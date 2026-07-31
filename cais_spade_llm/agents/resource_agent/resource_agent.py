"""Resource agent that receives work orders, selects a tool via LLM, and executes it."""

# agents/resource_agent/resource_agent.py
from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from spade.behaviour import CyclicBehaviour  # Behaviour base used for our inbox loop.
from spade.message import Message  # SPADE message objects (XMPP stanzas under the hood).
from spade.template import Template  # Filters incoming messages by metadata.

from cais_spade_llm.agents.intelligent_product.replanner.failure_context import (
    build_failure_event,
)
from cais_spade_llm.agents.shared_information.llm_agent import LlmAgent
from cais_spade_llm.agents.shared_information.local_dispatch import send_agent_message
from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    RECOVERY_OUTLINE_PHYSICAL_VALIDATE,
    RECOVERY_OUTLINE_PHYSICAL_VALIDATED,
    recovery_validation_fingerprint,
)


class ResourceAgent(LlmAgent):
    """
    SPADE ResourceAgent.

    - Receives tasks from ProductAgents (metadata[type] == "task").
    - Calls the LLM with tools enabled to select a function.
    - Dispatches to a registered executable (async function).
    - Sends ACK messages back to the ProductAgent with status.
    """

    agent_role = "resource"  # Used by LlmAgent to pick prompts and instructions for this class.

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str,
        function_names: Iterable[str] | None = None,
        static_capabilities: dict[str, Any] | None = None,
        allowed_senders: Iterable[str] | None = None,
        llm_timeout_s: int = 30,
        tool_timeout_s: int = 300,
        cca_jid: str | None = None,  # <-- NEW
        **kw: Any,
    ) -> None:
        """
        :param function_names: Names of methods on this class to expose as tools.
        :param static_capabilities: Optional metadata for discovery/logging.
        :param allowed_senders: Optional allowlist of JIDs allowed to send tasks.
        :param llm_timeout_s: Timeout for LLM tool selection.
        :param tool_timeout_s: Timeout for tool execution.
        """
        super().__init__(
            jid,
            password,
            name=name,
            agent_role="resource",
            function_names=list(function_names or []),
            **kw,
        )

        self.cca_jid = cca_jid

        # Optional metadata (payload limits, tool list, etc.) exposed to other agents or dashboards.
        self.static_capabilities: dict[str, Any] = static_capabilities or {}
        # Optional sender allow-list: if populated, only those JIDs can submit work.
        self.allowed_senders = set(allowed_senders or [])
        # Separate timeouts keep LLM latency (planning) independent from tool runtime (execution).
        self.llm_timeout_s = int(llm_timeout_s)
        self.tool_timeout_s = int(tool_timeout_s)

        self._safety_decisions: dict[str, str] = {}
        self._recovery_execution_primitive_catalog_cache: list[dict[str, Any]] | None = None
        self._recovery_synthesis_primitive_catalog_cache: list[dict[str, Any]] | None = None

        # Register recovery executor for all resource types.
        # RobotAgent overrides the method but no longer needs to re-register.
        self.executables["execute_recovery_macro"] = self.execute_recovery_macro

    # ------------------------------------------------------------------ #
    # SPADE lifecycle
    # ------------------------------------------------------------------ #
    async def setup(self) -> None:
        await super().setup()

        # Tasks from ProductAgent
        t_task = Template()
        t_task.set_metadata("type", "task")
        self.add_behaviour(self._TaskInbox(), t_task)

        # Safety decisions from CCA
        t_safety = Template()
        t_safety.set_metadata("type", "safety_decision")
        self.add_behaviour(self._SafetyDecisionInbox(), t_safety)

        t_recovery_physical_validate = Template()
        t_recovery_physical_validate.set_metadata(
            "type", RECOVERY_OUTLINE_PHYSICAL_VALIDATE
        )
        self.add_behaviour(
            self._RecoveryOutlinePhysicalValidationInbox(),
            t_recovery_physical_validate,
        )

    def _snapshot_state(self) -> dict[str, Any]:
        """
        Best-effort snapshot of resource state for failure context.
        Subclasses can override to provide richer state.
        """
        return {}

    def recovery_resource_type(self) -> str:
        snapshot = self._snapshot_state() if hasattr(self, "_snapshot_state") else {}
        if isinstance(snapshot, dict):
            token = str(snapshot.get("resource_type", "") or "").strip().lower()
            if token:
                return token
        token = str(self.static_capabilities.get("resource_type", "") or "").strip().lower()
        if token:
            return token
        return "resource"

    def get_recovery_snapshot(self) -> dict[str, Any]:
        """Return the current descriptor-driven recovery snapshot for this resource."""
        from cais_spade_llm.resources.resource_primitives import (
            get_resource_recovery_snapshot,
        )

        return get_resource_recovery_snapshot(self)

    def recovery_execution_primitive_catalog(self) -> list[dict[str, Any]]:
        """Return the resource-owned execution primitive catalog."""
        if self._recovery_execution_primitive_catalog_cache is None:
            from cais_spade_llm.resources.resource_primitives import (
                build_execution_primitive_catalog,
            )

            self._recovery_execution_primitive_catalog_cache = build_execution_primitive_catalog(self)
        return deepcopy(self._recovery_execution_primitive_catalog_cache)

    def recovery_synthesis_primitive_catalog(self) -> list[dict[str, Any]]:
        """Return the resource-owned LLM-facing primitive catalog."""
        if self._recovery_synthesis_primitive_catalog_cache is None:
            from cais_spade_llm.resources.resource_primitives import (
                build_synthesis_primitive_catalog,
            )

            self._recovery_synthesis_primitive_catalog_cache = build_synthesis_primitive_catalog(
                primitive_catalog=self.recovery_execution_primitive_catalog()
            )
        return deepcopy(self._recovery_synthesis_primitive_catalog_cache)

    def recovery_des_model(
        self,
        *,
        snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return this RA's local recovery extended finite automaton."""
        from cais_spade_llm.resources.resource_primitives import (
            build_recovery_des_model,
        )

        return build_recovery_des_model(
            self,
            snapshot=snapshot,
        )

    def invalidate_recovery_primitive_catalog(self) -> None:
        """Clear cached primitive catalogs after resource primitive changes."""
        self._recovery_execution_primitive_catalog_cache = None
        self._recovery_synthesis_primitive_catalog_cache = None

    def recovery_validation_resource_matches(self, resource_jid: str) -> bool:
        """Return whether a physical-validation request targets this exact RA."""
        receiving_jid = str(getattr(self, "jid", "") or "").strip().split("/", 1)[0]
        requested_jid = str(resource_jid or "").strip().split("/", 1)[0]
        return bool(requested_jid and requested_jid == receiving_jid)

    @staticmethod
    def _recovery_transition_state_value(
        *,
        field_name: str,
        scope: str,
        recovery_snapshot: dict[str, Any],
        part_context: dict[str, Any],
    ) -> tuple[bool, Any]:
        """Return one exact current value from Resource-Agent-owned evidence."""
        source = part_context if scope == "part" else recovery_snapshot
        aliases = {
            ("resource", "resource_state"): (
                "resource_state",
                "current_state",
                "state",
            ),
            ("resource", "resource_location"): (
                "resource_location",
                "current_location",
                "location",
            ),
            ("resource", "held_part"): ("held_part",),
            ("part", "part_state"): ("part_state", "current_state", "state"),
            ("part", "part_location"): (
                "part_location",
                "current_location",
                "location",
                "current_pose_ref",
            ),
        }
        for key in aliases.get((scope, field_name), (field_name,)):
            if key in source:
                return True, deepcopy(source.get(key))
        if scope == "resource" and field_name == "resource_location":
            occupancy = recovery_snapshot.get("occupancy")
            if isinstance(occupancy, dict) and "location" in occupancy:
                return True, deepcopy(occupancy.get("location"))
        if (
            scope == "part"
            and field_name == "part_location"
            and isinstance(part_context.get("observed_pose"), dict)
        ):
            return True, "observed_pose"
        return False, None

    @staticmethod
    def _recovery_string_tokens(value: Any) -> set[str]:
        """Return exact string tokens from one capability declaration value."""
        if isinstance(value, dict):
            return {
                str(token).strip()
                for token in value
                if str(token).strip()
            }
        if not isinstance(value, (list, tuple, set)):
            return set()
        return {
            str(token).strip()
            for token in value
            if isinstance(token, str) and str(token).strip()
        }

    def _recovery_known_location_tokens(
        self,
        *,
        recovery_des_model: dict[str, Any],
        recovery_snapshot: dict[str, Any],
        part_context: dict[str, Any],
    ) -> set[str]:
        """Return exact locations dynamically exposed by this Resource Agent."""
        tokens: set[str] = set()
        state_variables = dict(recovery_des_model.get("state_variables") or {})
        for field_name in ("resource_location", "part_location"):
            declaration = dict(state_variables.get(field_name) or {})
            tokens.update(
                str(value).strip()
                for value in (declaration.get("domain") or [])
                if isinstance(value, str) and str(value).strip()
            )
        for source in (
            recovery_snapshot,
        ):
            for field_name in (
                "named_poses",
                "available_named_poses",
                "reachability",
                "reachable_locations",
                "known_locations",
            ):
                tokens.update(
                    ResourceAgent._recovery_string_tokens(source.get(field_name))
                )
            tokens.update(
                str(token).strip()
                for token in dict(source.get("staging_areas") or {})
                if str(token).strip()
            )
            for field_name in (
                "resource_location",
                "current_location",
                "location",
            ):
                value = source.get(field_name)
                if isinstance(value, str) and value.strip():
                    tokens.add(value.strip())
        for field_name in (
            "part_location",
            "current_location",
            "location",
            "current_pose_ref",
        ):
            value = part_context.get(field_name)
            if isinstance(value, str) and value.strip():
                tokens.add(value.strip())
        if isinstance(part_context.get("observed_pose"), dict):
            tokens.add("observed_pose")
        observed_store_as = str(part_context.get("observed_store_as") or "").strip()
        if observed_store_as:
            tokens.add(observed_store_as)
        tokens.update(
            str(alias).strip()
            for alias in (part_context.get("observed_aliases") or [])
            if str(alias or "").strip()
        )
        return tokens

    def check_recovery_transition_feasibility(  # noqa: C901
        self,
        *,
        task: dict[str, Any],
        recovery_snapshot: dict[str, Any],
        part_context: dict[str, Any],
        recovery_des_model: dict[str, Any],
    ) -> dict[str, Any]:
        """Check candidate consistency using freshly retrieved RA capabilities.

        Generated event and state names remain opaque exact symbols. This check
        does not select configured events or search for a capability sequence.
        """
        state_variables = dict(recovery_des_model.get("state_variables") or {})
        if not state_variables:
            return {
                "allowed": False,
                "constraint_code": "resource_validation_unavailable",
                "reason": "ResourceAgent recovery capabilities are unavailable",
            }

        start_state = task.get("expected_start_state")
        end_state = task.get("expected_end_state")
        if not isinstance(start_state, dict) or not isinstance(end_state, dict):
            return {
                "allowed": False,
                "constraint_code": "candidate_schema_violation",
                "reason": "candidate transition states must be objects",
            }
        part_name = str(task.get("part_name") or "").strip()

        unknown_fields = sorted(
            {
                str(field_name)
                for state in (start_state, end_state)
                for field_name in state
                if str(field_name) not in state_variables
            }
        )
        if unknown_fields:
            return {
                "allowed": False,
                "constraint_code": "disallowed_outline_state_field",
                "reason": (
                    "candidate state field is not declared by the responsible "
                    f"ResourceAgent: {', '.join(unknown_fields)}"
                ),
                "evidence": {"state_fields": unknown_fields},
            }

        part_scoped_without_part = sorted(
            {
                str(field_name)
                for state in (start_state, end_state)
                for field_name in state
                if str(
                    dict(state_variables.get(field_name) or {}).get("scope")
                    or "resource"
                )
                == "part"
                and not part_name
            }
        )
        if part_scoped_without_part:
            return {
                "allowed": False,
                "constraint_code": "disallowed_outline_state_field",
                "reason": (
                    "part-scoped state fields require part_name: "
                    + ", ".join(part_scoped_without_part)
                ),
                "evidence": {"state_fields": part_scoped_without_part},
            }

        mismatches: list[dict[str, Any]] = []
        for field_name, expected in sorted(start_state.items()):
            scope = str(
                dict(state_variables.get(field_name) or {}).get("scope")
                or "resource"
            )
            available, actual = ResourceAgent._recovery_transition_state_value(
                field_name=field_name,
                scope=scope,
                recovery_snapshot=recovery_snapshot,
                part_context=part_context,
            )
            if not available or actual != expected:
                mismatches.append(
                    {
                        "field": field_name,
                        "expected": deepcopy(expected),
                        "actual": deepcopy(actual) if available else None,
                        "available": available,
                    }
                )
        if mismatches:
            return {
                "allowed": False,
                "constraint_family": "transition_staleness",
                "constraint_code": "validation_state_stale",
                "reason": (
                    "ResourceAgent state changed after the candidate start state "
                    "was projected"
                ),
                "evidence": {"mismatches": mismatches},
                "retriable": True,
            }

        known_locations = ResourceAgent._recovery_known_location_tokens(
            self,
            recovery_des_model=recovery_des_model,
            recovery_snapshot=recovery_snapshot,
            part_context=part_context,
        )
        for field_name in ("resource_location", "part_location"):
            value = end_state.get(field_name)
            if value in (None, ""):
                continue
            if not isinstance(value, str) or value not in known_locations:
                return {
                    "allowed": False,
                    "constraint_code": "unknown_location_token",
                    "reason": (
                        f"expected_end_state.{field_name} is not exposed by the "
                        "responsible ResourceAgent"
                    ),
                    "evidence": {
                        "field": f"expected_end_state.{field_name}",
                        "location_token": deepcopy(value),
                    },
                }

        named_pose_tokens = set()
        for source in (recovery_snapshot,):
            named_pose_tokens.update(
                ResourceAgent._recovery_string_tokens(source.get("named_poses"))
            )
            named_pose_tokens.update(
                ResourceAgent._recovery_string_tokens(
                    source.get("available_named_poses")
                )
            )
        end_resource_state = str(end_state.get("resource_state") or "").strip()
        if (
            end_resource_state in named_pose_tokens
            and end_state.get("resource_location") != end_resource_state
        ):
            return {
                "allowed": False,
                "constraint_code": "part_traceability_violation",
                "reason": (
                    "named-pose resource state and resource location do not match"
                ),
                "evidence": {
                    "resource_state": end_resource_state,
                    "resource_location": deepcopy(
                        end_state.get("resource_location")
                    ),
                },
            }

        end_held_part = end_state.get("held_part")
        if part_name and end_held_part not in (None, "", part_name):
            return {
                "allowed": False,
                "constraint_code": "part_traceability_violation",
                "reason": (
                    "expected_end_state.held_part contradicts the candidate part_name"
                ),
                "evidence": {
                    "field": "expected_end_state.held_part",
                    "held_part": deepcopy(end_held_part),
                    "part_name": part_name,
                },
                "invariant_id": "part_traceability",
            }
        if part_name and end_held_part == part_name:
            current_holder_resource_jid = str(
                part_context.get("current_holder_resource_jid") or ""
            ).strip()
            validator_jid = str(getattr(self, "jid", "") or "").strip()
            if (
                current_holder_resource_jid
                and current_holder_resource_jid != validator_jid
            ):
                return {
                    "allowed": False,
                    "constraint_code": "part_traceability_violation",
                    "reason": (
                        "candidate custody conflicts with the ResourceAgent's "
                        "current part-holder evidence"
                    ),
                    "evidence": {
                        "current_holder_resource_jid": current_holder_resource_jid,
                        "proposed_holder_resource_jid": validator_jid,
                    },
                    "invariant_id": "part_traceability",
                }
            from cais_spade_llm.resources.resource_profile import (
                get_resource_profile_for_agent,
                resource_snapshot_carried_entity_location,
            )

            carried_part_location = resource_snapshot_carried_entity_location(
                resource_jid=str(getattr(self, "jid", "") or "").strip(),
                snapshot=recovery_snapshot,
                profile=get_resource_profile_for_agent(self),
            )
            part_location_declaration = dict(
                state_variables.get("part_location") or {}
            )
            part_location_domain = part_location_declaration.get("domain")
            if (
                str(part_location_declaration.get("scope") or "") != "part"
                or not isinstance(part_location_domain, list)
                or carried_part_location not in part_location_domain
            ):
                carried_part_location = ""
            proposed_part_location = end_state.get("part_location")
            if not carried_part_location:
                return {
                    "allowed": False,
                    "constraint_code": "part_traceability_violation",
                    "reason": (
                        "ResourceAgent capabilities do not declare a carried-part "
                        "location"
                    ),
                    "evidence": {
                        "field": "expected_end_state.part_location",
                        "proposed_part_location": deepcopy(proposed_part_location),
                        "expected_carried_part_location": None,
                    },
                    "invariant_id": "part_traceability",
                }
            if proposed_part_location != carried_part_location:
                return {
                    "allowed": False,
                    "constraint_code": "held_part_location_mismatch",
                    "reason": (
                        "held_part and part_location do not describe the same "
                        "ResourceAgent transition"
                    ),
                    "evidence": {
                        "field": "expected_end_state.part_location",
                        "proposed_part_location": deepcopy(proposed_part_location),
                        "expected_carried_part_location": carried_part_location,
                    },
                    "invariant_id": "part_traceability",
                }

        return {
            "allowed": True,
            "reason": "ResourceAgent transition consistency is satisfied",
        }

    @staticmethod
    def recovery_physical_validation_snapshot(
        *,
        live_snapshot: dict[str, Any],
        physical_input: dict[str, Any],
        recovery_des_model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Overlay PA-projected dynamic facts on fresh RA capability evidence."""
        validation_snapshot = deepcopy(live_snapshot or {})
        if physical_input.get("use_projected_recovery_snapshot") is not True:
            return validation_snapshot
        projected_snapshot = physical_input.get("projected_recovery_snapshot")
        if not isinstance(projected_snapshot, dict):
            return validation_snapshot

        declared_fields = set(
            dict((recovery_des_model or {}).get("state_variables") or {})
        )
        declared_fields.update(
            {
                "resource_state",
                "current_state",
                "resource_location",
                "current_location",
                "held_part",
                "occupancy",
            }
        )
        for field_name in sorted(declared_fields):
            if field_name in projected_snapshot:
                validation_snapshot[field_name] = deepcopy(
                    projected_snapshot.get(field_name)
                )
        return validation_snapshot

    def check_recovery_physical_feasibility(
        self,
        *,
        part_context: dict[str, Any] | None = None,
        recovery_snapshot: dict[str, Any] | None = None,
        grounded_action: dict[str, Any] | None = None,
        **_compat_kwargs: Any,
    ) -> dict[str, Any]:
        """Fail closed when a resource has no physical recovery validator.

        Subclasses (RobotAgent, PrintingAgent) can override with
        resource-specific checks.
        """
        del part_context, recovery_snapshot, grounded_action
        return {
            "allowed": False,
            "constraint_code": "resource_validation_unavailable",
            "reason": "resource_validation_unavailable",
        }

    def validate_recovery_outline_physical_candidates(  # noqa: C901
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate one same-resource candidate batch against one fresh snapshot."""
        validator_jid = str(self.jid)
        snapshot: dict[str, Any] = {}
        recovery_des_model: dict[str, Any] = {}
        capability_error = ""
        try:
            raw_snapshot = self.get_recovery_snapshot()
            if not isinstance(raw_snapshot, dict):
                raise TypeError("ResourceAgent returned a malformed recovery snapshot")
            snapshot = deepcopy(raw_snapshot)
            recovery_des_model_method = getattr(self, "recovery_des_model", None)
            if not callable(recovery_des_model_method):
                raise RuntimeError("ResourceAgent recovery capabilities are unavailable")
            raw_recovery_des_model = recovery_des_model_method(snapshot=snapshot)
            if not isinstance(raw_recovery_des_model, dict) or not raw_recovery_des_model:
                raise RuntimeError("ResourceAgent recovery capabilities are unavailable")
            recovery_des_model = deepcopy(raw_recovery_des_model)
        except Exception as exc:  # noqa: BLE001 - capability boundary must fail closed
            capability_error = str(exc).strip() or type(exc).__name__
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            candidates = []
        results: list[dict[str, Any]] = []
        for row in candidates:
            candidate = row if isinstance(row, dict) else {}
            task = candidate.get("task")
            task = task if isinstance(task, dict) else {}
            candidate_index = int(candidate.get("candidate_index") or 0)
            resource_jid = str(task.get("resource_jid") or "").strip()
            physical_input = candidate.get("physical_input")
            physical_input = (
                physical_input if isinstance(physical_input, dict) else {}
            )
            snapshot_projector = getattr(
                self,
                "recovery_physical_validation_snapshot",
                ResourceAgent.recovery_physical_validation_snapshot,
            )
            validation_snapshot = snapshot_projector(
                live_snapshot=snapshot,
                physical_input=physical_input,
                recovery_des_model=recovery_des_model,
            )
            if not self.recovery_validation_resource_matches(resource_jid):
                transition_result = {
                    "allowed": False,
                    "constraint_code": "wrong_resource_validator",
                    "reason": (
                        f"candidate resource_jid '{resource_jid}' does not match "
                        f"receiving ResourceAgent '{validator_jid}'"
                    ),
                }
                physical_result = {
                    "allowed": False,
                    "skipped": True,
                    "reason": "transition_feasibility rejected",
                }
            elif capability_error:
                transition_result = {
                    "allowed": False,
                    "constraint_code": "resource_validation_unavailable",
                    "reason": capability_error,
                }
                physical_result = {
                    "allowed": False,
                    "skipped": True,
                    "reason": "transition_feasibility rejected",
                }
            else:
                transition_result = ResourceAgent.check_recovery_transition_feasibility(
                    self,
                    task=deepcopy(task),
                    recovery_snapshot=validation_snapshot,
                    part_context=deepcopy(physical_input.get("part_context") or {}),
                    recovery_des_model=deepcopy(recovery_des_model),
                )
                if not isinstance(transition_result, dict):
                    transition_result = {
                        "allowed": False,
                        "constraint_code": "resource_validation_unavailable",
                        "reason": (
                            "ResourceAgent returned a malformed transition result"
                        ),
                    }
                if bool(transition_result.get("allowed") is True):
                    physical_result = self.check_recovery_physical_feasibility(
                        part_context=deepcopy(
                            physical_input.get("part_context") or {}
                        ),
                        recovery_snapshot=validation_snapshot,
                        grounded_action=deepcopy(
                            physical_input.get("grounded_action") or {}
                        ),
                        operation_kind=str(
                            physical_input.get("operation_kind") or ""
                        ),
                        part_name=(
                            str(physical_input.get("part_name") or "").strip()
                            or None
                        ),
                    )
                    if not isinstance(physical_result, dict):
                        physical_result = {
                            "allowed": False,
                            "constraint_code": "resource_validation_unavailable",
                            "reason": (
                                "ResourceAgent returned a malformed physical result"
                            ),
                        }
                else:
                    physical_result = {
                        "allowed": False,
                        "skipped": True,
                        "reason": "transition_feasibility rejected",
                    }

            transition_findings: list[dict[str, Any]] = []
            if not bool(transition_result.get("allowed") is True):
                transition_finding = {
                    "validation_category": "transition_feasibility",
                    "constraint_owner": "resource",
                    "constraint_family": str(
                        transition_result.get("constraint_family")
                        or "transition_consistency"
                    ),
                    "constraint_code": str(
                        transition_result.get("constraint_code")
                        or "transition_feasibility_rejected"
                    ),
                    "reason": str(
                        transition_result.get("reason")
                        or "ResourceAgent rejected transition feasibility"
                    ),
                    "resource_jid": resource_jid or validator_jid,
                    "part_name": task.get("part_name"),
                    "evidence": deepcopy(
                        transition_result.get("evidence") or {}
                    ),
                }
                invariant_id = str(
                    transition_result.get("invariant_id") or ""
                ).strip()
                if invariant_id:
                    transition_finding["invariant_id"] = invariant_id
                if transition_result.get("retriable") is True:
                    transition_finding["retriable"] = True
                transition_findings.append(transition_finding)

            physical_findings: list[dict[str, Any]] = []
            if (
                not bool(physical_result.get("skipped"))
                and not bool(physical_result.get("allowed") is True)
            ):
                physical_findings.append(
                    {
                        "validation_category": "physical_feasibility",
                        "constraint_owner": "resource",
                        "constraint_family": "resource_feasibility",
                        "constraint_code": str(
                            physical_result.get("constraint_code")
                            or "resource_feasibility_rejected"
                        ),
                        "reason": str(
                            physical_result.get("reason")
                            or "ResourceAgent rejected physical feasibility"
                        ),
                        "resource_jid": resource_jid or validator_jid,
                        "part_name": task.get("part_name"),
                        "evidence": deepcopy(
                            physical_result.get("evidence") or {}
                        ),
                    }
                )
            transition_result = {
                **deepcopy(transition_result),
                "findings": deepcopy(transition_findings),
            }
            physical_result = {
                **deepcopy(physical_result),
                "findings": deepcopy(physical_findings),
            }
            allowed = bool(
                transition_result.get("allowed") is True
                and physical_result.get("allowed") is True
            )
            findings = [*transition_findings, *physical_findings]
            results.append(
                {
                    "candidate_index": candidate_index,
                    "event_id": str(candidate.get("event_id") or "").strip(),
                    "allowed": allowed,
                    "findings": findings,
                    "transition_feasibility": transition_result,
                    "physical_feasibility": physical_result,
                    "resource_result": deepcopy(
                        physical_result
                        if transition_result.get("allowed") is True
                        else transition_result
                    ),
                }
            )
        return {
            "validator_jid": validator_jid,
            "snapshot": deepcopy(snapshot),
            "snapshot_fingerprint": recovery_validation_fingerprint(snapshot),
            "recovery_des_model": deepcopy(recovery_des_model),
            "recovery_des_model_fingerprint": str(
                recovery_des_model.get("descriptor_fingerprint") or ""
            ),
            "results": results,
            "enabled_event_ids": sorted(
                {
                    str(row.get("event_id") or "").strip()
                    for row in results
                    if bool(row.get("allowed"))
                    and str(row.get("event_id") or "").strip()
                }
            ),
        }

    async def generate_recovery_primitives_batch(
        self,
        *,
        recovery_session_id: str = "",
        resource_jid: str = "",
        assigned_outline_events: list[dict[str, Any]] | None = None,
        prepared_recovery_request: dict[str, Any] | None = None,
        carried_session_state: dict[str, Any] | None = None,
        max_turns: int = 24,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
            generate_primitive_batch_with_llm_agent,
        )

        normalized_resource_jid = str(resource_jid or getattr(self, "jid", "") or "").strip()
        if (
            normalized_resource_jid
            and normalized_resource_jid != str(getattr(self, "jid", "") or "").strip()
        ):
            raise ValueError(
                f"resource-owned primitive batch was assigned to {normalized_resource_jid!r} "
                f"but invoked on {str(getattr(self, 'jid', '') or '').strip()!r}"
            )
        return await generate_primitive_batch_with_llm_agent(
            llm_agent=self,
            prepared_recovery_request=dict(prepared_recovery_request or {}),
            assigned_outline_events=[
                dict(row) for row in (assigned_outline_events or []) if isinstance(row, dict)
            ],
            recovery_session_id=str(recovery_session_id or "").strip(),
            carried_session_state=dict(carried_session_state or {}),
            max_turns=max_turns,
        )

    async def execute_recovery_macro(
        self,
        *,
        macro_name: str = "",
        primitive_steps: list | None = None,
        expected_start_state: str = "",
        expected_snapshot: dict[str, Any] | None = None,
        product_jid: str | None = None,
        task_id: str | None = None,
        in_state: str | None = None,
        out_state: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generic recovery macro executor.

        Validates the starting snapshot, semantically validates the primitive
        sequence, executes each primitive on the resolved owner, and applies
        projected recovery state back onto the resource agent.
        """
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_primitives import (
            apply_effects_to_snapshot,
            event_fact_key_for_primitive,
            expand_composite_steps,
            extract_step_output,
            resolve_param_refs,
            snapshot_matches_expected,
            validate_and_project_steps,
        )
        from cais_spade_llm.resources.resource_primitives import (
            get_resource_recovery_snapshot,
            sync_agent_from_recovery_snapshot,
        )
        from cais_spade_llm.resources.resource_profile import (
            get_resource_profile_for_agent,
            resource_snapshot_set_field,
        )

        steps = list(primitive_steps or [])
        runtime_snapshot = get_resource_recovery_snapshot(self)
        actual_state = str(
            runtime_snapshot.get("current_state") or getattr(self, "_current_state", "") or ""
        ).strip()

        if expected_start_state and actual_state != expected_start_state:
            msg = (
                f"Recovery macro '{macro_name}' expected start state "
                f"'{expected_start_state}' but resource is in '{actual_state}'"
            )
            return {
                "status": "failed",
                "content": msg,
                "observations": {
                    "macro_name": macro_name,
                    "expected_start_state": expected_start_state,
                    "actual_state": actual_state,
                    "step_index": -1,
                },
            }

        if expected_snapshot:
            matches, mismatch_message = snapshot_matches_expected(
                runtime_snapshot, expected_snapshot
            )
            if not matches:
                msg = (
                    f"Recovery macro '{macro_name}' expected snapshot mismatch: {mismatch_message}"
                )
                return {
                    "status": "failed",
                    "content": msg,
                    "observations": {
                        "macro_name": macro_name,
                        "expected_snapshot": expected_snapshot,
                        "actual_snapshot": runtime_snapshot,
                        "step_index": -1,
                    },
                }

        if not steps:
            return {
                "status": "failed",
                "content": f"Recovery macro '{macro_name}' has no primitive steps",
            }

        primitive_catalog = self.recovery_execution_primitive_catalog()
        try:
            steps = expand_composite_steps(steps, primitive_catalog)
        except Exception as exc:
            return {
                "status": "failed",
                "content": (
                    f"Recovery macro '{macro_name}' could not expand composite primitives: {exc}"
                ),
                "observations": {
                    "macro_name": macro_name,
                    "step_index": -1,
                },
            }
        primitive_meta_by_name = {
            str(entry.get("name", "")).strip(): entry
            for entry in primitive_catalog
            if isinstance(entry, dict) and str(entry.get("name", "")).strip()
        }
        grounding_context = deepcopy(dict(kwargs or {}))
        semantic_ok, _projected_runtime_snapshot, semantic_error = validate_and_project_steps(
            steps,
            primitive_catalog,
            runtime_snapshot,
            grounding_context=grounding_context,
        )
        if not semantic_ok:
            msg = (
                f"Recovery macro '{macro_name}' failed runtime semantic validation: "
                f"{semantic_error}"
            )
            return {
                "status": "failed",
                "content": msg,
                "observations": {
                    "macro_name": macro_name,
                    "expected_snapshot": expected_snapshot,
                    "actual_snapshot": runtime_snapshot,
                    "semantic_error": semantic_error,
                    "step_index": -1,
                },
            }

        profile = get_resource_profile_for_agent(self)
        owner = self
        if profile.primitive_owner_resolver is not None:
            owner = profile.primitive_owner_resolver(self) or self

        results: list[dict[str, Any]] = []
        event_facts: dict[str, Any] = {}
        resource_type = (
            str(
                dict(runtime_snapshot.get("resource_core") or {}).get("resource_type")
                or runtime_snapshot.get("resource_type")
                or "resource"
            )
            .strip()
            .lower()
            or "resource"
        )
        for step_idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            primitive = str(step.get("primitive", "")).strip()
            raw_params = dict(step.get("params") or {})

            fn = getattr(owner, primitive, None) or getattr(self, primitive, None)
            if not callable(fn):
                msg = f"Unknown primitive '{primitive}' at step {step_idx} in macro '{macro_name}'"
                return {
                    "status": "failed",
                    "content": msg,
                    "observations": {
                        "macro_name": macro_name,
                        "step_index": step_idx,
                        "primitive": primitive,
                        "completed_steps": len(results),
                    },
                }

            try:
                params = resolve_param_refs(
                    raw_params,
                    grounding_context,
                    event_facts=event_facts,
                )
            except Exception as exc:
                msg = (
                    f"Macro '{macro_name}' could not resolve params at step {step_idx} "
                    f"({primitive}): {exc}"
                )
                return {
                    "status": "failed",
                    "content": msg,
                    "observations": {
                        "macro_name": macro_name,
                        "step_index": step_idx,
                        "primitive": primitive,
                        "raw_params": raw_params,
                        "event_facts": deepcopy(event_facts),
                    },
                }

            try:
                maybe_result = fn(**params)
                step_result = (
                    await maybe_result if inspect.isawaitable(maybe_result) else maybe_result
                )
            except Exception as exc:
                return {
                    "status": "failed",
                    "content": (
                        f"Macro '{macro_name}' failed at step {step_idx} "
                        f"({primitive}): {type(exc).__name__}: {exc}"
                    ),
                    "observations": {
                        "macro_name": macro_name,
                        "step_index": step_idx,
                        "primitive": primitive,
                        "completed_steps": len(results),
                        "total_steps": len(steps),
                    },
                }

            if isinstance(step_result, dict):
                normalized_result = dict(step_result)
                normalized_result.setdefault("success", True)
            elif isinstance(step_result, bool):
                normalized_result = {
                    "success": step_result,
                    "message": f"{primitive} {'ok' if step_result else 'failed'}",
                }
            elif isinstance(step_result, list):
                normalized_result = {
                    "success": True,
                    "message": f"{primitive} returned {len(step_result)} items",
                    "data": step_result,
                }
            else:
                normalized_result = {
                    "success": True,
                    "message": f"{primitive} completed",
                    "data": step_result,
                }

            results.append({"primitive": primitive, "result": normalized_result})
            if not normalized_result.get("success", False):
                return {
                    "status": "failed",
                    "content": (
                        f"Macro '{macro_name}' failed at step {step_idx} "
                        f"({primitive}): {normalized_result.get('message') or 'unknown failure'}"
                    ),
                    "observations": {
                        "macro_name": macro_name,
                        "step_index": step_idx,
                        "primitive": primitive,
                        "primitive_result": normalized_result,
                        "completed_steps": len(results) - 1,
                        "total_steps": len(steps),
                    },
                }

            primitive_meta = primitive_meta_by_name.get(primitive)
            if primitive_meta is not None:
                runtime_snapshot = apply_effects_to_snapshot(
                    {**dict(step), "params": params},
                    primitive_meta,
                    runtime_snapshot,
                )
                sync_agent_from_recovery_snapshot(self, runtime_snapshot)

            event_fact_key, event_fact_error = event_fact_key_for_primitive(
                primitive=primitive,
                params=params,
                resource_type=resource_type,
            )
            if event_fact_error:
                return {
                    "status": "failed",
                    "content": (
                        f"Macro '{macro_name}' could not publish event facts at step {step_idx} "
                        f"({primitive}): {event_fact_error}"
                    ),
                    "observations": {
                        "macro_name": macro_name,
                        "step_index": step_idx,
                        "primitive": primitive,
                        "primitive_result": normalized_result,
                        "completed_steps": len(results),
                    },
                }
            if event_fact_key:
                step_output, output_error = extract_step_output(
                    primitive=primitive,
                    params=params,
                    step_result=normalized_result,
                    resource_type=resource_type,
                )
                if output_error:
                    return {
                        "status": "failed",
                        "content": (
                            f"Macro '{macro_name}' could not publish event facts at step {step_idx} "
                            f"({primitive}): {output_error}"
                        ),
                        "observations": {
                            "macro_name": macro_name,
                            "step_index": step_idx,
                            "primitive": primitive,
                            "primitive_result": normalized_result,
                            "event_fact_path": f"event_facts.{event_fact_key}",
                            "completed_steps": len(results),
                        },
                    }
                target = event_facts
                tokens = [token for token in str(event_fact_key).split(".") if token]
                for token in tokens[:-1]:
                    child = target.get(token)
                    if not isinstance(child, dict):
                        child = {}
                        target[token] = child
                    target = child
                if tokens:
                    target[tokens[-1]] = deepcopy(step_output)

        if out_state:
            runtime_snapshot = resource_snapshot_set_field(
                runtime_snapshot,
                "current_state",
                out_state,
                profile=profile,
            )
            sync_agent_from_recovery_snapshot(self, runtime_snapshot)

        return {
            "status": "completed",
            "content": f"Recovery macro '{macro_name}' completed successfully",
            "observations": {
                "macro_name": macro_name,
                "completed_steps": len(results),
                "total_steps": len(steps),
                "event_facts": deepcopy(event_facts),
            },
        }

    def _build_failure_context(
        self,
        *,
        fn_name: str,
        fn_args: dict[str, Any],
        result: dict[str, Any] | None,
        final_status: str,
        state_before: dict[str, Any],
        state_after: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a normalized failure context payload for any failure type.
        """
        is_failed = isinstance(final_status, str) and final_status.startswith("failed")
        if not is_failed:
            if isinstance(result, dict) and isinstance(result.get("failure_context"), dict):
                return deepcopy(result.get("failure_context") or {})
            return {}

        raw_failure_context = {}
        raw_observations = {}
        if isinstance(result, dict):
            if isinstance(result.get("failure_context"), dict):
                raw_failure_context = deepcopy(result.get("failure_context") or {})
            if isinstance(result.get("observations"), dict):
                raw_observations = deepcopy(result.get("observations") or {})

        failure_event = build_failure_event(
            failed_task_id=str(fn_args.get("task_id") or "").strip(),
            failed_resource_jid=str(getattr(self, "jid", "") or "").strip(),
            failed_function_name=str(fn_name or "").strip(),
            final_status=str(final_status or "").strip(),
            part_name=str(fn_args.get("part_name") or "").strip(),
            base_failure_context=raw_failure_context,
            observations=raw_observations,
            state_before=state_before,
            state_after=state_after,
        )
        return deepcopy(failure_event.get("failure_context") or {})

    async def _wait_for_safety_decision(self, task_id: str) -> str | None:
        """
        Block until a safety_decision is available for this task_id.
        No timeout: waits indefinitely until CCA replies.
        """
        while True:
            decision = self._safety_decisions.pop(task_id, None)
            if decision is not None:
                return decision
            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------ #
    # Behaviours
    # ------------------------------------------------------------------ ##
    class _TaskInbox(CyclicBehaviour):
        """Long-running behaviour that processes incoming tasks sequentially."""

        async def run(self) -> None:
            agent: ResourceAgent = self.agent  # type: ignore

            # Poll inbox frequently but yield control if nothing arrives to keep agent responsive.
            msg = await self.receive(timeout=0.05)
            if not msg:
                return

            # ----- trust boundary ----- #
            # Give operators a simple safety net: reject unexpected senders early.
            if agent.allowed_senders and str(msg.sender) not in agent.allowed_senders:
                agent.logger.warning(f"[Resource] Rejecting task from {msg.sender} (not allowed)")
                await self._ack(msg, task_id="?", status="rejected:unauthorized")
                return

            # ----- envelope ----- #
            # Protocol metadata lets us bump behaviours later (e.g., plan/exec distinctions).
            protocol = msg.metadata.get("protocol", "")

            # ----- parse body ----- #
            try:
                data = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Resource] Malformed task body (not JSON).")
                await self._ack(msg, task_id="?", status="failed:bad_json")
                return

            task_id = data.get("task_id")
            instruction = data.get("instruction", "")
            phase_id = data.get("phase_id")  # optional multi-phase flow identifier

            if not task_id:
                agent.logger.warning("[Resource] Task without task_id.")
                await self._ack(msg, task_id="?", status="failed:missing_task_id")
                return

            agent.logger.info(f"[Resource] ← Task ({task_id}) from={msg.sender} proto={protocol}")

            # ----- Tool selection ----- #
            # If the instruction already specifies a tool, honor it and skip the LLM.
            fn_name = None
            fn_args: dict[str, Any] = {}
            if isinstance(instruction, dict):
                fn_name = instruction.get("function_name") or instruction.get("function")
                if isinstance(instruction.get("params"), dict):
                    fn_args = dict(instruction.get("params") or {})

            if not fn_name:
                # ----- EARLY ACK (non-recovery only) ----- #
                await self._ack(msg, task_id=task_id, status="accepted")
                try:
                    # Force the LLM to pick an explicit tool so we never free-text a task.
                    llm_resp = await asyncio.wait_for(
                        agent.ask_llm(
                            instruction,
                            with_functions=True,
                            force_tool=True,
                        ),
                        timeout=agent.llm_timeout_s,
                    )
                except asyncio.TimeoutError:
                    await self._ack(msg, task_id=task_id, status="llm_timeout")
                    return
                except Exception as e:
                    agent.logger.exception("[Resource] LLM failure")
                    await self._ack(
                        msg,
                        task_id=task_id,
                        status=f"failed:llm:{type(e).__name__}",
                    )
                    return

                fn_name, fn_args = _parse_function_call(llm_resp)
            if not fn_name:
                agent.logger.info(f"[Resource] ({task_id}) no_tool_match; responding.")
                await self._ack(msg, task_id=task_id, status="no_tool_match")
                return

            start_safety_mode = str(fn_args.get("start_safety_mode") or "").strip().lower()
            # Recovery macros keep the legacy default fast path unless
            # they explicitly request cca_check. Any task can now opt into the
            # same bypass with start_safety_mode=fast_path.
            is_recovery_macro = fn_name == "execute_recovery_macro"
            use_fast_path = bool(
                start_safety_mode == "fast_path"
                or (is_recovery_macro and start_safety_mode != "cca_check")
            )

            # ----- plumb routing/context ----- #
            # Pass routing info into the tool implementation for downstream logging/rpc calls.
            fn_args.setdefault("product_jid", str(msg.sender))
            fn_args.setdefault("task_id", task_id)
            if phase_id and "phase_id" not in fn_args:
                fn_args["phase_id"] = phase_id

            # ----- dispatch ----- #
            func = agent.executables.get(fn_name)
            if not func:
                agent.logger.warning(f"[Resource] Unknown tool '{fn_name}'")
                await self._ack(
                    msg,
                    task_id=task_id,
                    status=f"failed:unknown_tool:{fn_name}",
                )
                return

            if use_fast_path:
                # Fast path: self-allow, log, and proceed directly to execution.
                # Skip the safety_check round-trip but still notify CCA of
                # the running state so the plan FSA can track .start events.
                agent.logger.info(
                    "[Resource] Fast-path task %s fn=%s (skipping safety round-trip)",
                    task_id,
                    fn_name,
                )
                agent._safety_decisions[task_id] = "allow"
                await self._ack(msg, task_id=task_id, status="running")
                try:
                    running_msg = Message(to=agent.cca_jid)
                    running_msg.set_metadata("type", "resource_event")
                    running_msg.body = json.dumps(
                        {
                            "task_id": task_id,
                            "resource_jid": str(agent.jid),
                            "function_name": fn_name,
                            "params": fn_args,
                            "status": "running",
                        }
                    )
                    await send_agent_message(
                        self,
                        running_msg,
                        transport_label="resource_running",
                    )
                except Exception:
                    agent.logger.exception(
                        "[Resource] Failed to send running event for fast-path task %s (ignored).",
                        task_id,
                    )
            else:
                # ----- EARLY ACK (normal tasks) ----- #
                await self._ack(msg, task_id=task_id, status="accepted")

                # ----- RESOURCE EVENT (FOR SAFETY) NOTIFICATION TO CCA ----- #
                try:
                    # 1) Send request permission, not running
                    resource_msg = Message(to=agent.cca_jid)
                    resource_msg.set_metadata("type", "resource_event")
                    resource_msg.body = json.dumps(
                        {
                            "task_id": task_id,
                            "resource_jid": str(agent.jid),
                            "function_name": fn_name,
                            "params": fn_args,
                            "status": "safety_check",  # <-- REQUEST permission
                        }
                    )
                    await send_agent_message(
                        self,
                        resource_msg,
                        transport_label="resource_safety",
                    )
                except Exception:
                    agent.logger.exception(
                        "[Resource] Failed to send resource_event to CCA (ignored)."
                    )

            # 2) Wait for CCA decision (instant for recovery macros, blocks for normal tasks)
            decision = await agent._wait_for_safety_decision(task_id)

            if decision == "block":
                await self._ack(msg, task_id=task_id, status="blocked")
                return

            if not use_fast_path:
                # ---------------------------
                #  SAFETY PASSED → RUNNING
                # ---------------------------
                await self._ack(msg, task_id=task_id, status="running")

                running_msg = Message(to=agent.cca_jid)
                running_msg.set_metadata("type", "resource_event")
                running_msg.body = json.dumps(
                    {
                        "task_id": task_id,
                        "resource_jid": str(agent.jid),
                        "function_name": fn_name,
                        "params": fn_args,
                        "status": "running",
                    }
                )
                await send_agent_message(
                    self,
                    running_msg,
                    transport_label="resource_running",
                )

            # ---------------------------
            #  EXECUTE THE TOOL
            # ---------------------------
            state_before = agent._snapshot_state()
            state_after = state_before
            result: dict[str, Any] | None = None
            try:
                # Filter fn_args to only params the function accepts.
                # Functions that declare **kwargs receive everything;
                # others get only the params in their signature.
                sig = inspect.signature(func)
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
                if accepts_var_kw:
                    filtered_args = fn_args
                else:
                    accepted = set(sig.parameters.keys())
                    filtered_args = {k: v for k, v in fn_args.items() if k in accepted}

                result = await asyncio.wait_for(
                    func(**filtered_args),
                    timeout=agent.tool_timeout_s,
                )
                state_after = agent._snapshot_state()
                final_status = (result or {}).get("status") or "completed"

                # ----- RESOURCE EVENT: TASK FINISHED (notify CCA) ----- #
                if not (isinstance(final_status, str) and final_status.startswith("failed")):
                    try:
                        failure_context = agent._build_failure_context(
                            fn_name=fn_name,
                            fn_args=fn_args,
                            result=result if isinstance(result, dict) else None,
                            final_status=final_status,
                            state_before=state_before,
                            state_after=state_after,
                        )

                        done_msg = Message(to=agent.cca_jid)
                        done_msg.set_metadata("type", "resource_event")
                        done_msg.body = json.dumps(
                            {
                                "task_id": task_id,
                                "resource_jid": str(agent.jid),
                                "function_name": fn_name,
                                "params": fn_args,
                                "status": final_status,  # e.g. "completed", "blocked", etc.
                                "failure_context": failure_context,
                                "current_state": state_after.get("current_state", "idle"),
                            }
                        )
                        # fire-and-forget so we don't block on CCA
                        asyncio.create_task(
                            send_agent_message(
                                self,
                                done_msg,
                                transport_label="resource_done",
                            )
                        )
                    except Exception:
                        agent.logger.exception(
                            "[Resource] Failed to send final resource_event to CCA (ignored)."
                        )

            except asyncio.TimeoutError:
                agent.logger.exception("[Resource] Tool execution timeout")
                final_status = "failed:tool_timeout"
            except Exception as e:
                agent.logger.exception("[Resource] Tool execution failed")
                final_status = f"failed:tool:{type(e).__name__}"

            # Notify CCA of failure so it can clean up running_aps / FSA state.
            if isinstance(final_status, str) and final_status.startswith("failed"):
                try:
                    state_after = agent._snapshot_state()
                    failure_context = agent._build_failure_context(
                        fn_name=fn_name,
                        fn_args=fn_args,
                        result=result if isinstance(result, dict) else None,
                        final_status=final_status,
                        state_before=state_before,
                        state_after=state_after,
                    )
                    fail_msg = Message(to=agent.cca_jid)
                    fail_msg.set_metadata("type", "resource_event")
                    fail_msg.body = json.dumps(
                        {
                            "task_id": task_id,
                            "resource_jid": str(agent.jid),
                            "function_name": fn_name,
                            "params": fn_args,
                            "status": final_status,
                            "failure_context": failure_context,
                            "current_state": state_after.get("current_state", "idle"),
                        }
                    )
                    await send_agent_message(
                        self,
                        fail_msg,
                        transport_label="resource_fail",
                    )
                except Exception:
                    agent.logger.exception("[Resource] Failed to send fail resource_event to CCA.")

            # ---------------------------
            #  SEND FINAL ACK TO PA
            # ---------------------------
            ack_content = ""
            ack_observations = None
            if isinstance(result, dict):
                ack_content = str(result.get("content") or "").strip()
                raw_observations = result.get("observations")
                if isinstance(raw_observations, dict):
                    ack_observations = raw_observations
            await self._ack(
                msg,
                task_id=task_id,
                status=final_status,
                content=ack_content,
                observations=ack_observations,
            )

        async def _ack(
            self,
            msg: Message,
            *,
            task_id: str | None,
            status: str,
            content: str = "",
            observations: dict[str, Any] | None = None,
        ) -> None:
            """Send an acknowledgement/status update back to the originating ProductAgent."""
            reply = Message(to=str(msg.sender))
            reply.set_metadata("type", "ack")
            payload = {"task_id": task_id, "status": status}
            if content:
                payload["content"] = str(content).strip()
            if isinstance(observations, dict) and observations:
                payload["observations"] = observations
            reply.body = json.dumps(payload)
            await send_agent_message(
                self,
                reply,
                transport_label="resource_ack",
            )

    class _RecoveryOutlinePhysicalValidationInbox(CyclicBehaviour):
        """Validate recovery candidates using this ResourceAgent's live state."""

        async def run(self) -> None:
            agent: ResourceAgent = self.agent  # type: ignore
            msg = await self.receive(timeout=0.05)
            if not msg:
                return
            started_at = time.perf_counter()
            try:
                payload = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning(
                    "[Resource] Malformed recovery_outline_physical_validate body"
                )
                return
            if not isinstance(payload, dict):
                agent.logger.warning(
                    "[Resource] Non-object recovery_outline_physical_validate body"
                )
                return

            validation = agent.validate_recovery_outline_physical_candidates(payload)
            validator_jid = str(validation.get("validator_jid") or agent.jid)
            snapshot = dict(validation.get("snapshot") or {})
            snapshot_fingerprint = str(
                validation.get("snapshot_fingerprint") or ""
            )
            recovery_des_model = dict(
                validation.get("recovery_des_model") or {}
            )
            recovery_des_model_fingerprint = str(
                validation.get("recovery_des_model_fingerprint") or ""
            )
            results = [
                deepcopy(row)
                for row in (validation.get("results") or [])
                if isinstance(row, dict)
            ]
            response = {
                "request_id": str(payload.get("request_id") or ""),
                "recovery_session_id": str(
                    payload.get("recovery_session_id") or ""
                ),
                "turn_index": int(payload.get("turn_index") or 0),
                "state_fingerprint": str(payload.get("state_fingerprint") or ""),
                "validator_jid": validator_jid,
                "snapshot": deepcopy(snapshot),
                "snapshot_fingerprint": snapshot_fingerprint,
                "recovery_des_model": deepcopy(recovery_des_model),
                "recovery_des_model_fingerprint": recovery_des_model_fingerprint,
                "results": results,
                "enabled_event_ids": deepcopy(
                    validation.get("enabled_event_ids") or []
                ),
                "latency_ms": (time.perf_counter() - started_at) * 1000.0,
                "mocked": False,
            }
            reply = Message(to=str(payload.get("product_jid") or msg.sender or ""))
            reply.set_metadata("type", RECOVERY_OUTLINE_PHYSICAL_VALIDATED)
            reply.body = json.dumps(response, default=str)
            await send_agent_message(
                self,
                reply,
                trace_category=RECOVERY_OUTLINE_PHYSICAL_VALIDATED,
                transport_label=RECOVERY_OUTLINE_PHYSICAL_VALIDATED,
            )

    class _SafetyDecisionInbox(CyclicBehaviour):
        """Receives safety_decision messages from CCA and stores them on the agent."""

        async def run(self) -> None:
            agent: ResourceAgent = self.agent  # type: ignore

            msg = await self.receive(timeout=0.05)
            if not msg:
                return

            try:
                data = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Resource] Malformed safety_decision body")
                return

            task_id = data.get("task_id")
            decision = data.get("decision")

            if not task_id or decision not in ("allow", "block"):
                agent.logger.warning("[Resource] Invalid safety_decision message: %s", data)
                return

            agent._safety_decisions[task_id] = decision
            agent.logger.info(
                "[Resource] Stored safety_decision=%s for task=%s",
                decision,
                task_id,
            )


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _parse_function_call(resp):
    """Normalize the OpenAI response into (tool_name, args_dict)."""
    fc = resp.get("function_call")
    if not isinstance(fc, dict):
        return None, {}

    name = fc.get("name")
    args_raw = fc.get("arguments", {})

    if isinstance(args_raw, str):
        # OpenAI may return arguments as a JSON string; parse defensively.
        try:
            return name, json.loads(args_raw)
        except json.JSONDecodeError:
            return name, {}

    return name, args_raw

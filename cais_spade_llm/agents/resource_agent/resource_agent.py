from __future__ import annotations

"""Resource agent that receives work orders, selects a tool via LLM, and executes it."""

# agents/resource_agent/resource_agent.py
import asyncio
import inspect
import json
import time
import uuid
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

    def configure_nominal(self, context: Any) -> None:
        """Attach an explicit nominal context with this agent's exact resource_id."""
        if context.resource_id != self.agent_name:
            raise ValueError("Nominal resource_id must match the ResourceAgent name exactly")
        self.nominal_context = context

    def nominal_des_model(self) -> dict[str, Any]:
        """Expose configured nominal capabilities without dispatching a task."""
        context = getattr(self, "nominal_context", None)
        if context is None:
            raise ValueError("ResourceAgent has no configured nominal context")
        return context.nominal_des_model()

    def validate_nominal_event(self, models: dict, valuation: dict, task: dict) -> dict:
        """Delegate nominal feasibility to this resource's configured context."""
        context = getattr(self, "nominal_context", None)
        if context is None:
            raise ValueError("ResourceAgent has no configured nominal context")
        return context.validate_nominal_event(models, valuation, task)

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
        self._recovery_composition_grants: dict[str, dict[str, Any]] = {}
        self._consumed_recovery_composition_refs: dict[str, dict[str, Any]] = {}
        self._pending_recovery_composition_refs: dict[str, dict[str, Any]] = {}
        self._pending_recovery_composition_request_ids: dict[str, str] = {}
        self._recovery_composition_observation_senders: dict[str, Any] = {}
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

        if getattr(self, "environment_context", None) is not None:
            from cais_spade_llm.agents.shared_information.environment_capabilities import CapabilityRequestInbox

            self.add_behaviour(CapabilityRequestInbox(), Template(metadata={"type": "capability_request"}))

        # Tasks from ProductAgent
        t_task = Template()
        t_task.set_metadata("type", "task")
        self.add_behaviour(self._TaskInbox(), t_task)

        # Safety decisions from CCA
        t_safety = Template()
        t_safety.set_metadata("type", "safety_decision")
        self.add_behaviour(self._SafetyDecisionInbox(), t_safety)

    def _snapshot_state(self) -> dict[str, Any]:
        """
        Best-effort snapshot of resource state for failure context.
        Subclasses can override to provide richer state.
        """
        context = getattr(self, 'environment_context', None) or getattr(self, 'nominal_context', None)
        if context is not None:
            values = context.snapshot()
            return {'current_state': values.get('resource_state', 'idle'), **values}
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

    def get_recovery_physical_snapshot(self) -> dict[str, Any]:
        """Export existing resource evidence without asserting physical completeness."""
        return {
            "resource_jid": str(self.jid),
            "resource_id": self.agent_name,
            "snapshot": deepcopy(self._snapshot_state()),
            "evidence": {"source": "ResourceAgent._snapshot_state"},
        }

    def capture_recovery_safety_state(self, *, max_age: float = 2.) -> dict:
        """Report missing observation support without inventing physical facts."""
        observer = getattr(self, 'recovery_safety_observer', None)
        if observer is not None:
            return observer.capture(max_age=max_age)
        return {"status": "NEEDS_CONTEXT", "resource_id": self.agent_name,
                "reason": "Resource-owned safety observation support unavailable"}

    def prepare_recovery_safety_program(self, program: dict, checkpoint: dict) -> dict:
        """Retain an unprepared request when this owner has no preparation hook."""
        from cais_spade_llm.recovery_framework import fingerprint

        return {"status": "NEEDS_CONTEXT", "resource_id": self.agent_name,
                "resource_jid": str(self.jid), "program": deepcopy(program),
                "program_fingerprint": fingerprint(program), "checkpoint_id": checkpoint["checkpoint_id"],
                "steps": [], "reason": "Resource-owned non-dispatching preparation unavailable"}

    def recovery_safety_configuration(self) -> dict:
        """Expose owner configuration without assuming a robot controller."""
        observer = getattr(self, 'recovery_safety_observer', None)
        if observer is not None:
            return {"static_capabilities": deepcopy(self.static_capabilities),
                    "observation_configuration": deepcopy(observer.configuration)}
        return {"static_capabilities": deepcopy(self.static_capabilities)}

    def get_recovery_safety_primitive_model(self):
        """Return no physical model until the owner explicitly supplies one."""
        return None

    def get_recovery_entity_reader(self):
        """Return no scene reader until an initialized owner provides one."""
        observer = getattr(self, 'recovery_safety_observer', None)
        if observer is not None:
            return observer.reader
        return None

    def validate_recovery_safety_step(self, planned: dict, step: dict) -> None:
        """Reject preparation references without owner-specific validation."""
        raise ValueError("Resource-owned prepared-step validation unavailable")

    async def prepare_recovery_composition_evidence(
        self, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Ask an owner-configured provider to prepare, without executing primitives.

        Request payloads cannot install a provider or enable mock execution. The
        CCA independently decides whether the provider's execution mode is usable.

        Args:
            request: Exact CCA task, primitive program, and schedule bindings.

        Returns:
            Owner-provided preparation evidence, or an explicit NEEDS_CONTEXT
            record when preparation or matching execution evidence is absent.
        """
        physical_snapshot = self.get_recovery_physical_snapshot()
        unavailable = {
            "status": "NEEDS_CONTEXT",
            "resource_jid": str(self.jid),
            "reason": "Resource-owned nonexecuting physical preparation is unavailable",
            "physical_snapshot": physical_snapshot,
        }
        if not isinstance(request, dict) or request.get("resource_jid") != str(self.jid):
            return {**unavailable, "reason": "Preparation resource_jid does not match this resource"}
        provider = getattr(self, "recovery_composition_evidence_provider", None)
        prepare = getattr(provider, "prepare", None)
        if not callable(prepare):
            return unavailable
        result = prepare(
            resource_agent=self,
            request=deepcopy(request),
            physical_snapshot=deepcopy(physical_snapshot),
        )
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, dict):
            return {**unavailable, "reason": "Preparation provider returned no evidence record"}
        result = deepcopy(result)
        if result.get("status") == "prepared" and (
            result.get("resource_jid") != str(self.jid)
            or result.get("program_hash") != request.get("program_hash")
            or result.get("execution_mode") not in {"live", "mock"}
            or not isinstance(result.get("preparation_id"), str)
            or not result["preparation_id"]
        ):
            return {**unavailable, "reason": "Preparation identity or execution evidence is incomplete"}
        result["request_fingerprint"] = recovery_validation_fingerprint(request)
        return result

    def recovery_execution_primitive_catalog(self) -> list[dict[str, Any]]:
        """Return the resource-owned execution primitive catalog."""
        if self._recovery_execution_primitive_catalog_cache is None:
            from cais_spade_llm.resources.resource_primitives import (
                build_execution_primitive_catalog,
            )

            self._recovery_execution_primitive_catalog_cache = build_execution_primitive_catalog(self)
        return deepcopy(self._recovery_execution_primitive_catalog_cache)

    def supports_recovery_composition_execution(self) -> bool:
        """Require an executor that implements grant consumption and feedback."""
        return getattr(self.execute_recovery_macro, "__func__", None) is ResourceAgent.execute_recovery_macro

    async def execute_recovery_composition_step(
        self, *, primitive: str, params: dict, task_id: str, step_index: int,
        grant: dict, owner: Any,
    ) -> Any:
        """Execute an already validated step on the configured primitive owner."""
        fn = getattr(owner, primitive, None) or getattr(self, primitive, None)
        if not callable(fn):
            raise ValueError("Registered primitive executor is unavailable")
        result = fn(**params)
        return await result if inspect.isawaitable(result) else result

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

    def check_recovery_primitive_feasibility(
        self,
        *,
        primitive: str,
        params: dict[str, Any],
        task: dict[str, Any],
        recovery_snapshot: dict[str, Any],
        part_context: dict[str, Any],
        grounded_action: dict[str, Any],
        resource_evidence: dict[str, Any] | None = None,
        physical_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Require a resource-owned evaluator for a bound primitive."""
        del primitive, params, task, recovery_snapshot, part_context, grounded_action, resource_evidence, physical_evidence
        return {
            "allowed": False,
            "feasibility_status": "NEEDS_CONTEXT",
            "constraint_code": "resource_validation_unavailable",
            "reason": "ResourceAgent has no primitive capability evaluator",
        }

    def get_recovery_resource_evidence(self) -> dict[str, Any]:
        """Read existing owner values without observing equipment or applying effects."""
        result = {"state": {}, "source": {"resource_jid": str(self.jid)}, "sources": {}}
        for field in ("_task_ctx", "_held_part", "_position"):
            if hasattr(self, field):
                result["state"][field] = deepcopy(getattr(self, field))
                result["sources"]["state/" + field] = "ResourceAgent." + field
        for field in ("controller_config", "gripper_config", "motion_config", "execution_mode"):
            if hasattr(self, field):
                result[field] = deepcopy(getattr(self, field))
                result["sources"][field] = "ResourceAgent." + field
        result["resource_type"] = getattr(self, "static_capabilities", {}).get("resource_type")
        return result

    def prepare_recovery_primitive_parameters(
        self, *, task: dict[str, Any], recovery_des_model: dict[str, Any],
        recovery_snapshot: dict[str, Any], physical_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Bind declared expressions to recorded facts on the validating owner."""
        from cais_spade_llm.resources.recovery_parameter_evidence import prepare_resource_primitive_parameters

        return prepare_resource_primitive_parameters(
            task=task, recovery_des_model=recovery_des_model, recovery_snapshot=recovery_snapshot,
            resource_evidence=physical_input.get("resource_evidence") or {}, resource_jid=str(self.jid))


    def validate_recovery_primitive_composition(self, **kwargs: Any) -> dict[str, Any]:
        """Validate an authored composition using this resource's recorded contracts."""
        from cais_spade_llm.resources.recovery_composition_owner import validate_owner_composition

        return validate_owner_composition(self, **kwargs)

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
            composition_owner=self,
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
        reference = kwargs.get("recovery_composition_ref")
        grant = None
        if reference is not None:
            if str(task_id) in self._consumed_recovery_composition_refs:
                return {"status": "failed:recovery_composition_grant_consumed",
                        "content": "Registered recovery task was already attempted; retry requires a new task ID"}
            grant = self._recovery_composition_grants.pop(str(task_id), None)
            if (not isinstance(grant, dict)
                    or grant.get("recovery_composition_ref") != reference
                    or grant.get("primitive_steps") != steps
                    or not isinstance(grant.get("resolved_primitive_steps"), list)
                    or len(grant["resolved_primitive_steps"]) != len(steps)
                    or str(task_id) not in self._recovery_composition_observation_senders):
                return {"status": "failed:recovery_composition_grant",
                        "content": "Registered recovery requires a matching CCA execution grant"}
            self._consumed_recovery_composition_refs[str(task_id)] = deepcopy(reference)
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
        if grant is not None and steps != grant["primitive_steps"]:
            return {"status": "failed:recovery_composition_grant",
                    "content": "Expanded primitive program differs from the CCA grant"}
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

            if grant is not None and grant["resolved_primitive_steps"][step_idx] != {
                "primitive": primitive, "params": params
            }:
                return {
                    "status": "failed:recovery_composition_bindings",
                    "content": "Resolved primitive parameters differ from the CCA grant",
                    "observations": {"step_index": step_idx, "primitive": primitive,
                                     "params": deepcopy(params), "completed_steps": len(results)},
                }

            started_at = time.monotonic()
            try:
                if grant is not None:
                    step_result = await self.execute_recovery_composition_step(
                        primitive=primitive, params=params, task_id=str(task_id),
                        step_index=step_idx, grant=grant, owner=owner,
                    )
                else:
                    maybe_result = fn(**params)
                    step_result = await maybe_result if inspect.isawaitable(maybe_result) else maybe_result
            except Exception as exc:
                if grant is not None:
                    await self._recovery_composition_observation_senders[str(task_id)]({
                        "step_index": step_idx, "primitive": primitive, "params": deepcopy(params),
                        "result": {"success": False, "error": f"{type(exc).__name__}: {exc}"},
                        "physical_snapshot": self.get_recovery_physical_snapshot(),
                        "started_at_monotonic": started_at, "finished_at_monotonic": time.monotonic(),
                    })
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

            if grant is not None:
                await self._recovery_composition_observation_senders[str(task_id)]({
                    "step_index": step_idx, "primitive": primitive, "params": deepcopy(params),
                    "result": deepcopy(step_result),
                    "physical_snapshot": self.get_recovery_physical_snapshot(),
                    "started_at_monotonic": started_at, "finished_at_monotonic": time.monotonic(),
                })

            if isinstance(step_result, dict):
                normalized_result = dict(step_result)
                normalized_result.setdefault("success", grant is None)
            elif isinstance(step_result, bool):
                normalized_result = {
                    "success": step_result,
                    "message": f"{primitive} {'ok' if step_result else 'failed'}",
                }
            elif isinstance(step_result, list):
                normalized_result = {
                    "success": grant is None,
                    "message": f"{primitive} returned {len(step_result)} items",
                    "data": step_result,
                }
            else:
                normalized_result = {
                    "success": grant is None,
                    "message": f"{primitive} completed",
                    "data": step_result,
                }

            results.append({"primitive": primitive, "result": normalized_result})
            step_succeeded = (
                normalized_result.get("success") is True
                if grant is not None else bool(normalized_result.get("success", False))
            )
            if not step_succeeded:
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
            if primitive_meta is not None and grant is None:
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

        if out_state and grant is None:
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
            if protocol == "environment":
                from cais_spade_llm.recovery_framework.environment_runtime import execute_environment_task

                await execute_environment_task(self, msg, data)
                return
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
            is_recovery_macro = fn_name == "execute_recovery_macro"
            reference = fn_args.get("recovery_composition_ref")
            if task_id in agent._consumed_recovery_composition_refs:
                await self._ack(msg, task_id=task_id, status="blocked",
                                content="Registered recovery task was already attempted")
                return
            diagnostic_macro = bool(
                is_recovery_macro and fn_args.get("validation_policy") == "no_validation"
                and reference is None and not fn_args.get("recovery_safety_scope_id")
            )
            if is_recovery_macro and not diagnostic_macro:
                required = ("recovery_id", "task_id", "outline_id", "program_hash", "problem_id")
                if (not isinstance(reference, dict)
                        or any(not isinstance(reference.get(key), str) or not reference[key] for key in required)
                        or reference["task_id"] != task_id):
                    await self._ack(msg, task_id=task_id, status="blocked",
                                    content="Validated recovery requires a CCA composition reference")
                    return
                fn_args["start_safety_mode"] = "cca_check"
                start_safety_mode = "cca_check"
            start_guard = getattr(agent, "recovery_composition_start_guard", None)
            if callable(start_guard) and start_guard():
                fn_args["start_safety_mode"] = "cca_check"
                start_safety_mode = "cca_check"
            use_fast_path = start_safety_mode == "fast_path" and (
                not is_recovery_macro or diagnostic_macro
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

            if reference is not None and (
                getattr(func, "__func__", None) is not getattr(agent.execute_recovery_macro, "__func__", None)
                or not agent.supports_recovery_composition_execution()
            ):
                await self._ack(msg, task_id=task_id, status="blocked",
                                content="Resource executor cannot report registered primitive evidence")
                return
            agent._safety_decisions.pop(task_id, None)
            agent._recovery_composition_grants.pop(task_id, None)
            recovery_composition_request_id = None
            if reference is not None:
                agent._pending_recovery_composition_refs[task_id] = deepcopy(reference)
                recovery_composition_request_id = uuid.uuid4().hex
                agent._pending_recovery_composition_request_ids[task_id] = recovery_composition_request_id

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
                            **({"recovery_composition_request_id": recovery_composition_request_id}
                               if recovery_composition_request_id is not None else {}),
                        }
                    )
                    await send_agent_message(
                        self,
                        resource_msg,
                        transport_label="resource_safety",
                    )
                except Exception:
                    agent.logger.exception(
                        "[Resource] Failed to send resource_event to CCA."
                    )
                    agent._pending_recovery_composition_refs.pop(task_id, None)
                    agent._pending_recovery_composition_request_ids.pop(task_id, None)
                    await self._ack(msg, task_id=task_id, status="blocked")
                    return

            # 2) Wait for CCA decision (instant for recovery macros, blocks for normal tasks)
            decision = await agent._wait_for_safety_decision(task_id)

            agent._pending_recovery_composition_refs.pop(task_id, None)
            agent._pending_recovery_composition_request_ids.pop(task_id, None)
            grant = agent._recovery_composition_grants.get(task_id)
            if decision != "allow" or (reference is not None and (
                not isinstance(grant, dict)
                or grant.get("recovery_composition_ref") != reference
                or grant.get("primitive_steps") != fn_args.get("primitive_steps")
            )):
                agent._recovery_composition_grants.pop(task_id, None)
                await self._ack(msg, task_id=task_id, status="blocked")
                return

            primitive_observations = []
            physical_identity = ({"run_id": grant["run_id"],
                                  "recovery_composition_ref": deepcopy(reference)}
                                 if reference is not None and isinstance(grant, dict) and grant.get("run_id") else {})

            async def report_primitive_observation(observation: dict[str, Any]) -> None:
                primitive_observations.append(deepcopy(observation))
                observed_msg = Message(to=agent.cca_jid)
                observed_msg.set_metadata("type", "resource_event")
                observed_msg.body = json.dumps({
                    **physical_identity, "task_id": task_id, "resource_jid": str(agent.jid),
                    "function_name": fn_name, "params": fn_args,
                    "status": "recovery_primitive_observed", "observations": observation,
                })
                await send_agent_message(self, observed_msg, transport_label="recovery_primitive_observed")

            if reference is not None:
                agent._recovery_composition_observation_senders[task_id] = report_primitive_observation

            if not use_fast_path:
                # ---------------------------
                #  SAFETY PASSED → RUNNING
                # ---------------------------
                await self._ack(msg, task_id=task_id, status="running")

                running_msg = Message(to=agent.cca_jid)
                running_msg.set_metadata("type", "resource_event")
                running_msg.body = json.dumps(
                    {
                        **physical_identity, "task_id": task_id,
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

                if physical_identity:
                    execution = asyncio.create_task(func(**filtered_args))
                    try:
                        result = await asyncio.wait_for(asyncio.shield(execution), timeout=agent.tool_timeout_s)
                    except asyncio.TimeoutError:
                        # A motion may still be executing in its native owner. Keep
                        # its grant and claims until its actual outcome is known.
                        await execution
                        raise
                else:
                    result = await asyncio.wait_for(func(**filtered_args), timeout=agent.tool_timeout_s)
                if reference is not None and isinstance(result, dict):
                    result.setdefault("observations", {})["primitive_observations"] = deepcopy(primitive_observations)
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
                                **physical_identity, "task_id": task_id,
                                "resource_jid": str(agent.jid),
                                "function_name": fn_name,
                                "params": fn_args,
                                "status": final_status,  # e.g. "completed", "blocked", etc.
                                "failure_context": failure_context,
                                "current_state": state_after.get("current_state", "idle"),
                                "observations": deepcopy((result or {}).get("observations")),
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
            finally:
                agent._recovery_composition_observation_senders.pop(task_id, None)
                agent._recovery_composition_grants.pop(task_id, None)

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
                            **physical_identity, "task_id": task_id,
                            "resource_jid": str(agent.jid),
                            "function_name": fn_name,
                            "params": fn_args,
                            "status": final_status,
                            "failure_context": failure_context,
                            "current_state": state_after.get("current_state", "idle"),
                            "observations": {"primitive_observations": deepcopy(primitive_observations),
                                             "result": deepcopy(result)},
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


    class _SafetyDecisionInbox(CyclicBehaviour):
        """Receives safety_decision messages from CCA and stores them on the agent."""

        async def run(self) -> None:
            agent: ResourceAgent = self.agent  # type: ignore

            msg = await self.receive(timeout=0.05)
            if not msg:
                return

            if (
                str(msg.sender).split("/", 1)[0] != str(agent.cca_jid).split("/", 1)[0]
            ):
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

            reference = agent._pending_recovery_composition_refs.get(task_id)
            if reference is not None:
                request_id = agent._pending_recovery_composition_request_ids.get(task_id)
                if not request_id or data.get("recovery_composition_request_id") != request_id:
                    return
            elif data.get("recovery_composition_request_id"):
                return
            elif data.get("recovery_composition_grant") is not None:
                runtime = getattr(agent, "environment_runtime", None)
                grant = data["recovery_composition_grant"]
                task = runtime.context.pending_for(task_id) if runtime is not None else None
                if (not isinstance(grant, dict) or task is None
                        or grant.get("nominal_task") != task
                        or grant.get("run_id") != runtime.context.run_id
                        or task.get("resource_id") != agent.agent_name
                        or runtime.jids.get(agent.agent_name) != str(agent.jid)):
                    return
                if decision == "allow":
                    agent._recovery_composition_grants[task_id] = deepcopy(grant)
            if reference is not None and decision == "allow":
                grant = data.get("recovery_composition_grant")
                if not isinstance(grant, dict) or grant.get("recovery_composition_ref") != reference:
                    decision = "block"
                else:
                    agent._recovery_composition_grants[task_id] = deepcopy(grant)
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

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
)
from cais_spade_llm.resources.capability_engine import (
    compound_recovery_state,
    compound_state_values,
    configured_recovery_state,
)


def _public_configured_state(
    capabilities: dict[str, Any],
    state: dict[str, Any],
    *,
    include_part_fields: bool,
) -> dict[str, Any]:
    state_variables = dict(capabilities.get("state_variables") or {})
    public_state = {
        str(field_name): deepcopy(value)
        for field_name, value in state.items()
        if (
            dict(state_variables.get(str(field_name)) or {}).get("private")
            is not True
            and (
                include_part_fields
                or str(
                    dict(
                        state_variables.get(str(field_name)) or {}
                    ).get("scope")
                    or "resource"
                )
                != "part"
            )
        )
    }
    return compound_recovery_state(public_state)


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
        from cais_spade_llm.resources.capability_engine import (
            configured_capability_errors,
        )

        self._configured_capability_declarations = {
            "state_variables": deepcopy(
                self.static_capabilities.get("state_variables")
            ),
            "events": deepcopy(self.static_capabilities.get("events")),
        }
        runtime_fact_names = self.capability_runtime_fact_names()
        executable_required_facts = {
            str(event.get("event_name") or ""): (
                self._executable_required_runtime_facts(
                    self.executables.get(str(event.get("event_name") or ""))
                )
            )
            for event in (
                self._configured_capability_declarations.get("events") or []
            )
            if isinstance(event, dict)
        }
        capability_errors = configured_capability_errors(
            self._configured_capability_declarations,
            executable_names=set(self.executables),
            runtime_fact_names=runtime_fact_names,
            executable_required_facts=executable_required_facts,
        )
        if capability_errors:
            raise ValueError(
                "Invalid static_capabilities: " + "; ".join(capability_errors)
            )
        ResourceAgent.public_locations(
            self,
            resource_snapshot={"resource_jid": str(jid)},
            part_contexts=[],
        )
        ResourceAgent.public_action_target(
            self,
            resource_snapshot={"resource_jid": str(jid)},
        )
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
        """Return the current resource-snapshot recovery snapshot for this resource."""
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

    @classmethod
    def capability_runtime_fact_names(cls) -> set[str]:
        """Return exact runtime fact names supported by this resource type."""
        return {"resource_jid"}

    @staticmethod
    def _executable_required_runtime_facts(executable: Any) -> set[str]:
        """Return exact required argument names for one executable."""
        if not callable(executable):
            return set()
        tool_spec = getattr(executable, "__tool_spec__", None)
        required_names = getattr(tool_spec, "required_argument_names", None)
        if callable(required_names):
            return {
                str(name)
                for name in required_names()
                if str(name)
            }
        try:
            signature = inspect.signature(executable)
        except (TypeError, ValueError):
            return set()
        return {
            str(name)
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.default is inspect.Parameter.empty
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }

    @staticmethod
    def _executable_runtime_fact_names(executable: Any) -> set[str]:
        """Return exact argument names accepted by one executable."""
        if not callable(executable):
            return set()
        tool_spec = getattr(executable, "__tool_spec__", None)
        arguments = getattr(tool_spec, "arguments", None)
        if isinstance(arguments, tuple):
            return {
                str(getattr(argument, "name", "") or "")
                for argument in arguments
                if str(getattr(argument, "name", "") or "")
            }
        try:
            signature = inspect.signature(executable)
        except (TypeError, ValueError):
            return set()
        return {
            str(name)
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }

    def capability_runtime_facts(
        self,
        *,
        task: dict[str, Any],
        resource_snapshot: dict[str, Any],
        part_context: dict[str, Any],
        grounded_action: dict[str, Any],
    ) -> dict[str, Any]:
        """Publish exact named facts available to configured capabilities."""
        del grounded_action
        supported_names = self.capability_runtime_fact_names()
        facts: dict[str, Any] = {
            "resource_jid": str(
                resource_snapshot.get("resource_jid")
                or self.jid
                or ""
            ).strip()
        }
        for fact_name in supported_names - {"resource_jid"}:
            for source in (part_context, resource_snapshot):
                if fact_name in source and source.get(fact_name) not in (None, ""):
                    facts[fact_name] = deepcopy(source.get(fact_name))
                    break
        return facts

    def capability_runtime_fact_tables(
        self,
        *,
        task: dict[str, Any],
        resource_snapshot: dict[str, Any],
        part_context: dict[str, Any],
        grounded_action: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return ordered trusted fact alternatives for capability evaluation."""
        facts = self.capability_runtime_facts(
            task=task,
            resource_snapshot=resource_snapshot,
            part_context=part_context,
            grounded_action=grounded_action,
        )
        return [facts] if isinstance(facts, dict) else []

    def capability_state_valuation(
        self,
        *,
        capabilities: dict[str, Any],
        resource_snapshot: dict[str, Any],
        part_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Project exact configured fields from this resource's live context."""
        valuation: dict[str, Any] = {}
        for field_name, declaration in dict(
            capabilities.get("state_variables") or {}
        ).items():
            source = (
                part_context
                if str(dict(declaration or {}).get("scope") or "resource")
                == "part"
                else resource_snapshot
            )
            valuation[str(field_name)] = deepcopy(source.get(str(field_name)))
        return valuation

    def public_resource_state_context(
        self,
        *,
        resource_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Return non-authorable public context for one resource projection."""
        del resource_snapshot
        return {}

    def public_part_state_context(
        self,
        *,
        resource_snapshot: dict[str, Any],
        part_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Return non-authorable public context for one part projection."""
        del resource_snapshot, part_context
        return {}

    def public_capability_state_projections(
        self,
        *,
        resource_snapshot: dict[str, Any],
        part_contexts: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Return public configured state values for exact resource bindings."""
        capabilities = self._bind_capabilities(snapshot=resource_snapshot)
        state_variables = dict(capabilities.get("state_variables") or {})
        if not state_variables:
            raise ValueError("configured state_variables is unavailable")

        public_resource_fields: list[str] = []
        public_part_fields: list[str] = []
        for raw_field_name, raw_declaration in state_variables.items():
            field_name = str(raw_field_name)
            declaration = dict(raw_declaration or {})
            if declaration.get("private") is True:
                continue
            if str(declaration.get("scope") or "resource") == "part":
                public_part_fields.append(field_name)
            else:
                public_resource_fields.append(field_name)
        if not public_resource_fields and not public_part_fields:
            raise ValueError("configured state_variables has no public fields")

        def _flat_public_state(
            *,
            part_context: dict[str, Any],
            selected_fields: list[str],
        ) -> dict[str, Any]:
            valuation = self.capability_state_valuation(
                capabilities=capabilities,
                resource_snapshot=resource_snapshot,
                part_context=part_context,
            )
            state = {
                field_name: deepcopy(valuation.get(field_name))
                for field_name in selected_fields
            }
            invalid_fields = [
                field_name
                for field_name, value in state.items()
                if not isinstance(value, (str, int, float, bool, type(None)))
            ]
            if invalid_fields:
                raise ValueError(
                    "public capability state is not scalar-valued: "
                    + ", ".join(invalid_fields)
                )
            return compound_recovery_state(state)

        resource_jid = str(
            resource_snapshot.get("resource_jid")
            or self.jid
            or ""
        ).strip()
        if not resource_jid:
            raise ValueError("resource_jid is unavailable")

        resource_context_provider = getattr(
            self,
            "public_resource_state_context",
            None,
        )
        resource_context = (
            resource_context_provider(resource_snapshot=resource_snapshot)
            if callable(resource_context_provider)
            else {}
        )
        if not isinstance(resource_context, dict):
            raise ValueError("public resource state context is malformed")
        projections = [
            {
                "resource_jid": resource_jid,
                "state": _flat_public_state(
                    part_context={},
                    selected_fields=public_resource_fields,
                ),
                **deepcopy(resource_context),
            }
        ]
        if not public_part_fields:
            return projections
        normalized_part_contexts = sorted(
            (
                deepcopy(part_context)
                for part_context in (part_contexts or [])
                if isinstance(part_context, dict)
                and str(part_context.get("part_name") or "").strip()
            ),
            key=lambda part_context: str(
                part_context.get("part_name") or ""
            ),
        )
        seen_part_names: set[str] = set()
        for part_context in normalized_part_contexts:
            part_name = str(part_context.get("part_name") or "").strip()
            if part_name in seen_part_names:
                raise ValueError(
                    f"duplicate part_name capability-state binding: {part_name}"
                )
            seen_part_names.add(part_name)
            part_context_provider = getattr(
                self,
                "public_part_state_context",
                None,
            )
            public_part_context = (
                part_context_provider(
                    resource_snapshot=resource_snapshot,
                    part_context=part_context,
                )
                if callable(part_context_provider)
                else {}
            )
            if not isinstance(public_part_context, dict):
                raise ValueError("public part state context is malformed")
            projections.append(
                {
                    "resource_jid": resource_jid,
                    "part_name": part_name,
                    "state": _flat_public_state(
                        part_context=part_context,
                        selected_fields=public_part_fields,
                    ),
                    **deepcopy(public_part_context),
                }
            )
        return projections

    def public_state_values(
        self,
        *,
        resource_snapshot: dict[str, Any],
        part_contexts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return configured condition values and public relation-field values."""
        capabilities = self._bind_capabilities(snapshot=resource_snapshot)
        state_variables = dict(capabilities.get("state_variables") or {})
        if not state_variables:
            raise ValueError("configured state_variables is unavailable")

        # Location fields are published alongside their condition so each
        # compound state carries its own configured vocabulary. A part location
        # is not the resource workspace list: it admits observation-derived
        # references and the holder this resource binds at runtime.
        public_fields = [
            str(field_name)
            for field_name, declaration in state_variables.items()
            if dict(declaration or {}).get("private") is not True
        ]
        if not public_fields:
            raise ValueError("configured state_variables has no public fields")

        del part_contexts

        resource_jid = str(
            resource_snapshot.get("resource_jid")
            or self.jid
            or ""
        ).strip()
        if not resource_jid:
            raise ValueError("resource_jid is unavailable")

        def _append_exact(values: list[Any], value: Any) -> None:
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError(
                    "public state values are not scalar-valued"
                )
            if not any(
                type(existing) is type(value) and existing == value
                for existing in values
            ):
                values.append(deepcopy(value))

        def _runtime_fact_tables(
            part_context: dict[str, Any],
        ) -> list[dict[str, Any]]:
            part_name = str(part_context.get("part_name") or "").strip()
            task = {
                "resource_jid": resource_jid,
                **({"part_name": part_name} if part_name else {}),
            }
            rows = self.capability_runtime_fact_tables(
                task=task,
                resource_snapshot=resource_snapshot,
                part_context=part_context,
                grounded_action={},
            )
            if not isinstance(rows, list) or not all(
                isinstance(row, dict) for row in rows
            ):
                raise ValueError(
                    "Resource Agent runtime fact tables are malformed"
                )
            return [deepcopy(row) for row in rows]

        fact_tables = _runtime_fact_tables({})

        state_values: dict[str, Any] = {}
        for field_name in public_fields:
            declaration = dict(state_variables.get(field_name) or {})
            values: list[Any] = []
            for domain_value in declaration.get("domain") or []:
                _append_exact(values, domain_value)
            parameter_values = [
                str(fact_name)
                for fact_name in (declaration.get("parameter_values") or [])
                if str(fact_name)
            ]
            for fact_table in fact_tables:
                for fact_name in parameter_values:
                    if fact_name == "part_name":
                        continue
                    if fact_name in fact_table:
                        _append_exact(values, fact_table.get(fact_name))
            state_values[field_name] = (
                {
                    "values": values,
                    "template": "part_name",
                }
                if "part_name" in parameter_values
                else values
            )

        return {
            "resource_jid": resource_jid,
            "state_values": compound_state_values(state_values),
        }

    def public_locations(
        self,
        *,
        resource_snapshot: dict[str, Any],
        part_contexts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return configured named locations available to this resource."""
        del part_contexts
        capabilities = self._bind_capabilities(snapshot=resource_snapshot)
        state_variables = dict(capabilities.get("state_variables") or {})
        resource_location = dict(
            state_variables.get("resource_location") or {}
        )
        locations: list[dict[str, Any]] = []
        for value in resource_location.get("domain") or []:
            if not isinstance(value, str) or not value:
                continue
            locations.append({"location": value})

        location_areas = self.static_capabilities.get("location_areas")
        if location_areas is not None and not isinstance(location_areas, dict):
            raise ValueError("location_areas capability data is malformed")
        configured_locations = {
            str(row.get("location") or "")
            for row in locations
            if str(row.get("location") or "")
        }
        for raw_location, raw_area in (location_areas or {}).items():
            location = str(raw_location or "")
            if location not in configured_locations or not isinstance(
                raw_area,
                dict,
            ):
                raise ValueError("location_areas capability data is malformed")
            area = dict(raw_area)
            if not str(area.get("frame") or "") or not str(
                area.get("units") or ""
            ):
                raise ValueError("location_areas capability data is malformed")
            bounds = area.get("bounds")
            if not isinstance(bounds, dict) or not bounds:
                raise ValueError("location_areas capability data is malformed")
            for raw_field_name, raw_interval in bounds.items():
                field_name = str(raw_field_name or "")
                if not field_name or not isinstance(raw_interval, dict):
                    raise ValueError(
                        "location_areas capability data is malformed"
                    )
                interval = dict(raw_interval)
                minimum = interval.get("min")
                maximum = interval.get("max")
                if (
                    not isinstance(minimum, (int, float))
                    or isinstance(minimum, bool)
                    or not isinstance(maximum, (int, float))
                    or isinstance(maximum, bool)
                    or float(minimum) > float(maximum)
                ):
                    raise ValueError(
                        "location_areas capability data is malformed"
                    )
        for row in locations:
            location = str(row.get("location") or "")
            area = dict((location_areas or {}).get(location) or {})
            if area:
                row["area"] = deepcopy(area)

        resource_jid = str(
            resource_snapshot.get("resource_jid")
            or self.jid
            or ""
        ).strip()
        if not resource_jid:
            raise ValueError("resource_jid is unavailable")
        return {
            "resource_jid": resource_jid,
            "locations": locations,
        }

    def public_action_target(
        self,
        *,
        resource_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Return this resource's configured public action-target contract."""
        resource_jid = str(
            resource_snapshot.get("resource_jid")
            or self.jid
            or ""
        ).strip()
        if not resource_jid:
            raise ValueError("resource_jid is unavailable")
        raw_contract = self.static_capabilities.get("action_target")
        if raw_contract is None:
            return {}
        if not isinstance(raw_contract, dict):
            raise ValueError("action_target capability data is malformed")
        fields = raw_contract.get("fields")
        required = raw_contract.get("required")
        if (
            not isinstance(fields, dict)
            or not fields
            or not isinstance(required, list)
            or not required
        ):
            raise ValueError("action_target capability data is malformed")
        normalized_fields: dict[str, dict[str, Any]] = {}
        for raw_field_name, raw_declaration in fields.items():
            field_name = str(raw_field_name or "")
            declaration = dict(raw_declaration or {})
            field_type = str(declaration.get("type") or "")
            if not field_name or field_type not in {
                "string",
                "number",
                "integer",
                "boolean",
            }:
                raise ValueError("action_target capability data is malformed")
            normalized_fields[field_name] = {"type": field_type}
        normalized_required = [str(name or "") for name in required]
        if (
            any(not name for name in normalized_required)
            or len(set(normalized_required)) != len(normalized_required)
            or any(name not in normalized_fields for name in normalized_required)
        ):
            raise ValueError("action_target capability data is malformed")
        return {
            "resource_jid": resource_jid,
            "action_target": {
                "fields": normalized_fields,
                "required": normalized_required,
                **(
                    {"frame": str(raw_contract.get("frame"))}
                    if str(raw_contract.get("frame") or "")
                    else {}
                ),
                **(
                    {"units": str(raw_contract.get("units"))}
                    if str(raw_contract.get("units") or "")
                    else {}
                ),
            },
        }

    def _generated_successor_from_action_target(
        self,
        *,
        task: dict[str, Any],
        initial_valuation: dict[str, Any],
        successor: dict[str, Any],
        runtime_fact_tables: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Apply a resource-owned action target to a generated successor."""
        del initial_valuation, runtime_fact_tables
        action_target = task.get("action_target")
        if not isinstance(action_target, dict) or not action_target:
            return deepcopy(successor), None
        return deepcopy(successor), {
            "constraint_code": "unsupported_resource_target",
            "reason": "this Resource Agent does not support an action_target",
            "evidence": {"resource_jid": str(self.jid)},
        }

    def _configured_event_required_runtime_facts(
        self,
        event: dict[str, Any],
    ) -> set[str]:
        from cais_spade_llm.resources.capability_engine import (
            configured_event_runtime_fact_names,
        )

        event_name = str(event.get("event_name") or "")
        required = self._executable_required_runtime_facts(
            self.executables.get(event_name)
        )
        required.update(configured_event_runtime_fact_names(event))
        if event.get("requires_part_binding") is True:
            required.add("part_name")
        return required

    def capability_goal_requirements(
        self,
        *,
        capabilities: dict[str, Any],
        goal_conditions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Bind exact nominal goal fields owned by this resource type."""
        state_variables = dict(capabilities.get("state_variables") or {})
        requirements: list[dict[str, Any]] = []
        for condition in goal_conditions:
            field_name = str(condition.get("field") or "").strip()
            declaration = dict(state_variables.get(field_name) or {})
            if not declaration or isinstance(condition.get("expected"), dict):
                continue
            entity_kind = str(
                condition.get("entity_kind") or ""
            ).strip().lower()
            scope = str(declaration.get("scope") or "resource").strip()
            if entity_kind in {"part", "resource"} and scope != entity_kind:
                continue
            requirements.append(
                {
                    "condition_id": str(
                        condition.get("condition_id")
                        or condition.get("id")
                        or ""
                    ).strip(),
                    "entity_kind": entity_kind or scope,
                    "entity": str(condition.get("entity") or "").strip(),
                    "field": field_name,
                    "expected": deepcopy(condition.get("expected")),
                }
            )
        return requirements

    def _capability_execution_arguments(
        self,
        *,
        event_name: str,
        runtime_facts: dict[str, Any],
    ) -> dict[str, Any]:
        executable = self.executables.get(event_name)
        accepted_names = self._executable_runtime_fact_names(executable)
        return {
            fact_name: deepcopy(runtime_facts.get(fact_name))
            for fact_name in accepted_names
            if runtime_facts.get(fact_name) not in (None, "")
        }

    def _capability_atomic_transition(
        self,
        *,
        event_name: str,
        before: dict[str, Any],
        successor: dict[str, Any],
        runtime_facts: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "event_name": event_name,
            "before": deepcopy(before),
            "after": deepcopy(successor),
            "execution_arguments": self._capability_execution_arguments(
                event_name=event_name,
                runtime_facts=runtime_facts,
            ),
        }

    def _future_goal_capability_instances(
        self,
        *,
        resource_snapshot: dict[str, Any],
        part_contexts: list[dict[str, Any]],
        goal_conditions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return private configured atomic instances relevant to nominal goals."""
        from cais_spade_llm.resources.capability_engine import (
            capability_state_mismatches,
            configured_event_successor,
        )

        capabilities = self._bind_capabilities(snapshot=resource_snapshot)
        state_variables = dict(capabilities.get("state_variables") or {})
        requirements = self.capability_goal_requirements(
            capabilities=capabilities,
            goal_conditions=goal_conditions,
        )
        if not requirements:
            return []

        normalized_parts = sorted(
            (
                deepcopy(row)
                for row in part_contexts
                if isinstance(row, dict)
                and str(row.get("part_name") or "").strip()
            ),
            key=lambda row: str(row.get("part_name") or ""),
        )
        results: list[dict[str, Any]] = []
        result_index_by_event_id: dict[str, int] = {}
        for event in capabilities.get("events") or []:
            if not isinstance(event, dict) or event.get("controllable") is False:
                continue
            event_name = str(event.get("event_name") or "").strip()
            if not event_name:
                continue
            contexts = (
                normalized_parts
                if event.get("requires_part_binding") is True
                else [{}]
            )
            updated_fields = set(dict(event.get("updates") or {}))
            for part_context in contexts:
                part_name = str(
                    part_context.get("part_name") or ""
                ).strip()
                task: dict[str, Any] = {
                    "event_name": event_name,
                    "resource_jid": str(self.jid),
                }
                if part_name:
                    task["part_name"] = part_name
                initial_valuation = self.capability_state_valuation(
                    capabilities=capabilities,
                    resource_snapshot=resource_snapshot,
                    part_context=part_context,
                )
                runtime_fact_tables = self.capability_runtime_fact_tables(
                    task=task,
                    resource_snapshot=resource_snapshot,
                    part_context=part_context,
                    grounded_action={},
                )
                for runtime_facts in runtime_fact_tables or [{}]:
                    successor_result = configured_event_successor(
                        event,
                        initial_valuation,
                        state_variables=state_variables,
                        runtime_facts=runtime_facts,
                        required_runtime_facts=(
                            self._configured_event_required_runtime_facts(event)
                        ),
                        evaluate_guards=False,
                    )
                    if successor_result.get("enabled") is not True:
                        continue
                    successor = deepcopy(
                        successor_result.get("successor") or {}
                    )
                    enabled_result = configured_event_successor(
                        event,
                        initial_valuation,
                        state_variables=state_variables,
                        runtime_facts=runtime_facts,
                        required_runtime_facts=(
                            self._configured_event_required_runtime_facts(event)
                        ),
                        evaluate_guards=True,
                    )
                    resource_enabled = bool(
                        enabled_result.get("enabled") is True
                    )
                    matched_requirements = [
                        requirement
                        for requirement in requirements
                        if str(requirement.get("field") or "") in updated_fields
                        and (
                            str(
                                requirement.get("entity_kind") or ""
                            ).strip()
                            != "part"
                            or (
                                part_name
                                and str(requirement.get("entity") or "").strip()
                                in {"", part_name}
                            )
                        )
                        and (
                            str(
                                requirement.get("entity_kind") or ""
                            ).strip()
                            != "resource"
                            or str(requirement.get("entity") or "").strip()
                            in {"", str(self.jid)}
                        )
                        and not capability_state_mismatches(
                            {
                                str(requirement.get("field") or ""): deepcopy(
                                    requirement.get("expected")
                                )
                            },
                            successor,
                        )
                    ]
                    if not matched_requirements:
                        continue
                    goal_effect = {
                        field_name: deepcopy(successor.get(field_name))
                        for field_name in sorted(
                            {
                                str(requirement.get("field") or "")
                                for requirement in matched_requirements
                                if str(requirement.get("field") or "")
                            }
                        )
                    }
                    event_id = json.dumps(
                        {
                            "resource_jid": str(self.jid),
                            "event_name": event_name,
                            **(
                                {"part_name": part_name}
                                if part_name
                                else {}
                            ),
                            "goal_effect": goal_effect,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    include_part_fields = bool(part_name)
                    public_before = _public_configured_state(
                        capabilities,
                        initial_valuation,
                        include_part_fields=include_part_fields,
                    )
                    public_successor = _public_configured_state(
                        capabilities,
                        successor,
                        include_part_fields=include_part_fields,
                    )
                    future_task = {
                        **task,
                        "expected_start_state": public_before,
                        "expected_end_state": public_successor,
                    }
                    part_scope_changed = bool(
                        part_name
                        and any(
                            str(
                                dict(
                                    state_variables.get(field_name) or {}
                                ).get("scope")
                                or "resource"
                            )
                            == "part"
                            and initial_valuation.get(field_name)
                            != successor.get(field_name)
                            for field_name in updated_fields
                        )
                    )
                    result_row = {
                        "event_id": event_id,
                        "task": future_task,
                        "resource_enabled": resource_enabled,
                        "signature": {
                            "inferable_primary_part": part_name or None,
                            "task_kind": (
                                "part_handling"
                                if part_name
                                else "resource_action"
                            ),
                            "changes_part_world": part_scope_changed,
                        },
                        "_cca_atomic_transitions": [
                            {
                                "event_name": event_name,
                                "before": deepcopy(initial_valuation),
                                "after": deepcopy(successor),
                            }
                        ],
                    }
                    existing_index = result_index_by_event_id.get(event_id)
                    if existing_index is None:
                        result_index_by_event_id[event_id] = len(results)
                        results.append(result_row)
                    elif (
                        resource_enabled
                        and results[existing_index].get("resource_enabled")
                        is not True
                    ):
                        results[existing_index] = result_row
        return results

    def _resolve_generated_capability(
        self,
        *,
        capabilities: dict[str, Any],
        task: dict[str, Any],
        initial_valuation: dict[str, Any],
        expected_end_state: dict[str, Any],
        runtime_fact_tables: list[dict[str, Any]],
        grounded_action: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate a consistent generated successor without configured search."""
        successor = deepcopy(initial_valuation)
        successor.update(deepcopy(expected_end_state))
        successor, target_finding = (
            self._generated_successor_from_action_target(
                task=task,
                initial_valuation=initial_valuation,
                successor=successor,
                runtime_fact_tables=runtime_fact_tables,
            )
        )
        if target_finding is not None:
            return {
                "allowed": False,
                "constraint_code": str(
                    target_finding.get("constraint_code")
                    or "unsupported_resource_target"
                ),
                "reason": str(
                    target_finding.get("reason")
                    or "resource action target is unsupported"
                ),
                "evidence": deepcopy(target_finding.get("evidence") or {}),
                "calculated_successor": {},
                "_atomic_transitions": [],
            }
        consistency_finding = self._generated_successor_consistency_finding(
            capabilities=capabilities,
            task=task,
            initial_valuation=initial_valuation,
            successor=successor,
            runtime_fact_tables=runtime_fact_tables,
            grounded_action=grounded_action,
        )
        if consistency_finding is not None:
            return {
                "allowed": False,
                "constraint_code": str(
                    consistency_finding.get("constraint_code")
                    or "generated_successor_inconsistent"
                ),
                "reason": str(
                    consistency_finding.get("reason")
                    or "generated successor is inconsistent"
                ),
                "evidence": deepcopy(
                    consistency_finding.get("evidence") or {}
                ),
                "calculated_successor": {},
                "_atomic_transitions": [],
            }
        event_name = str(task.get("event_name") or "").strip()
        return {
            "allowed": True,
            "constraint_code": "",
            "reason": "generated successor satisfies resource-local consistency",
            "evidence": {},
            "calculated_successor": successor,
            "_atomic_transitions": [
                {
                    "event_name": event_name,
                    "before": deepcopy(initial_valuation),
                    "after": deepcopy(successor),
                }
            ],
        }

    def _generated_successor_consistency_finding(
        self,
        *,
        capabilities: dict[str, Any],
        task: dict[str, Any],
        initial_valuation: dict[str, Any],
        successor: dict[str, Any],
        runtime_fact_tables: list[dict[str, Any]],
        grounded_action: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return a resource-local inconsistency in a generated successor."""
        del (
            capabilities,
            task,
            initial_valuation,
            successor,
            runtime_fact_tables,
            grounded_action,
        )
        return None

    def _evaluate_capability_transition(
        self,
        *,
        task: dict[str, Any],
        resource_snapshot: dict[str, Any],
        part_context: dict[str, Any] | None = None,
        grounded_action: dict[str, Any] | None = None,
        _runtime_fact_tables: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Evaluate one exact or generated event using resource-owned facts."""
        from cais_spade_llm.resources.capability_engine import (
            capability_state_error,
            capability_state_mismatches,
            configured_event_successor,
        )

        capabilities = self._bind_capabilities(snapshot=resource_snapshot)
        state_variables = dict(capabilities.get("state_variables") or {})
        events = [
            deepcopy(event)
            for event in (capabilities.get("events") or [])
            if isinstance(event, dict)
        ]
        if not state_variables or not events:
            return {
                "allowed": False,
                "constraint_code": "resource_validation_unavailable",
                "reason": "configured static_capabilities is unavailable",
                "evidence": {},
                "calculated_successor": {},
                "_atomic_transitions": [],
            }
        normalized_part_context = deepcopy(part_context or {})
        include_part_fields = bool(
            str(
                normalized_part_context.get("part_name")
                or task.get("part_name")
                or ""
            ).strip()
        )
        normalized_grounded_action = deepcopy(grounded_action or {})
        if _runtime_fact_tables is None:
            table_provider = getattr(
                self,
                "capability_runtime_fact_tables",
                None,
            )
            runtime_fact_tables = (
                table_provider(
                    task=task,
                    resource_snapshot=resource_snapshot,
                    part_context=normalized_part_context,
                    grounded_action=normalized_grounded_action,
                )
                if callable(table_provider)
                else [
                    self.capability_runtime_facts(
                        task=task,
                        resource_snapshot=resource_snapshot,
                        part_context=normalized_part_context,
                        grounded_action=normalized_grounded_action,
                    )
                ]
            )
        else:
            runtime_fact_tables = deepcopy(_runtime_fact_tables)
        if not isinstance(runtime_fact_tables, list) or not all(
            isinstance(row, dict) for row in runtime_fact_tables
        ):
            return {
                "allowed": False,
                "constraint_code": "resource_validation_unavailable",
                "reason": "Resource Agent runtime fact tables are malformed",
                "evidence": {},
                "calculated_successor": {},
                "_atomic_transitions": [],
            }
        runtime_fact_tables = [
            deepcopy(row) for row in runtime_fact_tables
        ]
        initial_valuation = self.capability_state_valuation(
            capabilities=capabilities,
            resource_snapshot=resource_snapshot,
            part_context=normalized_part_context,
        )
        expected_start_state = configured_recovery_state(
            dict(task.get("expected_start_state") or {})
        )
        expected_end_state = configured_recovery_state(
            dict(task.get("expected_end_state") or {})
        )
        event_name = str(task.get("event_name") or "")
        exact_event = next(
            (
                event
                for event in events
                if str(event.get("event_name") or "") == event_name
            ),
            None,
        )
        for state_name, state in (
            ("expected_start_state", expected_start_state),
            ("resource_valuation", initial_valuation),
        ):
            state_error = capability_state_error(
                capabilities,
                state_name=state_name,
                state=state,
                runtime_fact_tables=runtime_fact_tables,
                current_valuation=initial_valuation,
                allow_generated_values=True,
            )
            if state_error is not None:
                state_error["calculated_successor"] = {}
                state_error["_atomic_transitions"] = []
                return state_error
        end_state_error = capability_state_error(
            capabilities,
            state_name="expected_end_state",
            state=expected_end_state,
            runtime_fact_tables=runtime_fact_tables,
            current_valuation=initial_valuation,
            allow_generated_values=False,
        )
        if end_state_error is not None:
            end_state_error["calculated_successor"] = {}
            end_state_error["_atomic_transitions"] = []
            return end_state_error
        start_mismatches = capability_state_mismatches(
            expected_start_state,
            initial_valuation,
        )
        if start_mismatches:
            return {
                "allowed": False,
                "constraint_code": "expected_start_state_mismatch",
                "reason": (
                    "expected_start_state does not match the Resource Agent "
                    "valuation"
                ),
                "evidence": {"mismatches": start_mismatches},
                "calculated_successor": {},
                "_atomic_transitions": [],
            }

        if exact_event is None:
            if not expected_end_state:
                return {
                    "allowed": False,
                    "constraint_code": "generated_event_target_unavailable",
                    "reason": (
                        "a generated event requires expected_end_state"
                    ),
                    "evidence": {"event_name": event_name},
                    "calculated_successor": {},
                    "_atomic_transitions": [],
                }
            generated_result = self._resolve_generated_capability(
                capabilities=capabilities,
                task=task,
                initial_valuation=initial_valuation,
                expected_end_state=expected_end_state,
                runtime_fact_tables=runtime_fact_tables,
                grounded_action=normalized_grounded_action,
            )
            generated_result["calculated_successor"] = (
                _public_configured_state(
                    capabilities,
                    dict(generated_result.get("calculated_successor") or {}),
                    include_part_fields=include_part_fields,
                )
            )
            return generated_result

        first_disabled_result: dict[str, Any] | None = None
        first_effect_mismatch: dict[str, Any] | None = None
        for runtime_facts in runtime_fact_tables or [{}]:
            successor_result = configured_event_successor(
                exact_event,
                initial_valuation,
                state_variables=state_variables,
                runtime_facts=runtime_facts,
                required_runtime_facts=(
                    self._configured_event_required_runtime_facts(exact_event)
                ),
            )
            if successor_result.get("enabled") is not True:
                if first_disabled_result is None:
                    first_disabled_result = deepcopy(successor_result)
                continue
            successor = deepcopy(successor_result.get("successor") or {})
            atomic_transition = self._capability_atomic_transition(
                event_name=event_name,
                before=initial_valuation,
                successor=successor,
                runtime_facts=runtime_facts,
            )
            end_mismatches = capability_state_mismatches(
                expected_end_state,
                successor,
            )
            if not end_mismatches:
                return {
                    "allowed": True,
                    "constraint_code": "",
                    "reason": (
                        "configured capability reaches expected_end_state"
                    ),
                    "evidence": {},
                    "calculated_successor": _public_configured_state(
                        capabilities,
                        successor,
                        include_part_fields=include_part_fields,
                    ),
                    "_atomic_transitions": [atomic_transition],
                }
            if first_effect_mismatch is None:
                first_effect_mismatch = {
                    "successor": successor,
                    "atomic_transition": atomic_transition,
                    "mismatches": end_mismatches,
                }
        if first_effect_mismatch is not None:
            return {
                "allowed": False,
                "constraint_code": "transition_effect_mismatch",
                "reason": (
                    "configured capability successor does not match "
                    "expected_end_state"
                ),
                "evidence": {
                    "event_name": event_name,
                    "mismatches": deepcopy(
                        first_effect_mismatch["mismatches"]
                    ),
                },
                "calculated_successor": _public_configured_state(
                    capabilities,
                    dict(first_effect_mismatch["successor"]),
                    include_part_fields=include_part_fields,
                ),
                "_atomic_transitions": [
                    deepcopy(first_effect_mismatch["atomic_transition"])
                ],
            }
        disabled_result = dict(first_disabled_result or {})
        return {
            "allowed": False,
            "constraint_code": str(
                disabled_result.get("constraint_code")
                or "runtime_fact_unavailable"
            ),
            "reason": str(
                disabled_result.get("reason")
                or "configured capability runtime facts are unavailable"
            ),
            "evidence": {"event_name": event_name},
            "calculated_successor": {},
            "_atomic_transitions": [],
        }

    def _enabled_capability_instances(
        self,
        *,
        resource_snapshot: dict[str, Any],
        part_contexts: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve enabled grounded capabilities in configured order."""
        capabilities = self._bind_capabilities(snapshot=resource_snapshot)
        normalized_parts = sorted(
            (
                deepcopy(row)
                for row in (part_contexts or [])
                if isinstance(row, dict)
                and str(row.get("part_name") or "").strip()
            ),
            key=lambda row: str(row.get("part_name") or ""),
        )
        results: list[dict[str, Any]] = []
        seen_instance_keys: set[str] = set()
        for event in capabilities.get("events") or []:
            if not isinstance(event, dict):
                continue
            contexts = (
                normalized_parts
                if event.get("requires_part_binding") is True
                else [{}]
            )
            for part_context in contexts:
                task: dict[str, Any] = {
                    "event_name": str(event.get("event_name") or ""),
                    "resource_jid": str(self.jid),
                }
                part_name = str(part_context.get("part_name") or "").strip()
                if part_name:
                    task["part_name"] = part_name
                table_provider = getattr(
                    self,
                    "capability_runtime_fact_tables",
                    None,
                )
                runtime_fact_tables = (
                    table_provider(
                        task=task,
                        resource_snapshot=resource_snapshot,
                        part_context=part_context,
                        grounded_action={},
                    )
                    if callable(table_provider)
                    else [
                        self.capability_runtime_facts(
                            task=task,
                            resource_snapshot=resource_snapshot,
                            part_context=part_context,
                            grounded_action={},
                        )
                    ]
                )
                for runtime_facts in runtime_fact_tables or [{}]:
                    transition = self._evaluate_capability_transition(
                        task=task,
                        resource_snapshot=resource_snapshot,
                        part_context=part_context,
                        _runtime_fact_tables=[runtime_facts],
                    )
                    if transition.get("allowed") is not True:
                        continue
                    atomic_transitions = [
                        deepcopy(row)
                        for row in (
                            transition.get("_atomic_transitions") or []
                        )
                        if isinstance(row, dict)
                    ]
                    initial = _public_configured_state(
                        capabilities,
                        dict(atomic_transitions[0].get("before") or {}),
                        include_part_fields=bool(part_name),
                    )
                    successor = deepcopy(
                        dict(
                            transition.get("calculated_successor") or {}
                        )
                    )
                    instance_key = json.dumps(
                        {
                            "resource_jid": str(self.jid),
                            "event_name": task["event_name"],
                            "part_name": part_name,
                            "successor": successor,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                    if instance_key in seen_instance_keys:
                        continue
                    seen_instance_keys.add(instance_key)
                    grounded_task = {
                        **deepcopy(task),
                        "expected_start_state": initial,
                        "expected_end_state": successor,
                    }
                    event_id = json.dumps(
                        {
                            "resource_jid": str(self.jid),
                            "event_name": grounded_task["event_name"],
                            **(
                                {"part_name": part_name}
                                if part_name
                                else {}
                            ),
                            "expected_end_state": successor,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                    results.append(
                        {
                            "event_id": event_id,
                            "task": grounded_task,
                            "part_context": deepcopy(part_context),
                            "transition_feasibility": {
                                key: deepcopy(value)
                                for key, value in transition.items()
                                if not str(key).startswith("_")
                            },
                            "_atomic_transitions": atomic_transitions,
                        }
                    )
        return results

    def _bind_capabilities(
        self,
        *,
        snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return this Resource Agent's private startup-loaded capabilities."""
        del snapshot
        from cais_spade_llm.resources.capability_engine import (
            bind_configured_capabilities,
        )

        return bind_configured_capabilities(
            self,
            capabilities=self._configured_capability_declarations,
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
    def recovery_physical_validation_snapshot(
        *,
        live_snapshot: dict[str, Any],
        physical_input: dict[str, Any],
        bound_capabilities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Overlay PA-projected dynamic facts on fresh RA capability evidence."""
        validation_snapshot = deepcopy(live_snapshot or {})
        if physical_input.get("use_projected_recovery_snapshot") is not True:
            return validation_snapshot
        projected_snapshot = physical_input.get("projected_recovery_snapshot")
        if not isinstance(projected_snapshot, dict):
            return validation_snapshot

        declared_fields = set(
            dict((bound_capabilities or {}).get("state_variables") or {})
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

    def _grounded_action_from_atomic_transitions(
        self,
        *,
        task: dict[str, Any],
        atomic_transitions: list[dict[str, Any]],
        part_context: dict[str, Any],
        provided_grounded_action: dict[str, Any],
    ) -> dict[str, Any]:
        """Add Resource Agent-calculated effects to physical validation input."""
        grounded_action = deepcopy(provided_grounded_action)
        resource_jid = str(task.get("resource_jid") or "").strip()
        part_name = str(task.get("part_name") or "").strip()
        grounded_action["resource_jid"] = resource_jid
        grounded_action["part_name"] = part_name or None
        grounded_action["task_kind"] = (
            "part_handling" if part_name else "resource_action"
        )
        if not atomic_transitions:
            return grounded_action
        state_variables = dict(
            self._configured_capability_declarations.get("state_variables")
            or {}
        )
        first_before = dict(atomic_transitions[0].get("before") or {})
        final_after = dict(atomic_transitions[-1].get("after") or {})
        resource_fields = {
            str(field_name)
            for field_name, declaration in state_variables.items()
            if str(dict(declaration or {}).get("scope") or "resource")
            == "resource"
        }
        part_fields = set(state_variables) - resource_fields
        resource_before = {
            field_name: deepcopy(first_before.get(field_name))
            for field_name in resource_fields
        }
        resource_after = {
            field_name: deepcopy(final_after.get(field_name))
            for field_name in resource_fields
        }
        part_before = {
            field_name: deepcopy(first_before.get(field_name))
            for field_name in part_fields
        }
        part_after = {
            field_name: deepcopy(final_after.get(field_name))
            for field_name in part_fields
        }
        changed_resource = {
            field_name: deepcopy(value)
            for field_name, value in resource_after.items()
            if resource_before.get(field_name) != value
        }
        changed_part = {
            field_name: deepcopy(value)
            for field_name, value in part_after.items()
            if part_before.get(field_name) != value
        }
        grounded_action["expected_effect"] = {
            "resource": {
                {
                    "resource_state": "current_state",
                    "resource_location": "location",
                }.get(field_name, field_name): deepcopy(value)
                for field_name, value in changed_resource.items()
            },
            "part": {
                {
                    "part_state": "state",
                    "part_location": "location",
                }.get(field_name, field_name): deepcopy(value)
                for field_name, value in changed_part.items()
            },
        }
        grounded_action["effect_scope"] = (
            "resource_and_part"
            if changed_resource and changed_part
            else "part_only"
            if changed_part
            else "resource_only"
        )
        grounded_action["task_kind"] = (
            "part_handling" if changed_part else "resource_action"
        )
        if (
            part_name
            and resource_after.get("held_part") == part_name
            and resource_before.get("held_part") != part_name
        ):
            grounded_action.setdefault("preconditions", {}).setdefault(
                "part",
                {},
            )["requires_acquisition"] = True
        preconditions = grounded_action.setdefault("preconditions", {})
        if changed_part:
            source_ref = preconditions.setdefault("source_ref", {})
            if source_ref.get("location") in (None, ""):
                source_ref["location"] = deepcopy(
                    part_context.get("part_location")
                    or part_context.get("current_location")
                )
            if source_ref.get("pose") in (None, "", {}):
                source_ref["pose"] = deepcopy(
                    part_context.get("observed_pose")
                    or part_context.get("pose")
                )
        else:
            preconditions.pop("source_ref", None)
            preconditions.pop("part", None)
            changed_resource_location = str(
                changed_resource.get("resource_location") or ""
            ).strip()
            part_context_location = str(
                part_context.get("part_location")
                or part_context.get("current_location")
                or ""
            ).strip()
            part_context_pose = (
                part_context.get("observed_pose")
                or part_context.get("pose")
            )
            if (
                changed_resource_location
                and changed_resource_location == part_context_location
                and isinstance(part_context_pose, dict)
            ):
                target = grounded_action.setdefault("target", {})
                target["target_location"] = changed_resource_location
                target["pose"] = deepcopy(part_context_pose)
            grounded_action["part_name"] = None
        return grounded_action

    def validate_recovery_outline_physical_candidates(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate transitions and physical feasibility against one fresh snapshot."""
        validator_jid = str(self.jid)
        snapshot = self.get_recovery_snapshot()
        bound_capabilities = self._bind_capabilities(snapshot=snapshot)
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
            validation_snapshot = snapshot
            if not self.recovery_validation_resource_matches(resource_jid):
                transition_result = {
                    "allowed": False,
                    "constraint_code": "wrong_resource_validator",
                    "reason": (
                        f"candidate resource_jid '{resource_jid}' does not match "
                        f"receiving ResourceAgent '{validator_jid}'"
                    ),
                    "evidence": {},
                    "calculated_successor": {},
                    "_atomic_transitions": [],
                }
                physical_result = {
                    "allowed": False,
                    "skipped": True,
                    "constraint_code": "",
                    "reason": "transition_feasibility rejected",
                }
            else:
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
                    bound_capabilities=bound_capabilities,
                )
                transition_result = self._evaluate_capability_transition(
                    task=task,
                    resource_snapshot=validation_snapshot,
                    part_context=deepcopy(
                        physical_input.get("part_context") or {}
                    ),
                    grounded_action=deepcopy(
                        physical_input.get("grounded_action") or {}
                    ),
                )
                if bool(transition_result.get("allowed")):
                    grounded_action = self._grounded_action_from_atomic_transitions(
                        task=task,
                        atomic_transitions=[
                            deepcopy(transition_row)
                            for transition_row in (
                                transition_result.get("_atomic_transitions") or []
                            )
                            if isinstance(transition_row, dict)
                        ],
                        part_context=deepcopy(
                            physical_input.get("part_context") or {}
                        ),
                        provided_grounded_action=deepcopy(
                            physical_input.get("grounded_action") or {}
                        ),
                    )
                    physical_result = self.check_recovery_physical_feasibility(
                        part_context=deepcopy(
                            physical_input.get("part_context") or {}
                        ),
                        recovery_snapshot=validation_snapshot,
                        grounded_action=grounded_action,
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
                                "ResourceAgent returned a malformed physical "
                                "validation result"
                            ),
                        }
                else:
                    physical_result = {
                        "allowed": False,
                        "skipped": True,
                        "constraint_code": "",
                        "reason": "transition_feasibility rejected",
                    }
            transition_allowed = bool(
                transition_result.get("allowed") is True
            )
            physical_allowed = bool(physical_result.get("allowed") is True)
            atomic_transitions = [
                deepcopy(transition_row)
                for transition_row in (
                    transition_result.get("_atomic_transitions") or []
                )
                if isinstance(transition_row, dict)
            ]
            public_transition_result = {
                key: deepcopy(value)
                for key, value in transition_result.items()
                if not str(key).startswith("_")
            }
            transition_findings: list[dict[str, Any]] = []
            if not transition_allowed:
                transition_findings.append(
                    {
                        "validation_category": "transition_feasibility",
                        "constraint_owner": "resource",
                        "constraint_family": "resource_transition",
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
                )
            physical_findings: list[dict[str, Any]] = []
            if transition_allowed and not physical_allowed:
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
            allowed = transition_allowed and physical_allowed
            results.append(
                {
                    "candidate_index": candidate_index,
                    "event_id": str(candidate.get("event_id") or "").strip(),
                    "allowed": allowed,
                    "findings": [
                        *transition_findings,
                        *physical_findings,
                    ],
                    "transition_feasibility": public_transition_result,
                    "transition_findings": transition_findings,
                    "physical_feasibility": deepcopy(physical_result),
                    "physical_findings": physical_findings,
                    "_cca_atomic_transitions": [
                        {
                            "event_name": str(
                                transition_row.get("event_name") or ""
                            ),
                            "before": deepcopy(
                                transition_row.get("before") or {}
                            ),
                            "after": deepcopy(
                                transition_row.get("after") or {}
                            ),
                        }
                        for transition_row in atomic_transitions
                    ],
                }
            )
        enabled_capability_results: list[dict[str, Any]] = []
        enabledness_query = payload.get("enabledness_query")
        if isinstance(enabledness_query, dict):
            projected_resource_snapshot = enabledness_query.get(
                "projected_resource_snapshot"
            )
            enabledness_snapshot = deepcopy(snapshot)
            if isinstance(projected_resource_snapshot, dict):
                enabledness_snapshot = self.recovery_physical_validation_snapshot(
                    live_snapshot=snapshot,
                    physical_input={
                        "use_projected_recovery_snapshot": True,
                        "projected_recovery_snapshot": projected_resource_snapshot,
                    },
                    bound_capabilities=bound_capabilities,
                )
            raw_part_contexts = enabledness_query.get("part_contexts")
            if isinstance(raw_part_contexts, dict):
                part_contexts = [
                    {
                        "part_name": str(part_name),
                        **deepcopy(dict(part_context or {})),
                    }
                    for part_name, part_context in raw_part_contexts.items()
                    if str(part_name) and isinstance(part_context, dict)
                ]
            elif isinstance(raw_part_contexts, list):
                part_contexts = [
                    deepcopy(row)
                    for row in raw_part_contexts
                    if isinstance(row, dict)
                ]
            else:
                part_contexts = []
            for enabled_index, instance in enumerate(
                self._enabled_capability_instances(
                    resource_snapshot=enabledness_snapshot,
                    part_contexts=part_contexts,
                )
            ):
                task = dict(instance.get("task") or {})
                part_context = deepcopy(instance.get("part_context") or {})
                grounded_action = self._grounded_action_from_atomic_transitions(
                    task=task,
                    atomic_transitions=[
                        deepcopy(transition_row)
                        for transition_row in (
                            instance.get("_atomic_transitions") or []
                        )
                        if isinstance(transition_row, dict)
                    ],
                    part_context=part_context,
                    provided_grounded_action={},
                )
                physical_result = self.check_recovery_physical_feasibility(
                    part_context=part_context,
                    recovery_snapshot=enabledness_snapshot,
                    grounded_action=grounded_action,
                    operation_kind=str(task.get("event_name") or ""),
                    part_name=(
                        str(task.get("part_name") or "").strip() or None
                    ),
                )
                if not isinstance(physical_result, dict):
                    physical_result = {
                        "allowed": False,
                        "constraint_code": "resource_validation_unavailable",
                        "reason": (
                            "ResourceAgent returned a malformed physical "
                            "validation result"
                        ),
                    }
                transition_result = deepcopy(
                    instance.get("transition_feasibility") or {}
                )
                enabled_capability_results.append(
                    {
                        "candidate_index": enabled_index,
                        "event_id": str(instance.get("event_id") or ""),
                        "task": task,
                        "allowed": bool(
                            transition_result.get("allowed") is True
                            and physical_result.get("allowed") is True
                        ),
                        "transition_feasibility": transition_result,
                        "physical_feasibility": deepcopy(physical_result),
                        "_cca_atomic_transitions": [
                            {
                                "event_name": str(
                                    transition_row.get("event_name") or ""
                                ),
                                "before": deepcopy(
                                    transition_row.get("before") or {}
                                ),
                                "after": deepcopy(
                                    transition_row.get("after") or {}
                                ),
                            }
                            for transition_row in (
                                instance.get("_atomic_transitions") or []
                            )
                            if isinstance(transition_row, dict)
                        ],
                    }
                )
        future_goal_capability_results: list[dict[str, Any]] = []
        future_goal_query = payload.get("future_goal_query")
        if isinstance(future_goal_query, dict):
            projected_resource_snapshot = future_goal_query.get(
                "projected_resource_snapshot"
            )
            future_goal_snapshot = deepcopy(snapshot)
            if isinstance(projected_resource_snapshot, dict):
                future_goal_snapshot = (
                    self.recovery_physical_validation_snapshot(
                        live_snapshot=snapshot,
                        physical_input={
                            "use_projected_recovery_snapshot": True,
                            "projected_recovery_snapshot": (
                                projected_resource_snapshot
                            ),
                        },
                        bound_capabilities=bound_capabilities,
                    )
                )
            raw_part_contexts = future_goal_query.get("part_contexts")
            future_part_contexts = [
                deepcopy(row)
                for row in (
                    raw_part_contexts
                    if isinstance(raw_part_contexts, list)
                    else []
                )
                if isinstance(row, dict)
                and str(row.get("part_name") or "").strip()
            ]
            raw_goal_conditions = future_goal_query.get("goal_conditions")
            goal_conditions = [
                deepcopy(row)
                for row in (
                    raw_goal_conditions
                    if isinstance(raw_goal_conditions, list)
                    else []
                )
                if isinstance(row, dict)
            ]
            future_goal_capability_results = (
                self._future_goal_capability_instances(
                    resource_snapshot=future_goal_snapshot,
                    part_contexts=future_part_contexts,
                    goal_conditions=goal_conditions,
                )
            )
        response = {
            "validator_jid": validator_jid,
            "snapshot": deepcopy(snapshot),
            "results": results,
            "enabled_capability_results": enabled_capability_results,
            "enabled_event_ids": sorted(
                {
                    str(row.get("event_id") or "").strip()
                    for row in [*results, *enabled_capability_results]
                    if bool(row.get("allowed"))
                    and str(row.get("event_id") or "").strip()
                }
            ),
        }
        if isinstance(future_goal_query, dict):
            response["future_goal_capability_results"] = (
                future_goal_capability_results
            )
        return response

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
        event_name: str = "",
        part_name: str | None = None,
        outline_expected_start_state: dict[str, Any] | None = None,
        expected_end_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generic recovery macro executor.

        Re-resolve and execute a configured capability sequence when supplied.
        The primitive sequence path remains available for resource-owned
        primitive programs that do not represent configured events.
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
        configured_event_name = str(event_name or "").strip()
        if configured_event_name:
            configured_task = deepcopy(kwargs)
            configured_task.update(
                {
                    "event_name": configured_event_name,
                    "resource_jid": str(self.jid),
                    "expected_start_state": deepcopy(
                        outline_expected_start_state or {}
                    ),
                    "expected_end_state": deepcopy(expected_end_state or {}),
                }
            )
            if part_name:
                configured_task["part_name"] = str(part_name)
            configured_part_context = deepcopy(
                dict(kwargs.get("part_context") or {})
            )
            for state_field, value in dict(
                outline_expected_start_state or {}
            ).items():
                declaration = dict(
                    self._configured_capability_declarations.get(
                        "state_variables",
                        {},
                    ).get(state_field)
                    or {}
                )
                if (
                    str(declaration.get("scope") or "resource") == "part"
                    and state_field not in configured_part_context
                ):
                    configured_part_context[state_field] = deepcopy(value)
            if part_name:
                configured_part_context.setdefault(
                    "part_name",
                    str(part_name),
                )
            transition = self._evaluate_capability_transition(
                task=configured_task,
                resource_snapshot=runtime_snapshot,
                part_context=configured_part_context,
            )
            if transition.get("allowed") is not True:
                return {
                    "status": "revalidation_required",
                    "content": (
                        f"Recovery macro '{macro_name}' requires ResourceAgent "
                        "revalidation: "
                        + str(
                            transition.get("reason")
                            or transition.get("constraint_code")
                            or "transition feasibility changed"
                        )
                    ),
                    "observations": {
                        "macro_name": macro_name,
                        "constraint_code": str(
                            transition.get("constraint_code") or ""
                        ),
                    },
                }

            configured_results: list[dict[str, Any]] = []
            atomic_transitions = [
                deepcopy(row)
                for row in (transition.get("_atomic_transitions") or [])
                if isinstance(row, dict)
            ]
            state_variables = dict(
                self._configured_capability_declarations.get(
                    "state_variables"
                )
                or {}
            )
            for transition_index, transition_row in enumerate(
                atomic_transitions
            ):
                atomic_event_name = str(
                    transition_row.get("event_name") or ""
                )
                execution_arguments = deepcopy(
                    dict(transition_row.get("execution_arguments") or {})
                )
                executable = self.executables.get(atomic_event_name)
                if not callable(executable):
                    return {
                        "status": "revalidation_required",
                        "content": (
                            f"Recovery macro '{macro_name}' requires "
                            "ResourceAgent revalidation: configured event "
                            f"'{atomic_event_name}' has no executable function"
                        ),
                        "observations": {
                            "macro_name": macro_name,
                            "completed_events": transition_index,
                        },
                    }
                try:
                    maybe_result = executable(**execution_arguments)
                    event_result = (
                        await maybe_result
                        if inspect.isawaitable(maybe_result)
                        else maybe_result
                    )
                except Exception as exc:
                    return {
                        "status": "failed",
                        "content": (
                            f"Recovery macro '{macro_name}' failed configured "
                            f"event {transition_index} ({atomic_event_name}): "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        "observations": {
                            "macro_name": macro_name,
                            "completed_events": transition_index,
                        },
                    }
                event_succeeded = (
                    bool(event_result)
                    if isinstance(event_result, bool)
                    else bool(event_result.get("success"))
                    if isinstance(event_result, dict)
                    and "success" in event_result
                    else str(event_result.get("status") or "")
                    in {"completed", "success"}
                    if isinstance(event_result, dict)
                    and "status" in event_result
                    else True
                )
                configured_results.append(
                    {
                        "event_name": atomic_event_name,
                        "result": deepcopy(event_result),
                    }
                )
                if not event_succeeded:
                    return {
                        "status": "failed",
                        "content": (
                            f"Recovery macro '{macro_name}' failed configured "
                            f"event {transition_index}: {atomic_event_name}"
                        ),
                        "observations": {
                            "macro_name": macro_name,
                            "completed_events": transition_index,
                            "results": configured_results,
                        },
                    }
                configured_successor = dict(
                    transition_row.get("after") or {}
                )
                synchronized_snapshot = deepcopy(runtime_snapshot)
                for state_field, value in configured_successor.items():
                    snapshot_field = {
                        "resource_state": "current_state",
                        "resource_location": "current_location",
                    }.get(state_field, state_field)
                    if str(
                        dict(state_variables.get(state_field) or {}).get(
                            "scope"
                        )
                        or "resource"
                    ) == "resource":
                        synchronized_snapshot[snapshot_field] = deepcopy(value)
                sync_agent_from_recovery_snapshot(
                    self,
                    synchronized_snapshot,
                )
                runtime_snapshot = synchronized_snapshot
            return {
                "status": "completed",
                "content": (
                    f"Recovery macro '{macro_name}' completed configured "
                    "capability sequence"
                ),
                "observations": {
                    "macro_name": macro_name,
                    "completed_events": len(configured_results),
                    "results": configured_results,
                },
            }

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
                "results": results,
                "enabled_capability_results": deepcopy(
                    validation.get("enabled_capability_results") or []
                ),
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

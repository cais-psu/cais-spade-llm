"""Bridge turn parsing and proposal normalization for LLM-guided recovery."""

from __future__ import annotations

from copy import deepcopy
import logging
import json
from typing import Any, Callable, Coroutine, Optional

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.bridge_adapters import (
    canonical_bridge_event,
    canonical_bridge_resource,
)
from cais_spade_llm.resources.resource_profile import (
    all_registered_operation_kinds,
    get_resource_profile,
    resource_snapshot_field_value,
    resource_snapshot_fields_map,
    resource_snapshot_set_field,
)
from cais_spade_llm.agents.intelligent_product.replanner.des_search.resource_bidding import (
    _tool_signature,
)

logger = logging.getLogger(__name__)


class _BridgeWarningCapture(logging.Handler):
    """Collect bridge-normalization warnings for temporary UI debugging."""

    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink.append(record.getMessage())
        except Exception:
            pass


_BRIDGE_TASK_PARAM_RESERVED_KEYS = frozenset(
    {
        "macro_name",
        "primitive_steps",
        "expected_start_state",
        "expected_snapshot",
        "projected_snapshot",
        "product_jid",
        "task_id",
        "out_state",
        "part_name",
        "touched_part",
    }
)


def _normalize_bridge_react_trace(raw: Any) -> dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    normalized: dict[str, Any] = {}
    for key in (
        "observed_facts",
        "gap_to_close",
        "decision_basis",
        "expected_progress",
    ):
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        cleaned = [
            str(item).strip()
            for item in values
            if str(item).strip()
        ][:6]
        if cleaned:
            normalized[key] = cleaned
    return normalized


def _normalize_bridge_outline_steps(raw_steps: Any) -> list[dict[str, str]]:
    if not isinstance(raw_steps, list):
        return []

    normalized: list[dict[str, str]] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue
        step_name = str(raw_step.get("step_name", "") or "").strip()
        objective = str(raw_step.get("objective", "") or "").strip()
        resource_jid = str(raw_step.get("resource_jid", "") or "").strip()
        part_name = _normalize_optional_name(raw_step.get("part_name", ""))
        operation_family = str(raw_step.get("operation_family", "") or "").strip()
        success_signal = str(raw_step.get("success_signal", "") or "").strip()
        rationale = str(raw_step.get("rationale", "") or "").strip()
        if not any(
            (
                step_name,
                objective,
                resource_jid,
                part_name,
                operation_family,
                success_signal,
                rationale,
            )
        ):
            continue
        entry: dict[str, str] = {}
        if step_name:
            entry["step_name"] = step_name
        if objective:
            entry["objective"] = objective
        if resource_jid:
            entry["resource_jid"] = resource_jid
        if part_name:
            entry["part_name"] = part_name
        if operation_family:
            entry["operation_family"] = operation_family
        if success_signal:
            entry["success_signal"] = success_signal
        if rationale:
            entry["rationale"] = rationale
        normalized.append(entry)
    return normalized

def _bridge_resource_entries(
    *,
    ra_jid: str,
    primitive_catalog: list[dict] | None,
    bridge_snapshot: dict[str, Any] | None,
    bridge_resources: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.primitive_semantics import (
        filter_synthesis_primitive_catalog,
    )

    resources: dict[str, dict[str, Any]] = {}
    for raw_jid, raw_entry in (bridge_resources or {}).items():
        if not isinstance(raw_entry, dict):
            continue
        resource_jid = str(raw_jid or raw_entry.get("resource_jid") or "").strip()
        if not resource_jid:
            continue
        resources[resource_jid] = {
            "resource_jid": resource_jid,
            "resource_type": str(
                raw_entry.get("resource_type")
                or dict(raw_entry.get("bridge_snapshot") or {}).get("resource_type")
                or "resource"
            ).strip(),
            "primitive_catalog": list(raw_entry.get("primitive_catalog") or []),
            "execution_primitive_catalog": list(
                raw_entry.get("execution_primitive_catalog")
                or raw_entry.get("primitive_catalog")
                or []
            ),
            "bridge_snapshot": dict(
                raw_entry.get("bridge_snapshot")
                or raw_entry.get("primitive_snapshot")
                or {}
            ),
            "resource_core": deepcopy(raw_entry.get("resource_core") or {}),
            "resource_facets": deepcopy(raw_entry.get("resource_facets") or {}),
            "modeled_state": deepcopy(raw_entry.get("modeled_state") or {}),
            "pending_tasks": deepcopy(raw_entry.get("pending_tasks") or []),
            "static_capabilities": deepcopy(raw_entry.get("static_capabilities") or {}),
            "bridge_adapter": deepcopy(raw_entry.get("bridge_adapter") or {}),
        }

    if not resources and primitive_catalog is not None:
        resource_jid = str(ra_jid or "").strip()
        if resource_jid:
            fallback_snapshot = canonical_bridge_resource(
                resource_jid=resource_jid,
                resource_type=str(
                    dict(bridge_snapshot or {}).get("resource_type")
                    or dict(dict(bridge_snapshot or {}).get("resource_core") or {}).get("resource_type")
                    or "resource"
                ),
                snapshot=deepcopy(bridge_snapshot or {}),
                modeled_state={},
            )
            resources[resource_jid] = {
                "resource_jid": resource_jid,
                "resource_type": str(
                    fallback_snapshot.get("resource_type")
                    or dict(fallback_snapshot.get("resource_core") or {}).get("resource_type")
                    or "resource"
                ).strip(),
                "primitive_catalog": filter_synthesis_primitive_catalog(primitive_catalog or []),
                "execution_primitive_catalog": list(primitive_catalog or []),
                "bridge_snapshot": fallback_snapshot,
                "resource_core": deepcopy(fallback_snapshot.get("resource_core") or {}),
                "resource_facets": deepcopy(fallback_snapshot.get("resource_facets") or {}),
                "modeled_state": {},
                "pending_tasks": [],
                "static_capabilities": {},
            }
    return resources


def _macro_tasks_from_primitive_proposal(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tasks = parsed.get("macro_tasks")
    if isinstance(raw_tasks, list) and raw_tasks:
        return [dict(task) for task in raw_tasks if isinstance(task, dict)]

    primitive_steps = parsed.get("primitive_steps") or parsed.get("steps") or []
    if isinstance(primitive_steps, list) and primitive_steps:
        return [dict(parsed)]
    return []


def _normalize_optional_name(value: Any) -> str:
    token = str(value or "").strip()
    if token.lower() in {"", "none", "null", "n/a"}:
        return ""
    return token


def _normalize_bridge_event_summary(
    raw_summary: Any,
    *,
    bridge_resources: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(raw_summary, list):
        return []

    normalized: list[dict[str, Any]] = []
    for raw_event in raw_summary:
        if not isinstance(raw_event, dict):
            continue
        event_name = str(raw_event.get("event_name", "")).strip()
        resource_jid = str(raw_event.get("resource_jid", "")).strip()
        part_name = _normalize_optional_name(raw_event.get("part_name", ""))
        rationale = str(raw_event.get("rationale", "")).strip()
        raw_closes = raw_event.get("closes_conditions") or []
        closes_conditions: list[dict[str, Any]] = []
        if isinstance(raw_closes, list):
            for raw_condition in raw_closes:
                if not isinstance(raw_condition, dict):
                    continue
                entity_kind = str(raw_condition.get("entity_kind", "")).strip()
                entity = str(raw_condition.get("entity", "")).strip()
                field = str(raw_condition.get("field", "")).strip()
                expected = raw_condition.get("expected")
                if not entity_kind or not entity or not field or expected in (None, ""):
                    continue
                closes_conditions.append(
                    {
                        "entity_kind": entity_kind,
                        "entity": entity,
                        "field": field,
                        "expected": deepcopy(expected),
                    }
                )
        if not any((event_name, resource_jid, part_name, closes_conditions, rationale)):
            continue

        # --- strategic state-delta fields (optional) ---
        raw_resource_delta = raw_event.get("expected_resource_delta")
        resource_delta: dict[str, str] | None = None
        if isinstance(raw_resource_delta, dict):
            delta_from = str(raw_resource_delta.get("from", "")).strip()
            delta_to = str(raw_resource_delta.get("to", "")).strip()
            if delta_from and delta_to:
                resource_delta = {"from": delta_from, "to": delta_to}

        raw_part_delta = raw_event.get("expected_part_delta")
        part_delta: dict[str, Any] | None = None
        if isinstance(raw_part_delta, dict):
            pd_part = _normalize_optional_name(raw_part_delta.get("part_name", "")) or part_name
            pd_from = str(raw_part_delta.get("from", "")).strip()
            pd_to = str(raw_part_delta.get("to", "")).strip()
            if pd_from and pd_to:
                part_delta = {"part_name": pd_part, "from": pd_from, "to": pd_to}
                pd_loc = str(raw_part_delta.get("location_to", "")).strip()
                if pd_loc:
                    part_delta["location_to"] = pd_loc

        entry: dict[str, Any] = {
            "event_name": event_name,
            "resource_jid": resource_jid,
            "part_name": part_name,
            "closes_conditions": closes_conditions,
            "rationale": rationale,
        }
        operation_family = str(raw_event.get("operation_family", "") or "").strip()
        if operation_family:
            entry["operation_family"] = operation_family
        if resource_delta is not None:
            entry["expected_resource_delta"] = resource_delta
        if part_delta is not None:
            entry["expected_part_delta"] = part_delta
        raw_projected_effects = raw_event.get("projected_effects")
        if isinstance(raw_projected_effects, dict) and raw_projected_effects:
            entry["projected_effects"] = deepcopy(raw_projected_effects)

        normalized_event = canonical_bridge_event(
            entry,
            bridge_resources=bridge_resources,
        )
        normalized_event["_operation_family_explicit"] = bool(operation_family)
        normalized.append(normalized_event)
    return normalized


def _bridge_event_contract_error(event: dict[str, Any]) -> str | None:
    if not bool(event.get("_operation_family_explicit")):
        return "bridge_events.events[].operation_family is required"

    operation_family = str(event.get("operation_family", "") or "").strip().lower()
    if not operation_family:
        return "bridge_events.events[].operation_family must not be empty"
    if operation_family not in all_registered_operation_kinds():
        return (
            f"bridge_events.events[].operation_family '{operation_family}' "
            "is not recognized for any registered resource profile"
        )
    return None


def _bridge_profile_event_contract_error(
    event: dict[str, Any],
    *,
    bridge_resources: dict[str, Any] | None = None,
) -> str | None:
    resource_jid = str(event.get("resource_jid", "") or "").strip()
    resource_entry = dict((bridge_resources or {}).get(resource_jid) or {})
    resource_type = str(
        resource_entry.get("resource_type")
        or dict(resource_entry.get("bridge_snapshot") or {}).get("resource_type")
        or event.get("resource_type")
        or ""
    ).strip()
    profile = get_resource_profile(resource_type or "resource")
    if profile.event_contract_validator is None:
        return None
    return profile.event_contract_validator(
        event=deepcopy(event),
        resource_jid=resource_jid,
        resource_entry=deepcopy(resource_entry),
        bridge_resources=deepcopy(bridge_resources or {}),
        profile=profile,
    )


def _normalize_primary_obligation(
    raw_primary: Any,
    obligation_targets: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    active_targets = [
        deepcopy(target)
        for target in (obligation_targets or [])
        if isinstance(target, dict)
        and str(target.get("rule_id", "")).strip()
        and str(target.get("resource_jid", "")).strip()
    ]
    if not active_targets:
        return None, None

    if raw_primary in (None, "", {}):
        if len(active_targets) == 1:
            return active_targets[0], None
        return None, "proposal must declare primary_obligation when multiple active obligation targets exist"

    if isinstance(raw_primary, str):
        rule_id = str(raw_primary).strip()
        matches = [target for target in active_targets if str(target.get("rule_id", "")).strip() == rule_id]
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, f"primary_obligation '{rule_id}' matched multiple active targets; resource_jid is required"
        return None, f"primary_obligation '{rule_id}' did not match any active obligation target"

    if not isinstance(raw_primary, dict):
        return None, "primary_obligation must be an object with rule_id/resource_jid"

    rule_id = str(raw_primary.get("rule_id", "")).strip()
    resource_jid = str(raw_primary.get("resource_jid", "")).strip()
    if not rule_id:
        return None, "primary_obligation.rule_id is required"
    if resource_jid:
        for target in active_targets:
            if (
                str(target.get("rule_id", "")).strip() == rule_id
                and str(target.get("resource_jid", "")).strip() == resource_jid
            ):
                return target, None
        return None, (
            f"primary_obligation ({rule_id}, {resource_jid}) did not match any active obligation target"
        )

    matches = [target for target in active_targets if str(target.get("rule_id", "")).strip() == rule_id]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, f"primary_obligation '{rule_id}' matched multiple active targets; resource_jid is required"
    return None, f"primary_obligation '{rule_id}' did not match any active obligation target"


def _normalize_bridge_task_metadata(raw_task_metadata: Any) -> dict[str, Any]:
    task_metadata = raw_task_metadata if isinstance(raw_task_metadata, dict) else {}
    task_metadata = dict(task_metadata)

    required_context_keys = task_metadata.get("required_context_keys") or []
    if not isinstance(required_context_keys, list):
        required_context_keys = []
    task_metadata["required_context_keys"] = [
        str(key).strip()
        for key in required_context_keys
        if str(key).strip()
    ]

    context_mapping = task_metadata.get("context_mapping") or {}
    if not isinstance(context_mapping, dict):
        context_mapping = {}
    task_metadata["context_mapping"] = dict(context_mapping)

    raw_part_transition = task_metadata.get("part_transition")
    normalized_part_transition: dict[str, Any] = {}
    if isinstance(raw_part_transition, dict):
        part_transition = dict(raw_part_transition)
        completed = part_transition.get("completed")
        if isinstance(completed, dict):
            normalized_part_transition["completed"] = dict(completed)
        else:
            shorthand_state = (
                part_transition.get("state")
                or part_transition.get("to")
                or part_transition.get("to_state")
            )
            if shorthand_state not in (None, ""):
                normalized_completed: dict[str, Any] = {"state": deepcopy(shorthand_state)}
                for key in (
                    "location_param",
                    "location_template",
                    "observation_required",
                    "last_known_param",
                    "last_known_template",
                ):
                    if key in part_transition:
                        normalized_completed[key] = deepcopy(part_transition[key])
                normalized_part_transition["completed"] = normalized_completed
    elif isinstance(raw_part_transition, str):
        text = raw_part_transition.strip()
        if "->" in text:
            _, _, rhs = text.rpartition("->")
            target_state = rhs.strip()
            if target_state:
                normalized_part_transition["completed"] = {"state": target_state}
    task_metadata["part_transition"] = normalized_part_transition
    return task_metadata


def _apply_part_transition_projection(
    projected_parts: dict[str, Any],
    *,
    part_name: str,
    task_metadata: dict[str, Any],
    task_params: dict[str, Any],
    resource_jid: str,
) -> None:
    if not part_name:
        return
    transition_map = task_metadata.get("part_transition") or {}
    transition = transition_map.get("completed")
    if not isinstance(transition, dict) or not transition:
        return

    entry = dict(projected_parts.get(part_name) or {})
    explicit_location = False
    if "state" in transition:
        entry["state"] = transition["state"]
    if "location_template" in transition:
        entry["location"] = str(transition["location_template"]).format(resource_jid=resource_jid)
        explicit_location = True
    if "location_param" in transition:
        entry["location"] = deepcopy(task_params.get(str(transition["location_param"])))
        explicit_location = True
    if "product_jid" in task_params and entry.get("state") == "assembled" and not explicit_location:
        product_location = str(task_params.get("product_jid") or "").strip()
        if product_location:
            entry["location"] = product_location.split("@", 1)[0]
    if entry.get("state") == "assembled" and not explicit_location:
        target = dict(entry.get("target") or {})
        target_location = str(target.get("location") or "").strip()
        if target_location:
            entry["location"] = target_location
    if transition.get("observation_required"):
        entry["observation_required"] = True
        entry["location"] = None
    if "last_known_param" in transition:
        entry["last_known_location"] = deepcopy(task_params.get(str(transition["last_known_param"])))
    if "last_known_template" in transition:
        entry["last_known_location"] = str(transition["last_known_template"]).format(
            resource_jid=resource_jid
        )
    projected_parts[part_name] = entry


def _projected_bridge_grounding_context(
    *,
    grounding_context: dict[str, Any],
    projected_resource_snapshots: dict[str, dict[str, Any]],
    projected_parts: dict[str, Any],
    bridge_resources: dict[str, dict[str, Any]],
    focused_resource_jid: str,
    primary_obligation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Expose the evolving projected whole-system state to later macro_tasks."""
    def _flat_facet_fields(snapshot: dict[str, Any]) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for facet in (snapshot.get("resource_facets") or {}).values():
            if not isinstance(facet, dict):
                continue
            for key, value in facet.items():
                flat[str(key)] = deepcopy(value)
        return flat

    context = deepcopy(grounding_context or {})
    context["parts"] = deepcopy(projected_parts)

    resources_payload: dict[str, dict[str, Any]] = {}
    for resource_jid, snapshot in (projected_resource_snapshots or {}).items():
        raw_entry = dict(bridge_resources.get(resource_jid) or {})
        bridge_snapshot = canonical_bridge_resource(
            resource_jid=resource_jid,
            resource_type=str(
                (dict(snapshot or {}).get("resource_core") or {}).get("resource_type")
                or raw_entry.get("resource_type")
                or raw_entry.get("bridge_snapshot", {}).get("resource_type")
                or "resource"
            ),
            snapshot=deepcopy(snapshot or {}),
            modeled_state=deepcopy(raw_entry.get("modeled_state") or {}),
        )
        profile = get_resource_profile(
            str(
                (bridge_snapshot.get("resource_core") or {}).get("resource_type")
                or bridge_snapshot.get("resource_type")
                or "resource"
            ).strip().lower()
            or "resource"
        )
        resources_payload[resource_jid] = {
            "jid": resource_jid,
            "resource_core": deepcopy(bridge_snapshot.get("resource_core") or {}),
            "resource_facets": deepcopy(bridge_snapshot.get("resource_facets") or {}),
            "resource_type": bridge_snapshot.get("resource_type"),
            "current_state": bridge_snapshot.get("current_state"),
            "current_location": deepcopy(bridge_snapshot.get("current_location")),
            "primitive_snapshot": bridge_snapshot,
            "modeled_state": deepcopy(raw_entry.get("modeled_state") or {}),
            "pending_tasks": deepcopy(raw_entry.get("pending_tasks") or []),
            "static_capabilities": deepcopy(raw_entry.get("static_capabilities") or {}),
            "bridge_adapter": deepcopy(raw_entry.get("bridge_adapter") or {}),
            **resource_snapshot_fields_map(
                bridge_snapshot,
                profile.snapshot_fields,
                profile=profile,
            ),
            **_flat_facet_fields(bridge_snapshot),
        }

    context["resources"] = resources_payload
    if focused_resource_jid:
        context["resource"] = deepcopy(resources_payload.get(focused_resource_jid) or {})
    if primary_obligation:
        context["primary_obligation"] = deepcopy(primary_obligation)
    return context


def _obligation_required_out_states(primary_obligation: dict[str, Any]) -> set[str]:
    required_states = {
        str(state).strip()
        for state in (primary_obligation.get("required_out_states") or [])
        if str(state).strip() and str(state).strip().lower() != "any"
    }
    if required_states:
        return required_states

    for ap in primary_obligation.get("required_state_aps") or []:
        if not isinstance(ap, dict):
            continue
        full = str(ap.get("full", "")).strip()
        segments = full.split("/", 5)
        if len(segments) != 6:
            continue
        prefix = str(segments[0]).strip().lower()
        symbol = str(segments[4]).strip()
        if prefix in {"ap_state", "sp"} and symbol and symbol.lower() != "any":
            required_states.add(symbol)

    if required_states:
        return required_states

    return {
        str(tool.get("out_state", "")).strip()
        for tool in (primary_obligation.get("candidate_tools") or [])
        if isinstance(tool, dict)
        and str(tool.get("out_state", "")).strip()
        and str(tool.get("out_state", "")).strip().lower() != "any"
    }


def _primary_obligation_projection_error(
    *,
    primary_obligation: dict[str, Any] | None,
    projected_resource_snapshots: dict[str, dict[str, Any]],
) -> str | None:
    if not primary_obligation:
        return None

    resource_jid = str(primary_obligation.get("resource_jid", "")).strip()
    if not resource_jid:
        return "primary_obligation.resource_jid is required for projection validation"

    projected_snapshot = dict(projected_resource_snapshots.get(resource_jid) or {})
    if not projected_snapshot:
        return f"primary obligation resource '{resource_jid}' has no projected final snapshot"

    required_out_states = _obligation_required_out_states(primary_obligation)
    if not required_out_states:
        return None

    final_state = str(projected_snapshot.get("current_state", "")).strip()
    if final_state in required_out_states:
        return None

    return (
        f"final projected state for primary obligation resource '{resource_jid}' "
        f"was '{final_state or 'unknown'}', expected one of {sorted(required_out_states)}"
    )


def normalize_bridge_turn_response(
    *,
    raw: str | dict[str, Any],
    available_resource_jids: list[str] | set[str],
    allowed_observation_primitives: list[str] | set[str],
    bridge_resources: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse one ReAct turn response into a strict observe/bridge_outline/bridge_events/final_plan union."""
    if isinstance(raw, dict):
        parsed = deepcopy(raw)
    else:
        try:
            parsed = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, "turn response was not valid JSON"

    if not isinstance(parsed, dict):
        return None, "turn response must be a JSON object"

    response_type = str(parsed.get("type", "")).strip().lower()
    allowed_resources = {
        str(resource_jid).strip()
        for resource_jid in (available_resource_jids or [])
        if str(resource_jid).strip()
    }
    allowed_primitives = {
        str(name).strip()
        for name in (allowed_observation_primitives or [])
        if str(name).strip()
    }

    if response_type == "observe":
        resource_jid = str(parsed.get("resource_jid", "")).strip()
        primitive = str(parsed.get("primitive", "")).strip()
        params = parsed.get("params") or {}
        if not resource_jid:
            return None, "observe.resource_jid is required"
        if allowed_resources and resource_jid not in allowed_resources:
            return None, f"observe.resource_jid '{resource_jid}' is not an available bridge resource"
        if not primitive:
            return None, "observe.primitive is required"
        if allowed_primitives and primitive not in allowed_primitives:
            return None, f"observe.primitive '{primitive}' is not allowed in this phase"
        if not isinstance(params, dict):
            return None, "observe.params must be an object"
        if str(parsed.get("store_as", "") or "").strip():
            return None, (
                "observe.store_as is no longer supported; "
                "observation keys are assigned automatically"
            )
        return {
            "type": "observe",
            "resource_jid": resource_jid,
            "primitive": primitive,
            "params": deepcopy(params),
            "reason_summary": str(parsed.get("reason_summary", "") or "").strip(),
            "react_trace": _normalize_bridge_react_trace(parsed.get("react_trace")),
        }, None

    if response_type == "bridge_outline":
        normalized_steps = _normalize_bridge_outline_steps(parsed.get("steps"))
        if not normalized_steps:
            return None, "bridge_outline.steps must contain at least one valid step"
        return {
            "type": "bridge_outline",
            "steps": deepcopy(normalized_steps),
            "reason_summary": str(parsed.get("reason_summary", "") or "").strip(),
            "react_trace": _normalize_bridge_react_trace(parsed.get("react_trace")),
        }, None

    if response_type == "bridge_events":
        normalized_events = _normalize_bridge_event_summary(
            parsed.get("events"),
            bridge_resources=bridge_resources,
        )
        if not normalized_events:
            return None, "bridge_events.events must contain at least one valid event"
        for event in normalized_events:
            contract_error = _bridge_event_contract_error(event)
            if contract_error:
                return None, contract_error
            resource_jid = str(event.get("resource_jid", "")).strip()
            if not resource_jid:
                return None, "bridge_events.events[].resource_jid is required"
            if allowed_resources and resource_jid not in allowed_resources:
                return None, (
                    f"bridge_events.events[].resource_jid '{resource_jid}' "
                    "is not an available bridge resource"
                )
            profile_contract_error = _bridge_profile_event_contract_error(
                event,
                bridge_resources=bridge_resources,
            )
            if profile_contract_error:
                return None, profile_contract_error
            event.pop("_operation_family_explicit", None)
        return {
            "type": "bridge_events",
            "events": deepcopy(normalized_events),
            "reason_summary": str(parsed.get("reason_summary", "") or "").strip(),
            "react_trace": _normalize_bridge_react_trace(parsed.get("react_trace")),
        }, None

    if response_type == "final_plan":
        plan = parsed.get("plan")
        if not isinstance(plan, dict):
            return None, "final_plan.plan must be an object"
        return {
            "type": "final_plan",
            "plan": deepcopy(plan),
            "reason_summary": str(parsed.get("reason_summary", "") or "").strip(),
            "react_trace": _normalize_bridge_react_trace(parsed.get("react_trace")),
        }, None

    return None, "turn response type must be 'observe', 'bridge_outline', 'bridge_events', or 'final_plan'"


async def llm_explore_states_and_events(
    stuck_state: dict,
    P_id: list[str],
    ra_jid: str,
    ask_llm: Callable[..., Coroutine[Any, Any, str]],
    goal_state: str,
    tools_catalog: list[dict],
    resource_infos: list[dict],
    part_tracker: dict | None = None,
    obligation_targets: list[dict] | None = None,
    operator_feedback: str = "",
    primitive_catalog: list[dict] | None = None,
    bridge_snapshot: dict[str, Any] | None = None,
    grounding_context: dict[str, Any] | None = None,
    bridge_resources: dict[str, Any] | None = None,
    debug_trace: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Ask the LLM for a recovery macro proposal when DES finds no modeled path.

    When primitive_catalog is provided, the bridge generates primitive-based
    macros that execute through execute_recovery_macro.  Otherwise, falls back
    to the legacy catalog-function-based macro shape.
    """
    from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.prompts.bridge_react import (
        build_state_exploration_prompt,
    )

    primitive_mode = bool(primitive_catalog or bridge_resources)
    debug_payload = debug_trace if isinstance(debug_trace, dict) else None
    warnings: list[str] = []

    if debug_payload is not None:
        debug_payload["primitive_mode"] = primitive_mode
        debug_payload["llm_inputs"] = {
            "stuck_state": deepcopy(stuck_state),
            "P_id": deepcopy(P_id),
            "ra_jid": str(ra_jid or "").strip(),
            "goal_state": str(goal_state or "").strip(),
            "part_tracker": deepcopy(part_tracker),
            "obligation_targets": deepcopy(obligation_targets),
            "operator_feedback": str(operator_feedback or "").strip(),
            "resource_infos": deepcopy(resource_infos),
            "tools_catalog": deepcopy(tools_catalog),
            "primitive_catalog": deepcopy(primitive_catalog),
            "bridge_snapshot": deepcopy(bridge_snapshot),
            "grounding_context": deepcopy(grounding_context),
            "bridge_resources": deepcopy(bridge_resources),
        }

    prompt = build_state_exploration_prompt(
        stuck_state=stuck_state,
        part_tracker=part_tracker,
        P_id=P_id,
        goal_state=goal_state,
        ra_jid=ra_jid,
        tools_catalog=tools_catalog,
        resource_infos=resource_infos,
        obligation_targets=obligation_targets,
        operator_feedback=operator_feedback,
        primitive_catalog=primitive_catalog,
        bridge_snapshot=bridge_snapshot,
        grounding_context=grounding_context,
        bridge_resources=bridge_resources,
    )
    if debug_payload is not None:
        debug_payload["prompt"] = prompt

    capture = _BridgeWarningCapture(warnings)
    logger.addHandler(capture)
    try:
        raw = await ask_llm(prompt=prompt, with_functions=False)
        if debug_payload is not None:
            debug_payload["raw_response"] = (
                raw
                if isinstance(raw, str)
                else json.dumps(raw, indent=2, default=str)
            )

        if primitive_mode:
            proposal = _normalize_primitive_bridge_proposal(
                raw=raw,
                ra_jid=ra_jid,
                primitive_catalog=primitive_catalog,
                bridge_snapshot=bridge_snapshot or {},
                grounding_context=grounding_context or {},
                bridge_resources=bridge_resources,
                obligation_targets=obligation_targets,
            )
        else:
            proposal = _normalize_bridge_proposal(
                raw=raw,
                ra_jid=ra_jid,
                tools_catalog=tools_catalog,
            )
    except Exception as exc:
        if debug_payload is not None:
            debug_payload["status"] = "exception"
            debug_payload["exception"] = repr(exc)
            debug_payload["warning_messages"] = list(warnings)
        raise
    finally:
        logger.removeHandler(capture)
        capture.close()

    if debug_payload is not None:
        debug_payload["warning_messages"] = list(warnings)
        debug_payload["normalized_proposal"] = (
            deepcopy(proposal) if isinstance(proposal, dict) else None
        )
        debug_payload["status"] = "accepted" if proposal else "rejected"

    if proposal:
        if proposal.get("macro_tasks"):
            macro_names = [
                str(task.get("macro_name", "")).strip()
                for task in (proposal.get("macro_tasks") or [])
                if isinstance(task, dict) and str(task.get("macro_name", "")).strip()
            ]
            total_steps = sum(
                len(task.get("primitive_steps") or [])
                for task in (proposal.get("macro_tasks") or [])
                if isinstance(task, dict)
            )
            name_key = ", ".join(macro_names[:2]) if macro_names else "bridge_macro_tasks"
            steps_len = total_steps
        else:
            name_key = proposal.get("macro_name") or proposal.get("function_name")
            steps_len = len(proposal.get("primitive_steps") or proposal.get("macro_steps") or [])
        logger.info(
            "[EnvironmentModel] LLM bridge proposed macro '%s' with %d step(s).",
            name_key,
            steps_len,
        )
        return proposal

    logger.warning("[EnvironmentModel] LLM bridge response was invalid or not compilable.")
    return None


def _normalize_primitive_bridge_proposal(
    *,
    raw: str,
    ra_jid: str,
    primitive_catalog: list[dict] | None,
    bridge_snapshot: dict[str, Any],
    grounding_context: dict[str, Any],
    bridge_resources: dict[str, Any] | None = None,
    obligation_targets: list[dict[str, Any]] | None = None,
) -> Optional[dict[str, Any]]:
    """Validate and normalize a primitive-based bridge proposal."""
    from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.primitive_semantics import (
        expected_snapshot_from_bridge_snapshot,
        resolve_param_refs,
        resolve_step_param_refs,
        validate_and_project_steps,
    )

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("[EnvironmentModel] LLM bridge response was not valid JSON.")
        return None

    if not isinstance(parsed, dict):
        return None

    resources = _bridge_resource_entries(
        ra_jid=ra_jid,
        primitive_catalog=primitive_catalog,
        bridge_snapshot=bridge_snapshot,
        bridge_resources=bridge_resources,
    )
    if not resources:
        logger.warning("[EnvironmentModel] Primitive bridge proposal had no available bridge resources.")
        return None

    primary_obligation, obligation_error = _normalize_primary_obligation(
        parsed.get("primary_obligation"),
        obligation_targets,
    )
    if obligation_error:
        logger.warning("[EnvironmentModel] Primitive bridge proposal rejected: %s", obligation_error)
        return None

    raw_macro_tasks = _macro_tasks_from_primitive_proposal(parsed)
    if not raw_macro_tasks:
        logger.warning("[EnvironmentModel] Bridge proposal has no macro_tasks/primitive_steps.")
        return None

    projected_resource_snapshots: dict[str, dict[str, Any]] = {
        resource_jid: deepcopy(entry.get("bridge_snapshot") or {})
        for resource_jid, entry in resources.items()
    }
    projected_parts: dict[str, Any] = deepcopy((grounding_context or {}).get("parts") or {})
    normalized_macro_tasks: list[dict[str, Any]] = []

    for index, raw_task in enumerate(raw_macro_tasks, start=1):
        if not isinstance(raw_task, dict):
            logger.warning("[EnvironmentModel] Bridge macro_task %d must be an object.", index)
            return None

        resource_jid = str(raw_task.get("resource_jid") or parsed.get("resource_jid") or "").strip() or ra_jid
        resource_entry = resources.get(resource_jid)
        if resource_entry is None:
            logger.warning(
                "[EnvironmentModel] Bridge macro_task %d targeted unknown resource '%s'.",
                index,
                resource_jid,
            )
            return None

        current_snapshot = deepcopy(projected_resource_snapshots.get(resource_jid) or resource_entry.get("bridge_snapshot") or {})
        primitive_steps = raw_task.get("primitive_steps") or raw_task.get("steps") or []
        if not isinstance(primitive_steps, list) or not primitive_steps:
            logger.warning("[EnvironmentModel] Bridge macro_task %d has no primitive_steps.", index)
            return None

        validated_steps: list[dict[str, Any]] = []
        execution_catalog = (
            resource_entry.get("execution_primitive_catalog")
            or resource_entry.get("primitive_catalog")
            or []
        )
        primitive_rows = {
            str(entry.get("name", "")).strip(): dict(entry)
            for entry in execution_catalog
            if isinstance(entry, dict) and str(entry.get("name", "")).strip()
        }
        for step in primitive_steps:
            if not isinstance(step, dict):
                logger.warning("[EnvironmentModel] Bridge macro_task %d contains a non-object step.", index)
                return None
            primitive = str(step.get("primitive", "")).strip()
            params = step.get("params") or {}
            if not isinstance(params, dict):
                logger.warning("[EnvironmentModel] Bridge macro_task %d step params must be an object.", index)
                return None
            normalized_step = {"primitive": primitive, "params": dict(params)}
            if str(step.get("store_as") or "").strip():
                logger.warning(
                    "[EnvironmentModel] Bridge macro_task %d step %d rejected legacy store_as on primitive '%s'.",
                    index,
                    len(validated_steps) + 1,
                    primitive,
                )
                return None
            validated_steps.append(normalized_step)

        dynamic_grounding_context = _projected_bridge_grounding_context(
            grounding_context=grounding_context or {},
            projected_resource_snapshots=projected_resource_snapshots,
            projected_parts=projected_parts,
            bridge_resources=resources,
            focused_resource_jid=resource_jid,
            primary_obligation=primary_obligation,
        )
        resolved_steps, resolution_error = resolve_step_param_refs(
            validated_steps,
            dynamic_grounding_context,
        )
        if resolution_error:
            logger.warning(
                "[EnvironmentModel] Primitive bridge proposal rejected during context_ref resolution: macro_task %d %s",
                index,
                resolution_error,
            )
            return None

        _em_profile = get_resource_profile(
            str(resource_entry.get("resource_type") or "resource").strip().lower() or "resource"
        )
        _snap_fields = resource_snapshot_fields_map(
            current_snapshot, _em_profile.snapshot_fields, profile=_em_profile,
        )
        logger.debug(
            "[EnvironmentModel] macro_task %d (%s) — %d resolved steps, initial snapshot: %s",
            index,
            str(raw_task.get("macro_name") or "").strip() or "unnamed",
            len(resolved_steps),
            ", ".join(f"{k}={v!r}" for k, v in _snap_fields.items()),
        )
        for _si, _step in enumerate(resolved_steps, start=1):
            logger.debug(
                "[EnvironmentModel]   step %d: %s params=%s",
                _si,
                _step.get("primitive"),
                {k: v for k, v in (_step.get("params") or {}).items() if k != "product_geometry"},
            )

        semantic_ok, projected_snapshot, semantic_error = validate_and_project_steps(
            resolved_steps,
            execution_catalog,
            current_snapshot,
            grounding_context=dynamic_grounding_context,
        )
        if not semantic_ok:
            _macro_label = str(raw_task.get("macro_name") or "").strip()
            logger.warning(
                "[EnvironmentModel] Primitive bridge proposal rejected: macro_task %d (%s on %s) %s "
                "(initial snapshot: %s)",
                index,
                _macro_label or "unnamed",
                resource_jid,
                semantic_error,
                ", ".join(f"{k}={v!r}" for k, v in _snap_fields.items()),
            )
            return None

        macro_name = str(raw_task.get("macro_name") or parsed.get("macro_name") or "").strip()
        if not macro_name:
            macro_name = (
                "bridge_recovery_macro"
                if len(raw_macro_tasks) == 1
                else f"bridge_recovery_macro_{index}"
            )

        task_metadata = _normalize_bridge_task_metadata(
            raw_task.get("task_metadata") or parsed.get("task_metadata") or {}
        )

        part_name = _normalize_optional_name(
            raw_task.get("part_name")
            or raw_task.get("touched_part")
            or parsed.get("part_name")
            or parsed.get("touched_part")
            or ""
        )

        raw_task_params = raw_task.get("task_params")
        if raw_task_params is None:
            raw_task_params = parsed.get("task_params") if len(raw_macro_tasks) == 1 else {}
        if raw_task_params is None:
            raw_task_params = {}
        if not isinstance(raw_task_params, dict):
            logger.warning("[EnvironmentModel] Bridge macro_task %d task_params must be an object.", index)
            return None
        reserved_task_params = sorted(
            str(key).strip()
            for key in raw_task_params
            if str(key).strip() in _BRIDGE_TASK_PARAM_RESERVED_KEYS
        )
        if reserved_task_params:
            logger.warning(
                "[EnvironmentModel] Bridge macro_task %d task_params used reserved keys: %s",
                index,
                reserved_task_params,
            )
            return None
        try:
            task_params = resolve_param_refs(raw_task_params, dynamic_grounding_context)
        except Exception as exc:
            logger.warning(
                "[EnvironmentModel] Primitive bridge proposal rejected during task_params resolution: macro_task %d %s",
                index,
                exc,
            )
            return None

        if not part_name:
            part_name = _normalize_optional_name(task_params.get("part_name") or "")

        filtered_required_context_keys: list[str] = []
        dropped_required_context_keys: list[str] = []
        for key in task_metadata.get("required_context_keys", []):
            if key in task_params:
                filtered_required_context_keys.append(key)
            else:
                dropped_required_context_keys.append(key)
        if dropped_required_context_keys:
            logger.warning(
                "[EnvironmentModel] Bridge macro_task %d ignored unsupported required_context_keys: %s",
                index,
                dropped_required_context_keys,
            )
        task_metadata["required_context_keys"] = filtered_required_context_keys

        missing_context_keys = [
            key for key in task_metadata.get("required_context_keys", []) if key not in task_params
        ]
        if missing_context_keys:
            logger.warning(
                "[EnvironmentModel] Bridge macro_task %d missing required task_params keys: %s",
                index,
                missing_context_keys,
            )
            return None

        if task_metadata.get("part_transition") and not part_name:
            logger.warning(
                "[EnvironmentModel] Bridge macro_task %d declared part_transition without part_name.",
                index,
            )
            return None

        for status_key, transition in (task_metadata.get("part_transition") or {}).items():
            if not isinstance(transition, dict):
                continue
            for param_key in ("location_param", "last_known_param"):
                required_param = str(transition.get(param_key) or "").strip()
                if required_param and required_param not in task_params:
                    logger.warning(
                        "[EnvironmentModel] Bridge macro_task %d missing task_params['%s'] required by part_transition[%s].",
                        index,
                        required_param,
                        status_key,
                    )
                    return None

        expected_start_state = str(raw_task.get("expected_start_state") or parsed.get("expected_start_state") or "").strip()
        if expected_start_state:
            actual_state = str(
                resource_snapshot_field_value(current_snapshot, "current_state") or ""
            ).strip()
            if actual_state and expected_start_state != actual_state:
                logger.warning(
                    "[EnvironmentModel] Bridge macro_task %d expected_start_state '%s' mismatched actual '%s'.",
                    index,
                    expected_start_state,
                    actual_state,
                )
                return None

        projected_snapshot_with_state = deepcopy(projected_snapshot)
        if task_metadata.get("out_state"):
            projected_snapshot_with_state = resource_snapshot_set_field(
                projected_snapshot_with_state,
                "current_state",
                str(task_metadata["out_state"]),
                profile=get_resource_profile(
                    str(resource_entry.get("resource_type") or "resource")
                ),
            )

        projected_resource_snapshots[resource_jid] = projected_snapshot_with_state
        _post_snap = resource_snapshot_fields_map(
            projected_snapshot_with_state, _em_profile.snapshot_fields, profile=_em_profile,
        )
        logger.debug(
            "[EnvironmentModel] macro_task %d (%s) — projected snapshot after validation: %s",
            index,
            str(raw_task.get("macro_name") or "").strip() or "unnamed",
            ", ".join(f"{k}={v!r}" for k, v in _post_snap.items()),
        )
        _apply_part_transition_projection(
            projected_parts,
            part_name=part_name,
            task_metadata=task_metadata,
            task_params=task_params,
            resource_jid=resource_jid,
        )
        projected_part_entry = (
            deepcopy(projected_parts.get(part_name) or {})
            if part_name
            else None
        )
        normalized_task = {
            "resource_jid": resource_jid,
            "macro_name": macro_name,
            "description": str(raw_task.get("description") or parsed.get("description") or "").strip(),
            "rationale": str(raw_task.get("rationale") or parsed.get("rationale") or "").strip(),
            "expected_start_state": expected_start_state,
            "expected_snapshot": expected_snapshot_from_bridge_snapshot(
                current_snapshot,
                resource_type=str(resource_entry.get("resource_type") or "resource"),
            ),
            "projected_snapshot": projected_snapshot_with_state,
            "projected_part_entry": projected_part_entry,
            "task_metadata": task_metadata,
            "part_name": part_name,
            "task_params": task_params,
            "primitive_steps": resolved_steps,
        }
        normalized_macro_tasks.append(normalized_task)

    summary: list[str] = []
    if primary_obligation:
        summary.append(f"rule:{primary_obligation.get('rule_id')}")
    for task in normalized_macro_tasks:
        summary.append(str(task.get("macro_name", "")).strip())
        summary.extend(
            str(step.get("primitive", "")).strip()
            for step in (task.get("primitive_steps") or [])
            if isinstance(step, dict) and str(step.get("primitive", "")).strip()
        )

    obligation_projection_error = _primary_obligation_projection_error(
        primary_obligation=primary_obligation,
        projected_resource_snapshots=projected_resource_snapshots,
    )
    if obligation_projection_error:
        logger.warning(
            "[EnvironmentModel] Primitive bridge proposal rejected: %s",
            obligation_projection_error,
        )
        return None

    bridge_event_summary = _normalize_bridge_event_summary(
        parsed.get("bridge_event_summary"),
        bridge_resources=bridge_resources,
    )

    proposal: dict[str, Any] = {
        "primary_obligation": primary_obligation,
        "bridge_event_summary": bridge_event_summary,
        "macro_tasks": normalized_macro_tasks,
        "projected_resource_snapshots": projected_resource_snapshots,
        "projected_parts": projected_parts,
        "summary": summary,
    }

    if len(normalized_macro_tasks) == 1:
        proposal.update(normalized_macro_tasks[0])

    return proposal


def _normalize_bridge_proposal(
    *,
    raw: str,
    ra_jid: str,
    tools_catalog: list[dict],
) -> Optional[dict[str, Any]]:
    """Legacy: validate and normalize a catalog-function-based bridge proposal."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("[EnvironmentModel] LLM bridge response was not valid JSON.")
        return None

    if isinstance(parsed, list):
        parsed = {
            "function_name": "bridge_recovery_macro",
            "resource_jid": ra_jid,
            "description": "LLM-generated recovery macro",
            "rationale": "",
            "macro_steps": parsed,
        }
    if not isinstance(parsed, dict):
        return None

    resource_token = str(ra_jid or "").split("@", 1)[0].strip().lower()
    resource_tools = [
        row
        for row in (tools_catalog or [])
        if isinstance(row, dict)
        and str(row.get("function_owner_agent", "")).strip().lower() == resource_token
    ]
    tools_by_name = {
        str(row.get("function", "")).strip(): row
        for row in resource_tools
        if str(row.get("function", "")).strip()
    }
    macro_steps = parsed.get("macro_steps") or parsed.get("steps") or []
    if not isinstance(macro_steps, list) or not macro_steps:
        return None

    compiled_macro: list[dict[str, Any]] = []
    for index, step in enumerate(macro_steps, start=1):
        if not isinstance(step, dict):
            return None
        function_name = str(step.get("function_name", "")).strip()
        if not function_name or function_name not in tools_by_name:
            logger.warning(
                "[EnvironmentModel] Bridge macro step %d used non-catalog function '%s'.",
                index,
                function_name,
            )
            return None
        params = step.get("params") or {}
        if not isinstance(params, dict):
            return None
        tool_row = tools_by_name[function_name]
        compiled_macro.append(
            {
                "resource_jid": ra_jid,
                "function_name": function_name,
                "params": dict(params),
                "description": str(tool_row.get("description", "")).strip(),
                "tool_signature": _tool_signature(tool_row),
                "in_state": str(tool_row.get("in_state", "")).strip(),
                "out_state": str(tool_row.get("out_state", "")).strip(),
            }
        )

    function_name = str(parsed.get("function_name", "")).strip()
    if not function_name:
        return None

    proposal_resource_jid = str(parsed.get("resource_jid") or ra_jid).strip() or ra_jid
    if proposal_resource_jid != ra_jid:
        logger.warning(
            "[EnvironmentModel] Bridge proposal targeted unexpected resource '%s' (expected '%s').",
            proposal_resource_jid,
            ra_jid,
        )
        return None

    return {
        "function_name": function_name,
        "resource_jid": proposal_resource_jid,
        "description": str(parsed.get("description", "")).strip(),
        "rationale": str(parsed.get("rationale", "")).strip(),
        "macro_steps": compiled_macro,
        "summary": [
            function_name,
            *[str(step.get("function_name", "")).strip() for step in compiled_macro],
        ],
    }

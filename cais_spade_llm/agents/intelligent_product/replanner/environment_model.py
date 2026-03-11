"""Environment model (M_e): compile resource bids and search for recovery paths."""

from __future__ import annotations

from copy import deepcopy
import json
import logging
from collections import deque
from typing import Any, Callable, Coroutine, Optional

from .resource_bidding import Bid, _tool_signature

logger = logging.getLogger(__name__)


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


def _state_key(state: dict) -> str:
    """Stable string key from a state dict."""
    return json.dumps(state, sort_keys=True)


def compile_environment_model(bids: list[Bid]) -> dict:
    """
    Fuse RA bids into environment model M_e (Algorithm 1, Kovalenko et al.).

    M_e = {
        "states":      {state_key: state_dict},
        "transitions": {state_key: {event_key: next_state_key}},
        "events":      {event_key: event_dict},   # includes ra_jid
    }
    """
    states: dict[str, dict] = {}
    transitions: dict[str, dict[str, str]] = {}
    events: dict[str, dict] = {}

    for bid in bids:
        if not bid.str_x or not bid.str_e:
            continue
        for i, event_dict in enumerate(bid.str_e):
            from_state = bid.str_x[i]
            to_state = bid.str_x[i + 1]

            from_key = _state_key(from_state)
            to_key = _state_key(to_state)

            states.setdefault(from_key, from_state)
            states.setdefault(to_key, to_state)

            fn_name = event_dict.get("function_name", "")
            event_key = f"{bid.ra_jid}::{fn_name}::{i}"

            events[event_key] = {**event_dict, "ra_jid": bid.ra_jid}
            transitions.setdefault(from_key, {})[event_key] = to_key

    return {
        "states": states,
        "transitions": transitions,
        "events": events,
    }


def plan_on_environment_model(
    M_e: dict,
    x_c: dict,
    P_id: list[str],
    goal_state: str,
) -> list[dict] | None:
    """
    BFS on M_e from x_c to a goal state where all P_id parts are at goal_state.
    Finds the path with fewest steps.

    Returns ordered list of event dicts (each includes ra_jid and params),
    or None if no path exists.
    """
    states = M_e["states"]
    transitions = M_e["transitions"]
    events = M_e["events"]

    start_key = _find_start_state(states, x_c)
    if start_key is None:
        logger.warning("[EnvironmentModel] Could not match x_c to any state in M_e.")
        return None

    def is_goal(state_key: str) -> bool:
        part_states = states[state_key].get("part_states", {})
        return all(part_states.get(p) == goal_state for p in P_id)

    # BFS: queue of (state_key, path_as_event_keys)
    queue: deque = deque([(start_key, [])])
    visited: set[str] = set([start_key])

    while queue:
        current_key, path = queue.popleft()

        if is_goal(current_key):
            return [events[ek] for ek in path]

        for event_key, next_key in transitions.get(current_key, {}).items():
            if next_key not in visited:
                visited.add(next_key)
                queue.append((next_key, path + [event_key]))

    return None


def _find_start_state(states: dict, x_c: dict) -> str | None:
    """Match x_c to a state key in M_e. Exact match first, then partial."""
    exact = _state_key(x_c)
    if exact in states:
        return exact

    # Partial: same resource_state and part_states
    for key, state in states.items():
        if (state.get("resource_state") == x_c.get("resource_state") and
                state.get("part_states") == x_c.get("part_states")):
            return key

    return None


def _bridge_resource_entries(
    *,
    ra_jid: str,
    primitive_catalog: list[dict] | None,
    bridge_snapshot: dict[str, Any] | None,
    bridge_resources: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    resources: dict[str, dict[str, Any]] = {}
    for raw_jid, raw_entry in (bridge_resources or {}).items():
        if not isinstance(raw_entry, dict):
            continue
        resource_jid = str(raw_jid or raw_entry.get("resource_jid") or "").strip()
        if not resource_jid:
            continue
        resources[resource_jid] = {
            "resource_jid": resource_jid,
            "primitive_catalog": list(raw_entry.get("primitive_catalog") or []),
            "bridge_snapshot": dict(
                raw_entry.get("bridge_snapshot")
                or raw_entry.get("primitive_snapshot")
                or {}
            ),
            "modeled_state": deepcopy(raw_entry.get("modeled_state") or {}),
            "pending_tasks": deepcopy(raw_entry.get("pending_tasks") or []),
            "static_capabilities": deepcopy(raw_entry.get("static_capabilities") or {}),
        }

    if not resources and primitive_catalog is not None:
        resource_jid = str(ra_jid or "").strip()
        if resource_jid:
            resources[resource_jid] = {
                "resource_jid": resource_jid,
                "primitive_catalog": list(primitive_catalog or []),
                "bridge_snapshot": dict(bridge_snapshot or {}),
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

    part_transition = task_metadata.get("part_transition") or {}
    if not isinstance(part_transition, dict):
        part_transition = {}
    task_metadata["part_transition"] = dict(part_transition)
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
    if "state" in transition:
        entry["state"] = transition["state"]
    if "location_template" in transition:
        entry["location"] = str(transition["location_template"]).format(resource_jid=resource_jid)
    if "location_param" in transition:
        entry["location"] = deepcopy(task_params.get(str(transition["location_param"])))
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
    context = deepcopy(grounding_context or {})
    context["parts"] = deepcopy(projected_parts)

    resources_payload: dict[str, dict[str, Any]] = {}
    for resource_jid, snapshot in (projected_resource_snapshots or {}).items():
        raw_entry = dict(bridge_resources.get(resource_jid) or {})
        bridge_snapshot = deepcopy(snapshot or {})
        resources_payload[resource_jid] = {
            "jid": resource_jid,
            "current_state": bridge_snapshot.get("current_state"),
            "held_part": bridge_snapshot.get("held_part"),
            "gripper_state": bridge_snapshot.get("gripper_state"),
            "current_pose": deepcopy(bridge_snapshot.get("current_pose")),
            "current_pose_ref": bridge_snapshot.get("current_pose_ref"),
            "named_poses": {
                str(pose_name): str(pose_name)
                for pose_name in (bridge_snapshot.get("named_poses") or [])
                if str(pose_name).strip()
            },
            "primitive_snapshot": bridge_snapshot,
            "modeled_state": deepcopy(raw_entry.get("modeled_state") or {}),
            "pending_tasks": deepcopy(raw_entry.get("pending_tasks") or []),
            "static_capabilities": deepcopy(raw_entry.get("static_capabilities") or {}),
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
) -> dict[str, Any] | None:
    """
    Ask the LLM for a recovery macro proposal when DES finds no modeled path.

    When primitive_catalog is provided, the bridge generates primitive-based
    macros that execute through execute_recovery_macro.  Otherwise, falls back
    to the legacy catalog-function-based macro shape.
    """
    from cais_spade_llm.prompts import build_state_exploration_prompt

    primitive_mode = bool(primitive_catalog or bridge_resources)

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

    raw = await ask_llm(prompt=prompt, with_functions=False)

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
    from cais_spade_llm.agents.intelligent_product.replanner.primitive_semantics import (
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
            store_as = str(step.get("store_as") or "").strip()
            if store_as:
                normalized_step["store_as"] = store_as
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

        semantic_ok, projected_snapshot, semantic_error = validate_and_project_steps(
            resolved_steps,
            resource_entry.get("primitive_catalog") or [],
            current_snapshot,
            grounding_context=dynamic_grounding_context,
        )
        if not semantic_ok:
            logger.warning(
                "[EnvironmentModel] Primitive bridge proposal rejected: macro_task %d %s",
                index,
                semantic_error,
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

        part_name = str(raw_task.get("part_name") or raw_task.get("touched_part") or parsed.get("part_name") or parsed.get("touched_part") or "").strip()

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
            part_name = str(task_params.get("part_name") or "").strip()

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
            actual_state = str((current_snapshot or {}).get("current_state", "")).strip()
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
            projected_snapshot_with_state["current_state"] = str(task_metadata["out_state"])

        projected_resource_snapshots[resource_jid] = projected_snapshot_with_state
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
            "expected_snapshot": expected_snapshot_from_bridge_snapshot(current_snapshot),
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

    proposal: dict[str, Any] = {
        "primary_obligation": primary_obligation,
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

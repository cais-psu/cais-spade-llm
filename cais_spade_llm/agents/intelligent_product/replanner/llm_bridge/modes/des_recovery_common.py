"""Shared runtime helpers for DES-style bridge recovery modes.

This module contains the grounding, solving, validation, and finalization
logic shared by hybrid and procedural DES modes. It intentionally does not
depend on single-shot or multi-turn session shapes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from hashlib import sha1
from copy import deepcopy
import json
import logging
from typing import Any

from cais_spade_llm.agents.central_controller.outline_macro_safety import (
    validate_outline_macro_cca_constraints,
)
from cais_spade_llm.agents.central_controller.base_safety_checker import (
    BaseSafetyChecker,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_artifacts import (
    write_bridge_artifacts,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_contract_semantics import (
    apply_event_contract_effects,
    event_location_ref,
    infer_event_semantics,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_des_solver import (
    compose_and_solve,
)

_logger = logging.getLogger(__name__)

_DEFAULT_MAX_TURNS = 6
_DEFAULT_SOLVER_MAX_EXPLORED_STATES = 50_000
_DEFAULT_VALIDATOR_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class RecoveryBudget:
    """Shared execution budget for DES recovery modes."""

    max_turns: int = _DEFAULT_MAX_TURNS
    solver_max_explored_states: int = _DEFAULT_SOLVER_MAX_EXPLORED_STATES
    validator_timeout_s: float = _DEFAULT_VALIDATOR_TIMEOUT_S


def recovery_budget_from_prepared_request(
    prepared_bridge_request: dict[str, Any],
) -> RecoveryBudget:
    """Read the shared DES recovery budget from prepared bridge request state."""
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    max_turns = int(bridge_session.get("max_turns") or _DEFAULT_MAX_TURNS)
    solver_max_explored_states = int(
        bridge_session.get("solver_max_explored_states")
        or bridge_session.get("solver_max_explored_states_cap")
        or _DEFAULT_SOLVER_MAX_EXPLORED_STATES
    )
    raw_timeout = bridge_session.get("validator_timeout_s")
    try:
        validator_timeout_s = float(
            raw_timeout if raw_timeout not in (None, "") else _DEFAULT_VALIDATOR_TIMEOUT_S
        )
    except (TypeError, ValueError):
        validator_timeout_s = _DEFAULT_VALIDATOR_TIMEOUT_S
    return RecoveryBudget(
        max_turns=max(1, max_turns),
        solver_max_explored_states=max(1, solver_max_explored_states),
        validator_timeout_s=max(0.0, validator_timeout_s),
    )


def build_des_session_seed(
    prepared_bridge_request: dict[str, Any],
    *,
    engine_name: str,
) -> dict[str, Any]:
    """Build a DES session seed shared by hybrid and procedural modes."""
    budget = recovery_budget_from_prepared_request(prepared_bridge_request)
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})

    symbolic_resources: dict[str, dict[str, Any]] = {}
    for row in observed_runtime_state.get("resources") or []:
        if not isinstance(row, dict):
            continue
        jid = str(row.get("resource_jid") or "").strip()
        if jid:
            symbolic_resources[jid] = deepcopy(row)

    symbolic_parts: dict[str, dict[str, Any]] = {}
    for row in llm_input.get("part_facts") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("part_name") or "").strip()
        if name:
            symbolic_parts[name] = deepcopy(row)

    return {
        "des_engine": engine_name,
        "current_phase": "evaluate_grounding",
        "turn_index": 0,
        "max_turns": int(budget.max_turns),
        "solver_max_explored_states": int(budget.solver_max_explored_states),
        "validator_timeout_s": float(budget.validator_timeout_s),
        "status": "pending",
        "turns": [],
        "domain_revision_count": 0,
        "current_plant": None,
        "plant_findings": [],
        "solver_result": None,
        "feasibility_findings": [],
        "action_sequence": [],
        "proposal": None,
        "symbolic_resources": symbolic_resources,
        "symbolic_parts": symbolic_parts,
        "revision_history": [],
        "persistent_constraint_summary": [],
        "last_rejected_plant": None,
        "last_solver_trace": [],
        "grounding_checkpoint": None,
        "grounding_observation_count": 0,
    }


def _truthy_flag(row: dict[str, Any], *field_names: str) -> bool:
    return any(bool(row.get(field_name)) for field_name in field_names)


def compute_observation_blockers(
    symbolic_parts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Identify parts whose localization is unknown or explicitly untrusted."""
    blockers: list[dict[str, Any]] = []
    for part_name, row in (symbolic_parts or {}).items():
        if not isinstance(row, dict):
            continue
        observed_pose = row.get("observed_pose")
        has_pose = isinstance(observed_pose, dict) and bool(observed_pose)
        current_location = str(row.get("current_location") or "").strip()
        last_known_location = str(row.get("last_known_location") or "").strip()
        location_basis = str(row.get("location_basis") or "").strip().lower()

        reason_parts: list[str] = []
        if _truthy_flag(
            row,
            "needs_observation",
            "requires_observation",
            "observation_required",
            "pose_untrusted",
            "location_unverified",
            "localization_unknown",
        ):
            reason_parts.append("explicit observation/localization flag is set")

        if location_basis in {"sensor_observation", "live_observation"} and not has_pose:
            reason_parts.append(
                f"location_basis='{location_basis}' but no observed_pose is stored"
            )

        if not current_location and not last_known_location and not has_pose:
            reason_parts.append("no current location, no last known location, and no observed pose")

        if reason_parts:
            blockers.append({
                "kind": "observation_required",
                "part_name": part_name,
                "reason": "; ".join(reason_parts),
                "description": (
                    f"OBSERVATION REQUIRED: part '{part_name}' has untrusted localization "
                    f"({'; '.join(reason_parts)}). Recovery must ground this part before "
                    "relying on its pose or live workspace occupancy."
                ),
            })
    return blockers


def compute_terminal_state_blockers(
    symbolic_resources: dict[str, dict[str, Any]],
    *,
    extra_terminal_state_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Resources currently blocked from participation in recovery."""
    extras = {str(s).strip().lower() for s in (extra_terminal_state_names or set())}
    blockers: list[dict[str, Any]] = []
    for jid, row in (symbolic_resources or {}).items():
        if not isinstance(row, dict):
            continue
        state = str(row.get("current_state") or "").strip().lower()
        reasons: list[str] = []
        if row.get("available") is False:
            reasons.append("available=False")
        for flag in ("is_blocked", "is_faulted", "is_error", "is_terminal"):
            if bool(row.get(flag)):
                reasons.append(f"{flag}=True")
        for fault_field in ("fault", "error", "error_code"):
            value = row.get(fault_field)
            if value not in (None, "", 0, False, [], {}):
                reasons.append(f"{fault_field}={value!r}")
        if state and state in extras:
            reasons.append(f"current_state='{state}' matches configured terminal-state name")
        if reasons:
            blockers.append({
                "kind": "resource_terminal_state",
                "resource_jid": jid,
                "current_state": state,
                "reason": "; ".join(reasons),
                "description": (
                    f"RESOURCE BLOCKED: '{jid}' shows terminal-state evidence "
                    f"({'; '.join(reasons)}). Recovery must clear or route around it."
                ),
            })
    return blockers


def _part_goal_reached(row: dict[str, Any]) -> bool:
    current_location = str(row.get("current_location") or "").strip()
    goal_location = str(row.get("goal_location") or "").strip()
    return bool(current_location) and bool(goal_location) and current_location == goal_location


def compute_assembly_order_blockers(
    llm_input: dict[str, Any],
    symbolic_parts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect unmet predecessor dependencies from explicit requirement structure."""
    del llm_input
    blockers: list[dict[str, Any]] = []
    for part_name, row in (symbolic_parts or {}).items():
        if not isinstance(row, dict) or _part_goal_reached(row):
            continue
        predecessors = row.get("predecessor_parts") or row.get("predecessors") or []
        if isinstance(predecessors, str):
            predecessors = [predecessors]
        for predecessor in predecessors if isinstance(predecessors, list) else []:
            predecessor_name = str(predecessor or "").strip()
            if not predecessor_name:
                continue
            predecessor_row = dict((symbolic_parts or {}).get(predecessor_name) or {})
            if predecessor_row and not _part_goal_reached(predecessor_row):
                blockers.append({
                    "kind": "assembly_order",
                    "part_name": part_name,
                    "predecessor": predecessor_name,
                    "reason": (
                        f"part '{part_name}' depends on predecessor '{predecessor_name}' "
                        "which has not yet reached its goal location"
                    ),
                    "description": (
                        f"ORDER BLOCKED: '{part_name}' cannot be considered resumable until "
                        f"predecessor '{predecessor_name}' reaches its goal location."
                    ),
                })
    return blockers


def _resource_workspace_bounds(resource_entry: dict[str, Any]) -> dict[str, Any] | None:
    entry = dict(resource_entry or {})
    caps = dict(entry.get("static_capabilities") or entry.get("capabilities") or {})
    bounds = caps.get("workspace_bounds") or entry.get("workspace_bounds")
    return dict(bounds) if isinstance(bounds, dict) else None


def _pose_in_bounds(pose: dict[str, Any], bounds: dict[str, Any]) -> bool:
    try:
        x = float(pose["x"])
        y = float(pose["y"])
        z = float(pose["z"])
    except (KeyError, TypeError, ValueError):
        return False
    for axis, value in (("x", x), ("y", y), ("z", z)):
        lo = bounds.get(f"{axis}_min_m")
        hi = bounds.get(f"{axis}_max_m")
        if lo is not None and value < float(lo):
            return False
        if hi is not None and value > float(hi):
            return False
    return True


def _reachable_resource_jids_for_pose(
    pose: dict[str, Any],
    bridge_resources: dict[str, dict[str, Any]],
) -> list[str]:
    """Return resources whose workspace bounds contain the supplied pose."""
    if not isinstance(pose, dict) or not pose:
        return []
    reachable: list[str] = []
    for jid, resource_entry in (bridge_resources or {}).items():
        bounds = _resource_workspace_bounds(dict(resource_entry or {}))
        if bounds and _pose_in_bounds(pose, bounds):
            reachable.append(str(jid))
    return sorted(jid for jid in reachable if jid)


def compute_shared_workspace_blockers(
    bridge_resources: dict[str, dict[str, Any]],
    symbolic_parts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect cross-resource workspace occupancy that can require transfer/resequencing."""
    blockers: list[dict[str, Any]] = []
    for part_name, row in (symbolic_parts or {}).items():
        if not isinstance(row, dict):
            continue
        assigned = str(
            row.get("assigned_resource_jid")
            or row.get("goal_resource_jid")
            or row.get("nominal_requirement_resource_jid")
            or row.get("resource_jid")
            or ""
        ).strip()
        pose = row.get("observed_pose")
        if not assigned or not isinstance(pose, dict):
            continue
        for jid, resource_entry in (bridge_resources or {}).items():
            if jid == assigned:
                continue
            bounds = _resource_workspace_bounds(dict(resource_entry or {}))
            if bounds and _pose_in_bounds(pose, bounds):
                blockers.append({
                    "kind": "shared_workspace",
                    "part_name": part_name,
                    "assigned_resource_jid": assigned,
                    "host_resource_jid": jid,
                    "reason": (
                        f"part '{part_name}' is assigned to '{assigned}' but currently lies in "
                        f"'{jid}' workspace"
                    ),
                    "description": (
                        f"WORKSPACE CONFLICT: '{part_name}' currently lies in '{jid}' workspace "
                        f"while assigned to '{assigned}'. Recovery may require part_transfer, "
                        "reassignment, or resequencing. Do not assign the observed-pose "
                        f"pick/grasp for '{part_name}' to '{assigned}' while that pose is "
                        f"outside '{assigned}' workspace; use '{jid}' for reachable recovery "
                        "or model a physically grounded part_transfer."
                    ),
                })
                break
    return blockers


def compute_reachability_blockers(
    symbolic_parts: dict[str, dict[str, Any]],
    bridge_resources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect parts whose observed pose is outside every known resource workspace."""
    blockers: list[dict[str, Any]] = []
    for part_name, row in (symbolic_parts or {}).items():
        if not isinstance(row, dict):
            continue
        pose = row.get("observed_pose")
        if not isinstance(pose, dict):
            continue
        reachable_jids: list[str] = []
        for jid, resource_entry in (bridge_resources or {}).items():
            bounds = _resource_workspace_bounds(dict(resource_entry or {}))
            if bounds and _pose_in_bounds(pose, bounds):
                reachable_jids.append(jid)
        if not reachable_jids:
            blockers.append({
                "kind": "reachability",
                "part_name": part_name,
                "observed_pose": deepcopy(pose),
                "reason": f"part '{part_name}' observed pose is outside every resource workspace",
                "description": (
                    f"UNREACHABLE: '{part_name}' currently has no grounded reachable resource "
                    "for its observed pose."
                ),
            })
    return blockers


def collect_recovery_blockers(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate generalized recovery blockers from the synchronized symbolic state."""
    symbolic_parts = dict(session_state.get("symbolic_parts") or {})
    symbolic_resources = dict(session_state.get("symbolic_resources") or {})
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})

    extra_terminal_state_names = bridge_session.get("extra_terminal_resource_state_names")
    if isinstance(extra_terminal_state_names, (list, tuple, set)):
        extras = {str(s).strip().lower() for s in extra_terminal_state_names if str(s).strip()}
    else:
        extras = set()

    blockers: list[dict[str, Any]] = []
    blockers.extend(compute_observation_blockers(symbolic_parts))
    blockers.extend(
        compute_terminal_state_blockers(
            symbolic_resources,
            extra_terminal_state_names=extras,
        )
    )
    blockers.extend(compute_assembly_order_blockers(llm_input, symbolic_parts))
    blockers.extend(compute_shared_workspace_blockers(bridge_resources, symbolic_parts))
    blockers.extend(compute_reachability_blockers(symbolic_parts, bridge_resources))
    return blockers


def _prompt_visible_part_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a prompt-facing part row with stale loose-part locations hidden."""
    sanitized = deepcopy(dict(row or {}))
    location_basis = str(sanitized.get("location_basis") or "").strip().lower()
    observed_pose = sanitized.get("observed_pose")
    holder = str(sanitized.get("current_holder_resource_jid") or "").strip()
    if (
        location_basis in {"sensor_observation", "live_observation"}
        and isinstance(observed_pose, dict)
        and observed_pose
        and not holder
    ):
        sanitized["current_location"] = None
    return sanitized


def build_recovery_gap_state(session_state: dict[str, Any]) -> dict[str, Any]:
    """Build resource/part prompt context from symbolic state."""
    return {
        "resource_state": [
            deepcopy(row)
            for row in (session_state.get("symbolic_resources") or {}).values()
        ],
        "part_state": [
            _prompt_visible_part_row(row)
            for row in (session_state.get("symbolic_parts") or {}).values()
        ],
    }


def extract_safety_dfas(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Extract loaded safety DFA rules from planner or llm_input context."""
    product_agent = getattr(planner, "product_agent", None)
    cca = None
    if product_agent is not None:
        cca = getattr(product_agent, "cca_agent", None) or getattr(product_agent, "_cca", None)
    for owner in (planner, product_agent, cca):
        if owner is None:
            continue
        for attr_name in (
            "plan_safety_validator",
            "safety_checker",
            "online_safety_monitor",
            "online_safety_supervisor",
            "safety_monitor",
            "safety",
        ):
            safety_checker = getattr(owner, attr_name, None)
            if safety_checker is None:
                continue
            dfas = getattr(safety_checker, "dfas", None)
            if isinstance(dfas, dict) and dfas:
                return deepcopy(dfas)

    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    rules = llm_input.get("loaded_safety_rules") or []
    dfas: dict[str, dict[str, Any]] = {}
    dfa_dots: dict[str, str] = {}
    dot_rules: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id") or rule.get("rule_id") or "").strip()
        dfa_data = rule.get("dfa")
        if rule_id and isinstance(dfa_data, dict):
            dfas[rule_id] = deepcopy(dfa_data)
            continue
        if rule_id and isinstance(dfa_data, str) and dfa_data.strip():
            dfa_dots[rule_id] = dfa_data
            dot_rules.append(rule)
            continue
        dfa_dot = str(rule.get("dfa_dot") or rule.get("dot") or "").strip()
        if rule_id and dfa_dot:
            dfa_dots[rule_id] = dfa_dot
            dot_rules.append(rule)
    if dfa_dots:
        try:
            parsed = BaseSafetyChecker(dfa_dots, dot_rules).dfas
            for rule_id, dfa in parsed.items():
                if rule_id and isinstance(dfa, dict):
                    dfas.setdefault(rule_id, deepcopy(dfa))
        except Exception as exc:
            _logger.warning("[DESCommon] Failed to parse safety DFA DOT sources: %s", exc)
    return dfas


def _parse_ap_full(full: Any) -> dict[str, str]:
    token = str(full or "").strip()
    segments = [segment.strip() for segment in token.split("/") if segment.strip()]
    if len(segments) < 5:
        return {}
    prefix = segments[0]
    if prefix not in {"ap", "ap_event"}:
        return {}
    return {
        "prefix": prefix,
        "process": segments[1] if len(segments) > 1 else "",
        "product": segments[2] if len(segments) > 2 else "",
        "resource": segments[3] if len(segments) > 3 else "",
        "function_name": segments[4] if len(segments) > 4 else "",
        "context": segments[5] if len(segments) > 5 else "",
    }


def _normalize_ap_descriptor(raw_ap: dict[str, Any], *, label: str | None = None) -> dict[str, Any] | None:
    if not isinstance(raw_ap, dict):
        return None
    entry = deepcopy(raw_ap)
    ap_label = str(label or entry.get("label") or entry.get("ap_label") or "").strip()
    if not ap_label:
        return None
    entry["label"] = ap_label

    parsed = _parse_ap_full(entry.get("full"))
    selector = dict(entry.get("selector") or {})
    for target_key, source_key in (
        ("resource", "resource"),
        ("product", "product"),
        ("part", "product"),
        ("function_name", "function_name"),
        ("function", "function_name"),
        ("context", "context"),
    ):
        if str(entry.get(target_key) or "").strip():
            continue
        value = str(parsed.get(source_key) or selector.get(target_key) or "").strip()
        if value:
            entry[target_key] = value
    if not str(entry.get("context") or "").strip():
        destination = str(selector.get("destination") or "").strip()
        if destination:
            entry["context"] = destination
    if not str(entry.get("full") or "").strip() and not any(
        str(entry.get(key) or "").strip()
        for key in ("resource", "product", "part", "function_name", "function")
    ):
        return None
    return entry


def extract_ap_descriptors(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract AP descriptor list from the safety-rule context."""
    del planner
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    rules = llm_input.get("loaded_safety_rules") or []
    descriptors: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        ap_defs = rule.get("bridge_aps") or rule.get("ap_definitions") or rule.get("aps") or []
        if isinstance(ap_defs, list):
            for raw_ap in ap_defs:
                normalized = _normalize_ap_descriptor(raw_ap)
                if normalized:
                    descriptors.append(normalized)
        elif isinstance(ap_defs, dict):
            for label, desc in ap_defs.items():
                if not isinstance(desc, dict):
                    continue
                normalized = _normalize_ap_descriptor(desc, label=str(label))
                if normalized:
                    descriptors.append(normalized)
    return descriptors


def apply_des_action_effects(
    action: dict[str, Any],
    *,
    resources: dict[str, dict[str, Any]],
    parts: dict[str, dict[str, Any]],
) -> None:
    """Apply projected symbolic effects of a DES recovery action."""
    apply_event_contract_effects(
        action,
        resources=resources,
        parts=parts,
    )


def _transition_task_kind(semantic_witness: dict[str, Any]) -> str:
    category = str(semantic_witness.get("category") or "").strip().lower()
    if category == "acquisition":
        return "part_acquire"
    if category == "release":
        return "part_release"
    if category == "transfer":
        return "part_transfer"
    if category == "carry":
        return "part_interaction"
    if category == "resource_only_transition":
        return "resource_transition"
    return "observation"


def _mapped_effect_value(field_name: str, value: Any) -> tuple[str, Any]:
    if field_name == "current_location":
        return "location", deepcopy(value)
    if field_name == "current_holder_resource_jid":
        return "holder", deepcopy(value)
    if field_name == "observed_pose":
        return "pose", deepcopy(value)
    if field_name == "part_state":
        return "state", deepcopy(value)
    return field_name, deepcopy(value)


def _expected_effect_from_witness(semantic_witness: dict[str, Any]) -> dict[str, Any]:
    expected_effect: dict[str, Any] = {}
    resource_effect: dict[str, Any] = {}
    part_effect: dict[str, Any] = {}

    for field_name, payload in dict(semantic_witness.get("resource_delta") or {}).items():
        mapped_name, mapped_value = _mapped_effect_value(field_name, payload.get("after"))
        resource_effect[mapped_name] = mapped_value
    for field_name, payload in dict(semantic_witness.get("part_delta") or {}).items():
        mapped_name, mapped_value = _mapped_effect_value(field_name, payload.get("after"))
        part_effect[mapped_name] = mapped_value

    if resource_effect:
        expected_effect["resource"] = resource_effect
    if part_effect:
        expected_effect["part"] = part_effect
    return expected_effect


def _preconditions_from_witness(
    action: dict[str, Any],
    *,
    semantic_witness: dict[str, Any],
) -> dict[str, Any]:
    category = str(semantic_witness.get("category") or "").strip().lower()
    resource_jid = str(action.get("resource_jid") or "").strip()
    location_ref = event_location_ref(action)
    pre_part = dict(semantic_witness.get("part_before") or {})
    preconditions: dict[str, Any] = {}

    if category == "acquisition":
        part_block: dict[str, Any] = {"requires_acquisition": True}
        before_holder = str(pre_part.get("current_holder_resource_jid") or "").strip()
        if before_holder:
            part_block["holder"] = before_holder
        else:
            part_block["holder"] = None
        preconditions["part"] = part_block
        source_ref: dict[str, Any] = {}
        source_location = location_ref or str(pre_part.get("current_location") or "").strip()
        if source_location:
            source_ref["location"] = source_location
        source_pose = action.get("pose")
        if not isinstance(source_pose, dict):
            source_pose = pre_part.get("observed_pose")
        if isinstance(source_pose, dict) and source_pose:
            source_ref["pose"] = deepcopy(source_pose)
        if source_ref:
            preconditions["source_ref"] = source_ref
        return preconditions

    if category == "transfer":
        preconditions["part"] = {
            "part_transfer": True,
            "source_holder": str(pre_part.get("current_holder_resource_jid") or "").strip() or None,
            "target_holder": resource_jid or None,
        }
        source_location = (
            location_ref
            or str(pre_part.get("current_location") or "").strip()
            or (
                f"{str(pre_part.get('current_holder_resource_jid') or '').strip()}_gripper"
                if str(pre_part.get("current_holder_resource_jid") or "").strip()
                else ""
            )
        )
        if source_location:
            preconditions["source_ref"] = {"location": source_location}
        return preconditions

    if category in {"release", "carry"}:
        preconditions["part"] = {"holder": resource_jid}
        return preconditions

    if category == "resource_only_transition":
        return {"resource": {}}

    return preconditions


def _task_target_from_action(
    action: dict[str, Any],
    *,
    semantic_witness: dict[str, Any],
) -> dict[str, Any]:
    target: dict[str, Any] = {}
    location_ref = event_location_ref(action)
    if location_ref:
        target["target_location"] = location_ref
    category = str(semantic_witness.get("category") or "").strip().lower()
    if category in {"acquisition", "transfer"}:
        source_location = (
            str(
                dict(_preconditions_from_witness(action, semantic_witness=semantic_witness).get("source_ref") or {}).get("location")
                or ""
            ).strip()
        )
        if source_location:
            target["source_location"] = source_location
        source_pose = action.get("pose")
        if isinstance(source_pose, dict) and source_pose:
            target["source_pose"] = deepcopy(source_pose)
    elif isinstance(action.get("pose"), dict):
        target["pose"] = deepcopy(action.get("pose"))
    return target


def _expected_end_state_from_witness(
    semantic_witness: dict[str, Any],
) -> dict[str, Any]:
    resource_after = dict(semantic_witness.get("resource_after") or {})
    part_after = dict(semantic_witness.get("part_after") or {})
    end_state: dict[str, Any] = {}
    if resource_after:
        current_state = str(resource_after.get("current_state") or "").strip()
        current_location = str(resource_after.get("current_location") or "").strip()
        if current_state:
            end_state["current_state"] = current_state
        if current_location:
            end_state["current_location"] = current_location
            end_state.setdefault("location", current_location)
    if part_after:
        part_location = str(part_after.get("current_location") or "").strip()
        holder = str(part_after.get("current_holder_resource_jid") or "").strip()
        if part_location:
            end_state["part_location"] = part_location
        if holder:
            end_state["holder"] = holder
    return end_state


def _apply_finding_provenance(
    findings: list[dict[str, Any]],
    *,
    action: dict[str, Any],
    task_id: str,
    step_index: int,
) -> list[dict[str, Any]]:
    event_name = str(action.get("event_name") or action.get("name") or "").strip()
    enriched: list[dict[str, Any]] = []
    for raw_finding in findings:
        if not isinstance(raw_finding, dict):
            continue
        finding = deepcopy(raw_finding)
        finding["task_id"] = str(finding.get("task_id") or task_id).strip() or task_id
        finding["step_index"] = int(step_index)
        if event_name:
            finding["event_name"] = event_name
        enriched.append(finding)
    return enriched


async def validate_action_feasibility(
    *,
    action: dict[str, Any],
    step_index: int,
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
    pre_resources: dict[str, dict[str, Any]] | None = None,
    pre_parts: dict[str, dict[str, Any]] | None = None,
    action_sequence: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Check physical feasibility of a single action via CCA and RA oracle."""
    findings: list[dict[str, Any]] = []
    resource_jid = str(action.get("resource_jid") or "").strip()
    task_id = f"RECOVERY_SEQ{step_index + 1}"
    if not resource_jid:
        return _apply_finding_provenance([{
            "constraint_owner": "hybrid_validator",
            "constraint_code": "missing_resource",
            "reason": f"Step {step_index + 1} has no resource_jid.",
        }], action=action, task_id=task_id, step_index=step_index)

    resolver = getattr(planner, "_resource_by_jid", None)
    resource_agent = resolver(resource_jid) if callable(resolver) else None

    part_name = str(action.get("part_name") or "").strip()
    pose = action.get("pose")
    sym_resources = pre_resources or dict(session_state.get("symbolic_resources") or {})
    sym_parts = pre_parts or dict(session_state.get("symbolic_parts") or {})
    projected_resources = deepcopy(sym_resources)
    projected_parts = deepcopy(sym_parts)
    apply_event_contract_effects(
        action,
        resources=projected_resources,
        parts=projected_parts,
    )
    semantic_witness = dict(action.get("semantic_witness") or {})
    if not semantic_witness:
        semantic_witness = infer_event_semantics(
            action,
            pre_resources=sym_resources,
            pre_parts=sym_parts,
            post_resources=projected_resources,
            post_parts=projected_parts,
        )
    task_kind = _transition_task_kind(semantic_witness)
    expected_effect = _expected_effect_from_witness(semantic_witness)
    preconditions = _preconditions_from_witness(action, semantic_witness=semantic_witness)
    task_target = _task_target_from_action(action, semantic_witness=semantic_witness)

    task: dict[str, Any] = {
        "outline_id": task_id,
        "resource_jid": resource_jid,
        "description": str(action.get("description") or "").strip(),
        "expected_end_state": _expected_end_state_from_witness(semantic_witness),
    }
    if part_name:
        task["part_name"] = part_name
    if task_target:
        task["action_target"] = deepcopy(task_target)
    if isinstance(pose, dict):
        task["pose"] = deepcopy(pose)

    grounded_action: dict[str, Any] = deepcopy(task)
    grounded_action["preconditions"] = deepcopy(preconditions)
    grounded_action["expected_effect"] = deepcopy(expected_effect)
    grounded_action["task_kind"] = task_kind
    grounded_action["effect_scope"] = str(semantic_witness.get("effect_scope") or "").strip()
    grounded_action["transition_witness"] = deepcopy(semantic_witness)
    if task_target:
        grounded_action["target"] = deepcopy(task_target)

    try:
        llm_input = dict(prepared_bridge_request.get("llm_input") or {})
        full_action_sequence = action_sequence or list(session_state.get("action_sequence") or [])
        outline_tasks: list[dict[str, Any]] = []
        task_types_by_id: dict[str, str] = {}
        task_index_by_id: dict[str, int] = {}
        dependency_map: dict[str, list[str]] = {}
        for idx, act in enumerate(full_action_sequence):
            act_id = f"RECOVERY_SEQ{idx + 1}"
            act_semantic_witness = dict(act.get("semantic_witness") or {})
            act_task: dict[str, Any] = {
                "outline_id": act_id,
                "resource_jid": str(act.get("resource_jid") or "").strip(),
                "description": str(act.get("description") or "").strip(),
            }
            act_part = str(act.get("part_name") or "").strip()
            if act_part:
                act_task["part_name"] = act_part
            act_target = _task_target_from_action(act, semantic_witness=act_semantic_witness)
            if act_target:
                act_task["action_target"] = deepcopy(act_target)
            act_task["expected_end_state"] = _expected_end_state_from_witness(act_semantic_witness)
            outline_tasks.append(act_task)
            task_types_by_id[act_id] = _transition_task_kind(act_semantic_witness)
            task_index_by_id[act_id] = idx
            dependency_map[act_id] = [f"RECOVERY_SEQ{idx}"] if idx > 0 else []

        signature: dict[str, Any] = {
            "task_kind": task_kind,
            "changes_part_world": bool(part_name and expected_effect.get("part")),
            "inferable_primary_part": part_name if part_name else None,
        }
        if task_kind in {"resource_transition", "observation"}:
            signature["changes_part_world"] = False

        cca_result = validate_outline_macro_cca_constraints(
            task=task,
            grounded_action=grounded_action,
            signature=signature,
            pre_resources=sym_resources,
            pre_parts=sym_parts,
            projected_resources=projected_resources,
            projected_parts=projected_parts,
            llm_input=llm_input,
            outline_tasks=outline_tasks,
            task_types_by_id=task_types_by_id,
            task_index_by_id=task_index_by_id,
            dependency_map=dependency_map,
        )
        findings.extend(list(cca_result.get("findings") or []))
    except Exception as exc:
        _logger.warning(
            "[DESCommon] CCA validation failed for step %d: %s",
            step_index, exc,
        )

    if task_kind == "part_transfer":
        receiver_held_part = str(sym_resources.get(resource_jid, {}).get("held_part") or "").strip()
        if receiver_held_part and receiver_held_part != part_name:
            findings.append({
                "task_id": task_id,
                "resource_jid": resource_jid,
                "part_name": part_name or None,
                "constraint_owner": "validator",
                "constraint_family": "resource_feasibility",
                "constraint_code": "holder_conflict",
                "reason": (
                    f"resource already holds '{receiver_held_part}' and cannot acquire transfer of "
                    f"'{part_name}'"
                ),
                "evidence": {"conflicting_part": receiver_held_part},
            })
        return _apply_finding_provenance(findings, action=action, task_id=task_id, step_index=step_index)

    if resource_agent is not None:
        oracle = getattr(resource_agent, "bridge_feasibility_oracle", None)
        if callable(oracle):
            try:
                resource_snapshot = deepcopy(dict(sym_resources.get(resource_jid) or {}))
                get_bridge_snapshot = getattr(resource_agent, "get_bridge_snapshot", None)
                if callable(get_bridge_snapshot):
                    try:
                        live_snapshot = get_bridge_snapshot()
                    except Exception:
                        live_snapshot = {}
                    if isinstance(live_snapshot, dict):
                        for field_name in (
                            "workspace_bounds",
                            "available_named_poses",
                            "bridge_adapter",
                            "resource_type",
                            "role",
                        ):
                            if field_name not in resource_snapshot and field_name in live_snapshot:
                                resource_snapshot[field_name] = deepcopy(live_snapshot.get(field_name))

                part_row = deepcopy(dict(sym_parts.get(part_name) or {}))
                part_context: dict[str, Any] = {
                    **part_row,
                    "target": deepcopy(task_target),
                }
                if isinstance(pose, dict):
                    part_context["observed_pose"] = deepcopy(pose)

                oracle_result = oracle(
                    operation_kind=str(semantic_witness.get("category") or task_kind),
                    part_name=part_name or None,
                    part_context=part_context,
                    bridge_snapshot=resource_snapshot,
                    grounded_action=deepcopy(grounded_action),
                )
                result = dict(oracle_result or {})
                if not bool(result.get("allowed", True)):
                    constraint_code = str(result.get("constraint_code") or "").strip() or "resource_unavailable"
                    reason = str(result.get("reason") or "").strip() or "resource feasibility oracle rejected the action"
                    evidence = deepcopy(result.get("evidence") or {})
                    if constraint_code == "workspace_unreachable":
                        grounded_action_row = dict(evidence.get("grounded_action") or {})
                        preconditions = dict(grounded_action_row.get("preconditions") or {})
                        source_ref = dict(preconditions.get("source_ref") or {})
                        source_location = str(source_ref.get("location") or "").strip().lower()
                        source_is_pose_anchor = (
                            source_location == "observed_pose"
                            or source_location.endswith("_observed_pose")
                        )
                        checked_pose = dict(evidence.get("checked_pose") or {})
                        if checked_pose:
                            reachable_resource_jids = _reachable_resource_jids_for_pose(
                                checked_pose,
                                dict(prepared_bridge_request.get("bridge_resources") or {}),
                            )
                            evidence["reachable_resource_jids"] = reachable_resource_jids
                            if source_is_pose_anchor:
                                host_resource_jids = [
                                    jid for jid in reachable_resource_jids
                                    if jid != resource_jid
                                ]
                                if host_resource_jids:
                                    evidence["reachable_host_resource_jids"] = host_resource_jids
                                    host_text = ", ".join(host_resource_jids)
                                    reason = (
                                        f"{reason}; observed pose is reachable by {host_text} "
                                        f"but not by '{resource_jid}'"
                                    )
                                elif not reachable_resource_jids:
                                    reason = (
                                        f"{reason}; no listed resource can reach this observed pose"
                                    )
                    finding: dict[str, Any] = {
                        "task_id": task_id,
                        "resource_jid": resource_jid,
                        "part_name": part_name or None,
                        "constraint_owner": "resource",
                        "constraint_family": "resource_feasibility",
                        "constraint_code": constraint_code,
                        "reason": reason,
                        "evidence": evidence,
                    }
                    guard = result.get("guard")
                    if isinstance(guard, dict) and guard:
                        finding["guard"] = deepcopy(guard)
                    findings.append(finding)
            except Exception as exc:
                _logger.warning(
                    "[DESCommon] RA oracle failed for step %d: %s",
                    step_index, exc,
                )
    return _apply_finding_provenance(findings, action=action, task_id=task_id, step_index=step_index)


def feasibility_findings_summary(findings: list[dict[str, Any]]) -> str:
    """Render feasibility findings for prompt feedback."""
    if not findings:
        return "(none)"
    lines: list[str] = []
    for finding in findings:
        code = str(finding.get("constraint_code") or "").strip()
        reason = str(finding.get("reason") or "").strip()
        event_name = str(finding.get("event_name") or "").strip()
        step_index = finding.get("step_index")
        provenance: list[str] = []
        if isinstance(step_index, int) and step_index >= 0:
            provenance.append(f"step {step_index + 1}")
        if event_name:
            provenance.append(f"event={event_name}")
        provenance_text = f" ({', '.join(provenance)})" if provenance else ""
        lines.append(f"- [{code}]{provenance_text} {reason}")
    return "\n".join(lines)


def append_revision_entry(
    session_state: dict[str, Any],
    *,
    plant: dict[str, Any] | None,
    plant_findings: list[dict[str, Any]] | None = None,
    solver_result: dict[str, Any] | None = None,
    feasibility_findings: list[dict[str, Any]] | None = None,
) -> None:
    """Accumulate ruled-out compositions across failed recovery attempts."""
    plant_payload = deepcopy(plant or {})
    serializable = deepcopy(plant_payload)
    if isinstance(serializable.get("states"), set):
        serializable["states"] = sorted(serializable["states"])
    if isinstance(serializable.get("marked"), set):
        serializable["marked"] = sorted(serializable["marked"])
    fingerprint = sha1(
        json.dumps(serializable, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    ).hexdigest()

    rejected_patterns: list[str] = []
    state_metadata = dict(plant_payload.get("state_metadata") or {})
    for state_name, metadata in state_metadata.items():
        if not isinstance(metadata, dict):
            continue
        bindings = dict(metadata.get("atomic_bindings") or {})
        if bindings:
            rejected_patterns.append(
                f"state={state_name}; atomic_bindings={bindings}"
            )
    for event_name, event in dict(plant_payload.get("events") or {}).items():
        event = dict(event or {})
        fact_parts = [f"event={event_name}"]
        resource_jid = str(event.get("resource_jid") or "").strip()
        part_name = str(event.get("part_name") or "").strip()
        location_ref = event_location_ref(event)
        if resource_jid:
            fact_parts.append(f"resource_jid={resource_jid}")
        if part_name:
            fact_parts.append(f"part_name={part_name}")
        if location_ref:
            fact_parts.append(f"location_ref={location_ref}")
        if len(fact_parts) > 1:
            rejected_patterns.append("; ".join(fact_parts))
    for finding in plant_findings or []:
        code = str(finding.get("constraint_code") or "").strip()
        reason = str(finding.get("reason") or "").strip()
        event_name = str(finding.get("event_name") or "").strip()
        state_name = str(dict(finding.get("evidence") or {}).get("state_name") or "").strip()
        detail = f"; event={event_name}" if event_name else ""
        if state_name:
            detail += f"; state={state_name}"
        rejected_patterns.append(f"plant_finding={code}: {reason}{detail}")
    if isinstance(solver_result, dict):
        for blocked in solver_result.get("blocked_transitions") or []:
            if not isinstance(blocked, dict):
                continue
            event_name = str(blocked.get("event") or "").strip()
            violated_rule = str(blocked.get("violated_rule") or "").strip()
            rejected_patterns.append(
                f"blocked_transition={event_name}; violated_rule={violated_rule}"
            )
    for finding in feasibility_findings or []:
        code = str(finding.get("constraint_code") or "").strip()
        reason = str(finding.get("reason") or "").strip()
        event_name = str(finding.get("event_name") or "").strip()
        part_name = str(finding.get("part_name") or "").strip()
        resource_jid = str(finding.get("resource_jid") or "").strip()
        step_index = finding.get("step_index")
        step_text = (
            f"; step_index={int(step_index)}"
            if isinstance(step_index, int) and step_index >= 0
            else ""
        )
        event_text = f"; event={event_name}" if event_name else ""
        rejected_patterns.append(f"feasibility_finding={code}: {reason}{step_text}{event_text}")
        evidence = dict(finding.get("evidence") or {})
        guard = dict(finding.get("guard") or {})
        grounded_action = dict(evidence.get("grounded_action") or {})
        preconditions = dict(grounded_action.get("preconditions") or {})
        source_ref = dict(preconditions.get("source_ref") or {})
        source_location = str(dict(source_ref).get("location") or "").strip().lower()
        reachable_hosts = [
            str(jid).strip()
            for jid in (
                evidence.get("reachable_host_resource_jids")
                or evidence.get("reachable_resource_jids")
                or []
            )
            if str(jid).strip() and str(jid).strip() != resource_jid
        ]
        if (
            code == "workspace_unreachable"
            and (guard.get("kind") == "observed_pose_unreachable" or source_location.endswith("_observed_pose") or source_location == "observed_pose")
            and event_name
            and part_name
            and resource_jid
        ):
            pattern_parts = [
                f"event={event_name}",
                f"part_name={part_name}",
                f"unreachable_resource_jid={resource_jid}",
            ]
            if reachable_hosts:
                pattern_parts.append(
                    f"reachable_host_resource_jids={','.join(sorted(set(reachable_hosts)))}"
                )
            rejected_patterns.append(
                "workspace_pick_rejection=" + "; ".join(pattern_parts)
            )

    persistent = list(session_state.get("persistent_constraint_summary") or [])
    seen = {str(item) for item in persistent}
    for pattern in rejected_patterns:
        if pattern not in seen:
            persistent.append(pattern)
            seen.add(pattern)

    history = list(session_state.get("revision_history") or [])
    history.append({
        "turn_index": int(session_state.get("turn_index") or 0),
        "plant_fingerprint": fingerprint,
        "plant_findings": deepcopy(plant_findings or []),
        "solver_status": str((solver_result or {}).get("status") or "").strip(),
        "solver_trace": deepcopy((solver_result or {}).get("trace") or []),
        "feasibility_findings": deepcopy(feasibility_findings or []),
        "rejected_composition_patterns": rejected_patterns,
    })

    session_state["persistent_constraint_summary"] = persistent
    session_state["revision_history"] = history
    session_state["last_rejected_plant"] = plant_payload or None
    session_state["last_solver_trace"] = deepcopy((solver_result or {}).get("trace") or [])


def revision_history_summary_text(session_state: dict[str, Any]) -> str:
    """Compact summary of accumulated revision history."""
    history = list(session_state.get("revision_history") or [])
    if not history:
        return "(none)"
    lines: list[str] = []
    for entry in history[-5:]:
        turn_index = int(entry.get("turn_index") or 0)
        solver_status = str(entry.get("solver_status") or "").strip() or "n/a"
        pattern_count = len(entry.get("rejected_composition_patterns") or [])
        lines.append(
            f"- turn {turn_index}: solver={solver_status}, ruled_out_patterns={pattern_count}"
        )
    return "\n".join(lines)


def _json_block(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str, ensure_ascii=True)


def _grounding_safety_rule_lines(llm_input: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for row in llm_input.get("loaded_safety_rules") or []:
        if not isinstance(row, dict):
            continue
        rule_id = str(row.get("rule_id") or row.get("id") or "").strip() or "rule"
        summary = str(
            row.get("summary")
            or row.get("generated_interpretation")
            or row.get("raw_text")
            or ""
        ).strip()
        lines.append(f"- {rule_id}: {summary}" if summary else f"- {rule_id}")
    return lines or ["(none)"]


def _grounding_requirement_lines(llm_input: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for row in llm_input.get("relevant_assembly_requirements") or []:
        if not isinstance(row, dict):
            continue
        requirement_id = str(row.get("requirement_id") or "").strip() or "requirement"
        status = str(row.get("status") or "unknown").strip()
        summary = str(row.get("summary") or "").strip()
        lines.append(f"- {requirement_id} [{status}]: {summary}" if summary else f"- {requirement_id} [{status}]")
    return lines or ["(none)"]


def _grounding_fact_summary(
    symbolic_resources: dict[str, dict[str, Any]],
    symbolic_parts: dict[str, dict[str, Any]],
) -> list[str]:
    facts: list[str] = []
    for jid, row in sorted((symbolic_resources or {}).items()):
        if not isinstance(row, dict):
            continue
        state = str(row.get("current_state") or "unknown").strip()
        held_part = row.get("held_part")
        location = row.get("current_location")
        facts.append(
            f"Resource {jid}: state={state}, location={location or '-'}, held_part={held_part or '-'}."
        )
    for part_name, row in sorted((symbolic_parts or {}).items()):
        if not isinstance(row, dict):
            continue
        location_basis = str(row.get("location_basis") or "").strip().lower()
        holder = row.get("current_holder_resource_jid")
        location = row.get("current_location")
        pose = row.get("observed_pose")
        if location_basis in {"sensor_observation", "live_observation"} and pose and not holder:
            location = None
        facts.append(
            f"Part {part_name}: location={location or '-'}, observed_pose={pose or '-'}, holder={holder or '-'}."
        )
    return facts


def _prompt_grounding_part_facts(
    symbolic_parts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        _prompt_visible_part_row(row)
        for row in (symbolic_parts or {}).values()
        if isinstance(row, dict)
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        current_state = str(row.get("current_state") or "").strip().lower()
        if current_state in {"", "unknown"}:
            row.pop("current_state", None)
    return rows


def _continuation_condition_summary(row: dict[str, Any]) -> str:
    blocking_reason = str(row.get("blocking_reason") or "").strip()
    if blocking_reason:
        return blocking_reason
    entity = str(row.get("entity") or row.get("entity_kind") or "continuation").strip()
    field = str(row.get("field") or "condition").strip()
    expected = row.get("expected")
    actual = row.get("actual")
    if expected not in (None, "") or actual not in (None, ""):
        return f"{entity} {field}: expected {expected!r}, actual {actual!r}"
    kind = str(row.get("kind") or "unmet_continuation_condition").strip()
    return f"{kind}: {entity}"


def build_evaluate_grounding_artifacts(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    decision: str,
    blockers: list[dict[str, Any]],
    observation_tasks: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Build deterministic grounding prompt/response artifacts for DES modes."""
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    symbolic_resources = dict(session_state.get("symbolic_resources") or {})
    symbolic_parts = dict(session_state.get("symbolic_parts") or {})
    observation_blockers = [
        blocker for blocker in blockers
        if isinstance(blocker, dict) and blocker.get("kind") == "observation_required"
    ]
    observe_requests = [
        {
            "fact_type": "part_pose",
            "entity": str(blocker.get("part_name") or "").strip(),
            "reason": str(blocker.get("reason") or blocker.get("description") or "").strip(),
        }
        for blocker in observation_blockers
        if str(blocker.get("part_name") or "").strip()
    ]
    artifact_decision = "observe" if decision == "grounding_required" else "grounded"

    prompt_sections = [
        "Task and Role",
        (
            "Deterministic grounding assessment for a DES fallback recovery session.\n"
            "Current phase: Observation / State Estimation.\n"
            "Decision source: runtime-derived symbolic state; no LLM call was made."
        ),
        "",
        "Fault Event",
        _json_block(llm_input.get("fault_event") or {}),
        "",
        "Current Resource Facts",
        _json_block(list(symbolic_resources.values())),
        "",
        "Current Part Facts",
        _json_block(_prompt_grounding_part_facts(symbolic_parts)),
        "",
        "Modeled Continuation Gap",
        _json_block(llm_input.get("modeled_continuation_gap") or {}),
        "",
        "World Observation Surface",
        _json_block(observation_tasks),
        "",
        "Session Observation Store",
        _json_block(session_state.get("observation_store") or {}),
        "",
        "Safety Rules",
        "\n".join(_grounding_safety_rule_lines(llm_input)),
        "",
        "Assembly Requirements",
        "\n".join(_grounding_requirement_lines(llm_input)),
        "",
        "Open Recovery Conditions",
        _json_block(blockers),
        "",
        "Decision Result",
        _json_block({
            "decision": artifact_decision,
            "observe_request_count": len(observe_requests),
            "dispatch_observation_task_count": len(observation_tasks),
        }),
    ]
    prompt_text = "\n".join(prompt_sections).strip() + "\n"

    blocking_reasons = [
        str(blocker.get("reason") or blocker.get("description") or "").strip()
        for blocker in observation_blockers
        if str(blocker.get("reason") or blocker.get("description") or "").strip()
    ]
    modeled_gap = dict(llm_input.get("modeled_continuation_gap") or {})
    continuation_reasons = [
        _continuation_condition_summary(row)
        for row in (modeled_gap.get("unmet_continuation_conditions") or [])
        if isinstance(row, dict)
    ]
    response: dict[str, Any] = {
        "thought": (
            "Grounding is required before hybrid DES domain generation because at least one "
            "recovery-critical part has untrusted or missing localization."
            if artifact_decision == "observe"
            else "The current symbolic resource and part facts are grounded enough for hybrid DES domain generation."
        ),
        "decision": artifact_decision,
        "blocking_reasons": blocking_reasons + continuation_reasons,
        "grounded_facts": _grounding_fact_summary(symbolic_resources, symbolic_parts),
        "recovery_implications": [
            str(blocker.get("description") or blocker.get("reason") or "").strip()
            for blocker in blockers
            if isinstance(blocker, dict)
            and str(blocker.get("description") or blocker.get("reason") or "").strip()
        ] + continuation_reasons,
        "observe_reason": "; ".join(blocking_reasons) if artifact_decision == "observe" else "",
        "observe_requests": observe_requests if artifact_decision == "observe" else [],
        "dispatch_observation_tasks": deepcopy(observation_tasks),
    }
    return prompt_text, response


async def handle_evaluate_grounding(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    engine_name: str,
    session_state_key: str,
) -> tuple[str, dict[str, Any]]:
    """Pause DES recovery when fresh observation is required."""
    all_blockers = collect_recovery_blockers(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    observation_blockers = [
        blocker for blocker in all_blockers if blocker.get("kind") == "observation_required"
    ]
    if not observation_blockers:
        prompt_text, response = build_evaluate_grounding_artifacts(
            session_state=session_state,
            prepared_bridge_request=prepared_bridge_request,
            decision="grounding_satisfied",
            blockers=all_blockers,
            observation_tasks=[],
        )
        return "grounding_satisfied", {
            "current_recovery_blockers": all_blockers,
            "prompt_text": prompt_text,
            "raw_response": response,
        }

    part_names: list[str] = []
    for blocker in observation_blockers:
        part_name = str(blocker.get("part_name") or "").strip()
        if part_name and part_name not in part_names:
            part_names.append(part_name)

    observation_tasks: list[dict[str, Any]] = []
    for part_name in part_names:
        observation_tasks.append({
            "id": f"OBS_{part_name}",
            "resource_jid": str(prepared_bridge_request.get("ra_jid") or ""),
            "function_name": "detect_parts",
            "params": {"part_name": part_name},
            "store_as": f"detected_{part_name.lower()}",
            "background": False,
        })

    session_state["status"] = "paused_after_grounding"
    prepared_bridge_request[session_state_key] = deepcopy(session_state)
    prepared_bridge_request[f"{engine_name}_pending_observation_tasks"] = deepcopy(observation_tasks)
    prepared_bridge_request.setdefault("bridge_debug", {})["status"] = "paused_after_grounding"
    prompt_text, response = build_evaluate_grounding_artifacts(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        decision="grounding_required",
        blockers=all_blockers,
        observation_tasks=observation_tasks,
    )
    return "grounding_required", {
        "dispatch_observation_tasks": observation_tasks,
        "current_recovery_blockers": all_blockers,
        "prompt_text": prompt_text,
        "raw_response": response,
    }


async def handle_compose_and_solve(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Run the DES solver against the current plant and shared budget."""
    plant = dict(session_state.get("current_plant") or {})
    safety_dfas = extract_safety_dfas(planner, prepared_bridge_request)
    ap_descriptors = extract_ap_descriptors(planner, prepared_bridge_request)
    max_states = int(
        session_state.get("solver_max_explored_states")
        or recovery_budget_from_prepared_request(prepared_bridge_request).solver_max_explored_states
    )
    solver_result = compose_and_solve(
        plant=plant,
        safety_dfas=safety_dfas,
        ap_descriptors=ap_descriptors,
        max_states=max_states,
    )
    session_state["solver_result"] = deepcopy(solver_result)
    if solver_result.get("status") == "solved":
        session_state["action_sequence"] = deepcopy(solver_result.get("action_sequence") or [])
        return "solved", {
            "solver_status": solver_result["status"],
            "product_states_explored": solver_result.get("product_states_explored", 0),
            "trace_length": len(solver_result.get("trace") or []),
        }
    session_state["domain_revision_count"] = int(session_state.get("domain_revision_count") or 0) + 1
    return str(solver_result.get("status") or "unsolvable"), {
        "solver_status": solver_result.get("status"),
        "product_states_explored": solver_result.get("product_states_explored", 0),
        "trace_length": len(solver_result.get("trace") or []),
    }


async def handle_validate_plan(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    planner: Any,
) -> tuple[str, dict[str, Any]]:
    """Validate the solved action sequence against rolling symbolic state."""
    action_sequence = list(session_state.get("action_sequence") or [])
    all_findings: list[dict[str, Any]] = []

    async def _validate_all() -> tuple[
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        projected_resources = deepcopy(dict(session_state.get("symbolic_resources") or {}))
        projected_parts = deepcopy(dict(session_state.get("symbolic_parts") or {}))
        collected: list[dict[str, Any]] = []
        for index, action in enumerate(action_sequence):
            step_findings = await validate_action_feasibility(
                action=action,
                step_index=index,
                planner=planner,
                prepared_bridge_request=prepared_bridge_request,
                session_state=session_state,
                pre_resources=projected_resources,
                pre_parts=projected_parts,
                action_sequence=action_sequence,
            )
            collected.extend(step_findings)
            if not step_findings:
                apply_des_action_effects(
                    action,
                    resources=projected_resources,
                    parts=projected_parts,
                )
        return collected, projected_resources, projected_parts

    timeout_s = float(session_state.get("validator_timeout_s") or 0.0)
    try:
        if timeout_s > 0:
            all_findings, projected_resources, projected_parts = await asyncio.wait_for(
                _validate_all(),
                timeout=timeout_s,
            )
        else:
            all_findings, projected_resources, projected_parts = await _validate_all()
    except asyncio.TimeoutError:
        all_findings = [{
            "constraint_owner": "validator",
            "constraint_code": "validator_timeout",
            "reason": f"validation exceeded timeout_s={timeout_s}",
        }]
        projected_resources = deepcopy(dict(session_state.get("symbolic_resources") or {}))
        projected_parts = deepcopy(dict(session_state.get("symbolic_parts") or {}))

    session_state["feasibility_findings"] = deepcopy(all_findings)
    if not all_findings:
        session_state["symbolic_resources"] = deepcopy(projected_resources)
        session_state["symbolic_parts"] = deepcopy(projected_parts)
        return "all_feasible", {
            "feasibility_finding_count": 0,
            "feasibility_findings": [],
        }
    session_state["domain_revision_count"] = int(session_state.get("domain_revision_count") or 0) + 1
    return "infeasible", {
        "feasibility_finding_count": len(all_findings),
        "feasibility_findings": deepcopy(all_findings),
    }


def _declared_composite_states(plant: dict[str, Any]) -> list[str]:
    state_metadata = dict(plant.get("state_metadata") or {})
    declared: list[str] = []
    for state_name, metadata in state_metadata.items():
        if not isinstance(metadata, dict):
            continue
        if metadata.get("atomic_bindings"):
            declared.append(str(state_name))
    return sorted(set(declared))


def _declared_marking_predicates(plant: dict[str, Any]) -> dict[str, str]:
    marked_state_metadata = dict(plant.get("marked_state_metadata") or {})
    declared: dict[str, str] = {}
    for state_name, metadata in marked_state_metadata.items():
        if not isinstance(metadata, dict):
            continue
        predicate = str(metadata.get("marking_predicate") or "").strip()
        if predicate:
            declared[str(state_name)] = predicate
    return declared


def _cleared_blockers_summary(
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
) -> list[str]:
    blockers = collect_recovery_blockers(
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
    )
    return sorted({str(blocker.get("kind") or "") for blocker in blockers if blocker.get("kind")})


async def handle_finalize(
    *,
    session_state: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
    engine_name: str,
) -> tuple[str, dict[str, Any]]:
    """Build the final proposal from the accepted recovery trace."""
    action_sequence = list(session_state.get("action_sequence") or [])
    solver_result = dict(session_state.get("solver_result") or {})
    current_plant = dict(session_state.get("current_plant") or {})

    outline_tasks: list[dict[str, Any]] = []
    for index, action in enumerate(action_sequence):
        task: dict[str, Any] = {
            "outline_id": f"RECOVERY_SEQ{index + 1}",
            "resource_jid": str(action.get("resource_jid") or "").strip(),
            "description": str(action.get("description") or "").strip(),
        }
        event_name = str(action.get("event_name") or "").strip()
        part_name = str(action.get("part_name") or "").strip()
        location_ref = event_location_ref(action)
        if event_name:
            task["event_name"] = event_name
        if part_name:
            task["part_name"] = part_name
        if location_ref:
            task["location_ref"] = location_ref
        pose = action.get("pose")
        if isinstance(pose, dict):
            task["pose"] = deepcopy(pose)
        outline_tasks.append(task)

    proposal: dict[str, Any] = {
        "outline_tasks": outline_tasks,
        "solver_status": "solved",
        "engine": engine_name,
        "domain_revisions": int(session_state.get("domain_revision_count") or 0),
        "validation_status": "all_feasible",
        "accepted_trace_length": len(solver_result.get("trace") or []),
        "accepted_trace": deepcopy(solver_result.get("trace") or []),
        "accepted_action_sequence": deepcopy(action_sequence),
        "accepted_primitive_program": deepcopy(
            session_state.get("accepted_primitive_program") or []
        ),
        "cleared_blockers_summary": _cleared_blockers_summary(prepared_bridge_request, session_state),
        "resume_conditions_met": not bool(session_state.get("feasibility_findings")),
        "grounding_observation_count": int(session_state.get("grounding_observation_count") or 0),
        "solver_explored_state_count": int(solver_result.get("product_states_explored") or 0),
        "feasibility_finding_count": len(session_state.get("feasibility_findings") or []),
        "revision_history_summary": revision_history_summary_text(session_state),
        "declared_composite_states": _declared_composite_states(current_plant),
        "declared_marking_predicates": _declared_marking_predicates(current_plant),
    }
    session_state["proposal"] = deepcopy(proposal)
    return "accepted", {"proposal_task_count": len(outline_tasks)}


def write_des_per_turn_artifact(
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
    *,
    phase_label: str,
    debug_session_key: str,
    write_session_transcript: bool = False,
) -> None:
    """Write per-turn artifacts for DES recovery modes."""
    try:
        payload = deepcopy(prepared_bridge_request)
        payload["bridge_debug"] = payload.get("bridge_debug") or {}
        payload["bridge_debug"][debug_session_key] = deepcopy(session_state)
        write_bridge_artifacts(
            payload,
            phase_label=phase_label,
            write_session_transcript=write_session_transcript,
        )
    except Exception:
        _logger.debug("[DESCommon] Failed to write per-turn artifact.", exc_info=True)


__all__ = [
    "RecoveryBudget",
    "append_revision_entry",
    "apply_des_action_effects",
    "build_evaluate_grounding_artifacts",
    "build_des_session_seed",
    "build_recovery_gap_state",
    "collect_recovery_blockers",
    "compute_observation_blockers",
    "extract_ap_descriptors",
    "extract_safety_dfas",
    "feasibility_findings_summary",
    "handle_compose_and_solve",
    "handle_evaluate_grounding",
    "handle_finalize",
    "handle_validate_plan",
    "recovery_budget_from_prepared_request",
    "revision_history_summary_text",
    "validate_action_feasibility",
    "write_des_per_turn_artifact",
]

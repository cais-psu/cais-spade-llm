"""Bridge-only primitive semantics helpers.

This module turns controller primitive docstrings into a private in-memory
catalogue that includes preconditions/effects, then uses the same semantics for
prompt grounding, proposal validation, and runtime state projection.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
from typing import Any

from function_analyzer import FunctionAnalyzer
from resources.robot import UR5eController, XArm6Controller


_SUPPORTED_PRECONDITION_OPS = frozenset({"equals", "not_equals", "exists"})
_SUPPORTED_EFFECT_OPS = frozenset(
    {
        "set",
        "set_from_param",
        "pose_absolute_from_params",
        "pose_relative_from_params",
        "set_unknown",
    }
)


def _controller_owner(robot_agent: Any) -> Any | None:
    controller = getattr(robot_agent, "_controller", None)
    if controller is not None:
        return controller

    scope_name = ""
    try:
        scope_name = str(robot_agent._robot_scope_name()).strip().lower()
    except Exception:
        scope_name = str(getattr(robot_agent, "agent_name", "")).split("@", 1)[0].lower()

    if scope_name.startswith("ur5e"):
        return UR5eController
    if scope_name.startswith("xarm6"):
        return XArm6Controller
    return None


def _required_params_from_signature(fn: Any) -> list[str]:
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        if name in {"self", "cls"}:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect._empty:
            required.append(name)
    return required


def _param_schema(fn: Any, analyzed: dict[str, Any]) -> dict[str, Any]:
    schema = analyzed.get("parameters") or {}
    properties = dict(schema.get("properties") or {})
    return {
        "type": "object",
        "properties": properties,
        "required": _required_params_from_signature(fn),
    }


def _normalize_semantics_map(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for state_key, rule in payload.items():
        if not isinstance(rule, dict):
            continue
        normalized = {k: v for k, v in rule.items() if k in (_SUPPORTED_PRECONDITION_OPS | _SUPPORTED_EFFECT_OPS)}
        if normalized:
            out[str(state_key)] = normalized
    return out


def _effect_phrase(field: str, rule: dict[str, Any]) -> str:
    if "set" in rule:
        return f"{field} becomes {rule['set']!r}"
    if "set_from_param" in rule:
        return f"{field} becomes parameter '{rule['set_from_param']}'"
    if "pose_absolute_from_params" in rule:
        coords = ", ".join(map(str, rule["pose_absolute_from_params"]))
        return f"{field} becomes pose({coords})"
    if "pose_relative_from_params" in rule:
        coords = ", ".join(map(str, rule["pose_relative_from_params"]))
        return f"{field} shifts by ({coords})"
    if rule.get("set_unknown"):
        return f"{field} becomes unknown"
    return ""


def _precondition_phrase(field: str, rule: dict[str, Any]) -> str:
    if "equals" in rule:
        return f"{field} must equal {rule['equals']!r}"
    if "not_equals" in rule:
        return f"{field} must not equal {rule['not_equals']!r}"
    if "exists" in rule:
        return f"{field} must {'exist' if rule['exists'] else 'not exist'}"
    return ""


def _semantic_summary(preconditions: dict[str, dict[str, Any]], effects: dict[str, dict[str, Any]]) -> str:
    clauses: list[str] = []
    for field, rule in preconditions.items():
        phrase = _precondition_phrase(field, rule)
        if phrase:
            clauses.append(phrase)
    for field, rule in effects.items():
        phrase = _effect_phrase(field, rule)
        if phrase:
            clauses.append(phrase)
    return "; ".join(clauses)


def build_primitive_catalog(robot_agent: Any) -> list[dict[str, Any]]:
    """Build the private primitive catalogue for one robot resource."""
    controller_owner = _controller_owner(robot_agent)
    if controller_owner is None:
        return []

    analyzer = FunctionAnalyzer()
    primitive_names = sorted(getattr(robot_agent, "_BRIDGE_PRIMITIVES", []) or [])
    rows: list[dict[str, Any]] = []

    for primitive_name in primitive_names:
        fn = getattr(controller_owner, primitive_name, None)
        if not callable(fn):
            continue

        analyzed = analyzer.analyze_function(fn)
        meta = FunctionAnalyzer._extract_yaml_frontmatter(fn)
        preconditions = _normalize_semantics_map(meta.get("preconditions"))
        effects = _normalize_semantics_map(meta.get("effects"))
        params_schema = _param_schema(fn, analyzed)
        params_summary: dict[str, dict[str, Any]] = {}
        for param_name, schema in (params_schema.get("properties") or {}).items():
            params_summary[param_name] = {
                "type": schema.get("type", "string"),
                "description": schema.get("description", ""),
            }

        rows.append(
            {
                "name": primitive_name,
                "description": analyzed.get("description", ""),
                "params": params_summary,
                "parameters": params_schema,
                "required_params": list(params_schema.get("required") or []),
                "preconditions": preconditions,
                "effects": effects,
                "semantic_summary": _semantic_summary(preconditions, effects),
            }
        )

    return rows


def get_robot_bridge_snapshot(robot_agent: Any) -> dict[str, Any]:
    """Return the current primitive-level bridge snapshot for one robot."""
    current_pose = None
    controller = getattr(robot_agent, "_controller", None)
    if (
        controller is not None
        and str(getattr(robot_agent, "execution_mode", "")).strip().lower() != "dry_run"
    ):
        try:
            pose_result = controller.get_current_pose()
        except Exception:
            pose_result = None
        if isinstance(pose_result, dict) and pose_result.get("success"):
            pose = dict(pose_result.get("pose") or {})
            if {"x", "y", "z"} <= set(pose.keys()):
                current_pose = {
                    "x": float(pose["x"]),
                    "y": float(pose["y"]),
                    "z": float(pose["z"]),
                }

    if current_pose is None and getattr(robot_agent, "_bridge_pose_ref", None) is None:
        position = getattr(robot_agent, "_position", None)
        if isinstance(position, dict) and {"x", "y", "z"} <= set(position.keys()):
            current_pose = {
                "x": float(position["x"]),
                "y": float(position["y"]),
                "z": float(position["z"]),
            }

    return {
        "current_state": str(getattr(robot_agent, "_current_state", "") or "").strip() or "idle",
        "held_part": getattr(robot_agent, "_held_part", None),
        "gripper_state": str(getattr(robot_agent, "_gripper_state", "") or "").strip() or "unknown",
        "current_pose": current_pose,
        "current_pose_ref": getattr(robot_agent, "_bridge_pose_ref", None),
        "named_poses": sorted((getattr(robot_agent, "named_positions", {}) or {}).keys()),
    }


def _validate_step_params(params: Any, primitive_meta: dict[str, Any]) -> str | None:
    if not isinstance(params, dict):
        return "params must be an object"

    schema = primitive_meta.get("parameters") or {}
    properties = dict(schema.get("properties") or {})
    required = list(schema.get("required") or [])

    for required_name in required:
        if required_name not in params:
            return f"missing required param '{required_name}'"

    for key in params:
        if key not in properties:
            return f"unknown param '{key}'"

    return None


def _check_preconditions(snapshot: dict[str, Any], primitive_meta: dict[str, Any]) -> str | None:
    preconditions = primitive_meta.get("preconditions") or {}
    for field, rule in preconditions.items():
        value = snapshot.get(field)
        if "equals" in rule and value != rule["equals"]:
            return f"precondition failed: {field} must equal {rule['equals']!r}"
        if "not_equals" in rule and value == rule["not_equals"]:
            return f"precondition failed: {field} must not equal {rule['not_equals']!r}"
        if "exists" in rule:
            exists = value is not None
            if exists != bool(rule["exists"]):
                return f"precondition failed: {field} existence mismatch"
    return None


def apply_effects_to_snapshot(
    step: dict[str, Any], primitive_meta: dict[str, Any], snapshot: dict[str, Any]
) -> dict[str, Any]:
    """Apply one primitive's semantic effects to a snapshot."""
    next_snapshot = deepcopy(snapshot)
    params = dict(step.get("params") or {})
    effects = primitive_meta.get("effects") or {}

    for field, rule in effects.items():
        if "set" in rule:
            next_snapshot[field] = deepcopy(rule["set"])
            continue
        if "set_from_param" in rule:
            next_snapshot[field] = params.get(str(rule["set_from_param"]))
            continue
        if "pose_absolute_from_params" in rule:
            x_key, y_key, z_key = list(rule["pose_absolute_from_params"])
            next_snapshot[field] = {
                "x": float(params[x_key]),
                "y": float(params[y_key]),
                "z": float(params[z_key]),
            }
            continue
        if "pose_relative_from_params" in rule:
            dx_key, dy_key, dz_key = list(rule["pose_relative_from_params"])
            current_pose = dict(next_snapshot.get(field) or {})
            next_snapshot[field] = {
                "x": float(current_pose["x"]) + float(params[dx_key]),
                "y": float(current_pose["y"]) + float(params[dy_key]),
                "z": float(current_pose["z"]) + float(params[dz_key]),
            }
            continue
        if rule.get("set_unknown"):
            next_snapshot[field] = None

    return next_snapshot


def validate_and_project_steps(
    steps: list[dict[str, Any]],
    primitive_catalog: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> tuple[bool, dict[str, Any], str | None]:
    """Validate a primitive sequence and project the resulting snapshot."""
    projected = deepcopy(snapshot or {})
    catalog_by_name = {
        str(entry.get("name", "")).strip(): entry
        for entry in (primitive_catalog or [])
        if isinstance(entry, dict) and str(entry.get("name", "")).strip()
    }

    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            return False, projected, f"step {index} must be an object"

        primitive = str(step.get("primitive", "")).strip()
        primitive_meta = catalog_by_name.get(primitive)
        if primitive_meta is None:
            return False, projected, f"step {index} used unknown primitive '{primitive}'"

        params_error = _validate_step_params(step.get("params") or {}, primitive_meta)
        if params_error:
            return False, projected, f"step {index} {primitive}: {params_error}"

        precondition_error = _check_preconditions(projected, primitive_meta)
        if precondition_error:
            return False, projected, f"step {index} {primitive}: {precondition_error}"

        try:
            projected = apply_effects_to_snapshot(step, primitive_meta, projected)
        except Exception as exc:
            return False, projected, f"step {index} {primitive}: failed to apply effects ({exc})"

    return True, projected, None


def expected_snapshot_from_bridge_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields that are stable enough for start-state validation."""
    return {
        "current_state": snapshot.get("current_state"),
        "held_part": snapshot.get("held_part"),
        "gripper_state": snapshot.get("gripper_state"),
    }


def snapshot_matches_expected(actual: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, str | None]:
    """Return whether actual snapshot satisfies expected snapshot fields."""
    for key in ("current_state", "held_part", "gripper_state"):
        if key not in expected:
            continue
        if actual.get(key) != expected.get(key):
            return (
                False,
                f"expected {key}={expected.get(key)!r} but found {actual.get(key)!r}",
            )
    return True, None


def sync_agent_from_bridge_snapshot(robot_agent: Any, snapshot: dict[str, Any]) -> None:
    """Apply bridge snapshot fields back onto RobotAgent runtime state."""
    if "current_state" in snapshot and snapshot.get("current_state") is not None:
        robot_agent._current_state = str(snapshot["current_state"])
    if "held_part" in snapshot:
        robot_agent._held_part = snapshot.get("held_part")
    if "gripper_state" in snapshot and snapshot.get("gripper_state") is not None:
        robot_agent._gripper_state = str(snapshot["gripper_state"])

    current_pose = snapshot.get("current_pose")
    if isinstance(current_pose, dict) and {"x", "y", "z"} <= set(current_pose.keys()):
        robot_agent._position = {
            "x": float(current_pose["x"]),
            "y": float(current_pose["y"]),
            "z": float(current_pose["z"]),
        }

    robot_agent._bridge_pose_ref = snapshot.get("current_pose_ref")

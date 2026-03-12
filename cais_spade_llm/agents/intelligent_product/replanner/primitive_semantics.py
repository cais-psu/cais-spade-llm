"""Bridge-only primitive semantics helpers.

This module turns controller primitive docstrings into a private in-memory
catalogue that includes preconditions/effects, then uses the same semantics for
prompt grounding, proposal validation, and runtime state projection.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
from typing import Any

from cais_spade_llm.function_analyzer import FunctionAnalyzer
from cais_spade_llm.resources.robot import UR5eController, XArm6Controller


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


def _is_scalar_json_value(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _context_ref_tokens(context_ref: str) -> list[str]:
    ref = str(context_ref or "").strip()
    if not ref:
        raise ValueError("context_ref is empty")

    if ref.startswith("/"):
        return [
            token.replace("~1", "/").replace("~0", "~")
            for token in ref.lstrip("/").split("/")
            if token != ""
        ]

    tokens = [token for token in ref.split(".") if token]
    if not tokens:
        raise ValueError(f"context_ref '{ref}' is invalid")
    return tokens


def _is_step_output_ref(context_ref: str) -> bool:
    tokens = _context_ref_tokens(context_ref)
    return bool(tokens) and tokens[0] == "step_outputs"


def _walk_context_tokens(root: Any, tokens: list[str], *, context_ref: str) -> Any:
    current = root
    for token in tokens:
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(f"context_ref '{context_ref}' could not resolve token '{token}'")
            current = current[token]
            continue
        if isinstance(current, list):
            try:
                index = int(token)
            except (TypeError, ValueError) as exc:
                raise KeyError(
                    f"context_ref '{context_ref}' expected list index at token '{token}'"
                ) from exc
            if index < 0 or index >= len(current):
                raise KeyError(
                    f"context_ref '{context_ref}' list index '{token}' is out of range"
                )
            current = current[index]
            continue
        raise KeyError(
            f"context_ref '{context_ref}' cannot descend into non-container value at token '{token}'"
        )
    return current


def resolve_context_ref(
    context_ref: str,
    grounding_context: dict[str, Any],
    *,
    step_outputs: dict[str, Any] | None = None,
) -> Any:
    """Resolve one context_ref against planner context and optional step outputs.

    Supports JSON Pointer (`/parts/SG/observed_pose/x`) and dot paths
    (`parts.SG.observed_pose.x`) for compatibility with the current bridge TODO.
    """
    ref = str(context_ref or "").strip()
    tokens = _context_ref_tokens(ref)
    root = dict(grounding_context or {})
    if step_outputs is not None:
        root["step_outputs"] = deepcopy(step_outputs)
    return deepcopy(_walk_context_tokens(root, tokens, context_ref=ref))


def resolve_param_refs(
    value: Any,
    grounding_context: dict[str, Any],
    *,
    step_outputs: dict[str, Any] | None = None,
    preserve_step_output_refs: bool = False,
) -> Any:
    """Recursively resolve context_ref wrappers inside a primitive params value."""
    if isinstance(value, dict):
        if set(value.keys()) == {"context_ref"}:
            context_ref = str(value.get("context_ref") or "")
            if preserve_step_output_refs and _is_step_output_ref(context_ref):
                return deepcopy(value)
            resolved = resolve_context_ref(
                context_ref,
                grounding_context,
                step_outputs=step_outputs,
            )
            if not _is_scalar_json_value(resolved):
                raise ValueError(
                    f"context_ref '{value.get('context_ref')}' resolved to a non-scalar value"
                )
            return resolved
        return {
            str(key): resolve_param_refs(
                subvalue,
                grounding_context,
                step_outputs=step_outputs,
                preserve_step_output_refs=preserve_step_output_refs,
            )
            for key, subvalue in value.items()
        }

    if isinstance(value, list):
        return [
            resolve_param_refs(
                item,
                grounding_context,
                step_outputs=step_outputs,
                preserve_step_output_refs=preserve_step_output_refs,
            )
            for item in value
        ]

    return deepcopy(value)


def resolve_step_param_refs(
    steps: list[dict[str, Any]],
    grounding_context: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    """Resolve planner-known context_ref values while preserving step_outputs refs."""
    resolved_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        try:
            normalized = dict(step)
            normalized["params"] = resolve_param_refs(
                step.get("params") or {},
                grounding_context,
                preserve_step_output_refs=True,
            )
            resolved_steps.append(normalized)
        except Exception as exc:
            primitive = str(step.get("primitive", "")).strip()
            return [], f"step {index} {primitive}: {exc}"
    return resolved_steps, None


def _validate_store_as(store_as: Any) -> str | None:
    alias = str(store_as or "").strip()
    if not alias:
        return None
    if not alias.replace("_", "").isalnum() or alias[0].isdigit():
        return "store_as must be a snake_case-like identifier"
    return None


def _normalized_xyz_pose(payload: Any) -> dict[str, float] | None:
    if not isinstance(payload, dict) or not {"x", "y", "z"} <= set(payload.keys()):
        return None
    try:
        return {
            "x": float(payload["x"]),
            "y": float(payload["y"]),
            "z": float(payload["z"]),
        }
    except (TypeError, ValueError):
        return None


def _normalized_orientation(payload: Any) -> dict[str, float] | None:
    if not isinstance(payload, dict) or not {"qx", "qy", "qz", "qw"} <= set(payload.keys()):
        return None
    try:
        return {
            "qx": float(payload["qx"]),
            "qy": float(payload["qy"]),
            "qz": float(payload["qz"]),
            "qw": float(payload["qw"]),
        }
    except (TypeError, ValueError):
        return None


def _normalize_detected_part_output(part: Any, *, fallback_part_name: str = "") -> dict[str, Any] | None:
    if not isinstance(part, dict):
        return None
    try:
        x = float(part["x"])
        y = float(part["y"])
        z = float(part["z"])
    except (KeyError, TypeError, ValueError):
        return None
    part_name = str(part.get("part_name") or fallback_part_name or "").strip()
    output: dict[str, Any] = {
        "part_name": part_name,
        "x": x,
        "y": y,
        "z": z,
        "pose": {"x": x, "y": y, "z": z},
    }
    orientation = _normalized_orientation(part)
    if orientation is not None:
        output.update(orientation)
        output["orientation"] = deepcopy(orientation)
        output["pose"].update(orientation)
    model_name = str(part.get("model_name") or "").strip()
    if model_name:
        output["model_name"] = model_name
    return output


def _normalize_pose_output(pose: Any) -> dict[str, Any] | None:
    pose_xyz = _normalized_xyz_pose(pose)
    if pose_xyz is None:
        return None
    out: dict[str, Any] = {
        "x": pose_xyz["x"],
        "y": pose_xyz["y"],
        "z": pose_xyz["z"],
        "pose": dict(pose_xyz),
    }
    orientation = _normalized_orientation(pose)
    if orientation is not None:
        out.update(orientation)
        out["orientation"] = deepcopy(orientation)
        out["pose"].update(orientation)
    return out


def _preview_detect_parts_output(
    params: dict[str, Any],
    grounding_context: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    part_name = str(params.get("part_name") or "").strip()
    if not part_name:
        return None, "detect_parts with store_as requires params.part_name in v1"

    part_info = (grounding_context or {}).get("parts", {}).get(part_name, {})
    observed_pose = _normalized_xyz_pose((part_info or {}).get("observed_pose"))
    if observed_pose is None:
        observed_pose = {"x": 0.0, "y": 0.0, "z": 0.0}

    output: dict[str, Any] = {
        "part_name": part_name,
        "x": observed_pose["x"],
        "y": observed_pose["y"],
        "z": observed_pose["z"],
        "pose": dict(observed_pose),
    }
    orientation = _normalized_orientation((part_info or {}).get("observed_pose"))
    if orientation is not None:
        output.update(orientation)
        output["orientation"] = deepcopy(orientation)
        output["pose"].update(orientation)
    target = (part_info or {}).get("target") or {}
    model_name = str(target.get("model_name") or "").strip()
    if model_name:
        output["model_name"] = model_name
    return output, None


def _preview_get_current_pose_output(
    snapshot: dict[str, Any],
    grounding_context: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    raw_pose = (
        ((grounding_context or {}).get("resource") or {}).get("current_pose")
        or snapshot.get("current_pose")
        or {}
    )
    pose = _normalize_pose_output(raw_pose)
    if pose is None:
        pose = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "orientation": {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
            "pose": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        }
    elif "qx" not in pose:
        orientation = {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
        pose.update(orientation)
        pose["orientation"] = deepcopy(orientation)
        pose["pose"].update(orientation)
    return pose, None


def preview_step_output(
    *,
    primitive: str,
    params: dict[str, Any],
    snapshot: dict[str, Any],
    grounding_context: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Return a validation-time preview output for supported observational primitives."""
    if primitive == "detect_parts":
        return _preview_detect_parts_output(params, grounding_context)
    if primitive == "get_current_pose":
        return _preview_get_current_pose_output(snapshot, grounding_context)
    return None, f"primitive '{primitive}' does not support store_as in v1"


def extract_step_output(
    *,
    primitive: str,
    params: dict[str, Any],
    step_result: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Extract one normalized runtime step output for supported observational primitives."""
    if primitive == "detect_parts":
        part_name = str(params.get("part_name") or "").strip()
        if not part_name:
            return None, "detect_parts with store_as requires params.part_name in v1"
        items = step_result.get("data")
        if not isinstance(items, list):
            return None, "detect_parts store_as expected list result data"
        if len(items) != 1:
            return None, (
                f"detect_parts store_as expected exactly one result for '{part_name}', "
                f"found {len(items)}"
            )
        output = _normalize_detected_part_output(items[0], fallback_part_name=part_name)
        if output is None:
            return None, "detect_parts store_as result did not contain x/y/z fields"
        return output, None

    if primitive == "get_current_pose":
        output = _normalize_pose_output(step_result.get("pose"))
        if output is None:
            return None, "get_current_pose store_as result did not contain pose x/y/z"
        return output, None

    return None, f"primitive '{primitive}' does not support store_as in v1"


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
    *,
    grounding_context: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any], str | None]:
    """Validate a primitive sequence and project the resulting snapshot."""
    projected = deepcopy(snapshot or {})
    step_outputs: dict[str, Any] = {}
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

        store_as = str(step.get("store_as") or "").strip()
        store_as_error = _validate_store_as(store_as)
        if store_as_error:
            return False, projected, f"step {index} {primitive}: {store_as_error}"
        if store_as and store_as in step_outputs:
            return False, projected, f"step {index} {primitive}: duplicate store_as '{store_as}'"

        try:
            resolved_params = resolve_param_refs(
                step.get("params") or {},
                grounding_context or {},
                step_outputs=step_outputs,
            )
        except Exception as exc:
            return False, projected, f"step {index} {primitive}: {exc}"

        params_error = _validate_step_params(resolved_params, primitive_meta)
        if params_error:
            return False, projected, f"step {index} {primitive}: {params_error}"

        precondition_error = _check_preconditions(projected, primitive_meta)
        if precondition_error:
            return False, projected, f"step {index} {primitive}: {precondition_error}"

        preview_step = {**dict(step), "params": resolved_params}
        try:
            projected = apply_effects_to_snapshot(preview_step, primitive_meta, projected)
        except Exception as exc:
            return False, projected, f"step {index} {primitive}: failed to apply effects ({exc})"

        if store_as:
            preview_output, preview_error = preview_step_output(
                primitive=primitive,
                params=resolved_params,
                snapshot=projected,
                grounding_context=grounding_context or {},
            )
            if preview_error:
                return False, projected, f"step {index} {primitive}: {preview_error}"
            step_outputs[store_as] = preview_output

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

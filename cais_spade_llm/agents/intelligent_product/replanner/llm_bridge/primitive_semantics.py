"""Bridge-only primitive semantics helpers.

This module turns controller primitive docstrings into a private in-memory
catalogue that includes preconditions/effects, then uses the same semantics for
prompt grounding, proposal validation, and runtime state projection.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from cais_spade_llm.resources.resource_profile import (
    get_resource_profile,
    get_resource_profile_for_agent,
    resource_snapshot_field_value,
    resource_snapshot_fields_map,
    resource_snapshot_has_field,
    resource_snapshot_set_field,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_adapters import (
    bridge_adapter_capabilities,
    bridge_resource_type,
    canonical_bridge_resource,
)
from cais_spade_llm.function_analyzer import FunctionAnalyzer


_SUPPORTED_PRECONDITION_OPS = frozenset({"equals", "not_equals", "exists"})
_SUPPORTED_EFFECT_OPS = frozenset(
    {
        "set",
        "set_from_param",
        "set_from_param_any_of",
        "pose_absolute_from_params",
        "pose_relative_from_params",
        "set_unknown",
    }
)


def _is_scalar_json_value(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _canonicalize_step_output_alias(alias: Any) -> str:
    raw = str(alias or "").strip()
    if not raw:
        return ""
    raw = raw.replace("-", "_").replace(" ", "_")
    raw = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", raw)
    raw = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", raw)
    raw = re.sub(r"_+", "_", raw)
    return raw.strip("_").lower()


def _context_ref_display(context_ref: Any) -> str:
    if isinstance(context_ref, (list, tuple)):
        return repr(list(context_ref))
    if context_ref is None:
        return ""
    return str(context_ref).strip()


def _context_ref_tokens(context_ref: Any) -> list[str]:
    if isinstance(context_ref, (list, tuple)):
        tokens = [str(token).strip() for token in context_ref if str(token).strip()]
        if not tokens:
            raise ValueError("context_ref is empty")
        return tokens

    if context_ref is None:
        raise ValueError("context_ref is empty")
    if not isinstance(context_ref, str):
        raise ValueError("context_ref must be a string or token list")

    ref = context_ref.strip()
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


def _is_step_output_ref(context_ref: Any) -> bool:
    if context_ref is None or not isinstance(context_ref, (str, list, tuple)):
        return False
    tokens = _context_ref_tokens(context_ref)
    return bool(tokens) and tokens[0] == "step_outputs"


def _looks_like_step_output_alias(context_ref: Any, grounding_context: dict[str, Any]) -> bool:
    if context_ref is None or not isinstance(context_ref, (str, list, tuple)):
        return False
    tokens = _context_ref_tokens(context_ref)
    if not tokens or tokens[0] == "step_outputs" or len(tokens) < 2:
        return False
    root_keys = {str(key).strip() for key in (grounding_context or {})}
    return tokens[0] not in root_keys


def _looks_like_context_ref_string(
    value: Any,
    grounding_context: dict[str, Any],
    *,
    step_outputs: dict[str, Any] | None = None,
) -> bool:
    if not isinstance(value, str):
        return False
    ref = value.strip()
    if not ref:
        return False
    if ref.startswith("/"):
        return True
    try:
        tokens = _context_ref_tokens(ref)
    except Exception:
        return False
    if not tokens or len(tokens) < 2:
        return False
    root_keys = {str(key).strip() for key in (grounding_context or {})}
    if tokens[0] == "step_outputs":
        return True
    if tokens[0] in root_keys:
        return True
    if _looks_like_step_output_alias(ref, grounding_context):
        if isinstance(step_outputs, dict) and tokens[0] in step_outputs:
            return True
    return False


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
    context_ref: Any,
    grounding_context: dict[str, Any],
    *,
    step_outputs: dict[str, Any] | None = None,
) -> Any:
    """Resolve one context_ref against planner context and optional step outputs.

    Supports JSON Pointer (`/parts/SG/observed_pose/x`) and dot paths
    (`parts.SG.observed_pose.x`) for compatibility with the current bridge TODO.
    """
    ref_display = _context_ref_display(context_ref)
    tokens = _context_ref_tokens(context_ref)
    root = dict(grounding_context or {})
    if step_outputs is not None:
        root["step_outputs"] = deepcopy(step_outputs)
    available_step_outputs = root.get("step_outputs") if isinstance(root.get("step_outputs"), dict) else {}

    def _resolve_step_output_alias_token(token: str) -> str | None:
        if not isinstance(available_step_outputs, dict) or not token:
            return None
        if token in available_step_outputs:
            return token
        canonical = _canonicalize_step_output_alias(token)
        if not canonical:
            return None
        for key in available_step_outputs:
            if _canonicalize_step_output_alias(key) == canonical:
                return str(key)
        return None

    if (
        tokens
        and tokens[0] not in root
        and isinstance(available_step_outputs, dict)
    ):
        resolved_alias = _resolve_step_output_alias_token(tokens[0])
        if resolved_alias is not None:
            tokens = ["step_outputs", resolved_alias, *tokens[1:]]
    elif len(tokens) >= 2 and tokens[0] == "step_outputs":
        resolved_alias = _resolve_step_output_alias_token(tokens[1])
        if resolved_alias is not None:
            tokens = ["step_outputs", resolved_alias, *tokens[2:]]
    return deepcopy(_walk_context_tokens(root, tokens, context_ref=ref_display))


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
            raw_context_ref = value.get("context_ref")
            if preserve_step_output_refs and (
                _is_step_output_ref(raw_context_ref)
                or _looks_like_step_output_alias(raw_context_ref, grounding_context)
            ):
                return deepcopy(value)
            resolved = resolve_context_ref(
                raw_context_ref,
                grounding_context,
                step_outputs=step_outputs,
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

    if preserve_step_output_refs and (
        _is_step_output_ref(value)
        or _looks_like_step_output_alias(value, grounding_context)
    ):
        return deepcopy(value)

    if _looks_like_context_ref_string(
        value,
        grounding_context,
        step_outputs=step_outputs,
    ):
        resolved = resolve_context_ref(
            value,
            grounding_context,
            step_outputs=step_outputs,
        )
        return resolved

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


def preview_step_output(
    *,
    primitive: str,
    params: dict[str, Any],
    snapshot: dict[str, Any],
    grounding_context: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Return a validation-time preview output using profile-owned hooks."""
    resource_type = str(
        dict(snapshot.get("resource_core") or {}).get("resource_type")
        or snapshot.get("resource_type")
        or ""
    ).strip().lower()
    profile = get_resource_profile(resource_type or "resource")
    resolver = (profile.preview_output_map or {}).get(primitive)
    if not callable(resolver):
        return None, f"primitive '{primitive}' does not support store_as"
    return resolver(params, snapshot, grounding_context)


def extract_step_output(
    *,
    primitive: str,
    params: dict[str, Any],
    step_result: dict[str, Any],
    resource_type: str = "",
) -> tuple[dict[str, Any] | None, str | None]:
    """Extract one normalized runtime step output using profile-owned hooks."""
    profile = get_resource_profile(resource_type or "resource")
    resolver = (profile.extract_output_map or {}).get(primitive)
    if not callable(resolver):
        return None, f"primitive '{primitive}' does not support store_as"
    return resolver(params, step_result)


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


def _bridge_semantic_tags(
    primitive_name: str,
    *,
    preconditions: dict[str, dict[str, Any]],
    effects: dict[str, dict[str, Any]],
    resource_type: str = "",
) -> dict[str, Any]:
    def _field_delta(field: str) -> dict[str, Any] | None:
        rule = dict(effects.get(field) or {})
        return rule or None

    profile = get_resource_profile(resource_type) if resource_type else None
    observation_output_schema: dict[str, Any] | None = None
    if profile is not None:
        observation_output_schema = deepcopy(
            dict(getattr(profile, "observation_output_schema_map", {}) or {}).get(primitive_name)
            or None
        )

    carried_entity_field = str(getattr(profile, "carried_entity_field", "") or "").strip()
    carried_entity_rule = dict(preconditions.get(carried_entity_field) or {}) if carried_entity_field else {}
    requires_empty_carrier = carried_entity_rule.get("equals", object()) is None
    required_carried_entity = (
        carried_entity_rule.get("equals")
        if "equals" in carried_entity_rule and carried_entity_rule.get("equals") not in (None, "")
        else None
    )
    observation_kind = None
    operation_kind = "motion"
    goal_state_hint = None
    requires_live_observation = False
    part_effect = None
    location_effect = None
    carried_entity_effect = None

    mapped_kind = (
        str((profile.primitive_kind_map or {}).get(primitive_name) or "").strip()
        if profile is not None
        else ""
    )
    if mapped_kind:
        operation_kind = mapped_kind

    if primitive_name in (profile.preview_output_map if profile is not None else {}):
        observation_kind = primitive_name
        operation_kind = "observe" if operation_kind == "motion" else operation_kind
        requires_live_observation = True
    elif primitive_name == "move_to_named_pose":
        operation_kind = "home"
        location_effect = "named_pose"
    elif primitive_name in {"move_cartesian", "move_pose", "move_relative"}:
        operation_kind = "motion"
        location_effect = "cartesian_motion"
    elif primitive_name == "rotate_wrist":
        operation_kind = "orient"

    return {
        "produces_observation": primitive_name in (profile.preview_output_map if profile is not None else {}),
        "operation_kind": operation_kind,
        "operation_family": operation_kind,
        "observation_kind": observation_kind,
        "observation_output_schema": observation_output_schema,
        "resource_state_delta": _field_delta("current_state"),
        "part_state_delta": _field_delta("part_state"),
        "carried_entity_delta": _field_delta(carried_entity_field) if carried_entity_field else None,
        "part_effect": part_effect,
        "location_effect": location_effect,
        "carried_entity_effect": carried_entity_effect,
        "requires_live_observation": requires_live_observation,
        "goal_state_hint": goal_state_hint,
        "requires_empty_carrier": requires_empty_carrier,
        "required_carried_entity": deepcopy(required_carried_entity),
    }


def _primitive_owner(resource_agent: Any, resource_type: str) -> Any | None:
    profile = get_resource_profile_for_agent(resource_agent)
    if profile.primitive_owner_resolver is not None:
        owner = profile.primitive_owner_resolver(resource_agent)
        if owner is not None:
            return owner
    primitive_names = getattr(resource_agent, "_BRIDGE_PRIMITIVES", None)
    if primitive_names:
        return resource_agent
    return None


def build_primitive_catalog(resource_agent: Any) -> list[dict[str, Any]]:
    """Build the private primitive catalogue for one bridge-capable resource."""
    resource_type = bridge_resource_type(
        resource=resource_agent,
        static_capabilities=deepcopy(getattr(resource_agent, "static_capabilities", {}) or {}),
    )

    owner = _primitive_owner(resource_agent, resource_type)
    if owner is None:
        return []

    analyzer = FunctionAnalyzer()
    primitive_names = sorted(getattr(resource_agent, "_BRIDGE_PRIMITIVES", []) or [])
    rows: list[dict[str, Any]] = []

    for primitive_name in primitive_names:
        fn = getattr(owner, primitive_name, None)
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
                "resource_type": resource_type,
                "description": analyzed.get("description", ""),
                "params": params_summary,
                "parameters": params_schema,
                "required_params": list(params_schema.get("required") or []),
                "preconditions": preconditions,
                "effects": effects,
                "semantic_summary": _semantic_summary(preconditions, effects),
                "bridge_semantics": _bridge_semantic_tags(
                    primitive_name,
                    preconditions=preconditions,
                    effects=effects,
                    resource_type=resource_type,
                ),
            }
        )

    return rows


_OPERATION_KIND_ORDER = [
    "observe",
    "motion",
    "home",
    "orient",
]


def build_primitive_reference_card(
    primitive_catalog: list[dict[str, Any]],
) -> str:
    """Build a human-readable reference card from an existing primitive catalog.

    Groups primitives by ``bridge_semantics.operation_kind`` and emits a
    compact multi-line summary suitable for injection into an LLM prompt.
    """
    if not primitive_catalog:
        return ""

    by_kind: dict[str, list[dict[str, Any]]] = {}
    for entry in primitive_catalog:
        if not isinstance(entry, dict):
            continue
        semantics = entry.get("bridge_semantics") or {}
        kind = str(semantics.get("operation_kind") or "motion").strip()
        by_kind.setdefault(kind, []).append(entry)

    lines: list[str] = []
    ordered_kinds = [k for k in _OPERATION_KIND_ORDER if k in by_kind]
    ordered_kinds.extend(k for k in sorted(by_kind) if k not in ordered_kinds)

    for kind in ordered_kinds:
        label = kind.upper().replace("_", " ")
        lines.append(f"[{label}]")
        for entry in by_kind[kind]:
            name = str(entry.get("name") or "").strip()
            if not name:
                continue

            required = entry.get("required_params") or []
            optional = [
                p
                for p in (entry.get("params") or {})
                if p not in required
            ]
            sig_parts = [p for p in required]
            sig_parts.extend(f"{p}?" for p in optional)
            sig = ", ".join(sig_parts)

            parts: list[str] = [f"  {name}({sig})"]

            summary = str(entry.get("semantic_summary") or "").strip()
            if summary:
                parts.append(f"    Semantics: {summary}")

            semantics = entry.get("bridge_semantics") or {}
            output_schema = semantics.get("observation_output_schema")
            if output_schema and isinstance(output_schema, dict):
                schema_str = ", ".join(
                    f"{k}: {v}" if not isinstance(v, dict) else f"{k}: {{{', '.join(v)}}}"
                    for k, v in output_schema.items()
                )
                parts.append(f"    Output (store_as): {{{schema_str}}}")

            lines.append("\n".join(parts))
        lines.append("")

    return "\n".join(lines).rstrip()


def get_resource_bridge_snapshot(resource_agent: Any) -> dict[str, Any]:
    """Return the current primitive-level bridge snapshot for any resource.
    """
    resource_type = bridge_resource_type(
        resource=resource_agent,
        static_capabilities=deepcopy(getattr(resource_agent, "static_capabilities", {}) or {}),
    )
    profile = get_resource_profile(resource_type)

    if profile.snapshot_builder is not None:
        raw_snapshot = profile.snapshot_builder(resource_agent) or {}
        if not isinstance(raw_snapshot, dict):
            raw_snapshot = {}
        raw_snapshot = deepcopy(raw_snapshot)
    else:
        raw_snapshot = {}
        if hasattr(resource_agent, "_snapshot_state"):
            raw_snapshot = resource_agent._snapshot_state() or {}
        if not isinstance(raw_snapshot, dict):
            raw_snapshot = {}
        raw_snapshot = deepcopy(raw_snapshot)
        raw_snapshot.setdefault("resource_type", resource_type)
        raw_snapshot.setdefault(
            "current_state",
            str(getattr(resource_agent, "_current_state", "") or "").strip() or "idle",
        )

    canonical = canonical_bridge_resource(
        resource_jid=str(getattr(resource_agent, "jid", "") or ""),
        resource_type=resource_type,
        snapshot=raw_snapshot,
        modeled_state={},
    )
    primitive_catalog = build_primitive_catalog(resource_agent)
    canonical["bridge_adapter"] = bridge_adapter_capabilities(
        resource_type,
        primitive_catalog=primitive_catalog,
    )
    return canonical

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
    profile = get_resource_profile(
        str(
            dict(snapshot.get("resource_core") or {}).get("resource_type")
            or snapshot.get("resource_type")
            or "resource"
        ).strip().lower()
        or "resource"
    )
    for field, rule in preconditions.items():
        value = resource_snapshot_field_value(snapshot, field, profile=profile)
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
    profile = get_resource_profile(
        str(
            dict(next_snapshot.get("resource_core") or {}).get("resource_type")
            or next_snapshot.get("resource_type")
            or "resource"
        ).strip().lower()
        or "resource"
    )

    for field, rule in effects.items():
        if "set" in rule:
            next_snapshot = resource_snapshot_set_field(
                next_snapshot,
                field,
                deepcopy(rule["set"]),
                profile=profile,
            )
            continue
        if "set_from_param" in rule:
            next_snapshot = resource_snapshot_set_field(
                next_snapshot,
                field,
                params.get(str(rule["set_from_param"])),
                profile=profile,
            )
            continue
        if "set_from_param_any_of" in rule:
            resolved_value = None
            for raw_param_name in (rule.get("set_from_param_any_of") or []):
                param_name = str(raw_param_name or "").strip()
                if not param_name:
                    continue
                if param_name in params and params.get(param_name) not in (None, ""):
                    resolved_value = deepcopy(params.get(param_name))
                    break
            next_snapshot = resource_snapshot_set_field(
                next_snapshot,
                field,
                resolved_value,
                profile=profile,
            )
            continue
        if "pose_absolute_from_params" in rule:
            x_key, y_key, z_key = list(rule["pose_absolute_from_params"])
            next_snapshot = resource_snapshot_set_field(
                next_snapshot,
                field,
                {
                    "x": float(params[x_key]),
                    "y": float(params[y_key]),
                    "z": float(params[z_key]),
                },
                profile=profile,
            )
            continue
        if "pose_relative_from_params" in rule:
            dx_key, dy_key, dz_key = list(rule["pose_relative_from_params"])
            current_pose = dict(
                resource_snapshot_field_value(next_snapshot, field, profile=profile) or {}
            )
            next_snapshot = resource_snapshot_set_field(
                next_snapshot,
                field,
                {
                    "x": float(current_pose["x"]) + float(params[dx_key]),
                    "y": float(current_pose["y"]) + float(params[dy_key]),
                    "z": float(current_pose["z"]) + float(params[dz_key]),
                },
                profile=profile,
            )
            continue
        if rule.get("set_unknown"):
            next_snapshot = resource_snapshot_set_field(
                next_snapshot,
                field,
                None,
                profile=profile,
            )

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
            profile = get_resource_profile(
                str(
                    dict(projected.get("resource_core") or {}).get("resource_type")
                    or projected.get("resource_type")
                    or "resource"
                ).strip().lower() or "resource"
            )
            _snap = resource_snapshot_fields_map(
                projected, profile.snapshot_fields, profile=profile,
            )
            logger.debug(
                "[PrimitiveSemantics] step %d %s FAILED precondition — "
                "snapshot: %s, preconditions=%r",
                index,
                primitive,
                ", ".join(f"{k}={v!r}" for k, v in _snap.items()),
                primitive_meta.get("preconditions"),
            )
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


def expected_snapshot_from_bridge_snapshot(
    snapshot: dict[str, Any],
    *,
    resource_type: str = "resource",
) -> dict[str, Any]:
    """Keep only the fields that are stable enough for start-state validation."""
    profile = get_resource_profile(resource_type)
    return {
        field: resource_snapshot_field_value(snapshot, field, profile=profile)
        for field in profile.snapshot_fields
    }


def snapshot_matches_expected(actual: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, str | None]:
    """Return whether actual snapshot satisfies expected snapshot fields."""
    profile = get_resource_profile(
        str(
            dict(actual.get("resource_core") or {}).get("resource_type")
            or actual.get("resource_type")
            or "resource"
        ).strip().lower()
        or "resource"
    )
    for key in expected:
        actual_value = resource_snapshot_field_value(actual, key, profile=profile)
        if actual_value != expected.get(key):
            return (
                False,
                f"expected {key}={expected.get(key)!r} but found {actual_value!r}",
            )
    return True, None


def sync_agent_from_bridge_snapshot(resource_agent: Any, snapshot: dict[str, Any]) -> None:
    """Apply bridge snapshot fields back onto resource agent runtime state."""
    current_state = resource_snapshot_field_value(snapshot, "current_state")
    if current_state is not None:
        resource_agent._current_state = str(current_state)

    profile = get_resource_profile_for_agent(resource_agent)
    for field, target in dict(profile.sync_map or {}).items():
        if not resource_snapshot_has_field(snapshot, field, profile=profile):
            continue
        value = resource_snapshot_field_value(snapshot, field, profile=profile)
        if callable(target):
            target(resource_agent, value)
        else:
            setattr(resource_agent, str(target), value)

"""Validate and compile LLM-authored SynthesizedTaskFn into runtime macros.

Reuses existing primitive validation from ``primitive_semantics`` and
compiles validated functions into the same ``macro_task`` format consumed
by ``execute_recovery_macro``.

Binding rules
-------------
* ``store_as`` writes to function-local scope.
* ``context_ref`` resolves in order: (1) function-local scope
  (prior ``store_as`` outputs), (2) function inputs,
  (3) session ``observation_store``.
* Forward references to later ``store_as`` are validation errors.
* Cross-function references are not allowed — data flows between
  functions only via explicit function inputs.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.mutation_types import (
    SynthesizedTaskFn,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.primitive_semantics import (
    apply_effects_to_snapshot,
    validate_and_project_steps,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_synthesized_function(
    fn_def: SynthesizedTaskFn,
    primitive_catalog: list[dict[str, Any]],
    resource_snapshot: dict[str, Any],
    *,
    observation_store: dict[str, Any] | None = None,
    capability_flags: dict[str, bool] | None = None,
    resource_jid: str = "",
) -> tuple[bool, dict[str, Any], list[str]]:
    """Validate a :class:`SynthesizedTaskFn` against the primitive catalog.

    Parameters
    ----------
    fn_def:
        The synthesized function to validate.
    primitive_catalog:
        Primitive catalog entries for the target resource.
    resource_snapshot:
        Current canonical snapshot for the target resource.
    observation_store:
        Outputs from prior top-level ``observe`` turns (available as
        fallback ``context_ref`` scope).
    capability_flags:
        Resource capability flags (e.g. ``{"can_pick": True}``).

    Returns
    -------
    tuple:
        ``(is_valid, projected_snapshot, errors)`` — if *errors* is
        non-empty the function should not be compiled.
    """
    errors: list[str] = []

    # 1. Basic structural checks.
    if not fn_def.name:
        errors.append("function has no name")
    if not fn_def.primitive_program:
        errors.append(f"function '{fn_def.name}': primitive_program is empty")
        return False, resource_snapshot, errors

    # 2. Check all primitives exist in catalog.
    catalog_names = frozenset(
        str(e.get("name", "")).strip()
        for e in primitive_catalog
        if isinstance(e, dict)
    )
    for i, step in enumerate(fn_def.primitive_program):
        prim = str(step.get("primitive", "")).strip()
        if prim and prim not in catalog_names:
            errors.append(
                f"function '{fn_def.name}' step {i}: "
                f"primitive '{prim}' not in catalog"
            )

    if errors:
        return False, resource_snapshot, errors

    # 3. Check capability constraints.
    if fn_def.resource_constraints and capability_flags:
        for cap, required in fn_def.resource_constraints.items():
            if cap == "resource_type":
                continue
            if isinstance(required, bool) and required:
                if not capability_flags.get(cap, False):
                    errors.append(
                        f"function '{fn_def.name}': resource lacks "
                        f"capability '{cap}'"
                    )

    if errors:
        return False, resource_snapshot, errors

    # 4. Validate binding rules (no forward refs, no cross-function refs).
    binding_errors = _validate_bindings(
        fn_def.primitive_program, observation_store or {},
    )
    errors.extend(binding_errors)
    if errors:
        return False, resource_snapshot, errors

    # 5. Build grounding context for context_ref resolution.
    grounding_context = _build_fn_grounding_context(
        fn_def.inputs, observation_store or {},
    )

    # 6. Validate precondition chain and project state using existing
    #    primitive_semantics pipeline.
    is_valid, projected, projection_error = validate_and_project_steps(
        fn_def.primitive_program,
        primitive_catalog,
        resource_snapshot,
        grounding_context=grounding_context,
    )
    if projection_error:
        errors.append(f"function '{fn_def.name}': {projection_error}")
        return False, projected, errors

    # 7. Check declared preconditions against starting snapshot.
    #    Normalize keys: LLMs may emit 'xarm6@localhost.current_state'
    #    instead of 'current_state'.
    precondition_errors = _check_declared_preconditions(
        _normalize_field_keys(fn_def.preconditions, resource_jid),
        resource_snapshot,
    )
    errors.extend(
        f"function '{fn_def.name}': {e}" for e in precondition_errors
    )

    # 8. Check projected post-state matches declared expected_post_state.
    post_state_errors = _check_post_state_match(
        _normalize_field_keys(fn_def.expected_post_state, resource_jid),
        projected,
    )
    errors.extend(
        f"function '{fn_def.name}': {e}" for e in post_state_errors
    )

    # 9. Check declared effects consistency with projected snapshot.
    effect_errors = _check_declared_effects(
        _normalize_field_keys(fn_def.effects, resource_jid),
        resource_snapshot, projected,
    )
    errors.extend(
        f"function '{fn_def.name}': {e}" for e in effect_errors
    )

    return len(errors) == 0, projected, errors


def compile_synthesized_function_to_macro(
    fn_def: SynthesizedTaskFn,
    resource_jid: str,
    *,
    start_state: str = "",
    macro_index: int = 0,
    bridge_sequence_id: str = "",
) -> dict[str, Any]:
    """Compile a validated SynthesizedTaskFn into a ``macro_task`` dict.

    The output format matches what ``_apply_primitive_bridge_proposal()``
    (process_planner.py:903) expects and what ``execute_recovery_macro``
    (product_agent.py) consumes at runtime.
    """
    out_state = str(
        fn_def.expected_post_state.get("current_state") or start_state or "idle"
    ).strip()

    macro: dict[str, Any] = {
        "resource_jid": resource_jid,
        "macro_name": fn_def.name,
        "description": fn_def.intent,
        "primitive_steps": deepcopy(fn_def.primitive_program),
        "task_metadata": {
            "in_state": start_state or "idle",
            "out_state": out_state,
            "part_transition": {},
        },
    }

    # Extract part_transition from effects.
    for field_name, effect_spec in fn_def.effects.items():
        if field_name in ("held_part", "carried_entity"):
            op = effect_spec.get("set") if isinstance(effect_spec, dict) else None
            if op is not None:
                macro["task_metadata"]["part_transition"] = {
                    "held": {"to": op},
                }

    if bridge_sequence_id:
        macro["bridge_sequence_id"] = bridge_sequence_id
        macro["bridge_sequence_index"] = macro_index

    return macro


# ---------------------------------------------------------------------------
# Internal validators
# ---------------------------------------------------------------------------

def _validate_bindings(
    primitive_program: list[dict[str, Any]],
    observation_store: dict[str, Any],
) -> list[str]:
    """Validate store_as / context_ref binding rules."""
    errors: list[str] = []
    defined_outputs: set[str] = set()

    for i, step in enumerate(primitive_program):
        if not isinstance(step, dict):
            continue
        prim = str(step.get("primitive", "")).strip()

        # Check context_refs in params.
        params = step.get("params") or {}
        _check_context_refs_in_params(
            params, defined_outputs, observation_store,
            step_index=i, primitive=prim, errors=errors,
        )

        # Register store_as output.
        store_as = str(step.get("store_as") or "").strip()
        if store_as:
            if store_as in defined_outputs:
                errors.append(
                    f"step {i} {prim}: duplicate store_as '{store_as}'"
                )
            defined_outputs.add(store_as)

    return errors


def _check_context_refs_in_params(
    params: Any,
    defined_outputs: set[str],
    observation_store: dict[str, Any],
    *,
    step_index: int,
    primitive: str,
    errors: list[str],
    path: str = "",
) -> None:
    """Recursively check context_refs in param values."""
    if isinstance(params, dict):
        context_ref = params.get("context_ref")
        if context_ref is not None:
            ref_str = str(context_ref).strip()
            # Extract the root key (before first dot).
            root_key = ref_str.split(".")[0] if "." in ref_str else ref_str
            if (
                root_key not in defined_outputs
                and root_key not in observation_store
            ):
                errors.append(
                    f"step {step_index} {primitive}: context_ref '{ref_str}' "
                    f"references undefined output '{root_key}' "
                    f"(available: {sorted(defined_outputs | set(observation_store.keys()))})"
                )
        else:
            for key, value in params.items():
                _check_context_refs_in_params(
                    value, defined_outputs, observation_store,
                    step_index=step_index, primitive=primitive,
                    errors=errors, path=f"{path}.{key}" if path else key,
                )
    elif isinstance(params, (list, tuple)):
        for idx, item in enumerate(params):
            _check_context_refs_in_params(
                item, defined_outputs, observation_store,
                step_index=step_index, primitive=primitive,
                errors=errors, path=f"{path}[{idx}]",
            )


def _normalize_field_keys(
    fields: dict[str, Any],
    resource_jid: str,
) -> dict[str, Any]:
    """Strip resource JID prefix from field keys if present.

    LLMs may emit ``xarm6@localhost.current_state`` instead of
    ``current_state``.  This also handles dotted paths like
    ``xarm6@localhost.occupancy.location`` → ``occupancy.location``.
    """
    if not resource_jid:
        return fields
    normalized: dict[str, Any] = {}
    prefix = resource_jid + "."
    for key, value in fields.items():
        if key.startswith(prefix):
            key = key[len(prefix):]
        normalized[key] = value
    return normalized


def _check_declared_preconditions(
    preconditions: dict[str, dict[str, Any]],
    snapshot: dict[str, Any],
) -> list[str]:
    """Check declared preconditions against the starting snapshot."""
    errors: list[str] = []
    flat = _flatten_snapshot(snapshot)

    for field_name, rule in preconditions.items():
        if not isinstance(rule, dict):
            continue
        actual = flat.get(field_name)

        if "equals" in rule:
            expected = rule["equals"]
            if actual != expected:
                errors.append(
                    f"precondition failed: {field_name} must be "
                    f"{expected!r}, current value is {actual!r}"
                )
        elif "not_equals" in rule:
            forbidden = rule["not_equals"]
            if actual == forbidden:
                errors.append(
                    f"precondition failed: {field_name} must not be "
                    f"{forbidden!r}, current value is {actual!r}"
                )
        elif "exists" in rule:
            if actual is None:
                errors.append(
                    f"precondition failed: {field_name} must exist"
                )

    return errors


def _check_post_state_match(
    expected_post: dict[str, Any],
    projected: dict[str, Any],
) -> list[str]:
    """Check that projected snapshot matches declared expected_post_state."""
    errors: list[str] = []
    flat = _flatten_snapshot(projected)

    for field_name, expected_value in expected_post.items():
        actual = flat.get(field_name)
        if actual != expected_value:
            errors.append(
                f"expected_post_state mismatch: {field_name} projected "
                f"as {actual!r}, declared {expected_value!r}"
            )

    return errors


def _check_declared_effects(
    effects: dict[str, dict[str, Any]],
    start_snapshot: dict[str, Any],
    projected_snapshot: dict[str, Any],
) -> list[str]:
    """Check that declared effects are consistent with actual projection."""
    errors: list[str] = []
    start_flat = _flatten_snapshot(start_snapshot)
    projected_flat = _flatten_snapshot(projected_snapshot)

    for field_name, effect_spec in effects.items():
        if not isinstance(effect_spec, dict):
            continue
        if "set" in effect_spec:
            declared = effect_spec["set"]
            actual = projected_flat.get(field_name)
            if actual != declared:
                errors.append(
                    f"effect mismatch: {field_name} declared set={declared!r}, "
                    f"projected={actual!r}"
                )

    return errors


def _flatten_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Flatten a canonical snapshot into a single-level dict.

    Looks in resource_core, resource_facets, and top-level keys.
    """
    flat: dict[str, Any] = {}
    # Top-level (scalars and nested dicts like occupancy).
    for k, v in snapshot.items():
        if k in ("resource_core", "resource_facets"):
            continue
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if not isinstance(sub_v, dict):
                    flat.setdefault(f"{k}.{sub_k}", sub_v)
        else:
            flat[k] = v

    # resource_core.
    for k, v in (snapshot.get("resource_core") or {}).items():
        if not isinstance(v, dict):
            flat.setdefault(k, v)

    # resource_facets (flatten one level).
    for facet_key, facet_data in (snapshot.get("resource_facets") or {}).items():
        if isinstance(facet_data, dict):
            for k, v in facet_data.items():
                if not isinstance(v, dict):
                    flat.setdefault(k, v)

    return flat


def _build_fn_grounding_context(
    fn_inputs: dict[str, Any],
    observation_store: dict[str, Any],
) -> dict[str, Any]:
    """Build a grounding context dict for context_ref resolution.

    The existing ``resolve_param_refs`` in primitive_semantics looks up
    refs in the grounding_context dict.  We populate it with the
    observation store and function inputs so that context_refs like
    ``lcp_scan.poses`` resolve correctly.
    """
    ctx: dict[str, Any] = {}
    ctx.update(observation_store)
    ctx.update(fn_inputs)
    return ctx

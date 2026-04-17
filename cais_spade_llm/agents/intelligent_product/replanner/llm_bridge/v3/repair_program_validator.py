"""Full deterministic validation pipeline for RepairProgram proposals.

Two-layer validation:

**Layer A** — Local structural validation (fast, always runs first):
  schema, function synthesis, symbol resolution, mutation, primitive projection.

**Layer B** — Projected global validation (runs only if Layer A passes):
  projected task graph copy, temporary FSA compilation, LTLf/DFA safety,
  continuation viability via ``restore_runtime_progress()``, obligation
  discharge.
"""

from __future__ import annotations

import logging
from math import hypot
import re
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.mutation_types import (
    RepairProgram,
    RepairStep,
    RepairStepKind,
    RiskLevel,
    SynthesizedTaskFn,
    TaskMutationStep,
    TaskMutationType,
    ValidatedRepairProgram,
    compute_primitive_fingerprint,
    compute_signature_hash,
    repair_program_from_dict,
    repair_program_to_dict,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.function_synthesis import (
    compile_synthesized_function_to_macro,
    validate_synthesized_function,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.mutation_compiler import (
    compile_mutations,
    validate_mutation_step,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.part_state_semantics import (
    part_state_requires_external_localization,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.recovery_context_builder import (
    part_observation_status,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.primitive_semantics import (
    preview_step_output,
    resolve_param_refs,
)
from cais_spade_llm.resources.robot.place_geometry_resolution import (
    has_place_geometry_fields,
    resolve_place_geometry,
)

logger = logging.getLogger(__name__)

_POSE_DEPENDENT_OPERATION_KINDS = frozenset({"pick", "pick_place"})
_OBSERVED_PICKUP_XY_TOL_M = 0.06
_OBSERVED_PICKUP_Z_TOL_M = 0.25
_MEANINGFUL_POST_GRASP_XY_M = 0.08
_PLACEMENT_SLOT_XY_TOL_M = 0.06
_PLACE_TARGET_Z_TOL_M = 0.06
_PLACE_APPROACH_Z_TOL_M = 0.08
_REASONING_PHASE_GROUP = {
    "resolve_safety": 0,
    "restore_capability": 0,
    "free_executor": 0,
    "recover_entities": 1,
    "restore_resume_entry": 1,
    "adapt_goals": 1,
    "replace_suffix": 2,
    "resume_modeled_suffix": 2,
}
_CANONICAL_PRIMITIVE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MUTATION_TYPE_ALIASES = {
    "append_action": TaskMutationType.INSERT.value,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _canonical_primitive_name(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    match = _CANONICAL_PRIMITIVE_RE.match(token)
    if match:
        return match.group(0)
    return token.split()[0]


def _normalized_mutation_type_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    return _MUTATION_TYPE_ALIASES.get(token, token)

def validate_repair_program(
    program: RepairProgram,
    *,
    primitive_catalogs: dict[str, list[dict[str, Any]]],
    resource_snapshots: dict[str, dict[str, Any]],
    current_nodes: list[dict[str, Any]],
    available_task_actions: list[dict[str, Any]] | None = None,
    observation_store: dict[str, Any] | None = None,
    capability_flags_map: dict[str, dict[str, bool]] | None = None,
    safety_validator: Any | None = None,
    safety_rules: list[dict[str, Any]] | None = None,
    runtime_monitor_state: dict[str, Any] | None = None,
    compile_fsa_fn: Any | None = None,
    recovery_library: Any | None = None,
    active_obligations: list[dict[str, Any]] | None = None,
    part_states: dict[str, dict[str, Any]] | None = None,
    product_geometry: dict[str, Any] | None = None,
    workspace_bounds: dict[str, dict[str, float]] | None = None,
    grounded_environment_facts: dict[str, Any] | None = None,
) -> ValidatedRepairProgram:
    """Run the full two-layer validation pipeline.

    Parameters
    ----------
    program:
        The LLM-emitted RepairProgram.
    primitive_catalogs:
        Resource JID → primitive catalog entries.
    resource_snapshots:
        Resource JID → canonical snapshot.
    current_nodes:
        Current ``ProcessPlanner.nodes``.
    available_task_actions:
        Tools-catalog rows for direct nominal task-action reuse.
    observation_store:
        Outputs from prior observe turns.
    capability_flags_map:
        Resource JID → capability flags.
    safety_validator:
        Optional ``PlanSafetyValidator`` for Layer B LTLf/DFA checks.
    safety_rules:
        Safety rule dicts (used for Layer B).
    runtime_monitor_state:
        ``{"completed_task_ids": [...], "running_task_ids": [...],
        "failed_task_ids": [...], "current_state": str}``.
    compile_fsa_fn:
        Callable that compiles task nodes into an FSA dict.
    recovery_library:
        Optional ``RecoveryLibrary`` for risk classification.

    Returns
    -------
    ValidatedRepairProgram:
        The validated wrapper with authoritative risk_level, approval,
        continuation_viable, and rejection_reasons.
    """
    program = _normalize_anchor_only_staging_place_geometry(
        program,
        grounded_environment_facts=grounded_environment_facts,
    )
    result = ValidatedRepairProgram(program=program)

    # ===================================================================
    # LAYER A — Local structural validation
    # ===================================================================

    # A1: Schema validation.
    schema_errors = _validate_schema(program)
    if schema_errors:
        result.rejection_reasons.extend(
            _make_rejections("A", "schema", schema_errors)
        )
        return result

    reasoning_errors = _validate_reasoning_contract(program)
    if reasoning_errors:
        result.rejection_reasons.extend(
            _make_rejections("A", "reasoning_contract", reasoning_errors)
        )
        return result

    # A2: Function synthesis validation.
    # Build execution order from steps so functions sharing a resource
    # are validated against chained projected snapshots (e.g. stow_mcp
    # on ur5e → pick_lg on ur5e uses post-stow snapshot).
    fn_map = {f.name: f for f in program.function_defs}
    fn_execution_order: list[tuple[str, str]] = []  # (fn_name, resource_jid)
    for step in program.steps:
        if step.kind == RepairStepKind.CALL_FUNCTION:
            fn_name = str(step.payload.get("function_name", "")).strip()
            fn_def_for_step = fn_map.get(fn_name)
            if fn_def_for_step:
                rj = _resolve_function_resource(fn_def_for_step, program)
                fn_execution_order.append((fn_name, rj))

    # Track running snapshot per resource (chains across functions).
    running_snapshots: dict[str, dict[str, Any]] = dict(resource_snapshots)
    fn_projected_snapshots: dict[str, dict[str, Any]] = {}
    validated_fn_names: set[str] = set()

    # First validate in execution order (step order).
    for fn_name, resource_jid in fn_execution_order:
        fn_def = fn_map.get(fn_name)
        if fn_def is None or fn_name in validated_fn_names:
            continue
        validated_fn_names.add(fn_name)
        catalog = primitive_catalogs.get(resource_jid, [])
        snapshot = running_snapshots.get(resource_jid, {})
        caps = (capability_flags_map or {}).get(resource_jid, {})

        is_valid, projected, fn_errors = validate_synthesized_function(
            fn_def, catalog, snapshot,
            observation_store=observation_store,
            capability_flags=caps,
            resource_jid=resource_jid,
        )
        if fn_errors:
            result.rejection_reasons.extend(
                _make_rejections("A", "function_synthesis", fn_errors)
            )
        fn_projected_snapshots[fn_name] = projected
        # Chain: update running snapshot for this resource.
        if is_valid:
            running_snapshots[resource_jid] = projected

    # Validate any remaining function_defs not referenced in steps.
    for fn_def in program.function_defs:
        if fn_def.name in validated_fn_names:
            continue
        resource_jid = _resolve_function_resource(fn_def, program)
        catalog = primitive_catalogs.get(resource_jid, [])
        snapshot = running_snapshots.get(resource_jid, {})
        caps = (capability_flags_map or {}).get(resource_jid, {})

        is_valid, projected, fn_errors = validate_synthesized_function(
            fn_def, catalog, snapshot,
            observation_store=observation_store,
            capability_flags=caps,
            resource_jid=resource_jid,
        )
        if fn_errors:
            result.rejection_reasons.extend(
                _make_rejections("A", "function_synthesis", fn_errors)
            )
        fn_projected_snapshots[fn_def.name] = projected

    if result.rejection_reasons:
        return result

    grounding_errors = _validate_grounding_dependencies(
        program,
        fn_execution_order,
        primitive_catalogs,
        part_states=part_states,
    )
    if grounding_errors:
        result.rejection_reasons.extend(
            _make_rejections("A", "grounding_dependency", grounding_errors)
        )
        return result

    # A2.5: Workspace bounds feasibility.
    if workspace_bounds:
        ws_errors = _validate_workspace_feasibility(
            program, fn_execution_order, workspace_bounds,
        )
        if ws_errors:
            result.rejection_reasons.extend(
                _make_rejections("A", "workspace_feasibility", ws_errors)
            )
            return result

    # A3: Symbol resolution and step sequencing.
    seq_errors = _validate_step_sequencing(
        program,
        available_task_actions=available_task_actions,
    )
    if seq_errors:
        result.rejection_reasons.extend(
            _make_rejections("A", "step_sequencing", seq_errors)
        )
        return result

    direct_call_policy_errors = _validate_direct_task_call_policy(
        program,
        available_task_actions=available_task_actions,
    )
    if direct_call_policy_errors:
        result.rejection_reasons.extend(
            _make_rejections("A", "direct_task_call_policy", direct_call_policy_errors)
        )
        return result

    # A4: Mutation validation.
    mutation_steps = _extract_mutation_steps(program)
    for ms in mutation_steps:
        m_errors = validate_mutation_step(ms, current_nodes)
        if m_errors:
            result.rejection_reasons.extend(
                _make_rejections("A", "mutation", m_errors)
            )

    if result.rejection_reasons:
        return result

    # A5: Primitive precondition/effect projection — already done in A2.
    # (validate_synthesized_function runs validate_and_project_steps internally)

    # ===================================================================
    # LAYER B — Projected global validation (only if Layer A passed)
    # ===================================================================

    # B6-B7: Project mutations onto a copy of the task graph and compile FSA.
    projected_nodes = deepcopy(current_nodes)
    if mutation_steps:
        patches, compile_errors = compile_mutations(
            mutation_steps, projected_nodes,
        )
        if compile_errors:
            result.rejection_reasons.extend(
                _make_rejections("B", "mutation_compile", compile_errors)
            )
            return result
        projected_nodes = _apply_patches_to_nodes(patches, projected_nodes)

    # Insert synthesized function calls as execute_recovery_macro tasks and
    # allow direct catalog-backed task actions to flow through unchanged.
    call_steps = [
        s for s in program.steps
        if s.kind == RepairStepKind.CALL_FUNCTION
    ]
    for i, step in enumerate(call_steps):
        fn_name = str(step.payload.get("function_name", "")).strip()
        resource_jid = str(step.payload.get("resource_jid", "")).strip()
        fn_def = _find_function_def(fn_name, program)
        if fn_def is None:
            tool_row = _select_catalog_task_row(
                available_task_actions,
                resource_jid=resource_jid,
                function_name=fn_name,
            )
            if tool_row is None:
                continue  # Already caught in A3.
            projected_nodes.append({
                "id": f"repair_task_{fn_name}_{i}",
                "type": "task",
                "function_name": fn_name,
                "resource_jid": resource_jid,
                "params": dict(step.payload.get("args") or {}),
                "status": "pending",
                "predecessors": [],
                "successors": [],
                "in_state": str(tool_row.get("in_state") or "").strip(),
                "out_state": str(tool_row.get("out_state") or "").strip(),
            })
            continue
        macro = compile_synthesized_function_to_macro(
            fn_def, resource_jid, macro_index=i,
        )
        # Add as a task node for FSA compilation.
        task_id = f"repair_fn_{fn_name}_{i}"
        projected_nodes.append({
            "id": task_id,
            "type": "task",
            "function_name": "execute_recovery_macro",
            "resource_jid": resource_jid,
            "params": {
                "macro_name": macro["macro_name"],
                "primitive_steps": macro["primitive_steps"],
            },
            "status": "pending",
            "predecessors": [],
            "successors": [],
            "in_state": macro["task_metadata"].get("in_state", ""),
            "out_state": macro["task_metadata"].get("out_state", ""),
        })

    # B8: Compile temporary FSA and validate against LTLf/DFA safety.
    if safety_validator is not None and compile_fsa_fn is not None:
        try:
            temp_fsa = compile_fsa_fn(projected_nodes)
            if temp_fsa:
                is_safe, violations = safety_validator.validate_plan_fsa(
                    temp_fsa, plan={"nodes": projected_nodes},
                )
                if not is_safe:
                    for v in violations:
                        result.rejection_reasons.append({
                            "layer": "B",
                            "check": "ltlf_safety",
                            "message": str(v.get("message") or v.get("description", "")),
                            "rule_id": str(v.get("rule_id", "")),
                        })
                    return result
        except Exception as exc:
            logger.warning(
                "Layer B FSA compilation/safety check failed: %s", exc,
                exc_info=True,
            )
            result.rejection_reasons.append({
                "layer": "B",
                "check": "fsa_compilation",
                "message": f"FSA compilation or safety check failed: {exc}",
            })
            return result

        # B9: Continuation viability.
        if runtime_monitor_state and temp_fsa:
            viable, viability_error = _check_continuation_viability(
                temp_fsa, runtime_monitor_state,
            )
            result.continuation_viable = viable
            if not viable and viability_error:
                result.rejection_reasons.append({
                    "layer": "B",
                    "check": "continuation_viability",
                    "message": viability_error,
                })
                return result
        else:
            # If we can't check viability, assume viable (best effort).
            result.continuation_viable = True
    else:
        # No safety validator — Layer B skipped.
        result.continuation_viable = True

    # B10: Obligation discharge (success_conditions reachable).
    # This is a structural check — are the declared success conditions
    # theoretically achievable given the projected state?
    discharge_errors = _check_obligation_discharge(
        program.success_conditions,
        fn_projected_snapshots,
        resource_snapshots,
        active_obligations=active_obligations,
        part_states=part_states,
    )
    if discharge_errors:
        result.rejection_reasons.extend(
            _make_rejections("B", "obligation_discharge", discharge_errors)
        )
        return result

    pre_resume_errors = _check_pre_resume_obligation_coverage(
        program,
        active_obligations=active_obligations,
        resource_snapshots=resource_snapshots,
        part_states=part_states,
    )
    if pre_resume_errors:
        result.rejection_reasons.extend(
            _make_rejections("B", "pre_resume_obligation", pre_resume_errors)
        )
        return result

    under_modeled_errors = _check_under_modeled_part_recovery_semantics(
        program,
        active_obligations=active_obligations,
        part_states=part_states,
        resource_snapshots=resource_snapshots,
        product_geometry=product_geometry,
        observation_store=observation_store,
    )

    blocker_staging_errors = _check_blocker_staging_witnesses(
        program,
        resource_snapshots=resource_snapshots,
        grounded_environment_facts=grounded_environment_facts,
    )
    if under_modeled_errors or blocker_staging_errors:
        if under_modeled_errors:
            result.rejection_reasons.extend(
                _make_rejections("B", "under_modeled_part_recovery", under_modeled_errors)
            )
        if blocker_staging_errors:
            result.rejection_reasons.extend(
                _make_rejections("B", "blocker_staging_witness", blocker_staging_errors)
            )
        return result

    # ===================================================================
    # Risk classification (validator-owned)
    # ===================================================================
    result.risk_level = _classify_risk(program, recovery_library)
    result.requires_operator_approval = result.risk_level == RiskLevel.HIGH

    return result


def _normalize_anchor_only_staging_place_geometry(
    program: RepairProgram,
    *,
    grounded_environment_facts: dict[str, Any] | None,
) -> RepairProgram:
    details = {
        str(name).strip(): dict(row)
        for name, row in dict((grounded_environment_facts or {}).get("staging_destination_details") or {}).items()
        if str(name).strip()
    }
    if not details:
        return program

    program_dict = repair_program_to_dict(program)
    changed = False
    for fn_def in list(program_dict.get("function_defs") or []):
        if not isinstance(fn_def, dict):
            continue
        for step in list(fn_def.get("primitive_program") or []):
            if not isinstance(step, dict):
                continue
            if str(step.get("primitive") or "").strip() != "compute_place_targets":
                continue
            params = step.get("params") or {}
            if not isinstance(params, dict):
                continue
            destination = str(params.get("destination_location") or "").strip()
            if not destination:
                continue
            existing_geometry = params.get("product_geometry")
            if has_place_geometry_fields(existing_geometry):
                continue
            part_name = str(params.get("part_name") or "").strip()
            if not part_name:
                pick_ctx = params.get("pick_ctx")
                if isinstance(pick_ctx, dict):
                    part_name = str(pick_ctx.get("part_name") or "").strip()
            if not part_name:
                continue
            detail = details.get(part_name) or {}
            if str(detail.get("placement_support") or "").strip() != "anchor_only":
                continue
            if str(detail.get("destination") or "").strip() != destination:
                continue
            explicit_geometry = dict(detail.get("explicit_product_geometry") or {})
            if not has_place_geometry_fields(explicit_geometry):
                continue
            params["product_geometry"] = explicit_geometry
            step["params"] = params
            changed = True

    if not changed:
        return program
    return repair_program_from_dict(program_dict)


# ---------------------------------------------------------------------------
# Layer A validators
# ---------------------------------------------------------------------------

def _validate_workspace_feasibility(
    program: RepairProgram,
    fn_execution_order: list[tuple[str, str]],
    workspace_bounds: dict[str, dict[str, float]],
) -> list[str]:
    """A2.5: Check that literal coordinates fall within workspace bounds.

    Only catches hardcoded x/y/z values — ``context_ref`` parameters are
    resolved at runtime and cannot be checked here.
    """
    errors: list[str] = []
    fn_map = {f.name: f for f in program.function_defs}

    # Build fn → resource mapping from execution order.
    fn_resource: dict[str, str] = {fn: rjid for fn, rjid in fn_execution_order}
    # Fill in any remaining functions.
    for fn_def in program.function_defs:
        if fn_def.name not in fn_resource:
            fn_resource[fn_def.name] = _resolve_function_resource(fn_def, program)

    for fn_def in program.function_defs:
        resource_jid = fn_resource.get(fn_def.name, "")
        bounds = workspace_bounds.get(resource_jid)
        if not bounds:
            continue
        for i, prim in enumerate(fn_def.primitive_program or []):
            params = prim.get("params") or {}
            # Check direct x/y/z params (from move_cartesian etc.).
            _check_coords(errors, fn_def.name, i, resource_jid, params, bounds)
            # Check nested pose objects (target_pose, position).
            for key in ("target_pose", "position", "pose"):
                nested = params.get(key)
                if isinstance(nested, dict):
                    _check_coords(
                        errors, fn_def.name, i, resource_jid, nested, bounds,
                    )
    return errors


def _extract_part_refs_from_text(
    text: str,
    *,
    known_parts: set[str],
) -> set[str]:
    refs: set[str] = set()
    tokenized = [token for token in re.split(r"[^A-Za-z0-9_@-]+", text) if token]
    for token in tokenized:
        if token in known_parts:
            refs.add(token)
    if text in known_parts:
        refs.add(text)
    return refs


def _collect_param_part_refs(
    value: Any,
    *,
    known_parts: set[str],
) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            refs.update(_collect_param_part_refs(item, known_parts=known_parts))
        return refs
    if isinstance(value, (list, tuple, set)):
        for item in value:
            refs.update(_collect_param_part_refs(item, known_parts=known_parts))
        return refs
    if isinstance(value, str):
        refs.update(_extract_part_refs_from_text(value.strip(), known_parts=known_parts))
    return refs


def _collect_pose_ref_parts(
    value: Any,
    *,
    known_parts: set[str],
) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            refs.update(_collect_pose_ref_parts(item, known_parts=known_parts))
        return refs
    if isinstance(value, (list, tuple, set)):
        for item in value:
            refs.update(_collect_pose_ref_parts(item, known_parts=known_parts))
        return refs
    if not isinstance(value, str):
        return refs

    text = value.strip()
    if "observed_pose" not in text and "last_known_pose" not in text:
        return refs
    refs.update(_extract_part_refs_from_text(text, known_parts=known_parts))
    return refs


def _validate_grounding_dependencies(
    program: RepairProgram,
    fn_execution_order: list[tuple[str, str]],
    primitive_catalogs: dict[str, list[dict[str, Any]]],
    *,
    part_states: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Reject only proposals that need exact pose for an untrusted part."""
    errors: list[str] = []
    part_states = dict(part_states or {})
    untrusted_parts = {
        str(part_name).strip()
        for part_name, info in part_states.items()
        if str(part_name).strip()
        and part_state_requires_external_localization((info or {}).get("state"))
        and part_observation_status(dict(info or {})) != "observed"
    }
    if not untrusted_parts:
        return errors

    fn_resource: dict[str, str] = {fn_name: rjid for fn_name, rjid in fn_execution_order}
    for fn_def in program.function_defs:
        if fn_def.name not in fn_resource:
            fn_resource[fn_def.name] = _resolve_function_resource(fn_def, program)

    catalog_by_resource: dict[str, dict[str, dict[str, Any]]] = {}
    for resource_jid, catalog in primitive_catalogs.items():
        catalog_by_resource[resource_jid] = {
            str(entry.get("name") or "").strip(): entry
            for entry in (catalog or [])
            if isinstance(entry, dict) and str(entry.get("name") or "").strip()
        }

    for fn_def in program.function_defs:
        resource_jid = fn_resource.get(fn_def.name, "")
        catalog = catalog_by_resource.get(resource_jid, {})
        for step_index, primitive_step in enumerate(fn_def.primitive_program or [], start=1):
            primitive_name = str(primitive_step.get("primitive", "")).strip()
            params = primitive_step.get("params") or {}
            primitive_meta = dict(catalog.get(primitive_name) or {})
            semantics = primitive_meta.get("bridge_semantics") or {}
            operation_kind = str(semantics.get("operation_kind") or "").strip()

            direct_part_refs = _collect_param_part_refs(
                params,
                known_parts=untrusted_parts,
            )
            pose_ref_parts = _collect_pose_ref_parts(
                params,
                known_parts=untrusted_parts,
            )

            pose_dependent_parts = set(pose_ref_parts)
            if operation_kind in _POSE_DEPENDENT_OPERATION_KINDS:
                pose_dependent_parts.update(direct_part_refs)

            for part_name in sorted(pose_dependent_parts):
                info = dict(part_states.get(part_name) or {})
                status = part_observation_status(info)
                if status == "observed":
                    continue
                if (
                    not part_state_requires_external_localization(
                        info.get("state")
                    )
                    and part_name not in pose_ref_parts
                ):
                    continue
                errors.append(
                    f"function '{fn_def.name}' step {step_index} primitive "
                    f"'{primitive_name}' depends on exact pose for part "
                    f"'{part_name}', but that part is {status}; emit observe "
                    f"first or use trusted observation data"
                )
    return errors


def _check_coords(
    errors: list[str],
    fn_name: str,
    prim_idx: int,
    resource_jid: str,
    params: dict[str, Any],
    bounds: dict[str, float],
) -> None:
    """Check x/y/z in params against workspace bounds."""
    for axis in ("x", "y", "z"):
        val = params.get(axis)
        if val is None or isinstance(val, dict):
            # dict means context_ref — skip, resolved at runtime.
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        lo = bounds.get(f"{axis}_min_m")
        hi = bounds.get(f"{axis}_max_m")
        if lo is not None and val < float(lo):
            errors.append(
                f"fn '{fn_name}' step {prim_idx}: {axis}={val:.4f} "
                f"< {axis}_min_m={float(lo):.4f} for {resource_jid}"
            )
        if hi is not None and val > float(hi):
            errors.append(
                f"fn '{fn_name}' step {prim_idx}: {axis}={val:.4f} "
                f"> {axis}_max_m={float(hi):.4f} for {resource_jid}"
            )


def _validate_schema(program: RepairProgram) -> list[str]:
    """A1: Basic structural validation."""
    errors: list[str] = []

    if not program.steps:
        errors.append("repair_program has no steps")

    for i, step in enumerate(program.steps):
        if not isinstance(step.kind, RepairStepKind):
            errors.append(f"step {i}: invalid kind '{step.kind}'")
        if step.kind == RepairStepKind.CALL_FUNCTION:
            fn_name = step.payload.get("function_name", "")
            resource_jid = step.payload.get("resource_jid", "")
            if not fn_name:
                errors.append(f"step {i}: call_function missing function_name")
            if not resource_jid:
                errors.append(f"step {i}: call_function missing resource_jid")
        elif step.kind == RepairStepKind.TASK_MUTATION:
            mt = _normalized_mutation_type_token(step.payload.get("mutation_type", ""))
            if mt not in {e.value for e in TaskMutationType}:
                errors.append(f"step {i}: invalid mutation_type '{mt}'")

    for fn_def in program.function_defs:
        if not fn_def.name:
            errors.append("function_def has no name")
        if not fn_def.primitive_program:
            errors.append(f"function '{fn_def.name}': empty primitive_program")

    return errors


def _validate_reasoning_contract(program: RepairProgram) -> list[str]:
    """A1.5: Require explicit ReAct/TSS reasoning before primitive acceptance."""
    errors: list[str] = []
    reasoning = dict(program.reasoning or {})
    if not reasoning:
        return ["repair_program is missing the required reasoning object"]

    abstract_repair_order = reasoning.get("abstract_repair_order")
    if isinstance(abstract_repair_order, list) and abstract_repair_order:
        last_group = -1
        previous_phase_type = ""
        for index, phase in enumerate(abstract_repair_order, start=1):
            if not isinstance(phase, dict):
                errors.append(f"reasoning.abstract_repair_order[{index}] must be an object")
                continue
            phase_type = str(phase.get("phase_type") or "").strip()
            if phase_type not in _REASONING_PHASE_GROUP:
                errors.append(
                    f"reasoning.abstract_repair_order[{index}] has invalid phase_type '{phase_type}'"
                )
                continue
            phase_group = _REASONING_PHASE_GROUP[phase_type]
            if phase_group < last_group:
                errors.append(
                    "reasoning.abstract_repair_order must progress from preparatory phases "
                    f"toward recovery/adaptation and then terminal resume/replace phases; "
                    f"phase_type '{phase_type}' at row {index} cannot appear after "
                    f"'{previous_phase_type}'"
                )
                break
            last_group = max(last_group, phase_group)
            previous_phase_type = phase_type
            if phase_type in ("resume_modeled_suffix", "replace_suffix") and index != len(abstract_repair_order):
                errors.append(
                    f"reasoning.abstract_repair_order[{index}] uses terminal phase_type "
                    f"'{phase_type}' before the end of the outline"
                )
                break

    has_resume_suffix = any(step.kind == RepairStepKind.RESUME_SUFFIX for step in program.steps)
    if has_resume_suffix and isinstance(abstract_repair_order, list) and abstract_repair_order:
        last_phase_type = str((abstract_repair_order[-1] or {}).get("phase_type") or "").strip()
        if last_phase_type != "resume_modeled_suffix":
                errors.append(
                    "reasoning.abstract_repair_order must end with phase_type='resume_modeled_suffix' "
                    "when the program emits resume_suffix"
                )
    elif has_resume_suffix:
        errors.append(
            "reasoning.abstract_repair_order must be available when the program emits resume_suffix"
        )

    return errors


def _validate_step_sequencing(
    program: RepairProgram,
    *,
    available_task_actions: list[dict[str, Any]] | None = None,
) -> list[str]:
    """A3: Check that steps reference valid functions and task IDs."""
    errors: list[str] = []
    fn_names = {fn.name for fn in program.function_defs}
    resume_indices = [
        index for index, step in enumerate(program.steps)
        if step.kind == RepairStepKind.RESUME_SUFFIX
    ]
    if len(resume_indices) > 1:
        errors.append("repair_program may contain at most one resume_suffix step")
    if resume_indices and resume_indices[-1] != len(program.steps) - 1:
        errors.append("resume_suffix must be the final step in repair_program")

    for i, step in enumerate(program.steps):
        if step.kind == RepairStepKind.CALL_FUNCTION:
            fn_name = str(step.payload.get("function_name", "")).strip()
            resource_jid = str(step.payload.get("resource_jid", "")).strip()
            if (
                fn_name
                and fn_name not in fn_names
                and _select_catalog_task_row(
                    available_task_actions,
                    resource_jid=resource_jid,
                    function_name=fn_name,
                ) is None
            ):
                errors.append(
                    f"step {i}: call_function references undefined "
                    f"function '{fn_name}' (defined: {sorted(fn_names)})"
                )

    return errors


def _required_catalog_arg_keys(tool_row: dict[str, Any]) -> list[str]:
    required: list[str] = []
    params = dict(tool_row.get("params") or {})
    required_context_keys = {
        str(item).strip().lower()
        for item in (tool_row.get("required_context_keys") or [])
        if str(item).strip()
    }
    context_mapping = dict(tool_row.get("context_mapping") or {})
    location_param = str(context_mapping.get("location_param") or "").strip()
    if required_context_keys and location_param:
        required.append(location_param)

    if (
        "part_name" in params
        and (
            required_context_keys
            or tool_row.get("part_transition")
            or tool_row.get("part_in_state")
        )
    ):
        required.append("part_name")

    seen: set[str] = set()
    ordered: list[str] = []
    for key in required:
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _validate_direct_task_call_policy(
    program: RepairProgram,
    *,
    available_task_actions: list[dict[str, Any]] | None = None,
) -> list[str]:
    errors: list[str] = []
    fn_names = {fn.name for fn in program.function_defs}

    for i, step in enumerate(program.steps):
        if step.kind != RepairStepKind.CALL_FUNCTION:
            continue
        fn_name = str(step.payload.get("function_name") or "").strip()
        resource_jid = str(step.payload.get("resource_jid") or "").strip()
        if not fn_name or fn_name in fn_names:
            continue
        tool_row = _select_catalog_task_row(
            available_task_actions,
            resource_jid=resource_jid,
            function_name=fn_name,
        )
        if tool_row is None:
            continue
        errors.append(
            f"step {i}: direct nominal task call '{fn_name}' on '{resource_jid}' is not allowed in repair_program; "
            "synthesize a primitive-backed function instead"
        )

    return errors


def _extract_mutation_steps(program: RepairProgram) -> list[TaskMutationStep]:
    """Extract TaskMutationStep objects from RepairProgram steps."""
    mutations: list[TaskMutationStep] = []
    for step in program.steps:
        if step.kind != RepairStepKind.TASK_MUTATION:
            continue
        payload = step.payload
        try:
            mt = TaskMutationType(
                _normalized_mutation_type_token(payload.get("mutation_type", ""))
            )
        except ValueError:
            continue
        mutations.append(TaskMutationStep(
            mutation_type=mt,
            target_task_ids=list(payload.get("target_task_ids") or []),
            payload=dict(payload.get("payload") or payload),
        ))
    return mutations


# ---------------------------------------------------------------------------
# Layer B validators
# ---------------------------------------------------------------------------

def _check_continuation_viability(
    fsa: dict[str, Any],
    runtime_state: dict[str, Any],
) -> tuple[bool, str]:
    """B9: Check that the repaired FSA admits continuation.

    Uses ``OnlineFsaMonitor.restore_runtime_progress()`` logic:
    replay completed/running/failed task IDs into the projected FSA
    and verify the resulting state has at least one enabled transition.
    """
    try:
        from cais_spade_llm.agents.central_controller.online_fsa_monitor import (
            OnlineFsaMonitor,
        )
    except ImportError:
        return True, ""

    try:
        monitor = OnlineFsaMonitor(fsa)
        monitor.restore_runtime_progress(
            completed_task_ids=runtime_state.get("completed_task_ids"),
            running_task_ids=runtime_state.get("running_task_ids"),
            failed_task_ids=runtime_state.get("failed_task_ids"),
        )

        # Check if current state has at least one enabled transition.
        current = monitor.current_state
        if current is None:
            return False, "monitor reached null state after restore"

        A = (fsa or {}).get("A") or {}
        Xm = set(A.get("Xm") or [])

        # If already in a marked state, viable.
        if current in Xm:
            return True, ""

        # Check for enabled transitions from current state.
        transitions = A.get("Tr") or []
        enabled = [t for t in transitions if t.get("from") == current]
        if not enabled:
            return False, (
                f"dead-end state '{current}' after restore — "
                f"no enabled transitions toward a marked state"
            )

        return True, ""

    except Exception as exc:
        return False, f"continuation viability check failed: {exc}"


def _check_obligation_discharge(
    success_conditions: list[dict[str, Any]],
    fn_projected_snapshots: dict[str, dict[str, Any]],
    resource_snapshots: dict[str, dict[str, Any]],
    active_obligations: list[dict[str, Any]] | None = None,
    part_states: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """B10: Check that success conditions are structurally reachable.

    Also verifies that every non-safety active obligation is covered by
    at least one success_condition (entity + field match).  Obligations
    that are already satisfied in the current state are skipped.

    Success conditions are auto-derived from active obligations when the
    LLM does not provide them explicitly.
    """
    errors: list[str] = []

    # Check that active goal/reentry obligations are covered.
    if active_obligations:
        # Build coverage keys from LLM-provided success_conditions AND
        # auto-derived from non-safety active obligations.
        sc_keys: set[tuple[str, str]] = set()
        for sc in (success_conditions or []):
            entity = str(sc.get("entity", "")).strip()
            field = str(sc.get("field", "")).strip()
            if entity and field:
                sc_keys.add((entity, field))

        # Auto-derive from active obligations so LLM need not emit them.
        for ob in active_obligations:
            if str(ob.get("type", "")).strip() == "safety":
                continue
            entity = str(ob.get("entity", "")).strip()
            field = str(ob.get("field", "")).strip()
            if entity and field:
                sc_keys.add((entity, field))

        for ob in active_obligations:
            ob_type = str(ob.get("type", "")).strip()
            if ob_type == "safety":
                continue
            entity = str(ob.get("entity", "")).strip()
            field = str(ob.get("field", "")).strip()
            expected = str(ob.get("expected", "")).strip()
            if not entity or not field:
                continue

            # Check if obligation is already satisfied in current state.
            entity_kind = str(ob.get("entity_kind", "")).strip()
            current_val = None
            if entity_kind == "resource":
                snap = resource_snapshots.get(entity) or {}
                current_val = snap.get(field)
                if current_val is None:
                    current_val = dict(snap.get("resource_core") or {}).get(field)
            elif entity_kind == "part":
                current_val = (part_states or {}).get(entity, {}).get(field)
            if current_val is not None and str(current_val) == expected:
                continue  # Already satisfied — no success_condition needed.

            if (entity, field) not in sc_keys:
                errors.append(
                    f"obligation not discharged: {entity}.{field} must reach {expected}"
                )

    return errors


def _current_obligation_value(
    obligation: dict[str, Any],
    *,
    resource_snapshots: dict[str, dict[str, Any]],
    part_states: dict[str, dict[str, Any]] | None,
) -> Any:
    entity_kind = str(obligation.get("entity_kind", "")).strip()
    entity = str(obligation.get("entity", "")).strip()
    field = str(obligation.get("field", "")).strip()
    if not entity_kind or not entity or not field:
        return None

    if entity_kind == "resource":
        snap = resource_snapshots.get(entity) or {}
        current_val = snap.get(field)
        if current_val is None:
            current_val = dict(snap.get("resource_core") or {}).get(field)
        return current_val

    if entity_kind == "part":
        return (part_states or {}).get(entity, {}).get(field)

    return None


def _collect_touched_entities_before_resume(
    program: RepairProgram,
    *,
    known_parts: set[str],
) -> tuple[set[str], set[str], set[tuple[str, str, str]]]:
    touched_resources: set[str] = set()
    touched_parts: set[str] = set()
    explicit_waits: set[tuple[str, str, str]] = set()
    fn_map = {fn.name: fn for fn in program.function_defs}

    for step in program.steps:
        if step.kind == RepairStepKind.RESUME_SUFFIX:
            break

        if step.kind == RepairStepKind.CALL_FUNCTION:
            resource_jid = str(step.payload.get("resource_jid") or "").strip()
            if resource_jid:
                touched_resources.add(resource_jid)
            touched_parts.update(
                _collect_param_part_refs(
                    step.payload.get("args") or {},
                    known_parts=known_parts,
                )
            )
            fn_name = str(step.payload.get("function_name") or "").strip()
            fn_def = fn_map.get(fn_name)
            if fn_def is None:
                continue
            touched_parts.update(_collect_param_part_refs(fn_def.inputs, known_parts=known_parts))
            touched_parts.update(_collect_param_part_refs(fn_def.preconditions, known_parts=known_parts))
            touched_parts.update(_collect_param_part_refs(fn_def.effects, known_parts=known_parts))
            touched_parts.update(_collect_param_part_refs(fn_def.expected_post_state, known_parts=known_parts))
            for primitive_step in fn_def.primitive_program or []:
                touched_parts.update(
                    _collect_param_part_refs(primitive_step, known_parts=known_parts)
                )
            continue

        if step.kind == RepairStepKind.WAIT:
            until = step.payload.get("until") or {}
            entity_kind = str(until.get("entity_kind") or "").strip()
            entity = str(until.get("entity") or "").strip()
            field = str(until.get("field") or "").strip()
            if entity_kind and entity and field:
                explicit_waits.add((entity_kind, entity, field))
            continue

        if step.kind == RepairStepKind.TASK_MUTATION:
            payload = step.payload or {}
            touched_parts.update(_collect_param_part_refs(payload, known_parts=known_parts))
            for key in ("resource_jid", "new_resource_jid"):
                value = str(payload.get(key) or "").strip()
                if value:
                    touched_resources.add(value)
            mutation_payload = payload.get("payload") or {}
            if isinstance(mutation_payload, dict):
                for key in ("resource_jid", "new_resource_jid"):
                    value = str(mutation_payload.get(key) or "").strip()
                    if value:
                        touched_resources.add(value)

    return touched_resources, touched_parts, explicit_waits


def _check_pre_resume_obligation_coverage(
    program: RepairProgram,
    *,
    active_obligations: list[dict[str, Any]] | None,
    resource_snapshots: dict[str, dict[str, Any]],
    part_states: dict[str, dict[str, Any]] | None,
) -> list[str]:
    """Reject plans that outsource unfinished bridge work to resume_suffix."""
    if not any(step.kind == RepairStepKind.RESUME_SUFFIX for step in program.steps):
        return []

    errors: list[str] = []
    # Build coverage keys from LLM-provided success_conditions AND
    # auto-derived from non-safety active obligations.
    success_keys = {
        (
            str(condition.get("entity_kind") or "").strip(),
            str(condition.get("entity") or "").strip(),
            str(condition.get("field") or "").strip(),
        )
        for condition in (program.success_conditions or [])
        if str(condition.get("entity_kind") or "").strip()
        and str(condition.get("entity") or "").strip()
        and str(condition.get("field") or "").strip()
    }
    # Auto-derive from active obligations so LLM need not emit them.
    for ob in (active_obligations or []):
        if str(ob.get("type") or "").strip() == "safety":
            continue
        ek = str(ob.get("entity_kind") or "").strip()
        ent = str(ob.get("entity") or "").strip()
        fld = str(ob.get("field") or "").strip()
        if ek and ent and fld:
            success_keys.add((ek, ent, fld))
    known_parts = {
        str(name).strip()
        for name in (part_states or {})
        if str(name).strip()
    }
    touched_resources, touched_parts, explicit_waits = _collect_touched_entities_before_resume(
        program,
        known_parts=known_parts,
    )

    for obligation in active_obligations or []:
        if str(obligation.get("type") or "").strip().lower() == "safety":
            continue
        if not bool(obligation.get("must_satisfy_before_resume", True)):
            continue

        current_val = _current_obligation_value(
            obligation,
            resource_snapshots=resource_snapshots,
            part_states=part_states,
        )
        expected = obligation.get("expected")
        if current_val is not None and current_val == expected:
            continue

        entity_kind = str(obligation.get("entity_kind") or "").strip()
        entity = str(obligation.get("entity") or "").strip()
        field = str(obligation.get("field") or "").strip()
        obligation_class = str(obligation.get("obligation_class") or "bridge_goal").strip()
        if not entity_kind or not entity or not field:
            continue

        if (entity_kind, entity, field) not in success_keys:
            errors.append(
                f"pre-resume {obligation_class} obligation is missing from success_conditions: "
                f"{entity}.{field} must reach {expected}"
            )
            continue

        entity_touched = (
            entity in touched_resources
            if entity_kind == "resource"
            else entity in touched_parts
        )
        explicitly_waited = (entity_kind, entity, field) in explicit_waits
        if not entity_touched:
            if explicitly_waited:
                errors.append(
                    f"resume_suffix cannot start after wait-only gating for "
                    f"{entity}.{field} ({obligation_class}); a causal repair action "
                    "must handle that obligation before resume"
                )
                continue
            errors.append(
                f"resume_suffix cannot start before the repair prefix explicitly handles "
                f"{entity}.{field} ({obligation_class})"
            )

    return errors


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_string_leaves(value: Any) -> list[str]:
    leaves: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            leaves.extend(_iter_string_leaves(item))
        return leaves
    if isinstance(value, (list, tuple, set)):
        for item in value:
            leaves.extend(_iter_string_leaves(item))
        return leaves
    if isinstance(value, str):
        leaves.append(value)
    return leaves


def _absolute_pose_from_params(params: dict[str, Any]) -> dict[str, float] | None:
    if not isinstance(params, dict):
        return None
    x = _float_or_none(params.get("x"))
    y = _float_or_none(params.get("y"))
    z = _float_or_none(params.get("z"))
    if x is None or y is None or z is None:
        return None
    return {"x": x, "y": y, "z": z}


def _params_reference_observed_pose(params: dict[str, Any], *, part_name: str) -> bool:
    for leaf in _iter_string_leaves(params):
        text = str(leaf).strip()
        if "observed_pose" in text and part_name in text:
            return True
    return False


def _resolve_recovery_motion_params(
    params: dict[str, Any],
    *,
    part_name: str,
    observed_pose: dict[str, Any] | None,
    step_outputs: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    grounding_context: dict[str, Any] = {}
    if isinstance(observed_pose, dict):
        grounding_context = {
            "parts": {
                part_name: {
                    "observed_pose": deepcopy(observed_pose),
                }
            }
        }
    resolved = dict(params)
    for key in ("x", "y", "z", "dx", "dy", "dz"):
        if key not in params:
            continue
        try:
            resolved[key] = resolve_param_refs(
                params[key],
                grounding_context,
                step_outputs=step_outputs,
            )
        except Exception:
            resolved[key] = params[key]
    for key in ("target_pose", "position", "pose", "pick_ctx", "product_geometry"):
        if key not in params:
            continue
        try:
            resolved[key] = resolve_param_refs(
                params[key],
                grounding_context,
                step_outputs=step_outputs,
            )
        except Exception:
            resolved[key] = params[key]
    return resolved


def _helper_pick_is_grounded(
    params: dict[str, Any],
    *,
    part_name: str,
    observed_pose: dict[str, Any] | None,
    step_outputs: dict[str, Any] | None,
) -> bool:
    if _params_reference_observed_pose(params, part_name=part_name):
        return True
    resolved = _resolve_recovery_motion_params(
        params,
        part_name=part_name,
        observed_pose=observed_pose,
        step_outputs=step_outputs,
    )
    target_pose = resolved.get("target_pose")
    if isinstance(target_pose, dict) and _observed_pose_match(target_pose, observed_pose):
        return True
    return False


def _assign_local_event_fact(root: dict[str, Any], dotted_path: str, value: Any) -> None:
    current = root
    tokens = [str(token).strip() for token in str(dotted_path or "").split(".") if str(token).strip()]
    if not tokens:
        return
    for token in tokens[:-1]:
        next_value = current.get(token)
        if not isinstance(next_value, dict):
            next_value = {}
            current[token] = next_value
        current = next_value
    current[tokens[-1]] = deepcopy(value)


def _helper_place_has_grounded_geometry(
    params: dict[str, Any],
    *,
    part_name: str = "",
) -> bool:
    raw_geometry = params.get("product_geometry")
    geometry = resolve_place_geometry(
        part_name=str(params.get("part_name") or part_name or "").strip(),
        destination_location=str(params.get("destination_location") or "").strip(),
        product_geometry=raw_geometry if isinstance(raw_geometry, dict) else None,
        execution_mode="simulation",
    )
    return has_place_geometry_fields(geometry)


def _args_match_expected_destination(args: dict[str, Any], expected_location: str) -> bool:
    if not expected_location:
        return False
    for leaf in _iter_string_leaves(args):
        if str(leaf).strip() == expected_location:
            return True
    return False


def _motion_primitive_name(primitive_name: Any) -> str:
    return str(primitive_name or "").strip()


def _primitive_references_location(
    primitive_step: dict[str, Any],
    *,
    location_token: str,
) -> bool:
    if not location_token:
        return False
    params = dict((primitive_step or {}).get("params") or {})
    return any(str(leaf).strip() == location_token for leaf in _iter_string_leaves(params))


def _primitive_is_motion(primitive_step: dict[str, Any]) -> bool:
    return _motion_primitive_name((primitive_step or {}).get("primitive")) in {
        "move_to_named_pose",
        "move_cartesian",
        "move_pose",
        "move_relative",
    }


def _primitive_is_descend_witness(primitive_step: dict[str, Any]) -> bool:
    primitive_name = _motion_primitive_name((primitive_step or {}).get("primitive"))
    params = dict((primitive_step or {}).get("params") or {})
    if primitive_name == "move_relative":
        dz = _float_or_none(params.get("dz"))
        return dz is not None and dz < 0.0
    return primitive_name in {"move_cartesian", "move_pose"}


def _geometry_for_part(
    product_geometry: dict[str, Any] | None,
    *,
    part_name: str,
    expected_location: str,
) -> dict[str, Any] | None:
    token = str(expected_location or "").strip().lower()
    if not token or "assembly_board" not in token:
        return None
    if not isinstance(product_geometry, dict):
        return None

    board = dict(product_geometry.get("assembly_board") or {})
    parts = dict(product_geometry.get("parts") or {})
    slot_xy = dict(board.get("slots") or {}).get(part_name)
    if not isinstance(slot_xy, (list, tuple)) or len(slot_xy) < 2:
        return None

    return {
        "slot_xy": list(slot_xy[:2]),
        "part_height_m": dict(parts.get("heights_m") or {}).get(part_name),
        "model_name": dict(parts.get("model_map") or {}).get(part_name),
        "slot_floor_z_m": board.get("slot_floor_z_m"),
        "board_center": board.get("center") or {},
    }


def _default_preview_snapshot(resource_snapshots: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    for snap in (resource_snapshots or {}).values():
        row = dict(snap or {})
        if str(row.get("resource_type") or "").strip().lower() == "robot":
            return row
    return {
        "resource_type": "robot",
        "resource_core": {"resource_type": "robot"},
    }


def _preview_place_targets_for_observed_part(
    *,
    part_name: str,
    expected_location: str,
    observed_pose: dict[str, Any] | None,
    product_geometry: dict[str, Any] | None,
    resource_snapshots: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    geometry = _geometry_for_part(
        product_geometry,
        part_name=part_name,
        expected_location=expected_location,
    )
    if geometry is None or not isinstance(observed_pose, dict):
        return None

    preview_snapshot = _default_preview_snapshot(resource_snapshots)
    pick_ctx, pick_err = preview_step_output(
        primitive="compute_pick_targets",
        params={"part_name": part_name, "product_geometry": geometry},
        snapshot=preview_snapshot,
        grounding_context={"parts": {part_name: {"observed_pose": observed_pose}}},
    )
    if pick_err or not isinstance(pick_ctx, dict):
        return None

    place_preview, place_err = preview_step_output(
        primitive="compute_place_targets",
        params={
            "part_name": part_name,
            "product_geometry": geometry,
            "pick_ctx": pick_ctx,
        },
        snapshot=preview_snapshot,
        grounding_context={},
    )
    if place_err or not isinstance(place_preview, dict):
        return None
    return place_preview


def _preview_pick_targets_for_observed_part(
    *,
    part_name: str,
    expected_location: str,
    observed_pose: dict[str, Any] | None,
    product_geometry: dict[str, Any] | None,
    resource_snapshots: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not isinstance(observed_pose, dict):
        return None

    geometry = _geometry_for_part(
        product_geometry,
        part_name=part_name,
        expected_location=expected_location,
    ) or {}
    preview_snapshot = _default_preview_snapshot(resource_snapshots)
    pick_ctx, pick_err = preview_step_output(
        primitive="compute_pick_targets",
        params={"part_name": part_name, "product_geometry": geometry},
        snapshot=preview_snapshot,
        grounding_context={"parts": {part_name: {"observed_pose": observed_pose}}},
    )
    if pick_err or not isinstance(pick_ctx, dict):
        return None
    return pick_ctx


def _pose_matches_expected(
    current_pose: dict[str, float] | None,
    expected_pose: dict[str, Any] | None,
    *,
    z_tol: float,
) -> bool:
    if not isinstance(current_pose, dict) or not isinstance(expected_pose, dict):
        return False
    cur_x = _float_or_none(current_pose.get("x"))
    cur_y = _float_or_none(current_pose.get("y"))
    cur_z = _float_or_none(current_pose.get("z"))
    exp_x = _float_or_none(expected_pose.get("x"))
    exp_y = _float_or_none(expected_pose.get("y"))
    exp_z = _float_or_none(expected_pose.get("z"))
    if None in {cur_x, cur_y, cur_z, exp_x, exp_y, exp_z}:
        return False
    return (
        hypot(cur_x - exp_x, cur_y - exp_y) <= _PLACEMENT_SLOT_XY_TOL_M
        and abs(cur_z - exp_z) <= z_tol
    )


def _function_targets_part(fn_def: SynthesizedTaskFn, *, part_name: str) -> bool:
    known_parts = {part_name}
    if _extract_part_refs_from_text(fn_def.name, known_parts=known_parts):
        return True
    if _extract_part_refs_from_text(fn_def.intent, known_parts=known_parts):
        return True
    for payload in (
        fn_def.inputs,
        fn_def.preconditions,
        fn_def.effects,
        fn_def.expected_post_state,
    ):
        if _collect_param_part_refs(payload, known_parts=known_parts):
            return True
    for primitive_step in fn_def.primitive_program or []:
        if _collect_param_part_refs(primitive_step, known_parts=known_parts):
            return True
    return False


def _function_references_location_token(
    fn_def: SynthesizedTaskFn,
    *,
    location_token: str,
) -> bool:
    token = str(location_token or "").strip()
    if not token:
        return False
    haystacks: list[Any] = [
        fn_def.name,
        fn_def.intent,
        fn_def.inputs,
        fn_def.preconditions,
        fn_def.effects,
        fn_def.expected_post_state,
    ]
    for primitive_step in fn_def.primitive_program or []:
        haystacks.append(dict(primitive_step or {}).get("params") or {})
    for payload in haystacks:
        for leaf in _iter_string_leaves(payload):
            leaf_text = str(leaf).strip()
            if leaf_text == token or token in leaf_text:
                return True
    return False


def _new_part_recovery_evidence() -> dict[str, Any]:
    return {
        "has_grasp": False,
        "pickup_grounded": False,
        "has_release_after_grasp": False,
        "has_nonlocal_transport": False,
        "has_place_approach_witness": False,
        "has_destination_witness": False,
        "has_placement_completion": False,
        "_after_grasp": False,
        "_cumulative_xy": 0.0,
        "_current_pose": None,
        "_placement_ready": False,
        "_carrier_resource_jid": "",
        "_ungrounded_place_helper": False,
    }


def _observation_step_outputs_for_part(
    observation_store: dict[str, Any] | None,
    *,
    part_name: str,
    observed_pose: dict[str, Any] | None,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for observation_key, payload in dict(observation_store or {}).items():
        token = str(observation_key or "").strip()
        if not token or not isinstance(payload, dict):
            continue
        observed_part = str(payload.get("part_name") or "").strip()
        payload_pose = dict(payload.get("pose") or {})
        if observed_part == part_name:
            rows[token] = deepcopy(payload)
            _assign_local_event_fact(rows, f"detected_part.{part_name}", payload)
            continue
        if payload_pose and _observed_pose_match(payload_pose, observed_pose):
            rows[token] = deepcopy(payload)
    return rows


def _observed_pose_match(
    current_pose: dict[str, float] | None,
    observed_pose: dict[str, Any] | None,
) -> bool:
    if not isinstance(current_pose, dict) or not isinstance(observed_pose, dict):
        return False
    obs_x = _float_or_none(observed_pose.get("x"))
    obs_y = _float_or_none(observed_pose.get("y"))
    obs_z = _float_or_none(observed_pose.get("z"))
    cur_x = _float_or_none(current_pose.get("x"))
    cur_y = _float_or_none(current_pose.get("y"))
    cur_z = _float_or_none(current_pose.get("z"))
    if None in {obs_x, obs_y, obs_z, cur_x, cur_y, cur_z}:
        return False
    return (
        hypot(cur_x - obs_x, cur_y - obs_y) <= _OBSERVED_PICKUP_XY_TOL_M
        and abs(cur_z - obs_z) <= _OBSERVED_PICKUP_Z_TOL_M
    )


def _advance_synthesized_part_recovery(
    evidence: dict[str, Any],
    *,
    fn_def: SynthesizedTaskFn,
    resource_jid: str,
    part_name: str,
    observed_pose: dict[str, Any] | None,
    pick_preview: dict[str, Any] | None,
    place_preview: dict[str, Any] | None,
    observation_store: dict[str, Any] | None,
) -> None:
    if not _function_targets_part(fn_def, part_name=part_name):
        if not (
            evidence.get("_after_grasp")
            and str(evidence.get("_carrier_resource_jid") or "").strip() == resource_jid
        ):
            return

    after_grasp = bool(evidence.get("_after_grasp"))
    cumulative_xy = float(evidence.get("_cumulative_xy") or 0.0)
    current_pose = evidence.get("_current_pose")
    if not isinstance(current_pose, dict):
        current_pose = None
    placement_ready = bool(evidence.get("_placement_ready"))
    local_step_outputs: dict[str, Any] = _observation_step_outputs_for_part(
        observation_store,
        part_name=part_name,
        observed_pose=observed_pose,
    )
    expected_pick_pose = dict((pick_preview or {}).get("target_pose") or {})
    expected_approach_pose = dict((place_preview or {}).get("approach_pose") or {})
    expected_target_pose = dict((place_preview or {}).get("target_pose") or {})
    for primitive_step in fn_def.primitive_program or []:
        primitive_name = str(primitive_step.get("primitive") or "").strip()
        params = dict(primitive_step.get("params") or {})
        step_targets_part = bool(
            _collect_param_part_refs(params, known_parts={part_name})
        ) or primitive_name in {"grasp_part", "release_part"}

        if primitive_name == "compute_pick_targets":
            if (
                isinstance(pick_preview, dict)
                and _helper_pick_is_grounded(
                    params,
                    part_name=part_name,
                    observed_pose=observed_pose,
                    step_outputs=local_step_outputs,
                )
            ):
                _assign_local_event_fact(
                    local_step_outputs,
                    f"pick_targets.{part_name}",
                    pick_preview,
                )
            continue
        if primitive_name == "compute_place_targets":
            resolved_helper_params = _resolve_recovery_motion_params(
                params,
                part_name=part_name,
                observed_pose=observed_pose,
                step_outputs=local_step_outputs,
            )
            if (
                isinstance(place_preview, dict)
                and _helper_place_has_grounded_geometry(
                    resolved_helper_params,
                    part_name=part_name,
                )
            ):
                _assign_local_event_fact(
                    local_step_outputs,
                    f"place_targets.{part_name}",
                    place_preview,
                )
            elif after_grasp and step_targets_part:
                evidence["_ungrounded_place_helper"] = True
            continue
        if primitive_name == "detect_parts":
            if isinstance(observed_pose, dict) and step_targets_part:
                _assign_local_event_fact(local_step_outputs, f"detected_part.{part_name}", {
                    "part_name": part_name,
                    "pose": deepcopy(observed_pose),
                })
            continue
        if primitive_name == "get_current_pose":
            if isinstance(current_pose, dict):
                _assign_local_event_fact(
                    local_step_outputs,
                    "current_pose",
                    {"pose": deepcopy(current_pose)},
                )
            continue

        resolved_params = _resolve_recovery_motion_params(
            params,
            part_name=part_name,
            observed_pose=observed_pose,
            step_outputs=local_step_outputs,
        )

        if not after_grasp:
            if primitive_name in {"move_pose", "move_cartesian"}:
                absolute_pose = _absolute_pose_from_params(resolved_params)
                if absolute_pose is not None:
                    current_pose = dict(absolute_pose)
                if _params_reference_observed_pose(params, part_name=part_name):
                    evidence["pickup_grounded"] = True
                elif _pose_matches_expected(
                    current_pose,
                    expected_pick_pose,
                    z_tol=_OBSERVED_PICKUP_Z_TOL_M,
                ):
                    evidence["pickup_grounded"] = True
                elif _observed_pose_match(current_pose, observed_pose):
                    evidence["pickup_grounded"] = True
            elif primitive_name == "move_relative" and current_pose is not None:
                dx = _float_or_none(resolved_params.get("dx")) or 0.0
                dy = _float_or_none(resolved_params.get("dy")) or 0.0
                dz = _float_or_none(resolved_params.get("dz")) or 0.0
                current_pose = {
                    "x": current_pose["x"] + dx,
                    "y": current_pose["y"] + dy,
                    "z": current_pose["z"] + dz,
                }
                if _pose_matches_expected(
                    current_pose,
                    expected_pick_pose,
                    z_tol=_OBSERVED_PICKUP_Z_TOL_M,
                ):
                    evidence["pickup_grounded"] = True
                if _observed_pose_match(current_pose, observed_pose):
                    evidence["pickup_grounded"] = True
            if primitive_name == "grasp_part" and step_targets_part:
                evidence["has_grasp"] = True
                after_grasp = True
                evidence["_carrier_resource_jid"] = resource_jid
                if _observed_pose_match(current_pose, observed_pose):
                    evidence["pickup_grounded"] = True
            continue

        if primitive_name == "release_part":
            evidence["has_release_after_grasp"] = True
            if placement_ready and _pose_matches_expected(
                current_pose,
                expected_target_pose,
                z_tol=_PLACE_TARGET_Z_TOL_M,
            ):
                evidence["has_placement_completion"] = True
            after_grasp = False
            evidence["_carrier_resource_jid"] = ""
            break

        if primitive_name in {"move_pose", "move_cartesian"}:
            absolute_pose = _absolute_pose_from_params(resolved_params)
            if absolute_pose is None:
                evidence["has_nonlocal_transport"] = True
                current_pose = None
                placement_ready = False
                continue
            current_pose = dict(absolute_pose)
            if isinstance(observed_pose, dict):
                obs_x = _float_or_none(observed_pose.get("x"))
                obs_y = _float_or_none(observed_pose.get("y"))
                if obs_x is not None and obs_y is not None:
                    if (
                        hypot(absolute_pose["x"] - obs_x, absolute_pose["y"] - obs_y)
                        >= _MEANINGFUL_POST_GRASP_XY_M
                    ):
                        evidence["has_nonlocal_transport"] = True
            else:
                evidence["has_nonlocal_transport"] = True
            if _pose_matches_expected(
                current_pose,
                expected_approach_pose,
                z_tol=_PLACE_APPROACH_Z_TOL_M,
            ):
                evidence["has_place_approach_witness"] = True
            if _pose_matches_expected(
                current_pose,
                expected_target_pose,
                z_tol=_PLACE_TARGET_Z_TOL_M,
            ):
                evidence["has_destination_witness"] = True
                placement_ready = True
            else:
                placement_ready = False
            continue

        if primitive_name == "move_relative":
            dx = _float_or_none(resolved_params.get("dx")) or 0.0
            dy = _float_or_none(resolved_params.get("dy")) or 0.0
            dz = _float_or_none(resolved_params.get("dz")) or 0.0
            cumulative_xy += hypot(dx, dy)
            if cumulative_xy >= _MEANINGFUL_POST_GRASP_XY_M:
                evidence["has_nonlocal_transport"] = True
            if current_pose is not None:
                current_pose = {
                    "x": current_pose["x"] + dx,
                    "y": current_pose["y"] + dy,
                    "z": current_pose["z"] + dz,
                }
                if _pose_matches_expected(
                    current_pose,
                    expected_approach_pose,
                    z_tol=_PLACE_APPROACH_Z_TOL_M,
                ):
                    evidence["has_place_approach_witness"] = True
                if _pose_matches_expected(
                    current_pose,
                    expected_target_pose,
                    z_tol=_PLACE_TARGET_Z_TOL_M,
                ):
                    evidence["has_destination_witness"] = True
                    if dz < 0.0 or evidence["has_place_approach_witness"]:
                        placement_ready = True
                    continue
            placement_ready = False

    evidence["_after_grasp"] = after_grasp
    evidence["_cumulative_xy"] = cumulative_xy
    evidence["_current_pose"] = current_pose
    evidence["_placement_ready"] = placement_ready


def _check_under_modeled_part_recovery_semantics(
    program: RepairProgram,
    *,
    active_obligations: list[dict[str, Any]] | None,
    part_states: dict[str, dict[str, Any]] | None,
    resource_snapshots: dict[str, dict[str, Any]] | None,
    product_geometry: dict[str, Any] | None,
    observation_store: dict[str, Any] | None,
) -> list[str]:
    """Reject observed-part repairs that never leave the pickup neighborhood."""
    if not active_obligations or not part_states:
        return []

    target_parts: dict[str, dict[str, str]] = {}
    for obligation in active_obligations or []:
        if str(obligation.get("type") or "").strip().lower() == "safety":
            continue
        if not bool(obligation.get("must_satisfy_before_resume", True)):
            continue
        entity_kind = str(obligation.get("entity_kind") or "").strip()
        entity = str(obligation.get("entity") or "").strip()
        field = str(obligation.get("field") or "").strip()
        expected = str(obligation.get("expected") or "").strip()
        if entity_kind != "part" or field not in {"location", "state"} or not entity or not expected:
            continue
        info = dict((part_states or {}).get(entity) or {})
        if not info:
            continue
        if not part_state_requires_external_localization(info.get("state")):
            continue
        if part_observation_status(info) != "observed":
            continue
        current_value = info.get(field)
        if current_value is not None and str(current_value) == expected:
            continue
        target_parts.setdefault(entity, {})[field] = expected

    if not target_parts:
        return []

    fn_map = {fn.name: fn for fn in program.function_defs}
    errors: list[str] = []

    for part_name, targets in target_parts.items():
        observed_pose = dict((part_states or {}).get(part_name, {}).get("observed_pose") or {})
        expected_location = str(targets.get("location") or "").strip()
        expected_state = str(targets.get("state") or "").strip()
        pick_preview = _preview_pick_targets_for_observed_part(
            part_name=part_name,
            expected_location=expected_location,
            observed_pose=observed_pose,
            product_geometry=product_geometry,
            resource_snapshots=resource_snapshots,
        )
        place_preview = _preview_place_targets_for_observed_part(
            part_name=part_name,
            expected_location=expected_location,
            observed_pose=observed_pose,
            product_geometry=product_geometry,
            resource_snapshots=resource_snapshots,
        )
        claim_bits: list[str] = []
        if expected_location:
            claim_bits.append(f"{part_name}.location = {expected_location}")
        if expected_state:
            claim_bits.append(f"{part_name}.state = {expected_state}")
        claim_text = " and ".join(claim_bits) if claim_bits else f"{part_name} recovery"

        evidence = _new_part_recovery_evidence()

        for step in program.steps:
            if step.kind == RepairStepKind.RESUME_SUFFIX:
                break
            if step.kind != RepairStepKind.CALL_FUNCTION:
                continue

            fn_name = str(step.payload.get("function_name") or "").strip()
            resource_jid = str(step.payload.get("resource_jid") or "").strip()
            args = dict(step.payload.get("args") or {})
            if fn_name in {"place_approach", "place_insert"} and _args_match_expected_destination(
                args, expected_location,
            ):
                if fn_name == "place_approach":
                    evidence["has_place_approach_witness"] = True
                evidence["has_destination_witness"] = True
                evidence["has_nonlocal_transport"] = True
                if fn_name == "place_insert":
                    evidence["has_release_after_grasp"] = True
                    evidence["has_placement_completion"] = True

            fn_def = fn_map.get(fn_name)
            if fn_def is None:
                continue
            _advance_synthesized_part_recovery(
                evidence,
                fn_def=fn_def,
                resource_jid=resource_jid,
                part_name=part_name,
                observed_pose=observed_pose,
                pick_preview=pick_preview,
                place_preview=place_preview,
                observation_store=observation_store,
            )

        if evidence["has_grasp"] and not evidence["pickup_grounded"]:
            errors.append(
                f"repair for '{part_name}' must ground the pickup from the trusted observed pose before grasp_part"
            )
            continue

        if not (expected_location or expected_state):
            continue

        if evidence["has_grasp"] and not evidence["has_release_after_grasp"] and not evidence["has_destination_witness"]:
            errors.append(
                f"repair for '{part_name}' grasps the part but never completes a placement step consistent with {claim_text}"
            )
            continue

        if place_preview is None and expected_location:
            errors.append(
                f"repair for '{part_name}' lacks a geometry-derived placement preview for {expected_location}; destination-grounded release cannot be validated"
            )
            continue

        if place_preview is not None and evidence["has_grasp"] and not evidence["has_place_approach_witness"]:
            if evidence.get("_ungrounded_place_helper"):
                errors.append(
                    f"repair for '{part_name}' must ground compute_place_targets with destination geometry or destination_location for {expected_location}; pick_ctx or part_name alone is not enough"
                )
                continue
            errors.append(
                f"repair for '{part_name}' never reaches the computed place-approach pose for {expected_location}"
            )
            continue

        if place_preview is not None and evidence["has_grasp"] and not evidence["has_destination_witness"]:
            errors.append(
                f"repair for '{part_name}' transports the part after pickup but never reaches the computed placement target for {expected_location}"
            )
            continue

        if place_preview is not None and evidence["has_release_after_grasp"] and not evidence["has_placement_completion"]:
            errors.append(
                f"repair for '{part_name}' releases the part away from the computed placement target for {expected_location}; this does not establish {claim_text}"
            )
            continue

        if evidence["has_release_after_grasp"] and not (
            evidence["has_destination_witness"] or evidence["has_nonlocal_transport"]
        ):
            errors.append(
                f"repair for '{part_name}' picks from the observed pose but only performs local motion before release; this does not establish {claim_text}"
            )

    return errors


def _released_carried_parts_before_resume(
    program: RepairProgram,
    *,
    carried_parts: set[str],
) -> dict[str, list[SynthesizedTaskFn]]:
    released: dict[str, list[SynthesizedTaskFn]] = {
        part_name: [] for part_name in sorted(carried_parts)
    }
    fn_map = {fn.name: fn for fn in program.function_defs}
    for step in program.steps:
        if step.kind == RepairStepKind.RESUME_SUFFIX:
            break
        if step.kind != RepairStepKind.CALL_FUNCTION:
            continue
        fn_name = str(step.payload.get("function_name") or "").strip()
        fn_def = fn_map.get(fn_name)
        if fn_def is None:
            continue
        primitive_names = {
            _motion_primitive_name(item.get("primitive"))
            for item in (fn_def.primitive_program or [])
            if isinstance(item, dict)
        }
        if "release_part" not in primitive_names:
            continue
        for part_name in carried_parts:
            if _function_targets_part(fn_def, part_name=part_name):
                released.setdefault(part_name, []).append(fn_def)
    return released


def _check_blocker_staging_witnesses(
    program: RepairProgram,
    *,
    resource_snapshots: dict[str, dict[str, Any]] | None,
    grounded_environment_facts: dict[str, Any] | None,
) -> list[str]:
    resource_snapshots = dict(resource_snapshots or {})
    grounded_environment_facts = dict(grounded_environment_facts or {})
    carried_parts = {
        str(snapshot.get("held_part") or "").strip()
        for snapshot in resource_snapshots.values()
        if str(snapshot.get("held_part") or "").strip()
    }
    if not carried_parts:
        return []

    staging_destinations = {
        str(name).strip(): str(value).strip()
        for name, value in dict(grounded_environment_facts.get("staging_destinations") or {}).items()
        if str(name).strip() and str(value).strip()
    }
    released_by_part = _released_carried_parts_before_resume(
        program,
        carried_parts=carried_parts,
    )
    errors: list[str] = []

    for part_name in sorted(carried_parts):
        fn_defs = list(released_by_part.get(part_name) or [])
        if not fn_defs:
            continue
        staging_destination = str(staging_destinations.get(part_name) or "").strip()
        if not staging_destination:
            errors.append(
                f"staging of '{part_name}' requires an explicit non-assembly staging destination before release_part"
            )
            continue
        if "assembly_board" in staging_destination.lower():
            errors.append(
                f"staging of '{part_name}' must use a non-assembly destination, not {staging_destination}"
            )
            continue

        destination_satisfied = False
        descend_satisfied = False
        retreat_satisfied = False

        for fn_def in fn_defs:
            primitive_program = list(fn_def.primitive_program or [])
            anchor_seen = False
            descend_seen = False
            release_index = -1
            for index, primitive_step in enumerate(primitive_program):
                primitive_name = _motion_primitive_name(primitive_step.get("primitive"))
                if (
                    not anchor_seen
                    and _primitive_references_location(
                        primitive_step,
                        location_token=staging_destination,
                    )
                ):
                    anchor_seen = True
                    destination_satisfied = True
                if anchor_seen and _primitive_is_descend_witness(primitive_step):
                    descend_seen = True
                if primitive_name == "release_part":
                    release_index = index
                    if anchor_seen and descend_seen:
                        descend_satisfied = True
                    break

            if release_index >= 0:
                post_release = primitive_program[release_index + 1 :]
                if any(_primitive_is_motion(step) for step in post_release):
                    retreat_satisfied = True

        if not destination_satisfied:
            errors.append(
                f"staging of '{part_name}' must move to explicit staging destination '{staging_destination}' before release_part"
            )
            continue
        if not descend_satisfied:
            errors.append(
                f"staging of '{part_name}' must descend to the staging destination '{staging_destination}' before release_part"
            )
            continue
        if not retreat_satisfied:
            errors.append(
                f"staging of '{part_name}' must retreat after release_part at staging destination '{staging_destination}'"
            )

    return errors


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

def _classify_risk(
    program: RepairProgram,
    recovery_library: Any | None,
) -> RiskLevel:
    """Compute the authoritative risk level.

    Low-risk (single definition) when ALL of:
    (a) every synthesized function exactly matches a library entry
        with ``runtime_success_count >= 1``
    (b) mutations are insert-only
    (c) no safety-critical obligations are touched

    Everything else is high-risk.
    """
    # (a) Check all functions are runtime-proven in library.
    if recovery_library is not None and program.function_defs:
        for fn_def in program.function_defs:
            fingerprint = compute_primitive_fingerprint(fn_def.primitive_program)
            sig_hash = compute_signature_hash(
                resource_profile_id="",  # TODO: resolve from context
                primitive_fingerprint=fingerprint,
                touched_entities=[],
                preconditions=fn_def.preconditions,
                expected_post_state=fn_def.expected_post_state,
                catalog_version="",
            )
            if not recovery_library.is_runtime_proven(sig_hash):
                return RiskLevel.HIGH
    elif program.function_defs:
        # No library available — all new functions are high-risk.
        return RiskLevel.HIGH

    # (b) Check mutations are insert-only.
    for step in program.steps:
        if step.kind == RepairStepKind.TASK_MUTATION:
            mt = step.payload.get("mutation_type", "")
            if mt not in ("insert",):
                return RiskLevel.HIGH

    # (c) For now, all repairs with mutations beyond insert are high-risk.
    # Safety-critical obligation detection can be refined later.

    return RiskLevel.LOW


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_function_resource(
    fn_def: SynthesizedTaskFn,
    program: RepairProgram,
) -> str:
    """Find the resource JID for a function from the program's call steps."""
    for step in program.steps:
        if step.kind == RepairStepKind.CALL_FUNCTION:
            if step.payload.get("function_name") == fn_def.name:
                return str(step.payload.get("resource_jid", "")).strip()
    # Fallback: try resource_constraints.
    return str(fn_def.resource_constraints.get("resource_jid", "")).strip()


def _find_function_def(
    name: str,
    program: RepairProgram,
) -> SynthesizedTaskFn | None:
    """Find a function definition by name."""
    for fn in program.function_defs:
        if fn.name == name:
            return fn
    return None


def _resource_short_name(resource_jid: str) -> str:
    return str(resource_jid or "").split("@", 1)[0].strip().lower()


def _catalog_task_rows_for_action(
    available_task_actions: list[dict[str, Any]] | None,
    *,
    resource_jid: str,
    function_name: str,
) -> list[dict[str, Any]]:
    """Return task-catalog rows matching a direct nominal action call."""
    target_name = str(function_name or "").strip()
    target_jid = str(resource_jid or "").strip()
    if not target_name or not target_jid:
        return []

    target_owner = _resource_short_name(target_jid)
    rows: list[dict[str, Any]] = []
    for row in available_task_actions or []:
        if not isinstance(row, dict):
            continue
        row_name = str(
            row.get("function")
            or row.get("function_name")
            or row.get("name")
            or ""
        ).strip()
        if row_name != target_name:
            continue
        row_resource_jid = str(row.get("resource_jid") or "").strip()
        if row_resource_jid:
            if row_resource_jid != target_jid:
                continue
        else:
            row_owner = _resource_short_name(
                str(row.get("function_owner_agent") or row.get("owner") or "").strip()
            )
            if row_owner and row_owner != target_owner:
                continue
        rows.append(row)
    return rows


def _select_catalog_task_row(
    available_task_actions: list[dict[str, Any]] | None,
    *,
    resource_jid: str,
    function_name: str,
) -> dict[str, Any] | None:
    rows = _catalog_task_rows_for_action(
        available_task_actions,
        resource_jid=resource_jid,
        function_name=function_name,
    )
    if not rows:
        return None
    return rows[0]


def _make_rejections(
    layer: str,
    check: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    """Convert error strings into structured rejection dicts."""
    return [
        {"layer": layer, "check": check, "message": err}
        for err in errors
    ]


def _apply_patches_to_nodes(
    patches: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply patches to a node list (projected copy, not live)."""
    node_map = {n["id"]: deepcopy(n) for n in nodes if isinstance(n, dict) and n.get("id")}

    for patch in patches:
        tid = patch.get("id")
        if not tid:
            continue
        if patch.get("delete"):
            node_map.pop(tid, None)
            for node in node_map.values():
                preds = node.get("predecessors") or []
                if tid in preds:
                    preds.remove(tid)
                succs = node.get("successors") or []
                if tid in succs:
                    succs.remove(tid)
        else:
            if tid in node_map:
                node_map[tid].update(patch)
            else:
                node_map[tid] = dict(patch)

    return list(node_map.values())

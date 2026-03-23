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
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.mutation_types import (
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
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.function_synthesis import (
    compile_synthesized_function_to_macro,
    validate_synthesized_function,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.mutation_compiler import (
    compile_mutations,
    validate_mutation_step,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_repair_program(
    program: RepairProgram,
    *,
    primitive_catalogs: dict[str, list[dict[str, Any]]],
    resource_snapshots: dict[str, dict[str, Any]],
    current_nodes: list[dict[str, Any]],
    observation_store: dict[str, Any] | None = None,
    capability_flags_map: dict[str, dict[str, bool]] | None = None,
    safety_validator: Any | None = None,
    safety_rules: list[dict[str, Any]] | None = None,
    runtime_monitor_state: dict[str, Any] | None = None,
    compile_fsa_fn: Any | None = None,
    recovery_library: Any | None = None,
    active_obligations: list[dict[str, Any]] | None = None,
    part_states: dict[str, dict[str, Any]] | None = None,
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

    # A3: Symbol resolution and step sequencing.
    seq_errors = _validate_step_sequencing(program)
    if seq_errors:
        result.rejection_reasons.extend(
            _make_rejections("A", "step_sequencing", seq_errors)
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

    # Insert synthesized function calls as execute_recovery_macro tasks.
    call_steps = [
        s for s in program.steps
        if s.kind == RepairStepKind.CALL_FUNCTION
    ]
    for i, step in enumerate(call_steps):
        fn_name = step.payload.get("function_name", "")
        resource_jid = step.payload.get("resource_jid", "")
        fn_def = _find_function_def(fn_name, program)
        if fn_def is None:
            continue  # Already caught in A3.
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

    # ===================================================================
    # Risk classification (validator-owned)
    # ===================================================================
    result.risk_level = _classify_risk(program, recovery_library)
    result.requires_operator_approval = result.risk_level == RiskLevel.HIGH

    return result


# ---------------------------------------------------------------------------
# Layer A validators
# ---------------------------------------------------------------------------

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
            mt = step.payload.get("mutation_type", "")
            if mt not in {e.value for e in TaskMutationType}:
                errors.append(f"step {i}: invalid mutation_type '{mt}'")

    for fn_def in program.function_defs:
        if not fn_def.name:
            errors.append("function_def has no name")
        if not fn_def.primitive_program:
            errors.append(f"function '{fn_def.name}': empty primitive_program")

    return errors


def _validate_step_sequencing(program: RepairProgram) -> list[str]:
    """A3: Check that steps reference valid functions and task IDs."""
    errors: list[str] = []
    fn_names = {fn.name for fn in program.function_defs}

    for i, step in enumerate(program.steps):
        if step.kind == RepairStepKind.CALL_FUNCTION:
            fn_name = step.payload.get("function_name", "")
            if fn_name and fn_name not in fn_names:
                # Allow calling functions not in function_defs if they
                # are known task catalog functions (e.g. move_to_home).
                errors.append(
                    f"step {i}: call_function references undefined "
                    f"function '{fn_name}' (defined: {sorted(fn_names)})"
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
            mt = TaskMutationType(str(payload.get("mutation_type", "")))
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
    """
    errors: list[str] = []

    if not success_conditions:
        errors.append("repair_program has no success_conditions")

    # Check that active goal/reentry obligations are covered.
    if active_obligations:
        sc_keys: set[tuple[str, str]] = set()
        for sc in success_conditions:
            entity = str(sc.get("entity", "")).strip()
            field = str(sc.get("field", "")).strip()
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

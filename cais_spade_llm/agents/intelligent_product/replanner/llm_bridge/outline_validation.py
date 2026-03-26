from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.recovery_context_builder import (
    build_grounding_assessment,
)


_OUTLINE_PHASE_GROUP = {
    "resolve_safety": 0,
    "restore_capability": 0,
    "free_executor": 0,
    "recover_entities": 1,
    "restore_resume_entry": 1,
    "adapt_goals": 1,
    "replace_suffix": 2,
    "resume_modeled_suffix": 2,
}

_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class OutlineValidationResult:
    errors: list[str] = field(default_factory=list)
    phase_signature: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    outline_actions: list[dict[str, Any]] = field(default_factory=list)
    terminal_phase: str = ""
    unresolved_bridge_goal_parts: list[str] = field(default_factory=list)
    closes_bridge: bool = False

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "phase_signature": [
                {"phase_type": phase_type, "target_entities": list(targets)}
                for phase_type, targets in self.phase_signature
            ],
            "outline_actions": deepcopy(self.outline_actions),
            "terminal_phase": self.terminal_phase,
            "unresolved_bridge_goal_parts": list(self.unresolved_bridge_goal_parts),
            "closes_bridge": self.closes_bridge,
        }


def outline_signature(reasoning: dict[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    signature: list[tuple[str, tuple[str, ...]]] = []
    for phase in materialize_abstract_repair_order(reasoning):
        phase_type = str(phase.get("phase_type") or "").strip()
        target_entities = tuple(
            sorted(
                str(entity).strip()
                for entity in (phase.get("target_entities") or [])
                if str(entity).strip()
            )
        )
        if phase_type:
            signature.append((phase_type, target_entities))
    return signature


def _outline_phase_targets(phase: dict[str, Any]) -> set[str]:
    return {
        str(entity).strip()
        for entity in (phase.get("target_entities") or [])
        if str(entity).strip()
    }


def _slug_token(value: Any) -> str:
    token = _NON_WORD_RE.sub("_", str(value or "").strip().lower()).strip("_")
    return token


def _fallback_outline_action_id(
    *,
    phase_type: str,
    index: int,
    target_entities: list[str],
) -> str:
    target_hints = [
        _slug_token(entity)
        for entity in target_entities[:2]
        if _slug_token(entity)
    ]
    parts = [f"a{index}", _slug_token(phase_type)]
    parts.extend(target_hints)
    return "_".join(part for part in parts if part)


def materialize_outline_actions(reasoning: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized named task-level actions for the outline stage.

    If the model provides ``outline_actions``, normalize and keep them.
    Otherwise derive stable fallback action ids from ``abstract_repair_order``.
    """
    normalized_actions: list[dict[str, Any]] = []
    provided_actions = reasoning.get("outline_actions")
    if isinstance(provided_actions, list) and provided_actions:
        for index, action in enumerate(provided_actions, start=1):
            if not isinstance(action, dict):
                continue
            phase_type = str(action.get("phase_type") or "").strip()
            target_entities = [
                str(entity).strip()
                for entity in (action.get("target_entities") or [])
                if str(entity).strip()
            ]
            action_id = str(action.get("action_id") or "").strip()
            if not action_id:
                action_id = _fallback_outline_action_id(
                    phase_type=phase_type or f"outline_action_{index}",
                    index=index,
                    target_entities=target_entities,
                )
            normalized_actions.append({
                "action_id": action_id,
                "phase_type": phase_type,
                "objective": str(action.get("objective") or "").strip(),
                "target_entities": target_entities,
                "advances_obligations": [
                    str(item).strip()
                    for item in (action.get("advances_obligations") or [])
                    if str(item).strip()
                ],
                "must_complete_before": [
                    str(item).strip()
                    for item in (action.get("must_complete_before") or [])
                    if str(item).strip()
                ],
            })
        if normalized_actions:
            return normalized_actions

    abstract_repair_order = list(reasoning.get("abstract_repair_order") or [])
    derived: list[dict[str, Any]] = []
    for index, phase in enumerate(abstract_repair_order, start=1):
        if not isinstance(phase, dict):
            continue
        phase_type = str(phase.get("phase_type") or "").strip()
        target_entities = [
            str(entity).strip()
            for entity in (phase.get("target_entities") or [])
            if str(entity).strip()
        ]
        derived.append({
            "action_id": _fallback_outline_action_id(
                phase_type=phase_type or f"outline_action_{index}",
                index=index,
                target_entities=target_entities,
            ),
            "phase_type": phase_type,
            "objective": str(phase.get("objective") or "").strip(),
            "target_entities": target_entities,
            "advances_obligations": [
                str(item).strip()
                for item in (phase.get("advances_obligations") or [])
                if str(item).strip()
            ],
            "must_complete_before": [],
        })
    for index, action in enumerate(derived[:-1]):
        next_action_id = str(derived[index + 1].get("action_id") or "").strip()
        if next_action_id:
            action["must_complete_before"] = [next_action_id]
    return derived


def materialize_abstract_repair_order(reasoning: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized abstract phase blocks for the outline stage.

    When the model omits ``abstract_repair_order`` in ``repair_outline``, derive
    it from the ordered ``outline_actions``. If adjacent abstract rows share the
    same phase_type, merge them into one phase block.
    """
    normalized: list[dict[str, Any]] = []

    def _append_phase(
        *,
        phase_type: str,
        objective: str,
        target_entities: list[str],
        advances_obligations: list[str],
    ) -> None:
        if not phase_type:
            return
        targets = [
            str(entity).strip()
            for entity in target_entities
            if str(entity).strip()
        ]
        advances = [
            str(item).strip()
            for item in advances_obligations
            if str(item).strip()
        ]
        if normalized and str(normalized[-1].get("phase_type") or "").strip() == phase_type:
            merged_targets = sorted(set(normalized[-1].get("target_entities") or []) | set(targets))
            normalized[-1]["target_entities"] = merged_targets
            merged_advances = list(normalized[-1].get("advances_obligations") or [])
            for item in advances:
                if item not in merged_advances:
                    merged_advances.append(item)
            normalized[-1]["advances_obligations"] = merged_advances
            if objective and not str(normalized[-1].get("objective") or "").strip():
                normalized[-1]["objective"] = objective
            return

        normalized.append({
            "phase_type": phase_type,
            "objective": objective,
            "target_entities": targets,
            "advances_obligations": advances,
        })

    provided = reasoning.get("abstract_repair_order")
    if isinstance(provided, list) and provided:
        for phase in provided:
            if not isinstance(phase, dict):
                continue
            _append_phase(
                phase_type=str(phase.get("phase_type") or "").strip(),
                objective=str(phase.get("objective") or "").strip(),
                target_entities=list(phase.get("target_entities") or []),
                advances_obligations=list(phase.get("advances_obligations") or []),
            )
        if normalized:
            return normalized

    for action in materialize_outline_actions(reasoning):
        _append_phase(
            phase_type=str(action.get("phase_type") or "").strip(),
            objective=str(action.get("objective") or "").strip(),
            target_entities=list(action.get("target_entities") or []),
            advances_obligations=list(action.get("advances_obligations") or []),
        )
    return normalized


def _outline_actions_refine_abstract_order(
    abstract_repair_order: list[dict[str, Any]],
    outline_actions: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Return whether named outline actions refine the abstract phase order.

    One abstract phase may refine into multiple consecutive outline actions as
    long as the action phase blocks stay in the same order and each abstract
    phase's target entities are covered by its corresponding action block.
    """
    phase_rows = materialize_abstract_repair_order(
        {"abstract_repair_order": abstract_repair_order}
    )
    action_rows = [
        action
        for action in outline_actions
        if isinstance(action, dict) and str(action.get("phase_type") or "").strip()
    ]
    if not phase_rows or not action_rows:
        return False, (
            "repair_outline.reasoning.outline_actions must refine the abstract_repair_order "
            "with at least one named action per phase"
        )

    action_blocks: list[tuple[str, set[str]]] = []
    for action in action_rows:
        phase_type = str(action.get("phase_type") or "").strip()
        target_entities = {
            str(entity).strip()
            for entity in (action.get("target_entities") or [])
            if str(entity).strip()
        }
        if action_blocks and action_blocks[-1][0] == phase_type:
            action_blocks[-1][1].update(target_entities)
        else:
            action_blocks.append((phase_type, set(target_entities)))

    if len(action_blocks) != len(phase_rows):
        return False, (
            "repair_outline.reasoning.outline_actions must refine abstract_repair_order "
            "in ordered phase blocks; one phase may expand into multiple consecutive actions, "
            "but the phase sequence itself cannot change"
        )

    for index, phase in enumerate(phase_rows, start=1):
        phase_type = str(phase.get("phase_type") or "").strip()
        phase_targets = {
            str(entity).strip()
            for entity in (phase.get("target_entities") or [])
            if str(entity).strip()
        }
        action_phase_type, action_targets = action_blocks[index - 1]
        if action_phase_type != phase_type:
            return False, (
                "repair_outline.reasoning.outline_actions must refine abstract_repair_order "
                f"in order; phase block {index} is '{action_phase_type}' but abstract phase "
                f"{index} is '{phase_type}'"
            )
        if phase_targets and not phase_targets.issubset(action_targets):
            return False, (
                "repair_outline.reasoning.outline_actions must cover each abstract phase's "
                f"target_entities; phase '{phase_type}' is missing "
                + ", ".join(sorted(phase_targets - action_targets))
            )
    return True, ""


def _outline_obligation_is_currently_satisfied(
    recovery_context: Any,
    obligation: dict[str, Any],
) -> bool:
    entity = str(obligation.get("entity") or "").strip()
    field = str(obligation.get("field") or "").strip()
    expected = obligation.get("expected")
    if not entity or not field:
        return False

    if entity in (recovery_context.part_states or {}):
        part_entry = dict((recovery_context.part_states or {}).get(entity) or {})
        actual = part_entry.get(field)
        return actual == expected

    resource_entry = dict((recovery_context.resource_snapshots or {}).get(entity) or {})
    if resource_entry:
        actual = resource_entry.get(field)
        if actual is None and isinstance(resource_entry.get("resource_core"), dict):
            actual = dict(resource_entry.get("resource_core") or {}).get(field)
        return actual == expected

    return False


def _pose_within_workspace_bounds(pose: dict[str, Any], bounds: dict[str, Any]) -> bool:
    for axis in ("x", "y", "z"):
        val = pose.get(axis)
        if val is None:
            continue
        lo = bounds.get(f"{axis}_min_m")
        hi = bounds.get(f"{axis}_max_m")
        coord = float(val)
        if lo is not None and coord < float(lo):
            return False
        if hi is not None and coord > float(hi):
            return False
    return True


def _grounded_part_executor_bindings(recovery_context: Any) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    grounded_facts = dict(getattr(recovery_context, "grounded_environment_facts", {}) or {})
    observed_sources = dict(grounded_facts.get("observed_part_sources") or {})
    resource_snapshots = dict(getattr(recovery_context, "resource_snapshots", {}) or {})

    for part_name, row in observed_sources.items():
        pose = dict(row.get("observed_pose") or {})
        if not pose:
            continue
        reachable_resources = sorted(
            resource_jid
            for resource_jid, snapshot in resource_snapshots.items()
            if isinstance(snapshot.get("workspace_bounds"), dict)
            and _pose_within_workspace_bounds(pose, dict(snapshot.get("workspace_bounds") or {}))
        )
        if len(reachable_resources) != 1:
            continue
        bindings[str(part_name).strip()] = {
            "bound_executor": reachable_resources[0],
            "binding_basis": "trusted_observed_pose",
        }

    for resource_jid, snapshot in resource_snapshots.items():
        held_part = str(snapshot.get("held_part") or "").strip()
        if not held_part:
            continue
        bindings[held_part] = {
            "bound_executor": str(resource_jid).strip(),
            "binding_basis": "held_part",
        }

    return bindings


def _is_explicit_non_assembly_location_token(
    value: Any,
    *,
    known_parts: set[str],
    known_resources: set[str],
) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    lower = token.lower()
    if lower in {"unknown", "none", "null"}:
        return False
    if token in known_parts or token in known_resources:
        return False
    if "assembly_board" in lower:
        return False
    if "_gripper" in lower or lower.endswith("gripper"):
        return False
    return True


def _check_blocker_staging_destinations(
    outline_actions: list[dict[str, Any]],
    *,
    recovery_context: Any,
) -> list[str]:
    grounded_facts = dict(getattr(recovery_context, "grounded_environment_facts", {}) or {})
    staging_destinations = {
        str(name).strip(): str(value).strip()
        for name, value in dict(grounded_facts.get("staging_destinations") or {}).items()
        if str(name).strip() and str(value).strip()
    }
    resource_snapshots = dict(getattr(recovery_context, "resource_snapshots", {}) or {})
    carried_parts = {
        str(snapshot.get("held_part") or "").strip()
        for snapshot in resource_snapshots.values()
        if str(snapshot.get("held_part") or "").strip()
    }
    if not carried_parts:
        return []

    known_parts = {
        str(name).strip()
        for name in (getattr(recovery_context, "part_states", {}) or {})
        if str(name).strip()
    }
    known_resources = {
        str(jid).strip()
        for jid in resource_snapshots
        if str(jid).strip()
    }
    errors: list[str] = []

    for action in outline_actions:
        phase_type = str(action.get("phase_type") or "").strip()
        if phase_type != "free_executor":
            continue
        action_id = str(action.get("action_id") or "").strip() or "<unnamed_action>"
        targets = {
            str(entity).strip()
            for entity in (action.get("target_entities") or [])
            if str(entity).strip()
        }
        blocker_parts = sorted(targets & carried_parts)
        if not blocker_parts:
            continue

        explicit_target_destinations = {
            token for token in targets
            if _is_explicit_non_assembly_location_token(
                token,
                known_parts=known_parts,
                known_resources=known_resources,
            )
        }
        for part_name in blocker_parts:
            fact_destination = str(staging_destinations.get(part_name) or "").strip()
            if fact_destination:
                continue
            if explicit_target_destinations:
                continue
            errors.append(
                f"repair_outline action '{action_id}' targets blocker part '{part_name}' in phase "
                f"'free_executor' but no explicit non-assembly staging destination is grounded"
            )

    return errors


def _check_grounded_executor_switches(
    outline_actions: list[dict[str, Any]],
    *,
    recovery_context: Any,
) -> list[str]:
    bindings = _grounded_part_executor_bindings(recovery_context)
    if not bindings:
        return []

    resource_jids = {
        str(jid).strip()
        for jid in (getattr(recovery_context, "resource_snapshots", {}) or {})
        if str(jid).strip()
    }
    manipulation_phases = {"free_executor", "recover_entities", "restore_resume_entry"}
    errors: list[str] = []

    for part_name, binding in sorted(bindings.items()):
        bound_executor = str(binding.get("bound_executor") or "").strip()
        if not bound_executor:
            continue
        assignments: list[tuple[str, str, list[str]]] = []
        assigned_resources: set[str] = set()
        for action in outline_actions:
            phase_type = str(action.get("phase_type") or "").strip()
            if phase_type not in manipulation_phases:
                continue
            targets = {
                str(entity).strip()
                for entity in (action.get("target_entities") or [])
                if str(entity).strip()
            }
            if part_name not in targets:
                continue
            action_resources = sorted(targets & resource_jids)
            if not action_resources:
                continue
            assignments.append((
                str(action.get("action_id") or "").strip() or "<unnamed_action>",
                phase_type,
                action_resources,
            ))
            assigned_resources.update(action_resources)

        foreign_resources = sorted(
            resource_jid for resource_jid in assigned_resources
            if resource_jid != bound_executor
        )
        if not foreign_resources:
            continue

        if str(binding.get("binding_basis") or "").strip() == "held_part":
            basis_text = f"it is currently held by {bound_executor}"
        else:
            basis_text = (
                "its trusted observed pose is currently reachable only by "
                f"{bound_executor}"
            )
        action_refs = ", ".join(
            f"{action_id}({phase_type}:{'/'.join(resources)})"
            for action_id, phase_type, resources in assignments
            if any(resource in foreign_resources for resource in resources)
        )
        errors.append(
            f"repair_outline assigns grounded part '{part_name}' to executor(s) "
            + ", ".join(foreign_resources)
            + f" via {action_refs or 'manipulation phases'}, but {basis_text}; "
              "cross-executor reassignment is not grounded in the current bridge model"
        )

    return errors


def validate_repair_outline(
    parsed: dict[str, Any],
    *,
    recovery_context: Any,
) -> OutlineValidationResult:
    errors: list[str] = []
    reasoning = dict(parsed.get("reasoning") or {})
    if not reasoning:
        return OutlineValidationResult(
            errors=["repair_outline is missing the required reasoning object"],
        )

    outline_actions = materialize_outline_actions(reasoning)
    reasoning["outline_actions"] = deepcopy(outline_actions)
    abstract_repair_order = materialize_abstract_repair_order(reasoning)
    reasoning["abstract_repair_order"] = deepcopy(abstract_repair_order)
    if not abstract_repair_order:
        return OutlineValidationResult(
            errors=errors + [
                "repair_outline must provide either outline_actions or abstract_repair_order"
            ],
        )

    signature = outline_signature(reasoning)
    last_group = -1
    terminal_phase_seen = False
    first_non_safety_phase = ""
    previous_phase_type = ""
    terminal_phase = ""
    for index, phase in enumerate(abstract_repair_order, start=1):
        if not isinstance(phase, dict):
            errors.append(
                f"repair_outline.reasoning.abstract_repair_order[{index}] must be an object"
            )
            continue
        phase_type = str(phase.get("phase_type") or "").strip()
        if phase_type not in _OUTLINE_PHASE_GROUP:
            errors.append(
                f"repair_outline.reasoning.abstract_repair_order[{index}] has invalid phase_type '{phase_type}'"
            )
            continue
        if phase_type != "resolve_safety" and not first_non_safety_phase:
            first_non_safety_phase = phase_type
        phase_group = _OUTLINE_PHASE_GROUP[phase_type]
        if phase_group < last_group:
            errors.append(
                "repair_outline.reasoning.abstract_repair_order must progress from preparatory phases "
                f"toward recovery/adaptation and then terminal resume/replace phases; "
                f"phase_type '{phase_type}' at row {index} cannot appear after '{previous_phase_type}'"
            )
        last_group = max(last_group, phase_group)
        previous_phase_type = phase_type
        if phase_type in {"resume_modeled_suffix", "replace_suffix"}:
            terminal_phase = phase_type
            if index != len(abstract_repair_order):
                errors.append(
                    f"repair_outline.reasoning.abstract_repair_order[{index}] uses terminal phase_type '{phase_type}' before the end of the outline"
                )
            terminal_phase_seen = True

    if not terminal_phase_seen:
        errors.append(
            "repair_outline.reasoning.abstract_repair_order must end with phase_type='resume_modeled_suffix' or 'replace_suffix'"
        )

    if not outline_actions:
        errors.append(
            "repair_outline.reasoning.outline_actions must name the task-level recovery actions"
        )
    else:
        seen_action_ids: set[str] = set()
        for action in outline_actions:
            action_id = str(action.get("action_id") or "").strip()
            if not action_id:
                errors.append(
                    "repair_outline.reasoning.outline_actions entries must have non-empty action_id values"
                )
                continue
            if action_id in seen_action_ids:
                errors.append(
                    f"repair_outline.reasoning.outline_actions uses duplicate action_id '{action_id}'"
                )
                continue
            seen_action_ids.add(action_id)
        refines_outline, refine_error = _outline_actions_refine_abstract_order(
            abstract_repair_order,
            outline_actions,
        )
        if not refines_outline:
            errors.append(refine_error)

    grounding = build_grounding_assessment(recovery_context)
    executor_first_parts = {
        str(name).strip()
        for name in (grounding.get("executor_first_parts") or [])
        if str(name).strip()
    }
    if executor_first_parts and first_non_safety_phase not in {
        "restore_capability",
        "free_executor",
        "adapt_goals",
        "replace_suffix",
    }:
        errors.append(
            "repair_outline must address executor feasibility before entity recovery when the current state exposes an executor-first blocker"
        )

    unresolved_bridge_goal_parts: set[str] = set()
    for obligation in (recovery_context.active_obligations or []):
        if not isinstance(obligation, dict):
            continue
        if not bool(obligation.get("must_satisfy_before_resume", False)):
            continue
        if str(obligation.get("obligation_class") or "").strip() != "bridge_goal":
            continue
        entity = str(obligation.get("entity") or "").strip()
        field = str(obligation.get("field") or "").strip()
        if not entity or field not in {"state", "location"}:
            continue
        if entity not in (recovery_context.part_states or {}):
            continue
        if _outline_obligation_is_currently_satisfied(recovery_context, obligation):
            continue
        unresolved_bridge_goal_parts.add(entity)

    if terminal_phase == "resume_modeled_suffix" and unresolved_bridge_goal_parts:
        recovered_before_resume: set[str] = set()
        for phase in abstract_repair_order[:-1]:
            if not isinstance(phase, dict):
                continue
            if str(phase.get("phase_type") or "").strip() != "recover_entities":
                continue
            recovered_before_resume.update(
                _outline_phase_targets(phase) & unresolved_bridge_goal_parts
            )
        missing_parts = sorted(unresolved_bridge_goal_parts - recovered_before_resume)
        if missing_parts:
            errors.append(
                "repair_outline ending with resume_modeled_suffix must explicitly recover "
                "unresolved bridge-goal parts before resume; missing recover_entities for: "
                + ", ".join(missing_parts)
            )

    errors.extend(
        _check_grounded_executor_switches(
            outline_actions,
            recovery_context=recovery_context,
        )
    )
    errors.extend(
        _check_blocker_staging_destinations(
            outline_actions,
            recovery_context=recovery_context,
        )
    )

    if parsed.get("function_defs") not in (None, [], ()):
        errors.append("repair_outline must not include function_defs")
    if parsed.get("steps") not in (None, [], ()):
        errors.append("repair_outline must not include steps")
    if parsed.get("success_conditions") not in (None, [], ()):
        errors.append("repair_outline must not include success_conditions")

    closes_bridge = False
    if not errors:
        if terminal_phase == "replace_suffix":
            closes_bridge = True
        elif terminal_phase == "resume_modeled_suffix":
            closes_bridge = not unresolved_bridge_goal_parts

    return OutlineValidationResult(
        errors=errors,
        phase_signature=signature,
        outline_actions=deepcopy(outline_actions),
        terminal_phase=terminal_phase,
        unresolved_bridge_goal_parts=sorted(unresolved_bridge_goal_parts),
        closes_bridge=closes_bridge,
    )

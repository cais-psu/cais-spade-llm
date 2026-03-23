"""Compile task-graph mutations into ``_apply_replan_patch()`` patches.

v1 mutation surface: insert, delete, reassign, replace_suffix.

Each mutation validates inputs and produces a list of patch dicts in the
format expected by ``ProcessPlanner._apply_replan_patch()`` — which supports
node insertion (new ``id``), modification (existing ``id``), and deletion
(``"delete": True``).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any
from uuid import uuid4

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.mutation_types import (
    TaskMutationStep,
    TaskMutationType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_mutations(
    steps: list[TaskMutationStep],
    current_nodes: list[dict[str, Any]],
    *,
    anchor_task_id: str = "",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compile mutation steps into patch dicts for ``_apply_replan_patch()``.

    Parameters
    ----------
    steps:
        Ordered list of validated :class:`TaskMutationStep` to apply.
    current_nodes:
        Current task graph nodes (``ProcessPlanner.nodes``).
    anchor_task_id:
        Optional task ID to use as the insertion anchor point.

    Returns
    -------
    tuple:
        ``(patch_list, errors)`` — if *errors* is non-empty the patch
        should not be applied.
    """
    patches: list[dict[str, Any]] = []
    errors: list[str] = []
    node_map = {
        n["id"]: deepcopy(n) for n in current_nodes if isinstance(n, dict) and n.get("id")
    }

    for step in steps:
        step_patches, step_errors = _compile_single_mutation(
            step, node_map, anchor_task_id=anchor_task_id,
        )
        errors.extend(step_errors)
        if not step_errors:
            patches.extend(step_patches)
            # Apply patches to node_map so subsequent mutations see the
            # updated graph.
            _apply_patches_to_map(step_patches, node_map)

    return patches, errors


def validate_mutation_step(
    step: TaskMutationStep,
    current_nodes: list[dict[str, Any]],
) -> list[str]:
    """Validate a single mutation step without producing patches.

    Returns a list of error strings (empty if valid).
    """
    node_map = {
        n["id"]: deepcopy(n) for n in current_nodes if isinstance(n, dict) and n.get("id")
    }
    _, errors = _compile_single_mutation(step, node_map)
    return errors


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------

def _compile_single_mutation(
    step: TaskMutationStep,
    node_map: dict[str, dict[str, Any]],
    *,
    anchor_task_id: str = "",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Dispatch to the appropriate compiler for the mutation type."""
    compilers = {
        TaskMutationType.INSERT: _compile_insert,
        TaskMutationType.DELETE: _compile_delete,
        TaskMutationType.REASSIGN: _compile_reassign,
        TaskMutationType.REPLACE_SUFFIX: _compile_replace_suffix,
    }
    compiler = compilers.get(step.mutation_type)
    if compiler is None:
        return [], [f"Unsupported mutation type: {step.mutation_type.value}"]
    return compiler(step, node_map, anchor_task_id=anchor_task_id)


# ---------------------------------------------------------------------------
# INSERT
# ---------------------------------------------------------------------------

def _compile_insert(
    step: TaskMutationStep,
    node_map: dict[str, dict[str, Any]],
    *,
    anchor_task_id: str = "",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Insert new task nodes into the graph.

    Payload format::

        {
            "new_tasks": [
                {
                    "function_name": str,
                    "resource_jid": str,
                    "params": dict,
                    "predecessors": [str],  # optional
                    "successors": [str],    # optional
                    ...extra fields
                }
            ],
            "insert_after": str  # optional, task_id to insert after
        }
    """
    patches: list[dict[str, Any]] = []
    errors: list[str] = []
    payload = step.payload
    new_tasks = payload.get("new_tasks") or []

    if not new_tasks:
        errors.append("insert mutation: 'new_tasks' is empty")
        return patches, errors

    insert_after = str(
        payload.get("insert_after") or anchor_task_id or ""
    ).strip()

    # Validate insert_after reference.
    if insert_after and insert_after not in node_map:
        errors.append(
            f"insert mutation: insert_after task '{insert_after}' not found"
        )
        return patches, errors

    prev_task_id = insert_after
    for i, task_spec in enumerate(new_tasks):
        if not isinstance(task_spec, dict):
            errors.append(f"insert mutation: task at index {i} is not a dict")
            continue

        fn_name = str(task_spec.get("function_name", "")).strip()
        resource_jid = str(task_spec.get("resource_jid", "")).strip()
        if not fn_name:
            errors.append(f"insert mutation: task at index {i} missing function_name")
            continue
        if not resource_jid:
            errors.append(f"insert mutation: task at index {i} missing resource_jid")
            continue

        task_id = str(task_spec.get("id") or f"repair_insert_{uuid4().hex[:8]}").strip()

        predecessors = list(task_spec.get("predecessors") or [])
        if not predecessors and prev_task_id:
            predecessors = [prev_task_id]

        # Validate predecessor references.
        for pred_id in predecessors:
            if pred_id not in node_map and pred_id != task_id:
                # Allow references to tasks being inserted in the same batch.
                if not any(
                    (t.get("id") or "").strip() == pred_id
                    for t in new_tasks
                    if isinstance(t, dict)
                ):
                    errors.append(
                        f"insert mutation: predecessor '{pred_id}' not found for task '{task_id}'"
                    )

        patch = {
            "id": task_id,
            "type": "task",
            "function_name": fn_name,
            "resource_jid": resource_jid,
            "params": dict(task_spec.get("params") or {}),
            "predecessors": predecessors,
            "successors": list(task_spec.get("successors") or []),
            "status": "pending",
            "change_reason": f"v2 repair insert: {fn_name}",
        }
        # Carry extra fields.
        for key in (
            "in_state", "out_state", "part_name", "part_transition",
            "sequence_index", "primary_obligation",
        ):
            if key in task_spec:
                patch[key] = deepcopy(task_spec[key])

        patches.append(patch)
        prev_task_id = task_id

    return patches, errors


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------

def _compile_delete(
    step: TaskMutationStep,
    node_map: dict[str, dict[str, Any]],
    **_kwargs: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Delete tasks from the graph.

    All referenced task IDs must exist and must have status ``pending``.
    """
    patches: list[dict[str, Any]] = []
    errors: list[str] = []

    for task_id in step.target_task_ids:
        tid = str(task_id).strip()
        if not tid:
            continue
        if tid not in node_map:
            errors.append(f"delete mutation: task '{tid}' not found")
            continue
        node = node_map[tid]
        status = str(node.get("status", "")).strip()
        if status not in ("pending", ""):
            errors.append(
                f"delete mutation: task '{tid}' has status '{status}', "
                f"only 'pending' tasks can be deleted"
            )
            continue
        patches.append({
            "id": tid,
            "delete": True,
            "change_reason": str(step.payload.get("reason", "v2 repair delete")),
        })

    return patches, errors


# ---------------------------------------------------------------------------
# REASSIGN — validation-enriched delete+insert
# ---------------------------------------------------------------------------

def _compile_reassign(
    step: TaskMutationStep,
    node_map: dict[str, dict[str, Any]],
    **_kwargs: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reassign tasks to a different resource.

    The LLM provides both the task(s) to remove and their replacements
    already grounded for the new resource.  This compiler validates
    portability and produces delete+insert patches.

    Payload format::

        {
            "replacements": [
                {
                    "old_task_id": str,
                    "new_resource_jid": str,
                    "function_name": str,
                    "params": dict,
                    ...extra fields
                }
            ]
        }
    """
    patches: list[dict[str, Any]] = []
    errors: list[str] = []
    replacements = step.payload.get("replacements") or []

    if not replacements:
        errors.append("reassign mutation: 'replacements' is empty")
        return patches, errors

    for i, repl in enumerate(replacements):
        if not isinstance(repl, dict):
            errors.append(f"reassign mutation: replacement at index {i} is not a dict")
            continue

        old_task_id = str(repl.get("old_task_id", "")).strip()
        new_resource_jid = str(repl.get("new_resource_jid", "")).strip()
        fn_name = str(repl.get("function_name", "")).strip()

        if not old_task_id:
            errors.append(f"reassign mutation: replacement {i} missing old_task_id")
            continue
        if not new_resource_jid:
            errors.append(f"reassign mutation: replacement {i} missing new_resource_jid")
            continue
        if old_task_id not in node_map:
            errors.append(f"reassign mutation: old task '{old_task_id}' not found")
            continue

        old_node = node_map[old_task_id]
        old_status = str(old_node.get("status", "")).strip()
        if old_status not in ("pending", ""):
            errors.append(
                f"reassign mutation: task '{old_task_id}' has status '{old_status}', "
                f"only 'pending' tasks can be reassigned"
            )
            continue

        # Delete the old task.
        patches.append({
            "id": old_task_id,
            "delete": True,
            "change_reason": f"v2 repair reassign: replaced by {new_resource_jid}",
        })

        # Insert the replacement, preserving graph position.
        new_task_id = f"repair_reassign_{uuid4().hex[:8]}"
        patches.append({
            "id": new_task_id,
            "type": "task",
            "function_name": fn_name or old_node.get("function_name", ""),
            "resource_jid": new_resource_jid,
            "params": dict(repl.get("params") or old_node.get("params") or {}),
            "predecessors": list(old_node.get("predecessors") or []),
            "successors": list(old_node.get("successors") or []),
            "status": "pending",
            "change_reason": f"v2 repair reassign from {old_task_id}",
        })
        for key in (
            "in_state", "out_state", "part_name", "part_transition",
            "sequence_index", "primary_obligation",
        ):
            if key in repl:
                patches[-1][key] = deepcopy(repl[key])
            elif key in old_node:
                patches[-1][key] = deepcopy(old_node[key])

        # Fix successor references: tasks that had old_task_id as predecessor
        # now point to new_task_id.
        for nid, node in node_map.items():
            if old_task_id in (node.get("predecessors") or []):
                patches.append({
                    "id": nid,
                    "predecessors": [
                        new_task_id if p == old_task_id else p
                        for p in node.get("predecessors", [])
                    ],
                    "change_reason": f"v2 repair: update predecessor ref {old_task_id} -> {new_task_id}",
                })

    return patches, errors


# ---------------------------------------------------------------------------
# REPLACE_SUFFIX
# ---------------------------------------------------------------------------

def _compile_replace_suffix(
    step: TaskMutationStep,
    node_map: dict[str, dict[str, Any]],
    *,
    anchor_task_id: str = "",
    **_kwargs: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Replace the pending task suffix with a new sequence.

    Identifies suffix nodes (all pending tasks), deletes them, and inserts
    the replacement tasks.  Completed/running tasks are preserved.

    Payload format::

        {
            "new_suffix": [
                {
                    "function_name": str,
                    "resource_jid": str,
                    "params": dict,
                    ...
                }
            ]
        }
    """
    patches: list[dict[str, Any]] = []
    errors: list[str] = []
    new_suffix = step.payload.get("new_suffix") or []

    # Identify suffix nodes = all pending tasks.
    suffix_ids: list[str] = []
    last_completed_id = ""
    for node in sorted(
        node_map.values(),
        key=lambda n: n.get("sequence_index", 0),
    ):
        status = str(node.get("status", "")).strip()
        if status in ("completed", "running"):
            last_completed_id = node["id"]
        elif status in ("pending", ""):
            suffix_ids.append(node["id"])

    # Also use explicit target_task_ids if provided.
    if step.target_task_ids:
        explicit_ids = set(str(t).strip() for t in step.target_task_ids if str(t).strip())
        if explicit_ids:
            suffix_ids = [tid for tid in suffix_ids if tid in explicit_ids]

    # Delete suffix nodes.
    for tid in suffix_ids:
        patches.append({
            "id": tid,
            "delete": True,
            "change_reason": "v2 repair: replace_suffix delete",
        })

    # Determine anchor for new tasks.
    insert_after = anchor_task_id or last_completed_id

    # Insert replacement tasks.
    prev_id = insert_after
    for i, task_spec in enumerate(new_suffix):
        if not isinstance(task_spec, dict):
            errors.append(f"replace_suffix: task at index {i} is not a dict")
            continue

        fn_name = str(task_spec.get("function_name", "")).strip()
        resource_jid = str(task_spec.get("resource_jid", "")).strip()
        if not fn_name:
            errors.append(f"replace_suffix: task at index {i} missing function_name")
            continue
        if not resource_jid:
            errors.append(f"replace_suffix: task at index {i} missing resource_jid")
            continue

        task_id = str(
            task_spec.get("id") or f"repair_suffix_{uuid4().hex[:8]}"
        ).strip()
        predecessors = list(task_spec.get("predecessors") or [])
        if not predecessors and prev_id:
            predecessors = [prev_id]

        patch = {
            "id": task_id,
            "type": "task",
            "function_name": fn_name,
            "resource_jid": resource_jid,
            "params": dict(task_spec.get("params") or {}),
            "predecessors": predecessors,
            "successors": list(task_spec.get("successors") or []),
            "status": "pending",
            "change_reason": f"v2 repair: replace_suffix insert {fn_name}",
        }
        for key in (
            "in_state", "out_state", "part_name", "part_transition",
            "sequence_index", "primary_obligation",
        ):
            if key in task_spec:
                patch[key] = deepcopy(task_spec[key])

        patches.append(patch)
        prev_id = task_id

    return patches, errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_patches_to_map(
    patches: list[dict[str, Any]],
    node_map: dict[str, dict[str, Any]],
) -> None:
    """Apply patches to an in-memory node map (for sequential mutation chaining)."""
    for patch in patches:
        tid = patch.get("id")
        if not tid:
            continue
        if patch.get("delete"):
            node_map.pop(tid, None)
            for node in node_map.values():
                if tid in (node.get("predecessors") or []):
                    node["predecessors"].remove(tid)
                if tid in (node.get("successors") or []):
                    node["successors"].remove(tid)
        else:
            if tid in node_map:
                node_map[tid].update(patch)
            else:
                node_map[tid] = dict(patch)

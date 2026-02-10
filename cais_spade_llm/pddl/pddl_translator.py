"""
Translates a pyperplan plan (list of PDDL actions) into task DAG patches
in the same format that process_planner._replan_with_feedback() produces:

  {"tasks": [
      {
        "id": "RECOVERY_1",
        "function_name": "move_to_pick_location",
        "params": {...},
        "resource_jid": "ur5e@localhost",
        "predecessors": [...],
        "successors": [...],
        "change_reason": "..."
      },
      ...
  ]}

PDDL action → function expansion:
  pick-part(robot, part, loc)   →  move_to_pick_location + pick_part
  place-part(robot, part, loc)  →  move_loaded_to_destination + place_part (final)
  stage-part(robot, part, loc)  →  move_loaded_to_destination + place_part (staging)
  unlock-placement(p1, p2)      →  skipped (internal PDDL bookkeeping, no real action)

PDDL parameter names use hyphens and no @; we reverse-map them back to
the original robot JIDs and part names using the provided name maps.
"""

from __future__ import annotations
import uuid
from typing import Any


def _short_id() -> str:
    return uuid.uuid4().hex[:6].upper()


def translate(
    pddl_plan: list[tuple[str, list[str]]],
    *,
    robot_jid_map: dict[str, str],   # pddl_name → "ur5e@localhost"
    part_name_map: dict[str, str],   # pddl_name → "SG"
    location_map: dict[str, str],    # pddl_name → "assembly_board-v1" or staging zone name
    product_jid: str,
    failed_task_id: str,             # the task whose successor was blocked (for predecessor linking)
    existing_task_ids: set[str],     # ids already in the plan (to avoid collisions)
) -> dict[str, Any]:
    """
    Convert a PDDL plan into task DAG patches.

    Returns:
        {"tasks": [...]} matching the format expected by apply_plan_patch().
    """

    tasks: list[dict] = []
    prev_id: str | None = None   # predecessor chain

    for action_name, params in pddl_plan:

        # ------------------------------------------------------------------
        # skip internal PDDL bookkeeping actions
        # ------------------------------------------------------------------
        if action_name == "unlock-placement":
            continue

        robot_pddl = params[0] if len(params) > 0 else ""
        part_pddl  = params[1] if len(params) > 1 else ""
        loc_pddl   = params[2] if len(params) > 2 else ""

        robot_jid  = robot_jid_map.get(robot_pddl, robot_pddl)
        part_name  = part_name_map.get(part_pddl, part_pddl.upper())
        location   = location_map.get(loc_pddl, loc_pddl)

        if action_name == "pick-part":
            # Expand to: move_to_pick_location → pick_part
            move_id = _unique_id("RECOVERY_MOVE", existing_task_ids)
            pick_id = _unique_id("RECOVERY_PICK", existing_task_ids)
            existing_task_ids.update({move_id, pick_id})

            move_task = {
                "id": move_id,
                "function_name": "move_to_pick_location",
                "params": {
                    "origin_resource_location": location,
                    "part_name": part_name,
                    "product_jid": product_jid,
                    "task_id": move_id,
                },
                "resource_jid": robot_jid,
                "predecessors": [prev_id] if prev_id else [],
                "successors": [pick_id],
                "change_reason": f"INSERTION: Move {robot_jid} to pick {part_name} from {location}",
            }
            pick_task = {
                "id": pick_id,
                "function_name": "pick_part",
                "params": {
                    "part_name": part_name,
                    "origin_resource_location": location,
                    "product_jid": product_jid,
                    "task_id": pick_id,
                },
                "resource_jid": robot_jid,
                "predecessors": [move_id],
                "successors": [],
                "change_reason": f"INSERTION: {robot_jid} picks {part_name} from {location}",
            }
            tasks.append(move_task)
            tasks.append(pick_task)
            prev_id = pick_id

        elif action_name in ("place-part", "stage-part"):
            is_staging = (action_name == "stage-part")
            label = "stage" if is_staging else "place"

            move_id  = _unique_id("RECOVERY_MOVE", existing_task_ids)
            place_id = _unique_id(f"RECOVERY_{label.upper()}", existing_task_ids)
            existing_task_ids.update({move_id, place_id})

            move_task = {
                "id": move_id,
                "function_name": "move_loaded_to_destination",
                "params": {
                    "destination_location": location,
                    "part_name": part_name,
                    "product_jid": product_jid,
                    "task_id": move_id,
                },
                "resource_jid": robot_jid,
                "predecessors": [prev_id] if prev_id else [],
                "successors": [place_id],
                "change_reason": f"INSERTION: Move {robot_jid} carrying {part_name} to {location}",
            }
            place_task = {
                "id": place_id,
                "function_name": "place_part",
                "params": {
                    "destination_location": location,
                    "part_name": part_name,
                    "product_jid": product_jid,
                    "task_id": place_id,
                },
                "resource_jid": robot_jid,
                "predecessors": [move_id],
                "successors": [],
                "change_reason": (
                    f"INSERTION: {robot_jid} {'stages' if is_staging else 'places'} "
                    f"{part_name} at {location}"
                ),
            }
            tasks.append(move_task)
            tasks.append(place_task)
            prev_id = place_id

    # ------------------------------------------------------------------
    # The last recovery task becomes the new predecessor for the
    # originally blocked downstream task (e.g. REQ_2_T4).
    # We return this as a modification of the blocked task.
    # ------------------------------------------------------------------
    # (The caller in process_planner is responsible for finding
    #  which task was blocked and updating its predecessors.)

    return {"tasks": tasks, "recovery_tail_id": prev_id}


def _unique_id(base: str, existing: set[str]) -> str:
    """Generate a unique task ID that doesn't collide with existing ones."""
    candidate = f"{base}_{_short_id()}"
    while candidate in existing:
        candidate = f"{base}_{_short_id()}"
    return candidate

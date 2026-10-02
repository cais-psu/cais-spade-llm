"""Rebuild synthetic Part slippage regression inputs from the current plant.

These files exercise saved-context diagnostics; they are not recorded Gazebo runs
or approved recovery programs.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from ltlf2dfa.parser.ltlf import LTLfParser

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.workflow_execution import _robot_configuration
from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry
from cais_spade_llm.ui.recovery_setup import default_setup, validate_setup


def build() -> None:
    """Write both two-robot interruption directions and their diagnostic bundles."""
    inputs = validate_setup(default_setup())
    scene, models = inputs["scene"], inputs["models"]
    directory = Path(__file__).parent

    def write(path, value):
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2) + "\n")

    jids = {rid: f"recovery-resource-{index}@localhost" for index, rid in enumerate(models, 1)}
    for slipping, slipped, peer, held, name in (
        ("ur5e-3", "KET4_Square_4mm", "ur5e-4", "gear_large", ""),
        ("ur5e-4", "gear_large", "ur5e-3", "KET4_Square_4mm", "_reverse"),
    ):
        bundle = "bundle" + name
        origins = {rid: models[rid]["assignments"]["source"] for rid in (slipping, peer)}
        robot_configs, catalog, nodes, requirements, snapshots, tracker, statuses = {}, [], [], [], [], {}, {}
        for requirement_id, rid, part in (("REQ_1", peer, held), ("REQ_2", slipping, slipped)):
            robot = next(row for row in scene["robots"] if row["resource_id"] == rid)
            controller, positions, static = _robot_configuration(scene, robot)
            static["reachability"] = [value for value in models[rid]["state_variables"]["resource_location"]["domain"]
                                      if value is not None and value != "home"]
            static["staging_areas"] = {}
            origin = origins[rid]
            point = (scene[origin]["output_poses"][part] if origin == "3D Printing Station"
                     else scene[origin]["pickup_pose"])
            static["staging_areas"][origin] = {
                "anchor_pose": dict(zip(("x", "y", "z"), point[:3])),
                "board_center": dict(zip(("x", "y", "z"), point[:3])),
                "slot_xy": [0., 0.], "slot_floor_z_m": point[2],
            }
            config = {"type": "robot", "jid": jids[rid], "functions": "auto", "failure_scenarios": [],
                      "gazebo": {"controller": controller, "named_positions": positions,
                                 "static_capabilities": static}}
            robot_configs[rid] = config
            functions = ("pick_approach", "pick_grasp", "place_approach", "place_insert", "move_home")
            for index, function in enumerate(functions, 1):
                task = robot_task_registry()[function]
                catalog.append({"function": function, "function_owner_agent": jids[rid].split("@")[0],
                                **task.tool_frontmatter()})
                task_id = f"{requirement_id}_T{index}"
                params = {"product_jid": "assembly_board-v1@localhost", "task_id": task_id}
                if function != "move_home":
                    params["part_name"] = part
                    params["origin_resource_location" if function.startswith("pick") else "destination_location"] = (
                        origin if function.startswith("pick") else "assembly_board-v1")
                nodes.append({"id": task_id, "type": "task", "requirement_id": requirement_id,
                              "function_name": function, "resource_jid": jids[rid], "params": params,
                              "sequence_index": index - 1, "status": "pending",
                              "predecessors": [f"{requirement_id}_T{index-1}"] if index > 1 else [],
                              "successors": [f"{requirement_id}_T{index+1}"] if index < 5 else []})
                statuses[task_id] = "completed" if index <= 2 else "pending"
            requirements.append({"id": requirement_id, "type": "requirement", "phase": "ASSEMBLY",
                                 "process_type": "PICK_PLACE", "product": part,
                                 "raw_text": f"{rid} assembles {part} from {origin}.",
                                 "context": {"origin": origin, "destination": "assembly_board-v1",
                                             "robot_id": jids[rid].split("@")[0]}})
            pose = {"x": 0., "y": -.2 if slipping == "ur5e-3" else .2, "z": 1.04}
            snapshots.append({
                "resource_id": rid, "resource_jid": jids[rid],
                "resource_config": f"test/fixtures/part_slippage/resources{name}.json",
                "resource_config_key": rid, "execution_env": "gazebo",
                "current_state": "failed" if rid == slipping else "picked",
                "held_part": None if rid == slipping else part,
                "gripper_state": "open" if rid == slipping else "closed",
                "current_pose_ref": None, "current_pose": dict(zip(("x", "y", "z"), point[:3])),
                "current_location": origin,
                "observations": {part: {"part_name": part, **pose, "pose": pose}} if rid == slipping else {},
            })
            tracker[part] = {"state": "misplaced" if rid == slipping else "in_gripper",
                             "location": None if rid == slipping else jids[rid],
                             "last_known_location": None if rid == slipping else jids[rid],
                             "last_successful_task": f"{requirement_id}_T2", "origin_resource_location": origin}
        slipped_owner, held_owner = jids[slipping].split("@")[0], jids[peer].split("@")[0]
        ap = lambda part, owner, event: f"ap_event/assembly/{part.lower()}/{owner}/{event}/destination=assembly_board-v1"
        rules = [
            {"id": "SAFE_1", "raw_text": f"{slipped} placement precedes {held} placement in this diagnostic test.",
             "constraint_type": "ordering_place_approach_priority", "process": "assembly",
             "product": [slipped, held], "resources": [slipped_owner, held_owner], "resource_types": ["robot"],
             "event": "place_approach", "context": {"destination": "assembly_board-v1"},
             "aps": [{"label": "ap004", "full": ap(held, held_owner, "place_approach")},
                     {"label": "ap003", "full": ap(slipped, slipped_owner, "place_approach")}],
             "ltlf": "(!(ap004) U ap003)"},
            {"id": "SAFE_2", "raw_text": "The two robots must not occupy the assembly destination together.",
             "constraint_type": "mutual_exclusion_in_destination_area", "process": None, "product": [],
             "resources": [held_owner, slipped_owner], "resource_types": ["robot"], "event": None,
             "context": {"destination": "assembly_board-v1"},
             "aps": [{"label": label, "full": f"ap_{kind}/assembly/any/{owner}/{event}/destination=assembly_board-v1"}
                     for owner, entries in ((held_owner, (("ap001", "event", "place_approach"), ("ap006", "state", "positioned"), ("ap005", "state", "placed"))),
                                             (slipped_owner, (("ap002", "event", "place_approach"), ("ap008", "state", "positioned"), ("ap007", "state", "placed"))))
                     for label, kind, event in entries],
             "ltlf": "G (!(((ap001 | ap006 | ap005) & (ap002 | ap008 | ap007))))"},
        ]
        write(f"resources{name}.json", robot_configs)
        write(f"{bundle}/catalog/tools.json", catalog)
        write(f"{bundle}/plan/plan.json", {"nodes": nodes})
        write(f"{bundle}/plan/requirements.json", {"nodes": requirements})
        write(f"{bundle}/safety/cca_safety_logic.json", {"test_only": True, "rules": rules})
        for rule in rules:
            dot = LTLfParser()(rule["ltlf"]).to_dfa()
            if "->" not in dot:
                raise ValueError("MONA did not compile the diagnostic safety rule")
            (directory / bundle / "safety" / (rule["id"] + "_dfa.dot")).write_text(dot)

        write(f"{bundle}/bundle_manifest.json", {
            "bundle_id": "part_slippage_test" + name, "status": "test_fixture", "verified": False,
            "product_name": "assembly_board-v1", "execution_mode": "simulation", "robot_env": "gazebo",
            "artifacts": {"tools_json": "catalog/tools.json", "plan_json": "plan/plan.json",
                          "requirements_json": "plan/requirements.json", "safety_logic_json": "safety/cca_safety_logic.json"},
        })
        write(f"runtime_context{name}.json", {
            "fixture_kind": "synthetic_checkpoint", "bundle_root": f"test/fixtures/part_slippage/{bundle}",
            "failure_scenario_id": "part_slippage",
            "product_geometry": "cais_spade_llm/specification/products/geometry/assembly_board-v1-recovery-framework.json",
            "goal_state": "assembled", "default_resource_state": "idle", "task_statuses": statuses,
            "failure_event": {"failed_task_id": "REQ_2_T4", "final_status": "interrupted",
                              "checkpoint": "after_both_pickups_before_place", "part_name": slipped,
                              "state_before": {"execution_mode": "simulation", "controller_ready": True,
                                               "held_part": slipped, "current_state": "picked", "gripper_state": "closed"},
                              "state_after": {"execution_mode": "simulation", "controller_ready": True,
                                              "held_part": None, "current_state": "failed", "gripper_state": "open"}},
            "part_tracker": tracker, "resource_snapshots": snapshots,
            "obligation_targets": [], "recovery_feedback": "", "recovery_safety_context": {},
        })


if __name__ == "__main__":
    build()

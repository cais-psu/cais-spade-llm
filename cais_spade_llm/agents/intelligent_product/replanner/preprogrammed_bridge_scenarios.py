from __future__ import annotations

from typing import Any


_RECOVER_LG_V1 = "recover_lg_v1"
_RECOVER_LG_V1_ALIASES = frozenset(
    {
        _RECOVER_LG_V1,
        "case3_llm_bridge",
        "case3_two_arm_llm_bridge",
    }
)
_XARM6_CLEAR_SPEED = 0.5
_UR5E_TRAVERSE_SPEED = 1.2
_UR5E_CAREFUL_SPEED = 1.2
_UR5E_HOME_SPEED = 0.8
_MCP_RETURN_SHIFT_DY_M = 0.20
_MCP_RETURN_DESCEND_DZ_M = -0.10
_MCP_RETURN_LIFT_DZ_M = 0.10
_POST_PICK_LIFT_DZ_M = 0.05
_POST_PLACE_LIFT_DZ_M = 0.08
_LG_PICK_FINAL_NUDGE_DZ_M = 0.0
_LG_INSERT_Z_ADJUSTMENT_M = 0.006
_LG_RECOVERY_APPROACH_HEIGHT_M = 0.06
_LG_RECOVERY_MIN_PICK_TCP_Z_M = 1.032
_UR5E_LG_GRIPPER_POSITION = 0.055


def canonical_preprogrammed_bridge_scenario_id(scenario_id: str) -> str:
    scenario_key = str(scenario_id or "").strip()
    if scenario_key in _RECOVER_LG_V1_ALIASES:
        return _RECOVER_LG_V1
    return scenario_key


def build_preprogrammed_bridge_proposal(
    *,
    scenario_id: str,
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    scenario_key = canonical_preprogrammed_bridge_scenario_id(scenario_id)
    if scenario_key != _RECOVER_LG_V1:
        raise ValueError(f"unsupported preprogrammed bridge scenario '{scenario_key}'")
    if not isinstance(prepared_bridge_request, dict) or not prepared_bridge_request:
        raise ValueError("prepared bridge request is missing")
    return _build_recover_lg_v1(prepared_bridge_request)


def _build_recover_lg_v1(prepared_bridge_request: dict[str, Any]) -> dict[str, Any]:
    grounding_context = dict(prepared_bridge_request.get("grounding_context") or {})
    parts = dict(grounding_context.get("parts") or {})
    if "MCP" not in parts or "LG" not in parts:
        raise ValueError("prepared bridge request is missing MCP/LG grounding context")

    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    if "ur5e@localhost" not in bridge_resources or "xarm6@localhost" not in bridge_resources:
        raise ValueError("recover_lg_v1 requires ur5e@localhost and xarm6@localhost")

    primary_obligation = _select_primary_obligation(
        prepared_bridge_request,
        preferred_resource_jid="xarm6@localhost",
    )
    mcp_origin = _recover_part_origin_location(
        prepared_bridge_request,
        part_name="MCP",
        preferred_resource_jid="ur5e@localhost",
    )
    _part_model_name(grounding_context, "MCP")
    _part_model_name(grounding_context, "LG")

    mcp_pick_q = _current_pose_orientation_params(fact_path="current_pose")
    lg_observed_pose = _part_observed_pose(
        prepared_bridge_request,
        grounding_context=grounding_context,
        part_name="LG",
    )
    mcp_observed_pose = _part_observed_pose(
        prepared_bridge_request,
        grounding_context=grounding_context,
        part_name="MCP",
    )

    if lg_observed_pose:
        lg_pick_steps = [
            {
                "primitive": "get_current_pose",
                "params": {},
            },
            {
                "primitive": "compute_pick_targets",
                "params": {
                    "part_name": "LG",
                    "product_geometry": _part_pick_geometry(
                        grounding_context,
                        part_name="LG",
                    ),
                    "target_pose": lg_observed_pose,
                    "approach_height_override_m": _LG_RECOVERY_APPROACH_HEIGHT_M,
                    "ignore_current_height_for_travel_z": True,
                    "min_pick_tcp_z_override_m": _LG_RECOVERY_MIN_PICK_TCP_Z_M,
                },
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": {"context_ref": "/event_facts/pick_targets/LG/approach_pose/x"},
                    "y": {"context_ref": "/event_facts/pick_targets/LG/approach_pose/y"},
                    "z": {"context_ref": "/event_facts/pick_targets/LG/approach_pose/z"},
                    "speed": _UR5E_TRAVERSE_SPEED,
                },
            },
            {
                "primitive": "move_pose",
                "params": {
                    "x": {"context_ref": "/event_facts/pick_targets/LG/target_pose/x"},
                    "y": {"context_ref": "/event_facts/pick_targets/LG/target_pose/y"},
                    "z": {"context_ref": "/event_facts/pick_targets/LG/target_pose/z"},
                    **_current_pose_orientation_params(fact_path="current_pose"),
                    "speed": _UR5E_CAREFUL_SPEED,
                },
            },
            {
                "primitive": "move_relative",
                "params": {
                    "dx": 0.0,
                    "dy": 0.0,
                    "dz": _LG_PICK_FINAL_NUDGE_DZ_M,
                    "speed": _UR5E_CAREFUL_SPEED,
                },
            },
            {
                "primitive": "close_gripper",
                "params": {"position": _UR5E_LG_GRIPPER_POSITION},
            },
            {
                "primitive": "attach_part",
                "params": {
                    "model_name": {"context_ref": "/parts/LG/target/model_name"},
                },
            },
            {
                "primitive": "move_relative",
                "params": {
                    "dx": 0.0,
                    "dy": 0.0,
                    "dz": _POST_PICK_LIFT_DZ_M,
                    "speed": _UR5E_CAREFUL_SPEED,
                },
            },
        ]
    else:
        lg_pick_steps = [
            {
                "primitive": "detect_parts",
                "params": {"part_name": "LG"},
            },
            {
                "primitive": "get_current_pose",
                "params": {},
            },
            {
                "primitive": "compute_pick_targets",
                "params": {
                    "part_name": "LG",
                    "product_geometry": _part_pick_geometry(
                        grounding_context,
                        part_name="LG",
                    ),
                    "approach_height_override_m": _LG_RECOVERY_APPROACH_HEIGHT_M,
                    "ignore_current_height_for_travel_z": True,
                    "min_pick_tcp_z_override_m": _LG_RECOVERY_MIN_PICK_TCP_Z_M,
                },
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": {"context_ref": "/event_facts/detected_part/LG/pose/x"},
                    "y": {"context_ref": "/event_facts/detected_part/LG/pose/y"},
                    "z": {"context_ref": "/event_facts/pick_targets/LG/travel_z"},
                    "speed": _UR5E_TRAVERSE_SPEED,
                },
            },
            {
                "primitive": "move_pose",
                "params": {
                    "x": {"context_ref": "/event_facts/detected_part/LG/pose/x"},
                    "y": {"context_ref": "/event_facts/detected_part/LG/pose/y"},
                    "z": {"context_ref": "/event_facts/pick_targets/LG/pick_z"},
                    **_current_pose_orientation_params(fact_path="current_pose"),
                    "speed": _UR5E_CAREFUL_SPEED,
                },
            },
            {
                "primitive": "move_relative",
                "params": {
                    "dx": 0.0,
                    "dy": 0.0,
                    "dz": _LG_PICK_FINAL_NUDGE_DZ_M,
                    "speed": _UR5E_CAREFUL_SPEED,
                },
            },
            {
                "primitive": "close_gripper",
                "params": {"position": _UR5E_LG_GRIPPER_POSITION},
            },
            {
                "primitive": "attach_part",
                "params": {
                    "model_name": {"context_ref": "/parts/LG/target/model_name"},
                },
            },
            {
                "primitive": "move_relative",
                "params": {
                    "dx": 0.0,
                    "dy": 0.0,
                    "dz": _POST_PICK_LIFT_DZ_M,
                    "speed": _UR5E_CAREFUL_SPEED,
                },
            },
        ]

    if mcp_observed_pose:
        mcp_resume_steps = [
            {
                "primitive": "move_to_named_pose",
                "params": {
                    "pose_name": {
                        "context_ref": "/resources/ur5e@localhost/named_poses/home",
                    },
                    "speed": _UR5E_TRAVERSE_SPEED,
                },
            },
            {
                "primitive": "get_current_pose",
                "params": {},
            },
            {
                "primitive": "compute_pick_targets",
                "params": {
                    "part_name": "MCP",
                    "product_geometry": _part_pick_geometry(
                        grounding_context,
                        part_name="MCP",
                    ),
                    "target_pose": mcp_observed_pose,
                },
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": {"context_ref": "/event_facts/pick_targets/MCP/approach_pose/x"},
                    "y": {"context_ref": "/event_facts/pick_targets/MCP/approach_pose/y"},
                    "z": {"context_ref": "/event_facts/pick_targets/MCP/approach_pose/z"},
                    "speed": _UR5E_TRAVERSE_SPEED,
                },
            },
            {
                "primitive": "move_pose",
                "params": {
                    "x": {"context_ref": "/event_facts/pick_targets/MCP/target_pose/x"},
                    "y": {"context_ref": "/event_facts/pick_targets/MCP/target_pose/y"},
                    "z": {"context_ref": "/event_facts/pick_targets/MCP/target_pose/z"},
                    **mcp_pick_q,
                },
            },
            {"primitive": "close_gripper", "params": {}},
            {
                "primitive": "attach_part",
                "params": {
                    "model_name": {"context_ref": "/parts/MCP/target/model_name"},
                },
            },
            {
                "primitive": "move_relative",
                "params": {
                    "dx": 0.0,
                    "dy": 0.0,
                    "dz": _POST_PICK_LIFT_DZ_M,
                    "speed": _UR5E_CAREFUL_SPEED,
                },
            },
        ]
    else:
        mcp_resume_steps = [
            {
                "primitive": "move_to_named_pose",
                "params": {
                    "pose_name": {
                        "context_ref": "/resources/ur5e@localhost/named_poses/home",
                    },
                    "speed": _UR5E_TRAVERSE_SPEED,
                },
            },
            {
                "primitive": "get_current_pose",
                "params": {},
            },
            {
                "primitive": "detect_parts",
                "params": {"part_name": "MCP"},
            },
            {
                "primitive": "compute_pick_targets",
                "params": {
                    "part_name": "MCP",
                    "product_geometry": _part_pick_geometry(
                        grounding_context,
                        part_name="MCP",
                    ),
                },
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": {"context_ref": "/event_facts/detected_part/MCP/pose/x"},
                    "y": {"context_ref": "/event_facts/detected_part/MCP/pose/y"},
                    "z": {"context_ref": "/event_facts/current_pose/pose/z"},
                    "speed": _UR5E_TRAVERSE_SPEED,
                },
            },
            {
                "primitive": "move_pose",
                "params": {
                    "x": {"context_ref": "/event_facts/detected_part/MCP/pose/x"},
                    "y": {"context_ref": "/event_facts/detected_part/MCP/pose/y"},
                    "z": {"context_ref": "/event_facts/pick_targets/MCP/pick_z"},
                    **mcp_pick_q,
                },
            },
            {"primitive": "close_gripper", "params": {}},
            {
                "primitive": "attach_part",
                "params": {
                    "model_name": {"context_ref": "/parts/MCP/target/model_name"},
                },
            },
            {
                "primitive": "move_relative",
                "params": {
                    "dx": 0.0,
                    "dy": 0.0,
                    "dz": _POST_PICK_LIFT_DZ_M,
                    "speed": _UR5E_CAREFUL_SPEED,
                },
            },
        ]

    proposal = {
        "macro_tasks": [
            {
                "outline_id": "clear_xarm6_zone",
                "predecessors": [],
                "resource_jid": "xarm6@localhost",
                "macro_name": "clear_xarm6_zone",
                "description": "Retreat xarm6 to the recovery-clear pose to free the shared workspace.",
                "rationale": "Clears the collision zone before ur5e performs the LG recovery.",
                "expected_start_state": "failed",
                "task_params": {
                    "clear_pose": {"context_ref": "/resources/xarm6@localhost/named_poses/home"},
                },
                "task_metadata": {
                    "in_state": "failed",
                    "out_state": "idle",
                    "required_context_keys": [],
                    "context_mapping": {},
                    "part_transition": None,
                },
                "primitive_steps": [
                    {
                        "primitive": "move_to_named_pose",
                        "params": {
                            "pose_name": {
                                "context_ref": "/resources/xarm6@localhost/named_poses/home",
                            },
                            "speed": _XARM6_CLEAR_SPEED,
                        },
                    }
                ],
            },
            {
                "outline_id": "return_mcp_to_printer",
                "predecessors": [],
                "resource_jid": "ur5e@localhost",
                "macro_name": "return_mcp_to_printer",
                "description": "Release MCP back to its printer origin so ur5e can recover LG first.",
                "rationale": "Unloads MCP and restores ur5e to an empty-gripper state for the LG pickup.",
                "expected_start_state": "picked",
                "part_name": "MCP",
                "task_params": {
                    "destination_location": mcp_origin,
                },
                "task_metadata": {
                    "in_state": "picked",
                    "out_state": "idle",
                    "required_context_keys": ["destination_location"],
                    "context_mapping": {
                        "location_param": "destination_location",
                    },
                    "part_transition": {
                        "completed": {
                            "state": "ready",
                            "location_param": "destination_location",
                        }
                    },
                },
                "primitive_steps": [
                    {
                        "primitive": "move_relative",
                        "params": {
                            "dx": 0.0,
                            "dy": 0.0,
                            "dz": _MCP_RETURN_DESCEND_DZ_M,
                            "speed": _UR5E_CAREFUL_SPEED,
                        },
                    },
                    {"primitive": "open_gripper", "params": {}},
                    {
                        "primitive": "detach_part",
                        "params": {
                            "model_name": {"context_ref": "/parts/MCP/target/model_name"},
                            "assume_released_if_open": True,
                        },
                    },
                    {
                        "primitive": "move_relative",
                        "params": {
                            "dx": 0.0,
                            "dy": 0.0,
                            "dz": _POST_PLACE_LIFT_DZ_M,
                            "speed": _UR5E_TRAVERSE_SPEED,
                        },
                    },
                    {
                        "primitive": "move_to_named_pose",
                        "params": {
                            "pose_name": {
                                "context_ref": "/resources/ur5e@localhost/named_poses/home",
                            },
                            "speed": _UR5E_HOME_SPEED,
                        },
                    },
                ],
            },
            {
                "outline_id": "pick_lg",
                "predecessors": [
                    "clear_xarm6_zone",
                    "return_mcp_to_printer",
                ],
                "resource_jid": "ur5e@localhost",
                "macro_name": "pick_lg",
                "description": "Approach the dropped LG from above and secure it with ur5e.",
                "rationale": "Transfers LG recovery to ur5e using a normal top-down pickup.",
                "expected_start_state": "idle",
                "part_name": "LG",
                "task_params": {
                    "origin_resource_location": {"context_ref": "/parts/LG/location"},
                    "recovery_style": "top_down",
                },
                "task_metadata": {
                    "in_state": "idle",
                    "out_state": "picked",
                    "required_context_keys": ["origin_resource_location"],
                    "context_mapping": {
                        "location_param": "origin_resource_location",
                        "location_type": "part_location",
                    },
                    "part_transition": {
                        "completed": {
                            "state": "in_gripper",
                            "location_template": "{resource_jid}_gripper",
                        }
                    },
                },
                "primitive_steps": lg_pick_steps,
            },
            {
                "outline_id": "insert_lg",
                "predecessors": ["pick_lg"],
                "resource_jid": "ur5e@localhost",
                "macro_name": "insert_lg",
                "description": "Carry the recovered LG into the board with a normal vertical insertion.",
                "rationale": "Completes the LG assembly so the normal MCP suffix can resume.",
                "expected_start_state": "picked",
                "part_name": "LG",
                "task_params": {
                    "destination_location": "assembly_board-v1",
                    "recovery_style": "top_down",
                },
                "task_metadata": {
                    "in_state": "picked",
                    "out_state": "idle",
                    "required_context_keys": ["destination_location"],
                    "context_mapping": {
                        "location_param": "destination_location",
                        "location_type": "current_location",
                    },
                    "part_transition": {
                        "completed": {
                            "state": "assembled",
                            "location_param": "destination_location",
                        }
                    },
                },
                "primitive_steps": [
                    {
                        "primitive": "get_current_pose",
                        "params": {},
                    },
                    {
                        "primitive": "compute_place_targets",
                        "params": {
                            "part_name": "LG",
                            "product_geometry": _part_pick_geometry(
                                grounding_context,
                                part_name="LG",
                            ),
                            "z_adjustment_m": _LG_INSERT_Z_ADJUSTMENT_M,
                        },
                    },
                    {
                        "primitive": "move_cartesian",
                        "params": {
                            "x": {"context_ref": "/event_facts/place_targets/LG/slot_x"},
                            "y": {"context_ref": "/event_facts/place_targets/LG/slot_y"},
                            "z": {"context_ref": "/event_facts/current_pose/pose/z"},
                            "speed": _UR5E_TRAVERSE_SPEED,
                        },
                    },
                    {
                        "primitive": "move_pose",
                        "params": {
                            "x": {"context_ref": "/event_facts/place_targets/LG/slot_x"},
                            "y": {"context_ref": "/event_facts/place_targets/LG/slot_y"},
                            "z": {"context_ref": "/event_facts/place_targets/LG/place_z"},
                            **_current_pose_orientation_params(fact_path="current_pose"),
                            "speed": _UR5E_CAREFUL_SPEED,
                        },
                    },
                    {"primitive": "open_gripper", "params": {}},
                    {
                        "primitive": "detach_part",
                        "params": {
                            "model_name": {"context_ref": "/parts/LG/target/model_name"},
                            "assume_released_if_open": True,
                        },
                    },
                    {
                        "primitive": "move_relative",
                        "params": {
                            "dx": 0.0,
                            "dy": 0.0,
                            "dz": _POST_PLACE_LIFT_DZ_M,
                            "speed": _UR5E_TRAVERSE_SPEED,
                        },
                    },
                    {
                        "primitive": "move_to_named_pose",
                        "params": {
                            "pose_name": {
                                "context_ref": "/resources/ur5e@localhost/named_poses/home",
                            },
                            "speed": _UR5E_HOME_SPEED,
                        },
                    },
                ],
            },
            {
                "outline_id": "resume_mcp_assembly",
                "predecessors": ["insert_lg"],
                "resource_jid": "ur5e@localhost",
                "macro_name": "resume_mcp_assembly",
                "description": "Repick MCP into ur5e so the remaining MCP assembly tail can be completed inside the bridge.",
                "rationale": "Restores MCP to the gripper so the bridge can finish the remaining MCP assembly work end-to-end.",
                "expected_start_state": "idle",
                "part_name": "MCP",
                "task_params": {
                    "origin_resource_location": mcp_origin,
                },
                "task_metadata": {
                    "in_state": "idle",
                    "out_state": "picked",
                    "required_context_keys": ["origin_resource_location"],
                    "context_mapping": {
                        "location_param": "origin_resource_location",
                        "location_type": "part_location",
                    },
                    "part_transition": {
                        "completed": {
                            "state": "in_gripper",
                            "location_template": "{resource_jid}_gripper",
                        }
                    },
                },
                "primitive_steps": mcp_resume_steps,
            },
            {
                "outline_id": "insert_mcp",
                "predecessors": ["resume_mcp_assembly"],
                "resource_jid": "ur5e@localhost",
                "macro_name": "insert_mcp",
                "description": "Carry MCP into the assembly board and complete the remaining MCP insertion.",
                "rationale": "Completes the focused MCP suffix inside the bridge so no pending MCP tail remains.",
                "expected_start_state": "picked",
                "part_name": "MCP",
                "task_params": {
                    "destination_location": "assembly_board-v1",
                },
                "task_metadata": {
                    "in_state": "picked",
                    "out_state": "idle",
                    "required_context_keys": ["destination_location"],
                    "context_mapping": {
                        "location_param": "destination_location",
                        "location_type": "current_location",
                    },
                    "part_transition": {
                        "completed": {
                            "state": "assembled",
                            "location_param": "destination_location",
                        }
                    },
                },
                "primitive_steps": [
                    {
                        "primitive": "get_current_pose",
                        "params": {},
                    },
                    {
                        "primitive": "compute_place_targets",
                        "params": {
                            "part_name": "MCP",
                            "product_geometry": _part_pick_geometry(
                                grounding_context,
                                part_name="MCP",
                            ),
                        },
                    },
                    {
                        "primitive": "move_cartesian",
                        "params": {
                            "x": {"context_ref": "/event_facts/place_targets/MCP/slot_x"},
                            "y": {"context_ref": "/event_facts/place_targets/MCP/slot_y"},
                            "z": {"context_ref": "/event_facts/current_pose/pose/z"},
                            "speed": _UR5E_TRAVERSE_SPEED,
                        },
                    },
                    {
                        "primitive": "move_pose",
                        "params": {
                            "x": {"context_ref": "/event_facts/place_targets/MCP/slot_x"},
                            "y": {"context_ref": "/event_facts/place_targets/MCP/slot_y"},
                            "z": {"context_ref": "/event_facts/place_targets/MCP/place_z"},
                            **_current_pose_orientation_params(fact_path="current_pose"),
                            "speed": _UR5E_CAREFUL_SPEED,
                        },
                    },
                    {"primitive": "open_gripper", "params": {}},
                    {
                        "primitive": "detach_part",
                        "params": {
                            "model_name": {"context_ref": "/parts/MCP/target/model_name"},
                            "assume_released_if_open": True,
                        },
                    },
                    {
                        "primitive": "move_relative",
                        "params": {
                            "dx": 0.0,
                            "dy": 0.0,
                            "dz": _POST_PLACE_LIFT_DZ_M,
                            "speed": _UR5E_TRAVERSE_SPEED,
                        },
                    },
                    {
                        "primitive": "move_to_named_pose",
                        "params": {
                            "pose_name": {
                                "context_ref": "/resources/ur5e@localhost/named_poses/home",
                            },
                            "speed": _UR5E_HOME_SPEED,
                        },
                    },
                ],
            },
        ],
    }
    if primary_obligation:
        proposal["primary_obligation"] = primary_obligation
    return proposal


def _select_primary_obligation(
    prepared_bridge_request: dict[str, Any],
    *,
    preferred_resource_jid: str,
) -> dict[str, Any] | None:
    targets = prepared_bridge_request.get("obligation_targets") or []
    if not isinstance(targets, list):
        raise ValueError("prepared bridge request is missing obligation_targets")
    normalized_targets: list[dict[str, Any]] = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        rule_id = str(target.get("rule_id", "")).strip()
        resource_jid = str(target.get("resource_jid", "")).strip()
        if not rule_id or not resource_jid:
            continue
        normalized_targets.append(
            {
                "rule_id": rule_id,
                "resource_jid": resource_jid,
            }
        )

    if not normalized_targets:
        return None

    preferred_matches = [
        target
        for target in normalized_targets
        if str(target.get("resource_jid", "")).strip() == preferred_resource_jid
    ]
    if len(preferred_matches) == 1:
        return preferred_matches[0]
    if len(preferred_matches) > 1:
        raise ValueError(
            f"multiple active obligation targets matched required resource '{preferred_resource_jid}'"
        )
    if len(normalized_targets) == 1:
        return normalized_targets[0]
    raise ValueError(
        f"no active obligation target matched required resource '{preferred_resource_jid}'"
    )


def _recover_part_origin_location(
    prepared_bridge_request: dict[str, Any],
    *,
    part_name: str,
    preferred_resource_jid: str,
) -> str:
    tracker = dict(prepared_bridge_request.get("part_tracker") or {})
    part_entry = dict(tracker.get(part_name) or {})
    direct_keys = (
        "origin_resource_location",
        "origin_location",
        "origin",
        "source_location",
    )
    for key in direct_keys:
        candidate = str(part_entry.get(key) or "").strip()
        if candidate:
            return candidate

    plan_nodes = [
        dict(node)
        for node in (prepared_bridge_request.get("plan_nodes") or [])
        if isinstance(node, dict)
    ]
    if not plan_nodes:
        raise ValueError(
            f"could not recover {part_name} origin_resource_location: prepared bridge request has no plan_nodes"
        )

    task_lookup = {
        str(node.get("id", "")).strip(): node
        for node in plan_nodes
        if str(node.get("id", "")).strip()
    }
    candidate_locations: list[str] = []
    last_successful_task = str(part_entry.get("last_successful_task") or "").strip()
    if last_successful_task:
        candidate_locations.extend(
            _origin_candidates_from_task_lookup(
                task_lookup,
                task_id=last_successful_task,
                part_name=part_name,
            )
        )

    if not candidate_locations:
        candidate_locations.extend(
            _origin_candidates_from_plan(
                plan_nodes,
                part_name=part_name,
                resource_jid=preferred_resource_jid,
            )
        )

    deduped = []
    for candidate in candidate_locations:
        value = str(candidate or "").strip()
        if value and value not in deduped:
            deduped.append(value)
    if len(deduped) == 1:
        return deduped[0]

    raise ValueError(
        f"could not recover deterministic {part_name} origin_resource_location from the prepared bridge request"
    )


def _recover_resume_task_ids(
    prepared_bridge_request: dict[str, Any],
    *,
    part_name: str,
    preferred_resource_jid: str,
) -> list[str]:
    tracker = dict(prepared_bridge_request.get("part_tracker") or {})
    part_entry = dict(tracker.get(part_name) or {})
    last_successful_task = str(part_entry.get("last_successful_task") or "").strip()
    if not last_successful_task:
        return _pending_task_ids_for_resource(
            prepared_bridge_request,
            resource_jid=preferred_resource_jid,
        )

    plan_nodes = [
        dict(node)
        for node in (prepared_bridge_request.get("plan_nodes") or [])
        if isinstance(node, dict)
    ]
    task_lookup = {
        str(node.get("id", "")).strip(): node
        for node in plan_nodes
        if str(node.get("id", "")).strip()
    }
    last_task = task_lookup.get(last_successful_task) or {}
    requirement_id = _task_requirement_id(last_task) or _task_requirement_id({"id": last_successful_task})
    try:
        last_sequence_index = int(last_task.get("sequence_index"))
    except (TypeError, ValueError):
        last_sequence_index = -1

    resume_candidates: list[tuple[int, str]] = []
    for node in plan_nodes:
        task_id = str(node.get("id", "")).strip()
        if not task_id or task_id == last_successful_task:
            continue
        if str(node.get("resource_jid", "")).strip() != preferred_resource_jid:
            continue
        if requirement_id and _task_requirement_id(node) != requirement_id:
            continue
        try:
            sequence_index = int(node.get("sequence_index"))
        except (TypeError, ValueError):
            sequence_index = 10**9
        if last_sequence_index >= 0 and sequence_index <= last_sequence_index:
            continue
        resume_candidates.append((sequence_index, task_id))

    deduped = [task_id for _sequence_index, task_id in sorted(resume_candidates)]
    if deduped:
        return deduped
    return _pending_task_ids_for_resource(
        prepared_bridge_request,
        resource_jid=preferred_resource_jid,
    )


def _pending_task_ids_for_resource(
    prepared_bridge_request: dict[str, Any],
    *,
    resource_jid: str,
) -> list[str]:
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    resource_entry = dict(bridge_resources.get(resource_jid) or {})
    pending_tasks = resource_entry.get("pending_tasks") or []
    deduped: list[str] = []
    for task in pending_tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id", "")).strip()
        if task_id and task_id not in deduped:
            deduped.append(task_id)
    return deduped


def _task_requirement_id(task: dict[str, Any]) -> str:
    requirement_id = str(task.get("requirement_id") or "").strip()
    if requirement_id:
        return requirement_id
    task_id = str(task.get("id", "") or "").strip()
    if "_T" in task_id:
        return task_id.rsplit("_T", 1)[0]
    return ""


def _origin_candidates_from_task_lookup(
    task_lookup: dict[str, dict[str, Any]],
    *,
    task_id: str,
    part_name: str,
) -> list[str]:
    task = task_lookup.get(str(task_id or "").strip())
    if not isinstance(task, dict):
        return []
    candidates = _origin_candidates_from_task(task, part_name=part_name)
    if candidates:
        return candidates

    requirement_id = str(task.get("requirement_id") or "").strip()
    if not requirement_id:
        task_id_text = str(task.get("id", "")).strip()
        if "_T" in task_id_text:
            requirement_id = task_id_text.split("_T", 1)[0]
    if not requirement_id:
        return []

    requirement_matches = [
        node
        for node in task_lookup.values()
        if str(node.get("requirement_id", "")).strip() == requirement_id
    ]
    candidates = []
    for node in requirement_matches:
        candidates.extend(_origin_candidates_from_task(node, part_name=part_name))
    return candidates


def _origin_candidates_from_plan(
    plan_nodes: list[dict[str, Any]],
    *,
    part_name: str,
    resource_jid: str,
) -> list[str]:
    candidates: list[str] = []
    for node in plan_nodes:
        if str(node.get("resource_jid", "")).strip() != resource_jid:
            continue
        candidates.extend(_origin_candidates_from_task(node, part_name=part_name))
    return candidates


def _origin_candidates_from_task(task: dict[str, Any], *, part_name: str) -> list[str]:
    params = dict(task.get("params") or {})
    task_part = str(
        task.get("part_name")
        or params.get("part_name")
        or ""
    ).strip()
    if task_part and task_part != part_name:
        return []
    candidate = str(params.get("origin_resource_location") or "").strip()
    return [candidate] if candidate else []


def _part_model_name(grounding_context: dict[str, Any], part_name: str) -> str:
    parts = dict(grounding_context.get("parts") or {})
    target = dict((parts.get(part_name) or {}).get("target") or {})
    model_name = str(target.get("model_name") or "").strip()
    if not model_name:
        raise ValueError(f"grounding context is missing target.model_name for {part_name}")
    return model_name


def _part_pick_geometry(grounding_context: dict[str, Any], *, part_name: str) -> dict[str, Any]:
    parts = dict(grounding_context.get("parts") or {})
    target = dict((parts.get(part_name) or {}).get("target") or {})
    board_top_z = target.get("board_top_z")
    slot_pose = dict(target.get("slot_pose") or {})
    try:
        board_z = float(
            board_top_z
            if board_top_z is not None
            else slot_pose.get("z", 1.02)
        )
    except (TypeError, ValueError):
        board_z = 1.02

    geometry = {
        "board_center": {"x": 0.0, "y": 0.0, "z": board_z},
        "slot_floor_z_m": board_z,
    }
    slot_x = slot_pose.get("x")
    slot_y = slot_pose.get("y")
    try:
        if slot_x is not None and slot_y is not None:
            geometry["slot_xy"] = [float(slot_x), float(slot_y)]
    except (TypeError, ValueError):
        pass
    part_height = target.get("part_height")
    if part_height is not None:
        try:
            geometry["part_height_m"] = float(part_height)
        except (TypeError, ValueError):
            pass
    model_name = str(target.get("model_name") or "").strip()
    if model_name:
        geometry["model_name"] = model_name
    return geometry


def _part_orientation_params(
    grounding_context: dict[str, Any],
    *,
    part_name: str,
    fact_path: str,
    fallback: dict[str, float],
) -> dict[str, Any]:
    parts = dict(grounding_context.get("parts") or {})
    observed_pose = dict((parts.get(part_name) or {}).get("observed_pose") or {})
    if {"qx", "qy", "qz", "qw"} <= set(observed_pose.keys()):
        return {
            "qx": {"context_ref": f"/event_facts/{fact_path.replace('.', '/')}/pose/qx"},
            "qy": {"context_ref": f"/event_facts/{fact_path.replace('.', '/')}/pose/qy"},
            "qz": {"context_ref": f"/event_facts/{fact_path.replace('.', '/')}/pose/qz"},
            "qw": {"context_ref": f"/event_facts/{fact_path.replace('.', '/')}/pose/qw"},
        }
    return dict(fallback)


def _part_observed_pose(
    prepared_bridge_request: dict[str, Any],
    *,
    grounding_context: dict[str, Any],
    part_name: str,
) -> dict[str, float] | None:
    preprogrammed_observations = dict(
        prepared_bridge_request.get("preprogrammed_part_observations") or {}
    )
    preferred_pose = preprogrammed_observations.get(part_name)
    if isinstance(preferred_pose, dict) and {"x", "y", "z"} <= set(preferred_pose.keys()):
        try:
            return {
                "x": float(preferred_pose["x"]),
                "y": float(preferred_pose["y"]),
                "z": float(preferred_pose["z"]),
            }
        except (TypeError, ValueError):
            pass

    parts = dict(grounding_context.get("parts") or {})
    observed_pose = dict((parts.get(part_name) or {}).get("observed_pose") or {})
    if {"x", "y", "z"} <= set(observed_pose.keys()):
        try:
            return {
                "x": float(observed_pose["x"]),
                "y": float(observed_pose["y"]),
                "z": float(observed_pose["z"]),
            }
        except (TypeError, ValueError):
            return None
    return None


def _current_pose_orientation_params(*, fact_path: str) -> dict[str, Any]:
    return {
        "qx": {"context_ref": f"/event_facts/{fact_path.replace('.', '/')}/pose/qx"},
        "qy": {"context_ref": f"/event_facts/{fact_path.replace('.', '/')}/pose/qy"},
        "qz": {"context_ref": f"/event_facts/{fact_path.replace('.', '/')}/pose/qz"},
        "qw": {"context_ref": f"/event_facts/{fact_path.replace('.', '/')}/pose/qw"},
    }

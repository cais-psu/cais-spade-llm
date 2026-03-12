from __future__ import annotations

from typing import Any


_RECOVER_LCP_SIDEWAYS_V1 = "recover_lcp_sideways_v1"
_SIDEWAYS_QUATERNION = {
    "qx": 0.70710678,
    "qy": 0.0,
    "qz": 0.0,
    "qw": 0.70710678,
}


def build_preprogrammed_bridge_proposal(
    *,
    scenario_id: str,
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    scenario_key = str(scenario_id or "").strip()
    if scenario_key != _RECOVER_LCP_SIDEWAYS_V1:
        raise ValueError(f"unsupported preprogrammed bridge scenario '{scenario_key}'")
    if not isinstance(prepared_bridge_request, dict) or not prepared_bridge_request:
        raise ValueError("prepared bridge request is missing")
    return _build_recover_lcp_sideways_v1(prepared_bridge_request)


def _build_recover_lcp_sideways_v1(prepared_bridge_request: dict[str, Any]) -> dict[str, Any]:
    grounding_context = dict(prepared_bridge_request.get("grounding_context") or {})
    parts = dict(grounding_context.get("parts") or {})
    if "MCP" not in parts or "LCP" not in parts:
        raise ValueError("prepared bridge request is missing MCP/LCP grounding context")

    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    if "ur5e@localhost" not in bridge_resources or "xarm6@localhost" not in bridge_resources:
        raise ValueError("recover_lcp_sideways_v1 requires ur5e@localhost and xarm6@localhost")

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
    _part_model_name(grounding_context, "LCP")

    lcp_pick_q = _part_orientation_params(
        grounding_context,
        part_name="LCP",
        alias="detected_lcp",
        fallback=_SIDEWAYS_QUATERNION,
    )
    mcp_pick_q = _current_pose_orientation_params(alias="ur5e_home_pose")

    return {
        "primary_obligation": primary_obligation,
        "macro_tasks": [
            {
                "resource_jid": "xarm6@localhost",
                "macro_name": "clear_xarm6_zone",
                "description": "Retreat xarm6 to the recovery-clear pose to free the shared workspace.",
                "rationale": "Clears the collision zone before ur5e performs the sideways recovery.",
                "expected_start_state": "recovery_required",
                "task_params": {
                    "clear_pose": {"context_ref": "/resources/xarm6@localhost/named_poses/home"},
                },
                "task_metadata": {
                    "in_state": "recovery_required",
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
                            }
                        },
                    }
                ],
            },
            {
                "resource_jid": "ur5e@localhost",
                "macro_name": "return_mcp_to_printer",
                "description": "Release MCP back to its printer origin so ur5e can recover LCP first.",
                "rationale": "Unloads MCP and restores ur5e to an empty-gripper state for the sideways LCP pick.",
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
                    {"primitive": "open_gripper", "params": {}},
                    {
                        "primitive": "detach_part",
                        "params": {
                            "model_name": {"context_ref": "/parts/MCP/target/model_name"},
                        },
                    },
                ],
            },
            {
                "resource_jid": "ur5e@localhost",
                "macro_name": "pick_lcp_sideways",
                "description": "Approach the tipped LCP from the side and secure it with ur5e.",
                "rationale": "Transfers LCP recovery to ur5e using an orientation-aware side pickup.",
                "expected_start_state": "idle",
                "part_name": "LCP",
                "task_params": {
                    "origin_resource_location": {"context_ref": "/parts/LCP/location"},
                    "recovery_style": "sideways",
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
                "primitive_steps": [
                    {
                        "primitive": "detect_parts",
                        "params": {"part_name": "LCP"},
                        "store_as": "detected_lcp",
                    },
                    {
                        "primitive": "move_to_named_pose",
                        "params": {
                            "pose_name": {
                                "context_ref": "/resources/ur5e@localhost/named_poses/side_pick_approach",
                            }
                        },
                    },
                    {
                        "primitive": "move_pose",
                        "params": {
                            "x": {"context_ref": "/step_outputs/detected_lcp/pose/x"},
                            "y": {"context_ref": "/step_outputs/detected_lcp/pose/y"},
                            "z": {"context_ref": "/step_outputs/detected_lcp/pose/z"},
                            **lcp_pick_q,
                        },
                    },
                    {"primitive": "close_gripper", "params": {}},
                    {
                        "primitive": "attach_part",
                        "params": {
                            "model_name": {"context_ref": "/parts/LCP/target/model_name"},
                        },
                    },
                    {
                        "primitive": "move_relative",
                        "params": {"dx": 0.0, "dy": 0.0, "dz": 0.04},
                    },
                ],
            },
            {
                "resource_jid": "ur5e@localhost",
                "macro_name": "insert_lcp_sideways",
                "description": "Carry the recovered LCP into the board with the sideways insertion posture.",
                "rationale": "Completes the special-case LCP assembly so the normal MCP suffix can resume.",
                "expected_start_state": "picked",
                "part_name": "LCP",
                "task_params": {
                    "destination_location": "assembly_board-v1",
                    "recovery_style": "sideways",
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
                        "primitive": "move_to_named_pose",
                        "params": {
                            "pose_name": {
                                "context_ref": "/resources/ur5e@localhost/named_poses/side_insert_pre",
                            }
                        },
                    },
                    {
                        "primitive": "move_pose",
                        "params": {
                            "x": {"context_ref": "/parts/LCP/target/slot_pose/x"},
                            "y": {"context_ref": "/parts/LCP/target/slot_pose/y"},
                            "z": {"context_ref": "/parts/LCP/target/slot_pose/z"},
                            **_SIDEWAYS_QUATERNION,
                        },
                    },
                    {"primitive": "open_gripper", "params": {}},
                    {
                        "primitive": "detach_part",
                        "params": {
                            "model_name": {"context_ref": "/parts/LCP/target/model_name"},
                        },
                    },
                    {
                        "primitive": "move_to_named_pose",
                        "params": {
                            "pose_name": {
                                "context_ref": "/resources/ur5e@localhost/named_poses/side_insert_post",
                            }
                        },
                    },
                ],
            },
            {
                "resource_jid": "ur5e@localhost",
                "macro_name": "resume_mcp_assembly",
                "description": "Repick MCP into ur5e so the existing MCP assembly suffix can resume.",
                "rationale": "Restores the MCP flow to the normal picked state expected by the pending DES tasks.",
                "expected_start_state": "idle",
                "part_name": "MCP",
                "task_params": {
                    "origin_resource_location": mcp_origin,
                    "resume_mode": "normal_suffix",
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
                "primitive_steps": [
                    {
                        "primitive": "move_to_named_pose",
                        "params": {
                            "pose_name": {
                                "context_ref": "/resources/ur5e@localhost/named_poses/home",
                            }
                        },
                    },
                    {
                        "primitive": "get_current_pose",
                        "params": {},
                        "store_as": "ur5e_home_pose",
                    },
                    {
                        "primitive": "detect_parts",
                        "params": {"part_name": "MCP"},
                        "store_as": "detected_mcp",
                    },
                    {
                        "primitive": "move_pose",
                        "params": {
                            "x": {"context_ref": "/step_outputs/detected_mcp/pose/x"},
                            "y": {"context_ref": "/step_outputs/detected_mcp/pose/y"},
                            "z": {"context_ref": "/step_outputs/detected_mcp/pose/z"},
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
                        "params": {"dx": 0.0, "dy": 0.0, "dz": 0.04},
                    },
                ],
            },
        ],
    }


def _select_primary_obligation(
    prepared_bridge_request: dict[str, Any],
    *,
    preferred_resource_jid: str,
) -> dict[str, Any]:
    targets = prepared_bridge_request.get("obligation_targets") or []
    if not isinstance(targets, list):
        raise ValueError("prepared bridge request is missing obligation_targets")
    for target in targets:
        if (
            isinstance(target, dict)
            and str(target.get("resource_jid", "")).strip() == preferred_resource_jid
        ):
            return {
                "rule_id": str(target.get("rule_id", "")).strip(),
                "resource_jid": preferred_resource_jid,
            }
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


def _part_orientation_params(
    grounding_context: dict[str, Any],
    *,
    part_name: str,
    alias: str,
    fallback: dict[str, float],
) -> dict[str, Any]:
    parts = dict(grounding_context.get("parts") or {})
    observed_pose = dict((parts.get(part_name) or {}).get("observed_pose") or {})
    if {"qx", "qy", "qz", "qw"} <= set(observed_pose.keys()):
        return {
            "qx": {"context_ref": f"/step_outputs/{alias}/pose/qx"},
            "qy": {"context_ref": f"/step_outputs/{alias}/pose/qy"},
            "qz": {"context_ref": f"/step_outputs/{alias}/pose/qz"},
            "qw": {"context_ref": f"/step_outputs/{alias}/pose/qw"},
        }
    return dict(fallback)


def _current_pose_orientation_params(*, alias: str) -> dict[str, Any]:
    return {
        "qx": {"context_ref": f"/step_outputs/{alias}/pose/qx"},
        "qy": {"context_ref": f"/step_outputs/{alias}/pose/qy"},
        "qz": {"context_ref": f"/step_outputs/{alias}/pose/qz"},
        "qw": {"context_ref": f"/step_outputs/{alias}/pose/qw"},
    }

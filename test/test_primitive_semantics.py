from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cais_spade_llm.agents.intelligent_product.replanner.environment_model import (
    _normalize_primitive_bridge_proposal,
    llm_explore_states_and_events,
)
from cais_spade_llm.agents.intelligent_product.replanner.primitive_semantics import (
    apply_effects_to_snapshot,
    build_primitive_catalog,
    extract_step_output,
    get_robot_bridge_snapshot,
    sync_agent_from_bridge_snapshot,
    validate_and_project_steps,
)


class _DummyRobot:
    _BRIDGE_PRIMITIVES = frozenset(
        {
            "move_cartesian",
            "move_pose",
            "move_relative",
            "move_to_named_pose",
            "rotate_wrist",
            "open_gripper",
            "close_gripper",
            "detect_parts",
            "compute_pick_targets",
            "compute_place_targets",
            "attach_part",
            "detach_part",
            "get_current_pose",
        }
    )

    def __init__(self) -> None:
        self.agent_name = "ur5e"
        self._controller = None
        self.execution_mode = "dry_run"
        self._current_state = "picked"
        self._held_part = "MRP"
        self._gripper_state = "closed"
        self._position = {"x": 1.0, "y": 2.0, "z": 3.0}
        self._bridge_pose_ref = None
        self.named_positions = {"home": [0, 1, 2, 3, 4, 5]}

    def _robot_scope_name(self) -> str:
        return "ur5e"

    def get_bridge_snapshot(self) -> dict:
        return get_robot_bridge_snapshot(self)


def test_build_primitive_catalog_includes_preconditions_and_effects():
    catalog = build_primitive_catalog(_DummyRobot())
    by_name = {entry["name"]: entry for entry in catalog}

    attach = by_name["attach_part"]
    assert attach["preconditions"]["held_part"]["equals"] is None
    assert attach["effects"]["held_part"]["set_from_param"] == "model_name"
    assert "held_part must equal None" in attach["semantic_summary"]

    open_gripper = by_name["open_gripper"]
    assert open_gripper["effects"]["gripper_state"]["set"] == "open"

    move_pose = by_name["move_pose"]
    assert move_pose["required_params"] == ["x", "y", "z", "qx", "qy", "qz", "qw"]
    assert move_pose["effects"]["current_pose"]["pose_absolute_from_params"] == ["x", "y", "z"]


def test_validate_and_project_steps_rejects_attach_when_holding_part():
    robot = _DummyRobot()
    catalog = build_primitive_catalog(robot)
    ok, projected, error = validate_and_project_steps(
        [{"primitive": "attach_part", "params": {"model_name": "LCP"}}],
        catalog,
        robot.get_bridge_snapshot(),
    )

    assert ok is False
    assert projected["held_part"] == "MRP"
    assert "held_part" in str(error)


def test_validate_and_project_steps_rejects_move_relative_without_pose():
    robot = _DummyRobot()
    snapshot = robot.get_bridge_snapshot()
    snapshot["current_pose"] = None

    ok, _projected, error = validate_and_project_steps(
        [{"primitive": "move_relative", "params": {"dx": 0.1, "dy": 0.0, "dz": 0.2}}],
        build_primitive_catalog(robot),
        snapshot,
    )

    assert ok is False
    assert "current_pose" in str(error)


def test_validate_and_project_steps_projects_gripper_and_held_part_state():
    robot = _DummyRobot()
    ok, projected, error = validate_and_project_steps(
        [
            {"primitive": "open_gripper", "params": {}},
            {"primitive": "detach_part", "params": {}},
        ],
        build_primitive_catalog(robot),
        robot.get_bridge_snapshot(),
    )

    assert ok is True
    assert error is None
    assert projected["gripper_state"] == "open"
    assert projected["held_part"] is None


def test_validate_and_project_steps_accepts_move_pose_and_projects_xyz_only():
    robot = _DummyRobot()
    ok, projected, error = validate_and_project_steps(
        [
            {
                "primitive": "move_pose",
                "params": {
                    "x": 0.4,
                    "y": -0.1,
                    "z": 0.8,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.7071,
                    "qw": 0.7071,
                },
            }
        ],
        build_primitive_catalog(robot),
        robot.get_bridge_snapshot(),
    )

    assert ok is True
    assert error is None
    assert projected["current_pose"] == {"x": 0.4, "y": -0.1, "z": 0.8}


def test_normalize_primitive_bridge_proposal_rejects_semantically_invalid_steps():
    robot = _DummyRobot()
    proposal = _normalize_primitive_bridge_proposal(
        raw=(
            '{"macro_name":"bad_attach","resource_jid":"ur5e@localhost",'
            '"expected_start_state":"picked","primitive_steps":'
            '[{"primitive":"attach_part","params":{"model_name":"LCP"}}]}'
        ),
        ra_jid="ur5e@localhost",
        primitive_catalog=build_primitive_catalog(robot),
        bridge_snapshot=robot.get_bridge_snapshot(),
        grounding_context={},
    )

    assert proposal is None


def test_normalize_primitive_bridge_proposal_resolves_context_refs_before_validation():
    robot = _DummyRobot()
    proposal = _normalize_primitive_bridge_proposal(
        raw=json.dumps(
            {
                "macro_name": "move_to_grounded_pose",
                "resource_jid": "ur5e@localhost",
                "expected_start_state": "picked",
                "primitive_steps": [
                    {
                        "primitive": "move_cartesian",
                        "params": {
                            "x": {"context_ref": "/resource/current_pose/x"},
                            "y": {"context_ref": "/resource/current_pose/y"},
                            "z": {"context_ref": "/resource/current_pose/z"},
                        },
                    }
                ],
            }
        ),
        ra_jid="ur5e@localhost",
        primitive_catalog=build_primitive_catalog(robot),
        bridge_snapshot=robot.get_bridge_snapshot(),
        grounding_context={
            "resource": {
                "current_pose": {"x": 1.0, "y": 2.0, "z": 3.0},
            }
        },
    )

    assert proposal is not None
    assert proposal["primitive_steps"] == [
        {"primitive": "move_cartesian", "params": {"x": 1.0, "y": 2.0, "z": 3.0}}
    ]


def test_normalize_primitive_bridge_proposal_rejects_unknown_context_ref():
    robot = _DummyRobot()
    proposal = _normalize_primitive_bridge_proposal(
        raw=json.dumps(
            {
                "macro_name": "bad_ref",
                "resource_jid": "ur5e@localhost",
                "expected_start_state": "picked",
                "primitive_steps": [
                    {
                        "primitive": "move_to_named_pose",
                        "params": {
                            "pose_name": {"context_ref": "/resource/named_poses/missing"},
                        },
                    }
                ],
            }
        ),
        ra_jid="ur5e@localhost",
        primitive_catalog=build_primitive_catalog(robot),
        bridge_snapshot=robot.get_bridge_snapshot(),
        grounding_context={"resource": {"named_poses": {"home": "home"}}},
    )

    assert proposal is None


def test_validate_and_project_steps_supports_store_as_and_step_output_refs():
    robot = _DummyRobot()
    ok, projected, error = validate_and_project_steps(
        [
            {
                "primitive": "detect_parts",
                "params": {"part_name": "SG"},
                "store_as": "detected_sg",
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": {"context_ref": "/step_outputs/detected_sg/pose/x"},
                    "y": {"context_ref": "/step_outputs/detected_sg/pose/y"},
                    "z": {"context_ref": "/step_outputs/detected_sg/pose/z"},
                },
            },
        ],
        build_primitive_catalog(robot),
        robot.get_bridge_snapshot(),
        grounding_context={
            "parts": {
                "SG": {
                    "observed_pose": {"x": 0.31, "y": -0.12, "z": 0.15},
                    "target": {"model_name": "spur_gear"},
                }
            }
        },
    )

    assert ok is True
    assert error is None
    assert projected["current_pose"] == {"x": 0.31, "y": -0.12, "z": 0.15}


def test_validate_and_project_steps_supports_compute_place_targets_without_pick_ctx():
    robot = _DummyRobot()
    ok, projected, error = validate_and_project_steps(
        [
            {
                "primitive": "compute_place_targets",
                "params": {
                    "part_name": "LCP",
                    "product_geometry": {
                        "slot_xy": [0.1, -0.08],
                        "slot_floor_z_m": 1.025,
                        "board_center": {"z": 1.025},
                        "part_height_m": 0.1,
                        "model_name": "circ_pin_large",
                    },
                },
                "store_as": "lcp_place_targets",
            },
            {
                "primitive": "move_pose",
                "params": {
                    "x": {"context_ref": "/step_outputs/lcp_place_targets/slot_x"},
                    "y": {"context_ref": "/step_outputs/lcp_place_targets/slot_y"},
                    "z": {"context_ref": "/step_outputs/lcp_place_targets/place_z"},
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                },
            },
        ],
        build_primitive_catalog(robot),
        robot.get_bridge_snapshot(),
    )

    assert ok is True
    assert error is None
    assert math.isclose(projected["current_pose"]["x"], 0.1, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(projected["current_pose"]["y"], -0.08, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(projected["current_pose"]["z"], 1.2525, rel_tol=0.0, abs_tol=1e-9)


def test_normalize_primitive_bridge_proposal_accepts_orientation_refs_from_detect_parts_store_as():
    robot = _DummyRobot()
    proposal = _normalize_primitive_bridge_proposal(
        raw=json.dumps(
            {
                "macro_name": "side_pick",
                "resource_jid": "ur5e@localhost",
                "expected_start_state": "picked",
                "primitive_steps": [
                    {
                        "primitive": "detect_parts",
                        "params": {"part_name": "LCP"},
                        "store_as": "detected_lcp",
                    },
                    {
                        "primitive": "move_pose",
                        "params": {
                            "x": {"context_ref": "/step_outputs/detected_lcp/pose/x"},
                            "y": {"context_ref": "/step_outputs/detected_lcp/pose/y"},
                            "z": {"context_ref": "/step_outputs/detected_lcp/pose/z"},
                            "qx": {"context_ref": "/step_outputs/detected_lcp/pose/qx"},
                            "qy": {"context_ref": "/step_outputs/detected_lcp/pose/qy"},
                            "qz": {"context_ref": "/step_outputs/detected_lcp/pose/qz"},
                            "qw": {"context_ref": "/step_outputs/detected_lcp/pose/qw"},
                        },
                    },
                ],
            }
        ),
        ra_jid="ur5e@localhost",
        primitive_catalog=build_primitive_catalog(robot),
        bridge_snapshot=robot.get_bridge_snapshot(),
        grounding_context={
            "parts": {
                "LCP": {
                    "observed_pose": {
                        "x": 0.41,
                        "y": -0.18,
                        "z": 0.12,
                        "qx": 0.0,
                        "qy": 0.7071,
                        "qz": 0.0,
                        "qw": 0.7071,
                    },
                    "target": {"model_name": "circ_pin_large"},
                }
            }
        },
    )

    assert proposal is not None
    assert proposal["projected_snapshot"]["current_pose"] == {"x": 0.41, "y": -0.18, "z": 0.12}
    assert proposal["primitive_steps"][1]["params"]["qx"] == {
        "context_ref": "/step_outputs/detected_lcp/pose/qx"
    }


def test_normalize_primitive_bridge_proposal_accepts_orientation_refs_from_get_current_pose_store_as():
    robot = _DummyRobot()
    proposal = _normalize_primitive_bridge_proposal(
        raw=json.dumps(
            {
                "macro_name": "resume_pick",
                "resource_jid": "ur5e@localhost",
                "expected_start_state": "picked",
                "primitive_steps": [
                    {
                        "primitive": "get_current_pose",
                        "params": {},
                        "store_as": "home_pose",
                    },
                    {
                        "primitive": "move_pose",
                        "params": {
                            "x": 0.2,
                            "y": 0.1,
                            "z": 0.5,
                            "qx": {"context_ref": "/step_outputs/home_pose/pose/qx"},
                            "qy": {"context_ref": "/step_outputs/home_pose/pose/qy"},
                            "qz": {"context_ref": "/step_outputs/home_pose/pose/qz"},
                            "qw": {"context_ref": "/step_outputs/home_pose/pose/qw"},
                        },
                    },
                ],
            }
        ),
        ra_jid="ur5e@localhost",
        primitive_catalog=build_primitive_catalog(robot),
        bridge_snapshot=robot.get_bridge_snapshot(),
        grounding_context={"resource": {"current_pose": {"x": 1.0, "y": 2.0, "z": 3.0}}},
    )

    assert proposal is not None
    assert proposal["projected_snapshot"]["current_pose"] == {"x": 0.2, "y": 0.1, "z": 0.5}
    assert proposal["primitive_steps"][1]["params"]["qw"] == {
        "context_ref": "/step_outputs/home_pose/pose/qw"
    }


def test_extract_step_output_preserves_detected_part_orientation():
    output, error = extract_step_output(
        primitive="detect_parts",
        params={"part_name": "LCP"},
        step_result={
            "data": [
                {
                    "part_name": "LCP",
                    "model_name": "circ_pin_large",
                    "x": 0.12,
                    "y": -0.05,
                    "z": 1.02,
                    "qx": 0.0,
                    "qy": 0.7071,
                    "qz": 0.0,
                    "qw": 0.7071,
                }
            ]
        },
    )

    assert error is None
    assert output is not None
    assert output["pose"]["qx"] == 0.0
    assert output["pose"]["qy"] == 0.7071
    assert output["orientation"]["qw"] == 0.7071


def test_normalize_primitive_bridge_proposal_rejects_unknown_step_output_ref():
    robot = _DummyRobot()
    proposal = _normalize_primitive_bridge_proposal(
        raw=json.dumps(
            {
                "macro_name": "bad_step_output_ref",
                "resource_jid": "ur5e@localhost",
                "expected_start_state": "picked",
                "primitive_steps": [
                    {
                        "primitive": "detect_parts",
                        "params": {"part_name": "SG"},
                        "store_as": "detected_sg",
                    },
                    {
                        "primitive": "move_cartesian",
                        "params": {
                            "x": {"context_ref": "/step_outputs/missing_alias/pose/x"},
                            "y": {"context_ref": "/step_outputs/detected_sg/pose/y"},
                            "z": {"context_ref": "/step_outputs/detected_sg/pose/z"},
                        },
                    },
                ],
            }
        ),
        ra_jid="ur5e@localhost",
        primitive_catalog=build_primitive_catalog(robot),
        bridge_snapshot=robot.get_bridge_snapshot(),
        grounding_context={
            "parts": {
                "SG": {
                    "observed_pose": {"x": 0.31, "y": -0.12, "z": 0.15},
                    "target": {"model_name": "spur_gear"},
                }
            }
        },
    )

    assert proposal is None


def test_normalize_primitive_bridge_proposal_preserves_part_name_and_task_params():
    robot = _DummyRobot()
    proposal = _normalize_primitive_bridge_proposal(
        raw=json.dumps(
            {
                "macro_name": "stash_held_part",
                "resource_jid": "ur5e@localhost",
                "expected_start_state": "picked",
                "part_name": "MCP",
                "task_params": {
                    "destination_location": {"context_ref": "/locations/prusa_mk4_2/name"},
                },
                "task_metadata": {
                    "in_state": "picked",
                    "out_state": "idle",
                    "required_context_keys": ["destination_location"],
                    "context_mapping": {"location_param": "destination_location"},
                    "part_transition": {
                        "completed": {
                            "state": "ready",
                            "location_param": "destination_location",
                        }
                    },
                },
                "primitive_steps": [
                    {"primitive": "open_gripper", "params": {}},
                ],
            }
        ),
        ra_jid="ur5e@localhost",
        primitive_catalog=build_primitive_catalog(robot),
        bridge_snapshot=robot.get_bridge_snapshot(),
        grounding_context={"locations": {"prusa_mk4_2": {"name": "prusa-mk4-2"}}},
    )

    assert proposal is not None
    assert proposal["part_name"] == "MCP"
    assert proposal["task_params"] == {"destination_location": "prusa-mk4-2"}
    assert proposal["projected_part_entry"]["state"] == "ready"
    assert proposal["projected_part_entry"]["location"] == "prusa-mk4-2"


def test_normalize_primitive_bridge_proposal_accepts_legacy_touched_part_alias():
    robot = _DummyRobot()
    proposal = _normalize_primitive_bridge_proposal(
        raw=json.dumps(
            {
                "macro_name": "legacy_stash",
                "resource_jid": "ur5e@localhost",
                "expected_start_state": "picked",
                "touched_part": "MCP",
                "primitive_steps": [
                    {"primitive": "open_gripper", "params": {}},
                ],
            }
        ),
        ra_jid="ur5e@localhost",
        primitive_catalog=build_primitive_catalog(robot),
        bridge_snapshot=robot.get_bridge_snapshot(),
        grounding_context={},
    )

    assert proposal is not None
    assert proposal["part_name"] == "MCP"


def test_normalize_primitive_bridge_proposal_rejects_missing_part_transition_task_param():
    robot = _DummyRobot()
    proposal = _normalize_primitive_bridge_proposal(
        raw=json.dumps(
            {
                "macro_name": "bad_stash",
                "resource_jid": "ur5e@localhost",
                "expected_start_state": "picked",
                "part_name": "MCP",
                "task_metadata": {
                    "in_state": "picked",
                    "out_state": "idle",
                    "required_context_keys": ["destination_location"],
                    "part_transition": {
                        "completed": {
                            "state": "ready",
                            "location_param": "destination_location",
                        }
                    },
                },
                "primitive_steps": [
                    {"primitive": "open_gripper", "params": {}},
                ],
            }
        ),
        ra_jid="ur5e@localhost",
        primitive_catalog=build_primitive_catalog(robot),
        bridge_snapshot=robot.get_bridge_snapshot(),
        grounding_context={},
    )

    assert proposal is None


def test_normalize_primitive_bridge_proposal_accepts_ordered_macro_tasks_with_evolving_context():
    robot = _DummyRobot()
    proposal = _normalize_primitive_bridge_proposal(
        raw=json.dumps(
            {
                "primary_obligation": {
                    "rule_id": "SAFE_1",
                    "resource_jid": "ur5e@localhost",
                },
                "macro_tasks": [
                    {
                        "resource_jid": "ur5e@localhost",
                        "macro_name": "stash_held_part",
                        "expected_start_state": "picked",
                        "part_name": "MRP",
                        "task_params": {"destination_location": "stash_bin"},
                        "task_metadata": {
                            "in_state": "picked",
                            "out_state": "idle",
                            "required_context_keys": ["destination_location"],
                            "context_mapping": {"location_param": "destination_location"},
                            "part_transition": {
                                "completed": {
                                    "state": "ready",
                                    "location_param": "destination_location",
                                }
                            },
                        },
                        "primitive_steps": [
                            {"primitive": "open_gripper", "params": {}},
                            {"primitive": "detach_part", "params": {}},
                        ],
                    },
                    {
                        "resource_jid": "ur5e@localhost",
                        "macro_name": "return_home",
                        "expected_start_state": "idle",
                        "task_params": {
                            "last_stash_location": {"context_ref": "/parts/MRP/location"},
                        },
                        "task_metadata": {
                            "in_state": "idle",
                            "out_state": "idle",
                            "required_context_keys": [],
                            "context_mapping": {},
                            "part_transition": None,
                        },
                        "primitive_steps": [
                            {
                                "primitive": "move_to_named_pose",
                                "params": {
                                    "pose_name": {"context_ref": "/resource/named_poses/home"},
                                },
                            }
                        ],
                    },
                ],
            }
        ),
        ra_jid="ur5e@localhost",
        primitive_catalog=build_primitive_catalog(robot),
        bridge_snapshot=robot.get_bridge_snapshot(),
        grounding_context={},
        obligation_targets=[
            {
                "rule_id": "SAFE_1",
                "resource_jid": "ur5e@localhost",
                "required_out_states": ["idle"],
                "candidate_tools": [{"out_state": "idle"}],
            }
        ],
    )

    assert proposal is not None
    assert len(proposal["macro_tasks"]) == 2
    assert proposal["macro_tasks"][1]["task_params"]["last_stash_location"] == "stash_bin"
    assert proposal["projected_parts"]["MRP"]["location"] == "stash_bin"
    assert proposal["projected_resource_snapshots"]["ur5e@localhost"]["current_state"] == "idle"


def test_normalize_primitive_bridge_proposal_rejects_disconnected_macro_task_sequence():
    robot = _DummyRobot()
    proposal = _normalize_primitive_bridge_proposal(
        raw=json.dumps(
            {
                "macro_tasks": [
                    {
                        "resource_jid": "ur5e@localhost",
                        "macro_name": "release_grip",
                        "expected_start_state": "picked",
                        "task_metadata": {
                            "in_state": "picked",
                            "out_state": "idle",
                            "required_context_keys": [],
                            "context_mapping": {},
                            "part_transition": None,
                        },
                        "primitive_steps": [{"primitive": "open_gripper", "params": {}}],
                    },
                    {
                        "resource_jid": "ur5e@localhost",
                        "macro_name": "bad_followup",
                        "expected_start_state": "picked",
                        "task_metadata": {
                            "in_state": "picked",
                            "out_state": "idle",
                            "required_context_keys": [],
                            "context_mapping": {},
                            "part_transition": None,
                        },
                        "primitive_steps": [
                            {"primitive": "move_to_named_pose", "params": {"pose_name": "home"}},
                        ],
                    },
                ],
            }
        ),
        ra_jid="ur5e@localhost",
        primitive_catalog=build_primitive_catalog(robot),
        bridge_snapshot=robot.get_bridge_snapshot(),
        grounding_context={},
    )

    assert proposal is None


def test_normalize_primitive_bridge_proposal_rejects_final_state_that_does_not_discharge_primary_obligation():
    robot = _DummyRobot()
    proposal = _normalize_primitive_bridge_proposal(
        raw=json.dumps(
            {
                "primary_obligation": {
                    "rule_id": "SAFE_1",
                    "resource_jid": "ur5e@localhost",
                },
                "macro_tasks": [
                    {
                        "resource_jid": "ur5e@localhost",
                        "macro_name": "open_only",
                        "expected_start_state": "picked",
                        "task_metadata": {
                            "in_state": "picked",
                            "out_state": "picked",
                            "required_context_keys": [],
                            "context_mapping": {},
                            "part_transition": None,
                        },
                        "primitive_steps": [{"primitive": "open_gripper", "params": {}}],
                    }
                ],
            }
        ),
        ra_jid="ur5e@localhost",
        primitive_catalog=build_primitive_catalog(robot),
        bridge_snapshot=robot.get_bridge_snapshot(),
        grounding_context={},
        obligation_targets=[
            {
                "rule_id": "SAFE_1",
                "resource_jid": "ur5e@localhost",
                "required_out_states": ["idle"],
                "candidate_tools": [{"out_state": "idle"}],
            }
        ],
    )

    assert proposal is None


def test_llm_explore_states_and_events_accepts_primitive_catalog_and_snapshot():
    robot = _DummyRobot()
    captured_prompt: dict[str, str] = {}

    async def fake_ask_llm(*, prompt: str, with_functions: bool) -> str:
        captured_prompt["value"] = prompt
        assert with_functions is False
        return json.dumps(
            {
                "macro_name": "open_for_recovery",
                "resource_jid": "ur5e@localhost",
                "description": "Open the gripper to unblock recovery",
                "rationale": "This clears the held pose without changing resource ownership.",
                "expected_start_state": "picked",
                "task_metadata": {
                    "in_state": "picked",
                    "out_state": "picked",
                    "required_context_keys": [],
                    "context_mapping": {},
                    "part_transition": None,
                },
                "primitive_steps": [
                    {"primitive": "open_gripper", "params": {}},
                ],
            }
        )

    proposal = asyncio.run(
        llm_explore_states_and_events(
            stuck_state={"resource_state": "picked", "part_states": {"MRP": "in_gripper"}},
            P_id=["MRP"],
            ra_jid="ur5e@localhost",
            ask_llm=fake_ask_llm,
            goal_state="assembled",
            tools_catalog=[],
            resource_infos=[],
            primitive_catalog=build_primitive_catalog(robot),
            bridge_snapshot=robot.get_bridge_snapshot(),
            grounding_context={
                "resource": {
                    "current_pose": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "named_poses": {"home": "home"},
                },
                "parts": {"MRP": {"state": "in_gripper", "observed_pose": None}},
            },
        )
    )

    assert "CONTROLLER PRIMITIVES" in captured_prompt["value"]
    assert "GROUNDING CONTEXT" in captured_prompt["value"]
    assert '"part_name": "<optional canonical part name for tracking>"' in captured_prompt["value"]
    assert "task_params" in captured_prompt["value"]
    assert proposal is not None
    assert proposal["macro_name"] == "open_for_recovery"
    assert proposal["primitive_steps"] == [{"primitive": "open_gripper", "params": {}}]
    assert proposal["expected_snapshot"]["current_state"] == "picked"


def test_semantic_projection_syncs_runtime_state_back_to_agent_fields():
    robot = _DummyRobot()
    catalog = {entry["name"]: entry for entry in build_primitive_catalog(robot)}
    snapshot = robot.get_bridge_snapshot()
    snapshot = apply_effects_to_snapshot(
        {"primitive": "open_gripper", "params": {}},
        catalog["open_gripper"],
        snapshot,
    )
    snapshot = apply_effects_to_snapshot(
        {"primitive": "detach_part", "params": {}},
        catalog["detach_part"],
        snapshot,
    )
    snapshot["current_state"] = "idle"

    sync_agent_from_bridge_snapshot(robot, snapshot)

    assert robot._held_part is None
    assert robot._gripper_state == "open"
    assert robot._current_state == "idle"

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "cais_spade_llm"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from cais_spade_llm.agents.intelligent_product.replanner.environment_model import (
    _normalize_primitive_bridge_proposal,
)
from cais_spade_llm.agents.intelligent_product.replanner.primitive_semantics import (
    apply_effects_to_snapshot,
    build_primitive_catalog,
    get_robot_bridge_snapshot,
    sync_agent_from_bridge_snapshot,
    validate_and_project_steps,
)


class _DummyRobot:
    _BRIDGE_PRIMITIVES = frozenset(
        {
            "move_cartesian",
            "move_relative",
            "move_to_named_pose",
            "open_gripper",
            "close_gripper",
            "detect_parts",
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
    )

    assert proposal is None


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

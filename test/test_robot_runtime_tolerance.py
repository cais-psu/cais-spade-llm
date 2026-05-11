from __future__ import annotations

import asyncio
import logging

import pytest

from cais_spade_llm.resources.robot.robot_primitives import _preview_place_targets_output
from cais_spade_llm.resources.robot.robot_tasks import (
    execute_robot_task,
    robot_task_capability_decompositions,
)
from cais_spade_llm.resources.robot.ros2_pick_place_controller import Ros2PickPlaceController


class _FakeRobotAgent:
    def __init__(self, execution_mode: str) -> None:
        self.execution_mode = execution_mode
        self.logger = logging.getLogger(f"test.robot_task.{execution_mode}")
        self._held_part = "MG"
        self._current_state = "positioned"
        self._position = {"x": 0.0, "y": 0.08, "z": 1.0}
        self._gripper_state = "closed"
        self._bridge_pose_ref = None
        self._task_ctx = {
            "model_name": "gear_medium",
            "slot_x": 0.0,
            "slot_y": 0.08,
            "part_height": 0.015,
            "board_top_z": 1.025,
            "place_part_origin_z": 1.0325,
            "travel_z": 1.2,
        }
        self.calls: list[tuple[str, dict]] = []

    async def _maybe_inject_failure(self, **kwargs):
        return None

    async def _simulate_action(self, *args, **kwargs) -> None:
        return None

    def _task_failure(
        self,
        message: str,
        *,
        step: str,
        observations=None,
        failure_context=None,
    ) -> dict:
        return {
            "status": "failed",
            "content": str(message),
            "failure_context": {"observations": dict(observations or {})},
            "step": step,
        }

    async def _execute_primitive(self, primitive: str, params: dict) -> dict:
        self.calls.append((primitive, dict(params or {})))
        if primitive == "release_part":
            return {
                "success": True,
                "message": "released MG",
                "release_mode": "assumed_open_after_detach_timeout",
            }
        if primitive == "snap_part_to_slot":
            return {"success": False, "message": "snap failed"}
        if primitive == "move_relative":
            return {"success": False, "message": "lift failed"}
        return {"success": True, "message": f"{primitive} ok"}


class _WarnLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, message: str) -> None:
        self.messages.append(str(message))

    def info(self, message: str) -> None:
        self.messages.append(str(message))

    def warn(self, message: str) -> None:
        self.messages.append(str(message))

    def warning(self, message: str) -> None:
        self.messages.append(str(message))

    def error(self, message: str) -> None:
        self.messages.append(str(message))


def test_simulation_place_insert_completes_after_release_when_lift_soft_fails():
    agent = _FakeRobotAgent("simulation")

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            part_name="MG",
            destination_location="assembly_board-v1",
        )
    )

    assert result["status"] == "completed"
    assert [name for name, _params in agent.calls] == [
        "release_part",
        "snap_part_to_slot",
        "move_relative",
    ]
    assert agent.calls[1][1]["model_name"] == "gear_medium"
    assert agent.calls[1][1]["slot_x"] == 0.0
    assert agent.calls[1][1]["slot_y"] == 0.08
    assert agent.calls[1][1]["part_origin_z"] == 1.0325
    assert agent._held_part is None
    assert agent._current_state == "placed"
    assert agent._gripper_state == "open"
    assert agent._task_ctx == {}


def test_compute_place_targets_clamps_inserted_mg_origin_to_visible_slot_height():
    controller = Ros2PickPlaceController.__new__(Ros2PickPlaceController)
    controller.execution_mode = "simulation"
    controller.place_surface_gap_m = -0.01
    controller.insertion_depth_m = 0.0025
    controller.pick_tcp_z_bias_min_m = 0.003
    controller.pick_tcp_z_bias_max_m = 0.02
    controller.wait_for_services = lambda: True
    controller._get_ee_tcp_world_z_offset = lambda: -0.218

    result = controller.compute_place_targets(
        pick_ctx={
            "part_name": "MG",
            "model_name": "gear_medium",
            "pick_tcp_z": 1.076,
            "tz": 1.474,
            "tcp_offset_z": -0.218,
        },
        product_geometry={
            "slot_xy": [0.0, 0.08],
            "part_height_m": 0.015,
            "model_name": "gear_medium",
            "slot_floor_z_m": 1.025,
            "board_center": {"x": 0.0, "y": 0.0, "z": 1.02},
            "target_reference": {
                "source": "slot_geometry",
                "target_point": "inserted_part_origin",
                "surface_role": "assembly_slot",
            },
        },
        part_name="MG",
        destination_location="assembly_board-v1",
    )

    assert result["success"] is True
    assert result["slot_x"] == 0.0
    assert result["slot_y"] == 0.08
    assert result["place_part_origin_z"] == pytest.approx(1.0325)


def test_move_to_named_pose_uses_configured_named_pose_duration():
    controller = Ros2PickPlaceController.__new__(Ros2PickPlaceController)
    controller.named_positions = {"home": [0.0, 0.1, 0.2]}
    controller.named_pose_duration_sec = 2.0
    controller.trajectory_time_scale = 0.8
    controller._arm_pub = object()
    controller._exec_client = None
    controller.wait_for_services = lambda: True

    durations: list[float] = []

    def _publish(positions, *, duration_sec, tolerance_rad=0.08):
        durations.append(float(duration_sec))
        return True

    controller._publish_arm_joint_trajectory_and_wait = _publish

    result = controller.move_to_named_pose("home", speed=0.8)

    assert result["success"] is True
    assert durations == [pytest.approx(1.6)]


def test_move_home_uses_configured_home_duration():
    controller = Ros2PickPlaceController.__new__(Ros2PickPlaceController)
    controller.named_positions = {"home": [0.0, 0.1, 0.2]}
    controller.move_home_duration_sec = 2.0
    controller.trajectory_time_scale = 0.8
    controller._arm_pub = object()
    controller._exec_client = None
    controller._JointTrajectory = None
    controller._last_start_pose = object()
    controller.wait_for_services = lambda: True
    controller._get_arm_joint_positions = lambda timeout_sec=0.0: ([1.0, 1.0, 1.0], [])

    durations: list[float] = []

    def _publish(positions, *, duration_sec, tolerance_rad=0.08):
        durations.append(float(duration_sec))
        return True

    controller._publish_arm_joint_trajectory_and_wait = _publish

    result = controller.move_home(speed=0.5)

    assert result["success"] is True
    assert durations == [pytest.approx(1.0)]
    assert controller._last_start_pose is None


def test_move_home_capability_uses_faster_home_speed():
    decomposition = robot_task_capability_decompositions(function_name="move_home")
    steps = list(decomposition.get("bridge_visible_steps") or [])

    assert steps
    assert steps[0]["primitive"] == "move_to_named_pose"
    assert steps[0]["params"]["speed"] == pytest.approx(0.25)


def test_preview_place_targets_clamps_inserted_mg_origin_to_visible_slot_height():
    result, error = _preview_place_targets_output(
        {
            "part_name": "MG",
            "product_geometry": {
                "slot_xy": [0.0, 0.08],
                "part_height_m": 0.015,
                "model_name": "gear_medium",
                "slot_floor_z_m": 1.025,
                "board_center": {"x": 0.0, "y": 0.0, "z": 1.02},
                "target_reference": {
                    "source": "slot_geometry",
                    "target_point": "inserted_part_origin",
                    "surface_role": "assembly_slot",
                },
            },
            "pick_ctx": {
                "part_name": "MG",
                "model_name": "gear_medium",
                "pick_tcp_z": 1.076,
                "tz": 1.474,
                "tcp_offset_z": -0.218,
            },
            "destination_location": "assembly_board-v1",
        },
        {},
        {},
    )

    assert error is None
    assert result["slot_x"] == 0.0
    assert result["slot_y"] == 0.08
    assert result["place_part_origin_z"] == pytest.approx(1.0325)


def test_physical_place_insert_still_fails_when_lift_fails():
    agent = _FakeRobotAgent("physical")

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            part_name="MG",
            destination_location="assembly_board-v1",
        )
    )

    assert result["status"] == "failed"
    assert result["content"] == "lift failed"
    assert result["step"] == "place_insert.lift"
    assert agent._held_part == "MG"


def test_release_part_uses_open_command_fallback_when_detach_verification_unavailable():
    controller = Ros2PickPlaceController.__new__(Ros2PickPlaceController)
    controller.release_preopen_settle_sec = 0.0
    controller.release_postopen_settle_sec = 0.0
    controller.release_postdetach_settle_sec = 0.0
    controller._attached_model = "gear_medium"
    controller._attached_link = "xarm6_link6"
    controller._last_failure_message = ""
    warn_logger = _WarnLogger()
    controller._log = lambda: warn_logger
    controller.wait_for_services = lambda: True
    controller.open_gripper = lambda: True
    controller.detach_part = lambda *args, **kwargs: {
        "success": False,
        "message": "failed to detach gear_medium",
    }
    controller._verify_detach_timeout_release = lambda model_name: None

    def _unexpected_close_gripper():
        raise AssertionError("release fallback should not reclose the gripper")

    controller.close_gripper = _unexpected_close_gripper

    result = controller.release_part(
        "gear_medium",
        part_name="MG",
        assume_released_if_open=True,
    )

    assert result["success"] is True
    assert result["release_mode"] == "assumed_open_after_detach_timeout"
    assert controller._attached_model is None
    assert controller._attached_link is None


def test_simulation_release_part_fails_when_detach_and_verification_are_unavailable():
    controller = Ros2PickPlaceController.__new__(Ros2PickPlaceController)
    controller.execution_mode = "simulation"
    controller.release_preopen_settle_sec = 0.0
    controller.release_postopen_settle_sec = 0.0
    controller.release_postdetach_settle_sec = 0.0
    controller.release_detach_timeout_sec = 5.0
    controller.release_best_effort_detach_timeout_sec = 0.2
    controller._attached_model = "gear_medium"
    controller._attached_link = "xarm6_link6"
    controller._last_failure_message = ""
    controller._log = lambda: _WarnLogger()
    controller.wait_for_services = lambda: True
    controller.open_gripper = lambda: True
    controller.detach_part = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("simulation release should bypass detach_part retry loop")
    )
    detach_calls: list[dict] = []

    def _best_effort_detach(model_name, **kwargs):
        detach_calls.append({"model_name": model_name, **kwargs})
        return False

    controller._detach_part = _best_effort_detach
    controller._verify_detach_timeout_release = (
        lambda model_name, timeout_log_level="error": None
    )
    close_calls: list[bool] = []

    def _close_gripper():
        close_calls.append(True)
        return True

    controller.close_gripper = _close_gripper

    result = controller.release_part(
        "gear_medium",
        part_name="MG",
        assume_released_if_open=True,
    )

    assert result["success"] is False
    assert detach_calls == [
        {
            "model_name": "gear_medium",
            "timeout_sec": 0.2,
            "attached_link_only": False,
            "log_failure": False,
            "timeout_log_level": "warn",
            "break_on_timeout": False,
            "prefer_attached_link": False,
        }
    ]
    assert "release verification is unavailable" in result["message"]
    assert close_calls == [True]


def test_simulation_release_part_successful_detach_keeps_normal_release_mode():
    controller = Ros2PickPlaceController.__new__(Ros2PickPlaceController)
    controller.execution_mode = "simulation"
    controller.release_preopen_settle_sec = 0.0
    controller.release_postopen_settle_sec = 0.0
    controller.release_postdetach_settle_sec = 0.0
    controller.release_detach_timeout_sec = 5.0
    controller.release_best_effort_detach_timeout_sec = 0.2
    controller._attached_model = "gear_medium"
    controller._attached_link = "xarm6_link6"
    controller._last_failure_message = ""
    controller._log = lambda: _WarnLogger()
    controller.wait_for_services = lambda: True
    controller.open_gripper = lambda: True
    controller._detach_part = lambda *args, **kwargs: True
    controller._verify_detach_timeout_release = lambda *args, **kwargs: (
        (_ for _ in ()).throw(AssertionError("verification should not run"))
    )

    result = controller.release_part(
        "gear_medium",
        part_name="MG",
        assume_released_if_open=True,
    )

    assert result["success"] is True
    assert "release_mode" not in result


def test_simulation_release_part_verifies_release_with_rcutils_style_logger():
    controller = Ros2PickPlaceController.__new__(Ros2PickPlaceController)
    controller.execution_mode = "simulation"
    controller.release_preopen_settle_sec = 0.0
    controller.release_postopen_settle_sec = 0.0
    controller.release_postdetach_settle_sec = 0.0
    controller.release_detach_timeout_sec = 5.0
    controller.release_best_effort_detach_timeout_sec = 0.2
    controller.release_detach_verify_timeout_sec = 0.0
    controller.release_detach_verify_poll_sec = 0.05
    controller.release_detach_verify_distance_m = 0.04
    controller.primary_attach_link = "xarm6_link6"
    controller.attach_link_candidates = ["xarm6_link6"]
    controller._attached_model = "rect_pin_small"
    controller._attached_link = "xarm6_link6"
    controller._last_failure_message = ""
    logger = _WarnLogger()
    controller._log = lambda: logger
    controller.wait_for_services = lambda: True
    controller.open_gripper = lambda: True
    controller.detach_part = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("simulation release should bypass detach_part retry loop")
    )
    controller._detach_part = lambda *args, **kwargs: False
    controller._get_entity_world_position = (
        lambda model_name, timeout_log_level="error": (0.0, 0.0, 0.0)
    )
    controller._get_link_world_position = lambda link_name: (0.10, 0.0, 0.0)
    controller.close_gripper = lambda: (_ for _ in ()).throw(
        AssertionError("verified release should not reclose the gripper")
    )

    result = controller.release_part(
        "rect_pin_small",
        part_name="SRP",
        assume_released_if_open=True,
    )

    assert result["success"] is True
    assert result["release_mode"] == "verified_open_after_detach_timeout"
    assert any("Verified detach fallback" in msg for msg in logger.messages)
    assert controller._attached_model is None
    assert controller._attached_link is None


def test_detach_verification_failure_logs_with_rcutils_style_logger():
    controller = Ros2PickPlaceController.__new__(Ros2PickPlaceController)
    controller.execution_mode = "simulation"
    controller.release_detach_verify_timeout_sec = 0.0
    controller.release_detach_verify_poll_sec = 0.05
    controller.release_detach_verify_distance_m = 0.04
    controller.primary_attach_link = "xarm6_link6"
    controller.attach_link_candidates = ["xarm6_link6"]
    controller._attached_link = "xarm6_link6"
    logger = _WarnLogger()
    controller._log = lambda: logger
    controller._get_entity_world_position = (
        lambda model_name, timeout_log_level="error": (0.0, 0.0, 0.0)
    )
    controller._get_link_world_position = lambda link_name: (0.01, 0.0, 0.0)

    result = controller._verify_detach_timeout_release("rect_pin_small")

    assert result is False
    assert any("Detach fallback verification failed" in msg for msg in logger.messages)


def test_physical_release_part_stays_strict_even_with_assume_released_if_open():
    controller = Ros2PickPlaceController.__new__(Ros2PickPlaceController)
    controller.execution_mode = "physical"
    controller.release_preopen_settle_sec = 0.0
    controller.release_postopen_settle_sec = 0.0
    controller.release_postdetach_settle_sec = 0.0
    controller._attached_model = "gear_medium"
    controller._attached_link = "xarm6_link6"
    controller._last_failure_message = ""
    controller._log = lambda: _WarnLogger()
    controller.wait_for_services = lambda: True
    controller.open_gripper = lambda: True
    controller.detach_part = lambda *args, **kwargs: {
        "success": False,
        "message": "failed to detach gear_medium",
    }
    controller._verify_detach_timeout_release = lambda *args, **kwargs: (
        (_ for _ in ()).throw(AssertionError("physical mode should not assume release"))
    )
    close_calls: list[bool] = []

    def _close_gripper():
        close_calls.append(True)
        return True

    controller.close_gripper = _close_gripper

    result = controller.release_part(
        "gear_medium",
        part_name="MG",
        assume_released_if_open=True,
    )

    assert result["success"] is False
    assert "rollback: reclosed gripper after failed detach" in result["message"]
    assert close_calls == [True]

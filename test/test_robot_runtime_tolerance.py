from __future__ import annotations

import asyncio
import json
import logging
import sys
import types
from pathlib import Path

import pytest

from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent
from cais_spade_llm.resources.robot.robot_primitives import (
    _compile_place_macro,
    _compile_release_macro,
    _preview_place_targets_output,
)
from cais_spade_llm.resources.robot.robot_tasks import (
    execute_robot_task,
    robot_task_capability_decompositions,
)
from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
    GazeboPickPlaceController,
    _gazebo_timing_scale_from_env,
)


class _FakeRobotAgent:
    def __init__(
        self,
        execution_mode: str,
        *,
        lift_success: bool = False,
        snap_success: bool = True,
    ) -> None:
        self.execution_mode = execution_mode
        self.logger = logging.getLogger(f"test.robot_task.{execution_mode}")
        self.lift_success = lift_success
        self.snap_success = snap_success
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
            return {
                "success": self.snap_success,
                "message": "snap ok" if self.snap_success else "snap failed",
            }
        if primitive == "move_relative":
            return {
                "success": self.lift_success,
                "message": "lift ok" if self.lift_success else "lift failed",
            }
        return {"success": True, "message": f"{primitive} ok"}


class _FakePickRobotAgent:
    def __init__(self) -> None:
        self.execution_mode = "simulation"
        self.logger = logging.getLogger("test.robot_task.pick_grasp")
        self._held_part = None
        self._current_state = "at_pick"
        self._position = {"x": 0.0, "y": 0.08, "z": 1.0}
        self._gripper_state = "open"
        self._bridge_pose_ref = None
        self._task_ctx = {
            "model_name": "gear_medium",
            "gripper_close_position": 0.048,
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


class _FakeServiceClient:
    def __init__(self, response) -> None:
        self.response = response
        self.requests: list[object] = []

    def wait_for_service(self, timeout_sec: float = 0.0) -> bool:
        return True

    def call_async(self, request):
        self.requests.append(request)
        return self.response


class _FakeRequest:
    pass


class _FakeSetEntityState:
    class Request:
        def __init__(self) -> None:
            self.state = None


class _FakeEntityState:
    def __init__(self) -> None:
        self.name = ""
        self.pose = types.SimpleNamespace(
            position=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
        )
        self.twist = types.SimpleNamespace(
            linear=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
            angular=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
        )
        self.reference_frame = ""


class _FakeAttachSrv:
    Request = _FakeRequest


def _install_fake_gazebo_msgs(monkeypatch) -> None:
    gazebo_module = types.ModuleType("gazebo_msgs")
    msg_module = types.ModuleType("gazebo_msgs.msg")
    msg_module.EntityState = _FakeEntityState
    monkeypatch.setitem(sys.modules, "gazebo_msgs", gazebo_module)
    monkeypatch.setitem(sys.modules, "gazebo_msgs.msg", msg_module)


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
    assert agent.calls[1][1]["destination_location"] == "assembly_board-v1"
    assert agent._held_part is None
    assert agent._current_state == "placed"
    assert agent._gripper_state == "open"
    assert agent._task_ctx == {}


def test_simulation_place_insert_fails_when_snap_part_to_slot_fails():
    agent = _FakeRobotAgent("simulation", snap_success=False)

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            part_name="MG",
            destination_location="assembly_board-v1",
        )
    )

    assert result["status"] == "failed"
    assert result["content"] == "snap failed"
    assert result["step"] == "place_insert.snap_part_to_slot"
    assert [name for name, _params in agent.calls] == [
        "release_part",
        "snap_part_to_slot",
    ]
    assert agent._held_part == "MG"


def test_lg_slippage_after_place_insert_returns_failed_drop_observation():
    class _SlipController:
        def __init__(self) -> None:
            self.pose_calls: list[tuple[str, dict]] = []

        def set_entity_pose(self, model_name: str, **kwargs) -> dict:
            self.pose_calls.append((model_name, dict(kwargs)))
            return {"success": True, "message": f"entity pose reset for {model_name}"}

    agent = RobotAgent.__new__(RobotAgent)
    agent.jid = "xarm6@localhost"
    agent.agent_name = "xarm6"
    agent.execution_mode = "simulation"
    agent.logger = logging.getLogger("test.robot_task.lg_slippage")
    agent.failure_scenarios = [
        {"scenario_id": "lg_slippage", "mode": "once", "scope": "xarm6"}
    ]
    agent._triggered_failure_scenarios = set()
    agent._held_part = "LG"
    agent._current_state = "positioned"
    agent._position = {"x": 0.1, "y": 0.08, "z": 1.212}
    agent._gripper_state = "closed"
    agent._bridge_pose_ref = None
    agent._task_ctx = {
        "model_name": "gear_large",
        "slot_x": 0.1,
        "slot_y": 0.08,
        "part_height": 0.02,
        "board_top_z": 1.025,
        "place_part_origin_z": 1.035,
        "travel_z": 1.41,
    }
    agent._controller = _SlipController()

    async def _execute_primitive(primitive: str, params: dict) -> dict:
        return {"success": True, "message": f"{primitive} ok"}

    agent._execute_primitive = _execute_primitive

    result = asyncio.run(
        execute_robot_task(
            agent,
            "place_insert",
            part_name="LG",
            destination_location="assembly_board-v1",
            task_id="REQ_1_T4",
        )
    )

    assert result["status"] == "failed"
    assert result["content"] == "Assembly verification failed."
    assert result["observations"]["dropped_location"] == {
        "x": 0.0,
        "y": 0.2,
        "z": 1.035,
    }
    assert result["observations"]["last_commanded_location"] == "assembly_board-v1"
    assert result["failure_context"]["affected_entities"] == [
        {"entity_type": "part", "entity_id": "LG", "state": "unknown"}
    ]
    assert result["failure_context"]["observations"]["dropped_location"] == {
        "x": 0.0,
        "y": 0.2,
        "z": 1.035,
    }
    assert agent._held_part is None
    assert agent._current_state == "failed"
    assert agent._gripper_state == "open"
    assert agent._task_ctx == {}
    assert agent._controller.pose_calls == [
        (
            "gear_large",
            {
                "x": 0.0,
                "y": 0.2,
                "z": 1.035,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        )
    ]


def test_compute_place_targets_clamps_inserted_mg_origin_to_visible_slot_height():
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
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


def test_compute_place_targets_clamps_default_mg_origin_to_visible_slot_height():
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
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
        },
        part_name="MG",
        destination_location="assembly_board-v1",
    )

    assert result["success"] is True
    assert result["place_part_origin_z"] == pytest.approx(1.0325)


def test_move_to_named_pose_uses_configured_named_pose_duration():
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
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
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
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


def test_gazebo_timing_scale_applies_only_to_simulation_gazebo(monkeypatch):
    monkeypatch.setenv("ROBOT_ENV", "gazebo")
    monkeypatch.setenv("CAIS_GAZEBO_WAIT_SCALE", "0.35")

    assert _gazebo_timing_scale_from_env("simulation") == pytest.approx(0.35)
    assert _gazebo_timing_scale_from_env("physical") == pytest.approx(1.0)

    monkeypatch.setenv("ROBOT_ENV", "real")
    assert _gazebo_timing_scale_from_env("simulation") == pytest.approx(1.0)


def test_fast_timing_profile_scales_waits_and_enforces_lower_bounds():
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.execution_mode = "simulation"
    controller.gripper_move_time_sec = 0.4
    controller.gripper_settle_sec = 0.08
    controller.release_preopen_settle_sec = 0.15
    controller.release_postopen_settle_sec = 0.6
    controller.release_postdetach_settle_sec = 0.35
    controller.release_detach_retry_delay_sec = 0.35
    controller.release_detach_verify_timeout_sec = 0.75
    controller.release_detach_verify_poll_sec = 0.1
    controller.snap_to_slot_retry_delay_sec = 0.25
    controller.trajectory_time_scale = 0.45
    controller.named_pose_duration_sec = 1.0
    controller.move_home_duration_sec = 1.0

    controller._apply_gazebo_fast_timing_profile(0.35)

    assert controller.gripper_move_time_sec == pytest.approx(0.15)
    assert controller.gripper_settle_sec == pytest.approx(0.028)
    assert controller.release_preopen_settle_sec == pytest.approx(0.0525)
    assert controller.release_postopen_settle_sec == pytest.approx(0.21)
    assert controller.release_postdetach_settle_sec == pytest.approx(0.1225)
    assert controller.release_detach_retry_delay_sec == pytest.approx(0.1225)
    assert controller.release_detach_verify_timeout_sec == pytest.approx(0.2625)
    assert controller.release_detach_verify_poll_sec == pytest.approx(0.035)
    assert controller.snap_to_slot_retry_delay_sec == pytest.approx(0.0875)
    assert controller.trajectory_time_scale == pytest.approx(0.20)
    assert controller.named_pose_duration_sec == pytest.approx(0.25)
    assert controller.move_home_duration_sec == pytest.approx(0.25)
    assert controller._scaled_wall_wait_sec(0.05) == pytest.approx(0.0175)


def test_move_relative_does_not_start_no_collision_fallback_after_timeout():
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.trajectory_time_scale = 0.45
    controller._last_failure_message = ""
    controller.wait_for_services = lambda: True
    controller._get_ee_pose = lambda: types.SimpleNamespace(
        position=types.SimpleNamespace(x=0.0, y=0.0, z=1.0),
        orientation=types.SimpleNamespace(w=1.0),
    )
    controller._make_pose = lambda x, y, z, orientation: (x, y, z, orientation)
    calls: list[dict] = []

    def _cartesian_move(_target, label, **kwargs):
        calls.append({"label": label, **kwargs})
        controller._last_failure_message = f"[result:{label}] timed out"
        return False

    controller._cartesian_move = _cartesian_move

    result = controller.move_relative(0.0, 0.0, 0.08, speed=0.45)

    assert result["success"] is False
    assert len(calls) == 1
    assert calls[0]["label"] == "move_relative(dx=0.0, dy=0.0, dz=0.08)"


def test_move_relative_keeps_vertical_no_collision_fallback_for_non_timeout_failure():
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.trajectory_time_scale = 0.45
    controller._last_failure_message = ""
    controller.wait_for_services = lambda: True
    controller._get_ee_pose = lambda: types.SimpleNamespace(
        position=types.SimpleNamespace(x=0.0, y=0.0, z=1.0),
        orientation=types.SimpleNamespace(w=1.0),
    )
    controller._make_pose = lambda x, y, z, orientation: (x, y, z, orientation)
    controller._log = lambda: _WarnLogger()
    calls: list[dict] = []

    def _cartesian_move(_target, label, **kwargs):
        calls.append({"label": label, **kwargs})
        if len(calls) == 1:
            controller._last_failure_message = "planning fraction too low"
            return False
        controller._last_failure_message = ""
        return True

    controller._cartesian_move = _cartesian_move

    result = controller.move_relative(0.0, 0.0, 0.08, speed=0.45)

    assert result["success"] is True
    assert len(calls) == 2
    assert calls[1]["label"] == "move_relative(dx=0.0, dy=0.0, dz=0.08) (no-collision)"
    assert calls[1]["avoid_collisions"] is False
    assert calls[1]["allow_partial"] is True


def test_recovery_release_and_place_macros_keep_trailing_home_by_default(monkeypatch):
    monkeypatch.delenv("CAIS_SKIP_RECOVERY_HOME_AFTER_PLACE", raising=False)
    compiler = types.SimpleNamespace(
        _resource_by_jid=lambda _jid: types.SimpleNamespace(named_positions={"home": [0.0]}),
        _bridge_ref=lambda path: {"context_ref": path},
    )
    prepared = _prepared_bridge_request_for_macro_test()
    event = _place_event_for_macro_test()

    release_macro = _compile_release_macro(
        compiler,
        prepared,
        event=event,
        resource_jid="ur5e@localhost",
        start_state="holding",
    )
    place_macro = _compile_place_macro(
        compiler,
        prepared,
        event=event,
        resource_jid="ur5e@localhost",
        start_state="holding",
    )

    assert release_macro["primitive_steps"][-1]["primitive"] == "move_to_named_pose"
    assert place_macro["primitive_steps"][-1]["primitive"] == "move_to_named_pose"


def test_fast_recovery_release_and_place_macros_omit_optional_trailing_home(monkeypatch):
    monkeypatch.setenv("CAIS_SKIP_RECOVERY_HOME_AFTER_PLACE", "1")
    compiler = types.SimpleNamespace(
        _resource_by_jid=lambda _jid: types.SimpleNamespace(named_positions={"home": [0.0]}),
        _bridge_ref=lambda path: {"context_ref": path},
    )
    prepared = _prepared_bridge_request_for_macro_test()
    event = _place_event_for_macro_test()

    release_macro = _compile_release_macro(
        compiler,
        prepared,
        event=event,
        resource_jid="ur5e@localhost",
        start_state="holding",
    )
    place_macro = _compile_place_macro(
        compiler,
        prepared,
        event=event,
        resource_jid="ur5e@localhost",
        start_state="holding",
    )

    assert all(step["primitive"] != "move_to_named_pose" for step in release_macro["primitive_steps"])
    assert all(step["primitive"] != "move_to_named_pose" for step in place_macro["primitive_steps"])


def _prepared_bridge_request_for_macro_test() -> dict:
    return {
        "bridge_resources": {},
        "grounding_context": {
            "parts": {
                "LG": {
                    "target": {
                        "location": "assembly_board-v1",
                        "slot_pose": {"x": 0.0, "y": 0.08, "z": 1.04},
                        "board_top_z": 1.04,
                        "part_height": 0.015,
                        "model_name": "gear_large",
                    }
                }
            }
        },
    }


def _place_event_for_macro_test() -> dict:
    return {
        "event_name": "place_lg_to_assembly_board_v1",
        "part_name": "LG",
        "expected_resource_delta": {"to": "idle"},
        "expected_part_delta": {
            "to": "placed",
            "location_to": "assembly_board-v1",
        },
    }


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


def test_preview_place_targets_clamps_default_mg_origin_to_visible_slot_height():
    result, error = _preview_place_targets_output(
        {
            "part_name": "MG",
            "product_geometry": {
                "slot_xy": [0.0, 0.08],
                "part_height_m": 0.015,
                "model_name": "gear_medium",
                "slot_floor_z_m": 1.025,
                "board_center": {"x": 0.0, "y": 0.0, "z": 1.02},
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
    assert [name for name, _params in agent.calls] == [
        "delay",
        "release_part",
        "delay",
        "move_relative",
    ]
    assert agent.calls[0] == ("delay", {"duration_sec": 0.25})
    assert agent.calls[2] == ("delay", {"duration_sec": 0.25})
    assert agent._held_part == "MG"


def test_xarm6_config_uses_link6_as_only_attach_link():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "cais_spade_llm/initialization/resources/robot_xarm6.json"
    config = json.loads(config_path.read_text())

    attach = config["xarm6"]["gazebo"]["controller"]["attach"]

    assert attach["primary_attach_link"] == "xarm6_link6"
    assert attach["attach_link_candidates"] == ["xarm6_link6"]
    assert attach["release_detach_link_candidates"] == [
        "xarm6_link6",
        "xarm6_right_inner_knuckle",
        "xarm6_left_inner_knuckle",
        "xarm6_right_finger",
        "xarm6_left_finger",
    ]


def test_ur5e_config_has_explicit_release_detach_link_candidates():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "cais_spade_llm/initialization/resources/robot_ur5e.json"
    config = json.loads(config_path.read_text())

    attach = config["ur5e"]["gazebo"]["controller"]["attach"]

    assert attach["release_detach_link_candidates"] == [
        "ur5e_rg2_gripper_tcp",
        "ur5e_tool0",
        "ur5e_wrist_3_link",
    ]


def test_auto_link_attacher_xarm_uses_link6_without_right_gripper_links():
    repo_root = Path(__file__).resolve().parents[1]
    source_path = repo_root / "ros2/cais_lab_gazebo/launch/auto_link_attacher_node.py"
    source = source_path.read_text()

    xarm_block = source.split("'xarm': [", 1)[1].split("],", 1)[0]

    assert "'xarm6_link6'" in xarm_block
    assert "xarm6_right_inner_knuckle" not in xarm_block
    assert "xarm6_left_inner_knuckle" not in xarm_block
    assert "xarm6_right_finger" not in xarm_block
    assert "xarm6_left_finger" not in xarm_block


def test_ur5e_derives_gripper_close_position_from_gazebo_model_geometry():
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.gripper_open = 0.11
    controller.gripper_close = 0.02

    position = controller._derive_gripper_close_position(
        model_name="gear_medium",
        product_geometry={},
    )

    assert position == pytest.approx(0.048)


def test_xarm6_does_not_apply_metric_width_gripper_close_position():
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.gripper_open = 0.0
    controller.gripper_close = 0.85

    position = controller._derive_gripper_close_position(
        model_name="gear_medium",
        product_geometry={},
    )

    assert position is None


def test_pick_grasp_passes_gripper_close_position_to_grasp_part():
    agent = _FakePickRobotAgent()

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_grasp",
            part_name="MG",
            origin_resource_location="prusa-mk4-1",
        )
    )

    assert result["status"] == "completed"
    assert agent.calls[0] == (
        "grasp_part",
        {
            "model_name": "gear_medium",
            "part_name": "MG",
            "position": pytest.approx(0.048),
        },
    )


def test_pick_grasp_runs_delay_between_grasp_and_lift():
    agent = _FakePickRobotAgent()

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_grasp",
            part_name="MG",
            origin_resource_location="prusa-mk4-1",
        )
    )

    assert result["status"] == "completed"
    assert [primitive for primitive, _params in agent.calls] == [
        "grasp_part",
        "delay",
        "move_relative",
    ]
    assert agent.calls[1] == ("delay", {"duration_sec": 0.25})


def test_delay_uses_scaled_wall_time(monkeypatch):
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller._gazebo_wait_scale = 2.0
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "cais_spade_llm.resources.robot.gazebo_pick_place_controller.time.sleep",
        lambda seconds: sleep_calls.append(float(seconds)),
    )

    result = controller.delay(duration_sec=0.5)

    assert result["success"] is True
    assert result["duration_sec"] == pytest.approx(0.5)
    assert result["wait_sec"] == pytest.approx(1.0)
    assert sleep_calls == [pytest.approx(1.0)]


def test_delay_rejects_invalid_duration(monkeypatch):
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "cais_spade_llm.resources.robot.gazebo_pick_place_controller.time.sleep",
        lambda seconds: sleep_calls.append(float(seconds)),
    )

    negative = controller.delay(duration_sec=-0.1)
    non_numeric = controller.delay(duration_sec="later")

    assert negative["success"] is False
    assert "finite and non-negative" in negative["message"]
    assert non_numeric["success"] is False
    assert "must be numeric" in non_numeric["message"]
    assert sleep_calls == []


def test_snap_part_to_slot_attaches_part_to_assembly_board(monkeypatch):
    _install_fake_gazebo_msgs(monkeypatch)
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller._link_attacher_enabled = True
    controller._attached_model = "gear_medium"
    controller._attached_link = "ur5e_rg2_gripper_tcp"
    controller.release_best_effort_detach_timeout_sec = 0.2
    controller.release_detach_timeout_sec = 0.2
    controller.snap_to_slot_retry_count = 0
    controller.snap_to_slot_timeout_sec = 0.2
    controller.snap_to_slot_retry_delay_sec = 0.0
    controller.release_detach_link_candidates = ["ur5e_tool0"]
    controller._SetEntityState = _FakeSetEntityState
    controller._attach_srv = _FakeAttachSrv
    controller._detach_srv = _FakeAttachSrv
    controller._set_state_client = _FakeServiceClient(
        types.SimpleNamespace(success=True, message="")
    )
    controller._attach_client = _FakeServiceClient(
        types.SimpleNamespace(success=True, message="")
    )
    controller._detach_client = _FakeServiceClient(
        types.SimpleNamespace(success=False, message="not attached")
    )
    controller._wait_future = lambda future, **kwargs: future
    controller._log = lambda: _WarnLogger()
    detach_calls: list[dict] = []

    def _detach(model_name, **kwargs):
        detach_calls.append({"model_name": model_name, **kwargs})
        return False

    controller._detach_part = _detach

    result = controller._snap_part_to_slot(
        "gear_medium",
        0.0,
        0.08,
        0.015,
        1.025,
        part_origin_z=1.0325,
        destination_location="assembly_board-v1",
    )

    assert result is True
    assert len(controller._set_state_client.requests) == 2
    state = controller._set_state_client.requests[-1].state
    assert state.name == "gear_medium"
    assert state.pose.position.x == 0.0
    assert state.pose.position.y == 0.08
    assert state.pose.position.z == pytest.approx(1.0325)
    assert state.twist.linear.x == 0.0
    assert state.twist.angular.z == 0.0
    attach_req = controller._attach_client.requests[0]
    assert attach_req.model1_name == "assembly_board_v1"
    assert attach_req.link1_name == "anchor_gear_medium"
    assert attach_req.model2_name == "gear_medium"
    assert attach_req.link2_name == "link"
    assert {
        request.link1_name for request in controller._detach_client.requests
    } == {"anchor_gear_medium", "link"}
    assert detach_calls[0]["extra_link_candidates"] == ["ur5e_tool0"]
    assert controller._attached_model is None
    assert controller._attached_link is None


def test_set_entity_pose_detaches_assembly_board_links_before_pose_reset(monkeypatch):
    _install_fake_gazebo_msgs(monkeypatch)
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller._link_attacher_enabled = True
    controller._SetEntityState = _FakeSetEntityState
    controller._detach_srv = _FakeAttachSrv
    controller._set_state_client = _FakeServiceClient(
        types.SimpleNamespace(success=True, message="")
    )
    controller._detach_client = _FakeServiceClient(
        types.SimpleNamespace(success=True, message="detached")
    )
    controller._wait_future = lambda future, **kwargs: future
    controller._log = lambda: _WarnLogger()
    controller.wait_for_services = lambda: True

    result = controller.set_entity_pose(
        "gear_large",
        x=0.0,
        y=0.2,
        z=1.035,
        qx=0.0,
        qy=0.0,
        qz=0.0,
        qw=1.0,
    )

    assert result["success"] is True
    assert [
        (request.model1_name, request.link1_name, request.model2_name, request.link2_name)
        for request in controller._detach_client.requests
    ] == [
        ("assembly_board_v1", "anchor_gear_large", "gear_large", "link"),
        ("assembly_board_v1", "link", "gear_large", "link"),
    ]
    state = controller._set_state_client.requests[0].state
    assert state.name == "gear_large"
    assert state.pose.position.x == pytest.approx(0.0)
    assert state.pose.position.y == pytest.approx(0.2)
    assert state.pose.position.z == pytest.approx(1.035)


def test_snap_part_to_slot_fails_when_assembly_board_attach_fails(monkeypatch):
    _install_fake_gazebo_msgs(monkeypatch)
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller._link_attacher_enabled = True
    controller._attached_model = "gear_medium"
    controller._attached_link = "ur5e_rg2_gripper_tcp"
    controller.release_best_effort_detach_timeout_sec = 0.2
    controller.release_detach_timeout_sec = 0.2
    controller.snap_to_slot_retry_count = 0
    controller.snap_to_slot_timeout_sec = 0.2
    controller.snap_to_slot_retry_delay_sec = 0.0
    controller.release_detach_link_candidates = []
    controller._SetEntityState = _FakeSetEntityState
    controller._attach_srv = _FakeAttachSrv
    controller._detach_srv = _FakeAttachSrv
    controller._set_state_client = _FakeServiceClient(
        types.SimpleNamespace(success=True, message="")
    )
    controller._attach_client = _FakeServiceClient(
        types.SimpleNamespace(success=False, message="attach failed")
    )
    controller._detach_client = _FakeServiceClient(
        types.SimpleNamespace(success=False, message="not attached")
    )
    controller._wait_future = lambda future, **kwargs: future
    controller._log = lambda: _WarnLogger()
    controller._detach_part = lambda *args, **kwargs: False

    result = controller._snap_part_to_slot(
        "gear_medium",
        0.0,
        0.08,
        0.015,
        1.025,
        part_origin_z=1.0325,
        destination_location="assembly_board-v1",
    )

    assert result is False
    assert len(controller._set_state_client.requests) == 1
    assert len(controller._attach_client.requests) == 4


def test_snap_part_to_slot_falls_back_to_board_link_when_anchor_link_is_missing(monkeypatch):
    _install_fake_gazebo_msgs(monkeypatch)
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller._link_attacher_enabled = True
    controller._attached_model = "gear_medium"
    controller._attached_link = "ur5e_rg2_gripper_tcp"
    controller.release_best_effort_detach_timeout_sec = 0.2
    controller.release_detach_timeout_sec = 0.2
    controller.snap_to_slot_retry_count = 0
    controller.snap_to_slot_timeout_sec = 0.2
    controller.snap_to_slot_retry_delay_sec = 0.0
    controller.release_detach_link_candidates = []
    controller._SetEntityState = _FakeSetEntityState
    controller._attach_srv = _FakeAttachSrv
    controller._detach_srv = _FakeAttachSrv
    controller._set_state_client = _FakeServiceClient(
        types.SimpleNamespace(success=True, message="")
    )
    attach_responses = [
        types.SimpleNamespace(success=False, message="Failed to find link with name: anchor_gear_medium"),
        types.SimpleNamespace(success=True, message="attached"),
    ]

    class _SequenceClient(_FakeServiceClient):
        def call_async(self, request):
            self.requests.append(request)
            return attach_responses.pop(0)

    controller._attach_client = _SequenceClient(None)
    controller._detach_client = _FakeServiceClient(
        types.SimpleNamespace(success=False, message="not attached")
    )
    controller._wait_future = lambda future, **kwargs: future
    controller._log = lambda: _WarnLogger()
    controller._detach_part = lambda *args, **kwargs: False

    result = controller._snap_part_to_slot(
        "gear_medium",
        0.0,
        0.08,
        0.015,
        1.025,
        part_origin_z=1.0325,
        destination_location="assembly_board-v1",
    )

    assert result is True
    assert [request.link1_name for request in controller._attach_client.requests] == [
        "anchor_gear_medium",
        "link",
    ]


def test_release_part_uses_open_command_fallback_when_detach_verification_unavailable():
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
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


def test_simulation_release_part_continues_when_detach_verification_is_unavailable():
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.execution_mode = "simulation"
    controller.release_preopen_settle_sec = 0.0
    controller.release_postopen_settle_sec = 0.0
    controller.release_postdetach_settle_sec = 0.0
    controller.release_detach_timeout_sec = 5.0
    controller.release_best_effort_detach_timeout_sec = 0.2
    controller.release_detach_link_candidates = [
        "xarm6_link6",
        "xarm6_right_inner_knuckle",
    ]
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
    controller.close_gripper = lambda: (_ for _ in ()).throw(
        AssertionError("simulation release fallback should not reclose the gripper")
    )

    result = controller.release_part(
        "gear_medium",
        part_name="MG",
    )

    assert result["success"] is True
    assert result["release_mode"] == "verification_unavailable_after_detach_timeout"
    assert detach_calls == [
        {
            "model_name": "gear_medium",
            "timeout_sec": 0.2,
            "attached_link_only": False,
            "log_failure": False,
            "timeout_log_level": "warn",
            "break_on_timeout": False,
            "prefer_attached_link": False,
            "extra_link_candidates": [
                "xarm6_link6",
                "xarm6_right_inner_knuckle",
            ],
        }
    ]
    assert "detach verification unavailable" in result["message"]


def test_simulation_release_part_tries_release_detach_link_candidates():
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
    controller.execution_mode = "simulation"
    controller.release_preopen_settle_sec = 0.0
    controller.release_postopen_settle_sec = 0.0
    controller.release_postdetach_settle_sec = 0.0
    controller.release_detach_timeout_sec = 5.0
    controller.release_best_effort_detach_timeout_sec = 0.2
    controller.release_detach_link_candidates = [
        "xarm6_link6",
        "xarm6_right_inner_knuckle",
        "xarm6_left_inner_knuckle",
    ]
    controller._attached_model = "gear_medium"
    controller._attached_link = "xarm6_link6"
    controller._last_failure_message = ""
    controller._log = lambda: _WarnLogger()
    controller.wait_for_services = lambda: True
    controller.open_gripper = lambda: True
    detach_kwargs: list[dict] = []

    def _best_effort_detach(model_name, **kwargs):
        detach_kwargs.append(dict(kwargs))
        return True

    controller._detach_part = _best_effort_detach
    controller._verify_detach_timeout_release = lambda *args, **kwargs: (
        (_ for _ in ()).throw(AssertionError("verification should not run"))
    )

    result = controller.release_part("gear_medium", part_name="MG")

    assert result["success"] is True
    assert detach_kwargs[0]["extra_link_candidates"] == [
        "xarm6_link6",
        "xarm6_right_inner_knuckle",
        "xarm6_left_inner_knuckle",
    ]


def test_simulation_release_part_successful_detach_keeps_normal_release_mode():
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
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
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
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
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
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
    controller = GazeboPickPlaceController.__new__(GazeboPickPlaceController)
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

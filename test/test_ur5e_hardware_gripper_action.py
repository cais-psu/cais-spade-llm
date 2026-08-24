"""Focused contracts for physical UR5e RG2 action routing."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.resources.robot.hardware_pick_place_controller import (
    UR5eHardwareController,
)

ROOT = Path(__file__).resolve().parents[1]


class _ImmediateFuture:
    def __init__(self, result: Any) -> None:
        self._result = result

    @staticmethod
    def done() -> bool:
        return True

    def result(self) -> Any:
        return self._result


class _SettableFuture:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._result: Any = None

    def done(self) -> bool:
        return self._event.is_set()

    def result(self) -> Any:
        assert self._event.is_set()
        return self._result

    def set_result(self, result: Any) -> None:
        self._result = result
        self._event.set()


class _Goal:
    def __init__(self) -> None:
        self.trajectory = SimpleNamespace(joint_names=[], points=[])


class _Point:
    def __init__(self) -> None:
        self.positions: list[float] = []
        self.time_from_start: Any | None = None


class _ActionClient:
    def __init__(
        self,
        *,
        goal_status: int = 4,
        error_code: int = 0,
        error_string: str = "",
    ) -> None:
        self.goal_status = goal_status
        self.error_code = error_code
        self.error_string = error_string
        self.goals: list[Any] = []
        self.wait_ready = True
        self.destroyed = False

    def wait_for_server(self, timeout_sec: float) -> bool:
        assert timeout_sec == 2.0
        return self.wait_ready

    def send_goal_async(self, goal: Any) -> _ImmediateFuture:
        self.goals.append(goal)
        wrapped_result = SimpleNamespace(
            status=self.goal_status,
            result=SimpleNamespace(
                error_code=self.error_code,
                error_string=self.error_string,
            )
        )
        goal_handle = SimpleNamespace(
            accepted=True,
            get_result_async=lambda: _ImmediateFuture(wrapped_result),
        )
        return _ImmediateFuture(goal_handle)

    def destroy(self) -> None:
        self.destroyed = True


def _controller_double(client: _ActionClient) -> UR5eHardwareController:
    controller = object.__new__(UR5eHardwareController)
    controller._rg2_action_name = (
        "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory"
    )
    controller._rg2_action_client = client
    controller._FollowJointTrajectory = SimpleNamespace(Goal=_Goal)
    controller._JointTrajectoryPoint = _Point
    controller._Duration = lambda *, sec, nanosec: SimpleNamespace(
        sec=sec,
        nanosec=nanosec,
    )
    controller.gripper_joint = "ur5e_rg2_finger_width"
    controller.gripper_open = 0.11
    controller.gripper_close = 0.02
    controller.gripper_move_time_sec = 0.35
    controller.gripper_feedback_timeout_pad_sec = 1.0
    controller._last_failure_message = ""
    controller.wait_for_services = lambda: True
    controller._wait_future = lambda future, **_kwargs: future.result()
    return controller


def _arm_controller_double(client: _ActionClient) -> UR5eHardwareController:
    controller = object.__new__(UR5eHardwareController)
    controller._ur5e_hardware_trajectory_action = (
        "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory"
    )
    controller._ur5e_hardware_trajectory_client = client
    controller._ur5e_hardware_cartesian_action = (
        "/cais_ur5e_rtde_cartesian_controller/move_cartesian"
    )
    controller._ur5e_hardware_cartesian_client = _ActionClient()
    controller._ur5e_hardware_insert_action = (
        "/cais_ur5e_rtde_cartesian_controller/move_insert"
    )
    controller._ur5e_hardware_insert_client = _ActionClient()
    controller._FollowJointTrajectory = SimpleNamespace(Goal=_Goal)
    controller._MoveUR5eCartesian = SimpleNamespace(Goal=object)
    controller._MoveUR5eInsert = SimpleNamespace(Goal=object)
    controller._JointTrajectoryPoint = _Point
    controller._Duration = lambda *, sec, nanosec: SimpleNamespace(
        sec=sec,
        nanosec=nanosec,
    )
    controller.arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    controller.named_positions = {
        "home": [-1.45, -0.85, -2.34, -1.51, 1.56, -3.02],
        "prusa-mk4-2": [0.08, -0.88, -2.15, -1.66, 1.55, -3.14],
    }
    controller.named_pose_duration_sec = 1.0
    controller.move_home_duration_sec = 1.0
    controller.trajectory_time_scale = 0.45
    controller._last_failure_message = ""
    controller._last_start_pose = object()
    controller.wait_for_services = lambda: True
    controller._wait_future = lambda future, **_kwargs: future.result()
    controller._get_arm_joint_positions = lambda **_kwargs: (
        [-1.45, -0.85, -2.34, -1.51, 1.56, -3.02],
        [],
    )
    return controller


def test_physical_ur5e_named_pose_uses_guarded_two_point_rtde_action() -> None:
    client = _ActionClient()
    controller = _arm_controller_double(client)

    result = controller.move_to_named_pose("prusa-mk4-2")

    assert result["success"] is True
    assert len(client.goals) == 1
    goal = client.goals[0]
    assert goal.trajectory.joint_names == controller.arm_joint_names
    assert goal.trajectory.points[0].positions == [
        -1.45,
        -0.85,
        -2.34,
        -1.51,
        1.56,
        -3.02,
    ]
    assert goal.trajectory.points[0].time_from_start.sec == 0
    assert goal.trajectory.points[0].time_from_start.nanosec == 0
    assert goal.trajectory.points[1].positions == controller.named_positions["prusa-mk4-2"]


def test_physical_ur5e_named_pose_allows_action_acknowledgement_latency() -> None:
    client = _ActionClient()
    controller = _arm_controller_double(client)
    waits: dict[str, float] = {}

    def _wait(
        future: _ImmediateFuture,
        *,
        timeout_sec: float,
        label: str,
    ) -> Any:
        waits[label] = timeout_sec
        return future.result()

    controller._wait_future = _wait

    result = controller.move_to_named_pose("prusa-mk4-2")

    assert result["success"] is True
    assert waits["send:move_to_named_pose:prusa-mk4-2"] == 10.0


def test_physical_ur5e_named_pose_send_timeout_reports_unknown_motion_state() -> None:
    client = _ActionClient()
    controller = _arm_controller_double(client)

    def _wait(
        future: _ImmediateFuture,
        *,
        timeout_sec: float,
        label: str,
    ) -> Any:
        assert timeout_sec == 10.0
        assert label == "send:move_to_named_pose:prusa-mk4-2"
        return None

    controller._wait_future = _wait

    result = controller.move_to_named_pose("prusa-mk4-2")

    assert result["success"] is False
    assert "send acknowledgement timeout after 10.0s" in result["message"]
    assert "physical motion may still be executing" in result["message"]


def test_physical_ur5e_named_pose_reports_exact_rtde_action_failure() -> None:
    client = _ActionClient(error_code=-4, error_string="trajectory rejected")
    controller = _arm_controller_double(client)

    result = controller.move_to_named_pose("prusa-mk4-2")

    assert result["success"] is False
    assert controller._ur5e_hardware_trajectory_action in result["message"]
    assert "error_code=-4 trajectory rejected" in result["message"]


def test_physical_ur5e_joint_replay_requires_current_joint_state() -> None:
    client = _ActionClient()
    controller = _arm_controller_double(client)
    controller._get_arm_joint_positions = lambda **_kwargs: (None, ["elbow_joint"])

    assert controller.move_joints([0.0] * 6) is False
    assert "current joint state is unavailable" in controller._last_failure_message
    assert "elbow_joint" in controller._last_failure_message
    assert client.goals == []


def test_physical_ur5e_joint_replay_rejects_non_finite_current_joint_state() -> None:
    client = _ActionClient()
    controller = _arm_controller_double(client)
    controller._get_arm_joint_positions = lambda **_kwargs: (
        [0.0, 0.0, float("nan"), 0.0, 0.0, 0.0],
        [],
    )

    assert controller.move_joints([0.0] * 6) is False
    assert "current joint state contains invalid values" in controller._last_failure_message
    assert client.goals == []


def test_physical_ur5e_arm_result_timeout_cancels_and_confirms_terminal_goal() -> None:
    client = _ActionClient()
    controller = _arm_controller_double(client)
    cancel_calls = 0
    terminal = SimpleNamespace(
        status=5,
        result=SimpleNamespace(error_code=-1, error_string="canceled"),
    )

    class _GoalHandle:
        accepted = True

        @staticmethod
        def get_result_async() -> _ImmediateFuture:
            return _ImmediateFuture(terminal)

        @staticmethod
        def cancel_goal_async() -> _ImmediateFuture:
            nonlocal cancel_calls
            cancel_calls += 1
            return _ImmediateFuture(SimpleNamespace(goals_canceling=[object()]))

    goal_handle = _GoalHandle()

    def _send(goal: Any) -> _ImmediateFuture:
        client.goals.append(goal)
        return _ImmediateFuture(goal_handle)

    def _wait(future: _ImmediateFuture, *, label: str, **_kwargs: Any) -> Any:
        if label.startswith("result:"):
            return None
        return future.result()

    client.send_goal_async = _send
    controller._wait_future = _wait

    assert controller.move_joints([0.0] * 6) is False
    assert cancel_calls == 1
    assert "result timeout" in controller._last_failure_message
    assert "terminal status 5" in controller._last_failure_message


def test_physical_ur5e_move_home_uses_rtde_even_when_cached_state_is_home() -> None:
    client = _ActionClient()
    controller = _arm_controller_double(client)

    result = controller.move_home()

    assert result == {"success": True, "message": "moved to named home pose"}
    assert controller._last_start_pose is None
    assert len(client.goals) == 1
    assert client.goals[0].trajectory.points[1].positions == controller.named_positions["home"]


def test_physical_ur5e_move_home_does_not_trust_cached_home_when_action_is_missing() -> None:
    client = _ActionClient()
    client.wait_ready = False
    controller = _arm_controller_double(client)

    result = controller.move_home()

    assert result["success"] is False
    assert controller._ur5e_hardware_trajectory_action in result["message"]
    assert client.goals == []


def test_generic_ur5e_readiness_does_not_require_optional_insert_action() -> None:
    controller = _arm_controller_double(_ActionClient())
    controller.init = lambda: True
    controller._services_ready = False
    controller._rg2_action_client = _ActionClient()
    controller._get_arm_joint_positions = lambda **_kwargs: ([0.0] * 6, [])
    controller._ur5e_hardware_insert_client.wait_for_server = (
        lambda **_kwargs: pytest.fail("generic readiness waited for move_insert")
    )

    ready = UR5eHardwareController.wait_for_services(controller, timeout_sec=2.0)

    assert ready is True
    assert controller._services_ready is True


def test_reset_ur5e_hardware_trajectory_client_leaves_optional_insert_client() -> None:
    previous_arm_client = _ActionClient()
    previous_cartesian_client = _ActionClient()
    previous_insert_client = _ActionClient()
    gripper_client = _ActionClient()
    replacement_arm_client = _ActionClient()
    replacement_cartesian_client = _ActionClient()
    wait_timeouts: list[float] = []
    for replacement_client in (
        replacement_arm_client,
        replacement_cartesian_client,
    ):
        replacement_client.wait_for_server = (
            lambda timeout_sec: wait_timeouts.append(timeout_sec) or True
        )
    controller = _arm_controller_double(previous_arm_client)
    controller._ur5e_hardware_cartesian_client = previous_cartesian_client
    controller._ur5e_hardware_insert_client = previous_insert_client
    controller._node = object()
    controller._cb_group = object()
    controller._rg2_action_client = gripper_client
    created: list[tuple[Any, Any, str, Any]] = []

    def _create_client(
        node: Any,
        action_type: Any,
        action_name: str,
        *,
        callback_group: Any,
    ) -> _ActionClient:
        created.append((node, action_type, action_name, callback_group))
        if action_name == controller._ur5e_hardware_trajectory_action:
            return replacement_arm_client
        if action_name == controller._ur5e_hardware_cartesian_action:
            return replacement_cartesian_client
        raise AssertionError(f"unexpected action client: {action_name}")

    controller._ActionClient = _create_client

    ok, message = controller.reset_ur5e_hardware_trajectory_client(timeout_sec=8.0)

    assert ok is True
    assert "recreated" in message
    assert previous_arm_client.destroyed is True
    assert previous_cartesian_client.destroyed is True
    assert previous_insert_client.destroyed is False
    assert gripper_client.destroyed is False
    assert controller._rg2_action_client is gripper_client
    assert controller._ur5e_hardware_trajectory_client is replacement_arm_client
    assert controller._ur5e_hardware_cartesian_client is replacement_cartesian_client
    assert controller._ur5e_hardware_insert_client is previous_insert_client
    assert len(wait_timeouts) == 2
    assert 0.0 <= wait_timeouts[1] <= wait_timeouts[0] <= 8.0
    assert created == [
        (
            controller._node,
            controller._FollowJointTrajectory,
            controller._ur5e_hardware_trajectory_action,
            controller._cb_group,
        ),
        (
            controller._node,
            controller._MoveUR5eCartesian,
            controller._ur5e_hardware_cartesian_action,
            controller._cb_group,
        ),
    ]


def test_reset_ur5e_hardware_trajectory_client_reports_discovery_timeout() -> None:
    previous_arm_client = _ActionClient()
    replacement_client = _ActionClient()
    replacement_client.wait_for_server = lambda timeout_sec: False
    controller = _arm_controller_double(previous_arm_client)
    previous_cartesian_client = controller._ur5e_hardware_cartesian_client
    previous_insert_client = controller._ur5e_hardware_insert_client
    controller._node = object()
    controller._cb_group = object()
    controller._ActionClient = lambda *_args, **_kwargs: replacement_client

    ok, message = controller.reset_ur5e_hardware_trajectory_client(timeout_sec=1.0)

    assert ok is False
    assert "was not discovered" in message
    assert previous_arm_client.destroyed is True
    assert previous_cartesian_client.destroyed is True
    assert previous_insert_client.destroyed is False
    assert controller._ur5e_hardware_trajectory_client is replacement_client


def test_physical_ur5e_gripper_uses_exact_action_without_direct_rtde() -> None:
    client = _ActionClient()
    controller = _controller_double(client)

    assert controller.open_gripper() is True
    assert controller.close_gripper(position=0.03) is True

    assert len(client.goals) == 2
    assert client.goals[0].trajectory.joint_names == ["ur5e_rg2_finger_width"]
    assert client.goals[0].trajectory.points[0].positions == [0.11]
    assert client.goals[1].trajectory.points[0].positions == [0.03]
    assert not hasattr(UR5eHardwareController, "_get_rg2_gripper")


def test_physical_ur5e_gripper_reports_action_failure() -> None:
    client = _ActionClient(error_code=-4, error_string="RG2 command failed")
    controller = _controller_double(client)

    assert controller.close_gripper() is False
    assert "error_code=-4 RG2 command failed" in controller._last_failure_message


def test_physical_ur5e_gripper_rejects_non_succeeded_goal_status() -> None:
    client = _ActionClient(goal_status=6, error_code=0)
    controller = _controller_double(client)

    assert controller.open_gripper() is False
    assert "goal_status=6 error_code=0" in controller._last_failure_message


def test_physical_ur5e_gripper_blocks_when_action_server_is_missing() -> None:
    client = _ActionClient()
    client.wait_ready = False
    controller = _controller_double(client)

    assert controller.open_gripper() is False
    assert controller._rg2_action_name in controller._last_failure_message
    assert client.goals == []


@pytest.mark.parametrize("position", [float("nan"), 0.019, 0.111])
def test_physical_ur5e_gripper_rejects_unsafe_position(position: float) -> None:
    client = _ActionClient()
    controller = _controller_double(client)

    assert controller.close_gripper(position=position) is False
    assert "position" in controller._last_failure_message
    assert client.goals == []


def test_physical_ur5e_gripper_uses_terminal_settlement_wait() -> None:
    client = _ActionClient()
    controller = _controller_double(client)
    waits: list[float] = []

    def _wait(
        _goal_handle: Any,
        future: _ImmediateFuture,
        *,
        timeout_sec: float,
    ) -> Any:
        waits.append(timeout_sec)
        return future.result()

    controller._wait_ur5e_action_terminal_settlement = _wait

    assert controller.open_gripper() is True
    assert waits == [8.0]


def test_physical_ur5e_gripper_retains_call_until_delayed_acceptance_settles() -> None:
    client = _ActionClient()
    controller = _controller_double(client)
    controller._ur5e_action_send_timeout_sec = 0.01
    send_future = _SettableFuture()
    wrapped = SimpleNamespace(
        status=4,
        result=SimpleNamespace(error_code=0, error_string=""),
    )
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: _ImmediateFuture(wrapped),
    )
    client.send_goal_async = lambda goal: client.goals.append(goal) or send_future
    results: list[bool] = []

    worker = threading.Thread(target=lambda: results.append(controller.open_gripper()))
    worker.start()
    time.sleep(0.05)

    assert worker.is_alive()
    assert len(client.goals) == 1

    send_future.set_result(goal_handle)
    worker.join(timeout=1.0)

    assert worker.is_alive() is False
    assert results == [True]


def test_physical_ur5e_gripper_retains_call_without_terminal_observer() -> None:
    client = _ActionClient()
    controller = _controller_double(client)
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: (_ for _ in ()).throw(RuntimeError("observer lost")),
    )
    client.send_goal_async = (
        lambda goal: client.goals.append(goal) or _ImmediateFuture(goal_handle)
    )
    entered = threading.Event()
    release = threading.Event()
    controller._retain_accepted_action_without_terminal_observer = (
        lambda: entered.set() or release.wait()
    )
    results: list[bool] = []
    worker = threading.Thread(target=lambda: results.append(controller.open_gripper()))
    worker.start()

    assert entered.wait(timeout=1.0)
    assert worker.is_alive()

    release.set()
    worker.join(timeout=1.0)

    assert results == [False]
    assert "observer lost" in controller._last_failure_message


def test_physical_ur5e_shutdown_releases_action_client() -> None:
    client = _ActionClient()
    controller = _controller_double(client)
    controller._initialized = False

    controller.shutdown()

    assert client.destroyed is True
    assert controller._rg2_action_client is None
    assert controller._FollowJointTrajectory is None


def test_physical_ur5e_shutdown_releases_arm_and_gripper_action_clients() -> None:
    arm_client = _ActionClient()
    gripper_client = _ActionClient()
    controller = _arm_controller_double(arm_client)
    controller._rg2_action_client = gripper_client
    controller._initialized = False

    controller.shutdown()

    assert arm_client.destroyed is True
    assert gripper_client.destroyed is True
    assert controller._ur5e_hardware_trajectory_client is None
    assert controller._rg2_action_client is None
    assert controller._FollowJointTrajectory is None


def test_real_manifest_configures_the_preflighted_rg2_action() -> None:
    manifest = json.loads(
        (ROOT / "cais_spade_llm/initialization/resources/robot_ur5e.json").read_text(
            encoding="utf-8"
        )
    )
    resource = manifest["ur5e"]
    controller = UR5eHardwareController(
        controller_config=resource["real"]["controller"],
        named_positions=resource["real"]["named_positions"],
    )

    assert (
        controller._rg2_action_name
        == "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory"
    )
    assert (
        controller._ur5e_hardware_trajectory_action
        == "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory"
    )
    assert controller._ur5e_action_send_timeout_sec == 10.0
    assert controller._ur5e_hardware_trajectory_client is None
    assert controller._rg2_action_client is None


def test_physical_controller_reserves_an_executor_worker_for_action_responses() -> None:
    source = (
        ROOT / "cais_spade_llm/resources/robot/hardware_pick_place_controller.py"
    ).read_text(encoding="utf-8")

    assert "MultiThreadedExecutor(num_threads=2)" in source

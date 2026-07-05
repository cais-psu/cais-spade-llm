from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from cais_spade_llm.resources.robot.hardware_pick_place_controller import (
    HardwarePickPlaceController,
    TAUGHT_FUNCTIONS_ROOT,
    UR5eRG2GripperController,
    UR5eRG2GripperControllerSettings,
)
from cais_spade_llm.resources.robot.robot_tasks import execute_robot_task
from cais_spade_llm.ui.bridge import SystemBridge
from ros2.cais_lab_gazebo.scripts import digital_twin_sync
from ros2.cais_lab_gazebo.scripts import ur5e_rtde_trajectory_server
from ros2.cais_lab_gazebo.scripts.ur5e_rg2_rtde_gripper import (
    _build_arg_parser,
    _default_status_path,
    _force_for_position,
    _normalize_xmlrpc_path,
    _rg2_status_payload,
    _resolve_backend,
    _settle_sec_for_position,
    _width_mm_from_position,
    _xmlrpc_url,
)


class FakeRTDE:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def sendCustomScriptFunction(self, name: str, body: str) -> None:
        self.calls.append((name, body))

    def sendCustomScript(self, body: str) -> bool:
        self.calls.append(("script", body))
        return True


class FakeRTDEFactory:
    def __init__(self) -> None:
        self.kwargs: list[dict[str, object]] = []
        self.rtde = FakeRTDE()

    def __call__(self, **kwargs: object) -> FakeRTDE:
        self.kwargs.append(dict(kwargs))
        return self.rtde


def _make_guard_point(
    time_sec: float,
    positions: list[float],
    *,
    velocities: list[float] | None = None,
    accelerations: list[float] | None = None,
):
    point = ur5e_rtde_trajectory_server.JointTrajectoryPoint()
    point.positions = list(positions)
    if velocities is not None:
        point.velocities = list(velocities)
    if accelerations is not None:
        point.accelerations = list(accelerations)
    ur5e_rtde_trajectory_server._set_duration(point.time_from_start, time_sec)
    return point


def _make_guard_trajectory(
    points: list,
    *,
    joint_names: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        joint_names=list(joint_names or ur5e_rtde_trajectory_server.ARM_JOINTS),
        points=list(points),
    )


def _zero_ur5e_current_positions() -> dict[str, float]:
    return {joint: 0.0 for joint in ur5e_rtde_trajectory_server.ARM_JOINTS}


def _guard_point_time(point) -> float:
    return ur5e_rtde_trajectory_server._point_seconds(point)


def _make_joint_state(names: list[str], positions: list[float]):
    msg = ur5e_rtde_trajectory_server.JointState()
    msg.name = list(names)
    msg.position = list(positions)
    return msg


def test_ur5e_rtde_trajectory_helper_accepts_valid_trajectory() -> None:
    current = _zero_ur5e_current_positions()
    trajectory = _make_guard_trajectory(
        [
            _make_guard_point(1.0, [0.0] * 6),
            _make_guard_point(2.0, [0.04, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    )

    ok, guarded, status = ur5e_rtde_trajectory_server.prepare_guarded_trajectory(trajectory, current)

    assert ok is True
    assert guarded is not None
    assert status["state"] == "ready"
    assert status["blocked_reason"] == ""
    assert status["inserted_current_hold"] is False
    assert status["time_scale_applied"] == pytest.approx(1.0)
    assert status["max_segment_velocity_rad_s"] <= ur5e_rtde_trajectory_server.UR5E_RTDE_MAX_JOINT_VEL_RAD_S


def test_ur5e_rtde_trajectory_helper_prepends_current_hold_when_first_point_is_zero() -> None:
    current = _zero_ur5e_current_positions()
    trajectory = _make_guard_trajectory(
        [
            _make_guard_point(0.0, [0.0] * 6),
            _make_guard_point(1.0, [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    )

    ok, guarded, status = ur5e_rtde_trajectory_server.prepare_guarded_trajectory(trajectory, current)

    assert ok is True
    assert guarded is not None
    assert status["inserted_current_hold"] is True
    assert _guard_point_time(guarded.points[0]) == pytest.approx(
        ur5e_rtde_trajectory_server.UR5E_RTDE_CURRENT_HOLD_SEC
    )
    assert list(guarded.points[0].positions) == [0.0] * 6
    assert _guard_point_time(guarded.points[1]) >= (
        ur5e_rtde_trajectory_server.UR5E_RTDE_CURRENT_HOLD_SEC
        + ur5e_rtde_trajectory_server.UR5E_RTDE_MIN_POINT_SPACING_SEC
        - 1e-9
    )
    assert status["first_point_time"] == pytest.approx(ur5e_rtde_trajectory_server.UR5E_RTDE_CURRENT_HOLD_SEC)


def test_ur5e_rtde_trajectory_helper_stretches_time_when_velocity_exceeds_cap() -> None:
    current = _zero_ur5e_current_positions()
    trajectory = _make_guard_trajectory(
        [
            _make_guard_point(0.25, [0.0] * 6),
            _make_guard_point(0.50, [0.50, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    )

    ok, guarded, status = ur5e_rtde_trajectory_server.prepare_guarded_trajectory(trajectory, current)

    assert ok is True
    assert guarded is not None
    assert status["time_scale_applied"] > 1.0
    assert _guard_point_time(guarded.points[-1]) > 0.50
    assert status["max_segment_velocity_rad_s"] <= (
        ur5e_rtde_trajectory_server.UR5E_RTDE_MAX_JOINT_VEL_RAD_S + 1e-9
    )
    assert status["shoulder_pan_extra_scale_applied"] is True


def test_ur5e_rtde_trajectory_helper_stretches_time_for_acceleration_and_jerk() -> None:
    current = _zero_ur5e_current_positions()
    trajectory = _make_guard_trajectory(
        [
            _make_guard_point(1.00, [0.0] * 6),
            _make_guard_point(1.10, [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _make_guard_point(1.20, [-0.02, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _make_guard_point(1.30, [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    )

    ok, guarded, status = ur5e_rtde_trajectory_server.prepare_guarded_trajectory(
        trajectory,
        current,
        max_joint_velocity_rad_s=10.0,
        max_joint_acceleration_rad_s2=0.05,
        max_joint_jerk_rad_s3=0.05,
        shoulder_pan_extra_scale=1.0,
    )

    assert ok is True
    assert guarded is not None
    assert status["acceleration_time_scale"] > 1.0
    assert status["jerk_time_scale"] > 1.0
    assert status["max_segment_acceleration_rad_s2"] <= 0.05 + 1e-9
    assert status["max_segment_jerk_rad_s3"] <= 0.05 + 1e-9


def test_ur5e_rtde_trajectory_helper_scales_velocities_and_accelerations_when_retiming() -> None:
    current = _zero_ur5e_current_positions()
    trajectory = _make_guard_trajectory(
        [
            _make_guard_point(0.25, [0.0] * 6),
            _make_guard_point(
                0.50,
                [0.50, 0.0, 0.0, 0.0, 0.0, 0.0],
                velocities=[0.50] * 6,
                accelerations=[1.00] * 6,
            ),
        ]
    )

    ok, guarded, status = ur5e_rtde_trajectory_server.prepare_guarded_trajectory(trajectory, current)

    assert ok is True
    assert guarded is not None
    scale = float(status["time_scale_applied"])
    assert scale > 1.0
    assert list(guarded.points[-1].velocities)[0] == pytest.approx(0.50 / scale)
    assert list(guarded.points[-1].accelerations)[0] == pytest.approx(1.00 / (scale * scale))


def test_ur5e_rtde_trajectory_helper_blocks_start_pose_mismatch() -> None:
    current = _zero_ur5e_current_positions()
    trajectory = _make_guard_trajectory(
        [
            _make_guard_point(1.0, [0.30, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _make_guard_point(2.0, [0.32, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    )

    ok, guarded, status = ur5e_rtde_trajectory_server.prepare_guarded_trajectory(trajectory, current)

    assert ok is False
    assert guarded is None
    assert status["state"] == "blocked"
    assert status["start_delta_rad"] == pytest.approx(0.30)
    assert status["start_delta_joint"] == "shoulder_pan_joint"
    assert "trajectory start differs from hardware" in status["blocked_reason"]


def test_ur5e_rtde_trajectory_helper_writes_detailed_status_json(tmp_path) -> None:
    status = ur5e_rtde_trajectory_server._trajectory_status_base()
    status.update(
        state="ready",
        start_delta_rad=0.01,
        first_point_time=0.25,
        min_point_spacing=0.04,
        max_segment_velocity_rad_s=0.12,
        time_scale_applied=1.5,
        rtde_result="True",
    )
    path = tmp_path / "cais_ur5e_rtde_trajectory_status.json"

    ur5e_rtde_trajectory_server._atomic_json_write(path, status)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["state"] == "ready"
    assert payload["start_delta_rad"] == pytest.approx(0.01)
    assert payload["first_point_time"] == pytest.approx(0.25)
    assert payload["min_point_spacing"] == pytest.approx(0.04)
    assert payload["max_segment_velocity_rad_s"] == pytest.approx(0.12)
    assert payload["time_scale_applied"] == pytest.approx(1.5)
    assert payload["rtde_result"] == "True"


def test_ur5e_rtde_trajectory_server_retimes_fast_trajectory() -> None:
    current = _zero_ur5e_current_positions()
    trajectory = _make_guard_trajectory(
        [
            _make_guard_point(0.0, [0.0] * 6),
            _make_guard_point(0.1, [0.4, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    )

    ok, guarded, status = ur5e_rtde_trajectory_server.prepare_rtde_trajectory(trajectory, current)

    assert ok is True
    assert guarded is not None
    assert status["state"] == "ready"
    assert status["time_scale_applied"] > 1.0
    assert status["max_segment_velocity_rad_s"] <= (
        ur5e_rtde_trajectory_server.UR5E_RTDE_MAX_JOINT_VEL_RAD_S + 1e-9
    )


def test_ur5e_rtde_trajectory_server_rejects_mismatched_joints() -> None:
    current = _zero_ur5e_current_positions()
    trajectory = _make_guard_trajectory(
        [_make_guard_point(1.0, [0.0] * 6)],
        joint_names=[
            "bad_shoulder_pan_joint",
            "bad_shoulder_lift_joint",
            "bad_elbow_joint",
            "bad_wrist_1_joint",
            "bad_wrist_2_joint",
            "bad_wrist_3_joint",
        ],
    )

    ok, guarded, status = ur5e_rtde_trajectory_server.prepare_rtde_trajectory(trajectory, current)

    assert ok is False
    assert guarded is None
    assert status["state"] == "blocked"
    assert status["blocked_reason"]


def test_ur5e_rtde_movej_path_preserves_joint_order_and_final_blend() -> None:
    trajectory = _make_guard_trajectory(
        [
            _make_guard_point(1.0, [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]),
            _make_guard_point(2.0, [0.11, 0.12, 0.13, 0.14, 0.15, 0.16]),
        ],
        joint_names=list(ur5e_rtde_trajectory_server.ARM_JOINTS),
    )

    path = ur5e_rtde_trajectory_server.rtde_movej_path(trajectory)

    assert len(path) == 2
    assert path[0][:6] == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    assert path[0][6] == pytest.approx(ur5e_rtde_trajectory_server.UR5E_RTDE_MOVEJ_SPEED_RAD_S)
    assert path[0][7] == pytest.approx(ur5e_rtde_trajectory_server.UR5E_RTDE_MOVEJ_ACCEL_RAD_S2)
    assert path[0][8] == pytest.approx(ur5e_rtde_trajectory_server.UR5E_RTDE_INTERMEDIATE_BLEND_RAD)
    assert path[1][8] == pytest.approx(0.0)


def test_ur5e_rtde_execute_movej_path_uses_positional_async_flag() -> None:
    path = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 0.20, 0.0]]

    class FakeControl:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def moveJ(self, *args: object, **kwargs: object) -> bool:
            self.calls.append((args, kwargs))
            return True

    fake_control = FakeControl()
    fake_node = SimpleNamespace(control=fake_control)

    result, mode = ur5e_rtde_trajectory_server.UR5eRTDETrajectoryServer._execute_movej_path(fake_node, path)

    assert result == "True"
    assert mode == "asynchronous_positional"
    assert fake_control.calls == [((path, True), {})]


def test_ur5e_rtde_execute_movej_path_falls_back_to_keyword_async() -> None:
    path = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 0.20, 0.0]]

    class FakeControl:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def moveJ(self, *args: object, **kwargs: object) -> bool:
            self.calls.append((args, kwargs))
            if len(args) == 2:
                raise TypeError("positional async unsupported")
            return bool(kwargs.get("asynchronous"))

    fake_control = FakeControl()
    fake_node = SimpleNamespace(control=fake_control)

    result, mode = ur5e_rtde_trajectory_server.UR5eRTDETrajectoryServer._execute_movej_path(fake_node, path)

    assert result == "True"
    assert mode == "asynchronous_keyword"
    assert fake_control.calls == [((path, True), {}), ((path,), {"asynchronous": True})]


def test_hardware_controller_replays_taught_function_step_from_taught_functions_root(
    tmp_path,
) -> None:
    old_root = HardwarePickPlaceController.taught_functions_root
    HardwarePickPlaceController.taught_functions_root = tmp_path
    try:
        path = tmp_path / "ur5e" / "place_insert" / "nist_assembly_board__hardware.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "robot": "ur5e",
                    "function_name": "place_insert",
                    "name": "nist_assembly_board",
                    "capture_source": "hardware",
                    "replay_targets": ["hardware", "digital_twin"],
                    "steps": [
                        {
                            "step_name": "retreat_pose",
                            "primitive": "move_relative",
                            "waypoint": {
                                "joint_names": ["j1", "j2"],
                                "joint_positions": [0.1, 0.2],
                                "pose": None,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        controller = HardwarePickPlaceController.__new__(HardwarePickPlaceController)
        controller.robot_name = "ur5e"
        calls: list[tuple[list[float], float]] = []
        controller.move_joints = lambda positions, duration_sec=2.0: calls.append(
            ([float(value) for value in positions], float(duration_sec))
        ) or True

        result = controller.replay_taught_function_step(
            "place_insert",
            "nist_assembly_board",
            "retreat_pose",
        )

        assert result["success"] is True
        assert calls == [([0.1, 0.2], 2.0)]
        assert Path(str(result["file"])) == path
    finally:
        HardwarePickPlaceController.taught_functions_root = old_root


def test_hardware_controller_reports_missing_taught_function_file(tmp_path) -> None:
    old_root = HardwarePickPlaceController.taught_functions_root
    HardwarePickPlaceController.taught_functions_root = tmp_path
    try:
        controller = HardwarePickPlaceController.__new__(HardwarePickPlaceController)
        controller.robot_name = "ur5e"

        result = controller.replay_taught_function_step("move_home", "default", "home")

        assert result["success"] is False
        assert "taught function file not found" in result["message"]
        assert "move_home/default__hardware.json" in result["message"]
    finally:
        HardwarePickPlaceController.taught_functions_root = old_root


def test_hardware_controller_replay_function_payload_accepts_delay_between_waypoints(
    monkeypatch,
) -> None:
    controller = HardwarePickPlaceController.__new__(HardwarePickPlaceController)
    calls: list[tuple[str, object]] = []
    controller.move_joints = lambda positions, duration_sec=2.0: calls.append(
        ("move", [float(value) for value in positions])
    ) or True
    monkeypatch.setattr(
        "cais_spade_llm.resources.robot.gazebo_pick_place_controller.time.sleep",
        lambda seconds: calls.append(("delay", float(seconds))),
    )

    result = controller.replay_function_payload(
        {
            "steps": [
                {
                    "primitive": "move_cartesian",
                    "waypoint": {"joint_positions": [0.1, 0.2]},
                },
                {"primitive": "delay", "params": {"duration_sec": 0.5}},
                {
                    "primitive": "move_relative",
                    "waypoint": {"joint_positions": [0.3, 0.4]},
                },
            ],
        }
    )

    assert result["success"] is True
    assert calls == [
        ("move", [0.1, 0.2]),
        ("delay", pytest.approx(0.5)),
        ("move", [0.3, 0.4]),
    ]


class FakeRobotTaskAgent:
    execution_mode = "physical"

    class Logger:
        def debug(self, *_args, **_kwargs) -> None:
            return None

        def info(self, *_args, **_kwargs) -> None:
            return None

        def warning(self, *_args, **_kwargs) -> None:
            return None

        def error(self, *_args, **_kwargs) -> None:
            return None

        def exception(self, *_args, **_kwargs) -> None:
            return None

    def __init__(self) -> None:
        self.logger = self.Logger()
        self._held_part = None
        self._current_state = "idle"
        self._position: dict[str, float] = {}
        self._gripper_state = "open"
        self._bridge_pose_ref = None
        self._task_ctx: dict[str, object] = {}
        self.primitive_calls: list[tuple[str, dict[str, object]]] = []
        self.taught_calls: list[tuple[str, str, str]] = []

    async def _maybe_inject_failure(self, **_kwargs):
        return None

    async def _execute_primitive(self, primitive: str, params: dict[str, object]):
        self.primitive_calls.append((primitive, dict(params)))
        if primitive == "detect_parts":
            return {"success": True, "message": "detected"}
        if primitive == "compute_pick_targets":
            return {
                "success": True,
                "part_name": params.get("part_name"),
                "model_name": "gear_large",
                "tx": 0.1,
                "ty": 0.2,
                "tz": 0.3,
                "pick_z": 0.4,
                "travel_z": 0.8,
                "part_height": 0.02,
                "tcp_offset_z": 0.0,
                "pick_tcp_z": 0.4,
                "origin_pose": {"x": 0.1, "y": 0.2, "z": 0.3},
                "approach_pose": {"x": 0.1, "y": 0.2, "z": 0.8},
                "target_pose": {"x": 0.1, "y": 0.2, "z": 0.4},
            }
        if primitive == "open_gripper":
            return {"success": True, "message": "opened"}
        if primitive == "move_cartesian":
            return {"success": False, "message": "move_cartesian should use taught function replay"}
        return {"success": True, "message": primitive}

    async def _execute_taught_function_step(
        self,
        *,
        function_name: str,
        taught_function_name: str,
        step_name: str,
    ):
        self.taught_calls.append((function_name, taught_function_name, step_name))
        return {"success": True, "message": f"taught {step_name}"}

    def _task_failure(self, message: str, *, step: str, observations=None, failure_context=None):
        return {
            "status": "failed",
            "content": message,
            "step": step,
            "observations": observations,
            "failure_context": failure_context,
        }


def test_physical_robot_task_routes_motion_steps_to_taught_function_replay() -> None:
    agent = FakeRobotTaskAgent()

    result = asyncio.run(
        execute_robot_task(
            agent,
            "pick_approach",
            origin_resource_location="nist_assembly_board",
            part_name="LG",
        )
    )

    assert result["status"] == "completed"
    assert agent.taught_calls == [
        ("pick_approach", "nist_assembly_board", "approach_pose"),
        ("pick_approach", "nist_assembly_board", "pick_pose"),
    ]
    assert "move_cartesian" not in [primitive for primitive, _params in agent.primitive_calls]


def test_rg2_rtde_maps_position_to_width_force_and_script_body() -> None:
    fake = FakeRTDE()
    gripper = UR5eRG2GripperController(
        hostname="192.168.1.172",
        open_position=0.11,
        close_position=0.02,
        open_width_mm=70.0,
        close_width_mm=10.0,
        open_force=10.0,
        close_force=40.0,
        open_settle_sec=0.0,
        close_settle_sec=0.0,
        rtde_factory=lambda *, hostname: fake,
    )

    assert gripper.width_mm_from_position(0.11) == pytest.approx(70.0)
    assert gripper.width_mm_from_position(0.02) == pytest.approx(10.0)
    assert gripper.width_mm_from_position(0.065) == pytest.approx(40.0)
    assert gripper.width_mm_from_position(0.50) == pytest.approx(70.0)
    assert gripper.force_for_position(0.11) == pytest.approx(10.0)
    assert gripper.force_for_position(0.02) == pytest.approx(40.0)

    width = gripper.command_position(0.11, blocking=False)

    assert width == pytest.approx(70.0)
    assert len(fake.calls) == 1
    name, body = fake.calls[0]
    assert name == "rg2_cmd"
    assert 'rpc_factory("xmlrpc","http://localhost:41414")' in body
    assert "rg.rg_grip(0, 70.0, 10.0)" in body


def test_rg2_rtde_can_disable_remote_control_check() -> None:
    factory = FakeRTDEFactory()

    UR5eRG2GripperController(
        hostname="192.168.1.172",
        disable_remote_control_check=True,
        rtde_factory=factory,
    )

    assert len(factory.kwargs) == 1
    assert factory.kwargs[0]["hostname"] == "192.168.1.172"
    assert "flags" in factory.kwargs[0]


def test_rg2_program_body_wraps_script_for_urscript_interface() -> None:
    program = UR5eRG2GripperController.program_body(70.0, 10.0)

    assert program.startswith("def rg2_cmd():")
    assert 'rpc_factory("xmlrpc","http://localhost:41414")' in program
    assert "rg.rg_grip(0, 70.0, 10.0)" in program
    assert program.rstrip().endswith("end")


def test_rg2_inline_program_body_wraps_send_custom_script_program() -> None:
    fake = FakeRTDE()
    gripper = UR5eRG2GripperController(
        hostname="192.168.1.172",
        open_position=0.11,
        close_position=0.02,
        open_width_mm=70.0,
        close_width_mm=10.0,
        open_force=10.0,
        close_force=40.0,
        open_settle_sec=0.0,
        close_settle_sec=0.0,
        rtde_method="script",
        rtde_factory=lambda *, hostname: fake,
    )

    gripper.command_position(0.11, blocking=False)

    assert len(fake.calls) == 1
    name, body = fake.calls[0]
    assert name == "script"
    assert body.startswith("def program():")
    assert 'rpc_factory("xmlrpc","http://localhost:41414")' in body
    assert "rg.rg_grip(0, 70.0, 10.0)" in body
    assert body.rstrip().endswith("run program")


def test_ur5e_rg2_xmlrpc_url_uses_root_path_not_rpc2() -> None:
    assert _normalize_xmlrpc_path("") == "/"
    assert _normalize_xmlrpc_path("/") == "/"
    assert _normalize_xmlrpc_path("rg") == "/rg"
    assert _xmlrpc_url("192.168.1.172", 41414, "/") == "http://192.168.1.172:41414/"


def test_ur5e_rg2_bridge_can_publish_arm_joint_states_for_dual_sync() -> None:
    args = _build_arg_parser().parse_args(["--publish-arm-joint-states"])
    root = Path(__file__).resolve().parents[1]
    bridge_body = (
        root / "ros2" / "cais_lab_gazebo" / "scripts" / "ur5e_rg2_rtde_gripper.py"
    ).read_text(encoding="utf-8")

    assert args.publish_arm_joint_states is True
    assert "getActualQ()" in bridge_body
    assert '"shoulder_pan_joint"' in bridge_body
    assert "publish_arm_joint_states" in bridge_body


def test_rg2_secondary_program_body_does_not_replace_external_control() -> None:
    program = UR5eRG2GripperController.secondary_program_body(70.0, 10.0)

    assert program.startswith("sec rg2_cmd():")
    assert 'rpc_factory("xmlrpc","http://localhost:41414")' in program
    assert "rg.rg_grip(0, 70.0, 10.0)" in program
    assert program.rstrip().endswith("end")


def test_ur5e_rg2_bridge_maps_position_without_opening_rtde() -> None:
    settings = UR5eRG2GripperControllerSettings(
        open_position=0.11,
        close_position=0.02,
        open_width_mm=70.0,
        close_width_mm=10.0,
        open_force=10.0,
        close_force=40.0,
        open_settle_sec=1.2,
        close_settle_sec=2.0,
    )

    assert _width_mm_from_position(settings, 0.11) == pytest.approx(70.0)
    assert _width_mm_from_position(settings, 0.02) == pytest.approx(10.0)
    assert _width_mm_from_position(settings, 0.065) == pytest.approx(40.0)
    assert _width_mm_from_position(settings, 0.50) == pytest.approx(70.0)
    assert _force_for_position(settings, 0.11) == pytest.approx(10.0)
    assert _force_for_position(settings, 0.02) == pytest.approx(40.0)
    assert _settle_sec_for_position(settings, 0.11) == pytest.approx(1.2)
    assert _settle_sec_for_position(settings, 0.02) == pytest.approx(2.0)


def test_ur5e_rg2_bridge_honors_config_backend_with_secondary_urscript_fallback() -> None:
    parser = _build_arg_parser()
    default_args = parser.parse_args([])
    xmlrpc_args = parser.parse_args(["--backend", "xmlrpc"])
    rtde_args = parser.parse_args(["--backend", "rtde"])
    secondary_args = parser.parse_args(["--backend", "secondary_urscript"])
    urscript_args = parser.parse_args(["--backend", "urscript_interface"])
    auto_args = parser.parse_args(["--backend", "auto"])
    bridge_body = (
        Path(__file__).resolve().parents[1]
        / "ros2"
        / "cais_lab_gazebo"
        / "scripts"
        / "ur5e_rg2_rtde_gripper.py"
    ).read_text(encoding="utf-8")

    assert default_args.backend is None
    assert xmlrpc_args.backend == "xmlrpc"
    assert rtde_args.backend == "rtde"
    assert secondary_args.backend == "secondary_urscript"
    assert urscript_args.backend == "urscript_interface"
    assert auto_args.backend == "auto"
    assert _resolve_backend(default_args.backend, {}) == "secondary_urscript"
    assert _resolve_backend(default_args.backend, {"backend": "xmlrpc"}) == "xmlrpc"
    assert _resolve_backend(default_args.backend, {"backend": "rtde"}) == "rtde"
    assert _resolve_backend(default_args.backend, {"backend": "urscript_interface"}) == "urscript_interface"
    assert _resolve_backend(default_args.backend, {"backend": "auto"}) == "auto"
    assert _resolve_backend(secondary_args.backend, {"backend": "rtde"}) == "secondary_urscript"
    assert _resolve_backend(default_args.backend, {"backend": "bad"}) == "secondary_urscript"
    assert default_args.status_file == str(_default_status_path())
    assert "backend={self.backend}" in bridge_body
    assert "XMLRPC RG2 command" in bridge_body
    assert "xmlrpc_url={self.xmlrpc_url}" in bridge_body
    assert "RTDE RG2 command" in bridge_body
    assert "rtde_timeout_sec" in bridge_body
    assert "/urscript_interface/script_command" in bridge_body
    assert "_command_width_urscript_interface" in bridge_body
    assert "secondary_program_body(width_mm, force)" in bridge_body
    assert "secondary_urscript_port" in bridge_body
    assert "status_file={self.status_file}" in bridge_body
    assert "_rtde_command_worker" in bridge_body
    assert 'mp.get_context("spawn")' in bridge_body
    assert "UR5eRG2GripperController.from_settings(settings)" in bridge_body


def test_ur5e_rg2_bridge_status_payload_records_command_details() -> None:
    payload = _rg2_status_payload(
        source="action",
        joint_name="ur5e_rg2_finger_width",
        position=0.02,
        width_mm=10.0,
        force=40.0,
        backend="secondary_urscript",
        used_backend="secondary_urscript",
        rtde_method="script",
        disable_remote_control_check=True,
        xmlrpc_url="http://192.168.1.172:41414/",
        hostname="192.168.1.172",
        state="success",
        success=True,
    )

    assert payload["source"] == "action"
    assert payload["joint"] == "ur5e_rg2_finger_width"
    assert payload["position"] == pytest.approx(0.02)
    assert payload["width_mm"] == pytest.approx(10.0)
    assert payload["force"] == pytest.approx(40.0)
    assert payload["backend"] == "secondary_urscript"
    assert payload["used_backend"] == "secondary_urscript"
    assert payload["fallback_error"] == ""
    assert payload["rtde_method"] == "script"
    assert payload["disable_remote_control_check"] is True
    assert payload["xmlrpc_url"] == "http://192.168.1.172:41414/"
    assert payload["hostname"] == "192.168.1.172"
    assert payload["state"] == "success"
    assert payload["success"] is True
    assert payload["error"] == ""
    assert payload["updated_at"] > 0.0


def test_ur5e_real_rg2_config_uses_xmlrpc_backend() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "cais_spade_llm"
        / "initialization"
        / "resources"
        / "robot_ur5e.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gripper = config["ur5e"]["real"]["controller"]["gripper"]

    assert gripper["backend"] == "xmlrpc"
    assert gripper["rtde"]["method"] == "script"
    assert gripper["rtde"]["disable_remote_control_check"] is True
    assert gripper["xmlrpc"]["port"] == 41414
    assert gripper["xmlrpc"]["path"] == "/"
    assert gripper["action"] == "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory"
    assert gripper["topic"] == "/ur5e_rg2_gripper_traj_controller/joint_trajectory"


def test_hardware_stack_status_includes_ur5e_gripper(monkeypatch) -> None:
    bridge = SystemBridge()
    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")
    monkeypatch.setattr(
        bridge,
        "_ur5e_rtde_trajectory_status",
        lambda: {
            "state": "ready",
            "message": "UR5e RTDE trajectory server ready",
            "max_segment_velocity_rad_s": 0.12,
            "time_scale_applied": 1.0,
            "blocked_reason": "",
        },
    )

    status = bridge.hardware_stack_status("ur5e")

    assert status["overall"] == "running"
    assert status["driver"] == "running"
    assert status["rtde_trajectory_server"] == "ready"
    assert status["rtde_trajectory_message"] == "UR5e RTDE trajectory server ready"
    assert status["rtde_trajectory_max_segment_velocity_rad_s"] == pytest.approx(0.12)
    assert status["gripper"] == "running"
    assert status["gripper_action"] == "ready"
    assert status["moveit"] == "running"


def test_digital_twin_status_includes_ur5e_gripper(monkeypatch) -> None:
    bridge = SystemBridge()
    process_names = set(
        bridge._DIGITAL_TWIN_TARGETS["ur5e only"]["hardware_processes"].values()
    )
    monkeypatch.setattr(
        bridge,
        "ros2_proc_status",
        lambda name: "running" if name in process_names else "stopped",
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_rtde_trajectory_status",
        lambda: {
            "state": "ready",
            "message": "UR5e RTDE trajectory server ready",
            "max_segment_velocity_rad_s": 0.10,
            "time_scale_applied": 1.0,
            "blocked_reason": "",
        },
    )

    status = bridge.digital_twin_statuses()["ur5e only"]["hardware"]["status"]

    assert status["overall"] == "running"
    assert status["driver"] == "running"
    assert status["rtde_trajectory_server"] == "ready"
    assert status["rtde_trajectory_message"] == "UR5e RTDE trajectory server ready"
    assert status["rtde_trajectory_max_segment_velocity_rad_s"] == pytest.approx(0.10)
    assert status["gripper"] == "running"
    assert status["gripper_action"] == "ready"
    assert status["moveit"] == "running"


def test_digital_twin_ur5e_hardware_stack_uses_rtde_gripper_moveit_order(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["ur5e only"]
    events: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        bridge,
        "hardware_connection_statuses",
        lambda force=False: {"ur5e": {"reachable": True, "ip": "192.168.1.172", "message": "OK"}},
    )
    monkeypatch.setattr(bridge, "_digital_twin_sim_mode", lambda _target: "monitor")
    monkeypatch.setattr(
        bridge,
        "_start_ur5e_rtde_trajectory_server",
        lambda process_name, **_kwargs: events.append(("rtde", process_name, "")) or None,
    )

    def fake_start(process_name: str, launch_name: str, **_kwargs) -> None:
        events.append(("start", process_name, launch_name))
        return None

    monkeypatch.setattr(bridge, "_start_digital_twin_launch", fake_start)
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_topic_publisher",
        lambda topic, **_kwargs: events.append(("wait_topic", topic, "")) or None,
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_action",
        lambda action, **_kwargs: events.append(("wait_action", action, "")) or None,
    )

    err = bridge._start_digital_twin_hardware_stack(
        "ur5e only",
        cfg,
        ros_domain_id=42,
    )

    assert err is None
    assert events == [
        ("rtde", "digital_twin_ur5e_only_hardware_ur5e_rtde_trajectory_server", ""),
        ("start", "digital_twin_ur5e_only_hardware_ur5e_moveit", "hardware_ur5e_moveit"),
        ("wait_action", "/execute_trajectory", ""),
        ("wait_topic", "/joint_states", ""),
        ("start", "digital_twin_ur5e_only_hardware_ur5e_rg2_gripper", "hardware_ur5e_rg2_gripper"),
        ("wait_action", "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory", ""),
    ]


def test_digital_twin_ur5e_hardware_stack_retries_ros2_daemon_once(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["ur5e only"]
    topic_wait_calls = 0
    daemon_restarts = 0

    monkeypatch.setattr(
        bridge,
        "hardware_connection_statuses",
        lambda force=False: {"ur5e": {"reachable": True, "ip": "192.168.1.172", "message": "OK"}},
    )
    monkeypatch.setattr(bridge, "_digital_twin_sim_mode", lambda _target: "monitor")
    monkeypatch.setattr(bridge, "_start_digital_twin_launch", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bridge, "_start_ur5e_rtde_trajectory_server", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bridge, "_wait_for_ros_action", lambda *_args, **_kwargs: None)

    def fake_wait_for_ros_topic_publisher(*_args, **_kwargs) -> str | None:
        nonlocal topic_wait_calls
        topic_wait_calls += 1
        if topic_wait_calls == 1:
            return "/joint_states publisher state not available within 12s"
        return None

    def fake_restart_daemon(**_kwargs) -> None:
        nonlocal daemon_restarts
        daemon_restarts += 1
        return None

    monkeypatch.setattr(bridge, "_wait_for_ros_topic_publisher", fake_wait_for_ros_topic_publisher)
    monkeypatch.setattr(bridge, "_restart_ros2_daemon_for_discovery", fake_restart_daemon)

    err = bridge._start_digital_twin_hardware_stack(
        "ur5e only",
        cfg,
        ros_domain_id=42,
    )

    assert err is None
    assert topic_wait_calls == 2
    assert daemon_restarts == 1


def test_digital_twin_sync_includes_ur5e_rg2_gripper() -> None:
    gripper = digital_twin_sync.ROBOTS["ur5e"]["gripper"]

    assert gripper["gazebo_joint"] == "ur5e_rg2_finger_width"
    assert gripper["gazebo_trajectory_topics"] == [
        "/ur5e_rg2_gripper_traj_controller/joint_trajectory",
    ]


def test_digital_twin_sync_reports_exact_missing_joints_and_topics() -> None:
    sync_path = (
        Path(__file__).resolve().parents[1]
        / "ros2"
        / "cais_lab_gazebo"
        / "scripts"
        / "digital_twin_sync.py"
    )
    body = sync_path.read_text(encoding="utf-8")

    assert "hardware /joint_states missing required {robot} joints" in body
    assert "hardware /joint_states has no {robot} arm joints yet" in body
    assert "unmatched_seen_names" in body
    assert "gazebo trajectory controller not connected for {robot}: {topics}" in body


def test_digital_twin_sync_distinguishes_unrelated_joint_state_messages() -> None:
    ur5e_snapshot = {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": 0.0,
        "elbow_joint": 0.0,
        "wrist_1_joint": 0.0,
        "wrist_2_joint": 0.0,
        "wrist_3_joint": 0.0,
    }
    xarm6_snapshot = {
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 0.0,
        "joint4": 0.0,
        "joint5": 0.0,
        "joint6": 0.0,
    }

    assert digital_twin_sync._gazebo_joint_match_count(ur5e_snapshot, "xarm6") == 0
    assert digital_twin_sync._gazebo_joint_match_count(xarm6_snapshot, "ur5e") == 0
    assert digital_twin_sync._gazebo_joint_match_count(xarm6_snapshot, "xarm6") == 6
    assert digital_twin_sync._gazebo_joint_match_count(ur5e_snapshot, "ur5e") == 6


def test_digital_twin_sync_hardware_update_accepts_split_arm_and_gripper_states() -> None:
    xarm_gripper_item, xarm_gripper_joint, xarm_gripper_position = (
        digital_twin_sync._hardware_update_from_snapshot({"drive_joint": 0.42}, "xarm6")
    )
    assert xarm_gripper_item["diagnostic"] == "no_matching_hardware_joints"
    assert xarm_gripper_joint == "xarm6_drive_joint"
    assert xarm_gripper_position == pytest.approx(0.42)

    xarm_arm_item, _joint, _position = digital_twin_sync._hardware_update_from_snapshot(
        {
            "joint1": 1.0,
            "joint2": 2.0,
            "joint3": 3.0,
            "joint4": 4.0,
            "joint5": 5.0,
            "joint6": 6.0,
        },
        "xarm6",
        remembered_gripper_joint=xarm_gripper_joint,
        remembered_gripper_position=xarm_gripper_position,
    )
    assert xarm_arm_item["joint_names"] == [
        "xarm6_joint1",
        "xarm6_joint2",
        "xarm6_joint3",
        "xarm6_joint4",
        "xarm6_joint5",
        "xarm6_joint6",
    ]
    assert xarm_arm_item["positions"] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert xarm_arm_item["gripper_joint"] == "xarm6_drive_joint"
    assert xarm_arm_item["gripper_position"] == pytest.approx(0.42)

    ur_gripper_item, ur_gripper_joint, ur_gripper_position = (
        digital_twin_sync._hardware_update_from_snapshot({"ur5e_rg2_finger_width": 0.08}, "ur5e")
    )
    assert ur_gripper_item["diagnostic"] == "no_matching_hardware_joints"
    assert ur_gripper_joint == "ur5e_rg2_finger_width"
    assert ur_gripper_position == pytest.approx(0.08)

    ur_arm_item, _joint, _position = digital_twin_sync._hardware_update_from_snapshot(
        {
            "shoulder_pan_joint": 1.0,
            "shoulder_lift_joint": 2.0,
            "elbow_joint": 3.0,
            "wrist_1_joint": 4.0,
            "wrist_2_joint": 5.0,
            "wrist_3_joint": 6.0,
        },
        "ur5e",
        remembered_gripper_joint=ur_gripper_joint,
        remembered_gripper_position=ur_gripper_position,
    )
    assert ur_arm_item["joint_names"] == [
        "ur5e_shoulder_pan_joint",
        "ur5e_shoulder_lift_joint",
        "ur5e_elbow_joint",
        "ur5e_wrist_1_joint",
        "ur5e_wrist_2_joint",
        "ur5e_wrist_3_joint",
    ]
    assert ur_arm_item["positions"] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert ur_arm_item["gripper_joint"] == "ur5e_rg2_finger_width"
    assert ur_arm_item["gripper_position"] == pytest.approx(0.08)


def _paired_dual_recording() -> dict[str, object]:
    return {
        "target": "dual robots",
        "robot": "dual robots",
        "recording_type": "paired_dual_robots",
        "robots": {
            "xarm6": {
                "joint_names": [
                    "xarm6_joint1",
                    "xarm6_joint2",
                    "xarm6_joint3",
                    "xarm6_joint4",
                    "xarm6_joint5",
                    "xarm6_joint6",
                ],
            },
            "ur5e": {
                "joint_names": [
                    "ur5e_shoulder_pan_joint",
                    "ur5e_shoulder_lift_joint",
                    "ur5e_elbow_joint",
                    "ur5e_wrist_1_joint",
                    "ur5e_wrist_2_joint",
                    "ur5e_wrist_3_joint",
                ],
            },
        },
        "waypoints": [
            {
                "robots": {
                    "xarm6": {"positions": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], "gripper": 0.2},
                    "ur5e": {"positions": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2], "gripper": 0.10},
                }
            },
            {
                "robots": {
                    "xarm6": {"positions": [0.15, 0.15, 0.15, 0.15, 0.15, 0.15], "gripper": 0.8},
                    "ur5e": {"positions": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25], "gripper": 0.02},
                }
            },
        ],
    }


def _paired_replay_args(
    recording_file: Path,
    replay_target: str = "both",
    prepared_file: Path | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        recording_file=str(recording_file),
        prepared_file=str(prepared_file or ""),
        robot="dual robots",
        replay_target=replay_target,
        gazebo_domain_id=41,
        hardware_domain_id=42,
        waypoint_duration_sec=2.0,
        max_joint_delta_deg=175.0,
        max_joint_vel_deg_s=25.0,
    )


def _ur5e_hardware_joint_names() -> list[str]:
    return [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]


def _planned_ur5e_trajectory(
    start_positions: list[float],
    target_positions: list[float],
    *,
    joint_names: list[str] | None = None,
    duration: float = 1.0,
) -> dict[str, object]:
    names = list(joint_names or _ur5e_hardware_joint_names())
    return {
        "joint_names": names,
        "points": [
            {
                "positions": [float(value) for value in start_positions],
                "velocities": [0.0] * len(names),
                "accelerations": [0.0] * len(names),
                "time": 0.0,
            },
            {
                "positions": [float(value) for value in target_positions],
                "velocities": [0.0] * len(names),
                "accelerations": [0.0] * len(names),
                "time": float(duration),
            },
        ],
    }


def _successful_ur5e_plan_result(
    start_positions: list[float],
    target_positions: list[float],
    *,
    waypoint_index: int | None = None,
    joint_names: list[str] | None = None,
) -> dict[str, object]:
    return {
        "success": True,
        "message": "/move_action: planned; action accepted; action succeeded; moveit_error_code=1.",
        "mode": "planned",
        "waypoint_index": waypoint_index,
        "trajectory": _planned_ur5e_trajectory(
            start_positions,
            target_positions,
            joint_names=joint_names,
        ),
    }


def _single_ur5e_replay_args(recording_file: Path, replay_target: str = "hardware") -> SimpleNamespace:
    return SimpleNamespace(
        recording_file=str(recording_file),
        robot="ur5e",
        replay_target=replay_target,
        gazebo_domain_id=41,
        hardware_domain_id=42,
        waypoint_duration_sec=2.0,
        max_joint_delta_deg=175.0,
        max_joint_vel_deg_s=25.0,
    )


def test_single_ur5e_replay_uses_moveit_plan_and_execute(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    recording = {
        "target": "ur5e only",
        "robot": "ur5e",
        "recording_type": "single_robot",
        "saved_steps": 2,
        "joint_names": [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        "waypoints": [
            {"positions": [0.10, 0.10, 0.10, 0.10, 0.10, 0.10]},
            {"positions": [0.15, 0.15, 0.15, 0.15, 0.15, 0.15]},
        ],
    }
    path = tmp_path / "ur5e_single.json"
    path.write_text(json.dumps(recording), encoding="utf-8")

    move_calls: list[dict[str, object]] = []
    publish_calls: list[object] = []

    monkeypatch.setattr(
        digital_twin_sync,
        "_read_snapshot",
        lambda *_args, **_kwargs: {
            "success": True,
            "snapshot": {
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": 0.0,
                "elbow_joint": 0.0,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
            },
        },
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_move_group_action",
        lambda *_args, **_kwargs: {"success": True, "message": "/move_action: action server available."},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_trajectory",
        lambda *args, **_kwargs: publish_calls.append(args) or {"success": True, "message": "trajectory published."},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_execute_move_group_joint_goal",
        lambda domain_id, group_name, joint_names, start_positions, target_positions, **kwargs: move_calls.append(
            {
                "domain_id": domain_id,
                "group_name": group_name,
                "joint_names": list(joint_names),
                "start_positions": list(start_positions),
                "target_positions": list(target_positions),
                "waypoint_index": kwargs.get("waypoint_index"),
            }
        )
        or {
            "success": True,
            "message": "/move_action: plan_and_execute; action accepted; action succeeded; moveit_error_code=1.",
            "mode": "plan_and_execute",
        },
    )

    code = digital_twin_sync.run_replay(_single_ur5e_replay_args(path))
    output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert code == 0
    assert output["success"] is True
    assert output["saved_steps"] == 2
    assert output["waypoints"] == 2
    assert "ur5e/hardware_arm used MoveIt plan_and_execute" in output["message"]
    assert "group=ur_manipulator" in output["message"]
    assert publish_calls == []
    assert len(move_calls) == 2
    assert move_calls[0]["start_positions"] == [0.0] * 6
    assert move_calls[1]["start_positions"] == recording["waypoints"][0]["positions"]


def test_ur5e_move_group_plan_results_stitch_to_strict_timeline() -> None:
    names = _ur5e_hardware_joint_names()
    plan_results = [
        _successful_ur5e_plan_result([0.0] * 6, [0.2] * 6, waypoint_index=1),
        _successful_ur5e_plan_result([0.2] * 6, [0.25] * 6, waypoint_index=2),
    ]

    stitched = digital_twin_sync._stitch_ur5e_move_group_plan_results(names, plan_results)

    assert stitched["success"] is True
    points = [dict(point) for point in list(stitched["points"])]
    times = [float(point["time"]) for point in points]
    assert times == sorted(times)
    assert all(b > a for a, b in zip(times, times[1:]))
    assert points[0]["positions"] == pytest.approx([0.0] * 6)
    assert points[1]["positions"] == pytest.approx([0.2] * 6)
    assert points[2]["positions"] == pytest.approx([0.25] * 6)
    assert points[-1]["positions"] == pytest.approx([0.25] * 6)
    assert float(points[-1]["time"]) == pytest.approx(
        float(points[-2]["time"]) + digital_twin_sync.UR5E_TEACH_REPLAY_FINAL_HOLD_SEC
    )


def test_ur5e_move_group_plan_stitch_skips_duplicate_segment_start() -> None:
    names = _ur5e_hardware_joint_names()
    plan_results = [
        _successful_ur5e_plan_result([0.0] * 6, [0.2] * 6, waypoint_index=1),
        _successful_ur5e_plan_result([0.2] * 6, [0.25] * 6, waypoint_index=2),
    ]

    stitched = digital_twin_sync._stitch_ur5e_move_group_plan_results(names, plan_results)

    points = [dict(point) for point in list(stitched["points"])]
    duplicate_waypoint_points = [
        point for point in points
        if point["positions"] == pytest.approx([0.2] * 6)
    ]
    assert len(duplicate_waypoint_points) == 1


def test_ur5e_move_group_plan_stitch_rejects_mismatched_joints() -> None:
    names = _ur5e_hardware_joint_names()
    bad_names = list(names)
    bad_names[-1] = "wrong_joint"
    plan_results = [
        _successful_ur5e_plan_result(
            [0.0] * 6,
            [0.2] * 6,
            waypoint_index=1,
            joint_names=bad_names,
        )
    ]

    stitched = digital_twin_sync._stitch_ur5e_move_group_plan_results(names, plan_results)

    assert stitched["success"] is False
    assert "joint names do not match hardware joint names" in stitched["message"]


def test_digital_twin_snapshot_worker_accumulates_interleaved_joint_states() -> None:
    root = Path(__file__).resolve().parents[1]
    body = (
        root / "ros2" / "cais_lab_gazebo" / "scripts" / "digital_twin_sync.py"
    ).read_text(encoding="utf-8")

    assert "self.accumulated_snapshot: dict[str, float] = {}" in body
    assert "self.accumulated_snapshot.update(snapshot)" in body
    assert "_resolve_hardware_positions(" in body
    assert "self.accumulated_snapshot," in body
    assert "HARDWARE_SNAPSHOT_TIMEOUT_SEC" in body


def test_ur5e_gazebo_mirror_uses_smoothing_helpers() -> None:
    assert digital_twin_sync.UR5E_MIRROR_POINT_TIME_SEC == pytest.approx(0.12)
    assert digital_twin_sync.UR5E_MIRROR_MIN_PUBLISH_PERIOD_SEC == pytest.approx(0.05)
    assert digital_twin_sync.UR5E_MIRROR_MIN_JOINT_DELTA_RAD == pytest.approx(0.0010)
    assert digital_twin_sync._mirror_point_time_sec("ur5e") == pytest.approx(
        digital_twin_sync.UR5E_MIRROR_POINT_TIME_SEC
    )
    assert digital_twin_sync._mirror_point_time_sec("xarm6") == pytest.approx(
        digital_twin_sync.MIRROR_POINT_TIME_SEC
    )
    assert digital_twin_sync._mirror_min_publish_period_sec("ur5e") == pytest.approx(
        digital_twin_sync.UR5E_MIRROR_MIN_PUBLISH_PERIOD_SEC
    )
    assert digital_twin_sync._mirror_min_publish_period_sec("xarm6") == pytest.approx(
        digital_twin_sync.MIRROR_MIN_PUBLISH_PERIOD_SEC
    )
    assert digital_twin_sync._mirror_min_joint_delta_rad("ur5e") == pytest.approx(
        digital_twin_sync.UR5E_MIRROR_MIN_JOINT_DELTA_RAD
    )
    assert digital_twin_sync._mirror_min_joint_delta_rad("xarm6") == pytest.approx(
        digital_twin_sync.MIRROR_MIN_JOINT_DELTA_RAD
    )

    ok, reason = digital_twin_sync._should_publish_mirror_update(
        "ur5e",
        positions=[0.0] * 6,
        last_positions=None,
        last_publish_ts=0.0,
        now=10.0,
    )
    assert ok is True
    assert reason == ""

    ok, reason = digital_twin_sync._should_publish_mirror_update(
        "ur5e",
        positions=[0.0009] * 6,
        last_positions=[0.0] * 6,
        last_publish_ts=9.0,
        now=10.0,
    )
    assert ok is False
    assert reason == "below_delta"

    ok, reason = digital_twin_sync._should_publish_mirror_update(
        "ur5e",
        positions=[0.05] * 6,
        last_positions=[0.0] * 6,
        last_publish_ts=9.96,
        now=10.0,
    )
    assert ok is False
    assert reason == "rate_limited"

    ok, reason = digital_twin_sync._should_publish_mirror_update(
        "ur5e",
        positions=[0.05] * 6,
        last_positions=[0.0] * 6,
        last_publish_ts=9.0,
        now=10.0,
    )
    assert ok is True
    assert reason == ""

    root = Path(__file__).resolve().parents[1]
    body = (
        root / "ros2" / "cais_lab_gazebo" / "scripts" / "digital_twin_sync.py"
    ).read_text(encoding="utf-8")
    assert "mirror_min_joint_delta_rad=" in body
    assert "mirror_max_joint_delta_rad=" in body
    assert "last_skip_reason=" in body


def test_dual_replay_timing_defaults_are_stable_slow() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    assert digital_twin_sync.REPLAY_SPEED_SCALE == pytest.approx(1.0)
    assert digital_twin_sync.DEFAULT_REPLAY_WAYPOINT_DURATION_SEC == pytest.approx(2.0)
    assert digital_twin_sync.MAX_REPLAY_JOINT_VEL_DEG_S == pytest.approx(25.0)
    assert digital_twin_sync.UR5E_REPLAY_MAX_JOINT_VEL_DEG_S == pytest.approx(10.0)
    assert digital_twin_sync.MOVE_GROUP_REPLAY_VELOCITY_SCALING == pytest.approx(0.25)
    assert SystemBridge._DIGITAL_TWIN_REPLAY_SPEED_SCALE == pytest.approx(1.0)
    assert SystemBridge._DIGITAL_TWIN_REPLAY_WAYPOINT_DURATION_SEC == pytest.approx(2.0)
    assert SystemBridge._DIGITAL_TWIN_REPLAY_MAX_JOINT_VEL_DEG_S == pytest.approx(25.0)


def test_snapshot_worker_init_failure_returns_detailed_message(monkeypatch) -> None:
    class FakeQueue:
        def __init__(self) -> None:
            self.payload = None

        def put(self, payload) -> None:
            self.payload = dict(payload)

    result_queue = FakeQueue()
    monkeypatch.setattr(
        digital_twin_sync,
        "_init_ros_domain",
        lambda _domain_id: (_ for _ in ()).throw(RuntimeError("rclpy init failed")),
    )

    digital_twin_sync._joint_state_snapshot_worker(
        42,
        "ur5e",
        "hardware",
        0.1,
        result_queue,
    )

    assert result_queue.payload["success"] is False
    assert "hardware /joint_states snapshot failed for ur5e: rclpy init failed" in result_queue.payload["message"]
    assert "ROS_DOMAIN_ID=42" in result_queue.payload["message"]
    assert "topics=/joint_states" in result_queue.payload["message"]


def test_read_snapshot_returns_queued_result_before_killing_slow_worker(monkeypatch) -> None:
    queued = {
        "success": True,
        "snapshot": {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "elbow_joint": 0.0,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    }

    class FakeQueue:
        def get_nowait(self):
            return queued

    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False

        def start(self) -> None:
            pass

        def join(self, timeout=None) -> None:
            pass

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            self.terminated = True

    fake_process = FakeProcess()
    monkeypatch.setattr(digital_twin_sync.mp, "Queue", lambda maxsize=1: FakeQueue())
    monkeypatch.setattr(digital_twin_sync.mp, "Process", lambda *args, **kwargs: fake_process)

    result = digital_twin_sync._read_snapshot(42, "ur5e", "hardware", 20.0)

    assert result == queued
    assert fake_process.terminated is True


def test_read_snapshot_retries_once_after_empty_worker_result(monkeypatch) -> None:
    queued = {
        "success": True,
        "snapshot": {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "elbow_joint": 0.0,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    }

    class EmptyQueue:
        def get_nowait(self):
            raise digital_twin_sync.queue.Empty()

    class SuccessQueue:
        def get_nowait(self):
            return queued

    class FakeProcess:
        starts = 0

        def start(self) -> None:
            FakeProcess.starts += 1

        def join(self, timeout=None) -> None:
            pass

        def is_alive(self) -> bool:
            return False

    queues = [EmptyQueue(), SuccessQueue()]
    monkeypatch.setattr(digital_twin_sync.mp, "Queue", lambda maxsize=1: queues.pop(0))
    monkeypatch.setattr(digital_twin_sync.mp, "Process", lambda *args, **kwargs: FakeProcess())

    result = digital_twin_sync._read_snapshot(42, "ur5e", "hardware", 20.0)

    assert result == queued
    assert FakeProcess.starts == 2


def test_read_snapshot_reports_context_after_repeated_empty_worker_result(monkeypatch) -> None:
    class EmptyQueue:
        def get_nowait(self):
            raise digital_twin_sync.queue.Empty()

    class FakeProcess:
        starts = 0

        def start(self) -> None:
            FakeProcess.starts += 1

        def join(self, timeout=None) -> None:
            pass

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(digital_twin_sync.mp, "Queue", lambda maxsize=1: EmptyQueue())
    monkeypatch.setattr(digital_twin_sync.mp, "Process", lambda *args, **kwargs: FakeProcess())

    result = digital_twin_sync._read_snapshot(42, "ur5e", "hardware", 20.0)

    assert result["success"] is False
    assert FakeProcess.starts == 2
    assert "hardware /joint_states returned no data for ur5e after 2 attempts" in result["message"]
    assert "ROS_DOMAIN_ID=42" in result["message"]
    assert "topics=/joint_states" in result["message"]


def test_paired_dual_replay_preflights_both_robots_before_publish(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    path = tmp_path / "paired.json"
    path.write_text(json.dumps(_paired_dual_recording()), encoding="utf-8")
    read_calls: list[str] = []
    publish_calls: list[str] = []

    def fake_read_snapshot(_domain_id: int, robot: str, source: str, _timeout: float) -> dict[str, object]:
        assert source == "hardware"
        assert _timeout == digital_twin_sync.HARDWARE_SNAPSHOT_TIMEOUT_SEC
        read_calls.append(robot)
        if robot == "ur5e":
            return {"success": False, "message": "ur5e missing hardware state"}
        return {
            "success": True,
            "snapshot": {
                "joint1": 0.0,
                "joint2": 0.0,
                "joint3": 0.0,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": 0.0,
            },
        }

    monkeypatch.setattr(digital_twin_sync, "_read_snapshot", fake_read_snapshot)
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_trajectory",
        lambda *_args, **_kwargs: publish_calls.append("publish") or {"success": True},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_follow_joint_trajectory_action",
        lambda *_args, **_kwargs: publish_calls.append("action") or {"success": True},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_xarm_gripper_sequence",
        lambda *_args, **_kwargs: publish_calls.append("gripper") or {"success": True},
    )

    code = digital_twin_sync.run_replay(_paired_replay_args(path))
    output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert code == 6
    assert read_calls == ["xarm6", "ur5e"]
    assert publish_calls == []
    assert output["success"] is False
    assert "ur5e missing hardware state" in output["message"]


def test_paired_dual_ur5e_replay_slows_large_initial_approach(monkeypatch, tmp_path) -> None:
    recording = _paired_dual_recording()
    for waypoint in recording["waypoints"]:
        waypoint["robots"]["ur5e"]["positions"] = [1.0] * 6

    def fake_read_snapshot(_domain_id: int, robot: str, source: str, _timeout: float) -> dict[str, object]:
        assert robot == "ur5e"
        assert source == "hardware"
        names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        return {"success": True, "snapshot": {name: 0.0 for name in names}}

    monkeypatch.setattr(digital_twin_sync, "_read_snapshot", fake_read_snapshot)

    plan = digital_twin_sync._paired_replay_robot_plan(
        _paired_replay_args(tmp_path / "unused.json"),
        recording,
        "ur5e",
        need_hardware=True,
        step=2.0,
    )

    assert plan["success"] is True
    assert plan["max_joint_delta_deg"] == pytest.approx(math.degrees(1.0))
    assert plan["approach_time"] == pytest.approx(
        math.degrees(1.0) / digital_twin_sync.UR5E_REPLAY_MAX_JOINT_VEL_DEG_S
    )


def test_paired_dual_ur5e_hardware_points_use_positive_current_hold(monkeypatch, tmp_path) -> None:
    recording = _paired_dual_recording()

    def fake_read_snapshot(_domain_id: int, robot: str, source: str, _timeout: float) -> dict[str, object]:
        assert source == "hardware"
        names = (
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            if robot == "xarm6"
            else [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
        )
        return {"success": True, "snapshot": {name: 0.0 for name in names}}

    monkeypatch.setattr(digital_twin_sync, "_read_snapshot", fake_read_snapshot)
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_follow_joint_trajectory_action",
        lambda *_args, **_kwargs: {"success": True, "message": "action server available."},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_move_group_action",
        lambda *_args, **_kwargs: {"success": True, "message": "/move_action: action server available."},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_execute_trajectory_action",
        lambda *_args, **_kwargs: {"success": True, "message": "/execute_trajectory: action server available."},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_xarm_gripper_endpoint",
        lambda *_args, **_kwargs: {"success": True, "message": "service available.", "method": "service"},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_plan_move_group_joint_goal",
        lambda _domain_id, _group_name, joint_names, start_positions, target_positions, **kwargs: (
            _successful_ur5e_plan_result(
                list(start_positions),
                list(target_positions),
                waypoint_index=kwargs.get("waypoint_index"),
                joint_names=list(joint_names),
            )
        ),
    )

    prepared = digital_twin_sync._build_paired_replay_preparation(
        _paired_replay_args(tmp_path / "unused.json"),
        recording,
    )

    assert prepared["success"] is True
    approach_time = float(prepared["approach_time"])
    xarm_points = list(prepared["plans"]["xarm6"]["hardware_points"])
    ur5e_points = list(prepared["plans"]["ur5e"]["hardware_points"])
    assert float(xarm_points[0]["time"]) == digital_twin_sync.HARDWARE_TRAJECTORY_CURRENT_POINT_SEC
    assert float(xarm_points[1]["time"]) == pytest.approx(approach_time)
    assert float(ur5e_points[0]["time"]) == digital_twin_sync.UR5E_HARDWARE_TRAJECTORY_CURRENT_POINT_SEC
    assert float(ur5e_points[0]["time"]) > 0.0
    assert float(ur5e_points[1]["time"]) == pytest.approx(
        digital_twin_sync.UR5E_HARDWARE_TRAJECTORY_CURRENT_POINT_SEC + approach_time
    )
    execute_points = list(prepared["plans"]["ur5e"]["execute_trajectory_points"])
    assert execute_points
    assert all(
        float(second["time"]) > float(first["time"])
        for first, second in zip(execute_points, execute_points[1:])
    )


def test_ur5e_moveit_preflight_retries_goal_acceptance_timeout(monkeypatch, tmp_path) -> None:
    joint_names = _ur5e_hardware_joint_names()
    plan = {
        "hardware_names": joint_names,
        "hardware_positions": [0.0] * len(joint_names),
        "waypoints": [{"positions": [0.1] * len(joint_names)}],
    }
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_move_group_action",
        lambda *_args, **_kwargs: {"success": True, "message": "/move_action: action server available."},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_execute_trajectory_action",
        lambda *_args, **_kwargs: {"success": True, "message": "/execute_trajectory: action server available."},
    )
    monkeypatch.setattr(digital_twin_sync.time, "sleep", lambda _seconds: None)

    def fake_plan(
        _domain_id: int,
        _group_name: str,
        names: list[str],
        start_positions: list[float],
        target_positions: list[float],
        **kwargs,
    ) -> dict[str, object]:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return {
                "success": False,
                "message": "/move_action: action goal acceptance timed out; acceptance_timeout_sec=20.000.",
                "action_name": "/move_action",
                "waypoint_index": kwargs.get("waypoint_index"),
            }
        return _successful_ur5e_plan_result(
            list(start_positions),
            list(target_positions),
            waypoint_index=kwargs.get("waypoint_index"),
            joint_names=list(names),
        )

    monkeypatch.setattr(digital_twin_sync, "_plan_move_group_joint_goal", fake_plan)

    result = digital_twin_sync._preflight_ur5e_move_group_replay(
        _paired_replay_args(tmp_path / "unused.json"),
        plan,
        timeout_sec=digital_twin_sync.MOVE_GROUP_PLAN_TIMEOUT_SEC,
    )

    assert result["success"] is True
    assert len(calls) == 2
    assert calls[0]["acceptance_timeout_sec"] == pytest.approx(
        digital_twin_sync.MOVE_GROUP_GOAL_ACCEPTANCE_TIMEOUT_SEC
    )
    assert result["execute_action_name"] == "/execute_trajectory"
    assert result["execute_trajectory_points"]
    assert [attempt["success"] for attempt in result["move_group_attempt_results"]] == [
        False,
        True,
    ]


def test_paired_dual_replay_blocks_xarm6_when_ur5e_moveit_preflight_fails(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    path = tmp_path / "paired.json"
    path.write_text(json.dumps(_paired_dual_recording()), encoding="utf-8")
    publish_calls: list[str] = []
    action_calls: list[str] = []
    execute_calls: list[int | None] = []

    def fake_read_snapshot(_domain_id: int, robot: str, source: str, _timeout: float) -> dict[str, object]:
        assert source == "hardware"
        names = (
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            if robot == "xarm6"
            else [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
        )
        return {"success": True, "snapshot": {name: 0.0 for name in names}}

    monkeypatch.setattr(digital_twin_sync, "_read_snapshot", fake_read_snapshot)
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_follow_joint_trajectory_action",
        lambda *_args, **_kwargs: {"success": True, "message": "action server available."},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_move_group_action",
        lambda *_args, **_kwargs: {"success": True, "message": "/move_action: action server available."},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_execute_trajectory_action",
        lambda *_args, **_kwargs: {"success": True, "message": "/execute_trajectory: action server available."},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_xarm_gripper_endpoint",
        lambda *_args, **_kwargs: {"success": True, "message": "service available.", "method": "service"},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_trajectory",
        lambda *_args, **_kwargs: publish_calls.append("gazebo") or {"success": True, "message": "trajectory published."},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_follow_joint_trajectory_action",
        lambda _domain_id, action_name, *_args, **_kwargs: action_calls.append(str(action_name)) or {"success": True},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_paired_follow_joint_trajectory_actions",
        lambda *_args, **_kwargs: pytest.fail("paired dual replay must not use paired UR5e follow_joint_trajectory"),
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_xarm_gripper_sequence",
        lambda *_args, **_kwargs: pytest.fail("xarm6 gripper must not move when ur5e MoveIt preflight fails"),
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_plan_move_group_joint_goal",
        lambda *_args, **kwargs: {
            "success": False,
            "message": "/move_action: planned; action accepted; action aborted; moveit_error_code=-4.",
            "mode": "planned",
            "waypoint_index": kwargs.get("waypoint_index"),
        },
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_execute_move_group_joint_goal",
        lambda *_args, **_kwargs: pytest.fail("paired dual replay must not use per-waypoint /move_action execute"),
    )

    code = digital_twin_sync.run_replay(_paired_replay_args(path))
    output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert code == 6
    assert output["success"] is False
    assert publish_calls == []
    assert action_calls == []
    assert execute_calls == []
    assert "ur5e MoveIt preflight failed waypoint 1/2" in output["message"]
    assert "/move_action" in output["message"]
    assert "/execute_trajectory" not in output["message"]


def test_xarm_gripper_endpoint_falls_back_to_gripper_action(monkeypatch) -> None:
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_xarm_gripper_service",
        lambda *_args, **_kwargs: {"success": False, "message": "xarm position service unavailable"},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_gripper_command_action",
        lambda _domain_id, action_name, **_kwargs: {
            "success": True,
            "message": f"{action_name}: action server available.",
            "action": action_name,
        },
    )

    result = digital_twin_sync._wait_xarm_gripper_endpoint(42)

    assert result["success"] is True
    assert result["method"] == "action"
    assert result["action"] == "/xarm6/xarm_gripper/gripper_action"
    assert result["fallback_error"] == "xarm position service unavailable"


def test_xarm_gripper_replay_falls_back_to_gripper_action(monkeypatch) -> None:
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_xarm_gripper_service_sequence",
        lambda *_args, **_kwargs: {"success": False, "message": "xarm position service unavailable"},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_gripper_command_action_sequence",
        lambda _domain_id, action_name, points, **_kwargs: {
            "success": True,
            "message": f"{action_name}: gripper action replay accepted {len(points)} goals.",
            "action": action_name,
        },
    )

    result = digital_twin_sync._publish_xarm_gripper_sequence(
        42,
        [{"positions": [0.8], "time": 0.2}],
    )

    assert result["success"] is True
    assert result["action"] == "/xarm6/xarm_gripper/gripper_action"
    assert result["fallback_error"] == "xarm position service unavailable"


def test_paired_dual_preview_in_gazebo_does_not_use_hardware_actions(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    path = tmp_path / "paired.json"
    path.write_text(json.dumps(_paired_dual_recording()), encoding="utf-8")
    publish_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        digital_twin_sync,
        "_read_snapshot",
        lambda *_args, **_kwargs: pytest.fail("gazebo preview should not read hardware snapshots"),
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_follow_joint_trajectory_action",
        lambda *_args, **_kwargs: pytest.fail("gazebo preview should not preflight hardware actions"),
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_execute_trajectory_action",
        lambda *_args, **_kwargs: pytest.fail("gazebo preview should not preflight /execute_trajectory"),
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_move_group_action",
        lambda *_args, **_kwargs: pytest.fail("gazebo preview should not preflight /move_action"),
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_follow_joint_trajectory_action",
        lambda *_args, **_kwargs: pytest.fail("gazebo preview should not command hardware actions"),
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_execute_trajectory_action",
        lambda *_args, **_kwargs: pytest.fail("gazebo preview should not command /execute_trajectory"),
    )

    def fake_publish(
        domain_id: int,
        topics: list[str],
        joint_names: list[str],
        points: list[dict[str, object]],
        **_kwargs,
    ) -> dict[str, object]:
        publish_calls.append(
            {
                "domain_id": domain_id,
                "topics": list(topics),
                "joint_names": list(joint_names),
                "points": [dict(point) for point in points],
            }
        )
        return {"success": True, "message": "trajectory published."}

    monkeypatch.setattr(digital_twin_sync, "_publish_trajectory", fake_publish)

    code = digital_twin_sync.run_replay(_paired_replay_args(path, replay_target="gazebo"))
    output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert code == 0
    assert output["success"] is True
    assert len(publish_calls) == 4
    assert {call["domain_id"] for call in publish_calls} == {41}


def test_replay_in_twin_initializes_gazebo_from_hardware_before_replay(
    monkeypatch,
    tmp_path,
) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    bridge._digital_twin_sim_modes["dual robots"] = "teach"
    recording_path = tmp_path / "paired.json"
    recording_path.write_text(json.dumps(_paired_dual_recording()), encoding="utf-8")
    events: list[str] = []
    sleeps: list[float] = []

    monkeypatch.setattr(bridge, "_digital_twin_hardware_status", lambda _cfg: {"overall": "running"})
    monkeypatch.setattr(
        bridge,
        "_wait_for_digital_twin_dual_robots_hardware_ready",
        lambda *_args, **_kwargs: pytest.fail("Replay in Twin must not use startup hardware readiness"),
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_digital_twin_dual_robots_replay_ready",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        bridge,
        "ros2_proc_status",
        lambda name: "running" if name == "digital_twin_dual_robots_gazebo" else "stopped",
    )
    monkeypatch.setattr(
        bridge,
        "_initialize_digital_twin_gazebo_from_hardware",
        lambda *_args, **_kwargs: events.append("initialize") or None,
    )
    monkeypatch.setattr(
        bridge,
        "_run_digital_twin_sync",
        lambda *_args, **_kwargs: events.append("replay") or {"success": True, "message": "ok"},
    )
    monkeypatch.setattr(
        bridge,
        "_stop_digital_twin_mirror_workers_for_target",
        lambda *_args, **_kwargs: events.append("stop_mirror"),
    )
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_sync_when_ready",
        lambda *_args, **kwargs: events.append(f"sync:{kwargs.get('direction')}") or None,
    )
    monkeypatch.setattr(
        "cais_spade_llm.ui.bridge.time.sleep",
        lambda seconds: sleeps.append(float(seconds)),
    )

    result = bridge._replay_recording_file("dual robots", cfg, recording_path, "twin")

    assert result["success"] is True
    assert result["sync_resumed"] is True
    assert "hardware -> Gazebo sync resumed" in result["message"]
    assert sleeps == [pytest.approx(bridge._DIGITAL_TWIN_MIRROR_STABILIZATION_SEC)]
    assert events == ["initialize", "stop_mirror", "replay", "sync:hardware -> gazebo"]


def test_replay_in_twin_reports_initialization_failure(monkeypatch, tmp_path) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    bridge._digital_twin_sim_modes["dual robots"] = "teach"
    recording_path = tmp_path / "paired.json"
    recording_path.write_text(json.dumps(_paired_dual_recording()), encoding="utf-8")
    statuses: list[dict[str, object]] = []

    monkeypatch.setattr(bridge, "_digital_twin_hardware_status", lambda _cfg: {"overall": "running"})
    monkeypatch.setattr(
        bridge,
        "_wait_for_digital_twin_dual_robots_hardware_ready",
        lambda *_args, **_kwargs: pytest.fail("Replay in Twin must not use startup hardware readiness"),
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_digital_twin_dual_robots_replay_ready",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        bridge,
        "ros2_proc_status",
        lambda name: "running" if name == "digital_twin_dual_robots_gazebo" else "stopped",
    )
    monkeypatch.setattr(
        bridge,
        "_initialize_digital_twin_gazebo_from_hardware",
        lambda *_args, **_kwargs: "xarm6 gazebo initial hardware pose failed",
    )
    monkeypatch.setattr(bridge, "_write_digital_twin_status", lambda _target, payload: statuses.append(dict(payload)))

    result = bridge._replay_recording_file("dual robots", cfg, recording_path, "twin")

    assert result["success"] is False
    assert "could not initialize gazebo from hardware before replay" in result["message"]
    assert [status["message"] for status in statuses] == [
        "checking replay readiness.",
        "Replay in Twin: initializing gazebo from hardware.",
    ]


def test_dual_replay_resumes_mirror_after_failed_replay(monkeypatch, tmp_path) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    recording_path = tmp_path / "paired.json"
    recording_path.write_text(json.dumps(_paired_dual_recording()), encoding="utf-8")
    events: list[str] = []

    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")
    monkeypatch.setattr(
        bridge,
        "_wait_for_digital_twin_dual_robots_replay_ready",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        bridge,
        "_initialize_digital_twin_gazebo_from_hardware",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        bridge,
        "_stop_digital_twin_mirror_workers_for_target",
        lambda *_args, **_kwargs: events.append("stop_mirror"),
    )
    monkeypatch.setattr(
        bridge,
        "_run_digital_twin_sync",
        lambda *_args, **_kwargs: events.append("replay")
        or {"success": False, "message": "trajectory failed"},
    )
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_sync_when_ready",
        lambda *_args, **kwargs: events.append(f"resume:{kwargs.get('direction')}") or None,
    )

    result = bridge._replay_recording_file("dual robots", cfg, recording_path, "twin")

    assert result["success"] is False
    assert result["sync_resumed"] is True
    assert "trajectory failed" in result["message"]
    assert "hardware -> Gazebo sync resumed after failed replay" in result["message"]
    assert events == ["stop_mirror", "replay", "resume:hardware -> gazebo"]


def test_replay_repeat_runs_setup_once_and_passes_replay_speed(monkeypatch, tmp_path) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    recording_path = tmp_path / "paired.json"
    recording_path.write_text(json.dumps(_paired_dual_recording()), encoding="utf-8")
    events: list[str] = []
    sync_calls: list[list[str]] = []

    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")
    monkeypatch.setattr(
        bridge,
        "_wait_for_digital_twin_dual_robots_replay_ready",
        lambda *_args, **_kwargs: events.append("ready") or None,
    )
    monkeypatch.setattr(
        bridge,
        "_digital_twin_sync_status_snapshot",
        lambda *_args, **_kwargs: {
            "process_status": "stopped",
            "status_data": {},
            "status_age_ms": None,
        },
    )
    monkeypatch.setattr(
        bridge,
        "_initialize_digital_twin_gazebo_from_hardware",
        lambda *_args, **_kwargs: events.append("initialize") or None,
    )
    monkeypatch.setattr(
        bridge,
        "_stop_digital_twin_mirror_workers_for_target",
        lambda *_args, **_kwargs: events.append("stop_mirror"),
    )

    def fake_sync(args: list[str], **_kwargs) -> dict[str, object]:
        events.append("replay")
        sync_calls.append(list(args))
        return {
            "success": True,
            "message": "ok",
            "waypoints": 3,
            "approach_time": 1.0,
        }

    monkeypatch.setattr(bridge, "_run_digital_twin_sync", fake_sync)
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_sync_when_ready",
        lambda *_args, **kwargs: events.append(f"resume:{kwargs.get('direction')}") or None,
    )

    result = bridge._replay_recording_file(
        "dual robots",
        cfg,
        recording_path,
        "twin",
        source="dual_function_xarm6_test_ur5e_test",
        repeat_count=3,
    )

    assert result["success"] is True
    assert result["repeat_count"] == 3
    assert result["repeat_iteration"] == 3
    assert "Replay Dual Function repeated 3 times" in result["message"]
    assert events == [
        "ready",
        "initialize",
        "ready",
        "stop_mirror",
        "replay",
        "ready",
        "replay",
        "ready",
        "replay",
        "resume:hardware -> gazebo",
    ]
    assert len(sync_calls) == 3
    for args in sync_calls:
        assert (
            args[args.index("--waypoint-duration-sec") + 1]
            == f"{bridge._DIGITAL_TWIN_REPLAY_WAYPOINT_DURATION_SEC:.6f}"
        )
        assert (
            args[args.index("--max-joint-vel-deg-s") + 1]
            == f"{bridge._DIGITAL_TWIN_REPLAY_MAX_JOINT_VEL_DEG_S:.6f}"
        )


def test_dual_replay_stops_only_mirror_workers_for_target(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    stopped: list[tuple[str, str]] = []
    pkill_calls: list[list[str]] = []

    monkeypatch.setattr(
        bridge,
        "ros2_stop",
        lambda process, reason="": stopped.append((process, reason)) or None,
    )
    monkeypatch.setattr(
        "cais_spade_llm.ui.bridge.subprocess.run",
        lambda args, **_kwargs: pkill_calls.append(list(args)) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr("cais_spade_llm.ui.bridge.time.sleep", lambda *_args, **_kwargs: None)

    bridge._stop_digital_twin_mirror_workers_for_target("dual robots", cfg)

    assert stopped == [
        ("digital_twin_dual_robots_sync_xarm6", "digital_twin_replay"),
        ("digital_twin_dual_robots_sync_ur5e", "digital_twin_replay"),
    ]
    assert len(pkill_calls) == 4
    patterns = [call[-1] for call in pkill_calls]
    assert all("digital_twin_sync\\.py.*--mode mirror.*--status-file" in pattern for pattern in patterns)
    assert any("cais_digital_twin_dual_robots_xarm6\\.json" in pattern for pattern in patterns)
    assert any("cais_digital_twin_dual_robots_ur5e\\.json" in pattern for pattern in patterns)


def test_preview_in_gazebo_skips_hardware_initialization_in_monitor_mode(
    monkeypatch,
    tmp_path,
) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    bridge._digital_twin_sim_modes["dual robots"] = "teach"
    recording_path = tmp_path / "paired.json"
    recording_path.write_text(json.dumps(_paired_dual_recording()), encoding="utf-8")
    events: list[str] = []

    monkeypatch.setattr(bridge, "_digital_twin_hardware_status", lambda _cfg: {"overall": "running"})
    monkeypatch.setattr(
        bridge,
        "ros2_proc_status",
        lambda name: "running" if name == "digital_twin_dual_robots_gazebo" else "stopped",
    )
    monkeypatch.setattr(
        bridge,
        "_initialize_digital_twin_gazebo_from_hardware",
        lambda *_args, **_kwargs: pytest.fail("Preview in Gazebo must stay Gazebo-only in monitor mode"),
    )
    def fake_sync(args: list[str], **_kwargs) -> dict[str, object]:
        events.append("replay")
        assert str(args[args.index("--replay-target") + 1]) == "gazebo"
        return {"success": True, "message": "ok"}

    monkeypatch.setattr(bridge, "_run_digital_twin_sync", fake_sync)
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_sync_when_ready",
        lambda *_args, **_kwargs: pytest.fail("Preview in Gazebo must not resume hardware sync"),
    )

    result = bridge._replay_recording_file("dual robots", cfg, recording_path, "gazebo")

    assert result["success"] is True
    assert events == ["replay"]


def test_preview_in_gazebo_reports_gazebo_only_replay_failure(monkeypatch, tmp_path) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    bridge._digital_twin_sim_modes["dual robots"] = "teach"
    recording_path = tmp_path / "paired.json"
    recording_path.write_text(json.dumps(_paired_dual_recording()), encoding="utf-8")
    statuses: list[dict[str, object]] = []

    monkeypatch.setattr(bridge, "_digital_twin_hardware_status", lambda _cfg: {"overall": "running"})
    monkeypatch.setattr(
        bridge,
        "ros2_proc_status",
        lambda name: "running" if name == "digital_twin_dual_robots_gazebo" else "stopped",
    )
    monkeypatch.setattr(
        bridge,
        "_initialize_digital_twin_gazebo_from_hardware",
        lambda *_args, **_kwargs: pytest.fail("Preview in Gazebo must stay Gazebo-only in monitor mode"),
    )
    monkeypatch.setattr(
        bridge,
        "_run_digital_twin_sync",
        lambda *_args, **_kwargs: {"success": False, "message": "gazebo replay failed"},
    )
    monkeypatch.setattr(bridge, "_write_digital_twin_status", lambda _target, payload: statuses.append(dict(payload)))

    result = bridge._replay_recording_file("dual robots", cfg, recording_path, "gazebo")

    assert result["success"] is False
    assert "gazebo replay failed" in result["message"]
    assert statuses[0]["message"] == "Preview in Gazebo: publishing to Gazebo only."
    assert statuses[0]["direction"] == "hardware -> gazebo"


def test_dual_robots_hardware_rviz_uses_moveit_robot_display_only() -> None:
    rviz_path = (
        Path(__file__).resolve().parents[1]
        / "ros2"
        / "cais_lab_gazebo"
        / "rviz"
        / "dual_robots_hardware_moveit.rviz"
    )
    body = rviz_path.read_text(encoding="utf-8")

    assert "rviz_default_plugins/RobotModel" not in body
    assert "moveit_rviz_plugin/MotionPlanning" in body
    assert "Robot Alpha: 1" in body
    assert "Velocity_Scaling_Factor: 0.20" in body
    assert "Acceleration_Scaling_Factor: 0.20" in body


def test_dual_robots_hardware_moveit_launch_does_not_start_combined_robot_state_publisher() -> None:
    launch_path = (
        Path(__file__).resolve().parents[1]
        / "ros2"
        / "cais_lab_gazebo"
        / "launch"
        / "dual_robots_hardware_moveit.launch.py"
    )
    body = launch_path.read_text(encoding="utf-8")

    assert "moveit_ros_move_group" in body
    assert "name=\"move_group\"" in body
    assert "robot_state_publisher" not in body


def test_xarm6_hardware_driver_does_not_spawn_real_gripper_trajectory_controller() -> None:
    launch_path = (
        Path(__file__).resolve().parents[1]
        / "ros2"
        / "cais_lab_gazebo"
        / "launch"
        / "xarm6_hardware_driver.launch.py"
    )
    body = launch_path.read_text(encoding="utf-8")
    required_section = body.split("required_controller_spawner = Node(", 1)[1].split(
        "joint_state_relay = ExecuteProcess",
        1,
    )[0]

    assert '"joint_state_broadcaster"' in required_section
    assert '"xarm6_traj_controller"' in required_section
    assert '"xarm_gripper_traj_controller"' not in required_section
    assert '"--activate-as-group"' in required_section
    assert '"xarm_gripper_traj_controller"' not in body
    assert "optional_gripper_spawner" not in body
    assert "OnProcessExit" not in body
    assert '"/xarm6/xarm/joint_states"' in body
    assert 'msg.name = ["drive_joint"]' in body
    assert "self.create_timer(0.05, self._publish_drive_joint)" in body
    assert "GripperCommand_FeedbackMessage" in body
    assert '"/xarm6/xarm_gripper/gripper_action/_action/feedback"' in body


def test_keyboard_teleop_prefers_hardware_gripper_actions() -> None:
    teleop_path = (
        Path(__file__).resolve().parents[1]
        / "ros2"
        / "cais_lab_gazebo"
        / "scripts"
        / "keyboard_teleop.py"
    )
    body = teleop_path.read_text(encoding="utf-8")
    move_gripper_body = body.split("def move_gripper(", 1)[1]

    assert "f'/xarm6/xarm/{suffix}'" in body
    assert "/xarm6/xarm_gripper/gripper_action" in body
    assert "/xarm_gripper/gripper_action" in body
    assert move_gripper_body.index("_move_xarm_gripper_service") < move_gripper_body.index(
        "_move_xarm_gripper_action"
    )
    assert "FollowJointTrajectory" in body
    assert "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory" in body
    assert move_gripper_body.index("_move_ur5e_rg2_gripper_action") < move_gripper_body.index(
        "_publish_joint_trajectory"
    )
    assert "/xarm6/xarm_gripper_traj_controller/joint_trajectory" in body


def test_cleanup_kills_stale_xarm6_joint_state_relay() -> None:
    root = Path(__file__).resolve().parents[1]
    bridge_body = (root / "cais_spade_llm" / "ui" / "bridge.py").read_text(encoding="utf-8")
    ui_main_body = (root / "cais_spade_llm" / "ui_main.py").read_text(encoding="utf-8")

    assert "xarm6_hardware_driver.launch.py" in bridge_body
    assert "XArm6JointStateRelay" in bridge_body
    assert "xarm6_hardware_driver.launch.py" in ui_main_body
    assert "XArm6JointStateRelay" in ui_main_body


def test_ur5e_hardware_stack_is_rtde_gripper_moveit_only() -> None:
    bridge = SystemBridge()

    assert bridge._hardware_stack_for_robot("ur5e") == (
        "hardware_ur5e_rtde_trajectory_server",
        "hardware_ur5e_rg2_gripper",
        "hardware_ur5e_moveit",
    )
    assert set(bridge._HARDWARE_STACKS["ur5e"]) == {
        "hardware_ur5e_rtde_trajectory_server",
        "hardware_ur5e_rg2_gripper",
        "hardware_ur5e_moveit",
    }
    assert "hardware_ur5e_rtde_trajectory_server" in bridge.ROS2_LAUNCH_CMDS
    assert "hardware_ur5e_rg2_gripper" in bridge.ROS2_LAUNCH_CMDS
    assert "hardware_ur5e_moveit" in bridge.ROS2_LAUNCH_CMDS
    assert ("hardware_ur5e_" + "driver") not in bridge.ROS2_LAUNCH_CMDS
    assert ("hardware_ur5e_" + "driver_watchdog") not in bridge.ROS2_LAUNCH_CMDS
    assert ("hardware_ur5e_" + "trajectory_guard") not in bridge.ROS2_LAUNCH_CMDS


def test_dual_robots_hardware_stack_contains_no_ur_driver_watchdog_or_guard() -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    ur5e_processes = dict(cfg["hardware_processes"]["ur5e"])
    process_names = set(bridge._digital_twin_process_names(cfg))

    assert ur5e_processes == {
        "rtde": "digital_twin_dual_robots_hardware_ur5e_rtde_trajectory_server",
        "gripper": "digital_twin_dual_robots_hardware_ur5e_rg2_gripper",
        "moveit": "digital_twin_dual_robots_hardware_moveit",
    }
    assert "digital_twin_dual_robots_hardware_ur5e_rtde_trajectory_server" in process_names
    assert "digital_twin_dual_robots_hardware_ur5e_rg2_gripper" in process_names
    assert "digital_twin_dual_robots_hardware_moveit" in process_names
    assert not any(("hardware_ur5e_" + "driver") in name for name in process_names)
    assert not any("watchdog" in name for name in process_names)
    assert not any("trajectory_guard" in name for name in process_names)


def test_ur5e_moveit_launch_files_always_use_rtde_trajectory_controller() -> None:
    root = Path(__file__).resolve().parents[1]
    launch_paths = [
        root / "ros2" / "cais_lab_gazebo" / "launch" / "ur5e_rg2_hardware_moveit.launch.py",
        root / "ros2" / "cais_lab_gazebo" / "launch" / "dual_robots_hardware_moveit.launch.py",
    ]

    for launch_path in launch_paths:
        body = launch_path.read_text(encoding="utf-8")
        assert "UR5E_RTDE_TRAJECTORY_CONTROLLER" in body
        assert '"cais_ur5e_rtde_trajectory_controller"' in body
        assert "RTDE_ALLOWED_EXECUTION_DURATION_SCALING = 8.0" in body
        assert "RTDE_ALLOWED_GOAL_DURATION_MARGIN = 20.0" in body
        assert ("ur5e_arm_" + "backend") not in body
        assert ("use_ur5e_" + "trajectory_guard") not in body
        assert "UR5E_GUARDED_TRAJECTORY_CONTROLLER" not in body
        assert "UR5E_REAL_TRAJECTORY_CONTROLLER" not in body
        assert ("scaled_joint_" + "trajectory_controller") not in body


def test_digital_twin_sync_ur5e_defaults_to_rtde_action_and_balanced_mirror() -> None:
    assert (
        digital_twin_sync.ROBOTS["ur5e"]["hardware_trajectory_action"]
        == "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory"
    )
    assert digital_twin_sync.UR5E_MIRROR_POINT_TIME_SEC == pytest.approx(0.12)
    assert digital_twin_sync.UR5E_MIRROR_MIN_PUBLISH_PERIOD_SEC == pytest.approx(0.05)
    assert digital_twin_sync.UR5E_MIRROR_MIN_JOINT_DELTA_RAD == pytest.approx(0.0010)
    assert digital_twin_sync.MIRROR_POINT_TIME_SEC == pytest.approx(0.1)
    assert digital_twin_sync.MIRROR_MIN_PUBLISH_PERIOD_SEC == pytest.approx(0.0)
    assert digital_twin_sync.MIRROR_MIN_JOINT_DELTA_RAD == pytest.approx(0.0)
    assert digital_twin_sync._mirror_point_time_sec("ur5e") == pytest.approx(0.12)
    assert digital_twin_sync._mirror_min_publish_period_sec("ur5e") == pytest.approx(0.05)
    assert digital_twin_sync._mirror_min_joint_delta_rad("ur5e") == pytest.approx(0.0010)


def test_no_runtime_ur5e_ur_driver_or_guard_paths_remain() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime_paths = [
        root / "cais_spade_llm" / "ui" / "bridge.py",
        root / "cais_spade_llm" / "ui" / "pages" / "control.py",
        root / "cais_spade_llm" / "ui_main.py",
        root / "ros2" / "cais_lab_gazebo" / "scripts" / "digital_twin_sync.py",
        root / "ros2" / "cais_lab_gazebo" / "scripts" / "dual_drag_markers.py",
        root / "ros2" / "cais_lab_gazebo" / "scripts" / "ur5e_rtde_trajectory_server.py",
        root / "ros2" / "cais_lab_gazebo" / "launch" / "ur5e_rg2_hardware_moveit.launch.py",
        root / "ros2" / "cais_lab_gazebo" / "launch" / "dual_robots_hardware_moveit.launch.py",
    ]
    forbidden_terms = [
        "ros" + ".urp",
        "ur_robot_" + "driver",
        "hardware_ur5e_" + "driver",
        "hardware_ur5e_" + "driver_watchdog",
        "hardware_ur5e_" + "trajectory_guard",
        "cais_ur5e_" + "guarded_" + "scaled_joint_" + "trajectory_controller",
        "use_ur5e_" + "trajectory_guard",
        "CAIS_UR5E_ARM_" + "BACKEND",
        "repair_ur5e_" + "trajectory_controller",
    ]

    for runtime_path in runtime_paths:
        body = runtime_path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in body, f"{term} remains in {runtime_path}"


def test_bridge_starts_dual_drag_markers_with_status_file(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    calls: list[tuple[str, str, int | None]] = []

    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "stopped")
    monkeypatch.setattr(
        bridge,
        "_start_tracked_ros2_command",
        lambda process_name, command, **kwargs: calls.append(
            (process_name, command, kwargs.get("ros_domain_id"))
        )
        or None,
    )

    err = bridge._start_digital_twin_dual_drag_markers(
        cfg,
        ros_domain_id=42,
        mode="monitor",
    )

    assert err is None
    assert len(calls) == 1
    process_name, command, ros_domain_id = calls[0]
    assert process_name == "digital_twin_dual_robots_paired_markers"
    assert ros_domain_id == 42
    assert "--status-file" in command
    assert str(bridge._digital_twin_dual_drag_markers_status_path("dual robots")) in command


def test_digital_twin_statuses_include_dual_drag_markers_last_error(monkeypatch) -> None:
    bridge = SystemBridge()
    status_path = bridge._digital_twin_dual_drag_markers_status_path("dual robots")
    old_body = status_path.read_text(encoding="utf-8") if status_path.exists() else None
    status_path.write_text(
        json.dumps(
            {
                "updated_at": 100.0,
                "node": "dual_drag_markers",
                "mode": "monitor",
                "execution_policy": "paired",
                "state": "failed",
                "action": "Plan+Execute dual_robots",
                "stage": "execute",
                "message": "Plan+Execute dual_robots failed during execute: /execute_trajectory error_code=-4",
                "last_error": "Plan+Execute dual_robots failed during execute: /execute_trajectory error_code=-4",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")
    monkeypatch.setattr(
        bridge,
        "_digital_twin_hardware_status",
        lambda _cfg: {"overall": "running", "moveit": "running", "status": {}},
    )

    try:
        row = bridge.digital_twin_statuses()["dual robots"]
    finally:
        if old_body is None:
            status_path.unlink(missing_ok=True)
        else:
            status_path.write_text(old_body, encoding="utf-8")

    assert row["dual_drag_markers"]["state"] == "failed"
    assert row["dual_drag_markers"]["action"] == "Plan+Execute dual_robots"
    assert row["dual_drag_markers"]["stage"] == "execute"
    assert "/execute_trajectory error_code=-4" in row["dual_drag_markers"]["last_error"]


def test_dual_rviz_configs_show_dual_drag_markers() -> None:
    root = Path(__file__).resolve().parents[1]
    hardware_rviz = (
        root / "ros2" / "cais_lab_gazebo" / "rviz" / "dual_robots_hardware_moveit.rviz"
    ).read_text(encoding="utf-8")
    teach_rviz = (
        root / "ros2" / "cais_lab_gazebo" / "rviz" / "dual_moveit.rviz"
    ).read_text(encoding="utf-8")

    for body in (hardware_rviz, teach_rviz):
        assert "rviz_default_plugins/InteractiveMarkers" in body
        assert "Name: Dual Drag Markers" in body
        assert "Update Topic: /dual_drag_markers/update" in body


def test_digital_twin_sim_mode_uses_monitor_with_legacy_mirror_compatibility() -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]

    assert bridge._digital_twin_allowed_sim_modes(cfg) == ("monitor",)
    assert bridge._digital_twin_sim_mode("dual robots") == "monitor"
    assert bridge.digital_twin_set_sim_mode("dual robots", "author") == "unknown digital twin sim mode: author"
    assert bridge._digital_twin_sim_mode("dual robots") == "monitor"
    assert bridge._digital_twin_gazebo_launch("dual robots", cfg) == "gazebo_dual_passive"
    assert bridge.digital_twin_set_sim_mode("dual robots", "mirror") is None
    assert bridge._digital_twin_sim_mode("dual robots") == "monitor"
    assert bridge._digital_twin_gazebo_launch("dual robots", cfg) == "gazebo_dual_passive"


def test_control_page_only_exposes_monitor_and_teach_mode_labels() -> None:
    body = (
        Path(__file__).resolve().parents[1]
        / "cais_spade_llm"
        / "ui"
        / "pages"
        / "control.py"
    ).read_text(encoding="utf-8")
    label_line = body.split("_DT_MODE_LABELS =", 1)[1].split("\n", 1)[0]

    assert '"monitor": "Monitor"' in label_line
    assert '"teach": "Teach"' in label_line
    assert "mirror" not in label_line
    assert "author" not in label_line


def test_control_page_teach_copy_says_sim_rviz_controls_gazebo_only() -> None:
    root = Path(__file__).resolve().parents[1]
    body = (
        root / "cais_spade_llm" / "ui" / "pages" / "control.py"
    ).read_text(encoding="utf-8")
    bridge_body = (
        root / "cais_spade_llm" / "ui" / "bridge.py"
    ).read_text(encoding="utf-8")

    assert "Function Record / Replay" in body
    assert "Replay Function" in body
    assert "Replay Function in Twin?" in body
    assert "Replay Dual Function" in body
    assert "saved steps:" in body
    assert "def _suggest_next_step_name" in body
    assert '"taught_functions"' in bridge_body
    assert TAUGHT_FUNCTIONS_ROOT.name == "taught_functions"
    assert "functions" not in TAUGHT_FUNCTIONS_ROOT.parts


def test_ur5e_only_gazebo_launch_uses_single_table_center_and_home_pose() -> None:
    body = (
        Path(__file__).resolve().parents[1]
        / "ros2"
        / "cais_lab_gazebo"
        / "launch"
        / "ur5e_rg2_gazebo.launch.py"
    ).read_text(encoding="utf-8")

    assert "UR5E_BASE_XYZ = '0.0 0.0 1.021'" in body
    assert "UR5E_BASE_RPY = '0 0 3.142'" in body
    assert "'shoulder_pan_joint': 1.637161" in body
    assert "'shoulder_lift_joint': -2.150816" in body
    assert "'wrist_3_joint': 1.637331" in body
    assert "_set_ros2_control_initial_positions(" in body
    assert "{f'{onrobot_prefix}finger_width': 0.11}" in body
    assert "_inject_mimic_plugins(onrobot_root, max_effort='3.0', sensitiveness='0.003')" in body
    assert "_tune_rg2_contact_properties(onrobot_root, onrobot_prefix)" in body
    assert "_strip_grasp_fix_plugins(onrobot_root)" in body


def test_ur5e_only_digital_twin_start_keeps_stack_when_hardware_readiness_fails(monkeypatch) -> None:
    bridge = SystemBridge()
    statuses: list[dict] = []
    err_text = "ur5e MoveIt is not ready: command timed out"

    monkeypatch.setattr(bridge, "_digital_twin_blocked_reason", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(bridge, "_stop_teleop_server", lambda: None)
    monkeypatch.setattr(bridge, "_write_digital_twin_direction", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bridge, "_write_digital_twin_status", lambda _target, payload: statuses.append(dict(payload)))
    monkeypatch.setattr(bridge, "_shutdown_gazebo_prewarm_controllers", lambda: None)
    monkeypatch.setattr(bridge, "_force_kill_digital_twin_helpers", lambda: None)
    monkeypatch.setattr(bridge, "_kill_stale_gazebo_helpers", lambda: None)
    monkeypatch.setattr(bridge, "_force_kill_gazebo_core", lambda **_kwargs: None)
    monkeypatch.setattr(bridge, "_start_digital_twin_launch", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bridge, "_start_digital_twin_hardware_stack", lambda *_args, **_kwargs: err_text)
    monkeypatch.setattr(
        bridge,
        "_initialize_digital_twin_gazebo_from_hardware",
        lambda *_args, **_kwargs: pytest.fail("initialization should not run after hardware readiness failure"),
    )
    monkeypatch.setattr(
        bridge,
        "digital_twin_stop",
        lambda *_args, **_kwargs: pytest.fail("ur5e only digital twin should not close RViz after hardware readiness failure"),
    )

    err = bridge.digital_twin_start("ur5e only")

    assert err == err_text
    assert statuses[-1]["state"] == "partial"
    assert statuses[-1]["message"] == err_text


def test_ur5e_only_digital_twin_start_keeps_stack_when_gazebo_launch_fails(monkeypatch) -> None:
    bridge = SystemBridge()
    statuses: list[dict] = []
    err_text = "gazebo launch failed"

    monkeypatch.setattr(bridge, "_digital_twin_blocked_reason", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(bridge, "_stop_teleop_server", lambda: None)
    monkeypatch.setattr(bridge, "_write_digital_twin_direction", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bridge, "_write_digital_twin_status", lambda _target, payload: statuses.append(dict(payload)))
    monkeypatch.setattr(bridge, "_shutdown_gazebo_prewarm_controllers", lambda: None)
    monkeypatch.setattr(bridge, "_force_kill_digital_twin_helpers", lambda: None)
    monkeypatch.setattr(bridge, "_kill_stale_gazebo_helpers", lambda: None)
    monkeypatch.setattr(bridge, "_force_kill_gazebo_core", lambda **_kwargs: None)
    monkeypatch.setattr(bridge, "_start_digital_twin_launch", lambda *_args, **_kwargs: err_text)
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_hardware_stack",
        lambda *_args, **_kwargs: pytest.fail("hardware stack should not run after gazebo launch failure"),
    )
    monkeypatch.setattr(
        bridge,
        "digital_twin_stop",
        lambda *_args, **_kwargs: pytest.fail("ur5e only digital twin should not close RViz/Gazebo after gazebo launch failure"),
    )

    err = bridge.digital_twin_start("ur5e only")

    assert err == err_text
    assert statuses[-1]["state"] == "partial"
    assert statuses[-1]["message"] == err_text


def test_ur5e_only_digital_twin_start_keeps_stack_when_sync_start_fails(monkeypatch) -> None:
    bridge = SystemBridge()
    sync_statuses: list[dict] = []
    err_text = "sync command failed"

    monkeypatch.setattr(bridge, "_digital_twin_blocked_reason", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(bridge, "_stop_teleop_server", lambda: None)
    monkeypatch.setattr(bridge, "_write_digital_twin_direction", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bridge, "_write_digital_twin_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bridge, "_write_digital_twin_sync_status", lambda _target, _cfg, payload: sync_statuses.append(dict(payload)))
    monkeypatch.setattr(bridge, "_shutdown_gazebo_prewarm_controllers", lambda: None)
    monkeypatch.setattr(bridge, "_force_kill_digital_twin_helpers", lambda: None)
    monkeypatch.setattr(bridge, "_kill_stale_gazebo_helpers", lambda: None)
    monkeypatch.setattr(bridge, "_force_kill_gazebo_core", lambda **_kwargs: None)
    monkeypatch.setattr(bridge, "_start_digital_twin_launch", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bridge, "_start_digital_twin_hardware_stack", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bridge, "_initialize_digital_twin_gazebo_from_hardware", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bridge, "_start_digital_twin_sync_when_ready", lambda *_args, **_kwargs: err_text)
    monkeypatch.setattr(
        bridge,
        "digital_twin_stop",
        lambda *_args, **_kwargs: pytest.fail("ur5e only digital twin should not close RViz after sync start failure"),
    )

    err = bridge.digital_twin_start("ur5e only")

    assert err == err_text
    assert sync_statuses[-1]["state"] == "waiting"
    assert sync_statuses[-1]["last_error"] == err_text


def test_save_function_appends_unsaved_steps_to_existing_saved_function_and_clears_buffer(
    monkeypatch,
    tmp_path,
) -> None:
    bridge = SystemBridge()
    function_root = tmp_path / "taught_functions"
    saved_path = function_root / "ur5e" / "ur5e_test1" / "default__hardware.json"
    saved_path.parent.mkdir(parents=True)
    saved_path.write_text(
        json.dumps(
            {
                "robot": "ur5e",
                "function_name": "ur5e_test1",
                "name": "default",
                "capture_source": "hardware",
                "replay_targets": ["hardware", "digital_twin"],
                "steps": [
                    {
                        "step_name": "step_1",
                        "primitive": "move_cartesian",
                        "waypoint": {
                            "joint_names": ["j1"],
                            "joint_positions": [0.1],
                            "pose": None,
                        },
                    },
                    {
                        "step_name": "step_2",
                        "primitive": "move_cartesian",
                        "waypoint": {
                            "joint_names": ["j1"],
                            "joint_positions": [0.2],
                            "pose": None,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("cais_spade_llm.ui.bridge._ROBOT_TAUGHT_FUNCTIONS_DIR", function_root)
    monkeypatch.setattr(
        bridge,
        "_snapshot_robot_waypoint",
        lambda *_args, **_kwargs: {
            "joint_names": ["j1"],
            "positions": [0.3],
            "gripper_joint": "ur5e_rg2_finger_width",
            "gripper": 0.08,
        },
    )

    capture = bridge.digital_twin_capture_function_step(
        "ur5e only",
        "ur5e",
        "ur5e_test1",
        "default",
        "step_3",
        "move_cartesian",
    )
    assert capture["success"] is True
    assert bridge.digital_twin_function_step_count("ur5e only", "ur5e", "ur5e_test1", "default") == 1

    result = bridge.digital_twin_save_function("ur5e only", "ur5e", "ur5e_test1", "default")

    assert result["success"] is True
    assert result["saved_steps"] == 3
    assert result["unsaved_steps"] == 0
    assert bridge.digital_twin_function_step_count("ur5e only", "ur5e", "ur5e_test1", "default") == 0
    payload = json.loads(saved_path.read_text(encoding="utf-8"))
    assert [step["step_name"] for step in payload["steps"]] == ["step_1", "step_2", "step_3"]
    assert payload["steps"][2]["waypoint"]["joint_positions"] == [0.3]


def test_ur5e_only_monitor_status_schedules_sync_restart_when_mirror_stopped(monkeypatch) -> None:
    bridge = SystemBridge()
    restarts: list[tuple[str, str]] = []

    monkeypatch.setattr(
        bridge,
        "_digital_twin_hardware_status",
        lambda _cfg: {
            "overall": "running",
            "driver": "running",
            "moveit": "running",
            "ur5e": {"overall": "running", "driver": "running", "moveit": "running"},
        },
    )
    monkeypatch.setattr(bridge, "_ur5e_rg2_gripper_status", lambda: {})
    monkeypatch.setattr(bridge, "_digital_twin_blocked_reason", lambda *_args, **_kwargs: "")

    def fake_proc_status(name: str) -> str:
        if name == "digital_twin_ur5e_only_gazebo":
            return "running"
        if name == "digital_twin_ur5e_only_sync":
            return "stopped"
        return "stopped"

    monkeypatch.setattr(bridge, "ros2_proc_status", fake_proc_status)
    monkeypatch.setattr(
        bridge,
        "_schedule_digital_twin_monitor_sync_restart",
        lambda target, _cfg, **kwargs: restarts.append((target, str(kwargs.get("reason") or ""))) or True,
    )

    row = bridge.digital_twin_statuses()["ur5e only"]

    assert row["sync/status"]["process_status"] == "stopped"
    assert restarts == [("ur5e only", "sync process is stopped")]


def test_schedule_digital_twin_monitor_sync_restart_starts_background_sync(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["ur5e only"]
    statuses: list[dict] = []
    starts: list[tuple[str, str]] = []

    monkeypatch.setattr(
        bridge,
        "_write_digital_twin_sync_status",
        lambda _target, _cfg, payload: statuses.append(dict(payload)),
    )
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_sync_when_ready",
        lambda target, _cfg, **kwargs: starts.append((target, str(kwargs.get("direction") or ""))) or None,
    )

    scheduled = bridge._schedule_digital_twin_monitor_sync_restart(
        "ur5e only",
        cfg,
        gazebo_process="digital_twin_ur5e_only_gazebo",
        domains={"gazebo": 41, "hardware": 42},
        reason="sync process is stopped",
    )
    thread = bridge._digital_twin_sync_restart_threads.get("ur5e only")
    if thread is not None:
        thread.join(timeout=2.0)

    assert scheduled is True
    assert statuses[0]["message"] == "restarting hardware -> gazebo mirror: sync process is stopped"
    assert starts == [("ur5e only", "hardware -> gazebo")]


def test_control_page_replay_notify_uses_captured_client_before_dialog_close() -> None:
    body = (
        Path(__file__).resolve().parents[1]
        / "cais_spade_llm"
        / "ui"
        / "pages"
        / "control.py"
    ).read_text(encoding="utf-8")

    assert "from nicegui import context, ui" in body
    assert "def _current_client() -> Client | None:" in body
    assert 'client.outbox.enqueue_message("notify", options, client.id)' in body
    replay_confirmed = body.split("async def _replay_twin_confirmed", 1)[1].split(
        "ui.button(",
        1,
    )[0]
    assert "notify_client = _current_client()" in replay_confirmed
    assert "replay_confirm.close()" in replay_confirmed
    assert "await _replay_function(" in replay_confirmed
    assert '"twin"' in replay_confirmed
    assert "client=notify_client" in replay_confirmed


def test_digital_twin_dual_teach_mode_is_not_exposed_for_now(monkeypatch) -> None:
    bridge = SystemBridge()
    events: list[str] = []

    err = bridge.digital_twin_set_sim_mode("dual robots", "teach")

    assert err == "unknown digital twin sim mode: teach"
    assert bridge._digital_twin_sim_mode("dual robots") == "monitor"
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_sync_process",
        lambda *_args, **_kwargs: events.append("sync"),
    )
    monkeypatch.setattr(
        bridge,
        "_initialize_digital_twin_gazebo_from_hardware",
        lambda *_args, **_kwargs: pytest.fail("monitor mode start must not initialize from teach mode"),
    )
    assert events == []


def test_digital_twin_dual_stale_teach_mode_falls_back_to_monitor(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]

    bridge._digital_twin_sim_modes["dual robots"] = "teach"

    assert bridge._digital_twin_sim_mode("dual robots") == "monitor"
    assert bridge._digital_twin_gazebo_launch("dual robots", cfg) == "gazebo_dual_passive"


def test_digital_twin_dual_capture_waypoint_stores_xarm6_and_ur5e(monkeypatch) -> None:
    bridge = SystemBridge()
    seen_robots: list[str] = []

    monkeypatch.setattr(
        bridge,
        "ros2_proc_status",
        lambda name: "running" if name == "digital_twin_dual_robots_gazebo" else "stopped",
    )

    def fake_sync(args: list[str], **_kwargs) -> dict[str, object]:
        robot = str(args[args.index("--robot") + 1])
        seen_robots.append(robot)
        return {
            "success": True,
            "joint_names": [f"{robot}_joint_{i}" for i in range(6)],
            "positions": [float(i) for i in range(6)],
            "gripper_joint": f"{robot}_gripper",
            "gripper_position": 0.1,
        }

    monkeypatch.setattr(bridge, "_run_digital_twin_sync", fake_sync)

    result = bridge.digital_twin_capture_waypoint("dual robots")
    waypoints = bridge.digital_twin_list_waypoints("dual robots")
    recording = bridge._build_buffer_recording(
        "dual robots",
        bridge._DIGITAL_TWIN_TARGETS["dual robots"],
    )

    assert result["success"] is True
    assert seen_robots == ["xarm6", "ur5e"]
    assert set(waypoints[0]["robots"]) == {"xarm6", "ur5e"}
    assert recording is not None
    assert recording["recording_type"] == "paired_dual_robots"
    assert recording["robot"] == "dual robots"
    assert set(recording["robots"]) == {"xarm6", "ur5e"}
    assert "recovery_metadata" in recording
    assert set(recording["waypoints"][0]["robots"]) == {"xarm6", "ur5e"}


def test_digital_twin_save_recording_loads_persisted_capture_buffer(monkeypatch, tmp_path) -> None:
    bridge = SystemBridge()
    seen_robots: list[str] = []
    buffer_path = tmp_path / "buffer.json"

    monkeypatch.setattr(bridge, "_digital_twin_buffer_path", lambda _target, _cfg: buffer_path)
    monkeypatch.setattr(bridge, "_digital_twin_recordings_dir", lambda: tmp_path)
    monkeypatch.setattr(
        bridge,
        "ros2_proc_status",
        lambda name: "running" if name == "digital_twin_dual_robots_gazebo" else "stopped",
    )

    def fake_sync(args: list[str], **_kwargs) -> dict[str, object]:
        robot = str(args[args.index("--robot") + 1])
        seen_robots.append(robot)
        return {
            "success": True,
            "joint_names": [f"{robot}_joint_{i}" for i in range(6)],
            "positions": [float(i) for i in range(6)],
            "gripper_joint": f"{robot}_gripper",
            "gripper_position": 0.1,
        }

    monkeypatch.setattr(bridge, "_run_digital_twin_sync", fake_sync)

    capture = bridge.digital_twin_capture_waypoint("dual robots")
    assert capture["success"] is True
    assert buffer_path.is_file()

    bridge._digital_twin_waypoints.clear()
    assert bridge.digital_twin_waypoint_count("dual robots") == 1

    save = bridge.digital_twin_save_recording("dual robots", "saved")

    assert save["success"] is True
    assert "saved 1 waypoints" in save["message"]
    assert (tmp_path / "dual_robots__saved.json").is_file()
    assert seen_robots == ["xarm6", "ur5e"]


def test_digital_twin_sync_supports_initial_gazebo_from_hardware_mode() -> None:
    root = Path(__file__).resolve().parents[1]
    body = (
        root / "ros2" / "cais_lab_gazebo" / "scripts" / "digital_twin_sync.py"
    ).read_text(encoding="utf-8")

    assert "def run_initialize_gazebo_from_hardware" in body
    assert '"initialize-gazebo-from-hardware"' in body
    assert "SetModelConfiguration" in body
    assert "/gazebo/set_model_configuration" in body
    assert "ROBOTS[args.robot][\"gazebo_trajectory_topics\"]" in body
    assert "INITIALIZE_GAZEBO_TOLERANCE_RAD" in body
    assert "max_joint_delta_rad" in body
    assert "gazebo pose still differs from hardware" in body


def test_digital_twin_sync_xarm6_hardware_snapshot_uses_candidate_topics() -> None:
    topics = digital_twin_sync.ROBOTS["xarm6"]["hardware_joint_state_topics"]

    assert "/joint_states" in topics
    assert "/xarm/joint_states" in topics
    assert "/xarm6/joint_states" in topics
    assert "/xarm6/xarm/joint_states" in topics
    assert "/xarm6/xarm_gripper/joint_states" in topics
    assert digital_twin_sync._joint_state_topics("xarm6", "hardware") == topics
    assert digital_twin_sync._joint_state_topics("xarm6", "gazebo") == ["/joint_states"]


def test_digital_twin_dual_digital_twin_removes_boards_and_parts() -> None:
    root = Path(__file__).resolve().parents[1]
    launch_body = (
        root / "ros2" / "cais_lab_gazebo" / "launch" / "xarm6_ur5e_gazebo.launch.py"
    ).read_text(encoding="utf-8")
    dual_moveit_body = (
        root / "ros2" / "cais_lab_gazebo" / "launch" / "dual_moveit_gazebo.launch.py"
    ).read_text(encoding="utf-8")

    assert "include_assembly_parts:=false" in SystemBridge.ROS2_LAUNCH_CMDS["gazebo_dual_passive"]
    assert "include_assembly_parts:=false" in SystemBridge.ROS2_LAUNCH_CMDS["gazebo_dual"]
    assert "include_assembly_parts:=false" in SystemBridge.ROS2_LAUNCH_CMDS["gazebo_dual_gazebo_only"]
    assert "include_assembly_parts:=false" in SystemBridge.ROS2_LAUNCH_CMDS["gazebo_dual_moveit_only"]
    assert "run_perception:=false" in SystemBridge.ROS2_LAUNCH_CMDS["gazebo_dual"]
    assert "launch_moveit:=false launch_rviz:=false" in SystemBridge.ROS2_LAUNCH_CMDS["gazebo_dual_gazebo_only"]
    assert "launch_gazebo:=false launch_moveit:=true launch_rviz:=true" in SystemBridge.ROS2_LAUNCH_CMDS["gazebo_dual_moveit_only"]
    assert "ASSEMBLY_PART_MODELS" in launch_body
    assert "_world_without_assembly_parts" in launch_body
    assert "DeclareLaunchArgument(\n            'include_assembly_parts'" in launch_body
    assert "'include_assembly_parts': include_assembly_parts" in dual_moveit_body
    assert "DeclareLaunchArgument(\n            'launch_gazebo'" in dual_moveit_body
    assert "DeclareLaunchArgument(\n            'launch_moveit'" in dual_moveit_body
    assert "'gear_small'" in launch_body
    assert "'rect_pin_large'" in launch_body
    assert "'circ_pin_medium'" in launch_body
    assert "'assembly_board_v1'" in launch_body
    assert "'table_xarm6'" in launch_body
    assert "'table_ur5e'" in launch_body
    assert "'prusa_mk3'" in launch_body
    assert "'prusa_mk4_1'" in launch_body
    assert "'prusa_mk4_2'" in launch_body
    assert "'cam_mk3'" in launch_body
    assert "'cam_assembly'" in launch_body


def test_digital_twin_dual_hardware_ready_does_not_block_on_xarm_controller_service(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    events: list[tuple[str, str]] = []

    monkeypatch.setattr(
        bridge,
        "ros2_proc_status",
        lambda _name: "running",
    )

    def fake_retry(label: str, wait_fn, **_kwargs) -> str | None:
        events.append(("retry", label))
        return wait_fn()

    monkeypatch.setattr(bridge, "_wait_with_ros2_daemon_retry", fake_retry)
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_service",
        lambda service, **_kwargs: events.append(("service", service)) or "timeout",
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_action",
        lambda action, **_kwargs: events.append(("action", action)) or None,
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_topic_publisher",
        lambda topic, **_kwargs: events.append(("topic", topic)) or None,
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_driver_ready",
        lambda robot, **_kwargs: events.append(("driver", robot)) or None,
    )
    err = bridge._wait_for_digital_twin_dual_robots_hardware_ready(
        cfg,
        ros_domain_id=42,
    )

    assert err is None
    assert ("service", "/xarm6/controller_manager/list_controllers") in events
    assert ("action", "/xarm6/xarm6_traj_controller/follow_joint_trajectory") in events
    assert ("action", "/xarm6/xarm_gripper/gripper_action") not in events
    assert ("action", "/execute_trajectory") in events


def test_digital_twin_dual_sync_status_uses_sync_only_states(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    statuses: list[dict[str, object]] = []

    monkeypatch.setattr(
        bridge,
        "_write_digital_twin_sync_status",
        lambda _target, _cfg, payload: statuses.append(dict(payload)),
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_digital_twin_dual_robots_hardware_ready",
        lambda *_args, **_kwargs: pytest.fail(
            "dual robots sync must not wait for operator hardware readiness"
        ),
    )
    monkeypatch.setattr(bridge, "_wait_for_ros_service", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        bridge,
        "_wait_for_digital_twin_gazebo_controller_actions",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_sync_process",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(bridge, "_write_digital_twin_status", lambda *_args, **_kwargs: None)

    err = bridge._start_digital_twin_sync_when_ready(
        "dual robots",
        cfg,
        gazebo_process="digital_twin_dual_robots_gazebo",
        domains={"gazebo": 41, "hardware": 42},
    )

    assert err is None
    assert [status["message"] for status in statuses] == [
        "starting sync workers",
        "sync process running",
    ]
    assert all(status["last_error"] == "" for status in statuses)


def test_digital_twin_dual_sync_starts_even_if_xarm6_action_readiness_would_fail(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    statuses: list[dict[str, object]] = []
    sync_started = False

    monkeypatch.setattr(
        bridge,
        "_write_digital_twin_sync_status",
        lambda _target, _cfg, payload: statuses.append(dict(payload)),
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_digital_twin_dual_robots_hardware_ready",
        lambda *_args, **_kwargs: pytest.fail(
            "dual robots sync must not wait for xarm6 trajectory controller"
        ),
    )
    monkeypatch.setattr(bridge, "_wait_for_ros_service", lambda *_args, **_kwargs: None)

    def fake_start_sync(*_args, **_kwargs) -> None:
        nonlocal sync_started
        sync_started = True
        return None

    monkeypatch.setattr(bridge, "_start_digital_twin_sync_process", fake_start_sync)
    monkeypatch.setattr(bridge, "_write_digital_twin_status", lambda *_args, **_kwargs: None)

    err = bridge._start_digital_twin_sync_when_ready(
        "dual robots",
        cfg,
        gazebo_process="digital_twin_dual_robots_gazebo",
        domains={"gazebo": 41, "hardware": 42},
    )

    assert err is None
    assert sync_started is True
    assert statuses[0]["message"] == "starting sync workers"
    assert statuses[-1]["message"] == "sync process running"


def test_digital_twin_dual_sync_starts_xarm6_and_ur5e_processes(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    commands: list[tuple[str, str]] = []

    monkeypatch.setattr("pathlib.Path.is_file", lambda _self: True)
    monkeypatch.setattr(
        bridge,
        "_start_tracked_ros2_command",
        lambda process_name, command, **_kwargs: commands.append((process_name, command)) or None,
    )

    err = bridge._start_digital_twin_sync_process(
        "dual robots",
        cfg,
        gazebo_domain_id=41,
        hardware_domain_id=42,
        domain_ids={"gazebo": 41, "hardware": 42, "hardware_xarm6": 42, "hardware_ur5e": 43},
    )

    assert err is None
    assert commands[0][0] == "digital_twin_dual_robots_sync_xarm6"
    assert "--robot xarm6" in commands[0][1]
    assert "--hardware-domain-id 42" in commands[0][1]
    assert "cais_digital_twin_dual_robots_xarm6.json" in commands[0][1]
    assert commands[1][0] == "digital_twin_dual_robots_sync_ur5e"
    assert "--robot ur5e" in commands[1][1]
    assert "--hardware-domain-id 42" in commands[1][1]
    assert "cais_digital_twin_dual_robots_ur5e.json" in commands[1][1]


def test_topic_publisher_count_parser_reads_topic_info() -> None:
    output = "Type: sensor_msgs/msg/JointState\n\nPublisher count: 2\n\nSubscription count: 1\n"

    assert SystemBridge._topic_publisher_count_from_output(output) == 2
    assert SystemBridge._topic_publisher_count_from_output("Unknown topic '/joint_states'") is None


def test_teleop_resolves_normal_gazebo_for_selected_robot(monkeypatch) -> None:
    bridge = SystemBridge()
    running = {"gazebo_xarm6"}
    monkeypatch.setattr(bridge, "ros2_proc_status", lambda name: "running" if name in running else "stopped")

    target = bridge.teleop_target("xarm6", "cartesian")

    assert target["environment"] == "gazebo"
    assert target["ros_domain_id"] == 0
    assert target["moveit_process"] == "gazebo_xarm6"
    assert target["ready"] is True
    assert target["warning"] == ""


def test_teleop_resolves_xarm_only_digital_twin_to_hardware_domain(monkeypatch) -> None:
    bridge = SystemBridge()
    running = {
        "digital_twin_xarm_only_gazebo",
        "digital_twin_xarm_only_hardware_xarm6_moveit",
        "digital_twin_xarm_only_sync",
    }
    monkeypatch.setattr(bridge, "ros2_proc_status", lambda name: "running" if name in running else "stopped")

    target = bridge.teleop_target("xarm6", "cartesian")

    assert target["environment"] == "real"
    assert target["ros_domain_id"] == 42
    assert target["moveit_process"] == "digital_twin_xarm_only_hardware_xarm6_moveit"
    assert target["ready"] is True


def test_teleop_missing_moveit_warns_before_backend_start(monkeypatch) -> None:
    bridge = SystemBridge()
    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "stopped")

    called = False

    def fake_request_payload(*_args, **_kwargs):
        nonlocal called
        called = True
        return True, "OK", {}

    monkeypatch.setattr(bridge, "_teleop_request_payload", fake_request_payload)

    ok, msg = bridge.teleop_jog("xarm6", "z", 5.0)

    assert ok is False
    assert msg == "MoveIt is not running. Start the matching Gazebo, Hardware Stack, or Digital Twin launch first."
    assert called is False


def test_teleop_wrong_robot_warns_before_backend_start(monkeypatch) -> None:
    bridge = SystemBridge()
    running = {"gazebo_xarm6"}
    monkeypatch.setattr(bridge, "ros2_proc_status", lambda name: "running" if name in running else "stopped")

    ok, msg = bridge.teleop_jog("ur5e", "z", 5.0)

    assert ok is False
    assert msg == "MoveIt for ur5e is not running. Start the matching ur5e launch first."


def test_teleop_backend_restarts_when_ros_domain_changes(monkeypatch) -> None:
    bridge = SystemBridge()
    stopped = 0
    commands: list[str] = []

    class FakeProc:
        stdin = None
        stdout = object()
        stderr = None
        pid = 12345

        def poll(self):
            return None

    bridge._teleop_server_proc = FakeProc()
    bridge._teleop_server_ros_domain_id = 0

    def fake_stop() -> None:
        nonlocal stopped
        stopped += 1
        bridge._teleop_server_proc = None
        bridge._teleop_server_ros_domain_id = None

    def fake_popen(args, **_kwargs):
        commands.append(args[-1])
        return FakeProc()

    monkeypatch.setattr(bridge, "_stop_teleop_server_locked", fake_stop)
    monkeypatch.setattr("cais_spade_llm.ui.bridge.subprocess.Popen", fake_popen)
    monkeypatch.setattr(bridge, "_read_teleop_response_locked", lambda timeout_sec: (True, "ready", {}))

    err = bridge._ensure_teleop_server_locked(42)

    assert err is None
    assert stopped == 1
    assert bridge._teleop_server_ros_domain_id == 42
    assert "export ROS_DOMAIN_ID=42;" in commands[0]

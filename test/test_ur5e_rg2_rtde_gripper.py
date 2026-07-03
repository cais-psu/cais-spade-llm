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
        "_ur5e_external_control_status",
        lambda: {"state": "running", "message": "External Control: running"},
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_trajectory_controller_status",
        lambda **_kwargs: {
            "state": "active",
            "message": "scaled_joint_trajectory_controller active",
            "error": "",
        },
    )

    status = bridge.hardware_stack_status("ur5e")

    assert status["overall"] == "running"
    assert status["driver"] == "running"
    assert status["gripper"] == "running"
    assert status["gripper_action"] == "ready"
    assert status["moveit"] == "running"
    assert status["external_control"] == "running"


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
        "_ur5e_external_control_status",
        lambda: {"state": "running", "message": "External Control: running"},
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_trajectory_controller_status",
        lambda **_kwargs: {
            "state": "active",
            "message": "scaled_joint_trajectory_controller active",
            "error": "",
        },
    )

    status = bridge.digital_twin_statuses()["ur5e only"]["hardware"]["status"]

    assert status["overall"] == "running"
    assert status["driver"] == "running"
    assert status["gripper"] == "running"
    assert status["gripper_action"] == "ready"
    assert status["moveit"] == "running"
    assert status["external_control"] == "running"


def test_digital_twin_ur5e_hardware_stack_uses_driver_gripper_moveit_order(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["ur5e only"]
    events: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        bridge,
        "hardware_connection_statuses",
        lambda force=False: {"ur5e": {"reachable": True, "ip": "192.168.1.172", "message": "OK"}},
    )
    monkeypatch.setattr(bridge, "_digital_twin_sim_mode", lambda _target: "monitor")

    def fake_start(process_name: str, launch_name: str, **_kwargs) -> None:
        events.append(("start", process_name, launch_name))
        return None

    monkeypatch.setattr(bridge, "_start_digital_twin_launch", fake_start)
    monkeypatch.setattr(
        bridge,
        "_wait_for_driver_ready",
        lambda robot, **_kwargs: events.append(("wait_driver", robot, "")) or None,
    )
    monkeypatch.setattr(
        bridge,
        "_ensure_ur5e_external_control_running",
        lambda **_kwargs: events.append(("external_control", "ros.urp", "")) or None,
    )
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
    monkeypatch.setattr(
        bridge,
        "_ensure_ros_controller_active",
        lambda controller, **_kwargs: events.append(("controller", controller, "")) or None,
    )

    err = bridge._start_digital_twin_hardware_stack(
        "ur5e only",
        cfg,
        ros_domain_id=42,
    )

    assert err is None
    assert events == [
        ("start", "digital_twin_ur5e_only_hardware_ur5e_driver", "hardware_ur5e_driver"),
        ("wait_driver", "ur5e", ""),
        ("external_control", "ros.urp", ""),
        ("wait_topic", "/joint_states", ""),
        ("start", "digital_twin_ur5e_only_hardware_ur5e_rg2_gripper", "hardware_ur5e_rg2_gripper"),
        ("wait_action", "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory", ""),
        ("start", "digital_twin_ur5e_only_hardware_ur5e_moveit", "hardware_ur5e_moveit"),
        ("wait_action", "/execute_trajectory", ""),
        ("controller", "scaled_joint_trajectory_controller", ""),
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
    monkeypatch.setattr(bridge, "_wait_for_driver_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bridge, "_ensure_ur5e_external_control_running", lambda **_kwargs: None)
    monkeypatch.setattr(bridge, "_wait_for_ros_action", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bridge, "_ensure_ros_controller_active", lambda *_args, **_kwargs: None)

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


def test_digital_twin_sync_xarm6_hardware_replay_uses_namespaced_controller_action() -> None:
    assert digital_twin_sync.ROBOTS["xarm6"]["hardware_trajectory_action"] == (
        "/xarm6/xarm6_traj_controller/follow_joint_trajectory"
    )
    assert digital_twin_sync.ROBOTS["ur5e"]["hardware_trajectory_action"] == (
        "/scaled_joint_trajectory_controller/follow_joint_trajectory"
    )
    body = (
        Path(__file__).resolve().parents[1]
        / "ros2"
        / "cais_lab_gazebo"
        / "scripts"
        / "digital_twin_sync.py"
    ).read_text(encoding="utf-8")
    paired_body = body.split("def run_paired_replay", 1)[1].split("def run_replay", 1)[0]

    assert "_wait_execute_trajectory_action" not in paired_body
    assert "_wait_move_group_action" in paired_body
    assert "_wait_follow_joint_trajectory_action" in paired_body
    assert "_publish_follow_joint_trajectory_action" in paired_body
    assert "_publish_paired_follow_joint_trajectory_actions" not in paired_body
    assert "_execute_move_group_joint_goal" in paired_body
    assert "_publish_execute_trajectory_action" not in paired_body
    assert '"xarm6/hardware_arm"' in paired_body
    assert '"ur5e/hardware_arm"' in paired_body
    assert "UR5E_HARDWARE_MOVE_GROUP" in paired_body
    assert '"hardware_moveit"' not in paired_body
    assert "list(xarm_plan[\"hardware_names\"]) + list(ur5e_plan[\"hardware_names\"])" not in paired_body


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
    assert digital_twin_sync._mirror_point_time_sec("ur5e") == pytest.approx(
        digital_twin_sync.UR5E_MIRROR_POINT_TIME_SEC
    )
    assert digital_twin_sync._mirror_point_time_sec("xarm6") == pytest.approx(
        digital_twin_sync.MIRROR_POINT_TIME_SEC
    )
    assert digital_twin_sync._mirror_min_publish_period_sec("ur5e") > 0.0
    assert digital_twin_sync._mirror_min_joint_delta_rad("ur5e") > 0.0

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
        positions=[0.0001] * 6,
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
        last_publish_ts=9.95,
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


def test_follow_joint_trajectory_diagnostic_detail_includes_rejection_context() -> None:
    detail = digital_twin_sync._follow_joint_trajectory_diagnostic_detail(
        domain_id=42,
        action_name="/scaled_joint_trajectory_controller/follow_joint_trajectory",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        points=[
            {"positions": [0.0] * 6, "time": 0.25},
            {"positions": [0.1] * 6, "time": 2.25},
            {"positions": [0.2] * 6, "time": 4.25},
            {"positions": [0.2] * 6, "time": 4.75},
        ],
        start_delay_sec=1.0,
        goal_time_tolerance_sec=2.0,
        header_stamp=True,
        controller_state="active",
    )

    assert "ROS_DOMAIN_ID=42" in detail
    assert "action=/scaled_joint_trajectory_controller/follow_joint_trajectory" in detail
    assert "joints=shoulder_pan_joint,shoulder_lift_joint,elbow_joint,wrist_1_joint,wrist_2_joint,wrist_3_joint" in detail
    assert "points=4" in detail
    assert "first_point_time=0.250" in detail
    assert "second_point_time=2.250" in detail
    assert "final_point_time=4.750" in detail
    assert "min_point_spacing=0.500" in detail
    assert "header_stamp=True" in detail
    assert "start_delay_sec=1.000" in detail
    assert "goal_time_tolerance_sec=2.000" in detail
    assert "controller_state=active" in detail


def test_paired_dispatch_does_not_block_on_ur5e_controller_inactive() -> None:
    body = (
        Path(__file__).resolve().parents[1]
        / "ros2"
        / "cais_lab_gazebo"
        / "scripts"
        / "digital_twin_sync.py"
    ).read_text(encoding="utf-8")
    worker_body = body.split("def _send_paired_follow_joint_trajectory_worker", 1)[1].split(
        "def _publish_paired_follow_joint_trajectory_actions",
        1,
    )[0]

    assert "_paired_follow_joint_trajectory_inactive_controller_results" not in worker_body
    assert "scaled_joint_trajectory_controller inactive before paired replay dispatch" not in body


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
        "_wait_xarm_gripper_endpoint",
        lambda *_args, **_kwargs: {"success": True, "message": "service available.", "method": "service"},
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


def test_paired_dual_replay_starts_both_robots_on_shared_timeline(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    path = tmp_path / "paired.json"
    path.write_text(json.dumps(_paired_dual_recording()), encoding="utf-8")
    publish_calls: list[dict[str, object]] = []
    action_calls: list[dict[str, object]] = []
    plan_calls: list[dict[str, object]] = []
    move_calls: list[dict[str, object]] = []
    xarm_gripper_calls: list[dict[str, object]] = []
    events: list[tuple[str, object]] = []

    def fake_read_snapshot(_domain_id: int, robot: str, source: str, _timeout: float) -> dict[str, object]:
        assert source == "hardware"
        assert _timeout == digital_twin_sync.HARDWARE_SNAPSHOT_TIMEOUT_SEC
        if robot == "xarm6":
            names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        else:
            names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
        return {"success": True, "snapshot": {name: 0.0 for name in names}}

    def fake_publish(
        domain_id: int,
        topics: list[str],
        joint_names: list[str],
        points: list[dict[str, object]],
        **_kwargs,
    ) -> dict[str, object]:
        events.append(("publish", tuple(joint_names)))
        publish_calls.append(
            {
                "domain_id": domain_id,
                "topics": list(topics),
                "joint_names": list(joint_names),
                "points": [dict(point) for point in points],
            }
        )
        return {"success": True, "message": "trajectory published."}

    def fake_action(
        domain_id: int,
        action_name: str,
        joint_names: list[str],
        points: list[dict[str, object]],
        **kwargs,
    ) -> dict[str, object]:
        events.append(("action", action_name))
        action_calls.append(
            {
                "domain_id": domain_id,
                "action_name": action_name,
                "joint_names": list(joint_names),
                "points": [dict(point) for point in points],
                "start_delay_sec": kwargs.get("start_delay_sec", 0.0),
            }
        )
        return {"success": True, "message": "action accepted; action succeeded."}

    def fake_paired_action(
        domain_id: int,
        arms: list[dict[str, object]],
        **kwargs,
    ) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for arm in arms:
            action_name = str(arm["action_name"])
            joint_names = [str(name) for name in list(arm["joint_names"])]
            points = [dict(point) for point in list(arm["points"])]
            key = str(arm["key"])
            events.append(("action", action_name))
            action_calls.append(
                {
                    "domain_id": domain_id,
                    "action_name": action_name,
                    "joint_names": joint_names,
                    "points": points,
                    "start_delay_sec": kwargs.get("start_delay_sec", 0.0),
                }
            )
            out[key] = {
                "success": True,
                "message": f"{action_name}: action accepted; action succeeded.",
                "header_stamp": True,
                "start_delay_sec": kwargs.get("start_delay_sec", 0.0),
            }
        return out

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
    monkeypatch.setattr(digital_twin_sync, "_publish_trajectory", fake_publish)
    monkeypatch.setattr(digital_twin_sync, "_publish_follow_joint_trajectory_action", fake_action)
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_paired_follow_joint_trajectory_actions",
        lambda *_args, **_kwargs: pytest.fail("paired dual replay must not use paired UR5e follow_joint_trajectory"),
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_plan_move_group_joint_goal",
        lambda domain_id, group_name, joint_names, start_positions, target_positions, **kwargs: plan_calls.append(
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
            "message": "/move_action: planned; action accepted; action succeeded; moveit_error_code=1.",
            "mode": "planned",
        },
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
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_execute_trajectory_action",
        lambda *_args, **_kwargs: pytest.fail("UR5e Teach commit should not use manual /execute_trajectory"),
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_xarm_gripper_sequence",
        lambda domain_id, points, **_kwargs: events.append(("xarm_gripper", domain_id))
        or xarm_gripper_calls.append(
            {"domain_id": domain_id, "points": [dict(point) for point in points]}
        ) or {"success": True, "message": "xarm gripper replay sent."},
    )

    code = digital_twin_sync.run_replay(_paired_replay_args(path))
    output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert code == 0
    assert output["success"] is True
    assert sorted(output["robots"]) == ["ur5e", "xarm6"]
    assert "ur5e/hardware_arm" in output["message"]
    assert "/move_action" in output["message"]
    assert "plan_and_execute" in output["message"]
    assert len(publish_calls) == 4
    action_names = {str(call["action_name"]) for call in action_calls}
    assert action_names <= {
        "/xarm6/xarm6_traj_controller/follow_joint_trajectory",
        "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory",
    }
    assert "/xarm6/xarm6_traj_controller/follow_joint_trajectory" in action_names
    assert "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory" in action_names
    assert any(
        call["action_name"] == "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory"
        for call in action_calls
    )
    assert not any(
        call["action_name"] == "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        for call in action_calls
    )
    assert len(plan_calls) == 2
    assert len(move_calls) == 2
    assert len(xarm_gripper_calls) == 1
    gazebo_arm_first_point_times = [
        float(call["points"][0]["time"])
        for call in publish_calls
        if len(call["joint_names"]) == 6
    ]
    assert len(set(gazebo_arm_first_point_times)) == 1
    xarm_hardware_arm = next(
        call for call in action_calls
        if call["action_name"] == "/xarm6/xarm6_traj_controller/follow_joint_trajectory"
    )
    assert xarm_hardware_arm["joint_names"] == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    assert xarm_hardware_arm["points"][0]["positions"] == pytest.approx([0.0] * 6)
    assert float(xarm_hardware_arm["points"][0]["time"]) == digital_twin_sync.HARDWARE_TRAJECTORY_CURRENT_POINT_SEC
    assert float(xarm_hardware_arm["points"][1]["time"]) == gazebo_arm_first_point_times[0]
    assert xarm_hardware_arm["points"][1]["positions"] == pytest.approx([0.1] * 6)
    assert float(xarm_hardware_arm["start_delay_sec"]) == digital_twin_sync.HARDWARE_TRAJECTORY_START_DELAY_SEC

    expected_ur5e_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    assert plan_calls[0]["joint_names"] == expected_ur5e_names
    assert plan_calls[0]["start_positions"] == pytest.approx([0.0] * 6)
    assert plan_calls[0]["target_positions"] == pytest.approx([0.2] * 6)
    assert plan_calls[1]["start_positions"] == pytest.approx([0.2] * 6)
    assert plan_calls[1]["target_positions"] == pytest.approx([0.25] * 6)
    assert move_calls[0]["joint_names"] == expected_ur5e_names
    assert move_calls[0]["start_positions"] == pytest.approx([0.0] * 6)
    assert move_calls[0]["target_positions"] == pytest.approx([0.2] * 6)
    assert move_calls[1]["start_positions"] == pytest.approx([0.2] * 6)
    assert move_calls[1]["target_positions"] == pytest.approx([0.25] * 6)
    assert all(not str(name).startswith("ur5e_") for name in move_calls[0]["joint_names"])
    assert {call["domain_id"] for call in publish_calls} == {41}
    assert {call["domain_id"] for call in action_calls} == {42}
    assert {call["domain_id"] for call in xarm_gripper_calls} == {42}
    assert float(xarm_gripper_calls[0]["points"][0]["time"]) < min(gazebo_arm_first_point_times)
    assert any("/xarm6_xarm6_traj_controller/joint_trajectory" in call["topics"] for call in publish_calls)
    assert any("/xarm6_xarm_gripper_traj_controller/joint_trajectory" in call["topics"] for call in publish_calls)
    assert any("/ur5e_joint_trajectory_controller/joint_trajectory" in call["topics"] for call in publish_calls)
    assert any("/ur5e_rg2_gripper_traj_controller/joint_trajectory" in call["topics"] for call in publish_calls)
    assert any(
        call["action_name"] == "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory"
        for call in action_calls
    )


def test_paired_dual_replay_skips_ur5e_hardware_gripper_when_ur5e_arm_replay_fails(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    path = tmp_path / "paired.json"
    path.write_text(json.dumps(_paired_dual_recording()), encoding="utf-8")
    action_calls: list[str] = []
    move_calls: list[int | None] = []
    xarm_gripper_calls: list[list[dict[str, object]]] = []

    def fake_read_snapshot(_domain_id: int, robot: str, source: str, _timeout: float) -> dict[str, object]:
        assert source == "hardware"
        if robot == "xarm6":
            names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        else:
            names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
        return {"success": True, "snapshot": {name: 0.0 for name in names}}

    def fake_action(
        _domain_id: int,
        action_name: str,
        _joint_names: list[str],
        _points: list[dict[str, object]],
        **_kwargs,
    ) -> dict[str, object]:
        action_calls.append(action_name)
        if action_name == "/scaled_joint_trajectory_controller/follow_joint_trajectory":
            return {
                "success": False,
                "message": f"{action_name}: action accepted; action aborted; error_code=-4.",
            }
        return {"success": True, "message": f"{action_name}: action accepted; action succeeded."}

    def fake_paired_action(
        _domain_id: int,
        arms: list[dict[str, object]],
        **_kwargs,
    ) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for arm in arms:
            action_name = str(arm["action_name"])
            key = str(arm["key"])
            action_calls.append(action_name)
            if action_name == "/scaled_joint_trajectory_controller/follow_joint_trajectory":
                out[key] = {
                    "success": False,
                    "message": f"{action_name}: action accepted; action aborted; error_code=-4.",
                }
            else:
                out[key] = {
                    "success": True,
                    "message": f"{action_name}: action accepted; action succeeded.",
                }
        return out

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
        lambda *_args, **_kwargs: {"success": True, "message": "trajectory published."},
    )
    monkeypatch.setattr(digital_twin_sync, "_publish_follow_joint_trajectory_action", fake_action)
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_paired_follow_joint_trajectory_actions",
        lambda *_args, **_kwargs: pytest.fail("paired dual replay must not use paired UR5e follow_joint_trajectory"),
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_plan_move_group_joint_goal",
        lambda *_args, **_kwargs: {"success": True, "message": "/move_action: planned.", "mode": "planned"},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_execute_trajectory_action",
        lambda *_args, **_kwargs: pytest.fail("UR5e Teach commit should not use manual /execute_trajectory"),
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_execute_move_group_joint_goal",
        lambda *_args, **kwargs: move_calls.append(kwargs.get("waypoint_index")) or {
            "success": False,
            "message": "/move_action: plan_and_execute; action accepted; action aborted; moveit_error_code=-4.",
            "mode": "plan_and_execute",
        },
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_xarm_gripper_sequence",
        lambda _domain_id, points, **_kwargs: xarm_gripper_calls.append([dict(point) for point in points])
        or {"success": True, "message": "xarm gripper replay sent."},
    )

    code = digital_twin_sync.run_replay(_paired_replay_args(path))
    output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert code == 8
    assert output["success"] is False
    assert "/xarm6/xarm6_traj_controller/follow_joint_trajectory" in action_calls
    assert "/scaled_joint_trajectory_controller/follow_joint_trajectory" not in action_calls
    assert move_calls == [1]
    assert "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory" not in action_calls
    assert len(xarm_gripper_calls) == 1
    assert "xarm6/hardware_gripper: xarm gripper replay sent." in output["message"]
    assert "/move_action: plan_and_execute" in output["message"]
    assert "ur5e/hardware_gripper: skipped because ur5e hardware arm failed." in output["message"]
    assert "plan_and_execute" in output["message"]


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
        lambda *_args, **kwargs: execute_calls.append(kwargs.get("waypoint_index")) or {"success": True},
    )

    code = digital_twin_sync.run_replay(_paired_replay_args(path))
    output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert code == 8
    assert output["success"] is False
    assert publish_calls == []
    assert action_calls == []
    assert execute_calls == []
    assert "ur5e MoveIt preflight failed waypoint 1/2" in output["message"]
    assert "xarm6/hardware_arm: not sent because ur5e MoveIt preflight failed" in output["message"]
    assert "ur5e/hardware_arm: not sent because ur5e MoveIt preflight failed" in output["message"]
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


def test_paired_dual_replay_does_not_require_execute_trajectory(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    path = tmp_path / "paired.json"
    path.write_text(json.dumps(_paired_dual_recording()), encoding="utf-8")
    publish_calls: list[str] = []
    action_calls: list[str] = []
    move_calls: list[int | None] = []

    def fake_read_snapshot(_domain_id: int, robot: str, source: str, _timeout: float) -> dict[str, object]:
        assert source == "hardware"
        if robot == "xarm6":
            names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        else:
            names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
        return {"success": True, "snapshot": {name: 0.0 for name in names}}

    def fake_action_preflight(_domain_id: int, action_name: str, **_kwargs) -> dict[str, object]:
        action_calls.append(action_name)
        return {"success": True, "message": "action server available."}

    def fake_paired_action(
        _domain_id: int,
        arms: list[dict[str, object]],
        **_kwargs,
    ) -> dict[str, dict[str, object]]:
        publish_calls.append("hardware")
        return {
            str(arm["key"]): {
                "success": True,
                "message": f"{str(arm['action_name'])}: action accepted; action succeeded.",
            }
            for arm in arms
        }

    monkeypatch.setattr(digital_twin_sync, "_read_snapshot", fake_read_snapshot)
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_execute_trajectory_action",
        lambda *_args, **_kwargs: {"success": False, "message": "/execute_trajectory unavailable"},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_move_group_action",
        lambda *_args, **_kwargs: action_calls.append("/move_action") or {
            "success": True,
            "message": "/move_action: action server available.",
        },
    )
    monkeypatch.setattr(digital_twin_sync, "_wait_follow_joint_trajectory_action", fake_action_preflight)
    monkeypatch.setattr(
        digital_twin_sync,
        "_wait_xarm_gripper_endpoint",
        lambda *_args, **_kwargs: {"success": True, "message": "service available.", "method": "service"},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_trajectory",
        lambda *_args, **_kwargs: publish_calls.append("gazebo") or {"success": True},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_follow_joint_trajectory_action",
        lambda *_args, **_kwargs: publish_calls.append("hardware") or {"success": True},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_paired_follow_joint_trajectory_actions",
        lambda *_args, **_kwargs: pytest.fail("paired dual replay must not use paired UR5e follow_joint_trajectory"),
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_execute_trajectory_action",
        lambda *_args, **_kwargs: publish_calls.append("execute") or {"success": True},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_xarm_gripper_sequence",
        lambda *_args, **_kwargs: publish_calls.append("xarm_gripper") or {"success": True},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_execute_move_group_joint_goal",
        lambda *_args, **kwargs: move_calls.append(kwargs.get("waypoint_index")) or {
            "success": True,
            "message": "/move_action: plan_and_execute; action accepted; action succeeded.",
            "mode": "plan_and_execute",
        },
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_plan_move_group_joint_goal",
        lambda *_args, **_kwargs: {"success": True, "message": "/move_action: planned.", "mode": "planned"},
    )

    code = digital_twin_sync.run_replay(_paired_replay_args(path))
    output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert code == 0
    assert output["success"] is True
    assert "/execute_trajectory" not in action_calls
    assert "/move_action" in action_calls
    assert "/scaled_joint_trajectory_controller/follow_joint_trajectory" not in action_calls
    assert move_calls == [1, 2]
    assert "plan_and_execute" in output["message"]
    assert "execute" not in publish_calls
    assert "gazebo" in publish_calls
    assert "hardware" in publish_calls


def test_paired_dual_prepared_replay_reprepares_when_hardware_start_pose_changes(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    recording = _paired_dual_recording()
    recording_path = tmp_path / "paired.json"
    prepared_path = tmp_path / "paired_prepared.json"
    recording_path.write_text(json.dumps(recording), encoding="utf-8")
    current_positions = {
        "xarm6": [0.0] * 6,
        "ur5e": [0.0] * 6,
    }
    action_calls: list[dict[str, object]] = []
    move_calls: list[dict[str, object]] = []

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
        return {
            "success": True,
            "snapshot": {
                name: current_positions[str(robot)][index]
                for index, name in enumerate(names)
            },
        }

    def fake_action(
        _domain_id: int,
        action_name: str,
        joint_names: list[str],
        points: list[dict[str, object]],
        **_kwargs,
    ) -> dict[str, object]:
        action_calls.append(
            {
                "action_name": action_name,
                "joint_names": list(joint_names),
                "points": [dict(point) for point in points],
            }
        )
        return {"success": True, "message": f"{action_name}: action accepted; action succeeded."}

    def fake_paired_action(
        _domain_id: int,
        arms: list[dict[str, object]],
        **_kwargs,
    ) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for arm in arms:
            action_name = str(arm["action_name"])
            joint_names = [str(name) for name in list(arm["joint_names"])]
            points = [dict(point) for point in list(arm["points"])]
            key = str(arm["key"])
            action_calls.append(
                {
                    "action_name": action_name,
                    "joint_names": joint_names,
                    "points": points,
                }
            )
            out[key] = {
                "success": True,
                "message": f"{action_name}: action accepted; action succeeded.",
            }
        return out

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
        "_wait_xarm_gripper_endpoint",
        lambda *_args, **_kwargs: {"success": True, "message": "service available.", "method": "service"},
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_trajectory",
        lambda *_args, **_kwargs: {"success": True, "message": "trajectory published."},
    )
    monkeypatch.setattr(digital_twin_sync, "_publish_follow_joint_trajectory_action", fake_action)
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_paired_follow_joint_trajectory_actions",
        lambda *_args, **_kwargs: pytest.fail("paired dual replay must not use paired UR5e follow_joint_trajectory"),
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_publish_xarm_gripper_sequence",
        lambda *_args, **_kwargs: {"success": True, "message": "xarm gripper replay sent."},
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
            "message": "/move_action: plan_and_execute; action accepted; action succeeded.",
            "mode": "plan_and_execute",
        },
    )
    monkeypatch.setattr(
        digital_twin_sync,
        "_plan_move_group_joint_goal",
        lambda *_args, **_kwargs: {"success": True, "message": "/move_action: planned.", "mode": "planned"},
    )

    prepare_args = _paired_replay_args(recording_path, prepared_file=prepared_path)
    prepared = digital_twin_sync._build_paired_replay_preparation(prepare_args, recording)
    assert prepared["success"] is True
    digital_twin_sync._atomic_json_write(prepared_path, prepared)

    current_positions["xarm6"] = [0.01] * 6
    current_positions["ur5e"] = [0.01] * 6
    first_code = digital_twin_sync.run_replay(
        _paired_replay_args(recording_path, prepared_file=prepared_path)
    )
    first_output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    current_positions["xarm6"] = [0.15] * 6
    current_positions["ur5e"] = [0.25] * 6
    second_code = digital_twin_sync.run_replay(
        _paired_replay_args(recording_path, prepared_file=prepared_path)
    )
    second_output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert first_code == 0
    assert first_output["used_prepared_file"] is True
    assert second_code == 0
    assert second_output["used_prepared_file"] is False
    assert not any(
        call["action_name"] == "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        for call in action_calls
    )
    assert len(move_calls) == 4
    assert move_calls[0]["start_positions"] == pytest.approx([0.01] * 6)
    assert move_calls[0]["target_positions"] == pytest.approx([0.2] * 6)
    assert move_calls[2]["start_positions"] == pytest.approx([0.25] * 6)
    assert move_calls[2]["target_positions"] == pytest.approx([0.2] * 6)


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

    result = bridge._replay_recording_file("dual robots", cfg, recording_path, "twin")

    assert result["success"] is True
    assert result["sync_resumed"] is True
    assert "hardware -> Gazebo sync resumed" in result["message"]
    assert events == ["initialize", "stop_mirror", "replay", "sync:hardware -> gazebo"]


def test_replay_in_twin_does_not_repair_ur5e_controller_after_gazebo_initialization(
    monkeypatch,
    tmp_path,
) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    recording_path = tmp_path / "paired.json"
    recording_path.write_text(json.dumps(_paired_dual_recording()), encoding="utf-8")
    events: list[str] = []

    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")
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
        "_ur5e_trajectory_controller_status",
        lambda **_kwargs: pytest.fail("dual replay readiness must not inspect scaled_joint_trajectory_controller"),
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_program_running_once",
        lambda **_kwargs: pytest.fail("dual replay readiness must not use dashboard program_running"),
    )
    monkeypatch.setattr(
        bridge,
        "repair_ur5e_trajectory_controller",
        lambda target: pytest.fail("dual replay readiness must not repair scaled_joint_trajectory_controller"),
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
    monkeypatch.setattr(
        bridge,
        "_run_digital_twin_sync",
        lambda *_args, **_kwargs: events.append("replay") or {"success": True, "message": "ok"},
    )
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_sync_when_ready",
        lambda *_args, **kwargs: events.append(f"resume:{kwargs.get('direction')}") or None,
    )

    result = bridge._replay_recording_file("dual robots", cfg, recording_path, "twin")

    assert result["success"] is True
    assert events == [
        "initialize",
        "stop_mirror",
        "replay",
        "resume:hardware -> gazebo",
    ]


def test_replay_in_twin_does_not_block_when_final_ur5e_controller_is_inactive(
    monkeypatch,
    tmp_path,
) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    recording_path = tmp_path / "paired.json"
    recording_path.write_text(json.dumps(_paired_dual_recording()), encoding="utf-8")
    events: list[str] = []

    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")
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
        "_ur5e_trajectory_controller_status",
        lambda **_kwargs: pytest.fail("dual replay readiness must not inspect scaled_joint_trajectory_controller"),
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_program_running_once",
        lambda **_kwargs: pytest.fail("dual replay readiness must not use dashboard program_running"),
    )
    monkeypatch.setattr(
        bridge,
        "repair_ur5e_trajectory_controller",
        lambda target: pytest.fail("dual replay readiness must not repair scaled_joint_trajectory_controller"),
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
    monkeypatch.setattr(
        bridge,
        "_run_digital_twin_sync",
        lambda *_args, **_kwargs: events.append("replay") or {"success": True, "message": "ok"},
    )
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_sync_when_ready",
        lambda *_args, **kwargs: events.append(f"resume:{kwargs.get('direction')}") or None,
    )

    result = bridge._replay_recording_file("dual robots", cfg, recording_path, "twin")

    assert result["success"] is True
    assert result["gazebo_initialization"] == "ran"
    assert events == [
        "initialize",
        "stop_mirror",
        "replay",
        "resume:hardware -> gazebo",
    ]


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


def test_dual_replay_ready_does_not_call_strict_ros_graph_preflights(
    monkeypatch,
    tmp_path,
) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    recording_path = tmp_path / "paired.json"
    recording_path.write_text(json.dumps(_paired_dual_recording()), encoding="utf-8")
    events: list[str] = []

    monkeypatch.setattr(
        bridge,
        "ros2_proc_status",
        lambda _name: "running",
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_digital_twin_dual_robots_hardware_ready",
        lambda *_args, **_kwargs: pytest.fail("dual replay must not use startup hardware readiness"),
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_action",
        lambda *_args, **_kwargs: pytest.fail("dual replay readiness must not use ros2 action list"),
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_topic_publisher",
        lambda *_args, **_kwargs: pytest.fail("dual replay readiness must not preflight /joint_states"),
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_program_running_once",
        lambda **_kwargs: pytest.fail("active scaled_joint_trajectory_controller must not require dashboard preflight"),
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_trajectory_controller_status",
        lambda **_kwargs: {
            "state": "active",
            "message": "scaled_joint_trajectory_controller active",
            "error": "",
        },
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
        lambda *_args, **_kwargs: events.append("replay") or {"success": True, "message": "ok"},
    )
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_sync_when_ready",
        lambda *_args, **kwargs: events.append(f"resume:{kwargs.get('direction')}") or None,
    )

    result = bridge._replay_recording_file("dual robots", cfg, recording_path, "twin")

    assert result["success"] is True
    assert result["sync_resumed"] is True
    assert events == ["stop_mirror", "replay", "resume:hardware -> gazebo"]


@pytest.mark.parametrize(
    ("dashboard_result", "dashboard_detail"),
    [
        (False, "program_running=False"),
        (None, "dashboard command timed out"),
    ],
)
def test_dual_replay_ready_allows_active_ur5e_controller_when_dashboard_is_stale(
    monkeypatch,
    dashboard_result,
    dashboard_detail,
) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    program_checks: list[str] = []

    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")
    monkeypatch.setattr(
        bridge,
        "_ur5e_trajectory_controller_status",
        lambda **_kwargs: {
            "state": "active",
            "message": "scaled_joint_trajectory_controller active",
            "error": "",
        },
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_program_running_once",
        lambda **_kwargs: program_checks.append(str(dashboard_detail)) or (dashboard_result, dashboard_detail),
    )
    monkeypatch.setattr(
        bridge,
        "repair_ur5e_trajectory_controller",
        lambda *_args, **_kwargs: pytest.fail("active controller must not be repaired"),
    )

    err = bridge._wait_for_digital_twin_dual_robots_replay_ready(
        "dual robots",
        cfg,
        ros_domain_id=42,
    )

    assert err is None
    assert program_checks == []


def test_dual_replay_ready_does_not_repair_inactive_ur5e_controller(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]

    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_action",
        lambda *_args, **_kwargs: pytest.fail("dual replay readiness must not use ros2 action list"),
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_topic_publisher",
        lambda *_args, **_kwargs: pytest.fail("dual replay readiness must not preflight /joint_states"),
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_program_running_once",
        lambda **_kwargs: pytest.fail("dual replay readiness must not use dashboard program_running"),
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_trajectory_controller_status",
        lambda **_kwargs: pytest.fail("dual replay readiness must not inspect scaled_joint_trajectory_controller"),
    )
    monkeypatch.setattr(
        bridge,
        "repair_ur5e_trajectory_controller",
        lambda *_args, **_kwargs: pytest.fail("dual replay readiness must not repair scaled_joint_trajectory_controller"),
    )

    err = bridge._wait_for_digital_twin_dual_robots_replay_ready(
        "dual robots",
        cfg,
        ros_domain_id=42,
    )

    assert err is None


def test_dual_replay_ready_does_not_check_external_control_when_scaled_controller_is_inactive(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    external_status: list[dict[str, object]] = []

    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")
    monkeypatch.setattr(
        bridge,
        "_ur5e_program_running_once",
        lambda **_kwargs: pytest.fail("dual replay readiness must not use dashboard program_running"),
    )
    monkeypatch.setattr(
        bridge,
        "_write_ur5e_external_control_status",
        lambda **kwargs: external_status.append(dict(kwargs)),
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_trajectory_controller_status",
        lambda **_kwargs: pytest.fail("dual replay readiness must not inspect scaled_joint_trajectory_controller"),
    )
    monkeypatch.setattr(
        bridge,
        "repair_ur5e_trajectory_controller",
        lambda *_args, **_kwargs: pytest.fail("controller repair must not run when External Control is stopped"),
    )

    err = bridge._wait_for_digital_twin_dual_robots_replay_ready(
        "dual robots",
        cfg,
        ros_domain_id=42,
    )

    assert err is None
    assert external_status == []


def test_dual_replay_ready_does_not_report_dashboard_unavailable_when_ur5e_controller_inactive(
    monkeypatch,
) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    external_status: list[dict[str, object]] = []

    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")
    monkeypatch.setattr(
        bridge,
        "_ur5e_trajectory_controller_status",
        lambda **_kwargs: pytest.fail("dual replay readiness must not inspect scaled_joint_trajectory_controller"),
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_program_running_once",
        lambda **_kwargs: pytest.fail("dual replay readiness must not use dashboard program_running"),
    )
    monkeypatch.setattr(
        bridge,
        "_write_ur5e_external_control_status",
        lambda **kwargs: external_status.append(dict(kwargs)),
    )
    monkeypatch.setattr(
        bridge,
        "repair_ur5e_trajectory_controller",
        lambda *_args, **_kwargs: pytest.fail("controller repair must not run without dashboard confirmation"),
    )

    err = bridge._wait_for_digital_twin_dual_robots_replay_ready(
        "dual robots",
        cfg,
        ros_domain_id=42,
    )

    assert err is None
    assert external_status == []


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


def test_dual_robots_hardware_moveit_maps_xarm_gripper_to_namespaced_gripper_action() -> None:
    launch_path = (
        Path(__file__).resolve().parents[1]
        / "ros2"
        / "cais_lab_gazebo"
        / "launch"
        / "dual_robots_hardware_moveit.launch.py"
    )
    body = launch_path.read_text(encoding="utf-8")

    assert '"xarm6/xarm_gripper"' in body
    assert '"action_ns": "gripper_action"' in body
    assert '"type": "GripperCommand"' in body
    assert '"joints": ["drive_joint"]' in body
    assert '"max_velocity": UR5E_MAX_VELOCITY' in body
    assert '"max_acceleration": UR5E_MAX_ACCELERATION' in body
    assert "UR5E_MAX_VELOCITY = 0.50" in body
    assert "UR5E_MAX_ACCELERATION = 0.80" in body
    assert "DEFAULT_VELOCITY_SCALING = 0.20" in body
    assert "DEFAULT_ACCELERATION_SCALING = 0.20" in body
    assert '"ur5e_rg2_gripper_traj_controller"' in body
    assert '"action_ns": "follow_joint_trajectory"' in body
    assert '"projection_evaluator": "joints(ur5e_rg2_finger_width)"' in body
    assert '"longest_valid_segment_fraction": 0.005' in body


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


def test_digital_twin_dual_robots_is_supported_with_monitor(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "stopped")

    row = bridge.digital_twin_statuses()["dual robots"]

    assert cfg["hardware_supported"] is True
    assert row["supported"] is True
    assert row["blocked_reason"] == ""
    assert bridge._digital_twin_gazebo_launch("dual robots", cfg) == "gazebo_dual_passive"
    assert row["sim_modes"] == ["monitor"]
    assert row["hardware"]["domains"] == {"xarm6": 42, "ur5e": 42}
    process_names = set(bridge._digital_twin_process_names(cfg))
    assert "digital_twin_dual_robots_hardware_moveit" in process_names
    assert "digital_twin_dual_robots_gazebo_moveit" in process_names
    assert "digital_twin_dual_robots_paired_markers" in process_names
    assert "digital_twin_dual_robots_hardware_xarm6_moveit" not in process_names
    assert "digital_twin_dual_robots_hardware_ur5e_moveit" not in process_names


def test_dual_drag_markers_script_defines_two_interactive_drag_balls() -> None:
    body = (
        Path(__file__).resolve().parents[1]
        / "ros2"
        / "cais_lab_gazebo"
        / "scripts"
        / "dual_drag_markers.py"
    ).read_text(encoding="utf-8")

    assert "InteractiveMarkerServer" in body
    assert "InteractiveMarkerControl.MOVE_3D" in body
    assert 'marker.name = f"{robot}_drag_ball"' in body
    assert 'for robot in ("xarm6", "ur5e")' in body
    assert "ExecuteTrajectory.Goal" in body
    assert "--status-file" in body
    assert "def _write_status(" in body
    assert "failed during {stage}" in body
    assert "/execute_trajectory error_code=" in body


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


def test_digital_twin_initial_pose_waits_for_gazebo_and_ur_external_control(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    events: list[tuple[str, str]] = []
    helper_timeouts: list[float] = []

    monkeypatch.setattr(
        bridge,
        "_wait_for_digital_twin_gazebo_controller_actions",
        lambda *_args, **_kwargs: pytest.fail(
            "initial gazebo pose must not wait for gazebo trajectory actions"
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_wait_with_ros2_daemon_retry",
        lambda label, fn, **_kwargs: events.append(("wait_retry", label)) or fn(),
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_topic_publisher",
        lambda *_args, **_kwargs: pytest.fail(
            "initial gazebo pose must use digital_twin_sync hardware snapshot"
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_driver_ready",
        lambda robot, **kwargs: events.append(("wait_driver", f"{robot}:{kwargs.get('process_name')}")) or None,
    )
    monkeypatch.setattr(
        bridge,
        "_refresh_ur5e_external_control_running_status",
        lambda **_kwargs: events.append(("external_control", "refresh")) or True,
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_service",
        lambda service, **_kwargs: events.append(("wait_service", service)) or None,
    )
    def fake_sync(args: list[str], **kwargs) -> dict[str, object]:
        helper_timeouts.append(float(kwargs.get("timeout_sec") or 0.0))
        events.append(("init", str(args[args.index("--robot") + 1])))
        return {"success": True}

    monkeypatch.setattr(bridge, "_run_digital_twin_sync", fake_sync)

    err = bridge._initialize_digital_twin_gazebo_from_hardware(
        "dual robots",
        cfg,
        gazebo_process="digital_twin_dual_robots_gazebo",
        domains={"gazebo": 41, "hardware": 42},
    )

    assert err is None
    assert events.index(("wait_service", "/controller_manager/list_controllers")) < events.index(
        ("init", "xarm6")
    )
    assert ("wait_driver", "ur5e:digital_twin_dual_robots_hardware_ur5e_driver") in events
    assert ("external_control", "refresh") in events
    assert ("init", "xarm6") in events
    assert ("init", "ur5e") in events
    assert helper_timeouts == [
        bridge._DIGITAL_TWIN_INITIALIZE_TIMEOUT_S,
        bridge._DIGITAL_TWIN_INITIALIZE_TIMEOUT_S,
    ]
    assert min(helper_timeouts) >= 60.0


def test_digital_twin_initial_pose_does_not_wait_for_gazebo_trajectory_actions(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]

    monkeypatch.setattr(
        bridge,
        "_wait_for_digital_twin_gazebo_controller_actions",
        lambda *_args, **_kwargs: pytest.fail(
            "initial hardware pose should not wait for /xarm6_xarm6_traj_controller/follow_joint_trajectory"
        ),
    )
    monkeypatch.setattr(bridge, "_wait_with_ros2_daemon_retry", lambda _label, fn, **_kwargs: fn())
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_topic_publisher",
        lambda *_args, **_kwargs: pytest.fail(
            "initial gazebo pose should not use bridge-side /joint_states publisher checks"
        ),
    )
    monkeypatch.setattr(bridge, "_wait_for_driver_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bridge, "_refresh_ur5e_external_control_running_status", lambda **_kwargs: True)
    monkeypatch.setattr(bridge, "_wait_for_ros_service", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        bridge,
        "_run_digital_twin_sync",
        lambda args, **_kwargs: {"success": True, "robot": str(args[args.index("--robot") + 1])},
    )

    err = bridge._initialize_digital_twin_gazebo_from_hardware(
        "dual robots",
        cfg,
        gazebo_process="digital_twin_dual_robots_gazebo",
        domains={"gazebo": 41, "hardware": 42},
    )

    assert err is None


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


def test_digital_twin_dual_robots_uses_hardware_ur5e_rg2_gripper() -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    ur5e_processes = bridge._digital_twin_hardware_processes_for_robot(cfg, "ur5e")

    assert ur5e_processes["gripper"] == "digital_twin_dual_robots_hardware_ur5e_rg2_gripper"
    assert bridge.ROS2_LAUNCH_CMDS["hardware_ur5e_rg2_gripper"].endswith(
        "ros2/cais_lab_gazebo/scripts/ur5e_rg2_rtde_gripper.py --robot-ip {ur5e_ip} --backend xmlrpc"
    )


def test_digital_twin_dual_hardware_stack_uses_xarm_ur5e_rg2_combined_moveit_order(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    events: list[tuple[str, str, str, int | None]] = []
    start_extra_args: dict[str, str] = {}

    monkeypatch.setattr(
        bridge,
        "hardware_connection_statuses",
        lambda force=False: {
            "xarm6": {"reachable": True, "ip": "192.168.1.240", "message": "OK"},
            "ur5e": {"reachable": True, "ip": "192.168.1.172", "message": "OK"},
        },
    )
    monkeypatch.setattr(bridge, "_digital_twin_sim_mode", lambda _target: "monitor")

    def fake_start(process_name: str, launch_name: str, **kwargs) -> None:
        events.append(("start", process_name, launch_name, kwargs.get("ros_domain_id")))
        start_extra_args[process_name] = str(kwargs.get("extra_args") or "")
        return None

    monkeypatch.setattr(bridge, "_start_digital_twin_launch", fake_start)
    monkeypatch.setattr(
        bridge,
        "_ensure_ur5e_external_control_running",
        lambda **kwargs: events.append(
            ("external_control", str(kwargs.get("process_name")), "", kwargs.get("ros_domain_id"))
        )
        or None,
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_service",
        lambda service, **kwargs: events.append(("wait_service", service, "", kwargs.get("ros_domain_id"))) or None,
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_driver_ready",
        lambda robot, **kwargs: events.append(("wait_driver", robot, "", kwargs.get("ros_domain_id"))) or None,
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_topic_publisher",
        lambda topic, **kwargs: events.append(("wait_topic", topic, "", kwargs.get("ros_domain_id"))) or None,
    )
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_action",
        lambda action, **kwargs: events.append(("wait_action", action, "", kwargs.get("ros_domain_id"))) or None,
    )
    monkeypatch.setattr(
        bridge,
        "_ensure_ros_controller_active",
        lambda controller, **kwargs: events.append(("controller", controller, "", kwargs.get("ros_domain_id"))) or None,
    )

    err = bridge._start_digital_twin_hardware_stack(
        "dual robots",
        cfg,
        ros_domain_id=42,
        domain_ids={"gazebo": 41, "hardware": 42, "hardware_xarm6": 42, "hardware_ur5e": 43},
    )

    assert err is None
    starts = [event for event in events if event[0] == "start"]
    assert starts == [
        ("start", "digital_twin_dual_robots_hardware_xarm6_driver", "hardware_xarm6_driver", 42),
        ("start", "digital_twin_dual_robots_hardware_ur5e_driver", "hardware_ur5e_driver", 42),
        ("start", "digital_twin_dual_robots_hardware_ur5e_rg2_gripper", "hardware_ur5e_rg2_gripper", 42),
        ("start", "digital_twin_dual_robots_hardware_moveit", "hardware_dual_robots_moveit", 42),
    ]
    assert (
        "external_control",
        "digital_twin_dual_robots_hardware_ur5e_driver",
        "",
        42,
    ) in events
    ur_driver_index = events.index(
        ("start", "digital_twin_dual_robots_hardware_ur5e_driver", "hardware_ur5e_driver", 42)
    )
    external_control_index = events.index(
        ("external_control", "digital_twin_dual_robots_hardware_ur5e_driver", "", 42)
    )
    wait_driver_event = ("wait_driver", "ur5e", "", 42)
    wait_driver_index = events.index(wait_driver_event)
    rg2_start_index = events.index(
        ("start", "digital_twin_dual_robots_hardware_ur5e_rg2_gripper", "hardware_ur5e_rg2_gripper", 42)
    )
    assert ur_driver_index < wait_driver_index < external_control_index < rg2_start_index
    assert (
        start_extra_args["digital_twin_dual_robots_hardware_ur5e_rg2_gripper"]
        == "--publish-arm-joint-states"
    )
    moveit_index = events.index(
        ("start", "digital_twin_dual_robots_hardware_moveit", "hardware_dual_robots_moveit", 42)
    )
    for index, event in enumerate(events):
        if event == wait_driver_event:
            continue
        if event[0] in {"wait_service", "wait_driver", "wait_topic", "wait_action", "controller"}:
            assert moveit_index < index
    assert (
        "wait_action",
        "/xarm6/xarm_gripper_traj_controller/follow_joint_trajectory",
        "",
        42,
    ) not in events
    assert (
        "wait_action",
        "/xarm6/xarm_gripper/gripper_action",
        "",
        42,
    ) not in events
    assert ("controller", "scaled_joint_trajectory_controller", "", 42) in events
    assert (
        "wait_action",
        "/scaled_joint_trajectory_controller/follow_joint_trajectory",
        "",
        42,
    ) in events
    assert (
        "wait_action",
        "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory",
        "",
        42,
    ) in events


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
    monkeypatch.setattr(
        bridge,
        "_ensure_ros_controller_active",
        lambda controller, **_kwargs: events.append(("controller", controller)) or None,
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


def test_digital_twin_dual_hardware_ready_attempts_ur5e_controller_after_xarm_action_timeout(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    events: list[tuple[str, str]] = []

    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")

    def fake_retry(label: str, wait_fn, **_kwargs) -> str | None:
        events.append(("retry", label))
        return wait_fn()

    def fake_wait_action(action: str, **_kwargs) -> str | None:
        events.append(("action", action))
        if action == "/xarm6/xarm6_traj_controller/follow_joint_trajectory":
            return "timeout"
        return None

    monkeypatch.setattr(bridge, "_wait_with_ros2_daemon_retry", fake_retry)
    monkeypatch.setattr(
        bridge,
        "_wait_for_ros_service",
        lambda service, **_kwargs: events.append(("service", service)) or None,
    )
    monkeypatch.setattr(bridge, "_wait_for_ros_action", fake_wait_action)
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
    monkeypatch.setattr(
        bridge,
        "_ensure_ros_controller_active",
        lambda controller, **_kwargs: events.append(("controller", controller)) or None,
    )

    err = bridge._wait_for_digital_twin_dual_robots_hardware_ready(
        cfg,
        ros_domain_id=42,
    )

    assert "xarm6 trajectory controller is not ready: timeout" in str(err)
    assert ("controller", "scaled_joint_trajectory_controller") in events
    assert ("action", "/scaled_joint_trajectory_controller/follow_joint_trajectory") in events
    assert ("action", "/execute_trajectory") in events


def test_digital_twin_dual_start_does_not_gate_sync_on_passive_gazebo_actions(monkeypatch) -> None:
    bridge = SystemBridge()
    events: list[tuple[str, ...]] = []

    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "stopped")
    monkeypatch.setattr(
        bridge,
        "hardware_connection_statuses",
        lambda force=False: {
            "xarm6": {"reachable": True, "ip": "192.168.1.240", "message": "OK"},
            "ur5e": {"reachable": True, "ip": "192.168.1.172", "message": "OK"},
        },
    )
    monkeypatch.setattr(bridge, "_shutdown_gazebo_prewarm_controllers", lambda: None)
    monkeypatch.setattr(bridge, "_force_kill_digital_twin_helpers", lambda: events.append(("cleanup", "digital_twin_helpers")))
    monkeypatch.setattr(bridge, "_kill_stale_gazebo_helpers", lambda: None)
    monkeypatch.setattr(bridge, "_force_kill_gazebo_core", lambda reason="": None)

    def fake_start(process_name: str, launch_name: str, **kwargs) -> None:
        events.append(("start", launch_name, str(kwargs.get("ros_domain_id"))))
        return None

    def fake_wait_service(service: str, **kwargs) -> None:
        events.append(("wait_service", service, str(kwargs.get("ros_domain_id"))))
        return None

    def fake_wait_action(action: str, **kwargs) -> None:
        events.append(("wait_action", action, str(kwargs.get("ros_domain_id"))))
        return None

    monkeypatch.setattr(bridge, "_start_digital_twin_launch", fake_start)
    monkeypatch.setattr(bridge, "_ensure_ur5e_external_control_running", lambda **_kwargs: None)
    monkeypatch.setattr(bridge, "_wait_for_ros_service", fake_wait_service)
    monkeypatch.setattr(bridge, "_wait_for_ros_action", fake_wait_action)
    monkeypatch.setattr(bridge, "_wait_for_driver_ready", lambda robot, **kwargs: events.append(("wait_driver", robot, str(kwargs.get("ros_domain_id")))) or None)
    monkeypatch.setattr(bridge, "_wait_for_ros_topic_publisher", lambda topic, **kwargs: events.append(("wait_topic", topic, str(kwargs.get("ros_domain_id")))) or None)
    monkeypatch.setattr(bridge, "_ensure_ros_controller_active", lambda controller, **kwargs: events.append(("controller", controller, str(kwargs.get("ros_domain_id")))) or None)
    monkeypatch.setattr(
        bridge,
        "_initialize_digital_twin_gazebo_from_hardware",
        lambda *_args, **_kwargs: events.append(("init", "hardware_to_gazebo")) or None,
    )
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_dual_drag_markers",
        lambda _cfg, **kwargs: events.append(
            ("markers", str(kwargs.get("mode")), str(kwargs.get("ros_domain_id")))
        )
        or None,
    )
    monkeypatch.setattr(bridge, "_start_digital_twin_sync_process", lambda *_args, **_kwargs: events.append(("sync", "start")) or None)

    err = bridge.digital_twin_start("dual robots")

    assert err is None
    sync_index = events.index(("sync", "start"))
    moveit_index = events.index(("start", "hardware_dual_robots_moveit", "42"))
    markers_index = events.index(("markers", "monitor", "42"))
    gazebo_index = events.index(("start", "gazebo_dual_passive", "41"))
    init_index = events.index(("init", "hardware_to_gazebo"))
    assert moveit_index < markers_index < gazebo_index
    assert gazebo_index < init_index < sync_index
    assert (
        "wait_action",
        "/ur5e_joint_trajectory_controller/follow_joint_trajectory",
        "41",
    ) not in events
    assert (
        "wait_action",
        "/xarm6_xarm6_traj_controller/follow_joint_trajectory",
        "41",
    ) not in events


def test_digital_twin_dual_start_writes_status_when_hardware_launch_fails(monkeypatch) -> None:
    bridge = SystemBridge()
    statuses: list[dict[str, object]] = []
    sync_statuses: list[dict[str, object]] = []
    events: list[tuple[str, str]] = []
    expected = (
        "UR5e External Control auto reconnect failed loading ros.urp: command timed out after 8s. "
        "Set pendant to Remote Control or press Play on ros.urp manually."
    )

    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "stopped")
    monkeypatch.setattr(
        bridge,
        "hardware_connection_statuses",
        lambda force=False: {
            "xarm6": {"reachable": True, "ip": "192.168.1.240", "message": "OK"},
            "ur5e": {"reachable": True, "ip": "192.168.1.172", "message": "OK"},
        },
    )
    monkeypatch.setattr(bridge, "_shutdown_gazebo_prewarm_controllers", lambda: None)
    monkeypatch.setattr(bridge, "_force_kill_digital_twin_helpers", lambda: None)
    monkeypatch.setattr(bridge, "_kill_stale_gazebo_helpers", lambda: None)
    monkeypatch.setattr(bridge, "_force_kill_gazebo_core", lambda reason="": None)
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_dual_robots_hardware_launches",
        lambda *_args, **_kwargs: expected,
    )
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_dual_drag_markers",
        lambda *_args, **_kwargs: events.append(("markers", "started")) or None,
    )
    monkeypatch.setattr(
        bridge,
        "_ensure_digital_twin_launch",
        lambda _process, launch, **_kwargs: events.append(("launch", launch)) or None,
    )
    monkeypatch.setattr(
        bridge,
        "_write_digital_twin_status",
        lambda _target, payload: statuses.append(dict(payload)),
    )
    monkeypatch.setattr(
        bridge,
        "_write_digital_twin_sync_status",
        lambda _target, _cfg, payload: sync_statuses.append(dict(payload)),
    )

    err = bridge.digital_twin_start("dual robots")

    assert err == expected
    assert statuses[-1]["state"] == "waiting"
    assert statuses[-1]["message"] == expected
    assert statuses[-1]["last_error"] == expected
    assert sync_statuses[-1]["state"] == "waiting"
    assert sync_statuses[-1]["message"] == expected
    assert sync_statuses[-1]["last_error"] == expected
    assert events == []


def test_digital_twin_dual_start_retries_sync_without_cleanup_when_stack_running(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    events: list[tuple[str, str]] = []
    running = {
        "digital_twin_dual_robots_gazebo",
        "digital_twin_dual_robots_hardware_xarm6_driver",
        "digital_twin_dual_robots_hardware_ur5e_driver",
        "digital_twin_dual_robots_hardware_ur5e_rg2_gripper",
        "digital_twin_dual_robots_hardware_moveit",
    }

    monkeypatch.setattr(
        bridge,
        "ros2_proc_status",
        lambda name: "running" if name in running else "stopped",
    )
    monkeypatch.setattr(bridge, "_shutdown_gazebo_prewarm_controllers", lambda: events.append(("cleanup", "prewarm")))
    monkeypatch.setattr(bridge, "_force_kill_digital_twin_helpers", lambda: events.append(("cleanup", "helpers")))
    monkeypatch.setattr(bridge, "_kill_stale_gazebo_helpers", lambda: events.append(("cleanup", "gazebo_helpers")))
    monkeypatch.setattr(bridge, "_force_kill_gazebo_core", lambda reason="": events.append(("cleanup", reason)))
    monkeypatch.setattr(
        bridge,
        "hardware_connection_statuses",
        lambda force=False: {
            "xarm6": {"reachable": True, "ip": "192.168.1.240", "message": "OK"},
            "ur5e": {"reachable": True, "ip": "192.168.1.172", "message": "OK"},
        },
    )
    monkeypatch.setattr(bridge, "_start_digital_twin_launch", lambda process, launch, **kwargs: events.append(("start", launch)) or None)
    monkeypatch.setattr(bridge, "_ensure_ur5e_external_control_running", lambda **_kwargs: None)
    monkeypatch.setattr(bridge, "_wait_for_ros_service", lambda service, **kwargs: events.append(("wait_service", service)) or None)
    monkeypatch.setattr(bridge, "_wait_for_driver_ready", lambda robot, **kwargs: events.append(("wait_driver", robot)) or None)
    monkeypatch.setattr(bridge, "_wait_for_ros_topic_publisher", lambda topic, **kwargs: events.append(("wait_topic", topic)) or None)
    monkeypatch.setattr(bridge, "_wait_for_ros_action", lambda action, **kwargs: events.append(("wait_action", action)) or None)
    monkeypatch.setattr(bridge, "_ensure_ros_controller_active", lambda controller, **kwargs: events.append(("controller", controller)) or None)
    monkeypatch.setattr(
        bridge,
        "_initialize_digital_twin_gazebo_from_hardware",
        lambda *_args, **_kwargs: events.append(("init", "unexpected")) or None,
    )
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_dual_drag_markers",
        lambda _cfg, **kwargs: events.append(("markers", str(kwargs.get("mode")))) or None,
    )
    monkeypatch.setattr(bridge, "_start_digital_twin_sync_process", lambda *_args, **_kwargs: events.append(("sync", "start")) or None)
    monkeypatch.setattr(
        bridge,
        "_restart_ros2_daemon_for_discovery",
        lambda **_kwargs: events.append(("restart", "ros2_daemon")) or None,
    )

    err = bridge.digital_twin_start("dual robots")

    assert err is None
    assert ("restart", "ros2_daemon") in events
    assert ("markers", "monitor") in events
    assert ("sync", "start") in events
    assert not any(event[0] == "cleanup" for event in events)
    assert not any(event[0] == "start" for event in events)
    assert ("init", "unexpected") not in events


def test_digital_twin_dual_gazebo_wait_failure_keeps_hardware_moveit_running(monkeypatch) -> None:
    bridge = SystemBridge()
    events: list[tuple[str, str]] = []
    statuses: list[dict[str, object]] = []

    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "stopped")
    monkeypatch.setattr(
        bridge,
        "hardware_connection_statuses",
        lambda force=False: {
            "xarm6": {"reachable": True, "ip": "192.168.1.240", "message": "OK"},
            "ur5e": {"reachable": True, "ip": "192.168.1.172", "message": "OK"},
        },
    )
    monkeypatch.setattr(bridge, "_shutdown_gazebo_prewarm_controllers", lambda: None)
    monkeypatch.setattr(bridge, "_force_kill_digital_twin_helpers", lambda: None)
    monkeypatch.setattr(bridge, "_kill_stale_gazebo_helpers", lambda: None)
    monkeypatch.setattr(bridge, "_force_kill_gazebo_core", lambda reason="": None)
    monkeypatch.setattr(bridge, "digital_twin_stop", lambda target: events.append(("stop", target)) or None)
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_launch",
        lambda _process_name, launch_name, **_kwargs: events.append(("start", launch_name)) or None,
    )
    monkeypatch.setattr(bridge, "_ensure_ur5e_external_control_running", lambda **_kwargs: None)

    def fake_wait_service(service: str, **kwargs) -> str | None:
        if kwargs.get("ros_domain_id") == 41:
            events.append(("wait_gazebo_service", service))
            return "timeout"
        events.append(("wait_service", service))
        return None

    monkeypatch.setattr(bridge, "_wait_for_ros_service", fake_wait_service)
    monkeypatch.setattr(bridge, "_wait_for_driver_ready", lambda robot, **_kwargs: events.append(("wait_driver", robot)) or None)
    monkeypatch.setattr(bridge, "_wait_for_ros_topic_publisher", lambda topic, **_kwargs: events.append(("wait_topic", topic)) or None)
    monkeypatch.setattr(bridge, "_wait_for_ros_action", lambda action, **_kwargs: events.append(("wait_action", action)) or None)
    monkeypatch.setattr(bridge, "_ensure_ros_controller_active", lambda controller, **_kwargs: events.append(("controller", controller)) or None)
    monkeypatch.setattr(
        bridge,
        "_start_digital_twin_dual_drag_markers",
        lambda _cfg, **kwargs: events.append(("markers", str(kwargs.get("mode")))) or None,
    )
    monkeypatch.setattr(bridge, "_start_digital_twin_sync_process", lambda *_args, **_kwargs: events.append(("sync", "start")) or None)
    monkeypatch.setattr(
        bridge,
        "_write_digital_twin_sync_status",
        lambda _target, _cfg, payload: statuses.append(dict(payload)),
    )

    err = bridge.digital_twin_start("dual robots")

    assert err is None
    assert ("start", "hardware_dual_robots_moveit") in events
    assert ("markers", "monitor") in events
    assert ("sync", "start") not in events
    assert not any(event[0] == "stop" for event in events)
    assert statuses
    assert statuses[-1]["state"] == "waiting"
    assert "gazebo is not ready for sync" in str(statuses[-1]["message"])


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


def test_controller_state_parser_reads_scaled_controller_state() -> None:
    output = (
        "controller_manager_msgs.srv.ListControllers_Response(controller=["
        "controller_manager_msgs.msg.ControllerState("
        "name='scaled_joint_trajectory_controller', state='inactive', "
        "type='ur_controllers/ScaledJointTrajectoryController')])"
    )

    state = SystemBridge._controller_state_from_list_controllers_output(
        output,
        "scaled_joint_trajectory_controller",
    )

    assert state == "inactive"


def test_switch_controller_ok_parser_reads_response() -> None:
    ok_output = "controller_manager_msgs.srv.SwitchController_Response(ok=True)"
    false_output = "controller_manager_msgs.srv.SwitchController_Response(ok=False)"

    assert SystemBridge._switch_controller_ok_from_output(ok_output) is True
    assert SystemBridge._switch_controller_ok_from_output(false_output) is False


def test_ur_program_running_parser_reads_response() -> None:
    running_output = "ur_dashboard_msgs.srv.IsProgramRunning_Response(program_running=True, success=True)"
    stopped_output = "ur_dashboard_msgs.srv.IsProgramRunning_Response(program_running=False, success=True)"

    assert SystemBridge._ur_program_running_from_output(running_output) is True
    assert SystemBridge._ur_program_running_from_output(stopped_output) is False


def test_ros_service_success_parser_reads_response() -> None:
    ok_output = "std_srvs.srv.Trigger_Response(success=True, message='ok')"
    false_output = "ur_dashboard_msgs.srv.Load_Response(answer='failed', success=False)"

    assert SystemBridge._ros_service_success_from_output(ok_output) is True
    assert SystemBridge._ros_service_success_from_output(false_output) is False


def test_repair_ur5e_controller_reactivates_when_external_control_running(monkeypatch) -> None:
    bridge = SystemBridge()
    events: list[tuple[str, str, int | None]] = []

    monkeypatch.setattr(
        bridge,
        "_ur5e_controller_context",
        lambda _target=None: (
            {
                "target": "dual robots",
                "process_name": "digital_twin_dual_robots_hardware_ur5e_driver",
                "ros_domain_id": 42,
                "controller_name": "scaled_joint_trajectory_controller",
            },
            "",
        ),
    )
    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")
    monkeypatch.setattr(
        bridge,
        "_ur5e_trajectory_controller_status",
        lambda **_kwargs: {
            "state": "inactive",
            "message": "scaled_joint_trajectory_controller inactive",
            "error": "",
        },
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_program_running_once",
        lambda **_kwargs: (True, "program_running=True"),
    )
    monkeypatch.setattr(bridge, "_write_ur5e_external_control_status", lambda **_kwargs: None)

    def fake_activate(controller_name: str, **kwargs):
        events.append(("activate", controller_name, kwargs.get("ros_domain_id")))
        return None

    monkeypatch.setattr(bridge, "_activate_ros_controller", fake_activate)
    monkeypatch.setattr(bridge, "_wait_for_ros_controller_active", lambda *_args, **_kwargs: None)

    result = bridge.repair_ur5e_trajectory_controller("dual robots")

    assert result["success"] is True
    assert result["message"] == "reactivated scaled_joint_trajectory_controller"
    assert result["state"] == "active"
    assert events == [("activate", "scaled_joint_trajectory_controller", 42)]


def test_repair_ur5e_controller_requires_manual_play_when_external_control_stopped(monkeypatch) -> None:
    bridge = SystemBridge()
    external_status: list[dict] = []

    monkeypatch.setattr(
        bridge,
        "_ur5e_controller_context",
        lambda _target=None: (
            {
                "target": "ur5e only",
                "process_name": "digital_twin_ur5e_only_hardware_ur5e_driver",
                "ros_domain_id": 42,
                "controller_name": "scaled_joint_trajectory_controller",
            },
            "",
        ),
    )
    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")
    monkeypatch.setattr(
        bridge,
        "_ur5e_trajectory_controller_status",
        lambda **_kwargs: {
            "state": "inactive",
            "message": "scaled_joint_trajectory_controller inactive",
            "error": "",
        },
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_program_running_once",
        lambda **_kwargs: (False, "program_running=False"),
    )
    monkeypatch.setattr(bridge, "_write_ur5e_external_control_status", lambda **kwargs: external_status.append(kwargs))
    monkeypatch.setattr(
        bridge,
        "_activate_ros_controller",
        lambda *_args, **_kwargs: pytest.fail("switch_controller should not be called"),
    )

    result = bridge.repair_ur5e_trajectory_controller("ur5e only")

    assert result["success"] is False
    assert result["message"] == "External Control is not running; press Play on ros.urp"
    assert external_status[-1]["state"] == "manual Play required"


def test_repair_ur5e_controller_skips_when_already_active(monkeypatch) -> None:
    bridge = SystemBridge()

    monkeypatch.setattr(
        bridge,
        "_ur5e_controller_context",
        lambda _target=None: (
            {
                "target": "ur5e only",
                "process_name": "digital_twin_ur5e_only_hardware_ur5e_driver",
                "ros_domain_id": 42,
                "controller_name": "scaled_joint_trajectory_controller",
            },
            "",
        ),
    )
    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")
    monkeypatch.setattr(
        bridge,
        "_ur5e_trajectory_controller_status",
        lambda **_kwargs: {
            "state": "active",
            "message": "scaled_joint_trajectory_controller active",
            "error": "",
        },
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_program_running_once",
        lambda **_kwargs: pytest.fail("External Control should not be checked"),
    )
    monkeypatch.setattr(
        bridge,
        "_activate_ros_controller",
        lambda *_args, **_kwargs: pytest.fail("switch_controller should not be called"),
    )

    result = bridge.repair_ur5e_trajectory_controller("ur5e only")

    assert result["success"] is True
    assert result["message"] == "scaled_joint_trajectory_controller already active"
    assert result["state"] == "active"


def test_repair_ur5e_controller_reports_switch_controller_false(monkeypatch) -> None:
    bridge = SystemBridge()

    monkeypatch.setattr(
        bridge,
        "_ur5e_controller_context",
        lambda _target=None: (
            {
                "target": "dual robots",
                "process_name": "digital_twin_dual_robots_hardware_ur5e_driver",
                "ros_domain_id": 42,
                "controller_name": "scaled_joint_trajectory_controller",
            },
            "",
        ),
    )
    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")
    monkeypatch.setattr(
        bridge,
        "_ur5e_trajectory_controller_status",
        lambda **_kwargs: {
            "state": "inactive",
            "message": "scaled_joint_trajectory_controller inactive",
            "error": "",
        },
    )
    monkeypatch.setattr(
        bridge,
        "_ur5e_program_running_once",
        lambda **_kwargs: (True, "program_running=True"),
    )
    monkeypatch.setattr(bridge, "_write_ur5e_external_control_status", lambda **_kwargs: None)
    monkeypatch.setattr(
        bridge,
        "_activate_ros_controller",
        lambda *_args, **_kwargs: "switch_controller returned ok=False for scaled_joint_trajectory_controller",
    )

    result = bridge.repair_ur5e_trajectory_controller("dual robots")

    assert result["success"] is False
    assert result["message"] == "switch_controller returned ok=False for scaled_joint_trajectory_controller"
    assert result["state"] == "inactive"


def test_digital_twin_hardware_status_refreshes_stale_ur5e_external_control_before_auto_repair(monkeypatch) -> None:
    bridge = SystemBridge()
    cfg = bridge._DIGITAL_TWIN_TARGETS["dual robots"]
    refreshed = False
    repairs: list[str] = []

    monkeypatch.setattr(bridge, "ros2_proc_status", lambda _name: "running")

    def fake_external_control_status() -> dict[str, object]:
        return (
            {
                "state": "running",
                "message": "External Control: running",
                "error": "",
            }
            if refreshed
            else {
                "state": "manual Play required",
                "message": "External Control is not running; press Play on ros.urp",
                "error": "command timed out after 5s",
            }
        )

    def fake_refresh(**kwargs) -> bool:
        nonlocal refreshed
        assert kwargs.get("ros_domain_id") == 42
        refreshed = True
        return True

    monkeypatch.setattr(bridge, "_ur5e_external_control_status", fake_external_control_status)
    monkeypatch.setattr(bridge, "_refresh_ur5e_external_control_running_status", fake_refresh)
    monkeypatch.setattr(
        bridge,
        "_attach_ur5e_trajectory_controller_status",
        lambda status, **_kwargs: status.update(
            {
                "trajectory_controller": "inactive",
                "trajectory_controller_message": "scaled_joint_trajectory_controller inactive",
                "trajectory_controller_error": "",
            }
        )
        or status,
    )
    monkeypatch.setattr(
        bridge,
        "_schedule_ur5e_controller_auto_repair",
        lambda target: repairs.append(target),
    )

    status = bridge._digital_twin_hardware_status(cfg)

    assert refreshed is True
    assert status["ur5e"]["external_control"] == "running"
    assert status["ur5e"]["trajectory_controller"] == "inactive"
    assert repairs == ["dual robots"]


def test_ur5e_external_control_already_running_skips_load_play(monkeypatch) -> None:
    bridge = SystemBridge()
    calls: list[tuple[str, str, str]] = []

    monkeypatch.setattr(bridge, "_wait_for_ros_services", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        bridge,
        "_ur5e_program_running_once",
        lambda **_kwargs: (True, "program_running=True"),
    )

    def fake_call(service_name: str, service_type: str, request: str, **_kwargs):
        calls.append((service_name, service_type, request))
        return True, "success=True"

    monkeypatch.setattr(bridge, "_call_dashboard_service", fake_call)

    err = bridge._ensure_ur5e_external_control_running(
        process_name="hardware_ur5e_driver",
        ros_domain_id=42,
    )

    assert err is None
    assert calls == []


def test_ur5e_external_control_stopped_loads_then_plays(monkeypatch) -> None:
    bridge = SystemBridge()
    calls: list[tuple[str, str, str]] = []

    monkeypatch.setattr(bridge, "_wait_for_ros_services", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        bridge,
        "_ur5e_program_running_once",
        lambda **_kwargs: (False, "program_running=False"),
    )
    monkeypatch.setattr(bridge, "_wait_for_ur_program_running", lambda *_args, **_kwargs: None)

    def fake_call(service_name: str, service_type: str, request: str, **_kwargs):
        calls.append((service_name, service_type, request))
        return True, "success=True"

    monkeypatch.setattr(bridge, "_call_dashboard_service", fake_call)

    err = bridge._ensure_ur5e_external_control_running(
        process_name="hardware_ur5e_driver",
        ros_domain_id=42,
    )

    assert err is None
    assert calls == [
        ("/dashboard_client/load_program", "ur_dashboard_msgs/srv/Load", "{filename: ros.urp}"),
        ("/dashboard_client/play", "std_srvs/srv/Trigger", "{}"),
    ]


def test_ur5e_external_control_play_failure_reports_manual_play(monkeypatch) -> None:
    bridge = SystemBridge()

    monkeypatch.setattr(bridge, "_wait_for_ros_services", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        bridge,
        "_ur5e_program_running_once",
        lambda **_kwargs: (False, "program_running=False"),
    )

    def fake_call(service_name: str, service_type: str, request: str, **_kwargs):
        if service_name == "/dashboard_client/play":
            return True, "std_srvs.srv.Trigger_Response(success=False, message='remote control disabled')"
        return True, "success=True"

    monkeypatch.setattr(bridge, "_call_dashboard_service", fake_call)

    err = bridge._ensure_ur5e_external_control_running(
        process_name="hardware_ur5e_driver",
        ros_domain_id=42,
    )

    assert err is not None
    assert "Set pendant to Remote Control or press Play on ros.urp manually" in err


def test_ur5e_external_control_play_failure_clears_when_program_running(monkeypatch) -> None:
    bridge = SystemBridge()
    running_states = iter(
        [
            (False, "program_running=False"),
            (True, "program_running=True"),
        ]
    )
    status_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(bridge, "_wait_for_ros_services", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bridge, "_ur5e_program_running_once", lambda **_kwargs: next(running_states))
    monkeypatch.setattr(
        bridge,
        "_write_ur5e_external_control_status",
        lambda **kwargs: status_payloads.append(dict(kwargs)),
    )

    def fake_call(service_name: str, service_type: str, request: str, **_kwargs):
        if service_name == "/dashboard_client/play":
            return True, "std_srvs.srv.Trigger_Response(success=False, message='remote control disabled')"
        return True, "success=True"

    monkeypatch.setattr(bridge, "_call_dashboard_service", fake_call)

    err = bridge._ensure_ur5e_external_control_running(
        process_name="hardware_ur5e_driver",
        ros_domain_id=42,
    )

    assert err is None
    assert status_payloads[-1]["state"] == "running"
    assert status_payloads[-1]["message"] == "External Control: running"


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


def test_teleop_resolves_hardware_stack_for_selected_robot(monkeypatch) -> None:
    bridge = SystemBridge()
    running = {"hardware_ur5e_driver", "hardware_ur5e_rg2_gripper", "hardware_ur5e_moveit"}
    monkeypatch.setattr(bridge, "ros2_proc_status", lambda name: "running" if name in running else "stopped")

    target = bridge.teleop_target("ur5e", "joint")

    assert target["environment"] == "real"
    assert target["moveit_process"] == "hardware_ur5e_moveit"
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


def test_teleop_resolves_ur5e_only_digital_twin_to_hardware_domain(monkeypatch) -> None:
    bridge = SystemBridge()
    running = {
        "digital_twin_ur5e_only_gazebo",
        "digital_twin_ur5e_only_hardware_ur5e_driver",
        "digital_twin_ur5e_only_hardware_ur5e_rg2_gripper",
        "digital_twin_ur5e_only_hardware_ur5e_moveit",
        "digital_twin_ur5e_only_sync",
    }
    monkeypatch.setattr(bridge, "ros2_proc_status", lambda name: "running" if name in running else "stopped")

    target = bridge.teleop_target("ur5e", "cartesian")

    assert target["environment"] == "real"
    assert target["ros_domain_id"] == 42
    assert target["moveit_process"] == "digital_twin_ur5e_only_hardware_ur5e_moveit"
    assert target["ready"] is True


def test_teleop_resolves_dual_robots_digital_twin_to_shared_hardware_domain(monkeypatch) -> None:
    bridge = SystemBridge()
    running = {
        "digital_twin_dual_robots_gazebo",
        "digital_twin_dual_robots_hardware_xarm6_driver",
        "digital_twin_dual_robots_hardware_ur5e_driver",
        "digital_twin_dual_robots_hardware_ur5e_rg2_gripper",
        "digital_twin_dual_robots_hardware_moveit",
        "digital_twin_dual_robots_sync_xarm6",
        "digital_twin_dual_robots_sync_ur5e",
    }
    monkeypatch.delenv("CAIS_DIGITAL_TWIN_HARDWARE_XARM6_DOMAIN_ID", raising=False)
    monkeypatch.delenv("CAIS_DIGITAL_TWIN_HARDWARE_UR5E_DOMAIN_ID", raising=False)
    monkeypatch.setattr(bridge, "ros2_proc_status", lambda name: "running" if name in running else "stopped")

    xarm_target = bridge.teleop_target("xarm6", "cartesian")
    xarm_gripper_target = bridge.teleop_target("xarm6", "gripper")
    ur5e_target = bridge.teleop_target("ur5e", "cartesian")

    assert xarm_target["environment"] == "real"
    assert xarm_target["ros_domain_id"] == 42
    assert xarm_target["moveit_process"] == "digital_twin_dual_robots_hardware_moveit"
    assert xarm_target["ready"] is True
    assert xarm_gripper_target["ros_domain_id"] == 42
    assert xarm_gripper_target["gripper_process"] == "digital_twin_dual_robots_hardware_xarm6_driver"
    assert xarm_gripper_target["required_processes"] == ["digital_twin_dual_robots_hardware_xarm6_driver"]
    assert xarm_gripper_target["ready"] is True
    assert ur5e_target["environment"] == "real"
    assert ur5e_target["ros_domain_id"] == 42
    assert ur5e_target["moveit_process"] == "digital_twin_dual_robots_hardware_moveit"
    assert ur5e_target["ready"] is True


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


def test_teleop_missing_ur5e_rg2_bridge_warns_before_backend_start(monkeypatch) -> None:
    bridge = SystemBridge()
    running = {"hardware_ur5e_driver", "hardware_ur5e_moveit"}
    monkeypatch.setattr(bridge, "ros2_proc_status", lambda name: "running" if name in running else "stopped")

    ok, msg = bridge.teleop_gripper("ur5e", "open")

    assert ok is False
    assert msg == "RG2 gripper bridge is not running. Start UR5e Hardware Stack, ur5e only Digital Twin, or dual robots Digital Twin first."


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

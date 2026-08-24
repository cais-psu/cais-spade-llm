"""Focused no-motion contracts for headless and dual Hardware Stacks."""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
    XARM6_JOINT_NAMES,
)
from cais_spade_llm.resources.robot.hardware_pick_place_controller import (
    XARM6_HARDWARE_JOINT_NAMES,
    XArm6HardwareController,
)
from cais_spade_llm.ui import bridge as bridge_module
from cais_spade_llm.ui import ros2_processes
from cais_spade_llm.ui.bridge import SystemBridge

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_operator_insertion_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bridge_module,
        "_MOVE_INSERT_TRIALS_DIR",
        tmp_path / "operator_move_insert_trials",
    )
    monkeypatch.setattr(
        bridge_module,
        "_INSERTION_DEMONSTRATIONS_DIR",
        tmp_path / "operator_move_insert_demonstrations",
    )
    monkeypatch.setattr(
        bridge_module,
        "_HARDWARE_STATE_DIR",
        tmp_path / "operator_hardware_state",
    )


def _digital_twin_sync_module() -> Any:
    path = ROOT / "ros2/cais_lab_robotics/scripts/digital_twin_sync.py"
    spec = importlib.util.spec_from_file_location("xarm6_tf_readiness_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wsl_robot_route_script_waits_for_exact_interface_and_source() -> None:
    source = (ROOT / "scripts/cais-robot-routes.sh").read_text(encoding="utf-8")

    assert "route replace \"$ROBOT_IP/32\" dev \"$IFACE\" src \"$SRC_IP\"" in source
    assert '$0 ~ (" dev " iface " ")' in source
    assert '$0 ~ (" src " src "([[:space:]]|$)")' in source
    assert "ip route get 192.168.1.240 >/dev/null" not in source
    assert source.rstrip().endswith("exit 1")


def test_hardware_stack_launch_commands_use_direct_control_without_move_group() -> None:
    commands = ros2_processes.build_ros2_launch_cmds(
        project_root=ROOT,
        venv_python=Path("/venv/python"),
        ur5e_rg2_gripper_script=Path("/scripts/rg2.py"),
        ur5e_rtde_trajectory_script=Path("/scripts/rtde.py"),
        ur5e_rtde_trajectory_status=Path("/tmp/rtde.json"),
    )

    stack_processes = {
        process_name
        for stack in SystemBridge._HARDWARE_STACKS.values()
        for process_name in stack
    }
    assert all("moveit" not in process_name for process_name in stack_processes)
    assert (
        "dual_robots_hardware_state_publisher.launch.py"
        in commands["hardware_robot_state_publisher"]
    )
    assert "moveit" not in commands["hardware_robot_state_publisher"].lower()
    for process_name in stack_processes:
        command = commands[process_name]
        assert "gazebo" not in command.lower()
        assert "realsense" not in command.lower()
        assert "perception" not in command.lower()


def test_xarm6_cartesian_service_uses_embedded_driver_namespace() -> None:
    robot_config = json.loads(
        (ROOT / "cais_spade_llm/initialization/resources/robot_xarm6.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_config = yaml.safe_load(
        (
            ROOT
            / "ros2/cais_lab_robotics/config/hardware_runtime/"
            "xarm6_ur5e_hardware_runtime.yaml"
        ).read_text(encoding="utf-8")
    )
    launch_source = (
        ROOT / "ros2/cais_lab_robotics/launch/xarm6_hardware_driver.launch.py"
    ).read_text(encoding="utf-8")

    expected_service = "/xarm6/xarm/set_position"
    assert (
        robot_config["xarm6"]["real"]["controller"]["hardware_cartesian_service"]
        == expected_service
    )
    assert runtime_config["xarm6"]["hardware_cartesian_service"] == expected_service
    assert runtime_config["xarm6"]["moveit"]["arm_joint_limits"] == {
        "max_velocity": 1.391,
        "max_acceleration": 6.5,
    }
    assert runtime_config["xarm6"]["cartesian"]["speed_mm_s"] == 50.0
    assert runtime_config["xarm6"]["cartesian"]["max_speed_mm_s"] == 100.0
    assert runtime_config["xarm6"]["cartesian"]["acceleration_mm_s2"] == 42.25
    assert runtime_config["ur5e"]["rtde"]["cartesian_speed_m_s"] == 0.10
    assert runtime_config["ur5e"]["rtde"]["cartesian_max_speed_m_s"] == 0.10
    assert 'PushRosNamespace(namespace)' in launch_source
    assert "from xarm_msgs.msg import RobotMsg" in launch_source
    assert '"/xarm6/xarm/robot_states"' in launch_source
    assert '"joint1",' in launch_source
    assert "SetRemap" in launch_source
    assert 'src="/controller_manager/list_controllers"' in launch_source
    assert 'dst=f"/{namespace}/controller_manager/list_controllers"' in launch_source
    assert 'src="/controller_manager/switch_controller"' in launch_source
    assert 'dst=f"/{namespace}/controller_manager/switch_controller"' in launch_source
    assert '"hw_ns": "xarm"' in launch_source
    assert '"extra_robot_api_params_path"' in launch_source
    service_config = yaml.safe_load(
        (
            ROOT
            / "ros2/cais_lab_robotics/config/hardware_runtime/"
            "xarm6_robot_api_services.yaml"
        ).read_text(encoding="utf-8")
    )
    services = service_config["ufactory_driver"]["ros__parameters"]["services"]
    assert services["set_mode"] is True
    assert services["set_state"] is True
    assert services["set_tcp_maxacc"] is True
    assert services["vc_set_cartesian_velocity"] is True
    assert "except KeyboardInterrupt:" in launch_source
    assert "if rclpy.ok():" in launch_source


def test_xarm6_force_torque_sensor_is_recorded_but_move_insert_stays_unavailable() -> None:
    robot_config = json.loads(
        (ROOT / "cais_spade_llm/initialization/resources/robot_xarm6.json").read_text(
            encoding="utf-8"
        )
    )

    sensor = robot_config["xarm6"]["real"]["controller"][
        "ufactory_six_axis_force_torque_sensor"
    ]
    assert sensor == {
        "installed": True,
        "wrench_feedback_exposed_to_cais": False,
        "zeroing_exposed_to_cais": False,
        "force_control_exposed_to_cais": False,
        "move_insert_action": None,
    }


def test_dual_hardware_stack_uses_only_exact_physical_processes() -> None:
    assert SystemBridge._HARDWARE_STACKS["dual robots"] == (
        "hardware_xarm6_driver",
        "hardware_ur5e_rtde_trajectory_server",
        "hardware_ur5e_rg2_gripper",
        "hardware_robot_state_publisher",
    )


def test_dual_hardware_status_reports_direct_control_and_state_publisher() -> None:
    bridge = object.__new__(SystemBridge)
    states = {
        "hardware_xarm6_driver": "running",
        "hardware_ur5e_rtde_trajectory_server": "running",
        "hardware_ur5e_rg2_gripper": "running",
        "hardware_robot_state_publisher": "running",
    }
    bridge.ros2_proc_status = lambda name: states.get(name, "stopped")
    bridge._ur5e_rtde_trajectory_status = lambda: {
        "state": "ready",
        "joint_states_fresh": True,
        "cartesian_jog_ready": True,
        "cartesian_function_ready": True,
        "cartesian_frame_validation_message": "ready",
        "updated_at": time.time(),
        "rtde_reset_required": False,
    }
    bridge._hardware_cartesian_readiness = {
        "xarm6": {
            "cartesian_jog_ready": True,
            "cartesian_function_ready": True,
            "message": "ready",
        }
    }

    status = bridge.hardware_stack_status("dual robots")

    assert status["overall"] == "running"
    assert status["xarm6"]["driver"] == "running"
    assert status["xarm6"]["control"] == "direct trajectory action"
    assert status["xarm6"]["joint_control"] == "ready"
    assert status["xarm6"]["cartesian_control"] == "ready"
    assert status["xarm6"]["cartesian_jog_ready"] is True
    assert status["xarm6"]["cartesian_function_ready"] is True
    assert status["ur5e"]["driver"] == "running"
    assert status["ur5e"]["gripper"] == "running"
    assert status["ur5e"]["rtde_trajectory_server"] == "ready"
    assert status["ur5e"]["joint_states_fresh"] is True
    assert status["control"] == "direct"
    assert status["state_publisher"] == "running"
    assert "moveit" not in status


def test_stopped_rtde_does_not_show_an_unrelated_stale_failure() -> None:
    bridge = object.__new__(SystemBridge)
    bridge.ros2_proc_status = lambda _name: "stopped"
    bridge._hardware_stack_selected = "dual robots"
    bridge._hardware_stack_lifecycle_state = "failed"
    bridge._hardware_stack_last_error = "xArm6 controllers were not active"
    bridge._hardware_stack_failed_process = "hardware_xarm6_driver"
    bridge._hardware_stack_failed_return_code = None
    bridge._hardware_stack_process_log_path = "/tmp/xarm6.log"
    bridge._ur5e_rtde_trajectory_status = lambda: {
        "state": "failed",
        "message": "stale RTDE failure",
        "process_id": 1234,
    }

    dual_status = bridge.hardware_stack_status("dual robots")
    ur5e_status = bridge.hardware_stack_status("ur5e")

    assert dual_status["ur5e"]["rtde_trajectory_server"] == "stopped"
    assert dual_status["ur5e"]["rtde_trajectory_message"] == (
        "UR5e RTDE trajectory server stopped"
    )
    assert ur5e_status["rtde_trajectory_server"] == "stopped"
    assert "stale RTDE failure" not in ur5e_status["rtde_trajectory_message"]


def _dual_start_bridge() -> tuple[SystemBridge, list[str], list[tuple[str, str]]]:
    bridge = object.__new__(SystemBridge)
    bridge._ros2_procs = {}
    bridge._hardware_stack_process_generations = {}
    bridge._hardware_stack_validated_process_pids = {}
    bridge._hardware_stack_stationary_results = {}
    bridge._hardware_stack_cartesian_jog_reset_results = {}
    bridge._ur5e_cartesian_jog_state_uncertain = False
    bridge._ur5e_cartesian_jog_state_uncertain_reason = ""
    bridge._xarm6_cartesian_jog_state_uncertain = False
    bridge._xarm6_cartesian_jog_state_uncertain_reason = ""
    bridge._hardware_stack_lifecycle_lock = threading.Lock()
    bridge.robot_env = "gazebo"
    bridge._xarm6_robot_function_agent = None
    bridge._ur5e_robot_function_agent = None
    bridge.hardware_connection_statuses = lambda force=False: {
        "xarm6": {"reachable": True, "ip": "192.168.1.240", "message": "OK"},
        "ur5e": {"reachable": True, "ip": "192.168.1.172", "message": "OK"},
    }
    bridge._any_running = lambda _names: False
    states: dict[str, str] = {}
    started: list[str] = []
    stopped: list[tuple[str, str]] = []
    next_pid = {"value": 1000}
    bridge.ros2_proc_status = lambda name: states.get(name, "stopped")

    def _process(pid: int) -> SimpleNamespace:
        process = SimpleNamespace(pid=pid, return_code=None)
        process.poll = lambda process=process: process.return_code
        return process

    def _start(name: str) -> None:
        started.append(name)
        states[name] = "running"
        next_pid["value"] += 1
        bridge._ros2_procs[name] = _process(next_pid["value"])
        return None

    bridge.ros2_start = _start

    def _start_rtde(name: str, **_kwargs: Any) -> None:
        if states.get(name) != "running":
            started.append(name)
            states[name] = "running"
            next_pid["value"] += 1
            bridge._ros2_procs[name] = _process(next_pid["value"])
        return None

    bridge._start_ur5e_rtde_trajectory_server = _start_rtde
    def _stop(name: str, reason: str = "explicit_stop") -> None:
        stopped.append((name, reason))
        states[name] = "stopped"
        process = bridge._ros2_procs.pop(name, None)
        if process is not None:
            process.return_code = 0

    bridge.ros2_stop = _stop
    bridge._wait_with_ros2_daemon_retry = lambda _label, callback: callback()
    bridge._wait_for_ros_service = lambda *_args, **_kwargs: None
    bridge._wait_for_ros_services = lambda *_args, **_kwargs: None
    bridge._wait_for_ros_controllers_active = lambda *_args, **_kwargs: None
    bridge._wait_for_ros_action = lambda *_args, **_kwargs: None
    bridge._wait_for_ros_topic_publisher = lambda *_args, **_kwargs: None
    bridge._prepare_xarm6_trajectory_mode = lambda **_kwargs: None
    bridge._wait_for_xarm6_relayed_tf_ready = lambda **_kwargs: None
    bridge._hardware_cartesian_readiness = {}

    def _validate_xarm6_cartesian_frames(**_kwargs: Any) -> None:
        bridge._hardware_cartesian_readiness["xarm6"] = {
            "cartesian_jog_ready": True,
            "cartesian_function_ready": True,
            "generation": bridge._hardware_stack_lifecycle_generation,
            "message": "ready",
        }
        return None

    def _wait_for_ur5e_cartesian_frame_ready(*_args: Any, **_kwargs: Any) -> None:
        bridge._hardware_cartesian_readiness["ur5e"] = {
            "cartesian_jog_ready": True,
            "cartesian_function_ready": True,
            "generation": bridge._hardware_stack_lifecycle_generation,
            "message": "ready",
        }
        return None

    bridge._validate_xarm6_cartesian_frames = _validate_xarm6_cartesian_frames
    bridge._wait_for_ur5e_cartesian_frame_ready = (
        _wait_for_ur5e_cartesian_frame_ready
    )
    bridge._ur5e_rtde_trajectory_status = lambda: {}
    bridge._snapshot_robot_waypoint = lambda *_args, **_kwargs: {
        "positions": [0.0] * 6,
        "joint_names": [f"joint{index}" for index in range(1, 7)],
        "pose": {},
    }
    bridge._stop_teleop_server = lambda: None

    def _validate_hardware_stationary(
        robot: str,
        *,
        ros_domain_id: int,
    ) -> None:
        bridge._hardware_stack_stationary_results[robot] = {
            "generation": bridge._hardware_stack_lifecycle_generation,
            "stationary_ready": True,
            "message": "stationary feedback ready",
            "diagnostics": {"ros_domain_id": ros_domain_id},
        }
        return None

    bridge._validate_hardware_stationary = _validate_hardware_stationary
    bridge._clear_ros_action_service_snapshot = lambda: None
    bridge._default_ros_domain_id = lambda: 0
    return bridge, started, stopped


def test_xarm6_hardware_start_uses_driver_action_and_state_publisher_only() -> None:
    bridge, started, stopped = _dual_start_bridge()

    error = bridge.ros2_start_hardware_stack("xarm6")

    assert error is None
    assert started == [
        "hardware_xarm6_driver",
        "hardware_robot_state_publisher",
    ]
    assert stopped == []
    assert all("moveit" not in process_name for process_name in started)


@pytest.mark.parametrize("stack_name", ["ur5e", "dual robots", "xarm6"])
def test_terminal_ur5e_trial_does_not_control_general_hardware_stack(
    stack_name: str,
    tmp_path: Path,
) -> None:
    bridge, started, stopped = _dual_start_bridge()
    bridge._move_insert_trials_dir = tmp_path / "move_insert_trials"
    bridge._hardware_state_dir = tmp_path / "hardware_state"
    trial_id = "move-insert-terminal-ur5e-repair"
    bridge._move_insert_trials = {
        trial_id: {
            "target": "ur5e only",
            "robot": "ur5e",
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
            "trial_id": trial_id,
            "active": False,
            "completion_motion_active": False,
            "review_required": False,
            "recovery_required": False,
            "normal_repair_required": True,
            "hardware_stack_repair_required": True,
            "part_clamped": True,
            "released": False,
            "lifted": False,
            "completion_eligible": False,
            "qualified": False,
            "status": "failed",
            "message": "Terminal UR5e move_insert requires Repair Hardware Stack.",
        }
    }
    bridge._move_insert_last_trial_by_selection = {}
    bridge._insertion_demonstration = {
        "recording_id": "stale-recording-marker",
        "robot": "ur5e",
        "part_name": "MG",
        "active": True,
    }

    status = bridge.hardware_stack_status(stack_name)
    assert status["overall"] == "stopped"
    assert status["lifecycle_state"] == "stopped"
    assert status["hardware_stack_repair_required"] is False
    assert status["hardware_stack_repair_reason"] == ""
    assert status["hardware_stack_operation_blocked_reason"] == ""
    for operation in ("start", "stop", "repair"):
        assert bridge._hardware_stack_lifecycle_motion_error(
            operation,
            stack_name,
        ) == ""

    start_error = bridge.ros2_start_hardware_stack(stack_name)

    assert start_error is None
    assert started
    assert stopped == []
    pending = bridge._move_insert_pending_review()
    if stack_name == "xarm6":
        assert pending is not None
        assert pending["trial_id"] == trial_id
    else:
        assert pending is None
        trial = bridge._move_insert_trials[trial_id]
        assert trial["normal_repair_required"] is False
        assert trial["hardware_stack_repair_required"] is False
        assert trial["repair_evidence"]["hardware_stack"] == stack_name


@pytest.mark.parametrize("stationary_error", [None, "UR5e is still moving"])
def test_ur5e_hardware_start_clears_generic_uncertainty_only_after_fresh_readiness(
    stationary_error: str | None,
) -> None:
    bridge, started, _stopped = _dual_start_bridge()
    assert (
        bridge._write_ur5e_hardware_state_uncertainty(
            reason="generic uncertainty",
            source="hardware",
            source_id="uncertainty-test",
        )
        == ""
    )
    bridge._ur5e_robot_function_state_uncertain = False
    bridge._ur5e_robot_function_state_uncertain_reason = ""
    bridge._ur5e_cartesian_jog_state_uncertain = False
    bridge._ur5e_cartesian_jog_state_uncertain_reason = ""
    stationary_calls: list[tuple[str, int]] = []

    def _stationary(robot: str, *, ros_domain_id: int) -> str | None:
        stationary_calls.append((robot, ros_domain_id))
        return stationary_error

    bridge._validate_hardware_stationary = _stationary

    error = bridge.ros2_start_hardware_stack("ur5e")

    assert started
    assert stationary_calls == [("ur5e", 0)]
    if stationary_error:
        assert stationary_error in str(error)
        assert bridge._hardware_stack_lifecycle_state == "failed"
        assert bridge._ur5e_robot_function_state_uncertain is True
        assert bridge._ur5e_cartesian_jog_state_uncertain is True
        durable, durable_error = bridge._read_ur5e_hardware_state_uncertainty()
        assert durable_error == ""
        assert durable is not None
    else:
        assert error is None
        assert bridge._hardware_stack_lifecycle_state == "running"
        assert bridge._ur5e_robot_function_state_uncertain is False
        assert bridge._ur5e_cartesian_jog_state_uncertain is False
        durable, durable_error = bridge._read_ur5e_hardware_state_uncertainty()
        assert durable_error == ""
        assert durable is None


def test_ur5e_hardware_start_fails_closed_when_uncertainty_cannot_be_cleared() -> None:
    bridge, started, _stopped = _dual_start_bridge()
    assert (
        bridge._write_ur5e_hardware_state_uncertainty(
            reason="generic uncertainty",
            source="hardware",
            source_id="uncertainty-clear-test",
        )
        == ""
    )
    bridge._clear_ur5e_hardware_state_uncertainty = (
        lambda: "Could not clear durable UR5e hardware state: storage failed"
    )

    error = bridge.ros2_start_hardware_stack("ur5e")

    assert started
    assert "storage failed" in str(error)
    assert bridge._hardware_stack_lifecycle_state == "failed"
    assert bridge._ur5e_robot_function_state_uncertain is True
    assert bridge._ur5e_cartesian_jog_state_uncertain is True


def test_dual_hardware_component_start_keeps_warm_teleop_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cais_spade_llm.ui import bridge as bridge_module
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge.robot_env = "real"
    bridge._HARDWARE_PROCESS_NAMES = {"hardware_ur5e_rtde_trajectory_server"}
    bridge._GAZEBO_PROCESS_NAMES = set()
    bridge.ROS2_LAUNCH_CMDS = {
        "hardware_ur5e_rtde_trajectory_server": "unused",
    }
    bridge._BASE_HARDWARE_PROCESS_NAMES = set()
    bridge._ROS2_ENV = ""
    bridge._ros2_procs = {}
    bridge.ros2_proc_status = lambda _name: "stopped"
    bridge._ros2_launch_prereq_error = lambda _name: None
    bridge._any_running = lambda _names: False
    bridge._render_ros2_launch_cmd = lambda _name: "true"
    bridge._register_ui_process = lambda *_args: None
    stop_calls: list[str] = []
    bridge._stop_teleop_server = lambda: stop_calls.append("teleop")
    monkeypatch.setattr(
        bridge_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: SimpleNamespace(pid=1234, poll=lambda: None),
    )

    error = bridge.ros2_start("hardware_ur5e_rtde_trajectory_server")

    assert error is None
    assert stop_calls == []


@pytest.mark.parametrize("stack_name", ["xarm6", "dual robots"])
def test_xarm6_hardware_start_waits_for_embedded_cartesian_service(
    stack_name: str,
) -> None:
    bridge, _started, _stopped = _dual_start_bridge()
    observed_services: list[str] = []
    observed_topics: list[str] = []

    def _wait_for_service(service_name: str, **_kwargs: Any) -> None:
        observed_services.append(service_name)
        return None

    bridge._wait_for_ros_service = _wait_for_service
    bridge._wait_for_ros_services = lambda service_names, **_kwargs: (
        observed_services.extend(service_names) or None
    )
    bridge._wait_for_ros_topic_publisher = lambda topic_name, **_kwargs: (
        observed_topics.append(topic_name) or None
    )

    assert bridge.ros2_start_hardware_stack(stack_name) is None
    assert "/xarm6/xarm/set_position" in observed_services
    assert "/xarm6/set_position" not in observed_services
    assert "/xarm6/xarm/vc_set_cartesian_velocity" in observed_services
    assert "/xarm6/controller_manager/list_controllers" in observed_services
    assert "/xarm6/controller_manager/switch_controller" in observed_services
    assert "/xarm6/xarm/robot_states" in observed_topics


def test_ur5e_hardware_start_uses_rtde_rg2_and_state_publisher_only() -> None:
    bridge, started, stopped = _dual_start_bridge()
    bridge._ur5e_robot_function_state_uncertain = True
    bridge._ur5e_robot_function_state_uncertain_reason = "motion result was not observed"

    error = bridge.ros2_start_hardware_stack("ur5e")

    assert error is None
    assert bridge._ur5e_robot_function_state_uncertain is False
    assert bridge._ur5e_robot_function_state_uncertain_reason == ""
    assert started == [
        "hardware_ur5e_rtde_trajectory_server",
        "hardware_ur5e_rg2_gripper",
        "hardware_robot_state_publisher",
    ]
    assert stopped == []
    assert all("moveit" not in process_name for process_name in started)


@pytest.mark.parametrize("stack_name", ["xarm6", "ur5e", "dual robots"])
def test_ur5e_hardware_start_stops_calibration_tf_before_and_after_full_publisher(
    stack_name: str,
) -> None:
    bridge, started, stopped = _dual_start_bridge()
    original_start = bridge.ros2_start
    before_full_process = (
        "hardware_ur5e_rg2_gripper"
        if stack_name == "ur5e"
        else "hardware_xarm6_driver"
    )
    injected_before = False
    injected_after = False

    def _start_with_calibration_race(process_name: str) -> None:
        nonlocal injected_before, injected_after
        error = original_start(process_name)
        if process_name == before_full_process and not injected_before:
            injected_before = True
            assert original_start("ur5e_calibration_state_publisher") is None
        if process_name == "hardware_robot_state_publisher" and not injected_after:
            injected_after = True
            assert original_start("ur5e_calibration_state_publisher") is None
        return error

    bridge.ros2_start = _start_with_calibration_race

    error = bridge.ros2_start_hardware_stack(stack_name)

    assert error is None
    assert injected_before is True
    assert injected_after is True
    assert bridge.ros2_proc_status("ur5e_calibration_state_publisher") == "stopped"
    assert stopped == [
        (
            "ur5e_calibration_state_publisher",
            "hardware_robot_state_publisher_authority",
        ),
        (
            "ur5e_calibration_state_publisher",
            "hardware_robot_state_publisher_authority",
        ),
    ]
    assert started.index(before_full_process) < started.index(
        "hardware_robot_state_publisher"
    )


@pytest.mark.parametrize("stack_name", ["xarm6", "ur5e", "dual robots"])
def test_ur5e_hardware_repair_reclaims_full_tf_authority(
    stack_name: str,
) -> None:
    bridge, _started, stopped = _dual_start_bridge()
    assert bridge.ros2_start_hardware_stack(stack_name) is None
    bridge._hardware_stack_lifecycle_state = "failed"
    bridge._hardware_stack_last_error = "repair required"
    original_start = bridge.ros2_start
    before_full_process = (
        "hardware_ur5e_rg2_gripper"
        if stack_name == "ur5e"
        else "hardware_xarm6_driver"
    )
    calibration_started = False

    def _start_with_calibration_race(process_name: str) -> None:
        nonlocal calibration_started
        error = original_start(process_name)
        if process_name == before_full_process and not calibration_started:
            calibration_started = True
            assert original_start("ur5e_calibration_state_publisher") is None
        return error

    bridge.ros2_start = _start_with_calibration_race

    error = bridge.ros2_repair_hardware_stack(stack_name)

    assert error is None
    assert calibration_started is True
    assert bridge.ros2_proc_status("ur5e_calibration_state_publisher") == "stopped"
    assert (
        "ur5e_calibration_state_publisher",
        "hardware_robot_state_publisher_authority",
    ) in stopped


@pytest.mark.parametrize("stack_name", ["ur5e", "dual robots"])
def test_ur5e_hardware_start_retries_a_transient_disconnected_tf_tree(
    stack_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _started, stopped = _dual_start_bridge()
    snapshot_errors = iter(
        (
            (
                "Could not find a connection between 'world' and 'tool0' because "
                "they are not part of the same tree"
            ),
            "",
        )
    )
    snapshots: list[str] = []

    def _snapshot(robot: str, **_kwargs: Any) -> dict[str, Any]:
        if robot == "xarm6":
            return {
                "joint_names": [f"joint{index}" for index in range(1, 7)],
                "positions": [0.0] * 6,
            }
        error = next(snapshot_errors)
        snapshots.append(error)
        return {"error": error} if error else {"pose": {}}

    bridge._snapshot_robot_waypoint = _snapshot
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    error = bridge.ros2_start_hardware_stack(stack_name)

    assert error is None
    assert len(snapshots) == 2
    assert stopped == []


def test_xarm6_hardware_partial_start_rolls_back_direct_processes() -> None:
    bridge, started, stopped = _dual_start_bridge()
    bridge._wait_for_xarm6_relayed_tf_ready = lambda **_kwargs: (
        "TF world -> link_eef unavailable"
    )

    error = bridge.ros2_start_hardware_stack("xarm6")

    assert "world -> link_eef is not ready" in str(error)
    assert started == [
        "hardware_xarm6_driver",
        "hardware_robot_state_publisher",
    ]
    assert stopped == [
        ("hardware_robot_state_publisher", "xarm6_hardware_start_rollback"),
        ("hardware_xarm6_driver", "xarm6_hardware_start_rollback"),
    ]


def test_xarm6_cartesian_frame_validation_failure_never_reaches_running() -> None:
    bridge, started, stopped = _dual_start_bridge()
    bridge._validate_xarm6_cartesian_frames = lambda **_kwargs: (
        "controller TCP reconstructed world -> link_eef differs from TF"
    )

    error = bridge.ros2_start_hardware_stack("xarm6")

    assert "Cartesian frame validation failed" in str(error)
    assert bridge._hardware_stack_lifecycle_state == "failed"
    assert started == ["hardware_xarm6_driver", "hardware_robot_state_publisher"]
    assert stopped == [
        ("hardware_robot_state_publisher", "xarm6_hardware_start_rollback"),
        ("hardware_xarm6_driver", "xarm6_hardware_start_rollback"),
    ]


def test_xarm6_cartesian_frame_validation_waits_for_first_robot_states_sample() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._hardware_stack_lifecycle_generation = 7
    bridge._hardware_cartesian_readiness = {}
    responses = iter(
        (
            (
                False,
                "Cartesian frame validation failed: "
                "xArm6 robot_states feedback has not been received",
                {},
            ),
            (
                True,
                "xArm6 Cartesian frame validation ready",
                {
                    "cartesian_jog_ready": True,
                    "cartesian_function_ready": True,
                    "diagnostics": {"controller_mode": 1},
                },
            ),
        )
    )
    calls: list[tuple[dict[str, Any], int]] = []

    def _request(
        payload: dict[str, Any],
        timeout_sec: float,
        ros_domain_id: int,
    ) -> tuple[bool, str, dict[str, Any]]:
        _ = timeout_sec
        calls.append((dict(payload), ros_domain_id))
        return next(responses)

    bridge._teleop_request_payload = _request

    error = bridge._validate_xarm6_cartesian_frames(
        ros_domain_id=42,
        timeout_sec=1.0,
    )

    assert error is None
    assert calls == [
        ({"op": "cartesian_readiness", "robot": "xarm6"}, 42),
        ({"op": "cartesian_readiness", "robot": "xarm6"}, 42),
    ]
    assert bridge._hardware_cartesian_readiness["xarm6"] == {
        "cartesian_jog_ready": True,
        "cartesian_function_ready": True,
        "generation": 7,
        "message": "xArm6 Cartesian frame validation ready",
        "diagnostics": {"controller_mode": 1},
    }


def test_ur5e_cartesian_startup_retains_world_base_failure_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_name = "hardware_ur5e_rtde_trajectory_server"
    bridge = object.__new__(SystemBridge)
    bridge._ros2_procs = {
        process_name: SimpleNamespace(pid=7201, poll=lambda: None),
    }
    bridge._hardware_stack_lifecycle_generation = 9
    bridge._hardware_cartesian_readiness = {}
    expected = {
        "x": 0.0,
        "y": 0.5,
        "z": 1.021,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    observed = {**expected, "y": 0.0}
    mount_message = (
        "Cartesian world -> base mount validation failed: observed translation "
        "differs from protected ur5e.rtde.cartesian_world_base by 0.500000 m"
    )
    bridge._ur5e_rtde_trajectory_status = lambda: {
        "process_id": 7201,
        "updated_at": time.time(),
        "cartesian_jog_ready": False,
        "cartesian_function_ready": False,
        "cartesian_world_base_ready": False,
        "relative_cartesian_action_ready": True,
        "cartesian_jog_service_ready": True,
        "relative_cartesian_action_name": (
            "/cais_ur5e_rtde_cartesian_controller/move_relative_cartesian"
        ),
        "cartesian_jog_service_name": (
            "/cais_ur5e_rtde_cartesian_controller/set_cartesian_jog"
        ),
        "cartesian_frame_validation_message": mount_message,
        "cartesian_frame_position_error_m": None,
        "cartesian_frame_orientation_error_rad": None,
        "cartesian_world_base_message": mount_message,
        "cartesian_world_base_expected": expected,
        "cartesian_world_base_observed": observed,
        "cartesian_world_base_position_error_m": 0.5,
        "cartesian_world_base_orientation_error_rad": 0.0,
    }
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )

    error = bridge._wait_for_ur5e_cartesian_frame_ready(
        process_name,
        timeout_sec=1.0,
    )

    assert mount_message in str(error)
    readiness = bridge._hardware_cartesian_readiness["ur5e"]
    assert readiness["cartesian_jog_ready"] is False
    assert readiness["cartesian_function_ready"] is False
    assert readiness["message"].count(mount_message) == 1
    assert readiness["diagnostics"] == {
        "position_error_m": None,
        "orientation_error_rad": None,
        "cartesian_world_base_ready": False,
        "cartesian_world_base_message": mount_message,
        "cartesian_world_base_expected": expected,
        "cartesian_world_base_observed": observed,
        "cartesian_world_base_position_error_m": 0.5,
        "cartesian_world_base_orientation_error_rad": 0.0,
    }


@pytest.mark.parametrize("stack_name", ["xarm6", "dual robots"])
def test_xarm6_trajectory_mode_failure_rolls_back_before_motion(
    stack_name: str,
) -> None:
    bridge, started, stopped = _dual_start_bridge()
    bridge._prepare_xarm6_trajectory_mode = lambda **_kwargs: (
        "xArm6 trajectory Mode 1 preparation did not converge"
    )

    error = bridge.ros2_start_hardware_stack(stack_name)

    assert "trajectory Mode 1 is not ready" in str(error)
    assert started == [
        "hardware_xarm6_driver",
        "hardware_robot_state_publisher",
    ]
    rollback_reason = (
        "xarm6_hardware_start_rollback"
        if stack_name == "xarm6"
        else "dual_hardware_start_rollback"
    )
    assert stopped == [
        ("hardware_robot_state_publisher", rollback_reason),
        ("hardware_xarm6_driver", rollback_reason),
    ]


def test_ur5e_hardware_partial_start_rolls_back_direct_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, started, stopped = _dual_start_bridge()
    snapshot_attempts: list[bool] = []
    bridge._snapshot_robot_waypoint = lambda *_args, **_kwargs: (
        snapshot_attempts.append(True) or {"error": "world -> tool0 unavailable"}
    )
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )

    error = bridge.ros2_start_hardware_stack("ur5e")

    assert "world -> tool0 is not ready" in str(error)
    assert started == [
        "hardware_ur5e_rtde_trajectory_server",
        "hardware_ur5e_rg2_gripper",
        "hardware_robot_state_publisher",
    ]
    assert stopped == [
        ("hardware_robot_state_publisher", "ur5e_hardware_start_rollback"),
        ("hardware_ur5e_rg2_gripper", "ur5e_hardware_start_rollback"),
        ("hardware_ur5e_rtde_trajectory_server", "ur5e_hardware_start_rollback"),
    ]
    assert len(snapshot_attempts) > 1


def test_ur5e_hardware_snapshot_wait_retains_the_last_exact_tf_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = object.__new__(SystemBridge)
    bridge.ros2_proc_status = lambda _name: "running"
    exact_error = (
        "Could not find a connection between 'world' and 'tool0' because they are "
        "not part of the same tree"
    )
    bridge._snapshot_robot_waypoint = lambda *_args, **_kwargs: {
        "error": exact_error
    }
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )

    error = bridge._wait_for_ur5e_hardware_snapshot_ready(
        driver_process_name="hardware_ur5e_rtde_trajectory_server",
        state_publisher_process_name="hardware_robot_state_publisher",
        ros_domain_id=42,
        timeout_sec=1.0,
    )

    assert error == exact_error


@pytest.mark.parametrize("stack_name", ["ur5e", "dual robots"])
def test_ur5e_hardware_start_rolls_back_if_tf_process_exits_during_retry(
    stack_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, started, stopped = _dual_start_bridge()
    original_process_status = bridge.ros2_proc_status
    snapshot_attempted = {"value": False}

    def _snapshot(robot: str, **_kwargs: Any) -> dict[str, Any]:
        if robot == "xarm6":
            return {
                "joint_names": [f"joint{index}" for index in range(1, 7)],
                "positions": [0.0] * 6,
            }
        snapshot_attempted["value"] = True
        return {"error": "world and tool0 are not part of the same tree"}

    def _process_status(process_name: str) -> str:
        if (
            process_name == "hardware_robot_state_publisher"
            and snapshot_attempted["value"]
        ):
            return "stopped"
        return original_process_status(process_name)

    bridge._snapshot_robot_waypoint = _snapshot
    bridge.ros2_proc_status = _process_status
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    error = bridge.ros2_start_hardware_stack(stack_name)

    assert "hardware_robot_state_publisher exited" in str(error)
    assert snapshot_attempted["value"] is True
    rollback_reason = (
        "ur5e_hardware_start_rollback"
        if stack_name == "ur5e"
        else "dual_hardware_start_rollback"
    )
    assert stopped == [
        (process_name, rollback_reason)
        for process_name in reversed(started)
    ]


def test_dual_hardware_start_order_has_no_gazebo_or_perception() -> None:
    bridge, started, stopped = _dual_start_bridge()

    error = bridge.ros2_start_hardware_stack("dual robots")

    assert error is None
    assert started == [
        "hardware_xarm6_driver",
        "hardware_robot_state_publisher",
        "hardware_ur5e_rtde_trajectory_server",
        "hardware_ur5e_rg2_gripper",
    ]
    assert stopped == []
    assert bridge.robot_env == "real"
    assert all("gazebo" not in name and "perception" not in name for name in started)
    assert all("moveit" not in name for name in started)


def test_dual_hardware_partial_start_rolls_back_only_started_processes() -> None:
    bridge, started, stopped = _dual_start_bridge()
    bridge._wait_for_ros_action = lambda action, **_kwargs: (
        "missing hidden action services"
        if action == "/xarm6/xarm6_traj_controller/follow_joint_trajectory"
        else None
    )

    error = bridge.ros2_start_hardware_stack("dual robots")

    assert "xarm6 trajectory controller is not ready" in str(error)
    assert started == [
        "hardware_xarm6_driver",
        "hardware_robot_state_publisher",
    ]
    assert stopped == [
        ("hardware_robot_state_publisher", "dual_hardware_start_rollback"),
        ("hardware_xarm6_driver", "dual_hardware_start_rollback"),
    ]
    assert bridge._hardware_stack_selected == "dual robots"
    assert bridge._hardware_stack_lifecycle_state == "failed"
    assert bridge._hardware_stack_last_error == error
    assert bridge.hardware_stack_status("dual robots")["lifecycle_state"] == "failed"


def test_dual_hardware_does_not_start_ur5e_until_xarm6_controllers_are_active() -> None:
    bridge, started, stopped = _dual_start_bridge()
    bridge._wait_for_ros_controllers_active = lambda *_args, **_kwargs: (
        "controllers did not become active; inactive={'xarm6_traj_controller': 'inactive'}"
    )

    error = bridge.ros2_start_hardware_stack("dual robots")

    assert "xarm6 controllers are not active" in str(error)
    assert started == [
        "hardware_xarm6_driver",
        "hardware_robot_state_publisher",
    ]
    assert "hardware_ur5e_rtde_trajectory_server" not in started
    assert stopped == [
        ("hardware_robot_state_publisher", "dual_hardware_start_rollback"),
        ("hardware_xarm6_driver", "dual_hardware_start_rollback"),
    ]


def test_controller_readiness_uses_native_list_controllers_service() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._ros2_procs = {}
    bridge.ros2_proc_status = lambda _name: "running"
    commands: list[str] = []
    response = (
        "response:\n"
        "controller_manager_msgs.srv.ListControllers_Response(controller=["
        "controller_manager_msgs.msg.ControllerState("
        "name='joint_state_broadcaster', state='active', type=''), "
        "controller_manager_msgs.msg.ControllerState("
        "name='xarm6_traj_controller', state='active', type='')])\n"
    )

    def _command_output(command: str, **_kwargs: Any) -> tuple[bool, str]:
        commands.append(command)
        return True, response

    bridge._ros2_command_output = _command_output

    error = bridge._wait_for_ros_controllers_active(
        "/xarm6/controller_manager",
        ("joint_state_broadcaster", "xarm6_traj_controller"),
        timeout_sec=1.0,
        process_name="hardware_xarm6_driver",
    )

    assert error is None
    assert commands == [
        "ros2 service call /xarm6/controller_manager/list_controllers "
        "controller_manager_msgs/srv/ListControllers '{}'"
    ]
    assert all("ros2 control" not in command for command in commands)


def test_dual_hardware_requires_exact_namespaced_xarm6_joint_feedback() -> None:
    bridge, started, stopped = _dual_start_bridge()
    bridge._xarm6_hardware_feedback_error = lambda **_kwargs: (
        "xArm6 hardware snapshot is missing required joints: "
        "missing=['joint4', 'joint5', 'joint6']"
    )

    error = bridge.ros2_start_hardware_stack("dual robots")

    assert "xarm6 hardware feedback is not ready" in str(error)
    assert "joint4" in str(error)
    assert started == [
        "hardware_xarm6_driver",
        "hardware_robot_state_publisher",
    ]
    assert stopped == [
        ("hardware_robot_state_publisher", "dual_hardware_start_rollback"),
        ("hardware_xarm6_driver", "dual_hardware_start_rollback"),
    ]


@pytest.mark.parametrize("stack_name", ["xarm6", "dual robots"])
def test_hardware_start_gates_on_xarm6_broadcaster_topic(stack_name: str) -> None:
    bridge, _started, _stopped = _dual_start_bridge()
    observed_calls: list[tuple[str, int]] = []

    def _feedback_error(*, process_name: str, ros_domain_id: int) -> None:
        observed_calls.append((process_name, ros_domain_id))
        return None

    bridge._xarm6_hardware_feedback_error = _feedback_error

    assert bridge.ros2_start_hardware_stack(stack_name) is None
    assert observed_calls == [("hardware_xarm6_driver", 0)]


def test_xarm6_feedback_gate_uses_multi_topic_snapshot_worker() -> None:
    bridge = object.__new__(SystemBridge)
    bridge.ros2_proc_status = lambda _name: "running"
    snapshots: list[tuple[str, dict[str, Any]]] = []

    def _snapshot(robot: str, **kwargs: Any) -> dict[str, Any]:
        snapshots.append((robot, kwargs))
        return {
            "joint_names": [f"joint{index}" for index in range(1, 7)],
            "positions": [0.0] * 6,
        }

    bridge._snapshot_robot_waypoint = _snapshot

    error = bridge._xarm6_hardware_feedback_error(
        process_name="hardware_xarm6_driver",
        ros_domain_id=42,
    )

    assert error is None
    assert snapshots == [
        (
            "xarm6",
            {
                "source": "hardware",
                "hardware_domain_id": 42,
                "include_world_tool_pose": False,
            },
        )
    ]


def test_xarm6_tf_readiness_requires_exact_root_joint_names_and_finite_positions() -> None:
    module = _digital_twin_sync_module()
    exact_names = [f"joint{index}" for index in range(1, 7)]

    positions, problem = module._xarm6_relayed_positions(
        [*exact_names, "drive_joint"],
        [0.0] * 7,
    )
    assert positions == [0.0] * 6
    assert problem == ""

    positions, problem = module._xarm6_relayed_positions(
        [f"xarm6_joint{index}" for index in range(1, 7)],
        [0.0] * 6,
    )
    assert positions is None
    assert "missing=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']" in problem

    positions, problem = module._xarm6_relayed_positions(
        exact_names,
        [0.0, 0.0, float("nan"), 0.0, 0.0, 0.0],
    )
    assert positions is None
    assert "non-finite" in problem
    assert "joint3" in problem


def test_xarm6_tf_readiness_probe_monitors_driver_and_state_publisher() -> None:
    bridge = object.__new__(SystemBridge)
    observed: list[tuple[list[str], float, tuple[str, ...]]] = []

    def _run(
        args: list[str],
        timeout_sec: float,
        *,
        monitored_processes: tuple[str, ...],
    ) -> dict[str, Any]:
        observed.append((list(args), timeout_sec, monitored_processes))
        return {"success": True, "message": "ready"}

    bridge._run_digital_twin_sync = _run

    assert bridge._wait_for_xarm6_relayed_tf_ready(
        driver_process_name="hardware_xarm6_driver",
        state_publisher_process_name="hardware_robot_state_publisher",
        ros_domain_id=42,
    ) is None
    args, timeout, monitored = observed[0]
    assert args[args.index("--mode") + 1] == "xarm6-tf-readiness"
    assert args[args.index("--tf-readiness-timeout-sec") + 1] == "20.0"
    assert "/xarm6/xarm/joint_states" not in args
    assert timeout == 24.0
    assert monitored == (
        "hardware_xarm6_driver",
        "hardware_robot_state_publisher",
    )


def test_xarm6_tf_readiness_probe_stops_when_an_owner_exits(monkeypatch) -> None:
    bridge = object.__new__(SystemBridge)
    bridge.ros2_proc_status = lambda name: (
        "stopped" if name == "hardware_xarm6_driver" else "running"
    )

    class _ProbeProcess:
        pid = 9012
        returncode = None

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def communicate(timeout: float) -> tuple[str, str]:
            assert timeout == 1.0
            return "", ""

    monkeypatch.setattr(
        "cais_spade_llm.ui.bridge.subprocess.Popen",
        lambda *_args, **_kwargs: _ProbeProcess(),
    )
    monkeypatch.setattr("cais_spade_llm.ui.bridge.os.getpgid", lambda _pid: 9012)
    monkeypatch.setattr("cais_spade_llm.ui.bridge.os.killpg", lambda *_args: None)

    result = bridge._run_digital_twin_sync(
        [
            "--mode",
            "xarm6-tf-readiness",
            "--robot",
            "xarm6",
            "--gazebo-domain-id",
            "0",
            "--hardware-domain-id",
            "0",
        ],
        timeout_sec=24.0,
        monitored_processes=(
            "hardware_xarm6_driver",
            "hardware_robot_state_publisher",
        ),
    )

    assert result["success"] is False
    assert result["message"] == (
        "hardware_xarm6_driver exited before xArm6 TF readiness completed"
    )


def test_dual_hardware_concurrent_start_is_rejected_before_duplicate_launch() -> None:
    bridge, started, _stopped = _dual_start_bridge()
    connection_probe_entered = threading.Event()
    release_connection_probe = threading.Event()
    original_connection_statuses = bridge.hardware_connection_statuses

    def _blocked_connection_statuses(*, force: bool = False) -> dict[str, Any]:
        connection_probe_entered.set()
        assert release_connection_probe.wait(timeout=2.0)
        return original_connection_statuses(force=force)

    bridge.hardware_connection_statuses = _blocked_connection_statuses
    first_result: list[str | None] = []
    first_start = threading.Thread(
        target=lambda: first_result.append(
            bridge.ros2_start_hardware_stack("dual robots")
        )
    )
    first_start.start()
    assert connection_probe_entered.wait(timeout=2.0)
    try:
        assert bridge._hardware_stack_selected == "dual robots"
        assert bridge._hardware_stack_lifecycle_state == "starting"
        assert bridge.hardware_stack_status("dual robots")["lifecycle_state"] == "starting"
        assert bridge.hardware_stack_status("dual robots")["overall"] == "stopped"
        second_result = bridge.ros2_start_hardware_stack("dual robots")
    finally:
        release_connection_probe.set()
        first_start.join(timeout=2.0)

    assert not first_start.is_alive()
    assert first_result == [None]
    assert second_result == "Hardware Stack start or stop is already running."
    assert started == [
        "hardware_xarm6_driver",
        "hardware_robot_state_publisher",
        "hardware_ur5e_rtde_trajectory_server",
        "hardware_ur5e_rg2_gripper",
    ]
    assert bridge._hardware_stack_lifecycle_state == "running"


def test_post_green_component_exit_marks_failed_and_preserves_survivors() -> None:
    class _Process:
        def __init__(self, return_code: int | None) -> None:
            self.return_code = return_code
            self.pid = 1234

        def poll(self) -> int | None:
            return self.return_code

    bridge = object.__new__(SystemBridge)
    bridge._ros2_procs = {
        "hardware_xarm6_driver": _Process(17),
        "hardware_ur5e_rtde_trajectory_server": _Process(None),
    }
    bridge._ros2_intentional_stops = set()
    bridge._hardware_stack_selected = "dual robots"
    bridge._hardware_stack_lifecycle_state = "running"
    bridge._hardware_stack_last_error = ""
    bridge._hardware_stack_failed_process = ""
    bridge._hardware_stack_failed_return_code = None
    bridge._hardware_stack_process_log_path = ""
    bridge._cleanup_exited_ui_process = lambda *_args: None

    status = bridge.ros2_proc_status("hardware_xarm6_driver")

    assert status == "stopped"
    assert bridge._hardware_stack_lifecycle_state == "failed"
    assert bridge._hardware_stack_failed_process == "hardware_xarm6_driver"
    assert bridge._hardware_stack_failed_return_code == 17
    assert "return code 17" in bridge._hardware_stack_last_error
    assert bridge._hardware_stack_process_log_path.endswith(
        "hardware_xarm6_driver.log"
    )
    assert "hardware_ur5e_rtde_trajectory_server" in bridge._ros2_procs


def test_post_green_rtde_transport_loss_marks_failed_and_preserves_processes(
    tmp_path: Path,
) -> None:
    rtde_process = SimpleNamespace(pid=7201, poll=lambda: None)
    bridge = object.__new__(SystemBridge)
    bridge._hardware_stack_selected = "dual robots"
    bridge._hardware_stack_lifecycle_state = "running"
    bridge._hardware_stack_last_error = ""
    bridge._hardware_stack_failed_process = ""
    bridge._hardware_stack_failed_return_code = None
    bridge._hardware_stack_process_log_path = ""
    bridge._ros2_procs = {
        "hardware_xarm6_driver": SimpleNamespace(pid=7200, poll=lambda: None),
        "hardware_ur5e_rtde_trajectory_server": rtde_process,
        "hardware_ur5e_rg2_gripper": SimpleNamespace(pid=7202, poll=lambda: None),
        "hardware_robot_state_publisher": SimpleNamespace(pid=7203, poll=lambda: None),
    }
    bridge._ros2_process_log_paths = {
        "hardware_ur5e_rtde_trajectory_server": tmp_path / "rtde-run.log"
    }
    bridge._ros2_intentional_stops = set()
    bridge.ros2_proc_status = lambda name: (
        "running" if name in bridge._ros2_procs else "stopped"
    )
    bridge._ur5e_rtde_trajectory_status = lambda: {
        "state": "failed",
        "process_id": 7201,
        "updated_at": time.time(),
        "rtde_reset_required": True,
        "blocked_reason": "RTDE receive transport failed. Repair Hardware Stack.",
    }

    status = bridge.hardware_stack_status("dual robots")

    assert status["lifecycle_state"] == "failed"
    assert status["overall"] == "running"
    assert status["failed_process"] == "hardware_ur5e_rtde_trajectory_server"
    assert status["process_return_code"] is None
    assert status["process_log_path"] == str(tmp_path / "rtde-run.log")
    assert len(bridge._ros2_procs) == 4


def test_rtde_status_staleness_requires_six_seconds_before_latching_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _started, _stopped = _dual_start_bridge()
    assert bridge.ros2_start_hardware_stack("ur5e") is None
    rtde_process = bridge._ros2_procs["hardware_ur5e_rtde_trajectory_server"]
    now = 100.0
    rtde_status = {
        "state": "ready",
        "process_id": rtde_process.pid,
        "updated_at": now - 3.1,
        "rtde_reset_required": False,
        "joint_states_fresh": True,
    }
    bridge._ur5e_rtde_trajectory_status = lambda: dict(rtde_status)
    monkeypatch.setattr(time, "time", lambda: now)

    transient = bridge.hardware_stack_status("ur5e")

    assert transient["lifecycle_state"] == "running"
    assert transient["last_error"] == ""

    rtde_status["updated_at"] = now - 6.1
    sustained = bridge.hardware_stack_status("ur5e")

    assert sustained["lifecycle_state"] == "failed"
    assert sustained["failed_process"] == "hardware_ur5e_rtde_trajectory_server"
    assert sustained["last_error"] == (
        "UR5e RTDE status stopped advancing. Repair Hardware Stack."
    )


def test_failed_dual_hardware_stack_supports_one_serialized_repair() -> None:
    bridge, started, stopped = _dual_start_bridge()
    assert bridge.ros2_start_hardware_stack("dual robots") is None
    bridge._hardware_stack_lifecycle_state = "failed"
    bridge._hardware_stack_last_error = "hardware_xarm6_driver exited"

    error = bridge.ros2_repair_hardware_stack("dual robots")

    assert error is None
    assert bridge._hardware_stack_selected == "dual robots"
    assert bridge._hardware_stack_lifecycle_state == "running"
    assert bridge._hardware_stack_last_error == ""
    assert stopped == [
        ("hardware_robot_state_publisher", "explicit_stop"),
        ("hardware_ur5e_rg2_gripper", "explicit_stop"),
        ("hardware_ur5e_rtde_trajectory_server", "explicit_stop"),
        ("hardware_xarm6_driver", "explicit_stop"),
    ]
    assert started == [
        "hardware_xarm6_driver",
        "hardware_robot_state_publisher",
        "hardware_ur5e_rtde_trajectory_server",
        "hardware_ur5e_rg2_gripper",
    ] * 2


def test_stale_generation_failure_cannot_relatch_repaired_stack() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._hardware_stack_lifecycle_lock = threading.Lock()
    bridge._hardware_stack_lifecycle_generation = 8
    bridge._hardware_stack_selected = "dual robots"
    bridge._hardware_stack_lifecycle_state = "running"
    bridge._hardware_stack_last_error = ""
    bridge._hardware_stack_process_generations = {
        "hardware_ur5e_rtde_trajectory_server": 8
    }
    bridge._ros2_procs = {
        "hardware_ur5e_rtde_trajectory_server": SimpleNamespace(
            pid=8200,
            poll=lambda: None,
        )
    }

    marked = bridge._mark_hardware_stack_failed_if_current(
        message="old RTDE poll reported a transport failure",
        process_name="hardware_ur5e_rtde_trajectory_server",
        return_code=None,
        log_path=Path("/tmp/old-rtde.log"),
        expected_generation=7,
        expected_process_id=7100,
    )

    assert marked is False
    assert bridge._hardware_stack_lifecycle_state == "running"
    assert bridge._hardware_stack_last_error == ""


def test_status_poll_started_before_repair_cannot_relatch_new_rtde_process() -> None:
    old_status_requested = threading.Event()
    release_old_status = threading.Event()
    bridge = object.__new__(SystemBridge)
    bridge._hardware_stack_lifecycle_lock = threading.Lock()
    bridge._hardware_stack_lifecycle_generation = 7
    bridge._hardware_stack_selected = "dual robots"
    bridge._hardware_stack_lifecycle_state = "running"
    bridge._hardware_stack_last_error = ""
    bridge._hardware_stack_failed_process = ""
    bridge._hardware_stack_failed_return_code = None
    bridge._hardware_stack_process_log_path = ""
    old_process = SimpleNamespace(pid=7100, poll=lambda: None)
    bridge._ros2_procs = {
        process_name: SimpleNamespace(pid=7100 + index, poll=lambda: None)
        for index, process_name in enumerate(
            SystemBridge._HARDWARE_STACKS["dual robots"]
        )
    }
    bridge._ros2_procs["hardware_ur5e_rtde_trajectory_server"] = old_process
    bridge._hardware_stack_process_generations = {
        process_name: 7 for process_name in bridge._ros2_procs
    }
    bridge.ros2_proc_status = lambda name: (
        "running" if name in bridge._ros2_procs else "stopped"
    )

    def _old_rtde_status() -> dict[str, Any]:
        old_status_requested.set()
        assert release_old_status.wait(timeout=2.0)
        return {
            "state": "failed",
            "process_id": 7100,
            "updated_at": time.time(),
            "rtde_reset_required": True,
            "blocked_reason": "old transport failure",
        }

    bridge._ur5e_rtde_trajectory_status = _old_rtde_status
    results: list[dict[str, Any]] = []
    worker = threading.Thread(
        target=lambda: results.append(bridge.hardware_stack_status("dual robots"))
    )
    worker.start()
    assert old_status_requested.wait(timeout=2.0)
    bridge._hardware_stack_lifecycle_generation = 8
    bridge._hardware_stack_process_generations[
        "hardware_ur5e_rtde_trajectory_server"
    ] = 8
    bridge._ros2_procs["hardware_ur5e_rtde_trajectory_server"] = SimpleNamespace(
        pid=8200,
        poll=lambda: None,
    )
    release_old_status.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert results[0]["lifecycle_generation"] == 8
    assert bridge._hardware_stack_lifecycle_state == "running"
    assert bridge._hardware_stack_last_error == ""


def test_successful_repair_clears_generic_ur5e_uncertainty_after_fresh_readiness() -> None:
    bridge, _started, _stopped = _dual_start_bridge()
    assert bridge.ros2_start_hardware_stack("dual robots") is None
    bridge._hardware_stack_lifecycle_state = "failed"
    bridge._hardware_stack_last_error = "Cartesian Smooth Hold watchdog expired"
    bridge._ur5e_cartesian_jog_state_uncertain = True
    bridge._ur5e_cartesian_jog_state_uncertain_reason = "jogStop was not confirmed"
    bridge._ur5e_robot_function_state_uncertain = True
    bridge._ur5e_robot_function_state_uncertain_reason = "pick_approach failed"

    error = bridge.ros2_repair_hardware_stack("dual robots")

    assert error is None
    assert bridge._hardware_stack_lifecycle_state == "running"
    assert bridge._ur5e_cartesian_jog_state_uncertain is False
    assert bridge._ur5e_cartesian_jog_state_uncertain_reason == ""
    assert bridge._ur5e_robot_function_state_uncertain is False
    assert bridge._ur5e_robot_function_state_uncertain_reason == ""
    assert "cleared at generation" in (
        bridge._hardware_stack_cartesian_jog_reset_results["ur5e"]
    )
    assert set(bridge._hardware_stack_validated_process_pids) == set(
        SystemBridge._HARDWARE_STACKS["dual robots"]
    )


def test_hardware_stack_repair_is_rejected_during_motion() -> None:
    bridge, started, stopped = _dual_start_bridge()
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock.acquire()
    bridge._ur5e_robot_function_execution_active = "pick_approach"
    try:
        error = bridge.ros2_repair_hardware_stack("dual robots")
    finally:
        bridge._ur5e_robot_function_execution_lock.release()

    assert error == (
        "Cannot repair Hardware Stack while motion is active: pick_approach."
    )
    assert started == []
    assert stopped == []


def test_hardware_stack_start_button_enters_loading_before_scheduling() -> None:
    source = Path("cais_spade_llm/ui/pages/control.py").read_text(encoding="utf-8")
    row_source = source.split("def _hardware_stack_row(", maxsplit=1)[1].split(
        "# =====================================================================",
        maxsplit=1,
    )[0]

    assert row_source.index('start_button.props("loading")') < row_source.index(
        "asyncio.create_task(_start_async())"
    )
    assert 'if operation_state["busy"]:' in row_source


def test_stopped_hardware_rows_are_not_coupled_to_move_insert_repair_state() -> None:
    source = Path("cais_spade_llm/ui/pages/control.py").read_text(encoding="utf-8")
    launch_source = source.split("def _launch_section(", maxsplit=1)[1].split(
        "def _proc_row(",
        maxsplit=1,
    )[0]
    row_source = source.split("def _hardware_stack_row(", maxsplit=1)[1].split(
        "# =====================================================================",
        maxsplit=1,
    )[0]

    for field_name in (
        "hardware_stack_repair_required",
        "hardware_stack_repair_reason",
        "hardware_stack_operation_blocked_reason",
    ):
        assert field_name not in launch_source
    assert "hardware_statuses = {" in launch_source
    assert "signature, statuses, hardware_statuses = await asyncio.to_thread(" in (
        launch_source
    )
    assert "_launch_snapshot" in launch_source

    assert 'repair_needed = lifecycle_state == "failed"' in row_source
    assert "hardware_stack_repair_required" not in row_source
    assert "hardware_stack_repair_reason" not in row_source
    assert "if repair_needed" in row_source
    assert "bridge.ros2_repair_hardware_stack" in row_source
    assert '"Repair Hardware Stack" if repair_needed else "Start"' in row_source
    assert "stop_disabled = repair_needed" in row_source
    assert 'ui.label(blocked_reason).classes("text-xs text-amber-700")' in (
        row_source
    )


def test_hardware_connectivity_reports_wrong_same_subnet_route() -> None:
    bridge = object.__new__(SystemBridge)
    bridge.hardware_ips = {
        "xarm6": "192.168.1.240",
        "ur5e": "192.168.1.172",
    }
    bridge._hw_ping_cache = {}
    bridge._hw_ping_last_ts = 0.0
    bridge._ping_once = lambda ip, **kwargs: (
        (True, 3.2, None)
        if ip == "192.168.1.240" and kwargs.get("interface") == "eth1"
        else (True, 0.3, None)
        if ip == "192.168.1.172"
        else (False, None, "timeout")
    )
    bridge._hardware_route_conflict = lambda ip: (
        ("eth2", "192.168.1.220", "eth1")
        if ip == "192.168.1.240"
        else None
    )
    bridge._tcp_probe_once = lambda *_args, **_kwargs: pytest.fail(
        "A confirmed route conflict must not be hidden by a TCP fallback"
    )

    statuses = bridge.hardware_connection_statuses(force=True)

    assert statuses["xarm6"]["reachable"] is False
    assert statuses["xarm6"]["probe"] == "route conflict"
    assert "Linux selects eth2 src 192.168.1.220" in statuses["xarm6"]["message"]
    assert "replies on eth1" in statuses["xarm6"]["message"]
    assert statuses["ur5e"]["reachable"] is True


def test_hardware_ping_does_not_kill_ping_before_its_wait_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def _run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        observed["command"] = command
        observed["timeout"] = kwargs["timeout"]
        return SimpleNamespace(returncode=0, stdout="time=0.300 ms\n")

    monkeypatch.setattr("cais_spade_llm.ui.bridge.subprocess.run", _run)

    reachable, latency_ms, error = SystemBridge._ping_once(
        "192.168.1.172",
        timeout_sec=0.35,
    )

    assert reachable is True
    assert latency_ms == pytest.approx(0.3)
    assert error is None
    assert observed["command"] == [
        "ping",
        "-c",
        "1",
        "-W",
        "1",
        "192.168.1.172",
    ]
    assert observed["timeout"] >= 1.25


def test_hardware_connectivity_accepts_robot_tcp_when_icmp_is_blocked() -> None:
    bridge = object.__new__(SystemBridge)
    bridge.hardware_ips = {
        "xarm6": "192.168.1.240",
        "ur5e": "192.168.1.172",
    }
    bridge._hw_ping_cache = {}
    bridge._hw_ping_last_ts = 0.0
    bridge._ping_once = lambda *_args, **_kwargs: (False, None, "timeout")
    bridge._hardware_route_conflict = lambda _ip: None
    bridge._tcp_probe_once = lambda _ip, port, **_kwargs: (
        True,
        float(port) / 1000.0,
        None,
    )

    statuses = bridge.hardware_connection_statuses(force=True)

    assert statuses["xarm6"]["reachable"] is True
    assert statuses["xarm6"]["probe"] == "TCP 502"
    assert statuses["ur5e"]["reachable"] is True
    assert statuses["ur5e"]["probe"] == "TCP 30004"


def test_cleanup_removes_stale_ur5e_rtde_and_dual_moveit_processes() -> None:
    bridge_source = Path("cais_spade_llm/ui/bridge.py").read_text(encoding="utf-8")
    ui_main_source = Path("cais_spade_llm/ui_main.py").read_text(encoding="utf-8")
    explicit_cleanup = bridge_source.split(
        "def ros2_cleanup_processes(",
        maxsplit=1,
    )[1].split("def _cleanup_shm_and_tmp(", maxsplit=1)[0]
    startup_cleanup = ui_main_source.split("_KILL_CMDS:", maxsplit=1)[1].split(
        "def _kill_stale_ros2_processes(",
        maxsplit=1,
    )[0]

    for process_pattern in (
        "dual_robots_hardware_moveit.launch.py",
        "ur5e_rtde_trajectory_server.py",
    ):
        assert process_pattern in explicit_cleanup
        assert process_pattern in startup_cleanup


def test_matching_hardware_stack_bypasses_every_digital_twin_gate() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._active_normal_hardware_stack = lambda: "dual robots"
    bridge.hardware_stack_status = lambda _stack: {"overall": "running"}
    bridge.digital_twin_statuses = lambda: pytest.fail(
        "Hardware Stack readiness must not inspect a Digital Twin"
    )

    error = bridge._digital_twin_robot_function_target_error(
        "dual robots",
        {"hardware": ("xarm6", "ur5e")},
    )

    assert error == ""


def test_xarm6_physical_capture_uses_world_to_link_eef() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._digital_twin_target = lambda _target: {"hardware": ("xarm6",)}
    bridge._robot_function_hardware_domain_id = lambda *_args: 0
    bridge._snapshot_robot_waypoint = lambda *_args, **_kwargs: {
        "positions": [0.1] * 6,
        "joint_names": [f"joint{index}" for index in range(1, 7)],
        "pose": {
            "frame_id": "world",
            "child_frame_id": "link_eef",
            "x": 0.1,
            "y": -0.2,
            "z": 1.1,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
    }

    result = bridge._robot_function_capture_snapshot("xarm only", "xarm6")

    assert result["success"] is True
    assert result["joint_states_fresh"] is True
    assert result["world_link_eef_ready"] is True
    assert result["waypoint"]["pose"]["child_frame_id"] == "link_eef"


def test_xarm6_uncertainty_clears_only_after_completed_move_home() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._xarm6_robot_function_state_uncertain = True
    agent = object()

    bridge._record_xarm6_robot_function_result(
        agent,
        "pick_approach",
        {"status": "completed"},
    )
    assert bridge._xarm6_robot_function_state_uncertain is True

    bridge._record_xarm6_robot_function_result(
        agent,
        "move_home",
        {"status": "completed"},
    )
    assert bridge._xarm6_robot_function_state_uncertain is False

    bridge._record_xarm6_robot_function_result(
        agent,
        "move_home",
        {"status": "failed", "content": "Mode 1 recovery failed"},
    )
    assert bridge._xarm6_robot_function_state_uncertain is True
    assert bridge._xarm6_robot_function_state_uncertain_reason == (
        "Mode 1 recovery failed"
    )


def test_xarm6_cartesian_preflight_restores_mode_one_and_refreshes_frames() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._xarm6_robot_function_state_uncertain = False
    bridge.teleop_target = lambda _robot, _op: {
        "ready": True,
        "warning": "",
        "environment": "real",
        "ros_domain_id": 42,
    }
    calls: list[tuple[str, int]] = []
    bridge._prepare_xarm6_trajectory_mode = lambda *, ros_domain_id: (
        calls.append(("mode", ros_domain_id)) or None
    )
    bridge._validate_xarm6_cartesian_frames = lambda *, ros_domain_id: (
        calls.append(("frames", ros_domain_id)) or None
    )
    bridge._hardware_cartesian_readiness = {
        "xarm6": {
            "cartesian_jog_ready": True,
            "cartesian_function_ready": True,
            "message": "ready",
        }
    }

    target = bridge._teleop_preflight("xarm6", "cartesian")

    assert target["ready"] is True
    assert calls == []


class _ImmediateFuture:
    def __init__(self, value: Any) -> None:
        self.value = value


def _xarm6_feedback_controller(
    angles: list[float] | None,
    *,
    age_sec: float = 0.0,
    mode: int = 1,
    state: int = 0,
) -> XArm6HardwareController:
    controller = object.__new__(XArm6HardwareController)
    controller.arm_joint_names = list(XARM6_HARDWARE_JOINT_NAMES)
    controller._xarm6_robot_state_lock = threading.Lock()
    controller._xarm6_robot_state = (
        None
        if angles is None
        else SimpleNamespace(
            angle=list(angles),
            mode=mode,
            state=state,
        )
    )
    controller._xarm6_robot_state_received_monotonic = time.monotonic() - age_sec
    controller._xarm6_set_mode_service = "/xarm6/xarm/set_mode"
    controller._xarm6_set_state_service = "/xarm6/xarm/set_state"
    controller._xarm6_set_mode_client = SimpleNamespace(
        wait_for_service=lambda **_kwargs: True
    )
    controller._xarm6_set_state_client = SimpleNamespace(
        wait_for_service=lambda **_kwargs: True
    )
    controller._xarm6_controller_list_service = (
        "/xarm6/controller_manager/list_controllers"
    )
    controller._xarm6_controller_list_client = SimpleNamespace(
        wait_for_service=lambda **_kwargs: True
    )
    controller._xarm6_joint_duration_scale = 4.0
    controller._SetInt16 = SimpleNamespace(
        Request=lambda: SimpleNamespace(data=None)
    )
    controller._ListControllers = SimpleNamespace(
        Request=lambda: SimpleNamespace()
    )
    controller._last_failure_message = ""
    return controller


def test_xarm6_physical_and_gazebo_joint_names_remain_exact() -> None:
    assert XARM6_HARDWARE_JOINT_NAMES == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    assert XARM6_JOINT_NAMES == [
        "xarm6_joint1",
        "xarm6_joint2",
        "xarm6_joint3",
        "xarm6_joint4",
        "xarm6_joint5",
        "xarm6_joint6",
    ]


def test_xarm6_physical_joint_feedback_uses_robot_states_angles() -> None:
    angles = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    controller = _xarm6_feedback_controller(angles)
    controller._joint_positions = {name: 9.0 for name in XARM6_JOINT_NAMES}

    positions, missing = controller._get_arm_joint_positions(timeout_sec=0.0)

    assert positions == angles
    assert missing == []
    assert controller._last_failure_message == ""


@pytest.mark.parametrize(
    ("angles", "age_sec", "expected_message"),
    [
        (None, 0.0, "has not been received"),
        ([0.0] * 6, 2.1, "is stale"),
        ([0.0] * 5, 0.0, "angle has 5 values; expected at least 6"),
        ([0.0, 0.0, float("nan"), 0.0, 0.0, 0.0], 0.0, "non-finite"),
    ],
)
def test_xarm6_physical_joint_feedback_fails_closed(
    angles: list[float] | None,
    age_sec: float,
    expected_message: str,
) -> None:
    controller = _xarm6_feedback_controller(angles, age_sec=age_sec)

    positions, missing = controller._get_arm_joint_positions(timeout_sec=0.0)

    assert positions is None
    assert missing == XARM6_HARDWARE_JOINT_NAMES
    assert expected_message in controller._last_failure_message


def test_xarm6_wait_for_services_uses_robot_states_without_joint_states() -> None:
    controller = _xarm6_feedback_controller([0.0] * 6)
    controller._services_ready = False
    controller._xarm6_clients_ready = False
    controller.init = lambda: True
    arm_client = SimpleNamespace(server_is_ready=lambda: True)
    gripper_client = SimpleNamespace(server_is_ready=lambda: True)
    controller._xarm6_hardware_trajectory_actions = ["arm_action"]
    controller._xarm6_hardware_trajectory_clients = [("arm_action", arm_client)]
    controller._xarm6_gripper_actions = ["gripper_action"]
    controller._xarm6_gripper_clients = [("gripper_action", gripper_client)]
    controller._xarm6_hardware_cartesian_service = "/xarm6/xarm/set_position"
    controller._xarm6_hardware_cartesian_client = SimpleNamespace(
        wait_for_service=lambda **_kwargs: True
    )
    controller._joint_positions = {}

    assert controller.wait_for_services(timeout_sec=0.0) is True
    assert controller._xarm6_hardware_trajectory_action == "arm_action"
    assert controller._xarm6_gripper_action == "gripper_action"


def test_xarm6_wait_for_services_rechecks_cached_robot_states_freshness() -> None:
    controller = _xarm6_feedback_controller([0.0] * 6, age_sec=2.1)
    controller._services_ready = True
    controller._xarm6_clients_ready = True
    controller.init = lambda: True

    assert controller.wait_for_services(timeout_sec=0.0) is False
    assert "robot_states joint feedback is stale" in controller._last_failure_message


def test_xarm6_action_requires_terminal_success_and_joint_convergence() -> None:
    start_positions = [0.01, -0.02, 0.03, -0.04, 0.05, -0.06]
    controller = _xarm6_feedback_controller(start_positions)
    controller._wait_for_xarm6_trajectory_controller_state = (
        lambda state, **_kwargs: (state == "active", "OK")
    )
    controller._xarm6_hardware_trajectory_action = (
        "/xarm6/xarm6_traj_controller/follow_joint_trajectory"
    )
    controller.wait_for_services = lambda: True
    controller._JointTrajectoryPoint = lambda: SimpleNamespace(
        positions=[],
        time_from_start=None,
    )
    controller._Duration = lambda **kwargs: SimpleNamespace(**kwargs)

    class _Goal:
        def __init__(self) -> None:
            self.trajectory = SimpleNamespace(joint_names=[], points=[])

    controller._FollowJointTrajectory = SimpleNamespace(Goal=_Goal)
    result_future = _ImmediateFuture(
        SimpleNamespace(
            status=4,
            result=SimpleNamespace(error_code=0, error_string=""),
        )
    )
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: result_future,
    )
    send_future = _ImmediateFuture(goal_handle)
    sent: list[Any] = []
    controller._xarm6_hardware_trajectory_client = SimpleNamespace(
        send_goal_async=lambda goal: sent.append(goal) or send_future
    )
    controller._wait_future = lambda future, **_kwargs: future.value
    converged: list[list[float]] = []
    controller._wait_for_arm_joint_targets = lambda targets, **_kwargs: (
        converged.append(list(targets)) or True
    )

    assert controller._command_xarm6_hardware_trajectory_action(
        [0.1] * 6,
        duration_sec=1.0,
        label="move_home",
    ) is True
    assert len(sent) == 1
    assert sent[0].trajectory.joint_names == XARM6_HARDWARE_JOINT_NAMES
    assert sent[0].trajectory.points[0].positions == start_positions
    assert sent[0].trajectory.points[1].time_from_start.sec == 4
    assert converged == [[0.1] * 6]


def test_xarm6_mode_one_is_restored_after_joint_action() -> None:
    controller = _xarm6_feedback_controller([0.0] * 6, mode=0, state=2)
    calls: list[tuple[str, int]] = []

    def _call(service_name: str, value: int) -> _ImmediateFuture:
        calls.append((service_name, value))
        if service_name == "/xarm6/xarm/set_state":
            controller._xarm6_robot_state.mode = 1
            controller._xarm6_robot_state.state = 0
            controller._xarm6_robot_state_received_monotonic = time.monotonic()
        return _ImmediateFuture(SimpleNamespace(ret=0, message=""))

    controller._xarm6_set_mode_client = SimpleNamespace(
        call_async=lambda request: _call("/xarm6/xarm/set_mode", request.data)
    )
    controller._xarm6_set_state_client = SimpleNamespace(
        call_async=lambda request: _call("/xarm6/xarm/set_state", request.data)
    )
    controller._wait_future = lambda future, **_kwargs: future.value
    controller._wait_for_xarm6_trajectory_controller_state = (
        lambda state, **_kwargs: (state == "active", "OK")
    )

    assert controller._prepare_xarm6_mode_one(timeout_sec=0.2) is True
    assert calls == [
        ("/xarm6/xarm/set_mode", 1),
        ("/xarm6/xarm/set_state", 0),
    ]


def test_xarm6_cartesian_handoff_releases_and_restores_trajectory_controller() -> None:
    controller = _xarm6_feedback_controller([0.0] * 6)
    calls: list[tuple[str, int | str]] = []
    controller._set_xarm6_control_value = lambda service_name, _client, value: (
        calls.append((service_name, value)) or (True, "OK")
    )
    controller._wait_for_xarm6_mode = lambda mode: (
        calls.append(("wait_mode", mode)) or (True, "OK")
    )
    controller._wait_for_xarm6_trajectory_controller_state = lambda state: (
        calls.append(("wait_controller", state)) or (True, "OK")
    )

    assert controller._prepare_xarm6_firmware_cartesian_mode() == (
        True,
        "xArm6 firmware Cartesian Mode 0 ready",
    )
    assert controller._restore_xarm6_trajectory_control() == (
        True,
        "xArm6 trajectory controller Mode 1 restored",
    )
    assert calls == [
        ("/xarm6/xarm/set_mode", 0),
        ("/xarm6/xarm/set_state", 0),
        ("wait_mode", 0),
        ("wait_controller", "inactive"),
        ("/xarm6/xarm/set_mode", 1),
        ("/xarm6/xarm/set_state", 0),
        ("wait_mode", 1),
        ("wait_controller", "active"),
    ]


def test_xarm6_function_controller_observes_ufactory_controller_transition() -> None:
    controller = _xarm6_feedback_controller([0.0] * 6)
    states = iter(("inactive", "active"))
    observed_states: list[str] = []

    def _list_controllers(_request: Any) -> _ImmediateFuture:
        state = next(states)
        observed_states.append(state)
        return _ImmediateFuture(
            SimpleNamespace(
                controller=[
                    SimpleNamespace(name="xarm6_traj_controller", state=state)
                ]
            )
        )

    controller._xarm6_controller_list_client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: timeout_sec == pytest.approx(2.0),
        call_async=_list_controllers,
    )
    controller._wait_future = lambda future, **_kwargs: future.value

    assert controller._wait_for_xarm6_trajectory_controller_state("active") == (
        True,
        "xArm6 trajectory controller is active",
    )
    assert observed_states == ["inactive", "active"]
    source = (
        ROOT
        / "cais_spade_llm/resources/robot/hardware_pick_place_controller.py"
    ).read_text(encoding="utf-8")
    assert "SwitchController" not in source
    assert "_switch_xarm6_trajectory_controller" not in source


def test_xarm6_real_config_keeps_physical_home_and_action_candidates() -> None:
    payload = json.loads(
        (ROOT / "cais_spade_llm/initialization/resources/robot_xarm6.json").read_text(
            encoding="utf-8"
        )
    )["xarm6"]
    real = payload["real"]
    gazebo = payload["gazebo"]

    assert real["named_positions"]["home"] != gazebo["named_positions"]["home"]
    assert real["controller"]["move_group"] == {
        "group_name": "xarm6",
        "ee_link": "link_eef",
        "tcp_link": "link_tcp",
        "frame_id": "world",
    }
    assert real["controller"]["joint_state_topics"] == [
        "/joint_states",
        "/xarm/joint_states",
        "/xarm6/joint_states",
        "/xarm6/xarm/joint_states",
        "/xarm6/xarm_gripper/joint_states",
    ]
    assert real["controller"]["hardware_trajectory_actions"]
    assert real["controller"]["hardware_joint_duration_scale"] == pytest.approx(
        4.0 / 1.3
    )
    assert real["static_capabilities"]["workspace_bounds"]["z_max_m"] == 1.6
    assert real["static_capabilities"]["gripper_reach"]["z_max_m"] == 1.6
    assert gazebo["static_capabilities"]["workspace_bounds"]["z_max_m"] == 1.5
    assert real["controller"]["hardware_cartesian_speed_mm_s"] == 50.0
    assert real["controller"]["hardware_cartesian_max_speed_mm_s"] == 100.0
    assert real["controller"]["hardware_cartesian_acceleration_mm_s2"] == 42.25
    assert real["controller"]["gripper"]["hardware_service"] == "set_gripper_position"
    assert (
        real["controller"]["gripper"]["hardware_action"]
        == "/xarm6/xarm_gripper/gripper_action"
    )
    assert real["controller"]["gripper"]["action_candidates"]

"""Focused no-motion contracts for UR5e pick/place and Smooth Hold latency."""

from __future__ import annotations

import ast
import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
UR5E_RESOURCE = ROOT / "cais_spade_llm/initialization/resources/robot_ur5e.json"
HARDWARE_RUNTIME = (
    ROOT
    / "ros2/cais_lab_robotics/config/hardware_runtime/xarm6_ur5e_hardware_runtime.yaml"
)
RTDE_SERVER = ROOT / "ros2/cais_lab_robotics/scripts/ur5e_rtde_trajectory_server.py"
KEYBOARD_TELEOP = ROOT / "ros2/cais_lab_robotics/scripts/keyboard_teleop.py"
CONTROL_PAGE = ROOT / "cais_spade_llm/ui/pages/control.py"
BRIDGE = ROOT / "cais_spade_llm/ui/bridge.py"
PERCEPTION_MANAGER = ROOT / "cais_spade_llm/ui/perception_manager.py"


def _python_tree(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def _function(path: Path, name: str) -> tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    source, tree = _python_tree(path)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, f"expected one {name} in {path}, found {len(matches)}"
    return source, matches[0]


def _segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def _attribute_calls(function: ast.AST, attribute: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    ]


def _rtde_server_module() -> Any:
    spec = importlib.util.spec_from_file_location("ur5e_latency_rtde_server", RTDE_SERVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_nonblocking_acquire(call: tuple[tuple[Any, ...], dict[str, Any]]) -> None:
    args, kwargs = call
    blocking = kwargs.get("blocking", args[0] if args else True)
    assert blocking is False
    assert "timeout" not in kwargs


class _RecordingLock:
    def __init__(self, acquire_result: bool) -> None:
        self.acquire_result = acquire_result
        self.acquire_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.release_calls = 0

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        self.acquire_calls.append((args, kwargs))
        return self.acquire_result

    def release(self) -> None:
        self.release_calls += 1


def test_physical_pick_place_uses_faster_cartesian_motion_configuration() -> None:
    resource = json.loads(UR5E_RESOURCE.read_text(encoding="utf-8"))
    controller = resource["ur5e"]["real"]["controller"]
    runtime = yaml.safe_load(HARDWARE_RUNTIME.read_text(encoding="utf-8"))["ur5e"]

    assert controller["hardware_cartesian_speed_m_s"] == pytest.approx(0.15)
    assert controller["hardware_cartesian_max_speed_m_s"] == pytest.approx(0.15)
    assert controller["hardware_cartesian_acceleration_m_s2"] == pytest.approx(0.20)
    assert runtime["rtde"]["cartesian_speed_m_s"] == pytest.approx(0.15)
    assert runtime["rtde"]["cartesian_max_speed_m_s"] == pytest.approx(0.15)
    assert runtime["rtde"]["cartesian_accel_m_s2"] == pytest.approx(0.20)
    assert runtime["cartesian"]["speed_mm_s"] == pytest.approx(100.0)


def test_physical_gripper_and_release_do_not_stack_duplicate_waits() -> None:
    resource = json.loads(UR5E_RESOURCE.read_text(encoding="utf-8"))
    controller = resource["ur5e"]["real"]["controller"]
    runtime = yaml.safe_load(HARDWARE_RUNTIME.read_text(encoding="utf-8"))["ur5e"]

    assert controller["gripper"]["rtde"]["open_settle_sec"] == pytest.approx(0.60)
    assert controller["gripper"]["rtde"]["close_settle_sec"] == pytest.approx(1.00)
    assert runtime["gripper"]["rtde"]["open_settle_sec"] == pytest.approx(0.60)
    assert runtime["gripper"]["rtde"]["close_settle_sec"] == pytest.approx(1.00)
    assert controller["motion"]["release_preopen_settle_sec"] == pytest.approx(0.0)
    assert controller["motion"]["release_postopen_settle_sec"] == pytest.approx(0.0)
    assert controller["motion"]["release_postdetach_settle_sec"] == pytest.approx(0.0)


def test_rtde_stationary_hold_is_loaded_from_runtime_configuration() -> None:
    module = _rtde_server_module()

    module._apply_hardware_arms_config(HARDWARE_RUNTIME)

    assert pytest.approx(0.10) == module.UR5E_RTDE_STATIONARY_HOLD_SEC
    assert pytest.approx(0.15) == module.UR5E_RTDE_CARTESIAN_SPEED_M_S
    assert pytest.approx(0.15) == module.UR5E_RTDE_CARTESIAN_MAX_SPEED_M_S
    assert pytest.approx(0.20) == module.UR5E_RTDE_CARTESIAN_ACCEL_M_S2


def test_main_cartesian_move_keeps_only_dispatch_time_frame_revalidation() -> None:
    _source, execute_cartesian = _function(RTDE_SERVER, "_execute_cartesian")
    validation_calls = _attribute_calls(
        execute_cartesian,
        "_cartesian_frame_validation",
    )
    resolve_calls = _attribute_calls(execute_cartesian, "_resolve_cartesian_target")
    move_calls = _attribute_calls(execute_cartesian, "_execute_movel")

    assert len(resolve_calls) == 1
    assert len(validation_calls) == 1
    assert len(move_calls) == 1
    assert resolve_calls[0].lineno < validation_calls[0].lineno < move_calls[0].lineno

    _source, resolve_target = _function(RTDE_SERVER, "_resolve_cartesian_target")
    assert len(
        _attribute_calls(resolve_target, "_validated_cartesian_world_base")
    ) == 1


def test_confirmed_execution_defers_duplicate_pose_checks_to_dispatch() -> None:
    source, preflight = _function(
        BRIDGE,
        "_digital_twin_robot_function_execution_preflight",
    )
    parameter_names = [argument.arg for argument in preflight.args.kwonlyargs]
    assert "defer_manual_pre_execute_checks" in parameter_names
    preflight_source = _segment(source, preflight)
    assert preflight_source.count("and not defer_manual_pre_execute_checks") == 2

    source, execute = _function(BRIDGE, "digital_twin_execute_robot_function")
    assert "defer_manual_pre_execute_checks=True" in _segment(source, execute)
    source, readiness = _function(
        BRIDGE,
        "digital_twin_robot_function_execution_readiness",
    )
    assert "defer_manual_pre_execute_checks=True" not in _segment(source, readiness)


def test_identical_smooth_hold_refresh_skips_rtde_redispatch_but_rechecks_safety() -> None:
    module = _rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._shutdown_requested = False
    server._active_lock = threading.Lock()
    server._active_goal = None
    server._active_motion_kind = ""
    server._active_goal_status = None
    server._jog_session_token = object()
    server._connect_control_for_goal = lambda: None
    server._joint_states_fresh = lambda: True
    server._cartesian_frame_ready = True
    server._cartesian_world_base_ready = True
    server._cartesian_jog_ready = True
    server._cartesian_world_base_observed = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    server._cartesian_frame_message = "UR5e Cartesian frame validation ready"
    server._validated_cartesian_world_base = lambda: pytest.fail(
        "Smooth Hold must not perform full TF frame validation"
    )
    server._read_actual_tcp_pose = lambda: [0.0] * 6
    safety_checks: list[list[float]] = []
    jog_calls: list[tuple[list[float], int, float]] = []
    server.control = SimpleNamespace(
        FEATURE_BASE=0,
        isPoseWithinSafetyLimits=lambda pose: safety_checks.append(list(pose)) or True,
        jogStart=lambda speeds, feature, acceleration: (
            jog_calls.append((list(speeds), int(feature), float(acceleration))) or True
        ),
    )
    statuses: list[dict[str, Any]] = []
    server._write_status = lambda status: statuses.append(dict(status))
    request = SimpleNamespace(
        stop=False,
        world_linear_velocity_m_s=SimpleNamespace(x=0.08, y=0.0, z=0.0),
        acceleration_m_s2=module.UR5E_RTDE_CARTESIAN_ACCEL_M_S2,
        watchdog_sec=0.25,
    )

    first = SimpleNamespace(accepted=None, message="")
    second = SimpleNamespace(accepted=None, message="")
    server._set_cartesian_jog(request, first)
    server._set_cartesian_jog(request, second)

    assert first.accepted is True
    assert second.accepted is True
    assert len(safety_checks) == 2
    assert jog_calls == [
        ([80.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0, pytest.approx(0.20))
    ]
    assert len(statuses) == 1


def test_smooth_hold_without_cached_frame_readiness_fails_without_tf_validation() -> None:
    module = _rtde_server_module()
    server = object.__new__(module.UR5eRTDETrajectoryServer)
    server._shutdown_requested = False
    server._active_lock = threading.Lock()
    server._active_goal = None
    server._active_motion_kind = ""
    server._active_goal_status = None
    server._jog_session_token = object()
    server._connect_control_for_goal = lambda: None
    server._joint_states_fresh = lambda: True
    server._cartesian_frame_ready = False
    server._cartesian_world_base_ready = False
    server._cartesian_jog_ready = False
    server._cartesian_world_base_observed = None
    server._validated_cartesian_world_base = lambda: pytest.fail(
        "Smooth Hold must not perform full TF frame validation"
    )
    server._write_status = lambda _status: None

    response = SimpleNamespace(accepted=None, message="")
    request = SimpleNamespace(
        stop=False,
        world_linear_velocity_m_s=SimpleNamespace(x=0.08, y=0.0, z=0.0),
        acceleration_m_s2=module.UR5E_RTDE_CARTESIAN_ACCEL_M_S2,
        watchdog_sec=0.25,
    )

    server._set_cartesian_jog(request, response)

    assert response.accepted is False
    assert "Repair Hardware Stack" in response.message


def test_ur5e_smooth_start_uses_nonblocking_service_readiness() -> None:
    source, set_jog = _function(KEYBOARD_TELEOP, "_set_ur5e_cartesian_jog")
    function_source = _segment(source, set_jog)

    assert "client.service_is_ready()" in function_source
    assert "client.wait_for_service" not in function_source


def test_cartesian_preflight_skips_named_position_discovery() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge.teleop_target = lambda _robot, _op: {
        "environment": "real",
        "warning": "",
        "hardware_cartesian_readiness": {
            "cartesian_jog_ready": True,
            "message": "ready",
        },
    }
    bridge._current_insertion_demonstration = lambda: None
    bridge._insertion_demonstration_blocking_error = lambda **_kwargs: ""
    bridge._move_insert_pending_review_error = lambda: ""
    bridge._ur5e_robot_function_state_uncertain = False
    bridge._ur5e_cartesian_jog_state_uncertain = False
    named_position_calls: list[str] = []
    bridge.teleop_named_position_readiness = lambda robot: (
        named_position_calls.append(robot) or (True, "ready")
    )

    smooth = bridge._teleop_preflight("ur5e", "cartesian_smooth")

    assert smooth.get("warning") in {None, ""}
    assert named_position_calls == []

    joint = bridge._teleop_preflight("ur5e", "joint")

    assert joint.get("warning") in {None, ""}
    assert named_position_calls == ["ur5e"]


def test_ur5e_cartesian_mode_prewarms_direct_interfaces() -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._teleop_cartesian_modes = {"xarm6": "off", "ur5e": "off"}
    bridge.teleop_target = lambda _robot, _op: {
        "environment": "real",
        "warning": "",
        "ros_domain_id": 7,
    }
    preflight_calls: list[tuple[str, str]] = []
    bridge._teleop_preflight = lambda robot, op: (
        preflight_calls.append((robot, op))
        or {
            "environment": "real",
            "warning": "",
            "ros_domain_id": 7,
        }
    )
    bridge._current_insertion_demonstration = lambda: None
    bridge._insertion_demonstration_blocking_error = lambda **_kwargs: ""
    bridge._move_insert_pending_review_error = lambda: ""
    requests: list[tuple[dict[str, Any], float, int | None]] = []

    def _request(
        payload: dict[str, Any],
        timeout_sec: float,
        ros_domain_id: int | None = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        requests.append((dict(payload), float(timeout_sec), ros_domain_id))
        return (
            True,
            "UR5e Cartesian direct interfaces ready",
            {"cartesian_jog_ready": True, "cartesian_function_ready": True},
        )

    bridge._teleop_request_payload = _request

    ok, _message = bridge.teleop_cartesian_mode("ur5e", "smooth")

    assert ok is True
    assert preflight_calls == [("ur5e", "cartesian_smooth")]
    assert len(requests) == 1
    payload, _timeout_sec, ros_domain_id = requests[0]
    assert payload == {"op": "cartesian_readiness", "robot": "ur5e"}
    assert ros_domain_id == 7
    assert bridge._teleop_cartesian_modes["ur5e"] == "smooth"


@pytest.mark.parametrize("blocked_lock", ["execution", "agent"])
def test_ur5e_smooth_hold_lock_contention_returns_without_waiting(
    blocked_lock: str,
) -> None:
    from cais_spade_llm.ui.bridge import SystemBridge

    bridge = object.__new__(SystemBridge)
    bridge._teleop_smooth_session_lock = threading.Lock()
    bridge._teleop_smooth_session = None
    bridge._teleop_preflight = lambda _robot, _op: {
        "environment": "real",
        "warning": "",
        "ros_domain_id": 7,
    }
    bridge._xarm6_cartesian_session_mode = lambda: "off"
    bridge._ur5e_robot_function_execution_active = "pick_approach"
    execution_lock = _RecordingLock(blocked_lock != "execution")
    agent_lock = _RecordingLock(blocked_lock != "agent")
    bridge._ur5e_robot_function_execution_lock = execution_lock
    bridge._physical_ur5e_robot_agent = lambda: SimpleNamespace(
        _robot_motion_lock=agent_lock
    )

    ok, message = bridge.teleop_cartesian_smooth(
        "ur5e",
        "x",
        25.0,
        "start",
    )

    assert ok is False
    assert "already" in message
    _assert_nonblocking_acquire(execution_lock.acquire_calls[0])
    if blocked_lock == "agent":
        _assert_nonblocking_acquire(agent_lock.acquire_calls[0])
        assert execution_lock.release_calls == 1
    else:
        assert agent_lock.acquire_calls == []


def test_smooth_hold_release_during_start_stops_before_marking_active() -> None:
    source, run_smooth_hold = _function(CONTROL_PAGE, "_run_smooth_hold")
    start_awaits = [
        node
        for node in ast.walk(run_smooth_hold)
        if isinstance(node, ast.Await)
        and "teleop_cartesian_smooth" in _segment(source, node)
        and '"start"' in _segment(source, node)
    ]
    active_assignments = [
        node
        for node in ast.walk(run_smooth_hold)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "smooth_hold"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "active"
            for target in node.targets
        )
    ]
    assert len(start_awaits) == 1
    assert len(active_assignments) == 1
    start_line = start_awaits[0].lineno
    active_line = active_assignments[0].lineno
    stale_press_guards = [
        node
        for node in ast.walk(run_smooth_hold)
        if isinstance(node, ast.If)
        and start_line < node.lineno < active_line
        and "pressed" in _segment(source, node.test)
        and "generation" in _segment(source, node.test)
    ]
    assert len(stale_press_guards) == 1
    guard_source = _segment(source, stale_press_guards[0])
    assert "teleop_cartesian_smooth" in guard_source
    assert '"stop"' in guard_source


def test_control_page_prewarms_selected_smooth_hold_mode() -> None:
    source, tree = _python_tree(CONTROL_PAGE)
    prewarm_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_task"
        and "_apply_cartesian_mode" in _segment(source, node)
        and '"Smooth Hold"' in _segment(source, node)
    ]

    assert len(prewarm_calls) == 1


def test_teleop_state_poll_does_not_compete_with_smooth_hold_requests() -> None:
    source, refresh_state = _function(CONTROL_PAGE, "_refresh_teleop_state_async")
    function_source = _segment(source, refresh_state)

    assert 'cartesian_jog_mode["mode"] == "smooth"' in function_source
    assert function_source.index('cartesian_jog_mode["mode"] == "smooth"') < (
        function_source.index("bridge.teleop_state")
    )


def test_ur5e_smooth_hold_refresh_budget_stays_below_watchdog() -> None:
    _source, refresh_once = _function(KEYBOARD_TELEOP, "_ur5e_refresh_cartesian_jog_once")
    service_calls = _attribute_calls(refresh_once, "_send_ur5e_cartesian_jog")
    assert len(service_calls) == 1
    timeout_keyword = next(
        keyword for keyword in service_calls[0].keywords if keyword.arg == "timeout_sec"
    )
    assert isinstance(timeout_keyword.value, ast.Constant)
    assert float(timeout_keyword.value.value) == pytest.approx(0.30)

    _source, refresh_loop = _function(KEYBOARD_TELEOP, "_ur5e_cartesian_jog_refresh_loop")
    wait_calls = _attribute_calls(refresh_loop, "wait")
    assert len(wait_calls) == 1
    assert isinstance(wait_calls[0].args[0], ast.Constant)
    assert float(wait_calls[0].args[0].value) == pytest.approx(0.05)


def test_realsense_preview_command_requests_ten_hz_maximum_rate() -> None:
    source, start_camera_stack = _function(PERCEPTION_MANAGER, "_start_camera_stack")

    assert "--maximum-rate-hz 10.0" in _segment(source, start_camera_stack)

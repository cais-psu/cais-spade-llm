"""Focused no-motion UR5e RTDE reset and mirror recovery tests."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.ui import bridge as bridge_module
from cais_spade_llm.ui.bridge import SystemBridge


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


class _LifecycleLock:
    def __init__(self, locked: bool = False) -> None:
        self._locked = locked

    def locked(self) -> bool:
        return self._locked


def _reset_bridge(
    *,
    source: str,
    cfg: dict[str, Any] | None,
    rtde_process: str,
    mirror_process: str = "",
) -> tuple[SystemBridge, dict[str, Any]]:
    bridge = object.__new__(SystemBridge)
    bridge.system_running = False
    bridge._starting = False
    bridge._stopping = False
    bridge._ur5e_rtde_reset_lock = threading.Lock()
    bridge._ur5e_robot_function_preflight_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._ur5e_robot_function_state_uncertain = False
    bridge._ur5e_robot_function_agent_lifecycle_lock = _LifecycleLock()
    controller_calls: list[float] = []
    controller = SimpleNamespace(
        reset_ur5e_hardware_trajectory_client=lambda *, timeout_sec: (
            controller_calls.append(timeout_sec) or True,
            "client recreated",
        )
    )
    agent = SimpleNamespace(
        _controller=controller,
        _robot_motion_lock=threading.Lock(),
    )
    bridge._physical_ur5e_robot_agent = lambda: agent
    target = {
        "environment": "real",
        "source": source,
        "ros_domain_id": 43,
    }
    bridge._ur5e_rtde_reset_target = lambda: (
        target,
        cfg,
        rtde_process,
        mirror_process,
        "",
    )
    bridge.hardware_connection_statuses = lambda force=False: {
        "ur5e": {
            "ip": "192.168.1.172",
            "reachable": True,
            "message": "OK",
        }
    }
    bridge._ur5e_rtde_trajectory_status = lambda: {"process_id": 101}
    calls: dict[str, Any] = {
        "stopped": [],
        "started": [],
        "teleop_stops": 0,
        "cache_clears": 0,
        "controller_timeouts": controller_calls,
        "mirror_starts": [],
    }

    def _stop_teleop() -> None:
        calls["teleop_stops"] += 1

    bridge._stop_teleop_server = _stop_teleop
    bridge.ros2_stop = (
        lambda name, reason="explicit_stop": calls["stopped"].append((name, reason))
        or None
    )
    bridge._clear_ros_action_service_snapshot = lambda: calls.__setitem__(
        "cache_clears", calls["cache_clears"] + 1
    )
    bridge._start_ur5e_rtde_trajectory_server = (
        lambda name, **kwargs: calls["started"].append((name, kwargs)) or None
    )
    bridge._wait_for_reset_ur5e_rtde_status = lambda **_kwargs: None
    bridge._wait_for_ros_action = lambda *_args, **_kwargs: None
    bridge._wait_for_ros_topic_publisher = lambda *_args, **_kwargs: None
    bridge._digital_twin_domain_ids = lambda: {
        "gazebo": 41,
        "hardware": 42,
        "hardware_ur5e": 43,
    }
    bridge._start_reset_ur5e_mirror = (
        lambda *, target, cfg, domains: calls["mirror_starts"].append(
            (target, cfg, domains)
        )
        or None
    )
    bridge._wait_for_reset_ur5e_mirror = lambda **_kwargs: None
    return bridge, calls


@pytest.mark.parametrize(
    ("source", "rtde_process", "mirror_process"),
    [
        ("hardware", "hardware_ur5e_rtde_trajectory_server", ""),
        (
            "digital_twin:ur5e only",
            "digital_twin_ur5e_only_hardware_ur5e_rtde_trajectory_server",
            "digital_twin_ur5e_only_sync",
        ),
        (
            "digital_twin:dual robots",
            "digital_twin_dual_robots_hardware_ur5e_rtde_trajectory_server",
            "digital_twin_dual_robots_sync_ur5e",
        ),
    ],
)
def test_targeted_reset_restarts_only_ur5e_rtde_and_exact_mirror(
    source: str,
    rtde_process: str,
    mirror_process: str,
) -> None:
    cfg = None if source == "hardware" else {"gazebo_process": "gazebo-kept-running"}
    bridge, calls = _reset_bridge(
        source=source,
        cfg=cfg,
        rtde_process=rtde_process,
        mirror_process=mirror_process,
    )
    pending_review_calls: list[dict[str, Any]] = []
    bridge._move_insert_pending_review_error = (  # type: ignore[method-assign]
        lambda **kwargs: pending_review_calls.append(dict(kwargs)) or ""
    )

    ok, message = bridge.teleop_reset_ur5e_rtde_connection()

    assert ok is True
    assert "without robot motion" in message
    assert "move_home" not in message
    assert "physical joint state were reacquired" in message
    assert "normal fresh readiness checks" in message
    assert calls["teleop_stops"] == 1
    expected_stops = (
        [(mirror_process, "ur5e_rtde_reset")] if mirror_process else []
    ) + [(rtde_process, "ur5e_rtde_reset")]
    assert calls["stopped"] == expected_stops
    assert calls["started"] == [
        (
            rtde_process,
            {
                "ros_domain_id": 43,
                "stop_read_only_monitor": False,
                "wait_for_action": False,
            },
        )
    ]
    assert calls["controller_timeouts"] == [8.0]
    assert calls["cache_clears"] == 2
    assert pending_review_calls == [{"allow_terminal_recovery": True}]
    assert bridge._ur5e_robot_function_state_uncertain is False
    assert bridge._ur5e_robot_function_state_uncertain_reason == ""
    untouched = {
        "gazebo-kept-running",
        "hardware_ur5e_moveit",
        "hardware_ur5e_rg2_gripper",
        "digital_twin_dual_robots_hardware_xarm6_driver",
        "digital_twin_dual_robots_sync_xarm6",
        "realsense_camera",
        "physical_perception",
    }
    assert untouched.isdisjoint(name for name, _reason in calls["stopped"])


def test_reset_checks_hardware_reachability_before_stopping_anything() -> None:
    bridge, calls = _reset_bridge(
        source="hardware",
        cfg=None,
        rtde_process="hardware_ur5e_rtde_trajectory_server",
    )
    bridge.hardware_connection_statuses = lambda force=False: {
        "ur5e": {
            "ip": "192.168.1.172",
            "reachable": False,
            "message": "unreachable",
        }
    }

    ok, message = bridge.teleop_reset_ur5e_rtde_connection()

    assert ok is False
    assert "hardware reachability" in message
    assert "Nothing was stopped" in message
    assert calls["teleop_stops"] == 0
    assert calls["stopped"] == []


def test_reset_refuses_active_ur5e_motion() -> None:
    bridge, calls = _reset_bridge(
        source="hardware",
        cfg=None,
        rtde_process="hardware_ur5e_rtde_trajectory_server",
    )
    bridge._ur5e_robot_function_execution_active = "Interactive Teleop joint"
    bridge._ur5e_robot_function_execution_lock.acquire()
    try:
        ok, message = bridge.teleop_reset_ur5e_rtde_connection()
    finally:
        bridge._ur5e_robot_function_execution_lock.release()

    assert ok is False
    assert "motion is active: Interactive Teleop joint" in message
    assert calls["stopped"] == []
    assert bridge._ur5e_robot_function_execution_active == "Interactive Teleop joint"


def test_reset_refuses_hardware_stack_lifecycle_operation() -> None:
    bridge, calls = _reset_bridge(
        source="hardware",
        cfg=None,
        rtde_process="hardware_ur5e_rtde_trajectory_server",
    )
    bridge._hardware_stack_lifecycle_lock = threading.Lock()
    bridge._hardware_stack_lifecycle_lock.acquire()
    try:
        ok, message = bridge.teleop_reset_ur5e_rtde_connection()
    finally:
        bridge._hardware_stack_lifecycle_lock.release()

    assert ok is False
    assert "Hardware Stack start or stop is running" in message
    assert calls["stopped"] == []


@pytest.mark.parametrize("busy_source", ["system", "lifecycle", "preflight"])
def test_reset_refuses_full_system_and_robot_functions_preparation(
    busy_source: str,
) -> None:
    bridge, calls = _reset_bridge(
        source="hardware",
        cfg=None,
        rtde_process="hardware_ur5e_rtde_trajectory_server",
    )
    if busy_source == "system":
        bridge.system_running = True
    elif busy_source == "lifecycle":
        bridge._ur5e_robot_function_agent_lifecycle_lock = _LifecycleLock(True)
    else:
        bridge._ur5e_robot_function_preflight_lock.acquire()
    try:
        ok, message = bridge.teleop_reset_ur5e_rtde_connection()
    finally:
        if busy_source == "preflight":
            bridge._ur5e_robot_function_preflight_lock.release()

    assert ok is False
    assert "full CAIS system" in message or "Robot Functions preparation" in message
    assert calls["stopped"] == []


@pytest.mark.parametrize(
    ("failure_stage", "configure_failure"),
    [
        (
            "RTDE startup",
            lambda bridge: setattr(
                bridge,
                "_start_ur5e_rtde_trajectory_server",
                lambda *_args, **_kwargs: "process start failed",
            ),
        ),
        (
            "fresh RTDE feedback",
            lambda bridge: setattr(
                bridge,
                "_wait_for_reset_ur5e_rtde_status",
                lambda **_kwargs: "joint feedback is stale",
            ),
        ),
        (
            "hidden action services",
            lambda bridge: setattr(
                bridge,
                "_wait_for_ros_action",
                lambda *_args, **_kwargs: "send_goal is missing",
            ),
        ),
        (
            "/joint_states",
            lambda bridge: setattr(
                bridge,
                "_wait_for_ros_topic_publisher",
                lambda *_args, **_kwargs: "publisher is missing",
            ),
        ),
        (
            "Gazebo mirror convergence",
            lambda bridge: setattr(
                bridge,
                "_wait_for_reset_ur5e_mirror",
                lambda **_kwargs: "mirror did not converge",
            ),
        ),
    ],
)
def test_reset_reports_exact_failed_recovery_stage(
    failure_stage: str,
    configure_failure: Any,
) -> None:
    bridge, _calls = _reset_bridge(
        source="digital_twin:ur5e only",
        cfg={"gazebo_process": "digital_twin_ur5e_only_gazebo"},
        rtde_process="digital_twin_ur5e_only_hardware_ur5e_rtde_trajectory_server",
        mirror_process="digital_twin_ur5e_only_sync",
    )
    configure_failure(bridge)

    ok, message = bridge.teleop_reset_ur5e_rtde_connection()

    assert ok is False
    assert f"failed at {failure_stage}" in message
    assert bridge._ur5e_robot_function_state_uncertain is True


def test_reset_reports_robot_functions_action_client_failure() -> None:
    bridge, _calls = _reset_bridge(
        source="hardware",
        cfg=None,
        rtde_process="hardware_ur5e_rtde_trajectory_server",
    )
    agent = bridge._physical_ur5e_robot_agent()
    agent._controller.reset_ur5e_hardware_trajectory_client = (
        lambda **_kwargs: (False, "client discovery timed out")
    )

    ok, message = bridge.teleop_reset_ur5e_rtde_connection()

    assert ok is False
    assert "failed at Robot Functions action client" in message
    assert "client discovery timed out" in message


def test_reset_reacquires_physical_state_without_requiring_move_home() -> None:
    bridge, _calls = _reset_bridge(
        source="hardware",
        cfg=None,
        rtde_process="hardware_ur5e_rtde_trajectory_server",
    )
    bridge._ur5e_robot_function_state_uncertain = True
    bridge._ur5e_robot_function_state_uncertain_reason = "motion result was not observed"

    ok, message = bridge.teleop_reset_ur5e_rtde_connection()
    assert ok is True
    assert "physical joint state were reacquired" in message
    assert bridge._ur5e_robot_function_state_uncertain is False
    assert bridge._ur5e_robot_function_state_uncertain_reason == ""


def test_successful_reset_clears_a_previous_reset_only_uncertain_gate() -> None:
    bridge, _calls = _reset_bridge(
        source="hardware",
        cfg=None,
        rtde_process="hardware_ur5e_rtde_trajectory_server",
    )
    bridge._ur5e_robot_function_state_uncertain = True
    bridge._ur5e_robot_function_state_uncertain_reason = (
        "Reset UR5e RTDE replaced the control process without robot motion; "
        "inspect the physical UR5e before commanding motion."
    )

    ok, message = bridge.teleop_reset_ur5e_rtde_connection()

    assert ok is True
    assert "physical joint state were reacquired" in message
    assert bridge._ur5e_robot_function_state_uncertain is False
    assert bridge._ur5e_robot_function_state_uncertain_reason == ""


def test_fresh_physical_state_clears_reset_only_gate_after_hardware_restart() -> None:
    bridge, _calls = _reset_bridge(
        source="hardware",
        cfg=None,
        rtde_process="hardware_ur5e_rtde_trajectory_server",
    )
    bridge._ur5e_robot_function_state_uncertain = True
    bridge._ur5e_robot_function_state_uncertain_reason = (
        "Reset UR5e RTDE replaced the control process without robot motion; "
        "inspect the physical UR5e before commanding motion."
    )
    bridge._move_insert_pending_review_error = lambda: ""
    bridge.teleop_target = lambda *_args: {
        "environment": "real",
        "ready": True,
        "warning": "",
        "hardware_cartesian_readiness": {
            "cartesian_jog_ready": True,
            "message": "ready",
        },
    }
    bridge.teleop_named_position_readiness = lambda _robot: (
        True,
        "fresh physical joint state",
    )

    readiness = bridge._teleop_preflight("ur5e", "cartesian")

    assert readiness["ready"] is True
    assert readiness["warning"] == ""
    assert bridge._ur5e_robot_function_state_uncertain is False
    assert bridge._ur5e_robot_function_state_uncertain_reason == ""


def test_fresh_physical_state_does_not_clear_motion_uncertainty() -> None:
    bridge, _calls = _reset_bridge(
        source="hardware",
        cfg=None,
        rtde_process="hardware_ur5e_rtde_trajectory_server",
    )
    bridge._ur5e_robot_function_state_uncertain = True
    bridge._ur5e_robot_function_state_uncertain_reason = "motion result was not observed"
    bridge._move_insert_pending_review_error = lambda: ""
    bridge.teleop_target = lambda *_args: {
        "environment": "real",
        "ready": True,
        "warning": "",
        "hardware_cartesian_readiness": {
            "cartesian_jog_ready": True,
            "message": "ready",
        },
    }
    bridge.teleop_named_position_readiness = lambda _robot: (
        True,
        "fresh physical joint state",
    )

    readiness = bridge._teleop_preflight("ur5e", "cartesian")

    assert readiness["ready"] is False
    assert "motion result was not observed" in readiness["warning"]
    assert bridge._ur5e_robot_function_state_uncertain is True


def test_disposing_cached_agent_does_not_clear_uncertain_physical_state() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._ur5e_robot_function_agent = None
    bridge._ur5e_robot_function_agent_domain_id = 43
    bridge._ur5e_robot_function_state_uncertain = True

    asyncio.run(bridge._dispose_ur5e_robot_function_agent())

    assert bridge._ur5e_robot_function_agent is None
    assert bridge._ur5e_robot_function_agent_domain_id is None
    assert bridge._ur5e_robot_function_state_uncertain is True
    assert "complete move_home" in bridge._ur5e_robot_function_agent_handoff_error()


@pytest.mark.parametrize(
    ("source", "expected_rtde", "expected_mirror"),
    [
        ("hardware", "hardware_ur5e_rtde_trajectory_server", ""),
        (
            "digital_twin:ur5e only",
            "digital_twin_ur5e_only_hardware_ur5e_rtde_trajectory_server",
            "digital_twin_ur5e_only_sync",
        ),
        (
            "digital_twin:dual robots",
            "digital_twin_dual_robots_hardware_ur5e_rtde_trajectory_server",
            "digital_twin_dual_robots_sync_ur5e",
        ),
    ],
)
def test_reset_target_resolves_standalone_and_digital_twin_process_names(
    source: str,
    expected_rtde: str,
    expected_mirror: str,
) -> None:
    bridge = object.__new__(SystemBridge)
    bridge.teleop_target = lambda *_args: {
        "environment": "real",
        "source": source,
        "ros_domain_id": 43,
    }

    _target, _cfg, rtde_process, mirror_process, error = (
        bridge._ur5e_rtde_reset_target()
    )

    assert error == ""
    assert rtde_process == expected_rtde
    assert mirror_process == expected_mirror


def test_reset_status_wait_requires_new_pid_fresh_feedback_and_matching_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = object.__new__(SystemBridge)
    bridge.ros2_proc_status = lambda _name: "running"
    statuses = iter(
        (
            {
                "process_id": 101,
                "ros_domain_id": 43,
                "updated_at": time.time(),
                "state": "ready",
                "rtde_receive_connected": True,
                "rtde_control_connected": True,
                "joint_states_fresh": True,
            },
            {
                "process_id": 202,
                "ros_domain_id": 43,
                "updated_at": time.time(),
                "state": "ready",
                "rtde_reset_required": False,
                "rtde_receive_connected": True,
                "rtde_control_connected": True,
                "joint_states_fresh": True,
            },
        )
    )
    bridge._ur5e_rtde_trajectory_status = lambda: next(statuses)
    monkeypatch.setattr("cais_spade_llm.ui.bridge.time.sleep", lambda _seconds: None)

    error = bridge._wait_for_reset_ur5e_rtde_status(
        process_name="rtde",
        previous_process_id=101,
        ros_domain_id=43,
        started_at=time.time() - 1.0,
        timeout_sec=1.0,
    )

    assert error is None


def test_reset_mirror_start_filters_out_xarm6_worker() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._wait_for_ros_service = lambda *_args, **_kwargs: None
    directions: list[tuple[str, str]] = []
    bridge._write_digital_twin_direction = (
        lambda target, direction: directions.append((target, direction))
    )
    starts: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    bridge._start_digital_twin_sync_process = (
        lambda target, cfg, **kwargs: starts.append((target, cfg, kwargs)) or None
    )
    cfg = {"gazebo_process": "digital_twin_dual_robots_gazebo"}
    domains = {"gazebo": 41, "hardware": 42, "hardware_ur5e": 43}

    error = bridge._start_reset_ur5e_mirror(
        target="dual robots",
        cfg=cfg,
        domains=domains,
    )

    assert error is None
    assert directions == [("dual robots", "hardware -> gazebo")]
    assert starts == [
        (
            "dual robots",
            cfg,
            {
                "gazebo_domain_id": 41,
                "hardware_domain_id": 42,
                "domain_ids": domains,
                "only_robot": "ur5e",
            },
        )
    ]


def test_control_page_exposes_physical_ur5e_reset_button_and_progress() -> None:
    source = Path("cais_spade_llm/ui/pages/control.py").read_text(
        encoding="utf-8"
    )

    assert '"Reset UR5e RTDE"' in source
    assert "bridge.teleop_reset_ur5e_rtde_connection" in source
    assert 'str(target.get("environment") or "") == "real"' in source
    assert 'rtde_reset_button.props("loading")' in source
    assert 'refresh_callbacks.get("robot_functions")' in source

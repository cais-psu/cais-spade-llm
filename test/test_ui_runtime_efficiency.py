"""Focused contracts for long-running UI and digital-twin efficiency."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _digital_twin_sync_module() -> Any:
    path = ROOT / "ros2/cais_lab_robotics/scripts/digital_twin_sync.py"
    spec = importlib.util.spec_from_file_location("digital_twin_sync_efficiency_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ur5e_hardware_updates_are_filtered_before_multiprocessing_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _digital_twin_sync_module()
    monkeypatch.setattr(module, "UR5E_MIRROR_MIN_PUBLISH_PERIOD_SEC", 0.05)
    monkeypatch.setattr(module, "UR5E_MIRROR_MIN_JOINT_DELTA_RAD", 0.001)
    previous = [0.0] * 6

    assert module._should_enqueue_hardware_update(
        "ur5e",
        positions=[0.1] * 6,
        last_positions=None,
        gripper_position=None,
        last_gripper_position=None,
        last_enqueue_ts=0.0,
        now=1.0,
    )
    assert not module._should_enqueue_hardware_update(
        "ur5e",
        positions=[0.1] * 6,
        last_positions=previous,
        gripper_position=None,
        last_gripper_position=None,
        last_enqueue_ts=1.0,
        now=1.02,
    )
    assert not module._should_enqueue_hardware_update(
        "ur5e",
        positions=[0.0005] * 6,
        last_positions=previous,
        gripper_position=0.01,
        last_gripper_position=0.01,
        last_enqueue_ts=1.0,
        now=1.05,
    )
    assert module._should_enqueue_hardware_update(
        "ur5e",
        positions=[0.001] * 6,
        last_positions=previous,
        gripper_position=0.01,
        last_gripper_position=0.01,
        last_enqueue_ts=1.0,
        now=1.05,
    )
    assert module._should_enqueue_hardware_update(
        "ur5e",
        positions=previous,
        last_positions=previous,
        gripper_position=0.02,
        last_gripper_position=0.01,
        last_enqueue_ts=1.0,
        now=1.05,
    )


def test_unchanged_ur5e_hardware_update_emits_configured_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _digital_twin_sync_module()
    monkeypatch.setattr(module, "UR5E_MIRROR_MIN_PUBLISH_PERIOD_SEC", 0.05)
    monkeypatch.setattr(module, "UR5E_MIRROR_MIN_JOINT_DELTA_RAD", 0.001)
    monkeypatch.setattr(module, "MIRROR_HARDWARE_HEARTBEAT_SEC", 1.0)
    positions = [0.1] * 6

    assert not module._should_enqueue_hardware_update(
        "ur5e",
        positions=positions,
        last_positions=positions,
        gripper_position=None,
        last_gripper_position=None,
        last_enqueue_ts=5.0,
        now=5.99,
    )
    assert module._should_enqueue_hardware_update(
        "ur5e",
        positions=positions,
        last_positions=positions,
        gripper_position=None,
        last_gripper_position=None,
        last_enqueue_ts=5.0,
        now=6.0,
    )


def test_stationary_hardware_stays_mirroring_until_joint_states_are_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _digital_twin_sync_module()
    monkeypatch.setattr(module, "MIRROR_HARDWARE_STALE_SEC", 3.0)

    fresh = module._hardware_mirror_status_without_update(
        last_hardware_update_ts=10.0,
        last_published_positions=[0.1] * 6,
        hardware_diagnostic_message="",
        now=12.5,
    )
    assert fresh[0] == "mirroring"
    assert "hardware pose unchanged" in fresh[1]
    assert fresh[2] == ""
    assert fresh[3] == pytest.approx(2.5)

    stale = module._hardware_mirror_status_without_update(
        last_hardware_update_ts=10.0,
        last_published_positions=[0.1] * 6,
        hardware_diagnostic_message="",
        now=13.01,
    )
    assert stale[0] == "waiting"
    assert "hardware /joint_states is stale" in stale[1]
    assert stale[2] == stale[1]
    assert stale[3] == pytest.approx(3.01)


def test_gazebo_mirror_status_uses_observed_output_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _digital_twin_sync_module()
    monkeypatch.setattr(module, "MIRROR_GAZEBO_CONVERGENCE_TOLERANCE_RAD", 0.01)
    monkeypatch.setattr(module, "MIRROR_GAZEBO_CONVERGENCE_TIMEOUT_SEC", 2.0)
    joint_names = list(module.ROBOTS["ur5e"]["gazebo_joints"])
    target = [0.5] * 6

    following = module._gazebo_target_status(
        "ur5e",
        target_positions=target,
        gazebo_joint_snapshot=dict.fromkeys(joint_names, 0.0),
        gazebo_joint_state_received_at=10.0,
        target_changed_at=10.0,
        now=11.0,
    )
    assert following["state"] == "mirroring"
    assert following["gazebo_converged"] is False
    assert following["gazebo_max_joint_error_rad"] == pytest.approx(0.5)

    failed = module._gazebo_target_status(
        "ur5e",
        target_positions=target,
        gazebo_joint_snapshot=dict.fromkeys(joint_names, 0.0),
        gazebo_joint_state_received_at=12.0,
        target_changed_at=10.0,
        now=12.01,
    )
    assert failed["state"] == "waiting"
    assert "did not reach the hardware joint target" in failed["message"]

    converged = module._gazebo_target_status(
        "ur5e",
        target_positions=target,
        gazebo_joint_snapshot=dict.fromkeys(joint_names, 0.495),
        gazebo_joint_state_received_at=12.0,
        target_changed_at=10.0,
        now=12.01,
    )
    assert converged["state"] == "mirroring"
    assert converged["gazebo_converged"] is True
    assert converged["last_error"] == ""

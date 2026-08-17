"""Focused contracts for long-running UI and digital-twin efficiency."""

from __future__ import annotations

import importlib.util
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pytest

from cais_spade_llm.utils.runtime_cleanup import (
    ACTION_LOG_BACKUP_COUNT,
    ACTION_LOG_MAX_BYTES,
    cleanup_runtime_artifacts,
    install_action_log_handlers,
    prune_hardware_run_logs,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_at(path: Path, *, modified_at: float, content: str = "log") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, (modified_at, modified_at))


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


def test_hardware_run_log_cleanup_keeps_newest_ten_per_component(
    tmp_path: Path,
) -> None:
    log_root = tmp_path / "cais_spade_llm" / "log"
    unrelated = log_root / "operator_notes.log"
    _write_at(unrelated, modified_at=1.0, content="preserve")
    for component in ("hardware_xarm6_driver", "hardware_ur5e_rtde_trajectory_server"):
        for index in range(12):
            _write_at(
                log_root / f"{component}__run_{index}.log",
                modified_at=float(index + 1),
                content=str(index),
            )

    dry_run = prune_hardware_run_logs(log_root, apply=False)

    assert dry_run["counts"]["hardware_run_logs"] == 4
    assert len(list(log_root.glob("*__run_*.log"))) == 24

    applied = prune_hardware_run_logs(log_root, apply=True)

    assert applied["counts"]["hardware_run_logs"] == 4
    assert unrelated.read_text(encoding="utf-8") == "preserve"
    for component in ("hardware_xarm6_driver", "hardware_ur5e_rtde_trajectory_server"):
        retained = sorted(path.name for path in log_root.glob(f"{component}__run_*.log"))
        assert len(retained) == 10
        assert f"{component}__run_0.log" not in retained
        assert f"{component}__run_1.log" not in retained
        assert f"{component}__run_11.log" in retained


def test_runtime_cleanup_rejects_a_symlinked_cleanup_root(tmp_path: Path) -> None:
    real_log_root = tmp_path / "real-log-root"
    real_log_root.mkdir()
    symlinked_log_root = tmp_path / "linked-log-root"
    symlinked_log_root.symlink_to(real_log_root, target_is_directory=True)

    with pytest.raises(ValueError, match="directory symlink"):
        prune_hardware_run_logs(symlinked_log_root, apply=True)


def test_action_log_handler_rotates_and_is_not_duplicated(
    tmp_path: Path,
) -> None:
    logger = logging.getLogger("runtime_cleanup_test")
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()
    try:
        log_path = tmp_path / "runtime_cleanup_actions.log"
        install_action_log_handlers(logger, log_path)
        install_action_log_handlers(logger, log_path)
        rotating = [
            handler
            for handler in logger.handlers
            if isinstance(handler, RotatingFileHandler)
        ]
        assert len(rotating) == 1
        assert rotating[0].maxBytes == ACTION_LOG_MAX_BYTES
        assert rotating[0].backupCount == ACTION_LOG_BACKUP_COUNT

        rotating[0].maxBytes = 64
        logger.info("x" * 80)
        logger.info("y" * 80)
        rotating[0].flush()
        assert Path(f"{log_path}.1").is_file()
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def test_cais_ros_retention_removes_only_old_owned_entries(
    tmp_path: Path,
) -> None:
    now = time.time()
    project = tmp_path / "project"
    home = tmp_path / "home"
    ros_root = project / "cais_spade_llm" / "log" / "ros"
    old = ros_root / "old-session"
    recent = ros_root / "recent-session"
    active = ros_root / "active-long-session"
    _write_at(old / "node.log", modified_at=now - (9 * 86400))
    _write_at(recent / "node.log", modified_at=now - 60)
    _write_at(active / "node.log", modified_at=now - 30)
    os.utime(old, (now - (9 * 86400), now - (9 * 86400)))
    os.utime(recent, (now - 60, now - 60))
    os.utime(active, (now - (9 * 86400), now - (9 * 86400)))
    latest = ros_root / "latest"
    latest.symlink_to(recent, target_is_directory=True)

    report = cleanup_runtime_artifacts(
        root=project,
        home=home,
        apply=True,
        include_cais=True,
        include_caches=False,
        include_global_ros=False,
        now=now,
    )

    assert report["counts"]["cais_ros_logs"] == 1
    assert not old.exists()
    assert recent.is_dir()
    assert active.is_dir()
    assert latest.is_symlink()


def test_global_ros_retention_requires_explicit_opt_in(tmp_path: Path) -> None:
    now = time.time()
    project = tmp_path / "project"
    home = tmp_path / "home"
    global_root = home / ".ros" / "log"
    old = global_root / "old-session"
    recent = global_root / "recent-session"
    _write_at(old / "node.log", modified_at=now - (9 * 86400))
    _write_at(recent / "node.log", modified_at=now - 60)
    os.utime(old, (now - (9 * 86400), now - (9 * 86400)))
    os.utime(recent, (now - 60, now - 60))

    cleanup_runtime_artifacts(
        root=project,
        home=home,
        apply=True,
        include_cais=False,
        include_caches=False,
        include_global_ros=False,
        now=now,
    )
    assert old.is_dir()

    dry_run = cleanup_runtime_artifacts(
        root=project,
        home=home,
        apply=False,
        include_cais=False,
        include_caches=False,
        include_global_ros=True,
        now=now,
    )
    assert dry_run["counts"]["global_ros_logs"] == 1
    assert old.is_dir()

    applied = cleanup_runtime_artifacts(
        root=project,
        home=home,
        apply=True,
        include_cais=False,
        include_caches=False,
        include_global_ros=True,
        now=now,
    )
    assert applied["counts"]["global_ros_logs"] == 1
    assert not old.exists()
    assert recent.is_dir()


def test_cache_cleanup_removes_only_named_generated_directories(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    home = tmp_path / "home"
    removable = (
        project / ".pytest_cache",
        project / ".ruff_cache",
        project / "cais_spade_llm" / "__pycache__",
        project / ".vscode" / "__pycache__",
    )
    for directory in removable:
        _write_at(directory / "cache", modified_at=1.0)
    preserved = project / ".mypy_cache"
    _write_at(preserved / "cache", modified_at=1.0)
    excluded = project / ".venv" / "package" / "__pycache__"
    _write_at(excluded / "cache", modified_at=1.0)

    report = cleanup_runtime_artifacts(
        root=project,
        home=home,
        apply=True,
        include_cais=False,
        include_caches=True,
        include_global_ros=False,
    )

    assert report["counts"]["test_caches"] == len(removable)
    assert all(not directory.exists() for directory in removable)
    assert preserved.is_dir()
    assert excluded.is_dir()


def test_ros2_commands_route_future_logs_to_cais_owned_directory() -> None:
    from cais_spade_llm.ui import ros2_processes

    expected = ROOT / "cais_spade_llm" / "log" / "ros"
    assert f"export ROS_LOG_DIR={expected}" in ros2_processes.ROS2_ENV

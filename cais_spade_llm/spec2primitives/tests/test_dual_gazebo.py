"""Tests for the isolated Spec2Primitives dual Gazebo adapter and placeholder chat."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cais_spade_llm.spec2primitives.adapters.dual_gazebo import (
    DUAL_GAZEBO_NAME,
    read_dual_gazebo_status,
    start_dual_gazebo,
    stop_dual_gazebo,
)
from cais_spade_llm.spec2primitives.ui import _placeholder_reply


class FakeRuntime:
    """In-memory runtime implementing only the Spec2Primitives adapter protocol."""

    def __init__(self) -> None:
        self.statuses = {DUAL_GAZEBO_NAME: "stopped"}
        self.hardware_statuses = {
            "xarm6": {"overall": "stopped"},
            "ur5e": {"overall": "stopped"},
            "dual robots": {"overall": "stopped"},
        }
        self.start_error: str | None = None
        self.status_error: RuntimeError | None = None
        self.start_calls: list[str] = []
        self.stop_calls: list[str] = []

    def ros2_all_statuses(self) -> dict[str, str]:
        if self.status_error is not None:
            raise self.status_error
        return dict(self.statuses)

    def hardware_stack_status(self, robot: str) -> dict[str, object]:
        return dict(self.hardware_statuses[robot])

    def ros2_start(self, name: str) -> str | None:
        self.start_calls.append(name)
        return self.start_error

    def ros2_stop(self, name: str) -> None:
        self.stop_calls.append(name)


def test_reads_stopped_and_running_status() -> None:
    runtime = FakeRuntime()

    assert read_dual_gazebo_status(runtime).state == "stopped"

    runtime.statuses[DUAL_GAZEBO_NAME] = "running"
    status = read_dual_gazebo_status(runtime)

    assert status.state == "running"
    assert status.blocked_reason is None


@pytest.mark.parametrize("hardware_stack", ["xarm6", "ur5e", "dual robots"])
def test_hardware_stack_blocks_start(hardware_stack: str) -> None:
    runtime = FakeRuntime()
    runtime.hardware_statuses[hardware_stack]["overall"] = "running"

    error = start_dual_gazebo(runtime)

    assert error == "Blocked: hardware stack is running. Stop hardware first."
    assert runtime.start_calls == []


def test_start_uses_exact_dual_gazebo_name() -> None:
    runtime = FakeRuntime()

    assert start_dual_gazebo(runtime) is None
    assert runtime.start_calls == [DUAL_GAZEBO_NAME]


def test_duplicate_start_is_rejected() -> None:
    runtime = FakeRuntime()
    runtime.statuses[DUAL_GAZEBO_NAME] = "running"

    error = start_dual_gazebo(runtime)

    assert error == "Dual Robots (xArm6 + UR5e) is already running."
    assert runtime.start_calls == []


def test_runtime_start_error_is_returned_unchanged() -> None:
    runtime = FakeRuntime()
    runtime.start_error = "ROS2 workspace is not built yet."

    assert start_dual_gazebo(runtime) == runtime.start_error
    assert runtime.start_calls == [DUAL_GAZEBO_NAME]


def test_start_rechecks_hardware_state() -> None:
    runtime = FakeRuntime()
    assert read_dual_gazebo_status(runtime).blocked_reason is None
    runtime.hardware_statuses["dual robots"]["overall"] = "running"

    error = start_dual_gazebo(runtime)

    assert error == "Blocked: hardware stack is running. Stop hardware first."
    assert runtime.start_calls == []


def test_failed_fresh_state_check_never_starts() -> None:
    runtime = FakeRuntime()
    runtime.status_error = RuntimeError("fresh status unavailable")

    with pytest.raises(RuntimeError, match="fresh status unavailable"):
        start_dual_gazebo(runtime)

    assert runtime.start_calls == []


def test_stop_uses_exact_dual_gazebo_name() -> None:
    runtime = FakeRuntime()

    stop_dual_gazebo(runtime)

    assert runtime.stop_calls == [DUAL_GAZEBO_NAME]


def test_placeholder_reply_states_that_nothing_executed() -> None:
    reply = _placeholder_reply("assembly the medium gear")

    assert "not connected yet" in reply
    assert "no plan or robot action was executed" in reply


def test_spec2primitives_does_not_import_bridge() -> None:
    spec2primitives_root = Path(__file__).resolve().parents[1]

    for path in spec2primitives_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imported_modules.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert "cais_spade_llm.ui.bridge" not in imported_modules


def test_spec2primitives_production_code_has_no_ground_truth_reference() -> None:
    spec2primitives_root = Path(__file__).resolve().parents[1]
    forbidden_references = (
        "gazebo_msgs",
        "/gazebo/model_states",
        "/get_entity_state",
        "gazebo_camera_detector",
        "table_spec2primitives.world",
    )

    production_paths = [
        path
        for path in spec2primitives_root.rglob("*.py")
        if "tests" not in path.relative_to(spec2primitives_root).parts
    ]

    for path in production_paths:
        source = path.read_text(encoding="utf-8")
        for forbidden_reference in forbidden_references:
            assert forbidden_reference not in source, (
                f"{path} contains forbidden ground-truth reference "
                f"{forbidden_reference!r}"
            )

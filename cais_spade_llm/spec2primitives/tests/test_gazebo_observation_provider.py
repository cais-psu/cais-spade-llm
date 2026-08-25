"""Tests for demand-driven live Gazebo RGB-D observation capture."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cais_spade_llm.spec2primitives.tools.observation_context import (
    CAMERA_IDS,
    MAX_BUNDLE_SKEW_NS,
    MAX_RGB_DEPTH_SKEW_NS,
    ObservationContextError,
    read_observation_bundle,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    gazebo_observation_provider,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.gazebo_observation_provider import (
    GazeboObservationProviderError,
    capture_gazebo_observation,
)

_EVENT_DRAIN_TIMEOUT_SEC = 0.02


class FakeBridgeError(Exception):
    """Represent one controlled image decoding failure."""


class FakeBridge:
    """Return the controlled array carried by a fake Image message."""

    def imgmsg_to_cv2(
        self,
        message: FakeImage,
        *,
        desired_encoding: str,
    ) -> np.ndarray:
        assert desired_encoding == "passthrough"
        if message.payload_error:
            raise FakeBridgeError("controlled invalid payload")
        return message.array


@dataclass
class FakeImage:
    """Minimal controlled equivalent of one sensor_msgs Image."""

    header: Any
    encoding: str
    width: int
    height: int
    array: np.ndarray
    payload_error: bool = False


@dataclass
class FakeCameraInfo:
    """Minimal controlled equivalent of one sensor_msgs CameraInfo."""

    header: Any
    width: int = 640
    height: int = 480
    distortion_model: str = "plumb_bob"
    d: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    k: tuple[float, ...] = (
        500.0,
        0.0,
        320.0,
        0.0,
        500.0,
        240.0,
        0.0,
        0.0,
        1.0,
    )
    r: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    p: tuple[float, ...] = (
        500.0,
        0.0,
        320.0,
        0.0,
        0.0,
        500.0,
        240.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )


class FakeContext:
    """Track request-owned ROS context shutdown."""

    def __init__(self) -> None:
        self.shutdown = False

    def try_shutdown(self) -> None:
        self.shutdown = True


class FakeNode:
    """Track subscriptions made for one explicit capture request."""

    def __init__(self) -> None:
        self.callbacks: dict[str, Callable[[Any], None]] = {}
        self.destroyed = False

    def create_subscription(
        self,
        _message_type: Any,
        topic: str,
        callback: Callable[[Any], None],
        qos: Any,
    ) -> object:
        assert qos == "sensor_data"
        self.callbacks[topic] = callback
        return object()

    def destroy_node(self) -> None:
        self.destroyed = True


class FakeExecutor:
    """Deliver controlled messages only after capture starts spinning."""

    def __init__(self, runtime: FakeRosRuntime, context: FakeContext) -> None:
        self.runtime = runtime
        self.context = context
        self.node: FakeNode | None = None
        self.removed = False
        self.shutdown_called = False

    def add_node(self, node: FakeNode) -> None:
        self.node = node

    def spin_once(self, *, timeout_sec: float) -> None:
        assert timeout_sec > 0
        if not self.runtime.events:
            return
        topic, message = self.runtime.events.pop(0)
        assert self.node is not None
        self.node.callbacks[topic](message)

    def remove_node(self, node: FakeNode) -> None:
        assert node is self.node
        self.removed = True

    def shutdown(self, *, timeout_sec: float) -> None:
        assert timeout_sec == 0
        self.shutdown_called = True


class FakeRclpy:
    """Create request-owned fake nodes without global state."""

    def __init__(self, runtime: FakeRosRuntime) -> None:
        self.runtime = runtime

    def init(
        self,
        *,
        context: FakeContext,
        signal_handler_options: Any,
    ) -> None:
        assert signal_handler_options == "none"
        self.runtime.contexts.append(context)

    def create_node(self, name: str, *, context: FakeContext) -> FakeNode:
        assert name == "spec2primitives_gazebo_observation"
        assert context in self.runtime.contexts
        node = FakeNode()
        self.runtime.nodes.append(node)
        return node


class FakeRosRuntime:
    """Provide controlled request-scoped ROS dependencies and messages."""

    def __init__(self, events: list[tuple[str, Any]]) -> None:
        self.events = list(events)
        self.contexts: list[FakeContext] = []
        self.nodes: list[FakeNode] = []
        self.executors: list[FakeExecutor] = []
        self.rclpy = FakeRclpy(self)

    def dependencies(self) -> gazebo_observation_provider._RosDependencies:
        runtime = self

        def executor_type(*, context: FakeContext) -> FakeExecutor:
            executor = FakeExecutor(runtime, context)
            runtime.executors.append(executor)
            return executor

        return gazebo_observation_provider._RosDependencies(
            rclpy=self.rclpy,
            context_type=FakeContext,
            executor_type=executor_type,
            image_type=FakeImage,
            camera_info_type=FakeCameraInfo,
            bridge_type=FakeBridge,
            bridge_error_type=FakeBridgeError,
            sensor_data_qos="sensor_data",
            no_signal_handlers="none",
        )


def test_import_has_no_ros_side_effects() -> None:
    source_path = Path(gazebo_observation_provider.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    top_level_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    top_level_imports.update(
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not top_level_imports.intersection(
        {"rclpy", "sensor_msgs.msg", "cv_bridge"}
    )
    error = GazeboObservationProviderError("capture_timeout", "controlled")
    assert error.reason == "capture_timeout"


def test_explicit_capture_writes_and_reloads_live_four_camera_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _install_runtime(monkeypatch, _complete_events())
    observations_root = tmp_path / "observations"

    bundle_path = capture_gazebo_observation(
        observations_root,
        "observation_live_0001",
    )
    loaded = read_observation_bundle(bundle_path)

    assert bundle_path == observations_root / "observation_live_0001"
    assert loaded.evidence_label == "live"
    assert tuple(item.camera_id for item in loaded.camera_observations) == CAMERA_IDS
    assert loaded.captured_at_ns == max(
        timestamp
        for item in loaded.camera_observations
        for timestamp in (item.rgb_timestamp_ns, item.depth_timestamp_ns)
    )
    for camera_index, item in enumerate(loaded.camera_observations):
        assert item.rgb.shape == (480, 640, 3)
        assert item.rgb.dtype == np.uint8
        assert np.all(item.rgb[..., 0] == 20 + camera_index)
        assert item.depth_m.shape == (480, 640)
        assert item.depth_m.dtype == np.float32
        assert item.camera_calibration.frame == item.rgb_frame
        assert item.camera_calibration.extrinsics_available is False
        assert (bundle_path / f"{item.camera_id}_rgb.png").is_file()

    assert set(runtime.nodes[0].callbacks) == _expected_topics()
    _assert_runtime_cleaned(runtime)


def test_second_request_cannot_reuse_first_request_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _install_runtime(monkeypatch, _complete_events())
    capture_gazebo_observation(tmp_path, "observation_live_first")

    with pytest.raises(GazeboObservationProviderError) as exc_info:
        capture_gazebo_observation(
            tmp_path,
            "observation_live_second",
            timeout_sec=0.001,
        )

    assert exc_info.value.reason == "calibration_missing"
    assert not (tmp_path / "observation_live_second").exists()
    assert len(runtime.nodes) == 2
    _assert_runtime_cleaned(runtime)


@pytest.mark.parametrize(
    ("depth_difference_ns", "expected_reason"),
    [
        (MAX_RGB_DEPTH_SKEW_NS, None),
        (MAX_RGB_DEPTH_SKEW_NS + 1, "capture_timeout"),
    ],
)
def test_rgb_depth_synchronization_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    depth_difference_ns: int,
    expected_reason: str | None,
) -> None:
    runtime = _install_runtime(
        monkeypatch,
        _complete_events(
            camera_timestamps_ns=(
                1_000_000_000,
                1_020_000_000,
                1_040_000_000,
                1_060_000_000,
            ),
            depth_difference_ns=depth_difference_ns,
        ),
    )

    if expected_reason is None:
        bundle_path = capture_gazebo_observation(
            tmp_path,
            "observation_rgb_depth_boundary",
        )
        assert bundle_path.is_dir()
    else:
        with pytest.raises(GazeboObservationProviderError) as exc_info:
            capture_gazebo_observation(
                tmp_path,
                "observation_rgb_depth_boundary",
                timeout_sec=_EVENT_DRAIN_TIMEOUT_SEC,
            )
        assert exc_info.value.reason == expected_reason
        assert not (tmp_path / "observation_rgb_depth_boundary").exists()
    _assert_runtime_cleaned(runtime)


@pytest.mark.parametrize(
    ("last_camera_offset_ns", "expected_reason"),
    [
        (MAX_BUNDLE_SKEW_NS, None),
        (MAX_BUNDLE_SKEW_NS + 1, "capture_timeout"),
    ],
)
def test_four_camera_synchronization_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    last_camera_offset_ns: int,
    expected_reason: str | None,
) -> None:
    runtime = _install_runtime(
        monkeypatch,
        _complete_events(
            camera_timestamps_ns=(
                1_000_000_000,
                1_050_000_000,
                1_100_000_000,
                1_000_000_000 + last_camera_offset_ns,
            ),
            depth_difference_ns=0,
        ),
    )

    if expected_reason is None:
        path = capture_gazebo_observation(
            tmp_path,
            "observation_bundle_boundary",
        )
        assert read_observation_bundle(path).bundle_skew_ns == MAX_BUNDLE_SKEW_NS
    else:
        with pytest.raises(GazeboObservationProviderError) as exc_info:
            capture_gazebo_observation(
                tmp_path,
                "observation_bundle_boundary",
                timeout_sec=_EVENT_DRAIN_TIMEOUT_SEC,
            )
        assert exc_info.value.reason == expected_reason
    _assert_runtime_cleaned(runtime)


@pytest.mark.parametrize("calibration_problem", ["missing", "invalid", "wrong_frame"])
def test_missing_or_invalid_calibration_returns_calibration_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibration_problem: str,
) -> None:
    events = _complete_events()
    camera_info_topic = f"/{CAMERA_IDS[-1]}/{CAMERA_IDS[-1]}/camera_info"
    events = [event for event in events if event[0] != camera_info_topic]
    if calibration_problem != "missing":
        camera_info = _camera_info(CAMERA_IDS[-1], 1_060_000_000)
        if calibration_problem == "invalid":
            camera_info.d = ()
        else:
            camera_info.header.frame_id = "wrong_rgb_frame"
        events.insert(9, (camera_info_topic, camera_info))
    runtime = _install_runtime(monkeypatch, events)

    with pytest.raises(GazeboObservationProviderError) as exc_info:
        capture_gazebo_observation(
            tmp_path / "observations",
            "observation_calibration_missing",
            timeout_sec=_EVENT_DRAIN_TIMEOUT_SEC,
        )

    assert exc_info.value.reason == "calibration_missing"
    assert not (tmp_path / "observations").exists()
    _assert_runtime_cleaned(runtime)


def test_incomplete_camera_returns_capture_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_depth_topic = (
        f"/{CAMERA_IDS[-1]}/{CAMERA_IDS[-1]}/depth/image_raw"
    )
    events = [
        event for event in _complete_events() if event[0] != missing_depth_topic
    ]
    runtime = _install_runtime(monkeypatch, events)

    with pytest.raises(GazeboObservationProviderError) as exc_info:
        capture_gazebo_observation(
            tmp_path,
            "observation_incomplete",
            timeout_sec=_EVENT_DRAIN_TIMEOUT_SEC,
        )

    assert exc_info.value.reason == "capture_timeout"
    assert not (tmp_path / "observation_incomplete").exists()
    _assert_runtime_cleaned(runtime)


@pytest.mark.parametrize(
    "message_problem",
    [
        "rgb_encoding",
        "depth_encoding",
        "rgb_shape",
        "empty_frame",
        "invalid_timestamp",
        "invalid_payload",
    ],
)
def test_malformed_image_returns_invalid_message_without_partial_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message_problem: str,
) -> None:
    events = _complete_events()
    images = [message for _topic, message in events if isinstance(message, FakeImage)]
    rgb = next(message for message in images if message.encoding == "rgb8")
    depth = next(message for message in images if message.encoding == "32FC1")
    if message_problem == "rgb_encoding":
        rgb.encoding = "bgr8"
    elif message_problem == "depth_encoding":
        depth.encoding = "16UC1"
    elif message_problem == "rgb_shape":
        rgb.array = np.zeros((10, 10, 3), dtype=np.uint8)
    elif message_problem == "empty_frame":
        rgb.header.frame_id = ""
    elif message_problem == "invalid_timestamp":
        rgb.header.stamp.nanosec = 1_000_000_000
    else:
        rgb.payload_error = True
    runtime = _install_runtime(monkeypatch, events)

    with pytest.raises(GazeboObservationProviderError) as exc_info:
        capture_gazebo_observation(
            tmp_path / "observations",
            "observation_invalid_message",
        )

    assert exc_info.value.reason == "invalid_message"
    assert not (tmp_path / "observations").exists()
    _assert_runtime_cleaned(runtime)


def test_duplicate_ref_preserves_existing_bundle_and_cleans_second_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_runtime = _install_runtime(monkeypatch, _complete_events())
    bundle_path = capture_gazebo_observation(tmp_path, "observation_duplicate")
    original_manifest = (bundle_path / "manifest.json").read_bytes()

    second_runtime = _install_runtime(monkeypatch, _complete_events())
    with pytest.raises(ObservationContextError):
        capture_gazebo_observation(tmp_path, "observation_duplicate")

    assert (bundle_path / "manifest.json").read_bytes() == original_manifest
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "observation_duplicate"
    ]
    _assert_runtime_cleaned(first_runtime)
    _assert_runtime_cleaned(second_runtime)


def test_storage_rejection_is_not_reclassified_and_ros_is_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _install_runtime(monkeypatch, _complete_events())

    def reject_storage(_root: Path, _bundle: Any) -> Path:
        raise ObservationContextError("controlled storage rejection")

    monkeypatch.setattr(
        gazebo_observation_provider,
        "write_observation_bundle",
        reject_storage,
    )

    with pytest.raises(ObservationContextError, match="controlled storage rejection"):
        capture_gazebo_observation(tmp_path, "observation_storage_rejection")

    _assert_runtime_cleaned(runtime)


@pytest.mark.parametrize("timeout_sec", [0, -1.0, float("inf"), float("nan"), True])
def test_invalid_timeout_is_rejected_before_ros_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeout_sec: float,
) -> None:
    loader_called = False

    def load_dependencies() -> Any:
        nonlocal loader_called
        loader_called = True
        raise AssertionError("ROS dependency loader must not run")

    monkeypatch.setattr(
        gazebo_observation_provider,
        "_load_ros_dependencies",
        load_dependencies,
    )

    with pytest.raises(ValueError, match="timeout_sec"):
        capture_gazebo_observation(
            tmp_path,
            "observation_invalid_timeout",
            timeout_sec=timeout_sec,
        )

    assert loader_called is False


def test_ros_dependency_failure_uses_ros_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_dependencies() -> Any:
        raise GazeboObservationProviderError(
            "ros_unavailable",
            "controlled missing ROS2",
        )

    monkeypatch.setattr(
        gazebo_observation_provider,
        "_load_ros_dependencies",
        load_dependencies,
    )

    with pytest.raises(GazeboObservationProviderError) as exc_info:
        capture_gazebo_observation(tmp_path, "observation_ros_unavailable")

    assert exc_info.value.reason == "ros_unavailable"


def test_provider_has_no_agent_ui_detector_state_or_ground_truth_dependency() -> None:
    source_path = Path(gazebo_observation_provider.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
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

    for forbidden_module in (
        "cais_spade_llm.agents",
        "cais_spade_llm.ui",
        "gazebo_msgs",
    ):
        assert not any(
            module == forbidden_module or module.startswith(f"{forbidden_module}.")
            for module in imported_modules
        )
    for forbidden_reference in (
        "/gazebo/model_states",
        "/get_entity_state",
        "gazebo_camera_detector",
        "table_spec2primitives.world",
        "SystemBridge",
        "ProductAgent",
        "RobotAgent",
    ):
        assert forbidden_reference not in source


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    events: list[tuple[str, Any]],
) -> FakeRosRuntime:
    runtime = FakeRosRuntime(events)
    monkeypatch.setattr(
        gazebo_observation_provider,
        "_load_ros_dependencies",
        runtime.dependencies,
    )
    return runtime


def _complete_events(
    *,
    camera_timestamps_ns: tuple[int, int, int, int] = (
        1_000_000_000,
        1_020_000_000,
        1_040_000_000,
        1_060_000_000,
    ),
    depth_difference_ns: int = 10_000_000,
) -> list[tuple[str, Any]]:
    events = []
    for camera_index, (camera_id, rgb_timestamp_ns) in enumerate(
        zip(CAMERA_IDS, camera_timestamps_ns, strict=True)
    ):
        events.extend(
            [
                (
                    f"/{camera_id}/{camera_id}/camera_info",
                    _camera_info(camera_id, rgb_timestamp_ns),
                ),
                (
                    f"/{camera_id}/{camera_id}/image_raw",
                    _rgb_image(camera_id, camera_index, rgb_timestamp_ns),
                ),
                (
                    f"/{camera_id}/{camera_id}/depth/image_raw",
                    _depth_image(
                        camera_id,
                        camera_index,
                        rgb_timestamp_ns + depth_difference_ns,
                    ),
                ),
            ]
        )
    return events


def _camera_info(camera_id: str, timestamp_ns: int) -> FakeCameraInfo:
    return FakeCameraInfo(
        header=_header(f"{camera_id}_rgb_optical_frame", timestamp_ns)
    )


def _rgb_image(
    camera_id: str,
    camera_index: int,
    timestamp_ns: int,
) -> FakeImage:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[..., 0] = 20 + camera_index
    rgb[..., 1] = 40 + camera_index
    rgb[..., 2] = 60 + camera_index
    return FakeImage(
        header=_header(f"{camera_id}_rgb_optical_frame", timestamp_ns),
        encoding="rgb8",
        width=640,
        height=480,
        array=rgb,
    )


def _depth_image(
    camera_id: str,
    camera_index: int,
    timestamp_ns: int,
) -> FakeImage:
    depth = np.full((480, 640), 0.5 + camera_index * 0.1, dtype=np.float32)
    depth[0, 0] = np.nan
    return FakeImage(
        header=_header(f"{camera_id}_depth_optical_frame", timestamp_ns),
        encoding="32FC1",
        width=640,
        height=480,
        array=depth,
    )


def _header(frame_id: str, timestamp_ns: int) -> SimpleNamespace:
    sec, nanosec = divmod(timestamp_ns, 1_000_000_000)
    return SimpleNamespace(
        frame_id=frame_id,
        stamp=SimpleNamespace(sec=sec, nanosec=nanosec),
    )


def _expected_topics() -> set[str]:
    return {
        topic
        for camera_id in CAMERA_IDS
        for topic in (
            f"/{camera_id}/{camera_id}/image_raw",
            f"/{camera_id}/{camera_id}/depth/image_raw",
            f"/{camera_id}/{camera_id}/camera_info",
        )
    }


def _assert_runtime_cleaned(runtime: FakeRosRuntime) -> None:
    assert runtime.nodes
    assert all(node.destroyed for node in runtime.nodes)
    assert all(context.shutdown for context in runtime.contexts)
    assert all(executor.removed for executor in runtime.executors)
    assert all(executor.shutdown_called for executor in runtime.executors)

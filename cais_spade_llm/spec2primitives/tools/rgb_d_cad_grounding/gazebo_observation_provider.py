"""Capture one request-scoped live Gazebo RGB-D observation bundle."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cais_spade_llm.spec2primitives.tools.observation_context import (
    CAMERA_IDS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_BUNDLE_SKEW_NS,
    MAX_RGB_DEPTH_SKEW_NS,
    CameraCalibration,
    CameraObservation,
    ObservationBundle,
    write_observation_bundle,
)

_BUFFER_SIZE = 10
_SPIN_INTERVAL_SEC = 0.05


class GazeboObservationProviderError(RuntimeError):
    """Report a request-scoped live observation capture failure."""

    def __init__(self, reason: str, message: str) -> None:
        """Initialize the failure with its stable reason and operator message."""
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class _RosDependencies:
    rclpy: Any
    context_type: Any
    executor_type: Any
    image_type: Any
    camera_info_type: Any
    bridge_type: Any
    bridge_error_type: type[Exception]
    sensor_data_qos: Any
    no_signal_handlers: Any


@dataclass(frozen=True)
class _ImageSample:
    timestamp_ns: int
    frame: str
    array: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class _ImagePair:
    rgb: _ImageSample
    depth: _ImageSample

    @property
    def first_timestamp_ns(self) -> int:
        return min(self.rgb.timestamp_ns, self.depth.timestamp_ns)

    @property
    def last_timestamp_ns(self) -> int:
        return max(self.rgb.timestamp_ns, self.depth.timestamp_ns)


@dataclass
class _CameraSamples:
    rgb: deque[_ImageSample] = field(
        default_factory=lambda: deque(maxlen=_BUFFER_SIZE)
    )
    depth: deque[_ImageSample] = field(
        default_factory=lambda: deque(maxlen=_BUFFER_SIZE)
    )
    calibration: CameraCalibration | None = None


class _ObservationCollector:
    """Collect only messages delivered during one capture request."""

    def __init__(self, bridge: Any, bridge_error_type: type[Exception]) -> None:
        self._bridge = bridge
        self._bridge_error_type = bridge_error_type
        self._samples = {camera_id: _CameraSamples() for camera_id in CAMERA_IDS}
        self.error: GazeboObservationProviderError | None = None

    def add_rgb(self, camera_id: str, message: Any) -> None:
        """Validate and buffer one RGB message from the active request."""
        if self.error is not None:
            return
        try:
            self._samples[camera_id].rgb.append(
                self._image_sample(message, expected_encoding="rgb8", is_depth=False)
            )
        except GazeboObservationProviderError as exc:
            self.error = exc

    def add_depth(self, camera_id: str, message: Any) -> None:
        """Validate and buffer one metric-depth message from the active request."""
        if self.error is not None:
            return
        try:
            self._samples[camera_id].depth.append(
                self._image_sample(
                    message,
                    expected_encoding="32FC1",
                    is_depth=True,
                )
            )
        except GazeboObservationProviderError as exc:
            self.error = exc

    def add_camera_info(self, camera_id: str, message: Any) -> None:
        """Retain the newest approved CameraInfo received during the request."""
        try:
            calibration = _camera_calibration(message)
        except (
            GazeboObservationProviderError,
            AttributeError,
            OverflowError,
            TypeError,
            ValueError,
        ):
            return
        self._samples[camera_id].calibration = calibration

    def build_bundle(self, observation_ref: str) -> ObservationBundle | None:
        """Return the newest complete synchronized bundle, when available."""
        candidates = {
            camera_id: self._image_pairs(camera_id)
            for camera_id in CAMERA_IDS
        }
        if any(not candidates[camera_id] for camera_id in CAMERA_IDS):
            return None

        possible_window_starts = sorted(
            {
                pair.first_timestamp_ns
                for camera_pairs in candidates.values()
                for pair in camera_pairs
            },
            reverse=True,
        )
        for window_start_ns in possible_window_starts:
            window_end_ns = window_start_ns + MAX_BUNDLE_SKEW_NS
            selected: dict[str, _ImagePair] = {}
            for camera_id in CAMERA_IDS:
                matching_pairs = [
                    pair
                    for pair in candidates[camera_id]
                    if pair.first_timestamp_ns >= window_start_ns
                    and pair.last_timestamp_ns <= window_end_ns
                ]
                if not matching_pairs:
                    break
                selected[camera_id] = max(
                    matching_pairs,
                    key=lambda pair: pair.last_timestamp_ns,
                )
            if len(selected) == len(CAMERA_IDS):
                return self._observation_bundle(observation_ref, selected)
        return None

    def calibration_missing(self) -> bool:
        """Return whether any camera lacks calibration approved for its RGB frame."""
        for samples in self._samples.values():
            calibration = samples.calibration
            if calibration is None:
                return True
            if samples.rgb and not any(
                sample.frame == calibration.frame for sample in samples.rgb
            ):
                return True
        return False

    def _image_sample(
        self,
        message: Any,
        *,
        expected_encoding: str,
        is_depth: bool,
    ) -> _ImageSample:
        try:
            encoding = message.encoding
            width = message.width
            height = message.height
            frame = message.header.frame_id
        except AttributeError as exc:
            raise _invalid_message("Image message fields are incomplete.") from exc

        if encoding != expected_encoding:
            raise _invalid_message(
                f"Image encoding must be {expected_encoding}."
            )
        if width != IMAGE_WIDTH or height != IMAGE_HEIGHT:
            raise _invalid_message("Image dimensions must be 640 x 480.")
        if not isinstance(frame, str) or not frame:
            raise _invalid_message("Image frame must be non-empty.")

        timestamp_ns = _message_timestamp_ns(message)
        try:
            converted = self._bridge.imgmsg_to_cv2(
                message,
                desired_encoding="passthrough",
            )
        except self._bridge_error_type as exc:
            raise _invalid_message("Image payload could not be decoded.") from exc
        except (AttributeError, TypeError, ValueError) as exc:
            raise _invalid_message("Image payload could not be decoded.") from exc

        array = np.asarray(converted)
        expected_shape = (
            (IMAGE_HEIGHT, IMAGE_WIDTH)
            if is_depth
            else (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        )
        expected_dtype = np.float32 if is_depth else np.uint8
        if array.shape != expected_shape or array.dtype != expected_dtype:
            raise _invalid_message("Image payload shape or dtype is invalid.")
        return _ImageSample(
            timestamp_ns=timestamp_ns,
            frame=frame,
            array=np.array(array, copy=True, order="C"),
        )

    def _image_pairs(self, camera_id: str) -> list[_ImagePair]:
        samples = self._samples[camera_id]
        calibration = samples.calibration
        if calibration is None:
            return []

        pairs: dict[tuple[int, int], _ImagePair] = {}
        for rgb in samples.rgb:
            if rgb.frame != calibration.frame or not samples.depth:
                continue
            depth = min(
                samples.depth,
                key=lambda sample: abs(sample.timestamp_ns - rgb.timestamp_ns),
            )
            if abs(depth.timestamp_ns - rgb.timestamp_ns) <= MAX_RGB_DEPTH_SKEW_NS:
                pairs[(rgb.timestamp_ns, depth.timestamp_ns)] = _ImagePair(rgb, depth)

        for depth in samples.depth:
            matching_rgb = [
                rgb for rgb in samples.rgb if rgb.frame == calibration.frame
            ]
            if not matching_rgb:
                continue
            rgb = min(
                matching_rgb,
                key=lambda sample: abs(sample.timestamp_ns - depth.timestamp_ns),
            )
            if abs(depth.timestamp_ns - rgb.timestamp_ns) <= MAX_RGB_DEPTH_SKEW_NS:
                pairs[(rgb.timestamp_ns, depth.timestamp_ns)] = _ImagePair(rgb, depth)

        return sorted(
            pairs.values(),
            key=lambda pair: pair.last_timestamp_ns,
            reverse=True,
        )

    def _observation_bundle(
        self,
        observation_ref: str,
        selected: dict[str, _ImagePair],
    ) -> ObservationBundle:
        observations = []
        for camera_id in CAMERA_IDS:
            pair = selected[camera_id]
            calibration = self._samples[camera_id].calibration
            if calibration is None:
                raise AssertionError("Synchronized capture requires calibration.")
            observations.append(
                CameraObservation(
                    camera_id=camera_id,
                    rgb=pair.rgb.array,
                    depth_m=pair.depth.array,
                    rgb_timestamp_ns=pair.rgb.timestamp_ns,
                    depth_timestamp_ns=pair.depth.timestamp_ns,
                    rgb_frame=pair.rgb.frame,
                    depth_frame=pair.depth.frame,
                    camera_calibration=calibration,
                )
            )

        captured_at_ns = max(
            timestamp
            for observation in observations
            for timestamp in (
                observation.rgb_timestamp_ns,
                observation.depth_timestamp_ns,
            )
        )
        return ObservationBundle(
            observation_ref=observation_ref,
            evidence_label="live",
            captured_at_ns=captured_at_ns,
            camera_observations=tuple(observations),
        )


def capture_gazebo_observation(
    observations_root: Path,
    observation_ref: str,
    *,
    timeout_sec: float = 5.0,
) -> Path:
    """Capture and store one explicitly requested live Gazebo observation.

    Args:
        observations_root: Caller-owned interaction observation directory.
        observation_ref: Exact observation ref assigned for this request.
        timeout_sec: Maximum time to wait for a complete fresh capture.

    Returns:
        The atomically created observation bundle directory.

    Raises:
        GazeboObservationProviderError: If live capture cannot produce a bundle.
        ObservationContextError: If the Phase 1.1 bundle contract rejects storage.
        ValueError: If `timeout_sec` is not a finite positive number.
    """
    if (
        isinstance(timeout_sec, bool)
        or not isinstance(timeout_sec, (int, float))
        or not math.isfinite(float(timeout_sec))
        or timeout_sec <= 0
    ):
        raise ValueError("timeout_sec must be a finite positive number.")

    dependencies = _load_ros_dependencies()
    context = dependencies.context_type()
    node = None
    executor = None
    initialized = False
    try:
        dependencies.rclpy.init(
            context=context,
            signal_handler_options=dependencies.no_signal_handlers,
        )
        initialized = True
        node = dependencies.rclpy.create_node(
            "spec2primitives_gazebo_observation",
            context=context,
        )
        executor = dependencies.executor_type(context=context)
        executor.add_node(node)

        collector = _ObservationCollector(
            dependencies.bridge_type(),
            dependencies.bridge_error_type,
        )
        _create_subscriptions(node, collector, dependencies)

        deadline = time.monotonic() + float(timeout_sec)
        while True:
            remaining_sec = deadline - time.monotonic()
            if remaining_sec <= 0:
                break
            executor.spin_once(
                timeout_sec=min(_SPIN_INTERVAL_SEC, remaining_sec)
            )
            if collector.error is not None:
                raise collector.error
            bundle = collector.build_bundle(observation_ref)
            if bundle is not None:
                return write_observation_bundle(observations_root, bundle)

        if collector.calibration_missing():
            raise GazeboObservationProviderError(
                "calibration_missing",
                "Approved CameraInfo is unavailable for one or more required cameras.",
            )
        raise GazeboObservationProviderError(
            "capture_timeout",
            "A complete synchronized four-camera observation was not captured.",
        )
    except GazeboObservationProviderError:
        raise
    except (OSError, RuntimeError) as exc:
        raise GazeboObservationProviderError(
            "ros_unavailable",
            "The request-scoped ROS2 observation capture is unavailable.",
        ) from exc
    finally:
        if executor is not None and node is not None:
            with suppress(RuntimeError):
                executor.remove_node(node)
        if executor is not None:
            with suppress(RuntimeError):
                executor.shutdown(timeout_sec=0)
        if node is not None:
            with suppress(RuntimeError):
                node.destroy_node()
        if initialized:
            with suppress(RuntimeError):
                context.try_shutdown()


def _load_ros_dependencies() -> _RosDependencies:
    try:
        import rclpy
        from cv_bridge import CvBridge, CvBridgeError
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import CameraInfo, Image
    except (ImportError, OSError) as exc:
        raise GazeboObservationProviderError(
            "ros_unavailable",
            "ROS2 image capture dependencies are unavailable.",
        ) from exc

    return _RosDependencies(
        rclpy=rclpy,
        context_type=Context,
        executor_type=SingleThreadedExecutor,
        image_type=Image,
        camera_info_type=CameraInfo,
        bridge_type=CvBridge,
        bridge_error_type=CvBridgeError,
        sensor_data_qos=qos_profile_sensor_data,
        no_signal_handlers=SignalHandlerOptions.NO,
    )


def _create_subscriptions(
    node: Any,
    collector: _ObservationCollector,
    dependencies: _RosDependencies,
) -> None:
    for camera_id in CAMERA_IDS:
        node.create_subscription(
            dependencies.image_type,
            f"/{camera_id}/{camera_id}/image_raw",
            _camera_callback(collector.add_rgb, camera_id),
            dependencies.sensor_data_qos,
        )
        node.create_subscription(
            dependencies.image_type,
            f"/{camera_id}/{camera_id}/depth/image_raw",
            _camera_callback(collector.add_depth, camera_id),
            dependencies.sensor_data_qos,
        )
        node.create_subscription(
            dependencies.camera_info_type,
            f"/{camera_id}/{camera_id}/camera_info",
            _camera_callback(collector.add_camera_info, camera_id),
            dependencies.sensor_data_qos,
        )


def _camera_callback(
    operation: Callable[[str, Any], None],
    camera_id: str,
) -> Callable[[Any], None]:
    def callback(message: Any) -> None:
        operation(camera_id, message)

    return callback


def _camera_calibration(message: Any) -> CameraCalibration:
    _message_timestamp_ns(message)
    if message.width != IMAGE_WIDTH or message.height != IMAGE_HEIGHT:
        raise ValueError("CameraInfo dimensions must be 640 x 480.")
    frame = message.header.frame_id
    distortion_model = message.distortion_model
    if not isinstance(frame, str) or not frame:
        raise ValueError("CameraInfo frame must be non-empty.")
    if not isinstance(distortion_model, str) or not distortion_model:
        raise ValueError("CameraInfo distortion_model must be non-empty.")

    calibration = CameraCalibration(
        width=message.width,
        height=message.height,
        frame=frame,
        distortion_model=distortion_model,
        D=_finite_numeric_tuple(message.d, minimum_length=1),
        K=_finite_numeric_tuple(message.k, expected_length=9),
        R=_finite_numeric_tuple(message.r, expected_length=9),
        P=_finite_numeric_tuple(message.p, expected_length=12),
        extrinsics_available=False,
    )
    return calibration


def _finite_numeric_tuple(
    values: Any,
    *,
    expected_length: int | None = None,
    minimum_length: int | None = None,
) -> tuple[float, ...]:
    items = tuple(values)
    if expected_length is not None and len(items) != expected_length:
        raise ValueError("CameraInfo calibration length is invalid.")
    if minimum_length is not None and len(items) < minimum_length:
        raise ValueError("CameraInfo calibration is empty.")
    result = []
    for value in items:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
        ):
            raise ValueError("CameraInfo calibration must contain finite values.")
        result.append(float(value))
    return tuple(result)


def _message_timestamp_ns(message: Any) -> int:
    try:
        sec = message.header.stamp.sec
        nanosec = message.header.stamp.nanosec
    except AttributeError as exc:
        raise _invalid_message("Message timestamp is incomplete.") from exc
    if (
        not isinstance(sec, int)
        or isinstance(sec, bool)
        or sec < 0
        or not isinstance(nanosec, int)
        or isinstance(nanosec, bool)
        or nanosec < 0
        or nanosec >= 1_000_000_000
    ):
        raise _invalid_message("Message timestamp is invalid.")
    return sec * 1_000_000_000 + nanosec


def _invalid_message(message: str) -> GazeboObservationProviderError:
    return GazeboObservationProviderError("invalid_message", message)

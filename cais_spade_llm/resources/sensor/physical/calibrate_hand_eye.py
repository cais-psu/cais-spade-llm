"""Guided ChArUco eye-in-hand calibration for the wrist RealSense."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH_M = 0.025
MARKER_LENGTH_M = 0.018
MINIMUM_ACCEPTED_POSES = 20
MAXIMUM_REPROJECTION_ERROR_PX = 1.0
MAXIMUM_TRANSLATION_RMS_M = 0.005
MAXIMUM_ROTATION_RMS_DEG = 1.0
MAXIMUM_SOLVER_TRANSLATION_DISAGREEMENT_M = 0.001
MAXIMUM_SOLVER_ROTATION_DISAGREEMENT_DEG = 0.1
GEAR_HEIGHT_M = 0.020
MINIMUM_TABLE_PLANE_FRAMES = 10
MAXIMUM_TABLE_PLANE_MAD_M = 0.002
MAXIMUM_TABLE_PLANE_CLASS_DISAGREEMENT_M = 0.005


def _opencv() -> Any:
    try:
        import cv2 as cv
    except ImportError as exc:
        raise RuntimeError(
            "opencv-contrib-python-headless is required; run poetry install"
        ) from exc
    if not hasattr(cv, "aruco"):
        raise RuntimeError("OpenCV was installed without the aruco contrib module")
    return cv


def _charuco_board(cv: Any) -> Any:
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    return cv.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
        dictionary,
    )


def generate_board(path: str | Path, *, pixels_per_mm: int = 10) -> Path:
    """Generate the fixed 5x7 ChArUco board image."""
    cv = _opencv()
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    width = SQUARES_X * 25 * int(pixels_per_mm)
    height = SQUARES_Y * 25 * int(pixels_per_mm)
    image = _charuco_board(cv).generateImage((width, height), marginSize=0, borderBits=1)
    if not cv.imwrite(str(output), image):
        raise RuntimeError(f"failed to write ChArUco board: {output}")
    logger.info(
        "Generated %s. Print at exactly 125 x 175 mm, disable fit-to-page, and verify one square is 25 mm with a ruler.",
        output,
    )
    return output


def _matrix_from_transform_message(message: Any) -> np.ndarray:
    from .realsense_pose_estimator import quaternion_matrix

    transform = getattr(message, "transform", message)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_matrix(
        (
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        )
    )
    matrix[:3, 3] = [
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    ]
    return matrix


def _quaternion_from_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    from scipy.spatial.transform import Rotation

    quaternion = Rotation.from_matrix(rotation).as_quat()
    return tuple(float(value) for value in quaternion)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_yaml_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _detect_board_pose(image: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray) -> tuple[np.ndarray, float, int]:
    cv = _opencv()
    board = _charuco_board(cv)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corners, ids, _rejected = cv.aruco.detectMarkers(
        gray,
        board.getDictionary(),
    )
    if ids is None or len(ids) < 4:
        raise RuntimeError("fewer than four ChArUco markers are visible")
    _count, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(
        corners,
        ids,
        gray,
        board,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion,
    )
    if charuco_ids is None or len(charuco_ids) < 6:
        raise RuntimeError("fewer than six ChArUco corners are visible")
    object_points = board.getChessboardCorners()[charuco_ids.reshape(-1)].astype(np.float32)
    image_points = charuco_corners.reshape(-1, 2).astype(np.float32)
    ok, rotation_vector, translation_vector = cv.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
    )
    if not ok:
        raise RuntimeError("solvePnP failed for the ChArUco board")
    projected, _ = cv.projectPoints(
        object_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion,
    )
    error = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    camera_to_board = np.eye(4, dtype=np.float64)
    camera_to_board[:3, :3] = cv.Rodrigues(rotation_vector)[0]
    camera_to_board[:3, 3] = translation_vector.reshape(3)
    return camera_to_board, float(np.median(error)), int(len(charuco_ids))


class _CaptureNode:
    def __init__(
        self,
        *,
        camera_role: str,
        world_frame: str,
        tool_frame: str,
        camera_link_frame: str,
        camera_optical_frame: str,
        color_topic: str,
        camera_info_topic: str,
    ) -> None:
        import rclpy
        import tf2_ros
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image

        self.rclpy = rclpy
        self.tf2_ros = tf2_ros
        self.node = Node(f"{camera_role}_realsense_calibration_capture")
        self.camera_role = camera_role
        self.world_frame = world_frame
        self.tool_frame = tool_frame
        self.camera_link_frame = camera_link_frame
        self.camera_optical_frame = camera_optical_frame
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
        self.image: np.ndarray | None = None
        self.camera_matrix: np.ndarray | None = None
        self.distortion: np.ndarray | None = None
        self.node.create_subscription(
            Image,
            color_topic,
            self._on_image,
            10,
        )
        self.node.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._on_info,
            10,
        )

    def _on_image(self, message: Any) -> None:
        self.image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

    def _on_info(self, message: Any) -> None:
        self.camera_matrix = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
        self.distortion = np.asarray(message.d, dtype=np.float64)

    def spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def transform(self, target: str, source: str) -> np.ndarray:
        try:
            message = self.tf_buffer.lookup_transform(
                target,
                source,
                self.rclpy.time.Time(),
                timeout=self.rclpy.duration.Duration(seconds=1.0),
            )
        except (
            self.tf2_ros.LookupException,
            self.tf2_ros.ConnectivityException,
            self.tf2_ros.ExtrapolationException,
        ) as exc:
            if self.camera_role == "ur5e" and target == "world" and source == "tool0":
                raise RuntimeError(
                    "TF world -> tool0 is unavailable. The read-only UR5e calibration monitor "
                    "did not become ready; check its RTDE/state-publisher status and retry. "
                    "Keep the teach pendant in Local Control. Save Pose + Capture does not "
                    "request robot motion."
                ) from exc
            raise RuntimeError(f"TF {target} -> {source} is unavailable: {exc}") from exc
        return _matrix_from_transform_message(message.transform)


def capture_samples(  # noqa: PLR0913 - camera topics and TF frames must remain explicit.
    output_path: str | Path,
    *,
    accepted_poses: int = 25,
    world_frame: str = "world",
    tool_frame: str = "tool0",
    camera_link_frame: str = "camera_link",
    camera_optical_frame: str = "camera_color_optical_frame",
    camera_role: str = "ur5e",
    color_topic: str = "/camera/camera/color/image_raw",
    camera_info_topic: str = "/camera/camera/color/camera_info",
) -> Path:
    """Interactively capture stationary board observations and robot poses."""
    try:
        import rclpy
    except ImportError as exc:
        raise RuntimeError("ROS2 Python is unavailable; source the ROS2 workspace") from exc
    if accepted_poses < MINIMUM_ACCEPTED_POSES:
        raise ValueError(f"at least {MINIMUM_ACCEPTED_POSES} accepted poses are required")
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    capture = _CaptureNode(
        camera_role=camera_role,
        world_frame=world_frame,
        tool_frame=tool_frame,
        camera_link_frame=camera_link_frame,
        camera_optical_frame=camera_optical_frame,
        color_topic=color_topic,
        camera_info_topic=camera_info_topic,
    )
    samples: list[dict[str, Any]] = []
    try:
        capture.spin_for(2.0)
        logger.info(
            "Collecting %d accepted stationary poses. Move the wrist between samples and vary rotation around all axes.",
            accepted_poses,
        )
        while len(samples) < accepted_poses:
            input(f"Pose {len(samples) + 1}/{accepted_poses}: stop the UR5e, then press Enter...")
            capture.spin_for(0.4)
            if capture.image is None or capture.camera_matrix is None or capture.distortion is None:
                logger.warning("Rejected: color image or CameraInfo is unavailable")
                continue
            tool_before = capture.transform(world_frame, tool_frame)
            capture.spin_for(0.5)
            tool_after = capture.transform(world_frame, tool_frame)
            translation_m = float(np.linalg.norm(tool_after[:3, 3] - tool_before[:3, 3]))
            rotation_delta = tool_before[:3, :3].T @ tool_after[:3, :3]
            rotation_deg = math.degrees(
                math.acos(float(np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0)))
            )
            if translation_m > 0.001 or rotation_deg > 0.5:
                logger.warning(
                    "Rejected: UR5e moved %.2f mm / %.2f deg during capture",
                    translation_m * 1000.0,
                    rotation_deg,
                )
                continue
            try:
                camera_to_board, reprojection_error, corner_count = _detect_board_pose(
                    capture.image,
                    capture.camera_matrix,
                    capture.distortion,
                )
                link_to_optical = capture.transform(camera_link_frame, camera_optical_frame)
            except RuntimeError as exc:
                logger.warning("Rejected: %s", exc)
                continue
            samples.append(
                {
                    "base_to_tool": tool_after.tolist(),
                    "camera_to_board": camera_to_board.tolist(),
                    "camera_link_to_optical": link_to_optical.tolist(),
                    "reprojection_error_px": reprojection_error,
                    "corner_count": corner_count,
                    "captured_at": time.time(),
                }
            )
            logger.info(
                "Accepted pose %d/%d: %d corners, %.3f px median reprojection error",
                len(samples),
                accepted_poses,
                corner_count,
                reprojection_error,
            )
    finally:
        capture.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    _atomic_json_write(
        output,
        {
            "camera_role": camera_role,
            "world_frame": world_frame,
            "tool_frame": tool_frame,
            "samples": samples,
        },
    )
    return output


def capture_one_sample(  # noqa: PLR0913 - camera topics and TF frames must remain explicit.
    output_path: str | Path,
    *,
    camera_role: str,
    world_frame: str,
    tool_frame: str,
    camera_link_frame: str,
    camera_optical_frame: str,
    color_topic: str,
    camera_info_topic: str,
    stationary_camera: bool = False,
) -> dict[str, Any]:
    """Capture one reviewed stationary ChArUco observation atomically."""
    try:
        import rclpy
    except ImportError as exc:
        raise RuntimeError("ROS2 Python is unavailable; source the ROS2 workspace") from exc

    output = Path(output_path).expanduser()
    try:
        existing = json.loads(output.read_text(encoding="utf-8"))
    except FileNotFoundError:
        existing = {}
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read calibration samples: {output}") from exc
    samples = existing.get("samples", []) if isinstance(existing, dict) else []
    if not isinstance(samples, list):
        raise RuntimeError(f"calibration samples are invalid: {output}")

    rclpy.init()
    capture = _CaptureNode(
        camera_role=camera_role,
        world_frame=world_frame,
        tool_frame=tool_frame,
        camera_link_frame=camera_link_frame,
        camera_optical_frame=camera_optical_frame,
        color_topic=color_topic,
        camera_info_topic=camera_info_topic,
    )
    try:
        capture.spin_for(2.0)
        if capture.image is None or capture.camera_matrix is None or capture.distortion is None:
            raise RuntimeError("color image or CameraInfo is unavailable")
        tool_after: np.ndarray | None = None
        if not stationary_camera:
            tool_before = capture.transform(world_frame, tool_frame)
            capture.spin_for(0.5)
            tool_after = capture.transform(world_frame, tool_frame)
            translation_m = float(np.linalg.norm(tool_after[:3, 3] - tool_before[:3, 3]))
            rotation_delta = tool_before[:3, :3].T @ tool_after[:3, :3]
            rotation_deg = math.degrees(
                math.acos(float(np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0)))
            )
            if translation_m > 0.001 or rotation_deg > 0.5:
                raise RuntimeError(
                    f"{camera_role} moved {translation_m * 1000.0:.2f} mm / "
                    f"{rotation_deg:.2f} deg during capture"
                )
        camera_to_board, reprojection_error, corner_count = _detect_board_pose(
            capture.image,
            capture.camera_matrix,
            capture.distortion,
        )
        link_to_optical = capture.transform(camera_link_frame, camera_optical_frame)
        sample: dict[str, Any] = {
            "camera_to_board": camera_to_board.tolist(),
            "camera_link_to_optical": link_to_optical.tolist(),
            "reprojection_error_px": reprojection_error,
            "corner_count": corner_count,
            "captured_at": time.time(),
        }
        if tool_after is not None:
            sample["base_to_tool"] = tool_after.tolist()
        samples.append(sample)
        _atomic_json_write(
            output,
            {
                "camera_role": camera_role,
                "world_frame": world_frame,
                "tool_frame": tool_frame,
                "stationary_camera": stationary_camera,
                "samples": samples,
            },
        )
        return {"sample_count": len(samples), **sample}
    finally:
        capture.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _rotation_rms_deg(rotations: list[np.ndarray]) -> float:
    from scipy.spatial.transform import Rotation

    scipy_rotations = Rotation.from_matrix(np.asarray(rotations))
    mean_rotation = scipy_rotations.mean()
    errors = (mean_rotation.inv() * scipy_rotations).magnitude()
    return float(np.sqrt(np.mean(np.square(np.degrees(errors)))))


def _solve_hand_eye(
    cv: Any,
    rotations_gripper_to_base: list[np.ndarray],
    translations_gripper_to_base: list[np.ndarray],
    rotations_target_to_camera: list[np.ndarray],
    translations_target_to_camera: list[np.ndarray],
    *,
    method: int,
) -> np.ndarray:
    rotation, translation = cv.calibrateHandEye(
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        method=method,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation).reshape(3)
    return transform


def _transform_disagreement(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(first) @ second
    translation_m = float(np.linalg.norm(delta[:3, 3]))
    cosine = float(np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
    rotation_deg = math.degrees(math.acos(cosine))
    return translation_m, rotation_deg


def solve_calibration(
    samples_path: str | Path,
    output_path: str | Path,
    *,
    camera_role: str = "ur5e",
    parent_frame: str = "tool0",
) -> dict[str, Any]:
    """Solve PARK hand-eye calibration and overwrite only when all gates pass."""
    cv = _opencv()
    samples_payload = json.loads(Path(samples_path).expanduser().read_text(encoding="utf-8"))
    samples = samples_payload.get("samples", [])
    if not isinstance(samples, list) or len(samples) < MINIMUM_ACCEPTED_POSES:
        raise RuntimeError(f"at least {MINIMUM_ACCEPTED_POSES} accepted poses are required")

    base_to_tool = [np.asarray(row["base_to_tool"], dtype=np.float64) for row in samples]
    camera_to_board = [np.asarray(row["camera_to_board"], dtype=np.float64) for row in samples]
    rotations_gripper_to_base = [matrix[:3, :3] for matrix in base_to_tool]
    translations_gripper_to_base = [matrix[:3, 3] for matrix in base_to_tool]
    rotations_target_to_camera = [matrix[:3, :3] for matrix in camera_to_board]
    translations_target_to_camera = [matrix[:3, 3] for matrix in camera_to_board]
    tool_to_optical = _solve_hand_eye(
        cv,
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        method=cv.CALIB_HAND_EYE_PARK,
    )
    horaud_tool_to_optical = _solve_hand_eye(
        cv,
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        method=cv.CALIB_HAND_EYE_HORAUD,
    )
    solver_translation_disagreement, solver_rotation_disagreement = (
        _transform_disagreement(tool_to_optical, horaud_tool_to_optical)
    )
    if (
        solver_translation_disagreement > MAXIMUM_SOLVER_TRANSLATION_DISAGREEMENT_M
        or solver_rotation_disagreement > MAXIMUM_SOLVER_ROTATION_DISAGREEMENT_DEG
    ):
        raise RuntimeError(
            "calibration rejected: PARK/HORAUD disagreement="
            f"{solver_translation_disagreement * 1000.0:.3f} mm / "
            f"{solver_rotation_disagreement:.3f} deg"
        )

    board_poses = [
        base_tool @ tool_to_optical @ camera_board
        for base_tool, camera_board in zip(base_to_tool, camera_to_board, strict=True)
    ]
    board_translations = np.asarray([pose[:3, 3] for pose in board_poses])
    mean_translation = np.mean(board_translations, axis=0)
    translation_rms = float(
        np.sqrt(np.mean(np.sum(np.square(board_translations - mean_translation), axis=1)))
    )
    rotation_rms = _rotation_rms_deg([pose[:3, :3] for pose in board_poses])
    reprojection_error = float(
        np.median([float(row["reprojection_error_px"]) for row in samples])
    )
    accepted = (
        reprojection_error <= MAXIMUM_REPROJECTION_ERROR_PX
        and translation_rms <= MAXIMUM_TRANSLATION_RMS_M
        and rotation_rms <= MAXIMUM_ROTATION_RMS_DEG
    )
    if not accepted:
        raise RuntimeError(
            "calibration rejected: "
            f"reprojection={reprojection_error:.3f} px, "
            f"translation RMS={translation_rms * 1000.0:.3f} mm, "
            f"rotation RMS={rotation_rms:.3f} deg"
        )

    link_to_optical_samples = [
        np.asarray(row["camera_link_to_optical"], dtype=np.float64) for row in samples
    ]
    link_to_optical = link_to_optical_samples[0]
    tool_to_link = tool_to_optical @ np.linalg.inv(link_to_optical)
    optical_quaternion = _quaternion_from_matrix(tool_to_optical[:3, :3])
    link_quaternion = _quaternion_from_matrix(tool_to_link[:3, :3])

    def _transform_payload(matrix: np.ndarray, quaternion: tuple[float, float, float, float]) -> dict[str, Any]:
        return {
            "translation": {
                "x": float(matrix[0, 3]),
                "y": float(matrix[1, 3]),
                "z": float(matrix[2, 3]),
            },
            "quaternion": {
                "x": quaternion[0],
                "y": quaternion[1],
                "z": quaternion[2],
                "w": quaternion[3],
            },
        }

    payload = {
        "calibration_id": str(uuid.uuid4()),
        "camera_role": camera_role,
        "parent_frame": parent_frame,
        "method": "CALIB_HAND_EYE_PARK",
        "board": {
            "dictionary": "DICT_4X4_50",
            "squares_x": SQUARES_X,
            "squares_y": SQUARES_Y,
            "square_length_m": SQUARE_LENGTH_M,
            "marker_length_m": MARKER_LENGTH_M,
        },
        "accepted_pose_count": len(samples),
        "tool0_to_camera_color_optical_frame": _transform_payload(
            tool_to_optical,
            optical_quaternion,
        ),
        "tool0_to_camera_link": _transform_payload(tool_to_link, link_quaternion),
        "parent_to_camera_color_optical_frame": _transform_payload(
            tool_to_optical,
            optical_quaternion,
        ),
        "parent_to_camera_link": _transform_payload(tool_to_link, link_quaternion),
        "validation": {
            "accepted": True,
            "cross_check_method": "CALIB_HAND_EYE_HORAUD",
            "solver_translation_disagreement_m": solver_translation_disagreement,
            "solver_rotation_disagreement_deg": solver_rotation_disagreement,
            "median_reprojection_error_px": reprojection_error,
            "fixed_board_translation_rms_m": translation_rms,
            "fixed_board_rotation_rms_deg": rotation_rms,
        },
    }
    output = Path(output_path).expanduser()
    _atomic_yaml_write(output, payload)
    logger.info("Accepted calibration written to %s", output)
    return payload


def _transform_from_xyz_rpy(pose: dict[str, Any]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler(
        "xyz",
        [float(pose["roll"]), float(pose["pitch"]), float(pose["yaw"])],
    ).as_matrix()
    transform[:3, 3] = [float(pose["x"]), float(pose["y"]), float(pose["z"])]
    return transform


def solve_stationary_calibration(
    samples_path: str | Path,
    output_path: str | Path,
    *,
    board_world_pose: dict[str, Any],
    camera_role: str = "stationary",
    parent_frame: str = "world",
) -> dict[str, Any]:
    """Solve a fixed camera extrinsic from a surveyed ChArUco board pose."""
    samples_payload = json.loads(Path(samples_path).expanduser().read_text(encoding="utf-8"))
    samples = samples_payload.get("samples", [])
    if not isinstance(samples, list) or len(samples) < 10:
        raise RuntimeError("at least 10 accepted stationary camera observations are required")
    if not bool(board_world_pose.get("configured", False)):
        raise RuntimeError("the stationary ChArUco board world pose is not configured")

    world_to_board = _transform_from_xyz_rpy(board_world_pose)
    world_to_optical = [
        world_to_board @ np.linalg.inv(np.asarray(row["camera_to_board"], dtype=np.float64))
        for row in samples
    ]
    world_to_link = [
        optical @ np.linalg.inv(np.asarray(row["camera_link_to_optical"], dtype=np.float64))
        for optical, row in zip(world_to_optical, samples, strict=True)
    ]
    translations = np.asarray([transform[:3, 3] for transform in world_to_link])
    mean_translation = np.mean(translations, axis=0)
    translation_rms = float(
        np.sqrt(np.mean(np.sum(np.square(translations - mean_translation), axis=1)))
    )
    rotation_rms = _rotation_rms_deg([transform[:3, :3] for transform in world_to_link])
    reprojection_error = float(
        np.median([float(row["reprojection_error_px"]) for row in samples])
    )
    if (
        reprojection_error > MAXIMUM_REPROJECTION_ERROR_PX
        or translation_rms > MAXIMUM_TRANSLATION_RMS_M
        or rotation_rms > MAXIMUM_ROTATION_RMS_DEG
    ):
        raise RuntimeError(
            "stationary calibration rejected: "
            f"reprojection={reprojection_error:.3f} px, "
            f"translation RMS={translation_rms * 1000.0:.3f} mm, "
            f"rotation RMS={rotation_rms:.3f} deg"
        )

    from scipy.spatial.transform import Rotation

    mean_link_rotation = Rotation.from_matrix(
        np.asarray([transform[:3, :3] for transform in world_to_link])
    ).mean()
    mean_world_to_link = np.eye(4, dtype=np.float64)
    mean_world_to_link[:3, :3] = mean_link_rotation.as_matrix()
    mean_world_to_link[:3, 3] = mean_translation
    link_to_optical = np.asarray(samples[0]["camera_link_to_optical"], dtype=np.float64)
    mean_world_to_optical = mean_world_to_link @ link_to_optical

    def _payload(transform: np.ndarray) -> dict[str, Any]:
        quaternion = _quaternion_from_matrix(transform[:3, :3])
        return {
            "translation": {
                "x": float(transform[0, 3]),
                "y": float(transform[1, 3]),
                "z": float(transform[2, 3]),
            },
            "quaternion": dict(zip(("x", "y", "z", "w"), quaternion, strict=True)),
        }

    payload = {
        "calibration_id": str(uuid.uuid4()),
        "camera_role": camera_role,
        "parent_frame": parent_frame,
        "method": "surveyed_charuco_extrinsic",
        "board_world_pose": dict(board_world_pose),
        "accepted_pose_count": len(samples),
        "parent_to_camera_color_optical_frame": _payload(mean_world_to_optical),
        "parent_to_camera_link": _payload(mean_world_to_link),
        "validation": {
            "accepted": True,
            "median_reprojection_error_px": reprojection_error,
            "fixed_board_translation_rms_m": translation_rms,
            "fixed_board_rotation_rms_deg": rotation_rms,
        },
    }
    output = Path(output_path).expanduser()
    _atomic_yaml_write(output, payload)
    logger.info("Accepted stationary calibration written to %s", output)
    return payload


def estimate_table_plane(
    frames: list[list[dict[str, Any]]],
    *,
    world_frame: str = "world",
    gear_height_m: float = GEAR_HEIGHT_M,
) -> dict[str, Any]:
    """Estimate a horizontal table plane from stationary SG/MG detections."""
    if len(frames) < MINIMUM_TABLE_PLANE_FRAMES:
        raise RuntimeError(
            f"at least {MINIMUM_TABLE_PLANE_FRAMES} accepted detection frames are required"
        )
    samples: list[tuple[str, float]] = []
    accepted_frames = 0
    for rows in frames:
        frame_values: list[float] = []
        for row in rows:
            part_name = str(row.get("part_name") or "").strip()
            if part_name not in {"SG", "MG"} or row.get("frame_id") != world_frame:
                continue
            raw_center_z = row.get("observed_center_z", row.get("z"))
            try:
                surface_z = float(raw_center_z) - float(gear_height_m) * 0.5
            except (TypeError, ValueError):
                continue
            if not math.isfinite(surface_z):
                continue
            samples.append((part_name, surface_z))
            frame_values.append(surface_z)
        if frame_values:
            accepted_frames += 1
    if accepted_frames < MINIMUM_TABLE_PLANE_FRAMES or len(samples) < MINIMUM_TABLE_PLANE_FRAMES:
        raise RuntimeError(
            f"only {accepted_frames} usable world-frame detection frames were collected; "
            f"{MINIMUM_TABLE_PLANE_FRAMES} required"
        )

    values = np.asarray([value for _, value in samples], dtype=np.float64)
    preliminary_median = float(np.median(values))
    preliminary_mad = float(np.median(np.abs(values - preliminary_median)))
    outlier_threshold = max(3.0 * preliminary_mad, MAXIMUM_TABLE_PLANE_MAD_M)
    filtered_samples = [
        (part_name, value)
        for part_name, value in samples
        if abs(value - preliminary_median) <= outlier_threshold
    ]
    filtered = [value for _, value in filtered_samples]
    if len(filtered) < MINIMUM_TABLE_PLANE_FRAMES:
        raise RuntimeError("table-plane outlier rejection left too few samples")
    surface_z_m = float(np.median(filtered))
    class_medians = {}
    for part_name in ("SG", "MG"):
        part_values = [
            value for sample_part, value in filtered_samples if sample_part == part_name
        ]
        if part_values:
            class_medians[part_name] = float(np.median(part_values))
    if {"SG", "MG"}.issubset(class_medians):
        disagreement_m = abs(class_medians["SG"] - class_medians["MG"])
        if disagreement_m > MAXIMUM_TABLE_PLANE_CLASS_DISAGREEMENT_M:
            raise RuntimeError(
                "table-plane calibration rejected: SG/MG median disagreement="
                f"{disagreement_m * 1000.0:.3f} mm exceeds 5 mm"
            )
    else:
        disagreement_m = None

    mad_m = float(np.median(np.abs(np.asarray(filtered) - surface_z_m)))
    if mad_m > MAXIMUM_TABLE_PLANE_MAD_M:
        raise RuntimeError(
            f"table-plane calibration rejected: MAD={mad_m * 1000.0:.3f} mm exceeds 2 mm"
        )

    timestamp = time.time()
    return {
        "accepted": True,
        "world_frame": world_frame,
        "surface_z_m": surface_z_m,
        "frame_count": accepted_frames,
        "sample_count": len(filtered),
        "rejected_sample_count": len(samples) - len(filtered),
        "mad_m": mad_m,
        "class_medians_m": class_medians,
        "class_disagreement_m": disagreement_m,
        "timestamp": timestamp,
    }


def write_table_plane_calibration(
    calibration_path: str | Path,
    table_plane: dict[str, Any],
) -> dict[str, Any]:
    """Atomically add an accepted table plane to a hand-eye calibration file."""
    if not bool(table_plane.get("accepted", False)):
        raise RuntimeError("refusing to write a rejected table-plane calibration")
    path = Path(calibration_path).expanduser()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise RuntimeError(f"hand-eye calibration not found: {path}") from exc
    if not bool((payload.get("validation") or {}).get("accepted", False)):
        raise RuntimeError("hand-eye calibration is not marked accepted")
    updated = dict(payload)
    updated["table_plane"] = dict(table_plane)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    logger.info(
        "Accepted table plane z=%.6f m (MAD %.3f mm) written to %s",
        float(table_plane["surface_z_m"]),
        float(table_plane["mad_m"]) * 1000.0,
        path,
    )
    return updated


def capture_table_plane(
    calibration_path: str | Path,
    *,
    frame_count: int = MINIMUM_TABLE_PLANE_FRAMES,
    maximum_attempts: int = 30,
    world_frame: str = "world",
    measurement_service: str = "/perception/ur5e/table_plane_measurement",
) -> dict[str, Any]:
    """Collect stationary non-executable measurements and store an accepted table plane."""
    try:
        import rclpy
        from rclpy.node import Node
        from std_srvs.srv import Trigger
    except ImportError as exc:
        raise RuntimeError("ROS2 Python is unavailable; source the ROS2 workspace") from exc
    if frame_count < MINIMUM_TABLE_PLANE_FRAMES:
        raise ValueError(f"at least {MINIMUM_TABLE_PLANE_FRAMES} frames are required")
    if maximum_attempts < frame_count:
        raise ValueError("maximum attempts must be at least the requested frame count")

    rclpy.init()
    node = Node("ur5e_realsense_table_plane_capture")
    service_name = str(measurement_service or "").strip()
    if not service_name:
        raise ValueError("table-plane measurement service is empty")
    client = node.create_client(Trigger, service_name)
    frames: list[list[dict[str, Any]]] = []
    try:
        if not client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError(
                f"{service_name} is unavailable; restart physical perception and try again"
            )
        logger.info(
            "Collecting %d stationary table-plane frames. Keep SG or MG flat on the table and do not move the UR5e.",
            frame_count,
        )
        attempts = 0
        while len(frames) < frame_count and attempts < maximum_attempts:
            attempts += 1
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)
            if not future.done() or future.result() is None:
                logger.warning("Attempt %d rejected: %s timed out", attempts, service_name)
                continue
            response = future.result()
            if not response.success:
                logger.warning("Attempt %d rejected: %s", attempts, response.message)
                continue
            try:
                rows = json.loads(response.message)
            except json.JSONDecodeError:
                logger.warning(
                    "Attempt %d rejected: %s returned invalid JSON",
                    attempts,
                    service_name,
                )
                continue
            usable = [
                row
                for row in rows
                if isinstance(row, dict)
                and row.get("part_name") in {"SG", "MG"}
                and row.get("frame_id") == world_frame
            ] if isinstance(rows, list) else []
            if not usable:
                logger.warning("Attempt %d rejected: no SG/MG world-frame detection", attempts)
                continue
            frames.append(usable)
            logger.info("Accepted table-plane frame %d/%d", len(frames), frame_count)
            time.sleep(0.2)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if len(frames) < frame_count:
        raise RuntimeError(
            f"table-plane calibration collected only {len(frames)}/{frame_count} frames"
        )
    table_plane = estimate_table_plane(frames, world_frame=world_frame)
    write_table_plane_calibration(calibration_path, table_plane)
    return table_plane


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    board = subparsers.add_parser("board", help="generate the printable ChArUco board")
    board.add_argument("output")
    capture = subparsers.add_parser("capture", help="capture stationary eye-in-hand samples")
    capture.add_argument("output")
    capture.add_argument("--poses", type=int, default=25)
    capture_once = subparsers.add_parser(
        "capture-once",
        help="capture one reviewed ChArUco observation for the Perception page",
    )
    capture_once.add_argument("output")
    for command in (capture, capture_once):
        command.add_argument("--camera-role", default="ur5e")
        command.add_argument("--world-frame", default="world")
        command.add_argument("--tool-frame", default="tool0")
        command.add_argument("--camera-link-frame", default="camera_link")
        command.add_argument(
            "--camera-optical-frame",
            default="camera_color_optical_frame",
        )
        command.add_argument("--color-topic", default="/camera/camera/color/image_raw")
        command.add_argument(
            "--camera-info-topic",
            default="/camera/camera/color/camera_info",
        )
    capture_once.add_argument("--stationary-camera", action="store_true")
    solve = subparsers.add_parser("solve", help="solve and validate a captured sample set")
    solve.add_argument("samples")
    solve.add_argument(
        "--output",
        default="~/.config/cais-spade-llm/ur5e_realsense_hand_eye.yaml",
    )
    solve.add_argument("--camera-role", default="ur5e")
    solve.add_argument("--parent-frame", default="tool0")
    stationary_solve = subparsers.add_parser(
        "solve-stationary",
        help="solve a fixed camera extrinsic from a surveyed ChArUco board",
    )
    stationary_solve.add_argument("samples")
    stationary_solve.add_argument(
        "--output",
        default="~/.config/cais-spade-llm/stationary_realsense_extrinsic.yaml",
    )
    stationary_solve.add_argument("--camera-role", default="stationary")
    stationary_solve.add_argument("--parent-frame", default="world")
    stationary_solve.add_argument("--board-x", type=float, required=True)
    stationary_solve.add_argument("--board-y", type=float, required=True)
    stationary_solve.add_argument("--board-z", type=float, required=True)
    stationary_solve.add_argument("--board-roll", type=float, required=True)
    stationary_solve.add_argument("--board-pitch", type=float, required=True)
    stationary_solve.add_argument("--board-yaw", type=float, required=True)
    table_plane = subparsers.add_parser(
        "table-plane",
        help="collect 10 stationary gear detections and calibrate the physical table plane",
    )
    table_plane.add_argument(
        "--output",
        default=os.environ.get(
            "REALSENSE_HAND_EYE_CONFIG",
            "~/.config/cais-spade-llm/ur5e_realsense_hand_eye.yaml",
        ),
    )
    table_plane.add_argument("--frames", type=int, default=MINIMUM_TABLE_PLANE_FRAMES)
    table_plane.add_argument("--maximum-attempts", type=int, default=30)
    table_plane.add_argument(
        "--service",
        default="/perception/ur5e/table_plane_measurement",
    )
    return parser


def main() -> None:
    """Run board generation, guided capture, or accepted calibration solving."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args()
    if args.command == "board":
        generate_board(args.output)
    elif args.command == "capture":
        capture_samples(
            args.output,
            accepted_poses=args.poses,
            camera_role=args.camera_role,
            world_frame=args.world_frame,
            tool_frame=args.tool_frame,
            camera_link_frame=args.camera_link_frame,
            camera_optical_frame=args.camera_optical_frame,
            color_topic=args.color_topic,
            camera_info_topic=args.camera_info_topic,
        )
    elif args.command == "capture-once":
        capture_one_sample(
            args.output,
            camera_role=args.camera_role,
            world_frame=args.world_frame,
            tool_frame=args.tool_frame,
            camera_link_frame=args.camera_link_frame,
            camera_optical_frame=args.camera_optical_frame,
            color_topic=args.color_topic,
            camera_info_topic=args.camera_info_topic,
            stationary_camera=args.stationary_camera,
        )
    elif args.command == "solve":
        solve_calibration(
            args.samples,
            args.output,
            camera_role=args.camera_role,
            parent_frame=args.parent_frame,
        )
    elif args.command == "solve-stationary":
        solve_stationary_calibration(
            args.samples,
            args.output,
            camera_role=args.camera_role,
            parent_frame=args.parent_frame,
            board_world_pose={
                "configured": True,
                "x": args.board_x,
                "y": args.board_y,
                "z": args.board_z,
                "roll": args.board_roll,
                "pitch": args.board_pitch,
                "yaw": args.board_yaw,
            },
        )
    elif args.command == "table-plane":
        capture_table_plane(
            args.output,
            frame_count=args.frames,
            maximum_attempts=args.maximum_attempts,
            measurement_service=args.service,
        )


if __name__ == "__main__":
    main()

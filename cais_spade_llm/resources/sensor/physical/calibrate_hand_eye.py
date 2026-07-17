"""Guided ChArUco eye-in-hand calibration for the wrist RealSense."""

from __future__ import annotations

import argparse
import json
import logging
import math
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
    def __init__(self, *, world_frame: str, tool_frame: str, camera_link_frame: str, camera_optical_frame: str) -> None:
        import rclpy
        import tf2_ros
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image

        self.rclpy = rclpy
        self.node = Node("ur5e_realsense_hand_eye_capture")
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
            "/camera/camera/color/image_raw",
            self._on_image,
            10,
        )
        self.node.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
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
        message = self.tf_buffer.lookup_transform(
            target,
            source,
            self.rclpy.time.Time(),
            timeout=self.rclpy.duration.Duration(seconds=1.0),
        )
        return _matrix_from_transform_message(message.transform)


def capture_samples(
    output_path: str | Path,
    *,
    accepted_poses: int = 25,
    world_frame: str = "world",
    tool_frame: str = "tool0",
    camera_link_frame: str = "camera_link",
    camera_optical_frame: str = "camera_color_optical_frame",
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
        world_frame=world_frame,
        tool_frame=tool_frame,
        camera_link_frame=camera_link_frame,
        camera_optical_frame=camera_optical_frame,
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
    output.write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")
    return output


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


def solve_calibration(samples_path: str | Path, output_path: str | Path) -> dict[str, Any]:
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    logger.info("Accepted calibration written to %s", output)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    board = subparsers.add_parser("board", help="generate the printable ChArUco board")
    board.add_argument("output")
    capture = subparsers.add_parser("capture", help="capture stationary eye-in-hand samples")
    capture.add_argument("output")
    capture.add_argument("--poses", type=int, default=25)
    solve = subparsers.add_parser("solve", help="solve and validate a captured sample set")
    solve.add_argument("samples")
    solve.add_argument(
        "--output",
        default="~/.config/cais-spade-llm/ur5e_realsense_hand_eye.yaml",
    )
    return parser


def main() -> None:
    """Run board generation, guided capture, or accepted calibration solving."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args()
    if args.command == "board":
        generate_board(args.output)
    elif args.command == "capture":
        capture_samples(args.output, accepted_poses=args.poses)
    elif args.command == "solve":
        solve_calibration(args.samples, args.output)


if __name__ == "__main__":
    main()

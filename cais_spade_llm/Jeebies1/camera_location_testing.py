"""Stream Intel RealSense D435 video and inspect camera-frame locations.

Run:
    python cais_spade_llm/Jeebies1/camera_location_testing.py

Controls:
    left click  Print the clicked pixel's depth and XYZ location in meters
    c           Print the center pixel's depth and XYZ location
    s           Save the current color frame as a PNG
    q or Esc    Quit
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ClickState:
    pixel: tuple[int, int] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream color/depth video from an Intel RealSense D435 camera."
    )
    parser.add_argument("--width", type=int, default=640, help="Stream width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Stream height in pixels.")
    parser.add_argument("--fps", type=int, default=30, help="Stream frames per second.")
    parser.add_argument(
        "--serial",
        default=None,
        help="Optional RealSense serial number when more than one camera is connected.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print connected RealSense devices and exit.",
    )
    parser.add_argument(
        "--save-dir",
        default=".",
        help="Directory for screenshots saved with the 's' key.",
    )
    return parser.parse_args()


def import_runtime_dependencies():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import pyrealsense2 as rs  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing camera dependency. Install the runtime packages first:\n"
            "  pip install opencv-python numpy pyrealsense2\n"
            f"Original import error: {exc}"
        ) from exc

    return cv2, np, rs


def list_devices(rs) -> None:
    context = rs.context()
    devices = list(context.query_devices())

    if not devices:
        print("No Intel RealSense devices were found.")
        return

    print("Connected Intel RealSense devices:")
    for index, device in enumerate(devices, start=1):
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        firmware = device.get_info(rs.camera_info.firmware_version)
        print(f"  {index}. {name} | serial={serial} | firmware={firmware}")


def deproject_pixel(
    depth_frame, rs, pixel: tuple[int, int]
) -> tuple[float, float, float, float] | None:
    x, y = pixel
    width = depth_frame.get_width()
    height = depth_frame.get_height()

    if not (0 <= x < width and 0 <= y < height):
        return None

    depth_m = depth_frame.get_distance(x, y)
    if depth_m <= 0:
        return None

    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    point_m = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_m)
    return depth_m, point_m[0], point_m[1], point_m[2]


def print_location(depth_frame, rs, pixel: tuple[int, int], label: str) -> None:
    location = deproject_pixel(depth_frame, rs, pixel)
    x, y = pixel

    if location is None:
        print(f"{label} pixel=({x}, {y}) has no valid depth reading.")
        return

    depth_m, x_m, y_m, z_m = location
    print(
        f"{label} pixel=({x}, {y}) depth={depth_m:.3f} m "
        f"camera_xyz=({x_m:.3f}, {y_m:.3f}, {z_m:.3f}) m"
    )


def stream_realsense(args: argparse.Namespace) -> None:
    cv2, np, rs = import_runtime_dependencies()

    if args.list_devices:
        list_devices(rs)
        return

    pipeline = rs.pipeline()
    config = rs.config()

    if args.serial:
        config.enable_device(args.serial)

    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    align_to_color = rs.align(rs.stream.color)
    click_state = ClickState()
    window_name = "Intel RealSense D435 - camera_location_testing"

    def on_mouse(event, x, y, flags, userdata) -> None:
        del flags, userdata
        if event == cv2.EVENT_LBUTTONDOWN:
            click_state.pixel = (x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    started = False
    try:
        print("Starting Intel RealSense D435 stream...")
        profile = pipeline.start(config)
        started = True
        device = profile.get_device()
        device_name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"Streaming from {device_name} serial={serial}")
        print("Left-click for XYZ, press 'c' for center XYZ, 's' to save, 'q' or Esc to quit.")

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align_to_color.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            display = np.hstack((color_image, depth_vis))
            cv2.putText(
                display,
                "color | depth    left-click: XYZ   c: center   s: save   q/Esc: quit",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if click_state.pixel is not None:
                print_location(depth_frame, rs, click_state.pixel, "click")
                cv2.circle(display, click_state.pixel, 5, (0, 255, 255), 2)
                click_state.pixel = None

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("c"):
                center_pixel = (args.width // 2, args.height // 2)
                print_location(depth_frame, rs, center_pixel, "center")

            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = save_dir / f"realsense_color_{timestamp}.png"
                cv2.imwrite(str(path), color_image)
                print(f"Saved {path}")

    except RuntimeError as exc:
        raise SystemExit(
            "Unable to start the RealSense stream. Check that the D435 is plugged in, "
            "not already open in another app, and that Intel RealSense drivers are installed.\n"
            f"RealSense error: {exc}"
        ) from exc
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()


def main() -> None:
    stream_realsense(parse_args())


if __name__ == "__main__":
    main()

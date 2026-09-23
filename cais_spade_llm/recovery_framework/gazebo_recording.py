"""Record a validation-owned Gazebo window and retain only verified attempts."""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)
VIDEO_NAMES = ("capture.partial.mp4", "preview.partial.mp4", "assembly.mp4", "assembly-10x.mp4")


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def _process_identity(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _owner_alive(owner: dict) -> bool:
    return bool(owner.get("start_ticks")) and _process_identity(owner["pid"]) == owner["start_ticks"]


def _remove_videos(directory: Path) -> None:
    for name in VIDEO_NAMES:
        (directory / name).unlink(missing_ok=True)


def cleanup_abandoned(root: Path) -> None:
    """Delete only videos belonging to abandoned, unvalidated recording attempts."""
    for path in root.glob("attempt-*/owner.json"):
        try:
            owner = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (owner.get("kind") == "cais_gazebo_recording"
                and owner.get("attempt") == path.parent.name
                and not (path.parent / "success.json").exists()
                and not _owner_alive(owner)):
            _remove_videos(path.parent)


class RecordingAttempt:
    """Own capture, cleanup, and success-only promotion for one validation run."""

    def __init__(self, root: Path, *, python: str = "/usr/bin/python3") -> None:
        root.mkdir(parents=True, exist_ok=True)
        cleanup_abandoned(root)
        self.directory = root / f"attempt-{uuid4().hex}"
        self.directory.mkdir()
        self.python = python
        self.process: subprocess.Popen | None = None
        self.log = None
        self.owner = {"kind": "cais_gazebo_recording", "attempt": self.directory.name,
                      "pid": os.getpid(), "start_ticks": _process_identity(os.getpid())}
        _write_json(self.directory / "owner.json", self.owner)

    def _spawn(self, operation: str) -> subprocess.Popen:
        if self.log is None:
            self.log = (self.directory / "recorder.log").open("a")
        return subprocess.Popen(
            [self.python, str(Path(__file__).resolve()), operation, str(self.directory)],
            stdout=self.log, stderr=self.log, start_new_session=True,
        )

    def start(self, *, timeout: float = 20.0) -> None:
        """Wait for the first encoded frame before permitting production to start."""
        if self.process is not None:
            raise RuntimeError("This recording attempt has already started")
        self.process = self._spawn("capture")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                break
            if (self.directory / "ready.json").exists():
                return
            time.sleep(.05)
        self.cancel()
        raise RuntimeError(f"Gazebo recording did not become ready; see {self.directory / 'recorder.log'}")

    def check(self) -> None:
        """Reject a lost recorder during production without changing robot state."""
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Gazebo recorder exited before recording completion")

    def finish(self) -> dict:
        """Close the continuous recording after final observation and home returns."""
        self.check()
        (self.directory / "finish").touch()
        try:
            self.process.wait(timeout=20)
            metadata = json.loads((self.directory / "capture.json").read_text())
            if self.process.returncode or metadata.get("status") != "captured":
                raise RuntimeError("Gazebo recording failed")
            return metadata
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
            self.cancel()
            raise

    def promote(self, acceptance: dict) -> dict:
        """Publish both recordings only after physical and complete video validation.

        Args:
            acceptance: Correlated validation evidence with validated=True and run_id.
        """
        if acceptance.get("validated") is not True or not acceptance.get("run_id"):
            self.cancel()
            raise ValueError("Successful physical validation and a run identity are required")
        if self.process is None or self.process.poll() is None or self.process.returncode:
            self.cancel()
            raise RuntimeError("Capture must close successfully before video validation")
        try:
            self.process = self._spawn("preview")
            self.process.wait(timeout=1200)
            if self.process.returncode:
                raise RuntimeError(f"Video validation failed; see {self.directory / 'recorder.log'}")
            video = json.loads((self.directory / "video_validation.json").read_text())
            if video.get("validated") is not True:
                raise RuntimeError("Video validation did not pass")
            (self.directory / "capture.partial.mp4").replace(self.directory / "assembly.mp4")
            (self.directory / "preview.partial.mp4").replace(self.directory / "assembly-10x.mp4")
            result = {**acceptance, "video": video, "full_video": "assembly.mp4",
                      "preview_video": "assembly-10x.mp4", "preview_speed": 10}
            _write_json(self.directory / "success.json", result)
            return result
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
            self.cancel()
            raise
        finally:
            if self.log is not None:
                self.log.close()
                self.log = None

    def cancel(self) -> None:
        """Stop the owned process and discard unvalidated videos, keeping diagnostics."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if not (self.directory / "success.json").exists():
            _remove_videos(self.directory)
        if self.log is not None:
            self.log.close()
            self.log = None


class X11Capture:
    """Read the visible Gazebo window without recording the operator's desktop."""

    class XImage(ctypes.Structure):
        _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int),
                    ("xoffset", ctypes.c_int), ("format", ctypes.c_int), ("data", ctypes.c_void_p),
                    ("byte_order", ctypes.c_int), ("bitmap_unit", ctypes.c_int),
                    ("bitmap_bit_order", ctypes.c_int), ("bitmap_pad", ctypes.c_int),
                    ("depth", ctypes.c_int), ("bytes_per_line", ctypes.c_int),
                    ("bits_per_pixel", ctypes.c_int)]

    def __init__(self) -> None:
        tree = subprocess.check_output(["xwininfo", "-root", "-tree"], text=True, timeout=5)
        windows = re.findall(r'(0x[0-9a-f]+) "Gazebo"', tree)
        if len(windows) != 1:
            raise RuntimeError(f"Expected one visible Gazebo window; found {len(windows)}")
        self.window = int(windows[0], 16)
        self.x = ctypes.CDLL("libX11.so.6")
        self.x.XOpenDisplay.restype = ctypes.c_void_p
        self.x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.display = self.x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError("Cannot open Gazebo X display")
        self.x.XGetGeometry.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            *[ctypes.POINTER(ctypes.c_uint)] * 4]
        self.x.XGetImage.restype = ctypes.POINTER(self.XImage)
        self.x.XGetImage.argtypes = [ctypes.c_void_p, ctypes.c_ulong, *[ctypes.c_int] * 2,
                                     *[ctypes.c_uint] * 2, ctypes.c_ulong, ctypes.c_int]
        self.x.XDestroyImage.argtypes = [ctypes.POINTER(self.XImage)]
        self.x.XCloseDisplay.argtypes = [ctypes.c_void_p]

    def frame(self):
        """Copy one current BGR frame before releasing the X server image."""
        import numpy as np

        root, x, y = ctypes.c_ulong(), ctypes.c_int(), ctypes.c_int()
        width, height, border, depth = [ctypes.c_uint() for _ in range(4)]
        if not self.x.XGetGeometry(self.display, self.window, ctypes.byref(root), ctypes.byref(x),
                                  ctypes.byref(y), *[ctypes.byref(v) for v in (width, height, border, depth)]):
            raise RuntimeError("Gazebo window is unavailable")
        image = self.x.XGetImage(self.display, self.window, 0, 0, width.value, height.value,
                                ctypes.c_ulong(-1), 2)
        if not image:
            raise RuntimeError("Gazebo window capture failed")
        try:
            data = image.contents
            if data.bits_per_pixel != 32 or data.byte_order != 0:
                raise RuntimeError("Unsupported X11 image format")
            raw = ctypes.string_at(data.data, data.bytes_per_line * data.height)
            pixels = np.frombuffer(raw, dtype=np.uint8).reshape(data.height, data.bytes_per_line)
            return pixels[:, :data.width * 4].reshape(data.height, data.width, 4)[:, :, :3].copy()
        finally:
            self.x.XDestroyImage(image)

    def close(self) -> None:
        """Release the owned display connection."""
        self.x.XCloseDisplay(self.display)


def _capture(directory: Path) -> None:
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    owner = json.loads((directory / "owner.json").read_text())
    camera = X11Capture()
    writer = None
    interrupted = False

    def stop(_signal, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    frames = captures = repeated = 0
    max_gap = 0.0
    started = last = time.monotonic()
    started_wall = time.time()
    fps = 15.0
    status = "failed"
    try:
        frame = camera.frame()
        h, w = frame.shape[:2]
        luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = float(np.percentile(luminance, 95) - np.percentile(luminance, 5))
        if w < 640 or h < 360 or contrast < 20:
            raise RuntimeError("Gazebo capture is too small or blank")
        if not cv2.imwrite(str(directory / "first_frame.png"), frame):
            raise RuntimeError("Cannot preserve the initial rendered-frame check")
        scale = min(1., 1920 / w)
        size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
        writer = cv2.VideoWriter(str(directory / "capture.partial.mp4"), cv2.CAP_FFMPEG,
                                cv2.VideoWriter_fourcc(*"avc1"), fps, size)
        if not writer.isOpened():
            raise RuntimeError("H.264 MP4 encoder is unavailable")
        previous = None
        with (directory / "frames.jsonl").open("w") as observations:
            while not interrupted and _owner_alive(owner):
                now = time.monotonic()
                if (directory / "finish").exists():
                    status = "captured"
                    break
                if frames and now < started + frames / fps:
                    time.sleep(min(.02, started + frames / fps - now))
                    continue
                raw = camera.frame()
                h, w = raw.shape[:2]
                fit = min(size[0] / w, size[1] / h)
                small = cv2.resize(raw, (round(w * fit), round(h * fit)), interpolation=cv2.INTER_AREA)
                frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                y, x = (size[1] - small.shape[0]) // 2, (size[0] - small.shape[1]) // 2
                frame[y:y + small.shape[0], x:x + small.shape[1]] = small
                now = time.monotonic()
                gap = now - last
                max_gap = max(max_gap, gap)
                target_frame = int((now - started) * fps)
                missed = max(0, target_frame - frames)
                if gap > 2.0:
                    raise RuntimeError("Recording lost more than two seconds of observation")
                for _ in range(missed):
                    writer.write(previous if previous is not None else frame)
                writer.write(frame)
                frames += missed + 1
                repeated += missed
                captures += 1
                previous, last = frame, now
                observations.write(json.dumps({"frame": frames - 1, "observed_at_unix": time.time(),
                                               "elapsed_sec": now - started, "repeated_frames": missed}) + "\n")
                if captures % 15 == 0:
                    observations.flush()
                if captures == 1:
                    _write_json(directory / "ready.json", {"started_at_unix": started_wall, "size": size,
                                 "initial_frame_contrast": contrast, "initial_frame": "first_frame.png"})
    finally:
        if writer is not None:
            writer.release()
        camera.close()
        _write_json(directory / "capture.json", {
            "status": status, "started_at_unix": started_wall, "ended_at_unix": time.time(),
            "fps": fps, "frames": frames, "captures": captures, "repeated_frames": repeated,
            "max_capture_gap_sec": max_gap, "wall_duration_sec": time.monotonic() - started,
        })
        if status != "captured":
            _remove_videos(directory)


def _preview(directory: Path) -> None:
    import cv2

    cv2.setNumThreads(1)
    metadata = json.loads((directory / "capture.json").read_text())
    if metadata["status"] != "captured":
        raise ValueError("Recording is incomplete")
    owner = json.loads((directory / "owner.json").read_text())
    reader = cv2.VideoCapture(str(directory / "capture.partial.mp4"))
    fps = reader.get(cv2.CAP_PROP_FPS)
    size = (int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if not reader.isOpened() or fps != 15 or not all(size):
        raise ValueError("Full recording is not readable")
    writer = cv2.VideoWriter(str(directory / "preview.partial.mp4"), cv2.CAP_FFMPEG,
                            cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    count = preview_count = 0
    try:
        if not writer.isOpened():
            raise RuntimeError("Preview encoder is unavailable")
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            if count % 10 == 0:
                cv2.rectangle(frame, (8, 8), (400, 48), (0, 0, 0), -1)
                cv2.putText(frame, "10x speed | Gazebo assembly", (16, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(frame)
                preview_count += 1
            count += 1
            if count % 150 == 0 and not _owner_alive(owner):
                raise RuntimeError("Recording owner exited")
    finally:
        reader.release()
        writer.release()
    if count != metadata["frames"] or count < 2:
        raise ValueError("Full recording has missing or unreadable frames")
    preview = cv2.VideoCapture(str(directory / "preview.partial.mp4"))
    decoded_preview = 0
    while preview.read()[0]:
        decoded_preview += 1
    preview.release()
    if decoded_preview != preview_count:
        raise ValueError("Preview has missing or unreadable frames")
    duration = count / fps
    if abs(duration - metadata["wall_duration_sec"]) > 1:
        raise ValueError("Recording duration disagrees with wall-clock observations")
    _write_json(directory / "video_validation.json", {
        "validated": True, "full_frames_decoded": count, "preview_frames_decoded": decoded_preview,
        "fps": fps, "size": size, "duration_sec": duration, "preview_duration_sec": preview_count / fps,
        "capture": metadata,
    })


def main() -> None:
    """Run only the capture or preview worker owned by a validation attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("capture", "preview"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        (_capture if args.operation == "capture" else _preview)(args.directory)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        _remove_videos(args.directory)
        logger.exception("Gazebo recording failed")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

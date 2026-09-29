"""Record a validation-owned Gazebo window and retain only verified attempts."""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import math
import os
import re
import resource
import signal
import subprocess
import time
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)
VIDEO_NAMES = ("capture.partial.mp4", "preview.partial.mp4", "assembly.mp4", "assembly-10x.mp4",
               "clock.partial.mp4", "assembly-rtf2.mp4")


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

    def __init__(self, root: Path, *, python: str = "/usr/bin/python3",
                 record_simulation_clock: bool = False) -> None:
        root.mkdir(parents=True, exist_ok=True)
        cleanup_abandoned(root)
        self.directory = root / f"attempt-{uuid4().hex}"
        self.directory.mkdir()
        self.python = python
        self.process: subprocess.Popen | None = None
        self.clock_process: subprocess.Popen | None = None
        self.record_simulation_clock = record_simulation_clock
        self.log = None
        self.owner = {"kind": "cais_gazebo_recording", "attempt": self.directory.name,
                      "pid": os.getpid(), "start_ticks": _process_identity(os.getpid())}
        _write_json(self.directory / "owner.json", self.owner)
        if record_simulation_clock:
            _write_json(self.directory / "clock_config.json", {"real_time_factor": 2.0})

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
        deadline = time.monotonic() + timeout
        if self.record_simulation_clock:
            self.clock_process = self._spawn("clock")
            while time.monotonic() < deadline:
                if self.clock_process.poll() is not None:
                    break
                if (self.directory / "clock_ready.json").exists():
                    break
                time.sleep(.05)
            if (self.clock_process.poll() is not None
                    or not (self.directory / "clock_ready.json").exists()):
                self.cancel()
                raise RuntimeError("Gazebo simulation clock did not become ready")
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
        if self.clock_process is not None and self.clock_process.poll() is not None:
            raise RuntimeError("Gazebo clock recorder exited before recording completion")

    def finish(self) -> dict:
        """Close the continuous recording after final observation and home returns."""
        self.check()
        (self.directory / "finish").touch()
        try:
            self.process.wait(timeout=20)
            metadata = json.loads((self.directory / "capture.json").read_text())
            if self.process.returncode or metadata.get("status") != "captured":
                raise RuntimeError("Gazebo recording failed")
            if self.clock_process is not None:
                observations = (self.directory / "frames.jsonl").read_text().splitlines()
                last_frame = json.loads(observations[-1])["observed_at_unix"]
                _write_json(self.directory / "finish_clock.json", {"last_frame_at_unix": last_frame})
                self.clock_process.wait(timeout=10)
                clock = json.loads((self.directory / "clock_capture.json").read_text())
                if self.clock_process.returncode or clock.get("status") != "captured":
                    raise RuntimeError("Gazebo clock recording failed")
            return metadata
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
            self.cancel()
            raise

    def promote(self, acceptance: dict) -> dict:
        """Publish requested recordings after physical and complete video validation.

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
            if self.record_simulation_clock:
                if not video.get("simulation_clock", {}).get("validated"):
                    raise RuntimeError("Simulation-clock video validation did not pass")
                (self.directory / "clock.partial.mp4").replace(self.directory / "assembly-rtf2.mp4")
            (self.directory / "capture.partial.mp4").replace(self.directory / "assembly.mp4")
            (self.directory / "preview.partial.mp4").replace(self.directory / "assembly-10x.mp4")
            result = {**acceptance, "video": video, "full_video": "assembly.mp4",
                      "preview_video": "assembly-10x.mp4", "preview_speed": 10}
            if self.record_simulation_clock:
                result.update(simulation_clock_video="assembly-rtf2.mp4", simulation_clock_factor=2.0)
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
        if self.clock_process is not None and self.clock_process.poll() is None:
            self.clock_process.terminate()
            try:
                self.clock_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.clock_process.kill()
                self.clock_process.wait(timeout=5)
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


class _H264Writer:
    """Stream BGR frames to the NVIDIA encoder and check its completion."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        self.log = path.with_suffix('.encoder.log').open('w')
        self.closed = False
        self.progress = path.with_suffix('.encoder.progress')
        self.progress.unlink(missing_ok=True)
        try:
            self.process = subprocess.Popen([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                '-filter_threads', '1', '-stats_period', '0.05', '-progress', str(self.progress),
                '-f', 'rawvideo', '-pixel_format', 'bgr24',
                '-video_size', f'{size[0]}x{size[1]}', '-framerate', str(fps),
                '-i', 'pipe:0', '-an', '-c:v', 'h264_nvenc', '-preset', 'p1',
                '-rc', 'vbr', '-cq', '23', '-b:v', '0', '-pix_fmt', 'yuv420p',
                '-bf', '0', '-delay', '0', '-zerolatency', '1',
                '-movflags', '+faststart', str(path),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        except OSError:
            self.log.close()
            raise

    def write(self, frame) -> None:
        if self.closed or self.process.poll() is not None:
            raise RuntimeError('NVIDIA H.264 encoder exited during recording')
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError('NVIDIA H.264 encoder rejected a frame') from exc

    def wait_ready(self) -> None:
        """Wait for the first encoded frame before starting the capture clock."""
        self.process.stdin.flush()
        deadline = time.monotonic() + 15.
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError('NVIDIA H.264 encoder exited during startup')
            if self.progress.exists() and re.search(r'^frame=[1-9][0-9]*$', self.progress.read_text(), re.MULTILINE):
                return
            time.sleep(.02)
        raise RuntimeError('NVIDIA H.264 encoder did not produce its first frame')

    def release(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
                raise
            if code:
                raise RuntimeError(f'NVIDIA H.264 encoder failed with status {code}')
        finally:
            self.log.close()


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
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    status = "failed"
    try:
        frame = camera.frame()
        first_observed_wall = time.time()
        h, w = frame.shape[:2]
        luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = float(np.percentile(luminance, 95) - np.percentile(luminance, 5))
        if w < 640 or h < 360 or contrast < 20:
            raise RuntimeError("Gazebo capture is too small or blank")
        if not cv2.imwrite(str(directory / "first_frame.png"), frame):
            raise RuntimeError("Cannot preserve the initial rendered-frame check")
        scale = min(1., 1920 / w)
        size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
        writer = _H264Writer(directory / "capture.partial.mp4", fps, size)
        previous = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        writer.write(previous)
        writer.wait_ready()
        started = last = time.monotonic()
        started_wall = time.time()
        frames = captures = 1
        with (directory / "frames.jsonl").open("w") as observations:
            observations.write(json.dumps({"frame": 0, "observed_at_unix": first_observed_wall,
                                           "elapsed_sec": 0., "repeated_frames": 0}) + "\n")
            _write_json(directory / "ready.json", {"started_at_unix": started_wall, "size": size,
                         "encoder": "h264_nvenc", "initial_frame_contrast": contrast,
                         "initial_frame": "first_frame.png"})
            while not interrupted and _owner_alive(owner):
                now = time.monotonic()
                if (directory / "finish").exists():
                    status = "captured"
                    break
                if frames and now < started + frames / fps:
                    time.sleep(min(.02, started + frames / fps - now))
                    continue
                raw = camera.frame()
                observed_wall = time.time()
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
                observations.write(json.dumps({"frame": frames - 1, "observed_at_unix": observed_wall,
                                               "elapsed_sec": now - started, "repeated_frames": missed}) + "\n")
                if captures % 15 == 0:
                    observations.flush()
    finally:
        finished = time.monotonic()
        finished_wall = time.time()
        try:
            if writer is not None:
                writer.release()
        except (OSError, RuntimeError, subprocess.SubprocessError):
            status = "failed"
            raise
        finally:
            camera.close()
            _write_json(directory / "capture.json", {
                "status": status, "started_at_unix": started_wall, "ended_at_unix": finished_wall,
                "encoder_finalize_time_sec": time.monotonic() - finished,
                "fps": fps, "frames": frames, "captures": captures, "repeated_frames": repeated,
                "encoder": "h264_nvenc", "preset": "p1",
                "capture_cpu_time_sec": (resource.getrusage(resource.RUSAGE_SELF).ru_utime
                                         + resource.getrusage(resource.RUSAGE_SELF).ru_stime
                                         - cpu_started.ru_utime - cpu_started.ru_stime),
                "encoder_cpu_time_sec": (resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
                                         + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime),
                "max_capture_gap_sec": max_gap, "wall_duration_sec": finished - started,
            })
            if status != "captured":
                _remove_videos(directory)


def _record_clock(directory: Path) -> None:
    """Preserve fresh /clock observations in a separate, attempt-owned process."""
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock

    owner = json.loads((directory / "owner.json").read_text())
    interrupted = False

    def stop(_signal, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    rclpy.init()
    node = rclpy.create_node("cais_recording_clock")
    samples = 0
    last_wall = last_simulation = None
    status = "failed"
    try:
        with (directory / "clock_samples.jsonl").open("w") as stream:
            def observed(message):
                nonlocal samples, last_wall, last_simulation
                wall = time.time()
                simulation = message.clock.sec + message.clock.nanosec / 1e9
                if last_wall is not None and (wall <= last_wall or simulation < last_simulation):
                    raise ValueError("Gazebo recording clock reset or moved backwards")
                stream.write(json.dumps({"observed_at_unix": wall,
                                         "simulation_time_sec": simulation}) + "\n")
                stream.flush()
                last_wall, last_simulation = wall, simulation
                samples += 1
                if samples == 2:
                    _write_json(directory / "clock_ready.json", {"observed_at_unix": wall})

            node.create_subscription(Clock, "/clock", observed, qos_profile_sensor_data)
            started = time.monotonic()
            while not interrupted and _owner_alive(owner) and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=.1)
                if last_wall is not None and time.time() - last_wall > 2.0:
                    raise RuntimeError("Gazebo recording clock observations became stale")
                if last_wall is None and time.monotonic() - started > 15.0:
                    raise RuntimeError("Gazebo recording received no simulation clock")
                finish = directory / "finish_clock.json"
                if finish.exists() and last_wall is not None:
                    if last_wall >= json.loads(finish.read_text())["last_frame_at_unix"]:
                        status = "captured"
                        break
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        _write_json(directory / "clock_capture.json", {
            "status": status, "samples": samples, "last_observed_at_unix": last_wall,
            "last_simulation_time_sec": last_simulation,
        })


def _simulation_clock_frames(directory: Path, *, fps: float, frame_count: int) -> dict:
    """Select captured frames on a uniform two-simulation-seconds-per-video-second grid."""
    import numpy as np

    clocks = [json.loads(line) for line in (directory / "clock_samples.jsonl").read_text().splitlines()]
    frames = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
    if len(clocks) < 2 or len(frames) < 2 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Insufficient simulation clock or frame observations")
    wall = np.asarray([row["observed_at_unix"] for row in clocks], dtype=float)
    simulation = np.asarray([row["simulation_time_sec"] for row in clocks], dtype=float)
    observed = np.asarray([row["observed_at_unix"] for row in frames], dtype=float)
    indices = np.asarray([row["frame"] for row in frames], dtype=int)
    if (not all(np.isfinite(values).all() for values in (wall, simulation, observed))
            or np.any(np.diff(wall) <= 0) or np.any(np.diff(simulation) < 0)
            or np.any(np.diff(observed) <= 0) or np.any(np.diff(indices) <= 0)
            or indices[0] != 0 or indices[-1] != frame_count - 1):
        raise ValueError("Recording clocks or frame indices are invalid or moved backwards")
    if wall[0] > observed[0] or wall[-1] < observed[-1]:
        raise ValueError("Simulation clock does not cover the complete recording")
    if float(np.max(np.diff(wall))) > 2.0:
        raise ValueError("Simulation clock contains stale observation gaps")
    frame_simulation = np.interp(observed, wall, simulation)
    span = float(frame_simulation[-1] - frame_simulation[0])
    if span <= 0:
        raise ValueError("Simulation clock did not advance")
    factor = 2.0
    count = math.floor(span * fps / factor) + 1
    targets = frame_simulation[0] + np.arange(count) * factor / fps
    right = np.searchsorted(frame_simulation, targets, side="left").clip(0, len(frames) - 1)
    left = (right - 1).clip(0, len(frames) - 1)
    nearest = np.where(abs(frame_simulation[left] - targets) <= abs(frame_simulation[right] - targets),
                       left, right)
    selected = indices[nearest]
    errors = abs(frame_simulation[nearest] - targets)
    # Clock and GUI are separate observers. Preserve the measured sampling error.
    if float(errors.max()) > .25:
        raise ValueError("Captured frames are too sparse for simulation-clock playback")
    return {
        "real_time_factor": factor, "fps": fps, "frames": count,
        "source_frame_indices": selected.tolist(),
        "simulation_start_sec": float(frame_simulation[0]),
        "simulation_end_sec": float(frame_simulation[-1]),
        "simulation_duration_sec": span, "video_duration_sec": count / fps,
        "average_real_time_factor": span / (count / fps),
        "max_frame_clock_error_sec": float(errors.max()),
        "max_clock_observation_gap_sec": float(np.max(np.diff(wall))),
        "clock_samples": len(clocks), "captured_frames": len(frames),
        "mapping": "linear interpolation of /clock at frame capture times; nearest captured frame",
    }


def _decoded_frames(path: Path) -> int:
    import cv2

    reader = cv2.VideoCapture(str(path))
    count = 0
    try:
        while reader.read()[0]:
            count += 1
    finally:
        reader.release()
    return count


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
        reader.release()
        raise ValueError("Full recording is not readable")
    writer = clock_writer = None
    clock = None
    count = preview_count = clock_count = 0
    try:
        if (directory / "clock_config.json").exists():
            clock = _simulation_clock_frames(directory, fps=fps, frame_count=metadata["frames"])
            _write_json(directory / "simulation_clock_mapping.json", clock)
            clock_writer = _H264Writer(directory / "clock.partial.mp4", fps, size)
        writer = _H264Writer(directory / "preview.partial.mp4", fps, size)
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            if count % 10 == 0:
                preview = frame.copy()
                cv2.rectangle(preview, (8, 8), (400, 48), (0, 0, 0), -1)
                cv2.putText(preview, "10x speed | Gazebo assembly", (16, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(preview)
                preview_count += 1
            if clock is not None:
                while clock_count < clock["frames"] and clock["source_frame_indices"][clock_count] == count:
                    paced = frame.copy()
                    cv2.rectangle(paced, (8, 8), (525, 48), (0, 0, 0), -1)
                    cv2.putText(paced, "RTF 2 | 2 simulated seconds per video second", (16, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
                    clock_writer.write(paced)
                    clock_count += 1
            count += 1
            if count % 150 == 0 and not _owner_alive(owner):
                raise RuntimeError("Recording owner exited")
    finally:
        reader.release()
        try:
            if writer is not None:
                writer.release()
        finally:
            if clock_writer is not None:
                clock_writer.release()
    if count != metadata["frames"] or count < 2:
        raise ValueError("Full recording has missing or unreadable frames")
    decoded_preview = _decoded_frames(directory / "preview.partial.mp4")
    if decoded_preview != preview_count:
        raise ValueError("Preview has missing or unreadable frames")
    duration = count / fps
    observed_duration = metadata["wall_duration_sec"]
    duration_source = "capture.json:wall_duration_sec"
    frame_observations = directory / "frames.jsonl"
    if frame_observations.exists():
        # Earlier capture metadata included encoder shutdown after the last frame.
        last_frame = json.loads(frame_observations.read_text().splitlines()[-1])
        elapsed = float(last_frame["elapsed_sec"])
        if last_frame["frame"] != count - 1 or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Recording has invalid final frame timing")
        observed_duration = elapsed + 1 / fps
        duration_source = "frames.jsonl:last elapsed_sec + frame interval"
    if abs(duration - observed_duration) > 1:
        raise ValueError("Recording duration disagrees with wall-clock observations")
    result = {
        "validated": True, "full_frames_decoded": count, "preview_frames_decoded": decoded_preview,
        "fps": fps, "size": size, "duration_sec": duration, "preview_duration_sec": preview_count / fps,
        "capture": metadata, "preview_encoder": "h264_nvenc", "preview_preset": "p1",
        "duration_validation": {"source": duration_source,
                                "observed_duration_sec": observed_duration,
                                "difference_sec": duration - observed_duration},
    }
    if clock is not None:
        decoded_clock = _decoded_frames(directory / "clock.partial.mp4")
        if decoded_clock != clock_count or clock_count != clock["frames"]:
            raise ValueError("Simulation-clock video has missing or unreadable frames")
        if abs(clock_count / fps - clock["simulation_duration_sec"] / 2) > 1 / fps + 1e-6:
            raise ValueError("Simulation-clock video duration does not match RTF 2")
        result["simulation_clock"] = {key: value for key, value in clock.items()
                                      if key != "source_frame_indices"}
        result["simulation_clock"].update(validated=True, frames_decoded=decoded_clock,
                                           encoder="h264_nvenc", preset="p1")
    _write_json(directory / "video_validation.json", result)


def main() -> None:
    """Run a capture, clock, or preview worker owned by a validation attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("capture", "preview", "clock"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        {"capture": _capture, "preview": _preview, "clock": _record_clock}[args.operation](args.directory)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        _remove_videos(args.directory)
        logger.exception("Gazebo recording failed")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

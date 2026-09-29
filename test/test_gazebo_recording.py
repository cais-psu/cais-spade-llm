"""Recording ownership, success gates, and complete H.264 video validation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from cais_spade_llm.recovery_framework import gazebo_recording as recording


def test_recording_start_failure_discards_only_owned_video(tmp_path, monkeypatch):
    attempt = recording.RecordingAttempt(tmp_path)
    video = attempt.directory / 'capture.partial.mp4'
    video.write_bytes(b'incomplete')
    unrelated = tmp_path / 'previous-success.mp4'
    unrelated.write_bytes(b'preserve')
    monkeypatch.setattr(attempt, '_spawn', lambda operation: Mock(poll=lambda: 1))
    with pytest.raises(RuntimeError, match='did not become ready'):
        attempt.start()
    assert not video.exists() and unrelated.read_bytes() == b'preserve'


def test_cancel_terminates_owned_recorder_and_preserves_logs(tmp_path):
    attempt = recording.RecordingAttempt(tmp_path)
    attempt.process = Mock(poll=lambda: None)
    for name in recording.VIDEO_NAMES:
        (attempt.directory / name).write_bytes(b'partial')
    evidence = attempt.directory / 'frames.jsonl'
    evidence.write_text('observed\n')
    attempt.cancel()
    attempt.process.terminate.assert_called_once()
    attempt.process.wait.assert_called_once_with(timeout=10)
    assert not any((attempt.directory / name).exists() for name in recording.VIDEO_NAMES)
    assert evidence.read_text() == 'observed\n'


def test_abandoned_cleanup_preserves_active_successful_and_unowned_files(tmp_path):
    active = recording.RecordingAttempt(tmp_path)
    failed = recording.RecordingAttempt(tmp_path)
    successful = recording.RecordingAttempt(tmp_path)
    for attempt in (active, failed, successful):
        (attempt.directory / 'capture.partial.mp4').write_bytes(b'video')
    owner = {**failed.owner, 'pid': 99999999}
    recording._write_json(failed.directory / 'owner.json', owner)
    recording._write_json(successful.directory / 'owner.json', {**successful.owner, 'pid': 99999999})
    (successful.directory / 'success.json').write_text('{}')
    unknown = tmp_path / 'attempt-unowned'
    unknown.mkdir()
    (unknown / 'assembly.mp4').write_bytes(b'unrelated')
    recording.cleanup_abandoned(tmp_path)
    assert not (failed.directory / 'capture.partial.mp4').exists()
    assert (active.directory / 'capture.partial.mp4').exists()
    assert (successful.directory / 'capture.partial.mp4').exists()
    assert (unknown / 'assembly.mp4').exists()


def test_failed_physical_validation_cannot_retain_video(tmp_path):
    attempt = recording.RecordingAttempt(tmp_path)
    (attempt.directory / 'capture.partial.mp4').write_bytes(b'video')
    with pytest.raises(ValueError, match='physical validation'):
        attempt.promote({'validated': False, 'run_id': 'failed'})
    assert not (attempt.directory / 'capture.partial.mp4').exists()


def test_video_failure_discards_recording_even_if_physical_validation_passed(tmp_path, monkeypatch):
    attempt = recording.RecordingAttempt(tmp_path)
    attempt.process = Mock(poll=lambda: 0, returncode=0)
    (attempt.directory / 'capture.partial.mp4').write_bytes(b'broken')
    monkeypatch.setattr(attempt, '_spawn', lambda operation: Mock(poll=lambda: 1, returncode=1))
    with pytest.raises(RuntimeError, match='Video validation failed'):
        attempt.promote({'validated': True, 'run_id': 'physically-complete'})
    assert not (attempt.directory / 'capture.partial.mp4').exists()
    assert not (attempt.directory / 'success.json').exists()


def test_success_publishes_both_videos_and_survives_cleanup(tmp_path, monkeypatch):
    attempt = recording.RecordingAttempt(tmp_path)
    attempt.process = Mock(poll=lambda: 0, returncode=0)
    for name in recording.VIDEO_NAMES[:2]:
        (attempt.directory / name).write_bytes(b'validated')
    recording._write_json(attempt.directory / 'video_validation.json', {'validated': True})
    monkeypatch.setattr(attempt, '_spawn', lambda operation: Mock(poll=lambda: 0, returncode=0))
    result = attempt.promote({'validated': True, 'run_id': 'eleven-complete'})
    attempt.cancel()
    assert result['preview_speed'] == 10
    assert (attempt.directory / 'assembly.mp4').exists()
    assert (attempt.directory / 'assembly-10x.mp4').exists()
    assert not (attempt.directory / 'capture.partial.mp4').exists()


@pytest.mark.parametrize('duration_case', ['legacy', 'encoder_finalize_delay', 'invalid_frame_duration'])
def test_system_encoder_decodes_entire_video_and_labeled_preview(tmp_path, duration_case):
    attempt = recording.RecordingAttempt(tmp_path)
    script = '''
import sys, json
from pathlib import Path
import cv2
import numpy as np
from cais_spade_llm.recovery_framework.gazebo_recording import _preview, _write_json, _H264Writer
root = Path(sys.argv[1])
writer = _H264Writer(root/'capture.partial.mp4', 15., (640, 360))
for i in range(30):
    writer.write(np.full((360, 640, 3), 100 + i, dtype=np.uint8))
writer.release()
case = sys.argv[2]
_write_json(root/'capture.json', {'status':'captured', 'frames':30,
                                'wall_duration_sec':2. if case == 'legacy' else 4.2})
if case != 'legacy':
    rows = [{'frame':i, 'elapsed_sec':i/15, 'observed_at_unix':100+i/15} for i in range(30)]
    if case == 'invalid_frame_duration':
        rows[-1]['elapsed_sec'] += 3.
    (root/'frames.jsonl').write_text(''.join(json.dumps(row) + chr(10) for row in rows))
try:
    _preview(root)
except ValueError as error:
    assert case == 'invalid_frame_duration' and 'duration disagrees' in str(error)
else:
    assert case != 'invalid_frame_duration'
'''
    subprocess.run(['/usr/bin/python3', '-c', script, str(attempt.directory), duration_case],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=20)
    if duration_case == 'invalid_frame_duration':
        assert not (attempt.directory / 'video_validation.json').exists()
        return
    result = json.loads((attempt.directory / 'video_validation.json').read_text())
    assert result['preview_encoder'] == 'h264_nvenc'
    assert result['full_frames_decoded'] == 30
    assert result['preview_frames_decoded'] == 3
    assert result['duration_sec'] == pytest.approx(result['preview_duration_sec'] * 10)


def test_blank_rendering_cannot_start_production_recording(tmp_path):
    attempt = recording.RecordingAttempt(tmp_path)
    script = '''
import sys
from pathlib import Path
import numpy as np
from cais_spade_llm.recovery_framework import gazebo_recording as recording
class BlankWindow:
    def frame(self): return np.zeros((720, 1280, 3), dtype=np.uint8)
    def close(self): pass
recording.X11Capture = BlankWindow
try:
    recording._capture(Path(sys.argv[1]))
except RuntimeError as error:
    assert "blank" in str(error)
else:
    raise AssertionError("Blank Gazebo rendering was accepted")
'''
    subprocess.run(["/usr/bin/python3", "-c", script, str(attempt.directory)],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=20)
    assert not (attempt.directory / "ready.json").exists()
    assert not (attempt.directory / "capture.partial.mp4").exists()
    assert json.loads((attempt.directory / "capture.json").read_text())["status"] == "failed"


def test_encoder_failure_is_reported_and_owned_process_is_reaped(tmp_path, monkeypatch):
    process = Mock()
    process.wait.return_value = 1
    monkeypatch.setattr(recording.subprocess, 'Popen', Mock(return_value=process))
    writer = recording._H264Writer(tmp_path / 'capture.partial.mp4', 15., (640, 360))
    with pytest.raises(RuntimeError, match='encoder failed'):
        writer.release()
    process.stdin.close.assert_called_once()
    process.wait.assert_called_once_with(timeout=15)
    assert writer.log.closed
    writer.release()


@pytest.mark.parametrize('delay_stage', ['startup', 'shutdown'])
def test_encoder_startup_delay_precedes_capture_clock_and_ready(tmp_path, delay_stage):
    attempt = recording.RecordingAttempt(tmp_path)
    script = r"""
import sys, time, json
from pathlib import Path
import numpy as np
from cais_spade_llm.recovery_framework import gazebo_recording as recording
root = Path(sys.argv[1])
class Window:
    count = 0
    def frame(self):
        self.count += 1
        if self.count == 6:
            (root/'finish').touch()
        return np.tile(np.arange(640, dtype=np.uint8), (360, 1))[..., None].repeat(3, axis=2)
    def close(self): pass
recording.X11Capture = Window
stage = sys.argv[2]
method = 'wait_ready' if stage == 'startup' else 'release'
original = getattr(recording._H264Writer, method)
def delayed(self):
    original(self)
    time.sleep(2.2)
    if stage == 'startup':
        assert not (root/'ready.json').exists()
setattr(recording._H264Writer, method, delayed)
recording._capture(root)
setattr(recording._H264Writer, method, original)
recording._preview(root)
metadata = json.loads((root/'capture.json').read_text())
assert metadata['max_capture_gap_sec'] < 2.
assert metadata['wall_duration_sec'] < 2.
assert metadata['ended_at_unix'] - metadata['started_at_unix'] < 2.
if stage == 'shutdown':
    assert metadata['encoder_finalize_time_sec'] >= 2.2
"""
    subprocess.run(['/usr/bin/python3', '-c', script, str(attempt.directory), delay_stage],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=30)
    assert json.loads((attempt.directory / 'ready.json').read_text())['encoder'] == 'h264_nvenc'
    assert json.loads((attempt.directory / 'video_validation.json').read_text())['validated']


def _clock_observations(directory: Path) -> None:
    """Record a simulation that changes speed halfway through a four-second capture."""
    clocks = []
    frames = []
    for index in range(61):
        elapsed = index / 15
        simulation = min(elapsed, 2) * .2 + max(0, elapsed - 2) * .8
        clocks.append({'observed_at_unix': 100 + elapsed, 'simulation_time_sec': simulation})
        if index < 60:
            frames.append({'frame': index, 'observed_at_unix': 100 + elapsed,
                           'elapsed_sec': elapsed, 'repeated_frames': 0})
    for name, rows in (('clock_samples.jsonl', clocks), ('frames.jsonl', frames)):
        (directory / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_clock_pacing_tracks_speed_changes_instead_of_average_speed(tmp_path):
    _clock_observations(tmp_path)
    mapping = recording._simulation_clock_frames(tmp_path, fps=15, frame_count=60)
    # Advancing 0.1333 simulation seconds initially takes ten input frames.
    assert mapping['source_frame_indices'][:4] == [0, 10, 20, 30]
    # After the clock speeds up, the same advance takes two or three frames.
    steps = [b - a for a, b in zip(mapping['source_frame_indices'][3:],
                                  mapping['source_frame_indices'][4:])]
    assert set(steps) <= {2, 3}
    assert mapping['video_duration_sec'] == pytest.approx(mapping['simulation_duration_sec'] / 2,
                                                         abs=1 / 15)
    assert mapping['max_frame_clock_error_sec'] <= .027


@pytest.mark.parametrize('failure', ['reset', 'wall_reset', 'missing_start', 'missing_end', 'stale', 'nan'])
def test_clock_pacing_rejects_unverifiable_clock(tmp_path, failure):
    _clock_observations(tmp_path)
    path = tmp_path / 'clock_samples.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if failure == 'reset':
        rows[30]['simulation_time_sec'] = 0
    elif failure == 'wall_reset':
        rows[30]['observed_at_unix'] = rows[29]['observed_at_unix']
    elif failure == 'missing_start':
        rows = rows[1:]
    elif failure == 'missing_end':
        rows = rows[:-2]
    elif failure == 'stale':
        rows = rows[:10] + rows[50:]
    else:
        rows[30]['simulation_time_sec'] = float('nan')
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        recording._simulation_clock_frames(tmp_path, fps=15, frame_count=60)


def test_clock_pacing_selects_actual_captures_not_repeated_fill_frames(tmp_path):
    _clock_observations(tmp_path)
    path = tmp_path / 'frames.jsonl'
    frames = [json.loads(line) for line in path.read_text().splitlines()]
    frames.pop(10)
    path.write_text(''.join(json.dumps(row) + '\n' for row in frames))
    mapping = recording._simulation_clock_frames(tmp_path, fps=15, frame_count=60)
    assert 10 not in mapping['source_frame_indices']
    assert mapping['source_frame_indices'] == sorted(mapping['source_frame_indices'])


def test_clock_video_validation_is_required_before_any_promotion(tmp_path, monkeypatch):
    attempt = recording.RecordingAttempt(tmp_path, record_simulation_clock=True)
    attempt.process = Mock(poll=lambda: 0, returncode=0)
    for name in recording.VIDEO_NAMES:
        (attempt.directory / name).write_bytes(b'video')
    recording._write_json(attempt.directory / 'video_validation.json', {'validated': True})
    monkeypatch.setattr(attempt, '_spawn', lambda operation: Mock(poll=lambda: 0, returncode=0))
    with pytest.raises(RuntimeError, match='Simulation-clock video validation'):
        attempt.promote({'validated': True, 'run_id': 'eleven-complete'})
    assert not any((attempt.directory / name).exists() for name in recording.VIDEO_NAMES)
    assert not (attempt.directory / 'success.json').exists()


def test_clock_video_encodes_decodes_and_promotes_with_existing_copies(tmp_path):
    attempt = recording.RecordingAttempt(tmp_path, record_simulation_clock=True)
    _clock_observations(attempt.directory)
    script = '''
import sys
from pathlib import Path
import numpy as np
from cais_spade_llm.recovery_framework.gazebo_recording import _H264Writer, _write_json
root = Path(sys.argv[1])
writer = _H264Writer(root/'capture.partial.mp4', 15., (640, 360))
for i in range(60):
    writer.write(np.full((360, 640, 3), 100 + i, dtype=np.uint8))
writer.release()
_write_json(root/'capture.json', {'status':'captured', 'frames':60, 'wall_duration_sec':4.})
'''
    subprocess.run(['/usr/bin/python3', '-c', script, str(attempt.directory)],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=20)
    attempt.process = Mock(poll=lambda: 0, returncode=0)
    success = attempt.promote({'validated': True, 'run_id': 'eleven-complete'})
    assert success['simulation_clock_factor'] == 2
    assert success['video']['simulation_clock']['validated']
    assert success['video']['simulation_clock']['frames_decoded'] == 15
    for name in ('assembly.mp4', 'assembly-10x.mp4', 'assembly-rtf2.mp4'):
        assert (attempt.directory / name).stat().st_size > 0
    attempt.cancel()
    assert (attempt.directory / 'assembly-rtf2.mp4').exists()

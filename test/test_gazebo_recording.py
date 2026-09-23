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


def test_system_encoder_decodes_entire_video_and_labeled_preview(tmp_path):
    attempt = recording.RecordingAttempt(tmp_path)
    script = '''
import sys
from pathlib import Path
import cv2
import numpy as np
from cais_spade_llm.recovery_framework.gazebo_recording import _preview, _write_json
root = Path(sys.argv[1])
writer = cv2.VideoWriter(str(root/'capture.partial.mp4'), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*'avc1'), 15., (640, 360))
assert writer.isOpened()
for i in range(30):
    writer.write(np.full((360, 640, 3), 100 + i, dtype=np.uint8))
writer.release()
_write_json(root/'capture.json', {'status':'captured', 'frames':30, 'wall_duration_sec':2.})
_preview(root)
'''
    subprocess.run(['/usr/bin/python3', '-c', script, str(attempt.directory)],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=20)
    result = json.loads((attempt.directory / 'video_validation.json').read_text())
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

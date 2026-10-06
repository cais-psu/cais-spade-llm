"""Recording ownership, success gates, and complete H.264 video validation."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from cais_spade_llm.recovery_framework import gazebo_recording as recording
from cais_spade_llm.recovery_framework import failure_videos


def test_caption_timing_uses_captured_frames_despite_wall_clock_drift():
    frames = [dict(frame=0, observed_at_unix=100.),
              dict(frame=150, observed_at_unix=115.),
              dict(frame=300, observed_at_unix=130.)]
    assert failure_videos._caption_elapsed(115.02, frames, 15.) == 10.
    with pytest.raises(ValueError, match='no nearby recorded observation'):
        failure_videos._caption_elapsed(140., frames, 15.)


def test_navigation_readiness_timeout_does_not_accept_unobserved_controllers(tmp_path, monkeypatch):
    probe = Mock(side_effect=[subprocess.TimeoutExpired('readiness', 30),
                              subprocess.CompletedProcess([], 0, stdout=json.dumps({'ready': True}))])
    monkeypatch.setattr(failure_videos.subprocess, 'run', probe)
    asyncio.run(failure_videos._wait_for_navigation(tmp_path))
    observations = json.loads((tmp_path / 'navigation_readiness.json').read_text())['observations']
    assert [row['ready'] for row in observations] == [False, True]
    assert observations[0]['error'] == 'Nav2 readiness observation timed out'
    assert probe.call_count == 2


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


@pytest.mark.parametrize('change', ['run_id', 'injection', 'marker', 'cca'])
def test_failure_video_rejects_unconfirmed_or_unrelated_failure(change):
    sample = {
        'run_id': 'observed-run', 'diagnostic_cca_bypass': False,
        'unavailable_resources': ['Conveyor'],
        'fault': {'status': 'triggered', 'run_id': 'observed-run',
                  'scenario': 'Conveyor breakdown', 'visual': {'status': 'completed'},
                  'evidence': {'run_id': 'observed-run', 'injection_status': 'completed',
                               'checkpoint': 'after_M1_pick_before_release', 'source': 'M1',
                               'part_name': 'KET4_Square_4mm',
                               'resource_values_before': {'ur5e-1': {'held_part': 'KET4_Square_4mm'}},
                               'part_tracker_before': {'KET4_Square_4mm': {
                                   'location': 'ur5e-1', 'processCompleted': [{'process': 'trim', 'result': 'square'}]}}}},
    }
    config = {'scenario': 'Conveyor breakdown', 'resource_id': 'Conveyor',
              'checkpoint': 'after_M1_pick_before_release'}
    assert failure_videos.validate_failure(sample, config)['validated']
    if change == 'run_id':
        sample['fault']['evidence']['run_id'] = 'previous-run'
    elif change == 'injection':
        sample['fault']['evidence']['injection_status'] = 'failed'
    elif change == 'marker':
        sample['fault']['visual']['status'] = 'failed'
    else:
        sample['diagnostic_cca_bypass'] = True
    with pytest.raises(ValueError):
        failure_videos.validate_failure(sample, config)


@pytest.mark.parametrize('fraction', [None, .4, .6, float('nan')])
def test_machining_video_requires_observed_halfway_interruption(fraction):
    config = {'scenario': 'Machining breakdown during part processing', 'resource_id': 'M1',
              'checkpoint': 'during_processing_halfway'}
    evidence = {'run_id': 'observed-run', 'injection_status': 'completed',
                'checkpoint': config['checkpoint'], 'process_completed': False,
                'source': 'gazebo_workholding_observation', 'processing_fraction': .5}
    sample = {'run_id': 'observed-run', 'diagnostic_cca_bypass': False,
              'unavailable_resources': ['M1'],
              'fault': {'status': 'triggered', 'run_id': 'observed-run',
                        'scenario': config['scenario'], 'visual': {'status': 'completed'},
                        'evidence': evidence}}
    assert failure_videos.validate_failure(sample, config)['validated']
    evidence['processing_fraction'] = fraction
    with pytest.raises(ValueError, match='halfway'):
        failure_videos.validate_failure(sample, config)


@pytest.mark.parametrize('change', ['pickup', 'other_pickup', 'region', 'assumed_release'])
def test_slippage_video_requires_both_pickups_and_observed_region(change):
    config = {'scenario': 'Part slippage', 'resource_id': 'ur5e-3', 'part_name': 'KET4_Square_4mm',
              'checkpoint': 'after_both_pickups_before_place', 'drop_pose': {'x': 0., 'y': -.2, 'z': 1.04},
              'additional_condition': {'resource_id': 'ur5e-4', 'part_name': 'gear_small'}}
    evidence = {'run_id': 'observed-run', 'injection_status': 'completed', 'checkpoint': config['checkpoint'],
                'detach': {'success': True, 'release_mode': 'detached'},
                'observed_drop_pose': {'x': 0., 'y': -.2, 'z': 1.015},
                'resource_values_before': {'ur5e-3': {'held_part': 'KET4_Square_4mm'},
                                           'ur5e-4': {'held_part': 'gear_small'}}}
    sample = {'run_id': 'observed-run', 'diagnostic_cca_bypass': False,
              'unavailable_resources': ['ur5e-3'], 'values': {'ur5e-4': {'held_part': 'gear_small'}},
              'fault': {'status': 'triggered', 'run_id': 'observed-run', 'scenario': config['scenario'],
                        'visual': {'status': 'completed'}, 'evidence': evidence}}
    assert failure_videos.validate_failure(sample, config)['validated']
    if change == 'pickup':
        evidence['resource_values_before']['ur5e-3']['held_part'] = None
    elif change == 'other_pickup':
        evidence['resource_values_before']['ur5e-4']['held_part'] = None
    elif change == 'region':
        evidence['observed_drop_pose']['y'] = .2
    else:
        evidence['detach']['release_mode'] = 'assumed_released_if_open'
    with pytest.raises(ValueError):
        failure_videos.validate_failure(sample, config)


def test_failure_video_cannot_publish_unvalidated_capture(tmp_path):
    attempt = recording.RecordingAttempt(tmp_path)
    source = attempt.directory / 'capture.partial.mp4'
    source.write_bytes(b'owned original')
    output = tmp_path / 'failure-20x.mp4'
    with pytest.raises(ValueError, match='Observed run validation'):
        failure_videos.export_20x(attempt, output, 'Conveyor breakdown',
                                 {'validated': False, 'run_id': 'blocked'}, [])
    assert source.read_bytes() == b'owned original'
    assert not output.exists()


def test_verified_20x_export_decodes_and_removes_only_its_original(tmp_path):
    script = '''
import json, sys
from pathlib import Path
import numpy as np
from cais_spade_llm.recovery_framework.gazebo_recording import RecordingAttempt, _H264Writer, _write_json
from cais_spade_llm.recovery_framework.failure_videos import export_20x
root=Path(sys.argv[1])
attempt=RecordingAttempt(root)
writer=_H264Writer(attempt.directory/'capture.partial.mp4', 15., (640,360))
for index in range(60):
    writer.write(np.full((360,640,3), 80+index, dtype=np.uint8))
writer.release()
_write_json(attempt.directory/'capture.json', {'status':'captured','frames':60,'fps':15.})
(attempt.directory/'frames.jsonl').write_text(json.dumps({'frame':59,'elapsed_sec':59/15})+'\\n')
previous=root/'previous.mp4'
previous.write_bytes(b'preserve')
output=root/'videos'/'Conveyor breakdown-20x.mp4'
result=export_20x(attempt,output,'Conveyor breakdown', {'validated':True,'run_id':'encoding-fixture'},[])
assert abs(result['duration_sec']-result['source_duration_sec']/20) <= 2/15
assert result['frames_decoded'] >= 2 and output.exists()
assert not (attempt.directory/'capture.partial.mp4').exists()
assert previous.read_bytes() == b'preserve'
assert list(output.parent.iterdir()) == [output]
'''
    subprocess.run(['/usr/bin/python3', '-c', script, str(tmp_path)],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=30)


def test_combined_export_requires_every_trial_validation_before_reading_video(tmp_path):
    attempt = recording.RecordingAttempt(tmp_path)
    source = attempt.directory / 'capture.partial.mp4'
    source.write_bytes(b'preserve incomplete capture')
    trial = {'attempt': attempt, 'title': 'Incomplete trial', 'captions': [],
             'validation': {'validated': False, 'run_id': 'incomplete'}}
    with pytest.raises(ValueError, match='observed run validation'):
        failure_videos.export_combined_20x([trial], tmp_path / 'combined-20x.mp4', 'Part slippage')
    assert source.read_bytes() == b'preserve incomplete capture'
    assert not (tmp_path / 'combined-20x.mp4').exists()


def test_combined_20x_export_keeps_one_video_and_distinct_trial_evidence(tmp_path):
    script = '''
import json, sys
from pathlib import Path
import numpy as np
from cais_spade_llm.recovery_framework.gazebo_recording import RecordingAttempt, _H264Writer, _write_json
from cais_spade_llm.recovery_framework.failure_videos import export_combined_20x
root=Path(sys.argv[1])
trials=[]
for index, name in enumerate(('mutex', 'precedence', 'safe')):
    attempt=RecordingAttempt(root/name)
    writer=_H264Writer(attempt.directory/'capture.partial.mp4',15.,(640,360))
    for frame in range(60):
        writer.write(np.full((360,640,3),60+index*30+frame,dtype=np.uint8))
    writer.release()
    _write_json(attempt.directory/'capture.json',{'status':'captured','frames':60,'fps':15.})
    (attempt.directory/'frames.jsonl').write_text(json.dumps({'frame':59,'elapsed_sec':59/15})+'\\n')
    trials.append({'attempt':attempt,'title':name+' | synthetic encoding fixture',
                   'captions':[{'text':'Encoding test only','start':1.,'end':3.}],
                   'validation':{'validated':True,'run_id':'encoding-'+name}})
previous=root/'historical.mp4'
previous.write_bytes(b'preserve history')
output=root/'videos'/'part_slippage-20x.mp4'
result=export_combined_20x(trials,output,'Synthetic encoding test')
assert result['separately_staged_trials'] is True
assert [row['run_id'] for row in result['trials']] == ['encoding-mutex','encoding-precedence','encoding-safe']
assert abs(result['source_duration_sec']-12.) < .01
assert abs(result['duration_sec']-.6) <= 2/15
assert result['frames_decoded'] > 0
assert len(list(output.parent.glob('*.mp4'))) == 1
assert previous.read_bytes() == b'preserve history'
assert all(not (row['attempt'].directory/'capture.partial.mp4').exists() for row in trials)
assert Path(result['validation_file']).exists()
assert list(output.parent.iterdir()) == [output]
'''
    subprocess.run(['/usr/bin/python3', '-c', script, str(tmp_path)],
                   cwd=Path(__file__).resolve().parents[1], check=True, timeout=60)


def test_mutex_video_requires_cca_hold_and_subsequent_access():
    def sample(timestamp, first, second):
        return {'run_id': 'mutex-run', 'observed_at_unix': timestamp,
                'diagnostic_cca_bypass': False,
                'values': {'ur5e-3': {'resource_location': first},
                           'ur5e-4': {'resource_location': second}}}
    samples = [sample(10, 'home', 'assembly_board-v1'),
               sample(10.5, 'home', 'home'), sample(11, 'assembly_board-v1', 'home')]
    negotiations = [
        {'kind': 'CCA', 'timestamp': 10, 'task_ids': ['entry-3']},
        {'kind': 'candidate_held', 'task_id': 'entry-3',
         'decision': {'status': 'held', 'included_specifications': ['workspace_mutex'],
                      'counterexample': [{'action': 'entry'}],
                      'task_bindings': {'entry': {'resource_id': 'ur5e-3', 'event_name': 'place_approach',
                                                 'parameters': {'destination_location': 'assembly_board-v1'}}}}},
        {'kind': 'CCA', 'timestamp': 10.6, 'decision': {'decisions': {'allowed-entry-3': {'status': 'allowed'}}}},
        {'kind': 'task_sent', 'task_id': 'allowed-entry-3', 'timestamp': 10.7,
         'resource_id': 'ur5e-3', 'event_name': 'place_approach'},
    ]
    result = failure_videos.validate_mutex(samples, negotiations)
    assert result['waiting_robot'] == 'ur5e-3' and result['first_robot'] == 'ur5e-4'
    with pytest.raises(ValueError, match='No observed CCA mutex hold'):
        failure_videos.validate_mutex(samples, [])
    samples.append(sample(12, 'assembly_board-v1', 'assembly_board-v1'))
    with pytest.raises(ValueError, match='overlapping occupancy'):
        failure_videos.validate_mutex(samples, negotiations)


def test_mutex_recording_binds_entry_and_persistent_occupancy_to_exact_resources():
    from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor

    checker = OnlineSafetyMonitor({}, [failure_videos.mutex_rule()])
    assert checker._map_task_to_aps('ur5e-3@localhost', 'place_approach',
                                    {'destination_location': 'assembly_board-v1'}) == ['ap5']
    assert checker._map_task_to_aps('ur5e-3@localhost', 'place_approach',
                                    {'destination_location': 'Conveyor'}) == []
    assert checker._map_state_to_aps('ur5e-4@localhost', 'placed',
                                     {'resource_location': 'assembly_board-v1'}) == ['ap4']
    assert checker._map_state_to_aps('ur5e-4@localhost', 'home',
                                     {'resource_location': 'home'}) == []
    checker.resource_bindings = {'recovery-resource-3@localhost': 'ur5e-3',
                                 'recovery-resource-4@localhost': 'ur5e-4'}
    assert checker._map_task_to_aps('recovery-resource-3@localhost', 'place_approach',
                                    {'destination_location': 'assembly_board-v1'}) == ['ap5']
    assert checker._map_state_to_aps('recovery-resource-4@localhost', 'placed',
                                     {'resource_location': 'assembly_board-v1'}) == ['ap4']
    assert checker._map_task_to_aps('unregistered@localhost', 'place_approach',
                                    {'resource_id': 'ur5e-3', 'destination_location': 'assembly_board-v1'}) == []


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

from __future__ import annotations

"""Capture one selected failure in an owned simulation without LLM or recovery."""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.capture_files import (
    BELIEFS_NAME,
    RAW_SCHEMA_VERSION,
    write_capture,
)
from cais_spade_llm.recovery_framework.failure_checkpoints import (
    CHECKPOINTS,
    ROBOT_BREAKDOWN_GRIPPER_CONDITION,
    SUPPORTED_CHECKPOINTS,
)

logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str | None:
    """Hash a retained file without loading its full contents."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_record(path: Path, value: dict) -> None:
    """Publish an experiment record atomically."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
    temporary.replace(path)


def active_processes() -> list[dict]:
    """Find live UI, agent, or simulator processes without inspecting secrets."""
    matches = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            argv = [item.decode(errors='replace') for item in (directory / 'cmdline').read_bytes().split(b'\0') if item]
            comm = (directory / 'comm').read_text().strip()
        except (OSError, ProcessLookupError):
            continue
        if not argv or comm in {'bash', 'sh', 'codex', 'codex-linux-sandbox'}:
            continue
        text = ' '.join(argv)
        if 'cais_spade_llm.ui.xmpp_server_runner' in text:
            continue
        process_match = comm in {'gzserver', 'gzclient', 'move_group', 'robot_state_pub', 'controller_mana', 'ros2_control_no'}
        python_match = ('python' in comm and (
            'cais_spade_llm.ui_main' in argv or any(Path(item).name in {'ui_main.py', 'runtime.py'} for item in argv)
            or 'cais_spade_llm.recovery_framework.evaluation_capture' in argv
            or ('cais-spade-llm' in text and any('/capture.py' in item for item in argv))))
        launch_match = any(item == 'launch' for item in argv) and any('ros2' in item for item in argv[:2])
        if process_match or python_match or launch_match:
            matches.append({'pid': int(directory.name), 'comm': comm})
    return matches


def preflight(port: int, domain: int) -> dict:
    """Reject another runtime or occupied selected endpoints before starting."""
    processes = active_processes()
    with socket.socket() as probe:
        probe.settimeout(0.3)
        ui_running = probe.connect_ex(('127.0.0.1', 8080)) == 0
    with socket.socket() as probe:
        probe.settimeout(0.3)
        gazebo_running = probe.connect_ex(('127.0.0.1', port)) == 0
    result = {'processes': processes, 'ui_port_8080_active': ui_running,
              'gazebo_master_port_active': gazebo_running, 'ROS_DOMAIN_ID': domain,
              'GAZEBO_MASTER_URI': f'http://127.0.0.1:{port}'}
    if processes or ui_running or gazebo_running:
        raise RuntimeError('A separate UI/runtime or selected endpoint is active: ' + json.dumps(result))
    return result


def preserved_files() -> dict[str, str | None]:
    """Record immutable observations and normal UI configuration fingerprints."""
    paths = [ROOT / 'cais_spade_llm/user_verified_plan/index.json',
             ROOT / 'cais_spade_llm/initialization/tools.json',
             ROOT / 'cais_spade_llm/monitor/environment_runs/latest/run.json',
             ROOT / 'cais_spade_llm/monitor/environment_runs/.latest-owner.json']
    paths += list((ROOT / 'cais_spade_llm/monitor/environment_runs/latest/processes').glob('*.json'))
    paths += list((ROOT / 'cais_spade_llm/monitor/environment_runs/recovery').glob('capture_*/observations.json'))
    paths += list((ROOT / 'cais_spade_llm/initialization').glob('*.json'))
    paths += list((ROOT / 'cais_spade_llm/specification/products/geometry').glob('*recovery-framework.json'))
    for subdirectory in ('history', 'plan', 'state'):
        paths += [path for path in (ROOT / 'cais_spade_llm/monitor' / subdirectory).glob('*') if path.is_file()]
    return {str(path): sha256_file(path) for path in sorted(set(paths))}


def verify_capture(captured: dict, scenario: str) -> dict:
    """Require the selected reconciled checkpoint and preserve its exact facts.

    Args:
        captured: Saved runtime and physical observations.
        scenario: Exact selected failure scenario.

    Returns:
        Verified checkpoint identities and their source paths.
    """
    failure = captured.get('failure')
    setup = captured.get('setup')
    if not isinstance(failure, dict) or not isinstance(setup, dict):
        raise ValueError('Captured failure and setup are missing')
    configured = setup.get('failure_scenario')
    if not isinstance(configured, dict) or scenario not in CHECKPOINTS:
        raise ValueError('Configured failure scenario is missing')
    if (captured.get('scenario') != scenario or configured.get('scenario') != scenario
            or failure.get('checkpoint') != configured.get('checkpoint')
            or failure.get('checkpoint') not in SUPPORTED_CHECKPOINTS[scenario]
            or failure.get('injection_status') != 'completed'
            or failure.get('run_id') != captured.get('run_id')
            or setup.get('execution_mode') != 'simulation'):
        raise ValueError('The captured failure is not the selected reconciled simulation checkpoint')
    part = failure.get('part_name')
    if configured.get('checkpoint') == 'during_processing':
        records = captured.get('failure_processing_observations', captured.get('processing_observations', {}))
        for rid in ('M1', 'M2'):
            record = records.get(rid, {})
            job = record.get('jobs', {}).get(record.get('active_job'), {})
            if (job.get('job_state') != 'running' or job.get('process_completed') is not False
                    or not 0 <= job.get('processing_fraction', -1) < 1):
                raise ValueError('Failure capture lacks overlapping unfinished processing: ' + rid)
        if captured.get('capture_checkpoint'):
            from cais_spade_llm.recovery_framework.recovery_observations import (
                processing_capture_compatibility,
            )

            if processing_capture_compatibility(captured)['status'] != 'supported':
                raise ValueError('Frozen processing checkpoint is inconsistent')
            record = captured['processing_observations']['M2']
            job = record['jobs'].get(record['active_job'], {})
            if job.get('job_state') != 'running' or job.get('process_completed') is not False:
                raise ValueError('M2 is no longer processing at the capture checkpoint')
    if not isinstance(part, str) or part not in captured.get('part_tracker', {}):
        raise ValueError('The checkpoint does not identify a captured affected part')
    if scenario == 'ur5e-1 breakdown':
        _verify_robot_breakdown_capture(captured)
    if scenario == 'Conveyor breakdown' and (configured.get('part_name') is not None
                                             or configured.get('additional_condition') is not None):
        _verify_conveyor_capture(captured)
    return {'checkpoint': failure['checkpoint'], 'run_id': failure['run_id'],
            'part_name': part, 'resource_id': configured['resource_id'],
            'source_paths': ['/failure', '/setup/failure_scenario', '/part_tracker/' + part]}


def _verify_conveyor_capture(captured: dict) -> None:
    """Require ur5e-1's retained pickup from completed M1 and the unavailable Conveyor transport."""
    from cais_spade_llm.resources.recovery_geometry import _checkpoint

    configured = captured['setup']['failure_scenario']
    failure = captured['failure']
    part = failure['part_name']
    other = configured.get('additional_condition')
    pickups = failure.get('pickups', {})
    machine = pickups.get('ur5e-1', {}).get('before', {}).get('M1', {})
    if machine.get('resource_state') != 'completed' or machine.get('part_name') != part:
        raise ValueError('Conveyor capture requires acknowledged pickup from completed M1')
    retained = [('ur5e-1', configured['part_name'], 'M1')]
    if other is not None:
        retained.append((other['resource_id'], other['part_name'], '3D Printing Station'))
    for rid, held, origin in retained:
        acknowledgement = pickups.get(rid, {}).get('acknowledgement', {})
        if (captured.get('resource_states', {}).get(rid, {}).get('held_part') != held
                or captured['part_tracker'].get(held, {}).get('location') != rid
                or acknowledgement.get('run_id') != captured['run_id']
                or acknowledgement.get('resource_id') != rid
                or acknowledgement.get('event_name') != 'pick_grasp'
                or acknowledgement.get('evidence') != 'resource'
                or acknowledgement.get('parameters', {}).get('part_name') != held
                or acknowledgement.get('parameters', {}).get('origin_resource_location') != origin
                or not pickups[rid].get('observations')):
            raise ValueError('Conveyor capture lacks acknowledged retained pickup: ' + rid)
        if any(row.get('process') == 'assembly'
               for row in captured['part_tracker'][held].get('processCompleted', [])):
            raise ValueError('Conveyor capture requires each retained part unassembled: ' + held)
    if (part != configured['part_name'] or 'Conveyor' not in captured.get('unavailable_resources', [])
            or captured.get('inputs', {}).get('scene', {}).get('Conveyor', {}).get('transport_enabled') is not False):
        raise ValueError('Conveyor capture does not retain the unavailable transport')
    if (captured.get('schema_version') == RAW_SCHEMA_VERSION
            and captured.get('fault_reports', {}).get('Conveyor', {}).get('available') is not False):
        raise ValueError('Conveyor capture lacks the Conveyor fault report')
    physical = captured['physical_evidence']
    if other is not None:
        observed = _checkpoint(physical['snapshot'], physical['geometry'], physical['resource_models'], index=0)
        if observed['part_region_occupancy']['assembly_board-v1'][other['part_name']] is not False:
            raise ValueError('Conveyor capture requires the printed part observed outside assembly_board-v1')
        return
    # With no held peer, the fault must find no task in progress and every other ordered part
    # measured as fixed to the board, so only the machined part's delivery remains.
    if failure.get('pending_tasks'):
        raise ValueError('Conveyor capture requires no task in progress at the fault')
    models = {name: row.get('name') for name, row in physical.get('parts', {}).items()}
    board = models.get(physical.get('product_name'))
    fixed = {row.get('model2') for row in (physical.get('physics_evidence') or {}).get('attachments') or []
             if board and row.get('model1') == board and (row.get('joint_evidence') or {}).get('fixed_joint') is True}
    for name in captured.get('requirements', {}):
        if name != part and (models.get(name) not in fixed or not any(
                row.get('process') == 'assembly'
                for row in captured['part_tracker'].get(name, {}).get('processCompleted', []))):
            raise ValueError('Conveyor capture requires every other part assembled and fixed to the board: ' + name)


def _verify_robot_breakdown_capture(captured: dict) -> None:
    """Require the observed failed approach, retained part, and frozen M1 occupancy."""
    from cais_spade_llm.recovery_framework.recovery_observations import checkpoint_fact_fingerprint
    from cais_spade_llm.resources.recovery_geometry import _checkpoint

    failure = captured['failure']
    part = failure['part_name']
    values = captured.get('resource_states', {})
    robot, machine = values.get('ur5e-1', {}), values.get('M1', {})
    # Schema 2 keeps the robot's own fault report apart from the product's beliefs.
    reported = (captured.get('fault_reports', {}).get('ur5e-1', {}).get('reported_state')
                if captured.get('schema_version') == RAW_SCHEMA_VERSION else robot.get('resource_state'))
    if (captured['setup']['failure_scenario']['resource_id'] != 'ur5e-1'
            or reported != 'failed' or robot.get('held_part') is not None
            or failure.get('resource_values_before', {}).get('ur5e-1', {}).get('resource_state') != 'at_pick'
            or machine.get('resource_state') != 'completed' or machine.get('part_name') != part
            or captured['part_tracker'][part].get('location') != 'M1'
            or 'ur5e-1' in captured.get('unavailable_resources', [])):
        raise ValueError('Robot breakdown must retain a failed empty-gripper approach and the completed part at M1')
    for rid, name in (('M1', 'complete_job'), ('ur5e-1', 'pick_approach')):
        record = failure.get(name, {})
        task = record.get('acknowledgement', {})
        if (task.get('resource_id') != rid or task.get('event_name') != name
                or task.get('run_id') != captured['run_id'] or task.get('evidence') != 'resource'
                or task.get('parameters', {}).get('part_name') != part or not record.get('observations')
                or record not in captured.get('transitions', [])):
            raise ValueError('Robot breakdown lacks its current acknowledged ' + name)
    approach = failure['pick_approach']
    if (approach['acknowledgement']['parameters'].get('origin_resource_location') != 'M1'
            or approach.get('before', {}).get('M1', {}).get('resource_state') != 'completed'
            or approach['before']['M1'].get('part_name') != part):
        raise ValueError('Robot breakdown approach must target the completed part at M1')
    physical = captured['physical_evidence']
    physical_robot = physical.get('snapshot', {}).get('resources', {}).get('ur5e-1', {})
    if ('held_part' not in physical_robot or physical_robot['held_part'] is not None
            or physical.get('resources', {}).get('ur5e-1', {}).get('custody_complete') is not True):
        raise ValueError('Robot breakdown requires the observed empty attachment inventory')
    frozen = captured.get('capture_checkpoint', {})
    native = physical.get('physics_evidence', {})
    if (not frozen.get('instance_id') or frozen.get('world_paused') is not True
            or frozen.get('native_observations_unchanged') is not True
            or native.get('world_paused') is not True
            or any(native.get(key) != frozen.get(key) for key in ('instance_id', 'simulation_time'))
            or frozen.get('facts_fingerprint') != checkpoint_fact_fingerprint(captured)):
        raise ValueError('Robot breakdown requires unchanged observations at the frozen checkpoint')
    observed = _checkpoint(physical['snapshot'], physical['geometry'], physical['resource_models'], index=0)
    if not any(observed['region_occupancy'].get(region, {}).get('ur5e-1') is True
               for region in ('M1.front_access', 'M1.side_access', 'M1.workholding')):
        raise ValueError('NEEDS_CONTEXT: ur5e-1 has no observed occupancy of the protected M1 area')
    if not frozen.get('machine_observations'):
        return
    # The gripper-fault design: M2 still trims while the KMR, idle at a dock, can take over at M1.
    report = captured.get('fault_reports', {}).get('ur5e-1', {})
    if report.get('source') != 'gripper object detection':
        raise ValueError('Robot breakdown fault must come from the gripper object detection')
    if report.get('gripper') != ROBOT_BREAKDOWN_GRIPPER_CONDITION:
        raise ValueError('Robot breakdown fault must report the gripper hardware fault: ' + ROBOT_BREAKDOWN_GRIPPER_CONDITION)
    records = captured.get('processing_observations', {})
    m1, m2 = records.get('M1', {}), records.get('M2', {})
    running = m2.get('jobs', {}).get(m2.get('active_job'), {})
    if (m1.get('active_job') is not None
            or not any(job.get('part_name') == part and job.get('process_completed') is True
                       for job in m1.get('jobs', {}).values())
            or running.get('job_state') != 'running' or running.get('process_completed') is not False):
        raise ValueError('Robot breakdown requires M1 finished and M2 still processing at the frozen checkpoint')
    kmr = values.get('KMR', {})
    location = kmr.get('resource_location')
    docks = {'Storage': captured['inputs']['scene']['Storage']['KMR_docking_pose'],
             **{row['resource_id']: row['KMR_docking_pose'] for row in captured['inputs']['scene']['machines']}}
    base = physical.get('resources', {}).get('KMR', {}).get('base_pose') or []
    if (kmr.get('resource_state') != 'idle' or kmr.get('held_part') is not None or location not in docks
            or len(base) < 2 or math.dist(base[:2], docks[location][:2]) > .05):
        raise ValueError('Robot breakdown requires the KMR idle at a dock, not on a route')


async def _freeze_robot_capture(runtime, captured: dict, directory: Path) -> dict:
    """Freeze the owned stopped robot scene and recapture its measured facts."""
    from cais_spade_llm.recovery_framework.recovery_observations import (
        _cancel_checkpoint_motion,
        capture_observations,
        fingerprint,
    )

    preparation = getattr(runtime, 'recovery_observation_preparation', None)
    reader = getattr(preparation, 'reader', None)
    if reader is None or not runtime.stopped or not runtime.nominal_idle.is_set():
        raise ValueError('Robot checkpoint requires its stopped runtime and registered observation reader')
    revision = runtime.context.revision
    runtime.recovery_checkpoint_capturing = True
    try:
        cancellation = await _cancel_checkpoint_motion(runtime, reader, directory)
    finally:
        runtime.recovery_checkpoint_capturing = False
    if runtime.context.revision != revision:
        raise ValueError('Robot checkpoint changed during motion cancellation')
    runtime.recovery_capture_frozen = True
    before = await asyncio.to_thread(reader.set_world_paused, True)
    write_record(directory / 'checkpoint_native_before.json', before)
    fresh = await asyncio.to_thread(capture_observations, runtime, runtime.part_slippage_recovery._failure)
    fresh['pending_nodes'] = deepcopy(captured['pending_nodes'])
    fresh['additional_observations'] = deepcopy(captured.get('additional_observations', {}))
    fresh['checkpoint_motion_cancellation'] = cancellation
    after = await asyncio.to_thread(reader.snapshot, refresh=True)
    write_record(directory / 'checkpoint_native_after.json', after)
    unchanged = fingerprint({key: value for key, value in before.items() if key != 'observed_monotonic'}) == \
        fingerprint({key: value for key, value in after.items() if key != 'observed_monotonic'})
    if (not unchanged or after.get('world_paused') is not True or runtime.context.revision != revision):
        raise ValueError('Robot physical checkpoint changed during its frozen capture')
    fresh['capture_checkpoint'] = {'checkpoint_id': uuid4().hex, 'run_id': runtime.context.run_id,
        'context_revision': revision, 'instance_id': before['instance_id'],
        'simulation_time': before['simulation_time'], 'world_paused': True,
        'native_observations_unchanged': True}
    readiness = stationary_readiness(fresh)
    if not readiness['ready']:
        write_record(directory / 'last_unresolved_capture.json', fresh)
        raise ValueError('Frozen robot checkpoint lacks stationary evidence: ' + repr(readiness['missing']))
    write_record(directory / 'checkpoint_observations_raw.json', fresh)
    return fresh


def _owned_gazebo_environment(bridge, *, proc_root: Path = Path('/proc')) -> tuple[int, dict[str, str]]:
    """Read launch-added model and plugin paths only from the owned server."""
    groups = {os.getpgid(proc.pid) for proc in bridge._ros2_procs.values() if proc.poll() is None}
    servers = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if ((directory / 'comm').read_text().strip() != 'gzserver'
                    or os.getpgid(int(directory.name)) not in groups):
                continue
            if directory.stat().st_uid != os.getuid():
                raise ValueError('Owned Gazebo server has a different process owner')
            entries = (directory / 'environ').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        environment = dict(entry.decode(errors='surrogateescape').split('=', 1)
                           for entry in entries if b'=' in entry)
        servers.append((int(directory.name), environment))
    if len(servers) != 1:
        raise ValueError('Scene image requires exactly one owned Gazebo server')
    return servers[0]


def _capture_gazebo_frame(window: int, path: Path):
    """Read X11 in a child process because a mapping error can exit Xlib."""
    import cv2

    code = '''
import sys
import cv2
from cais_spade_llm.recovery_framework.gazebo_recording import X11Capture
camera = X11Capture()
try:
    if camera.window != int(sys.argv[1], 16):
        raise RuntimeError('Owned Gazebo window changed')
    if not cv2.imwrite(sys.argv[2], camera.frame()):
        raise RuntimeError('Gazebo frame was not written')
finally:
    camera.close()
'''
    result = subprocess.run([sys.executable, '-c', code, hex(window), str(path)],
                            capture_output=True, text=True, timeout=10., cwd=ROOT)
    if result.returncode:
        raise RuntimeError('Gazebo frame read failed: ' + result.stderr[-400:])
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError('Owned Gazebo frame is unavailable')
    return frame


def capture_scene_image(bridge, runtime, captured: dict, observation_path: Path) -> dict:
    """Save the final Gazebo window only when it belongs to this capture launch.

    Args:
        bridge: SystemBridge owning this capture's launch processes.
        runtime: Stopped runtime at the observed failure checkpoint.
        captured: Saved observations identifying the failure and physical scene.
        observation_path: Final saved observation file beside the image.

    Returns:
        Image provenance bound to the owned window, run, and saved observations.
    """
    import cv2

    if not runtime.stopped or not runtime.nominal_idle.is_set():
        raise ValueError('Scene image requires the stopped, idle owned runtime')
    if runtime.conveyor_fault.snapshot().get('visual', {}).get('status') != 'completed':
        raise ValueError('Scene image requires the acknowledged failure marker')
    frozen = captured.get('capture_checkpoint') if captured.get('scenario') == 'ur5e-1 breakdown' else None
    # A machine checkpoint unpauses the world after its frozen record; only a robot freeze is still paused.
    paused_freeze = frozen is not None and not frozen.get('machine_observations')
    if paused_freeze:
        native = runtime.recovery_observation_preparation.reader.snapshot(refresh=True)
        if (native.get('world_paused') is not True
                or any(native.get(key) != frozen.get(key) for key in ('instance_id', 'simulation_time'))):
            raise ValueError('Scene image no longer matches the frozen robot checkpoint')
    name = 'capture_gazebo_gui'
    server_pid, environment = _owned_gazebo_environment(bridge)
    command = ['gzclient', '--gui-client-plugin=libgazebo_ros_eol_gui.so']
    window = None
    frame_path = observation_path.parent / 'failure_scene.frame.png'
    with (observation_path.parent / 'gzclient.log').open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, env=environment,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    bridge._ros2_procs[name] = process
    try:
        bridge._register_ui_process(name, process, shlex.join(command))
        deadline = time.monotonic() + 30.
        while window is None:
            if process.poll() is not None:
                raise RuntimeError('Owned Gazebo GUI exited before its window became available')
            tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True, timeout=5)
            windows = re.findall(r'(0x[0-9a-f]+) "Gazebo"', tree)
            if len(windows) == 1:
                window = int(windows[0], 16)
            elif time.monotonic() >= deadline:
                raise RuntimeError('Owned Gazebo window did not become available')
            else:
                time.sleep(.1)
        property_text = subprocess.check_output(
            ['xprop', '-id', hex(window), '_NET_WM_PID'], text=True, timeout=5)
        match = re.search(r'=\s*(\d+)', property_text)
        if match is None:
            raise ValueError('Gazebo window process ownership is unavailable')
        pid = int(match.group(1))
        groups = {os.getpgid(proc.pid) for proc in bridge._ros2_procs.values() if proc.poll() is None}
        if os.getpgid(pid) not in groups:
            raise ValueError('Gazebo window belongs to a different launch')
        # A mapped window can precede scene rendering by many seconds on WSL.
        deadline = time.monotonic() + 180.
        rendered_since = None
        while True:
            if process.poll() is not None:
                raise RuntimeError('Owned Gazebo GUI exited before scene rendering')
            try:
                frame = _capture_gazebo_frame(window, frame_path)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                logger.debug('Gazebo frame is not ready: %s', exc)
                frame = None
            now = time.monotonic()
            if frame is not None:
                height, width = frame.shape[:2]
                scene = frame[height // 8:7 * height // 8, width // 4:7 * width // 8]
            if frame is not None and scene.size and (scene > 16).mean() > .05 and scene.std() > 8.:
                if rendered_since is None:
                    rendered_since = now
                if now - rendered_since >= 5.:
                    break
            else:
                rendered_since = None
            if now >= deadline:
                raise TimeoutError('Owned Gazebo window did not render the failure scene')
            time.sleep(.5)
        temporary = observation_path.parent / 'failure_scene.partial.png'
        destination = observation_path.parent / 'failure_scene.png'
        if not cv2.imwrite(str(temporary), frame):
            raise RuntimeError('Gazebo scene image was not written')
        temporary.replace(destination)
        record = {'path': str(destination), 'sha256': sha256_file(destination),
            'width': frame.shape[1], 'height': frame.shape[0], 'window_id': hex(window),
            'server_pid': server_pid,
            'window_pid': pid, 'window_process_group': os.getpgid(pid),
            'captured_at': datetime.now(timezone.utc).isoformat(), 'run_id': captured['run_id'],
            'source': str(observation_path), 'source_sha256': sha256_file(observation_path),
            'observation_captured_at': captured['captured_at'],
            'physical_evidence_fingerprint': captured['physical_evidence']['evidence_fingerprint'],
            'rendering': {'verified': True, 'pixel_stddev': float(scene.std()),
                          'nonblack_fraction': float((scene > 16).mean())},
            'execution_authorized': False}
        if paused_freeze:
            native = runtime.recovery_observation_preparation.reader.snapshot(refresh=True)
            if (native.get('world_paused') is not True
                    or any(native.get(key) != frozen.get(key) for key in ('instance_id', 'simulation_time'))):
                raise ValueError('Robot checkpoint changed while its scene image was captured')
        if frozen is not None:
            record['capture_checkpoint'] = deepcopy(frozen)
        write_record(observation_path.parent / 'failure_scene.json', record)
        return record
    finally:
        frame_path.unlink(missing_ok=True)
        bridge.ros2_stop(name, reason='owned_failure_scene_saved')


def stationary_readiness(captured: dict) -> dict:
    """Require every unheld part's unchanged declared rest checks to succeed."""
    physical = captured['physical_evidence']
    snapshot = physical.get('snapshot') or {}
    resources = snapshot.get('resources') or {}
    held = {row.get('held_part') for row in resources.values()}
    free = set(snapshot.get('parts') or {}) - held
    support = physical.get('stationary_part_evidence') or {}
    reasons = []
    rows = {}
    for part in sorted(free):
        entry = support.get(part) or {}
        rows[part] = {key: deepcopy(entry[key]) for key in
            ('status', 'reason', 'source', 'link_speeds', 'model_static', 'current_pose') if key in entry}
        if entry.get('status') != 'supported':
            reasons.append('stationary_part_evidence/' + part)
    if set(physical.get('stationary_parts', [])) != free:
        reasons.append('stationary_parts does not cover exactly all unheld parts')
    reasons.extend(physical.get('unresolved') or [])
    if set(physical.get('resources', {})) != set(captured['jids']):
        reasons.append('physical resource identities incomplete')
    for rid, model in physical.get('resource_models', {}).items():
        names = model.get('configuration', {}).get('joint_names')
        if names:
            owner = resources.get(rid) or {}
            if owner.get('joint_names') != names or len(owner.get('joint_positions', [])) != len(names):
                reasons.append(rid + ' complete joint configuration missing')
            if not physical.get('resources', {}).get(rid, {}).get('custody_complete'):
                reasons.append(rid + ' complete attachment inventory missing')
    return {'ready': not reasons, 'missing': reasons, 'parts': rows,
        'captured_at': captured['captured_at'], 'source_revision': captured['context_revision']}


def recovery_capture_ready(runtime) -> bool:
    """Wait for the installed preparation task instead of racing begin() capture."""
    recovery = getattr(runtime, 'part_slippage_recovery', None)
    if recovery is None:
        return False
    state = recovery._state
    if state.get('status') in {'blocked', 'stopped'}:
        raise RuntimeError('Input preparation failed: ' + str(state.get('message')))
    task = recovery._task
    if task is None or not task.done():
        return False
    if task.cancelled():
        raise RuntimeError('Owned input preparation was cancelled')
    error = task.exception()
    if error is not None:
        raise RuntimeError(f'Owned input preparation failed: {type(error).__name__}: {error}') from error
    if state.get('status') != 'outline_ready':
        return False
    if not recovery._observations.get('prepared_recovery_request'):
        raise ValueError('Preparation reported outline_ready without prepared_recovery_request')
    return bool(recovery._failure and runtime.stopped and runtime.nominal_idle.is_set())


def prepare_private_workspace(directory: Path) -> Path:
    """Keep relative monitor writes private while reading the unchanged project inputs."""
    workspace = directory / 'workspace'
    package = workspace / 'cais_spade_llm'
    package.mkdir(parents=True, exist_ok=True)
    for source in ROOT.iterdir():
        if source.name in {'cais_spade_llm', '.git', '.venv', '.codex', '.agents'}:
            continue
        target = workspace / source.name
        if not target.exists():
            target.symlink_to(source, target_is_directory=source.is_dir())
    for source in (ROOT / 'cais_spade_llm').iterdir():
        target = package / source.name
        if source.name in {'monitor', 'log'}:
            target.mkdir(exist_ok=True)
        elif not target.exists():
            target.symlink_to(source, target_is_directory=source.is_dir())
    return workspace


def install_capture_guards() -> list[str]:
    """Block generation and recovery dispatch while allowing nominal failure setup."""
    from cais_spade_llm.agents.shared_information.llm_agent import LlmAgent
    from cais_spade_llm.recovery_framework import slippage_execution
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery

    calls: list[str] = []

    def blocked(name):
        async def reject(*args, **kwargs):
            calls.append(name)
            raise RuntimeError('Capture-only guard rejected ' + name)
        return reject

    for method in ('ask_llm', 'ask_llm_structured'):
        setattr(LlmAgent, method, blocked('LlmAgent.' + method))
    for method in ('_run_stage', '_generate_primitives', '_execute'):
        setattr(PartSlippageRecovery, method, blocked('PartSlippageRecovery.' + method))
    slippage_execution.execute_generated_recovery = blocked('execute_generated_recovery')
    return calls


def scope_bridge_cleanup(bridge, directory: Path) -> None:
    """Own only this harness's launch group instead of killing every matching process.

    The bridge's normal startup and stop kill any process whose command mentions Gazebo and
    delete shared DDS memory, which also kills a recorder or another chat's simulation.

    Args:
        bridge: The harness's own SystemBridge.
        directory: Output folder that holds the owned-process registry.
    """
    from cais_spade_llm.ui.process_registry import UIProcessRegistry

    bridge._ui_process_registry = UIProcessRegistry(directory / 'owned_processes.json')
    # The tracked launch stop already sends SIGINT/SIGKILL to the owned group.
    def skip_global_cleanup(*args, **kwargs):
        logger.info('Skipped global cleanup; only the experiment launch group is owned')

    for name in ('_kill_stale_gazebo_helpers', '_force_kill_gazebo_core',
                 '_force_kill_digital_twin_helpers', '_cleanup_shm_and_tmp'):
        setattr(bridge, name, skip_global_cleanup)


def configure_bridge(bridge, directory: Path, domain: int, port: int, scenario: str) -> None:
    """Scope cleanup, process ownership, logs, and bundle state to this harness."""
    scope_bridge_cleanup(bridge, directory)
    bridge._sim_readiness_diagnostic_path = directory / 'readiness_probes.jsonl'
    bridge._archive_monitors = lambda: {'history': 0, 'plan': 0, 'state': 0}
    bridge.ROS2_LAUNCH_CMDS = dict(bridge.ROS2_LAUNCH_CMDS)
    bridge.ROS2_LAUNCH_CMDS['gazebo_dual'] = (
        'ros2 launch cais_lab_robotics recovery_framework_gazebo.launch.py '
        'world_file:=table_recovery_framework.world run_perception:=false '
        'include_assembly_parts:=true include_loose_parts:=true launch_rviz:=false '
        'recovery_observations:=true launch_nav2:=true launch_gazebo_gui:=false '
        '> ' + shlex.quote(str(directory / 'gazebo.log')) + ' 2>&1')
    bridge._ROS2_ENV += (f'export ROS_DOMAIN_ID={domain}; '
                         f'export GAZEBO_MASTER_URI=http://127.0.0.1:{port}; '
                         'export ROS_LOG_DIR=' + shlex.quote(str(directory / 'ros_logs')) + '; ')


def install_condition_setup(bridge, directory: Path, scenario: str, condition: str, seed: int | None = None) -> None:
    """Apply only the selected private capture setup before its owned launch."""
    from cais_spade_llm.recovery_framework import part_slippage_example
    from cais_spade_llm.recovery_framework.evaluation_conditions import condition_setup

    original = part_slippage_example.example_setup

    def selected_setup(**kwargs):
        if kwargs.get('scenario') != scenario:
            raise ValueError('Capture setup changed its selected scenario')
        setup = condition_setup(original(**kwargs), scenario, condition, directory)
        if seed is not None:
            # Only a randomly timed failure draws a seed; a selected seed reproduces that timing.
            if 'seed' not in setup['failure_scenario']:
                raise ValueError('The selected failure has no randomly chosen time to seed')
            setup['failure_scenario']['seed'] = seed
        scene = Path(setup['scene_file'])
        if not scene.is_absolute():
            scene = ROOT / scene
        if not scene.is_file():
            raise ValueError('Capture scene configuration is missing')
        command = bridge.ROS2_LAUNCH_CMDS['gazebo_dual']
        command = command.replace(' > ', ' robots_file:=' + shlex.quote(str(scene.resolve())) + ' > ', 1)
        bridge.ROS2_LAUNCH_CMDS['gazebo_dual'] = command
        write_record(directory / 'capture_setup.json', {'setup': setup, 'scene_file': str(scene.resolve()),
            'scene_sha256': sha256_file(scene)})
        return setup

    part_slippage_example.example_setup = selected_setup


def capture_implementation(directory: Path, purpose: str) -> dict | None:
    """Freeze implementation only for explicitly reserved evaluation captures."""
    if purpose == 'development':
        return None
    if purpose != 'evaluation':
        raise ValueError('Capture purpose must be development or evaluation')
    from cais_spade_llm.recovery_framework.recovery_evaluation import (
        _implementation_files,
        frozen_files,
        verify_frozen,
    )

    manifest = {'files': frozen_files([]),
                'implementation_files': sorted(str(path.relative_to(ROOT)) for path in _implementation_files(ROOT))}
    verify_frozen(manifest)
    write_record(directory / 'implementation_manifest.json', manifest)
    return manifest


async def run_capture(args) -> dict:
    """Run to the observed failure, capture once, and stop only owned work."""
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    implementation = capture_implementation(directory, args.purpose)
    preflight_record = preflight(args.gazebo_port, args.ros_domain)
    before = preserved_files()
    os.environ.update(ROS_DOMAIN_ID=str(args.ros_domain),
                      GAZEBO_MASTER_URI=f'http://127.0.0.1:{args.gazebo_port}',
                      CAIS_KEEP_GAZEBO_ON_EXIT='1')
    # Imported for its sys.path setup, which the bridge's top-level agent_creator import needs;
    # never invoke UI startup or cleanup.
    import cais_spade_llm.ui_main  # noqa: F401
    import cais_spade_llm.ui.bridge as bridge_module
    from cais_spade_llm.recovery_framework import environment_runtime
    from cais_spade_llm.recovery_framework.evaluation_conditions import (
        apply_changed_condition,
        verify_capture_condition,
    )
    from cais_spade_llm.recovery_framework.failure_grounding import apply_grounding
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import _prepare_saved_capture
    from cais_spade_llm.recovery_framework.part_slippage_example import example_for_bridge
    from cais_spade_llm.recovery_framework.recovery_evaluation import value_hash, verify_frozen
    from cais_spade_llm.recovery_framework.recovery_motion_preview import capture_preview_scene
    from cais_spade_llm.recovery_framework.recovery_observations import (
        RECORDS,
        capture_observations,
        offline_preparation_missing,
        save_native_preview_model_from_capture,
    )

    # These are process-local constants; production source and shared files stay unchanged.
    bridge_module._USER_VERIFIED_PLAN = directory / 'bundle_store'
    bridge_module._TOOLS_OUT = directory / 'tools.json'
    workspace = prepare_private_workspace(directory)
    bridge_module._MONITOR = workspace / 'cais_spade_llm/monitor'
    bridge_module._LOG_DIR = workspace / 'cais_spade_llm/log'
    bridge_module._RECOVERY_RUNTIME_DATA_DIR = bridge_module._MONITOR / 'debug/recovery_runtime_data'
    environment_runtime.RUN_DIRECTORY = directory / 'environment_runs'
    os.chdir(workspace)
    calls = install_capture_guards()
    bridge = bridge_module.SystemBridge()
    configure_bridge(bridge, directory, args.ros_domain, args.gazebo_port, args.scenario)
    install_condition_setup(bridge, directory, args.scenario, args.condition, getattr(args, 'seed', None))
    bridge.set_part_slippage_modes('manual', 'manual')
    example = example_for_bridge(bridge)
    from cais_spade_llm.recovery_framework.recovery_case_safety import default_safety_files
    example.set_options(source='gazebo', scenario=args.scenario,
                        safety_files=default_safety_files(args.scenario))
    cancelled = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, cancelled.set)
    report = {'status': 'starting', 'started_at': datetime.now(timezone.utc).isoformat(),
              'pid': os.getpid(), 'scenario': args.scenario, 'condition': args.condition,
              'purpose': args.purpose,
              'readiness_diagnostics': str(directory / 'readiness_probes.jsonl'),
              'preflight': preflight_record, 'preserved_before': before,
              'llm_calls': calls, 'recovery_execution_started': False,
              'outline_mode': 'manual', 'primitive_mode': 'manual'}
    write_record(directory / 'report.json', report)
    condition_record = None
    runtime = None
    try:
        # Repeat after Python imports, immediately before the launch operation.
        preflight(args.gazebo_port, args.ros_domain)
        if implementation is not None:
            verify_frozen(implementation)
        await bridge.start_failure_example()
        deadline = time.monotonic() + args.timeout
        while not cancelled.is_set():
            runtime = example._runtime
            recovery = getattr(runtime, 'part_slippage_recovery', None)
            state = recovery._state if recovery is not None else example._status
            progress = {key: deepcopy(state.get(key)) for key in
                        ('status', 'message', 'outline_status', 'primitive_status', 'observation_path')}
            progress.update(recorded_at=datetime.now(timezone.utc).isoformat(),
                            system_running=bridge.system_running, bridge_error=bridge.last_error)
            write_record(directory / 'state.json', progress)
            if calls:
                raise RuntimeError('Capture guard was invoked: ' + repr(calls))
            if runtime is not None and recovery_capture_ready(runtime):
                break
            if example._status.get('status') in {'blocked', 'stopped'}:
                raise RuntimeError(str(example._status.get('message')))
            if time.monotonic() >= deadline:
                raise TimeoutError('The owned example did not produce a prepared paused failure within the capture deadline')
            await asyncio.sleep(2)
        if cancelled.is_set():
            raise asyncio.CancelledError()
        if runtime.setup['execution_mode'] != 'simulation' or not runtime.stopped:
            raise ValueError('An owned paused simulation checkpoint is required')
        if args.condition == 'changed':
            report['condition_intervention'] = await bridge._run_on_agent_runtime(
                apply_changed_condition(runtime, args.scenario))
            write_record(directory / 'condition_intervention.json', report['condition_intervention'])
        settle_deadline = time.monotonic() + args.settle_timeout
        sample_index = 0
        frozen_checkpoint = recovery._observations.get('capture_checkpoint') is not None
        while True:
            captured = (deepcopy(recovery._observations) if frozen_checkpoint else
                        await asyncio.to_thread(capture_observations, runtime, recovery._failure))
            captured['pending_nodes'] = recovery._pending_nodes()
            captured['additional_observations'] = deepcopy(getattr(runtime, 'recovery_additional_observations', {}))
            readiness = stationary_readiness(captured)
            sample_index += 1
            readiness['sample_index'] = sample_index
            with (directory / 'stationary_samples.jsonl').open('a') as stream:
                stream.write(json.dumps(readiness, ensure_ascii=False) + '\n')
            write_record(directory / 'state.json', {'status': 'observing_stationary_parts', **readiness})
            if readiness['ready']:
                report['stationary_readiness'] = readiness
                break
            if frozen_checkpoint:
                write_record(directory / 'last_unresolved_capture.json', captured)
                # The checkpoint records current activity; future stationary
                # coverage remains a proposal-specific validation obligation.
                report['stationary_readiness'] = readiness
                break
            logger.info('Stationary sample %s unresolved: %s', sample_index, readiness['missing'])
            if cancelled.is_set():
                raise asyncio.CancelledError()
            if time.monotonic() >= settle_deadline:
                write_record(directory / 'last_unresolved_capture.json', captured)
                raise TimeoutError('Recorded rest checks did not pass within the observation deadline: ' + repr(readiness['missing']))
            await asyncio.sleep(5)

        if args.scenario == 'ur5e-1 breakdown' and not frozen_checkpoint:
            captured = await bridge._run_on_agent_runtime(_freeze_robot_capture(runtime, captured, directory))
        if frozen_checkpoint:
            scene = captured['planning_scene_capture']
        else:
            settings = runtime.context.inputs['scene']['safety_preparation']['outline_preview']
            scene = await capture_preview_scene(execution_mode='simulation',
                planning_scene_service=settings['planning_scene_service'],
                preview_parameter_service=settings['preview_parameter_service'],
                timeout_seconds=settings['timeout_seconds'])
            captured['planning_scene_capture'] = {**scene, 'status': 'captured', 'compatibility': 'not_checked',
                'physical_evidence_fingerprint': captured['physical_evidence']['evidence_fingerprint']}
        if args.scenario == 'ur5e-1 breakdown':
            from cais_spade_llm.recovery_framework.recovery_observations import (
                checkpoint_fact_fingerprint,
            )

            captured['capture_checkpoint']['facts_fingerprint'] = checkpoint_fact_fingerprint(captured)
            report['capture_checkpoint'] = deepcopy(captured['capture_checkpoint'])
            write_record(directory / 'checkpoint_observations_raw.json', captured)
        destination = RECORDS / ('capture_' + uuid4().hex)
        destination.mkdir()
        observation_path = destination / 'observations.json'
        report.update(observation_path=str(observation_path), capture_verification=verify_capture(captured, args.scenario))
        write_capture(observation_path, captured)
        write_record(directory / 'report.json', report)
        captured['prepared_recovery_request'] = await bridge._run_on_agent_runtime(
            _prepare_saved_capture(runtime, captured, recovery._observations['prepared_recovery_request'], observation_path))
        # The capture's beliefs.json stores its grounding; offline attempts reuse it.
        apply_grounding(captured['prepared_recovery_request'], captured)
        captured['missing'] = offline_preparation_missing(captured)
        captured['offline_ready'] = not captured['missing']
        write_capture(observation_path, captured)
        report['native_model_archive'] = await save_native_preview_model_from_capture(
            observation_path, cancel_event=cancelled)
        if getattr(args, 'scene_image', False):
            report['scene_image'] = await asyncio.to_thread(
                capture_scene_image, bridge, runtime, captured, observation_path)
        condition_evidence = verify_capture_condition(captured, args.scenario, args.condition)
        if not isinstance(condition_evidence, dict) or not condition_evidence:
            raise ValueError('Condition verification returned no factual evidence')
        condition_record = {'version': 1, 'source': str(observation_path),
            'source_sha256': sha256_file(observation_path),
            'preparation_sha256': sha256_file(observation_path.with_name(BELIEFS_NAME)), 'scenario': args.scenario,
            'condition': args.condition, 'purpose': args.purpose, 'capture_verified': True,
            'checkpoint_evidence': report['capture_verification'], 'condition_evidence': condition_evidence}
        if implementation is not None:
            condition_record.update(implementation_manifest=str(directory / 'implementation_manifest.json'),
                                    implementation_fingerprint=value_hash(implementation['files']))
        physical = captured['physical_evidence']
        missing = list(captured['missing'])
        if physical.get('unresolved'):
            missing.append({'physical_evidence.unresolved': physical['unresolved']})
        if len(physical.get('resources', {})) != len(runtime.jids):
            missing.append('Physical resource identities are incomplete')
        report.update(status='captured' if not missing else 'NEEDS_CONTEXT',
            observation_path=str(observation_path), observation_sha256=sha256_file(observation_path),
            preparation_sha256=sha256_file(observation_path.with_name(BELIEFS_NAME)),
            captured_at=captured['captured_at'], missing=missing,
            source_run_id=captured['run_id'], resources=list(runtime.jids),
            physical_evidence_fingerprint=physical.get('evidence_fingerprint'),
            scene_fingerprint=scene.get('scene_fingerprint'), model_fingerprint=scene.get('model_fingerprint'),
            source_revision=captured['context_revision'])
    except asyncio.CancelledError:
        report.update(status='cancelled')
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError, asyncio.TimeoutError, subprocess.SubprocessError) as exc:
        logger.exception('Capture failed')
        report.update(status='failed', error_type=type(exc).__name__, reason=str(exc))
    finally:
        try:
            try:
                if args.scenario == 'ur5e-1 breakdown' and getattr(runtime, 'recovery_capture_frozen', False):
                    reader = runtime.recovery_observation_preparation.reader
                    await bridge._run_on_agent_runtime(asyncio.to_thread(reader.set_world_paused, False))
                    runtime.recovery_capture_frozen = False
            finally:
                await bridge.part_slippage_action('stop')
                await asyncio.to_thread(bridge.ros2_stop_all, reason='owned_fresh_capture_complete')
                if getattr(bridge, '_xmpp_proc', None) is not None:
                    await bridge._stop_xmpp_server()
                example._restore_settings()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError, asyncio.TimeoutError, subprocess.SubprocessError) as exc:
            logger.exception('Owned cleanup failed')
            report['cleanup_error'] = str(exc)
        after = {path: sha256_file(Path(path)) for path in before}
        report.update(finished_at=datetime.now(timezone.utc).isoformat(), preserved_after=after,
                      preserved_files_unchanged=before == after, llm_calls=calls,
                      owned_launch_groups_remaining={name: proc.pid for name, proc in bridge._ros2_procs.items()
                                                     if proc.poll() is None})
        xmpp = getattr(bridge, '_xmpp_proc', None)
        report['owned_xmpp_server_remaining'] = xmpp.pid if xmpp is not None and xmpp.poll() is None else None
        if implementation is not None:
            try:
                verify_frozen(implementation)
            except ValueError as exc:
                report.update(status='failed', reason=str(exc), implementation_unchanged=False)
            else:
                report['implementation_unchanged'] = True
        if (condition_record is not None and report['status'] == 'captured'
                and report['preserved_files_unchanged'] and not report['owned_launch_groups_remaining']
                and not report['owned_xmpp_server_remaining']
                and 'cleanup_error' not in report):
            write_record(directory / 'condition_record.json', condition_record)
            report['evaluation_row'] = {'scenario': args.scenario, 'condition': args.condition,
                'source': condition_record['source'], 'condition_record': str(directory / 'condition_record.json')}
        write_record(directory / 'report.json', report)
    return report


def main() -> None:
    """Require explicit run selection; default inspection starts no processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario', choices=list(CHECKPOINTS), required=True)
    parser.add_argument('--condition', choices=('normal', 'changed'), required=True)
    parser.add_argument('--purpose', choices=('development', 'evaluation'), default='development')
    parser.add_argument('--ros-domain', type=int, default=89)
    parser.add_argument('--gazebo-port', type=int, default=11379)
    parser.add_argument('--timeout', type=float, default=600.)
    parser.add_argument('--settle-timeout', type=float, default=300.)
    parser.add_argument('--scene-image', action='store_true',
                        help='Save an owned Gazebo-window PNG at the final failure checkpoint')
    parser.add_argument('--seed', type=int, default=None,
                        help='Reproduce a randomly timed failure (Machining breakdown during processing)')
    args = parser.parse_args()
    if not 1 <= args.ros_domain <= 232 or not 1024 <= args.gazebo_port <= 65535:
        parser.error('Use an isolated ROS domain and an unprivileged Gazebo port')
    if any(not math.isfinite(value) or value <= 0 for value in (args.timeout, args.settle_timeout)):
        parser.error('Observation deadlines must be finite and positive')
    if args.seed is not None and not 0 <= args.seed < 2 ** 32:
        parser.error('The failure seed must be a 32-bit unsigned integer')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if not args.run:
        logger.info('Staged capture only: %s', preflight(args.gazebo_port, args.ros_domain))
        return
    result = asyncio.run(run_capture(args))
    if result.get('status') != 'captured' or not result.get('preserved_files_unchanged') or 'evaluation_row' not in result:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

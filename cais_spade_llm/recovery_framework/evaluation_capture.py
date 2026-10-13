from __future__ import annotations

"""Capture one selected failure in an owned simulation without LLM or recovery."""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import shlex
import signal
import subprocess
import socket
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.failure_checkpoints import CHECKPOINTS, SUPPORTED_CHECKPOINTS

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
            matches.append({'pid': int(directory.name), 'comm': comm, 'argv': argv})
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
             ROOT / 'cais_spade_llm/initialization/tools.json']
    paths += list((ROOT / 'cais_spade_llm/monitor/environment_runs/recovery').glob('capture_*/observations.json'))
    paths += list((ROOT / 'cais_spade_llm/initialization').glob('*.json'))
    paths += list((ROOT / 'cais_spade_llm/specification/products/geometry').glob('*recovery-framework.json'))
    for subdirectory in ('history', 'plan', 'state'):
        paths += [path for path in (ROOT / 'cais_spade_llm/monitor' / subdirectory).glob('*') if path.is_file()]
    return {str(path): sha256_file(path) for path in sorted(set(paths))}


def verify_capture(captured: dict, scenario: str) -> dict:
    """Require the selected reconciled checkpoint and preserve its exact facts."""
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
            from cais_spade_llm.recovery_framework.recovery_observations import processing_capture_compatibility

            if processing_capture_compatibility(captured)['status'] != 'supported':
                raise ValueError('Frozen processing checkpoint is inconsistent')
            record = captured['processing_observations']['M2']
            job = record['jobs'].get(record['active_job'], {})
            if job.get('job_state') != 'running' or job.get('process_completed') is not False:
                raise ValueError('M2 is no longer processing at the capture checkpoint')
    if not isinstance(part, str) or part not in captured.get('part_tracker', {}):
        raise ValueError('The checkpoint does not identify a captured affected part')
    return {'checkpoint': failure['checkpoint'], 'run_id': failure['run_id'],
            'part_name': part, 'resource_id': configured['resource_id'],
            'source_paths': ['/failure', '/setup/failure_scenario', '/part_tracker/' + part]}


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
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery
    from cais_spade_llm.recovery_framework import slippage_execution

    calls: list[str] = []

    def blocked(name):
        async def reject(*args, **kwargs):
            calls.append(name)
            raise RuntimeError('Capture-only guard rejected ' + name)
        return reject

    for method in ('ask_llm', 'ask_llm_structured'):
        setattr(LlmAgent, method, blocked('LlmAgent.' + method))
    for method in ('_run_stage', '_generate_outline', '_generate_primitives', '_execute'):
        setattr(PartSlippageRecovery, method, blocked('PartSlippageRecovery.' + method))
    slippage_execution.execute_generated_recovery = blocked('execute_generated_recovery')
    return calls


def configure_bridge(bridge, directory: Path, domain: int, port: int, scenario: str) -> None:
    """Scope cleanup, process ownership, logs, and bundle state to this harness."""
    from cais_spade_llm.ui.process_registry import UIProcessRegistry

    bridge._ui_process_registry = UIProcessRegistry(directory / 'owned_processes.json')
    bridge._sim_readiness_diagnostic_path = directory / 'readiness_probes.jsonl'
    # The normal methods kill unrelated groups and delete shared DDS memory.
    # Its tracked launch stop already sends SIGINT/SIGKILL to the owned group.
    def skip_global_cleanup(*args, **kwargs):
        logger.info('Skipped global cleanup; only the experiment launch group is owned')

    for name in ('_kill_stale_gazebo_helpers', '_force_kill_gazebo_core',
                 '_force_kill_digital_twin_helpers', '_cleanup_shm_and_tmp'):
        setattr(bridge, name, skip_global_cleanup)
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


def install_condition_setup(bridge, directory: Path, scenario: str, condition: str) -> None:
    """Apply only the selected private capture setup before its owned launch."""
    from cais_spade_llm.recovery_framework import part_slippage_example
    from cais_spade_llm.recovery_framework.evaluation_conditions import condition_setup

    original = part_slippage_example.example_setup

    def selected_setup(**kwargs):
        if kwargs.get('scenario') != scenario:
            raise ValueError('Capture setup changed its selected scenario')
        setup = condition_setup(original(**kwargs), scenario, condition, directory)
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
    from cais_spade_llm.recovery_framework.recovery_evaluation import _implementation_files, frozen_files, verify_frozen

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
    import cais_spade_llm.ui_main  # Only import compatibility; never invoke UI startup/cleanup.
    import cais_spade_llm.ui.bridge as bridge_module
    from cais_spade_llm.recovery_framework.part_slippage_example import example_for_bridge
    from cais_spade_llm.recovery_framework.recovery_observations import (
        RECORDS, capture_observations, offline_preparation_missing, save_native_preview_model_from_capture,
    )
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import _prepare_saved_capture
    from cais_spade_llm.recovery_framework.recovery_motion_preview import capture_preview_scene
    from cais_spade_llm.recovery_framework.evaluation_conditions import apply_changed_condition, verify_capture_condition
    from cais_spade_llm.recovery_framework.recovery_evaluation import value_hash, verify_frozen

    # These are process-local constants; production source and shared files stay unchanged.
    bridge_module._USER_VERIFIED_PLAN = directory / 'bundle_store'
    bridge_module._TOOLS_OUT = directory / 'tools.json'
    workspace = prepare_private_workspace(directory)
    bridge_module._MONITOR = workspace / 'cais_spade_llm/monitor'
    bridge_module._LOG_DIR = workspace / 'cais_spade_llm/log'
    bridge_module._RECOVERY_RUNTIME_DATA_DIR = bridge_module._MONITOR / 'debug/recovery_runtime_data'
    os.chdir(workspace)
    calls = install_capture_guards()
    bridge = bridge_module.SystemBridge()
    configure_bridge(bridge, directory, args.ros_domain, args.gazebo_port, args.scenario)
    install_condition_setup(bridge, directory, args.scenario, args.condition)
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
        destination = RECORDS / ('capture_' + uuid4().hex)
        destination.mkdir()
        observation_path = destination / 'observations.json'
        report.update(observation_path=str(observation_path), capture_verification=verify_capture(captured, args.scenario))
        write_record(observation_path, captured)
        write_record(directory / 'report.json', report)
        captured['prepared_recovery_request'] = await bridge._run_on_agent_runtime(
            _prepare_saved_capture(runtime, captured, recovery._observations['prepared_recovery_request'], observation_path))
        captured['missing'] = offline_preparation_missing(captured)
        captured['offline_ready'] = not captured['missing']
        write_record(observation_path, captured)
        report['native_model_archive'] = await save_native_preview_model_from_capture(
            observation_path, cancel_event=cancelled)
        condition_evidence = verify_capture_condition(captured, args.scenario, args.condition)
        if not isinstance(condition_evidence, dict) or not condition_evidence:
            raise ValueError('Condition verification returned no factual evidence')
        condition_record = {'version': 1, 'source': str(observation_path),
            'source_sha256': sha256_file(observation_path), 'scenario': args.scenario,
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
            await bridge.part_slippage_action('stop')
            await asyncio.to_thread(bridge.ros2_stop_all, reason='owned_fresh_capture_complete')
            example._restore_settings()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError, asyncio.TimeoutError, subprocess.SubprocessError) as exc:
            logger.exception('Owned cleanup failed')
            report['cleanup_error'] = str(exc)
        after = {path: sha256_file(Path(path)) for path in before}
        report.update(finished_at=datetime.now(timezone.utc).isoformat(), preserved_after=after,
                      preserved_files_unchanged=before == after, llm_calls=calls,
                      owned_launch_groups_remaining={name: proc.pid for name, proc in bridge._ros2_procs.items()
                                                     if proc.poll() is None})
        if implementation is not None:
            try:
                verify_frozen(implementation)
            except ValueError as exc:
                report.update(status='failed', reason=str(exc), implementation_unchanged=False)
            else:
                report['implementation_unchanged'] = True
        if (condition_record is not None and report['status'] == 'captured'
                and report['preserved_files_unchanged'] and not report['owned_launch_groups_remaining']
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
    args = parser.parse_args()
    if not 1 <= args.ros_domain <= 232 or not 1024 <= args.gazebo_port <= 65535:
        parser.error('Use an isolated ROS domain and an unprivileged Gazebo port')
    if any(not math.isfinite(value) or value <= 0 for value in (args.timeout, args.settle_timeout)):
        parser.error('Observation deadlines must be finite and positive')
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

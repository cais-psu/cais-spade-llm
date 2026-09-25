"""Prepare the saved delivery experiment before the existing SystemBridge start."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

from cais_spade_llm.recovery_framework import read_json
from cais_spade_llm.recovery_framework.delivery import (
    check_stopped, is_delivery_order, prepare_start, record_preparation_failure, verify_configuration,
)
from cais_spade_llm.ui.recovery_setup import reference_path, startup_block_reason, validate_setup

logger = logging.getLogger(__name__)
_TRANSIENT_READINESS_ERRORS = (
    'Simulation startup is not done yet. Waiting for core services:',
    'Simulation readiness query failed or timed out',
)
_STARTUP_LOG_SCOPE: ContextVar[object | None] = ContextVar('delivery_startup_log_scope', default=None)


class _DeferredReadinessErrors(logging.Filter):
    """Hold only this Start request's recoverable discovery errors until its retry decision."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def filter(self, record: logging.LogRecord) -> bool:
        """Leave unrelated errors and concurrent Start requests untouched."""
        error = record.exc_info[1] if record.exc_info else None
        if (_STARTUP_LOG_SCOPE.get() is self
                and record.getMessage() == 'Failed to start system'
                and isinstance(error, RuntimeError)
                and str(error).startswith(_TRANSIENT_READINESS_ERRORS)):
            self.records.append(record)
            return False
        return True


@contextmanager
def _defer_readiness_errors() -> Iterator[list[logging.LogRecord]]:
    bridge_logger = logging.getLogger('ui.bridge')
    deferred = _DeferredReadinessErrors()
    token = _STARTUP_LOG_SCOPE.set(deferred)
    bridge_logger.addFilter(deferred)
    try:
        yield deferred.records
    finally:
        bridge_logger.removeFilter(deferred)
        _STARTUP_LOG_SCOPE.reset(token)
        for record in deferred.records:
            bridge_logger.handle(record)


def supports_delivery(setup: dict) -> bool:
    """Read whether the selected setup uses the supported delivery order."""
    try:
        order = read_json(reference_path(setup['selected_product_order_file']))
        return setup['execution_mode'] == 'simulation' and is_delivery_order(order)
    except (OSError, KeyError, TypeError, ValueError):
        return False


def configuration_fingerprints(setup: dict) -> dict:
    """Fingerprint referenced source files without recording manifest credentials."""
    inputs = validate_setup(setup)
    references = [setup['selected_product'], setup['selected_product_order_file'],
                  setup['scene_file'], inputs['product_geometry_file'],
                  setup['recovery_experiment_settings_file']]
    if setup['selected_safety_file'] != '__NONE__':
        references.append(setup['selected_safety_file'])
    references.extend(['ros2/cais_lab_robotics/worlds/table_recovery_framework.world',
                       'ros2/cais_lab_robotics/CMakeLists.txt',
                       'ros2/cais_lab_robotics/package.xml',
                       'ros2/cais_lab_robotics/models/bantam_tools_desktop_cnc/model.sdf',
                       'cais_spade_llm/recovery_framework/kmr_gazebo.py',
                       'cais_spade_llm/recovery_framework/kmr_motion.py',
                       'cais_spade_llm/recovery_framework/kmr_agent.py',
                       'cais_spade_llm/recovery_framework/kmr_tasks.py',
                       'cais_spade_llm/recovery_framework/gazebo_worker.py',
                       'cais_spade_llm/recovery_framework/workflow_execution.py',
                       'cais_spade_llm/recovery_framework/planning_permit.py',
                       'cais_spade_llm/recovery_framework/geometry.py',
                       'cais_spade_llm/resources/robot/gazebo_pick_place_controller.py',
                       'cais_spade_llm/resources/robot/cartesian_waypoints.py',
                       'cais_spade_llm/recovery_framework/environment_runtime.py',
                       'cais_spade_llm/agents/shared_information/environment_capabilities.py',
                       'cais_spade_llm/product/environment.py',
                       'cais_spade_llm/resources/robot/robot_task_runtime.py',
                       'cais_spade_llm/resources/robot/execution_timing.py',
                       'cais_spade_llm/recovery_framework/simulation.py',
                       'cais_spade_llm/resources/robot/simulation_timing.py',
                       'cais_spade_llm/initialization/recovery_framework_setup.json'])
    references.extend('ros2/cais_lab_robotics/'+path
                      for path in inputs['scene']['KMR']['task_execution']['scene_assets'])
    return {str(path): hashlib.sha256(reference_path(path).read_bytes()).hexdigest() for path in references}


def configuration_snapshots(setup: dict, fingerprints: dict) -> dict:
    """Record referenced text and binary assets while excluding manifest credentials."""
    snapshots = {}
    for path in fingerprints:
        if path == setup['selected_product']:
            continue
        data = reference_path(path).read_bytes()
        try:
            snapshots[path] = data.decode('utf-8')
        except UnicodeDecodeError:
            snapshots[path] = {'encoding': 'base64', 'data': base64.b64encode(data).decode('ascii')}
    return snapshots


async def prepare_delivery_start(
    bridge,
    setup: dict,
    *,
    worker=None,
    requested_at_unix: float | None = None,
) -> dict:
    """Launch if absent, verify observations, and prepare an immutable factory input."""
    from cais_spade_llm.recovery_framework.gazebo_worker import GazeboWorker

    from cais_spade_llm.recovery_framework.delivery import (
        retain_prepared_worker,
        take_unclaimed_prepared_workers,
    )

    for stale_worker in take_unclaimed_prepared_workers():
        await stale_worker.cancel()
    prepare_start(None)
    requested_at = time.time() if requested_at_unix is None else float(requested_at_unix)
    preparation_started = time.monotonic()
    setup = deepcopy(setup)
    validation_started = time.monotonic()
    reason = startup_block_reason(setup)
    if reason:
        raise ValueError(reason)
    inputs = validate_setup(setup)
    if not supports_delivery(setup):
        raise ValueError('Unsupported Gazebo delivery setup')
    fingerprints = configuration_fingerprints(setup)
    snapshot = {'setup': setup, 'source_fingerprints': fingerprints,
                'source_snapshots': configuration_snapshots(setup, fingerprints),
                'inputs': {'scene': inputs['scene'], 'product_order': inputs['product_order'], 'geometry': inputs['geometry']},
                'startup_timing': {
                    'start_requested_at_unix': requested_at,
                    'configuration_validation_wall_time_sec': time.monotonic()-validation_started,
                }}
    owned_worker = None
    try:
        def ensure_scene() -> bool:
            if bridge.simulation_environment_running():
                return False
            error = bridge.ros2_start('gazebo_dual')
            if error:
                raise RuntimeError(error)
            # The legacy prewarm creates ur5e/xarm6 perception controllers. This
            # order uses KMR's explicit probe and the same system service gate.
            bridge._shutdown_gazebo_prewarm_controllers()
            return True

        scene_started = time.monotonic()
        snapshot['startup_timing']['fresh_scene_started'] = ensure_scene()
        snapshot['startup_timing']['scene_launch_request_wall_time_sec'] = (
            time.monotonic() - scene_started
        )
        check_stopped()
        probe_worker = worker or GazeboWorker()
        owned_worker = probe_worker if worker is None else None
        probe_started = time.monotonic()
        snapshot['probe'] = await probe_worker.run({'mode': 'probe', 'inputs': snapshot['inputs']})
        snapshot['startup_timing']['probe_wall_time_sec'] = time.monotonic() - probe_started
        from cais_spade_llm.recovery_framework.simulation import simulation_settings
        import json

        applied = snapshot['probe'].get('performance_settings')
        if applied is not None and json.loads(applied) != simulation_settings(setup):
            raise ValueError('Simulation settings changed; explicitly start a fresh Gazebo scene')
        check_stopped()
        if fingerprints != configuration_fingerprints(setup):
            raise ValueError('Configuration changed during startup; reload setup and start again')
    except (OSError, ValueError, RuntimeError, TimeoutError, asyncio.CancelledError) as exc:
        if owned_worker is not None:
            await owned_worker.cancel()
        record_preparation_failure(snapshot, exc)
        raise
    snapshot['startup_timing']['preparation_wall_time_sec'] = time.monotonic() - preparation_started
    if owned_worker is not None:
        snapshot['prepared_resource_token'] = retain_prepared_worker(owned_worker, snapshot)
    prepare_start(snapshot)
    return snapshot


async def start_delivery_agents(
    bridge, setup: dict, prepared: dict, *, on_progress: Callable[[str], None] | None = None,
) -> None:
    """Retry transient discovery before agents start, retaining every readiness gate.

    Args:
        bridge: Existing SystemBridge owning this explicit Start request.
        setup: Selected delivery settings.
        prepared: Verified scene and immutable configuration from preparation.
        on_progress: Optional UI notice when a transient discovery check is retried.
    """
    from cais_spade_llm.recovery_framework.delivery import take_unclaimed_prepared_workers

    async def cleanup_unclaimed() -> None:
        prepare_start(None)
        for unclaimed in take_unclaimed_prepared_workers():
            await unclaimed.cancel()

    setup, prepared = deepcopy(setup), deepcopy(prepared)
    try:
        for attempt in range(3):
            check_stopped()
            verify_configuration(prepared)
            with _defer_readiness_errors() as pending_errors:
                await bridge.start_system()
                if bridge.system_running:
                    ready_at = time.time()
                    for product in getattr(bridge, 'product_agents', ()):
                        runtime = getattr(product, 'delivery_runtime', None)
                        if runtime is None:
                            continue
                        timing = runtime.prepared.setdefault('startup_timing', {})
                        timing['agents_ready_at_unix'] = ready_at
                        requested = timing.get('start_requested_at_unix')
                        if requested is not None:
                            timing['start_to_readiness_wall_time_sec'] = ready_at-requested
                        runtime.save()
                    await cleanup_unclaimed()
                    return
                if attempt == 2 or not str(bridge.last_error).startswith(_TRANSIENT_READINESS_ERRORS):
                    break
                check_stopped()
                if not bridge.simulation_environment_running():
                    break
                # The bridge logs before its caller can decide to retry. Only replace
                # that record once this request has committed to another readiness check.
                pending_errors.clear()
            message = (
                f'Waiting for simulation service discovery; retrying Start System '
                f'({attempt + 2}/3). {bridge.last_error}'
            )
            logger.info(message)
            if on_progress is not None:
                on_progress(message)
            await asyncio.sleep(.1)
    except (ValueError, asyncio.CancelledError):
        await cleanup_unclaimed()
        raise
    await cleanup_unclaimed()

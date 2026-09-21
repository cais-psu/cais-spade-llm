"""Prepare the saved delivery experiment before the existing SystemBridge start."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from copy import deepcopy

from cais_spade_llm.recovery_framework import read_json
from cais_spade_llm.recovery_framework.delivery import (
    check_stopped, is_delivery_order, prepare_start, record_preparation_failure,
)
from cais_spade_llm.ui.recovery_setup import reference_path, startup_block_reason, validate_setup


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
                       'ros2/cais_lab_robotics/models/bantam_tools_desktop_cnc/model.sdf',
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


async def prepare_delivery_start(bridge, setup: dict, *, worker=None) -> dict:
    """Launch if absent, verify observations, and prepare an immutable factory input."""
    from cais_spade_llm.recovery_framework.kmr_agent import GazeboWorker

    prepare_start(None)
    setup = deepcopy(setup)
    reason = await asyncio.to_thread(startup_block_reason, setup)
    if reason:
        raise ValueError(reason)
    inputs = await asyncio.to_thread(validate_setup, setup)
    if not await asyncio.to_thread(supports_delivery, setup):
        raise ValueError('Unsupported Gazebo delivery setup')
    fingerprints = await asyncio.to_thread(configuration_fingerprints, setup)
    snapshot = {'setup': setup, 'source_fingerprints': fingerprints,
                'source_snapshots': await asyncio.to_thread(configuration_snapshots, setup, fingerprints),
                'inputs': {'scene': inputs['scene'], 'product_order': inputs['product_order'], 'geometry': inputs['geometry']}}
    try:
        if not await asyncio.to_thread(bridge.simulation_environment_running):
            error = await asyncio.to_thread(bridge.ros2_start, 'gazebo_dual')
            if error:
                raise RuntimeError(error)
            # The legacy prewarm creates ur5e/xarm6 perception controllers. This
            # order uses KMR's explicit probe and the same system service gate.
            await asyncio.to_thread(bridge._shutdown_gazebo_prewarm_controllers)
        check_stopped()
        snapshot['probe'] = await (worker or GazeboWorker()).run({'mode': 'probe', 'inputs': snapshot['inputs']})
        deadline = time.monotonic()+240
        while True:
            check_stopped()
            ready, reason = await asyncio.to_thread(bridge.simulation_start_ready, True)
            if ready:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f'System readiness did not complete: {reason}')
            await asyncio.sleep(1)
        check_stopped()
        if fingerprints != await asyncio.to_thread(configuration_fingerprints, setup):
            raise ValueError('Configuration changed during startup; reload setup and start again')
    except (OSError, ValueError, RuntimeError, TimeoutError, asyncio.CancelledError) as exc:
        await asyncio.to_thread(record_preparation_failure, snapshot, exc)
        raise
    await asyncio.to_thread(prepare_start, snapshot)
    return snapshot

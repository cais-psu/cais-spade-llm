from __future__ import annotations
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
sys.path.insert(0, '/home/jongh/projects/cais-spade-llm')
import cais_spade_llm.ui_main
from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.recovery_setup import default_setup
from cais_spade_llm.recovery_framework.startup import prepare_delivery_start
from cais_spade_llm.recovery_framework.delivery import ORDER_PATH, ROOT, reset_stop, request_stop, prepare_start

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('acceptance')

async def main():
    setup = default_setup()
    setup['selected_product_order_file'] = str(ORDER_PATH.relative_to(ROOT))
    bridge = SystemBridge()
    bridge.ROS2_LAUNCH_CMDS = dict(bridge.ROS2_LAUNCH_CMDS)
    bridge.ROS2_LAUNCH_CMDS['gazebo_dual'] += ' launch_gazebo_gui:=false launch_rviz:=false'
    bridge.selected_product = setup['selected_product']
    bridge.selected_product_order_file = setup['selected_product_order_file']
    bridge.selected_safety_file = setup['selected_safety_file']
    bridge.set_active_bundle(None)
    bridge.set_runtime_recovery_mode(setup['runtime_recovery_mode'])
    bridge.set_runtime_recovery_validation_policy(setup['runtime_recovery_validation_policy'])
    bridge.set_runtime_recovery_archive_selection('', '')
    runtime = None
    reset_stop()
    try:
        prepared = await prepare_delivery_start(bridge, setup)
        log.info('START PREPARATION COMPLETE: %s', prepared['probe']['launch_id'])
        await bridge.start_system()
        prepare_start(None)
        log.info('START RESULT: running=%s error=%s', bridge.system_running, bridge.last_error)
        if not bridge.system_running:
            raise RuntimeError(bridge.last_error)
        await bridge.start_system()
        log.info('DUPLICATE START returned without a second launch')
        deadline = time.monotonic()+900
        while time.monotonic() < deadline:
            runtime = next((getattr(a, 'delivery_runtime', None) for a in bridge.product_agents if getattr(a, 'delivery_runtime', None)), None)
            if runtime:
                log.info('DELIVERY STATUS %s report=%s', runtime.outcome, runtime.path)
                if runtime.outcome['status'] not in {'prepared','planned','running'}:
                    Path('/tmp/cais_acceptance_outcome.json').write_text(json.dumps({'outcome':runtime.outcome, 'report':str(runtime.path/'run.json')}, indent=2))
                    break
            await asyncio.sleep(10)
        else:
            raise TimeoutError('Gazebo delivery did not finish')
    finally:
        request_stop()
        await bridge.stop_system()
        log.info('STOP COMPLETE, Gazebo retained=%s', bridge.simulation_environment_running())
        prepare_start(None)

    if runtime is not None and runtime.outcome['status'] == 'completed':
        reset_stop()
        try:
            await prepare_delivery_start(bridge, setup)
        except (ValueError, RuntimeError) as exc:
            if 'explicitly reset Gazebo' not in str(exc):
                raise
            log.info('REPEAT BLOCKED: %s', exc)
            (runtime.path/'control_verification.json').write_text(json.dumps({
                'duplicate_start': 'no second launch', 'stop_system': 'agents stopped; Gazebo retained',
                'repeat_without_reset': str(exc), 'interface': 'SystemBridge Start System / Stop System'}, indent=2))
        else:
            raise AssertionError('A completed transfer was allowed to repeat without reset')
        finally:
            prepare_start(None)
            request_stop()

asyncio.run(main())

from __future__ import annotations
import asyncio
import json
import logging
import sys
import time
import subprocess
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
        await asyncio.to_thread(subprocess.run, ['bash', '-c', 'source /opt/ros/humble/setup.bash\nsource /home/jongh/ros2_ws/install/setup.bash\nexec /usr/bin/python3 /tmp/cais_machine_paths_final.py'], check=True, timeout=100)
        await bridge.start_system()
        prepare_start(None)
        log.info('START RESULT: running=%s error=%s', bridge.system_running, bridge.last_error)
        if not bridge.system_running:
            raise RuntimeError(bridge.last_error)
        await bridge.start_system()
        log.info('DUPLICATE START returned without a second launch')
        runtime = next(a.delivery_runtime for a in bridge.product_agents if getattr(a, 'delivery_runtime', None))
        command = ['bash', '-c', 'source /opt/ros/humble/setup.bash\nsource /home/jongh/ros2_ws/install/setup.bash\nexec /usr/bin/python3 /tmp/cais_action_observation.py "$1" "$2"', 'cais-observe', '2', '']
        command[-1:] = []
        # Pass no goal filter for the first observed executing trajectory.
        command[2] = command[2].replace(' "$2"', '')
        active = await asyncio.to_thread(subprocess.run, command, capture_output=True, text=True, timeout=65, check=True)
        executing = json.loads(active.stdout)
        log.info('ACTIVE ARM GOAL: %s', executing)
        request_stop()
        await bridge.stop_system()
        command[2] += ' "$2"'
        command[-1] = '5,6'
        command.append(executing['goal_id'])
        stopped = await asyncio.to_thread(subprocess.run, command, capture_output=True, text=True, timeout=65, check=True)
        observed = json.loads(stopped.stdout)
        assert runtime.context.revision == 0
        assert runtime.context.snapshot()['Storage']['inventory.KET4_Square_4mm'] is True
        assert runtime.context.snapshot()['KMR']['held_part'] is None
        assert runtime.outcome['status'] == 'stopped'
        assert not bridge.system_running and bridge.simulation_environment_running()
        result = {'executing': executing, 'after_stop': observed, 'outcome': runtime.outcome['status'],
                  'acknowledged_tasks': 0, 'Storage_inventory': True, 'KMR_held_part': None,
                  'Gazebo_retained': True}
        (runtime.path/'control_verification.json').write_text(json.dumps(result, indent=2))
        Path('/tmp/cais_stop_acceptance_outcome.json').write_text(json.dumps({'report':str(runtime.path/'run.json'), **result}, indent=2))
        log.info('ACTIVE STOP VERIFIED: %s', result)
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

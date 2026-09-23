"""KMR functions composed from reusable, observed execution primitives."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict

from cais_spade_llm.resources.robot.robot_task_model import (
    RobotTaskArgument, RobotTaskDefinition, RobotTaskEffect, RobotTaskGuard,
    RobotTaskProgram, RobotTaskStep, _arg, _resolve_value, _state, _step_output,
)


def _step(identifier, primitive, **params):
    return RobotTaskStep(id=identifier, op=primitive, params=params, store_as=identifier)


def _path(name, *path):
    return _step_output(name, *map(str, path))


def delivery_bindings(part_name: str) -> dict[str, dict]:
    """Retain every binding in the existing three-task acknowledgement protocol."""
    return {
        'pick_part': {'resource_id': 'KMR', 'part_name': part_name,
                      'origin_resource_location': 'Storage', 'handoff_acknowledged': True},
        'move_to_resource': {'resource_id': 'KMR', 'target_resource': 'M1',
                             'source_resource': 'Storage', 'arrival_acknowledged': True,
                             'arm_parked': True},
        'place_release': {'resource_id': 'KMR', 'part_name': part_name,
                          'destination_location': 'M1', 'handoff_acknowledged': True,
                          'robot_clear': True},
    }


PRIMITIVE_CONTRACTS = {
    'open_gripper': ({'held_part': 'checked by the bound task'}, {'gripper_state': 'open'}),
    'close_gripper': ({'endpoint': 'observed pickup pose'}, {'gripper_state': 'closed'}),
    'compute_pick_targets': ({'resource_location': 'Storage', 'part_pose': 'fresh'}, {'pick_targets': 'vertical and downward-facing'}),
    'move_to_pose': ({'start_joints': 'fresh'}, {'endpoint': 'collision checked and observed'}),
    'rotate_arm_base': ({'arm_clearance': 'collision checked'}, {'joint_a1': 'observed', 'orientation': 'downward'}),
    'move_to_configuration': ({'start_joints': 'fresh'}, {'joints': 'collision checked and observed'}),
    'execute_plan': ({'start_joints': 'fresh and matching'}, {'endpoint': 'observed'}),
    'observe_grasp': ({'part_pose': 'unchanged since planning'}, {'grasp_transform': 'observed'}),
    'attach_part': ({'gripper_state': 'closed'}, {'attached': True}),
    'detach_part': ({'gripper_state': 'open'}, {'attached': False}),
    'part_collision': ({'part_pose': 'fresh'}, {'planning_scene': 'acknowledged'}),
    'custody': ({'grasp_transform': 'acknowledged'}, {'held_part': 'observed'}),
    'confirm_carrying': ({'arm_joints': 'fresh'}, {'arm_parked': True}),
    'observe_carrying': ({'attached': True}, {'part_location': 'KMR', 'arm_parked': True}),
    'observe_custody': ({'attached': True}, {'grasp_transform': 'validated'}),
    'validate_transport': ({'arm_parked': True}, {'transport_sweep': 'collision checked'}),
    'dock': ({'arm_parked': True, 'attached': True}, {'resource_location': 'observed'}),
    'observe_dock': ({'attached': True}, {'resource_location': 'M1'}),
    'compute_place_targets': ({'resource_location': 'configured destination', 'workholding': 'empty'}, {'place_targets': 'vertical and downward-facing'}),
    'move_cartesian': ({'start_joints': 'fresh'}, {'endpoint': 'observed'}),
    'observe_release': ({'attached': False}, {'part_location': 'M1', 'robot_clear': True}),
}

_pick = (
    _step('open', 'open_gripper'),
    _step('pick', 'compute_pick_targets', initial=_state('initial')),
    _step('approach', 'move_to_pose', target=_path('pick', 'approach'), seed=_path('pick', 'seed')),
    _step('descend', 'move_cartesian', target=_path('pick', 'target')),
    _step('close', 'close_gripper'),
    _step('grasp', 'observe_grasp', initial=_state('initial')),
    _step('attach', 'attach_part'),
    _step('attached_collision', 'part_collision', attached=True),
    _step('lift', 'move_cartesian', target=_path('pick', 'lift')),
    _step('lift_custody', 'custody', transform=_path('grasp')),
    _step('observations', 'observe_carrying', transform=_path('grasp')),
)
_move = (
    _step('grasp', 'observe_custody', previous=_state('previous')),
    _step('posture', 'confirm_carrying'),
    _step('sweep', 'validate_transport'),
    _step('transport', 'dock', target_resource=_arg('target_resource'), transform=_path('grasp')),
    _step('observations', 'observe_dock', transform=_path('grasp')),
)
_place = (
    _step('grasp', 'observe_custody', previous=_state('previous')),
    _step('place', 'compute_place_targets', transform=_path('grasp')),
    _step('clearance', 'move_to_configuration', joints=_path('place', 'before_turn'), hold_arm_base=True),
    _step('turn', 'rotate_arm_base', joint_a1=_path('place', 'joint_a1')),
    _step('descend', 'move_cartesian', target=_path('place', 'target')),
    _step('held', 'custody', transform=_path('grasp')),
    _step('open', 'open_gripper'),
    _step('detach', 'detach_part'),
    _step('released_collision', 'part_collision', attached=False),
    _step('withdraw', 'move_cartesian', target=_path('place', 'retreat')),
    _step('park', 'move_to_configuration', joints=_path('place', 'parked')),
    _step('observations', 'observe_release', destination=_path('place', 'destination')),
)


def _definition(name, before, after, steps):
    bindings = delivery_bindings('')[name]
    return RobotTaskDefinition(
        name=name, description=f'Observed KMR {name} through reusable primitives.',
        arguments=tuple(RobotTaskArgument(arg, 'boolean' if isinstance(value, bool) else 'string', required=True)
                        for arg, value in bindings.items()),
        source='recovery_framework/kmr_tasks.py',
        program=RobotTaskProgram(
            entry_state=before, success_state=after,
            entry_guards=(RobotTaskGuard('bound_task', description='Matching task, scene, and fresh resource observations'),),
            steps=steps, effects=(RobotTaskEffect('resource_state', 'set', after),),
            notes=('Nominal effects are committed only by the existing task acknowledgement.',),
        ),
    )


KMR_TASKS = {
    'pick_part': _definition('pick_part', 'idle', 'carrying', _pick),
    'move_to_resource': _definition('move_to_resource', 'carrying', 'carrying', _move),
    'place_release': _definition('place_release', 'carrying', 'idle', _place),
}


def capability_decompositions(*, function_name='', resource_jid='', **_kwargs) -> dict:
    """Derive recovery metadata from the same programs that dispatch primitives."""
    decompositions = {name: {
        **definition.capability_decomposition(resource_jid=resource_jid),
        'program': asdict(definition.program),
        'primitive_contracts': {
            step.op: {'preconditions': deepcopy(PRIMITIVE_CONTRACTS[step.op][0]),
                      'effects': deepcopy(PRIMITIVE_CONTRACTS[step.op][1])}
            for step in definition.program.steps
        },
    } for name, definition in KMR_TASKS.items() if not function_name or name == function_name}
    return decompositions.get(function_name, {}) if function_name else decompositions


def execute_composition(name, *, arguments, state, primitives, records, operations,
                        now, check, serialize, planning_time=lambda: 0.0) -> dict:
    """Run declared steps in order, preserving partial evidence on failure."""
    definition = KMR_TASKS[name]
    outputs = {}
    for step in definition.program.steps:
        parameters = _resolve_value(step.params, args=arguments, runtime_state=state, step_outputs=outputs)
        wall, simulated, planned = time.monotonic(), now(), planning_time()
        first_operation = len(operations)
        record = {
            'resource_id': 'KMR', 'function_name': name, 'step_id': step.id,
            'task_id': state.get('task_id'),
            'primitive': step.op, 'parameters': serialize(parameters),
            'preconditions': deepcopy(PRIMITIVE_CONTRACTS[step.op][0]),
            'effects': deepcopy(PRIMITIVE_CONTRACTS[step.op][1]), 'status': 'running',
        }
        records.append(record)
        try:
            check()
            result = primitives[step.op](**parameters)
            if isinstance(result, dict) and result.get('success') is False:
                raise RuntimeError(result.get('message', f'{step.op} failed'))
            outputs[step.store_as or step.id] = result
            record.update(status='completed', result=serialize(result))
        except (RuntimeError, ValueError, TypeError, KeyError, TimeoutError, InterruptedError) as exc:
            record.update(status='failed', error=str(exc))
            raise
        finally:
            elapsed = time.monotonic() - wall
            simulated_elapsed = now() - simulated
            record['observations'] = deepcopy(operations[first_operation:])
            record['timing'] = {
                'wall_time_sec': elapsed, 'simulation_time_sec': simulated_elapsed,
                'planning_wall_time_sec': planning_time() - planned,
                'trajectory_duration_sec': sum(
                    row['trajectory']['points'][-1]['time_from_start']
                    for row in record['observations'] if row.get('trajectory', {}).get('points')),
                'real_time_factor': simulated_elapsed / elapsed if elapsed > 0 and simulated_elapsed >= 0 else None,
                'clock_reset': simulated_elapsed < 0,
            }
    return outputs['observations']

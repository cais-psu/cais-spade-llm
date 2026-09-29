"""KMR functions composed from reusable, observed execution primitives."""

from __future__ import annotations

import time
import json
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
    """Bind the five KMR functions used for the Storage-to-M1 delivery."""
    return {
        'pick_approach': {'resource_id': 'KMR', 'part_name': part_name,
                          'origin_resource_location': 'Storage'},
        'pick_part': {'resource_id': 'KMR', 'part_name': part_name,
                      'origin_resource_location': 'Storage', 'handoff_acknowledged': True},
        'move_to_resource': {'resource_id': 'KMR', 'target_resource': 'M1',
                             'source_resource': 'Storage', 'arrival_acknowledged': True,
                             'arm_parked': True},
        'place_approach': {'resource_id': 'KMR', 'part_name': part_name,
                           'destination_location': 'M1'},
        'place_release': {'resource_id': 'KMR', 'part_name': part_name,
                          'destination_location': 'M1', 'handoff_acknowledged': True,
                          'robot_clear': True},
    }


PRIMITIVE_CONTRACTS = {
    'detect_parts': ({}, {'parts': 'observed Gazebo poses'}),
    'compute_pick_targets': ({'part_pose': 'fresh'}, {'pick_targets': 'observed poses'}),
    'compute_place_targets': ({'held_part': 'observed'}, {'place_targets': 'observed poses'}),
    'move_to_named_pose': ({'arm_joints': 'fresh'}, {'tcp_pose': 'observed'}),
    'move_cartesian': ({'arm_joints': 'fresh'}, {'tcp_pose': 'observed'}),
    'move_relative': ({'arm_joints': 'fresh'}, {'tcp_pose': 'observed'}),
    'move_joints': ({'arm_joints': 'fresh'}, {'joints': 'observed'}),
    'rotate_joint': ({'arm_joints': 'fresh'}, {'joints': 'observed'}),
    'open_gripper': ({}, {'gripper_state': 'open'}),
    'close_gripper': ({}, {'gripper_state': 'closed'}),
    'grasp_part': ({'held_part': None}, {'held_part': 'observed'}),
    'release_part': ({'held_part': 'observed'}, {'held_part': None}),
    'attach_part': ({'gripper_state': 'closed'}, {'attached': True}),
    'detach_part': ({'gripper_state': 'open'}, {'attached': False}),
    'move_base': ({'base_pose': 'fresh'}, {'base_pose': 'observed'}),
}

_pick_approach = (
    _step('park', 'move_to_named_pose', pose_name='transport'),
    _step('transport', 'move_base', target_pose=_state('pickup_pose'),
          waypoints=_state('pickup_waypoints')),
    _step('open', 'open_gripper'),
    _step('pick', 'compute_pick_targets', initial=_state('initial')),
    _step('approach', 'move_cartesian', target=_path('pick', 'approach'),
          seed=_path('pick', 'seed')),
)
_pick = (
    _step('descend', 'move_cartesian', target=_state('previous', 'pick_targets', 'target')),
    _step('grasp', 'grasp_part', initial=_state('initial')),
    _step('lift', 'move_cartesian', target=_state('previous', 'pick_targets', 'lift'),
          waypoints=_state('previous', 'pick_targets', 'lift_waypoints')),
)
_move = (
    _step('transport', 'move_base', target_pose=_state('base_target_pose'),
          waypoints=_state('base_waypoints'), transform=_state('previous', 'grasp_transform')),
)
_place_approach = (
    _step('place', 'compute_place_targets', transform=_state('previous', 'grasp_transform')),
    _step('clearance', 'move_cartesian', target=_path('place', 'approach'),
          waypoints=_path('place', 'transfer_waypoints')),
)
_place = (
    _step('descend', 'move_cartesian', target=_state('previous', 'place_targets', 'target')),
    _step('release', 'release_part', transform=_state('previous', 'grasp_transform')),
    _step('withdraw', 'move_cartesian', target=_state('previous', 'place_targets', 'retreat')),
    _step('park', 'move_cartesian', target=_state('previous', 'place_targets', 'transport'),
          waypoints=_state('previous', 'place_targets', 'withdraw_waypoints')),
)
_empty_move = (
    _step('transport', 'move_base', target_pose=_state('base_target_pose'),
          waypoints=_state('base_waypoints')),
)
_empty_return = (
    _step('transport', 'move_base', target_pose=_state('base_target_pose'),
          waypoints=_state('base_waypoints')),
    _step('home', 'move_to_named_pose', pose_name='home'),
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
    'pick_approach': _definition('pick_approach', 'idle', 'at_pick', _pick_approach),
    'pick_part': _definition('pick_part', 'at_pick', 'carrying', _pick),
    'move_to_resource': _definition('move_to_resource', 'carrying', 'carrying', _move),
    'place_approach': _definition('place_approach', 'carrying', 'positioned', _place_approach),
    'place_release': _definition('place_release', 'positioned', 'idle', _place),
}

KMR_MOVE_VARIANTS = {
    'empty': _definition('move_to_resource', 'idle', 'idle', _empty_move),
    'empty_return': _definition('move_to_resource', 'idle', 'idle', _empty_return),
}


def _location_definition(resource_state: str) -> RobotTaskDefinition:
    """Declare coordinate travel with the current KMR held-part state preserved."""
    steps = (_step('transport', 'move_base', target_pose=_state('base_target_pose'),
                   waypoints=_state('base_waypoints')),) if resource_state == 'idle' else (
        _step('transport', 'move_base', target_pose=_state('base_target_pose'),
              waypoints=_state('base_waypoints'),
              transform=_state('previous', 'grasp_transform')),
    )
    return RobotTaskDefinition(
        name='move_to_location',
        description='Move the KMR base to a validated world-frame coordinate.',
        arguments=tuple(RobotTaskArgument(name, 'number', required=True)
                        for name in ('x', 'y', 'yaw')),
        source='recovery_framework/kmr_tasks.py',
        program=RobotTaskProgram(
            entry_state=resource_state, success_state=resource_state,
            entry_guards=(RobotTaskGuard(
                'bound_task', description='Fresh KMR observations and saved scene revision'),),
            steps=steps, effects=(),
            notes=('Coordinate movement updates measured base pose and only a matched named location.',),
        ),
    )


KMR_LOCATION_TASK = _location_definition('idle')
KMR_LOCATION_VARIANTS = {'carrying': _location_definition('carrying')}


def capability_decompositions(*, function_name='', resource_jid='', scene=None, **_kwargs) -> dict:
    """Derive recovery metadata from the same programs that dispatch primitives."""
    from cais_spade_llm.recovery_framework import SCENE_PATH
    from cais_spade_llm.resources.gazebo_programs import saved_robot_definition

    scene = scene if scene is not None else json.loads(SCENE_PATH.read_text())
    definitions = (
        {name: saved_robot_definition(scene, 'KMR', name)
         for name in (*KMR_TASKS, 'move_to_location')}
        if 'resource_programs' in scene else {**KMR_TASKS, 'move_to_location': KMR_LOCATION_TASK}
    )
    decompositions = {name: {
        **definition.capability_decomposition(resource_jid=resource_jid),
        'program': asdict(definition.program),
        'primitive_contracts': {
            step.op: {'preconditions': deepcopy(PRIMITIVE_CONTRACTS[step.op][0]),
                      'effects': deepcopy(PRIMITIVE_CONTRACTS[step.op][1])}
            for step in definition.program.steps
        },
    } for name, definition in definitions.items() if not function_name or name == function_name}
    return decompositions.get(function_name, {}) if function_name else decompositions


def execute_composition(name, *, arguments, state, primitives, records, operations,
                        now, check, serialize, planning_time=lambda: 0.0,
                        definition=None) -> dict:
    """Run declared steps in order, preserving partial evidence on failure."""
    definition = definition or KMR_TASKS[name]
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
    return outputs

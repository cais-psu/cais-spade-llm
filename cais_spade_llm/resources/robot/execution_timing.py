"""Per-call simulation evidence without changing authored assembly programs."""

from __future__ import annotations

import time
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps

from .robot_task_model import _resolve_value

_steps = ContextVar('simulation_primitive_timings', default=None)


def _sample(agent) -> dict:
    controller = getattr(agent, '_controller', None)
    node = getattr(controller, '_node', None)
    return {
        'wall_time_sec': time.monotonic(),
        'simulation_time_sec': node.get_clock().now().nanoseconds / 1e9 if node is not None else None,
        'planning_wall_time_sec': getattr(controller, '_planning_wall_time_sec', 0.),
        'trajectory_duration_sec': getattr(controller, '_trajectory_duration_sec', 0.),
    }


def _elapsed(before: dict, after: dict) -> dict:
    result = {key: after[key] - value if value is not None and after[key] is not None else None
              for key, value in before.items()}
    simulated, wall = result['simulation_time_sec'], result['wall_time_sec']
    result['real_time_factor'] = simulated / wall if simulated is not None and simulated >= 0 and wall > 0 else None
    result['clock_reset'] = simulated is not None and simulated < 0
    return result


def record_simulation_step(function):
    """Retain the actual step result and elapsed times, including partial failures."""
    @wraps(function)
    async def timed(**kwargs):
        records = _steps.get()
        if records is None:
            return await function(**kwargs)
        agent, step, task = kwargs['agent'], kwargs['step'], kwargs['task']
        before = _sample(agent)
        started_at = time.time()
        controller = getattr(agent, '_controller', None)
        if controller is not None:
            controller._last_command_evidence = None
        record = {'resource_id': getattr(agent, 'agent_name', ''), 'function_name': task.name,
                  'step_id': step.id, 'primitive': step.op, 'parameters': _resolve_value(
                      step.params, args=kwargs['args'], runtime_state=kwargs['runtime_state'],
                      step_outputs=kwargs['step_outputs']),
                  'status': 'interrupted', 'started_at_unix': started_at}
        records.append(record)
        try:
            result = await function(**kwargs)
            record.update(status='skipped' if result.get('skipped') else
                          'completed' if result.get('success') else 'failed', result=deepcopy(result))
            command_evidence = getattr(controller, '_last_command_evidence', None)
            if isinstance(command_evidence, dict):
                record['command_evidence'] = deepcopy(command_evidence)
            return result
        finally:
            record['timing'] = _elapsed(before, _sample(agent))
            record['completed_at_unix'] = time.time()
    return timed


def record_simulation_function(function):
    """Add timing observations only to simulation calls; preserve other execution modes."""
    @wraps(function)
    async def timed(agent, *args, **kwargs):
        if getattr(agent, 'execution_mode', '') != 'simulation':
            return await function(agent, *args, **kwargs)
        records = []
        token = _steps.set(records)
        before = _sample(agent)
        dispatched_at = time.time()
        result = None
        try:
            result = await function(agent, *args, **kwargs)
            return result
        finally:
            evidence = {'resource_id': getattr(agent, 'agent_name', ''),
                        'function_name': args[0] if args else kwargs.get('task_name'),
                        'dispatched_at_unix': dispatched_at,
                        'acknowledged_at_unix': time.time(),
                        'timing': _elapsed(before, _sample(agent)), 'primitive_results': records}
            first_motion = getattr(getattr(agent, '_controller', None), '_first_motion_at_unix', None)
            if first_motion is not None:
                evidence['first_motion_at_unix'] = first_motion
                runtime = getattr(agent, 'environment_runtime', None)
                requested = getattr(runtime, 'startup_timing', {}).get('start_requested_at_unix')
                if requested is not None:
                    evidence['start_to_first_movement_wall_time_sec'] = first_motion-requested
            agent.last_simulation_timing = deepcopy(evidence)
            if isinstance(result, dict):
                if result.get('observations') is None:
                    result['observations'] = {}
                result['observations']['simulation_execution'] = evidence
            _steps.reset(token)
    return timed

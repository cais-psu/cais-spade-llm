from __future__ import annotations

"""KMR ResourceAgent execution through the existing task and CCA protocol."""

import json
import math
import time
from copy import deepcopy

from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.recovery_framework.delivery import check_stopped, validate_execution_evidence, verify_configuration
from cais_spade_llm.recovery_framework import ROOT, SCENE_PATH
from cais_spade_llm.resources.gazebo_programs import (
    current_program_revision, resource_program_revision, validate_resource_programs,
)
from cais_spade_llm.recovery_framework.gazebo_worker import GazeboExecutionError, GazeboWorker
from cais_spade_llm.recovery_framework.kmr_primitives import (
    KMRPrimitives, KMR_RECOVERY_PRIMITIVES, KMR_RESOURCE_PROFILE,
)


class KMRResourceAgent(ResourceAgent):
    """Execute only descriptor-bound KMR tasks after the standard CCA check."""

    _RESOURCE_PROFILE = KMR_RESOURCE_PROFILE
    _RECOVERY_PRIMITIVES = KMR_RECOVERY_PRIMITIVES

    def __init__(self, jid: str, password: str, *, worker: GazeboWorker | None = None, **kwargs) -> None:
        super().__init__(jid, password, name='KMR', function_names=[], **kwargs)
        self.worker = worker or GazeboWorker()
        self.workflow_worker = self.worker
        self.kmr_primitives = KMRPrimitives(self)
        self._primitive_state = {}
        self._primitive_evidence = {}
        self.current_state = 'idle'
        for name in ('pick_approach', 'pick_part', 'move_to_resource',
                     'place_approach', 'place_release', 'move_to_location'):
            self.executables[name] = getattr(self, name)

    def configure_recovery_safety_observer(self, observer, record: dict) -> None:
        """Bind the KMR owner to its configured continuous observation model."""
        from cais_spade_llm.recovery_framework.kmr_live_safety import configure_observer

        configure_observer(self, observer, record)

    def capture_recovery_safety_state(self, *, max_age: float = 2.) -> dict:
        """Read the configured KMR physical state without dispatching motion."""
        from cais_spade_llm.recovery_framework.kmr_live_safety import capture_state

        return capture_state(self, max_age=max_age)

    def get_recovery_safety_primitive_model(self):
        """Expose KMR owner kinematics for physical AP evaluation."""
        from cais_spade_llm.recovery_framework.kmr_live_safety import primitive_model

        return primitive_model(self)

    def prepare_recovery_safety_program(self, program: dict, checkpoint: dict) -> dict:
        """Prepare a native KMR program on an existing preparation worker thread."""
        from cais_spade_llm.recovery_framework.kmr_live_safety import prepare_program_sync

        return prepare_program_sync(self, program, checkpoint)

    async def prepare_recovery_safety_program_async(self, program: dict, checkpoint: dict) -> dict:
        """Prepare KMR native motion without blocking the agent event loop."""
        from cais_spade_llm.recovery_framework.kmr_live_safety import prepare_program

        return await prepare_program(self, program, checkpoint)

    def validate_recovery_safety_step(self, planned: dict, step: dict) -> None:
        """Require the exact worker-retained trajectory and primitive binding."""
        evidence = step["model_evidence"]
        if (planned["primitive"] != step["primitive"] or planned["params"] != step["resolved_params"]
                or planned["joint_trajectory"] != evidence["joint_trajectory"]
                or planned["continuous_motion"] != evidence["continuous_motion"]):
            raise ValueError("KMR physical evidence differs from its native preparation")

    async def execute_recovery_composition_step(self, *, primitive: str, params: dict,
            task_id: str, step_index: int, grant: dict, owner) -> dict:
        """Execute only worker-retained KMR motion authorized by the CCA."""
        provider = getattr(self, "recovery_composition_evidence_provider", None)
        execute = getattr(provider, "execute_async_step", None)
        if not callable(execute):
            raise ValueError("KMR prepared execution provider is unavailable")
        return await execute(resource_agent=self, task_id=task_id, step_index=step_index,
                             primitive=primitive, params=params, grant=grant)

    def recovery_execution_primitive_catalog(self) -> list[dict]:
        """Expose only primitives selectable in this pinned Gazebo scene."""
        rows = super().recovery_execution_primitive_catalog()
        runtime = getattr(self, 'environment_runtime', None) or getattr(self, 'delivery_runtime', None)
        scene = (runtime.context.inputs['scene'] if runtime is not None
                 else json.loads(SCENE_PATH.read_text()))
        if 'resource_programs' not in scene:
            return rows
        validate_resource_programs(scene)
        catalog = scene['resource_programs']['resources']['KMR']['primitives']
        return [row for row in rows if catalog.get(row['name'], {}).get('status') == 'executable'
                and catalog[row['name']].get('recovery_selectable') is True]

    async def pick_approach(self, **params) -> dict:
        """Dock at the configured Storage pickup and approach the selected part.

        Args:
            **params: Exact bound nominal task parameters.

        Returns:
            Gazebo execution and observed approach evidence.
        """
        return await self._execute('pick_approach', params)

    async def pick_part(self, **params) -> dict:
        """Pick the bound Storage part and park with acknowledged custody.

        ---
        description: Pick the exact configured part from Storage.
        in_state: at_pick
        out_state: carrying
        params:
          part_name: {type: string}
          origin_resource_location: {type: string}
        ---
        """
        return await self._execute('pick_part', params)

    async def move_to_resource(self, **params) -> dict:
        """Follow DockKMR with confirmed attachment and the arm parked.

        ---
        description: Move along the configured route with acknowledged custody.
        in_state: carrying
        out_state: carrying
        params:
          target_resource: {type: string}
        ---
        """
        return await self._execute('move_to_resource', params)

    async def place_approach(self, **params) -> dict:
        """Approach loaded M1/M2 workholding while retaining custody.

        Args:
            **params: Exact bound nominal task parameters.

        Returns:
            Gazebo execution and observed approach evidence.
        """
        return await self._execute('place_approach', params)

    async def place_release(self, **params) -> dict:
        """Release into M1 and withdraw before acknowledging the shared handoff.

        ---
        description: Place and release the held part into M1 workholding.
        in_state: positioned
        out_state: idle
        params:
          part_name: {type: string}
          destination_location: {type: string}
        ---
        """
        return await self._execute('place_release', params)

    async def move_to_location(self, x: float, y: float, yaw: float) -> dict:
        """Execute the saved coordinate function from an active pinned Gazebo context."""
        runtime = getattr(self, 'environment_runtime', None) or getattr(self, 'delivery_runtime', None)
        if runtime is None or runtime.stopped:
            return {'status': 'failed', 'content': 'No active KMR Gazebo context'}
        if any(type(value) not in {int, float} or not math.isfinite(value)
               for value in (x, y, yaw)):
            return {'status': 'failed', 'content': 'KMR coordinates must be finite numbers'}
        scene = runtime.context.inputs['scene']
        try:
            validate_resource_programs(scene)
            if (runtime.scene_file
                    and current_program_revision(ROOT / runtime.scene_file)
                    != resource_program_revision(scene)):
                raise ValueError('Stale Gazebo program revision')
            valuation = runtime.context.snapshot()
            kmr = valuation['KMR']
            if kmr['resource_state'] not in {'idle', 'carrying'}:
                raise ValueError('KMR coordinate travel requires idle or carrying state')
            if (kmr['resource_state'] == 'idle') != (kmr['held_part'] is None):
                raise ValueError('KMR held part disagrees with its function state')
            pending = getattr(runtime.context, 'pending_tasks', {})
            if pending and any(task.get('resource_id') == 'KMR' for task in pending.values()):
                raise ValueError('KMR already has a pending function')
            if getattr(runtime.context, 'pending', None):
                raise ValueError('KMR already has a pending function')
            custody = deepcopy(getattr(self, 'workflow_custody', None) or {})
            if kmr['held_part'] is not None and custody.get('grasp_transform') is None:
                raise ValueError('KMR carrying state lacks measured grasp evidence')
            request = {
                'mode': 'function', 'inputs': runtime.context.inputs,
                'valuation': valuation, 'geometry': deepcopy(getattr(runtime.context, 'geometry', {})),
                'probe': getattr(runtime, 'kmr_probe', None) or runtime.prepared.get('probe'),
                'custody': custody,
                'pending': {'event_name': 'move_to_location',
                            'parameters': {'x': x, 'y': y, 'yaw': yaw}},
            }
            self._kmr_execution_request = deepcopy(request)
            result = await self.worker.run(request)
            self.record_primitive_evidence(result)
            observation = result['observations']
            pose = observation.get('base_pose')
            if not isinstance(pose, list) or len(pose) != 3:
                raise ValueError('KMR coordinate movement lacks measured base pose')
            location = observation.get('resource_location')
            if location is not None and location not in scene['resource_programs']['resources']:
                raise ValueError('KMR coordinate movement reported an unknown location')
            model_parameters = {
                'resource_id': 'KMR', 'x': x, 'y': y, 'yaw': yaw,
                'resource_state': kmr['resource_state'],
                'observed_resource_location': location,
            }
            if hasattr(runtime.context, 'pending_tasks'):
                from cais_spade_llm.resources.environment_models import project_transition

                event = next(row for row in runtime.context.models['KMR']['events']
                             if row['event_name'] == 'move_to_location'
                             and row['parameter_bindings']['resource_state']['equals'] == kmr['resource_state'])
                projected, _ = project_transition(
                    runtime.context.models, valuation, runtime.context.part_tracker,
                    {'resource_id': 'KMR', 'event_id': event['event_id'],
                     'event_name': 'move_to_location', 'parameters': model_parameters},
                    runtime.context.product_name, runtime.context.requirements,
                )
            else:
                from cais_spade_llm.resources.nominal_des import project_nominal_event

                projected = project_nominal_event(
                    runtime.context.models, valuation, 'KMR', 'move_to_location',
                    model_parameters,
                )
            context = runtime.context.resources['KMR']
            context.valuation = projected['KMR']
            context.revision += 1
            context.evidence = 'Gazebo move_to_location at measured base pose'
            self._primitive_state.update(base_pose=deepcopy(pose), resource_location=location)
            if hasattr(runtime, 'queue_save'):
                runtime.queue_save()
            else:
                runtime.save()
            return {'status': 'completed', 'observations': observation}
        except (GazeboExecutionError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            if isinstance(exc, GazeboExecutionError):
                self.record_primitive_evidence(exc.result)
                return {'status': 'failed:gazebo', 'content': str(exc), 'observations': exc.result}
            return {'status': 'failed', 'content': str(exc)}

    async def _execute(self, name: str, params: dict) -> dict:
        runtime = self.delivery_runtime
        check_stopped()
        verify_configuration(runtime.prepared)
        pending = deepcopy(runtime.context.pending)
        if runtime.stopped or not pending or pending['event_name'] != name:
            raise ValueError('No matching pending delivery task')
        received = {k: v for k, v in params.items() if k not in {'task_id', 'product_jid', 'phase_id'}}
        if params.get('task_id') != pending['task_id'] or json.dumps(received, sort_keys=True) != json.dumps(pending['parameters'], sort_keys=True):
            raise ValueError('KMR task bindings disagree with ProductAgent')
        self.validate_nominal_event(runtime.context.models, runtime.context.snapshot(), pending)
        try:
            request = {'mode': 'task', 'inputs': runtime.context.inputs,
                                           'pending': pending, 'probe': runtime.prepared['probe'],
                                           'startup_timing': runtime.prepared.get('startup_timing', {}),
                                           'dispatch_requested_at_unix': time.time(),
                                           'custody': (runtime.context.transitions[-1]['acknowledgement']['observations']
                                                       if runtime.context.transitions else None)}
            self._kmr_execution_request = deepcopy(request)
            result = await self.worker.run(request)
            self.record_primitive_evidence(result)
        except GazeboExecutionError as exc:
            self.record_primitive_evidence(exc.result)
            return {'status': 'failed:gazebo', 'content': str(exc), 'observations': exc.result}
        ack = {**pending, 'status': 'completed', 'evidence': 'gazebo', 'observations': result['observations']}
        validate_execution_evidence(ack, scene=runtime.context.inputs['scene'])
        self.current_state = {
            'pick_approach': 'at_pick',
            'pick_part': 'carrying',
            'move_to_resource': 'carrying',
            'place_approach': 'positioned',
            'place_release': 'idle',
        }[name]
        return {'status': 'completed', 'observations': {'nominal_acknowledgement': ack}}

    def record_primitive_evidence(self, result: dict) -> None:
        """Retain physical partial progress independently of nominal acknowledgements."""
        self._primitive_evidence = deepcopy(result)
        records = result.get('primitive_results', result.get('observations', {}).get('primitive_results', []))
        request = getattr(self, '_kmr_execution_request', {})
        part = (request.get('primitive_parameters', {}).get('part_name')
                or request.get('pending', {}).get('parameters', {}).get('part_name'))
        for record in records:
            if record.get('status') != 'completed':
                continue
            primitive = record['primitive']
            if primitive in {'open_gripper', 'close_gripper'}:
                self._primitive_state['gripper_state'] = 'open' if primitive == 'open_gripper' else 'closed'
            output = record.get('result')
            if primitive in {'attach_part', 'grasp_part'}:
                self._primitive_state['held_part'] = (output or {}).get('held_part', part)
                self._primitive_state['gripper_state'] = 'closed'
            elif primitive in {'detach_part', 'release_part'}:
                self._primitive_state['held_part'] = None
                self._primitive_state['gripper_state'] = 'open'
            if primitive == 'move_base' and isinstance(output, dict):
                if output.get('base_pose') is not None:
                    self._primitive_state['base_pose'] = deepcopy(output['base_pose'])
                if 'resource_location' in output:
                    self._primitive_state['resource_location'] = output['resource_location']
            if isinstance(output, dict) and 'attached' in output:
                measured = deepcopy(getattr(self, 'workflow_custody', None) or {})
                measured.update(deepcopy(output))
                if output['attached'] is False:
                    measured.pop('grasp_transform', None)
                    measured.pop('carrying_arm_configuration', None)
                self.workflow_custody = measured
            if isinstance(output, dict) and output.get('tcp_pose'):
                self._primitive_state['current_pose'] = deepcopy(output['tcp_pose'])
        observation = result.get('observations', {})
        if observation.get('grasp_transform'):
            self.workflow_custody = deepcopy(observation)
        self._primitive_state['last_execution_evidence'] = deepcopy(result)

    def _snapshot_state(self) -> dict:
        snapshot = super()._snapshot_state()
        snapshot.update(resource_type='kmr', **self._primitive_state)
        runtime = getattr(self, 'delivery_runtime', None)
        if runtime is not None:
            snapshot['execution_outcome'] = deepcopy(runtime.outcome)
            if runtime.context.transitions:
                ack = runtime.context.transitions[-1]['acknowledgement']
                snapshot['state_evidence'] = (
                    f"Last Gazebo acknowledgement: {ack['event_name']}; "
                    f"revision {runtime.context.revision}; launch {runtime.prepared['probe']['launch_id']}."
                )
        return snapshot

    def get_recovery_physical_snapshot(self) -> dict:
        """Export retained KMR observations without filling missing poses or custody.

        This is a checkpoint export, not a fresh controller query or a prepared
        trajectory. A preparation provider must establish freshness and coverage.
        """
        state = deepcopy(getattr(self, '_primitive_state', {}))
        custody = deepcopy(getattr(self, 'workflow_custody', None) or {})
        snapshot = {
            key: deepcopy(state[key])
            for key in ('current_pose', 'base_pose', 'held_part', 'gripper_state',
                        'resource_location', 'current_state')
            if key in state
        }
        if 'grasp_transform' in custody:
            snapshot['grasp_transform'] = deepcopy(custody['grasp_transform'])
        runtime = getattr(self, 'environment_runtime', None) or getattr(self, 'delivery_runtime', None)
        evidence = {
            'source': 'KMRResourceAgent.record_primitive_evidence',
            'primitive_evidence': deepcopy(getattr(self, '_primitive_evidence', {})),
            'workflow_custody': custody,
        }
        if runtime is not None:
            evidence['revision'] = getattr(runtime.context, 'revision', None)
            evidence['part_tracker'] = deepcopy(getattr(runtime.context, 'part_tracker', {}))
        return {'resource_id': 'KMR', 'resource_jid': str(self.jid),
                'snapshot': snapshot, 'evidence': evidence}

    async def teardown(self) -> None:
        """Stop pending execution without dropping or resetting a held part."""
        runtime = getattr(self, 'delivery_runtime', None)
        if runtime is not None:
            runtime.stop()
        environment = getattr(self, 'environment_runtime', None)
        if environment is not None:
            environment.stop()
        await self.worker.cancel()
        if (runtime is not None and runtime.context.pending is not None
                and self.worker.task_id == runtime.context.pending['task_id']
                and self.worker.last_result is not None):
            runtime.outcome['pending_execution_result'] = deepcopy(self.worker.last_result)
            runtime.save()

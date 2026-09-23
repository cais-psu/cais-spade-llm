"""KMR ResourceAgent execution through the existing task and CCA protocol."""

from __future__ import annotations

import json
import time
from copy import deepcopy

from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.recovery_framework.delivery import check_stopped, validate_execution_evidence, verify_configuration
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
        for name in ('pick_part', 'move_to_resource', 'place_release'):
            self.executables[name] = getattr(self, name)

    async def pick_part(self, **params) -> dict:
        """Pick the bound Storage part and park with acknowledged custody.

        ---
        description: Pick the exact configured part from Storage.
        in_state: idle
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

    async def place_release(self, **params) -> dict:
        """Release into M1 and withdraw before acknowledging the shared handoff.

        ---
        description: Place and release the held part into M1 workholding.
        in_state: carrying
        out_state: idle
        params:
          part_name: {type: string}
          destination_location: {type: string}
        ---
        """
        return await self._execute('place_release', params)

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
        validate_execution_evidence(ack)
        self.current_state = 'idle' if name == 'place_release' else 'carrying'
        return {'status': 'completed', 'observations': {'nominal_acknowledgement': ack}}

    def record_primitive_evidence(self, result: dict) -> None:
        """Retain physical partial progress independently of nominal acknowledgements."""
        self._primitive_evidence = deepcopy(result)
        records = result.get('primitive_results', result.get('observations', {}).get('primitive_results', []))
        request = getattr(self, '_kmr_execution_request', {})
        part = request.get('pending', {}).get('parameters', {}).get('part_name')
        for record in records:
            if record.get('status') != 'completed':
                continue
            primitive = record['primitive']
            if primitive in {'open_gripper', 'close_gripper'}:
                self._primitive_state['gripper_state'] = 'open' if primitive == 'open_gripper' else 'closed'
            if primitive == 'attach_part':
                self._primitive_state.update(held_part=part, current_state='carrying')
            elif primitive == 'detach_part':
                self._primitive_state.update(held_part=None, current_state='idle')
            output = record.get('result')
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

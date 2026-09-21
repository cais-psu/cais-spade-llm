"""KMR ResourceAgent execution through the existing task and CCA protocol."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.delivery import check_stopped, register_worker, validate_execution_evidence, verify_configuration


class GazeboExecutionError(RuntimeError):
    """Carry failed observations without establishing nominal completion."""

    def __init__(self, result: dict) -> None:
        super().__init__(result.get('error', 'KMR operation failed'))
        self.result = result


class GazeboWorker:
    """Own one cancellable ROS process without importing ROS into the UI loop."""

    def __init__(self) -> None:
        self.process = None
        self.last_result = None
        self.task_id = None
        self._result_path = None
        register_worker(self)

    async def run(self, request: dict) -> dict:
        """Execute an explicit probe or task and read its acknowledged evidence."""
        check_stopped()
        if self.process is not None:
            raise RuntimeError('KMR already has an active operation')
        self.last_result = None
        self.task_id = request.get('pending', {}).get('task_id')
        with tempfile.TemporaryDirectory(prefix='cais_kmr_') as directory:
            request_path = Path(directory) / 'request.json'
            result_path = Path(directory) / 'result.json'
            self._result_path = result_path
            request_path.write_text(json.dumps(request), encoding='utf-8')
            script = 'source /opt/ros/humble/setup.bash\nsource "$1"\nexec /usr/bin/python3 -m cais_spade_llm.recovery_framework.kmr_gazebo "$2" "$3"'
            with (Path(directory) / 'worker.log').open('w') as log:
                self.process = subprocess.Popen(
                    ['bash', '-c', script, 'cais-kmr', str(Path.home() / 'ros2_ws/install/setup.bash'),
                     str(request_path), str(result_path)], cwd=ROOT,
                    stdout=log, stderr=log, start_new_session=True,
                )
                try:
                    # SPADE can change the event-loop policy after UI startup;
                    # a thread-owned wait does not depend on asyncio child watchers.
                    await asyncio.to_thread(self.process.wait, timeout=360)
                except (asyncio.CancelledError, subprocess.TimeoutExpired):
                    await self.cancel()
                    raise
                finally:
                    self.process = None
                    if result_path.is_file():
                        self.last_result = json.loads(result_path.read_text())
                    self._result_path = None
            if not result_path.is_file():
                detail = (Path(directory) / 'worker.log').read_text()[-3000:]
                raise RuntimeError(f'KMR ROS worker returned no evidence: {detail}')
            result = json.loads(result_path.read_text())
            if result.get('status') != 'completed':
                raise GazeboExecutionError(result)
            return result

    async def cancel(self) -> None:
        """Cancel controller goals before terminating the owned ROS process."""
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            await asyncio.to_thread(process.wait, timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            await asyncio.to_thread(process.wait, timeout=5)
        if self._result_path is not None and self._result_path.is_file():
            self.last_result = json.loads(self._result_path.read_text())


class KMRResourceAgent(ResourceAgent):
    """Execute only descriptor-bound KMR tasks after the standard CCA check."""

    def __init__(self, jid: str, password: str, **kwargs) -> None:
        super().__init__(jid, password, name='KMR', function_names=[], **kwargs)
        self.worker = GazeboWorker()
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
            result = await self.worker.run({'mode': 'task', 'inputs': runtime.context.inputs,
                                           'pending': pending, 'probe': runtime.prepared['probe'],
                                           'custody': (runtime.context.transitions[-1]['acknowledgement']['observations']
                                                       if runtime.context.transitions else None)})
        except GazeboExecutionError as exc:
            return {'status': 'failed:gazebo', 'content': str(exc), 'observations': exc.result}
        ack = {**pending, 'status': 'completed', 'evidence': 'gazebo', 'observations': result['observations']}
        validate_execution_evidence(ack)
        self.current_state = 'idle' if name == 'place_release' else 'carrying'
        return {'status': 'completed', 'observations': {'nominal_acknowledgement': ack}}

    def _snapshot_state(self) -> dict:
        return {'current_state': self.current_state, **self.nominal_context.snapshot()}

    async def teardown(self) -> None:
        """Stop pending execution without dropping or resetting a held part."""
        runtime = getattr(self, 'delivery_runtime', None)
        if runtime is not None:
            runtime.stop()
        await self.worker.cancel()
        if (runtime is not None and runtime.context.pending is not None
                and self.worker.task_id == runtime.context.pending['task_id']
                and self.worker.last_result is not None):
            runtime.outcome['pending_execution_result'] = deepcopy(self.worker.last_result)
            runtime.save()

from __future__ import annotations

"""Persistent ROS worker owned by one recovery resource."""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.delivery import check_stopped, register_worker

logger = logging.getLogger(__name__)


class GazeboExecutionError(RuntimeError):
    """Carry failed observations without establishing nominal completion."""

    def __init__(self, result: dict) -> None:
        super().__init__(result.get('error', 'KMR operation failed'))
        self.result = result


class GazeboWorker:
    """Own one cancellable ROS process without importing ROS into the UI loop."""

    def __init__(
        self,
        *,
        module: str = "cais_spade_llm.recovery_framework.kmr_gazebo",
        label: str = "KMR",
        timeout_sec: float = 360.0,
    ) -> None:
        self.module = module
        self.label = label
        self.timeout_sec = timeout_sec
        self.process = None
        self.last_result = None
        self.progress = None
        self.task_id = None
        self._result_path = None
        self._directory = None
        self._log = None
        self._log_reader = None
        self._operation_lock = threading.Lock()
        self._control_lock = asyncio.Lock()
        self.machine_jobs = None
        self.processing_shutdown = None
        register_worker(self)

    async def run(self, request: dict) -> dict:
        """Execute an explicit probe or task and read its acknowledged evidence."""
        check_stopped()
        guard = getattr(self, "physical_request_guard", None)
        if callable(guard):
            guard(request)
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError(f'{self.label} already has an active operation')
        try:
            attempts = 2 if request.get('mode') == 'probe' else 1
            for attempt in range(attempts):
                try:
                    result = await self._exchange(request)
                    break
                except RuntimeError:
                    dead = self.process is None or self.process.poll() is not None
                    if attempt + 1 >= attempts or not dead:
                        raise
                    self._cleanup()
                    await asyncio.sleep(.5)
            if result.get('status') != 'completed':
                logger.error('[%s] %s failed: %s', self.label,
                             request.get('pending', {}).get('event_name') or request.get('mode'),
                             result.get('error', 'No completion evidence'))
                raise GazeboExecutionError(result)
            return result
        except (asyncio.CancelledError, TimeoutError):
            await self.cancel()
            raise
        finally:
            self._operation_lock.release()

    async def _exchange(self, request: dict) -> dict:
        """Exchange one request with the owned process; probe callers may restart it."""
        self.last_result = None
        self.task_id = request.get('pending', request.get('task', {})).get('task_id')
        if self.process is None or self.process.poll() is not None:
            if self.machine_jobs and self.machine_jobs.get('active_job'):
                identity = self.machine_jobs['active_job']
                self.machine_jobs['jobs'][identity].update(job_state='interrupted',
                    interruption='Worker exited without acknowledged completion')
                self.machine_jobs.update(active_job=None, revision=self.machine_jobs['revision'] + 1)
            self._cleanup()
            self._directory = tempfile.TemporaryDirectory(prefix='cais_gazebo_worker_')
            directory = Path(self._directory.name)
            self._log = (directory / 'worker.log').open('w')
            if self.label == 'KMR':
                self._log_reader = (directory / 'worker.log').open()
            script = (
                'source /opt/ros/humble/setup.bash\n'
                'source "$1"\n'
                'exec /usr/bin/python3 -m "$2" --serve "$3"'
            )
            self.process = subprocess.Popen(
                ['bash', '-c', script, 'cais-kmr', str(Path.home() / 'ros2_ws/install/setup.bash'),
                 self.module, str(directory)], cwd=ROOT, stdout=self._log, stderr=self._log,
                start_new_session=True,
            )
        directory = Path(self._directory.name)
        identifier = uuid4().hex
        result_path = directory / 'result.json'
        progress_path = directory / 'progress.json'
        self._result_path = result_path
        result_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        temporary = directory / 'request.tmp'
        temporary.write_text(json.dumps({
            'id': identifier,
            'request': {**request, '_progress_path': str(progress_path),
                        '_jobs_path': request.get('_jobs_path', str(directory / 'jobs.json')),
                        '_control_path': str(directory / 'control.json'),
                        'machine_jobs': deepcopy(self.machine_jobs)},
        }))
        temporary.replace(directory / 'request.json')
        self._jobs_path = Path(request.get('_jobs_path', str(directory / 'jobs.json')))
        deadline = time.monotonic() + self.timeout_sec
        while True:
            self._relay_log()
            self.read_machine_jobs()
            if self._directory is None:
                return self.last_result or {
                    'status': 'failed', 'error': f'{self.label} worker cancelled'
                }
            if result_path.is_file():
                envelope = json.loads(result_path.read_text())
                if envelope['id'] == identifier:
                    self._relay_log()
                    self.last_result = envelope['result']
                    return self.last_result
            if progress_path.is_file():
                try:
                    self.progress = json.loads(progress_path.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
            return_code = self.process.poll()
            if return_code is not None:
                detail = (directory / 'worker.log').read_text()[-3000:].strip()
                if not detail:
                    detail = f'process exited with return code {return_code}'
                    if return_code < 0:
                        try:
                            detail += f' ({signal.Signals(-return_code).name})'
                        except ValueError:
                            pass
                raise RuntimeError(f'{self.label} ROS worker returned no evidence: {detail}')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{self.label} ROS worker exceeded its wall-clock watchdog')
            await asyncio.sleep(.02)

    def read_machine_jobs(self) -> dict | None:
        """Read the owner's latest atomic record without commanding the worker."""
        if self._directory is not None:
            path = getattr(self, '_jobs_path', Path(self._directory.name) / 'jobs.json')
            if path.is_file():
                self.machine_jobs = json.loads(path.read_text())
        return deepcopy(self.machine_jobs)

    async def machine_control(self, request: dict, *, command_id: str) -> dict:
        """Service job controls while the machine_part request is still running."""
        check_stopped()
        guard = getattr(self, 'physical_request_guard', None)
        if callable(guard):
            guard(request)
        return await self._machine_control_exchange(request, command_id=command_id)

    async def observe_machine_checkpoint(self, *, run_id: str, resource_id: str, checkpoint_id: str) -> dict:
        """Ask the owner to observe a frozen checkpoint without a job command."""
        return await self._machine_control_exchange({"run_id": run_id, "resource_id": resource_id,
            "observation_only": True, "checkpoint_id": checkpoint_id}, command_id=checkpoint_id)

    async def _machine_control_exchange(self, request: dict, *, command_id: str) -> dict:
        async with self._control_lock:
            if self._directory is None or self.process is None or self.process.poll() is not None:
                raise RuntimeError('Machine worker is not running; no job was resumed')
            directory = Path(self._directory.name)
            temporary = directory / 'control.tmp'
            temporary.write_text(json.dumps({'id': command_id, 'request': request}))
            temporary.replace(directory / 'control.json')
            deadline = time.monotonic() + 30.
            while self._directory is not None and self.process.poll() is None:
                path = directory / 'control_result.json'
                if path.is_file():
                    envelope = json.loads(path.read_text())
                    if envelope['id'] == command_id:
                        self.read_machine_jobs()
                        return envelope['result']
                if time.monotonic() >= deadline:
                    # Expired commands must not execute after the caller abandons them.
                    await self.cancel()
                    raise TimeoutError('Machine control acknowledgement timed out')
                await asyncio.sleep(.02)
            raise RuntimeError('Machine worker stopped before acknowledging control')

    def _relay_log(self) -> None:
        """Tee the KMR worker's existing ROS logs to the operator console."""
        if self._log_reader is not None:
            output = self._log_reader.read()
            if output:
                sys.stderr.write(output)
                sys.stderr.flush()

    def _cleanup(self) -> None:
        self._relay_log()
        if self._log_reader is not None:
            self._log_reader.close()
            self._log_reader = None
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
        self.process = None
        self._result_path = None
        self.progress = None

    async def cancel(self) -> None:
        """Cancel controller goals before terminating the owned ROS process."""
        process = self.process
        self.read_machine_jobs()
        if process is None or process.poll() is not None:
            self._cleanup()
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
            self.last_result = json.loads(self._result_path.read_text())['result']
        self.processing_shutdown = self.read_machine_jobs()
        self._cleanup()

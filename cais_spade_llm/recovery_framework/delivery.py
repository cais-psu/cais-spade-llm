"""Product-owned Storage-to-M1 delivery planning and Gazebo acknowledgements."""

from __future__ import annotations

import hashlib
import os
import signal
import threading
import time
import weakref
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from cais_spade_llm.product.nominal import NominalProductContext
from cais_spade_llm.recovery_framework import ROOT, fingerprint
from cais_spade_llm.recovery_framework.reports import LatestReport

ORDER_PATH = ROOT / 'cais_spade_llm/specification/products/orders/assembly_board-v1-kmr-storage-m1.json'
RUN_DIRECTORY = ROOT / 'cais_spade_llm/monitor/recovery_gazebo_runs'
_prepared: dict | None = None
_cancelled = threading.Event()
_workers = weakref.WeakSet()
_prepared_workers: dict[str, dict] = {}
_prepared_workers_lock = threading.Lock()


def register_worker(worker) -> None:
    """Track owned processes so Stop System can interrupt active controller goals."""
    _workers.add(worker)


def retain_prepared_worker(worker, snapshot: dict) -> str:
    """Retain a probed worker outside the serializable preparation snapshot."""
    token = uuid4().hex
    binding = {
        'worker': worker,
        'source_fingerprints': deepcopy(snapshot.get('source_fingerprints', {})),
        'launch_id': snapshot.get('probe', {}).get('launch_id'),
        'scene_fingerprint': snapshot.get('probe', {}).get('scene_fingerprint'),
        'scene_asset_fingerprint': snapshot.get('probe', {}).get('scene_asset_fingerprint'),
    }
    with _prepared_workers_lock:
        _prepared_workers[token] = binding
    return token


def claim_prepared_worker(snapshot: dict):
    """Transfer one exactly-bound preparation worker to its resource agent."""
    token = str(snapshot.get('prepared_resource_token') or '')
    if not token:
        return None
    with _prepared_workers_lock:
        binding = _prepared_workers.pop(token, None)
    if binding is None:
        return None
    expected = {
        'source_fingerprints': snapshot.get('source_fingerprints', {}),
        'launch_id': snapshot.get('probe', {}).get('launch_id'),
        'scene_fingerprint': snapshot.get('probe', {}).get('scene_fingerprint'),
        'scene_asset_fingerprint': snapshot.get('probe', {}).get('scene_asset_fingerprint'),
    }
    if any(binding[key] != expected[key] for key in expected):
        raise ValueError('Prepared KMR worker does not match this Start request')
    return binding['worker']


def take_unclaimed_prepared_workers() -> list:
    """Remove and return preparation workers which no resource agent claimed."""
    with _prepared_workers_lock:
        workers = [entry['worker'] for entry in _prepared_workers.values()]
        _prepared_workers.clear()
    return workers


def request_stop() -> None:
    """Suppress dispatch and signal cancellation across the UI and agent loops."""
    _cancelled.set()
    for worker in list(_workers):
        process = worker.process
        if process is not None and process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass


def reset_stop() -> None:
    """Begin an explicitly requested preparation with no previous stop request."""
    _cancelled.clear()


def check_stopped() -> None:
    """Reject any task or startup step after Stop System."""
    if _cancelled.is_set():
        raise ValueError('Stop System cancelled delivery startup or execution')


def is_delivery_order(order: dict) -> bool:
    """Check the exact supported delivery goal, independently of its filename."""
    supported = {
        'product': 'assembly_board-v1', 'product_jid': 'assembly_board-v1@localhost',
        'quantity': 1, 'parts': ['KET4_Square_4mm'],
        'completion_conditions': {
            'M1': {'resource_state': 'loaded', 'part_name': 'KET4_Square_4mm'},
            'Storage': {'inventory.KET4_Square_4mm': False},
            'KMR': {'resource_state': 'idle', 'held_part': None, 'resource_location': 'M1'},
        },
    }
    return all(order.get(key) == supported[key] for key in (
        'product', 'product_jid', 'quantity', 'parts', 'completion_conditions'
    ))


def verify_configuration(snapshot: dict) -> None:
    """Reject mutable source changes after preparation and before dispatch."""
    for reference, expected in snapshot.get('source_fingerprints', {}).items():
        path = Path(reference)
        if not path.is_absolute():
            path = ROOT / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Configuration changed after preparation: {reference}; save setup and start again')


def prepare_start(snapshot: dict | None) -> None:
    """Supply an explicitly prepared Start System snapshot to the agent factory."""
    global _prepared
    _prepared = deepcopy(snapshot)


def prepared_start() -> dict | None:
    """Read the prepared snapshot without reading mutable setup files."""
    return deepcopy(_prepared)


def record_preparation_failure(snapshot: dict, failure: BaseException) -> Path:
    """Retain a failed explicit Start request without inventing task execution."""
    context = NominalProductContext(**snapshot['inputs'])
    store = LatestReport(RUN_DIRECTORY)
    report = {'schema_version': 1, 'evidence': 'gazebo', 'setup': snapshot['setup'],
              'source_fingerprints': snapshot['source_fingerprints'],
              'source_snapshots': snapshot['source_snapshots'], 'inputs': context.inputs,
              'input_fingerprint': fingerprint(context.inputs), 'models': context.models,
              'descriptor_fingerprint': fingerprint(context.models),
              'initial_observations': deepcopy(getattr(failure, 'result', snapshot.get('probe', {}))),
              'plans': [], 'pending': None, 'transitions': [], 'history': context.history,
              'final_valuation': context.snapshot(), 'final_product_states': context.part_tracker,
              'outcome': {'status': 'stopped' if _cancelled.is_set() else 'preparation_failed',
                          'acknowledged_tasks': 0, 'reason': str(failure),
                          'valuation_source': 'configured initial values; no task was acknowledged'}}
    store.save(report)
    return store.path/'run.json'


def validate_execution_evidence(ack: dict) -> None:
    """Reject incomplete or unsuccessful KMR delivery observations."""
    if not isinstance(ack, dict) or ack.get('evidence') != 'gazebo' or ack.get('status') != 'completed':
        raise ValueError('Expected a completed Gazebo acknowledgement')
    observation = ack.get('observations', {})
    if not isinstance(observation, dict) or not observation.get('launch_id') or not observation.get('scene_fingerprint'):
        raise ValueError('Gazebo acknowledgement lacks scene provenance')
    if ack.get('resource_id') != 'KMR':
        raise ValueError('Only KMR delivery execution is integrated')
    name = ack.get('event_name')
    requirements = {
        'pick_part': {'part_location': 'KMR', 'attached': True, 'arm_parked': True},
        'move_to_resource': {'part_location': 'KMR', 'attached': True, 'arm_parked': True, 'resource_location': 'M1'},
        'place_release': {'part_location': 'M1', 'attached': False, 'robot_clear': True},
    }
    if name not in requirements:
        raise ValueError('Unsupported Gazebo nominal event')
    required = {'part_name': 'KET4_Square_4mm', 'controllers_succeeded': True, **requirements[name]}
    if any(type(observation.get(k)) is not type(v) or observation.get(k) != v for k, v in required.items()):
        raise ValueError('Gazebo observations do not establish the task effects')
    operations = observation.get('operations')
    if not isinstance(operations, list) or not operations or any(row.get('success') is not True for row in operations):
        raise ValueError('Missing or failed controller/observation evidence')
    primitives = observation.get('primitive_results')
    if primitives is not None:
        from cais_spade_llm.recovery_framework.kmr_tasks import KMR_TASKS

        expected = KMR_TASKS[name].program.steps
        if (not isinstance(primitives, list)
                or not all(isinstance(row, dict) for row in primitives)
                or [(row.get('step_id'), row.get('primitive')) for row in primitives]
                != [(step.id, step.op) for step in expected]
                or any(row.get('status') != 'completed' or row.get('task_id') != ack.get('task_id')
                       for row in primitives)):
            raise ValueError('Gazebo primitives do not match the declared task composition')


class DeliveryRuntime:
    """Bind ProductAgent planning and acknowledgements to shared resource contexts."""

    def __init__(self, agent, prepared: dict, resources: list) -> None:
        verify_configuration(prepared)
        self.agent = agent
        self.prepared = deepcopy(prepared)
        self.context = NominalProductContext(**prepared['inputs'])
        if not is_delivery_order(self.context.product_order):
            raise ValueError('Gazebo delivery requires the supported Storage-to-M1 order')
        self.actor = next(resource for resource in resources if resource.agent_name == 'KMR')
        for resource in resources:
            resource.configure_nominal(self.context.resources[resource.agent_name])
        self.actor.delivery_runtime = self
        self.agent.nominal_context = self.context
        self.agent.part_tracker = deepcopy(self.context.part_tracker)
        self.reports = LatestReport(RUN_DIRECTORY)
        self.run_id = self.reports.run_id
        self.path = self.reports.path
        self.outcome = {'status': 'prepared', 'acknowledged_tasks': 0}
        self.stopped = False
        self.dispatch_timing: dict[str, dict] = {}

    def build_plan(self) -> tuple:
        """Build the existing task DAG/FSA from nominal resource capabilities."""
        check_stopped()
        plan = self.context.plan()
        if plan['status'] != 'planned':
            raise ValueError(f'Delivery planning {plan["status"]}')
        nodes = []
        for index, task in enumerate(plan['tasks'], 1):
            if task['resource_id'] != 'KMR':
                raise ValueError('Delivery plan contains unsupported execution')
            node = {'id': f'nominal_{self.run_id}_{index}', 'type': 'task', 'requirement_id': 'delivery',
                    'function_name': task['event_name'], 'params': deepcopy(task['parameters']),
                    'resource_jid': str(self.actor.jid), 'sequence_index': index-1,
                    'status': 'pending', 'predecessors': [nodes[-1]['id']] if nodes else [],
                    'successors': [], 'nominal_task': task,
                    'in_state': 'idle' if index == 1 else 'carrying',
                    'out_state': 'idle' if task['event_name'] == 'place_release' else 'carrying'}
            if nodes:
                nodes[-1]['successors'].append(node['id'])
            nodes.append(node)
        planner = self.agent.process_planner
        planner.nodes = nodes
        planner.compile_global_fsa()
        self.outcome['status'] = 'planned'
        self.save()
        return nodes, planner.global_fsa

    def prepare_dispatch(self, node: dict) -> None:
        """Revalidate the shared valuation immediately before task dispatch."""
        check_stopped()
        verify_configuration(self.prepared)
        if self.stopped:
            raise ValueError('Delivery is stopped')
        pending = self.context.prepare(node['nominal_task'], task_id=node['id'])
        if pending['task_id'] != node['id']:
            raise ValueError('Delivery task identity mismatch')
        self.outcome['status'] = 'running'
        self.dispatch_timing[node['id']] = {'dispatched_at_unix': time.time()}
        self.save()

    def accept(self, sender: str, payload: dict) -> bool:
        """Authenticate and commit a completed resource acknowledgement atomically."""
        if sender.split('/', 1)[0] != str(self.actor.jid).split('/', 1)[0]:
            raise ValueError('Unexpected delivery acknowledgement')
        if self.stopped or _cancelled.is_set():
            self.stop()
            self.outcome.setdefault('uncommitted_acknowledgements', []).append(deepcopy(payload))
            self.save()
            return False
        if payload.get('status') != 'completed':
            if str(payload.get('status', '')).startswith(('failed', 'blocked', 'rejected')):
                self.outcome = {'status': payload['status'], 'acknowledged_tasks': self.context.revision,
                                'details': deepcopy(payload)}
                self.stopped = True
                self.save()
            return True
        observations = payload.get('observations') or {}
        acknowledgement = observations.get('nominal_acknowledgement')
        if not isinstance(acknowledgement, dict) or acknowledgement.get('task_id') != payload.get('task_id'):
            raise ValueError('Missing matching nominal Gazebo acknowledgement')
        observed = acknowledgement.get('observations', {})
        probe = self.prepared['probe']
        if any(observed.get(key) != probe[key] for key in ('launch_id', 'scene_fingerprint')):
            raise ValueError('Gazebo scene changed during the delivery')
        committed = self.context.acknowledge_gazebo(acknowledgement)
        if not committed:
            return False
        self.agent.part_tracker = deepcopy(self.context.part_tracker)
        timing = self.dispatch_timing.setdefault(payload.get('task_id', ''), {})
        timing['acknowledged_at_unix'] = time.time()
        if timing.get('dispatched_at_unix') is not None:
            timing['dispatch_to_acknowledgement_wall_time_sec'] = (
                timing['acknowledged_at_unix'] - timing['dispatched_at_unix']
            )
        self.outcome = {'status': 'completed' if self.context.plan()['status'] == 'completed' else 'running',
                        'acknowledged_tasks': self.context.revision}
        self.save()
        return committed

    def save(self) -> None:
        """Save the configuration, observations, and acknowledged Gazebo transitions."""
        report = {'schema_version': 1, 'evidence': 'gazebo', 'setup': self.prepared['setup'],
                  'source_fingerprints': self.prepared.get('source_fingerprints', {}),
                  'source_snapshots': self.prepared.get('source_snapshots', {}),
                  'inputs': self.context.inputs, 'input_fingerprint': fingerprint(self.context.inputs),
                  'models': self.context.models, 'descriptor_fingerprint': fingerprint(self.context.models),
                  'initial_observations': self.prepared['probe'], 'plans': self.context.plans,
                  'pending': self.context.pending, 'transitions': self.context.transitions,
                  'history': self.context.history, 'final_valuation': self.context.snapshot(),
                  'final_product_states': self.context.part_tracker,
                  'dispatch_timing': deepcopy(self.dispatch_timing), 'outcome': self.outcome}
        self.reports.save(report)

    def stop(self, reason: str | None = None) -> None:
        """Suppress later dispatch while retaining all acknowledged custody."""
        self.stopped = True
        if self.outcome['status'] in {'prepared', 'planned', 'running'}:
            self.outcome = {'status': 'rejected' if reason else 'stopped',
                            'acknowledged_tasks': self.context.revision}
            if reason:
                self.outcome['reason'] = reason
        self.save()

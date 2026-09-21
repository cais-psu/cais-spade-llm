"""Delivery goals, authenticated Gazebo handoffs, startup, and retired settings."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from cais_spade_llm.product.nominal import NominalProductContext
from cais_spade_llm.recovery_framework import PRODUCT_PATH, ROOT, SCENE_PATH, delivery, read_json, startup
from cais_spade_llm.ui import recovery_setup


@pytest.fixture
def inputs():
    meta = next(iter(read_json(PRODUCT_PATH).values()))
    return {
        'scene': read_json(SCENE_PATH),
        'product_order': read_json(delivery.ORDER_PATH),
        'geometry': read_json(ROOT / meta['product_geometry_file'])['gazebo'],
    }


@pytest.fixture(autouse=True)
def clear_start(monkeypatch, tmp_path):
    monkeypatch.setattr(delivery, 'RUN_DIRECTORY', tmp_path/'runs')
    prepare = delivery.prepare_start
    delivery.reset_stop()
    prepare(None)
    yield
    delivery.reset_stop()
    prepare(None)


def observation(task):
    name = task['event_name']
    return {'launch_id': 'launch', 'scene_fingerprint': 'scene',
            'part_name': 'KET4_Square_4mm', 'part_location': 'M1' if name == 'place_release' else 'KMR',
            'attached': name != 'place_release', 'arm_parked': True,
            'robot_clear': True, 'controllers_succeeded': True, 'resource_location': 'M1',
            'operations': [{'operation': name, 'success': True}]}


def gazebo_ack(pending):
    return {**pending, 'status': 'completed', 'evidence': 'gazebo', 'observations': observation(pending)}


def test_delivery_goal_has_three_tasks_and_preserves_assembly_semantics(inputs):
    context = NominalProductContext(**inputs)
    tasks = context.plan()['tasks']
    assert [task['event_name'] for task in tasks] == ['pick_part', 'move_to_resource', 'place_release']
    for task in tasks:
        assert context.acknowledge_gazebo(gazebo_ack(context.prepare(task)))
    assert context.plan()['status'] == 'completed'
    assert context.revision == 3
    values = context.snapshot()
    assert values['M1']['part_name'] == 'KET4_Square_4mm'
    assert values['M1']['resource_state'] == 'loaded'
    assert values['KMR']['held_part'] is None
    assert values['Storage']['inventory.KET4_Square_4mm'] is False
    assert values['M2']['part_name'] is None
    state = context.part_tracker['KET4_Square_4mm']
    assert state['location'] == 'M1' and state['state'] == 'loaded'
    assert state['processCompleted'] == []


@pytest.mark.parametrize('conditions', [
    {}, {'m1': {'resource_state': 'loaded'}}, {'M1': {'unknown': True}},
    {'M1': {'part_name': 'RGOCG4-50_Round_4mm'}}, {'M1': {'resource_state': 'assembled'}},
    {'Storage': {'inventory.KET4_Square_4mm': 0}},
])
def test_invalid_completion_conditions_rejected(inputs, conditions):
    inputs['product_order']['completion_conditions'] = conditions
    with pytest.raises(ValueError):
        NominalProductContext(**inputs)


def test_gazebo_handoffs_are_atomic_and_offline_acks_stay_strict(inputs):
    context = NominalProductContext(**inputs)
    for task in context.plan()['tasks']:
        before = context.snapshot()
        pending = context.prepare(task)
        ack = gazebo_ack(pending)
        with pytest.raises(ValueError):
            context.acknowledge(ack)
        bad = deepcopy(ack)
        bad['observations']['controllers_succeeded'] = False
        with pytest.raises(ValueError):
            context.acknowledge_gazebo(bad)
        assert context.snapshot() == before
        assert context.acknowledge_gazebo(ack)
        revision = context.revision
        assert not context.acknowledge_gazebo(ack)
        assert context.revision == revision
        bad['observations']['controllers_succeeded'] = True
        bad['revision'] += 1
        with pytest.raises(ValueError):
            context.acknowledge_gazebo(bad)
    assert context.plan()['status'] == 'completed'


def test_stale_ack_and_participant_disagreement_never_commit(inputs):
    context = NominalProductContext(**inputs)
    pending = context.prepare(context.plan()['tasks'][0])
    ack = gazebo_ack(pending)
    ack['revision'] += 1
    before = context.snapshot()
    with pytest.raises(ValueError):
        context.acknowledge_gazebo(ack)
    context.resources['Storage']._valuation['inventory.KET4_Square_4mm'] = False
    with pytest.raises(ValueError):
        context.acknowledge_gazebo(gazebo_ack(pending))
    assert context.revision == 0 and context.history['KET4_Square_4mm'] == []
    assert context.snapshot()['KMR'] == before['KMR']


def test_machine_assignment_cannot_be_changed_by_delivery_order(inputs):
    inputs['product_order']['parts'] = ['RGOCG4-50_Round_4mm']
    context = NominalProductContext(**inputs)
    assert context.plan(max_search_states=1000)['status'] in {'blocked', 'budget_exhausted'}
    assert context.revision == 0


def test_retired_lg_slippage_has_no_active_bindings():
    root = delivery.ROOT / 'cais_spade_llm/initialization'
    assert not (root / 'failure_scenarios/lg_slippage.json').exists()
    for path in (root / 'resources').glob('*.json'):
        assert all(row.get('scenario_id') != 'lg_slippage'
                   for meta in read_json(path).values() for row in meta.get('failure_scenarios', []))


def test_startup_launches_once_reuses_scene_and_checks_configuration(inputs, monkeypatch, tmp_path):
    setup = recovery_setup.default_setup()
    setup['selected_product_order_file'] = str(delivery.ORDER_PATH.relative_to(delivery.ROOT))
    bridge = SimpleNamespace(simulation_environment_running=Mock(return_value=False), ros2_start=Mock(return_value=None),
                             _shutdown_gazebo_prewarm_controllers=Mock(),
                             simulation_start_ready=Mock(return_value=(True, 'ready')))
    worker = SimpleNamespace(run=AsyncMock(return_value={'status': 'completed', 'launch_id': 'launch', 'scene_fingerprint': 'scene'}))
    source = tmp_path/'scene.json'
    source.write_text('{}')
    monkeypatch.setattr(startup, 'configuration_fingerprints', lambda _setup: {str(source): 'same'})
    result = asyncio.run(startup.prepare_delivery_start(bridge, setup, worker=worker))
    assert result['inputs'] == inputs
    bridge.ros2_start.assert_called_once_with('gazebo_dual')
    bridge.simulation_environment_running.return_value = True
    asyncio.run(startup.prepare_delivery_start(bridge, setup, worker=worker))
    assert bridge.ros2_start.call_count == 1
    stamps = iter([{str(source): 'old'}, {str(source): 'new'}])
    monkeypatch.setattr(startup, 'configuration_fingerprints', lambda _setup: next(stamps))
    with pytest.raises(ValueError, match='changed'):
        asyncio.run(startup.prepare_delivery_start(bridge, setup, worker=worker))
    assert delivery.prepared_start() is None
    report = read_json(next(delivery.RUN_DIRECTORY.glob('*/run.json')))
    assert report['outcome']['status'] == 'preparation_failed'
    assert report['plans'] == [] and report['transitions'] == []
    assert report['source_snapshots'][str(source)] == '{}'


def test_stop_during_preparation_prevents_factory_handoff(inputs, monkeypatch):
    setup = recovery_setup.default_setup()
    setup['selected_product_order_file'] = str(delivery.ORDER_PATH.relative_to(delivery.ROOT))
    bridge = SimpleNamespace(simulation_environment_running=lambda: True)
    async def probe(_request):
        delivery.request_stop()
        return {'status': 'completed'}
    monkeypatch.setattr(startup, 'configuration_fingerprints', lambda _setup: {})
    with pytest.raises(ValueError, match='Stop System'):
        asyncio.run(startup.prepare_delivery_start(bridge, setup, worker=SimpleNamespace(run=probe)))
    assert delivery.prepared_start() is None
    report = read_json(next(delivery.RUN_DIRECTORY.glob('*/run.json')))
    assert report['outcome']['status'] == 'stopped'
    assert report['outcome']['acknowledged_tasks'] == 0


def test_stop_request_blocks_a_completion_arriving_before_agent_teardown(inputs):
    actor = SimpleNamespace(agent_name='KMR', jid='kmr@localhost', configure_nominal=Mock())
    runtime = delivery.DeliveryRuntime(SimpleNamespace(), {'inputs': inputs, 'setup': {}, 'probe': {}}, [actor])
    pending = runtime.context.prepare(runtime.context.plan()['tasks'][0])
    before = runtime.context.snapshot()
    delivery.request_stop()
    assert not runtime.accept('kmr@localhost', {'task_id': pending['task_id'], 'status': 'completed',
                                              'observations': {'nominal_acknowledgement': gazebo_ack(pending)}})
    assert runtime.context.snapshot() == before
    assert runtime.context.revision == 0
    assert runtime.outcome['status'] == 'stopped'


def test_agent_handlers_authenticate_bindings_and_preserve_cca_dispatch(inputs, tmp_path, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
    from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
    from cais_spade_llm.recovery_framework.kmr_agent import KMRResourceAgent

    monkeypatch.setattr(delivery, 'RUN_DIRECTORY', tmp_path)

    async def scenario():
        actor = KMRResourceAgent('KMR@localhost', 'none', cca_jid='cca@localhost')
        resources = [actor, ResourceAgent('Storage@localhost', 'none', name='Storage'),
                     ResourceAgent('M1@localhost', 'none', name='M1')]
        product = ProductAgent('assembly_board-v1@localhost', 'none', name='assembly_board-v1',
                               resource_agents=resources, resource_jids=[str(r.jid) for r in resources],
                               product_order_file=str(delivery.ORDER_PATH))
        prepared = {'inputs': inputs, 'setup': {}, 'probe': {'launch_id': 'launch', 'scene_fingerprint': 'scene'}}
        runtime = delivery.DeliveryRuntime(product, prepared, resources)
        product.delivery_runtime = runtime
        nodes, fsa = runtime.build_plan()
        assert len(nodes) == 3 and fsa
        actor.worker.run = AsyncMock(side_effect=lambda req: {'status': 'completed', 'observations': observation(req['pending'])})
        for node in nodes:
            runtime.prepare_dispatch(node)
            with pytest.raises(ValueError):
                await actor._execute(node['function_name'], {**node['params'], 'task_id': node['id'], 'part_name': 'LG'})
            result = await actor._execute(node['function_name'], {**node['params'], 'task_id': node['id']})
            payload = {'task_id': node['id'], **result}
            before = runtime.context.snapshot()
            with pytest.raises(ValueError):
                runtime.accept('ur5e-1@localhost', payload)
            assert runtime.context.snapshot() == before
            assert runtime.accept(str(actor.jid), payload)
            assert not runtime.accept(str(actor.jid), payload)
        assert runtime.outcome['status'] == 'completed'
        assert runtime.context.resources['M1'] is resources[2].nominal_context
        report = read_json(runtime.path/'run.json')
        assert report['evidence'] == 'gazebo' and len(report['transitions']) == 3
        assert 'password' not in str(report)
        runtime.stop()
        assert runtime.outcome['status'] == 'completed'
    asyncio.run(scenario())


def test_complete_run_page_uses_existing_start_stop_without_render_dispatch(monkeypatch):
    from nicegui import context, ui
    from nicegui.client import Client
    from test_recovery_setup import _capture_buttons, _run_bridge
    from cais_spade_llm.ui.pages import recovery_run

    setup = recovery_setup.default_setup()
    setup['selected_product_order_file'] = str(delivery.ORDER_PATH.relative_to(delivery.ROOT))
    monkeypatch.setattr(recovery_setup, 'load_setup', lambda _path=None, **kwargs: deepcopy(setup))
    monkeypatch.setattr(recovery_setup, 'startup_block_reason', lambda *args, **kwargs: '')
    bridge = _run_bridge()
    bridge.simulation_environment_running.return_value = False
    bridge.simulation_start_ready.return_value = (False, 'Gazebo is absent')
    bridge.ros2_proc_status.return_value = 'stopped'
    bridge.consume_notice.return_value = ''
    async def start():
        bridge.system_running = True
    async def stop():
        bridge.system_running = False
    bridge.start_system = AsyncMock(side_effect=start)
    bridge.stop_system = AsyncMock(side_effect=stop)
    prepare = AsyncMock(return_value={})
    monkeypatch.setattr(recovery_run, 'prepare_delivery_start', prepare)
    buttons = _capture_buttons(monkeypatch)
    monkeypatch.setattr(ui, 'timer', lambda *args, **kwargs: Mock())
    client = Client(context.client.page)

    async def scenario():
        with client:
            recovery_run.render(bridge)
            bridge.start_system.assert_not_called()
            bridge.ros2_start.assert_not_called()
            prepare.assert_not_called()
            await buttons['Start System']()
            await asyncio.sleep(.05)
            prepare.assert_awaited_once()
            bridge.start_system.assert_awaited_once()
            await buttons['Start System']()
            assert bridge.start_system.await_count == 1
            await buttons['Stop System']()
            bridge.stop_system.assert_awaited_once()
        client.delete()
    asyncio.run(scenario())


def test_supported_delivery_contract_does_not_follow_edited_order_file(inputs, tmp_path, monkeypatch):
    modified = deepcopy(inputs['product_order'])
    modified['completion_conditions']['M1']['resource_state'] = 'completed'
    path = tmp_path/'edited.json'
    import json
    path.write_text(json.dumps(modified))
    monkeypatch.setattr(delivery, 'ORDER_PATH', path)
    assert not delivery.is_delivery_order(modified)
    assert delivery.is_delivery_order(inputs['product_order'])


def test_configuration_changes_after_preparation_block_dispatch(tmp_path):
    import hashlib
    path = tmp_path/'scene.json'
    path.write_text('{}')
    snapshot = {'source_fingerprints': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}}
    delivery.verify_configuration(snapshot)
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='changed after preparation'):
        delivery.verify_configuration(snapshot)


def test_gazebo_report_selection_and_refresh_are_read_only(tmp_path, monkeypatch):
    import json
    from nicegui import context, ui
    from nicegui.client import Client
    from cais_spade_llm.ui.components.gazebo_delivery_run import render_gazebo_delivery_runs
    from test_recovery_setup import _capture_buttons

    path = tmp_path/'trial'/'run.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'schema_version': 1, 'evidence': 'gazebo',
                               'outcome': {'status': 'completed'}, 'transitions': []}))
    before = path.read_bytes()
    buttons = _capture_buttons(monkeypatch)
    monkeypatch.setattr(delivery, 'prepare_start', Mock(side_effect=AssertionError('render dispatched')))
    with Client(context.client.page) as client:
        render_gazebo_delivery_runs(tmp_path)
        selector = next(e for e in client.elements.values() if isinstance(e, ui.select))
        selector.set_value(str(path))
        buttons['Refresh Gazebo reports']()
        assert any(getattr(e, 'text', '') == 'Outcome: completed' for e in client.elements.values())
        assert path.read_bytes() == before
        path.write_text('{}')
        buttons['Refresh Gazebo reports']()
        assert any('Unsupported Gazebo report format' in getattr(e, 'text', '') for e in client.elements.values())
        delivery.prepare_start.assert_not_called()
        client.delete()


def test_ros_worker_does_not_depend_on_asyncio_child_watcher(monkeypatch):
    import subprocess
    import sys
    from cais_spade_llm.recovery_framework.kmr_agent import GazeboWorker

    launch = subprocess.Popen
    def worker_process(args, **kwargs):
        code = 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(\'{"status":"completed"}\')'
        return launch([sys.executable, '-c', code, args[-1]], **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', worker_process)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', AsyncMock(side_effect=AssertionError('child watcher used')))
    assert asyncio.run(GazeboWorker().run({'mode': 'probe'})) == {'status': 'completed'}


def test_scene_identity_read_can_retry_but_effect_requests_are_not_repeated():
    import ast
    from pathlib import Path
    from cais_spade_llm.recovery_framework import kmr_gazebo

    module = ast.parse(Path(kmr_gazebo.__file__).read_text())
    run = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    service = next(node for node in run.body if isinstance(node, ast.FunctionDef) and node.name == 'service')

    def invoke(retry_read):
        unanswered = SimpleNamespace(done=lambda: False)
        answered = SimpleNamespace(done=lambda: True, result=lambda: 'observed scene')
        client = SimpleNamespace(service_is_ready=lambda: True,
                                 call_async=Mock(side_effect=[unanswered, answered]),
                                 remove_pending_request=Mock())
        def spin_until(predicate, _timeout, _label):
            if not predicate():
                raise TimeoutError('No response')
        namespace = {'clients': {}, 'node': SimpleNamespace(create_client=lambda *_: client),
                     'time': SimpleNamespace(monotonic=lambda: 0.), 'spin_until': spin_until,
                     '_log': Mock(), 'operations': []}
        exec(compile(ast.Module(body=[service], type_ignores=[]), str(kmr_gazebo.__file__), 'exec'), namespace)
        if retry_read:
            assert namespace['service'](object, '/KMR_base_controller/get_parameters', {}, retry_read=True) == 'observed scene'
            assert client.call_async.call_count == 2
            assert namespace['operations'][0]['read_attempts'] == 2
        else:
            with pytest.raises(TimeoutError):
                namespace['service'](object, '/ATTACHLINK', {})
            assert client.call_async.call_count == 1
        client.remove_pending_request.assert_called_once_with(unanswered)
    invoke(True)
    invoke(False)


def test_gripper_controller_success_waits_for_measured_width():
    import ast
    from pathlib import Path
    from cais_spade_llm.recovery_framework import kmr_gazebo

    module = ast.parse(Path(kmr_gazebo.__file__).read_text())
    run = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    gripper = next(node for node in run.body if isinstance(node, ast.FunctionDef) and node.name == 'gripper')
    for measured, succeeds in (([.08, .08, .015], True), ([.08, .08, .08], False)):
        samples = iter(measured)
        values = {'KMR_rg2_finger_width': .08}
        def fresh_state():
            values['KMR_rg2_finger_width'] = next(samples)
        def spin_until(predicate, _timeout, _label):
            for _ in range(3):
                if predicate():
                    return
            raise TimeoutError('Measured target was not reached')
        namespace = {
            'FollowJointTrajectory': SimpleNamespace(Goal=lambda: SimpleNamespace(trajectory=SimpleNamespace())),
            'JointTrajectoryPoint': lambda **kwargs: SimpleNamespace(**kwargs),
            'Duration': lambda **kwargs: SimpleNamespace(**kwargs), 'action': Mock(),
            'kmr': {'gripper_joint': 'KMR_rg2_finger_width', 'gripper_controller': '/KMR/KMR_rg2_gripper_traj_controller'},
            'fresh_state': fresh_state, 'spin_until': spin_until, 'joint_values': values,
            'joint_stamps': {'KMR_rg2_finger_width': 10.}, 'operations': [],
        }
        exec(compile(ast.Module(body=[gripper], type_ignores=[]), str(kmr_gazebo.__file__), 'exec'), namespace)
        if succeeds:
            namespace['gripper'](.015)
            assert namespace['operations'][-1]['position'] == .015
        else:
            with pytest.raises(TimeoutError):
                namespace['gripper'](.015)
            assert namespace['operations'] == []
        namespace['action'].assert_called_once()


def test_delivery_supervisor_request_stops_without_legacy_recovery(inputs, tmp_path, monkeypatch):
    import json
    from spade.message import Message
    from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
    from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
    from cais_spade_llm.recovery_framework.kmr_agent import KMRResourceAgent

    monkeypatch.setattr(delivery, 'RUN_DIRECTORY', tmp_path)
    async def scenario():
        actor = KMRResourceAgent('KMR@localhost', 'none', cca_jid='cca@localhost')
        resources = [actor, ResourceAgent('Storage@localhost', 'none', name='Storage'),
                     ResourceAgent('M1@localhost', 'none', name='M1')]
        product = ProductAgent('assembly_board-v1@localhost', 'none', name='assembly_board-v1',
                               resource_agents=resources, resource_jids=[str(r.jid) for r in resources],
                               product_order_file=str(delivery.ORDER_PATH))
        runtime = delivery.DeliveryRuntime(product, {'inputs': inputs, 'setup': {}, 'probe': {}}, resources)
        product.delivery_runtime = runtime
        runtime.prepare_dispatch(runtime.build_plan()[0][0])
        before = deepcopy(product.part_tracker)
        product._handle_runtime_des_replan_request = AsyncMock(side_effect=AssertionError('legacy recovery invoked'))
        actor.worker.cancel = AsyncMock()
        message = Message(sender='cca@localhost', body=json.dumps({'reason': 'inevitable_violation'}))
        inbox = ProductAgent._ReplanInbox()
        inbox.set_agent(product)
        assert runtime.context.pending['task_id'] != 'nominal_1'
        stale = Message(sender='cca@localhost', body=json.dumps(
            {'reason': 'inevitable_violation', 'event': {'task_id': 'nominal_1'}}))
        inbox.receive = AsyncMock(return_value=stale)
        await inbox.run()
        assert not runtime.stopped
        actor.worker.cancel.assert_not_awaited()
        inbox.receive = AsyncMock(return_value=message)
        await inbox.run()
        actor.worker.cancel.assert_awaited_once()
        assert runtime.stopped and runtime.context.revision == 0
        assert product.part_tracker == before
        assert not runtime.accept(str(actor.jid), {'task_id': 'nominal_1', 'status': 'failed:gazebo'})
        assert runtime.outcome['uncommitted_acknowledgements'][0]['status'] == 'failed:gazebo'
        product._handle_runtime_des_replan_request.assert_not_called()
    asyncio.run(scenario())

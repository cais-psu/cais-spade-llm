"""Delivery goals, authenticated Gazebo handoffs, startup, and retired settings."""

from __future__ import annotations

import asyncio
import logging
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
    for worker in delivery.take_unclaimed_prepared_workers():
        asyncio.run(worker.cancel())
    prepare(None)
    yield
    delivery.reset_stop()
    for worker in delivery.take_unclaimed_prepared_workers():
        asyncio.run(worker.cancel())
    prepare(None)


@pytest.fixture
def startup_logs(monkeypatch, caplog):
    for name in ('ui.bridge', startup.__name__):
        target = logging.getLogger(name)
        monkeypatch.setattr(target, 'handlers', [caplog.handler])
        monkeypatch.setattr(target, 'propagate', False)
        caplog.set_level(logging.INFO, logger=name)
    return caplog


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
    from cais_spade_llm.recovery_framework.kmr_tasks import KMR_TASKS, delivery_bindings

    context = NominalProductContext(**inputs)
    tasks = context.plan()['tasks']
    assert [task['event_name'] for task in tasks] == ['pick_part', 'move_to_resource', 'place_release']
    for task in tasks:
        assert task['parameters'] == delivery_bindings('KET4_Square_4mm')[task['event_name']]
        assert set(task['parameters']) == {arg.name for arg in KMR_TASKS[task['event_name']].arguments}
        assert context.acknowledge_gazebo(gazebo_ack(context.prepare(task)))
    assert context.plan()['status'] == 'completed'
    assert context.revision == 3
    values = context.snapshot()
    assert values['M1']['part_name'] == 'KET4_Square_4mm'
    assert values['M1']['resource_state'] == 'loaded'
    assert values['KMR']['held_part'] is None
    assert values['KMR']['resource_state'] == 'idle'
    assert values['KMR']['resource_location'] == 'M1'
    assert values['Storage']['inventory.KET4_Square_4mm'] is False
    assert values['M2']['part_name'] is None
    state = context.part_tracker['KET4_Square_4mm']
    assert state['location'] == 'M1' and state['state'] == 'loaded'
    assert state['processCompleted'] == []


def test_saved_setup_selects_eight_pegs_and_preserves_full_order_and_delivery_demo():
    setup = recovery_setup.load_setup()
    full_order = 'cais_spade_llm/specification/products/orders/assembly_board-v1-recovery-framework.json'
    eight_order = 'cais_spade_llm/specification/products/orders/assembly_board-v1-eight-pegs.json'
    assert setup['selected_product_order_file'] == eight_order
    assert recovery_setup.default_setup()['selected_product_order_file'] == full_order
    delivery_setup = {**setup, 'selected_product_order_file': str(delivery.ORDER_PATH.relative_to(delivery.ROOT))}
    assert startup.supports_delivery(delivery_setup)
    assert recovery_setup.startup_block_reason(delivery_setup) == ''
    inputs = recovery_setup.validate_setup(delivery_setup)
    tasks = NominalProductContext(**{
        key: inputs[key] for key in ('scene', 'product_order', 'geometry')
    }).plan()['tasks']
    assert [task['event_name'] for task in tasks] == ['pick_part', 'move_to_resource', 'place_release']
    assert {task['resource_id'] for task in tasks} == {'KMR'}
    assert recovery_setup.load_setup()['selected_product_order_file'] == eight_order


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


def test_startup_transfers_the_exact_probed_worker_once(inputs, monkeypatch, tmp_path):
    from cais_spade_llm.recovery_framework import gazebo_worker

    setup = recovery_setup.default_setup()
    setup['selected_product_order_file'] = str(delivery.ORDER_PATH.relative_to(delivery.ROOT))
    bridge = SimpleNamespace(simulation_environment_running=Mock(return_value=True))
    source = tmp_path/'scene.json'
    source.write_text('{}')
    monkeypatch.setattr(startup, 'configuration_fingerprints', lambda _setup: {str(source): 'same'})
    worker = SimpleNamespace(
        run=AsyncMock(return_value={
            'status': 'completed', 'launch_id': 'launch',
            'scene_fingerprint': 'scene', 'scene_asset_fingerprint': 'assets',
        }),
        cancel=AsyncMock(),
    )
    monkeypatch.setattr(gazebo_worker, 'GazeboWorker', lambda: worker)
    prepared = asyncio.run(startup.prepare_delivery_start(bridge, setup))
    assert prepared['prepared_resource_token']
    assert delivery.claim_prepared_worker(prepared) is worker
    assert delivery.claim_prepared_worker(prepared) is None
    worker.cancel.assert_not_awaited()


def test_dead_preparation_worker_restarts_once_but_task_execution_does_not(monkeypatch):
    from cais_spade_llm.recovery_framework.gazebo_worker import GazeboWorker

    async def exercise(mode):
        worker = GazeboWorker()
        dead = SimpleNamespace(poll=lambda: -9)

        async def exchange(_request):
            if exchange.calls == 0:
                exchange.calls += 1
                worker.process = dead
                raise RuntimeError('worker returned no evidence')
            return {'status': 'completed'}

        exchange.calls = 0
        monkeypatch.setattr(worker, '_exchange', exchange)
        if mode == 'probe':
            assert await worker.run({'mode': mode}) == {'status': 'completed'}
            assert exchange.calls == 1
        else:
            with pytest.raises(RuntimeError, match='no evidence'):
                await worker.run({'mode': mode})

    asyncio.run(exercise('probe'))
    asyncio.run(exercise('task'))


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


@pytest.mark.parametrize('reason', [
    'Simulation startup is not done yet. Waiting for core services: /compute_cartesian_path',
    'Simulation readiness query failed or timed out after 10 seconds. Retrying ROS service discovery.',
])
def test_delivery_start_rechecks_transient_discovery_without_relaunch(monkeypatch, startup_logs, reason):
    prepared = {'source_fingerprints': {}, 'probe': {'launch_id': 'launch', 'scene_fingerprint': 'scene'}}
    refreshed = deepcopy(prepared)
    refreshed['probe']['timings'] = {'wall_time_sec': 2.}
    prepare = AsyncMock(return_value=refreshed)
    monkeypatch.setattr(startup, 'prepare_delivery_start', prepare)
    monkeypatch.setattr(startup.asyncio, 'sleep', AsyncMock())
    bridge = SimpleNamespace(system_running=False, last_error=reason,
                             simulation_environment_running=Mock(return_value=True))
    async def start():
        bridge.system_running = bridge.start_system.await_count == 2
        if not bridge.system_running:
            try:
                raise RuntimeError(reason)
            except RuntimeError:
                logging.getLogger('ui.bridge').exception('Failed to start system')
    bridge.start_system = AsyncMock(side_effect=start)
    progress = Mock()
    asyncio.run(startup.start_delivery_agents(bridge, {}, prepared, on_progress=progress))
    assert bridge.system_running
    assert bridge.start_system.await_count == 2
    prepare.assert_not_awaited()
    bridge.simulation_environment_running.assert_called_once()
    progress.assert_called_once()
    assert reason in progress.call_args.args[0]
    assert 'Waiting for simulation service discovery' in startup_logs.text
    assert '(2/3)' in startup_logs.text
    assert 'Failed to start system' not in startup_logs.text
    assert 'Traceback' not in startup_logs.text
    assert not any(record.levelno >= logging.ERROR for record in startup_logs.records)


@pytest.mark.parametrize('condition', ['failure', 'running', 'absent', 'persistent', 'stop'])
def test_delivery_start_retry_is_bounded_and_preserves_stop(monkeypatch, startup_logs, condition):
    prepared = {'source_fingerprints': {}, 'probe': {'launch_id': 'launch'}}
    prepare = AsyncMock(return_value=prepared)
    monkeypatch.setattr(startup, 'prepare_delivery_start', prepare)
    monkeypatch.setattr(startup.asyncio, 'sleep', AsyncMock())
    bridge = SimpleNamespace(system_running=False,
                             last_error='Simulation startup is not done yet. Waiting for core services: /attach',
                             simulation_environment_running=Mock(return_value=condition != 'absent'))
    async def start():
        if condition == 'failure':
            bridge.last_error = 'Agent authentication failed'
        elif condition == 'running':
            bridge.system_running = True
        elif condition == 'stop':
            delivery.request_stop()
        if not bridge.system_running:
            try:
                raise RuntimeError(bridge.last_error)
            except RuntimeError:
                logging.getLogger('ui.bridge').exception('Failed to start system')
    bridge.start_system = AsyncMock(side_effect=start)
    if condition == 'stop':
        with pytest.raises(ValueError, match='Stop System'):
            asyncio.run(startup.start_delivery_agents(bridge, {}, prepared))
    else:
        asyncio.run(startup.start_delivery_agents(bridge, {}, prepared))
    assert bridge.start_system.await_count == (3 if condition == 'persistent' else 1)
    assert prepare.await_count == 0
    failures = [record for record in startup_logs.records if record.getMessage() == 'Failed to start system']
    assert len(failures) == (0 if condition == 'running' else 1)
    if failures:
        assert failures[0].levelno == logging.ERROR
        assert str(failures[0].exc_info[1]) == bridge.last_error


def test_delivery_readiness_retry_preserves_errors_from_other_requests(monkeypatch, startup_logs):
    reason = 'Simulation startup is not done yet. Waiting for core services: /compute_cartesian_path'
    prepared = {'source_fingerprints': {}, 'probe': {'launch_id': 'launch'}}
    monkeypatch.setattr(startup, 'prepare_delivery_start', AsyncMock(return_value=prepared))
    bridge = SimpleNamespace(system_running=False, last_error=reason,
                             simulation_environment_running=lambda: True)
    bridge_logger = logging.getLogger('ui.bridge')
    original_filters = list(bridge_logger.filters)
    unrelated_error = RuntimeError(reason)

    async def scenario():
        entered, released = asyncio.Event(), asyncio.Event()

        async def other_request():
            await entered.wait()
            try:
                raise unrelated_error
            except RuntimeError:
                bridge_logger.exception('Failed to start system')
            released.set()

        async def start():
            if bridge.start_system.await_count == 1:
                try:
                    raise RuntimeError(reason)
                except RuntimeError:
                    bridge_logger.exception('Failed to start system')
                entered.set()
                await released.wait()
            else:
                bridge.system_running = True

        bridge.start_system = AsyncMock(side_effect=start)
        other = asyncio.create_task(other_request())
        await startup.start_delivery_agents(bridge, {}, prepared)
        await other

    asyncio.run(scenario())
    failures = [record for record in startup_logs.records if record.getMessage() == 'Failed to start system']
    assert len(failures) == 1 and failures[0].exc_info[1] is unrelated_error
    assert bridge_logger.filters == original_filters
    assert bridge.system_running


def test_delivery_start_rejects_configuration_change_without_reprobing(monkeypatch):
    prepared = {'source_fingerprints': {}, 'probe': {
        'launch_id': 'launch', 'scene_fingerprint': 'scene', 'scene_asset_fingerprint': 'assets',
    }}
    prepare = AsyncMock()
    monkeypatch.setattr(startup, 'prepare_delivery_start', prepare)
    monkeypatch.setattr(startup.asyncio, 'sleep', AsyncMock())
    verify = Mock(side_effect=[None, ValueError('Configuration changed after preparation')])
    monkeypatch.setattr(startup, 'verify_configuration', verify)
    delivery.prepare_start(prepared)
    bridge = SimpleNamespace(system_running=False, start_system=AsyncMock(),
                             last_error='Simulation readiness query failed or timed out',
                             simulation_environment_running=lambda: True)
    with pytest.raises(ValueError, match='Configuration changed'):
        asyncio.run(startup.start_delivery_agents(bridge, {}, prepared))
    bridge.start_system.assert_awaited_once()
    prepare.assert_not_awaited()
    assert delivery.prepared_start() is None


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
        from cais_spade_llm.ui.bridge import SystemBridge
        from cais_spade_llm.ui.resource_status import ResourceStatusReader

        bridge = SimpleNamespace(
            get_robot_states=lambda: SystemBridge.get_robot_states(SimpleNamespace(resource_agents=resources)),
            load_config=lambda _path: deepcopy(inputs['scene']),
        )
        status = ResourceStatusReader(bridge).read()
        display = status['resources']
        assert set(display) == {'KMR', 'Storage', 'M1'}
        assert display['M1']['state']['resource_state'] == 'loaded'
        assert display['Storage']['state']['inventory.KET4_Square_4mm'] is False
        assert display['KMR']['state']['resource_location'] == 'M1'
        assert status['outcome']['status'] == 'completed'
        assert display['KMR']['evidence'] == 'Last Gazebo acknowledgement: place_release; revision 3; launch launch.'
        report = read_json(runtime.path/'run.json')
        assert report['evidence'] == 'gazebo' and len(report['transitions']) == 3
        assert 'password' not in str(report)
        runtime.stop()
        assert runtime.outcome['status'] == 'completed'
    asyncio.run(scenario())


@pytest.mark.parametrize('preparation_failure', [False, True])
@pytest.mark.parametrize('gazebo_running', [False, True])
@pytest.mark.parametrize('discovery_retry', [False, True])
def test_complete_run_page_uses_existing_start_stop_without_render_dispatch(
    monkeypatch, preparation_failure, gazebo_running, discovery_retry
):
    from nicegui import context, ui
    from nicegui.client import Client
    from test_recovery_setup import _capture_buttons, _run_bridge
    from cais_spade_llm.ui.pages import recovery_run

    setup = recovery_setup.default_setup()
    setup['selected_product_order_file'] = str(delivery.ORDER_PATH.relative_to(delivery.ROOT))
    monkeypatch.setattr(recovery_setup, 'load_setup', lambda _path=None, **kwargs: deepcopy(setup))
    monkeypatch.setattr(recovery_setup, 'startup_block_reason', lambda *args, **kwargs: '')
    bridge = _run_bridge()
    bridge.simulation_environment_running.return_value = gazebo_running
    bridge.simulation_start_ready.return_value = (
        gazebo_running, 'Perception is still warming up: /detect_all. Start is allowed.'
        if gazebo_running else 'Gazebo is absent',
    )
    bridge.ros2_proc_status.return_value = 'running' if gazebo_running else 'stopped'
    bridge.consume_notice.return_value = ''
    async def start():
        bridge.system_running = not discovery_retry or bridge.start_system.await_count == 2
        bridge.last_error = None
        if not bridge.system_running:
            bridge.simulation_environment_running.return_value = True
            bridge.last_error = 'Simulation startup is not done yet. Waiting for core services: /compute_cartesian_path'
            try:
                raise RuntimeError(bridge.last_error)
            except RuntimeError:
                logging.getLogger('ui.bridge').exception('Failed to start system')
    async def stop():
        bridge.system_running = False
    bridge.start_system = AsyncMock(side_effect=start)
    bridge.stop_system = AsyncMock(side_effect=stop)
    prepared = {'source_fingerprints': {}, 'probe': {'launch_id': 'launch'}}
    prepare = AsyncMock(return_value=prepared)
    if preparation_failure:
        prepare.side_effect = RuntimeError('KMR initial observations unavailable')
    monkeypatch.setattr(recovery_run, 'prepare_delivery_start', prepare)
    buttons = _capture_buttons(monkeypatch)
    monkeypatch.setattr(ui, 'timer', lambda *args, **kwargs: Mock())
    client = Client(context.client.page)

    async def retry_preparation(*_args):
        texts = [getattr(element, 'text', '') for element in client.elements.values()]
        assert any('Waiting for simulation service discovery' in text for text in texts)
        assert not any('Start failed:' in text for text in texts)
        await buttons['Start System']()
        assert bridge.start_system.await_count == 1
        return prepared

    retry_prepare = AsyncMock(side_effect=retry_preparation)
    monkeypatch.setattr(startup, 'prepare_delivery_start', retry_prepare)

    async def scenario():
        with client:
            recovery_run.render(bridge)
            await buttons['Refresh saved setup']()
            texts = [getattr(element, 'text', '') for element in client.elements.values()]
            assert not any('Perception is still warming up:' in text for text in texts)
            bridge.start_system.assert_not_called()
            bridge.ros2_start.assert_not_called()
            prepare.assert_not_called()
            await buttons['Start System']()
            for _ in range(200):
                await asyncio.sleep(.01)
                texts = [getattr(element, 'text', '') for element in client.elements.values()]
                if 'Agents started.' in texts or any('Start failed:' in text for text in texts):
                    break
            prepare.assert_awaited_once()
            assert 'Live Resource Status' in texts
            assert any('Destination: M1' in text and 'KET4_Square_4mm' in text for text in texts)
            assert 'System started successfully.' not in texts
            if preparation_failure:
                bridge.start_system.assert_not_called()
                assert any('Start failed: KMR initial observations unavailable' in text for text in texts)
                return
            assert bridge.start_system.await_count == (2 if discovery_retry else 1)
            assert retry_prepare.await_count == 0
            assert 'Agents started.' in texts
            await buttons['Start System']()
            assert bridge.start_system.await_count == (2 if discovery_retry else 1)
            await buttons['Stop System']()
            bridge.stop_system.assert_awaited_once()
    asyncio.run(scenario())
    client.delete()


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
        code = '''import json,pathlib,sys,time
directory=pathlib.Path(sys.argv[1])
previous=None
while True:
    request=directory/'request.json'
    if request.exists():
        envelope=json.loads(request.read_text())
        if envelope['id'] != previous:
            previous=envelope['id']
            temporary=directory/'result.tmp'
            temporary.write_text(json.dumps({'id':previous,'result':{'status':'completed'}}))
            temporary.replace(directory/'result.json')
    time.sleep(.01)
'''
        return launch([sys.executable, '-c', code, args[-1]], **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', worker_process)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', AsyncMock(side_effect=AssertionError('child watcher used')))
    async def scenario():
        worker = GazeboWorker()
        try:
            assert await worker.run({'mode': 'probe'}) == {'status': 'completed'}
            pid = worker.process.pid
            assert await worker.run({'mode': 'probe'}) == {'status': 'completed'}
            assert worker.process.pid == pid
        finally:
            await worker.cancel()
        assert worker.process is None
    asyncio.run(scenario())


@pytest.mark.parametrize('response', ['delayed', 'missing', 'rejected'])
def test_collision_scene_update_retries_only_identical_idempotent_request(response):
    import ast
    from pathlib import Path
    from cais_spade_llm.recovery_framework import kmr_gazebo

    tree = ast.parse(Path(kmr_gazebo.__file__).read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'apply')
    answers = ([TimeoutError('DDS response missing'), SimpleNamespace(success=True)]
               if response == 'delayed' else [TimeoutError('DDS response missing')]*3
               if response == 'missing' else [SimpleNamespace(success=False)])
    service = Mock(side_effect=answers)
    operations = []
    kind = SimpleNamespace(Request=lambda **kwargs: SimpleNamespace(**kwargs))
    namespace = {'service': service, 'ApplyPlanningScene': kind,
                 'config': {'services': {'apply_scene': '/apply_planning_scene'}},
                 '_log': Mock(), 'operations': operations}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(kmr_gazebo.__file__), 'exec'), namespace)
    scene = object()
    if response == 'delayed':
        namespace['apply'](scene)
        assert service.call_count == 2
        first, second = service.call_args_list
        assert first.args[2] is second.args[2]
        assert first.args[2].scene is scene
        assert operations[-1]['scene_update_attempts'] == 2
    else:
        with pytest.raises(TimeoutError if response == 'missing' else RuntimeError):
            namespace['apply'](scene)
        assert service.call_count == (3 if response == 'missing' else 1)
        assert operations == []


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


@pytest.mark.parametrize('values', [[], ['launch'], ['launch', 'scene'],
                                   ['', 'scene', 'assets'], ['launch', '', 'assets'],
                                   ['launch', 'scene', ''], ['launch', 'scene', 'assets', 'extra']])
def test_incomplete_controller_identity_reports_required_gazebo_refresh(values):
    from cais_spade_llm.recovery_framework.kmr_gazebo import _scene_identity

    with pytest.raises(ValueError) as error:
        _scene_identity(values)
    assert '/KMR_base_controller' in str(error.value)
    assert 'launch_id, scene_fingerprint, scene_asset_fingerprint' in str(error.value)
    assert 'make bootstrap-gazebo' in str(error.value)
    assert 'Reset Gazebo before Start System' in str(error.value)


def test_controller_identity_keeps_observed_values():
    from cais_spade_llm.recovery_framework.kmr_gazebo import _scene_identity
    from cais_spade_llm.recovery_framework.simulation import DEFAULT_SETTINGS
    import json

    performance = json.dumps(DEFAULT_SETTINGS)
    assert _scene_identity(['launch-1', 'scene-2', 'assets-3', performance]) == {
        'launch_id': 'launch-1', 'scene_fingerprint': 'scene-2',
        'scene_asset_fingerprint': 'assets-3', 'performance_settings': performance,
    }


def test_delivery_motion_uses_model_limits_and_low_bounded_carrying_pose(inputs):
    from cais_spade_llm.recovery_framework.kmr_motion import bounded_joints, joint_limits

    kmr = inputs['scene']['KMR']
    config = kmr['task_execution']
    limits = joint_limits(ROOT/'ros2/cais_lab_robotics/urdf/KMR_recovery.urdf.xacro',
                          kmr['arm_joint_names'], config['joint_acceleration_limits'])
    assert config['velocity_scaling'] == config['acceleration_scaling'] == 1.0
    assert bounded_joints(kmr['arm_joint_names'], config['carrying_arm_configuration'], limits)
    assert config['carrying_arm_configuration'] != kmr['parked_arm_configuration']
    invalid = list(config['carrying_arm_configuration'])
    invalid[0] += 2 * 3.141592653589793
    assert not bounded_joints(kmr['arm_joint_names'], invalid, limits)
    assert 'pick_approach_configuration' not in config


def test_trajectory_timing_respects_velocity_acceleration_and_duration_order():
    from cais_spade_llm.recovery_framework.kmr_motion import retime_trajectory, seconds, trajectory_cost

    def path(end, duration, velocity, acceleration):
        return SimpleNamespace(joint_names=['joint_a1'], points=[
            SimpleNamespace(positions=[0.0], velocities=[0.0], accelerations=[0.0],
                            time_from_start=SimpleNamespace(sec=0, nanosec=0)),
            SimpleNamespace(positions=[end], velocities=[velocity], accelerations=[acceleration],
                            time_from_start=SimpleNamespace(sec=duration, nanosec=0)),
        ])
    limits = {'joint_a1': {'lower': -2.9, 'upper': 2.9, 'velocity': 1.0, 'acceleration': 2.0}}
    trajectory = path(1.0, 1, 2.0, 8.0)
    retime_trajectory(trajectory, limits, 1.0, 1.0)
    assert seconds(trajectory.points[-1].time_from_start) == 2.0
    assert trajectory.points[-1].velocities == [1.0]
    assert trajectory.points[-1].accelerations == [2.0]
    slow = path(.1, 3, 0., 0.)
    short = path(.5, 2, 0., 0.)
    assert min([slow, trajectory, short], key=lambda t: trajectory_cost([t])) is short
    with pytest.raises(ValueError, match='bounded joint'):
        retime_trajectory(path(6.0, 1, 0., 0.), limits, 1.0, 1.0)
    with pytest.raises(ValueError, match='scaling'):
        retime_trajectory(path(.5, 1, 0., 0.), limits, 10.0, 1.0)


def test_worker_preserves_base_abort_reason_and_docking_evidence():
    import ast
    from pathlib import Path
    from cais_spade_llm.recovery_framework import kmr_gazebo

    module = ast.parse(Path(kmr_gazebo.__file__).read_text())
    run = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    action = next(n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == 'action')
    action.body[0] = ast.copy_location(ast.Global(names=['first_motion_at_unix']), action.body[0])
    failure = 'KMR stopping footprint intersects an obstacle'
    wrapped = SimpleNamespace(status=6, result=SimpleNamespace(success=False, message=failure))
    def done(value):
        return SimpleNamespace(done=lambda: True, result=lambda: value)
    samples = [(0., {'measured_velocity': [99., 0., 0.]})]
    sample = {'measured_velocity': [1., 0., 0.], 'odometry_age_sec': .02}
    def result():
        samples.append((1.1, sample))
        return done(wrapped)
    handle = SimpleNamespace(accepted=True, get_result_async=result)
    client = SimpleNamespace(server_is_ready=lambda: True, send_goal_async=lambda *a, **kw: done(handle))
    kind = object()
    operations = []
    namespace = {
        'action_clients': {}, 'ActionClient': lambda *args: client, 'node': SimpleNamespace(
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000))),
        'spin_until': lambda *args, **kwargs: None, 'UUID': lambda **kw: SimpleNamespace(**kw),
        'uuid4': lambda: SimpleNamespace(bytes=bytes(16)),
        'time': SimpleNamespace(monotonic=lambda: 1., time=lambda: 1.),
        'first_motion_at_unix': None,
        'pending_goals': [], 'active_goals': [], 'operations': operations,
        'GoalStatus': SimpleNamespace(STATUS_SUCCEEDED=4), 'DockKMR': kind,
        'deepcopy': deepcopy, 'base_motion_status': {'measured_velocity': [1., 0., 0.]},
        'base_motion_samples': samples,
    }
    exec(compile(ast.Module(body=[action], type_ignores=[]), str(kmr_gazebo.__file__), 'exec'), namespace)
    with pytest.raises(RuntimeError, match=failure):
        namespace['action'](kind, '/KMR/dock', object())
    assert operations[-1]['message'] == failure
    assert operations[-1]['base_motion_status']['measured_velocity'] == [1., 0., 0.]
    assert len(operations[-1]['base_motion_samples']) == 1
    assert operations[-1]['base_motion_samples'][0]['received_after_action_start_sec'] == pytest.approx(.1)
    sample['measured_velocity'][0] = 2.
    assert operations[-1]['base_motion_samples'][0]['measured_velocity'] == [1., 0., 0.]
    assert namespace['active_goals'] == []


@pytest.mark.parametrize('feedback', ['delayed', 'stale_start', 'wrong_endpoint'])
def test_arm_completion_requires_measured_endpoint_without_weakening_start_guard(feedback):
    import ast
    from pathlib import Path
    from cais_spade_llm.recovery_framework import kmr_gazebo

    tree = ast.parse(Path(kmr_gazebo.__file__).read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'execute_plan')
    trajectory = SimpleNamespace(joint_trajectory=SimpleNamespace(
        joint_names=['joint_a1'], points=[SimpleNamespace(positions=[0.]), SimpleNamespace(positions=[.5])],
    ))
    samples = iter([.1 if feedback == 'stale_start' else 0., .4, .48,
                    .5 if feedback == 'delayed' else .48])
    def fresh_state(**kwargs):
        return SimpleNamespace(name=['joint_a1'], position=[next(samples)])
    def spin_until(predicate, timeout, label):
        for _ in range(3):
            if predicate():
                return
        raise TimeoutError(label)
    operations = []
    execute = Mock()
    namespace = {'fresh_state': fresh_state, 'execute': execute, 'operations': operations,
                 'deepcopy': deepcopy, 'spin_until': spin_until, 'joint_stamps': {'joint_a1': 2.}}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(kmr_gazebo.__file__), 'exec'), namespace)
    plan = (trajectory, {'operation': 'cartesian_path', 'target': None})
    if feedback == 'stale_start':
        with pytest.raises(ValueError, match='stale trajectory'):
            namespace['execute_plan'](plan)
        execute.assert_not_called()
        assert operations == []
    elif feedback == 'wrong_endpoint':
        with pytest.raises(TimeoutError, match='joint feedback at trajectory endpoint'):
            namespace['execute_plan'](plan)
        execute.assert_called_once_with(trajectory)
        assert len(operations) == 1
    else:
        namespace['execute_plan'](plan)
        execute.assert_called_once_with(trajectory)
        assert operations[-1]['operation'] == 'observed_arm_endpoint'
        assert operations[-1]['joints'] == {'joint_a1': .5}
        assert operations[-1]['stamps'] == {'joint_a1': 2.}


def test_cartesian_segment_requires_complete_collision_check_and_copies_target():
    import ast
    from pathlib import Path
    from cais_spade_llm.recovery_framework import kmr_gazebo

    tree = ast.parse(Path(kmr_gazebo.__file__).read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'plan_motion')
    method.body[0] = ast.copy_location(ast.Global(names=['planning_seconds']), method.body[0])
    point = SimpleNamespace(positions=[0.0], velocities=[0.0], accelerations=[0.0],
                            time_from_start=SimpleNamespace(sec=0, nanosec=0))
    response = SimpleNamespace(error_code=SimpleNamespace(val=1), fraction=1.0,
                               solution=SimpleNamespace(joint_trajectory=SimpleNamespace(
                                   joint_names=['joint_a1'], points=[point])))
    query = SimpleNamespace(header=SimpleNamespace())
    service = Mock(return_value=response)
    namespace = {
        'planning_seconds': 0.0, 'time': SimpleNamespace(monotonic=lambda: 1.0),
        'GetCartesianPath': SimpleNamespace(Request=lambda: query), 'service': service,
        'config': {'planning_group': 'KMR_iiwa_arm', 'tcp_link': 'KMR_tcp',
                   'services': {'cartesian_path': '/compute_cartesian_path'},
                   'velocity_scaling': 1.0, 'acceleration_scaling': 1.0},
        'limits': {}, 'retime_trajectory': Mock(), 'deepcopy': deepcopy,
        'pose_message': lambda target: deepcopy(target),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(kmr_gazebo.__file__), 'exec'), namespace)
    target = [-8.95, 1.88, 1.19, 0.0, 0.0, 0.0, 1.0]
    expected = target.copy()
    state = {'observed_joints': [0.0]}
    _, evidence = namespace['plan_motion'](state, target=target, cartesian=True)
    assert query.avoid_collisions is True and query.waypoints == [expected]
    assert query.start_state == state and query.start_state is not state
    assert query.revolute_jump_threshold > 0
    target[2] += .2
    assert evidence['target'] == expected
    response.fraction = .8
    with pytest.raises(ValueError, match='Cartesian path incomplete'):
        namespace['plan_motion'](state, target=target, cartesian=True)
    assert namespace['retime_trajectory'].call_count == 1


def test_transport_sweep_rejects_attached_payload_collision_before_dispatch(inputs):
    import ast
    import math
    from pathlib import Path
    from cais_spade_llm.recovery_framework import kmr_gazebo

    tree = ast.parse(Path(kmr_gazebo.__file__).read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'validate_transport')
    method.body[0] = ast.copy_location(ast.Global(names=['planning_seconds']), method.body[0])
    queries = []
    def validity(_kind, _endpoint, query, *, retry_read):
        assert retry_read is True
        queries.append(query)
        assert query.robot_state['attached_part'] == 'KET4_Square_4mm'
        return SimpleNamespace(valid=len(queries) < 3, contacts=[SimpleNamespace(
            contact_body_1='KET4_Square_4mm', contact_body_2='Storage')])
    kmr = inputs['scene']['KMR']
    namespace = {
        'planning_seconds': 0.0, 'time': SimpleNamespace(monotonic=lambda: 1.0),
        'kmr': kmr, 'config': kmr['task_execution'], 'math': math, 'deepcopy': deepcopy,
        'GetStateValidity': SimpleNamespace(Request=lambda **kw: SimpleNamespace(**kw)),
        'updated_state': lambda state, names, positions: {**state, **dict(zip(names, positions))},
        'route_between': lambda _source, _target: [[0., 0., 0.], [.1, 0., 0.]],
        'source_resource': 'Storage', 'target_resource': 'M1',
        'service': validity,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(kmr_gazebo.__file__), 'exec'), namespace)
    with pytest.raises(ValueError, match='KET4_Square_4mm.*Storage'):
        namespace['validate_transport']({'attached_part': 'KET4_Square_4mm'})
    assert len(queries) == 3
    assert queries[0].robot_state['KMR_base_x_joint'] < queries[-1].robot_state['KMR_base_x_joint']


def test_transport_heartbeat_stops_when_observed_part_leaves_attachment():
    import ast
    import json
    import math
    from pathlib import Path
    from cais_spade_llm.recovery_framework import kmr_gazebo

    tree = ast.parse(Path(kmr_gazebo.__file__).read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'publish_custody')
    method.body[0] = ast.copy_location(ast.Global(names=['custody_future']), method.body[0])
    initial = [0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0]
    displaced = initial.copy()
    displaced[0] += .02
    response = SimpleNamespace(success=True, state=SimpleNamespace(pose=displaced))
    publisher = Mock()
    timer = Mock()
    namespace = {
        'custody_future': SimpleNamespace(done=lambda: True, result=lambda: response),
        'pose_values': lambda value: value, 'math': math, 'json': json,
        'expected_relative': initial, 'custody_evidence': {'checked_samples': 1, 'success': True},
        'probe': {'launch_id': 'current', 'scene_fingerprint': 'scene'}, 'part': 'KET4_Square_4mm',
        'custody_id': 'current-custody', 'transport_posture': {},
        'custody_client': Mock(), 'custody_query': object(), 'transport_timer': timer,
        'transport': publisher, 'String': lambda **kw: SimpleNamespace(**kw),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(kmr_gazebo.__file__), 'exec'), namespace)
    namespace['publish_custody']()
    payload = json.loads(publisher.publish.call_args.args[0].data)
    assert payload['attached'] is False and payload['error']
    assert payload['custody_id'] == 'current-custody'
    assert namespace['custody_evidence']['success'] is False
    assert namespace['custody_evidence']['last_relative_pose'] == displaced
    timer.cancel.assert_called_once()
    namespace['custody_client'].call_async.assert_not_called()


@pytest.mark.parametrize('acknowledged', [False, True])
def test_docking_waits_for_matching_guarded_custody_acknowledgement(acknowledged):
    import ast
    import json
    from copy import deepcopy
    from pathlib import Path
    from cais_spade_llm.recovery_framework import kmr_gazebo

    tree = ast.parse(Path(kmr_gazebo.__file__).read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'dock')
    method.body[0] = ast.copy_location(ast.Global(names=['transport_timer']), method.body[0])
    response = SimpleNamespace(success=True, state=SimpleNamespace(pose=[0.] * 7))
    status = {'transport_custody_ack': {'custody_id': 'previous', 'arm_parked': True}}
    dispatch = Mock()

    def spin_until(predicate, timeout):
        assert timeout == 2.
        assert not predicate()
        status['transport_custody_ack'] = {'custody_id': 'current', 'arm_parked': False}
        assert not predicate()
        dispatch.assert_not_called()
        if not acknowledged:
            raise TimeoutError('Controller custody acknowledgement unavailable')
        status['transport_custody_ack']['arm_parked'] = True
        assert predicate()

    namespace = {
        'custody': Mock(), 'part': 'KET4_Square_4mm', 'pose_values': lambda value: value,
        'GetEntityState': SimpleNamespace(Request=SimpleNamespace),
        'config': {'attach_link': 'iiwa_link_7', 'services': {'get_state': '/get_state'}},
        'clients': {'/get_state': Mock()}, 'service': Mock(return_value=response),
        'operations': [], 'uuid4': lambda: SimpleNamespace(hex='current'),
        'request': {'custody': {'carrying_arm_configuration': [0.] * 7}},
        'validate_transport': Mock(return_value={'operation': 'validated_transport_sweep', 'success': True}),
        'observed_state': Mock(),
        'node': Mock(), 'Clock': Mock(), 'ClockType': SimpleNamespace(STEADY_TIME=1),
        'transport': Mock(), 'String': SimpleNamespace, 'json': json,
        'probe': {'launch_id': 'scene'}, 'spin_until': spin_until,
        'base_motion_status': status, 'deepcopy': deepcopy,
        'action': dispatch, 'DockKMR': SimpleNamespace(Goal=SimpleNamespace),
        'prepare_place_turn': Mock(),
        'kmr': {'docking_action': '/KMR/dock'},
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(kmr_gazebo.__file__), 'exec'), namespace)
    if acknowledged:
        namespace['dock']('M1', [])
        dispatch.assert_called_once()
        evidence = next(row['controller_acknowledgement'] for row in namespace['operations']
                        if row['operation'] == 'observed_transport_custody')
        status['transport_custody_ack']['arm_parked'] = False
        assert evidence == {'custody_id': 'current', 'arm_parked': True}
    else:
        with pytest.raises(TimeoutError, match='custody acknowledgement'):
            namespace['dock']('M1', [])
        dispatch.assert_not_called()
    sent = json.loads(namespace['transport'].publish.call_args.args[0].data)
    assert sent['custody_id'] == 'current' and sent['attached'] is True


def test_gripper_controller_success_waits_for_measured_width():
    import ast
    from pathlib import Path
    from cais_spade_llm.recovery_framework import kmr_gazebo

    module = ast.parse(Path(kmr_gazebo.__file__).read_text())
    run = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    gripper = next(node for node in run.body if isinstance(node, ast.FunctionDef) and node.name == 'gripper')
    for measured, succeeds in (([.08, .08, .015], True), ([.08, .08, .08, .08], False)):
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
            'joint_stamps': {'KMR_rg2_finger_width': 10.}, 'joint_samples': {},
            'active_goals': [], 'pending_goals': [], 'time': SimpleNamespace(monotonic=lambda: 10.),
            'operations': [],
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


def test_kmr_gripper_skips_a_redundant_command_only_with_stable_fresh_feedback():
    import ast
    from pathlib import Path
    from cais_spade_llm.recovery_framework import kmr_gazebo

    module = ast.parse(Path(kmr_gazebo.__file__).read_text())
    run = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    gripper = next(node for node in run.body if isinstance(node, ast.FunctionDef) and node.name == 'gripper')
    action = Mock()
    operations = []
    namespace = {
        'fresh_state': Mock(),
        'joint_samples': {'finger': [(9.8, 9.8, .015), (9.9, 9.9, .015)]},
        'active_goals': [], 'pending_goals': [],
        'time': SimpleNamespace(monotonic=lambda: 10.),
        'kmr': {'gripper_joint': 'finger', 'gripper_controller': '/gripper'},
        'operations': operations, 'action': action,
        'FollowJointTrajectory': SimpleNamespace(), 'JointTrajectoryPoint': Mock(),
        'Duration': Mock(), 'spin_until': Mock(), 'joint_values': {}, 'joint_stamps': {},
    }
    exec(compile(ast.Module(body=[gripper], type_ignores=[]), str(kmr_gazebo.__file__), 'exec'), namespace)
    namespace['gripper'](.015)
    action.assert_not_called()
    assert operations == [{
        'operation': 'observed_gripper', 'success': True, 'command_sent': False,
        'reason': 'fresh stable endpoint already observed', 'target': .015,
        'position': .015, 'stamp': 9.9,
    }]


def test_placement_target_computation_does_not_command_motion_and_scene_planning_is_serialized():
    import ast
    from pathlib import Path
    from cais_spade_llm.recovery_framework import kmr_gazebo
    from cais_spade_llm.recovery_framework.planning_permit import scene_planning_permit

    module = ast.parse(Path(kmr_gazebo.__file__).read_text())
    prepare = next(
        node for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == 'compute_place_targets'
    )
    called_names = {
        node.func.id for node in ast.walk(prepare)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {'action', 'execute', 'execute_plan'}.isdisjoint(called_names)
    with scene_planning_permit('one-scene') as first:
        with scene_planning_permit('one-scene') as second:
            assert first is True and second is False


def test_ur_background_preparation_requires_the_validated_scene_setting():
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
        GazeboPickPlaceController,
    )

    controller = object.__new__(GazeboPickPlaceController)
    controller.execution_mode = 'simulation'
    controller.controller_config = {'background_preparation_enabled': False}
    controller._queued_motion_preparation = None
    controller._attached_model = None
    controller._preparation_generation = 0
    assert not controller.queue_next_motion_preparation(
        'move_cartesian', {'x': 1., 'y': 2., 'z': 3.},
    )
    assert controller._queued_motion_preparation is None
    controller.controller_config['background_preparation_enabled'] = True
    assert controller.queue_next_motion_preparation(
        'move_cartesian', {'x': 1., 'y': 2., 'z': 3.},
    )
    assert controller._queued_motion_preparation['params'] == {'x': 1., 'y': 2., 'z': 3.}


def test_ur_named_pose_reuses_a_fresh_stable_observation_without_dispatch():
    import threading
    import time
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
        GazeboPickPlaceController,
    )

    controller = object.__new__(GazeboPickPlaceController)
    controller.execution_mode = 'simulation'
    controller._simulation_goal = None
    controller.named_positions = {'home': [.1, -.2]}
    controller.arm_joint_names = ['joint_1', 'joint_2']
    controller._joint_lock = threading.Lock()
    controller._joint_positions = {'joint_1': .1, 'joint_2': -.2}
    now = time.monotonic()
    controller._joint_received_times = {'joint_1': now, 'joint_2': now}
    controller._joint_stable_since = {'joint_1': now - 1., 'joint_2': now - 1.}
    controller.wait_for_services = Mock(return_value=True)
    controller._publish_arm_joint_trajectory_and_wait = Mock()
    controller._exec_client = None
    result = controller.move_to_named_pose('home')
    assert result['success'] is True
    assert result['command_sent'] is False
    controller._publish_arm_joint_trajectory_and_wait.assert_not_called()


def test_persistent_worker_rejects_a_reset_clock_before_reusing_observations():
    import ast
    from pathlib import Path
    from cais_spade_llm.recovery_framework import kmr_gazebo

    tree = ast.parse(Path(kmr_gazebo.__file__).read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'check_clock')
    session = {'last_clock': 10.}
    samples = {'joint_a1': [(10., 5., .2)]}
    node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0)))
    method.body[0] = ast.copy_location(ast.Global(names=['stopped']), method.body[0])
    namespace = {'session': session, 'joint_samples': samples, 'node': node,
                 'os': SimpleNamespace(getppid=lambda: 7), 'owner_pid': 7, 'stopped': False}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(kmr_gazebo.__file__), 'exec'), namespace)
    with pytest.raises(RuntimeError, match='clock reset'):
        namespace['check_clock']()
    assert samples == {} and session['clock_reset'] is True
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=20_000_000_000))
    with pytest.raises(RuntimeError, match='clock reset'):
        namespace['check_clock']()


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


def test_latest_reports_reject_stale_writers_and_archive_exact_run(tmp_path):
    from cais_spade_llm.recovery_framework.reports import LatestReport, archive_latest

    first = LatestReport(tmp_path)
    assert first.save({'outcome': {'status': 'running'}}, {'pick_part': {'name': 'pick_part'}})
    second = LatestReport(tmp_path)
    assert first.run_id != second.run_id and first.path == second.path
    assert second.save({'outcome': {'status': 'completed'}}, {'place_release': {'name': 'place_release'}})
    assert not first.save({'outcome': {'status': 'failed'}})
    with pytest.raises(ValueError, match='changed'):
        archive_latest(tmp_path, expected_run_id=first.run_id)
    archived = archive_latest(tmp_path, expected_run_id=second.run_id)
    assert read_json(archived)['outcome']['status'] == 'completed'
    assert not (second.path/'processes/pick_part.json').exists()
    assert second.save({'outcome': {'status': 'stopped'}})
    assert read_json(archived)['outcome']['status'] == 'completed'
    assert len(list(tmp_path.glob('*/run.json'))) == 1


@pytest.mark.parametrize('function_name', ['pick_part', 'move_to_resource', 'place_release'])
@pytest.mark.parametrize('fail', [False, True])
def test_kmr_execution_uses_declared_composition_and_preserves_partial_results(function_name, fail):
    from cais_spade_llm.recovery_framework.kmr_tasks import KMR_TASKS, capability_decompositions, execute_composition

    definition = KMR_TASKS[function_name]
    metadata = capability_decompositions(function_name=function_name)
    assert [row['primitive'] for row in metadata['recovery_visible_steps']] == [s.op for s in definition.program.steps]
    outputs = {'compute_pick_targets': {key: [1, 2, 3] for key in (
                   'approach', 'seed', 'target', 'lift', 'retreat', 'carrying')},
               'compute_place_targets': {key: [1, 2, 3] for key in (
                   'approach', 'seed', 'target', 'retreat', 'destination', 'parked', 'before_turn', 'joint_a1')},
               'observe_grasp': [0]*7, 'observe_custody': [0]*7}
    calls, records, operations = [], [], []
    failed_step = min(3, len(definition.program.steps)-1)
    def primitive(name):
        def execute(**params):
            calls.append(name)
            operations.append({'operation': name, 'success': not (fail and len(calls)-1 == failed_step)})
            if fail and len(calls)-1 == failed_step:
                raise ValueError('observed primitive failure')
            return outputs.get(name, {'success': True})
        return execute
    primitives = {step.op: primitive(step.op) for step in definition.program.steps}
    def run():
        return execute_composition(
            function_name, arguments={'target_resource': 'M1'}, state={'initial': [1, 2, 3], 'previous': {}},
            primitives=primitives, records=records, operations=operations, now=lambda: 1.,
            check=lambda: None, serialize=deepcopy,
        )
    if fail:
        with pytest.raises(ValueError, match='observed primitive failure'):
            run()
        assert len(records) == failed_step+1 and records[-1]['status'] == 'failed'
        assert records[-1]['observations'][-1]['success'] is False
        assert all(r['status'] == 'completed' for r in records[:-1])
    else:
        run()
        assert calls == [step.op for step in definition.program.steps]
        assert all(r['status'] == 'completed' for r in records)
    assert all({'parameters', 'preconditions', 'effects', 'timing', 'observations'} <= r.keys() for r in records)


def test_kmr_owned_primitive_catalog_and_recovery_preserve_partial_custody():
    from cais_spade_llm.recovery_framework.gazebo_worker import GazeboExecutionError
    from cais_spade_llm.recovery_framework.kmr_agent import KMRResourceAgent

    async def scenario():
        target = [1., 2., 3., 1., 0., 0., 0.]
        worker = SimpleNamespace(run=AsyncMock(side_effect=[
            GazeboExecutionError({
                'status': 'failed', 'error': 'injected incomplete Cartesian path',
                'primitive_results': [{'primitive': 'move_cartesian', 'status': 'failed'}],
            }),
            {'status': 'completed', 'result': {'tcp_pose': target, 'avoid_collisions': True, 'fraction': 1.},
             'primitive_results': [{'primitive': 'move_cartesian', 'status': 'completed',
                                    'result': {'tcp_pose': target}}]},
        ]))
        agent = KMRResourceAgent('kmr-recovery-test@localhost', 'none', worker=worker)
        agent._kmr_execution_request = {'pending': {'parameters': {'part_name': 'RGOCG4-50_Round_4mm'}}}
        agent._primitive_state.update(current_state='carrying', held_part='RGOCG4-50_Round_4mm',
                                      gripper_state='closed')
        catalog = {entry['name']: entry for entry in agent.recovery_execution_primitive_catalog()}
        assert {'compute_pick_targets', 'compute_place_targets', 'move_cartesian', 'rotate_arm_base',
                'dock', 'custody', 'open_gripper', 'close_gripper'} <= catalog.keys()
        assert catalog['move_cartesian']['params']['target']['type'] == 'array'
        assert catalog['move_cartesian']['output_schema']['fraction'] == 'number'
        failed = await agent.kmr_primitives.move_cartesian(target)
        assert not failed['success']
        assert agent._snapshot_state()['held_part'] == 'RGOCG4-50_Round_4mm'
        result = await agent.execute_recovery_macro(
            macro_name='retry_observed_vertical_lift', expected_start_state='carrying',
            primitive_steps=[{'primitive': 'move_cartesian', 'params': {'target': target}}],
        )
        assert result['status'] == 'completed'
        assert agent._snapshot_state()['held_part'] == 'RGOCG4-50_Round_4mm'
        assert agent._snapshot_state()['current_pose'] == target
        assert worker.run.await_count == 2
        assert all(call.args[0]['mode'] == 'primitive' for call in worker.run.await_args_list)
    asyncio.run(scenario())


@pytest.mark.parametrize('module_name', ['kmr_gazebo', 'workflow_gazebo'])
def test_owned_workers_parse_unchanged_request_once_and_exit_with_owner(tmp_path, monkeypatch, module_name):
    import importlib
    import json
    import sys

    module = importlib.import_module(f'cais_spade_llm.recovery_framework.{module_name}')
    request = {'id': 'single-command', 'request': {'mode': 'probe'}}
    (tmp_path / 'request.json').write_text(json.dumps(request))
    parse = Mock(side_effect=json.loads)
    execute = Mock(return_value={'status': 'completed'})
    monkeypatch.setattr(module, 'json', SimpleNamespace(loads=parse, dumps=json.dumps))
    monkeypatch.setattr(module, 'run', execute)
    monkeypatch.setattr(module.os, 'getppid', Mock(side_effect=[42, 42, 42, 42, 1]))
    monkeypatch.setitem(sys.modules, 'rclpy', SimpleNamespace(ok=lambda: False))
    if module_name == 'workflow_gazebo':
        monkeypatch.setattr(module.signal, 'signal', Mock())
    module.serve(tmp_path)
    parse.assert_called_once()
    execute.assert_called_once_with(request['request'], {})
    assert json.loads((tmp_path / 'result.json').read_text()) == {
        'id': 'single-command', 'result': {'status': 'completed'},
    }


def test_KMR_home_uses_configured_downward_pick_posture_at_Storage(inputs):
    from cais_spade_llm.recovery_framework.kmr_gazebo import storage_home
    from cais_spade_llm.recovery_framework.geometry import rotate

    scene = inputs['scene']
    round_part = 'RGOCG4-50_Round_4mm'
    home = storage_home(scene, {'parts': [round_part]})
    assert home['part_name'] == round_part
    assert home['joints'] == scene['Storage']['KMR_pick_arm_configurations'][round_part]
    assert rotate(home['tcp_pose'][3:], [0., 0., 1.])[2] == pytest.approx(-1.)
    assert home['tcp_pose'][2] > scene['Storage']['slots'][round_part][2] + .05
    # Returning to Storage must use a posture at that dock, even for a later column.
    other = storage_home(scene, {'parts': ['RGOCG16-50_16mm']})
    assert scene['Storage']['KMR_pick_docking_poses'][other['part_name']][:2] == scene['Storage']['KMR_docking_pose'][:2]
    scene['Storage']['KMR_docking_pose'][0] += 1.
    with pytest.raises(ValueError, match='no configured downward home posture'):
        storage_home(scene, {'parts': [round_part]})


def test_KMR_downward_home_is_owned_and_callable():
    from cais_spade_llm.recovery_framework.kmr_primitives import KMRPrimitives, KMR_RECOVERY_PRIMITIVES

    agent = SimpleNamespace(
        _kmr_execution_request={'inputs': {}, 'valuation': {'KMR': {'held_part': None}}},
        worker=SimpleNamespace(run=AsyncMock(return_value={
            'result': {'home_pose_observed': True, 'downward_facing': True},
        })),
        record_primitive_evidence=Mock(),
    )
    assert 'move_home' in KMR_RECOVERY_PRIMITIVES
    result = asyncio.run(KMRPrimitives(agent).move_home())
    assert result['success'] and result['home_pose_observed'] and result['downward_facing']
    request = agent.worker.run.call_args.args[0]
    assert request['mode'] == 'primitive' and request['primitive'] == 'move_home'
    assert request['primitive_parameters'] == {}
    agent.record_primitive_evidence.assert_called_once()


@pytest.mark.parametrize(
    ("event_name", "resource_id", "parameters", "start_x", "expected_x"),
    [
        (
            "machine_part", "M1",
            {"part_name": "part", "process": "trim", "result": "square"}, 0.0, 0.0,
        ),
        (
            "advance_conveyor", "Conveyor",
            {"next_locations": {"part": "loading_position_2"}, "delivered_part": None},
            0.0, 1.0,
        ),
        (
            "advance_part", "Buffer For Machined parts",
            {"part_name": "part", "zone": 1, "downstream_zone": 2}, 1.0, 2.0,
        ),
    ],
)
def test_workflow_gazebo_executes_declared_primitive_order(
    event_name, resource_id, parameters, start_x, expected_x
):
    from cais_spade_llm.recovery_framework.workflow_gazebo import WorkflowPrimitiveRunner
    from cais_spade_llm.resources.workflow_task_programs import workflow_task_program

    def entity_state(x, y=0.0, z=0.0):
        return SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=z),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ))

    scene = {
        "machines": [
            {
                "resource_id": "M1", "workholding_pose": [0.0, 0.0, 0.0],
                "handling_robot": "ur5e-1", "KMR_process_clearance_m": 0.1,
                "simulation_process": {"processing_time_sec": 5.0, "observation_interval_sec": 0.25},
                "conveyor_loading_pose": [0.0, 0.0, 0.0],
            },
            {"resource_id": "M2", "conveyor_loading_pose": [1.0, 0.0, 0.0]},
        ],
        "KMR": {"task_execution": {"clearance_link_bounds": {
            "link": [{"center": [0.0, 0.0, 0.0], "radius": 0.1}]
        }}},
        "Conveyor": {
            "output_nest_pose": [3.0, 0.0, 0.0],
            "simulation_transport": {"speed_mps": 1.0, "acceleration_mps2": 1.0},
        },
        "Buffer For Machined parts": {
            "slot_poses": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            "simulation_transport": {"speed_mps": 1.0, "acceleration_mps2": 1.0},
        },
    }
    entities = {"part": entity_state(start_x), "KMR::link": entity_state(5.0, 5.0, 5.0)}
    request = {
        "scene": scene,
        "geometry": {"part": {"model_name": "part"}},
        "valuation": {"Conveyor": {"part_location.part": "loading_position_1"}},
        "task": {
            "event_name": event_name, "resource_id": resource_id,
            "task_id": "task-1", "parameters": parameters,
        },
    }

    def wait_simulation(duration, observe):
        if observe is not None:
            observe(0.0)
            observe(duration)
        return 0.0, duration

    runner = WorkflowPrimitiveRunner(
        request,
        entity=lambda name: entities[name],
        set_pose=lambda name, pose: setattr(entities[name], "pose", pose),
        robot_clearance=lambda names, _states: {name: 1.0 for name in names},
        current_clock=lambda: 0.0,
        wait_simulation=wait_simulation,
    )
    observations = runner.execute()
    assert [row["primitive"] for row in runner.primitive_trace] == [
        step["op"] for step in workflow_task_program(event_name)["steps"]
    ]
    assert all(row["status"] == "completed" for row in runner.primitive_trace)
    assert entities["part"].pose.position.x == pytest.approx(expected_x)
    if event_name == "machine_part":
        assert observations["correct_part_present"] is True
        assert observations["robot_access_clear"] is True
        assert observations["processing_time_sec"] == 5.0
    else:
        assert observations["arrival_observed"] is True
        assert observations["source_clear"] is True
        assert observations["robot_clear"] is True


def test_workflow_gazebo_records_the_failed_primitive_without_acknowledging_arrival():
    from cais_spade_llm.recovery_framework.workflow_gazebo import WorkflowPrimitiveRunner

    state = SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=1.0, y=0.0, z=0.0)))
    request = {
        "scene": {"Buffer For Machined parts": {
            "slot_poses": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            "simulation_transport": {"speed_mps": 1.0, "acceleration_mps2": 1.0},
        }},
        "geometry": {"part": {"model_name": "part"}},
        "task": {"event_name": "advance_part", "resource_id": "Buffer For Machined parts",
                 "task_id": "task-1", "parameters": {"part_name": "part", "downstream_zone": 2}},
    }

    def fail_pose(_name, _pose):
        raise ValueError("Gazebo rejected controlled transport")

    def wait_simulation(duration, observe):
        observe(duration)
        return 0.0, duration

    runner = WorkflowPrimitiveRunner(
        request,
        entity=lambda _name: state,
        set_pose=fail_pose,
        robot_clearance=lambda names, _states: {name: 1.0 for name in names},
        current_clock=lambda: 0.0,
        wait_simulation=wait_simulation,
    )
    with pytest.raises(ValueError, match="Gazebo rejected controlled transport"):
        runner.execute()
    assert [row["status"] for row in runner.primitive_trace] == [
        "completed", "completed", "completed", "failed"
    ]
    assert runner.primitive_trace[-1]["primitive"] == "move_buffer_part"
    assert runner.observations == {}



def test_workflow_gazebo_rejects_missing_workholding_and_blocked_conveyor():
    from cais_spade_llm.recovery_framework.workflow_gazebo import WorkflowPrimitiveRunner

    part = SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=1.0, y=0.0, z=0.0)))
    geometry = {"part": {"model_name": "part"}}
    machine = WorkflowPrimitiveRunner(
        {
            "scene": {"machines": [{"resource_id": "M1", "workholding_pose": [0.0, 0.0, 0.0]}]},
            "geometry": geometry,
            "task": {"event_name": "machine_part", "resource_id": "M1",
                     "parameters": {"part_name": "part"}},
        },
        entity=lambda _name: part,
        set_pose=lambda _name, _pose: None,
        robot_clearance=lambda _names, _states: {},
        current_clock=lambda: 0.0,
        wait_simulation=lambda duration, _observe: (0.0, duration),
    )
    with pytest.raises(ValueError, match="not observed at machine workholding"):
        machine.execute()
    assert [(row["primitive"], row["status"]) for row in machine.primitive_trace] == [
        ("observe_workholding", "failed")
    ]
    assert machine.observations == {}

    def blocked_clearance(_names, _states):
        raise ValueError("Robot has not cleared the conveyor")

    conveyor = WorkflowPrimitiveRunner(
        {
            "scene": {
                "machines": [
                    {"conveyor_loading_pose": [1.0, 0.0, 0.0]},
                    {"conveyor_loading_pose": [2.0, 0.0, 0.0]},
                ],
                "Conveyor": {
                    "output_nest_pose": [3.0, 0.0, 0.0],
                    "simulation_transport": {"speed_mps": 1.0, "acceleration_mps2": 1.0},
                },
            },
            "geometry": geometry,
            "valuation": {"Conveyor": {"part_location.part": "loading_position_1"}},
            "task": {"event_name": "advance_conveyor", "resource_id": "Conveyor",
                     "parameters": {"next_locations": {"part": "loading_position_2"}}},
        },
        entity=lambda _name: part,
        set_pose=lambda _name, _pose: pytest.fail("Blocked conveyor must not move"),
        robot_clearance=blocked_clearance,
        current_clock=lambda: 0.0,
        wait_simulation=lambda duration, _observe: (0.0, duration),
    )
    with pytest.raises(ValueError, match="Robot has not cleared the conveyor"):
        conveyor.execute()
    assert [row["status"] for row in conveyor.primitive_trace] == [
        "completed", "completed", "failed"
    ]
    assert conveyor.primitive_trace[-1]["primitive"] == "verify_transport_clearance"
    assert conveyor.observations == {}


def test_planned_print_part_program_has_no_gazebo_primitive_dispatch():
    from cais_spade_llm.recovery_framework.workflow_gazebo import WorkflowPrimitiveRunner
    from cais_spade_llm.resources.workflow_task_programs import workflow_task_program

    assert workflow_task_program("print_part")["status"] == "planned"
    runner = WorkflowPrimitiveRunner(
        {
            "scene": {},
            "task": {
                "event_name": "print_part", "resource_id": "3D Printing Station",
                "parameters": {"part_name": "gear_small"},
            },
        },
        entity=Mock(), set_pose=Mock(), robot_clearance=Mock(),
        current_clock=Mock(), wait_simulation=Mock(),
    )
    with pytest.raises(ValueError, match="Unsupported workflow Gazebo event: print_part"):
        runner.execute()
    assert runner.primitive_trace == []


@pytest.mark.parametrize("failure_phase", ["before_motion", "after_motion", "arrival", "collision_scene"])
def test_transport_failure_retains_observed_progress_without_event_completion(failure_phase):
    from cais_spade_llm.recovery_framework.workflow_gazebo import WorkflowPrimitiveRunner

    physical = SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=1.0, y=0.0, z=0.0)))
    calls = 0

    def entity(_name):
        return deepcopy(physical)

    def set_pose(_name, pose):
        nonlocal calls
        calls += 1
        if failure_phase == "before_motion":
            raise ValueError("transport interrupted")
        physical.pose = deepcopy(pose)
        if failure_phase == "after_motion":
            raise ValueError("transport interrupted after physical movement")
        if failure_phase == "arrival" and calls == 2:
            physical.pose.position.x -= .05

    def wait_simulation(duration, observe):
        observe(duration / 2)
        observe(duration)
        return 0.0, duration

    runner = WorkflowPrimitiveRunner({
        "scene": {"Buffer For Machined parts": {
            "slot_poses": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            "simulation_transport": {"speed_mps": 1.0, "acceleration_mps2": 1.0},
        }},
        "geometry": {"part": {"model_name": "part"}},
        "task": {"task_id": "transport-failure", "resource_id": "Buffer For Machined parts",
                 "event_name": "advance_part", "parameters": {"part_name": "part", "zone": 1, "downstream_zone": 2}},
    }, entity=entity, set_pose=set_pose,
        robot_clearance=lambda names, _states: {name: 1.0 for name in names},
        current_clock=lambda: 0.0, wait_simulation=wait_simulation,
        update_part_collisions=(Mock(side_effect=ValueError("collision scene rejected"))
                                if failure_phase == "collision_scene" else None))
    with pytest.raises(ValueError):
        runner.execute()
    assert not runner.observations.get("arrival_observed")
    assert runner.primitive_trace[-1]["status"] == "failed"
    assert all(row["task_id"] == "transport-failure" for row in runner.primitive_trace)
    evidence = runner.primitive_trace[-1]["observations"]
    assert evidence["initial_x"] == {"part": 1.0}
    assert evidence["target_x"] == {"part": 2.0}
    assert evidence["observed_x"]["part"] == physical.pose.position.x
    assert (physical.pose.position.x == 1.0) is (failure_phase == "before_motion")


@pytest.mark.parametrize("empty_return", [False, True])
def test_scene_refresh_removes_old_payload_boxes_and_preserves_observed_parts(empty_return):
    import ast
    from pathlib import Path
    from cais_spade_llm.recovery_framework import kmr_gazebo

    tree = ast.parse(Path(kmr_gazebo.__file__).read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'install_scene')
    observed = {"first": [0., 0., 1.024], "second": [-6., 1.8, 1.2]}
    boxes = [{"id": f"{part}/link/collision", "pose": pose, "size": [.02] * 3}
             for part, pose in observed.items()]
    scene = SimpleNamespace(world=SimpleNamespace(collision_objects=[]))
    collision_object = Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
    collision_object.REMOVE = 1
    components = Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
    components.WORLD_OBJECT_NAMES = 8
    namespace = {
        "PlanningScene": lambda **kw: scene,
        "component_parts": list(observed), "part": "second", "entity": observed.__getitem__,
        "mode": "environment_task", "name": "move_to_resource" if empty_return else "place_release",
        "held_part": None if empty_return else "second", "ROOT": Path('/tmp'),
        "collision_boxes": lambda *args: [row for row in boxes if row["id"].split("/")[0] in args[2]],
        "session": {},
        "GetPlanningScene": SimpleNamespace(Request=lambda **kw: SimpleNamespace(**kw)),
        "PlanningSceneComponents": components, "CollisionObject": collision_object,
        "box_object": lambda identifier, pose, size: SimpleNamespace(id=identifier, operation=0, pose=pose),
        "service": lambda *args, **kw: SimpleNamespace(scene=SimpleNamespace(world=SimpleNamespace(
            collision_objects=[SimpleNamespace(id=v) for v in ['first', 'fixture', 'second/link/collision']]))),
        "apply": Mock(), "operations": [],
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(kmr_gazebo.__file__), 'exec'), namespace)
    namespace['install_scene']()
    updates = {obj.id: obj for obj in scene.world.collision_objects}
    assert updates['first'].operation == collision_object.REMOVE
    assert updates['first/link/collision'].pose == observed['first']
    assert 'fixture' not in updates
    assert updates['second/link/collision'].operation == (0 if empty_return else collision_object.REMOVE)
    assert namespace['operations'][0]['removed_released_part_objects'] == ['first']
    namespace['install_scene']()
    updates = {obj.id: obj for obj in scene.world.collision_objects}
    assert 'first' not in updates
    assert 'first/link/collision' not in updates
    assert namespace['operations'][-1]['initialized'] is False
    assert namespace['operations'][-1]['observed_part_poses'] == ({} if empty_return else {'second': observed['second']})

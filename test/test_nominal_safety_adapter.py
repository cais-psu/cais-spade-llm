from __future__ import annotations

"""Saved nominal expansion, exact motion lowering, and observed completion."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_continuous_motion import configuration, trajectory

from cais_spade_llm.recovery_framework.nominal_safety_adapter import (
    NominalRobotAdapter,
    install_nominal_robot_adapter,
    motion_effects,
    prepare_step,
    resolve_motion,
)
from cais_spade_llm.resources.resource_safety_preparation import LiveCommandLedger, fingerprint
from cais_spade_llm.resources.robot.robot_task_model import (
    RobotTaskDefinition,
    RobotTaskEffect,
    RobotTaskProgram,
    RobotTaskStep,
)


def _configuration():
    return {"joint_names": ["slide"], "continuous_motion": configuration(),
            "named_positions": {"home": [1.0]}, "controller_configuration": {"cartesian_motion": {}}}


def test_relative_lowering_uses_observed_pose_without_mutating_original_operation():
    start, params = [1, 2, 3, 0, 0, 0, 1], {"dx": -1, "dz": 0.5, "speed": 2}
    before = deepcopy([start, params])
    assert resolve_motion("move_relative", params, start, _configuration()) == {
        "x": 0, "y": 2, "z": 3.5, "qx": 0, "qy": 0, "qz": 0, "qw": 1, "speed": 2}
    assert [start, params] == before
    with pytest.raises(ValueError, match="finite"):
        resolve_motion("move_relative", {"dz": float("nan")}, start, _configuration())


def test_named_pose_uses_configured_fk_and_rejects_orientation_change():
    config = _configuration()
    result = resolve_motion("move_to_named_pose", {"pose_name": "home"}, [0, 0, 0, 0, 0, 0, 1], config)
    assert result["x"] == pytest.approx(1)
    config["continuous_motion"]["joints"][0].update(type="revolute", axis=[0, 0, 1])
    with pytest.raises(ValueError, match="changes orientation"):
        resolve_motion("move_to_named_pose", {"pose_name": "home"}, [0, 0, 0, 0, 0, 0, 1], config)


@pytest.mark.parametrize("primitive", ["open_gripper", "close_gripper", "grasp_part", "release_part", "move_insert"])
def test_unmodeled_native_effects_never_create_synthetic_hold_evidence(primitive):
    controller = SimpleNamespace(
        arm_joint_names=["slide"], controller_config={}, named_positions={},
        recovery_safety_observer=SimpleNamespace(motion_configuration=configuration()))
    result = prepare_step(controller, primitive=primitive, params={},
                          start={"current_pose": [0, 0, 0, 0, 0, 0, 1]}, binding={})
    assert result["status"] == "NEEDS_CONTEXT" and result["command_sent"] is False
    assert "joint_trajectory" not in result and "continuous_motion" not in result


def test_model_rejects_changed_relative_lowering():
    config, raw = _configuration(), trajectory(0, 1, None)
    evidence = {"joint_trajectory": raw, "continuous_motion": {
        "joint_trajectory": raw, "configuration": config["continuous_motion"]},
        "resolved_motion": {"primitive": "move_cartesian",
                            "params": {"x": 1, "y": 0, "z": 0, "qx": 0, "qy": 0, "qz": 0, "qw": 1}}}
    effects = motion_effects(primitive="move_relative", params={"dx": 1}, evidence=evidence,
                             start_time=0, end_time=1, configuration=config)
    assert effects["continuous_motion"] == evidence["continuous_motion"]
    evidence["resolved_motion"]["params"]["x"] = 2
    with pytest.raises(ValueError, match="differs from the original"):
        motion_effects(primitive="move_relative", params={"dx": 1}, evidence=evidence,
                       start_time=0, end_time=1, configuration=config)


def _adapter(monkeypatch, *, steps=None):
    definition = RobotTaskDefinition(
        name="move_home", description="configured fixture", arguments=(),
        program=RobotTaskProgram(entry_state="ready", success_state="idle",
            steps=tuple(steps or [RobotTaskStep(id="hidden_motion", op="move_relative",
                                               exposed=False, params={"dx": 1})]),
            effects=(RobotTaskEffect(target="current_state", action="set", value="idle"),),
            success_response={"content": "completed observed configured steps"}))
    monkeypatch.setattr("cais_spade_llm.resources.gazebo_programs.saved_robot_definition",
                        lambda *args, **kwargs: definition)
    observed = {"joint_names": ["slide"], "joint_positions": [1],
                "current_pose": [1, 0, 0, 0, 0, 0, 1], "attachment": {"model_name": None},
                "launch_id": "launch", "frame": "world"}
    owner = SimpleNamespace(agent_name="robot", execution_mode="simulation",
                            _current_state="ready", _controller=SimpleNamespace(
                                controller_config={}, capture_recovery_safety_state=lambda: deepcopy(observed)))
    ledger = LiveCommandLedger()
    owner.recovery_composition_evidence_provider = SimpleNamespace(ledger=ledger)
    task = {"task_id": "task", "resource_id": "robot", "event_name": "move_home", "parameters": {}}
    return NominalRobotAdapter(owner, {}), task, ledger, observed


def test_nominal_full_saved_program_includes_hidden_motion(monkeypatch):
    adapter, task, _, _ = _adapter(monkeypatch)
    output = adapter.prepare(task)
    assert output["primitive_steps"][0]["primitive"] == "move_relative"
    assert output["primitive_steps"][0]["source"]["step_id"] == "hidden_motion"
    assert adapter.owner._current_state == "ready"


def test_nominal_full_saved_program_cannot_drop_unmodeled_hidden_step(monkeypatch):
    adapter, task, _, _ = _adapter(monkeypatch, steps=[
        RobotTaskStep(id="hidden_open", op="open_gripper", exposed=False)])
    with pytest.raises(ValueError, match="complete native preparation contract: open_gripper"):
        adapter.prepare(task)


def test_nominal_completion_requires_observed_authorized_ledger_and_commits_once(monkeypatch):
    adapter, task, ledger, observed = _adapter(monkeypatch)
    adapter.prepare(task)
    result = {"success": True, "preparation_id": "native", "joint_trajectory": trajectory(0, 1, None),
              "observations": observed}
    with pytest.raises(ValueError, match="authenticated owner ledger"):
        adapter.complete(task, [result])
    command = {"primitive": "move_relative", "params": {"dx": 1},
               "joint_trajectory": result["joint_trajectory"]}
    identifier = ledger.prepare(resource_id="robot", task_id="task", command=command,
                                owner_identity={"preparation_id": "native"})
    ledger.authorize(identifier, reservation_token="reservation", expected_revision=ledger.revision)
    ledger.start(identifier, command=command, controller_goal_id=[1] * 16)
    ledger.finish(identifier, success=True, observations=observed)
    response = adapter.complete(task, [result])
    assert response["status"] == "completed" and adapter.owner._current_state == "idle"
    with pytest.raises(ValueError, match="already consumed"):
        adapter.complete(task, [result])


@pytest.mark.parametrize('mode', [True, False, None, 1])
def test_installed_preparation_retains_only_checkpoint_owned_model_execution(monkeypatch, mode):
    owner, program, checkpoint, observed = _installed_preparation(monkeypatch)
    if mode is not None:
        checkpoint['model_execution'] = mode
    checkpoint['observations']['robot']['physical']['model_execution'] = True
    if mode is True:
        observed['joint_positions'] = [.004]
        observed['current_pose'][0] = .004
        observed['component_bounds'][0]['bounds'][0] = [-.046, .054]
    before = deepcopy(checkpoint)
    prepared = owner._controller.prepare_recovery_safety_program(program, checkpoint, resource_jid='robot@local')
    assert prepared['status'] == 'prepared'
    assert len(prepared['steps']) == 2
    for index, record in enumerate(prepared['steps']):
        assert record['start']['model_execution'] is (mode is True)
        assert record['binding']['model_execution'] is (mode is True)
        assert record['binding']['step_index'] == index
        if mode is True:
            assert record['start']['model_execution_regions'] == checkpoint['geometry']['regions']
            assert record['binding']['model_execution_regions'] == checkpoint['geometry']['regions']
            assert record['binding']['stationary_contract'] == record['start']['stationary_contract']
        assert record['preparation_id'] == fingerprint({key: value for key, value in record.items()
                                                      if key != 'preparation_id'})
        assert record['command_sent'] is False
    assert checkpoint == before


@pytest.mark.parametrize(('change', 'reason'), [
    ('strict_drift', 'Checkpoint changed before planning: joint_positions'),
    ('joint', 'joint_position_error'), ('pose', 'initial FK enclosure'),
    ('component', 'component_bounds'), ('identity', 'launch_id'),
])
def test_installed_preparation_rejects_changed_identity_or_outside_modeled_start(monkeypatch, change, reason):
    owner, program, checkpoint, observed = _installed_preparation(monkeypatch)
    checkpoint['model_execution'] = change != 'strict_drift'
    if change in {'strict_drift', 'joint'}:
        observed['joint_positions'] = [.02]
    elif change == 'pose':
        observed['current_pose'][0] = .02
    elif change == 'component':
        observed['component_bounds'][0]['bounds'][0] = [-.05, .08]
    else:
        observed['launch_id'] = 'another launch'
    prepared = owner._controller.prepare_recovery_safety_program(program, checkpoint, resource_jid='robot@local')
    assert prepared['status'] == 'NEEDS_CONTEXT' and reason in prepared['reason']
    assert all(record['command_sent'] is False for record in prepared['steps'])


def _installed_preparation(monkeypatch):
    """Exercise the installed adapter while replacing only the native planner."""
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller

    config = configuration()
    config['joint_position_error'] = {'slide': .01}
    start = {'joint_names': ['slide'], 'joint_positions': [0.],
             'frame': 'world', 'launch_id': 'launch', 'attachment': {'model_name': None},
             'current_pose': [0., 0., 0., 0., 0., 0., 1.],
             'stationary_contract': {'kind': 'idle_commanded_hold', 'requires_no_running_tasks': True,
                 'requires_no_active_goals': True, 'future_execution_tracking': 'not_established'},
             'component_bounds': [{'id': 'observed::body::collision', 'link': 'body', 'bounds': [[-.05, .05]] * 3}]}
    observed = deepcopy(start)
    controller = SimpleNamespace(arm_joint_names=['slide'], controller_config={}, named_positions={},
        get_recovery_safety_primitive_model=lambda: None,
        recovery_safety_observer=SimpleNamespace(motion_configuration=config),
        capture_recovery_safety_state=lambda: deepcopy(observed))
    owner = SimpleNamespace(_controller=controller)
    install_nominal_robot_adapter(owner, {})

    def native_prepare(controller, *, primitive, params, start, binding):
        raw = trajectory(start['joint_positions'][0], params['x'], None)
        return {'status': 'prepared', 'observation_status': 'prepared', 'command_sent': False,
                'start': deepcopy(start), 'binding': deepcopy(binding), 'joint_trajectory': raw,
                'continuous_motion': {'joint_trajectory': deepcopy(raw), 'configuration': deepcopy(config)},
                'target_pose': [params['x'], 0., 0., 0., 0., 0., 1.]}

    monkeypatch.setattr(gazebo_pick_place_controller, 'prepare_ur5e_motion', native_prepare)
    program = {'resource_id': 'robot', 'primitive_steps': [
        {'primitive': 'move_relative', 'params': {'dx': .1}},
        {'primitive': 'move_cartesian', 'params': {'x': .2, 'y': 0., 'z': 0.}},
    ]}
    checkpoint = {'checkpoint_id': 'checkpoint', 'runtime': {'run_id': 'run'},
                  'geometry': {'regions': {'assembly_board-v1': {'frame': 'world', 'bounds': [[1., 2.]] * 3}}},
                  'observations': {'robot': {'physical': start}}}
    return owner, program, checkpoint, observed

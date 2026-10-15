from __future__ import annotations

"""KMR owner previews preserve declared formats and isolated physical state."""

import asyncio
from copy import deepcopy
import hashlib
import math
from types import SimpleNamespace

import pytest

from cais_spade_llm.recovery_framework.kmr_outline_evidence import (
    _apply_motion, _base_pose, _case_evidence, _motion_request, prepare_kmr_event_evidence, project_kmr_event,
)
from cais_spade_llm.recovery_framework.recovery_motion_preview import preview_result_record


def motion_fixture():
    names = [f"joint_a{i}" for i in range(1, 8)]
    config = {"root": "KMR_base_link", "reference_link": "tool", "root_xyz": [0., 0., 0.],
        "root_quaternion": [0., 0., 0., 1.], "joint_names": names, "interpolation": "splines",
        "joints": [{"name": name, "parent": "KMR_base_link" if i == 0 else "link"+str(i),
                    "child": "tool" if i == 6 else "link"+str(i+1), "type": "revolute", "axis": [0., 0., 1.],
                    "xyz": [0., 0., .1], "rpy": [0., 0., 0.], "position_bounds": [-3., 3.]}
                   for i, name in enumerate(names)],
        "components": [{"id": "tool", "link": "tool", "bounds": [[-.02, .02]]*3}]}
    owner = {"joint_names": names, "joint_positions": [0.]*7, "held_part": None,
             "current_pose": [0., 0., .7, 0., 0., 0., 1.]}
    physical = {"snapshot": {"frame": "world", "resources": {"KMR": owner, "M1": {"part_name": None}}, "parts": {}},
        "resource_models": {"KMR": {"configuration": config}, "M1": {}},
        "resources": {"KMR": {}, "M1": {"model_static": True}},
        "geometry": {"frame": "world", "regions": {}, "parts": {}, "resources": {
            "KMR": {}, "M1": {"stationary_only": True, "component_bounds": [{"id": "body", "bounds": [[4., 5.]]*3}]}}}}
    state = {"is_diff": False, "joint_state": {"name": names, "position": [0.]*7}}
    request = {"planning_scene": {"is_diff": False, "robot_state": state}, "start_state": deepcopy(state),
        "group_name": "KMR_iiwa_arm", "link_name": "tool", "frame_id": "world", "resource_id": "KMR",
        "capture_fingerprint": "capture", "proposal_fingerprint": "proposal", "scene_fingerprint": "scene", "model_fingerprint": "model",
        "mode": "joint_target", "joint_target": {"name": names, "position": [.1]*7},
        "max_translation_step": .01, "max_rotation_step": .01, "max_joint_step": .01,
        "velocity_scaling": .1, "acceleration_scaling": .1, "timeout_seconds": 10.}
    raw = {key: request[key] for key in ("resource_id", "capture_fingerprint", "proposal_fingerprint", "scene_fingerprint", "model_fingerprint")}
    raw.update(status="PREVIEW_COMPLETE", path_complete=True, fraction=1., continuous_collision_verified=False,
        trajectory={"joint_trajectory": {"joint_names": names, "points": [
            {"positions": [0.]*7, "velocities": [0.]*7, "time_from_start": {"sec": 0, "nanosec": 0}},
            {"positions": [.1]*7, "velocities": [0.]*7, "time_from_start": {"sec": 1, "nanosec": 0}}]}})
    return physical, request, preview_result_record(request, raw)


def test_seven_joint_motion_replays_peer_scene_without_mutating_capture():
    physical, request, response = motion_fixture()
    original = deepcopy(physical)
    successor, points, interval = _apply_motion({}, physical, request, response, 0.)
    assert physical == original
    assert successor["snapshot"]["resources"]["KMR"]["joint_positions"] == [.1]*7
    assert successor["snapshot"]["resources"]["M1"] == original["snapshot"]["resources"]["M1"]
    assert points[0]["phase"] == "between" and points[1]["phase"] == "at"
    assert interval["stationary_resources"] == ["M1"]


def test_motion_intervals_retain_global_time_after_a_previous_motion():
    physical, request, response = motion_fixture()
    _, points, interval = _apply_motion({}, physical, request, response, .125)
    assert points[0]["continuous_interval"] == ["1/8", "9/8"]
    assert points[0]["continuous_cells"][0]["interval"][0] == "1/8"
    assert points[0]["continuous_cells"][-1]["interval"][1] == "9/8"
    assert points[1]["time_exact"] == interval["end_time_exact"] == "9/8"


def test_projected_custody_does_not_reuse_captured_velocities():
    from cais_spade_llm.resources.recovery_geometry import _checkpoint

    physical, _, _ = motion_fixture()
    owner = physical["snapshot"]["resources"]["KMR"]
    owner.update(base_pose=[0., 0., 0., 0., 0., 0., 1.], gripper_state="open")
    physical["physics_evidence"] = {"models": {"KMR": {
        "joints": {name: {"position": 0., "velocity": 0.} for name in owner["joint_names"]},
        "links": {"KMR_base_link": {"pose": owner["base_pose"], "linear_velocity": [0.]*3,
                                    "angular_velocity": [0.]*3}}}}}
    point = _checkpoint(physical["snapshot"], physical["geometry"], physical["resource_models"], index=2)
    evidence = {"inputs": {"scene": {"KMR": {"arm_joint_names": owner["joint_names"]}}}}
    source = {"op": "open_gripper", "step_fingerprint": "step", "source_fingerprint": "preview"}
    _case_evidence([point], physical=physical, evidence=evidence, declared={"task": {"parameters": {}}},
                   custody_source=source)
    raw = point["owner_safety_evidence"]["resource_evidence"]["KMR"]
    assert "base_velocity_bounds" not in raw and "joint_velocity_bounds" not in raw
    assert point["owner_safety_evidence"]["physical_state_updates"] == {"KMR": {"gripper_state": "open"}}


@pytest.mark.parametrize("defect", ["partial", "request", "trajectory", "start", "endpoint", "request_boolean"])
def test_native_result_rejects_partial_stale_or_changed_evidence(defect):
    physical, request, response = motion_fixture()
    if defect == "partial":
        response.update(path_complete=False, fraction=.5)
    elif defect == "request":
        response["request"]["proposal_fingerprint"] = "different"
    elif defect == "trajectory":
        response["joint_trajectory"]["points"][-1]["positions"][0] = .2
    elif defect == "start":
        physical["snapshot"]["resources"]["KMR"]["joint_positions"][0] = .3
    elif defect == "request_boolean":
        response["request"]["start_state"]["joint_state"]["position"][0] = False
    else:
        response["joint_trajectory"]["points"][-1]["positions"][0] = .2
        response["trajectory"]["joint_trajectory"]["points"][-1]["positions"][0] = .2
    with pytest.raises(ValueError):
        _apply_motion({}, physical, request, response, 0.)


def test_exact_kmr_move_joints_and_rotate_joint_names():
    physical, _, _ = motion_fixture()
    params = {"joints": [.2]*7}
    request = _motion_request({"op": "move_joints"}, params, physical, {"inputs": {"scene": {}}})
    assert request["joint_target"]["position"] == [.2]*7
    assert params == {"joints": [.2]*7}
    rotated = _motion_request({"op": "rotate_joint"}, {"joint_name": "joint_a7", "delta_deg": 90.},
                              physical, {"inputs": {"scene": {}}})
    assert rotated["joint_target"]["position"][:6] == [0.]*6
    assert rotated["joint_target"]["position"][6] == pytest.approx(1.5707963267948966)


@pytest.mark.parametrize('defect', ['', 'part', 'declared_part', 'coordinate', 'duplicate'])
def test_initial_state_separates_proposal_part_context_from_movement_arguments(monkeypatch, defect):
    from cais_spade_llm.recovery_framework import kmr_outline_evidence, kmr_recovery

    physical, _, _ = motion_fixture()
    task = {'resource_jid': 'KMR@localhost', 'part_name': 'gear_small'}
    declared = {'task': {'event_name': 'move_to_location', 'parameters': {'x': 1., 'y': 2., 'yaw': 0.}}}
    parameters = {**declared['task']['parameters'], 'part_name': task['part_name']}
    if defect == 'part':
        parameters['part_name'] = 'KET4_Square_4mm'
    elif defect == 'coordinate':
        parameters['x'] = 3.
    names = {name: {} for name in declared['task']['parameters']}
    if defect == 'declared_part':
        names['part_name'] = {'required': True}
    record = {'event_name': 'move_to_location', 'parameter_bindings': parameters,
              'calculated_state': {'previous': {'held_part': 'gear_small'}}}
    calculations = [record, deepcopy(record)] if defect == 'duplicate' else [record]
    monkeypatch.setattr(kmr_recovery, 'prepare_kmr_primitive_parameters',
        lambda **kwargs: {'parameter_calculations': deepcopy(calculations), 'conflicts': []})
    monkeypatch.setattr(kmr_outline_evidence, '_program', lambda *args: {'parameter_bindings': names})
    before = deepcopy((task, declared, calculations, physical))
    if defect:
        with pytest.raises(ValueError, match='unique recorded contributor'):
            kmr_outline_evidence._initial_state(task, {}, {}, physical, {}, declared)
    else:
        state = kmr_outline_evidence._initial_state(task, {}, {}, physical, {}, declared)
        assert state == record['calculated_state']
    assert (task, declared, calculations, physical) == before


def test_route_or_seed_is_not_substituted_for_native_motion():
    physical, _, _ = motion_fixture()
    with pytest.raises(ValueError, match="base-controller"):
        _motion_request({"op": "move_base"}, {"target_pose": [1., 2., 0.]}, physical, {"inputs": {"scene": {}}})
    with pytest.raises(ValueError, match="seven-joint"):
        _motion_request({"op": "move_cartesian"}, {"target": [0., 0., .7, 0., 0., 0., 1.], "seed": [0.]*6},
                        physical, {"inputs": {"scene": {}}})


def test_base_model_does_not_override_a_conflicting_observation():
    physical, _, _ = motion_fixture()
    physical["snapshot"]["resources"]["KMR"]["base_pose"] = [1., 0., 0., 0., 0., 0., 1.]
    with pytest.raises(ValueError, match="disagree"):
        _base_pose(physical)


@pytest.mark.parametrize('change', [None, 'source', 'root', 'axis', 'origin', 'mimic', 'nonplanar',
    'boolean', 'frame', 'missing_joint', 'duplicate_joint', 'malformed'])
def test_private_base_projection_uses_the_bound_physical_pose_and_declared_chain(change):
    from cais_spade_llm.recovery_framework.kmr_outline_evidence import _project_base_state

    physical, _, _ = motion_fixture()
    pose = [1.5, -2., 0., 0., 0., math.sin(.15), math.cos(.15)]
    configuration = physical['resource_models']['KMR']['configuration']
    if change == 'nonplanar':
        pose[2] = .1
    elif change == 'boolean':
        pose[0] = True
    configuration.update(root_xyz=pose[:3], root_quaternion=pose[3:])
    physical['snapshot']['resources']['KMR']['base_pose'] = deepcopy(pose)
    names = ['KMR_base_x_joint', 'KMR_base_y_joint', 'KMR_base_yaw_joint', 'unrelated_joint']
    description = '<robot name="recorded">' + ''.join(
        f'<joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/>'
        f'<axis xyz="{axis}"/></joint>' for name, kind, parent, child, axis in (
            (names[0], 'prismatic', 'world', 'KMR_base_x_link', '1 0 0'),
            (names[1], 'prismatic', 'KMR_base_x_link', 'KMR_base_y_link', '0 1 0'),
            (names[2], 'continuous', 'KMR_base_y_link', 'KMR_base_link', '0 0 1'))) + '</robot>'
    if change == 'axis':
        description = description.replace('xyz="1 0 0"', 'xyz="0 1 0"')
    elif change == 'origin':
        description = description.replace('<axis', '<origin xyz="1 0 0"/><axis', 1)
    elif change == 'mimic':
        description = description.replace('<axis', '<mimic joint="unrelated_joint"/><axis', 1)
    elif change == 'malformed':
        description = '<robot'
    configuration['description_sha256'] = hashlib.sha256(description.encode()).hexdigest()
    if change == 'source':
        description += ' '
    elif change == 'root':
        physical['snapshot']['resources']['KMR']['base_pose'][0] += 1.
    elif change == 'frame':
        physical['snapshot']['frame'] = 'different'
    state = {'joint_state': {'name': names, 'position': [99., 88., 77., 66.]}}
    projected = {'start_state': deepcopy(state), 'planning_scene': {'robot_state': deepcopy(state)}}
    if change == 'missing_joint':
        projected['start_state']['joint_state']['name'].pop(0)
        projected['start_state']['joint_state']['position'].pop(0)
    elif change == 'duplicate_joint':
        projected['start_state']['joint_state']['name'][3] = names[0]
    capture = {'robot_description': description}
    original = deepcopy(physical), deepcopy(capture)
    if change is not None:
        with pytest.raises(ValueError):
            _project_base_state(capture, physical, projected)
    else:
        _project_base_state(capture, physical, projected)
        assert projected['start_state']['joint_state']['position'] == pytest.approx([1.5, -2., .3, 66.])
        assert projected['start_state'] == projected['planning_scene']['robot_state']
    assert (physical, capture) == original


def test_scoped_peer_holds_do_not_remove_the_kmr_base_readiness_obligation():
    from cais_spade_llm.recovery_framework.kmr_outline_evidence import _holds

    physical, _, _ = motion_fixture()
    with pytest.raises(ValueError, match='base'):
        _holds(physical, robot_description='<robot name="recorded"/>')


def test_offline_preview_never_falls_back_to_live_interfaces(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("live interface called")
    monkeypatch.setattr("cais_spade_llm.recovery_framework.recovery_motion_preview.capture_preview_scene", forbidden)
    monkeypatch.setattr("cais_spade_llm.recovery_framework.recovery_motion_preview.preview_recovery_motion", forbidden)
    with pytest.raises(ValueError, match="saved-model"):
        asyncio.run(prepare_kmr_event_evidence(SimpleNamespace(), task={}, recovery_snapshot={},
            recovery_des_model={}, physical_evidence={}, resource_evidence={"execution_mode": "simulation"},
            observations={}, allow_live_capture=False))
    result = project_kmr_event(task={}, recovery_snapshot={}, recovery_des_model={}, resource_evidence={}, physical={}, declared={})
    assert result["status"] == "NEEDS_CONTEXT" and result["execution_authorized"] is False


def released_scene_fixture():
    part = "KET4_Square_4mm"
    declaration = {"status": "recorded", "model_name": part, "collision_scene_acknowledged": False,
        "collision_objects": [{"id": part + "/link/collision", "model": part, "link": "link", "collision": "collision"}]}
    command = {"resource_id": "KMR", "task_id": "environment_14", "status": "completed"}
    observations = {"run_id": "capture", "context_revision": 4, "jids": {}, "physical_evidence": {
        "parts": {part: {"name": part, "collision_scene_identity": declaration}},
        "physics_evidence": {"models": {part: {"links": {"link": {
            "collisions": [{"id": part + "::link::collision"}]}}}}}},
        "transitions": [{"acknowledgement": {**command, "run_id": "capture", "part_name": part, "revision": 3},
            "observations": {**command, "observations": {"part_name": part, "attached": False,
                "primitive_results": [{**command, "primitive": "release_part", "result": {
                    "attached": False, "held_part": None}}]}}}]}
    scene = {"scene_fingerprint": "scene", "planning_scene": {
        "world": {"collision_objects": [{"id": part}]}, "robot_state": {"attached_collision_objects": []}}}
    return observations, scene, part


def test_released_scene_identity_uses_exact_owner_acknowledgement_without_changing_capture():
    from cais_spade_llm.recovery_framework.kmr_recovery import recorded_scene_payloads
    from cais_spade_llm.recovery_framework.outline_gazebo_evidence import _scene_owner_records

    observations, scene, part = released_scene_fixture()
    original = deepcopy(observations), deepcopy(scene)
    payload = recorded_scene_payloads(observations, scene)[part]
    assert payload["collision_objects"][0]["id"] == part
    assert payload["declared_scene_identity"]["collision_objects"][0]["id"] == part + "/link/collision"
    assert payload["recorded_owner_commands"][0]["task_id"] == "environment_14"
    assert payload["calculation"]["capture_revision"] == 4
    payloads, _ = _scene_owner_records(observations, scene)
    assert payloads["parts/" + part] == payload
    assert (observations, scene) == original


@pytest.mark.parametrize("defect", ["different_run", "different_part", "different_task", "not_completed",
    "not_released", "missing_null", "compound_body", "ambiguous_scene", "acknowledged", "sdf_present"])
def test_released_scene_identity_does_not_guess_missing_or_conflicting_bindings(defect):
    from cais_spade_llm.recovery_framework.kmr_recovery import recorded_scene_payloads

    observations, scene, part = released_scene_fixture()
    transition = observations["transitions"][0]
    step = transition["observations"]["observations"]["primitive_results"][0]
    if defect == "different_run":
        transition["acknowledgement"]["run_id"] = "other"
    elif defect == "different_part":
        transition["acknowledgement"]["part_name"] = "RGOCG4-50_Round_4mm"
    elif defect == "different_task":
        step["task_id"] = "other"
    elif defect == "not_completed":
        step["status"] = "running"
    elif defect == "not_released":
        step["result"]["attached"] = True
    elif defect == "missing_null":
        del step["result"]["held_part"]
    elif defect == "compound_body":
        observations["physical_evidence"]["physics_evidence"]["models"][part]["links"]["link"]["collisions"].append({"id": "second"})
    elif defect == "ambiguous_scene":
        scene["planning_scene"]["robot_state"]["attached_collision_objects"] = [{"object": {"id": part}}]
    elif defect == "acknowledged":
        observations["physical_evidence"]["parts"][part]["collision_scene_identity"]["collision_scene_acknowledged"] = True
    else:
        scene["planning_scene"]["world"]["collision_objects"].append({"id": part + "/link/collision"})
    assert recorded_scene_payloads(observations, scene) == {}


@pytest.mark.parametrize('held,target,expected', [(None, 'M1', 'empty'), (None, 'Storage', 'empty_return'),
                                               ('KET4_Square_4mm', 'M1', '')])
def test_owner_preview_selects_recorded_move_variant_from_custody(held, target, expected):
    from cais_spade_llm.recovery_framework.kmr_outline_evidence import _program

    event = {'event_id': 3, 'event_name': 'move_to_resource', 'program': {'steps': [{'id': 'loaded'}]},
             'program_variants': {'empty': {'steps': [{'id': 'empty'}]}, 'empty_return': {'steps': [{'id': 'return'}]}}}
    physical = {'snapshot': {'resources': {'KMR': {'held_part': held}}}}
    evidence = {'environment_models': {'KMR': {'events': [event]}}}
    declared = {'task': {'event_id': 3, 'parameters': {'target_resource': target}}}
    original = deepcopy((physical, evidence, declared))
    result = _program(declared, evidence, physical)
    assert result['selected_program_variant'] == expected
    assert result['program'] == (event['program_variants'][expected] if expected else event['program'])
    assert (physical, evidence, declared) == original
    if expected:
        del event['program_variants'][expected]
        with pytest.raises(ValueError, match='not captured'):
            _program(declared, evidence, physical)
    del physical['snapshot']['resources']['KMR']['held_part']
    with pytest.raises(ValueError, match='held_part'):
        _program(declared, evidence, physical)


def test_coordinate_parameters_use_requested_destination_without_fallback():
    from cais_spade_llm.recovery_framework.kmr_recovery import coordinate_parameters

    scene = {'machines': [{'resource_id': 'M1', 'KMR_docking_pose': [1., 2., 0., 0., 0., .3]},
                          {'resource_id': 'M2', 'KMR_docking_pose': [3., 4., 0., 0., 0., .5]}]}
    bindings = {'observed_resource_location': 'M2', 'resource_state': 'idle'}
    original = deepcopy((bindings, scene))
    parameters, sources = coordinate_parameters(bindings, scene)
    assert [parameters[key] for key in ('x', 'y', 'yaw')] == [3., 4., .5]
    assert all(row['ref'] == '/inputs/scene/machines/1/KMR_docking_pose' for row in sources.values())
    assert (bindings, scene) == original
    supplied, sources = coordinate_parameters({'observed_resource_location': None, 'x': -1., 'y': 0., 'yaw': .2}, scene)
    assert supplied['x'] == -1. and sources == {}
    for invalid in ({'observed_resource_location': None}, {'observed_resource_location': 'absent'},
                    dict(bindings, x=None), dict(bindings, x=True), dict(bindings, x=1.)):
        with pytest.raises(ValueError):
            coordinate_parameters(invalid, scene)


@pytest.mark.parametrize('failure', ['moving', 'source_changed', None])
def test_kmr_capture_keeps_raw_base_evidence_when_hold_cannot_be_established(monkeypatch, tmp_path, failure):
    import hashlib
    import sys
    from cais_spade_llm.recovery_framework.kmr_live_safety import capture_state

    installed = tmp_path / 'lib/cais_lab_robotics/kmr_base_controller.py'
    installed.parent.mkdir(parents=True)
    installed.write_text('recorded owner source')
    digest = hashlib.sha256(installed.read_bytes()).hexdigest()
    pose = [0., 0., 0., 0., 0., 0., 1.]
    config = {'root': 'KMR_base_link', 'reference_link': 'rg2_gripper_tcp', 'joint_names': ['joint_a1'],
              'interpolation': 'splines', 'description_sha256': hashlib.sha256(b'model').hexdigest()}
    physics = {'instance_id': 'capture', 'simulation_time': 4., 'observed_monotonic': 10.,
        'models': {'KMR': {'joints': {'joint_a1': {'position': .2}}, 'links': {'KMR_base_link': {
            'pose': pose, 'linear_velocity': [.1, 0., 0.], 'angular_velocity': [0., 0., .2]}}}}}
    controller = {'controller_source_sha256': 'old' if failure == 'source_changed' else digest,
                  'instance_id': 'capture', 'simulation_time': 4., 'has_active_goal': True,
                  'last_output': [.1, 0., .2]}
    original = deepcopy((physics, controller))
    reader = SimpleNamespace(parameter=lambda node, key: 'splines' if key == 'interpolation_method' else 'model',
        snapshot=lambda **_: deepcopy(physics), controller_state=lambda _: deepcopy(controller))
    observer = SimpleNamespace(motion_configuration=config,
        configuration={'description_node': 'description', 'controller_node': 'controller', 'model': 'KMR',
                       'physics_description_node': 'physics'},
        provider=SimpleNamespace(reader=reader, launch_id='capture'),
        capture=lambda **kwargs: {**deepcopy(kwargs['native']), 'observation_state': {}})
    monkeypatch.setitem(sys.modules, 'ament_index_python.packages', SimpleNamespace(get_package_prefix=lambda _: str(tmp_path)))
    monkeypatch.setattr('cais_spade_llm.resources.continuous_geometry.observed_joint_configuration', lambda *_: config)
    monkeypatch.setattr('cais_spade_llm.resources.continuous_motion.kinematic_transforms', lambda *_: {'rg2_gripper_tcp': pose})
    monkeypatch.setattr('cais_spade_llm.resources.continuous_motion.representative_pose', lambda value: deepcopy(value))
    calls = []
    def validate(**kwargs):
        calls.append(deepcopy(kwargs))
        if failure == 'moving':
            raise ValueError('Base controller has an active goal')
        return {'source': 'local fixture'}
    monkeypatch.setattr('cais_spade_llm.agents.central_controller.recovery_case_evidence.recorded_base_hold', validate)
    result = capture_state(SimpleNamespace(recovery_safety_observer=observer))
    hold = result['base_hold_evidence']
    assert hold['controller_observation'] == controller
    assert hold['controller_source_sha256'] == digest
    assert hold['status'] == ('supported' if failure is None else 'NEEDS_CONTEXT')
    if failure != 'source_changed':
        assert hold['linear_velocity'] == [.1, 0., 0.]
        assert hold['physics_source']['simulation_time'] == 4.
        assert len(calls) == 1
    else:
        assert calls == [] and 'differs' in hold['reason']
    assert (physics, controller) == original

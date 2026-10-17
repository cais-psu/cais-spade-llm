from __future__ import annotations


"""Fact selection and retrieval without narrowing validator authority."""

import asyncio
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import jsonschema
import pytest

from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
    multi_turn,
    multi_turn_context,
    multi_turn_outline_generation,
    multi_turn_prompts,
    multi_turn_validated_event,
)


@pytest.fixture
def context():
    jids = [f"recovery-resource-{number}@localhost" for number in (4, 3, 2)]
    names = ["gear_small", "KET4_Square_4mm", "gear_large"]
    resources = {jid: {"resource_jid": jid, "resource_state": "idle", "held_part": None,
                       "current_pose": {"x": i, "y": 0, "z": 1}} for i, jid in enumerate(jids)}
    resources[jids[0]]["resource_state"] = "failed"
    resources[jids[1]]["held_part"] = names[1]
    parts = {name: {"part_name": name, "part_state": "ready", "part_location": "Storage"} for name in names}
    parts[names[0]].update(part_state="misplaced", part_location=None)
    parts[names[1]].update(part_state="in_gripper", part_location=jids[1])
    entries = {jid: {"resource_jid": jid, "resource_type": "robot", "recovery_snapshot": deepcopy(row),
        "static_capabilities": {"reachability": ["assembly_board-v1"]},
        "recovery_des_model": {"state_variables": {"resource_state": {"scope": "resource", "domain": ["idle", "failed"]},
                                                  "task_ctx.part_name": {"private": True}}},
        "execution_primitive_catalog": [{"name": "move_to_named_pose", "primitive_kind": "actuate", "required_params": ["pose_name"]}],
        "pending_tasks": []} for jid, row in resources.items()}
    entries[jids[2]]["static_capabilities"]["reachability"] = ["Exit"]
    for jid, name in zip(jids[:2], names[:2], strict=True):
        entries[jid]["pending_tasks"] = [{"id": jid, "function_name": "place_approach", "params": {
            "part_name": name, "destination_location": "assembly_board-v1"}, "status": "pending"}]
    prepared = {"ra_jid": jids[0], "P_id": [names[0]], "offline_recovery": True,
        "source_observation_captured_at": "2026-10-08T00:00:00+00:00", "source_observation_run_id": "capture",
        "recovery_session": {"outline_context_retrieval": True}, "recovery_resources": entries,
        "llm_input": {"fault_event": {"focused_resource_jid": jids[0], "affected_part_names": [names[0]]},
            "part_facts": list(parts.values()), "observed_runtime_state": {"resources": list(resources.values())}},
        "grounding_context": {"parts": {}}, "part_tracker": deepcopy(parts)}
    session = multi_turn.build_multi_turn_session_seed(prepared)
    return prepared, session, jids, names


def test_unexpected_disruption_includes_shared_work_and_retains_alternatives(context):
    prepared, session, jids, names = context
    original = deepcopy(prepared)
    scope = multi_turn_context.select_outline_context(prepared, session)
    assert scope["resource_jids"] == sorted(jids[:2])
    assert scope["part_names"] == sorted(names[:2])
    assert {row["resource_jid"] for row in scope["resource_index"]} == set(jids)
    assert "pending_tasks" in scope["inclusion_reasons"][f"/resources/{jids[1]}"]
    session["current_phase"] = "outline"
    payload, prompt = multi_turn._build_phase_prompt(prepared, session)
    assert names[2] not in prompt
    assert jids[2] in prompt
    assert len(payload["session_state"]["symbolic_resources"]) == 2
    assert len(session["symbolic_resources"]) == 3
    assert len(prepared["recovery_resources"]) == 3
    assert prepared == original


@pytest.mark.parametrize('resource_type', ['robot', 'kmr', 'resource'])
@pytest.mark.parametrize('field', ['reachability', 'reachable_locations'])
def test_matching_capabilities_include_every_owner_without_retrieval_or_ranking(context, resource_type, field):
    prepared, session, jids, _ = context
    entry = prepared['recovery_resources'][jids[2]]
    entry.update(resource_type=resource_type, static_capabilities={field: ['assembly_board-v1']})
    original = deepcopy((prepared, session))
    scope = multi_turn_context.select_outline_context(prepared, session)
    assert scope['resource_jids'] == sorted(jids)
    prepared['recovery_resources'] = dict(reversed(list(prepared['recovery_resources'].items())))
    assert multi_turn_context.select_outline_context(prepared, session) == scope
    assert not session.get('outline_requested_refs')
    assert (prepared, session) == original


def test_conflicting_locations_remain_visible_and_both_participate_in_selection(context):
    prepared, session, jids, _ = context
    entry = prepared['recovery_resources'][jids[2]]
    entry['recovery_snapshot']['reachability'] = ['assembly_board-v1']
    scope = multi_turn_context.select_outline_context(prepared, session)
    assert jids[2] in scope['resource_jids']
    row = next(row for row in scope['resource_index'] if row['resource_jid'] == jids[2])
    assert row['conflicts']['reachability'] == {'static_capabilities': ['Exit'], 'recovery_snapshot': ['assembly_board-v1']}


def test_automatic_capabilities_are_concise_and_do_not_change_candidate_schema(context):
    prepared, session, jids, _ = context
    session.update(validation_contract_version=2, current_phase='outline')
    before = multi_turn_prompts.multi_turn_phase_response_schema('outline', validation_contract_version=2)
    payload, prompt = multi_turn._build_phase_prompt(prepared, session)
    assert prompt.count('Use recorded facts and request missing public context.') == 1
    section = prompt.split('Resource Capabilities\n\n', 1)[1].split('\n\nPublic state declarations', 1)[0]
    assert all(section.count(jid) == 1 for jid in jids)
    assert not payload['outline_served_context'] and not session.get('outline_requested_refs')
    assert payload['outline_resource_capabilities']
    assert multi_turn_prompts.multi_turn_phase_response_schema('outline', validation_contract_version=2) == before


@pytest.mark.parametrize('change', ['valid', 'label_only', 'wrong_resource', 'wrong_run', 'future_revision', 'location_conflict', 'failed_geometry'])
def test_historical_capabilities_require_measured_consistent_scoped_records(change):
    from cais_spade_llm.recovery_framework.recovery_observations import recorded_capability_evidence

    record = {'acknowledgement': {'resource_id': 'KMR', 'task_id': 't', 'run_id': 'capture',
        'revision': 1, 'event_name': 'move_to_resource', 'status': 'completed'},
        'after': {'KMR': {'resource_location': 'M2', 'held_part': 'part'}},
        'observations': {'resource_id': 'KMR', 'task_id': 't', 'status': 'completed', 'observations': {
            'resource_location': 'M2', 'part_name': 'part', 'scene_fingerprint': 'historical-scene',
            'primitive_results': [{'resource_id': 'KMR', 'task_id': 't', 'status': 'completed',
                'result': {'resource_location': 'M2', 'base_pose': [1., 2., 3.]},
                'observations': [{'operation': 'validated_transport_sweep', 'success': True}]}]}}}
    source = {'run_id': 'capture', 'context_revision': 2, 'evidence_source': 'gazebo', 'transitions': [record]}
    if change == 'label_only':
        record['observations']['observations']['primitive_results'] = []
    elif change == 'wrong_resource':
        record['observations']['resource_id'] = 'different'
    elif change == 'wrong_run':
        record['acknowledgement']['run_id'] = 'different'
    elif change == 'future_revision':
        record['acknowledgement']['revision'] = 3
    elif change == 'location_conflict':
        record['after']['KMR']['resource_location'] = 'M1'
    elif change == 'failed_geometry':
        record['observations']['observations']['primitive_results'][0]['observations'][0]['success'] = False
    original = deepcopy(source)
    result = recorded_capability_evidence(source)['KMR'][0]
    assert result['recorded_execution'] is (change in {'valid', 'failed_geometry'})
    assert bool(result['geometric_validation']) is (change == 'valid')
    assert result['current_proposal_validated'] is False
    assert result['scope']['scene_fingerprint'] == 'historical-scene'
    assert result['held_part'] == 'part'
    assert source == original


def test_capability_evidence_retrieval_filters_private_records(context):
    prepared, session, jids, _ = context
    prepared['recovery_resources'][jids[0]]['capability_evidence'] = [{
        'ref': '/transitions/1', 'resource_location': 'M1', 'recorded_execution': True,
        'private_sequence': 'PRIVATE_CAPABILITY', 'scope': {'part_name': 'gear_small', 'replacement_event': 'PRIVATE_CAPABILITY'},
        'checks': [{'operation': 'validated_transport_sweep', 'success': True, 'program': 'PRIVATE_CAPABILITY'}],
        'measurements': [{'base_pose': [1., 2., 3.], 'private_pose': 'PRIVATE_CAPABILITY'}]}]
    fact = multi_turn_context.resolve_outline_context(prepared, session, f'/resources/{jids[0]}/capability_evidence')
    assert 'PRIVATE_CAPABILITY' not in str(fact)
    assert fact['value'][0]['scope']['part_name'] == 'gear_small'


@pytest.mark.parametrize('rid', ['M1', 'M2', 'Conveyor', 'Storage', 'Buffer For Machined parts', '3D Printing Station'])
def test_public_condition_evaluation_support_uses_implemented_declarations_for_every_resource(rid):
    from cais_spade_llm.resources.environment_recovery import public_evaluation_support

    entry = {'recovery_des_model': {'resource_id': rid, 'state_variables': {'resource_state': {'scope': 'resource'}},
        'events': [{'program_status': 'implemented', 'updates': {'resource_state': {'set': 'ready'}},
                    'primitive_support': [{'primitive': 'declared'}]}]}}
    assert public_evaluation_support(entry) == {'public_conditions': True}
    entry['recovery_des_model']['events'][0]['program_status'] = 'unimplemented'
    assert public_evaluation_support(entry) == {}


def test_resource_index_retains_recorded_resource_id_without_selecting_it(context):
    prepared, session, jids, names = context
    prepared['observation_jids'] = {'ur5e-4': jids[0], 'ur5e-3': jids[1], 'ur5e-2': jids[2]}
    original = deepcopy(prepared)
    rows = multi_turn_context.resource_index(prepared)
    assert {row['resource_id']: row['resource_jid'] for row in rows} == prepared['observation_jids']
    assert prepared == original


def test_validated_prompt_keeps_recorded_rejections_and_filters_private_evidence(context):
    prepared, session, jids, _ = context
    session.update(validation_contract_version=2, current_phase="outline")
    reason = "pose is outside gripper_reach: xy_radius=0.7244 m, max=0.7000 m"
    row = {"resource_jid": jids[0], "event_name": "proposed_identifier", "constraint_code": "transition_feasibility_rejected",
        "evidence": {"capability_result": {"feasibility_status": "INFEASIBLE", "checks": [{
            "feasibility_status": "INFEASIBLE", "reason": reason, "checked_pose": {"x": .72, "y": 0., "z": 1.},
            "gripper_reach": {"frame": "world", "max_xy_radius_m": .7}}]},
            "private_sequence": "PRIVATE_SEQUENCE_SENTINEL"}}
    session["candidate_rejection_feedback"] = [deepcopy(row)]
    session["outline_validation_findings"] = [deepcopy(row)]
    _, prompt = multi_turn._build_phase_prompt(prepared, session)
    assert reason not in prompt and 'gripper_reach' not in prompt
    assert prompt.count('The calculated target is outside the recorded owner motion capability.') == 1
    assert "PRIVATE_SEQUENCE_SENTINEL" not in prompt
    session["outline_validation_findings"] = []
    _, after_retrieval = multi_turn._build_phase_prompt(prepared, session)
    assert reason not in after_retrieval
    assert 'The calculated target is outside the recorded owner motion capability.' in after_retrieval


def test_captured_requirements_are_visible_without_a_nominal_route(context):
    from cais_spade_llm.recovery_framework.recovery_observations import restore_recorded_public_facts

    prepared, session, jids, names = context
    source = {'requirements': {names[0]: [{'processesToComplete': [{'process': 'trim', 'result': 'square'}],
                                        'route_hints': 'PRIVATE_ROUTE'}]}}
    restore_recorded_public_facts(prepared, source)
    session.update(validation_contract_version=2, current_phase='outline')
    _, prompt = multi_turn._build_phase_prompt(prepared, session)
    goals = prompt.split('Recovery goals\n\n', 1)[1].split('\n\nResource Capabilities', 1)[0]
    assert 'processesToComplete' in goals and 'square' in goals
    assert 'PRIVATE_ROUTE' not in prompt and 'route_hints' not in prompt


@pytest.mark.parametrize('semantics', ['recorded_unfinished_tasks', None])
def test_nominal_assignment_explanation_preserves_explicit_goals_and_resource_choices(context, semantics):
    prepared, session, jids, _ = context
    session.update(validation_contract_version=2, current_phase='outline',
                   nominal_continuation_semantics=semantics)
    before = deepcopy((prepared, session))
    schema = multi_turn._get_response_schema('outline', session)
    _, prompt = multi_turn._build_phase_prompt(prepared, session)
    instruction = "A recorded task's resource assignment is not itself a goal;"
    assert prompt.count(instruction) == int(semantics == 'recorded_unfinished_tasks')
    if semantics == 'recorded_unfinished_tasks':
        explanation = prompt[prompt.index(instruction):].split('\n\n', 1)[0]
        assert 'accepted matching product-completion evidence' in explanation
        assert 'Explicit resource goals remain required.' in explanation
        assert all(jid not in explanation for jid in jids)
    assert 'Respect recorded resource availability.' in prompt
    assert schema == multi_turn._get_response_schema('outline', session)
    assert (prepared, session) == before


def test_public_guard_feedback_preserves_values_and_excludes_private_conditions():
    public = multi_turn_prompts._outline_public_evidence({'capability_result': {
        'feasibility_status': 'INFEASIBLE', 'guard_failures': [
            {'resource_id': 'KMR', 'field': 'resource_state', 'operator': 'equals',
             'expected': 'positioned', 'actual': 'carrying', 'declaration_fingerprint': 'source',
             'private_program': 'PRIVATE_PROGRAM'},
            {'resource_id': 'KMR', 'field': 'task_ctx.private', 'operator': 'equals',
             'expected': 'PRIVATE_CONDITION', 'actual': None, 'declaration_fingerprint': 'source'}]}},
        {'resource_state'})
    assert public['capability_result']['guard_failures'] == [{'resource_id': 'KMR',
        'field': 'resource_state', 'operator': 'equals', 'expected': 'positioned',
        'actual': 'carrying', 'declaration_fingerprint': 'source'}]
    assert 'PRIVATE_' not in str(public)


def test_validated_renderer_retains_discovery_without_unrelated_detailed_states(context):
    prepared, session, jids, names = context
    session.update(validation_contract_version=2, current_phase="outline")
    scope = multi_turn_context.select_outline_context(prepared, session)
    payload = {"llm_input": deepcopy(prepared["llm_input"]),
               "recovery_resources": deepcopy(prepared["recovery_resources"]), "outline_context": scope}
    prompt = multi_turn_prompts._render_validated_event_prompt(payload, session)
    resource_section = prompt.split("Current resource states\n\n", 1)[1].split("\n\nCurrent part states", 1)[0]
    assert jids[2] not in resource_section
    assert jids[2] in prompt and names[2] not in prompt
    fact = multi_turn_context.resolve_outline_context(prepared, session, f"/resources/{jids[2]}/snapshot")
    assert fact["value"]["current_pose"] == {"x": 2, "y": 0, "z": 1}


def test_condition_pairs_keep_whole_public_counterexamples_without_private_fields():
    public_pair = [{'field': 'held_part', 'actual': 'gear_small'}, {'field': 'resource_state', 'actual': 'idle'}]
    evidence = {'condition_support': {
        'conditions': [{**row, 'supported': True} for row in public_pair],
        'combination_supported': False,
        'unsupported_combinations': [
            [{**row, 'private_program': 'PRIVATE_PROGRAM'} for row in public_pair],
            [public_pair[0], {'field': 'task_ctx.part_name', 'actual': 'PRIVATE_VALUE'}]],
        'source': {'descriptor_fingerprint': 'recorded-descriptor'}}}
    before = deepcopy(evidence)
    rendered = multi_turn_prompts._outline_public_evidence(evidence, {'held_part', 'resource_state'})
    assert rendered['condition_support']['unsupported_combinations'] == [public_pair]
    assert 'PRIVATE_' not in json.dumps(rendered)
    assert evidence == before


def test_validated_prompt_keeps_recent_rejections_after_acceptance_clears_current_feedback(context):
    prepared, session, jids, _ = context
    session.update(validation_contract_version=2, current_phase="outline", candidate_rejection_feedback=[],
        outline_validation_findings=[])
    session['turns'] = [{'phase': 'outline', 'turn_index': index, 'decision': 'need_revision',
        'next_transition': {'outline_id': f'rejected_{index}', 'resource_jid': jids[0],
            'event_name': 'proposed_event', 'expected_end_state': {'resource_state': 'idle'},
            'private_program': 'PRIVATE_PROGRAM'},
        'validation_findings': [{'resource_jid': jids[0], 'constraint_code': 'unsupported_successor_condition',
            'reason': 'PRIVATE_VALIDATOR_PROSE', 'expected_end_state': {'resource_state': 'idle'}}]}
        for index in (1, 2, 3)]
    before = deepcopy((prepared, session))
    _, prompt = multi_turn._build_phase_prompt(prepared, session)
    history = prompt.split('Recent rejected proposals (checked in their recorded starting states)\n\n', 1)[1]
    assert 'rejected_1' not in history and 'rejected_2' in history and 'rejected_3' in history
    assert 'PRIVATE_' not in prompt
    assert (prepared, session) == before
    session['candidate_rejection_feedback'] = deepcopy(session['turns'][-1]['validation_findings'])
    _, current = multi_turn._build_phase_prompt(prepared, session)
    assert 'Recent rejected proposals (checked in their recorded starting states)' not in current


def test_initial_declarations_omit_repeated_provenance_without_losing_fields(context):
    prepared, session, jids, _ = context
    session.update(validation_contract_version=2, current_phase="outline")
    fact = multi_turn_context.resolve_outline_context(prepared, session, f"/resources/{jids[0]}/state_variables")
    original = deepcopy(fact)
    payload = {"llm_input": prepared["llm_input"], "recovery_resources": prepared["recovery_resources"],
               "outline_resource_fields": {jids[0]: fact}, "outline_context": multi_turn_context.select_outline_context(prepared, session)}
    prompt = multi_turn_prompts._render_validated_event_prompt(payload, session)
    rendered = json.loads(prompt.split("Public state declarations\n\n", 1)[1].split("\n\nSafety Rules", 1)[0])
    assert set(rendered[jids[0]]) == set(fact["value"])
    assert rendered[jids[0]]["resource_state"] == {
        "type": "string", "scope": "resource", "domain": ["idle", "failed"]}
    assert "domain_complete" not in rendered[jids[0]]["resource_state"]
    assert multi_turn_context.resolve_outline_context(prepared, session, f"/resources/{jids[0]}/state_variables") == fact
    assert "revision" not in rendered[jids[0]]
    assert fact == original


@pytest.mark.parametrize("resource_type", ["robot", "machine", "conveyor"])
@pytest.mark.parametrize("domain_complete", [False, True])
def test_recorded_domains_remain_facts_without_restricting_candidate_schema(context, resource_type, domain_complete):
    prepared, session, jids, _ = context
    session.update(validation_contract_version=2, current_phase="outline")
    entry = prepared["recovery_resources"][jids[0]]
    entry["resource_type"] = resource_type
    declaration = entry["recovery_des_model"]["state_variables"]["resource_state"]
    declaration["domain_complete"] = domain_complete
    entry["recovery_des_model"]["state_variables"]["task_ctx.part_name"]["domain"] = ["PRIVATE_VALUE"]
    session["recovery_des_models"][jids[0]] = deepcopy(entry["recovery_des_model"])
    before = deepcopy((prepared, session))
    schema = multi_turn._get_response_schema("outline", session)
    _, prompt = multi_turn._build_phase_prompt(prepared, session)
    rendered = json.loads(prompt.split("Public state declarations\n\n", 1)[1].split("\n\nSafety Rules", 1)[0])
    assert rendered[jids[0]]["resource_state"]["domain"] == declaration["domain"]
    assert rendered[jids[0]]["resource_state"]["domain_complete"] is domain_complete
    assert "PRIVATE_VALUE" not in prompt
    assert schema == multi_turn._get_response_schema("outline", session)
    assert (prepared, session) == before


def test_validated_prompt_uses_one_index_and_retrieval_list_and_serves_details_once(context):
    prepared, session, jids, _ = context
    session.update(validation_contract_version=2, current_phase="outline")
    entry = prepared["recovery_resources"][jids[0]]
    entry["recovery_des_model"]["state_variables"]["active_job"] = {
        "scope": "resource", "type": ["string", "null"], "domain": ["environment_23"],
        "domain_complete": False, "description": "Identity of the currently processing job.",
    }
    entry["static_capabilities"].update(program_options={"circle": {"required_tool": "simulation_trim_tool"}})
    session["recovery_des_models"][jids[0]] = deepcopy(entry["recovery_des_model"])
    before = deepcopy(prepared)
    payload, initial = multi_turn._build_phase_prompt(prepared, session)
    schema = multi_turn._get_response_schema("outline", session)
    assert initial.count("Resource Capabilities") == 1
    index = initial.split('Resource Capabilities\n\n', 1)[1].split('\n\nPublic state declarations', 1)[0]
    assert all(index.count(jid) == 1 for jid in jids)
    assert initial.count("Context retrieval") == 1
    assert initial.count("expected_end_state may contain only the intended conditions") == 1
    assert "currently processing job" in initial
    assert "simulation_trim_tool" not in initial
    assert all(jid in initial for jid in jids)
    refs = [f"/resources/{jids[0]}/state_variables", f"/resources/{jids[0]}/static_capabilities", "/resources"]
    session["outline_requested_refs"] = refs + refs
    payload, retrieved = multi_turn._build_phase_prompt(prepared, session)
    assert set(payload["outline_served_context"]) == set(refs)
    assert retrieved.count("currently processing job") == 1
    assert retrieved.count("simulation_trim_tool") == 1
    assert "Value already shown in Resource Capabilities." in retrieved
    assert schema == multi_turn._get_response_schema("outline", session)
    assert "pause_job" not in retrieved and "start_job" not in retrieved
    assert prepared == before


def test_recorded_processing_summary_moves_to_retrieved_record_without_repetition(context):
    from cais_spade_llm.resources.machine.machine_jobs import public_processing_observations

    prepared, session, jids, names = context
    session.update(validation_contract_version=2, current_phase="outline")
    record = {"resource_id": "M2", "run_id": "capture", "revision": 1, "model_fingerprint": "recorded-model",
        "active_job": "environment_23", "selected_program": "circle", "jobs": {"environment_23": {
            "job_id": "environment_23", "part_name": names[0], "job_state": "running", "process": "trim",
            "result": "circle", "processing_fraction": .125, "process_completed": False}}}
    prepared["llm_input"]["fault_event"]["checkpoint"] = {"processing_observations": {"M2": record}}
    prepared["recovery_resources"][jids[0]]["static_capabilities"]["processing_observations"] = public_processing_observations(record)
    before = deepcopy(prepared)
    _, initial = multi_turn._build_phase_prompt(prepared, session)
    assert initial.count('"processing_fraction": 0.125') == 1
    assert initial.count('These are capture-time job records; use Current resource states after accepted events.') == 1
    ref = f"/resources/{jids[0]}/static_capabilities"
    session["outline_requested_refs"] = [ref]
    payload, retrieved = multi_turn._build_phase_prompt(prepared, session)
    assert retrieved.count('"processing_fraction": 0.125') == 1
    assert payload["outline_served_context"][ref]["value"]["processing_observations"] == public_processing_observations(record)
    assert prepared == before


@pytest.mark.parametrize("phase", ["grounding", "outline"])
def test_selected_safety_requirements_appear_once_without_loading_validation(context, phase):
    from cais_spade_llm.recovery_framework.part_slippage_recovery import _prompt_safety_rules

    prepared, session, jids, _ = context
    rules = _prompt_safety_rules([
        "cais_spade_llm/specification/safety/safety_assembly_board-v1_mutex.txt",
        "cais_spade_llm/specification/safety/safety_assembly_board-v1_gear_small_before_KET4_Square_4mm.txt"])
    prepared["prompt_safety_rules"] = deepcopy(rules)
    prepared["llm_input"]["loaded_safety_rules"] = []
    session["current_phase"] = phase
    before = deepcopy(prepared)
    payload, prompt = multi_turn._build_phase_prompt(prepared, session)
    assert len(rules) == 2
    for rule in rules:
        assert prompt.count(f"- {rule['id']}: {rule['requirement']}") == 1
        assert rule["source_sha256"] not in prompt
    assert payload["llm_input"]["prompt_safety_rules"] == rules
    assert payload["llm_input"]["loaded_safety_rules"] == []
    assert jids[2] in prompt
    assert prepared == before


def test_safety_renderer_preserves_legacy_rules_and_explicit_empty_selection():
    legacy = {"loaded_safety_rules": [{"id": "SAFE_legacy", "raw_text": "Legacy exact requirement"}]}
    assert multi_turn_prompts._compact_safety_rules(legacy) == "- SAFE_legacy: Legacy exact requirement"
    assert multi_turn_prompts._compact_safety_rules({**legacy, "prompt_safety_rules": []}) == "(none)"
    rule = {"id": "SAFE_shared_area_mutex", "requirement": "Exact requirement"}
    assert multi_turn_prompts._compact_safety_rules({"prompt_safety_rules": [rule, rule]}) == "- SAFE_shared_area_mutex: Exact requirement"


def test_retrieval_expands_prompt_and_never_commits_an_event(context):
    prepared, session, jids, names = context
    response = {"thought": "Retrieve recorded facts", "decision": "need_context", "context_requests": [f"/resources/{jids[2]}/snapshot", "/parts"]}
    before = deepcopy(session["accepted_outline_prefix"])
    decision, turn = asyncio.run(multi_turn_outline_generation._handle_outline_phase(
        session_state=session, prepared_recovery_request=prepared, planner=object(), parsed_response=response))
    assert decision == "need_context"
    assert multi_turn.transition_multi_turn_phase("outline", decision) == "outline"
    assert session["accepted_outline_prefix"] == before
    assert turn["served_context"][response["context_requests"][0]]["value"]["current_pose"] == {"x": 2, "y": 0, "z": 1}
    assert jids[2] in multi_turn_context.select_outline_context(prepared, session)["resource_jids"]
    assert prepared["recovery_resources"][jids[2]]["recovery_snapshot"]["current_pose"]["x"] == 2


def test_null_prepared_pose_does_not_hide_recorded_pose_or_change_projected_custody(context):
    prepared, session, _, names = context
    pose = {"x": 0.0, "y": 0.3, "z": 1.025}
    prepared["grounding_context"]["parts"][names[0]] = {"observed_pose": pose}
    session["symbolic_parts"][names[0]].update(observed_pose=None, current_holder_resource_jid=None)
    session["accepted_outline_prefix"] = [{"outline_id": "RECOVERY_SEQ1"}]
    fact = multi_turn_context.resolve_outline_context(prepared, session, f"/parts/{names[0]}/observed_pose")
    assert fact["value"] == pose and fact["source"] == "saved_observations"
    snapshot = multi_turn_context.resolve_outline_context(prepared, session, f"/parts/{names[0]}/snapshot")
    assert snapshot["source"] == "projected_state"
    assert snapshot["value"]["current_holder_resource_jid"] is None
    assert snapshot["field_sources"]["observed_pose"] == "saved_observations"


@pytest.mark.parametrize("projected_pose", [None, {"x": 0.2, "y": 0.4, "z": 1.1}])
def test_projected_part_move_does_not_restore_historical_pose(context, projected_pose):
    prepared, session, _, names = context
    part = names[0]
    historical = {"x": 0, "y": 0.3, "z": 1.025}
    prepared["grounding_context"]["parts"][part] = {"observed_pose": historical}
    prepared["part_tracker"][part]["observed_pose"] = historical
    session["observation_store"] = {"recorded": {"part_name": part, "pose": historical}}
    session["symbolic_parts"][part].update(observed_pose=None, observed_pose_current=False,
                                          last_observed_pose=historical, projected_pose=projected_pose)
    session.update(current_phase="outline", outline_mode="incremental_candidates_validated",
                   accepted_outline_prefix=[{"outline_id": "RECOVERY_SEQ1", "part_name": part}])
    before = deepcopy(prepared)
    snapshot = multi_turn_context.resolve_outline_context(prepared, session, f"/parts/{part}/snapshot")
    assert snapshot["value"]["observed_pose"] is None
    assert snapshot["value"]["last_observed_pose"] == historical
    assert snapshot["value"]["projected_pose"] == projected_pose
    assert snapshot["field_sources"]["observed_pose"] == "unavailable_after_projected_transition"
    assert snapshot["field_sources"]["last_observed_pose"] == "saved_observations"
    assert snapshot["field_sources"]["projected_pose"] == "projected_state"
    with pytest.raises(ValueError, match="NEEDS_CONTEXT.*after the projected part transition"):
        multi_turn_context.resolve_outline_context(prepared, session, f"/parts/{part}/observed_pose")
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    assert "last_observed_pose is historical; projected_pose is unexecuted" in prompt
    assert '"observed_pose": null' in prompt and '"observed_pose_current": false' in prompt
    assert prepared == before


def test_requested_drop_pose_never_becomes_an_observation(context):
    prepared, session, _, names = context
    prepared["failure"] = {"requested_drop_pose": {"x": 0, "y": 0.3, "z": 1.04}}
    prepared["part_tracker"][names[0]]["observed_pose"] = None
    with pytest.raises(ValueError, match="NEEDS_CONTEXT"):
        multi_turn_context.resolve_outline_context(prepared, session, f"/parts/{names[0]}/observed_pose")


def test_explicit_unfinished_task_dependencies_are_selected(context):
    prepared, session, jids, names = context
    prepared["recovery_resources"][jids[2]]["pending_tasks"] = [{
        "id": "unfinished-task", "predecessors": [jids[0]], "params": {"part_name": names[2]}}]
    scope = multi_turn_context.select_outline_context(prepared, session)
    assert jids[2] in scope["resource_jids"]
    assert names[2] in scope["part_names"]


@pytest.mark.parametrize("field", ["source_resource", "target_resource"])
def test_unfinished_transport_at_affected_part_location_is_selected(context, field):
    prepared, session, jids, names = context
    for row in (prepared["llm_input"]["part_facts"][0], prepared["part_tracker"][names[0]],
                session["symbolic_parts"][names[0]]):
        row["part_location"] = "M2"
    prepared["recovery_resources"][jids[2]]["pending_tasks"] = [{
        "id": "environment_22", "function_name": "move_to_resource", "status": "pending",
        "params": {field: "M2", "resource_id": "KMR"}, "predecessors": [], "successors": [],
    }]
    original = deepcopy((prepared, session))
    scope = multi_turn_context.select_outline_context(prepared, session)
    assert jids[2] in scope["resource_jids"]
    assert "pending_tasks" in scope["inclusion_reasons"][f"/resources/{jids[2]}"]
    assert (prepared, session) == original
    prepared["recovery_resources"][jids[2]]["pending_tasks"][0]["params"][field] = "Exit"
    assert jids[2] not in multi_turn_context.select_outline_context(prepared, session)["resource_jids"]


@pytest.mark.parametrize("ref", ["/evaluator_data", "/recovery_des_models", "/resources/recovery-resource-4@localhost/task_ctx.part_name",
                                 "/parts/gear_small/observed_pose", "/resources/unknown/snapshot"])
def test_missing_or_private_context_never_invokes_live_observation(context, monkeypatch, ref):
    prepared, session, _, _ = context
    live = AsyncMock(side_effect=AssertionError("offline retrieval invoked a live observation"))
    monkeypatch.setattr(multi_turn, "_handle_grounding_phase", live)
    response = {"thought": "Retrieve recorded facts", "decision": "need_context", "context_requests": [ref]}
    _, turn = asyncio.run(multi_turn_context.handle_outline_context(prepared=prepared, session=session,
                                                                  response=response, planner=object()))
    assert turn["context_errors"] and not turn["served_context"]
    live.assert_not_called()
    asyncio.run(multi_turn_context.handle_outline_context(prepared=prepared, session=session, response=response, planner=object()))
    assert session["status"] == "NEEDS_CONTEXT"


def test_public_snapshot_and_prompt_exclude_private_canaries(context):
    prepared, session, jids, _ = context
    canary = "EVALUATOR_PRIVATE_SEQUENCE_79831"
    prepared["evaluator_data"] = {"golden_answer": canary}
    prepared["recovery_resources"][jids[0]]["recovery_snapshot"]["task_ctx.part_name"] = canary
    prepared["recovery_resources"][jids[0]]["recovery_des_model"]["events"] = [{"event_name": canary}]
    value = multi_turn_context.resolve_outline_context(prepared, session, f"/resources/{jids[0]}/snapshot")
    assert canary not in str(value)
    session["current_phase"] = "outline"
    assert canary not in multi_turn._build_phase_prompt(prepared, session)[1]


@pytest.mark.parametrize("scenario", [
    "Conveyor breakdown", "ur5e-1 breakdown",
    "Machining breakdown during part processing", "Part slippage",
])
@pytest.mark.parametrize("phase", ["grounding", "outline"])
def test_all_cases_keep_evaluation_routes_and_owner_records_private(context, scenario, phase):
    prepared, session, jids, _ = context
    canary = "PRIVATE_REFERENCE_RECOVERY_73841"
    private = {"nominal_sequence": [canary], "reference_recovery": [canary],
               "evaluator_data": {"expected_events": [canary]}, "route_hints": [canary],
               "operator_feedback": canary, "fixture_comments": canary}
    prepared.update(deepcopy(private), scenario=scenario, validation_contract_version=2)
    prepared["llm_input"]["fault_event"].update(deepcopy(private))
    entry = prepared["recovery_resources"][jids[0]]
    entry.update(deepcopy(private))
    entry["recovery_snapshot"].update(deepcopy(private))
    entry["recovery_des_model"]["private_owner_records"] = deepcopy(private)
    entry["static_capabilities"].update(deepcopy(private))
    entry["static_capabilities"]["capabilities"] = {"trim": {"supported_results": ["square"],
                                                             **deepcopy(private)}}
    session.update(current_phase=phase, validation_contract_version=2)
    before = deepcopy(prepared)
    payload, prompt = multi_turn._build_phase_prompt(prepared, session)
    assert canary not in prompt
    for ref in ("/resources", "/parts", f"/resources/{jids[0]}/snapshot",
                f"/resources/{jids[0]}/static_capabilities",
                f"/resources/{jids[0]}/state_variables", f"/resources/{jids[0]}/pending_tasks"):
        assert canary not in json.dumps(multi_turn_context.resolve_outline_context(prepared, session, ref))
    for ref in ("/nominal_sequence", "/evaluator_data", "/reference_recovery", "/route_hints",
                f"/resources/{jids[0]}/private_owner_records"):
        with pytest.raises(ValueError):
            multi_turn_context.resolve_outline_context(prepared, session, ref)
    assert prepared == before


def test_scenario_metadata_does_not_change_prompt_schema_or_resource_selection(context):
    prepared, session, _, _ = context
    session.update(current_phase="outline", validation_contract_version=2)
    original = deepcopy(prepared)
    prompts, scopes, schemas = [], [], []
    for scenario in ("Conveyor breakdown", "ur5e-1 breakdown",
                     "Machining breakdown during part processing", "Part slippage"):
        prepared["scenario"] = scenario
        prepared["evaluation_condition"] = {"scenario": scenario, "preferred_resource": "PRIVATE"}
        payload, prompt = multi_turn._build_phase_prompt(prepared, session)
        prompts.append(prompt)
        scopes.append(multi_turn_context.select_outline_context(prepared, session))
        schemas.append(multi_turn._get_response_schema("outline", session))
    assert all(prompt == prompts[0] for prompt in prompts)
    assert all(scope == scopes[0] for scope in scopes)
    assert all(schema == schemas[0] for schema in schemas)
    assert prepared["recovery_resources"] == original["recovery_resources"]


def test_unavailable_resource_remains_discoverable_with_recorded_availability(context):
    prepared, session, jids, _ = context
    prepared["recovery_resources"][jids[2]]["recovery_snapshot"]["availability"] = False
    index = multi_turn_context.resource_index(prepared)
    row = next(row for row in index if row["resource_jid"] == jids[2])
    assert row["availability"] is False
    fact = multi_turn_context.resolve_outline_context(prepared, session, f"/resources/{jids[2]}/snapshot")
    assert fact["value"]["availability"] is False
    assert {row["resource_jid"] for row in index} == set(jids)


def test_prompt_keeps_recorded_locations_separate_from_poses_and_projections(context):
    prepared, session, jids, names = context
    session.update(current_phase="outline", outline_mode="incremental_candidates_validated")
    session["symbolic_resources"][jids[0]]["resource_location"] = "Storage"
    prepared["recovery_resources"][jids[0]]["recovery_snapshot"]["current_pose_captured_at"] = 123.0
    session["symbolic_parts"][names[0]]["location_basis"] = "sensor_observation"
    original = deepcopy(prepared)
    payload, prompt = multi_turn._build_phase_prompt(prepared, session)
    assert '"current_pose_captured_at": 123.0' in prompt
    assert '"location_basis": "sensor_observation"' in prompt
    assert '"resource_location": "Storage"' in prompt
    assert "resource_location and part_location are symbolic; use observed_pose for measured part positions" in prompt
    assert "recorded before recovery" in prompt
    assert payload["outline_observation_source"]["source_run_id"] == "capture"
    assert "null: not recorded" in prompt
    session["accepted_outline_prefix"] = [{"outline_id": "RECOVERY_SEQ1", "resource_jid": jids[0]}]
    _, projected = multi_turn._build_phase_prompt(prepared, session)
    assert "projected after the accepted transitions, not newly observed" in projected
    assert "Accepted Transition Prefix (projected; not executed)" in projected
    assert "These transitions have already been applied" not in projected
    assert prepared == original


def test_prompt_retains_all_retrieved_facts_but_displays_repeated_pose_once(context):
    prepared, session, jids, names = context
    session.update(current_phase="outline", outline_mode="incremental_candidates_validated")
    refs = [f"/resources/{jids[0]}/current_pose", f"/resources/{jids[0]}/snapshot"]
    async def retrieve(ref):
        await multi_turn_context.handle_outline_context(prepared=prepared, session=session,
            response={"thought": "Read the saved fact", "decision": "need_context", "context_requests": [ref]},
            planner=object())
    for ref in refs:
        asyncio.run(retrieve(ref))
    payload, prompt = multi_turn._build_phase_prompt(prepared, session)
    assert set(payload["outline_served_context"]) == set(refs)
    assert all(ref in prompt for ref in refs)
    pose = multi_turn_prompts._inline_json(session["symbolic_resources"][jids[0]]["current_pose"])
    assert prompt.count(pose) == 1
    assert "Value already shown in Current DES State: current_pose" in prompt
    assert "Fields already shown in Current DES State:" in prompt
    assert "Shared source metadata:" in prompt
    assert payload["outline_served_context"][refs[0]]["value"] == session["symbolic_resources"][jids[0]]["current_pose"]
    assert jids[2] in prompt and names[2] not in prompt
    assert prompt.count('"resource_type": "robot"') == len(jids)


def test_no_progress_feedback_does_not_instruct_larger_claimed_effects(context):
    prepared, session, _, _ = context
    session.update(current_phase="outline", outline_mode="incremental_candidates_validated",
        candidate_rejection_feedback=[{"constraint_code": "no_progressing_candidate", "reason": "Recorded continuation was not established"}])
    _, prompt = multi_turn._build_phase_prompt(prepared, session)
    assert "no_progressing_candidate" in prompt
    assert "Recorded checks did not establish recovery progress" in prompt
    assert "request missing facts when needed" in prompt
    assert "Do not claim additional effects merely to satisfy this check" in prompt
    assert "revise its symbolic expected_end_state" not in prompt
    assert "materially different effect" not in prompt


@pytest.mark.parametrize("phase", ["grounding", "outline", "primitive_generation"])
def test_brief_explanations_are_prompt_guidance_without_truncation(context, phase):
    prepared, session, _, _ = context
    session["current_phase"] = phase
    payload = multi_turn_prompts.build_multi_turn_phase_prompt_input(
        phase=phase, llm_input=prepared["llm_input"], session_state=session,
        recovery_resources=prepared["recovery_resources"])
    prompt = multi_turn_prompts.render_multi_turn_phase_prompt(payload)
    assert "at most two short sentences" in prompt
    if phase != "grounding":
        assert "one short sentence" in prompt
    schema = multi_turn_prompts.multi_turn_phase_response_schema("grounding")["schema"]
    jsonschema.validate({"thought": "Recorded facts. " * 100, "decision": "grounded", "observe_requests": []}, schema)


def test_retrieved_projection_and_revision_are_recomputed(context):
    prepared, session, jids, _ = context
    ref = f"/resources/{jids[0]}/resource_state"
    saved = multi_turn_context.resolve_outline_context(prepared, session, ref)
    session["accepted_outline_prefix"] = [{"resource_jid": jids[0]}]
    session["symbolic_resources"][jids[0]]["resource_state"] = "idle"
    projected = multi_turn_context.resolve_outline_context(prepared, session, ref)
    assert projected["source"] == "projected_state"
    assert saved["value"] == "failed" and projected["value"] == "idle"
    assert saved["revision"] != projected["revision"]
    assert prepared["recovery_resources"][jids[0]]["recovery_snapshot"]["resource_state"] == "failed"
    snapshot = multi_turn_context.resolve_outline_context(prepared, session, f"/resources/{jids[0]}/snapshot")
    assert snapshot["source"] == "projected_state"
    assert snapshot["field_sources"]["current_pose"] == "saved_observations"


def test_stale_live_revision_is_rejected(context):
    prepared, session, _, _ = context
    prepared.update(offline_recovery=False, source_context_revision=1)
    planner = SimpleNamespace(product_agent=SimpleNamespace(environment_runtime=SimpleNamespace(context=SimpleNamespace(revision=2))))
    with pytest.raises(ValueError, match="observations changed"):
        asyncio.run(multi_turn_context.handle_outline_context(prepared=prepared, session=session,
            response={"thought": "Retrieve recorded facts", "decision": "need_context", "context_requests": ["/parts"]}, planner=planner))


def test_observed_empty_custody_is_not_treated_as_a_missing_fact(context):
    prepared, session, jids, _ = context
    fact = multi_turn_context.resolve_outline_context(prepared, session, f"/resources/{jids[0]}/held_part")
    assert fact["value"] is None
    assert fact["source"] == "saved_observations"


@pytest.mark.parametrize("accepted_prefix", [[], [{"outline_id": "RECOVERY_SEQ1"}]])
def test_pending_tasks_are_retrievable_without_exposing_internal_fields(context, accepted_prefix):
    prepared, session, jids, _ = context
    session["accepted_outline_prefix"] = deepcopy(accepted_prefix)
    task = prepared["recovery_resources"][jids[1]]["pending_tasks"][0]
    task["evaluator_data"] = "PRIVATE_CANARY"
    fact = multi_turn_context.resolve_outline_context(prepared, session, f"/resources/{jids[1]}/pending_tasks")
    assert fact["source"] == "saved_observations"
    assert fact["captured_at"] == prepared["source_observation_captured_at"]
    assert fact["value"][0]["function_name"] == "place_approach"
    assert fact["value"][0]["params"]["part_name"] == "KET4_Square_4mm"
    assert "PRIVATE_CANARY" not in str(fact)


@pytest.mark.parametrize("mode", ["single_pass", "incremental", "incremental_validated", "incremental_candidates_validated"])
def test_schema_accepts_retrieval_and_legacy_candidates(mode):
    schema = multi_turn_prompts.multi_turn_phase_response_schema("outline", outline_mode=mode,
        outline_context_retrieval=True, recovery_selection_mode="neurosymbolic")["schema"]
    jsonschema.validate({"thought": "A recorded pose is required", "decision": "need_context", "context_requests": ["/parts"]}, schema)
    jsonschema.validate({"thought": "", "decision": "need_context", "context_requests": []}, schema)
    for invalid in ({"thought": "", "decision": "need_context", "context_requests": [False]},):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)
    original = multi_turn_prompts.multi_turn_phase_response_schema("outline", outline_mode=mode,
        recovery_selection_mode="neurosymbolic")["schema"]
    assert schema["required"] == ["thought"]
    assert schema["type"] == "object"
    assert not {"anyOf", "oneOf", "allOf", "enum", "const", "not"} & schema.keys()
    assert all(schema["properties"][key] == value for key, value in original["properties"].items())


@pytest.mark.parametrize("response", [
    {"thought": "", "decision": "need_context", "context_requests": []},
    {"thought": "", "decision": "need_context", "context_requests": [False]},
    {"thought": "", "decision": "need_context", "context_requests": ["/parts"], "candidate_events": []},
    {"thought": "", "decision": "need_context", "context_requests": ["/parts"], "next_transition": {}},
    {"thought": "", "decision": "need_context", "context_requests": ["/parts"], "selected_candidate_index": 0},
    {"decision": "need_context", "context_requests": ["/parts"]},
    {"thought": None, "decision": "need_context", "context_requests": ["/parts"]},
])
def test_local_validation_rejects_mixed_or_invalid_retrieval_before_serving(context, monkeypatch, response):
    prepared, session, _, _ = context
    resolve = Mock(side_effect=AssertionError("Invalid response retrieved context"))
    monkeypatch.setattr(multi_turn_context, "resolve_outline_context", resolve)
    decision, turn = asyncio.run(multi_turn_outline_generation._handle_outline_phase(
        session_state=session, parsed_response=response, prepared_recovery_request=prepared, planner=object()))
    assert decision == "need_context"
    assert turn["context_errors"] and turn["served_context"] == {}
    assert session["accepted_outline_prefix"] == []
    assert multi_turn.transition_multi_turn_phase("outline", decision) == "outline"
    resolve.assert_not_called()


@pytest.mark.parametrize("mode", ["single_pass", "incremental", "incremental_validated", "incremental_candidates_validated"])
def test_local_candidate_validation_retains_original_required_fields(context, monkeypatch, mode):
    prepared, session, _, _ = context
    session.update(outline_mode=mode, recovery_selection_mode="pure_llm")
    schema = multi_turn._get_response_schema("outline", {**session, "outline_context_retrieval": False})["schema"]
    handler_name = {
        "single_pass": "_handle_outline_single_pass", "incremental": "_handle_outline_incremental",
        "incremental_validated": "_handle_outline_incremental_validated",
        "incremental_candidates_validated": "_handle_outline_incremental_candidates_validated",
    }[mode]
    handler = AsyncMock(return_value=("need_revision", {}))
    monkeypatch.setattr(multi_turn_outline_generation, handler_name, handler)
    candidate = {key: [] for key in schema["required"]}
    candidate["thought"] = "Use recorded state"
    if "selected_candidate_index" in candidate:
        candidate["selected_candidate_index"] = 0
    async def run():
        for missing in schema["required"]:
            response = {key: value for key, value in candidate.items() if key != missing}
            decision, turn = await multi_turn_outline_generation._handle_outline_phase(
                session_state=session, parsed_response=response, prepared_recovery_request=prepared, planner=object())
            assert decision == "need_revision" and missing in turn["error"]
        for extra in ({"context_requests": ["/parts"]}, {"decision": "candidate"}):
            decision, turn = await multi_turn_outline_generation._handle_outline_phase(
                session_state=session, parsed_response={**candidate, **extra}, prepared_recovery_request=prepared, planner=object())
            assert decision == "need_revision" and turn["error"]
        handler.assert_not_called()
        assert session["accepted_outline_prefix"] == []
        await multi_turn_outline_generation._handle_outline_phase(
            session_state=session, parsed_response=candidate, prepared_recovery_request=prepared, planner=object())
        handler.assert_awaited_once()
        assert handler.call_args.kwargs["parsed_response"] == candidate
    asyncio.run(run())


def test_grounding_observation_is_used_only_for_missing_live_pose(context, monkeypatch):
    prepared, session, _, names = context
    prepared["offline_recovery"] = False
    async def observe(**kwargs):
        observed = {"part_name": names[0], "pose": {"x": 1, "y": 2, "z": 3}}
        kwargs["session_state"]["observation_history"].append({"output": observed})
        kwargs["session_state"]["observation_store"]["part_pose"] = observed
        return "observe", {}
    live = AsyncMock(side_effect=observe)
    monkeypatch.setattr(multi_turn, "_handle_grounding_phase", live)
    ref = f"/parts/{names[0]}/observed_pose"
    _, turn = asyncio.run(multi_turn_context.handle_outline_context(prepared=prepared, session=session,
        response={"thought": "Retrieve recorded facts", "decision": "need_context", "context_requests": [ref]}, planner=object()))
    assert turn["served_context"][ref]["value"] == {"x": 1, "y": 2, "z": 3}
    assert turn["served_context"][ref]["source"] == "resource_observation"
    live.assert_awaited_once()


@pytest.mark.parametrize("offline", [False, True])
def test_missing_robot_pose_uses_only_its_owner_and_never_calls_live_offline(context, monkeypatch, offline):
    prepared, session, jids, _ = context
    prepared["offline_recovery"] = offline
    for view in (prepared["recovery_resources"][jids[1]]["recovery_snapshot"], session["symbolic_resources"][jids[1]]):
        view.pop("current_pose", None)
    observe = Mock(return_value={"current_pose": {"x": 4, "y": 5, "z": 6}, "current_pose_captured_at": 100.0})
    if offline:
        observe.side_effect = AssertionError("offline invoked resource observation")
    unrelated = Mock(side_effect=AssertionError("wrong resource observed"))
    monkeypatch.setattr(multi_turn, "_resource_agent_map", lambda _: {
        jids[0]: SimpleNamespace(get_recovery_snapshot=unrelated), jids[1]: SimpleNamespace(get_recovery_snapshot=observe)})
    ref = f"/resources/{jids[1]}/current_pose"
    _, turn = asyncio.run(multi_turn_context.handle_outline_context(prepared=prepared, session=session,
        response={"thought": "Retrieve recorded facts", "decision": "need_context", "context_requests": [ref]}, planner=object()))
    unrelated.assert_not_called()
    if offline:
        observe.assert_not_called()
        assert turn["context_errors"] and not turn["served_context"]
    else:
        observe.assert_called_once()
        assert turn["served_context"][ref]["value"] == {"x": 4, "y": 5, "z": 6}
        assert turn["served_context"][ref]["captured_at"] == 100.0


def test_multi_turn_loop_records_retrieval_and_stops_without_progress(context, tmp_path, monkeypatch):
    prepared, session, _, _ = context
    prepared["recovery_debug"] = {"per_turn_debug_dir": str(tmp_path)}
    prepared["recovery_session"]["session_id"] = "retrieval-test"
    session.update(current_phase="outline", max_turns=5)
    llm = AsyncMock(return_value={"thought": "Need the recorded part identities", "decision": "need_context", "context_requests": ["/parts"]})
    planner = SimpleNamespace(product_agent=SimpleNamespace(ask_llm_structured=llm))
    monkeypatch.setattr(multi_turn, "_try_handle_modeled_continuation", AsyncMock(return_value=None))
    asyncio.run(multi_turn.execute_multi_turn_recovery(planner, prepared, session_state=session))
    final = prepared["multi_turn_session_state"]
    assert final["status"] == "NEEDS_CONTEXT"
    assert final["current_phase"] == "outline"
    assert final["accepted_outline_prefix"] == []
    assert llm.await_count == 2
    records = list((tmp_path / "recovery_outline/turns").glob("*.json"))
    assert len(records) == 2
    first = json.loads(sorted(records)[0].read_text())
    assert first["served_context"]["/parts"]["source_run_id"] == "capture"


def test_public_task_contracts_retain_saved_alternatives_and_bindings(context):
    prepared, session, jids, _ = context
    entry = prepared["recovery_resources"][jids[0]]
    entry["recovery_des_model"]["state_variables"].update({
        "held_part": {"scope": "resource", "domain": [None]},
        "part_state": {"scope": "part", "domain": ["in_gripper", "in_transit"]},
    })
    entry["recovery_des_model"]["events"] = [
        {"event_name": "place_approach", "guards": {"held_part": {"equals_from_param": "part_name"}},
         "updates": {"resource_state": {"set": "positioned"}, "part_state": {"set": "in_transit"}},
         "parameter_bindings": {"destination_location": {"location_type": "reachable_location"}},
         "requires_part_binding": True,
         "primitive_support": [{"primitive": "compute_place_targets", "params": {"part_name": {"$arg": "part_name"}}}]},
        {"event_name": "move_home", "guards": {"held_part": {"equals": None}},
         "updates": {"resource_state": {"set": "idle"}},
         "primitive_support": [{"primitive": "move_to_named_pose", "params": {"pose_name": "home"}}]},
    ]
    entry["execution_primitive_catalog"] = [
        {"name": "compute_place_targets", "params": {"part_name": {"type": "string"}}},
        {"name": "move_to_named_pose", "params": {"pose_name": {"type": "string"}}},
    ]
    before = deepcopy(prepared)
    listing = multi_turn_context._task_contract_events(entry, "internal")
    assert [row["event_name"] for row in listing] == ["place_approach", "move_home"]
    contract = multi_turn_context._public_task_contract(entry, listing[0])
    expected = deepcopy(entry["recovery_des_model"]["events"][0])
    expected["partial"] = False
    assert contract == expected
    with pytest.raises(ValueError, match="Unsupported outline"):
        multi_turn_context.resolve_outline_context(prepared, session, f"/task_contracts/{jids[0]}/place_approach")
    fields = multi_turn_context.resolve_outline_context(prepared, session, f"/resources/{jids[0]}/state_variables")
    assert fields["value"]["held_part"]["domain"] == [None]
    assert prepared == before


def test_public_task_contract_omits_private_fields_and_incomplete_primitive_bindings(context):
    prepared, session, jids, _ = context
    entry = prepared["recovery_resources"][jids[0]]
    canary = "EVALUATOR_PRIVATE_SEQUENCE_99193"
    entry["recovery_des_model"]["state_variables"].update({
        "internal_custody": {"private": True}, "task_ctx.destination_location": {"scope": "resource"},
    })
    entry["recovery_des_model"]["events"] = [
        {"event_name": "place_insert", "guards": {"resource_state": {"equals": "positioned"},
             "internal_custody": {"equals": canary}, "task_ctx.destination_location": {"equals_from_param": "destination_location"}},
         "updates": {"resource_state": {"set": "placed"}, "task_ctx.destination_location": {"set": None}},
         "evaluator_data": {"golden_answer": canary},
         "composition_contract": {"private": canary},
         "primitive_support": [{"primitive": "release_part", "params": {
             "part_name": {"$arg": "part_name"}, "model_name": {"$state": "_task_ctx", "path": ["model_name"]},
             "internal_custody": {"$state": "internal_custody"}, "private_argument": canary, "evaluator_data": canary,
         }, "when": [{"predicate": canary}], "evaluator_data": canary}]},
        {"event_name": canary, "private": True},
    ]
    entry["execution_primitive_catalog"] = [{"name": "release_part", "params": {
        "part_name": {"type": "string"}, "model_name": {"type": "string"}, "internal_custody": {"type": "string"},
        "private_argument": {"type": "string", "private": True}}}]
    before = deepcopy(prepared)
    value = multi_turn_context._public_task_contract(entry, entry["recovery_des_model"]["events"][0])
    assert value["event_name"] == "place_insert"
    assert value["guards"] == {"resource_state": {"equals": "positioned"}}
    assert value["updates"] == {"resource_state": {"set": "placed"}}
    assert value["primitive_support"] == [{"primitive": "release_part", "params": {"part_name": {"$arg": "part_name"}}}]
    assert value["partial"] is True
    assert value["omitted_sections"] == ["guards", "primitive_support", "updates"]
    assert "not a complete task contract" in value["note"]
    assert not any(token in json.dumps(value) for token in (canary, "task_ctx", "internal_custody", "evaluator_data"))
    listing = multi_turn_context._task_contract_events(entry, "internal")
    assert [row["event_name"] for row in listing] == ["place_insert"]
    assert prepared == before


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("reference", [
    "/resources/{jid}/task_contracts", "/task_contracts/{jid}/place_approach",
    "/primitive_contracts/{jid}/place_approach", "/parts/gear_small/target", "/evaluator_data",
])
def test_missing_declarations_never_request_a_new_observation(context, monkeypatch, offline, reference):
    prepared, session, jids, _ = context
    prepared["offline_recovery"] = offline
    observe = AsyncMock(side_effect=AssertionError("declarations cannot be supplied by sensors"))
    monkeypatch.setattr(multi_turn_context, "_observe_missing", observe)
    ref = reference.format(jid=jids[0])
    _, turn = asyncio.run(multi_turn_context.handle_outline_context(prepared=prepared, session=session,
        response={"thought": "Read saved declarations", "decision": "need_context", "context_requests": [ref]}, planner=object()))
    assert not turn["served_context"]
    assert turn["context_errors"]
    assert "requires a new observation capture" not in json.dumps(turn["context_errors"])
    assert "observation_error" not in turn["context_errors"][0]
    observe.assert_not_called()


def test_task_contract_retrieval_keeps_alternative_resource_available(context, monkeypatch):
    prepared, session, jids, _ = context
    entry = prepared["recovery_resources"][jids[2]]
    entry["recovery_des_model"]["events"] = [{"event_name": "alternative_public_event", "guards": {},
        "updates": {"resource_state": {"set": "idle"}}, "primitive_support": []}]
    live = AsyncMock(side_effect=AssertionError("offline task retrieval invoked live interfaces"))
    monkeypatch.setattr(multi_turn_context, "_observe_missing", live)
    ref = f"/task_contracts/{jids[2]}/alternative_public_event"
    _, turn = asyncio.run(multi_turn_context.handle_outline_context(prepared=prepared, session=session,
        response={"thought": "Read an alternative declaration", "decision": "need_context", "context_requests": [ref]}, planner=object()))
    assert not turn["served_context"] and turn["context_errors"]
    assert session["accepted_outline_prefix"] == []
    assert jids[2] in {row["resource_jid"] for row in multi_turn_context.resource_index(prepared)}
    session["current_phase"] = "outline"
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    assert "alternative_public_event" not in prompt
    public_ref = f"/resources/{jids[2]}/static_capabilities"
    _, turn = asyncio.run(multi_turn_context.handle_outline_context(prepared=prepared, session=session,
        response={"thought": "Read alternative capabilities", "decision": "need_context", "context_requests": [public_ref]}, planner=object()))
    assert public_ref in turn["served_context"]
    assert jids[2] in multi_turn_context.select_outline_context(prepared, session)["resource_jids"]
    live.assert_not_called()


def test_partial_task_contract_does_not_invent_missing_declarations(context):
    prepared, session, jids, _ = context
    prepared["recovery_resources"][jids[0]]["recovery_des_model"]["events"] = [{"event_name": "recorded_name_only"}]
    entry = prepared["recovery_resources"][jids[0]]
    value = multi_turn_context._public_task_contract(entry, entry["recovery_des_model"]["events"][0])
    assert value["partial"] is True
    assert value["omitted_sections"] == ["guards", "primitive_support", "updates"]
    assert value["guards"] == value["updates"] == {}
    assert "primitive_support" not in value


def test_diagnostic_refs_stay_unsupported_outside_diagnostic_mode(context):
    prepared, session, jids, names = context
    prepared["diagnostic_reach"] = {jids[0]: {"max_xy_radius_m": 1.0}}
    for ref in (f"/resources/{jids[0]}/gripper_reach", f"/resources/{jids[0]}/primitives",
                f"/parts/{names[0]}/pose_relations", "/geometry"):
        with pytest.raises(ValueError):
            multi_turn_context.resolve_outline_context(prepared, session, ref)
    session.update(current_phase="outline", outline_mode="incremental_candidates_validated")
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    assert "Diagnostic refs:" not in prompt and "Diagnostic Mode" not in prompt


def test_retrieval_prompt_states_request_limits_and_separates_declared_contracts(context):
    prepared, session, _, _ = context
    session.update(current_phase="outline", outline_mode="incremental_candidates_validated")
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    assert "When required facts are available" in prompt
    assert 'only `thought`, `decision`="need_context", `context_requests` (1 to 8 refs)' in prompt
    assert "Omit all candidate fields, even null/empty" in prompt
    assert "state_variables,pending_tasks" in prompt
    assert "/task_contracts/" not in prompt and "/primitive_catalog/" not in prompt and "/primitive_contracts/" not in prompt
    assert "Resource validators check whether the proposed condition is supported" in prompt
    assert "primitive composition checks feasibility" in prompt
    assert "do not invent facts or effects" in prompt
    prepared["recovery_session"]["outline_context_retrieval"] = False
    session["outline_context_retrieval"] = False
    legacy = multi_turn._build_phase_prompt(prepared, session)[1]
    assert "- Return one JSON object" in legacy
    assert "When required facts are available" not in legacy
    assert "Context retrieval" not in legacy
    assert "those tasks need not execute next" not in legacy


@pytest.mark.parametrize("last_indices", [(1, 0), (0,), (1,)])
def test_outline_retrieval_stops_on_reordered_or_subset_known_facts(context, last_indices):
    prepared, session, jids, _ = context
    refs = [f"/resources/{jid}/snapshot" for jid in jids[:2]]
    async def retrieve(requested):
        return await multi_turn_context.handle_outline_context(prepared=prepared, session=session,
            response={"thought": "Read saved facts", "decision": "need_context", "context_requests": requested}, planner=object())
    _, first = asyncio.run(retrieve(refs))
    assert session["status"] != "NEEDS_CONTEXT"
    requested = [refs[index] for index in last_indices]
    _, repeated = asyncio.run(retrieve(requested))
    assert session["status"] == "NEEDS_CONTEXT"
    assert repeated["context_requests"] == requested
    assert repeated["served_context"] == {ref: first["served_context"][ref] for ref in requested}
    assert "no new information" in repeated["context_errors"][-1]["error"]
    assert session["accepted_outline_prefix"] == []


@pytest.mark.parametrize("missing", [False, True])
def test_outline_retrieval_allows_new_information_among_known_references(context, missing):
    prepared, session, jids, _ = context
    known = f"/resources/{jids[0]}/snapshot"
    new = "/parts/gear_small/observed_pose" if missing else f"/resources/{jids[1]}/snapshot"
    async def retrieve(refs):
        return await multi_turn_context.handle_outline_context(prepared=prepared, session=session,
            response={"thought": "Read saved facts", "decision": "need_context", "context_requests": refs}, planner=object())
    asyncio.run(retrieve([known]))
    _, mixed = asyncio.run(retrieve([known, new]))
    assert session["status"] != "NEEDS_CONTEXT"
    if missing:
        assert mixed["context_errors"][0]["ref"] == new
        assert "not recorded" in mixed["context_errors"][0]["error"]
    else:
        assert new in mixed["served_context"]
    _, repeated = asyncio.run(retrieve([new]))
    assert session["status"] == "NEEDS_CONTEXT"
    assert "no new information" in repeated["context_errors"][-1]["error"]


@pytest.mark.parametrize("missing", [False, True])
def test_outline_retrieval_refreshes_after_an_accepted_transition(context, missing):
    prepared, session, jids, _ = context
    ref = "/parts/gear_small/observed_pose" if missing else f"/resources/{jids[0]}/static_capabilities"
    response = {"thought": "Read saved capabilities", "decision": "need_context", "context_requests": [ref]}
    async def retrieve():
        return await multi_turn_context.handle_outline_context(prepared=prepared, session=session, response=response, planner=object())
    _, before = asyncio.run(retrieve())
    session["accepted_outline_prefix"] = [{"outline_id": "RECOVERY_SEQ1", "resource_jid": jids[0]}]
    session["symbolic_resources"][jids[0]]["resource_state"] = "idle"
    _, after = asyncio.run(retrieve())
    assert session["status"] != "NEEDS_CONTEXT"
    if missing:
        assert before["context_errors"] == after["context_errors"]
    else:
        assert before["served_context"][ref]["value"] == after["served_context"][ref]["value"]
        assert before["served_context"][ref]["revision"] != after["served_context"][ref]["revision"]
    asyncio.run(retrieve())
    assert session["status"] == "NEEDS_CONTEXT"


@pytest.mark.parametrize("change", ["value", "source", "captured_at"])
def test_outline_retrieval_retains_changed_fact_evidence_at_same_revision(context, monkeypatch, change):
    prepared, session, jids, _ = context
    ref = f"/resources/{jids[0]}/current_pose"
    fact = multi_turn_context.resolve_outline_context(prepared, session, ref)
    resolve = Mock(side_effect=lambda *args: deepcopy(fact))
    monkeypatch.setattr(multi_turn_context, "resolve_outline_context", resolve)
    response = {"thought": "Read recorded pose evidence", "decision": "need_context", "context_requests": [ref]}
    async def retrieve():
        return await multi_turn_context.handle_outline_context(prepared=prepared, session=session, response=response, planner=object())
    asyncio.run(retrieve())
    fact[change] = {"x": 0.1, "y": 0.2, "z": 0.3} if change == "value" else "new recorded evidence"
    _, changed = asyncio.run(retrieve())
    assert session["status"] != "NEEDS_CONTEXT"
    assert changed["served_context"][ref] == fact
    asyncio.run(retrieve())
    assert session["status"] == "NEEDS_CONTEXT"


@pytest.mark.parametrize("missing", [False, True])
def test_multi_turn_loop_stops_alternating_known_requests_after_compaction(context, tmp_path, monkeypatch, missing):
    prepared, session, jids, _ = context
    prepared["recovery_debug"] = {"per_turn_debug_dir": str(tmp_path)}
    prepared["recovery_session"]["session_id"] = "alternating-retrieval-test"
    session.update(current_phase="outline", max_turns=8)
    refs = ([f"/parts/{name}/target" for name in ("gear_small", "KET4_Square_4mm")] if missing else
            [f"/resources/{jid}/snapshot" for jid in jids[:2]])
    responses = [{"thought": "Read recorded facts", "decision": "need_context", "context_requests": [ref]}
                 for ref in (refs[0], refs[1], refs[0])]
    llm = AsyncMock(side_effect=responses)
    planner = SimpleNamespace(product_agent=SimpleNamespace(ask_llm_structured=llm))
    observe = AsyncMock(side_effect=AssertionError("offline loop attempted live observation"))
    monkeypatch.setattr(multi_turn_context, "_observe_missing", observe)
    monkeypatch.setattr(multi_turn, "_try_handle_modeled_continuation", AsyncMock(return_value=None))
    asyncio.run(multi_turn.execute_multi_turn_recovery(planner, prepared, session_state=session))
    final = prepared["multi_turn_session_state"]
    assert final["status"] == "NEEDS_CONTEXT"
    assert final["outline_context_seen_results"]
    assert final["accepted_outline_prefix"] == []
    assert llm.await_count == 3
    records = [json.loads(path.read_text()) for path in sorted((tmp_path / "recovery_outline/turns").glob("*.json"))]
    assert len(records) == 3
    assert records[2]["context_requests"] == records[0]["context_requests"]
    assert "no new information" in records[2]["context_errors"][-1]["error"]
    if missing:
        assert records[2]["context_errors"][0] == records[0]["context_errors"][0]
    else:
        assert records[2]["served_context"] == records[0]["served_context"]
    observe.assert_not_called()


@pytest.mark.parametrize("binding", ["guard", "update", "both"])
def test_initial_public_fields_explain_only_declared_custody_bindings(context, binding):
    prepared, session, jids, _ = context
    entry = prepared["recovery_resources"][jids[0]]
    model = entry["recovery_des_model"]
    model["state_variables"]["held_part"] = {"scope": "resource", "domain": [None]}
    model["state_variables"]["resource_location"] = {"scope": "resource", "domain": ["home"]}
    model["events"] = []
    if binding in {"guard", "both"}:
        model["events"].append({"event_name": "DECLARED_EVENT_NAME_NOT_IN_INITIAL_PROMPT",
            "guards": {"held_part": {"equals_from_param": "part_name"}},
            "updates": {"resource_state": {"set": "idle"}}})
    if binding in {"update", "both"}:
        model["events"].append({"event_name": "DECLARED_EVENT_NAME_NOT_IN_INITIAL_PROMPT",
            "guards": {}, "updates": {"resource_state": {"set": "idle"}, "held_part": {"set_from_param": "part_name"}}})
    session.update(current_phase="outline", outline_mode="incremental_candidates_validated")
    original = deepcopy(prepared)
    payload, prompt = multi_turn._build_phase_prompt(prepared, session)
    facts = payload["outline_resource_fields"]
    assert set(facts) == set(jids[:2])
    assert jids[2] in prompt
    ref = f"/resources/{jids[0]}/state_variables"
    fact = multi_turn_context.resolve_outline_context(prepared, session, ref)
    assert facts[jids[0]] == fact
    assert fact["value"]["held_part"] == {"scope": "resource", "domain": [None]}
    assert fact["value"]["resource_location"]["domain"] == ["home"]
    assert fact["source"] == "declaration"
    assert fact["source_run_id"] == "capture"
    assert fact["captured_at"] == prepared["source_observation_captured_at"]
    explanation = multi_turn_context._held_part_interpretation(entry, fact["value"])
    assert explanation["source"] == "runtime_validation"
    assert explanation["source_function"] == "cais_spade_llm.resources.recovery_feasibility._successor_bindings"
    assert explanation["declared_part_name_guards"] == ([{"equals_from_param": "part_name"}] if binding != "update" else [])
    assert explanation["declared_part_name_updates"] == ([{"set_from_param": "part_name"}] if binding != "guard" else [])
    assert ("When held_part is preserved" in explanation["rule"]) is (binding != "update")
    assert ("A declared set_from_param update" in explanation["rule"]) is (binding != "guard")
    assert "only when that event matches the proposed successor" in explanation["rule"]
    assert "Other guards, bindings and domain checks still apply" in explanation["rule"]
    assert "does not establish action acceptance or physical feasibility" in explanation["rule"]
    assert "not an observation or proposed action" in prompt
    assert "source_function" not in prompt and "declared_part_name_guards" not in prompt
    assert "DECLARED_EVENT_NAME_NOT_IN_INITIAL_PROMPT" not in prompt
    assert "primitive_support" not in prompt
    assert prepared == original


@pytest.mark.parametrize("change", [
    "missing_held_part", "private_held_part", "part_scope", "missing_resource_state", "private_resource_state",
    "missing_state_update", "unsupported_state_update", "private_event", "different_guard_parameter",
    "different_update_parameter", "guard_overridden_by_update", "set_takes_precedence", "private_guard", "private_update",
])
def test_custody_interpretation_requires_the_existing_successor_binding_case(context, change):
    prepared, session, jids, _ = context
    model = prepared["recovery_resources"][jids[0]]["recovery_des_model"]
    model["state_variables"]["held_part"] = {"scope": "resource", "domain": [None]}
    event = {"event_name": "recorded_event", "guards": {"held_part": {"equals_from_param": "part_name"}},
             "updates": {"resource_state": {"set": "idle"}}}
    model["events"] = [event]
    if change == "missing_held_part":
        model["state_variables"].pop("held_part")
    elif change == "private_held_part":
        model["state_variables"]["held_part"]["private"] = True
    elif change == "part_scope":
        model["state_variables"]["held_part"]["scope"] = "part"
    elif change == "missing_resource_state":
        model["state_variables"].pop("resource_state")
    elif change == "private_resource_state":
        model["state_variables"]["resource_state"]["private"] = True
    elif change == "missing_state_update":
        event["updates"].pop("resource_state")
    elif change == "unsupported_state_update":
        event["updates"]["resource_state"] = {"unavailable": True}
    elif change == "private_event":
        event["private"] = True
    elif change == "different_guard_parameter":
        event["guards"]["held_part"]["equals_from_param"] = "different_part"
    elif change == "different_update_parameter":
        event["updates"]["held_part"] = {"set_from_param": "different_part"}
    elif change == "guard_overridden_by_update":
        event["updates"]["held_part"] = {"set": None}
    elif change == "set_takes_precedence":
        event["updates"]["held_part"] = {"set": None, "set_from_param": "part_name"}
    elif change == "private_guard":
        event["guards"]["held_part"]["private"] = True
    elif change == "private_update":
        event["updates"]["held_part"] = {"set_from_param": "part_name", "private": True}
    original = deepcopy(prepared)
    fact = multi_turn_context.resolve_outline_context(prepared, session, f"/resources/{jids[0]}/state_variables")
    assert "interpretation" not in fact
    assert prepared == original


def test_initial_declarations_do_not_expose_private_fields_or_fabricate_missing_models(context):
    prepared, session, jids, _ = context
    model = prepared["recovery_resources"][jids[0]]["recovery_des_model"]
    canary = "PRIVATE_EVALUATOR_FIELD_CANARY"
    model["state_variables"].update({"internal": {"private": True, "domain": [canary]},
                                    "task_ctx.hidden": {"domain": [canary]}, "_hidden": {"domain": [canary]}})
    model["evaluator_data"] = {"solution": canary}
    prepared["recovery_resources"][jids[1]]["recovery_des_model"] = None
    session["current_phase"] = "outline"
    before = deepcopy(prepared)
    payload, prompt = multi_turn._build_phase_prompt(prepared, session)
    facts = payload["outline_resource_fields"]
    assert canary not in json.dumps(facts)
    assert canary not in prompt
    assert facts[jids[1]]["value"] == {}
    assert "not recorded" in facts[jids[1]]["missing_context"]
    assert "interpretation" not in facts[jids[1]]
    assert prepared == before


def test_latest_twelve_resource_capture_keeps_custody_domains_and_essential_fields():
    path = Path(__file__).resolve().parents[1] / "cais_spade_llm/monitor/environment_runs/recovery/1c3bec8faca6495cb1b0e22046277d5c/observations.json"
    if not path.is_file():
        pytest.skip("The user-owned October 8 observation capture is not present in this checkout")
    original_bytes = path.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    observations = json.loads(original_bytes)
    prepared = deepcopy(observations["prepared_recovery_request"])
    prepared.update(offline_recovery=True, source_observation_captured_at=observations["captured_at"],
                    source_observation_run_id=observations["run_id"])
    prepared.setdefault("recovery_session", {})["outline_context_retrieval"] = True
    session = multi_turn.build_multi_turn_session_seed(prepared)
    session["current_phase"] = "outline"
    before = deepcopy(prepared)
    payload, prompt = multi_turn._build_phase_prompt(prepared, session)
    assert len(prepared["recovery_resources"]) == len(payload["outline_context"]["resource_index"]) == 12
    assert set(payload["outline_resource_fields"]) == {"recovery-resource-3@localhost", "recovery-resource-4@localhost"}
    for jid, fact in payload["outline_resource_fields"].items():
        assert fact == multi_turn_context.resolve_outline_context(prepared, session, f"/resources/{jid}/state_variables")
        assert fact["value"]["held_part"] == prepared["recovery_resources"][jid]["recovery_des_model"]["state_variables"]["held_part"]
        interpretation = multi_turn_context._held_part_interpretation(prepared["recovery_resources"][jid], fact["value"])
        assert interpretation["declared_part_name_guards"] == [{"equals_from_param": "part_name"}]
        assert interpretation["declared_part_name_updates"] == [{"set_from_param": "part_name"}]
        assert set(fact["interpretation"]) == {"source", "field", "rule"}
    assert payload["outline_resource_fields"]["recovery-resource-4@localhost"]["value"]["held_part"]["domain"] == [None]
    assert "Runtime binding interpretation" in prompt
    assert "recover_pick" not in prompt and "recover_place_insert" not in prompt
    assert prepared == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash


@pytest.fixture(params=[
    ('Conveyor', 'Conveyor breakdown', 'part_name', 'part_name'),
    ('M1', 'Machining breakdown during part processing', 'current_job', 'job_id'),
    ('3D Printing Station', 'unexpected printing disruption', 'output_part', 'part_name'),
])
def recorded(request):
    jid, failure, field, parameter = request.param
    event = {'event_name': 'DECLARED_EVENT_NOT_A_SUGGESTION', 'guards': {}, 'updates': {
        'resource_state': {'set': 'idle'}, field: {'set_from_param': parameter}}, 'primitive_support': []}
    model = {'state_variables': {'resource_state': {'scope': 'resource', 'domain': ['failed', 'idle']},
                                field: {'scope': 'resource', 'domain': [None]}}, 'events': [event]}
    snapshot = {'resource_jid': jid, 'resource_state': 'failed', field: None}
    prepared = {'offline_recovery': True, 'ra_jid': jid, 'P_id': [],
                'recovery_session': {'outline_context_retrieval': True},
                'llm_input': {'fault_event': {'focused_resource_jid': jid, 'failure_type': failure},
                              'observed_runtime_state': {'resources': [snapshot]}},
                'recovery_resources': {jid: {'resource_type': 'resource', 'recovery_snapshot': snapshot,
                                            'recovery_des_model': model, 'execution_primitive_catalog': []}}}
    session = multi_turn.build_multi_turn_session_seed(prepared)
    session.update(current_phase='outline', outline_mode='incremental_candidates_validated')
    return prepared, session, jid, field, parameter, model, event


def test_nonrobot_initial_context_and_offline_retrieval_preserve_declared_facts(recorded, monkeypatch):
    prepared, session, jid, field, parameter, model, event = recorded
    original = deepcopy(prepared)
    payload, prompt = multi_turn._build_phase_prompt(prepared, session)
    assert payload['outline_resource_fields'][jid]['value'] == model['state_variables']
    assert jid in {row['resource_jid'] for row in payload['outline_context']['resource_index']}
    assert 'DECLARED_EVENT_NOT_A_SUGGESTION' not in prompt
    observe = AsyncMock(side_effect=AssertionError('Offline resource retrieval queried live observations'))
    monkeypatch.setattr(multi_turn_context, '_observe_missing', observe)
    refs = [f'/resources/{jid}/snapshot', f'/resources/{jid}/state_variables',
            f'/resources/{jid}/task_contracts', f'/task_contracts/{jid}/{event["event_name"]}']
    _, turn = asyncio.run(multi_turn_context.handle_outline_context(prepared=prepared, session=session,
        response={'thought': 'Read saved declarations', 'decision': 'need_context', 'context_requests': refs}, planner=object()))
    assert [row['ref'] for row in turn['context_errors']] == refs[2:]
    assert turn['served_context'][refs[0]]['value'][field] is None
    assert set(turn['served_context']) == set(refs[:2])
    contract = multi_turn_context._public_task_contract(prepared['recovery_resources'][jid], event)
    assert contract['guards'] == event['guards']
    assert contract['updates'] == event['updates']
    assert contract['partial'] is False
    assert prepared == original
    observe.assert_not_called()


def test_nonrobot_declared_parameter_update_preserves_existing_validator_semantics(recorded):
    from cais_spade_llm.resources.recovery_feasibility import _successor_bindings

    _, _, _, field, parameter, model, event = recorded
    task = {'expected_end_state': {'resource_state': 'idle', field: 'recorded_value'}}
    assert _successor_bindings(task, event, model) == {parameter: 'recorded_value'}
    assert model['state_variables'][field]['domain'] == [None]
    task['params'] = {parameter: 'conflicting_value'}
    assert _successor_bindings(task, event, model) is None


@pytest.mark.parametrize('operator', ['equals_from_param', 'not_equals_from_param'])
def test_nonrobot_public_parameter_guards_are_retrievable_and_domains_remain_closed(recorded, operator):
    from cais_spade_llm.resources.recovery_feasibility import _successor_bindings

    prepared, session, jid, field, parameter, model, event = recorded
    event['updates'].pop(field)
    event['guards'][field] = {operator: parameter}
    model['state_variables'][field]['domain'] = ['recorded_value']
    task = {'expected_end_state': {'resource_state': 'idle', field: 'unrecorded_value'}}
    if operator == 'not_equals_from_param':
        task['params'] = {parameter: 'other_value'}
    assert _successor_bindings(task, event, model) is None
    task['expected_end_state'][field] = 'recorded_value'
    assert _successor_bindings(task, event, model) is not None
    fact = multi_turn_context._public_task_contract(prepared['recovery_resources'][jid], event)
    assert fact['guards'][field] == {operator: parameter}
    assert fact['partial'] is False
    with pytest.raises(ValueError, match='Unsupported outline'):
        multi_turn_context.resolve_outline_context(prepared, session, f'/task_contracts/{jid}/{event["event_name"]}')


@pytest.mark.parametrize('private_name', ['internal', 'task_ctx.hidden', '_hidden', 'named_poses', 'current_pose'])
def test_nonrobot_private_snapshot_values_are_not_retrievable(recorded, private_name):
    prepared, session, jid, _, _, model, _ = recorded
    declaration = {'scope': 'resource', 'domain': ['PRIVATE_CANARY']}
    if private_name in {'internal', 'named_poses', 'current_pose'}:
        declaration['private'] = True
    model['state_variables'][private_name] = declaration
    prepared['recovery_resources'][jid]['recovery_snapshot'][private_name] = 'PRIVATE_CANARY'
    snapshot = multi_turn_context.resolve_outline_context(prepared, session, f'/resources/{jid}/snapshot')
    assert private_name not in snapshot['value']
    with pytest.raises(ValueError, match='NEEDS_CONTEXT'):
        multi_turn_context.resolve_outline_context(prepared, session, f'/resources/{jid}/{private_name}')


def test_nonrobot_unknown_and_missing_declarations_stay_missing(recorded):
    prepared, session, jid, _, _, model, _ = recorded
    with pytest.raises(ValueError, match='NEEDS_CONTEXT'):
        multi_turn_context.resolve_outline_context(prepared, session, '/resources/unknown_resource/snapshot')
    with pytest.raises(ValueError, match='Unsupported outline'):
        multi_turn_context.resolve_outline_context(prepared, session, f'/task_contracts/{jid}/unknown_event')
    model.pop('state_variables')
    fact = multi_turn_context.resolve_outline_context(prepared, session, f'/resources/{jid}/state_variables')
    assert fact['value'] == {} and 'NEEDS_CONTEXT' in fact['missing_context']
    assert 'interpretation' not in fact


@pytest.mark.parametrize("reference", [
    "/task_contracts/{jid}/DECLARED_FUNCTION_CANARY", "/resources/{jid}/task_contracts",
    "/primitive_catalog/{jid}", "/primitive_contracts/{jid}/PRIMITIVE_CANARY",
    "/resources/{jid}/primitive_catalog", "/resources/{jid}/execution_primitive_catalog",
])
def test_outline_rejects_recorded_execution_catalogs_without_live_calls(context, monkeypatch, reference):
    prepared, session, jids, _ = context
    entry = prepared["recovery_resources"][jids[0]]
    entry["recovery_des_model"]["events"] = [{"event_name": "DECLARED_FUNCTION_CANARY"}]
    entry["execution_primitive_catalog"] = [{"name": "PRIMITIVE_CANARY"}]
    ref = reference.format(jid=jids[0])
    observe = AsyncMock(side_effect=AssertionError("A rejected reference must not trigger observation"))
    monkeypatch.setattr(multi_turn_context, "_observe_missing", observe)
    before = deepcopy(prepared)
    _, turn = asyncio.run(multi_turn_context.handle_outline_context(prepared=prepared, session=session,
        response={"thought": "Read context", "decision": "need_context", "context_requests": [ref]}, planner=object()))
    assert not turn["served_context"]
    assert turn["context_errors"][0]["ref"] == ref
    assert "Unsupported outline" in turn["context_errors"][0]["error"]
    session.update(current_phase="outline", outline_mode="incremental_candidates_validated")
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    assert "DECLARED_FUNCTION_CANARY" not in prompt and "PRIMITIVE_CANARY" not in prompt
    assert prepared == before
    observe.assert_not_called()


def test_primitive_composition_still_retrieves_its_own_catalog(context):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_primitive_generation,
    )

    prepared, session, jids, _ = context
    entry = prepared["recovery_resources"][jids[0]]
    entry["primitive_catalog"] = deepcopy(entry["execution_primitive_catalog"])
    served, errors = multi_turn_primitive_generation._serve_context_requests(
        prepared_recovery_request=prepared, session_state=session,
        outline_event={"outline_id": "RECOVERY_SEQ1", "resource_jid": jids[0]},
        context_requests=["/primitive_catalog", "/primitive_contracts/move_to_named_pose"],
    )
    assert errors == []
    assert served["/primitive_contracts/move_to_named_pose"]["name"] == "move_to_named_pose"


def test_public_capabilities_keep_recorded_limits_sources_empty_values_and_conflicts(context):
    prepared, session, jids, _ = context
    entry = prepared["recovery_resources"][jids[2]]
    entry["static_capabilities"] = {"reachability": [], "staging_areas": {"Exit": {"x": 1, "y": 2, "z": 3}}}
    entry["recovery_snapshot"].update(reachable_locations=["assembly_board-v1"],
        gripper_reach={"frame": "world", "origin_pose": {"x": 0, "y": 0, "z": 1}, "max_xy_radius_m": 0.7})
    before = deepcopy(prepared)
    ref = f"/resources/{jids[2]}/static_capabilities"
    fact = multi_turn_context.resolve_outline_context(prepared, session, ref)
    values = fact["value"]
    assert values["reachability"] == []
    assert values["reachable_locations"] == ["assembly_board-v1"]
    assert "gripper_reach" not in json.dumps(values)
    assert values["staging_areas"] == entry["static_capabilities"]["staging_areas"]
    assert fact["captured_at"] == prepared["source_observation_captured_at"]
    assert "capabilities" not in values
    index = next(row for row in multi_turn_context.resource_index(prepared) if row["resource_jid"] == jids[2])
    assert "gripper_reach" not in index and "staging_areas" not in index
    assert index["available_capability_fields"] == ["staging_areas"]
    assert index["reachability"] == []
    assert prepared == before
    entry["recovery_adapter"] = {"reachable_locations": ["Exit"]}
    conflict = multi_turn_context.resolve_outline_context(prepared, session, ref)["value"]
    assert "reachable_locations" not in conflict
    assert conflict["conflicts"]["reachable_locations"] == {
        "recovery_adapter": ["Exit"], "recovery_snapshot": ["assembly_board-v1"],
    }
    index = next(row for row in multi_turn_context.resource_index(prepared) if row["resource_jid"] == jids[2])
    assert "reachable_locations" not in index and index["conflicts"] == conflict["conflicts"]


def test_outline_feedback_preserves_public_reach_facts_without_execution_internals(context):
    prepared, session, jids, names = context
    canary = "PRIVATE_FUNCTION_AND_PRIMITIVE_CANARY"
    pose = {"x": -0.8, "y": 0.3, "z": 1.025}
    reach = {"frame": "world", "origin_pose": {"x": 0.5, "y": -0.5, "z": 0}, "max_xy_radius_m": 0.85}
    feedback = {"constraint_code": "workspace_unreachable", "resource_jid": jids[0], "part_name": names[0],
                "reason": canary, "evidence": {"checked_pose": pose, "gripper_reach": reach,
                "pose_source": "/parts/gear_small/observed_pose", "role": "source", "evidence_source": "saved_observations",
                "task_contracts": canary, "primitive_support": canary, "evaluator_data": canary}}
    session.update(current_phase="outline", outline_mode="incremental_candidates_validated",
                   outline_validation_findings=[feedback], primitive_escalation_diagnostics=[deepcopy(feedback)])
    session["accepted_outline_prefix"] = [
        {"outline_id": "RECOVERY_SEQ1", "event_name": "LLM_AUTHORED_EVENT", "candidate_source": "llm", "resource_jid": jids[0]},
        {"outline_id": "RECOVERY_SEQ2", "event_name": canary, "candidate_source": "robot_task_program", "resource_jid": jids[0],
         "part_name": names[0], "expected_end_state": {"resource_state": "positioned"}},
    ]
    before = deepcopy(session)
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    assert canary not in prompt
    assert "LLM_AUTHORED_EVENT" in prompt and '"candidate_source": "robot_task_program"' in prompt
    assert '"resource_state": "positioned"' in prompt
    assert multi_turn_prompts._inline_json(pose) in prompt
    assert "gripper_reach" not in prompt and "max_xy_radius_m" not in prompt
    assert "workspace_unreachable" in prompt and "saved_observations" in prompt
    assert session["outline_validation_findings"] == before["outline_validation_findings"]
    assert session["accepted_outline_prefix"] == before["accepted_outline_prefix"]


def test_outline_keeps_interrupted_and_pending_function_names_as_recorded_work(context):
    prepared, session, jids, _ = context
    prepared["llm_input"]["fault_event"]["blocked_at_function"] = "INTERRUPTED_FUNCTION"
    prepared["recovery_resources"][jids[0]]["pending_tasks"][0]["function_name"] = "PENDING_FUNCTION"
    session.update(current_phase="outline", outline_mode="incremental_candidates_validated",
                   outline_requested_refs=[f"/resources/{jids[0]}/pending_tasks"])
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    assert "INTERRUPTED_FUNCTION" in prompt and "PENDING_FUNCTION" in prompt
    assert "/task_contracts/" not in prompt and "/primitive_catalog/" not in prompt


def test_outline_feedback_rejects_opaque_evidence_and_retains_typed_public_facts(context):
    prepared, session, jids, _ = context
    canary = "PRIVATE_NOMINAL_FUNCTION_AND_PRIMITIVE_CANARY"
    opaque = {"actual": {"functions": [canary]}, "expected": {"primitives": [canary]},
              "available": [canary], "declarations": [["task_contracts", {"function_name": canary}]]}
    reach = {"frame": "world", "origin_pose": {"x": 0.5, "y": -0.5, "z": 0}, "max_xy_radius_m": 0.85}
    findings = [
        {"constraint_code": "resource_validation_unavailable", "evidence": opaque},
        {"constraint_code": "validation_state_stale", "evidence": {"field": "held_part", **opaque}},
        {"constraint_code": "resource_validation_unavailable", "evidence": {
            "field": "gripper_reach", "declarations": [["recovery_snapshot.gripper_reach", {
                **reach, "primitive_catalog": canary, "origin_pose": {**reach["origin_pose"], "private": canary}}]],
            "available": False}},
        {"constraint_code": "held_part_location_mismatch", "evidence": {
            "field": "expected_end_state.part_location", "proposed_part_location": "Buffer For Machined parts",
            "expected_carried_part_location": jids[0], "mismatches": [
                {"field": "held_part", "actual": None, "expected": "gear_small"}, opaque]}},
    ]
    session.update(current_phase="outline", outline_mode="incremental_candidates_validated",
                   outline_validation_findings=findings, primitive_escalation_diagnostics=deepcopy(findings))
    before = deepcopy(session)
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    assert canary not in prompt
    assert "expected_carried_part_location" in prompt and jids[0] in prompt
    assert "gripper_reach" not in prompt and "max_xy_radius_m" not in prompt
    assert '"available": false' not in prompt
    assert '"actual": null' in prompt and '"expected": "gear_small"' in prompt
    assert session["outline_validation_findings"] == before["outline_validation_findings"]
    assert session["primitive_escalation_diagnostics"] == before["primitive_escalation_diagnostics"]


def test_successor_feedback_keeps_field_conflict_and_excludes_private_derivation(context):
    prepared, session, jids, _ = context
    canary = 'PRIVATE_OWNER_EVENT_AND_SEQUENCE'
    model = prepared['recovery_resources'][jids[0]]['recovery_des_model']
    model['state_variables']['active_job'] = {'scope': 'resource', 'domain': ['job-1', None]}
    session['recovery_des_models'][jids[0]] = deepcopy(model)
    session.update(current_phase='outline', outline_mode='incremental_candidates_validated',
        outline_validation_findings=[{'constraint_code': 'unsupported_successor_condition',
            'resource_jid': jids[0], 'reason': canary, 'evidence': {'mismatches': [{
                'field': 'active_job', 'actual': 'job-1', 'expected': None, 'condition_operator': 'equals',
                'evidence_source': 'resource declarations supporting requested changes',
                'source': {'event_name': canary, 'primitive_sequence': [canary]}}]}}])
    before = deepcopy(session)
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    assert canary not in prompt
    assert 'active_job' in prompt and '"actual": "job-1"' in prompt and '"expected": null' in prompt
    assert 'resource declarations supporting requested changes' in prompt
    assert session == before


def test_outline_feedback_respects_each_resources_private_fields(context):
    prepared, session, jids, _ = context
    canary = "PRIVATE_OTHER_RESOURCE_VALUE"
    for jid, private in zip(jids[:2], [True, False], strict=True):
        declaration = {"private": private, "scope": "resource", "domain": []}
        prepared["recovery_resources"][jid]["recovery_des_model"]["state_variables"]["inspection_result"] = declaration
        session["recovery_des_models"][jid]["state_variables"]["inspection_result"] = deepcopy(declaration)
    session.update(current_phase="outline", outline_mode="incremental_candidates_validated",
                   primitive_escalation_diagnostics=[{
                       "resource_jid": jids[0], "constraint_code": "validation_state_stale",
                       "expected_end_state": {"inspection_result": canary},
                       "evidence": {"field": "inspection_result", "actual": canary},
                       "validation_findings": [{"evidence": {"field": "inspection_result", "expected": canary}}]},
                       {"resource_jid": jids[1], "constraint_code": "validation_state_stale",
                        "evidence": {"field": "inspection_result", "actual": "RECORDED_PUBLIC_VALUE"}}])
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    assert canary not in prompt
    assert "RECORDED_PUBLIC_VALUE" in prompt


def test_validated_outline_initial_prompt_uses_public_snapshot_and_pose_provenance(context):
    prepared, session, jids, _ = context
    jid = jids[0]
    prepared["recovery_resources"][jid]["recovery_des_model"]["state_variables"]["current_pose"] = {
        "scope": "resource", "domain": [], "private": True,
    }
    session["symbolic_resources"][jid]["current_pose"] = {"private": "PRIVATE_POSE_CANARY"}
    prepared["recovery_resources"][jid]["recovery_snapshot"]["current_pose"] = {"private": "PRIVATE_POSE_CANARY"}
    public_jid = jids[1]
    session["symbolic_resources"][public_jid].update(
        current_pose={"x": 0.2, "y": 0.3, "z": 0.4}, current_pose_source="projected_state",
        projected_pose={"x": 0.2, "y": 0.3, "z": 0.4})
    session.update(validation_contract_version=2, current_phase="outline")
    before = deepcopy(prepared)
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    assert "PRIVATE_POSE_CANARY" not in prompt
    assert '"current_pose_source": "projected_state"' in prompt
    assert '"projected_pose"' in prompt
    assert prepared == before


@pytest.mark.parametrize("empty_domain", [False, True])
@pytest.mark.parametrize("field,declared_type,domain,accepted,rejected", [
    ("active_job", ["string", "null"], ["environment_23"], [None, "environment_23"], [5, False, [], {}]),
    ("active_job", ["null", "string"], ["environment_23"], [None, "environment_23"], [5]),
    ("temperature", ["number", "null"], [20], [None, 20.5], ["20", True, [], {}]),
    ("temperature", ["integer", "null"], [20], [None, 21], [20.5, True, "20"]),
    ("occupied", ["boolean", "null"], [False], [None, True], [1, "false"]),
    ("current_pose", ["array", "null"], [[0, 0, 0]], [None, [1, 2, 3]], [{}, "pose"]),
    ("current_pose", ["object", "null"], [{}], [None, {"x": 1}], [[], "pose"]),
    ("active_job", "string", ["environment_23"], ["environment_23"], [None, 1]),
    ("active_job", ["string"], ["environment_23"], ["environment_23"], [None]),
    ("active_job", "null", [None], [None], ["environment_23", 0]),
    ("held_part", "string", [None], ["gear_small"], [None, ""]),
    ("resource_state", ["string", "null"], ["idle"], [None, "idle"], [""]),
])
def test_outline_field_schema_preserves_declared_types(
    field, declared_type, domain, accepted, rejected, empty_domain,
):
    declaration = {"scope": "resource", "type": declared_type,
                   "domain": [] if empty_domain else domain, "domain_complete": False}
    before = deepcopy(declaration)
    schema = multi_turn_prompts._outline_state_field_schema(field, declaration)
    assert schema["type"] == declared_type
    assert type(schema["type"]) is type(declared_type)
    if isinstance(declared_type, list):
        assert schema["type"] is not declared_type
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    for value in accepted:
        assert validator.is_valid(value), (schema, value)
    for value in rejected:
        assert not validator.is_valid(value), (schema, value)
    assert declaration == before


@pytest.mark.parametrize("declared_type", [None, "", [], ["string", "string"],
    ["string", None], ["String"], ["float"], {"type": "string"}, [dict()], 1, True,
    ("string", "null")])
def test_outline_field_schema_rejects_malformed_explicit_types(declared_type):
    declaration = {"type": declared_type, "domain": ["environment_23"]}
    with pytest.raises(ValueError, match="Invalid declared type for outline field 'active_job'"):
        multi_turn_prompts._outline_state_field_schema("active_job", declaration)


@pytest.mark.parametrize("field,declaration,expected", [
    ("active_job", {"domain": [None, "environment_23"]}, {"type": ["null", "string"]}),
    ("temperature", {"domain": [20, 20.5]}, {"type": ["integer", "number"]}),
    ("resource_state", {"domain": [None]}, {"type": "string", "minLength": 1}),
    ("part_state", {}, {"type": "string", "minLength": 1}),
    ("held_part", {"domain": ["gear_small"]}, {"type": ["string", "null"], "minLength": 1}),
    ("inspection_result", {"domain": []},
     {"type": ["string", "number", "boolean", "object", "array", "null"]}),
])
def test_outline_field_schema_preserves_legacy_fallback(field, declaration, expected):
    assert multi_turn_prompts._outline_state_field_schema(field, declaration) == expected


@pytest.mark.parametrize("resource_type", ["printer", "conveyor", "machining"])
def test_validated_outline_uses_exact_declared_fields_without_robot_state_requirements(context, resource_type):
    prepared, session, jids, names = context
    jid = jids[0]
    declarations = {"temperature": {"scope": "resource", "domain": [10, 20]},
                    "inspection_result": {"scope": "part", "domain": [None, "done"]}}
    prepared["recovery_resources"][jid].update(resource_type=resource_type,
        recovery_des_model={"state_variables": deepcopy(declarations)})
    session["recovery_des_models"][jid] = {"state_variables": deepcopy(declarations)}
    session["symbolic_resources"][jid] = {"resource_jid": jid, "temperature": 10}
    session["symbolic_parts"][names[0]] = {"part_name": names[0], "inspection_result": None}
    session.update(validation_contract_version=2, current_phase="outline")
    event = {"outline_id": "event_1", "event_name": "observed_disruption_response", "resource_jid": jid,
             "part_name": names[0], "expected_end_state": {"temperature": 20, "inspection_result": "done"},
             "rationale": "Proposed declared end conditions"}
    task, findings = multi_turn._derive_candidate_outline_task(candidate_task=event,
        session_state=session, prepared_recovery_request=prepared)
    assert findings == []
    assert task["expected_start_state"] == {"temperature": 10, "inspection_result": None}
    schema = multi_turn._get_response_schema("outline", session)["schema"]
    jsonschema.validate({"thought": "Use supplied state declarations", "next_transition": event}, schema)
    assert "temperature" in multi_turn._build_phase_prompt(prepared, session)[1]
    assert '"inspection_result": null' in multi_turn._build_phase_prompt(prepared, session)[1]
    fact = multi_turn_context.resolve_outline_context(prepared, session, f"/parts/{names[0]}/inspection_result")
    assert fact["value"] is None and fact["source"] == "saved_observations"


def test_validated_outline_does_not_force_undeclared_part_location_for_custody(context):
    prepared, session, jids, names = context
    jid = jids[0]
    declarations = {"held_part": {"scope": "resource", "domain": [None, names[0]]}}
    prepared["recovery_resources"][jid]["recovery_des_model"] = {"state_variables": declarations}
    session["recovery_des_models"][jid] = {"state_variables": declarations}
    session["symbolic_resources"][jid] = {"resource_jid": jid, "held_part": None}
    session.update(validation_contract_version=2, current_phase="outline")
    event = {"outline_id": "event_1", "event_name": "observed_disruption_response", "resource_jid": jid,
             "part_name": names[0], "expected_end_state": {"held_part": names[0]},
             "rationale": "ResourceAgent derives custody from declared effects"}
    task, findings = multi_turn._derive_candidate_outline_task(candidate_task=event,
        session_state=session, prepared_recovery_request=prepared)
    assert findings == [] and task["expected_start_state"] == {"held_part": None}
    jsonschema.validate({"thought": "Propose the declared field", "next_transition": event},
                        multi_turn._get_response_schema("outline", session)["schema"])


@pytest.mark.parametrize("field,with_part", [("_private", True), ("task_ctx.hidden", True), ("inspection_result", False)])
def test_validated_outline_structural_checks_reject_private_or_unbound_part_fields(context, field, with_part):
    prepared, session, jids, names = context
    declarations = {field: {"scope": "part" if field == "inspection_result" else "resource", "domain": [None]}}
    prepared["recovery_resources"][jids[0]]["recovery_des_model"] = {"state_variables": declarations}
    session["recovery_des_models"][jids[0]] = {"state_variables": declarations}
    session["validation_contract_version"] = 2
    event = {"outline_id": "event_1", "event_name": "e", "resource_jid": jids[0],
             "expected_end_state": {field: None}, "rationale": "Check declared shape"}
    if with_part:
        event["part_name"] = names[0]
    task, findings = multi_turn._derive_candidate_outline_task(candidate_task=event,
        session_state=session, prepared_recovery_request=prepared)
    assert task is None and findings


@pytest.mark.parametrize("field", ["inspection_result", "observed_pose", "_internal", "task_ctx.hidden"])
def test_validated_outline_part_facts_hide_private_declarations(context, field):
    prepared, session, jids, names = context
    model = prepared["recovery_resources"][jids[0]]["recovery_des_model"]
    model["state_variables"][field] = {"scope": "part", "private": True, "domain": []}
    session["symbolic_parts"][names[0]][field] = "PRIVATE_PART_CANARY"
    prepared["part_tracker"][names[0]][field] = "PRIVATE_PART_CANARY"
    session.update(validation_contract_version=2, current_phase="outline")
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    assert "PRIVATE_PART_CANARY" not in prompt
    snapshot = multi_turn_context.resolve_outline_context(prepared, session, f"/parts/{names[0]}/snapshot")
    assert field not in snapshot["value"]
    with pytest.raises(ValueError, match="NEEDS_CONTEXT"):
        multi_turn_context.resolve_outline_context(prepared, session, f"/parts/{names[0]}/{field}")


@pytest.mark.parametrize("field,aliases,value", [
    ("part_location", ["current_location", "location"], "3D Printing Station"),
    ("part_state", ["current_state", "state"], "misplaced"),
    ("part_holder_resource_jid", ["current_holder_resource_jid", "holder_resource_jid"], "recovery-resource-4@localhost"),
    ("inspection_result", [], "pending"),
])
def test_validated_part_projection_does_not_restore_unknown_saved_fields(context, field, aliases, value):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_validated_event

    prepared, session, jids, names = context
    name = names[0]
    prepared["recovery_resources"][jids[0]]["recovery_des_model"]["state_variables"][field] = {"scope": "part"}
    saved = {key: value for key in (field, *aliases)}
    prepared["grounding_context"]["parts"][name] = deepcopy(saved)
    prepared["part_tracker"][name].update(saved)
    session["symbolic_parts"][name].update(saved)
    session.update(validation_contract_version=2, current_phase="outline")
    before = deepcopy(prepared)
    task = {"outline_id": "event_1", "event_name": "observed_disruption_response", "resource_jid": jids[0], "part_name": name}
    ra = {"primitive_support": {"projected_snapshot": {"resource_state": "idle"},
          "projected_part_context": {}, "terminal_state": {"unknown": {"part": [field]}}},
          "geometry_evidence": {"status": "projected", "successor_evidence": {"snapshot": {"resources": {}, "parts": {}}}}}
    multi_turn_validated_event._project(task, ra, {"monitor_context_after": {}}, session)
    session["accepted_outline_prefix"].append(task)
    fact = multi_turn_context.resolve_outline_context(prepared, session, f"/parts/{name}/snapshot")
    assert fact["source"] == "projected_state"
    for key in (field, *aliases):
        assert key not in fact["value"]
        with pytest.raises(ValueError, match="NEEDS_CONTEXT"):
            multi_turn_context.resolve_outline_context(prepared, session, f"/parts/{name}/{key}")
    session["symbolic_parts"][name][field] = None
    assert multi_turn_context.resolve_outline_context(prepared, session, f"/parts/{name}/{field}")["value"] is None
    assert prepared == before


def test_validated_part_retrieval_keeps_untouched_saved_facts(context):
    prepared, session, jids, names = context
    prepared["part_tracker"][names[1]]["processCompleted"] = ["M1"]
    session.update(validation_contract_version=2, accepted_outline_prefix=[{
        "outline_id": "event_1", "resource_jid": jids[0], "part_name": names[0]}])
    fact = multi_turn_context.resolve_outline_context(prepared, session, f"/parts/{names[1]}/snapshot")
    assert fact["value"]["processCompleted"] == ["M1"]


def test_validated_retrieval_renders_public_facts_once_and_sanitizes_errors(context):
    prepared, session, jids, _ = context
    session.update(validation_contract_version=2, current_phase="outline")
    prepared["recovery_resources"][jids[0]]["static_capabilities"]["named_poses"] = {"home": {"x": 0.24681357}}
    refs = [f"/resources/{jids[0]}/static_capabilities", f"/task_contracts/{jids[0]}/PRIVATE_CONTRACT_CANARY"]
    _, turn = asyncio.run(multi_turn_context.handle_outline_context(prepared=prepared, session=session,
        response={"thought": "Read the recorded public facts", "decision": "need_context", "context_requests": refs}, planner=object()))
    session["outline_context_errors"].append({"ref": f"/parts/gear_small/observed_pose", "error": "PRIVATE_RUNTIME_ERROR_CANARY"})
    before = deepcopy(session["outline_context_errors"])
    payload, prompt = multi_turn._build_phase_prompt(prepared, session)
    assert prompt.count('"x": 0.24681357') == 1
    assert prompt.count("Retrieved context") == prompt.count("Missing context") == 1
    assert "PRIVATE_CONTRACT_CANARY" not in prompt and "PRIVATE_RUNTIME_ERROR_CANARY" not in prompt
    assert "Unsupported context reference" in prompt and "NEEDS_CONTEXT: requested public fact is unavailable" in prompt
    assert turn["context_errors"][0]["ref"] == refs[1]
    assert payload["outline_context_errors"] == session["outline_context_errors"] == before


def test_validated_outline_preserves_resource_and_part_goal_conditions(context):
    prepared, session, jids, names = context
    goals = [{"entity_kind": "resource", "entity": jids[0], "field": "temperature", "expected": 0},
             {"entity_kind": "part", "entity": names[0], "field": "inspection_result", "expected": None},
             {"entity_kind": "resource", "entity": jids[1], "field": "controller_ready", "expected": False},
             {"entity_kind": "part", "entity": names[0], "field": "state", "expected": "assembled"},
             {"entity_kind": "part", "entity": names[0], "field": "location", "expected": "assembly_board-v1"}]
    prepared["llm_input"]["goal_conditions"] = deepcopy(goals)
    session.update(validation_contract_version=2, current_phase="outline")
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    section = prompt.split("Recovery goals\n\n")[1].split("\n\n")[0]
    assert [json.loads(line) for line in section.splitlines()] == goals
    assert prepared["llm_input"]["goal_conditions"] == goals
    legacy = multi_turn_prompts._compact_recovery_goals(prepared["llm_input"], projected_parts=[], current_recovery_blockers=[])
    assert "restore gear_small to assembly_board-v1 with part_state=assembled" in legacy


@pytest.mark.parametrize("retrieval", [True, False])
def test_validated_outline_rejection_prompt_keeps_rule_valuation_without_private_models(context, monkeypatch, retrieval):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_validated_event

    prepared, session, jids, _ = context
    session.update(validation_contract_version=2, current_phase="outline", outline_context_retrieval=retrieval)
    canary = "PRIVATE_EVALUATOR_SEQUENCE_CANARY"
    rule_id = "SAFE_shared_area_mutex:recorded_rule_binding"
    event = {"outline_id": "event_1", "event_name": "LLM_PROPOSED_EVENT", "resource_jid": jids[0],
             "expected_end_state": {"resource_state": "idle"}, "rationale": "Proposed state change"}
    interval = ["64795114467/3200000000", "7199457163/320000000"]
    row = {"rule_id": rule_id, "ap_values": {"ap001": True, "ap002": True},
           "possible_violation": True, "definite_violation": True, "cell_index": 9, "interval": interval,
           "from": [canary], "to": [], "private_model": canary}
    check = {"validation_status": "INFEASIBLE", "is_safe": False, "counterexample": deepcopy(row),
             "source_path": "specification/safety/safety_assembly_board-v1_mutex.txt",
             "rule_checks": [{**row, "observation_index": 2, "phase": "at", "time": 1.0}],
             "projection": {"private": canary}, "reason": canary}
    validate = AsyncMock(return_value=(
        {"allowed": True, "feasibility_status": "FEASIBLE", "primitive_support": {"private": canary}},
        {"is_safe": False, "validation_status": "rejected", "reason": canary, "checks": [check]},
    ))
    monkeypatch.setattr(multi_turn_validated_event, "validate_event", validate)
    decision, turn = asyncio.run(multi_turn_validated_event.handle_validated_event(
        session=session, prepared=prepared, planner=object(), response={"thought": "Propose one event", "next_transition": event}))
    assert decision == "need_revision" and session["accepted_outline_prefix"] == []
    finding = turn["validation_findings"][0]
    assert finding["expected_start_state"] == {"resource_state": "failed"}
    assert finding["expected_end_state"] == event["expected_end_state"]
    public = finding["evidence"]["safety_checks"][0]["counterexample"]
    assert public == {"rule_id": rule_id, "ap_values": {"ap001": True, "ap002": True},
                      "possible_violation": True, "definite_violation": True, "cell_index": 9,
                      "interval": interval, "observation_index": 2, "phase": "at", "time": 1.0}
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    assert rule_id in prompt and '"ap001": true' in prompt and '"ap002": true' in prompt
    assert '"feasibility_status": "FEASIBLE"' in prompt and '"validation_status": "rejected"' in prompt
    assert "PRIVATE_EVALUATOR_SEQUENCE_CANARY" not in prompt
    assert "expected_start_state" in prompt and "expected_end_state" in prompt
    assert '"definite_violation": true' in prompt and '"cell_index": 9' in prompt
    assert json.dumps(interval, separators=(",", ":")) in "".join(prompt.split())
    assert (multi_turn_prompts._outline_safety_checks_for_prompt([check])[0]["counterexample"]["interval"]
            is not check["counterexample"]["interval"])


def test_outline_safety_feedback_preserves_uncertain_boolean_values_and_excludes_opaque_payloads():
    helper = multi_turn_prompts._outline_safety_checks_for_prompt
    rows = [{"validation_status": "NEEDS_CONTEXT", "counterexample": {
        "rule_id": "SAFE_shared_area_mutex:uncertain", "ap_values": {"ap001": [False, True], "ap002": True},
        "possible_violation": True, "from": ["PRIVATE_STATE"], "to": ["PRIVATE_STATE"]}},
        {"counterexample": {"rule_id": "PRIVATE_RULE", "ap_values": {"ap001": {"primitive": "PRIVATE_CANARY"}}}}]
    result = helper(rows)
    assert len(result) == 1
    assert result[0]["counterexample"]["ap_values"] == {"ap001": [False, True], "ap002": True}
    assert "PRIVATE" not in json.dumps(result)
    assert not {"definite_violation", "cell_index", "interval"} & result[0]["counterexample"].keys()


@pytest.mark.parametrize("fields", [
    {"definite_violation": "PRIVATE_CANARY", "cell_index": True, "interval": ["PRIVATE_CANARY", "1"]},
    {"definite_violation": None, "cell_index": -1, "interval": ["0"]},
    {"definite_violation": 1, "cell_index": "9", "interval": [{"private": "PRIVATE_CANARY"}, "1"]},
])
def test_outline_safety_feedback_does_not_invent_or_expose_invalid_cell_facts(fields):
    row = {"counterexample": {"rule_id": "SAFE_shared_area_mutex", "ap_values": {"ap001": [True]}, **fields}}
    before = deepcopy(row)
    public = multi_turn_prompts._outline_safety_checks_for_prompt([row])[0]["counterexample"]
    assert public == {"rule_id": "SAFE_shared_area_mutex", "ap_values": {"ap001": [True]}}
    assert row == before


def test_validator_retrieval_updates_scoped_inputs_without_changing_source(context):
    prepared, session, jids, _ = context
    jid = jids[0]
    witnesses = [{"primitive": "move_to_named_pose", "event_name": "declared_motion", "step_index": 0,
                  "params": {"pose_name": "home"}, "source": "saved_observations",
                  "captured_at": prepared["source_observation_captured_at"], "revision": 4}]
    prepared["recovery_resources"][jid]["parameter_witnesses"] = witnesses
    before = deepcopy(prepared)
    session.update(primitive_support_semantics="successor_primitive_union",
                   validation_resource_context={jid: {"parameter_witnesses": []}})
    ref = f"/resources/{jid}/parameter_witnesses"
    turn = {}
    changed = asyncio.run(multi_turn_validated_event._retrieve(
        prepared, session, object(), [ref, "/private/arbitrary_json"], turn))
    assert changed is True
    assert multi_turn_validated_event._validation_resource_input(prepared, session, jid)["parameter_witnesses"] == witnesses
    evidence = turn["validation_context_exchanges"][0]
    assert evidence["returned_facts"][ref]["captured_at"] == prepared["source_observation_captured_at"]
    assert evidence["missing_facts"] == [{"ref": "/private/arbitrary_json", "reason": "Unsupported validator context reference"}]
    assert asyncio.run(multi_turn_validated_event._retrieve(prepared, session, object(), [ref], turn)) is False
    assert prepared == before
    assert session["accepted_outline_prefix"] == []


def test_validator_retrieval_retries_same_proposal_with_new_parameter_witnesses(context):
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    prepared, session, jids, _ = context
    jid = jids[0]
    ref = f"/resources/{jid}/parameter_witnesses"
    witnesses = [{"primitive": "move_to_named_pose", "event_name": "declared_motion", "step_index": 0,
                  "params": {"pose_name": "home"}, "source": "saved_observations", "revision": 4}]
    prepared["recovery_resources"][jid]["parameter_witnesses"] = witnesses
    session.update(primitive_support_semantics="successor_primitive_union",
                   validation_resource_context={jid: {"parameter_witnesses": []}})
    task = {"outline_id": "RECOVERY1", "event_name": "unlisted_event", "resource_jid": jid,
            "expected_start_state": {"resource_state": "failed"}, "expected_end_state": {"resource_state": "idle"}}
    requests = []

    async def resource_validation(*, resource_jid, payload):
        assert resource_jid == jid
        requests.append(deepcopy(payload))
        present = payload["candidates"][0]["physical_input"]["parameter_witnesses"] == witnesses
        return {"validator_jid": jid, "results": [{
            "candidate_fingerprint": recovery_validation_fingerprint(task),
            "primitive_support_semantics": "successor_primitive_union",
            "allowed": present, "feasibility_status": "FEASIBLE" if present else "NEEDS_CONTEXT",
            "context_requests": [] if present else [ref], "geometry_evidence": {"status": "NEEDS_CONTEXT"}}]}

    safety = AsyncMock(return_value={"validator_jid": "cca", "results": [{
        "candidate_fingerprint": recovery_validation_fingerprint(task),
        "is_safe": False, "validation_status": "unresolved", "checks": []}]})
    product = SimpleNamespace(request_recovery_outline_physical_validation=resource_validation,
                              request_recovery_outline_safety_validation=safety)
    turn = {}
    ra, cca = asyncio.run(multi_turn_validated_event.validate_event(
        task=task, session=session, prepared=prepared, planner=SimpleNamespace(product_agent=product), turn=turn))
    assert len(requests) == 2
    assert all(request["candidates"][0]["task"] == task for request in requests)
    assert all(request["primitive_support_semantics"] == "successor_primitive_union" for request in requests)
    assert ra["feasibility_status"] == "FEASIBLE" and cca["validation_status"] == "unresolved"
    assert session["accepted_outline_prefix"] == []
    assert safety.await_count == 1


def test_marked_validator_request_rejects_historical_reply_before_cca(context):
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    prepared, session, jids, _ = context
    session["primitive_support_semantics"] = "successor_primitive_union"
    task = {"outline_id": "RECOVERY1", "event_name": "unlisted_event", "resource_jid": jids[0],
            "expected_end_state": {"resource_state": "idle"}}
    resource = AsyncMock(return_value={"validator_jid": jids[0], "results": [{
        "candidate_fingerprint": recovery_validation_fingerprint(task), "allowed": True,
        "primitive_support": {"semantics": "primitive_transition"}}]})
    safety = AsyncMock(side_effect=AssertionError("Historical RA reply reached CCA"))
    product = SimpleNamespace(request_recovery_outline_physical_validation=resource,
                              request_recovery_outline_safety_validation=safety)
    with pytest.raises(ValueError, match="different primitive_support_semantics"):
        asyncio.run(multi_turn_validated_event.validate_event(
            task=task, session=session, prepared=prepared, planner=SimpleNamespace(product_agent=product), turn={}))
    safety.assert_not_awaited()
    assert session["accepted_outline_prefix"] == []


def test_validator_private_retrieval_stays_out_of_proposal_prompt(context):
    prepared, session, jids, _ = context
    session.update(current_phase="outline", validation_contract_version=2,
                   primitive_support_semantics="successor_primitive_union",
                   validation_resource_context={jids[0]: {"recovery_des_model": {
                       "private": "PRIVATE_VALIDATOR_MODEL_CANARY"}, "parameter_witnesses": [{
                           "params": {"private": "PRIVATE_VALIDATOR_PARAMS_CANARY"}}]}})
    prompt = multi_turn._build_phase_prompt(prepared, session)[1]
    assert "PRIVATE_VALIDATOR" not in prompt


def _accepted_public_case(context, *, pickup=False):
    prepared, session, jids, names = context
    jid, part = jids[1], names[0] if pickup else names[1]
    session.update(validation_contract_version=2, public_safety_context_version=1)
    declarations = prepared['recovery_resources'][jid]['recovery_des_model']['state_variables']
    declarations.update({
        'held_part': {'scope': 'resource', 'domain': [None, part]},
        'resource_location': {'scope': 'resource', 'domain': ['Buffer For Machined parts', 'assembly_board-v1']},
        'part_state': {'scope': 'part', 'domain': ['ready', 'in_gripper']},
        'part_location': {'scope': 'part', 'domain': ['Buffer For Machined parts', jid]},
    })
    prior = session['symbolic_resources'][jid]
    prior.update(current_state='picked' if not pickup else 'placed',
                 current_location='Buffer For Machined parts', gripper_state='closed' if not pickup else 'open',
                 held_part_location=jid,
                 resource_state='picked' if pickup else 'placed', held_part=part if pickup else None,
                 resource_location='assembly_board-v1' if pickup else 'Buffer For Machined parts')
    current = session['symbolic_parts'][part]
    current.update(current_state='misplaced' if pickup else 'in_gripper',
                   current_location=None if pickup else 'ur5e-3',
                   current_holder_resource_jid=None if pickup else jid,
                   part_holder_resource_jid=None if pickup else jid,
                   part_state='in_gripper' if pickup else 'ready',
                   part_location=jid if pickup else 'Buffer For Machined parts',
                   projected_pose=[0., 0., 1., 0., 0., 0., 1.], observed_pose_current=False)
    session['derived_resource_snapshots'] = {jid: deepcopy(prior)}
    session['accepted_outline_prefix'] = [{'resource_jid': jid, 'part_name': part}]
    session['projected_physical_evidence'] = {
        'jids': {'ur5e-3': jid}, 'provenance': {'kind': 'projected'},
        'snapshot': {'resources': {'ur5e-3': {
            'held_part': part if pickup else None, 'gripper_state': 'closed' if pickup else 'open',
            'current_pose': [0., 0., 1.5, 0., 0., 0., 1.]}},
            'parts': {part: {'current_pose': current['projected_pose'], 'contained_by': None}}},
    }
    return prepared, session, jid, part


@pytest.mark.parametrize('pickup', [False, True])
def test_current_public_facts_do_not_relabel_old_capture(context, pickup):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_context import public_part_snapshot, public_resource_snapshot

    prepared, session, jid, part = _accepted_public_case(context, pickup=pickup)
    original = deepcopy((prepared, session))
    resource = public_resource_snapshot(prepared, session, jid)
    product = public_part_snapshot(prepared, session, part)
    assert resource['resource_state'] == ('picked' if pickup else 'placed')
    assert resource['held_part'] == (part if pickup else None)
    assert resource['gripper_state'] == ('closed' if pickup else 'open')
    assert not {'current_state', 'current_location', 'held_part_location'} & resource.keys()
    assert product['part_state'] == ('in_gripper' if pickup else 'ready')
    assert product['part_location'] == (jid if pickup else 'Buffer For Machined parts')
    assert not {'current_state', 'current_location', 'current_holder_resource_jid', 'part_holder_resource_jid'} & product.keys()
    assert product['projected_pose'] == session['symbolic_parts'][part]['projected_pose']
    assert (prepared, session) == original


def test_projected_public_facts_preserve_exact_nonrobot_declared_fields(context):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_context import public_resource_snapshot

    prepared, session, jid, _ = _accepted_public_case(context)
    entry = prepared['recovery_resources'][jid]
    entry['resource_type'] = 'printer'
    entry['recovery_des_model']['state_variables'] = {
        'build_stage': {'scope': 'resource', 'domain': ['cooling']},
        'current_state': {'scope': 'resource', 'domain': ['finished']},
        'material_remaining': {'scope': 'resource', 'domain': [None]},
        'private_temperature': {'scope': 'resource', 'private': True, 'domain': [99]},
    }
    session['symbolic_resources'][jid].update(build_stage='cooling', current_state='finished',
        material_remaining=None, private_temperature=99)
    resource = public_resource_snapshot(prepared, session, jid)
    assert resource['build_stage'] == 'cooling'
    assert resource['current_state'] == 'finished'
    assert resource['material_remaining'] is None
    assert 'private_temperature' not in resource




def test_projected_public_pose_and_peer_sources_do_not_become_new_measurements(context):
    prepared, session, jid, part = _accepted_public_case(context)
    peer = next(value for value in prepared['recovery_resources'] if value != jid)
    ref = f'/resources/{jid}/current_pose'
    expected = session['projected_physical_evidence']['snapshot']['resources']['ur5e-3']['current_pose']
    session['outline_observed_context'] = {ref: {'value': deepcopy(expected), 'captured_at': 'earlier observation'}}
    pose = multi_turn_context.resolve_outline_context(prepared, session, ref)
    snapshot = multi_turn_context.resolve_outline_context(prepared, session, f'/resources/{jid}/snapshot')
    other = multi_turn_context.resolve_outline_context(prepared, session, f'/resources/{peer}/snapshot')
    assert pose['value'] == snapshot['value']['current_pose'] == expected
    assert pose['source'] == snapshot['field_sources']['current_pose'] == 'projected_state'
    assert pose['captured_at'] is None
    assert pose['source_observation_captured_at'] == prepared['source_observation_captured_at']
    assert other['source'] == 'saved_observations'
    assert other['captured_at'] == prepared['source_observation_captured_at']
    assert 'current_pose_captured_at' not in snapshot['value']
    held = multi_turn_context.resolve_outline_context(prepared, session, f'/resources/{jid}/held_part')
    assert held['value'] is None and held['source'] == 'projected_state'
    for missing in (f'/resources/{jid}/current_state', f'/parts/{part}/current_holder_resource_jid'):
        with pytest.raises(ValueError, match='NEEDS_CONTEXT'):
            multi_turn_context.resolve_outline_context(prepared, session, missing)


def test_current_public_facts_do_not_resurrect_missing_declared_effects(context):
    prepared, session, jid, part = _accepted_public_case(context)
    prepared['recovery_resources'][jid]['recovery_snapshot']['resource_location'] = 'captured location'
    session['symbolic_resources'][jid].pop('resource_location')
    prepared['part_tracker'] = {part: {'part_state': 'captured state'}}
    session['symbolic_parts'][part].pop('part_state')
    assert 'resource_location' not in multi_turn_context.public_resource_snapshot(prepared, session, jid)
    assert 'part_state' not in multi_turn_context.public_part_snapshot(prepared, session, part)


def test_current_public_view_is_versioned_and_matches_the_prompt(context):
    prepared, session, jid, part = _accepted_public_case(context)
    session.pop('public_safety_context_version')
    historical_resource = multi_turn_context.public_resource_snapshot(prepared, session, jid)
    historical_part = multi_turn_context.public_part_snapshot(prepared, session, part)
    assert historical_resource['current_state'] == 'picked'
    assert historical_part['current_holder_resource_jid'] == jid
    session['public_safety_context_version'] = 1
    payload = {'llm_input': prepared['llm_input'], 'recovery_resources': prepared['recovery_resources']}
    # Exercise the actual public-view renderer without invoking safety derivation,
    # generation, observations, or validators.
    prompt = multi_turn_prompts._render_validated_event_prompt(payload, session)
    block = prompt.split('Current resource states\n\n', 1)[1].split('\n\nCurrent part states', 1)[0]
    rows = [json.loads(line) for line in block.splitlines()]
    resource = next(row for row in rows if row['resource_jid'] == jid)
    assert resource == multi_turn_context.public_resource_snapshot(prepared, session, jid)
    block = prompt.split('Current part states\n\n', 1)[1].split('\n\nRecovery goals', 1)[0]
    rows = [json.loads(line) for line in block.splitlines()]
    product = next(row for row in rows if row['part_name'] == part)
    assert product == multi_turn_context.public_part_snapshot(prepared, session, part)
    assert 'current_state' not in product and 'current_holder_resource_jid' not in product


def _public_safety_context_case():
    from cais_spade_llm.recovery_framework.outline_validation import load_outline_specifications

    prepared, session, jids, _ = context.__wrapped__()
    resources = {
        rid: {
            "joint_names": ["slide"],
            "joint_positions": [0.0],
            "held_part": None,
            "current_pose": [offset, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "grasp_transform": None,
        }
        for rid, offset in (("ur5e-3", 0.0), ("ur5e-4", 3.0))
    }
    parts = {
        name: {"current_pose": [-3.0, index, 0.0, 0.0, 0.0, 0.0, 1.0], "contained_by": None}
        for index, name in enumerate(("gear_small", "KET4_Square_4mm"))
    }
    models = {
        rid: {
            "configuration": {
                "joint_names": ["slide"],
                "root": "world",
                "root_xyz": [offset, 0.0, 0.0],
                "root_rpy": [0.0, 0.0, 0.0],
                "reference_link": "body",
                "joints": [
                    {
                        "name": "slide",
                        "parent": "world",
                        "child": "body",
                        "type": "prismatic",
                        "xyz": [0.0, 0.0, 0.0],
                        "rpy": [0.0, 0.0, 0.0],
                        "axis": [1.0, 0.0, 0.0],
                    }
                ],
                "components": [{"id": "body_box", "link": "body", "bounds": [[-0.05, 0.05]] * 3}],
            }
        }
        for rid, offset in (("ur5e-3", 0.0), ("ur5e-4", 3.0))
    }
    geometry = {
        "frame": "world",
        "resources": {rid: {} for rid in resources},
        "parts": {name: {"local_bounds": [[-0.05, 0.05]] * 3} for name in parts},
        "regions": {"assembly_board-v1": {"bounds": [[0.8, 1.2], [-0.1, 0.1], [-0.1, 0.1]]}},
    }
    geometry["parts"]["gear_small"]["target"] = "Gear_Plate/Gear_Shaft_1"
    prepared["physical_evidence"] = {
        "snapshot": {"frame": "world", "resources": resources, "parts": parts},
        "geometry": geometry,
        "resource_models": models,
        "jids": {"ur5e-3": jids[1], "ur5e-4": jids[0]},
    }
    paths = [
        "cais_spade_llm/specification/safety/safety_assembly_board-v1_mutex.txt",
        "cais_spade_llm/specification/safety/safety_assembly_board-v1_gear_small_before_KET4_Square_4mm.txt",
    ]
    prepared["outline_safety_specifications"] = load_outline_specifications(
        paths, Path(__file__).resolve().parents[1]
    )
    session.update(public_safety_context_version=1, validation_contract_version=2)
    return prepared, session


@pytest.mark.parametrize("missing_participant", [False, True])
def test_public_mutex_bindings_use_recorded_custody_and_motion_declarations(missing_participant):
    from cais_spade_llm.recovery_framework.outline_validation import load_outline_specifications

    prepared, session = _public_safety_context_case()
    physical = prepared["physical_evidence"]
    physical["geometry"]["regions"] = {name: {"bounds": [[.8, 1.2], [-.1, .1], [-.1, .1]]}
        for name in ("M2.front_access", "M2.side_access", "M2.workholding")}
    physical["environment_models"] = {rid: {"state_variables": {"held_part": {"type": ["string", "null"]}},
        "private_sequences": "PRIVATE_OWNER_SENTINEL"} for rid in physical["jids"]}
    physical["jids"]["M2"] = "machine@localhost"
    physical["snapshot"]["resources"]["M2"] = {}
    physical["geometry"]["resources"]["M2"] = {"stationary_only": True,
        "component_bounds": [{"bounds": [[9., 10.], [9., 10.], [0., 1.]]}]}
    physical["resource_models"]["M2"] = {}
    physical["environment_models"]["M2"] = {"state_variables": {}}
    if missing_participant:
        physical["resource_models"]["ur5e-3"].pop("configuration")
        physical["environment_models"]["ur5e-3"] = {"state_variables": {}}
    prepared["outline_safety_specifications"] = load_outline_specifications(
        ["cais_spade_llm/specification/safety/safety_M2_mutex.txt"], Path(__file__).resolve().parents[1])
    original = deepcopy((prepared, session))
    if missing_participant:
        with pytest.raises(ValueError, match="AP bindings are unavailable"):
            multi_turn_context.public_safety_context_fact(prepared, session, "/safety_rules/SAFE_M2_mutex/instances")
    else:
        fact = multi_turn_context.public_safety_context_fact(prepared, session, "/safety_rules/SAFE_M2_mutex/instances")
        assert len(fact["value"]) == 1
        aps = fact["value"][0]["aps"]
        assert {ap["binding"]["resource"] for ap in aps} == {"ur5e-3", "ur5e-4"}
        assert {ap["binding"]["region"] for ap in aps} == set(physical["geometry"]["regions"])
        assert "PRIVATE_OWNER_SENTINEL" not in json.dumps(fact)
    assert (prepared, session) == original


def test_public_safety_exact_meanings_bindings_and_rule_namespaces_without_private_models():
    prepared, session = _public_safety_context_case()
    before = deepcopy((prepared, session))
    prepared["physical_evidence"]["resource_models"]["ur5e-3"]["private_model"] = (
        "SECRET_PRIVATE_MODEL"
    )
    result = multi_turn_context.public_outline_safety_context(prepared, session)
    rules = result["specifications"]
    instances = result["instances"]
    assert len(rules) == len(instances) == 2
    assert (
        rules[0]["aps"]
        == prepared["outline_safety_specifications"][0]["document"]["catalog"]["specifications"][0][
            "aps"
        ]
    )
    mutex = next(r for r in instances if r["id"] == "SAFE_shared_area_mutex")
    precedence = next(r for r in instances if r["id"] == "SAFE_gear_small_before_KET4_Square_4mm")
    assert mutex["aps"][0]["binding"] == {"region": "assembly_board-v1", "resource": "ur5e-3"}
    assert mutex["aps"][1]["binding"] == {"region": "assembly_board-v1", "resource": "ur5e-4"}
    assert precedence["aps"][0]["binding"] == {
        "part": "KET4_Square_4mm",
        "region": "assembly_board-v1",
    }
    assert precedence["aps"][1]["binding"] == {
        "part": "gear_small",
        "process": "assembly",
        "target": "Gear_Plate/Gear_Shaft_1",
    }
    assert mutex["aps"][0]["label"] == precedence["aps"][0]["label"] == "ap001"
    text = json.dumps(result)
    assert (
        "SECRET_PRIVATE_MODEL" not in text
        and "dfa_dot" not in text
        and "event_projection_contracts" not in text
    )
    prepared["physical_evidence"]["resource_models"]["ur5e-3"].pop("private_model")
    assert (prepared, session) == before


def test_public_safety_endpoint_occupancy_uses_robot_and_carried_part_geometry_not_tcp_location():
    prepared, session = _public_safety_context_case()
    scene = prepared["physical_evidence"]
    scene["snapshot"]["resources"]["ur5e-3"].update(
        held_part="KET4_Square_4mm", grasp_transform=[1, 0, 0, 0, 0, 0, 1]
    )
    result = multi_turn_context.public_outline_safety_context(prepared, session)
    assert scene["snapshot"]["resources"]["ur5e-3"]["current_pose"][0] == 0
    assert result["current_occupancy"]["assembly_board-v1"]["resources"]["ur5e-3"] == [True]
    assert result["current_occupancy"]["assembly_board-v1"]["parts"]["KET4_Square_4mm"] == [True]
    assert (
        result["source"]["basis"] == "saved_observations"
        and result["source"]["coverage"] == "endpoint"
    )
    assert result["source"]["execution_authorized"] is False


def test_public_safety_scope_includes_all_resources_and_regions_without_scenario_lists():
    prepared, session = _public_safety_context_case()
    scene = prepared["physical_evidence"]
    scene["snapshot"]["resources"]["Conveyor"] = {"held_part": None}
    scene["geometry"]["resources"]["Conveyor"] = {
        "component_bounds": [{"id": "conveyor_body", "bounds": [[5, 6], [0, 1], [0, 1]]}]
    }
    scene["jids"]["Conveyor"] = "conveyor@localhost"
    scene["geometry"]["regions"]["Buffer For Machined parts"] = {
        "bounds": [[5, 7], [-1, 2], [-1, 2]]
    }
    prepared["outline_safety_specifications"][0]["document"]["requirement_scopes"].append(
        {"specification": "SAFE_shared_area_mutex", "region": "Buffer For Machined parts"}
    )
    result = multi_turn_context.public_outline_safety_context(prepared, session)
    assert len([r for r in result["instances"] if r["id"] == "SAFE_shared_area_mutex"]) == 6
    assert result["regions"] == ["Buffer For Machined parts", "assembly_board-v1"]
    assert {r["resource"] for r in result["resource_bindings"]} == {"Conveyor", "ur5e-3", "ur5e-4"}
    assert result["current_occupancy"]["Buffer For Machined parts"]["resources"]["Conveyor"] == [
        True
    ]


def test_public_safety_unknown_values_and_missing_evidence_remain_distinct(monkeypatch):
    from cais_spade_llm.resources import recovery_geometry

    prepared, session = _public_safety_context_case()
    original = recovery_geometry._checkpoint

    def uncertain(*args, **kwargs):
        row = original(*args, **kwargs)
        row["occupancy_possibilities"]["assembly_board-v1"]["ur5e-3"] = [False, True]
        return row

    monkeypatch.setattr(recovery_geometry, "_checkpoint", uncertain)
    result = multi_turn_context.public_outline_safety_context(prepared, session)
    assert result["current_occupancy"]["assembly_board-v1"]["resources"]["ur5e-3"] == [False, True]
    prepared["physical_evidence"]["resource_models"]["ur5e-3"].pop("configuration")
    missing = multi_turn_context.public_outline_safety_context(prepared, session)
    assert missing["current_occupancy"]["assembly_board-v1"]["resources"]["ur5e-3"] is None
    assert "NEEDS_CONTEXT" in missing["missing_context"][0]["reason"]


def test_public_safety_accepted_scene_changes_revision_without_reusing_original_observations():
    prepared, session = _public_safety_context_case()
    before = deepcopy(prepared)
    first = multi_turn_context.public_outline_safety_context(prepared, session)
    session["projected_physical_evidence"] = deepcopy(prepared["physical_evidence"])
    session["projected_physical_evidence"]["snapshot"]["resources"]["ur5e-3"]["joint_positions"] = [
        1.0
    ]
    session["accepted_outline_prefix"] = [{"outline_id": "RECOVERY1"}]
    second = multi_turn_context.public_outline_safety_context(prepared, session)
    assert second["source"]["revision"] != first["source"]["revision"]
    assert second["source"]["basis"] == "projected_state"
    assert second["source"]["captured_at"] == first["source"]["captured_at"]
    assert second["current_occupancy"]["assembly_board-v1"]["resources"]["ur5e-3"] == [True]
    session.pop("projected_physical_evidence")
    missing = multi_turn_context.public_outline_safety_context(prepared, session)
    assert missing["missing_context"] and not missing["current_occupancy"]
    assert prepared == before


@pytest.mark.parametrize(
    "ref",
    [
        "/physical_evidence",
        "/regions/assembly_board-v1/geometry/parts",
        "/safety_rules/SAFE_shared_area_mutex/dfa_dot",
        "/resources/ur5e-3/private_model",
    ],
)
def test_public_safety_no_arbitrary_private_access(ref):
    prepared, session = _public_safety_context_case()
    with pytest.raises(ValueError, match="Unsupported"):
        multi_turn_context.public_safety_context_fact(prepared, session, ref)


def test_public_safety_offline_retrieval_and_repeated_requests_never_call_live_or_commit(
    monkeypatch,
):
    prepared, session = _public_safety_context_case()

    async def forbidden(*args, **kwargs):
        raise AssertionError("live observation is forbidden")

    monkeypatch.setattr(multi_turn_context, "_observe_missing", forbidden)
    monkeypatch.setattr(multi_turn_context, "_observe_resource_pose", forbidden)
    response = {
        "thought": "Read selected safety facts",
        "decision": "need_context",
        "context_requests": [
            "/safety_rules/SAFE_shared_area_mutex/instances",
            "/regions/assembly_board-v1/occupancy",
        ],
    }
    before = deepcopy((prepared, session["accepted_outline_prefix"]))
    decision, entry = asyncio.run(
        multi_turn_context.handle_outline_context(
            prepared=prepared, session=session, response=response, planner=object()
        )
    )
    assert decision == "need_context" and not entry["context_errors"]
    assert entry["served_context"][response["context_requests"][1]]["value"]["resources"][
        "ur5e-3"
    ] == [False]
    assert (prepared, session["accepted_outline_prefix"]) == before
    _, entry = asyncio.run(
        multi_turn_context.handle_outline_context(
            prepared=prepared, session=session, response=response, planner=object()
        )
    )
    assert session["status"] == "NEEDS_CONTEXT"
    assert "no new information" in entry["context_errors"][-1]["error"]


def test_public_safety_feedback_binds_only_the_exact_recorded_rule_instance():
    prepared, session = _public_safety_context_case()
    value = multi_turn_context.public_outline_safety_context(prepared, session)
    mutex = next(r for r in value["instances"] if r["id"] == "SAFE_shared_area_mutex")
    findings = [
        {
            "evidence": {
                "safety_checks": [
                    {
                        "counterexample": {
                            "rule_id": mutex["rule_id"],
                            "ap_values": {"ap001": [True], "ap002": [True]},
                        }
                    }
                ]
            }
        }
    ]
    assert multi_turn_context.safety_feedback_bindings(value, findings) == [mutex]
    assert (
        multi_turn_context.safety_feedback_bindings(
            value, [{"evidence": {"safety_checks": [{"counterexample": {"rule_id": "unknown"}}]}}]
        )
        == []
    )


def test_public_safety_old_sessions_do_not_gain_new_references():
    prepared, session = _public_safety_context_case()
    session.pop("public_safety_context_version")
    with pytest.raises(ValueError, match="Unsupported"):
        multi_turn_context.resolve_outline_context(prepared, session, "/regions")


def test_public_safety_prompt_exposes_facts_once_and_exact_rejection_bindings():
    prepared, session = _public_safety_context_case()
    session.update(current_phase="outline", outline_mode="incremental_candidates_validated")
    records = multi_turn_context.public_outline_safety_context(prepared, session)
    mutex = next(row for row in records["instances"] if row["id"] == "SAFE_shared_area_mutex")
    session["outline_validation_findings"] = [
        {
            "constraint_code": "safety_rule_violation",
            "evidence": {
                "safety_checks": [
                    {
                        "validation_status": "INFEASIBLE",
                        "counterexample": {
                            "rule_id": mutex["rule_id"],
                            "ap_values": {"ap001": [True], "ap002": [True]},
                        },
                    }
                ]
            },
        }
    ]
    before = deepcopy((prepared, session))
    payload, prompt = multi_turn._build_phase_prompt(prepared, session)
    assert payload["outline_safety_feedback_bindings"] == [mutex]
    for record in prepared["outline_safety_specifications"]:
        rule = record["document"]["catalog"]["specifications"][0]
        assert prompt.count(f"- {rule['id']}: {rule['requirement']}") == 1
        retrieved = multi_turn_context.resolve_outline_context(prepared, session, f"/safety_rules/{rule['id']}")
        for ap in rule["aps"]:
            assert ap["meaning"] in json.dumps(retrieved)
    for reference in (
        "/safety_rules",
        "/safety_rules/{specification}/instances",
        "/regions/{region}/occupancy",
    ):
        assert reference in prompt
    assert "Current geometric occupancy" in prompt
    assert "Safety rule bindings in validation feedback" in prompt
    assert "[false, true] is unresolved; null means evidence is unavailable" in prompt
    assert "dfa_dot" not in prompt and "event_projection_contracts" not in prompt
    assert "model_fingerprint" not in prompt and "snapshot_fingerprint" not in prompt
    assert "model_fingerprint" in json.dumps(payload["outline_safety_context"])
    assert (prepared, session) == before


def test_public_safety_missing_snapshot_owner_does_not_omit_a_declared_alternative():
    prepared, session = _public_safety_context_case()
    prepared["physical_evidence"]["snapshot"]["resources"].pop("ur5e-4")
    result = multi_turn_context.public_outline_safety_context(prepared, session)
    assert {row["resource"] for row in result["resource_bindings"]} == {"ur5e-3", "ur5e-4"}
    assert result["current_occupancy"]["assembly_board-v1"]["resources"]["ur5e-4"] is None
    assert result["missing_context"]


def test_public_safety_invalid_selected_schema_cannot_expose_arbitrary_saved_fields():
    prepared, session = _public_safety_context_case()
    prepared["outline_safety_specifications"][0]["document"]["catalog"]["specifications"][0]["aps"][
        0
    ]["private_model"] = "SECRET_PRIVATE_MODEL"
    result = multi_turn_context.public_outline_safety_context(prepared, session)
    assert "SECRET_PRIVATE_MODEL" not in json.dumps(result)
    assert any("supported public schema" in row["reason"] for row in result["missing_context"])


def test_public_safety_scope_never_exposes_private_evaluator_fields():
    prepared, session = _public_safety_context_case()
    scope = prepared["outline_safety_specifications"][0]["document"]["requirement_scopes"][0]
    scope["ap_groundings"] = {"ap001": {"state_field": "SECRET_PRIVATE_EVALUATOR"}}
    result = multi_turn_context.public_outline_safety_context(prepared, session)
    assert "SECRET_PRIVATE_EVALUATOR" not in json.dumps(result)
    assert any("bindings require" in row["reason"] for row in result["missing_context"])


@pytest.mark.parametrize("scenario", ["Conveyor breakdown", "ur5e-1 breakdown",
    "Machining breakdown during part processing", "Part slippage"])
def test_large_selected_safety_sets_keep_every_requirement_and_retrieve_details(scenario):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import _outline_selected_safety_text
    from cais_spade_llm.recovery_framework.recovery_case_safety import default_safety_files
    from cais_spade_llm.recovery_framework import ROOT

    rows = []
    for name in default_safety_files(scenario):
        document = json.loads((ROOT / name).read_text())
        rows.extend({**row, "source_path": name, "source_sha256": "source",
                     "requirement_scopes": document["requirement_scopes"]}
                    for row in document["catalog"]["specifications"])
    result = _outline_selected_safety_text({"specifications": rows})
    for row in rows:
        assert row["id"] in result and row["requirement"] in result
    assert len(result) < len(json.dumps(rows)) / 2
    assert "source_sha256" not in result


def test_compact_safety_occupancy_preserves_scope_uncertainty_and_complete_discovery():
    prepared, session = _public_safety_context_case()
    full = multi_turn_context.public_outline_safety_context(prepared, session)
    region = 'assembly_board-v1'
    full['current_occupancy'][region]['resources']['ur5e-3'] = [False, True]
    full['current_occupancy'][region]['parts']['KET4_Square_4mm'] = None
    before = deepcopy(full)
    jid = prepared['physical_evidence']['jids']['ur5e-3']
    compact = multi_turn_context.compact_public_safety_context(full,
        resource_jids={jid}, part_names={'KET4_Square_4mm'})
    assert compact['current_occupancy'][region] == {
        'resources':{'ur5e-3':[False,True]}, 'parts':{'KET4_Square_4mm':None}}
    for field in ('regions','source','resource_bindings','missing_context'):
        assert compact[field] == before[field]
    assert full == before
    retrieved = multi_turn_context.public_safety_context_fact(prepared, session, '/regions/assembly_board-v1/occupancy')
    assert set(retrieved['value']['resources']) == {'ur5e-3','ur5e-4'}
    assert set(retrieved['value']['parts']) == {'KET4_Square_4mm','gear_small'}
    assert multi_turn_context.compact_public_safety_context(full)['current_occupancy'] == full['current_occupancy']


def test_scoped_prompt_uses_exact_selection_for_initial_safety_occupancy(monkeypatch):
    prepared, session = _public_safety_context_case()
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import _build_phase_prompt
    session['current_phase'] = 'outline'
    selection = multi_turn_context.select_outline_context(prepared, session)
    jid = prepared['physical_evidence']['jids']['ur5e-3']
    selection.update(resource_jids=[jid],part_names=['KET4_Square_4mm'])
    monkeypatch.setattr(multi_turn_context, 'select_outline_context', lambda *args:deepcopy(selection))
    payload, _ = _build_phase_prompt(prepared, session)
    occupancy = payload['outline_safety_context']['current_occupancy']['assembly_board-v1']
    assert set(occupancy['resources']) == {'ur5e-3'}
    assert set(occupancy['parts']) == {'KET4_Square_4mm'}


_RECORDED_OWNER_FIELDS = json.loads((Path(__file__).parent / "fixtures/recovery_framework/conveyor_b66d_declared_fields.json").read_text())


@pytest.mark.parametrize("jid,row", list(_RECORDED_OWNER_FIELDS["resources"].items()))
def test_recorded_conveyor_capture_declared_fields_survive_preparation_and_discovery(jid, row):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_session import RecoverySessionMixin
    from cais_spade_llm.resources.resource_profile import resource_snapshot_declared_fields

    raw = deepcopy(row["snapshot"])
    raw["resource_type"] = row["resource_type"]
    original = deepcopy(raw)
    seen = []

    def descriptor(*, snapshot):
        seen.append(deepcopy(snapshot))
        return {"state_variables": deepcopy(row["state_variables"]), "descriptor_fingerprint": "recorded"}

    owner = SimpleNamespace(jid=jid, static_capabilities=deepcopy(row["static_capabilities"]),
                            recovery_des_model=descriptor, recovery_execution_primitive_catalog=lambda: [])
    planner = SimpleNamespace(resource_agents=[owner], _resource_by_jid=lambda value: owner,
        _build_resource_search_state=lambda **kwargs: {}, _pending_resource_tasks=lambda value: [],
        logger=Mock(), product_agent=SimpleNamespace())
    _, snapshot, entries = asyncio.run(RecoverySessionMixin._recovery_primitive_context(
        planner, target_jid=jid, resource_states={}, default_resource_state="", part_states={},
        part_locations={}, recorded_resource_snapshots={jid: raw}))
    assert seen == [original]
    assert resource_snapshot_declared_fields(snapshot, row["state_variables"]) == row["expected"]
    effective = RecoverySessionMixin._recovery_effective_resource_facts(
        planner, recovery_resources=entries, projected_resource_snapshots=None)
    assert resource_snapshot_declared_fields(effective[jid], row["state_variables"]) == row["expected"]
    grounded = RecoverySessionMixin._recovery_grounding_context(planner, focused_resource_jid=jid,
        recovery_snapshot=snapshot, recovery_resources=entries, part_tracker={}, goal_state="", P_id=[])
    assert resource_snapshot_declared_fields(grounded["resources"][jid], row["state_variables"]) == row["expected"]
    prompt_snapshot = RecoverySessionMixin._build_prompt_recovery_snapshot(entries[jid])
    public_expected = {key: value for key, value in row["expected"].items()
                       if not row["state_variables"][key].get("private") and not key.startswith(("_", "task_ctx."))}
    assert {key: prompt_snapshot[key] for key in public_expected} == public_expected
    prepared = {"recovery_resources": entries, "llm_input": {"observed_runtime_state": {"resources": []}}}
    public = multi_turn_context.public_resource_snapshot(prepared, {}, jid)
    assert {key: public[key] for key in public_expected} == public_expected
    assert raw == original
    snapshot[next(iter(row["expected"]))] = "mutated detached result"
    assert raw == original


@pytest.mark.parametrize("value", [None, False, 0, "", [], {}])
def test_declared_resource_snapshot_preserves_explicit_values_and_missing(value):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_resource_adapter import adapt_recovery_resource_snapshot
    from cais_spade_llm.resources.resource_profile import resource_snapshot_declared_fields

    declarations = {"current_state": {"scope": "resource"}, "held_part": {"scope": "resource"},
                    "part_state": {"scope": "part"}, "inventory.part": {"scope": "resource"}}
    raw = {"current_state": value, "part_state": "not a resource field", "inventory.part": None}
    result = adapt_recovery_resource_snapshot(resource_jid="KMR@localhost", resource_type="kmr",
        snapshot=raw, modeled_state={"resource_state": "idle"}, state_variables=declarations)
    assert resource_snapshot_declared_fields(result, declarations) == {"current_state": value, "inventory.part": None}
    assert "held_part" not in result
    assert "part_state" not in result


@pytest.mark.parametrize("other", ["idle", False, 0])
def test_declared_resource_snapshot_rejects_conflicting_exact_fields(other):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_resource_adapter import adapt_recovery_resource_snapshot

    with pytest.raises(ValueError, match="Conflicting recorded resource field: held_part"):
        adapt_recovery_resource_snapshot(resource_jid="KMR@localhost", resource_type="kmr",
            snapshot={"held_part": None, "resource_core": {"held_part": other}},
            state_variables={"held_part": {"scope": "resource"}})


def test_public_recorded_process_capabilities_are_discoverable_without_private_annotations(context):
    prepared, session, jids, _ = context
    entry = prepared["recovery_resources"][jids[2]]
    entry["static_capabilities"].update(process_capabilities={"trim": {"supported_results": ["square"],
        "reference_recovery": "forbidden"}}, program_options=[{"program_id": "recorded_square"}])
    row = next(row for row in multi_turn_context.resource_index(prepared) if row["resource_jid"] == jids[2])
    assert {"process_capabilities", "program_options"} <= set(row["available_capability_fields"])
    fact = multi_turn_context.resolve_outline_context(prepared, session, f"/resources/{jids[2]}/static_capabilities")
    assert fact["value"]["process_capabilities"] == {"trim": {"supported_results": ["square"]}}
    assert fact["value"]["program_options"] == [{"program_id": "recorded_square"}]


def test_declared_projected_resource_fields_do_not_fall_back_to_captured_values():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_session import RecoverySessionMixin

    entries = {"M1@localhost": {"resource_type": "resource", "recovery_snapshot": {"resource_state": "idle"},
        "recovery_des_model": {"state_variables": {"resource_state": {"scope": "resource"}}}}}
    facts = RecoverySessionMixin._recovery_effective_resource_facts(object(), recovery_resources=entries,
        projected_resource_snapshots={"M1@localhost": {}})
    assert "resource_state" not in facts["M1@localhost"]
    with pytest.raises(ValueError, match="not a mapping"):
        RecoverySessionMixin._recovery_effective_resource_facts(object(), recovery_resources=entries,
            projected_resource_snapshots={"M1@localhost": None})


def test_completed_failure_checkpoint_is_separate_from_blocked_task_and_excludes_metadata():
    from cais_spade_llm.agents.intelligent_product.replanner.failure_context import build_failure_event
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_session import RecoverySessionMixin
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import _slim_fault_event

    checkpoint = {"name": "after_M1_pick_before_release", "completed_task": {
        "task_id": "environment_17", "resource_id": "ur5e-1", "event_name": "pick_grasp", "status": "completed",
        "evaluator_data": "PRIVATE_CANARY"}, "reference_recovery": "PRIVATE_CANARY"}
    event = build_failure_event(failed_resource_jid="Conveyor@localhost", final_status="failed:Conveyor breakdown",
        part_name="KET4_Square_4mm", observations={"checkpoint_context": checkpoint})
    anchor = RecoverySessionMixin._failure_anchor_from_payload(event)
    assert anchor["failed_task_id"] == "" and anchor["failed_function_name"] == ""
    assert anchor["failure_type"] == "Conveyor breakdown"
    compact = _slim_fault_event({"failure_type": anchor["failure_type"], "checkpoint": anchor["checkpoint"]})
    assert compact["checkpoint"]["completed_task"] == {
        "task_id": "environment_17", "resource_id": "ur5e-1", "event_name": "pick_grasp", "status": "completed"}
    assert "PRIVATE_CANARY" not in repr(compact)


def test_checkpoint_processing_facts_preserve_incomplete_and_unknown_without_private_records():
    from cais_spade_llm.recovery_framework.part_slippage_recovery import _recorded_checkpoint
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import _slim_fault_event

    facts = {"simulation_started_sec": 35.7, "simulation_elapsed_sec": 2.5, "processing_time_sec": 5.0,
        "processing_fraction": 0.5, "process": "trim", "result": "square", "process_completed": False,
        "observation_status": "completed"}
    captured = {"failure": {"checkpoint": "during_processing_halfway", **facts,
        "reference_recovery": "PRIVATE_CANARY", "primitive_sequence": ["PRIVATE_CANARY"]}}
    checkpoint = _recorded_checkpoint(captured)
    assert checkpoint == {"name": "during_processing_halfway", **facts}
    assert _slim_fault_event({"checkpoint": checkpoint})["checkpoint"] == checkpoint
    unknown = {**checkpoint, "processing_fraction": None, "process_completed": None}
    assert _slim_fault_event({"checkpoint": unknown})["checkpoint"] == unknown
    malformed = {**checkpoint, "processing_fraction": True, "process": {"private": "PRIVATE_CANARY"}}
    compact = _slim_fault_event({"checkpoint": malformed})["checkpoint"]
    assert "processing_fraction" not in compact and "process" not in compact
    assert "PRIVATE_CANARY" not in repr(compact)

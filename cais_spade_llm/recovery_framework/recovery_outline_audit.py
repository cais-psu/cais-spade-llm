from __future__ import annotations

"""Replay frozen evaluation evidence using the existing saved-prefix audit.

The replay retains exact provider, RA, physical projection, CCA and nominal
continuation checks from the previously audited Part slippage experiment.
Selected specifications are supplied by each capture; no reference events are used.
"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from unittest.mock import patch

from cais_spade_llm.recovery_framework import ROOT


def exact(first, second):
    """Compare JSON values without equating absent fields, null, or numeric types."""
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(exact(first[key], second[key]) for key in first)
    if isinstance(first, list):
        return len(first) == len(second) and all(exact(a, b) for a, b in zip(first, second))
    return first == second


def recorded_initial_checkpoint(scene, projection):
    """Reconstruct only declared captured fields for the new owner clock."""
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
    from cais_spade_llm.resources.recovery_geometry import _checkpoint, _part_entries

    checkpoint = _checkpoint(scene['snapshot'], scene['geometry'], scene['resource_models'], index=0)
    _part_entries([checkpoint])
    clock_version = projection.get('projection_clock_version')
    if clock_version is None:
        return checkpoint
    require(type(clock_version) is int and clock_version == 1, 'Unsupported owner projection clock version')
    contract = projection.get('owner_interval_contract', {})
    require(type(scene.get('owner_projection_clock_version')) is int and scene['owner_projection_clock_version'] == 1
            and type(contract.get('projection_clock_version')) is int and contract['projection_clock_version'] == 1
            and contract.get('source_contract_fingerprint') == projection.get('provenance', {}).get('contract_fingerprint')
            and recovery_validation_fingerprint(contract.get('source_contract')) == contract.get('source_contract_fingerprint')
            and isinstance(contract.get('preview_fingerprint'), str) and bool(contract['preview_fingerprint']),
            'Owner clock has no matching recorded projection contract')
    protected = {'current_pose', 'joint_names', 'joint_positions', 'base_pose', 'grasp_transform',
                 'held_part', 'gripper_state', 'contained_parts', 'frame'}
    models = scene.get('environment_models', {})
    for rid, values in scene.get('environment_valuation', {}).items():
        require(rid in checkpoint['resources'] and rid in models,
                'Recorded owner valuation has no matching resource declaration')
        fields = {field.replace('{part_name}', part) if '{part_name}' in field else field
            for field in models[rid].get('state_variables', {})
            for part in (scene['geometry']['parts'] if '{part_name}' in field else [None])}
        for field, value in values.items():
            require(field in fields, 'Recorded owner valuation contains an undeclared field: ' + rid + '/' + field)
            row = checkpoint['resources'][rid]
            if field in row:
                require(exact(row[field], value), 'Recorded owner field conflicts with physical checkpoint: ' + rid + '/' + field)
            else:
                require(field not in protected, 'Recorded owner valuation cannot supply missing physical evidence: ' + rid + '/' + field)
                row[field] = deepcopy(value)
    return checkpoint


def authored_match(authored, accepted):
    for field in ('outline_id', 'event_name', 'resource_jid', 'part_name', 'expected_end_state', 'rationale'):
        if (field in authored) != (field in accepted):
            return False
        if field in authored and not exact(authored[field], accepted[field]):
            return False
    for field in ('params', 'expected_start_state'):
        if field in authored and (field not in accepted or not exact(authored[field], accepted[field])):
            return False
    return True


def require(value, reason):
    if not value:
        raise ValueError(reason)


def load(path, *, fields=None):
    path = Path(path)

    def fingerprint():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    before = fingerprint()
    with path.open() as stream:
        if fields is None:
            value = json.load(stream)
        else:
            # Session artifacts are written with indent=2. Skip the archived
            # turn bodies while retaining every field consumed by this audit.
            require(next(stream).rstrip() == '{', 'Unsupported session artifact format')
            value = {}
            pattern = re.compile(r'^  "([^"\\]+)": (.*)$')
            last = None
            for line in stream:
                last = line
                match = pattern.match(line)
                if match is None or match[1] not in fields:
                    continue
                key, first = match.groups()
                require(key not in value, 'Repeated session field: ' + key)
                first = first.rstrip('\n')
                pieces = [first]
                if first in ('{', '['):
                    closing = '}' if first == '{' else ']'
                    for following in stream:
                        pieces.append(following)
                        if following.startswith('  ' + closing):
                            break
                    else:
                        raise ValueError('Incomplete session field: ' + key)
                value[key] = json.loads(''.join(pieces).rstrip().removesuffix(','))
            require(last is not None and last.rstrip() == '}', 'Incomplete session artifact')
    require(fingerprint() == before, 'Artifact bytes changed while loading: ' + str(path))
    return value, {'path': str(path.resolve()), 'sha256': before}



def require_nominal_node_task(node, task, current, prepared):
    """Bind each nominal witness to its exact recorded task and current state."""
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import _nominal_task

    require(task.get('resource_jid') == node.get('resource_jid')
            and bool(node.get('resource_jid')), 'Nominal task owner differs from its recorded node')
    require(task.get('required_task_contract') == node.get('function_name')
            and bool(node.get('function_name')), 'Nominal task contract differs from its recorded node')
    expected, reason = _nominal_task(node, current, prepared)
    require(expected is not None and not reason, reason or 'Nominal task cannot be derived from its recorded node')
    require(exact(task, expected), 'Nominal task fields differ from the exact recorded-node derivation')


def require_manifest(provenance, recorded_artifacts):
    """Require an indexed immutable hash for every selected proof artifact."""
    require(provenance['path'] in recorded_artifacts,
            'Proof artifact is absent from the frozen lineage manifest: ' + provenance['path'])
    require(provenance['sha256'] == recorded_artifacts[provenance['path']],
            'Proof artifact bytes changed: ' + provenance['path'])



def read_bound_nominal(record, *, current, prepared, directory, recorded_artifacts):
    """Load a complete proof only from the exact originating stage and manifest."""
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (
        nominal_continuation_context_binding, read_nominal_continuation,
    )
    proof = read_nominal_continuation(record, directory=directory,
        context_binding=nominal_continuation_context_binding(current, prepared))
    provenance = None
    if "artifact_reference" in record:
        reference = record["artifact_reference"]
        provenance = {"path": str((directory / reference["path"]).resolve()), "sha256": reference["sha256"]}
        require_manifest(provenance, recorded_artifacts)
    return proof, provenance


def completion_within_deadline(report, process):
    """Separate a terminal generation timestamp from cleanup/reporting duration."""
    elapsed = report.get('terminal_generation_elapsed_seconds')
    return bool(type(elapsed) in (int, float) and math.isfinite(elapsed)
        and elapsed >= 0
        and report.get('generation_deadline_seconds') is None
        and report.get('terminal_generation_status') == 'outline_review'
        and report.get('timed_out') is False and report.get('error') is None
        and process.get('returncode') == 0 and type(process.get('returncode')) is int
        and process.get('hard_timeout') is False
        and process.get('cancellation_completed') is True
        and process.get('generation_deadline_seconds') is None
        and process.get('source') == report.get('source')
        and process.get('source_sha256') == report.get('source_sha256'))


def completion_attested(result, report):
    """Require complete validated chains and unchanged source/code/native evidence."""
    return bool(result.get('accepted_events') and result.get('accepted_chain_valid') is True
        and result.get('nominal_chain_valid') is True and result.get('nominal_complete') is True
        and report['result'].get('status') == 'outline_review'
        and all(result.get(key) is True for key in ('source_unchanged',
            'parent_source_unchanged', 'recorded_run_integrity',
            'current_code_matches_recorded', 'current_native_matches_recorded',
            'manifest_membership_valid', 'completed_within_deadline', 'fresh_generation_valid'))
        and not result.get('errors') and report.get('error') is None)


def require_fresh_generation(report, process, inputs, attempt, provider_pairs, recorded_artifacts):
    """Bind fresh completion to this process's actual provider calls and new session.

    Failure here does not invalidate historical prefix evidence. It prevents
    that evidence from being counted as a new generated success.
    """
    require(report.get('fresh_attempt') is True and report.get('generation_enabled') is True,
            'This report does not attest fresh generation')
    require(report.get('generation_path') == 'production Offline Generate'
            and type(report.get('external_validation_retries')) is int
            and report['external_validation_retries'] == 0,
            'Fresh generation did not use production Offline Generate alone')
    companion = Path(report['source']).resolve().parent / 'native_preview_model/manifest.json'
    companion_key = str(companion.relative_to(ROOT) if companion.is_relative_to(ROOT) else companion)
    companion_sha = report.get('native_preview_model_manifest_sha256')
    require(isinstance(companion_sha, str) and len(companion_sha) == 64
            and report.get('native_code_sha256', {}).get(companion_key) == companion_sha,
            'Fresh generation lacks its frozen source-bound native model companion')
    require(type(report.get('pid')) is int and report['pid'] == process.get('pid'),
            'Generation process identity differs')
    output = str(Path(attempt).resolve())
    require(report.get('attempt_directory') == output == process.get('attempt_directory'),
            'Generation output differs from the reaped process')
    require(report.get('source') == process.get('source')
            and report.get('source_sha256') == process.get('source_sha256'),
            'Generation source differs from the reaped process')
    session = report.get('fresh_session') or {}
    run_id = session.get('run_id')
    require(isinstance(run_id, str) and bool(run_id) and run_id == inputs.get('run_id')
            == report.get('result', {}).get('run_id'), 'Fresh session run identity differs')
    directory = Path(report['result']['artifact_directory']).resolve()
    require(directory.name == 'generation_0001' and directory.parent.name == run_id
            and session.get('artifact_directory') == str(directory),
            'Fresh generation artifact root differs')
    prepared = inputs['prepared_recovery_request']
    seed = prepared['multi_turn_session_seed']
    require(seed.get('accepted_outline_prefix') == [] and seed.get('turn_index') == 0,
            'Fresh generation started from an existing outline or turn')
    session_id = prepared.get('recovery_session', {}).get('session_id')
    require(session_id == session.get('session_id') == run_id + '_0001',
            'Fresh generation session identifier differs')
    source_fingerprint = report.get('source_observation_fingerprint')
    require(isinstance(source_fingerprint, str) and bool(source_fingerprint)
            and source_fingerprint == inputs.get('source_observation_fingerprint')
            == prepared.get('source_observation_fingerprint'),
            'Fresh generation saved observation binding differs')
    root = directory / 'recovery_outline'
    roots = report.get('lineage', {}).get('evidence_directories')
    require(isinstance(roots, list) and roots and {Path(value).resolve() for value in roots} == {root},
            'Fresh generation includes a different evidence root')
    require(bool(recorded_artifacts) and all(Path(value).resolve().is_relative_to(root)
            for value in recorded_artifacts), 'Fresh manifest includes evidence from another generation')
    receipts = list(report.get('live_provider_calls') or [])
    require(provider_pairs and len(receipts) == len(provider_pairs),
            'Fresh generation lacks exact live provider receipts')

    def value_hash(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    for request, request_source, response, response_source in provider_pairs:
        require_manifest(request_source, recorded_artifacts)
        require_manifest(response_source, recorded_artifacts)
        require(request.get('kind') == 'provider_request' and request.get('capture_status') == 'captured'
                and response.get('kind') == 'provider_response', 'Provider capture kinds differ')
        require(request.get('run_id') == response.get('run_id') == session_id,
                'Provider capture belongs to another session')
        for field in ('stage', 'turn', 'call_id', 'tool_round', 'attempt'):
            require(field in request and field in response and exact(request[field], response[field]),
                    'Provider request/response identity differs: ' + field)
        request_path, response_path = Path(request_source['path']), Path(response_source['path'])
        require(request_path == response_path.with_name(response_path.name.removesuffix('_response.json') + '_request.json')
                and response_path.parent.parent == root / 'requests'
                and response_path.parent.name == response['call_id'], 'Provider capture paths differ')
        payload = request.get('payload') or {}
        require(payload.get('model') == report.get('model'), 'Provider model differs from this attempt')
        require(('reasoning_effort' not in payload) if report.get('model') == 'gpt-4o'
                else payload.get('reasoning_effort') == 'low', 'Provider reasoning effort differs')
        matching = [index for index, row in enumerate(receipts)
            if row.get('request_fingerprint') == value_hash(payload)
            and row.get('response_fingerprint') == value_hash(response.get('raw_response'))
            and row.get('request_id') == response.get('request_id')
            and row.get('run_id') == session_id and row.get('stage') == response['stage']
            and exact(row.get('turn'), response['turn'])]
        require(len(matching) == 1, 'Provider response has no unique current-process live receipt')
        receipt = receipts.pop(matching[0])
        require(receipt.get('request_sent') is True and receipt.get('response_source') == 'live',
                'Provider receipt records a replay or unsent request')
        require(receipt.get('pid') == report['pid'] and receipt.get('attempt_directory') == output,
                'Provider receipt belongs to another process')
        begin, end = receipt.get('started_elapsed_seconds'), receipt.get('returned_elapsed_seconds')
        stopped = report.get('generation_stopped_elapsed_seconds')
        require(all(type(value) in (int, float) and math.isfinite(value) for value in (begin, end, stopped))
                and 0 <= begin <= end <= stopped, 'Provider receipt time is outside this generation')
    require(not receipts, 'Unmatched live provider receipts remain')


def audit(attempt):
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint as fp
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_validated_event import _project
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_nominal_evidence import (
        _completion_goals, _exact_nominal_witness, _fulfilled_nominal_tasks, _monitor_completion,
    )
    from cais_spade_llm.resources.recovery_geometry import _checkpoint, _part_entries, outline_event_binding
    from cais_spade_llm.recovery_framework.outline_validation import validate_safety_outline_payload

    report, report_source = load(attempt / 'report.json')
    process, process_source = load(attempt.parent / 'process_result.json')
    final, session_source = load(attempt / 'session.json', fields={
        'accepted_outline_prefix', 'nominal_continuation_evidence', 'symbolic_resources',
        'symbolic_parts', 'derived_resource_snapshots', 'outline_monitor_context', 'projected_physical_evidence',
    })
    directory = Path(report['result']['artifact_directory'])
    inputs, input_source = load(directory / 'inputs.json')
    prepared = inputs['prepared_recovery_request']
    current = deepcopy(prepared['multi_turn_session_seed'])
    current['projected_physical_evidence'] = deepcopy(prepared['physical_evidence'])
    records = {'report': report_source, 'session': session_source, 'inputs': input_source, 'process': process_source}
    result = {'accepted_chain_valid': False, 'nominal_chain_valid': None, 'nominal_complete': False,
              'outline_complete': False, 'accepted_events': [], 'nominal_tasks': [], 'errors': [],
              'source_records': records, 'execution_authorized': False, 'physical_execution_verified': False,
              'scope': 'Saved RA evidence, derived state chain, and local CCA replay; no new RA derivation or native preview'}
    result['completed_within_deadline'] = completion_within_deadline(report, process)
    result['manifest_membership_valid'] = False
    result['fresh_generation_valid'] = False
    result['fresh_generation_errors'] = []
    source = Path(report['source'])
    result['source_unchanged'] = hashlib.sha256(source.read_bytes()).hexdigest() == report['source_sha256']
    result['recorded_run_integrity'] = all(report.get(key) is True for key in (
        'source_unchanged', 'parent_source_unchanged', 'code_unchanged', 'native_code_unchanged')) and report.get('forbidden_calls') == 0
    result['recorded_run_error'] = report.get('error')
    result['parent_source_unchanged'] = (Path(report['parent_source']).is_file()
        and hashlib.sha256(Path(report['parent_source']).read_bytes()).hexdigest() == report['parent_source_sha256'])
    result['current_code_matches_recorded'] = bool(report.get('code_sha256')) and all((ROOT / name).is_file()
        and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
        for name, digest in report.get('code_sha256', {}).items())
    result['current_native_matches_recorded'] = bool(report.get('native_code_sha256')) and all((ROOT / name).is_file()
        and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
        for name, digest in report.get('native_code_sha256', {}).items())
    documents = prepared['outline_safety_specifications']
    expected_rules = {row['source_sha256'] for row in documents}
    require(documents and len(expected_rules) == len(documents), 'Selected safety specifications must be present and distinct')
    for row in documents:
        require(hashlib.sha256(Path(row['source_path']).read_bytes()).hexdigest() == row['source_sha256'],
                'Selected specification bytes changed')
    dirs = {directory / 'recovery_outline'}
    dirs.update(Path(value) for value in report.get('lineage', {}).get('evidence_directories', []))
    recorded_artifacts = {row['path']: row['sha256'] for row in [
        *report.get('lineage', {}).get('provider_request_records', []),
        *report.get('lineage', {}).get('provider_records', []),
        *report.get('lineage', {}).get('validation_artifacts', [])]}
    proposals, turns, provider_pairs = [], [], []
    request_paths_by_turn = {}
    for artifact_dir in sorted(dirs):
        for path in sorted(artifact_dir.glob('requests/*/*_response.json')):
            record, provenance = load(path)
            if record.get('kind') != 'provider_response':
                continue
            require_manifest(provenance, recorded_artifacts)
            request_path = path.with_name(path.name.removesuffix('_response.json') + '_request.json')
            try:
                request, request_source = load(request_path)
                provider_pairs.append((request, request_source, record, provenance))
            except (OSError, ValueError, TypeError) as exc:
                result['fresh_generation_errors'].append(str(exc))
            for choice in record['raw_response'].get('choices', []):
                content = choice.get('message', {}).get('content')
                if not isinstance(content, str):
                    continue
                try:
                    decoded = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if isinstance(decoded, dict) and isinstance(decoded.get('next_transition'), dict):
                    proposals.append({'event': decoded['next_transition'], 'turn': record.get('turn'), 'source': provenance})
        for path in sorted(artifact_dir.glob('turns/turn_*.json')):
            turn, provenance = load(path, fields={'transition_validation', 'next_transition', 'source_turn_index', 'turn_index',
                                                  'phase', 'request_record_paths'})
            if turn.get('request_record_paths'):
                request_paths_by_turn[(turn.get('turn_index'), turn.get('phase'))] = (
                    turn['request_record_paths'], provenance)
            if turn.get('transition_validation', {}).get('status') == 'passed':
                require_manifest(provenance, recorded_artifacts)
                turns.append({'path': path, 'source': provenance,
                    'next_transition': deepcopy(turn.get('next_transition') or {}),
                    'source_turn': turn.get('source_turn_index', turn.get('turn_index'))})
            del turn

    try:
        require(not result['fresh_generation_errors'], 'Fresh provider capture pairs are incomplete')
        require_fresh_generation(report, process, inputs, attempt, provider_pairs, recorded_artifacts)
        result['fresh_generation_valid'] = True
    except (OSError, KeyError, ValueError, TypeError) as exc:
        result['fresh_generation_errors'].append(str(exc))
    del provider_pairs

    def validate(task, stages, state):
        candidate = fp(task)
        previous = None
        errors = []
        for stage in stages:
            if stage.get('validator_role') == 'ResourceAgent':
                previous = stage
                continue
            if stage.get('validator_role') != 'CentralControllerAgent' or previous is None:
                continue
            ra, cca = previous.get('evidence') or {}, stage.get('evidence') or {}
            if not ra.get('allowed') or not cca.get('is_safe'):
                continue
            try:
                require(ra.get('allowed') is True and ra.get('execution_authorized') is False
                        and ra.get('feasibility_status') == 'FEASIBLE' and ra.get('validation_contract_version') == 2,
                        'RA does not establish version-2 feasibility')
                require(ra.get('primitive_support', {}).get('semantics') == 'successor_primitive_union',
                        'RA primitive support semantics differ')
                require(previous.get('validator_jid') == task['resource_jid'], 'RA owner differs')
                require(previous.get('state_fingerprint') and previous['state_fingerprint'] == stage.get('state_fingerprint'),
                        'RA and CCA checked different states')
                require(ra.get('candidate_fingerprint') == candidate == cca.get('candidate_fingerprint'),
                        'RA or CCA candidate fingerprint differs')
                projection = ra.get('geometry_evidence') or {}
                require(projection.get('status') == 'projected' and projection.get('interval_complete') is True
                        and not projection.get('unresolved') and projection.get('execution_authorized') is False,
                        'Complete physical projection is missing')
                scene = state['projected_physical_evidence']
                resource = prepared['recovery_resources'][task['resource_jid']]
                snapshot = state.get('derived_resource_snapshots', {}).get(task['resource_jid'], resource['recovery_snapshot'])
                expected = outline_event_binding(task, snapshot, resource['recovery_des_model'], scene)
                require(exact(projection.get('provenance', {}).get('binding'), expected),
                        'Physical projection is not bound to the replayed prefix state')
                checkpoint = recorded_initial_checkpoint(scene, projection)
                initial = {key: value for key, value in projection['observations'][0].items()
                           if key not in {'state_snapshot', 'owner_safety_evidence'}}
                require(exact(checkpoint, initial),
                        'Physical projection does not start at the replayed complete scene')
                before = state.get('outline_monitor_context', {})
                require(exact(cca.get('monitor_context_before'), before), 'CCA history was reset or replaced')
                replay = validate_safety_outline_payload({'candidates': [{
                    'candidate_index': cca.get('candidate_index', 0), 'event_id': task['outline_id'],
                    'safety_input': {'candidate_fingerprint': candidate, 'projection': projection,
                        'specifications': documents, 'monitor_history': prepared.get('monitor_history', {}),
                        'monitor_context': before}}]})['results'][0]
                require(replay.get('is_safe') is True and replay.get('validation_status') == 'passed',
                        'Local CCA replay failed: ' + replay.get('reason', ''))
                require(exact(replay, cca), 'Saved CCA result differs from complete local monitor replay')
                require(set(replay['monitor_context_after']) == expected_rules, 'CCA omitted a selected specification')
                after = deepcopy(state)
                _project(task, ra, replay, after)
                return ra, replay, after
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                errors.append(str(exc))
        raise ValueError('; '.join(errors) or 'No matching successful RA/CCA pair')

    try:
        require(result['source_unchanged'], 'Source observations changed')
        accepted = final.get('accepted_outline_prefix', [])
        require(len({row['outline_id'] for row in accepted}) == len(accepted), 'Accepted outline IDs repeat')
        for accepted_event in accepted:
            task = {key: deepcopy(value) for key, value in accepted_event.items() if key != 'candidate_source'}
            authored = [row for row in proposals if authored_match(row['event'], task)]
            require(authored, 'Accepted event has no exact provider proposal (including missing versus null)')
            candidates = [entry for entry in turns if authored_match(entry['next_transition'], task)]
            failures, selected = [], None
            for entry in candidates:
                captures = [row for row in authored if row['turn'] == entry['source_turn']]
                if not captures:
                    continue
                turn, provenance = load(entry['path'])
                require(provenance == entry['source'], 'Validation artifact changed after indexing')
                try:
                    derived, findings = multi_turn._derive_candidate_outline_task(candidate_task=captures[0]['event'],
                        session_state=current, prepared_recovery_request=prepared)
                    require(not findings and exact(derived, task),
                            'Accepted task differs from the proposal derived against the replayed state: ' + str(findings))
                    ra, cca, after = validate(task, turn.get('validation_stages', []), current)
                    selected = (captures, provenance, ra, cca, after)
                    break
                except (KeyError, ValueError, TypeError, IndexError) as exc:
                    failures.append(str(exc))
                finally:
                    del turn
            require(selected is not None, 'Accepted event failed prefix replay: ' + '; '.join(failures))
            captures, provenance, ra, cca, current = selected
            if result['fresh_generation_valid']:
                try:
                    paths, turn_source = request_paths_by_turn[(captures[0]['turn'], 'outline')]
                    require_manifest(turn_source, recorded_artifacts)
                    require(any(row['source']['path'] in paths and str(Path(row['source']['path']).with_name(
                        Path(row['source']['path']).name.removesuffix('_response.json') + '_request.json')) in paths
                        for row in captures), 'Accepted proposal lacks its current turn request/response capture')
                except (KeyError, ValueError, TypeError) as exc:
                    result['fresh_generation_valid'] = False
                    result['fresh_generation_errors'].append(str(exc))
            current.setdefault('accepted_outline_prefix', []).append(deepcopy(accepted_event))
            result['accepted_events'].append({'outline_id': task['outline_id'], 'chain_valid': True,
                'provider_records': [row['source'] for row in captures], 'validation_record': provenance,
                'RA_candidate_fingerprint': ra['candidate_fingerprint'],
                'physical_binding': ra['geometry_evidence']['provenance']['binding'],
                'monitor_context_after_fingerprint': fp(cca['monitor_context_after'])})
            del selected, ra, cca, after
        require(exact(current.get('accepted_outline_prefix', []), accepted),
                'Accepted event prefix differs from verified provider/RA/CCA replay')
        for field in ('symbolic_resources', 'symbolic_parts', 'derived_resource_snapshots', 'outline_monitor_context'):
            require(exact(current.get(field, {}), final.get(field, {})), 'Final accepted state differs at ' + field)
        probe = {}
        replay_scene = outline_event_binding(probe, {}, {}, current['projected_physical_evidence'])['scene_fingerprint']
        final_scene = outline_event_binding(probe, {}, {}, final.get('projected_physical_evidence', prepared['physical_evidence']))['scene_fingerprint']
        require(replay_scene == final_scene, 'Final physical state differs from accepted-prefix replay')
        result['accepted_chain_valid'] = True
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        result['errors'].append(str(exc))
        return result

    try:
        nominal, nominal_source = read_bound_nominal(final.get('nominal_continuation_evidence') or {},
            current=current, prepared=prepared, directory=directory / 'recovery_outline', recorded_artifacts=recorded_artifacts)
        if nominal_source is not None:
            result['source_records']['nominal_proof'] = nominal_source
        result['nominal_recorded_status'] = nominal.get('status')
        require(prepared.get('nominal_continuation_semantics') == 'recorded_unfinished_tasks'
                and 'saved_pending_nodes' in prepared, 'Complete recorded unfinished-task contract is missing')
        nodes = {row['id']: row for row in prepared['saved_pending_nodes']
                 if row.get('id') and row.get('status') not in {'done', 'completed'}}
        for resource in prepared['recovery_resources'].values():
            for row in resource.get('pending_tasks', []):
                if row.get('id') and row.get('status') not in {'done', 'completed'}:
                    nodes.setdefault(row['id'], row)
        replaced = _fulfilled_nominal_tasks(nodes, current, prepared)
        if nominal.get('complete') is True:
            require(set(nominal.get('replaced_task_ids', [])) == set(replaced)
                    and exact(nominal.get('replacement_evidence'), replaced), 'Nominal replacement evidence differs')
        completed = set(replaced)
        for row in nominal.get('tasks', []):
            identifier, task = row.get('task_id'), row.get('task') or {}
            require(identifier in nodes and identifier not in completed, 'Nominal witness repeats or invents a task')
            node = nodes[identifier]
            require_nominal_node_task(node, task, current, prepared)
            require(set(node.get('predecessors', [])).issubset(completed | set(prepared.get('completed_task_ids', []))),
                    'Nominal predecessor is unfinished')
            require(not any(other not in completed and other != identifier
                and prior.get('resource_jid') == node.get('resource_jid')
                and prior.get('sequence_index', 0) < node.get('sequence_index', 0) for other, prior in nodes.items()),
                'Nominal resource order differs')
            ra, cca, after = validate(task, row.get('validation_stages', []), current)
            reason = _exact_nominal_witness(node, ra, prepared)
            require(not reason, reason or '')
            current = after
            completed.add(identifier)
            result['nominal_tasks'].append({'task_id': identifier, 'chain_valid': True,
                'monitor_context_after_fingerprint': fp(cca['monitor_context_after'])})
        result['nominal_chain_valid'] = True
        result['unfinished_task_ids'] = sorted(set(nodes) - completed)
        if nominal.get('complete') is True:
            require(nominal.get('status') == 'passed' and nominal.get('execution_authorized') is False,
                    'Nominal completion has no non-executing passed verdict')
            require(set(nodes) == completed, 'Nominal continuation leaves retained tasks unfinished')
            incomplete = (_completion_goals(current, prepared, multi_turn._active_continuation_conditions(prepared))
                          or _monitor_completion(current, prepared))
            require(not incomplete, str(incomplete))
            result['nominal_complete'] = True
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        result['nominal_chain_valid'] = False
        result['errors'].append('Nominal continuation: ' + str(exc))
    result['manifest_membership_valid'] = True
    result['outline_complete'] = completion_attested(result, report)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    forbidden_calls, local_compilations = [], []
    import shutil
    import subprocess
    mona = shutil.which('mona')
    original_popen = subprocess.Popen

    def local_compiler(arguments, *positional, **keywords):
        command = list(arguments) if isinstance(arguments, (list, tuple)) else []
        allowed = (mona is not None and len(command) == 5 and command[:4] == [mona, '-q', '-u', '-w']
                   and not keywords.get('shell') and Path(command[-1]).name == 'formula.mona'
                   and Path(command[-1]).parent.name.startswith('reviewed-primitive-safety-')
                   and Path(command[-1]).resolve().is_relative_to(Path('/tmp')))
        if not allowed:
            return forbidden(arguments, *positional, **keywords)
        local_compilations.append({'executable': mona,
            'executable_sha256': hashlib.sha256(Path(mona).read_bytes()).hexdigest(),
            'formula_sha256': hashlib.sha256(Path(command[-1]).read_bytes()).hexdigest()})
        return original_popen(arguments, *positional, **keywords)

    def forbidden(*arguments, **keywords):
        forbidden_calls.append(True)
        raise AssertionError('Post-run audit forbids network and subprocess calls other than exact local MONA compilation')

    with ExitStack() as stack:
        for target in ('socket.socket.connect', 'socket.create_connection', 'asyncio.create_subprocess_exec'):
            stack.enter_context(patch(target, side_effect=forbidden))
        stack.enter_context(patch('subprocess.Popen', side_effect=local_compiler))
        try:
            result = audit(args.attempt.resolve())
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            result = {'accepted_chain_valid': False, 'outline_complete': False, 'errors': [str(exc)]}
    result.update(forbidden_calls=len(forbidden_calls), local_MONA_compilations=local_compilations, audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == '__main__':
    main()

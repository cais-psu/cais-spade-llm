from __future__ import annotations

"""Proposal-scoped KMR bindings over detached owner declarations and evidence."""

from copy import deepcopy

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint as fingerprint
from cais_spade_llm.recovery_framework.kmr_recovery import coordinate_parameters, recorded_location, pick_targets, place_targets


def validate_calculation_records(physical: dict, *, session_id: str, inputs: dict) -> None:
    """Reject modified, transplanted, or differently configured attempt evidence."""
    for record in physical.get('additional_binding_records', []):
        body = {key: value for key, value in record.items() if key != 'fingerprint'}
        if (record.get('fingerprint') != fingerprint(body) or record.get('session_id') != session_id
                or record.get('inputs_fingerprint') != fingerprint(inputs)):
            raise ValueError('NEEDS_CONTEXT: additional binding evidence dependencies changed')
    records = physical.get('additional_binding_records') or []
    if records and records[-1]['environment_models_fingerprint'] != fingerprint(physical.get('environment_models')):
        raise ValueError('NEEDS_CONTEXT: additional binding calculation model changed')


def _handoff(models: dict, name: str, location: str, part: str, *, scene: dict,
             physical: dict) -> tuple[list[dict], list[dict]]:
    """Bind KMR's implemented program to recorded peer custody conditions.

    Peer-specific parameters, collection effects, and process changes cannot be
    inferred for KMR. All copied peer guards remain obligations of projection.
    """
    field = 'origin_resource_location' if name.startswith('pick_') else 'destination_location'
    peers, sources = [], []
    for rid, model in sorted(models.items()):
        if rid == 'KMR':
            continue
        for event in model.get('events', []):
            bindings = event.get('parameter_bindings', {})
            actor = bindings.get('resource_id', {}).get('equals')
            operation = 'pick_grasp' if name.startswith('pick_') else 'place_release'
            if (event.get('event_name') not in {operation, 'pick_part' if name.startswith('pick_') else operation}
                    or bindings.get(field) != {'equals': location} or actor == 'KMR'
                    or set(event.get('participants', [])) != {rid, actor}
                    or event.get('collection_effects') or event.get('product_effects', {}).get('processCompleted')
                    or set(bindings) - {'resource_id', field, 'part_name', 'handoff_acknowledged', 'source_clear', 'robot_clear'}):
                continue
            peers.append((rid, event))
            valuation = physical.get('environment_valuation', {}).get(rid, {})
            if (name.startswith('pick_') and event.get('guards', {}).get('resource_state') == {'equals': 'completed'}
                    and valuation.get('job_state') == 'interrupted'):
                from cais_spade_llm.resources.machine.machine_jobs import interrupted_pickup_conditions

                machines = [row for row in scene.get('machines', []) if row.get('resource_id') == rid]
                if len(machines) != 1:
                    raise KeyError('Exact recorded machine declaration for ' + rid)
                conditions = interrupted_pickup_conditions(machines[0],
                    physical.get('processing_observations', {}).get(rid, {}), valuation, part,
                    run_id=physical.get('processing_observation_source', {}).get('run_id', ''))
                peer = deepcopy(event)
                peer['guards'].update(deepcopy(conditions['guards']))
                peers.append((rid, peer))
                sources.append({'machine_pickup_conditions': conditions,
                                'source_declaration_fingerprint': fingerprint(event)})
    templates = sorted((event for event in models['KMR']['events'] if event.get('event_name') == name
                        and event.get('program_status') == 'implemented'), key=fingerprint)
    if name in {'place_approach', 'place_release'} and location == 'Storage' and templates:
        from cais_spade_llm.resources.environment_recovery import storage_placement_conditions

        conditions = storage_placement_conditions(models['Storage'], scene,
            physical.get('environment_valuation', {}).get('Storage', {}), part)
        peer = deepcopy(templates[0])
        peer.update(participants=['KMR', 'Storage'], guards=deepcopy(conditions['guards']),
                    updates=deepcopy(conditions['updates']))
        peer['parameter_bindings'].update(resource_id={'equals': 'KMR'},
                                         destination_location={'equals': location}, part_name={'equals': part})
        peers.append(('Storage', peer))
        sources.append({'storage_placement_conditions': conditions})
    if not templates or not peers:
        raise KeyError('Recorded peer custody and access conditions for ' + location)
    # The program and local KMR state effects are invariant across configured
    # bindings. Differing programs require an explicit owner implementation.
    if len({fingerprint({'program': event.get('program'), 'updates': event.get('updates'),
                         'guards': {key: value for key, value in event.get('guards', {}).items() if key != 'resource_location'}})
            for event in templates}) != 1:
        raise KeyError('Unambiguous implemented KMR program for ' + name)
    result, seen = [], set()
    for rid, peer in peers:
        template = deepcopy(templates[0])
        bindings = deepcopy(peer['parameter_bindings'])
        bindings.update(resource_id={'equals': 'KMR'}, part_name={'equals': part})
        template.update(parameter_bindings=bindings, participants=['KMR', rid])
        template['guards']['resource_location'] = {'equals': location}
        # Capability metadata describes this new binding only; source records
        # and their robot identities are retained unchanged in the audit.
        template['capability_transition'] = {'source': {}, 'target': {}}
        peer_row = deepcopy(peer)
        peer_row.update(event_name=name, parameter_bindings=deepcopy(bindings), participants=['KMR', rid],
                        capability_transition=deepcopy(template['capability_transition']))
        if name in {'pick_approach', 'place_approach'}:
            peer_row['updates'] = {}
        key = fingerprint([template, peer_row])
        if key not in seen:
            seen.add(key)
            result.append({'KMR': template, rid: peer_row})
    return result, sources


def _check_peer_conditions(rows: list[dict], valuation: dict, part: str, location: str) -> None:
    missing, supported = False, False
    for bindings in rows:
        valid, unknown = True, False
        for rid, event in bindings.items():
            if rid == 'KMR':
                continue
            for field, condition in event.get('guards', {}).items():
                field = field.replace('{part_name}', part)
                if field not in valuation.get(rid, {}) or len(condition) != 1:
                    unknown = True
                    continue
                operator, expected = next(iter(condition.items()))
                if operator.endswith('_from_param'):
                    if expected != 'part_name':
                        unknown = True
                        continue
                    expected = part
                if operator not in {'equals', 'not_equals', 'equals_from_param', 'not_equals_from_param'}:
                    unknown = True
                    continue
                actual = valuation[rid][field]
                equal = type(actual) is type(expected) and actual == expected
                valid &= not equal if operator.startswith('not_') else equal
        supported |= valid and not unknown
        missing |= valid and unknown
    if not supported:
        if missing:
            raise KeyError('Recorded peer access and custody at ' + location)
        raise ValueError('Recorded peer conditions reject requested custody at ' + location)


def prepare_calculation(*, task: dict, model: dict, snapshot: dict, physical_input: dict,
                        session_id: str) -> tuple[dict, dict, dict | None]:
    """Calculate additional bindings without changing captured or accepted state.

    Returns:
        The proposal's model, matching private evidence, and its bound audit.
        Missing geometry or custody raises KeyError; no effects are committed.
    """
    from cais_spade_llm.resources.environment_recovery import environment_recovery_descriptor, kmr_validation_evidence
    from cais_spade_llm.resources.recovery_feasibility import derive_primitive_union

    if model.get('resource_id') != 'KMR':
        return model, physical_input, None
    original = physical_input.get('resource_evidence') or {}
    physical = physical_input.get('physical_evidence') or {}
    if not original.get('environment_models'):
        return model, physical_input, None
    validate_calculation_records(physical, session_id=session_id, inputs=original.get('inputs', {}))
    records = physical.get('additional_binding_records') or []
    evidence = kmr_validation_evidence(original, snapshot, physical)
    if records:
        if any(record['source_model_fingerprint'] != fingerprint(model) for record in records):
            raise ValueError('NEEDS_CONTEXT: additional binding source model changed')
        evidence['environment_models'] = deepcopy(physical['environment_models'])
    retained_model = environment_recovery_descriptor(evidence, snapshot, resource_jid=task['resource_jid']) if records else model
    base_support = derive_primitive_union(task=task, recovery_des_model=retained_model, recovery_snapshot=snapshot)
    end, start = task.get('expected_end_state', {}), snapshot
    location, operations = end.get('resource_location', start.get('resource_location')), []
    part = task.get('part_name')
    if part and start.get('held_part') != part and (end.get('held_part') == part
            or end.get('approached_part') == part or end.get('resource_state') == 'at_pick'):
        operations = ['pick_approach', 'pick_part']
    elif part and start.get('held_part') == part and ('held_part' in end and end['held_part'] is None
                                                   or end.get('resource_state') == 'positioned'):
        operations = ['place_approach', 'place_release']
    matching = base_support['matching_transitions']
    if operations:
        field = 'origin_resource_location' if operations[0] == 'pick_approach' else 'destination_location'
        # Successor-union support deliberately omits predecessor guards. A
        # declaration for a different location cannot supply this new binding.
        matching = [row for row in matching if row['event_name'] in operations
                    and row['parameter_bindings'].get(field) == location]
    reuse = bool(records and matching)
    if matching and not records:
        return model, physical_input, None
    if reuse:
        operations = []
    elif not operations and ('resource_location' not in end or end['resource_location'] in model.get('state_variables', {}).get('resource_location', {}).get('domain', [])):
        return model, physical_input, None
    if not isinstance(location, str):
        return model, physical_input, None
    if not session_id:
        raise KeyError('Current recovery_session_id for additional bindings')
    models = deepcopy(evidence['environment_models'])
    scene = evidence.get('inputs', {}).get('scene', {})
    sources, calculations = [], {}
    declaration, reference = recorded_location(scene, location)
    if reuse:
        sources = [{'additional_binding_fingerprint': records[-1]['fingerprint']}]
    elif not operations:
        available = [event for event in models['KMR']['events'] if event.get('event_name') == 'move_to_location'
                     and event.get('program_status') == 'implemented']
        if not available:
            return model, physical_input, None
        values, sources = coordinate_parameters({'observed_resource_location': location, **task.get('params', {})}, scene)
        calculations['move_to_location'] = values
    else:
        handoffs = {}
        for name in operations:
            handoffs[name], peer_sources = _handoff(models, name, location, part, scene=scene, physical=physical)
            sources.extend(peer_sources)
        for rows in handoffs.values():
            _check_peer_conditions(rows, physical.get('environment_valuation', evidence.get('environment_valuation', {})), part, location)
        recorded = evidence.get('observed_parts', {}).get(part, {})
        state = {**evidence.get('snapshot', {}), **snapshot}
        if operations[0] == 'pick_approach':
            if physical.get('products', evidence.get('environment_products', {})).get(part, {}).get('location') != location:
                raise KeyError('Recorded part location and custody for ' + str(part))
            calculations['pick_part'] = pick_targets(scene, part, recorded.get('observed_pose', recorded.get('current_pose')),
                state['current_pose'], state['base_pose'], location=location)
        else:
            attachment = evidence.get('attachment') or {}
            if attachment.get('attached') is not True or attachment.get('held_part', attachment.get('part_name')) != part:
                raise KeyError('Recorded attachment and part identity')
            calculations['place_release'] = place_targets(scene, declaration, attachment['grasp_transform'],
                state['current_pose'], state['base_pose'], declaration.get('workholding_pose'), part_name=part)
        for name in operations:
            for rows in handoffs[name]:
                event_id = 1 + max(event['event_id'] for owner in models.values() for event in owner.get('events', []))
                for rid, event in rows.items():
                    event['event_id'] = event_id
                    models[rid]['events'].append(event)
        sources = [reference, *sources]
    # resource_location is an instance enumeration for KMR's implemented
    # coordinate operation. All other domains and every physical limit survive.
    domain = models['KMR']['state_variables']['resource_location']['domain']
    for value in (start.get('resource_location'), location):
        if isinstance(value, str) and value not in domain:
            coordinate_parameters({'observed_resource_location': value}, scene)
            domain.append(value)
    evidence['environment_models'] = models
    calculation = environment_recovery_descriptor(evidence, snapshot, resource_jid=task['resource_jid'])
    audit = {'session_id': session_id, 'task': deepcopy(task), 'snapshot_fingerprint': fingerprint(snapshot),
             'source_model_fingerprint': fingerprint(model), 'source_evidence_fingerprint': fingerprint(original),
             'inputs_fingerprint': fingerprint(evidence.get('inputs', {})),
             'physical_evidence_fingerprint': fingerprint(physical),
             'attachment_fingerprint': fingerprint(evidence.get('attachment')),
             'environment_models_fingerprint': fingerprint(models),
             'calculation_model_fingerprint': fingerprint(calculation),
             'sources': sources, 'calculations': calculations, 'command_sent': False, 'execution_authorized': False}
    audit['fingerprint'] = fingerprint(audit)
    updated = deepcopy(physical_input)
    # The original model remains bound in the audit. Downstream checks use the
    # exact detached model whose effects and geometry they are validating.
    updated['base_model_fingerprint'] = fingerprint(calculation)
    updated['resource_evidence'] = evidence
    updated.setdefault('physical_evidence', {}).update(environment_models=deepcopy(models))
    updated['physical_evidence'].setdefault('additional_binding_records', []).append(deepcopy(audit))
    return calculation, updated, audit

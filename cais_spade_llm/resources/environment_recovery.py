from __future__ import annotations

"""Recorded environment declarations used by live and detached recovery owners."""

import math
from copy import deepcopy
from fractions import Fraction
from itertools import combinations_with_replacement, product

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
from cais_spade_llm.resources.environment_models import feasibility
from cais_spade_llm.resources.nominal_conveyor import conveyor_location_at
from cais_spade_llm.resources.recovery_feasibility import derive_primitive_union


def environment_owner_snapshot(owner) -> dict:
    """Export retained owner observations without acquiring or inventing state."""
    context = getattr(owner, 'environment_context', None)
    values = deepcopy(context.snapshot()) if context is not None else {}
    runtime = getattr(owner, 'environment_runtime', None)
    failure = getattr(getattr(runtime, 'conveyor_fault', None), 'evidence', {})
    record = failure.get('checkpoint_processing_observations', failure.get('processing_observations', {})).get(owner.agent_name)
    worker = getattr(owner, 'workflow_worker', None)
    if record is None and worker is not None and hasattr(worker, 'read_machine_jobs'):
        record = worker.read_machine_jobs()
    if record:
        from cais_spade_llm.resources.machine.machine_jobs import job_public_state
        values.update(job_public_state(record))
    primitive = getattr(owner, '_primitive_state', {})
    for key, value in primitive.items():
        if key in values and recovery_validation_fingerprint(values[key]) != recovery_validation_fingerprint(value):
            raise ValueError(f'Conflicting retained owner observations: {owner.agent_name}.{key}')
        values[key] = deepcopy(value)
    values.update(resource_id=owner.agent_name, resource_jid=str(owner.jid),
                  resource_type='kmr' if owner.agent_name == 'KMR' else 'resource')
    return values


def environment_owner_evidence(owner) -> dict:
    """Capture the environment model and facts already held by the owning resource."""
    runtime = getattr(owner, 'environment_runtime', None)
    context = getattr(runtime, 'context', None)
    result = {'snapshot': environment_owner_snapshot(owner), 'state': {},
              'resource_id': owner.agent_name,
              'resource_type': 'kmr' if owner.agent_name == 'KMR' else 'resource',
              'source': {'resource_jid': str(owner.jid), 'kind': 'resource_owned_observations'}}
    result['state'] = deepcopy(result['snapshot'])
    if context is None:
        return result
    for field, attribute in (('environment_models', 'models'), ('environment_products', 'part_tracker'),
                             ('geometry', 'geometry'), ('requirements', 'requirements'),
                             ('inputs', 'inputs'), ('product_name', 'product_name')):
        if hasattr(context, attribute):
            result[field] = deepcopy(getattr(context, attribute))
    result['environment_valuation'] = deepcopy(context.snapshot())
    worker = getattr(owner, 'workflow_worker', None)
    if worker is not None and hasattr(worker, 'read_machine_jobs'):
        failure = getattr(getattr(runtime, 'conveyor_fault', None), 'evidence', {})
        record = failure.get('checkpoint_processing_observations', failure.get('processing_observations', {})).get(owner.agent_name) or worker.read_machine_jobs()
        if record:
            from cais_spade_llm.resources.machine.machine_jobs import job_public_state
            result['processing_observations'] = record
            fields = job_public_state(record)
            for key in ('snapshot', 'state'):
                result[key].update(deepcopy(fields))
            result['environment_valuation'][owner.agent_name].update(deepcopy(fields))
    result['source'].update(run_id=context.run_id, revision=context.revision)
    result['execution_mode'] = getattr(runtime, 'setup', {}).get('execution_mode')
    result['unavailable_resources'] = sorted(getattr(context, 'unavailable_resources', []))
    setup = getattr(runtime, 'setup', {})
    permitted = setup.get('recovery_permitted_resources', setup.get('permitted_resources'))
    if permitted is not None:
        result['recovery_permitted_resources'] = deepcopy(permitted)
    if hasattr(owner, 'workflow_custody'):
        result['attachment'] = deepcopy(owner.workflow_custody)
    result['model_fingerprint'] = recovery_validation_fingerprint(result.get('environment_models', {}))
    return result


def environment_primitive_catalog(evidence: dict) -> list[dict]:
    """Keep declared execution and recovery-selection flags distinct."""
    rid = evidence.get('resource_id')
    model = evidence.get('environment_models', {}).get(rid, {})
    catalog = []
    for name, declaration in model.get('primitive_catalog', {}).items():
        row = {'name': name, 'params': {}, 'required_params': [], **deepcopy(declaration)}
        if rid == 'KMR':
            from cais_spade_llm.recovery_framework.kmr_primitives import KMRPrimitives
            from cais_spade_llm.resources.resource_primitives import _schema_properties_for_function
            function = getattr(KMRPrimitives, name, None)
            if function is not None:
                row['params'], row['required_params'], row['description'] = _schema_properties_for_function(function)
        elif name == 'select_program':
            row['params'] = {'program': {'type': 'string'}}
            row['required_params'] = ['program']
        elif name in {'start_job', 'pause_job', 'resume_job', 'stop_job'}:
            row['params'] = {'job_id': {'type': 'string'}}
            row['required_params'] = ['job_id']
            if name == 'start_job':
                row['params'].update({key: {'type': 'string'} for key in ('task_id', 'part_name', 'process', 'result')})
                row['required_params'] = ['part_name', 'process', 'result']
        elif name == 'dwell':
            row['params'] = {'duration_sec': {'type': ['number', 'null']}}
        elif name == 'move_relative':
            row['params'] = {'speed_mps': {'type': ['number', 'null']}}
        catalog.append(row)
    return catalog


def environment_declared_capabilities(evidence: dict) -> dict:
    """Export exact process and program declarations without owner event recipes."""
    rid = evidence.get('resource_id')
    model = evidence.get('environment_models', {}).get(rid, {})
    result = {field: deepcopy(model[field]) for field in ('process_capabilities', 'current_configuration')
              if field in model and model[field]}
    locations = model.get('state_variables', {}).get('resource_location', {})
    if not locations.get('private') and isinstance(locations.get('domain'), list):
        result['reachable_locations'] = [deepcopy(value) for value in locations['domain']
                                         if isinstance(value, str)]
    for machine in evidence.get('inputs', {}).get('scene', {}).get('machines', []):
        if machine.get('resource_id') == rid and 'program_options' in machine:
            result['program_options'] = deepcopy(machine['program_options'])
            if 'job_control' in machine.get('simulation_process', {}):
                result['job_control'] = deepcopy(machine['simulation_process']['job_control'])
    if evidence.get('processing_observations'):
        from cais_spade_llm.resources.machine.machine_jobs import public_processing_observations
        result['processing_observations'] = public_processing_observations(evidence['processing_observations'])
    return result


def public_evaluation_support(entry: dict) -> dict:
    """Describe implemented owner checks without exposing their private programs."""
    model = entry.get('recovery_des_model') or {}
    events = model.get('events') or []
    # Both environment owners and robot owners implement condition validation.
    # An absent executable declaration must not become an advertised capability.
    supported = [event for event in events if event.get('primitive_support')
                 and event.get('program_status', 'implemented') == 'implemented']
    fields = sorted({field for event in supported for field in event.get('updates', {})
                     if field in model.get('state_variables', {})
                     and not model['state_variables'][field].get('private')
                     and not field.startswith(('task_ctx.', '_'))})
    result = {'public_conditions': True} if fields else {}
    if model.get('resource_id') == 'KMR':
        names = {event.get('event_name') for event in supported}
        operations = [name for name in ('move_to_location', 'pick_approach', 'pick_part',
                                       'place_approach', 'place_release') if name in names]
        if operations:
            result['additional_bindings'] = operations
    return result


def storage_placement_conditions(model: dict, scene: dict, valuation: dict, part: str) -> dict:
    """Calculate receiving conditions for an exact recorded empty Storage slot.

    This owner calculation supplies inventory effects only. KMR must separately
    establish its placement program, docking, attachment and supported rest.

    Args:
        model: The captured Storage owner declaration.
        scene: The unchanged recorded scene configuration.
        valuation: Current projected Storage inventory.
        part: Exact proposed part identifier.

    Returns:
        Bound inventory conditions and their recorded source fingerprints.
    """
    storage = scene['Storage']
    domain = model.get('state_variables', {}).get('inventory.{part_name}', {}).get('domain')
    if model.get('resource_id') != 'Storage' or domain != [False, True]:
        raise KeyError('Recorded Storage inventory declaration')
    slot = storage.get('slots', {}).get(part)
    if (not isinstance(slot, list) or len(slot) != 6
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in slot)):
        raise KeyError('Exact recorded Storage slot for ' + part)
    field = 'inventory.' + part
    if type(valuation.get(field)) is not bool:
        raise KeyError('Recorded Storage inventory for ' + part)
    if valuation[field]:
        raise ValueError('Recorded Storage slot is occupied: ' + part)
    return {'guards': {field: {'equals': False}}, 'updates': {field: {'set': True}},
        'source': '/inputs/scene/Storage/slots/' + part,
        'model_fingerprint': recovery_validation_fingerprint(model),
        'scene_fingerprint': recovery_validation_fingerprint(storage),
        'valuation_fingerprint': recovery_validation_fingerprint(valuation),
        'physical_access_validated': False, 'execution_authorized': False}


# Settled positions vary by a few millimetres around a declared area's edge.
_REST_MARGIN_M = .005


def observed_receiver(scene: dict, conveyor_model: dict | None, part: str, pose: list) -> tuple[str, str] | None:
    """Name the declared receiving area a released part rests in, from its settled position.

    Buffer zones, machine workholding and staging trays, and Conveyor loading
    areas come from the declared safety regions; Storage is the part's own
    recorded slot; the rest of the belt, which runs along world x, is ordered
    by the Conveyor's own loading and output-nest positions.

    Args:
        scene: The recorded scene configuration.
        conveyor_model: The Conveyor descriptor, or None when the cell has none.
        part: The released part.
        pose: The settled world position of the part, x, y, z first.

    Returns:
        ``(resource_id, area)``: a Buffer zone field such as ``zone_2_part``, a
        destination token such as ``M1`` or ``M1 staging tray``, ``Storage``, or
        a Conveyor location. None for a surface no receiver declares.
    """
    regions = scene.get('safety_preparation', {}).get('regions', {})
    machines = {row['resource_id'] for row in scene.get('machines', [])}

    def inside(bounds):
        return all(low - _REST_MARGIN_M <= value <= high + _REST_MARGIN_M
                   for value, (low, high) in zip(pose[:3], bounds, strict=True))

    for name, row in sorted(regions.items()):
        owner, _, area = name.partition('.')
        if not inside(row['bounds']):
            continue
        if owner == 'Buffer For Machined parts' and area.startswith('zone_'):
            return owner, f'{area}_part'
        if owner in machines and area == 'workholding':
            return owner, owner
        if owner in machines and area == 'staging':
            return owner, f'{owner} staging tray'
    slot = scene.get('Storage', {}).get('slots', {}).get(part)
    if isinstance(slot, list) and math.dist(pose[:3], slot[:3]) <= 3 * _REST_MARGIN_M:
        return 'Storage', 'Storage'
    conveyor = scene.get('Conveyor')
    if conveyor is None or conveyor_model is None:
        return None
    x0, y0 = conveyor['world_pose'][:2]
    surface = conveyor['surface_height']
    if not (abs(pose[0] - x0) <= conveyor['length'] / 2 and abs(pose[1] - y0) <= conveyor['width'] / 2
            and surface - _REST_MARGIN_M <= pose[2] <= surface + .05):
        return None
    loading = regions.get('Conveyor.loading_position_1', {}).get('bounds')
    half_width = (loading[0][1] - loading[0][0]) / 2 if loading else .15
    location = conveyor_location_at(conveyor_model, pose[0], half_width)
    return ('Conveyor', location) if location else None


def _binding_values(name: str, declaration: dict, model: dict, evidence: dict) -> list | None:
    if isinstance(declaration.get('domain'), list):
        return deepcopy(declaration['domain'])
    if name == 'job_id':
        return list(evidence.get('processing_observations', {}).get('jobs', {}))
    if set(declaration) == {'equals'}:
        return [deepcopy(declaration['equals'])]
    if declaration.get('reference') == 'part_name':
        return list(evidence.get('environment_products', {}))
    if name == 'result' and declaration.get('type') == 'string':
        return list(dict.fromkeys(value for row in model.get('process_capabilities', {}).values()
                                  for value in row.get('supported_results', [])))
    if name == 'target' and declaration.get('type') == 'string':
        return list(dict.fromkeys(row['target'] for row in evidence.get('geometry', {}).values()
                                  if row.get('target') is not None))
    return None


def environment_recovery_descriptor(evidence: dict, snapshot: dict, *, resource_jid: str) -> dict:
    """Bind declared finite identities without changing task or primitive symbols.

    Completion acknowledgements remain composition obligations. Enumeration uses
    recorded identities and declared capability domains, never a recovery sequence.
    """
    rid = evidence.get('resource_id')
    model = deepcopy(evidence.get('environment_models', {}).get(rid, {}))
    parts = list(evidence.get('environment_products', {}))
    variables = {}
    for field, declaration in model.get('state_variables', {}).items():
        fields = [field.replace('{part_name}', part) for part in parts] if '{part_name}' in field else [field]
        for bound in fields:
            row = deepcopy(declaration)
            row.setdefault('scope', 'resource')
            if 'domain' in row:
                row['domain_complete'] = True
            else:
                row['domain'] = ([None, *parts] if row.get('reference') == 'part_name'
                                 else [deepcopy(snapshot[bound])] if bound in snapshot else [])
                row['domain_complete'] = False
            variables[bound] = row
    machine = next((row for row in evidence.get('inputs', {}).get('scene', {}).get('machines', [])
                    if row.get('resource_id') == rid), {})
    if machine.get('simulation_process', {}).get('job_control', {}).get('version') == 1:
        meanings = {
            'active_job': 'Identity of the currently processing job, or null when none is processing. Retained jobs are separate records.',
            'job_state': 'State of active_job when present, otherwise the last retained job.',
            'selected_program': 'Currently selected declared program; installed tool and workholding are recorded separately.',
        }
        for field, description in meanings.items():
            if field in variables:
                variables[field].setdefault('description', description)
    # Product claims remain obligations; they are never resource effects.
    for field in ('part_state', 'part_location'):
        if field not in variables:
            variables[field] = {'scope': 'part', 'domain': [], 'domain_complete': False}
    events, unresolved, assignments = [], [], 0
    for declared in model.get('events', []):
        if declared.get('parameter_bindings', {}).get('resource_id', {}).get('equals') != rid:
            continue
        if declared.get('collection_effects'):
            from cais_spade_llm.resources.nominal_conveyor import (
                CONVEYOR_LOCATIONS, conveyor_parts, conveyor_advance_parameters,
            )
            if (rid != 'Conveyor' or declared.get('event_name') != 'advance_conveyor'
                    or declared['collection_effects'].get('validation_source') != 'cais_spade_llm.resources.nominal_conveyor'):
                unresolved.append({'reason': 'Unsupported owner collection effect declaration'})
                continue
            try:
                residents = conveyor_parts(model, snapshot)
            except (KeyError, ValueError) as exc:
                unresolved.append({'reason': str(exc)})
                continue
            delivery = declared['parameter_bindings']['delivered_part']
            delivered = delivery['equals'] if 'equals' in delivery else (residents[0] if residents else None)
            remaining = [part for part in residents if part != delivered]
            for indices in combinations_with_replacement(range(len(CONVEYOR_LOCATIONS)), len(remaining)):
                assignments += 1
                if assignments > 4096:
                    unresolved.append({'reason': 'Environment parameter assignment limit reached'})
                    break
                targets = dict(zip(remaining, (CONVEYOR_LOCATIONS[i] for i in reversed(indices)), strict=True))
                try:
                    calculated = conveyor_advance_parameters(model, snapshot, targets, delivered)
                except ValueError:
                    continue
                event = deepcopy(declared)
                bindings = {name: deepcopy(row['equals']) for name, row in declared['parameter_bindings'].items() if 'equals' in row}
                bindings.update(next_locations=targets, delivered_part=delivered)
                event['parameter_bindings'] = {name: {'equals': value} for name, value in bindings.items()}
                event['guards'] = {}
                for field, guard in declared.get('guards', {}).items():
                    for name, value in bindings.items():
                        field = field.replace('{' + name + '}', str(value))
                    event['guards'][field] = deepcopy(guard)
                event['updates'] = {field.removeprefix('next_'): {'set': deepcopy(value)} for field, value in calculated.items()}
                event['primitive_support'] = [dict(deepcopy(step), primitive=step['op']) for step in declared['program']['steps']]
                event['composition_contract'] = {'event_id': declared['event_id'], 'parameter_bindings': bindings,
                    'completion_evidence': deepcopy(declared.get('completion_evidence', {})),
                    'participants': deepcopy(declared['participants']), 'product_effects': deepcopy(declared['product_effects']),
                    'source_model_fingerprint': recovery_validation_fingerprint(model),
                    'source_snapshot_fingerprint': recovery_validation_fingerprint(snapshot),
                    'collection_effects': deepcopy(declared['collection_effects'])}
                events.append(event)
            continue
        choices = {name: (None if name == 'job_id' and declared['event_name'] == 'start_job'
                         else _binding_values(name, row, model, evidence))
                   for name, row in declared.get('parameter_bindings', {}).items()}
        finite = {name: values for name, values in choices.items() if values}
        for values in product(*finite.values()):
            assignments += 1
            if assignments > 4096:
                unresolved.append({'reason': 'Environment parameter assignment limit reached'})
                break
            bindings = dict(zip(finite, values, strict=True))
            event = deepcopy(declared)
            event['parameter_bindings'] = {name: ({'equals': deepcopy(bindings[name])} if name in bindings else deepcopy(row))
                                           for name, row in declared['parameter_bindings'].items()}
            for section in ('guards', 'updates'):
                event[section] = {}
                for field, value in declared.get(section, {}).items():
                    for name, bound in bindings.items():
                        field = field.replace('{' + name + '}', str(bound))
                    event[section][field] = deepcopy(value)
            event['primitive_support'] = [dict(deepcopy(step), primitive=step['op'])
                                          for step in declared.get('program', {}).get('steps', [])]
            event['composition_contract'] = {'event_id': declared['event_id'],
                'parameter_bindings': deepcopy(bindings), 'completion_evidence': deepcopy(declared.get('completion_evidence', {})),
                'participants': deepcopy(declared.get('participants', [])),
                'product_effects': deepcopy(declared.get('product_effects', {})),
                'source_model_fingerprint': recovery_validation_fingerprint(model)}
            events.append(event)
        if assignments > 4096:
            break
    descriptor = {'resource_id': rid, 'resource_jid': resource_jid,
        'resource_type': evidence.get('resource_type'), 'state_variables': variables,
        'current_valuation': {field: deepcopy(snapshot[field]) for field in variables if field in snapshot},
        'events': events, 'primitive_catalog': environment_primitive_catalog(evidence),
        'model_complete': not unresolved, 'unresolved_declarations': unresolved,
        'source_model_fingerprint': recovery_validation_fingerprint(model),
        'primitive_support_semantics': 'successor_primitive_union', 'validation_contract_version': 2}
    descriptor['descriptor_fingerprint'] = recovery_validation_fingerprint(descriptor)
    return descriptor


def _result(status: str, reason: str, evidence: dict, snapshot: dict, **facts) -> dict:
    return {'allowed': status == 'FEASIBLE', 'feasibility_status': status, 'reason': reason,
            'evidence': {'source': deepcopy(evidence.get('source', {})),
                         'source_fingerprint': recovery_validation_fingerprint(evidence),
                         'snapshot_fingerprint': recovery_validation_fingerprint(snapshot),
                         'command_sent': False, 'execution_authorized': False, **facts}}


def kmr_validation_evidence(evidence: dict, snapshot: dict, physical: dict | None) -> dict:
    """Read accepted KMR projections separately from immutable capture facts."""
    result = deepcopy(evidence)
    if (physical or {}).get('endpoint_evidence_policy_version') in (1, 2):
        result['endpoint_evidence_policy_version'] = physical['endpoint_evidence_policy_version']
    provenance = (physical or {}).get('provenance', {})
    if provenance.get('kind') != 'projected':
        return result
    if (not isinstance(provenance.get('binding'), dict) or not provenance['binding']
            or not isinstance(provenance.get('contract_fingerprint'), str) or not provenance['contract_fingerprint']):
        raise ValueError('Projected KMR evidence lacks its accepted owner binding')
    identities = [record['resource_jid'] for record in (
        snapshot, snapshot.get('resource_core', {}), evidence.get('source', {}), evidence.get('snapshot', {}))
        if 'resource_jid' in record]
    if (not identities or not physical.get('jids', {}).get('KMR')
            or any(identity != physical['jids']['KMR'] for identity in identities)):
        raise ValueError('Projected KMR identity differs from its validating owner')
    owner = physical['snapshot']['resources']['KMR']
    declarations = evidence['environment_models']['KMR']['state_variables']
    for field in declarations:
        if field in snapshot and (field not in owner or recovery_validation_fingerprint(snapshot[field])
                                  != recovery_validation_fingerprint(owner[field])):
            raise ValueError('Projected KMR public snapshot conflicts with accepted ' + field)
    result['snapshot'] = {**deepcopy(snapshot), **deepcopy(owner)}
    result['continuous_motion_configuration'] = deepcopy(physical['resource_models']['KMR']['configuration'])
    result['snapshot']['continuous_motion_configuration'] = deepcopy(result['continuous_motion_configuration'])
    result['observed_parts'] = {part: {'current_pose': deepcopy(row['current_pose']),
        'source': deepcopy(provenance), 'kind': 'projected'} for part, row in physical['snapshot']['parts'].items()
        if row.get('current_pose') is not None}
    result.pop('attachment', None)
    if 'held_part' in owner:
        if owner['held_part'] is None:
            result['attachment'] = {'attached': False, 'source': deepcopy(provenance)}
        elif 'grasp_transform' in owner:
            result['attachment'] = {'attached': True, 'held_part': owner['held_part'],
                'grasp_transform': deepcopy(owner['grasp_transform']), 'source': deepcopy(provenance)}
    result['validation_projection'] = {'source': deepcopy(provenance),
        'source_capture_evidence_fingerprint': recovery_validation_fingerprint(evidence),
        'physical_evidence_fingerprint': recovery_validation_fingerprint(physical)}
    return result


def check_environment_primitive_capability(*, primitive: str, params: dict, task: dict,
                                          recovery_snapshot: dict, resource_evidence: dict,
                                          physical_evidence: dict | None = None,
                                          **_unused) -> dict:
    """Check one declared primitive against one unchanged owner snapshot."""
    evidence, snapshot = resource_evidence, recovery_snapshot
    rid = evidence.get('resource_id')
    def answer(status, reason, **facts):
        return _result(status, reason, evidence, snapshot, primitive=primitive, params=deepcopy(params), **facts)
    if rid in evidence.get('unavailable_resources', []):
        return answer('INFEASIBLE', 'Resource is recorded unavailable')
    if 'recovery_permitted_resources' in evidence and rid not in evidence['recovery_permitted_resources']:
        return answer('INFEASIBLE', 'Resource is not permitted for recovery')
    model = evidence.get('environment_models', {}).get(rid)
    if not isinstance(model, dict):
        return answer('NEEDS_CONTEXT', 'Recorded environment resource model is unavailable')
    declaration = model.get('primitive_catalog', {}).get(primitive)
    if not isinstance(declaration, dict) or declaration.get('status') != 'executable':
        return answer('NEEDS_CONTEXT', 'Primitive has no declared executable owner capability')
    if rid == 'KMR':
        from cais_spade_llm.recovery_framework.kmr_recovery import check_kmr_primitive_capability
        try:
            evidence = kmr_validation_evidence(evidence, snapshot, physical_evidence)
        except (KeyError, ValueError, TypeError) as exc:
            return answer('NEEDS_CONTEXT', str(exc))
        return check_kmr_primitive_capability(primitive=primitive, params=params, task=task,
            recovery_snapshot=snapshot, resource_evidence=evidence)
    descriptor = environment_recovery_descriptor(evidence, snapshot, resource_jid=task.get('resource_jid', ''))
    support = derive_primitive_union(task=task, recovery_des_model=descriptor, recovery_snapshot=snapshot)
    contributors = [row for row in support['matching_transitions'] if primitive in row['primitives']]
    if not contributors:
        return answer('NEEDS_CONTEXT', 'No declared owner successor supplies this primitive', support=support)
    scene = evidence.get('inputs', {}).get('scene', {})
    checks = []
    for contributor in contributors:
        bindings = contributor['parameter_bindings']
        from cais_spade_llm.resources.machine.machine_jobs import MACHINE_PRIMITIVES
        if primitive in MACHINE_PRIMITIVES:
            try:
                record, reply = _machine_command_projection(evidence, snapshot, primitive,
                    {**bindings, **params}, task, physical_evidence or {})
            except (KeyError, TypeError, ValueError) as exc:
                missing = isinstance(exc, (KeyError, TypeError)) or 'NEEDS_CONTEXT:' in str(exc)
                checks.append({'event_name': contributor['event_name'], 'parameter_bindings': deepcopy(bindings),
                    'status': 'NEEDS_CONTEXT' if missing else 'INFEASIBLE', 'reasons': [str(exc)]})
                continue
            return answer('FEASIBLE', 'Independent job command capability; no processing completion established',
                          projected_command=reply, model_fingerprint=record['model_fingerprint'])
        part = bindings.get('part_name')
        geometry = evidence.get('geometry', {}).get(part, {})
        if rid == 'Conveyor' and part is None:
            from cais_spade_llm.resources.nominal_conveyor import conveyor_parts

            resident = conveyor_parts(model, evidence.get('environment_valuation', {}).get(rid, {}))
            checked_parts = [feasibility(model, {'event_name': contributor['event_name'], 'parameters': bindings},
                evidence.get('geometry', {}).get(name, {})) for name in resident]
            if not checked_parts:
                status, reasons = 'NEEDS_CONTEXT', ['No recorded Conveyor part custody is available']
            else:
                status = ('INFEASIBLE' if any(row[0] == 'INFEASIBLE' for row in checked_parts) else
                          'NEEDS_CONTEXT' if any(row[0] == 'NEEDS_CONTEXT' for row in checked_parts) else 'FEASIBLE')
                reasons = [reason for row in checked_parts for reason in row[1]]
        else:
            status, reasons = feasibility(model, {'event_name': contributor['event_name'], 'parameters': bindings}, geometry)
        if primitive == 'dwell':
            machine = next((row for row in scene.get('machines', []) if row.get('resource_id') == rid), {})
            minimum = machine.get('simulation_process', {}).get('processing_time_sec')
            duration = params.get('duration_sec', minimum)
            if duration is None:
                duration = minimum
            if type(minimum) not in (int, float) or not math.isfinite(minimum) or minimum <= 0:
                status, reasons = 'NEEDS_CONTEXT', ['Declared processing_time_sec is unavailable']
            elif type(duration) not in (int, float) or not math.isfinite(duration) or duration < minimum:
                status, reasons = 'INFEASIBLE', ['Machining dwell is below the declared processing time']
        elif primitive == 'move_relative':
            configured = scene.get(rid, {}).get('simulation_transport', {}).get('speed_mps')
            speed = params.get('speed_mps', configured)
            if speed is None:
                speed = configured
            if type(configured) not in (int, float) or not math.isfinite(configured) or configured <= 0:
                status, reasons = 'NEEDS_CONTEXT', ['Declared transport speed is unavailable']
            elif type(speed) not in (int, float) or not math.isfinite(speed) or not 0 < speed <= configured:
                status, reasons = 'INFEASIBLE', ['Transport speed exceeds the declared owner capability']
        else:
            status, reasons = 'NEEDS_CONTEXT', ['No owner calculation supports this primitive']
        checks.append({'event_name': contributor['event_name'], 'parameter_bindings': deepcopy(bindings),
                       'status': status, 'reasons': reasons})
        if status == 'FEASIBLE':
            return answer(status, 'Declared primitive capability is supported; ordering and physical intervals remain separate', checks=checks)
    status = 'NEEDS_CONTEXT' if any(row['status'] == 'NEEDS_CONTEXT' for row in checks) else 'INFEASIBLE'
    return answer(status, '; '.join(reason for row in checks for reason in row['reasons']), checks=checks)


def _machine_command_projection(evidence: dict, snapshot: dict, primitive: str, params: dict,
                                task: dict, physical: dict) -> tuple[dict, dict]:
    from cais_spade_llm.resources.machine.machine_jobs import project_job_command

    rid = evidence['resource_id']
    machine = next(row for row in evidence['inputs']['scene']['machines'] if row['resource_id'] == rid)
    record = deepcopy(physical.get('processing_observations', {}).get(rid, evidence.get('processing_observations')))
    if not record:
        raise ValueError('NEEDS_CONTEXT: processing_observations/' + rid)
    params = deepcopy(params)
    from cais_spade_llm.resources.machine.machine_jobs import job_public_state
    for field, value in job_public_state(record).items():
        if field in snapshot and snapshot[field] != value:
            raise ValueError('NEEDS_CONTEXT: processing observation differs from projected ' + field)
    if primitive == 'start_job':
        # New identities bind predictions to this proposal; they are not observations.
        identity = recovery_validation_fingerprint(task)
        params.setdefault('job_id', identity)
        params.setdefault('task_id', identity)
    job = record['jobs'].get(params.get('job_id'), {})
    part = params.get('part_name', job.get('part_name'))
    observation = deepcopy(job.get('observations', {}))
    if primitive in {'start_job', 'resume_job'}:
        current_access = physical.get('machine_access_observations', {}).get(rid)
        if physical.get('provenance', {}).get('kind') == 'projected' and current_access is None:
            current_access = _machine_access_projection(evidence, snapshot, task, physical, part)
        observation = deepcopy(current_access if current_access is not None else observation)
        if snapshot.get('part_name') != part:
            raise ValueError('Observed workholding does not contain the requested part')
    return project_job_command(machine, record, primitive, params, observation,
        command_id=recovery_validation_fingerprint([task, primitive, params]),
        allow_relative_time=physical.get('outline_evidence_policy_version') == 2)


def _machine_distance_bounds(bounds: list, point: list) -> tuple[float, float]:
    """Enclose distance from a recorded point to a calculated position box."""
    if len(bounds) != 3 or len(point) != 3:
        raise ValueError('NEEDS_CONTEXT: three-dimensional machine access geometry')
    lower, upper = Fraction(0), Fraction(0)
    for pair, coordinate in zip(bounds, point, strict=True):
        if (len(pair) != 2 or any(type(value) not in (int, float) or not math.isfinite(value)
                                  for value in (*pair, coordinate)) or pair[0] > pair[1]):
            raise ValueError('NEEDS_CONTEXT: finite ordered machine access geometry')
        low, high = (Fraction(value) - Fraction(coordinate) for value in pair)
        lower += (min(abs(low), abs(high)) if low > 0 or high < 0 else Fraction(0)) ** 2
        upper += max(abs(low), abs(high)) ** 2
    start, end = math.sqrt(float(lower)), math.sqrt(float(upper))
    while Fraction(start) ** 2 > lower:
        start = math.nextafter(start, -math.inf)
    while Fraction(end) ** 2 < upper:
        end = math.nextafter(end, math.inf)
    return start, end


def _machine_access_transforms(physical: dict, rid: str) -> tuple[dict, dict]:
    """Use an established peer hold, never an old pose."""
    from cais_spade_llm.resources.continuous_motion import interval, kinematic_transforms
    from cais_spade_llm.resources.recovery_geometry import commanded_hold_evidence

    state, model = physical['snapshot']['resources'][rid], physical['resource_models'][rid]
    resource = physical['resources'][rid]
    hold = commanded_hold_evidence(resource_id=rid, state=state, model=model, physical=resource,
        robot_description=physical.get('endpoint_robot_description'), include_observation_roundoff=True)
    if hold.get('status') != 'supported':
        raise ValueError('NEEDS_CONTEXT: machine access configuration for ' + rid + ': ' + hold.get('reason', ''))
    configuration = model['configuration']
    if (state.get('joint_names') != configuration['joint_names']
            or len(state['joint_positions']) != len(state['joint_names'])):
        raise ValueError('NEEDS_CONTEXT: machine access joint identities for ' + rid)
    if 'base_pose' in state and state['base_pose'] != [*configuration['root_xyz'], *configuration['root_quaternion']]:
        raise ValueError('NEEDS_CONTEXT: machine access base configuration for ' + rid)
    positions = {name: interval(*bounds) for name, bounds in hold['joint_intervals'].items()}
    transforms = kinematic_transforms(configuration, positions)
    return transforms, {'resource_id': rid, 'hold': hold,
        'state_fingerprint': recovery_validation_fingerprint(state),
        'configuration_fingerprint': recovery_validation_fingerprint(configuration)}


def _machine_access_projection(evidence: dict, snapshot: dict, task: dict, physical: dict, part: str) -> dict:
    """Evaluate the native machine workholding and clearance checks on model bounds."""
    from cais_spade_llm.resources.continuous_motion import transformed_box

    rid = evidence['resource_id']
    scene = evidence['inputs']['scene']
    machine = next(row for row in scene['machines'] if row['resource_id'] == rid)
    if (physical['snapshot'].get('frame') != 'world' or physical['geometry'].get('frame') != 'world'
            or physical['resources'][rid].get('model_static') is not True
            or physical['geometry']['resources'][rid].get('stationary_only') is not True):
        raise ValueError('NEEDS_CONTEXT: recorded static machine geometry in world')
    if snapshot.get('part_name') != part or physical['snapshot']['resources'][rid].get('part_name') != part:
        raise ValueError('Observed workholding does not contain the requested part')
    if any(row.get('held_part') == part for row in physical['snapshot']['resources'].values()):
        raise ValueError('The requested workholding part remains held by a resource')
    state = physical['snapshot']['parts'][part]
    if state.get('location') != rid:
        raise ValueError('NEEDS_CONTEXT: machine and part location declarations conflict')
    support = physical.get('stationary_part_evidence', {}).get(part, {})
    if (support.get('status') != 'supported' or not support.get('source')
            or support.get('current_pose') != state.get('current_pose')):
        raise ValueError('NEEDS_CONTEXT: matching stationary workholding part evidence')
    expected = machine['workholding_pose'][:3]
    present = _machine_distance_bounds([[value, value] for value in state['current_pose'][:3]], expected)
    # These are the same checks used by WorkflowPrimitiveRunner.observe_workholding
    # and its native robot_clearance callback; no clearance is inferred from labels.
    if present[0] > .08:
        raise ValueError('The correct part is not observed at machine workholding')
    if present[1] > .08:
        raise ValueError('NEEDS_CONTEXT: workholding position bounds cross the native tolerance')
    kmr, kmr_source = _machine_access_transforms(physical, 'KMR')
    configured = scene['KMR']['task_execution']['clearance_link_bounds']
    clearance = machine['KMR_process_clearance_m']
    if not configured or type(clearance) not in (int, float) or not math.isfinite(clearance) or clearance < 0:
        raise ValueError('NEEDS_CONTEXT: configured KMR process clearance')
    kmr_bounds = {}
    for link, spheres in configured.items():
        if not spheres:
            raise ValueError('NEEDS_CONTEXT: KMR clearance bounds for ' + link)
        distances = []
        for sphere in spheres:
            radius = sphere['radius']
            if type(radius) not in (int, float) or not math.isfinite(radius) or radius < 0:
                raise ValueError('NEEDS_CONTEXT: finite KMR clearance radius')
            box = transformed_box(kmr[link], [[value, value] for value in sphere['center']], link)
            bounds = [[low[0], high[1]] for low, high in zip(box['minimum'], box['maximum'], strict=True)]
            low, high = _machine_distance_bounds(bounds, expected)
            distances.append((math.nextafter(low - radius, -math.inf), math.nextafter(high - radius, math.inf)))
        kmr_bounds[link] = [min(pair[0] for pair in distances), min(pair[1] for pair in distances)]
    handling = machine['handling_robot']
    robot = next(row for row in scene['robots'] if row['resource_id'] == handling)
    transforms, robot_source = _machine_access_transforms(physical, handling)
    link = robot['prefix'] + robot['clearance_observation_link']
    robot_bounds = _machine_distance_bounds([list(transforms[link][index][3]) for index in range(3)],
                                            state['current_pose'][:3])
    blocked = any(pair[1] < clearance for pair in kmr_bounds.values()) or robot_bounds[1] < .12
    if not blocked and (any(pair[0] < clearance for pair in kmr_bounds.values()) or robot_bounds[0] < .12):
        raise ValueError('NEEDS_CONTEXT: machine access bounds cross the native clearance limits')
    return {'part_name': part, 'correct_part_present': True, 'robot_access_clear': not blocked,
        'robot_clearance_m': {handling: robot_bounds[0]},
        'KMR_arm_clearance_m': {link: pair[0] for link, pair in kmr_bounds.items()},
        'clearance_bounds_m': {'KMR': kmr_bounds, handling: list(robot_bounds)},
        'source': {'kind': 'projected', 'calculation': 'resource_owned_machine_access',
            'task_fingerprint': recovery_validation_fingerprint(task),
            'physical_fingerprint': recovery_validation_fingerprint(physical),
            'machine_fingerprint': recovery_validation_fingerprint(machine),
            'part_support_fingerprint': recovery_validation_fingerprint(support),
            'resources': [kmr_source, robot_source]},
        'physical_execution_verified': False, 'execution_authorized': False}



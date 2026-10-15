from __future__ import annotations

"""Recorded environment declarations used by live and detached recovery owners."""

import math
from copy import deepcopy
from fractions import Fraction
from itertools import combinations_with_replacement, product

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint
from cais_spade_llm.resources.environment_models import feasibility, project_transition
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
    invalidated = 'KMR' in physical.get('invalidated_geometry', {}).get('resources', {})
    if invalidated:
        from cais_spade_llm.recovery_framework.outline_evidence_policy import remove_invalidated_pose

        remove_invalidated_pose(result)
    else:
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
    """Use the current endpoint or an established peer hold, never an old pose."""
    from cais_spade_llm.recovery_framework.outline_endpoint_evidence import _arm_hold, _readiness
    from cais_spade_llm.resources.continuous_motion import interval, kinematic_transforms
    from cais_spade_llm.resources.recovery_geometry import commanded_hold_evidence

    if rid in physical.get('invalidated_geometry', {}).get('resources', {}):
        raise ValueError('NEEDS_CONTEXT: current machine access geometry for ' + rid)
    state, model = physical['snapshot']['resources'][rid], physical['resource_models'][rid]
    resource = physical['resources'][rid]
    if resource.get('projected_endpoint') is not None:
        try:
            _readiness(physical, rid)
        except ValueError as exc:
            raise ValueError('NEEDS_CONTEXT: ' + str(exc)) from exc
        hold = _arm_hold(physical, rid)
    else:
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


def declared_environment_successors(*, task: dict, recovery_snapshot: dict, recovery_des_model: dict,
                                   physical_input: dict, guard_failures: list | None = None) -> tuple[list[dict], list[str]]:
    """Calculate declared successor effects without establishing a motion interval."""
    evidence = physical_input.get('resource_evidence') or {}
    rid = evidence.get('resource_id')
    defer_resource_guards = (physical_input.get('outline_evidence_policy_version') == 2
                            and not task.get('required_task_contract'))
    calculation_model = recovery_des_model
    if rid == 'Conveyor' and evidence.get('environment_models'):
        calculation_model = environment_recovery_descriptor(evidence, recovery_snapshot, resource_jid=task['resource_jid'])
        if calculation_model['source_model_fingerprint'] != recovery_des_model.get('source_model_fingerprint'):
            raise ValueError('Conveyor collection declarations changed after capture')
    support = derive_primitive_union(task=task, recovery_des_model=calculation_model, recovery_snapshot=recovery_snapshot)
    candidates, unresolved = [], []
    if guard_failures is not None and (support.get('missing_context') or support.get('search', {}).get('limit_reached')
                                       or support.get('model_complete') is not True):
        unresolved.append('Owner declaration search lacks complete evidence')
    for contributor in support['matching_transitions']:
        contract = contributor.get('composition_contract', {})
        declared_task = {'resource_id': rid, 'event_id': contract.get('event_id'),
                         'event_name': contributor['event_name'], 'parameters': deepcopy(contributor['parameter_bindings'])}
        try:
            physical = physical_input.get('physical_evidence') or {}
            valuation = deepcopy(physical.get('environment_valuation', evidence['environment_valuation']))
            declaration = next(row for row in evidence['environment_models'][rid]['events']
                               if row['event_id'] == declared_task['event_id'])
            declared_task['parameters'] = {key: value for key, value in declared_task['parameters'].items()
                                          if key in declaration['parameter_bindings']}
            if rid == 'KMR' and declared_task['event_name'] == 'move_to_location':
                from cais_spade_llm.recovery_framework.kmr_recovery import coordinate_parameters
                declared_task['parameters'], _ = coordinate_parameters(declared_task['parameters'], evidence['inputs']['scene'])
            if declared_task['event_name'] == 'advance_conveyor' and 'next_locations' not in declared_task['parameters']:
                delivered = declared_task['parameters'].get('delivered_part')
                targets = {field.removeprefix('part_location.'): value for field, value in task['expected_end_state'].items()
                           if field.startswith('part_location.') and value is not None}
                resident = {field.removeprefix('part_location.'): value for field, value in valuation[rid].items()
                            if field.startswith('part_location.') and value is not None}
                if not targets:
                    raise ValueError('Conveyor projection requires explicit next_locations or proposed recorded part_location fields')
                declared_task['parameters']['next_locations'] = {part: targets.get(part, location)
                    for part, location in resident.items() if part != delivered}
            for field in evidence['environment_models'][rid]['state_variables']:
                if field in recovery_snapshot:
                    valuation[rid][field] = deepcopy(recovery_snapshot[field])
            after, products = project_transition(evidence['environment_models'], valuation,
                physical.get('products', evidence['environment_products']), declared_task,
                evidence['product_name'], evidence['requirements'], defer_resource_guards=defer_resource_guards)
        except (KeyError, ValueError, TypeError) as exc:
            from cais_spade_llm.resources.environment_models import ResourceGuardError

            if guard_failures is not None and isinstance(exc, ResourceGuardError):
                variables = evidence['environment_models'][exc.resource_id]['state_variables']
                public = (exc.field in variables and not variables[exc.field].get('private')
                          and not exc.field.startswith(('task_ctx.', '_')))
                guard_failures.append({'declaration_fingerprint': recovery_validation_fingerprint(declaration),
                    **({'resource_id': exc.resource_id, 'field': exc.field, 'operator': exc.operator,
                        'expected': exc.expected, 'actual': exc.actual} if public else {})})
            unresolved.append(str(exc))
            continue
        candidates.append({'task': declared_task, 'environment_valuation': after,
                           'environment_products': products, 'kind': 'projected',
                           **({'composition_prerequisites': [
                               {'resource_id': participant, 'guards': deepcopy(local['guards']),
                                'collection_guards': deepcopy(local.get('collection_guards', {}))}
                               for participant in declaration['participants']
                               for local in evidence['environment_models'][participant]['events']
                               if local['event_id'] == declaration['event_id']]}
                              if defer_resource_guards else {}),
                           'source_model_fingerprint': recovery_validation_fingerprint(evidence['environment_models']),
                           'source_fingerprint': recovery_validation_fingerprint(evidence)})
    return candidates, unresolved


def project_environment_event(*, task: dict, recovery_snapshot: dict, recovery_des_model: dict,
                              physical_input: dict) -> dict:
    """Retain owner-derived symbolic effects separately from missing motion evidence."""
    if physical_input.get('physical_evidence', {}).get('endpoint_evidence_policy_version') in (1, 2):
        from cais_spade_llm.recovery_framework.outline_evidence_policy import event_evidence_requirements
        from cais_spade_llm.recovery_framework.outline_endpoint_evidence import project_bounded_successor

        required = event_evidence_requirements(task=task, snapshot=recovery_snapshot, model=recovery_des_model,
            physical_input=physical_input, specifications=physical_input.get('outline_safety_specifications',
                physical_input['physical_evidence'].get('outline_safety_specifications')))
        return project_bounded_successor(task=task, snapshot=recovery_snapshot, model=recovery_des_model,
            physical_input=physical_input, requirements=required)
    evidence = physical_input.get('resource_evidence') or {}
    rid = evidence.get('resource_id')
    candidates, unresolved = declared_environment_successors(task=task, recovery_snapshot=recovery_snapshot,
        recovery_des_model=recovery_des_model, physical_input=physical_input)
    rejected = []
    for declared in candidates:
        from cais_spade_llm.resources.machine.machine_jobs import MACHINE_PRIMITIVES
        projector = (_project_machine_dwell if declared['task']['event_name'] == 'machine_part' else
                     _project_machine_command if declared['task']['event_name'] in MACHINE_PRIMITIVES else
                     _project_transport if declared['task']['event_name'] in {'advance_part', 'advance_conveyor'} else None)
        if rid == 'KMR':
            from cais_spade_llm.recovery_framework.kmr_outline_evidence import project_kmr_event
            projector = project_kmr_event
        if projector is None:
            continue
        try:
            projected = projector(task=task, recovery_snapshot=recovery_snapshot,
                recovery_des_model=recovery_des_model, resource_evidence=evidence,
                physical=physical_input.get('physical_evidence') or {}, declared=declared)
            if projected.get('status') == 'projected':
                return projected
            rejected.append(projected)
            if projected.get('status') != 'rejected':
                unresolved.extend(projected.get('unresolved') or
                    ([projected['reason']] if projected.get('reason') else []))
        except (KeyError, ValueError, TypeError, StopIteration) as exc:
            unresolved.append(str(exc))
    if rejected and not unresolved and len(rejected) == len(candidates):
        return rejected[0]
    return {'status': 'needs_context', 'requires_projection': True, 'observations': [],
            'interval_complete': False, 'execution_authorized': False, 'command_sent': False,
            'declared_projections': candidates,
            'unresolved': [*unresolved, 'Owner motion or static interval evidence is required independently of unordered primitive support'],
            'context_requests': [f'/resources/{task.get("resource_jid")}/outline_event_evidence']}


def _project_machine_command(*, task: dict, recovery_snapshot: dict, recovery_des_model: dict,
                              resource_evidence: dict, physical: dict, declared: dict) -> dict:
    """Project instantaneous simulated job controls separately from motion or completion."""
    from cais_spade_llm.resources.machine.machine_jobs import job_public_state
    from cais_spade_llm.resources.recovery_geometry import _checkpoint

    selected = declared['task']
    rid = resource_evidence['resource_id']
    record, reply = _machine_command_projection(resource_evidence, recovery_snapshot,
        selected['event_name'], selected['parameters'], task, physical)
    snapshot, geometry, models = deepcopy(physical['snapshot']), physical['geometry'], physical['resource_models']
    bounded = physical.get('endpoint_evidence_policy_version') == 2
    complete, gap = None, None
    if physical['resources'][rid].get('model_static') is True:
        try:
            complete = _checkpoint(snapshot, geometry, models, index=0)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            if not bounded:
                raise
            gap = str(exc)
    if complete is not None:
        first = complete
        bounded = False
    elif bounded:
        from cais_spade_llm.recovery_framework.outline_endpoint_evidence import (
            OCCUPANCY_FIELDS, _bounded_checkpoint, _occupancy_binding,
        )

        first = _bounded_checkpoint(physical, index=0)
        if gap:
            first['bound_gaps'].append(gap)
        cached = physical.get('endpoint_occupancy')
        if cached is not None:
            if cached['binding'] != _occupancy_binding(physical):
                raise ValueError('Retained endpoint occupancy is stale')
            for field in OCCUPANCY_FIELDS:
                first[field] = deepcopy(cached[field])
    else:
        raise ValueError('NEEDS_CONTEXT: declared static simulation machine geometry')
    first['owner_completion_applied'] = False
    first['state_snapshot'] = deepcopy(physical.get('environment_valuation', resource_evidence['environment_valuation']))
    declared['environment_valuation'][rid].update(job_public_state(record))
    snapshot['resources'][rid].update(deepcopy(declared['environment_valuation'][rid]))
    if bounded:
        # These owner commands update the simulated job ledger at one event
        # boundary. No motion or elapsed processing interval is projected.
        last = deepcopy(first)
        last['resources'] = deepcopy(snapshot['resources'])
        last['parts'] = deepcopy(snapshot['parts'])
    else:
        last = _checkpoint(snapshot, geometry, models, index=0)
    last['owner_completion_applied'] = True
    last['state_snapshot'] = deepcopy(declared['environment_valuation'])
    result = _finish_environment_projection(task=task, recovery_snapshot=recovery_snapshot,
        recovery_des_model=recovery_des_model, resource_evidence=resource_evidence, physical=physical,
        declared=declared, geometry=geometry, models=models, observations=[first, last],
        successor_snapshot=snapshot, contract={'kind': 'resource_owned_simulation_job_control',
            'command': reply, 'model_fingerprint': record['model_fingerprint'],
            'physical_start_fingerprint': recovery_validation_fingerprint(physical)})
    if result.get('status') == 'projected':
        result['processing_observations'] = {rid: deepcopy(record)}
        for field in ('successor_evidence',):
            if isinstance(result.get(field), dict):
                result[field].setdefault('processing_observations', {})[rid] = deepcopy(record)
                successor = result[field]
                for peer, values in physical['resources'].items():
                    endpoint = values.get('projected_endpoint')
                    if (peer == rid or not endpoint
                            or endpoint.get('binding') != physical.get('provenance', {}).get('binding')
                            or endpoint.get('state_fingerprint') != recovery_validation_fingerprint(physical['snapshot']['resources'][peer])
                            or endpoint.get('configuration_fingerprint') != recovery_validation_fingerprint(physical['resource_models'][peer]['configuration'])
                            or successor['snapshot']['resources'][peer] != physical['snapshot']['resources'][peer]
                            or successor['resource_models'][peer] != physical['resource_models'][peer]
                            or successor['geometry']['resources'][peer] != physical['geometry']['resources'][peer]
                            or successor['snapshot']['parts'] != physical['snapshot']['parts']):
                        continue
                    # A zero-duration job control does not move an unchanged
                    # peer. Retain the endpoint's source instead of claiming
                    # that this machine command established a new robot pose.
                    retained = successor['resources'][peer]['projected_endpoint']
                    retained['preserved_endpoint'] = {
                        'fingerprint': recovery_validation_fingerprint(endpoint),
                        'binding': deepcopy(endpoint['binding']),
                        'duration_sec': 0,
                    }
                    retained['binding'] = deepcopy(successor['provenance']['binding'])
                if physical.get('endpoint_evidence_policy_version') == 2 and (
                        bounded or 'endpoint_occupancy' in physical):
                    from cais_spade_llm.recovery_framework.outline_endpoint_evidence import OCCUPANCY_FIELDS, _occupancy_binding

                    result[field]['endpoint_occupancy'] = {'binding': _occupancy_binding(result[field]),
                        **{name: deepcopy(last[name]) for name in OCCUPANCY_FIELDS}}
    return result


def _project_machine_dwell(*, task: dict, recovery_snapshot: dict, recovery_des_model: dict,
                           resource_evidence: dict, physical: dict, declared: dict) -> dict:
    """Project owner processing and supported peer occupancy over its duration."""
    from cais_spade_llm.resources.recovery_geometry import _checkpoint, commanded_hold_evidence

    rid = resource_evidence['resource_id']
    selected = declared['task']
    event = next(row for row in resource_evidence['environment_models'][rid]['events']
                 if row['event_id'] == selected['event_id'])
    operations = [row['op'] for row in event['program']['steps']]
    if selected['event_name'] != 'machine_part' or operations not in (['dwell'], ['start_job']):
        raise ValueError('Owner physical interval calculation is not declared for this event')
    calculation_model = deepcopy(resource_evidence['environment_models'][rid])
    job_record = deepcopy(physical.get('processing_observations', {}).get(rid,
                         resource_evidence.get('processing_observations')))
    if operations == ['start_job']:
        if not job_record:
            raise ValueError('NEEDS_CONTEXT: processing_observations/' + rid)
        calculation_model['current_configuration'] = deepcopy(job_record['current_configuration'])
    status, reasons = feasibility(calculation_model, selected,
                                  resource_evidence['geometry'][selected['parameters']['part_name']])
    if status != 'FEASIBLE':
        raise ValueError('; '.join(reasons))
    scene = resource_evidence['inputs']['scene']
    machine = next(row for row in scene['machines'] if row['resource_id'] == rid)
    duration = event['program']['steps'][0].get('params', {}).get('duration_sec')
    if duration is None:
        duration = machine['simulation_process']['processing_time_sec']
    completed_jobs = None
    if operations == ['start_job']:
        from cais_spade_llm.resources.machine.machine_jobs import project_job_completion, job_public_state
        active = job_record['active_job']
        if active is None:
            job_record, _reply = _machine_command_projection(resource_evidence, recovery_snapshot,
                'start_job', selected['parameters'], task, physical)
            active = job_record['active_job']
        job = job_record['jobs'][active]
        if any(job.get(key) != selected['parameters'].get(key) for key in ('part_name', 'process', 'result')):
            raise ValueError('Active job differs from the nominal machining obligation')
        duration = job['processing_time_sec'] - job['processing_active_sec']
        completed_jobs = project_job_completion(machine, job_record, duration, job['observations'])
        declared['environment_valuation'][rid].update(job_public_state(completed_jobs))
    requested_duration = task.get('params', {}).get('duration_sec')
    if requested_duration is not None and recovery_validation_fingerprint(requested_duration) != recovery_validation_fingerprint(duration):
        raise ValueError('Requested duration differs from the declared owner dwell program')
    if (type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0
            or operations == ['dwell'] and duration < machine['simulation_process']['processing_time_sec']):
        raise ValueError('Machine dwell duration must satisfy the declared processing time')
    snapshot, geometry, models = deepcopy(physical['snapshot']), deepcopy(physical['geometry']), deepcopy(physical['resource_models'])
    before = deepcopy(physical.get('environment_valuation', resource_evidence['environment_valuation']))
    if any(field in snapshot['resources'].get(owner, {}) and
           recovery_validation_fingerprint(snapshot['resources'][owner][field]) != recovery_validation_fingerprint(value)
           for owner, values in before.items() for field, value in values.items()):
        raise ValueError('Physical and symbolic starting observations disagree')
    if physical['resources'][rid].get('model_static') is not True or geometry['resources'][rid].get('stationary_only') is not True:
        raise ValueError('Machine dwell requires recorded static owner geometry')
    partial = physical.get('outline_evidence_policy_version') == 2
    holds, unknown = {}, set()
    for peer, state in snapshot['resources'].items():
        row = physical['resources'][peer]
        if row.get('model_static') is True:
            continue
        if 'controller_hold_model' in row or row.get('idle') is not True:
            hold = commanded_hold_evidence(resource_id=peer, state=state,
                model=models.get(peer, {}) if partial else models[peer], physical=row)
            if hold.get('status') != 'supported':
                if partial:
                    unknown.add(peer)
                    continue
                raise ValueError('Machine dwell lacks supported peer hold: ' + peer + ': ' + hold.get('reason', ''))
            holds[peer] = hold
        elif row.get('stationary_contract', {}).get('requires_no_active_goals') is not True:
            if partial:
                unknown.add(peer)
                continue
            raise ValueError('Machine dwell lacks a stationary peer contract: ' + peer)
    held = {row.get('held_part') for row in snapshot['resources'].values()}
    for part, state in snapshot['parts'].items():
        if part in held:
            continue
        support = physical.get('stationary_part_evidence', {}).get(part, {})
        if (support.get('status') != 'supported' or support.get('current_pose') != state.get('current_pose')
                or not support.get('source')):
            if not partial:
                raise ValueError('Machine dwell requires matching stationary part evidence: ' + part)
    if partial:
        from cais_spade_llm.recovery_framework.outline_endpoint_evidence import _bounded_checkpoint

        first = _bounded_checkpoint(physical, index=0)
        interval = _bounded_checkpoint(physical, index=0, holds=holds, unknown=unknown)
    else:
        first = _checkpoint(snapshot, geometry, models, index=0)
        interval = _checkpoint(snapshot, geometry, models, index=0, peer_holds=holds)
    interval.update(phase='between', continuous_interval=['0', str(duration)],
        continuous_cells=[{'interval': ['0', str(duration)],
            'occupancy_possibilities': deepcopy(interval['occupancy_possibilities']),
            'part_occupancy_possibilities': deepcopy(interval['part_occupancy_possibilities'])}])
    after, products = declared['environment_valuation'], declared['environment_products']
    successor_snapshot = deepcopy(snapshot)
    for owner, values in after.items():
        if owner in successor_snapshot['resources']:
            successor_snapshot['resources'][owner].update(deepcopy(values))
    for name, values in products.items():
        if name in successor_snapshot['parts']:
            for field in ('state', 'location', 'processCompleted', 'target'):
                if field in values:
                    successor_snapshot['parts'][name][field] = deepcopy(values[field])
    last = (_bounded_checkpoint({**physical, 'snapshot': successor_snapshot}, index=duration,
                                holds=holds, unknown=unknown) if partial else
            _checkpoint(successor_snapshot, geometry, models, index=duration))
    for observation, valuation in ((first, before), (interval, before), (last, after)):
        observation['state_snapshot'] = deepcopy(valuation)
    contract = {'event': deepcopy(event), 'task': selected, 'duration_sec': duration,
                'physical_start_fingerprint': recovery_validation_fingerprint(physical),
                'stationary_resource_evidence': holds, 'kind': 'resource_owned_static_dwell'}
    if partial:
        contract['unknown_resources'] = sorted(unknown)
        for point in (first, interval, last):
            point['time_basis'] = 'owner_processing_duration_seconds'
    projected = _finish_environment_projection(task=task, recovery_snapshot=recovery_snapshot,
        recovery_des_model=recovery_des_model, resource_evidence=resource_evidence, physical=physical,
        declared=declared, geometry=geometry, models=models, observations=[first, interval, last],
        successor_snapshot=successor_snapshot, contract=contract)
    if completed_jobs is not None and projected.get('status') == 'projected':
        projected['successor_evidence'].setdefault('processing_observations', {})[rid] = completed_jobs
    if partial and projected.get('status') == 'projected':
        from cais_spade_llm.recovery_framework.outline_evidence_policy import invalidate_geometry

        successor = projected['successor_evidence']
        invalidate_geometry(successor, unknown, binding=projected['provenance']['binding'])
        projected['projected_snapshot'] = deepcopy(successor['snapshot'])
        projected['geometry'] = deepcopy(successor['geometry'])
        projected['resource_models'] = deepcopy(successor['resource_models'])
        projected['elapsed_time'] = {'status': 'calculated', 'seconds': duration,
                                    'source': 'machine owner processing_time_sec'}
    return projected


def _finish_environment_projection(*, task: dict, recovery_snapshot: dict, recovery_des_model: dict,
        resource_evidence: dict, physical: dict, declared: dict, geometry: dict, models: dict,
        observations: list[dict], successor_snapshot: dict, contract: dict) -> dict:
    """Commit only owner-supported effects with their complete interval ledgers."""
    from cais_spade_llm.resources.recovery_geometry import _part_entries, outline_event_binding

    _part_entries(observations)

    rid, selected = resource_evidence['resource_id'], declared['task']
    duration = observations[-1]['time']
    before = physical.get('environment_valuation', resource_evidence['environment_valuation'])
    after, products = declared['environment_valuation'], declared['environment_products']
    binding = outline_event_binding(task, recovery_snapshot, recovery_des_model, physical)
    task_id = recovery_validation_fingerprint(binding)
    part = selected['parameters'].get('part_name', task.get('part_name'))
    identity = {'task_id': task_id, 'resource_id': rid, 'process': selected['parameters'].get('process', 'any'),
                'product': part if part is not None else 'any', 'function': selected['event_name'], 'context': 'any'}
    updates = {field: deepcopy(value) for field, value in after[rid].items()
               if recovery_validation_fingerprint(value) != recovery_validation_fingerprint(before[rid][field])}
    part_updates = {field: deepcopy(products[part][key]) for field, key in
                    (('part_state', 'state'), ('part_location', 'location'), ('processCompleted', 'processCompleted'))
                    if part in products and key in products[part]}
    unresolved_fields = []
    for field, value in task['expected_end_state'].items():
        supported = updates.get(field, after[rid].get(field)) if field in recovery_des_model['state_variables'] and recovery_des_model['state_variables'][field].get('scope') != 'part' else part_updates.get(field)
        if field in after[rid] or field in part_updates:
            if recovery_validation_fingerprint(value) != recovery_validation_fingerprint(supported):
                reason = 'Declared owner completion contradicts proposed ' + field
                return {'status': 'rejected', 'validation_status': 'rejected', 'owner_effect_rejected': True,
                    'reason': reason, 'owner_effect_result': {'feasibility_status': 'INFEASIBLE',
                        'reason': reason, 'field': field, 'proposed': deepcopy(value), 'derived': deepcopy(supported)},
                    'observations': [], 'interval_complete': False, 'execution_authorized': False,
                    'command_sent': False, 'context_requests': []}
        else:
            unresolved_fields.append(field)
    if unresolved_fields:
        return {'status': 'needs_context', 'observations': [], 'interval_complete': False,
            'execution_authorized': False, 'command_sent': False,
            'deferred_valuation_fields': unresolved_fields,
            'unresolved': ['Owner completion does not establish proposed fields: ' + ', '.join(unresolved_fields)],
            'context_requests': [f'/resources/{task.get("resource_jid")}/outline_event_evidence']}
    provenance = {'kind': 'projected', 'source': contract['kind'],
                  'binding': binding, 'contract_fingerprint': recovery_validation_fingerprint(contract)}
    successor = deepcopy(physical)
    successor.update(snapshot=successor_snapshot, geometry=deepcopy(geometry), resource_models=deepcopy(models),
                     environment_valuation=deepcopy(after), products=deepcopy(products), provenance=provenance)
    ledger_models = deepcopy(resource_evidence['environment_models'])
    initial = [{'resource_id': owner, 'process': 'any', 'product': 'any', 'context': 'any', 'values': deepcopy(values)}
               for owner, values in before.items()]
    declared_event = next(row for row in ledger_models[rid]['events'] if row['event_id'] == selected['event_id'])
    participants = declared_event['participants']
    participant_updates = {owner: {field: deepcopy(value) for field, value in values.items()
        if recovery_validation_fingerprint(value) != recovery_validation_fingerprint(before[owner][field])}
        for owner, values in after.items()}
    changed_owners = {owner for owner, values in participant_updates.items() if values}
    if not changed_owners.issubset(participants):
        raise ValueError('Projected effects changed an undeclared participant')
    task_rows, state_rows = [], []
    for participant in participants:
        local_identity = {**identity, 'resource_id': participant,
            'task_id': recovery_validation_fingerprint({'task_id': task_id, 'participant': participant})}
        task_rows.append({**local_identity, 'start_time': 0, 'end_time': duration,
                          'declared_task': deepcopy(selected)})
        if participant_updates[participant]:
            update = {key: local_identity[key] for key in ('task_id', 'resource_id', 'process', 'product', 'context')}
            state_rows.append({**update, 'time': duration, 'values': participant_updates[participant]})
    ledgers = {'resource_models': ledger_models, 'interval_complete': True,
        'task_evidence': {'complete': True, 'source_kind': 'resource_owned_projection', 'horizon': [0, duration],
                          'events': task_rows},
        'state_evidence': {'complete': True, 'source_kind': 'resource_owned_projection', 'horizon': [0, duration],
                           'initial': initial, 'updates': state_rows}}
    if 'resource_configurations' in contract:
        ledgers['resource_configurations'] = deepcopy(contract['resource_configurations'])
    if rid == 'Conveyor' and contract['kind'] == 'resource_owned_shared_transport':
        from cais_spade_llm.resources.environment_models import owners

        previous_products = physical.get('products', resource_evidence['environment_products'])
        custody_before = owners(ledger_models, before, previous_products, resource_evidence['product_name'])
        custody_after = owners(ledger_models, after, products, resource_evidence['product_name'])
        transfers = [part for part, owner in custody_before.items() if custody_after.get(part) != owner]
        if transfers != ([selected['parameters']['delivered_part']] if selected['parameters'].get('delivered_part') else []):
            raise ValueError('Conveyor physical projection disagrees with its declared hand-off identity')
        inventory_before = {owner: [part for part, holder in custody_before.items() if holder == owner] for owner in before}
        inventory_after = {owner: [part for part, holder in custody_after.items() if holder == owner] for owner in after}
        for observation in observations:
            for participant in participants:
                record = observation['owner_safety_evidence']['resource_evidence'][participant]
                record['handoff_declarations'] = deepcopy(ledger_models[participant]['events'])
                records = []
                for part in transfers:
                    task_row = next(row for row in task_rows if row['resource_id'] == participant)
                    completed = observation['time'] == duration
                    handoff = {'task_id': task_row['task_id'], 'part_name': part,
                        'source_resource': custody_before[part], 'destination_resource': custody_after[part],
                        'custody_before': deepcopy(inventory_before),
                        'custody_after': deepcopy(inventory_after if completed else inventory_before),
                        'phase': 'completed' if completed else 'pending'}
                    if completed:
                        handoff['acknowledgement'] = {'kind': 'projected', 'source_fingerprint': recovery_validation_fingerprint(contract),
                            'task_id': task_row['task_id'], 'part_name': part,
                            'source_resource': custody_before[part], 'destination_resource': custody_after[part],
                            'source_departed': True, 'destination_arrived': True,
                            'owner_effect_fingerprint': recovery_validation_fingerprint(
                                [row for row in state_rows if row['task_id'] == task_row['task_id']])}
                    records.append(handoff)
                record['handoffs'] = {'complete': True, 'records': records}
    result = {'status': 'projected', 'observations': observations, 'geometry': geometry,
        'resource_models': models, 'owner_safety_evidence': ledgers, 'projected_snapshot': successor_snapshot,
        'successor_evidence': successor, 'resource_updates': updates, 'part_updates': part_updates,
        'projected_resource_snapshot': {**deepcopy(recovery_snapshot),
            **deepcopy(successor_snapshot['resources'][rid]), **deepcopy(after[rid])},
        'resource_valuation_updates': {owner: values for owner, values in participant_updates.items() if values},
        'deferred_valuation_fields': [], 'deferred_part_fields': [],
        'interval_complete': True, 'provenance': provenance, 'owner_interval_contract': contract,
        'execution_authorized': False, 'command_sent': False, 'unresolved': []}

    from cais_spade_llm.agents.central_controller.outline_evidence_safety import bind_outline_owner_safety_evidence
    from cais_spade_llm.agents.central_controller.recovery_case_evidence import bind_case_projection_evidence

    try:
        result = bind_case_projection_evidence(result, physical_evidence=physical,
            scene_configuration=resource_evidence['inputs']['scene'])
        return bind_outline_owner_safety_evidence(result)
    except ValueError as exc:
        result['owner_safety_evidence_unresolved'] = str(exc)
        return result


def _project_transport(*, task: dict, recovery_snapshot: dict, recovery_des_model: dict,
                       resource_evidence: dict, physical: dict, declared: dict) -> dict:
    """Bound the owner's declared shared X displacement over its saved scene.

    This calls the same displacement and acceleration calculations as the
    simulation owner. It does not infer a motion sequence from primitive support.
    """
    from cais_spade_llm.recovery_framework.workflow_gazebo import _motion_duration, _motion_distance, _shared_displacement
    from cais_spade_llm.resources.continuous_motion import occupancy, quaternion_transform, transformed_box
    from cais_spade_llm.resources.recovery_geometry import (
        _checkpoint, commanded_hold_evidence, projected_part_support,
    )

    rid, selected = resource_evidence['resource_id'], declared['task']
    event = next(row for row in resource_evidence['environment_models'][rid]['events']
                 if row['event_id'] == selected['event_id'])
    if [step['op'] for step in event['program']['steps']] != ['move_relative']:
        raise ValueError('Recorded transport program has no supported single shared displacement')
    scene = resource_evidence['inputs']['scene']
    snapshot, geometry, models = deepcopy(physical['snapshot']), deepcopy(physical['geometry']), deepcopy(physical['resource_models'])
    before = physical.get('environment_valuation', resource_evidence['environment_valuation'])
    if set(snapshot['resources']) != set(resource_evidence['environment_models']):
        raise ValueError('Transport projection requires the complete peer scene')
    if any(field in snapshot['resources'][owner] and recovery_validation_fingerprint(snapshot['resources'][owner][field])
           != recovery_validation_fingerprint(value) for owner, values in before.items() for field, value in values.items()):
        raise ValueError('Transport physical and symbolic starting states disagree')
    if physical['resources'][rid].get('model_static') is not True:
        raise ValueError('Transport owner requires its recorded static frame geometry')
    holds = {}
    for peer, state in snapshot['resources'].items():
        row = physical['resources'][peer]
        if row.get('model_static') is True:
            continue
        if 'controller_hold_model' in row or row.get('idle') is not True:
            hold = commanded_hold_evidence(resource_id=peer, state=state, model=models[peer], physical=row)
            if hold.get('status') != 'supported':
                raise ValueError('Transport lacks supported peer hold: ' + peer)
            holds[peer] = hold
        elif row.get('stationary_contract', {}).get('requires_no_active_goals') is not True:
            raise ValueError('Transport lacks a stationary peer contract: ' + peer)
    params = selected['parameters']
    if selected['event_name'] == 'advance_conveyor':
        names = [field.removeprefix('part_location.') for field, value in before[rid].items()
                 if field.startswith('part_location.') and value is not None]
        targets = deepcopy(params['next_locations'])
        if params.get('delivered_part') is not None:
            targets[params['delivered_part']] = 'Buffer For Machined parts'
        if set(targets) != set(names) or not names:
            raise ValueError('Shared Conveyor displacement requires every recorded resident part')
        initial_x = {name: snapshot['parts'][name]['current_pose'][0] for name in names}
        displacement = _shared_displacement(scene, initial_x, targets)
    else:
        names = [params['part_name']]
        initial_x = {name: snapshot['parts'][name]['current_pose'][0] for name in names}
        target = scene[rid]['slot_poses'][int(params['downstream_zone']) - 1]
        displacement = target[0] - initial_x[names[0]]
    if type(displacement) not in (int, float) or not math.isfinite(displacement) or displacement <= 0:
        raise ValueError('Declared downstream transport requires a positive finite displacement')
    if any(snapshot['resources'][peer].get('held_part') in names for peer in snapshot['resources']):
        raise ValueError('A held part cannot be moved by the transport owner')
    held = {row.get('held_part') for row in snapshot['resources'].values()}
    for name, state in snapshot['parts'].items():
        if name in held:
            continue
        support = physical.get('stationary_part_evidence', {}).get(name, {})
        if (support.get('status') != 'supported' or support.get('current_pose') != state.get('current_pose')
                or not support.get('source')):
            raise ValueError('Transport lacks matching initial support: ' + name)
    profile = scene[rid]['simulation_transport']
    speed = event['program']['steps'][0]['params'].get('speed_mps')
    if speed is None:
        speed = profile['speed_mps']
    requested_speed = task.get('params', {}).get('speed_mps')
    if requested_speed is not None and recovery_validation_fingerprint(requested_speed) != recovery_validation_fingerprint(speed):
        raise ValueError('Requested speed differs from the declared owner transport program')
    acceleration = profile['acceleration_mps2']
    if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in (speed, acceleration))
            or speed > profile['speed_mps']):
        raise ValueError('Transport speed and acceleration must satisfy the captured declaration')
    duration = _motion_duration(displacement, speed, acceleration)
    first = _checkpoint(snapshot, geometry, models, index=0)
    total = deepcopy(snapshot)
    for name in names:
        pose = total['parts'][name]['current_pose']
        if (not isinstance(pose, list) or len(pose) != 7
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in pose)):
            raise ValueError('Transport needs each recorded xyz/xyzw part pose')
        pose[0] = initial_x[name] + displacement
    after, products = declared['environment_valuation'], declared['environment_products']
    for peer, values in after.items():
        total['resources'][peer].update(deepcopy(values))
    for name, values in products.items():
        if name in total['parts']:
            for field in ('state', 'location', 'processCompleted', 'target'):
                if field in values:
                    total['parts'][name][field] = deepcopy(values[field])
    from cais_spade_llm.resources.environment_models import owners

    custody = owners(resource_evidence['environment_models'], after, products, resource_evidence['product_name'])
    for name in names:
        total['parts'][name]['contained_by'] = custody[name]
    for peer, state in total['resources'].items():
        if 'contained_parts' in state:
            state['contained_parts'] = [name for name, owner in custody.items()
                                       if owner == peer and state.get('held_part') != name]
    support = {}
    for name in names:
        supported = projected_part_support(part_name=name, state=total, evidence=physical)
        if supported.get('status') != 'supported':
            raise ValueError('Transport endpoint lacks supported placement for ' + name + ': ' + supported.get('reason', ''))
        supported['current_pose'] = deepcopy(total['parts'][name]['current_pose'])
        support[name] = supported
    interval = _checkpoint(snapshot, geometry, models, index=0, peer_holds=holds)
    interval.update(phase='between', continuous_interval=['0', str(duration)])
    def boxes_between(start, end):
        low = max(0., math.nextafter(_motion_distance(start, displacement, speed, acceleration), -math.inf))
        high = min(displacement, math.nextafter(_motion_distance(end, displacement, speed, acceleration), math.inf))
        result = {}
        for name in names:
            pose = snapshot['parts'][name]['current_pose']
            matrix = quaternion_transform(pose[:3], pose[3:])
            matrix[0][3] = (math.nextafter(pose[0] + low, -math.inf), math.nextafter(pose[0] + high, math.inf))
            result[name] = transformed_box(matrix, geometry['parts'][name]['local_bounds'], name)
        return result
    # A fixed bound is complete over the whole interval. Unknown region crossings
    # remain explicit until bounded refinement separates their time cells.
    cells = []
    pending = [(0., duration, 0)]
    while pending:
        start, end, depth = pending.pop()
        boxes = boxes_between(start, end)
        values = deepcopy(interval['part_occupancy_possibilities'])
        unknown = False
        for region, shape in geometry['regions'].items():
            for name, box in boxes.items():
                values[region][name] = occupancy([box], shape['bounds'])
                unknown |= len(values[region][name]) > 1
        if unknown and depth < 4 and len(cells) + len(pending) < 15:
            middle = (start + end) / 2
            pending.extend([(middle, end, depth + 1), (start, middle, depth + 1)])
            continue
        cells.append({'interval': [str(start), str(end)],
                      'occupancy_possibilities': deepcopy(interval['occupancy_possibilities']),
                      'part_occupancy_possibilities': values,
                      'part_motion_bounds': boxes})
    cells.sort(key=lambda row: float(row['interval'][0]))
    interval['continuous_cells'] = cells
    for region in geometry['regions']:
        for name in names:
            values = sorted({value for cell in cells for value in cell['part_occupancy_possibilities'][region][name]})
            interval['part_occupancy_possibilities'][region][name] = values
            interval['part_region_occupancy'][region][name] = values[0] if len(values) == 1 else None
    last = _checkpoint(total, geometry, models, index=duration)
    observations = [first, interval, last]
    for observation, valuation in ((first, before), (interval, before), (last, after)):
        observation['state_snapshot'] = deepcopy(valuation)
    contract = {'kind': 'resource_owned_shared_transport', 'event': deepcopy(event), 'task': selected,
                'duration_sec': duration, 'displacement_m': displacement, 'speed_mps': speed,
                'acceleration_mps2': acceleration, 'part_names': names,
                'stationary_resource_evidence': holds, 'projected_part_support': support,
                'physical_start_fingerprint': recovery_validation_fingerprint(physical),
                'future_execution_tracking': 'not_established',
                'model_assumptions': ['The declared simulation transport follows its controlled part-pose profile',
                                     'Peer owners retain their supported stationary states']}
    if rid == 'Conveyor':
        contract['resource_configurations'] = {participant: deepcopy(scene[participant])
            for participant in event['participants']}
        source = {'kind': 'projected', 'fingerprint': recovery_validation_fingerprint(contract)}
        for observation in observations:
            records = observation.setdefault('owner_safety_evidence', {}).setdefault('resource_evidence', {})
            for participant in event['participants']:
                records[participant] = {
                    'source': deepcopy(source), 'configuration': deepcopy(scene[participant]),
                    'interval': deepcopy(observation['continuous_interval']) if observation.get('phase') == 'between'
                                else [observation['time_exact'], observation['time_exact']]}
            records[rid]['transport_motion'] = {'source': deepcopy(source), 'interval': [0, duration],
                'displacement_m': displacement, 'speed_mps': speed,
                'acceleration_mps2': acceleration, 'duration_sec': duration}
    result = _finish_environment_projection(task=task, recovery_snapshot=recovery_snapshot,
        recovery_des_model=recovery_des_model, resource_evidence=resource_evidence, physical=physical,
        declared=declared, geometry=geometry, models=models, observations=observations,
        successor_snapshot=total, contract=contract)
    if result.get('status') == 'projected':
        result['successor_evidence'].setdefault('stationary_part_evidence', {}).update(support)
    return result

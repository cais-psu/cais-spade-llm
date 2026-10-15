from __future__ import annotations

"""CCA checks of projected recovery geometry with unchanged physical AP meanings."""

from copy import deepcopy
from fractions import Fraction
from typing import Any

from cais_spade_llm.agents.central_controller._recovery_monitor_history import continuous_transition
from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
from cais_spade_llm.agents.central_controller.local_composition import AnalysisLimit, Budget
from cais_spade_llm.agents.central_controller.offline_safety_grounding import (
    _ap_binding, _horizon, _ledger, _rules, _states, _state_value, _structured_value, _tasks, _valuations,
)
from cais_spade_llm.agents.central_controller.ppr_ap import physical_ap_kind
from cais_spade_llm.agents.central_controller.recovery_case_evidence import (
    CASE_AP_MEANINGS, transport_belt_stopped,
)
from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
    _catalog,
    _digest,
    _valuation,
)


def _physical_state(observation: dict) -> dict:
    return {key: deepcopy(observation[key]) for key in ("resources", "parts", "frame")
            if key in observation}


def _owner_checkpoint(observation: dict) -> dict:
    result = {**_physical_state(observation), "state_snapshot": deepcopy(observation.get("state_snapshot"))}
    if "owner_completion_applied" in observation:
        result["owner_completion_applied"] = observation["owner_completion_applied"]
    return result


def _physical_state_updates(observation: dict, models: dict, parts: dict) -> dict:
    record = observation["owner_safety_evidence"]
    updates, sources = record.get("physical_state_updates", {}), record.get("physical_state_sources", {})
    if not isinstance(updates, dict) or not isinstance(sources, dict) or set(updates) != set(sources):
        raise ValueError("Physical custody fields require complete owner step sources")
    contract = record.get("owner_interval_contract", {})
    for rid, values in updates.items():
        source, task = sources[rid], contract.get("declared_task", {})
        steps = contract.get("owner_declaration", {}).get("program", {}).get("steps", [])
        if "owner_declarations" in contract:
            declarations = contract["owner_declarations"]
            if (type(contract.get("projection_clock_version")) is not int
                    or contract["projection_clock_version"] != 1 or not isinstance(declarations, list)
                    or not declarations or not contract.get("source_contract_fingerprint")):
                raise ValueError("Physical custody fields lack the exact owner event declarations")
            by_fingerprint = {}
            for declaration in declarations:
                for step in declaration.get("composition_contract", {}).get("steps", []):
                    by_fingerprint[_digest(step)] = step
            steps = list(by_fingerprint.values())
        matches = [step for step in steps if _digest(step) == source.get("step_fingerprint")
                   and step.get("op") == source.get("op")]
        empty_home = (len(matches) == 1 and source.get("op") == "move_to_named_pose"
                      and matches[0].get("params", {}).get("pose_name") == "home")
        if (task.get("resource_id") != rid or len(matches) != 1
                or not contract.get("preview_fingerprint")
                or source.get("source_fingerprint") != contract["preview_fingerprint"]
                or (source["op"] not in {"grasp_part", "release_part", "open_gripper", "close_gripper"} and not empty_home)
                or not isinstance(values, dict) or not values or set(values)-{"held_part", "gripper_state"}):
            raise ValueError("Physical custody fields differ from the validated owner program")
        op = source["op"]
        if "held_part" in values:
            if op == "grasp_part":
                expected = task.get("parameters", {}).get("part_name")
                if not isinstance(expected, str) or expected not in parts or values["held_part"] != expected:
                    raise ValueError("Physical grasp differs from the declared part identity")
            elif op == "release_part":
                if values["held_part"] is not None:
                    raise ValueError("Physical release cannot retain held_part")
            else:
                raise ValueError("A gripper-only step cannot establish changed custody")
        expected_gripper = "closed" if op in {"grasp_part", "close_gripper"} else "open"
        if "gripper_state" in values and values["gripper_state"] != expected_gripper:
            raise ValueError("Physical gripper state differs from its declared operation")
        for field, value in values.items():
            _state_value(value, models[rid]["state_variables"][field], parts)
            if field not in observation["resources"][rid] or _digest(observation["resources"][rid][field]) != _digest(value):
                raise ValueError("Physical custody source contradicts the exact checkpoint")
    return updates


def _owner_valuation_ledgers(observation: dict, models: dict, parts: dict) -> tuple:
    tasks, initial, updates, task_ledger, state_ledger = _owner_ledgers(observation, models, parts)
    current = deepcopy(initial)
    for update in updates:
        if (Fraction(str(update["time"])) <= Fraction(observation["time_exact"])
                and observation.get("owner_completion_applied") is not False):
            rid = update["resource_id"]
            current[rid] = {**update, "values": {**current[rid]["values"], **deepcopy(update["values"])}}
    for rid, values in _physical_state_updates(observation, models, parts).items():
        current[rid]["values"].update(deepcopy(values))
    if observation.get("owner_completion_applied") is False:
        tasks = {key: row for key, row in tasks.items() if row["start_time"] != row["end_time"]}
    return tasks, current, [], task_ledger, state_ledger


def _transport_values(binding: dict, observation: dict, ledgers: tuple, *, interval=None):
    if binding.get("state_field") != "belt_stopped":
        return None
    stopped = transport_belt_stopped(binding["resource_id"], observation, interval=interval)
    if stopped is None:
        return None
    tasks, current, _, task_ledger, state_ledger = ledgers
    values = []
    for value in stopped:
        state = deepcopy(current)
        state[binding["resource_id"]]["values"]["belt_stopped"] = value
        supported, _ = _structured_value(binding, Fraction(observation["time_exact"]),
            tasks, state, task_ledger, state_ledger)
        values.append(supported)
    return sorted(set(values))


def _owner_ledgers(observation: dict, models: dict, parts: dict) -> tuple:
    record = observation.get("owner_safety_evidence")
    if not isinstance(record, dict) or record.get("interval_complete") is not True:
        raise ValueError("Structured APs require complete resource-owned interval evidence")
    declared = {rid: {"state_variables": model.get("state_variables", {})} for rid, model in models.items()}
    if record.get("resource_models_fingerprint") != _digest(declared):
        raise ValueError("Resource-state evidence declarations changed")
    if record.get("observation_fingerprint") != _digest(_owner_checkpoint(observation)):
        raise ValueError("Resource-state evidence is bound to another physical checkpoint")
    instantaneous = record.get("owner_interval_contract", {}).get("kind") == "resource_owned_simulation_job_control"
    horizon = _horizon(record.get("horizon"), "owner_safety_evidence.horizon", instantaneous=instantaneous)
    if not instantaneous and "owner_completion_applied" in observation:
        raise ValueError("Only declared instantaneous owner controls may separate completion at the same time")
    if instantaneous and horizon[0] != horizon[1]:
        raise ValueError("Instantaneous owner control cannot establish a nonzero motion interval")
    if horizon[0] == horizon[1] and type(observation.get("owner_completion_applied")) is not bool:
        raise ValueError("Instantaneous owner evidence requires explicit before/after completion checkpoints")
    if not horizon[0] <= Fraction(observation["time_exact"]) <= horizon[1]:
        raise ValueError("Resource-state evidence omits the physical checkpoint time")
    if not isinstance(record.get("task_evidence"), dict) or not isinstance(record.get("state_evidence"), dict):
        raise ValueError("Task and state ledgers must explicitly establish complete evidence")
    task_ledger = _ledger(record["task_evidence"], "task_evidence", horizon, {"events"}, instantaneous=instantaneous)
    state_ledger = _ledger(record["state_evidence"], "state_evidence", horizon, {"initial", "updates"}, instantaneous=instantaneous)
    tasks, _ = _tasks(task_ledger, models, horizon, instantaneous=instantaneous)
    initial, updates, _ = _states(state_ledger, tasks, models, horizon, parts)
    if set(initial) != set(models):
        raise ValueError("Resource-state evidence omits part of the declared resource population")
    if observation.get("phase") == "between":
        start, end = map(Fraction, observation["continuous_interval"])
        boundaries = [Fraction(str(row[field])) for row in tasks.values()
                      for field in ("start_time", "end_time")]
        boundaries.extend(Fraction(str(row["time"])) for row in updates)
        if any(start < boundary < end for boundary in boundaries):
            raise ValueError("Physical intervals must split at every owner task/state boundary")
    current = {rid: deepcopy(row["values"]) for rid, row in initial.items()}
    for update in updates:
        if (Fraction(str(update["time"])) <= Fraction(observation["time_exact"])
                and observation.get("owner_completion_applied") is not False):
            current[update["resource_id"]].update(deepcopy(update["values"]))
    for rid, values in _physical_state_updates(observation, models, parts).items():
        current[rid].update(deepcopy(values))
    snapshot = observation.get("state_snapshot")
    if not isinstance(snapshot, dict) or _digest(current) != _digest(snapshot):
        raise ValueError("Resource-state ledger disagrees with the exact observation state_snapshot")
    for rid, values in snapshot.items():
        # Symbolic owner declarations remain discoverable even when an AP does
        # not require their geometry. Participant coverage is checked by _rules.
        physical = observation["resources"].get(rid, {})
        if any(field in physical and _digest(physical[field]) != _digest(value)
               for field, value in values.items()):
            raise ValueError("Resource-state ledger conflicts with the physical owner checkpoint")
    return tasks, initial, updates, task_ledger, state_ledger


def _owner_state_declarations(models: dict, parts: dict) -> dict:
    """Expand owner declarations against the recorded part population."""
    declared = {}
    for rid, model in models.items():
        fields = {}
        for field, declaration in model.get("state_variables", {}).items():
            names = ([field.replace("{part_name}", part) for part in parts]
                     if "{part_name}" in field else [field])
            for name in names:
                if name in fields and _digest(fields[name]) != _digest(declaration):
                    raise ValueError("Parameterized owner state declarations conflict")
                fields[name] = deepcopy(declaration)
        declared[rid] = {"state_variables": fields}
    return declared


def bind_outline_owner_safety_evidence(projection: dict) -> dict:
    """Bind complete owner ledgers to each unchanged physical observation.

    This records an owner calculation, not an execution observation. The owner
    must already have established every interval represented by the ledgers.
    """
    result = deepcopy(projection)
    evidence = result.get("owner_safety_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("Resource-owned safety ledgers are unavailable")
    models = evidence.get("resource_models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Resource-owned safety declarations are unavailable")
    declared = _owner_state_declarations(models, result["geometry"]["parts"])
    event_fingerprints = deepcopy(evidence.get("resource_event_fingerprints", {}))
    for rid, model in models.items():
        if "events" in model:
            event_fingerprints[rid] = _digest(model["events"])
    observations = result["observations"]
    if not observations or not set(observations[0]["resources"]).issubset(declared):
        raise ValueError("Resource-state declarations omit the complete physical population")
    if not isinstance(evidence.get("task_evidence"), dict):
        raise ValueError("Complete resource-owned task evidence is unavailable")
    horizon = deepcopy(evidence["task_evidence"].get("horizon"))
    instantaneous = result.get("owner_interval_contract", {}).get("kind") == "resource_owned_simulation_job_control"
    if instantaneous and [row.get("owner_completion_applied") for row in observations] != [False, True]:
        raise ValueError("Instantaneous owner evidence requires one before and one after checkpoint")
    if _horizon(horizon, "owner horizon", instantaneous=instantaneous) != (Fraction(observations[0]["time_exact"]),
                                               Fraction(observations[-1]["time_exact"])):
        raise ValueError("Owner ledger horizon differs from the exact projected horizon")
    record = {key: deepcopy(evidence.get(key)) for key in
              ("task_evidence", "state_evidence", "interval_complete", "resource_configurations", "collision_object_ids")}
    record.update(horizon=horizon, resource_models_fingerprint=_digest(declared))
    record["resource_kinematic_model_fingerprints"] = {
        rid: _digest(row["configuration"]) for rid, row in result.get("resource_models", {}).items()
        if isinstance(row.get("configuration"), dict) and row["configuration"].get("joint_names")}
    record["resource_event_fingerprints"] = event_fingerprints
    if "owner_interval_contract" in result:
        record["owner_interval_contract"] = deepcopy(result["owner_interval_contract"])
    for observation in observations:
        supplied = observation.get("owner_safety_evidence", {})
        raw = deepcopy(supplied.get("resource_evidence", {}))
        local = {key: deepcopy(supplied[key]) for key in
                 ("collision_object_ids", "physical_state_updates", "physical_state_sources") if key in supplied}
        observation["owner_safety_evidence"] = {
            **deepcopy(record), **local, "resource_evidence": raw}
        for rid, values in _physical_state_updates(observation, declared, result["geometry"]["parts"]).items():
            observation["state_snapshot"][rid].update(deepcopy(values))
        observation["owner_safety_evidence"]["observation_fingerprint"] = _digest(_owner_checkpoint(observation))
        _owner_ledgers(observation, declared, result["geometry"]["parts"])
    result["owner_safety_evidence"]["resource_models"] = declared
    result["owner_safety_evidence"]["resource_event_fingerprints"] = event_fingerprints
    return result


def bind_validated_outline_projection(projection: dict, *, task: dict,
                                      physical_evidence: dict, scene_configuration: dict | None = None) -> dict:
    """Retain owner ledgers only when every symbolic boundary is established.

    The caller is the resource owner after independent physical projection.
    Intermediate state/custody changes require their explicit task effects;
    they are never flattened into a guessed final-only state transition here.
    """
    if projection.get("status") != "projected" or projection.get("interval_complete") is not True:
        raise ValueError("Owner ledgers require an independently supported physical interval")
    if "owner_safety_evidence" in projection:
        if scene_configuration is not None:
            from .recovery_case_evidence import bind_case_projection_evidence
            projection = bind_case_projection_evidence(projection, physical_evidence=physical_evidence,
                                                       scene_configuration=scene_configuration)
        return bind_outline_owner_safety_evidence(projection)
    models = physical_evidence.get("environment_models")
    before = physical_evidence.get("environment_valuation")
    after = projection.get("successor_evidence", {}).get("environment_valuation")
    if not isinstance(models, dict) or not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("Complete recorded owner declarations and endpoint valuation are unavailable")
    observations = projection["observations"]
    if not observations or set(models) != set(before) or set(models) != set(after):
        raise ValueError("Owner declaration and valuation populations differ")
    first, last = Fraction(observations[0]["time_exact"]), Fraction(observations[-1]["time_exact"])
    if first >= last:
        raise ValueError("Owner task ledger requires a positive recorded duration")
    # The task ledger's existing numeric schema cannot represent arbitrary
    # rational clocks exactly; decline instead of rounding the interval.
    horizon = [float(first), float(last)]
    if [Fraction(str(value)) for value in horizon] != [first, last]:
        raise ValueError("Owner ledger numeric clock cannot represent the exact physical horizon")
    result = deepcopy(projection)
    for index, observation in enumerate(result["observations"]):
        endpoint = (index == len(observations) - 1 or
                    (type(result.get("owner_interval_contract", {}).get("projection_clock_version")) is int
                     and result["owner_interval_contract"]["projection_clock_version"] == 1
                     and observation.get("phase") != "between"
                     and Fraction(observation["time_exact"]) == last))
        expected = deepcopy(after if endpoint else before)
        envelope = observation.setdefault("owner_safety_evidence", {})
        envelope["owner_interval_contract"] = deepcopy(result.get("owner_interval_contract", {}))
        for rid, values in _physical_state_updates(observation, models, result["geometry"]["parts"]).items():
            expected[rid].update(deepcopy(values))
        values = {}
        for rid, fields in expected.items():
            physical = observation["resources"][rid]
            if any(field not in physical for field in fields):
                raise ValueError("Physical checkpoint omits declared owner state: " + rid)
            values[rid] = {field: deepcopy(physical[field]) for field in fields}
        if _digest(values) != _digest(expected):
            raise ValueError("Intermediate owner state changes need their complete task-effect ledger")
        observation["state_snapshot"] = values
    binding = projection.get("provenance", {}).get("binding")
    if not isinstance(binding, dict) or not projection.get("provenance", {}).get("contract_fingerprint"):
        raise ValueError("Owner interval evidence lacks its checked task/contract binding")
    jid = task.get("resource_jid")
    if binding.get("resource_jid") != jid:
        raise ValueError("Owner interval evidence is bound to another task resource")
    jids = physical_evidence.get("jids", {})
    owners = [rid for rid in models if jids.get(rid) == jid]
    if len(owners) != 1:
        raise ValueError("Owner task resource association is missing or ambiguous")
    rid = owners[0]
    if any(physical_evidence.get("resources", {}).get(peer, {}).get("idle") is not True
           for peer in models if peer != rid):
        raise ValueError("Complete peer task absence is unavailable for the projected interval")
    changed_peers = [peer for peer in models if peer != rid and _digest(before[peer]) != _digest(after[peer])]
    if changed_peers or projection.get("product_custody_evidence"):
        from cais_spade_llm.resources.recovery_geometry import _owner_product_custody

        _, calculated, witness = _owner_product_custody(task=task,
            state=projection.get("projected_snapshot", {}), evidence=physical_evidence,
            resource_updates=projection.get("resource_updates", {}), rid=rid,
            support=projection.get("successor_evidence", {}).get("stationary_part_evidence", {}))
        if (not witness or _digest(witness) != _digest(projection.get("product_custody_evidence"))
                or any(_digest(calculated[peer]) != _digest(after[peer]) for peer in models if peer != rid)):
            raise ValueError("Cross-owner effects need their complete independently derived custody ledger")
    function = task.get("event_name")
    if not isinstance(function, str) or not function:
        raise ValueError("Owner task identity is missing")
    task_id = _digest({"binding": binding, "task": task})
    identity = {"resource_id": rid, "process": "*", "product": "*", "context": "any"}
    events, updates = [], []
    for owner in [rid, *changed_peers]:
        owner_task_id = task_id if owner == rid else _digest({"binding":binding,"task":task,"resource_id":owner})
        owner_identity = {**identity,"resource_id":owner,"task_id":owner_task_id}
        events.append({**owner_identity,"function":function,"start_time":horizon[0],"end_time":horizon[1]})
        changed = {field:deepcopy(value) for field,value in after[owner].items()
                   if field not in before[owner] or _digest(value) != _digest(before[owner][field])}
        if changed:
            updates.append({**owner_identity,"time":horizon[1],"values":changed})
    result["owner_safety_evidence"] = {
        "resource_models": deepcopy(models), "interval_complete": True,
        "task_evidence": {"complete": True, "source_kind": "resource_owned_projection",
            "horizon": horizon, "events": events},
        "state_evidence": {"complete": True, "source_kind": "resource_owned_projection",
            "horizon": horizon,
            "initial": [{"resource_id": peer, "process": "*", "product": "*", "context": "any",
                         "values": deepcopy(values)} for peer, values in before.items()],
            "updates": updates}}
    if scene_configuration is not None:
        from .recovery_case_evidence import bind_case_projection_evidence
        result = bind_case_projection_evidence(result, physical_evidence=physical_evidence,
                                              scene_configuration=scene_configuration)
    return bind_outline_owner_safety_evidence(result)


def _replay(checker: BaseSafetyChecker, rules: list, observations: list,
            states: list, budget: Budget, models: dict | None = None,
            parts: dict | None = None) -> tuple[list, list, dict | None, dict | None]:
    if any(physical_ap_kind(ap) == "loading_reserved_by" for rule in rules for ap in rule["aps"]):
        for observation in observations:
            _owner_ledgers(observation, models or {}, parts or {})
    physical = [{**rule, "aps": [ap for ap in rule["aps"] if physical_ap_kind(ap) is not None]}
                for rule in rules]
    valuations, _ = _valuations(physical, observations, {}, {}, [], None, None)
    structured = [{**rule, "aps": [ap for ap in rule["aps"] if physical_ap_kind(ap) is None]}
                  for rule in rules]
    if any(rule["aps"] for rule in structured):
        for observation, values in zip(observations, valuations, strict=True):
            ledgers = _owner_valuation_ledgers(observation, models or {}, parts or {})
            owner_values, _ = _valuations(structured, [observation], *ledgers)
            for rid, row in owner_values[0].items():
                values[rid].update(row)
            for rule in structured:
                for ap in rule["aps"]:
                    binding = rule["ap_groundings"][ap["label"]]
                    stopped = _transport_values(binding, observation, ledgers)
                    if stopped is not None:
                        values[rule["rule_id"]][ap["label"]] = stopped
    checks = []
    first_uncertainty = None
    for index, (observation, values) in enumerate(zip(observations, valuations, strict=True)):
        cells = observation.get("continuous_cells")
        if cells is not None:
            bounds = observation.get("continuous_interval")
            if (observation.get("phase") != "between" or not isinstance(cells, list) or not 1 <= len(cells) <= 16
                    or not isinstance(bounds, list) or len(bounds) != 2):
                raise ValueError("Continuous outline cells require a bounded complete motion interval")
            start, end = map(Fraction, bounds)
            previous = start
            observation["rule_cells"] = []
            for cell in cells:
                a, b = map(Fraction, cell["interval"])
                if a != previous or not a < b <= end:
                    raise ValueError("Continuous outline cells do not cover the exact motion interval")
                previous = b
                cell_values = deepcopy(values)
                for rule in rules:
                    for ap in rule["aps"]:
                        kind = physical_ap_kind(ap)
                        binding = _ap_binding(rule, ap)
                        if kind == "resource_region":
                            cell_values[rule["rule_id"]][ap["label"]] = deepcopy(
                                cell["occupancy_possibilities"][binding["region"]][binding["resource"]])
                        elif kind == "part_region_entry":
                            cell_values[rule["rule_id"]][ap["label"]] = deepcopy(
                                cell["part_entry_possibilities"][binding["region"]][binding["part"]])
                        elif kind == "capacity_exceeded":
                            cell_values[rule["rule_id"]][ap["label"]] = _valuation(
                                {**rule, "aps": [ap]}, {**observation, **cell}, None)[ap["label"]]
                        elif kind is None:
                            owner_binding = rule["ap_groundings"][ap["label"]]
                            cell_ledgers = _owner_valuation_ledgers(observation, models or {}, parts or {})
                            stopped = _transport_values(owner_binding, observation, cell_ledgers, interval=cell["interval"])
                            if stopped is not None:
                                cell_values[rule["rule_id"]][ap["label"]] = stopped
                            region = owner_binding.get("arguments", {}).get("region")
                            if region is not None:
                                tasks, initial, updates, task_ledger, state_ledger = _owner_valuation_ledgers(
                                    observation, models or {}, parts or {})
                                current = deepcopy(initial)
                                for update in updates:
                                    if Fraction(str(update["time"])) <= Fraction(observation["time_exact"]):
                                        prior = current[update["resource_id"]]
                                        current[update["resource_id"]] = {**update, "values": {**prior["values"], **update["values"]}}
                                owner_value, _ = _structured_value(owner_binding, Fraction(observation["time_exact"]),
                                    tasks, current, task_ledger, state_ledger)
                                cell_values[rule["rule_id"]][ap["label"]] = sorted({owner and value for owner in
                                    (stopped if stopped is not None else [owner_value]) for value in
                                    cell["occupancy_possibilities"][region][owner_binding["resource_id"]]})
                observation["rule_cells"].append(cell_values)
            if previous != end:
                raise ValueError("Continuous outline cells omit part of the motion interval")
        states, rows, violation, uncertainty = continuous_transition(
            checker, [rule["rule_id"] for rule in rules], states, values, observation, budget)
        checks.extend({**row, "observation_index": index,
                       "phase": observation["phase"], "time": observation["time"]} for row in rows)
        if violation is not None or uncertainty is not None:
            def counterexample(row: dict | None, field: str) -> dict | None:
                if row is None:
                    return None
                recorded = deepcopy(row)
                cell = recorded.get(field)
                if cell is not None:
                    recorded["observation_ap_values"] = deepcopy(recorded["ap_values"])
                    recorded.update(deepcopy(cell))
                recorded.update(observation_index=index, phase=observation["phase"],
                                time=observation["time"], definite_violation=field == "violating_cell")
                return recorded

            violation = counterexample(violation, "violating_cell")
            uncertainty = counterexample(uncertainty, "uncertain_cell")
            if first_uncertainty is None:
                first_uncertainty = uncertainty
            if violation is not None:
                return states, checks, violation, first_uncertainty
            # Unsafe alternatives cannot become safe later. Continue only the
            # surviving states so later evidence can establish that all paths
            # violate a requirement; surviving paths retain this uncertainty.
    return states, checks, None, first_uncertainty


def _restore_history(context: dict, fingerprint: str, checker: BaseSafetyChecker,
                     rules: list, states: list, budget: Budget, first: dict,
                     models: dict | None = None, parts: dict | None = None,
                     selective: bool = False) -> tuple[list, list]:
    if context.get("rules_fingerprint") != fingerprint:
        raise ValueError("Selected safety rules, bindings, or geometry changed since the retained history")
    history = deepcopy(context.get("observations"))
    if not isinstance(history, list) or not history:
        raise ValueError("Active safety requires retained observation history")
    if selective:
        from cais_spade_llm.recovery_framework.outline_evidence_policy import check_monitor_evidence

        check_monitor_evidence(context)
    states, _, violation, uncertainty = _replay(checker, rules, history, states, budget, models, parts)
    if violation is not None or uncertainty is not None:
        raise ValueError("Retained monitor history contains rejected or unresolved safety evidence")
    identifiers = [rule["rule_id"] for rule in rules]
    if context.get("states") != {rid: list(state) for rid, state in zip(identifiers, states, strict=True)}:
        raise ValueError("Retained monitor states disagree with replayed observations")
    from cais_spade_llm.recovery_framework.outline_evidence_policy import predicate_checkpoint

    checkpoint = (lambda row: predicate_checkpoint(row, rules)) if selective else _physical_state
    if checkpoint(history[-1]) != checkpoint(first):
        raise ValueError("Projected candidate does not continue the retained physical checkpoint")
    if any(physical_ap_kind(ap) in {None, "loading_reserved_by"} for rule in rules for ap in rule["aps"]):
        if _digest(history[-1].get("state_snapshot")) != _digest(first.get("state_snapshot")):
            raise ValueError("Projected candidate does not continue the retained owner state checkpoint")
    return states, history


def validate_outline_evidence_safety(*, projection: dict, catalog: dict,
                                     requirement_scopes: list,
                                     monitor_context: dict | None = None,
                                     prospective_activation: bool = False) -> dict[str, Any]:
    """Check one projected primitive witness and retain replayable monitor history.

    Each grounded rule has its own AP namespace. Prior states must be reproduced
    from retained observations, never supplied as unchecked DFA state names.
    A checked prefix does not prove nominal completion or authorize execution.
    """
    result: dict[str, Any] = {
        "is_safe": False, "status": "needs_context", "validation_status": "NEEDS_CONTEXT",
        "reason": "", "states_before": {}, "states_after": {}, "rule_checks": [],
        "execution_authorized": False, "physical_execution_verified": False,
        "nominal_continuation_verified": False, "projection": deepcopy(projection),
    }
    try:
        if catalog == {"version": 2, "specifications": []} and not requirement_scopes:
            result.update(is_safe=True, status="not_selected", validation_status="not_selected",
                          reason="No safety specification is selected")
            return result
        definitions = _catalog(deepcopy(catalog), structured_groundings=True)
        from cais_spade_llm.recovery_framework.outline_evidence_policy import policy_enabled, evidence_coverage

        selective = policy_enabled(projection)
        if selective:
            required = projection["evidence_requirements"]
            checked_coverage = evidence_coverage(requirements=required, projection=projection)
            if checked_coverage != projection.get("evidence_coverage") or checked_coverage["status"] != "covered":
                raise ValueError("Selected predicates lack independently checked evidence coverage")
        if projection.get("status") != "projected":
            raise ValueError("; ".join(projection.get("unresolved") or ["Projected geometry is unavailable"]))
        structured = any(physical_ap_kind(ap) in {None, "loading_reserved_by"}
                         for rule in definitions.values() for ap in rule["aps"])
        if structured:
            projection = bind_outline_owner_safety_evidence(projection)
            result["projection"] = deepcopy(projection)
        observations = deepcopy(projection["observations"])
        if not observations:
            raise ValueError("Projected geometry has no initial checkpoint")
        geometry = projection["geometry"]
        models = (projection["owner_safety_evidence"]["resource_models"] if structured else
                  {rid: {} for rid in observations[0]["resources"]})
        if any("stationary_only" in scope for scope in requirement_scopes):
            declarations = projection.get("owner_safety_evidence", {}).get("resource_models")
            if declarations is not None:
                if not isinstance(declarations, dict) or set(declarations) != set(models):
                    raise ValueError("Mutex participant declarations must cover the recorded resource population")
                models = declarations
            for rid, state in observations[0]["resources"].items():
                if (geometry.get("resources", {}).get(rid, {}).get("stationary_only") is True
                        and "held_part" in state):
                    raise ValueError(f"{rid}: stationary_only conflicts with recorded custody fields")
        rules, coverage = _rules(definitions, requirement_scopes, models, geometry)
        identity = {"rules": rules, "geometry": geometry}
        if selective:
            # Rules already contain exact grounded regions, parts and process
            # bindings. Moving configurations are facts, not new specifications.
            from cais_spade_llm.recovery_framework.outline_evidence_policy import predicate_geometry_identity

            identity = {"rules": rules, "outline_evidence_policy_version": projection["outline_evidence_policy_version"],
                        "predicate_geometry": predicate_geometry_identity(geometry, rules)}
            if "endpoint_evidence_policy_version" in projection:
                identity["endpoint_evidence_policy_version"] = projection["endpoint_evidence_policy_version"]
        if structured or any("stationary_only" in scope for scope in requirement_scopes):
            identity["resource_state_declarations"] = models
        if any(physical_ap_kind(ap) in CASE_AP_MEANINGS for rule in rules for ap in rule["aps"]):
            owner = projection.get("owner_safety_evidence", {})
            declarations = owner.get("resource_configurations")
            if not isinstance(declarations, dict) or not declarations:
                raise ValueError("Case APs require captured owner safety configurations")
            identity["owner_safety_configurations"] = declarations
            identity["collision_object_ids"] = owner.get("collision_object_ids")
            identity["resource_event_fingerprints"] = owner.get("resource_event_fingerprints")
        fingerprint = _digest(identity)
        checker = BaseSafetyChecker({rule["rule_id"]: rule["dfa_dot"] for rule in rules}, rules)
        identifiers = [rule["rule_id"] for rule in rules]
        states = [(checker.dfas[rid]["initial"],) for rid in identifiers]
        budget = Budget(seconds=10)
        history = []
        if monitor_context is not None:
            states, history = _restore_history(monitor_context, fingerprint, checker, rules,
                                               states, budget, observations[0], models, geometry["parts"], selective)
        elif not prospective_activation:
            raise ValueError("Safety history is missing; prospective activation must be explicit")
        result["states_before"] = {rid: list(state) for rid, state in zip(identifiers, states, strict=True)}
        # Replay one complete history so entry predicates retain the preceding
        # occupancy, without consuming the shared checkpoint twice.
        joined = history + (observations[1:] if history else observations)
        initial = [(checker.dfas[rid]["initial"],) for rid in identifiers]
        after, checks, violation, uncertainty = _replay(checker, rules, joined, initial, budget, models, geometry["parts"])
        result.update(rule_checks=checks, coverage=coverage,
                      states_after={rid: list(state) for rid, state in zip(identifiers, after, strict=True)})
        if violation is not None:
            result.update(status="violated", validation_status="INFEASIBLE",
                          reason="The projected primitive witness violates a selected safety requirement",
                          counterexample=violation)
            return result
        if uncertainty is not None:
            result["counterexample"] = uncertainty
            bounded = projection.get("endpoint_evidence_policy_version") in (1, 2)
            result["requires_projection"] = not bounded
            if bounded:
                result["requires_predicate_bounds"] = True
            result["context_requests"] = ["/physical_evidence"]
            result["missing_facts"] = [
                ("More precise supported occupancy bounds for " if bounded else "Refined physical occupancy for ") + uncertainty["rule_id"]
                + " at observation " + str(uncertainty["observation_index"])
                + ", cell " + str(uncertainty.get("cell_index", 0))
                + ", interval " + str(uncertainty.get("interval", [uncertainty["time"]]))]
            raise ValueError("Geometric occupancy alternatives include a safety violation")
        if not selective and projection.get("interval_complete") is not True:
            result["requires_projection"] = True
            raise ValueError("Endpoint checks are recorded; primitive motion intervals remain unverified")
        pending = [rid for rid, current in zip(identifiers, after, strict=True)
                   if any(state not in checker.dfas[rid]["accepting_states"] for state in current)]
        context = {"rules_fingerprint": fingerprint, "observations": joined,
                   "states": deepcopy(result["states_after"]),
                   "pending_rule_ids": pending,
                   "activation": (monitor_context or {}).get("activation", "prospective"),
                   "evidence_kind": "projected"}
        if selective:
            context["outline_evidence_policy_version"] = projection["outline_evidence_policy_version"]
            context["evidence_history"] = deepcopy((monitor_context or {}).get("evidence_history", []))
            from cais_spade_llm.recovery_framework.outline_evidence_policy import monitor_evidence_record

            context["evidence_history"].append(monitor_evidence_record(projection))
        result.update(is_safe=True, status="prefix_checked", validation_status="FEASIBLE",
                      reason="Selected requirements hold over the supplied projected primitive witness",
                      pending_rule_ids=deepcopy(pending),
                      monitor_context=context, monitor_context_after=deepcopy(context))
    except (KeyError, TypeError, ValueError, IndexError, AnalysisLimit) as exc:
        result["reason"] = str(exc)
        result.setdefault("requires_projection", bool(projection.get("requires_projection")))
        result.setdefault("context_requests", [] if result["requires_projection"] else
                          deepcopy(projection.get("context_requests", ["/physical_evidence"])))
    return result

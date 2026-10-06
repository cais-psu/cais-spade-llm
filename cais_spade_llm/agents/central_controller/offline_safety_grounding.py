"""Ground reviewed safety APs against a complete frozen offline scene.

Task and state descriptors retain their declared meaning. Their evidence is
never inferred from a destination name or substituted for geometric occupancy.
The result is conditional on the supplied models, complete ledgers, and scope;
it is neither an execution grant nor an observation of physical execution.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from itertools import combinations
from math import isfinite
from typing import Any

from cais_spade_llm.agents.central_controller._product_effect_evidence import (
    _join_product_effects,
    _product_effect_updates,
    _validate_owner_task_effects,
    _validate_product_checkpoint,
)
from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
    _AP_BINDING_FIELDS,
    _MEANINGS,
    _TARGET_COMPLETED,
    _ap_binding,
    _catalog,
    _digest,
    _evaluate_frozen_trace,
    _list,
    _new_result,
    _object,
    _physical_bindings,
    _scope_geometry,
    _symbol,
    _valuation,
)
from cais_spade_llm.resources.environment_models import build_environment_models
from cais_spade_llm.resources.primitive_observations import model_primitive_observations

_CLOCK_VERSION = "grounded_primitive_observations_joint_trace_v1"
_TASK_FIELDS = {
    "task_id",
    "resource_id",
    "process",
    "product",
    "function",
    "context",
    "start_time",
    "end_time",
}
_STATE_FIELDS = {"resource_id", "process", "product", "context", "values"}


def _time(value: Any, name: str) -> Fraction:
    try:
        finite = type(value) in (int, float) and isfinite(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is outside the supported numeric range") from exc
    if not finite:
        raise ValueError(f"{name} must be a finite numeric time")
    return Fraction(str(value))


def _horizon(value: Any, name: str) -> tuple[Fraction, Fraction]:
    interval = _list(value, name)
    if len(interval) != 2:
        raise ValueError(f"{name} requires start and end times")
    start, end = (_time(item, name) for item in interval)
    if start >= end:
        raise ValueError(f"{name} requires a positive duration")
    return start, end


def _scene_models(
    scene: dict[str, Any], snapshot: dict[str, Any], geometry: dict[str, Any]
) -> dict[str, Any]:
    models = build_environment_models(_object(scene, "scene"))
    population = set(models)
    shapes = _object(_object(geometry, "geometry").get("resources"), "geometry.resources")
    resources = _object(_object(snapshot, "snapshot").get("resources"), "snapshot.resources")
    if not population or set(shapes) != population or not population <= set(resources):
        raise ValueError(
            "Every scene resource requires physical geometry and snapshot evidence; extra spatial identities are unsupported"
        )
    for identifier in population:
        shape = _object(shapes[identifier], f"geometry.resources[{identifier!r}]")
        if (
            shape.get("stationary_only") is True
            and "held_part" in models[identifier]["state_variables"]
        ):
            raise ValueError(
                f"{identifier!r} declares custody and cannot use stationary_only to omit it"
            )
    return models


def _groundings(
    definition: dict[str, Any],
    raw: Any,
    models: dict[str, Any],
    parts: dict[str, Any],
    resource_associations: dict[str, str],
) -> dict[str, Any]:
    mappings = _object(raw, "ap_groundings")
    structured = {ap["label"]: ap for ap in definition["aps"] if ap["full"] not in _MEANINGS}
    if set(mappings) != set(structured):
        raise ValueError(
            "Every structured AP needs exactly one explicit grounding; physical AP meanings cannot be replaced"
        )
    for label, ap in structured.items():
        binding = _object(mappings[label], "AP grounding")
        prefix, process, product, resource, symbol, context = ap["full"].split("/", 5)
        required = {"source", "resource_id", "resource_symbol", "process", "product", "context"}
        required |= {"function"} if prefix == "ap_event" else {"state_field", "state_value"}
        if set(binding) != required:
            raise ValueError("Structured AP grounding fields do not match the descriptor kind")
        for name, value in binding.items():
            _symbol(value, f"AP grounding.{name}")
        if binding["resource_id"] not in models or resource in {"any", "robot"}:
            raise ValueError("Structured APs require an explicit configured resource association")
        if resource in models and resource != binding["resource_id"]:
            raise ValueError("A configured resource identifier cannot denote another resource")
        associated = resource_associations.setdefault(resource, binding["resource_id"])
        if associated != binding["resource_id"]:
            raise ValueError("A preserved resource symbol has ambiguous scene associations")
        expected = {
            "resource_symbol": resource,
            "process": process,
            "product": product,
            "context": context,
        }
        if any(binding[field] != value for field, value in expected.items()):
            raise ValueError("Structured AP grounding changes the preserved descriptor")
        if prefix == "ap_event":
            if binding["source"] != "task_event" or binding["function"] != symbol:
                raise ValueError(
                    "An event AP requires matching task-event evidence, not physical occupancy"
                )
        else:
            if binding["source"] != "resource_state":
                raise ValueError("A state AP requires explicit resource-state evidence")
            if binding["state_field"] not in models[binding["resource_id"]]["state_variables"]:
                raise ValueError("The AP state field is not declared by its resource model")
            expected_symbol = (
                f"{binding['state_field']}={binding['state_value']}"
                if "=" in symbol
                else binding["state_value"]
            )
            if symbol != expected_symbol or (
                "=" not in symbol and binding["state_field"] != "resource_state"
            ):
                raise ValueError("The AP state symbol and explicit field/value disagree")
            _state_binding_value(
                binding["state_value"],
                models[binding["resource_id"]]["state_variables"][binding["state_field"]],
                parts,
            )
    return deepcopy(mappings)


def _rules(
    definitions: dict[str, Any],
    requirement_scopes: Any,
    models: dict[str, Any],
    geometry: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    population = sorted(models)
    rules, used, seen = [], set(), set()
    resource_associations: dict[str, str] = {}
    regions = _object(geometry.get("regions"), "geometry.regions")
    for scope in _list(requirement_scopes, "requirement_scopes"):
        scope = _object(scope, "requirement scope")
        identifier = _symbol(scope.get("specification"), "scope.specification")
        if identifier not in definitions:
            raise ValueError("A scoped specification is missing from the reviewed catalog")
        definition = definitions[identifier]
        required = set().union(
            *(_AP_BINDING_FIELDS[ap["full"]] for ap in definition["aps"] if ap["full"] in _MEANINGS)
        )
        physical_bindings = (
            _physical_bindings(definition, scope["physical_ap_bindings"])
            if "physical_ap_bindings" in scope else None
        )
        paired = "resources" in required
        if physical_bindings is not None:
            required = set()
        required.discard("resources")
        expected_fields = required | {"specification"}
        if physical_bindings is not None:
            expected_fields.add("physical_ap_bindings")
        has_structured = any(ap["full"] not in _MEANINGS for ap in definition["aps"])
        if has_structured:
            expected_fields.add("ap_groundings")
        if set(scope) != expected_fields:
            raise ValueError(
                "Requirement scopes contain unsupported fields or missing exact bindings; participant and pair lists are not accepted"
            )
        for field in required:
            _symbol(scope[field], f"scope.{field}")
        for binding in physical_bindings.values() if physical_bindings is not None else [scope]:
            _scope_geometry(binding, geometry, regions, models)
        groundings = _groundings(
            definition,
            scope.get("ap_groundings", {}),
            models,
            _object(geometry.get("parts"), "geometry.parts"),
            resource_associations,
        )
        pairs = list(combinations(population, 2)) if paired else [None]
        if not pairs:
            raise ValueError("Pairwise requirements need two configured scene resources")
        for pair in pairs:
            binding = {
                key: deepcopy(value) for key, value in scope.items()
                if key not in {"ap_groundings", "physical_ap_bindings"}
            }
            if pair is not None:
                binding["resources"] = list(pair)
            identity = {"binding": binding, "ap_groundings": groundings}
            if physical_bindings is not None:
                identity["physical_ap_bindings"] = physical_bindings
            rule_id = f"{identifier}:{_digest(identity)}"
            if rule_id in seen:
                raise ValueError("Duplicate grounded requirement instance")
            seen.add(rule_id)
            rules.append(
                {
                    **deepcopy(definition),
                    "rule_id": rule_id,
                    "binding": binding,
                    "ap_groundings": groundings,
                    **({"physical_ap_bindings": physical_bindings} if physical_bindings is not None else {}),
                }
            )
        used.add(identifier)
    if used != set(definitions):
        raise ValueError("Every reviewed requirement needs an explicit requirement scope")
    rules.sort(key=lambda row: row["rule_id"])
    return rules, {
        "population": population,
        "requirement_scopes": deepcopy(requirement_scopes),
        "rule_ids": [row["rule_id"] for row in rules],
        "complete_relative_to": "build_environment_models(scene)",
    }


def _ledger(
    raw: Any, name: str, horizon: tuple[Fraction, Fraction], fields: set[str]
) -> dict[str, Any] | None:
    if raw is None:
        return None
    ledger = _object(raw, name)
    if (
        set(ledger) != fields | {"complete", "source_kind", "horizon"}
        or ledger["complete"] is not True
    ):
        raise ValueError(f"{name} requires complete, explicit evidence for the full horizon")
    _symbol(ledger["source_kind"], f"{name}.source_kind")
    if _horizon(ledger["horizon"], f"{name}.horizon") != horizon:
        raise ValueError(f"{name} does not cover the physical observation horizon")
    return ledger


def _tasks(
    ledger: dict[str, Any] | None, models: dict[str, Any], horizon: tuple[Fraction, Fraction]
) -> tuple[dict[str, Any], list[float]]:
    tasks, boundaries = {}, []
    if ledger is None:
        return tasks, boundaries
    for row in _list(ledger["events"], "task_evidence.events"):
        row = _object(row, "task event")
        if set(row) not in (_TASK_FIELDS, _TASK_FIELDS | {"declared_task"}):
            raise ValueError(
                "Task events require exact identities, context, and start/end intervals"
            )
        for name in _TASK_FIELDS - {"start_time", "end_time"}:
            _symbol(row[name], f"task event.{name}")
        if row["task_id"] in tasks or row["resource_id"] not in models:
            raise ValueError("Task evidence has a duplicate task or unknown resource")
        start, end = _time(row["start_time"], "task start"), _time(row["end_time"], "task end")
        if not horizon[0] <= start < end <= horizon[1]:
            raise ValueError("Task intervals must have positive duration inside the frozen horizon")
        if any(
            row["resource_id"] == other["resource_id"]
            and max(start, _time(other["start_time"], "task start"))
            < min(end, _time(other["end_time"], "task end"))
            for other in tasks.values()
        ):
            raise ValueError("Overlapping task intervals for one resource are ambiguous")
        tasks[row["task_id"]] = row
        boundaries.extend((row["start_time"], row["end_time"]))
    return tasks, boundaries


def _scalar_symbol(value: Any) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    return str(value)


def _state_binding_value(symbol: str, declaration: dict[str, Any], parts: dict[str, Any]) -> None:
    if "domain" in declaration:
        candidates = declaration["domain"]
    elif declaration.get("reference") == "part_name":
        candidates = [None, *parts]
    else:
        candidates = [None, symbol, True, False]
        try:
            candidates.append(int(symbol))
            candidates.append(float(symbol))
        except ValueError:
            pass
    matching = [value for value in candidates if _scalar_symbol(value) == symbol]
    for value in matching:
        try:
            _state_value(value, declaration, parts)
        except ValueError:
            continue
        return
    raise ValueError("The AP state value is outside its declared type or domain")


def _state_value(value: Any, declaration: dict[str, Any], parts: dict[str, Any]) -> None:
    if value is not None and type(value) not in (str, bool, int, float):
        raise ValueError("Resource-state values must be explicit scalar observations")
    if type(value) is float and not isfinite(value):
        raise ValueError("Resource-state numeric observations must be finite")
    if "domain" in declaration:
        if not any(type(value) is type(item) and value == item for item in declaration["domain"]):
            raise ValueError("Resource-state value is outside its declared domain")
        return
    types = declaration.get("type", [])
    types = [types] if isinstance(types, str) else types
    actual = {type(None): "null", str: "string", bool: "boolean", int: "integer", float: "number"}[
        type(value)
    ]
    if actual not in types and not (actual == "integer" and "number" in types):
        raise ValueError("Resource-state value does not match its declared type")
    if declaration.get("reference") == "part_name" and value is not None and value not in parts:
        raise ValueError("Resource-state part reference has no complete modeled part evidence")
    if type(value) in (int, float) and (
        ("minimum" in declaration and value < declaration["minimum"])
        or ("maximum" in declaration and value > declaration["maximum"])
    ):
        raise ValueError("Resource-state value is outside its declared numeric bounds")


def _state_row(
    row: Any, models: dict[str, Any], parts: dict[str, Any], *, update: bool
) -> dict[str, Any]:
    row = _object(row, "resource-state evidence")
    if set(row) != _STATE_FIELDS | ({"time", "task_id"} if update else set()):
        raise ValueError("Resource-state evidence has incomplete or unsupported fields")
    for field in _STATE_FIELDS - {"values"}:
        _symbol(row[field], f"resource-state evidence.{field}")
    if row["resource_id"] not in models:
        raise ValueError("Resource-state evidence references an unknown scene resource")
    values = _object(row["values"], "resource-state values")
    if not values or not set(values) <= set(models[row["resource_id"]]["state_variables"]):
        raise ValueError("Resource-state values must use declared resource fields")
    for field, value in values.items():
        _state_value(value, models[row["resource_id"]]["state_variables"][field], parts)
    return row


def _states(
    ledger: dict[str, Any] | None,
    tasks: dict[str, Any],
    models: dict[str, Any],
    horizon: tuple[Fraction, Fraction],
    parts: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    initial, updates, boundaries = {}, [], []
    if ledger is None:
        return initial, updates, boundaries
    for raw in _list(ledger["initial"], "state_evidence.initial"):
        row = _state_row(raw, models, parts, update=False)
        if row["resource_id"] in initial:
            raise ValueError("Resource-state initial evidence is ambiguous")
        initial[row["resource_id"]] = row
    changed = set()
    for raw in _list(ledger["updates"], "state_evidence.updates"):
        row = _state_row(raw, models, parts, update=True)
        time = _time(row["time"], "state update time")
        task_id = _symbol(row["task_id"], "state update task_id")
        task = tasks.get(task_id)
        if task is None or _time(task["end_time"], "task completion") != time:
            raise ValueError("State updates require their exact task-completion evidence")
        if any(
            row[field] != task[field] for field in ("resource_id", "process", "product", "context")
        ):
            raise ValueError("State updates disagree with the completed task identity or context")
        if row["resource_id"] not in initial or not horizon[0] <= time <= horizon[1]:
            raise ValueError(
                "State update lacks a complete initial state or lies outside the horizon"
            )
        key = (row["resource_id"], time)
        if key in changed:
            raise ValueError(
                "Simultaneous updates to one resource require one joint, unambiguous state record"
            )
        changed.add(key)
        updates.append(row)
        boundaries.append(row["time"])
    updates.sort(key=lambda row: (_time(row["time"], "state update time"), row["resource_id"]))
    return initial, updates, boundaries


def _matches(binding: dict[str, Any], row: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return binding["resource_id"] == row["resource_id"] and all(
        binding[field] == row[field] or (field != "function" and binding[field] == "any")
        for field in fields
    )


def _structured_value(
    binding: dict[str, Any],
    time: Fraction,
    tasks: dict[str, Any],
    state: dict[str, Any],
    task_ledger: dict[str, Any] | None,
    state_ledger: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    if binding["source"] == "task_event":
        if task_ledger is None:
            raise ValueError("Task-event AP evidence is unavailable")
        matching = [
            row["task_id"]
            for row in tasks.values()
            if _matches(binding, row, ("process", "product", "function", "context"))
            and _time(row["start_time"], "task start") <= time < _time(row["end_time"], "task end")
        ]
        return bool(matching), {
            "source_kind": task_ledger["source_kind"],
            "ledger": "task_evidence",
            "complete": True,
            "task_ids": sorted(matching),
        }
    if state_ledger is None or binding["resource_id"] not in state:
        raise ValueError("Resource-state AP evidence is unavailable")
    row = state[binding["resource_id"]]
    if binding["state_field"] not in row["values"]:
        raise ValueError("Resource-state AP field has no complete observation evidence")
    value = (
        _matches(binding, row, ("process", "product", "context"))
        and _scalar_symbol(row["values"][binding["state_field"]]) == binding["state_value"]
    )
    return value, {
        "source_kind": state_ledger["source_kind"],
        "ledger": "state_evidence",
        "complete": True,
        "task_id": row.get("task_id"),
        "time": row.get("time"),
        "field": binding["state_field"],
    }


def _valuations(  # noqa: PLR0913
    rules: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    tasks: dict[str, Any],
    initial: dict[str, Any],
    updates: list[dict[str, Any]],
    task_ledger: dict[str, Any] | None,
    state_ledger: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valuations, evidence = [], []
    state, next_update = deepcopy(initial), 0
    for index, observation in enumerate(observations):
        time = Fraction(observation["time_exact"])
        while (
            next_update < len(updates)
            and _time(updates[next_update]["time"], "state update") <= time
        ):
            row = updates[next_update]
            prior = state[row["resource_id"]]
            state[row["resource_id"]] = {
                **deepcopy(row),
                "values": {**prior["values"], **row["values"]},
            }
            next_update += 1
        for resource, row in state.items():
            if "held_part" in row["values"] and row["values"]["held_part"] != observation[
                "resources"
            ][resource].get("held_part"):
                raise ValueError(
                    "Resource-state custody evidence contradicts the physical observations"
                )
        values = {}
        for rule in rules:
            values[rule["rule_id"]] = {}
            for ap in rule["aps"]:
                if ap["full"] in _MEANINGS:
                    bound = _ap_binding(rule, ap)
                    if 'occupancy_possibilities' in observation and ap['full'] in {
                        'ap_state/physical_observation/shared_area_first_resource',
                        'ap_state/physical_observation/shared_area_second_resource',
                    }:
                        resource = bound['resources'][0 if ap['full'].endswith('first_resource') else 1]
                        value = deepcopy(observation['occupancy_possibilities'][bound['region']][resource])
                    elif ap['full'] == 'ap_event/physical_observation/part_region_entry' and 'part_entry_possibilities' in observation:
                        value = deepcopy(observation['part_entry_possibilities'][bound['region']][bound['part']])
                    else:
                        value = _valuation(
                            {**rule, "aps": [ap]}, observation,
                            observations[index - 1] if index else None,
                        )[ap["label"]]
                    source = {
                        "source_kind": "resource_model",
                        "observation_index": index,
                        "predicate": ap["full"],
                    }
                    if ap["full"] == _TARGET_COMPLETED:
                        part = observation["parts"][bound["part"]]
                        source["checkpoint"] = deepcopy(part["processCompleted_evidence"])
                        source["product_effect_evidence"] = deepcopy(part.get("product_effect_evidence", []))
                else:
                    bound = rule["ap_groundings"][ap["label"]]
                    value, source = _structured_value(
                        bound, time, tasks, state, task_ledger, state_ledger
                    )
                values[rule["rule_id"]][ap["label"]] = value
                evidence.append(
                    {
                        "rule_id": rule["rule_id"],
                        "descriptor": deepcopy(ap),
                        "binding": deepcopy(bound),
                        "evidence_source": source,
                        "observation_index": index,
                        "time": observation["time"],
                        "phase": observation["phase"],
                        "value": (value[0] if len(value) == 1 else None) if isinstance(value, list) else value,
                        **({'possible_values': deepcopy(value)} if isinstance(value, list) else {}),
                    }
                )
        valuations.append(values)
    return valuations, evidence


class _GroundingPreparationError(ValueError):
    """Preserve established coverage when later evidence is unavailable."""

    def __init__(
        self, reason: str, bindings: list[dict[str, Any]], coverage: dict[str, Any]
    ) -> None:
        super().__init__(reason)
        self.bindings = deepcopy(bindings)
        self.coverage = deepcopy(coverage)


def _validate_composition_formulas(catalog: dict[str, Any]) -> None:
    """Reject formulas outside the composition experiment's Boolean/G/F/U fragment."""
    from lark.exceptions import LarkError
    from ltlf2dfa.ltlf import (
        LTLfAlways,
        LTLfAnd,
        LTLfAtomic,
        LTLfEquivalence,
        LTLfEventually,
        LTLfFalse,
        LTLfImplies,
        LTLfNot,
        LTLfOr,
        LTLfTrue,
        LTLfUntil,
    )
    from ltlf2dfa.parser.ltlf import LTLfParser

    supported = {
        LTLfAlways,
        LTLfAnd,
        LTLfAtomic,
        LTLfEquivalence,
        LTLfEventually,
        LTLfFalse,
        LTLfImplies,
        LTLfNot,
        LTLfOr,
        LTLfTrue,
        LTLfUntil,
    }
    rows = _list(_object(catalog, "catalog").get("specifications"), "catalog.specifications")
    if not rows:
        raise ValueError("Composition requires a nonempty reviewed catalog")
    for row in rows:
        formula = _symbol(_object(row, "specification").get("formula"), "specification.formula")
        try:
            nodes = [LTLfParser()(formula)]
        except (LarkError, ValueError) as exc:
            raise ValueError(f"Invalid composition LTLf formula: {exc}") from exc
        while nodes:
            node = nodes.pop()
            if type(node) not in supported:
                raise ValueError(
                    f"Composition supports only Boolean/G/F/U formulas; {type(node).__name__} requires a different observation-clock contract"
                )
            nodes.extend(getattr(node, "formulas", ()))
            if hasattr(node, "f"):
                nodes.append(node.f)


def _prepare_grounded_primitive_trace(  # noqa: PLR0913
    *,
    scene: dict[str, Any],
    catalog: dict[str, Any],
    requirement_scopes: list[dict[str, Any]],
    programs: list[dict[str, Any]],
    snapshot: dict[str, Any],
    geometry: dict[str, Any],
    horizon: list[float],
    stationary: dict[str, Any],
    task_evidence: dict[str, Any] | None = None,
    state_evidence: dict[str, Any] | None = None,
    product_effect_evidence: dict[str, Any] | None = None,
    observation_boundaries: list[float] | None = None,
    allow_predicted_product_effects: bool = False,
    primitive_models: dict | None = None,
    motion_budget=None,
) -> dict[str, Any]:
    """Prepare complete grounded observations without evaluating DFA acceptance.

    Inputs are copied. The returned trace retains unsafe valuations so a caller
    can represent every unavoidable branch. Optional decision boundaries join
    the same joint clock before interval representatives are constructed.
    """
    bindings: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    try:
        if type(allow_predicted_product_effects) is not bool:
            raise ValueError("Predicted product effects require an explicit Boolean mode")
        frozen = deepcopy(
            {
                "scene": scene,
                "catalog": catalog,
                "requirement_scopes": requirement_scopes,
                "programs": programs,
                "snapshot": snapshot,
                "geometry": geometry,
                "horizon": horizon,
                "stationary": stationary,
                "task_evidence": task_evidence,
                "state_evidence": state_evidence,
            }
        )
        if observation_boundaries is not None:
            frozen["observation_boundaries"] = deepcopy(observation_boundaries)
        if product_effect_evidence is not None:
            frozen["product_effect_evidence"] = deepcopy(product_effect_evidence)
            frozen["allow_predicted_product_effects"] = allow_predicted_product_effects
        models = _scene_models(frozen["scene"], frozen["snapshot"], frozen["geometry"])
        _validate_product_checkpoint(frozen["snapshot"], allow_predicted=allow_predicted_product_effects)
        definitions = _catalog(frozen["catalog"], structured_groundings=True)
        rules, coverage = _rules(
            definitions, frozen["requirement_scopes"], models, frozen["geometry"]
        )
        bindings = [
            {
                "rule_id": row["rule_id"],
                **deepcopy(row["binding"]),
                "ap_groundings": deepcopy(row["ap_groundings"]),
                **({"physical_ap_bindings": deepcopy(row["physical_ap_bindings"])}
                   if "physical_ap_bindings" in row else {}),
            }
            for row in rules
        ]
        interval = _horizon(frozen["horizon"], "horizon")
        task_ledger = _ledger(frozen["task_evidence"], "task_evidence", interval, {"events"})
        state_ledger = _ledger(
            frozen["state_evidence"], "state_evidence", interval, {"initial", "updates"}
        )
        tasks, boundaries = _tasks(task_ledger, models, interval)
        product_ledger = _ledger(
            frozen.get("product_effect_evidence"), "product_effect_evidence", interval, {"updates"}
        )
        product_updates, product_boundaries = _product_effect_updates(
            product_ledger, tasks, models, frozen["snapshot"], frozen["geometry"],
            allow_predicted=allow_predicted_product_effects,
        )
        initial, updates, state_boundaries = _states(
            state_ledger,
            tasks,
            models,
            interval,
            _object(frozen["snapshot"].get("parts"), "snapshot.parts"),
        )
        decision_boundaries = (
            []
            if observation_boundaries is None
            else _list(frozen["observation_boundaries"], "observation_boundaries")
        )
        for value in decision_boundaries:
            if not interval[0] <= _time(value, "observation boundary") <= interval[1]:
                raise ValueError("Observation boundary lies outside the frozen horizon")
        physical_geometry = deepcopy(frozen["geometry"])
        for part in physical_geometry["parts"].values():
            if "target" in part:
                _symbol(part.pop("target"), "part geometry.target")
        model = model_primitive_observations(
            programs=frozen["programs"],
            snapshot=frozen["snapshot"],
            geometry=physical_geometry,
            horizon=frozen["horizon"],
            stationary=frozen["stationary"],
            bindings=[],
            observation_boundaries=sorted(set(boundaries + state_boundaries + decision_boundaries + product_boundaries)),
            primitive_models=primitive_models,
            motion_budget=motion_budget,
        )
        if model.get("valid") is not True:
            raise ValueError(model.get("reason", "Physical observations are unavailable"))
        model = _join_product_effects(model, product_updates, frozen["snapshot"])
        _validate_owner_task_effects(model, tasks, models, product_updates, initial, updates)
        valuations, evidence = _valuations(
            rules, model["observations"], tasks, initial, updates, task_ledger, state_ledger
        )
        if model.get('evidence', {}).get('continuous_motion'):
            _validate_composition_formulas(frozen['catalog'])
            for observation, values in zip(model['observations'], valuations, strict=True):
                observation['rule_cells'] = []
                for cell in observation.get('continuous_cells', []):
                    cell_values = deepcopy(values)
                    for rule in rules:
                        for ap in rule['aps']:
                            if ap['full'] in {
                                'ap_state/physical_observation/shared_area_first_resource',
                                'ap_state/physical_observation/shared_area_second_resource',
                            }:
                                binding = _ap_binding(rule, ap)
                                resource = binding['resources'][0 if ap['full'].endswith('first_resource') else 1]
                                cell_values[rule['rule_id']][ap['label']] = cell['occupancy_possibilities'][binding['region']][resource]
                    observation['rule_cells'].append(cell_values)
        return {
            "frozen": frozen,
            "models": models,
            "rules": rules,
            "bindings": bindings,
            "coverage": coverage,
            "model": model,
            "observations": model["observations"],
            "projected_snapshot": model["projected_snapshot"],
            "valuations": valuations,
            "ap_evidence": evidence,
            "fingerprint": {**frozen, "population": sorted(models),
                            **({"primitive_models": model["evidence"]["primitive_models"],
                                "owner_effects": model["evidence"].get("owner_effects", [])}
                               if model.get("evidence", {}).get("primitive_models") else {})},
            "clock_version": (model['evidence']['clock_version'] if model.get('evidence', {}).get('continuous_motion') else _CLOCK_VERSION if product_effect_evidence is None
                              else "grounded_primitive_observations_product_effects_v1"),
        }
    except (ImportError, OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        raise _GroundingPreparationError(str(exc), bindings, coverage) from exc


def validate_grounded_primitive_program_safety(  # noqa: PLR0913
    *,
    scene: dict[str, Any],
    catalog: dict[str, Any],
    requirement_scopes: list[dict[str, Any]],
    programs: list[dict[str, Any]],
    snapshot: dict[str, Any],
    geometry: dict[str, Any],
    horizon: list[float],
    stationary: dict[str, Any],
    task_evidence: dict[str, Any] | None = None,
    state_evidence: dict[str, Any] | None = None,
    product_effect_evidence: dict[str, Any] | None = None,
    continuation: dict[str, Any] | None = None,
    trace_complete: bool = False,
    observation_slice: list[int] | None = None,
    primitive_models: dict | None = None,
) -> dict[str, Any]:
    """Derive concrete AP bindings and check one complete offline scene.

    Args:
        scene: Frozen configured scene consumed by build_environment_models.
        catalog: Reviewed formulas and exact physical or structured AP meanings.
        requirement_scopes: Requirement region/part bindings and explicit
            structured AP evidence associations; never participant or pair lists.
        programs: Exact bound primitive programs and resource-owned evidence.
        snapshot: Initial resource, part, custody, and process evidence.
        geometry: Complete physical envelopes for every configured resource.
        horizon: Complete shared observation interval.
        stationary: Explicit coverage outside each modeled primitive interval.
        task_evidence: Complete optional task intervals on this horizon.
        state_evidence: Complete optional initial states and task-completion updates.
        product_effect_evidence: Complete optional acknowledged assembly effects;
            predictions cannot be supplied as observed completion.
        continuation: Previously accepted continuation of this exact frozen trace.
        trace_complete: Whether the actual trace endpoint is being finalized.
        observation_slice: Optional contiguous half-open observation index range.
        primitive_models: Trusted pure resource models supplied separately from evidence.

    Returns:
        Offline verdict, automatically derived coverage, per-AP provenance, and
        monitor continuation. Rejections do not advance the supplied continuation.
    """
    result = _new_result(continuation, _CLOCK_VERSION)
    result["ap_evidence"] = []
    try:
        if type(trace_complete) is not bool:
            raise ValueError("trace_complete must be an explicit Boolean")
        prepared = _prepare_grounded_primitive_trace(
            scene=scene,
            catalog=catalog,
            requirement_scopes=requirement_scopes,
            programs=programs,
            snapshot=snapshot,
            geometry=geometry,
            horizon=horizon,
            stationary=stationary,
            task_evidence=task_evidence,
            state_evidence=state_evidence,
            product_effect_evidence=product_effect_evidence,
            primitive_models=primitive_models,
        )
        result["coverage"] = prepared["coverage"]
        result["bindings"] = prepared["bindings"]
        return _evaluate_frozen_trace(
            result=result,
            model=prepared["model"],
            snapshot=prepared["frozen"]["snapshot"],
            rules=prepared["rules"],
            fingerprint=prepared["fingerprint"],
            continuation=continuation,
            trace_complete=trace_complete,
            observation_slice=observation_slice,
            clock_version=prepared["clock_version"],
            valuations=prepared["valuations"],
            ap_evidence=prepared["ap_evidence"],
        )
    except (ImportError, OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        if isinstance(exc, _GroundingPreparationError):
            result["coverage"] = exc.coverage
            result["bindings"] = exc.bindings
        result["reason"] = str(exc)
        return result

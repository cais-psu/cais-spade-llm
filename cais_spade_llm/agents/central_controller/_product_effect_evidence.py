"""Validated resource-declared product acknowledgements joined to an offline physical trace.

These records are explicit evidence assumptions, not proof that a robot ran.
Primitives never manufacture completion. Composition may separately request
declared predictions, which remain marked predicted in every returned record.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from math import isfinite

from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
    _complete_target_ledger,
    _list,
    _object,
    _symbol,
)
from cais_spade_llm.resources.environment_models import (
    _apply_resource_effects,
    _check_product_requirements,
    _task_event,
    check_value,
    declaration_for,
)


def _validate_owner_state_values(model: dict, contracts: dict, state_initial: dict, state_updates: list) -> None:
    values = {rid: deepcopy(row["values"]) for rid, row in state_initial.items() if rid in contracts}
    cursor = 0
    for observation in model["observations"]:
        time = Fraction(observation["time_exact"])
        while cursor < len(state_updates) and Fraction(str(state_updates[cursor]["time"])) <= time:
            row = state_updates[cursor]
            if row["resource_id"] in values:
                values[row["resource_id"]].update(deepcopy(row["values"]))
            cursor += 1
        for rid, fields in values.items():
            physical = observation["resources"][rid]
            if any(field in physical and physical[field] != value for field, value in fields.items()):
                raise ValueError("Resource state AP evidence contradicts owner observations")


def _validate_native_updates(models: dict, after: dict, first: dict, last: dict, event: dict) -> None:
    for rid in event["participants"]:
        for field, value in after[rid].items():
            try:
                declaration = declaration_for(models[rid], field)
            except ValueError:
                continue  # Physical pose/evidence fields are separate from the native model.
            if not check_value(declaration, value, first["parts"]):
                raise ValueError("Declared task effect violates the resource state domain")
            if last["resources"][rid].get(field) != value:
                raise ValueError("Owner state evidence differs from the declared task effect")


def _validate_owner_task_effects(model: dict, tasks: dict, models: dict, updates: list[dict],
                                 state_initial: dict, state_updates: list[dict]) -> None:
    """Check new owner state/product effects against exact configured tasks.

    Historical acknowledgement-only inputs retain their established evidence
    contract. Owner-modeled updates additionally require native guards, the
    complete declared primitive program and observed resource effects. Ordered
    product requirements are supplied by the trusted owner's frozen configuration.
    """
    evidence = model.get("evidence", {})
    contracts = evidence.get("primitive_models", {})
    steps = evidence.get("owner_effects", [])
    product_updates = {row["task_id"]: row for row in updates}
    needed = {identity for identity, row in product_updates.items() if row["resource_id"] in contracts}
    needed.update(row["task_id"] for row in state_updates if row["resource_id"] in contracts)
    _validate_owner_state_values(model, contracts, state_initial, state_updates)
    for step in steps:
        if not step["effects"]["resource_updates"]:
            continue
        resource = step["resource_id"]
        for field in step["effects"]["resource_updates"]:
            declared = declaration_for(models[resource], field)
            owner_declared = declaration_for({"state_variables": contracts[resource]["configuration"].get("state_variables", {})}, field)
            if owner_declared != declared:
                raise ValueError("Owner state declaration differs from its configured resource")
        matches = [identity for identity, task in tasks.items()
                   if task["resource_id"] == step["resource_id"]
                   and Fraction(str(task["end_time"])) == Fraction(step["end_time"])
                   and Fraction(str(task["start_time"])) <= Fraction(step["start_time"])]
        if len(matches) != 1:
            raise ValueError("Owner state updates require an exact declared task completion")
        needed.add(matches[0])
    for identity in sorted(needed, key=lambda key: (tasks[key]["end_time"], key)):
        task = tasks[identity]
        declaration = task.get("declared_task", product_updates.get(identity, {}).get("declaration"))
        if declaration is None:
            raise ValueError("Owner effects require their exact declared_task association")
        resource = task["resource_id"]
        start, end = (Fraction(str(task[field])) for field in ("start_time", "end_time"))
        first = next(row for row in model["observations"] if Fraction(row["time_exact"]) == start)
        last = next(row for row in model["observations"] if Fraction(row["time_exact"]) == end)
        event = _task_event(models, first["parts"], {**declaration, "resource_id": resource})
        params = declaration["parameters"]
        configured = event.get("program", {}).get("steps")
        actual = [row for row in steps if row["resource_id"] == resource
                  and start <= Fraction(row["start_time"]) and Fraction(row["end_time"]) <= end]
        if (not configured or len(actual) != len(configured)
                or Fraction(actual[0]["start_time"]) != start or Fraction(actual[-1]["end_time"]) != end
                or any(row["primitive"] != expected["op"] or row["params"] != expected.get("params", {})
                       for row, expected in zip(actual, configured, strict=True))):
            raise ValueError("Owner effects do not cover the exact supported task program")
        after = deepcopy(first["resources"])
        _apply_resource_effects(models, first["resources"], after, event, params)
        _validate_native_updates(models, after, first, last, event)
        if identity in product_updates:
            requirements = contracts[resource]["configuration"].get("requirements")
            if not isinstance(requirements, dict) or task["product"] not in requirements:
                raise ValueError("Owner process completion needs the configured product requirements")
            _check_product_requirements({**declaration, "resource_id": resource}, first["parts"],
                                        task["product"], requirements, event)


def _declared_product_effects(
    models: dict, parts: dict, geometry: dict, resource_id: str, declaration: dict,
) -> dict:
    """Resolve exact supported product fields from a configured task declaration."""
    declaration = _object(declaration, "declared_task")
    if set(declaration) != {"event_id", "event_name", "parameters"}:
        raise ValueError("declared_task requires the exact event and parameter bindings")
    if type(declaration["event_id"]) is not int or resource_id not in models:
        raise ValueError("Unknown resource or invalid declared event identifier")
    _symbol(declaration["event_name"], "declared_task.event_name")
    params = _object(declaration["parameters"], "declared_task.parameters")
    event = _task_event(models, parts, {**declaration, "resource_id": resource_id})
    part = params["part_name"]
    if "target" in params and _object(geometry["parts"][part], "part geometry").get("target") != params["target"]:
        raise ValueError("Assembly effect target differs from the part's configured geometry")
    declared = event.get("product_effects", {})
    if "processCompleted" not in declared or set(declared) - {"state", "target", "processCompleted"}:
        raise ValueError("Unsupported declared product effects")
    effects = {}
    for field in ("state", "target"):
        if field not in declared:
            continue
        value = declared[field]
        if isinstance(value, dict):
            if set(value) != {"set_from_param"}:
                raise ValueError("Unsupported declared product-effect parameter")
            value = params[value["set_from_param"]]
        _symbol(value, f"product_effects.{field}")
        effects[field] = value
    effects["processCompleted"] = []
    for record in _list(declared["processCompleted"], "declared processCompleted"):
        record = _object(record, "declared process record")
        resolved = {}
        for field, value in record.items():
            if isinstance(value, dict):
                if set(value) != {"set_from_param"}:
                    raise ValueError("Unsupported declared process-effect parameter")
                value = params[value["set_from_param"]]
            resolved[field] = _symbol(value, f"product_effects.processCompleted.{field}")
        effects["processCompleted"].append(resolved)
    if not effects["processCompleted"]:
        raise ValueError("The declared task establishes no process completion")
    _complete_target_ledger({**parts[part], "processCompleted": effects["processCompleted"]})
    if "target" in effects and effects["target"] != geometry["parts"][part].get("target"):
        raise ValueError("Declared product target differs from configured geometry")
    return {part: effects}


def _product_effect_updates(
    ledger: dict | None, tasks: dict, models: dict, snapshot: dict, geometry: dict,
    *, allow_predicted: bool = False,
) -> tuple[list[dict], list[float]]:
    """Validate complete explicit evidence; do not infer missing task effects."""
    updates, boundaries, seen = [], [], set()
    if ledger is None:
        return updates, boundaries
    start, end = (Fraction(str(value)) for value in ledger["horizon"])
    for raw in _list(ledger["updates"], "product_effect_evidence.updates"):
        row = _object(raw, "product-effect update")
        if set(row) != {"task_id", "resource_id", "time", "kind", "declaration", "product_effects"}:
            raise ValueError("Product-effect updates require exact task, declaration and effect evidence")
        for field in ("task_id", "resource_id", "kind"):
            _symbol(row[field], f"product-effect update.{field}")
        if row["kind"] not in ({"acknowledged", "predicted"} if allow_predicted else {"acknowledged"}):
            raise ValueError("Predicted product effects are not acknowledged execution history")
        time = row["time"]
        try:
            finite = type(time) in (int, float) and isfinite(time)
        except OverflowError as exc:
            raise ValueError("Product-effect time is outside the supported numeric range") from exc
        if not finite or not start < Fraction(str(time)) <= end:
            raise ValueError("Product-effect time lies outside the acknowledged horizon")
        task = tasks.get(row["task_id"])
        if task is None or task["resource_id"] != row["resource_id"] or task["end_time"] != time:
            raise ValueError("Product-effect acknowledgement does not match a task completion")
        if row["task_id"] in seen:
            raise ValueError("Duplicate product-effect acknowledgement")
        seen.add(row["task_id"])
        declaration = _object(row["declaration"], "product-effect declaration")
        if "declared_task" in task:
            if task["declared_task"] != declaration:
                raise ValueError("Product-effect declaration differs from the task's reviewed association")
        elif task["function"] != declaration.get("event_name"):
            raise ValueError("Generated task effects require an explicit declared_task association")
        effects = _declared_product_effects(
            models, snapshot["parts"], geometry, row["resource_id"], declaration,
        )
        part = next(iter(effects))
        destination = declaration["parameters"].get("destination_location")
        processes = {record["process"] for record in effects[part]["processCompleted"]}
        if (task["product"] != part or processes != {task["process"]}
                or destination is not None and task["context"] != f"destination={destination}"):
            raise ValueError("Product-effect part/process/destination differs from its task evidence")
        if row["product_effects"] != effects:
            raise ValueError("Product-effect evidence differs from the exact declared effects")
        _complete_target_ledger(snapshot["parts"][part])
        updates.append({**deepcopy(row), "source_kind": ledger["source_kind"]})
        boundaries.append(time)
    updates.sort(key=lambda row: (Fraction(str(row["time"])), row["task_id"]))
    return updates, boundaries


def _validate_product_checkpoint(snapshot: dict, *, allow_predicted: bool = False) -> None:
    for part in _object(snapshot.get("parts"), "snapshot.parts").values():
        part = _object(part, "snapshot part")
        for row in _list(part.get("product_effect_evidence", []), "checkpoint product effects"):
            row = _object(row, "checkpoint product effect")
            if row.get("kind") not in ({"acknowledged", "predicted"} if allow_predicted else {"acknowledged"}):
                raise ValueError("A predicted product effect cannot establish an acknowledged checkpoint")


def _join_product_effects(model: dict, updates: list[dict], snapshot: dict) -> dict:
    """Join acknowledged effects at joint boundaries, preserving primitive data."""
    if not updates:
        return model
    result = deepcopy(model)
    products = deepcopy(snapshot["parts"])
    cursor = 0
    changed: set[str] = set()
    for observation in result["observations"]:
        time = Fraction(observation["time_exact"])
        while cursor < len(updates) and Fraction(str(updates[cursor]["time"])) <= time:
            row = updates[cursor]
            for part, effects in row["product_effects"].items():
                if effects.get("state") == "assembled" and any(part in held for held in observation["carried_parts"].values()):
                    raise ValueError("An assembled part remains in resource custody")
                ledger = _complete_target_ledger(products[part])
                if any(record in ledger for record in effects["processCompleted"]):
                    raise ValueError("Duplicate process completion must not advance product history")
                products[part]["processCompleted"] = [*ledger, *deepcopy(effects["processCompleted"])]
                products[part].update({field: deepcopy(value) for field, value in effects.items() if field != "processCompleted"})
                products[part].setdefault("product_effect_evidence", []).append(deepcopy(row))
                changed.add(part)
            cursor += 1
        for part in changed:
            if products[part].get("state") == "assembled" and any(part in held for held in observation["carried_parts"].values()):
                raise ValueError("Post-assembly custody changes require separate supported evidence")
            for field in ("state", "target", "processCompleted", "product_effect_evidence"):
                if field in products[part]:
                    observation["parts"][part][field] = deepcopy(products[part][field])
    if cursor != len(updates):
        raise ValueError("Product-effect acknowledgement has no physical observation boundary")
    for part in changed:
        for field in ("state", "target", "processCompleted", "product_effect_evidence"):
            if field in products[part]:
                result["projected_snapshot"]["parts"][part][field] = deepcopy(products[part][field])
    result.setdefault("evidence", {})["product_effect_evidence"] = deepcopy(updates)
    return result

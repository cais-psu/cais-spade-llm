"""Native task history and replayed physical checkpoints for recovery analysis."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
from cais_spade_llm.agents.central_controller.offline_safety_grounding import (
    _prepare_grounded_primitive_trace,
    _validate_composition_formulas,
)
from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
    _digest,
    _object,
    _symbol,
)
from cais_spade_llm.recovery_framework.environment_composition import state_labels
from cais_spade_llm.resources.primitive_observations import _number


def physical_trace_fingerprint(prepared: dict) -> str:
    """Identify the exact reconstructible trace used by an observed checkpoint."""
    return _digest({key: prepared[key] for key in (
        "fingerprint", "rules", "observations", "valuations", "clock_version",
    )})


def _physical_semantics(prepared: dict) -> str:
    return _digest({"clock_version": prepared["clock_version"], "rules": prepared["rules"],
                    **({"primitive_models": prepared["model"]["evidence"]["primitive_models"]}
                       if prepared.get("model", {}).get("evidence", {}).get("primitive_models") else {}),
                    **{key: prepared["frozen"][key] for key in (
                        "scene", "catalog", "requirement_scopes", "geometry",
                    )}})


def _checkpoint_values(observation: dict) -> dict:
    result = {key: deepcopy(observation[key]) for key in ("resources", "parts")}
    for part in result["parts"].values():
        part.pop("stationary_until", None)
    return result


def _append_product_effects(parts: dict, updates: dict, end: Fraction, writes: set) -> dict:
    effects = {}
    for part, fields in updates.items():
        before = parts.get(part, {}).get("processCompleted")
        after = fields.get("processCompleted", before)
        if after == before:
            continue
        if (not isinstance(before, list) or not isinstance(after, list)
                or len(after) <= len(before) or after[:len(before)] != before):
            raise ValueError("Native product completion must append to its existing processCompleted ledger")
        if (part, end) in writes:
            raise ValueError("Simultaneous product completion effects are ambiguous")
        writes.add((part, end))
        effects[part] = {**{field: deepcopy(fields[field]) for field in ("state", "target") if field in fields},
                         "processCompleted": deepcopy(after[len(before):])}
        parts[part].update(deepcopy(fields))
    return effects


def acknowledged_product_effects_for_comparison(
    expected: dict, observed: dict, acknowledged_task_ids: set[str],
) -> dict:
    """Compare predicted effects only after independently validated acknowledgements.

    The caller owns acknowledgement validation and the frozen initial history.
    Existing acknowledged records must remain exact. Returned data is a detached
    comparison copy; observed history retains its acknowledged provenance.
    """
    if (not isinstance(acknowledged_task_ids, set)
            or any(not isinstance(identity, str) or not identity for identity in acknowledged_task_ids)):
        raise ValueError("Product effect comparison requires validated acknowledgement task identities")
    result = deepcopy(_object(observed, "observed checkpoint"))
    predicted_parts = _object(expected.get("parts"), "predicted parts")
    actual_parts = _object(result.get("parts"), "observed parts")
    for part in set(predicted_parts) | set(actual_parts):
        predicted = _object(predicted_parts.get(part, {}), "predicted part").get("product_effect_evidence", [])
        actual = _object(actual_parts.get(part, {}), "observed part").get("product_effect_evidence", [])
        if not isinstance(predicted, list) or not isinstance(actual, list) or len(predicted) != len(actual):
            raise ValueError("Observed product effects differ from the predicted completion history")
        for planned, completed in zip(predicted, actual, strict=True):
            _object(planned, "predicted product effect")
            _object(completed, "observed product effect")
            required = {"task_id", "resource_id", "time", "kind", "declaration", "product_effects", "source_kind"}
            if (set(planned) != required or set(completed) != required
                    or planned.get("kind") not in {"predicted", "acknowledged"}
                    or completed.get("kind") != "acknowledged"
                    or planned.get("kind") == "predicted" and planned.get("task_id") not in acknowledged_task_ids
                    or {key: value for key, value in planned.items() if key != "kind"}
                    != {key: value for key, value in completed.items() if key != "kind"}):
                raise ValueError("Observed product effects lack their exact validated task acknowledgement")
            completed["kind"] = planned["kind"]
    return result


def physical_history(checkpoint: dict | None, prepared: dict, checker: BaseSafetyChecker,
                     budget, *, primitive_models: dict | None = None) -> tuple[dict, bool]:
    """Replay supplied history; never accept caller-asserted state IDs alone."""
    states = {identifier: row["initial"] for identifier, row in checker.dfas.items()}
    if checkpoint is None:
        return states, False
    checkpoint = _object(checkpoint, "physical_checkpoint")
    required = {"version", "grounding_inputs", "observation_boundaries", "observation_count",
                "trace_fingerprint", "safety_dfa_states"}
    if set(checkpoint) != required or type(checkpoint["version"]) is not int or checkpoint["version"] != 1:
        raise ValueError("Unsupported physical checkpoint")
    prior = _prepare_grounded_primitive_trace(
        **checkpoint["grounding_inputs"], observation_boundaries=checkpoint["observation_boundaries"],
        primitive_models=primitive_models)
    budget.check()
    count = checkpoint["observation_count"]
    if type(count) is not int or not 0 < count <= len(prior["observations"]):
        raise ValueError("Physical checkpoint requires a nonempty contiguous observed prefix")
    if checkpoint["trace_fingerprint"] != physical_trace_fingerprint(prior):
        raise ValueError("Physical checkpoint trace fingerprint changed")
    if _physical_semantics(prior) != _physical_semantics(prepared):
        raise ValueError("Physical checkpoint formulas, AP bindings, geometry or clock changed")
    last, first = prior["observations"][count - 1], prepared["observations"][0]
    if (Fraction(last["time_exact"]) != Fraction(first["time_exact"])
            or _checkpoint_values(last) != _checkpoint_values(first)):
        raise ValueError("Physical checkpoint does not meet the new observed checkpoint")
    for values in prior["valuations"][:count]:
        budget.check()
        for identifier, state in states.items():
            transition = checker.transition_evidence(identifier, state, frozenset(
                label for label, value in values[identifier].items() if value))
            if transition["status"] != "passed":
                raise ValueError("Physical checkpoint contains rejected monitor history")
            states[identifier] = transition["to"]
    if checkpoint["safety_dfa_states"] != states:
        raise ValueError("Physical checkpoint states do not match replayed observations")
    return states, True


class TaskMonitorHistory:
    """Evaluate CCA-owned native task effects on their acknowledgement clock."""

    def __init__(self, context: dict, events: dict, running: dict) -> None:
        context = deepcopy(_object(context, "task_monitor_context"))
        if context.get("additional_scopes"):
            raise ValueError("Additional active recovery safety scopes are unsupported")
        monitor = context.get("monitor")
        if not isinstance(monitor, BaseSafetyChecker) or getattr(monitor, "history_error", None):
            raise ValueError("Native task monitor history is unavailable")
        self.checker = monitor
        if {row.get("id") for row in monitor.safety_rules} != set(monitor.dfas):
            raise ValueError("Native compiled rules and declared formulas disagree")
        if monitor.safety_rules:
            formulas = []
            for row in monitor.safety_rules:
                if "ltlf" in row and "formula" in row and row["ltlf"] != row["formula"]:
                    raise ValueError("Native rule formula is ambiguous")
                formulas.append({"formula": row.get("ltlf", row.get("formula"))})
            _validate_composition_formulas({"specifications": formulas})
        self.jids = _object(context["jids"], "native resource JIDs")
        self.tasks = _object(context["tasks"], "native task bindings")
        event_ids = context.get("event_ids", list(self.tasks))
        if (not isinstance(event_ids, list) or any(not isinstance(value, str) for value in event_ids)
                or len(set(event_ids)) != len(event_ids) or set(event_ids) - set(self.tasks)):
            raise ValueError("Native scope event eligibility is unavailable or ambiguous")
        self.event_ids = frozenset(event_ids)
        self.rules = sorted(monitor.dfas)
        states = _object(context["current_states"], "native DFA states")
        if set(states) != set(self.rules):
            raise ValueError("Native task monitor history omits an active rule")
        if hasattr(monitor, "current_states") and monitor.current_states != states:
            raise ValueError("Native task states disagree with the detached CCA monitor")
        for identifier, state in states.items():
            if state not in monitor.dfas[identifier]["accepting_reachable_states"]:
                raise ValueError("Native task monitor state is unknown or already rejected")
        resources = _object(context["resources"], "native resource values")
        products = _object(context["products"], "native product values")
        contexts = _object(context["contexts"], "native state contexts")
        if set(resources) != set(self.jids) or set(contexts) - set(resources):
            raise ValueError("Native resource bindings are incomplete")
        if set(self.tasks) != set(events) | set(running):
            raise ValueError("Native task bindings must cover every recovery and running event")
        self.labels = {}
        self._bind_tasks(resources, products, events, running)
        running_labels = set().union(*(self.labels[identity] for identity in running if identity in self.event_ids))
        if hasattr(monitor, "running_aps") and set(monitor.running_aps) != running_labels:
            raise ValueError("Native running labels disagree with the admitted running tasks")
        self.initial = {"states": states, "resources": resources, "products": products,
                        "contexts": contexts, "running": sorted(running), "completed": []}
        self.identity = {key: value for key, value in context.items() if key != "monitor"}
        self.identity["monitor"] = {
            "safety_rules": monitor.safety_rules, "dfas": monitor.dfas,
            "tools_catalog": monitor.tools_catalog,
            "resource_bindings": getattr(monitor, "resource_bindings", {}),
        }

    def _bind_tasks(self, resources: dict, products: dict, events: dict, running: dict) -> None:
        task_ids = set()
        for identity, record in {**events, **running}.items():
            row = _object(self.tasks[identity], "native task binding")
            required = {"task", "participants", "resource_updates", "product_updates"}
            if not required <= set(row) or set(row) - required - {"declared_task"}:
                raise ValueError("Native completion effects require explicit task and participant bindings")
            if "declared_task" in row:
                declaration = _object(row["declared_task"], "native declared_task")
                if set(declaration) != {"event_id", "event_name", "parameters"}:
                    raise ValueError("Native declared_task requires an exact declared event and parameters")
            task = _object(row["task"], "native task")
            _symbol(task.get("task_id"), "native task_id")
            if (not task.get("task_id") or task.get("resource_id") != record["resource_id"]
                    or task.get("event_name") != record["event_name"]):
                raise ValueError("Native task identity disagrees with the recovery or running event")
            if task["task_id"] in task_ids:
                raise ValueError("Native completion bindings repeat one task identity")
            task_ids.add(task["task_id"])
            participants = row["participants"]
            if (not isinstance(participants, list) or not participants
                    or len(set(participants)) != len(participants)
                    or set(participants) - set(resources)
                    or task["resource_id"] not in participants):
                raise ValueError("Native task participants are unavailable")
            params = {**_object(task["parameters"], "native task parameters"), "task_id": task["task_id"]}
            if task["parameters"].get("task_id", task["task_id"]) != task["task_id"]:
                raise ValueError("Native task parameters name a different task")
            function_name = _symbol(task.get("function_name", task["event_name"]), "native function_name")
            self.labels[identity] = frozenset(self.checker._map_task_to_aps(
                self.jids[task["resource_id"]], function_name, params))
            for key, declared in (("resource_updates", resources), ("product_updates", products)):
                updates = _object(row[key], "native " + key)
                if set(updates) - set(declared):
                    raise ValueError("Native completion effects reference unknown resources or products")
                if key == "resource_updates" and set(updates) - set(participants):
                    raise ValueError("Native resource effects exceed declared task participants")
                for name, fields in updates.items():
                    if set(_object(fields, "native completion fields")) - set(declared[name]):
                        raise ValueError("Native completion effects introduce unknown fields")

    def _labels(self, state: dict) -> frozenset:
        return state_labels(self.checker, state["resources"], state["products"], self.jids, state["contexts"])

    def _effects(self, state: dict, identity: str) -> dict:
        result = deepcopy(state)
        row, task = self.tasks[identity], self.tasks[identity]["task"]
        for key, destination in (("resource_updates", "resources"), ("product_updates", "products")):
            for identifier, values in row[key].items():
                result[destination][identifier].update(deepcopy(values))
        for resource in row["participants"]:
            result["contexts"][resource] = {**task["parameters"], "task_id": task["task_id"]}
        return result

    def _checks(self, states: dict, labels: frozenset) -> list:
        checks = [self.checker.transition_evidence(rule, states[rule], labels) for rule in self.rules]
        if any(row["reason"] in {"dfa_missing_transition", "dfa_ambiguous_transition"} for row in checks):
            raise ValueError("Native task monitor transition is incomplete")
        return checks

    def start(self, state: dict, identities: tuple) -> tuple[dict, list]:
        result = deepcopy(state)
        if set(identities) & (set(state["running"]) | set(state["completed"])):
            raise ValueError("Native task was already started or completed")
        result["running"] = sorted(set(state["running"]) | set(identities))
        running = frozenset(label for identity in result["running"] if identity in self.event_ids
                            for label in self.labels[identity])
        checks = []
        for identity in identities:
            if identity not in self.event_ids:
                continue
            projected = self._effects(state, identity)
            checks.extend(self._checks(state["states"], running | self._labels(state) | self._labels(projected)))
        return result, checks

    def finish(self, state: dict, identity: str) -> tuple[dict, list]:
        if identity not in state["running"]:
            raise ValueError("Native completion lacks a granted task")
        result = self._effects(state, identity)
        result["running"].remove(identity)
        result["completed"] = sorted([*state["completed"], identity])
        if identity not in self.event_ids:
            return result, []
        labels = self._labels(result) | self.labels[identity] | frozenset(
            label for other in result["running"] if other in self.event_ids for label in self.labels[other])
        checks = self._checks(state["states"], labels)
        result["states"] = {row["rule_id"]: row["to"] for row in checks}
        return result, checks

    def pending(self, state: dict) -> list:
        return [rule for rule in self.rules
                if state["states"][rule] not in self.checker.dfas[rule]["accepting_states"]]


class TaskMonitorHistories:
    """Keep exact CCA scope identities and their separate AP namespaces."""

    def __init__(self, context: dict, events: dict, running: dict) -> None:
        context = deepcopy(_object(context, "task_monitor_context"))
        if "monitors" in context:
            if "monitor" in context or "current_states" in context:
                raise ValueError("Native monitor context has ambiguous scope records")
            records = context.pop("monitors")
        else:
            records = [{"scope_id": None, "monitor": context.pop("monitor"),
                        "current_states": context.pop("current_states")}]
        if not isinstance(records, list) or not records:
            raise ValueError("Native task history requires an explicit main monitor")
        self.monitors = []
        scopes = set()
        for record in records:
            if (not {"scope_id", "monitor", "current_states"} <= set(_object(record, "native scope"))
                    or set(record) - {"scope_id", "monitor", "current_states", "event_ids"}):
                raise ValueError("Native scopes require exact identities and monitor states")
            scope = record["scope_id"]
            if scope is not None and (not isinstance(scope, str) or not scope):
                raise ValueError("Native scope identity is unavailable")
            if scope in scopes:
                raise ValueError("Duplicate native safety scope")
            scopes.add(scope)
            history = TaskMonitorHistory({**context, **{key: record[key] for key in (
                "monitor", "current_states", "event_ids",
            ) if key in record}}, events, running)
            self.monitors.append((scope, history))
        if None not in scopes:
            raise ValueError("Native task history omits the main CCA monitor")
        self.initial = {"running": sorted(running), "completed": [], "monitors": [
            {"scope_id": scope, "state": deepcopy(history.initial)} for scope, history in self.monitors]}
        self.identity = [{"scope_id": scope, "context": history.identity} for scope, history in self.monitors]
        self.tasks = self.monitors[0][1].tasks

    def _apply(self, operation: str, state: dict, argument) -> tuple[dict, list]:
        result, checks = deepcopy(state), []
        for index, (scope, history) in enumerate(self.monitors):
            new_state, evidence = getattr(history, operation)(state["monitors"][index]["state"], argument)
            result["monitors"][index]["state"] = new_state
            checks.extend({"scope_id": scope, **row} for row in evidence)
        result["running"] = deepcopy(result["monitors"][0]["state"]["running"])
        result["completed"] = deepcopy(result["monitors"][0]["state"]["completed"])
        return result, checks

    def start(self, state: dict, identities: tuple) -> tuple[dict, list]:
        return self._apply("start", state, identities)

    def finish(self, state: dict, identity: str) -> tuple[dict, list]:
        return self._apply("finish", state, identity)

    def pending(self, state: dict) -> list:
        return [{"scope_id": scope, "rule_id": rule}
                for index, (scope, history) in enumerate(self.monitors)
                for rule in history.pending(state["monitors"][index]["state"])]

    def validate_product_effect_evidence(self, inputs: dict, choice: dict, intervals: dict) -> None:
        """Bind projected process effects to the owner's exact native completions."""
        ledger = choice.get("product_effect_evidence", inputs.get("product_effect_evidence"))
        updates = (ledger or {}).get("updates", [])
        if not isinstance(updates, list):
            raise ValueError("Product completion evidence requires explicit updates")
        evidence = {}
        for update in updates:
            task_id = _symbol(_object(update, "product effect").get("task_id"), "product effect task_id")
            if task_id in evidence:
                raise ValueError("Product completion evidence repeats a task")
            evidence[task_id] = update
        task_ledger = choice.get("task_evidence", inputs.get("task_evidence"))
        task_rows = (task_ledger or {}).get("events", [])
        if not isinstance(task_rows, list):
            raise ValueError("Product completion evidence requires its task ledger")
        tasks = {row["task_id"]: row for row in task_rows}
        parts = deepcopy(inputs["snapshot"]["parts"])
        used, writes = set(), set()
        for identity in sorted(intervals, key=lambda key: (intervals[key][1], key)):
            binding = self.tasks[identity]
            task, end = binding["task"], intervals[identity][1]
            effects = _append_product_effects(parts, binding["product_updates"], end, writes)
            if not effects:
                continue
            task_id = task["task_id"]
            update, observed_task = evidence.get(task_id), tasks.get(task_id)
            if update is None or observed_task is None:
                raise ValueError("Native product completion lacks explicit task and product effect evidence")
            declaration = binding.get("declared_task")
            if declaration is None and {"event_id", "event_name", "parameters"} <= set(task):
                declaration = {key: task[key] for key in ("event_id", "event_name", "parameters")}
            if declaration is None or update.get("declaration") != declaration:
                raise ValueError("Product completion evidence disagrees with the native declared_task")
            if (update.get("resource_id") != task["resource_id"]
                    or _number(update.get("time"), "product effect time") != end
                    or update.get("product_effects") != effects
                    or observed_task.get("resource_id") != task["resource_id"]
                    or observed_task.get("function") != task.get("function_name", task["event_name"])
                    or _number(observed_task.get("start_time"), "task start") != intervals[identity][0]
                    or _number(observed_task.get("end_time"), "task end") != end
                    or observed_task.get("declared_task", declaration) != declaration):
                raise ValueError("Product completion evidence disagrees with its native task or exact effects")
            used.add(task_id)
        if set(evidence) != used:
            raise ValueError("Product completion evidence contains an undeclared native effect")

    def projected_process_values(self, parts: dict, intervals: dict, time: Fraction) -> dict:
        """Project only declared process fields whose native tasks have completed."""
        projected = deepcopy(parts)
        for identity in sorted(intervals, key=lambda key: (intervals[key][1], key)):
            if intervals[identity][1] > time:
                continue
            for part, fields in self.tasks[identity]["product_updates"].items():
                if part in projected:
                    projected[part].update({key: deepcopy(value) for key, value in fields.items()
                                           if key in {"processCompleted", "processCompleted_complete",
                                                      "processCompleted_evidence"}})
        return projected

    def validate_physical_effects(self, schedules: list) -> None:
        """Reject native custody/effects contradicting the same physical branch."""
        physical_fields = {
            "resources": {"held_part", "gripper_state", "current_pose", "base_pose", "grasp_transform", "frame", "contained_parts"},
            "parts": {"current_pose", "contained_by", "frame", "processCompleted", "processCompleted_complete"},
        }

        def agree(kind: str, values: dict, observed: dict) -> None:
            for identifier, fields in values.items():
                for field in set(fields) & physical_fields[kind]:
                    if field not in observed.get(identifier, {}) or fields[field] != observed[identifier][field]:
                        raise ValueError("Native task custody or effects disagree with physical observations")

        first = self.monitors[0][1].initial
        for schedule in schedules:
            observations = schedule["prepared"]["observations"]
            agree("resources", first["resources"], observations[0]["resources"])
            agree("parts", first["products"], observations[0]["parts"])
            for identity, row in self.tasks.items():
                index = schedule["times"].index(schedule["intervals"][identity][1])
                agree("resources", row["resource_updates"], observations[index]["resources"])
                agree("parts", row["product_updates"], observations[index]["parts"])

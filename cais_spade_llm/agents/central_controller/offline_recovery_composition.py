from __future__ import annotations

"""Bounded offline composition of explicitly evidenced recovery schedules.

Only complete recovery-event starts and supplied waiting alternatives are
controllable. Every joint observation after a choice is forced. The finite
schedules are evidence, not a timing synthesizer or an execution permission.
"""

import json
from collections import deque
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
from typing import Any

from cais_spade_llm.agents.central_controller._recovery_monitor_history import (
    TaskMonitorHistories,
    continuous_transition,
    physical_history,
    physical_trace_fingerprint,
)
from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
from cais_spade_llm.agents.central_controller.local_composition import (
    Action,
    Analysis,
    AnalysisLimit,
    Budget,
    Edge,
    Node,
    Scope,
    nonblocking_region,
)
from cais_spade_llm.agents.central_controller.offline_safety_grounding import (
    _prepare_grounded_primitive_trace,
    _validate_composition_formulas,
)
from cais_spade_llm.agents.central_controller.region_admission import (
    event_region_relevance,
    selected_region_mutexes,
)
from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
    _digest,
    _list,
    _object,
    _symbol,
)
from cais_spade_llm.resources.primitive_observations import _interval, _number

_CLOCK_VERSION = "composition_v1"
_NOMINAL_FIELDS = (
    "nominal_tasks", "task_statuses", "resume_entry_task_ids_by_resource",
    "resumable_task_ids_by_resource",
)
_PROCESS_FIELDS = ("processCompleted", "processCompleted_complete", "processCompleted_evidence")


def _records(value: Any, *, running: bool) -> dict:
    result = {}
    identity = "task_id" if running else "outline_id"
    required = {identity, "resource_id", "event_name", "primitive_step_indices"}
    required |= {"start_time", "end_time"} if running else {"predecessors"}
    for raw in _list(value, "running_work" if running else "recovery_events"):
        row = _object(raw, "event record")
        if not required <= set(row) or set(row) - required - (set() if running else {"des_event_id"}):
            raise ValueError("Event records require exact identities and primitive-step membership")
        for field in (identity, "resource_id", "event_name"):
            _symbol(row[field], field)
        if row[identity] in result:
            raise ValueError("Duplicate event identity")
        indices = _list(row["primitive_step_indices"], "primitive_step_indices")
        if (not indices or any(type(index) is not int or index < 0 for index in indices)
                or indices != list(range(indices[0], indices[-1] + 1))):
            raise ValueError("Each event requires a nonempty contiguous primitive-step interval")
        if not running:
            if "des_event_id" in row:
                _symbol(row["des_event_id"], "des_event_id")
            predecessors = _list(row["predecessors"], "predecessors")
            if len(set(predecessors)) != len(predecessors):
                raise ValueError("Duplicate recovery predecessor")
            for predecessor in predecessors:
                _symbol(predecessor, "predecessor")
        result[row[identity]] = deepcopy(row)
    if not running:
        for identifier, row in result.items():
            if identifier in row["predecessors"] or set(row["predecessors"]) - set(result):
                raise ValueError("Recovery predecessors must name other supplied events")
    return result


def _subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _subset(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            _subset(first, second) for first, second in zip(expected, actual, strict=True)
        )
    if type(expected) is bool or type(actual) is bool:
        return type(expected) is type(actual) and expected == actual
    return expected == actual


def _physical(observation: dict) -> dict:
    state = {name: deepcopy(observation[name]) for name in (
        "resources", "parts", "region_occupancy", "part_region_occupancy", "carried_parts",
    )}
    # This is a future coverage assertion, not an observed part-state variable.
    for part in state["parts"].values():
        part.pop("stationary_until", None)
    return state


def _task_state(prepared: dict, time: Fraction) -> dict:
    frozen = prepared["frozen"]
    tasks = (frozen.get("task_evidence") or {}).get("events", [])
    active = [{key: value for key, value in row.items() if key not in {"start_time", "end_time"}}
              for row in tasks if _number(row["start_time"], "task start") <= time
              < _number(row["end_time"], "task end")]
    ledger = frozen.get("state_evidence") or {}
    states = {row["resource_id"]: deepcopy(row) for row in ledger.get("initial", [])}
    for row in sorted(ledger.get("updates", []), key=lambda item: item["time"]):
        if _number(row["time"], "state time") <= time:
            previous = states[row["resource_id"]]
            states[row["resource_id"]] = {
                **deepcopy(row), "values": {**previous["values"], **row["values"]},
            }
    return {"active_tasks": sorted(active, key=lambda row: row["task_id"]), "states": states}


def _event_source(trace: dict, row: dict, identity: str, local_index: int, *, running: bool) -> None:
    source = _object(trace.get("source"), "step source")
    field = "task_id" if running else "outline_id"
    if source.get(field) != identity or source.get("event_name") != row["event_name"]:
        raise ValueError("Primitive provenance disagrees with its exact event identity")
    if type(source.get("step_index")) is not int or source["step_index"] != local_index:
        raise ValueError("Primitive provenance has an inconsistent local step index")
    if not running:
        des_event_id = _symbol(source.get("des_event_id"), "source.des_event_id")
        if "des_event_id" in row and des_event_id != row["des_event_id"]:
            raise ValueError("Primitive provenance has an inconsistent des_event_id")


def _intervals(choice: dict, events: dict, running: dict, horizon: tuple) -> dict:
    programs = {row["resource_id"]: row for row in choice["programs"]}
    if len(programs) != len(choice["programs"]):
        raise ValueError("Duplicate resource program")
    covered, intervals = set(), {}
    for identity, row in {**events, **running}.items():
        resource = row["resource_id"]
        if resource not in programs:
            raise ValueError("Every recovery event and running task requires its complete program")
        traces = programs[resource]["step_results"]
        indices = row["primitive_step_indices"]
        if indices[-1] >= len(traces):
            raise ValueError("Event primitive-step membership exceeds its resource program")
        for local_index, index in enumerate(indices):
            if (resource, index) in covered:
                raise ValueError("A primitive step belongs to multiple events")
            covered.add((resource, index))
            _event_source(traces[index], row, identity, local_index, running=identity in running)
        if identity in events and len({traces[index]["source"]["des_event_id"] for index in indices}) != 1:
            raise ValueError("One recovery event has inconsistent des_event_id provenance")
        start = _number(traces[indices[0]]["start_time"], "event start")
        end = _number(traces[indices[-1]]["end_time"], "event end")
        if not horizon[0] <= start < end <= horizon[1]:
            raise ValueError("Event evidence needs a positive complete interval within the horizon")
        if identity in events:
            if start != _number(choice["starts"][identity], "declared event start"):
                raise ValueError("A schedule start disagrees with its authored primitive timing")
        elif (not _number(row["start_time"], "running start") <= horizon[0] <= start
              or end != _number(row["end_time"], "running end")):
            raise ValueError("Running work must already be started and retain its fixed completion")
        intervals[identity] = (start, end)
    expected = {(resource, index) for resource, program in programs.items()
                for index in range(len(program["primitive_steps"]))}
    if covered != expected:
        raise ValueError("Every primitive requires exact recovery-event or running-task membership")
    for identity, row in events.items():
        if any(intervals[prior][1] > intervals[identity][0] for prior in row["predecessors"]):
            raise ValueError("A recovery event starts before its predecessor completes")
    return intervals


def _relative_steps(schedule: dict, identity: str, event: dict) -> list:
    start = schedule["intervals"][identity][0]
    program = next(row for row in schedule["choice"]["programs"]
                   if row["resource_id"] == event["resource_id"])

    def without_future_coverage(value):
        if isinstance(value, dict):
            return {key: without_future_coverage(item) for key, item in value.items()
                    if key != "stationary_until"}
        if isinstance(value, list):
            return [without_future_coverage(item) for item in value]
        return value

    traces = []
    for index in event["primitive_step_indices"]:
        trace = without_future_coverage(program["step_results"][index])
        for field in ("start_time", "end_time"):
            trace[field] = str(_number(trace[field], field) - start)
        for field in ("trajectory", "base_trajectory"):
            for point in trace["model_evidence"].get(field, []):
                point["time"] = str(_number(point["time"], "trajectory time") - start)
        traces.append(trace)
    return traces


def _fixed_ledgers(schedule: dict, events: dict, task_history: TaskMonitorHistories | None) -> dict:
    frozen = schedule["prepared"]["frozen"]
    task_evidence, state_evidence, product_effect_evidence = (deepcopy(frozen.get(field)) for field in (
        "task_evidence", "state_evidence", "product_effect_evidence",
    ))
    identities = ({row["task"]["task_id"]: identity for identity, row in task_history.tasks.items()}
                  if task_history else {})
    for task in (task_evidence or {}).get("events", []):
        identity = identities.get(task["task_id"], task["task_id"])
        if identity not in events:
            continue
        event = events[identity]
        start, end = schedule["intervals"][identity]
        native = task_history.tasks[identity]["task"] if task_history else None
        function = native.get("function_name", native["event_name"]) if native else event["event_name"]
        if (task["resource_id"] != event["resource_id"] or task["function"] != function
                or _number(task["start_time"], "task start") != start
                or _number(task["end_time"], "task end") != end):
            raise ValueError("A recovery task ledger must bind its exact event identity and interval")
        task["start_time"], task["end_time"] = "0", str(end - start)
    for ledger in (state_evidence, product_effect_evidence):
        for update in (ledger or {}).get("updates", []):
            identity = identities.get(update["task_id"], update["task_id"])
            if identity in events:
                update["time"] = str(_number(update["time"], "completion update") - schedule["intervals"][identity][0])
    return {"task_evidence": task_evidence, "state_evidence": state_evidence,
            "product_effect_evidence": product_effect_evidence}


def _equivalent_prefixes(prepared: list, events: dict, running: dict, budget: Budget,
                         task_history: TaskMonitorHistories | None) -> None:
    ledgers = [_digest(_fixed_ledgers(schedule, events, task_history)) for schedule in prepared]
    if len(set(ledgers)) != 1:
        raise ValueError("Scheduling choices change fixed task/state evidence or admitted completion effects")
    for first, second in combinations(prepared, 2):
        budget.check()
        for identity, event in events.items():
            if _relative_steps(first, identity, event) != _relative_steps(second, identity, event):
                raise ValueError("A scheduling choice changes an admitted event's relative timing or effects")
        differing = [identity for identity in events
                     if first["intervals"][identity][0] != second["intervals"][identity][0]]
        split = min((min(first["intervals"][identity][0], second["intervals"][identity][0])
                     for identity in differing), default=None)
        left = [row for row in first["semantic"] if split is None or Fraction(row["time_exact"]) < split]
        right = [row for row in second["semantic"] if split is None or Fraction(row["time_exact"]) < split]
        if left != right:
            raise ValueError("Scheduling alternatives disagree before their first differing controllable start")
        for work in running.values():
            resource = work["resource_id"]
            programs = []
            for schedule in (first, second):
                program = next(item for item in schedule["choice"]["programs"] if item["resource_id"] == resource)
                programs.append([program["step_results"][index] for index in work["primitive_step_indices"]])
            if programs[0] != programs[1]:
                raise ValueError("Already-running evidence cannot depend on a future recovery choice")


def _base_programs(inputs: dict, choices: list) -> None:
    for base in _list(inputs.get("programs", []), "grounding_inputs.programs"):
        resource = base["resource_id"]
        for choice in choices:
            current = next((row for row in choice["programs"] if row["resource_id"] == resource), None)
            if current is None or current["primitive_steps"] != base["primitive_steps"]:
                raise ValueError("Schedules disagree with the frozen base primitive program")
            if [row.get("source") for row in current["step_results"]] != [
                row.get("source") for row in base["step_results"]
            ]:
                raise ValueError("Schedules disagree with frozen base primitive provenance")


def _completion_fields(inputs: dict, completion: dict, primitive_models: dict | None = None) -> None:
    if set(completion) != {"resources", "parts"} or not any(completion.values()):
        raise ValueError("Completion requires explicit resource and part conditions")
    allowed = {
        "resources": {"frame", "current_pose", "base_pose", "held_part", "gripper_state",
                      "grasp_transform", "contained_parts"},
        "parts": {"frame", "current_pose", "contained_by", "state", "target", *_PROCESS_FIELDS},
    }
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(inputs["scene"]) if primitive_models else {}
    for kind in ("resources", "parts"):
        for identifier, fields in _object(completion[kind], "completion." + kind).items():
            if identifier not in inputs["snapshot"][kind] or not _object(fields, "completion fields"):
                raise ValueError("Completion must refer to declared resource or part fields")
            native = set()
            if kind == "resources" and identifier in (primitive_models or {}):
                declarations = _object(primitive_models[identifier].descriptor()["configuration"].get("state_variables", {}),
                                       "owner state declarations")
                if any(models.get(identifier, {}).get("state_variables", {}).get(key) != value
                       for key, value in declarations.items()):
                    raise ValueError("Completion requires configured owner state declarations")
                native = {field for field in fields if field in declarations or (
                    "." in field and field.split(".", 1)[0] + ".{part_name}" in declarations)}
            if set(fields) - allowed[kind] - native or set(fields) - set(inputs["snapshot"][kind][identifier]):
                raise ValueError("Completion contains an unsupported modeled checkpoint field")


def _resource_provenance(inputs: dict, choice: dict) -> None:
    for program in choice["programs"]:
        resource = program["resource_id"]
        expected_jid = inputs["snapshot"]["resources"].get(resource, {}).get("resource_jid")
        for trace in program["step_results"]:
            source = trace["source"]
            if ("resource_id" in source and source["resource_id"] != resource
                    or expected_jid is not None and "resource_jid" in source and source["resource_jid"] != expected_jid):
                raise ValueError("Primitive provenance disagrees with its resource association")


def _preserved_checkpoint(inputs: dict, prepared: dict, intervals: dict,
                          task_history: TaskMonitorHistories | None) -> None:
    for field in _NOMINAL_FIELDS:
        if prepared["projected_snapshot"].get(field) != inputs["snapshot"].get(field):
            raise ValueError("Recovery evidence changes preserved nominal task metadata")
    for observation in prepared["observations"]:
        expected = inputs["snapshot"]["parts"]
        if task_history:
            expected = task_history.projected_process_values(
                expected, intervals, Fraction(observation["time_exact"]))
        for part, values in expected.items():
            if any(observation["parts"][part].get(field) != values.get(field) for field in _PROCESS_FIELDS):
                raise ValueError("Recovery evidence changes processCompleted without its declared native completion")


def _prepare(inputs: dict, choices: Any, events: dict, running: dict,
             completion: dict, budget: Budget,
             task_history: TaskMonitorHistories | None, primitive_models: dict | None = None) -> tuple[list, list, str]:
    if "allow_predicted_product_effects" in inputs:
        raise ValueError("Predicted product effects require native completion validation")
    _validate_composition_formulas(inputs["catalog"])
    horizon = _interval(inputs["horizon"], "horizon")
    if horizon[0] >= horizon[1] or not events or set(events) & set(running):
        raise ValueError("Composition requires a positive horizon and distinct recovery/running identities")
    _completion_fields(inputs, completion, primitive_models)
    raw_choices = _list(choices, "event_start_choices")
    if not raw_choices:
        raise ValueError("At least one completely evidenced scheduling alternative is required")
    identifiers, decision_times = set(), {horizon[0]}
    authored, prepared = None, []
    for choice in raw_choices:
        budget.check()
        _object(choice, "schedule")
        if set(choice) - {"id", "starts", "programs", "stationary", "task_evidence", "state_evidence",
                          "product_effect_evidence"}:
            raise ValueError("Scheduling alternatives cannot change shared grounding inputs")
        if not {"id", "starts", "programs", "stationary"} <= set(choice):
            raise ValueError("Every schedule requires explicit starts, programs and stationary coverage")
        identifier = _symbol(choice["id"], "schedule.id")
        if identifier in identifiers or set(_object(choice["starts"], "starts")) != set(events):
            raise ValueError("Schedules require distinct IDs and all recovery-event starts")
        identifiers.add(identifier)
        _list(choice["programs"], "programs")
        current = {row["resource_id"]: row["primitive_steps"] for row in choice["programs"]}
        if authored is not None and current != authored:
            raise ValueError("Scheduling choices must preserve all authored primitives and parameters")
        authored = current
        intervals = _intervals(choice, events, running, horizon)
        if task_history:
            task_history.validate_product_effect_evidence(inputs, choice, intervals)
        elif choice.get("product_effect_evidence", inputs.get("product_effect_evidence")) is not None:
            raise ValueError("Product completion effects require a native task monitor context")
        _resource_provenance(inputs, choice)
        decision_times.update(start for identity, (start, _) in intervals.items() if identity in events)
        prepared.append({"id": identifier, "choice": deepcopy(choice), "intervals": intervals})
    _base_programs(inputs, raw_choices)
    # These times originate in explicit numeric schedule inputs. Geometric
    # crossing times stay as exact rational strings inside each prepared trace.
    numeric_times = [inputs["horizon"][0], *(time for choice in raw_choices for time in choice["starts"].values())]
    for row in prepared:
        budget.check()
        call = deepcopy(inputs)
        call.update({name: deepcopy(row["choice"][name]) for name in (
            "programs", "stationary", "task_evidence", "state_evidence", "product_effect_evidence",
        ) if name in row["choice"]})
        row["prepared"] = _prepare_grounded_primitive_trace(
            **call, observation_boundaries=numeric_times,
            allow_predicted_product_effects=task_history is not None, primitive_models=primitive_models,
            motion_budget=budget)
        budget.check()
        value = row["prepared"]
        row["times"] = [Fraction(item["time_exact"]) for item in value["observations"]]
        row["semantic"] = [
            {"time_exact": observation["time_exact"], "physical": _physical(observation),
             "task_state": _task_state(value, time), "valuations": value["valuations"][index]}
            for index, (observation, time) in enumerate(zip(value["observations"], row["times"], strict=True))
        ]
        row["region_relevance"] = {
            identity: event_region_relevance(value, event["resource_id"], row["intervals"][identity])
            for identity, event in {**events, **running}.items()
        }
        _preserved_checkpoint(inputs, value, row["intervals"], task_history)
    _equivalent_prefixes(prepared, events, running, budget, task_history)
    problem_id = _digest({"clock_version": _CLOCK_VERSION, "grounding_inputs": inputs,
                          "recovery_events": events, "running_work": running,
                          "event_start_choices": raw_choices, "completion": completion,
                          "prepared_traces": [{"rules": row["prepared"]["rules"],
                                               "observations": row["prepared"]["observations"],
                                               "valuations": row["prepared"]["valuations"],
                                               "clock_version": row["prepared"]["clock_version"],
                                               **({"primitive_models": row["prepared"]["model"]["evidence"]["primitive_models"],
                                                   "owner_effects": row["prepared"]["model"]["evidence"].get("owner_effects", [])}
                                                  if row["prepared"]["model"].get("evidence", {}).get("primitive_models") else {})}
                                              for row in prepared]})
    return prepared, sorted(decision_times), problem_id


class _Composition:
    def __init__(self, inputs: dict, events: dict, running: dict, schedules: list,  # noqa: PLR0913
                 decisions: list, completion: dict, budget: Budget,
                 task_history: TaskMonitorHistories | None = None,
                 physical_checkpoint: dict | None = None, primitive_models: dict | None = None) -> None:
        self.inputs, self.events, self.running_work = inputs, events, running
        self.schedules, self.decisions = schedules, decisions
        self.completion, self.budget = completion, budget
        first = schedules[0]["prepared"]
        self.continuous = bool(first['model'].get('evidence', {}).get('continuous_motion'))
        self.uncertain = set()
        self.definite_schedules = set()
        self.rules = first["rules"]
        self.rule_ids = [rule["rule_id"] for rule in self.rules]
        self.checker = BaseSafetyChecker({row["rule_id"]: row["dfa_dot"] for row in self.rules}, self.rules)
        self.task_history = task_history
        if self.task_history:
            self.task_history.validate_physical_effects(schedules)
        physical_states, self.skip_initial_observation = physical_history(
            physical_checkpoint, first, self.checker, budget, primitive_models=primitive_models)
        pending = [identifier for identifier, status in inputs["snapshot"].get("task_statuses", {}).items()
                   if status == "pending"]
        scope = Scope(resources=set(first["models"]), products=set(inputs["snapshot"]["parts"]),
                      rules=set(self.rule_ids), tasks=set(events) | set(running) | set(pending),
                      goals=[deepcopy(completion)], reasons=[{"kind": "complete_frozen_scene"}])
        scope.task_bindings = {**deepcopy(events), **deepcopy(running)}
        self.analysis = Analysis("inconclusive", "analysis_incomplete", scope)
        self.details, self.edge_ids, self.node_ids = {}, {}, {}
        self.marked, self.pending = set(), {}
        self.task_pending = {}
        self.terminal_schedule = {}
        self.decision_nodes = set()
        monitors = tuple(physical_states[identifier] for identifier in self.rule_ids)
        state = {"resources": deepcopy(inputs["snapshot"]["resources"]),
                 "parts": deepcopy(inputs["snapshot"]["parts"])}
        self.root = self._node(state, monitors, list(range(len(schedules))), decisions[0], "decision",
                               self.task_history.initial if self.task_history else None)
        self.analysis.root = self.root

    def _node(self, state: dict, monitors: tuple, choices: list, time: Fraction, phase: str,
              task_state: dict | None = None):
        self.budget.check(len(self.analysis.nodes))
        schedule = self.schedules[choices[0]]
        active = []
        for identity, row in {**self.events, **self.running_work}.items():
            start, end = schedule["intervals"][identity]
            started = start < time or phase != "decision" and start == time or identity in self.running_work
            if started and time < end:
                active.append(Action(identity, deepcopy(row), frozenset({row["resource_id"]}), frozenset()))
        physical = deepcopy(state)
        if phase == "decision":
            endpoints = []
            for index in choices:
                candidate = self.schedules[index]
                observation_index = candidate["times"].index(time)
                endpoints.append(_physical(candidate["prepared"]["observations"][observation_index]))
            if any(value != endpoints[0] for value in endpoints[1:]):
                raise ValueError("A controllable start depends on incompatible current physical checkpoints")
            # Expose the observed endpoint for freshness checks without ticking
            # a DFA before the chosen start establishes this boundary's APs.
            physical.update(endpoints[0])
        if task_state is not None:
            physical["task_monitor"] = deepcopy(task_state)
        for part in physical["parts"].values():
            part.pop("stationary_until", None)
        physical.update(time_exact=str(time), phase=phase, compatible_choices=choices)
        key = (_digest(physical), tuple(sorted(action.key for action in active)), monitors)
        if key not in self.analysis.nodes:
            self.analysis.nodes[key] = Node(physical, tuple(active), monitors)
            self.analysis.graph[key] = []
            self.node_ids[key] = _digest(key)
        if phase == "decision":
            self.decision_nodes.add(key)
        return key

    def _edge(self, source, target, action: str, kind: str, controllable: bool, **details) -> None:
        edge = Edge(target, action, kind, controllable)
        self.analysis.graph[source].append(edge)
        identifier = _digest([self.node_ids[source], self.node_ids.get(target), action, kind])
        self.edge_ids[(source, edge)] = identifier
        self.details[identifier] = {
            "edge_id": identifier, "source": self.node_ids[source], "target": self.node_ids.get(target),
            "action": action, "kind": kind, "controllable": controllable, **details,
        }

    def _completion_nodes(self, node, indices: list, time: Fraction) -> list:
        if self.task_history is None:
            return [node]
        state = self.analysis.nodes[node].state["task_monitor"]
        schedule = self.schedules[indices[0]]
        due = [identity for identity in state["running"]
               if schedule["intervals"][identity][1] == time]
        if not due:
            return [node]
        result = []
        # The acknowledgement order is not a controller choice. Retain every
        # possible order, including successors that violate a native rule.
        for identity in sorted(due):
            self.budget.check(len(self.analysis.nodes))
            updated, checks = self.task_history.finish(state, identity)
            violation = next((row for row in checks if row["status"] != "passed"), None)
            current = self.analysis.nodes[node]
            target = None if violation else self._node(
                current.state, current.monitors, indices, time, current.state["phase"], updated)
            task = self.task_history.tasks[identity]
            self._edge(node, target, identity, "task_completion", False,
                       event_id=identity, task_id=task["task"]["task_id"], time=float(time), time_exact=str(time),
                       task_rule_checks=checks, task_monitor_state=deepcopy(updated),
                       resource_updates=deepcopy(task["resource_updates"]),
                       product_updates=deepcopy(task["product_updates"]), violation=violation)
            if target is not None:
                result.extend(self._completion_nodes(target, indices, time))
        return list(dict.fromkeys(result))

    def build(self, node=None, choices=None, decision_index: int = 0) -> None:
        node = self.root if node is None else node
        choices = list(range(len(self.schedules))) if choices is None else choices
        time = self.decisions[decision_index]
        completed = self._completion_nodes(node, choices, time)
        if completed != [node]:
            for successor in completed:
                self.build(successor, choices, decision_index)
            return
        groups = {}
        for index in choices:
            starts = tuple(sorted(identity for identity in self.events
                                  if self.schedules[index]["intervals"][identity][0] == time))
            groups.setdefault(starts, []).append(index)
        for starts, indices in sorted(groups.items()):
            self.budget.check(len(self.analysis.nodes))
            monitors = self.analysis.nodes[node].monitors
            task_state = self.analysis.nodes[node].state.get("task_monitor")
            checks = []
            if self.task_history:
                task_state, checks = self.task_history.start(task_state, starts)
            violation = next((row for row in checks if row["status"] != "passed"), None)
            after = None if violation else self._node(
                self.analysis.nodes[node].state, monitors, indices, time, "progress", task_state)
            action = json.dumps({"time_exact": str(time), "event_ids": starts}, separators=(",", ":"))
            self._edge(node, after, action, "start" if starts else "wait", True,
                       time=float(time), time_exact=str(time), event_ids=list(starts),
                       **({"task_rule_checks": checks, "task_monitor_state": deepcopy(task_state),
                           "violation": violation} if self.task_history else {}))
            if after is not None:
                self._progress(after, indices, decision_index)

    def _progress(self, after, indices: list, decision_index: int, start_index: int = 0) -> None:
        time = self.decisions[decision_index]
        monitors = self.analysis.nodes[after].monitors
        schedule = self.schedules[indices[0]]
        prepared = schedule["prepared"]
        next_time = self.decisions[decision_index + 1] if decision_index + 1 < len(self.decisions) else None
        for index in range(start_index, len(schedule["times"])):
            observed_time = schedule["times"][index]
            if (observed_time < time or next_time is not None and observed_time >= next_time
                    or index == 0 and self.skip_initial_observation):
                continue
            observation = prepared["observations"][index]
            if self.continuous:
                next_monitors, checks, violation, uncertainty = self._continuous_transition(
                    monitors, prepared['valuations'][index], observation)
                if uncertainty:
                    self.uncertain.add(after)
                    self._edge(after, None, observation['time_exact'] + ':uncertainty', 'observation', False,
                               observation_index=index, schedule_id=schedule['id'],
                               time=observation['time'], time_exact=observation['time_exact'],
                               active_steps=deepcopy(observation['active_steps']),
                               violation=uncertainty, certainty='unresolved')
                if violation:
                    self.definite_schedules.update(indices)
            else:
                next_monitors, checks, violation = [], [], None
            for identifier, current in zip(self.rule_ids, monitors, strict=True):
                if self.continuous:
                    break
                values = prepared["valuations"][index][identifier]
                transition = self.checker.transition_evidence(
                    identifier, current, frozenset(label for label, value in values.items() if value))
                checks.append({"rule_id": identifier, "ap_values": deepcopy(values),
                               "transition": deepcopy(transition)})
                if transition["reason"] in {"dfa_missing_transition", "dfa_ambiguous_transition"}:
                    raise ValueError("Incomplete monitor transition in composition")
                if transition["status"] != "passed":
                    violation = violation or checks[-1]
                next_monitors.append(transition["to"])
            target = None if violation else self._node(
                _physical(observation), tuple(next_monitors), indices, observed_time, "observation",
                self.analysis.nodes[after].state.get("task_monitor"))
            self._edge(after, target, observation["time_exact"], "observation", False,
                       time=observation["time"], time_exact=observation["time_exact"],
                       observation_index=index, schedule_id=schedule["id"], rule_checks=checks,
                       active_steps=deepcopy(observation["active_steps"]), violation=violation,
                       observation=deepcopy(observation),
                       **({'certainty':'definite'} if self.continuous and violation else {}))
            if target is None:
                break
            after, monitors = target, tuple(next_monitors)
            completed = self._completion_nodes(after, indices, observed_time)
            if completed != [after]:
                for successor in completed:
                    self._progress(successor, indices, decision_index, index + 1)
                return
        else:
            if next_time is not None:
                target = self._node(self.analysis.nodes[after].state, monitors, indices, next_time, "decision")
                self._edge(after, target, str(next_time), "decision", False,
                           time=float(next_time), time_exact=str(next_time))
                self.build(target, indices, decision_index + 1)
            else:
                snapshot = prepared["projected_snapshot"]
                if all(_subset(self.completion[kind], snapshot[kind]) for kind in ("resources", "parts")):
                    pending = self.pending_states(monitors)
                    task_pending = (self.task_history.pending(self.analysis.nodes[after].state["task_monitor"])
                                    if self.task_history else [])
                    if pending or task_pending:
                        self.pending[after] = pending
                        self.task_pending[after] = task_pending
                    else:
                        self.marked.add(after)
                        self.terminal_schedule[after] = indices[0]

    def pending_states(self, monitors):
        """Every possible physical monitor state must accept the same endpoint."""
        return [identifier for identifier, current in zip(self.rule_ids, monitors, strict=True)
                if any(state not in self.checker.dfas[identifier]['accepting_states']
                       for state in (current if self.continuous else (current,)))]

    def _continuous_transition(self, monitors, values, observation):
        """Universally retain bounded physical words on the same branch search.

        An open cell permits any finite nonempty word over its bounded alphabet.
        Closure over DFA states overapproximates unknown crossing order/count.
        No numerical refinement step is emitted as a task or physical clock tick.
        """
        return continuous_transition(
            self.checker, self.rule_ids, monitors, values, observation, self.budget,
            explored_states=len(self.analysis.nodes))

    def paths(self, root, *, winning: bool = False) -> dict:
        paths = {root: []}
        queue = deque([root])
        while queue:
            self.budget.check()
            source = queue.popleft()
            for edge in self.analysis.graph[source]:
                if edge.target is None or edge.target in paths or winning and edge.target not in self.analysis.winning:
                    continue
                paths[edge.target] = paths[source] + [self.edge_ids[(source, edge)]]
                queue.append(edge.target)
        return paths

    def replay(self, token: Any, problem_id: str):
        if token is None:
            return self.root, []
        token = _object(token, "accepted_prefix")
        if set(token) != {"version", "problem_id", "path"} or type(token["version"]) is not int or token["version"] != 1:
            raise ValueError("Unsupported accepted-prefix token")
        if token["problem_id"] != problem_id:
            raise ValueError("Accepted prefix belongs to a different frozen composition problem")
        node = self.root
        for identifier in _list(token["path"], "accepted_prefix.path"):
            edge = next((edge for edge in self.analysis.graph[node]
                         if self.edge_ids[(node, edge)] == identifier), None)
            if edge is None or edge.target not in self.analysis.winning:
                raise ValueError("Accepted prefix cannot replay an unknown or losing transition")
            node = edge.target
        if node not in self.decision_nodes:
            raise ValueError("An accepted prefix must end at a controllable event decision")
        return node, list(token["path"])

    def decisions_at(self, node) -> list:
        result = []
        for edge in self.analysis.graph[node]:
            details = self.details[self.edge_ids[(node, edge)]]
            if not edge.controllable:
                continue
            reachable = self.paths(edge.target) if edge.target is not None else {}
            status = "allowed" if edge.target in self.analysis.winning else (
                "inconclusive" if set(reachable) & (set(self.pending) | self.uncertain)
                and not set(self.analysis.nodes[node].state['compatible_choices']) <= self.definite_schedules else "held")
            result.append({key: deepcopy(details[key]) for key in (
                "edge_id", "action", "event_ids", "kind", "time", "time_exact", "target",
            )} | {"status": status})
        return result


def analyze_grounded_recovery_composition(  # noqa: PLR0913
    *, grounding_inputs: dict, recovery_events: list[dict], running_work: list[dict],
    event_start_choices: list[dict], completion: dict, accepted_prefix: dict | None = None,
    task_monitor_context: dict | None = None, physical_checkpoint: dict | None = None,
    budget: Budget | None = None, primitive_models: dict | None = None,
) -> dict[str, Any]:
    """Analyze only explicitly supplied offline recovery scheduling alternatives.

    Args:
        grounding_inputs: Shared frozen inputs of the automatic grounding checker.
        recovery_events: Exact event identities, predecessors and primitive indices.
            An optional ``des_event_id`` binds its separately preserved symbol;
            otherwise all source records must agree on one nonempty value.
        running_work: Already-started task identities and fixed remaining programs.
        event_start_choices: Complete authored schedules, programs and stationary evidence.
            Optional product_effect_evidence must match the owner-declared native
            task effects and their exact completion times.
        completion: Required subsets of final resource and part observations.
        accepted_prefix: An engine-issued, replayable path in this unchanged problem.
        task_monitor_context: Detached CCA task monitors, existing states and approved
            task completion effects. Each scope retains its own AP namespace;
            optional exact ``event_ids`` retain that scope's native task clock.
        physical_checkpoint: Reconstructible observed physical prefix and its exact
            fingerprint; asserted state IDs are checked by replay.
        budget: Shared preparation, graph exploration and nonblocking-analysis limits.
        primitive_models: Trusted owner models, kept outside serialized composition inputs.

    Returns:
        JSON-ready offline decisions, forced counterexamples, completion witness,
        scope and replayable decision prefixes. No task is executed or admitted.
    """
    budget = budget or Budget()
    result = {"status": "inconclusive", "reason": "analysis_incomplete", "offline": True,
              "clock_version": _CLOCK_VERSION, "problem_id": None, "choices": [],
              "decision_prefixes": [], "counterexample": [], "completion_witness": None,
              "scope": {}, "explored_states": 0}
    try:
        inputs = deepcopy(_object(grounding_inputs, "grounding_inputs"))
        events, running = _records(recovery_events, running=False), _records(running_work, running=True)
        completion = deepcopy(_object(completion, "completion"))
        task_history = (TaskMonitorHistories(task_monitor_context, events, running)
                        if task_monitor_context is not None else None)
        schedules, decisions, problem_id = _prepare(
            inputs, event_start_choices, events, running, completion, budget, task_history, primitive_models)
        composition = _Composition(inputs, events, running, schedules, decisions, completion, budget,
                                   task_history, physical_checkpoint, primitive_models)
        if task_monitor_context is not None or physical_checkpoint is not None:
            problem_id = _digest({"physical_problem": problem_id,
                                  "task_monitor_context": composition.task_history.identity
                                  if composition.task_history else None,
                                  "physical_checkpoint": physical_checkpoint})
        result["problem_id"] = problem_id
        result["root_node"] = composition.node_ids[composition.root]
        result["physical_trace_fingerprints"] = {
            row["id"]: physical_trace_fingerprint(row["prepared"]) for row in schedules}
        result["scope"] = composition.analysis.scope.evidence()
        result["scope"]["dependency_closure"] = {
            "mode": "complete_frozen_scene",
            "reason": "no_compositional_independence_certificate",
            "includes": ["ongoing_nominal", "stationary_resources", "pending_nominal",
                         "future_entrants", "shared_resources", "guards", "products"],
            "unmodeled_future_starts": "held_by_runtime_admission",
        }
        result["region_relevance"] = {
            row["id"]: deepcopy(row["region_relevance"]) for row in schedules
        }
        result["region_mutexes"] = selected_region_mutexes(composition.rules)
        composition.build()
        analysis = composition.analysis
        analysis.winning = nonblocking_region(analysis.graph, composition.marked, budget)
        current, accepted_path = composition.replay(accepted_prefix, problem_id)
        reachable = composition.paths(current)
        result["status"] = "allowed" if current in analysis.winning else (
            "inconclusive" if set(reachable) & (set(composition.pending) | composition.uncertain)
            and not set(analysis.nodes[current].state['compatible_choices']) <= composition.definite_schedules else "held")
        result["reason"] = "" if result["status"] == "allowed" else (
            "required_continuation_outside_supplied_behavior" if result["status"] == "inconclusive"
            else "no_joint_completion")
        result["choices"] = composition.decisions_at(current)
        result["pending_rule_ids"] = composition.pending_states(analysis.nodes[current].monitors)
        if composition.continuous:
            result['clock_version'] = schedules[0]['prepared']['clock_version']
            result['physical_evidence'] = deepcopy(schedules[0]['prepared']['model']['evidence'])
            result['ap_evidence'] = deepcopy(schedules[0]['prepared']['ap_evidence'])
            if result['status'] == 'inconclusive' and composition.uncertain:
                result['reason'] = 'continuous_motion_occupancy_unresolved'
        result["safety_dfa_states_before"] = dict(zip(
            composition.rule_ids, analysis.nodes[current].monitors, strict=True))
        if composition.task_history:
            task_state = analysis.nodes[current].state["task_monitor"]
            result["task_monitor_state_before"] = deepcopy(task_state)
            result["task_pending_rule_ids"] = composition.task_history.pending(task_state)
        preceding = next((composition.details[identifier] for identifier in reversed(accepted_path)
                          if composition.details[identifier]["kind"] == "observation"), None)
        result["previous_observation"] = deepcopy(preceding)
        winning_paths = composition.paths(current, winning=True)
        for node in sorted(composition.decision_nodes & set(winning_paths), key=lambda key: composition.node_ids[key]):
            path = accepted_path + winning_paths[node]
            result["decision_prefixes"].append({
                "time": float(Fraction(analysis.nodes[node].state["time_exact"])),
                "started_event_ids": sorted({identity for edge_id in path
                                             for identity in composition.details[edge_id].get("event_ids", [])}),
                "accepted_prefix": {"version": 1, "problem_id": problem_id, "path": path},
                "choices": composition.decisions_at(node),
            })
        for node, path in reachable.items():
            bad = next((edge for edge in analysis.graph[node] if not edge.controllable and edge.target is None
                        and (result['status'] != 'held' or composition.details[composition.edge_ids[(node,edge)]].get('certainty') != 'unresolved')), None)
            if bad is not None:
                result["counterexample"] = [deepcopy(composition.details[edge_id]) for edge_id in (
                    accepted_path + path + [composition.edge_ids[(node, bad)]])]
                break
        target = next((node for node in winning_paths if node in composition.marked), None)
        if target is not None:
            selected = schedules[composition.terminal_schedule[target]]
            prepared = selected["prepared"]
            path = accepted_path + winning_paths[target]
            result["completion_witness"] = {
                "schedule_id": selected["id"], "path": [deepcopy(composition.details[edge_id]) for edge_id in path],
                "observations": deepcopy(prepared["observations"]),
                "observation_count": len(prepared["observations"]),
                "projected_snapshot": deepcopy(prepared["projected_snapshot"]),
                "completed_event_ids": list(events), "completed_running_task_ids": list(running),
                "pending_rule_ids": [],
                "safety_dfa_states_after": dict(zip(composition.rule_ids, analysis.nodes[target].monitors, strict=True)),
            }
            if composition.task_history:
                result["completion_witness"]["task_monitor_state_after"] = deepcopy(
                    analysis.nodes[target].state["task_monitor"])
        result["explored_states"] = len(analysis.nodes)
        result["bindings"] = deepcopy(schedules[0]["prepared"]["bindings"])
        result["graph"] = {
            "nodes": [{"id": composition.node_ids[key], "phase": node.state["phase"],
                       "time_exact": node.state["time_exact"], "running": [action.key for action in node.running],
                       "monitors": dict(zip(composition.rule_ids, node.monitors, strict=True)),
                       "state": deepcopy(node.state),
                       "task_monitor_state": deepcopy(node.state.get("task_monitor")),
                       "winning": key in analysis.winning, "marked": key in composition.marked}
                      for key, node in analysis.nodes.items()],
            "edges": list(composition.details.values()),
        }
        budget.check()
    except (AnalysisLimit, ValueError, KeyError, TypeError, IndexError, ImportError, OSError) as exc:
        result.update(status="inconclusive", reason=str(exc), choices=[], decision_prefixes=[], completion_witness=None)
    result["analysis_time_sec"] = budget.clock() - budget.started
    return result

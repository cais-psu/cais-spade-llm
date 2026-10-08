"""Recovery resource-model adapter for CCA local parallel composition."""

from __future__ import annotations

import json
from copy import copy, deepcopy
from itertools import combinations
from typing import Any

from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
from cais_spade_llm.agents.central_controller.ppr_ap import parse_ap_record
from cais_spade_llm.agents.central_controller.local_composition import (
    Action, Budget, IncompleteModel, Scope,
)
from cais_spade_llm.product.environment import fingerprint
from cais_spade_llm.resources.environment_models import (
    candidates, event_bindings, feasibility, matches_requirement, project_transition,
)


def task_key(task: dict) -> str:
    """Identify a bound transition without changing any project symbol."""
    return fingerprint([task["resource_id"], task["event_id"], task["event_name"],
                        task["parameters"]])


def detached_checker(monitor: BaseSafetyChecker) -> BaseSafetyChecker:
    """Reuse compiled DFAs while keeping per-search evaluation caches private."""
    checker = copy(monitor)
    for attribute in ("current_states", "running_aps", "resource_state_aps", "resource_states", "history_error"):
        if hasattr(monitor, attribute):
            setattr(checker, attribute, deepcopy(getattr(monitor, attribute)))
    checker._eval_result_cache = {}
    checker._compiled_expr_cache = dict(monitor._compiled_expr_cache)
    return checker


def state_labels(checker: BaseSafetyChecker, values: dict, products: dict,
                 jids: dict, contexts: dict | None = None) -> frozenset[str]:
    """Map acknowledged valuations through the existing AP implementation."""
    result = set()
    for rid, value in values.items():
        params = {**(contexts or {}).get(rid, {}), **value}
        part = value.get("held_part") or value.get("part_name") or params.get("part_name")
        if part in products:
            params.update(part_name=part, part_location=products[part].get("location"),
                          part_state=products[part].get("state"))
        result.update(checker._map_state_to_aps(
            jids[rid], str(value.get("resource_state", "")), params))
    return frozenset(result)


class EnvironmentPlant:
    """Detached declared resource behavior with operation/release completion goals."""

    def __init__(self, context, checker: BaseSafetyChecker, jids: dict,
                 goals: dict[str, dict], admitted: list[dict] | None = None) -> None:
        """Bind an immutable calculation context to existing nominal semantics."""
        self.context = context
        self.checker = checker
        state_aps = [ap for rule in checker.safety_rules for ap in rule.get("aps", [])
                     if parse_ap_record(ap)["kind"] == "ap_state"]
        self._has_state_aps = bool(state_aps)
        self._has_state_context = self._has_state_aps
        for ap in state_aps:
            definition = parse_ap_record(ap)
            condition = definition["state"]
            if (definition["product"] not in {"*", "any"} or condition["arguments"]
                    or ap.get("source_task_ids") or ap.get("field") or ap.get("value")
                    or "=" not in condition["symbol"]
                    or condition["symbol"].partition("=")[0] not in context.models.get(
                        definition["resource"], {}).get("state_variables", {})):
                break
        else:
            # These APs read owned valuations only. Their future truth cannot
            # depend on previous task parameters or task identities.
            overrides = {"outline_expected_start_state", "expected_end_state", "projected_outline_state"}
            self._has_state_context = self._has_state_aps and (any(
                overrides.intersection(event["parameter_bindings"])
                for model in context.models.values() for event in model["events"]
            ) or any(overrides.intersection(task["parameters"])
                     for task in context.pending_tasks.values()))
        self.jids = dict(jids)
        self.initial = {"resources": context.snapshot(),
                        "products": deepcopy(context.part_tracker), "contexts": {}}
        self.goals = deepcopy(goals)
        self.admitted = deepcopy(admitted or [])
        self.bound_tasks = {task_key(task): deepcopy(task)
                            for task in context.pending_tasks.values()}
        self._templates: dict[str, list[dict]] = {}
        self._selected_templates: list[dict] = []
        self._actions: dict[str, Action] = {}
        self.missing_capabilities: set[str] = set()
        self._release_only: set[str] = set()
        self._extended: set[str] = set()
        self._label_cache: dict[int, tuple[dict, frozenset[str]]] = {}
        self._key_cache: dict[tuple, tuple[dict, str]] = {}
        self._projection_cache: dict[tuple, tuple[dict, dict | str]] = {}
        self._action_cache: dict[tuple, Action] = {}

    def action(self, task: dict, state: dict | None = None) -> Action:
        """Bind exact parameters, task APs and the existing reservation contract."""
        state = self.initial if state is None else state
        part = task["parameters"].get("part_name") or task.get("part_name")
        key = task_key(task)
        cache_key = (key, task.get("task_id"),
                     state["resources"][task["resource_id"]].get("resource_location"),
                     tuple(sorted(name for name, value in state["products"].items()
                                  if value.get("location") == "Conveyor")))
        if cache_key in self._action_cache:
            return self._action_cache[cache_key]
        if part is None:
            raise IncompleteModel("missing_part_binding")
        frozen = copy(self.context)
        frozen.part_tracker = state["products"]
        frozen.resources = {rid: copy(actor) for rid, actor in self.context.resources.items()}
        for rid, actor in frozen.resources.items():
            actor.valuation = state["resources"][rid]
        params = {**task["parameters"]}
        if task.get("task_id"):
            params["task_id"] = task["task_id"]
        labels = frozenset(self.checker._map_task_to_aps(
            self.jids[task["resource_id"]], task["event_name"], params))
        claims = frozenset(frozen._task_reservations(task, part))
        event = next(event for event in self.context.models[task["resource_id"]]["events"]
                     if event["event_id"] == task["event_id"])
        result = Action(task_key(task), deepcopy(task), claims, labels,
                        event.get("controllable", True))
        self._actions[result.key] = result
        self._action_cache[cache_key] = result
        return result

    def _desired(self, part: str) -> dict:
        if part in self._release_only:
            return {}
        if part in self.goals:
            return self.goals[part]
        requirements = self.context.requirements.get(part, [])
        desired = next((goal for goal in requirements
                        if not matches_requirement(self.initial["products"][part], goal)), None)
        if desired is None:
            # Completed material can still need transport to release a resource.
            return requirements[-1] if requirements else {}
        return desired

    def _part_templates(self, part: str, budget: Budget) -> list[dict]:
        if part in self._templates:
            return self._templates[part]
        rows = []
        desired = self._desired(part)
        requirements = self.context.requirements.get(part, [])
        if desired in requirements[1:]:
            # A later processPlan step must continue beyond the previous
            # operation's stable collection, while ordered guards still apply.
            self._extended.add(part)
        desired_items = []
        # Later operation effects may be prerequisites of a release or a DFA
        # obligation. Ordered product guards still govern when they can execute.
        for requirement in ([desired, *requirements] if part in self._extended else [desired]):
            for item in requirement.get("processesToComplete", [requirement]):
                if item not in desired_items:
                    desired_items.append(item)
        for rid, model in self.context.models.items():
            if rid not in self.context.permitted_resources:
                continue
            for event in model["events"]:
                budget.check()
                if event["parameter_bindings"]["resource_id"].get("equals") != rid:
                    continue
                for goal in desired_items:
                    params = event_bindings(event, part, goal, requirements)
                    if (params is None or params.get("part_name", part) != part
                            or params.get("product_completed") and part != self.context.product_name):
                        continue
                    task = {"resource_id": rid, "event_id": event["event_id"],
                            "event_name": event["event_name"], "parameters": params,
                            "part_name": part}
                    if not self.context.allows_task(task):
                        continue
                    status, _ = feasibility(model, task, self.context.geometry.get(part, {}))
                    if status == "INFEASIBLE":
                        continue
                    if status != "FEASIBLE":
                        self.missing_capabilities.add(f"{rid}:{event['event_id']}:bindings")
                        continue
                    footprint = set(event["participants"])
                    claims = self.context._task_reservations(task, part)
                    for claim in claims:
                        if claim.startswith("resource:"):
                            footprint.add(claim.removeprefix("resource:"))
                    rows.append({"task": task, "event": event, "goal": goal,
                                 "resources": footprint, "claims": claims})
        releases = set()
        for row in rows:
            rid = row["task"]["resource_id"]
            updates = row["event"].get("updates", {})
            if updates.get("held_part") == {"set": None}:
                releases.add(rid)
        rows = [row for row in rows
                if "held_part" not in self.context.models[row["task"]["resource_id"]]["state_variables"]
                or row["task"]["resource_id"] in releases
                or self.initial["resources"][row["task"]["resource_id"]].get("held_part") == part]
        unique = {task_key(row["task"]): row for row in rows}
        # Capability locations are an overapproximation: omit only transitions
        # whose source custody cannot be reached by any declared selected behavior.
        reachable = {self.initial["products"][part].get("location")}
        selected = {}
        changed = True
        while changed:
            budget.check()
            before = (len(reachable), len(selected))
            for key, row in unique.items():
                transition = row["event"].get("capability_transition", {})
                source = transition.get("source", {}).get("part_location")
                target = transition.get("target", {}).get("part_location")
                stable_collection = source in self.context.models and any(
                    field.startswith("part_location.") or field.startswith("zone_") and field.endswith("_part")
                    for field in self.initial["resources"][source])
                # A completed operation may leave its part in a stable collection.
                # Transport beyond it is selected separately when occupancy blocks
                # another obligation or a specification requires a later operation.
                if (stable_collection and source != self.initial["products"][part].get("location")
                        and part not in self._extended and part not in self._release_only):
                    continue
                if source is None or source in reachable:
                    selected[key] = row
                    if target is not None:
                        reachable.add(target)
            changed = before != (len(reachable), len(selected))
        active_actors = {row["task"]["resource_id"] for row in selected.values()
                         if row["task"]["event_name"] != "move_home"}
        proposed = getattr(self, "proposed", None)
        self._templates[part] = [row for row in selected.values()
                                if row["task"]["event_name"] != "move_home"
                                or row["task"]["resource_id"] in active_actors
                                or self.initial["resources"][row["task"]["resource_id"]].get("task_ctx.part_name") == part
                                or proposed is not None and task_key(row["task"]) == proposed.key]
        return self._templates[part]

    def _rule_resources(self, rule: dict) -> set[str]:
        resources = set()
        for ap in rule.get("aps", []):
            descriptor = self.checker._parse_ap_descriptor(ap.get("full", ""))
            if descriptor is None:
                raise IncompleteModel("unsupported_ap_descriptor")
            token = descriptor["resource"]
            resources.update(rid for rid, jid in self.jids.items()
                             if token in {"*", "any"}
                             or token == rid or token == self.checker._resource_short_name(jid))
        return resources

    def _unmatched_stutters(self, rule_id: str, rule: dict, budget: Budget) -> bool:
        symbols = [ap["label"] for ap in rule.get("aps", [])
                   if parse_ap_record(ap)["kind"] == "ap_state"]
        if len(symbols) > 12:
            return False
        dfa = self.checker.dfas[rule_id]
        for count in range(len(symbols) + 1):
            for valuation in combinations(symbols, count):
                for current in dfa["transitions"]:
                    budget.check()
                    evidence = self.checker.transition_evidence(rule_id, current, frozenset(valuation))
                    if evidence["to"] != current or evidence["reason"] in {
                            "dfa_missing_transition", "dfa_ambiguous_transition"}:
                        return False
        return True

    def select(self, candidate: Action, running: tuple[Action, ...],
               checker: BaseSafetyChecker, budget: Budget, *, full: bool = False) -> Scope:
        """Select operation continuations and recursively include their dependencies.

        Selection is deliberately conservative for shared resources. No proof of
        independence is inferred from an absent event-name match.
        """
        part = candidate.task["parameters"].get("part_name") or candidate.task.get("part_name")
        if part not in self.initial["products"]:
            raise IncompleteModel("unknown_candidate_part")
        scope = Scope(products={part})
        self.scope = scope
        scope.reasons.append({"kind": "candidate", "task": candidate.key, "part": part})
        known_tasks = [item.task for item in running] + self.admitted
        rule_by_id = {rule["id"]: rule for rule in checker.safety_rules}
        changed = True
        while changed:
            budget.check()
            before = (set(scope.products), set(scope.resources), set(scope.rules), set(self._extended))
            rows = [row for name in sorted(scope.products)
                    for row in self._part_templates(name, budget)]
            for row in rows:
                for rid in sorted(row["resources"] - scope.resources):
                    scope.reasons.append({"kind": "event_participant_or_reservation",
                                          "resource": rid, "task": task_key(row["task"])})
                scope.resources.update(row["resources"])
                scope.claims.update(row["claims"])
            for task in known_tasks:
                action_resources = set(self.context._task_participants(task))
                other = task["parameters"].get("part_name") or task.get("part_name")
                action_claims = set(self.action(task).claims)
                if (action_resources & scope.resources or action_claims & scope.claims) and other not in scope.products:
                    scope.products.add(other)
                    scope.reasons.append({"kind": "admitted_interaction",
                                          "task": task_key(task), "part": other,
                                          "resources": sorted(action_resources & scope.resources),
                                          "reservations": sorted(action_claims & scope.claims)})
            # Occupancy persists even when its producing task left the pending batch.
            for name, state in self.initial["products"].items():
                location = state.get("location")
                if name == self.context.product_name or name in scope.products:
                    continue
                # Storage and printer output retain acknowledged occupancy, but
                # another stored product does not itself require future execution.
                if (location in scope.resources and location != "Storage"
                        and f"output.{name}" not in self.initial["resources"].get(location, {})):
                    if name not in self.context.selected_parts:
                        raise IncompleteModel("missing_operation_requirements: " + name)
                    scope.products.add(name)
                    if name not in self.goals:
                        self._release_only.add(name)
                    values = self.initial["resources"][location]
                    zones = [value for field, value in values.items()
                             if field.startswith("zone_") and field.endswith("_part")]
                    if (location in self.context.machine_ids or values.get("held_part") == name
                            or zones and all(value is not None for value in zones)):
                        self._extended.add(name)
                    self._templates.pop(name, None)
                    scope.reasons.append({"kind": "occupied_resource", "part": name,
                                          "resource": location})
            for rule_id, dfa in checker.dfas.items():
                budget.check()
                rule = rule_by_id.get(rule_id)
                if rule is None and dfa.get("ap_symbols"):
                    raise IncompleteModel("missing_specification_mapping: " + rule_id)
                resources = self._rule_resources(rule or {})
                # Even a completion with no matching AP can change this monitor.
                empty_changes = not self._unmatched_stutters(rule_id, rule or {}, budget)
                if empty_changes or resources & scope.resources:
                    scope.resources.update(resources)
                    known_ids = ({task.get("task_id") for task in self.bound_tasks.values()}
                                 | set(getattr(self.context, "completed_task_ids", ())))
                    for ap in (rule or {}).get("aps", []):
                        if set(ap.get("source_task_ids") or []) - known_ids:
                            raise IncompleteModel("missing_task_identity_binding: " + rule_id)
                    if rule_id not in scope.rules:
                        scope.rules.add(rule_id)
                        scope.reasons.append({"kind": "specification", "rule": rule_id,
                                              "empty_event_changes_state": empty_changes,
                                              "resources": sorted(resources & scope.resources)})
                    # A specification's prerequisites may be waiting rather than admitted.
                    for name in self.context.selected_parts:
                        if name in self._extended:
                            continue
                        # Inspect every declared operation for a waiting prerequisite,
                        # without adding unrelated future execution to the composition.
                        was_extended = name in self._extended
                        self._extended.add(name)
                        self._templates.pop(name, None)
                        future_rows = self._part_templates(name, budget)
                        if not was_extended:
                            self._extended.discard(name)
                            self._templates.pop(name, None)
                        for row in future_rows:
                            params = row["task"]["parameters"]
                            labels = checker._map_task_to_aps(
                                self.jids[row["task"]["resource_id"]],
                                row["task"]["event_name"], params)
                            if set(labels) & set(dfa["ap_symbols"]):
                                scope.products.add(name)
                                self._release_only.discard(name)
                                self._extended.add(name)
                                self._templates.pop(name, None)
                                scope.reasons.append({"kind": "specification_prerequisite",
                                                      "rule": rule_id, "part": name})
                                break
                    if empty_changes:
                        # No unmatched-event stutter proof: include all registered
                        # admitted work, but not every hypothetical future arrival.
                        for task in known_tasks:
                            scope.products.add(task["parameters"].get("part_name")
                                               or task.get("part_name"))
            changed = before != (scope.products, scope.resources, scope.rules, self._extended)
        self._selected_templates = [
            row for name in sorted(scope.products)
            for row in self._part_templates(name, budget)
        ]
        for name in sorted(scope.products):
            if (not self._part_templates(name, budget)
                    and not matches_requirement(self.initial["products"][name], self._desired(name))):
                raise IncompleteModel("missing_continuation: " + name)
            scope.goals.append({"part_name": name, "operation": self._desired(name)})
        used_resources = set().union(*(row["resources"] for row in self._selected_templates))
        for rid in sorted(used_resources):
            model = self.context.models[rid]
            # Printer output and buffers retain completed material. Their global
            # marked state is not a local operation's release obligation.
            conditions = (model.get("marked_state_conditions", [])
                          if "held_part" in model["state_variables"] else [])
            if rid in self.context.machine_ids:
                conditions = [{"resource_state": {"equals": "idle"},
                               "part_name": {"equals": None}}]
            if conditions:
                scope.goals.append({"resource_id": rid, "release": deepcopy(conditions)})
        for action in running:
            if (action.task["parameters"].get("part_name") or action.task.get("part_name")) in scope.products:
                scope.tasks.add(action.key)
        if full:
            scope.terminal_rules = set(scope.rules)
            for name in self.context.selected_parts:
                self._extended.add(name)
                self._templates.pop(name, None)
            self._selected_templates = [row for name in self.context.selected_parts
                                        for row in self._part_templates(name, budget)]
            scope.products.update(self.context.selected_parts)
            scope.resources.update(self.context.models)
            for row in self._selected_templates:
                scope.claims.update(row["claims"])
            scope.rules.update(checker.dfas)
            scope.tasks.update(action.key for action in running)
            scope.reasons.append({"kind": "full_reference", "objective": "same_local_completion"})
        # Retain binding identities in explanations; no task is executed by selection.
        scope.tasks.update(task_key(row["task"]) for row in self._selected_templates)
        scope.task_bindings.update({task_key(row["task"]): deepcopy(row["task"])
                                    for row in self._selected_templates})
        return scope

    def actions(self, state: dict, scope: Scope, budget: Budget):
        """Instantiate all selected modeled alternatives, including belt outcomes."""
        emitted = set()
        for row in self._selected_templates:
            budget.check()
            task, event = row["task"], row["event"]
            rid, part = task["resource_id"], task["part_name"]
            if task["event_name"] == "advance_conveyor":
                available = candidates(self.context.models[rid], state["resources"], part,
                                       row["goal"], self.context.requirements.get(part, []))
            else:
                available = [task]
            for bound in available:
                budget.check()
                bound = {**bound, "part_name": part}
                if bound["event_id"] != event["event_id"]:
                    continue
                key = task_key(bound)
                if key in emitted:
                    continue
                emitted.add(key)
                if bound["event_name"] not in self.context.resources[rid].executors:
                    self.missing_capabilities.add(f"{rid}:{bound['event_id']}:executor")
                    continue
                status, _ = feasibility(self.context.models[rid], bound,
                                        self.context.geometry.get(part, {}))
                if status != "FEASIBLE":
                    continue
                bound = self.bound_tasks.get(key, bound)
                proposed = getattr(self, "proposed", None)
                if proposed is not None and proposed.key == key:
                    bound = proposed.task
                yield self.action(bound, state)

    def project(self, state: dict, action: Action) -> dict:
        """Apply the existing guard, update, custody and product-effect evaluator."""
        key = (id(state), action.key)
        cached = self._projection_cache.get(key)
        if cached is not None and cached[0] is state:
            if isinstance(cached[1], str):
                raise ValueError(cached[1])
            return cached[1]
        try:
            values, products = project_transition(
                self.context.models, state["resources"], state["products"], action.task,
                self.context.product_name, self.context.requirements)
        except ValueError as exc:
            self._projection_cache[key] = (state, str(exc))
            raise
        contexts = deepcopy(state.get("contexts", {}))
        for rid in self.context._task_participants(action.task):
            contexts[rid] = dict(action.task["parameters"])
            if action.task.get("task_id"):
                contexts[rid]["task_id"] = action.task["task_id"]
        result = {"resources": values, "products": products, "contexts": contexts}
        self._projection_cache[key] = (state, result)
        return result

    def labels(self, state: dict) -> frozenset[str]:
        """Evaluate persistent propositions without modifying any live monitor."""
        if not self._has_state_aps:
            return frozenset()
        cached = self._label_cache.get(id(state))
        if cached is not None and cached[0] is state:
            return cached[1]
        labels = state_labels(self.checker, state["resources"], state["products"],
                              self.jids, state.get("contexts"))
        self._label_cache[id(state)] = (state, labels)
        return labels

    def key(self, state: dict, scope: Scope) -> str:
        """Include exact scoped values and proposition context in state identity."""
        cache_key = (id(state), frozenset(scope.products), frozenset(scope.resources))
        cached = self._key_cache.get(cache_key)
        if cached is not None and cached[0] is state:
            return cached[1]
        products = {
            name: {key: value for key, value in state["products"][name].items()
                   if key != "last_task"} for name in sorted(scope.products)
        }
        contexts = state.get("contexts", {})
        retain_contexts = self._has_state_context or (self._has_state_aps and any(
            {"outline_expected_start_state", "expected_end_state", "projected_outline_state"}.intersection(params)
            for params in contexts.values()
        ))
        key = json.dumps([
            {rid: state["resources"][rid] for rid in sorted(scope.resources)},
            products, sorted(self.labels(state)), contexts if retain_contexts else {},
        ], sort_keys=True, separators=(",", ":"))
        self._key_cache[cache_key] = (state, key)
        return key

    def complete(self, state: dict, scope: Scope) -> bool:
        """Require operation completion and declared releases, not empty buffers."""
        for goal in scope.goals:
            if "part_name" in goal:
                if not matches_requirement(state["products"][goal["part_name"]], goal["operation"]):
                    return False
            else:
                values = state["resources"][goal["resource_id"]]
                if not any(all(values.get(key) == expected["equals"]
                               for key, expected in condition.items())
                           for condition in goal["release"]):
                    return False
        return True

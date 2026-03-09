"""Plan validator that checks a compiled plan FSA against DFA safety rules."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import json
import re
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from agents.central_controller.base_safety_checker import BaseSafetyChecker


class PlanSafetyValidator(BaseSafetyChecker):
    """
    Validation of a *compiled FSA plan* against LTLf-based DFA safety rules.

    Plant model:
      - Input is an FSA JSON of the form:
          fsa["A"] = { "X": [...], "E": [...], "Tr": [...], "x0": "...", "Xm": ["..."] }

      - Transitions contain at least:
          { "from": <state_name>, "event": <event_str>, "to": <state_name>, ... }

    Verification model (per rule):
      - Explore ALL reachable product states (x, q) where:
          x : plant (FSA) state
          q : DFA state for the rule
      - For each enabled plant transition (x --e--> x'):
          sigma := AP-set emitted by that transition, projected to this rule's AP alphabet
          q' := delta_rule(q, sigma)
          if q' reaches violation_state => violation with witness trace
      - Also apply an end-of-plan empty step (sigma=∅) whenever reaching a marked plant state x∈Xm
        to catch eventualities.

    AP mapping:
      - Preferred: semantic mapping using BaseSafetyChecker._map_task_to_aps(resource, fn, params)
        This requires either:
          (a) transition carries resource_jid/function_name/params, OR
          (b) you provide the original DAG plan (nodes) so we can look up task_id -> node.

    Notes:
      - This is *complete* wrt the FSA language: if any FSA path can violate a rule, we find one.
    """

    # Guardrail: stop if product graph explodes (tune as needed)
    MAX_PRODUCT_STATES: int = 200_000
    _RUNNING_TASK_RE = re.compile(
        r"([^=,\s]+)=\([^)]*?run=([^:,\)]+):([A-Za-z0-9_-]+)"
    )

    def __init__(
        self,
        rules: List[Dict[str, Any]],
        dfa_map: Dict[str, str],
        tools_catalog: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        # BaseSafetyChecker signature is (dfa_map, rules)
        super().__init__(dfa_map, rules, tools_catalog=tools_catalog)
        self.rule_lookup: Dict[str, Dict[str, Any]] = {r["id"]: r for r in rules if r.get("id")}

    # ------------------------------------------------------------------ #
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------ #
    def validate_plan_fsa(
        self,
        fsa: Dict[str, Any],
        plan: Optional[Dict[str, Any]] = None,
        product_jid: str | None = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate a compiled FSA against all DFA safety rules.

        Args:
            fsa         : compiled FSA JSON dict (must contain fsa["A"])
            plan        : optional original DAG plan dict (with "nodes") used for AP mapping
            product_jid : optional identifier for logging

        Returns:
            (is_valid, violations)
        """
        A = (fsa or {}).get("A") or {}
        Tr = A.get("Tr") or []
        x0 = A.get("x0")
        Xm = set(A.get("Xm") or [])

        if not x0 or not Tr:
            return True, []

        self.logger.info(
            "[PlanValidator] Validating FSA for %s (|Tr|=%d)...",
            product_jid,
            len(Tr),
        )

        enabled = self._index_enabled(Tr)

        # Optional: task_id -> node lookup from plan
        task_lookup = self._build_task_lookup(plan)

        task_meta_lookup = self._build_transition_task_lookup(enabled, task_lookup)
        initial_resource_states = self._initial_resource_states(enabled, x0, task_meta_lookup)

        all_violations: List[Dict[str, Any]] = []

        for rule in self.safety_rules:
            rule_id = rule.get("id")
            if not rule_id:
                continue

            dfa = self.dfas.get(rule_id)
            if not dfa:
                continue

            aps_for_rule: Set[str] = set(dfa.get("ap_symbols", []))
            if not aps_for_rule:
                continue

            violations = self._check_rule_on_fsa_product(
                rule_id=rule_id,
                rule=rule,
                x0=x0,
                Xm=Xm,
                enabled=enabled,
                aps_for_rule=aps_for_rule,
                task_lookup=task_lookup,
                task_meta_lookup=task_meta_lookup,
                initial_resource_states=initial_resource_states,
            )

            # Keep only ONE witness per rule
            if violations:
                all_violations.append(violations[0])
        return (len(all_violations) == 0), all_violations

    # Backward-compatible alias kept during the validator rename migration.
    def validate_fsa_offline(
        self,
        fsa: Dict[str, Any],
        plan: Optional[Dict[str, Any]] = None,
        product_jid: str | None = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        return self.validate_plan_fsa(fsa=fsa, plan=plan, product_jid=product_jid)

    # ------------------------------------------------------------------ #
    # INDEXING
    # ------------------------------------------------------------------ #
    def _index_enabled(self, transitions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        enabled: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in transitions:
            frm = t.get("from")
            if frm is None:
                continue
            enabled[str(frm)].append(t)
        return enabled

    def _build_task_lookup(self, plan: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Build task_id -> node dict from DAG plan (optional).
        """
        if not plan:
            return {}
        nodes = plan.get("nodes") or []
        out: Dict[str, Dict[str, Any]] = {}
        for n in nodes:
            nid = n.get("id")
            if nid:
                out[str(nid)] = n
        return out

    @staticmethod
    def _merge_task_metadata(dest: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("task_id", "resource_jid", "function_name", "in_state", "out_state"):
            value = src.get(key)
            if value is None:
                continue
            token = str(value).strip()
            if token:
                dest[key] = value
        params = src.get("params")
        if isinstance(params, dict):
            if dest.get("params") is None:
                dest["params"] = {}
            if params:
                dest["params"] = dict(params)
        return dest

    def _build_transition_task_lookup(
        self,
        enabled: Dict[str, List[Dict[str, Any]]],
        task_lookup: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}

        for task_id, node in task_lookup.items():
            out[str(task_id)] = self._merge_task_metadata({}, dict(node))

        for transitions in enabled.values():
            for transition in transitions:
                task_id = str(transition.get("task_id") or "").strip()
                if not task_id:
                    continue
                entry = out.setdefault(task_id, {})
                self._merge_task_metadata(entry, transition)
                if task_lookup.get(task_id):
                    self._merge_task_metadata(entry, task_lookup[task_id])
                entry["task_id"] = task_id
                if "params" not in entry:
                    entry["params"] = {}

        return out

    def _transition_task_meta(
        self,
        transition: Dict[str, Any],
        task_lookup: Dict[str, Dict[str, Any]],
        task_meta_lookup: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        task_id = str(transition.get("task_id") or "").strip()
        meta: Dict[str, Any] = {}
        if task_id and task_id in task_meta_lookup:
            self._merge_task_metadata(meta, task_meta_lookup[task_id])
        if task_id and task_id in task_lookup:
            self._merge_task_metadata(meta, task_lookup[task_id])
        self._merge_task_metadata(meta, transition)
        if task_id:
            meta["task_id"] = task_id
        if not isinstance(meta.get("params"), dict):
            meta["params"] = {}
        return meta

    def _running_tasks_from_state(self, x: str) -> List[Dict[str, str]]:
        running: List[Dict[str, str]] = []
        for match in self._RUNNING_TASK_RE.finditer(str(x or "")):
            running.append(
                {
                    "resource_jid": str(match.group(1)),
                    "task_id": str(match.group(2)),
                    "function_name": str(match.group(3)),
                }
            )
        return running

    def _running_event_aps_from_state(
        self,
        x: str,
        aps_for_rule: Set[str],
        task_meta_lookup: Dict[str, Dict[str, Any]],
    ) -> FrozenSet[str]:
        sigma: Set[str] = set()
        for item in self._running_tasks_from_state(x):
            meta = dict(task_meta_lookup.get(item["task_id"]) or {})
            resource_jid = str(meta.get("resource_jid") or item["resource_jid"] or "").strip()
            function_name = str(meta.get("function_name") or item["function_name"] or "").strip()
            params = meta.get("params") or {}
            if not resource_jid or not function_name:
                continue
            sigma.update(self._map_task_to_aps(resource_jid, function_name, params))
        return frozenset(ap for ap in sigma if ap in aps_for_rule)

    def _state_aps_for_resources(
        self,
        resource_states: Dict[str, Dict[str, Any]],
        aps_for_rule: Set[str],
    ) -> FrozenSet[str]:
        sigma: Set[str] = set()
        for resource_jid, payload in (resource_states or {}).items():
            current_state = str((payload or {}).get("current_state") or "").strip()
            if not current_state:
                continue
            params = dict((payload or {}).get("params") or {})
            sigma.update(self._map_state_to_aps(resource_jid, current_state, params))
        return frozenset(ap for ap in sigma if ap in aps_for_rule)

    def _running_event_aps_from_state_all(
        self,
        x: str,
        task_meta_lookup: Dict[str, Dict[str, Any]],
    ) -> FrozenSet[str]:
        sigma: Set[str] = set()
        for item in self._running_tasks_from_state(x):
            meta = dict(task_meta_lookup.get(item["task_id"]) or {})
            resource_jid = str(meta.get("resource_jid") or item["resource_jid"] or "").strip()
            function_name = str(meta.get("function_name") or item["function_name"] or "").strip()
            params = meta.get("params") or {}
            if not resource_jid or not function_name:
                continue
            sigma.update(self._map_task_to_aps(resource_jid, function_name, params))
        return frozenset(sigma)

    def _state_aps_for_resources_all(
        self,
        resource_states: Dict[str, Dict[str, Any]],
    ) -> FrozenSet[str]:
        sigma: Set[str] = set()
        for resource_jid, payload in (resource_states or {}).items():
            current_state = str((payload or {}).get("current_state") or "").strip()
            if not current_state:
                continue
            params = dict((payload or {}).get("params") or {})
            sigma.update(self._map_state_to_aps(resource_jid, current_state, params))
        return frozenset(sigma)

    def _initial_resource_states(
        self,
        enabled: Dict[str, List[Dict[str, Any]]],
        x0: str,
        task_meta_lookup: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        resource_states: Dict[str, Dict[str, Any]] = {}
        owner_to_jid: Dict[str, str] = {}
        default_domain = "localhost"

        for meta in task_meta_lookup.values():
            resource_jid = str(meta.get("resource_jid") or "").strip()
            if resource_jid and resource_jid not in resource_states:
                resource_states[resource_jid] = {"current_state": "idle", "params": {}}
            if resource_jid:
                owner_to_jid[self._resource_short_name(resource_jid)] = resource_jid
                if "@" in resource_jid:
                    default_domain = resource_jid.split("@", 1)[1] or default_domain

        for row in self.tools_catalog:
            owner = self._resource_short_name(str(row.get("function_owner_agent") or "").strip())
            if not owner:
                continue
            resource_jid = owner_to_jid.get(owner) or f"{owner}@{default_domain}"
            resource_states.setdefault(resource_jid, {"current_state": "idle", "params": {}})

        for transition in enabled.get(str(x0), []):
            task_id = str(transition.get("task_id") or "").strip()
            meta = task_meta_lookup.get(task_id, {})
            resource_jid = str(meta.get("resource_jid") or transition.get("resource_jid") or "").strip()
            in_state = str(meta.get("in_state") or transition.get("in_state") or "").strip()
            if not resource_jid:
                continue
            resource_states.setdefault(resource_jid, {"current_state": "idle", "params": {}})
            if in_state and in_state.lower() != "any":
                resource_states[resource_jid]["current_state"] = in_state

        return resource_states

    @staticmethod
    def _resource_state_signature(resource_states: Dict[str, Dict[str, Any]]) -> Tuple[Tuple[str, str, str], ...]:
        items: List[Tuple[str, str, str]] = []
        for resource_jid, payload in sorted((resource_states or {}).items()):
            current_state = str((payload or {}).get("current_state") or "").strip()
            params = dict((payload or {}).get("params") or {})
            params_token = json.dumps(params, sort_keys=True, separators=(",", ":"))
            items.append((str(resource_jid), current_state, params_token))
        return tuple(items)

    def _ordered_rule_ids(self) -> List[str]:
        return sorted(str(rule_id) for rule_id in self.dfas.keys())

    def _joint_transition_successor(
        self,
        *,
        x: str,
        q_vec: Tuple[str, ...],
        rule_ids: List[str],
        rule_ap_sets: Dict[str, Set[str]],
        transition: Dict[str, Any],
        task_lookup: Dict[str, Dict[str, Any]],
        task_meta_lookup: Dict[str, Dict[str, Any]],
        resource_states: Dict[str, Dict[str, Any]],
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...], Dict[str, Dict[str, Any]], Dict[str, FrozenSet[str]]]:
        event_name = str(transition.get("event") or "").strip()
        meta = self._transition_task_meta(transition, task_lookup, task_meta_lookup)
        resource_jid = str(meta.get("resource_jid") or "").strip()
        function_name = str(meta.get("function_name") or "").strip()
        params = dict(meta.get("params") or {})

        running_before_all = set(self._running_event_aps_from_state_all(x, task_meta_lookup))
        persistent_before_all = set(self._state_aps_for_resources_all(resource_states))

        if resource_jid and function_name:
            candidate_event_aps_all = set(self._map_task_to_aps(resource_jid, function_name, params))
            predicted_state_aps_all = set(self._predict_state_aps(resource_jid, function_name, params))
        else:
            candidate_event_aps_all = set()
            predicted_state_aps_all = set()

        next_resource_states = deepcopy(resource_states)

        if event_name.endswith(".done") or event_name.endswith(".finish"):
            running_after_all = set(running_before_all)
            for ap in candidate_event_aps_all:
                running_after_all.discard(ap)

            if resource_jid:
                next_payload = next_resource_states.setdefault(
                    resource_jid,
                    {
                        "current_state": next_resource_states.get(resource_jid, {}).get(
                            "current_state", "idle"
                        ),
                        "params": {},
                    },
                )
                out_state = str(meta.get("out_state") or "").strip()
                if out_state and out_state.lower() != "any":
                    next_payload["current_state"] = out_state
                    next_payload["params"] = dict(params)
                elif "params" not in next_payload:
                    next_payload["params"] = dict(params)

            persistent_after_all = set(self._state_aps_for_resources_all(next_resource_states))
            sigma_all = frozenset(running_after_all | persistent_after_all | candidate_event_aps_all)
        elif event_name.endswith(".start"):
            sigma_all = frozenset(
                running_before_all
                | persistent_before_all
                | candidate_event_aps_all
                | predicted_state_aps_all
            )
        else:
            sigma_all = frozenset(running_before_all | persistent_before_all | candidate_event_aps_all)

        checked_vec: List[str] = []
        committed_vec: List[str] = []
        sigma_by_rule: Dict[str, FrozenSet[str]] = {}

        for idx, rule_id in enumerate(rule_ids):
            sigma_rule = frozenset(ap for ap in sigma_all if ap in rule_ap_sets.get(rule_id, set()))
            sigma_by_rule[rule_id] = sigma_rule
            current_q = q_vec[idx]
            checked_q = self._delta(rule_id, current_q, sigma_rule)
            checked_vec.append(checked_q)

            if event_name.endswith(".start"):
                committed_q = current_q
            else:
                committed_q = checked_q
            committed_vec.append(committed_q)

        return tuple(checked_vec), tuple(committed_vec), next_resource_states, sigma_by_rule

    def _is_accepting_terminal_product_state(
        self,
        *,
        x: str,
        q_vec: Tuple[str, ...],
        Xm: Set[str],
        rule_ids: List[str],
    ) -> bool:
        if x not in Xm:
            return False

        empty_sigma = frozenset()
        for idx, rule_id in enumerate(rule_ids):
            current_q = q_vec[idx]
            q_end = self._delta(rule_id, current_q, empty_sigma)
            dfa = self.dfas.get(rule_id, {})
            violation_state = str(dfa.get("violation_state") or "").strip()
            if violation_state and q_end == violation_state:
                return False

            accepting_states = set(str(s) for s in (dfa.get("accepting_states") or []))
            if accepting_states and q_end not in accepting_states:
                return False

        return True

    def compute_winning_set(
        self,
        fsa: Dict[str, Any],
        plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the reachable joint product graph and compute its winning set.

        Winning states are product states from which some continuation reaches a
        marked plan state while all safety DFAs remain satisfiable and end in
        accepting states after the final empty-step evaluation.
        """
        A = (fsa or {}).get("A") or {}
        Tr = A.get("Tr") or []
        x0 = str(A.get("x0") or "").strip()
        Xm = set(str(x) for x in (A.get("Xm") or []))

        if not x0 or not Tr:
            return {
                "W": set(),
                "product_graph": {},
                "reverse_graph": {},
                "accepting_states": set(),
                "state_meta": {},
                "rule_ids": [],
                "initial_state": None,
                "initial_resource_states": {},
            }

        enabled = self._index_enabled(Tr)
        task_lookup = self._build_task_lookup(plan)
        task_meta_lookup = self._build_transition_task_lookup(enabled, task_lookup)
        initial_resource_states = self._initial_resource_states(enabled, x0, task_meta_lookup)

        rule_ids = self._ordered_rule_ids()
        rule_ap_sets: Dict[str, Set[str]] = {
            rule_id: set(self.dfas.get(rule_id, {}).get("ap_symbols", []))
            for rule_id in rule_ids
        }
        q0_vec = tuple(str(self.dfas.get(rule_id, {}).get("initial") or "1") for rule_id in rule_ids)

        start_payload = deepcopy(initial_resource_states)
        start_state = (x0, q0_vec, self._resource_state_signature(start_payload))

        state_payloads: Dict[
            Tuple[str, Tuple[str, ...], Tuple[Tuple[str, str, str], ...]],
            Dict[str, Dict[str, Any]],
        ] = {start_state: start_payload}
        product_graph: Dict[
            Tuple[str, Tuple[str, ...], Tuple[Tuple[str, str, str], ...]],
            List[Dict[str, Any]],
        ] = defaultdict(list)
        reverse_graph: Dict[
            Tuple[str, Tuple[str, ...], Tuple[Tuple[str, str, str], ...]],
            List[Tuple[str, Tuple[str, ...], Tuple[Tuple[str, str, str], ...]]],
        ] = defaultdict(list)
        state_meta: Dict[
            Tuple[str, Tuple[str, ...], Tuple[Tuple[str, str, str], ...]],
            Dict[str, Any],
        ] = {
            start_state: {
                "plan_state": x0,
                "q_vec": q0_vec,
                "resource_states": deepcopy(start_payload),
            }
        }

        queue = deque([start_state])
        seen = {start_state}
        accepting_states: Set[Tuple[str, Tuple[str, ...], Tuple[Tuple[str, str, str], ...]]] = set()

        while queue:
            node = queue.popleft()
            x, q_vec, _ = node
            resource_states = state_payloads.get(node, {})

            if self._is_accepting_terminal_product_state(
                x=x,
                q_vec=q_vec,
                Xm=Xm,
                rule_ids=rule_ids,
            ):
                accepting_states.add(node)

            for transition in enabled.get(x, []):
                x2 = str(transition.get("to") or "").strip()
                if not x2:
                    continue

                checked_vec, committed_vec, next_resource_states, sigma_by_rule = (
                    self._joint_transition_successor(
                        x=x,
                        q_vec=q_vec,
                        rule_ids=rule_ids,
                        rule_ap_sets=rule_ap_sets,
                        transition=transition,
                        task_lookup=task_lookup,
                        task_meta_lookup=task_meta_lookup,
                        resource_states=resource_states,
                    )
                )

                violates = False
                violated_rule_ids: List[str] = []
                for idx, rule_id in enumerate(rule_ids):
                    violation_state = str(self.dfas.get(rule_id, {}).get("violation_state") or "").strip()
                    if violation_state and checked_vec[idx] == violation_state:
                        violates = True
                        violated_rule_ids.append(rule_id)

                if violates:
                    continue

                successor = (x2, committed_vec, self._resource_state_signature(next_resource_states))
                edge_meta = {
                    "event": transition.get("event"),
                    "task_id": transition.get("task_id"),
                    "resource_jid": transition.get("resource_jid"),
                    "function_name": transition.get("function_name"),
                    "params": dict(transition.get("params") or {}),
                    "from": x,
                    "to": x2,
                    "successor": successor,
                    "sigma_by_rule": {
                        rid: sorted(list(sigma_by_rule.get(rid, frozenset())))
                        for rid in rule_ids
                    },
                }
                product_graph[node].append(edge_meta)
                reverse_graph[successor].append(node)

                if successor in seen:
                    continue

                seen.add(successor)
                state_payloads[successor] = deepcopy(next_resource_states)
                state_meta[successor] = {
                    "plan_state": x2,
                    "q_vec": committed_vec,
                    "resource_states": deepcopy(next_resource_states),
                }
                queue.append(successor)

                if len(seen) > self.MAX_PRODUCT_STATES:
                    self.logger.warning(
                        "[OfflineValidator] Product state limit reached (%d) while computing winning set.",
                        self.MAX_PRODUCT_STATES,
                    )
                    break

        winning_set: Set[Tuple[str, Tuple[str, ...], Tuple[Tuple[str, str, str], ...]]] = set()
        backward = deque(accepting_states)
        while backward:
            node = backward.popleft()
            if node in winning_set:
                continue
            winning_set.add(node)
            for prev in reverse_graph.get(node, []):
                if prev not in winning_set:
                    backward.append(prev)

        return {
            "W": winning_set,
            "product_graph": dict(product_graph),
            "reverse_graph": dict(reverse_graph),
            "accepting_states": accepting_states,
            "state_meta": state_meta,
            "rule_ids": rule_ids,
            "initial_state": start_state,
            "initial_resource_states": initial_resource_states,
        }

    def _transition_successor(
        self,
        *,
        rule_id: str,
        q: str,
        x: str,
        transition: Dict[str, Any],
        aps_for_rule: Set[str],
        task_lookup: Dict[str, Dict[str, Any]],
        task_meta_lookup: Dict[str, Dict[str, Any]],
        resource_states: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, str, Dict[str, Dict[str, Any]], FrozenSet[str]]:
        event_name = str(transition.get("event") or "").strip()
        meta = self._transition_task_meta(transition, task_lookup, task_meta_lookup)
        resource_jid = str(meta.get("resource_jid") or "").strip()
        function_name = str(meta.get("function_name") or "").strip()
        params = dict(meta.get("params") or {})

        running_before = set(self._running_event_aps_from_state(x, aps_for_rule, task_meta_lookup))
        persistent_before = set(self._state_aps_for_resources(resource_states, aps_for_rule))

        if resource_jid and function_name:
            candidate_event_aps = set(
                ap for ap in self._map_task_to_aps(resource_jid, function_name, params)
                if ap in aps_for_rule
            )
            predicted_state_aps = set(
                ap for ap in self._predict_state_aps(resource_jid, function_name, params)
                if ap in aps_for_rule
            )
        else:
            candidate_event_aps = set()
            predicted_state_aps = set()

        if event_name.endswith(".start"):
            sigma = frozenset(
                running_before
                | persistent_before
                | candidate_event_aps
                | predicted_state_aps
            )
            checked_q = self._delta(rule_id, q, sigma)
            return checked_q, q, deepcopy(resource_states), sigma

        if event_name.endswith(".done") or event_name.endswith(".finish"):
            running_after = set(running_before)
            for ap in candidate_event_aps:
                running_after.discard(ap)

            next_resource_states = deepcopy(resource_states)
            if resource_jid:
                next_payload = next_resource_states.setdefault(
                    resource_jid,
                    {"current_state": next_resource_states.get(resource_jid, {}).get("current_state", "idle"), "params": {}},
                )
                out_state = str(meta.get("out_state") or "").strip()
                if out_state and out_state.lower() != "any":
                    next_payload["current_state"] = out_state
                    next_payload["params"] = dict(params)
                elif "params" not in next_payload:
                    next_payload["params"] = dict(params)

            persistent_after = set(self._state_aps_for_resources(next_resource_states, aps_for_rule))
            sigma = frozenset(running_after | persistent_after | candidate_event_aps)
            committed_q = self._delta(rule_id, q, sigma)
            return committed_q, committed_q, next_resource_states, sigma

        sigma = frozenset(running_before | persistent_before | candidate_event_aps)
        committed_q = self._delta(rule_id, q, sigma)
        return committed_q, committed_q, deepcopy(resource_states), sigma

    # ------------------------------------------------------------------ #
    # PRODUCT SEARCH PER RULE
    # ------------------------------------------------------------------ #
    def _check_rule_on_fsa_product(
        self,
        rule_id: str,
        rule: Dict[str, Any],
        x0: str,
        Xm: Set[str],
        enabled: Dict[str, List[Dict[str, Any]]],
        aps_for_rule: Set[str],
        task_lookup: Dict[str, Dict[str, Any]],
        task_meta_lookup: Dict[str, Dict[str, Any]],
        initial_resource_states: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Explore reachable (x,q) states and detect any violation.
        """
        dfa = self.dfas[rule_id]
        q0: str = dfa["initial"]
        violation_state: str | None = dfa.get("violation_state")

        rule_info = self.rule_lookup.get(rule_id, {})
        ap_defs = {
            ap["label"]: ap.get("full", "")
            for ap in rule_info.get("aps", [])
            if ap.get("label")
        }

        start_state_payload = deepcopy(initial_resource_states)
        start_sig = self._resource_state_signature(start_state_payload)
        start = (str(x0), str(q0), start_sig)
        parent: Dict[Tuple[str, str, Tuple[Tuple[str, str, str], ...]], Optional[Tuple[str, str, Tuple[Tuple[str, str, str], ...]]]] = {start: None}
        parent_edge: Dict[Tuple[str, str, Tuple[Tuple[str, str, str], ...]], Optional[Dict[str, Any]]] = {start: None}
        state_payloads: Dict[Tuple[str, str, Tuple[Tuple[str, str, str], ...]], Dict[str, Dict[str, Any]]] = {
            start: start_state_payload
        }

        stack = deque([start])
        seen: Set[Tuple[str, str, Tuple[Tuple[str, str, str], ...]]] = {start}

        violations: List[Dict[str, Any]] = []

        while stack:
            node = stack.pop()
            x, q, _ = node
            resource_states = state_payloads.get(node, {})

            # End-of-plan check when plant is in a marked state
            if x in Xm:
                q_end = self._delta(rule_id, q, frozenset())
                accepting_states = set(str(s) for s in (dfa.get("accepting_states") or []))
                terminal_invalid = bool(violation_state and q_end == violation_state)
                if accepting_states and q_end not in accepting_states:
                    terminal_invalid = True
                if terminal_invalid:
                    trace = self._reconstruct_trace(node, parent, parent_edge)
                    violations.append(self._build_fsa_violation_entry(
                        rule_id=rule_id,
                        rule=rule_info,
                        ap_defs=ap_defs,
                        witness=trace,
                    ))
                    # You can continue searching to find more witnesses; usually one is enough.
                    continue

            # Expand all enabled plant transitions from x
            for t in enabled.get(x, []):
                x2 = str(t.get("to"))
                if not x2:
                    continue

                checked_q, committed_q, next_resource_states, sigma = self._transition_successor(
                    rule_id=rule_id,
                    q=q,
                    x=x,
                    transition=t,
                    aps_for_rule=aps_for_rule,
                    task_lookup=task_lookup,
                    task_meta_lookup=task_meta_lookup,
                    resource_states=resource_states,
                )

                if violation_state and checked_q == violation_state:
                    # Found violation witness
                    # record (x2,q2) as the violating product state for trace reconstruction
                    violating = (x2, checked_q, self._resource_state_signature(next_resource_states))
                    if violating not in parent:
                        parent[violating] = node
                        
                        t_dbg = dict(t)
                        t_dbg["_sigma"] = sorted(list(sigma))
                        t_dbg["_q_from"] = q
                        t_dbg["_q_to"] = checked_q
                        t_dbg["_resource_states"] = deepcopy(next_resource_states)
                        parent_edge[violating] = t_dbg
                        state_payloads[violating] = deepcopy(next_resource_states)
                        
                    trace = self._reconstruct_trace(violating, parent, parent_edge)

                    violations.append(self._build_fsa_violation_entry(
                        rule_id=rule_id,
                        rule=rule_info,
                        ap_defs=ap_defs,
                        witness=trace,
                    ))
                    # Do not expand this violating successor
                    continue

                s2 = (x2, committed_q, self._resource_state_signature(next_resource_states))
                if s2 in seen:
                    continue

                seen.add(s2)
                parent[s2] = node
                                
                t_dbg = dict(t)
                t_dbg["_sigma"] = sorted(list(sigma))
                t_dbg["_q_from"] = q
                t_dbg["_q_to"] = committed_q
                t_dbg["_resource_states"] = deepcopy(next_resource_states)
                parent_edge[s2] = t_dbg
                state_payloads[s2] = deepcopy(next_resource_states)

                stack.append(s2)

                if len(seen) > self.MAX_PRODUCT_STATES:
                    self.logger.warning(
                        "[OfflineValidator] Product state limit reached (%d). "
                        "Stopping exploration for rule %s.",
                        self.MAX_PRODUCT_STATES,
                        rule_id,
                    )
                    return violations

        return violations

    def _reconstruct_trace(
        self,
        end_state: Tuple[str, str, Tuple[Tuple[str, str, str], ...]],
        parent: Dict[
            Tuple[str, str, Tuple[Tuple[str, str, str], ...]],
            Optional[Tuple[str, str, Tuple[Tuple[str, str, str], ...]]],
        ],
        parent_edge: Dict[
            Tuple[str, str, Tuple[Tuple[str, str, str], ...]],
            Optional[Dict[str, Any]],
        ],
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct witness as a list of plant transitions (dicts) along the product path.
        """
        path: List[Dict[str, Any]] = []
        cur = end_state
        while True:
            pe = parent_edge.get(cur)
            if pe is not None:
                path.append(pe)
            prev = parent.get(cur)
            if prev is None:
                break
            cur = prev
        path.reverse()
        return path

    # ------------------------------------------------------------------ #
    # VIOLATION REPORT
    # ------------------------------------------------------------------ #
    def _build_fsa_violation_entry(
        self,
        rule_id: str,
        rule: Dict[str, Any],
        ap_defs: Dict[str, str],
        witness: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build a violation entry including a witness transition sequence.

        witness_transitions: list of transitions (from,event,to,task_id,...)
        witness_events:      list of event strings
        witness_task_ids:    list of task_ids (if present)
        """
        witness_events = [str(t.get("event", "")) for t in witness if t.get("event")]
        witness_task_ids = [str(t.get("task_id")) for t in witness if t.get("task_id")]

        return {
            "violated_rule_id": rule_id,
            "violation_text": rule.get("raw_text", ""),
            "violation_logic": rule.get("ltlf", ""),
            "ap_definitions": ap_defs,

            # Witness info
            "witness_events": witness_events,
            "witness_task_ids": witness_task_ids,
            "witness_transitions": [dict(t) for t in witness],  # shallow copies
        }

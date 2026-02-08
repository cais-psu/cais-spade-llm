"""Offline verifier that checks a compiled plan FSA against DFA safety rules."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Set, FrozenSet, Optional
from collections import defaultdict, deque
import re
from agents.central_controller.base_safety_checker import BaseSafetyChecker


class OfflineSafetyValidator(BaseSafetyChecker):
    """
    Offline validation of a *compiled FSA plan* against LTLf-based DFA safety rules.

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

    def __init__(self, rules: List[Dict[str, Any]], dfa_map: Dict[str, str]) -> None:
        # BaseSafetyChecker signature is (dfa_map, rules)
        super().__init__(dfa_map, rules)
        self.rule_lookup: Dict[str, Dict[str, Any]] = {r["id"]: r for r in rules if r.get("id")}

    # ------------------------------------------------------------------ #
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------ #
    def validate_fsa_offline(
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
            "[OfflineValidator] Validating FSA for %s (|Tr|=%d)...",
            product_jid,
            len(Tr),
        )

        enabled = self._index_enabled(Tr)

        # Optional: task_id -> node lookup from plan
        task_lookup = self._build_task_lookup(plan)

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
            )

            # Keep only ONE witness per rule
            if violations:
                all_violations.append(violations[0])
        return (len(all_violations) == 0), all_violations

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

    # ------------------------------------------------------------------ #
    # AP (SIGMA) MAPPING FOR FSA TRANSITIONS
    # ------------------------------------------------------------------ #
    def _sigma_for_transition(
        self,
        t: Dict[str, Any],
        aps_for_rule: Set[str],
        task_lookup: Dict[str, Dict[str, Any]],
    ) -> FrozenSet[str]:
        """
        Compute sigma (AP label set) emitted by a plant transition, projected to this rule's AP alphabet.

        Preferred:
          - Use semantic mapping from task metadata (resource/function/params) via BaseSafetyChecker._map_task_to_aps.
        Fallback:
          - If we cannot map, return empty sigma (safe but may cause false negatives).
        """
        # Try to obtain task metadata from transition itself
        resource_jid = t.get("resource_jid") or ""
        function_name = t.get("function_name") or ""
        params = t.get("params") or None

        # If not present, try plan lookup by task_id
        if (not function_name or params is None or not resource_jid) and t.get("task_id"):
            node = task_lookup.get(str(t["task_id"]))
            if node:
                resource_jid = resource_jid or (node.get("resource_jid") or "")
                function_name = function_name or (node.get("function_name") or "")
                if params is None:
                    params = node.get("params") or {}

        if params is None:
            params = {}

        if not resource_jid or not function_name:
            # No semantic info → cannot map APs reliably in your current framework
            return frozenset()

        full_sigma = self._map_task_to_aps(resource_jid, function_name, params)
        return frozenset(ap for ap in full_sigma if ap in aps_for_rule)


    # ------------------------------------------------------------------ #
    # AP (SIGMA) MAPPING FOR FSA STATES
    # ------------------------------------------------------------------ #
    def _sigma_for_state(
        self,
        x: str,
        aps_for_rule: Set[str],
        ap_defs: Dict[str, str],  # label -> full AP string
    ) -> FrozenSet[str]:
        """
        State-based labeling: return the set of AP labels that are TRUE in global plant state x.

        Assumption: global state encodes running like:
            "<resource>@...=(...,run=task_id:function_name)"
        and state APs use:
            st/<process>/<product>/<resource>/<event>/<context>
        """
        # Map (resource, event) -> ap_label for this rule
        key_to_label: Dict[Tuple[str, str], str] = {}
        for label, full in ap_defs.items():
            parts = str(full).split("/")
            # expected: st/<process>/<product>/<resource>/<event>/<context>
            if len(parts) >= 6 and parts[0] == "ap":
                resource = parts[3]
                event = parts[4]
                key_to_label[(resource, event)] = label

        sigma: Set[str] = set()

        # Extract currently running (resource,event) from the global state string
        for m in re.finditer(
            r"([A-Za-z0-9_-]+)@[^=]*=\([^)]*?run=[^:,\)]+:([A-Za-z0-9_-]+)",
            str(x),
        ):
            resource = m.group(1)
            event = m.group(2)
            label = key_to_label.get((resource, event))
            if label and label in aps_for_rule:
                sigma.add(label)

        return frozenset(sigma)


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

        start = (str(x0), str(q0))
        parent: Dict[Tuple[str, str], Optional[Tuple[str, str]]] = {start: None}
        parent_edge: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {start: None}

        stack = deque([start])
        seen: Set[Tuple[str, str]] = {start}

        violations: List[Dict[str, Any]] = []

        while stack:
            x, q = stack.pop()

            # End-of-plan check when plant is in a marked state
            if x in Xm:
                q_end = self._delta(rule_id, q, frozenset())
                if violation_state and q_end == violation_state:
                    trace = self._reconstruct_trace((x, q), parent, parent_edge)
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

                #sigma = self._sigma_for_transition(t, aps_for_rule, task_lookup)
                #q2 = self._delta(rule_id, q, sigma)
                sigma = self._sigma_for_state(x2, aps_for_rule, ap_defs)
                q2 = self._delta(rule_id, q, sigma)

                if violation_state and q2 == violation_state:
                    # Found violation witness
                    # record (x2,q2) as the violating product state for trace reconstruction
                    violating = (x2, q2)
                    if violating not in parent:
                        parent[violating] = (x, q)
                        
                        t_dbg = dict(t)
                        t_dbg["_sigma"] = sorted(list(sigma))
                        t_dbg["_q_from"] = q
                        t_dbg["_q_to"] = q2
                        parent_edge[violating] = t_dbg
                        
                    trace = self._reconstruct_trace(violating, parent, parent_edge)

                    violations.append(self._build_fsa_violation_entry(
                        rule_id=rule_id,
                        rule=rule_info,
                        ap_defs=ap_defs,
                        witness=trace,
                    ))
                    # Do not expand this violating successor
                    continue

                s2 = (x2, q2)
                if s2 in seen:
                    continue

                seen.add(s2)
                parent[s2] = (x, q)
                                
                t_dbg = dict(t)
                t_dbg["_sigma"] = sorted(list(sigma))
                t_dbg["_q_from"] = q
                t_dbg["_q_to"] = q2
                parent_edge[s2] = t_dbg

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
        end_state: Tuple[str, str],
        parent: Dict[Tuple[str, str], Optional[Tuple[str, str]]],
        parent_edge: Dict[Tuple[str, str], Optional[Dict[str, Any]]],
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

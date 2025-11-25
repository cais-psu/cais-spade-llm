from __future__ import annotations

from typing import Dict, List, Tuple, Iterable, FrozenSet, Any, Optional
import re

#df
class SafetyMonitor:
    """
    High-level safety monitor that wraps all safety DFAs.

    Usage:

        monitor = SafetyMonitor(dfa_dots)  # { "SAFE_1": dfa_dot_str, ... }

        allowed, meta = monitor.check(
            running_aps=["ap001"],
            candidate_aps=["ap003"],
        )

        if allowed:
            # safe to start task
        else:
            # blocked by meta["violated_rule"]
    """

    def __init__(self, dfa_dots: Dict[str, str], safety_rules: list[dict]) -> None:
        """
        Initialize from:
          - dfa_dots: {rule_id: DFA_DOT_string}
          - safety_rules: list of rules from safety_logic.json
        """
        self.safety_rules = safety_rules

        # Per-rule DFA data:
        self.transitions: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        self.init_state: Dict[str, str] = {}
        self.violation_state: Dict[str, Optional[str]] = {}
        self.ap_symbols: Dict[str, List[str]] = {}
        self.current_state: Dict[str, str] = {}

        for rule_id, dot_src in dfa_dots.items():
            self._parse_dot_for_rule(rule_id, dot_src)
    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def check(
        self,
        running_aps: Iterable[str],
        candidate_aps: Iterable[str],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Generic safety check for one scheduling decision.

        Inputs:
          running_aps   - AP labels for tasks currently executing
          candidate_aps - AP labels for the task(s) we want to start

        Returns:
          (allowed: bool, meta: dict)

        meta["violated_rule"] = id of the first rule that would be violated,
        or None if allowed.
        """
        sigma: FrozenSet[str] = frozenset(set(running_aps) | set(candidate_aps))

        allowed = True
        violated_rule: Optional[str] = None
        next_states: Dict[str, str] = {}

        # 1) Lookahead on each DFA
        for rule_id in self.transitions.keys():
            cur = self.current_state[rule_id]
            nxt = self._delta(rule_id, cur, sigma)

            # violation if we have a known violation_state and we reach it
            vio_state = self.violation_state.get(rule_id)
            violated = (vio_state is not None and nxt == vio_state)

            next_states[rule_id] = nxt

            if violated and allowed:
                allowed = False
                violated_rule = rule_id

                # ===== ▼ PRINT VIOLATION ▼ =====
                print(f"[SafetyMonitor] VIOLATION: rule={rule_id}, sigma={sorted(list(sigma))}")
                # ===============================

        # 2) Commit if safe
        if allowed:
            for rule_id, ns in next_states.items():
                self.current_state[rule_id] = ns

        return allowed, {"violated_rule": violated_rule}

    # ------------------------------------------------------------------ #
    # DOT parsing per rule
    # ------------------------------------------------------------------ #
    def _parse_dot_for_rule(self, rule_id: str, dot_src: str) -> None:
        """
        Parse a single DFA DOT string for a given rule_id and populate
        internal structures for that rule.
        """
        transitions: Dict[str, List[Tuple[str, str]]] = {}
        ap_list: List[str] = []
        init: Optional[str] = None
        violation: Optional[str] = None

        # Normalize whitespace
        text = " ".join(dot_src.split())

        # initial state: 'init -> 1;'
        m = re.search(r"init\s*->\s*([A-Za-z0-9_]+)\s*;", text)
        if m:
            init = m.group(1)

        # transitions: '1 -> 2 [label="..."];'
        pattern = re.compile(
            r"([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)\s*\[label=\"(.*?)\"\];"
        )
        for src, dst, label in pattern.findall(text):
            transitions.setdefault(src, []).append((label, dst))

            # detect violation state as any state with self-loop 'true'
            if src == dst and label.strip().lower() == "true":
                violation = src

            # collect AP symbols
            for ap in re.findall(r"(ap\d+)", label):
                if ap not in ap_list:
                    ap_list.append(ap)

        # Fallback if no init_state found
        if init is None and transitions:
            init = next(iter(transitions.keys()))

        if init is None:
            # malformed DFA; ignore this rule
            return

        # Store per-rule data
        self.transitions[rule_id] = transitions
        self.init_state[rule_id] = init
        self.violation_state[rule_id] = violation
        self.ap_symbols[rule_id] = ap_list
        self.current_state[rule_id] = init

    # ------------------------------------------------------------------ #
    # DFA transition + label evaluation
    # ------------------------------------------------------------------ #
    def _delta(self, rule_id: str, state: str, sigma: FrozenSet[str]) -> str:
        """
        Deterministic next-state function for a given rule.
        We pick the first transition whose label evaluates to True.
        If none match, stay in the same state.
        """
        for label, dst in self.transitions.get(rule_id, {}).get(state, []):
            if self._eval_label(rule_id, label, sigma):
                return dst
        return state  # no matching transition

    def _eval_label(self, rule_id: str, label: str, sigma: FrozenSet[str]) -> bool:
        """
        Evaluate a transition label of the form:

          (~ap001 & ~ap002) | (ap003 & ap004) | true

        against the set of currently true APs (sigma).
        """
        label = label.strip()
        if label.lower() == "true":
            return True
        if not label:
            return False

        expr = label
        # Replace MONA-style logic with Python boolean operators
        expr = expr.replace("&", " and ")
        expr = expr.replace("|", " or ")
        expr = expr.replace("~", " not ")
        expr = re.sub(r"\btrue\b", "True", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bfalse\b", "False", expr, flags=re.IGNORECASE)

        # Build environment: each apNNN is True iff apNNN ∈ sigma
        env = {ap: (ap in sigma) for ap in self.ap_symbols.get(rule_id, [])}

        try:
            result = eval(expr, {"__builtins__": {}}, env)
        except Exception:
            # conservative fallback: if label can't be evaluated, treat as False
            result = False

        return bool(result)

    # ------------------------------------------------------------------ #
    # Task → AP mapping (no hard-coded destinations)
    # ------------------------------------------------------------------ #
    def _map_task_to_aps(
        self,
        resource_jid: str,
        function_name: str,
        params: dict,
    ) -> list[str]:
        """
        Map a task (resource, function) to AP labels using self.safety_rules.

        No hard-coded destination names, no hard-coded param keys.
        It only looks at the AP 'full' string:

            evt/<process>/<product>/<resource>/<event>/<context>
        """
        labels: list[str] = []
        res_short = resource_jid.split("@")[0]  # "xarm6", "ur5e"

        for rule in self.safety_rules:
            for ap in rule.get("aps", []):
                full = ap.get("full", "")
                label = ap.get("label")
                if not full or not label:
                    continue

                parts = full.split("/")
                if len(parts) < 6:
                    continue

                ap_resource = parts[3]  # resource from AP
                ap_event    = parts[4]  # event from AP

                if ap_resource == res_short and ap_event == function_name:
                    labels.append(label)

        return labels

from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Tuple, Optional, FrozenSet

class BaseSafetyChecker:
    """
    Base class containing shared logic for:
    1. Parsing DFA DOT strings.
    2. Mapping tasks to Atomic Propositions (APs).
    3. Evaluating DFA transitions (The 'Physics' of the safety logic).
    
    This class is STATELESS regarding the robot execution. 
    It only holds the rules.
    """

    def __init__(self, dfa_map: Dict[str, str], safety_rules: List[Dict[str, Any]]) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.safety_rules = safety_rules
        
        # Internal DFA storage: { rule_id: { "initial": "1", "transitions": {...} } }
        self.dfas: Dict[str, Dict[str, Any]] = {}

        # Parse all DOT strings immediately
        for rule_id, dot_str in dfa_map.items():
            self.dfas[rule_id] = self._parse_dot(rule_id, dot_str)

    # ------------------------------------------------------------------ #
    # Shared: Task -> AP Mapping
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Shared: Task -> AP Mapping (generic, no hard-coded fn names)
    # ------------------------------------------------------------------ #
    def _map_task_to_aps(self, resource_jid: str, function_name: str, params: dict) -> List[str]:
        """
        Maps a task execution to a list of AP labels based on loaded safety rules.

        Matching logic (per AP):
          - resource:  exact match or wildcard ("any", "robot")
          - event:     must equal function_name
          - product:   must match params["part_name"] (or "product") unless "any"
          - context:   if not "any", must match at least one param value
                       or one of rule["context"].values().
        """
        labels: List[str] = []

        # Normalize "ur5e@localhost" -> "ur5e"
        res_short = resource_jid.split("@")[0] if "@" in resource_jid else resource_jid

        # Task-level fields
        task_product = (
            params.get("part_name")
            or params.get("product")
            or "any"
        )

        # Normalize all param values to strings for comparison
        param_value_strings = {str(v) for v in params.values() if v is not None}

        for rule in self.safety_rules:
            rule_context = rule.get("context") or {}
            # Also consider rule context values as potential matches
            rule_ctx_value_strings = {str(v) for v in rule_context.values()}

            for ap in rule.get("aps", []):
                full = ap.get("full") or ""
                label = ap.get("label")
                if not full or not label:
                    continue

                parts = full.split("/")
                # Expected: evt/<process>/<product>/<resource>/<event>/<context>
                if len(parts) < 6:
                    continue

                _, ap_process, ap_product, ap_resource, ap_event, ap_context = parts[:6]

                # 1) Resource match (with wildcard support)
                if ap_resource not in ("any", "robot") and ap_resource != res_short:
                    continue

                # 2) Event / function name match
                if ap_event != function_name:
                    continue

                # 3) Product match (MCP vs SG, etc.)
                if ap_product != "any" and ap_product != task_product:
                    continue

                # 4) Context match (generic, no function_name branching)
                if ap_context != "any":
                    # Context must match either:
                    #   - one of this rule's context values, OR
                    #   - one of this task's parameter values
                    if (
                        str(ap_context) not in rule_ctx_value_strings
                        and str(ap_context) not in param_value_strings
                    ):
                        continue

                # If all predicates passed, this AP applies to this task.
                labels.append(label)

        return labels

    # ------------------------------------------------------------------ #
    # Shared: DFA Transition Logic (The Math)
    # ------------------------------------------------------------------ #
    def _delta(self, rule_id: str, current_state: str, sigma: FrozenSet[str]) -> str:
        """
        Calculates the next state for a specific rule given the current state and active APs.
        """
        dfa_data = self.dfas.get(rule_id)
        if not dfa_data:
            return current_state

        transitions = dfa_data.get("transitions", {}).get(current_state, [])
        ap_symbols = dfa_data.get("ap_symbols", [])
        
        for label, dst in transitions:
            if self._eval_label(label, sigma, ap_symbols):
                return dst
        
        # If no transition matches, remain in current state (Stuttering)
        return current_state

    def _eval_label(self, label: str, sigma: FrozenSet[str], rule_aps: List[str]) -> bool:
        """
        Evaluates boolean label expression (e.g., "ap001 & !ap002").
        """
        label = label.strip()
        if label.lower() == "true":
            return True
        if not label or label.lower() == "false":
            return False

        # Prepare expression for Python eval
        expr = label.replace("&", " and ").replace("|", " or ").replace("~", " not ")
        expr = re.sub(r"\btrue\b", "True", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bfalse\b", "False", expr, flags=re.IGNORECASE)

        # Build environment: apNNN is True only if it is in sigma
        env = {ap: (ap in sigma) for ap in rule_aps}

        try:
            return bool(eval(expr, {"__builtins__": {}}, env))
        except Exception:
            self.logger.error(f"Failed to evaluate label: {label}")
            return False

    # ------------------------------------------------------------------ #
    # Shared: DOT Parsing
    # ------------------------------------------------------------------ #
    def _parse_dot(self, rule_id: str, dot_src: str) -> Dict[str, Any]:
        """
        Parses a DOT string into a dictionary structure.
        """
        transitions: Dict[str, List[Tuple[str, str]]] = {}
        ap_list: List[str] = []
        init: Optional[str] = None
        violation: Optional[str] = None

        text = " ".join(dot_src.split())

        # Match initial state: 'init -> 1;'
        m_init = re.search(r"init\s*->\s*([A-Za-z0-9_]+)\s*;", text)
        if m_init:
            init = m_init.group(1)

        # Match transitions: '1 -> 2 [label="..."];'
        pattern = re.compile(r"([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)\s*\[label=\"(.*?)\"\];")
        
        for src, dst, label in pattern.findall(text):
            transitions.setdefault(src, []).append((label, dst))

            if src == dst and label.strip().lower() == "true":
                violation = src

            for ap in re.findall(r"(ap\d+)", label):
                if ap not in ap_list:
                    ap_list.append(ap)

        if init is None and transitions:
            init = next(iter(transitions.keys()))

        return {
            "initial": init,
            "transitions": transitions,
            "violation_state": violation,
            "ap_symbols": ap_list
        }
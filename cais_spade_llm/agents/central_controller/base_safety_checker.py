"""Shared DFA parsing and AP-mapping utilities for safety checking."""

from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Tuple, Optional, FrozenSet
from urllib.parse import parse_qsl

class BaseSafetyChecker:
    """
    Base class containing shared logic for:
    1. Parsing DFA DOT strings.
    2. Mapping tasks to Atomic Propositions (APs).
    3. Evaluating DFA transitions (The 'Physics' of the safety logic).
    
    This class is STATELESS regarding the robot execution. 
    It only holds the rules.
    """

    def __init__(
        self,
        dfa_map: Dict[str, str],
        safety_rules: List[Dict[str, Any]],
        tools_catalog: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Load safety rules and parse DFA DOT sources into transition tables."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.safety_rules = safety_rules
        self.tools_catalog: List[Dict[str, Any]] = list(tools_catalog or [])
        
        # Internal DFA storage: { rule_id: { "initial": "1", "transitions": {...} } }
        self.dfas: Dict[str, Dict[str, Any]] = {}

        # Parse all DOT strings immediately
        for rule_id, dot_str in dfa_map.items():
            self.dfas[rule_id] = self._parse_dot(rule_id, dot_str)

    @staticmethod
    def _context_scalar_text(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).strip()

    @staticmethod
    def _resource_short_name(resource_jid: str) -> str:
        token = str(resource_jid or "").strip()
        if "@" in token:
            token = token.split("@", 1)[0]
        return token.lower()

    @staticmethod
    def _task_product_name(params: dict[str, Any]) -> str:
        product = (
            params.get("part_name")
            or params.get("product")
            or "any"
        )
        return str(product).lower()

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
          - resource:  exact match or wildcard ("any", legacy "robot")
          - event:     must equal function_name
          - product:   must match params["part_name"] (or "product") unless "any"
          - context:   if not "any", either:
                       * composite key=value pairs joined by "&" must all match, or
                       * legacy single-token matching falls back to raw value comparison
        """
        labels: List[str] = []

        # Normalize "<resource>@<host>" -> "<resource>"
        res_short = self._resource_short_name(resource_jid)

        # Task-level fields
        task_product = self._task_product_name(params)

        # Normalize all param values to strings for comparison
        param_value_strings = {
            self._context_scalar_text(v) for v in params.values() if v is not None
        }

        for rule in self.safety_rules:
            rule_context = rule.get("context") or {}
            # Also consider rule context values as potential matches
            rule_ctx_value_strings = {
                self._context_scalar_text(v) for v in rule_context.values()
            }

            for ap in rule.get("aps", []):
                full = ap.get("full") or ""
                label = ap.get("label")
                if not full or not label:
                    continue

                parts = full.split("/")
                # Expected: evt/<process>/<product>/<resource>/<event>/<context>
                if len(parts) < 6:
                    continue

                ap_prefix, _, ap_product, ap_resource, ap_event, ap_context = parts[:6]
                if ap_prefix not in {"ap", "ap_event"}:
                    continue

                # 1) Resource match (with wildcard support)
                if ap_resource not in ("any", "robot") and ap_resource != res_short:
                    continue

                # 2) Event / function name match
                if ap_event != function_name:
                    continue

                # 3) Product match (MCP vs SG, etc.)
                if ap_product != "any" and str(ap_product).lower() != task_product:
                    continue

                # 4) Context match (supports composite serialized context segments)
                if not self._context_matches(
                    ap_context=str(ap_context),
                    params=params,
                    param_value_strings=param_value_strings,
                    rule_context=rule_context,
                    rule_ctx_value_strings=rule_ctx_value_strings,
                ):
                    continue

                # If all predicates passed, this AP applies to this task.
                labels.append(label)

        return labels

    def _map_state_to_aps(self, resource_jid: str, current_state: str, params: dict) -> List[str]:
        """
        Maps a resource's persistent state to matching state AP labels.

        Matching logic mirrors _map_task_to_aps, but:
          - only matches ap_state/sp prefixes
          - compares the AP event/state segment against current_state
        """
        labels: List[str] = []
        res_short = self._resource_short_name(resource_jid)
        task_product = self._task_product_name(params)

        param_value_strings = {
            self._context_scalar_text(v) for v in params.values() if v is not None
        }

        for rule in self.safety_rules:
            rule_context = rule.get("context") or {}
            rule_ctx_value_strings = {
                self._context_scalar_text(v) for v in rule_context.values()
            }

            for ap in rule.get("aps", []):
                full = ap.get("full") or ""
                label = ap.get("label")
                if not full or not label:
                    continue

                parts = full.split("/")
                if len(parts) < 6:
                    continue

                ap_prefix, _, ap_product, ap_resource, ap_state, ap_context = parts[:6]
                if ap_prefix not in {"ap_state", "sp"}:
                    continue

                if ap_resource not in ("any", "robot") and ap_resource != res_short:
                    continue

                if ap_state != current_state:
                    continue

                if ap_product != "any" and str(ap_product).lower() != task_product:
                    continue

                if not self._context_matches(
                    ap_context=str(ap_context),
                    params=params,
                    param_value_strings=param_value_strings,
                    rule_context=rule_context,
                    rule_ctx_value_strings=rule_ctx_value_strings,
                ):
                    continue

                labels.append(label)

        return labels

    def _tool_rows_for_action(self, resource_jid: str, function_name: str) -> List[Dict[str, Any]]:
        res_short = self._resource_short_name(resource_jid)
        rows: List[Dict[str, Any]] = []
        for row in self.tools_catalog:
            if not isinstance(row, dict):
                continue
            if str(row.get("function", "")).strip() != str(function_name or "").strip():
                continue
            owner = self._resource_short_name(str(row.get("function_owner_agent", "")).strip())
            if owner and owner != res_short:
                continue
            rows.append(row)
        return rows

    def _predict_state_aps(
        self,
        resource_jid: str,
        function_name: str,
        params: dict[str, Any],
    ) -> List[str]:
        """
        Predict which state APs would become true if the task finishes successfully.
        """
        predicted: List[str] = []
        for row in self._tool_rows_for_action(resource_jid, function_name):
            out_state = str(row.get("out_state", "")).strip()
            if not out_state or out_state.lower() == "any":
                continue
            predicted.extend(self._map_state_to_aps(resource_jid, out_state, params))
        deduped: List[str] = []
        seen: set[str] = set()
        for label in predicted:
            if label in seen:
                continue
            seen.add(label)
            deduped.append(label)
        return deduped

    @staticmethod
    def _context_pairs(ap_context: str) -> Optional[list[tuple[str, str]]]:
        token = str(ap_context or "").strip()
        if not token or token == "any" or "=" not in token:
            return None
        try:
            pairs = parse_qsl(token, keep_blank_values=True, strict_parsing=False)
        except Exception:
            return None
        return [(str(k), str(v)) for k, v in pairs if str(k)]

    @classmethod
    def _context_matches(
        cls,
        *,
        ap_context: str,
        params: dict[str, Any],
        param_value_strings: set[str],
        rule_context: dict[str, Any],
        rule_ctx_value_strings: set[str],
    ) -> bool:
        token = str(ap_context or "").strip()
        if token == "any":
            return True

        pairs = cls._context_pairs(token)
        if not pairs:
            return token in rule_ctx_value_strings or token in param_value_strings

        for key, value in pairs:
            if key in params:
                if cls._context_scalar_text(params.get(key)) != value:
                    return False
                continue
            if key in rule_context:
                if cls._context_scalar_text(rule_context.get(key)) != value:
                    return False
                continue
            if value not in param_value_strings and value not in rule_ctx_value_strings:
                return False
        return True

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
        expr = label.replace("&", " and ").replace("|", " or ").replace("~", " not ").replace("!", " not ")
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
        accepting: set[str] = set()

        text = " ".join(dot_src.split())
        state_pat = r"[A-Za-z0-9_\.]+"

        # Match initial state: 'init -> 1;'
        m_init = re.search(rf"init\s*->\s*({state_pat})\s*;", text)
        if m_init:
            init = m_init.group(1)

        for state_blob in re.findall(
            rf"node\s*\[shape\s*=\s*doublecircle\]\s*;\s*([^;]+)\s*;",
            text,
            flags=re.IGNORECASE,
        ):
            for state in re.findall(state_pat, state_blob):
                accepting.add(state)

        # Match transitions: '1 -> 2 [label="..."];'
        pattern = re.compile(rf"({state_pat})\s*->\s*({state_pat})\s*\[label=\"(.*?)\"\];")
        
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
            "accepting_states": sorted(accepting),
            "ap_symbols": ap_list
        }

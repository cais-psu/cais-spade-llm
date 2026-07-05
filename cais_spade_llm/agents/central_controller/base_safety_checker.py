"""Shared DFA parsing and AP-mapping utilities for safety checking."""

from __future__ import annotations

import json
import logging
import re
from typing import Any
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
        dfa_dots: dict[str, str],
        safety_rules: list[dict],
        tools_catalog: list[dict[str, Any]] | None = None,
    ) -> None:
        """Load safety rules and parse DFA DOT sources into transition tables."""
        self.logger = logging.getLogger("BaseSafetyChecker")
        self.safety_rules = safety_rules or []
        self.tools_catalog = tools_catalog or []
        
        # Performance optimization caches
        self._compiled_expr_cache: dict[str, Any] = {}
        self._eval_result_cache: dict[tuple[str, frozenset[str]], bool] = {}

        # Internal DFA storage: { rule_id: { "initial": "1", "transitions": {...} } }
        self.dfas: dict[str, dict[str, Any]] = {}

        # Parse all DOT strings immediately
        for rule_id, dot_str in dfa_dots.items():
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
            or params.get("part")
            or params.get("product")
            or "any"
        )
        return str(product).lower()

    @staticmethod
    def _fixed_state_fields() -> tuple[str, ...]:
        return (
            "resource_state",
            "resource_location",
            "held_part",
            "part_state",
            "part_location",
        )

    @classmethod
    def _task_event_tokens(cls, function_name: str, params: dict[str, Any]) -> set[str]:
        tokens = {
            str(function_name or "").strip(),
            str(params.get("event_name") or "").strip(),
            str(params.get("function") or "").strip(),
            str(params.get("function_name") or "").strip(),
        }
        return {token for token in tokens if token}

    @classmethod
    def _task_id_tokens(cls, params: dict[str, Any]) -> set[str]:
        tokens = {
            str(params.get("task_id") or "").strip(),
            str(params.get("bridge_outline_id") or "").strip(),
            str(params.get("outline_id") or "").strip(),
            str(params.get("llm_outline_id") or "").strip(),
        }
        return {token for token in tokens if token}

    @classmethod
    def _state_surface_from_runtime(
        cls,
        current_state: str,
        params: dict[str, Any],
    ) -> dict[str, str]:
        surface: dict[str, str] = {}
        current_state_token = str(current_state or "").strip()
        if current_state_token:
            surface["resource_state"] = current_state_token
        for source in (
            params.get("outline_expected_start_state"),
            params,
            params.get("expected_end_state"),
            params.get("projected_outline_state"),
        ):
            if not isinstance(source, dict):
                continue
            for field in cls._fixed_state_fields():
                if field not in source:
                    continue
                value = cls._context_scalar_text(source.get(field))
                if value:
                    surface[field] = value
        return surface

    def _state_surface_from_prediction(self, params: dict[str, Any]) -> dict[str, str]:
        surface: dict[str, str] = {}
        for source in (
            params.get("expected_end_state"),
            params.get("projected_outline_state"),
        ):
            if not isinstance(source, dict):
                continue
            for field in self._fixed_state_fields():
                if field not in source:
                    continue
                value = self._context_scalar_text(source.get(field))
                if value:
                    surface[field] = value
        return surface

    @staticmethod
    def _state_surface_tokens(surface: dict[str, str]) -> set[str]:
        return {
            f"{str(field).strip()}={str(value).strip()}"
            for field, value in (surface or {}).items()
            if str(field).strip() and str(value).strip()
        }

    @classmethod
    def _ap_requires_source_task_ids(
        cls,
        ap: dict[str, Any],
        params: dict[str, Any],
    ) -> bool:
        source_task_ids = {
            str(token).strip()
            for token in (ap.get("source_task_ids") or [])
            if str(token).strip()
        }
        if not source_task_ids:
            return True
        return bool(source_task_ids & cls._task_id_tokens(params))

    @staticmethod
    def _parse_ap_descriptor(full: str) -> dict[str, str] | None:
        token = str(full or "").strip()
        parts = token.split("/")
        if len(parts) < 6:
            return None
        prefix, process, product, resource, symbol, context = parts[:6]
        return {
            "prefix": str(prefix or "").strip(),
            "process": str(process or "").strip(),
            "product": str(product or "").strip(),
            "resource": str(resource or "").strip(),
            "symbol": str(symbol or "").strip(),
            "context": str(context or "").strip(),
        }

    @staticmethod
    def _tool_signature(row: dict[str, Any]) -> str:
        payload = {
            "function_owner_agent": str(row.get("function_owner_agent") or "").strip(),
            "function": str(row.get("function") or "").strip(),
            "in_state": str(row.get("in_state") or "").strip(),
            "out_state": str(row.get("out_state") or "").strip(),
            "part_in_state": str(row.get("part_in_state") or "").strip(),
            "location_type": str(
                (row.get("context_mapping") or {}).get("location_type") or ""
            ).strip(),
            "location_param": str(
                (row.get("context_mapping") or {}).get("location_param") or ""
            ).strip(),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    # ------------------------------------------------------------------ #
    # Shared: Task -> AP Mapping
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Shared: Task -> AP Mapping (generic, no hard-coded fn names)
    # ------------------------------------------------------------------ #
    def _map_task_to_aps(self, resource_jid: str, function_name: str, params: dict) -> list[str]:
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
        labels: list[str] = []

        # Normalize "<resource>@<host>" -> "<resource>"
        res_short = self._resource_short_name(resource_jid)

        # Task-level fields
        task_product = self._task_product_name(params)
        event_tokens = self._task_event_tokens(function_name, params)

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
                expected_event = str(
                    ap.get("event_name")
                    or ap.get("function")
                    or ap_event
                    or ""
                ).strip()
                if expected_event not in event_tokens:
                    continue

                # 3) Product match (MCP vs SG, etc.)
                if ap_product != "any" and str(ap_product).lower() != task_product:
                    continue

                if not self._ap_requires_source_task_ids(ap, params):
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

    def _map_state_to_aps(self, resource_jid: str, current_state: str, params: dict) -> list[str]:
        """
        Maps a resource's persistent state to matching state AP labels.

        Matching logic mirrors _map_task_to_aps, but:
          - only matches ap_state/sp prefixes
          - compares the AP event/state segment against current_state
        """
        surface = self._state_surface_from_runtime(current_state, params)
        return self._map_state_surface_to_aps(resource_jid, surface, params)

    def _map_state_surface_to_aps(
        self,
        resource_jid: str,
        surface: dict[str, str],
        params: dict[str, Any],
    ) -> list[str]:
        """Maps a structured state surface to matching state AP labels."""
        labels: list[str] = []
        res_short = self._resource_short_name(resource_jid)
        task_product = self._task_product_name(params)
        state_tokens = self._state_surface_tokens(surface)
        current_state = str(surface.get("resource_state") or "").strip()

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

                field_name = str(ap.get("field") or "").strip()
                field_value = str(ap.get("value") or "").strip()
                expected_state = f"{field_name}={field_value}" if field_name and field_value else str(ap_state)
                if "=" in expected_state:
                    token = expected_state.strip()
                    if token not in state_tokens:
                        continue
                elif ap_state != current_state:
                    continue

                if ap_product != "any" and str(ap_product).lower() != task_product:
                    continue

                if not self._ap_requires_source_task_ids(ap, params):
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

    def _resource_state_signature_token(
        self,
        resource_jid: str,
        current_state: str,
        params: dict[str, Any] | None,
    ) -> str:
        """
        Build a stable, safety-relevant signature token for a resource state.

        Runtime task payloads often carry extra fields such as geometry snapshots
        that do not affect any state AP. Using the matched AP labels instead of
        the raw params keeps online and offline product-state signatures aligned.
        """
        labels = sorted(
            set(
                self._map_state_to_aps(
                    resource_jid,
                    str(current_state or "").strip(),
                    dict(params or {}),
                )
            )
        )
        return json.dumps(labels, separators=(",", ":"))

    def _tool_rows_for_action(self, resource_jid: str, function_name: str) -> list[dict[str, Any]]:
        res_short = self._resource_short_name(resource_jid)
        rows: list[dict[str, Any]] = []
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
    ) -> list[str]:
        """
        Predict which state APs would become true if the task finishes successfully.
        """
        predicted: list[str] = []
        for row in self._tool_rows_for_action(resource_jid, function_name):
            out_state = str(row.get("out_state", "")).strip()
            if not out_state or out_state.lower() == "any":
                continue
            predicted.extend(self._map_state_to_aps(resource_jid, out_state, params))
        projected_surface = self._state_surface_from_prediction(params)
        if projected_surface:
            predicted.extend(
                self._map_state_surface_to_aps(resource_jid, projected_surface, params)
            )
        deduped: list[str] = []
        seen: set[str] = set()
        for label in predicted:
            if label in seen:
                continue
            seen.add(label)
            deduped.append(label)
        return deduped

    @staticmethod
    def _context_pairs(ap_context: str) -> list[tuple[str, str]] | None:
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
            if str(value).strip().lower() == "any":
                continue
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
    def _delta(self, rule_id: str, current_state: str, sigma: frozenset[str]) -> str:
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

    def _eval_label(self, label: str, sigma: frozenset[str], rule_aps: list[str]) -> bool:
        """
        Evaluates boolean label expression (e.g., "ap001 & !ap002").
        """
        label = label.strip()
        if label.lower() == "true":
            return True
        if not label or label.lower() == "false":
            return False

        # Fast path 1: check result cache for exact inputs
        active_aps = frozenset([ap for ap in rule_aps if ap in sigma])
        cache_key = (label, active_aps)
        if cache_key in self._eval_result_cache:
            return self._eval_result_cache[cache_key]

        # Fast path 2: parse and compile expression only once
        if label not in self._compiled_expr_cache:
            expr = label.replace("&", " and ").replace("|", " or ").replace("~", " not ").replace("!", " not ")
            expr = re.sub(r"\btrue\b", "True", expr, flags=re.IGNORECASE)
            expr = re.sub(r"\bfalse\b", "False", expr, flags=re.IGNORECASE)
            try:
                self._compiled_expr_cache[label] = compile(expr.strip(), "<string>", "eval")
            except Exception:
                self.logger.error(f"Failed to compile label expression: {label}")
                self._compiled_expr_cache[label] = None
        
        compiled_expr = self._compiled_expr_cache.get(label)
        if compiled_expr is None:
            self._eval_result_cache[cache_key] = False
            return False

        # Build environment: apNNN is True only if it is in sigma
        env = {ap: (ap in active_aps) for ap in rule_aps}

        try:
            val = bool(eval(compiled_expr, {"__builtins__": {}}, env))
            self._eval_result_cache[cache_key] = val
            return val
        except Exception:
            self.logger.error(f"Failed to evaluate label: {label}")
            self._eval_result_cache[cache_key] = False
            return False

    # ------------------------------------------------------------------ #
    # Shared: DOT Parsing
    # ------------------------------------------------------------------ #
    def _parse_dot(self, rule_id: str, dot_src: str) -> dict[str, Any]:
        """
        Parses a DOT string into a dictionary structure.
        """
        transitions: dict[str, list[tuple[str, str]]] = {}
        ap_list: list[str] = []
        init: str | None = None
        violation: str | None = None
        accepting: set[str] = set()

        text = " ".join(dot_src.split())
        state_pat = r"[A-Za-z0-9_\.]+"

        # Match initial state: 'init -> 1;'
        m_init = re.search(rf"init\s*->\s*({state_pat})\s*;", text)
        if m_init:
            init = m_init.group(1)

        for state_blob in re.findall(
            r"node\s*\[shape\s*=\s*doublecircle\]\s*;\s*([^;]+)\s*;",
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

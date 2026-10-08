"""Shared DFA parsing and AP-mapping utilities for safety checking."""

from __future__ import annotations

import ast
import json
import logging
import re
from functools import lru_cache
from typing import Any
from cais_spade_llm.agents.central_controller.ppr_ap import parse_ap_definition, parse_ap_record, physical_ap_kind


@lru_cache(maxsize=4096)
def _satisfiable_label(label: str) -> bool:
    """Check Boolean DFA guards without treating impossible edges as paths."""
    expression = label.replace("&", " and ").replace("|", " or ")
    expression = expression.replace("~", " not ").replace("!", " not ")
    expression = re.sub(r"\btrue\b", "True", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\bfalse\b", "False", expression, flags=re.IGNORECASE)
    tree = ast.parse(expression.strip(), mode="eval").body
    names = sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.BoolOp, ast.UnaryOp, ast.Name, ast.Constant, ast.And, ast.Or, ast.Not, ast.Load),
        ):
            raise ValueError(f"Unsupported DFA guard: {label!r}")
        if isinstance(node, ast.Constant) and type(node.value) is not bool:
            raise ValueError(f"Unsupported DFA guard constant: {label!r}")
        if isinstance(node, ast.Name) and not re.fullmatch(r"ap\d+", node.id):
            raise ValueError(f"Unsupported DFA proposition: {node.id!r}")

    def evaluate(node: ast.AST, values: dict[str, bool]) -> bool | None:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return values.get(node.id)
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand, values)
            return None if value is None else not value
        results = [evaluate(child, values) for child in node.values]
        if isinstance(node.op, ast.And):
            return False if False in results else None if None in results else True
        return True if True in results else None if None in results else False

    def search(values: dict[str, bool], index: int) -> bool:
        result = evaluate(tree, values)
        if result is not None:
            return result
        name = names[index]
        return search({**values, name: False}, index + 1) or search(
            {**values, name: True}, index + 1
        )

    return search({}, 0)


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
        self.logger = logging.getLogger(__name__)
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
        return token

    @staticmethod
    def _task_product_name(params: dict[str, Any]) -> str:
        product = params.get("part_name") or params.get("part") or params.get("product") or "any"
        return str(product)

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
        token = str(function_name or "").strip()
        return {token} if token else set()

    @classmethod
    def _task_id_tokens(cls, params: dict[str, Any]) -> set[str]:
        tokens = {
            str(params.get("task_id") or "").strip(),
            str(params.get("recovery_outline_id") or "").strip(),
            str(params.get("outline_id") or "").strip(),
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
            str(token).strip() for token in (ap.get("source_task_ids") or []) if str(token).strip()
        }
        if not source_task_ids:
            return True
        return bool(source_task_ids & cls._task_id_tokens(params))

    @staticmethod
    def _parse_ap_descriptor(full: str) -> dict[str, Any]:
        definition = parse_ap_definition(full)
        condition = definition["state" if definition["kind"] == "ap_state" else "event"]
        return {
            "prefix": definition["kind"],
            "product": definition["product"],
            "process": definition["process"],
            "resource": definition["resource"],
            "symbol": condition["symbol"],
            "arguments": condition["arguments"],
        }

    @staticmethod
    def _condition_arguments_match(arguments: dict, values: dict) -> bool:
        missing = []
        for key, expected in arguments.items():
            if expected == "*":
                continue
            if key not in values:
                missing.append(key)
                continue
            actual = values[key]
            if ((type(actual) is bool) != (type(expected) is bool)
                    or actual != expected):
                return False
        if missing:
            raise ValueError("AP condition argument evidence is unavailable: " + ", ".join(missing))
        return True

    @staticmethod
    def _ppr_entities_match(definition: dict, resource: str, params: dict) -> bool:
        product = params.get("part_name") or params.get("part") or params.get("product")
        return (definition["resource"] in {"*", "any", resource}
                and definition["product"] in {"*", "any", product})

    @classmethod
    def _ppr_scope_matches(cls, definition: dict, resource: str, params: dict) -> bool:
        if not cls._ppr_entities_match(definition, resource, params):
            return False
        if definition["process"] in {"*", "any"}:
            return True
        if not params.get("process"):
            raise ValueError("Concrete AP process has no registered execution evidence")
        return definition["process"] == params["process"]

    def _registered_process_params(self, resource_jid: str, function_name: str, params: dict) -> dict:
        """Bind the process through the exact registered owner function."""
        processes = {row["process"] for row in self._tool_rows_for_action(resource_jid, function_name)
                     if isinstance(row.get("process"), str) and row["process"]}
        if len(processes) > 1:
            raise ValueError("Registered function has ambiguous process evidence")
        if not processes:
            return {key: value for key, value in params.items() if key != "process"}
        process = next(iter(processes))
        if params.get("process") not in (None, "", process):
            raise ValueError("Task process disagrees with its registered owner function")
        return {**params, "process": process}

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
        """Match registered function events with their exact PPR condition arguments."""
        labels: list[str] = []
        params = self._registered_process_params(resource_jid, function_name, params)
        resource = getattr(self, "resource_bindings", {}).get(
            str(resource_jid).split("/", 1)[0], self._resource_short_name(resource_jid)
        )
        event_tokens = self._task_event_tokens(function_name, params)
        for rule in self.safety_rules:
            for ap in rule.get("aps", []):
                definition = parse_ap_record(ap)
                if (definition["kind"] != "ap_event"
                        or not self._ppr_entities_match(definition, resource, params)):
                    continue
                if physical_ap_kind(definition):
                    if self._ppr_scope_matches(definition, resource, params):
                        raise ValueError("Physical AP requires grounded observations, not task events")
                    continue
                condition = definition["event"]
                if (condition["symbol"] in event_tokens
                        and self._condition_arguments_match(condition["arguments"], params)
                        and self._ap_requires_source_task_ids(ap, params)
                        and self._ppr_scope_matches(definition, resource, params)):
                    labels.append(ap["label"])
        return labels

    def _map_state_to_aps(self, resource_jid: str, current_state: str, params: dict) -> list[str]:
        """
        Maps a resource's persistent state to matching state AP labels.

        Matching logic mirrors _map_task_to_aps, but:
          - only matches structured ap_state definitions
          - compares the registered state condition against current_state
        """
        surface = self._state_surface_from_runtime(current_state, params)
        return self._map_state_surface_to_aps(resource_jid, surface, params)

    def _map_state_surface_to_aps(
        self, resource_jid: str, surface: dict[str, str], params: dict[str, Any],
    ) -> list[str]:
        """Match modeled state values; spatial conditions require the physical evaluator."""
        labels: list[str] = []
        resource = getattr(self, "resource_bindings", {}).get(
            str(resource_jid).split("/", 1)[0], self._resource_short_name(resource_jid)
        )
        state_tokens = self._state_surface_tokens(surface)
        for rule in self.safety_rules:
            for ap in rule.get("aps", []):
                definition = parse_ap_record(ap)
                if (definition["kind"] != "ap_state"
                        or not self._ppr_entities_match(definition, resource, params)):
                    continue
                condition = definition["state"]
                symbol, arguments = condition["symbol"], condition["arguments"]
                if physical_ap_kind(definition):
                    if self._ppr_scope_matches(definition, resource, params):
                        raise ValueError("Physical AP requires grounded observations, not task state")
                    continue
                matches = symbol in state_tokens if "=" in symbol else symbol == surface.get("resource_state")
                if (matches and self._condition_arguments_match(arguments, {**params, **surface})
                        and self._ap_requires_source_task_ids(ap, params)
                        and self._ppr_scope_matches(definition, resource, params)):
                    labels.append(ap["label"])
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
        res_short = getattr(self, "resource_bindings", {}).get(
            str(resource_jid).split("/", 1)[0], self._resource_short_name(resource_jid))
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
        params = self._registered_process_params(resource_jid, function_name, params)
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

    # ------------------------------------------------------------------ #
    # Shared: DFA Transition Logic (The Math)
    # ------------------------------------------------------------------ #
    def transition_evidence(
        self, rule_id: str, current_state: str, sigma: frozenset[str]
    ) -> dict[str, Any]:
        """Evaluate one DFA step without inventing a stutter transition.

        Args:
            rule_id: Exact identifier of the applicable requirement.
            current_state: Monitor state reached by the accepted history.
            sigma: Complete proposition valuation, including an empty step.

        Returns:
            Matched guards, unique successor, and accepting-state reachability.
        """
        dfa = self.dfas.get(rule_id) or {}
        matches = [
            {"guard": label, "to": destination}
            for label, destination in dfa.get("transitions", {}).get(current_state, [])
            if self._eval_label(label, sigma, dfa.get("ap_symbols", []))
        ]
        destinations = {row["to"] for row in matches}
        next_state = next(iter(destinations)) if len(destinations) == 1 else None
        reachable = next_state in dfa.get("accepting_reachable_states", [])
        reason = (
            "dfa_missing_transition" if not destinations
            else "dfa_ambiguous_transition" if len(destinations) != 1
            else "accepting_state_unreachable" if not reachable
            else ""
        )
        return {
            "rule_id": rule_id,
            "from": current_state,
            "label": sorted(sigma),
            "rule_label": sorted(set(sigma) & set(dfa.get("ap_symbols", []))),
            "matched_transitions": matches,
            "to": next_state,
            "accepting": next_state in dfa.get("accepting_states", []),
            "accepting_reachable": reachable,
            "status": "passed" if not reason else "rejected",
            "reason": reason,
        }

    def _delta(self, rule_id: str, current_state: str, sigma: frozenset[str]) -> str:
        """Return a unique successor or stop validation on a malformed DFA step."""
        evidence = self.transition_evidence(rule_id, current_state, sigma)
        if evidence["to"] is None:
            raise ValueError(
                f"{evidence['reason']}: rule {rule_id!r}, state {current_state!r}, "
                f"label {sorted(sigma)!r}"
            )
        return evidence["to"]

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
            expr = (
                label.replace("&", " and ")
                .replace("|", " or ")
                .replace("~", " not ")
                .replace("!", " not ")
            )
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
            r"node\s*\[shape\s*=\s*doublecircle\]\s*;\s*(.*?)"
            r"(?=node\s*\[|\binit\b|[A-Za-z0-9_.]+\s*->|\}|$)",
            text,
            flags=re.IGNORECASE,
        ):
            for state in re.findall(state_pat, state_blob):
                accepting.add(state)
        accepting.update(re.findall(
            rf"\b(?!node\b)({state_pat})\s*\[\s*shape\s*=\s*doublecircle\s*\]", text
        ))

        # Match transitions: '1 -> 2 [label="..."];'
        pattern = re.compile(rf"({state_pat})\s*->\s*({state_pat})\s*\[label=\"(.*?)\"\];")

        for src, dst, label in pattern.findall(text):
            transitions.setdefault(src, []).append((label, dst))

            if src == dst and label.strip().lower() == "true" and src not in accepting:
                violation = src

            for ap in re.findall(r"(ap\d+)", label):
                if ap not in ap_list:
                    ap_list.append(ap)

        if init is None and transitions:
            init = next(iter(transitions.keys()))

        predecessors: dict[str, set[str]] = {}
        for src, edges in transitions.items():
            for label, dst in edges:
                if _satisfiable_label(label):
                    predecessors.setdefault(dst, set()).add(src)
        accepting_reachable = set(accepting)
        pending = list(accepting)
        while pending:
            for src in predecessors.get(pending.pop(), set()) - accepting_reachable:
                accepting_reachable.add(src)
                pending.append(src)

        return {
            "initial": init,
            "transitions": transitions,
            "violation_state": violation,
            "accepting_states": sorted(accepting),
            "accepting_reachable_states": sorted(accepting_reachable),
            "ap_symbols": ap_list,
        }

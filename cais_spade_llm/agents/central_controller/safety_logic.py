"""Safety rule parsing and LTLf/DFA generation utilities."""

from __future__ import annotations

import contextlib
import io
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re
import shutil
from urllib.parse import quote
try:
    from ltlf2dfa.parser.ltlf import LTLfParser
except Exception:  # pragma: no cover - dependency may be absent in lightweight test envs
    LTLfParser = None  # type: ignore[assignment]

try:
    from graphviz import Source
except Exception:  # pragma: no cover - dependency may be absent in lightweight test envs
    Source = None  # type: ignore[assignment]

from cais_spade_llm.prompts import (
    build_safety_parse_prompt,
    build_safety_logic_prompt,
    build_safety_interpretation_prompt,
)

class SafetyLogic:
    """
    SafetyLogic:

    1. Load NL safety requirements from file.
    2. NL -> structured safety rule nodes (via LLM).
    3. Structured rules -> AP strings + LTLf (via LLM).
    4. Attach AP labels + full AP definitions and LTLf to each rule node.
    5. (Optional) Combine all rule formulas into one global safety spec.
    """

    def __init__(self, controller_agent, safety_file: str | Path) -> None:
        """Bind to a controller agent and set up safety rule storage paths."""
        self.controller_agent = controller_agent
        self.logger = controller_agent.logger
        self.safety_file = Path(safety_file)

        # Where structured + logic JSON will be stored
        self.structured_safety_path: Path = self.safety_file.with_suffix(".json")

        # Parsed structured rules (NL -> rules)
        self.rules: List[Dict[str, Any]] = []

        # Raw logic from LLM: rule_id -> {"aps": [full___str...], "ltlf": "..."}
        self.logic_raw: Dict[str, Dict[str, Any]] = {}

        # Optional combined safety spec: {"aps": {label: full}, "formula": "φ_safety"}
        self.global_safety_spec: Dict[str, Any] = {}
        self.preview_interpretation_summary: str = ""

        # store one DFA (DOT string) per rule
        self.rule_dfas: Dict[str, str] = {}

    @staticmethod
    def _to_dfa_quiet(ltlf_formula) -> str:
        """
        Convert LTLf to DFA while swallowing noisy parser prints from ltlf2dfa.
        Some versions print regex parse warnings to stdout/stderr even on success.
        """
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            return ltlf_formula.to_dfa()

    @staticmethod
    def _fallback_rule_interpretation(rule: dict[str, Any]) -> str:
        ltlf = str(rule.get("ltlf", "") or "").strip()
        aps = rule.get("aps", []) if isinstance(rule.get("aps"), list) else []

        ap_map: dict[str, str] = {}
        for ap in aps:
            if not isinstance(ap, dict):
                continue
            label = str(ap.get("label", "")).strip()
            full = str(ap.get("full", "")).strip()
            if label and full:
                ap_map[label] = full

        normalized = re.sub(r"\s+", "", ltlf)
        order_match = (
            re.fullmatch(r"\(!?(ap\d+)\)U(ap\d+)", normalized)
            or re.fullmatch(r"\(\(!?(ap\d+)\)U(ap\d+)\)", normalized)
        )
        response_match = re.fullmatch(r"G\((ap\d+)->F(ap\d+)\)", normalized)

        if ltlf.startswith("G !(") or ltlf.startswith("G!("):
            explained = [full for _, full in sorted(ap_map.items())]
            if explained:
                joined = "; ".join(explained)
                return (
                    "This generated rule treats the following grounded events as mutually exclusive: "
                    f"{joined}. Those events are not allowed to overlap in time."
                )
            return "This generated rule is a global mutual-exclusion constraint over the generated AP events."

        if response_match:
            trigger = ap_map.get(response_match.group(1), response_match.group(1))
            response = ap_map.get(response_match.group(2), response_match.group(2))
            return f"Whenever {trigger} happens, {response} must eventually happen afterwards."

        if order_match:
            earlier = ap_map.get(order_match.group(1), order_match.group(1))
            later = ap_map.get(order_match.group(2), order_match.group(2))
            return f"The generated ordering requires {earlier} to occur before {later}."

        if " U " in ltlf:
            return (
                "This generated rule uses an until-condition: the left-hand condition must hold "
                "until the right-hand condition becomes true."
            )

        if ltlf.startswith("G(") or ltlf.startswith("G "):
            return "This generated rule is a global constraint that must hold throughout execution."

        if ltlf:
            return (
                "This generated rule constrains execution according to the grounded AP events in the "
                f"formula {ltlf}."
            )
        return "No generated interpretation is available for this rule."

    @classmethod
    def _fallback_preview_interpretation_summary(cls, rules: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            rid = str(rule.get("id", "")).strip() or "rule"
            interp = str(rule.get("generated_interpretation", "")).strip() or cls._fallback_rule_interpretation(rule)
            lines.append(f"- {rid}: {interp}")
        return "\n".join(lines) if lines else "No generated rule interpretation available."

    @staticmethod
    def _is_placeholder_dfa_dot(dot_text: str) -> bool:
        raw = str(dot_text or "").strip()
        if not raw:
            return True
        if re.search(r"\[label=\".+?\"\]", raw):
            return False
        return "init -> 1;" in raw and "0.0;" in raw

    @staticmethod
    def _placeholder_dfa_message() -> str:
        if shutil.which("mona") is None:
            return "MONA is not installed, so ltlf2dfa returned placeholder output"
        return "ltlf2dfa returned placeholder DFA output"

    @staticmethod
    def _normalize_resource_token(value: Any) -> str:
        token = str(value or "").strip()
        if not token:
            return ""
        return token.split("@")[0].lower()

    @staticmethod
    def _dedupe_keep_order(items: List[str]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    @staticmethod
    def _debug_json(value: Any) -> str:
        def _default(obj: Any) -> Any:
            if isinstance(obj, set):
                return sorted(obj)
            return repr(obj)

        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_default)
        except TypeError:
            return repr(value)

    @staticmethod
    def _normalize_context_scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).strip()

    @classmethod
    def _normalize_context_object(cls, value: Any) -> Optional[dict[str, str]]:
        if not isinstance(value, dict):
            return None

        normalized: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key or "").strip()
            if not key:
                continue
            if raw_value is None or isinstance(raw_value, (dict, list, tuple, set)):
                continue
            value_text = cls._normalize_context_scalar(raw_value)
            if not value_text:
                continue
            normalized[key] = value_text

        return normalized or None

    @classmethod
    def _serialize_context_object(cls, value: Any) -> str:
        normalized = cls._normalize_context_object(value)
        if not normalized:
            return "any"

        items = []
        for key, raw_val in sorted(normalized.items()):
            key_text = quote(str(key), safe="-_.~")
            val_text = quote(str(raw_val), safe="-_.~")
            items.append(f"{key_text}={val_text}")
        return "&".join(items) if items else "any"

    @staticmethod
    def _ap_segments(ap: str) -> Optional[dict[str, str]]:
        parts = str(ap or "").split("/", 5)
        if len(parts) != 6:
            return None
        prefix, process, product, resource, event, context = parts
        return {
            "prefix": str(prefix).strip(),
            "process": str(process).strip().lower(),
            "product": str(product).strip().lower(),
            "resource": str(resource).strip().lower(),
            "event": str(event).strip(),
            "context": str(context).strip(),
        }

    @staticmethod
    def _normalized_rule_text(
        rule: dict[str, Any], refinement_feedback: str = ""
    ) -> str:
        pieces = [
            str(rule.get("constraint_type", "") or ""),
            str(rule.get("raw_text", "") or ""),
        ]
        return " ".join(piece.strip().lower() for piece in pieces if piece).strip()

    @staticmethod
    def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
        return any(cue in text for cue in cues)

    @staticmethod
    def _tokenize_for_overlap(value: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", str(value or "").lower())

    @classmethod
    def _term_overlap_score(cls, term: str, text: str) -> int:
        term_tokens = cls._tokenize_for_overlap(str(term).replace("_", " "))
        text_tokens = cls._tokenize_for_overlap(text)
        if not term_tokens or not text_tokens:
            return 0

        score = 0
        for needle in term_tokens:
            for token in text_tokens:
                if token == needle or token.startswith(needle) or needle.startswith(token):
                    score += 1
                    break
        return score

    @classmethod
    def _pick_scored_ap(
        cls,
        parsed_aps: list[tuple[str, dict[str, str]]],
        text: str,
        *,
        exclude: set[str] | None = None,
    ) -> Optional[str]:
        blocked = exclude or set()
        best_ap: Optional[str] = None
        best_score = -1

        for ap, segments in parsed_aps:
            if ap in blocked:
                continue
            score = 0
            score += 4 * cls._term_overlap_score(segments.get("event", ""), text)
            score += 2 * cls._term_overlap_score(segments.get("product", ""), text)
            score += cls._term_overlap_score(segments.get("resource", ""), text)
            score += cls._term_overlap_score(segments.get("context", "").replace("&", " "), text)
            if score > best_score:
                best_score = score
                best_ap = ap

        return best_ap if best_score > 0 else None

    @classmethod
    def _infer_ltlf_family(
        cls,
        rule: dict[str, Any],
        aps: list[str],
        refinement_feedback: str = "",
    ) -> Optional[str]:
        if not aps:
            return None

        text = cls._normalized_rule_text(rule, refinement_feedback)
        if not text:
            return None

        precedence_cues = (
            "before",
            "precedence",
            "precedes",
            "ordering",
            "ordered",
            "prior to",
        )
        precedence_gate_cues = (
            "must not",
            "should not",
            "cannot",
            "can't",
            "not begin",
            "not start",
            "only after",
        )
        mutex_cues = (
            "same time",
            "simultaneous",
            "simultaneously",
            "concurrent",
            "mutual exclusion",
            "mutex",
            "overlap",
            "together",
        )
        response_cues = (
            " after ",
            "_after_",
            "followed by",
            "follow_up",
            "follow-up",
            "response",
            "post_",
            "post-",
        )
        absence_cues = (
            "never",
            "forbidden",
            "must not",
            "should not",
            "cannot",
            "can't",
        )
        until_cues = (" until ", "_until_", "until")

        if len(aps) >= 2 and cls._contains_any(text, precedence_cues):
            return "precedence"
        if (
            len(aps) >= 2
            and "until" in text
            and cls._contains_any(text, precedence_gate_cues)
        ):
            return "precedence"
        if len(aps) >= 2 and cls._contains_any(text, mutex_cues):
            return "mutex"
        if len(aps) >= 2 and cls._contains_any(text, response_cues):
            return "response"
        if len(aps) >= 2 and cls._contains_any(text, until_cues):
            return "until"
        if cls._contains_any(text, absence_cues):
            return "absence"
        return None

    @classmethod
    def _pick_precedence_pair(
        cls,
        rule: dict[str, Any],
        aps: list[str],
        refinement_feedback: str = "",
    ) -> Optional[tuple[str, str]]:
        parsed_aps = [
            (ap, segments)
            for ap in aps
            if (segments := cls._ap_segments(ap)) is not None
        ]
        if len(parsed_aps) < 2:
            return None

        rule_event = str(rule.get("event", "") or "").strip()
        products = [
            str(product).strip().lower()
            for product in (rule.get("product") or [])
            if str(product or "").strip()
        ]

        def select_by_product(product_name: str) -> Optional[str]:
            candidates = [
                ap
                for ap, segments in parsed_aps
                if segments.get("product") == product_name
            ]
            if rule_event:
                event_candidates = [
                    ap
                    for ap in candidates
                    if (cls._ap_segments(ap) or {}).get("event") == rule_event
                ]
                if event_candidates:
                    return event_candidates[0]
            return candidates[0] if candidates else None

        if len(products) >= 2:
            earlier = select_by_product(products[0])
            later = select_by_product(products[1])
            if earlier and later and earlier != later:
                return earlier, later

        text = cls._normalized_rule_text(rule, refinement_feedback)
        if "before" in text:
            before_text, after_text = re.split(r"\bbefore\b", text, maxsplit=1)
            later = cls._pick_scored_ap(parsed_aps, after_text)
            earlier = cls._pick_scored_ap(
                parsed_aps,
                before_text,
                exclude={later} if later else None,
            )
            if earlier and later and earlier != later:
                return earlier, later

        return parsed_aps[0][0], parsed_aps[1][0]

    @classmethod
    def _pair_response_aps(
        cls,
        rule: dict[str, Any],
        aps: list[str],
        refinement_feedback: str = "",
    ) -> list[tuple[str, str]]:
        parsed_aps = [
            (ap, segments)
            for ap in aps
            if (segments := cls._ap_segments(ap)) is not None
        ]
        if len(parsed_aps) < 2:
            return []

        text = cls._normalized_rule_text(rule, refinement_feedback)
        response_event = str(rule.get("event", "") or "").strip()
        if not response_event:
            before_text = text
            after_text = ""
            if "after" in text:
                before_text, after_text = re.split(r"\bafter\b", text, maxsplit=1)
            event_counts: dict[str, int] = {}
            for _, segments in parsed_aps:
                event_name = segments.get("event", "")
                if event_name:
                    event_counts[event_name] = event_counts.get(event_name, 0) + 1

            best_event = ""
            best_score = 0
            for event_name in event_counts:
                score = cls._term_overlap_score(event_name, before_text)
                score -= cls._term_overlap_score(event_name, after_text)
                if score > best_score:
                    best_score = score
                    best_event = event_name
            response_event = best_event

        if not response_event:
            return []

        response_aps = [
            (ap, segments)
            for ap, segments in parsed_aps
            if segments.get("event") == response_event
        ]
        trigger_aps = [
            (ap, segments)
            for ap, segments in parsed_aps
            if segments.get("event") != response_event
        ]
        if not response_aps or not trigger_aps:
            return []

        pairs: list[tuple[str, str]] = []
        used_triggers: set[str] = set()

        for response_ap, response_segments in response_aps:
            matched_trigger: Optional[str] = None
            for trigger_ap, trigger_segments in trigger_aps:
                if trigger_ap in used_triggers:
                    continue
                if trigger_segments.get("resource") == response_segments.get("resource"):
                    matched_trigger = trigger_ap
                    break
            if not matched_trigger:
                for trigger_ap, trigger_segments in trigger_aps:
                    if trigger_ap in used_triggers:
                        continue
                    if (
                        response_segments.get("product") != "any"
                        and trigger_segments.get("product") == response_segments.get("product")
                    ):
                        matched_trigger = trigger_ap
                        break
            if not matched_trigger:
                for trigger_ap, _ in trigger_aps:
                    if trigger_ap not in used_triggers:
                        matched_trigger = trigger_ap
                        break

            if not matched_trigger:
                continue

            used_triggers.add(matched_trigger)
            pairs.append((matched_trigger, response_ap))

        return pairs

    @classmethod
    def _pick_until_pair(
        cls,
        rule: dict[str, Any],
        aps: list[str],
        refinement_feedback: str = "",
    ) -> Optional[tuple[str, str]]:
        parsed_aps = [
            (ap, segments)
            for ap in aps
            if (segments := cls._ap_segments(ap)) is not None
        ]
        if len(parsed_aps) < 2:
            return None

        text = cls._normalized_rule_text(rule, refinement_feedback)
        if "until" in text:
            left_text, right_text = re.split(r"\buntil\b", text, maxsplit=1)
            right = cls._pick_scored_ap(parsed_aps, right_text)
            left = cls._pick_scored_ap(
                parsed_aps,
                left_text,
                exclude={right} if right else None,
            )
            if left and right and left != right:
                return left, right

        return parsed_aps[0][0], parsed_aps[1][0]

    @classmethod
    def _compile_ltlf_for_rule(
        cls,
        rule: dict[str, Any],
        aps: list[str],
        refinement_feedback: str = "",
    ) -> Optional[str]:
        family = cls._infer_ltlf_family(rule, aps, refinement_feedback)
        if family == "precedence":
            pair = cls._pick_precedence_pair(rule, aps, refinement_feedback)
            if not pair:
                return None
            earlier, later = pair
            return f"((!{later}) U {earlier})"

        if family == "mutex":
            if len(aps) < 2:
                return None
            if len(aps) == 2:
                return f"G !({aps[0]} & {aps[1]})"
            pair_terms: list[str] = []
            for idx, left in enumerate(aps):
                for right in aps[idx + 1 :]:
                    pair_terms.append(f"({left} & {right})")
            if not pair_terms:
                return None
            return f"G !({' | '.join(pair_terms)})"

        if family == "response":
            pairs = cls._pair_response_aps(rule, aps, refinement_feedback)
            if not pairs:
                return None
            if len(pairs) == 1:
                trigger, response = pairs[0]
                return f"G ({trigger} -> F {response})"
            pair_terms = [
                f"({trigger} -> F {response})" for trigger, response in pairs
            ]
            return f"G ({' & '.join(pair_terms)})"

        if family == "absence":
            if not aps:
                return None
            body = aps[0] if len(aps) == 1 else " | ".join(aps)
            return f"G !({body})"

        if family == "until":
            pair = cls._pick_until_pair(rule, aps, refinement_feedback)
            if not pair:
                return None
            left, right = pair
            return f"({left} U {right})"

        return None

    def _tool_grounding(self) -> tuple[set[str], dict[str, str], set[str], set[str]]:
        """
        Build capability grounding from tools_catalog.
        Returns:
          - allowed function names
          - function -> process map
          - allowed resource identifiers (owner-agent localparts)
          - allowed process names
        """
        tools_catalog = getattr(self.controller_agent, "tools_catalog", []) or []
        allowed_functions: set[str] = set()
        function_process: dict[str, str] = {}
        allowed_resources: set[str] = set()

        for row in tools_catalog:
            if not isinstance(row, dict):
                continue
            fn = str(row.get("function", "")).strip()
            if fn:
                allowed_functions.add(fn)
                proc = str(row.get("process", "")).strip().lower()
                if proc and fn not in function_process:
                    function_process[fn] = proc

            owner = self._normalize_resource_token(row.get("function_owner_agent"))
            if owner:
                allowed_resources.add(owner)

        allowed_processes = set(function_process.values())
        return allowed_functions, function_process, allowed_resources, allowed_processes

    def _allowed_resource_types(self) -> set[str]:
        allowed: set[str] = set()
        for row in self._tool_rows():
            token = self._normalize_resource_type_token(row.get("resource_type"))
            if token:
                allowed.add(token)
        return allowed

    def _tool_rows(self) -> list[dict[str, Any]]:
        return [
            row for row in (getattr(self.controller_agent, "tools_catalog", []) or [])
            if isinstance(row, dict)
        ]

    def _allowed_state_names(self) -> set[str]:
        states: set[str] = set()
        for row in self._tool_rows():
            for key in ("in_state", "out_state"):
                token = str(row.get(key, "") or "").strip()
                if token and token.lower() != "any":
                    states.add(token)
        return states

    def _tool_rows_for_resource(self, resource: str) -> list[dict[str, Any]]:
        token = self._normalize_resource_token(resource)
        rows: list[dict[str, Any]] = []
        for row in self._tool_rows():
            owner = self._normalize_resource_token(row.get("function_owner_agent"))
            if token and owner and owner != token:
                continue
            rows.append(row)
        return rows

    def _tool_row_for_action(self, resource: str, function_name: str) -> Optional[dict[str, Any]]:
        token = self._normalize_resource_token(resource)
        fn = str(function_name or "").strip()
        fallback: Optional[dict[str, Any]] = None
        for row in self._tool_rows():
            if str(row.get("function", "")).strip() != fn:
                continue
            owner = self._normalize_resource_token(row.get("function_owner_agent"))
            if owner and token and owner == token:
                return row
            fallback = fallback or row
        return fallback

    @staticmethod
    def _normalize_state_name(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _normalize_string_list(value: Any) -> list[str]:
        if isinstance(value, (list, tuple, set)):
            items = value
        elif value is None:
            items = []
        else:
            items = [value]
        out: list[str] = []
        for item in items:
            token = str(item or "").strip()
            if token:
                out.append(token)
        return out

    @staticmethod
    def _collect_resource_var_names(node: Any) -> set[str]:
        names: set[str] = set()
        if isinstance(node, dict):
            token = str(node.get("resource_var", "") or "").strip()
            if token:
                names.add(token)
            for value in node.values():
                names |= SafetyLogic._collect_resource_var_names(value)
            return names
        if isinstance(node, list):
            for value in node:
                names |= SafetyLogic._collect_resource_var_names(value)
        return names

    @staticmethod
    def _row_required_context_keys(row: dict[str, Any]) -> tuple[str, ...]:
        keys = [
            str(key).strip()
            for key in (row.get("required_context_keys") or [])
            if str(key).strip()
        ]
        return tuple(SafetyLogic._dedupe_keep_order(keys))

    @staticmethod
    def _row_location_type(row: dict[str, Any]) -> str:
        mapping = row.get("context_mapping") or {}
        if not isinstance(mapping, dict):
            return ""
        return str(mapping.get("location_type") or "").strip().lower()

    @staticmethod
    def _row_persistent_state_name(row: dict[str, Any]) -> str:
        state = str(row.get("out_state", "") or "").strip()
        if not state or state.lower() == "any":
            return ""
        return state

    @classmethod
    def _row_family_signature(cls, row: dict[str, Any]) -> tuple[str, ...]:
        roles = cls._row_required_context_keys(row)
        if roles:
            return roles
        location_type = cls._row_location_type(row)
        if location_type:
            return (f"location_type={location_type}",)
        return ("<none>",)

    @classmethod
    def _row_family_payload(cls, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "function": str(row.get("function", "")).strip(),
            "required_context_keys": list(cls._row_required_context_keys(row)),
            "location_type": cls._row_location_type(row),
            "in_state": cls._normalize_state_name(row.get("in_state")),
            "out_state": cls._normalize_state_name(row.get("out_state")),
        }

    def _resolve_rule_resources(self, rule: dict[str, Any]) -> list[str]:
        _, _, allowed_resources, _ = self._tool_grounding()
        allowed_resource_types = {
            self._normalize_resource_type_token(resource_type)
            for resource_type in (rule.get("resource_types") or [])
            if self._normalize_resource_type_token(resource_type)
        }
        resources = [
            self._normalize_resource_token(resource)
            for resource in (rule.get("resources") or [])
            if self._normalize_resource_token(resource)
        ]
        concrete = [resource for resource in resources if resource != "any"]
        if allowed_resource_types:
            typed_resources = {
                self._normalize_resource_token(row.get("function_owner_agent"))
                for row in self._tool_rows()
                if self._normalize_resource_token(row.get("function_owner_agent"))
                and self._normalize_resource_type_token(row.get("resource_type")) in allowed_resource_types
            }
        else:
            typed_resources = set(allowed_resources)
        if concrete:
            return [resource for resource in self._dedupe_keep_order(concrete) if resource in typed_resources]
        return sorted(typed_resources)

    @staticmethod
    def _ast_contains_resource_var(node: Any) -> bool:
        if isinstance(node, dict):
            if "resource_var" in node and str(node.get("resource_var", "")).strip():
                return True
            return any(SafetyLogic._ast_contains_resource_var(value) for value in node.values())
        if isinstance(node, list):
            return any(SafetyLogic._ast_contains_resource_var(value) for value in node)
        return False

    @classmethod
    def _bind_resource_var(cls, node: Any, resource: str) -> Any:
        if isinstance(node, list):
            return [cls._bind_resource_var(value, resource) for value in node]
        if not isinstance(node, dict):
            return node
        bound: dict[str, Any] = {}
        for key, value in node.items():
            if key == "resource_var":
                bound["resource"] = resource
                continue
            bound[key] = cls._bind_resource_var(value, resource)
        return bound

    def _normalize_atom_product(self, rule: dict[str, Any], node: dict[str, Any]) -> str:
        explicit = str(node.get("product", "") or "").strip().lower()
        if explicit:
            return explicit
        products = [
            str(product).strip().lower()
            for product in (rule.get("product") or [])
            if str(product or "").strip()
        ]
        return products[0] if len(products) == 1 else "any"

    def _normalize_atom_context(self, rule: dict[str, Any], node: dict[str, Any]) -> dict[str, str] | None:
        explicit = self._normalize_context_object(node.get("context"))
        if explicit:
            return explicit
        return self._normalize_context_object(rule.get("context"))

    def _normalize_atom_resource(self, rule: dict[str, Any], node: dict[str, Any]) -> str:
        explicit = self._normalize_resource_token(node.get("resource"))
        if explicit:
            return explicit
        resources = self._resolve_rule_resources(rule)
        return resources[0] if len(resources) == 1 else "any"

    @staticmethod
    def _normalize_process_token(value: Any) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _normalize_resource_type_token(value: Any) -> str:
        return str(value or "").strip().lower()

    def _selector_match_spec(
        self,
        rule: dict[str, Any],
        node: dict[str, Any],
    ) -> dict[str, Any]:
        match = node.get("match") or {}
        if not isinstance(match, dict):
            match = {}

        explicit_states = {
            self._normalize_state_name(state)
            for state in self._normalize_string_list(match.get("states"))
            if self._normalize_state_name(state)
        }
        functions = {
            str(fn).strip()
            for fn in self._normalize_string_list(match.get("functions"))
            if str(fn).strip()
        }

        process_token = self._normalize_process_token(
            match.get("process") or node.get("process") or rule.get("process")
        )
        resource_type_token = self._normalize_resource_type_token(
            match.get("resource_type") or node.get("resource_type")
        )

        return {
            "context": self._normalize_context_object(match.get("context")),
            "states": explicit_states,
            "functions": functions,
            "process": process_token,
            "resource_type": resource_type_token,
        }

    def _match_selector_row(
        self,
        row: dict[str, Any],
        selector_context: dict[str, str] | None,
        *,
        selector_process: str = "",
        selector_resource_type: str = "",
        selector_functions: Optional[set[str]] = None,
        selector_states: Optional[set[str]] = None,
    ) -> bool:
        row_process = self._normalize_process_token(row.get("process"))
        row_resource_type = self._normalize_resource_type_token(row.get("resource_type"))
        row_function = str(row.get("function", "")).strip()
        row_out_state = self._row_persistent_state_name(row)

        if selector_process and row_process != selector_process:
            return False
        if selector_resource_type and row_resource_type != selector_resource_type:
            return False
        if selector_functions and row_function not in selector_functions:
            return False
        if selector_states and row_out_state not in selector_states:
            return False

        if not selector_context:
            return True
        required_context_keys = list(self._row_required_context_keys(row))
        if not required_context_keys:
            return False
        required = set(required_context_keys)
        return set(selector_context.keys()).issubset(required)

    def _resolved_selector_match_spec_and_rows(
        self,
        rule: dict[str, Any],
        selector_node: dict[str, Any],
        *,
        resource: str = "",
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        match_spec = self._selector_match_spec(rule, selector_node)
        rows = self._tool_rows_for_resource(resource) if resource else self._tool_rows()

        base_rows = [
            row
            for row in rows
            if self._match_selector_row(
                row,
                None,
                selector_process=match_spec["process"],
                selector_resource_type=match_spec["resource_type"],
                selector_functions=match_spec["functions"] or None,
            )
        ]
        if not base_rows:
            raise RuntimeError(
                f"selector for rule {rule.get('id')} did not match any tool rows before context grounding "
                f"(match={self._debug_json(match_spec)}, selector={self._debug_json(selector_node)})"
            )

        explicit_states = match_spec["states"]
        candidate_rows = base_rows
        if explicit_states:
            candidate_rows = [
                row
                for row in base_rows
                if self._row_persistent_state_name(row) in explicit_states
            ]
            if not candidate_rows:
                available_states = sorted(
                    {
                        state
                        for row in base_rows
                        if (state := self._row_persistent_state_name(row))
                    }
                )
                raise RuntimeError(
                    f"selector for rule {rule.get('id')} requested states {sorted(explicit_states)} "
                    f"but matching tool rows expose out_states {available_states or ['<none>']} "
                    f"before context grounding"
                )

        selector_context = match_spec["context"]
        if selector_context:
            supported_keys = self._dedupe_keep_order(
                [
                    key
                    for row in candidate_rows
                    for key in self._row_required_context_keys(row)
                ]
            )
            selector_keys = list(selector_context.keys())
            unsupported_keys = [key for key in selector_keys if key not in supported_keys]
            if unsupported_keys:
                if (
                    len(selector_keys) == 1
                    and len(unsupported_keys) == 1
                    and len(supported_keys) == 1
                ):
                    selector_context = {
                        supported_keys[0]: selector_context[selector_keys[0]]
                    }
                else:
                    raise RuntimeError(
                        f"selector for rule {rule.get('id')} uses unresolved context keys {unsupported_keys}; "
                        f"candidate canonical context roles are {supported_keys or ['<none>']} "
                        f"(match={self._debug_json(match_spec)}, selector={self._debug_json(selector_node)})"
                    )

        filtered_rows = [
            row
            for row in candidate_rows
            if self._match_selector_row(
                row,
                selector_context,
                selector_process=match_spec["process"],
                selector_resource_type=match_spec["resource_type"],
                selector_functions=match_spec["functions"] or None,
                selector_states=explicit_states or None,
            )
        ]
        if not filtered_rows:
            raise RuntimeError(
                f"selector for rule {rule.get('id')} could not ground context "
                f"{self._debug_json(selector_context)} to any tool rows"
            )

        family_signatures = {self._row_family_signature(row) for row in filtered_rows}
        if len(family_signatures) > 1:
            raise RuntimeError(
                f"selector for rule {rule.get('id')} mixes incompatible tool families "
                f"(families={self._debug_json(sorted(family_signatures))}, "
                f"rows={self._debug_json([self._row_family_payload(row) for row in filtered_rows])})"
            )

        resolved_match = dict(match_spec)
        resolved_match["context"] = selector_context
        return resolved_match, filtered_rows

    def _selector_context_token(
        self,
        rule: dict[str, Any],
        selector_node: dict[str, Any],
        *,
        resource: str = "",
    ) -> str:
        resolved_match, _ = self._resolved_selector_match_spec_and_rows(
            rule,
            selector_node,
            resource=resource,
        )
        selector_context = self._normalize_context_object(resolved_match.get("context"))
        if selector_context:
            return self._serialize_context_object(selector_context)
        rule_context = self._normalize_context_object(rule.get("context"))
        return self._serialize_context_object(rule_context)

    def _matching_rows_for_selector(
        self,
        rule: dict[str, Any],
        selector_node: dict[str, Any],
        *,
        resource: str = "",
    ) -> list[dict[str, Any]]:
        _, rows = self._resolved_selector_match_spec_and_rows(
            rule,
            selector_node,
            resource=resource,
        )
        return rows

    def _candidate_resources_from_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> list[str]:
        owners: list[str] = []
        for row in rows:
            owner = self._normalize_resource_token(row.get("function_owner_agent"))
            if owner:
                owners.append(owner)
        return self._dedupe_keep_order(owners)

    def _candidate_resources_for_ast_node(
        self,
        rule: dict[str, Any],
        node: Any,
    ) -> list[set[str]]:
        candidate_sets: list[set[str]] = []

        if isinstance(node, list):
            for item in node:
                candidate_sets.extend(self._candidate_resources_for_ast_node(rule, item))
            return candidate_sets

        if not isinstance(node, dict):
            return candidate_sets

        node_type = str(node.get("type", "") or "").strip()
        has_resource_var = bool(str(node.get("resource_var", "") or "").strip())

        if has_resource_var and node_type == "ap_selector":
            rows = self._matching_rows_for_selector(rule, node)
            owners = self._candidate_resources_from_rows(rows)
            if owners:
                candidate_sets.append(set(owners))
        elif has_resource_var and node_type == "ap_event_atom":
            function_name = str(node.get("function", "")).strip()
            node_process = self._normalize_process_token(node.get("process") or rule.get("process"))
            node_resource_type = self._normalize_resource_type_token(node.get("resource_type"))
            node_context = self._normalize_context_object(node.get("context"))
            rows = []
            for row in self._tool_rows():
                if function_name and str(row.get("function", "")).strip() != function_name:
                    continue
                if not self._match_selector_row(
                    row,
                    node_context,
                    selector_process=node_process,
                    selector_resource_type=node_resource_type,
                ):
                    continue
                rows.append(row)
            owners = self._candidate_resources_from_rows(rows)
            if owners:
                candidate_sets.append(set(owners))
        elif has_resource_var and node_type == "ap_state_atom":
            state_name = self._normalize_state_name(node.get("state"))
            node_process = self._normalize_process_token(node.get("process") or rule.get("process"))
            node_resource_type = self._normalize_resource_type_token(node.get("resource_type"))
            node_context = self._normalize_context_object(node.get("context"))
            rows = []
            for row in self._tool_rows():
                row_states = {
                    self._normalize_state_name(row.get("in_state")),
                    self._normalize_state_name(row.get("out_state")),
                }
                row_states.discard("")
                row_states.discard("any")
                if state_name and state_name not in row_states:
                    continue
                if not self._match_selector_row(
                    row,
                    node_context,
                    selector_process=node_process,
                    selector_resource_type=node_resource_type,
                ):
                    continue
                rows.append(row)
            owners = self._candidate_resources_from_rows(rows)
            if owners:
                candidate_sets.append(set(owners))

        for value in node.values():
            candidate_sets.extend(self._candidate_resources_for_ast_node(rule, value))
        return candidate_sets

    def _resolve_ast_resource_bindings(
        self,
        rule: dict[str, Any],
        ast: dict[str, Any],
    ) -> list[str]:
        base_resources = self._resolve_rule_resources(rule)
        base_set = set(base_resources)
        candidate_sets = self._candidate_resources_for_ast_node(rule, ast)
        if not candidate_sets:
            return base_resources

        allowed = set.intersection(*candidate_sets) if candidate_sets else set()
        if base_set:
            allowed &= base_set
        if not allowed:
            return []
        return [resource for resource in base_resources if resource in allowed]

    def _expand_selector(self, rule: dict[str, Any], selector_node: dict[str, Any]) -> list[str]:
        resource = self._normalize_atom_resource(rule, selector_node)
        match_spec, matching_rows = self._resolved_selector_match_spec_and_rows(
            rule,
            selector_node,
            resource=resource,
        )
        explicit_states = match_spec["states"]
        include_entry_events = bool(selector_node.get("include_entry_events", True))
        include_state_aps = bool(selector_node.get("include_state_aps", True))
        context_token = self._selector_context_token(rule, selector_node, resource=resource)
        product_token = self._normalize_atom_product(rule, selector_node)

        persistent_states: list[str] = []
        state_process: dict[str, str] = {}
        for row in matching_rows:
            out_state = self._normalize_state_name(row.get("out_state"))
            if not out_state or out_state.lower() == "any":
                continue
            if explicit_states and out_state not in explicit_states:
                continue
            if out_state not in state_process:
                state_process[out_state] = str(
                    row.get("process") or rule.get("process") or "any"
                ).strip().lower() or "any"
            persistent_states.append(out_state)

        persistent_state_set = set(persistent_states)
        expanded: list[str] = []

        if include_entry_events and persistent_state_set:
            for row in matching_rows:
                out_state = self._normalize_state_name(row.get("out_state"))
                in_state = self._normalize_state_name(row.get("in_state"))
                if out_state not in persistent_state_set:
                    continue
                if in_state in persistent_state_set:
                    continue
                process_token = str(
                    row.get("process") or rule.get("process") or "any"
                ).strip().lower() or "any"
                function_name = str(row.get("function", "")).strip()
                if not function_name:
                    continue
                expanded.append(
                    "/".join(
                        [
                            "ap_event",
                            process_token,
                            product_token,
                            resource,
                            function_name,
                            context_token,
                        ]
                    )
                )

        if include_state_aps and persistent_state_set:
            for state_name in self._dedupe_keep_order(persistent_states):
                expanded.append(
                    "/".join(
                        [
                            "ap_state",
                            state_process.get(state_name, str(rule.get("process") or "any").strip().lower() or "any"),
                            product_token,
                            resource,
                            state_name,
                            context_token,
                        ]
                    )
                )

        return self._dedupe_keep_order(expanded)

    def _compile_ast_event_atom(self, rule: dict[str, Any], node: dict[str, Any]) -> tuple[str, list[str]]:
        function_name = str(node.get("function", "")).strip()
        if not function_name:
            raise RuntimeError("ap_event_atom is missing function")
        resource = self._normalize_atom_resource(rule, node)
        tool_row = self._tool_row_for_action(resource, function_name)
        process_token = str(
            (tool_row or {}).get("process") or rule.get("process") or "any"
        ).strip().lower() or "any"
        product_token = self._normalize_atom_product(rule, node)
        context_token = self._serialize_context_object(self._normalize_atom_context(rule, node))
        ap = "/".join(
            [
                "ap_event",
                process_token,
                product_token,
                resource,
                function_name,
                context_token,
            ]
        )
        return ap, [ap]

    def _compile_ast_state_atom(self, rule: dict[str, Any], node: dict[str, Any]) -> tuple[str, list[str]]:
        state_name = self._normalize_state_name(node.get("state"))
        if not state_name:
            raise RuntimeError("ap_state_atom is missing state")
        resource = self._normalize_atom_resource(rule, node)
        process_token = self._normalize_process_token(node.get("process") or rule.get("process")) or "any"
        product_token = self._normalize_atom_product(rule, node)
        context_token = self._serialize_context_object(self._normalize_atom_context(rule, node))
        ap = "/".join(
            [
                "ap_state",
                process_token,
                product_token,
                resource,
                state_name,
                context_token,
            ]
        )
        return ap, [ap]

    def _compile_formula_ast_node(
        self,
        rule: dict[str, Any],
        node: dict[str, Any],
    ) -> tuple[str, list[str]]:
        if not isinstance(node, dict):
            raise RuntimeError(f"invalid formula_ast node: {node!r}")

        node_type = str(node.get("type", "") or "").strip()
        if node_type == "ap_event_atom":
            return self._compile_ast_event_atom(rule, node)
        if node_type == "ap_state_atom":
            return self._compile_ast_state_atom(rule, node)
        if node_type == "ap_selector":
            aps = self._expand_selector(rule, node)
            if not aps:
                resource = self._normalize_atom_resource(rule, node)
                match_spec = self._selector_match_spec(rule, node)
                raise RuntimeError(
                    f"ap_selector for rule {rule.get('id')} did not expand to any APs "
                    f"(resource={resource}, rule_resources={self._resolve_rule_resources(rule)}, "
                    f"match={self._debug_json(match_spec)}, selector={self._debug_json(node)})"
                )
            if len(aps) == 1:
                return aps[0], aps
            return f"({' | '.join(aps)})", aps

        op = str(node.get("op", "") or "").strip()
        if not op:
            raise RuntimeError(f"formula_ast node is missing op/type: {node!r}")

        if op in {"G", "F", "X", "!"}:
            arg_formula, arg_aps = self._compile_formula_ast_node(rule, node.get("arg") or {})
            if op == "!":
                return f"!({arg_formula})", arg_aps
            return f"{op} ({arg_formula})", arg_aps

        if op in {"&", "|"}:
            raw_args = node.get("args")
            if raw_args is None:
                raw_args = [node.get("left"), node.get("right")]
            args = [arg for arg in raw_args if isinstance(arg, dict)]
            compiled = [self._compile_formula_ast_node(rule, arg) for arg in args]
            formulas = [formula for formula, _ in compiled if formula]
            aps: list[str] = []
            for _, leaf_aps in compiled:
                aps.extend(leaf_aps)
            if not formulas:
                raise RuntimeError(f"operator {op} has no operands")
            joiner = f" {op} "
            if len(formulas) == 1:
                return formulas[0], self._dedupe_keep_order(aps)
            return f"({joiner.join(formulas)})", self._dedupe_keep_order(aps)

        if op in {"->", "U"}:
            left_formula, left_aps = self._compile_formula_ast_node(rule, node.get("left") or {})
            right_formula, right_aps = self._compile_formula_ast_node(rule, node.get("right") or {})
            return (
                f"({left_formula} {op} {right_formula})",
                self._dedupe_keep_order(left_aps + right_aps),
            )

        raise RuntimeError(f"unsupported formula_ast operator '{op}'")

    def _compile_formula_ast_for_rule(
        self,
        rule: dict[str, Any],
        formula_ast: dict[str, Any],
    ) -> Dict[str, Any]:
        ast = deepcopy(formula_ast)
        resource_vars = self._collect_resource_var_names(ast)
        if len(resource_vars) > 1:
            raise RuntimeError(
                f"formula_ast for rule {rule.get('id')} uses multiple distinct resource_var names "
                f"{sorted(resource_vars)}; use concrete resources or a single shared resource_var"
            )
        if resource_vars:
            resources = self._resolve_ast_resource_bindings(rule, ast)
            compiled_terms: list[str] = []
            aps: list[str] = []
            for resource in resources:
                bound_ast = self._bind_resource_var(ast, resource)
                formula, node_aps = self._compile_formula_ast_node(rule, bound_ast)
                compiled_terms.append(f"({formula})")
                aps.extend(node_aps)
            if not compiled_terms:
                raise RuntimeError(
                    f"formula_ast for rule {rule.get('id')} has resource_var but no grounded resources"
                )
            return {
                "formula_ast": formula_ast,
                "aps": self._dedupe_keep_order(aps),
                "ltlf": " & ".join(compiled_terms),
            }

        formula, aps = self._compile_formula_ast_node(rule, ast)
        return {
            "formula_ast": formula_ast,
            "aps": self._dedupe_keep_order(aps),
            "ltlf": formula,
        }

    @classmethod
    def _resources_for_formula(cls, formula: str, aps: list[str]) -> list[str]:
        resources: list[str] = []
        for ap in aps:
            if ap and ap not in formula:
                continue
            segments = cls._ap_segments(ap)
            if not segments:
                continue
            resource = str(segments.get("resource", "")).strip().lower()
            if resource and resource not in {"any", "robot"}:
                resources.append(resource)
        return cls._dedupe_keep_order(resources)

    def _validate_compiled_rule_logic(
        self,
        rule: dict[str, Any],
        compiled: dict[str, Any],
        *,
        refinement_feedback: str = "",
    ) -> None:
        aps = [
            str(ap).strip()
            for ap in (compiled.get("aps") or [])
            if str(ap or "").strip()
        ]
        ltlf = str(compiled.get("ltlf", "") or "").strip()
        if not aps or not ltlf:
            return

        family = self._infer_ltlf_family(
            rule,
            aps,
            refinement_feedback=refinement_feedback,
        )
        if family != "mutex":
            return

        concrete_rule_resources = [
            resource
            for resource in self._resolve_rule_resources(rule)
            if resource not in {"any", "robot"}
        ]
        if len(concrete_rule_resources) < 2:
            return

        used_resources = self._resources_for_formula(ltlf, aps)
        if len(used_resources) < 2:
            raise RuntimeError(
                f"compiled mutex rule {rule.get('id')} references fewer than two concrete resources "
                f"(used_resources={used_resources}, expected_resources={concrete_rule_resources}, ltlf={ltlf})"
            )

        conjuncts = self._split_ltlf_formula_by_top_level_and(ltlf)
        conjunct_resource_sets: list[set[str]] = []
        for conjunct in conjuncts:
            resources = set(self._resources_for_formula(conjunct, aps))
            if resources:
                conjunct_resource_sets.append(resources)

        if conjunct_resource_sets and all(len(resources) <= 1 for resources in conjunct_resource_sets):
            raise RuntimeError(
                f"compiled mutex rule {rule.get('id')} degenerates into independent single-resource conjuncts "
                f"(conjunct_resources={self._debug_json([sorted(resources) for resources in conjunct_resource_sets])}, "
                f"ltlf={ltlf})"
            )


    # ------------------------------------------------------------------ #
    # 1. Load NL safety requirements
    # ------------------------------------------------------------------ #
    def load_nl_safety_text(self) -> Optional[str]:
        """
        Load raw NL safety text from safety_file.
        """
        try:
            if not self.safety_file.exists():
                self.logger.warning(
                    "[SafetyLogic] Safety file missing: %s", self.safety_file
                )
                return None

            txt = self.safety_file.read_text(encoding="utf-8").strip()
            if not txt:
                self.logger.warning(
                    "[SafetyLogic] Safety file is empty: %s", self.safety_file
                )
                return None

            self.logger.info(
                "[SafetyLogic] Loaded NL safety text from %s", self.safety_file
            )
            return txt

        except Exception as exc:
            self.logger.exception(
                "[SafetyLogic] Failed to read safety file %s: %s",
                self.safety_file,
                exc,
            )
            return None

    # ------------------------------------------------------------------ #
    # 2. NL → structured safety rules (via LLM)
    # ------------------------------------------------------------------ #
    async def build_safety_rules(
        self,
        safety_text: str,
        refinement_feedback: str = "",
        previous_preview_rules: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Use the LLM to parse natural-language safety rules into structured
        safety rule nodes.

        Each rule node:

        - id
        - raw_text
        - constraint_type
        - process
        - product
        - resources
        - resource_types
        - event
        - context  (dict or None)
        """
        self.rules.clear()
        self.logic_raw.clear()
        self.global_safety_spec.clear()

        try:
            structured = await self._llm_parse_safety_rules(
                safety_text,
                refinement_feedback=refinement_feedback,
                previous_preview_rules=previous_preview_rules,
            )
        except Exception as exc:
            if self.logger:
                self.logger.exception(
                    "[SafetyLogic] LLM safety parsing failed: %s", exc
                )
            structured = []

        allowed_functions, function_process, allowed_resources, allowed_processes = (
            self._tool_grounding()
        )
        allowed_resource_types = self._allowed_resource_types()

        for idx, r in enumerate(structured, start=1):
            rule_id = r.get("id") or f"SAFE_{idx}"

            raw_text        = r.get("raw_text", "")
            constraint_type = r.get("constraint_type")
            process_raw     = str(r.get("process", "") or "").strip().lower()

            product_raw = r.get("product")
            products = []
            if isinstance(product_raw, list):
                # Filter out empty strings/nulls
                products = [str(p).strip() for p in product_raw if p]

            resources       = r.get("resources") or []
            resource_types_raw = r.get("resource_types")
            event_raw       = str(r.get("event", "") or "").strip()
            context         = r.get("context")       # expected to be dict or None

            event: str | None = event_raw if event_raw in allowed_functions else None
            if event_raw and event is None and self.logger:
                self.logger.warning(
                    "[SafetyLogic] Rule %s uses unsupported event '%s'; set event=null. "
                    "Supported functions: %s",
                    rule_id,
                    event_raw,
                    sorted(allowed_functions),
                )

            if event and event in function_process:
                process = function_process[event]
            elif process_raw in allowed_processes:
                process = process_raw
            else:
                process = None

            # Normalize resources
            if not isinstance(resources, list):
                resources = []
            normalized_resources: list[str] = []
            for res in resources:
                token = self._normalize_resource_token(res)
                if not token:
                    continue
                if token in {"any", "robot"}:
                    normalized_resources = ["any"]
                    break
                if token in allowed_resources:
                    normalized_resources.append(token)
                elif self.logger:
                    self.logger.warning(
                        "[SafetyLogic] Rule %s references unsupported resource '%s'; dropped.",
                        rule_id,
                        token,
                    )
            resources = self._dedupe_keep_order(normalized_resources)

            resource_types = self._normalize_string_list(resource_types_raw)
            normalized_resource_types: list[str] = []
            for resource_type in resource_types:
                token = self._normalize_resource_type_token(resource_type)
                if not token:
                    continue
                if token == "any":
                    normalized_resource_types = []
                    break
                if token in allowed_resource_types:
                    normalized_resource_types.append(token)
                elif self.logger:
                    self.logger.warning(
                        "[SafetyLogic] Rule %s references unsupported resource_type '%s'; dropped.",
                        rule_id,
                        token,
                    )
            resource_types = self._dedupe_keep_order(normalized_resource_types)

            # Normalize context (the LLM should return dict or None)
            context = self._normalize_context_object(context)

            node: Dict[str, Any] = {
                "id": rule_id,
                "raw_text": raw_text,
                "constraint_type": constraint_type,
                "process": process,
                "product": products,
                "resources": resources,
                "resource_types": resource_types,
                "event": event,
                "context": context,   # dict or None
            }

            self.rules.append(node)

        msg = f"[SafetyLogic] Parsed {len(self.rules)} structured safety rule(s) via LLM."
        if self.logger:
            self.logger.info(msg)

        return msg


    async def build_safety_rules_and_logic(
        self,
        safety_text: str,
        refinement_feedback: str = "",
        previous_preview_rules: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        High-level helper:

          1) NL -> structured rules
          2) structured rules -> AP strings + LTLf (via LLM)
          3) inject AP labels + full AP strings + LTLf into each rule
          4) build an optional global safety specification
        """
        msg = await self.build_safety_rules(
            safety_text,
            refinement_feedback=refinement_feedback,
            previous_preview_rules=previous_preview_rules,
        )
        if not self.rules:
            return msg

        # 2) structured rules -> APs + LTLf (raw)
        try:
            self.logic_raw = await self._llm_build_safety_logic(
                refinement_feedback=refinement_feedback,
                previous_preview_rules=previous_preview_rules,
            )
        except Exception as exc:
            if self.logger:
                self.logger.exception(
                    "[SafetyLogic] LLM safety logic generation failed: %s", exc
                )
            self.logic_raw = {}
            raise RuntimeError(f"safety logic generation failed: {exc}") from exc

        # 2.5) split LTLf formulas that are conjunctions of independent AP groups
        self._split_rules_on_independent_conjuncts()

        # 3) inject labels + full APs + LTLf into rule nodes
        self._apply_labels_into_rules()

        grounded_rules = [
            rule
            for rule in self.rules
            if isinstance(rule, dict)
            and rule.get("aps")
            and str(rule.get("ltlf", "") or "").strip()
        ]
        if not grounded_rules:
            allowed_functions, _, _, _ = self._tool_grounding()
            raise RuntimeError(
                "no grounded safety rules were generated from the current safety text. "
                f"Supported function events: {sorted(allowed_functions)}"
            )

        # 4) build a combined safety specification (optional)
        self.global_safety_spec = self._combine_safety_rules()

        if self.logger:
            self.logger.info(
                "[SafetyLogic] Attached APs + LTLf to %d rule(s). Global safety formula length: %d",
                len(self.rules),
                len(self.global_safety_spec.get("formula", "")),
            )

        return msg

    async def build_preview_interpretations(self) -> dict[str, Any]:
        """Generate preview-only natural-language explanations for the grounded rules."""
        if not self.rules:
            self.preview_interpretation_summary = ""
            return {"preview_summary": "", "rules": []}

        prompt = build_safety_interpretation_prompt(self.rules)
        parsed: dict[str, Any] = {}
        try:
            raw = await self.controller_agent.ask_llm(
                prompt=prompt,
                with_functions=False,
                temperature=0.0,
            )
            if isinstance(raw, dict):
                raise RuntimeError("ask_llm returned dict; expected JSON string.")
            parsed = json.loads(raw)
        except Exception as exc:
            if self.logger:
                self.logger.warning(
                    "[SafetyLogic] Preview interpretation generation failed; using fallback text: %s",
                    exc,
                )

        by_id: dict[str, str] = {}
        for item in parsed.get("rules", []) if isinstance(parsed.get("rules"), list) else []:
            if not isinstance(item, dict):
                continue
            rid = str(item.get("id", "")).strip()
            interpretation = str(item.get("interpretation", "")).strip()
            if rid and interpretation:
                by_id[rid] = interpretation

        for rule in self.rules:
            if not isinstance(rule, dict):
                continue
            rid = str(rule.get("id", "")).strip()
            interpretation = by_id.get(rid) or self._fallback_rule_interpretation(rule)
            rule["generated_interpretation"] = interpretation

        self.preview_interpretation_summary = self._fallback_preview_interpretation_summary(self.rules)

        return {
            "preview_summary": self.preview_interpretation_summary,
            "rules": [
                {
                    "id": str(rule.get("id", "")).strip(),
                    "interpretation": str(rule.get("generated_interpretation", "")).strip(),
                }
                for rule in self.rules
                if isinstance(rule, dict)
            ],
        }

    # ------------------------------------------------------------------ #
    # LLM call: NL → structured safety rules
    # ------------------------------------------------------------------ #
    async def _llm_parse_safety_rules(
        self,
        safety_text: str,
        *,
        refinement_feedback: str = "",
        previous_preview_rules: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Call the LLM with SAFETY_PARSE_PROMPT and tools_catalog, return
        a cleaned list of structured safety rules.
        """
        tools_catalog = getattr(self.controller_agent, "tools_catalog", [])
        capability_overview = ""
        if hasattr(self.controller_agent, "_static_caps_overview"):
            try:
                capability_overview = self.controller_agent._static_caps_overview()
            except Exception:
                capability_overview = ""

        prompt = build_safety_parse_prompt(
            safety_text,
            tools_catalog,
            capability_overview,
            refinement_feedback=refinement_feedback,
            previous_preview_rules=previous_preview_rules,
        )

        raw = await self.controller_agent.ask_llm(
            prompt=prompt,
            with_functions=False,
            temperature=0.0,
        )

        if isinstance(raw, dict):
            if self.logger:
                self.logger.error(
                    "[SafetyLogic] ask_llm returned dict, expected JSON string."
                )
            raise RuntimeError("ask_llm returned dict; expected JSON string.")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            if self.logger:
                self.logger.error(
                    "[SafetyLogic] LLM did not return valid JSON: %s\nRaw: %s",
                    exc,
                    raw,
                )
            raise

        rules = parsed.get("rules", [])
        cleaned: list[dict[str, Any]] = []

        for idx, r in enumerate(rules, start=1):
            if not isinstance(r, dict):
                continue

            resources = r.get("resources") or []
            if not isinstance(resources, list):
                resources = []
            resource_types = r.get("resource_types")
            if not isinstance(resource_types, list):
                resource_types = []

            # context is now expected to be an object (dict) or null
            context = self._normalize_context_object(r.get("context"))

            cleaned.append(
                {
                    "id":              r.get("id") or f"SAFE_{idx}",
                    "raw_text":        r.get("raw_text", ""),
                    "constraint_type": r.get("constraint_type"),
                    "process":         r.get("process"),
                    "product":         r.get("product"),
                    "resources":       resources,
                    "resource_types":  resource_types,
                    "event":           r.get("event"),
                    "context":         context,
                }
            )

        return cleaned

    # ------------------------------------------------------------------ #
    # LLM call: structured rules → AP strings + LTLf
    # ------------------------------------------------------------------ #
    async def _llm_build_safety_logic(
        self,
        *,
        refinement_feedback: str = "",
        previous_preview_rules: list[dict[str, Any]] | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Use the LLM to convert self.rules into AP lists + LTLf formulas.

        Returns (raw form, before labeling):
          {
            "SAFE_1": { "aps": [full_ap_str...], "ltlf": "..." },
            ...
          }
        """
        if not self.rules:
            return {}

        tools_catalog = getattr(self.controller_agent, "tools_catalog", [])
        prompt = build_safety_logic_prompt(
            self.rules,
            tools_catalog,
            refinement_feedback=refinement_feedback,
            previous_preview_rules=previous_preview_rules,
        )

        raw = await self.controller_agent.ask_llm(
            prompt=prompt,
            with_functions=False,
            temperature=0.0,
        )

        if isinstance(raw, dict):
            if self.logger:
                self.logger.error(
                    "[SafetyLogic] ask_llm for logic returned dict, expected JSON string."
                )
            raise RuntimeError("ask_llm returned dict; expected JSON string.")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            if self.logger:
                self.logger.error(
                    "[SafetyLogic] LLM did not return valid JSON for safety logic: %s\nRaw: %s",
                    exc,
                    raw,
                )
            raise

        result: Dict[str, Dict[str, Any]] = {}
        items = parsed.get("rules", [])
        allowed_functions, function_process, allowed_resources, _ = self._tool_grounding()
        allowed_states = self._allowed_state_names()
        rules_by_id = {
            str(r.get("id")): r
            for r in self.rules
            if isinstance(r, dict) and r.get("id")
        }
        unresolved: dict[str, list[str]] = {}

        for item in items:
            if not isinstance(item, dict):
                continue
            rid = str(item.get("id", "")).strip()
            aps = item.get("aps", [])
            ltlf = item.get("ltlf", "")

            if not rid:
                continue

            rule = rules_by_id.get(rid, {})
            formula_ast = item.get("formula_ast")
            if isinstance(formula_ast, dict):
                try:
                    compiled = self._compile_formula_ast_for_rule(rule, formula_ast)
                except Exception:
                    if self.logger:
                        self.logger.error(
                            "[SafetyLogic] Failed to compile formula_ast for rule %s. "
                            "rule=%s formula_ast=%s raw_item=%s",
                            rid,
                            self._debug_json(rule),
                            self._debug_json(formula_ast),
                            self._debug_json(item),
                        )
                    raise
                result[str(rid)] = compiled
                continue

            if not isinstance(aps, list):
                aps = []

            raw_aps = [str(a).strip() for a in aps if a]
            ltlf_text = str(ltlf).strip()
            rule_event = str(rule.get("event", "") or "").strip()
            rule_process = str(rule.get("process", "") or "").strip().lower()
            rule_context = self._normalize_context_object(rule.get("context"))
            rule_context_token = (
                self._serialize_context_object(rule_context) if rule_context else ""
            )
            rule_resources = rule.get("resources") or []
            fallback_resource = "any"
            if isinstance(rule_resources, list):
                for raw_res in rule_resources:
                    token = self._normalize_resource_token(raw_res)
                    if not token:
                        continue
                    if token in {"any", "robot"}:
                        fallback_resource = "any"
                        break
                    if token in allowed_resources:
                        fallback_resource = token
                        break

            sanitized_aps: list[str] = []
            unresolved_events_for_rule: list[str] = []

            for raw_ap in raw_aps:
                parts = raw_ap.split("/")
                if len(parts) < 6:
                    if self.logger:
                        self.logger.warning(
                            "[SafetyLogic] Rule %s AP '%s' ignored (expected 6 segments).",
                            rid,
                            raw_ap,
                        )
                    continue

                prefix, ap_process, ap_product, ap_resource, ap_event, ap_context = parts[:6]
                raw_prefix = str(prefix).strip()
                product_token = str(ap_product).strip().lower() or "any"
                context_token = rule_context_token or str(ap_context).strip() or "any"

                resource_token = self._normalize_resource_token(ap_resource)
                if resource_token not in {"any", "robot"} and resource_token not in allowed_resources:
                    resource_token = fallback_resource
                if resource_token == "robot":
                    resource_token = "any"
                if not resource_token:
                    resource_token = "any"

                if raw_prefix in {"ap", "ap_event"}:
                    event_token = str(ap_event).strip()
                    if event_token not in allowed_functions:
                        if rule_event in allowed_functions:
                            event_token = rule_event
                        else:
                            unresolved_events_for_rule.append(event_token or raw_ap)
                            continue

                    process_token = (
                        function_process.get(event_token)
                        or str(ap_process).strip().lower()
                        or rule_process
                        or "any"
                    )
                    normalized_ap = "/".join(
                        [
                            "ap_event",
                            process_token,
                            product_token,
                            resource_token,
                            event_token,
                            context_token,
                        ]
                    )
                    ltlf_text = ltlf_text.replace(raw_ap, normalized_ap)
                    sanitized_aps.append(normalized_ap)
                    continue

                if raw_prefix in {"ap_state", "sp"}:
                    state_token = self._normalize_state_name(ap_event)
                    if allowed_states and state_token not in allowed_states:
                        if self.logger:
                            self.logger.warning(
                                "[SafetyLogic] Rule %s state AP '%s' ignored (unsupported state '%s').",
                                rid,
                                raw_ap,
                                state_token,
                            )
                        continue
                    process_token = (
                        str(ap_process).strip().lower()
                        or rule_process
                        or "any"
                    )
                    normalized_ap = "/".join(
                        [
                            "ap_state",
                            process_token,
                            product_token,
                            resource_token,
                            state_token,
                            context_token,
                        ]
                    )
                    ltlf_text = ltlf_text.replace(raw_ap, normalized_ap)
                    sanitized_aps.append(normalized_ap)
                    continue

                if self.logger:
                    self.logger.warning(
                        "[SafetyLogic] Rule %s AP '%s' ignored (unsupported prefix '%s').",
                        rid,
                        raw_ap,
                        raw_prefix,
                    )

            sanitized_aps = self._dedupe_keep_order(sanitized_aps)

            if not sanitized_aps and rule_event in allowed_functions:
                fallback_ap = "/".join(
                    [
                        "ap_event",
                        function_process.get(rule_event, rule_process or "any"),
                        "any",
                        fallback_resource,
                        rule_event,
                        rule_context_token or "any",
                    ]
                )
                sanitized_aps = [fallback_ap]
                if not ltlf_text or fallback_ap not in ltlf_text:
                    ltlf_text = fallback_ap

            if unresolved_events_for_rule:
                unresolved[rid] = self._dedupe_keep_order(unresolved_events_for_rule)

            deterministic_ltlf = self._compile_ltlf_for_rule(
                rule,
                sanitized_aps,
                refinement_feedback=refinement_feedback,
            )
            if deterministic_ltlf and all(
                str(ap).startswith(("ap_event/", "ap/")) for ap in sanitized_aps
            ):
                ltlf_text = deterministic_ltlf
            elif sanitized_aps and (not ltlf_text or not any(ap in ltlf_text for ap in sanitized_aps)):
                ltlf_text = " & ".join(sanitized_aps) if len(sanitized_aps) > 1 else sanitized_aps[0]

            result[str(rid)] = {
                "aps": sanitized_aps,
                "ltlf": ltlf_text,
            }

        blocking = {rid: evs for rid, evs in unresolved.items() if not result.get(rid, {}).get("aps")}
        if blocking:
            detail = ", ".join(
                f"{rid}={events}" for rid, events in sorted(blocking.items())
            )
            raise RuntimeError(
                "Safety logic references unsupported events that cannot be grounded to catalog actions: "
                f"{detail}. Supported functions: {sorted(allowed_functions)}"
            )

        for rid, compiled in result.items():
            rule = rules_by_id.get(rid, {})
            if not rule or not isinstance(compiled, dict):
                continue
            self._validate_compiled_rule_logic(
                rule,
                compiled,
                refinement_feedback=refinement_feedback,
            )

        return result

    # ------------------------------------------------------------------ #
    # Split LTLf formula
    # ------------------------------------------------------------------ #
    def _split_ltlf_formula_by_top_level_and(self, formula: str) -> List[str]:
        """
        Split an LTLf formula string by top-level '&' operators.
        We ignore '&' that are inside parentheses.
        Example:
          "G (a -> X b) & G (c -> X d)"
          -> ["G (a -> X b)", "G (c -> X d)"]
        """
        if not formula:
            return []

        f = formula.strip()
        parts: List[str] = []
        depth = 0
        last = 0

        for i, ch in enumerate(f):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "&" and depth == 0:
                # split before this '&'
                segment = f[last:i].strip()
                if segment:
                    parts.append(segment)
                last = i + 1  # skip '&'

        tail = f[last:].strip()
        if tail:
            parts.append(tail)

        # If we didn't actually split, just return the whole thing
        return parts or [f]
    
    def _split_rules_on_independent_conjuncts(self) -> None:
        """
        For each rule's raw LTLf (before AP labels), check if it is a
        top-level '&' of independent conjuncts (disjoint AP sets).

        If so, split that rule into multiple rules:
          SAFE_2 -> SAFE_2-1, SAFE_2-2, ...

        Independence = two conjuncts do not share any AP strings.
        This works regardless of whether APs differ by agent, product, zone, etc.
        """
        if not self.logic_raw or not self.rules:
            return

        new_rules: List[Dict[str, Any]] = []
        new_logic: Dict[str, Dict[str, Any]] = {}

        for rule in self.rules:
            rid = rule.get("id")
            if not rid:
                new_rules.append(rule)
                continue

            raw_logic = self.logic_raw.get(rid)
            if not raw_logic:
                new_rules.append(rule)
                continue

            raw_ltlf = str(raw_logic.get("ltlf", "") or "").strip()
            raw_aps: List[str] = [str(a).strip() for a in (raw_logic.get("aps") or [])]

            if not raw_ltlf or not raw_aps:
                new_rules.append(rule)
                new_logic[rid] = {
                    "aps": raw_aps,
                    "ltlf": raw_ltlf,
                }
                continue

            # 1) split by top-level '&'
            conjuncts = self._split_ltlf_formula_by_top_level_and(raw_ltlf)
            if len(conjuncts) <= 1:
                # nothing to split
                new_rules.append(rule)
                new_logic[rid] = {
                    "aps": raw_aps,
                    "ltlf": raw_ltlf,
                }
                continue

            # 2) for each conjunct, collect the APs that appear in it
            ap_sets: List[set] = []
            for conj in conjuncts:
                used = {ap for ap in raw_aps if ap and ap in conj}
                ap_sets.append(used)

            # if any conjunct has no APs, splitting is risky -> keep whole rule
            if any(len(s) == 0 for s in ap_sets):
                new_rules.append(rule)
                new_logic[rid] = {
                    "aps": raw_aps,
                    "ltlf": raw_ltlf,
                }
                continue

            # 3) group conjuncts by AP overlap (very simple grouping)
            groups: List[List[int]] = []
            assigned: set[int] = set()

            for i in range(len(conjuncts)):
                if i in assigned:
                    continue
                group = [i]
                assigned.add(i)
                merged_aps = set(ap_sets[i])

                # put any conjunct that shares APs with this group into the same group
                for j in range(i + 1, len(conjuncts)):
                    if j in assigned:
                        continue
                    if ap_sets[j] & merged_aps:
                        assigned.add(j)
                        group.append(j)
                        merged_aps |= ap_sets[j]

                groups.append(group)

            # if everything is one group, no split
            if len(groups) <= 1:
                new_rules.append(rule)
                new_logic[rid] = {
                    "aps": raw_aps,
                    "ltlf": raw_ltlf,
                }
                continue

            if self.logger:
                self.logger.info(
                    "[SafetyLogic] Splitting rule %s into %d independent sub-rules.",
                    rid,
                    len(groups),
                )

            # 4) create new rules: SAFE_2-1, SAFE_2-2, ...
            for group_index, grp in enumerate(groups, start=1):
                new_id = f"{rid}-{group_index}"

                # subformula: AND of this group's conjuncts (keep order)
                sub_conjs = [conjuncts[k] for k in sorted(grp)]
                sub_formula = " & ".join(sub_conjs)

                # APs used by this group
                used_aps: set[str] = set()
                for k in grp:
                    used_aps |= ap_sets[k]
                sub_aps = [ap for ap in raw_aps if ap in used_aps]

                # clone rule and update id
                new_rule = dict(rule)
                new_rule["id"] = new_id
                new_rules.append(new_rule)

                new_logic[new_id] = {
                    "aps": sub_aps,
                    "ltlf": sub_formula,
                }

        # commit split
        self.rules = new_rules
        self.logic_raw = new_logic

    # ------------------------------------------------------------------ #
    # Inject AP labels + full AP strings into each rule
    # ------------------------------------------------------------------ #
    def _apply_labels_into_rules(self) -> None:
        """
        Convert raw AP strings into AP labels and store them inside each rule node:

          rule["aps"] = [
             {"label": "AP_001", "full": "evt/..."},
             ...
          ]
          rule["ltlf"] = "G !(AP_001 & AP_002)"

        Labels are global across all rules (AP_001, AP_002, ...).
        """
        # Collect all AP strings
        all_aps: list[str] = []
        for entry in self.logic_raw.values():
            all_aps.extend(entry.get("aps", []))

        unique_aps = sorted(set(all_aps))

        # Assign labels (global)
        ap_reverse: Dict[str, str] = {}  # full_ap -> label
        for idx, ap in enumerate(unique_aps, start=1):
            label = f"ap{idx:03d}"
            ap_reverse[ap] = label

        # Inject into rules
        for rule in self.rules:
            rid = rule.get("id")
            if not rid:
                continue

            raw_logic = self.logic_raw.get(rid)
            if not raw_logic:
                continue

            raw_aps = raw_logic.get("aps", [])
            raw_ltlf = raw_logic.get("ltlf", "")
            raw_formula_ast = raw_logic.get("formula_ast")

            # Build list of {label, full}
            labeled_aps = [
                {"label": ap_reverse[a], "full": a}
                for a in raw_aps
                if a in ap_reverse
            ]

            # Replace full AP strings by labels in formula
            formula = raw_ltlf
            for full_ap, label in ap_reverse.items():
                formula = formula.replace(full_ap, label)

            rule["aps"] = labeled_aps
            rule["ltlf"] = formula
            if raw_formula_ast is not None:
                rule["formula_ast"] = raw_formula_ast

    # ------------------------------------------------------------------ #
    # Combine all rules into one global safety spec
    # ------------------------------------------------------------------ #
    def _combine_safety_rules(self) -> Dict[str, Any]:
        """
        Combine all per-rule LTLf formulas into a single global safety
        specification:

          Φ_safety = ∧_i φ_i

        Returns:
          {
            "aps": { "AP_001": "evt/...", ... },
            "formula": "(φ_SAFE_1) & (φ_SAFE_2) & ..."
          }
        """
        global_ap_map: Dict[str, str] = {}
        formula_list: List[str] = []

        for rule in self.rules:
            # collect APs for this rule
            aps = rule.get("aps", [])
            for ap in aps:
                label = ap.get("label")
                full  = ap.get("full")
                if label and full:
                    global_ap_map[label] = full

            # collect formula
            phi = rule.get("ltlf")
            if phi:
                formula_list.append(f"({phi})")

        global_formula = " & ".join(formula_list) if formula_list else ""

        return {
            "aps": global_ap_map,
            "formula": global_formula,
        }

    # ------------------------------------------------------------------ #
    # ltlf to dfa
    # ------------------------------------------------------------------ #
    def build_dfas_per_rule(self, out_dir: Path | str | None = None):
        """
        Build one DFA per safety rule (SAFE_1, SAFE_2, ...) using ltlf2dfa.

        For each rule, we:
          - parse its LTLf formula
          - convert to DFA (DOT string)
          - save DOT and PNG under cais_spade_llm/safety/
        """
        if not self.rules:
            if self.logger:
                self.logger.warning("[SafetyLogic] No safety rules to build DFAs for.")
            return {}

        if LTLfParser is None:
            raise RuntimeError(
                "ltlf2dfa is not installed; DFA generation is unavailable in this environment."
            )

        parser = LTLfParser()
        out_dir = Path(out_dir) if out_dir else Path("cais_spade_llm/safety")
        out_dir.mkdir(parents=True, exist_ok=True)

        self.rule_dfas = {}

        for rule in self.rules:
            rid = rule.get("id")
            phi = rule.get("ltlf")
            if not rid or not phi:
                continue

            # e.g. '"G (a -> b)"' -> 'G (a -> b)'
            phi = phi.strip().strip('"').strip("'")

            try:
                ltlf_formula = parser(phi)
                dfa_dot = self._to_dfa_quiet(ltlf_formula)  # DOT string for this rule only
            except Exception as exc:
                if self.logger:
                    self.logger.exception(
                        "[SafetyLogic] Failed to build DFA for rule %s (formula '%s'): %s",
                        rid, phi, exc,
                    )
                continue

            # Store in memory
            self.rule_dfas[rid] = dfa_dot

            # Save DOT file
            dot_path = out_dir / f"{rid}_dfa.dot"
            dot_path.write_text(dfa_dot, encoding="utf-8")

            if self._is_placeholder_dfa_dot(dfa_dot):
                png_path = out_dir / f"{rid}_dfa.png"
                png_path.unlink(missing_ok=True)
                if self.logger:
                    self.logger.warning(
                        "[SafetyLogic] DFA (DOT) for %s saved to %s, but %s; skipping Graphviz render.",
                        rid,
                        dot_path,
                        self._placeholder_dfa_message(),
                    )
                continue

            if self.logger:
                self.logger.info(
                    "[SafetyLogic] DFA (DOT) for %s built and saved to %s",
                    rid, dot_path,
                )

            # Render PNG via Graphviz
            if Source is None:
                if self.logger:
                    self.logger.warning(
                        "[SafetyLogic] graphviz is not installed; skipping DFA render for %s.",
                        rid,
                    )
                continue
            try:
                src = Source(dfa_dot)
                render_path = src.render(
                    filename=str(out_dir / f"{rid}_dfa"),
                    format="png",
                    cleanup=True,
                )
                if self.logger:
                    self.logger.info(
                        "[SafetyLogic] DFA graph for %s rendered to %s",
                        rid, render_path,
                    )
            except Exception as exc:
                if self.logger:
                    self.logger.exception(
                        "[SafetyLogic] Graphviz rendering failed for %s: %s",
                        rid, exc,
                    )

        return self.rule_dfas

    def build_global_dfa(self):
        """
        Convert the combined LTLf formula into a DFA (DOT string) using ltlf2dfa.
        Then save it and render a PNG visualization.
        """
        formula_str = self.global_safety_spec.get("formula", "")
        if not formula_str:
            if self.logger:
                self.logger.warning("[SafetyLogic] No global LTLf safety formula to convert.")
            return None

        if LTLfParser is None:
            raise RuntimeError(
                "ltlf2dfa is not installed; DFA generation is unavailable in this environment."
            )

        try:
            parser = LTLfParser()
            ltlf_formula = parser(formula_str)
            dfa_dot = self._to_dfa_quiet(ltlf_formula)  # DOT string
        except Exception as exc:
            if self.logger:
                self.logger.exception(
                    "[SafetyLogic] Failed to build DFA from formula '%s': %s",
                    formula_str, exc
                )
            return None

        self.dfa = dfa_dot

        # Save DOT file
        out_dir = Path("cais_spade_llm/safety")
        out_dir.mkdir(parents=True, exist_ok=True)
        dot_path = out_dir / "cca_safety_dfa.dot"
        dot_path.write_text(dfa_dot, encoding="utf-8")

        if self._is_placeholder_dfa_dot(dfa_dot):
            png_path = out_dir / "cca_safety_dfa.png"
            png_path.unlink(missing_ok=True)
            if self.logger:
                self.logger.warning(
                    "[SafetyLogic] DFA (DOT) saved to %s, but %s; skipping Graphviz render.",
                    dot_path,
                    self._placeholder_dfa_message(),
                )
            return dfa_dot

        if self.logger:
            self.logger.info("[SafetyLogic] DFA (DOT) built and saved to %s", dot_path)

        # -------------------------
        # Graphviz Visualization
        # -------------------------
        if Source is None:
            if self.logger:
                self.logger.warning(
                    "[SafetyLogic] graphviz is not installed; skipping global DFA render."
                )
            return dfa_dot
        try:
            src = Source(dfa_dot)
            render_path = src.render(
                filename=str(out_dir / "cca_safety_dfa"),
                format="png",
                cleanup=True
            )
            if self.logger:
                self.logger.info("[SafetyLogic] DFA graph rendered to %s", render_path)

        except Exception as exc:
            if self.logger:
                self.logger.exception("[SafetyLogic] Graphviz rendering failed: %s", exc)

        return dfa_dot
    


    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def save(self, path: Path | str | None = None) -> None:
        """
        Save structured rules (including APs + LTLf) to JSON.
        """
        p = Path(path) if path else self.structured_safety_path
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "preview_interpretation_summary": self.preview_interpretation_summary,
            "rules": self.rules,
        }

        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        if self.logger:
            self.logger.info(
                "[SafetyLogic] Saved structured safety rules to %s",
                p.resolve(),
            )

    def load(self, path: Path | str | None = None) -> None:
        """
        Load structured rules (with APs + LTLf if present) from JSON.
        """
        p = Path(path) if path else self.structured_safety_path
        if not p.exists():
            if self.logger:
                self.logger.warning(
                    "[SafetyLogic] Safety file missing: %s", p
                )
            return

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.preview_interpretation_summary = str(
            data.get("preview_interpretation_summary", "") or ""
        ).strip()
        self.rules = data.get("rules", [])

        # Rebuild global spec if LTLf is already present
        self.global_safety_spec = self._combine_safety_rules()

        if self.logger:
            self.logger.info(
                "[SafetyLogic] Loaded structured safety rules from %s",
                p.resolve(),
            )

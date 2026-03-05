"""Safety rule parsing and LTLf/DFA generation utilities."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from ltlf2dfa.parser.ltlf import LTLfParser
from graphviz import Source

from prompts import (
    build_safety_parse_prompt,
    build_safety_logic_prompt,
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
    async def build_safety_rules(self, safety_text: str) -> str:
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
        - event
        - context  (dict or None)
        """
        self.rules.clear()
        self.logic_raw.clear()
        self.global_safety_spec.clear()

        try:
            structured = await self._llm_parse_safety_rules(safety_text)
        except Exception as exc:
            if self.logger:
                self.logger.exception(
                    "[SafetyLogic] LLM safety parsing failed: %s", exc
                )
            structured = []

        allowed_functions, function_process, allowed_resources, allowed_processes = (
            self._tool_grounding()
        )

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

            # Normalize context (the LLM should return dict or None)
            if not isinstance(context, dict):
                context = None   # DO NOT override dicts

            node: Dict[str, Any] = {
                "id": rule_id,
                "raw_text": raw_text,
                "constraint_type": constraint_type,
                "process": process,
                "product": products,
                "resources": resources,
                "event": event,
                "context": context,   # dict or None
            }

            self.rules.append(node)

        msg = f"[SafetyLogic] Parsed {len(self.rules)} structured safety rule(s) via LLM."
        if self.logger:
            self.logger.info(msg)

        return msg


    async def build_safety_rules_and_logic(self, safety_text: str) -> str:
        """
        High-level helper:

          1) NL -> structured rules
          2) structured rules -> AP strings + LTLf (via LLM)
          3) inject AP labels + full AP strings + LTLf into each rule
          4) build an optional global safety specification
        """
        msg = await self.build_safety_rules(safety_text)
        if not self.rules:
            return msg

        # 2) structured rules -> APs + LTLf (raw)
        try:
            self.logic_raw = await self._llm_build_safety_logic()
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

    # ------------------------------------------------------------------ #
    # LLM call: NL → structured safety rules
    # ------------------------------------------------------------------ #
    async def _llm_parse_safety_rules(self, safety_text: str) -> list[dict[str, Any]]:
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

        prompt = build_safety_parse_prompt(safety_text, tools_catalog, capability_overview)

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

            # context is now expected to be an object (dict) or null
            context = r.get("context")
            if not isinstance(context, dict):
                context = None

            cleaned.append(
                {
                    "id":              r.get("id") or f"SAFE_{idx}",
                    "raw_text":        r.get("raw_text", ""),
                    "constraint_type": r.get("constraint_type"),
                    "process":         r.get("process"),
                    "product":         r.get("product"),
                    "resources":       resources,
                    "event":           r.get("event"),
                    "context":         context,
                }
            )

        return cleaned

    # ------------------------------------------------------------------ #
    # LLM call: structured rules → AP strings + LTLf
    # ------------------------------------------------------------------ #
    async def _llm_build_safety_logic(self) -> Dict[str, Dict[str, Any]]:
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
        prompt = build_safety_logic_prompt(self.rules, tools_catalog)

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
            if not isinstance(aps, list):
                aps = []

            raw_aps = [str(a).strip() for a in aps if a]
            ltlf_text = str(ltlf).strip()

            rule = rules_by_id.get(rid, {})
            rule_event = str(rule.get("event", "") or "").strip()
            rule_process = str(rule.get("process", "") or "").strip().lower()
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
                product_token = str(ap_product).strip() or "any"
                context_token = str(ap_context).strip() or "any"
                prefix_token = str(prefix).strip() or "ap"

                resource_token = self._normalize_resource_token(ap_resource)
                if resource_token not in {"any", "robot"} and resource_token not in allowed_resources:
                    resource_token = fallback_resource
                if resource_token == "robot":
                    resource_token = "any"
                if not resource_token:
                    resource_token = "any"

                normalized_ap = "/".join(
                    [
                        prefix_token,
                        process_token,
                        product_token,
                        resource_token,
                        event_token,
                        context_token,
                    ]
                )
                ltlf_text = ltlf_text.replace(raw_ap, normalized_ap)
                sanitized_aps.append(normalized_ap)

            sanitized_aps = self._dedupe_keep_order(sanitized_aps)

            if not sanitized_aps and rule_event in allowed_functions:
                fallback_ap = "/".join(
                    [
                        "ap",
                        function_process.get(rule_event, rule_process or "any"),
                        "any",
                        fallback_resource,
                        rule_event,
                        "any",
                    ]
                )
                sanitized_aps = [fallback_ap]
                if not ltlf_text or fallback_ap not in ltlf_text:
                    ltlf_text = fallback_ap

            if unresolved_events_for_rule:
                unresolved[rid] = self._dedupe_keep_order(unresolved_events_for_rule)

            if sanitized_aps and (not ltlf_text or not any(ap in ltlf_text for ap in sanitized_aps)):
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
                "Safety logic references unsupported events that cannot be grounded to robot actions: "
                f"{detail}. Supported functions: {sorted(allowed_functions)}"
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

            if self.logger:
                self.logger.info(
                    "[SafetyLogic] DFA (DOT) for %s built and saved to %s",
                    rid, dot_path,
                )

            # Render PNG via Graphviz
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

        if self.logger:
            self.logger.info("[SafetyLogic] DFA (DOT) built and saved to %s", dot_path)

        # -------------------------
        # Graphviz Visualization
        # -------------------------
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

        payload = {"rules": self.rules}

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
        self.rules = data.get("rules", [])

        # Rebuild global spec if LTLf is already present
        self.global_safety_spec = self._combine_safety_rules()

        if self.logger:
            self.logger.info(
                "[SafetyLogic] Loaded structured safety rules from %s",
                p.resolve(),
            )

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from ltlf2dfa.parser.ltlf import LTLfParser

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

        for idx, r in enumerate(structured, start=1):
            rule_id = r.get("id") or f"SAFE_{idx}"

            raw_text        = r.get("raw_text", "")
            constraint_type = r.get("constraint_type")
            process         = r.get("process")
            product         = r.get("product")
            resources       = r.get("resources") or []
            event           = r.get("event")
            context         = r.get("context")       # expected to be dict or None

            # Normalize resources
            if not isinstance(resources, list):
                resources = []
            resources = [str(res).strip() for res in resources if res]

            # Normalize context (the LLM should return dict or None)
            if not isinstance(context, dict):
                context = None   # DO NOT override dicts

            node: Dict[str, Any] = {
                "id": rule_id,
                "raw_text": raw_text,
                "constraint_type": constraint_type,
                "process": process,
                "product": product,
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
            return msg + " LTLf logic generation failed."

        # 3) inject labels + full APs + LTLf into rule nodes
        self._apply_labels_into_rules()

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

        for item in items:
            if not isinstance(item, dict):
                continue
            rid  = item.get("id")
            aps  = item.get("aps", [])
            ltlf = item.get("ltlf", "")

            if not rid:
                continue
            if not isinstance(aps, list):
                aps = []

            aps = [str(a).strip() for a in aps if a]

            result[str(rid)] = {
                "aps": aps,
                "ltlf": str(ltlf).strip(),
            }

        return result

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
    def build_dfa(self):
        """
        Convert the combined LTLf formula into a DFA (DOT string) using ltlf2dfa.

        We store the DOT string and write it to a .dot file for visualization.
        """
        formula_str = self.global_safety_spec.get("formula", "")
        if not formula_str:
            if self.logger:
                self.logger.warning("[SafetyLogic] No global LTLf safety formula to convert.")
            return None

        try:
            parser = LTLfParser()
            ltlf_formula = parser(formula_str)
            dfa_dot = ltlf_formula.to_dfa()  # This is a DOT string, not a Python DFA object
        except Exception as exc:
            if self.logger:
                self.logger.exception(
                    "[SafetyLogic] Failed to build DFA from formula '%s': %s",
                    formula_str, exc
                )
            return None

        # Store the DOT string on the instance (for future use if needed)
        self.dfa = dfa_dot

        # Save DOT to file so you can inspect / render it
        out_dir = Path("cais_spade_llm/safety")
        out_dir.mkdir(parents=True, exist_ok=True)
        dot_path = out_dir / "cca_safety_dfa.dot"
        dot_path.write_text(dfa_dot, encoding="utf-8")

        if self.logger:
            # Optionally log a small preview of the DOT string
            preview = dfa_dot.splitlines()[0] if dfa_dot else "<empty>"
            self.logger.info("[SafetyLogic] DFA (DOT) built and saved to %s", dot_path)
            self.logger.debug("[SafetyLogic] DFA DOT first line: %s", preview)

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
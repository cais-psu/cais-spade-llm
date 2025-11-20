from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from prompts import build_safety_parse_prompt
import json

class SafetyPlanner:
    """
    SafetyPlanner (placeholder):

    1. Load NL safety requirements from file.
    2. Convert NL into internal safety rule nodes (placeholder, no LLM yet).
    """

    def __init__(self, controller_agent, safety_file: str | Path) -> None:
        self.controller_agent = controller_agent
        self.logger = controller_agent.logger
        self.safety_file = Path(safety_file)
        self.rules: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # 1. Load NL safety requirements
    # ------------------------------------------------------------------ #
    def load_nl_safety_text(self) -> Optional[str]:
        """
        Load raw NL safety text from safety_file.
        This is a simple helper, similar to ProductAgent._read_spec_text.
        """
        try:
            if not self.safety_file.exists():
                self.logger.warning(
                    "[SafetyPlanner] Safety file missing: %s", self.safety_file
                )
                return None

            txt = self.safety_file.read_text(encoding="utf-8").strip()
            if not txt:
                self.logger.warning(
                    "[SafetyPlanner] Safety file is empty: %s", self.safety_file
                )
                return None

            self.logger.info(
                "[SafetyPlanner] Loaded NL safety text from %s", self.safety_file
            )
            return txt

        except Exception as exc:
            self.logger.exception(
                "[SafetyPlanner] Failed to read safety file %s: %s",
                self.safety_file,
                exc,
            )
            return None

    # ------------------------------------------------------------------ #
    # NL → structured safety rules (via LLM)
    # ------------------------------------------------------------------ #
    async def build_safety_rules(self, safety_text: str) -> str:
        """
        Use the LLM to parse natural-language safety rules into structured
        safety rule nodes.

        Each rule has:
          - id
          - raw_text
          - constraint_type   (e.g. "no_simultaneous_action")
          - process           (e.g. "ASSEMBLY")
          - product           (e.g. "any")
          - resources         (e.g. ["xarm6", "ur5e"])
          - event             (e.g. "move_loaded_to_destination")
          - context           (e.g. "destination", "origin", "zone_id")
        """
        self.rules.clear()

        try:
            structured = await self._llm_parse_safety_rules(safety_text)
        except Exception as exc:
            if self.logger:
                self.logger.exception("[SafetyPlanner] LLM safety parsing failed: %s", exc)
            structured = []

        for idx, r in enumerate(structured, start=1):
            rule_id = f"SAFE_{idx}"

            raw_text        = r.get("raw_text", "")
            constraint_type = r.get("constraint_type")
            process         = r.get("process")
            product         = r.get("product")
            resources       = r.get("resources") or []
            event           = r.get("event")
            context         = r.get("context")

            if not isinstance(resources, list):
                resources = []
            resources = [str(res).strip() for res in resources if res]

            if isinstance(context, str):
                context = context.strip()
            else:
                context = None

            node: Dict[str, Any] = {
                "id": rule_id,
                "raw_text": raw_text,
                "constraint_type": constraint_type,
                "process": process,
                "product": product,
                "resources": resources,
                "event": event,
                "context": context,
            }

            self.rules.append(node)

        msg = f"[SafetyPlanner] Parsed {len(self.rules)} structured safety rule(s) via LLM."
        if self.logger:
            self.logger.info(msg)
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
        prompt = build_safety_parse_prompt(safety_text, tools_catalog)

        raw = await self.controller_agent.ask_llm(
            prompt=prompt,
            with_functions=False,
            temperature=0.0,
        )

        if isinstance(raw, dict):
            if self.logger:
                self.logger.error(
                    "[SafetyPlanner] ask_llm returned dict, expected JSON string."
                )
            raise RuntimeError("ask_llm returned dict; expected JSON string.")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            if self.logger:
                self.logger.error(
                    "[SafetyPlanner] LLM did not return valid JSON: %s\nRaw: %s",
                    exc,
                    raw,
                )
            raise

        rules = parsed.get("rules", [])
        cleaned: list[dict[str, Any]] = []

        for r in rules:
            if not isinstance(r, dict):
                continue

            resources = r.get("resources") or []
            if not isinstance(resources, list):
                resources = []

            context = r.get("context")
            if not isinstance(context, str):
                context = None

            cleaned.append(
                {
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
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def save(self, path: Path | str | None = None) -> None:
        p = Path(path) if path else self.structured_safety_path
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {"rules": self.rules}

        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        if self.logger:
            self.logger.info(f"[SafetyPlanner] Saved structured safety rules to {p.resolve()}")

    def load(self, path: Path | str | None = None) -> None:
        p = Path(path) if path else self.structured_safety_path
        if not p.exists():
            if self.logger:
                self.logger.warning(f"[SafetyPlanner] Safety file missing: {p}")
            return

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.rules = data.get("rules", [])

        if self.logger:
            self.logger.info(f"[SafetyPlanner] Loaded structured safety rules from {p.resolve()}")

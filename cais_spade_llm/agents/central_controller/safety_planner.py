from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


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
    # 2. NL → structured safety rules (placeholder)
    # ------------------------------------------------------------------ #
    async def build_safety_rules(self, safety_text: str) -> str:
        """
        Placeholder for parsing NL safety requirements.

        For now:
          - each non-empty line becomes one safety rule node
          - only 'id' and 'raw_text' are filled
        Later:
          - you will replace this with LLM parsing + AP/LTLf building.
        """
        self.rules.clear()

        lines = [ln.strip() for ln in safety_text.splitlines() if ln.strip()]

        for idx, line in enumerate(lines, start=1):
            rule_id = f"SAFE_{idx}"
            node: Dict[str, Any] = {
                "id": rule_id,
                "raw_text": line,
                # placeholders for later
                "rule_type": None,
                "actions": [],
                "scope_location": None,
                "resources": [],
            }
            self.rules.append(node)

        msg = f"[SafetyPlanner] Parsed {len(self.rules)} safety rule(s) (placeholder)."
        self.logger.info(msg)
        return msg

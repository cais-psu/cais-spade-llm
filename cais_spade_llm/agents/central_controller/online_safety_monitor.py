"""Online safety monitor that tracks running actions and DFA states."""

from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, FrozenSet, Set

# Ensure this import works in your project structure
from agents.central_controller.base_safety_checker import BaseSafetyChecker

class OnlineSafetyMonitor(BaseSafetyChecker):
    """
    Maintains the LIVE state of the factory.
    Inherits parsing/transition logic from BaseSafetyChecker.
    """
    def __init__(self, dfa_dots: Dict[str, str], safety_rules: list[dict]) -> None:
        """Initialize runtime safety state and history logging."""
        # Initialize the base class (parses DOTs)
        super().__init__(dfa_dots, safety_rules)
        
        self.logger = logging.getLogger("OnlineSafetyMonitor")

        # STATE: Track currently running actions across the factory
        self.running_aps: Set[str] = set()
        
        # STATE: Current DFA state pointer for every rule
        self.current_states: Dict[str, str] = {
            rid: data["initial"] for rid, data in self.dfas.items()
        }

        # Runtime DFA history log (JSONL)
        history_dir = Path("cais_spade_llm/history")
        history_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = history_dir / "safety_trace.jsonl"

    def _log_safety_event(
        self,
        *,
        rule_id: str,
        event_label: str,
        from_state: str,
        to_state: str,
        running_aps: List[str],
        candidate_aps: Optional[List[str]] = None,
        reason: str = "event",
        violated: bool = False,
    ) -> None:
        """
        Append a safety-related record to the history log (JSONL).
        """
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "rule_id": rule_id,
            "event": event_label,
            "reason": reason,
            "from": from_state,
            "to": to_state,
            "running_aps": running_aps,
            "candidate_aps": candidate_aps if candidate_aps is not None else [],
            "violated": violated,
        }
        try:
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            self.logger.exception("[Monitor] Failed to write DFA history log.")

    # ------------------------------------------------------------------ #
    # 1. Parsing Logic
    # ------------------------------------------------------------------ #
    def parse_resource_event(self, msg) -> Optional[dict[str, Any]]:
        """Parse and validate a resource_event message body into a dict."""
        try:
            data = json.loads(msg.body or "{}")
        except Exception:
            self.logger.error("Malformed resource_event body.")
            return None

        task_id       = data.get("task_id")
        resource_jid  = data.get("resource_jid")
        function_name = data.get("function_name")
        params        = data.get("params") or {}
        status        = data.get("status") or "running"

        if not (resource_jid and function_name):
            self.logger.warning("resource_event missing resource_jid or function_name.")
            return None

        return {
            "task_id": task_id,
            "resource_jid": resource_jid,
            "function_name": function_name,
            "params": params,
            "status": status,
            "failure_context": data.get("failure_context") or {},
        }

    # ------------------------------------------------------------------ #
    # 2. Logic Handling (Start/Finish)
    # ------------------------------------------------------------------ #
    def online_safety_validation(self, candidate_aps: List[str]) -> Tuple[bool, dict[str, Any]]:
        """
        Pure safety validation step: check if adding candidate APs would violate any rule.
        Does NOT mutate state.
        """
        # If no safety rules apply to this task, allow it.
        if not candidate_aps:
            return True, {
                "running_snapshot": list(self.running_aps),
                "next_states": {},
                "candidate_aps": candidate_aps,
            }

        # Combine Running + Candidate to see the "Next World State"
        sigma = frozenset(set(self.running_aps) | set(candidate_aps))

        next_states: Dict[str, str] = {}
        violated_rule = None
        violated_from = None
        violated_to = None

        for rule_id in self.dfas:
            curr = self.current_states.get(rule_id, "1")
            nxt = self._delta(rule_id, curr, sigma)

            # Check for violation state
            vio_state = self.dfas[rule_id].get("violation_state")
            if vio_state and nxt == vio_state:
                violated_rule = rule_id
                violated_from = curr
                violated_to = nxt
                # We stop at the first violation found
                break

            next_states[rule_id] = nxt

        if violated_rule:
            return False, {
                "violated_rule": violated_rule,
                "violated_from": violated_from,
                "violated_to": violated_to,
                "candidate_aps": candidate_aps,
                "running_snapshot": list(self.running_aps),
            }

        return True, {
            "running_snapshot": list(self.running_aps),
            "next_states": next_states,
            "candidate_aps": candidate_aps,
        }

    def process_start_event(self, event: dict[str, Any]) -> Tuple[bool, dict[str, Any]]:
        """
        Check if a task can start. If yes, update running_aps only.
        The DFA state is NOT advanced here; it advances on successful
        completion (process_finish_event) so that a failed task does not
        prematurely satisfy ordering requirements.
        """
        candidate_aps = self._map_task_to_aps(
            event["resource_jid"], event["function_name"], event["params"]
        )
        allowed, info = self.online_safety_validation(candidate_aps)

        if not allowed:
            self._log_safety_event(
                rule_id=info.get("violated_rule", "unknown"),
                event_label="start",
                from_state=info.get("violated_from", "unknown"),
                to_state=info.get("violated_to", "unknown"),
                running_aps=list(self.running_aps),
                candidate_aps=candidate_aps,
                reason="start_blocked",
                violated=True,
            )
            return False, info

        # Only update running_aps; do NOT advance DFA.
        self.running_aps.update(candidate_aps)
        next_states = info.get("next_states") or {}
        if next_states:
            for rule_id, nxt in next_states.items():
                prev = self.current_states.get(rule_id, "1")
                self._log_safety_event(
                    rule_id=rule_id,
                    event_label="start",
                    from_state=prev,
                    to_state=nxt,
                    running_aps=list(self.running_aps),
                    candidate_aps=candidate_aps,
                    reason="start_validated",
                    violated=False,
                )
        return True, {"running_snapshot": list(self.running_aps)}

    def process_finish_event(self, event: dict[str, Any]) -> None:
        """
        Task finished successfully. Remove its APs from the running set
        and advance the DFA using the original function's APs (not .done).
        Only successful completion should satisfy ordering requirements.
        """
        finished_aps = self._map_task_to_aps(
            event["resource_jid"], event["function_name"], event["params"]
        )
        for ap in finished_aps:
            self.running_aps.discard(ap)

        if not finished_aps:
            return

        # Advance DFA with the original APs to mark this action as completed.
        sigma = frozenset(set(self.running_aps) | set(finished_aps))
        next_states: Dict[str, str] = {}
        for rule_id in self.dfas:
            prev = self.current_states.get(rule_id, "1")
            nxt = self._delta(rule_id, prev, sigma)
            next_states[rule_id] = nxt
            self._log_safety_event(
                rule_id=rule_id,
                event_label="finish",
                from_state=prev,
                to_state=nxt,
                running_aps=list(self.running_aps),
                candidate_aps=finished_aps,
                reason="finish",
                violated=bool(self.dfas[rule_id].get("violation_state") == nxt),
            )
        self.current_states = next_states

    def process_fail_event(self, event: dict[str, Any]) -> None:
        """
        Task failed. Remove its running APs but do NOT advance the DFA.
        Since the action did not complete, the DFA stays in the
        pre-completion state, so dependent actions remain blocked.
        """
        failed_aps = self._map_task_to_aps(
            event["resource_jid"], event["function_name"], event["params"]
        )
        for ap in failed_aps:
            self.running_aps.discard(ap)

        if failed_aps:
            for rule_id in self.dfas:
                curr = self.current_states.get(rule_id, "1")
                self._log_safety_event(
                    rule_id=rule_id,
                    event_label="fail",
                    from_state=curr,
                    to_state=curr,
                    running_aps=list(self.running_aps),
                    candidate_aps=failed_aps,
                    reason="fail_no_advance",
                    violated=False,
                )

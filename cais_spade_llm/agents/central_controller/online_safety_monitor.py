"""Online safety monitor that tracks running actions and DFA states."""

from __future__ import annotations
import json
import logging
from typing import Tuple, Dict, Any, List, Optional, Set

# Ensure this import works in your project structure
from agents.central_controller.base_safety_checker import BaseSafetyChecker

class OnlineSafetyMonitor(BaseSafetyChecker):
    """
    Maintains the LIVE state of the factory.
    Inherits parsing/transition logic from BaseSafetyChecker.
    """
    def __init__(self, dfa_dots: Dict[str, str], safety_rules: list[dict]) -> None:
        super().__init__(dfa_dots, safety_rules)

        self.logger = logging.getLogger("OnlineSafetyMonitor")

        # STATE: Track currently running actions across the factory
        self.running_aps: Set[str] = set()

        # STATE: Current DFA state pointer for every rule
        self.current_states: Dict[str, str] = {
            rid: data["initial"] for rid, data in self.dfas.items()
        }

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
            return False, info

        # Only update running_aps; do NOT advance DFA.
        self.running_aps.update(candidate_aps)
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

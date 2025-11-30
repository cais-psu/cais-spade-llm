from __future__ import annotations
import json
import logging
from typing import Tuple, Dict, Any, List, Optional, FrozenSet, Set

# Ensure this import works in your project structure
from agents.central_controller.base_safety_checker import BaseSafetyChecker

class OnlineSafetyMonitor(BaseSafetyChecker):
    """
    Maintains the LIVE state of the factory.
    Inherits parsing/transition logic from BaseSafetyChecker.
    """
    def __init__(self, dfa_dots: Dict[str, str], safety_rules: list[dict]) -> None:
        # Initialize the base class (parses DOTs)
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
        }

    # ------------------------------------------------------------------ #
    # 2. Logic Handling (Start/Finish)
    # ------------------------------------------------------------------ #
    def process_start_event(self, event: dict[str, Any]) -> Tuple[bool, dict[str, Any]]:
        """
        Check if a task can start. If yes, update state.
        """
        candidate_aps = self._map_task_to_aps(
            event["resource_jid"], event["function_name"], event["params"]
        )
        
        # If no safety rules apply to this task, allow it.
        if not candidate_aps:
            return True, {}

        # Combine Running + Candidate to see the "Next World State"
        sigma = frozenset(set(self.running_aps) | set(candidate_aps))
        
        # Check Transitions
        next_states = {}
        violated_rule = None
        
        for rule_id in self.dfas:
            curr = self.current_states.get(rule_id, "1")
            
            # Calculate Next State (using logic from BaseSafetyChecker)
            nxt = self._delta(rule_id, curr, sigma)
            
            # Check for violation state
            vio_state = self.dfas[rule_id].get("violation_state")
            if vio_state and nxt == vio_state:
                violated_rule = rule_id
                # We stop at the first violation found
                break 
            
            next_states[rule_id] = nxt

        if violated_rule:
             return False, {
                 "violated_rule": violated_rule, 
                 "candidate_aps": candidate_aps,
                 "running_snapshot": list(self.running_aps)
             }

        # Commit State Updates
        self.running_aps.update(candidate_aps)
        self.current_states = next_states
        return True, {"running_snapshot": list(self.running_aps)}

    def process_finish_event(self, event: dict[str, Any]) -> None:
        """
        Task finished. Remove its APs from the running set.
        """
        candidate_aps = self._map_task_to_aps(
            event["resource_jid"], event["function_name"], event["params"]
        )
        for ap in candidate_aps:
            self.running_aps.discard(ap)
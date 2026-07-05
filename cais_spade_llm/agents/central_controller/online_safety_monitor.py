"""Online safety monitor that tracks running actions and DFA states."""

from __future__ import annotations

import json
import logging
from typing import Any

from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker


class OnlineSafetyMonitor(BaseSafetyChecker):
    """
    Maintains the LIVE state of the factory.
    Inherits parsing/transition logic from BaseSafetyChecker.
    """
    def __init__(
        self,
        dfa_dots: dict[str, str],
        safety_rules: list[dict],
        tools_catalog: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(dfa_dots, safety_rules, tools_catalog=tools_catalog)

        self.logger = logging.getLogger("OnlineSafetyMonitor")

        # STATE: Track currently running actions across the factory
        self.running_aps: set[str] = set()

        # STATE: Track currently active state APs across the factory
        self.resource_state_aps: dict[str, set[str]] = {}
        self.resource_states: dict[str, dict[str, Any]] = {}

        # STATE: Current DFA state pointer for every rule
        self.current_states: dict[str, str] = {
            rid: data["initial"] for rid, data in self.dfas.items()
        }

    # ------------------------------------------------------------------ #
    # 1. Parsing Logic
    # ------------------------------------------------------------------ #
    def parse_resource_event(self, msg) -> dict[str, Any] | None:
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
        current_state = data.get("current_state")

        if not (resource_jid and function_name):
            self.logger.warning("resource_event missing resource_jid or function_name.")
            return None

        return {
            "task_id": task_id,
            "resource_jid": resource_jid,
            "function_name": function_name,
            "params": params,
            "status": status,
            "current_state": current_state,
            "failure_context": data.get("failure_context") or {},
        }

    def seed_resource_states(self, resource_snapshots: dict[str, dict[str, Any]]) -> None:
        """Seed initial persistent state APs from resource snapshots when available."""
        for resource_jid, snapshot in (resource_snapshots or {}).items():
            if not isinstance(snapshot, dict):
                continue
            current_state = str(snapshot.get("current_state", "") or "").strip()
            if not current_state:
                continue
            self._update_resource_state(resource_jid, current_state, params={})

    def _all_state_aps(self) -> set[str]:
        active: set[str] = set()
        for labels in self.resource_state_aps.values():
            active |= set(labels)
        return active

    def _update_resource_state(
        self,
        resource_jid: str,
        current_state: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> None:
        payload = dict(params or {})
        self.resource_states[str(resource_jid)] = {
            "current_state": str(current_state),
            "params": payload,
        }
        self.resource_state_aps[str(resource_jid)] = set(
            self._map_state_to_aps(resource_jid, str(current_state), payload)
        )

    # ------------------------------------------------------------------ #
    # 2. Logic Handling (Start/Finish)
    # ------------------------------------------------------------------ #
    def online_safety_validation(
        self,
        candidate_aps: list[str],
        *,
        predicted_state_aps: list[str] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Pure safety validation step: check if adding candidate APs would violate any rule.
        Does NOT mutate state.
        """
        predicted = list(predicted_state_aps or [])

        # If no safety rules apply to this task, allow it.
        if not candidate_aps and not predicted:
            return True, {
                "running_snapshot": list(self.running_aps),
                "state_snapshot": sorted(self._all_state_aps()),
                "next_states": {},
                "candidate_aps": candidate_aps,
                "predicted_state_aps": predicted,
            }

        # Combine Running + Candidate to see the "Next World State"
        sigma = frozenset(
            set(self.running_aps)
            | self._all_state_aps()
            | set(candidate_aps)
            | set(predicted)
        )

        next_states: dict[str, str] = {}
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
                "state_snapshot": sorted(self._all_state_aps()),
                "predicted_state_aps": predicted,
            }

        return True, {
            "running_snapshot": list(self.running_aps),
            "state_snapshot": sorted(self._all_state_aps()),
            "next_states": next_states,
            "candidate_aps": candidate_aps,
            "predicted_state_aps": predicted,
        }

    def process_start_event(self, event: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """
        Check if a task can start. If yes, update running_aps only.
        The DFA state is NOT advanced here; it advances on successful
        completion (process_finish_event) so that a failed task does not
        prematurely satisfy ordering requirements.
        """
        event_params = dict(event.get("params") or {})
        task_id = str(event.get("task_id") or "").strip()
        if task_id:
            event_params.setdefault("task_id", task_id)
        candidate_aps = self._map_task_to_aps(
            event["resource_jid"], event["function_name"], event_params
        )
        predicted_state_aps = self._predict_state_aps(
            event["resource_jid"], event["function_name"], event_params
        )
        allowed, info = self.online_safety_validation(
            candidate_aps,
            predicted_state_aps=predicted_state_aps,
        )

        if not allowed:
            return False, info

        # Only update running_aps; do NOT advance DFA.
        self.running_aps.update(candidate_aps)
        return True, {
            "running_snapshot": list(self.running_aps),
            "state_snapshot": sorted(self._all_state_aps()),
        }

    def process_finish_event(self, event: dict[str, Any]) -> None:
        """
        Task finished successfully. Remove its APs from the running set
        and advance the DFA using the original function's APs (not .done).
        Only successful completion should satisfy ordering requirements.
        """
        event_params = dict(event.get("params") or {})
        task_id = str(event.get("task_id") or "").strip()
        if task_id:
            event_params.setdefault("task_id", task_id)
        finished_aps = self._map_task_to_aps(
            event["resource_jid"], event["function_name"], event_params
        )
        for ap in finished_aps:
            self.running_aps.discard(ap)

        current_state = str(event.get("current_state", "") or "").strip()
        projected_surface = self._state_surface_from_prediction(event_params)
        if current_state or projected_surface:
            current_state = current_state or str(
                projected_surface.get("resource_state") or ""
            ).strip()
            self._update_resource_state(
                event["resource_jid"],
                current_state,
                params=event_params,
            )

        # Advance DFA with the original APs to mark this action as completed.
        sigma = frozenset(
            set(self.running_aps)
            | self._all_state_aps()
            | set(finished_aps)
        )
        next_states: dict[str, str] = {}
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
        event_params = dict(event.get("params") or {})
        task_id = str(event.get("task_id") or "").strip()
        if task_id:
            event_params.setdefault("task_id", task_id)
        failed_aps = self._map_task_to_aps(
            event["resource_jid"], event["function_name"], event_params
        )
        for ap in failed_aps:
            self.running_aps.discard(ap)

        current_state = str(event.get("current_state", "") or "").strip()
        projected_surface = self._state_surface_from_prediction(event_params)
        if current_state or projected_surface:
            current_state = current_state or str(
                projected_surface.get("resource_state") or ""
            ).strip()
            self._update_resource_state(
                event["resource_jid"],
                current_state,
                params=event_params,
            )

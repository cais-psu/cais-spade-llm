"""Online safety monitor that tracks running actions and DFA states."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from itertools import product
from typing import Any

from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
from cais_spade_llm.agents.central_controller.local_composition import Budget
from cais_spade_llm.recovery_framework import fingerprint


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
        self.history_error: dict[str, Any] | None = None
        self.physical_monitor = LivePhysicalMonitor()

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

        task_id = data.get("task_id")
        resource_jid = data.get("resource_jid")
        function_name = data.get("function_name")
        params = data.get("params") or {}
        status = data.get("status") or "running"
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
        successor_state_aps: list[str] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Check that accepting states remain reachable after the projected label.

        Args:
            candidate_aps: Propositions for the candidate event.
            predicted_state_aps: Additional facts for legacy task admission.
            successor_state_aps: Complete projected state facts for recovery;
                replaces current state facts when supplied, including an empty list.

        Returns:
            Admissibility and evidence without mutating DFA states or running_aps.
        """
        if self.history_error is not None:
            return False, {"reason": "monitor_history_unavailable", "history_error": self.history_error}
        predicted = list(predicted_state_aps or [])

        # Outline projection supplies the full successor valuation, including
        # unchanged resources. Superseded current facts must not survive here.
        state_aps = (
            set(successor_state_aps)
            if successor_state_aps is not None
            else self._all_state_aps() | set(predicted)
        )
        sigma = frozenset(set(self.running_aps) | state_aps | set(candidate_aps))

        checks = [
            self.transition_evidence(rule_id, self.current_states[rule_id], sigma)
            for rule_id in self.dfas
        ]
        failed = next((row for row in checks if row["status"] != "passed"), None)
        info = {
            "rule_checks": checks,
            "label": sorted(sigma),
            "running_snapshot": sorted(self.running_aps),
            "state_snapshot": sorted(self._all_state_aps()),
            "successor_state_aps": sorted(state_aps),
            "candidate_aps": list(candidate_aps),
            "predicted_state_aps": predicted,
        }
        if failed is not None:
            info.update(
                violated_rule=failed["rule_id"], violated_from=failed["from"],
                violated_to=failed["to"], reason=failed["reason"],
            )
            return False, info
        info["next_states"] = {row["rule_id"]: row["to"] for row in checks}
        return True, info

    def process_start_event(self, event: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """
        Check if a task can start. If yes, update running_aps only.
        The DFA state is NOT advanced here; it advances on successful
        completion (process_finish_event) so that a failed task does not
        prematurely satisfy ordering requirements.
        """
        event_params = self._registered_process_params(
            event["resource_jid"], event["function_name"], dict(event.get("params") or {}))
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
        event_params = self._registered_process_params(
            event["resource_jid"], event["function_name"], dict(event.get("params") or {}))
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
            current_state = (
                current_state or str(projected_surface.get("resource_state") or "").strip()
            )
            self._update_resource_state(
                event["resource_jid"],
                current_state,
                params=event_params,
            )

        # Advance DFA with the original APs to mark this action as completed.
        sigma = frozenset(set(self.running_aps) | self._all_state_aps() | set(finished_aps))
        next_states: dict[str, str] = {}
        for rule_id in self.dfas:
            prev = self.current_states.get(rule_id, "1")
            try:
                nxt = self._delta(rule_id, prev, sigma)
            except ValueError:
                self.history_error = self.transition_evidence(rule_id, prev, sigma)
                raise
            next_states[rule_id] = nxt
        self.current_states = next_states

    def process_fail_event(self, event: dict[str, Any]) -> None:
        """
        Task failed. Remove its running APs but do NOT advance the DFA.
        Since the action did not complete, the DFA stays in the
        pre-completion state, so dependent actions remain blocked.
        """
        event_params = self._registered_process_params(
            event["resource_jid"], event["function_name"], dict(event.get("params") or {}))
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
            current_state = (
                current_state or str(projected_surface.get("resource_state") or "").strip()
            )
            self._update_resource_state(
                event["resource_jid"],
                current_state,
                params=event_params,
            )


class LivePhysicalMonitor:
    """Retain all monitor states compatible with accepted physical executions.

    A continuous cell denotes every finite nonempty AP word over its certified
    alphabet. Predictions use copies; only an authenticated execution outcome
    commits the resulting state set. A failed command retains every safe prefix.
    """

    def __init__(self) -> None:
        self.rules = None
        self.checker = None
        self.states: dict[str, set] = {}
        self.revision = 0
        self.history: list[dict] = []
        self.invalid_reason = ""

    def check(self, prepared: dict, *, budget: Budget | None = None) -> dict:
        """Check all selected rules across every certified primitive interval."""
        budget = budget or Budget()
        if self.invalid_reason:
            raise ValueError(self.invalid_reason)
        rules = prepared["rules"]
        identity = fingerprint(rules)
        if self.rules is not None and identity != fingerprint(self.rules):
            raise ValueError("Physical AP definitions changed during the run")
        checker = self.checker or BaseSafetyChecker(
            {row["rule_id"]: row["dfa_dot"] for row in rules}, rules)
        states = ({key: set(value) for key, value in self.states.items()} if self.states else
                  {identifier: {row["initial"]} for identifier, row in checker.dfas.items()})
        prefixes = deepcopy(states)
        for observation, values in zip(prepared["observations"], prepared["valuations"], strict=True):
            for cell in observation.get("rule_cells") or [values]:
                for identifier, current in states.items():
                    labels = list(cell[identifier])
                    options = [value if isinstance(value, list) else [value]
                               for value in cell[identifier].values()]
                    if any(not values or any(type(v) is not bool for v in values) for values in options):
                        raise ValueError("Physical AP has unresolved Boolean semantics")
                    alphabets = [frozenset(label for label, bit in zip(labels, bits, strict=True) if bit)
                                 for bits in product(*options)]
                    pending, reached = list(current), set()
                    while pending:
                        budget.check(len(reached))
                        state = pending.pop()
                        for alphabet in alphabets:
                            transition = checker.transition_evidence(identifier, state, alphabet)
                            if transition["status"] != "passed":
                                return {"status": "held", "reason": "possible_physical_requirement_violation",
                                        "rule_id": identifier, "transition": transition,
                                        "time_exact": observation["time_exact"]}
                            target = transition["to"]
                            if target not in reached:
                                reached.add(target)
                                if observation["phase"] == "between":
                                    pending.append(target)
                    states[identifier] = reached
                    prefixes[identifier].update(reached)
        pending_rules = sorted(identifier for identifier, values in states.items()
                               if not values <= set(checker.dfas[identifier]["accepting_states"]))
        if pending_rules:
            return {"status": "inconclusive", "reason": "temporal_continuation_proof_unavailable",
                    "prefix_safe": True, "pending_rule_ids": pending_rules,
                    "end_states": {key: sorted(value) for key, value in states.items()},
                    "end_accepting": False, "history_revision": self.revision}
        return {"status": "allowed", "prefix_safe": True, "pending_rule_ids": [],
                "end_accepting": True, "rules_fingerprint": identity,
                "history_revision": self.revision, "rules": deepcopy(rules),
                "end_states": {key: sorted(value) for key, value in states.items()},
                "prefix_states": {key: sorted(value) for key, value in prefixes.items()},
                "certificate_fingerprint": fingerprint({"rules": rules,
                    "observations": prepared["observations"], "valuations": prepared["valuations"]})}

    def commit(self, proof: dict, *, success: bool, execution_evidence: dict) -> None:
        """Retain conservative history only after the trusted owner reports execution."""
        if (proof.get("status") != "allowed" or proof["history_revision"] != self.revision
                or type(success) is not bool or not execution_evidence):
            raise ValueError("Physical execution history or proof changed before commit")
        if self.rules is not None and fingerprint(self.rules) != proof["rules_fingerprint"]:
            raise ValueError("Physical monitor definitions changed before commit")
        if self.rules is None:
            self.rules = deepcopy(proof["rules"])
            self.checker = BaseSafetyChecker({row["rule_id"]: row["dfa_dot"] for row in self.rules}, self.rules)
        self.states = {key: set(value) for key, value in proof[
            "end_states" if success else "prefix_states"].items()}
        self.history.append({"revision": self.revision, "success": success,
            "certificate_fingerprint": proof["certificate_fingerprint"],
            "execution_evidence": deepcopy(execution_evidence),
            "states": {key: sorted(value) for key, value in self.states.items()}})
        self.revision += 1
        if not success:
            self.invalid_reason = "physical_failure_stopping_coverage_unverified"

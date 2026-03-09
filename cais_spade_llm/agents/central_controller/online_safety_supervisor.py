"""Online supervisor over the plan/safety product winning set."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from agents.central_controller.online_fsa_monitor import OnlineFsaMonitor
from agents.central_controller.online_safety_monitor import OnlineSafetyMonitor


ProductState = Tuple[str, Tuple[str, ...], Tuple[Tuple[str, str, str], ...]]


class OnlineSafetySupervisor:
    """
    Supervisory layer over the live plan FSA state and the per-rule safety DFA states.

    It uses a precomputed winning set W over the product automaton to decide:
      - whether the current state is still recoverable
      - whether a candidate start transition keeps the execution recoverable
      - which next tasks remain safe from the current state
    """

    def __init__(
        self,
        *,
        winning_set_data: Dict[str, Any],
        fsa_monitor: OnlineFsaMonitor,
        safety_monitor: OnlineSafetyMonitor,
        enforcement_mode: str = "preventive",
    ) -> None:
        self.logger = logging.getLogger("OnlineSafetySupervisor")
        self.fsa = fsa_monitor
        self.safety = safety_monitor
        normalized_mode = str(enforcement_mode or "preventive").strip().lower()
        if normalized_mode not in {"preventive", "reactive"}:
            normalized_mode = "preventive"
        self.enforcement_mode = normalized_mode
        self.W: Set[ProductState] = set(winning_set_data.get("W") or set())
        self.graph: Dict[ProductState, List[Dict[str, Any]]] = dict(
            winning_set_data.get("product_graph") or {}
        )
        self.accepting_states: Set[ProductState] = set(
            winning_set_data.get("accepting_states") or set()
        )
        self.rule_ids: List[str] = list(winning_set_data.get("rule_ids") or [])
        self.base_resource_states: Dict[str, Dict[str, Any]] = deepcopy(
            winning_set_data.get("initial_resource_states") or {}
        )

    @staticmethod
    def _resource_state_signature(
        resource_states: Dict[str, Dict[str, Any]],
    ) -> Tuple[Tuple[str, str, str], ...]:
        items: List[Tuple[str, str, str]] = []
        for resource_jid, payload in sorted((resource_states or {}).items()):
            current_state = str((payload or {}).get("current_state") or "").strip()
            params = dict((payload or {}).get("params") or {})
            params_token = json.dumps(params, sort_keys=True, separators=(",", ":"))
            items.append((str(resource_jid), current_state, params_token))
        return tuple(items)

    def _merged_resource_states(self) -> Dict[str, Dict[str, Any]]:
        merged = deepcopy(self.base_resource_states)
        for resource_jid, payload in (getattr(self.safety, "resource_states", {}) or {}).items():
            merged[str(resource_jid)] = {
                "current_state": str((payload or {}).get("current_state") or ""),
                "params": dict((payload or {}).get("params") or {}),
            }
        return merged

    def _current_q_vector(self) -> Tuple[str, ...]:
        return tuple(str(self.safety.current_states.get(rule_id, "1")) for rule_id in self.rule_ids)

    def _current_safety_state_map(self) -> Dict[str, str]:
        return {rule_id: str(self.safety.current_states.get(rule_id, "1")) for rule_id in self.rule_ids}

    def current_product_state(self) -> ProductState:
        x = str(self.fsa.current_state or "")
        q_vec = self._current_q_vector()
        resource_sig = self._resource_state_signature(self._merged_resource_states())
        return (x, q_vec, resource_sig)

    def _pending_rule_ids(self, q_vec: Tuple[str, ...]) -> List[str]:
        pending: List[str] = []
        for idx, rule_id in enumerate(self.rule_ids):
            accepting_states = set(
                str(s) for s in (self.safety.dfas.get(rule_id, {}).get("accepting_states") or [])
            )
            violation_state = str(self.safety.dfas.get(rule_id, {}).get("violation_state") or "").strip()
            if violation_state and q_vec[idx] == violation_state:
                continue
            if accepting_states and q_vec[idx] not in accepting_states:
                pending.append(rule_id)
        return pending

    def _violated_rule_ids(self, q_vec: Tuple[str, ...]) -> List[str]:
        violated: List[str] = []
        for idx, rule_id in enumerate(self.rule_ids):
            violation_state = str(self.safety.dfas.get(rule_id, {}).get("violation_state") or "").strip()
            if violation_state and q_vec[idx] == violation_state:
                violated.append(rule_id)
        return violated

    def _safe_edges_from(self, product_state: ProductState) -> List[Dict[str, Any]]:
        edges = list(self.graph.get(product_state, []))
        return [edge for edge in edges if edge.get("successor") in self.W]

    def safe_next_task_ids(self) -> List[str]:
        current = self.current_product_state()
        task_ids = {
            str(edge.get("task_id"))
            for edge in self._safe_edges_from(current)
            if str(edge.get("event") or "").endswith(".start") and edge.get("task_id")
        }
        return sorted(task_ids)

    def safe_suffix_hint(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        start = self.current_product_state()
        if start not in self.W:
            return []
        if start in self.accepting_states:
            return []

        queue = deque([start])
        parent: Dict[ProductState, Optional[ProductState]] = {start: None}
        parent_edge: Dict[ProductState, Optional[Dict[str, Any]]] = {start: None}
        seen: Set[ProductState] = {start}

        target: Optional[ProductState] = None
        while queue:
            node = queue.popleft()
            if node in self.accepting_states:
                target = node
                break
            for edge in self._safe_edges_from(node):
                successor = edge.get("successor")
                if successor in seen:
                    continue
                seen.add(successor)
                parent[successor] = node
                parent_edge[successor] = edge
                queue.append(successor)

        if target is None:
            return []

        trace: List[Dict[str, Any]] = []
        cur = target
        while cur != start:
            edge = parent_edge.get(cur)
            if edge is not None:
                trace.append(
                    {
                        "task_id": edge.get("task_id"),
                        "event": edge.get("event"),
                        "resource_jid": edge.get("resource_jid"),
                        "function_name": edge.get("function_name"),
                        "params": dict(edge.get("params") or {}),
                    }
                )
            prev = parent.get(cur)
            if prev is None:
                break
            cur = prev
        trace.reverse()
        if limit is not None:
            return trace[:limit]
        return trace

    def classify(self, *, event_kind: Optional[str] = None) -> Dict[str, Any]:
        current = self.current_product_state()
        x, q_vec, _ = current
        violated_rule_ids = self._violated_rule_ids(q_vec)
        pending_rule_ids = self._pending_rule_ids(q_vec)
        safe_next_task_ids = self.safe_next_task_ids()
        event_token = str(event_kind or "").strip().lower()

        if violated_rule_ids:
            status = "violated"
            reason = "Current safety DFA state is already in a violation state."
        else:
            if self.enforcement_mode == "reactive":
                if event_token == "fail" and current not in self.W:
                    status = "inevitable_violation"
                    reason = (
                        "No remaining accepting continuation exists after the runtime failure."
                    )
                elif pending_rule_ids and current not in self.W:
                    status = "inevitable_violation"
                    reason = (
                        "A safety obligation is now open, but no remaining accepting continuation "
                        "exists in the current modeled plan."
                    )
                elif pending_rule_ids:
                    status = "pending_obligation"
                    reason = "Safety obligations remain open, but a safe continuation still exists."
                else:
                    status = "safe"
                    reason = (
                        "Reactive runtime supervision is active; no current safety obligation or "
                        "violation has been triggered."
                    )
            else:
                if current not in self.W:
                    status = "inevitable_violation"
                    reason = "No remaining accepting continuation exists in the current modeled plan."
                elif pending_rule_ids:
                    status = "pending_obligation"
                    reason = "Safety obligations remain open, but a safe continuation still exists."
                else:
                    status = "safe"
                    reason = (
                        "Current execution state is recoverable and all tracked safety DFAs are accepting."
                    )

        diagnosis = {
            "status": status,
            "rule_ids": violated_rule_ids or pending_rule_ids,
            "current_plan_state": x,
            "current_safety_states": self._current_safety_state_map(),
            "safe_next_task_ids": safe_next_task_ids,
            "safe_suffix_hint": self.safe_suffix_hint(limit=3) if current in self.W else [],
            "reason": reason,
            "enforcement_mode": self.enforcement_mode,
        }
        return diagnosis

    def check_candidate(self, event: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        current = self.current_product_state()
        diagnosis = self.classify()
        if diagnosis["status"] == "violated":
            return False, diagnosis

        if self.enforcement_mode == "reactive":
            task_id = str(event.get("task_id") or "").strip()
            event_label = f"{task_id}.start" if task_id else ""
            matching_safe_edges = [
                edge for edge in self._safe_edges_from(current)
                if str(edge.get("event") or "").strip() == event_label
            ]
            matching_edges = [
                edge for edge in (self.graph.get(current, []) or [])
                if str(edge.get("event") or "").strip() == event_label
            ]
            if diagnosis["status"] == "inevitable_violation" or not matching_safe_edges:
                return True, {
                    **diagnosis,
                    "status": "deferred_monitoring",
                    "candidate_event": event_label,
                    "reason": (
                        "Reactive runtime supervision is allowing the modeled task to run and will "
                        "diagnose/replan after subsequent runtime events."
                        if matching_edges or diagnosis["status"] == "inevitable_violation"
                        else "Reactive runtime supervision could not match a safe product edge for the candidate event."
                    ),
                }
            return True, {
                **diagnosis,
                "status": "safe",
                "candidate_event": event_label,
            }

        if diagnosis["status"] == "inevitable_violation":
            return False, diagnosis

        task_id = str(event.get("task_id") or "").strip()
        event_label = f"{task_id}.start" if task_id else ""

        matching_safe_edges = [
            edge for edge in self._safe_edges_from(current)
            if str(edge.get("event") or "").strip() == event_label
        ]
        if matching_safe_edges:
            return True, {
                **diagnosis,
                "status": "safe",
                "candidate_event": event_label,
            }

        matching_edges = [
            edge for edge in (self.graph.get(current, []) or [])
            if str(edge.get("event") or "").strip() == event_label
        ]
        reason = (
            "Candidate task would leave the recoverable winning set."
            if matching_edges
            else "Candidate task is not modeled as a safe successor from the current product state."
        )
        return False, {
            **diagnosis,
            "status": "blocked_candidate",
            "candidate_event": event_label,
            "reason": reason,
        }

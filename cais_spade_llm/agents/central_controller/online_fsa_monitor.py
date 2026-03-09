"""Runtime monitor for the plan FSA, tracking progress and reachable tasks."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class OnlineFsaMonitor:
    """
    Runtime monitor that tracks execution through the plan FSA
    (the global automaton saved in *_global_fsa.json).
    This is NOT the safety monitor; it is the execution monitor.
    """

    def __init__(self, fsa: Dict[str, Any]) -> None:
        self.logger = logging.getLogger("OnlineFsaMonitor")
        self.fsa = fsa or {}

        A = (self.fsa or {}).get("A") or {}
        self.current_state: Optional[str] = A.get("x0")
        self.transitions: List[Dict[str, Any]] = A.get("Tr") or []

        # Build (from_state, event) -> transition map
        self._tr_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._from_map: Dict[str, List[Dict[str, Any]]] = {}
        self._all_task_ids: set[str] = set()
        for tr in self.transitions:
            key = (tr.get("from"), tr.get("event"))
            if key[0] and key[1] and key not in self._tr_map:
                self._tr_map[key] = tr
            from_state = tr.get("from")
            if from_state:
                self._from_map.setdefault(from_state, []).append(tr)
            if tr.get("task_id"):
                self._all_task_ids.add(str(tr.get("task_id")))

        # Runtime plan FSA history log (JSONL)
        history_dir = Path("cais_spade_llm/monitor/history")
        history_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = history_dir / "online_fsa_trace.jsonl"
        self.last_event: Optional[Dict[str, Any]] = None
        # Track task outcomes for higher-level replanning context.
        self.completed_task_ids: set[str] = set()
        self.failed_task_ids: set[str] = set()

    def matches_fsa(self, fsa: Dict[str, Any]) -> bool:
        """Return True when the provided FSA is structurally identical."""
        return (self.fsa or {}) == (fsa or {})

    def _apply_event_label(self, event_label: str, *, task_id: str, outcome: str | None = None) -> bool:
        if not self.current_state:
            return False

        tr = self._tr_map.get((self.current_state, event_label))
        if not tr:
            return False

        self.current_state = tr.get("to") or self.current_state
        if outcome == "done":
            self.completed_task_ids.add(str(task_id))
            self.failed_task_ids.discard(str(task_id))
        elif outcome == "fail":
            self.failed_task_ids.add(str(task_id))
        return True

    def restore_runtime_progress(
        self,
        *,
        completed_task_ids: List[str] | None = None,
        running_task_ids: List[str] | None = None,
        failed_task_ids: List[str] | None = None,
    ) -> None:
        """Replay the executed runtime prefix into a fresh monitor for a repaired FSA."""
        A = (self.fsa or {}).get("A") or {}
        self.current_state = A.get("x0")
        self.completed_task_ids = set()
        self.failed_task_ids = set()

        completed = [str(task_id).strip() for task_id in (completed_task_ids or []) if str(task_id).strip()]
        running = [str(task_id).strip() for task_id in (running_task_ids or []) if str(task_id).strip()]
        failed = [str(task_id).strip() for task_id in (failed_task_ids or []) if str(task_id).strip()]

        for task_id in completed:
            started = self._apply_event_label(f"{task_id}.start", task_id=task_id)
            finished = self._apply_event_label(f"{task_id}.done", task_id=task_id, outcome="done")
            if not (started and finished):
                self.logger.warning(
                    "[OnlineFSA] Could not fully restore completed task %s into repaired FSA.",
                    task_id,
                )

        for task_id in running:
            if task_id in self.completed_task_ids:
                continue
            if not self._apply_event_label(f"{task_id}.start", task_id=task_id):
                self.logger.warning(
                    "[OnlineFSA] Could not restore running task %s into repaired FSA.",
                    task_id,
                )

        for task_id in failed:
            if task_id in self.completed_task_ids:
                continue
            started = self._apply_event_label(f"{task_id}.start", task_id=task_id)
            failed_ok = self._apply_event_label(f"{task_id}.fail", task_id=task_id, outcome="fail")
            if not (started and failed_ok):
                self.logger.warning(
                    "[OnlineFSA] Could not restore failed task %s into repaired FSA.",
                    task_id,
                )

    def _log_event(self, record: Dict[str, Any]) -> None:
        try:
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            self.logger.exception("[OnlineFSA] Failed to write online FSA history log.")

    def process_event(
        self,
        *,
        event_type: str,
        task_id: str,
        function_name: str,
        resource_jid: str,
        status: str,
    ) -> None:
        """
        Update the plan FSA with a runtime event and log the transition.
        """
        if not self.current_state:
            self.logger.warning("[OnlineFSA] Missing current state; cannot update.")
            return

        if event_type == "start":
            event_label = f"{task_id}.start"
        elif event_type == "done":
            event_label = f"{task_id}.done"
        else:
            event_label = f"{task_id}.fail"

        tr = self._tr_map.get((self.current_state, event_label))
        from_state = self.current_state
        to_state = from_state
        readable_event = None

        if tr:
            to_state = tr.get("to") or to_state
            readable_event = tr.get("readable_event")
            self.current_state = to_state

        # Track completion/failure for replanning context.
        if event_type == "done":
            self.completed_task_ids.add(str(task_id))
            # A completed task should not be considered failed.
            self.failed_task_ids.discard(str(task_id))
        elif event_type == "fail":
            self.failed_task_ids.add(str(task_id))

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "event": event_label,
            "readable_event": readable_event,
            "task_id": task_id,
            "function_name": function_name,
            "resource_jid": resource_jid,
            "status": status,
            "from": from_state,
            "to": to_state,
        }
        self.last_event = record
        self._log_event(record)

    def _parse_state(self, state: str) -> Dict[str, Dict[str, Any]]:
        """
        Parse a plan FSA state string into per-resource info.
        Example state:
          "(ur5e@localhost=(k=3,run=REQ_2_T3:place_insert),xarm6@localhost=(k=3,idle))"

        Output per resource:
          {
            "ur5e@localhost": {
              "k": 3,
              "status": "running",
              "run_task_id": "REQ_2_T3",
              "run_function": "place_insert",
            },
            "xarm6@localhost": {
              "k": 3,
              "status": "idle",
              "run_task_id": None,
              "run_function": None,
            }
          }
        """
        result: Dict[str, Dict[str, Any]] = {}
        if not state:
            return result

        pattern = re.compile(
            r"([^\s=,]+)=\(k=(\d+),(?:run=([^:]+):([^)]+)|idle)\)"
        )
        for match in pattern.finditer(state):
            res = match.group(1)
            k = int(match.group(2))
            run_task_id = match.group(3)
            run_fn = match.group(4)
            if run_task_id and run_fn:
                result[res] = {
                    "k": k,
                    "status": "running",
                    "run_task_id": run_task_id,
                    "run_function": run_fn,
                }
            else:
                result[res] = {
                    "k": k,
                    "status": "idle",
                    "run_task_id": None,
                    "run_function": None,
                }
        return result

    def _reachable_from_state(self, start_state: Optional[str]) -> tuple[set[str], set[str]]:
        """
        Return (reachable_task_ids, reachable_states) from a given state.
        """
        return self._reachable_from_state_filtered(start_state, blocked_events=set())

    def _reachable_from_state_filtered(
        self,
        start_state: Optional[str],
        *,
        blocked_events: set[str],
    ) -> tuple[set[str], set[str]]:
        """
        Return (reachable_task_ids, reachable_states) from a given state,
        skipping any transitions whose event appears in blocked_events.
        """
        reachable_tasks: set[str] = set()
        reachable_states: set[str] = set()
        queue: List[str] = [start_state] if start_state else []
        while queue:
            s = queue.pop(0)
            if not s or s in reachable_states:
                continue
            reachable_states.add(s)
            for tr in self._from_map.get(s, []):
                if tr.get("event") in blocked_events:
                    continue
                tid = tr.get("task_id")
                if tid:
                    reachable_tasks.add(str(tid))
                nxt = tr.get("to")
                if nxt and nxt not in reachable_states:
                    queue.append(nxt)
        return reachable_tasks, reachable_states

    def _next_task_ids_from_state(self, state: Optional[str]) -> List[str]:
        """
        Tasks whose START transitions are enabled from the given state.
        """
        return sorted({
            str(tr.get("task_id"))
            for tr in self._from_map.get(state or "", [])
            if tr.get("task_id") and str(tr.get("event", "")).endswith(".start")
        })

    def running_task_ids_from_state(self, state: Optional[str] = None) -> List[str]:
        """
        Task IDs currently marked as running in a plan FSA state.
        """
        parsed = self._parse_state(state or self.current_state or "")
        return sorted(
            str(info["run_task_id"])
            for info in parsed.values()
            if info.get("status") == "running" and info.get("run_task_id")
        )

    def has_blocked_descendants(self, task_id: str) -> bool:
        """
        Return True if the failure of task_id blocks any future tasks.

        Logic: compare tasks reachable from the current FSA state
        (1) unrestricted, vs (2) when task_id's .done transition is blocked.
        Any task that disappears from the reachable set when we block
        task_id.done is a task that depended on task_id completing.
        """
        cur = self.current_state
        reachable, _ = self._reachable_from_state(cur)
        reachable_without, _ = self._reachable_from_state_filtered(
            cur, blocked_events={f"{task_id}.done"}
        )
        return bool(set(reachable) - set(reachable_without))

    def build_replan_context(
        self,
        *,
        failure_event: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Gather online replanning context using the current plan FSA state.

        Key terms:
          - reachable_task_ids: tasks that can still occur in SOME future execution from the CURRENT state
            AND are not blocked by failures (failed tasks + their FSA-descendants).
          - failed_task_descendant_ids: tasks impacted by the failure (FSA-descendants).
          - completed_task_ids: tasks already completed at runtime.
        """
        cur_state = self.current_state

        # Reachable tasks from current state (FSA-based).
        reachable_tasks, _ = self._reachable_from_state(cur_state)

        # Optional: descendants of the failed task in the FSA.
        # These are "relevant tasks" to focus on during replanning.
        failed_task_descendant_ids: List[str] = []
        failed_task_id = None
        if failure_event:
            failed_task_id = (
                failure_event.get("failed_task_id")
                or failure_event.get("task_id")
            )

        # Build the blocked-by-failure set: failed tasks + all FSA-descendants.
        failed_task_ids = set(self.failed_task_ids)
        if failed_task_id:
            failed_task_ids.add(str(failed_task_id))

        blocked_events = {f"{tid}.done" for tid in failed_task_ids}
        reachable_unblocked, _ = self._reachable_from_state_filtered(
            cur_state, blocked_events=blocked_events
        )

        # Descendants for the single failed task (if provided) via FSA reachability delta.
        if failed_task_id:
            reachable_without_failed, _ = self._reachable_from_state_filtered(
                cur_state, blocked_events={f"{failed_task_id}.done"}
            )
            failed_task_descendant_ids = sorted(
                set(reachable_tasks) - set(reachable_without_failed)
            )

        completed_task_ids = set(self.completed_task_ids)

        return {
            "reachable_task_ids": sorted(reachable_unblocked),
            "completed_task_ids": sorted(completed_task_ids),
            "failed_task_id": failed_task_id,
            "failed_task_descendant_ids": failed_task_descendant_ids,
        }

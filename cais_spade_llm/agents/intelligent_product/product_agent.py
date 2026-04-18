"""High-level product agent that reads build instructions and instructs a resource agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os, uuid
import shutil
import threading
import time
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import TimeoutError as FutureTimeoutError
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message
from spade.template import Template

from cais_spade_llm.agents.shared_information.llm_agent import LlmAgent
from cais_spade_llm.agents.shared_information.local_dispatch import (
    send_agent_message,
    send_agent_message_sync,
)
from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.agents.intelligent_product.replanner.preprogrammed_bridge_scenarios import (
    build_preprogrammed_bridge_proposal,
    canonical_preprogrammed_bridge_scenario_id,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_artifacts import (
    DEFAULT_BRIDGE_DEBUG_DIR,
    write_bridge_artifacts,
)
from cais_spade_llm.resources.sensor.camera_module import CameraModule

_UNSET = object()
_DEFAULT_REPAIR_MAX_ATTEMPTS = 5
_CASE3_PREPROGRAMMED_SCENARIO_ID = "case3_llm_bridge"
_CASE3_PREPROGRAMMED_REQUIREMENT_FILES = frozenset({"case3_two_arm_llm_bridge.txt"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trace_with_timestamp(trace: Any, key: str) -> dict[str, Any]:
    out = dict(trace or {}) if isinstance(trace, dict) else {}
    out[key] = _utc_now_iso()
    return out


def _trace_delta_ms(trace: dict[str, Any], start_key: str, end_key: str) -> float | None:
    try:
        start = datetime.fromisoformat(str(trace.get(start_key) or ""))
        end = datetime.fromisoformat(str(trace.get(end_key) or ""))
    except Exception:
        return None
    return max(0.0, (end - start).total_seconds() * 1000.0)


class ProductAgent(LlmAgent):
    """
    SPADE ProductAgent
    - Reads a product specification (optional) and sends a task to ResourceAgents.
    - Receives ACKs from Resource/Robot agents and logs status.
    - Keeps the PA simple: RA is the single broker that asks the LLM with tools.
    """

    agent_role = "product"  # Registered role so the shared LLM base class can fetch the right prompts.

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str,
        resource_jids: Optional[Iterable[str]] = None,
        resource_agents: Optional[Iterable[Any]] = None,
        product_specification_file: Optional[str] = None,
        product_geometry_file: Optional[str] = None,
        safety_file: Optional[str] = None,
        instruction_override: Optional[str] = None,
        cca_jid: Optional[str] = None,
        camera: Optional["CameraModule"] = None,
        precomputed_bundle: Optional[Dict[str, Any]] = None,
        **kw,
    ) -> None:
        """
        :param resource_jids: List of RA JIDs to target (first is used).
        :param product_specification_file: Path to spec text (utf-8). Optional.
        :param instruction_override: If provided, this text is used instead of reading a file.
        """
        super().__init__(jid, password, name=name, agent_role="product", **kw)

        self.cca_jid = cca_jid or "cca@localhost"

        # Resource agents are the downstream executors; keep them in order and avoid mutating caller lists.
        self.resource_jids = list(resource_jids or [])
        self._resource_agent_refs = list(resource_agents or [])
        self.product_specification_file = product_specification_file
        self.product_geometry_file = product_geometry_file
        self.product_geometry: Dict[str, Any] = self._load_product_geometry(product_geometry_file)
        self.safety_file = safety_file
        # Manual instruction text provided at runtime overrides any file read.
        self.instruction_override = instruction_override

        # Cache safety text for use during replanning
        self.safety_text: str = ""

        self.precomputed_bundle: dict[str, Any] = dict(precomputed_bundle or {})

        # Planner scaffolding
        base_plan_dir = Path("cais_spade_llm/monitor/plan")
        base_state_dir = Path("cais_spade_llm/monitor/state")
        # For now: requirements file (NL → structured requirements)
        self.structured_requirements_path = base_plan_dir / f"{name}_requirements.json"
        # Reserved for later: full DAG task plan (requirements → task graph)
        self.plan_path = base_plan_dir / f"{name}_plan.json"
        self.global_fsa_path = base_plan_dir / f"{name}_global_fsa.json"
        # Runtime snapshots (overwritten each step)
        self.product_state_path = base_state_dir / f"{name}_product_state.json"
        self.resource_state_path = base_state_dir / f"{name}_resource_state.json"

        planner_resources = self._match_resource_objects(
            self._resource_agent_refs, self.resource_jids
        )
        self.process_planner = ProcessPlanner(self, planner_resources)
        
        #keep resolved resource agents on the ProductAgent for caps overview
        self.resource_agents = planner_resources

        # Simple in-memory map of task_id -> latest status string so UI/debug tooling can query progress.
        self.task_states: dict[str, str] = {}

        # Sensor: camera module for post-placement verification
        self.camera = camera if camera is not None else CameraModule()

        # Runtime tracking for replanning context (PRODUCT STATE ONLY)
        self.part_tracker: dict[str, dict[str, Any]] = {}  # part_name -> {location, state, last_task}
        self.execution_timeline: list[dict[str, Any]] = []  # [{timestamp, task_id, status, ...}]
        # NOTE: Robot states are queried directly from ResourceAgents, not cached here
        self._plan_result_inbox_registered = False
        self._runtime_repair_inflight = False
        self._runtime_repair_fail_streak = 0
        self._runtime_repair_max_attempts = _DEFAULT_REPAIR_MAX_ATTEMPTS
        self._bridge_generation_mode = "auto"
        self._bridge_reasoning_mode = "single_shot"
        self.runtime_repair_state = "idle"
        self.plan_safety_alert: dict[str, Any] | None = None
        self.runtime_recovery: dict[str, Any] = self._empty_runtime_recovery()
        self._runtime_recovery_context: dict[str, Any] = {}
        self.kickoff_result: dict[str, Any] = {
            "success": False,
            "message": "kickoff pending",
            "retries_used": 0,
            "retries_max": 0,
            "violated_rules": [],
            "witness_count": 0,
            "updated_at_utc": "",
            "product_name": name,
            "product_jid": str(self.jid),
            "stage": "kickoff",
            "alert": None,
        }
        self.startup_readiness: dict[str, Any] = {
            "startup_ready": False,
            "success": False,
            "continuing": True,
            "execution_blocked": True,
            "message": "kickoff pending",
            "retries_used": 0,
            "retries_max": 0,
            "violated_rules": [],
            "witness_count": 0,
            "updated_at_utc": "",
            "product_name": name,
            "product_jid": str(self.jid),
            "stage": "kickoff",
            "alert": None,
            "used_precomputed_bundle": False,
        }
        self._startup_readiness_event = asyncio.Event()
        self._kickoff_result_event = asyncio.Event()
        precomputed_policy = (
            self.precomputed_bundle.get("replan_policy", {})
            if isinstance(self.precomputed_bundle.get("replan_policy"), dict)
            else {}
        )
        if precomputed_policy:
            try:
                self._runtime_repair_max_attempts = max(
                    0,
                    min(
                        int(
                            precomputed_policy.get(
                                "auto_replan_max_attempts",
                                _DEFAULT_REPAIR_MAX_ATTEMPTS,
                            )
                            or 0
                        ),
                        10,
                    ),
                )
            except Exception:
                self._runtime_repair_max_attempts = _DEFAULT_REPAIR_MAX_ATTEMPTS
            self._bridge_generation_mode = str(
                precomputed_policy.get("bridge_generation_mode", "auto") or "auto"
            ).strip().lower()
            if self._bridge_generation_mode not in {"auto", "manual"}:
                self._bridge_generation_mode = "auto"
            self._bridge_reasoning_mode = str(
                precomputed_policy.get("bridge_reasoning_mode", "single_shot")
                or "single_shot"
            ).strip().lower()
            if self._bridge_reasoning_mode not in {"single_shot", "multi_turn"}:
                self._bridge_reasoning_mode = "single_shot"

        self.logger.info(f"ProductAgent '{name}' initialized.")

    # ------------------------------------------------------------------ #
    # Persistence helper
    # ------------------------------------------------------------------ #
    def _build_plan_validation_payload(
        self,
        *,
        skip_revalidation: bool = False,
        skip_offline_validation: bool | None = None,
    ):
        """Package plan + FSA for plan validation by the CCA."""
        if skip_offline_validation is not None:
            skip_revalidation = bool(skip_offline_validation)

        fsa = self.process_planner.global_fsa
        nodes = self.process_planner.nodes

        if fsa is None and not skip_revalidation:
            raise RuntimeError("Global FSA is None. Did you call save_global_fsa()?")

        return {
            "fsa": fsa or {},                # <-- upload FSA here
            "product_jid": str(self.jid),
            "plan": {"nodes": nodes},
            "runtime_context": self._build_runtime_plan_context(),
            "skip_revalidation": bool(skip_revalidation),
        }

    def _build_runtime_plan_context(self) -> dict[str, Any]:
        completed_task_ids: list[str] = []
        seen_completed: set[str] = set()
        for event in self.execution_timeline:
            if not isinstance(event, dict):
                continue
            task_id = str(event.get("task_id", "")).strip()
            status = str(event.get("status", "")).strip().lower()
            if not task_id:
                continue
            if status in {"completed", "finished"} and task_id not in seen_completed:
                seen_completed.add(task_id)
                completed_task_ids.append(task_id)

        running_task_ids: list[str] = []
        failed_task_ids: list[str] = []
        for node in self.process_planner.nodes:
            if not isinstance(node, dict) or node.get("type") != "task":
                continue
            task_id = str(node.get("id", "")).strip()
            if not task_id:
                continue
            status = str(node.get("status", "")).strip().lower()
            if status == "running":
                running_task_ids.append(task_id)
            elif status == "failed" or status.startswith("failed"):
                failed_task_ids.append(task_id)

        return {
            "completed_task_ids": completed_task_ids,
            "running_task_ids": running_task_ids,
            "failed_task_ids": failed_task_ids,
        }

    async def _send_agent_message(
        self,
        msg: Message,
        *,
        trace_category: str = "agent",
    ) -> None:
        """Send a SPADE message from agent methods that are outside a Behaviour."""
        self._dispatch_agent_message_sync(msg, trace_category=trace_category)

    def _dispatch_agent_message_sync(
        self,
        msg: Message,
        *,
        trace_category: str = "agent",
    ) -> None:
        """Send a SPADE message synchronously from local helper code."""
        send_agent_message_sync(
            self,
            msg,
            trace_category=trace_category,
            transport_label=trace_category,
        )

    def _run_coroutine_on_agent_loop_sync(
        self,
        coroutine: Any,
        *,
        timeout_sec: float = 10.0,
        operation_name: str = "product agent coroutine",
    ) -> Any:
        loop = getattr(self, "loop", None)
        if loop is None:
            raise RuntimeError("product agent loop is unavailable")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=max(1.0, float(timeout_sec)))
        except FutureTimeoutError as exc:
            future.cancel()
            raise RuntimeError(
                f"{str(operation_name or 'product agent coroutine').strip() or 'product agent coroutine'} "
                f"timed out after {max(1.0, float(timeout_sec)):.0f}s"
            ) from exc

    def _run_callable_on_agent_loop_sync(
        self,
        callback: Any,
        *,
        timeout_sec: float = 10.0,
        operation_name: str = "product agent callback",
    ) -> Any:
        loop = getattr(self, "loop", None)
        if loop is None:
            raise RuntimeError("product agent loop is unavailable")
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            return callback()

        future: ConcurrentFuture[Any] = ConcurrentFuture()

        def _invoke() -> None:
            try:
                future.set_result(callback())
            except Exception as exc:
                future.set_exception(exc)

        loop.call_soon_threadsafe(_invoke)
        try:
            return future.result(timeout=max(1.0, float(timeout_sec)))
        except FutureTimeoutError as exc:
            raise RuntimeError(
                f"{str(operation_name or 'product agent callback').strip() or 'product agent callback'} "
                f"timed out after {max(1.0, float(timeout_sec)):.0f}s"
            ) from exc

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _violation_summary(violations: list[dict[str, Any]] | None) -> tuple[list[str], int]:
        items = violations if isinstance(violations, list) else []
        violated_rules = sorted(
            {
                str(v.get("violated_rule_id"))
                for v in items
                if isinstance(v, dict) and v.get("violated_rule_id")
            }
        )
        return violated_rules, len(items)

    def _task_nodes_hash(self) -> str:
        task_nodes = [
            node
            for node in self.process_planner.nodes
            if isinstance(node, dict) and node.get("type") == "task"
        ]
        canonical = json.dumps(task_nodes, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _build_plan_safety_alert(
        self,
        *,
        stage: str,
        message: str,
        retries_used: int,
        retries_max: int,
        violations: list[dict[str, Any]] | None = None,
        paused: bool = False,
    ) -> dict[str, Any]:
        violated_rules, witness_count = self._violation_summary(violations)
        return {
            "product_name": self.agent_name,
            "product_jid": str(self.jid),
            "stage": str(stage),
            "message": str(message),
            "retries_used": int(retries_used),
            "retries_max": int(retries_max),
            "violated_rules": violated_rules,
            "witness_count": witness_count,
            "paused": bool(paused),
            "updated_at_utc": self._utc_now_iso(),
        }

    def _set_plan_safety_alert(
        self,
        *,
        stage: str,
        message: str,
        retries_used: int,
        retries_max: int,
        violations: list[dict[str, Any]] | None = None,
        paused: bool = False,
    ) -> dict[str, Any]:
        alert = self._build_plan_safety_alert(
            stage=stage,
            message=message,
            retries_used=retries_used,
            retries_max=retries_max,
            violations=violations,
            paused=paused,
        )
        self.plan_safety_alert = alert
        return alert

    def _clear_plan_safety_alert(self) -> None:
        self.plan_safety_alert = None

    def get_plan_safety_alert(self) -> dict[str, Any] | None:
        return dict(self.plan_safety_alert) if isinstance(self.plan_safety_alert, dict) else None

    def _empty_runtime_recovery(self) -> dict[str, Any]:
        return {
            "product_name": self.agent_name,
            "product_jid": str(self.jid),
            "status": "idle",
            "resolution_class": "none",
            "trigger": "",
            "failed_task_id": "",
            "message": "No active runtime recovery session.",
            "violated_rules": [],
            "witness_count": 0,
            "attempts_used": 0,
            "attempts_max": int(self._runtime_repair_max_attempts),
            "used_llm_bridge": False,
            "operator_guidance": "",
            "bridge_proposal": None,
            "bridge_debug": None,
            "bridge_artifacts": {},
            "bridge_approval_state": "none",
            "active_bridge_sequence": None,
            "bridge_feedback_history": [],
            "history": [],
            "updated_at_utc": self._utc_now_iso(),
        }

    def _sync_runtime_repair_state(self) -> None:
        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        approval_state = str(
            self.runtime_recovery.get("bridge_approval_state", "none") or "none"
        ).strip().lower()
        if status == "llm_bridge" and approval_state == "pending":
            self.runtime_repair_state = "paused_after_failure"
        elif status in {"des_search", "llm_bridge", "validating"}:
            self.runtime_repair_state = "repairing"
        elif status in {"bridge_ready", "human_required"}:
            self.runtime_repair_state = "paused_after_failure"
        else:
            self.runtime_repair_state = "idle"

    def _set_runtime_recovery(
        self,
        *,
        reset: bool = False,
        status: str | None = None,
        resolution_class: str | None = None,
        trigger: str | None = None,
        failed_task_id: str | None = None,
        message: str | None = None,
        attempts_used: int | None = None,
        attempts_max: int | None = None,
        used_llm_bridge: bool | None = None,
        operator_guidance: str | None = None,
        bridge_proposal: dict[str, Any] | None | object = _UNSET,
        bridge_debug: dict[str, Any] | None | object = _UNSET,
        bridge_artifacts: dict[str, Any] | None | object = _UNSET,
        bridge_approval_state: str | None = None,
        active_bridge_sequence: dict[str, Any] | None | object = _UNSET,
        bridge_feedback_history: list[str] | object = _UNSET,
        violations: list[dict[str, Any]] | None = None,
        append_history: bool = False,
        history_message: str | None = None,
    ) -> dict[str, Any]:
        current = self._empty_runtime_recovery() if reset else deepcopy(self.runtime_recovery)
        current["product_name"] = self.agent_name
        current["product_jid"] = str(self.jid)
        current["attempts_max"] = int(self._runtime_repair_max_attempts)

        if status is not None:
            current["status"] = str(status or "idle").strip() or "idle"
        if resolution_class is not None:
            current["resolution_class"] = str(resolution_class or "none").strip() or "none"
        if trigger is not None:
            current["trigger"] = str(trigger).strip()
        if failed_task_id is not None:
            current["failed_task_id"] = str(failed_task_id).strip()
        if message is not None:
            current["message"] = str(message).strip() or current.get("message", "")
        if attempts_used is not None:
            current["attempts_used"] = max(0, int(attempts_used))
        if attempts_max is not None:
            current["attempts_max"] = max(0, int(attempts_max))
        if used_llm_bridge is not None:
            current["used_llm_bridge"] = bool(used_llm_bridge)
        if operator_guidance is not None:
            current["operator_guidance"] = str(operator_guidance).strip()
        if bridge_proposal is not _UNSET:
            current["bridge_proposal"] = deepcopy(bridge_proposal) if isinstance(bridge_proposal, dict) else None
        if bridge_debug is not _UNSET:
            current["bridge_debug"] = deepcopy(bridge_debug) if isinstance(bridge_debug, dict) else None
        if bridge_artifacts is not _UNSET:
            current["bridge_artifacts"] = (
                deepcopy(bridge_artifacts)
                if isinstance(bridge_artifacts, dict)
                else {}
            )
        if bridge_approval_state is not None:
            current["bridge_approval_state"] = str(bridge_approval_state or "none").strip() or "none"
        if active_bridge_sequence is not _UNSET:
            current["active_bridge_sequence"] = (
                deepcopy(active_bridge_sequence)
                if isinstance(active_bridge_sequence, dict)
                else None
            )
        if bridge_feedback_history is not _UNSET:
            current["bridge_feedback_history"] = [
                str(item).strip()
                for item in (bridge_feedback_history or [])
                if str(item).strip()
            ]
        if violations is not None:
            violated_rules, witness_count = self._violation_summary(violations)
            current["violated_rules"] = violated_rules
            current["witness_count"] = witness_count

        event_message = str(history_message if history_message is not None else message or "").strip()
        if append_history and event_message:
            history = list(current.get("history") or [])
            history.append(
                {
                    "timestamp": self._utc_now_iso(),
                    "status": current.get("status", ""),
                    "message": event_message,
                }
            )
            current["history"] = history[-12:]

        current["updated_at_utc"] = self._utc_now_iso()
        self.runtime_recovery = current
        self._sync_runtime_repair_state()
        return deepcopy(current)

    def _clear_runtime_recovery(self) -> None:
        self.runtime_recovery = self._empty_runtime_recovery()
        self._runtime_recovery_context = {}
        self._sync_runtime_repair_state()

    def _runtime_recovery_blocks_execution(self) -> bool:
        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        return status not in {"", "idle", "resolved"}

    def get_runtime_recovery(self) -> dict[str, Any]:
        return deepcopy(self.runtime_recovery) if isinstance(self.runtime_recovery, dict) else self._empty_runtime_recovery()

    def _set_kickoff_result(
        self,
        *,
        success: bool,
        message: str,
        retries_used: int,
        retries_max: int,
        violations: list[dict[str, Any]] | None = None,
        alert: dict[str, Any] | None = None,
    ) -> None:
        violated_rules, witness_count = self._violation_summary(violations)
        self.kickoff_result = {
            "success": bool(success),
            "message": str(message),
            "retries_used": int(retries_used),
            "retries_max": int(retries_max),
            "violated_rules": violated_rules,
            "witness_count": witness_count,
            "updated_at_utc": self._utc_now_iso(),
            "product_name": self.agent_name,
            "product_jid": str(self.jid),
            "stage": "kickoff",
            "alert": dict(alert) if isinstance(alert, dict) else None,
        }
        if not self._kickoff_result_event.is_set():
            self._kickoff_result_event.set()

    def _set_startup_readiness(
        self,
        *,
        startup_ready: bool,
        success: bool,
        continuing: bool,
        message: str,
        retries_used: int,
        retries_max: int,
        violations: list[dict[str, Any]] | None = None,
        alert: dict[str, Any] | None = None,
        used_precomputed_bundle: bool = False,
    ) -> None:
        violated_rules, witness_count = self._violation_summary(violations)
        self.startup_readiness = {
            "startup_ready": bool(startup_ready),
            "success": bool(success),
            "continuing": bool(continuing),
            "execution_blocked": not bool(success),
            "message": str(message),
            "retries_used": int(retries_used),
            "retries_max": int(retries_max),
            "violated_rules": violated_rules,
            "witness_count": witness_count,
            "updated_at_utc": self._utc_now_iso(),
            "product_name": self.agent_name,
            "product_jid": str(self.jid),
            "stage": "kickoff",
            "alert": dict(alert) if isinstance(alert, dict) else None,
            "used_precomputed_bundle": bool(used_precomputed_bundle),
        }
        if not self._startup_readiness_event.is_set():
            self._startup_readiness_event.set()

    async def wait_for_startup_readiness(self, timeout: float | None = None) -> dict[str, Any]:
        if not self._startup_readiness_event.is_set():
            try:
                if timeout is None:
                    await self._startup_readiness_event.wait()
                else:
                    await asyncio.wait_for(self._startup_readiness_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                message = (
                    f"{self.agent_name}: startup readiness timed out after {timeout:.1f}s."
                )
                return {
                    "startup_ready": False,
                    "success": False,
                    "continuing": False,
                    "execution_blocked": True,
                    "message": message,
                    "retries_used": 0,
                    "retries_max": 0,
                    "violated_rules": [],
                    "witness_count": 0,
                    "updated_at_utc": self._utc_now_iso(),
                    "product_name": self.agent_name,
                    "product_jid": str(self.jid),
                    "stage": "kickoff",
                    "alert": self._build_plan_safety_alert(
                        stage="kickoff",
                        message=message,
                        retries_used=0,
                        retries_max=0,
                        violations=[],
                    ),
                    "used_precomputed_bundle": False,
                }
        return dict(self.startup_readiness)

    async def wait_for_kickoff_result(self, timeout: float | None = None) -> dict[str, Any]:
        if not self._kickoff_result_event.is_set():
            try:
                if timeout is None:
                    await self._kickoff_result_event.wait()
                else:
                    await asyncio.wait_for(self._kickoff_result_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                message = (
                    f"{self.agent_name}: startup plan validation timed out after {timeout:.1f}s."
                )
                return {
                    "success": False,
                    "message": message,
                    "retries_used": 0,
                    "retries_max": 0,
                    "violated_rules": [],
                    "witness_count": 0,
                    "updated_at_utc": self._utc_now_iso(),
                    "product_name": self.agent_name,
                    "product_jid": str(self.jid),
                    "stage": "kickoff",
                    "alert": self._build_plan_safety_alert(
                        stage="kickoff",
                        message=message,
                        retries_used=0,
                        retries_max=0,
                        violations=[],
                    ),
                }
        return dict(self.kickoff_result)



    def _persist_plan_snapshot(self) -> None:
        """Persist the current process planner graph (DAG nodes only) to disk."""
        if not self.plan_path:
            return

        try:
            import json

            self.plan_path.parent.mkdir(parents=True, exist_ok=True)
            with self.plan_path.open("w", encoding="utf-8") as f:
                json.dump({"nodes": self.process_planner.nodes}, f, indent=2)

            self.logger.debug(f"[Product] Saved plan to {self.plan_path.resolve()}")
        except Exception:
            self.logger.exception("[Product] Failed to persist plan snapshot.")

    def _persist_product_state(self) -> None:
        """Persist product state (part tracker, execution timeline) to disk."""
        if not self.product_state_path:
            return

        try:
            from datetime import datetime, timezone
            import json

            payload = {
                "part_tracker": self.part_tracker,
                "execution_timeline": self.execution_timeline,
                "runtime_repair_state": self.runtime_repair_state,
                "runtime_recovery": self.runtime_recovery,
                "plan_safety_alert": self.plan_safety_alert,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.product_state_path.parent.mkdir(parents=True, exist_ok=True)
            with self.product_state_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            self.logger.debug(f"[Product] Saved product state to {self.product_state_path.resolve()}")
        except Exception:
            self.logger.exception("[Product] Failed to persist product state.")

    def _bridge_debug_directory(self) -> Path:
        return DEFAULT_BRIDGE_DEBUG_DIR

    def _build_runtime_bridge_artifact_payload(
        self,
        *,
        phase: str,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, Any]:
        bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
        return {
            "phase": str(phase or "").strip(),
            "product_name": self.agent_name,
            "product_jid": str(self.jid),
            "runtime_recovery": deepcopy(self.runtime_recovery),
            "prepared_bridge_request": deepcopy(prepared_bridge_request),
            "bridge_debug": bridge_debug,
            "updated_at_utc": self._utc_now_iso(),
        }

    def _record_runtime_bridge_artifacts(
        self,
        *,
        phase: str,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, str]:
        artifact_payload = self._build_runtime_bridge_artifact_payload(
            phase=phase,
            prepared_bridge_request=prepared_bridge_request,
        )
        artifact_paths = write_bridge_artifacts(
            artifact_payload,
            phase_label=phase,
            debug_dir=self._bridge_debug_directory(),
            write_latest=False,
            write_session_transcript=True,
            filename_prefix=f"bridge_runtime_{phase}",
        )

        bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
        bridge_artifacts = deepcopy(bridge_debug.get("artifacts") or {})
        if not isinstance(bridge_artifacts, dict):
            bridge_artifacts = {}
        bridge_artifacts[str(phase or "").strip() or "runtime"] = deepcopy(artifact_paths)
        bridge_debug["artifacts"] = bridge_artifacts
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)

        if hasattr(self.process_planner, "_set_last_bridge_debug"):
            self.process_planner._set_last_bridge_debug(bridge_debug)

        runtime_bridge_artifacts = deepcopy(self.runtime_recovery.get("bridge_artifacts") or {})
        if not isinstance(runtime_bridge_artifacts, dict):
            runtime_bridge_artifacts = {}
        runtime_bridge_artifacts[str(phase or "").strip() or "runtime"] = deepcopy(
            artifact_paths
        )
        self._set_runtime_recovery(
            bridge_debug=bridge_debug,
            bridge_artifacts=runtime_bridge_artifacts,
        )
        self.logger.info(
            "[Product] Wrote bridge %s artifacts: prompt=%s response=%s session=%s",
            str(phase or "").strip() or "runtime",
            artifact_paths.get("prompt_artifact_path", ""),
            artifact_paths.get("response_artifact_path", ""),
            artifact_paths.get("session_transcript_artifact_path", ""),
        )
        return artifact_paths

    def _persist_resource_state(self) -> None:
        """Persist each resource agent's current state snapshot to disk."""
        if not self.resource_state_path:
            return

        try:
            import json

            state = {str(ra.jid): ra._snapshot_state() for ra in self.resource_agents}
            self.resource_state_path.parent.mkdir(parents=True, exist_ok=True)
            with self.resource_state_path.open("w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            self.logger.debug(f"[Product] Saved resource state to {self.resource_state_path.resolve()}")
        except Exception:
            self.logger.exception("[Product] Failed to persist resource state.")


    def _get_part_transition(
        self, function_name: str, task_node: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Look up the part_transition map, preferring node-level for bridge macros."""
        # Prefer node-level part_transition (set by bridge macro proposals).
        if task_node and task_node.get("part_transition"):
            return dict(task_node["part_transition"])
        self.__class__._load_shared_tools_catalogue()
        row = LlmAgent._TOOLS_BY_FUNC.get(function_name, {})
        return row.get("part_transition", {})

    def _tracked_part_name_for_task(
        self,
        task_node: Optional[Dict[str, Any]],
    ) -> str:
        """Resolve the canonical part name for one task node, including bridge macros."""
        if not isinstance(task_node, dict):
            return ""
        params = task_node.get("params") or {}
        if isinstance(params, dict):
            part_name = str(params.get("part_name") or "").strip()
            if part_name:
                return part_name
            touched_part = str(params.get("touched_part") or "").strip()
            if touched_part:
                return touched_part
        return str(task_node.get("touched_part") or "").strip()

    def _apply_part_tracker_update(
        self,
        part_name: str,
        function_name: str,
        status: str,
        params: Dict[str, Any],
        resource_jid: str,
        task_id: str,
        task_node: Optional[Dict[str, Any]] = None,
        observations: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Generic interpreter: applies the part_transition declared in each function's docstring."""
        transition_map = self._get_part_transition(function_name, task_node=task_node)
        if not transition_map:
            return  # function has no declared part_transition — nothing to track

        # Most-specific match wins: "failed:misplaced" → "failed" → nothing
        transition = transition_map.get(status) or transition_map.get(status.split(":")[0])
        if not transition:
            return

        self.part_tracker.setdefault(part_name, {"state": "unknown", "location": None})
        entry: Dict[str, Any] = {"state": transition["state"]}

        if "location_template" in transition:
            entry["location"] = transition["location_template"].format(
                resource_jid=resource_jid,
            )

        if "location_param" in transition:
            entry["location"] = params.get(transition["location_param"])

        exec_mode = str(os.environ.get("EXECUTION_MODE", "dry_run")).strip().lower()
        perception_backend = str(os.environ.get("PERCEPTION_BACKEND", "none")).strip().lower()

        if transition.get("verify_camera"):
            if exec_mode == "dry_run" or perception_backend in {"", "none"}:
                entry["state"] = "assembled"
                entry["camera_verification"] = "skipped_dry_run"
            elif perception_backend == "yolo":
                # Placeholder for future physical-camera verification.
                entry["state"] = "assembled"
                entry["camera_verification"] = "todo_yolo_placeholder"
            else:
                position = self.camera.observe(part_name)
                if position is not None:
                    entry["state"] = "assembled"
                    entry["position"] = position
                    entry["camera_verification"] = "detected"
                else:
                    entry["state"] = "untracked"
                    entry["observation_required"] = True
                    entry["camera_verification"] = "not_detected"
                    self.logger.error(
                        "[Product] %s is untracked after placement; camera backend '%s' could not locate it.",
                        part_name,
                        perception_backend or "unknown",
                    )

        if transition.get("camera_locate"):
            last_known = (
                params.get(transition["last_known_param"])
                if "last_known_param" in transition else None
            )
            entry["location"] = None
            if last_known:
                entry["last_known_location"] = last_known

            if exec_mode == "dry_run" or perception_backend in {"", "none", "yolo"}:
                entry["state"] = "untracked"
                entry["observation_required"] = True
                entry["camera_verification"] = "unavailable"
            else:
                position = self.camera.observe(part_name)
                if position is not None:
                    entry["state"] = "misplaced"
                    entry["position"] = position
                    entry["camera_verification"] = "detected"
                else:
                    entry["state"] = "untracked"
                    entry["observation_required"] = True
                    entry["camera_verification"] = "not_detected"
                    self.logger.error(
                        "[Product] %s is untracked; camera backend '%s' could not locate it.",
                        part_name,
                        perception_backend or "unknown",
                    )

        if transition.get("observation_required"):
            entry["observation_required"] = True
            entry["location"] = None
            if "last_known_param" in transition:
                entry["last_known_location"] = params.get(transition["last_known_param"])
            elif "last_known_template" in transition:
                entry["last_known_location"] = transition["last_known_template"].format(resource_jid=resource_jid)

        if status == "completed":
            entry["last_successful_task"] = task_id

        # Preserve origin info when a part transitions to in_gripper so
        # recovery planners know where to return it.
        if transition.get("state") == "in_gripper":
            origin = str(params.get("origin_resource_location") or "").strip()
            if origin:
                entry["origin_resource_location"] = origin
            obs = observations or {}
            origin_pose = obs.get("origin_pose")
            if isinstance(origin_pose, dict) and {"x", "y", "z"} <= set(origin_pose):
                entry["origin_pose"] = {
                    "x": float(origin_pose["x"]),
                    "y": float(origin_pose["y"]),
                    "z": float(origin_pose["z"]),
                }

        self.part_tracker[part_name].update(entry)



    def _restore_product_state(self) -> None:
        """Restore product state (part tracker, timeline) and plan nodes from disk."""
        import json

        # --- Restore product state ---
        # Prefer the dedicated product_state file; fall back to legacy plan.json key.
        product_state: dict = {}
        if self.product_state_path and self.product_state_path.exists():
            try:
                with self.product_state_path.open("r", encoding="utf-8") as f:
                    product_state = json.load(f)
            except Exception:
                self.logger.exception("[Product] Failed to read product state file.")
        elif self.plan_path and self.plan_path.exists():
            try:
                with self.plan_path.open("r", encoding="utf-8") as f:
                    product_state = json.load(f).get("product_state", {})
            except Exception:
                self.logger.exception("[Product] Failed to read legacy product state from plan file.")

        if product_state:
            self.part_tracker = product_state.get("part_tracker", {})
            self.execution_timeline = product_state.get("execution_timeline", [])
            runtime_repair_state = str(product_state.get("runtime_repair_state", "")).strip()
            if runtime_repair_state:
                self.runtime_repair_state = runtime_repair_state
            runtime_recovery = product_state.get("runtime_recovery")
            if isinstance(runtime_recovery, dict):
                restored_runtime_recovery = deepcopy(runtime_recovery)
                restored_runtime_recovery.pop("replan_mode", None)
                self.runtime_recovery = restored_runtime_recovery
                self._sync_runtime_repair_state()
            alert = product_state.get("plan_safety_alert")
            self.plan_safety_alert = dict(alert) if isinstance(alert, dict) else None
            self.logger.info(
                f"[Product] Restored product state: {len(self.part_tracker)} parts, "
                f"{len(self.execution_timeline)} timeline events"
            )

        # --- Restore plan nodes ---
        if self.plan_path and self.plan_path.exists():
            try:
                with self.plan_path.open("r", encoding="utf-8") as f:
                    nodes = json.load(f).get("nodes", [])
                if nodes:
                    self.process_planner.nodes = nodes
                    self.logger.info(f"[Product] Restored {len(nodes)} plan nodes")
            except Exception:
                self.logger.exception("[Product] Failed to restore plan nodes.")

    # --------------------------------------------------------------------- #
    # SPADE lifecycle
    # --------------------------------------------------------------------- #

    async def setup(self):
        await super().setup()

        # Kickoff behaviour (runs once) to build the plan.
        # Template ensures _Kickoff only receives plan_safety_result messages
        # and does not steal ACKs or other messages from the queue.
        t_kickoff = Template()
        t_kickoff.set_metadata("type", "plan_safety_result")
        self.add_behaviour(self._Kickoff(), t_kickoff)

        # ACK inbox
        t_ack = Template()
        t_ack.set_metadata("type", "ack")
        self.add_behaviour(self._AckInbox(), t_ack)

        # Replan request inbox (from CCA)
        t_replan = Template()
        t_replan.set_metadata("type", "replan_request")
        self.add_behaviour(self._ReplanInbox(), t_replan)

        # Retry-ready inbox (from CCA) for transient safety blocks that cleared.
        t_retry_ready = Template()
        t_retry_ready.set_metadata("type", "task_retry_ready")
        self.add_behaviour(self._TaskRetryReadyInbox(), t_retry_ready)

        # Plan executor (runs cycles, dispatches DAG tasks)
        # self.add_behaviour(self._PlanExecutor())

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _read_safety_text(self) -> str:
        """
        Read the NL safety file if provided. Returns empty string if missing.
        """
        if not self.safety_file:
            return ""
        
        try:
            p = Path(self.safety_file)
            if p.exists():
                txt = p.read_text(encoding="utf-8").strip()
                self.logger.info(f"[Product] Loaded safety constraints from {p}")
                return txt
            else:
                self.logger.warning(f"[Product] Safety file path provided but not found: {p}")
        except Exception as e:
            self.logger.exception(f"[Product] Failed to read safety file: {e}")
        
        return ""
    
    def _read_spec_text(self) -> Optional[str]:
        """Return the instruction text: prefer override, else read file, else None."""
        if self.instruction_override:
            # Honour explicit overrides so tests and manual runs can inject free-form prompts.
            txt = self.instruction_override.strip()
            if txt:
                return txt

        if self.product_specification_file:
            try:
                cwd = os.getcwd()
                self.logger.debug(f"[Product] Current working directory: {cwd}")
                p = Path(self.product_specification_file)
                self.logger.debug(f"[Product] Attempting to open: {p.resolve()}")
                txt = p.read_text(encoding="utf-8").strip()
                if txt:
                    return txt
                self.logger.warning(f"[Product] Spec file is empty: {p}")
            except Exception as e:
                self.logger.exception(f"[Product] Failed to read spec: {e}")

        return None

    def _load_product_geometry(self, geometry_file: Optional[str]) -> Dict[str, Any]:
        """Load product geometry JSON, selecting the environment block (gazebo/real)."""
        if not geometry_file:
            return {}
        env = os.environ.get("ROBOT_ENV", "gazebo").strip().lower()
        try:
            p = Path(geometry_file)
            if not p.exists():
                self.logger.warning("[Product] Geometry file not found: %s", p)
                return {}
            raw = json.loads(p.read_text(encoding="utf-8"))
            geo = raw.get(env, {})
            if geo:
                self.logger.info(
                    "[Product] Loaded geometry for env='%s' from %s", env, p,
                )
                return geo

            self.logger.warning(
                "[Product] Geometry file %s has no usable '%s' block.", p, env
            )
            return {}
        except Exception:
            self.logger.exception("[Product] Failed to load geometry from %s", geometry_file)
            return {}

    def _geometry_for_part(self, part_name: str) -> Dict[str, Any]:
        """Extract placement geometry for a single part from loaded product geometry."""
        if not self.product_geometry:
            return {}
        board = self.product_geometry.get("assembly_board", {})
        parts = self.product_geometry.get("parts", {})
        slot_xy = board.get("slots", {}).get(part_name)
        if slot_xy is None:
            return {}
        return {
            "slot_xy": slot_xy,
            "part_height_m": parts.get("heights_m", {}).get(part_name),
            "model_name": parts.get("model_map", {}).get(part_name),
            "slot_floor_z_m": board.get("slot_floor_z_m"),
            "board_center": board.get("center", {}),
        }

    def _compose_task_msg(
        self,
        *,
        to: str,
        task_id: str,
        instruction: str | dict,
        phase_id: str | None = None,
        protocol: str = "plan/1.0",
    ) -> Message:
        """Create the structured SPADE Message that drives resource-agent task execution."""
        # Build a JSON payload with the instruction plus optional phase metadata.
        body = {
            "task_id": task_id,
            "instruction": instruction,
            "trace": _trace_with_timestamp({}, "product_dispatch_sent_at"),
        }
        if phase_id:
            body["phase_id"] = phase_id
        msg = Message(to=to)
        # Metadata drives the SPADE routing/filtering logic downstream.
        msg.set_metadata("type", "task")
        msg.set_metadata("protocol", protocol)
        msg.body = json.dumps(body)
        return msg

    def _build_product_state(self) -> Dict[str, Any]:
        """
        Build product-specific state for replanning context.

        NOTE: This only includes product/part state. System coordination state
        (robot states, running tasks, FSA states) is provided by CentralControllerAgent.

        Includes:
        - Part locations and states
        - Execution timeline
        - Requirements progress
        """
        requirements_status: Dict[str, Dict[str, Any]] = {}
        task_nodes = [n for n in self.process_planner.nodes if n.get("type") == "task"]

        # Seed from explicit requirement nodes when available.
        for node in self.process_planner.nodes:
            if node.get("type") != "requirement":
                continue
            req_id = node.get("id")
            if not req_id:
                continue
            requirements_status[str(req_id)] = {
                "goal": node.get("description") or node.get("raw_text", ""),
                "status": "unknown",
                "completion": 0,
            }

        # Ensure every requirement_id referenced by tasks is represented.
        req_ids_from_tasks = {
            str(n.get("requirement_id"))
            for n in task_nodes
            if n.get("requirement_id")
        }
        for req_id in req_ids_from_tasks:
            requirements_status.setdefault(
                req_id,
                {"goal": "", "status": "unknown", "completion": 0},
            )

        # Aggregate per-requirement progress from task statuses.
        for req_id in req_ids_from_tasks:
            req_tasks = [n for n in task_nodes if str(n.get("requirement_id")) == req_id]
            total_tasks = len(req_tasks)
            completed_tasks = sum(1 for n in req_tasks if n.get("status") == "completed")
            has_failed = any(
                isinstance(n.get("status"), str) and n.get("status", "").startswith("failed")
                for n in req_tasks
            )
            all_completed = bool(req_tasks) and all(n.get("status") == "completed" for n in req_tasks)

            if total_tasks > 0:
                requirements_status[req_id]["completion"] = int((completed_tasks / total_tasks) * 100)

            if has_failed:
                requirements_status[req_id]["status"] = "failed"
            elif all_completed:
                requirements_status[req_id]["status"] = "completed"
            elif completed_tasks > 0:
                requirements_status[req_id]["status"] = "in_progress"
            else:
                requirements_status[req_id]["status"] = "pending"

        final_timeline = [
            e for e in self.execution_timeline
            if e.get("status", "") in {"completed", "blocked"}
            or str(e.get("status", "")).startswith("failed:")
        ]
        return {
            "parts": dict(self.part_tracker),
            "execution_timeline": final_timeline,
            "requirements_status": requirements_status,
        }

    def _extract_requirement_text(self) -> Optional[str]:
        """
        Load requirement snippets from either an override path or the default
        specification/products/requirements/<product>.txt file.
        """
        candidates = [Path(self.product_specification_file)]
        for req_path in candidates:
            try:
                if not req_path.exists():
                    continue
                txt = req_path.read_text(encoding="utf-8").strip()
                if txt:
                    self.logger.info(f"[Product] Using requirement file: {req_path.resolve()}")
                    return txt
                self.logger.warning(f"[Product] Requirement file empty: {req_path}")
            except Exception as exc:
                self.logger.exception(
                    "[Product] Failed to read requirements from %s: %s",
                    req_path,
                    exc,
                )
        return None

    def _ensure_plan_result_inbox(self) -> None:
        """Register runtime plan_safety_result inbox exactly once."""
        if self._plan_result_inbox_registered:
            return
        t_plan_result = Template()
        t_plan_result.set_metadata("type", "plan_safety_result")
        self.add_behaviour(self._PlanSafetyResultInbox(), t_plan_result)
        self._plan_result_inbox_registered = True

    def _reactivate_blocked_tasks(
        self,
        *,
        candidate_task_ids: Optional[set[str]] = None,
    ) -> int:
        """
        Convert blocked tasks back to pending so they can be retried after
        replanning. If candidate_task_ids is provided, only reactivate those.
        """
        if candidate_task_ids is not None:
            candidate_task_ids = {str(tid) for tid in candidate_task_ids if tid}

        reactivated = 0
        for node in self.process_planner.nodes:
            if node.get("type") != "task":
                continue
            if node.get("status") != "blocked":
                continue

            node_id = str(node.get("id") or "")
            if candidate_task_ids is not None and node_id not in candidate_task_ids:
                continue

            node["status"] = "pending"
            reactivated += 1

        return reactivated

    def _handle_task_retry_ready(self, task_ids: Iterable[str]) -> int:
        """
        Requeue blocked tasks after CCA reports that a transient safety block
        has cleared and the task may be retried with a fresh safety_check.
        """
        candidate_task_ids = {
            str(task_id).strip()
            for task_id in (task_ids or [])
            if str(task_id).strip()
        }
        if not candidate_task_ids:
            return 0

        reactivated = self._reactivate_blocked_tasks(candidate_task_ids=candidate_task_ids)
        if not reactivated:
            return 0

        now_iso = datetime.now(timezone.utc).isoformat()
        for task_id in candidate_task_ids:
            self.task_states[task_id] = "pending"
            self.execution_timeline.append({
                "timestamp": now_iso,
                "task_id": task_id,
                "status": "requeued",
                "resource_jid": str(self.cca_jid),
            })

        return reactivated

    def _candidate_task_ids_from_violations(
        self,
        violations: list[dict[str, Any]] | None,
    ) -> set[str]:
        candidate_ids: set[str] = set()
        for violation in violations or []:
            if not isinstance(violation, dict):
                continue
            for key in ("failed_task_id", "task_id"):
                task_id = violation.get(key)
                if task_id:
                    candidate_ids.add(str(task_id))
            for key in ("affected_task_ids", "blocked_task_ids", "unreachable_task_ids"):
                values = violation.get(key)
                if isinstance(values, (list, tuple, set)):
                    candidate_ids.update(str(value) for value in values if value)
        return candidate_ids

    def _active_bridge_sequence(self) -> dict[str, Any] | None:
        payload = self.runtime_recovery.get("active_bridge_sequence")
        return deepcopy(payload) if isinstance(payload, dict) else None

    @staticmethod
    def _bridge_used_llm(
        active_bridge_sequence: dict[str, Any] | None,
        runtime_recovery: dict[str, Any] | None = None,
    ) -> bool:
        if isinstance(active_bridge_sequence, dict) and "used_llm_bridge" in active_bridge_sequence:
            return bool(active_bridge_sequence.get("used_llm_bridge", False))
        if isinstance(runtime_recovery, dict):
            return bool(runtime_recovery.get("used_llm_bridge", False))
        return False

    @staticmethod
    def _bridge_execution_policy(active_bridge_sequence: dict[str, Any] | None) -> dict[str, Any]:
        payload = (
            active_bridge_sequence.get("execution_policy")
            if isinstance(active_bridge_sequence, dict)
            else {}
        )
        return deepcopy(payload) if isinstance(payload, dict) else {}

    def _bridge_requires_complete_full_tail(
        self,
        active_bridge_sequence: dict[str, Any] | None,
    ) -> bool:
        return bool(self._bridge_execution_policy(active_bridge_sequence).get("complete_full_tail"))

    def _bridge_task_debug_rows(self, task_ids: list[str] | None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        lookup = {
            str(node.get("id", "")).strip(): node
            for node in self.process_planner.nodes
            if isinstance(node, dict) and str(node.get("id", "")).strip()
        }
        for task_id in task_ids or []:
            task_key = str(task_id or "").strip()
            if not task_key:
                continue
            node = lookup.get(task_key)
            if not isinstance(node, dict):
                rows.append({"id": task_key, "missing": True})
                continue
            rows.append(
                {
                    "id": task_key,
                    "function_name": str(node.get("function_name", "")).strip(),
                    "resource_jid": str(node.get("resource_jid", "")).strip(),
                    "status": str(node.get("status", "")).strip(),
                    "predecessors": list(node.get("predecessors") or []),
                    "successors": list(node.get("successors") or []),
                    "params": deepcopy(node.get("params") or {}),
                    "bridge_sequence_id": str(node.get("bridge_sequence_id", "")).strip(),
                    "bridge_sequence_index": int(node.get("bridge_sequence_index") or 0),
                    "bridge_sequence_length": int(node.get("bridge_sequence_length") or 0),
                    "primary_obligation": deepcopy(node.get("primary_obligation") or {}),
                    "projected_snapshot": deepcopy(node.get("projected_snapshot") or {}),
                    "projected_part_entry": deepcopy(node.get("projected_part_entry") or {}),
                    "change_reason": str(node.get("change_reason", "")).strip(),
                }
            )
        return rows

    def _bridge_sequence_tail_task_ids(
        self,
        *,
        bridge_sequence_id: str,
        completed_task_id: str,
    ) -> list[str]:
        completed_task_id = str(completed_task_id or "").strip()
        current_node = self.process_planner._find_node(completed_task_id)
        cutoff_index = int(current_node.get("bridge_sequence_index") or 0) if current_node else 0
        tail_ids: list[str] = []
        for node in self.process_planner._bridge_sequence_nodes(bridge_sequence_id):
            node_id = str(node.get("id", "")).strip()
            sequence_index = int(node.get("bridge_sequence_index") or 0)
            if not node_id or node_id == completed_task_id:
                continue
            if cutoff_index and sequence_index <= cutoff_index:
                continue
            tail_ids.append(node_id)
        return tail_ids

    def _refresh_bridge_snapshot(self, resource_jid: str) -> dict[str, Any] | None:
        resource = self.process_planner._resource_by_jid(resource_jid)
        if resource is None or not hasattr(resource, "get_bridge_snapshot"):
            return None
        try:
            snapshot = resource.get_bridge_snapshot()
        except Exception:
            self.logger.exception(
                "[Product] Failed to refresh bridge snapshot for %s.",
                resource_jid,
            )
            return None
        return dict(snapshot) if isinstance(snapshot, dict) else None

    def _system_coordination_state_with_bridge_snapshot(
        self,
        *,
        base_state: dict[str, Any] | None,
        resource_jid: str,
        bridge_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        from cais_spade_llm.resources.resource_profile import (
            get_resource_profile,
            resource_snapshot_field_value,
            resource_snapshot_has_field,
        )

        system_state = deepcopy(base_state or {})
        resource_states = dict(
            self.process_planner._extract_resource_states(system_state)
        )
        resource_entry = dict(resource_states.get(resource_jid) or {})
        resource_core = dict(bridge_snapshot.get("resource_core") or {})
        resource_facets = dict(bridge_snapshot.get("resource_facets") or {})
        resource_type = str(
            resource_core.get("resource_type") or bridge_snapshot.get("resource_type") or ""
        ).strip().lower()
        profile = get_resource_profile(resource_type or "resource")
        update: dict[str, Any] = {
            "resource_type": resource_type or None,
            "current_state": resource_core.get("current_state") or bridge_snapshot.get("current_state"),
            "current_location": (
                resource_core.get("current_location")
                if resource_core.get("current_location") is not None
                else bridge_snapshot.get("current_location")
                if bridge_snapshot.get("current_location") is not None
                else resource_snapshot_field_value(
                    bridge_snapshot,
                    "current_pose_ref",
                    profile=profile,
                )
            ),
            "active_work": resource_core.get("active_work") or bridge_snapshot.get("active_work"),
        }
        for field in profile.snapshot_fields:
            if field == "current_state":
                continue
            if not resource_snapshot_has_field(
                bridge_snapshot,
                field,
                profile=profile,
            ):
                continue
            value = resource_snapshot_field_value(
                bridge_snapshot,
                field,
                profile=profile,
            )
            update[field] = value
        if resource_snapshot_has_field(bridge_snapshot, "current_pose_ref", profile=profile):
            update["current_pose_ref"] = resource_snapshot_field_value(
                bridge_snapshot,
                "current_pose_ref",
                profile=profile,
            )
        resource_entry.update(update)
        resource_states[resource_jid] = resource_entry
        system_state["resource_states"] = resource_states
        return system_state

    @staticmethod
    def _bridge_snapshot_mismatch(
        *,
        actual_snapshot: dict[str, Any],
        projected_snapshot: dict[str, Any],
    ) -> str:
        from cais_spade_llm.resources.resource_profile import (
            get_resource_profile,
            resource_snapshot_field_value,
        )

        resource_type = str(
            projected_snapshot.get("resource_type")
            or dict(projected_snapshot.get("resource_core") or {}).get("resource_type")
            or actual_snapshot.get("resource_type")
            or dict(actual_snapshot.get("resource_core") or {}).get("resource_type")
            or "resource"
        ).strip().lower() or "resource"
        profile = get_resource_profile(resource_type)
        comparable_keys = ["current_state", "current_location", "active_work"]
        comparable_keys.extend(
            key
            for key in profile.snapshot_fields
            if key not in comparable_keys
        )
        for extra_key in ("current_pose_ref",):
            if extra_key in projected_snapshot and extra_key not in comparable_keys:
                comparable_keys.append(extra_key)
        for key in comparable_keys:
            if key not in projected_snapshot:
                continue
            actual_value = resource_snapshot_field_value(actual_snapshot, key, profile=profile)
            projected_value = resource_snapshot_field_value(projected_snapshot, key, profile=profile)
            if actual_value != projected_value:
                return (
                    f"projected {key}={projected_value!r} "
                    f"but runtime observed {actual_value!r}"
                )
        return ""

    def _bridge_part_entry_mismatch(
        self,
        *,
        part_name: str,
        projected_part_entry: dict[str, Any],
    ) -> str:
        actual_entry = dict(self.part_tracker.get(part_name) or {})
        for key in ("state", "location"):
            if key not in projected_part_entry:
                continue
            if actual_entry.get(key) != projected_part_entry.get(key):
                return (
                    f"projected part {part_name}.{key}={projected_part_entry.get(key)!r} "
                    f"but runtime tracker has {actual_entry.get(key)!r}"
                )
        return ""

    def _dispatch_runtime_plan_validation_check(
        self, *, skip_revalidation: bool = False,
    ) -> None:
        # Always recompile the FSA so that newly-inserted tasks (e.g.,
        # recovery bridge macros) are present in the transition table.
        # skip_revalidation only skips the CCA-side safety-rule check.
        self.process_planner.compile_global_fsa()
        self.process_planner.save_global_fsa(self.global_fsa_path)

        payload = self._build_plan_validation_payload(
            skip_revalidation=skip_revalidation,
        )
        msg_check = Message(to=self.cca_jid)
        msg_check.set_metadata("type", "plan_safety_check")
        msg_check.body = json.dumps(payload)

        def _dispatch() -> None:
            self._ensure_plan_result_inbox()
            self._dispatch_agent_message_sync(
                msg_check,
                trace_category="ProductAgent/_dispatch_runtime_plan_validation_check",
            )

        self._run_callable_on_agent_loop_sync(
            _dispatch,
            timeout_sec=10.0,
            operation_name="runtime plan validation dispatch",
        )

        self.logger.info(
            "[Product] Recompiled plan FSA after runtime recovery and sent plan_safety_check to CCA."
        )

    async def _send_runtime_plan_validation_check(
        self, *, skip_revalidation: bool = False,
    ) -> None:
        self._dispatch_runtime_plan_validation_check(
            skip_revalidation=skip_revalidation,
        )

    def _send_runtime_plan_validation_check_sync(
        self, *, skip_revalidation: bool = False,
    ) -> None:
        self._dispatch_runtime_plan_validation_check(
            skip_revalidation=skip_revalidation,
        )

    async def _fail_closed_bridge_sequence(
        self,
        *,
        task_node: dict[str, Any],
        status: str,
        message: str,
        active_bridge_sequence: dict[str, Any],
        content: str = "",
        observations: dict[str, Any] | None = None,
    ) -> None:
        sequence = deepcopy(active_bridge_sequence)
        sequence_id = str(sequence.get("bridge_sequence_id", "")).strip()
        if sequence_id:
            deletions = self.process_planner.remove_bridge_sequence_tail(
                bridge_sequence_id=sequence_id,
                completed_task_id=str(task_node.get("id", "")).strip(),
            )
            if deletions:
                sequence["trimmed_tail_task_ids"] = [
                    str(item.get("id", "")).strip()
                    for item in deletions
                    if str(item.get("id", "")).strip()
                ]
        sequence["state"] = "failed"
        sequence["last_task_id"] = str(task_node.get("id", "")).strip()
        sequence["last_status"] = str(status or "").strip()
        if content:
            sequence["last_content"] = str(content).strip()
        if isinstance(observations, dict) and observations:
            sequence["last_observations"] = deepcopy(observations)

        violations = deepcopy(list(sequence.get("violations") or []))
        trigger = str(sequence.get("trigger", "")).strip()
        failed_task_id = str(sequence.get("failed_task_id", "")).strip()
        used_llm_bridge = self._bridge_used_llm(sequence, self.runtime_recovery)
        self._runtime_recovery_context = {}
        self._set_runtime_recovery(
            status="human_required",
            resolution_class="human_required",
            trigger=trigger,
            failed_task_id=failed_task_id,
            message=message,
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=used_llm_bridge,
            bridge_proposal=None,
            bridge_approval_state="approved",
            active_bridge_sequence=sequence,
            bridge_feedback_history=self.runtime_recovery.get("bridge_feedback_history") or [],
            violations=violations,
            append_history=True,
            history_message=message,
        )
        self._set_plan_safety_alert(
            stage="runtime",
            message=message,
            retries_used=self._runtime_repair_fail_streak,
            retries_max=self._runtime_repair_max_attempts,
            violations=violations,
            paused=True,
        )
        await asyncio.to_thread(self._persist_plan_snapshot)
        await asyncio.to_thread(self._persist_product_state)
        await asyncio.to_thread(self._persist_resource_state)

    async def _handle_bridge_macro_ack(
        self,
        *,
        task_node: dict[str, Any],
        status: str,
        content: str = "",
        observations: dict[str, Any] | None = None,
    ) -> bool:
        if str(task_node.get("function_name", "")).strip() != "execute_recovery_macro":
            return False

        active_bridge_sequence = self._active_bridge_sequence()
        if not active_bridge_sequence:
            return False

        sequence_id = str(task_node.get("bridge_sequence_id", "")).strip()
        if not sequence_id or sequence_id != str(active_bridge_sequence.get("bridge_sequence_id", "")).strip():
            return False

        macro_name = str((task_node.get("params") or {}).get("macro_name") or task_node.get("id") or "bridge_macro").strip()
        failed_task_id = str(
            active_bridge_sequence.get("failed_task_id")
            or self.runtime_recovery.get("failed_task_id", "")
        ).strip()
        trigger = str(active_bridge_sequence.get("trigger", "")).strip()
        violations = deepcopy(list(active_bridge_sequence.get("violations") or []))
        feedback_history = self.runtime_recovery.get("bridge_feedback_history") or []
        used_llm_bridge = self._bridge_used_llm(active_bridge_sequence, self.runtime_recovery)

        if isinstance(status, str) and status.startswith("failed"):
            detail = str(content or "").strip()
            if not detail and isinstance(observations, dict):
                detail = json.dumps(observations, sort_keys=True, default=str)
            message = (
                f"{self.agent_name}: bridge macro '{macro_name}' failed during execution"
                + (f" ({detail})." if detail else ".")
                + " Human intervention required."
            )
            await self._fail_closed_bridge_sequence(
                task_node=task_node,
                status=status,
                message=message,
                active_bridge_sequence=active_bridge_sequence,
                content=content,
                observations=observations,
            )
            return True

        if str(status).strip().lower() != "completed":
            return False

        resource_jid = str(task_node.get("resource_jid", "")).strip()
        actual_snapshot = self._refresh_bridge_snapshot(resource_jid)
        if not isinstance(actual_snapshot, dict):
            message = (
                f"{self.agent_name}: bridge macro '{macro_name}' completed but the runtime "
                f"bridge snapshot for {resource_jid} could not be refreshed. Human intervention required."
            )
            await self._fail_closed_bridge_sequence(
                task_node=task_node,
                status=status,
                message=message,
                active_bridge_sequence=active_bridge_sequence,
                content=content,
                observations=observations,
            )
            return True

        projected_snapshot = dict(task_node.get("projected_snapshot") or {})
        mismatch = ""
        if projected_snapshot:
            mismatch = self._bridge_snapshot_mismatch(
                actual_snapshot=actual_snapshot,
                projected_snapshot=projected_snapshot,
            )
        part_name = str(self._tracked_part_name_for_task(task_node) or "").strip()
        projected_part_entry = dict(task_node.get("projected_part_entry") or {})
        if not mismatch and part_name and projected_part_entry:
            mismatch = self._bridge_part_entry_mismatch(
                part_name=part_name,
                projected_part_entry=projected_part_entry,
            )
        if mismatch:
            divergence = dict(observations or {})
            divergence["runtime_snapshot"] = deepcopy(actual_snapshot)
            divergence["projected_snapshot"] = deepcopy(projected_snapshot)
            if projected_part_entry:
                divergence["projected_part_entry"] = deepcopy(projected_part_entry)
                divergence["actual_part_entry"] = deepcopy(self.part_tracker.get(part_name) or {})
            message = (
                f"{self.agent_name}: bridge macro '{macro_name}' diverged from its approved "
                f"projected post-state ({mismatch}). Human intervention required."
            )
            await self._fail_closed_bridge_sequence(
                task_node=task_node,
                status=status,
                message=message,
                active_bridge_sequence=active_bridge_sequence,
                content=content,
                observations=divergence,
            )
            return True

        tail_task_ids = self._bridge_sequence_tail_task_ids(
            bridge_sequence_id=sequence_id,
            completed_task_id=str(task_node.get("id", "")).strip(),
        )
        refreshed_system_state = self._system_coordination_state_with_bridge_snapshot(
            base_state=active_bridge_sequence.get("system_coordination_state") or {},
            resource_jid=resource_jid,
            bridge_snapshot=actual_snapshot,
        )
        plan_rewrite = dict(active_bridge_sequence.get("plan_rewrite") or {})
        if tail_task_ids and self._bridge_requires_complete_full_tail(active_bridge_sequence):
            next_sequence = deepcopy(active_bridge_sequence)
            next_sequence["state"] = "executing"
            next_sequence["last_task_id"] = str(task_node.get("id", "")).strip()
            next_sequence["last_completed_macro_name"] = macro_name
            next_sequence["system_coordination_state"] = deepcopy(refreshed_system_state)
            continue_message = (
                f"Bridge macro '{macro_name}' matched projection; continuing the approved bridge tail."
            )
            self._set_runtime_recovery(
                status="resolved",
                resolution_class="des_with_llm_bridge",
                trigger=trigger,
                failed_task_id=failed_task_id,
                message=continue_message,
                attempts_used=self._runtime_repair_fail_streak,
                attempts_max=self._runtime_repair_max_attempts,
                used_llm_bridge=used_llm_bridge,
                bridge_proposal=None,
                bridge_approval_state="approved",
                active_bridge_sequence=next_sequence,
                bridge_feedback_history=feedback_history,
                violations=[],
                append_history=True,
                history_message=continue_message,
            )
            await asyncio.to_thread(self._persist_product_state)
            return True

        if bool(plan_rewrite.get("replace_failed_branch")):
            self._runtime_recovery_context = {
                "trigger": trigger,
                "failed_task_id": failed_task_id,
                "violations": deepcopy(violations),
                "system_coordination_state": deepcopy(refreshed_system_state),
                "bridge_feedback_history": list(feedback_history),
            }
            validation_message = (
                f"Bridge macro '{macro_name}' completed the approved branch replacement; "
                "validating updated plan."
            )
            self._set_runtime_recovery(
                status="validating",
                resolution_class="none",
                trigger=trigger,
                failed_task_id=failed_task_id,
                message=validation_message,
                attempts_used=self._runtime_repair_fail_streak,
                attempts_max=self._runtime_repair_max_attempts,
                used_llm_bridge=used_llm_bridge,
                bridge_proposal=None,
                bridge_approval_state="approved",
                active_bridge_sequence=None,
                bridge_feedback_history=feedback_history,
                violations=violations,
                append_history=True,
                history_message=validation_message,
            )
            self._clear_plan_safety_alert()
            await self._send_runtime_plan_validation_check()
            await asyncio.to_thread(self._persist_plan_snapshot)
            await asyncio.to_thread(self._persist_product_state)
            await asyncio.to_thread(self._persist_resource_state)
            return True

        if failed_task_id and self.process_planner.can_execute_task_from_system_state(
            task_id=failed_task_id,
            system_coordination_state=refreshed_system_state,
            part_tracker=self.part_tracker,
        ):
            if tail_task_ids:
                self.process_planner.remove_bridge_sequence_tail(
                    bridge_sequence_id=sequence_id,
                    completed_task_id=str(task_node.get("id", "")).strip(),
                )
            reactivated = self._reactivate_blocked_tasks(
                candidate_task_ids=self._candidate_task_ids_from_violations(violations) or None
            )
            if reactivated:
                self.logger.info(
                    "[Product] Reactivated %d blocked task(s) after bridge macro %s restored direct executability.",
                    reactivated,
                    macro_name,
                )
            self._runtime_recovery_context = {
                "trigger": trigger,
                "failed_task_id": failed_task_id,
                "violations": deepcopy(violations),
                "system_coordination_state": deepcopy(refreshed_system_state),
                "bridge_feedback_history": list(feedback_history),
            }
            validation_message = (
                f"Bridge macro '{macro_name}' restored direct task executability; validating updated plan."
            )
            self._set_runtime_recovery(
                status="validating",
                resolution_class="none",
                trigger=trigger,
                failed_task_id=failed_task_id,
                message=validation_message,
                attempts_used=self._runtime_repair_fail_streak,
                attempts_max=self._runtime_repair_max_attempts,
                used_llm_bridge=used_llm_bridge,
                bridge_proposal=None,
                bridge_approval_state="approved",
                active_bridge_sequence=None,
                bridge_feedback_history=feedback_history,
                violations=violations,
                append_history=True,
                history_message=validation_message,
            )
            self._clear_plan_safety_alert()
            await self._send_runtime_plan_validation_check()
            await asyncio.to_thread(self._persist_plan_snapshot)
            await asyncio.to_thread(self._persist_product_state)
            await asyncio.to_thread(self._persist_resource_state)
            return True

        des_result = await self.process_planner.replan_with_feedback_des(
            violations,
            system_coordination_state=refreshed_system_state,
            allow_bridge_fallback=False,
            ignored_task_ids=set(tail_task_ids),
        )
        if bool(des_result.get("plan_changed", False)):
            if tail_task_ids:
                self.process_planner.remove_bridge_sequence_tail(
                    bridge_sequence_id=sequence_id,
                    completed_task_id=str(task_node.get("id", "")).strip(),
                )
            reactivated = self._reactivate_blocked_tasks(
                candidate_task_ids=self._candidate_task_ids_from_violations(violations) or None
            )
            if reactivated:
                self.logger.info(
                    "[Product] Reactivated %d blocked task(s) after DES resumed from bridge macro %s.",
                    reactivated,
                    macro_name,
                )
            self._runtime_recovery_context = {
                "trigger": trigger,
                "failed_task_id": failed_task_id,
                "violations": deepcopy(violations),
                "system_coordination_state": deepcopy(refreshed_system_state),
                "bridge_feedback_history": list(feedback_history),
            }
            validation_message = (
                f"Bridge macro '{macro_name}' restored a DES continuation; validating updated plan."
            )
            self._set_runtime_recovery(
                status="validating",
                resolution_class="none",
                trigger=trigger,
                failed_task_id=failed_task_id,
                message=validation_message,
                attempts_used=self._runtime_repair_fail_streak,
                attempts_max=self._runtime_repair_max_attempts,
                used_llm_bridge=used_llm_bridge,
                bridge_proposal=None,
                bridge_approval_state="approved",
                active_bridge_sequence=None,
                bridge_feedback_history=feedback_history,
                violations=violations,
                append_history=True,
                history_message=validation_message,
            )
            self._clear_plan_safety_alert()
            await self._send_runtime_plan_validation_check()
            await asyncio.to_thread(self._persist_plan_snapshot)
            await asyncio.to_thread(self._persist_product_state)
            await asyncio.to_thread(self._persist_resource_state)
            return True

        if bool(des_result.get("des_recovery_missing", False)) and tail_task_ids:
            next_sequence = deepcopy(active_bridge_sequence)
            next_sequence["state"] = "executing"
            next_sequence["last_task_id"] = str(task_node.get("id", "")).strip()
            next_sequence["last_completed_macro_name"] = macro_name
            next_sequence["system_coordination_state"] = deepcopy(refreshed_system_state)
            continue_message = (
                f"Bridge macro '{macro_name}' matched projection; continuing the approved bridge tail."
            )
            self._set_runtime_recovery(
                status="resolved",
                resolution_class="des_with_llm_bridge",
                trigger=trigger,
                failed_task_id=failed_task_id,
                message=continue_message,
                attempts_used=self._runtime_repair_fail_streak,
                attempts_max=self._runtime_repair_max_attempts,
                used_llm_bridge=used_llm_bridge,
                bridge_proposal=None,
                bridge_approval_state="approved",
                active_bridge_sequence=next_sequence,
                bridge_feedback_history=feedback_history,
                violations=[],
                append_history=True,
                history_message=continue_message,
            )
            await asyncio.to_thread(self._persist_product_state)
            return True

        if bool(des_result.get("des_recovery_missing", False)):
            message = (
                f"{self.agent_name}: final bridge macro '{macro_name}' completed as approved, "
                "but DES still found no valid continuation. Human intervention required."
            )
            await self._fail_closed_bridge_sequence(
                task_node=task_node,
                status=status,
                message=message,
                active_bridge_sequence=active_bridge_sequence,
                content=content,
                observations=observations,
            )
            return True

        fallback_message = str(des_result.get("message", "")).strip() or (
            f"{self.agent_name}: bridge macro '{macro_name}' completed, but runtime DES handoff failed."
        )
        await self._fail_closed_bridge_sequence(
            task_node=task_node,
            status=status,
            message=fallback_message,
            active_bridge_sequence=active_bridge_sequence,
            content=content,
            observations=observations,
        )
        return True

    async def _run_des_runtime_recovery_attempt(
        self,
        *,
        violations: list[dict[str, Any]],
        trigger: str,
        failed_task_id: str,
        system_coordination_state: dict | None = None,
        reset_attempts: bool = False,
        history_message: str | None = None,
        bridge_feedback: str = "",
    ) -> dict[str, Any]:
        if reset_attempts:
            self._runtime_repair_fail_streak = 0

        attempt_number = self._runtime_repair_fail_streak + 1
        self._runtime_repair_fail_streak = attempt_number
        feedback_history = [
            str(item).strip()
            for item in (self.runtime_recovery.get("bridge_feedback_history") or [])
            if str(item).strip()
        ]
        feedback_text = str(bridge_feedback or "").strip()
        if feedback_text:
            feedback_history.append(feedback_text)
        self._runtime_recovery_context = {
            "trigger": str(trigger or "").strip(),
            "failed_task_id": str(failed_task_id or "").strip(),
            "violations": deepcopy(list(violations or [])),
            "system_coordination_state": deepcopy(system_coordination_state or {}),
            "bridge_feedback_history": list(feedback_history),
        }
        self._set_runtime_recovery(
            status="des_search",
            resolution_class="none",
            trigger=trigger,
            failed_task_id=failed_task_id,
            message=(
                f"Running DES runtime recovery attempt "
                f"{attempt_number}/{self._runtime_repair_max_attempts}."
            ),
            attempts_used=attempt_number,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=False,
            bridge_proposal=None,
            bridge_debug=None,
            bridge_approval_state="none",
            active_bridge_sequence=None,
            bridge_feedback_history=feedback_history,
            violations=violations,
            append_history=True,
            history_message=history_message or (
                f"DES runtime recovery attempt {attempt_number}/{self._runtime_repair_max_attempts} started."
            ),
        )

        scenario_hint = canonical_preprogrammed_bridge_scenario_id(
            self._auto_runtime_preprogrammed_scenario_id()
        )
        bridge_generation_mode = str(self._bridge_generation_mode or "auto").strip().lower()
        if scenario_hint and bridge_generation_mode != "manual":
            self.logger.info(
                "[Product] Forcing manual bridge handoff for preprogrammed scenario: scenario_id=%s mode=%s->manual",
                scenario_hint,
                bridge_generation_mode or "auto",
            )
            bridge_generation_mode = "manual"

        self._runtime_repair_inflight = True
        try:
            result = await self.process_planner.replan_with_feedback_online(
                violations,
                system_coordination_state=system_coordination_state,
                bridge_feedback=feedback_text,
                bridge_generation_mode=bridge_generation_mode,
            )
            if not isinstance(result, dict):
                result = {}

            plan_changed = bool(result.get("plan_changed", False))
            used_llm_bridge = bool(result.get("used_llm_bridge", False))
            human_required = bool(result.get("human_required", False))
            awaiting_bridge_approval = bool(result.get("awaiting_bridge_approval", False))
            awaiting_bridge_generation = bool(result.get("awaiting_bridge_generation", False))
            base_message = str(result.get("message", "")).strip()
            bridge_summary = result.get("bridge_summary") or []
            bridge_proposal = result.get("bridge_proposal")
            bridge_debug = result.get("bridge_debug")
            prepared_bridge_request = result.get("prepared_bridge_request")
            if awaiting_bridge_generation and isinstance(prepared_bridge_request, dict):
                self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(
                    prepared_bridge_request
                )
                if hasattr(self.process_planner, "emit_prepare_trace_summary"):
                    try:
                        self.process_planner.emit_prepare_trace_summary(prepared_bridge_request)
                    except Exception:
                        self.logger.exception(
                            "[Product] Failed to emit prepare-trace summary for runtime bridge request."
                        )
                bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                active_bridge_phase = str(bridge_session.get("phase") or "").strip().lower()
                allow_preprogrammed_autoload = active_bridge_phase not in {"prepare_trace"}
                if scenario_hint and allow_preprogrammed_autoload:
                    self.logger.info(
                        "[Product] Auto-loading preprogrammed recovery scenario after DES handoff: scenario_id=%s",
                        scenario_hint,
                    )
                    try:
                        (
                            scenario_key,
                            normalized,
                            bridge_debug,
                            _plan_rewrite,
                            bridge_summary,
                        ) = self._resolve_preprogrammed_runtime_bridge_bundle(
                            scenario_id=scenario_hint,
                            prepared_bridge_request=prepared_bridge_request,
                        )
                        bridge_text = (
                            ", ".join(str(item) for item in bridge_summary if item)
                            or scenario_key
                        )
                        recovery = self._set_runtime_recovery(
                            status="llm_bridge",
                            trigger=trigger,
                            failed_task_id=failed_task_id,
                            message="Preprogrammed recovery scenario loaded; auto-approving.",
                            attempts_used=attempt_number,
                            attempts_max=self._runtime_repair_max_attempts,
                            used_llm_bridge=False,
                            bridge_proposal=normalized,
                            bridge_debug=bridge_debug,
                            bridge_approval_state="pending",
                            active_bridge_sequence=None,
                            bridge_feedback_history=feedback_history,
                            violations=violations,
                            append_history=True,
                            history_message=(
                                f"Automatically loaded preprogrammed recovery scenario: {bridge_text}."
                            ),
                        )
                        self._clear_plan_safety_alert()
                        self.logger.info(
                            "[Product] Auto-approving preprogrammed recovery scenario after DES handoff: scenario_id=%s",
                            scenario_key,
                        )
                        return self.approve_runtime_bridge_proposal_sync()
                    except Exception:
                        self.logger.exception(
                            "[Product] Auto-loading preprogrammed recovery scenario failed: scenario_id=%s",
                            scenario_hint,
                        )
                elif scenario_hint and not allow_preprogrammed_autoload:
                    self.logger.info(
                        "[Product] Active bridge mode left runtime recovery at prepare-trace checkpoint; "
                        "skipping preprogrammed auto-load for scenario_id=%s",
                        scenario_hint,
                    )
                message = (
                    base_message
                    or "DES found no modeled continuation. Bridge session is ready for LLM reasoning."
                )
                recovery = self._set_runtime_recovery(
                    status="bridge_ready",
                    trigger=trigger,
                    failed_task_id=failed_task_id,
                    message=message,
                    attempts_used=attempt_number,
                    attempts_max=self._runtime_repair_max_attempts,
                    used_llm_bridge=False,
                    bridge_proposal=None,
                    bridge_debug=bridge_debug,
                    bridge_approval_state="ready",
                    active_bridge_sequence=None,
                    bridge_feedback_history=feedback_history,
                    violations=violations,
                    append_history=True,
                    history_message=message,
                )
                self._record_runtime_bridge_artifacts(
                    phase="prepare",
                    prepared_bridge_request=prepared_bridge_request,
                )
                self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(
                    prepared_bridge_request
                )
                recovery = deepcopy(self.runtime_recovery)
                self._clear_plan_safety_alert()
                await asyncio.to_thread(self._persist_product_state)
                return recovery
            if awaiting_bridge_approval and isinstance(bridge_proposal, dict):
                bridge_text = ", ".join(str(item) for item in bridge_summary if item) or "bridge step(s)"
                recovery = self._set_runtime_recovery(
                    status="llm_bridge",
                    trigger=trigger,
                    failed_task_id=failed_task_id,
                    message=base_message or "Validated bridge proposal is ready for final approval.",
                    attempts_used=attempt_number,
                    attempts_max=self._runtime_repair_max_attempts,
                    used_llm_bridge=True,
                    bridge_proposal=bridge_proposal,
                    bridge_debug=bridge_debug,
                    bridge_approval_state="pending",
                    active_bridge_sequence=None,
                    bridge_feedback_history=feedback_history,
                    violations=violations,
                    append_history=True,
                    history_message=f"LLM bridge proposed {bridge_text}. Awaiting approval.",
                )
                self._clear_plan_safety_alert()
                await asyncio.to_thread(self._persist_product_state)
                return recovery

            if human_required or not plan_changed:
                message = (
                    base_message
                    or "DES recovery could not produce a valid continuation. Human intervention required."
                )
                recovery = self._set_runtime_recovery(
                    status="human_required",
                    resolution_class="human_required",
                    trigger=trigger,
                    failed_task_id=failed_task_id,
                    message=message,
                    attempts_used=attempt_number,
                    attempts_max=self._runtime_repair_max_attempts,
                    used_llm_bridge=used_llm_bridge,
                    bridge_proposal=None,
                    bridge_debug=bridge_debug,
                    bridge_approval_state="none",
                    active_bridge_sequence=None,
                    bridge_feedback_history=feedback_history,
                    violations=violations,
                    append_history=True,
                    history_message=message,
                )
                self._set_plan_safety_alert(
                    stage="runtime",
                    message=message,
                    retries_used=attempt_number,
                    retries_max=self._runtime_repair_max_attempts,
                    violations=violations,
                    paused=True,
                )
                await asyncio.to_thread(self._persist_product_state)
                return recovery

            reactivated = self._reactivate_blocked_tasks(
                candidate_task_ids=self._candidate_task_ids_from_violations(violations) or None
            )
            if reactivated:
                self.logger.info(
                    "[Product] Reactivated %d blocked task(s) to pending after DES recovery.",
                    reactivated,
                )

            validation_message = (
                "DES recovery candidate generated; validating updated plan."
                if not used_llm_bridge
                else "DES + LLM bridge candidate generated; validating updated plan."
            )
            recovery = self._set_runtime_recovery(
                status="validating",
                resolution_class="none",
                trigger=trigger,
                failed_task_id=failed_task_id,
                message=validation_message,
                attempts_used=attempt_number,
                attempts_max=self._runtime_repair_max_attempts,
                used_llm_bridge=used_llm_bridge,
                bridge_proposal=None,
                bridge_debug=bridge_debug if used_llm_bridge else None,
                bridge_approval_state="approved" if used_llm_bridge else "none",
                active_bridge_sequence=None,
                bridge_feedback_history=feedback_history,
                violations=violations,
                append_history=True,
                history_message=validation_message,
            )
            self._clear_plan_safety_alert()
            await self._send_runtime_plan_validation_check()
            await asyncio.to_thread(self._persist_plan_snapshot)
            await asyncio.to_thread(self._persist_product_state)
            await asyncio.to_thread(self._persist_resource_state)
            return recovery
        except Exception as exc:
            self.logger.exception("[Product] DES runtime recovery attempt failed.")
            message = (
                f"{self.agent_name}: DES runtime recovery attempt "
                f"{attempt_number}/{self._runtime_repair_max_attempts} failed ({exc})."
            )
            recovery = self._set_runtime_recovery(
                status="human_required",
                resolution_class="human_required",
                trigger=trigger,
                failed_task_id=failed_task_id,
                message=message,
                attempts_used=attempt_number,
                attempts_max=self._runtime_repair_max_attempts,
                bridge_proposal=None,
                bridge_approval_state="none",
                active_bridge_sequence=None,
                bridge_feedback_history=feedback_history,
                violations=violations,
                append_history=True,
                history_message=message,
            )
            self._set_plan_safety_alert(
                stage="runtime",
                message=message,
                retries_used=attempt_number,
                retries_max=self._runtime_repair_max_attempts,
                violations=violations,
                paused=True,
            )
            await asyncio.to_thread(self._persist_product_state)
            return recovery
        finally:
            self._runtime_repair_inflight = False

    async def _handle_runtime_des_replan_request(
        self,
        *,
        reason: str,
        failed_task_id: str,
        violations: list[dict[str, Any]],
        system_coordination_state: dict | None = None,
    ) -> dict[str, Any]:
        current_status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        if current_status in {"des_search", "bridge_ready", "llm_bridge", "validating", "human_required"}:
            self.logger.warning(
                "[Product] Runtime recovery already active for %s; ignoring duplicate replan request.",
                self.runtime_recovery.get("failed_task_id") or failed_task_id,
            )
            return self.get_runtime_recovery()

        self._clear_plan_safety_alert()
        self._set_runtime_recovery(
            reset=True,
            status="des_search",
            resolution_class="none",
            trigger=reason,
            failed_task_id=failed_task_id,
            message=f"Runtime DES recovery triggered by {reason}.",
            attempts_used=0,
            attempts_max=self._runtime_repair_max_attempts,
            bridge_proposal=None,
            bridge_approval_state="none",
            active_bridge_sequence=None,
            bridge_feedback_history=[],
            violations=violations,
            append_history=True,
            history_message=f"Runtime DES recovery triggered by {reason}.",
        )
        return await self._run_des_runtime_recovery_attempt(
            violations=violations,
            trigger=reason,
            failed_task_id=failed_task_id,
            system_coordination_state=system_coordination_state,
            reset_attempts=True,
        )

    async def _handle_runtime_plan_validation_result(
        self,
        *,
        ok: bool,
        violations: list[dict[str, Any]],
    ) -> bool:
        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        if not self._runtime_recovery_context and status in {"", "idle", "resolved"}:
            return False

        if ok:
            active_bridge_sequence = self._active_bridge_sequence()
            resolution_class = (
                "des_with_llm_bridge"
                if bool(self.runtime_recovery.get("used_llm_bridge", False))
                else "des_only"
            )
            attempts_used = self._runtime_repair_fail_streak
            self._runtime_repair_fail_streak = 0
            success_message = (
                "Plan validation passed; approved bridge sequence is executing."
                if active_bridge_sequence
                else "Plan validation passed; runtime recovery resolved."
            )
            self.logger.info(
                "[Product] Runtime plan validation passed: active_bridge_sequence=%s status_before=%s",
                bool(active_bridge_sequence),
                status,
            )
            self._set_runtime_recovery(
                status="resolved",
                resolution_class=resolution_class,
                message=success_message,
                attempts_used=attempts_used,
                used_llm_bridge=bool(self.runtime_recovery.get("used_llm_bridge", False)),
                bridge_proposal=None,
                bridge_approval_state=(
                    "approved"
                    if bool(self.runtime_recovery.get("used_llm_bridge", False))
                    else "none"
                ),
                active_bridge_sequence=active_bridge_sequence,
                violations=[],
                append_history=True,
                history_message=success_message,
            )
            self._runtime_recovery_context = {}
            self._clear_plan_safety_alert()
            await asyncio.to_thread(self._persist_product_state)
            return True

        self.logger.warning(
            "[Product] Runtime plan validation failed: status=%s violations=%d",
            status,
            len(violations),
        )
        if self._runtime_repair_inflight:
            self.logger.warning(
                "[Product] Runtime plan validation failed while DES recovery is already running; ignoring duplicate result."
            )
            return True

        if not self._runtime_recovery_context:
            message = (
                f"{self.agent_name}: plan validation failed without an active DES recovery context; "
                "execution paused for human intervention."
            )
            self._set_runtime_recovery(
                reset=True,
                status="human_required",
                resolution_class="human_required",
                message=message,
                attempts_used=self._runtime_repair_fail_streak,
                attempts_max=self._runtime_repair_max_attempts,
                bridge_proposal=None,
                bridge_approval_state="none",
                active_bridge_sequence=None,
                violations=violations,
                append_history=True,
                history_message=message,
            )
            self._set_plan_safety_alert(
                stage="runtime",
                message=message,
                retries_used=self._runtime_repair_fail_streak,
                retries_max=self._runtime_repair_max_attempts,
                violations=violations,
                paused=True,
            )
            await asyncio.to_thread(self._persist_product_state)
            return True

        if self._runtime_repair_fail_streak >= self._runtime_repair_max_attempts:
            message = (
                f"{self.agent_name}: runtime plan validation still fails after "
                f"{self._runtime_repair_fail_streak}/{self._runtime_repair_max_attempts} "
                "DES recovery attempt(s); execution paused."
            )
            self._set_runtime_recovery(
                status="human_required",
                resolution_class="human_required",
                trigger=str(self._runtime_recovery_context.get("trigger", "")),
                failed_task_id=str(self._runtime_recovery_context.get("failed_task_id", "")),
                message=message,
                attempts_used=self._runtime_repair_fail_streak,
                attempts_max=self._runtime_repair_max_attempts,
                used_llm_bridge=bool(self.runtime_recovery.get("used_llm_bridge", False)),
                bridge_proposal=None,
                bridge_approval_state="none",
                active_bridge_sequence=None,
                violations=violations,
                append_history=True,
                history_message=message,
            )
            self._set_plan_safety_alert(
                stage="runtime",
                message=message,
                retries_used=self._runtime_repair_fail_streak,
                retries_max=self._runtime_repair_max_attempts,
                violations=violations,
                paused=True,
            )
            await asyncio.to_thread(self._persist_product_state)
            return True

        self._runtime_recovery_context["violations"] = deepcopy(list(violations or []))
        await self._run_des_runtime_recovery_attempt(
            violations=violations,
            trigger=str(self._runtime_recovery_context.get("trigger", "")),
            failed_task_id=str(self._runtime_recovery_context.get("failed_task_id", "")),
            system_coordination_state=dict(self._runtime_recovery_context.get("system_coordination_state") or {}),
            reset_attempts=False,
            history_message=(
                f"Plan validation failed; rerunning DES recovery attempt "
                f"{self._runtime_repair_fail_streak + 1}/{self._runtime_repair_max_attempts}."
            ),
        )
        return True

    async def submit_runtime_recovery_guidance(self, message: str) -> dict[str, Any]:
        guidance = str(message or "").strip()
        if not guidance:
            raise ValueError("operator guidance is empty")
        feedback_history = [
            str(item).strip()
            for item in (self.runtime_recovery.get("bridge_feedback_history") or [])
            if str(item).strip()
        ]
        feedback_history.append(guidance)
        if self._runtime_recovery_context:
            self._runtime_recovery_context["bridge_feedback_history"] = list(feedback_history)
            prepared_bridge_request = deepcopy(
                self._runtime_recovery_context.get("prepared_bridge_request") or {}
            )
            if prepared_bridge_request:
                bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                operator_feedback_history = [
                    str(item).strip()
                    for item in (bridge_session.get("operator_feedback_history") or [])
                    if str(item).strip()
                ]
                operator_feedback_history.append(guidance)
                bridge_session["operator_feedback_history"] = operator_feedback_history[-12:]
                prepared_bridge_request["bridge_session"] = bridge_session
                prepared_bridge_request["bridge_feedback"] = guidance
                self._runtime_recovery_context["prepared_bridge_request"] = prepared_bridge_request
        recovery = self._set_runtime_recovery(
            operator_guidance=guidance,
            bridge_feedback_history=feedback_history,
            append_history=True,
            history_message=f"Operator guidance recorded: {guidance}",
        )
        await asyncio.to_thread(self._persist_product_state)
        return recovery

    async def generate_runtime_bridge_proposal(self) -> dict[str, Any]:
        if not self._runtime_recovery_context:
            raise RuntimeError("no active runtime DES recovery context is available")
        prepared_bridge_request = deepcopy(
            self._runtime_recovery_context.get("prepared_bridge_request") or {}
        )
        if not prepared_bridge_request:
            raise RuntimeError("no prepared bridge request is available")
        if self._runtime_repair_inflight:
            raise RuntimeError("runtime recovery is already in progress")

        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        if status not in {"bridge_ready", "llm_bridge", "human_required"}:
            raise RuntimeError(
                "bridge exploration can only be started from bridge_ready, llm_bridge, or human_required"
            )
        if status == "bridge_ready":
            scenario_hint = canonical_preprogrammed_bridge_scenario_id(
                self._auto_runtime_preprogrammed_scenario_id()
            )
            if scenario_hint:
                self.logger.info(
                    "[Product] Auto-routing runtime bridge generation to preprogrammed scenario: scenario_id=%s",
                    scenario_hint,
                )
                return self.load_preprogrammed_runtime_bridge_scenario_sync(scenario_hint)

        failed_task_id = str(
            self.runtime_recovery.get("failed_task_id")
            or self._runtime_recovery_context.get("failed_task_id", "")
        ).strip()
        violations = deepcopy(list(self._runtime_recovery_context.get("violations") or []))
        bridge_debug = deepcopy(
            (prepared_bridge_request.get("bridge_debug") or self.runtime_recovery.get("bridge_debug") or {})
        )

        self._set_runtime_recovery(
            status="bridge_ready",
            resolution_class="none",
            trigger=str(self._runtime_recovery_context.get("trigger", "")),
            failed_task_id=failed_task_id,
            message="Running bounded LLM bridge reasoning from the prepared session.",
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=False,
            bridge_proposal=None,
            bridge_debug=bridge_debug if bridge_debug else None,
            bridge_approval_state="generating",
            active_bridge_sequence=None,
            bridge_feedback_history=list(self._runtime_recovery_context.get("bridge_feedback_history") or []),
            violations=violations,
            append_history=True,
            history_message="Operator started LLM bridge exploration from the prepared request.",
        )
        await asyncio.to_thread(self._persist_product_state)

        self._runtime_repair_inflight = True
        try:
            proposal = await self.process_planner.execute_prepared_bridge_request(
                prepared_bridge_request
            )
            bridge_debug = self.process_planner.get_last_bridge_debug()
            self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(
                prepared_bridge_request
            )
            self._record_runtime_bridge_artifacts(
                phase="single_shot",
                prepared_bridge_request=prepared_bridge_request,
            )
            self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(
                prepared_bridge_request
            )
            bridge_debug = deepcopy(
                prepared_bridge_request.get("bridge_debug") or bridge_debug or {}
            )
            if isinstance(proposal, dict):
                bridge_summary = self.process_planner._bridge_summary(proposal)
                bridge_text = ", ".join(str(item) for item in bridge_summary if item) or "bridge step(s)"
                recovery = self._set_runtime_recovery(
                    status="llm_bridge",
                    resolution_class="none",
                    trigger=str(self._runtime_recovery_context.get("trigger", "")),
                    failed_task_id=failed_task_id,
                    message="Validated LLM bridge proposal is ready for final approval.",
                    attempts_used=self._runtime_repair_fail_streak,
                    attempts_max=self._runtime_repair_max_attempts,
                    used_llm_bridge=True,
                    bridge_proposal=proposal,
                    bridge_debug=bridge_debug if bridge_debug else None,
                    bridge_approval_state="pending",
                    active_bridge_sequence=None,
                    bridge_feedback_history=list(
                        self._runtime_recovery_context.get("bridge_feedback_history") or []
                    ),
                    violations=violations,
                    append_history=True,
                    history_message=f"LLM bridge proposed {bridge_text}.",
                )
                self._clear_plan_safety_alert()
                await asyncio.to_thread(self._persist_product_state)
                return recovery

            bridge_status = str((bridge_debug or {}).get("status") or "").strip().lower()
            if bridge_status == "paused_after_grounding":
                message = (
                    "LLM bridge grounding completed and is paused before outline generation for review."
                )
                recovery = self._set_runtime_recovery(
                    status="llm_bridge",
                    resolution_class="none",
                    trigger=str(self._runtime_recovery_context.get("trigger", "")),
                    failed_task_id=failed_task_id,
                    message=message,
                    attempts_used=self._runtime_repair_fail_streak,
                    attempts_max=self._runtime_repair_max_attempts,
                    used_llm_bridge=True,
                    bridge_proposal=None,
                    bridge_debug=bridge_debug if bridge_debug else None,
                    bridge_approval_state="generating",
                    active_bridge_sequence=None,
                    bridge_feedback_history=list(
                        self._runtime_recovery_context.get("bridge_feedback_history") or []
                    ),
                    violations=violations,
                    append_history=True,
                    history_message=message,
                )
                self._clear_plan_safety_alert()
                await asyncio.to_thread(self._persist_product_state)
                return recovery

            message = (
                "LLM bridge reasoning produced no validated final plan. Review the trace, refine, or retry DES."
            )
            recovery = self._set_runtime_recovery(
                status="human_required",
                resolution_class="human_required",
                trigger=str(self._runtime_recovery_context.get("trigger", "")),
                failed_task_id=failed_task_id,
                message=message,
                attempts_used=self._runtime_repair_fail_streak,
                attempts_max=self._runtime_repair_max_attempts,
                used_llm_bridge=True,
                bridge_proposal=None,
                bridge_debug=bridge_debug if bridge_debug else None,
                bridge_approval_state="none",
                active_bridge_sequence=None,
                bridge_feedback_history=list(
                    self._runtime_recovery_context.get("bridge_feedback_history") or []
                ),
                violations=violations,
                append_history=True,
                history_message=message,
            )
            self._set_plan_safety_alert(
                stage="runtime",
                message=message,
                retries_used=self._runtime_repair_fail_streak,
                retries_max=self._runtime_repair_max_attempts,
                violations=violations,
                paused=True,
            )
            await asyncio.to_thread(self._persist_product_state)
            return recovery
        except Exception as exc:
            self.logger.exception("[Product] LLM bridge exploration failed.")
            bridge_debug = self.process_planner.get_last_bridge_debug()
            message = f"{self.agent_name}: LLM bridge exploration failed ({exc})."
            recovery = self._set_runtime_recovery(
                status="human_required",
                resolution_class="human_required",
                trigger=str(self._runtime_recovery_context.get("trigger", "")),
                failed_task_id=failed_task_id,
                message=message,
                attempts_used=self._runtime_repair_fail_streak,
                attempts_max=self._runtime_repair_max_attempts,
                used_llm_bridge=True,
                bridge_proposal=None,
                bridge_debug=bridge_debug if bridge_debug else None,
                bridge_approval_state="none",
                active_bridge_sequence=None,
                bridge_feedback_history=list(
                    self._runtime_recovery_context.get("bridge_feedback_history") or []
                ),
                violations=violations,
                append_history=True,
                history_message=message,
            )
            self._set_plan_safety_alert(
                stage="runtime",
                message=message,
                retries_used=self._runtime_repair_fail_streak,
                retries_max=self._runtime_repair_max_attempts,
                violations=violations,
                paused=True,
            )
            await asyncio.to_thread(self._persist_product_state)
            return recovery
        finally:
            self._runtime_repair_inflight = False

    def _build_preprogrammed_runtime_bridge_bundle(
        self,
        *,
        scenario_id: str,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, Any]:
        scenario_request = deepcopy(prepared_bridge_request)
        preprogrammed_part_observations = self._derive_preprogrammed_part_observations()
        if preprogrammed_part_observations:
            scenario_request["preprogrammed_part_observations"] = preprogrammed_part_observations
        proposal = build_preprogrammed_bridge_proposal(
            scenario_id=scenario_id,
            prepared_bridge_request=scenario_request,
        )
        plan_rewrite = deepcopy(proposal.get("plan_rewrite") or {})
        normalized = self.process_planner.validate_preprogrammed_bridge_proposal(
            proposal=proposal,
            prepared_bridge_request=scenario_request,
            source="preprogrammed_scenario",
            scenario_id=scenario_id,
        )
        raw_bridge_debug = self.process_planner.get_last_bridge_debug()
        bridge_debug = deepcopy(raw_bridge_debug) if isinstance(raw_bridge_debug, dict) else {}
        bridge_debug["source"] = "preprogrammed_scenario"
        bridge_debug["scenario_id"] = scenario_id
        bridge_debug["execution_policy"] = {"complete_full_tail": True}
        if isinstance(plan_rewrite, dict) and plan_rewrite:
            bridge_debug["plan_rewrite"] = plan_rewrite
        bridge_summary = self.process_planner._bridge_summary(normalized)
        return {
            "proposal": normalized,
            "bridge_debug": bridge_debug,
            "bridge_summary": bridge_summary,
            "plan_rewrite": plan_rewrite,
        }

    def _derive_preprogrammed_part_observations(self) -> dict[str, dict[str, float]]:
        derived: dict[str, dict[str, float]] = {}
        violations = list(self._runtime_recovery_context.get("violations") or [])
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            failure_context = violation.get("failure_context")
            if not isinstance(failure_context, dict):
                continue
            observations = failure_context.get("observations")
            if not isinstance(observations, dict):
                continue
            pose_candidate = None
            for key in ("observed_pose", "pose", "position", "dropped_location"):
                raw_pose = observations.get(key)
                if not isinstance(raw_pose, dict):
                    continue
                if not {"x", "y", "z"} <= set(raw_pose.keys()):
                    continue
                try:
                    pose_candidate = {
                        "x": float(raw_pose["x"]),
                        "y": float(raw_pose["y"]),
                        "z": float(raw_pose["z"]),
                    }
                except (TypeError, ValueError):
                    pose_candidate = None
                if pose_candidate is not None:
                    break
            if pose_candidate is None:
                continue
            for entity in failure_context.get("affected_entities") or []:
                if not isinstance(entity, dict):
                    continue
                if str(entity.get("entity_type") or "").strip().lower() != "part":
                    continue
                part_name = str(entity.get("entity_id") or "").strip()
                if not part_name or part_name in derived:
                    continue
                derived[part_name] = deepcopy(pose_candidate)
        return derived

    def _cache_preprogrammed_runtime_bridge_scenario(
        self,
        *,
        scenario_id: str,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, Any]:
        bundle = self._build_preprogrammed_runtime_bridge_bundle(
            scenario_id=scenario_id,
            prepared_bridge_request=prepared_bridge_request,
        )
        cache = dict(self._runtime_recovery_context.get("preprogrammed_bridge_cache") or {})
        cache[str(scenario_id)] = deepcopy(bundle)
        self._runtime_recovery_context["preprogrammed_bridge_cache"] = cache
        return bundle

    def _resolve_preprogrammed_runtime_bridge_bundle(
        self,
        *,
        scenario_id: str,
        prepared_bridge_request: dict[str, Any],
        started_at: float | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
        scenario_key = str(scenario_id or "").strip()
        if not scenario_key:
            raise ValueError("scenario_id is empty")

        elapsed = (
            time.perf_counter() - started_at
            if started_at is not None
            else 0.0
        )
        cached_bundle = deepcopy(
            ((self._runtime_recovery_context.get("preprogrammed_bridge_cache") or {}).get(scenario_key) or {})
        )
        if isinstance(cached_bundle.get("proposal"), dict):
            normalized = deepcopy(cached_bundle["proposal"])
            bridge_debug = deepcopy(cached_bundle.get("bridge_debug") or {})
            plan_rewrite = deepcopy(cached_bundle.get("plan_rewrite") or {})
            bridge_summary = list(cached_bundle.get("bridge_summary") or [])
            self.logger.info(
                "[Product] Preprogrammed runtime bridge scenario cache hit: scenario_id=%s elapsed=%.3fs",
                scenario_key,
                elapsed,
            )
        else:
            bundle = self._cache_preprogrammed_runtime_bridge_scenario(
                scenario_id=scenario_key,
                prepared_bridge_request=prepared_bridge_request,
            )
            normalized = deepcopy(bundle.get("proposal") or {})
            bridge_debug = deepcopy(bundle.get("bridge_debug") or {})
            plan_rewrite = deepcopy(bundle.get("plan_rewrite") or {})
            bridge_summary = list(bundle.get("bridge_summary") or [])
            self.logger.info(
                "[Product] Preprogrammed runtime bridge scenario built+validated: scenario_id=%s elapsed=%.3fs",
                scenario_key,
                elapsed,
            )

        if not isinstance(bridge_debug, dict):
            bridge_debug = {}
        bridge_debug["source"] = "preprogrammed_scenario"
        bridge_debug["scenario_id"] = scenario_key
        bridge_debug["execution_policy"] = {"complete_full_tail": True}
        if isinstance(plan_rewrite, dict) and plan_rewrite:
            bridge_debug["plan_rewrite"] = plan_rewrite
        return scenario_key, normalized, bridge_debug, plan_rewrite, bridge_summary

    def load_preprogrammed_runtime_bridge_scenario_sync(self, scenario_id: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        if not self._runtime_recovery_context:
            raise RuntimeError("no active runtime DES recovery context is available")

        prepared_bridge_request = deepcopy(
            self._runtime_recovery_context.get("prepared_bridge_request") or {}
        )
        if not prepared_bridge_request:
            raise RuntimeError("no prepared bridge request is available")
        if self._runtime_repair_inflight:
            raise RuntimeError("runtime recovery is already in progress")

        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        if status != "bridge_ready":
            raise RuntimeError("preprogrammed bridge scenarios can only be loaded from the bridge-ready state")

        scenario_key = str(scenario_id or "").strip()
        if not scenario_key:
            raise ValueError("scenario_id is empty")
        self.logger.info(
            "[Product] Loading preprogrammed runtime bridge scenario start: scenario_id=%s elapsed=%.3fs",
            scenario_key,
            time.perf_counter() - started_at,
        )
        try:
            scenario_key, normalized, bridge_debug, plan_rewrite, bridge_summary = (
                self._resolve_preprogrammed_runtime_bridge_bundle(
                    scenario_id=scenario_key,
                    prepared_bridge_request=prepared_bridge_request,
                    started_at=started_at,
                )
            )
        except Exception:
            self.logger.exception(
                "[Product] Loading preprogrammed runtime bridge scenario failed: scenario_id=%s",
                scenario_key,
            )
            raise

        failed_task_id = str(
            self.runtime_recovery.get("failed_task_id")
            or self._runtime_recovery_context.get("failed_task_id", "")
        ).strip()
        violations = deepcopy(list(self._runtime_recovery_context.get("violations") or []))
        if not bridge_summary:
            bridge_summary = self.process_planner._bridge_summary(normalized)
        bridge_text = ", ".join(str(item) for item in bridge_summary if item) or scenario_key
        recovery = self._set_runtime_recovery(
            status="llm_bridge",
            resolution_class="none",
            trigger=str(self._runtime_recovery_context.get("trigger", "")),
            failed_task_id=failed_task_id,
            message="Preprogrammed recovery scenario loaded; auto-approving.",
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=False,
            bridge_proposal=normalized,
            bridge_debug=bridge_debug if bridge_debug else None,
            bridge_approval_state="pending",
            active_bridge_sequence=None,
            bridge_feedback_history=list(
                self._runtime_recovery_context.get("bridge_feedback_history") or []
            ),
            violations=violations,
            append_history=True,
            history_message=f"Loaded preprogrammed recovery scenario: {bridge_text}.",
        )
        self._clear_plan_safety_alert()
        self.logger.info(
            "[Product] Preprogrammed runtime bridge scenario ready: scenario_id=%s elapsed=%.3fs",
            scenario_key,
            time.perf_counter() - started_at,
        )
        self.logger.info(
            "[Product] Auto-approving preprogrammed runtime bridge scenario: scenario_id=%s",
            scenario_key,
        )
        return self.approve_runtime_bridge_proposal_sync()

    async def load_preprogrammed_runtime_bridge_scenario(self, scenario_id: str) -> dict[str, Any]:
        return self.load_preprogrammed_runtime_bridge_scenario_sync(scenario_id)

    def approve_runtime_bridge_proposal_sync(self) -> dict[str, Any]:
        started_at = time.perf_counter()
        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        if status != "llm_bridge":
            raise RuntimeError("no pending bridge proposal is awaiting approval")

        proposal = self.runtime_recovery.get("bridge_proposal")
        if not isinstance(proposal, dict):
            raise RuntimeError("bridge proposal is missing")
        bridge_debug = deepcopy(self.runtime_recovery.get("bridge_debug") or {})
        used_llm_bridge = bool(self.runtime_recovery.get("used_llm_bridge", False))
        plan_rewrite = (
            deepcopy(bridge_debug.get("plan_rewrite") or {})
            if isinstance(bridge_debug, dict)
            else {}
        )

        failed_task_id = str(
            self.runtime_recovery.get("failed_task_id")
            or self._runtime_recovery_context.get("failed_task_id", "")
        ).strip()
        violations = list(self._runtime_recovery_context.get("violations") or [])
        self.logger.info(
            "[Product] Approving runtime bridge proposal start: failed_task_id=%s elapsed=%.3fs",
            failed_task_id,
            time.perf_counter() - started_at,
        )
        planner_nodes_snapshot = deepcopy(self.process_planner.nodes)
        planner_global_fsa_snapshot = deepcopy(self.process_planner.global_fsa)
        bridge_replaces_failed_branch = bool(plan_rewrite.get("replace_failed_branch"))
        anchor_task_id = failed_task_id
        if bridge_replaces_failed_branch and failed_task_id:
            direct_predecessors = self._direct_predecessors_from_nodes(
                planner_nodes_snapshot,
                failed_task_id,
            )
            if direct_predecessors:
                anchor_task_id = direct_predecessors[0]
            else:
                anchor_task_id = ""

        deleted_task_ids: list[str] = []
        if bridge_replaces_failed_branch and failed_task_id:
            deleted_task_ids = sorted(
                set(self._collect_descendants_from_nodes(planner_nodes_snapshot, failed_task_id))
                | {failed_task_id}
            )
        explicit_delete_task_ids: list[str] = []
        if isinstance(plan_rewrite, dict):
            for task_id in plan_rewrite.get("delete_task_ids") or []:
                candidate = str(task_id or "").strip()
                if (
                    candidate
                    and candidate not in explicit_delete_task_ids
                    and candidate != failed_task_id
                ):
                    explicit_delete_task_ids.append(candidate)
        if explicit_delete_task_ids:
            deleted_task_ids = sorted(set(deleted_task_ids) | set(explicit_delete_task_ids))

        resumable_task_ids = []
        resume_task_ids_explicit = False
        if isinstance(plan_rewrite, dict):
            resume_task_ids_explicit = "resume_task_ids" in plan_rewrite
            for task_id in plan_rewrite.get("resume_task_ids") or []:
                candidate = str(task_id or "").strip()
                if (
                    candidate
                    and candidate not in resumable_task_ids
                    and candidate not in deleted_task_ids
                ):
                    resumable_task_ids.append(candidate)
        if not resumable_task_ids and not resume_task_ids_explicit:
            resumable_task_ids = [
                str(node.get("id", "")).strip()
                for node in planner_nodes_snapshot
                if isinstance(node, dict)
                and node.get("type") == "task"
                and str(node.get("id", "")).strip()
                and str(node.get("id", "")).strip() != failed_task_id
                and str(node.get("id", "")).strip() not in deleted_task_ids
                and str(node.get("status", "")).strip().lower() in {"pending", "blocked"}
            ]
        self.logger.info(
            "[Product] Bridge approval rewrite: anchor=%s replace_failed_branch=%s delete=%s resume=%s",
            anchor_task_id or "<none>",
            bridge_replaces_failed_branch,
            deleted_task_ids,
            resumable_task_ids,
        )
        try:
            tasks = self.process_planner.apply_bridge_macro_proposal(
                proposal,
                anchor_task_id=anchor_task_id,
            )
            post_updates: list[dict[str, Any]] = []
            if deleted_task_ids:
                post_updates.extend(
                    [
                        {
                            "id": task_id,
                            "delete": True,
                            "change_reason": (
                                f"DELETION: Approved bridge recovery replaces failed task branch task {task_id}"
                            ),
                        }
                        for task_id in deleted_task_ids
                        if task_id and self.process_planner._find_node(task_id) is not None
                    ]
                )
            if tasks and resumable_task_ids:
                self.process_planner._gate_tasks_after_recovery_tail(
                    post_updates,
                    tail_task_id=str(tasks[-1].get("id", "")).strip(),
                    before_task_ids=resumable_task_ids,
                    change_prefix="Approved bridge recovery",
                )
            if post_updates:
                self.process_planner._apply_replan_patch(post_updates)
            for task_id in deleted_task_ids:
                self.task_states.pop(task_id, None)
            for task in tasks:
                task_id = str(task.get("id", "")).strip()
                if task_id:
                    self.task_states[task_id] = "pending"
            for task_id in resumable_task_ids:
                node = self.process_planner._find_node(task_id)
                if isinstance(node, dict):
                    self.task_states[task_id] = str(node.get("status", "pending") or "pending")
        except Exception as exc:
            self.logger.exception("[Product] Failed to materialize approved bridge proposal.")
            try:
                self.process_planner.nodes = planner_nodes_snapshot
                self.process_planner.global_fsa = deepcopy(planner_global_fsa_snapshot)
                if hasattr(self, "plan_path"):
                    self.process_planner.save(self.plan_path)
                if planner_global_fsa_snapshot is not None and hasattr(self, "global_fsa_path"):
                    self.process_planner.save_global_fsa(self.global_fsa_path)
            except Exception:
                self.logger.exception("[Product] Failed to roll back planner state after approval failure.")
            message = f"{self.agent_name}: approved bridge proposal could not be compiled ({exc})."
            recovery = self._set_runtime_recovery(
                status="human_required",
                resolution_class="human_required",
                message=message,
                used_llm_bridge=used_llm_bridge,
                bridge_approval_state="none",
                bridge_debug=bridge_debug if bridge_debug else None,
                active_bridge_sequence=None,
                violations=violations,
                append_history=True,
                history_message=message,
            )
            self._set_plan_safety_alert(
                stage="runtime",
                message=message,
                retries_used=self._runtime_repair_fail_streak,
                retries_max=self._runtime_repair_max_attempts,
                violations=violations,
                paused=True,
            )
            threading.Thread(
                target=self._persist_product_state,
                name=f"{self.agent_name}-persist-product-state",
                daemon=True,
            ).start()
            return recovery

        bridge_sequence_id = str(tasks[0].get("bridge_sequence_id", "")).strip() if tasks else ""
        active_bridge_sequence = None
        if bridge_sequence_id:
            active_bridge_sequence = {
                "bridge_sequence_id": bridge_sequence_id,
                "bridge_task_ids": [
                    str(task.get("id", "")).strip()
                    for task in tasks
                    if str(task.get("id", "")).strip()
                ],
                "bridge_sequence_length": len(tasks),
                "trigger": str(self._runtime_recovery_context.get("trigger", "")),
                "failed_task_id": failed_task_id,
                "violations": deepcopy(violations),
                "used_llm_bridge": used_llm_bridge,
                "system_coordination_state": deepcopy(
                    self._runtime_recovery_context.get("system_coordination_state") or {}
                ),
                "state": "approved",
                "plan_rewrite": deepcopy(plan_rewrite) if isinstance(plan_rewrite, dict) else {},
            }
            if isinstance(bridge_debug, dict):
                execution_policy = bridge_debug.get("execution_policy")
                if isinstance(execution_policy, dict) and execution_policy:
                    active_bridge_sequence["execution_policy"] = deepcopy(execution_policy)
                source = str(bridge_debug.get("source", "")).strip()
                if source:
                    active_bridge_sequence["source"] = source
                scenario_id = str(bridge_debug.get("scenario_id", "")).strip()
                if scenario_id:
                    active_bridge_sequence["scenario_id"] = scenario_id
        if bridge_debug:
            bridge_debug["approval"] = {
                "approved_at_utc": self._utc_now_iso(),
                "entry_task_ids": deepcopy(resumable_task_ids),
                "deleted_task_ids": deepcopy(deleted_task_ids),
                "anchor_task_id": anchor_task_id,
                "compiled_bridge_task_ids": (
                    deepcopy(active_bridge_sequence.get("bridge_task_ids") or [])
                    if isinstance(active_bridge_sequence, dict)
                    else []
                ),
                "compiled_bridge_tasks": self._bridge_task_debug_rows(
                    list(active_bridge_sequence.get("bridge_task_ids") or [])
                    if isinstance(active_bridge_sequence, dict)
                    else []
                ),
            }

        validation_message = (
            f"Approved bridge proposal '{proposal.get('macro_name') or proposal.get('function_name') or 'bridge_recovery'}' compiled to "
            f"{len(tasks)} task(s); validating updated plan."
        )
        recovery = self._set_runtime_recovery(
            status="validating",
            resolution_class="none",
            trigger=str(self._runtime_recovery_context.get("trigger", "")),
            failed_task_id=failed_task_id,
            message=validation_message,
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=used_llm_bridge,
            bridge_proposal=proposal,
            bridge_debug=bridge_debug if bridge_debug else None,
            bridge_approval_state="approved",
            active_bridge_sequence=active_bridge_sequence,
            violations=violations,
            append_history=True,
            history_message=validation_message,
        )
        self._clear_plan_safety_alert()
        self.logger.info(
            "[Product] Approved runtime bridge proposal compiled: tasks=%d elapsed=%.3fs",
            len(tasks),
            time.perf_counter() - started_at,
        )
        try:
            # Preprogrammed bridge scenarios are pre-verified; skip the
            # expensive BFS revalidation in CCA (saves ~30 s).
            _bridge_source = str(
                (active_bridge_sequence or {}).get("source", "")
            ).strip().lower()
            _skip_reval = _bridge_source == "preprogrammed_scenario"
            self.logger.info(
                "[Product] Approved bridge proposal finalization started: sending runtime plan validation%s.",
                " (skip_revalidation)" if _skip_reval else "",
            )
            self._send_runtime_plan_validation_check_sync(
                skip_revalidation=_skip_reval,
            )
            self.logger.info(
                "[Product] Approved bridge proposal validation sent successfully; persisting updated runtime state."
            )
            threading.Thread(
                target=self._persist_plan_snapshot,
                name=f"{self.agent_name}-persist-plan-snapshot",
                daemon=True,
            ).start()
            threading.Thread(
                target=self._persist_product_state,
                name=f"{self.agent_name}-persist-product-state",
                daemon=True,
            ).start()
            threading.Thread(
                target=self._persist_resource_state,
                name=f"{self.agent_name}-persist-resource-state",
                daemon=True,
            ).start()
        except Exception as exc:
            self.logger.exception("[Product] Approved bridge proposal validation dispatch failed.")
            message = (
                f"{self.agent_name}: approved bridge proposal could not start runtime "
                f"plan validation ({exc})."
            )
            recovery = self._set_runtime_recovery(
                status="human_required",
                resolution_class="human_required",
                trigger=str(self._runtime_recovery_context.get("trigger", "")),
                failed_task_id=str(self._runtime_recovery_context.get("failed_task_id", "")),
                message=message,
                attempts_used=self._runtime_repair_fail_streak,
                attempts_max=self._runtime_repair_max_attempts,
                used_llm_bridge=used_llm_bridge,
                bridge_proposal=None,
                bridge_approval_state="approved",
                bridge_debug=bridge_debug if bridge_debug else None,
                active_bridge_sequence=None,
                violations=violations,
                append_history=True,
                history_message=message,
            )
            self._set_plan_safety_alert(
                stage="runtime",
                message=message,
                retries_used=self._runtime_repair_fail_streak,
                retries_max=self._runtime_repair_max_attempts,
                violations=violations,
                paused=True,
            )
            threading.Thread(
                target=self._persist_product_state,
                name=f"{self.agent_name}-persist-product-state",
                daemon=True,
            ).start()
            return recovery
        self.logger.info(
            "[Product] Approved runtime bridge proposal dispatched runtime validation: elapsed=%.3fs",
            time.perf_counter() - started_at,
        )
        return recovery

    async def approve_runtime_bridge_proposal(self) -> dict[str, Any]:
        return self.approve_runtime_bridge_proposal_sync()

    async def reject_runtime_bridge_proposal(self, feedback: str) -> dict[str, Any]:
        if not self._runtime_recovery_context:
            raise RuntimeError("no active runtime DES recovery context is available")
        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        if status != "llm_bridge":
            raise RuntimeError("no pending bridge proposal is awaiting rejection")

        feedback_text = str(feedback or "").strip()
        if not feedback_text:
            raise ValueError("bridge rejection feedback is empty")

        feedback_history = [
            str(item).strip()
            for item in (self.runtime_recovery.get("bridge_feedback_history") or [])
            if str(item).strip()
        ]
        feedback_history.append(feedback_text)
        self._runtime_recovery_context["bridge_feedback_history"] = list(feedback_history)
        prepared_bridge_request = deepcopy(
            self._runtime_recovery_context.get("prepared_bridge_request") or {}
        )
        if prepared_bridge_request:
            bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
            operator_feedback_history = [
                str(item).strip()
                for item in (bridge_session.get("operator_feedback_history") or [])
                if str(item).strip()
            ]
            operator_feedback_history.append(feedback_text)
            bridge_session["operator_feedback_history"] = operator_feedback_history[-12:]
            prepared_bridge_request["bridge_session"] = bridge_session
            self._runtime_recovery_context["prepared_bridge_request"] = prepared_bridge_request

        recovery = self._set_runtime_recovery(
            status="human_required",
            resolution_class="human_required",
            trigger=str(self._runtime_recovery_context.get("trigger", "")),
            failed_task_id=str(self._runtime_recovery_context.get("failed_task_id", "")),
            message="Operator rejected the bridge proposal. Add guidance and rerun bridge reasoning when ready.",
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=bool(self.runtime_recovery.get("used_llm_bridge", False)),
            bridge_proposal=None,
            bridge_debug=(
                self.process_planner.get_last_bridge_debug()
                or self.runtime_recovery.get("bridge_debug")
            ),
            bridge_approval_state="none",
            active_bridge_sequence=None,
            violations=list(self._runtime_recovery_context.get("violations") or []),
            bridge_feedback_history=feedback_history,
            append_history=True,
            history_message=f"Operator rejected bridge proposal: {feedback_text}",
        )
        self._set_plan_safety_alert(
            stage="runtime",
            message=str(recovery.get("message", "") or "Bridge proposal rejected."),
            retries_used=self._runtime_repair_fail_streak,
            retries_max=self._runtime_repair_max_attempts,
            violations=list(self._runtime_recovery_context.get("violations") or []),
            paused=True,
        )
        await asyncio.to_thread(self._persist_product_state)
        return recovery

    async def retry_runtime_recovery_des(self) -> dict[str, Any]:
        if not self._runtime_recovery_context:
            raise RuntimeError("no active runtime DES recovery context is available")
        trigger = str(self._runtime_recovery_context.get("trigger", "") or "operator_retry")
        failed_task_id = str(self._runtime_recovery_context.get("failed_task_id", "")).strip()
        violations = deepcopy(list(self._runtime_recovery_context.get("violations") or []))
        system_coordination_state = dict(self._runtime_recovery_context.get("system_coordination_state") or {})
        self._clear_plan_safety_alert()
        return await self._run_des_runtime_recovery_attempt(
            violations=violations,
            trigger=trigger,
            failed_task_id=failed_task_id,
            system_coordination_state=system_coordination_state,
            reset_attempts=True,
            history_message="Operator requested a DES retry for the active runtime recovery session.",
        )

    async def _build_plan(self, requirement_text: str, safety_text: str = ""):
        """Build requirements, expand to tasks, and compile the global FSA."""
        # 1) NL → structured requirements
        await self.process_planner.build_high_level(requirement_text)
        self.process_planner.save(self.structured_requirements_path)

        # 2) Expand → DAG
        await self.process_planner.expand_requirements_to_tasks(safety_text=safety_text)
        self.process_planner.save(self.plan_path)

        # 3) Compile + save FSA (also sets self.process_planner.global_fsa)
        self.process_planner.save_global_fsa(self.global_fsa_path)

        # 4) Return both artifacts
        return self.process_planner.nodes, self.process_planner.global_fsa

    def _load_precomputed_plan_bundle(self) -> bool:
        """Load precomputed plan/global FSA artifacts when provided by startup bundle context."""
        bundle = dict(self.precomputed_bundle or {})
        artifacts = bundle.get("artifacts", {}) if isinstance(bundle, dict) else {}
        if not isinstance(artifacts, dict):
            return False

        plan_path = artifacts.get("plan_json")
        fsa_path = artifacts.get("global_fsa_json")
        req_path = artifacts.get("requirements_json")

        if not plan_path or not fsa_path:
            return False

        p_plan = Path(str(plan_path))
        p_fsa = Path(str(fsa_path))
        if not p_plan.exists() or not p_fsa.exists():
            self.logger.warning(
                "[Bundle] Precomputed plan artifacts missing plan=%s fsa=%s",
                p_plan,
                p_fsa,
            )
            return False

        self.process_planner.load(p_plan)
        self.process_planner.load_global_fsa(p_fsa)

        # Keep monitor snapshots aligned with runtime expectations.
        self.process_planner.save(self.plan_path)
        self.process_planner.save_global_fsa(self.global_fsa_path)

        if req_path:
            p_req = Path(str(req_path))
            if p_req.exists():
                self.structured_requirements_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p_req, self.structured_requirements_path)

        self.logger.info(
            "[Bundle] Using precomputed plan bundle_id=%s plan=%s fsa=%s",
            bundle.get("bundle_id", ""),
            p_plan,
            p_fsa,
        )
        return True

    def _auto_runtime_preprogrammed_scenario_id(self) -> str:
        bundle = dict(self.precomputed_bundle or {})
        bundle_id = str(bundle.get("bundle_id", "") or "").strip()
        if (
            canonical_preprogrammed_bridge_scenario_id(bundle_id)
            == canonical_preprogrammed_bridge_scenario_id(_CASE3_PREPROGRAMMED_SCENARIO_ID)
        ):
            return _CASE3_PREPROGRAMMED_SCENARIO_ID

        requirement_candidates = [
            self.product_specification_file,
            bundle.get("product_spec_file"),
        ]
        for candidate in requirement_candidates:
            raw = str(candidate or "").strip()
            if not raw:
                continue
            if Path(raw).name in _CASE3_PREPROGRAMMED_REQUIREMENT_FILES:
                return _CASE3_PREPROGRAMMED_SCENARIO_ID
        return ""



    # --------------------------------------------------------------------- #
    # Behaviours
    # --------------------------------------------------------------------- #
    class _Kickoff(OneShotBehaviour):
        async def run(self):
            agent: "ProductAgent" = self.agent
            retries_used = 0
            used_precomputed = False
            max_retries = 3

            try:
                instruction = agent._extract_requirement_text()
                safety_text = agent._read_safety_text()
                agent.safety_text = safety_text
                agent.runtime_repair_state = "idle"
                agent._runtime_repair_fail_streak = 0
                agent._clear_plan_safety_alert()
                agent._clear_runtime_recovery()

                used_precomputed = agent._load_precomputed_plan_bundle()
                max_retries = 0 if used_precomputed else 3
                if not used_precomputed:
                    if not instruction:
                        raise RuntimeError("no product requirement text available for startup planning")
                    await agent._build_plan(instruction, safety_text)

                while True:
                    if used_precomputed:
                        agent.logger.info(
                            "[Product] Initializing startup runtime monitors for verified plan bundle (offline revalidation skipped)."
                        )
                    else:
                        agent.logger.info(
                            "[Product] Validating startup plan (auto-replans %d/%d)...",
                            retries_used,
                            max_retries,
                        )

                    payload = agent._build_plan_validation_payload(
                        skip_revalidation=used_precomputed,
                    )
                    msg = Message(to=agent.cca_jid)
                    msg.set_metadata("type", "plan_safety_check")
                    msg.body = json.dumps(payload)
                    await send_agent_message(
                        self,
                        msg,
                        transport_label="product_plan_check",
                    )

                    reply = None
                    while reply is None:
                        reply = await self.receive(timeout=5.0)

                    if reply.metadata.get("type") != "plan_safety_result":
                        continue

                    data = json.loads(reply.body or "{}")
                    is_safe = bool(data.get("ok", False))
                    violations = data.get("violations")
                    if not isinstance(violations, list):
                        violations = []

                    if is_safe:
                        message = (
                            f"{agent.agent_name}: loaded verified plan set; startup skipped offline safety revalidation."
                            if used_precomputed
                            else f"{agent.agent_name}: startup plan passed safety validation."
                        )
                        agent.logger.info("[Product] Plan PASSED safety validation.")
                        agent._clear_plan_safety_alert()
                        agent._ensure_plan_result_inbox()
                        agent.add_behaviour(agent._PlanExecutor())
                        agent._set_startup_readiness(
                            startup_ready=True,
                            success=True,
                            continuing=False,
                            message=message,
                            retries_used=retries_used,
                            retries_max=max_retries,
                            violations=[],
                            used_precomputed_bundle=used_precomputed,
                        )
                        agent._set_kickoff_result(
                            success=True,
                            message=message,
                            retries_used=retries_used,
                            retries_max=max_retries,
                            violations=[],
                        )
                        return

                    if used_precomputed:
                        message = (
                            f"{agent.agent_name}: verified plan set failed startup safety validation "
                            f"({len(violations)} witness(es)); startup aborted."
                        )
                        agent.logger.error(
                            "[Bundle] Precomputed plan failed safety validation (%d violations). Aborting kickoff.",
                            len(violations),
                        )
                        alert = agent._set_plan_safety_alert(
                            stage="kickoff",
                            message=message,
                            retries_used=0,
                            retries_max=0,
                            violations=violations,
                            paused=False,
                        )
                        agent._set_startup_readiness(
                            startup_ready=True,
                            success=False,
                            continuing=False,
                            message=message,
                            retries_used=0,
                            retries_max=0,
                            violations=violations,
                            alert=alert,
                            used_precomputed_bundle=used_precomputed,
                        )
                        agent._set_kickoff_result(
                            success=False,
                            message=message,
                            retries_used=0,
                            retries_max=0,
                            violations=violations,
                            alert=alert,
                        )
                        return

                    if retries_used >= max_retries:
                        message = (
                            f"{agent.agent_name}: startup plan still violates safety after "
                            f"{retries_used}/{max_retries} auto-replan attempt(s); startup aborted."
                        )
                        agent.logger.error("[Product] Max replanning attempts reached. Aborting.")
                        alert = agent._set_plan_safety_alert(
                            stage="kickoff",
                            message=message,
                            retries_used=retries_used,
                            retries_max=max_retries,
                            violations=violations,
                            paused=False,
                        )
                        agent._set_startup_readiness(
                            startup_ready=True,
                            success=False,
                            continuing=False,
                            message=message,
                            retries_used=retries_used,
                            retries_max=max_retries,
                            violations=violations,
                            alert=alert,
                            used_precomputed_bundle=used_precomputed,
                        )
                        agent._set_kickoff_result(
                            success=False,
                            message=message,
                            retries_used=retries_used,
                            retries_max=max_retries,
                            violations=violations,
                            alert=alert,
                        )
                        return

                    next_attempt = retries_used + 1
                    before_hash = agent._task_nodes_hash()
                    agent.logger.warning(
                        "[Product] Startup plan failed safety check (%d violation(s)); triggering auto-replan %d/%d.",
                        len(violations),
                        next_attempt,
                        max_retries,
                    )
                    startup_message = (
                        f"{agent.agent_name}: startup plan failed initial safety validation "
                        f"({len(violations)} witness(es)); auto-replan {next_attempt}/{max_retries} "
                        "running in background."
                    )
                    startup_alert = agent._set_plan_safety_alert(
                        stage="kickoff",
                        message=startup_message,
                        retries_used=next_attempt,
                        retries_max=max_retries,
                        violations=violations,
                        paused=False,
                    )
                    agent._set_startup_readiness(
                        startup_ready=True,
                        success=False,
                        continuing=True,
                        message=startup_message,
                        retries_used=next_attempt,
                        retries_max=max_retries,
                        violations=violations,
                        alert=startup_alert,
                        used_precomputed_bundle=used_precomputed,
                    )
                    try:
                        retries_used = next_attempt
                        await agent.process_planner.replan_with_feedback_offline(violations)
                        if before_hash == agent._task_nodes_hash():
                            message = (
                                f"{agent.agent_name}: startup auto-replan produced no plan change on "
                                f"attempt {retries_used}/{max_retries}; startup aborted."
                            )
                            alert = agent._set_plan_safety_alert(
                                stage="kickoff",
                                message=message,
                                retries_used=retries_used,
                                retries_max=max_retries,
                                violations=violations,
                                paused=False,
                            )
                            agent._set_startup_readiness(
                                startup_ready=True,
                                success=False,
                                continuing=False,
                                message=message,
                                retries_used=retries_used,
                                retries_max=max_retries,
                                violations=violations,
                                alert=alert,
                                used_precomputed_bundle=used_precomputed,
                            )
                            agent._set_kickoff_result(
                                success=False,
                                message=message,
                                retries_used=retries_used,
                                retries_max=max_retries,
                                violations=violations,
                                alert=alert,
                            )
                            return
                        agent.process_planner.compile_global_fsa()
                        await asyncio.to_thread(agent._persist_plan_snapshot)
                        await asyncio.to_thread(agent.process_planner.save_global_fsa, agent.global_fsa_path)
                    except Exception as exc:
                        message = (
                            f"{agent.agent_name}: startup auto-replan attempt {retries_used}/{max_retries} "
                            f"failed ({exc}); startup aborted."
                        )
                        agent.logger.exception("[Product] Startup auto-replan attempt failed.")
                        alert = agent._set_plan_safety_alert(
                            stage="kickoff",
                            message=message,
                            retries_used=retries_used,
                            retries_max=max_retries,
                            violations=violations,
                            paused=False,
                        )
                        agent._set_startup_readiness(
                            startup_ready=True,
                            success=False,
                            continuing=False,
                            message=message,
                            retries_used=retries_used,
                            retries_max=max_retries,
                            violations=violations,
                            alert=alert,
                            used_precomputed_bundle=used_precomputed,
                        )
                        agent._set_kickoff_result(
                            success=False,
                            message=message,
                            retries_used=retries_used,
                            retries_max=max_retries,
                            violations=violations,
                            alert=alert,
                        )
                        return
            except Exception as exc:
                message = f"{agent.agent_name}: kickoff failed ({exc})."
                agent.logger.exception("[Product] Kickoff failed.")
                alert = agent._set_plan_safety_alert(
                    stage="kickoff",
                    message=message,
                    retries_used=retries_used,
                    retries_max=max_retries,
                    violations=[],
                    paused=False,
                )
                agent._set_startup_readiness(
                    startup_ready=False,
                    success=False,
                    continuing=False,
                    message=message,
                    retries_used=retries_used,
                    retries_max=max_retries,
                    violations=[],
                    alert=alert,
                    used_precomputed_bundle=used_precomputed,
                )
                agent._set_kickoff_result(
                    success=False,
                    message=message,
                    retries_used=retries_used,
                    retries_max=max_retries,
                    violations=[],
                    alert=alert,
                )

    class _AckInbox(CyclicBehaviour):
        """Background behaviour that listens for acknowledgements from resource agents."""

        async def run(self):
            agent: "ProductAgent" = self.agent  # type: ignore
            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            try:
                payload = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Product] Malformed ACK body (not JSON).")
                return

            task_id = payload.get("task_id", "?")
            status = payload.get("status", "unknown")
            content = str(payload.get("content") or "").strip()
            observations = payload.get("observations")
            if not isinstance(observations, dict):
                observations = None
            trace = _trace_with_timestamp(payload.get("trace"), "product_ack_received_at")

            # 1) Keep existing state map for UI/debug
            agent.task_states[task_id] = status

            # 2) ALSO update node status in the planner DAG if exists
            updated_node = False
            task_node = None
            for node in agent.process_planner.nodes:
                if node.get("id") == task_id:
                    # Map RA status → planner status; for now use it directly
                    if node.get("status") != status:
                        node["status"] = status
                        updated_node = True
                    task_node = node
                    break

            # 3) Track execution timeline
            timeline_row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": task_id,
                "status": status,
                "resource_jid": str(msg.sender),
            }
            if trace:
                timeline_row["trace"] = trace
            agent.execution_timeline.append(timeline_row)

            # 4) Update part location tracking
            if task_node:
                function_name = task_node.get("function_name", "")
                params = task_node.get("params", {})
                part_name = agent._tracked_part_name_for_task(task_node)

                if part_name:
                    agent._apply_part_tracker_update(
                        part_name=part_name,
                        function_name=function_name,
                        status=status,
                        params=params,
                        resource_jid=str(msg.sender),
                        task_id=task_id,
                        task_node=task_node,
                        observations=observations,
                    )

            # NOTE: Robot states are NOT cached here - they're collected by CentralControllerAgent
            # and provided in the replan_request message (system_coordination_state)

            handled_bridge_ack = False
            if task_node:
                handled_bridge_ack = await agent._handle_bridge_macro_ack(
                    task_node=task_node,
                    status=str(status),
                    content=content,
                    observations=observations,
                )

            if updated_node and not handled_bridge_ack:
                await asyncio.to_thread(agent._persist_plan_snapshot)
                await asyncio.to_thread(agent._persist_product_state)
                await asyncio.to_thread(agent._persist_resource_state)

            agent.logger.info(
                f"[Product] ACK ({task_id}) status='{status}' from={msg.sender}"
            )
            if trace:
                hop_labels = [
                    ("product_to_resource", "product_dispatch_sent_at", "resource_received_at"),
                    ("resource_to_cca", "resource_safety_sent_at", "cca_resource_event_received_at"),
                    ("cca_to_resource", "cca_decision_sent_at", "resource_decision_received_at"),
                    ("resource_to_product_ack", "resource_ack_sent_at", "product_ack_received_at"),
                ]
                parts = []
                for label, start_key, end_key in hop_labels:
                    delta = _trace_delta_ms(trace, start_key, end_key)
                    if delta is not None:
                        parts.append(f"{label}={delta:.0f}ms")
                if parts:
                    agent.logger.info(
                        "[Product] ACK trace (%s) %s",
                        task_id,
                        ", ".join(parts),
                    )
                transport_parts = [
                    f"{key[:-10]}={value}"
                    for key, value in sorted(trace.items())
                    if str(key).endswith("_transport") and value
                ]
                if transport_parts:
                    agent.logger.info(
                        "[Product] ACK transport (%s) %s",
                        task_id,
                        ", ".join(transport_parts),
                    )

    class _ReplanInbox(CyclicBehaviour):
        """Handle online replan requests from the CCA."""

        async def run(self):
            agent: "ProductAgent" = self.agent  # type: ignore
            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            try:
                payload = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Product] Malformed replan_request body.")
                return

            reason = payload.get("reason", "unknown")
            event = payload.get("event") or {}
            plan_ctx = payload.get("plan_ctx") or {}
            safety_ctx = payload.get("safety_ctx") or {}
            failure_ctx = event.get("failure_context") or {}
            system_coordination_state = payload.get("system_coordination_state") or {}

            failed_task_id = event.get("task_id") or plan_ctx.get("failed_task_id")
            if not failed_task_id:
                agent.logger.warning("[Product] Replan request missing failed_task_id.")
                return

            violations = agent._build_online_replan_violations(
                reason=reason,
                failed_task_id=str(failed_task_id),
                plan_ctx=plan_ctx,
                safety_ctx=safety_ctx,
                failure_ctx=failure_ctx,
            )

            agent.logger.info(
                "[Product] Online replan payload: %s", json.dumps(violations, ensure_ascii=False)
            )
            agent.logger.warning(
                "[Product] Online replan requested (%s) for failed task %s.",
                reason,
                failed_task_id,
            )
            await agent._handle_runtime_des_replan_request(
                reason=str(reason),
                failed_task_id=str(failed_task_id),
                violations=violations,
                system_coordination_state=system_coordination_state,
            )
            await asyncio.to_thread(agent._persist_product_state)
            return

            await asyncio.to_thread(agent._persist_plan_snapshot)
            await asyncio.to_thread(agent._persist_product_state)
            await asyncio.to_thread(agent._persist_resource_state)

    class _PlanSafetyResultInbox(CyclicBehaviour):
        """Handle runtime plan_safety_result replies from CCA."""

        async def run(self):
            agent: "ProductAgent" = self.agent  # type: ignore
            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            try:
                payload = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Product] Malformed plan_safety_result body.")
                return

            ok = bool(payload.get("ok", False))
            violations = payload.get("violations")
            if not isinstance(violations, list):
                violations = []

            if await agent._handle_runtime_plan_validation_result(
                ok=ok,
                violations=violations,
            ):
                return

            if ok:
                if agent._runtime_repair_fail_streak:
                    agent.logger.info(
                        "[Product] Runtime plan validation recovered after %d repair attempt(s).",
                        agent._runtime_repair_fail_streak,
                    )
                agent._runtime_repair_fail_streak = 0
                agent.runtime_repair_state = "idle"
                agent._clear_plan_safety_alert()
                await asyncio.to_thread(agent._persist_product_state)
                return

            if agent._runtime_repair_inflight:
                agent.logger.warning(
                    "[Product] Runtime plan validation failed while repair is already in progress; ignoring duplicate result."
                )
                return

            if agent._runtime_repair_fail_streak >= agent._runtime_repair_max_attempts:
                message = (
                    f"{agent.agent_name}: runtime plan validation still fails after "
                    f"{agent._runtime_repair_fail_streak}/{agent._runtime_repair_max_attempts} "
                    "auto-replan attempt(s); execution paused."
                )
                agent.logger.error(
                    "[Product] Runtime plan validation still failing after %d repair attempt(s); pausing execution.",
                    agent._runtime_repair_fail_streak,
                )
                agent.runtime_repair_state = "paused_after_failure"
                agent._set_plan_safety_alert(
                    stage="runtime",
                    message=message,
                    retries_used=agent._runtime_repair_fail_streak,
                    retries_max=agent._runtime_repair_max_attempts,
                    violations=violations,
                    paused=True,
                )
                await asyncio.to_thread(agent._persist_product_state)
                return

            agent._runtime_repair_inflight = True
            agent.runtime_repair_state = "repairing"
            agent._runtime_repair_fail_streak += 1
            try:
                before_hash = agent._task_nodes_hash()
                agent.logger.warning(
                    "[Product] Runtime plan validation failed (%d violation(s)); triggering corrective replan attempt %d/%d.",
                    len(violations),
                    agent._runtime_repair_fail_streak,
                    agent._runtime_repair_max_attempts,
                )
                await agent.process_planner.replan_with_feedback_offline(violations)
                if before_hash == agent._task_nodes_hash():
                    message = (
                        f"{agent.agent_name}: runtime auto-replan produced no plan change on "
                        f"attempt {agent._runtime_repair_fail_streak}/{agent._runtime_repair_max_attempts}; "
                        "execution paused."
                    )
                    agent.runtime_repair_state = "paused_after_failure"
                    agent._set_plan_safety_alert(
                        stage="runtime",
                        message=message,
                        retries_used=agent._runtime_repair_fail_streak,
                        retries_max=agent._runtime_repair_max_attempts,
                        violations=violations,
                        paused=True,
                    )
                    await asyncio.to_thread(agent._persist_product_state)
                    return
                agent.process_planner.compile_global_fsa()
                agent.process_planner.save_global_fsa(agent.global_fsa_path)

                check_payload = agent._build_plan_validation_payload()
                check_msg = Message(to=agent.cca_jid)
                check_msg.set_metadata("type", "plan_safety_check")
                check_msg.body = json.dumps(check_payload)
                await send_agent_message(
                    self,
                    check_msg,
                    transport_label="product_plan_check",
                )
                await asyncio.to_thread(agent._persist_plan_snapshot)
                await asyncio.to_thread(agent._persist_product_state)
                await asyncio.to_thread(agent._persist_resource_state)
            except Exception:
                agent.logger.exception(
                    "[Product] Corrective runtime replan attempt failed."
                )
                message = (
                    f"{agent.agent_name}: runtime auto-replan attempt "
                    f"{agent._runtime_repair_fail_streak}/{agent._runtime_repair_max_attempts} failed; "
                    "execution paused."
                )
                agent.runtime_repair_state = "paused_after_failure"
                agent._set_plan_safety_alert(
                    stage="runtime",
                    message=message,
                    retries_used=agent._runtime_repair_fail_streak,
                    retries_max=agent._runtime_repair_max_attempts,
                    violations=violations,
                    paused=True,
                )
                await asyncio.to_thread(agent._persist_product_state)
            finally:
                agent._runtime_repair_inflight = False

    class _TaskRetryReadyInbox(CyclicBehaviour):
        """Requeue blocked tasks after CCA clears a transient safety block."""

        async def run(self):
            agent: "ProductAgent" = self.agent  # type: ignore
            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            try:
                payload = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Product] Malformed task_retry_ready body.")
                return

            task_ids = payload.get("task_ids")
            if not isinstance(task_ids, list):
                task_id = payload.get("task_id")
                task_ids = [task_id] if task_id else []

            reactivated = agent._handle_task_retry_ready(task_ids)
            if not reactivated:
                return

            agent.logger.info(
                "[Product] Requeued %d blocked task(s) after CCA cleared transient safety block: %s",
                reactivated,
                ", ".join(str(task_id) for task_id in task_ids if task_id),
            )
            await asyncio.to_thread(agent._persist_plan_snapshot)
            await asyncio.to_thread(agent._persist_product_state)
            await asyncio.to_thread(agent._persist_resource_state)

    class _PlanExecutor(CyclicBehaviour):
        """
        Periodically checks the DAG for the next ready task and dispatches it
        as a SPADE message to the resource agent.
        """

        async def run(self):
            agent: "ProductAgent" = self.agent  # type: ignore

            # No resources? nothing to do
            if not agent.resource_jids:
                return

            if agent._runtime_recovery_blocks_execution():
                await asyncio.sleep(0.05)
                return

            # Ask planner for one ready task
            task_node = agent.process_planner.next_ready_task()
            if not task_node:
                await asyncio.sleep(0.05)
                return

            # NEW: prefer resource_jid chosen by the planner (LLM)
            to = task_node.get("resource_jid")
            if not to:
                # Fallback: first configured resource JID
                to = agent.resource_jids[0]
                agent.logger.warning(
                    "[Product] Task %s has no resource_jid, falling back to %s",
                    task_node.get("id"),
                    to,
                )

            # Build the instruction for the RobotAgent from the DAG node.
            # Enrich params with product geometry when a part_name is present.
            params = dict(task_node.get("params", {}))
            part_name = params.get("part_name")
            if part_name:
                geo = agent._geometry_for_part(part_name)
                if geo:
                    params["product_geometry"] = geo

            instruction = {
                "function_name": task_node.get("function_name"),
                "params": params,
            }

            task_id = task_node["id"]

            msg = agent._compose_task_msg(
                to=to,
                task_id=task_id,
                instruction=instruction,
                phase_id=None,
            )

            # Mark as "dispatched" (still waiting for ACK to flip to "completed")
            task_node["status"] = "dispatched"

            await send_agent_message(
                self,
                msg,
                transport_label="product_dispatch",
            )
            agent.logger.info(
                f"[Product] Dispatched DAG task {task_id} -> {to} ({instruction})"
            )

            # Short sleep so we don't hammer the RA with a storm of tasks
            await asyncio.sleep(0.1)

    @staticmethod
    def _match_resource_objects(
        resources: Iterable[Any], target_jids: Iterable[str]
    ) -> list[Any]:
        """Return the subset of *resources* whose JIDs appear in *target_jids* (case-insensitive)."""
        resources = list(resources or [])
        if not resources:
            return []

        target_jids_lower = {
            str(jid).lower() for jid in target_jids if jid is not None
        }
        if not target_jids_lower:
            return resources

        matched = []
        for agent in resources:
            # Each agent is expected to expose a `jid`; treat missing values as empty strings.
            agent_jid = str(getattr(agent, "jid", "")).lower()
            # Append resource agents whose JID matches one of the requested targets.
            if agent_jid and agent_jid in target_jids_lower:
                matched.append(agent)
        return matched

    def _collect_descendants(self, root_id: str) -> list[str]:
        """Return all descendant task IDs in the current planner DAG."""
        return self._collect_descendants_from_nodes(self.process_planner.nodes, root_id)

    @staticmethod
    def _collect_descendants_from_nodes(nodes: list[dict[str, Any]], root_id: str) -> list[str]:
        """Return all descendant task IDs from the provided task graph snapshot."""
        succ_map: dict[str, list[str]] = {}
        for node in nodes or []:
            if node.get("type") != "task":
                continue
            nid = node.get("id")
            if not nid:
                continue
            succs = node.get("successors") or []
            succ_map[str(nid)] = [str(s) for s in succs if s]

        seen: set[str] = set()
        stack = [root_id]
        while stack:
            cur = stack.pop()
            for nxt in succ_map.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return sorted(seen)

    @staticmethod
    def _direct_predecessors_from_nodes(nodes: list[dict[str, Any]], task_id: str) -> list[str]:
        lookup = {
            str(node.get("id", "")).strip(): node
            for node in nodes or []
            if isinstance(node, dict) and str(node.get("id", "")).strip()
        }
        task = lookup.get(str(task_id or "").strip()) or {}
        return [
            str(pred).strip()
            for pred in (task.get("predecessors") or [])
            if str(pred or "").strip() and str(pred or "").strip() in lookup
        ]

    def _build_online_replan_violations(
        self,
        *,
        reason: str,
        failed_task_id: str,
        plan_ctx: dict,
        safety_ctx: dict,
        failure_ctx: dict,
    ) -> list[dict]:
        """
        Build a standardized online replanning payload for the planner.
        """
        descendants = self._collect_descendants(failed_task_id)
        # Prefer explicit failed-task descendants; fall back to legacy names if present.
        unreachable = (
            plan_ctx.get("failed_task_descendant_ids")
            or plan_ctx.get("failure_blocked_task_ids")
            or plan_ctx.get("blocked_by_failure_task_ids")
            or plan_ctx.get("unreachable_task_ids")
            or []
        )
        # Ensure the failed task itself is included if present.
        plan_failed_task_id = plan_ctx.get("failed_task_id")
        if plan_failed_task_id:
            failed_task_id = str(plan_failed_task_id)
        if failed_task_id:
            unreachable = sorted(set(unreachable) | {str(failed_task_id)})
        affected_task_ids = sorted(set(unreachable) | set(descendants))

        return [
            {
                "type": reason,
                "failed_task_id": str(failed_task_id),
                "affected_task_ids": affected_task_ids,
                "failure_context": failure_ctx,
                "safety_ctx": safety_ctx,
            }
        ]

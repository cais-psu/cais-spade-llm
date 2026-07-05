"""High-level product agent that reads build instructions and instructs a resource agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import TimeoutError as FutureTimeoutError
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from spade.template import Template

from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.agents.intelligent_product.product_recovery_controller import (
    ProductRecoveryController,
)
from cais_spade_llm.agents.shared_information.llm_agent import LlmAgent
from cais_spade_llm.agents.shared_information.local_dispatch import (
    send_agent_message,
    send_agent_message_sync,
)
from cais_spade_llm.product.order import validate_product_order
from cais_spade_llm.product.profile import ProductProfile
from cais_spade_llm.resources.sensor.camera_module import CameraModule

_UNSET = object()
_LEGACY_PROCEDURAL_DES_BRIDGE_MODE = "procedural" + "_des_v1"
_ACK_PROGRESS_RANK = {
    "pending": 0,
    "dispatched": 1,
    "accepted": 2,
    "running": 3,
    "completed": 4,
    "finished": 4,
    "blocked": 4,
}


def _env_flag_enabled(*names: str, default: bool = False) -> bool:
    for name in names:
        token = str(os.environ.get(name) or "").strip().lower()
        if not token:
            continue
        return token in {"1", "true", "yes", "on"}
    return bool(default)


def _ack_status_rank(status: str) -> int:
    normalized = str(status or "").strip().lower()
    if normalized.startswith("failed"):
        return 4
    return int(_ACK_PROGRESS_RANK.get(normalized, -1))


def _ack_status_is_regression(current_status: str, incoming_status: str) -> bool:
    current_rank = _ack_status_rank(current_status)
    incoming_rank = _ack_status_rank(incoming_status)
    if current_rank < 0 or incoming_rank < 0:
        return False
    return incoming_rank < current_rank


def _should_persist_ack_state(task_node: dict[str, Any] | None, status: str) -> bool:
    if not isinstance(task_node, dict):
        return True
    if str(task_node.get("function_name") or "").strip() != "execute_recovery_macro":
        return True
    return str(status or "").strip().lower() not in {"accepted", "running", "dispatched"}


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
        resource_jids: Iterable[str] | None = None,
        resource_agents: Iterable[Any] | None = None,
        product_order_file: str | None = None,
        product_specification_file: str | None = None,
        product_geometry_file: str | None = None,
        safety_file: str | None = None,
        instruction_override: str | None = None,
        cca_jid: str | None = None,
        camera: CameraModule | None = None,
        precomputed_bundle: dict[str, Any] | None = None,
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
        self.product_profile = ProductProfile(
            name=name,
            product_specification_file=product_specification_file,
            product_order_file=product_order_file,
            product_geometry_file=product_geometry_file,
            safety_file=safety_file,
            instruction_override=instruction_override,
            precomputed_bundle=dict(precomputed_bundle or {}),
            logger=self.logger,
        )
        self.product_specification_file = self.product_profile.product_specification_file
        self.product_order_file = self.product_profile.product_order_file
        self.product_geometry_file = self.product_profile.product_geometry_file
        self.product_geometry: dict[str, Any] = dict(self.product_profile.product_geometry)
        self.safety_file = self.product_profile.safety_file
        self.safety_logic_path = Path("cais_spade_llm/safety/cca_safety_logic.json")
        self.robot_env = self.product_profile.robot_env
        # Manual instruction text provided at runtime overrides any file read.
        self.instruction_override = self.product_profile.instruction_override

        # Cache safety text for use during replanning
        self.safety_text: str = ""
        self.safety_text_has_requirements: bool = False

        self.precomputed_bundle: dict[str, Any] = dict(self.product_profile.precomputed_bundle or {})

        # Planner scaffolding
        base_plan_dir = Path("cais_spade_llm/monitor/plan")
        base_state_dir = Path("cais_spade_llm/monitor/state")
        # Legacy requirement snapshot path used only when no product order is provided.
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
        self.recovery_controller = ProductRecoveryController(self)
        self.recovery_controller.bind_methods()
        
        #keep resolved resource agents on the ProductAgent for caps overview
        self.resource_agents = planner_resources

        # Simple in-memory map of task_id -> latest status string so UI/debug tooling can query progress.
        self.task_states: dict[str, str] = {}
        self._pending_task_retry_ready_ids: set[str] = set()
        self._runtime_safety_fast_path_cache: dict[str, Any] = {}
        self._runtime_safety_history_cache: dict[str, Any] = {}

        # Sensor: camera module for post-placement verification
        self.camera = camera if camera is not None else CameraModule()

        # Runtime tracking for replanning context (PRODUCT STATE ONLY)
        self.part_tracker: dict[str, dict[str, Any]] = {}  # part_name -> {location, state, last_task}
        self.execution_timeline: list[dict[str, Any]] = []  # [{timestamp, task_id, status, ...}]
        # NOTE: Robot states are queried directly from ResourceAgents, not cached here
        self._plan_result_inbox_registered = False
        self._product_order_runtime_enabled = False
        self._product_order_commit_inflight = False
        self._product_order_commit_validation_request_id = ""
        self._product_order_commit_pending: dict[str, Any] = {}
        self._runtime_repair_inflight = False
        self._runtime_repair_fail_streak = 0
        self._runtime_repair_max_attempts = 3
        self._bridge_generation_mode = "auto"
        self._bridge_reasoning_mode = "multi_turn"
        self._runtime_bridge_mode = "pre_ran"
        self._runtime_bridge_validation_policy = "validated"
        self._runtime_bridge_archive_path = ""
        self._runtime_bridge_archive_label = ""
        self._orphaned_bridge_task_warning_ids: set[str] = set()
        self._generated_bridge_gazebo_verification_enabled = _env_flag_enabled(
            "CAIS_VERIFY_GENERATED_BRIDGE_IN_GAZEBO",
            "CAIS_GENERATED_BRIDGE_GAZEBO_VERIFICATION",
            default=False,
        )
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
                    min(int(precomputed_policy.get("auto_replan_max_attempts", 3) or 0), 10),
                )
            except Exception:
                self._runtime_repair_max_attempts = 3
            self._bridge_generation_mode = str(
                precomputed_policy.get("bridge_generation_mode", "auto") or "auto"
            ).strip().lower()
            if self._bridge_generation_mode not in {"auto", "manual"}:
                self._bridge_generation_mode = "auto"
            self._bridge_reasoning_mode = str(
                precomputed_policy.get("bridge_reasoning_mode", "multi_turn")
                or "multi_turn"
            ).strip().lower()
            if self._bridge_reasoning_mode in {
                "hybrid",
                "procedural",
                _LEGACY_PROCEDURAL_DES_BRIDGE_MODE,
            }:
                self._bridge_reasoning_mode = "multi_turn"
            if self._bridge_reasoning_mode != "multi_turn":
                self._bridge_reasoning_mode = "multi_turn"
            if "generated_bridge_gazebo_verification" in precomputed_policy:
                self._generated_bridge_gazebo_verification_enabled = bool(
                    precomputed_policy.get("generated_bridge_gazebo_verification")
                )

        self.logger.info(f"ProductAgent '{name}' initialized.")

    # ------------------------------------------------------------------ #
    # Persistence helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fsa_task_ids(fsa: dict[str, Any] | None) -> set[str]:
        transitions = (((fsa or {}).get("A") or {}).get("Tr") or [])
        return {
            str(transition.get("task_id") or "").strip()
            for transition in transitions
            if isinstance(transition, dict) and str(transition.get("task_id") or "").strip()
        }

    def _filter_runtime_context_completed_task_ids_for_fsa(
        self,
        runtime_context: dict[str, Any],
        fsa: dict[str, Any] | None,
    ) -> None:
        fsa_task_ids = self._fsa_task_ids(fsa)
        runtime_context["completed_task_ids"] = [
            str(task_id).strip()
            for task_id in (runtime_context.get("completed_task_ids") or [])
            if str(task_id).strip() and str(task_id).strip() in fsa_task_ids
        ]

    def _build_plan_validation_payload(
        self,
        *,
        skip_revalidation: bool = False,
        skip_recovery_safety_validation: bool = False,
        skip_offline_validation: bool | None = None,
        request_id: str | None = None,
        validation_scope: str | None = None,
        composition_backend: str | None = None,
        runtime_supervisor_mode: str | None = None,
    ):
        """Package plan + FSA for plan validation by the CCA."""
        if skip_offline_validation is not None:
            skip_revalidation = bool(skip_offline_validation)

        fsa = self.process_planner.global_fsa
        nodes = self.process_planner.nodes
        runtime_context = self._build_runtime_plan_context()
        validation_scope = str(validation_scope or "").strip()
        composition_backend = str(composition_backend or "").strip()
        if validation_scope:
            runtime_context["validation_scope"] = validation_scope
        if composition_backend:
            runtime_context["composition_backend"] = composition_backend
        runtime_supervisor_mode = str(runtime_supervisor_mode or "reactive").strip()
        if runtime_supervisor_mode:
            runtime_context["runtime_supervisor_mode"] = runtime_supervisor_mode
        if (
            validation_scope == "active_window"
            and composition_backend == "explicit_fsa_dfa"
        ):
            self._filter_runtime_context_completed_task_ids_for_fsa(
                runtime_context,
                fsa or {},
            )

        if fsa is None and not skip_revalidation:
            raise RuntimeError("Global FSA is None. Did you call save_global_fsa()?")

        payload = {
            "fsa": fsa or {},                # <-- upload FSA here
            "product_jid": str(self.jid),
            "plan": {"nodes": nodes},
            "runtime_context": runtime_context,
            "skip_revalidation": bool(skip_revalidation),
            "skip_recovery_safety_validation": bool(skip_recovery_safety_validation),
            "request_id": str(request_id or "").strip(),
        }
        if validation_scope:
            payload["validation_scope"] = validation_scope
        if composition_backend:
            payload["composition_backend"] = composition_backend
        if runtime_supervisor_mode:
            payload["runtime_supervisor_mode"] = runtime_supervisor_mode
        recovery_safety_result = (
            self._runtime_recovery_context.get("recovery_safety_result")
            if isinstance(getattr(self, "_runtime_recovery_context", None), dict)
            else None
        )
        if isinstance(recovery_safety_result, dict) and recovery_safety_result:
            payload["recovery_safety_result"] = deepcopy(recovery_safety_result)
        return payload

    def _safety_event_history_cache_key(self) -> tuple[tuple[Any, ...], bool]:
        timeline = list(getattr(self, "execution_timeline", []) or [])
        last_event = timeline[-1] if timeline and isinstance(timeline[-1], dict) else {}
        timeline_cursor = (
            len(timeline),
            id(last_event) if isinstance(last_event, dict) else 0,
            str((last_event or {}).get("task_id") or ""),
            str((last_event or {}).get("status") or ""),
            str((last_event or {}).get("timestamp") or ""),
        )

        classifier_available = False
        monitor_getter = getattr(self, "_runtime_safety_fast_path_monitor", None)
        if callable(monitor_getter):
            try:
                classifier_available = monitor_getter() is not None
            except Exception:
                classifier_available = False

        fast_path_cache = getattr(self, "_runtime_safety_fast_path_cache", None)
        classifier_identity = None
        if isinstance(fast_path_cache, dict):
            classifier_identity = fast_path_cache.get("cache_key")
        if classifier_identity is None:
            classifier_identity = ("classifier_unavailable",)

        return (timeline_cursor, classifier_identity, classifier_available), classifier_available

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

        safety_event_history_builder = getattr(self, "_build_safety_event_history", None)
        safety_event_history = (
            safety_event_history_builder()
            if callable(safety_event_history_builder)
            else []
        )
        return {
            "completed_task_ids": completed_task_ids,
            "running_task_ids": running_task_ids,
            "failed_task_ids": failed_task_ids,
            "safety_event_history": safety_event_history,
        }

    def _build_safety_event_history(self) -> list[dict[str, Any]]:
        """Build ordered task-event history for safety DFA progress across active FSA windows."""
        cache_key, classifier_available = self._safety_event_history_cache_key()
        history_cache = getattr(self, "_runtime_safety_history_cache", None)
        if not isinstance(history_cache, dict):
            history_cache = {}
            self._runtime_safety_history_cache = history_cache
        if history_cache.get("cache_key") == cache_key:
            return deepcopy(history_cache.get("history") or [])

        node_by_id = {
            str(node.get("id") or "").strip(): node
            for node in getattr(self.process_planner, "nodes", []) or []
            if isinstance(node, dict) and str(node.get("id") or "").strip()
        }
        history: list[dict[str, Any]] = []
        emitted: set[tuple[str, str]] = set()
        task_history_empty: dict[str, bool] = {}

        def _suffix_for_status(status: str) -> str:
            normalized = str(status or "").strip().lower()
            if normalized in {"dispatched", "accepted", "running"}:
                return "start"
            if normalized in {"completed", "finished"}:
                return "done"
            if normalized == "failed" or normalized.startswith("failed"):
                return "fail"
            return ""

        def _task_history_is_ap_empty(task_id: str, source_event: dict[str, Any]) -> bool:
            task_id = str(task_id or "").strip()
            if not classifier_available or not task_id:
                return False
            if task_id in task_history_empty:
                return task_history_empty[task_id]

            task_node = node_by_id.get(task_id) or {}
            params = dict(task_node.get("params") or {})
            params.setdefault("task_id", task_id)
            resource_jid = str(
                task_node.get("resource_jid")
                or source_event.get("resource_jid")
                or ""
            ).strip()
            function_name = str(
                task_node.get("function_name")
                or source_event.get("function_name")
                or ""
            ).strip()
            if resource_jid:
                params.setdefault("resource_jid", resource_jid)
            if function_name:
                params.setdefault("function_name", function_name)

            classifier = getattr(self, "_runtime_safety_ap_sets_for_task", None)
            if not callable(classifier) or not resource_jid or not function_name:
                task_history_empty[task_id] = False
                return False
            try:
                ap_sets = classifier(task_node, params)
            except Exception:
                ap_sets = None
            if ap_sets is None:
                task_history_empty[task_id] = False
                return False

            empty = not ap_sets.get("candidate_aps") and not ap_sets.get("predicted_state_aps")
            task_history_empty[task_id] = bool(empty)
            return bool(empty)

        def _append(task_id: str, suffix: str, source_event: dict[str, Any]) -> None:
            task_id = str(task_id or "").strip()
            suffix = str(suffix or "").strip()
            if not task_id or not suffix or (task_id, suffix) in emitted:
                return
            task_node = node_by_id.get(task_id) or {}
            params = dict(task_node.get("params") or {})
            params.setdefault("task_id", task_id)
            resource_jid = str(
                task_node.get("resource_jid")
                or source_event.get("resource_jid")
                or ""
            ).strip()
            function_name = str(task_node.get("function_name") or "").strip()
            part_name = str(self._tracked_part_name_for_task(task_node) or "").strip()
            emitted.add((task_id, suffix))
            history.append(
                {
                    "task_id": task_id,
                    "suffix": suffix,
                    "event": f"{task_id}.{suffix}",
                    "function_name": function_name,
                    "part_name": part_name,
                    "resource_jid": resource_jid,
                    "params": params,
                    "status": str(source_event.get("status") or "").strip(),
                    "timestamp": str(source_event.get("timestamp") or "").strip(),
                }
            )

        for event in self.execution_timeline:
            if not isinstance(event, dict):
                continue
            task_id = str(event.get("task_id") or "").strip()
            suffix = _suffix_for_status(str(event.get("status") or ""))
            if not task_id or not suffix:
                continue
            if _task_history_is_ap_empty(task_id, event):
                continue
            if suffix in {"done", "fail"} and (task_id, "start") not in emitted:
                _append(task_id, "start", event)
            _append(task_id, suffix, event)

        history_cache["cache_key"] = cache_key
        history_cache["history"] = deepcopy(history)
        return history


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

        t_recovery_safety = Template()
        t_recovery_safety.set_metadata("type", "recovery_safety_generated")
        self.add_behaviour(self._RecoverySafetyGeneratedInbox(), t_recovery_safety)

        # Plan executor (runs cycles, dispatches DAG tasks)
        # self.add_behaviour(self._PlanExecutor())

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _read_safety_text(self) -> str:
        """Compatibility wrapper for ProductProfile safety text loading."""
        return ProductProfile.read_safety_file(self.safety_file, logger=self.logger)

    @staticmethod
    def _safety_text_has_requirements(safety_text: str) -> bool:
        """Return True when the selected safety text contains a non-empty requirement line."""
        for raw_line in str(safety_text or "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("["):
                continue
            if line.startswith(("-", "*")):
                line = line[1:].strip()
            if line:
                return True
        return False
    


    def _geometry_for_part(self, part_name: str) -> dict[str, Any]:
        """Compatibility wrapper for ProductProfile per-part geometry lookup."""
        profile = getattr(self, "product_profile", None)
        if isinstance(profile, ProductProfile):
            return profile.geometry_for_part(part_name, product_geometry=self.product_geometry)
        return ProductProfile.geometry_for_part_from_geometry(part_name, self.product_geometry)

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
        body = {"task_id": task_id, "instruction": instruction}
        if phase_id:
            body["phase_id"] = phase_id
        msg = Message(to=to)
        # Metadata drives the SPADE routing/filtering logic downstream.
        msg.set_metadata("type", "task")
        msg.set_metadata("protocol", protocol)
        msg.body = json.dumps(body)
        return msg

    def _build_product_state(self) -> dict[str, Any]:
        """
        Build product-specific state for replanning context.

        NOTE: This only includes product/part state. System coordination state
        (robot states, running tasks, FSA states) is provided by CentralControllerAgent.

        Includes:
        - Part locations and states
        - Execution timeline
        - Requirements progress
        """
        requirements_status: dict[str, dict[str, Any]] = {}
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
            "product_order_runtime": deepcopy(
                getattr(self.process_planner, "last_product_order_artifact", {}) or {}
            ),
        }

    def _extract_requirement_text(self) -> str | None:
        """Compatibility wrapper for ProductProfile requirement text loading."""
        return ProductProfile.extract_requirement_file(
            self.product_specification_file,
            logger=self.logger,
        )

    def _load_product_order(self) -> dict[str, Any] | None:
        """Compatibility wrapper for ProductProfile product-order JSON loading."""
        profile = getattr(self, "product_profile", None)
        if isinstance(profile, ProductProfile):
            return profile.read_product_order(logger=self.logger)
        return ProductProfile.read_product_order_file(
            self.product_order_file,
            logger=self.logger,
        )

    def _ensure_plan_result_inbox(self) -> None:
        """Register runtime plan_safety_result inbox exactly once."""
        if self._plan_result_inbox_registered:
            return
        t_plan_result = Template()
        t_plan_result.set_metadata("type", "plan_safety_result")
        self.add_behaviour(self._PlanSafetyResultInbox(), t_plan_result)
        self._plan_result_inbox_registered = True























































    @staticmethod






























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

    async def _build_plan_from_product_order(
        self,
        product_order: dict[str, Any],
        safety_text: str = "",
    ):
        """Build a rolling runtime product-order skeleton from product-order JSON."""
        validate_product_order(product_order, self.product_geometry)
        self.process_planner.build_product_order_runtime_skeleton(
            product_order,
            safety_text=safety_text,
        )
        self._product_order_runtime_enabled = True
        self.process_planner.save(self.plan_path)
        return self.process_planner.nodes, self.process_planner.global_fsa

    def _product_order_runtime_active(self) -> bool:
        runtime = getattr(self.process_planner, "product_order_runtime", {}) or {}
        return bool(self._product_order_runtime_enabled and runtime.get("enabled"))

    def _product_order_unavailable_resource_jids(
        self,
        *,
        exclude_part: str = "",
    ) -> set[str]:
        """Return resources already reserved by uncompleted Product-order/runtime nodes."""
        unavailable: set[str] = set()
        final_statuses = {"completed", "finished"}
        exclude_part = str(exclude_part or "").strip()
        for node in getattr(self.process_planner, "nodes", []) or []:
            if not isinstance(node, dict) or node.get("type") != "task":
                continue
            if exclude_part and str(node.get("product_order_part") or "").strip() == exclude_part:
                continue
            resource_jid = str(node.get("resource_jid") or "").strip()
            if not resource_jid:
                continue
            task_id = str(node.get("id") or "").strip()
            status = str(self.task_states.get(task_id) or node.get("status") or "").strip().lower()
            if not status or status in final_statuses:
                continue
            unavailable.add(resource_jid)
        return unavailable

    def _product_order_part_nodes(self, part_name: str) -> list[dict[str, Any]]:
        part_name = str(part_name or "").strip()
        return [
            node
            for node in getattr(self.process_planner, "nodes", []) or []
            if isinstance(node, dict)
            and node.get("type") == "task"
            and str(node.get("product_order_part") or "").strip() == part_name
        ]

    def _product_order_part_chain_completed(self, part_name: str) -> bool:
        nodes = self._product_order_part_nodes(part_name)
        return bool(nodes) and all(
            str(node.get("status") or "").strip().lower() in {"completed", "finished"}
            for node in nodes
        )

    def _mark_product_order_part_completed_if_ready(self, part_name: str) -> bool:
        if not self._product_order_runtime_active():
            return False
        part_name = str(part_name or "").strip()
        if not part_name or not self._product_order_part_chain_completed(part_name):
            return False
        self.process_planner.mark_product_order_part_completed(part_name)
        return True

    async def _maybe_commit_product_order_runtime_parts(self, behaviour: CyclicBehaviour) -> bool:
        """Commit ready Product-order parts, validate the updated FSA, and hold dispatch until CCA OK."""
        if not self._product_order_runtime_active():
            return False
        if self._product_order_commit_inflight:
            return False

        ready_parts = self.process_planner.ready_product_order_parts()
        if not ready_parts:
            return False

        unavailable = self._product_order_unavailable_resource_jids()
        committed: list[dict[str, Any]] = []
        for part_name in ready_parts:
            try:
                record = self.process_planner.commit_product_order_part(
                    part_name,
                    unavailable_resource_jids=unavailable,
                    status="pending_validation",
                )
            except ValueError as exc:
                message = str(exc)
                if "no available product bid resources" in message:
                    self.logger.debug(
                        "[Product] Product-order part %s is ready but no resource is available for bidding yet.",
                        part_name,
                    )
                    continue
                self.logger.error(
                    "[Product] Product-order commit failed for %s: %s",
                    part_name,
                    exc,
                )
                self._set_plan_safety_alert(
                    stage="runtime",
                    message=f"{self.agent_name}: product-order commit failed for {part_name} ({exc}).",
                    retries_used=0,
                    retries_max=0,
                    violations=[],
                    paused=False,
                )
                continue
            committed.append(record)
            resource_jid = str(record.get("resource_jid") or "").strip()
            if resource_jid:
                unavailable.add(resource_jid)

        if not committed:
            return False

        try:
            self.process_planner.recompile_committed_product_order_fsa()
            self.process_planner.save_global_fsa(self.global_fsa_path)
        except Exception as exc:
            parts = [str(record.get("part") or "").strip() for record in committed]
            removed = self.process_planner.rollback_product_order_committed_parts(parts)
            self.logger.exception(
                "[Product] Product-order committed FSA compile failed; rolled back parts=%s.",
                removed,
            )
            self._set_plan_safety_alert(
                stage="runtime",
                message=f"{self.agent_name}: product-order committed FSA compile failed ({exc}).",
                retries_used=0,
                retries_max=0,
                violations=[],
                paused=False,
            )
            await asyncio.to_thread(self._persist_plan_snapshot)
            await asyncio.to_thread(self._persist_product_state)
            return False

        request_id = f"product_order_commit_{uuid.uuid4().hex}"
        task_ids = [
            str(task_id or "").strip()
            for record in committed
            for task_id in (record.get("task_ids") or [])
            if str(task_id or "").strip()
        ]
        parts = [
            str(record.get("part") or "").strip()
            for record in committed
            if str(record.get("part") or "").strip()
        ]
        self._product_order_commit_inflight = True
        self._product_order_commit_validation_request_id = request_id
        self._product_order_commit_pending = {
            "request_id": request_id,
            "parts": parts,
            "task_ids": task_ids,
            "created_at_utc": self._utc_now_iso(),
        }

        payload = self._build_plan_validation_payload(
            request_id=request_id,
            validation_scope="active_window",
            composition_backend="explicit_fsa_dfa",
            runtime_supervisor_mode="reactive",
        )
        msg = Message(to=self.cca_jid)
        msg.set_metadata("type", "plan_safety_check")
        msg.body = json.dumps(payload)
        await send_agent_message(
            behaviour,
            msg,
            transport_label="product_order_commit_plan_check",
        )
        self.logger.info(
            "[Product] Product-order committed part(s) pending CCA validation request_id=%s parts=%s task_ids=%s.",
            request_id,
            parts,
            task_ids,
        )
        await asyncio.to_thread(self._persist_plan_snapshot)
        await asyncio.to_thread(self._persist_product_state)
        await asyncio.to_thread(self._persist_resource_state)
        return True

    async def _handle_product_order_commit_validation_result(
        self,
        *,
        ok: bool,
        violations: list[dict[str, Any]],
        request_id: str,
    ) -> bool:
        """Consume CCA validation replies for rolling Product-order commits."""
        if not self._product_order_runtime_active():
            return False
        request_id = str(request_id or "").strip()
        expected = str(self._product_order_commit_validation_request_id or "").strip()
        if request_id != expected:
            if request_id.startswith("product_order_commit_"):
                self.logger.warning(
                    "[Product] Ignoring stale Product-order commit validation result request_id=%s expected=%s.",
                    request_id,
                    expected or "<none>",
                )
                return True
            return False

        pending = dict(self._product_order_commit_pending or {})
        parts = [
            str(part or "").strip()
            for part in (pending.get("parts") or [])
            if str(part or "").strip()
        ]
        if ok:
            validated = self.process_planner.mark_product_order_commit_validated(parts)
            self._product_order_commit_inflight = False
            self._product_order_commit_validation_request_id = ""
            self._product_order_commit_pending = {}
            self._clear_plan_safety_alert()
            self.logger.info(
                "[Product] Product-order commit validation PASSED request_id=%s parts=%s.",
                request_id,
                validated,
            )
            await asyncio.to_thread(self._persist_plan_snapshot)
            await asyncio.to_thread(self._persist_product_state)
            await asyncio.to_thread(self._persist_resource_state)
            return True

        removed = self.process_planner.rollback_product_order_committed_parts(parts)
        if self.process_planner.nodes:
            try:
                self.process_planner.recompile_committed_product_order_fsa()
                self.process_planner.save_global_fsa(self.global_fsa_path)
            except Exception:
                self.logger.exception(
                    "[Product] Product-order rollback FSA recompile failed after commit validation failure."
                )
                self.process_planner.global_fsa = None
        else:
            self.process_planner.global_fsa = None
        self._product_order_commit_inflight = False
        self._product_order_commit_validation_request_id = ""
        self._product_order_commit_pending = {}
        alert = self._set_plan_safety_alert(
            stage="runtime",
            message=(
                f"{self.agent_name}: product-order commit validation failed for "
                f"{', '.join(parts) or '<unknown>'}; rolled back {', '.join(removed) or '<none>'}."
            ),
            retries_used=0,
            retries_max=0,
            violations=violations,
            paused=False,
        )
        self.logger.warning(
            "[Product] Product-order commit validation FAILED request_id=%s parts=%s removed=%s violations=%d.",
            request_id,
            parts,
            removed,
            len(violations),
        )
        await asyncio.to_thread(self._persist_plan_snapshot)
        await asyncio.to_thread(self._persist_product_state)
        await asyncio.to_thread(self._persist_resource_state)
        return bool(alert is not None or True)

    async def _rebid_product_order_pending_assignment_if_resource_unavailable(
        self,
        task_node: dict[str, Any],
        behaviour: CyclicBehaviour,
    ) -> bool:
        """Rollback a not-yet-dispatched Product-order part if its resource becomes unavailable."""
        if not self._product_order_runtime_active() or self._product_order_commit_inflight:
            return False
        if not isinstance(task_node, dict):
            return False
        if str(task_node.get("status") or "").strip() != "pending":
            return False
        part_name = str(task_node.get("product_order_part") or "").strip()
        resource_jid = str(task_node.get("resource_jid") or "").strip()
        if not part_name or not resource_jid:
            return False
        unavailable = self._product_order_unavailable_resource_jids(exclude_part=part_name)
        if resource_jid not in unavailable:
            return False
        removed = self.process_planner.rollback_product_order_committed_parts([part_name])
        if not removed:
            return False
        self.logger.info(
            "[Product] Product-order assignment for %s was rolled back before dispatch because %s became unavailable; rebidding.",
            part_name,
            resource_jid,
        )
        await asyncio.to_thread(self._persist_plan_snapshot)
        await asyncio.to_thread(self._persist_product_state)
        return await self._maybe_commit_product_order_runtime_parts(behaviour)

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




    # --------------------------------------------------------------------- #
    # Behaviours
    # --------------------------------------------------------------------- #
    class _Kickoff(OneShotBehaviour):
        async def run(self):
            agent: ProductAgent = self.agent
            retries_used = 0
            used_precomputed = False
            max_retries = 3

            try:
                product_order = agent._load_product_order()
                instruction = None if product_order else agent._extract_requirement_text()
                safety_text = agent._read_safety_text()
                agent.safety_text = safety_text
                agent.safety_text_has_requirements = agent._safety_text_has_requirements(safety_text)
                agent.runtime_repair_state = "idle"
                agent._runtime_repair_fail_streak = 0
                agent._clear_plan_safety_alert()
                agent._clear_runtime_recovery()

                used_precomputed = agent._load_precomputed_plan_bundle()
                max_retries = 0 if used_precomputed else 3
                if not used_precomputed:
                    if product_order:
                        await agent._build_plan_from_product_order(product_order, safety_text)
                    elif not instruction:
                        raise RuntimeError("no product requirement text available for startup planning")
                    else:
                        await agent._build_plan(instruction, safety_text)

                if (
                    product_order
                    and not used_precomputed
                    and agent._product_order_runtime_active()
                    and not agent.process_planner.nodes
                ):
                    message = (
                        f"{agent.agent_name}: product-order runtime skeleton ready; "
                        "rolling Product bidding will validate committed parts at runtime."
                    )
                    agent.logger.info(
                        "[Product] Product-order runtime skeleton ready. Startup FSA validation deferred until first committed part."
                    )
                    agent._clear_plan_safety_alert()
                    agent._ensure_plan_result_inbox()
                    agent.add_behaviour(agent._PlanExecutor())
                    await asyncio.to_thread(agent._persist_plan_snapshot)
                    await asyncio.to_thread(agent._persist_product_state)
                    await asyncio.to_thread(agent._persist_resource_state)
                    agent._set_kickoff_result(
                        success=True,
                        message=message,
                        retries_used=0,
                        retries_max=0,
                        violations=[],
                    )
                    return

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
            agent: ProductAgent = self.agent  # type: ignore
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

            existing_task_node = None
            for node in agent.process_planner.nodes:
                if node.get("id") == task_id:
                    existing_task_node = node
                    break

            if agent._should_ignore_stale_recovery_macro_ack(
                task_node=existing_task_node,
                incoming_status=str(status),
            ):
                agent.logger.warning(
                    "[Product] Ignoring stale ACK for completed recovery task %s: current_status=%s incoming_status=%s",
                    task_id,
                    str((existing_task_node or {}).get("status") or "").strip() or "<unknown>",
                    str(status or "").strip() or "<unknown>",
                )
                return
            current_status = str((existing_task_node or {}).get("status") or "").strip()
            if _ack_status_is_regression(current_status, str(status)):
                agent.logger.warning(
                    "[Product] Ignoring regressive ACK for %s: current_status=%s incoming_status=%s",
                    task_id,
                    current_status or "<unknown>",
                    str(status or "").strip() or "<unknown>",
                )
                return

            # 1) Keep existing state map for UI/debug
            agent.task_states[task_id] = status

            # 2) ALSO update node status in the planner DAG if exists
            updated_node = False
            task_node = existing_task_node
            if task_node:
                # Map RA status → planner status; for now use it directly
                if task_node.get("status") != status:
                    task_node["status"] = status
                    updated_node = True

            # 3) Track execution timeline
            from datetime import datetime, timezone
            agent.execution_timeline.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": task_id,
                "status": status,
                "resource_jid": str(msg.sender),
            })

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

            if task_node and str(status).strip().lower() in {"completed", "finished"}:
                product_order_part = str(task_node.get("product_order_part") or "").strip()
                if product_order_part and agent._mark_product_order_part_completed_if_ready(product_order_part):
                    updated_node = True

            if updated_node and not handled_bridge_ack and _should_persist_ack_state(task_node, str(status)):
                await asyncio.to_thread(agent._persist_plan_snapshot)
                await asyncio.to_thread(agent._persist_product_state)
                await asyncio.to_thread(agent._persist_resource_state)

            agent.logger.info(
                f"[Product] ACK ({task_id}) status='{status}' from={msg.sender}"
            )

            if (
                task_node
                and str(status).strip().lower() == "blocked"
                and str(task_id).strip()
                in getattr(agent, "_pending_task_retry_ready_ids", set())
            ):
                reactivated = agent._handle_task_retry_ready([str(task_id).strip()])
                if reactivated:
                    agent.logger.info(
                        "[Product] Requeued %d blocked task(s) after delayed blocked ACK matched earlier CCA transient safety clear: %s",
                        reactivated,
                        str(task_id).strip(),
                    )
                    await asyncio.to_thread(agent._persist_plan_snapshot)
                    await asyncio.to_thread(agent._persist_product_state)
                    await asyncio.to_thread(agent._persist_resource_state)

    class _ReplanInbox(CyclicBehaviour):
        """Handle online replan requests from the CCA."""

        async def run(self):
            agent: ProductAgent = self.agent  # type: ignore
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
            agent: ProductAgent = self.agent  # type: ignore
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
            request_id = str(payload.get("request_id") or "").strip()

            if await agent._handle_product_order_commit_validation_result(
                ok=ok,
                violations=violations,
                request_id=request_id,
            ):
                return

            if await agent._handle_runtime_plan_validation_result(
                ok=ok,
                violations=violations,
                request_id=request_id,
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
                    transport_label="product_runtime_plan_check",
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
            agent: ProductAgent = self.agent  # type: ignore
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

    class _RecoverySafetyGeneratedInbox(CyclicBehaviour):
        """Handle recovery_safety_generated replies from CCA."""

        async def run(self):
            agent: ProductAgent = self.agent  # type: ignore
            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            try:
                payload = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Product] Malformed recovery_safety_generated body.")
                return

            await agent._handle_recovery_safety_generated_result(payload)

    class _PlanExecutor(CyclicBehaviour):
        """
        Periodically checks the DAG for the next ready task and dispatches it
        as a SPADE message to the resource agent.
        """

        async def run(self):
            agent: ProductAgent = self.agent  # type: ignore

            # No resources? nothing to do
            if not agent.resource_jids:
                return

            if agent._runtime_recovery_blocks_execution():
                await asyncio.sleep(0.05)
                return

            async def _maybe_commit_product_order_runtime_parts() -> bool:
                helper = getattr(agent, "_maybe_commit_product_order_runtime_parts", None)
                if not callable(helper):
                    return False
                return bool(await helper(self))

            async def _dispatch_task_node(task_node: dict[str, Any]) -> bool:
                task_id = str(task_node.get("id") or "").strip()
                planner_status = str(task_node.get("status") or "").strip().lower()
                tracked_status = str(agent.task_states.get(task_id) or "").strip().lower()
                current_sequence_for_guard = agent._active_bridge_sequence()
                bridge_recorded_status = ""
                if isinstance(current_sequence_for_guard, dict):
                    completed_task_ids = set(
                        agent._bridge_sequence_task_ids(
                            current_sequence_for_guard.get("completed_bridge_task_ids") or []
                        )
                    )
                    dispatched_task_ids = set(
                        agent._bridge_sequence_task_ids(
                            current_sequence_for_guard.get("dispatched_bridge_task_ids") or []
                        )
                    )
                    if task_id in completed_task_ids:
                        bridge_recorded_status = "completed"
                    elif task_id in dispatched_task_ids:
                        bridge_recorded_status = "dispatched"
                effective_tracked_status = bridge_recorded_status or tracked_status
                if planner_status == "pending" and effective_tracked_status in {
                    "accepted",
                    "running",
                    "dispatched",
                    "completed",
                    "blocked",
                }:
                    task_node["status"] = effective_tracked_status
                    agent.logger.warning(
                        "[Product] Suppressing duplicate dispatch for %s: planner_status=pending tracked_status=%s",
                        task_id,
                        effective_tracked_status,
                    )
                    return False

                # Prefer resource_jid chosen by the planner.
                to = task_node.get("resource_jid")
                if not to:
                    # Fallback: first configured resource JID
                    to = agent.resource_jids[0]
                    agent.logger.warning(
                        "[Product] Task %s has no resource_jid, falling back to %s",
                        task_node.get("id"),
                        to,
                    )
                to = str(to or "").strip()
                active_statuses = {"dispatched", "accepted", "running"}
                active_same_resource_task_ids: list[str] = []
                planner_nodes = getattr(
                    getattr(agent, "process_planner", None),
                    "nodes",
                    [],
                ) or []
                for other_node in planner_nodes:
                    if not isinstance(other_node, dict):
                        continue
                    other_task_id = str(other_node.get("id") or "").strip()
                    if other_task_id and other_task_id == task_id:
                        continue
                    if str(other_node.get("resource_jid") or "").strip() != to:
                        continue
                    other_status = str(other_node.get("status") or "").strip().lower()
                    other_tracked_status = str(
                        agent.task_states.get(other_task_id) or ""
                    ).strip().lower()
                    if (
                        other_status in active_statuses
                        or other_tracked_status in active_statuses
                    ):
                        active_same_resource_task_ids.append(other_task_id or "<unknown>")
                if active_same_resource_task_ids:
                    agent.logger.info(
                        "[Product] Dispatch guard suppressed task %s for resource_jid=%s while active task(s) are running on that resource: %s.",
                        task_id or "<unknown>",
                        to or "<unknown>",
                        active_same_resource_task_ids,
                    )
                    return False

                # Build the instruction for the RobotAgent from the DAG node.
                # Recovery macro primitive steps own their destination intent.
                try:
                    params = agent._dispatch_params_for_task_node(task_node)
                except RuntimeError as exc:
                    if "Recovery Safety Check dispatch blocked" not in str(exc):
                        raise
                    agent.logger.error(
                        "[Product] Dispatch blocked for %s: %s",
                        task_id or "<unknown>",
                        exc,
                    )
                    await asyncio.to_thread(agent._persist_product_state)
                    return False

                instruction = {
                    "function_name": task_node.get("function_name"),
                    "params": params,
                }

                msg = agent._compose_task_msg(
                    to=to,
                    task_id=task_id,
                    instruction=instruction,
                    phase_id=None,
                )

                # Mark as "dispatched" (still waiting for ACK to flip to "completed")
                task_node["status"] = "dispatched"
                agent.task_states[task_id] = "dispatched"

                is_bridge_task = (
                    str(task_node.get("function_name") or "").strip() == "execute_recovery_macro"
                    and str(task_node.get("bridge_sequence_id") or "").strip()
                )
                if is_bridge_task:
                    current_sequence = agent._active_bridge_sequence()
                    if (
                        isinstance(current_sequence, dict)
                        and str(current_sequence.get("bridge_sequence_id") or "").strip()
                        == str(task_node.get("bridge_sequence_id") or "").strip()
                    ):
                        repair_task_id = str(current_sequence.get("repair_task_id") or "").strip()
                        repair_target_task_id = str(
                            current_sequence.get("repair_target_task_id") or ""
                        ).strip()
                        if repair_task_id and repair_task_id == task_id:
                            agent.logger.info(
                                "[Product] Dispatching runtime DES repair %s before restored task %s.",
                                task_id,
                                repair_target_task_id or "<unknown>",
                            )
                        next_sequence = agent._bridge_sequence_with_dispatched_task(
                            current_sequence,
                            task_id=task_id,
                        )
                        if isinstance(next_sequence, dict):
                            agent._set_runtime_recovery(
                                message=str(agent.runtime_recovery.get("message", "") or "").strip(),
                                active_bridge_sequence=next_sequence,
                            )

                if isinstance(msg, Message):
                    await send_agent_message(
                        self,
                        msg,
                        transport_label="product_task",
                    )
                else:
                    await self.send(msg)
                agent.logger.info(f"[Product] Dispatched task {task_id} -> {to} ({instruction})")
                return is_bridge_task

            task_node = agent._next_dispatchable_task_node()
            if (
                task_node
                and str(task_node.get("function_name") or "").strip() != "execute_recovery_macro"
            ):
                if await _maybe_commit_product_order_runtime_parts():
                    await asyncio.sleep(0.01)
                    return
            if not task_node and agent._active_bridge_blocks_nominal_dispatch():
                agent.logger.debug(
                    "[Product] Active bridge sequence is executing; suppressing nominal DAG dispatch."
                )
                await asyncio.sleep(0.05)
                return
            if not task_node:
                if await _maybe_commit_product_order_runtime_parts():
                    await asyncio.sleep(0.01)
                    return
                await asyncio.sleep(0.05)
                return

            rebid_helper = getattr(
                agent,
                "_rebid_product_order_pending_assignment_if_resource_unavailable",
                None,
            )
            if callable(rebid_helper) and await rebid_helper(task_node, self):
                await asyncio.sleep(0.01)
                return

            is_bridge_task = await _dispatch_task_node(task_node)
            if is_bridge_task:
                await asyncio.to_thread(agent._persist_plan_snapshot)
                await asyncio.to_thread(agent._persist_product_state)

            # Keep the loop responsive in Gazebo without busy-spinning.
            await asyncio.sleep(0.01)

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

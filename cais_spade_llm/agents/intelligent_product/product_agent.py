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
from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.product.profile import ProductProfile
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
_CASE3_PREPROGRAMMED_SCENARIO_ID = "case3_llm_bridge"
_CASE3_PREPROGRAMMED_REQUIREMENT_FILES = frozenset({"case3_two_arm_llm_bridge.txt"})
_LEGACY_PROCEDURAL_DES_BRIDGE_MODE = "procedural" + "_des_v1"


def _env_flag_enabled(*names: str, default: bool = False) -> bool:
    for name in names:
        token = str(os.environ.get(name) or "").strip().lower()
        if not token:
            continue
        return token in {"1", "true", "yes", "on"}
    return bool(default)


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
        self.product_profile = ProductProfile(
            name=name,
            product_specification_file=product_specification_file,
            product_geometry_file=product_geometry_file,
            safety_file=safety_file,
            instruction_override=instruction_override,
            precomputed_bundle=dict(precomputed_bundle or {}),
            logger=self.logger,
        )
        self.product_specification_file = self.product_profile.product_specification_file
        self.product_geometry_file = self.product_profile.product_geometry_file
        self.product_geometry: Dict[str, Any] = dict(self.product_profile.product_geometry)
        self.safety_file = self.product_profile.safety_file
        self.robot_env = self.product_profile.robot_env
        # Manual instruction text provided at runtime overrides any file read.
        self.instruction_override = self.product_profile.instruction_override

        # Cache safety text for use during replanning
        self.safety_text: str = ""

        self.precomputed_bundle: dict[str, Any] = dict(self.product_profile.precomputed_bundle or {})

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
        self._runtime_repair_max_attempts = 3
        self._bridge_generation_mode = "auto"
        self._bridge_reasoning_mode = "multi_turn"
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
        if msg.empty_sender():
            msg.sender = str(self.jid)

        if self.container.has_agent(str(msg.to)):
            self.container.get_agent(str(msg.to)).dispatch(msg)
        else:
            if self.client is None:
                raise RuntimeError("agent client is not connected")
            slixmpp_msg = msg.prepare(self.client)
            slixmpp_msg.send()

        msg.sent = True
        self.traces.append(msg, category=trace_category)

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
            "fixture_replay": None,
            "generated_code_verification": None,
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
        elif status in {"bridge_ready", "human_required", "generated_bridge_verified"}:
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
        fixture_replay: dict[str, Any] | None | object = _UNSET,
        generated_code_verification: dict[str, Any] | None | object = _UNSET,
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
        if fixture_replay is not _UNSET:
            current["fixture_replay"] = (
                deepcopy(fixture_replay)
                if isinstance(fixture_replay, dict)
                else None
            )
        if generated_code_verification is not _UNSET:
            current["generated_code_verification"] = (
                deepcopy(generated_code_verification)
                if isinstance(generated_code_verification, dict)
                else None
            )
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

    @staticmethod
    def _normalized_xyz_pose(value: Any) -> dict[str, float] | None:
        if not isinstance(value, dict) or not {"x", "y", "z"} <= set(value.keys()):
            return None
        try:
            return {
                "x": float(value["x"]),
                "y": float(value["y"]),
                "z": float(value["z"]),
            }
        except (TypeError, ValueError):
            return None

    def _observed_pose_for_part(self, part_name: str) -> dict[str, float] | None:
        part_key = str(part_name or "").strip()
        if not part_key:
            return None

        tracker_entry = dict((getattr(self, "part_tracker", {}) or {}).get(part_key) or {})
        for key in ("observed_pose", "pose", "position", "dropped_location"):
            pose = self._normalized_xyz_pose(tracker_entry.get(key))
            if pose is not None:
                return pose

        derived_observations = (
            dict(self._derive_preprogrammed_part_observations() or {})
            if hasattr(self, "_runtime_recovery_context")
            else {}
        )
        pose = self._normalized_xyz_pose(derived_observations.get(part_key))
        if pose is not None:
            return pose

        prepared_request = dict(
            (getattr(self, "_runtime_recovery_context", {}) or {}).get("prepared_bridge_request")
            or {}
        )
        prepared_parts = dict(prepared_request.get("part_tracker") or {})
        prepared_entry = dict(prepared_parts.get(part_key) or {})
        for key in ("observed_pose", "pose", "position", "dropped_location"):
            pose = self._normalized_xyz_pose(prepared_entry.get(key))
            if pose is not None:
                return pose
        return None

    def _tracked_pose_for_part_at_location(
        self,
        *,
        part_name: str,
        source_location: str,
    ) -> dict[str, float] | None:
        part_key = str(part_name or "").strip()
        normalized_source = str(source_location or "").strip()
        if not part_key or not normalized_source:
            return None

        tracker_entry = dict((getattr(self, "part_tracker", {}) or {}).get(part_key) or {})
        tracked_location = str(tracker_entry.get("location") or "").strip()
        if tracked_location != normalized_source:
            return None

        for key in ("position", "pose", "observed_pose", "dropped_location"):
            pose = self._normalized_xyz_pose(tracker_entry.get(key))
            if pose is not None:
                return pose
        return None

    def _support_surface_place_pose_from_observations(
        self,
        *,
        part_name: str,
        params: dict[str, Any],
        observations: dict[str, Any] | None,
    ) -> tuple[dict[str, float] | None, str]:
        if not isinstance(observations, dict):
            return None, ""

        destination_location = str(params.get("destination_location") or "").strip()
        if not destination_location:
            return None, ""

        event_facts = dict(observations.get("event_facts") or {})
        place_targets = dict(event_facts.get("place_targets") or {})
        place_target = dict(place_targets.get(str(part_name or "").strip()) or {})
        if not place_target:
            return None, ""

        target_reference = dict(place_target.get("target_reference") or {})
        target_point = str(target_reference.get("target_point") or "").strip()
        surface_role = str(target_reference.get("surface_role") or "").strip()
        if target_point != "part_origin" and surface_role != "support_surface":
            return None, ""

        target_origin_pose = dict(place_target.get("target_origin_pose") or {})
        normalized_target_origin_pose = self._normalized_xyz_pose(target_origin_pose)
        if normalized_target_origin_pose is not None:
            pose_source = str(target_origin_pose.get("source") or "").strip() or "target_origin_pose"
            return normalized_target_origin_pose, pose_source

        try:
            return {
                "x": float(place_target["slot_x"]),
                "y": float(place_target["slot_y"]),
                "z": float(place_target["place_part_origin_z"]),
            }, "place_targets.derived_part_origin"
        except (KeyError, TypeError, ValueError):
            return None, ""

    def _part_geometry_for_pick_context(self, part_name: str) -> dict[str, Any]:
        geometry = self._geometry_for_part(part_name)
        if not isinstance(geometry, dict):
            return {}
        part_geometry: dict[str, Any] = {}
        for key in ("part_height_m", "model_name"):
            if key in geometry:
                part_geometry[key] = deepcopy(geometry[key])
        return part_geometry

    def _enrich_observed_pose_recovery_params(self, params: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(params or {})
        source_location = str(enriched.get("origin_resource_location") or "").strip()
        if source_location != "observed_pose" and not source_location.endswith("_observed_pose"):
            return enriched
        part_name = str(enriched.get("part_name") or "").strip()
        if not part_name:
            return enriched
        if not isinstance(enriched.get("observed_pose"), dict):
            observed_pose = self._observed_pose_for_part(part_name)
            if observed_pose is not None:
                enriched["observed_pose"] = observed_pose
        if "part_geometry" not in enriched:
            part_geometry = self._part_geometry_for_pick_context(part_name)
            if part_geometry:
                enriched["part_geometry"] = part_geometry
        return enriched

    def _dispatch_params_for_task_node(self, task_node: dict[str, Any]) -> dict[str, Any]:
        """Build task params for dispatch without overriding recovery primitive intent."""
        params = dict(task_node.get("params", {}))
        part_name = params.get("part_name")
        function_name = str(task_node.get("function_name") or "").strip()
        if function_name == "execute_recovery_macro":
            return self._enrich_observed_pose_recovery_params(params)
        if part_name and function_name != "execute_recovery_macro":
            geo = self._geometry_for_part(part_name)
            if geo:
                params["product_geometry"] = geo
        return params

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

        support_surface_pose, support_surface_pose_source = (
            self._support_surface_place_pose_from_observations(
                part_name=part_name,
                params=params,
                observations=observations,
            )
            if status == "completed"
            else (None, "")
        )
        if support_surface_pose is not None:
            entry["position"] = deepcopy(support_surface_pose)
            entry["pose_source"] = support_surface_pose_source
            destination_location = str(
                params.get("destination_location") or entry.get("location") or ""
            ).strip()
            if destination_location:
                entry["location"] = destination_location

        # Preserve origin info when a part transitions to in_gripper so
        # recovery planners know where to return it.
        if transition.get("state") == "in_gripper":
            origin = str(params.get("origin_resource_location") or "").strip()
            if origin:
                entry["origin_resource_location"] = origin
            model_name = self._model_name_from_mapping(params, part_name)
            if model_name:
                entry["model_name"] = model_name
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
        """Compatibility wrapper for ProductProfile safety text loading."""
        return ProductProfile.read_safety_file(self.safety_file, logger=self.logger)
    
    def _read_spec_text(self) -> Optional[str]:
        """Compatibility wrapper for ProductProfile spec text loading."""
        return ProductProfile.read_spec_file(
            self.product_specification_file,
            instruction_override=self.instruction_override,
            logger=self.logger,
        )

    def _load_product_geometry(self, geometry_file: Optional[str]) -> Dict[str, Any]:
        """Compatibility wrapper for ProductProfile geometry loading."""
        return ProductProfile.load_product_geometry(
            geometry_file,
            robot_env=getattr(self, "robot_env", None),
            logger=self.logger,
        )

    def _geometry_for_part(self, part_name: str) -> Dict[str, Any]:
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
        """Compatibility wrapper for ProductProfile requirement text loading."""
        return ProductProfile.extract_requirement_file(
            self.product_specification_file,
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

    def _bridge_is_verification_only(
        self,
        active_bridge_sequence: dict[str, Any] | None,
    ) -> bool:
        return bool(self._bridge_execution_policy(active_bridge_sequence).get("verification_only"))

    def _active_bridge_blocks_nominal_dispatch(self) -> bool:
        active_bridge_sequence = self._active_bridge_sequence()
        if not active_bridge_sequence:
            return False
        state = str(active_bridge_sequence.get("state") or "").strip().lower()
        return state in {
            "approved",
            "executing",
            "continuation_blocked",
            "human_required",
            "failed",
        }

    def _next_dispatchable_task_node(self) -> dict[str, Any] | None:
        return self._select_runtime_event()

    def _active_bridge_next_ready_task(self) -> dict[str, Any] | None:
        """Return the next pending active bridge task, prioritizing recovery over nominal work."""
        active_bridge_sequence = self._active_bridge_sequence()
        if not active_bridge_sequence:
            return None
        sequence_id = str(active_bridge_sequence.get("bridge_sequence_id") or "").strip()
        if not sequence_id:
            return None
        bridge_task_ids = [
            str(task_id or "").strip()
            for task_id in (active_bridge_sequence.get("bridge_task_ids") or [])
            if str(task_id or "").strip()
        ]
        if not bridge_task_ids:
            bridge_task_ids = [
                str(node.get("id") or "").strip()
                for node in self.process_planner._bridge_sequence_nodes(sequence_id)
                if str(node.get("id") or "").strip()
            ]
        bridge_task_id_set = set(bridge_task_ids)
        failed_task_id = str(
            active_bridge_sequence.get("failed_task_id")
            or self.runtime_recovery.get("failed_task_id")
            or ""
        ).strip()

        for task_id in bridge_task_ids:
            node = self.process_planner._find_node(task_id)
            if not isinstance(node, dict):
                continue
            if str(node.get("bridge_sequence_id") or "").strip() != sequence_id:
                continue
            if str(node.get("status") or "").strip() != "pending":
                continue
            if self._bridge_task_predecessors_ready(
                node,
                bridge_task_ids=bridge_task_id_set,
                failed_task_id=failed_task_id,
            ):
                return node
            return None
        return None

    def _bridge_task_predecessors_ready(
        self,
        task_node: dict[str, Any],
        *,
        bridge_task_ids: set[str],
        failed_task_id: str = "",
    ) -> bool:
        for pred_id in [
            str(pred or "").strip()
            for pred in (task_node.get("predecessors") or [])
            if str(pred or "").strip()
        ]:
            pred_node = self.process_planner._find_node(pred_id)
            if not isinstance(pred_node, dict):
                return False
            pred_status = str(pred_node.get("status") or "").strip().lower()
            if pred_status == "completed":
                continue
            if pred_id in bridge_task_ids:
                return False
            if pred_id == failed_task_id and pred_status.startswith("failed"):
                continue
            return False
        return True

    def _select_runtime_event(self) -> dict[str, Any] | None:
        """Select the next DES-style runtime event: bridge first, then guarded nominal."""
        bridge_task = self._active_bridge_next_ready_task()
        if bridge_task:
            return bridge_task
        if self._active_bridge_blocks_nominal_dispatch():
            return None

        graph_ready = self._graph_ready_task_nodes()
        if not graph_ready:
            return None

        plant_state = self._build_runtime_plant_state(
            resource_jids=[
                str(node.get("resource_jid") or "").strip()
                for node in graph_ready
                if str(node.get("resource_jid") or "").strip()
            ]
        )
        plant_enabled: list[dict[str, Any]] = []
        disabled_frontier: list[dict[str, Any]] = []
        for node in graph_ready:
            violations = self._event_guard_violations(
                node,
                plant_state=plant_state,
                enforce_unknown=False,
            )
            if violations:
                disabled_frontier.append(
                    {
                        "task_id": str(node.get("id") or "").strip(),
                        "function_name": str(node.get("function_name") or "").strip(),
                        "resource_jid": str(node.get("resource_jid") or "").strip(),
                        "part_name": self._tracked_part_name_for_task(node),
                        "guard_violations": violations,
                    }
                )
            else:
                plant_enabled.append(node)

        self._record_runtime_des_trace(
            graph_ready_event_ids=[
                str(node.get("id") or "").strip()
                for node in graph_ready
                if str(node.get("id") or "").strip()
            ],
            plant_enabled_event_ids=[
                str(node.get("id") or "").strip()
                for node in plant_enabled
                if str(node.get("id") or "").strip()
            ],
            disabled_frontier=disabled_frontier,
        )

        if plant_enabled:
            return plant_enabled[0]

        repair_node = self._try_compile_controllable_repair(
            disabled_frontier=disabled_frontier,
            trigger="runtime_disabled_frontier",
        )
        if repair_node:
            return repair_node

        if disabled_frontier:
            self._mark_runtime_des_human_required(
                disabled_frontier=disabled_frontier,
                message=(
                    "Runtime DES supervisor found graph-ready event(s), but their "
                    "plant guards are disabled and no deterministic repair event "
                    "was applicable."
                ),
            )
        return None

    def _graph_ready_task_nodes(self) -> list[dict[str, Any]]:
        if hasattr(self.process_planner, "graph_ready_task_nodes"):
            nodes = self.process_planner.graph_ready_task_nodes()
            return [node for node in nodes if isinstance(node, dict)]
        node = self.process_planner.next_ready_task()
        return [node] if isinstance(node, dict) else []

    def _tool_row_for_task_node(self, task_node: dict[str, Any]) -> dict[str, Any]:
        function_name = str(task_node.get("function_name") or "").strip()
        resource_jid = str(task_node.get("resource_jid") or "").strip()
        if not function_name:
            return {}
        if hasattr(self.process_planner, "_tool_row_for_task"):
            try:
                row = self.process_planner._tool_row_for_task(
                    resource_jid=resource_jid,
                    function_name=function_name,
                    tools_catalog=list(getattr(self, "tools_catalog", []) or []),
                )
                if isinstance(row, dict):
                    return dict(row)
            except Exception:
                self.logger.debug(
                    "[Product] Runtime DES tool lookup fell back for task=%s",
                    task_node.get("id"),
                    exc_info=True,
                )
        self.__class__._load_shared_tools_catalogue()
        return dict((LlmAgent._TOOLS_BY_FUNC or {}).get(function_name) or {})

    def _event_contract_for_task_node(self, task_node: dict[str, Any]) -> dict[str, Any]:
        row = self._tool_row_for_task_node(task_node)
        contract = {
            "task_id": str(task_node.get("id") or "").strip(),
            "function_name": str(task_node.get("function_name") or "").strip(),
            "resource_jid": str(task_node.get("resource_jid") or "").strip(),
            "part_name": self._tracked_part_name_for_task(task_node),
            "in_state": str(row.get("in_state") or "").strip(),
            "out_state": str(row.get("out_state") or "").strip(),
            "part_in_state": str(row.get("part_in_state") or "").strip(),
            "context_mapping": dict(row.get("context_mapping") or {}),
            "part_transition": dict(row.get("part_transition") or {}),
        }
        for key in ("in_state", "out_state", "part_in_state", "context_mapping", "part_transition"):
            if key in task_node and task_node.get(key):
                contract[key] = deepcopy(task_node[key])
        return contract

    def _build_runtime_plant_state(
        self,
        *,
        resource_jids: Iterable[str] | None = None,
        base_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        active_bridge_sequence = self._active_bridge_sequence()
        system_state = deepcopy(
            base_state
            if isinstance(base_state, dict)
            else (active_bridge_sequence or {}).get("system_coordination_state") or {}
        )
        if not isinstance(system_state, dict):
            system_state = {}
        system_state.setdefault("resource_states", {})
        for resource_jid in {
            str(item or "").strip()
            for item in (resource_jids or [])
            if str(item or "").strip()
        }:
            snapshot = self._refresh_bridge_snapshot(resource_jid)
            if isinstance(snapshot, dict):
                system_state = self._system_coordination_state_with_bridge_snapshot(
                    base_state=system_state,
                    resource_jid=resource_jid,
                    bridge_snapshot=snapshot,
                )
        return {
            "resources": dict(
                self.process_planner._extract_resource_states(system_state)
            ),
            "parts": deepcopy(dict(getattr(self, "part_tracker", {}) or {})),
            "system_coordination_state": system_state,
        }

    @staticmethod
    def _plant_resource_field(
        plant_state: dict[str, Any],
        resource_jid: str,
        field: str,
    ) -> Any:
        resource_entry = dict(
            dict(plant_state.get("resources") or {}).get(str(resource_jid or "").strip()) or {}
        )
        if field in resource_entry:
            return resource_entry.get(field)
        facets = dict(resource_entry.get("resource_facets") or {})
        manipulator = dict(facets.get("manipulator") or {})
        if field in manipulator:
            return manipulator.get(field)
        core = dict(resource_entry.get("resource_core") or {})
        if field in core:
            return core.get(field)
        return None

    @staticmethod
    def _plant_part_field(
        plant_state: dict[str, Any],
        part_name: str,
        field: str,
    ) -> Any:
        part_entry = dict(
            dict(plant_state.get("parts") or {}).get(str(part_name or "").strip()) or {}
        )
        return part_entry.get(field)

    @staticmethod
    def _unknown_runtime_value(value: Any) -> bool:
        return value in (None, "", [], {}, "unknown")

    def _event_guard_violations(
        self,
        task_node: dict[str, Any],
        *,
        plant_state: dict[str, Any] | None = None,
        enforce_unknown: bool = False,
    ) -> list[dict[str, Any]]:
        contract = self._event_contract_for_task_node(task_node)
        resource_jid = str(contract.get("resource_jid") or "").strip()
        part_name = str(contract.get("part_name") or "").strip()
        plant = plant_state or self._build_runtime_plant_state(
            resource_jids=[resource_jid] if resource_jid else []
        )
        violations: list[dict[str, Any]] = []

        def add_violation(
            *,
            entity_kind: str,
            entity: str,
            field: str,
            expected: Any,
            actual: Any,
            kind: str,
        ) -> None:
            if (
                field != "held_part"
                and self._unknown_runtime_value(actual)
                and not enforce_unknown
            ):
                return
            if actual == expected:
                return
            violations.append(
                {
                    "kind": kind,
                    "entity_kind": entity_kind,
                    "entity": entity,
                    "field": field,
                    "expected": expected,
                    "actual": actual,
                    "source_task_id": contract.get("task_id"),
                    "source_function_name": contract.get("function_name"),
                }
            )

        in_state = str(contract.get("in_state") or "").strip()
        if resource_jid and in_state and in_state.lower() != "any":
            add_violation(
                entity_kind="resource",
                entity=resource_jid,
                field="current_state",
                expected=in_state,
                actual=self._plant_resource_field(plant, resource_jid, "current_state"),
                kind="event_guard_resource_state",
            )

        part_in_state = str(contract.get("part_in_state") or "").strip()
        if part_name and part_in_state:
            add_violation(
                entity_kind="part",
                entity=part_name,
                field="state",
                expected=part_in_state,
                actual=self._plant_part_field(plant, part_name, "state"),
                kind="event_guard_part_state",
            )
            if part_in_state == "in_gripper" and resource_jid:
                add_violation(
                    entity_kind="resource",
                    entity=resource_jid,
                    field="held_part",
                    expected=part_name,
                    actual=self._plant_resource_field(plant, resource_jid, "held_part"),
                    kind="event_guard_carried_entity",
                )

        ctx_map = dict(contract.get("context_mapping") or {})
        location_param = str(ctx_map.get("location_param") or "").strip()
        location_type = str(ctx_map.get("location_type") or "").strip()
        location_value = (
            dict(task_node.get("params") or {}).get(location_param)
            if location_param
            else None
        )
        if part_name and location_type == "part_location" and location_value not in (None, ""):
            add_violation(
                entity_kind="part",
                entity=part_name,
                field="location",
                expected=location_value,
                actual=self._plant_part_field(plant, part_name, "location"),
                kind="event_guard_part_location",
            )
        return violations

    def _record_runtime_des_trace(
        self,
        *,
        graph_ready_event_ids: list[str],
        plant_enabled_event_ids: list[str],
        disabled_frontier: list[dict[str, Any]],
        selected_repair_operator: str = "",
        projected_repair_effects: dict[str, Any] | None = None,
    ) -> None:
        trace = {
            "graph_ready_event_ids": [
                str(item).strip() for item in graph_ready_event_ids if str(item).strip()
            ],
            "plant_enabled_event_ids": [
                str(item).strip() for item in plant_enabled_event_ids if str(item).strip()
            ],
            "disabled_frontier": deepcopy(disabled_frontier),
            "selected_repair_operator": str(selected_repair_operator or "").strip(),
            "projected_repair_effects": deepcopy(projected_repair_effects or {}),
            "updated_at_utc": self._utc_now_iso(),
        }
        bridge_debug = deepcopy(self.runtime_recovery.get("bridge_debug") or {})
        if isinstance(bridge_debug, dict):
            bridge_debug["runtime_des_supervisor"] = trace
            self.runtime_recovery["bridge_debug"] = bridge_debug

    def _resource_can_reach_location(self, resource_jid: str, location: str) -> bool:
        location_key = str(location or "").strip()
        if not location_key:
            return False
        resource = self.process_planner._resource_by_jid(resource_jid)
        caps = getattr(resource, "static_capabilities", {}) if resource is not None else {}
        if not isinstance(caps, dict) or not caps:
            return True
        reachability = caps.get("reachability")
        if isinstance(reachability, list) and reachability:
            normalized = {str(item).strip() for item in reachability if str(item).strip()}
            if location_key in normalized:
                return True
        staging_areas = caps.get("staging_areas")
        if isinstance(staging_areas, dict) and location_key in staging_areas:
            return True
        if isinstance(staging_areas, list):
            normalized = {str(item).strip() for item in staging_areas if str(item).strip()}
            if location_key in normalized:
                return True
        return not reachability

    @staticmethod
    def _model_name_from_mapping(value: Any, part_name: str) -> str:
        if not isinstance(value, dict):
            return ""
        for key in ("model_name", "gazebo_model_name"):
            token = str(value.get(key) or "").strip()
            if token:
                return token

        parts = value.get("parts")
        if isinstance(parts, dict):
            model_map = parts.get("model_map")
            if isinstance(model_map, dict):
                token = str(model_map.get(part_name) or "").strip()
                if token:
                    return token

        for key in (
            "part_geometry",
            "product_geometry",
            "geometry",
            "target",
            "part_target",
        ):
            token = ProductAgent._model_name_from_mapping(value.get(key), part_name)
            if token:
                return token

        grounding_context = value.get("grounding_context")
        if isinstance(grounding_context, dict):
            parts = grounding_context.get("parts")
            if isinstance(parts, dict):
                token = ProductAgent._model_name_from_mapping(parts.get(part_name), part_name)
                if token:
                    return token

        return ""

    def _part_model_name_for_repair(
        self,
        part_name: str,
        *,
        disabled_event: dict[str, Any] | None = None,
        active_bridge_sequence: dict[str, Any] | None = None,
    ) -> str:
        part_key = str(part_name or "").strip()
        if not part_key:
            return ""

        candidates: list[Any] = [
            dict((getattr(self, "part_tracker", {}) or {}).get(part_key) or {}),
            disabled_event or {},
            dict((disabled_event or {}).get("params") or {}),
            active_bridge_sequence or {},
            dict(getattr(self, "product_geometry", {}) or {}),
        ]
        try:
            candidates.append(self._part_geometry_for_pick_context(part_key))
        except Exception:
            self.logger.debug(
                "[Product] Could not resolve product geometry for DES repair part=%s",
                part_key,
                exc_info=True,
            )

        runtime_context = getattr(self, "_runtime_recovery_context", {}) or {}
        prepared_request = dict(runtime_context.get("prepared_bridge_request") or {})
        candidates.append(prepared_request)
        candidates.append(dict((prepared_request.get("part_tracker") or {}).get(part_key) or {}))

        for node in list(getattr(getattr(self, "process_planner", None), "nodes", []) or []):
            if not isinstance(node, dict):
                continue
            if self._tracked_part_name_for_task(node) != part_key:
                continue
            candidates.append(node)
            candidates.append(dict(node.get("params") or {}))

        for candidate in candidates:
            token = self._model_name_from_mapping(candidate, part_key)
            if token:
                return token
        return ""

    def _part_geometry_for_repair(self, part_name: str, model_name: str = "") -> dict[str, Any]:
        geometry: dict[str, Any] = {}
        try:
            geometry.update(self._part_geometry_for_pick_context(part_name))
        except Exception:
            self.logger.debug(
                "[Product] Could not build DES repair pick geometry for part=%s",
                part_name,
                exc_info=True,
            )
        if model_name and not geometry.get("model_name"):
            geometry["model_name"] = model_name
        return {key: deepcopy(value) for key, value in geometry.items() if value is not None}

    def _repair_execution_mode_for_resource(self, resource_jid: str) -> str:
        resource = self.process_planner._resource_by_jid(resource_jid)
        execution_mode = str(getattr(resource, "execution_mode", "") or "").strip().lower()
        return execution_mode or "simulation"

    def _resolve_acquire_entity_pick_source(
        self,
        *,
        resource_jid: str,
        part_name: str,
        source_location: str,
        part_geometry: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str]:
        normalized_source = str(source_location or "").strip()
        pick_params: dict[str, Any] = {"part_name": part_name}
        if part_geometry:
            pick_params["product_geometry"] = deepcopy(part_geometry)

        if not normalized_source:
            return (
                "unsupported",
                pick_params,
                (
                    "Cannot compile acquire_entity repair: missing source location "
                    f"for part '{part_name}'."
                ),
            )

        if normalized_source == "observed_pose" or normalized_source.endswith("_observed_pose"):
            return "observed_pose", pick_params, ""

        tracked_pose = self._tracked_pose_for_part_at_location(
            part_name=part_name,
            source_location=normalized_source,
        )
        if tracked_pose is not None:
            pick_params["target_pose"] = deepcopy(tracked_pose)
            pick_params["target_pose_source"] = f"tracked_current_pose:{normalized_source}"
            return "tracked_location", pick_params, ""

        tracker_entry = dict((getattr(self, "part_tracker", {}) or {}).get(part_name) or {})
        origin_location = str(tracker_entry.get("origin_resource_location") or "").strip()
        if origin_location == normalized_source:
            origin_pose = self._normalized_xyz_pose(tracker_entry.get("origin_pose"))
            if origin_pose is not None:
                pick_params["target_pose"] = deepcopy(origin_pose)
                pick_params["target_pose_source"] = (
                    f"tracked_origin_pose:{normalized_source}"
                )
                return "tracked_origin", pick_params, ""

        execution_mode = self._repair_execution_mode_for_resource(resource_jid)
        resolved_geometry = ProductProfile.resolve_place_geometry(
            part_name=part_name,
            destination_location=normalized_source,
            product_geometry=part_geometry,
            execution_mode=execution_mode,
        )
        target_pose = self._normalized_xyz_pose(
            dict(resolved_geometry.get("target_origin_pose") or {})
        )
        if target_pose is None:
            return (
                "unsupported",
                pick_params,
                (
                    "Cannot compile acquire_entity repair: source_location "
                    f"'{normalized_source}' has no deterministic target_origin_pose "
                    f"for part '{part_name}'."
                ),
            )

        pick_params["product_geometry"] = deepcopy(resolved_geometry)
        pick_params["target_pose"] = deepcopy(target_pose)
        pick_params["target_pose_source"] = normalized_source
        return "modeled_location", pick_params, ""

    def _repair_primitive_catalog_for_resource(self, resource_jid: str) -> list[dict[str, Any]]:
        resource = self.process_planner._resource_by_jid(resource_jid)
        if resource is None:
            return []
        method = getattr(resource, "bridge_execution_primitive_catalog", None)
        if callable(method):
            try:
                catalog = method()
                if isinstance(catalog, list):
                    return [dict(item) for item in catalog if isinstance(item, dict)]
            except Exception:
                self.logger.debug(
                    "[Product] Could not load primitive catalog for DES repair resource=%s",
                    resource_jid,
                    exc_info=True,
                )
        return []

    def _validate_repair_primitive_program(
        self,
        *,
        resource_jid: str,
        primitive_steps: list[dict[str, Any]],
    ) -> str:
        catalog = self._repair_primitive_catalog_for_resource(resource_jid)
        catalog_by_name = {
            str(entry.get("name") or "").strip(): dict(entry)
            for entry in catalog
            if isinstance(entry, dict) and str(entry.get("name") or "").strip()
        }
        fallback_required_params = {
            "compute_pick_targets": ["part_name"],
            "move_cartesian": ["x", "y", "z"],
            "grasp_part": ["model_name"],
        }
        for index, step in enumerate(primitive_steps or []):
            if not isinstance(step, dict):
                return f"primitive step {index} must be an object"
            primitive = str(step.get("primitive") or "").strip()
            if not primitive:
                return f"primitive step {index} is missing 'primitive'"
            if catalog_by_name and primitive not in catalog_by_name:
                return f"unknown primitive '{primitive}' at step {index}"
            params = dict(step.get("params") or {})
            required = (
                list(catalog_by_name.get(primitive, {}).get("required_params") or [])
                if catalog_by_name
                else list(fallback_required_params.get(primitive) or [])
            )
            if primitive in catalog_by_name:
                allowed_params = {
                    str(param_name).strip()
                    for param_name in dict(catalog_by_name.get(primitive, {}).get("params") or {})
                    if str(param_name).strip()
                }
                unexpected_params = sorted(
                    str(param_name).strip()
                    for param_name in params.keys()
                    if str(param_name).strip()
                    and str(param_name).strip() not in allowed_params
                )
                if unexpected_params:
                    allowed_description = (
                        f"allowed params={sorted(allowed_params)}"
                        if allowed_params
                        else "primitive accepts no params"
                    )
                    return (
                        f"unexpected params {unexpected_params} at step {index} "
                        f"({primitive}); {allowed_description}"
                    )
            for required_param in required:
                param_name = str(required_param or "").strip()
                if not param_name:
                    continue
                value = params.get(param_name)
                if value in (None, ""):
                    return (
                        f"missing required param '{param_name}' at step {index} "
                        f"({primitive})"
                    )
        return ""

    def _record_repair_compile_error(
        self,
        *,
        disabled_event: dict[str, Any],
        operator: str,
        message: str,
    ) -> None:
        bridge_debug = deepcopy(self.runtime_recovery.get("bridge_debug") or {})
        bridge_debug.setdefault("runtime_des_supervisor", {})
        bridge_debug["runtime_des_supervisor"].update(
            {
                "selected_repair_operator": str(operator or "").strip(),
                "repair_compile_error": str(message or "").strip(),
                "disabled_event": deepcopy(disabled_event),
            }
        )
        self.runtime_recovery["bridge_debug"] = bridge_debug

    @staticmethod
    def _failed_release_primitive_observation(observations: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(observations, dict):
            return {}
        primitive = str(observations.get("primitive") or "").strip()
        if primitive != "release_part":
            return {}
        return deepcopy(observations)

    def _try_compile_release_retry_event(
        self,
        *,
        task_node: dict[str, Any],
        active_bridge_sequence: dict[str, Any],
        observations: dict[str, Any] | None,
        trigger: str,
        used_llm_bridge: bool,
        feedback_history: list[Any],
    ) -> dict[str, Any] | None:
        release_observation = self._failed_release_primitive_observation(observations)
        if not release_observation:
            return None
        retry_attempts = int(active_bridge_sequence.get("release_retry_attempts") or 0)
        if retry_attempts >= 1:
            return None

        params = dict(task_node.get("params") or {})
        part_name = str(params.get("part_name") or self._tracked_part_name_for_task(task_node) or "").strip()
        if not part_name:
            return None
        model_name = str(params.get("model_name") or "").strip()
        if not model_name:
            model_name = self._part_model_name_for_repair(
                part_name,
                disabled_event=task_node,
                active_bridge_sequence=active_bridge_sequence,
            )
        if not model_name:
            self._record_repair_compile_error(
                disabled_event={
                    "task_id": str(task_node.get("id") or "").strip(),
                    "function_name": str(task_node.get("function_name") or "").strip(),
                    "resource_jid": str(task_node.get("resource_jid") or "").strip(),
                    "part_name": part_name,
                },
                operator="retry_event",
                message=(
                    "Cannot compile release retry event: missing required "
                    f"model_name for part '{part_name}'."
                ),
            )
            return None

        state_after = dict(release_observation.get("state_after") or {})
        if state_after:
            held_after = state_after.get("held_part")
            current_after = str(state_after.get("current_state") or "").strip()
            if held_after not in (part_name, model_name) or current_after not in {"", "picked"}:
                return None

        retry_task_id = f"REPAIR_EVENT_{uuid.uuid4().hex[:6].upper()}"
        sequence_id = f"DESRETRY_{uuid.uuid4().hex[:8].upper()}"
        resource_jid = str(task_node.get("resource_jid") or "").strip()
        original_task_id = str(task_node.get("id") or "").strip()
        release_params = {
            "part_name": part_name,
            "model_name": model_name,
            "assume_released_if_open": True,
        }
        primitive_steps = [{"primitive": "release_part", "params": release_params}]
        validation_error = self._validate_repair_primitive_program(
            resource_jid=resource_jid,
            primitive_steps=primitive_steps,
        )
        if validation_error:
            self._record_repair_compile_error(
                disabled_event={
                    "task_id": original_task_id,
                    "function_name": "execute_recovery_macro",
                    "resource_jid": resource_jid,
                    "part_name": part_name,
                },
                operator="retry_event",
                message=validation_error,
            )
            return None

        expected_snapshot: dict[str, Any] | None = None
        if state_after:
            expected_snapshot = {
                "current_state": state_after.get("current_state"),
                "held_part": state_after.get("held_part"),
                "gripper_state": state_after.get("gripper_state"),
            }
            expected_snapshot = {
                key: value for key, value in expected_snapshot.items() if value is not None
            }
        projected_snapshot = deepcopy(task_node.get("projected_snapshot") or {})
        if not projected_snapshot:
            projected_snapshot = {
                "resource_type": "robot",
                "resource_jid": resource_jid,
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
                "resource_core": {
                    "resource_jid": resource_jid,
                    "resource_type": "robot",
                    "current_state": "idle",
                },
                "resource_facets": {
                    "manipulator": {"held_part": None, "gripper_state": "open"}
                },
            }
        projected_part_entry = deepcopy(task_node.get("projected_part_entry") or {})
        if not projected_part_entry:
            destination = str(params.get("destination_location") or "").strip()
            projected_part_entry = {
                "state": "assembled" if destination else "ready",
                "location": destination or None,
                "model_name": model_name,
            }
        else:
            projected_part_entry.setdefault("model_name", model_name)

        retry_node = {
            "id": retry_task_id,
            "type": "task",
            "status": "pending",
            "function_name": "execute_recovery_macro",
            "resource_jid": resource_jid,
            "params": {
                "macro_name": f"retry_release_for_{original_task_id or 'bridge_macro'}",
                "primitive_steps": primitive_steps,
                "expected_start_state": str(state_after.get("current_state") or "picked"),
                "product_jid": str(self.jid),
                "task_id": retry_task_id,
                "part_name": part_name,
                "destination_location": params.get("destination_location"),
                "out_state": "idle",
            },
            "predecessors": [],
            "successors": [],
            "bridge_sequence_id": sequence_id,
            "bridge_sequence_index": 1,
            "bridge_sequence_length": 1,
            "in_state": "picked",
            "out_state": "idle",
            "part_transition": deepcopy(task_node.get("part_transition") or {}),
            "part_name": part_name,
            "projected_snapshot": projected_snapshot,
            "projected_part_entry": projected_part_entry,
            "repair_operator": "retry_event",
            "repair_intent": "release_retry",
            "retry_of_task_id": original_task_id,
            "change_reason": (
                "INSERTION: Runtime DES retry event for failed bridge release "
                f"{original_task_id or '-'}"
            ),
        }
        if expected_snapshot:
            retry_node["params"]["expected_snapshot"] = expected_snapshot
        self.process_planner.nodes.append(retry_node)

        next_sequence = deepcopy(active_bridge_sequence)
        next_sequence.update(
            {
                "bridge_sequence_id": sequence_id,
                "bridge_task_ids": [retry_task_id],
                "bridge_sequence_length": 1,
                "state": "executing",
                "trigger": str(trigger or "").strip() or next_sequence.get("trigger", ""),
                "used_llm_bridge": bool(used_llm_bridge),
                "release_retry_attempts": retry_attempts + 1,
                "repair_operator": "retry_event",
                "repair_intent": "release_retry",
                "retry_source_task_id": original_task_id,
            }
        )
        bridge_debug = deepcopy(self.runtime_recovery.get("bridge_debug") or {})
        bridge_debug.setdefault("runtime_des_supervisor", {})
        bridge_debug["runtime_des_supervisor"].update(
            {
                "selected_repair_operator": "retry_event",
                "repair_intent": "release_retry",
                "repair_task_id": retry_task_id,
                "retry_source_task_id": original_task_id,
                "release_observation": deepcopy(release_observation),
            }
        )
        self._set_runtime_recovery(
            status="resolved",
            resolution_class="runtime_des_repair",
            trigger=str(trigger or "").strip() or self.runtime_recovery.get("trigger", ""),
            failed_task_id=str(
                next_sequence.get("failed_task_id")
                or self.runtime_recovery.get("failed_task_id")
                or ""
            ),
            message=(
                "Runtime DES supervisor inserted retry_event for failed bridge release."
            ),
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=bool(used_llm_bridge),
            bridge_debug=bridge_debug,
            bridge_approval_state="approved",
            active_bridge_sequence=next_sequence,
            bridge_feedback_history=feedback_history,
            violations=[],
            append_history=True,
            history_message=(
                f"Runtime DES release retry event {retry_task_id} inserted for "
                f"{original_task_id or 'bridge macro'}."
            ),
        )
        return retry_node

    def _try_compile_controllable_repair(
        self,
        *,
        disabled_frontier: list[dict[str, Any]],
        trigger: str,
        active_bridge_sequence: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Compile a generic repair event when guard/effect matching proves it safe."""
        active = deepcopy(active_bridge_sequence or self._active_bridge_sequence() or {})
        repair_attempts = int(active.get("continuation_repair_attempts") or 0) if active else 0
        if repair_attempts >= 1:
            return None

        for disabled_event in disabled_frontier or []:
            violations = [
                dict(item)
                for item in (disabled_event.get("guard_violations") or [])
                if isinstance(item, dict)
            ]
            carried_violation = next(
                (
                    item
                    for item in violations
                    if str(item.get("field") or "").strip() == "held_part"
                    and str(item.get("expected") or "").strip()
                ),
                None,
            )
            if not carried_violation:
                continue

            resource_jid = str(
                carried_violation.get("entity")
                or disabled_event.get("resource_jid")
                or ""
            ).strip()
            part_name = str(carried_violation.get("expected") or "").strip()
            if not resource_jid or not part_name:
                continue

            plant_state = self._build_runtime_plant_state(resource_jids=[resource_jid])
            current_state = self._plant_resource_field(plant_state, resource_jid, "current_state")
            held_part = self._plant_resource_field(plant_state, resource_jid, "held_part")
            if str(current_state or "").strip() not in {"", "idle"}:
                continue
            if held_part not in (None, "", "unknown"):
                continue
            part_entry = dict((plant_state.get("parts") or {}).get(part_name) or {})
            source_location = str(part_entry.get("location") or "").strip()
            if (
                not source_location
                or source_location == f"{resource_jid}_gripper"
                or source_location.endswith("_gripper")
            ):
                continue
            if not self._resource_can_reach_location(resource_jid, source_location):
                continue

            repair_node = self._append_acquire_entity_repair_event(
                resource_jid=resource_jid,
                part_name=part_name,
                source_location=source_location,
                disabled_event=disabled_event,
                active_bridge_sequence=active,
                trigger=trigger,
            )
            if repair_node:
                return repair_node
        return None

    def _append_acquire_entity_repair_event(
        self,
        *,
        resource_jid: str,
        part_name: str,
        source_location: str,
        disabled_event: dict[str, Any],
        active_bridge_sequence: dict[str, Any] | None,
        trigger: str,
    ) -> dict[str, Any] | None:
        sequence_id = f"DESREPAIR_{uuid.uuid4().hex[:8].upper()}"
        task_id = f"REPAIR_EVENT_{uuid.uuid4().hex[:6].upper()}"
        event_fact_part = str(part_name)
        model_name = self._part_model_name_for_repair(
            part_name,
            disabled_event=disabled_event,
            active_bridge_sequence=active_bridge_sequence,
        )
        if not model_name:
            self._record_repair_compile_error(
                disabled_event=disabled_event,
                operator="acquire_entity",
                message=(
                    "Cannot compile acquire_entity repair: missing required "
                    f"model_name for part '{part_name}'."
                ),
            )
            self.logger.warning(
                "[Product] Runtime DES repair compile failed: missing model_name for part=%s",
                part_name,
            )
            return None
        part_geometry = self._part_geometry_for_repair(part_name, model_name=model_name)
        _source_mode, pick_params, source_error = self._resolve_acquire_entity_pick_source(
            resource_jid=resource_jid,
            part_name=part_name,
            source_location=source_location,
            part_geometry=part_geometry,
        )
        if source_error:
            self._record_repair_compile_error(
                disabled_event=disabled_event,
                operator="acquire_entity",
                message=source_error,
            )
            self.logger.warning(
                "[Product] Runtime DES repair compile failed: %s",
                source_error,
            )
            return None
        predecessor = str(
            (active_bridge_sequence or {}).get("last_task_id")
            or (active_bridge_sequence or {}).get("failed_task_id")
            or self.runtime_recovery.get("failed_task_id")
            or ""
        ).strip()
        primitive_steps = [
            {
                "primitive": "compute_pick_targets",
                "params": pick_params,
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": {
                        "context_ref": (
                            f"event_facts.pick_targets.{event_fact_part}.approach_pose.x"
                        )
                    },
                    "y": {
                        "context_ref": (
                            f"event_facts.pick_targets.{event_fact_part}.approach_pose.y"
                        )
                    },
                    "z": {
                        "context_ref": (
                            f"event_facts.pick_targets.{event_fact_part}.approach_pose.z"
                        )
                    },
                },
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": {
                        "context_ref": (
                            f"event_facts.pick_targets.{event_fact_part}.target_pose.x"
                        )
                    },
                    "y": {
                        "context_ref": (
                            f"event_facts.pick_targets.{event_fact_part}.target_pose.y"
                        )
                    },
                    "z": {
                        "context_ref": (
                            f"event_facts.pick_targets.{event_fact_part}.target_pose.z"
                        )
                    },
                },
            },
            {
                "primitive": "grasp_part",
                "params": {"part_name": part_name, "model_name": model_name},
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": {
                        "context_ref": (
                            f"event_facts.pick_targets.{event_fact_part}.approach_pose.x"
                        )
                    },
                    "y": {
                        "context_ref": (
                            f"event_facts.pick_targets.{event_fact_part}.approach_pose.y"
                        )
                    },
                    "z": {
                        "context_ref": (
                            f"event_facts.pick_targets.{event_fact_part}.approach_pose.z"
                        )
                    },
                },
            },
        ]
        validation_error = self._validate_repair_primitive_program(
            resource_jid=resource_jid,
            primitive_steps=primitive_steps,
        )
        if validation_error:
            self._record_repair_compile_error(
                disabled_event=disabled_event,
                operator="acquire_entity",
                message=validation_error,
            )
            self.logger.warning(
                "[Product] Runtime DES repair compile failed: %s",
                validation_error,
            )
            return None
        node: dict[str, Any] = {
            "id": task_id,
            "type": "task",
            "status": "pending",
            "function_name": "execute_recovery_macro",
            "resource_jid": resource_jid,
            "params": {
                "macro_name": (
                    "restore_guard_for_"
                    f"{disabled_event.get('task_id') or 'disabled_event'}_acquire_entity"
                ),
                "primitive_steps": primitive_steps,
                "expected_start_state": "idle",
                "product_jid": str(self.jid),
                "task_id": task_id,
                "part_name": part_name,
                "origin_resource_location": source_location,
                "part_geometry": deepcopy(part_geometry),
                "out_state": "picked",
            },
            "predecessors": [predecessor] if predecessor else [],
            "successors": [],
            "bridge_sequence_id": sequence_id,
            "bridge_sequence_index": 1,
            "bridge_sequence_length": 1,
            "in_state": "idle",
            "out_state": "picked",
            "part_transition": {
                "completed": {
                    "state": "in_gripper",
                    "location_template": "{resource_jid}_gripper",
                }
            },
            "part_name": part_name,
            "projected_snapshot": {
                "resource_type": "robot",
                "resource_jid": resource_jid,
                "current_state": "picked",
                "held_part": part_name,
                "gripper_state": "closed",
                "resource_core": {
                    "resource_jid": resource_jid,
                    "resource_type": "robot",
                    "current_state": "picked",
                },
                "resource_facets": {
                    "manipulator": {
                        "held_part": part_name,
                        "gripper_state": "closed",
                    }
                },
            },
            "projected_part_entry": {
                "state": "in_gripper",
                "location": f"{resource_jid}_gripper",
                "model_name": model_name,
            },
            "repair_operator": "acquire_entity",
            "repair_intent": "restore_event_guard",
            "restores_event_id": str(disabled_event.get("task_id") or "").strip(),
            "guard_violations": deepcopy(disabled_event.get("guard_violations") or []),
            "producer_semantics": ["pick_approach", "pick_grasp"],
            "disabled_event": deepcopy(disabled_event),
            "change_reason": (
                "INSERTION: Runtime DES guard-restoration event 'acquire_entity' "
                f"for disabled event {disabled_event.get('task_id') or '-'}"
            ),
        }
        self.process_planner.nodes.append(node)

        next_sequence = deepcopy(active_bridge_sequence or {})
        next_sequence.update(
            {
                "bridge_sequence_id": sequence_id,
                "bridge_task_ids": [task_id],
                "bridge_sequence_length": 1,
                "state": "executing",
                "trigger": str(trigger or "").strip() or next_sequence.get("trigger", ""),
                "used_llm_bridge": bool(next_sequence.get("used_llm_bridge", False)),
                "continuation_repair_attempts": int(
                    next_sequence.get("continuation_repair_attempts") or 0
                )
                + 1,
                "repair_operator": "acquire_entity",
                "repair_intent": "restore_event_guard",
                "repair_target_task_id": str(disabled_event.get("task_id") or "").strip(),
            }
        )
        bridge_debug = deepcopy(self.runtime_recovery.get("bridge_debug") or {})
        bridge_debug.setdefault("runtime_des_supervisor", {})
        bridge_debug["runtime_des_supervisor"].update(
            {
                "selected_repair_operator": "acquire_entity",
                "projected_repair_effects": {
                    "resource": {
                        "entity": resource_jid,
                        "current_state": "picked",
                        "held_part": part_name,
                    },
                    "part": {
                        "entity": part_name,
                        "state": "in_gripper",
                        "location": f"{resource_jid}_gripper",
                        "model_name": model_name,
                    },
                },
                "disabled_event": deepcopy(disabled_event),
                "repair_task_id": task_id,
                "repair_intent": "restore_event_guard",
                "restores_event_id": str(disabled_event.get("task_id") or "").strip(),
                "guard_violations": deepcopy(disabled_event.get("guard_violations") or []),
                "producer_semantics": ["pick_approach", "pick_grasp"],
            }
        )
        self._set_runtime_recovery(
            status="resolved",
            resolution_class="runtime_des_repair",
            trigger=str(trigger or "").strip() or self.runtime_recovery.get("trigger", ""),
            failed_task_id=str(
                next_sequence.get("failed_task_id")
                or self.runtime_recovery.get("failed_task_id")
                or ""
            ),
            message=(
                "Runtime DES supervisor inserted controllable repair event "
                f"'acquire_entity' before resuming nominal execution."
            ),
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=bool(next_sequence.get("used_llm_bridge", False)),
            bridge_debug=bridge_debug,
            bridge_approval_state="approved",
            active_bridge_sequence=next_sequence,
            violations=[],
            append_history=True,
            history_message=(
                f"Runtime DES repair event {task_id} inserted for disabled "
                f"event {disabled_event.get('task_id') or '-'}."
            ),
        )
        return node

    def _mark_runtime_des_human_required(
        self,
        *,
        disabled_frontier: list[dict[str, Any]],
        message: str,
    ) -> None:
        bridge_debug = deepcopy(self.runtime_recovery.get("bridge_debug") or {})
        bridge_debug.setdefault("runtime_des_supervisor", {})
        bridge_debug["runtime_des_supervisor"].update(
            {
                "disabled_frontier": deepcopy(disabled_frontier),
                "selected_repair_operator": "",
            }
        )
        self._set_runtime_recovery(
            status="human_required",
            resolution_class="human_required",
            message=message,
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            bridge_debug=bridge_debug,
            active_bridge_sequence=self._active_bridge_sequence(),
            violations=[],
            append_history=True,
            history_message=message,
        )

    def _bridge_continuation_disabled_frontier(
        self,
        active_bridge_sequence: dict[str, Any],
    ) -> list[dict[str, Any]]:
        requirements = [
            dict(item)
            for item in (active_bridge_sequence.get("continuation_requirements") or [])
            if isinstance(item, dict)
        ]
        if not requirements:
            return []
        plant_state = self._build_runtime_plant_state(
            resource_jids=[
                str(item.get("entity") or "").strip()
                for item in requirements
                if str(item.get("entity_kind") or "").strip() == "resource"
            ],
            base_state=active_bridge_sequence.get("system_coordination_state") or {},
        )
        grouped: dict[str, dict[str, Any]] = {}
        for requirement in requirements:
            source_task_id = str(requirement.get("source_task_id") or "").strip()
            if not source_task_id:
                continue
            entity_kind = str(requirement.get("entity_kind") or "").strip()
            entity = str(requirement.get("entity") or "").strip()
            field = str(requirement.get("field") or "").strip()
            expected = requirement.get("expected")
            if not entity_kind or not entity or not field:
                continue
            actual = (
                self._plant_resource_field(plant_state, entity, field)
                if entity_kind == "resource"
                else self._plant_part_field(plant_state, entity, field)
            )
            if actual == expected:
                continue
            node = self.process_planner._find_node(source_task_id)
            group = grouped.setdefault(
                source_task_id,
                {
                    "task_id": source_task_id,
                    "function_name": str(requirement.get("source_function_name") or "").strip(),
                    "resource_jid": str(node.get("resource_jid") or entity if isinstance(node, dict) else entity),
                    "part_name": self._tracked_part_name_for_task(node) if isinstance(node, dict) else "",
                    "guard_violations": [],
                },
            )
            group["guard_violations"].append(
                {
                    "kind": str(requirement.get("kind") or "continuation_guard"),
                    "entity_kind": entity_kind,
                    "entity": entity,
                    "field": field,
                    "expected": expected,
                    "actual": actual,
                    "source_task_id": source_task_id,
                    "source_function_name": requirement.get("source_function_name"),
                    "condition_family": "continuation",
                }
            )
        return list(grouped.values())

    @staticmethod
    def _runtime_is_gazebo_simulation() -> bool:
        exec_mode = str(os.environ.get("EXECUTION_MODE", "dry_run") or "").strip().lower()
        robot_env = str(os.environ.get("ROBOT_ENV", "gazebo") or "").strip().lower()
        return exec_mode == "simulation" and robot_env == "gazebo"

    def _should_enable_generated_bridge_verification(
        self,
        *,
        bridge_debug: dict[str, Any] | None = None,
    ) -> bool:
        verification_flag_enabled = bool(
            self._generated_bridge_gazebo_verification_enabled
            or _env_flag_enabled(
                "CAIS_VERIFY_GENERATED_BRIDGE_IN_GAZEBO",
                "CAIS_GENERATED_BRIDGE_GAZEBO_VERIFICATION",
                default=False,
            )
        )
        if not verification_flag_enabled:
            return False
        if not self._runtime_is_gazebo_simulation():
            return False
        payload = dict(bridge_debug or {})
        reasoning_mode = str(payload.get("reasoning_mode") or "").strip().lower()
        return reasoning_mode == "multi_turn"

    def _build_generated_code_verification(
        self,
        *,
        enabled: bool,
        bridge_debug: dict[str, Any] | None = None,
        prepared_bridge_request: dict[str, Any] | None = None,
        normalized_proposal: dict[str, Any] | None = None,
        status: str = "disabled",
        reason: str = "",
        verification_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        bridge_debug = dict(bridge_debug or {})
        prepared_bridge_request = dict(prepared_bridge_request or {})
        final_output = dict(bridge_debug.get("final_output") or {})
        artifacts = dict(bridge_debug.get("artifacts") or {})
        fixture_replay = dict(bridge_debug.get("fixture_replay") or {})
        session = dict(bridge_debug.get("multi_turn_session") or {})
        verification_payload: dict[str, Any] = {
            "enabled": bool(enabled),
            "status": str(status or "disabled").strip() or "disabled",
            "reason": str(reason or "").strip(),
            "verdict": "pending" if enabled else "disabled",
            "source_reasoning_mode": str(bridge_debug.get("reasoning_mode") or "").strip(),
            "source_final_output_stage": str(final_output.get("final_output_stage") or "").strip(),
            "source_artifact_path": (
                str(fixture_replay.get("source_path") or "").strip()
                or str(dict(artifacts.get("prepare") or {}).get("response_artifact_path") or "").strip()
            ),
            "source_turn_index": int(session.get("turn_index") or 0),
            "normalized_proposal_available": isinstance(normalized_proposal, dict),
            "executed_macro_ids": [],
            "snapshot_match_details": [],
            "updated_at_utc": self._utc_now_iso(),
        }
        if isinstance(verification_result, dict) and verification_result:
            verification_payload["result"] = deepcopy(verification_result)
            verdict = str(verification_result.get("verdict") or "").strip()
            if verdict:
                verification_payload["verdict"] = verdict
        if enabled and isinstance(final_output, dict) and final_output:
            verification_payload["source_final_output"] = {
                "final_output_stage": str(final_output.get("final_output_stage") or "").strip(),
                "accepted_trace_length": int(final_output.get("accepted_trace_length") or 0),
            }
        if isinstance(prepared_bridge_request, dict) and prepared_bridge_request:
            verification_payload["prepared_request_reasoning_mode"] = str(
                dict(prepared_bridge_request.get("bridge_session") or {}).get("reasoning_mode") or ""
            ).strip()
        return verification_payload

    @staticmethod
    def _runtime_bridge_fixture_final_output_path() -> str:
        return str(
            os.environ.get("CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT") or ""
        ).strip()

    def _runtime_bridge_fixture_final_output_source_path(self) -> str:
        fixture_path = self._runtime_bridge_fixture_final_output_path()
        if not fixture_path:
            return ""
        resolved_path = Path(fixture_path).expanduser()
        try:
            resolved_path = resolved_path.resolve()
        except Exception:
            pass
        return str(resolved_path)

    def _runtime_bridge_fixture_replay_enabled(self) -> bool:
        return bool(self._runtime_bridge_fixture_final_output_path())

    @staticmethod
    def _compact_fixture_replay_status(
        fixture_replay: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        payload = dict(fixture_replay or {})
        if not payload:
            return None
        return {
            "enabled": bool(payload.get("enabled")),
            "source_path": str(payload.get("source_path") or "").strip(),
            "load_status": str(payload.get("load_status") or "").strip(),
            "normalization_status": str(
                payload.get("normalization_status") or ""
            ).strip(),
            "reason": str(payload.get("reason") or "").strip(),
        }

    def _load_runtime_bridge_fixture_replay(
        self,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, Any]:
        fixture_path = self._runtime_bridge_fixture_final_output_path()
        result: dict[str, Any] = {
            "enabled": bool(fixture_path),
            "source_path": "",
            "load_status": "disabled",
            "normalization_status": "disabled",
            "reason": "",
            "final_output": None,
            "adapter_result": None,
            "proposal": None,
        }
        if not fixture_path:
            return result

        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        reasoning_mode = str(
            bridge_session.get("reasoning_mode") or ""
        ).strip().lower()
        resolved_path = Path(fixture_path).expanduser()
        try:
            resolved_path = resolved_path.resolve()
        except Exception:
            pass
        result["source_path"] = str(resolved_path)

        if reasoning_mode != "multi_turn":
            result["load_status"] = "skipped"
            result["normalization_status"] = "skipped"
            result["reason"] = (
                "runtime bridge fixture replay requires reasoning_mode=multi_turn"
            )
            return result

        if not resolved_path.exists():
            result["load_status"] = "missing"
            result["normalization_status"] = "skipped"
            result["reason"] = (
                f"fixture final_output artifact does not exist: {resolved_path}"
            )
            return result

        try:
            final_output_payload = json.loads(
                resolved_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            result["load_status"] = "invalid_json"
            result["normalization_status"] = "skipped"
            result["reason"] = (
                f"failed to parse fixture final_output artifact: {exc}"
            )
            return result

        if not isinstance(final_output_payload, dict) or not final_output_payload:
            result["load_status"] = "loaded"
            result["normalization_status"] = "rejected"
            result["reason"] = (
                "fixture final_output artifact did not contain a JSON object"
            )
            return result

        result["final_output"] = deepcopy(final_output_payload)
        result["load_status"] = "loaded"

        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn import (
            normalize_multi_turn_final_output_to_bridge_proposal,
        )

        adapter_result = normalize_multi_turn_final_output_to_bridge_proposal(
            final_output_payload=final_output_payload,
            prepared_bridge_request=prepared_bridge_request,
        )
        result["adapter_result"] = deepcopy(adapter_result)
        if isinstance(adapter_result, dict) and adapter_result.get("accepted") is True:
            result["normalization_status"] = "accepted"
            result["proposal"] = deepcopy(
                adapter_result.get("normalized_proposal") or {}
            )
            return result

        result["normalization_status"] = "rejected"
        result["reason"] = str(
            dict(adapter_result or {}).get("reason")
            or "fixture final_output normalization failed"
        ).strip()
        return result

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
        generated_code_verification = deepcopy(
            self.runtime_recovery.get("generated_code_verification") or {}
        )
        if generated_code_verification:
            generated_code_verification["status"] = "failed"
            generated_code_verification["verdict"] = "failed"
            generated_code_verification["updated_at_utc"] = self._utc_now_iso()
            generated_code_verification["result"] = {
                "verdict": "failed",
                "message": str(message or "").strip(),
                "failed_task_id": str(task_node.get("id", "")).strip(),
                "runtime_status": str(status or "").strip(),
                "observations": deepcopy(observations or {}),
            }
            if isinstance(self.runtime_recovery.get("bridge_debug"), dict):
                bridge_debug = deepcopy(self.runtime_recovery.get("bridge_debug") or {})
                bridge_debug["generated_code_verification"] = deepcopy(
                    generated_code_verification
                )
                self.runtime_recovery["bridge_debug"] = bridge_debug
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
            generated_code_verification=generated_code_verification or None,
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
        verification_only = self._bridge_is_verification_only(active_bridge_sequence)

        if isinstance(status, str) and status.startswith("failed"):
            retry_node = self._try_compile_release_retry_event(
                task_node=task_node,
                active_bridge_sequence=active_bridge_sequence,
                observations=observations,
                trigger=trigger,
                used_llm_bridge=used_llm_bridge,
                feedback_history=list(feedback_history),
            )
            if retry_node:
                await asyncio.to_thread(self._persist_plan_snapshot)
                await asyncio.to_thread(self._persist_product_state)
                return True

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
        generated_code_verification = deepcopy(
            self.runtime_recovery.get("generated_code_verification")
            or active_bridge_sequence.get("generated_code_verification")
            or {}
        )
        if generated_code_verification:
            executed_macro_ids = list(generated_code_verification.get("executed_macro_ids") or [])
            task_id = str(task_node.get("id", "")).strip()
            if task_id and task_id not in executed_macro_ids:
                executed_macro_ids.append(task_id)
            generated_code_verification["executed_macro_ids"] = executed_macro_ids
            snapshot_match_details = list(
                generated_code_verification.get("snapshot_match_details") or []
            )
            snapshot_match_details.append(
                {
                    "task_id": task_id,
                    "macro_name": macro_name,
                    "resource_jid": resource_jid,
                    "matched": True,
                    "actual_snapshot": deepcopy(actual_snapshot),
                    "projected_snapshot": deepcopy(projected_snapshot),
                    "part_name": part_name or None,
                    "projected_part_entry": deepcopy(projected_part_entry),
                }
            )
            generated_code_verification["snapshot_match_details"] = snapshot_match_details
            generated_code_verification["updated_at_utc"] = self._utc_now_iso()
        if verification_only:
            next_sequence = deepcopy(active_bridge_sequence)
            next_sequence["last_task_id"] = str(task_node.get("id", "")).strip()
            next_sequence["last_completed_macro_name"] = macro_name
            next_sequence["system_coordination_state"] = deepcopy(refreshed_system_state)
            if generated_code_verification:
                next_sequence["generated_code_verification"] = deepcopy(
                    generated_code_verification
                )
            bridge_debug = deepcopy(self.runtime_recovery.get("bridge_debug") or {})
            if generated_code_verification:
                bridge_debug["generated_code_verification"] = deepcopy(
                    generated_code_verification
                )
            if tail_task_ids:
                next_sequence["state"] = "executing"
                continue_message = (
                    f"Bridge macro '{macro_name}' matched projection; continuing generated-code verification tail."
                )
                self._set_runtime_recovery(
                    status="resolved",
                    resolution_class="generated_bridge_verification",
                    trigger=trigger,
                    failed_task_id=failed_task_id,
                    message=continue_message,
                    attempts_used=self._runtime_repair_fail_streak,
                    attempts_max=self._runtime_repair_max_attempts,
                    used_llm_bridge=used_llm_bridge,
                    bridge_proposal=None,
                    bridge_debug=bridge_debug if bridge_debug else None,
                    bridge_approval_state="approved",
                    active_bridge_sequence=next_sequence,
                    generated_code_verification=generated_code_verification or None,
                    bridge_feedback_history=feedback_history,
                    violations=[],
                    append_history=True,
                    history_message=continue_message,
                )
                await asyncio.to_thread(self._persist_product_state)
                return True

            disabled_frontier = self._bridge_continuation_disabled_frontier(next_sequence)
            if disabled_frontier:
                self._record_runtime_des_trace(
                    graph_ready_event_ids=list(
                        next_sequence.get("pending_nominal_task_ids") or []
                    ),
                    plant_enabled_event_ids=[],
                    disabled_frontier=disabled_frontier,
                )
                repair_node = self._try_compile_controllable_repair(
                    disabled_frontier=disabled_frontier,
                    trigger="bridge_continuation_guard",
                    active_bridge_sequence=next_sequence,
                )
                if repair_node:
                    bridge_debug = deepcopy(self.runtime_recovery.get("bridge_debug") or bridge_debug)
                    if generated_code_verification:
                        bridge_debug["generated_code_verification"] = deepcopy(
                            generated_code_verification
                        )
                        self.runtime_recovery["generated_code_verification"] = deepcopy(
                            generated_code_verification
                        )
                    await asyncio.to_thread(self._persist_plan_snapshot)
                    await asyncio.to_thread(self._persist_product_state)
                    return True

                message = (
                    "Generated bridge verification matched macro projections, but the "
                    "runtime DES continuation guard is still disabled. Human "
                    "intervention required."
                )
                self._mark_runtime_des_human_required(
                    disabled_frontier=disabled_frontier,
                    message=message,
                )
                await asyncio.to_thread(self._persist_product_state)
                return True

            next_sequence["state"] = "verified"
            if generated_code_verification:
                generated_code_verification["status"] = "verified"
                generated_code_verification["verdict"] = "passed"
                generated_code_verification["updated_at_utc"] = self._utc_now_iso()
                generated_code_verification["result"] = {
                    "verdict": "passed",
                    "executed_macro_ids": deepcopy(
                        generated_code_verification.get("executed_macro_ids") or []
                    ),
                    "snapshot_match_details": deepcopy(
                        generated_code_verification.get("snapshot_match_details") or []
                    ),
                    "message": (
                        "Generated bridge macro sequence executed in Gazebo and matched the projected post-state."
                    ),
                }
                next_sequence["generated_code_verification"] = deepcopy(
                    generated_code_verification
                )
                bridge_debug["generated_code_verification"] = deepcopy(
                    generated_code_verification
                )
            verified_message = (
                f"Generated bridge verification completed after macro '{macro_name}'; "
                "resuming nominal execution."
            )
            if next_sequence:
                bridge_debug["verified_bridge_sequence"] = deepcopy(next_sequence)
            self._runtime_recovery_context = {}
            self._set_runtime_recovery(
                status="resolved",
                resolution_class="generated_bridge_verified",
                trigger=trigger,
                failed_task_id=failed_task_id,
                message=verified_message,
                attempts_used=self._runtime_repair_fail_streak,
                attempts_max=self._runtime_repair_max_attempts,
                used_llm_bridge=used_llm_bridge,
                bridge_proposal=None,
                bridge_debug=bridge_debug if bridge_debug else None,
                bridge_approval_state="approved",
                active_bridge_sequence=None,
                generated_code_verification=generated_code_verification or None,
                bridge_feedback_history=feedback_history,
                violations=[],
                append_history=True,
                history_message=verified_message,
            )
            self._clear_plan_safety_alert()
            await asyncio.to_thread(self._persist_plan_snapshot)
            await asyncio.to_thread(self._persist_product_state)
            await asyncio.to_thread(self._persist_resource_state)
            return True

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
        fixture_replay_enabled = self._runtime_bridge_fixture_replay_enabled()
        fixture_source_path = self._runtime_bridge_fixture_final_output_source_path()
        self.logger.info(
            "[Product] Runtime bridge fixture replay gate: fixture_replay=%s source_path=%s env_var=CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT",
            fixture_replay_enabled,
            fixture_source_path or "<unset>",
        )
        bridge_generation_mode = str(self._bridge_generation_mode or "auto").strip().lower()
        if (scenario_hint or fixture_replay_enabled) and bridge_generation_mode != "manual":
            self.logger.info(
                "[Product] Forcing manual bridge handoff for runtime bridge replay: scenario_id=%s fixture_replay=%s mode=%s->manual",
                scenario_hint,
                fixture_replay_enabled,
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
                if scenario_hint and allow_preprogrammed_autoload and not fixture_replay_enabled:
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
                elif scenario_hint and allow_preprogrammed_autoload and fixture_replay_enabled:
                    self.logger.info(
                        "[Product] Skipping preprogrammed recovery auto-load because fixture replay is enabled: scenario_id=%s",
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
                verification_ready = self._should_enable_generated_bridge_verification(
                    bridge_debug=bridge_debug if isinstance(bridge_debug, dict) else None,
                )
                if fixture_replay_enabled or (not scenario_hint and verification_ready):
                    if fixture_replay_enabled:
                        self.logger.info(
                            "[Product] Auto-starting runtime bridge fixture replay from prepared request: source_path=%s verification_ready=%s",
                            self._runtime_bridge_fixture_final_output_path(),
                            verification_ready,
                        )
                    else:
                        self.logger.info(
                            "[Product] Auto-starting multi-turn bridge generation for Gazebo verification."
                        )
                    self._runtime_repair_inflight = False
                    return await self.generate_runtime_bridge_proposal()
                if not fixture_replay_enabled:
                    self.logger.info(
                        "[Product] Runtime bridge fixture replay disabled at prepare-trace checkpoint: "
                        "CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT is not set; staying at bridge_ready."
                    )
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
        active_bridge_sequence = self._active_bridge_sequence()
        active_bridge_state = str(
            (active_bridge_sequence or {}).get("state") or ""
        ).strip().lower()
        if (
            current_status == "resolved"
            and active_bridge_sequence
            and active_bridge_state in {"approved", "executing"}
        ):
            self.logger.warning(
                "[Product] Runtime bridge sequence already executing for %s; ignoring duplicate replan request for %s.",
                self.runtime_recovery.get("failed_task_id") or failed_task_id,
                failed_task_id,
            )
            return self.get_runtime_recovery()
        if current_status in {
            "des_search",
            "bridge_ready",
            "llm_bridge",
            "validating",
            "human_required",
            "generated_bridge_verified",
        }:
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
            verification_only = self._bridge_is_verification_only(active_bridge_sequence)
            resolution_class = (
                "generated_bridge_verification"
                if verification_only
                else "des_with_llm_bridge"
                if bool(self.runtime_recovery.get("used_llm_bridge", False))
                else "des_only"
            )
            attempts_used = self._runtime_repair_fail_streak
            self._runtime_repair_fail_streak = 0
            success_message = (
                "Plan validation passed; generated bridge verification is executing."
                if active_bridge_sequence and verification_only
                else "Plan validation passed; approved bridge sequence is executing."
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
                generated_code_verification=self.runtime_recovery.get("generated_code_verification"),
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

    async def _execute_runtime_bridge_generation(
        self,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, Any] | None:
        fixture_replay = self._load_runtime_bridge_fixture_replay(prepared_bridge_request)
        if bool(fixture_replay.get("enabled")):
            bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
            bridge_debug["fixture_replay"] = (
                self._compact_fixture_replay_status(fixture_replay) or {}
            )
            if isinstance(fixture_replay.get("final_output"), dict):
                bridge_debug["final_output"] = deepcopy(
                    fixture_replay.get("final_output") or {}
                )
            if isinstance(fixture_replay.get("adapter_result"), dict):
                bridge_debug["final_output_adapter"] = deepcopy(
                    fixture_replay.get("adapter_result") or {}
                )
            if isinstance(fixture_replay.get("proposal"), dict):
                bridge_debug["normalized_proposal"] = deepcopy(
                    fixture_replay.get("proposal") or {}
                )
                bridge_debug["status"] = "fixture_replay_ready"
                bridge_debug["message"] = (
                    "Loaded runtime bridge proposal from archived final_output fixture."
                )
            else:
                bridge_debug["status"] = "fixture_replay_failed"
                bridge_debug["message"] = str(
                    fixture_replay.get("reason")
                    or "Runtime bridge fixture replay failed."
                ).strip()
            prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
            if hasattr(self.process_planner, "_set_last_bridge_debug"):
                self.process_planner._set_last_bridge_debug(bridge_debug)
            proposal = fixture_replay.get("proposal")
            return deepcopy(proposal) if isinstance(proposal, dict) else None

        proposal = await self.process_planner.execute_prepared_bridge_request(
            prepared_bridge_request
        )
        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        reasoning_mode = str(bridge_session.get("reasoning_mode") or "").strip().lower()
        if reasoning_mode != "multi_turn":
            return proposal

        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
            execute_multi_turn_bridge as _resume_multi_turn_bridge,
        )

        max_resume = max(
            10,
            int(
                bridge_session.get("max_turns")
                or dict(prepared_bridge_request.get("multi_turn_session_seed") or {}).get("max_turns")
                or 0
            ),
        )
        for resume_idx in range(max_resume):
            session_state = dict(prepared_bridge_request.get("multi_turn_session_state") or {})
            pause_status = str(session_state.get("status") or "").strip().lower()
            if pause_status not in {"paused_after_outline_turn", "paused_after_primitive_turn"}:
                break
            self.logger.info(
                "[Product] Resuming multi-turn bridge session: status=%s round=%d/%d",
                pause_status,
                resume_idx + 1,
                max_resume,
            )
            proposal = await _resume_multi_turn_bridge(
                self.process_planner,
                prepared_bridge_request,
                session_state=session_state,
            )
        return proposal

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
            if scenario_hint and not self._runtime_bridge_fixture_replay_enabled():
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
        fixture_replay_payload = self._compact_fixture_replay_status(
            bridge_debug.get("fixture_replay")
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
            fixture_replay=fixture_replay_payload,
            violations=violations,
            append_history=True,
            history_message="Operator started LLM bridge exploration from the prepared request.",
        )
        await asyncio.to_thread(self._persist_product_state)

        self._runtime_repair_inflight = True
        try:
            proposal = await self._execute_runtime_bridge_generation(
                prepared_bridge_request
            )
            bridge_debug = deepcopy(
                prepared_bridge_request.get("bridge_debug")
                or self.process_planner.get_last_bridge_debug()
                or {}
            )
            fixture_replay_payload = self._compact_fixture_replay_status(
                bridge_debug.get("fixture_replay")
            )
            verification_enabled = self._should_enable_generated_bridge_verification(
                bridge_debug=bridge_debug,
            )
            verification_payload: dict[str, Any] | None = None
            if str(bridge_debug.get("reasoning_mode") or "").strip().lower() == "multi_turn":
                verification_payload = self._build_generated_code_verification(
                    enabled=verification_enabled,
                    bridge_debug=bridge_debug,
                    prepared_bridge_request=prepared_bridge_request,
                    normalized_proposal=proposal if isinstance(proposal, dict) else None,
                    status="ready" if isinstance(proposal, dict) else "generating",
                    reason=(
                        ""
                        if verification_enabled
                        else "Gazebo generated-code verification is disabled."
                    ),
                )
                bridge_debug["generated_code_verification"] = deepcopy(
                    verification_payload
                )
                if isinstance(proposal, dict) and verification_enabled:
                    bridge_debug["execution_policy"] = {
                        "complete_full_tail": False,
                        "verification_only": True,
                        "pause_after_verification": True,
                    }
            prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
            self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(
                prepared_bridge_request
            )
            self._record_runtime_bridge_artifacts(
                phase="multi_turn",
                prepared_bridge_request=prepared_bridge_request,
            )
            self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(
                prepared_bridge_request
            )
            bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or bridge_debug or {})
            if isinstance(proposal, dict):
                bridge_summary = self.process_planner._bridge_summary(proposal)
                bridge_text = ", ".join(str(item) for item in bridge_summary if item) or "bridge step(s)"
                recovery = self._set_runtime_recovery(
                    status="llm_bridge",
                    resolution_class="none",
                    trigger=str(self._runtime_recovery_context.get("trigger", "")),
                    failed_task_id=failed_task_id,
                    message=(
                        "Validated replayed bridge proposal is ready for final approval."
                        if fixture_replay_payload
                        else "Validated LLM bridge proposal is ready for final approval."
                    ),
                    attempts_used=self._runtime_repair_fail_streak,
                    attempts_max=self._runtime_repair_max_attempts,
                    used_llm_bridge=True,
                    bridge_proposal=proposal,
                    bridge_debug=bridge_debug if bridge_debug else None,
                    bridge_approval_state="pending",
                    active_bridge_sequence=None,
                    fixture_replay=fixture_replay_payload,
                    generated_code_verification=verification_payload,
                    bridge_feedback_history=list(
                        self._runtime_recovery_context.get("bridge_feedback_history") or []
                    ),
                    violations=violations,
                    append_history=True,
                    history_message=(
                        f"Fixture replay loaded {bridge_text} from archived final output."
                        if fixture_replay_payload
                        else f"LLM bridge proposed {bridge_text}."
                    ),
                )
                self._clear_plan_safety_alert()
                await asyncio.to_thread(self._persist_product_state)
                if verification_enabled:
                    self.logger.info(
                        "[Product] Auto-approving multi-turn bridge proposal for Gazebo verification."
                    )
                    self._runtime_repair_inflight = False
                    return self.approve_runtime_bridge_proposal_sync()
                return recovery

            if fixture_replay_payload:
                message = str(
                    fixture_replay_payload.get("reason")
                    or "Runtime bridge fixture replay failed before producing a validated proposal."
                ).strip()
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
                    fixture_replay=fixture_replay_payload,
                    generated_code_verification=verification_payload,
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
                    fixture_replay=fixture_replay_payload,
                    generated_code_verification=verification_payload,
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
            if bridge_status in {"paused_after_outline_turn", "paused_after_primitive_turn"}:
                message = (
                    "LLM bridge paused for another bounded multi-turn reasoning step."
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
                    fixture_replay=fixture_replay_payload,
                    generated_code_verification=verification_payload,
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
            if bridge_status in {"paused_after_primitive_blocked", "paused_after_primitive_stuck"}:
                message = (
                    "LLM bridge primitive generation stalled before producing a validated final plan."
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
                    fixture_replay=fixture_replay_payload,
                    generated_code_verification=verification_payload,
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
                fixture_replay=fixture_replay_payload,
                generated_code_verification=verification_payload,
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
                fixture_replay=self.runtime_recovery.get("fixture_replay"),
                generated_code_verification=self.runtime_recovery.get("generated_code_verification"),
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
        generated_code_verification = deepcopy(
            self.runtime_recovery.get("generated_code_verification") or {}
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
        execution_policy = (
            deepcopy(bridge_debug.get("execution_policy") or {})
            if isinstance(bridge_debug, dict)
            else {}
        )
        verification_only_approval = bool(execution_policy.get("verification_only"))
        bridge_replaces_failed_branch = (
            bool(plan_rewrite.get("replace_failed_branch"))
            and not verification_only_approval
        )
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
        if isinstance(plan_rewrite, dict) and not verification_only_approval:
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
        if isinstance(plan_rewrite, dict) and not verification_only_approval:
            resume_task_ids_explicit = "resume_task_ids" in plan_rewrite
            for task_id in plan_rewrite.get("resume_task_ids") or []:
                candidate = str(task_id or "").strip()
                if (
                    candidate
                    and candidate not in resumable_task_ids
                    and candidate not in deleted_task_ids
                ):
                    resumable_task_ids.append(candidate)
        if (
            not verification_only_approval
            and not resumable_task_ids
            and not resume_task_ids_explicit
        ):
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
            "[Product] Bridge approval rewrite: anchor=%s replace_failed_branch=%s verification_only=%s delete=%s resume=%s",
            anchor_task_id or "<none>",
            bridge_replaces_failed_branch,
            verification_only_approval,
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
                generated_code_verification=generated_code_verification or None,
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
            prepared_bridge_request = dict(
                self._runtime_recovery_context.get("prepared_bridge_request") or {}
            )
            modeled_gap = dict(
                dict(prepared_bridge_request.get("context_summary") or {}).get(
                    "modeled_continuation_gap"
                )
                or {}
            )
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
                "continuation_requirements": deepcopy(
                    modeled_gap.get("continuation_requirements") or []
                ),
                "pending_nominal_task_ids": deepcopy(
                    modeled_gap.get("pending_nominal_task_ids") or []
                ),
                "continuation_repair_attempts": 0,
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
                if generated_code_verification:
                    active_bridge_sequence["generated_code_verification"] = deepcopy(
                        generated_code_verification
                    )
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
            if generated_code_verification:
                generated_code_verification["status"] = "executing"
                generated_code_verification["updated_at_utc"] = self._utc_now_iso()
                bridge_debug["generated_code_verification"] = deepcopy(
                    generated_code_verification
                )
                if isinstance(active_bridge_sequence, dict):
                    active_bridge_sequence["generated_code_verification"] = deepcopy(
                        generated_code_verification
                    )

        if verification_only_approval and isinstance(active_bridge_sequence, dict):
            active_bridge_sequence["state"] = "executing"
            execution_message = (
                f"Approved generated bridge verification proposal compiled to {len(tasks)} "
                "task(s); executing bridge macros without full-plan suffix validation."
            )
            recovery = self._set_runtime_recovery(
                status="resolved",
                resolution_class="generated_bridge_verification",
                trigger=str(self._runtime_recovery_context.get("trigger", "")),
                failed_task_id=failed_task_id,
                message=execution_message,
                attempts_used=self._runtime_repair_fail_streak,
                attempts_max=self._runtime_repair_max_attempts,
                used_llm_bridge=used_llm_bridge,
                bridge_proposal=None,
                bridge_debug=bridge_debug if bridge_debug else None,
                bridge_approval_state="approved",
                active_bridge_sequence=active_bridge_sequence,
                generated_code_verification=generated_code_verification or None,
                violations=[],
                append_history=True,
                history_message=execution_message,
            )
            self._clear_plan_safety_alert()
            self.logger.info(
                "[Product] Approved generated bridge verification proposal compiled; "
                "skipping CCA full-plan validation and executing bridge macros."
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
            return recovery

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
            generated_code_verification=generated_code_verification or None,
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
                generated_code_verification=generated_code_verification or None,
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
                    await self.send(msg)

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

            if updated_node and not handled_bridge_ack:
                await asyncio.to_thread(agent._persist_plan_snapshot)
                await asyncio.to_thread(agent._persist_product_state)
                await asyncio.to_thread(agent._persist_resource_state)

            agent.logger.info(
                f"[Product] ACK ({task_id}) status='{status}' from={msg.sender}"
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
                await self.send(check_msg)
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

            # Prioritize active bridge sequences over nominal DAG work. The first
            # recovery macro may be anchored after the failed task, which is
            # intentionally not "completed" during runtime recovery.
            task_node = agent._next_dispatchable_task_node()
            if not task_node and agent._active_bridge_blocks_nominal_dispatch():
                agent.logger.debug(
                    "[Product] Active bridge sequence is executing; suppressing nominal DAG dispatch."
                )
                await asyncio.sleep(0.05)
                return
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
            # Recovery macro primitive steps own their destination intent.
            params = agent._dispatch_params_for_task_node(task_node)

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

            await self.send(msg)
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

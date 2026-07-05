"""Runtime recovery, bridge session, and validation helpers for ProductAgent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spade.message import Message

from cais_spade_llm.agents.central_controller.online_safety_monitor import (
    OnlineSafetyMonitor,
)
from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_artifacts import (
    DEFAULT_BRIDGE_DEBUG_DIR,
    write_bridge_artifacts,
)
from cais_spade_llm.agents.intelligent_product.replanner.preprogrammed_bridge_scenarios import (
    build_preprogrammed_bridge_proposal,
)
from cais_spade_llm.agents.shared_information.llm_agent import LlmAgent
from cais_spade_llm.product.profile import ProductProfile

_UNSET = object()


def _env_flag_enabled(*names: str, default: bool = False) -> bool:
    for name in names:
        token = str(os.environ.get(name) or "").strip().lower()
        if not token:
            continue
        return token in {"1", "true", "yes", "on"}
    return bool(default)

class ProductRecoveryController:
    EXPORTED_METHODS = (
        '_build_plan_safety_alert',
        '_set_plan_safety_alert',
        '_clear_plan_safety_alert',
        'get_plan_safety_alert',
        '_normalize_runtime_bridge_mode',
        '_normalize_runtime_bridge_validation_policy',
        'set_runtime_bridge_session_settings',
        '_runtime_bridge_session_mode',
        '_runtime_bridge_session_validation_policy',
        '_runtime_bridge_session_archive_path',
        '_runtime_bridge_session_archive_label',
        '_runtime_bridge_session_artifact_directory',
        '_bridge_debug_with_runtime_handoff',
        '_runtime_bridge_turn_token',
        '_runtime_bridge_progress_message',
        'report_runtime_bridge_turn_progress',
        '_runtime_action_feedback_payload',
        '_recovery_result_with_action_feedback',
        '_bridge_sequence_state',
        '_bridge_sequence_is_live',
        '_bridge_sequence_task_ids',
        '_bridge_proposal_fingerprint',
        '_bridge_sequence_summary',
        '_should_ignore_stale_recovery_macro_ack',
        '_bridge_sequence_with_dispatched_task',
        '_bridge_sequence_with_completed_task',
        '_derive_runtime_bridge_stage',
        '_empty_runtime_recovery',
        '_sync_runtime_repair_state',
        '_set_runtime_recovery',
        '_clear_runtime_recovery',
        '_normalized_xyz_pose',
        '_observed_pose_for_part',
        '_tracked_pose_for_part_at_location',
        '_support_surface_place_pose_from_observations',
        '_part_geometry_for_pick_context',
        '_enrich_observed_pose_recovery_params',
        '_runtime_safety_fast_path_monitor',
        '_runtime_safety_ap_sets_for_task',
        '_runtime_safety_task_ap_empty',
        '_dispatch_params_for_task_node',
        '_runtime_recovery_blocks_execution',
        'get_runtime_recovery',
        '_runtime_bridge_data_root',
        '_resolve_runtime_bridge_archive_path',
        '_bridge_feedback_history_with',
        '_append_bridge_feedback',
        '_bridge_session_state',
        '_reset_session_for_outline_retry',
        '_reset_session_for_primitive_retry',
        '_prepare_runtime_bridge_session_state',
        '_bridge_proposal_from_debug',
        '_load_runtime_bridge_archive_bundle',
        '_run_multi_turn_bridge_until',
        '_set_kickoff_result',
        'wait_for_kickoff_result',
        '_persist_plan_snapshot',
        '_persist_product_state',
        '_bridge_debug_directory',
        '_ensure_live_bridge_per_turn_debug_dir',
        '_build_runtime_bridge_artifact_payload',
        '_record_runtime_bridge_artifacts',
        '_runtime_recovery_safety_live_root',
        '_allocate_runtime_recovery_safety_worked_dir',
        '_recovery_safety_generation_dirs',
        '_build_recovery_safety_generation_payload',
        '_dispatch_recovery_safety_generation_request',
        '_handle_recovery_safety_generated_result',
        '_persist_resource_state',
        '_get_part_transition',
        '_tracked_part_name_for_task',
        '_apply_part_tracker_update',
        '_reactivate_blocked_tasks',
        '_handle_task_retry_ready',
        '_reactivate_restored_repair_target_from_context',
        '_candidate_task_ids_from_violations',
        '_active_bridge_sequence',
        '_bridge_used_llm',
        '_bridge_execution_policy',
        '_bridge_requires_complete_full_tail',
        '_bridge_is_verification_only',
        '_reconstruct_active_bridge_sequence_for_validation',
        '_active_bridge_blocks_nominal_dispatch',
        '_next_dispatchable_task_node',
        '_active_bridge_next_ready_task',
        '_bridge_task_predecessors_ready',
        '_select_runtime_event',
        '_graph_ready_task_nodes',
        '_tool_row_for_task_node',
        '_event_contract_for_task_node',
        '_build_runtime_plant_state',
        '_plant_resource_field',
        '_plant_part_field',
        '_unknown_runtime_value',
        '_event_guard_violations',
        '_record_runtime_des_trace',
        '_resource_can_reach_location',
        '_model_name_from_mapping',
        '_part_model_name_for_repair',
        '_part_geometry_for_repair',
        '_repair_execution_mode_for_resource',
        '_resolve_acquire_entity_pick_source',
        '_repair_primitive_catalog_for_resource',
        '_validate_repair_primitive_program',
        '_record_repair_compile_error',
        '_strip_continuation_requirement_actuals',
        '_runtime_des_disabled_event_message',
        '_failed_release_primitive_observation',
        '_try_compile_release_retry_event',
        '_try_compile_controllable_repair',
        '_append_acquire_entity_repair_event',
        '_mark_runtime_des_human_required',
        '_bridge_continuation_disabled_frontier',
        '_runtime_is_gazebo_simulation',
        '_should_enable_generated_bridge_verification',
        '_build_generated_code_verification',
        '_runtime_bridge_fixture_final_output_path',
        '_runtime_bridge_fixture_final_output_source_path',
        '_runtime_bridge_fixture_replay_enabled',
        '_compact_fixture_replay_status',
        '_load_runtime_bridge_fixture_replay',
        '_bridge_task_debug_rows',
        '_bridge_sequence_tail_task_ids',
        '_refresh_bridge_snapshot',
        '_system_coordination_state_with_bridge_snapshot',
        '_bridge_snapshot_mismatch',
        '_bridge_part_entry_mismatch',
        '_dispatch_runtime_plan_validation_check',
        '_send_runtime_plan_validation_check',
        '_send_runtime_plan_validation_check_sync',
        '_fail_closed_bridge_sequence',
        '_handle_bridge_macro_ack',
        '_run_des_runtime_recovery_attempt',
        '_handle_runtime_des_replan_request',
        '_handle_runtime_plan_validation_result',
        'submit_runtime_recovery_guidance',
        '_execute_runtime_bridge_generation',
        'generate_runtime_bridge_proposal',
        'load_runtime_bridge_archive_proposal',
        'approve_runtime_bridge_outline',
        'refine_runtime_bridge_outline',
        'reject_runtime_bridge_outline',
        'approve_runtime_bridge_primitives',
        'refine_runtime_bridge_primitives',
        'reject_runtime_bridge_primitives',
        '_build_preprogrammed_runtime_bridge_bundle',
        '_derive_preprogrammed_part_observations',
        '_cache_preprogrammed_runtime_bridge_scenario',
        '_resolve_preprogrammed_runtime_bridge_bundle',
        'load_preprogrammed_runtime_bridge_scenario_sync',
        'load_preprogrammed_runtime_bridge_scenario',
        'approve_runtime_bridge_proposal_sync',
        'approve_runtime_bridge_proposal',
        'reject_runtime_bridge_proposal',
        'retry_runtime_recovery_des',
    )

    def __init__(self, agent: Any) -> None:
        object.__setattr__(self, "_agent", agent)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_agent":
            object.__setattr__(self, name, value)
            return
        setattr(self._agent, name, value)

    def bind_methods(self) -> None:
        for name in self.EXPORTED_METHODS:
            setattr(self._agent, name, getattr(self, name))

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

    @staticmethod
    def _normalize_runtime_bridge_mode(value: Any) -> str:
        token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if token in {"auto", "manual", "pre_ran"}:
            return token
        if token == "preran":
            return "pre_ran"
        return "pre_ran"

    @staticmethod
    def _normalize_runtime_bridge_validation_policy(value: Any) -> str:
        token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if token in {"validated", "no_validation"}:
            return token
        if token in {"unvalidated", "skip_validation", "no_validation_mode"}:
            return "no_validation"
        return "validated"

    def set_runtime_bridge_session_settings(
        self,
        *,
        mode: str | None = None,
        validation_policy: str | None = None,
        selected_archive_path: str | None = None,
        selected_archive_label: str | None = None,
    ) -> None:
        if mode is not None:
            self._runtime_bridge_mode = self._normalize_runtime_bridge_mode(mode)
        if validation_policy is not None:
            self._runtime_bridge_validation_policy = (
                self._normalize_runtime_bridge_validation_policy(validation_policy)
            )
        if selected_archive_path is not None:
            self._runtime_bridge_archive_path = str(selected_archive_path or "").strip()
        if selected_archive_label is not None:
            self._runtime_bridge_archive_label = str(selected_archive_label or "").strip()

        if self._runtime_recovery_context:
            status = str(self.runtime_recovery.get("status", "") or "").strip().lower()
            self._runtime_recovery_context["bridge_mode"] = self._normalize_runtime_bridge_mode(
                self._runtime_bridge_mode
            )
            self._runtime_recovery_context["validation_policy"] = (
                self._normalize_runtime_bridge_validation_policy(
                    self._runtime_bridge_validation_policy
                )
            )
            self._runtime_recovery_context["selected_archive_path"] = str(
                self._runtime_bridge_archive_path or ""
            ).strip()
            self._runtime_recovery_context["selected_archive_label"] = str(
                self._runtime_bridge_archive_label or ""
            ).strip()
            prepared_bridge_request = deepcopy(
                self._runtime_recovery_context.get("prepared_bridge_request") or {}
            )
            if prepared_bridge_request:
                bridge_debug = self._bridge_debug_with_runtime_handoff(
                    dict(prepared_bridge_request.get("bridge_debug") or {}),
                    bridge_mode=self._normalize_runtime_bridge_mode(
                        self._runtime_bridge_mode
                    ),
                    validation_policy=self._normalize_runtime_bridge_validation_policy(
                        self._runtime_bridge_validation_policy
                    ),
                )
                prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
                self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(
                    prepared_bridge_request
                )
            if status in {"bridge_ready", "llm_bridge", "human_required"}:
                self._set_runtime_recovery(
                    message=str(self.runtime_recovery.get("message", "") or "").strip(),
                )

    def _runtime_bridge_session_mode(self) -> str:
        session_mode = str(
            dict(getattr(self, "_runtime_recovery_context", {}) or {}).get("bridge_mode") or ""
        ).strip()
        if session_mode:
            return self._normalize_runtime_bridge_mode(session_mode)
        return self._normalize_runtime_bridge_mode(self._runtime_bridge_mode)

    def _runtime_bridge_session_validation_policy(self) -> str:
        runtime_context = dict(getattr(self, "_runtime_recovery_context", {}) or {})
        if runtime_context:
            candidate = str(runtime_context.get("validation_policy") or "").strip()
            if candidate:
                return self._normalize_runtime_bridge_validation_policy(candidate)
        return self._normalize_runtime_bridge_validation_policy(
            self._runtime_bridge_validation_policy
        )

    def _runtime_bridge_session_archive_path(self) -> str:
        runtime_context = dict(getattr(self, "_runtime_recovery_context", {}) or {})
        if runtime_context:
            candidate = str(
                runtime_context.get("selected_archive_path") or ""
            ).strip()
            if candidate:
                return candidate
        return str(self._runtime_bridge_archive_path or "").strip()

    def _runtime_bridge_session_archive_label(self) -> str:
        runtime_context = dict(getattr(self, "_runtime_recovery_context", {}) or {})
        if runtime_context:
            candidate = str(
                runtime_context.get("selected_archive_label") or ""
            ).strip()
            if candidate:
                return candidate
        return str(self._runtime_bridge_archive_label or "").strip()

    def _runtime_bridge_session_artifact_directory(self) -> str:
        runtime_context = dict(getattr(self, "_runtime_recovery_context", {}) or {})
        if runtime_context:
            candidate = str(runtime_context.get("artifact_directory") or "").strip()
            if candidate:
                return candidate
        runtime_recovery = dict(getattr(self, "runtime_recovery", {}) or {})
        candidate = str(runtime_recovery.get("artifact_directory") or "").strip()
        if candidate:
            return candidate
        recovery_debug = dict(runtime_recovery.get("bridge_debug") or {})
        if isinstance(recovery_debug, dict):
            candidate = str(recovery_debug.get("artifact_directory") or "").strip()
            if candidate:
                return candidate
        return ""

    @staticmethod
    def _bridge_debug_with_runtime_handoff(
        bridge_debug: dict[str, Any] | None,
        **updates: Any,
    ) -> dict[str, Any]:
        updated = deepcopy(bridge_debug or {})
        runtime_handoff = dict(updated.get("runtime_handoff") or {})
        for key, value in updates.items():
            runtime_handoff[key] = deepcopy(value)
        updated["runtime_handoff"] = runtime_handoff
        return updated

    @staticmethod
    def _runtime_bridge_turn_token(turn_index: int, max_turns: int) -> str:
        max_turns = max(0, int(max_turns or 0))
        turn_index = max(0, int(turn_index or 0))
        width = max(2, len(str(max_turns or turn_index or 0)))
        return f"{turn_index:0{width}d}/{max_turns:0{width}d}"

    @classmethod
    def _runtime_bridge_progress_message(
        cls,
        *,
        session_state: dict[str, Any] | None,
        current_phase: str,
        status_label: str,
        decision: str = "",
        next_phase: str = "",
        elapsed_s: float | None = None,
    ) -> str:
        session = dict(session_state or {})
        turn_token = cls._runtime_bridge_turn_token(
            int(session.get("turn_index") or 0),
            int(session.get("max_turns") or 0),
        )
        phase_label = str(current_phase or session.get("current_phase") or "grounding").strip() or "grounding"
        status_key = str(status_label or "running").strip().lower()
        decision_label = str(decision or "").strip()
        next_phase_label = str(next_phase or "").strip()
        if status_key == "waiting_for_llm":
            return f"Live bridge turn {turn_token}: {phase_label} (waiting for LLM)."
        if status_key == "still_waiting_for_llm":
            elapsed_text = (
                f"{max(0.0, float(elapsed_s)):.1f}s elapsed"
                if elapsed_s is not None
                else "waiting"
            )
            return f"Live bridge turn {turn_token}: {phase_label} (still waiting for LLM, {elapsed_text})."
        if status_key == "response_received":
            return f"Live bridge turn {turn_token}: {phase_label} (LLM response received)."
        if status_key == "decision":
            tail = ""
            if decision_label and next_phase_label:
                tail = f"decision={decision_label}, next={next_phase_label}"
            elif decision_label:
                tail = f"decision={decision_label}"
            elif next_phase_label:
                tail = f"next={next_phase_label}"
            if tail:
                return f"Live bridge turn {turn_token}: {phase_label} ({tail})."
        return f"Live bridge turn {turn_token}: {phase_label}."

    async def report_runtime_bridge_turn_progress(
        self,
        *,
        session_state: dict[str, Any] | None,
        current_phase: str,
        status_label: str,
        decision: str = "",
        next_phase: str = "",
        elapsed_s: float | None = None,
    ) -> None:
        if not self._runtime_recovery_context:
            return
        status = str(self.runtime_recovery.get("status", "") or "").strip().lower()
        if status != "llm_bridge":
            return

        session_snapshot = deepcopy(session_state or {})
        prepared_bridge_request = deepcopy(
            self._runtime_recovery_context.get("prepared_bridge_request") or {}
        )
        bridge_debug = deepcopy(
            prepared_bridge_request.get("bridge_debug")
            or self.runtime_recovery.get("bridge_debug")
            or {}
        )
        bridge_debug["status"] = str(session_snapshot.get("status") or "running").strip() or "running"
        bridge_debug["multi_turn_session"] = deepcopy(session_snapshot)
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        prepared_bridge_request["multi_turn_session_state"] = deepcopy(session_snapshot)
        self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(
            prepared_bridge_request
        )
        if hasattr(self.process_planner, "_set_last_bridge_debug"):
            self.process_planner._set_last_bridge_debug(bridge_debug)

        message = self._runtime_bridge_progress_message(
            session_state=session_snapshot,
            current_phase=current_phase,
            status_label=status_label,
            decision=decision,
            next_phase=next_phase,
            elapsed_s=elapsed_s,
        )
        self._set_runtime_recovery(
            status="llm_bridge",
            resolution_class="none",
            message=message,
            used_llm_bridge=True,
            bridge_debug=bridge_debug if bridge_debug else None,
            bridge_approval_state="generating",
            bridge_stage="none",
            append_history=False,
        )
        await asyncio.to_thread(self._persist_product_state)

    @staticmethod
    def _runtime_action_feedback_payload(kind: str, text: str) -> dict[str, str] | None:
        message = str(text or "").strip()
        if not message:
            return None
        return {
            "kind": str(kind or "info").strip().lower() or "info",
            "text": message,
        }

    def _recovery_result_with_action_feedback(
        self,
        recovery: dict[str, Any] | None = None,
        *,
        kind: str,
        text: str,
    ) -> dict[str, Any]:
        result = (
            deepcopy(recovery)
            if isinstance(recovery, dict)
            else self.get_runtime_recovery()
        )
        payload = self._runtime_action_feedback_payload(kind, text)
        if payload is not None:
            result["action_feedback"] = payload
        return result

    @staticmethod
    def _bridge_sequence_state(active_bridge_sequence: dict[str, Any] | None) -> str:
        return str(
            (active_bridge_sequence or {}).get("state") or ""
        ).strip().lower()

    @classmethod
    def _bridge_sequence_is_live(cls, active_bridge_sequence: dict[str, Any] | None) -> bool:
        return cls._bridge_sequence_state(active_bridge_sequence) in {"approved", "executing"}

    @staticmethod
    def _bridge_sequence_task_ids(values: Any) -> list[str]:
        deduped: list[str] = []
        for value in values or []:
            task_id = str(value or "").strip()
            if task_id and task_id not in deduped:
                deduped.append(task_id)
        return deduped

    def _bridge_proposal_fingerprint(
        self,
        *,
        proposal: dict[str, Any],
        failed_task_id: str,
        bridge_debug: dict[str, Any] | None = None,
    ) -> str:
        archive_replay = dict((bridge_debug or {}).get("archive_replay") or {})
        payload = {
            "failed_task_id": str(failed_task_id or "").strip(),
            "bridge_mode": self._runtime_bridge_session_mode(),
            "validation_policy": self._runtime_bridge_session_validation_policy(),
            "selected_archive_path": str(
                archive_replay.get("source_path")
                or self._runtime_bridge_session_archive_path()
                or ""
            ).strip(),
            "source": str((bridge_debug or {}).get("source") or "").strip(),
            "proposal": deepcopy(proposal or {}),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _bridge_sequence_summary(
        self,
        active_bridge_sequence: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(active_bridge_sequence, dict):
            return None
        bridge_task_ids = self._bridge_sequence_task_ids(
            active_bridge_sequence.get("bridge_task_ids") or []
        )
        dispatched_task_ids = self._bridge_sequence_task_ids(
            active_bridge_sequence.get("dispatched_bridge_task_ids") or []
        )
        completed_task_ids = self._bridge_sequence_task_ids(
            active_bridge_sequence.get("completed_bridge_task_ids") or []
        )
        return {
            "bridge_sequence_id": str(
                active_bridge_sequence.get("bridge_sequence_id") or ""
            ).strip(),
            "proposal_fingerprint": str(
                active_bridge_sequence.get("proposal_fingerprint") or ""
            ).strip(),
            "state": str(active_bridge_sequence.get("state") or "").strip(),
            "bridge_task_ids": bridge_task_ids,
            "bridge_sequence_length": int(
                active_bridge_sequence.get("bridge_sequence_length") or len(bridge_task_ids)
            ),
            "compiled_bridge_task_count": len(bridge_task_ids),
            "dispatched_bridge_task_ids": dispatched_task_ids,
            "dispatched_bridge_task_count": len(dispatched_task_ids),
            "completed_bridge_task_ids": completed_task_ids,
            "completed_bridge_task_count": len(completed_task_ids),
            "execution_started_at_utc": str(
                active_bridge_sequence.get("execution_started_at_utc") or ""
            ).strip(),
            "last_dispatched_task_id": str(
                active_bridge_sequence.get("last_dispatched_task_id") or ""
            ).strip(),
            "last_completed_task_id": str(
                active_bridge_sequence.get("last_completed_task_id") or ""
            ).strip(),
            "last_completed_macro_name": str(
                active_bridge_sequence.get("last_completed_macro_name") or ""
            ).strip(),
            "source_mode": str(active_bridge_sequence.get("source_mode") or "").strip(),
            "validation_policy": str(
                active_bridge_sequence.get("validation_policy") or ""
            ).strip(),
            "source_archive_path": str(
                active_bridge_sequence.get("source_archive_path") or ""
            ).strip(),
            "source_archive_label": str(
                active_bridge_sequence.get("source_archive_label") or ""
            ).strip(),
            "source": str(active_bridge_sequence.get("source") or "").strip(),
            "used_llm_bridge": bool(active_bridge_sequence.get("used_llm_bridge", False)),
            "repair_task_id": str(
                active_bridge_sequence.get("repair_task_id") or ""
            ).strip(),
            "repair_target_task_id": str(
                active_bridge_sequence.get("repair_target_task_id") or ""
            ).strip(),
            "repair_operator": str(
                active_bridge_sequence.get("repair_operator") or ""
            ).strip(),
            "repair_intent": str(
                active_bridge_sequence.get("repair_intent") or ""
            ).strip(),
        }

    def _should_ignore_stale_recovery_macro_ack(
        self,
        *,
        task_node: dict[str, Any] | None,
        incoming_status: str,
    ) -> bool:
        if not isinstance(task_node, dict):
            return False
        if str(task_node.get("function_name") or "").strip() != "execute_recovery_macro":
            return False
        task_id = str(task_node.get("id") or "").strip()
        sequence_id = str(task_node.get("bridge_sequence_id") or "").strip()
        if not task_id or not sequence_id:
            return False
        normalized_incoming = str(incoming_status or "").strip().lower()
        if normalized_incoming in {"", "completed", "finished"}:
            return False
        if not (
            normalized_incoming.startswith("failed")
            or normalized_incoming in {"accepted", "running", "dispatched"}
        ):
            return False
        current_status = str(task_node.get("status") or "").strip().lower()
        if current_status not in {"completed", "finished"}:
            return False

        active_bridge_sequence = self._active_bridge_sequence()
        if isinstance(active_bridge_sequence, dict):
            active_state = self._bridge_sequence_state(active_bridge_sequence)
            active_sequence_id = str(
                active_bridge_sequence.get("bridge_sequence_id") or ""
            ).strip()
            if active_sequence_id and active_sequence_id == sequence_id and active_state in {
                "failed",
                "human_required",
            }:
                return True
            completed_ids = set(
                self._bridge_sequence_task_ids(
                    active_bridge_sequence.get("completed_bridge_task_ids") or []
                )
            )
            if task_id in completed_ids:
                return True
            active_sequence_id = str(
                active_bridge_sequence.get("bridge_sequence_id") or ""
            ).strip()
            if active_sequence_id and active_sequence_id != sequence_id:
                return True

        last_completed_sequence = self.runtime_recovery.get("last_completed_bridge_sequence")
        if isinstance(last_completed_sequence, dict):
            completed_ids = set(
                self._bridge_sequence_task_ids(
                    last_completed_sequence.get("completed_bridge_task_ids") or []
                )
            )
            if task_id in completed_ids:
                return True
            last_sequence_id = str(
                last_completed_sequence.get("bridge_sequence_id") or ""
            ).strip()
            if last_sequence_id and last_sequence_id == sequence_id:
                return True

        return True

    def _bridge_sequence_with_dispatched_task(
        self,
        active_bridge_sequence: dict[str, Any] | None,
        *,
        task_id: str,
    ) -> dict[str, Any] | None:
        if not isinstance(active_bridge_sequence, dict):
            return None
        next_sequence = deepcopy(active_bridge_sequence)
        dispatched_task_ids = self._bridge_sequence_task_ids(
            next_sequence.get("dispatched_bridge_task_ids") or []
        )
        if task_id and task_id not in dispatched_task_ids:
            dispatched_task_ids.append(task_id)
        next_sequence["dispatched_bridge_task_ids"] = dispatched_task_ids
        next_sequence["last_dispatched_task_id"] = str(task_id or "").strip()
        next_sequence["state"] = "executing"
        if not str(next_sequence.get("execution_started_at_utc") or "").strip():
            next_sequence["execution_started_at_utc"] = self._utc_now_iso()
        return next_sequence

    def _bridge_sequence_with_completed_task(
        self,
        active_bridge_sequence: dict[str, Any] | None,
        *,
        task_id: str,
        macro_name: str,
        terminal: bool = False,
    ) -> dict[str, Any] | None:
        if not isinstance(active_bridge_sequence, dict):
            return None
        next_sequence = deepcopy(active_bridge_sequence)
        dispatched_task_ids = self._bridge_sequence_task_ids(
            next_sequence.get("dispatched_bridge_task_ids") or []
        )
        if task_id and task_id not in dispatched_task_ids:
            dispatched_task_ids.append(task_id)
        next_sequence["dispatched_bridge_task_ids"] = dispatched_task_ids
        completed_task_ids = self._bridge_sequence_task_ids(
            next_sequence.get("completed_bridge_task_ids") or []
        )
        if task_id and task_id not in completed_task_ids:
            completed_task_ids.append(task_id)
        next_sequence["completed_bridge_task_ids"] = completed_task_ids
        next_sequence["last_completed_task_id"] = str(task_id or "").strip()
        next_sequence["last_completed_macro_name"] = str(macro_name or "").strip()
        next_sequence["last_task_id"] = str(task_id or "").strip()
        next_sequence["state"] = "completed" if terminal else "executing"
        if not str(next_sequence.get("execution_started_at_utc") or "").strip():
            next_sequence["execution_started_at_utc"] = self._utc_now_iso()
        return next_sequence

    @staticmethod
    def _derive_runtime_bridge_stage(recovery: dict[str, Any]) -> str:
        approval_state = str(recovery.get("bridge_approval_state", "none") or "none").strip().lower()
        if approval_state == "outline_pending":
            return "outline"
        if approval_state == "primitive_pending":
            return "primitive"
        if approval_state in {"pending", "approved"} or isinstance(recovery.get("bridge_proposal"), dict):
            return "final"
        return "none"

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
            "bridge_stage": "none",
            "bridge_mode": self._runtime_bridge_session_mode(),
            "validation_policy": self._runtime_bridge_session_validation_policy(),
            "selected_archive_path": self._runtime_bridge_session_archive_path(),
            "selected_archive_label": self._runtime_bridge_session_archive_label(),
            "artifact_directory": self._runtime_bridge_session_artifact_directory(),
            "active_bridge_sequence": None,
            "last_completed_bridge_sequence": None,
            "recovery_safety_scope_id": "",
            "recovery_safety_status": "none",
            "recovery_safety_logic_json": "",
            "recovery_safety_dir": "",
            "recovery_plan_dir": "",
            "recovery_safery_dir": "",
            "recovery_final_dir": "",
            "recovery_final_output_path": "",
            "recovery_enforced_task_ids": [],
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
        if status == "llm_bridge" and approval_state in {"pending", "outline_pending", "primitive_pending"}:
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
        bridge_stage: str | None = None,
        artifact_directory: str | None | object = _UNSET,
        active_bridge_sequence: dict[str, Any] | None | object = _UNSET,
        last_completed_bridge_sequence: dict[str, Any] | None | object = _UNSET,
        recovery_safety_scope_id: str | None | object = _UNSET,
        recovery_safety_status: str | None | object = _UNSET,
        recovery_safety_logic_json: str | None | object = _UNSET,
        recovery_safety_dir: str | None | object = _UNSET,
        recovery_plan_dir: str | None | object = _UNSET,
        recovery_safery_dir: str | None | object = _UNSET,
        recovery_final_dir: str | None | object = _UNSET,
        recovery_final_output_path: str | None | object = _UNSET,
        recovery_enforced_task_ids: list[str] | object = _UNSET,
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
        if bridge_stage is not None:
            current["bridge_stage"] = str(bridge_stage or "none").strip() or "none"
        if artifact_directory is not _UNSET:
            current["artifact_directory"] = str(artifact_directory or "").strip()
        if active_bridge_sequence is not _UNSET:
            current["active_bridge_sequence"] = (
                deepcopy(active_bridge_sequence)
                if isinstance(active_bridge_sequence, dict)
                else None
            )
        if last_completed_bridge_sequence is not _UNSET:
            current["last_completed_bridge_sequence"] = (
                deepcopy(last_completed_bridge_sequence)
                if isinstance(last_completed_bridge_sequence, dict)
                else None
            )
        if recovery_safety_scope_id is not _UNSET:
            current["recovery_safety_scope_id"] = str(recovery_safety_scope_id or "").strip()
        if recovery_safety_status is not _UNSET:
            current["recovery_safety_status"] = str(recovery_safety_status or "none").strip() or "none"
        if recovery_safety_logic_json is not _UNSET:
            current["recovery_safety_logic_json"] = str(recovery_safety_logic_json or "").strip()
        if recovery_safety_dir is not _UNSET:
            current["recovery_safety_dir"] = str(recovery_safety_dir or "").strip()
        if recovery_plan_dir is not _UNSET:
            current["recovery_plan_dir"] = str(recovery_plan_dir or "").strip()
        if recovery_safery_dir is not _UNSET:
            current["recovery_safery_dir"] = str(recovery_safery_dir or "").strip()
        if recovery_final_dir is not _UNSET:
            current["recovery_final_dir"] = str(recovery_final_dir or "").strip()
        if recovery_final_output_path is not _UNSET:
            current["recovery_final_output_path"] = str(recovery_final_output_path or "").strip()
        if recovery_enforced_task_ids is not _UNSET:
            current["recovery_enforced_task_ids"] = [
                str(task_id).strip()
                for task_id in (recovery_enforced_task_ids or [])
                if str(task_id).strip()
            ]
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

        current["bridge_mode"] = self._runtime_bridge_session_mode()
        current["validation_policy"] = self._runtime_bridge_session_validation_policy()
        current["selected_archive_path"] = self._runtime_bridge_session_archive_path()
        current["selected_archive_label"] = self._runtime_bridge_session_archive_label()
        if artifact_directory is _UNSET:
            current["artifact_directory"] = (
                self._runtime_bridge_session_artifact_directory()
                or str(dict(current.get("bridge_debug") or {}).get("artifact_directory") or "").strip()
            )
        if bridge_stage is None:
            current["bridge_stage"] = self._derive_runtime_bridge_stage(current)

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

    def _runtime_safety_fast_path_monitor(self) -> OnlineSafetyMonitor | None:
        logic_path_value = getattr(
            self,
            "safety_logic_path",
            Path("cais_spade_llm/safety/cca_safety_logic.json"),
        )
        precomputed_bundle = getattr(self, "precomputed_bundle", {})
        artifacts = (
            precomputed_bundle.get("artifacts", {})
            if isinstance(precomputed_bundle, dict)
            else {}
        )
        precomputed_logic = (
            artifacts.get("safety_logic_json")
            if isinstance(artifacts, dict)
            else None
        )
        if precomputed_logic:
            logic_path_value = precomputed_logic
        logic_path = Path(logic_path_value)
        safety_file = getattr(self, "safety_file", None)
        if not safety_file:
            return None
        try:
            safety_text = Path(safety_file).read_text(encoding="utf-8").strip()
            expected_hash = SafetyLogic.compute_safety_text_sha256(safety_text)
            stat = logic_path.stat()
            cache_key = (
                str(logic_path.resolve()),
                int(stat.st_mtime_ns),
                expected_hash,
            )
        except Exception:
            return None

        cache = getattr(self, "_runtime_safety_fast_path_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._runtime_safety_fast_path_cache = cache
        if cache.get("cache_key") == cache_key:
            monitor = cache.get("monitor")
            return monitor if isinstance(monitor, OnlineSafetyMonitor) else None

        def _store_monitor(monitor: OnlineSafetyMonitor | None) -> OnlineSafetyMonitor | None:
            cache["cache_key"] = cache_key
            cache["monitor"] = monitor
            return monitor

        try:
            payload = json.loads(logic_path.read_text(encoding="utf-8"))
            if str(payload.get("safety_text_sha256") or "").strip() != expected_hash:
                return _store_monitor(None)

            safety_rules = payload.get("rules")
            if not isinstance(safety_rules, list) or not safety_rules:
                return _store_monitor(None)

            dfa_map: dict[str, str] = {}
            for rule in safety_rules:
                rule_id = str((rule or {}).get("id") or "").strip()
                if not rule_id:
                    continue
                dot_path = logic_path.parent / f"{rule_id}_dfa.dot"
                if not dot_path.exists():
                    return _store_monitor(None)
                dfa_map[rule_id] = dot_path.read_text(encoding="utf-8")

            if not dfa_map:
                return _store_monitor(None)

            monitor = OnlineSafetyMonitor(
                dfa_map,
                safety_rules,
                tools_catalog=getattr(self, "tools_catalog", []),
            )
            return _store_monitor(monitor)
        except Exception:
            return _store_monitor(None)

    def _runtime_safety_ap_sets_for_task(
        self,
        task_node: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, list[str]] | None:
        monitor = self._runtime_safety_fast_path_monitor()
        if monitor is None:
            return None

        resource_jid = str(
            task_node.get("resource_jid")
            or params.get("resource_jid")
            or ""
        ).strip()
        function_name = str(
            task_node.get("function_name")
            or params.get("function_name")
            or ""
        ).strip()
        if not resource_jid or not function_name:
            return None

        candidate_aps = monitor._map_task_to_aps(resource_jid, function_name, params)
        predicted_state_aps = monitor._predict_state_aps(resource_jid, function_name, params)
        return {
            "candidate_aps": list(candidate_aps),
            "predicted_state_aps": list(predicted_state_aps),
        }

    def _runtime_safety_task_ap_empty(
        self,
        task_node: dict[str, Any],
        params: dict[str, Any],
    ) -> bool | None:
        ap_sets = self._runtime_safety_ap_sets_for_task(task_node, params)
        if ap_sets is None:
            return None
        return not ap_sets.get("candidate_aps") and not ap_sets.get("predicted_state_aps")

    def _dispatch_params_for_task_node(self, task_node: dict[str, Any]) -> dict[str, Any]:
        """Build task params for dispatch without overriding recovery primitive intent."""
        params = dict(task_node.get("params", {}))
        task_id = str(task_node.get("id") or "").strip()
        if task_id and "task_id" not in params:
            params["task_id"] = task_id
        part_name = params.get("part_name")
        function_name = str(task_node.get("function_name") or "").strip()
        if function_name == "execute_recovery_macro":
            params = self._enrich_observed_pose_recovery_params(params)
            bridge_outline_id = str(task_node.get("bridge_outline_id") or "").strip()
            if bridge_outline_id:
                params.setdefault("bridge_outline_id", bridge_outline_id)
                params.setdefault("outline_id", bridge_outline_id)
            llm_outline_id = str(task_node.get("llm_outline_id") or "").strip()
            if llm_outline_id:
                params.setdefault("llm_outline_id", llm_outline_id)
            event_name = str(
                params.get("event_name")
                or task_node.get("event_name")
                or ""
            ).strip()
            if event_name:
                params["event_name"] = event_name
            projected_outline_state = dict(
                params.get("projected_outline_state")
                or task_node.get("projected_outline_state")
                or {}
            )
            if projected_outline_state:
                params["projected_outline_state"] = deepcopy(projected_outline_state)
            if isinstance(task_node.get("part_name"), str) and str(task_node.get("part_name") or "").strip():
                params.setdefault("part_name", str(task_node.get("part_name") or "").strip())
        elif part_name and function_name != "execute_recovery_macro":
            geo = self._geometry_for_part(part_name)
            if geo:
                params["product_geometry"] = geo

        active_bridge_sequence = self._active_bridge_sequence()
        active_execution_policy = (
            active_bridge_sequence.get("execution_policy")
            if isinstance(active_bridge_sequence, dict)
            and isinstance(active_bridge_sequence.get("execution_policy"), dict)
            else {}
        )
        active_validation_policy = str(
            active_execution_policy.get("validation_policy")
            or (active_bridge_sequence or {}).get("validation_policy")
            or self.runtime_recovery.get("validation_policy")
            or self._runtime_bridge_session_validation_policy()
            or ""
        ).strip().lower()
        recovery_safety_scope_id = str(
            (active_bridge_sequence or {}).get("recovery_safety_scope_id")
            or self.runtime_recovery.get("recovery_safety_scope_id")
            or ""
        ).strip()
        bridge_dispatch_session = bool(
            active_bridge_sequence
            or self.runtime_recovery.get("active_bridge_sequence")
            or self.runtime_recovery.get("used_llm_bridge")
            or self.runtime_recovery.get("bridge_approval_state")
            or self.runtime_recovery.get("bridge_proposal")
        )
        sequence_task_id_parser = getattr(self, "_bridge_sequence_task_ids", None)

        def _sequence_task_ids(raw_ids: Any) -> list[str]:
            if callable(sequence_task_id_parser):
                return list(sequence_task_id_parser(raw_ids or []))
            if isinstance(raw_ids, str):
                return [raw_ids]
            return [str(item).strip() for item in (raw_ids or []) if str(item).strip()]

        recovery_enforced_task_ids: set[str] = set()
        for raw_ids in (
            (active_bridge_sequence or {}).get("recovery_enforced_task_ids"),
            self.runtime_recovery.get("recovery_enforced_task_ids"),
            (active_bridge_sequence or {}).get("bridge_task_ids"),
            (active_bridge_sequence or {}).get("dispatched_bridge_task_ids"),
        ):
            for item in _sequence_task_ids(raw_ids or []):
                token = str(item).strip()
                if token:
                    recovery_enforced_task_ids.add(token)

        has_recovery_fields = bool(
            str(task_node.get("bridge_sequence_id") or "").strip()
            or str(task_node.get("bridge_outline_id") or "").strip()
            or str(params.get("bridge_outline_id") or "").strip()
            or str(params.get("outline_id") or "").strip()
            or str(task_node.get("repair_operator") or "").strip()
        )
        recovery_safety_task = bool(
            task_id in recovery_enforced_task_ids
            or function_name == "execute_recovery_macro"
            or has_recovery_fields
        )
        assembly_board_task = (
            function_name in {"place_approach", "place_insert"}
            and str(params.get("destination_location") or "").strip() == "assembly_board-v1"
        )
        if active_validation_policy == "validated" and recovery_safety_task:
            if recovery_safety_scope_id:
                params["recovery_safety_scope_id"] = recovery_safety_scope_id
                params["start_safety_mode"] = "cca_check"
            else:
                message = (
                    "Recovery Safety Check dispatch blocked: validated recovery task "
                    f"{task_id or '<unknown>'} has no recovery_safety_scope_id; "
                    "refusing to dispatch without CCA runtime check."
                )
                self.logger.error("[Product] %s", message)
                self._set_runtime_recovery(
                    status="human_required",
                    resolution_class="human_required",
                    message=message,
                    attempts_used=self._runtime_repair_fail_streak,
                    attempts_max=self._runtime_repair_max_attempts,
                    bridge_approval_state="approved",
                    recovery_enforced_task_ids=sorted(recovery_enforced_task_ids),
                    violations=[],
                    append_history=True,
                    history_message=message,
                )
                raise RuntimeError(message)
        elif assembly_board_task and bool(getattr(self, "safety_text_has_requirements", False)):
            params["start_safety_mode"] = "cca_check"
        elif not recovery_safety_task:
            if active_validation_policy == "no_validation" and bridge_dispatch_session:
                return params
            if not bool(getattr(self, "safety_text_has_requirements", False)) or self._runtime_safety_task_ap_empty(task_node, params) is True:
                params.setdefault("start_safety_mode", "fast_path")
        return params

    def _runtime_recovery_blocks_execution(self) -> bool:
        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        return status not in {"", "idle", "resolved"}

    def get_runtime_recovery(self) -> dict[str, Any]:
        return deepcopy(self.runtime_recovery) if isinstance(self.runtime_recovery, dict) else self._empty_runtime_recovery()

    def _runtime_bridge_data_root(self) -> Path:
        return Path(DEFAULT_BRIDGE_DEBUG_DIR)

    def _runtime_recovery_safety_live_root(self) -> Path:
        return Path(
            "cais_spade_llm/agents/intelligent_product/replanner/llm_bridge/runtime_data/imported/worked"
        )

    def _allocate_runtime_recovery_safety_worked_dir(self) -> Path:
        current_dirs = [
            str(
                (self._runtime_recovery_context or {}).get("artifact_directory")
                or self.runtime_recovery.get("artifact_directory")
                or ""
            ).strip(),
            str(
                (self._runtime_recovery_context or {}).get("recovery_safety_dir")
                or self.runtime_recovery.get("recovery_safety_dir")
                or ""
            ).strip(),
            str(
                (self._runtime_recovery_context or {}).get("recovery_plan_dir")
                or self.runtime_recovery.get("recovery_plan_dir")
                or ""
            ).strip(),
            str(
                (self._runtime_recovery_context or {}).get("recovery_safery_dir")
                or self.runtime_recovery.get("recovery_safery_dir")
                or ""
            ).strip(),
        ]
        for raw_path in current_dirs:
            if not raw_path:
                continue
            candidate = Path(raw_path)
            if candidate.name.isdigit() and candidate.parent.name == "worked":
                return candidate
            parent = candidate.parent
            if parent.name.isdigit() and parent.parent.name == "worked":
                return parent
        root = self._runtime_recovery_safety_live_root()
        root.mkdir(parents=True, exist_ok=True)
        max_index = 0
        for child in root.iterdir():
            if child.is_dir() and child.name.isdigit():
                max_index = max(max_index, int(child.name))
        worked_dir = root / str(max_index + 1)
        worked_dir.mkdir(parents=True, exist_ok=True)
        return worked_dir

    def _recovery_safety_generation_dirs(self) -> dict[str, str]:
        worked_dir = self._allocate_runtime_recovery_safety_worked_dir()
        recovery_safety_dir = worked_dir / "recovery_safety"
        recovery_plan_dir = recovery_safety_dir
        recovery_safery_dir = recovery_safety_dir
        recovery_safety_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(self._runtime_recovery_context, dict):
            self._runtime_recovery_context["recovery_safety_dir"] = str(recovery_safety_dir)
            self._runtime_recovery_context["recovery_plan_dir"] = str(recovery_plan_dir)
            self._runtime_recovery_context["recovery_safery_dir"] = str(recovery_safery_dir)
        return {
            "recovery_safety_dir": str(recovery_safety_dir),
            "recovery_plan_dir": str(recovery_plan_dir),
            "recovery_safery_dir": str(recovery_safery_dir),
        }

    @staticmethod
    def _worked_dir_for_archived_bridge_artifact(
        artifact_path: str | Path,
    ) -> Path | None:
        candidate = Path(artifact_path)
        try:
            candidate = candidate.resolve()
        except Exception:
            pass
        search_roots = [candidate if candidate.is_dir() else candidate.parent]
        search_roots.extend(search_roots[0].parents)
        for root in search_roots:
            if root.name.isdigit() and root.parent.name == "worked":
                return root
            parent = root.parent
            if parent.name.isdigit() and parent.parent.name == "worked":
                return parent
        return None

    def _load_archived_recovery_safety_result(
        self,
        artifact_path: str | Path,
    ) -> dict[str, Any]:
        worked_dir = self._worked_dir_for_archived_bridge_artifact(artifact_path)
        if worked_dir is None:
            return {}
        recovery_safety_dir = worked_dir / "recovery_safety"
        result_path = recovery_safety_dir / "recovery_safety_generation_result.json"
        if not result_path.is_file():
            return {}
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            self.logger.warning(
                "[Product] Could not load archived recovery safety result: %s",
                result_path,
                exc_info=True,
            )
            return {}
        if not isinstance(payload, dict) or not bool(payload.get("ok", False)):
            return {}

        result = deepcopy(payload)
        if not str(result.get("recovery_safety_scope_id") or "").strip():
            result["recovery_safety_scope_id"] = f"recovery_scope_{uuid.uuid4().hex[:8]}"
        result["recovery_safety_status"] = "ready"
        result.setdefault("recovery_safety_dir", str(recovery_safety_dir))
        result.setdefault("recovery_plan_dir", str(recovery_safety_dir))
        result.setdefault("recovery_safery_dir", str(recovery_safety_dir))
        recovery_safety_logic_json = str(
            result.get("recovery_safety_logic_json") or ""
        ).strip()
        if not recovery_safety_logic_json:
            logic_path = recovery_safety_dir / "cca_safety_logic.json"
            if logic_path.is_file():
                result["recovery_safety_logic_json"] = str(logic_path.resolve())
        result["archived_recovery_safety_result_path"] = str(result_path)
        return result

    def _recovery_final_dir(self) -> Path:
        worked_dir = self._allocate_runtime_recovery_safety_worked_dir()
        recovery_final_dir = worked_dir / "recovery_final"
        recovery_final_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(self._runtime_recovery_context, dict):
            self._runtime_recovery_context["recovery_final_dir"] = str(recovery_final_dir)
        return recovery_final_dir

    @staticmethod
    def _primitive_program_ready_final_output_payload(
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, Any]:
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        for candidate in (
            bridge_debug.get("final_output"),
            dict(bridge_debug.get("multi_turn_session") or {}).get("final_output"),
            prepared_bridge_request.get("multi_turn_session_state", {}).get("final_output")
            if isinstance(prepared_bridge_request.get("multi_turn_session_state"), dict)
            else {},
        ):
            if not isinstance(candidate, dict) or not candidate:
                continue
            if str(candidate.get("final_output_stage") or "").strip() != "primitive_program_ready":
                continue
            if not bool(candidate.get("primitive_program_complete")):
                continue
            return deepcopy(candidate)
        return {}

    @staticmethod
    def _primitive_program_ready_final_output_source_path(
        prepared_bridge_request: dict[str, Any],
    ) -> str:
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        session_state = dict(
            bridge_debug.get("multi_turn_session")
            or prepared_bridge_request.get("multi_turn_session_state")
            or {}
        )
        turns = [
            dict(row)
            for row in (session_state.get("turns") or [])
            if isinstance(row, dict)
        ]
        for turn in reversed(turns):
            if str(turn.get("phase") or "").strip().lower() != "final_output":
                continue
            if str(turn.get("final_output_stage") or "").strip() != "primitive_program_ready":
                continue
            candidate = str(turn.get("response_artifact_path") or "").strip()
            if candidate:
                return candidate
        return ""

    @staticmethod
    def _primitive_program_ready_final_output_turn_index(
        prepared_bridge_request: dict[str, Any],
    ) -> int:
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        session_state = dict(
            bridge_debug.get("multi_turn_session")
            or prepared_bridge_request.get("multi_turn_session_state")
            or {}
        )
        turns = [
            dict(row)
            for row in (session_state.get("turns") or [])
            if isinstance(row, dict)
        ]
        for turn in reversed(turns):
            if str(turn.get("phase") or "").strip().lower() != "final_output":
                continue
            if str(turn.get("final_output_stage") or "").strip() != "primitive_program_ready":
                continue
            return int(turn.get("turn_index") or 0)
        return 0

    def _write_recovery_final_bundle(
        self,
        *,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, str]:
        if not isinstance(prepared_bridge_request, dict) or not prepared_bridge_request:
            return {}
        final_output_payload = self._primitive_program_ready_final_output_payload(
            prepared_bridge_request
        )
        if not final_output_payload:
            return {}
        recovery_safety_logic_json = str(
            self.runtime_recovery.get("recovery_safety_logic_json") or ""
        ).strip()
        if not recovery_safety_logic_json:
            return {}
        recovery_safety_logic_path = Path(recovery_safety_logic_json)
        if not recovery_safety_logic_path.exists():
            return {}

        recovery_final_dir = self._recovery_final_dir()
        source_final_output_path = self._primitive_program_ready_final_output_source_path(
            prepared_bridge_request
        )
        final_output_turn_index = self._primitive_program_ready_final_output_turn_index(
            prepared_bridge_request
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        if source_final_output_path:
            source_path = Path(source_final_output_path)
            target_final_output_path = recovery_final_dir / source_path.name
            if source_path.exists() and source_path.resolve() != target_final_output_path.resolve():
                shutil.copy2(source_path, target_final_output_path)
            elif source_path.exists():
                target_final_output_path = source_path
            else:
                target_final_output_path.write_text(
                    json.dumps(final_output_payload, indent=2, ensure_ascii=True),
                    encoding="utf-8",
                )
        else:
            artifact_name = (
                f"multi_turn_turn{max(final_output_turn_index, 1):02d}_"
                f"final_output_response_{timestamp}.txt"
            )
            target_final_output_path = recovery_final_dir / artifact_name
            target_final_output_path.write_text(
                json.dumps(final_output_payload, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )

        target_logic_path = recovery_final_dir / "cca_safety_logic.json"
        target_logic_tmp_path = recovery_final_dir / "cca_safety_logic.json.tmp"
        shutil.copy2(recovery_safety_logic_path, target_logic_tmp_path)
        target_logic_tmp_path.replace(target_logic_path)
        dfa_dot_files: list[str] = []
        for stale_dfa_path in sorted(recovery_final_dir.glob("*_dfa.dot")):
            stale_dfa_path.unlink(missing_ok=True)
        for source_dfa_path in sorted(recovery_safety_logic_path.parent.glob("*_dfa.dot")):
            target_dfa_path = recovery_final_dir / source_dfa_path.name
            shutil.copy2(source_dfa_path, target_dfa_path)
            dfa_dot_files.append(str(target_dfa_path.resolve()))

        bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
        bridge_debug["recovery_safety_logic_json"] = str(target_logic_path.resolve())
        bridge_debug["recovery_final_safety_logic_json"] = str(target_logic_path.resolve())
        bridge_debug["recovery_final_dir"] = str(recovery_final_dir)
        bridge_debug["recovery_final_output_path"] = str(target_final_output_path.resolve())
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        if isinstance(self._runtime_recovery_context, dict):
            self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(
                prepared_bridge_request
            )
            self._runtime_recovery_context["recovery_safety_logic_json"] = str(
                target_logic_path.resolve()
            )
            self._runtime_recovery_context["recovery_final_safety_logic_json"] = str(
                target_logic_path.resolve()
            )
            self._runtime_recovery_context["recovery_final_dir"] = str(recovery_final_dir)
            self._runtime_recovery_context["recovery_final_output_path"] = str(
                target_final_output_path.resolve()
            )

        self._set_runtime_recovery(
            recovery_safety_logic_json=str(target_logic_path.resolve()),
            recovery_final_dir=str(recovery_final_dir),
            recovery_final_output_path=str(target_final_output_path.resolve()),
        )
        self.logger.info(
            "[Product] Wrote recovery_final bundle: final_output=%s safety_logic=%s dfa_count=%d",
            target_final_output_path,
            target_logic_path,
            len(dfa_dot_files),
        )
        return {
            "recovery_safety_logic_json": str(target_logic_path.resolve()),
            "recovery_final_dir": str(recovery_final_dir),
            "recovery_final_output_path": str(target_final_output_path.resolve()),
            "recovery_final_safety_logic_json": str(target_logic_path.resolve()),
        }

    def _maybe_write_recovery_final_bundle(
        self,
        *,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, str]:
        recovery_safety_status = str(
            self.runtime_recovery.get("recovery_safety_status") or "none"
        ).strip().lower()
        if recovery_safety_status != "ready":
            return {}
        return self._write_recovery_final_bundle(
            prepared_bridge_request=prepared_bridge_request,
        )

    def _build_recovery_safety_generation_payload(
        self,
        *,
        prepared_bridge_request: dict[str, Any],
        recovery_safety_scope_id: str,
        request_id: str,
        recovery_safety_dir: str,
        recovery_plan_dir: str,
        recovery_safery_dir: str,
    ) -> dict[str, Any]:
        def _nominal_candidate_tasks(
            *,
            pending_nominal_tasks: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            pending_by_id = {
                str(row.get("id") or row.get("task_id") or "").strip(): deepcopy(row)
                for row in pending_nominal_tasks
                if str(row.get("id") or row.get("task_id") or "").strip()
            }
            requirement_task_index = dict(
                prepared_bridge_request.get("requirement_task_index") or {}
            )
            task_requirement_map = dict(
                prepared_bridge_request.get("task_requirement_map") or {}
            )
            bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
            pending_task_rows_by_id: dict[str, dict[str, Any]] = {}
            active_requirement_ids: set[str] = set()

            for resource_entry in bridge_resources.values():
                if not isinstance(resource_entry, dict):
                    continue
                for task in (resource_entry.get("pending_tasks") or []):
                    if not isinstance(task, dict):
                        continue
                    task_id = str(task.get("id") or "").strip()
                    if not task_id:
                        continue
                    pending_task_rows_by_id[task_id] = deepcopy(task)
                    requirement_id = str(
                        task.get("requirement_id")
                        or task_requirement_map.get(task_id)
                        or ""
                    ).strip()
                    if requirement_id:
                        active_requirement_ids.add(requirement_id)

            for task_id in pending_by_id:
                requirement_id = str(task_requirement_map.get(task_id) or "").strip()
                if requirement_id:
                    active_requirement_ids.add(requirement_id)

            candidate_rows: list[dict[str, Any]] = []
            seen_task_ids: set[str] = set()
            for requirement_id in sorted(active_requirement_ids):
                for raw_task in (requirement_task_index.get(requirement_id) or []):
                    if not isinstance(raw_task, dict):
                        continue
                    task_id = str(raw_task.get("task_id") or raw_task.get("id") or "").strip()
                    if not task_id or task_id in seen_task_ids:
                        continue
                    seen_task_ids.add(task_id)
                    enriched = dict(pending_task_rows_by_id.get(task_id) or {})
                    pending_row = dict(pending_by_id.get(task_id) or {})
                    params = dict(
                        enriched.get("params")
                        or raw_task.get("params")
                        or {}
                    )
                    product_jid = str(params.get("product_jid") or "").strip()
                    destination_location = str(
                        params.get("destination_location") or ""
                    ).strip()
                    candidate_rows.append(
                        {
                            "id": task_id,
                            "function": str(
                                raw_task.get("function_name")
                                or enriched.get("function_name")
                                or pending_row.get("function")
                                or ""
                            ).strip(),
                            "resource": str(
                                raw_task.get("resource_jid")
                                or enriched.get("resource_jid")
                                or pending_row.get("resource")
                                or ""
                            ).strip(),
                            "part": str(
                                raw_task.get("part_name")
                                or params.get("part_name")
                                or pending_row.get("part")
                                or ""
                            ).strip(),
                            "status": str(
                                raw_task.get("status")
                                or enriched.get("status")
                                or pending_row.get("status")
                                or ""
                            ).strip(),
                            "in_state": str(
                                raw_task.get("in_state")
                                or enriched.get("in_state")
                                or pending_row.get("in_state")
                                or ""
                            ).strip(),
                            "out_state": str(
                                raw_task.get("out_state")
                                or enriched.get("out_state")
                                or pending_row.get("out_state")
                                or ""
                            ).strip(),
                            "requirement_id": requirement_id,
                            "sequence_index": int(raw_task.get("sequence_index") or 0),
                            "product_jid": product_jid,
                            "destination_location": destination_location,
                            "blocked_by_condition_ids": [
                                str(token).strip()
                                for token in (pending_row.get("blocked_by_condition_ids") or [])
                                if str(token).strip()
                            ],
                            "expected_start_state": deepcopy(
                                pending_row.get("expected_start_state") or {}
                            ),
                            "expected_end_state": deepcopy(
                                pending_row.get("expected_end_state") or {}
                            ),
                            "projected_outline_state": deepcopy(
                                pending_row.get("projected_outline_state") or {}
                            ),
                        }
                    )
            return candidate_rows

        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        session_candidates = [
            dict(bridge_debug.get("multi_turn_session") or {}),
            dict(prepared_bridge_request.get("multi_turn_session_state") or {}),
            dict(prepared_bridge_request.get("multi_turn_session_seed") or {}),
        ]
        session_state = next(
            (candidate for candidate in session_candidates if candidate),
            {},
        )
        accepted_outline_prefix: list[dict[str, Any]] = []
        projected_outline_state: dict[str, Any] = {}
        for candidate in session_candidates:
            if not candidate:
                continue
            for rows in (
                candidate.get("accepted_outline_prefix"),
                candidate.get("transition_trace"),
                dict(candidate.get("final_output") or {}).get("transition_trace"),
            ):
                accepted_outline_prefix = [
                    deepcopy(row)
                    for row in (rows or [])
                    if isinstance(row, dict)
                ]
                if accepted_outline_prefix:
                    break
            if accepted_outline_prefix:
                projected_outline_state = deepcopy(
                    accepted_outline_prefix[-1].get("projected_outline_state") or {}
                )
                break
        if not accepted_outline_prefix:
            for candidate in session_candidates:
                if not candidate:
                    continue
                projected_outline_state = deepcopy(
                    candidate.get("projected_outline_state") or {}
                )
                if projected_outline_state:
                    break
        if not projected_outline_state and accepted_outline_prefix:
            projected_outline_state = deepcopy(
                accepted_outline_prefix[-1].get("projected_outline_state") or {}
            )
        if not projected_outline_state:
            projected_outline_state = deepcopy(session_state.get("projected_outline_state") or {})
        if not projected_outline_state:
            projected_outline_state = deepcopy(
                dict(session_state.get("final_output") or {}).get("projected_outline_state") or {}
            )
        llm_input = dict(prepared_bridge_request.get("llm_input") or {})
        modeled_continuation_gap = dict(llm_input.get("modeled_continuation_gap") or {})
        pending_nominal_tasks = [
            deepcopy(row)
            for row in (modeled_continuation_gap.get("pending_nominal_tasks") or [])
            if isinstance(row, dict)
        ]
        pending_nominal_task_ids = [
            str(task_id).strip()
            for task_id in (
                modeled_continuation_gap.get("pending_nominal_task_ids")
                or (
                    dict(prepared_bridge_request.get("context_summary") or {}).get(
                        "modeled_continuation_gap"
                    )
                    or {}
                ).get("pending_nominal_task_ids")
                or [row.get("task_id") or row.get("id") for row in pending_nominal_tasks]
            )
            if str(task_id).strip()
        ]
        nominal_candidate_tasks = _nominal_candidate_tasks(
            pending_nominal_tasks=pending_nominal_tasks,
        )
        return {
            "request_id": request_id,
            "product_jid": str(self.jid),
            "recovery_safety_scope_id": str(recovery_safety_scope_id or "").strip(),
            "accepted_outline_prefix": accepted_outline_prefix,
            "projected_outline_state": projected_outline_state,
            "pending_nominal_tasks": pending_nominal_tasks,
            "pending_nominal_task_ids": pending_nominal_task_ids,
            "nominal_candidate_tasks": nominal_candidate_tasks,
            "nominal_candidate_task_ids": [
                str(row.get("id") or "").strip()
                for row in nominal_candidate_tasks
                if str(row.get("id") or "").strip()
            ],
            "loaded_safety_rules": [
                deepcopy(rule)
                for rule in (prepared_bridge_request.get("loaded_safety_rules") or [])
                if isinstance(rule, dict)
            ],
            "bridge_safety_context": deepcopy(
                prepared_bridge_request.get("bridge_safety_context") or {}
            ),
            "tools_catalog": deepcopy(getattr(self, "tools_catalog", []) or []),
            "recovery_safety_dir": str(recovery_safety_dir or "").strip(),
            "recovery_plan_dir": str(recovery_plan_dir or "").strip(),
            "recovery_safery_dir": str(recovery_safery_dir or "").strip(),
        }

    def _dispatch_recovery_safety_generation_request(
        self,
        *,
        prepared_bridge_request: dict[str, Any],
        recovery_safety_scope_id: str,
    ) -> dict[str, Any]:
        request_id = f"recovery_safety_generate_{uuid.uuid4().hex}"
        dirs = self._recovery_safety_generation_dirs()
        payload = self._build_recovery_safety_generation_payload(
            prepared_bridge_request=prepared_bridge_request,
            recovery_safety_scope_id=recovery_safety_scope_id,
            request_id=request_id,
            recovery_safety_dir=str(dirs.get("recovery_safety_dir") or "").strip(),
            recovery_plan_dir=str(dirs.get("recovery_plan_dir") or "").strip(),
            recovery_safery_dir=str(dirs.get("recovery_safery_dir") or "").strip(),
        )
        if not isinstance(getattr(self, "_runtime_recovery_context", None), dict):
            self._runtime_recovery_context = {}
        self._runtime_recovery_context["active_recovery_safety_request_id"] = request_id
        self._runtime_recovery_context["recovery_safety_scope_id"] = recovery_safety_scope_id
        self._runtime_recovery_context["recovery_safety_request_payload"] = deepcopy(payload)

        msg = Message(to=self.cca_jid)
        msg.set_metadata("type", "recovery_safety_generate")
        msg.body = json.dumps(payload)

        def _dispatch() -> None:
            self._dispatch_agent_message_sync(
                msg,
                trace_category="ProductAgent/_dispatch_recovery_safety_generation_request",
            )

        self._run_callable_on_agent_loop_sync(
            _dispatch,
            timeout_sec=10.0,
            operation_name="recovery safety generation dispatch",
        )
        self.logger.info(
            "[Product] Sent recovery_safety_generate scope=%s request_id=%s.",
            recovery_safety_scope_id,
            request_id,
        )
        return payload

    async def _handle_recovery_safety_generated_result(
        self,
        payload: dict[str, Any],
    ) -> bool:
        incoming_request_id = str(payload.get("request_id") or "").strip()
        expected_request_id = ""
        if isinstance(getattr(self, "_runtime_recovery_context", None), dict):
            expected_request_id = str(
                self._runtime_recovery_context.get("active_recovery_safety_request_id") or ""
            ).strip()
        if incoming_request_id and expected_request_id and incoming_request_id != expected_request_id:
            self.logger.info(
                "[Product] Ignoring stale recovery_safety_generated request_id=%s; active request_id=%s.",
                incoming_request_id,
                expected_request_id,
            )
            return True
        if incoming_request_id and not expected_request_id:
            self.logger.info(
                "[Product] Ignoring unexpected recovery_safety_generated request_id=%s because no active recovery safety generation is tracked.",
                incoming_request_id,
            )
            return True
        if isinstance(getattr(self, "_runtime_recovery_context", None), dict):
            self._runtime_recovery_context.pop("active_recovery_safety_request_id", None)
            self._runtime_recovery_context["recovery_safety_result"] = deepcopy(payload)

        ok = bool(payload.get("ok", False))
        recovery_safety_scope_id = str(payload.get("recovery_safety_scope_id") or "").strip()
        recovery_plan_dir = str(payload.get("recovery_plan_dir") or "").strip()
        recovery_safery_dir = str(payload.get("recovery_safery_dir") or "").strip()
        recovery_safety_dir = str(
            payload.get("recovery_safety_dir")
            or recovery_safery_dir
            or recovery_plan_dir
            or ""
        ).strip()
        recovery_safety_logic_json = str(payload.get("recovery_safety_logic_json") or "").strip()
        rule_ids = [
            str(rule_id).strip()
            for rule_id in (payload.get("rule_ids") or [])
            if str(rule_id).strip()
        ]
        if ok:
            message = (
                f"Recovery safety bundle is ready for scope {recovery_safety_scope_id} "
                f"with {len(rule_ids)} rule(s)."
            )
            self._set_runtime_recovery(
                recovery_safety_scope_id=recovery_safety_scope_id,
                recovery_safety_status="ready",
                recovery_safety_logic_json=recovery_safety_logic_json,
                recovery_safety_dir=recovery_safety_dir,
                recovery_plan_dir=recovery_plan_dir,
                recovery_safery_dir=recovery_safery_dir,
                append_history=True,
                history_message=message,
            )
            prepared_bridge_request = deepcopy(
                self._runtime_recovery_context.get("prepared_bridge_request") or {}
            )
            if prepared_bridge_request:
                self._maybe_write_recovery_final_bundle(
                    prepared_bridge_request=prepared_bridge_request,
                )
            self.logger.info("[Product] %s", message)
            if (
                self._runtime_bridge_session_mode() == "pre_ran"
                and str(self.runtime_recovery.get("status") or "").strip().lower()
                == "llm_bridge"
                and str(
                    self.runtime_recovery.get("bridge_approval_state") or ""
                ).strip().lower()
                == "pending"
                and str(self.runtime_recovery.get("bridge_stage") or "").strip().lower()
                == "final"
                and isinstance(self.runtime_recovery.get("bridge_proposal"), dict)
            ):
                self.logger.info(
                    "[Product] Auto-approving pre-ran archived bridge after recovery safety ready: scope=%s",
                    recovery_safety_scope_id or "<missing>",
                )
                await asyncio.to_thread(self.approve_runtime_bridge_proposal_sync)
                return True
        else:
            failure_reason = str(payload.get("failure_reason") or "unknown failure").strip()
            message = (
                f"Recovery safety generation failed for scope {recovery_safety_scope_id}: "
                f"{failure_reason}."
            )
            self._set_runtime_recovery(
                recovery_safety_scope_id=recovery_safety_scope_id,
                recovery_safety_status="failed",
                recovery_safety_logic_json="",
                recovery_safety_dir=recovery_safety_dir,
                recovery_plan_dir=recovery_plan_dir,
                recovery_safery_dir=recovery_safery_dir,
                append_history=True,
                history_message=message,
            )
            self.logger.warning("[Product] %s", message)
        await asyncio.to_thread(self._persist_product_state)
        return True

    def _resolve_runtime_bridge_archive_path(self, artifact_path: str | None = None) -> Path:
        candidate = str(artifact_path or self._runtime_bridge_session_archive_path() or "").strip()
        if not candidate:
            raise ValueError("no archived bridge run is selected")
        resolved = Path(candidate).expanduser()
        try:
            resolved = resolved.resolve()
        except Exception:
            pass
        root = self._runtime_bridge_data_root()
        try:
            root_resolved = root.resolve()
        except Exception:
            root_resolved = root
        if resolved.parent.name != "recovery_final":
            recovery_final_candidate = resolved.parent / "recovery_final" / resolved.name
            if recovery_final_candidate.exists() and recovery_final_candidate.is_file():
                try:
                    resolved = recovery_final_candidate.resolve()
                except Exception:
                    resolved = recovery_final_candidate
        try:
            resolved.relative_to(root_resolved)
        except Exception as exc:
            raise ValueError(
                f"archived bridge run must be inside {root_resolved}"
            ) from exc
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"archived bridge run does not exist: {resolved}")
        return resolved

    @staticmethod
    def _bridge_feedback_history_with(
        items: list[str] | None,
        feedback: str,
    ) -> list[str]:
        history = [str(item).strip() for item in (items or []) if str(item).strip()]
        feedback_text = str(feedback or "").strip()
        if feedback_text:
            history.append(feedback_text)
        return history

    def _append_bridge_feedback(self, feedback: str) -> list[str]:
        feedback_text = str(feedback or "").strip()
        if not feedback_text:
            raise ValueError("bridge feedback is empty")
        feedback_history = self._bridge_feedback_history_with(
            self.runtime_recovery.get("bridge_feedback_history") or [],
            feedback_text,
        )
        if self._runtime_recovery_context:
            self._runtime_recovery_context["bridge_feedback_history"] = list(feedback_history)
            prepared_bridge_request = deepcopy(
                self._runtime_recovery_context.get("prepared_bridge_request") or {}
            )
            if prepared_bridge_request:
                bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
                operator_feedback_history = self._bridge_feedback_history_with(
                    bridge_session.get("operator_feedback_history") or [],
                    feedback_text,
                )
                bridge_session["operator_feedback_history"] = operator_feedback_history[-12:]
                prepared_bridge_request["bridge_session"] = bridge_session
                prepared_bridge_request["bridge_feedback"] = feedback_text
                self._runtime_recovery_context["prepared_bridge_request"] = prepared_bridge_request
        return feedback_history

    @staticmethod
    def _bridge_session_state(prepared_bridge_request: dict[str, Any]) -> dict[str, Any]:
        session_state = deepcopy(prepared_bridge_request.get("multi_turn_session_state") or {})
        if session_state:
            return session_state
        return deepcopy(prepared_bridge_request.get("multi_turn_session_seed") or {})

    @staticmethod
    def _reset_session_for_outline_retry(session_state: dict[str, Any]) -> dict[str, Any]:
        updated = deepcopy(session_state or {})
        updated["current_phase"] = "outline"
        updated["status"] = "pending"
        updated["accepted_outline_prefix"] = []
        updated["accepted_transition_prefix"] = []
        updated["des_event_sequence"] = []
        updated["transition_trace"] = []
        updated["outline_lookahead"] = []
        updated["outline_stagnation_count"] = 0
        updated["outline_progress_signature"] = ""
        updated["pruned_actions"] = []
        updated["outline_validation_findings"] = []
        updated["transition_validation"] = {}
        updated["unresolved_target_predicates"] = []
        updated["candidate_rejection_feedback"] = []
        updated["candidate_prune_history"] = {}
        updated["primitive_generation_cursor"] = 0
        updated["accepted_primitive_program"] = []
        updated["primitive_rejection_feedback"] = []
        updated["primitive_served_context"] = {}
        updated["primitive_context_errors"] = []
        updated["primitive_input_diagnostics"] = []
        updated["primitive_event_guard"] = {}
        updated["primitive_escalation_diagnostics"] = []
        updated.pop("proposal", None)
        updated.pop("final_output", None)
        updated.pop("final_output_adapter", None)
        return updated

    @staticmethod
    def _reset_session_for_primitive_retry(session_state: dict[str, Any]) -> dict[str, Any]:
        updated = deepcopy(session_state or {})
        updated["current_phase"] = "primitive_generation"
        updated["status"] = "pending"
        updated["primitive_generation_cursor"] = 0
        updated["accepted_primitive_program"] = []
        updated["primitive_rejection_feedback"] = []
        updated["primitive_served_context"] = {}
        updated["primitive_context_errors"] = []
        updated["primitive_input_diagnostics"] = []
        updated["primitive_event_guard"] = {}
        updated["primitive_escalation_diagnostics"] = []
        updated.pop("proposal", None)
        updated.pop("final_output", None)
        updated.pop("final_output_adapter", None)
        return updated

    def _prepare_runtime_bridge_session_state(
        self,
        *,
        stage: str,
    ) -> dict[str, Any]:
        prepared_bridge_request = deepcopy(
            self._runtime_recovery_context.get("prepared_bridge_request") or {}
        )
        if not prepared_bridge_request:
            raise RuntimeError("no prepared bridge request is available")
        session_state = self._bridge_session_state(prepared_bridge_request)
        if not session_state:
            raise RuntimeError("no multi-turn session state is available")
        if stage == "outline":
            session_state = self._reset_session_for_outline_retry(session_state)
        elif stage == "primitive":
            session_state = self._reset_session_for_primitive_retry(session_state)
        else:
            raise ValueError(f"unsupported bridge stage reset: {stage}")
        prepared_bridge_request["multi_turn_session_state"] = deepcopy(session_state)
        bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
        bridge_debug.pop("bridge_proposal", None)
        bridge_debug.pop("final_output", None)
        bridge_debug.pop("final_output_adapter", None)
        bridge_debug["multi_turn_session"] = deepcopy(session_state)
        bridge_debug["status"] = "pending"
        prepared_bridge_request["bridge_debug"] = bridge_debug
        self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(prepared_bridge_request)
        return prepared_bridge_request

    @staticmethod
    def _bridge_proposal_from_debug(
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, Any] | None:
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        adapter = dict(bridge_debug.get("final_output_adapter") or {})
        proposal = adapter.get("bridge_proposal")
        if isinstance(proposal, dict):
            return deepcopy(proposal)
        proposal = bridge_debug.get("bridge_proposal")
        if isinstance(proposal, dict):
            return deepcopy(proposal)
        final_output = bridge_debug.get("final_output")
        if isinstance(final_output, dict):
            from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn import (
                build_multi_turn_bridge_proposal,
            )

            adapter = build_multi_turn_bridge_proposal(
                final_output_payload=deepcopy(final_output),
                prepared_bridge_request=prepared_bridge_request,
            )
            proposal = adapter.get("bridge_proposal")
            if isinstance(proposal, dict):
                bridge_debug["final_output_adapter"] = deepcopy(adapter)
                bridge_debug["bridge_proposal"] = deepcopy(proposal)
                prepared_bridge_request["bridge_debug"] = bridge_debug
                return deepcopy(proposal)
        session_state = dict(prepared_bridge_request.get("multi_turn_session_state") or {})
        proposal = session_state.get("proposal")
        if isinstance(proposal, dict):
            return deepcopy(proposal)
        return None

    def _load_runtime_bridge_archive_bundle(
        self,
        *,
        prepared_bridge_request: dict[str, Any],
        artifact_path: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], Path]:
        resolved_path = self._resolve_runtime_bridge_archive_path(artifact_path)
        try:
            final_output_payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"failed to parse archived bridge run: {exc}"
            ) from exc
        if not isinstance(final_output_payload, dict):
            raise RuntimeError("archived bridge run did not contain a JSON object")
        engine = str(final_output_payload.get("engine") or "").strip().lower()
        if engine and not engine.startswith("multi_turn"):
            raise RuntimeError("archived bridge run engine must be multi_turn-compatible")
        if str(final_output_payload.get("final_output_stage") or "").strip() != "primitive_program_ready":
            raise RuntimeError("archived bridge run is not a primitive_program_ready artifact")
        if not bool(final_output_payload.get("primitive_program_complete")):
            raise RuntimeError("archived bridge run does not contain a complete primitive program")

        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn import (
            build_multi_turn_bridge_proposal,
        )

        adapter_result = build_multi_turn_bridge_proposal(
            final_output_payload=deepcopy(final_output_payload),
            prepared_bridge_request=prepared_bridge_request,
        )
        proposal = adapter_result.get("bridge_proposal")
        if not isinstance(proposal, dict):
            raise RuntimeError(
                str(adapter_result.get("reason") or "archived bridge run proposal build failed").strip()
                or "archived bridge run proposal build failed"
            )
        return deepcopy(final_output_payload), deepcopy(adapter_result), resolved_path

    async def _run_multi_turn_bridge_until(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        stop_after: str,
    ) -> dict[str, Any] | None:
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
            execute_multi_turn_bridge as _resume_multi_turn_bridge,
        )

        target = str(stop_after or "final").strip().lower()
        self._ensure_live_bridge_per_turn_debug_dir(prepared_bridge_request)
        prepared_bridge_request["_stop_after_multi_turn_phase"] = target if target in {"outline", "primitive"} else ""
        session_state = dict(prepared_bridge_request.get("multi_turn_session_state") or {})
        if session_state:
            proposal = await _resume_multi_turn_bridge(
                self.process_planner,
                prepared_bridge_request,
                session_state=session_state,
            )
        else:
            proposal = await self.process_planner.execute_prepared_bridge_request(
                prepared_bridge_request
            )

        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        max_resume = max(
            10,
            int(
                bridge_session.get("max_turns")
                or dict(prepared_bridge_request.get("multi_turn_session_seed") or {}).get("max_turns")
                or 0
            ),
        )
        for _resume_idx in range(max_resume):
            session_state = dict(prepared_bridge_request.get("multi_turn_session_state") or {})
            pause_status = str(session_state.get("status") or "").strip().lower()
            if target == "outline":
                if pause_status in {"paused_after_outline_turn", "ready_for_primitive_generation"}:
                    break
                resume_needed = False
            elif target == "primitive":
                resume_needed = pause_status in {
                    "paused_after_outline_turn",
                    "ready_for_primitive_generation",
                    "paused_after_primitive_turn",
                }
                if not resume_needed:
                    break
            else:
                resume_needed = pause_status in {
                    "paused_after_outline_turn",
                    "ready_for_primitive_generation",
                    "paused_after_primitive_turn",
                }
                if not resume_needed:
                    break
            proposal = await _resume_multi_turn_bridge(
                self.process_planner,
                prepared_bridge_request,
                session_state=session_state,
            )
        prepared_bridge_request["_stop_after_multi_turn_phase"] = ""
        return proposal

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
            import json
            from datetime import datetime, timezone

            payload = {
                "part_tracker": self.part_tracker,
                "execution_timeline": self.execution_timeline,
                "runtime_repair_state": self.runtime_repair_state,
                "runtime_recovery": self.runtime_recovery,
                "plan_safety_alert": self.plan_safety_alert,
                "product_order_runtime": deepcopy(
                    getattr(getattr(self, "process_planner", None), "last_product_order_artifact", {}) or {}
                ),
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

    def _ensure_live_bridge_per_turn_debug_dir(
        self,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(prepared_bridge_request, dict):
            return {}
        bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
        artifact_directory = str(
            bridge_debug.get("artifact_directory")
            or getattr(self, "_runtime_recovery_context", {}).get("artifact_directory")
            or self.runtime_recovery.get("artifact_directory")
            or ""
        ).strip()
        if not artifact_directory:
            artifact_directory = str(self._allocate_runtime_recovery_safety_worked_dir())
        candidate_dir = Path(artifact_directory)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        bridge_debug["artifact_directory"] = artifact_directory
        bridge_debug["per_turn_debug_dir"] = artifact_directory
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        if self._runtime_recovery_context is not None:
            self._runtime_recovery_context["artifact_directory"] = artifact_directory
        return bridge_debug

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
        bridge_debug = self._ensure_live_bridge_per_turn_debug_dir(prepared_bridge_request)
        artifact_directory = str(bridge_debug.get("artifact_directory") or "").strip()
        artifact_paths = write_bridge_artifacts(
            artifact_payload,
            phase_label=phase,
            debug_dir=artifact_directory or self._bridge_debug_directory(),
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
            artifact_directory=artifact_directory,
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
        self, function_name: str, task_node: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Look up the part_transition map, preferring node-level for bridge macros."""
        # Prefer node-level part_transition (set by bridge macro proposals).
        if task_node and task_node.get("part_transition"):
            return dict(task_node["part_transition"])
        type(self._agent)._load_shared_tools_catalogue()
        row = LlmAgent._TOOLS_BY_FUNC.get(function_name, {})
        return row.get("part_transition", {})

    def _tracked_part_name_for_task(
        self,
        task_node: dict[str, Any] | None,
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
        params: dict[str, Any],
        resource_jid: str,
        task_id: str,
        task_node: dict[str, Any] | None = None,
        observations: dict[str, Any] | None = None,
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
        entry: dict[str, Any] = {"state": transition["state"]}

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

    def _reactivate_blocked_tasks(
        self,
        *,
        candidate_task_ids: set[str] | None = None,
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

        pending_task_retry_ready_ids = getattr(self, "_pending_task_retry_ready_ids", None)
        if not isinstance(pending_task_retry_ready_ids, set):
            pending_task_retry_ready_ids = set()
            self._pending_task_retry_ready_ids = pending_task_retry_ready_ids

        reactivated_task_ids: set[str] = set()
        retain_task_retry_ready_ids: set[str] = set()
        for node in self.process_planner.nodes:
            if node.get("type") != "task":
                continue
            node_id = str(node.get("id") or "")
            if node_id not in candidate_task_ids:
                continue
            node_status = str(node.get("status") or "").strip().lower()
            if node_status in {"dispatched", "accepted"}:
                retain_task_retry_ready_ids.add(node_id)
                continue
            if node_status != "blocked":
                continue

            node["status"] = "pending"
            reactivated_task_ids.add(node_id)

        stale_task_retry_ready_ids = candidate_task_ids - retain_task_retry_ready_ids
        pending_task_retry_ready_ids.difference_update(stale_task_retry_ready_ids)
        pending_task_retry_ready_ids.update(retain_task_retry_ready_ids)

        if not reactivated_task_ids:
            return 0

        for task_id in reactivated_task_ids:
            pending_task_retry_ready_ids.discard(task_id)

        now_iso = datetime.now(timezone.utc).isoformat()
        for task_id in reactivated_task_ids:
            self.task_states[task_id] = "pending"
            self.execution_timeline.append({
                "timestamp": now_iso,
                "task_id": task_id,
                "status": "requeued",
                "resource_jid": str(self.cca_jid),
            })

        return len(reactivated_task_ids)

    def _reactivate_restored_repair_target_from_context(self) -> str:
        repair_target_task_id = str(
            self._runtime_recovery_context.get("repair_target_task_id") or ""
        ).strip()
        repair_task_id = str(
            self._runtime_recovery_context.get("repair_task_id") or ""
        ).strip()
        if not repair_target_task_id:
            return ""
        node = self.process_planner._find_node(repair_target_task_id)
        if not isinstance(node, dict):
            return ""
        current_status = str(node.get("status") or "").strip().lower()
        if current_status != "blocked":
            return ""
        node["status"] = "pending"
        self.task_states[repair_target_task_id] = "pending"
        self.logger.info(
            "[Product] Runtime DES repair %s reactivated restored target task %s after validation success.",
            repair_task_id or "<unknown>",
            repair_target_task_id,
        )
        return repair_target_task_id

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

    def _reconstruct_active_bridge_sequence_for_validation(self) -> dict[str, Any] | None:
        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        approval_state = str(
            self.runtime_recovery.get("bridge_approval_state", "none") or "none"
        ).strip().lower()
        if status != "validating" or approval_state != "approved":
            return None
        if isinstance(self.runtime_recovery.get("last_completed_bridge_sequence"), dict):
            return None

        bridge_debug = dict(self.runtime_recovery.get("bridge_debug") or {})
        approval = dict(bridge_debug.get("approval") or {})
        compiled_bridge_task_ids = self._bridge_sequence_task_ids(
            approval.get("compiled_bridge_task_ids") or []
        )
        if not compiled_bridge_task_ids:
            return None

        ordered_nodes: list[dict[str, Any]] = []
        for task_id in compiled_bridge_task_ids:
            node = self.process_planner._find_node(task_id)
            if isinstance(node, dict):
                ordered_nodes.append(node)
        if not ordered_nodes:
            return None

        bridge_sequence_id = str(ordered_nodes[0].get("bridge_sequence_id") or "").strip()
        if not bridge_sequence_id:
            return None

        sequence_nodes = self.process_planner._bridge_sequence_nodes(bridge_sequence_id)
        if sequence_nodes:
            ordered_nodes = [
                node
                for node in sequence_nodes
                if str(node.get("id") or "").strip() in set(compiled_bridge_task_ids)
            ]
            if not ordered_nodes:
                ordered_nodes = sequence_nodes

        prepared_bridge_request = dict(
            self._runtime_recovery_context.get("prepared_bridge_request") or {}
        )
        modeled_gap = dict(
            dict(prepared_bridge_request.get("context_summary") or {}).get(
                "modeled_continuation_gap"
            )
            or {}
        )
        archive_replay = dict(bridge_debug.get("archive_replay") or {})
        execution_policy = dict(bridge_debug.get("execution_policy") or {})
        if not bool(execution_policy.get("verification_only")):
            execution_policy.setdefault("complete_full_tail", True)

        bridge_task_ids = [
            str(node.get("id") or "").strip()
            for node in ordered_nodes
            if str(node.get("id") or "").strip()
        ]
        dispatched_bridge_task_ids: list[str] = []
        completed_bridge_task_ids: list[str] = []
        for node in ordered_nodes:
            task_id = str(node.get("id") or "").strip()
            task_status = str(node.get("status") or "").strip().lower()
            if task_status in {
                "dispatched",
                "accepted",
                "running",
                "completed",
                "failed",
                "blocked",
            }:
                dispatched_bridge_task_ids.append(task_id)
            if task_status == "completed":
                completed_bridge_task_ids.append(task_id)

        reconstructed_sequence: dict[str, Any] = {
            "bridge_sequence_id": bridge_sequence_id,
            "bridge_task_ids": bridge_task_ids,
            "bridge_sequence_length": len(bridge_task_ids),
            "trigger": str(self._runtime_recovery_context.get("trigger", "")),
            "failed_task_id": str(
                self.runtime_recovery.get("failed_task_id")
                or self._runtime_recovery_context.get("failed_task_id", "")
                or ""
            ).strip(),
            "violations": deepcopy(list(self._runtime_recovery_context.get("violations") or [])),
            "used_llm_bridge": bool(self.runtime_recovery.get("used_llm_bridge", False)),
            "system_coordination_state": deepcopy(
                self._runtime_recovery_context.get("system_coordination_state") or {}
            ),
            "proposal_fingerprint": str(approval.get("proposal_fingerprint") or "").strip(),
            "state": "executing" if dispatched_bridge_task_ids else "approved",
            "dispatched_bridge_task_ids": dispatched_bridge_task_ids,
            "completed_bridge_task_ids": completed_bridge_task_ids,
            "execution_started_at_utc": str(
                dict(self.runtime_recovery.get("bridge_artifacts") or {}).get(
                    "execution_started_at_utc"
                )
                or ""
            ).strip(),
            "last_dispatched_task_id": (
                dispatched_bridge_task_ids[-1] if dispatched_bridge_task_ids else ""
            ),
            "last_completed_task_id": (
                completed_bridge_task_ids[-1] if completed_bridge_task_ids else ""
            ),
            "continuation_requirements": deepcopy(
                self._strip_continuation_requirement_actuals(
                    modeled_gap.get("continuation_requirements") or []
                )
            ),
            "pending_nominal_task_ids": deepcopy(
                modeled_gap.get("pending_nominal_task_ids") or []
            ),
            "continuation_repair_attempts": 0,
            "source_mode": str(
                bridge_debug.get("bridge_mode")
                or self.runtime_recovery.get("bridge_mode")
                or self._runtime_bridge_session_mode()
                or ""
            ).strip(),
            "validation_policy": str(
                self.runtime_recovery.get("validation_policy")
                or self._runtime_bridge_session_validation_policy()
                or ""
            ).strip(),
            "source_archive_path": str(
                archive_replay.get("source_path")
                or self.runtime_recovery.get("selected_archive_path")
                or ""
            ).strip(),
            "source_archive_label": str(
                archive_replay.get("source_label")
                or self.runtime_recovery.get("selected_archive_label")
                or ""
            ).strip(),
        }
        if execution_policy:
            reconstructed_sequence["execution_policy"] = deepcopy(execution_policy)
        source = str(bridge_debug.get("source") or "").strip()
        if source:
            reconstructed_sequence["source"] = source
        scenario_id = str(bridge_debug.get("scenario_id") or "").strip()
        if scenario_id:
            reconstructed_sequence["scenario_id"] = scenario_id
        repair_task_id = str(self._runtime_recovery_context.get("repair_task_id") or "").strip()
        if repair_task_id:
            reconstructed_sequence["repair_task_id"] = repair_task_id
        repair_target_task_id = str(
            self._runtime_recovery_context.get("repair_target_task_id") or ""
        ).strip()
        if repair_target_task_id:
            reconstructed_sequence["repair_target_task_id"] = repair_target_task_id
        generated_code_verification = deepcopy(
            self.runtime_recovery.get("generated_code_verification") or {}
        )
        if generated_code_verification:
            reconstructed_sequence["generated_code_verification"] = generated_code_verification

        self.logger.warning(
            "[Product] Reconstructed missing active bridge sequence %s during runtime validation from compiled bridge task metadata.",
            bridge_sequence_id,
        )
        self._set_runtime_recovery(
            message=str(self.runtime_recovery.get("message", "") or "").strip(),
            active_bridge_sequence=reconstructed_sequence,
        )
        return reconstructed_sequence

    def _active_bridge_blocks_nominal_dispatch(self) -> bool:
        active_bridge_sequence = self._active_bridge_sequence()
        if not active_bridge_sequence:
            return False
        state = self._bridge_sequence_state(active_bridge_sequence)
        return state in {
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
        if self._active_bridge_blocks_nominal_dispatch():
            return None

        graph_ready = self._graph_ready_task_nodes()
        if not graph_ready:
            return None

        active_bridge_sequence = self._active_bridge_sequence()
        if not active_bridge_sequence:
            active_bridge_sequence = self._reconstruct_active_bridge_sequence_for_validation()
        active_bridge_sequence_id = str(
            (active_bridge_sequence or {}).get("bridge_sequence_id") or ""
        ).strip()
        filtered_graph_ready: list[dict[str, Any]] = []
        orphaned_bridge_task_ids: list[str] = []
        for node in graph_ready:
            bridge_sequence_id = str(node.get("bridge_sequence_id") or "").strip()
            if not bridge_sequence_id:
                filtered_graph_ready.append(node)
                continue
            if active_bridge_sequence_id and bridge_sequence_id == active_bridge_sequence_id:
                filtered_graph_ready.append(node)
                continue
            orphaned_bridge_task_ids.append(str(node.get("id") or "").strip())
        new_orphaned_bridge_task_ids = [
            task_id
            for task_id in orphaned_bridge_task_ids
            if task_id and task_id not in self._orphaned_bridge_task_warning_ids
        ]
        if new_orphaned_bridge_task_ids:
            self._orphaned_bridge_task_warning_ids.update(new_orphaned_bridge_task_ids)
            self.logger.warning(
                "[Product] Skipping orphaned bridge task(s) without matching active bridge sequence: %s",
                new_orphaned_bridge_task_ids,
            )
        graph_ready = filtered_graph_ready
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
            return None

        if disabled_frontier:
            self._mark_runtime_des_human_required(
                disabled_frontier=disabled_frontier,
                message=self._runtime_des_disabled_event_message(
                    disabled_frontier[0] if disabled_frontier else {},
                    human_required=True,
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
        type(self._agent)._load_shared_tools_catalogue()
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

        def caps_can_reach(caps: Any) -> tuple[bool, bool]:
            if not isinstance(caps, dict) or not caps:
                return False, False
            reachability = caps.get("reachability")
            has_reachability = isinstance(reachability, list) and bool(reachability)
            if has_reachability:
                normalized = {str(item).strip() for item in reachability if str(item).strip()}
                if location_key in normalized:
                    return True, True
            staging_areas = caps.get("staging_areas")
            if isinstance(staging_areas, dict) and location_key in staging_areas:
                return True, has_reachability
            if isinstance(staging_areas, list):
                normalized = {str(item).strip() for item in staging_areas if str(item).strip()}
                if location_key in normalized:
                    return True, has_reachability
            return False, has_reachability

        resource = self.process_planner._resource_by_jid(resource_jid)
        caps = getattr(resource, "static_capabilities", {}) if resource is not None else {}
        if not isinstance(caps, dict) or not caps:
            return True
        reachable, has_reachability = caps_can_reach(caps)
        if reachable:
            return True

        bridge_proposal = dict((getattr(self, "runtime_recovery", {}) or {}).get("bridge_proposal") or {})
        bridge_macro_tasks = []
        if bridge_proposal and hasattr(self.process_planner, "_primitive_bridge_macro_tasks"):
            try:
                bridge_macro_tasks = self.process_planner._primitive_bridge_macro_tasks(
                    bridge_proposal
                )
            except Exception:
                bridge_macro_tasks = []
        for task in bridge_macro_tasks:
            if not isinstance(task, dict):
                continue
            if str(task.get("resource_jid") or "").strip() != str(resource_jid or "").strip():
                continue
            projected_snapshot = dict(task.get("projected_snapshot") or {})
            for projected_caps in (
                projected_snapshot.get("static_capabilities"),
                projected_snapshot,
            ):
                projected_reachable, _projected_has_reachability = caps_can_reach(projected_caps)
                if projected_reachable:
                    return True

        return not has_reachability

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
            token = ProductRecoveryController._model_name_from_mapping(
                value.get(key), part_name
            )
            if token:
                return token

        grounding_context = value.get("grounding_context")
        if isinstance(grounding_context, dict):
            parts = grounding_context.get("parts")
            if isinstance(parts, dict):
                token = ProductRecoveryController._model_name_from_mapping(
                    parts.get(part_name), part_name
                )
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
                    for param_name in params
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
    def _strip_continuation_requirement_actuals(
        requirements: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for requirement in requirements or []:
            if not isinstance(requirement, dict):
                continue
            entry = deepcopy(requirement)
            entry.pop("actual", None)
            cleaned.append(entry)
        return cleaned

    @staticmethod
    def _runtime_des_disabled_event_message(
        disabled_event: dict[str, Any] | None,
        *,
        inserted_repair: bool = False,
        human_required: bool = False,
    ) -> str:
        event = dict(disabled_event or {})
        task_id = str(event.get("task_id") or "").strip()
        violations = [
            dict(item)
            for item in (event.get("guard_violations") or [])
            if isinstance(item, dict)
        ]
        held_part_violation = next(
            (
                item
                for item in violations
                if str(item.get("field") or "").strip() == "held_part"
                and str(item.get("expected") or "").strip()
            ),
            None,
        )
        if inserted_repair:
            if task_id:
                return (
                    f"Runtime guard disabled for {task_id}; spliced DES "
                    "acquire_entity repair before the disabled nominal task."
                )
            return (
                "Runtime DES supervisor spliced controllable repair event "
                "'acquire_entity' before the disabled nominal task."
            )
        if human_required:
            if task_id and held_part_violation:
                return (
                    f"Runtime guard disabled for {task_id}, and no DES "
                    "acquire_entity repair was applicable. Human intervention "
                    "required."
                )
            if task_id:
                return (
                    f"Runtime guard disabled for {task_id}, and no deterministic "
                    "runtime repair was applicable. Human intervention required."
                )
            return (
                "Runtime DES supervisor found graph-ready event(s), but their plant "
                "guards are disabled and no deterministic repair event was applicable."
            )
        return ""

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

    def _bridge_projected_plant_state_for_resource(
        self,
        *,
        resource_jid: str,
        bridge_tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        projected_plant = self._build_runtime_plant_state(resource_jids=[resource_jid])
        resources = dict(projected_plant.get("resources") or {})
        resource_entry = dict(resources.get(resource_jid) or {})
        parts = deepcopy(dict(projected_plant.get("parts") or {}))

        for bridge_task in bridge_tasks:
            if not isinstance(bridge_task, dict):
                continue
            projected_snapshot = dict(bridge_task.get("projected_snapshot") or {})
            if projected_snapshot:
                resource_entry.update(deepcopy(projected_snapshot))
            part_name = str(
                bridge_task.get("part_name")
                or self._tracked_part_name_for_task(bridge_task)
                or ""
            ).strip()
            projected_part_entry = dict(bridge_task.get("projected_part_entry") or {})
            if part_name and projected_part_entry:
                part_entry = dict(parts.get(part_name) or {})
                part_entry.update(deepcopy(projected_part_entry))
                parts[part_name] = part_entry

        resources[resource_jid] = resource_entry
        projected_plant["resources"] = resources
        projected_plant["parts"] = parts
        return projected_plant

    def _compile_acquire_entity_repair_node(
        self,
        *,
        resource_jid: str,
        part_name: str,
        source_location: str,
        disabled_event: dict[str, Any],
        active_bridge_sequence: dict[str, Any] | None,
        sequence_id: str = "",
        sequence_index: int = 1,
        sequence_length: int = 1,
        requirement_id: str = "",
        change_reason: str = "",
    ) -> dict[str, Any] | None:
        sequence_token = str(sequence_id or "").strip() or f"DESREPAIR_{uuid.uuid4().hex[:8].upper()}"
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
        if not self._resource_can_reach_location(resource_jid, source_location):
            message = (
                "Cannot compile acquire_entity repair: source_location "
                f"'{source_location}' is not reachable for resource '{resource_jid}'."
            )
            self._record_repair_compile_error(
                disabled_event=disabled_event,
                operator="acquire_entity",
                message=message,
            )
            self.logger.warning("[Product] Runtime DES repair compile failed: %s", message)
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
        disabled_task_id = str(disabled_event.get("task_id") or "").strip()
        if not disabled_task_id:
            message = "Cannot compile acquire_entity repair: disabled event is missing task_id."
            self._record_repair_compile_error(
                disabled_event=disabled_event,
                operator="acquire_entity",
                message=message,
            )
            self.logger.warning("[Product] Runtime DES repair compile failed: %s", message)
            return None
        disabled_task = self.process_planner._find_node(disabled_task_id)
        if not isinstance(disabled_task, dict):
            message = (
                "Cannot compile acquire_entity repair: disabled target task "
                f"'{disabled_task_id}' was not found in the planner graph."
            )
            self._record_repair_compile_error(
                disabled_event=disabled_event,
                operator="acquire_entity",
                message=message,
            )
            self.logger.warning("[Product] Runtime DES repair compile failed: %s", message)
            return None
        disabled_task_resource_jid = str(disabled_task.get("resource_jid") or "").strip()
        if disabled_task_resource_jid != resource_jid:
            message = (
                "Cannot compile acquire_entity repair: disabled target task "
                f"'{disabled_task_id}' belongs to resource '{disabled_task_resource_jid or '<missing>'}', "
                f"not '{resource_jid}'."
            )
            self._record_repair_compile_error(
                disabled_event=disabled_event,
                operator="acquire_entity",
                message=message,
            )
            self.logger.warning("[Product] Runtime DES repair compile failed: %s", message)
            return None
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
        node = {
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
            "predecessors": [],
            "successors": [],
            "bridge_sequence_id": sequence_token,
            "bridge_sequence_index": int(sequence_index or 1),
            "bridge_sequence_length": int(sequence_length or 1),
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
            "restores_event_id": disabled_task_id,
            "guard_violations": deepcopy(disabled_event.get("guard_violations") or []),
            "producer_semantics": ["pick_approach", "pick_grasp"],
            "disabled_event": deepcopy(disabled_event),
            "change_reason": (
                str(change_reason).strip()
                or (
                    "INSERTION: Runtime DES guard-restoration event 'acquire_entity' "
                    f"spliced before disabled event {disabled_task_id or '-'}"
                )
            ),
        }
        requirement_token = str(requirement_id or "").strip()
        if requirement_token:
            node["requirement_id"] = requirement_token
        return node

    def _append_bridge_resume_entry_acquire_entity_repair_if_needed(
        self,
        *,
        tasks: list[dict[str, Any]],
        resume_entry_task_ids_by_resource: dict[str, str],
        patch_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        task_groups_by_resource: dict[str, list[dict[str, Any]]] = {}
        for task in tasks:
            resource_jid = str(task.get("resource_jid") or "").strip()
            if resource_jid:
                task_groups_by_resource.setdefault(resource_jid, []).append(task)

        for resource_jid, entry_task_id in dict(resume_entry_task_ids_by_resource or {}).items():
            resume_task_id = str(entry_task_id or "").strip()
            if not resource_jid or not resume_task_id:
                continue
            bridge_tasks = list(task_groups_by_resource.get(resource_jid) or [])
            if not bridge_tasks:
                continue
            entry_task = self.process_planner._find_node(resume_task_id)
            if not isinstance(entry_task, dict):
                continue
            projected_plant = self._bridge_projected_plant_state_for_resource(
                resource_jid=resource_jid,
                bridge_tasks=bridge_tasks,
            )
            violations = self._event_guard_violations(
                entry_task,
                plant_state=projected_plant,
                enforce_unknown=False,
            )
            held_part_violation = next(
                (
                    item
                    for item in violations
                    if str(item.get("field") or "").strip() == "held_part"
                    and str(item.get("expected") or "").strip()
                ),
                None,
            )
            if not held_part_violation:
                continue

            part_name = str(held_part_violation.get("expected") or "").strip()
            source_location = str(
                self._plant_part_field(projected_plant, part_name, "location") or ""
            ).strip()
            disabled_event = {
                "task_id": resume_task_id,
                "function_name": str(entry_task.get("function_name") or "").strip(),
                "resource_jid": resource_jid,
                "part_name": part_name,
                "guard_violations": violations,
                "params": deepcopy(entry_task.get("params") or {}),
            }
            next_bridge_length = len(tasks) + 1
            sequence_id = str(bridge_tasks[0].get("bridge_sequence_id") or "").strip()
            repair_node = self._compile_acquire_entity_repair_node(
                resource_jid=resource_jid,
                part_name=part_name,
                source_location=source_location,
                disabled_event=disabled_event,
                active_bridge_sequence=None,
                sequence_id=sequence_id,
                sequence_index=next_bridge_length,
                sequence_length=next_bridge_length,
                requirement_id=str(entry_task.get("requirement_id") or "").strip(),
                change_reason=(
                    "INSERTION: Approved bridge recovery — splice acquire_entity "
                    f"before {resume_task_id}"
                ),
            )
            if repair_node is None:
                raise ValueError(
                    "Approved bridge recovery could not compile acquire_entity before "
                    f"{resume_task_id}"
                )

            repair_patch_rows: list[dict[str, Any]] = []
            self.process_planner._splice_runtime_des_repair_before_task(
                repair_patch_rows,
                repair_task=repair_node,
                target_task_id=resume_task_id,
                change_prefix="Approved bridge recovery",
            )
            for bridge_task in list(tasks) + [repair_node]:
                bridge_task_id = str(bridge_task.get("id") or "").strip()
                if not bridge_task_id:
                    continue
                repair_patch_rows.append(
                    {
                        "id": bridge_task_id,
                        "bridge_sequence_length": next_bridge_length,
                    }
                )
            if patch_rows is not None:
                patch_rows.extend(repair_patch_rows)
                repair_node["bridge_sequence_length"] = next_bridge_length
                tasks.append(deepcopy(repair_node))
                task_groups_by_resource.setdefault(resource_jid, []).append(
                    deepcopy(repair_node)
                )
                for bridge_task in tasks:
                    bridge_task["bridge_sequence_length"] = next_bridge_length
                continue

            self.process_planner._apply_replan_patch(repair_patch_rows)
            inserted_repair_node = self.process_planner._find_node(
                str(repair_node.get("id") or "").strip()
            )
            if isinstance(inserted_repair_node, dict):
                inserted_repair_node["bridge_sequence_length"] = next_bridge_length
                self.task_states[str(inserted_repair_node.get("id") or "").strip()] = str(
                    inserted_repair_node.get("status") or "pending"
                ).strip() or "pending"
                tasks.append(deepcopy(inserted_repair_node))
                task_groups_by_resource.setdefault(resource_jid, []).append(
                    deepcopy(inserted_repair_node)
                )
            for bridge_task in tasks:
                bridge_task["bridge_sequence_length"] = next_bridge_length

    def _prune_redundant_bridge_resume_move_home_if_satisfied(
        self,
        *,
        tasks: list[dict[str, Any]],
        resumable_task_ids: list[str],
        resumable_task_ids_by_resource: dict[str, list[str]],
        resume_entry_task_ids_by_resource: dict[str, str],
        deleted_task_ids: list[str],
        patch_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        task_groups_by_resource: dict[str, list[dict[str, Any]]] = {}
        for task in tasks:
            resource_jid = str(task.get("resource_jid") or "").strip()
            if resource_jid:
                task_groups_by_resource.setdefault(resource_jid, []).append(task)

        for resource_jid, entry_task_id in list(
            dict(resume_entry_task_ids_by_resource or {}).items()
        ):
            resume_task_id = str(entry_task_id or "").strip()
            if not resource_jid or not resume_task_id:
                continue
            bridge_tasks = list(task_groups_by_resource.get(resource_jid) or [])
            if not bridge_tasks:
                continue
            entry_task = self.process_planner._find_node(resume_task_id)
            if not isinstance(entry_task, dict):
                continue
            if str(entry_task.get("function_name") or "").strip() != "move_home":
                continue

            projected_plant = self._bridge_projected_plant_state_for_resource(
                resource_jid=resource_jid,
                bridge_tasks=bridge_tasks,
            )
            projected_state = str(
                self._plant_resource_field(
                    projected_plant,
                    resource_jid,
                    "current_state",
                )
                or ""
            ).strip()
            projected_pose_ref = str(
                self._plant_resource_field(
                    projected_plant,
                    resource_jid,
                    "current_pose_ref",
                )
                or ""
            ).strip()
            projected_location = str(
                self._plant_resource_field(
                    projected_plant,
                    resource_jid,
                    "current_location",
                )
                or self._plant_resource_field(
                    projected_plant,
                    resource_jid,
                    "location",
                )
                or ""
            ).strip()
            if projected_state != "idle":
                continue
            if projected_pose_ref != "home" and projected_location != "home":
                continue

            delete_row = {
                "id": resume_task_id,
                "delete": True,
                "change_reason": (
                    "DELETION: Approved bridge recovery already leaves "
                    f"{resource_jid} at home idle before redundant move_home task "
                    f"{resume_task_id}"
                ),
            }
            if patch_rows is not None:
                patch_rows.append(delete_row)
            else:
                self.process_planner._apply_replan_patch([delete_row])
            if resume_task_id in resumable_task_ids:
                resumable_task_ids.remove(resume_task_id)
            if resume_task_id not in deleted_task_ids:
                deleted_task_ids.append(resume_task_id)

            remaining_task_ids = [
                str(task_id).strip()
                for task_id in (resumable_task_ids_by_resource.get(resource_jid) or [])
                if str(task_id).strip()
                and str(task_id).strip() != resume_task_id
                and isinstance(self.process_planner._find_node(str(task_id).strip()), dict)
            ]
            if not remaining_task_ids:
                resumable_task_ids_by_resource.pop(resource_jid, None)
                resume_entry_task_ids_by_resource.pop(resource_jid, None)
                continue

            resumable_task_ids_by_resource[resource_jid] = remaining_task_ids
            next_entry_task_id = min(
                remaining_task_ids,
                key=lambda task_id: (
                    self.process_planner._task_sequence_index_key(
                        dict(self.process_planner._find_node(task_id) or {}).get(
                            "sequence_index"
                        )
                    ),
                    task_id,
                ),
            )
            resume_entry_task_ids_by_resource[resource_jid] = next_entry_task_id

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
        node = self._compile_acquire_entity_repair_node(
            resource_jid=resource_jid,
            part_name=part_name,
            source_location=source_location,
            disabled_event=disabled_event,
            active_bridge_sequence=active_bridge_sequence,
            sequence_id=sequence_id,
            sequence_index=1,
            sequence_length=1,
        )
        if node is None:
            return None
        task_id = str(node.get("id") or "").strip()
        disabled_task_id = str(disabled_event.get("task_id") or "").strip()
        try:
            patch_rows: list[dict[str, Any]] = []
            self.process_planner._splice_runtime_des_repair_before_task(
                patch_rows,
                repair_task=node,
                target_task_id=disabled_task_id,
                change_prefix="Runtime DES guard-restoration repair",
            )
            self.process_planner._apply_replan_patch(patch_rows)
        except Exception as exc:
            message = (
                "Cannot compile acquire_entity repair: failed to splice repair "
                f"before disabled target task '{disabled_task_id}': {exc}"
            )
            self._record_repair_compile_error(
                disabled_event=disabled_event,
                operator="acquire_entity",
                message=message,
            )
            self.logger.warning("[Product] Runtime DES repair compile failed: %s", message)
            return None
        inserted_node = self.process_planner._find_node(task_id)
        planner_owned_node = inserted_node if isinstance(inserted_node, dict) else None
        if planner_owned_node is not None:
            self.task_states[task_id] = str(
                planner_owned_node.get("status") or "pending"
            ).strip() or "pending"

        next_sequence = deepcopy(active_bridge_sequence or {})
        next_sequence.update(
            {
                "bridge_sequence_id": sequence_id,
                "bridge_task_ids": [task_id],
                "bridge_sequence_length": 1,
                "state": "approved",
                "last_task_id": task_id,
                "trigger": str(trigger or "").strip() or next_sequence.get("trigger", ""),
                "used_llm_bridge": bool(next_sequence.get("used_llm_bridge", False)),
                "continuation_repair_attempts": int(
                    next_sequence.get("continuation_repair_attempts") or 0
                )
                + 1,
                "repair_operator": "acquire_entity",
                "repair_intent": "restore_event_guard",
                "repair_task_id": task_id,
                "repair_target_task_id": disabled_task_id,
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
                        "model_name": str(
                            dict(node.get("projected_part_entry") or {}).get("model_name") or ""
                        ).strip(),
                    },
                },
                "disabled_event": deepcopy(disabled_event),
                "repair_task_id": task_id,
                "repair_intent": "restore_event_guard",
                "restores_event_id": disabled_task_id,
                "guard_violations": deepcopy(disabled_event.get("guard_violations") or []),
                "producer_semantics": ["pick_approach", "pick_grasp"],
            }
        )
        current_trigger = str(trigger or "").strip() or str(
            self.runtime_recovery.get("trigger", "") or ""
        ).strip()
        current_failed_task_id = str(
            next_sequence.get("failed_task_id")
            or self.runtime_recovery.get("failed_task_id")
            or ""
        ).strip()
        current_violations = deepcopy(
            self._runtime_recovery_context.get("violations")
            or self.runtime_recovery.get("violations")
            or []
        )
        current_system_state = deepcopy(
            next_sequence.get("system_coordination_state")
            or self._runtime_recovery_context.get("system_coordination_state")
            or {}
        )
        current_feedback_history = list(
            self.runtime_recovery.get("bridge_feedback_history") or []
        )
        validation_message = (
            f"Runtime DES repair {task_id} inserted before {disabled_task_id or '-'}; "
            "validating repaired continuation before dispatch."
        )
        self._runtime_recovery_context = {
            "trigger": current_trigger,
            "failed_task_id": current_failed_task_id,
            "violations": deepcopy(current_violations),
            "system_coordination_state": current_system_state,
            "bridge_feedback_history": list(current_feedback_history),
            "repair_task_id": task_id,
            "repair_target_task_id": disabled_task_id,
        }
        self._set_runtime_recovery(
            status="validating",
            resolution_class="runtime_des_repair",
            trigger=current_trigger,
            failed_task_id=current_failed_task_id,
            message=validation_message,
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=bool(next_sequence.get("used_llm_bridge", False)),
            bridge_debug=bridge_debug,
            bridge_approval_state="approved",
            active_bridge_sequence=next_sequence,
            violations=current_violations,
            append_history=True,
            history_message=(
                f"Runtime DES repair event {task_id} spliced before disabled "
                f"event {disabled_task_id or '-'}."
            ),
        )
        self.logger.info(
            "[Product] Runtime DES repair %s inserted before %s; validating repaired plan before dispatch.",
            task_id,
            disabled_task_id or "<unknown>",
        )
        try:
            self._persist_product_state()
            self._send_runtime_plan_validation_check_sync()
        except Exception as exc:
            message = (
                f"{self.agent_name}: runtime DES repair '{task_id}' was inserted before "
                f"'{disabled_task_id or '-'}' but plan validation dispatch failed ({exc})."
            )
            self.logger.warning("[Product] %s", message)
            self._set_runtime_recovery(
                status="human_required",
                resolution_class="human_required",
                trigger=current_trigger,
                failed_task_id=current_failed_task_id,
                message=message,
                attempts_used=self._runtime_repair_fail_streak,
                attempts_max=self._runtime_repair_max_attempts,
                used_llm_bridge=bool(next_sequence.get("used_llm_bridge", False)),
                bridge_debug=bridge_debug,
                bridge_approval_state="approved",
                active_bridge_sequence=next_sequence,
                violations=current_violations,
                append_history=True,
                history_message=message,
            )
            self._persist_product_state()
            return None
        return planner_owned_node or node

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
        bridge_proposal: dict[str, Any] | None = None,
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
            "bridge_proposal_available": isinstance(bridge_proposal, dict),
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
            "proposal_build_status": str(
                payload.get("proposal_build_status") or ""
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
            "proposal_build_status": "disabled",
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
            result["proposal_build_status"] = "skipped"
            result["reason"] = (
                "runtime bridge fixture replay requires reasoning_mode=multi_turn"
            )
            return result

        if not resolved_path.exists():
            result["load_status"] = "missing"
            result["proposal_build_status"] = "skipped"
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
            result["proposal_build_status"] = "skipped"
            result["reason"] = (
                f"failed to parse fixture final_output artifact: {exc}"
            )
            return result

        if not isinstance(final_output_payload, dict) or not final_output_payload:
            result["load_status"] = "loaded"
            result["proposal_build_status"] = "rejected"
            result["reason"] = (
                "fixture final_output artifact did not contain a JSON object"
            )
            return result

        result["final_output"] = deepcopy(final_output_payload)
        result["load_status"] = "loaded"

        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn import (
            build_multi_turn_bridge_proposal,
        )

        adapter_result = build_multi_turn_bridge_proposal(
            final_output_payload=final_output_payload,
            prepared_bridge_request=prepared_bridge_request,
        )
        result["adapter_result"] = deepcopy(adapter_result)
        if isinstance(adapter_result, dict) and adapter_result.get("accepted") is True:
            result["proposal_build_status"] = "accepted"
            result["proposal"] = deepcopy(
                adapter_result.get("bridge_proposal") or {}
            )
            return result

        result["proposal_build_status"] = "rejected"
        result["reason"] = str(
            dict(adapter_result or {}).get("reason")
            or "fixture final_output proposal build failed"
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
                    "bridge_outline_id": str(node.get("bridge_outline_id", "")).strip(),
                    "predecessor_outline_ids": deepcopy(node.get("predecessor_outline_ids") or []),
                    "recovery_group_id": str(node.get("recovery_group_id", "")).strip(),
                    "recovery_parent_failure_id": str(node.get("recovery_parent_failure_id", "")).strip(),
                    "recovery_kind": str(node.get("recovery_kind", "")).strip(),
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
        tail_ids: list[str] = []
        for node in self.process_planner._bridge_sequence_nodes(bridge_sequence_id):
            node_id = str(node.get("id", "")).strip()
            if not node_id or node_id == completed_task_id:
                continue
            status = str(node.get("status") or "").strip().lower()
            if status in {"completed", "finished"}:
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
        allow_missing_current_location: bool = False,
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
            if (
                allow_missing_current_location
                and key == "current_location"
                and actual_value is None
                and projected_value is not None
            ):
                continue
            equivalence_resolver = getattr(profile, "snapshot_equivalence_resolver", None)
            if callable(equivalence_resolver) and equivalence_resolver(
                field=key,
                actual_snapshot=actual_snapshot,
                projected_snapshot=projected_snapshot,
                actual_value=actual_value,
                projected_value=projected_value,
                profile=profile,
            ):
                continue
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
        self,
        *,
        skip_revalidation: bool = False,
        skip_recovery_safety_validation: bool = False,
    ) -> None:
        # Always recompile the FSA so that newly-inserted tasks (e.g.,
        # recovery bridge macros) are present in the transition table.
        # skip_revalidation only skips the CCA-side safety-rule check.
        self.process_planner.compile_global_fsa()
        self.process_planner.save_global_fsa(self.global_fsa_path)

        request_id = f"runtime_plan_validation_{uuid.uuid4().hex}"
        if not isinstance(getattr(self, "_runtime_recovery_context", None), dict):
            self._runtime_recovery_context = {}
        self._runtime_recovery_context["active_runtime_plan_validation_request_id"] = (
            request_id
        )
        payload = self._build_plan_validation_payload(
            skip_revalidation=skip_revalidation,
            skip_recovery_safety_validation=skip_recovery_safety_validation,
            request_id=request_id,
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
        self,
        *,
        skip_revalidation: bool = False,
        skip_recovery_safety_validation: bool = False,
    ) -> None:
        self._dispatch_runtime_plan_validation_check(
            skip_revalidation=skip_revalidation,
            skip_recovery_safety_validation=skip_recovery_safety_validation,
        )

    def _send_runtime_plan_validation_check_sync(
        self,
        *,
        skip_revalidation: bool = False,
        skip_recovery_safety_validation: bool = False,
    ) -> None:
        self._dispatch_runtime_plan_validation_check(
            skip_revalidation=skip_revalidation,
            skip_recovery_safety_validation=skip_recovery_safety_validation,
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

        completed_sequence = self._bridge_sequence_with_completed_task(
            active_bridge_sequence,
            task_id=str(task_node.get("id", "")).strip(),
            macro_name=macro_name,
        )
        if not isinstance(completed_sequence, dict):
            completed_sequence = deepcopy(active_bridge_sequence)

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
                active_bridge_sequence=completed_sequence,
                content=content,
                observations=observations,
            )
            return True

        projected_snapshot = dict(task_node.get("projected_snapshot") or {})
        part_name = str(self._tracked_part_name_for_task(task_node) or "").strip()
        projected_part_entry = dict(task_node.get("projected_part_entry") or {})
        mismatch = ""
        if projected_snapshot:
            mismatch = self._bridge_snapshot_mismatch(
                actual_snapshot=actual_snapshot,
                projected_snapshot=projected_snapshot,
                allow_missing_current_location=bool(projected_part_entry),
            )
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
                active_bridge_sequence=completed_sequence,
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
            next_sequence = deepcopy(completed_sequence)
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
                    self._runtime_des_disabled_event_message(
                        disabled_frontier[0] if disabled_frontier else {},
                        human_required=True,
                    )
                    or (
                        "Generated bridge verification matched macro projections, but "
                        "the runtime DES continuation guard is still disabled. Human "
                        "intervention required."
                    )
                )
                self._mark_runtime_des_human_required(
                    disabled_frontier=disabled_frontier,
                    message=message,
                )
                await asyncio.to_thread(self._persist_product_state)
                return True

            next_sequence["state"] = "completed"
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
                last_completed_bridge_sequence=self._bridge_sequence_summary(next_sequence),
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
            next_sequence = deepcopy(completed_sequence)
            next_sequence["state"] = "executing"
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

        terminal_sequence = self._bridge_sequence_with_completed_task(
            completed_sequence,
            task_id=str(task_node.get("id", "")).strip(),
            macro_name=macro_name,
            terminal=True,
        ) or deepcopy(completed_sequence)
        self._runtime_recovery_context = {
            "trigger": trigger,
            "failed_task_id": failed_task_id,
            "violations": deepcopy(violations),
            "system_coordination_state": deepcopy(refreshed_system_state),
            "bridge_feedback_history": list(feedback_history),
            "repair_task_id": str(terminal_sequence.get("repair_task_id") or "").strip(),
            "repair_target_task_id": str(
                terminal_sequence.get("repair_target_task_id") or ""
            ).strip(),
        }
        repair_task_id = str(terminal_sequence.get("repair_task_id") or "").strip()
        repair_target_task_id = str(
            terminal_sequence.get("repair_target_task_id") or ""
        ).strip()
        disabled_frontier = self._bridge_continuation_disabled_frontier(terminal_sequence)
        if disabled_frontier:
            self._record_runtime_des_trace(
                graph_ready_event_ids=list(
                    terminal_sequence.get("pending_nominal_task_ids") or []
                ),
                plant_enabled_event_ids=[],
                disabled_frontier=disabled_frontier,
            )
            repair_node = self._try_compile_controllable_repair(
                disabled_frontier=disabled_frontier,
                trigger="bridge_continuation_guard",
                active_bridge_sequence=terminal_sequence,
            )
            if repair_node:
                await asyncio.to_thread(self._persist_plan_snapshot)
                await asyncio.to_thread(self._persist_product_state)
                return True
        validation_message = (
            f"Bridge macro '{macro_name}' completed the approved recovery bridge; "
            "validating updated plan."
        )
        if repair_task_id and repair_task_id == str(task_node.get("id", "")).strip():
            validation_message = (
                f"Runtime DES repair '{macro_name}' completed; validating restored "
                f"continuation before reactivating {repair_target_task_id or '<unknown>'}."
            )
            self.logger.info(
                "[Product] Runtime DES repair %s completed; validating restored task %s.",
                repair_task_id,
                repair_target_task_id or "<unknown>",
            )
        active_validation_policy = self._normalize_runtime_bridge_validation_policy(
            dict(active_bridge_sequence.get("execution_policy") or {}).get("validation_policy")
            or active_bridge_sequence.get("validation_policy")
            or self.runtime_recovery.get("validation_policy")
            or self._runtime_bridge_session_validation_policy()
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
            last_completed_bridge_sequence=self._bridge_sequence_summary(terminal_sequence),
            bridge_feedback_history=feedback_history,
            violations=violations,
            append_history=True,
            history_message=validation_message,
        )
        self._clear_plan_safety_alert()
        await self._send_runtime_plan_validation_check(
            skip_recovery_safety_validation=(active_validation_policy == "no_validation")
        )
        await asyncio.to_thread(self._persist_plan_snapshot)
        await asyncio.to_thread(self._persist_product_state)
        await asyncio.to_thread(self._persist_resource_state)
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
        session_bridge_mode = self._normalize_runtime_bridge_mode(self._runtime_bridge_mode)
        session_validation_policy = self._normalize_runtime_bridge_validation_policy(
            self._runtime_bridge_validation_policy
        )
        self._runtime_recovery_context = {
            "trigger": str(trigger or "").strip(),
            "failed_task_id": str(failed_task_id or "").strip(),
            "violations": deepcopy(list(violations or [])),
            "system_coordination_state": deepcopy(system_coordination_state or {}),
            "bridge_feedback_history": list(feedback_history),
            "bridge_mode": session_bridge_mode,
            "validation_policy": session_validation_policy,
            "selected_archive_path": str(self._runtime_bridge_archive_path or "").strip(),
            "selected_archive_label": str(self._runtime_bridge_archive_label or "").strip(),
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
            bridge_stage="none",
            active_bridge_sequence=None,
            bridge_feedback_history=feedback_history,
            violations=violations,
            append_history=True,
            history_message=history_message or (
                f"DES runtime recovery attempt {attempt_number}/{self._runtime_repair_max_attempts} started."
            ),
        )

        scenario_hint = ""
        fixture_replay_enabled = self._runtime_bridge_fixture_replay_enabled()
        fixture_source_path = self._runtime_bridge_fixture_final_output_source_path()
        self.logger.info(
            "[Product] Runtime bridge fixture replay gate: fixture_replay=%s source_path=%s env_var=CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT",
            fixture_replay_enabled,
            fixture_source_path or "<unset>",
        )
        bridge_generation_mode = "auto" if session_bridge_mode == "auto" else "manual"
        if fixture_replay_enabled and bridge_generation_mode != "manual":
            self.logger.info(
                "[Product] Forcing manual bridge handoff for runtime bridge replay: fixture_replay=%s mode=%s->manual",
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
                bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or bridge_debug or {})
                bridge_debug = self._bridge_debug_with_runtime_handoff(
                    bridge_debug,
                    handoff_owner="product_agent",
                    bridge_mode=session_bridge_mode,
                    validation_policy=session_validation_policy,
                    auto_start_requested=bool(session_bridge_mode == "auto"),
                    auto_start_started=False,
                )
                prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
                bridge_debug = self._ensure_live_bridge_per_turn_debug_dir(prepared_bridge_request)
                self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(
                    prepared_bridge_request
                )
                self.logger.info(
                    "[Product] Prepared bridge request returned to ProductAgent. mode=%s",
                    session_bridge_mode,
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
                selected_archive_path = self._runtime_bridge_session_archive_path()
                selected_archive_label = self._runtime_bridge_session_archive_label()
                message = (
                    base_message
                    or (
                        "DES found no modeled continuation. Selected archived bridge run will auto-load and auto-run."
                        if session_bridge_mode == "pre_ran" and selected_archive_path
                        else "Pre-ran mode is selected, but no archived bridge run is selected."
                        if session_bridge_mode == "pre_ran"
                        else "DES found no modeled continuation. Bridge session is ready for LLM reasoning."
                    )
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
                    bridge_stage="none",
                    artifact_directory=str(bridge_debug.get("artifact_directory") or "").strip(),
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
                autostart_pre_ran = bool(
                    session_bridge_mode == "pre_ran" and selected_archive_path
                )
                if not autostart_pre_ran:
                    await asyncio.to_thread(self._persist_product_state)
                if session_bridge_mode == "pre_ran":
                    if not selected_archive_path:
                        warning = (
                            "Pre-ran mode is selected, but no archived bridge run is selected."
                        )
                        self.logger.warning(
                            "[Product] Runtime recovery paused at bridge_ready without archived bridge selection."
                        )
                        recovery = self._set_runtime_recovery(
                            status="bridge_ready",
                            resolution_class="none",
                            trigger=trigger,
                            failed_task_id=failed_task_id,
                            message=warning,
                            attempts_used=attempt_number,
                            attempts_max=self._runtime_repair_max_attempts,
                            used_llm_bridge=False,
                            bridge_proposal=None,
                            bridge_debug=bridge_debug,
                            bridge_approval_state="ready",
                            bridge_stage="none",
                            active_bridge_sequence=None,
                            bridge_feedback_history=feedback_history,
                            violations=violations,
                            append_history=True,
                            history_message=warning,
                        )
                        await asyncio.to_thread(self._persist_product_state)
                        return self._recovery_result_with_action_feedback(
                            recovery,
                            kind="warning",
                            text=warning,
                        )
                    archive_ref = selected_archive_label or selected_archive_path
                    self.logger.info(
                        "[Product] Auto-starting pre-ran archived bridge after DES handoff: archive=%s validation_policy=%s",
                        archive_ref,
                        session_validation_policy,
                    )
                    self._runtime_repair_inflight = False
                    try:
                        await self.load_runtime_bridge_archive_proposal(
                            selected_archive_path,
                        )
                    except Exception as exc:
                        warning = (
                            f"Pre-ran auto-start failed for archived bridge run '{selected_archive_path}': "
                            f"{exc}. Execution remains paused."
                        )
                        self.logger.warning(
                            "[Product] %s",
                            warning,
                        )
                        recovery = self._set_runtime_recovery(
                            status="bridge_ready",
                            resolution_class="none",
                            trigger=trigger,
                            failed_task_id=failed_task_id,
                            message=warning,
                            attempts_used=attempt_number,
                            attempts_max=self._runtime_repair_max_attempts,
                            used_llm_bridge=False,
                            bridge_proposal=None,
                            bridge_debug=bridge_debug,
                            bridge_approval_state="ready",
                            bridge_stage="none",
                            active_bridge_sequence=None,
                            bridge_feedback_history=feedback_history,
                            violations=violations,
                            append_history=True,
                            history_message=warning,
                        )
                        await asyncio.to_thread(self._persist_product_state)
                        return self._recovery_result_with_action_feedback(
                            recovery,
                            kind="warning",
                            text=warning,
                        )
                    self.logger.info(
                        "[Product] Auto-approving pre-ran archived bridge after DES handoff: archive=%s validation_policy=%s",
                        archive_ref,
                        session_validation_policy,
                    )
                    return self.approve_runtime_bridge_proposal_sync()
                verification_ready = self._should_enable_generated_bridge_verification(
                    bridge_debug=bridge_debug if isinstance(bridge_debug, dict) else None,
                )
                if (
                    session_bridge_mode == "auto"
                    or fixture_replay_enabled
                    or (not scenario_hint and verification_ready)
                ):
                    if fixture_replay_enabled:
                        self.logger.info(
                            "[Product] Auto-starting runtime bridge fixture replay from prepared request: source_path=%s verification_ready=%s",
                            self._runtime_bridge_fixture_final_output_path(),
                            verification_ready,
                        )
                    elif session_bridge_mode == "auto":
                        self.logger.info(
                            "[Product] Auto-starting live runtime bridge generation from the prepared request."
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
        active_bridge_state = self._bridge_sequence_state(active_bridge_sequence)
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
        request_id: str = "",
    ) -> bool:
        incoming_request_id = str(request_id or "").strip()
        expected_request_id = ""
        if isinstance(getattr(self, "_runtime_recovery_context", None), dict):
            expected_request_id = str(
                self._runtime_recovery_context.get(
                    "active_runtime_plan_validation_request_id"
                )
                or ""
            ).strip()
        if incoming_request_id:
            if expected_request_id and incoming_request_id != expected_request_id:
                self.logger.info(
                    "[Product] Ignoring stale plan_safety_result request_id=%s; active runtime validation request_id=%s.",
                    incoming_request_id,
                    expected_request_id,
                )
                return True
            if not expected_request_id:
                self.logger.info(
                    "[Product] Ignoring unexpected plan_safety_result request_id=%s because no runtime bridge plan validation is active.",
                    incoming_request_id,
                )
                return True
            self._runtime_recovery_context.pop(
                "active_runtime_plan_validation_request_id", None
            )

        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        validation_policy = self._normalize_runtime_bridge_validation_policy(
            self.runtime_recovery.get("validation_policy")
            or (
                self._runtime_recovery_context.get("validation_policy")
                if isinstance(self._runtime_recovery_context, dict)
                else ""
            )
            or self._runtime_bridge_session_validation_policy()
        )
        if (
            validation_policy == "no_validation"
            and not expected_request_id
            and status in {"", "idle", "resolved"}
        ):
            self.logger.info(
                "[Product] Ignoring plan_safety_result because validation_policy=no_validation and no runtime bridge plan validation is active."
            )
            return True
        if not self._runtime_recovery_context and status in {"", "idle", "resolved"}:
            return False

        active_bridge_sequence = self._active_bridge_sequence()
        if not active_bridge_sequence:
            active_bridge_sequence = self._reconstruct_active_bridge_sequence_for_validation()

        if ok:
            verification_only = self._bridge_is_verification_only(active_bridge_sequence)
            repair_task_id = str(
                (active_bridge_sequence or {}).get("repair_task_id") or ""
            ).strip()
            repair_target_task_id = str(
                (active_bridge_sequence or {}).get("repair_target_task_id") or ""
            ).strip()
            resolution_class = (
                "generated_bridge_verification"
                if verification_only
                else "des_with_llm_bridge"
                if bool(self.runtime_recovery.get("used_llm_bridge", False))
                else "des_only"
            )
            attempts_used = self._runtime_repair_fail_streak
            self._runtime_repair_fail_streak = 0
            reactivated_task_id = ""
            if not active_bridge_sequence:
                reactivated_task_id = self._reactivate_restored_repair_target_from_context()
            success_message = (
                "CCA runtime plan validation passed; generated bridge verification is executing."
                if active_bridge_sequence and verification_only
                else (
                    f"CCA runtime plan validation passed; runtime DES repair {repair_task_id} is validated "
                    f"for restored task {repair_target_task_id or '<unknown>'} and ready to dispatch."
                )
                if active_bridge_sequence and repair_task_id
                else "CCA runtime plan validation passed; approved bridge sequence is executing."
                if active_bridge_sequence
                else f"CCA runtime plan validation passed; restored continuation task {reactivated_task_id} reactivated."
                if reactivated_task_id
                else "CCA runtime plan validation passed; runtime recovery resolved."
            )
            self.logger.info(
                "[Product] Runtime plan validation passed: active_bridge_sequence=%s status_before=%s",
                bool(active_bridge_sequence),
                status,
            )
            if active_bridge_sequence and repair_task_id:
                self.logger.info(
                    "[Product] Runtime DES repair %s validated; enabling dispatch before restored task %s.",
                    repair_task_id,
                    repair_target_task_id or "<unknown>",
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
            if not active_bridge_sequence:
                self._runtime_recovery_context = {}
            elif isinstance(self._runtime_recovery_context, dict):
                self._runtime_recovery_context.pop(
                    "active_runtime_plan_validation_request_id", None
                )
            self._clear_plan_safety_alert()
            if reactivated_task_id:
                await asyncio.to_thread(self._persist_plan_snapshot)
            await asyncio.to_thread(self._persist_product_state)
            return True

        self.logger.warning(
            "[Product] Runtime plan validation failed: status=%s violations=%d",
            status,
            len(violations),
        )
        try:
            repaired_from_witness = (
                self.process_planner.apply_validation_witness_ordering_repairs(
                    violations
                )
            )
        except Exception:
            repaired_from_witness = False
            self.logger.exception(
                "[Product] Failed to apply CCA validation witness ordering repair."
            )
        if repaired_from_witness:
            self.logger.info(
                "[Product] Applied CCA validation witness ordering repair; resubmitting runtime plan validation."
            )
            self._runtime_recovery_context.pop(
                "active_runtime_plan_validation_request_id", None
            )
            try:
                await asyncio.to_thread(self.process_planner.compile_global_fsa)
                if hasattr(self, "global_fsa_path"):
                    await asyncio.to_thread(
                        self.process_planner.save_global_fsa,
                        self.global_fsa_path,
                    )
                await self._send_runtime_plan_validation_check()
                await asyncio.to_thread(self._persist_plan_snapshot)
                await asyncio.to_thread(self._persist_product_state)
                await asyncio.to_thread(self._persist_resource_state)
            except Exception:
                self.logger.exception(
                    "[Product] Failed to resubmit runtime plan validation after CCA witness repair."
                )
            return True

        if self._runtime_repair_inflight:
            self.logger.warning(
                "[Product] Runtime plan validation failed while DES recovery is already running; ignoring duplicate result."
            )
            return True

        if not self._runtime_recovery_context:
            message = (
                f"{self.agent_name}: CCA runtime plan validation failed without an active DES recovery context; "
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
                f"{self.agent_name}: the CCA runtime plan validation still fails after "
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

        self._runtime_recovery_context.pop("active_runtime_plan_validation_request_id", None)
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
        self._ensure_live_bridge_per_turn_debug_dir(prepared_bridge_request)
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
                bridge_debug["bridge_proposal"] = deepcopy(
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
            if pause_status not in {
                "paused_after_outline_turn",
                "ready_for_primitive_generation",
                "paused_after_primitive_turn",
            }:
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
        active_bridge_sequence = self._active_bridge_sequence()
        if self._bridge_sequence_is_live(active_bridge_sequence):
            warning = "Live bridge generation is blocked while the approved bridge sequence is still active."
            self.logger.warning(
                "[Product] %s product=%s bridge_sequence_id=%s",
                warning,
                self.agent_name,
                str((active_bridge_sequence or {}).get("bridge_sequence_id") or "").strip(),
            )
            return self._recovery_result_with_action_feedback(
                self.get_runtime_recovery(),
                kind="warning",
                text=warning,
            )
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
        session_bridge_mode = self._runtime_bridge_session_mode()
        if status == "bridge_ready" and session_bridge_mode == "pre_ran":
            raise RuntimeError(
                "pre-ran mode requires loading an archived bridge run instead of live bridge generation"
            )

        failed_task_id = str(
            self.runtime_recovery.get("failed_task_id")
            or self._runtime_recovery_context.get("failed_task_id", "")
        ).strip()
        violations = deepcopy(list(self._runtime_recovery_context.get("violations") or []))
        bridge_debug = deepcopy(
            prepared_bridge_request.get("bridge_debug") or self.runtime_recovery.get("bridge_debug") or {}
        )
        fixture_replay_payload = self._compact_fixture_replay_status(
            bridge_debug.get("fixture_replay")
        )
        bridge_debug = self._bridge_debug_with_runtime_handoff(
            bridge_debug,
            handoff_owner="product_agent",
            bridge_mode=session_bridge_mode,
            validation_policy=self._runtime_bridge_session_validation_policy(),
            auto_start_requested=bool(session_bridge_mode == "auto"),
            auto_start_started=bool(session_bridge_mode == "auto"),
            generation_started_at_utc=self._utc_now_iso(),
        )
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        bridge_debug = self._ensure_live_bridge_per_turn_debug_dir(prepared_bridge_request)
        self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(
            prepared_bridge_request
        )
        if hasattr(self.process_planner, "_set_last_bridge_debug"):
            self.process_planner._set_last_bridge_debug(bridge_debug)

        self._set_runtime_recovery(
            status="llm_bridge",
            resolution_class="none",
            trigger=str(self._runtime_recovery_context.get("trigger", "")),
            failed_task_id=failed_task_id,
            message=(
                "Running bounded LLM bridge reasoning to the outline review checkpoint."
                if session_bridge_mode == "manual"
                else "Running live LLM bridge reasoning from the prepared session."
            ),
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=True,
            bridge_proposal=None,
            bridge_debug=bridge_debug if bridge_debug else None,
            bridge_approval_state="generating",
            bridge_stage="none",
            active_bridge_sequence=None,
            bridge_feedback_history=list(self._runtime_recovery_context.get("bridge_feedback_history") or []),
            fixture_replay=fixture_replay_payload,
            violations=violations,
            append_history=True,
            history_message=(
                "Operator started live bridge reasoning to the outline checkpoint."
                if session_bridge_mode == "manual"
                else "Auto-started live runtime bridge generation from the prepared request."
            ),
        )
        await asyncio.to_thread(self._persist_product_state)

        if session_bridge_mode == "manual":
            self.logger.info(
                "[Product] Starting live runtime bridge generation from the prepared request after operator action."
            )
        per_turn_debug_dir = str(bridge_debug.get("per_turn_debug_dir") or "").strip()
        if per_turn_debug_dir:
            self.logger.info(
                "[Product] Live bridge per-turn artifacts will be written under %s",
                per_turn_debug_dir,
            )

        self._runtime_repair_inflight = True
        try:
            if session_bridge_mode == "manual":
                proposal = await self._run_multi_turn_bridge_until(
                    prepared_bridge_request,
                    stop_after="outline",
                )
            else:
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
            if (
                session_bridge_mode == "auto"
                and str(bridge_debug.get("reasoning_mode") or "").strip().lower() == "multi_turn"
            ):
                verification_payload = self._build_generated_code_verification(
                    enabled=verification_enabled,
                    bridge_debug=bridge_debug,
                    prepared_bridge_request=prepared_bridge_request,
                    bridge_proposal=proposal if isinstance(proposal, dict) else None,
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
            bridge_status = str((bridge_debug or {}).get("status") or "").strip().lower()
            if session_bridge_mode == "manual" and bridge_status in {
                "paused_after_outline_turn",
                "ready_for_primitive_generation",
            }:
                recovery = self._set_runtime_recovery(
                    status="llm_bridge",
                    resolution_class="none",
                    trigger=str(self._runtime_recovery_context.get("trigger", "")),
                    failed_task_id=failed_task_id,
                    message="Bridge outline is ready for operator review.",
                    attempts_used=self._runtime_repair_fail_streak,
                    attempts_max=self._runtime_repair_max_attempts,
                    used_llm_bridge=True,
                    bridge_proposal=None,
                    bridge_debug=bridge_debug if bridge_debug else None,
                    bridge_approval_state="outline_pending",
                    bridge_stage="outline",
                    active_bridge_sequence=None,
                    fixture_replay=fixture_replay_payload,
                    generated_code_verification=None,
                    bridge_feedback_history=list(
                        self._runtime_recovery_context.get("bridge_feedback_history") or []
                    ),
                    violations=violations,
                    append_history=True,
                    history_message="Bridge outline paused for operator approval.",
                )
                self._clear_plan_safety_alert()
                await asyncio.to_thread(self._persist_product_state)
                return recovery
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
                    bridge_stage="final",
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
                    bridge_stage="none",
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
                    bridge_stage="none",
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
            if bridge_status in {
                "paused_after_outline_turn",
                "ready_for_primitive_generation",
                "paused_after_primitive_turn",
            }:
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
                    bridge_stage="none",
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
                    bridge_stage="none",
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
                bridge_stage="none",
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
                bridge_stage="none",
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

    async def load_runtime_bridge_archive_proposal(
        self,
        artifact_path: str | None = None,
    ) -> dict[str, Any]:
        if not self._runtime_recovery_context:
            raise RuntimeError("no active runtime DES recovery context is available")
        active_bridge_sequence = self._active_bridge_sequence()
        if self._bridge_sequence_is_live(active_bridge_sequence):
            warning = "Archived bridge loading is blocked while the approved bridge sequence is still active."
            self.logger.warning(
                "[Product] %s product=%s bridge_sequence_id=%s",
                warning,
                self.agent_name,
                str((active_bridge_sequence or {}).get("bridge_sequence_id") or "").strip(),
            )
            return self._recovery_result_with_action_feedback(
                self.get_runtime_recovery(),
                kind="warning",
                text=warning,
            )
        if self._runtime_repair_inflight:
            raise RuntimeError("runtime recovery is already in progress")
        if self._runtime_bridge_session_mode() != "pre_ran":
            raise RuntimeError("archived bridge loading is only available in pre-ran mode")

        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        if status not in {"bridge_ready", "human_required", "llm_bridge"}:
            raise RuntimeError(
                "archived bridge loading is only available from bridge_ready, llm_bridge, or human_required"
            )

        prepared_bridge_request = deepcopy(
            self._runtime_recovery_context.get("prepared_bridge_request") or {}
        )
        if not prepared_bridge_request:
            raise RuntimeError("no prepared bridge request is available")

        final_output_payload, adapter_result, resolved_path = self._load_runtime_bridge_archive_bundle(
            prepared_bridge_request=prepared_bridge_request,
            artifact_path=artifact_path,
        )
        proposal = deepcopy(adapter_result.get("bridge_proposal") or {})
        if not isinstance(proposal, dict) or not proposal:
            raise RuntimeError("archived bridge run proposal build did not produce a proposal")

        validation_policy = self._runtime_bridge_session_validation_policy()
        bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
        bridge_debug["source"] = "archived_final_output"
        bridge_debug["status"] = "archive_replay_ready"
        bridge_debug["message"] = (
            "Loaded archived bridge final output for final operator approval."
        )
        bridge_debug["validation_policy"] = validation_policy
        bridge_debug["archive_replay"] = {
            "source_path": str(resolved_path),
            "selected_label": self._runtime_bridge_session_archive_label(),
            "load_status": "loaded",
            "proposal_build_status": "accepted",
            "validation_policy": validation_policy,
        }
        execution_policy = deepcopy(bridge_debug.get("execution_policy") or {})
        if not bool(execution_policy.get("verification_only")):
            execution_policy["complete_full_tail"] = True
        execution_policy["validation_policy"] = validation_policy
        execution_policy.pop("skip_runtime_plan_validation", None)
        bridge_debug["execution_policy"] = execution_policy
        bridge_debug = self._bridge_debug_with_runtime_handoff(
            bridge_debug,
            bridge_mode=self._runtime_bridge_session_mode(),
            validation_policy=validation_policy,
        )
        bridge_debug["final_output"] = deepcopy(final_output_payload)
        bridge_debug["final_output_adapter"] = deepcopy(adapter_result)
        bridge_debug["bridge_proposal"] = deepcopy(proposal)
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(
            prepared_bridge_request
        )

        archived_recovery_safety_result: dict[str, Any] = {}
        recovery_safety_scope_id = str(
            self.runtime_recovery.get("recovery_safety_scope_id") or ""
        ).strip()
        recovery_safety_status = str(
            self.runtime_recovery.get("recovery_safety_status") or "none"
        ).strip().lower()
        recovery_safety_logic_json = str(
            self.runtime_recovery.get("recovery_safety_logic_json") or ""
        ).strip()
        recovery_safety_dir = str(self.runtime_recovery.get("recovery_safety_dir") or "").strip()
        recovery_plan_dir = str(self.runtime_recovery.get("recovery_plan_dir") or "").strip()
        recovery_safery_dir = str(self.runtime_recovery.get("recovery_safery_dir") or "").strip()
        if validation_policy == "validated":
            archived_recovery_safety_result = self._load_archived_recovery_safety_result(
                resolved_path
            )
            if archived_recovery_safety_result:
                recovery_safety_scope_id = str(
                    archived_recovery_safety_result.get("recovery_safety_scope_id") or ""
                ).strip()
                recovery_safety_status = "ready"
                recovery_safety_logic_json = str(
                    archived_recovery_safety_result.get("recovery_safety_logic_json") or ""
                ).strip()
                recovery_safety_dir = str(
                    archived_recovery_safety_result.get("recovery_safety_dir") or ""
                ).strip()
                recovery_plan_dir = str(
                    archived_recovery_safety_result.get("recovery_plan_dir") or ""
                ).strip()
                recovery_safery_dir = str(
                    archived_recovery_safety_result.get("recovery_safery_dir") or ""
                ).strip()
                self._runtime_recovery_context["recovery_safety_result"] = deepcopy(
                    archived_recovery_safety_result
                )
                bridge_debug["archived_recovery_safety"] = {
                    "status": "loaded",
                    "recovery_safety_scope_id": recovery_safety_scope_id,
                    "recovery_safety_status": recovery_safety_status,
                    "recovery_safety_logic_json": recovery_safety_logic_json,
                    "result_path": str(
                        archived_recovery_safety_result.get(
                            "archived_recovery_safety_result_path"
                        )
                        or ""
                    ).strip(),
                }
                prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
                self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(
                    prepared_bridge_request
                )
            else:
                recovery_safety_scope_id = (
                    recovery_safety_scope_id
                    or f"recovery_scope_{uuid.uuid4().hex[:8]}"
                )
                recovery_safety_payload = self._dispatch_recovery_safety_generation_request(
                    prepared_bridge_request=prepared_bridge_request,
                    recovery_safety_scope_id=recovery_safety_scope_id,
                )
                recovery_safety_status = "generating"
                recovery_safety_logic_json = ""
                recovery_safety_dir = str(
                    recovery_safety_payload.get("recovery_safety_dir") or ""
                ).strip()
                recovery_plan_dir = str(
                    recovery_safety_payload.get("recovery_plan_dir") or ""
                ).strip()
                recovery_safery_dir = str(
                    recovery_safety_payload.get("recovery_safery_dir") or ""
                ).strip()
                bridge_debug["archived_recovery_safety"] = {
                    "status": "generating",
                    "recovery_safety_scope_id": recovery_safety_scope_id,
                    "recovery_safety_status": recovery_safety_status,
                }
                prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
                self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(
                    prepared_bridge_request
                )
        self.logger.info(
            "[Product] Archived bridge replay loaded: archive=%s validation_policy=%s",
            resolved_path,
            validation_policy,
        )
        if hasattr(self.process_planner, "_set_last_bridge_debug"):
            self.process_planner._set_last_bridge_debug(bridge_debug)
        self._record_runtime_bridge_artifacts(
            phase="multi_turn",
            prepared_bridge_request=prepared_bridge_request,
        )
        self._runtime_recovery_context["prepared_bridge_request"] = deepcopy(
            prepared_bridge_request
        )

        failed_task_id = str(
            self.runtime_recovery.get("failed_task_id")
            or self._runtime_recovery_context.get("failed_task_id", "")
        ).strip()
        violations = deepcopy(list(self._runtime_recovery_context.get("violations") or []))
        bridge_summary = self.process_planner._bridge_summary(proposal)
        bridge_text = ", ".join(str(item) for item in bridge_summary if item) or "bridge step(s)"
        recovery_message = (
            "Archived bridge proposal loaded and ready for final approval. Recovery Safety Check is enabled for recovery_safety runtime enforcement."
            if validation_policy == "validated" and recovery_safety_status == "ready"
            else "Archived bridge proposal loaded; Recovery Safety Check is generating recovery_safety runtime enforcement before final approval."
            if validation_policy == "validated"
            else "Archived bridge proposal loaded and ready for final approval. No Recovery Safety Check is enabled for recovery_safety runtime enforcement."
        )
        recovery = self._set_runtime_recovery(
            status="llm_bridge",
            resolution_class="none",
            trigger=str(self._runtime_recovery_context.get("trigger", "")),
            failed_task_id=failed_task_id,
            message=recovery_message,
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=True,
            bridge_proposal=proposal,
            bridge_debug=bridge_debug if bridge_debug else None,
            bridge_approval_state="pending",
            bridge_stage="final",
            active_bridge_sequence=None,
            recovery_safety_scope_id=recovery_safety_scope_id,
            recovery_safety_status=recovery_safety_status,
            recovery_safety_logic_json=recovery_safety_logic_json,
            recovery_safety_dir=recovery_safety_dir,
            recovery_plan_dir=recovery_plan_dir,
            recovery_safery_dir=recovery_safery_dir,
            fixture_replay=None,
            generated_code_verification=None,
            bridge_feedback_history=list(
                self._runtime_recovery_context.get("bridge_feedback_history") or []
            ),
            violations=violations,
            append_history=True,
            history_message=(
                f"Loaded archived bridge proposal: {bridge_text} (recovery_safety mode: {validation_policy})."
            ),
        )
        self._clear_plan_safety_alert()
        await asyncio.to_thread(self._persist_product_state)
        return recovery

    async def approve_runtime_bridge_outline(self) -> dict[str, Any]:
        if not self._runtime_recovery_context:
            raise RuntimeError("no active runtime DES recovery context is available")
        if self._runtime_repair_inflight:
            raise RuntimeError("runtime recovery is already in progress")

        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        approval_state = str(
            self.runtime_recovery.get("bridge_approval_state", "none") or "none"
        ).strip().lower()
        if status != "llm_bridge" or approval_state != "outline_pending":
            raise RuntimeError("no outline checkpoint is awaiting approval")

        failed_task_id = str(
            self.runtime_recovery.get("failed_task_id")
            or self._runtime_recovery_context.get("failed_task_id", "")
        ).strip()
        violations = deepcopy(list(self._runtime_recovery_context.get("violations") or []))
        prepared_bridge_request = deepcopy(
            self._runtime_recovery_context.get("prepared_bridge_request") or {}
        )
        if not prepared_bridge_request:
            raise RuntimeError("no prepared bridge request is available")
        recovery_safety_scope_id = f"recovery_scope_{uuid.uuid4().hex[:8]}"

        self._set_runtime_recovery(
            status="llm_bridge",
            resolution_class="none",
            trigger=str(self._runtime_recovery_context.get("trigger", "")),
            failed_task_id=failed_task_id,
            message="Outline approved. Continuing live bridge reasoning to the primitive review checkpoint.",
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=True,
            bridge_proposal=None,
            bridge_debug=self.runtime_recovery.get("bridge_debug"),
            bridge_approval_state="generating",
            bridge_stage="outline",
            active_bridge_sequence=None,
            recovery_safety_scope_id=recovery_safety_scope_id,
            recovery_safety_status="generating",
            recovery_safety_logic_json="",
            recovery_safety_dir="",
            recovery_plan_dir="",
            recovery_safery_dir="",
            recovery_final_dir="",
            recovery_final_output_path="",
            recovery_enforced_task_ids=[],
            fixture_replay=self.runtime_recovery.get("fixture_replay"),
            generated_code_verification=None,
            bridge_feedback_history=list(
                self._runtime_recovery_context.get("bridge_feedback_history") or []
            ),
            violations=violations,
            append_history=True,
            history_message="Operator approved the bridge outline.",
        )
        await asyncio.to_thread(self._persist_product_state)

        try:
            recovery_safety_payload = self._dispatch_recovery_safety_generation_request(
                prepared_bridge_request=prepared_bridge_request,
                recovery_safety_scope_id=recovery_safety_scope_id,
            )
        except Exception as exc:
            self.logger.exception("[Product] Outline approval recovery safety generation dispatch failed.")
            message = (
                f"{self.agent_name}: outline approval could not start recovery safety generation "
                f"({exc})."
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
                bridge_debug=self.runtime_recovery.get("bridge_debug"),
                bridge_approval_state="none",
                bridge_stage="none",
                active_bridge_sequence=None,
                recovery_safety_scope_id=recovery_safety_scope_id,
                recovery_safety_status="failed",
                recovery_safety_logic_json="",
                recovery_safety_dir="",
                recovery_plan_dir="",
                recovery_safery_dir="",
                recovery_final_dir="",
                recovery_final_output_path="",
                recovery_enforced_task_ids=[],
                fixture_replay=self.runtime_recovery.get("fixture_replay"),
                generated_code_verification=None,
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

        self._set_runtime_recovery(
            recovery_safety_scope_id=recovery_safety_scope_id,
            recovery_safety_status="generating",
            recovery_safety_logic_json="",
            recovery_safety_dir=str(recovery_safety_payload.get("recovery_safety_dir") or "").strip(),
            recovery_plan_dir=str(recovery_safety_payload.get("recovery_plan_dir") or "").strip(),
            recovery_safery_dir=str(recovery_safety_payload.get("recovery_safery_dir") or "").strip(),
            recovery_final_dir="",
            recovery_final_output_path="",
            recovery_enforced_task_ids=[],
        )
        await asyncio.to_thread(self._persist_product_state)

        self._runtime_repair_inflight = True
        try:
            proposal = await self._run_multi_turn_bridge_until(
                prepared_bridge_request,
                stop_after="primitive",
            )
            bridge_debug = deepcopy(
                prepared_bridge_request.get("bridge_debug")
                or self.process_planner.get_last_bridge_debug()
                or {}
            )
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
            self._maybe_write_recovery_final_bundle(
                prepared_bridge_request=prepared_bridge_request,
            )
            bridge_status = str(bridge_debug.get("status") or "").strip().lower()
            if bridge_status in {
                "paused_after_primitive_generation",
                "paused_after_primitive_turn",
            }:
                recovery = self._set_runtime_recovery(
                    status="llm_bridge",
                    resolution_class="none",
                    trigger=str(self._runtime_recovery_context.get("trigger", "")),
                    failed_task_id=failed_task_id,
                    message="Bridge primitives are ready for operator review.",
                    attempts_used=self._runtime_repair_fail_streak,
                    attempts_max=self._runtime_repair_max_attempts,
                    used_llm_bridge=True,
                    bridge_proposal=None,
                    bridge_debug=bridge_debug if bridge_debug else None,
                    bridge_approval_state="primitive_pending",
                    bridge_stage="primitive",
                    active_bridge_sequence=None,
                    fixture_replay=None,
                    generated_code_verification=None,
                    bridge_feedback_history=list(
                        self._runtime_recovery_context.get("bridge_feedback_history") or []
                    ),
                    violations=violations,
                    append_history=True,
                    history_message="Bridge primitive generation paused for operator approval.",
                )
                self._clear_plan_safety_alert()
                await asyncio.to_thread(self._persist_product_state)
                return recovery

            if isinstance(proposal, dict):
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
                    bridge_stage="final",
                    active_bridge_sequence=None,
                    fixture_replay=None,
                    generated_code_verification=None,
                    bridge_feedback_history=list(
                        self._runtime_recovery_context.get("bridge_feedback_history") or []
                    ),
                    violations=violations,
                    append_history=True,
                    history_message="Bridge primitive generation completed and produced a final proposal.",
                )
                self._clear_plan_safety_alert()
                await asyncio.to_thread(self._persist_product_state)
                return recovery

            message = (
                "LLM bridge primitive generation did not produce a reviewable primitive program."
                if bridge_status not in {"paused_after_primitive_blocked", "paused_after_primitive_stuck"}
                else "LLM bridge primitive generation stalled before producing a reviewable primitive program."
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
                bridge_stage="none",
                active_bridge_sequence=None,
                fixture_replay=None,
                generated_code_verification=None,
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
            self.logger.exception("[Product] Outline approval bridge continuation failed.")
            bridge_debug = self.process_planner.get_last_bridge_debug()
            message = f"{self.agent_name}: outline approval failed while continuing bridge reasoning ({exc})."
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
                bridge_stage="none",
                active_bridge_sequence=None,
                fixture_replay=None,
                generated_code_verification=None,
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

    async def refine_runtime_bridge_outline(self, feedback: str) -> dict[str, Any]:
        if not self._runtime_recovery_context:
            raise RuntimeError("no active runtime DES recovery context is available")
        if self._runtime_repair_inflight:
            raise RuntimeError("runtime recovery is already in progress")

        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        approval_state = str(
            self.runtime_recovery.get("bridge_approval_state", "none") or "none"
        ).strip().lower()
        if status != "llm_bridge" or approval_state != "outline_pending":
            raise RuntimeError("no outline checkpoint is awaiting refinement")

        feedback_history = self._append_bridge_feedback(feedback)
        failed_task_id = str(
            self.runtime_recovery.get("failed_task_id")
            or self._runtime_recovery_context.get("failed_task_id", "")
        ).strip()
        violations = deepcopy(list(self._runtime_recovery_context.get("violations") or []))
        prepared_bridge_request = self._prepare_runtime_bridge_session_state(stage="outline")

        self._set_runtime_recovery(
            status="llm_bridge",
            resolution_class="none",
            trigger=str(self._runtime_recovery_context.get("trigger", "")),
            failed_task_id=failed_task_id,
            message="Outline guidance recorded. Re-running live bridge reasoning to regenerate the outline checkpoint.",
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=True,
            bridge_proposal=None,
            bridge_debug=prepared_bridge_request.get("bridge_debug"),
            bridge_approval_state="generating",
            bridge_stage="outline",
            active_bridge_sequence=None,
            fixture_replay=None,
            generated_code_verification=None,
            bridge_feedback_history=feedback_history,
            violations=violations,
            append_history=True,
            history_message=f"Operator refined the bridge outline: {str(feedback).strip()}",
        )
        await asyncio.to_thread(self._persist_product_state)

        self._runtime_repair_inflight = True
        try:
            proposal = await self._run_multi_turn_bridge_until(
                prepared_bridge_request,
                stop_after="outline",
            )
            bridge_debug = deepcopy(
                prepared_bridge_request.get("bridge_debug")
                or self.process_planner.get_last_bridge_debug()
                or {}
            )
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
            bridge_status = str(bridge_debug.get("status") or "").strip().lower()
            if bridge_status in {"paused_after_outline_turn", "ready_for_primitive_generation"}:
                recovery = self._set_runtime_recovery(
                    status="llm_bridge",
                    resolution_class="none",
                    trigger=str(self._runtime_recovery_context.get("trigger", "")),
                    failed_task_id=failed_task_id,
                    message="Bridge outline is ready for operator review.",
                    attempts_used=self._runtime_repair_fail_streak,
                    attempts_max=self._runtime_repair_max_attempts,
                    used_llm_bridge=True,
                    bridge_proposal=None,
                    bridge_debug=bridge_debug if bridge_debug else None,
                    bridge_approval_state="outline_pending",
                    bridge_stage="outline",
                    active_bridge_sequence=None,
                    fixture_replay=None,
                    generated_code_verification=None,
                    bridge_feedback_history=feedback_history,
                    violations=violations,
                    append_history=True,
                    history_message="Bridge outline regenerated for operator approval.",
                )
                self._clear_plan_safety_alert()
                await asyncio.to_thread(self._persist_product_state)
                return recovery

            if isinstance(proposal, dict):
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
                    bridge_stage="final",
                    active_bridge_sequence=None,
                    fixture_replay=None,
                    generated_code_verification=None,
                    bridge_feedback_history=feedback_history,
                    violations=violations,
                    append_history=True,
                    history_message="Bridge refinement produced a final proposal.",
                )
                self._clear_plan_safety_alert()
                await asyncio.to_thread(self._persist_product_state)
                return recovery

            message = "LLM bridge outline regeneration did not produce a reviewable outline."
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
                bridge_stage="none",
                active_bridge_sequence=None,
                fixture_replay=None,
                generated_code_verification=None,
                bridge_feedback_history=feedback_history,
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
            self.logger.exception("[Product] Outline refinement bridge continuation failed.")
            bridge_debug = self.process_planner.get_last_bridge_debug()
            message = f"{self.agent_name}: outline refinement failed while continuing bridge reasoning ({exc})."
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
                bridge_stage="none",
                active_bridge_sequence=None,
                fixture_replay=None,
                generated_code_verification=None,
                bridge_feedback_history=feedback_history,
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

    async def reject_runtime_bridge_outline(self, feedback: str) -> dict[str, Any]:
        if not self._runtime_recovery_context:
            raise RuntimeError("no active runtime DES recovery context is available")
        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        approval_state = str(
            self.runtime_recovery.get("bridge_approval_state", "none") or "none"
        ).strip().lower()
        if status != "llm_bridge" or approval_state != "outline_pending":
            raise RuntimeError("no outline checkpoint is awaiting rejection")

        feedback_text = str(feedback or "").strip()
        if not feedback_text:
            raise ValueError("outline rejection feedback is empty")
        feedback_history = self._append_bridge_feedback(feedback_text)
        violations = deepcopy(list(self._runtime_recovery_context.get("violations") or []))
        recovery = self._set_runtime_recovery(
            status="human_required",
            resolution_class="human_required",
            trigger=str(self._runtime_recovery_context.get("trigger", "")),
            failed_task_id=str(self._runtime_recovery_context.get("failed_task_id", "")),
            message="Operator rejected the bridge outline. Runtime recovery is paused for manual intervention.",
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=True,
            bridge_proposal=None,
            bridge_debug=(
                self.process_planner.get_last_bridge_debug()
                or self.runtime_recovery.get("bridge_debug")
            ),
            bridge_approval_state="none",
            bridge_stage="none",
            active_bridge_sequence=None,
            fixture_replay=None,
            generated_code_verification=None,
            violations=violations,
            bridge_feedback_history=feedback_history,
            append_history=True,
            history_message=f"Operator rejected bridge outline: {feedback_text}",
        )
        self._set_plan_safety_alert(
            stage="runtime",
            message=str(recovery.get("message", "") or "Bridge outline rejected."),
            retries_used=self._runtime_repair_fail_streak,
            retries_max=self._runtime_repair_max_attempts,
            violations=violations,
            paused=True,
        )
        await asyncio.to_thread(self._persist_product_state)
        return recovery

    async def approve_runtime_bridge_primitives(self) -> dict[str, Any]:
        if not self._runtime_recovery_context:
            raise RuntimeError("no active runtime DES recovery context is available")

        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        approval_state = str(
            self.runtime_recovery.get("bridge_approval_state", "none") or "none"
        ).strip().lower()
        if status != "llm_bridge" or approval_state != "primitive_pending":
            raise RuntimeError("no primitive checkpoint is awaiting approval")

        prepared_bridge_request = deepcopy(
            self._runtime_recovery_context.get("prepared_bridge_request") or {}
        )
        if not prepared_bridge_request:
            raise RuntimeError("no prepared bridge request is available")

        proposal = self._bridge_proposal_from_debug(prepared_bridge_request)
        if not isinstance(proposal, dict):
            raise RuntimeError(
                "completed primitive bridge output could not be built into a final proposal"
            )
        bridge_debug = deepcopy(
            prepared_bridge_request.get("bridge_debug")
            or self.process_planner.get_last_bridge_debug()
            or {}
        )
        bridge_debug["bridge_proposal"] = deepcopy(proposal)
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
        self._maybe_write_recovery_final_bundle(
            prepared_bridge_request=prepared_bridge_request,
        )

        failed_task_id = str(
            self.runtime_recovery.get("failed_task_id")
            or self._runtime_recovery_context.get("failed_task_id", "")
        ).strip()
        violations = deepcopy(list(self._runtime_recovery_context.get("violations") or []))
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
            bridge_stage="final",
            active_bridge_sequence=None,
            fixture_replay=None,
            generated_code_verification=None,
            bridge_feedback_history=list(
                self._runtime_recovery_context.get("bridge_feedback_history") or []
            ),
            violations=violations,
            append_history=True,
            history_message=f"Operator approved bridge primitives: {bridge_text}.",
        )
        self._clear_plan_safety_alert()
        await asyncio.to_thread(self._persist_product_state)
        return recovery

    async def refine_runtime_bridge_primitives(self, feedback: str) -> dict[str, Any]:
        if not self._runtime_recovery_context:
            raise RuntimeError("no active runtime DES recovery context is available")
        if self._runtime_repair_inflight:
            raise RuntimeError("runtime recovery is already in progress")

        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        approval_state = str(
            self.runtime_recovery.get("bridge_approval_state", "none") or "none"
        ).strip().lower()
        if status != "llm_bridge" or approval_state != "primitive_pending":
            raise RuntimeError("no primitive checkpoint is awaiting refinement")

        feedback_history = self._append_bridge_feedback(feedback)
        failed_task_id = str(
            self.runtime_recovery.get("failed_task_id")
            or self._runtime_recovery_context.get("failed_task_id", "")
        ).strip()
        violations = deepcopy(list(self._runtime_recovery_context.get("violations") or []))
        prepared_bridge_request = self._prepare_runtime_bridge_session_state(stage="primitive")

        self._set_runtime_recovery(
            status="llm_bridge",
            resolution_class="none",
            trigger=str(self._runtime_recovery_context.get("trigger", "")),
            failed_task_id=failed_task_id,
            message="Primitive guidance recorded. Re-running live bridge primitive generation.",
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=True,
            bridge_proposal=None,
            bridge_debug=prepared_bridge_request.get("bridge_debug"),
            bridge_approval_state="generating",
            bridge_stage="primitive",
            active_bridge_sequence=None,
            fixture_replay=None,
            generated_code_verification=None,
            bridge_feedback_history=feedback_history,
            violations=violations,
            append_history=True,
            history_message=f"Operator refined bridge primitives: {str(feedback).strip()}",
        )
        await asyncio.to_thread(self._persist_product_state)

        self._runtime_repair_inflight = True
        try:
            proposal = await self._run_multi_turn_bridge_until(
                prepared_bridge_request,
                stop_after="primitive",
            )
            bridge_debug = deepcopy(
                prepared_bridge_request.get("bridge_debug")
                or self.process_planner.get_last_bridge_debug()
                or {}
            )
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
            self._maybe_write_recovery_final_bundle(
                prepared_bridge_request=prepared_bridge_request,
            )
            bridge_status = str(bridge_debug.get("status") or "").strip().lower()
            if bridge_status in {
                "paused_after_primitive_generation",
                "paused_after_primitive_turn",
            }:
                recovery = self._set_runtime_recovery(
                    status="llm_bridge",
                    resolution_class="none",
                    trigger=str(self._runtime_recovery_context.get("trigger", "")),
                    failed_task_id=failed_task_id,
                    message="Bridge primitives are ready for operator review.",
                    attempts_used=self._runtime_repair_fail_streak,
                    attempts_max=self._runtime_repair_max_attempts,
                    used_llm_bridge=True,
                    bridge_proposal=None,
                    bridge_debug=bridge_debug if bridge_debug else None,
                    bridge_approval_state="primitive_pending",
                    bridge_stage="primitive",
                    active_bridge_sequence=None,
                    fixture_replay=None,
                    generated_code_verification=None,
                    bridge_feedback_history=feedback_history,
                    violations=violations,
                    append_history=True,
                    history_message="Bridge primitive generation regenerated for operator approval.",
                )
                self._clear_plan_safety_alert()
                await asyncio.to_thread(self._persist_product_state)
                return recovery

            if isinstance(proposal, dict):
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
                    bridge_stage="final",
                    active_bridge_sequence=None,
                    fixture_replay=None,
                    generated_code_verification=None,
                    bridge_feedback_history=feedback_history,
                    violations=violations,
                    append_history=True,
                    history_message="Bridge primitive refinement produced a final proposal.",
                )
                self._clear_plan_safety_alert()
                await asyncio.to_thread(self._persist_product_state)
                return recovery

            message = (
                "LLM bridge primitive regeneration did not produce a reviewable primitive program."
                if bridge_status not in {"paused_after_primitive_blocked", "paused_after_primitive_stuck"}
                else "LLM bridge primitive generation stalled before producing a reviewable primitive program."
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
                bridge_stage="none",
                active_bridge_sequence=None,
                fixture_replay=None,
                generated_code_verification=None,
                bridge_feedback_history=feedback_history,
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
            self.logger.exception("[Product] Primitive refinement bridge continuation failed.")
            bridge_debug = self.process_planner.get_last_bridge_debug()
            message = f"{self.agent_name}: primitive refinement failed while continuing bridge reasoning ({exc})."
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
                bridge_stage="none",
                active_bridge_sequence=None,
                fixture_replay=None,
                generated_code_verification=None,
                bridge_feedback_history=feedback_history,
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

    async def reject_runtime_bridge_primitives(self, feedback: str) -> dict[str, Any]:
        if not self._runtime_recovery_context:
            raise RuntimeError("no active runtime DES recovery context is available")
        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        approval_state = str(
            self.runtime_recovery.get("bridge_approval_state", "none") or "none"
        ).strip().lower()
        if status != "llm_bridge" or approval_state != "primitive_pending":
            raise RuntimeError("no primitive checkpoint is awaiting rejection")

        feedback_text = str(feedback or "").strip()
        if not feedback_text:
            raise ValueError("primitive rejection feedback is empty")
        feedback_history = self._append_bridge_feedback(feedback_text)
        violations = deepcopy(list(self._runtime_recovery_context.get("violations") or []))
        recovery = self._set_runtime_recovery(
            status="human_required",
            resolution_class="human_required",
            trigger=str(self._runtime_recovery_context.get("trigger", "")),
            failed_task_id=str(self._runtime_recovery_context.get("failed_task_id", "")),
            message="Operator rejected the bridge primitive program. Runtime recovery is paused for manual intervention.",
            attempts_used=self._runtime_repair_fail_streak,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=True,
            bridge_proposal=None,
            bridge_debug=(
                self.process_planner.get_last_bridge_debug()
                or self.runtime_recovery.get("bridge_debug")
            ),
            bridge_approval_state="none",
            bridge_stage="none",
            active_bridge_sequence=None,
            fixture_replay=None,
            generated_code_verification=None,
            violations=violations,
            bridge_feedback_history=feedback_history,
            append_history=True,
            history_message=f"Operator rejected bridge primitives: {feedback_text}",
        )
        self._set_plan_safety_alert(
            stage="runtime",
            message=str(recovery.get("message", "") or "Bridge primitive program rejected."),
            retries_used=self._runtime_repair_fail_streak,
            retries_max=self._runtime_repair_max_attempts,
            violations=violations,
            paused=True,
        )
        await asyncio.to_thread(self._persist_product_state)
        return recovery

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
        bridge_summary = self.process_planner._bridge_summary(normalized)
        return {
            "proposal": normalized,
            "bridge_debug": bridge_debug,
            "bridge_summary": bridge_summary,
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
    ) -> tuple[str, dict[str, Any], dict[str, Any], list[str]]:
        scenario_key = str(scenario_id or "").strip()
        if not scenario_key:
            raise ValueError("scenario_id is empty")

        elapsed = (
            time.perf_counter() - started_at
            if started_at is not None
            else 0.0
        )
        cached_bundle = deepcopy(
            (self._runtime_recovery_context.get("preprogrammed_bridge_cache") or {}).get(scenario_key) or {}
        )
        if isinstance(cached_bundle.get("proposal"), dict):
            normalized = deepcopy(cached_bundle["proposal"])
            bridge_debug = deepcopy(cached_bundle.get("bridge_debug") or {})
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
        return scenario_key, normalized, bridge_debug, bridge_summary

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
            scenario_key, normalized, bridge_debug, bridge_summary = (
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
        self._set_runtime_recovery(
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
        proposal = self.runtime_recovery.get("bridge_proposal")
        bridge_debug = deepcopy(self.runtime_recovery.get("bridge_debug") or {})
        active_bridge_sequence = self._active_bridge_sequence()
        recovery_safety_scope_id = str(
            self.runtime_recovery.get("recovery_safety_scope_id") or ""
        ).strip()
        recovery_safety_status = str(
            self.runtime_recovery.get("recovery_safety_status") or "none"
        ).strip().lower()
        if status != "llm_bridge":
            if self._bridge_sequence_is_live(active_bridge_sequence):
                warning = "Bridge already approved/executing for this recovery session."
                return self._recovery_result_with_action_feedback(
                    self.get_runtime_recovery(),
                    kind="warning",
                    text=warning,
                )
            raise RuntimeError("no pending bridge proposal is awaiting approval")
        if not isinstance(proposal, dict):
            raise RuntimeError("bridge proposal is missing")
        if (
            self._runtime_bridge_session_validation_policy() == "validated"
            and not recovery_safety_scope_id
        ):
            warning = (
                "Recovery Safety Check is enabled, but no recovery_safety_scope_id "
                "is ready for this bridge proposal."
            )
            return self._recovery_result_with_action_feedback(
                self.get_runtime_recovery(),
                kind="warning",
                text=warning,
            )
        if recovery_safety_scope_id and recovery_safety_status != "ready":
            if recovery_safety_status == "failed":
                failure_reason = str(
                    dict(self._runtime_recovery_context.get("recovery_safety_result") or {}).get(
                        "failure_reason"
                    )
                    or "unknown failure"
                ).strip()
                message = (
                    f"{self.agent_name}: recovery safety generation failed for scope "
                    f"{recovery_safety_scope_id} ({failure_reason})."
                )
                recovery = self._set_runtime_recovery(
                    status="human_required",
                    resolution_class="human_required",
                    trigger=str(self._runtime_recovery_context.get("trigger", "")),
                    failed_task_id=str(
                        self.runtime_recovery.get("failed_task_id")
                        or self._runtime_recovery_context.get("failed_task_id", "")
                    ),
                    message=message,
                    attempts_used=self._runtime_repair_fail_streak,
                    attempts_max=self._runtime_repair_max_attempts,
                    used_llm_bridge=bool(self.runtime_recovery.get("used_llm_bridge", False)),
                    bridge_proposal=proposal,
                    bridge_debug=bridge_debug if bridge_debug else None,
                    bridge_approval_state="primitive_pending",
                    bridge_stage="primitive",
                    active_bridge_sequence=None,
                    recovery_safety_scope_id=recovery_safety_scope_id,
                    recovery_safety_status="failed",
                    recovery_safety_logic_json=str(
                        self.runtime_recovery.get("recovery_safety_logic_json") or ""
                    ).strip(),
                    recovery_safety_dir=str(
                        self.runtime_recovery.get("recovery_safety_dir") or ""
                    ).strip(),
                    recovery_plan_dir=str(
                        self.runtime_recovery.get("recovery_plan_dir") or ""
                    ).strip(),
                    recovery_safery_dir=str(
                        self.runtime_recovery.get("recovery_safery_dir") or ""
                    ).strip(),
                    recovery_enforced_task_ids=list(
                        self.runtime_recovery.get("recovery_enforced_task_ids") or []
                    ),
                    generated_code_verification=deepcopy(
                        self.runtime_recovery.get("generated_code_verification") or {}
                    ) or None,
                    violations=list(self._runtime_recovery_context.get("violations") or []),
                    append_history=True,
                    history_message=message,
                )
                self._set_plan_safety_alert(
                    stage="runtime",
                    message=message,
                    retries_used=self._runtime_repair_fail_streak,
                    retries_max=self._runtime_repair_max_attempts,
                    violations=list(self._runtime_recovery_context.get("violations") or []),
                    paused=True,
                )
                threading.Thread(
                    target=self._persist_product_state,
                    name=f"{self.agent_name}-persist-product-state",
                    daemon=True,
                ).start()
                return recovery
            warning = (
                f"Recovery safety generation for scope {recovery_safety_scope_id} is still "
                f"{recovery_safety_status or 'pending'}."
            )
            return self._recovery_result_with_action_feedback(
                self.get_runtime_recovery(),
                kind="warning",
                text=warning,
            )
        used_llm_bridge = bool(self.runtime_recovery.get("used_llm_bridge", False))
        generated_code_verification = deepcopy(
            self.runtime_recovery.get("generated_code_verification") or {}
        )

        failed_task_id = str(
            self.runtime_recovery.get("failed_task_id")
            or self._runtime_recovery_context.get("failed_task_id", "")
        ).strip()
        proposal_fingerprint = self._bridge_proposal_fingerprint(
            proposal=proposal,
            failed_task_id=failed_task_id,
            bridge_debug=bridge_debug if isinstance(bridge_debug, dict) else None,
        )
        if self._bridge_sequence_is_live(active_bridge_sequence):
            existing_fingerprint = str(
                (active_bridge_sequence or {}).get("proposal_fingerprint") or ""
            ).strip()
            if existing_fingerprint and existing_fingerprint == proposal_fingerprint:
                warning = "Bridge already approved/executing for this recovery session."
                self.logger.warning(
                    "[Product] Duplicate bridge approval ignored: product=%s failed_task_id=%s fingerprint=%s",
                    self.agent_name,
                    failed_task_id,
                    proposal_fingerprint[:12],
                )
                return self._recovery_result_with_action_feedback(
                    self.get_runtime_recovery(),
                    kind="warning",
                    text=warning,
                )
            warning = (
                "Another bridge sequence is already active for this recovery session. "
                "Reset or retry DES instead of stacking bridge branches."
            )
            self.logger.warning(
                "[Product] Conflicting bridge approval blocked: product=%s failed_task_id=%s active_bridge_sequence_id=%s fingerprint=%s",
                self.agent_name,
                failed_task_id,
                str((active_bridge_sequence or {}).get("bridge_sequence_id") or "").strip(),
                proposal_fingerprint[:12],
            )
            return self._recovery_result_with_action_feedback(
                self.get_runtime_recovery(),
                kind="warning",
                text=warning,
            )
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
        session_validation_policy = self._runtime_bridge_session_validation_policy()
        verification_only_approval = bool(execution_policy.get("verification_only"))
        if not verification_only_approval:
            execution_policy["complete_full_tail"] = True
            if isinstance(bridge_debug, dict):
                bridge_debug["execution_policy"] = deepcopy(execution_policy)
        anchor_task_id = failed_task_id
        if failed_task_id:
            direct_predecessors = self._direct_predecessors_from_nodes(
                planner_nodes_snapshot,
                failed_task_id,
            )
            if direct_predecessors:
                anchor_task_id = direct_predecessors[0]
            else:
                anchor_task_id = ""

        deleted_task_ids: list[str] = []
        if failed_task_id and not verification_only_approval:
            deleted_task_ids = [failed_task_id]

        prepared_bridge_request = dict(
            self._runtime_recovery_context.get("prepared_bridge_request") or {}
        )
        modeled_gap = dict(
            dict(prepared_bridge_request.get("context_summary") or {}).get(
                "modeled_continuation_gap"
            )
            or {}
        )
        resumable_task_ids = []
        resumable_task_ids_by_resource: dict[str, list[str]] = {}
        splice_before_task_ids_by_resource: dict[str, str] = {}
        if not verification_only_approval:
            resumable_nodes_by_resource: dict[str, list[dict[str, Any]]] = {}
            for node in planner_nodes_snapshot:
                if not isinstance(node, dict) or node.get("type") != "task":
                    continue
                task_id = str(node.get("id", "")).strip()
                if (
                    not task_id
                    or task_id == failed_task_id
                    or task_id in deleted_task_ids
                    or str(node.get("status", "")).strip().lower() not in {"pending", "blocked"}
                ):
                    continue
                resumable_task_ids.append(task_id)
                resource_jid = str(node.get("resource_jid", "")).strip()
                if resource_jid:
                    resumable_task_ids_by_resource.setdefault(resource_jid, []).append(task_id)
                    resumable_nodes_by_resource.setdefault(resource_jid, []).append(node)
            for resource_jid, nodes in resumable_nodes_by_resource.items():
                earliest_node = min(
                    nodes,
                    key=lambda node: (
                        self.process_planner._task_sequence_index_key(
                            node.get("sequence_index")
                        ),
                        str(node.get("id", "")).strip(),
                    ),
                )
                earliest_task_id = str(earliest_node.get("id", "")).strip()
                if earliest_task_id:
                    splice_before_task_ids_by_resource[resource_jid] = earliest_task_id
        self.logger.info(
            "[Product] Bridge approval plan: anchor=%s verification_only=%s delete=%s resume=%s splice=%s",
            anchor_task_id or "<none>",
            verification_only_approval,
            deleted_task_ids,
            resumable_task_ids,
            splice_before_task_ids_by_resource,
        )
        try:
            staged_nodes = deepcopy(planner_nodes_snapshot)
            all_patch_rows: list[dict[str, Any]] = []

            def _stage_approval_patch(patch_rows: list[dict[str, Any]]) -> None:
                nonlocal staged_nodes
                rows = [deepcopy(row) for row in patch_rows if isinstance(row, dict)]
                if not rows:
                    return
                all_patch_rows.extend(deepcopy(rows))
                staged_nodes = self.process_planner._preview_replan_patch(
                    staged_nodes,
                    rows,
                )
                self.process_planner.nodes = deepcopy(staged_nodes)
                self.process_planner.global_fsa = None

            def _refresh_staged_tasks(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
                refreshed: list[dict[str, Any]] = []
                for task in task_rows:
                    task_id = str(dict(task or {}).get("id", "")).strip()
                    if not task_id:
                        continue
                    node = self.process_planner._find_node(task_id)
                    refreshed.append(deepcopy(node if isinstance(node, dict) else task))
                return refreshed

            self.process_planner.nodes = deepcopy(staged_nodes)
            self.process_planner.global_fsa = None
            tasks, bridge_patch_rows = self.process_planner.build_bridge_macro_proposal_patch(
                proposal,
                anchor_task_id=anchor_task_id,
                splice_before_task_ids_by_resource=splice_before_task_ids_by_resource,
            )
            _stage_approval_patch(bridge_patch_rows)
            tasks = _refresh_staged_tasks(tasks)
            tail_task_ids_by_resource: dict[str, str] = {}
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                task_id = str(task.get("id", "")).strip()
                resource_jid = str(task.get("resource_jid", "")).strip()
                if task_id and resource_jid:
                    tail_task_ids_by_resource[resource_jid] = task_id
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
                    tail_task_ids_by_resource=tail_task_ids_by_resource,
                    before_task_ids=resumable_task_ids,
                    deleted_task_ids=deleted_task_ids,
                    change_prefix="Approved bridge recovery",
                    recovery_tasks=tasks,
                )
            if post_updates:
                _stage_approval_patch(post_updates)
                tasks = _refresh_staged_tasks(tasks)
            if tasks and splice_before_task_ids_by_resource:
                prune_patch_rows: list[dict[str, Any]] = []
                self._prune_redundant_bridge_resume_move_home_if_satisfied(
                    tasks=tasks,
                    resumable_task_ids=resumable_task_ids,
                    resumable_task_ids_by_resource=resumable_task_ids_by_resource,
                    resume_entry_task_ids_by_resource=splice_before_task_ids_by_resource,
                    deleted_task_ids=deleted_task_ids,
                    patch_rows=prune_patch_rows,
                )
                if prune_patch_rows:
                    _stage_approval_patch(prune_patch_rows)
                    tasks = _refresh_staged_tasks(tasks)
                acquire_entity_patch_rows: list[dict[str, Any]] = []
                self._append_bridge_resume_entry_acquire_entity_repair_if_needed(
                    tasks=tasks,
                    resume_entry_task_ids_by_resource=splice_before_task_ids_by_resource,
                    patch_rows=acquire_entity_patch_rows,
                )
                if acquire_entity_patch_rows:
                    _stage_approval_patch(acquire_entity_patch_rows)
                    tasks = _refresh_staged_tasks(tasks)
            self.process_planner.nodes = deepcopy(planner_nodes_snapshot)
            self.process_planner.global_fsa = deepcopy(planner_global_fsa_snapshot)
            self.process_planner._apply_replan_patch(all_patch_rows)
            tasks = _refresh_staged_tasks(tasks)
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
            archive_replay = dict(bridge_debug.get("archive_replay") or {}) if isinstance(bridge_debug, dict) else {}
            source_archive_path = str(
                archive_replay.get("source_path")
                or self._runtime_bridge_session_archive_path()
                or ""
            ).strip()
            active_bridge_sequence = {
                "bridge_sequence_id": bridge_sequence_id,
                "bridge_task_ids": [
                    str(task.get("id", "")).strip()
                    for task in tasks
                    if str(task.get("id", "")).strip()
                ],
                "bridge_sequence_length": len(tasks),
                "recovery_group_id": bridge_sequence_id,
                "trigger": str(self._runtime_recovery_context.get("trigger", "")),
                "failed_task_id": failed_task_id,
                "violations": deepcopy(violations),
                "used_llm_bridge": used_llm_bridge,
                "system_coordination_state": deepcopy(
                    self._runtime_recovery_context.get("system_coordination_state") or {}
                ),
                "proposal_fingerprint": proposal_fingerprint,
                "state": "approved",
                "dispatched_bridge_task_ids": [],
                "completed_bridge_task_ids": [],
                "execution_started_at_utc": "",
                "last_dispatched_task_id": "",
                "last_completed_task_id": "",
                "continuation_requirements": deepcopy(
                    self._strip_continuation_requirement_actuals(
                        modeled_gap.get("continuation_requirements") or []
                    )
                ),
                "pending_nominal_task_ids": deepcopy(
                    modeled_gap.get("pending_nominal_task_ids") or []
                ),
                "continuation_repair_attempts": 0,
                "source_mode": self._runtime_bridge_session_mode(),
                "validation_policy": self._runtime_bridge_session_validation_policy(),
                "source_archive_path": source_archive_path,
                "source_archive_label": self._runtime_bridge_session_archive_label(),
                "recovery_safety_scope_id": recovery_safety_scope_id,
                "recovery_safety_status": recovery_safety_status,
                "recovery_safety_logic_json": str(
                    self.runtime_recovery.get("recovery_safety_logic_json") or ""
                ).strip(),
                "recovery_plan_dir": str(
                    self.runtime_recovery.get("recovery_plan_dir") or ""
                ).strip(),
                "recovery_safery_dir": str(
                    self.runtime_recovery.get("recovery_safery_dir") or ""
                ).strip(),
                "recovery_final_dir": str(
                    self.runtime_recovery.get("recovery_final_dir") or ""
                ).strip(),
                "recovery_final_output_path": str(
                    self.runtime_recovery.get("recovery_final_output_path") or ""
                ).strip(),
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
                "splice_before_task_ids_by_resource": deepcopy(
                    splice_before_task_ids_by_resource
                ),
                "deleted_task_ids": deepcopy(deleted_task_ids),
                "anchor_task_id": anchor_task_id,
                "proposal_fingerprint": proposal_fingerprint,
                "validation_policy": session_validation_policy,
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

        bridge_source = str(
            (active_bridge_sequence or {}).get("source", "")
        ).strip().lower()
        archived_replay_approval = bridge_source == "archived_final_output"
        if isinstance(active_bridge_sequence, dict):
            sequence_execution_policy = deepcopy(
                active_bridge_sequence.get("execution_policy") or {}
            )
            sequence_execution_policy["validation_policy"] = session_validation_policy
            sequence_execution_policy.pop("skip_runtime_plan_validation", None)
            active_bridge_sequence["execution_policy"] = sequence_execution_policy
            active_bridge_sequence["recovery_enforced_task_ids"] = self._bridge_sequence_task_ids(
                list(active_bridge_sequence.get("bridge_task_ids") or [])
            )
        if isinstance(bridge_debug, dict):
            bridge_debug["validation_policy"] = session_validation_policy
            debug_execution_policy = deepcopy(bridge_debug.get("execution_policy") or {})
            debug_execution_policy["validation_policy"] = session_validation_policy
            debug_execution_policy.pop("skip_runtime_plan_validation", None)
            bridge_debug["execution_policy"] = debug_execution_policy

        if verification_only_approval:
            validation_message = (
                f"Approved generated bridge verification proposal compiled to {len(tasks)} task(s); "
                "waiting for the CCA runtime plan validation before executing bridge macros."
            )
        elif archived_replay_approval:
            validation_message = (
                f"Approved archived bridge proposal compiled to {len(tasks)} task(s); "
                "waiting for the CCA runtime plan validation before executing bridge macros."
            )
        else:
            validation_message = (
                f"Approved bridge proposal '{proposal.get('macro_name') or proposal.get('function_name') or 'bridge_recovery'}' compiled to "
                f"{len(tasks)} task(s); waiting for the CCA runtime plan validation before executing bridge macros."
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
            last_completed_bridge_sequence=None,
            recovery_safety_scope_id=recovery_safety_scope_id,
            recovery_safety_status=recovery_safety_status,
            recovery_safety_logic_json=str(
                self.runtime_recovery.get("recovery_safety_logic_json") or ""
            ).strip(),
            recovery_safety_dir=str(
                self.runtime_recovery.get("recovery_safety_dir") or ""
            ).strip(),
            recovery_plan_dir=str(
                self.runtime_recovery.get("recovery_plan_dir") or ""
            ).strip(),
            recovery_safery_dir=str(
                self.runtime_recovery.get("recovery_safery_dir") or ""
            ).strip(),
            recovery_final_dir=str(
                self.runtime_recovery.get("recovery_final_dir") or ""
            ).strip(),
            recovery_final_output_path=str(
                self.runtime_recovery.get("recovery_final_output_path") or ""
            ).strip(),
            recovery_enforced_task_ids=list(
                dict(active_bridge_sequence or {}).get("recovery_enforced_task_ids") or []
            ),
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
            self.logger.info(
                "[Product] Approved bridge proposal finalization started: sending the CCA runtime plan validation source=%s validation_policy=%s.",
                bridge_source or "<unknown>",
                session_validation_policy,
            )
            self._send_runtime_plan_validation_check_sync(
                skip_revalidation=False,
                skip_recovery_safety_validation=(session_validation_policy == "no_validation"),
            )
            self.logger.info(
                "[Product] Approved bridge proposal CCA runtime plan validation sent successfully; persisting updated runtime state."
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
            self.logger.exception("[Product] Approved bridge proposal CCA runtime plan validation dispatch failed.")
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
        active_bridge_sequence = self._active_bridge_sequence()
        if self._bridge_sequence_is_live(active_bridge_sequence):
            warning = (
                "DES retry is blocked while the approved bridge sequence is still active."
            )
            self.logger.warning(
                "[Product] %s product=%s bridge_sequence_id=%s",
                warning,
                self.agent_name,
                str((active_bridge_sequence or {}).get("bridge_sequence_id") or "").strip(),
            )
            return self._recovery_result_with_action_feedback(
                self.get_runtime_recovery(),
                kind="warning",
                text=warning,
            )
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

from __future__ import annotations

"""Generate and review unvalidated outlines from immutable saved observations."""

import asyncio
import json
import logging
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn_context as context
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
    _slim_fault_event,
)
from cais_spade_llm.recovery_framework.evaluation_capture import sha256_file, write_record
from cais_spade_llm.recovery_framework.offline_recovery import SavedRecoveryProduct, SavedRecoveryResource
from cais_spade_llm.recovery_framework.outline_checks import CHECK_FIELDS
from cais_spade_llm.recovery_framework.part_slippage_recovery import _prompt_safety_rules
from cais_spade_llm.recovery_framework.recovery_observations import RECORDS
from cais_spade_llm.resources.machine.machine_jobs import public_processing_observations

logger = logging.getLogger(__name__)

_HIDDEN = context._PRIVATE_CONTEXT_FIELDS | {
    "evaluation_support", "available_capability_fields", "capabilities", "job_control",
    "program_options", "supports_executable_recovery", "supports_manipulator_pick_place",
    "primitive_steps", "capability_decompositions", "allowed_execution_surface", "active_job",
}
_NOT_CHECKED = {name: "not_checked" for name in ("ra", "physical_evidence", "cca", "nominal_continuation")}
DRAFTS = RECORDS / "outline_drafts"


def _public(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items() if key not in _HIDDEN}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return deepcopy(value)


def _inputs(observations: dict) -> tuple[dict, dict]:
    original = observations.get("prepared_recovery_request") or {}
    if not original.get("recovery_resources") or not original.get("llm_input"):
        raise ValueError("Saved observations need recovery_resources and llm_input for generation")
    prepared = {key: deepcopy(original[key]) for key in (
        "ra_jid", "P_id", "failure_anchor", "grounding_context", "part_tracker",
        "marked_reentry_context", "tools_catalog", "requirement_nodes", "goal_state",
    ) if key in original}
    prepared.update(offline_recovery=True, observation_jids=deepcopy(observations.get("jids", {})),
        source_observation_run_id=observations.get("run_id"),
        source_observation_captured_at=observations.get("captured_at"))
    llm = original["llm_input"]
    prepared["llm_input"] = {key: _public(llm[key]) for key in (
        "observed_runtime_state", "part_facts", "goal_conditions", "relevant_assembly_requirements",
    ) if key in llm}
    prepared["llm_input"]["fault_event"] = _public(_slim_fault_event(llm.get("fault_event", {})))
    prepared["recovery_resources"] = {}
    for jid, entry in original["recovery_resources"].items():
        row = {key: _public(entry[key]) for key in (
            "resource_jid", "resource_type", "recovery_snapshot", "static_capabilities",
            "recovery_adapter", "pending_tasks", "capability_evidence",
        ) if key in entry}
        row["recovery_des_model"] = deepcopy(entry.get("recovery_des_model") or {})
        variables = row["recovery_des_model"].get("state_variables", {})
        if "active_job" in variables:
            variables["active_job"]["private"] = True
        for declaration in variables.values():
            if "active_job" in declaration.get("description", ""):
                declaration.pop("description")
        prepared["recovery_resources"][jid] = row
    # These are recorded failure-time owner fields, not projected successor facts.
    for rid, record in observations.get("processing_observations", {}).items():
        jid = observations.get("jids", {}).get(rid)
        entry = prepared["recovery_resources"].get(jid)
        if entry is None:
            continue
        public = _public(public_processing_observations(record))
        entry.setdefault("static_capabilities", {})["processing_observations"] = public
        fields = {key: public[key] for key in ("job_state", "selected_program")}
        entry.setdefault("recovery_snapshot", {}).update(fields)
        for row in prepared["llm_input"].get("observed_runtime_state", {}).get("resources", []):
            if row.get("resource_jid") == jid:
                row.update(fields)
    prepared["saved_pending_nodes"] = deepcopy(observations.get("pending_nodes", []))
    prepared["diagnostic_regions"] = _public(
        observations.get("physical_evidence", {}).get("geometry", {}).get("regions", {}))
    # Declared reach is a static capability recorded for every robot, like reachability.
    prepared["diagnostic_reach"] = {jid: deepcopy(entry["static_capabilities"]["gripper_reach"])
        for jid, entry in original["recovery_resources"].items()
        if isinstance((entry.get("static_capabilities") or {}).get("gripper_reach"), dict)}
    prepared["validation_contract_version"] = 2
    prepared["public_safety_context_version"] = 1
    prepared["recovery_session"] = {
        "outline_mode": "incremental_candidates_validated", "recovery_selection_mode": "neurosymbolic",
        "outline_context_retrieval": True, "outline_diagnostic_bypass": True, "max_turns": 80,
        "max_observations": 0, "action_horizon": "1", "candidate_proposal_budget": 5,
    }
    return prepared, {}


class OutlineDraft:
    """Adapt saved-observation UI actions to execute_multi_turn_recovery."""

    def __init__(self, observations: dict, safety_files: list[str], *, root: Path = DRAFTS,
                 model: str | None = None, reasoning_effort: str | None = None,
                 check_ra_feasibility: bool = False, check_physical_evidence: bool = False,
                 check_cca_safety: bool = False) -> None:
        """Prepare a detached framework session with explicitly selected checks.

        Args:
            observations: Immutable saved failure input and its recorded provenance.
            safety_files: Exact selected specification paths.
            root: Directory for the fresh attempt's records.
            model: Optional attempt-specific provider model.
            reasoning_effort: Optional attempt-specific provider reasoning setting.
            check_ra_feasibility: Enable ResourceAgent feasibility validation.
            check_physical_evidence: Enable physical evidence calculations.
            check_cca_safety: Enable validation of the selected safety predicates.
        """
        from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner

        self.prepared, self.session = _inputs(observations)
        choices = dict(zip(CHECK_FIELDS, (check_ra_feasibility, check_physical_evidence, check_cca_safety)))
        if any(type(value) is not bool for value in choices.values()) or all(choices.values()):
            raise ValueError("Use the validated Recovery workflow when all three checks are enabled")
        self.prepared["recovery_session"].update(choices)
        self.rules = _prompt_safety_rules(safety_files)
        self.prepared["prompt_safety_rules"] = deepcopy(self.rules)
        from .outline_validation import load_outline_specifications

        self.prepared["diagnostic_safety_specifications"] = load_outline_specifications(safety_files, ROOT)
        resources = []
        if any(choices.values()):
            for jid, entry in observations["prepared_recovery_request"]["recovery_resources"].items():
                owner = SavedRecoveryResource(jid, entry, observations, _copy_observations=False)
                resources.append(owner)
                snapshot = owner.get_recovery_snapshot()
                recorded_model = entry.get("recovery_des_model") or {}
                evidence = owner.get_recovery_resource_evidence()
                model_values = (owner.recovery_des_model(snapshot=snapshot)
                                if recorded_model.get("descriptor_fingerprint") or
                                (entry.get("resource_type") != "robot" and
                                 evidence.get("environment_models", {}).get(evidence.get("resource_id")))
                                else deepcopy(recorded_model))
                self.prepared["recovery_resources"][jid] = {**deepcopy(entry), "recovery_snapshot": snapshot,
                    "recovery_des_model": model_values,
                    "resource_evidence": evidence}
            self.prepared.update(primitive_support_semantics="successor_primitive_union",
                physical_evidence=deepcopy(observations.get("physical_evidence") or {}),
                outline_safety_specifications=load_outline_specifications(safety_files, ROOT),
                monitor_history=deepcopy(observations.get("monitor_history") or {}))
            if check_physical_evidence:
                self.prepared["outline_evidence_policy_version"] = 2
                self.prepared["physical_evidence"].update(endpoint_evidence_policy_version=2,
                    outline_evidence_policy_version=2, physical_preview_contract_version=7, owner_projection_clock_version=1)
                description = (observations.get("planning_scene_capture", {}).get("native_model_parameters", {})
                               .get("robot_description", {}).get("value"))
                if description:
                    self.prepared["physical_evidence"]["endpoint_robot_description"] = description
        self.provider = SavedRecoveryProduct(observations, resources, model=model, reasoning_effort=reasoning_effort)
        self.planner = ProcessPlanner(self.provider, [])
        self.planner.nodes = deepcopy(observations.get("pending_nodes", []))
        self.provider.part_tracker = deepcopy(observations.get("part_tracker", {}))
        self.directory = root / f"draft_{uuid4().hex}"
        self.prepared["recovery_session"]["session_id"] = self.directory.name
        self.prepared["recovery_debug"] = {"artifact_directory": str(self.directory), "per_turn_debug_dir": str(self.directory)}
        self.prepared["_stop_after_multi_turn_phase"] = "outline"
        self._task: asyncio.Task | None = None
        self._stopped = False
        self.record = {"version": 2, "kind": "outline_draft", "pipeline": "execute_multi_turn_recovery",
            "outline_diagnostic_bypass": True, "run_id": self.directory.name,
            "status": "draft_ready", "scenario": observations["scenario"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_path": observations["source_path"], "source_sha256": observations["source_sha256"],
            "source_run_id": observations.get("run_id"), "captured_at": observations.get("captured_at"),
            "selected_safety_files": list(safety_files), "safety_rules": deepcopy(self.rules),
            "validation_checks": choices,
            "llm_settings": deepcopy(self.provider.llm_settings), "checks": dict(_NOT_CHECKED),
            "execution_authorized": False, "transition_trace": [], "provider_calls": 0,
            "llm_turns": [], "retrievals": [], "message": "Ready for an unvalidated multi_turn outline."}
        ask = self.provider.ask_llm_structured

        async def recorded_ask(prompt: str, **kwargs):
            self.record["provider_calls"] += 1
            self._save()
            return await ask(prompt, **kwargs)

        self.provider.ask_llm_structured = recorded_ask

        async def progress(**kwargs):
            self.record["message"] = kwargs.get("message", "Generating multi_turn outline; validation is bypassed.")
            self._save()

        self.provider.report_runtime_recovery_turn_progress = progress

    def snapshot(self) -> dict:
        """Present tentative events separately from accepted recovery results."""
        return {"status": self.record["status"], "message": self.record["message"],
            "run_id": self.record["run_id"], "outline_draft": deepcopy(self.record),
            "outline": [], "primitives": [], "execution": [], "outline_status": "unvalidated",
            "primitive_status": "waiting", "outline_safety_status": "not_checked",
            "execution_authorized": False, "artifact_directory": str(self.directory),
            "outline_artifact_directory": str(self.directory / "recovery_outline"),
            "observation_path": self.record["source_path"],
            "validation_checks": deepcopy(self.record.get("validation_checks") or dict.fromkeys(CHECK_FIELDS, False)),
            "validation_checks_locked": bool(self.record["status"] != "draft_ready"
                                              or (self._task is not None and not self._task.done())),
            "can_generate_outline": self.record["status"] == "draft_ready",
            "draft_path": str(self.directory / "draft.json")}

    def _save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        write_record(self.directory / "draft.json", self.record)

    async def generate(self) -> None:
        """Run and resume the existing orchestration, stopping at outline handoff."""
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn

        if self.record["status"] != "draft_ready":
            raise ValueError("Start fresh before generating another outline")
        self.directory.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        self.record.update(status="generating_draft", message="Generating multi_turn outline; validation is bypassed.")
        self._save()
        try:
            while not self._stopped:
                await multi_turn.execute_multi_turn_recovery(self.planner, self.prepared,
                    session_state=self.prepared.get("multi_turn_session_state"))
                self.session = self.prepared.get("multi_turn_session_state", {})
                self.record["transition_trace"] = deepcopy(self.session.get("accepted_outline_prefix", []))
                self.record["checks"].update(self.session.get("outline_check_results", {}))
                self.record["nominal_handoff"] = deepcopy(self.session.get("diagnostic_nominal_handoff", {}))
                self.record["handoff_safety_blocks"] = deepcopy(self.session.get("diagnostic_handoff_safety_blocks", []))
                status = self.session.get("status")
                if status == "ready_for_primitive_generation":
                    self.record.update(status="draft", message="Unvalidated multi_turn outline with proposed nominal handoff.")
                    break
                if status not in {"paused_after_outline_turn", "running"}:
                    self.record.update(status="NEEDS_CONTEXT" if status == "NEEDS_CONTEXT" and any(self.record["validation_checks"].values()) else "draft_incomplete",
                        message="; ".join(row.get("reason", "") for row in self.session.get("outline_validation_findings", []))
                        or f"multi_turn stopped at {status}.")
                    break
                if self.session.get("turn_index", 0) >= self.session.get("max_turns", 80):
                    self.record.update(status="draft_incomplete", message="multi_turn reached its 80-turn budget.")
                    break
                self._save()
            if self._stopped:
                self.record.update(status="stopped", message="Draft generation stopped; captured responses retained.")
        except (RuntimeError, OSError, TypeError, KeyError, ValueError) as exc:
            logger.warning("Multi_turn outline draft failed: %s", exc)
            self.record.update(status="draft_incomplete", message=str(exc))
        finally:
            self.record["elapsed_seconds"] = time.monotonic() - started
            self.record["request_summary"] = {key: deepcopy(value) for key, value in self.provider._last_structured_request.items()
                if key not in {"messages", "response_format", "tools"}}
            self.record["usage"] = []
            requests = sorted(self.directory.glob("recovery_outline/requests/*/*_request.json"))
            self.record["llm_turns"] = [{"turn": r["turn"], "stage": r["stage"],
                "request": str(path.relative_to(self.directory))}
                for path in requests for r in [json.loads(path.read_text(encoding="utf-8"))]]
            for path in sorted(self.directory.glob("recovery_outline/requests/*/*_response.json")):
                payload = json.loads(path.read_text(encoding="utf-8")).get("raw_response", {})
                if isinstance(payload, dict) and payload.get("usage"):
                    self.record["usage"].append({"source": str(path.relative_to(self.directory)), "usage": payload["usage"]})
            self.record["retrievals"] = deepcopy(self.session.get("outline_requested_refs", []))
            self._save()

    def stop(self) -> None:
        """Stop after the in-flight provider request is retained."""
        self._stopped = True
        self.prepared["_outline_diagnostic_stop"] = True
        self.record["message"] = "Stopping after the current multi_turn request."



def load_draft(path: Path, observations: dict, *, root: Path = DRAFTS) -> dict:
    """Read a matching draft for display, without accepting any proposed event."""
    path = path.resolve()
    if not path.is_relative_to(root.resolve()) or path.name != "draft.json":
        raise ValueError("Select a saved outline draft")
    record = json.loads(path.read_text(encoding="utf-8"))
    choices = record.get("validation_checks") or dict.fromkeys(CHECK_FIELDS, False)
    checks = record.get("checks") or {}
    valid_checks = set(choices) == set(CHECK_FIELDS) and all(type(value) is bool for value in choices.values())
    valid_checks = valid_checks and not all(choices.values()) and checks.get("nominal_continuation") == "not_checked"
    for field, name in zip(CHECK_FIELDS, ("ra", "physical_evidence", "cca")):
        valid_checks = valid_checks and checks.get(name) in ({"passed", "not_selected", "not_checked"} if choices.get(field) else {"not_checked"})
    if (record.get("kind") != "outline_draft" or record.get("version") not in {1, 2} or record.get("status") != "draft"
            or record.get("execution_authorized") is not False or not valid_checks):
        raise ValueError("Saved record is not an unvalidated outline draft")
    if (record.get("source_sha256") != observations["source_sha256"]
            or record.get("scenario") != observations["scenario"]
            or sha256_file(Path(observations["source_path"])) != record["source_sha256"]):
        raise ValueError("Draft source observations changed")
    return record


def latest_draft(observations: dict, *, root: Path = DRAFTS) -> Path | None:
    """Find the latest complete draft for this exact source, independently of accepted results."""
    for path in sorted(root.glob("draft_*/draft.json"), key=lambda item: item.stat().st_mtime_ns, reverse=True):
        try:
            load_draft(path, observations, root=root)
        except (OSError, ValueError, KeyError):
            continue
        return path
    return None

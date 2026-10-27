from __future__ import annotations

"""Generate and review unvalidated outlines from immutable saved observations."""

import asyncio
import importlib.util
import json
import logging
import math
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
    _slim_primitive_catalog_for_prompt,
)
from cais_spade_llm.recovery_framework.evaluation_capture import sha256_file, write_record
from cais_spade_llm.recovery_framework.offline_recovery import SavedRecoveryProduct
from cais_spade_llm.recovery_framework.part_slippage_recovery import _prompt_safety_rules
from cais_spade_llm.recovery_framework.recovery_observations import RECORDS
from cais_spade_llm.resources.environment_recovery import environment_recovery_descriptor
from cais_spade_llm.resources.machine.machine_jobs import machine_public_state, public_processing_observations

logger = logging.getLogger(__name__)

_HIDDEN = context._PRIVATE_CONTEXT_FIELDS | {
    "evaluation_support", "available_capability_fields", "capabilities",
    "program_options", "supports_executable_recovery", "supports_manipulator_pick_place",
    "primitive_steps", "capability_decompositions", "allowed_execution_surface", "active_job",
}
DRAFTS = RECORDS / "outline_drafts"
# A cost limit, not a progress judgment: a good draft takes six to ten LLM calls, and a long one such as
# Machining (sixteen events) used all of twenty; each refusal under a check costs another call.
DRAFT_MAX_TURNS = 40
# Where equipment is and how large it is. Routes, docking poses, handling
# assignments and transfer targets are left out so geometry gives no answer.
_GEOMETRY_KEYS = ("world_pose", "size", "length", "width", "surface_height", "enclosure_dimensions_m", "stand_height_m")
_MOBILE_KEYS = ("platform_overall_dimensions_m", "arm_mount_xyz", "arm_reach_m")
_NAVIGATION_MAP = ROOT / "ros2/cais_lab_robotics/scripts/generate_recovery_framework_map.py"


def _public(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items() if key not in _HIDDEN}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return deepcopy(value)


def _planar_pose(pose: list) -> list[float]:
    x, y, _, qx, qy, qz, qw = pose
    return [round(x, 4), round(y, 4), round(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)), 4)]


def _navigation_floor() -> dict:
    """Read the fixed floor map that mobile bases plan their routes around."""
    if not _NAVIGATION_MAP.is_file():
        return {}
    spec = importlib.util.spec_from_file_location("recovery_framework_navigation_map", _NAVIGATION_MAP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extent = [module.ORIGIN_X_M, module.ORIGIN_X_M + module.WIDTH * module.RESOLUTION_M,
              module.ORIGIN_Y_M, module.ORIGIN_Y_M + module.HEIGHT * module.RESOLUTION_M]
    return {"floor_extent_xy": [round(value, 3) for value in extent],
            "floor_obstacles_xy": {name: list(bounds) for name, bounds in module.OBSTACLE_BOUNDS.items()}}


def scene_geometry(observations: dict) -> dict:
    """Describe the captured cell layout the same way for every resource.

    Args:
        observations: A saved failure capture.

    Returns:
        Station, machine and robot-base geometry, each mobile base's current
        planar pose and arm data, and the navigation floor map.
    """
    scene = (observations.get("inputs") or {}).get("scene") or {}
    measured = (observations.get("physical_evidence") or {}).get("resources") or {}
    stations = {name: {key: deepcopy(row[key]) for key in _GEOMETRY_KEYS if key in row}
                for name, row in scene.items() if isinstance(row, dict) and "world_pose" in row}
    stations.update({row["resource_id"]: {key: deepcopy(row[key]) for key in _GEOMETRY_KEYS if key in row}
                     for row in scene.get("machines", []) if row.get("resource_id")})
    mobile = {}
    for name, row in scene.items():
        if isinstance(row, dict) and "platform_overall_dimensions_m" in row:
            mobile[name] = {key: deepcopy(row[key]) for key in _MOBILE_KEYS if key in row}
            pose = (measured.get(name) or {}).get("base_pose")
            if isinstance(pose, list) and len(pose) == 7:
                mobile[name]["base_pose_xy_yaw"] = _planar_pose(pose)
    return {"frame": "world", "stations": stations,
            "robot_bases": {row["resource_id"]: deepcopy(row["base_xyz"]) for row in scene.get("robots", [])
                            if row.get("resource_id") and row.get("base_xyz")},
            "mobile_bases": mobile, **_navigation_floor()}


def _inputs(observations: dict) -> tuple[dict, dict]:
    original = observations.get("prepared_recovery_request") or {}
    if not original.get("recovery_resources") or not original.get("llm_input"):
        raise ValueError("Saved observations need recovery_resources and llm_input for generation")
    prepared = {key: deepcopy(original[key]) for key in (
        "ra_jid", "P_id", "failure_anchor", "grounding_context", "part_tracker",
        "marked_reentry_context", "tools_catalog", "requirement_nodes", "goal_state", "code_grounding",
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
        evidence = entry.get("resource_evidence") or {}
        owner_model = (evidence.get("environment_models") or {}).get(evidence.get("resource_id")) or {}
        # Only a robot's model is its own recorded declaration; other models derive from the plant model
        # and the recorded evidence, so they are derived again with the current plant semantics.
        if entry.get("resource_type") != "robot" and owner_model and isinstance(evidence.get("snapshot"), dict):
            row["recovery_des_model"] = environment_recovery_descriptor(evidence, evidence["snapshot"], resource_jid=jid)
        else:
            row["recovery_des_model"] = deepcopy(entry.get("recovery_des_model") or {})
        variables = row["recovery_des_model"].get("state_variables", {})
        if "active_job" in variables:
            variables["active_job"]["private"] = True
        for declaration in variables.values():
            if "active_job" in declaration.get("description", ""):
                declaration.pop("description")
        if owner_model.get("events"):
            # The owner's declared events state, for example, where another resource's placement puts a part.
            row["environment_model_events"] = deepcopy(owner_model["events"])
        prepared["recovery_resources"][jid] = row
    # These are recorded failure-time owner fields, not projected successor facts.
    for rid, record in observations.get("processing_observations", {}).items():
        jid = observations.get("jids", {}).get(rid)
        entry = prepared["recovery_resources"].get(jid)
        if entry is None:
            continue
        public = _public(public_processing_observations(record))
        entry.setdefault("static_capabilities", {})["processing_observations"] = public
        snapshot = entry.setdefault("recovery_snapshot", {})
        snapshot.update(machine_public_state(record, snapshot))
        for row in prepared["llm_input"].get("observed_runtime_state", {}).get("resources", []):
            if row.get("resource_jid") == jid:
                row.update(machine_public_state(record, row))
    prepared["saved_pending_nodes"] = deepcopy(observations.get("pending_nodes", []))
    prepared["diagnostic_regions"] = _public(
        observations.get("physical_evidence", {}).get("geometry", {}).get("regions", {}))
    # Declared reach is a static capability recorded for every robot, like reachability.
    prepared["diagnostic_reach"] = {jid: deepcopy(entry["static_capabilities"]["gripper_reach"])
        for jid, entry in original["recovery_resources"].items()
        if isinstance((entry.get("static_capabilities") or {}).get("gripper_reach"), dict)}
    prepared["diagnostic_scene_geometry"] = scene_geometry(observations)
    prepared["diagnostic_primitives"] = {
        jid: _slim_primitive_catalog_for_prompt(entry.get("execution_primitive_catalog") or [])
        for jid, entry in original["recovery_resources"].items() if entry.get("execution_primitive_catalog")}
    prepared["validation_contract_version"] = 2
    prepared["public_safety_context_version"] = 1
    prepared["recovery_session"] = {
        "outline_mode": "incremental_candidates_validated", "recovery_selection_mode": "neurosymbolic",
        "outline_context_retrieval": True, "outline_diagnostic_bypass": True, "max_turns": DRAFT_MAX_TURNS,
        "max_observations": 0, "action_horizon": "1", "candidate_proposal_budget": 5,
    }
    return prepared, {}


class OutlineDraft:
    """Adapt saved-observation UI actions to execute_multi_turn_recovery."""

    def __init__(self, observations: dict, safety_files: list[str], *, root: Path = DRAFTS,
                 model: str | None = None, reasoning_effort: str | None = None,
                 check_ra_feasibility: bool = False, check_nonblocking: bool = False,
                 check_cca_safety: bool = False, cca_assumptions: list[dict] | None = None) -> None:
        """Prepare a detached framework session; only the checks turned on check the outline.

        Args:
            observations: Immutable saved failure input and its recorded provenance.
            safety_files: Exact selected specification paths.
            root: Directory for the fresh attempt's records.
            model: Optional attempt-specific provider model.
            reasoning_effort: Optional attempt-specific provider reasoning setting.
            check_ra_feasibility: Check each proposed event with Alg. 1 and Eq. 12 against the capture's
                recorded cell; a refusal returns to the LLM as validation feedback.
            check_nonblocking: Have the CCA check the LLM's declared completion (Eq. 6) in the
                composition of the capture's runtime models; a refusal names what normal production
                cannot continue, and synthesis continues.
            check_cca_safety: Have the CCA check each event the RA finds feasible against every selected
                specification (Eq. 13); needs ``check_ra_feasibility``, whose reach results place the
                acting robot. A refusal states the broken specification and its facts.
            cca_assumptions: Declared facts about the failure state the capture could not show, each
                ``{"resource", "clear_of", "reason"}``: the resource counts as outside those regions until
                an event moves it (captures are taken with the CCA bypassed).
        """
        from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner

        self.prepared, self.session = _inputs(observations)
        from .failure_grounding import apply_grounding

        grounded = apply_grounding(self.prepared, observations)
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn, multi_turn_diagnostic

        # A failure blocks every task that needs the failed resource, not only its own tasks.
        moved = multi_turn_diagnostic.resume_after_faults(
            self.prepared, multi_turn.build_multi_turn_session_seed(self.prepared))
        self.rules = _prompt_safety_rules(safety_files)
        self.prepared["prompt_safety_rules"] = deepcopy(self.rules)
        # A resource's conditions are the variables its model and its primitives declare (assumption A.2).
        multi_turn_diagnostic.declare_derived_conditions(self.prepared, {
            jid: entry.get("execution_primitive_catalog") or []
            for jid, entry in observations["prepared_recovery_request"]["recovery_resources"].items()})
        self.provider = SavedRecoveryProduct(observations, [], model=model, reasoning_effort=reasoning_effort)
        if check_ra_feasibility:
            from .ra_feasibility import recorded_cell

            self.prepared["recovery_session"]["check_ra_feasibility"] = True
            self.provider.recorded_cell = recorded_cell(observations, _navigation_floor())
        if check_nonblocking:
            from .cca_nonblocking import capture_inputs

            self.prepared["recovery_session"]["check_nonblocking"] = True
            self.provider.cca_capture = capture_inputs(observations)
        if check_cca_safety:
            if not check_ra_feasibility:
                raise ValueError("The CCA safety check places each acting robot from the RA's reach results; "
                                 "turn on the RA feasibility check too")
            from .cca_safety import capture_inputs as safety_inputs

            self.prepared["recovery_session"]["check_cca_safety"] = True
            self.provider.cca_safety_capture = safety_inputs(observations, safety_files, cca_assumptions)
        self.planner = ProcessPlanner(self.provider, [])
        self.planner.nodes = deepcopy(observations.get("pending_nodes", []))
        self.provider.part_tracker = deepcopy(observations.get("part_tracker", {}))
        self.directory = root / f"draft_{uuid4().hex}"
        self.prepared["recovery_session"]["session_id"] = self.directory.name
        self.prepared["recovery_debug"] = {"artifact_directory": str(self.directory), "per_turn_debug_dir": str(self.directory)}
        self.prepared["_stop_after_multi_turn_phase"] = "outline"
        self._task: asyncio.Task | None = None
        self._generating_message = "Generating multi_turn outline; validation is bypassed."
        self._stopped = False
        self.record = {"version": 2, "kind": "outline_draft", "pipeline": "execute_multi_turn_recovery",
            "outline_diagnostic_bypass": True, "ra_feasibility_check": check_ra_feasibility,
            "nonblocking_check": check_nonblocking, "cca_safety_check": check_cca_safety,
            "cca_assumptions": deepcopy(cca_assumptions or []), "run_id": self.directory.name,
            "status": "draft_ready", "scenario": observations["scenario"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_path": observations["source_path"], "source_sha256": observations["source_sha256"],
            "source_run_id": observations.get("run_id"), "captured_at": observations.get("captured_at"),
            "selected_safety_files": list(safety_files), "safety_rules": deepcopy(self.rules),
            "llm_settings": deepcopy(self.provider.llm_settings),
            "execution_authorized": False, "transition_trace": [], "provider_calls": 0,
            "llm_turns": [], "retrievals": [], "message": "Ready for an unvalidated multi_turn outline."}
        if grounded is not None:
            self.record["fault_grounding"] = deepcopy(grounded["fault_event"])
        if moved:
            self.record["resumed_tasks_replaced"] = moved
        ask = self.provider.ask_llm_structured

        async def recorded_ask(prompt: str, **kwargs):
            self.record["provider_calls"] += 1
            self._save()
            return await ask(prompt, **kwargs)

        self.provider.ask_llm_structured = recorded_ask

        async def progress(**kwargs):
            self.record["message"] = kwargs.get("message", self._generating_message)
            self._save()

        self.provider.report_runtime_recovery_turn_progress = progress

    def snapshot(self) -> dict:
        """Present tentative events separately from accepted recovery results."""
        return {"status": self.record["status"], "message": self.record["message"],
            "run_id": self.record["run_id"], "outline_draft": self._named_record(),
            "outline": [], "primitives": [], "execution": [], "outline_status": "unvalidated",
            "primitive_status": "waiting", "outline_safety_status": "not_checked",
            "execution_authorized": False, "artifact_directory": str(self.directory),
            "outline_artifact_directory": str(self.directory / "recovery_outline"),
            "observation_path": self.record["source_path"],
            "can_generate_outline": self.record["status"] == "draft_ready",
            "draft_path": str(self.directory / "draft.json")}

    def _named_record(self) -> dict:
        """Return the draft with each event, and each connecting nominal step, naming its resources."""
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_outline_facts import (
            resource_names,
            with_resource_names,
        )

        names = resource_names(self.prepared)
        record = deepcopy(self.record)
        for row in [*record["transition_trace"], *record.get("nominal_handoff", {}).get("nominal_connection", [])]:
            row["resource"] = names.get(row.get("resource_jid"), row.get("resource_jid"))
            if isinstance(row.get("expected_end_state"), dict):
                row["expected_end_state"] = {field: names.get(value, value) if isinstance(value, str) else value
                                             for field, value in row["expected_end_state"].items()}
            # Drafts saved before October 21 call the field place_end_state.
            for key in ("neighbor_end_state", "place_end_state"):
                if isinstance(row.get(key), dict):
                    row[key] = {names.get(place, place): values for place, values in row[key].items()}
            for change in row.get("condition_changes") or []:
                change["entity"] = names.get(change.get("entity"), change.get("entity"))
                if isinstance(change.get("value"), str):
                    change["value"] = names.get(change["value"], change["value"])
                if change.get("reason"):
                    change["reason"] = with_resource_names(change["reason"], names)
        return record

    def _finished_message(self) -> str:
        """Say how the outline ended; a declared recovery nominal work cannot take over is kept for review."""
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_outline_facts import (
            resource_names,
            with_resource_names,
        )

        handoff = self.record["nominal_handoff"]
        checks = [name for name, on in (("RA", self.record["ra_feasibility_check"]),
                                        ("nonblocking", self.record["nonblocking_check"]),
                                        ("CCA safety", self.record.get("cca_safety_check"))) if on]
        kind = f"{'- and '.join(checks)}-checked" if checks else "Unvalidated"
        if handoff.get("complete"):
            assumed = "; ".join(row["reason"] for row in handoff.get("assumptions") or [] if row.get("reason"))
            if not assumed:
                return f"{kind} multi_turn outline with proposed nominal handoff."
            return with_resource_names(f"{kind} multi_turn outline with proposed nominal handoff; assumed, not "
                                       f"checked: {assumed}.", resource_names(self.prepared))
        reasons = "; ".join(handoff.get("not_connected_reasons") or ["its recorded handoff conditions do not hold"])
        return with_resource_names(f"The LLM declared the recovery complete; nominal work cannot take over: {reasons}.",
                                   resource_names(self.prepared))

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
        checks = [name for name, on in (("the RA feasibility check", self.record["ra_feasibility_check"]),
                                        ("the CCA nonblocking check", self.record["nonblocking_check"]),
                                        ("the CCA safety check", self.record.get("cca_safety_check"))) if on]
        checked = f"{' and '.join(checks)} {'is' if len(checks) == 1 else 'are'} on" if checks else "validation is bypassed"
        self._generating_message = f"Generating multi_turn outline; {checked}."
        self.record.update(status="generating_draft", message=self._generating_message)
        self._save()
        try:
            while not self._stopped:
                await multi_turn.execute_multi_turn_recovery(self.planner, self.prepared,
                    session_state=self.prepared.get("multi_turn_session_state"))
                self.session = self.prepared.get("multi_turn_session_state", {})
                self.record["transition_trace"] = deepcopy(self.session.get("accepted_outline_prefix", []))
                self.record["nominal_handoff"] = deepcopy(self.session.get("diagnostic_nominal_handoff", {}))
                if self.record["ra_feasibility_check"]:
                    self.record["ra_feasibility_checks"] = deepcopy(self.session.get("ra_feasibility_checks", []))
                if self.record["nonblocking_check"]:
                    self.record["cca_nonblocking_checks"] = deepcopy(self.session.get("cca_nonblocking_checks", []))
                if self.record.get("cca_safety_check"):
                    self.record["cca_safety_checks"] = deepcopy(self.session.get("cca_safety_checks", []))
                status = self.session.get("status")
                if status == "ready_for_primitive_generation":
                    self.record.update(status="draft", message=self._finished_message())
                    break
                if status not in {"paused_after_outline_turn", "running"}:
                    self.record.update(status="draft_incomplete",
                        message="; ".join(row.get("reason", "") for row in self.session.get("outline_validation_findings", []))
                        or f"multi_turn stopped at {status}.")
                    break
                budget = self.session.get("max_turns", DRAFT_MAX_TURNS)
                if self.session.get("turn_index", 0) >= budget:
                    self.record.update(status="draft_incomplete", message=f"multi_turn reached its {budget}-turn budget.")
                    break
                self._save()
            if self._stopped:
                self.record.update(status="stopped", message="Draft generation stopped; captured responses retained.")
        except (RuntimeError, OSError, TypeError, LookupError, AttributeError, ValueError) as exc:
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
    # Drafts saved before October 22 may record outline checks; those checks were removed.
    if (record.get("kind") != "outline_draft" or record.get("version") not in {1, 2} or record.get("status") != "draft"
            or record.get("execution_authorized") is not False or any((record.get("validation_checks") or {}).values())):
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

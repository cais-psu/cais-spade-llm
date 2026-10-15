from __future__ import annotations

"""Start and reset the Part slippage example through SystemBridge."""

import asyncio
import logging
import secrets
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.failure_checkpoints import CHECKPOINTS
from cais_spade_llm.recovery_framework.outline_checks import CHECK_FIELDS
from cais_spade_llm.recovery_framework.recovery_observations import (
    RECORDS,
    copy_generation_records,
    list_observations,
    observation_fingerprint,
    observation_summary,
    read_observations,
)
from cais_spade_llm.ui.recovery_setup import (
    FAILURE_SCENARIOS,
    default_setup,
    load_setup,
    slippage_preset,
    validate_setup,
)

logger = logging.getLogger(__name__)

_PART_SLIPPAGE_SAFETY_FILES = (
    "cais_spade_llm/specification/safety/safety_assembly_board-v1_mutex.txt",
    "cais_spade_llm/specification/safety/safety_assembly_board-v1_gear_small_before_KET4_Square_4mm.txt",
)

_DEFAULT_OBSERVATIONS = {
    "Part slippage": "capture_f923020bf9ce496eb443826291f8f2c2",
    "Machining breakdown during part processing": "capture_8beafdb8a01e444894cfaf41d29921fd",
    "Conveyor breakdown": "capture_1b3f3d1d729a4701b94cff514c783130",
    "ur5e-1 breakdown": "capture_0a7fde8074b6426586b204204f2b3417",
}


def default_observation_paths() -> dict[str, str]:
    """Return the retained capture paths available for fresh Recovery attempts.

    Returns:
        Exact failure scenario names mapped to existing original observations files.
    """
    return {scenario: str(path) for scenario, identifier in _DEFAULT_OBSERVATIONS.items()
            if (path := RECORDS / identifier / "observations.json").is_file()}


def example_setup(*, outline_mode: str = "auto", primitive_mode: str = "auto",
                  scenario: str = "Part slippage", safety_files: list[str] | None = None, root=ROOT) -> dict:
    """Build simulation settings without changing saved setup files.

    Args:
        outline_mode: Selected outline generation mode.
        primitive_mode: Selected primitive generation mode.
        scenario: Exact configured failure scenario.
        safety_files: Explicit specifications, or the scenario defaults.
        root: Project checkout containing the configuration.

    Returns:
        Validated simulation settings with the selected failure checkpoint.
    """
    if outline_mode not in {"auto", "manual"} or primitive_mode not in {"auto", "manual"}:
        raise ValueError("Recovery stages require Auto or Manual")
    setup = default_setup(root)
    setup.update(
        selected_product_order_file="cais_spade_llm/specification/products/orders/assembly_board-v1-two-parts.json",
        selected_safety_file="__NONE__", execution_mode="simulation", diagnostic_cca_bypass=True,
        runtime_recovery_mode=outline_mode, runtime_recovery_primitive_mode=primitive_mode,
        runtime_recovery_validation_policy="no_validation",
    )
    saved_path = root / "cais_spade_llm/initialization/recovery_framework_setup.json"
    if saved_path.exists():
        setup["simulation"] = deepcopy(load_setup(saved_path, root=root).get("simulation", setup["simulation"]))
    if scenario not in FAILURE_SCENARIOS:
        raise ValueError("Select an existing failure scenario")
    from cais_spade_llm.recovery_framework.recovery_case_safety import default_safety_files

    setup["selected_safety_files"] = default_safety_files(scenario) if safety_files is None else list(safety_files)
    setup["recovery_permitted_resources"] = list(setup["permitted_resources"])
    if scenario == "Part slippage":
        inputs = validate_setup(setup, root=root)
        setup["failure_scenario"] = slippage_preset(inputs["models"], root=root)
    else:
        resource_id = {"Conveyor breakdown": "Conveyor", "ur5e-1 breakdown": "ur5e-1",
                       "Machining breakdown during part processing": "M1"}[scenario]
        if scenario != "Machining breakdown during part processing":
            setup["permitted_resources"] = [rid for rid in setup["permitted_resources"] if rid not in {"M2", "ur5e-2"}]
        setup["failure_scenario"] = {"scenario": scenario, "resource_id": resource_id,
                                     "checkpoint": CHECKPOINTS[scenario], "mode": "once"}
        if scenario == "Conveyor breakdown":
            setup["failure_scenario"].update(
                part_name="KET4_Square_4mm",
                additional_condition={"resource_id": "ur5e-4", "part_name": "gear_small"},
            )
        if scenario == "Machining breakdown during part processing":
            setup["selected_product_order_file"] = "cais_spade_llm/specification/products/orders/assembly_board-v1-machine-breakdown.json"
            setup["failure_scenario"].update(checkpoint="during_processing", seed=secrets.randbits(32))
    setup["nominal_permitted_resources"] = list(setup["permitted_resources"])
    validate_setup(setup, root=root)
    return setup


class PartSlippageExample:
    """Keep one launch request alive independently of any NiceGUI page client."""

    def __init__(self, bridge: Any) -> None:
        """Keep the existing bridge as the owner of simulation and agent lifecycles."""
        self.bridge = bridge
        self.outline_mode = "auto"
        self.primitive_mode = "auto"
        self._gazebo_modes = (self.outline_mode, self.primitive_mode)
        self.source = "gazebo"
        self.scenario = "Part slippage"
        self.safety_files: list[str] = list(_PART_SLIPPAGE_SAFETY_FILES)
        self.observation_path = ""
        self.validation_checks: dict[str, bool] | None = None
        self._defer_preparation = False
        self._selection_revision = 0
        self._selected_observation_summary: dict | None = None
        self._fresh_cancel = threading.Event()
        self._selected_observations: dict = {}
        self._offline = None
        self._draft = None
        self._saved_draft_path: Path | None = None
        self._test_proposal: dict | None = None
        self._test_observations: dict | None = None
        self._test_source: dict | None = None
        self._task: asyncio.Task | None = None
        self._preview_task: asyncio.Task | None = None
        self._cancelled = False
        self._runtime = None
        self._saved_settings: dict = {}
        self._saved_bundle: str | None = None
        self._status = {"status": "idle", "message": "Select a failure scenario and Start failure."}

    def snapshot(self) -> dict:
        """Read progress without creating processes or modifying settings."""
        state = {
            "source": self.source, "scenario": self.scenario, "outline_only": False,
            "selected_safety_files": list(self.safety_files), "selected_observation_path": self.observation_path,
            "observations": {}, "observation_path": "",
            "outline_mode": self.outline_mode, "primitive_mode": self.primitive_mode,
            "outline_status": "waiting", "primitive_status": "waiting",
            "outline": [], "primitives": [], "execution": [], "artifact_directory": "",
            "safety_status": "skipped", **deepcopy(self._status),
            "validation_checks": deepcopy(self.validation_checks or dict.fromkeys(CHECK_FIELDS, False)),
            "validation_checks_selected": self.validation_checks is not None,
            "validation_checks_locked": False,
            "defer_preparation": self._defer_preparation,
            "selection_revision": self._selection_revision,
        }
        runtime = self._runtime
        offline_state = {}
        if self.source == "offline" and self._offline is not None:
            offline_state = self._offline.snapshot()
            state.update(offline_state)
        elif self.source == "gazebo" and runtime is not None and self._status["status"] in {"running_example", "blocked", "stopped"}:
            state.update(runtime.part_slippage_recovery.snapshot())
            if self._status["status"] in {"blocked", "stopped"}:
                state.update(self._status)
            if state["status"] == "running_example":
                outcome = {key: runtime.outcome.get(key) for key in ("status", "reason")}
                if runtime.conveyor_fault.status == "triggered":
                    state["message"] = f"Reconciling the observed {self.scenario}."
                elif outcome.get("status") in {"blocked", "execution_unavailable", "held", "stopped"}:
                    state.update(status="blocked", message=outcome.get("reason", f"Example could not reach {self.scenario}"))
                state["outcome"] = outcome
        session = self._offline if self.source == "offline" else getattr(runtime, "part_slippage_recovery", None)
        state["busy"] = bool((self._task is not None and not self._task.done())
                             or (self._preview_task is not None and not self._preview_task.done())
                             or (session is not None and session._task is not None and not session._task.done()))
        if self.source == "offline" and self._offline is None and self._selected_observations:
            if self._selected_observation_summary is None:
                self._selected_observation_summary = observation_summary(self._selected_observations)
            state.update(observations=deepcopy(self._selected_observation_summary), observation_path=self.observation_path)
        if self.source == "offline":
            if self._draft is not None:
                state.update(self._draft.snapshot())
                state["busy"] = state["busy"] or bool(self._draft._task and not self._draft._task.done())
            state.update(outline_mode="manual", primitive_mode="manual")
            missing = (state.get("observations") or {}).get("missing") or []
            if not self.observation_path:
                reason = "Select saved observations to prepare the recovery input."
            elif state["busy"]:
                reason = "Preparing input from saved observations." if state["status"] == "preparing_outline" else "Generation already running."
            elif missing:
                reason = "NEEDS_CONTEXT: " + ", ".join(missing)
            elif state["status"] == "blocked":
                reason = state["message"]
            elif state["status"] not in {"outline_ready", "stopped"}:
                reason = ("Preparing input from saved observations." if state["status"] == "idle"
                          else "Outline already generated; use Regenerate for another proposal.")
            else:
                reason = ""
            state["outline_generate_reason"] = reason
            state["can_generate_outline"] = not reason
            state["can_generate_outline_only"] = bool(self.observation_path and not state["busy"])
            state["can_load_draft"] = bool(self._saved_draft_path and not state["busy"])
            if self._draft is not None:
                state["can_generate_outline"] = self._draft.record["status"] == "draft_ready" and not state["busy"]
                state["outline_generate_reason"] = "" if state["can_generate_outline"] else "Start fresh for another attempt."
            elif self._offline is not None:
                state["validation_checks"] = deepcopy(offline_state.get("validation_checks") or dict.fromkeys(CHECK_FIELDS, True))
                state["validation_checks_locked"] = bool(offline_state.get("validation_checks_locked")
                                                         or state["busy"] or state.get("outline") or state.get("primitives"))
            if self._defer_preparation:
                state["validation_checks"] = deepcopy(self.validation_checks or dict.fromkeys(CHECK_FIELDS, False))
                state["validation_checks_locked"] = state["busy"]
                state["can_generate_outline"] = bool(self.observation_path and not state["busy"])
                state["outline_generate_reason"] = ("Generation already running." if state["busy"] else
                    "Select saved observations to prepare the recovery input." if not self.observation_path else "")
        return state

    def set_options(self, *, source: str, scenario: str, safety_files: list[str], observation_path: str = "",
                    check_ra_feasibility: bool = True, check_physical_evidence: bool = True,
                    check_cca_safety: bool = True, defer_preparation: bool = False) -> dict:
        """Select the next input source and checks without launching generation.

        Args:
            source: Existing Gazebo or Offline source selection.
            scenario: Exact failure scenario.
            safety_files: Selected safety specification paths.
            observation_path: Original saved failure observations.
            check_ra_feasibility: Run ResourceAgent feasibility validation.
            check_physical_evidence: Run physical evidence calculations.
            check_cca_safety: Run CCA validation for the selected specifications.
            defer_preparation: Store preferences without reading observations or preparing an attempt.

        Returns:
            The current workflow state with the selected checks.
        """
        state = self.snapshot()
        choices = dict(zip(CHECK_FIELDS, (check_ra_feasibility, check_physical_evidence, check_cca_safety)))
        if any(type(value) is not bool for value in choices.values()) or type(defer_preparation) is not bool:
            raise ValueError("Outline check selections must be booleans")
        checks_changed = choices != self.validation_checks
        if checks_changed and state.get("validation_checks_locked") and not (defer_preparation and not state["busy"]):
            if defer_preparation:
                raise ValueError("Wait for the current generation before changing checks")
            raise ValueError("Start fresh before changing this attempt's checks")
        preparing = self.source == "offline" and state["status"] == "preparing_outline"
        if (state["busy"] and (defer_preparation or not preparing)) or state["status"] in {"running_example", "executing", "resuming"}:
            raise ValueError("Stop the current action before changing the example")
        ready = state["status"] in {"idle", "stopped", "draft_ready", "outline_ready"} and not state.get("outline")
        if not ready and not defer_preparation and (scenario != self.scenario or safety_files != self.safety_files):
            raise ValueError("Reset example before changing its failure or safety selections")
        if source not in {"gazebo", "offline"} or scenario not in FAILURE_SCENARIOS:
            raise ValueError("Select Gazebo or Offline and an existing failure scenario")
        if scenario != self.scenario:
            from cais_spade_llm.recovery_framework.recovery_case_safety import default_safety_files

            selected_paths = {(ROOT / name).resolve() for name in safety_files}
            prior_defaults = {(ROOT / name).resolve() for name in default_safety_files(self.scenario)}
            safety_files = (default_safety_files(scenario) if selected_paths == prior_defaults else
                [*safety_files, *(name for name in default_safety_files(scenario)
                                 if (ROOT / name).resolve() not in selected_paths)])
        safety_root = (ROOT / "cais_spade_llm/specification/safety").resolve()
        for name in safety_files:
            path = (ROOT / name).resolve()
            if not path.is_relative_to(safety_root) or not path.is_file():
                raise ValueError(f"Safety specification is unavailable: {name}")
        input_changed = (self.source, self.scenario, self.observation_path, self.safety_files) != (source, scenario, observation_path, safety_files)
        if defer_preparation:
            selected = {} if input_changed else self._selected_observations
        else:
            selected = read_observations(observation_path) if observation_path else {}
        if selected and selected.get("scenario") != scenario:
            raise ValueError("Saved observations belong to a different failure scenario")
        changed = input_changed or checks_changed or defer_preparation != self._defer_preparation
        if source != self.source:
            if source == "offline":
                self._gazebo_modes = (self.outline_mode, self.primitive_mode)
                self.outline_mode = self.primitive_mode = "manual"
            else:
                self.outline_mode, self.primitive_mode = self._gazebo_modes
        self.source, self.scenario = source, scenario
        self.safety_files, self.observation_path = list(safety_files), observation_path
        self.validation_checks = choices
        self._selected_observations = selected
        self._defer_preparation = defer_preparation
        if changed:
            self._selection_revision += 1
        if input_changed or not defer_preparation:
            self._selected_observation_summary = None
        if defer_preparation:
            if input_changed:
                self._draft = None
                self._offline = None
            self._saved_draft_path = None
            if input_changed or (self._draft is None and self._offline is None):
                self._status = {"status": "idle", "message": "Choose checks, then Generate a fresh outline from the verified capture."}
            return self.snapshot()
        if changed:
            self._draft = None
            self._saved_draft_path = None
            if source == "offline" and selected:
                from .outline_draft import latest_draft

                self._saved_draft_path = latest_draft(selected)
            if self._offline is not None:
                self._offline.stop("Saved observation selection changed")
                if self._offline._task is not None and not self._offline._task.done():
                    previous = self._offline
                    selected_options = (source, scenario, observation_path)

                    async def prepare_after_drain() -> None:
                        await self._drain_offline(previous)
                        if (self.source == "offline" and self.observation_path and self._offline is None
                                and self._status["status"] == "idle"
                                and selected_options == (self.source, self.scenario, self.observation_path)):
                            self._new_offline(generate_outline=False)

                    self._task = asyncio.create_task(prepare_after_drain(),
                                                     name="drain_part_slippage_offline")
                    self._task.add_done_callback(self._task_finished)
            self._offline = None
            self._status = {"status": "idle", "message": "Ready to start the selected example."}
            if source == "gazebo" and self._runtime is not None and self.bridge.system_running:
                self._status = {"status": "running_example", "message": "Reconnected to the paused failure."}
        return self.snapshot()

    async def _drain_offline(self, previous: Any) -> None:
        """Retain an old offline worker until its cancellation has completed."""
        task = previous._task
        if task is None:
            return
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.done():
                    cancelled = True
        if not task.cancelled():
            task.result()
        if cancelled:
            raise asyncio.CancelledError()

    def saved_observations(self) -> list[dict]:
        """List retained observation sources without querying Gazebo."""
        return list_observations()

    def prepare_offline(self) -> dict:
        """Prepare the selected immutable capture without calling the LLM."""
        if self.source != "offline":
            raise ValueError("Select Offline before preparing saved observations")
        state = self.snapshot()
        if self._draft is not None:
            return state
        if state["busy"] or (self._offline is not None and state["status"] not in {"blocked", "stopped"}):
            return state
        return self._new_offline(generate_outline=False)

    def _new_offline(self, *, generate_outline: bool | None) -> dict:
        from cais_spade_llm.recovery_framework.offline_recovery import create_offline_recovery

        if not self.observation_path:
            raise ValueError("Select saved observations first")
        if self._offline is not None:
            self._offline.stop()
        self._offline = None
        try:
            observations = read_observations(self.observation_path)
            if observations.get("scenario") != self.scenario:
                raise ValueError("Saved observations belong to a different failure scenario")
            self._selected_observations = observations
            self._selected_observation_summary = observation_summary(observations)
            choices = self.validation_checks or dict.fromkeys(CHECK_FIELDS, True)
            if not all(choices.values()):
                from .outline_draft import OutlineDraft

                self._draft = OutlineDraft(observations, self.safety_files, **choices)
                self._draft._save()
                return self.snapshot()
            session = create_offline_recovery(observations, outline_mode=self.outline_mode,
                                               primitive_mode=self.primitive_mode,
                                               source_observation_path=self.observation_path)
            session.runtime.setup["selected_safety_files"] = list(self.safety_files)
            session._publish(selected_safety_files=self.safety_files, validation_checks=choices,
                             validation_checks_locked=generate_outline is True
                             or (generate_outline is None and self.outline_mode == "auto"))
            session.begin(observations["failure"], generate_outline=generate_outline)
            self._offline = session
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._draft = None
            self._status = {"status": "blocked", "message": str(exc)}
        return self.snapshot()

    def start_offline(self) -> dict:
        """Prepare a Manual session through the retained legacy entrypoint."""
        if self.source != "offline" or self.snapshot()["busy"] or self.snapshot()["status"] != "idle":
            raise ValueError("Reset the offline example before starting another test")
        return self._new_offline(generate_outline=None)

    def load_offline_result(self) -> dict:
        """Copy a saved outline or composition into a new Manual review session."""
        from cais_spade_llm.recovery_framework.diagnostics import read_object, write_record
        from cais_spade_llm.recovery_framework.offline_recovery import create_offline_recovery

        if self.source != "offline" or self.snapshot()["busy"] or self.snapshot()["status"] not in {"idle", "outline_ready", "stopped", "blocked"}:
            raise ValueError("Reset the offline example before loading a saved result")
        observations = read_observations(self.observation_path)
        if observations.get("scenario") != self.scenario:
            raise ValueError("Saved observations belong to a different failure scenario")
        parent = Path(self.observation_path).resolve().parent
        state = read_object(parent / "run.json") if (parent / "run.json").is_file() else {}
        if state.get("status") not in {"offline_completed", "outline_review"}:
            selected_path = str(Path(self.observation_path).resolve())
            saved = next((row for row in self.saved_observations()
                          if selected_path in row.get("related_observation_paths", [row["source_path"]])), {})
            if saved.get("result_observation_path"):
                parent = Path(saved["result_observation_path"]).resolve().parent
                state = read_object(parent / "run.json")
        if state.get("status") not in {"offline_completed", "outline_review"}:
            raise ValueError("This capture has no completed offline outline or composition")
        outline_only = state["status"] == "outline_review"
        source = Path(state["artifact_directory"]).resolve()
        if not source.is_relative_to(parent):
            raise ValueError("Saved composition refers to another run")
        outline_request = read_object(source / "outline_request.json")
        proposal = {} if outline_only else read_object(source / "proposal.json")
        if outline_only:
            from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
                recovery_validation_fingerprint,
            )

            recorded = outline_request.get("multi_turn_session_state", {})
            if not isinstance(recorded, dict):
                raise ValueError("Saved outline session evidence must be an object")
            outline = read_object(source / "outline.json").get("outline")
            accepted = recorded.get("accepted_outline_prefix")
            if (state.get("validation_contract_version") != 2 or state.get("outline_status") != "completed"
                    or outline_request.get("source_observation_fingerprint") != observation_fingerprint(observations)
                    or not isinstance(observations.get("pending_nodes"), list)
                    or recovery_validation_fingerprint(outline_request.get("saved_pending_nodes"))
                       != recovery_validation_fingerprint(observations["pending_nodes"])
                    or recorded.get("current_phase") != "primitive_generation"
                    or recorded.get("status") != "ready_for_primitive_generation"
                    or not isinstance(accepted, list) or not accepted
                    or any(not isinstance(row, dict) or row.get("candidate_source") != "llm" for row in accepted)
                    or recovery_validation_fingerprint(accepted) != recovery_validation_fingerprint(outline)
                    or recovery_validation_fingerprint(accepted) != recovery_validation_fingerprint(state.get("outline"))):
                raise ValueError("Saved outline does not match its completed source and accepted events")
        elif proposal.get("source_observation_fingerprint") != observation_fingerprint(observations):
            raise ValueError("Saved composition does not match its source observations")
        if proposal.get("composition_validation_version") == 1:
            from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
                recovery_validation_fingerprint,
            )
            from cais_spade_llm.resources.recovery_composition import composition_program_error

            composition = read_object(source / "composition.json")
            program = composition.get("accepted_primitive_program", [])
            error = composition_program_error(outline_request.get("multi_turn_session_state", {}).get("accepted_outline_prefix", []),
                program, prepared=outline_request, session=outline_request.get("multi_turn_session_state", {}))
            if (error or composition.get("execution_authorized") is not False
                    or proposal.get("execution_authorized") is not False
                    or composition.get("source_outline_fingerprint") != recovery_validation_fingerprint(outline_request)
                    or composition.get("accepted_outline_prefix") != outline_request["multi_turn_session_state"]["accepted_outline_prefix"]
                    or recovery_validation_fingerprint(program) != recovery_validation_fingerprint(state.get("primitives"))
                    or [{key: row.get(key) for key in ("outline_id", "resource_jid", "part_name", "primitive_steps", "composition_validation")}
                        for row in proposal.get("macro_tasks", [])] !=
                       [{key: row.get(key) for key in ("outline_id", "resource_jid", "part_name", "primitive_steps", "composition_validation")}
                        for row in program]):
                raise ValueError(error or "Saved composition evidence differs from the displayed result")
        session = create_offline_recovery(observations, outline_mode=self.outline_mode,
                                          primitive_mode=self.primitive_mode,
                                          source_observation_path=self.observation_path)
        session._failure = deepcopy(observations["failure"])
        session._save_observations()
        directory = session._new_artifact_directory()
        session._install_prepared(outline_request, directory, "", new_attempt=False, source_directory=source,
                                  require_nominal_complete=outline_only or proposal.get("composition_validation_version") == 1)
        session._proposal = proposal
        session._outline_request = deepcopy(outline_request)
        paths = copy_generation_records(state, parent, directory)
        if proposal.get("composition_validation_version") == 1:
            (directory / "composition.json").write_bytes((source / "composition.json").read_bytes())
        for name in ("prompt_safety_rules.json", "outline_evidence_requirements.json"):
            if (source / name).is_file():
                (directory / name).write_bytes((source / name).read_bytes())
        for name, value in (("proposal.json", session._proposal), ("outline_request.json", session._outline_request),
                            ("outline.json", {"outline": state["outline"]})):
            if name != "proposal.json" or not outline_only:
                write_record(directory / name, value)
        if outline_only or proposal.get("composition_validation_version") == 1:
            from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (
                _nominal_artifact_path,
                _nominal_artifact_sha256,
            )

            reference = session._prepared["multi_turn_session_state"]["nominal_continuation_evidence"]["artifact_reference"]
            copied = _nominal_artifact_path(directory / "recovery_outline", reference)
            if _nominal_artifact_sha256(copied) != reference["sha256"]:
                raise ValueError("Copied nominal continuation differs from its verified source")
            session_id = f"{session.runtime.context.run_id}_{session._generation:04d}"
            for request in (session._prepared, session._outline_request):
                request.setdefault("recovery_debug", {}).update(
                    artifact_directory=str(directory), per_turn_debug_dir=str(directory))
                request.setdefault("recovery_session", {})["session_id"] = session_id
                request["multi_turn_session_state"]["recovery_session_id"] = session_id
            if isinstance(state.get("selected_safety_files"), list):
                self.safety_files = deepcopy(state["selected_safety_files"])
        session.runtime.setup["selected_safety_files"] = list(self.safety_files)
        session._publish(status="outline_review" if outline_only else "offline_completed", outline_status="completed",
                         primitive_status="waiting" if outline_only else "completed",
                         outline=state["outline"], primitives=[] if outline_only else state["primitives"],
                         validation_contract_version=(outline_request.get("validation_contract_version") if outline_only
                                                      else session._proposal.get("validation_contract_version")),
                         execution_authorized=False,
                         composition_validation_version=session._proposal.get("composition_validation_version"),
                         composition_feasibility=session._proposal.get("composition_feasibility", "not_checked"),
                         outline_safety_status=state.get("outline_safety_status", "skipped"),
                         **paths,
                         selected_safety_files=self.safety_files,
                         message=("Saved offline outline loaded for review; execution remains disabled."
                                  if outline_only and session.snapshot()["outline_only"] else
                                  "Saved offline outline loaded for review. Continue prepares primitive composition; execution remains disabled."
                                  if outline_only else "Saved offline composition loaded. No LLM call or equipment command was sent."))
        if self._offline is not None:
            self._offline.stop()
        self._offline = session
        return self.snapshot()

    def set_modes(self, outline_mode: str, primitive_mode: str) -> dict:
        """Set stage modes before starting the example or offline generation."""
        if self.source == "offline" and (outline_mode, primitive_mode) != ("manual", "manual"):
            raise ValueError("Offline recovery always uses Manual for both stages")
        state = self.snapshot()
        offline_ready = self.source == "offline" and (state["status"] == "stopped" or
                        (state["status"] in {"outline_ready", "blocked"} and not state.get("outline")))
        if state["busy"] or (state["status"] != "idle" and not offline_ready):
            raise ValueError("Set generation modes before starting the example")
        if outline_mode not in {"auto", "manual"} or primitive_mode not in {"auto", "manual"}:
            raise ValueError("Recovery stages require Auto or Manual")
        self.outline_mode, self.primitive_mode = outline_mode, primitive_mode
        if self.source == "offline" and self._offline is not None and state["status"] != "stopped":
            self._offline.runtime.setup.update(runtime_recovery_mode=outline_mode,
                                                runtime_recovery_primitive_mode=primitive_mode)
            self._offline._publish(outline_mode=outline_mode, primitive_mode=primitive_mode)
        return self.snapshot()

    def start(self) -> dict:
        """Queue a single launch, rejecting concurrent work and stale worlds."""
        bridge = self.bridge
        if self.source != "gazebo":
            raise ValueError("Use Recovery outline Generate for saved observations")
        if self.snapshot()["status"] != "idle" or self.snapshot()["busy"]:
            raise ValueError("The example already started; use Reset example for another run")
        if bridge.system_running or bridge._starting or bridge._stopping:
            raise ValueError(f"Stop the current system before starting {self.scenario}")
        if bridge.get_conveyor_fault().get("status") == "triggered":
            raise ValueError(f"Reset example before starting another {self.scenario} run")
        self._cancelled = False
        self._saved_settings = {
            name: deepcopy(getattr(bridge, name)) for name in (
                "execution_mode", "robot_env", "selected_product", "selected_product_order_file",
                "selected_requirement_file", "selected_safety_file", "runtime_recovery_mode",
                "runtime_recovery_validation_policy", "runtime_recovery_archive_path", "runtime_recovery_archive_label",
            ) if hasattr(bridge, name)
        }
        if hasattr(bridge, "bundle_store"):
            self._saved_bundle = bridge.bundle_store.get_active_bundle_id()
        self._status = {"status": "starting", "message": "Starting Gazebo."}
        self._task = asyncio.create_task(self._start(), name="start_part_slippage")
        self._task.add_done_callback(self._task_finished)
        return self.snapshot()

    def _task_finished(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("Failure example task failed", exc_info=(type(error), error, error.__traceback__))
            if not self._cancelled:
                self._status = {"status": "blocked", "message": f"{type(error).__name__}: {error}"}

    async def _start(self) -> None:  # noqa: C901 - launch ownership and cancellation stay together
        from cais_spade_llm.recovery_framework.delivery import prepare_start
        from cais_spade_llm.recovery_framework.environment_runtime import (
            prepare_environment_start,
            record_environment_startup_ready,
        )

        bridge = self.bridge
        try:
            setup = await asyncio.to_thread(example_setup, outline_mode=self.outline_mode, primitive_mode=self.primitive_mode,
                                            scenario=self.scenario, safety_files=self.safety_files)
            if self._test_proposal is not None:
                setup.update(runtime_recovery_mode="manual", runtime_recovery_primitive_mode="manual")
            if self._cancelled:
                return
            bridge.execution_mode, bridge.robot_env = "simulation", "gazebo"
            bridge.selected_product = setup["selected_product"]
            bridge.selected_product_order_file = setup["selected_product_order_file"]
            bridge.selected_requirement_file = ""
            bridge.selected_safety_file = "__NONE__"
            bridge.set_active_bundle(None)
            bridge.set_runtime_recovery_mode(self.outline_mode)
            bridge.set_runtime_recovery_validation_policy("no_validation")
            bridge.set_runtime_recovery_archive_selection("", "")
            if not await asyncio.to_thread(bridge.simulation_environment_running):
                error = await asyncio.to_thread(bridge.ros2_start, "gazebo_dual")
                if error:
                    raise RuntimeError(error)
            if not bridge._recovery_framework_gazebo():
                raise ValueError(f"{self.scenario} requires the recovery-framework Gazebo scene")
            self._status = {"status": "starting", "message": "Waiting for Gazebo and controller readiness."}
            deadline = time.monotonic() + 180
            while not self._cancelled:
                ready, reason = await asyncio.to_thread(bridge.simulation_start_ready, True)
                if ready:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(reason)
                await asyncio.sleep(1)
            if self._cancelled:
                return
            prepare_start(None)
            self._status = {"status": "starting", "message": f"Preparing {self.scenario} and starting agents."}
            await asyncio.to_thread(
                prepare_environment_start, setup, prewarm_controllers=True,
                launch_identity=bridge._simulation_launch_key(), requested_at_unix=time.time(),
                diagnostic_cca_bypass=True,
            )
            if self._cancelled:
                await asyncio.to_thread(prepare_environment_start, None)
                return
            await bridge.start_system()
            if not bridge.system_running:
                raise RuntimeError(bridge.last_error or "Agent startup did not complete")
            if self._cancelled:
                await bridge.stop_system()
                return
            self._runtime = next((agent.environment_runtime for agent in bridge.product_agents
                                  if getattr(agent, "environment_runtime", None) is not None), None)
            if self._runtime is None or self._runtime.part_slippage_recovery is None:
                raise RuntimeError(f"{self.scenario} recovery was not installed")
            record_environment_startup_ready(bridge)
            self._status = {"status": "running_example", "message": f"Running {self.scenario}."}
            if self._test_proposal is not None:
                await self._wait_and_validate_saved_proposal()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as exc:
            logger.exception("%s startup failed", self.scenario)
            self._status = {"status": "blocked", "message": str(exc)}
            await asyncio.to_thread(prepare_environment_start, None)
            if not bridge.system_running:
                self._restore_settings()
        finally:
            if self._cancelled:
                self._status = {"status": "stopped", "message": "Stopped by operator."}
                self._restore_settings()

    def _restore_settings(self) -> None:
        if not self._saved_settings:
            return
        for name, value in self._saved_settings.items():
            setattr(self.bridge, name, value)
        if hasattr(self.bridge, "bundle_store"):
            self.bridge.bundle_store.set_active_bundle_id(self._saved_bundle)
        self._saved_settings = {}

    async def action(self, action: str, *, feedback: str = "") -> dict:  # noqa: C901, PLR0912 - explicit operator actions
        """Forward an operator action to the owning ProductAgent event loop."""
        if action == "generate_outline_fresh":
            if self.source != "offline" or self.snapshot()["busy"] or not self.observation_path:
                raise ValueError("Select saved observations in Offline and wait for the current action")
            options = (self.observation_path, self.scenario, list(self.safety_files),
                       deepcopy(self.validation_checks or dict.fromkeys(CHECK_FIELDS, True)))
            self._fresh_cancel = threading.Event()
            self._draft = self._offline = None
            self._status = {"status": "preparing_outline", "message": "Preparing a fresh outline from the verified capture."}
            self._task = asyncio.create_task(self._generate_outline_fresh(*options), name="generate_outline_fresh")
            self._task.add_done_callback(self._task_finished)
            return self.snapshot()
        if action == "stop" and self._task is not None and not self._task.done() and self._task.get_name() == "generate_outline_fresh":
            self._fresh_cancel.set()
            if self._draft is not None:
                self._draft.stop()
            if self._offline is not None:
                self._offline.stop()
            self._status = {"status": "stopped", "message": "Stopping the current attempt; captured responses retained."}
            return self.snapshot()
        if action == "start_fresh":
            if self.source != "offline" or self.snapshot()["busy"]:
                raise ValueError("Select Offline and wait for the current action before starting fresh")
            self._draft = None
            if self._offline is not None:
                self._offline.stop("Starting a fresh outline attempt")
            self._offline = None
            self._status = {"status": "idle", "message": "Preparing a fresh outline from the original failure capture."}
            return self._new_offline(generate_outline=False)
        if self.source == "offline" and action in {"generate_outline_only", "load_draft"}:
            from .outline_draft import OutlineDraft, latest_draft, load_draft

            if self.snapshot()["busy"]:
                raise ValueError("Wait for the current action to finish")
            observations = read_observations(self.observation_path)
            path = latest_draft(observations)
            previous = load_draft(path, observations) if path is not None else None
            settings = previous.get("llm_settings", {}) if previous else {}
            draft = OutlineDraft(observations, self.safety_files,
                model=settings.get("model"), reasoning_effort=settings.get("reasoning_effort"))
            if action == "load_draft":
                if previous is None:
                    raise ValueError("No complete draft for these saved observations")
                draft.record = previous
                draft.directory = path.parent
                self._saved_draft_path = path
            else:
                async def generate_draft() -> None:
                    await draft.generate()
                    if draft.record["status"] == "draft":
                        self._saved_draft_path = draft.directory / "draft.json"

                draft._task = asyncio.create_task(generate_draft(), name="generate_outline_only")
                draft._task.add_done_callback(self._task_finished)
            self._draft = draft
            return self.snapshot()
        if self._draft is not None:
            if action == "generate_outline" and self._draft.record["status"] == "draft_ready":
                self._draft._task = asyncio.create_task(self._draft.generate(), name="generate_outline")
                self._draft._task.add_done_callback(self._task_finished)
                return self.snapshot()
            if action == "stop":
                self._draft.stop()
                return self.snapshot()
            if action == "reset" and not self.snapshot()["busy"]:
                self._draft = None
            else:
                raise ValueError("Unvalidated drafts are for review only; reset before validated recovery")
        if action != "stop" and self._preview_task is not None and not self._preview_task.done():
            raise ValueError("A recovery action is already running")
        if action == "prepare_gazebo_evidence":
            return self._prepare_gazebo_evidence()
        if action == "use_new_observations":
            return self._use_new_observations()
        if action == "stop" and self._preview_task is not None and not self._preview_task.done():
            self._preview_task.cancel()
            try:
                await self._preview_task
            except asyncio.CancelledError:
                pass
        if action == "test_in_gazebo":
            return await self._test_in_gazebo()
        if self.source == "offline":
            if action == "reset":
                if self._task is not None and not self._task.done():
                    raise ValueError("Wait for the stopped operation to finish")
                previous = self._offline
                if previous is not None:
                    previous.stop()
                    if previous._task is not None and not previous._task.done():
                        self._task = asyncio.create_task(self._drain_offline(previous),
                                                         name="drain_part_slippage_offline")
                        self._task.add_done_callback(self._task_finished)
                        await asyncio.shield(self._task)
                self._offline = None
                self._status = {"status": "idle", "message": "Offline example reset. Saved observations retained."}
                if self.observation_path:
                    return self.prepare_offline()
            elif action == "stop":
                if self._offline is not None:
                    self._offline.stop()
                    self._offline._publish(status="stopped", message="Offline attempt stopped. Generate starts a fresh attempt.")
                self._status = {"status": "stopped", "message": "Offline attempt stopped. Generate starts a fresh attempt."}
            elif action == "prepare":
                return self.prepare_offline()
            elif action == "generate_outline" and self.snapshot()["status"] == "stopped":
                if self.snapshot()["busy"]:
                    raise ValueError("Wait for the stopped operation to finish")
                return self._new_offline(generate_outline=True)
            elif self._offline is None:
                raise ValueError("Select saved observations and prepare the recovery input first")
            else:
                self._offline.action(action, feedback=feedback)
                if action in {"generate_outline", "regenerate_outline", "continue_outline"}:
                    self._offline._publish(validation_checks_locked=True)
            return self.snapshot()
        if action == "stop":
            self._cancelled = True
            if self._runtime is not None:
                async def stop_runtime() -> None:
                    self._runtime.stop()
                    await self._runtime.cancel_owned()
                await self.bridge._run_on_agent_runtime(stop_runtime())
            await self.bridge.stop_system()
            if not self.snapshot()["busy"]:
                self._restore_settings()
            self._status = {"status": "stopped", "message": "Stopped by operator."}
        elif action == "reset":
            if self.snapshot()["busy"]:
                raise ValueError("Wait for startup or reset to finish")
            if self.bridge.system_running and self._runtime is None:
                raise ValueError("Stop the current system before resetting this example")
            self._status = {"status": "resetting", "message": "Resetting the Gazebo example."}
            self._task = asyncio.create_task(self._reset(), name="reset_part_slippage")
            self._task.add_done_callback(self._task_finished)
        else:
            if self._runtime is None or self._cancelled or not self.bridge.system_running:
                raise ValueError(f"Start {self.scenario} before generating recovery")
            if self.bridge._simulation_launch_key() != self._runtime.launch_identity:
                raise ValueError("The Gazebo launch changed; Reset example before continuing")
            async def dispatch() -> dict:
                return self._runtime.part_slippage_recovery.action(action, feedback=feedback)
            await self.bridge._run_on_agent_runtime(dispatch())
            self._status = {"status": "running_example", "message": "Recovery action queued."}
        return self.snapshot()

    async def _fresh_work(self, operation):
        """Drain private preparation/provider work before releasing its owner."""
        worker = asyncio.create_task(asyncio.to_thread(operation))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            self._fresh_cancel.set()
            if self._draft is not None:
                self._draft.stop()
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                    break
            if not worker.cancelled():
                worker.exception()
            raise

    async def _generate_outline_fresh(self, path: str, scenario: str, safety_files: list[str], choices: dict[str, bool]) -> None:
        from .offline_recovery import create_offline_recovery
        from .outline_draft import OutlineDraft

        def prepare():
            observations = read_observations(path)
            if observations.get("scenario") != scenario:
                raise ValueError("Saved observations belong to a different failure scenario")
            summary = observation_summary(observations)
            if self._fresh_cancel.is_set():
                return observations, summary, None
            if not all(choices.values()):
                owner = OutlineDraft(observations, safety_files, **choices)
                owner._save()
            else:
                owner = create_offline_recovery(observations, outline_mode="manual", primitive_mode="manual",
                                                source_observation_path=path)
                owner.runtime.setup["selected_safety_files"] = safety_files
                owner._publish(selected_safety_files=safety_files, validation_checks=choices, validation_checks_locked=True)
            return observations, summary, owner

        try:
            observations, summary, owner = await self._fresh_work(prepare)
            self._selected_observations, self._selected_observation_summary = observations, summary
            if self._fresh_cancel.is_set():
                if owner is not None:
                    owner.stop()
                return
            if not all(choices.values()):
                self._draft = owner
                owner._task = self._task
                await self._fresh_work(lambda: asyncio.run(owner.generate()))
            else:
                self._offline = owner
                owner.begin(observations["failure"], generate_outline=True, defer_observation_save=True)
                await owner._task
        except asyncio.CancelledError:
            self._status = {"status": "stopped", "message": "Stopped; saved observations and responses retained."}
            raise
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            self._status = {"status": "NEEDS_CONTEXT" if isinstance(exc, ValueError) and str(exc).startswith("NEEDS_CONTEXT:")
                            else "blocked", "message": str(exc)}
            logger.warning("Fresh recovery outline could not start: %s", exc)

    def _prepare_gazebo_evidence(self) -> dict:
        """Queue an explicit preview while leaving generation and execution idle."""
        if self.snapshot()["busy"]:
            raise ValueError("A recovery action is already running")
        recovery = self._offline if self.source == "offline" else getattr(self._runtime, "part_slippage_recovery", None)
        if recovery is None or not recovery.snapshot().get("can_prepare_gazebo_evidence"):
            raise ValueError("Gazebo evidence requires an unresolved outline proposal")
        runtime = self._runtime
        if (runtime is None or not self.bridge.system_running
                or self.bridge._simulation_launch_key() != runtime.launch_identity):
            raise ValueError("Use Start failure to obtain an owned paused Gazebo checkpoint")

        async def prepare() -> None:
            from cais_spade_llm.recovery_framework.outline_gazebo_evidence import (
                prepare_gazebo_evidence,
            )

            recovery._publish(message="Preparing Gazebo evidence without executing recovery.")
            cancelled = threading.Event()
            dispatch = asyncio.create_task(self.bridge._run_on_agent_runtime(
                prepare_gazebo_evidence(recovery, runtime, cancel_event=cancelled)))
            try:
                report = await asyncio.shield(dispatch)
                if report["status"] == "new_observations_required":
                    recovery._publish(new_observation_path=report["observation_path"],
                        checkpoint_differences=report["checkpoint_differences"],
                        message="The saved checkpoint cannot be reused. Use new observations, then Generate.")
                elif report.get("evidence"):
                    if recovery.offline:
                        await recovery.retry_outline_validation(report["evidence"])
                    else:
                        await self.bridge._run_on_agent_runtime(recovery.retry_outline_validation(report["evidence"]))
                else:
                    recovery._publish(message=report.get("reason", "NEEDS_CONTEXT: owner preview is unavailable"))
            except asyncio.CancelledError:
                cancelled.set()
                try:
                    await dispatch
                except asyncio.CancelledError:
                    pass
                except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as exc:
                    logger.warning("Gazebo preview stopped with an evidence error: %s", exc)
                recovery._publish(message="Gazebo preview cancelled; saved evidence retained.")
                raise
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as exc:
                recovery._publish(status="blocked", message=str(exc))

        self._preview_task = asyncio.create_task(prepare(), name="prepare_gazebo_outline_evidence")
        return self.snapshot()

    def _use_new_observations(self) -> dict:
        """Prepare a fresh Manual attempt without rewriting its source capture."""
        if self.snapshot()["busy"]:
            raise ValueError("Wait for the current recovery action to finish")
        recovery = self._offline if self.source == "offline" else getattr(self._runtime, "part_slippage_recovery", None)
        path = recovery.snapshot().get("new_observation_path") if recovery is not None else None
        if not path:
            raise ValueError("No fresh observations are available")
        from uuid import uuid4

        from cais_spade_llm.recovery_framework.capture_files import write_capture

        observations = read_observations(path)
        saved_path = recovery.path.parent / f"capture_{uuid4().hex}" / "observations.json"
        write_capture(saved_path, {key: value for key, value in observations.items()
                                   if key not in {"source_path", "source_sha256"}})
        if self.source != "offline":
            self._gazebo_modes = (self.outline_mode, self.primitive_mode)
        self.source = "offline"
        self.outline_mode = self.primitive_mode = "manual"
        self.observation_path = str(saved_path)
        self._selected_observations = observations
        return self._new_offline(generate_outline=False)

    async def _test_in_gazebo(self) -> dict:
        if self.scenario != "Part slippage":
            raise ValueError("This failure scenario stops at Recovery outline review")
        if (self.source != "offline" or self._offline is None
                or self.snapshot()["busy"] or self._offline.snapshot()["status"] != "offline_completed"):
            raise ValueError("Complete the offline primitive composition first")
        if (self._offline._proposal.get("validation_contract_version") == 2
                or self._offline._proposal.get("composition_validation_version") == 1):
            raise ValueError("Outline validation does not authorize execution; runtime CCA integration is pending")
        self._test_proposal = deepcopy(self._offline._proposal)
        self._test_observations = deepcopy(self._offline._observations)
        self._test_source = self._offline.snapshot()
        self.source, self.scenario = "gazebo", "Part slippage"
        runtime = self._runtime
        if (runtime is not None and self.bridge.system_running
                and self.bridge._simulation_launch_key() == runtime.launch_identity
                and runtime.conveyor_fault.status == "triggered"):
            self._status = {"status": "running_example", "message": "Checking saved composition against Gazebo."}
            self._task = asyncio.create_task(self._wait_and_validate_saved_proposal(), name="check_saved_recovery")
            self._task.add_done_callback(self._task_finished)
        else:
            if self.bridge.system_running:
                self.source = "offline"
                raise ValueError("Stop the current system before testing the saved composition")
            self._status = {"status": "idle", "message": "Starting the selected failure for the saved composition."}
            if runtime is not None:
                await self._reset()
                if self._status["status"] != "idle":
                    return self.snapshot()
            self.start()
        return self.snapshot()

    async def _wait_and_validate_saved_proposal(self) -> None:
        try:
            deadline = time.monotonic() + 600
            while not self._cancelled:
                recovery = self._runtime.part_slippage_recovery
                state = recovery.snapshot()
                if state["status"] == "blocked":
                    raise ValueError(state["message"])
                if (state["status"] in {"outline_ready", "outline_review", "primitive_ready", "ready_to_execute"}
                        and (recovery._task is None or recovery._task.done())):
                    async def validate(recovery=recovery) -> None:
                        from cais_spade_llm.recovery_framework.saved_recovery_execution import (
                            validate_saved_proposal,
                        )

                        recovery._task = asyncio.current_task()
                        await validate_saved_proposal(recovery, self._test_proposal, self._test_observations,
                                                      source_state=self._test_source)
                    await self.bridge._run_on_agent_runtime(validate())
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError("The failure observations are not ready for the saved composition")
                await asyncio.sleep(1)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as exc:
            logger.exception("Saved recovery validation failed")
            self._status = {"status": "blocked", "message": str(exc)}
        finally:
            self._test_proposal = None
            self._test_observations = None
            self._test_source = None

    async def _reset(self) -> None:
        try:
            await self.action("stop")
            self._status = {"status": "resetting", "message": "Recreating the Gazebo scene."}
            if self._runtime is not None:
                from cais_spade_llm.recovery_framework.conveyor_fault import reset_fault_scene

                ok, reason = await asyncio.to_thread(reset_fault_scene, self.bridge, self._runtime)
                if not ok:
                    raise RuntimeError(reason)
            elif await asyncio.to_thread(self.bridge.simulation_environment_running):
                if not self.bridge._recovery_framework_gazebo():
                    raise ValueError("The running Gazebo scene is not owned by this example")
                error = await asyncio.to_thread(self.bridge.ros2_stop, "gazebo_dual")
                if error:
                    raise RuntimeError(error)
            self._runtime = None
            self._cancelled = False
            self._status = {"status": "idle", "message": "Example reset. Previous records are retained."}
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            logger.exception("%s reset failed", self.scenario)
            self._status = {"status": "blocked", "message": str(exc)}
        finally:
            if not self.bridge.system_running:
                self._restore_settings()


def example_for_bridge(bridge: Any) -> PartSlippageExample:
    """Return the bridge-owned example, creating its idle state only once."""
    example = getattr(bridge, "_part_slippage_example", None)
    if example is None:
        example = PartSlippageExample(bridge)
        bridge._part_slippage_example = example
    return example

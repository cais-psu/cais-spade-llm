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
from cais_spade_llm.recovery_framework.recovery_observations import (
    RECORDS,
    list_observations,
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

# Schema-2 captures of the corrected resource models (2026-10-22/23); the schema-1 originals stay
# selectable as regression fixtures.
_DEFAULT_OBSERVATIONS = {
    "Part slippage": "capture_ad64bf1628564c3fa7f928b7bdebd5d9",
    "Machining breakdown during part processing": "capture_37e090a54d0242f4948ae80c95d3cd76",
    "Conveyor breakdown": "capture_3ad495d2bd7845269fa70f1a09963bc9",
    "ur5e-1 breakdown": "capture_b7dea70c1db74e1e9e98849380c09655",
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
        if scenario == "Conveyor breakdown":
            setup["permitted_resources"] = [rid for rid in setup["permitted_resources"] if rid not in {"M2", "ur5e-2"}]
        setup["failure_scenario"] = {"scenario": scenario, "resource_id": resource_id,
                                     "checkpoint": CHECKPOINTS[scenario], "mode": "once"}
        if scenario == "Conveyor breakdown":
            # No held peer: gear_small precedes KET4 on the board, so nothing would make ur5e-4 wait.
            setup["failure_scenario"]["part_name"] = "KET4_Square_4mm"
        if scenario in {"Machining breakdown during part processing", "ur5e-1 breakdown"}:
            setup["selected_product_order_file"] = "cais_spade_llm/specification/products/orders/assembly_board-v1-machine-breakdown.json"
        if scenario == "Machining breakdown during part processing":
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
        from cais_spade_llm.recovery_framework.recovery_case_safety import default_safety_files

        self.scenario = "Part slippage"
        self.safety_files: list[str] = default_safety_files(self.scenario)
        self.observation_path = ""
        self._defer_preparation = False
        self._selection_revision = 0
        self._selected_observation_summary: dict | None = None
        self._fresh_cancel = threading.Event()
        self._selected_observations: dict = {}
        self._draft = None
        self._saved_draft_path: Path | None = None
        self._task: asyncio.Task | None = None
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
            "defer_preparation": self._defer_preparation,
            "selection_revision": self._selection_revision,
        }
        runtime = self._runtime
        if self.source == "gazebo" and runtime is not None and self._status["status"] in {"running_example", "blocked", "stopped"}:
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
        session = None if self.source == "offline" else getattr(runtime, "part_slippage_recovery", None)
        state["busy"] = bool((self._task is not None and not self._task.done())
                             or (session is not None and session._task is not None and not session._task.done()))
        if self.source == "offline" and self._selected_observations:
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
            if self._defer_preparation:
                state["can_generate_outline"] = bool(self.observation_path and not state["busy"])
                state["outline_generate_reason"] = ("Generation already running." if state["busy"] else
                    "Select saved observations to prepare the recovery input." if not self.observation_path else "")
        state["generating_stage"] = self._generating_stage(session)
        return state

    def _generating_stage(self, session) -> str:
        """Name the stage a stage-level Stop can cancel, or "" once a stop is requested."""
        if self._task is not None and not self._task.done() and self._task.get_name() == "generate_outline_fresh":
            return "" if self._fresh_cancel.is_set() else "outline"
        if self.source == "offline" and self._draft is not None:
            running = self._draft._task is not None and not self._draft._task.done()
            return "outline" if running and not self._draft._stopped else ""
        return ""

    def set_options(self, *, source: str, scenario: str, safety_files: list[str], observation_path: str = "",
                    defer_preparation: bool = False) -> dict:
        """Select the next input source without launching generation.

        Args:
            source: Existing Gazebo or Offline source selection.
            scenario: Exact failure scenario.
            safety_files: Selected safety specification paths.
            observation_path: Original saved failure observations.
            defer_preparation: Store preferences without reading observations or preparing an attempt.

        Returns:
            The current workflow state.
        """
        state = self.snapshot()
        if type(defer_preparation) is not bool:
            raise ValueError("defer_preparation must be a boolean")
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
        changed = input_changed or defer_preparation != self._defer_preparation
        if source != self.source:
            if source == "offline":
                self._gazebo_modes = (self.outline_mode, self.primitive_mode)
                self.outline_mode = self.primitive_mode = "manual"
            else:
                self.outline_mode, self.primitive_mode = self._gazebo_modes
        self.source, self.scenario = source, scenario
        self.safety_files, self.observation_path = list(safety_files), observation_path
        self._selected_observations = selected
        self._defer_preparation = defer_preparation
        if changed:
            self._selection_revision += 1
        if input_changed or not defer_preparation:
            self._selected_observation_summary = None
        if defer_preparation:
            if input_changed:
                self._draft = None
            self._saved_draft_path = None
            if input_changed or self._draft is None:
                self._status = {"status": "idle", "message": "Generate a fresh outline from the verified capture."}
            return self.snapshot()
        if changed:
            self._draft = None
            self._saved_draft_path = None
            if source == "offline" and selected:
                from .outline_draft import latest_draft

                self._saved_draft_path = latest_draft(selected)
            self._status = {"status": "idle", "message": "Ready to start the selected example."}
            if source == "gazebo" and self._runtime is not None and self.bridge.system_running:
                self._status = {"status": "running_example", "message": "Reconnected to the paused failure."}
        return self.snapshot()

    def saved_observations(self) -> list[dict]:
        """List retained observation sources without querying Gazebo."""
        return list_observations()

    def prepare_offline(self) -> dict:
        """Prepare the selected immutable capture without calling the LLM."""
        if self.source != "offline":
            raise ValueError("Select Offline before preparing saved observations")
        state = self.snapshot()
        if self._draft is not None or state["busy"]:
            return state
        return self._new_draft()

    def _new_draft(self) -> dict:
        from .outline_draft import OutlineDraft

        if not self.observation_path:
            raise ValueError("Select saved observations first")
        try:
            observations = read_observations(self.observation_path)
            if observations.get("scenario") != self.scenario:
                raise ValueError("Saved observations belong to a different failure scenario")
            self._selected_observations = observations
            self._selected_observation_summary = observation_summary(observations)
            self._draft = OutlineDraft(observations, self.safety_files)
            self._draft._save()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._draft = None
            self._status = {"status": "blocked", "message": str(exc)}
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
        if action == "stop_generation":
            return await self._stop_generation()
        if action == "generate_outline_fresh":
            if self.source != "offline" or self.snapshot()["busy"] or not self.observation_path:
                raise ValueError("Select saved observations in Offline and wait for the current action")
            options = (self.observation_path, self.scenario, list(self.safety_files))
            self._fresh_cancel = threading.Event()
            self._draft = None
            self._status = {"status": "preparing_outline", "message": "Preparing a fresh outline from the verified capture."}
            self._task = asyncio.create_task(self._generate_outline_fresh(*options), name="generate_outline_fresh")
            self._task.add_done_callback(self._task_finished)
            return self.snapshot()
        if action == "stop" and self._task is not None and not self._task.done() and self._task.get_name() == "generate_outline_fresh":
            self._fresh_cancel.set()
            if self._draft is not None:
                self._draft.stop()
            self._status = {"status": "stopped", "message": "Stopping the current attempt; captured responses retained."}
            return self.snapshot()
        if action == "start_fresh":
            if self.source != "offline" or self.snapshot()["busy"]:
                raise ValueError("Select Offline and wait for the current action before starting fresh")
            self._draft = None
            self._status = {"status": "idle", "message": "Preparing a fresh outline from the original failure capture."}
            return self._new_draft()
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
                raise ValueError("Unvalidated drafts are for review only; reset before another attempt")
        if self.source == "offline":
            if action == "reset":
                if self._task is not None and not self._task.done():
                    raise ValueError("Wait for the stopped operation to finish")
                self._status = {"status": "idle", "message": "Offline example reset. Saved observations retained."}
                if self.observation_path:
                    return self.prepare_offline()
            elif action == "stop":
                self._status = {"status": "stopped", "message": "Offline attempt stopped. Generate starts a fresh attempt."}
            elif action in {"prepare", "generate_outline"}:
                if self.snapshot()["busy"]:
                    raise ValueError("Wait for the stopped operation to finish")
                return self.prepare_offline()
            else:
                raise ValueError("Select saved observations and prepare the recovery input first")
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

    async def _stop_generation(self) -> dict:
        """Stop the running outline draft; only Offline drafts generate outlines."""
        if not self._generating_stage(None):
            raise ValueError("No recovery generation is running")
        # Offline Generate always starts a fresh outline, so the full attempt stop applies.
        return await self.action("stop")

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

    async def _generate_outline_fresh(self, path: str, scenario: str, safety_files: list[str]) -> None:
        from .outline_draft import OutlineDraft

        def prepare():
            observations = read_observations(path)
            if observations.get("scenario") != scenario:
                raise ValueError("Saved observations belong to a different failure scenario")
            summary = observation_summary(observations)
            if self._fresh_cancel.is_set():
                return observations, summary, None
            owner = OutlineDraft(observations, safety_files)
            owner._save()
            return observations, summary, owner

        try:
            observations, summary, owner = await self._fresh_work(prepare)
            self._selected_observations, self._selected_observation_summary = observations, summary
            if self._fresh_cancel.is_set():
                if owner is not None:
                    owner.stop()
                return
            self._draft = owner
            owner._task = self._task
            await self._fresh_work(lambda: asyncio.run(owner.generate()))
        except asyncio.CancelledError:
            self._status = {"status": "stopped", "message": "Stopped; saved observations and responses retained."}
            raise
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            self._status = {"status": "NEEDS_CONTEXT" if isinstance(exc, ValueError) and str(exc).startswith("NEEDS_CONTEXT:")
                            else "blocked", "message": str(exc)}
            logger.warning("Fresh recovery outline could not start: %s", exc)

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

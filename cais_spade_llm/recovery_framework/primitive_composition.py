from __future__ import annotations

"""Readiness and immutable source handling for Recovery UI composition attempts."""

import asyncio
import logging
import shutil
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)
from cais_spade_llm.recovery_framework.recovery_observations import RECORDS

logger = logging.getLogger(__name__)

COMPOSITIONS = RECORDS / "draft_compositions"
# The outline checks composition needs: RA feasibility (Alg. 1, Eq. 12), CCA safety (Eq. 13) and CCA
# nonblocking (Eq. 6). ``outline_diagnostic_bypass`` only selects the frozen outline prompt.
_OUTLINE_CHECKS = ("check_ra_feasibility", "check_cca_safety", "check_nonblocking")
# The same limits as the Recovery session's primitive stage.
_COMPOSITION_TURN_BUDGET = 80
_CONTINUING_STATUSES = {"paused_after_outline_turn", "ready_for_primitive_generation", "paused_after_primitive_turn",
                        "paused_after_primitive_generation"}


def composition_readiness(prepared: dict) -> dict:
    """Check accepted outline coverage and recorded owner contracts without calls."""
    session = prepared.get("multi_turn_session_state", {})
    if not isinstance(session, dict):
        session = {}
    events = session.get("accepted_outline_prefix", [])
    if not isinstance(events, list) or any(not isinstance(row, dict) for row in events):
        events = []
    missing = []
    if any(session.get(field) is not True for field in _OUTLINE_CHECKS):
        missing.append("The outline must pass the RA feasibility, CCA safety and CCA nonblocking checks; "
                       "full validation is required before composition")
    ids = [row.get("outline_id") for row in events]
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        missing.append("A complete accepted outline with unique event identities is required")
    if session.get("current_phase") not in {"primitive_generation", "finalize"}:
        missing.append("The outline has not completed validation and nominal continuation")
    for event in events:
        jid = event.get("resource_jid")
        entry = prepared.get("recovery_resources", {}).get(jid, {})
        if not entry.get("execution_primitive_catalog") or not entry.get("recovery_des_model", {}).get("state_variables"):
            missing.append(f"{jid}: owner primitive contracts or state declarations are missing")
    return {"ready": not missing, "feasibility_status": "NEEDS_CONTEXT" if missing else "not_checked",
            "reasons": list(dict.fromkeys(missing)), "execution_authorized": False}


def prepare_composition_attempt(source: dict, directory: Path, session_id: str, *, feedback: str = "") -> dict:
    """Copy a saved outline and its proof into a separate attempt directory.

    Args:
        source: The accepted outline request; never modified.
        directory: The new attempt's directory.
        session_id: The attempt's recovery session identifier.
        feedback: Optional operator feedback for the composition prompt.

    Returns:
        The attempt's request, its planning state at the start of primitive generation.
    """
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import (
        _build_phase_prompt,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (
        _nominal_artifact_sha256,
        nominal_continuation_context_binding,
        read_nominal_continuation,
    )
    from cais_spade_llm.agents.shared_information.llm_request_records import (
        recovery_request_directory,
    )
    from cais_spade_llm.recovery_framework.diagnostics import write_record

    source_stage = recovery_request_directory(source, "outline")
    source_state = source["multi_turn_session_state"]
    nominal = source_state.get("nominal_continuation_evidence", {})
    binding = nominal_continuation_context_binding(source_state, source) if "artifact_reference" in nominal else None
    if binding is not None:
        read_nominal_continuation(nominal, directory=source_stage, context_binding=binding)
        relative = nominal["artifact_reference"]["path"]
        destination = directory / "recovery_outline" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_stage / relative, destination)
        if _nominal_artifact_sha256(destination) != nominal["artifact_reference"]["sha256"]:
            raise ValueError("Copied nominal continuation differs from its verified source")
    # Verify the large source proof before allocating the composition copy.
    # Completed-outline geometry stays in outline_request.json; primitive
    # owners use recorded physical evidence plus accepted primitive effects.
    prepared = deepcopy({key: value for key, value in source.items()
                         if key not in {"multi_turn_session_state", "recovery_debug"}})
    prepared["recovery_debug"] = deepcopy({key: value for key, value in source.get("recovery_debug", {}).items()
                                           if key != "multi_turn_session"})
    state = deepcopy({key: value for key, value in source_state.items()
                      if key not in {"turns", "projected_physical_evidence", "outline_monitor_context"}})
    prepared["multi_turn_session_state"] = state
    prepared.update(composition_validation_version=1, recovery_feedback=feedback,
                    composition_source_fingerprint=recovery_validation_fingerprint(source))
    state.update(composition_validation_version=1, current_phase="primitive_generation",
        status="ready_for_primitive_generation", accepted_primitive_program=[], primitive_generation_cursor=0,
        primitive_rejection_feedback=[], primitive_served_context={}, primitive_context_errors=[],
        primitive_event_guard={}, primitive_authoring_memo=[])
    state.pop("final_output", None)
    state.pop("proposal", None)
    # Historical validation and provider records remain in the immutable source.
    # A composition attempt records only its own turns in its active checkpoint.
    state["turns"] = []
    prepared.setdefault("recovery_session", {})["session_id"] = session_id
    state["recovery_session_id"] = session_id
    prepared.setdefault("recovery_debug", {}).update(artifact_directory=str(directory), per_turn_debug_dir=str(directory))
    write_record(directory / "outline_request.json", source)
    write_record(directory / "outline.json", {"outline": deepcopy(state["accepted_outline_prefix"])})
    write_record(directory / "composition_source.json", {"composition_validation_version": 1,
        "source_fingerprint": prepared["composition_source_fingerprint"],
        "source_outline_directory": str(Path(source_stage).parent), "execution_authorized": False})
    preview, prompt = _build_phase_prompt(prepared, state)
    write_record(directory / "primitive_input_preview.json", {
        "request_sent": False, "preview_status": "not sent", "prompt_input": preview, "prompt": prompt})
    return prepared


async def start_composition_attempt(session, *, feedback: str = "") -> None:
    """Copy the saved outline and proof into a separate versioned attempt."""
    source = session._outline_request
    readiness = composition_readiness(source)
    if not readiness["ready"]:
        raise ValueError("NEEDS_CONTEXT: " + "; ".join(readiness["reasons"]))
    directory = session._new_artifact_directory()
    session_id = f"{session.runtime.context.run_id}_{session._generation:04d}"
    session._publish(primitive_status="preparing", execution_authorized=False,
                     message="Preparing a separate primitive composition attempt from the saved outline.")
    prepared = await session._offline_work(
        lambda: prepare_composition_attempt(source, directory, session_id, feedback=feedback))
    session._prepared, session._proposal = prepared, {}
    session._imported_proposal = session._imported_observations = None
    session._publish(composition_validation_version=1, composition_feasibility="not_checked",
        composition_readiness=readiness, composition_findings=[], primitives=[], execution_authorized=False,
        primitive_artifact_directory=str(directory / "recovery_primitves"))


def replay_checked_draft(draft, observations: dict) -> tuple[dict, str]:
    """Rebuild a finished checked draft's planning state by replaying its events with the current checks.

    A draft records its events and check results, not the planning state composition starts from. Each
    event goes through the step its checks took during generation (``check_outline``), with no model call,
    and the end of the outline is checked as the declared completion.

    Args:
        draft: The loaded ``OutlineDraft``, built with the draft's recorded checks and assumptions.
        observations: The draft's saved failure input.

    Returns:
        The composition source and "", or an empty dict and why the current code does not reproduce the
        draft. The source is the draft's request with its planning state at primitive generation and each
        resource's recorded primitive contracts.
    """
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_validated_event import (
        check_outline,
    )

    events = deepcopy(draft.record.get("transition_trace") or [])
    session = multi_turn.build_multi_turn_session_seed(draft.prepared)
    result = asyncio.run(check_outline(deepcopy(events), session, draft.prepared,
                                       SimpleNamespace(product_agent=draft.provider)))
    findings = "; ".join(row.get("reason", "") for row in result["audit"].get("validation_findings") or [])
    if result["refused_event_index"] is not None:
        return {}, (f"Replayed with the current checks, event {result['refused_event_index'] + 1} is refused: "
                    f"{findings}")
    if result["decision"] != "outline_ready":
        return {}, f"Replayed with the current checks, the declared completion is refused: {findings}"
    replayed = result["session"].get("accepted_outline_prefix") or []
    changed = [index + 1 for index, (saved, again) in enumerate(zip(events, replayed, strict=False))
               if any(saved.get(key) != again.get(key) for key in ("expected_start_state", "condition_changes"))]
    if changed or len(replayed) != len(events):
        return {}, ("Replayed with the current checks, the planning state differs from the saved draft at event "
                    + ", ".join(str(index) for index in changed or [min(len(events), len(replayed)) + 1]))
    from cais_spade_llm.resources.resource_primitives import current_primitive_declarations

    source = deepcopy(draft.prepared)
    # The live loop takes this transition when the outline is declared complete.
    source["multi_turn_session_state"] = {**result["session"], "current_phase": "primitive_generation"}
    recorded = observations["prepared_recovery_request"]["recovery_resources"]
    declarations = current_primitive_declarations(observations["inputs"]["scene"])
    rids = {jid: rid for rid, jid in observations["jids"].items()}
    # The outline input leaves out procedures and owner evidence: composition authors from each owner's
    # primitives, and each owner checks its program against its own recorded evidence.
    for jid, entry in source["recovery_resources"].items():
        owner = recorded.get(jid) or {}
        entry["execution_primitive_catalog"] = _declared_catalog(owner.get("execution_primitive_catalog") or [],
                                                                 declarations.get(rids.get(jid), {}))
        evidence = owner.get("resource_evidence") or (observations.get("resource_evidence") or {}).get(jid)
        if evidence:
            entry["resource_evidence"] = deepcopy(evidence)
    source["physical_evidence"] = _composition_physical_evidence(observations, source)
    return source, ""


# What a primitive needs and changes; everything else in a catalog row is its executor interface.
_CONTRACT_KEYS = ("preconditions", "effects", "capability_constraints", "product_guards", "product_effects",
                  "parameters", "controllable", "after_sec", "semantic_summary")


def _declared_catalog(recorded: list[dict], declared: dict[str, list[dict]]) -> list[dict]:
    """Give each recorded primitive the contracts the current code declares for it (paper assumption A.2).

    A capture records each primitive's executor interface (parameters, outputs, description) as its program
    revision has it. What a primitive needs and changes is read from the code at validation time, as the RA
    reads it, one row per alternative contract. A primitive the code does not offer for recovery (an internal
    gripper control) has no row. An uncontrollable outcome, such as a machine's job completion, keeps its declared
    ``controllable`` false: a program waits for it and never commands it. No executor records one, so its row has
    only the parameters its own declared effects read.

    Args:
        recorded: The capture's execution catalog rows for one resource.
        declared: That resource's current declarations, per primitive name.

    Returns:
        The catalog composition validates programs against.
    """
    from cais_spade_llm.resources.resource_primitives import primitive_summary

    def row_for(interface: dict, variant: dict) -> dict:
        entry = {**deepcopy(interface), "preconditions": deepcopy(variant.get("preconditions") or {}),
                 "effects": deepcopy(variant.get("effects") or {}),
                 **{key: deepcopy(variant[key]) for key in ("capability_constraints", "product_guards", "product_effects",
                                                            "parameters", "controllable", "after_sec") if key in variant}}
        entry["semantic_summary"] = primitive_summary(description=str(entry.get("description") or ""),
                                                      preconditions=entry["preconditions"], effects=entry["effects"])
        return entry

    rows = []
    for row in recorded:
        variants = declared.get(row.get("name"))
        if variants is None:
            continue
        interface = {key: deepcopy(value) for key, value in row.items() if key not in _CONTRACT_KEYS}
        # A primitive its executor leaves undeclared needs and changes nothing the checks can read.
        rows.extend(row_for(interface, variant) for variant in variants or [{}])
    named = {row.get("name") for row in recorded}
    for name, variants in declared.items():
        if name in named or not variants or any(variant.get("controllable") is not False for variant in variants):
            continue
        for variant in variants:
            params = sorted(_parameter_slots([variant.get("effects"), variant.get("product_effects")]))
            rows.append(row_for({"name": name, "params": {param: {} for param in params}, "required_params": params},
                                variant))
    return rows


def _parameter_slots(value: Any) -> set[str]:
    """Name the parameters a declared value reads through its ``{"set_from_param": name}`` slots."""
    if isinstance(value, dict) and set(value) == {"set_from_param"} and isinstance(value["set_from_param"], str):
        return {value["set_from_param"]}
    items = value.values() if isinstance(value, dict) else value if isinstance(value, list) else []
    return set().union(*(_parameter_slots(item) for item in items))


def _composition_physical_evidence(observations: dict, prepared: dict) -> dict:
    """Return the capture's physical evidence with what a live capture recorded beside it.

    A raw capture keeps the measured cell. The runtime models, the failure-time resource values and the part
    tracker are rebuilt from its inputs and code grounding, as the CCA nonblocking check rebuilds them; an
    owner reads a receiving resource's declared zones and occupancy from them to complete a placement.
    """
    from cais_spade_llm.product.environment import EnvironmentProductContext

    physical = deepcopy(observations.get("physical_evidence") or {})
    context = EnvironmentProductContext(**observations["inputs"],
                                        permitted_resources=observations["nominal_permitted_resources"])
    grounded = prepared["code_grounding"]["resources"]
    physical["environment_models"] = {rid: deepcopy(resource.model) for rid, resource in context.resources.items()}
    physical["environment_valuation"] = {rid: {field: deepcopy(grounded[rid][field])
                                               for field in resource.model["current_valuation"]}
                                         for rid, resource in context.resources.items()}
    physical["products"] = deepcopy(observations.get("part_tracker") or {})
    return physical


class DraftComposition:
    """Compose primitives for a loaded checked outline draft, offline and for review; nothing executes."""

    def __init__(self, draft, observations: dict, source: dict, reason: str = "", *, root: Path = COMPOSITIONS) -> None:
        """Hold one draft's composition source; each Generate starts a separate attempt.

        Args:
            draft: The loaded ``OutlineDraft``; its record is never changed.
            observations: The draft's saved failure input.
            source: The replayed composition source, or an empty dict.
            reason: Why there is no source.
            root: Directory for the attempts' records.
        """
        self.draft = draft
        self._observations = observations
        self._source = source
        self._root = root
        self._readiness = (composition_readiness(source) if source else
                           {"ready": False, "feasibility_status": "NEEDS_CONTEXT", "reasons": [reason],
                            "execution_authorized": False})
        settings = draft.record.get("llm_settings") or {}
        self._model = settings.get("model")
        self._reasoning_effort = settings.get("reasoning_effort")
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._task: asyncio.Task | None = None
        self._attempt = 0
        self._state = {"primitive_status": "ready" if self._readiness["ready"] else "unavailable",
                       "primitives": [], "composition_findings": [], "composition_feasibility": "not_checked"}

    @property
    def running(self) -> bool:
        """Whether an attempt is composing."""
        return self._task is not None and not self._task.done()

    @property
    def stopping(self) -> bool:
        """Whether a stop was requested for the running attempt."""
        return self.running and self._cancel.is_set()

    def snapshot(self) -> dict:
        """Read the composition progress for the Recovery UI."""
        with self._lock:
            state = deepcopy(self._state)
        state.update(composition_readiness=deepcopy(self._readiness), execution_authorized=False,
                     can_generate_composition=self._readiness["ready"] and not self.running)
        return state

    def start(self, *, feedback: str = "") -> None:
        """Start a composition attempt with the draft's model; it calls the model.

        Args:
            feedback: Optional operator feedback for the composition prompt.
        """
        if not self._readiness["ready"]:
            raise ValueError("NEEDS_CONTEXT: " + "; ".join(self._readiness["reasons"]))
        if self.running:
            raise ValueError("Primitive composition is already running")
        self._cancel = threading.Event()
        self._attempt += 1
        directory = self._root / f"composition_{uuid4().hex}"
        self._set(primitive_status="generating", primitives=[], composition_findings=[],
                  composition_feasibility="in_progress", composition_validation_version=1,
                  composition_directory=str(directory), composition_path="",
                  primitive_artifact_directory=str(directory / "recovery_primitves"),
                  message=f"Composing primitives with {self._model} {self._reasoning_effort or ''}".rstrip() + ".")
        self._task = asyncio.create_task(asyncio.to_thread(self._compose, directory, feedback),
                                         name="compose_outline_draft")
        self._task.add_done_callback(self._finished)

    def stop(self) -> None:
        """Stop the running attempt after its current model call; its records are kept."""
        self._cancel.set()
        self._set(message="Stopping primitive composition after the current LLM call.")

    def _set(self, **values) -> None:
        with self._lock:
            self._state.update(deepcopy(values))

    def _finished(self, task: asyncio.Task) -> None:
        if task.cancelled():
            self._set(primitive_status="stopped", message="Primitive composition stopped; its records are retained.")
            return
        error = task.exception()
        if error is not None:
            logger.error("Draft composition failed", exc_info=(type(error), error, error.__traceback__))
            self._set(primitive_status="failed", composition_feasibility="NEEDS_CONTEXT",
                      message=f"{type(error).__name__}: {error}")

    def _compose(self, directory: Path, feedback: str) -> None:
        from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import (
            build_multi_turn_recovery_proposal,
            execute_multi_turn_recovery,
        )
        from cais_spade_llm.recovery_framework.diagnostics import write_record

        from .offline_recovery import SavedRecoveryProduct, SavedRecoveryResource
        from .recovery_observations import observation_fingerprint

        run_id = self.draft.record["run_id"]
        directory.mkdir(parents=True, exist_ok=True)
        prepared = prepare_composition_attempt(self._source, directory, f"{run_id}_{self._attempt:04d}",
                                               feedback=feedback)
        recorded = self._observations["prepared_recovery_request"]["recovery_resources"]
        # Each recorded owner checks its own events' programs with the RA's composition contract.
        owners = [SavedRecoveryResource(jid, entry, self._observations, _copy_observations=False)
                  for jid, entry in recorded.items()]
        # A provider of its own, so nothing writes to the immutable draft record.
        provider = SavedRecoveryProduct(self._observations, owners, model=self._model,
                                        reasoning_effort=self._reasoning_effort)
        provider._composition_cancel_event = self._cancel
        planner = ProcessPlanner(provider, owners)
        prepared["_stop_after_multi_turn_phase"] = "primitive"
        for _ in range(_COMPOSITION_TURN_BUDGET):
            if self._cancel.is_set():
                self._set(primitive_status="stopped", message="Primitive composition stopped; its records are retained.")
                return
            try:
                asyncio.run(execute_multi_turn_recovery(planner, prepared,
                                                        session_state=prepared.get("multi_turn_session_state")))
            except asyncio.CancelledError:
                self._set(primitive_status="stopped", message="Primitive composition stopped; its records are retained.")
                return
            session = prepared.get("multi_turn_session_state", {})
            findings = deepcopy(session.get("primitive_rejection_feedback", []))
            accepted = session.get("accepted_primitive_program", [])
            self._set(primitives=accepted, composition_findings=findings,
                      composition_feasibility=(findings[0].get("feasibility_status", "NEEDS_CONTEXT") if findings
                                               else "in_progress"),
                      message=f"Validated {len(accepted)} of {len(session.get('accepted_outline_prefix', []))} "
                              "recovery events.")
            if (session.get("final_output") or {}).get("primitive_program_complete") is True:
                break
            if session.get("status") not in _CONTINUING_STATUSES:
                reasons = "; ".join(row.get("reason", "") for row in findings)
                self._set(primitive_status="failed",
                          message=f"Primitive composition stopped at {session.get('status')}: {reasons or 'no completed output'}")
                return
        else:
            self._set(primitive_status="failed",
                      message=f"Primitive composition used its {_COMPOSITION_TURN_BUDGET}-turn budget.")
            return
        adapted = build_multi_turn_recovery_proposal(final_output_payload=session["final_output"],
                                                     prepared_recovery_request=prepared)
        if not adapted.get("accepted"):
            self._set(primitive_status="failed", message=adapted.get("reason") or "Primitive composition is incomplete")
            return
        proposal = {**adapted["recovery_proposal"], "offline_recovery": True,
                    "validation_contract_version": prepared.get("validation_contract_version"),
                    "composition_validation_version": 1, "execution_authorized": False,
                    "source_observation_fingerprint": observation_fingerprint(self._observations),
                    "source_outline_draft": run_id}
        if "primitive_support_semantics" in prepared:
            proposal["primitive_support_semantics"] = prepared["primitive_support_semantics"]
        write_record(directory / "proposal.json", proposal)
        write_record(directory / "composition.json", {
            "composition_validation_version": 1, "source_outline_draft": run_id,
            "source_outline_fingerprint": prepared.get("composition_source_fingerprint"),
            "accepted_outline_prefix": self._source["multi_turn_session_state"]["accepted_outline_prefix"],
            "accepted_primitive_program": session.get("accepted_primitive_program", []),
            "composition_feasibility": "FEASIBLE", "execution_authorized": False,
            "cca_composition_status": "not_checked"})
        self._set(primitive_status="completed", composition_feasibility="FEASIBLE",
                  primitives=session.get("accepted_primitive_program", []),
                  composition_path=str(directory / "composition.json"),
                  message="Composition completed for review. Execution requires the subsequent runtime CCA integration.")

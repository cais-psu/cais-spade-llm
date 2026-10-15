from __future__ import annotations

"""Readiness and immutable source handling for Recovery UI composition attempts."""

from copy import deepcopy
from pathlib import Path
import shutil

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint


def composition_readiness(prepared: dict) -> dict:
    """Check accepted outline coverage and recorded owner contracts without calls."""
    session = prepared.get("multi_turn_session_state", {})
    if not isinstance(session, dict):
        session = {}
    events = session.get("accepted_outline_prefix", [])
    if not isinstance(events, list) or any(not isinstance(row, dict) for row in events):
        events = []
    missing = []
    if session.get("outline_diagnostic_bypass") or any(session.get(field) is False for field in (
        "check_ra_feasibility", "check_physical_evidence", "check_cca_safety")):
        missing.append("An outline with bypassed checks is for review; full validation is required before composition")
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


async def start_composition_attempt(session, *, feedback: str = "") -> None:
    """Copy the saved outline and proof into a separate versioned attempt."""
    from cais_spade_llm.agents.shared_information.llm_request_records import recovery_request_directory
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (
        _nominal_artifact_sha256, nominal_continuation_context_binding, read_nominal_continuation,
    )
    from cais_spade_llm.recovery_framework.diagnostics import write_record

    source = session._outline_request
    readiness = composition_readiness(source)
    if not readiness["ready"]:
        raise ValueError("NEEDS_CONTEXT: " + "; ".join(readiness["reasons"]))
    source_stage = recovery_request_directory(source, "outline")
    directory = session._new_artifact_directory()
    session_id = f"{session.runtime.context.run_id}_{session._generation:04d}"
    session._publish(primitive_status="preparing", execution_authorized=False,
                     message="Preparing a separate primitive composition attempt from the saved outline.")

    def prepare() -> dict:
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn import _build_phase_prompt

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

    prepared = await session._offline_work(prepare)
    session._prepared, session._proposal = prepared, {}
    session._imported_proposal = session._imported_observations = None
    session._publish(composition_validation_version=1, composition_feasibility="not_checked",
        composition_readiness=readiness, composition_findings=[], primitives=[], execution_authorized=False,
        primitive_artifact_directory=str(directory / "recovery_primitves"))

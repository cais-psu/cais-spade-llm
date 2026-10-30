from __future__ import annotations

"""Audit a finished recovery outline with the RA and CCA checks, whatever configuration produced it.

Every finished outline is replayed from its capture's failure state through the checks the validated
configurations use: RA feasibility (Alg. 1, Eq. 12) and CCA safety (Eq. 13) for each event in order, then
the CCA's nonblocking check (Eq. 6) at the outline's end. Nothing goes back to any model. The attempt is
classified by the first failing check of its first failing event, in the fixed order RA feasibility,
physical evidence, CCA safety, nominal continuation; missing evidence is ``unresolved``, never a pass or a
violation, and events after the first failing one are not audited.
"""

import asyncio
import json
import logging
import re
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from cais_spade_llm.recovery_framework.evaluation_capture import write_record

logger = logging.getLogger(__name__)

AUDIT_NAME = "audit.json"
PASSED = "passed"
INFEASIBLE = "infeasible"
PHYSICALLY_UNSUPPORTED = "physically_unsupported"
UNSAFE = "unsafe"
NO_NOMINAL_CONTINUATION = "no_nominal_continuation"
UNRESOLVED = "unresolved"
DID_NOT_FINISH = "did_not_finish"
OUTCOME_CLASSES = (PASSED, INFEASIBLE, PHYSICALLY_UNSUPPORTED, UNSAFE, NO_NOMINAL_CONTINUATION, UNRESOLVED,
                   DID_NOT_FINISH)
# Capability checks that test a pose, a footprint or a route in the recorded cell. The others (``capacity``,
# the preconditions a resource's primitives share, ``handling``) are discrete.
_PHYSICAL_CHECKS = {"navigate", "move", "pick", "place", "transport"}
_SEVERITY = {"FEASIBLE": 0, "NEEDS_CONTEXT": 1, "INFEASIBLE": 2}


def ra_stages(result: dict) -> dict[str, str]:
    """Split one RA result into its symbolic and physical parts.

    A condition fails physically when some primitive makes the stated value with its preconditions holding,
    so only the continuous pose constraint refused it (Eq. 12); otherwise it fails symbolically (Alg. 1). A
    capability row the RA marks with ``refused_by`` follows that mark; otherwise pick, place, move, navigate
    and transport test the recorded cell and are physical, and every other capability check is symbolic.

    Args:
        result: One ``ra_feasibility.check_event`` result.

    Returns:
        ``{"symbolic": status, "physical": status}``, each the worst status of its rows.
    """
    symbolic, physical = ["FEASIBLE"], ["FEASIBLE"]
    for row in result.get("conditions") or []:
        if row.get("status", "FEASIBLE") == "FEASIBLE":
            continue
        continuous = [producer for producer in row.get("producers") or []
                      if producer.get("makes_value") and not _failed_preconditions(producer)]
        if continuous:
            # The condition holds if any of these primitives works, so its least severe refusal counts.
            physical.append(min((producer.get("status", "INFEASIBLE") for producer in continuous), key=_severity))
        else:
            symbolic.append(row["status"])
    for row in result.get("capability") or []:
        continuous_row = (row["refused_by"] == "pose_constraint" if "refused_by" in row
                          else row.get("check") in _PHYSICAL_CHECKS)
        (physical if continuous_row else symbolic).append(row.get("status", "FEASIBLE"))
    stages = {"symbolic": _worst(symbolic), "physical": _worst(physical)}
    if result.get("feasibility_status") not in (None, "FEASIBLE") and set(stages.values()) == {"FEASIBLE"}:
        # A refusal from a row the split does not know stays symbolic rather than being lost.
        stages["symbolic"] = result["feasibility_status"]
    return stages


def _failed_preconditions(producer: dict) -> bool:
    """Return whether a producer that makes the value was refused by its preconditions, not its pose constraint.

    The RA records which part refused it in ``refused_by``. Drafts recorded before that field read only the
    reason, "<primitive> needs …" (or "<primitive> (<declared values>) needs …" for one of several
    alternatives) for preconditions and "<primitive>: …" for a pose constraint.
    """
    if "refused_by" in producer:
        return producer["refused_by"] != "pose_constraint"
    name = re.escape(str(producer.get("primitive")))
    return re.match(rf"{name}(?: \(.*?\))? needs ", str(producer.get("reason", ""))) is not None


def _severity(status: str) -> int:
    return _SEVERITY.get(status, 2)


def _worst(statuses: list[str]) -> str:
    return max(statuses, key=_severity)


def classify_event(ra: dict, cca: dict | None) -> tuple[str, str] | None:
    """Return ``(outcome, stage)`` for an event a check refused, or None when it passed.

    Stages run in the fixed order; the first that does not pass decides, and inside a stage a violation
    beats missing evidence.
    """
    stages = ra_stages(ra)
    for stage, violation in (("ra_feasibility", INFEASIBLE), ("physical_evidence", PHYSICALLY_UNSUPPORTED)):
        status = stages["symbolic" if stage == "ra_feasibility" else "physical"]
        if status != "FEASIBLE":
            return (violation if status == "INFEASIBLE" else UNRESOLVED), stage
    if cca is not None and cca.get("safety_status") != "SAFE":
        return (UNSAFE if cca.get("safety_status") == "UNSAFE" else UNRESOLVED), "cca_safety"
    return None


def audit_draft(directory: Path, observations: dict, safety_files: list[str],
                cca_assumptions: list[dict] | None = None) -> dict:
    """Audit one finished draft from its capture's failure state and write ``audit.json`` beside it.

    Runs the checks synchronously; call it from a thread without a running event loop.

    Args:
        directory: The draft's directory, holding ``draft.json``.
        observations: The draft's saved failure input, already read.
        safety_files: The selected specification paths the CCA checks.
        cca_assumptions: Declared facts about the failure state, as for a CCA-checked draft.

    Returns:
        The audit record.
    """
    record = json.loads((Path(directory) / "draft.json").read_text(encoding="utf-8"))
    started = time.monotonic()
    audit = {"version": 1, "kind": "outline_audit", "draft": Path(directory).name,
             "draft_status": record.get("status"), "event_count": len(record.get("transition_trace") or []),
             "selected_safety_files": list(safety_files), "cca_assumptions": deepcopy(cca_assumptions or [])}
    if record.get("status") != "draft":
        audit.update(status="audited", outcome=DID_NOT_FINISH, stop_cause=deepcopy(record.get("stop_cause")),
                     message=record.get("message"))
    else:
        try:
            audit.update(status="audited", **_replay(Path(directory), record, observations, safety_files,
                                                     cca_assumptions))
        except (ValueError, LookupError, TypeError, AttributeError, RuntimeError, OSError) as exc:
            logger.warning("Outline audit of %s failed: %s", directory, exc)
            audit.update(status="audit_failed", error=f"{type(exc).__name__}: {exc}")
    audit["elapsed_seconds"] = time.monotonic() - started
    write_record(Path(directory) / AUDIT_NAME, audit)
    return audit


def _replay(directory: Path, record: dict, observations: dict, safety_files: list[str],
            cca_assumptions: list[dict] | None) -> dict:
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_validated_event import (
        check_outline,
    )
    from cais_spade_llm.recovery_framework import cca_nonblocking

    from .outline_draft import OutlineDraft

    events = [deepcopy(row) for row in record.get("transition_trace") or []]
    # The checker writes nothing; it only rebuilds the failure state and the checks' inputs.
    checker = OutlineDraft(observations, safety_files, root=directory, check_ra_feasibility=True,
                           check_nonblocking=True, check_cca_safety=True, cca_assumptions=cca_assumptions)
    session = multi_turn.build_multi_turn_session_seed(checker.prepared)
    # The nonblocking check runs below, so a search that runs out reads as unresolved, not as a refusal.
    session["check_nonblocking"] = False
    result = asyncio.run(check_outline(events, session, checker.prepared,
                                       SimpleNamespace(product_agent=checker.provider)))
    replayed = result["session"].get("accepted_outline_prefix") or []
    outcome = {"checks": result["checks"], "audited_events": len(result["checks"]),
               "replay_mismatches": [index for index, (saved, again) in enumerate(zip(events, replayed, strict=False))
                                     if any(saved.get(key) != again.get(key)
                                            for key in ("expected_start_state", "condition_changes"))]}
    index = result["refused_event_index"]
    if index is not None:
        step = result["checks"][index]
        cca = (step.get("cca_safety") or [None])[0]
        verdict, stage = classify_event(step["ra_feasibility"][0], cca) or (UNRESOLVED, "ra_feasibility")
        return {**outcome, "outcome": verdict, "failing_stage": stage, "first_failing_event": index + 1,
                "findings": deepcopy(result["audit"].get("validation_findings") or [])}
    started = time.monotonic()
    try:
        check = cca_nonblocking.check_completion(result["session"], checker.prepared, checker.provider.cca_capture)
    except RuntimeError as exc:
        # The search budget ran out or a model is incomplete: never a verdict.
        return {**outcome, "outcome": UNRESOLVED, "failing_stage": "nonblocking", "first_failing_event": None,
                "nonblocking": {"error": str(exc), "seconds": time.monotonic() - started}}
    nonblocking = {**deepcopy(check), "seconds": time.monotonic() - started}
    if check["nonblocking_status"] == cca_nonblocking.BLOCKING:
        return {**outcome, "outcome": NO_NOMINAL_CONTINUATION, "failing_stage": "nonblocking",
                "first_failing_event": None, "nonblocking": nonblocking,
                "findings": cca_nonblocking.refusal_findings(check)}
    return {**outcome, "outcome": PASSED, "failing_stage": None, "first_failing_event": None,
            "nonblocking": nonblocking}

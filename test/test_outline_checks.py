from __future__ import annotations

"""Independent check selections in the existing multi_turn outline workflow."""

import asyncio
from copy import deepcopy
from itertools import product
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from cais_spade_llm.recovery_framework import outline_checks, outline_validation
from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint


@pytest.mark.parametrize("ra_enabled,physical_enabled,cca_enabled", list(product((False, True), repeat=3)))
def test_each_selected_check_runs_independently(monkeypatch, ra_enabled, physical_enabled, cca_enabled):
    task = {"outline_id": "event", "resource_jid": "owner", "part_name": None,
            "expected_start_state": {"resource_state": "failed"}, "expected_end_state": {"resource_state": "idle"}}
    model = {"state_variables": {}, "events": []}
    support = {"semantics": "successor_primitive_union", "matching_transitions": [], "terminal_state": {"unknown": {}}}
    projection = {"status": "projected", "interval_complete": True}
    owner = SimpleNamespace(jid="owner", get_recovery_snapshot=lambda: {},
        recovery_des_model=lambda **_: deepcopy(model),
        validate_recovery_primitive_support=Mock(return_value={"allowed": True, "feasibility_status": "FEASIBLE", "primitive_support": support}),
        project_recovery_outline_event=Mock(return_value=projection))
    monkeypatch.setattr("cais_spade_llm.resources.recovery_feasibility.derive_primitive_union", Mock(return_value=support))
    monkeypatch.setattr("cais_spade_llm.recovery_framework.recovery_observations.projection_capture_compatibility",
                        lambda **_: {"status": "supported", "checks": []})
    state_projector = Mock(return_value={**projection, "state_only": True})
    monkeypatch.setattr(outline_checks, "state_only_projection", state_projector)
    safety = Mock(return_value={"is_safe": True, "validation_status": "FEASIBLE", "monitor_context_after": {}})
    monkeypatch.setattr("cais_spade_llm.agents.central_controller.outline_evidence_safety.validate_outline_evidence_safety", safety)

    async def resource_validation(*, payload, **_):
        return outline_validation.validate_resource_outline_payload(owner, payload, offline=True)

    async def safety_validation(*, payload):
        return outline_validation.validate_safety_outline_payload(payload, offline=True)

    agent = SimpleNamespace(request_recovery_outline_physical_validation=AsyncMock(side_effect=resource_validation),
                            request_recovery_outline_safety_validation=AsyncMock(side_effect=safety_validation))
    choices = dict(zip(outline_checks.CHECK_FIELDS, (ra_enabled, physical_enabled, cca_enabled)))
    session = {**choices, "outline_diagnostic_bypass": not all(choices.values()),
               "primitive_support_semantics": "successor_primitive_union", "symbolic_parts": {}}
    document = {"document": {"catalog": {"version": 2, "specifications": []}, "requirement_scopes": []},
                "source_sha256": "specification", "source_path": "selected"}
    prepared = {"recovery_resources": {"owner": {"recovery_snapshot": {}, "recovery_des_model": model}},
                "outline_safety_specifications": [document]}
    before = deepcopy(session)
    turn = {}
    checks, result, verdict = asyncio.run(outline_checks.validate_partial_event(task=task, session=session,
        prepared=prepared, planner=SimpleNamespace(product_agent=agent), turn=turn))
    assert session == before
    assert owner.validate_recovery_primitive_support.call_count == int(ra_enabled)
    assert owner.project_recovery_outline_event.call_count == int(physical_enabled)
    assert safety.call_count == int(cca_enabled)
    assert agent.request_recovery_outline_safety_validation.await_count == int(cca_enabled)
    assert checks == {"ra": "passed" if ra_enabled else "not_checked",
                      "physical_evidence": "passed" if physical_enabled else "not_checked",
                      "cca": "passed" if cca_enabled else "not_checked"}
    if not ra_enabled and (physical_enabled or cca_enabled) and result:
        assert result["allowed"] is None and result["feasibility_status"] == "not_checked"
    if not cca_enabled:
        assert verdict == {}


def test_disabled_physical_check_never_calls_projection_after_feasible_ra(monkeypatch):
    task = {"outline_id": "event", "resource_jid": "owner", "expected_end_state": {}}
    owner = SimpleNamespace(jid="owner", get_recovery_snapshot=lambda: {}, recovery_des_model=lambda **_: {},
        validate_recovery_primitive_support=Mock(return_value={"allowed": True, "feasibility_status": "FEASIBLE"}),
        project_recovery_outline_event=Mock(side_effect=AssertionError("Physical evidence was bypassed")))
    payload = {"primitive_support_semantics": "successor_primitive_union", "check_physical_evidence": False,
               "candidates": [{"task": task}]}
    result = outline_validation.validate_resource_outline_payload(owner, payload, offline=True)["results"][0]
    assert result["feasibility_status"] == "FEASIBLE"
    assert result["geometry_evidence"]["status"] == "not_checked"
    owner.project_recovery_outline_event.assert_not_called()
    with pytest.raises(ValueError, match="saved observations"):
        outline_validation.validate_resource_outline_payload(owner, payload)


def test_cca_without_physical_evidence_pauses_for_mutex(monkeypatch):
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "cais_spade_llm/specification/safety/safety_M1_mutex.txt"
    documents = [{"document": json.loads(path.read_text()), "source_sha256": "mutex", "source_path": str(path)}]
    projection = outline_checks.state_only_projection(task={}, snapshot={}, model={}, physical_input={},
        specifications=documents, monitor_context={})
    assert projection["status"] == "needs_context"
    assert projection["context_requests"] == ["/physical_evidence"]
    task = {"outline_id": "event"}
    source = {"task": task, "snapshot": {}, "model": {}, "physical_input": {}}
    response = outline_validation.validate_safety_outline_payload({"check_physical_evidence": False,
        "candidates": [{"safety_input": {"task": task, "candidate_fingerprint": recovery_validation_fingerprint(task),
            "specifications": documents, "projection": projection, "evidence_input": source}}]}, offline=True)
    assert response["results"][0]["is_safe"] is False
    assert response["results"][0]["validation_status"] == "unresolved"
    assert response["results"][0]["monitor_context_after"] == {}


def test_state_only_process_order_uses_owner_ledgers_without_spatial_data(monkeypatch):
    import json
    from pathlib import Path

    parts = {"KET4_Square_4mm": {"processCompleted": [{"process": "trim", "result": "square"}],
            "processCompleted_complete": True, "processCompleted_evidence": {"complete": True, "source_kind": "recorded", "checkpoint": "capture"}},
        "RGOCG4-50_Round_4mm": {"processCompleted": [], "processCompleted_complete": True,
            "processCompleted_evidence": {"complete": True, "source_kind": "recorded", "checkpoint": "capture"}}}
    after = deepcopy(parts)
    after["RGOCG4-50_Round_4mm"]["processCompleted"] = [{"process": "trim", "result": "circle"}]
    monkeypatch.setattr("cais_spade_llm.resources.environment_recovery.declared_environment_successors",
        lambda **_: ([{"environment_products": after, "environment_valuation": {"M2": {}},
                      "source_fingerprint": "owner", "source_model_fingerprint": "models"}], []))
    path = Path(__file__).resolve().parents[1] / "cais_spade_llm/specification/safety/safety_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing.txt"
    documents = [{"document": json.loads(path.read_text()), "source_sha256": "order", "source_path": str(path)}]
    physical = {"resource_evidence": {"environment_models": {"M2": {}}, "environment_valuation": {"M2": {}}},
                "physical_evidence": {"snapshot": {"parts": parts}, "geometry": {"parts": {name: {} for name in parts}}}}
    task = {"outline_id": "event"}
    source = {"task": task, "snapshot": {}, "model": {}, "physical_input": physical}
    projection = outline_checks.state_only_projection(**source, specifications=documents, monitor_context={})
    assert projection["status"] == "projected"
    assert projection["geometry"]["regions"] == projection["geometry"]["resources"] == {}
    response = outline_validation.validate_safety_outline_payload({"check_physical_evidence": False,
        "candidates": [{"safety_input": {"task": task, "candidate_fingerprint": recovery_validation_fingerprint(task),
            "specifications": documents, "projection": projection, "evidence_input": source}}]}, offline=True)
    assert response["results"][0]["is_safe"] is True
    assert response["results"][0]["validation_status"] == "passed"
    assert response["execution_authorized"] is False

    async def safety_validation(*, payload):
        return outline_validation.validate_safety_outline_payload(payload, offline=True)

    agent = SimpleNamespace(request_recovery_outline_safety_validation=AsyncMock(side_effect=safety_validation),
        request_recovery_outline_physical_validation=Mock(side_effect=AssertionError("Physical evidence is bypassed")))
    session = {"check_ra_feasibility": False, "check_physical_evidence": False, "check_cca_safety": True,
        "projected_state_evidence": projection,
        "outline_monitor_context": response["results"][0]["monitor_context_after"]}
    prepared = {"physical_evidence": deepcopy(physical["physical_evidence"]),
        "outline_safety_specifications": documents,
        "recovery_resources": {"owner": {"recovery_snapshot": {}, "recovery_des_model": {},
                                            "resource_evidence": physical["resource_evidence"]}}}
    original = deepcopy(prepared)
    second = {"outline_id": "second_event", "resource_jid": "owner"}
    checks, _, cca = asyncio.run(outline_checks.validate_partial_event(task=second, session=session, prepared=prepared,
        planner=SimpleNamespace(product_agent=agent), turn={}))
    assert checks == {"ra": "not_checked", "physical_evidence": "not_checked", "cca": "passed"}
    assert cca["state_only_projection"]["observations"][0]["parts"] == projection["observations"][-1]["parts"]
    assert prepared == original
    agent.request_recovery_outline_physical_validation.assert_not_called()


@pytest.mark.parametrize("mismatched", ("task", "candidate_fingerprint"))
def test_state_only_evidence_must_identify_the_proposed_event(mismatched):
    task = {"outline_id": "event"}
    source = {"task": deepcopy(task), "snapshot": {}, "model": {}, "physical_input": {}}
    projection = outline_checks.state_only_projection(**source, specifications=[], monitor_context={})
    safety = {"task": task, "candidate_fingerprint": recovery_validation_fingerprint(task),
              "projection": projection, "specifications": [], "evidence_input": source}
    if mismatched == "task":
        source["task"]["outline_id"] = "another_event"
    else:
        safety["candidate_fingerprint"] = "another_event"
    with pytest.raises(ValueError, match="different proposed candidate"):
        outline_validation.validate_safety_outline_payload({"check_physical_evidence": False,
            "candidates": [{"safety_input": safety}]}, offline=True)

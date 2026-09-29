"""Task-level baseline audits distinguish exhaustive absence from missing models."""

from __future__ import annotations

from copy import deepcopy

from cais_spade_llm.recovery_framework.task_des_audit import audit_task_des


def _model():
    return {
        "states": {
            "fault": {"part_state": "slipped", "obligations": ["placement", "resume"]},
            "done": {"part_state": "placed", "obligations": []},
        },
        "transitions": {"fault": {"supported_recovery": "done"}},
        "events": {"supported_recovery": {"event_name": "supported_recovery"}},
    }


def _audit(model, **kwargs):
    return audit_task_des(
        model=model,
        failure_valuation=_model()["states"]["fault"],
        goal_conditions={"part_state": "placed", "obligations": []},
        **kwargs,
    )


def test_legitimate_modeled_recovery_is_reported_without_suppressing_it():
    model = _model()
    original = deepcopy(model)
    result = _audit(model, complete=True)
    assert result["status"] == "path_found"
    assert result["path"][0]["event_id"] == "supported_recovery"
    assert model == original


def test_no_path_requires_complete_exploration_of_a_complete_model():
    model = _model()
    model["transitions"] = {}
    assert _audit(model)["status"] == "inconclusive"
    result = _audit(model, complete=True)
    assert result["status"] == "no_path"
    assert result["explored_states"] == 1


def test_unknown_start_missing_models_and_budgets_are_inconclusive():
    assert _audit(None)["status"] == "inconclusive"
    assert _audit(_model(), complete=True, max_states=1)["status"] == "inconclusive"
    assert _audit(_model(), complete=True, deadline_s=0)["status"] == "inconclusive"
    model = _model()
    model["states"]["fault"]["part_state"] = "ready"
    assert _audit(model, complete=True)["status"] == "inconclusive"


def test_saved_failure_audit_rejects_stale_snapshot_binding():
    from cais_spade_llm.recovery_framework.task_des_audit import (
        audit_saved_failure,
        failure_snapshot_fingerprint,
    )

    context = {"failure_scenario_id": "slippage", "part_tracker": {"part": {"location": "floor"}}}
    context["predefined_task_des_audit"] = {
        "model": {
            "states": {"fault": {"recovered": False}, "goal": {"recovered": True}},
            "transitions": {"fault": {"recover": "goal"}},
            "events": {"recover": {}},
        },
        "failure_valuation": {"recovered": False},
        "goal_conditions": {"recovered": True},
        "complete": True,
        "snapshot_fingerprint": failure_snapshot_fingerprint(context),
    }
    assert audit_saved_failure(context)["status"] == "path_found"
    context["part_tracker"]["part"]["location"] = "unknown"
    result = audit_saved_failure(context)
    assert result["status"] == "inconclusive"
    assert "not bound" in result["reason"]

from __future__ import annotations

"""Schema-2 captures keep raw facts apart from the views derived from them."""

import hashlib
import json

import pytest

from cais_spade_llm.recovery_framework import capture_files as files
from cais_spade_llm.recovery_framework.recovery_observations import (
    RECORDS,
    observation_fingerprint,
    read_observations,
)

FAULT_VALUES = ["failed"]


def task(task_id: str | None, resource_id: str, event_name: str, part_name: str) -> dict:
    row = {"resource_id": resource_id, "event_id": event_name, "event_name": event_name,
           "parameters": {"part_name": part_name}}
    return row if task_id is None else {"task_id": task_id, **row}


def capture() -> dict:
    """A schema-2 capture: a robot reported its own fault after losing a peg."""
    return {
        "schema_version": 2, "run_id": "run", "scenario": "Supplied scenario name",
        "jids": {"arm_a": "jid-a", "cell": "jid-cell"},
        "inputs": {"scene": {}}, "config_references": [{"path": "scene.json", "sha256": "0" * 64}],
        "failure": {"pending_tasks": [task("t1", "arm_a", "place_approach", "peg")], "continuations": {},
                    "resource_values_before": {"arm_a": {"resource_state": "picked", "held_part": "peg"}},
                    "resource_values_after": {"arm_a": {"resource_state": "failed", "held_part": None}},
                    "part_tracker_after": {"peg": {"state": "misplaced", "location": None}}},
        "acknowledgements": {}, "transitions": [],
        "fault_reports": {"arm_a": {"reported_state": "failed", "source": "robot controller"}},
        "physical_evidence": {
            "resources": {"arm_a": {"observation_state": {"resource_state": "failed"}}},
            "parts": {"peg": {"name": "peg", "pose": [1.0, 0.0, 1.0, 0, 0, 0, 1]}},
            "snapshot": {"parts": {"peg": {"state": "misplaced"}}},
            "environment_valuation": {"arm_a": {"resource_state": "failed"}},
            "products": {"peg": {"state": "misplaced"}}, "environment_models": {"cell": {}}},
        "part_tracker": {"peg": {"state": "misplaced", "location": None}},
        "resource_states": {"arm_a": {"resource_state": "failed", "held_part": None}},
        "resource_evidence": {"jid-a": {"state": {"_current_state": "failed"}, "inputs": {"scene": {}}}},
        "prepared_recovery_request": {"recovery_resources": {
            "jid-a": {"resource_type": "robot", "static_capabilities": {"reach": 1.0},
                      "execution_primitive_catalog": [{"name": "move"}], "recovery_des_model": {"events": []},
                      "recovery_snapshot": {"current_state": "failed"}},
            "jid-cell": {"resource_type": "machine", "static_capabilities": {},
                         "recovery_des_model": {"events": ["from the plant model"]}}}},
        "pending_nodes": [{"id": "t1"}], "offline_ready": True, "missing": [],
    }


def test_raw_file_keeps_facts_and_preparation_keeps_derived_views(tmp_path):
    path = tmp_path / "observations.json"
    record = capture()
    files.write_capture(path, record)
    raw = json.loads(path.read_bytes())
    preparation = json.loads((tmp_path / files.PREPARATION_NAME).read_bytes())
    assert not set(raw) & files.DERIVED_KEYS
    assert {"snapshot", "environment_valuation", "products", "environment_models"}.isdisjoint(raw["physical_evidence"])
    assert {"resource_values_after", "part_tracker_after"}.isdisjoint(raw["failure"])
    assert files.label_audit(raw, FAULT_VALUES) == []
    assert raw["agent_reports"] == {"jid-a": {"_current_state": "failed"}}
    assert set(raw["agent_declarations"]["jid-a"]) == {"resource_type", "static_capabilities",
                                                       "execution_primitive_catalog", "recovery_des_model"}
    assert "recovery_des_model" not in raw["agent_declarations"]["jid-cell"]
    assert preparation["observations_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert files.read_capture(path) == record


def test_views_that_do_not_match_the_raw_file_are_not_merged(tmp_path):
    path = tmp_path / "observations.json"
    files.write_capture(path, capture())
    raw = json.loads(path.read_bytes())
    path.write_text(json.dumps({**raw, "run_id": "edited"}, indent=2))
    assert "prepared_recovery_request" not in files.read_capture(path)
    (tmp_path / files.PREPARATION_NAME).unlink()
    assert "prepared_recovery_request" not in files.read_capture(path)
    assert "agent_reports" not in files.read_capture(path)


def test_older_schemas_are_written_unchanged(tmp_path):
    path = tmp_path / "observations.json"
    record = {**capture(), "schema_version": 1}
    files.write_capture(path, record)
    assert json.loads(path.read_bytes()) == record
    assert not (tmp_path / files.PREPARATION_NAME).exists()
    assert files.read_capture(path) == record


def test_label_audit_reports_labels_no_agent_reported_about_itself():
    raw = {"fault_reports": {"arm_a": {"reported_state": "failed"}},
           "physical_evidence": {"resources": {"arm_a": {"observation_state": {"resource_state": "failed"}}}},
           "transitions": [{"after": {"arm_a": {"resource_state": "failed"}}}],
           "checkpoint_motion_cancellation": {"worker_result": {"status": "failed"}},
           "failure": {"part_tracker_before": {"peg": {"state": "misplaced"}}}}
    assert files.label_audit(raw, FAULT_VALUES) == [
        {"path": "/transitions/0/after/arm_a/resource_state", "value": "failed"},
        {"path": "/failure/part_tracker_before/peg/state", "value": "misplaced"}]


def test_pending_nodes_skip_completed_and_repeated_work_and_put_resource_cleanup_last():
    failure = {
        "pending_tasks": [task("done", "arm_a", "pick", "peg"), task("t2", "arm_a", "place", "peg")],
        "continuations": {"resource:arm_a": {"path": [task(None, "arm_a", "move_home", "peg")]},
                          "part:peg": {"path": [task("t2b", "arm_a", "place", "peg"), task(None, "cell", "process", "peg")]}},
    }
    observations = {"run_id": "run", "jids": {"arm_a": "jid-a", "cell": "jid-cell"},
                    "acknowledgements": {"done": {"task_id": "done", "run_id": "run", "status": "completed",
                                                  "evidence": "resource"}}}
    nodes = files.pending_nodes(failure, observations)
    assert [(node["id"], node["function_name"], node["resource_jid"]) for node in nodes] == [
        ("t2", "place", "jid-a"), ("run_pending_3", "process", "jid-cell"), ("run_pending_4", "move_home", "jid-a")]
    assert files.pending_nodes(failure, observations, run_id="live")[1]["id"] == "live_pending_3"


def test_configuration_files_are_referenced_by_hash(tmp_path):
    scene = tmp_path / "scene.json"
    scene.write_text("{}")
    references = files.config_references({"scene_file": "scene.json", "selected_safety_files": ["missing.txt"],
                                          "selected_safety_file": "__NONE__"}, root=tmp_path)
    assert references == [{"path": "scene.json", "sha256": hashlib.sha256(b"{}").hexdigest()}]


def test_official_part_slippage_capture_converts_without_loss(tmp_path):
    from cais_spade_llm.recovery_framework.failure_grounding import ground_failure

    path = RECORDS / "capture_f923020bf9ce496eb443826291f8f2c2" / "observations.json"
    if not path.is_file():
        pytest.skip("retained capture is not present")
    original = read_observations(path)
    record = files.convert_legacy_capture(original, FAULT_VALUES)
    destination = tmp_path / "observations.json"
    files.write_capture(destination, record)
    raw = json.loads(destination.read_bytes())
    assert observation_fingerprint(files.read_capture(destination)) == observation_fingerprint(record)
    raw_only = {key: value for key, value in raw.items() if key not in files.RAW_EXTRACTS}
    assert files._without_sources(ground_failure(raw_only)) == files._without_sources(ground_failure(original))
    # Only the schema-1 transition-log alias of the injected state remains; re-capture removes it.
    assert [row["path"].split("/")[1] for row in files.label_audit(raw, FAULT_VALUES)] == ["transitions"]

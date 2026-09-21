"""Saved experiment evidence remains separate from runtime authority."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from cais_spade_llm.spec2primitives import project_results


def _write(root: Path, name: str, payload: object) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_saved_versions_and_execution_do_not_manufacture_assembly_success(tmp_path):
    root = tmp_path / "interaction_Exact-A"
    _write(
        root,
        "products/user_requirement/product_requirement.json",
        {"product_requirement": "assemble medium gear"},
    )
    _write(
        root, "interaction_record/context_completion_0001.json", {"status": "grounding complete"}
    )
    _write(
        root,
        "composition/selected_ra_assignments/assignment_0001.json",
        {"selected_resource_jid": "xarm6@localhost"},
    )
    for index in (1, 2):
        _write(
            root,
            f"composition/primitive_program_candidates/attempt_{index:04d}/candidate.json",
            {"status": "proposed", "primitive_steps": [{"primitive": f"exact_{index}"}]},
        )
    run = "composition/refinement_runs/run_0001"
    first = {"ref": "composition/primitive_program_candidates/attempt_0001/candidate.json"}
    second = {"ref": "composition/primitive_program_candidates/attempt_0002/candidate.json"}
    _write(root, f"{run}/validation_0001.json", {"status": "failed", "candidate_ref": first})
    _write(root, f"{run}/validation_0002.json", {"status": "passed", "candidate_ref": second})
    _write(
        root,
        f"{run}/result.json",
        {"status": "validated_for_declared_scope", "candidate_refs": [first, second]},
    )
    _write(root, f"{run}/event_0001.json", {"elapsed_sec": 0})
    _write(
        root,
        "execution/run_0001/result.json",
        {
            "record_type": "PrimitiveExecutionResult",
            "status": "completed",
            "assembly_success": None,
        },
    )
    before = {path: path.read_bytes() for path in root.rglob("*.json")}

    rows = project_results.read_interactions(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["interaction_identifier"] == "interaction_Exact-A"
    assert row["first_pass_validation"] == "failed"
    assert row["composition_status"] == "validated_for_declared_scope"
    assert row["candidate_count"] == 2
    assert row["simulation_status"] == "completed"
    assert row["assembly_success"] == "not recorded"
    assert row["physical_outcome"] == "not recorded"
    exported = list(csv.DictReader(io.StringIO(project_results.results_csv(rows))))[0]
    assert exported["elapsed_sec"] == "0"
    assert exported["candidate_count"] == "2"
    assert before == {path: path.read_bytes() for path in root.rglob("*.json")}
    assert len(project_results.artifact_paths(root)) == len(before)


def test_incomplete_latest_runs_do_not_reuse_earlier_success(tmp_path):
    root = tmp_path / "interaction_incomplete"
    _write(
        root,
        "products/user_requirement/product_requirement.json",
        {"product_requirement": "exact requirement"},
    )
    _write(
        root,
        "composition/refinement_runs/run_0001/result.json",
        {"status": "validated_for_declared_scope"},
    )
    _write(root, "composition/refinement_runs/run_0002/request.json", {"profile": {}})
    _write(
        root,
        "execution/run_0001/result.json",
        {
            "record_type": "PrimitiveExecutionResult",
            "status": "completed",
            "assembly_success": True,
        },
    )
    _write(root, "execution/run_0002/request.json", {})
    row = project_results.read_interactions(tmp_path)[0]
    assert row["composition_status"] == "not recorded"
    assert row["simulation_status"] == "not recorded"
    assert row["assembly_success"] == "not recorded"


def test_later_candidate_validation_is_not_a_first_pass_result(tmp_path):
    root = tmp_path / "interaction_revision"
    _write(
        root, "products/user_requirement/product_requirement.json", {"product_requirement": "exact"}
    )
    run = "composition/refinement_runs/run_0001"
    first = {"ref": "composition/primitive_program_candidates/attempt_0001/candidate.json"}
    second = {"ref": "composition/primitive_program_candidates/attempt_0002/candidate.json"}
    _write(
        root,
        f"{run}/result.json",
        {"status": "validated_for_declared_scope", "candidate_refs": [first, second]},
    )
    _write(root, f"{run}/validation_0001.json", {"status": "passed", "candidate_ref": second})
    assert project_results.read_interactions(tmp_path)[0]["first_pass_validation"] == "not recorded"


def test_missing_and_corrupt_interactions_remain_inspectable(tmp_path):
    assert project_results.read_interactions(tmp_path / "missing") == []
    root = tmp_path / "interaction_failed"
    _write(root, "products/user_requirement/product_requirement.json", [])
    _write(
        root, "interaction_record/turn_0001.json", {"PA_output": {"grounding_status": "incomplete"}}
    )
    row = project_results.read_interactions(tmp_path)[0]
    assert row["grounding_status"] == "incomplete"
    assert "Expected a JSON object" in row["record_error"]
    assert project_results.read_artifact(root, "products/user_requirement/product_requirement.json")["raw_text"] == "[]"
    assert "record_error" in project_results.read_artifact(root, "../../outside.json")


def test_false_outcome_is_preserved_and_external_symlinks_are_rejected(tmp_path):
    root = tmp_path / "interaction_false"
    _write(
        root, "products/user_requirement/product_requirement.json", {"product_requirement": "exact"}
    )
    _write(
        root,
        "execution/run_0001/result.json",
        {"record_type": "PrimitiveExecutionResult", "status": "failed", "assembly_success": False},
    )
    assert project_results.read_interactions(tmp_path)[0]["assembly_success"] is False
    (tmp_path / "interaction_link").symlink_to(tmp_path.parent, target_is_directory=True)
    assert len(project_results.read_interactions(tmp_path)) == 1


def test_results_ui_exports_filtered_rows_and_inspects_saved_programs(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from nicegui import context, ui
    from nicegui.client import Client

    for identifier in ("interaction_A", "interaction_B"):
        _write(
            tmp_path / identifier,
            "products/user_requirement/product_requirement.json",
            {"product_requirement": identifier},
        )
    candidate = {
        "status": "proposed",
        "primitive_steps": [{"primitive": "move_cartesian", "parameters": {"x": 1}}],
    }
    path = _write(
        tmp_path / "interaction_B",
        "composition/primitive_program_candidates/attempt_0001/candidate.json",
        candidate,
    )
    before = path.read_bytes()
    callbacks = {}
    original_click = ui.button.on_click
    original_select = ui.table.on_select

    def capture_click(button, callback):
        callbacks[button.text] = callback
        return original_click(button, callback)

    def capture_select(table, callback):
        callbacks["select"] = callback
        return original_select(table, callback)

    download = Mock()
    monkeypatch.setattr(ui.button, "on_click", capture_click)
    monkeypatch.setattr(ui.table, "on_select", capture_select)
    monkeypatch.setattr(ui.download, "content", download)
    client = Client(context.client.page)
    with client:
        project_results.render_results(tmp_path)
        search = next(
            element for element in client.elements.values() if isinstance(element, ui.input)
        )
        table = next(
            element for element in client.elements.values() if isinstance(element, ui.table)
        )
        search.value = "interaction_B"
        assert len(table.rows) == 1
        callbacks["Export CSV"]()
        exported = list(csv.DictReader(io.StringIO(download.call_args.args[0])))
        assert exported[0]["interaction_identifier"] == "interaction_B"
        table.selected = [table.rows[0]]
        callbacks["select"]()
        code = next(element for element in client.elements.values() if isinstance(element, ui.code))
        assert json.loads(code.content) == candidate
        assert path.read_bytes() == before
    client.delete()

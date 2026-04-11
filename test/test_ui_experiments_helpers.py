from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "nicegui" not in sys.modules:
    sys.modules["nicegui"] = types.SimpleNamespace(ui=types.SimpleNamespace())

experiments = importlib.import_module("cais_spade_llm.ui.pages.experiments")


def test_asset_catalog_signature_tracks_only_experiment_assets() -> None:
    context_a = {
        "manifest": {"scenarios": [{"id": "S1"}]},
        "product_catalog": [{"path": "product-a"}],
        "requirement_files_by_product": {"product-a": ["req-a"]},
        "verified_safety_catalog": [{"path": "safety-a"}],
        "verified_rules_by_safety_file": {"safety-a": [{"id": "SAFE_1"}]},
    }
    context_b = {
        "manifest": {"scenarios": [{"id": "S9"}], "defaults": {"trials_per_method": 99}},
        "product_catalog": [{"path": "product-a"}],
        "requirement_files_by_product": {"product-a": ["req-a"]},
        "verified_safety_catalog": [{"path": "safety-a"}],
        "verified_rules_by_safety_file": {"safety-a": [{"id": "SAFE_1"}]},
    }
    context_c = {
        **context_b,
        "requirement_files_by_product": {"product-a": ["req-a", "req-b"]},
    }

    assert experiments._asset_catalog_signature(context_a) == experiments._asset_catalog_signature(context_b)
    assert experiments._asset_catalog_signature(context_a) != experiments._asset_catalog_signature(context_c)


def test_selected_file_name_and_preview_text_helpers() -> None:
    assert experiments._selected_file_name("") == "(none selected)"
    assert experiments._selected_file_name("cais_spade_llm/specification/products/requirements/s1_rpc_232.txt") == (
        "s1_rpc_232.txt"
    )

    layout = experiments._requirement_layout(
        "cais_spade_llm/specification/products/requirements/s1_rpc_232.txt",
        valid_parts={"SG", "MRP", "LCP"},
    )
    assert layout["derived"] is True
    assert layout["part_order"] == ["SG", "MRP", "LCP"]

    preview = experiments._read_preview_text(
        "cais_spade_llm/specification/products/requirements/s1_rpc_232.txt"
    )
    assert preview.startswith("[Product Requirements]")
    assert "assemble SG" in preview

    fallback = experiments._read_preview_text("missing/file.txt", fallback="fallback text")
    assert fallback == "fallback text"


def test_repair_history_rows_summarize_validation_and_replans() -> None:
    rows = experiments._repair_history_rows(
        {
            "repair_history": [
                {
                    "phase": "validation",
                    "attempt_index": 0,
                    "ok": False,
                    "satisfied_rule_count": 2,
                    "safety_rule_count": 3,
                    "violated_rules": ["SAFE_3"],
                    "witness_count": 1,
                    "stop_reason": "violations_found",
                },
                {
                    "phase": "repair",
                    "attempt_index": 1,
                    "compile_ok": False,
                    "changed_task_ids": ["REQ_2_T3"],
                    "error_message": "same-resource block cycle",
                },
                {
                    "phase": "validation",
                    "attempt_index": 2,
                    "ok": True,
                    "satisfied_rule_count": 3,
                    "safety_rule_count": 3,
                    "violated_rules": [],
                    "witness_count": 0,
                    "stop_reason": "repaired_valid",
                },
            ]
        }
    )

    assert rows[0]["step"] == "Initial validation"
    assert rows[0]["result"] == "violations_found"
    assert rows[0]["safety_rules"] == "2/3"
    assert rows[0]["violated_rules"] == "SAFE_3"
    assert rows[1]["step"] == "Replan 1"
    assert rows[1]["result"] == "repair failed"
    assert rows[1]["changed_tasks"] == "REQ_2_T3"
    assert rows[1]["error"] == "same-resource block cycle"
    assert rows[2]["step"] == "Validation after replan 2"
    assert rows[2]["result"] == "valid"
    assert rows[2]["safety_rules"] == "3/3"


def test_optional_count_formatter_for_aggregate_repair_columns() -> None:
    assert experiments._format_optional_count(None) == "N/A"
    assert experiments._format_optional_count(1) == "1.00"
    assert experiments._format_optional_count(0.5) == "0.50"

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cais_spade_llm.agents.central_controller.plan_safety_validator import PlanSafetyValidator
from cais_spade_llm.bundles.bundle_compiler import BundleCompiler
from cais_spade_llm.experiments.offline_study import OfflineStudyRunner


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_secondary_robot_resources_are_available_in_runtime_and_experiments() -> None:
    default_dir = ROOT / "cais_spade_llm" / "initialization" / "resources"
    experiment_dir = ROOT / "writing" / "experiments" / "resources"

    assert (experiment_dir / "robot_xarm6_2.json").exists()
    assert (experiment_dir / "robot_ur5e_2.json").exists()
    assert (default_dir / "robot_xarm6_2.json").exists()
    assert (default_dir / "robot_ur5e_2.json").exists()


def test_experiment_manifest_load_and_rendering() -> None:
    manifest_path = ROOT / "writing" / "experiments" / "offline_planning_experiment_pack.json"
    runner = OfflineStudyRunner(manifest_path)
    study = runner.load_study_definition()
    editor_context = runner.load_editor_context()

    assert study["study_id"] == "offline_planning_experiment_pack"
    assert study["defaults"]["methods"]
    assert set(study["defaults"]["methods"]).issubset({"llm_nl_safety", "verified"})
    assert [scenario["id"] for scenario in study["scenarios"]] == ["S1", "S2", "S3"]
    assert [row["key"] for row in editor_context["robot_catalog"]] == ["xarm6", "ur5e", "xarm6-2", "ur5e-2"]
    assert editor_context["product_catalog"][0]["path"] == "cais_spade_llm/initialization/products/assembly_board-v1.json"
    assert (
        "cais_spade_llm/specification/products/requirements/case3_two_arm_llm_bridge.txt"
        in editor_context["requirement_files_by_product"]["cais_spade_llm/initialization/products/assembly_board-v1.json"]
    )
    assert any(
        row["path"] == "cais_spade_llm/specification/safety/safety_case_llm_bridge.txt"
        for row in editor_context["verified_safety_catalog"]
    )
    assert "SAFE_1" in {
        rule["id"]
        for rule in editor_context["verified_rules_by_safety_file"][
            "cais_spade_llm/specification/safety/safety_case_llm_bridge.txt"
        ]
    }

    scenario_resource_keys = {
        scenario["id"]: scenario["resource_keys"] for scenario in editor_context["manifest"]["scenarios"]
    }
    assert scenario_resource_keys["S1"] == ["xarm6", "ur5e", "xarm6-2"]
    assert scenario_resource_keys["S3"] == ["xarm6", "ur5e", "xarm6-2", "ur5e-2"]

    s1 = study["scenarios"][0]
    requirements_text = runner.render_requirements_text(s1)
    assert requirements_text.startswith("[Product Requirements]")
    assert "- assemble SG from prusa-mk4-1 to the Assembly Station." in requirements_text
    assert "- assemble MRP from prusa-mk3 to the Assembly Station." in requirements_text
    assert "- assemble LCP from prusa-mk4-2 to the Assembly Station." in requirements_text

    safety_text = runner.render_safety_text(
        s1,
        verified_rules_by_safety_file=study["verified_rules_by_safety_file"],
    )
    assert safety_text.startswith("[Safety Requirements]")
    assert "- SG must be pick_approach first before MRP pick_approach to the assembly station." in safety_text
    assert (
        "- xarm6, ur5e, and xarm6-2 should not be place_approach at the same time to the assembly station."
        in safety_text
    )
    assert "assembly board destination area" not in safety_text


def test_requirement_layout_is_auto_derived_from_selected_requirement_file() -> None:
    derived = OfflineStudyRunner.parse_requirement_file_layout(
        "cais_spade_llm/specification/products/requirements/s1_rpc_232.txt",
        project_root=ROOT,
        valid_parts={"SG", "MRP", "LCP"},
    )

    assert derived["derived"] is True
    assert derived["parts"] == ["SG", "MRP", "LCP"]
    assert derived["part_order"] == ["SG", "MRP", "LCP"]
    assert derived["part_sources"] == {
        "SG": "prusa-mk4-1",
        "MRP": "prusa-mk3",
        "LCP": "prusa-mk4-2",
    }


def test_validator_reports_stats_for_simple_fsa() -> None:
    rules = [
        {
            "id": "SAFE_1",
            "raw_text": "Dummy rule",
            "ltlf": "G ! bad",
            "aps": [
                {
                    "label": "ap1",
                    "full": "ap/assembly/any/robot/pick_approach/any",
                }
            ],
        }
    ]
    dfa_map = {
        "SAFE_1": (
            "digraph { "
            "init -> 1; "
            "node [shape=doublecircle]; 1; "
            "1 -> 1 [label=\"!ap1\"]; "
            "1 -> 2 [label=\"ap1\"]; "
            "2 -> 2 [label=\"true\"]; "
            "}"
        )
    }
    validator = PlanSafetyValidator(rules=rules, dfa_map=dfa_map, tools_catalog=[])
    ok, violations = validator.validate_plan_fsa(
        fsa={
            "A": {
                "Tr": [{"from": "x0", "event": "noop", "to": "x1"}],
                "x0": "x0",
                "Xm": ["x1"],
            }
        },
        plan=None,
        product_jid="assembly_board-v1@localhost",
    )

    assert ok is True
    assert violations == []
    assert validator.last_run_stats["fsa_reachable_states"] == 2
    assert validator.last_run_stats["fsa_transitions"] == 1
    assert validator.last_run_stats["total_product_states_explored"] >= 2
    assert validator.last_run_stats["verification_time_ms"] >= 0.0
    assert validator.last_run_stats["product_state_limit_hit"] is False
    assert validator.last_run_stats["per_rule_product_states_explored"]["SAFE_1"] >= 2


def test_build_paired_comparison_rows_and_markdown() -> None:
    aggregate_rows = [
        {
            "scenario_id": "S1",
            "method": "llm_nl_safety",
            "robots": 2,
            "parts": 2,
            "safety_rules": 1,
            "trials": 10,
            "valid_rate": 0.6,
            "first_pass_validity_rate": 0.6,
            "rule_satisfaction_rate": 0.6,
            "first_pass_rule_satisfaction_rate": 0.6,
            "avg_cumulative_validation_time_ms": 4.2,
        },
        {
            "scenario_id": "S1",
            "method": "verified",
            "robots": 2,
            "parts": 2,
            "safety_rules": 1,
            "trials": 10,
            "valid_rate": 0.9,
            "final_verified_validity_rate": 0.9,
            "rule_satisfaction_rate": 0.9,
            "final_verified_rule_satisfaction_rate": 0.9,
            "avg_auto_replans_used": 0.6,
            "avg_cumulative_validation_time_ms": 42.3,
        },
    ]

    paired_rows = OfflineStudyRunner.build_paired_comparison_rows(aggregate_rows)

    assert paired_rows == [
        {
            "scenario_id": "S1",
            "robots": 2,
            "parts": 2,
            "safety_rules": 1,
            "pure_llm_trials": 10,
            "verified_trials": 10,
            "paired_trials": 10,
            "pure_llm_rule_satisfaction_rate": 0.6,
            "verified_rule_satisfaction_rate": 0.9,
            "mean_verified_repair_attempts": 0.6,
            "mean_cumulative_verified_time_ms": 42.3,
            "delta_rule_satisfaction_rate": 0.3,
        }
    ]

    markdown = OfflineStudyRunner.build_markdown_table(aggregate_rows)
    assert "| Scenario | Scale (Robots / Parts / Rules) | Trials | Pure LLM Rule Satisfaction % (single-shot) | LLM + Formal Verification Rule Satisfaction % (<=5 repairs) | Mean Verified Repair Attempts | Mean Cumulative Verified Time (ms) | Delta |" in markdown
    assert "| S1 | 2 / 2 / 1 | 10 | 60.0% | 90.0% | 0.6 | 42.3 | +30.0 pp |" in markdown
    assert "Invalid Rate" not in markdown
    assert "Avg Verify Time" not in markdown
    assert "Avg Product States" not in markdown
    assert "Avg Auto-Replans" not in markdown


def test_build_markdown_table_includes_incomplete_scenarios() -> None:
    markdown = OfflineStudyRunner.build_markdown_table(
        [
            {
                "scenario_id": "S1",
                "method": "llm_nl_safety",
                "robots": 2,
                "parts": 2,
                "safety_rules": 1,
                "trials": 10,
                "valid_rate": 0.6,
                "first_pass_validity_rate": 0.6,
                "rule_satisfaction_rate": 0.6,
                "first_pass_rule_satisfaction_rate": 0.6,
            }
        ]
    )

    assert "| Scenario | Scale (Robots / Parts / Rules) | Trials |" in markdown
    assert "| S1 | 2 / 2 / 1 | 10 | 60.0% | N/A | N/A | N/A | N/A |" in markdown


def test_normalize_selected_safety_rules_repairs_stale_precedence_ltlf() -> None:
    stale_rule = {
        "id": "SAFE_1",
        "raw_text": "LG by xarm6 must place_approach first before MCP by ur5e to the assembly station.",
        "constraint_type": "ordering_precedence",
        "process": "assembly",
        "product": ["LG", "MCP"],
        "resources": ["xarm6", "ur5e"],
        "resource_types": ["robot"],
        "event": "place_approach",
        "context": {"destination": "assembly_board-v1"},
        "aps": [
            {
                "label": "ap004",
                "full": "ap_event/assembly/mcp/ur5e/place_approach/destination=assembly_board-v1",
            },
            {
                "label": "ap003",
                "full": "ap_event/assembly/lg/xarm6/place_approach/destination=assembly_board-v1",
            },
        ],
        "ltlf": "G ((ap004 -> ap003))",
        "generated_interpretation": "stale",
    }

    normalized = OfflineStudyRunner._normalize_selected_safety_rules([stale_rule])

    assert normalized[0]["ltlf"] == "((!ap004) U ap003)"
    assert "xarm6/place_approach" in normalized[0]["generated_interpretation"]
    assert "ur5e/place_approach" in normalized[0]["generated_interpretation"]


def test_run_offline_repair_loop_tracks_cumulative_validation_time() -> None:
    class _FakePlanner:
        def __init__(self) -> None:
            self.nodes = [{"id": "REQ_1_T1", "type": "task"}]
            self.global_fsa = {"A": {"x0": "x0", "Xm": ["x1"], "Tr": []}}

        def get_last_grounding_summary(self) -> dict[str, Any]:
            return {
                "corrected_task_count": 0,
                "invalid_task_count": 0,
                "unresolved_task_count": 0,
                "findings": [],
            }

        async def replan_with_feedback_offline(self, violations: list[dict[str, Any]]) -> None:
            assert violations
            self.nodes = [{"id": "REQ_1_T2", "type": "task"}]

        def compile_global_fsa(self) -> None:
            return None

    class _FakeProductAgent:
        def __init__(self) -> None:
            self.process_planner = _FakePlanner()

    class _FakeValidator:
        def __init__(self) -> None:
            self.calls = 0
            self.last_run_stats: dict[str, Any] = {}

        def validate_plan_fsa(self, **_: Any) -> tuple[bool, list[dict[str, Any]]]:
            self.calls += 1
            if self.calls == 1:
                self.last_run_stats = {"verification_time_ms": 4.5}
                return False, [{"violated_rule_id": "SAFE_1"}]
            self.last_run_stats = {"verification_time_ms": 7.5}
            return True, []

    payload = asyncio.run(
        BundleCompiler.run_offline_repair_loop(
            product_agent=_FakeProductAgent(),
            validator=_FakeValidator(),
            product_jid="assembly_board-v1@localhost",
            auto_replan_max_attempts=5,
        )
    )

    assert payload["ok"] is True
    assert payload["auto_replans_used"] == 1
    assert payload["stop_reason"] == "repaired_valid"
    assert payload["validation_call_count"] == 2
    assert payload["cumulative_validation_time_ms"] == 12.0
    assert [row["phase"] for row in payload["repair_history"]] == [
        "validation",
        "repair",
        "validation",
    ]


def test_run_offline_repair_loop_retries_after_compile_cycle_and_later_repairs() -> None:
    cycle_message = (
        "failed to derive deterministic same-resource requirement block order "
        "for ur5e@localhost: REQ_2, REQ_3; block dependency evidence: "
        "REQ_2 -> REQ_3 via REQ_2_T4 -> REQ_3_T3"
    )

    class _FakePlanner:
        def __init__(self) -> None:
            self.nodes = [{"id": "REQ_2_T3", "type": "task", "predecessors": []}]
            self.global_fsa = {"A": {"x0": "x0", "Xm": ["x1"], "Tr": []}}
            self.attempts = 0
            self.feedbacks: list[list[dict[str, Any]]] = []

        def get_last_grounding_summary(self) -> dict[str, Any]:
            return {
                "corrected_task_count": 0,
                "invalid_task_count": 0,
                "unresolved_task_count": 0,
                "findings": [],
            }

        async def replan_with_feedback_offline(self, violations: list[dict[str, Any]]) -> None:
            self.feedbacks.append(violations)
            self.attempts += 1
            if self.attempts == 1:
                self.nodes = [
                    {"id": "REQ_2_T3", "type": "task", "predecessors": ["REQ_3_T4"]},
                    {"id": "REQ_3_T3", "type": "task", "predecessors": ["REQ_2_T4"]},
                ]
            else:
                self.nodes = [{"id": "REQ_2_T3", "type": "task", "predecessors": []}]

        def compile_global_fsa(self) -> None:
            if self.attempts == 1:
                raise ValueError(cycle_message)
            self.global_fsa = {"A": {"x0": "x0", "Xm": ["x1"], "Tr": []}}

        def _validate_task_graph(self, nodes: list[dict[str, Any]]) -> None:
            return None

    class _FakeProductAgent:
        def __init__(self) -> None:
            self.process_planner = _FakePlanner()

    class _FakeValidator:
        rules = [{"id": "SAFE_1"}, {"id": "SAFE_2"}, {"id": "SAFE_3"}]

        def __init__(self) -> None:
            self.calls = 0
            self.last_run_stats: dict[str, Any] = {}

        def validate_plan_fsa(self, **_: Any) -> tuple[bool, list[dict[str, Any]]]:
            self.calls += 1
            self.last_run_stats = {"verification_time_ms": 1.0}
            if self.calls == 1:
                return (
                    False,
                    [
                        {
                            "violated_rule_id": "SAFE_3",
                            "witness_task_ids": ["REQ_2_T3"],
                        }
                    ],
                )
            return True, []

    product_agent = _FakeProductAgent()
    payload = asyncio.run(
        BundleCompiler.run_offline_repair_loop(
            product_agent=product_agent,
            validator=_FakeValidator(),
            product_jid="assembly_board-v1@localhost",
            auto_replan_max_attempts=5,
        )
    )

    assert payload["ok"] is True
    assert payload["stop_reason"] == "repaired_valid"
    assert payload["auto_replans_used"] == 2
    assert payload["validation_call_count"] == 2
    history = payload["repair_history"]
    assert [row["phase"] for row in history] == [
        "validation",
        "repair",
        "repair",
        "validation",
    ]
    assert history[0]["satisfied_rule_count"] == 2
    assert history[0]["safety_rule_count"] == 3
    assert history[1]["compile_ok"] is False
    assert "REQ_2 -> REQ_3" in history[1]["error_message"]
    assert history[1]["retry_planned"] is True
    assert history[2]["compile_ok"] is True
    assert history[-1]["satisfied_rule_count"] == 3
    synthetic_feedback = product_agent.process_planner.feedbacks[1]
    assert synthetic_feedback[0]["violated_rule_id"] == "REPAIR_GRAPH_INVALID"
    assert "same-resource requirement block order" in synthetic_feedback[0]["violation_text"]


def test_run_offline_repair_loop_uses_all_attempts_for_repeated_compile_errors() -> None:
    class _FakePlanner:
        def __init__(self) -> None:
            self.nodes = [{"id": "REQ_2_T3", "type": "task", "predecessors": []}]
            self.global_fsa = {"A": {"x0": "x0", "Xm": ["x1"], "Tr": []}}
            self.attempts = 0

        def get_last_grounding_summary(self) -> dict[str, Any]:
            return {
                "corrected_task_count": 0,
                "invalid_task_count": 0,
                "unresolved_task_count": 0,
                "findings": [],
            }

        async def replan_with_feedback_offline(self, violations: list[dict[str, Any]]) -> None:
            assert violations
            self.attempts += 1
            self.nodes = [
                {
                    "id": "REQ_2_T3",
                    "type": "task",
                    "predecessors": [f"REQ_3_T{self.attempts}"],
                }
            ]

        def compile_global_fsa(self) -> None:
            raise ValueError(
                "failed to derive deterministic same-resource requirement block order "
                "for ur5e@localhost: REQ_2, REQ_3"
            )

        def _validate_task_graph(self, nodes: list[dict[str, Any]]) -> None:
            return None

    class _FakeProductAgent:
        def __init__(self) -> None:
            self.process_planner = _FakePlanner()

    class _FakeValidator:
        rules = [{"id": "SAFE_1"}]

        def __init__(self) -> None:
            self.last_run_stats: dict[str, Any] = {}

        def validate_plan_fsa(self, **_: Any) -> tuple[bool, list[dict[str, Any]]]:
            self.last_run_stats = {"verification_time_ms": 1.0}
            return False, [{"violated_rule_id": "SAFE_1", "witness_task_ids": ["REQ_2_T3"]}]

    payload = asyncio.run(
        BundleCompiler.run_offline_repair_loop(
            product_agent=_FakeProductAgent(),
            validator=_FakeValidator(),
            product_jid="assembly_board-v1@localhost",
            auto_replan_max_attempts=5,
        )
    )

    assert payload["ok"] is False
    assert payload["stop_reason"] == "repair_exception"
    assert payload["auto_replans_used"] == 5
    assert payload["validation_call_count"] == 1
    repair_entries = [
        row for row in payload["repair_history"] if row.get("phase") == "repair"
    ]
    assert len(repair_entries) == 5
    assert all(row["compile_ok"] is False for row in repair_entries)
    assert repair_entries[-1]["retry_planned"] is False


def test_repair_progress_from_history_counts_initial_and_per_repair_corrections() -> None:
    progress = OfflineStudyRunner._repair_progress_from_history(
        [
            {
                "phase": "validation",
                "attempt_index": 0,
                "violated_rule_count": 2,
                "satisfied_rule_count": 1,
                "safety_rule_count": 3,
            },
            {
                "phase": "repair",
                "attempt_index": 1,
                "compile_ok": False,
            },
            {
                "phase": "repair",
                "attempt_index": 2,
                "compile_ok": True,
            },
            {
                "phase": "validation",
                "attempt_index": 2,
                "violated_rule_count": 1,
                "satisfied_rule_count": 2,
                "safety_rule_count": 3,
            },
            {
                "phase": "repair",
                "attempt_index": 3,
                "compile_ok": True,
            },
            {
                "phase": "validation",
                "attempt_index": 3,
                "violated_rule_count": 0,
                "satisfied_rule_count": 3,
                "safety_rule_count": 3,
            },
        ],
        fallback_violated_rule_count=0,
    )

    assert progress["initial_violated_rule_count"] == 2
    assert progress["final_violated_rule_count"] == 0
    assert progress["repair_corrections_by_attempt"] == {1: 0.0, 2: 1.0, 3: 1.0}
    assert progress["repair_correction_summary"] == "R1: +0.00, R2: +1.00, R3: +1.00"


@pytest.mark.usefixtures("monkeypatch")
def test_experiment_runner_creates_outputs_and_tools_catalogue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_manifest = _load_json(ROOT / "writing" / "experiments" / "offline_planning_experiment_pack.json")
    base_manifest["defaults"]["trials_per_method"] = 1
    base_manifest["defaults"]["methods"] = ["llm_nl_safety", "verified"]
    for scenario in base_manifest["scenarios"]:
        scenario["trials"] = 1

    manifest_path = tmp_path / "study.json"
    manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")

    compile_calls: list[dict[str, Any]] = []

    async def _fake_compile_bundle(self, **kwargs: Any) -> dict[str, Any]:
        compile_calls.append(dict(kwargs))
        auto_replan_max_attempts = int(kwargs.get("auto_replan_max_attempts", 0) or 0)
        ok = auto_replan_max_attempts > 0
        repair_history = (
            [
                {
                    "phase": "validation",
                    "attempt_index": 0,
                    "ok": False,
                    "violated_rule_count": 1,
                    "satisfied_rule_count": 0,
                    "safety_rule_count": 1,
                    "violated_rules": ["SAFE_001"],
                    "witness_count": 1,
                    "stop_reason": "violations_found",
                },
                {
                    "phase": "repair",
                    "attempt_index": 1,
                    "compile_ok": True,
                    "changed_task_ids": ["REQ_1_T2"],
                },
                {
                    "phase": "validation",
                    "attempt_index": 1,
                    "ok": True,
                    "violated_rule_count": 0,
                    "satisfied_rule_count": 1,
                    "safety_rule_count": 1,
                    "violated_rules": [],
                    "witness_count": 0,
                    "stop_reason": "repaired_valid",
                },
            ]
            if ok
            else [
                {
                    "phase": "validation",
                    "attempt_index": 0,
                    "ok": False,
                    "violated_rule_count": 1,
                    "satisfied_rule_count": 0,
                    "safety_rule_count": 1,
                    "violated_rules": ["SAFE_001"],
                    "witness_count": 1,
                    "stop_reason": "max_attempts_reached",
                }
            ]
        )
        validation_summary = {
            "ok": ok,
            "violated_rules": [] if ok else ["SAFE_001"],
            "witness_count": 0 if ok else 1,
            "auto_replans_used": 1 if ok else 0,
            "stop_reason": "repaired_valid" if ok else "max_attempts_reached",
            "cumulative_validation_time_ms": 9.0 if ok else 4.5,
            "validation_call_count": 2 if ok else 1,
            "repair_history": repair_history,
            "grounding_summary": {
                "corrected_task_count": 0,
                "invalid_task_count": 0,
                "unresolved_task_count": 0,
                "findings": [],
            },
            "validator_stats": {
                "fsa_reachable_states": 11,
                "fsa_transitions": 17,
                "per_rule_product_states_explored": {"SAFE_001": 23},
                "total_product_states_explored": 23,
                "verification_time_ms": 4.5,
                "product_state_limit_hit": False,
            },
        }
        return {
            "bundle_id": "fake-bundle",
            "bundle_dir": str(self.store.root_dir),
            "manifest": {
                "bundle_id": "fake-bundle",
                "validation_summary": validation_summary,
            },
            "summary": {},
        }

    monkeypatch.setattr(BundleCompiler, "compile_bundle", _fake_compile_bundle)

    def _fake_build_tools_catalogue(
        self,
        *,
        resources_dir: Path,
        cca_init_file: Path,
        robot_env: str,
        execution_mode: str,
        tools_path: Path,
    ) -> None:
        rows = []
        for path in sorted(resources_dir.glob("*.json")):
            payload = _load_json(path)
            resource_name = next(iter(payload.keys()))
            rows.append(
                {
                    "function_owner_agent": resource_name,
                    "function": "pick_approach",
                }
            )
        tools_path.parent.mkdir(parents=True, exist_ok=True)
        tools_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    monkeypatch.setattr(OfflineStudyRunner, "_build_tools_catalogue", _fake_build_tools_catalogue)

    runner = OfflineStudyRunner(
        manifest_path,
        project_root=ROOT,
        results_root=tmp_path / "results",
    )
    result = runner.run()

    run_root = Path(result["run_root"])
    assert run_root.exists()
    assert Path(result["aggregate_summary_json"]).exists()
    assert Path(result["aggregate_summary_csv"]).exists()
    assert Path(result["paper_table_markdown"]).exists()

    aggregate_rows = result["aggregate_rows"]
    assert len(aggregate_rows) == 6
    by_key = {(row["scenario_id"], row["method"]): row for row in aggregate_rows}
    s1_rule_count = len(base_manifest["scenarios"][0]["enabled_safety_rule_ids"])
    expected_s1_first_pass_rule_rate = (s1_rule_count - 1) / s1_rule_count
    assert by_key[("S1", "llm_nl_safety")]["first_pass_validity_rate"] == 0.0
    assert by_key[("S1", "llm_nl_safety")]["first_pass_rule_satisfaction_rate"] == expected_s1_first_pass_rule_rate
    assert by_key[("S1", "verified")]["final_verified_validity_rate"] == 1.0
    assert by_key[("S1", "verified")]["final_verified_rule_satisfaction_rate"] == 1.0
    assert by_key[("S1", "verified")]["avg_initial_violated_rules"] == 1.0
    assert by_key[("S1", "verified")]["avg_final_violated_rules"] == 0.0
    assert by_key[("S1", "verified")]["repair_correction_summary"] == "R1: +1.00"
    paired_rows = result["paired_comparison_rows"]
    assert len(paired_rows) == 3
    paired_by_scenario = {row["scenario_id"]: row for row in paired_rows}
    assert paired_by_scenario["S1"]["safety_rules"] == len(
        base_manifest["scenarios"][0]["enabled_safety_rule_ids"]
    )
    assert paired_by_scenario["S1"]["paired_trials"] == 1
    assert paired_by_scenario["S1"]["pure_llm_rule_satisfaction_rate"] == expected_s1_first_pass_rule_rate
    assert paired_by_scenario["S1"]["verified_rule_satisfaction_rate"] == 1.0
    assert paired_by_scenario["S1"]["mean_verified_repair_attempts"] == 1.0
    assert paired_by_scenario["S1"]["mean_cumulative_verified_time_ms"] == 9.0
    assert paired_by_scenario["S1"]["delta_rule_satisfaction_rate"] == round(
        1.0 - expected_s1_first_pass_rule_rate,
        12,
    )
    assert "| Scenario | Scale (Robots / Parts / Rules) | Trials |" in result["markdown_preview"]
    assert "Pure LLM Rule Satisfaction % (single-shot)" in result["markdown_preview"]
    assert "LLM + Formal Verification Rule Satisfaction % (<=5 repairs)" in result["markdown_preview"]
    assert "Mean Verified Repair Attempts" in result["markdown_preview"]
    assert "Mean Cumulative Verified Time (ms)" in result["markdown_preview"]
    assert "Avg Verify Time" not in result["markdown_preview"]
    assert "Avg Product States" not in result["markdown_preview"]
    assert "Avg Auto-Replans" not in result["markdown_preview"]

    s1_requirements = (
        run_root / "workspaces" / "S1" / "spec" / "products" / "requirements" / "S1_requirements.txt"
    ).read_text(encoding="utf-8")
    assert "- assemble SG from prusa-mk4-1 to the Assembly Station." in s1_requirements
    assert "- assemble MRP from prusa-mk3 to the Assembly Station." in s1_requirements
    assert "- assemble LCP from prusa-mk4-2 to the Assembly Station." in s1_requirements

    s1_safety = (run_root / "workspaces" / "S1" / "spec" / "safety" / "S1_safety.txt").read_text(
        encoding="utf-8"
    )
    assert "- SG must be pick_approach first before MRP pick_approach to the assembly station." in s1_safety
    assert (
        "- xarm6, ur5e, and xarm6-2 should not be place_approach at the same time to the assembly station."
        in s1_safety
    )
    assert "assembly board destination area" not in s1_safety

    tools_payload = _load_json(run_root / "workspaces" / "S3" / "catalog" / "tools.json")
    owners = {str(row.get("function_owner_agent")) for row in tools_payload}
    assert "xarm6-2" in owners
    assert "ur5e-2" in owners

    llm_calls = [call for call in compile_calls if int(call.get("auto_replan_max_attempts", 0) or 0) == 0]
    verified_calls = [call for call in compile_calls if int(call.get("auto_replan_max_attempts", 0) or 0) > 0]
    assert llm_calls
    assert verified_calls
    assert all(isinstance(call.get("precomputed_safety_artifacts"), dict) for call in llm_calls)
    assert all(isinstance(call.get("precomputed_safety_artifacts"), dict) for call in verified_calls)
    assert all(
        str(call["precomputed_safety_artifacts"].get("safety_logic_json", "")).endswith("cca_safety_logic.json")
        for call in llm_calls + verified_calls
    )
    s2_calls = [
        call
        for call in compile_calls
        if "S2_requirements" in str(call.get("product_requirement_file", ""))
    ]
    assert s2_calls
    for call in s2_calls:
        assert call["selected_resource_keys"] == ["xarm6", "ur5e", "xarm6-2"]
        assert any(
            Path(resource_file).name == "robot_xarm6_2.json"
            for resource_file in call["resource_files"]
        )

    raw_trial = run_root / "raw" / "S2" / "llm_nl_safety" / "trial_01.json"
    assert raw_trial.exists()
    raw_payload = _load_json(raw_trial)
    assert raw_payload["trial_record"]["method"] == "llm_nl_safety"
    assert raw_payload["trial_record"]["fsa_reachable_states"] == 11
    assert raw_payload["trial_record"]["cumulative_validation_time_ms"] == 4.5
    assert raw_payload["trial_record"]["validation_call_count"] == 1
    assert raw_payload["trial_record"]["repair_history"][0]["phase"] == "validation"
    assert raw_payload["trial_record"]["initial_violated_rule_count"] == 1
    assert raw_payload["trial_record"]["final_violated_rule_count"] == 1
    assert raw_payload["trial_record"]["repair_correction_summary"] == "none"


def test_grounding_invalid_trials_report_rule_satisfaction_as_not_applicable() -> None:
    runner = OfflineStudyRunner(ROOT / "writing" / "experiments" / "offline_planning_experiment_pack.json")
    trial_row = {
        "scenario_id": "S2",
        "method": "verified",
        "robots": 3,
        "parts": 4,
        "safety_rules": 3,
        "ok": False,
        "stop_reason": "grounding_invalid",
        "violated_rules": [],
        "violated_rule_count": 0,
        "rule_satisfaction_evaluated": False,
        "rule_satisfaction_rate": None,
        "validation_call_count": 0,
    }

    aggregate_rows = runner._aggregate_rows(
        study={
            "scenarios": [
                {
                    "id": "S2",
                    "resource_count": 3,
                    "part_count": 4,
                    "safety_rule_count": 3,
                }
            ]
        },
        trial_rows=[trial_row],
    )
    paired_rows = OfflineStudyRunner.build_paired_comparison_rows(aggregate_rows)

    assert OfflineStudyRunner._trial_rule_satisfaction_rate(trial_row) is None
    assert aggregate_rows[0]["rule_satisfaction_rate"] is None
    assert aggregate_rows[0]["final_verified_rule_satisfaction_rate"] is None
    assert paired_rows[0]["verified_rule_satisfaction_rate"] is None


@pytest.mark.usefixtures("monkeypatch")
def test_experiment_runner_can_run_selected_scenario_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_manifest = _load_json(ROOT / "writing" / "experiments" / "offline_planning_experiment_pack.json")
    base_manifest["defaults"]["trials_per_method"] = 1
    base_manifest["defaults"]["methods"] = ["llm_nl_safety"]
    for scenario in base_manifest["scenarios"]:
        scenario["trials"] = 1

    manifest_path = tmp_path / "study.json"
    manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")

    async def _fake_compile_bundle(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "bundle_id": "fake-bundle",
            "bundle_dir": str(self.store.root_dir / "bundles" / "fake-bundle"),
            "manifest": {
                "bundle_id": "fake-bundle",
                "validation_summary": {
                    "ok": False,
                    "violated_rules": ["SAFE_001"],
                    "witness_count": 1,
                    "auto_replans_used": 0,
                    "stop_reason": "max_attempts_reached",
                    "grounding_summary": {
                        "corrected_task_count": 0,
                        "invalid_task_count": 0,
                        "unresolved_task_count": 0,
                        "findings": [],
                    },
                    "validator_stats": {
                        "fsa_reachable_states": 5,
                        "fsa_transitions": 7,
                        "per_rule_product_states_explored": {"SAFE_001": 9},
                        "total_product_states_explored": 9,
                        "verification_time_ms": 2.5,
                        "product_state_limit_hit": False,
                    },
                },
            },
            "summary": {},
        }

    monkeypatch.setattr(BundleCompiler, "compile_bundle", _fake_compile_bundle)

    def _fake_build_tools_catalogue(
        self,
        *,
        resources_dir: Path,
        cca_init_file: Path,
        robot_env: str,
        execution_mode: str,
        tools_path: Path,
    ) -> None:
        tools_path.parent.mkdir(parents=True, exist_ok=True)
        tools_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(OfflineStudyRunner, "_build_tools_catalogue", _fake_build_tools_catalogue)

    runner = OfflineStudyRunner(
        manifest_path,
        project_root=ROOT,
        results_root=tmp_path / "results",
    )
    result = runner.run(scenario_id="S2")

    run_root = Path(result["run_root"])
    assert result["executed_scenario_ids"] == ["S2"]
    assert len(result["aggregate_rows"]) == 1
    assert result["aggregate_rows"][0]["scenario_id"] == "S2"
    assert (run_root / "raw" / "S2" / "llm_nl_safety" / "trial_01.json").exists()
    assert not (run_root / "raw" / "S1").exists()
    normalized_study = _load_json(run_root / "normalized_study.json")
    assert [scenario["id"] for scenario in normalized_study["scenarios"]] == ["S2"]


def test_run_summary_and_trial_detail_loading(tmp_path: Path) -> None:
    manifest_path = ROOT / "writing" / "experiments" / "offline_planning_experiment_pack.json"
    results_root = tmp_path / "results"
    run_root = results_root / "offline_planning_experiment_pack" / "20260410T000000Z"
    bundle_dir = run_root / "trials" / "S1" / "llm_nl_safety" / "trial_01" / "bundle_store" / "bundles" / "fake"
    (bundle_dir / "plan").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "validation").mkdir(parents=True, exist_ok=True)

    plan_json = {"nodes": [{"id": "REQ_1_T1", "type": "task", "function_name": "pick_approach", "predecessors": []}]}
    global_fsa = {"A": {"X": ["x0", "x1"], "Tr": [{"from": "x0", "to": "x1", "event": "noop"}], "x0": "x0", "Xm": ["x1"]}}
    requirements_json = {"nodes": [{"id": "REQ_1", "type": "requirement"}]}
    validation_json = {"ok": False}

    (bundle_dir / "plan" / "plan.json").write_text(json.dumps(plan_json, indent=2), encoding="utf-8")
    (bundle_dir / "plan" / "global_fsa.json").write_text(json.dumps(global_fsa, indent=2), encoding="utf-8")
    (bundle_dir / "plan" / "requirements.json").write_text(json.dumps(requirements_json, indent=2), encoding="utf-8")
    (bundle_dir / "validation" / "plan_validation.json").write_text(json.dumps(validation_json, indent=2), encoding="utf-8")

    workspace_requirements = run_root / "workspaces" / "S1" / "spec" / "products" / "requirements" / "S1_requirements.txt"
    workspace_safety = run_root / "workspaces" / "S1" / "spec" / "safety" / "S1_safety.txt"
    workspace_requirements.parent.mkdir(parents=True, exist_ok=True)
    workspace_safety.parent.mkdir(parents=True, exist_ok=True)
    workspace_requirements.write_text("[Product Requirements]\n- assemble LG from prusa-mk4-1 to the Assembly Station.\n", encoding="utf-8")
    workspace_safety.write_text("[Safety Requirements]\n- xarm6 and ur5e must not both be in the assembly board destination area at the same time.\n", encoding="utf-8")

    aggregate_summary = {
        "study_id": "offline_planning_experiment_pack",
        "run_id": "20260410T000000Z",
        "run_root": str(run_root),
        "aggregate_rows": [
            {
                "scenario_id": "S1",
                "method": "llm_nl_safety",
                "robots": 2,
                "parts": 2,
                "trials": 1,
                "valid_rate": 0.0,
                "unsafe_or_invalid_rate": 1.0,
                "avg_verification_time_ms": 5.0,
                "avg_total_product_states_explored": 20.0,
                "avg_auto_replans_used": 0.0,
            }
        ],
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "aggregate_summary.json").write_text(json.dumps(aggregate_summary, indent=2), encoding="utf-8")
    (run_root / "paper_table.md").write_text("| Scenario |\n", encoding="utf-8")

    raw_trial_dir = run_root / "raw" / "S1" / "llm_nl_safety"
    raw_trial_dir.mkdir(parents=True, exist_ok=True)
    raw_trial_payload = {
        "trial_record": {
            "scenario_id": "S1",
            "method": "llm_nl_safety",
            "trial_index": 1,
            "ok": False,
            "witness_count": 1,
            "verification_time_ms": 5.0,
        },
        "workspace": {
            "requirements_file": str(workspace_requirements),
            "safety_file": str(workspace_safety),
        },
        "compile_result": {
            "bundle_dir": str(bundle_dir),
            "manifest": {
                "validation_summary": {"ok": False},
                "artifacts": {
                    "plan_json": "plan/plan.json",
                    "global_fsa_json": "plan/global_fsa.json",
                    "requirements_json": "plan/requirements.json",
                    "plan_validation_json": "validation/plan_validation.json",
                },
            },
        },
    }
    (raw_trial_dir / "trial_01.json").write_text(json.dumps(raw_trial_payload, indent=2), encoding="utf-8")

    summary = OfflineStudyRunner.load_run_summary(
        manifest_path,
        run_root=run_root,
        project_root=ROOT,
        results_root=results_root,
    )
    assert summary is not None
    assert summary["groups"][0]["method"] == "llm_nl_safety"
    assert summary["groups"][0]["method_label"] == "Pure LLM"
    assert summary["groups"][0]["trials_list"][0]["trial_index"] == 1
    assert summary["paired_comparison_rows"][0]["scenario_id"] == "S1"
    expected_s1_safety_rules = len(
        _load_json(manifest_path)["scenarios"][0]["enabled_safety_rule_ids"]
    )
    assert summary["paired_comparison_rows"][0]["safety_rules"] == expected_s1_safety_rules
    assert summary["paired_comparison_rows"][0]["pure_llm_rule_satisfaction_rate"] == 0.0
    assert summary["paired_comparison_rows"][0]["verified_rule_satisfaction_rate"] is None

    detail = OfflineStudyRunner.load_trial_details(
        manifest_path,
        run_root=run_root,
        scenario_id="S1",
        method="llm_nl_safety",
        trial_index=1,
        project_root=ROOT,
        results_root=results_root,
    )
    assert detail["method"] == "llm_nl_safety"
    assert detail["plan_json"]["nodes"][0]["id"] == "REQ_1_T1"
    assert detail["global_fsa_json"]["A"]["x0"] == "x0"
    assert "assemble LG from prusa-mk4-1" in detail["requirements_text"]

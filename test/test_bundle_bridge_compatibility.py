from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "cais_spade_llm"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

import cais_spade_llm.ui.bridge as bridge_module
from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic
from cais_spade_llm.bundles.bundle_store import BundleStore
from cais_spade_llm.bundles.models import sha256_file
from cais_spade_llm.prompts import build_requirement_parse_prompt, build_task_expansion_prompt
from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.pages.products import (
    _delete_product_manifest,
    _default_product_meta,
    _normalize_product_meta,
    _unlink_geometry_from_products,
    _validate_geometry_payload,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _current_scope(bridge: SystemBridge) -> tuple[str, dict]:
    requirement_file = next((path for path in bridge.list_product_requirement_files() if str(path).strip()), "")
    safety_files = bridge.list_safety_requirement_files()
    if safety_files:
        safety_file = safety_files[0]
    else:
        fallback_safety = Path("/tmp/cais_spade_llm_test_safety.txt")
        fallback_safety.write_text("[Safety Requirements]\n- keep robots separated\n", encoding="utf-8")
        safety_file = str(fallback_safety.resolve())
    if not requirement_file:
        raise AssertionError("no product requirement file available for test scope")
    try:
        product_init = bridge._resolve_product_init_for_requirement(requirement_file)
    except Exception:
        product_files = bridge.list_product_files()
        if not product_files:
            raise
        product_init = bridge._resolve_product_context(product_files[0], include_hashes=False)
    return requirement_file, {
        "product_name": str(product_init["product_name"]),
        "product_spec_file": str(Path(requirement_file).resolve()),
        "safety_file": str(Path(safety_file).resolve()),
        "source_hashes": bridge._compute_source_hashes(
            Path(requirement_file).resolve(),
            Path(safety_file).resolve(),
        ),
    }


def test_bridge_bundle_compatibility_and_active_context(tmp_path):
    bridge = SystemBridge()
    bridge.bundle_store = BundleStore(tmp_path / "user_verified_plan")

    selected_product, ctx = _current_scope(bridge)
    bundle_id = "bundle_ok"
    bdir = bridge.bundle_store.bundle_dir(bundle_id)
    (bdir / "plan").mkdir(parents=True, exist_ok=True)
    (bdir / "safety").mkdir(parents=True, exist_ok=True)
    (bdir / "validation").mkdir(parents=True, exist_ok=True)
    (bdir / "catalog").mkdir(parents=True, exist_ok=True)

    _write_json(bdir / "plan" / "plan.json", {"nodes": []})
    _write_json(
        bdir / "plan" / "global_fsa.json",
        {"A": {"X": ["S0"], "E": [], "Tr": [], "x0": "S0", "Xm": ["S0"]}, "meta": {}},
    )
    _write_json(bdir / "plan" / "requirements.json", {"nodes": []})
    _write_json(
        bdir / "safety" / "cca_safety_logic.json",
        {"rules": [{"id": "SAFE_1", "ltlf": "G(true)", "aps": []}]},
    )
    (bdir / "safety" / "SAFE_1_dfa.dot").write_text("digraph {}", encoding="utf-8")
    _write_json(bdir / "validation" / "offline_validation.json", {"ok": True, "violations": []})
    _write_json(bdir / "catalog" / "tools.json", {"tools": ["snapshot"]})

    manifest = {
        "bundle_id": bundle_id,
        "display_name": "ok",
        "created_at_utc": "2026-03-03T00:00:00+00:00",
        "status": "verified",
        "verified": True,
        "product_spec_file": ctx["product_spec_file"],
        "safety_file": ctx["safety_file"],
        "product_name": ctx["product_name"],
        "execution_mode": "simulation",
        "robot_env": "gazebo",
        "llm_models": {"planner_model": "gpt-5", "safety_model": "gpt-5"},
        "source_hashes": {
            **ctx["source_hashes"],
            "tools_sha256": sha256_file(bdir / "catalog" / "tools.json"),
        },
        "artifacts": {
            "tools_json": "catalog/tools.json",
            "requirements_json": "plan/requirements.json",
            "plan_json": "plan/plan.json",
            "global_fsa_json": "plan/global_fsa.json",
            "safety_logic_json": "safety/cca_safety_logic.json",
            "safety_dfa_dot_files": ["safety/SAFE_1_dfa.dot"],
            "safety_dfa_png_files": [],
            "offline_validation_json": "validation/offline_validation.json",
        },
        "validation_summary": {"ok": True, "violated_rules": [], "witness_count": 0},
    }
    bridge.bundle_store.save_manifest(bdir, manifest)
    bridge.bundle_store.upsert_bundle_summary(
        {
            "bundle_id": bundle_id,
            "display_name": "ok",
            "created_at_utc": manifest["created_at_utc"],
            "status": "verified",
            "verified": True,
            "product_name": manifest["product_name"],
            "product_spec_file": manifest["product_spec_file"],
            "execution_mode": "simulation",
            "robot_env": "gazebo",
            "manifest_path": str(bridge.bundle_store.manifest_path(bundle_id)),
        }
    )
    bridge.bundle_store.set_active_bundle_id(bundle_id)

    ok, reasons = bridge.check_bundle_compatibility(
        bundle_id,
        selected_product,
        "simulation",
        "gazebo",
    )
    assert ok is True
    assert reasons == []

    bundle_ctx = bridge._resolve_active_bundle_context(
        product_spec_file=selected_product,
        execution_mode="simulation",
        robot_env="gazebo",
    )
    assert bundle_ctx is not None
    assert bundle_ctx["bundle_id"] == bundle_id
    assert Path(bundle_ctx["artifacts"]["plan_json"]).exists()
    assert Path(bundle_ctx["artifacts"]["tools_json"]).exists()

    bad_safe_path = tmp_path / "safety_other.txt"
    bad_safe_path.write_text("always keep robots separated", encoding="utf-8")
    bad_safe = str(bad_safe_path.resolve())
    eval_files = bridge.evaluate_bundle_for_files(
        bundle_id,
        ctx["product_spec_file"],
        bad_safe,
        "simulation",
        "gazebo",
    )
    assert eval_files["ok"] is False
    assert "safety_file" in eval_files["reasons"]


def test_bridge_bundle_mismatch_blocks_active_context(tmp_path):
    bridge = SystemBridge()
    bridge.bundle_store = BundleStore(tmp_path / "user_verified_plan")

    selected_product, ctx = _current_scope(bridge)

    bundle_id = "bundle_bad"
    bdir = bridge.bundle_store.bundle_dir(bundle_id)
    (bdir / "plan").mkdir(parents=True, exist_ok=True)
    (bdir / "safety").mkdir(parents=True, exist_ok=True)

    manifest = {
        "bundle_id": bundle_id,
        "status": "verified",
        "verified": True,
        "product_spec_file": ctx["product_spec_file"],
        "safety_file": ctx["safety_file"],
        "product_name": ctx["product_name"],
        "execution_mode": "simulation",
        "robot_env": "gazebo",
        "source_hashes": {**ctx["source_hashes"], "prompts_sha256": "mismatch"},
        "artifacts": {
            "requirements_json": "plan/requirements.json",
            "plan_json": "plan/plan.json",
            "global_fsa_json": "plan/global_fsa.json",
            "safety_logic_json": "safety/cca_safety_logic.json",
        },
    }
    bridge.bundle_store.save_manifest(bdir, manifest)
    bridge.bundle_store.upsert_bundle_summary(
        {
            "bundle_id": bundle_id,
            "display_name": "bad",
            "created_at_utc": "2026-03-03T00:00:00+00:00",
            "status": "verified",
            "verified": True,
            "product_name": ctx["product_name"],
            "product_spec_file": ctx["product_spec_file"],
            "execution_mode": "simulation",
            "robot_env": "gazebo",
            "manifest_path": str(bridge.bundle_store.manifest_path(bundle_id)),
        }
    )
    bridge.bundle_store.set_active_bundle_id(bundle_id)

    ok, reasons = bridge.check_bundle_compatibility(
        bundle_id,
        selected_product,
        "simulation",
        "gazebo",
    )
    assert ok is False
    assert "prompts_sha256" in reasons

    try:
        bridge._resolve_active_bundle_context(
            product_spec_file=selected_product,
            execution_mode="simulation",
            robot_env="gazebo",
        )
        assert False, "expected mismatch to raise RuntimeError"
    except RuntimeError as exc:
        assert "incompatible" in str(exc)


def test_bridge_startup_bundle_mismatch_auto_deactivates_active_bundle(tmp_path):
    bridge = SystemBridge()
    bridge.bundle_store = BundleStore(tmp_path / "user_verified_plan")

    selected_product, ctx = _current_scope(bridge)

    bundle_id = "bundle_startup_bad"
    bdir = bridge.bundle_store.bundle_dir(bundle_id)
    (bdir / "plan").mkdir(parents=True, exist_ok=True)
    (bdir / "safety").mkdir(parents=True, exist_ok=True)

    manifest = {
        "bundle_id": bundle_id,
        "status": "verified",
        "verified": True,
        "product_spec_file": ctx["product_spec_file"],
        "safety_file": ctx["safety_file"],
        "product_name": ctx["product_name"],
        "execution_mode": "simulation",
        "robot_env": "gazebo",
        "source_hashes": {**ctx["source_hashes"], "prompts_sha256": "mismatch"},
        "artifacts": {
            "requirements_json": "plan/requirements.json",
            "plan_json": "plan/plan.json",
            "global_fsa_json": "plan/global_fsa.json",
            "safety_logic_json": "safety/cca_safety_logic.json",
        },
    }
    bridge.bundle_store.save_manifest(bdir, manifest)
    bridge.bundle_store.upsert_bundle_summary(
        {
            "bundle_id": bundle_id,
            "display_name": "startup-bad",
            "created_at_utc": "2026-03-03T00:00:00+00:00",
            "status": "verified",
            "verified": True,
            "product_name": ctx["product_name"],
            "product_spec_file": ctx["product_spec_file"],
            "execution_mode": "simulation",
            "robot_env": "gazebo",
            "manifest_path": str(bridge.bundle_store.manifest_path(bundle_id)),
        }
    )
    bridge.bundle_store.set_active_bundle_id(bundle_id)

    bundle_ctx, notice = bridge._resolve_startup_bundle_context(
        product_spec_file=selected_product,
        execution_mode="simulation",
        robot_env="gazebo",
    )
    assert bundle_ctx is None
    assert notice is not None
    assert bundle_id in notice
    assert "prompts.py changed" in notice
    assert bridge.bundle_store.get_active_bundle_id() is None


def test_bridge_safety_preview_flags_placeholder_dfa_and_builds_interpretation_summary(tmp_path):
    bridge = SystemBridge()

    safety_file = tmp_path / "safety2.txt"
    safety_file.write_text(
        "[Safety Requirements]\n"
        "- xarm6 and ur5e should not enter into the assembly board at the same time.\n"
        "- xarm6 and ur5e should move back home after inserting the pins.\n",
        encoding="utf-8",
    )

    preview_dir = tmp_path / "safety_previews" / "preview_1"
    preview_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        preview_dir / "cca_safety_logic.json",
        {
            "rules": [
                {
                    "id": "SAFE_1",
                    "raw_text": "xarm6 and ur5e should not enter into the assembly board at the same time.",
                    "constraint_type": "no_concurrent_access_zone",
                    "aps": [
                        {
                            "label": "ap001",
                            "full": "ap/assembly/any/ur5e/place_approach/simultaneous=true&zone=assembly_board-v1",
                        },
                        {
                            "label": "ap002",
                            "full": "ap/assembly/any/xarm6/place_approach/simultaneous=true&zone=assembly_board-v1",
                        },
                    ],
                    "ltlf": "G !(ap001 & ap002)",
                },
                {
                    "id": "SAFE_2",
                    "raw_text": "xarm6 and ur5e should move back home after inserting the pins.",
                    "constraint_type": "post_insertion_return_home",
                    "aps": [
                        {
                            "label": "ap005",
                            "full": "ap/assembly/pins/any/place_insert/preceding_event=place_insert",
                        },
                        {
                            "label": "ap007",
                            "full": "ap/assembly/pins/xarm6/move_home/preceding_event=place_insert",
                        },
                    ],
                    "ltlf": "G (ap005 -> F ap007)",
                },
            ]
        },
    )
    placeholder_dot = (
        "digraph MONA_DFA {\n"
        " rankdir = LR;\n"
        " center = true;\n"
        " size = \"7.5,10.5\";\n"
        " edge [fontname = Courier];\n"
        " node [height = .5, width = .5];\n"
        " node [shape = doublecircle]; 0.0;\n"
        " node [shape = circle]; 1;\n"
        " init [shape = plaintext, label = \"\"];\n"
        " init -> 1;\n"
        "}\n"
    )
    (preview_dir / "SAFE_1_dfa.dot").write_text(placeholder_dot, encoding="utf-8")
    (preview_dir / "SAFE_2_dfa.dot").write_text(placeholder_dot, encoding="utf-8")

    previews_path = tmp_path / "intent_previews.json"
    original_previews = bridge_module._SAFETY_INTENT_PREVIEWS
    original_requirements_dir = bridge_module._SAFETY_REQUIREMENTS_DIR
    original_preview_dir = bridge_module._SAFETY_PREVIEW_DIR
    original_verified_root = bridge_module._USER_VERIFIED_SAFETY
    try:
        bridge_module._SAFETY_INTENT_PREVIEWS = previews_path
        bridge_module._SAFETY_REQUIREMENTS_DIR = tmp_path
        bridge_module._SAFETY_PREVIEW_DIR = preview_dir.parent
        bridge_module._USER_VERIFIED_SAFETY = tmp_path / "user_verified_safety"
        bridge._save_safety_intent_previews(
            {
                "schema_version": 1,
                "previews": {
                    str(safety_file.resolve()): [
                        {
                            "preview_id": "preview_1",
                            "generated_at_utc": "2026-03-06T19:37:31+00:00",
                            "preview_dir": str(preview_dir.resolve()),
                            "safety_logic_json": str((preview_dir / "cca_safety_logic.json").resolve()),
                            "safety_sha256": bridge_module.sha256_text(
                                safety_file.read_text(encoding="utf-8").strip()
                            ),
                            "refinement_feedback": "",
                            "parent_preview_id": "",
                        }
                    ]
                },
            }
        )

        payload = bridge.get_safety_rule_preview(str(safety_file))
    finally:
        bridge_module._SAFETY_INTENT_PREVIEWS = original_previews
        bridge_module._SAFETY_REQUIREMENTS_DIR = original_requirements_dir
        bridge_module._SAFETY_PREVIEW_DIR = original_preview_dir
        bridge_module._USER_VERIFIED_SAFETY = original_verified_root

    assert payload["available"] is True
    assert payload["preview_interpretation_summary"]
    assert "- SAFE_1:" in payload["preview_interpretation_summary"]
    assert "- SAFE_2:" in payload["preview_interpretation_summary"]

    rules = {row["id"]: row for row in payload["rules"]}
    assert rules["SAFE_1"]["generated_interpretation"]
    assert rules["SAFE_2"]["generated_interpretation"]
    expected_status = "backend_missing" if shutil.which("mona") is None else "invalid"
    assert rules["SAFE_1"]["dfa_status"] == expected_status
    assert rules["SAFE_2"]["dfa_status"] == expected_status
    assert rules["SAFE_1"]["dfa_diagnostic"]


def test_safety_logic_skips_render_for_placeholder_dfa(tmp_path, monkeypatch, caplog):
    logger = logging.getLogger("test.safety_logic.placeholder")
    controller_agent = SimpleNamespace(logger=logger)
    safety_file = tmp_path / "safety.txt"
    safety_file.write_text("[Safety Requirements]\n- placeholder\n", encoding="utf-8")

    logic = SafetyLogic(controller_agent, safety_file)
    logic.rules = [{"id": "SAFE_1", "ltlf": "G true"}]

    placeholder_dot = (
        "digraph MONA_DFA {\n"
        " rankdir = LR;\n"
        " node [shape = doublecircle]; 0.0;\n"
        " node [shape = circle]; 1;\n"
        " init [shape = plaintext, label = \"\"];\n"
        " init -> 1;\n"
        "}\n"
    )

    monkeypatch.setattr(logic, "_to_dfa_quiet", lambda _: placeholder_dot)

    render_calls: list[str] = []

    def _unexpected_render(*args, **kwargs):
        render_calls.append("called")
        raise AssertionError("Graphviz render should be skipped for placeholder DFA")

    monkeypatch.setattr("cais_spade_llm.agents.central_controller.safety_logic.Source.render", _unexpected_render)

    with caplog.at_level(logging.WARNING):
        dfas = logic.build_dfas_per_rule(tmp_path)

    assert dfas["SAFE_1"] == placeholder_dot
    assert (tmp_path / "SAFE_1_dfa.dot").exists()
    assert not (tmp_path / "SAFE_1_dfa.png").exists()
    assert render_calls == []
    assert "placeholder" in caplog.text


def test_bridge_wait_for_ros_services_checks_targets_in_single_probe():
    bridge = SystemBridge()

    calls: list[str] = []

    def _fake_ros2_command_output(command: str, **kwargs):
        calls.append(command)
        return True, "/compute_cartesian_path\n/detect_all\n"

    bridge._ros2_command_output = _fake_ros2_command_output  # type: ignore[method-assign]

    err = bridge._wait_for_ros_services(
        ["/compute_cartesian_path", "/detect_all"],
        timeout_sec=2.0,
    )
    assert err is None
    assert calls == ["ros2 service list"]


def test_bridge_save_safety_previews_prunes_old_history_and_orphans(tmp_path):
    bridge = SystemBridge()
    safety_file = tmp_path / "safety.txt"
    safety_file.write_text("[Safety Requirements]\n- keep apart\n", encoding="utf-8")

    verified_root = tmp_path / "user_verified_safety"
    preview_root = tmp_path / "safety_previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    previews_path = tmp_path / "intent_previews.json"

    entries: list[dict[str, str]] = []
    for idx in range(12):
        preview_dir = preview_root / f"preview_{idx:02d}"
        preview_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            preview_dir / "cca_safety_logic.json",
            {"rules": [{"id": "SAFE_1", "ltlf": "G true", "aps": []}]},
        )
        entries.append(
            {
                "preview_id": f"preview_{idx:02d}",
                "generated_at_utc": f"2026-03-06T19:{idx:02d}:00+00:00",
                "preview_dir": str(preview_dir.resolve()),
                "safety_logic_json": str((preview_dir / "cca_safety_logic.json").resolve()),
                "safety_sha256": "hash",
                "refinement_feedback": "",
                "parent_preview_id": "",
            }
        )

    orphan_dir = preview_root / "orphan_preview"
    orphan_dir.mkdir(parents=True, exist_ok=True)

    original_previews = bridge_module._SAFETY_INTENT_PREVIEWS
    original_requirements_dir = bridge_module._SAFETY_REQUIREMENTS_DIR
    original_preview_dir = bridge_module._SAFETY_PREVIEW_DIR
    original_verified_root = bridge_module._USER_VERIFIED_SAFETY
    try:
        bridge_module._SAFETY_INTENT_PREVIEWS = previews_path
        bridge_module._SAFETY_REQUIREMENTS_DIR = tmp_path
        bridge_module._SAFETY_PREVIEW_DIR = preview_root
        bridge_module._USER_VERIFIED_SAFETY = verified_root
        bridge._save_safety_intent_previews(
            {
                "schema_version": 1,
                "previews": {
                    str(safety_file.resolve()): entries,
                },
            }
        )

        payload = bridge._load_safety_intent_previews()
    finally:
        bridge_module._SAFETY_INTENT_PREVIEWS = original_previews
        bridge_module._SAFETY_REQUIREMENTS_DIR = original_requirements_dir
        bridge_module._SAFETY_PREVIEW_DIR = original_preview_dir
        bridge_module._USER_VERIFIED_SAFETY = original_verified_root

    kept_entries = payload["previews"][str(safety_file.resolve())]
    assert len(kept_entries) == 10
    assert kept_entries[0]["preview_id"] == "preview_00"
    assert kept_entries[-1]["preview_id"] == "preview_09"
    assert (preview_root / "preview_00").exists()
    assert (preview_root / "preview_09").exists()
    assert not (preview_root / "preview_10").exists()
    assert not (preview_root / "preview_11").exists()
    assert not orphan_dir.exists()


def test_bridge_load_safety_previews_migrates_legacy_storage(tmp_path):
    bridge = SystemBridge()
    safety_root = tmp_path / "spec_safety"
    safety_root.mkdir(parents=True, exist_ok=True)
    safety_file = safety_root / "safety1.txt"
    safety_file.write_text("[Safety Requirements]\n- keep apart\n", encoding="utf-8")

    legacy_plan_root = tmp_path / "user_verified_plan"
    legacy_preview_root = legacy_plan_root / "safety_previews"
    legacy_preview_dir = legacy_preview_root / "preview_legacy"
    legacy_preview_dir.mkdir(parents=True, exist_ok=True)
    (legacy_preview_dir / "cca_safety_logic.json").write_text(
        json.dumps({"rules": [{"id": "SAFE_1", "ltlf": "G true", "aps": []}]}),
        encoding="utf-8",
    )
    (legacy_preview_dir / "SAFE_1_dfa.dot").write_text("digraph {}", encoding="utf-8")

    verified_root = tmp_path / "user_verified_safety"
    new_preview_root = verified_root / "previews"
    new_previews_path = verified_root / "intent_previews.json"
    legacy_previews_path = safety_root / "intent_previews.json"
    _write_json(
        legacy_previews_path,
        {
            "schema_version": 1,
            "previews": {
                str(safety_file.resolve()): [
                    {
                        "preview_id": "preview_legacy",
                        "generated_at_utc": "2026-03-06T20:00:00+00:00",
                        "preview_dir": str(legacy_preview_dir.resolve()),
                        "safety_logic_json": str((legacy_preview_dir / "cca_safety_logic.json").resolve()),
                        "safety_sha256": bridge_module.sha256_text(
                            safety_file.read_text(encoding="utf-8").strip()
                        ),
                        "refinement_feedback": "",
                        "parent_preview_id": "",
                    }
                ]
            },
        },
    )

    original_requirements_dir = bridge_module._SAFETY_REQUIREMENTS_DIR
    original_user_verified_plan = bridge_module._USER_VERIFIED_PLAN
    original_user_verified_safety = bridge_module._USER_VERIFIED_SAFETY
    original_previews_path = bridge_module._SAFETY_INTENT_PREVIEWS
    original_preview_dir = bridge_module._SAFETY_PREVIEW_DIR
    try:
        bridge_module._SAFETY_REQUIREMENTS_DIR = safety_root
        bridge_module._USER_VERIFIED_PLAN = legacy_plan_root
        bridge_module._USER_VERIFIED_SAFETY = verified_root
        bridge_module._SAFETY_INTENT_PREVIEWS = new_previews_path
        bridge_module._SAFETY_PREVIEW_DIR = new_preview_root

        payload = bridge._load_safety_intent_previews()
    finally:
        bridge_module._SAFETY_REQUIREMENTS_DIR = original_requirements_dir
        bridge_module._USER_VERIFIED_PLAN = original_user_verified_plan
        bridge_module._USER_VERIFIED_SAFETY = original_user_verified_safety
        bridge_module._SAFETY_INTENT_PREVIEWS = original_previews_path
        bridge_module._SAFETY_PREVIEW_DIR = original_preview_dir

    record = payload["previews"][str(safety_file.resolve())][0]
    assert Path(record["preview_dir"]).parent == new_preview_root
    assert Path(record["preview_dir"]).exists()
    assert not legacy_preview_dir.exists()
    assert new_previews_path.exists()
    assert not legacy_previews_path.exists()


def test_bridge_save_safety_approvals_materializes_verified_files(tmp_path):
    bridge = SystemBridge()
    safety_root = tmp_path / "spec_safety"
    safety_root.mkdir(parents=True, exist_ok=True)
    approved_file = safety_root / "safety_ok.txt"
    approved_file.write_text("[Safety Requirements]\n- safe\n", encoding="utf-8")
    stale_file = safety_root / "safety_stale.txt"
    stale_file.write_text("[Safety Requirements]\n- stale\n", encoding="utf-8")

    verified_root = tmp_path / "user_verified_safety"
    approvals_path = verified_root / "intent_approvals.json"
    verified_files_dir = verified_root / "verified"

    original_user_verified_safety = bridge_module._USER_VERIFIED_SAFETY
    original_approvals_path = bridge_module._SAFETY_INTENT_APPROVALS
    original_verified_dir = bridge_module._SAFETY_VERIFIED_DIR
    try:
        bridge_module._USER_VERIFIED_SAFETY = verified_root
        bridge_module._SAFETY_INTENT_APPROVALS = approvals_path
        bridge_module._SAFETY_VERIFIED_DIR = verified_files_dir

        bridge._save_safety_intent_approvals(
            {
                "schema_version": 1,
                "approvals": {
                    str(approved_file.resolve()): {
                        "approved": True,
                        "safety_sha256": bridge_module.sha256_text(
                            approved_file.read_text(encoding="utf-8").strip()
                        ),
                    },
                    str(stale_file.resolve()): {
                        "approved": True,
                        "safety_sha256": "mismatch",
                    },
                },
            }
        )
        payload = bridge._load_safety_intent_approvals()
    finally:
        bridge_module._USER_VERIFIED_SAFETY = original_user_verified_safety
        bridge_module._SAFETY_INTENT_APPROVALS = original_approvals_path
        bridge_module._SAFETY_VERIFIED_DIR = original_verified_dir

    approved_record = payload["approvals"][str(approved_file.resolve())]
    stale_record = payload["approvals"][str(stale_file.resolve())]
    assert Path(approved_record["verified_file"]).exists()
    assert Path(approved_record["verified_file"]).read_text(encoding="utf-8") == approved_file.read_text(encoding="utf-8")
    assert "verified_file" not in stale_record
    verified_files = sorted(p.name for p in verified_files_dir.glob("*.txt"))
    assert len(verified_files) == 1


def test_safety_logic_does_not_leak_refinement_feedback_between_rules():
    mutex_rule = {
        "id": "SAFE_1",
        "raw_text": "xarm6 and ur5e should not enter into the assembly board at the same time.",
        "constraint_type": "no_concurrent_access_zone",
    }
    mutex_aps = [
        "ap/assembly/any/ur5e/pick_approach/zone=assembly_board-v1",
        "ap/assembly/any/xarm6/pick_approach/zone=assembly_board-v1",
    ]

    response_rule = {
        "id": "SAFE_2",
        "raw_text": "xarm6 should move back home after inserting the pins.",
        "constraint_type": "post_insertion_return_home",
        "event": "move_home",
    }
    response_aps = [
        "ap/assembly/pins/xarm6/place_insert/preceding_event=place_insert",
        "ap/assembly/pins/xarm6/move_home/preceding_event=place_insert",
    ]

    precedence_rule = {
        "id": "SAFE_3",
        "raw_text": "MCP must be grasped before LRP.",
        "constraint_type": "ordering",
        "product": ["mcp", "lrp"],
        "event": "pick_grasp",
    }
    precedence_aps = [
        "ap/assembly/mcp/any/pick_grasp/any",
        "ap/assembly/lrp/any/pick_grasp/any",
    ]

    refinement_feedback = "Add SAFE_3: MCP pick_grasp should happen before LRP pick_grasp."

    mutex_formula = SafetyLogic._compile_ltlf_for_rule(
        mutex_rule,
        mutex_aps,
        refinement_feedback=refinement_feedback,
    )
    response_formula = SafetyLogic._compile_ltlf_for_rule(
        response_rule,
        response_aps,
        refinement_feedback=refinement_feedback,
    )
    precedence_formula = SafetyLogic._compile_ltlf_for_rule(
        precedence_rule,
        precedence_aps,
        refinement_feedback=refinement_feedback,
    )

    assert mutex_formula == "G !(ap/assembly/any/ur5e/pick_approach/zone=assembly_board-v1 & ap/assembly/any/xarm6/pick_approach/zone=assembly_board-v1)"
    assert response_formula == "G (ap/assembly/pins/xarm6/place_insert/preceding_event=place_insert -> F ap/assembly/pins/xarm6/move_home/preceding_event=place_insert)"
    assert precedence_formula == "((!ap/assembly/lrp/any/pick_grasp/any) U ap/assembly/mcp/any/pick_grasp/any)"


def test_bridge_bundle_old_tools_hash_does_not_block_active_context(tmp_path):
    bridge = SystemBridge()
    bridge.bundle_store = BundleStore(tmp_path / "user_verified_plan")

    selected_product, ctx = _current_scope(bridge)

    bundle_id = "bundle_old_tools"
    bdir = bridge.bundle_store.bundle_dir(bundle_id)
    (bdir / "plan").mkdir(parents=True, exist_ok=True)
    (bdir / "safety").mkdir(parents=True, exist_ok=True)

    manifest = {
        "bundle_id": bundle_id,
        "status": "verified",
        "verified": True,
        "product_spec_file": ctx["product_spec_file"],
        "safety_file": ctx["safety_file"],
        "product_name": ctx["product_name"],
        "execution_mode": "simulation",
        "robot_env": "gazebo",
        "source_hashes": {**ctx["source_hashes"], "tools_sha256": "old-tools-hash"},
        "artifacts": {
            "requirements_json": "plan/requirements.json",
            "plan_json": "plan/plan.json",
            "global_fsa_json": "plan/global_fsa.json",
            "safety_logic_json": "safety/cca_safety_logic.json",
        },
    }
    bridge.bundle_store.save_manifest(bdir, manifest)
    bridge.bundle_store.upsert_bundle_summary(
        {
            "bundle_id": bundle_id,
            "display_name": "old-tools",
            "created_at_utc": "2026-03-03T00:00:00+00:00",
            "status": "verified",
            "verified": True,
            "product_name": ctx["product_name"],
            "product_spec_file": ctx["product_spec_file"],
            "execution_mode": "simulation",
            "robot_env": "gazebo",
            "manifest_path": str(bridge.bundle_store.manifest_path(bundle_id)),
        }
    )
    bridge.bundle_store.set_active_bundle_id(bundle_id)

    ok, reasons = bridge.check_bundle_compatibility(
        bundle_id,
        selected_product,
        "simulation",
        "gazebo",
    )
    assert ok is True
    assert "tools_sha256" not in reasons


def test_bridge_bundle_snapshot_tools_hash_allows_current_global_drift(tmp_path):
    bridge = SystemBridge()
    bridge.bundle_store = BundleStore(tmp_path / "user_verified_plan")

    selected_product, ctx = _current_scope(bridge)

    bundle_id = "bundle_snapshot_tools"
    bdir = bridge.bundle_store.bundle_dir(bundle_id)
    (bdir / "plan").mkdir(parents=True, exist_ok=True)
    (bdir / "safety").mkdir(parents=True, exist_ok=True)
    (bdir / "catalog").mkdir(parents=True, exist_ok=True)

    snapshot_path = bdir / "catalog" / "tools.json"
    snapshot_path.write_text(json.dumps([{"function": "bundle_only"}], indent=2), encoding="utf-8")

    manifest = {
        "bundle_id": bundle_id,
        "status": "verified",
        "verified": True,
        "product_spec_file": ctx["product_spec_file"],
        "safety_file": ctx["safety_file"],
        "product_name": ctx["product_name"],
        "execution_mode": "simulation",
        "robot_env": "gazebo",
        "source_hashes": {
            **ctx["source_hashes"],
            "tools_sha256": sha256_file(snapshot_path),
        },
        "artifacts": {
            "tools_json": "catalog/tools.json",
            "requirements_json": "plan/requirements.json",
            "plan_json": "plan/plan.json",
            "global_fsa_json": "plan/global_fsa.json",
            "safety_logic_json": "safety/cca_safety_logic.json",
        },
    }
    bridge.bundle_store.save_manifest(bdir, manifest)
    bridge.bundle_store.upsert_bundle_summary(
        {
            "bundle_id": bundle_id,
            "display_name": "snapshot-tools",
            "created_at_utc": "2026-03-03T00:00:00+00:00",
            "status": "verified",
            "verified": True,
            "product_name": ctx["product_name"],
            "product_spec_file": ctx["product_spec_file"],
            "execution_mode": "simulation",
            "robot_env": "gazebo",
            "manifest_path": str(bridge.bundle_store.manifest_path(bundle_id)),
        }
    )

    ok, reasons = bridge.check_bundle_compatibility(
        bundle_id,
        selected_product,
        "simulation",
        "gazebo",
    )
    assert ok is True
    assert reasons == []


def test_bridge_verify_bundle_updates_status(tmp_path):
    bridge = SystemBridge()
    bridge.bundle_store = BundleStore(tmp_path / "user_verified_plan")

    _, ctx = _current_scope(bridge)

    bundle_id = "bundle_draft"
    bdir = bridge.bundle_store.bundle_dir(bundle_id)
    (bdir / "plan").mkdir(parents=True, exist_ok=True)
    (bdir / "safety").mkdir(parents=True, exist_ok=True)
    (bdir / "validation").mkdir(parents=True, exist_ok=True)
    _write_json(bdir / "plan" / "plan.json", {"nodes": []})
    _write_json(
        bdir / "plan" / "global_fsa.json",
        {"A": {"X": ["S0"], "E": [], "Tr": [], "x0": "S0", "Xm": ["S0"]}, "meta": {}},
    )
    _write_json(
        bdir / "safety" / "cca_safety_logic.json",
        {"rules": [{"id": "SAFE_1", "ltlf": "G(true)", "aps": []}]},
    )
    (bdir / "safety" / "SAFE_1_dfa.dot").write_text("digraph {}", encoding="utf-8")
    _write_json(
        bdir / "validation" / "offline_validation.json",
        {"ok": True, "violations": [], "violated_rules": [], "witness_count": 0},
    )

    manifest = {
        "bundle_id": bundle_id,
        "display_name": "draft",
        "created_at_utc": "2026-03-03T00:00:00+00:00",
        "status": "draft",
        "verified": False,
        "product_spec_file": ctx["product_spec_file"],
        "safety_file": ctx["safety_file"],
        "product_name": ctx["product_name"],
        "execution_mode": "simulation",
        "robot_env": "gazebo",
        "source_hashes": dict(ctx["source_hashes"]),
        "artifacts": {
            "requirements_json": "plan/plan.json",
            "plan_json": "plan/plan.json",
            "global_fsa_json": "plan/global_fsa.json",
            "safety_logic_json": "safety/cca_safety_logic.json",
            "safety_dfa_dot_files": ["safety/SAFE_1_dfa.dot"],
            "safety_dfa_png_files": [],
            "offline_validation_json": "validation/offline_validation.json",
        },
        "validation_summary": {"ok": True, "violated_rules": [], "witness_count": 0},
    }
    bridge.bundle_store.save_manifest(bdir, manifest)
    bridge.bundle_store.upsert_bundle_summary(
        {
            "bundle_id": bundle_id,
            "display_name": "draft",
            "created_at_utc": manifest["created_at_utc"],
            "status": "draft",
            "verified": False,
            "product_name": ctx["product_name"],
            "product_spec_file": ctx["product_spec_file"],
            "execution_mode": "simulation",
            "robot_env": "gazebo",
            "manifest_path": str(bridge.bundle_store.manifest_path(bundle_id)),
        }
    )

    out = bridge.verify_bundle(bundle_id)
    assert out["summary"]["status"] == "verified"
    assert out["summary"]["verified"] is True
    loaded = bridge.bundle_store.load_manifest(bundle_id)
    assert loaded is not None
    assert loaded["status"] == "verified"
    assert loaded["verified"] is True


def test_bundle_store_reconciles_index_with_bundle_dirs(tmp_path):
    store = BundleStore(tmp_path / "user_verified_plan")

    present_id = "bundle_present"
    present_dir = store.bundle_dir(present_id)
    present_dir.mkdir(parents=True, exist_ok=True)
    store.save_manifest(
        present_dir,
        {
            "bundle_id": present_id,
            "display_name": "present",
            "created_at_utc": "2026-03-06T20:00:00+00:00",
            "status": "draft",
            "verified": False,
            "product_name": "assembly_board-v1",
            "product_spec_file": "/tmp/assembly.txt",
            "safety_file": "/tmp/safety.txt",
            "execution_mode": "simulation",
            "robot_env": "gazebo",
        },
    )

    manifestless_id = "bundle_manifestless"
    store.bundle_dir(manifestless_id).mkdir(parents=True, exist_ok=True)

    store._save_index(
        {
            "schema_version": 1,
            "active_bundle_id": "bundle_missing",
            "bundles": [
                {
                    "bundle_id": "bundle_missing",
                    "display_name": "missing",
                    "created_at_utc": "2026-03-06T19:00:00+00:00",
                    "status": "verified",
                    "verified": True,
                }
            ],
        }
    )

    rows = store.list_bundles()
    row_ids = {row["bundle_id"] for row in rows}
    assert present_id in row_ids
    assert manifestless_id in row_ids
    assert "bundle_missing" not in row_ids

    manifestless = store.get_bundle_summary(manifestless_id)
    assert manifestless is not None
    assert manifestless["status"] == "invalid"
    assert manifestless["verified"] is False
    assert store.get_active_bundle_id() is None


def test_bridge_delete_bundle_allows_verified_unlinked_plan_set(tmp_path):
    bridge = SystemBridge()
    bridge.bundle_store = BundleStore(tmp_path / "user_verified_plan")

    _, ctx = _current_scope(bridge)

    bundle_id = "bundle_unlinked_verified"
    bdir = bridge.bundle_store.bundle_dir(bundle_id)
    bdir.mkdir(parents=True, exist_ok=True)

    missing_safety = tmp_path / "deleted_safety.txt"
    manifest = {
        "bundle_id": bundle_id,
        "display_name": "unlinked",
        "created_at_utc": "2026-03-06T20:10:00+00:00",
        "status": "verified",
        "verified": True,
        "product_spec_file": ctx["product_spec_file"],
        "safety_file": str(missing_safety.resolve()),
        "product_name": ctx["product_name"],
        "execution_mode": "simulation",
        "robot_env": "gazebo",
        "source_hashes": dict(ctx["source_hashes"]),
        "artifacts": {},
        "validation_summary": {"ok": True, "violated_rules": [], "witness_count": 0},
    }
    bridge.bundle_store.save_manifest(bdir, manifest)
    bridge.bundle_store.upsert_bundle_summary(
        {
            "bundle_id": bundle_id,
            "display_name": "unlinked",
            "created_at_utc": manifest["created_at_utc"],
            "status": "verified",
            "verified": True,
            "product_name": ctx["product_name"],
            "product_spec_file": ctx["product_spec_file"],
            "safety_file": manifest["safety_file"],
            "execution_mode": "simulation",
            "robot_env": "gazebo",
            "manifest_path": str(bridge.bundle_store.manifest_path(bundle_id)),
        }
    )

    policy = bridge.get_bundle_delete_policy(bundle_id)
    assert policy["can_delete"] is True
    assert "safety_requirement_file_missing" in policy["missing_links"]

    out = bridge.delete_bundle(bundle_id)
    assert out["removed"] is True
    assert not bdir.exists()


def test_bridge_generate_verified_bundle_forwards_refinement_feedback():
    bridge = SystemBridge()
    product_files = bridge.list_product_files()
    assert product_files
    selected_product = product_files[0]
    product_ctx = bridge._resolve_product_context(selected_product, include_hashes=False)
    requirement_file = str(product_ctx["product_spec_file"])
    safety_file = str(product_ctx["safety_file"])
    captured: dict[str, object] = {}

    async def _fake_compile_bundle(**kwargs):
        captured.update(kwargs)
        return {"summary": {"bundle_id": "bundle_new"}, "manifest": {}}

    bridge.bundle_compiler = SimpleNamespace(compile_bundle=_fake_compile_bundle)

    bridge.generate_verified_bundle(
        selected_product,
        "simulation",
        "gazebo",
        product_requirement_file=requirement_file,
        safety_requirement_file=safety_file,
        auto_replan_max_attempts=2,
        refinement_feedback="use xarm6 for LRP first",
        parent_bundle_id="bundle_parent",
    )

    assert captured["refinement_feedback"] == "use xarm6 for LRP first"
    assert captured["parent_bundle_id"] == "bundle_parent"
    assert captured["auto_replan_max_attempts"] == 2


def test_bridge_product_requirement_listing_only_uses_product_manifests(tmp_path):
    bridge = SystemBridge()

    product_dir = tmp_path / "initialization" / "products"
    product_dir.mkdir(parents=True, exist_ok=True)
    req_dir = tmp_path / "specification" / "products" / "requirements"
    req_dir.mkdir(parents=True, exist_ok=True)
    linked_req = req_dir / "demo_product.txt"
    orphan_req = req_dir / "orphan.txt"
    linked_req.write_text("[Product Requirements]\n- demo\n", encoding="utf-8")
    orphan_req.write_text("[Product Requirements]\n- orphan\n", encoding="utf-8")
    product_path = product_dir / "demo_product.json"
    _write_json(
        product_path,
        {
            "demo_product": {
                "type": "product",
                "jid": "demo_product@localhost",
                "password": "none",
                "domain": "localhost",
                "functions": [],
                "instructions": "demo",
            }
        },
    )

    original_product_dir = bridge_module._PRODUCT_DIR
    original_req_dir = bridge_module._PRODUCT_REQUIREMENTS_DIR
    try:
        bridge_module._PRODUCT_DIR = product_dir
        bridge_module._PRODUCT_REQUIREMENTS_DIR = req_dir

        ctx = bridge._resolve_product_context(str(product_path), include_hashes=False)
        listed = bridge.list_product_requirement_files(str(product_path))
        listed_all = bridge.list_product_requirement_files()
    finally:
        bridge_module._PRODUCT_DIR = original_product_dir
        bridge_module._PRODUCT_REQUIREMENTS_DIR = original_req_dir

    assert ctx["product_spec_file"] == str(linked_req.resolve())
    assert listed == [str(linked_req.resolve())]
    assert listed_all == [str(linked_req.resolve())]


def test_plan_prompt_builders_include_refinement_context():
    req_prompt = build_requirement_parse_prompt(
        "[Product Requirements]\n- assemble MCP\n",
        [],
        refinement_feedback="Use canonical part names.",
        previous_preview_requirements=[{"id": "REQ_1", "product": "MCP part"}],
    )
    assert "HUMAN-IN-THE-LOOP REFINEMENT CONTEXT:" in req_prompt
    assert "Use canonical part names." in req_prompt
    assert "PREVIOUS_PARSED_REQUIREMENTS" in req_prompt

    task_prompt = build_task_expansion_prompt(
        requirements=[{"id": "REQ_1", "product": "MCP", "context": {}, "raw_text": "assemble MCP"}],
        tools_catalog=[],
        resource_infos=[],
        caps_overview="",
        safety_text="",
        refinement_feedback="Add move_home after insertion.",
        previous_preview_requirements=[{"id": "REQ_1"}],
        previous_preview_tasks=[{"id": "REQ_1_T1", "function_name": "place_insert"}],
    )
    assert "HUMAN-IN-THE-LOOP REFINEMENT CONTEXT:" in task_prompt
    assert "Add move_home after insertion." in task_prompt
    assert "PREVIOUS_TASK_DAG" in task_prompt


def test_product_meta_normalization_drops_cad_path_and_defaults_replan_mode():
    normalized = _normalize_product_meta(
        {
            "jid": "demo@localhost",
            "cad_path": "/tmp/unused.step",
            "functions": "not-a-list",
            "replan_mode": "unexpected",
        },
        product_name="demo",
    )
    assert "cad_path" not in normalized
    assert normalized["jid"] == "demo@localhost"
    assert normalized["functions"] == []
    assert normalized["replan_mode"] == "des"


def test_default_product_meta_sets_default_instructions_and_requirement_path():
    meta = _default_product_meta("demo_product")
    assert meta["instructions"] == "This product requires to be printed and assembled as specified."
    assert meta["product_specification_file"].endswith(
        "cais_spade_llm/specification/products/requirements/demo_product.txt"
    )
    assert meta["replan_mode"] == "des"


def test_delete_product_manifest_removes_existing_json_file(tmp_path):
    product_path = tmp_path / "demo_product.json"
    product_path.write_text('{"demo_product": {}}', encoding="utf-8")

    deleted = _delete_product_manifest(product_path)

    assert deleted is True
    assert product_path.exists() is False


def test_unlink_geometry_from_products_clears_matching_geometry_paths(tmp_path):
    product_a = tmp_path / "demo_a.json"
    product_b = tmp_path / "demo_b.json"
    geometry_path = "cais_spade_llm/specification/products/geometry/demo_geom.json"

    _write_json(
        product_a,
        {
            "demo_a": {
                "jid": "demo_a@localhost",
                "product_geometry_file": geometry_path,
                "instructions": "demo",
            }
        },
    )
    _write_json(
        product_b,
        {
            "demo_b": {
                "jid": "demo_b@localhost",
                "product_geometry_file": "cais_spade_llm/specification/products/geometry/other.json",
                "instructions": "demo",
            }
        },
    )

    class _Bridge:
        def list_product_files(self):
            return [str(product_a), str(product_b)]

        def load_config(self, path):
            return json.loads(Path(path).read_text(encoding="utf-8"))

        def save_config(self, path, data):
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    cleared = _unlink_geometry_from_products(_Bridge(), geometry_path)

    assert cleared == ["demo_a"]
    saved_a = json.loads(product_a.read_text(encoding="utf-8"))
    saved_b = json.loads(product_b.read_text(encoding="utf-8"))
    assert saved_a["demo_a"]["product_geometry_file"] == ""
    assert saved_b["demo_b"]["product_geometry_file"].endswith("other.json")


def test_geometry_payload_validator_matches_assembly_board_shape():
    valid = {
        "gazebo": {
            "assembly_board": {
                "center": {"x": 0.0, "y": 0.0, "z": 1.02},
                "thickness_m": 0.01,
                "slot_floor_z_m": 1.025,
                "slots": {"MCP": [0.0, -0.08]},
            },
            "parts": {
                "model_map": {"MCP": "circ_pin_medium"},
                "heights_m": {"MCP": 0.08},
            },
        },
        "real": {
            "assembly_board": {
                "center": {"x": 0.0, "y": 0.0, "z": 1.02},
                "thickness_m": 0.01,
                "slot_floor_z_m": 1.025,
                "slots": {"MCP": [0.0, -0.08]},
            },
            "parts": {
                "model_map": {"MCP": "circ_pin_medium"},
                "heights_m": {"MCP": 0.08},
            },
        },
    }
    invalid = {"gazebo": {"assembly_board": {"slots": {"MCP": [0.0]}}, "parts": {}}}

    assert _validate_geometry_payload(valid) == ""
    assert "center" in _validate_geometry_payload(invalid)


def test_gazebo_reset_restores_scene_in_place():
    bridge = SystemBridge()
    bridge.system_running = False
    bridge._starting = False

    bridge._any_running = lambda names: True
    bridge._ros2_command_output = lambda *args, **kwargs: (
        True,
        "/reset_simulation\n/reset_world\n",
    )

    calls: list[str] = []
    bridge.ros2_exec = lambda command, timeout_sec=20.0: (calls.append(command) or True, "OK")
    bridge._restore_gazebo_scene_in_place = lambda: (["xarm6: moved home", "restored 9 part pose(s)"], [])

    ok, msg = bridge.ros2_reset_gazebo_environment()
    assert ok is True
    assert calls == ['ros2 service call /reset_world std_srvs/srv/Empty "{}"']
    assert "Gazebo environment reset via /reset_world." in msg
    assert "restored 9 part pose(s)" in msg

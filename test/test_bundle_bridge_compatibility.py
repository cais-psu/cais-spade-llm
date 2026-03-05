from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "cais_spade_llm"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from cais_spade_llm.bundles.bundle_store import BundleStore
from cais_spade_llm.bundles.models import sha256_file
from cais_spade_llm.ui.bridge import SystemBridge


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _current_scope(bridge: SystemBridge) -> tuple[str, dict]:
    requirement_file = bridge.list_product_requirement_files()[0]
    safety_file = bridge.list_safety_requirement_files()[0]
    product_init = bridge._resolve_product_init_for_requirement(requirement_file)
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

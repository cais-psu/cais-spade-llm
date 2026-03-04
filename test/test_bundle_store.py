from __future__ import annotations

import json

from cais_spade_llm.bundles.bundle_store import BundleStore


def test_bundle_store_index_and_active_selection(tmp_path):
    store = BundleStore(tmp_path / "user_verified")

    summary = {
        "bundle_id": "bundle_a",
        "display_name": "Bundle A",
        "created_at_utc": "2026-03-03T00:00:00+00:00",
        "status": "verified",
        "verified": True,
        "product_name": "assembly_board-v1",
        "product_spec_file": "cais_spade_llm/specification/products/requirements/assembly_board-v1.txt",
        "execution_mode": "simulation",
        "robot_env": "gazebo",
        "manifest_path": str(store.manifest_path("bundle_a")),
    }
    store.upsert_bundle_summary(summary)

    listed = store.list_bundles()
    assert len(listed) == 1
    assert listed[0]["bundle_id"] == "bundle_a"

    store.set_active_bundle_id("bundle_a")
    assert store.get_active_bundle_id() == "bundle_a"

    store.set_active_bundle_id(None)
    assert store.get_active_bundle_id() is None

    with store.index_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["active_bundle_id"] is None
    assert len(payload["bundles"]) == 1


def test_bundle_store_manifest_roundtrip(tmp_path):
    store = BundleStore(tmp_path / "user_verified")
    bid = "bundle_b"
    bdir = store.bundle_dir(bid)
    manifest = {"bundle_id": bid, "status": "verified", "artifacts": {"plan_json": "plan/p.json"}}
    store.save_manifest(bdir, manifest)
    loaded = store.load_manifest(bid)
    assert loaded == manifest

    updated_manifest = dict(manifest)
    updated_manifest["status"] = "draft"
    store.overwrite_manifest(bid, updated_manifest)
    loaded2 = store.load_manifest(bid)
    assert loaded2["status"] == "draft"


def test_bundle_store_update_bundle_summary(tmp_path):
    store = BundleStore(tmp_path / "user_verified")
    summary = {
        "bundle_id": "bundle_c",
        "display_name": "Bundle C",
        "created_at_utc": "2026-03-03T00:00:00+00:00",
        "status": "draft",
        "verified": False,
        "product_name": "assembly_board-v1",
        "product_spec_file": "cais_spade_llm/specification/products/requirements/assembly_board-v1.txt",
        "execution_mode": "simulation",
        "robot_env": "gazebo",
        "manifest_path": str(store.manifest_path("bundle_c")),
    }
    store.upsert_bundle_summary(summary)
    updated = store.update_bundle_summary("bundle_c", {"status": "verified", "verified": True})
    assert updated["status"] == "verified"
    assert updated["verified"] is True

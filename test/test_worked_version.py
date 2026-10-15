from __future__ import annotations

"""Integrity and read-only review of the preserved worked Part slippage archive."""

import hashlib
import json
import zipfile

import pytest

from cais_spade_llm.recovery_framework.worked_version import read_worked_version


def archive(tmp_path):
    events = [{"outline_id": f"r_{index}"} for index in range(1, 6)]
    programs = [{"outline_id": row["outline_id"], "primitive_steps": [{}] * count}
                for row, count in zip(events, (1, 5, 5, 5, 5))]
    outline = {"outline": events, "outline_status": "completed", "outline_safety_status": "passed",
               "execution_authorized": False}
    composition = {"accepted_outline_prefix": events, "accepted_primitive_program": programs,
                   "composition_feasibility": "FEASIBLE", "cca_composition_status": "not_checked",
                   "execution_authorized": False}
    manifest = {"version": 1, "scenario": "Part slippage", "snapshot_at": "snapshot-time",
        "outline_run_id": "offline_470087fe7a92421abd624ea721aa2ca1",
        "composition_run_id": "offline_1bb1efac84d7443aa37238521ead3b68",
        "outline_artifact": "outline.json", "composition_artifact": "composition.json",
        "branch": "recovery-framework-journal", "commit": "0123456789abcdef", "files": {},
        "code_snapshot_note": "Current code; original generation times preserved",
        "execution_authorized": False, "cca_composition_status": "not_checked"}
    path = tmp_path / "worked.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, value in (("outline.json", outline), ("composition.json", composition)):
            data = json.dumps(value).encode()
            manifest["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            bundle.writestr(name, data)
        bundle.writestr("manifest.json", json.dumps(manifest))
    index = {key: value for key, value in manifest.items() if key != "files"}
    index.update(archive_name=path.name, archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 files_verified=2, archive_bytes=path.stat().st_size)
    (tmp_path / "manifest.json").write_text(json.dumps(index))
    return path


def test_worked_review_preserves_ids_programs_and_pending_execution(tmp_path):
    path = archive(tmp_path)
    before = path.read_bytes()
    result = read_worked_version(root=tmp_path)
    assert len(result["outline"]) == 5
    assert sum(len(row["primitive_steps"]) for row in result["primitives"]) == 21
    assert result["outline_run_id"] == "offline_470087fe7a92421abd624ea721aa2ca1"
    assert result["composition_run_id"] == "offline_1bb1efac84d7443aa37238521ead3b68"
    assert result["cca_composition_status"] == "not_checked" and result["execution_authorized"] is False
    assert path.read_bytes() == before
    result["outline"].clear()
    assert len(read_worked_version(root=tmp_path)["outline"]) == 5


def test_changed_archive_is_rejected_after_cached_review(tmp_path):
    path = archive(tmp_path)
    read_worked_version(root=tmp_path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        read_worked_version(root=tmp_path)


def test_worked_index_cannot_claim_execution_authorization(tmp_path):
    archive(tmp_path)
    index = tmp_path / "manifest.json"
    values = json.loads(index.read_text())
    values["execution_authorized"] = True
    index.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="provenance"):
        read_worked_version(root=tmp_path)


def test_no_archive_returns_empty_without_creating_files(tmp_path):
    assert read_worked_version(root=tmp_path) == {}
    assert list(tmp_path.iterdir()) == []

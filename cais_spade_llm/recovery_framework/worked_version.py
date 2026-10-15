from __future__ import annotations

"""Read the preserved worked Part slippage version without creating an attempt."""

import hashlib
import json
import zipfile
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from cais_spade_llm.recovery_framework import ROOT

WORKED = ROOT / "cais_spade_llm/monitor/environment_runs/recovery_worked/Part slippage"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _read(directory: str, index_stamp: int, archive_stamp: int, archive_size: int) -> dict:
    root = Path(directory)
    index = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    archive = root / "worked.zip"
    if (index.get("version") != 1 or index.get("scenario") != "Part slippage"
            or index.get("archive_name") != archive.name or archive.is_symlink()
            or index.get("execution_authorized") is not False
            or index.get("cca_composition_status") != "not_checked"
            or _sha256(archive) != index.get("archive_sha256")):
        raise ValueError("The worked Part slippage archive changed or has invalid provenance")
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        for key in ("snapshot_at", "outline_run_id", "composition_run_id", "branch", "commit"):
            if manifest.get(key) != index.get(key):
                raise ValueError("The worked archive and its index disagree: " + key)

        def read_record(name: str) -> dict:
            entry = manifest["files"][name]
            data = bundle.read(name)
            if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("The worked record differs from its preserved hash: " + name)
            return json.loads(data)

        outline = read_record(manifest["outline_artifact"])
        composition = read_record(manifest["composition_artifact"])
    events, programs = outline.get("outline"), composition.get("accepted_primitive_program")
    if (outline.get("outline_status") != "completed" or outline.get("outline_safety_status") != "passed"
            or not isinstance(events, list) or len(events) != 5
            or not isinstance(programs, list) or len(programs) != 5
            or composition.get("accepted_outline_prefix") != events
            or sum(len(row.get("primitive_steps", [])) for row in programs) != 21
            or composition.get("composition_feasibility") != "FEASIBLE"
            or composition.get("cca_composition_status") != "not_checked"
            or composition.get("execution_authorized") is not False
            or outline.get("execution_authorized") is not False):
        raise ValueError("The worked version does not preserve its accepted outline and composition")
    return {"scenario": "Part slippage", "snapshot_at": index["snapshot_at"],
        "outline_run_id": index["outline_run_id"], "composition_run_id": index["composition_run_id"],
        "outline": events, "primitives": programs, "outline_safety_status": outline["outline_safety_status"],
        "composition_feasibility": composition["composition_feasibility"],
        "cca_composition_status": composition["cca_composition_status"], "execution_authorized": False,
        "archive_path": str(archive), "archive_bytes": archive_size,
        "files_verified": index["files_verified"], "branch": index["branch"], "commit": index["commit"],
        "code_snapshot_note": index["code_snapshot_note"]}


def read_worked_version(*, root: Path = WORKED) -> dict:
    """Verify and read the separate archive for a read-only Recovery card.

    Args:
        root: Directory containing the preserved archive and its manifest.

    Returns:
        The recorded results and provenance, or an empty dictionary before archiving.
    """
    index, archive = root / "manifest.json", root / "worked.zip"
    if not index.is_file() or not archive.is_file():
        return {}
    if root.is_symlink() or index.is_symlink():
        raise ValueError("The worked archive must use its retained local directory")
    return deepcopy(_read(str(root.resolve()), index.stat().st_mtime_ns,
                          archive.stat().st_mtime_ns, archive.stat().st_size))

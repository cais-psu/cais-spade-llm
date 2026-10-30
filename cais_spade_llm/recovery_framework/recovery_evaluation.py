from __future__ import annotations

"""Frozen implementation manifests for saved-observation captures."""

import hashlib
import json
import logging
from pathlib import Path

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.capture_files import BELIEFS_NAME

logger = logging.getLogger(__name__)


def file_hash(path: Path) -> str:
    """Hash original bytes using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_hash(value: object) -> str:
    """Fingerprint exact JSON values without changing identifiers."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _implementation_files(root: Path) -> set[Path]:
    files = {root / "pyproject.toml", root / "poetry.lock"}
    package = root / "cais_spade_llm"
    files.update(path for path in package.rglob("*.py") if "monitor" not in path.relative_to(package).parts)
    for directory in (package / "initialization", package / "specification", root / "ros2"):
        files.update(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower()
                     in {".py", ".json", ".txt", ".yaml", ".yml", ".cpp", ".hpp", ".h", ".action",
                         ".urdf", ".srdf", ".xacro", ".world", ".sdf", ".stl", ".dae", ".obj", ".xml", ".pgm"})
    return files


def implementation_inventory(root: Path = ROOT) -> list[str]:
    """List the implementation files a frozen manifest checks for additions and removals."""
    return sorted(str(path.relative_to(root)) for path in _implementation_files(root))


def frozen_files(captures: list[dict], *, root: Path = ROOT) -> dict[str, str]:
    """Freeze implementation, declarations, specifications, and saved model files."""
    files = _implementation_files(root)
    for capture in captures:
        source = Path(capture["source"]).resolve()
        files.add(source)
        if source.with_name(BELIEFS_NAME).is_file():
            files.add(source.with_name(BELIEFS_NAME))
        files.update(path for path in (source.parent / "native_preview_model").rglob("*") if path.is_file())
        files.update(Path(path) for path in capture.get("evidence_files", []))
    return {str(path.relative_to(root) if path.is_relative_to(root) else path): file_hash(path)
            for path in sorted(files)}


def verify_frozen(manifest: dict, *, root: Path = ROOT) -> None:
    """Refuse subsequent requests when any frozen input changed."""
    if "implementation_files" in manifest:
        current = sorted(str(path.relative_to(root)) for path in _implementation_files(root))
        if current != manifest["implementation_files"]:
            raise ValueError("Frozen implementation file inventory changed")
    for name, digest in manifest["files"].items():
        path = root / name
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError("Frozen evaluation input changed: " + name)


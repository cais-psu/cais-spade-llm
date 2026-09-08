from __future__ import annotations

"""Append and verify the evidence owned by one non-executing refinement run."""

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def fingerprint(value: Any) -> str:
    """Hash finite canonical JSON, excluding no fields implicitly."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def owned_path(root: Path, ref: str) -> Path:
    """Resolve an exact relative reference without symlink or traversal access."""
    relative = Path(ref)
    if not ref or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Refinement reference must remain inside its interaction.")
    path = root / relative
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Refinement references cannot use symlinks.")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Refinement reference escapes its interaction.")
    return path


def pin(root: Path, path: Path) -> dict[str, str]:
    """Pin one existing interaction artifact by exact bytes."""
    ref = path.relative_to(root).as_posix()
    return {"ref": ref, "sha256": hashlib.sha256(owned_path(root, ref).read_bytes()).hexdigest()}


def read_pin(root: Path, reference: Mapping[str, str]) -> Any:
    """Read hash-verified JSON without admitting it as model evidence."""
    if set(reference) != {"ref", "sha256"}:
        raise ValueError("A refinement pin requires ref and sha256.")
    data = owned_path(root, reference["ref"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError("Pinned refinement evidence changed.")
    return json.loads(data)


def append_record(
    root: Path, directory: Path, name: str, payload: Mapping[str, Any]
) -> dict[str, str]:
    """Publish one immutable record atomically, never replacing an earlier record."""
    directory.mkdir(parents=True, exist_ok=True)
    path = owned_path(root, (directory / name).relative_to(root).as_posix())
    record = dict(payload)
    record.pop("fingerprint", None)
    record["fingerprint"] = fingerprint(record)
    temporary = path.with_suffix(path.suffix + ".pending")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link supplies atomic exclusive publication, unlike replace().
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return pin(root, path)


def verify_record(root: Path, reference: Mapping[str, str]) -> dict[str, Any]:
    """Verify both an immutable record's bytes and its content fingerprint."""
    record = read_pin(root, reference)
    if not isinstance(record, dict):
        raise ValueError("Refinement record must be an object.")
    payload = dict(record)
    if payload.pop("fingerprint", None) != fingerprint(payload):
        raise ValueError("Refinement record fingerprint changed.")
    return record


def verify_evidence_tree(root: Path, reference: Mapping[str, str]) -> Any:
    """Check selected evidence and every embedded local artifact pin, including meshes."""
    seen: dict[str, str] = {}

    def visit(source: Mapping[str, str]) -> Any:
        ref, sha = source["ref"], source["sha256"]
        if {"evaluations", "execution"}.intersection(Path(ref).parts):
            raise ValueError("Evaluation and execution records cannot supply composition evidence.")
        if ref in seen:
            if seen[ref] != sha:
                raise ValueError("Evidence contains conflicting source hashes.")
            return None
        if len(seen) >= 512:
            raise ValueError("Evidence dependency chain exceeds its record budget.")
        data = owned_path(root, ref).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError("Pinned evidence dependency changed: " + ref)
        seen[ref] = sha
        if Path(ref).suffix.lower() != ".json":
            return None
        value = json.loads(data)
        if isinstance(value, dict) and (
            str(value.get("record_type", "")).startswith("PrimitiveExecution")
            or value.get("record_type") == "GazeboInstanceBinding"
        ):
            raise ValueError("Execution records cannot supply composition evidence.")
        walk(value)
        return value

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("ref"), str) and isinstance(value.get("sha256"), str):
                visit(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    return visit(reference)

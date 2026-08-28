"""Resolve approved Spec2Primitives document and CAD refs from local sources."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import pdfplumber
from pdfminer.pdfparser import PDFSyntaxError

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SPEC2PRIMITIVES_ROOT = Path(__file__).resolve().parents[1]
_REFERENCES_ROOT = _SPEC2PRIMITIVES_ROOT / "references" / "products"
_CAD_ROOT = _REPOSITORY_ROOT / "ros2" / "cais_lab_robotics" / "cad_models"
_INVENTORY_PATH = _REFERENCES_ROOT / "approved_sources.json"

_DOCUMENT_KEYS = {
    "context_ref",
    "evidence_type",
    "repository_path",
    "source_url",
    "page_count",
}
_CAD_KEYS = {
    "context_ref",
    "evidence_type",
    "repository_path",
    "source_url",
    "units",
}


def approved_context_refs() -> tuple[str, ...]:
    """Return every exact ref in the fixed approved-source inventory."""
    return tuple(_load_sources())


def approved_context_ref_evidence_types() -> dict[str, str]:
    """Return exact approved refs mapped to their fixed evidence types."""
    return {
        context_ref: str(source["evidence_type"]) for context_ref, source in _load_sources().items()
    }


def approved_cad_refs() -> tuple[str, ...]:
    """Return every exact approved CAD ref in inventory order."""
    return tuple(
        context_ref
        for context_ref, source in _load_sources().items()
        if source["evidence_type"] == "CAD"
    )


def approved_document_refs() -> tuple[str, ...]:
    """Return every exact approved document ref in inventory order."""
    return tuple(
        context_ref
        for context_ref, source in _load_sources().items()
        if source["evidence_type"] == "document"
    )


def approved_document_path(context_ref: str) -> Path:
    """Return the local path for one exact approved document ref.

    This boundary never resolves caller-supplied paths. The value must be an
    exact document key from the fixed approved-source inventory.
    """
    if (
        not isinstance(context_ref, str)
        or not context_ref
        or context_ref != context_ref.strip()
        or _is_forbidden_ref(context_ref)
    ):
        raise ValueError("context_ref must be one exact approved document ref.")
    source = _load_sources().get(context_ref)
    if source is None or source.get("evidence_type") != "document":
        raise ValueError("context_ref is not an approved document ref.")
    source_path = (_REPOSITORY_ROOT / str(source["repository_path"])).resolve()
    try:
        source_path.relative_to(_REFERENCES_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Approved document path is outside its permitted directory.") from exc
    if source_path.name != context_ref or source_path.suffix.lower() != ".pdf":
        raise ValueError("Approved document source does not match its exact ref.")
    if not source_path.is_file():
        raise OSError("Approved document source is missing.")
    return source_path


def approved_document_metadata(context_ref: str) -> dict[str, object]:
    """Validate one approved PDF without extracting its page text."""
    source_path = approved_document_path(context_ref)
    source = _load_sources()[context_ref]
    try:
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        with pdfplumber.open(source_path) as document:
            page_count = len(document.pages)
    except (OSError, PDFSyntaxError, TypeError, ValueError) as exc:
        raise OSError("Approved document source could not be validated.") from exc
    if page_count != source["page_count"]:
        raise ValueError(
            "Approved document page count does not match its inventory."
        )
    return {
        "context_ref": context_ref,
        "source_sha256": source_sha256,
        "page_count": page_count,
    }


def approved_cad_path(context_ref: str) -> Path:
    """Return the local path for one exact approved CAD ref.

    The value must be an exact CAD key from the fixed approved-source inventory;
    caller-supplied filesystem paths are never resolved.
    """
    if (
        not isinstance(context_ref, str)
        or not context_ref
        or context_ref != context_ref.strip()
        or _is_forbidden_ref(context_ref)
    ):
        raise ValueError("context_ref must be one exact approved CAD ref.")
    source = _load_sources().get(context_ref)
    if source is None or source.get("evidence_type") != "CAD":
        raise ValueError("context_ref is not an approved CAD ref.")
    source_path = (_REPOSITORY_ROOT / str(source["repository_path"])).resolve()
    try:
        source_path.relative_to(_CAD_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Approved CAD path is outside its permitted directory.") from exc
    if source_path.name != context_ref or source_path.suffix.lower() != ".stl":
        raise ValueError("Approved CAD source does not match its exact ref.")
    if not source_path.is_file():
        raise OSError("Approved CAD source is missing.")
    return source_path


def resolve_context_ref(request: dict[str, object]) -> dict[str, object]:
    """Serve bounded evidence for one exact approved local source.

    Args:
        request: A mapping containing only a string `context_ref` value.

    Returns:
        A mapping containing either `served_context` or a structured
        `rejection`. The resolver does not write source content or results.
    """
    if not isinstance(request, dict) or set(request) != {"context_ref"}:
        return _rejection(
            None,
            "invalid_request",
            "The request must contain only one context_ref string.",
        )

    context_ref = request["context_ref"]
    if not isinstance(context_ref, str) or not context_ref:
        return _rejection(
            None,
            "invalid_request",
            "The request must contain only one context_ref string.",
        )
    if context_ref != context_ref.strip():
        return _rejection(
            context_ref,
            "invalid_request",
            "The context_ref must match an approved ref exactly.",
        )
    if _is_forbidden_ref(context_ref):
        return _rejection(
            context_ref,
            "forbidden_ref",
            "Direct paths and path traversal are not permitted.",
        )

    try:
        sources = _load_sources()
    except (OSError, ValueError):
        return _rejection(
            context_ref,
            "malformed_source",
            "The approved-source inventory could not be read.",
        )

    source = sources.get(context_ref)
    if source is None:
        return _rejection(
            context_ref,
            "unknown_ref",
            "The context_ref is not in the approved-source inventory.",
        )

    evidence_type = source["evidence_type"]
    allowed_root = _REFERENCES_ROOT if evidence_type == "document" else _CAD_ROOT
    try:
        source_path = (_REPOSITORY_ROOT / str(source["repository_path"])).resolve()
        source_path.relative_to(allowed_root.resolve())
    except (OSError, ValueError):
        return _rejection(
            context_ref,
            "forbidden_source",
            "The approved source path is outside its permitted directory.",
        )

    if source_path.name != context_ref:
        return _rejection(
            context_ref,
            "forbidden_source",
            "The context_ref does not match the approved source filename.",
        )
    if not source_path.is_file():
        return _rejection(
            context_ref,
            "missing_source",
            "The approved source file is missing.",
        )

    if evidence_type == "document":
        if source_path.suffix.lower() != ".pdf":
            return _rejection(
                context_ref,
                "unsupported_document",
                "The approved document is not a PDF.",
            )
        return _serve_document(context_ref, source, source_path)

    if source_path.suffix.lower() != ".stl":
        return _rejection(
            context_ref,
            "unsupported_CAD",
            "The approved CAD source is not an STL file.",
        )
    return _serve_cad(context_ref, source, source_path)


def _load_sources() -> dict[str, dict[str, Any]]:
    with _INVENTORY_PATH.open(encoding="utf-8") as stream:
        inventory = json.load(stream)

    if not isinstance(inventory, dict) or set(inventory) != {"sources"}:
        raise ValueError("Approved-source inventory must contain only sources.")
    entries = inventory["sources"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Approved-source inventory sources must be a list.")

    sources: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each approved source must be a mapping.")
        evidence_type = entry.get("evidence_type")
        expected_keys = _DOCUMENT_KEYS if evidence_type == "document" else _CAD_KEYS
        if evidence_type not in {"document", "CAD"} or set(entry) != expected_keys:
            raise ValueError("Approved source fields do not match its evidence type.")

        context_ref = entry["context_ref"]
        repository_path = entry["repository_path"]
        source_url = entry["source_url"]
        if not all(
            isinstance(value, str) and value for value in (context_ref, repository_path, source_url)
        ):
            raise ValueError("Approved source strings must be non-empty.")
        if context_ref in sources:
            raise ValueError("Approved context_ref values must be unique.")

        if evidence_type == "document":
            page_count = entry["page_count"]
            if not isinstance(page_count, int) or isinstance(page_count, bool) or page_count <= 0:
                raise ValueError("Approved document page_count must be positive.")
        elif entry["units"] != "mm":
            raise ValueError("Approved CAD units must be mm.")

        sources[context_ref] = entry
    return sources


def _is_forbidden_ref(context_ref: str) -> bool:
    return (
        Path(context_ref).is_absolute()
        or ".." in context_ref
        or "/" in context_ref
        or "\\" in context_ref
        or ":" in context_ref
        or "\x00" in context_ref
    )


def _serve_document(
    context_ref: str,
    source: dict[str, Any],
    source_path: Path,
) -> dict[str, object]:
    try:
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        with pdfplumber.open(source_path) as document:
            if len(document.pages) != source["page_count"]:
                return _rejection(
                    context_ref,
                    "malformed_source",
                    "The document page count does not match the approved inventory.",
                )
            pages = [
                {"page": page_number, "text": page.extract_text() or ""}
                for page_number, page in enumerate(document.pages, start=1)
            ]
    except (OSError, PDFSyntaxError, TypeError, ValueError):
        return _rejection(
            context_ref,
            "malformed_source",
            "The approved PDF could not be read.",
        )

    return {
        "served_context": {
            "context_ref": context_ref,
            "evidence_type": "document",
            "provenance": _provenance(source),
            "document_evidence": {
                "source_sha256": source_sha256,
                "page_count": len(pages),
                "pages": pages,
            },
        }
    }


def _serve_cad(
    context_ref: str,
    source: dict[str, Any],
    source_path: Path,
) -> dict[str, object]:
    try:
        evidence = _read_binary_stl(source_path, str(source["units"]))
    except (OSError, struct.error, ValueError):
        return _rejection(
            context_ref,
            "malformed_source",
            "The approved STL could not be read.",
        )

    return {
        "served_context": {
            "context_ref": context_ref,
            "evidence_type": "CAD",
            "provenance": _provenance(source),
            "CAD_evidence": {
                "filename": context_ref,
                **evidence,
            },
        }
    }


def _read_binary_stl(source_path: Path, units: str) -> dict[str, object]:
    data = source_path.read_bytes()
    if len(data) < 84:
        raise ValueError("Binary STL header is incomplete.")

    triangle_count = struct.unpack_from("<I", data, 80)[0]
    if triangle_count == 0 or len(data) != 84 + triangle_count * 50:
        raise ValueError("Binary STL triangle layout is invalid.")

    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    for triangle_index in range(triangle_count):
        vertex_values = struct.unpack_from(
            "<9f",
            data,
            84 + triangle_index * 50 + 12,
        )
        for vertex_index in range(0, 9, 3):
            for axis in range(3):
                value = vertex_values[vertex_index + axis]
                if not math.isfinite(value):
                    raise ValueError("Binary STL contains a non-finite vertex.")
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)

    return {
        "units": units,
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "triangle_count": triangle_count,
        "bounds_mm": {
            "minimum": [_clean_number(value) for value in minimum],
            "maximum": [_clean_number(value) for value in maximum],
            "size": [_clean_number(maximum[axis] - minimum[axis]) for axis in range(3)],
        },
    }


def _clean_number(value: float) -> float:
    rounded = round(value, 6)
    return 0.0 if rounded == 0 else rounded


def _provenance(source: dict[str, Any]) -> dict[str, str]:
    return {
        "repository_path": str(source["repository_path"]),
        "source_url": str(source["source_url"]),
    }


def _rejection(
    context_ref: str | None,
    reason: str,
    message: str,
) -> dict[str, object]:
    return {
        "rejection": {
            "context_ref": context_ref,
            "reason": reason,
            "message": message,
        }
    }

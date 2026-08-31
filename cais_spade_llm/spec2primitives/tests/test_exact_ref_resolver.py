"""Tests for the Spec2Primitives approved-source exact-ref resolver."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cais_spade_llm.spec2primitives.tools import exact_ref_resolver

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CAD_ROOT = REPOSITORY_ROOT / "ros2" / "cais_lab_robotics" / "cad_models"

EXPECTED_CAD_REFS = (
    "BNC_Female.STL",
    "BNC_Male.STL",
    "DB_Housing.STL",
    "DSUB_Female.STL",
    "DSUB_Male.STL",
    "GMC_Laser_Plate_Virtual.STL",
    "Gear_Large.STL",
    "Gear_Medium.STL",
    "Gear_Plate.STL",
    "Gear_Shaft.STL",
    "Gear_Small.STL",
    "KET12_Square_12mm.STL",
    "KET16_Square_16mm.STL",
    "KET4_Square_4mm.STL",
    "KET8_Square_8mm.STL",
    "M12_Hex_Nut.STL",
    "M12_Screw.STL",
    "M16_Hex_Nut.STL",
    "M16_Screw.STL",
    "M4_Hex_Nut.STL",
    "M4_Screw.STL",
    "M8_Hex_Nut.STL",
    "M8_Screw.STL",
    "RGOCG12-50_12mm.STL",
    "RGOCG16-50_16mm.STL",
    "RGOCG4-50_Round_4mm.STL",
    "RGOCG8-50_8mm.STL",
    "RJ45_Female.STL",
    "RJ45_Housing.STL",
    "RJ45_Male.STL",
    "USB_Female.STL",
    "USB_Male.STL",
    "Waterproof_Female.STL",
    "Waterproof_Male.STL",
)
EXPECTED_REFS = ("NIST_assembly_instructions.pdf", *EXPECTED_CAD_REFS)


def test_inventory_publishes_the_complete_current_catalog() -> None:
    inventory = json.loads(
        exact_ref_resolver._INVENTORY_PATH.read_text(encoding="utf-8")
    )

    approved_refs = exact_ref_resolver.approved_context_refs()
    assert approved_refs
    assert approved_refs == EXPECTED_REFS
    assert len(approved_refs) == 35
    assert tuple(sorted(path.name for path in CAD_ROOT.glob("*.STL"))) == (
        EXPECTED_CAD_REFS
    )
    for source in inventory["sources"]:
        source_path = REPOSITORY_ROOT / source["repository_path"]
        assert source["source_sha256"] == hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()


def test_complete_document_ref_returns_all_ordered_pages() -> None:
    result = exact_ref_resolver.resolve_context_ref(
        {"context_ref": "NIST_assembly_instructions.pdf"}
    )

    assert "rejection" not in result
    served_context = result["served_context"]
    assert served_context["context_ref"] == "NIST_assembly_instructions.pdf"
    assert served_context["evidence_type"] == "document"
    assert set(served_context) == {
        "context_ref",
        "evidence_type",
        "provenance",
        "document_evidence",
    }
    evidence = served_context["document_evidence"]
    assert evidence["page_count"] == 6
    assert [page["page"] for page in evidence["pages"]] == list(range(1, 7))
    assert sum(len(page["text"]) for page in evidence["pages"]) == 8487
    assert served_context["provenance"] == {
        "repository_path": (
            "cais_spade_llm/spec2primitives/references/products/"
            "NIST_assembly_instructions.pdf"
        ),
        "source_url": (
            "https://www.nist.gov/el/intelligent-systems-division-73500/"
            "robotic-grasping-and-manipulation-assembly/assembly"
        ),
    }


def test_document_metadata_validates_without_serving_page_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        exact_ref_resolver,
        "_serve_document",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not serve text")),
    )

    metadata = exact_ref_resolver.approved_document_metadata(
        "NIST_assembly_instructions.pdf"
    )

    assert metadata["context_ref"] == "NIST_assembly_instructions.pdf"
    assert metadata["page_count"] == 6
    assert isinstance(metadata["source_sha256"], str)
    assert len(metadata["source_sha256"]) == 64
    inventory = json.loads(
        exact_ref_resolver._INVENTORY_PATH.read_text(encoding="utf-8")
    )
    assert metadata["source_sha256"] == inventory["sources"][0]["source_sha256"]
    assert exact_ref_resolver.approved_document_refs() == (
        "NIST_assembly_instructions.pdf",
    )


def test_all_approved_cad_refs_return_bounded_geometry() -> None:
    inventory = json.loads(
        exact_ref_resolver._INVENTORY_PATH.read_text(encoding="utf-8")
    )
    for context_ref in EXPECTED_CAD_REFS:
        result = exact_ref_resolver.resolve_context_ref(
            {"context_ref": context_ref}
        )

        assert "rejection" not in result, context_ref
        served_context = result["served_context"]
        assert served_context["context_ref"] == context_ref
        assert served_context["evidence_type"] == "CAD"
        assert set(served_context) == {
            "context_ref",
            "evidence_type",
            "provenance",
            "CAD_evidence",
        }
        evidence = served_context["CAD_evidence"]
        assert evidence["filename"] == context_ref
        assert evidence["units"] == "mm"
        source = next(
            item for item in inventory["sources"] if item["context_ref"] == context_ref
        )
        assert evidence["source_sha256"] == source["source_sha256"]
        assert evidence["triangle_count"] > 0
        assert set(evidence["bounds_mm"]) == {"minimum", "maximum", "size"}
        assert all(size > 0 for size in evidence["bounds_mm"]["size"])
        assert "sha256" not in served_context["provenance"]


@pytest.mark.parametrize(
    ("context_request", "reason"),
    [
        ({}, "invalid_request"),
        ({"context_ref": ""}, "invalid_request"),
        ({"context_ref": " Gear_Medium.STL"}, "invalid_request"),
        ({"context_ref": "../Gear_Medium.STL"}, "forbidden_ref"),
        ({"context_ref": "/tmp/Gear_Medium.STL"}, "forbidden_ref"),
        ({"context_ref": "Not_Approved.STL"}, "unknown_ref"),
    ],
)
def test_invalid_forbidden_and_unknown_refs_are_rejected(
    context_request: dict[str, object],
    reason: str,
) -> None:
    result = exact_ref_resolver.resolve_context_ref(context_request)

    assert "served_context" not in result
    assert result["rejection"]["reason"] == reason


def test_missing_approved_source_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _use_temporary_inventory(
        monkeypatch,
        tmp_path,
        _cad_source("Missing.STL", "cad/Missing.STL"),
    )

    result = exact_ref_resolver.resolve_context_ref(
        {"context_ref": "Missing.STL"}
    )

    assert "served_context" not in result
    assert result["rejection"]["reason"] == "missing_source"


def test_malformed_approved_stl_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    malformed = b"not a binary STL"
    cad_root = _use_temporary_inventory(
        monkeypatch,
        tmp_path,
        _cad_source(
            "Malformed.STL",
            "cad/Malformed.STL",
            source_sha256=hashlib.sha256(malformed).hexdigest(),
        ),
    )
    (cad_root / "Malformed.STL").write_bytes(malformed)

    result = exact_ref_resolver.resolve_context_ref(
        {"context_ref": "Malformed.STL"}
    )

    assert "served_context" not in result
    assert result["rejection"]["reason"] == "malformed_source"


def test_authority_hash_mismatch_is_rejected_before_serving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cad_root = _use_temporary_inventory(
        monkeypatch,
        tmp_path,
        _cad_source("Changed.STL", "cad/Changed.STL"),
    )
    (cad_root / "Changed.STL").write_bytes(b"changed after approval")

    result = exact_ref_resolver.resolve_context_ref(
        {"context_ref": "Changed.STL"}
    )

    assert "served_context" not in result
    assert result["rejection"]["reason"] == "source_hash_mismatch"


def test_inventory_cannot_point_outside_the_permitted_cad_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _use_temporary_inventory(
        monkeypatch,
        tmp_path,
        _cad_source("Outside.STL", "../Outside.STL"),
    )

    result = exact_ref_resolver.resolve_context_ref(
        {"context_ref": "Outside.STL"}
    )

    assert "served_context" not in result
    assert result["rejection"]["reason"] == "forbidden_source"


def _cad_source(
    context_ref: str,
    repository_path: str,
    *,
    source_sha256: str = "0" * 64,
) -> dict[str, object]:
    return {
        "context_ref": context_ref,
        "evidence_type": "CAD",
        "repository_path": repository_path,
        "source_url": "https://example.invalid/approved-source",
        "units": "mm",
        "source_sha256": source_sha256,
    }


def _use_temporary_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: dict[str, object],
) -> Path:
    repository_root = tmp_path / "repository"
    references_root = repository_root / "references"
    cad_root = repository_root / "cad"
    references_root.mkdir(parents=True)
    cad_root.mkdir()
    inventory_path = tmp_path / "approved_sources.json"
    inventory_path.write_text(
        json.dumps({"sources": [source]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(exact_ref_resolver, "_REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(exact_ref_resolver, "_REFERENCES_ROOT", references_root)
    monkeypatch.setattr(exact_ref_resolver, "_CAD_ROOT", cad_root)
    monkeypatch.setattr(exact_ref_resolver, "_INVENTORY_PATH", inventory_path)
    return cad_root

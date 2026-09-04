"""Measure geometry for PA-selected candidates without interpreting a relation."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_GROUNDING_ROOT = Path("products/grounding/rgb_d_cad_grounding")
_PRODUCER = "rgb_d_cad_grounding"


class CandidateLayoutError(ValueError):
    """Raised when candidate-layout evidence cannot be validated or persisted."""


@dataclass(frozen=True)
class CandidateSpatialRelationResult:
    """Return one persisted requirement-blind candidate geometry record."""

    record_path: Path
    status: str
    record: Mapping[str, object]


def analyze_candidate_layout(
    *,
    interaction_root: Path,
    segmentation_record_path: Path,
    candidate_field_paths: Sequence[str],
    relation_number: int = 1,
) -> CandidateSpatialRelationResult:
    """Measure two or more PA-selected candidates from one observation frame."""
    _positive_integer(relation_number, "relation_number")
    if (
        not isinstance(candidate_field_paths, Sequence)
        or isinstance(candidate_field_paths, (str, bytes))
        or len(candidate_field_paths) < 2
        or any(not isinstance(path, str) or not path for path in candidate_field_paths)
        or len(set(candidate_field_paths)) != len(candidate_field_paths)
    ):
        raise CandidateLayoutError(
            "candidate_field_paths must contain at least two unique JSON pointers."
        )
    root = Path(interaction_root).resolve()
    segmentation_path, segmentation_ref, segmentation = _load_local_record(
        root,
        segmentation_record_path,
        "segmentation",
    )
    if (
        segmentation.get("record_type") != "RGBDSegmentationRecord"
        or segmentation.get("producer") != _PRODUCER
    ):
        raise CandidateLayoutError("Candidate-layout segmentation identity is invalid.")

    candidates = [
        _candidate_at_path(segmentation, segmentation_ref, field_path)
        for field_path in candidate_field_paths
    ]
    views = {
        (candidate["observation_handle"], candidate["frame"])
        for candidate in candidates
    }
    if len(views) != 1:
        raise CandidateLayoutError("Selected candidates must come from the same view and frame.")

    pairwise_measurements = []
    for first, second in itertools.combinations(candidates, 2):
        displacement = [
            float(second["center_m"][axis]) - float(first["center_m"][axis])
            for axis in range(3)
        ]
        pairwise_measurements.append(
            {
                "first": _candidate_reference(first),
                "second": _candidate_reference(second),
                "displacement_m": displacement,
                "distance_m": math.sqrt(sum(component * component for component in displacement)),
            }
        )

    collinearity_measurements = [
        _collinearity_measurement(first, second, third)
        for first, second, third in itertools.combinations(candidates, 3)
    ]
    record: dict[str, object] = {
        "schema_version": 2,
        "record_type": "CandidateSpatialRelationRecord",
        "producer": _PRODUCER,
        "relation_number": relation_number,
        "method": "euclidean_candidate_geometry",
        "parameters": {"minimum_candidate_count": 2},
        "segmentation": {
            "ref": segmentation_ref,
            "sha256": _sha256_path(segmentation_path),
        },
        "evidence_refs": [segmentation_ref],
        "candidate_count": len(candidates),
        "candidates": [_candidate_projection(candidate) for candidate in candidates],
        "pairwise_measurements": pairwise_measurements,
        "collinearity_measurements": collinearity_measurements,
        "status": "measured",
    }
    record["fingerprint"] = _fingerprint(record)
    destination = root / _GROUNDING_ROOT / f"layout_{relation_number:04d}"
    _persist_record(destination, record)
    return CandidateSpatialRelationResult(
        record_path=destination / "relation_record.json",
        status="accepted",
        record=record,
    )


def _candidate_at_path(
    segmentation: Mapping[str, object],
    segmentation_ref: str,
    field_path: str,
) -> dict[str, object]:
    parts = field_path.split("/")
    if (
        len(parts) != 5
        or parts[0] != ""
        or parts[1] != "cameras"
        or not parts[2].isdigit()
        or parts[3] != "candidates"
        or not parts[4].isdigit()
    ):
        raise CandidateLayoutError("Candidate field path is invalid.")
    cameras = segmentation.get("cameras")
    camera_index = int(parts[2])
    candidate_index = int(parts[4])
    if not isinstance(cameras, list) or camera_index >= len(cameras):
        raise CandidateLayoutError("Candidate field path does not exist.")
    camera = cameras[camera_index]
    candidates = camera.get("candidates") if isinstance(camera, Mapping) else None
    if not isinstance(candidates, list) or candidate_index >= len(candidates):
        raise CandidateLayoutError("Candidate field path does not exist.")
    candidate = candidates[candidate_index]
    if not isinstance(candidate, Mapping):
        raise CandidateLayoutError("Candidate record is invalid.")
    return {
        "observation_handle": _text(
            camera.get("observation_handle"),
            "observation_handle",
        ),
        "candidate_handle": _text(candidate.get("candidate_handle"), "candidate_handle"),
        "frame": _text(camera.get("frame"), "candidate frame"),
        "center_m": _finite_vector(candidate.get("centroid_m"), "candidate centroid_m"),
        "value_ref": {
            "record_ref": segmentation_ref,
            "field_path": field_path,
        },
    }


def _collinearity_measurement(
    first: Mapping[str, object],
    second: Mapping[str, object],
    third: Mapping[str, object],
) -> dict[str, object]:
    first_center = first["center_m"]
    second_center = second["center_m"]
    third_center = third["center_m"]
    ab = [float(second_center[index]) - float(first_center[index]) for index in range(3)]
    ac = [float(third_center[index]) - float(first_center[index]) for index in range(3)]
    cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ]
    ab_norm = math.sqrt(sum(value * value for value in ab))
    ac_norm = math.sqrt(sum(value * value for value in ac))
    denominator = ab_norm * ac_norm
    return {
        "candidates": [
            _candidate_reference(first),
            _candidate_reference(second),
            _candidate_reference(third),
        ],
        "normalized_cross_product": (
            None
            if denominator <= 1e-12
            else math.sqrt(sum(value * value for value in cross)) / denominator
        ),
    }


def _candidate_projection(candidate: Mapping[str, object]) -> dict[str, object]:
    return {
        **_candidate_reference(candidate),
        "frame": candidate["frame"],
        "center_m": list(candidate["center_m"]),
    }


def _candidate_reference(candidate: Mapping[str, object]) -> dict[str, object]:
    return {
        "observation_handle": candidate["observation_handle"],
        "candidate_handle": candidate["candidate_handle"],
        "value_ref": dict(candidate["value_ref"]),
    }


def _load_local_record(
    root: Path,
    value: object,
    label: str,
) -> tuple[Path, str, Mapping[str, object]]:
    path = Path(value).resolve() if isinstance(value, Path) else None
    if path is None:
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise CandidateLayoutError(f"{label} record reference is invalid.")
        path = (root / value).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise CandidateLayoutError(f"{label} record leaves the interaction.") from exc
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateLayoutError(f"{label} record could not be read.") from exc
    if not isinstance(record, Mapping):
        raise CandidateLayoutError(f"{label} record is invalid.")
    return path, relative, record


def _persist_record(destination: Path, record: Mapping[str, object]) -> None:
    if destination.exists():
        raise CandidateLayoutError("Candidate layout already exists.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".layout-", dir=destination.parent))
    try:
        (temporary / "relation_record.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
    except (OSError, TypeError, ValueError) as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise CandidateLayoutError("Candidate layout could not be persisted.") from exc


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    """Hash one exact local premise for immutable provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value: object, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise CandidateLayoutError(f"{label} must contain three coordinates.")
    coordinates: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CandidateLayoutError(f"{label} is invalid.")
        coordinate = float(item)
        if not math.isfinite(coordinate):
            raise CandidateLayoutError(f"{label} is non-finite.")
        coordinates.append(coordinate)
    return coordinates


def _positive_integer(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CandidateLayoutError(f"{label} must be a positive integer.")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CandidateLayoutError(f"{label} is invalid.")
    return value

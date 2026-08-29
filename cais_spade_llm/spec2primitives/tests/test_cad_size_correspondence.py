"""Tests for Phase 4.2B2A CAD-size candidate association."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    resolve_context_ref,
)
from cais_spade_llm.spec2primitives.tools.observation_context import (
    CAMERA_IDS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    CameraCalibration,
    CameraObservation,
    ObservationBundle,
    write_observation_bundle,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    CADSizeAssociationError,
    associate_segmented_candidate_by_size,
    preprocess_served_geometry,
    read_rgbd_segmentation_status,
    run_cad_size_association_pipeline,
    segment_preprocessed_observation,
)

_DEPTH_M = 0.8
_FOCAL_LENGTH_PX = 800.0


def test_medium_gear_selects_42_mm_candidate_and_reports_camera_location(
    tmp_path: Path,
) -> None:
    segmentation_path, cad_path = _prepare_inputs(
        tmp_path,
        _size_bundle((0.022, 0.042, 0.062)),
    )

    result = associate_segmented_candidate_by_size(
        interaction_root=tmp_path,
        segmentation_record_path=segmentation_path,
        cad_record_path=cad_path,
    )

    assert result.CAD_correspondence == "accepted"
    assert result.location == "available"
    selected = result.selected_candidate
    assert selected is not None
    assert selected["camera_id"] == "cam_mk3"
    assert selected["frame"] == "cam_mk3_optical_frame"
    assert selected["candidate_id"] == 2
    np.testing.assert_allclose(
        selected["candidate_center_m"],
        [-0.04, -0.06, 0.8],
        atol=1.1e-3,
    )
    np.testing.assert_allclose(
        selected["observed_dimensions_m"],
        [0.042, 0.042],
        atol=1.1e-3,
    )
    record = _read_json(result.record_path)
    assert record == result.record
    assert record["record_type"] == "CADSizeCorrespondenceRecord"
    assert record["CAD"]["context_ref"] == "Gear_Medium.STL"
    assert record["CAD"]["record"]["sha256"] == _sha256(cad_path)
    assert record["segmentation"]["record"]["sha256"] == _sha256(
        segmentation_path
    )
    assert record["pose"] == "not_evaluated"
    assert record["cross_camera_fusion"] == "not_evaluated"


def test_measurement_noise_within_fifteen_percent_is_accepted(tmp_path: Path) -> None:
    segmentation_path, cad_path = _prepare_inputs(
        tmp_path,
        _size_bundle((0.044,)),
    )

    result = associate_segmented_candidate_by_size(
        interaction_root=tmp_path,
        segmentation_record_path=segmentation_path,
        cad_record_path=cad_path,
    )

    assert result.CAD_correspondence == "accepted"
    assert result.selected_candidate is not None
    assert max(result.selected_candidate["dimension_errors"]) <= 0.15


def test_large_assembly_candidate_summary_is_order_stable(tmp_path: Path) -> None:
    segmentation_path, cad_path = _prepare_inputs(
        tmp_path,
        _size_bundle((0.042,), large_assembly_candidate=True),
    )
    segmentation_record = _read_json(segmentation_path)
    assembly_camera = next(
        camera for camera in segmentation_record["cameras"] if camera["camera_id"] == "cam_assembly"
    )

    result = associate_segmented_candidate_by_size(
        interaction_root=tmp_path,
        segmentation_record_path=segmentation_path,
        cad_record_path=cad_path,
    )

    assert assembly_camera["candidates"][0]["point_count"] > 200_000
    assert result.CAD_correspondence == "accepted"


def test_tampered_candidate_centroid_rejects_without_partial_output(
    tmp_path: Path,
) -> None:
    segmentation_path, cad_path = _prepare_inputs(
        tmp_path,
        _size_bundle((0.042,)),
    )
    segmentation_record = _read_json(segmentation_path)
    centroid = segmentation_record["cameras"][0]["candidates"][0]["centroid_m"]
    centroid[0] += 0.001
    segmentation_path.write_text(
        json.dumps(segmentation_record, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        CADSizeAssociationError,
        match="candidate summary is inconsistent",
    ):
        associate_segmented_candidate_by_size(
            interaction_root=tmp_path,
            segmentation_record_path=segmentation_path,
            cad_record_path=cad_path,
        )

    grounding_root = tmp_path / "products/grounding/rgb_d_cad_grounding"
    assert not (grounding_root / "correspondence_0001").exists()
    assert list(grounding_root.glob(".correspondence-*")) == []


def test_duplicate_size_candidates_are_ambiguous(tmp_path: Path) -> None:
    segmentation_path, cad_path = _prepare_inputs(
        tmp_path,
        _size_bundle((0.042, 0.042)),
    )

    result = associate_segmented_candidate_by_size(
        interaction_root=tmp_path,
        segmentation_record_path=segmentation_path,
        cad_record_path=cad_path,
    )

    assert result.CAD_correspondence == "ambiguous"
    assert result.location == "ambiguous"
    assert result.selected_candidate is None
    assert len(result.record["plausible_candidates"]) == 2


@pytest.mark.parametrize("diameters_m", [(0.022, 0.062), ()])
def test_no_matching_or_zero_candidates_are_rejected(
    tmp_path: Path,
    diameters_m: tuple[float, ...],
) -> None:
    bundle = _size_bundle(diameters_m, zero_candidates=not diameters_m)
    segmentation_path, cad_path = _prepare_inputs(tmp_path, bundle)

    result = associate_segmented_candidate_by_size(
        interaction_root=tmp_path,
        segmentation_record_path=segmentation_path,
        cad_record_path=cad_path,
    )

    assert result.CAD_correspondence == "rejected"
    assert result.location == "unavailable"
    assert result.selected_candidate is None


def test_candidate_touching_image_boundary_fails_closed_as_partial_visibility(
    tmp_path: Path,
) -> None:
    segmentation_path, cad_path = _prepare_inputs(
        tmp_path,
        _size_bundle((0.042,), clipped_candidate=True),
    )

    result = associate_segmented_candidate_by_size(
        interaction_root=tmp_path,
        segmentation_record_path=segmentation_path,
        cad_record_path=cad_path,
    )

    assert result.CAD_correspondence == "rejected"
    assert result.location == "unavailable"
    partial = [
        candidate
        for candidate in result.record["ranked_candidates"]
        if candidate["camera_id"] == "cam_mk3"
    ]
    assert len(partial) == 1
    assert partial[0]["measurement_status"] == "partial_visibility"
    assert partial[0]["observed_dimensions_m"] is None


def test_ranking_is_deterministic_and_existing_record_is_not_overwritten(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_inputs = _prepare_inputs(first_root, _size_bundle((0.022, 0.042, 0.062)))
    second_inputs = _prepare_inputs(second_root, _size_bundle((0.022, 0.042, 0.062)))
    first = associate_segmented_candidate_by_size(
        interaction_root=first_root,
        segmentation_record_path=first_inputs[0],
        cad_record_path=first_inputs[1],
    )
    second = associate_segmented_candidate_by_size(
        interaction_root=second_root,
        segmentation_record_path=second_inputs[0],
        cad_record_path=second_inputs[1],
    )

    assert first.record["ranked_candidates"] == second.record["ranked_candidates"]
    before = first.record_path.read_bytes()
    with pytest.raises(CADSizeAssociationError, match="already exists"):
        associate_segmented_candidate_by_size(
            interaction_root=first_root,
            segmentation_record_path=first_inputs[0],
            cad_record_path=first_inputs[1],
        )
    assert first.record_path.read_bytes() == before


def test_tampered_label_mask_rejects_without_partial_output(tmp_path: Path) -> None:
    segmentation_path, cad_path = _prepare_inputs(
        tmp_path,
        _size_bundle((0.042,)),
    )
    label_path = segmentation_path.parent / "cam_mk3_candidate_labels.npy"
    tampered = bytearray(label_path.read_bytes())
    tampered[-1] ^= 1
    label_path.write_bytes(tampered)

    with pytest.raises(CADSizeAssociationError, match="hash"):
        associate_segmented_candidate_by_size(
            interaction_root=tmp_path,
            segmentation_record_path=segmentation_path,
            cad_record_path=cad_path,
        )

    grounding_root = tmp_path / "products/grounding/rgb_d_cad_grounding"
    assert not (grounding_root / "correspondence_0001").exists()
    assert list(grounding_root.glob(".correspondence-*")) == []


def test_status_wrapper_exposes_only_compact_association_states(tmp_path: Path) -> None:
    interaction_root = tmp_path / "interaction"
    segmentation_path, cad_path = _prepare_inputs(
        interaction_root,
        _size_bundle((0.042,)),
    )

    result = run_cad_size_association_pipeline(
        contexts_root=tmp_path,
        interaction_root=interaction_root,
        segmentation_record_path=segmentation_path,
        cad_record_path=cad_path,
    )

    assert result == {
        "status": "ready",
        "interaction_root": str(interaction_root.resolve()),
        "correspondence_record_path": result["correspondence_record_path"],
        "CAD_correspondence": "accepted",
        "location": "available",
        "pose": "not_evaluated",
        "failure": None,
    }
    assert "selected_candidate" not in result
    assert "candidate_center_m" not in result
    status = read_rgbd_segmentation_status(tmp_path)
    assert status["identity"] == "not_evaluated"
    assert status["CAD_correspondence"] == "accepted"
    assert status["location"] == "available"
    assert status["pose"] == "not_evaluated"


def _prepare_inputs(
    interaction_root: Path,
    bundle: ObservationBundle,
) -> tuple[Path, Path]:
    interaction_root.mkdir(parents=True, exist_ok=True)
    bundle_path = write_observation_bundle(
        interaction_root / "products/observations",
        bundle,
    )
    manifest = _read_json(bundle_path / "manifest.json")
    bundle_ref = bundle_path.relative_to(interaction_root)
    served_observation = {
        "context_ref": None,
        "observation_ref": bundle.observation_ref,
        "evidence_type": "observation",
        "evidence_label": bundle.evidence_label,
        "provenance": {"manifest_path": str(bundle_ref / "manifest.json")},
        "observation_evidence": {
            "manifest": manifest,
            "artifact_references": [
                {
                    "camera_id": camera_id,
                    "rgb_artifact": str(bundle_ref / f"{camera_id}_rgb.png"),
                    "depth_artifact": str(bundle_ref / f"{camera_id}_depth_m.npy"),
                }
                for camera_id in CAMERA_IDS
            ],
        },
    }
    observation = preprocess_served_geometry(
        interaction_root=interaction_root,
        served_context=served_observation,
        operation_number=1,
    )
    segmentation = segment_preprocessed_observation(
        interaction_root=interaction_root,
        observation_record_path=observation.record_path,
    )
    resolution = resolve_context_ref({"context_ref": "Gear_Medium.STL"})
    assert "served_context" in resolution
    cad = preprocess_served_geometry(
        interaction_root=interaction_root,
        served_context=resolution["served_context"],
        operation_number=2,
    )
    return segmentation.record_path, cad.record_path


def _size_bundle(
    diameters_m: tuple[float, ...],
    *,
    clipped_candidate: bool = False,
    zero_candidates: bool = False,
    large_assembly_candidate: bool = False,
) -> ObservationBundle:
    observations = []
    for camera_index, camera_id in enumerate(CAMERA_IDS):
        rgb = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        depth_m = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), np.nan, dtype=np.float32)
        if zero_candidates:
            depth_m[100:105, 100:105] = np.float32(1.0)
        elif camera_id == "cam_assembly":
            if large_assembly_candidate:
                assembly_rows = slice(20, 460)
                assembly_columns = slice(40, 600)
            else:
                assembly_rows = slice(100, 300)
                assembly_columns = slice(200, 440)
            depth_m[assembly_rows, assembly_columns] = np.float32(0.9)
            rgb[assembly_rows, assembly_columns] = np.asarray(
                [80, 90, 100],
                dtype=np.uint8,
            )
        else:
            depth_m[80:320, 20:600] = np.float32(1.0)
            rgb[80:320, 20:600] = np.asarray([20, 30, 40], dtype=np.uint8)
            if camera_id == "cam_mk3":
                _add_size_candidates(
                    rgb,
                    depth_m,
                    diameters_m,
                    clipped_candidate=clipped_candidate,
                )
        timestamp_ns = 3_000_000_000 + camera_index * 1_000
        frame = f"{camera_id}_optical_frame"
        observations.append(
            CameraObservation(
                camera_id=camera_id,
                rgb=rgb,
                depth_m=depth_m,
                rgb_timestamp_ns=timestamp_ns,
                depth_timestamp_ns=timestamp_ns + 100,
                rgb_frame=frame,
                depth_frame=frame,
                camera_calibration=_calibration(frame),
            )
        )
    return ObservationBundle(
        observation_ref="observation_0001",
        evidence_label="live",
        captured_at_ns=3_000_004_000,
        camera_observations=tuple(observations),
    )


def _add_size_candidates(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    diameters_m: tuple[float, ...],
    *,
    clipped_candidate: bool,
) -> None:
    grid_v, grid_u = np.ogrid[:IMAGE_HEIGHT, :IMAGE_WIDTH]
    default_centers = ((140, 180), (280, 180), (440, 180))
    centers = (
        ((0, 180),)
        if clipped_candidate
        else default_centers[: len(diameters_m)]
    )
    for index, (diameter_m, (center_u, center_v)) in enumerate(
        zip(diameters_m, centers, strict=True)
    ):
        radius_px = round(diameter_m * _FOCAL_LENGTH_PX / (2.0 * _DEPTH_M))
        mask = (
            np.square(grid_u - center_u) + np.square(grid_v - center_v)
            <= radius_px**2
        )
        depth_m[mask] = np.float32(_DEPTH_M)
        color = np.asarray([200 - index * 30, 20 + index * 40, 50], dtype=np.uint8)
        rgb[mask] = color


def _calibration(frame: str) -> CameraCalibration:
    return CameraCalibration(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        frame=frame,
        distortion_model="plumb_bob",
        D=(0.0, 0.0, 0.0, 0.0, 0.0),
        K=(800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0),
        R=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        P=(800.0, 0.0, 320.0, 0.0, 0.0, 800.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

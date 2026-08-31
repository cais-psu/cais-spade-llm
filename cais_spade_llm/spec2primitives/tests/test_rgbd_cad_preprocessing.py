"""Tests for Phase 4.2A CAD and RGB-D preprocessing."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    initialize_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    ontology_config,
)
from cais_spade_llm.spec2primitives.tools import exact_ref_resolver
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_cad_path,
    approved_cad_refs,
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
    GeometryPreprocessingError,
    preprocess_served_geometry,
    preprocessor,
    run_rgbd_cad_preprocessing_diagnostic,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.gazebo_observation_provider import (
    GazeboObservationProviderError,
)


class ControlledCapture:
    """Persist one deterministic fresh bundle for a diagnostic request."""

    def __init__(self, bundle: ObservationBundle) -> None:
        self.bundle = bundle
        self.calls: list[tuple[Path, str, float]] = []

    def capture(
        self,
        observations_root: Path,
        observation_ref: str,
        *,
        timeout_sec: float,
    ) -> Path:
        """Write the controlled bundle under the exact requested ref."""
        self.calls.append((observations_root, observation_ref, timeout_sec))
        return write_observation_bundle(
            observations_root,
            replace(self.bundle, observation_ref=observation_ref, evidence_label="live"),
        )


class RejectedCapture:
    """Reject a live capture using a stable provider reason."""

    def capture(
        self,
        observations_root: Path,
        observation_ref: str,
        *,
        timeout_sec: float,
    ) -> Path:
        """Raise without writing observation evidence."""
        del observations_root, observation_ref, timeout_sec
        raise GazeboObservationProviderError(
            "observation_timeout",
            "Controlled live capture timed out.",
        )


def test_approved_cad_accessor_accepts_only_exact_inventory_refs() -> None:
    refs = approved_cad_refs()

    assert refs
    assert all(approved_cad_path(ref).name == ref for ref in refs)
    for invalid_ref in (
        "NIST_assembly_instructions.pdf",
        "../Gear_Medium.STL",
        "/tmp/Gear_Medium.STL",
        "Gear_Medium.STL ",
        "Not_Approved.STL",
    ):
        with pytest.raises(ValueError):
            approved_cad_path(invalid_ref)


def test_cad_preprocessing_preserves_full_mesh_units_hashes_and_delta(
    tmp_path: Path,
) -> None:
    served_context = _served_cad("Gear_Medium.STL")
    source_path = approved_cad_path("Gear_Medium.STL")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    result = preprocess_served_geometry(
        interaction_root=tmp_path,
        served_context=served_context,
        operation_number=1,
    )

    record = result.record
    artifact = result.artifact_paths[0]
    with np.load(artifact, allow_pickle=False) as arrays:
        triangles_m = arrays["triangles_m"]
        facet_normals = arrays["facet_normals"]

    assert record["record_type"] == "CADMeshRecord"
    assert record["triangle_count"] == served_context["CAD_evidence"]["triangle_count"]
    assert triangles_m.shape == (record["triangle_count"], 3, 3)
    assert facet_normals.shape == (record["triangle_count"], 3)
    assert triangles_m.dtype == np.float32
    assert facet_normals.dtype == np.float32
    np.testing.assert_allclose(
        triangles_m.min(axis=(0, 1)),
        np.asarray(served_context["CAD_evidence"]["bounds_mm"]["minimum"]) * 0.001,
        rtol=0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        triangles_m.max(axis=(0, 1)),
        np.asarray(served_context["CAD_evidence"]["bounds_mm"]["maximum"]) * 0.001,
        rtol=0.0,
        atol=1e-8,
    )
    assert record["stored_units"] == "m"
    assert record["source"]["source_units"] == "mm"
    assert record["source"]["source_sha256"] == source_sha256
    assert record["artifacts"]["mesh"]["sha256"] == _sha256(artifact)
    assert record["correspondence"] == "not_evaluated"
    assert record["pose"] == "not_evaluated"
    assert result.delta["assertions"] == []
    assert result.delta["typed_context_refs"] == [
        "products/grounding/rgb_d_cad_grounding/operation_0001/geometry_record.json"
    ]
    assert result.delta["unresolved_evidence_needs"] == []
    assert _read_json(result.record_path) == record


def test_cad_preprocessing_accepts_served_gear_shaft_bounds(tmp_path: Path) -> None:
    served_context = _served_cad("Gear_Shaft.STL")

    result = preprocess_served_geometry(
        interaction_root=tmp_path,
        served_context=served_context,
        operation_number=1,
    )

    assert result.evidence_ref == "Gear_Shaft.STL"
    assert result.record["source"]["context_ref"] == "Gear_Shaft.STL"
    assert result.record_path.is_file()
    assert all(path.is_file() for path in result.artifact_paths)


def test_cad_preprocessing_rejects_tampered_served_bounds(tmp_path: Path) -> None:
    served_context = _served_cad("Gear_Shaft.STL")
    served_context["CAD_evidence"]["bounds_mm"]["minimum"][0] += 0.001

    with pytest.raises(GeometryPreprocessingError, match="bounds do not match"):
        preprocess_served_geometry(
            interaction_root=tmp_path,
            served_context=served_context,
            operation_number=1,
        )

    operation_root = (
        tmp_path
        / "products/grounding/rgb_d_cad_grounding/operation_0001"
    )
    assert not operation_root.exists()


def test_cad_preprocessing_rejects_source_changed_after_serving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    served_context = _served_cad("Gear_Medium.STL")
    changed_source = tmp_path / "Gear_Medium.STL"
    shutil.copyfile(approved_cad_path("Gear_Medium.STL"), changed_source)
    source_bytes = bytearray(changed_source.read_bytes())
    source_bytes[-1] ^= 1
    changed_source.write_bytes(source_bytes)
    monkeypatch.setattr(preprocessor, "approved_cad_path", lambda _ref: changed_source)

    with pytest.raises(GeometryPreprocessingError, match="changed after"):
        preprocess_served_geometry(
            interaction_root=tmp_path / "interaction",
            served_context=served_context,
            operation_number=1,
        )

    operation_root = tmp_path / "interaction/products/grounding/rgb_d_cad_grounding/operation_0001"
    assert not operation_root.exists()


def test_cad_preprocessing_does_not_overwrite_an_operation(tmp_path: Path) -> None:
    served_context = _served_cad("Gear_Medium.STL")
    result = preprocess_served_geometry(
        interaction_root=tmp_path,
        served_context=served_context,
        operation_number=1,
    )
    before = result.record_path.read_bytes()

    with pytest.raises(GeometryPreprocessingError, match="already exists"):
        preprocess_served_geometry(
            interaction_root=tmp_path,
            served_context=served_context,
            operation_number=1,
        )

    assert result.record_path.read_bytes() == before


def test_rgbd_preprocessing_deprojects_all_valid_pixels_with_rgb_association(
    tmp_path: Path,
) -> None:
    bundle = _observation_bundle()
    served_context = _write_served_observation(tmp_path, bundle)

    result = preprocess_served_geometry(
        interaction_root=tmp_path,
        served_context=served_context,
        operation_number=1,
    )

    assert result.record["record_type"] == "ColoredPointCloudSetRecord"
    assert result.record["coordinate_convention"] == "+x right, +y down, +z forward"
    assert result.record["cross_camera_fusion"] == "not_evaluated"
    assert result.record["extrinsics_available"] is False
    assert result.record["correspondence"] == "not_evaluated"
    assert result.record["pose"] == "not_evaluated"
    assert result.delta["assertions"] == []

    for camera_index, camera_id in enumerate(CAMERA_IDS):
        artifact = next(path for path in result.artifact_paths if camera_id in path.name)
        with np.load(artifact, allow_pickle=False) as arrays:
            points_m = arrays["points_m"]
            colors_rgb = arrays["colors_rgb"]
            pixels_uv = arrays["pixels_uv"]
        first_depth = np.float32(1.0 + camera_index)
        second_depth = np.float32(2.0 + camera_index)
        np.testing.assert_allclose(
            points_m,
            np.asarray(
                [
                    [0.0, 0.0, first_depth],
                    [second_depth, second_depth * 0.5, second_depth],
                ],
                dtype=np.float32,
            ),
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            pixels_uv,
            np.asarray([[320, 240], [420, 340]], dtype=np.uint16),
        )
        np.testing.assert_array_equal(
            colors_rgb,
            np.asarray(
                [[10 + camera_index, 20, 30], [40, 50 + camera_index, 60]],
                dtype=np.uint8,
            ),
        )
        camera_record = result.record["cameras"][camera_index]
        assert camera_record["camera_id"] == camera_id
        assert camera_record["frame"] == f"{camera_id}_optical_frame"
        assert camera_record["point_count"] == 2
        assert camera_record["projection"]["K"] == list(
            bundle.camera_observations[camera_index].camera_calibration.K
        )
        assert camera_record["point_cloud_artifact"]["sha256"] == _sha256(artifact)


@pytest.mark.parametrize(
    "calibration",
    [
        CameraCalibration(
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            frame="unsupported_optical_frame",
            distortion_model="equidistant",
            D=(0.0, 0.0, 0.0, 0.0),
            K=(100.0, 0.0, 320.0, 0.0, 200.0, 240.0, 0.0, 0.0, 1.0),
            R=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            P=(100.0, 0.0, 320.0, 0.0, 0.0, 200.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        ),
        CameraCalibration(
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            frame="invalid_optical_frame",
            distortion_model="plumb_bob",
            D=(0.0, 0.0, 0.0, 0.0, 0.0),
            K=(0.0, 0.0, 320.0, 0.0, 200.0, 240.0, 0.0, 0.0, 1.0),
            R=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            P=(
                100.0,
                0.0,
                320.0,
                0.0,
                0.0,
                200.0,
                240.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ),
        ),
    ],
)
def test_rgbd_preprocessing_fails_closed_without_partial_operation(
    tmp_path: Path,
    calibration: CameraCalibration,
) -> None:
    bundle = _observation_bundle()
    first = replace(bundle.camera_observations[0], camera_calibration=calibration)
    invalid_bundle = replace(
        bundle,
        camera_observations=(first, *bundle.camera_observations[1:]),
    )
    served_context = _write_served_observation(tmp_path, invalid_bundle)

    with pytest.raises(GeometryPreprocessingError):
        preprocess_served_geometry(
            interaction_root=tmp_path,
            served_context=served_context,
            operation_number=1,
        )

    grounding_root = tmp_path / "products/grounding/rgb_d_cad_grounding"
    assert not (grounding_root / "operation_0001").exists()
    assert list(grounding_root.glob(".preprocessing-*")) == []


def test_geometry_deltas_merge_without_changing_the_initialized_graph(
    tmp_path: Path,
) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble Medium Gear", tbox)
    initial_graph = set(abox.graph)
    cad_result = preprocess_served_geometry(
        interaction_root=tmp_path,
        served_context=_served_cad("Gear_Medium.STL"),
        operation_number=1,
    )
    observation_result = preprocess_served_geometry(
        interaction_root=tmp_path,
        served_context=_write_served_observation(tmp_path, _observation_bundle()),
        operation_number=2,
    )

    first_merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        "rgb_d_cad_grounding",
        cad_result.delta,
        authorized_evidence_refs=[cad_result.evidence_ref],
    )
    second_merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        "rgb_d_cad_grounding",
        observation_result.delta,
        authorized_evidence_refs=[observation_result.evidence_ref],
    )

    assert set(second_merge.abox.graph) == initial_graph
    assert first_merge.assertion_count == 0
    assert second_merge.assertion_count == 0
    assert second_merge.abox.delta_count == 2
    assert _read_json(first_merge.delta_path)["typed_context_refs"] == list(
        cad_result.delta["typed_context_refs"]
    )
    assert _read_json(second_merge.delta_path)["typed_context_refs"] == list(
        observation_result.delta["typed_context_refs"]
    )


def test_diagnostic_uses_fresh_injected_capture_and_reports_only_preprocessing(
    tmp_path: Path,
) -> None:
    capture = ControlledCapture(_observation_bundle())

    result = run_rgbd_cad_preprocessing_diagnostic(
        interaction_root=tmp_path,
        cad_ref="Gear_Medium.STL",
        capture_timeout_sec=7.5,
        capture_runtime=capture,
    )

    assert result["status"] == "preprocessed"
    assert capture.calls == [(tmp_path / "products/observations", "observation_0001", 7.5)]
    assert result["CAD"]["triangle_count"] > 0
    assert [camera["camera_id"] for camera in result["RGB_D"]["cameras"]] == list(CAMERA_IDS)
    assert result["correspondence"] == "not_evaluated"
    assert result["pose"] == "not_evaluated"
    assert result["CAD"]["delta"]["assertions"] == []
    assert result["RGB_D"]["delta"]["assertions"] == []
    persisted = _read_json(Path(result["diagnostic_record_path"]))
    assert persisted == result
    assert "context understanding complete" not in json.dumps(result)
    assert "matching" not in json.dumps(result).lower()


def test_diagnostic_records_capture_failure_without_partial_rgbd_operation(
    tmp_path: Path,
) -> None:
    result = run_rgbd_cad_preprocessing_diagnostic(
        interaction_root=tmp_path,
        cad_ref="Gear_Medium.STL",
        capture_timeout_sec=2.0,
        capture_runtime=RejectedCapture(),
    )

    assert result["status"] == "rejected"
    assert result["failure"]["reason"] == "observation_timeout"
    assert result["CAD_record_path"].endswith("operation_0001/geometry_record.json")
    assert result["RGB_D_record_path"] is None
    assert result["correspondence"] == "not_evaluated"
    assert result["pose"] == "not_evaluated"
    assert not (tmp_path / "products/grounding/rgb_d_cad_grounding/operation_0002").exists()
    assert _read_json(Path(result["diagnostic_record_path"])) == result


def test_diagnostic_rejects_missing_or_non_exact_cad_without_capture(
    tmp_path: Path,
) -> None:
    capture = ControlledCapture(_observation_bundle())

    result = run_rgbd_cad_preprocessing_diagnostic(
        interaction_root=tmp_path,
        cad_ref="../Gear_Medium.STL",
        capture_timeout_sec=5.0,
        capture_runtime=capture,
    )

    assert result["status"] == "rejected"
    assert result["failure"]["reason"] == "invalid_diagnostic_request"
    assert capture.calls == []
    assert not (tmp_path / "products").exists()


def test_exact_resolver_rejects_nonfinite_stl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "Invalid.STL"
    source_path.write_bytes(
        b"\x00" * 80
        + (1).to_bytes(4, "little")
        + np.asarray(
            [0.0, 0.0, 1.0, np.nan, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype="<f4"
        ).tobytes()
        + b"\x00\x00"
    )
    inventory_path = tmp_path / "approved_sources.json"
    inventory_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "context_ref": "Invalid.STL",
                        "evidence_type": "CAD",
                        "repository_path": "Invalid.STL",
                        "source_url": "https://example.invalid/source",
                        "units": "mm",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(exact_ref_resolver, "_INVENTORY_PATH", inventory_path)
    monkeypatch.setattr(exact_ref_resolver, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(exact_ref_resolver, "_CAD_ROOT", tmp_path)

    result = resolve_context_ref({"context_ref": "Invalid.STL"})

    assert result["rejection"]["reason"] == "malformed_source"


def _served_cad(context_ref: str) -> dict[str, Any]:
    result = resolve_context_ref({"context_ref": context_ref})
    assert "rejection" not in result
    return result["served_context"]


def _observation_bundle() -> ObservationBundle:
    camera_observations = []
    for camera_index, camera_id in enumerate(CAMERA_IDS):
        rgb = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        rgb[240, 320] = np.asarray([10 + camera_index, 20, 30], dtype=np.uint8)
        rgb[340, 420] = np.asarray([40, 50 + camera_index, 60], dtype=np.uint8)
        depth_m = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), np.nan, dtype=np.float32)
        depth_m[0, 0] = -1.0
        depth_m[0, 1] = 0.0
        depth_m[0, 2] = np.inf
        depth_m[240, 320] = np.float32(1.0 + camera_index)
        depth_m[340, 420] = np.float32(2.0 + camera_index)
        timestamp_ns = 1_000_000_000 + camera_index * 1_000
        frame = f"{camera_id}_optical_frame"
        camera_observations.append(
            CameraObservation(
                camera_id=camera_id,
                rgb=rgb,
                depth_m=depth_m,
                rgb_timestamp_ns=timestamp_ns,
                depth_timestamp_ns=timestamp_ns + 100,
                rgb_frame=frame,
                depth_frame=frame,
                camera_calibration=CameraCalibration(
                    width=IMAGE_WIDTH,
                    height=IMAGE_HEIGHT,
                    frame=frame,
                    distortion_model="plumb_bob",
                    D=(0.0, 0.0, 0.0, 0.0, 0.0),
                    K=(100.0, 0.0, 320.0, 0.0, 200.0, 240.0, 0.0, 0.0, 1.0),
                    R=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                    P=(
                        100.0,
                        0.0,
                        320.0,
                        0.0,
                        0.0,
                        200.0,
                        240.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                    ),
                ),
            )
        )
    return ObservationBundle(
        observation_ref="observation_0001",
        evidence_label="live",
        captured_at_ns=1_000_004_000,
        camera_observations=tuple(camera_observations),
    )


def _write_served_observation(
    interaction_root: Path,
    bundle: ObservationBundle,
) -> dict[str, object]:
    bundle_path = write_observation_bundle(
        interaction_root / "products/observations",
        bundle,
    )
    manifest = _read_json(bundle_path / "manifest.json")
    bundle_ref = bundle_path.relative_to(interaction_root)
    return {
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

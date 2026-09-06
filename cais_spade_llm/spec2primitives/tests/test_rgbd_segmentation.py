from __future__ import annotations

"""Tests for minimal automatic Phase 4.2B1 RGB-D segmentation."""


import asyncio
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cais_spade_llm.spec2primitives import spec2primitives_ui
from cais_spade_llm.spec2primitives.config import load_model_runtime_config
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
    ObservationCandidateReviewError,
    ObservationCandidateReviewRequest,
    ObservationCandidateReviewResponse,
    OpenAIObservationVisionRuntime,
    RGBDSegmentationError,
    preprocess_served_geometry,
    read_rgbd_segmentation_status,
    review_observation_candidates,
    run_automatic_rgbd_segmentation_pipeline,
    segment_preprocessed_observation,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.gazebo_observation_provider import (
    GazeboObservationProviderError,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.observation_review import (
    ObservationCandidateImage,
    _validated_output,
)


class ControlledCapture:
    """Persist one analytic live observation and record the automatic request."""

    def __init__(self, bundle: ObservationBundle) -> None:
        self.bundle = bundle
        self.calls: list[tuple[Path, str, float]] = []
        self.status_during_capture: list[str] = []

    def capture(
        self,
        observations_root: Path,
        observation_ref: str,
        *,
        timeout_sec: float,
    ) -> Path:
        """Write the exact requested bundle after observing running status."""
        self.calls.append((observations_root, observation_ref, timeout_sec))
        contexts_root = observations_root.parents[2]
        self.status_during_capture.append(
            str(read_rgbd_segmentation_status(contexts_root)["status"])
        )
        return write_observation_bundle(
            observations_root,
            replace(self.bundle, observation_ref=observation_ref, evidence_label="live"),
        )


class RejectedCapture:
    """Reject the automatic observation request without writing a bundle."""

    def capture(
        self,
        observations_root: Path,
        observation_ref: str,
        *,
        timeout_sec: float,
    ) -> Path:
        """Return one stable provider failure."""
        del observations_root, observation_ref, timeout_sec
        raise GazeboObservationProviderError(
            "observation_timeout",
            "Controlled automatic capture timed out.",
        )


class ControlledObservationVision:
    """Describe every opaque candidate without adding ontology assignments."""

    def __init__(self, *, forbidden_assignment: str | None = None) -> None:
        self.forbidden_assignment = forbidden_assignment
        self.requests: list[ObservationCandidateReviewRequest] = []

    async def review_candidates(
        self,
        request: ObservationCandidateReviewRequest,
    ) -> ObservationCandidateReviewResponse:
        """Return one deterministic description for every supplied handle."""
        self.requests.append(request)
        candidates: list[dict[str, object]] = []
        for candidate in request.candidates:
            value: dict[str, object] = {
                "observation_handle": candidate.observation_handle,
                "candidate_handle": candidate.candidate_handle,
                "description": f"visible region {candidate.candidate_handle}",
                "uncertainty": "identity not assigned",
            }
            if self.forbidden_assignment is not None:
                value[self.forbidden_assignment] = True
            candidates.append(value)
        return ObservationCandidateReviewResponse(
            response_id="response_observation_0001",
            model="gpt-5.6-sol",
            output={"candidates": candidates},
        )


def test_segmentation_applies_one_neutral_policy_to_every_observation(
    tmp_path: Path,
) -> None:
    preprocessing = _preprocess_observation(tmp_path, _segmentation_bundle())

    result = segment_preprocessed_observation(
        interaction_root=tmp_path,
        observation_record_path=preprocessing.record_path,
    )

    assert result.candidate_count == 6
    assert "schema_version" not in result.record
    assert result.record["candidate_count"] == 6
    assert result.record["candidate_state"] == "candidates_available"
    assert result.record["identity"] == "not_evaluated"
    assert result.record["CAD_correspondence"] == "not_evaluated"
    assert result.record["pose"] == "not_evaluated"
    assert result.record["cross_camera_fusion"] == "not_evaluated"
    assert result.record["parameters"] == {
        "ransac_seed": 0,
        "ransac_iterations": 256,
        "maximum_ransac_sample_points": 10_000,
        "plane_distance_threshold_m": 0.005,
        "minimum_plane_inlier_fraction": 0.2,
        "minimum_plane_inlier_points": 100,
        "depth_neighbor_threshold_m": 0.02,
        "minimum_candidate_points": 50,
        "maximum_candidates": 32,
    }

    for camera_index, camera in enumerate(result.record["cameras"], start=1):
        camera_id = camera["camera_id"]
        mask_path = result.record_path.parent / f"{camera_id}_candidate_labels.npy"
        labels = np.load(mask_path, allow_pickle=False)
        assert labels.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
        assert labels.dtype == np.uint16
        assert camera["label_mask_artifact"]["sha256"] == _sha256(mask_path)
        assert camera["frame"] == f"{camera_id}_optical_frame"
        assert camera["observation_handle"] == f"view_{camera_index:04d}"
        assert "role" not in camera
        assert camera["identity"] == "not_evaluated"
        assert camera["CAD_correspondence"] == "not_evaluated"
        assert camera["pose"] == "not_evaluated"
        if camera_id == "cam_assembly":
            assert camera["candidate_count"] == 0
            assert camera["support_plane"]["inlier_count"] == 8_000
            assert camera["support_plane"]["retained_point_count"] == 0
            assert labels[120, 120] == 0
        else:
            assert camera["candidate_count"] == 2
            assert camera["support_plane"]["retained_point_count"] == 200
            assert labels[105, 105] == 0
            assert (int(labels[125, 125]), int(labels[125, 135])) == (1, 2)
        assert camera["support_plane"]["status"] == "detected"
        assert camera["support_plane"]["candidate_filtering_applied"] is True
        assert camera["support_plane"]["candidate_side"] == "camera_side"
        for candidate in camera["candidates"]:
            assert candidate["candidate_handle"].startswith(f"candidate_{camera_index:04d}_")
            assert candidate["point_count"] >= 50
            assert candidate["identity"] == "not_evaluated"
            assert candidate["CAD_correspondence"] == "not_evaluated"
            assert candidate["pose"] == "not_evaluated"


def test_openai_observation_adapter_uses_current_runtime_configuration() -> None:
    class ControlledResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(
                id="response_observation_0001",
                model="gpt-5.6-sol",
                output_text=json.dumps(
                    {
                        "candidates": [
                            {
                                "observation_handle": "view_0001",
                                "candidate_handle": "candidate_0001_0001",
                                "description": "visible circular part",
                                "uncertainty": "identity not assigned",
                            }
                        ]
                    }
                ),
            )

    responses = ControlledResponses()
    config = load_model_runtime_config().observation_vlm
    runtime = OpenAIObservationVisionRuntime(
        config,
        client=SimpleNamespace(responses=responses),
    )
    request = ObservationCandidateReviewRequest(
        candidates=(
            ObservationCandidateImage(
                observation_handle="view_0001",
                candidate_handle="candidate_0001_0001",
                source_data_url="data:image/png;base64,source",
                crop_data_url="data:image/png;base64,crop",
            ),
        )
    )

    response = asyncio.run(runtime.review_candidates(request))

    assert response.model == "gpt-5.6-sol"
    call = responses.calls[0]
    assert call["model"] == "gpt-5.6"
    assert call["reasoning"] == {"effort": "medium"}
    assert call["max_output_tokens"] == 4096
    assert call["store"] is False
    content = call["input"][0]["content"]
    image_inputs = [item for item in content if item["type"] == "input_image"]
    assert len(image_inputs) == 2
    assert all(item["detail"] == "high" for item in image_inputs)
    request_text = str(content[0]["text"])
    assert "cylindrical forms" in request_text
    assert "without naming a candidate with an inferred part identity" in request_text
    assert "Do not infer a part identity" in call["instructions"]
    schema = call["text"]["format"]["schema"]
    serialized_schema = json.dumps(schema)
    for forbidden in ("current_state", "desired_state", "CAD", "process", "resource"):
        assert forbidden not in serialized_schema


@pytest.mark.parametrize("identity_term", ["shaft", "pin"])
def test_observation_review_rejects_inferred_part_identity(identity_term: str) -> None:
    expected = (
        ObservationCandidateImage(
            observation_handle="view_0001",
            candidate_handle="candidate_0001_0001",
            source_data_url="data:image/png;base64,source",
            crop_data_url="data:image/png;base64,crop",
        ),
    )

    with pytest.raises(ObservationCandidateReviewError, match="morphology"):
        _validated_output(
            {
                "candidates": [
                    {
                        "observation_handle": "view_0001",
                        "candidate_handle": "candidate_0001_0001",
                        "description": f"visible cylindrical {identity_term}",
                        "uncertainty": "identity not assigned",
                    }
                ]
            },
            expected,
        )


def test_segmentation_is_deterministic_and_does_not_overwrite(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_preprocessing = _preprocess_observation(first_root, _segmentation_bundle())
    second_preprocessing = _preprocess_observation(second_root, _segmentation_bundle())
    first = segment_preprocessed_observation(
        interaction_root=first_root,
        observation_record_path=first_preprocessing.record_path,
    )
    second = segment_preprocessed_observation(
        interaction_root=second_root,
        observation_record_path=second_preprocessing.record_path,
    )

    for camera_id in CAMERA_IDS:
        first_mask = np.load(
            first.record_path.parent / f"{camera_id}_candidate_labels.npy",
            allow_pickle=False,
        )
        second_mask = np.load(
            second.record_path.parent / f"{camera_id}_candidate_labels.npy",
            allow_pickle=False,
        )
        np.testing.assert_array_equal(first_mask, second_mask)
    before = first.record_path.read_bytes()
    with pytest.raises(RGBDSegmentationError, match="already exists"):
        segment_preprocessed_observation(
            interaction_root=first_root,
            observation_record_path=first_preprocessing.record_path,
        )
    assert first.record_path.read_bytes() == before


def test_observation_review_covers_exact_handles_and_pins_candidate_crops(
    tmp_path: Path,
) -> None:
    from cais_spade_llm.spec2primitives.tools.observation_presentation import (
        ObservationPresentation,
    )

    preprocessing = _preprocess_observation(tmp_path, _segmentation_bundle())
    segmentation = segment_preprocessed_observation(
        interaction_root=tmp_path,
        observation_record_path=preprocessing.record_path,
    )
    vision = ControlledObservationVision()

    review = asyncio.run(
        review_observation_candidates(
            interaction_root=tmp_path,
            segmentation_record_path=segmentation.record_path,
            review_number=1,
            config=load_model_runtime_config().observation_vlm,
            vision_runtime=vision,
        )
    )

    expected_handles = [
        (camera["observation_handle"], candidate["candidate_handle"])
        for camera in segmentation.record["cameras"]
        for candidate in camera["candidates"]
    ]
    presentation = ObservationPresentation(tmp_path)
    assert [
        (candidate.observation_handle, candidate.candidate_handle)
        for candidate in vision.requests[0].candidates
    ] == sorted(
        (presentation.handle(view), presentation.handle(candidate))
        for view, candidate in expected_handles
    )
    assert [
        (candidate["observation_handle"], candidate["candidate_handle"])
        for candidate in review.record["candidates"]
    ] == expected_handles
    assert review.record["source_segmentation"] == {
        "ref": segmentation.record_path.relative_to(tmp_path).as_posix(),
        "sha256": _sha256(segmentation.record_path),
    }
    for candidate, crop_path in zip(
        review.record["candidates"],
        review.crop_paths,
        strict=True,
    ):
        assert crop_path.is_file()
        assert candidate["crop_artifact"]["sha256"] == _sha256(crop_path)
        assert set(candidate).isdisjoint(
            {"current_state", "desired_state", "CAD", "process", "resource"}
        )


@pytest.mark.parametrize(
    "forbidden_assignment",
    ["current_state", "desired_state", "CAD", "process", "resource"],
)
def test_observation_review_rejects_assignment_without_partial_output(
    tmp_path: Path,
    forbidden_assignment: str,
) -> None:
    preprocessing = _preprocess_observation(tmp_path, _segmentation_bundle())
    segmentation = segment_preprocessed_observation(
        interaction_root=tmp_path,
        observation_record_path=preprocessing.record_path,
    )

    with pytest.raises(
        ObservationCandidateReviewError,
        match="fields are invalid",
    ):
        asyncio.run(
            review_observation_candidates(
                interaction_root=tmp_path,
                segmentation_record_path=segmentation.record_path,
                review_number=1,
                config=load_model_runtime_config().observation_vlm,
                vision_runtime=ControlledObservationVision(
                    forbidden_assignment=forbidden_assignment
                ),
            )
        )

    grounding_root = tmp_path / "products/grounding/rgb_d_cad_grounding"
    assert not (grounding_root / "observation_review_0001").exists()
    assert list(grounding_root.glob(".observation-review-*")) == []


def test_segmentation_rejects_tampered_point_cloud_without_partial_output(
    tmp_path: Path,
) -> None:
    preprocessing = _preprocess_observation(tmp_path, _segmentation_bundle())
    point_cloud_path = preprocessing.artifact_paths[0]
    tampered = bytearray(point_cloud_path.read_bytes())
    tampered[-1] ^= 1
    point_cloud_path.write_bytes(tampered)

    with pytest.raises(RGBDSegmentationError, match="hash"):
        segment_preprocessed_observation(
            interaction_root=tmp_path,
            observation_record_path=preprocessing.record_path,
        )

    grounding_root = tmp_path / "products/grounding/rgb_d_cad_grounding"
    assert not (grounding_root / "segmentation_0001").exists()
    assert list(grounding_root.glob(".segmentation-*")) == []


def test_segmentation_excludes_support_and_background_when_parts_are_absent(
    tmp_path: Path,
) -> None:
    preprocessing = _preprocess_observation(
        tmp_path,
        _segmentation_bundle(source_objects=False),
    )

    result = segment_preprocessed_observation(
        interaction_root=tmp_path,
        observation_record_path=preprocessing.record_path,
    )

    assert result.candidate_count == 0
    assert all(camera["candidate_count"] == 0 for camera in result.record["cameras"])
    assert all(camera["candidate_state"] == "unresolved" for camera in result.record["cameras"])


def test_segmentation_fails_closed_when_support_plane_is_unavailable(
    tmp_path: Path,
) -> None:
    preprocessing = _preprocess_observation(tmp_path, _plane_unavailable_bundle())

    result = segment_preprocessed_observation(
        interaction_root=tmp_path,
        observation_record_path=preprocessing.record_path,
    )

    assert result.candidate_count == 0
    for camera in result.record["cameras"]:
        assert camera["support_plane"] == {
            "status": "unavailable",
            "candidate_filtering_applied": False,
            "retained_point_count": 0,
        }
        assert camera["candidate_state"] == "unresolved"


def test_segmentation_rejects_mutated_record_without_partial_output(tmp_path: Path) -> None:
    preprocessing = _preprocess_observation(tmp_path, _segmentation_bundle())
    record = _read_json(preprocessing.record_path)
    record["cameras"][0]["point_count"] += 1
    preprocessing.record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RGBDSegmentationError):
        segment_preprocessed_observation(
            interaction_root=tmp_path,
            observation_record_path=preprocessing.record_path,
        )

    grounding_root = tmp_path / "products/grounding/rgb_d_cad_grounding"
    assert not (grounding_root / "segmentation_0001").exists()
    assert list(grounding_root.glob(".segmentation-*")) == []


def test_automatic_pipeline_captures_preprocesses_and_segments_with_fixed_inputs(
    tmp_path: Path,
) -> None:
    capture = ControlledCapture(_segmentation_bundle())

    first = run_automatic_rgbd_segmentation_pipeline(
        contexts_root=tmp_path,
        capture_runtime=capture,
    )
    second = run_automatic_rgbd_segmentation_pipeline(
        contexts_root=tmp_path,
        capture_runtime=capture,
    )

    assert first["status"] == "ready"
    assert second["status"] == "ready"
    assert first["candidate_count"] == 6
    assert first["identity"] == "not_evaluated"
    assert first["CAD_correspondence"] == "not_evaluated"
    assert first["pose"] == "not_evaluated"
    assert first["failure"] is None
    assert capture.status_during_capture == ["running", "running"]
    assert [call[1:] for call in capture.calls] == [
        ("observation_0001", 5.0),
        ("observation_0001", 5.0),
    ]
    roots = [Path(first["interaction_root"]), Path(second["interaction_root"])]
    assert roots[0] != roots[1]
    assert all(root.parent == tmp_path for root in roots)
    for result, root in zip((first, second), roots, strict=True):
        preprocessing_record = _read_json(Path(result["preprocessing_record_path"]))
        assert preprocessing_record["evidence_type"] == "observation"
        assert not (root / "products/grounding/rgb_d_cad_grounding/operation_0002").exists()
        assert _read_json(Path(result["pipeline_record_path"])) == result
    status = read_rgbd_segmentation_status(tmp_path)
    assert status["status"] == "ready"
    assert status["candidate_count"] == 6


def test_automatic_pipeline_records_failure_and_preserves_not_evaluated_states(
    tmp_path: Path,
) -> None:
    result = run_automatic_rgbd_segmentation_pipeline(
        contexts_root=tmp_path,
        capture_runtime=RejectedCapture(),
    )

    assert result["status"] == "failed"
    assert result["failure"]["reason"] == "observation_timeout"
    assert result["preprocessing_record_path"] is None
    assert result["segmentation_record_path"] is None
    assert result["identity"] == "not_evaluated"
    assert result["CAD_correspondence"] == "not_evaluated"
    assert result["pose"] == "not_evaluated"
    status = read_rgbd_segmentation_status(tmp_path)
    assert status["status"] == "failed"
    assert status["failure"]["reason"] == "observation_timeout"


def test_status_reader_fails_closed_for_tampered_status(tmp_path: Path) -> None:
    assert read_rgbd_segmentation_status(tmp_path)["status"] == "idle"
    (tmp_path / "rgbd_segmentation_status.json").write_text(
        json.dumps({"status": "ready", "candidate_count": 99}),
        encoding="utf-8",
    )

    status = read_rgbd_segmentation_status(tmp_path)

    assert status["status"] == "failed"
    assert status["failure"]["reason"] == "status_record_invalid"
    assert status["identity"] == "not_evaluated"
    assert status["pose"] == "not_evaluated"


def test_demo_ui_omits_rgbd_status_card_but_keeps_status_tool() -> None:
    source = Path(spec2primitives_ui.__file__).read_text(encoding="utf-8")

    assert "_render_rgbd_segmentation_status" not in source
    assert "RGB-D Observation Processing" not in source
    assert callable(read_rgbd_segmentation_status)


def test_automatic_pipeline_has_no_operator_processing_parameters() -> None:
    signature = inspect.signature(run_automatic_rgbd_segmentation_pipeline)

    assert tuple(signature.parameters) == ("contexts_root", "capture_runtime")


def _preprocess_observation(interaction_root: Path, bundle: ObservationBundle) -> Any:
    bundle_path = write_observation_bundle(
        interaction_root / "products/observations",
        bundle,
    )
    manifest = _read_json(bundle_path / "manifest.json")
    bundle_ref = bundle_path.relative_to(interaction_root)
    served_context = {
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
    return preprocess_served_geometry(
        interaction_root=interaction_root,
        served_context=served_context,
        operation_number=1,
    )


def _segmentation_bundle(*, source_objects: bool = True) -> ObservationBundle:
    camera_observations = []
    for camera_index, camera_id in enumerate(CAMERA_IDS):
        rgb = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        depth_m = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), np.nan, dtype=np.float32)
        if camera_id == "cam_assembly":
            depth_m[100:180, 100:200] = np.float32(0.9)
            rgb[100:180, 100:200] = np.asarray([80, 90, 100], dtype=np.uint8)
        else:
            depth_m[100:200, 100:200] = np.float32(1.0)
            rgb[100:200, 100:200] = np.asarray([20, 30, 40], dtype=np.uint8)
            if source_objects:
                depth_m[120:130, 120:130] = np.float32(0.8)
                depth_m[120:130, 130:140] = np.float32(0.7)
                rgb[120:130, 120:130] = np.asarray([200, 10, 10], dtype=np.uint8)
                rgb[120:130, 130:140] = np.asarray([10, 200, 10], dtype=np.uint8)
        timestamp_ns = 2_000_000_000 + camera_index * 1_000
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
                camera_calibration=_calibration(frame),
            )
        )
    return ObservationBundle(
        observation_ref="observation_0001",
        evidence_label="live",
        captured_at_ns=2_000_004_000,
        camera_observations=tuple(camera_observations),
    )


def _plane_unavailable_bundle() -> ObservationBundle:
    camera_observations = []
    for camera_index, camera_id in enumerate(CAMERA_IDS):
        rgb = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        depth_m = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), np.nan, dtype=np.float32)
        for point_index in range(80):
            row = 100 + point_index // 10
            column = 100 + point_index % 10
            depth_m[row, column] = np.float32(0.7 + point_index * 0.001)
        timestamp_ns = 2_000_000_000 + camera_index * 1_000
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
                camera_calibration=_calibration(frame),
            )
        )
    return ObservationBundle(
        observation_ref="observation_0001",
        evidence_label="live",
        captured_at_ns=2_000_004_000,
        camera_observations=tuple(camera_observations),
    )


def _calibration(frame: str) -> CameraCalibration:
    return CameraCalibration(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        frame=frame,
        distortion_model="plumb_bob",
        D=(0.0, 0.0, 0.0, 0.0, 0.0),
        K=(100.0, 0.0, 320.0, 0.0, 100.0, 240.0, 0.0, 0.0, 1.0),
        R=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        P=(100.0, 0.0, 320.0, 0.0, 0.0, 100.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

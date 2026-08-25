"""Tests for offline Spec2Primitives RGB-D observation bundles."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cais_spade_llm.spec2primitives.tools import observation_context
from cais_spade_llm.spec2primitives.tools.observation_context import (
    CAMERA_IDS,
    CameraCalibration,
    CameraObservation,
    ObservationBundle,
    ObservationContextError,
    read_observation_bundle,
    write_observation_bundle,
)


@pytest.fixture
def observation_bundle() -> ObservationBundle:
    camera_observations = []
    for camera_index, camera_id in enumerate(CAMERA_IDS):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[..., 0] = 20 + camera_index
        rgb[..., 1] = 40 + camera_index
        rgb[..., 2] = 60 + camera_index

        depth_m = np.full(
            (480, 640),
            0.5 + camera_index * 0.1,
            dtype=np.float32,
        )
        depth_m[0, 0] = np.nan
        depth_m[0, 1] = np.inf
        depth_m[0, 2] = 0.0

        rgb_timestamp_ns = 1_000_000_000 + camera_index * 20_000_000
        camera_observations.append(
            CameraObservation(
                camera_id=camera_id,
                rgb=rgb,
                depth_m=depth_m,
                rgb_timestamp_ns=rgb_timestamp_ns,
                depth_timestamp_ns=rgb_timestamp_ns + 10_000_000,
                rgb_frame=f"{camera_id}_rgb_optical_frame",
                depth_frame=f"{camera_id}_depth_optical_frame",
                camera_calibration=_calibration(camera_id),
            )
        )

    return ObservationBundle(
        observation_ref="observation_0001",
        evidence_label="fixture",
        captured_at_ns=1_100_000_000,
        camera_observations=tuple(camera_observations),
    )


@pytest.mark.parametrize("evidence_label", ["fixture", "replay"])
def test_bundle_round_trip_preserves_rgb_depth_and_manifest(
    observation_bundle: ObservationBundle,
    tmp_path: Path,
    evidence_label: str,
) -> None:
    source_bundle = replace(observation_bundle, evidence_label=evidence_label)
    bundle_path = write_observation_bundle(tmp_path, source_bundle)
    loaded_bundle = read_observation_bundle(bundle_path)

    assert loaded_bundle.observation_ref == source_bundle.observation_ref
    assert loaded_bundle.evidence_label == evidence_label
    assert loaded_bundle.captured_at_ns == source_bundle.captured_at_ns
    assert loaded_bundle.bundle_skew_ns == source_bundle.bundle_skew_ns
    assert tuple(
        observation.camera_id for observation in loaded_bundle.camera_observations
    ) == CAMERA_IDS

    source_by_camera = {
        observation.camera_id: observation
        for observation in source_bundle.camera_observations
    }
    for loaded in loaded_bundle.camera_observations:
        source = source_by_camera[loaded.camera_id]
        assert np.array_equal(loaded.rgb, source.rgb)
        np.testing.assert_array_equal(loaded.depth_m, source.depth_m)
        assert loaded.rgb_timestamp_ns == source.rgb_timestamp_ns
        assert loaded.depth_timestamp_ns == source.depth_timestamp_ns
        assert loaded.timestamp_difference_ns == source.timestamp_difference_ns
        assert loaded.rgb_frame == source.rgb_frame
        assert loaded.depth_frame == source.depth_frame
        assert loaded.depth_registered_to_rgb is True
        assert loaded.valid_depth_fraction == source.valid_depth_fraction
        assert loaded.camera_calibration == source.camera_calibration

    manifest = json.loads(
        (bundle_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["evidence_label"] == evidence_label
    assert manifest["bundle_skew_ns"] == source_bundle.bundle_skew_ns
    for camera in manifest["cameras"]:
        camera_id = camera["camera_id"]
        assert camera["rgb_artifact"] == f"{camera_id}_rgb.png"
        assert camera["depth_artifact"] == f"{camera_id}_depth_m.npy"
        assert not Path(camera["rgb_artifact"]).is_absolute()
        assert not Path(camera["depth_artifact"]).is_absolute()
        assert camera["camera_calibration"]["extrinsics_available"] is False


def test_missing_duplicate_and_unknown_cameras_are_rejected(
    observation_bundle: ObservationBundle,
    tmp_path: Path,
) -> None:
    observations = observation_bundle.camera_observations
    invalid_camera_sets = (
        observations[:-1],
        (*observations[:-1], observations[0]),
        (*observations[:-1], replace(observations[-1], camera_id="cam_unknown")),
    )

    for camera_observations in invalid_camera_sets:
        invalid_bundle = replace(
            observation_bundle,
            camera_observations=tuple(camera_observations),
        )
        with pytest.raises(ObservationContextError):
            write_observation_bundle(tmp_path, invalid_bundle)
        assert not (tmp_path / invalid_bundle.observation_ref).exists()


def test_invalid_rgb_depth_shapes_dtypes_and_units_are_rejected(
    observation_bundle: ObservationBundle,
    tmp_path: Path,
) -> None:
    observation = observation_bundle.camera_observations[0]
    invalid_observations = (
        replace(observation, rgb=np.zeros((10, 10, 3), dtype=np.uint8)),
        replace(observation, rgb=observation.rgb.astype(np.float32)),
        replace(observation, depth_m=np.ones((10, 10), dtype=np.float32)),
        replace(observation, depth_m=observation.depth_m.astype(np.float64)),
        replace(observation, rgb_encoding="bgr8"),
        replace(observation, depth_encoding="16UC1"),
        replace(observation, depth_units="mm"),
    )

    for invalid_observation in invalid_observations:
        with pytest.raises(ObservationContextError):
            write_observation_bundle(
                tmp_path,
                _replace_camera(observation_bundle, 0, invalid_observation),
            )


def test_missing_malformed_or_extrinsic_calibration_is_rejected(
    observation_bundle: ObservationBundle,
    tmp_path: Path,
) -> None:
    observation = observation_bundle.camera_observations[0]
    calibration = observation.camera_calibration
    invalid_observations = (
        replace(observation, camera_calibration=None),
        replace(observation, camera_calibration=replace(calibration, K=(1.0,))),
        replace(
            observation,
            camera_calibration=replace(calibration, extrinsics_available=True),
        ),
    )

    for invalid_observation in invalid_observations:
        with pytest.raises(ObservationContextError):
            write_observation_bundle(
                tmp_path,
                _replace_camera(observation_bundle, 0, invalid_observation),
            )


def test_unregistered_empty_or_unsynchronized_depth_is_rejected(
    observation_bundle: ObservationBundle,
    tmp_path: Path,
) -> None:
    observation = observation_bundle.camera_observations[0]
    empty_depth = np.full((480, 640), np.nan, dtype=np.float32)
    invalid_observations = (
        replace(observation, depth_registered_to_rgb=False),
        replace(observation, depth_m=empty_depth),
        replace(
            observation,
            depth_timestamp_ns=(
                observation.rgb_timestamp_ns
                + observation_context.MAX_RGB_DEPTH_SKEW_NS
                + 1
            ),
        ),
    )

    for invalid_observation in invalid_observations:
        with pytest.raises(ObservationContextError):
            write_observation_bundle(
                tmp_path,
                _replace_camera(observation_bundle, 0, invalid_observation),
            )


def test_excessive_cross_camera_bundle_skew_is_rejected(
    observation_bundle: ObservationBundle,
    tmp_path: Path,
) -> None:
    last = observation_bundle.camera_observations[-1]
    late_rgb_timestamp_ns = (
        observation_bundle.camera_observations[0].rgb_timestamp_ns
        + observation_context.MAX_BUNDLE_SKEW_NS
        + 1
    )
    late = replace(
        last,
        rgb_timestamp_ns=late_rgb_timestamp_ns,
        depth_timestamp_ns=late_rgb_timestamp_ns + 10_000_000,
    )

    with pytest.raises(ObservationContextError):
        write_observation_bundle(
            tmp_path,
            _replace_camera(observation_bundle, len(CAMERA_IDS) - 1, late),
        )


@pytest.mark.parametrize(
    "observation_ref",
    [
        "",
        "observation_",
        "observation_..",
        "../observation_0001",
        "/tmp/observation_0001",
        "observation_0001/escape",
        "observation_0001\\escape",
        "observation 0001",
    ],
)
def test_unsafe_observation_refs_are_rejected(
    observation_bundle: ObservationBundle,
    tmp_path: Path,
    observation_ref: str,
) -> None:
    with pytest.raises(ObservationContextError):
        write_observation_bundle(
            tmp_path,
            replace(observation_bundle, observation_ref=observation_ref),
        )


def test_existing_bundle_is_not_overwritten(
    observation_bundle: ObservationBundle,
    tmp_path: Path,
) -> None:
    bundle_path = write_observation_bundle(tmp_path, observation_bundle)
    original_manifest = (bundle_path / "manifest.json").read_bytes()

    with pytest.raises(ObservationContextError):
        write_observation_bundle(tmp_path, observation_bundle)

    assert (bundle_path / "manifest.json").read_bytes() == original_manifest


def test_failed_artifact_write_leaves_no_partial_bundle(
    observation_bundle: ObservationBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observation_context.cv2, "imwrite", lambda *_args: False)

    with pytest.raises(ObservationContextError):
        write_observation_bundle(tmp_path, observation_bundle)

    assert not (tmp_path / observation_bundle.observation_ref).exists()
    assert list(tmp_path.iterdir()) == []


def test_manifest_artifact_path_cannot_escape_bundle(
    observation_bundle: ObservationBundle,
    tmp_path: Path,
) -> None:
    bundle_path = write_observation_bundle(tmp_path, observation_bundle)
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cameras"][0]["rgb_artifact"] = "../outside.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ObservationContextError):
        read_observation_bundle(bundle_path)


def test_production_module_has_no_live_or_agent_dependencies() -> None:
    source = Path(observation_context.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "rclpy",
        "sensor_msgs",
        "gazebo_msgs",
        "gazebo_camera_detector",
        "VLM",
        "ProductAgent",
        "RobotAgent",
        "nicegui",
        "SystemBridge",
        "segmentation",
        "recognition",
        "grounding",
        "execution",
    ):
        assert forbidden not in source


def _calibration(camera_id: str) -> CameraCalibration:
    return CameraCalibration(
        width=640,
        height=480,
        frame=f"{camera_id}_rgb_optical_frame",
        distortion_model="plumb_bob",
        D=(0.0, 0.0, 0.0, 0.0, 0.0),
        K=(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0),
        R=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        P=(
            500.0,
            0.0,
            320.0,
            0.0,
            0.0,
            500.0,
            240.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ),
    )


def _replace_camera(
    observation_bundle: ObservationBundle,
    camera_index: int,
    replacement: CameraObservation,
) -> ObservationBundle:
    camera_observations = list(observation_bundle.camera_observations)
    camera_observations[camera_index] = replacement
    return replace(
        observation_bundle,
        camera_observations=tuple(camera_observations),
    )

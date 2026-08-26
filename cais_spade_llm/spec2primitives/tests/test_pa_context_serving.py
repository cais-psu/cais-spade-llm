"""Tests for serving the first context requested by ProductAgent."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cais_spade_llm.spec2primitives.agents.pa import (
    context_serving,
    serve_pa_requested_context,
)
from cais_spade_llm.spec2primitives.tools.observation_context import (
    CAMERA_IDS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    CameraCalibration,
    CameraObservation,
    ObservationBundle,
    ObservationContextError,
    write_observation_bundle,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.gazebo_observation_provider import (
    GazeboObservationProviderError,
)


@pytest.mark.parametrize(
    ("context_ref", "evidence_type", "evidence_key"),
    [
        ("NIST_assembly_instructions.pdf", "document", "document_evidence"),
        ("Gear_Medium.STL", "CAD", "CAD_evidence"),
    ],
)
def test_exact_static_context_is_served_once_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    context_ref: str,
    evidence_type: str,
    evidence_key: str,
) -> None:
    product_requirement = "  assemble Medium Gear exactly  "
    needed_context = _needed_context(context_ref=context_ref)
    _write_phase_3_1_records(tmp_path, product_requirement, needed_context)
    resolver_calls: list[dict[str, object]] = []
    real_resolver = context_serving.resolve_context_ref

    def tracking_resolver(request: dict[str, object]) -> dict[str, object]:
        resolver_calls.append(request)
        return real_resolver(request)

    def unexpected_capture(*args: object, **kwargs: object) -> Path:
        raise AssertionError("Live observation capture must not run.")

    monkeypatch.setattr(context_serving, "resolve_context_ref", tracking_resolver)
    monkeypatch.setattr(
        context_serving,
        "capture_gazebo_observation",
        unexpected_capture,
    )

    result = serve_pa_requested_context(tmp_path)

    assert resolver_calls == [{"context_ref": context_ref}]
    served_context = result["served_context"]
    assert served_context["context_ref"] == context_ref
    assert served_context["evidence_type"] == evidence_type
    assert evidence_key in served_context
    assert served_context["provenance"] == {
        "repository_path": (
            "cais_spade_llm/spec2primitives/references/products/NIST_assembly_instructions.pdf"
            if evidence_type == "document"
            else "ros2/cais_lab_robotics/cad_models/Gear_Medium.STL"
        ),
        "source_url": (
            "https://www.nist.gov/el/intelligent-systems-division-73500/"
            "robotic-grasping-and-manipulation-assembly/assembly"
        ),
    }

    served_reference = _read_json(tmp_path / "products/served_references" / f"{context_ref}.json")
    assert served_reference == result
    retrieval_record = _read_json(tmp_path / "interaction_record/retrieval_0001.json")
    assert retrieval_record == {
        "retrieval": 1,
        "product_requirement": product_requirement,
        "needed_context": needed_context,
        "context_request": {"context_ref": context_ref},
        "served_context": served_context,
        "retrieval_error": None,
        "failure": None,
    }


def test_live_observation_is_captured_once_and_served_as_references(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    product_requirement = "assemble Medium Gear"
    needed_context = _needed_context(request_live_observation=True)
    _write_phase_3_1_records(tmp_path, product_requirement, needed_context)
    capture_calls: list[dict[str, object]] = []

    def controlled_capture(
        observations_root: Path,
        observation_ref: str,
        *,
        timeout_sec: float,
    ) -> Path:
        capture_calls.append(
            {
                "observations_root": observations_root,
                "observation_ref": observation_ref,
                "timeout_sec": timeout_sec,
            }
        )
        return write_observation_bundle(
            observations_root,
            _live_observation_bundle(observation_ref),
        )

    def unexpected_resolver(request: dict[str, object]) -> dict[str, object]:
        raise AssertionError(f"Static resolver must not run: {request}")

    monkeypatch.setattr(
        context_serving,
        "capture_gazebo_observation",
        controlled_capture,
    )
    monkeypatch.setattr(context_serving, "resolve_context_ref", unexpected_resolver)

    result = serve_pa_requested_context(
        tmp_path,
        live_observation_timeout_sec=7.5,
    )

    observations_root = tmp_path / "products/observations"
    assert capture_calls == [
        {
            "observations_root": observations_root,
            "observation_ref": "observation_0001",
            "timeout_sec": 7.5,
        }
    ]
    served_context = result["served_context"]
    assert served_context["context_ref"] is None
    assert served_context["observation_ref"] == "observation_0001"
    assert served_context["evidence_type"] == "observation"
    assert served_context["evidence_label"] == "live"
    assert served_context["provenance"] == {
        "manifest_path": ("products/observations/observation_0001/manifest.json")
    }
    observation_evidence = served_context["observation_evidence"]
    assert observation_evidence["manifest"]["evidence_label"] == "live"
    assert [
        reference["camera_id"] for reference in observation_evidence["artifact_references"]
    ] == list(CAMERA_IDS)
    for reference in observation_evidence["artifact_references"]:
        assert reference["rgb_artifact"].startswith("products/observations/observation_0001/")
        assert reference["depth_artifact"].startswith("products/observations/observation_0001/")
    json.dumps(result, allow_nan=False)

    retrieval_record = _read_json(tmp_path / "interaction_record/retrieval_0001.json")
    assert retrieval_record["product_requirement"] == product_requirement
    assert retrieval_record["needed_context"] == needed_context
    assert retrieval_record["context_request"] == {
        "request_live_observation": True,
        "observation_ref": "observation_0001",
        "live_observation_timeout_sec": 7.5,
    }
    assert retrieval_record["served_context"] == served_context
    assert retrieval_record["retrieval_error"] is None
    assert retrieval_record["failure"] is None
    assert not (tmp_path / "products/served_references").exists()


def test_clarification_decision_returns_context_not_requested_without_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    needed_context = _needed_context(
        clarification_question="Which Medium Gear should be assembled?"
    )
    _write_phase_3_1_records(tmp_path, "assemble Medium Gear", needed_context)
    monkeypatch.setattr(
        context_serving,
        "resolve_context_ref",
        lambda request: pytest.fail(f"Unexpected resolver call: {request}"),
    )
    monkeypatch.setattr(
        context_serving,
        "capture_gazebo_observation",
        lambda *args, **kwargs: pytest.fail("Unexpected capture call"),
    )

    result = serve_pa_requested_context(tmp_path)

    assert result["failure"]["reason"] == "context_not_requested"
    retrieval_record = _read_json(tmp_path / "interaction_record/retrieval_0001.json")
    assert retrieval_record["context_request"] == {
        "clarification_question": "Which Medium Gear should be assembled?",
    }
    assert retrieval_record["served_context"] is None
    assert retrieval_record["retrieval_error"] is None
    assert retrieval_record["failure"] == result["failure"]


@pytest.mark.parametrize(
    "invalid_case",
    [
        "missing_records",
        "malformed_requirement",
        "failed_turn",
        "mismatched_requirement",
        "mixed_decision",
    ],
)
def test_invalid_phase_3_1_interactions_are_rejected_without_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid_case: str,
) -> None:
    if invalid_case != "missing_records":
        needed_context = _needed_context(context_ref="Gear_Medium.STL")
        _write_phase_3_1_records(tmp_path, "assemble Medium Gear", needed_context)
    if invalid_case == "malformed_requirement":
        _write_json(
            tmp_path / "products/user_requirement/product_requirement.json",
            {"product_requirement": "   "},
        )
    elif invalid_case == "failed_turn":
        turn_path = tmp_path / "interaction_record/turn_0001.json"
        turn = _read_json(turn_path)
        turn["failure"] = {"reason": "pa_call_failed", "message": "controlled"}
        _write_json(turn_path, turn)
    elif invalid_case == "mismatched_requirement":
        turn_path = tmp_path / "interaction_record/turn_0001.json"
        turn = _read_json(turn_path)
        turn["product_requirement"] = "different requirement"
        _write_json(turn_path, turn)
    elif invalid_case == "mixed_decision":
        turn_path = tmp_path / "interaction_record/turn_0001.json"
        turn = _read_json(turn_path)
        turn["PA_output"]["needed_context"]["request_live_observation"] = True
        _write_json(turn_path, turn)

    monkeypatch.setattr(
        context_serving,
        "resolve_context_ref",
        lambda request: pytest.fail(f"Unexpected resolver call: {request}"),
    )
    monkeypatch.setattr(
        context_serving,
        "capture_gazebo_observation",
        lambda *args, **kwargs: pytest.fail("Unexpected capture call"),
    )

    result = serve_pa_requested_context(tmp_path)

    assert result["failure"]["reason"] == "invalid_interaction"
    assert not (tmp_path / "interaction_record/retrieval_0001.json").exists()


def test_resolver_rejection_is_preserved_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context_ref = "Gear_Medium.STL"
    needed_context = _needed_context(context_ref=context_ref)
    _write_phase_3_1_records(tmp_path, "assemble Medium Gear", needed_context)
    calls: list[dict[str, object]] = []

    def rejected_resolver(request: dict[str, object]) -> dict[str, object]:
        calls.append(request)
        return {
            "rejection": {
                "context_ref": context_ref,
                "reason": "missing_source",
                "message": "The approved source file is missing.",
            }
        }

    monkeypatch.setattr(context_serving, "resolve_context_ref", rejected_resolver)
    monkeypatch.setattr(
        context_serving,
        "capture_gazebo_observation",
        lambda *args, **kwargs: pytest.fail("Fallback capture must not run"),
    )

    result = serve_pa_requested_context(tmp_path)

    assert calls == [{"context_ref": context_ref}]
    assert result["failure"]["reason"] == "retrieval_failed"
    assert result["retrieval_error"] == {
        "context_ref": context_ref,
        "observation_ref": None,
        "reason": "missing_source",
        "message": "The approved source file is missing.",
    }
    retrieval_record = _read_json(tmp_path / "interaction_record/retrieval_0001.json")
    assert retrieval_record["served_context"] is None
    assert retrieval_record["retrieval_error"] == result["retrieval_error"]
    assert retrieval_record["failure"] == result["failure"]
    assert not (tmp_path / "products/served_references").exists()


def test_malformed_resolver_response_is_rejected_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_phase_3_1_records(
        tmp_path,
        "assemble Medium Gear",
        _needed_context(context_ref="Gear_Medium.STL"),
    )
    monkeypatch.setattr(
        context_serving,
        "resolve_context_ref",
        lambda request: {"served_context": {"context_ref": request["context_ref"]}},
    )

    result = serve_pa_requested_context(tmp_path)

    assert result["failure"]["reason"] == "retrieval_failed"
    assert result["retrieval_error"]["reason"] == "invalid_resolver_response"
    assert not (tmp_path / "products/served_references").exists()


@pytest.mark.parametrize(
    ("capture_error", "expected_reason"),
    [
        (
            GazeboObservationProviderError(
                "capture_timeout",
                "Controlled complete capture timeout.",
            ),
            "capture_timeout",
        ),
        (
            ObservationContextError("Controlled observation storage failure."),
            "observation_storage_failed",
        ),
        (RuntimeError("Controlled capture failure."), "capture_failed"),
    ],
)
def test_live_capture_failures_are_preserved_without_static_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capture_error: Exception,
    expected_reason: str,
) -> None:
    _write_phase_3_1_records(
        tmp_path,
        "assemble Medium Gear",
        _needed_context(request_live_observation=True),
    )
    calls: list[tuple[Path, str, float]] = []

    def failing_capture(
        observations_root: Path,
        observation_ref: str,
        *,
        timeout_sec: float,
    ) -> Path:
        calls.append((observations_root, observation_ref, timeout_sec))
        raise capture_error

    monkeypatch.setattr(
        context_serving,
        "capture_gazebo_observation",
        failing_capture,
    )
    monkeypatch.setattr(
        context_serving,
        "resolve_context_ref",
        lambda request: pytest.fail(f"Fallback resolver must not run: {request}"),
    )

    result = serve_pa_requested_context(tmp_path)

    assert calls == [(tmp_path / "products/observations", "observation_0001", 5.0)]
    assert result["failure"]["reason"] == "retrieval_failed"
    assert result["retrieval_error"]["context_ref"] is None
    assert result["retrieval_error"]["observation_ref"] == "observation_0001"
    assert result["retrieval_error"]["reason"] == expected_reason
    retrieval_record = _read_json(tmp_path / "interaction_record/retrieval_0001.json")
    assert retrieval_record["retrieval_error"] == result["retrieval_error"]


def test_malformed_live_manifest_is_rejected_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_phase_3_1_records(
        tmp_path,
        "assemble Medium Gear",
        _needed_context(request_live_observation=True),
    )

    def malformed_capture(
        observations_root: Path,
        observation_ref: str,
        *,
        timeout_sec: float,
    ) -> Path:
        del timeout_sec
        bundle_path = observations_root / observation_ref
        bundle_path.mkdir(parents=True)
        _write_json(bundle_path / "manifest.json", {"unexpected": True})
        return bundle_path

    monkeypatch.setattr(
        context_serving,
        "capture_gazebo_observation",
        malformed_capture,
    )

    result = serve_pa_requested_context(tmp_path)

    assert result["failure"]["reason"] == "retrieval_failed"
    assert result["retrieval_error"]["reason"] == "invalid_observation_bundle"
    retrieval_record = _read_json(tmp_path / "interaction_record/retrieval_0001.json")
    assert retrieval_record["served_context"] is None
    assert retrieval_record["retrieval_error"] == result["retrieval_error"]


@pytest.mark.parametrize(
    ("needed_context", "existing_relative_path"),
    [
        (
            {
                "context_ref": "Gear_Medium.STL",
                "request_live_observation": False,
                "clarification_question": None,
            },
            "interaction_record/retrieval_0001.json",
        ),
        (
            {
                "context_ref": "Gear_Medium.STL",
                "request_live_observation": False,
                "clarification_question": None,
            },
            "products/served_references/Gear_Medium.STL.json",
        ),
        (
            {
                "context_ref": None,
                "request_live_observation": True,
                "clarification_question": None,
            },
            "products/observations/observation_0001",
        ),
    ],
)
def test_existing_phase_3_2_records_are_never_overwritten_or_retrieved_again(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    needed_context: dict[str, object],
    existing_relative_path: str,
) -> None:
    _write_phase_3_1_records(
        tmp_path,
        "assemble Medium Gear",
        needed_context,
    )
    existing_path = tmp_path / existing_relative_path
    if existing_path.suffix:
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text("controlled existing record", encoding="utf-8")
    else:
        existing_path.mkdir(parents=True)
    monkeypatch.setattr(
        context_serving,
        "resolve_context_ref",
        lambda request: pytest.fail(f"Unexpected resolver call: {request}"),
    )
    monkeypatch.setattr(
        context_serving,
        "capture_gazebo_observation",
        lambda *args, **kwargs: pytest.fail("Unexpected capture call"),
    )

    result = serve_pa_requested_context(tmp_path)

    assert result["failure"]["reason"] == "interaction_exists"
    if existing_path.is_file():
        assert existing_path.read_text(encoding="utf-8") == ("controlled existing record")


def test_phase_3_2_has_only_the_approved_dependencies_and_no_pa_turn() -> None:
    source_path = Path(context_serving.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    imported_modules = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module
    }
    imported_modules.update(
        alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    )

    assert not any(
        forbidden in module
        for module in imported_modules
        for forbidden in (
            "intelligent_product",
            "llm_agent",
            "document_evidence",
            ".ui",
            ".agents.ra",
        )
    )
    for forbidden_source in (
        "ask_llm",
        ".setup(",
        "/gazebo/model_states",
        "/get_entity_state",
        "table_spec2primitives.world",
        "ground_truth",
        "primitive_steps",
    ):
        assert forbidden_source not in source
    assert serve_pa_requested_context is context_serving.serve_pa_requested_context


def _needed_context(
    *,
    context_ref: str | None = None,
    request_live_observation: bool = False,
    clarification_question: str | None = None,
) -> dict[str, object]:
    return {
        "context_ref": context_ref,
        "request_live_observation": request_live_observation,
        "clarification_question": clarification_question,
    }


def _write_phase_3_1_records(
    interaction_root: Path,
    product_requirement: str,
    needed_context: dict[str, object],
) -> None:
    _write_json(
        interaction_root / "products/user_requirement/product_requirement.json",
        {"product_requirement": product_requirement},
    )
    _write_json(
        interaction_root / "interaction_record/turn_0001.json",
        {
            "turn": 1,
            "product_requirement": product_requirement,
            "PA_input": {
                "prompt": "controlled Phase 3.1 prompt",
                "response_format": {"strict": True},
            },
            "PA_output": {"needed_context": needed_context},
            "failure": None,
        },
    )


def _live_observation_bundle(observation_ref: str) -> ObservationBundle:
    camera_observations = []
    for camera_index, camera_id in enumerate(CAMERA_IDS):
        frame = f"{camera_id}_frame"
        calibration = CameraCalibration(
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            frame=frame,
            distortion_model="plumb_bob",
            D=(0.0,),
            K=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            R=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            P=(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        )
        timestamp_ns = 1_000_000_000 + camera_index
        camera_observations.append(
            CameraObservation(
                camera_id=camera_id,
                rgb=np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8),
                depth_m=np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32),
                rgb_timestamp_ns=timestamp_ns,
                depth_timestamp_ns=timestamp_ns,
                rgb_frame=frame,
                depth_frame=frame,
                camera_calibration=calibration,
            )
        )
    return ObservationBundle(
        observation_ref=observation_ref,
        evidence_label="live",
        captured_at_ns=1_000_000_003,
        camera_observations=tuple(camera_observations),
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

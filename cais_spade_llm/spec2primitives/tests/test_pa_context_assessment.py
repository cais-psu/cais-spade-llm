"""Tests for ontology-backed Spec2Primitives PA context orchestration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cais_spade_llm.spec2primitives.agents.pa import (
    context_assessment,
    context_serving,
)
from cais_spade_llm.spec2primitives.agents.pa.context_assessment import (
    continue_pa_context_interaction,
)
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    start_pa_context_interaction,
)
from cais_spade_llm.spec2primitives.agents.pa.context_serving import (
    serve_pa_requested_context,
)
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    PPR_NAMESPACE,
    ControlledGroundingRuntime,
    complete_context,
    ontology_config,
    request_clarification,
    request_context,
    request_live_observation,
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


class FakeProductAgent:
    """Return controlled Phase 3.1 responses without lifecycle behavior."""

    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append({"prompt": prompt, "response_format": response_format})
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)


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


def test_medium_gear_sequence_interprets_merges_and_completes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    product_requirement = "  assemble Medium Gear exactly  "
    assessments = [
        request_context("Gear_Medium.STL", semantic_need="requested product identity"),
        request_context("Gear_Plate.STL", semantic_need="receiving product"),
        request_context("Gear_Shaft.STL", semantic_need="receiving feature"),
        request_context(
            "GMC_Laser_Plate_Virtual.STL",
            semantic_need="candidate correspondence",
        ),
        request_live_observation(semantic_need="current arrangement"),
        complete_context(),
    ]
    grounding = ControlledGroundingRuntime(assessments=assessments)
    product_agent = _initialize_static(
        tmp_path,
        product_requirement,
        "NIST_assembly_instructions.pdf",
        grounding,
    )
    resolver_calls: list[dict[str, object]] = []
    capture_calls: list[tuple[str, float]] = []
    real_resolver = context_serving.resolve_context_ref

    def tracking_resolver(request: dict[str, object]) -> dict[str, object]:
        resolver_calls.append(request)
        return real_resolver(request)

    def controlled_capture(
        observations_root: Path,
        observation_ref: str,
        *,
        timeout_sec: float,
    ) -> Path:
        capture_calls.append((observation_ref, timeout_sec))
        return write_observation_bundle(
            observations_root,
            _live_observation_bundle(observation_ref),
        )

    monkeypatch.setattr(context_serving, "resolve_context_ref", tracking_resolver)
    monkeypatch.setattr(
        context_serving,
        "capture_gazebo_observation",
        controlled_capture,
    )

    result = _continue(
        product_agent,
        tmp_path,
        grounding,
        max_pa_turns=12,
        live_observation_timeout_sec=7.5,
    )

    assert result == _terminal_complete()
    assert resolver_calls == [
        {"context_ref": context_ref}
        for context_ref in (
            "Gear_Medium.STL",
            "Gear_Plate.STL",
            "Gear_Shaft.STL",
            "GMC_Laser_Plate_Virtual.STL",
        )
    ]
    assert capture_calls == [("observation_0001", 7.5)]
    assert len(grounding.interpretation_calls) == 6
    assert len(grounding.assessment_calls) == 6
    assert [call["delta_count"] for call in grounding.assessment_calls] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert _read_json(tmp_path / "interaction_record/pa_context_settings.json") == {
        "max_pa_turns": 12,
        "live_observation_timeout_sec": 7.5,
    }
    for number in range(1, 7):
        assert (tmp_path / f"interaction_record/interpretation_{number:04d}.json").is_file()
        assert (tmp_path / f"interaction_record/decision_{number:04d}.json").is_file()
        assert (tmp_path / f"products/grounding/ontology/delta_{number:04d}.json").is_file()
    manifest = _read_json(tmp_path / "products/grounding/ontology/abox_manifest.json")
    assert manifest["delta_count"] == 6
    assert manifest["accepted_assertion_count"] == 6
    final_decision = _read_json(tmp_path / "interaction_record/decision_0006.json")
    assert final_decision["Phase_4_3_output"] == complete_context()
    assert "served_context" not in json.dumps(final_decision["Phase_4_3_input"])


def test_alternative_source_order_is_preserved(tmp_path: Path) -> None:
    grounding = ControlledGroundingRuntime(
        assessments=[
            request_context("Gear_Plate.STL", semantic_need="receiving product"),
            request_context("Gear_Medium.STL", semantic_need="requested product"),
            complete_context(),
        ]
    )
    product_agent = _initialize_static(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
        grounding,
    )

    result = _continue(product_agent, tmp_path, grounding, max_pa_turns=4)

    assert result == _terminal_complete()
    assert _read_json(tmp_path / "interaction_record/retrieval_0002.json")["context_request"] == {
        "context_ref": "Gear_Plate.STL"
    }
    assert _read_json(tmp_path / "interaction_record/retrieval_0003.json")["context_request"] == {
        "context_ref": "Gear_Medium.STL"
    }


def test_live_observation_uses_next_number_and_new_semantic_need(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture_calls: list[str] = []

    def controlled_capture(
        observations_root: Path,
        observation_ref: str,
        *,
        timeout_sec: float,
    ) -> Path:
        assert timeout_sec == 5.0
        capture_calls.append(observation_ref)
        return write_observation_bundle(
            observations_root,
            _live_observation_bundle(observation_ref),
        )

    monkeypatch.setattr(
        context_serving,
        "capture_gazebo_observation",
        controlled_capture,
    )
    grounding = ControlledGroundingRuntime(
        assessments=[
            request_live_observation(semantic_need="refresh occluded feature"),
            complete_context(),
        ]
    )
    product_agent = _initialize_live(tmp_path, grounding)

    result = _continue(product_agent, tmp_path, grounding, max_pa_turns=3)

    assert result == _terminal_complete()
    assert capture_calls == ["observation_0001", "observation_0002"]
    assert (
        _read_json(tmp_path / "interaction_record/retrieval_0002.json")["served_context"][
            "observation_ref"
        ]
        == "observation_0002"
    )


def test_repeated_live_observation_requires_new_semantic_need(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def controlled_capture(
        observations_root: Path,
        observation_ref: str,
        *,
        timeout_sec: float,
    ) -> Path:
        assert timeout_sec == 5.0
        return write_observation_bundle(
            observations_root,
            _live_observation_bundle(observation_ref),
        )

    monkeypatch.setattr(
        context_serving,
        "capture_gazebo_observation",
        controlled_capture,
    )
    repeated_request = request_live_observation(semantic_need="same unresolved scene need")
    grounding = ControlledGroundingRuntime(assessments=[repeated_request, repeated_request])
    product_agent = _initialize_live(tmp_path, grounding)

    result = _continue(product_agent, tmp_path, grounding, max_pa_turns=4)

    assert result["failure"]["reason"] == "invalid_assessment"
    assert "new semantic need" in result["failure"]["message"]
    assert (tmp_path / "interaction_record/retrieval_0002.json").is_file()
    assert not (tmp_path / "interaction_record/retrieval_0003.json").exists()


@pytest.mark.parametrize("max_pa_turns", [True, 1, 1.5])
def test_invalid_max_pa_turns_is_rejected_without_settings(
    tmp_path: Path,
    max_pa_turns: object,
) -> None:
    grounding = ControlledGroundingRuntime(assessments=[complete_context()])
    product_agent = _initialize_static(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
        grounding,
    )

    result = _continue(
        product_agent,
        tmp_path,
        grounding,
        max_pa_turns=max_pa_turns,
    )

    assert result["failure"]["reason"] == "invalid_max_pa_turns"
    assert not (tmp_path / "interaction_record/pa_context_settings.json").exists()
    assert grounding.interpretation_calls == []


def test_default_limit_and_persisted_completion(tmp_path: Path) -> None:
    grounding = ControlledGroundingRuntime(assessments=[complete_context()])
    product_agent = _initialize_static(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
        grounding,
    )

    result = _continue(product_agent, tmp_path, grounding)

    assert result == _terminal_complete()
    assert (
        _read_json(tmp_path / "interaction_record/pa_context_settings.json")["max_pa_turns"] == 12
    )
    decision = _read_json(tmp_path / "interaction_record/decision_0001.json")
    assert decision["failure"] is None
    assert decision["Phase_4_3_output"] == complete_context()


def test_turn_limit_records_request_without_serving_it(tmp_path: Path) -> None:
    grounding = ControlledGroundingRuntime(assessments=[request_context("Gear_Medium.STL")])
    product_agent = _initialize_static(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
        grounding,
    )

    result = _continue(product_agent, tmp_path, grounding, max_pa_turns=2)

    assert result["failure"]["reason"] == "pa_turn_limit_reached"
    assert (
        _read_json(tmp_path / "interaction_record/turn_0002.json")["failure"] == (result["failure"])
    )
    assert (tmp_path / "interaction_record/decision_0001.json").is_file()
    assert not (tmp_path / "interaction_record/retrieval_0002.json").exists()


@pytest.mark.parametrize(
    "assessment",
    [
        {"needed_context": None, "context understanding complete": True},
        {
            "unresolved_semantic_need": None,
            "needed_context": _needed_context(context_ref="Gear_Medium.STL"),
            "context understanding complete": False,
        },
        {
            "unresolved_semantic_need": "requested product",
            "needed_context": _needed_context(context_ref="Gear_Medium.STL"),
            "context understanding complete": False,
        },
        {
            "unresolved_semantic_need": {
                "kind": "class",
                "symbol": f"{PPR_NAMESPACE}product",
                "description": "requested product",
            },
            "needed_context": _needed_context(clarification_question="Which gear?"),
            "context understanding complete": False,
        },
        {
            **request_context("NIST_assembly_instructions.pdf"),
            "extra": True,
        },
    ],
)
def test_invalid_assessment_is_recorded_without_retrieval(
    tmp_path: Path,
    assessment: dict[str, object],
) -> None:
    grounding = ControlledGroundingRuntime(assessments=[assessment])
    product_agent = _initialize_static(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
        grounding,
    )

    result = _continue(product_agent, tmp_path, grounding)

    assert result["failure"]["reason"] == "invalid_assessment"
    decision = _read_json(tmp_path / "interaction_record/decision_0001.json")
    assert decision["Phase_4_3_output"] == assessment
    assert decision["failure"] == result["failure"]
    assert not (tmp_path / "interaction_record/retrieval_0002.json").exists()


def test_clarification_requires_persisted_assessment(tmp_path: Path) -> None:
    clarification = request_clarification("What outcome do you intend?")
    grounding = ControlledGroundingRuntime(assessments=[clarification])
    product_agent = _initialize_static(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
        grounding,
    )

    result = _continue(product_agent, tmp_path, grounding)

    assert result == {
        "needed_context": clarification["needed_context"],
        "context understanding complete": False,
    }
    assert (
        _read_json(tmp_path / "interaction_record/decision_0001.json")["Phase_4_3_output"]
        == clarification
    )
    assert not (tmp_path / "interaction_record/retrieval_0002.json").exists()


def test_interpretation_and_assessment_failures_are_recorded(tmp_path: Path) -> None:
    interpretation_runtime = ControlledGroundingRuntime(
        interpretation_error=RuntimeError("controlled interpretation failure")
    )
    product_agent = _initialize_static(
        tmp_path / "interpretation",
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
        interpretation_runtime,
    )
    interpretation_failure = _continue(
        product_agent,
        tmp_path / "interpretation",
        interpretation_runtime,
    )
    assert interpretation_failure["failure"]["reason"] == "interpretation_failed"
    assert (
        _read_json(tmp_path / "interpretation/interaction_record/interpretation_0001.json")[
            "accepted"
        ]
        is False
    )

    assessment_runtime = ControlledGroundingRuntime(
        assessment_error=RuntimeError("controlled assessment failure")
    )
    product_agent = _initialize_static(
        tmp_path / "assessment",
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
        assessment_runtime,
    )
    assessment_failure = _continue(
        product_agent,
        tmp_path / "assessment",
        assessment_runtime,
    )
    assert assessment_failure["failure"]["reason"] == "assessment_failed"
    assert (
        _read_json(tmp_path / "assessment/interaction_record/decision_0001.json")["failure"]
        == assessment_failure["failure"]
    )


def test_invalid_delta_is_rejected_atomically(tmp_path: Path) -> None:
    grounding = ControlledGroundingRuntime(
        interpretation_override={
            "assertions": [
                {
                    "subject": "https://outside.invalid/feature",
                    "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                    "object": {
                        "kind": "iri",
                        "value": "http://PAonto.com#feature",
                    },
                    "evidence_refs": ["NIST_assembly_instructions.pdf"],
                }
            ]
        }
    )
    product_agent = _initialize_static(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
        grounding,
    )

    result = _continue(product_agent, tmp_path, grounding)

    assert result["failure"]["reason"] == "invalid_interpretation"
    manifest = _read_json(tmp_path / "products/grounding/ontology/abox_manifest.json")
    assert manifest["delta_count"] == 0
    assert not (tmp_path / "products/grounding/ontology/delta_0001.json").exists()


def test_retrieval_failure_stops_after_persisted_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    grounding = ControlledGroundingRuntime(assessments=[request_context("Gear_Medium.STL")])
    product_agent = _initialize_static(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
        grounding,
    )
    monkeypatch.setattr(
        context_serving,
        "resolve_context_ref",
        lambda request: {
            "rejection": {
                "context_ref": request["context_ref"],
                "reason": "controlled_rejection",
                "message": "controlled retrieval rejection",
            }
        },
    )

    result = _continue(product_agent, tmp_path, grounding)

    assert result["failure"]["reason"] == "retrieval_failed"
    assert _read_json(tmp_path / "interaction_record/decision_0001.json")["failure"] is None
    assert (
        _read_json(tmp_path / "interaction_record/retrieval_0002.json")["retrieval_error"]["reason"]
        == "controlled_rejection"
    )


def test_existing_phase_3_3_record_is_not_overwritten(tmp_path: Path) -> None:
    grounding = ControlledGroundingRuntime(assessments=[complete_context()])
    product_agent = _initialize_static(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
        grounding,
    )
    settings_path = tmp_path / "interaction_record/pa_context_settings.json"
    settings_path.write_text('{"preserve": "exactly"}', encoding="utf-8")

    result = _continue(product_agent, tmp_path, grounding)

    assert result["failure"]["reason"] == "interaction_exists"
    assert _read_json(settings_path) == {"preserve": "exactly"}
    assert grounding.interpretation_calls == []


def test_phase_3_3_source_has_no_forbidden_runtime_dependencies() -> None:
    source = Path(context_assessment.__file__).read_text(encoding="utf-8")

    for forbidden_source in (
        "tools.document_evidence",
        "tools.rgb_d_cad_grounding",
        "ProcessPlanner",
        "primitive_catalog",
        "ResourceAgent",
        "RobotAgent",
        "/gazebo/model_states",
        "/get_entity_state",
        "evaluator",
        ".execute(",
    ):
        assert forbidden_source not in source


def _initialize_static(
    interaction_root: Path,
    product_requirement: str,
    context_ref: str,
    grounding: ControlledGroundingRuntime,
) -> FakeProductAgent:
    product_agent = FakeProductAgent(
        responses=[{"needed_context": _needed_context(context_ref=context_ref)}]
    )
    result = asyncio.run(
        start_pa_context_interaction(
            product_agent,
            interaction_root,
            product_requirement,
            ontology_config=ontology_config(),
            grounding_runtime=grounding,
        )
    )
    assert "needed_context" in result
    served = serve_pa_requested_context(interaction_root)
    assert "served_context" in served
    return product_agent


def _initialize_live(
    interaction_root: Path,
    grounding: ControlledGroundingRuntime,
) -> FakeProductAgent:
    product_agent = FakeProductAgent(
        responses=[{"needed_context": _needed_context(request_live_observation=True)}]
    )
    result = asyncio.run(
        start_pa_context_interaction(
            product_agent,
            interaction_root,
            "assemble Medium Gear",
            ontology_config=ontology_config(),
            grounding_runtime=grounding,
        )
    )
    assert "needed_context" in result
    served = serve_pa_requested_context(interaction_root)
    assert "served_context" in served
    return product_agent


def _continue(
    product_agent: FakeProductAgent,
    interaction_root: Path,
    grounding: ControlledGroundingRuntime,
    *,
    max_pa_turns: object = 12,
    live_observation_timeout_sec: float = 5.0,
) -> dict[str, object]:
    return asyncio.run(
        continue_pa_context_interaction(
            product_agent,
            interaction_root,
            ontology_config=ontology_config(),
            grounding_runtime=grounding,
            max_pa_turns=max_pa_turns,  # type: ignore[arg-type]
            live_observation_timeout_sec=live_observation_timeout_sec,
        )
    )


def _terminal_complete() -> dict[str, object]:
    return {
        "needed_context": None,
        "context understanding complete": True,
        "grounding_status": "complete",
    }


def _live_observation_bundle(observation_ref: str) -> ObservationBundle:
    camera_observations = []
    for camera_index, camera_id in enumerate(CAMERA_IDS):
        frame = f"{camera_id}_frame"
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
                camera_calibration=CameraCalibration(
                    width=IMAGE_WIDTH,
                    height=IMAGE_HEIGHT,
                    frame=frame,
                    distortion_model="plumb_bob",
                    D=(0.0,),
                    K=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                    R=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                    P=(
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
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
        observation_ref=observation_ref,
        evidence_label="live",
        captured_at_ns=1_000_000_003,
        camera_observations=tuple(camera_observations),
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

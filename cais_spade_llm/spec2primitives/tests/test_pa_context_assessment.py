"""Tests for configurable Spec2Primitives PA context reassessment."""

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
    """Return scripted reassessment decisions without lifecycle behavior."""

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


def test_medium_gear_sequence_accumulates_context_and_completes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    product_requirement = "  assemble Medium Gear exactly  "
    _write_initial_static_interaction(
        tmp_path,
        product_requirement,
        "NIST_assembly_instructions.pdf",
    )
    requested_refs = [
        "Gear_Medium.STL",
        "Gear_Plate.STL",
        "Gear_Shaft.STL",
        "GMC_Laser_Plate_Virtual.STL",
    ]
    product_agent = FakeProductAgent(
        responses=[
            *[_request_context_ref(context_ref) for context_ref in requested_refs],
            _request_live_observation(),
            _complete(),
        ]
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

    result = asyncio.run(
        continue_pa_context_interaction(
            product_agent,
            tmp_path,
            max_pa_turns=12,
            live_observation_timeout_sec=7.5,
        )
    )

    assert result == _complete()
    assert len(product_agent.calls) == 6
    assert resolver_calls == [
        {"context_ref": context_ref} for context_ref in requested_refs
    ]
    assert capture_calls == [("observation_0001", 7.5)]
    assert _read_json(tmp_path / "interaction_record/pa_context_settings.json") == {
        "max_pa_turns": 12,
        "live_observation_timeout_sec": 7.5,
    }
    for turn_number in range(1, 8):
        assert (tmp_path / f"interaction_record/turn_{turn_number:04d}.json").is_file()
    for retrieval_number in range(1, 7):
        retrieval = _read_json(
            tmp_path / f"interaction_record/retrieval_{retrieval_number:04d}.json"
        )
        assert retrieval["product_requirement"] == product_requirement
        assert retrieval["failure"] is None
    final_prompt = str(product_agent.calls[-1]["prompt"])
    for context_ref in ["NIST_assembly_instructions.pdf", *requested_refs]:
        assert context_ref in final_prompt
    assert "observation_0001" in final_prompt
    assert "Phase 4" in final_prompt
    assert "Phase 5" in final_prompt
    assert "Phase 6" in final_prompt
    assert "Phase 7" in final_prompt
    assert "primitive catalog" in final_prompt
    assert "Do not contact RA" in final_prompt
    assert "rgb" not in _read_json(
        tmp_path / "interaction_record/retrieval_0006.json"
    )["served_context"]


def test_pa_selects_an_alternative_context_order(tmp_path: Path) -> None:
    _write_initial_static_interaction(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
    )
    product_agent = FakeProductAgent(
        responses=[
            _request_context_ref("Gear_Plate.STL"),
            _request_context_ref("Gear_Medium.STL"),
            _complete(),
        ]
    )

    result = asyncio.run(
        continue_pa_context_interaction(product_agent, tmp_path, max_pa_turns=4)
    )

    assert result == _complete()
    assert _read_json(tmp_path / "interaction_record/retrieval_0002.json")[
        "context_request"
    ] == {"context_ref": "Gear_Plate.STL"}
    assert _read_json(tmp_path / "interaction_record/retrieval_0003.json")[
        "context_request"
    ] == {"context_ref": "Gear_Medium.STL"}


def test_live_observation_uses_the_next_number(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_initial_live_interaction(tmp_path, "assemble Medium Gear")
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
    product_agent = FakeProductAgent(
        responses=[_request_live_observation(), _complete()]
    )

    result = asyncio.run(
        continue_pa_context_interaction(product_agent, tmp_path, max_pa_turns=3)
    )

    assert result == _complete()
    assert capture_calls == ["observation_0002"]
    assert _read_json(tmp_path / "interaction_record/retrieval_0002.json")[
        "served_context"
    ]["observation_ref"] == "observation_0002"


@pytest.mark.parametrize("max_pa_turns", [True, 1, 1.5])
def test_invalid_max_pa_turns_is_rejected_without_call_or_settings(
    tmp_path: Path,
    max_pa_turns: object,
) -> None:
    _write_initial_static_interaction(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
    )
    product_agent = FakeProductAgent(responses=[_complete()])

    result = asyncio.run(
        continue_pa_context_interaction(
            product_agent,
            tmp_path,
            max_pa_turns=max_pa_turns,  # type: ignore[arg-type]
        )
    )

    assert result["failure"]["reason"] == "invalid_max_pa_turns"
    assert product_agent.calls == []
    assert not (tmp_path / "interaction_record/pa_context_settings.json").exists()


def test_default_limit_is_recorded_and_completion_is_not_grounding(
    tmp_path: Path,
) -> None:
    _write_initial_static_interaction(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
    )
    product_agent = FakeProductAgent(responses=[_complete()])

    result = asyncio.run(continue_pa_context_interaction(product_agent, tmp_path))

    assert result == _complete()
    assert _read_json(tmp_path / "interaction_record/pa_context_settings.json")[
        "max_pa_turns"
    ] == 12
    prompt = str(product_agent.calls[0]["prompt"])
    assert "Remaining metric grounding belongs to Phase 4" in prompt
    assert "execute robot behavior" in prompt


def test_turn_limit_records_requested_context_without_serving_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_initial_static_interaction(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
    )
    product_agent = FakeProductAgent(
        responses=[_request_context_ref("Gear_Medium.STL")]
    )
    monkeypatch.setattr(
        context_serving,
        "resolve_context_ref",
        lambda request: pytest.fail(f"Unexpected resolver call: {request}"),
    )

    result = asyncio.run(
        continue_pa_context_interaction(product_agent, tmp_path, max_pa_turns=2)
    )

    assert result["failure"]["reason"] == "pa_turn_limit_reached"
    assert not (tmp_path / "interaction_record/retrieval_0002.json").exists()
    turn = _read_json(tmp_path / "interaction_record/turn_0002.json")
    assert turn["PA_output"] == _request_context_ref("Gear_Medium.STL")
    assert turn["failure"] == result["failure"]


@pytest.mark.parametrize(
    "pa_output",
    [
        {
            "needed_context": {
                "context_ref": "NIST_assembly_instructions.pdf",
                "request_live_observation": False,
                "clarification_question": None,
            },
            "context understanding complete": False,
        },
        {
            "needed_context": {
                "context_ref": "Not_Approved.STL",
                "request_live_observation": False,
                "clarification_question": None,
            },
            "context understanding complete": False,
        },
        {
            "needed_context": {
                "context_ref": "Gear_Medium.STL",
                "request_live_observation": False,
                "clarification_question": None,
            },
            "context understanding complete": True,
        },
        {
            "needed_context": {
                "context_ref": "Gear_Medium.STL",
                "request_live_observation": False,
                "clarification_question": None,
            },
            "context understanding complete": False,
            "extra": True,
        },
        {"needed_context": None, "context understanding complete": False},
    ],
)
def test_invalid_reassessment_is_recorded_without_retrieval(
    tmp_path: Path,
    pa_output: dict[str, object],
) -> None:
    _write_initial_static_interaction(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
    )
    product_agent = FakeProductAgent(responses=[pa_output])

    result = asyncio.run(continue_pa_context_interaction(product_agent, tmp_path))

    assert result["failure"]["reason"] == "invalid_pa_response"
    assert _read_json(tmp_path / "interaction_record/turn_0002.json")["failure"] == (
        result["failure"]
    )
    assert not (tmp_path / "interaction_record/retrieval_0002.json").exists()


def test_clarification_stops_without_retrieval(tmp_path: Path) -> None:
    _write_initial_static_interaction(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
    )
    clarification = _request_clarification("Which Medium Gear should be assembled?")

    result = asyncio.run(
        continue_pa_context_interaction(
            FakeProductAgent(responses=[clarification]),
            tmp_path,
        )
    )

    assert result == clarification
    assert not (tmp_path / "interaction_record/retrieval_0002.json").exists()


def test_pa_exception_and_retrieval_failure_are_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_initial_static_interaction(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
    )
    pa_failure = asyncio.run(
        continue_pa_context_interaction(
            FakeProductAgent(error=RuntimeError("controlled")),
            tmp_path,
        )
    )
    assert pa_failure["failure"]["reason"] == "pa_call_failed"

    second_root = tmp_path / "second"
    _write_initial_static_interaction(
        second_root,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
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
    retrieval_failure = asyncio.run(
        continue_pa_context_interaction(
            FakeProductAgent(responses=[_request_context_ref("Gear_Medium.STL")]),
            second_root,
        )
    )
    assert retrieval_failure["failure"]["reason"] == "retrieval_failed"
    assert retrieval_failure["retrieval_error"]["reason"] == "controlled_rejection"
    assert _read_json(second_root / "interaction_record/retrieval_0002.json")[
        "retrieval_error"
    ] == retrieval_failure["retrieval_error"]


def test_existing_phase_3_3_record_is_not_overwritten(tmp_path: Path) -> None:
    _write_initial_static_interaction(
        tmp_path,
        "assemble Medium Gear",
        "NIST_assembly_instructions.pdf",
    )
    settings_path = tmp_path / "interaction_record/pa_context_settings.json"
    _write_json(settings_path, {"preserve": "exactly"})
    product_agent = FakeProductAgent(responses=[_complete()])

    result = asyncio.run(continue_pa_context_interaction(product_agent, tmp_path))

    assert result["failure"]["reason"] == "interaction_exists"
    assert _read_json(settings_path) == {"preserve": "exactly"}
    assert product_agent.calls == []


def test_phase_3_3_source_has_no_forbidden_runtime_dependencies() -> None:
    source = Path(context_assessment.__file__).read_text(encoding="utf-8")

    for forbidden_source in (
        "VLM",
        "grounding import",
        "ProcessPlanner",
        "primitive_catalog",
        "ResourceAgent",
        "RobotAgent",
        "CCA",
        "/gazebo/model_states",
        "/get_entity_state",
        "evaluator",
        ".execute(",
    ):
        assert forbidden_source not in source


def _write_initial_static_interaction(
    interaction_root: Path,
    product_requirement: str,
    context_ref: str,
) -> None:
    needed_context = _needed_context(context_ref=context_ref)
    served_context = _static_served_context(context_ref)
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
    _write_json(
        interaction_root / "interaction_record/retrieval_0001.json",
        {
            "retrieval": 1,
            "product_requirement": product_requirement,
            "needed_context": needed_context,
            "context_request": {"context_ref": context_ref},
            "served_context": served_context,
            "retrieval_error": None,
            "failure": None,
        },
    )


def _write_initial_live_interaction(
    interaction_root: Path,
    product_requirement: str,
) -> None:
    needed_context = _needed_context(request_live_observation=True)
    served_context = {
        "context_ref": None,
        "observation_ref": "observation_0001",
        "evidence_type": "observation",
        "evidence_label": "live",
        "provenance": {
            "manifest_path": "products/observations/observation_0001/manifest.json"
        },
        "observation_evidence": {"manifest": {}, "artifact_references": []},
    }
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
    _write_json(
        interaction_root / "interaction_record/retrieval_0001.json",
        {
            "retrieval": 1,
            "product_requirement": product_requirement,
            "needed_context": needed_context,
            "context_request": {
                "request_live_observation": True,
                "observation_ref": "observation_0001",
                "live_observation_timeout_sec": 5.0,
            },
            "served_context": served_context,
            "retrieval_error": None,
            "failure": None,
        },
    )


def _static_served_context(context_ref: str) -> dict[str, object]:
    evidence_type = "document" if context_ref.endswith(".pdf") else "CAD"
    evidence_key = "document_evidence" if evidence_type == "document" else "CAD_evidence"
    return {
        "context_ref": context_ref,
        "evidence_type": evidence_type,
        "provenance": {
            "repository_path": f"controlled/{context_ref}",
            "source_url": "https://example.invalid/controlled",
        },
        evidence_key: {},
    }


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


def _request_context_ref(context_ref: str) -> dict[str, object]:
    return {
        "needed_context": _needed_context(context_ref=context_ref),
        "context understanding complete": False,
    }


def _request_live_observation() -> dict[str, object]:
    return {
        "needed_context": _needed_context(request_live_observation=True),
        "context understanding complete": False,
    }


def _request_clarification(question: str) -> dict[str, object]:
    return {
        "needed_context": _needed_context(clarification_question=question),
        "context understanding complete": False,
    }


def _complete() -> dict[str, object]:
    return {
        "needed_context": None,
        "context understanding complete": True,
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
                    P=(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                ),
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

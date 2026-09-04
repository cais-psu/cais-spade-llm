"""Tests for starting one native ProductAgent grounding interaction."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    start_pa_context_interaction,
)
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingProducerDescriptor,
)
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    ontology_config,
)


class _UnusedProductAgent:
    async def ask_llm_structured(self, *args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise AssertionError("The controlled grounding runtime owns this test result.")


class _GroundingRuntime:
    def __init__(
        self,
        outputs: Sequence[Mapping[str, object] | Exception],
    ) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    def grounding_producer_descriptors(self) -> tuple[GroundingProducerDescriptor, ...]:
        return _descriptors()

    async def ground_product_context(
        self,
        product_agent: object,
        *,
        interaction_root: Path,
        tbox: object,
        abox: object,
        product_context: Mapping[str, object],
        max_pa_turns: int,
        clarification_history: tuple[Mapping[str, object], ...] = (),
    ) -> Mapping[str, object]:
        del product_agent, tbox
        self.calls.append(
            {
                "interaction_root": interaction_root,
                "delta_count": abox.delta_count,
                "product_context": dict(product_context),
                "max_pa_turns": max_pa_turns,
                "clarification_history": clarification_history,
            }
        )
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


@pytest.mark.parametrize(
    "output",
    [
        {
            "grounding_status": "clarification_required",
            "clarification_question": "Which product variant is intended?",
            "tool_call_refs": [],
        },
        *[
            {
                "grounding_status": "incomplete",
                "grounding_stage": "target_feature",
                "insufficient_evidence": f"Deterministic controller result: {code}.",
                "grounding_validation_code": code,
                "tool_call_refs": [],
            }
            for code in (
                "unsupported_process",
                "invalid_target_feature",
                "evidence_reference_invalid",
                "location_evidence_unavailable",
                "no_reachable_resource",
                "invalid_resource_selection",
            )
        ],
    ],
)
def test_start_runs_one_native_investigation_and_persists_direct_result(
    tmp_path: Path,
    output: dict[str, object],
) -> None:
    runtime = _GroundingRuntime([output])

    result = asyncio.run(
        start_pa_context_interaction(
            _UnusedProductAgent(),
            tmp_path,
            "assemble medium gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )

    assert result == output
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["delta_count"] == 0
    assert runtime.calls[0]["max_pa_turns"] == 12
    turn = _read_json(tmp_path / "interaction_record/turn_0001.json")
    assert turn["PA_input"]["mode"] == "native_tool_grounding"
    assert turn["PA_output"] == output
    assert turn["failure"] is None
    serialized = json.dumps(turn)
    assert "next_action" not in serialized
    assert "needed_context" not in serialized
    assert not (tmp_path / "products/grounding/session").exists()


@pytest.mark.parametrize("requirement", ["", " \t\n"])
def test_blank_requirement_is_rejected_without_records(
    tmp_path: Path,
    requirement: str,
) -> None:
    runtime = _GroundingRuntime([])

    result = asyncio.run(
        start_pa_context_interaction(
            _UnusedProductAgent(),
            tmp_path,
            requirement,
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )

    assert result["failure"]["reason"] == "invalid_product_requirement"
    assert runtime.calls == []
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "output",
    [
        {},
        {"grounding_status": "unknown"},
        {"grounding_status": "clarification_required"},
        {"grounding_status": "incomplete"},
        {
            "grounding_status": "incomplete",
            "insufficient_evidence": "Historical uncoded grounding result.",
            "tool_call_refs": [],
        },
        {
            "grounding_status": "incomplete",
            "insufficient_evidence": "A declared obligation is unresolved.",
            "unmet_grounding_obligation": 1,
        },
        {
            "grounding_status": "incomplete",
            "insufficient_evidence": "Controller-authored diagnostic.",
            "grounding_validation_code": "not_a_grounding_code",
        },
        {
            "grounding_status": "incomplete",
            "insufficient_evidence": "Controller-authored diagnostic.",
            "grounding_validation_code": "missing_obligation",
        },
        {
            "grounding_status": "incomplete",
            "insufficient_evidence": "Controller-authored diagnostic.",
            "grounding_validation_code": "missing_obligation",
            "unmet_grounding_obligation": "backlash",
        },
        {
            "grounding_status": "incomplete",
            "insufficient_evidence": "Controller-authored diagnostic.",
            "grounding_validation_code": "claims_evidence_invalid",
            "unmet_grounding_obligation": "current_state.state_values",
        },
        {
            "grounding_status": "complete",
            "grounding_validation_code": "investigation_exhausted",
            "ontology_projection_ref": "proposal.json",
            "resource_selection_ref": "selection.json",
            "tool_call_refs": [],
        },
        {
            "grounding_status": "clarification_required",
            "clarification_question": "Which product variant is intended?",
            "unmet_grounding_obligation": "current_state.statement",
        },
        {
            "grounding_status": "complete",
            "ontology_projection_ref": "proposal.json",
            "resource_selection_ref": "selection.json",
            "tool_call_refs": "not-a-list",
        },
    ],
)
def test_invalid_native_result_is_rejected_and_preserved_for_audit(
    tmp_path: Path,
    output: dict[str, object],
) -> None:
    runtime = _GroundingRuntime([output])

    result = asyncio.run(
        start_pa_context_interaction(
            _UnusedProductAgent(),
            tmp_path,
            "assemble medium gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )

    assert result["failure"]["reason"] == "invalid_pa_response"
    turn = _read_json(tmp_path / "interaction_record/turn_0001.json")
    assert turn["PA_output"] == output
    assert turn["failure"]["reason"] == "invalid_pa_response"


def test_grounding_exception_is_fail_closed_and_recorded(tmp_path: Path) -> None:
    runtime = _GroundingRuntime([RuntimeError("controlled failure")])

    result = asyncio.run(
        start_pa_context_interaction(
            _UnusedProductAgent(),
            tmp_path,
            "assemble medium gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )

    assert result["failure"]["reason"] == "pa_call_failed"
    assert "controlled failure" in result["failure"]["message"]
    turn = _read_json(tmp_path / "interaction_record/turn_0001.json")
    assert turn["PA_output"] is None
    assert turn["failure"]["reason"] == "pa_call_failed"


def test_missing_grounding_dependencies_fail_before_writes(tmp_path: Path) -> None:
    result = asyncio.run(
        start_pa_context_interaction(
            _UnusedProductAgent(),
            tmp_path,
            "assemble medium gear",
        )
    )

    assert result["failure"]["reason"] == "grounding_unavailable"
    assert not any(tmp_path.iterdir())


def _descriptors() -> tuple[GroundingProducerDescriptor, ...]:
    return (
        GroundingProducerDescriptor.from_mapping(
            {
                "provider_id": "document_test_provider",
                "description": "Produce document evidence.",
                "accepted_evidence_types": ["document"],
                "produced_record_types": ["DocumentOverviewRecord"],
                "prerequisites": {"DocumentOverviewRecord": []},
                "availability": True,
                "estimated_cost": 1,
            }
        ),
        GroundingProducerDescriptor.from_mapping(
            {
                "provider_id": "geometry_test_provider",
                "description": "Produce CAD and observation evidence.",
                "accepted_evidence_types": ["CAD", "observation"],
                "produced_record_types": ["CADMeshRecord"],
                "prerequisites": {"CADMeshRecord": []},
                "availability": True,
                "estimated_cost": 1,
            }
        ),
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))

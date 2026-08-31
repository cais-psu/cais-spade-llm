"""Focused tests for native tool-using ProductAgent grounding."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.spec2primitives.agents.pa import production_grounding
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingProducerDescriptor,
)
from cais_spade_llm.spec2primitives.agents.pa.ontology_grounding import (
    OntologyGroundingCandidate,
    OntologyGroundingInterruption,
    OntologyGroundingResult,
    commit_ontology_grounding_candidate,
    propose_and_validate_ontology_grounding,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    initialize_interaction_abox,
    load_interaction_abox,
)
from cais_spade_llm.spec2primitives.agents.pa.production_grounding import (
    ProductionProductContextGroundingRuntime,
    _approved_evidence_handles,
    _clarification_requests_system_choice,
    _EvidenceHandle,
    _GroundingGap,
    _NativeEvidenceInvestigation,
    _PreparedGrounding,
    _producer_descriptors,
    _proposal_cad_bindings,
    _raw_evidence_types_for_gap,
    _required_record_plan,
    _retrieve_tool,
    _target_required_binding,
)
from cais_spade_llm.spec2primitives.config import load_model_runtime_config
from cais_spade_llm.spec2primitives.ontology import (
    load_predefined_resource_registry,
    load_predefined_workcell,
)
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    PPR_NAMESPACE,
    ontology_config,
)


class _NoDocumentVision:
    async def interpret_document(self, request: object) -> object:
        del request
        raise AssertionError("CAD-only test must not invoke document vision.")


class _ToolUsingProductAgent:
    def __init__(
        self,
        *,
        retrieve_order: tuple[str, ...],
        final_result: Mapping[str, object] | None = None,
    ) -> None:
        self.retrieve_order = retrieve_order
        self.final_result = final_result
        self.calls: list[dict[str, object]] = []

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[
            [str, Mapping[str, object]], Awaitable[Mapping[str, object]]
        ]
        | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        assert tool_executor is not None
        retrieved_refs: list[str] = []
        for evidence_id in self.retrieve_order:
            result = await tool_executor("retrieve", {"evidence_id": evidence_id})
            refs = result.get("evidence_refs")
            if isinstance(refs, list):
                retrieved_refs.extend(ref for ref in refs if isinstance(ref, str))
        self.calls.append(
            {
                "prompt": prompt,
                "response_format": response_format,
                "tools": tools,
                "max_tool_rounds": max_tool_rounds,
            }
        )
        if self.final_result is not None:
            return {"result": dict(self.final_result)}
        citation = retrieved_refs[-1] if retrieved_refs else "requirement_0001"
        return {"result": _proposal(citation)}


class _SequencedProductAgent(_ToolUsingProductAgent):
    def __init__(self, results: tuple[Mapping[str, object], ...]) -> None:
        super().__init__(retrieve_order=())
        self.results = results

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[
            [str, Mapping[str, object]], Awaitable[Mapping[str, object]]
        ]
        | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "prompt": prompt,
                "response_format": response_format,
                "tools": tools,
                "max_tool_rounds": max_tool_rounds,
            }
        )
        return {"result": dict(self.results[len(self.calls) - 1])}


@pytest.mark.parametrize(
    "retrieve_order",
    [
        (),
        ("evidence_document",),
        ("evidence_CAD", "evidence_document", "evidence_observation"),
        ("evidence_observation", "evidence_CAD"),
    ],
)
def test_pa_can_use_zero_or_multiple_retrievals_in_any_order(
    tmp_path: Path,
    retrieve_order: tuple[str, ...],
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble medium gear", tbox)
    registry = load_predefined_resource_registry(tbox)
    workcell = load_predefined_workcell(tbox, registry)
    authorized = {"requirement_0001"}
    calls: list[str] = []

    async def retrieve(
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        assert tool_name == "retrieve"
        evidence_id = str(arguments["evidence_id"])
        calls.append(evidence_id)
        evidence_ref = f"products/grounding/test/{evidence_id}.json"
        authorized.add(evidence_ref)
        return {"evidence_refs": [evidence_ref], "record_refs": [evidence_ref]}

    agent = _ToolUsingProductAgent(retrieve_order=retrieve_order)
    result = asyncio.run(
        propose_and_validate_ontology_grounding(
            agent,
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            workcell=workcell,
            evidence_catalog=[
                {"evidence_id": evidence_id, "evidence_type": evidence_type}
                for evidence_id, evidence_type in (
                    ("evidence_document", "document"),
                    ("evidence_CAD", "CAD"),
                    ("evidence_observation", "observation"),
                )
            ],
            authorized_evidence_refs=authorized,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "retrieve",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            tool_executor=retrieve,
            max_tool_rounds=12,
            required_output_projection={
                "record_type": "RobotFrameLocationRecord",
                "target_frame": "world",
            },
        )
    )

    assert isinstance(result, OntologyGroundingCandidate)
    assert calls == list(retrieve_order)
    assert result.proposal_path.name == "proposal_0001.json"
    assert not result.proposal_path.exists()
    assert load_interaction_abox(root, tbox).delta_count == 0
    committed = commit_ontology_grounding_candidate(
        result,
        interaction_root=root,
        tbox=tbox,
        abox=abox,
        workcell=workcell,
        authorized_evidence_refs=authorized,
    )
    assert isinstance(committed, OntologyGroundingResult)
    assert committed.merge.abox.delta_count == 1
    call = agent.calls[0]
    assert call["response_format"]["name"] == "spec2primitives_grounding_result"
    schema = call["response_format"]["schema"]
    assert schema["type"] == "object"
    assert set(schema) == {
        "type",
        "additionalProperties",
        "required",
        "properties",
    }
    assert schema["required"] == ["result"]
    assert len(schema["properties"]["result"]["anyOf"]) == 3
    proposal_schema = schema["properties"]["result"]["anyOf"][0]
    relation_variants = proposal_schema["properties"]["relations"]["items"]["anyOf"]
    specification_variants = [
        variant
        for variant in relation_variants
        if variant["properties"]["subject_kind"]["enum"] == ["specification"]
    ]
    assert len(specification_variants) == 2
    expected_predicates = {
        f"{PPR_NAMESPACE}defines",
        f"{PPR_NAMESPACE}realizes",
    }
    for variant in relation_variants:
        assert set(variant["properties"]["predicate_iri"]["enum"]) == (
            expected_predicates
        )
    for variant in specification_variants:
        assert variant["properties"]["subject_individual_index"] == {"type": "null"}
        assert variant["properties"]["subject_iri"] == {"type": "null"}
    serialized = json.dumps(call)
    for removed_protocol in ("next_action", "propose_grounding", '"inspect"'):
        assert removed_protocol not in serialized
    assert "RobotFrameLocationRecord" in str(call["prompt"])


def test_direct_clarification_and_insufficient_evidence_stop_without_proposal(
    tmp_path: Path,
) -> None:
    for index, final_result in enumerate(
        (
            {"clarification_question": "Which product variant is intended?"},
            {"insufficient_evidence": "No retrieved source supports a feature."},
        ),
        start=1,
    ):
        root = tmp_path / f"case_{index}"
        tbox = ontology_config().load_tbox()
        abox = initialize_interaction_abox(root, "assemble product", tbox)
        registry = load_predefined_resource_registry(tbox)
        workcell = load_predefined_workcell(tbox, registry)
        result = asyncio.run(
            propose_and_validate_ontology_grounding(
                _ToolUsingProductAgent(
                    retrieve_order=(),
                    final_result=final_result,
                ),
                interaction_root=root,
                tbox=tbox,
                abox=abox,
                workcell=workcell,
                evidence_catalog=[],
                authorized_evidence_refs={"requirement_0001"},
                tools=[],
                tool_executor=_unused_tool,
                max_tool_rounds=3,
                required_output_projection={
                    "record_type": "RobotFrameLocationRecord",
                    "target_frame": "world",
                },
            )
        )
        assert isinstance(result, OntologyGroundingInterruption)
        assert not (root / "products/grounding/ontology_grounding").exists()


def test_retrieve_tool_exposes_only_prompt_local_evidence_id() -> None:
    handles = _approved_evidence_handles()
    tool = _retrieve_tool(handles)

    assert tool["function"]["name"] == "retrieve"
    parameters = tool["function"]["parameters"]
    assert set(parameters["properties"]) == {"evidence_id"}
    assert parameters["required"] == ["evidence_id"]
    serialized = json.dumps(tool)
    for forbidden in (
        "provider_id",
        "record_type",
        "target_frame",
        "manifest_ref",
        "repository_path",
    ):
        assert forbidden not in serialized
    assert {handle.evidence_type for handle in handles} == {
        "document",
        "CAD",
        "observation",
    }


def test_unauthorized_and_stale_ids_fail_closed_and_are_audited(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble product", tbox)
    runtime = ProductionProductContextGroundingRuntime(
        tbox=tbox,
        document_config=load_model_runtime_config().document_vlm,
        document_vision_runtime=_NoDocumentVision(),
    )
    stale = _EvidenceHandle(
        evidence_id="evidence_stale",
        evidence_type="CAD",
        display_name="Gear_Medium.STL",
        context_ref="Gear_Medium.STL",
        source_revision="0" * 64,
    )
    investigation = _NativeEvidenceInvestigation(
        runtime=runtime,
        interaction_root=root,
        tbox=tbox,
        abox=abox,
        requirement=abox.product_requirement,
        handles=[stale],
    )

    unknown = asyncio.run(
        investigation.execute("retrieve", {"evidence_id": "not_catalogued"})
    )
    stale_result = asyncio.run(
        investigation.execute("retrieve", {"evidence_id": "evidence_stale"})
    )

    assert unknown["error"]["reason"] == "unauthorized_evidence_id"
    assert stale_result["error"]["reason"] == "stale_evidence_id"
    audits = sorted((root / "interaction_record").glob("tool_call_*.json"))
    assert [path.name for path in audits] == [
        "tool_call_0001.json",
        "tool_call_0002.json",
    ]
    assert json.loads(audits[0].read_text())["resolved_evidence"] is None
    assert json.loads(audits[1].read_text())["resolved_evidence"]["evidence_id"] == (
        "evidence_stale"
    )


def test_static_cad_is_reused_across_calls_and_clarification_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble medium gear", tbox)
    medium = _EvidenceHandle(
        evidence_id="evidence_0001",
        evidence_type="CAD",
        display_name="synthetic static CAD",
        context_ref="synthetic.stl",
        source_revision="a" * 64,
    )
    handles = (medium,)
    retrieval_calls: list[int] = []

    def retrieve_fixture(
        interaction_root: Path,
        *,
        product_requirement: str,
        evidence_type: str,
        context_ref: str | None,
        retrieval_number: int,
        live_observation_timeout_sec: float,
    ) -> Mapping[str, object]:
        del product_requirement, live_observation_timeout_sec
        retrieval_calls.append(retrieval_number)
        return {
            "served_context": {
                "evidence_type": evidence_type,
                "context_ref": context_ref,
            }
        }

    class CADFixtureRuntime:
        async def interpret_retrieved_evidence(
            self,
            *,
            interaction_root: Path,
            tbox: object,
            abox: object,
            served_context: Mapping[str, object],
            operation_number: int,
        ) -> Mapping[str, object]:
            del tbox, abox, served_context
            path = (
                interaction_root
                / "products/grounding/rgb_d_cad_grounding"
                / f"cad_{operation_number:04d}.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "record_type": "CADMeshRecord",
                        "coordinate_frame": "CAD_local",
                        "stored_units": "m",
                        "triangle_count": 1,
                        "vertex_count": 3,
                        "bounds_m": [[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]],
                        "vertex_centroid_m": [0.03, 0.03, 0.0],
                    }
                ),
                encoding="utf-8",
            )
            return {"typed_context_refs": [path.relative_to(interaction_root).as_posix()]}

    monkeypatch.setattr(
        production_grounding.context_serving,
        "retrieve_pa_evidence",
        retrieve_fixture,
    )
    async def run_immediately(
        function: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        return function(*args, **kwargs)

    monkeypatch.setattr(production_grounding.asyncio, "to_thread", run_immediately)
    monkeypatch.setattr(
        production_grounding,
        "_handle_revision_is_current",
        lambda handle: handle == medium,
    )
    monkeypatch.setattr(
        production_grounding,
        "validate_and_merge_triple_delta",
        lambda *args, **kwargs: SimpleNamespace(abox=abox),
    )
    monkeypatch.setattr(
        production_grounding,
        "build_product_context_view",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        production_grounding,
        "persist_product_context_view",
        lambda *args, **kwargs: root / "view.json",
    )
    runtime = CADFixtureRuntime()
    first = _NativeEvidenceInvestigation(
        runtime=runtime,
        interaction_root=root,
        tbox=tbox,
        abox=abox,
        requirement=abox.product_requirement,
        handles=handles,
    )

    first_result = asyncio.run(
        first.execute("retrieve", {"evidence_id": medium.evidence_id})
    )
    repeated_result = asyncio.run(
        first.execute("retrieve", {"evidence_id": medium.evidence_id})
    )
    assert first_result == repeated_result
    assert retrieval_calls == [1]
    assert json.loads(
        (root / "interaction_record/tool_call_0002.json").read_text()
    )["reused"] is True

    resumed = _NativeEvidenceInvestigation(
        runtime=runtime,
        interaction_root=root,
        tbox=tbox,
        abox=abox,
        requirement=abox.product_requirement,
        handles=handles,
    )
    assert resumed.prior_evidence[0]["retrieval_state"] == "already_retrieved"
    resumed_result = asyncio.run(
        resumed.execute("retrieve", {"evidence_id": medium.evidence_id})
    )
    assert resumed_result == first_result
    assert retrieval_calls == [1]
    assert json.loads(
        (root / "interaction_record/tool_call_0003.json").read_text()
    )["reused"] is True


def test_descriptor_closure_is_location_driven_and_accepts_synthetic_provider() -> None:
    descriptors = _producer_descriptors(calibration_available=True)

    assert _required_record_plan(descriptors, "RobotFrameLocationRecord") == (
        "CADMeshRecord",
        "ColoredPointCloudSetRecord",
        "RGBDSegmentationRecord",
        "CADSizeCorrespondenceRecord",
        "CameraToRobotCalibrationRecord",
        "RobotFrameLocationRecord",
    )
    pose_plan = _required_record_plan(descriptors, "CADPoseEstimationRecord")
    assert pose_plan[-1] == "CADPoseEstimationRecord"
    assert "RobotFrameLocationRecord" not in pose_plan
    assert "CameraToRobotCalibrationRecord" not in pose_plan

    synthetic = GroundingProducerDescriptor.from_mapping(
        {
            "provider_id": "synthetic_consumer_provider",
            "description": "Consume any accepted location record.",
            "accepted_evidence_types": ["existing_record"],
            "produced_record_types": ["SyntheticReachRecord"],
            "prerequisites": {
                "SyntheticReachRecord": ["RobotFrameLocationRecord"]
            },
            "availability": True,
            "estimated_cost": 0,
        }
    )
    plan = _required_record_plan((*descriptors, synthetic), "SyntheticReachRecord")
    assert plan[-2:] == ("RobotFrameLocationRecord", "SyntheticReachRecord")


def test_descriptor_gap_derives_raw_evidence_types_without_product_rules() -> None:
    descriptors = _producer_descriptors(calibration_available=True)

    assert _raw_evidence_types_for_gap(
        descriptors,
        ("CADMeshRecord",),
    ) == {"CAD"}
    assert _raw_evidence_types_for_gap(
        descriptors,
        ("ColoredPointCloudSetRecord", "RGBDSegmentationRecord"),
    ) == {"observation"}
    assert _raw_evidence_types_for_gap(
        descriptors,
        (
            "CADMeshRecord",
            "ColoredPointCloudSetRecord",
            "RGBDSegmentationRecord",
            "CADSizeCorrespondenceRecord",
            "CameraToRobotCalibrationRecord",
            "RobotFrameLocationRecord",
        ),
    ) == {"CAD", "observation"}


def test_target_cad_requires_primary_feature_citation(tmp_path: Path) -> None:
    record_path = tmp_path / "products/grounding/cad/geometry_record.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps({"source": {"context_ref": "Gear_Medium.STL"}}),
        encoding="utf-8",
    )
    binding = SimpleNamespace(
        record_type="CADMeshRecord",
        status="accepted",
        record_ref=record_path.relative_to(tmp_path).as_posix(),
        evidence_refs=("Gear_Medium.STL",),
    )
    view = SimpleNamespace(typed_bindings=(binding,))
    uncited = SimpleNamespace(
        relations=(
            {
                "predicate_iri": f"{PPR_NAMESPACE}realizes",
                "object_individual_index": 1,
            },
        ),
        individuals=(
            {"individual_index": 1, "evidence_refs": ("requirement_0001",)},
        ),
    )
    cited = SimpleNamespace(
        relations=uncited.relations,
        individuals=(
            {"individual_index": 1, "evidence_refs": ("Gear_Medium.STL",)},
        ),
    )

    assert _proposal_cad_bindings(tmp_path, view, uncited) == ()
    assert _proposal_cad_bindings(tmp_path, view, cited) == (binding,)


def test_accepted_target_record_can_satisfy_consumer_without_modality_route() -> None:
    older = SimpleNamespace(
        record_type="RobotFrameLocationRecord",
        status="accepted",
        frame="world",
        observed_at_ns=10,
        record_ref="older.json",
        evidence_refs=("direct_target_evidence",),
    )
    newest = SimpleNamespace(
        record_type="RobotFrameLocationRecord",
        status="accepted",
        frame="world",
        observed_at_ns=20,
        record_ref="newest.json",
        evidence_refs=("direct_target_evidence",),
    )
    distractor = SimpleNamespace(
        record_type="RobotFrameLocationRecord",
        status="accepted",
        frame="world",
        observed_at_ns=30,
        record_ref="distractor.json",
        evidence_refs=("unrelated_evidence",),
    )
    proposal = SimpleNamespace(
        relations=(
            {
                "predicate_iri": f"{PPR_NAMESPACE}realizes",
                "object_individual_index": 1,
            },
        ),
        individuals=(
            {
                "individual_index": 1,
                "evidence_refs": ("direct_target_evidence",),
            },
        ),
    )
    need = SimpleNamespace(
        required_record_type="RobotFrameLocationRecord",
        target_frame="world",
    )
    view = SimpleNamespace(typed_bindings=(older, newest, distractor))

    assert _target_required_binding(view, proposal, need) is newest


def test_evidence_gap_reenters_pa_before_semantic_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble medium gear", tbox)
    runtime = ProductionProductContextGroundingRuntime(
        tbox=tbox,
        document_config=load_model_runtime_config().document_vlm,
        document_vision_runtime=_NoDocumentVision(),
    )
    agent = _ToolUsingProductAgent(retrieve_order=())
    prepare_calls = 0

    async def prepare(**kwargs: object) -> object:
        nonlocal prepare_calls
        prepare_calls += 1
        assert load_interaction_abox(root, tbox).accepted_assertion_count == 0
        if prepare_calls == 1:
            return _GroundingGap(
                required_record_type="RobotFrameLocationRecord",
                missing_record_types=("RGBDSegmentationRecord",),
                eligible_evidence_ids=("evidence_observation",),
            )
        return _PreparedGrounding(
            abox=abox,
            view=SimpleNamespace(),
            need=SimpleNamespace(),
            location_binding=SimpleNamespace(record_ref="unused_location.json"),
        )

    async def complete(**kwargs: object) -> Mapping[str, object]:
        del kwargs
        assert load_interaction_abox(root, tbox).accepted_assertion_count == 3
        return {
            "grounding_status": "complete",
            "resource_selection_ref": "products/grounding/resource_selection/test.json",
        }

    monkeypatch.setattr(runtime, "_prepare_required_context", prepare)
    monkeypatch.setattr(runtime, "_complete_resource_assignment", complete)
    result = asyncio.run(
        runtime.ground_product_context(
            agent,
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            product_context={},
            max_pa_turns=3,
        )
    )

    assert result["grounding_status"] == "complete"
    assert prepare_calls == 2
    assert len(agent.calls) == 2
    assert "current_validation_gap" not in str(agent.calls[0]["prompt"])
    assert "RGBDSegmentationRecord" in str(agent.calls[1]["prompt"])
    assert load_interaction_abox(root, tbox).accepted_assertion_count == 3
    assert (root / "products/grounding/ontology_grounding/proposal_0001.json").is_file()


def test_invalid_proposal_receives_bounded_feedback_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble medium gear", tbox)
    runtime = ProductionProductContextGroundingRuntime(
        tbox=tbox,
        document_config=load_model_runtime_config().document_vlm,
        document_vision_runtime=_NoDocumentVision(),
    )
    invalid = _proposal("requirement_0001")
    invalid["relations"] = [
        relation
        for relation in invalid["relations"]
        if relation["predicate_iri"] != f"{PPR_NAMESPACE}defines"
    ]
    agent = _SequencedProductAgent(
        (invalid, _proposal("requirement_0001"))
    )

    async def prepare(**kwargs: object) -> _PreparedGrounding:
        del kwargs
        assert load_interaction_abox(root, tbox).accepted_assertion_count == 0
        return _PreparedGrounding(
            abox=abox,
            view=SimpleNamespace(),
            need=SimpleNamespace(),
            location_binding=SimpleNamespace(record_ref="unused_location.json"),
        )

    async def complete(**kwargs: object) -> Mapping[str, object]:
        del kwargs
        assert load_interaction_abox(root, tbox).accepted_assertion_count == 3
        return {
            "grounding_status": "complete",
            "resource_selection_ref": (
                "products/grounding/resource_selection/test.json"
            ),
        }

    monkeypatch.setattr(runtime, "_prepare_required_context", prepare)
    monkeypatch.setattr(runtime, "_complete_resource_assignment", complete)
    result = asyncio.run(
        runtime.ground_product_context(
            agent,
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            product_context={},
            max_pa_turns=3,
        )
    )

    assert result["grounding_status"] == "complete"
    assert len(agent.calls) == 2
    assert "ontology_proposal_validation_error" in str(agent.calls[1]["prompt"])
    assert "Every grounded feature must be defined" in str(agent.calls[1]["prompt"])
    rejected = json.loads(
        (
            root
            / "products/grounding/ontology_grounding/proposal_0001.json"
        ).read_text(encoding="utf-8")
    )
    accepted = json.loads(
        (
            root
            / "products/grounding/ontology_grounding/proposal_0002.json"
        ).read_text(encoding="utf-8")
    )
    assert rejected["status"] == "rejected"
    assert accepted["status"] == "accepted"
    assert load_interaction_abox(root, tbox).accepted_assertion_count == 3


def test_system_owned_evidence_question_is_not_user_clarification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble medium gear", tbox)
    runtime = ProductionProductContextGroundingRuntime(
        tbox=tbox,
        document_config=load_model_runtime_config().document_vlm,
        document_vision_runtime=_NoDocumentVision(),
    )
    question = {
        "clarification_question": (
            "Do you also want retrieval of the live RGB-D observation evidence "
            "for the current RobotFrameLocationRecord validation gap?"
        )
    }
    agent = _SequencedProductAgent(
        (question, _proposal("requirement_0001"))
    )

    async def prepare(**kwargs: object) -> _PreparedGrounding:
        del kwargs
        return _PreparedGrounding(
            abox=abox,
            view=SimpleNamespace(),
            need=SimpleNamespace(),
            location_binding=SimpleNamespace(record_ref="unused_location.json"),
        )

    async def complete(**kwargs: object) -> Mapping[str, object]:
        del kwargs
        return {
            "grounding_status": "complete",
            "resource_selection_ref": (
                "products/grounding/resource_selection/test.json"
            ),
        }

    monkeypatch.setattr(runtime, "_prepare_required_context", prepare)
    monkeypatch.setattr(runtime, "_complete_resource_assignment", complete)
    result = asyncio.run(
        runtime.ground_product_context(
            agent,
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            product_context={},
            max_pa_turns=3,
        )
    )

    assert result["grounding_status"] == "complete"
    assert "clarification_question" not in result
    assert len(agent.calls) == 2
    assert "system_owned_grounding_choice" in str(agent.calls[1]["prompt"])
    assert _clarification_requests_system_choice(
        str(question["clarification_question"]),
        required_record_type="RobotFrameLocationRecord",
        handles=_approved_evidence_handles(),
    )
    assert not _clarification_requests_system_choice(
        "Which product variant do you mean?",
        required_record_type="RobotFrameLocationRecord",
        handles=_approved_evidence_handles(),
    )


def test_production_source_has_no_task_label_or_answer_recipe() -> None:
    package_root = Path(__file__).resolve().parents[1]
    source = (package_root / "agents/pa/production_grounding.py").read_text(
        encoding="utf-8"
    )
    prompt_source = (package_root / "agents/pa/ontology_grounding.py").read_text(
        encoding="utf-8"
    )

    assert "supports_manipulator_pick_place" not in source
    assert "mounting order" not in prompt_source
    assert "Phase 5" not in prompt_source
    assert "next_action" not in prompt_source
    assert '"inspect"' not in prompt_source
    assert "exactly one feature" not in prompt_source.lower()
    assert "compatible retrieved CAD and live observation evidence" not in source


async def _unused_tool(
    tool_name: str,
    arguments: Mapping[str, object],
) -> Mapping[str, object]:
    del tool_name, arguments
    raise AssertionError("No tool call was expected.")


def _proposal(evidence_ref: str) -> dict[str, object]:
    return {
        "individuals": [
            {
                "individual_index": 1,
                "class_iri": f"{PPR_NAMESPACE}feature",
                "grounded_meaning": "The requested assembly feature.",
                "evidence_refs": [evidence_ref],
            }
        ],
        "relations": [
            {
                "subject_kind": "specification",
                "subject_individual_index": None,
                "subject_iri": None,
                "predicate_iri": f"{PPR_NAMESPACE}defines",
                "object_kind": "new_individual",
                "object_individual_index": 1,
                "object_iri": None,
                "evidence_refs": [evidence_ref],
            },
            {
                "subject_kind": "existing_individual",
                "subject_individual_index": None,
                "subject_iri": "https://cais-spade-llm.local/process/assembly",
                "predicate_iri": f"{PPR_NAMESPACE}realizes",
                "object_kind": "new_individual",
                "object_individual_index": 1,
                "object_iri": None,
                "evidence_refs": [evidence_ref],
            },
        ],
        "literal_facts": [],
        "context_summary": "Evidence supports one requested assembly feature.",
        "evidence_refs": [evidence_ref],
        "missing_information": [],
    }

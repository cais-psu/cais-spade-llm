"""Focused tests for native tool-using ProductAgent grounding."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rdflib import Namespace, URIRef
from rdflib.namespace import RDF

from cais_spade_llm.spec2primitives.agents.pa import production_grounding
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingProducerDescriptor,
    build_product_context_view,
)
from cais_spade_llm.spec2primitives.agents.pa.ontology_grounding import (
    OntologyGroundingCandidate,
    OntologyGroundingError,
    OntologyGroundingInterruption,
    OntologyGroundingProposal,
    OntologyGroundingResult,
    commit_ontology_grounding_candidate,
    propose_and_validate_ontology_grounding,
    review_target_feature_semantics,
)
from cais_spade_llm.spec2primitives.agents.pa.presentation_records import (
    AllocationEvidenceEntry,
    EvidencePresentationRecord,
    load_or_create_evidence_presentation,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    initialize_interaction_abox,
    load_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.agents.pa.production_grounding import (
    ProductionGroundingError,
    ProductionProductContextGroundingRuntime,
    _allocation_entry_for_binding,
    _append_validation_feedback,
    _approved_evidence_handles,
    _approved_evidence_sources,
    _cad_size_comparison_projection,
    _CADComparisonBinding,
    _clarification_requests_system_choice,
    _compare_cad_size_tool,
    _EvidenceHandle,
    _NativeEvidenceInvestigation,
    _neutral_candidate_views,
    _pa_allocation_prompt,
    _producer_descriptors,
    _proposal_cad_bindings,
    _proposal_state_candidate_bindings,
    _raw_evidence_types_for_gap,
    _required_record_plan,
    _retrieve_tool,
    _StateCandidateBinding,
    _validate_pa_state_cad_comparison,
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


def _presentation_for_handles(
    root: Path,
    handles: tuple[_EvidenceHandle, ...],
) -> EvidencePresentationRecord:
    sources = tuple(
        (handle.context_ref, handle.evidence_type, handle.source_revision) for handle in handles
    )
    explicit_order = tuple(handle.context_ref or "__live_observation__" for handle in handles)
    return load_or_create_evidence_presentation(
        root,
        sources=sources,
        explicit_order=explicit_order,
    )


def _default_handles(root: Path) -> tuple[EvidencePresentationRecord, tuple[_EvidenceHandle, ...]]:
    sources = _approved_evidence_sources()
    presentation = load_or_create_evidence_presentation(
        root,
        sources=sources,
        explicit_order=tuple(source_ref or "__live_observation__" for source_ref, _, _ in sources),
    )
    return presentation, _approved_evidence_handles(presentation)


def _allow_unrelated_test_through_selected_candidate_gate(
    runtime: ProductionProductContextGroundingRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep legacy semantic/clarification tests focused on their own boundary."""
    binding = SimpleNamespace(
        state="current_state",
        name="selected_candidate",
        record_ref="selected_segmentation.json",
        field_path="/cameras/0/candidates/0",
    )
    monkeypatch.setattr(
        production_grounding,
        "_proposal_state_candidate_bindings",
        lambda proposal: (binding, binding),
    )

    def accepted_correspondence(**kwargs: object) -> None:
        del kwargs
        return None

    monkeypatch.setattr(
        runtime,
        "_validate_pa_state_cad_comparisons",
        accepted_correspondence,
    )


def test_pa_candidate_projection_removes_semantic_sensor_shortcuts() -> None:
    segmentation = {
        "cameras": [
            {
                "camera_id": "semantic_camera_name",
                "role": "assembly_target",
                "source": "Gazebo_ground_truth",
                "observation_handle": "view_0001",
                "candidate_count": 1,
                "candidates": [
                    {
                        "candidate_handle": "candidate_0001_0001",
                        "point_count": 100,
                        "pixel_bounds_uv": {
                            "minimum": [1, 2],
                            "maximum": [3, 4],
                        },
                        "bounds_m": {
                            "minimum": [0.0, 0.0, 0.0],
                            "maximum": [0.1, 0.1, 0.1],
                        },
                        "centroid_m": [0.05, 0.05, 0.05],
                        "depth_range_m": {"minimum": 0.4, "maximum": 0.5},
                        "evaluator_label": "desired region",
                    }
                ],
            }
        ]
    }

    projected = _neutral_candidate_views(
        segmentation,
        segmentation_record_ref="products/grounding/segmentation_record.json",
        review={
            "record_type": "ObservationCandidateReview",
            "status": "accepted",
            "candidates": [
                {
                    "observation_handle": "view_0001",
                    "candidate_handle": "candidate_0001_0001",
                    "description": "visible toothed circular part",
                    "uncertainty": "identity not assigned",
                }
            ],
        },
    )

    serialized = json.dumps(projected)
    assert "candidate_0001_0001" in serialized
    for forbidden in (
        "assembly_target",
        "semantic_camera_name",
        "Gazebo_ground_truth",
        "evaluator_label",
        "desired region",
        "camera_id",
        "role",
        "source",
    ):
        assert forbidden not in serialized


def test_allocation_prompt_reuses_approved_evidence_and_links_segmentation_candidate() -> None:
    canonical_record_ref = (
        "products/grounding/rgb_d_cad_grounding/segmentation_0001/rgbd_segmentation_record.json"
    )
    opaque_record_ref = "typed_record_0003"
    field_path = "/cameras/0/candidates/1"
    observation_result = {
        "evidence_type": "observation",
        "evidence_handle": "observation_candidate_0001",
        "evidence_refs": ["citation_0003", opaque_record_ref],
        "record_refs": ["typed_record_0002", opaque_record_ref],
        "segmentation": {
            "candidate_state": "available",
            "candidate_count": 2,
            "views": [
                {
                    "observation_handle": "view_0001",
                    "candidates": [
                        {
                            "candidate_handle": "candidate_0001_0002",
                            "candidate_value_ref": {
                                "record_ref": opaque_record_ref,
                                "field_path": field_path,
                            },
                        }
                    ],
                }
            ],
        },
    }
    presentation = SimpleNamespace(entries=())
    investigation = SimpleNamespace(
        presentation=presentation,
        prior_evidence=[
            {
                "retrieval_state": "already_retrieved",
                "evidence_type": "document",
                "evidence_handle": "document_candidate_0001",
                "record_refs": ["typed_record_0001"],
                "summary": "Approved assembly instructions.",
            }
        ],
        retrieved_results={
            "cad_candidate_0001": {
                "evidence_type": "CAD",
                "evidence_handle": "cad_candidate_0001",
                "record_refs": ["typed_record_0004"],
                "dimensions": {"size": [0.04, 0.04, 0.01]},
            },
            "observation_candidate_0001": observation_result,
        },
        project_canonical_reference=lambda record_ref: (
            opaque_record_ref
            if record_ref == canonical_record_ref
            else (_ for _ in ()).throw(AssertionError(record_ref))
        ),
    )
    entry = AllocationEvidenceEntry(
        pa_handle="state_evidence_0001",
        canonical_key=f"{canonical_record_ref}#{field_path}",
        record_type="RGBDSegmentationRecord",
        record_ref=canonical_record_ref,
        record_sha256="a" * 64,
        field_path=field_path,
        observation_handle="view_0001",
        candidate_handle="candidate_0001_0002",
        source_frame="camera_frame",
        neutral_projection={
            "visual_region_available": True,
            "point_count": 400,
        },
    )
    allocation_presentation = SimpleNamespace(
        evidence_entries=(entry,),
        resource_order=("xarm6", "ur5e"),
    )

    prompt = _pa_allocation_prompt(
        requirement="assemble medium gear",
        target_feature={"feature_iri": "urn:feature:1"},
        resource_catalog={
            "xarm6": {
                "resource_iri": "urn:resource:xarm6",
                "resource_jid": "xarm6@localhost",
            },
            "ur5e": {
                "resource_iri": "urn:resource:ur5e",
                "resource_jid": "ur5e@localhost",
            },
        },
        allocation_presentation=allocation_presentation,
        investigation=investigation,
        current_state_evidence=entry,
        desired_state_evidence=entry,
        validation_feedback=None,
    )

    prompt_input = json.loads(prompt.split("Allocation input:\n", maxsplit=1)[1])
    assert {item["evidence_type"] for item in prompt_input["approved_retrieved_evidence"]} == {
        "document",
        "CAD",
        "observation",
    }
    state_evidence = prompt_input["grounded_state_evidence"]["current_state"]
    assert state_evidence["observation_handle"] == "view_0001"
    assert state_evidence["candidate_handle"] == "candidate_0001_0002"
    assert state_evidence["candidate_value_ref"] == {
        "record_ref": opaque_record_ref,
        "field_path": field_path,
    }
    assert (
        state_evidence["candidate_value_ref"]
        == observation_result["segmentation"]["views"][0]["candidates"][0]["candidate_value_ref"]
    )
    assert canonical_record_ref not in prompt
    assert "Gazebo_ground_truth" not in prompt


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
        self.review_calls: list[dict[str, object]] = []

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]]
        | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        if response_format["name"] == "spec2primitives_target_feature_review":
            self.review_calls.append({"prompt": prompt, "response_format": response_format})
            return {"verdict": "complete", "gap": None}
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
        tool_executor: Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]]
        | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        if response_format["name"] == "spec2primitives_target_feature_review":
            self.review_calls.append({"prompt": prompt, "response_format": response_format})
            return {"verdict": "complete", "gap": None}
        self.calls.append(
            {
                "prompt": prompt,
                "response_format": response_format,
                "tools": tools,
                "max_tool_rounds": max_tool_rounds,
            }
        )
        return {"result": dict(self.results[len(self.calls) - 1])}


class _RevisingAllocationAgent:
    def __init__(self) -> None:
        self.resource_choices = ["xarm6", "ur5e"]
        self.prompts: list[str] = []

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]]
        | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del max_tool_rounds
        assert response_format["name"] == "spec2primitives_pa_resource_allocation"
        assert tool_executor is not None
        assert tools is not None
        parameters = tools[0]["function"]["parameters"]
        assert set(parameters["properties"]) == {"resource_symbol"}
        assert parameters["required"] == ["resource_symbol"]
        resource_symbol = self.resource_choices[len(self.prompts)]
        self.prompts.append(prompt)
        reachability = await tool_executor(
            "check_reachability",
            {"resource_symbol": resource_symbol},
        )
        assert reachability["status"] == "accepted"
        return {
            "result": {
                "provisional_resource_symbol": resource_symbol,
                "reachability_check_ref": reachability["reachability_check_ref"],
            }
        }


class _RejectXarmAcceptUr5Feasibility:
    def __init__(self) -> None:
        self.resources: list[str] = []

    async def validate_plan_only_allocation(
        self,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        resource_symbol = str(request["resource_symbol"])
        self.resources.append(resource_symbol)
        status = "rejected" if resource_symbol == "xarm6" else "accepted"
        endpoint = {
            "status": status,
            "message": f"{resource_symbol} plan-only {status}",
            "error_code": -1 if status == "rejected" else 1,
        }
        return {
            "status": status,
            "current_state": dict(endpoint),
            "desired_state": dict(endpoint),
            "feedback": endpoint["message"] if status == "rejected" else None,
        }


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
    semantic_review = asyncio.run(
        review_target_feature_semantics(
            agent,
            interaction_root=root,
            candidate=result,
            product_requirement=abox.product_requirement,
            evidence_catalog=[],
        )
    )
    committed = commit_ontology_grounding_candidate(
        result,
        interaction_root=root,
        tbox=tbox,
        abox=abox,
        workcell=workcell,
        authorized_evidence_refs=authorized,
        semantic_review=semantic_review,
    )
    assert isinstance(committed, OntologyGroundingResult)
    assert committed.merge.abox.delta_count == 1
    ppr = Namespace(PPR_NAMESPACE)
    feature = URIRef(f"{abox.namespace}feature_0001")
    currentstate = URIRef(f"{abox.namespace}currentstate_0001")
    desiredstate = URIRef(f"{abox.namespace}desiredstate_0001")
    assert set(committed.merge.abox.graph) - set(abox.graph) == {
        (feature, RDF.type, ppr.feature),
        (URIRef(abox.specification_iri), ppr.defines, feature),
        (URIRef(workcell.processes[0][1]), ppr.realizes, feature),
        (currentstate, RDF.type, ppr.state),
        (desiredstate, RDF.type, ppr.state),
        (feature, ppr.hascurrentstate, currentstate),
        (feature, ppr.hasdesiredstate, desiredstate),
    }
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
    assert proposal_schema["required"] == ["target_feature"]
    target_schema = proposal_schema["properties"]["target_feature"]
    assert set(target_schema["properties"]) == {
        "required_process",
        "current_state",
        "desired_state",
    }
    assert target_schema["properties"]["required_process"]["properties"]["process_iri"]["enum"] == [
        "https://cais-spade-llm.local/process/assembly"
    ]
    for state_name in ("current_state", "desired_state"):
        assert set(target_schema["properties"][state_name]["properties"]) == {
            "statement",
            "state_values",
        }
    serialized = json.dumps(call)
    for removed_protocol in (
        "next_action",
        "propose_grounding",
        '"inspect"',
        '"individuals"',
        '"relations"',
        '"context_summary"',
        '"missing_information"',
    ):
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


@pytest.mark.parametrize("state_name", ["current_state", "desired_state"])
@pytest.mark.parametrize("state_value_count", [0, 1, 2])
def test_target_feature_supports_zero_one_or_multiple_state_values(
    tmp_path: Path,
    state_name: str,
    state_value_count: int,
) -> None:
    root = tmp_path / f"{state_name}_{state_value_count}"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "paint medium gear", tbox)
    workcell = load_predefined_workcell(
        tbox,
        load_predefined_resource_registry(tbox),
    )
    proposal = _proposal("requirement_0001")
    values = [
        {
            "name": name,
            "value_ref": {
                "record_ref": "products/grounding/test/specification.json",
                "field_path": f"/{state_name}/{field_path}",
            },
            "evidence_refs": ["requirement_0001"],
        }
        for name, field_path in (
            ("specified_finish", "finish"),
            ("specified_coating", "coating"),
        )[:state_value_count]
    ]
    proposal["target_feature"][state_name]["state_values"] = values
    exact_record = {state_name: {"finish": "matte", "coating": "primer"}}

    def resolver(record_ref: str) -> Mapping[str, object]:
        assert record_ref == "products/grounding/test/specification.json"
        return {
            "record_type": "DocumentOverviewRecord",
            "record_sha256": "a" * 64,
            "record": exact_record,
        }

    result = asyncio.run(
        propose_and_validate_ontology_grounding(
            _ToolUsingProductAgent(retrieve_order=(), final_result=proposal),
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            workcell=workcell,
            evidence_catalog=[],
            authorized_evidence_refs={"requirement_0001"},
            tools=[],
            tool_executor=_unused_tool,
            max_tool_rounds=1,
            required_output_projection={},
            typed_record_resolver=resolver,
        )
    )

    assert isinstance(result, OntologyGroundingCandidate)
    assert result.output == proposal
    assert len(result.proposal.resolved_state_values) == state_value_count
    if state_value_count == 2:
        assert {item["state"] for item in result.proposal.resolved_state_values} == {state_name}
        assert [item["resolved_value"] for item in result.proposal.resolved_state_values] == [
            "matte",
            "primer",
        ]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("empty_statement", "statement text is empty"),
        ("duplicate_names", "state value names must be unique"),
        ("unsupported_process", "not an authorized process"),
        ("unauthorized_evidence", "unique authorized direct evidence refs"),
        ("missing_record", "not an accepted typed binding"),
        ("invalid_pointer", "field_path does not exist"),
        ("empty_value", "resolves to an empty value"),
    ],
)
def test_invalid_target_feature_fails_before_commit(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    root = tmp_path / case
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble medium gear", tbox)
    workcell = load_predefined_workcell(
        tbox,
        load_predefined_resource_registry(tbox),
    )
    proposal = _proposal("requirement_0001")
    value = {
        "name": "specified_finish",
        "value_ref": {
            "record_ref": "products/grounding/test/specification.json",
            "field_path": "/desired/finish",
        },
        "evidence_refs": ["requirement_0001"],
    }
    if case == "empty_statement":
        proposal["target_feature"]["desired_state"]["statement"]["text"] = " "
    elif case == "duplicate_names":
        proposal["target_feature"]["desired_state"]["state_values"] = [
            value,
            dict(value),
        ]
    elif case == "unsupported_process":
        proposal["target_feature"]["required_process"]["process_iri"] = (
            "https://cais-spade-llm.local/process/painting"
        )
    elif case == "unauthorized_evidence":
        proposal["target_feature"]["desired_state"]["statement"]["evidence_refs"] = [
            "manual_page_0004"
        ]
    else:
        proposal["target_feature"]["desired_state"]["state_values"] = [value]
        if case == "invalid_pointer":
            value["value_ref"]["field_path"] = "/desired/unknown"

    def resolver(record_ref: str) -> Mapping[str, object]:
        del record_ref
        if case == "missing_record":
            raise ValueError("not accepted")
        return {
            "record_type": "DocumentOverviewRecord",
            "record_sha256": "a" * 64,
            "record": {"desired": {"finish": "" if case == "empty_value" else "matte"}},
        }

    with pytest.raises(OntologyGroundingError, match=message):
        asyncio.run(
            propose_and_validate_ontology_grounding(
                _ToolUsingProductAgent(retrieve_order=(), final_result=proposal),
                interaction_root=root,
                tbox=tbox,
                abox=abox,
                workcell=workcell,
                evidence_catalog=[],
                authorized_evidence_refs={"requirement_0001"},
                tools=[],
                tool_executor=_unused_tool,
                max_tool_rounds=1,
                required_output_projection={},
                typed_record_resolver=resolver,
            )
        )
    rejected = json.loads(
        (root / "products/grounding/ontology_grounding/proposal_0001.json").read_text(
            encoding="utf-8"
        )
    )
    assert rejected["schema_version"] == 8
    assert rejected["status"] == "rejected"


def test_retrieve_tool_exposes_only_prompt_local_evidence_id(tmp_path: Path) -> None:
    _presentation, handles = _default_handles(tmp_path)
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


def test_compare_cad_size_tool_exposes_only_typed_opaque_input_handles(
    tmp_path: Path,
) -> None:
    _presentation, handles = _default_handles(tmp_path)
    tool = _compare_cad_size_tool(handles)

    assert tool["function"]["name"] == "compare_cad_size"
    parameters = tool["function"]["parameters"]
    assert parameters["required"] == ["cad_evidence_id", "observation_evidence_id"]
    by_id = {handle.evidence_id: handle for handle in handles}
    assert all(
        by_id[evidence_id].evidence_type == "CAD"
        for evidence_id in parameters["properties"]["cad_evidence_id"]["enum"]
    )
    assert all(
        by_id[evidence_id].evidence_type == "observation"
        for evidence_id in parameters["properties"]["observation_evidence_id"]["enum"]
    )
    serialized = json.dumps(tool)
    for forbidden in ("camera_id", "candidate_index", "selected_candidate", "state_role"):
        assert forbidden not in serialized


def test_compare_cad_size_rejects_unauthorized_unretrieved_and_stale_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble medium gear", tbox)
    handles = (
        _EvidenceHandle(
            evidence_id="cad_opaque",
            evidence_type="CAD",
            display_name="cad_opaque",
            context_ref="synthetic.stl",
            source_revision="a" * 64,
        ),
        _EvidenceHandle(
            evidence_id="observation_opaque",
            evidence_type="observation",
            display_name="observation_opaque",
            context_ref=None,
            source_revision="fresh_on_call",
        ),
    )
    runtime = ProductionProductContextGroundingRuntime(
        tbox=tbox,
        document_config=load_model_runtime_config().document_vlm,
        document_vision_runtime=_NoDocumentVision(),
    )
    investigation = _NativeEvidenceInvestigation(
        runtime=runtime,
        interaction_root=root,
        tbox=tbox,
        abox=abox,
        requirement=abox.product_requirement,
        handles=handles,
        presentation=_presentation_for_handles(root, handles),
    )
    monkeypatch.setattr(production_grounding, "_handle_revision_is_current", lambda handle: True)

    unauthorized = asyncio.run(
        investigation.execute(
            "compare_cad_size",
            {
                "cad_evidence_id": "not_catalogued",
                "observation_evidence_id": "observation_opaque",
            },
        )
    )
    unretrieved = asyncio.run(
        investigation.execute(
            "compare_cad_size",
            {
                "cad_evidence_id": "cad_opaque",
                "observation_evidence_id": "observation_opaque",
            },
        )
    )
    investigation.retrieved_results.update(
        {"cad_opaque": {"record_refs": []}, "observation_opaque": {"record_refs": []}}
    )
    monkeypatch.setattr(
        production_grounding,
        "_handle_revision_is_current",
        lambda handle: False,
    )
    stale = asyncio.run(
        investigation.execute(
            "compare_cad_size",
            {
                "cad_evidence_id": "cad_opaque",
                "observation_evidence_id": "observation_opaque",
            },
        )
    )

    assert unauthorized["error"]["reason"] == "unauthorized_cad_evidence_id"
    assert unretrieved["error"]["reason"] == "comparison_prerequisite_not_retrieved"
    assert stale["error"]["reason"] == "stale_comparison_input"
    audits = sorted((root / "interaction_record").glob("tool_call_*.json"))
    assert len(audits) == 3
    assert all(json.loads(path.read_text())["tool_name"] == "compare_cad_size" for path in audits)


def test_cad_size_projection_is_measurement_only_and_answer_blind() -> None:
    comparison_ref = "products/grounding/correspondence_0001.json"
    investigation = SimpleNamespace(
        presentation=SimpleNamespace(entries=()),
        project_canonical_reference=lambda ref: (
            "typed_record_comparison" if ref == comparison_ref else None
        ),
    )
    observation_result = {
        "evidence_refs": ["typed_record_observation"],
        "segmentation": {
            "views": [
                {
                    "observation_handle": "view_opaque",
                    "candidates": [
                        {
                            "candidate_handle": "candidate_opaque",
                            "candidate_value_ref": {
                                "record_ref": "typed_record_segmentation",
                                "field_path": "/cameras/0/candidates/0",
                            },
                        }
                    ],
                }
            ]
        },
    }
    record = {
        "CAD": {
            "context_ref": "Gear_Shaft.STL",
            "compared_dimensions_m": [0.020, 0.010],
        },
        "parameters": {"dimension_error_limit": 0.15},
        "CAD_correspondence": "accepted",
        "ranked_candidates": [
            {
                "rank": 1,
                "observation_handle": "view_opaque",
                "camera_id": "cam_mk4_1",
                "frame": "simulator_frame",
                "candidate_handle": "candidate_opaque",
                "candidate_id": 3,
                "observed_dimensions_m": [0.0201, 0.0102],
                "dimension_errors": [0.005, 0.02],
                "mean_dimension_error": 0.0125,
                "within_size_tolerance": True,
                "candidate_center_m": [0.1, 0.2, 0.3],
                "inferred_identity": "pin",
            }
        ],
    }

    result = _cad_size_comparison_projection(
        investigation,
        record,
        comparison_ref=comparison_ref,
        cad_evidence_id="cad_opaque",
        observation_evidence_id="observation_opaque",
        cad_result={"evidence_refs": ["typed_record_cad"]},
        observation_result=observation_result,
    )

    assert result["selection_made_by_tool"] is False
    candidate = result["ranked_candidates"][0]
    assert set(candidate) == {
        "rank",
        "observation_handle",
        "candidate_handle",
        "candidate_value_ref",
        "observed_dimensions_m",
        "dimension_errors",
        "mean_dimension_error",
        "within_size_tolerance",
        "candidate_center_m",
    }
    serialized = json.dumps(result)
    for forbidden in (
        "cam_mk4_1",
        "simulator_frame",
        "Gear_Shaft.STL",
        "inferred_identity",
        '"pin"',
        "selected_candidate",
    ):
        assert forbidden not in serialized


def test_ambiguous_cad_projection_exposes_neutral_plausible_order() -> None:
    comparison_ref = "products/grounding/correspondence_0001.json"
    investigation = SimpleNamespace(
        presentation=SimpleNamespace(entries=()),
        project_canonical_reference=lambda ref: (
            "typed_record_comparison" if ref == comparison_ref else None
        ),
    )
    candidate_handles = ["candidate_a", "candidate_b", "candidate_c"]
    observation_result = {
        "evidence_refs": ["typed_record_observation"],
        "segmentation": {
            "views": [
                {
                    "observation_handle": "view_opaque",
                    "candidates": [
                        {
                            "candidate_handle": handle,
                            "candidate_value_ref": {
                                "record_ref": "typed_record_segmentation",
                                "field_path": f"/cameras/0/candidates/{index}",
                            },
                        }
                        for index, handle in enumerate(candidate_handles)
                    ],
                }
            ]
        },
    }
    ranked_candidates = [
        {
            "rank": rank,
            "observation_handle": "view_opaque",
            "candidate_handle": handle,
            "observed_dimensions_m": [0.020 + rank / 10000, 0.010],
            "dimension_errors": [rank / 1000, 0.0],
            "mean_dimension_error": rank / 2000,
            "within_size_tolerance": True,
            "candidate_center_m": [float(rank), 0.0, 0.0],
        }
        for rank, handle in enumerate(reversed(candidate_handles), start=1)
    ]
    record = {
        "CAD": {"compared_dimensions_m": [0.020, 0.010]},
        "parameters": {"dimension_error_limit": 0.15},
        "CAD_correspondence": "ambiguous",
        "ranked_candidates": ranked_candidates,
    }

    result = _cad_size_comparison_projection(
        investigation,
        record,
        comparison_ref=comparison_ref,
        cad_evidence_id="cad_opaque",
        observation_evidence_id="observation_opaque",
        cad_result={"evidence_refs": ["typed_record_cad"]},
        observation_result=observation_result,
    )

    assert "ranked_candidates" not in result
    assert [
        candidate["candidate_handle"] for candidate in result["plausible_candidates"]
    ] == candidate_handles
    assert result["plausible_candidate_count"] == 3
    assert all(
        {"rank", "dimension_errors", "mean_dimension_error"}.isdisjoint(candidate)
        for candidate in result["plausible_candidates"]
    )


def test_compare_cad_size_persists_result_and_audits_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble medium gear", tbox)
    handles = (
        _EvidenceHandle(
            evidence_id="cad_opaque",
            evidence_type="CAD",
            display_name="cad_opaque",
            context_ref="synthetic.stl",
            source_revision="a" * 64,
        ),
        _EvidenceHandle(
            evidence_id="observation_opaque",
            evidence_type="observation",
            display_name="observation_opaque",
            context_ref=None,
            source_revision="fresh_on_call",
        ),
    )
    runtime = ProductionProductContextGroundingRuntime(
        tbox=tbox,
        document_config=load_model_runtime_config().document_vlm,
        document_vision_runtime=_NoDocumentVision(),
    )
    investigation = _NativeEvidenceInvestigation(
        runtime=runtime,
        interaction_root=root,
        tbox=tbox,
        abox=abox,
        requirement=abox.product_requirement,
        handles=handles,
        presentation=_presentation_for_handles(root, handles),
    )
    cad_ref = "products/grounding/test/cad.json"
    segmentation_ref = "products/grounding/test/segmentation.json"
    for relative, record_type in (
        (cad_ref, "CADMeshRecord"),
        (segmentation_ref, "RGBDSegmentationRecord"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"record_type": record_type}), encoding="utf-8")
    investigation._register_references(
        source_refs=(),
        record_refs=(cad_ref, segmentation_ref),
    )
    cad_pa_ref = investigation.project_canonical_reference(cad_ref)
    segmentation_pa_ref = investigation.project_canonical_reference(segmentation_ref)
    investigation.retrieved_results.update(
        {
            "cad_opaque": {
                "evidence_refs": [cad_pa_ref],
                "record_refs": [cad_pa_ref],
            },
            "observation_opaque": {
                "evidence_refs": [segmentation_pa_ref],
                "record_refs": [segmentation_pa_ref],
                "segmentation": {
                    "views": [
                        {
                            "observation_handle": "view_opaque",
                            "candidates": [
                                {
                                    "candidate_handle": "candidate_opaque",
                                    "candidate_value_ref": {
                                        "record_ref": segmentation_pa_ref,
                                        "field_path": "/cameras/0/candidates/0",
                                    },
                                }
                            ],
                        }
                    ]
                },
            },
        }
    )
    accepted_bindings = (
        SimpleNamespace(
            record_ref=cad_ref,
            record_type="CADMeshRecord",
            status="accepted",
        ),
        SimpleNamespace(
            record_ref=segmentation_ref,
            record_type="RGBDSegmentationRecord",
            status="accepted",
        ),
    )
    monkeypatch.setattr(production_grounding, "_handle_revision_is_current", lambda handle: True)
    monkeypatch.setattr(
        production_grounding,
        "build_product_context_view",
        lambda *args, **kwargs: SimpleNamespace(typed_bindings=accepted_bindings),
    )

    async def run_immediately(
        function: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        return function(*args, **kwargs)

    monkeypatch.setattr(production_grounding.asyncio, "to_thread", run_immediately)
    matcher_calls = 0

    def match_size(**kwargs: object) -> SimpleNamespace:
        nonlocal matcher_calls
        del kwargs
        matcher_calls += 1
        record_path = (
            root
            / "products/grounding/rgb_d_cad_grounding"
            / "correspondence_0001"
            / "correspondence_record.json"
        )
        record_path.parent.mkdir(parents=True)
        record = {
            "record_type": "CADSizeCorrespondenceRecord",
            "CAD": {
                "record": {"ref": cad_ref},
                "compared_dimensions_m": [0.020, 0.010],
            },
            "segmentation": {"record": {"ref": segmentation_ref}},
            "parameters": {"dimension_error_limit": 0.15},
            "CAD_correspondence": "accepted",
            "ranked_candidates": [
                {
                    "rank": 1,
                    "observation_handle": "view_opaque",
                    "candidate_handle": "candidate_opaque",
                    "observed_dimensions_m": [0.020, 0.010],
                    "dimension_errors": [0.0, 0.0],
                    "mean_dimension_error": 0.0,
                    "within_size_tolerance": True,
                    "candidate_center_m": [0.1, 0.2, 0.3],
                }
            ],
            "plausible_candidates": [],
            "selected_candidate": {
                "observation_handle": "view_opaque",
                "candidate_handle": "candidate_opaque",
            },
        }
        record_path.write_text(json.dumps(record), encoding="utf-8")
        return SimpleNamespace(
            record_path=record_path,
            record=record,
            CAD_correspondence="accepted",
        )

    monkeypatch.setattr(production_grounding, "associate_segmented_candidate_by_size", match_size)
    monkeypatch.setattr(
        production_grounding,
        "_merge_derived_record",
        lambda *args, **kwargs: (abox, SimpleNamespace(status="accepted")),
    )
    arguments = {
        "cad_evidence_id": "cad_opaque",
        "observation_evidence_id": "observation_opaque",
    }

    first = asyncio.run(investigation.execute("compare_cad_size", arguments))
    repeated = asyncio.run(investigation.execute("compare_cad_size", arguments))

    assert repeated == first
    assert matcher_calls == 1
    assert first["selection_made_by_tool"] is False
    audits = sorted((root / "interaction_record").glob("tool_call_*.json"))
    assert len(audits) == 2
    first_audit = json.loads(audits[0].read_text())
    second_audit = json.loads(audits[1].read_text())
    assert first_audit["record_type"] == "ProductAgentCADComparisonToolCall"
    assert first_audit["reused"] is False
    assert second_audit["reused"] is True
    assert first_audit["result_ref"].endswith("correspondence_record.json")


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
        presentation=_presentation_for_handles(root, (stale,)),
    )

    unknown = asyncio.run(investigation.execute("retrieve", {"evidence_id": "not_catalogued"}))
    stale_result = asyncio.run(investigation.execute("retrieve", {"evidence_id": "evidence_stale"}))

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
        presentation=_presentation_for_handles(root, handles),
    )

    first_result = asyncio.run(first.execute("retrieve", {"evidence_id": medium.evidence_id}))
    repeated_result = asyncio.run(first.execute("retrieve", {"evidence_id": medium.evidence_id}))
    assert first_result == repeated_result
    serialized_result = json.dumps(first_result)
    assert '"context_ref": "synthetic.stl"' in serialized_result
    assert "/home/" not in serialized_result
    assert "synthetic static CAD" not in serialized_result
    assert "CAD_identity" not in serialized_result
    assert first_result["evidence_handle"] == medium.evidence_id
    assert all(
        str(record_ref).startswith("typed_record_") for record_ref in first_result["record_refs"]
    )
    assert retrieval_calls == [1]
    assert (
        json.loads((root / "interaction_record/tool_call_0002.json").read_text())["reused"] is True
    )

    resumed = _NativeEvidenceInvestigation(
        runtime=runtime,
        interaction_root=root,
        tbox=tbox,
        abox=abox,
        requirement=abox.product_requirement,
        handles=handles,
        presentation=_presentation_for_handles(root, handles),
    )
    assert resumed.prior_evidence[0]["retrieval_state"] == "already_retrieved"
    resumed_result = asyncio.run(resumed.execute("retrieve", {"evidence_id": medium.evidence_id}))
    assert resumed_result == first_result
    assert retrieval_calls == [1]
    assert (
        json.loads((root / "interaction_record/tool_call_0003.json").read_text())["reused"] is True
    )


def test_live_observation_is_reused_within_run_and_refreshed_after_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble medium gear", tbox)
    observation = _EvidenceHandle(
        evidence_id="observation_opaque",
        evidence_type="observation",
        display_name="observation_opaque",
        context_ref=None,
        source_revision="fresh_on_call",
    )
    handles = (observation,)
    capture_count = 0

    def retrieve_fixture(
        interaction_root: Path,
        *,
        product_requirement: str,
        evidence_type: str,
        context_ref: str | None,
        retrieval_number: int,
        live_observation_timeout_sec: float,
    ) -> Mapping[str, object]:
        nonlocal capture_count
        del interaction_root, product_requirement, context_ref, retrieval_number
        del live_observation_timeout_sec
        capture_count += 1
        assert evidence_type == "observation"
        return {
            "served_context": {
                "evidence_type": "observation",
                "observation_ref": f"observation_{capture_count:04d}",
            }
        }

    class ObservationFixtureRuntime:
        async def interpret_retrieved_evidence(
            self,
            *,
            interaction_root: Path,
            tbox: object,
            abox: object,
            served_context: Mapping[str, object],
            operation_number: int,
        ) -> Mapping[str, object]:
            del tbox, abox, operation_number
            observation_ref = str(served_context["observation_ref"])
            destination = interaction_root / "products/grounding/test" / observation_ref
            destination.mkdir(parents=True)
            segmentation_path = destination / "segmentation.json"
            review_path = destination / "review.json"
            segmentation_path.write_text(
                json.dumps(
                    {
                        "record_type": "RGBDSegmentationRecord",
                        "candidate_state": "available",
                        "candidate_count": 0,
                        "observation_ref": observation_ref,
                        "cameras": [],
                    }
                ),
                encoding="utf-8",
            )
            review_path.write_text(
                json.dumps(
                    {
                        "record_type": "ObservationCandidateReview",
                        "status": "accepted",
                        "candidates": [],
                    }
                ),
                encoding="utf-8",
            )
            return {
                "typed_context_refs": [
                    segmentation_path.relative_to(interaction_root).as_posix(),
                    review_path.relative_to(interaction_root).as_posix(),
                ]
            }

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
    runtime = ObservationFixtureRuntime()
    first = _NativeEvidenceInvestigation(
        runtime=runtime,
        interaction_root=root,
        tbox=tbox,
        abox=abox,
        requirement=abox.product_requirement,
        handles=handles,
        presentation=_presentation_for_handles(root, handles),
    )

    first_result = asyncio.run(first.execute("retrieve", {"evidence_id": observation.evidence_id}))
    repeated_result = asyncio.run(
        first.execute("retrieve", {"evidence_id": observation.evidence_id})
    )
    assert repeated_result == first_result
    assert capture_count == 1

    resumed = _NativeEvidenceInvestigation(
        runtime=runtime,
        interaction_root=root,
        tbox=tbox,
        abox=abox,
        requirement=abox.product_requirement,
        handles=handles,
        presentation=_presentation_for_handles(root, handles),
    )
    assert resumed.prior_evidence == []
    assert resumed.retrieved_results == {}
    resumed_result = asyncio.run(
        resumed.execute("retrieve", {"evidence_id": observation.evidence_id})
    )
    assert resumed_result != first_result
    assert capture_count == 2


def test_descriptor_closure_is_location_driven_and_accepts_synthetic_provider() -> None:
    descriptors = _producer_descriptors(calibration_available=True)

    assert _required_record_plan(descriptors, "RobotFrameLocationRecord") == (
        "ColoredPointCloudSetRecord",
        "RGBDSegmentationRecord",
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
            "prerequisites": {"SyntheticReachRecord": ["RobotFrameLocationRecord"]},
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
    other_path = tmp_path / "products/grounding/cad/other_geometry_record.json"
    other_path.write_text(
        json.dumps({"source": {"context_ref": "Gear_Other.STL"}}),
        encoding="utf-8",
    )
    other_binding = SimpleNamespace(
        record_type="CADMeshRecord",
        status="accepted",
        record_ref=other_path.relative_to(tmp_path).as_posix(),
        evidence_refs=("Gear_Other.STL",),
    )
    view = SimpleNamespace(typed_bindings=(other_binding, binding))
    uncited = SimpleNamespace(
        target_feature={
            "current_state": {
                "statement": {"evidence_refs": ["requirement_0001"]},
                "state_values": [],
            }
        }
    )
    cited = SimpleNamespace(
        target_feature={
            "current_state": {
                "statement": {"evidence_refs": ["Gear_Medium.STL"]},
                "state_values": [],
            }
        }
    )

    assert _proposal_cad_bindings(tmp_path, view, uncited, state_name="current_state") == ()
    assert _proposal_cad_bindings(tmp_path, view, cited, state_name="current_state") == (binding,)


def test_desired_candidate_rejects_ambiguous_size_correspondence(
    tmp_path: Path,
) -> None:
    cad_ref = "products/grounding/cad/shaft.json"
    segmentation_ref = "products/grounding/observation/segmentation.json"
    comparison_ref = "products/grounding/comparison/shaft.json"
    for relative in (cad_ref, segmentation_ref, comparison_ref):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / cad_ref).write_text(
        json.dumps({"source": {"context_ref": "Gear_Shaft.STL"}}),
        encoding="utf-8",
    )
    candidate_handles = [
        "mounted_candidate_0001",
        "mounted_candidate_0002",
        "mounted_candidate_0003",
        "long_cylindrical_candidate_0004",
    ]
    (tmp_path / segmentation_ref).write_text(
        json.dumps(
            {
                "cameras": [
                    {
                        "observation_handle": "view_opaque",
                        "candidates": [
                            {"candidate_handle": handle} for handle in candidate_handles
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    plausible = [
        {
            "observation_handle": "view_opaque",
            "candidate_handle": handle,
            "within_size_tolerance": True,
        }
        for handle in candidate_handles[:3]
    ]
    comparison_record = {
        "record_type": "CADSizeCorrespondenceRecord",
        "CAD": {"record": {"ref": cad_ref}},
        "segmentation": {"record": {"ref": segmentation_ref}},
        "CAD_correspondence": "ambiguous",
        "plausible_candidates": plausible,
        "selected_candidate": None,
    }
    comparison_path = tmp_path / comparison_ref
    comparison_path.write_text(json.dumps(comparison_record), encoding="utf-8")
    cad_binding = SimpleNamespace(
        record_type="CADMeshRecord",
        status="accepted",
        record_ref=cad_ref,
        record_sha256=hashlib.sha256((tmp_path / cad_ref).read_bytes()).hexdigest(),
        evidence_refs=("Gear_Shaft.STL",),
    )
    segmentation_binding = SimpleNamespace(
        record_type="RGBDSegmentationRecord",
        status="accepted",
        record_ref=segmentation_ref,
        record_sha256=hashlib.sha256((tmp_path / segmentation_ref).read_bytes()).hexdigest(),
        evidence_refs=(),
    )
    comparison_typed_binding = SimpleNamespace(
        record_type="CADSizeCorrespondenceRecord",
        status="ambiguous",
        record_ref=comparison_ref,
        record_sha256=hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
        evidence_refs=(),
    )
    view = SimpleNamespace(
        typed_bindings=(cad_binding, segmentation_binding, comparison_typed_binding)
    )
    investigation = SimpleNamespace(
        root=tmp_path,
        comparison_bindings={
            comparison_ref: _CADComparisonBinding(
                record_ref=comparison_ref,
                cad_record_ref=cad_ref,
                segmentation_record_ref=segmentation_ref,
            )
        },
    )

    def proposal(candidate_index: int) -> OntologyGroundingProposal:
        return OntologyGroundingProposal(
            target_feature={
                "desired_state": {
                    "statement": {
                        "text": "The medium gear is assembled at the supported destination.",
                        "evidence_refs": [cad_ref],
                    },
                    "state_values": [
                        {
                            "name": "supported_destination",
                            "value_ref": {
                                "record_ref": segmentation_ref,
                                "field_path": f"/cameras/0/candidates/{candidate_index}",
                            },
                            "evidence_refs": [cad_ref, comparison_ref],
                        }
                    ],
                }
            },
            feature_iri="urn:feature:1",
            resolved_state_values=(),
            evidence_refs=(),
        )

    plausible_binding = _StateCandidateBinding(
        state="desired_state",
        name="supported_destination",
        record_ref=segmentation_ref,
        field_path="/cameras/0/candidates/1",
    )
    gap = _validate_pa_state_cad_comparison(
        investigation,
        view,
        proposal(1),
        state_name="desired_state",
        state_binding=plausible_binding,
    )
    assert gap is not None
    assert "not uniquely supported" in gap
    assert "Gear_Shaft" not in gap
    assert all(handle not in gap for handle in candidate_handles)


def test_current_candidate_still_requires_unique_size_correspondence(
    tmp_path: Path,
) -> None:
    cad_ref = "cad.json"
    segmentation_ref = "segmentation.json"
    comparison_ref = "comparison.json"
    (tmp_path / cad_ref).write_text(
        json.dumps({"source": {"context_ref": "Gear_Medium.STL"}}),
        encoding="utf-8",
    )
    (tmp_path / segmentation_ref).write_text(
        json.dumps(
            {
                "cameras": [
                    {
                        "observation_handle": "view_opaque",
                        "candidates": [
                            {"candidate_handle": "loose_candidate"},
                            {"candidate_handle": "other_candidate"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    comparison = {
        "record_type": "CADSizeCorrespondenceRecord",
        "CAD": {"record": {"ref": cad_ref}},
        "segmentation": {"record": {"ref": segmentation_ref}},
        "CAD_correspondence": "accepted",
        "plausible_candidates": [],
        "selected_candidate": {
            "observation_handle": "view_opaque",
            "candidate_handle": "loose_candidate",
        },
    }
    (tmp_path / comparison_ref).write_text(json.dumps(comparison), encoding="utf-8")
    view = SimpleNamespace(
        typed_bindings=(
            SimpleNamespace(
                record_type="CADMeshRecord",
                status="accepted",
                record_ref=cad_ref,
                record_sha256=hashlib.sha256((tmp_path / cad_ref).read_bytes()).hexdigest(),
                evidence_refs=("Gear_Medium.STL",),
            ),
            SimpleNamespace(
                record_type="RGBDSegmentationRecord",
                status="accepted",
                record_ref=segmentation_ref,
                record_sha256=hashlib.sha256(
                    (tmp_path / segmentation_ref).read_bytes()
                ).hexdigest(),
                evidence_refs=(),
            ),
            SimpleNamespace(
                record_type="CADSizeCorrespondenceRecord",
                status="accepted",
                record_ref=comparison_ref,
                record_sha256=hashlib.sha256((tmp_path / comparison_ref).read_bytes()).hexdigest(),
                evidence_refs=(),
            ),
        )
    )
    investigation = SimpleNamespace(
        root=tmp_path,
        comparison_bindings={
            comparison_ref: _CADComparisonBinding(
                record_ref=comparison_ref,
                cad_record_ref=cad_ref,
                segmentation_record_ref=segmentation_ref,
            )
        },
    )
    proposal = OntologyGroundingProposal(
        target_feature={
            "current_state": {
                "statement": {"text": "The medium gear is loose.", "evidence_refs": [cad_ref]},
                "state_values": [
                    {
                        "name": "medium_gear",
                        "value_ref": {
                            "record_ref": segmentation_ref,
                            "field_path": "/cameras/0/candidates/0",
                        },
                        "evidence_refs": [cad_ref, comparison_ref],
                    }
                ],
            }
        },
        feature_iri="urn:feature:1",
        resolved_state_values=(),
        evidence_refs=(),
    )

    accepted = _validate_pa_state_cad_comparison(
        investigation,
        view,
        proposal,
        state_name="current_state",
        state_binding=_StateCandidateBinding(
            state="current_state",
            name="medium_gear",
            record_ref=segmentation_ref,
            field_path="/cameras/0/candidates/0",
        ),
    )
    rejected = _validate_pa_state_cad_comparison(
        investigation,
        view,
        proposal,
        state_name="current_state",
        state_binding=_StateCandidateBinding(
            state="current_state",
            name="medium_gear",
            record_ref=segmentation_ref,
            field_path="/cameras/0/candidates/1",
        ),
    )

    assert accepted is None
    assert rejected is not None
    assert "not uniquely supported" in rejected


def test_state_candidate_bindings_and_allocation_ignore_presentation_order() -> None:
    current_ref = "products/grounding/segmentation_current.json"
    desired_ref = "products/grounding/segmentation_desired.json"
    current_path = "/cameras/1/candidates/2"
    desired_path = "/cameras/0/candidates/0"
    proposal = OntologyGroundingProposal(
        target_feature={},
        feature_iri="urn:feature:1",
        resolved_state_values=(
            {
                "state": "desired_state",
                "name": "supported_destination",
                "record_type": "RGBDSegmentationRecord",
                "record_sha256": "b" * 64,
                "value_ref": {
                    "record_ref": desired_ref,
                    "field_path": desired_path,
                },
                "resolved_value": {"candidate_handle": "candidate_desired"},
            },
            {
                "state": "current_state",
                "name": "medium_gear",
                "record_type": "RGBDSegmentationRecord",
                "record_sha256": "a" * 64,
                "value_ref": {
                    "record_ref": current_ref,
                    "field_path": current_path,
                },
                "resolved_value": {"candidate_handle": "candidate_current"},
            },
        ),
        evidence_refs=(),
    )

    current, desired = _proposal_state_candidate_bindings(proposal)

    assert (current.name, current.record_ref, current.field_path) == (
        "medium_gear",
        current_ref,
        current_path,
    )
    assert (desired.name, desired.record_ref, desired.field_path) == (
        "supported_destination",
        desired_ref,
        desired_path,
    )
    entries = (
        _allocation_entry("desired", desired_ref, desired_path),
        _allocation_entry("unselected", current_ref, "/cameras/0/candidates/1"),
        _allocation_entry("current", current_ref, current_path),
    )
    presentation = SimpleNamespace(evidence_entries=entries)
    assert _allocation_entry_for_binding(presentation, current).pa_handle == "current"
    assert _allocation_entry_for_binding(presentation, desired).pa_handle == "desired"


def test_state_candidate_binding_rejects_only_incompatible_candidate_reuse() -> None:
    def proposal(current_name: str, desired_name: str) -> OntologyGroundingProposal:
        return OntologyGroundingProposal(
            target_feature={},
            feature_iri="urn:feature:1",
            resolved_state_values=tuple(
                {
                    "state": state,
                    "name": name,
                    "record_type": "RGBDSegmentationRecord",
                    "record_sha256": "a" * 64,
                    "value_ref": {
                        "record_ref": "products/grounding/segmentation.json",
                        "field_path": "/cameras/0/candidates/0",
                    },
                    "resolved_value": {"candidate_handle": "candidate_0001_0001"},
                }
                for state, name in (
                    ("current_state", current_name),
                    ("desired_state", desired_name),
                )
            ),
            evidence_refs=(),
        )

    current, desired = _proposal_state_candidate_bindings(
        proposal("same_supported_value", "same_supported_value")
    )
    assert current.field_path == desired.field_path
    with pytest.raises(ProductionGroundingError, match="incompatible"):
        _proposal_state_candidate_bindings(proposal("medium_gear", "supported_destination"))


def _allocation_entry(
    handle: str,
    record_ref: str,
    field_path: str,
) -> AllocationEvidenceEntry:
    return AllocationEvidenceEntry(
        pa_handle=handle,
        canonical_key=f"{record_ref}#{field_path}",
        record_type="RGBDSegmentationRecord",
        record_ref=record_ref,
        record_sha256="a" * 64,
        field_path=field_path,
        observation_handle=f"view_{handle}",
        candidate_handle=f"candidate_{handle}",
        source_frame="camera_frame",
        neutral_projection={"visual_region_available": True},
    )


def test_completion_requires_pa_selected_state_candidates(
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

    assert result["grounding_status"] == "incomplete"
    assert result["insufficient_evidence"] == (
        "current_state must select exactly one supplied neutral candidate."
    )
    assert len(agent.calls) == 3
    assert "current_validation_gap" not in str(agent.calls[0]["prompt"])
    assert "TargetFeatureGeometryRecord" not in str(agent.calls[0]["prompt"])
    assert load_interaction_abox(root, tbox).accepted_assertion_count == 0
    assert not (root / "products/grounding/ontology_grounding/proposal_0001.json").exists()


@pytest.mark.parametrize("failure_mode", ["ambiguous", "malformed"])
def test_cad_correspondence_failure_remains_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble medium gear", tbox)
    runtime = ProductionProductContextGroundingRuntime(
        tbox=tbox,
        document_config=load_model_runtime_config().document_vlm,
        document_vision_runtime=_NoDocumentVision(),
    )
    binding = SimpleNamespace(
        state="current_state",
        name="medium_gear",
        record_ref="selected_segmentation.json",
        field_path="/cameras/0/candidates/0",
    )
    monkeypatch.setattr(
        production_grounding,
        "_proposal_state_candidate_bindings",
        lambda proposal: (binding, binding),
    )

    def correspondence(**kwargs: object) -> str | None:
        del kwargs
        if failure_mode == "malformed":
            raise ValueError("controlled malformed evidence")
        return (
            "The submitted state assignments are not uniquely supported by complete, "
            "internally consistent, hash-pinned approved evidence."
        )

    monkeypatch.setattr(runtime, "_validate_pa_state_cad_comparisons", correspondence)

    result = asyncio.run(
        runtime.ground_product_context(
            _ToolUsingProductAgent(retrieve_order=()),
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            product_context={},
            max_pa_turns=1,
        )
    )

    assert result["grounding_status"] == "incomplete"
    message = str(result["insufficient_evidence"])
    assert "not uniquely supported" in message
    assert "CAD" not in message
    assert "candidate_" not in message
    assert load_interaction_abox(root, tbox).accepted_assertion_count == 0


def test_robot_agent_rejection_returns_to_pa_without_resource_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    initial_abox = initialize_interaction_abox(
        root,
        "assemble medium gear",
        tbox,
    )
    current_path, current_source = _write_neutral_location(
        root,
        "current",
        (0.0, 0.0, 1.1),
    )
    desired_path, desired_source = _write_neutral_location(
        root,
        "desired",
        (0.0, 0.05, 1.1),
    )
    current_ref = current_path.relative_to(root).as_posix()
    desired_ref = desired_path.relative_to(root).as_posix()
    validate_and_merge_triple_delta(
        root,
        tbox,
        "neutral_test_location_provider",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [current_ref],
        },
        authorized_evidence_refs={current_source},
    )
    location_merge = validate_and_merge_triple_delta(
        root,
        tbox,
        "neutral_test_location_provider",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [desired_ref],
        },
        authorized_evidence_refs={desired_source},
    )
    proposal_value = _proposal("requirement_0001")
    records = {current_ref: current_path, desired_ref: desired_path}

    def resolve_typed_record(record_ref: str) -> Mapping[str, object]:
        path = records[record_ref]
        return {
            "record_type": "RobotFrameLocationRecord",
            "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "record": json.loads(path.read_text(encoding="utf-8")),
        }

    registry = load_predefined_resource_registry(tbox)
    workcell = load_predefined_workcell(tbox, registry)
    proposal_agent = _ToolUsingProductAgent(
        retrieve_order=(),
        final_result=proposal_value,
    )
    candidate = asyncio.run(
        propose_and_validate_ontology_grounding(
            proposal_agent,
            interaction_root=root,
            tbox=tbox,
            abox=location_merge.abox,
            workcell=workcell,
            evidence_catalog=[],
            authorized_evidence_refs={
                "requirement_0001",
                current_ref,
                desired_ref,
            },
            tools=[],
            tool_executor=_unused_tool,
            max_tool_rounds=1,
            required_output_projection={
                "record_type": "RobotFrameLocationRecord",
                "target_frame": "world",
            },
            typed_record_resolver=resolve_typed_record,
        )
    )
    assert isinstance(candidate, OntologyGroundingCandidate)
    semantic_review = asyncio.run(
        review_target_feature_semantics(
            proposal_agent,
            interaction_root=root,
            candidate=candidate,
            product_requirement=initial_abox.product_requirement,
            evidence_catalog=[],
        )
    )
    grounded = commit_ontology_grounding_candidate(
        candidate,
        interaction_root=root,
        tbox=tbox,
        abox=location_merge.abox,
        workcell=workcell,
        authorized_evidence_refs={
            "requirement_0001",
            current_ref,
            desired_ref,
        },
        semantic_review=semantic_review,
    )
    accepted_view = build_product_context_view(
        root,
        grounded.merge.abox,
        attempted_evidence=(),
        assessed_at_ns=1,
    )
    feasibility = _RejectXarmAcceptUr5Feasibility()
    runtime = ProductionProductContextGroundingRuntime(
        tbox=tbox,
        document_config=load_model_runtime_config().document_vlm,
        document_vision_runtime=_NoDocumentVision(),
        robot_agent_feasibility_runtime=feasibility,
    )
    investigation = _NativeEvidenceInvestigation(
        runtime=runtime,
        interaction_root=root,
        tbox=tbox,
        abox=grounded.merge.abox,
        requirement=initial_abox.product_requirement,
        handles=(),
        presentation=_default_handles(root)[0],
    )
    allocation_agent = _RevisingAllocationAgent()
    bindings = (
        SimpleNamespace(
            state="current_state",
            record_ref=current_ref,
            field_path="/translated_location_m",
        ),
        SimpleNamespace(
            state="desired_state",
            record_ref=desired_ref,
            field_path="/translated_location_m",
        ),
    )
    monkeypatch.setattr(
        production_grounding,
        "_proposal_state_candidate_bindings",
        lambda proposal: bindings,
    )

    def selected_location_entry(
        presentation: object,
        binding: object,
    ) -> AllocationEvidenceEntry:
        return next(
            entry
            for entry in presentation.evidence_entries
            if entry.record_ref == binding.record_ref
        )

    monkeypatch.setattr(
        production_grounding,
        "_allocation_entry_for_binding",
        selected_location_entry,
    )
    original_catalog = production_grounding.candidate_resource_catalog

    def physical_mode_catalog(*args: object, **kwargs: object) -> Mapping[str, Mapping[str, str]]:
        catalog = original_catalog(*args, **kwargs)
        return {
            symbol: {**dict(entry), "execution_mode": "physical"}
            for symbol, entry in catalog.items()
        }

    monkeypatch.setattr(
        production_grounding,
        "candidate_resource_catalog",
        physical_mode_catalog,
    )

    result = asyncio.run(
        runtime._complete_resource_assignment(
            product_agent=allocation_agent,
            investigation=investigation,
            root=root,
            tbox=tbox,
            abox=grounded.merge.abox,
            view=accepted_view,
            proposal=grounded.proposal,
            max_pa_turns=3,
        )
    )

    assert result["grounding_status"] == "complete"
    assert result["allocation_label"] == "validated endpoint-motion allocation"
    assert feasibility.resources == ["xarm6", "ur5e"]
    assert len(allocation_agent.prompts) == 2
    assert "xarm6 plan-only rejected" in allocation_agent.prompts[1]
    selection_paths = sorted(
        root.glob(
            "products/grounding/resource_selection/selection_*/resource_selection_record.json"
        )
    )
    assert len(selection_paths) == 2
    rejected_selection = json.loads(selection_paths[0].read_text(encoding="utf-8"))
    accepted_selection = json.loads(selection_paths[1].read_text(encoding="utf-8"))
    assert rejected_selection["provisional_resource_symbol"] == "xarm6"
    assert rejected_selection["selected_resource_symbol"] is None
    assert accepted_selection["provisional_resource_symbol"] == "ur5e"
    assert accepted_selection["selected_resource_symbol"] == "ur5e"
    final_abox = load_interaction_abox(root, tbox)
    execution = URIRef(f"{final_abox.namespace}process_execution_0001")
    assert (
        execution,
        Namespace(PPR_NAMESPACE).runsOnResource,
        URIRef("https://cais-spade-llm.local/resource/ur5e"),
    ) in final_abox.graph
    assert (
        execution,
        Namespace(PPR_NAMESPACE).runsOnResource,
        URIRef("https://cais-spade-llm.local/resource/xarm6"),
    ) not in final_abox.graph


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
    _allow_unrelated_test_through_selected_candidate_gate(runtime, monkeypatch)
    invalid = _proposal("requirement_0001")
    del invalid["target_feature"]["desired_state"]
    agent = _SequencedProductAgent((invalid, _proposal("requirement_0001")))

    async def complete(**kwargs: object) -> Mapping[str, object]:
        del kwargs
        assert load_interaction_abox(root, tbox).accepted_assertion_count == 7
        return {
            "grounding_status": "complete",
            "resource_selection_ref": ("products/grounding/resource_selection/test.json"),
        }

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
    assert "target_feature fields are invalid" in str(agent.calls[1]["prompt"])
    rejected = json.loads(
        (root / "products/grounding/ontology_grounding/proposal_0001.json").read_text(
            encoding="utf-8"
        )
    )
    accepted = json.loads(
        (root / "products/grounding/ontology_grounding/proposal_0002.json").read_text(
            encoding="utf-8"
        )
    )
    assert rejected["status"] == "rejected"
    assert accepted["status"] == "accepted"
    assert load_interaction_abox(root, tbox).accepted_assertion_count == 7


def test_validation_feedback_history_accumulates_and_deduplicates() -> None:
    history: list[Mapping[str, object]] = []
    first = {"kind": "first_gap", "message": "Retrieve comparison evidence."}
    second = {"kind": "second_gap", "message": "Revise the selected destination."}

    _append_validation_feedback(history, first)
    _append_validation_feedback(history, first)
    _append_validation_feedback(history, second)

    assert history == [first, second]


def test_ambiguous_state_evidence_stops_before_review_and_allocation(
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
    binding = SimpleNamespace(
        state="current_state",
        name="selected_candidate",
        record_ref="selected_segmentation.json",
        field_path="/cameras/0/candidates/0",
    )
    monkeypatch.setattr(
        production_grounding,
        "_proposal_state_candidate_bindings",
        lambda proposal: (binding, binding),
    )
    gap = (
        "The submitted desired_state assignment is not uniquely supported by complete, "
        "internally consistent, hash-pinned approved evidence."
    )

    def reject_ambiguous_evidence(**kwargs: object) -> str:
        del kwargs
        return gap

    monkeypatch.setattr(runtime, "_validate_pa_state_cad_comparisons", reject_ambiguous_evidence)

    async def unexpected_allocation(**kwargs: object) -> Mapping[str, object]:
        del kwargs
        raise AssertionError("Allocation must not run after ambiguous state evidence.")

    monkeypatch.setattr(runtime, "_complete_resource_assignment", unexpected_allocation)
    agent = _SequencedProductAgent(
        (
            _proposal("requirement_0001"),
            {"insufficient_evidence": "The approved evidence remains ambiguous."},
        )
    )

    result = asyncio.run(
        runtime.ground_product_context(
            agent,
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            product_context={},
            max_pa_turns=2,
        )
    )

    assert result["grounding_status"] == "incomplete"
    assert result["insufficient_evidence"] == "The approved evidence remains ambiguous."
    assert agent.review_calls == []
    assert not (root / "products/grounding/target_feature_review").exists()
    first_input = json.loads(str(agent.calls[0]["prompt"]).split("Grounding input:\n", 1)[1])
    projection = json.dumps(first_input["required_output_projection"]).casefold()
    feedback_input = json.loads(str(agent.calls[1]["prompt"]).split("Grounding input:\n", 1)[1])
    feedback = json.dumps(feedback_input["validation_feedback_history"]).casefold()
    assert "state_evidence_uniqueness" in feedback
    for forbidden in (
        "gear_medium",
        "gear_shaft",
        "stl",
        "rgb-d",
        "cad",
        "document",
        "target_frame",
        "mounted",
    ):
        assert forbidden not in projection
        assert forbidden not in feedback


def test_distinct_validation_feedback_is_preserved_across_pa_attempts(
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
    _allow_unrelated_test_through_selected_candidate_gate(runtime, monkeypatch)
    invalid = _proposal("requirement_0001")
    del invalid["target_feature"]["desired_state"]

    class CumulativeFeedbackAgent(_SequencedProductAgent):
        async def ask_llm_structured(
            self,
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            tool_executor: Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]]
            | None = None,
            max_tool_rounds: int = 3,
        ) -> dict[str, Any]:
            if response_format["name"] == "spec2primitives_target_feature_review":
                self.review_calls.append({"prompt": prompt, "response_format": response_format})
                if len(self.review_calls) == 1:
                    return {
                        "verdict": "incomplete",
                        "gap": "The desired state needs a supported destination.",
                    }
                return {"verdict": "complete", "gap": None}
            return await super().ask_llm_structured(
                prompt,
                response_format=response_format,
                tools=tools,
                tool_executor=tool_executor,
                max_tool_rounds=max_tool_rounds,
            )

    agent = CumulativeFeedbackAgent(
        (invalid, _proposal("requirement_0001"), _proposal("requirement_0001"))
    )

    async def complete(**kwargs: object) -> Mapping[str, object]:
        del kwargs
        return {
            "grounding_status": "complete",
            "resource_selection_ref": "products/grounding/resource_selection/test.json",
        }

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
    final_prompt = str(agent.calls[2]["prompt"])
    prompt_input = json.loads(final_prompt.split("Grounding input:\n", maxsplit=1)[1])
    assert [item["kind"] for item in prompt_input["validation_feedback_history"]] == [
        "ontology_proposal_validation_error",
        "target_feature_semantic_review",
    ]


def test_tool_round_exhaustion_gets_one_final_no_tools_pa_call(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble medium gear", tbox)
    runtime = ProductionProductContextGroundingRuntime(
        tbox=tbox,
        document_config=load_model_runtime_config().document_vlm,
        document_vision_runtime=_NoDocumentVision(),
    )

    class ExhaustionAgent:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def ask_llm_structured(
            self,
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            tool_executor: Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]]
            | None = None,
            max_tool_rounds: int = 3,
        ) -> dict[str, Any]:
            del response_format, tool_executor
            self.calls.append(
                {
                    "prompt": prompt,
                    "tools": tools,
                    "max_tool_rounds": max_tool_rounds,
                }
            )
            if len(self.calls) == 1:
                raise RuntimeError("Exceeded max tool rounds")
            return {
                "result": {
                    "insufficient_evidence": (
                        "Accumulated evidence does not support a final destination."
                    )
                }
            }

    agent = ExhaustionAgent()
    result = asyncio.run(
        runtime.ground_product_context(
            agent,
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            product_context={},
            max_pa_turns=1,
        )
    )

    assert result == {
        "grounding_status": "incomplete",
        "insufficient_evidence": ("Accumulated evidence does not support a final destination."),
        "tool_call_refs": [],
    }
    assert len(agent.calls) == 2
    assert {tool["function"]["name"] for tool in agent.calls[0]["tools"]} == {
        "retrieve",
        "compare_cad_size",
    }
    assert agent.calls[1]["tools"] == []
    assert agent.calls[1]["max_tool_rounds"] == 0
    assert "tool_budget_exhausted" in str(agent.calls[1]["prompt"])


def test_semantic_review_accepts_supported_destination_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble medium gear", tbox)
    proposal = OntologyGroundingProposal(
        target_feature={
            "required_process": {
                "process_iri": "https://cais-spade-llm.local/process/assembly",
                "evidence_refs": ["requirement_0001"],
            },
            "current_state": {
                "statement": {
                    "text": "The medium gear is currently loose.",
                    "evidence_refs": ["requirement_0001"],
                },
                "state_values": [],
            },
            "desired_state": {
                "statement": {
                    "text": "The medium gear will be assembled at the destination.",
                    "evidence_refs": ["requirement_0001"],
                },
                "state_values": [
                    {
                        "name": "supported_destination",
                        "value_ref": {
                            "record_ref": "typed_record_segmentation",
                            "field_path": "/cameras/0/candidates/0",
                        },
                        "evidence_refs": ["typed_record_comparison"],
                    }
                ],
            },
        },
        feature_iri="urn:feature:1",
        resolved_state_values=(
            {
                "state": "desired_state",
                "name": "supported_destination",
                "record_type": "RGBDSegmentationRecord",
                "record_sha256": "a" * 64,
                "value_ref": {
                    "record_ref": "typed_record_segmentation",
                    "field_path": "/cameras/0/candidates/0",
                },
                "resolved_value": {"candidate_handle": "candidate_opaque"},
            },
        ),
        evidence_refs=("requirement_0001", "typed_record_comparison"),
    )
    candidate = OntologyGroundingCandidate(
        provisional_abox=abox,
        proposal_path=(root / "products/grounding/ontology_grounding/proposal_0001.json"),
        proposal_number=1,
        output={},
        compiled_delta={},
        proposal=proposal,
    )
    agent = _ToolUsingProductAgent(retrieve_order=())

    review = asyncio.run(
        review_target_feature_semantics(
            agent,
            interaction_root=root,
            candidate=candidate,
            product_requirement=abox.product_requirement,
            evidence_catalog=[],
        )
    )

    assert review.verdict == "complete"
    prompt = str(agent.review_calls[0]["prompt"])
    assert "supported_destination" in prompt
    assert "scaffold-free consistency review" in prompt
    assert "required_output_projection" not in prompt
    assert "without depicting the completed assembly" not in prompt


def test_incomplete_semantic_review_reenters_pa_before_commit(
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
    _allow_unrelated_test_through_selected_candidate_gate(runtime, monkeypatch)

    class _ReviewRevisionAgent(_SequencedProductAgent):
        def __init__(self) -> None:
            shallow = _proposal("requirement_0001")
            shallow["target_feature"]["desired_state"]["statement"]["text"] = (
                "An assembly feature is requested."
            )
            super().__init__((shallow, _proposal("requirement_0001")))
            self.review_results = (
                {
                    "verdict": "incomplete",
                    "gap": "The desired state omits the medium gear.",
                },
                {"verdict": "complete", "gap": None},
            )

        async def ask_llm_structured(
            self,
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            tool_executor: Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]]
            | None = None,
            max_tool_rounds: int = 3,
        ) -> dict[str, Any]:
            if response_format["name"] == "spec2primitives_target_feature_review":
                self.review_calls.append({"prompt": prompt, "response_format": response_format})
                return dict(self.review_results[len(self.review_calls) - 1])
            return await super().ask_llm_structured(
                prompt,
                response_format=response_format,
                tools=tools,
                tool_executor=tool_executor,
                max_tool_rounds=max_tool_rounds,
            )

    async def complete(**kwargs: object) -> Mapping[str, object]:
        del kwargs
        return {
            "grounding_status": "complete",
            "resource_selection_ref": ("products/grounding/resource_selection/test.json"),
        }

    agent = _ReviewRevisionAgent()
    monkeypatch.setattr(runtime, "_complete_resource_assignment", complete)
    result = asyncio.run(
        runtime.ground_product_context(
            agent,
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            product_context={},
            max_pa_turns=2,
        )
    )

    assert result["grounding_status"] == "complete"
    assert len(agent.calls) == 2
    assert len(agent.review_calls) == 2
    assert "target_feature_semantic_review" in str(agent.calls[1]["prompt"])
    first = json.loads(
        (root / "products/grounding/ontology_grounding/proposal_0001.json").read_text(
            encoding="utf-8"
        )
    )
    second = json.loads(
        (root / "products/grounding/ontology_grounding/proposal_0002.json").read_text(
            encoding="utf-8"
        )
    )
    review = json.loads(
        (root / "products/grounding/target_feature_review/review_0001.json").read_text(
            encoding="utf-8"
        )
    )
    assert first["status"] == "rejected"
    assert second["status"] == "accepted"
    assert set(review) == {
        "schema_version",
        "record_type",
        "review_number",
        "proposal_number",
        "target_feature_fingerprint",
        "evidence_refs",
        "verdict",
        "gap",
        "reviewed_at_ns",
        "fingerprint",
    }
    assert "reasoning" not in review


def test_clarification_requires_successful_approved_evidence(
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
    _allow_unrelated_test_through_selected_candidate_gate(runtime, monkeypatch)
    question = {"clarification_question": "Which product variant is intended?"}
    agent = _SequencedProductAgent((question, _proposal("requirement_0001")))

    async def complete(**kwargs: object) -> Mapping[str, object]:
        del kwargs
        return {
            "grounding_status": "complete",
            "resource_selection_ref": ("products/grounding/resource_selection/test.json"),
        }

    monkeypatch.setattr(runtime, "_complete_resource_assignment", complete)
    result = asyncio.run(
        runtime.ground_product_context(
            agent,
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            product_context={},
            max_pa_turns=2,
        )
    )

    assert result["grounding_status"] == "complete"
    assert len(agent.calls) == 2
    revision_prompt = str(agent.calls[1]["prompt"])
    assert "evidence_first_clarification" in revision_prompt
    assert "Do not assume or supply an interpretation" in revision_prompt
    assert "completed assembly is already visible" not in revision_prompt


def test_clarification_after_successful_retrieval_is_returned(
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
    _allow_unrelated_test_through_selected_candidate_gate(runtime, monkeypatch)

    async def retrieve(
        investigation: _NativeEvidenceInvestigation,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        assert tool_name == "retrieve"
        evidence_id = str(arguments["evidence_id"])
        investigation.retrieved_handle_ids.append(evidence_id)
        return {"evidence_refs": ["requirement_0001"], "record_refs": []}

    monkeypatch.setattr(_NativeEvidenceInvestigation, "execute", retrieve)

    class _RetrieveThenClarifyAgent(_ToolUsingProductAgent):
        async def ask_llm_structured(
            self,
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            tool_executor: Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]]
            | None = None,
            max_tool_rounds: int = 3,
        ) -> dict[str, Any]:
            assert tool_executor is not None
            assert tools is not None
            evidence_ids = tools[0]["function"]["parameters"]["properties"]["evidence_id"]["enum"]
            await tool_executor("retrieve", {"evidence_id": evidence_ids[0]})
            self.calls.append(
                {
                    "prompt": prompt,
                    "response_format": response_format,
                    "tools": tools,
                    "max_tool_rounds": max_tool_rounds,
                }
            )
            return {"result": {"clarification_question": "Which product variant is intended?"}}

    agent = _RetrieveThenClarifyAgent(retrieve_order=())
    result = asyncio.run(
        runtime.ground_product_context(
            agent,
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            product_context={},
            max_pa_turns=2,
        )
    )

    assert result == {
        "grounding_status": "clarification_required",
        "clarification_question": "Which product variant is intended?",
        "tool_call_refs": [],
    }
    assert "hidden case knowledge" in str(agent.calls[0]["prompt"])


def test_premature_clarification_at_turn_limit_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble medium gear", tbox)
    runtime = ProductionProductContextGroundingRuntime(
        tbox=tbox,
        document_config=load_model_runtime_config().document_vlm,
        document_vision_runtime=_NoDocumentVision(),
    )
    agent = _SequencedProductAgent(
        ({"clarification_question": "Which product variant is intended?"},)
    )

    result = asyncio.run(
        runtime.ground_product_context(
            agent,
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            product_context={},
            max_pa_turns=1,
        )
    )

    assert result["grounding_status"] == "incomplete"
    assert "clarification_question" not in result
    assert result["insufficient_evidence"] == (
        "PA requested user clarification before retrieving and considering approved evidence."
    )


def test_answered_clarification_uses_persisted_record_ref(
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
    _allow_unrelated_test_through_selected_candidate_gate(runtime, monkeypatch)
    clarification = {
        "record_type": "PAClarification",
        "question_turn": 3,
        "reply": "Medium Gear",
    }
    clarification_path = root / "interaction_record/clarification_0003.json"
    clarification_path.parent.mkdir(parents=True)
    clarification_path.write_text(json.dumps(clarification), encoding="utf-8")
    evidence_ref = "interaction_record/clarification_0003.json"
    presentation = load_or_create_evidence_presentation(
        root,
        sources=_approved_evidence_sources(),
    )
    presented_ref = presentation.opaque_reference(evidence_ref, kind="citation")
    agent = _SequencedProductAgent((_proposal(presented_ref),))

    async def complete(**kwargs: object) -> Mapping[str, object]:
        del kwargs
        return {
            "grounding_status": "complete",
            "resource_selection_ref": ("products/grounding/resource_selection/test.json"),
        }

    monkeypatch.setattr(runtime, "_complete_resource_assignment", complete)
    result = asyncio.run(
        runtime.ground_product_context(
            agent,
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            product_context={},
            max_pa_turns=1,
            clarification_history=(clarification,),
        )
    )

    assert result["grounding_status"] == "complete"
    prompt = str(agent.calls[0]["prompt"])
    assert f'"evidence_ref": "{presented_ref}"' in prompt
    assert evidence_ref not in prompt
    proposal = json.loads(
        (root / "products/grounding/ontology_grounding/proposal_0001.json").read_text(
            encoding="utf-8"
        )
    )
    target_feature = proposal["output"]["target_feature"]
    assert evidence_ref in target_feature["required_process"]["evidence_refs"]
    assert evidence_ref in target_feature["desired_state"]["statement"]["evidence_refs"]
    review_prompt = str(agent.review_calls[0]["prompt"])
    assert evidence_ref not in review_prompt
    assert presented_ref in review_prompt


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
    _allow_unrelated_test_through_selected_candidate_gate(runtime, monkeypatch)
    question = {
        "clarification_question": (
            "Do you also want retrieval of the live RGB-D observation evidence "
            "for the current RGBDSegmentationRecord validation gap?"
        )
    }
    agent = _SequencedProductAgent((question, _proposal("requirement_0001")))

    async def complete(**kwargs: object) -> Mapping[str, object]:
        del kwargs
        return {
            "grounding_status": "complete",
            "resource_selection_ref": ("products/grounding/resource_selection/test.json"),
        }

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
        required_record_type="RGBDSegmentationRecord",
        handles=_default_handles(tmp_path / "first_catalog")[1],
    )
    assert not _clarification_requests_system_choice(
        "Which product variant do you mean?",
        required_record_type="RGBDSegmentationRecord",
        handles=_default_handles(tmp_path / "second_catalog")[1],
    )


def test_production_source_has_no_task_label_or_answer_recipe() -> None:
    package_root = Path(__file__).resolve().parents[1]
    source = (package_root / "agents/pa/production_grounding.py").read_text(encoding="utf-8")
    prompt_source = (package_root / "agents/pa/ontology_grounding.py").read_text(encoding="utf-8")

    assert "supports_manipulator_pick_place" not in source
    assert "mounting order" not in prompt_source
    assert "Phase 5" not in prompt_source
    assert "next_action" not in prompt_source
    assert '"inspect"' not in prompt_source
    assert "target_feature" in prompt_source
    assert "compatible retrieved CAD and live observation evidence" not in source
    assert "Gear_Medium" not in source
    assert "Gear_Shaft" not in source
    assert "required_CAD_size_comparisons" not in source
    assert "target_frame_if_a_verifier_derives_geometry" not in source


async def _unused_tool(
    tool_name: str,
    arguments: Mapping[str, object],
) -> Mapping[str, object]:
    del tool_name, arguments
    raise AssertionError("No tool call was expected.")


def _write_neutral_location(
    root: Path,
    name: str,
    translation_m: tuple[float, float, float],
) -> tuple[Path, str]:
    destination = root / "products/grounding/neutral_test_location" / name
    destination.mkdir(parents=True, exist_ok=True)
    source_path = destination / "source.json"
    source_path.write_text(
        json.dumps({"neutral_observation": name}),
        encoding="utf-8",
    )
    source_ref = source_path.relative_to(root).as_posix()
    location_path = destination / "robot_frame_location_record.json"
    location_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "record_type": "RobotFrameLocationRecord",
                "producer": "neutral_test_location_provider",
                "method": "neutral_test_location",
                "source_evidence": {
                    "ref": source_ref,
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                },
                "candidate_reference": {
                    "observation_handle": f"observation_{name}",
                    "candidate_handle": f"candidate_{name}",
                },
                "observation_timestamp_ns": 1,
                "source_frame": "neutral_camera_frame",
                "target_frame": "world",
                "translated_location_m": list(translation_m),
                "location": "available",
                "robot_frame_conversion": "accepted",
            }
        ),
        encoding="utf-8",
    )
    return location_path, source_ref


def _proposal(evidence_ref: str) -> dict[str, object]:
    return {
        "target_feature": {
            "required_process": {
                "process_iri": "https://cais-spade-llm.local/process/assembly",
                "evidence_refs": [evidence_ref],
            },
            "current_state": {
                "statement": {
                    "text": "The medium gear is currently separate from the assembly.",
                    "evidence_refs": [evidence_ref],
                },
                "state_values": [],
            },
            "desired_state": {
                "statement": {
                    "text": "The medium gear is assembled as requested.",
                    "evidence_refs": [evidence_ref],
                },
                "state_values": [],
            },
        }
    }

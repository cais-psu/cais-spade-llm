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
    ProductionProductContextGroundingRuntime,
    _approved_evidence_handles,
    _approved_evidence_sources,
    _clarification_requests_system_choice,
    _EvidenceHandle,
    _NativeEvidenceInvestigation,
    _neutral_candidate_views,
    _pa_allocation_prompt,
    _producer_descriptors,
    _proposal_cad_bindings,
    _raw_evidence_types_for_gap,
    _required_record_plan,
    _retrieve_tool,
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
        (handle.context_ref, handle.evidence_type, handle.source_revision)
        for handle in handles
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
        "products/grounding/rgb_d_cad_grounding/segmentation_0001/"
        "rgbd_segmentation_record.json"
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
        validation_feedback=None,
    )

    prompt_input = json.loads(prompt.split("Allocation input:\n", maxsplit=1)[1])
    assert {
        item["evidence_type"] for item in prompt_input["approved_retrieved_evidence"]
    } == {"document", "CAD", "observation"}
    state_evidence = prompt_input["neutral_state_evidence_pool"][0]
    assert state_evidence["observation_handle"] == "view_0001"
    assert state_evidence["candidate_handle"] == "candidate_0001_0002"
    assert state_evidence["candidate_value_ref"] == {
        "record_ref": opaque_record_ref,
        "field_path": field_path,
    }
    assert state_evidence["candidate_value_ref"] == observation_result["segmentation"][
        "views"
    ][0]["candidates"][0]["candidate_value_ref"]
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
        evidence_handles = tools[0]["function"]["parameters"]["properties"][
            "current_state_evidence_handle"
        ]["enum"]
        assert len(evidence_handles) == 2
        resource_symbol = self.resource_choices[len(self.prompts)]
        self.prompts.append(prompt)
        reachability = await tool_executor(
            "check_reachability",
            {
                "resource_symbol": resource_symbol,
                "current_state_evidence_handle": evidence_handles[0],
                "desired_state_evidence_handle": evidence_handles[1],
            },
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
    assert "synthetic.stl" not in serialized_result
    assert "synthetic static CAD" not in serialized_result
    assert "CAD_identity" not in serialized_result
    assert first_result["evidence_handle"] == medium.evidence_id
    assert all(
        str(record_ref).startswith("typed_record_")
        for record_ref in first_result["record_refs"]
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
    view = SimpleNamespace(typed_bindings=(binding,))
    uncited = SimpleNamespace(evidence_refs=("requirement_0001",))
    cited = SimpleNamespace(evidence_refs=("Gear_Medium.STL",))

    assert _proposal_cad_bindings(tmp_path, view, uncited) == ()
    assert _proposal_cad_bindings(tmp_path, view, cited) == (binding,)


def test_semantic_commit_does_not_require_precomputed_target_geometry(
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

    async def complete(**kwargs: object) -> Mapping[str, object]:
        proposal = kwargs["proposal"]
        assert load_interaction_abox(root, tbox).accepted_assertion_count == 7
        assert proposal.target_feature["current_state"]["state_values"] == []
        assert proposal.target_feature["desired_state"]["state_values"] == []
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
    assert len(agent.calls) == 1
    assert "current_validation_gap" not in str(agent.calls[0]["prompt"])
    assert "TargetFeatureGeometryRecord" not in str(agent.calls[0]["prompt"])
    assert load_interaction_abox(root, tbox).accepted_assertion_count == 7
    assert (root / "products/grounding/ontology_grounding/proposal_0001.json").is_file()


def test_robot_agent_rejection_returns_to_pa_without_resource_substitution(
    tmp_path: Path,
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
    assert "completed assembly" not in revision_prompt


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
            evidence_ids = tools[0]["function"]["parameters"]["properties"]["evidence_id"][
                "enum"
            ]
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

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
from cais_spade_llm.spec2primitives.agents.pa.ontology_grounding import (
    OntologyGroundingCandidate,
    OntologyGroundingError,
    OntologyGroundingInterruption,
    OntologyGroundingProposal,
    OntologyGroundingResult,
    commit_ontology_grounding_candidate,
    propose_and_validate_ontology_grounding,
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
    _analyze_candidate_layout_tool,
    _approved_evidence_handles,
    _approved_evidence_sources,
    _cad_size_comparison_projection,
    _CADComparisonBinding,
    _compare_cad_size_tool,
    _EvidenceHandle,
    _NativeEvidenceInvestigation,
    _neutral_candidate_views,
    _pa_allocation_prompt,
    _proposal_evidence_is_intact,
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

    async def query_document(self, request: object) -> object:
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


def test_pa_document_question_is_persisted_verbatim_and_scene_refs_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "an unforeseen requirement", tbox)
    context_ref = "controlled_manual.pdf"
    document_handle = _EvidenceHandle(
        evidence_id="document_opaque",
        evidence_type="document",
        display_name="document_opaque",
        context_ref=context_ref,
        source_revision="a" * 64,
    )
    handles = (document_handle,)
    presentation = _presentation_for_handles(root, handles)
    runtime = SimpleNamespace(
        _document_config=load_model_runtime_config().document_vlm,
        _document_vision_runtime=_NoDocumentVision(),
        _grounding_candidate_order=None,
    )
    investigation = _NativeEvidenceInvestigation(
        runtime=runtime,
        interaction_root=root,
        tbox=tbox,
        abox=abox,
        requirement=abox.product_requirement,
        handles=handles,
        presentation=presentation,
    )
    source_index_path = (
        root / "products/grounding/document_evidence/source_index_0001.json"
    )
    source_index_path.parent.mkdir(parents=True)
    source_index_path.write_text(
        json.dumps({"record_type": "DocumentSourceIndexRecord"}),
        encoding="utf-8",
    )
    source_index_ref = source_index_path.relative_to(root).as_posix()
    investigation._register_references(
        source_refs=(context_ref, f"{context_ref}#page=1"),
        record_refs=(source_index_ref,),
    )
    investigation.retrieved_results[document_handle.evidence_id] = {
        "record_refs": [investigation.project_canonical_reference(source_index_ref)]
    }
    source_binding = SimpleNamespace(
        record_type="DocumentSourceIndexRecord",
        status="accepted",
        record_ref=source_index_ref,
        record_sha256=hashlib.sha256(source_index_path.read_bytes()).hexdigest(),
        evidence_refs=(context_ref,),
    )
    monkeypatch.setattr(
        production_grounding,
        "build_product_context_view",
        lambda *args, **kwargs: SimpleNamespace(typed_bindings=(source_binding,)),
    )
    captured_questions: list[str] = []

    async def query_fixture(**kwargs: object) -> SimpleNamespace:
        question = str(kwargs["question"])
        captured_questions.append(question)
        query_path = root / "products/grounding/document_evidence/query_0001.json"
        record: dict[str, object] = {
            "schema_version": 1,
            "record_type": "DocumentQueryRecord",
            "producer": "document_evidence",
            "status": "supported",
            "question": question,
            "claims": [
                {
                    "predicate_text": "shown_in_order",
                    "arguments": ["first item", "second item", "third item"],
                    "evidence_refs": [f"{context_ref}#page=1"],
                    "uncertainty": [],
                }
            ],
            "uncertainty": [],
            "evidence_refs": [source_index_ref, f"{context_ref}#page=1"],
        }
        record["fingerprint"] = production_grounding._json_fingerprint(record)
        query_path.write_text(json.dumps(record), encoding="utf-8")
        return SimpleNamespace(record_path=query_path, status="supported", record=record)

    monkeypatch.setattr(production_grounding, "query_document_evidence", query_fixture)
    monkeypatch.setattr(
        production_grounding,
        "_merge_derived_record",
        lambda *args, **kwargs: (
            abox,
            SimpleNamespace(
                record_type="DocumentQueryRecord",
                status="accepted",
                record_ref="products/grounding/document_evidence/query_0001.json",
            ),
        ),
    )
    exact_question = "Which ordered items are explicitly shown?"
    result = asyncio.run(
        investigation.execute(
            "query_document",
            {
                "document_evidence_id": document_handle.evidence_id,
                "question": exact_question,
            },
        )
    )

    assert result["question"] == exact_question
    assert captured_questions == [exact_question]
    audit = json.loads(
        (root / "interaction_record/tool_call_0001.json").read_text(encoding="utf-8")
    )
    assert audit["arguments"]["question"] == exact_question
    assert audit["tool_name"] == "query_document"

    investigation.retrieved_results["issued_scene"] = {
        "segmentation": {
            "views": [
                {
                    "observation_handle": "issued_view_token",
                    "candidates": [{"candidate_handle": "issued_candidate_token"}],
                }
            ]
        }
    }
    rejected = asyncio.run(
        investigation.execute(
            "query_document",
            {
                "document_evidence_id": document_handle.evidence_id,
                "question": "Where should issued_candidate_token be placed?",
            },
        )
    )
    assert rejected["error"]["reason"] == "cross_modal_question_reference"
    assert captured_questions == [exact_question]


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
    )

    prompt_input = json.loads(prompt.split("Allocation input:\n", maxsplit=1)[1])
    assert {item["evidence_type"] for item in prompt_input["approved_retrieved_evidence"]} == {
        "document",
        "CAD",
        "observation",
    }
    location_evidence = prompt_input["location_evidence_catalog"]
    assert len(location_evidence) == 1
    assert location_evidence[0]["evidence_handle"] == "state_evidence_0001"
    assert location_evidence[0]["observation_handle"] == "view_0001"
    assert location_evidence[0]["candidate_handle"] == "candidate_0001_0002"
    assert location_evidence[0]["candidate_value_ref"] == {
        "record_ref": opaque_record_ref,
        "field_path": field_path,
    }
    assert "grounded_state_evidence" not in prompt_input
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


class _TwoStageAllocationAgent:
    def __init__(self, resource_symbol: str) -> None:
        self.resource_symbol = resource_symbol
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
        self.calls.append(
            {
                "prompt": prompt,
                "response_format": response_format,
                "tools": tools,
                "max_tool_rounds": max_tool_rounds,
            }
        )
        if response_format["name"] == "spec2primitives_grounding_result":
            return {"result": _proposal("requirement_0001")}
        assert response_format["name"] == "spec2primitives_pa_resource_allocation"
        assert tool_executor is not None and tools is not None
        parameters = tools[0]["function"]["parameters"]
        assert set(parameters["properties"]) == {
            "resource_symbol",
            "state_locations",
        }
        location_handles = parameters["properties"]["state_locations"]["properties"][
            "current_state"
        ]["items"]["enum"]
        assert len(location_handles) >= 2
        state_locations = {
            "current_state": [location_handles[0]],
            "desired_state": [location_handles[1]],
        }
        reachability = await tool_executor(
            "check_reachability",
            {
                "resource_symbol": self.resource_symbol,
                "state_locations": state_locations,
            },
        )
        assert reachability["status"] == "accepted"
        return {
            "result": {
                "resource_symbol": self.resource_symbol,
                "state_locations": state_locations,
                "reachability_check_ref": reachability["reachability_check_ref"],
            }
        }


class _AcceptingStateLocationFeasibility:
    def __init__(self) -> None:
        self.requests: list[Mapping[str, object]] = []

    async def validate_plan_only_allocation(
        self,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        self.requests.append(dict(request))
        state_locations = request["state_locations"]
        return {
            "status": "accepted",
            "state_locations": {
                state_name: [
                    {
                        "evidence_handle": item["evidence_handle"],
                        "status": "accepted",
                        "message": "selected location is reachable",
                        "error_code": 1,
                    }
                    for item in locations
                ]
                for state_name, locations in state_locations.items()
            },
            "feedback": None,
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
    required_process_schema = target_schema["properties"]["required_process"]
    assert required_process_schema["properties"]["process_iri"]["enum"] == [
        "https://cais-spade-llm.local/process/assembly"
    ]
    for state_name in ("current_state", "desired_state"):
        state_schema = target_schema["properties"][state_name]
        assert set(state_schema["properties"]) == {
            "statement",
            "state_values",
        }
    serialized_schema = json.dumps(schema)
    assert "insufficient_evidence" not in serialized_schema
    assert '"reason"' not in serialized_schema
    assert "unmet_obligation" not in serialized_schema
    assert '"enum": [true]' not in serialized_schema
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
    assert "resource locations" in str(call["prompt"])


def test_direct_clarification_stops_without_proposal(tmp_path: Path) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble product", tbox)
    workcell = load_predefined_workcell(
        tbox,
        load_predefined_resource_registry(tbox),
    )
    result = asyncio.run(
        propose_and_validate_ontology_grounding(
            _ToolUsingProductAgent(
                retrieve_order=(),
                final_result={
                    "clarification_question": "Which product variant is intended?"
                },
            ),
            interaction_root=tmp_path,
            tbox=tbox,
            abox=abox,
            workcell=workcell,
            evidence_catalog=[],
            authorized_evidence_refs={"requirement_0001"},
            tools=[],
            tool_executor=_unused_tool,
            max_tool_rounds=3,
        )
    )

    assert isinstance(result, OntologyGroundingInterruption)
    assert result.kind == "clarification_question"
    assert not (tmp_path / "products/grounding/ontology_grounding").exists()


@pytest.mark.parametrize(
    "removed_result",
    [
        {"insufficient_evidence": "Legacy unscoped reason."},
        {
            "insufficient_evidence": {
                "unmet_obligation": "clearance",
                "reason": "Mounting and meshing are not established.",
            }
        },
    ],
)
def test_removed_insufficient_evidence_result_cannot_interrupt_grounding(
    tmp_path: Path,
    removed_result: Mapping[str, object],
) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble product", tbox)
    workcell = load_predefined_workcell(
        tbox,
        load_predefined_resource_registry(tbox),
    )

    with pytest.raises(OntologyGroundingError, match="target-feature candidate"):
        asyncio.run(
            propose_and_validate_ontology_grounding(
                _ToolUsingProductAgent(
                    retrieve_order=(),
                    final_result=removed_result,
                ),
                interaction_root=tmp_path,
                tbox=tbox,
                abox=abox,
                workcell=workcell,
                evidence_catalog=[],
                authorized_evidence_refs={"requirement_0001"},
                tools=[],
                tool_executor=_unused_tool,
                max_tool_rounds=1,
            )
        )


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
                typed_record_resolver=resolver,
            )
        )
    rejected = json.loads(
        (root / "products/grounding/ontology_grounding/proposal_0001.json").read_text(
            encoding="utf-8"
        )
    )
    assert rejected["schema_version"] == 9
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


def test_candidate_layout_tool_metadata_is_generic_and_accepts_two_or_more() -> None:
    tool = _analyze_candidate_layout_tool()

    assert tool["function"]["name"] == "analyze_candidate_layout"
    parameters = tool["function"]["parameters"]
    assert parameters["properties"]["candidate_field_paths"]["minItems"] == 2
    serialized = json.dumps(tool).casefold()
    assert "same observation frame" in serialized
    assert "uniqueitems" not in serialized
    for forbidden in (
        "gear",
        "shaft",
        "middle",
        "between",
        "expected candidate",
        "preferred evidence",
    ):
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
    candidate = result["candidate_measurements"][0]
    assert set(candidate) == {
        "observation_handle",
        "candidate_handle",
        "candidate_value_ref",
        "observed_dimensions_m",
        "dimension_errors",
        "mean_dimension_error",
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


def test_cad_projection_exposes_every_measurement_in_neutral_observation_order() -> None:
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

    assert [
        candidate["candidate_handle"] for candidate in result["candidate_measurements"]
    ] == candidate_handles
    assert "ranked_candidates" not in result
    assert "plausible_candidates" not in result
    assert "selected_candidate" not in result
    assert all("rank" not in candidate for candidate in result["candidate_measurements"])


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
            "schema_version": 3,
            "record_type": "CADSizeCorrespondenceRecord",
            "CAD": {
                "record": {"ref": cad_ref},
                "compared_dimensions_m": [0.020, 0.010],
            },
            "segmentation": {"record": {"ref": segmentation_ref}},
            "parameters": {"dimension_error_limit": 0.15},
            "CAD_correspondence": "not_evaluated",
            "measurement": "accepted",
            "candidate_measurements": [
                {
                    "observation_handle": "view_opaque",
                    "candidate_handle": "candidate_opaque",
                    "observed_dimensions_m": [0.020, 0.010],
                    "dimension_errors": [0.0, 0.0],
                    "mean_dimension_error": 0.0,
                    "within_size_tolerance": True,
                    "candidate_center_m": [0.1, 0.2, 0.3],
                }
            ],
        }
        record_path.write_text(json.dumps(record), encoding="utf-8")
        return SimpleNamespace(
            record_path=record_path,
            record=record,
            measurement="accepted",
        )

    monkeypatch.setattr(
        production_grounding,
        "measure_segmented_candidates_against_cad",
        match_size,
    )
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


def test_generic_state_values_do_not_require_candidate_or_cad_evidence(
    tmp_path: Path,
) -> None:
    record_path = tmp_path / "products/grounding/document/value_record.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            {
                "record_type": "DocumentOverviewRecord",
                "value": "specified",
                "other_value": "measured",
            }
        ),
        encoding="utf-8",
    )
    record_ref = record_path.relative_to(tmp_path).as_posix()
    record_sha256 = hashlib.sha256(record_path.read_bytes()).hexdigest()
    binding = SimpleNamespace(
        record_type="DocumentOverviewRecord",
        status="accepted",
        record_ref=record_ref,
        record_sha256=record_sha256,
        evidence_refs=("approved_document#page=1",),
    )
    view = SimpleNamespace(typed_bindings=(binding,))
    investigation = SimpleNamespace(root=tmp_path, comparison_bindings={})
    no_values = OntologyGroundingProposal(
        target_feature={
            "current_state": {"state_values": []},
            "desired_state": {"state_values": []},
        },
        feature_iri="urn:feature:1",
        resolved_state_values=(),
        evidence_refs=(),
    )
    open_values = OntologyGroundingProposal(
        target_feature={
            "current_state": {
                "state_values": [
                    {
                        "name": "document_value",
                        "value_ref": {
                            "record_ref": record_ref,
                            "field_path": "/value",
                        },
                        "evidence_refs": ["approved_document#page=1"],
                    },
                    {
                        "name": "other_document_value",
                        "value_ref": {
                            "record_ref": record_ref,
                            "field_path": "/other_value",
                        },
                        "evidence_refs": ["approved_document#page=1"],
                    },
                ],
            }
        },
        feature_iri="urn:feature:1",
        resolved_state_values=(
            {
                "state": "current_state",
                "name": "document_value",
                "record_type": "DocumentOverviewRecord",
                "record_sha256": record_sha256,
                "value_ref": {"record_ref": record_ref, "field_path": "/value"},
                "resolved_value": "specified",
            },
            {
                "state": "current_state",
                "name": "other_document_value",
                "record_type": "DocumentOverviewRecord",
                "record_sha256": record_sha256,
                "value_ref": {
                    "record_ref": record_ref,
                    "field_path": "/other_value",
                },
                "resolved_value": "measured",
            },
        ),
        evidence_refs=(),
    )

    assert _proposal_evidence_is_intact(investigation, view, no_values) is True
    assert _proposal_evidence_is_intact(investigation, view, open_values) is True
    record_path.write_text(
        json.dumps({"record_type": "DocumentOverviewRecord", "value": "changed"}),
        encoding="utf-8",
    )
    assert _proposal_evidence_is_intact(investigation, view, open_values) is False


def test_state_value_does_not_require_unique_size_correspondence(
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
        field_path = f"/cameras/0/candidates/{candidate_index}"
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
                                "field_path": field_path,
                            },
                            "evidence_refs": [cad_ref, comparison_ref],
                        }
                    ],
                }
            },
            feature_iri="urn:feature:1",
            resolved_state_values=(
                {
                    "state": "desired_state",
                    "name": "supported_destination",
                    "record_type": "RGBDSegmentationRecord",
                    "record_sha256": segmentation_binding.record_sha256,
                    "value_ref": {
                        "record_ref": segmentation_ref,
                        "field_path": field_path,
                    },
                    "resolved_value": {
                        "candidate_handle": candidate_handles[candidate_index]
                    },
                },
            ),
            evidence_refs=(),
        )

    assert _proposal_evidence_is_intact(investigation, view, proposal(1)) is True


def test_state_value_does_not_require_spatial_relation_subject_match(
    tmp_path: Path,
) -> None:
    segmentation_ref = "products/grounding/observation/segmentation.json"
    comparison_ref = "products/grounding/comparison/ambiguous.json"
    relation_ref = "products/grounding/relation/layout.json"
    for relative in (segmentation_ref, comparison_ref, relation_ref):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
    candidates = ["candidate_left", "candidate_middle", "candidate_right"]
    segmentation_path = tmp_path / segmentation_ref
    segmentation_path.write_text(
        json.dumps(
            {
                "cameras": [
                    {
                        "observation_handle": "view_opaque",
                        "candidates": [
                            {"candidate_handle": candidate} for candidate in candidates
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    comparison_path = tmp_path / comparison_ref
    comparison_path.write_text(
        json.dumps(
            {
                "record_type": "CADSizeCorrespondenceRecord",
                "CAD_correspondence": "ambiguous",
                "segmentation": {"record": {"ref": segmentation_ref}},
                "plausible_candidates": [
                    {
                        "observation_handle": "view_opaque",
                        "candidate_handle": candidate,
                        "within_size_tolerance": True,
                    }
                    for candidate in candidates
                ],
            }
        ),
        encoding="utf-8",
    )

    def candidate_ref(index: int) -> dict[str, object]:
        return {
            "observation_handle": "view_opaque",
            "candidate_handle": candidates[index],
            "value_ref": {
                "record_ref": segmentation_ref,
                "field_path": f"/cameras/0/candidates/{index}",
            },
        }

    relation_record: dict[str, object] = {
        "schema_version": 1,
        "record_type": "CandidateSpatialRelationRecord",
        "producer": "rgb_d_cad_grounding",
        "comparison": {
            "ref": comparison_ref,
            "sha256": hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
        },
        "segmentation": {
            "ref": segmentation_ref,
            "sha256": hashlib.sha256(segmentation_path.read_bytes()).hexdigest(),
        },
        "evidence_refs": [comparison_ref, segmentation_ref],
        "candidate_count": 3,
        "candidates": [candidate_ref(index) for index in range(3)],
        "relations": [
            {
                "predicate_text": "between",
                "subject": candidate_ref(1),
                "objects": [candidate_ref(0), candidate_ref(2)],
                "group_size": 3,
                "residual": 0.0,
                "next_best_residual": 2.0,
                "residual_separation": 2.0,
            }
        ],
        "status": "accepted",
    }
    relation_record["fingerprint"] = production_grounding._json_fingerprint(
        relation_record
    )
    relation_path = tmp_path / relation_ref
    relation_path.write_text(json.dumps(relation_record), encoding="utf-8")
    segmentation_sha256 = hashlib.sha256(segmentation_path.read_bytes()).hexdigest()
    relation_binding = SimpleNamespace(
        record_type="CandidateSpatialRelationRecord",
        status="accepted",
        record_ref=relation_ref,
        record_sha256=hashlib.sha256(relation_path.read_bytes()).hexdigest(),
        evidence_refs=(comparison_ref, segmentation_ref),
    )
    view = SimpleNamespace(
        typed_bindings=(
            SimpleNamespace(
                record_type="RGBDSegmentationRecord",
                status="accepted",
                record_ref=segmentation_ref,
                record_sha256=segmentation_sha256,
                evidence_refs=(),
            ),
            SimpleNamespace(
                record_type="CADSizeCorrespondenceRecord",
                status="ambiguous",
                record_ref=comparison_ref,
                record_sha256=hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
                evidence_refs=(),
            ),
            relation_binding,
        )
    )
    investigation = SimpleNamespace(
        root=tmp_path,
        comparison_bindings={
            comparison_ref: _CADComparisonBinding(
                record_ref=comparison_ref,
                cad_record_ref="products/grounding/cad/unused.json",
                segmentation_record_ref=segmentation_ref,
            )
        },
    )

    def proposal(index: int) -> OntologyGroundingProposal:
        field_path = f"/cameras/0/candidates/{index}"
        return OntologyGroundingProposal(
            target_feature={
                "desired_state": {
                    "state_values": [
                        {
                            "name": "selected_value",
                            "value_ref": {
                                "record_ref": segmentation_ref,
                                "field_path": field_path,
                            },
                            "evidence_refs": [comparison_ref, relation_ref],
                        }
                    ]
                }
            },
            feature_iri="urn:feature:1",
            resolved_state_values=(
                {
                    "state": "desired_state",
                    "name": "selected_value",
                    "record_type": "RGBDSegmentationRecord",
                    "record_sha256": segmentation_sha256,
                    "value_ref": {
                        "record_ref": segmentation_ref,
                        "field_path": field_path,
                    },
                    "resolved_value": {"candidate_handle": candidates[index]},
                },
            ),
            evidence_refs=(relation_ref,),
        )

    assert _proposal_evidence_is_intact(investigation, view, proposal(1)) is True
    assert _proposal_evidence_is_intact(investigation, view, proposal(0)) is True
    assert _proposal_evidence_is_intact(investigation, view, proposal(2)) is True

    relation_record["comparison"] = {
        "ref": "products/grounding/comparison/unrelated.json",
        "sha256": "f" * 64,
    }
    relation_record["fingerprint"] = production_grounding._json_fingerprint(
        {
            key: value
            for key, value in relation_record.items()
            if key != "fingerprint"
        }
    )
    relation_path.write_text(json.dumps(relation_record), encoding="utf-8")
    relation_binding.record_sha256 = hashlib.sha256(relation_path.read_bytes()).hexdigest()
    assert _proposal_evidence_is_intact(investigation, view, proposal(1)) is True

    relation_record["relations"][0]["subject"] = candidate_ref(0)  # type: ignore[index]
    relation_path.write_text(json.dumps(relation_record), encoding="utf-8")
    assert _proposal_evidence_is_intact(investigation, view, proposal(1)) is False


def test_current_state_value_does_not_require_unique_size_correspondence(
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
    segmentation_sha256 = hashlib.sha256((tmp_path / segmentation_ref).read_bytes()).hexdigest()
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
                record_sha256=segmentation_sha256,
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
        resolved_state_values=(
            {
                "state": "current_state",
                "name": "medium_gear",
                "record_type": "RGBDSegmentationRecord",
                "record_sha256": segmentation_sha256,
                "value_ref": {
                    "record_ref": segmentation_ref,
                    "field_path": "/cameras/0/candidates/0",
                },
                "resolved_value": {"candidate_handle": "loose_candidate"},
            },
        ),
        evidence_refs=(),
    )

    accepted = _proposal_evidence_is_intact(investigation, view, proposal)
    rejected_proposal = OntologyGroundingProposal(
        target_feature={
            "current_state": {
                "statement": proposal.target_feature["current_state"]["statement"],
                "state_values": [
                    {
                        "name": "medium_gear",
                        "value_ref": {
                            "record_ref": segmentation_ref,
                            "field_path": "/cameras/0/candidates/1",
                        },
                        "evidence_refs": [cad_ref, comparison_ref],
                    }
                ],
            }
        },
        feature_iri="urn:feature:1",
        resolved_state_values=(
            {
                **dict(proposal.resolved_state_values[0]),
                "value_ref": {
                    "record_ref": segmentation_ref,
                    "field_path": "/cameras/0/candidates/1",
                },
                "resolved_value": {"candidate_handle": "other_candidate"},
            },
        ),
        evidence_refs=(),
    )
    alternative = _proposal_evidence_is_intact(
        investigation,
        view,
        rejected_proposal,
    )

    assert accepted is True
    assert alternative is True


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


def test_two_autonomous_pa_decisions_commit_exactly_eleven_assertions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    initial_abox = initialize_interaction_abox(root, "assemble medium gear", tbox)
    current_path, current_source = _write_neutral_location(
        root,
        "first",
        (0.0, -0.7, 1.1),
    )
    desired_path, desired_source = _write_neutral_location(
        root,
        "second",
        (0.0, -0.2, 1.1),
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
    locations = validate_and_merge_triple_delta(
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
    feasibility = _AcceptingStateLocationFeasibility()
    runtime = ProductionProductContextGroundingRuntime(
        tbox=tbox,
        document_config=load_model_runtime_config().document_vlm,
        document_vision_runtime=_NoDocumentVision(),
        robot_agent_feasibility_runtime=feasibility,
    )
    agent = _TwoStageAllocationAgent("xarm6")

    result = asyncio.run(
        runtime.ground_product_context(
            agent,
            interaction_root=root,
            tbox=tbox,
            abox=locations.abox,
            product_context={},
            max_pa_turns=1,
        )
    )

    assert result["grounding_status"] == "complete"
    assert result["resource_assignment_status"] == "complete"
    assert len(agent.calls) == 2
    assert len(feasibility.requests) == 1
    assert feasibility.requests[0]["validation_scope"] == (
        "state_location_reachability"
    )
    final_abox = load_interaction_abox(root, tbox)
    assert initial_abox.accepted_assertion_count == 0
    assert final_abox.accepted_assertion_count == 11
    proposal = json.loads((root / str(result["ontology_projection_ref"])).read_text())
    selection = json.loads((root / str(result["resource_selection_ref"])).read_text())
    reachability = json.loads(
        (root / str(selection["reachability_check_ref"])).read_text()
    )
    assert proposal["schema_version"] == 9
    assert len(proposal["compiled_delta"]["assertions"]) == 7
    assert selection["schema_version"] == 5
    assert selection["selected_resource_symbol"] == "xarm6"
    assert reachability["schema_version"] == 4
    assert all(
        reachability["state_locations"][state_name]
        for state_name in ("current_state", "desired_state")
    )
    allocation_contract = json.dumps(
        {
            "response_format": agent.calls[1]["response_format"],
            "tools": agent.calls[1]["tools"],
        }
    ).casefold()
    assert "uniqueitems" not in allocation_contract
    assert '"enum": [true]' not in allocation_contract
    allocation_prompt = str(agent.calls[1]["prompt"])
    assert "translated_location_m" in allocation_prompt
    assert "expected_resource" not in allocation_prompt
    assert "expected_candidate" not in allocation_prompt
    assert not (root / "products/grounding/target_feature_review").exists()
    assert not (root / "products/grounding/completion").exists()


def test_grounding_accepts_empty_state_values_then_reports_missing_location_evidence(
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
    assert result["grounding_stage"] == "resource_assignment"
    assert result["grounding_validation_code"] == "location_evidence_unavailable"
    assert isinstance(result["ontology_projection_ref"], str)
    assert len(agent.calls) == 1
    assert agent.review_calls == []
    first_input = json.loads(str(agent.calls[0]["prompt"]).split("Grounding input:\n", 1)[1])
    assert first_input["target_feature_contract"] == {
        "cardinality": "exactly_one",
        "state_value_cardinality": "zero_or_more",
        "value_names": "PA_authored_without_host_enum",
    }
    assert load_interaction_abox(root, tbox).accepted_assertion_count == 7
    assert (root / str(result["ontology_projection_ref"])).is_file()
    assert not (
        root / "products/grounding/presentation/allocation_presentation_record.json"
    ).exists()


def test_supported_current_statement_does_not_require_unlisted_physical_relations(
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
    partial = _proposal("requirement_0001")
    target_feature = partial["target_feature"]
    assert isinstance(target_feature, dict)
    current_state = target_feature["current_state"]
    assert isinstance(current_state, dict)
    current_state["statement"] = {
        "text": "The observed candidate is the current product state.",
        "evidence_refs": ["requirement_0001"],
    }
    current_state["state_values"] = []
    agent = _ToolUsingProductAgent(retrieve_order=(), final_result=partial)

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
    assert result["grounding_validation_code"] == "location_evidence_unavailable"
    assert agent.review_calls == []
    assert load_interaction_abox(root, tbox).accepted_assertion_count == 7
    assert (root / str(result["ontology_projection_ref"])).is_file()
    prompt = str(agent.calls[0]["prompt"])
    assert "mounting" not in prompt
    assert "meshing" not in prompt


def test_transport_failure_is_not_mislabeled_as_an_invalid_target_feature(
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

    class _RejectedTransport:
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
            del prompt, response_format, tools, tool_executor, max_tool_rounds
            raise RuntimeError("provider rejected the structured request")

    with pytest.raises(RuntimeError, match="provider rejected"):
        asyncio.run(
            runtime.ground_product_context(
                _RejectedTransport(),
                interaction_root=root,
                tbox=tbox,
                abox=abox,
                product_context={},
                max_pa_turns=1,
            )
        )

    assert load_interaction_abox(root, tbox).accepted_assertion_count == 0
    assert not (root / "products/grounding/ontology_grounding").exists()


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
    def evidence_integrity(**kwargs: object) -> bool:
        del kwargs
        if failure_mode == "malformed":
            raise ValueError("controlled malformed evidence")
        return False

    monkeypatch.setattr(runtime, "_proposal_evidence_is_intact", evidence_integrity)

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
    assert result["grounding_validation_code"] == "evidence_reference_invalid"
    assert result["insufficient_evidence"] == "The target feature cites invalid evidence."
    assert load_interaction_abox(root, tbox).accepted_assertion_count == 0


def test_invalid_state_evidence_stops_before_review_and_allocation(
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
    def reject_tampered_evidence(**kwargs: object) -> bool:
        del kwargs
        return False

    monkeypatch.setattr(runtime, "_proposal_evidence_is_intact", reject_tampered_evidence)

    async def unexpected_allocation(**kwargs: object) -> Mapping[str, object]:
        del kwargs
        raise AssertionError("Allocation must not run after ambiguous state evidence.")

    monkeypatch.setattr(runtime, "_complete_resource_assignment", unexpected_allocation)
    agent = _SequencedProductAgent((_proposal("requirement_0001"),))

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
    assert result["grounding_validation_code"] == "evidence_reference_invalid"
    assert result["insufficient_evidence"] == "The target feature cites invalid evidence."
    assert "unmet_grounding_obligation" not in result
    assert agent.review_calls == []
    assert not (root / "products/grounding/target_feature_review").exists()
    assert len(agent.calls) == 1
    prompt = str(agent.calls[0]["prompt"]).casefold()
    assert "validation_feedback" not in prompt
    assert "expected_revision" not in prompt


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (
            OntologyGroundingError(
                "This structural message happens to contain the word evidence."
            ),
            "invalid_target_feature",
        ),
        (
            OntologyGroundingError(
                "Opaque validation failure.",
                validation_code="evidence_reference_invalid",
            ),
            "evidence_reference_invalid",
        ),
    ],
)
def test_incomplete_code_is_typed_and_never_inferred_from_exception_prose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: OntologyGroundingError,
    expected_code: str,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble medium gear", tbox)
    runtime = ProductionProductContextGroundingRuntime(
        tbox=tbox,
        document_config=load_model_runtime_config().document_vlm,
        document_vision_runtime=_NoDocumentVision(),
    )

    def reject(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise error

    monkeypatch.setattr(production_grounding, "validate_ontology_grounding_attempt", reject)
    result = asyncio.run(
        runtime.ground_product_context(
            _SequencedProductAgent((_proposal("requirement_0001"),)),
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            product_context={},
            max_pa_turns=1,
        )
    )

    assert result["grounding_validation_code"] == expected_code
    assert str(error) not in result["insufficient_evidence"]
    assert load_interaction_abox(root, tbox).accepted_assertion_count == 0


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


def test_genuine_clarification_is_returned_without_controller_retry(tmp_path: Path) -> None:
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

    assert result == {
        "grounding_status": "clarification_required",
        "clarification_question": "Which product variant is intended?",
        "tool_call_refs": [],
    }
    assert len(agent.calls) == 1


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
    assert agent.review_calls == []


def test_production_source_has_no_task_label_or_answer_recipe() -> None:
    package_root = Path(__file__).resolve().parents[1]
    source = (package_root / "agents/pa/production_grounding.py").read_text(encoding="utf-8")
    prompt_source = (package_root / "agents/pa/ontology_grounding.py").read_text(encoding="utf-8")
    layout_source = (
        package_root / "tools/rgb_d_cad_grounding/candidate_layout.py"
    ).read_text(encoding="utf-8")
    active_sources = f"{source}\n{prompt_source}\n{layout_source}".casefold()

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
    for forbidden in (
        "medium gear",
        "gear_shaft",
        "shaft-centered",
        "expected_candidate",
        "expected_resource",
        "required evidence order",
        "groundingreadinesscontract",
        "targetfeaturesemanticreview",
        "expected_revision",
        "semantic_review",
    ):
        assert forbidden not in active_sources


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

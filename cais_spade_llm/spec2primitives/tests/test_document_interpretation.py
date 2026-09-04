"""Tests for evidence-first approved PDF preparation and interpretation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import BadRequestError
from PIL import Image

from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    initialize_interaction_abox,
)
from cais_spade_llm.spec2primitives.config import load_model_runtime_config
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    ontology_config,
)
from cais_spade_llm.spec2primitives.tools import exact_ref_resolver
from cais_spade_llm.spec2primitives.tools.document_evidence import (
    DOCUMENT_OVERVIEW_SCHEMA_VERSION,
    DocumentInterpretationError,
    DocumentQueryVisionRequest,
    DocumentVisionRequest,
    DocumentVisionResponse,
    OpenAIDocumentVisionRuntime,
    document_overview_cache_status,
    document_query_schema,
    index_document_evidence,
    interpret_document_evidence,
    prepare_document_overview,
    query_document_evidence,
    run_document_interpretation_diagnostic,
)
from cais_spade_llm.spec2primitives.tools.document_evidence.interpreter import (
    RenderedDocumentPage,
    document_interpretation_schema,
)
from cais_spade_llm.spec2primitives.tools.document_evidence.prepare import (
    main as prepare_document_main,
)
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    resolve_context_ref,
)

_TEST_DOCUMENT_REF = "NIST_assembly_instructions.pdf"


class ControlledVisionRuntime:
    """Return controlled neutral document records without network calls."""

    def __init__(
        self,
        output: dict[str, object] | None = None,
        *,
        query_output: dict[str, object] | None = None,
        model: str = "gpt-5.6-sol",
    ) -> None:
        self.output = output or _valid_output()
        self.query_output = query_output or _valid_query_output()
        self.model = model
        self.requests: list[DocumentVisionRequest] = []
        self.query_requests: list[DocumentQueryVisionRequest] = []

    async def interpret_document(
        self,
        request: DocumentVisionRequest,
    ) -> DocumentVisionResponse:
        self.requests.append(request)
        return DocumentVisionResponse(
            response_id="resp_overview_controlled",
            model=self.model,
            output=self.output,
        )

    async def query_document(
        self,
        request: DocumentQueryVisionRequest,
    ) -> DocumentVisionResponse:
        self.query_requests.append(request)
        return DocumentVisionResponse(
            response_id="resp_query_controlled",
            model=self.model,
            output=self.query_output,
        )


def test_openai_adapter_sends_one_neutral_nonstored_overview_request() -> None:
    class ControlledResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(
                id="resp_openai_controlled",
                model="gpt-5.6-sol",
                output_text=json.dumps(_valid_output()),
            )

    responses = ControlledResponses()
    runtime = OpenAIDocumentVisionRuntime(
        load_model_runtime_config().document_vlm,
        client=SimpleNamespace(responses=responses),
    )
    pages = tuple(
        RenderedDocumentPage(
            page_number=number,
            image_path=Path(f"page_{number:04d}.png"),
            image_sha256=f"sha{number}",
            image_data_url=f"data:image/png;base64,page{number}",
            text=f"page {number} text",
        )
        for number in range(1, 7)
    )
    request = DocumentVisionRequest(
        context_ref=_TEST_DOCUMENT_REF,
        source_sha256="a" * 64,
        pages=pages,
    )

    response = asyncio.run(runtime.interpret_document(request))

    assert response.response_id == "resp_openai_controlled"
    assert response.output == _valid_output()
    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call["store"] is False
    assert call["max_output_tokens"] == 4096
    assert call["reasoning"] == {"effort": "medium"}
    assert "tools" not in call
    serialized = json.dumps(call)
    for forbidden in (
        "product_requirement",
        "ContextNeed",
        "TBox",
        "ABox",
        "allowed_classes",
        "http://PAonto.com#",
        "entity_key",
        "triple_delta",
    ):
        assert forbidden not in serialized
    content = call["input"][0]["content"]
    assert [item["type"] for item in content] == [
        "input_text",
        *("input_image",) * 6,
    ]


def test_openai_adapter_sends_only_exact_question_and_document_pages() -> None:
    class ControlledResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(
                id="resp_query_controlled",
                model="gpt-5.6-sol",
                output_text=json.dumps(_valid_query_output()),
            )

    responses = ControlledResponses()
    runtime = OpenAIDocumentVisionRuntime(
        load_model_runtime_config().document_vlm,
        client=SimpleNamespace(responses=responses),
    )
    exact_question = "Which surface properties are explicitly specified?"
    request = DocumentQueryVisionRequest(
        question=exact_question,
        context_ref=_TEST_DOCUMENT_REF,
        source_sha256="a" * 64,
        pages=(
            RenderedDocumentPage(
                page_number=1,
                image_path=Path("page_0001.png"),
                image_sha256="b" * 64,
                image_data_url="data:image/png;base64,page1",
                text="page 1 text",
            ),
        ),
    )
    response = asyncio.run(runtime.query_document(request))
    assert response.output == _valid_query_output()
    call = responses.calls[0]
    assert call["store"] is False
    serialized = json.dumps(call)
    assert exact_question in serialized
    content = call["input"][0]["content"]
    request_payload = json.loads(content[0]["text"])
    assert set(request_payload) == {"question", "document_pages"}
    assert request_payload["question"] == exact_question
    assert request_payload["document_pages"] == [
        {"page": 1, "extracted_text": "page 1 text"}
    ]
    for forbidden in (
        "product_requirement",
        "authorized_processes",
        "CADMeshRecord",
        "RGBDSegmentationRecord",
        "candidate_handle",
        "expected_answer",
    ):
        assert forbidden not in serialized


def test_document_schemas_are_neutral_and_use_supported_constraints() -> None:
    overview_schema = document_interpretation_schema()
    serialized = json.dumps(overview_schema)

    assert "uniqueItems" not in serialized
    for forbidden in (
        "entities",
        "relations",
        "literal_facts",
        "class_iri",
        "predicate_iri",
        "TBox",
        "ABox",
    ):
        assert forbidden not in serialized
    assert set(overview_schema["properties"]) == {
        "summary",
        "observations",
        "uncertainty",
    }
    query_schema = document_query_schema()
    serialized_query = json.dumps(query_schema)
    assert "fact_kind" not in serialized_query
    assert "required_process" not in serialized_query
    assert set(query_schema["properties"]) == {"status", "claims", "uncertainty"}
    claim_schema = query_schema["properties"]["claims"]["items"]
    assert claim_schema["properties"]["predicate_text"] == {"type": "string"}
    assert claim_schema["properties"]["arguments"] == {
        "type": "array",
        "items": {"type": "string"},
    }


def test_source_index_is_requirement_blind_and_query_is_exact_and_document_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(
        root,
        "a requirement that must not enter document indexing",
        tbox,
    )
    source_index = index_document_evidence(
        interaction_root=root,
        tbox=tbox,
        abox=abox,
        served_context=_served_document(_TEST_DOCUMENT_REF),
        operation_number=1,
    )
    snapshot = _read_json(source_index.source_index_record_path)
    assert snapshot["record_type"] == "DocumentSourceIndexRecord"
    assert snapshot["status"] == "accepted"
    assert [page["page"] for page in snapshot["source_index"]["pages"]] == list(
        range(1, 7)
    )
    serialized_index = json.dumps(snapshot)
    assert abox.product_requirement not in serialized_index
    assert "summary" not in snapshot["source_index"]
    assert "observations" not in snapshot["source_index"]

    exact_question = "What ordered items are visibly shown on the cited page?"
    vision = ControlledVisionRuntime()
    query = asyncio.run(
        query_document_evidence(
            interaction_root=root,
            source_index_record_path=source_index.source_index_record_path,
            operation_number=1,
            question=exact_question,
            config=load_model_runtime_config().document_vlm,
            vision_runtime=vision,
        )
    )
    assert len(vision.query_requests) == 1
    request = vision.query_requests[0]
    assert request.question == exact_question
    assert len(request.pages) == 6
    assert not hasattr(request, "requirement")
    assert not hasattr(request, "ontology")
    assert not hasattr(request, "candidates")
    record = _read_json(query.record_path)
    assert record["record_type"] == "DocumentQueryRecord"
    assert record["question"] == exact_question
    assert record["status"] == "supported"
    assert record["claims"][0]["predicate_text"] == "shown_in_order"
    assert record["claims"][0]["evidence_refs"] == [
        f"{_TEST_DOCUMENT_REF}#page=4"
    ]


def test_historical_overview_cannot_be_used_as_a_dynamic_query_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    overview_path = root / "products/grounding/document_evidence/overview_0001.json"
    overview_path.parent.mkdir(parents=True)
    overview_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "record_type": "DocumentOverviewRecord",
                "producer": "document_evidence",
                "overview": {"pages": []},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DocumentInterpretationError, match="source index is invalid"):
        asyncio.run(
            query_document_evidence(
                interaction_root=root,
                source_index_record_path=overview_path,
                operation_number=1,
                question="What is stated?",
                config=load_model_runtime_config().document_vlm,
                vision_runtime=ControlledVisionRuntime(),
            )
        )


def test_unsupported_leading_document_question_returns_no_claim(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "paint a component", tbox)
    source_index = index_document_evidence(
        interaction_root=root,
        tbox=tbox,
        abox=abox,
        served_context=_served_document(_TEST_DOCUMENT_REF),
        operation_number=1,
    )
    vision = ControlledVisionRuntime(
        query_output={
            "status": "insufficient_evidence",
            "claims": [],
            "uncertainty": [
                {
                    "description": "The requested finish is not specified.",
                    "evidence_pages": [4],
                }
            ],
        }
    )
    result = asyncio.run(
        query_document_evidence(
            interaction_root=root,
            source_index_record_path=source_index.source_index_record_path,
            operation_number=1,
            question="Does the page require a blue finish?",
            config=load_model_runtime_config().document_vlm,
            vision_runtime=vision,
        )
    )
    assert result.status == "insufficient_evidence"
    assert result.record["claims"] == []


def test_document_query_rejects_changed_source_index(tmp_path: Path) -> None:
    root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(root, "assemble a component", tbox)
    source_index = index_document_evidence(
        interaction_root=root,
        tbox=tbox,
        abox=abox,
        served_context=_served_document(_TEST_DOCUMENT_REF),
        operation_number=1,
    )
    changed = _read_json(source_index.source_index_record_path)
    changed["source_index"]["pages"][0]["extracted_text"] = "changed"
    source_index.source_index_record_path.write_text(
        json.dumps(changed),
        encoding="utf-8",
    )
    with pytest.raises(DocumentInterpretationError, match="fingerprint"):
        asyncio.run(
            query_document_evidence(
                interaction_root=root,
                source_index_record_path=source_index.source_index_record_path,
                operation_number=1,
                question="What is shown?",
                config=load_model_runtime_config().document_vlm,
                vision_runtime=ControlledVisionRuntime(),
            )
        )


def test_overview_cache_hit_is_assertion_free_and_model_change_invalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_model_runtime_config().document_vlm
    served_context = _served_document(_TEST_DOCUMENT_REF)
    cache_root = tmp_path / "source_cache"
    vision = ControlledVisionRuntime()

    first = asyncio.run(
        prepare_document_overview(
            served_context=served_context,
            cache_root=cache_root,
            config=config,
            vision_runtime=vision,
        )
    )
    second = asyncio.run(
        prepare_document_overview(
            served_context=served_context,
            cache_root=cache_root,
            config=config,
            vision_runtime=vision,
        )
    )

    assert first.cache_status == "miss"
    assert second.cache_status == "hit"
    assert first.record_path == second.record_path
    assert len(vision.requests) == 1
    record = _read_json(first.record_path)
    assert record["record_type"] == "DocumentOverviewRecord"
    assert "entities" not in record
    assert "relations" not in record
    assert "assertions" not in record
    assert record["observations"][0]["observation_id"] == "observation_0001"
    assert [page["page"] for page in record["pages"]] == list(range(1, 7))
    assert all(page["extracted_text"] for page in record["pages"])
    assert all(len(page["image_sha256"]) == 64 for page in record["pages"])
    assert [page["evidence_ref"] for page in record["pages"]] == [
        f"{_TEST_DOCUMENT_REF}#page={page}" for page in range(1, 7)
    ]

    changed_config = replace(config, model="controlled-new-model")
    changed_vision = ControlledVisionRuntime(model="controlled-new-model")
    changed = asyncio.run(
        prepare_document_overview(
            served_context=served_context,
            cache_root=cache_root,
            config=changed_config,
            vision_runtime=changed_vision,
        )
    )
    assert changed.cache_status == "miss"
    assert changed.record_path != first.record_path
    assert len(changed_vision.requests) == 1

    monkeypatch.setattr(
        "cais_spade_llm.spec2primitives.tools.document_evidence.interpreter."
        "DOCUMENT_OVERVIEW_SCHEMA_VERSION",
        DOCUMENT_OVERVIEW_SCHEMA_VERSION + 1,
    )
    stale_status = document_overview_cache_status(
        _TEST_DOCUMENT_REF,
        cache_root=cache_root,
        config=config,
    )
    assert stale_status["source_status"] == "valid"
    assert stale_status["overview_status"] == "stale"
    schema_vision = ControlledVisionRuntime()
    schema_changed = asyncio.run(
        prepare_document_overview(
            served_context=served_context,
            cache_root=cache_root,
            config=config,
            vision_runtime=schema_vision,
        )
    )
    assert schema_changed.cache_status == "miss"
    assert schema_changed.record_path != first.record_path
    assert len(schema_vision.requests) == 1

    interaction_root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(
        interaction_root,
        "assemble Medium Gear",
        tbox,
    )
    interpretation = asyncio.run(
        interpret_document_evidence(
            interaction_root=interaction_root,
            tbox=tbox,
            abox=abox,
            served_context=served_context,
            operation_number=1,
            config=config,
            vision_runtime=vision,
        )
    )
    assert interpretation.delta["assertions"] == []
    snapshot = _read_json(interpretation.overview_record_path)
    assert snapshot["record_type"] == "DocumentOverviewRecord"
    assert snapshot["producer"] == "document_evidence"


def test_one_document_interpretation_persists_every_ordered_page(
    tmp_path: Path,
) -> None:
    config = load_model_runtime_config().document_vlm
    served_context = _served_document(_TEST_DOCUMENT_REF)
    vision = ControlledVisionRuntime()
    interaction_root = tmp_path / "interaction"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(
        interaction_root,
        "a requirement that must not shape document extraction",
        tbox,
    )

    result = asyncio.run(
        interpret_document_evidence(
            interaction_root=interaction_root,
            tbox=tbox,
            abox=abox,
            served_context=served_context,
            operation_number=1,
            config=config,
            vision_runtime=vision,
        )
    )
    snapshot = _read_json(result.overview_record_path)
    overview = snapshot["overview"]
    assert snapshot["schema_version"] == 3
    assert [page["page"] for page in overview["pages"]] == list(range(1, 7))
    assert [page["extracted_text"] for page in overview["pages"]] == [
        page.text for page in vision.requests[0].pages
    ]
    assert snapshot["evidence_refs"] == [
        _TEST_DOCUMENT_REF,
        *(f"{_TEST_DOCUMENT_REF}#page={page}" for page in range(1, 7)),
    ]
    assert result.delta["unresolved_evidence_needs"] == []


def test_invalid_overview_is_removed_atomically_and_abox_is_unchanged(
    tmp_path: Path,
) -> None:
    output = _valid_output()
    observations = output["observations"]
    assert isinstance(observations, list)
    observation = observations[0]
    assert isinstance(observation, dict)
    observation["evidence_pages"] = [4, 4]
    interaction_root = tmp_path / "interaction"
    config = load_model_runtime_config().document_vlm
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(
        interaction_root,
        "assemble Medium Gear",
        tbox,
    )

    with pytest.raises(DocumentInterpretationError, match="must be unique"):
        asyncio.run(
            interpret_document_evidence(
                interaction_root=interaction_root,
                tbox=tbox,
                abox=abox,
                served_context=_served_document(_TEST_DOCUMENT_REF),
                operation_number=1,
                config=config,
                vision_runtime=ControlledVisionRuntime(output),
            )
        )

    assert not list((tmp_path / "source_cache").rglob("overview.json"))
    manifest = _read_json(interaction_root / "products/grounding/ontology/abox_manifest.json")
    assert manifest["delta_count"] == 0
    assert manifest["accepted_assertion_count"] == 0


def test_bad_request_diagnostic_is_sanitized_chained_and_persisted(
    tmp_path: Path,
) -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(
        400,
        request=request,
        headers={"x-request-id": "req_document_controlled"},
    )
    api_error = BadRequestError(
        "raw exception text that must not persist",
        response=response,
        body={
            "message": "Unsupported schema keyword: uniqueItems",
            "type": "invalid_request_error",
            "param": "text.format.schema",
            "code": "invalid_json_schema",
            "private_body_field": "must not persist",
            "prompt": "controlled prompt that must not persist",
            "image_url": "data:image/png;base64,must-not-persist",
        },
    )

    class ControlledResponses:
        async def create(self, **kwargs: object) -> object:
            del kwargs
            raise api_error

    config = load_model_runtime_config().document_vlm
    runtime = OpenAIDocumentVisionRuntime(
        config,
        client=SimpleNamespace(responses=ControlledResponses()),
    )
    interaction_root = tmp_path / "bad_request"
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(
        interaction_root,
        "assemble Medium Gear",
        tbox,
    )

    with pytest.raises(DocumentInterpretationError) as exc_info:
        asyncio.run(
            interpret_document_evidence(
                interaction_root=interaction_root,
                tbox=tbox,
                abox=abox,
                served_context=_served_document(_TEST_DOCUMENT_REF),
                operation_number=1,
                config=config,
                vision_runtime=runtime,
            )
        )

    assert exc_info.value.__cause__ is api_error
    trace_path = interaction_root / "products/grounding/document_evidence/interpretation_0001.json"
    trace_text = trace_path.read_text(encoding="utf-8")
    assert "req_document_controlled" in trace_text
    for forbidden in (
        "private_body_field",
        "controlled prompt",
        "raw exception text",
        "OPENAI_API_KEY",
        "data:image",
        "base64",
    ):
        assert forbidden not in trace_text


def test_second_registered_pdf_uses_the_same_overview_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references_root = tmp_path / "references/products"
    references_root.mkdir(parents=True)
    pdf_path = references_root / "Second_Product_Manual.pdf"
    Image.new("RGB", (320, 200), color="white").save(pdf_path, format="PDF")
    inventory_path = references_root / "approved_sources.json"
    inventory_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "context_ref": pdf_path.name,
                        "evidence_type": "document",
                        "repository_path": "references/products/Second_Product_Manual.pdf",
                        "source_url": "https://example.test/second-manual",
                        "page_count": 1,
                        "source_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(exact_ref_resolver, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(exact_ref_resolver, "_REFERENCES_ROOT", references_root)
    monkeypatch.setattr(exact_ref_resolver, "_INVENTORY_PATH", inventory_path)
    served = _served_document(pdf_path.name)
    vision = ControlledVisionRuntime(
        {
            "summary": "A second product manual.",
            "observations": [
                {
                    "description": "The page shows a second product.",
                    "evidence_pages": [1],
                }
            ],
            "uncertainty": [],
        }
    )

    result = asyncio.run(
        prepare_document_overview(
            served_context=served,
            cache_root=tmp_path / "cache",
            config=load_model_runtime_config().document_vlm,
            vision_runtime=vision,
        )
    )

    assert result.context_ref == "Second_Product_Manual.pdf"
    assert result.record["page_count"] == 1
    document_evidence = served["document_evidence"]
    assert isinstance(document_evidence, dict)
    assert result.record["source_sha256"] == document_evidence["source_sha256"]
    assert vision.requests[0].context_ref == "Second_Product_Manual.pdf"

    Image.new("RGB", (320, 200), color="black").save(pdf_path, format="PDF")
    changed = resolve_context_ref({"context_ref": pdf_path.name})

    assert changed["rejection"]["reason"] == "source_hash_mismatch"
    assert "served_context" not in changed
    assert len(vision.requests) == 1


def test_prepare_command_supports_all_and_one_exact_context_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], Path]] = []

    async def controlled_prepare(
        context_refs: tuple[str, ...],
        *,
        cache_root: Path,
    ) -> int:
        calls.append((context_refs, cache_root))
        return 0

    monkeypatch.setattr(
        "cais_spade_llm.spec2primitives.tools.document_evidence.prepare.approved_document_refs",
        lambda: ("First.pdf", "Second.pdf"),
    )
    monkeypatch.setattr(
        "cais_spade_llm.spec2primitives.tools.document_evidence.prepare.prepare_documents",
        controlled_prepare,
    )

    assert prepare_document_main(["--all", "--cache-root", str(tmp_path)]) == 0
    assert (
        prepare_document_main(
            [
                "--context-ref",
                "Second.pdf",
                "--cache-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert calls == [
        (("First.pdf", "Second.pdf"), tmp_path),
        (("Second.pdf",), tmp_path),
    ]


def test_diagnostic_keeps_overview_proposal_and_assertions_as_separate_stages(
    tmp_path: Path,
) -> None:
    class ProposalAgent:
        async def ask_llm_structured(
            self,
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            tool_executor: Any = None,
            max_tool_rounds: int = 3,
        ) -> dict[str, Any]:
            assert "initialized_specification_iri" in prompt
            assert response_format["name"] == "spec2primitives_grounding_result"
            assert tools == []
            assert tool_executor is not None
            assert max_tool_rounds == 1
            evidence_ref = f"{_TEST_DOCUMENT_REF}#page=4"
            proposal = {
                "target_feature": {
                    "required_process": {
                        "process_iri": "https://cais-spade-llm.local/process/assembly",
                        "evidence_refs": [evidence_ref],
                    },
                    "current_state": {
                        "statement": {
                            "text": "The Medium Gear is currently separate.",
                            "evidence_refs": [evidence_ref],
                        },
                        "state_values": [],
                    },
                    "desired_state": {
                        "statement": {
                            "text": "The Medium Gear is assembled as requested.",
                            "evidence_refs": [evidence_ref],
                        },
                        "state_values": [],
                    },
                }
            }
            return {"result": proposal}

    result = asyncio.run(
        run_document_interpretation_diagnostic(
            interaction_root=tmp_path / "diagnostic",
            product_requirement="assemble Medium Gear",
            context_ref=_TEST_DOCUMENT_REF,
            product_agent=ProposalAgent(),
            ontology_config=ontology_config(),
            config=load_model_runtime_config().document_vlm,
            vision_runtime=ControlledVisionRuntime(),
        )
    )

    assert result["status"] == "accepted"
    assert result["overview"]["status"] == "accepted"
    assert result["ontology_proposal"]["status"] == "accepted"
    proposal_record = _read_json(
        tmp_path / "diagnostic/products/grounding/ontology_grounding/proposal_0001.json"
    )
    assert proposal_record["schema_version"] == 9
    assert len(result["accepted_assertions"]) == 7
    assert result["failure"] is None


def _served_document(context_ref: str) -> dict[str, object]:
    resolved = resolve_context_ref({"context_ref": context_ref})
    served = resolved.get("served_context")
    assert isinstance(served, dict), resolved
    return served


def _valid_output() -> dict[str, object]:
    return {
        "summary": "Assembly instructions for a gear product.",
        "observations": [
            {
                "description": "The specification identifies a Medium Gear.",
                "evidence_pages": [4],
            },
            {
                "description": "A depicted assembly sequence uses the gear.",
                "evidence_pages": [4],
            },
        ],
        "uncertainty": [
            {
                "description": "The page does not state an insertion tolerance.",
                "evidence_pages": [4],
            }
        ],
    }


def _valid_query_output() -> dict[str, object]:
    return {
        "status": "supported",
        "claims": [
            {
                "predicate_text": "shown_in_order",
                "arguments": ["first visible item", "second visible item", "third visible item"],
                "evidence_pages": [4],
                "uncertainty": [],
            }
        ],
        "uncertainty": [],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

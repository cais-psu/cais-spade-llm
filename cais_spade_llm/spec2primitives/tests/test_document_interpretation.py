"""Tests for evidence-first approved PDF preparation and interpretation."""

from __future__ import annotations

import asyncio
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
    DocumentVisionRequest,
    DocumentVisionResponse,
    OpenAIDocumentVisionRuntime,
    TargetedDocumentVisionRequest,
    document_overview_cache_status,
    inspect_document_evidence,
    interpret_document_evidence,
    prepare_document_overview,
    run_document_interpretation_diagnostic,
)
from cais_spade_llm.spec2primitives.tools.document_evidence.interpreter import (
    RenderedDocumentPage,
    document_interpretation_schema,
    targeted_document_evidence_schema,
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
        targeted_output: dict[str, object] | None = None,
        model: str = "gpt-5.4-mini-2026-03-17",
    ) -> None:
        self.output = output or _valid_output()
        self.targeted_output = targeted_output
        self.model = model
        self.requests: list[DocumentVisionRequest] = []
        self.targeted_requests: list[TargetedDocumentVisionRequest] = []

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

    async def inspect_document_pages(
        self,
        request: TargetedDocumentVisionRequest,
    ) -> DocumentVisionResponse:
        self.targeted_requests.append(request)
        output = self.targeted_output or {
            "observations": [
                {
                    "description": "The requested item is visible in the document.",
                    "evidence_pages": [request.pages[0].page_number],
                }
            ],
            "uncertainty": [],
        }
        return DocumentVisionResponse(
            response_id="resp_targeted_controlled",
            model=self.model,
            output=output,
        )


def test_openai_adapter_sends_one_neutral_nonstored_overview_request() -> None:
    class ControlledResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(
                id="resp_openai_controlled",
                model="gpt-5.4-mini-2026-03-17",
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
    assert call["reasoning"] == {"effort": "low"}
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


def test_document_schemas_are_neutral_and_use_supported_constraints() -> None:
    overview_schema = document_interpretation_schema()
    targeted_schema = targeted_document_evidence_schema()
    serialized = json.dumps([overview_schema, targeted_schema])

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
    assert set(targeted_schema["properties"]) == {
        "observations",
        "uncertainty",
    }


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


def test_targeted_evidence_inspects_every_nist_page_in_order(
    tmp_path: Path,
) -> None:
    config = load_model_runtime_config().document_vlm
    served_context = _served_document(_TEST_DOCUMENT_REF)
    vision = ControlledVisionRuntime()
    overview = asyncio.run(
        prepare_document_overview(
            served_context=served_context,
            cache_root=tmp_path / "source_cache",
            config=config,
            vision_runtime=vision,
        )
    )
    snapshot = {
        "schema_version": 1,
        "record_type": "DocumentOverviewRecord",
        "producer": "document_evidence",
        "operation_number": 1,
        "cache_status": overview.cache_status,
        "cache_record_ref": str(overview.record_path),
        "evidence_refs": [
            _TEST_DOCUMENT_REF,
            *(f"{_TEST_DOCUMENT_REF}#page={page}" for page in range(1, 7)),
        ],
        "overview": dict(overview.record),
    }

    evidence = asyncio.run(
        inspect_document_evidence(
            interaction_root=tmp_path / "interaction",
            overview_snapshot=snapshot,
            evidence_question="What does page 4 say about assembling Medium Gear?",
            config=config,
            vision_runtime=vision,
        )
    )

    assert [page.page_number for page in vision.targeted_requests[0].pages] == list(
        range(1, 7)
    )
    assert 4 in [page.page_number for page in vision.targeted_requests[0].pages]
    assert evidence.record["selected_pages"] == list(range(1, 7))

    invalid_vision = ControlledVisionRuntime(
        targeted_output={
            "observations": [
                {
                    "description": "This citation is outside the document.",
                    "evidence_pages": [7],
                }
            ],
            "uncertainty": [],
        }
    )
    with pytest.raises(DocumentInterpretationError, match="evidence page is invalid"):
        asyncio.run(
            inspect_document_evidence(
                interaction_root=tmp_path / "invalid_interaction",
                overview_snapshot=snapshot,
                evidence_question="What does the document say?",
                config=config,
                vision_runtime=invalid_vision,
            )
        )


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
    manifest = _read_json(
        interaction_root / "products/grounding/ontology/abox_manifest.json"
    )
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
    trace_path = (
        interaction_root
        / "products/grounding/document_evidence/interpretation_0001.json"
    )
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
    changed_served = _served_document(pdf_path.name)
    changed = asyncio.run(
        prepare_document_overview(
            served_context=changed_served,
            cache_root=tmp_path / "cache",
            config=load_model_runtime_config().document_vlm,
            vision_runtime=vision,
        )
    )
    assert changed.cache_status == "miss"
    assert changed.record_path != result.record_path
    assert changed.record["source_sha256"] != result.record["source_sha256"]
    assert len(vision.requests) == 2


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
        "cais_spade_llm.spec2primitives.tools.document_evidence.prepare."
        "approved_document_refs",
        lambda: ("First.pdf", "Second.pdf"),
    )
    monkeypatch.setattr(
        "cais_spade_llm.spec2primitives.tools.document_evidence.prepare."
        "prepare_documents",
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
        ) -> dict[str, Any]:
            assert "initialized_specification_iri" in prompt
            proposal_schema = response_format["schema"]["properties"][
                "ontology_grounding_proposal"
            ]["properties"]
            evidence_refs = proposal_schema["evidence_refs"]["items"]["enum"]
            return {
                "ontology_grounding_proposal": {
                    "individuals": [
                        {
                            "individual_index": 1,
                            "class_iri": "http://PAonto.com#feature",
                        }
                    ],
                    "relations": [
                        {
                            "subject_kind": "specification",
                            "subject_individual_index": None,
                            "subject_iri": None,
                            "predicate_iri": "http://PAonto.com#defines",
                            "object_kind": "new_individual",
                            "object_individual_index": 1,
                            "object_iri": None,
                        },
                        {
                            "subject_kind": "existing_individual",
                            "subject_individual_index": None,
                            "subject_iri": (
                                "https://cais-spade-llm.local/process/assembly"
                            ),
                            "predicate_iri": "http://PAonto.com#realizes",
                            "object_kind": "new_individual",
                            "object_individual_index": 1,
                            "object_iri": None,
                        },
                    ],
                    "literal_facts": [],
                    "context_summary": (
                        "The document describes the requested assembly context."
                    ),
                    "evidence_refs": [evidence_refs[0]],
                    "missing_information": [],
                }
            }

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
    assert result["targeted_evidence"] is None
    assert result["grounding_session"]["status"] == "ready_for_ontology"
    assert result["ontology_proposal"]["status"] == "accepted"
    assert len(result["accepted_assertions"]) == 3
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

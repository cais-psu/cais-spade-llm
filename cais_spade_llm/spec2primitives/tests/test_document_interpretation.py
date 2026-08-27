"""Tests for the Phase 4.1 OpenAI document interpretation boundary."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import BadRequestError
from PIL import Image
from rdflib import RDF, Graph, Namespace

from cais_spade_llm.spec2primitives import spec2primitives_ui
from cais_spade_llm.spec2primitives.adapters.ui_runtime import (
    Spec2PrimitivesUIRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    initialize_interaction_abox,
)
from cais_spade_llm.spec2primitives.config import load_model_runtime_config
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    PPR_NAMESPACE,
    ontology_config,
)
from cais_spade_llm.spec2primitives.tools.document_evidence import (
    DOCUMENT_CONTEXT_REF,
    DocumentInterpretationError,
    DocumentVisionRequest,
    DocumentVisionResponse,
    OpenAIDocumentVisionRuntime,
    interpret_document_evidence,
    run_document_interpretation_diagnostic,
)
from cais_spade_llm.spec2primitives.tools.document_evidence.interpreter import (
    RenderedDocumentPage,
    document_interpretation_schema,
)
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    resolve_context_ref,
)

PPR = Namespace(PPR_NAMESPACE)


class ControlledVisionRuntime:
    """Return one evidence-backed interpretation without a network call."""

    def __init__(self, output: dict[str, object] | None = None) -> None:
        self.output = output or _valid_output()
        self.requests: list[DocumentVisionRequest] = []

    async def interpret_document(
        self,
        request: DocumentVisionRequest,
    ) -> DocumentVisionResponse:
        self.requests.append(request)
        return DocumentVisionResponse(
            response_id="resp_controlled",
            model="gpt-5.4-mini-2026-03-17",
            output=self.output,
        )


def test_openai_adapter_sends_one_nonstored_structured_six_page_request() -> None:
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
    client = SimpleNamespace(responses=responses)
    runtime = OpenAIDocumentVisionRuntime(
        load_model_runtime_config().document_vlm,
        client=client,
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
        product_requirement="assemble Medium Gear",
        abox_view={"status": "unresolved"},
        tbox_classes=(str(PPR.feature), str(PPR.process)),
        tbox_object_properties=(str(PPR.defines), str(PPR.realizes)),
        tbox_datatype_properties=(),
        pages=pages,
    )

    response = asyncio.run(runtime.interpret_document(request))

    assert response.response_id == "resp_openai_controlled"
    assert response.output == _valid_output()
    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call["model"] == "gpt-5.4-mini-2026-03-17"
    assert call["store"] is False
    assert call["max_output_tokens"] == 4096
    assert call["reasoning"] == {"effort": "low"}
    assert "tools" not in call
    content = call["input"][0]["content"]
    assert [item["type"] for item in content] == ["input_text", *("input_image",) * 6]
    assert [item["detail"] for item in content[1:]] == ["high"] * 6
    assert call["text"]["format"]["strict"] is True
    assert call["text"]["format"]["schema"]["additionalProperties"] is False


def test_openai_document_schema_uses_supported_constraints() -> None:
    schema = document_interpretation_schema()
    schema_text = json.dumps(schema)

    assert "uniqueItems" not in schema_text
    properties = schema["properties"]
    assert isinstance(properties, dict)
    literal_facts = properties["literal_facts"]
    assert isinstance(literal_facts, dict)
    literal_items = literal_facts["items"]
    assert isinstance(literal_items, dict)
    literal_properties = literal_items["properties"]
    assert isinstance(literal_properties, dict)
    assert literal_properties["value"] == {
        "anyOf": [
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
        ]
    }
    entities = properties["entities"]
    assert isinstance(entities, dict)
    entity_items = entities["items"]
    assert isinstance(entity_items, dict)
    entity_properties = entity_items["properties"]
    assert isinstance(entity_properties, dict)
    assert entity_properties["evidence_pages"] == {
        "type": "array",
        "items": {"type": "integer", "minimum": 1},
        "minItems": 1,
    }


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
    calls: list[dict[str, object]] = []

    class ControlledResponses:
        async def create(self, **kwargs: object) -> object:
            calls.append(kwargs)
            raise api_error

    config = load_model_runtime_config().document_vlm
    vision_runtime = OpenAIDocumentVisionRuntime(
        config,
        client=SimpleNamespace(responses=ControlledResponses()),
    )
    interaction_root = tmp_path / "bad_request"
    configured_ontology = ontology_config()
    tbox = configured_ontology.load_tbox()
    abox = initialize_interaction_abox(
        interaction_root,
        "assemble Medium Gear",
        tbox,
    )
    resolved = resolve_context_ref({"context_ref": DOCUMENT_CONTEXT_REF})
    served_context = resolved["served_context"]
    assert isinstance(served_context, dict)

    with pytest.raises(DocumentInterpretationError) as exc_info:
        asyncio.run(
            interpret_document_evidence(
                interaction_root=interaction_root,
                tbox=tbox,
                abox=abox,
                served_context=served_context,
                operation_number=1,
                config=config,
                vision_runtime=vision_runtime,
            )
        )

    assert len(calls) == 1
    assert exc_info.value.__cause__ is api_error
    expected_diagnostic = {
        "stage": "Phase 4.1 document_evidence OpenAI Responses call",
        "exception": "BadRequestError",
        "status_code": 400,
        "request_id": "req_document_controlled",
        "error_type": "invalid_request_error",
        "param": "text.format.schema",
        "code": "invalid_json_schema",
        "message": "Unsupported schema keyword: uniqueItems",
    }
    assert exc_info.value.diagnostic == expected_diagnostic
    assert "status_code=400" in str(exc_info.value)
    assert "param=text.format.schema" in str(exc_info.value)

    trace_path = (
        interaction_root
        / "products/grounding/document_evidence/interpretation_0001.json"
    )
    trace = _read_json(trace_path)
    assert trace["diagnostic"] == expected_diagnostic
    trace_text = trace_path.read_text(encoding="utf-8")
    for forbidden in (
        "private_body_field",
        "controlled prompt",
        "raw exception text",
        "OPENAI_API_KEY",
        "data:image",
        "base64",
    ):
        assert forbidden not in trace_text

    manifest = _read_json(
        interaction_root / "products/grounding/ontology/abox_manifest.json"
    )
    assert manifest["delta_count"] == 0
    assert not list(
        (interaction_root / "products/grounding/ontology").glob("delta_*.json")
    )


def test_diagnostic_renders_all_pages_and_merges_validated_delta(
    tmp_path: Path,
) -> None:
    requirement = "  assemble Medium Gear exactly  "
    vision = ControlledVisionRuntime()
    interaction_root = tmp_path / "document_diagnostic"

    result = asyncio.run(
        run_document_interpretation_diagnostic(
            interaction_root=interaction_root,
            product_requirement=requirement,
            ontology_config=ontology_config(),
            config=load_model_runtime_config().document_vlm,
            vision_runtime=vision,
        )
    )

    assert result["status"] == "accepted"
    assert result["product_requirement"] == requirement
    assert result["page_count"] == 6
    assert result["model"] == "gpt-5.4-mini-2026-03-17"
    assert result["assertion_count"] == 4
    assert len(result["supported_findings"]) == 4
    assert len(result["uncertainty"]) == 1
    assert len(result["unresolved_evidence_needs"]) == 1
    assert result["failure"] is None
    assert len(vision.requests) == 1
    request = vision.requests[0]
    assert request.product_requirement == requirement
    assert request.abox_view["product_requirement"] == requirement
    assert [page.page_number for page in request.pages] == list(range(1, 7))
    assert all(page.image_data_url.startswith("data:image/png;base64,") for page in request.pages)
    for page in request.pages:
        with Image.open(page.image_path) as image:
            assert abs(image.width - 1600) <= 2

    graph = Graph().parse(str(result["abox_path"]), format="turtle")
    feature = next(graph.subjects(RDF.type, PPR.feature))
    process = next(graph.subjects(RDF.type, PPR.process))
    specification = next(graph.subjects(RDF.type, PPR.specification))
    assert (specification, PPR.defines, feature) in graph
    assert (process, PPR.realizes, feature) in graph

    trace = _read_json(Path(str(result["trace_path"])))
    assert trace["store"] is False
    assert len(trace["pages"]) == 6
    assert trace["compiled_delta"]["assertions"][0]["evidence_refs"] == [
        "NIST_assembly_instructions.pdf#page=4"
    ]
    assert "OPENAI_API_KEY" not in json.dumps(trace)
    diagnostic = _read_json(Path(str(result["diagnostic_record_path"])))
    assert diagnostic == result
    assert "context understanding complete" not in json.dumps(diagnostic)


def test_invalid_vision_vocabulary_is_rejected_without_abox_delta(
    tmp_path: Path,
) -> None:
    output = _valid_output()
    output["entities"][0]["class_iri"] = "https://unknown.example/Class"
    interaction_root = tmp_path / "rejected"

    result = asyncio.run(
        run_document_interpretation_diagnostic(
            interaction_root=interaction_root,
            product_requirement="assemble Medium Gear",
            ontology_config=ontology_config(),
            config=load_model_runtime_config().document_vlm,
            vision_runtime=ControlledVisionRuntime(output),
        )
    )

    assert result["status"] == "rejected"
    assert result["failure"]["reason"] == "document_interpretation_rejected"
    assert result["trace_path"] is not None
    assert result["abox_path"] is not None
    manifest = _read_json(interaction_root / "products/grounding/ontology/abox_manifest.json")
    assert manifest["delta_count"] == 0
    assert manifest["accepted_assertion_count"] == 0
    assert not list((interaction_root / "products/grounding/ontology").glob("delta_*.json"))
    trace = _read_json(
        interaction_root / "products/grounding/document_evidence/interpretation_0001.json"
    )
    assert "not declared in the TBox" in trace["failure"]


def test_duplicate_evidence_pages_remain_rejected_without_abox_delta(
    tmp_path: Path,
) -> None:
    output = _valid_output()
    entities = output["entities"]
    assert isinstance(entities, list)
    first_entity = entities[0]
    assert isinstance(first_entity, dict)
    first_entity["evidence_pages"] = [4, 4]
    interaction_root = tmp_path / "duplicate_pages"

    result = asyncio.run(
        run_document_interpretation_diagnostic(
            interaction_root=interaction_root,
            product_requirement="assemble Medium Gear",
            ontology_config=ontology_config(),
            config=load_model_runtime_config().document_vlm,
            vision_runtime=ControlledVisionRuntime(output),
        )
    )

    assert result["status"] == "rejected"
    assert "evidence pages must be unique" in result["failure"]["message"]
    manifest = _read_json(
        interaction_root / "products/grounding/ontology/abox_manifest.json"
    )
    assert manifest["delta_count"] == 0
    assert not list(
        (interaction_root / "products/grounding/ontology").glob("delta_*.json")
    )
    trace = _read_json(
        interaction_root
        / "products/grounding/document_evidence/interpretation_0001.json"
    )
    assert trace["diagnostic"] is None


def test_ui_document_diagnostic_is_separate_and_fails_closed(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    unavailable = Spec2PrimitivesUIRuntime(
        dual_gazebo=object(),
        product_agent=object(),  # type: ignore[arg-type]
        contexts_root=tmp_path,
        document_diagnostic_unavailable_reason="controlled unavailable",
    )
    result = asyncio.run(
        spec2primitives_ui._run_document_diagnostic_ui(
            unavailable,
            "assemble Medium Gear",
        )
    )
    assert result["status"] == "unavailable"
    assert result["failure"]["message"] == "controlled unavailable"

    calls: list[dict[str, object]] = []

    async def controlled_runner(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"status": "accepted", "failure": None}

    monkeypatch.setattr(
        spec2primitives_ui,
        "run_document_interpretation_diagnostic",
        controlled_runner,
    )
    configured = Spec2PrimitivesUIRuntime(
        dual_gazebo=object(),
        product_agent=object(),  # type: ignore[arg-type]
        contexts_root=tmp_path,
        ontology_config=ontology_config(),
        model_config=load_model_runtime_config(),
        document_vision_runtime=ControlledVisionRuntime(),
    )
    result = asyncio.run(
        spec2primitives_ui._run_document_diagnostic_ui(
            configured,
            "assemble Medium Gear",
        )
    )

    assert result == {"status": "accepted", "failure": None}
    assert len(calls) == 1
    assert str(calls[0]["interaction_root"]).startswith(str(tmp_path / "document_diagnostic_"))
    source = inspect.getsource(spec2primitives_ui._render_document_interpretation_diagnostic)
    assert "context understanding complete" not in source


def _valid_output() -> dict[str, object]:
    return {
        "entities": [
            {
                "key": "required_medium_gear",
                "class_iri": str(PPR.feature),
                "evidence_pages": [4],
            },
            {
                "key": "gear_assembly_process",
                "class_iri": str(PPR.process),
                "evidence_pages": [4],
            },
        ],
        "relations": [
            {
                "subject_key": "specification",
                "predicate_iri": str(PPR.defines),
                "object_key": "required_medium_gear",
                "evidence_pages": [4],
            },
            {
                "subject_key": "gear_assembly_process",
                "predicate_iri": str(PPR.realizes),
                "object_key": "required_medium_gear",
                "evidence_pages": [4],
            },
        ],
        "literal_facts": [],
        "uncertainty": [
            {
                "description": "The manual does not state an insertion tolerance.",
                "evidence_pages": [4],
            }
        ],
        "unresolved_evidence_needs": [
            {
                "description": "A measured receiving pose is not present in the manual.",
                "evidence_pages": [4],
            }
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

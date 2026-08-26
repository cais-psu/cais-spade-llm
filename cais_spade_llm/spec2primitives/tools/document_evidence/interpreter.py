"""Interpret approved PDF evidence through an injected vision boundary."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pdfplumber
from openai import AsyncOpenAI, OpenAIError
from pdfminer.pdfparser import PDFSyntaxError
from rdflib import RDF

from cais_spade_llm.spec2primitives.agents.pa.context_grounding import (
    compact_abox_view,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import ABoxSnapshot
from cais_spade_llm.spec2primitives.config import DocumentVLMConfig
from cais_spade_llm.spec2primitives.ontology import TBoxSnapshot
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_document_path,
)

DOCUMENT_CONTEXT_REF = "NIST_assembly_instructions.pdf"
_PRODUCER = "document_evidence"
_OUTPUT_NAME = "spec2primitives_document_interpretation"
_ENTITY_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_OUTPUT_KEYS = {
    "entities",
    "relations",
    "literal_facts",
    "uncertainty",
    "unresolved_evidence_needs",
}
_ENTITY_KEYS = {"key", "class_iri", "evidence_pages"}
_RELATION_KEYS = {"subject_key", "predicate_iri", "object_key", "evidence_pages"}
_LITERAL_KEYS = {
    "subject_key",
    "predicate_iri",
    "value",
    "datatype_iri",
    "language",
    "evidence_pages",
}
_NOTE_KEYS = {"description", "evidence_pages"}


class DocumentInterpretationError(ValueError):
    """Raised when approved document evidence cannot be interpreted safely."""


@dataclass(frozen=True)
class RenderedDocumentPage:
    """Hold one ordered rendered page and its extracted text."""

    page_number: int
    image_path: Path
    image_sha256: str
    image_data_url: str
    text: str


@dataclass(frozen=True)
class DocumentVisionRequest:
    """Contain the bounded inputs visible to a document vision runtime."""

    product_requirement: str
    abox_view: Mapping[str, object]
    tbox_classes: tuple[str, ...]
    tbox_object_properties: tuple[str, ...]
    tbox_datatype_properties: tuple[str, ...]
    pages: tuple[RenderedDocumentPage, ...]


@dataclass(frozen=True)
class DocumentVisionResponse:
    """Return one structured interpretation and provider audit fields."""

    response_id: str
    model: str
    output: Mapping[str, object]


class DocumentVisionRuntime(Protocol):
    """Narrow injected boundary for one document vision request."""

    async def interpret_document(
        self,
        request: DocumentVisionRequest,
    ) -> DocumentVisionResponse:
        """Interpret all ordered pages in one structured call."""
        ...


@dataclass(frozen=True)
class DocumentInterpretationResult:
    """Return a generic delta plus the persisted Phase 4.1 audit record."""

    delta: Mapping[str, object]
    trace_path: Path
    page_count: int
    assertion_count: int
    uncertainty_count: int
    unresolved_evidence_need_count: int
    provider: str
    model: str
    response_id: str


class OpenAIDocumentVisionRuntime:
    """Call the OpenAI Responses API without exposing it to orchestration."""

    def __init__(
        self,
        config: DocumentVLMConfig,
        *,
        client: Any | None = None,
    ) -> None:
        """Create the adapter with an optional offline-test client."""
        self._config = config
        self._client = client or AsyncOpenAI(
            max_retries=0,
            timeout=config.timeout_seconds,
        )

    async def interpret_document(
        self,
        request: DocumentVisionRequest,
    ) -> DocumentVisionResponse:
        """Submit one ordered multimodal request with strict JSON output."""
        content: list[dict[str, object]] = [
            {
                "type": "input_text",
                "text": _request_text(request),
            }
        ]
        content.extend(
            {
                "type": "input_image",
                "image_url": page.image_data_url,
                "detail": self._config.image_detail,
            }
            for page in request.pages
        )
        try:
            response = await self._client.responses.create(
                model=self._config.model,
                instructions=_OPENAI_INSTRUCTIONS,
                input=[{"role": "user", "content": content}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": _OUTPUT_NAME,
                        "strict": True,
                        "schema": document_interpretation_schema(),
                    }
                },
                reasoning={"effort": self._config.reasoning_effort},
                max_output_tokens=self._config.max_output_tokens,
                store=False,
            )
        except OpenAIError as exc:
            raise DocumentInterpretationError(
                f"OpenAI document interpretation failed: {type(exc).__name__}."
            ) from exc

        output_text = getattr(response, "output_text", None)
        response_id = getattr(response, "id", None)
        model = getattr(response, "model", None)
        if not all(isinstance(value, str) and value for value in (output_text, response_id, model)):
            raise DocumentInterpretationError(
                "OpenAI returned an incomplete document interpretation response."
            )
        try:
            output = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise DocumentInterpretationError(
                "OpenAI document interpretation output was not JSON."
            ) from exc
        if not isinstance(output, Mapping):
            raise DocumentInterpretationError(
                "OpenAI document interpretation output was not an object."
            )
        return DocumentVisionResponse(
            response_id=response_id,
            model=model,
            output=output,
        )


async def interpret_document_evidence(
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    served_context: Mapping[str, object],
    operation_number: int,
    config: DocumentVLMConfig,
    vision_runtime: DocumentVisionRuntime,
) -> DocumentInterpretationResult:
    """Render, interpret, compile, and persist one approved PDF result.

    The returned delta remains untrusted until the caller passes it through the
    shared ABox validator. This function does not decide context completeness.
    """
    _validate_operation_number(operation_number)
    document_pages = _validated_served_document(served_context)
    root = Path(interaction_root).resolve()
    if abox.interaction_root != root or abox.tbox_fingerprint != tbox.fingerprint:
        raise DocumentInterpretationError(
            "Document interpretation ABox does not match its interaction and TBox."
        )
    if abox.product_requirement != compact_abox_view(abox)["product_requirement"]:
        raise DocumentInterpretationError("ABox requirement view is inconsistent.")
    tbox.assert_unchanged()
    interpretation_root = root / "products/grounding/document_evidence"
    trace_path = interpretation_root / f"interpretation_{operation_number:04d}.json"
    operation_root = interpretation_root / f"operation_{operation_number:04d}"
    if trace_path.exists() or operation_root.exists():
        raise DocumentInterpretationError(
            f"Document interpretation {operation_number:04d} already exists."
        )

    rendered_pages: tuple[RenderedDocumentPage, ...] = ()
    response: DocumentVisionResponse | None = None
    try:
        rendered_pages = _render_pages(
            DOCUMENT_CONTEXT_REF,
            document_pages,
            operation_root,
        )
        request = DocumentVisionRequest(
            product_requirement=abox.product_requirement,
            abox_view=compact_abox_view(abox),
            tbox_classes=tuple(sorted(tbox.classes)),
            tbox_object_properties=tuple(sorted(tbox.object_properties)),
            tbox_datatype_properties=tuple(sorted(tbox.datatype_properties)),
            pages=rendered_pages,
        )
        response = await vision_runtime.interpret_document(request)
        if response.model != config.model:
            raise DocumentInterpretationError(
                "Document vision response model does not match configured model."
            )
        output = _validated_output(response.output, tbox, len(rendered_pages))
        delta = _compile_delta(
            output,
            abox=abox,
            operation_number=operation_number,
        )
    except (DocumentInterpretationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _write_trace_exclusive(
            trace_path,
            _trace_record(
                config=config,
                rendered_pages=rendered_pages,
                response=response,
                output=None if response is None else response.output,
                delta=None,
                failure=f"{type(exc).__name__}: {exc}",
            ),
        )
        if isinstance(exc, DocumentInterpretationError):
            raise
        raise DocumentInterpretationError(
            f"Document interpretation failed: {type(exc).__name__}: {exc}"
        ) from exc

    _write_trace_exclusive(
        trace_path,
        _trace_record(
            config=config,
            rendered_pages=rendered_pages,
            response=response,
            output=output,
            delta=delta,
            failure=None,
        ),
    )
    return DocumentInterpretationResult(
        delta=delta,
        trace_path=trace_path,
        page_count=len(rendered_pages),
        assertion_count=len(delta["assertions"]),  # type: ignore[arg-type]
        uncertainty_count=len(delta["uncertainty"]),  # type: ignore[arg-type]
        unresolved_evidence_need_count=len(  # type: ignore[arg-type]
            delta["unresolved_evidence_needs"]
        ),
        provider=config.provider,
        model=response.model,
        response_id=response.response_id,
    )


def document_interpretation_schema() -> dict[str, object]:
    """Return the strict structured-output schema sent to OpenAI."""
    page_numbers = {
        "type": "array",
        "items": {"type": "integer", "minimum": 1},
        "minItems": 1,
        "uniqueItems": True,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_OUTPUT_KEYS),
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(_ENTITY_KEYS),
                    "properties": {
                        "key": {"type": "string"},
                        "class_iri": {"type": "string"},
                        "evidence_pages": page_numbers,
                    },
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(_RELATION_KEYS),
                    "properties": {
                        "subject_key": {"type": "string"},
                        "predicate_iri": {"type": "string"},
                        "object_key": {"type": "string"},
                        "evidence_pages": page_numbers,
                    },
                },
            },
            "literal_facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(_LITERAL_KEYS),
                    "properties": {
                        "subject_key": {"type": "string"},
                        "predicate_iri": {"type": "string"},
                        "value": {"type": ["string", "number", "integer", "boolean"]},
                        "datatype_iri": {"type": ["string", "null"]},
                        "language": {"type": ["string", "null"]},
                        "evidence_pages": page_numbers,
                    },
                },
            },
            "uncertainty": {
                "type": "array",
                "items": _note_schema(page_numbers),
            },
            "unresolved_evidence_needs": {
                "type": "array",
                "items": _note_schema(page_numbers),
            },
        },
    }


def _note_schema(page_numbers: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_NOTE_KEYS),
        "properties": {
            "description": {"type": "string"},
            "evidence_pages": page_numbers,
        },
    }


def _request_text(request: DocumentVisionRequest) -> str:
    record = {
        "product_requirement": request.product_requirement,
        "current_abox": request.abox_view,
        "allowed_tbox_classes": list(request.tbox_classes),
        "allowed_tbox_object_properties": list(request.tbox_object_properties),
        "allowed_tbox_datatype_properties": list(request.tbox_datatype_properties),
        "document_pages": [
            {"page": page.page_number, "extracted_text": page.text} for page in request.pages
        ],
    }
    return json.dumps(record, ensure_ascii=False, allow_nan=False)


_OPENAI_INSTRUCTIONS = """You are a bounded document interpretation tool. Interpret only the supplied ordered PDF page images and extracted text for the exact product requirement. Use only the supplied TBox class and property IRIs. Use the reserved subject key `specification` only for the initialized requirement. Create short unique entity keys; do not create IRIs. Every output item must cite the visible page numbers supporting it. Put uncertain claims in uncertainty and missing evidence in unresolved_evidence_needs. Never infer robot resources, capabilities, primitive steps, execution state, simulator state, or hidden expected answers. Return only the requested structured object."""


def _validated_served_document(
    served_context: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    if (
        not isinstance(served_context, Mapping)
        or served_context.get("context_ref") != DOCUMENT_CONTEXT_REF
        or served_context.get("evidence_type") != "document"
    ):
        raise DocumentInterpretationError(
            "Phase 4.1 accepts only the exact approved NIST document context."
        )
    document_evidence = served_context.get("document_evidence")
    if not isinstance(document_evidence, Mapping):
        raise DocumentInterpretationError("served_context has no document_evidence.")
    pages = document_evidence.get("pages")
    page_count = document_evidence.get("page_count")
    if (
        not isinstance(pages, list)
        or not pages
        or isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or page_count != len(pages)
    ):
        raise DocumentInterpretationError("served document page records are invalid.")
    validated: list[Mapping[str, object]] = []
    for page_number, page in enumerate(pages, start=1):
        if (
            not isinstance(page, Mapping)
            or set(page) != {"page", "text"}
            or page.get("page") != page_number
            or not isinstance(page.get("text"), str)
        ):
            raise DocumentInterpretationError("served document pages are not ordered exactly.")
        validated.append(page)
    return tuple(validated)


def _render_pages(
    context_ref: str,
    page_records: Sequence[Mapping[str, object]],
    operation_root: Path,
) -> tuple[RenderedDocumentPage, ...]:
    source_path = approved_document_path(context_ref)
    parent = operation_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".document-", dir=parent))
    try:
        pages_root = temporary_root / "pages"
        pages_root.mkdir()
        rendered: list[RenderedDocumentPage] = []
        with pdfplumber.open(source_path) as document:
            if len(document.pages) != len(page_records):
                raise DocumentInterpretationError(
                    "Approved PDF page count changed after exact-ref serving."
                )
            for page_number, (page, page_record) in enumerate(
                zip(document.pages, page_records, strict=True),
                start=1,
            ):
                resolution = max(72, round(1600 * 72 / float(page.width)))
                image_path = pages_root / f"page_{page_number:04d}.png"
                page.to_image(resolution=resolution, antialias=True).save(
                    image_path,
                    format="PNG",
                )
                image_bytes = image_path.read_bytes()
                rendered.append(
                    RenderedDocumentPage(
                        page_number=page_number,
                        image_path=operation_root / "pages" / image_path.name,
                        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
                        image_data_url=(
                            "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
                        ),
                        text=str(page_record["text"]),
                    )
                )
        temporary_root.rename(operation_root)
        return tuple(rendered)
    except (OSError, PDFSyntaxError, TypeError, ValueError):
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def _validated_output(
    value: Mapping[str, object],
    tbox: TBoxSnapshot,
    page_count: int,
) -> dict[str, list[dict[str, object]]]:
    if not isinstance(value, Mapping) or set(value) != _OUTPUT_KEYS:
        raise DocumentInterpretationError("Document output fields are invalid.")
    output: dict[str, list[dict[str, object]]] = {}
    for field in _OUTPUT_KEYS:
        items = value[field]
        if not isinstance(items, list):
            raise DocumentInterpretationError(f"Document output {field} must be a list.")
        output[field] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise DocumentInterpretationError(f"{field}[{index}] must be an object.")
            output[field].append(dict(item))

    entity_keys = _validated_entities(output["entities"], tbox, page_count)
    subject_keys = {"specification", *entity_keys}
    _validate_relations(output["relations"], tbox, page_count, entity_keys)
    _validate_literal_facts(output["literal_facts"], tbox, page_count, subject_keys)
    _validate_notes(output["uncertainty"], "uncertainty", page_count)
    _validate_notes(
        output["unresolved_evidence_needs"],
        "unresolved_evidence_needs",
        page_count,
    )
    return output


def _validated_entities(
    entities: Sequence[Mapping[str, object]],
    tbox: TBoxSnapshot,
    page_count: int,
) -> set[str]:
    entity_keys: set[str] = set()
    for index, entity in enumerate(entities):
        _require_exact_keys(entity, _ENTITY_KEYS, f"entities[{index}]")
        key = _entity_key(entity["key"], f"entities[{index}].key")
        if key == "specification" or key in entity_keys:
            raise DocumentInterpretationError("Entity keys must be unique and non-reserved.")
        entity_keys.add(key)
        if entity["class_iri"] not in tbox.classes:
            raise DocumentInterpretationError("Entity class_iri is not declared in the TBox.")
        _evidence_pages(entity["evidence_pages"], page_count, f"entities[{index}]")
    return entity_keys


def _validate_relations(
    relations: Sequence[Mapping[str, object]],
    tbox: TBoxSnapshot,
    page_count: int,
    entity_keys: set[str],
) -> None:
    subject_keys = {"specification", *entity_keys}
    for index, relation in enumerate(relations):
        _require_exact_keys(relation, _RELATION_KEYS, f"relations[{index}]")
        if relation["subject_key"] not in subject_keys or relation["object_key"] not in entity_keys:
            raise DocumentInterpretationError("Relation entity keys are not declared.")
        if relation["predicate_iri"] not in tbox.object_properties:
            raise DocumentInterpretationError("Relation predicate_iri is not declared in the TBox.")
        _evidence_pages(relation["evidence_pages"], page_count, f"relations[{index}]")


def _validate_literal_facts(
    literal_facts: Sequence[Mapping[str, object]],
    tbox: TBoxSnapshot,
    page_count: int,
    subject_keys: set[str],
) -> None:
    for index, literal in enumerate(literal_facts):
        _require_exact_keys(literal, _LITERAL_KEYS, f"literal_facts[{index}]")
        if literal["subject_key"] not in subject_keys:
            raise DocumentInterpretationError("Literal fact subject key is not declared.")
        if literal["predicate_iri"] not in tbox.datatype_properties:
            raise DocumentInterpretationError("Literal predicate_iri is not declared in the TBox.")
        scalar = literal["value"]
        if isinstance(scalar, (dict, list)) or scalar is None:
            raise DocumentInterpretationError("Literal fact value must be a JSON scalar.")
        datatype = literal["datatype_iri"]
        language = literal["language"]
        if datatype is not None and (not isinstance(datatype, str) or not datatype):
            raise DocumentInterpretationError("Literal datatype_iri is invalid.")
        if language is not None and (not isinstance(language, str) or not language):
            raise DocumentInterpretationError("Literal language is invalid.")
        if datatype is not None and language is not None:
            raise DocumentInterpretationError("Literal cannot have datatype and language together.")
        _evidence_pages(literal["evidence_pages"], page_count, f"literal_facts[{index}]")


def _validate_notes(
    notes: Sequence[Mapping[str, object]],
    field: str,
    page_count: int,
) -> None:
    for index, note in enumerate(notes):
        _require_exact_keys(note, _NOTE_KEYS, f"{field}[{index}]")
        if not isinstance(note["description"], str) or not note["description"].strip():
            raise DocumentInterpretationError(f"{field}[{index}] description is invalid.")
        _evidence_pages(note["evidence_pages"], page_count, f"{field}[{index}]")


def _compile_delta(
    output: Mapping[str, list[dict[str, object]]],
    *,
    abox: ABoxSnapshot,
    operation_number: int,
) -> dict[str, object]:
    entity_iris = {
        str(entity["key"]): f"{abox.namespace}document_{operation_number:04d}_{entity['key']}"
        for entity in output["entities"]
    }
    subject_iris = {"specification": abox.specification_iri, **entity_iris}
    assertions: list[dict[str, object]] = []
    for entity in output["entities"]:
        assertions.append(
            _assertion(
                entity_iris[str(entity["key"])],
                str(RDF.type),
                {"kind": "iri", "value": entity["class_iri"]},
                entity["evidence_pages"],
            )
        )
    for relation in output["relations"]:
        assertions.append(
            _assertion(
                subject_iris[str(relation["subject_key"])],
                str(relation["predicate_iri"]),
                {
                    "kind": "iri",
                    "value": entity_iris[str(relation["object_key"])],
                },
                relation["evidence_pages"],
            )
        )
    for literal in output["literal_facts"]:
        object_record = {
            "kind": "literal",
            "value": literal["value"],
        }
        if literal["datatype_iri"] is not None:
            object_record["datatype"] = literal["datatype_iri"]
        if literal["language"] is not None:
            object_record["language"] = literal["language"]
        assertions.append(
            _assertion(
                subject_iris[str(literal["subject_key"])],
                str(literal["predicate_iri"]),
                object_record,
                literal["evidence_pages"],
            )
        )
    return {
        "assertions": assertions,
        "uncertainty": output["uncertainty"],
        "unresolved_evidence_needs": output["unresolved_evidence_needs"],
        "typed_context_refs": [
            f"products/grounding/document_evidence/interpretation_{operation_number:04d}.json"
        ],
    }


def _assertion(
    subject: str,
    predicate: str,
    object_record: Mapping[str, object],
    pages: object,
) -> dict[str, object]:
    return {
        "subject": subject,
        "predicate": predicate,
        "object": dict(object_record),
        "evidence_refs": [
            f"{DOCUMENT_CONTEXT_REF}#page={page}"
            for page in pages  # type: ignore[union-attr]
        ],
    }


def _trace_record(
    *,
    config: DocumentVLMConfig,
    rendered_pages: Sequence[RenderedDocumentPage],
    response: DocumentVisionResponse | None,
    output: Mapping[str, object] | None,
    delta: Mapping[str, object] | None,
    failure: str | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "producer": _PRODUCER,
        "provider": config.provider,
        "configured_model": config.model,
        "response_id": None if response is None else response.response_id,
        "response_model": None if response is None else response.model,
        "store": False,
        "document_context_ref": DOCUMENT_CONTEXT_REF,
        "pages": [
            {
                "page": page.page_number,
                "image_ref": str(page.image_path),
                "image_sha256": page.image_sha256,
            }
            for page in rendered_pages
        ],
        "structured_output": output,
        "compiled_delta": delta,
        "failure": failure,
    }


def _write_trace_exclusive(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise DocumentInterpretationError(
            f"Document interpretation trace already exists: {path.name}"
        ) from exc


def _validate_operation_number(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DocumentInterpretationError("operation_number must be a positive integer.")


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    field_name: str,
) -> None:
    if set(value) != expected:
        raise DocumentInterpretationError(f"{field_name} fields are invalid.")


def _entity_key(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _ENTITY_KEY.fullmatch(value):
        raise DocumentInterpretationError(f"{field_name} is invalid.")
    return value


def _evidence_pages(value: object, page_count: int, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise DocumentInterpretationError(f"{field_name} requires evidence_pages.")
    pages: list[int] = []
    for page in value:
        if isinstance(page, bool) or not isinstance(page, int) or page < 1 or page > page_count:
            raise DocumentInterpretationError(f"{field_name} evidence page is invalid.")
        pages.append(page)
    if len(pages) != len(set(pages)):
        raise DocumentInterpretationError(f"{field_name} evidence pages must be unique.")
    return tuple(pages)

"""Interpret approved PDF evidence into ontology-neutral overview records."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pdfplumber
from openai import AsyncOpenAI, OpenAIError
from pdfminer.pdfparser import PDFSyntaxError

from cais_spade_llm.spec2primitives.agents.pa.product_context import ABoxSnapshot
from cais_spade_llm.spec2primitives.config import DocumentVLMConfig
from cais_spade_llm.spec2primitives.ontology import TBoxSnapshot
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_document_metadata,
    approved_document_path,
)

DOCUMENT_OVERVIEW_SCHEMA_VERSION = 2
_PRODUCER = "document_evidence"
_OUTPUT_NAME = "spec2primitives_document_overview"
_OUTPUT_KEYS = {"summary", "observations", "uncertainty"}
_OBSERVATION_KEYS = {"description", "evidence_pages"}
_CACHE_RECORD_KEYS = {
    "schema_version",
    "record_type",
    "overview_schema_version",
    "context_ref",
    "source_sha256",
    "cache_fingerprint",
    "provider",
    "configured_model",
    "response_model",
    "response_id",
    "store",
    "page_count",
    "pages",
    "summary",
    "observations",
    "uncertainty",
}


class DocumentInterpretationError(ValueError):
    """Raised when approved document evidence cannot be interpreted safely."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: Mapping[str, object] | None = None,
    ) -> None:
        """Create an error with optional sanitized provider diagnostics."""
        super().__init__(message)
        self.diagnostic = None if diagnostic is None else dict(diagnostic)


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
    """Contain only source evidence visible to the overview VLM."""

    context_ref: str
    source_sha256: str
    pages: tuple[RenderedDocumentPage, ...]


@dataclass(frozen=True)
class DocumentVisionResponse:
    """Return one structured overview and provider audit fields."""

    response_id: str
    model: str
    output: Mapping[str, object]


class DocumentVisionRuntime(Protocol):
    """Narrow injected boundary for one ontology-neutral document request."""

    async def interpret_document(
        self,
        request: DocumentVisionRequest,
    ) -> DocumentVisionResponse:
        """Interpret the ordered pages without product or ontology context."""
        ...

@dataclass(frozen=True)
class DocumentOverviewRecord:
    """Reference one validated content-addressed document overview."""

    context_ref: str
    source_sha256: str
    cache_fingerprint: str
    cache_status: str
    record_path: Path
    record: Mapping[str, object]


@dataclass(frozen=True)
class DocumentInterpretationResult:
    """Return the assertion-free delta and overview metadata used by callers."""

    delta: Mapping[str, object]
    overview_record_path: Path
    cache_status: str
    page_count: int


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
        """Submit one ordered multimodal overview request with strict JSON."""
        content: list[dict[str, object]] = [
            {"type": "input_text", "text": _request_text(request)}
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
            diagnostic = _openai_error_diagnostic(exc)
            raise DocumentInterpretationError(
                "OpenAI document overview failed: "
                f"{_diagnostic_summary(diagnostic)}",
                diagnostic=diagnostic,
            ) from exc

        return _document_vision_response(response, "document overview")

async def prepare_document_overview(
    *,
    served_context: Mapping[str, object],
    cache_root: Path,
    config: DocumentVLMConfig,
    vision_runtime: DocumentVisionRuntime,
) -> DocumentOverviewRecord:
    """Load or create one content-addressed ontology-neutral overview.

    The cache key includes the source bytes and interpretation contract. A
    valid cache hit performs no model call.
    """
    context_ref, source_sha256, document_pages = _validated_served_document(
        served_context
    )
    cache_fingerprint = _cache_fingerprint(source_sha256, config)
    source_root = Path(cache_root).resolve() / "document" / source_sha256
    record_root = source_root / cache_fingerprint
    record_path = record_root / "overview.json"
    if record_path.is_file():
        record = _load_cache_record(
            record_path,
            context_ref=context_ref,
            source_sha256=source_sha256,
            cache_fingerprint=cache_fingerprint,
        )
        return DocumentOverviewRecord(
            context_ref=context_ref,
            source_sha256=source_sha256,
            cache_fingerprint=cache_fingerprint,
            cache_status="hit",
            record_path=record_path,
            record=record,
        )

    source_root.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".overview-", dir=source_root))
    try:
        rendered_pages = _render_pages(
            context_ref,
            document_pages,
            temporary_root / "rendered",
        )
        request = DocumentVisionRequest(
            context_ref=context_ref,
            source_sha256=source_sha256,
            pages=rendered_pages,
        )
        response = await vision_runtime.interpret_document(request)
        if response.model != config.model:
            raise DocumentInterpretationError(
                "Document vision response model does not match configured model."
            )
        output = _validated_overview_output(
            response.output,
            page_count=len(rendered_pages),
        )
        record = _overview_cache_record(
            context_ref=context_ref,
            source_sha256=source_sha256,
            cache_fingerprint=cache_fingerprint,
            config=config,
            response=response,
            rendered_pages=rendered_pages,
            output=output,
        )
        _write_json_exclusive(temporary_root / "overview.json", record)
        try:
            temporary_root.rename(record_root)
        except FileExistsError:
            shutil.rmtree(temporary_root, ignore_errors=True)
            record = _load_cache_record(
                record_path,
                context_ref=context_ref,
                source_sha256=source_sha256,
                cache_fingerprint=cache_fingerprint,
            )
            return DocumentOverviewRecord(
                context_ref=context_ref,
                source_sha256=source_sha256,
                cache_fingerprint=cache_fingerprint,
                cache_status="hit",
                record_path=record_path,
                record=record,
            )
    except (DocumentInterpretationError, OSError, RuntimeError, TypeError, ValueError):
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    return DocumentOverviewRecord(
        context_ref=context_ref,
        source_sha256=source_sha256,
        cache_fingerprint=cache_fingerprint,
        cache_status="miss",
        record_path=record_path,
        record=record,
    )


def document_overview_cache_status(
    context_ref: str,
    *,
    cache_root: Path,
    config: DocumentVLMConfig,
) -> dict[str, object]:
    """Inspect one approved document and overview cache without inference."""
    try:
        metadata = approved_document_metadata(context_ref)
        validated_ref = str(metadata["context_ref"])
        source_sha256 = str(metadata["source_sha256"])
        fingerprint = _cache_fingerprint(source_sha256, config)
        record_path = (
            Path(cache_root).resolve()
            / "document"
            / source_sha256
            / fingerprint
            / "overview.json"
        )
        if record_path.is_file():
            _load_cache_record(
                record_path,
                context_ref=validated_ref,
                source_sha256=source_sha256,
                cache_fingerprint=fingerprint,
            )
            overview_status = "prepared"
        else:
            overview_status = (
                "stale"
                if _has_other_overview_record(
                    Path(cache_root).resolve(),
                    context_ref=validated_ref,
                )
                else "missing"
            )
    except (DocumentInterpretationError, OSError, TypeError, ValueError) as exc:
        return {
            "context_ref": context_ref,
            "source_status": "invalid",
            "overview_status": "unavailable",
            "record_path": None,
            "rejection": f"{type(exc).__name__}: {exc}",
        }
    return {
        "context_ref": validated_ref,
        "source_status": "valid",
        "source_sha256": source_sha256,
        "overview_status": overview_status,
        "record_path": str(record_path) if record_path.is_file() else None,
        "rejection": None,
    }


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
    """Snapshot a neutral overview and return an assertion-free delta."""
    _validate_operation_number(operation_number)
    root = Path(interaction_root).resolve()
    if abox.interaction_root != root or abox.tbox_fingerprint != tbox.fingerprint:
        raise DocumentInterpretationError(
            "Document interpretation ABox does not match its interaction and TBox."
        )
    tbox.assert_unchanged()
    interpretation_root = root / "products/grounding/document_evidence"
    trace_path = interpretation_root / f"interpretation_{operation_number:04d}.json"
    overview_path = interpretation_root / f"overview_{operation_number:04d}.json"
    if trace_path.exists() or overview_path.exists():
        raise DocumentInterpretationError(
            f"Document interpretation {operation_number:04d} already exists."
        )

    overview: DocumentOverviewRecord | None = None
    try:
        overview = await prepare_document_overview(
            served_context=served_context,
            cache_root=root.parent / "source_cache",
            config=config,
            vision_runtime=vision_runtime,
        )
        evidence_refs = [overview.context_ref]
        evidence_refs.extend(
            f"{overview.context_ref}#page={page_number}"
            for page_number in range(1, int(overview.record["page_count"]) + 1)
        )
        snapshot = {
            "schema_version": 2,
            "record_type": "DocumentOverviewRecord",
            "producer": _PRODUCER,
            "operation_number": operation_number,
            "cache_status": overview.cache_status,
            "cache_record_ref": str(overview.record_path),
            "evidence_refs": evidence_refs,
            "overview": dict(overview.record),
        }
        _write_json_exclusive(overview_path, snapshot)
        uncertainty = [
            {
                "description": str(item["description"]),
                "evidence_refs": list(item["evidence_refs"]),
            }
            for item in _record_items(overview.record, "uncertainty")
        ]
        delta: dict[str, object] = {
            "assertions": [],
            "uncertainty": uncertainty,
            "unresolved_evidence_needs": [],
            "typed_context_refs": [str(overview_path.relative_to(root))],
        }
    except (DocumentInterpretationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        diagnostic = (
            exc.diagnostic if isinstance(exc, DocumentInterpretationError) else None
        )
        _write_trace_exclusive(
            trace_path,
            _trace_record(
                config=config,
                overview=overview,
                delta=None,
                failure=f"{type(exc).__name__}: {exc}",
                diagnostic=diagnostic,
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
            overview=overview,
            delta=delta,
            failure=None,
            diagnostic=None,
        ),
    )
    page_count = int(overview.record["page_count"])
    return DocumentInterpretationResult(
        delta=delta,
        overview_record_path=overview_path,
        cache_status=overview.cache_status,
        page_count=page_count,
    )


def document_interpretation_schema() -> dict[str, object]:
    """Return the strict ontology-neutral overview schema sent to OpenAI."""
    page_numbers = {
        "type": "array",
        "items": {"type": "integer", "minimum": 1},
        "minItems": 1,
    }
    observation = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_OBSERVATION_KEYS),
        "properties": {
            "description": {"type": "string"},
            "evidence_pages": page_numbers,
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_OUTPUT_KEYS),
        "properties": {
            "summary": {"type": "string"},
            "observations": {"type": "array", "items": observation},
            "uncertainty": {"type": "array", "items": observation},
        },
    }


def _request_text(request: DocumentVisionRequest) -> str:
    return json.dumps(
        {
            "document_context_ref": request.context_ref,
            "source_sha256": request.source_sha256,
            "document_pages": [
                {"page": page.page_number, "extracted_text": page.text}
                for page in request.pages
            ],
        },
        ensure_ascii=False,
        allow_nan=False,
    )


_OPENAI_INSTRUCTIONS = """You are a bounded document overview tool. Describe only the supplied ordered PDF page images and extracted text. Produce a generic source overview independent of any user requirement or ontology. Record short surface-form observations with the visible page numbers that support them. Put ambiguous or unclear content in uncertainty. Do not create entity keys, IRIs, ontology classes, ontology properties, RDF assertions, robot resources, capabilities, primitive steps, execution state, simulator state, or hidden expected answers. Return only the requested structured object."""


def _document_vision_response(response: object, label: str) -> DocumentVisionResponse:
    output_text = getattr(response, "output_text", None)
    response_id = getattr(response, "id", None)
    model = getattr(response, "model", None)
    if not all(
        isinstance(value, str) and value
        for value in (output_text, response_id, model)
    ):
        raise DocumentInterpretationError(
            f"OpenAI returned an incomplete {label} response."
        )
    try:
        output = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise DocumentInterpretationError(
            f"OpenAI {label} output was not JSON."
        ) from exc
    if not isinstance(output, Mapping):
        raise DocumentInterpretationError(
            f"OpenAI {label} output was not an object."
        )
    return DocumentVisionResponse(
        response_id=response_id,
        model=model,
        output=output,
    )


def _validated_served_document(
    served_context: Mapping[str, object],
) -> tuple[str, str, tuple[Mapping[str, object], ...]]:
    if (
        not isinstance(served_context, Mapping)
        or served_context.get("evidence_type") != "document"
    ):
        raise DocumentInterpretationError(
            "Phase 4.1 accepts only one approved document context."
        )
    context_ref = served_context.get("context_ref")
    if not isinstance(context_ref, str) or not context_ref:
        raise DocumentInterpretationError("served document context_ref is invalid.")
    try:
        source_path = approved_document_path(context_ref)
    except (OSError, ValueError) as exc:
        raise DocumentInterpretationError(
            "served document context_ref is not approved."
        ) from exc
    document_evidence = served_context.get("document_evidence")
    if not isinstance(document_evidence, Mapping):
        raise DocumentInterpretationError("served_context has no document_evidence.")
    if set(document_evidence) != {"source_sha256", "page_count", "pages"}:
        raise DocumentInterpretationError("served document evidence fields are invalid.")
    source_sha256 = document_evidence.get("source_sha256")
    actual_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha256 != actual_sha256:
        raise DocumentInterpretationError("served document source hash is invalid.")
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
            raise DocumentInterpretationError(
                "served document pages are not ordered exactly."
            )
        validated.append(page)
    return context_ref, actual_sha256, tuple(validated)


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
                            "data:image/png;base64,"
                            + base64.b64encode(image_bytes).decode("ascii")
                        ),
                        text=str(page_record["text"]),
                    )
                )
        temporary_root.rename(operation_root)
        return tuple(rendered)
    except (OSError, PDFSyntaxError, TypeError, ValueError):
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def _validated_overview_output(
    value: Mapping[str, object],
    *,
    page_count: int,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _OUTPUT_KEYS:
        raise DocumentInterpretationError("Document overview output fields are invalid.")
    summary = value["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise DocumentInterpretationError("Document overview summary is invalid.")
    output: dict[str, object] = {
        "summary": summary,
        "observations": [],
        "uncertainty": [],
    }
    for field in ("observations", "uncertainty"):
        items = value[field]
        if not isinstance(items, list):
            raise DocumentInterpretationError(f"Document overview {field} must be a list.")
        validated_items: list[dict[str, object]] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping) or set(item) != _OBSERVATION_KEYS:
                raise DocumentInterpretationError(f"{field}[{index}] fields are invalid.")
            description = item["description"]
            if not isinstance(description, str) or not description.strip():
                raise DocumentInterpretationError(
                    f"{field}[{index}] description is invalid."
                )
            pages = _evidence_pages(
                item["evidence_pages"],
                page_count,
                f"{field}[{index}]",
            )
            validated_items.append(
                {"description": description, "evidence_pages": list(pages)}
            )
        output[field] = validated_items
    if not output["observations"]:
        raise DocumentInterpretationError(
            "Document overview requires at least one cited observation."
        )
    return output


def _overview_cache_record(
    *,
    context_ref: str,
    source_sha256: str,
    cache_fingerprint: str,
    config: DocumentVLMConfig,
    response: DocumentVisionResponse,
    rendered_pages: Sequence[RenderedDocumentPage],
    output: Mapping[str, object],
) -> dict[str, object]:
    def _items(field: str, prefix: str) -> list[dict[str, object]]:
        values = output[field]
        if not isinstance(values, list):
            raise DocumentInterpretationError(f"Document overview {field} is invalid.")
        return [
            {
                f"{prefix}_id": f"{prefix}_{index:04d}",
                "description": str(item["description"]),
                "evidence_refs": [
                    f"{context_ref}#page={page}" for page in item["evidence_pages"]
                ],
            }
            for index, item in enumerate(values, start=1)
        ]

    return {
        "schema_version": 2,
        "record_type": "DocumentOverviewRecord",
        "overview_schema_version": DOCUMENT_OVERVIEW_SCHEMA_VERSION,
        "context_ref": context_ref,
        "source_sha256": source_sha256,
        "cache_fingerprint": cache_fingerprint,
        "provider": config.provider,
        "configured_model": config.model,
        "response_model": response.model,
        "response_id": response.response_id,
        "store": False,
        "page_count": len(rendered_pages),
        "pages": [
            {
                "page": page.page_number,
                "evidence_ref": f"{context_ref}#page={page.page_number}",
                "extracted_text": page.text,
                "image_ref": f"rendered/pages/page_{page.page_number:04d}.png",
                "image_sha256": page.image_sha256,
            }
            for page in rendered_pages
        ],
        "summary": output["summary"],
        "observations": _items("observations", "observation"),
        "uncertainty": _items("uncertainty", "uncertainty"),
    }


def _cache_fingerprint(source_sha256: str, config: DocumentVLMConfig) -> str:
    value = {
        "source_sha256": source_sha256,
        "overview_schema_version": DOCUMENT_OVERVIEW_SCHEMA_VERSION,
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "image_detail": config.image_detail,
        "max_output_tokens": config.max_output_tokens,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_cache_record(
    path: Path,
    *,
    context_ref: str,
    source_sha256: str,
    cache_fingerprint: str,
) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DocumentInterpretationError("Document overview cache is unreadable.") from exc
    if not isinstance(value, dict) or set(value) != _CACHE_RECORD_KEYS:
        raise DocumentInterpretationError("Document overview cache fields are invalid.")
    if (
        value["schema_version"] != 2
        or value["record_type"] != "DocumentOverviewRecord"
        or value["overview_schema_version"] != DOCUMENT_OVERVIEW_SCHEMA_VERSION
        or value["context_ref"] != context_ref
        or value["source_sha256"] != source_sha256
        or value["cache_fingerprint"] != cache_fingerprint
        or value["store"] is not False
    ):
        raise DocumentInterpretationError("Document overview cache identity is invalid.")
    page_count = value["page_count"]
    pages = value["pages"]
    if (
        isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or page_count <= 0
        or not isinstance(pages, list)
        or len(pages) != page_count
    ):
        raise DocumentInterpretationError("Document overview cached pages are invalid.")
    for field in (
        "summary",
        "provider",
        "configured_model",
        "response_model",
        "response_id",
    ):
        if not isinstance(value[field], str) or not value[field]:
            raise DocumentInterpretationError(
                f"Document overview cache {field} is invalid."
            )
    for field in ("observations", "uncertainty"):
        if not isinstance(value[field], list):
            raise DocumentInterpretationError(
                f"Document overview cache {field} is invalid."
            )
    for page_number, page in enumerate(pages, start=1):
        if (
            not isinstance(page, Mapping)
            or set(page)
            != {
                "page",
                "evidence_ref",
                "extracted_text",
                "image_ref",
                "image_sha256",
            }
            or page.get("page") != page_number
            or page.get("evidence_ref") != f"{context_ref}#page={page_number}"
            or not isinstance(page.get("extracted_text"), str)
            or not isinstance(page.get("image_ref"), str)
            or not isinstance(page.get("image_sha256"), str)
        ):
            raise DocumentInterpretationError(
                "Document overview cached page content is invalid."
            )
    return value


def _has_other_overview_record(cache_root: Path, *, context_ref: str) -> bool:
    for path in (cache_root / "document").glob("*/*/overview.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping) and value.get("context_ref") == context_ref:
            return True
    return False


def _record_items(
    record: Mapping[str, object],
    field: str,
) -> tuple[Mapping[str, object], ...]:
    items = record.get(field)
    if not isinstance(items, list) or not all(isinstance(item, Mapping) for item in items):
        raise DocumentInterpretationError(f"Document overview {field} is invalid.")
    return tuple(items)  # type: ignore[arg-type]


def _trace_record(
    *,
    config: DocumentVLMConfig,
    overview: DocumentOverviewRecord | None,
    delta: Mapping[str, object] | None,
    failure: str | None,
    diagnostic: Mapping[str, object] | None,
) -> dict[str, object]:
    record = None if overview is None else overview.record
    return {
        "schema_version": 2,
        "producer": _PRODUCER,
        "provider": config.provider,
        "configured_model": config.model,
        "response_id": None if record is None else record["response_id"],
        "response_model": None if record is None else record["response_model"],
        "store": False,
        "document_context_ref": None if record is None else record["context_ref"],
        "source_sha256": None if record is None else record["source_sha256"],
        "cache_status": None if overview is None else overview.cache_status,
        "cache_record_ref": None if overview is None else str(overview.record_path),
        "pages": [] if record is None else record["pages"],
        "structured_output": None
        if record is None
        else {
            "summary": record["summary"],
            "observations": record["observations"],
            "uncertainty": record["uncertainty"],
        },
        "compiled_delta": None if delta is None else dict(delta),
        "failure": failure,
        "diagnostic": None if diagnostic is None else dict(diagnostic),
    }


def _openai_error_diagnostic(error: OpenAIError) -> dict[str, object]:
    """Extract bounded OpenAI error fields without persisting request data."""
    body = getattr(error, "body", None)
    if isinstance(body, Mapping) and isinstance(body.get("error"), Mapping):
        body = body["error"]

    def _metadata(attribute: str, body_field: str | None = None) -> object:
        value = getattr(error, attribute, None)
        if value is None and isinstance(body, Mapping):
            value = body.get(body_field or attribute)
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            return None
        return value

    body_message = body.get("message") if isinstance(body, Mapping) else None
    error_message = getattr(error, "message", None)
    message = body_message if isinstance(body_message, str) else error_message
    if not isinstance(message, str) or not message:
        message = "OpenAI request failed without a message."
    return {
        "stage": "Phase 4.1 document_evidence OpenAI Responses call",
        "exception": type(error).__name__,
        "status_code": _metadata("status_code"),
        "request_id": _metadata("request_id"),
        "error_type": _metadata("type"),
        "param": _metadata("param"),
        "code": _metadata("code"),
        "message": message[:2000],
    }


def _diagnostic_summary(diagnostic: Mapping[str, object]) -> str:
    """Format only the approved diagnostic fields for operator visibility."""
    return "; ".join(
        f"{field}={diagnostic[field]}"
        for field in (
            "stage",
            "exception",
            "status_code",
            "request_id",
            "error_type",
            "param",
            "code",
            "message",
        )
        if diagnostic.get(field) is not None
    )


def _write_json_exclusive(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _write_trace_exclusive(path: Path, record: Mapping[str, object]) -> None:
    try:
        _write_json_exclusive(path, record)
    except FileExistsError as exc:
        raise DocumentInterpretationError(
            f"Document interpretation trace already exists: {path.name}"
        ) from exc


def _validate_operation_number(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DocumentInterpretationError("operation_number must be a positive integer.")


def _evidence_pages(value: object, page_count: int, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise DocumentInterpretationError(f"{field_name} requires evidence_pages.")
    pages: list[int] = []
    for page in value:
        if (
            isinstance(page, bool)
            or not isinstance(page, int)
            or page < 1
            or page > page_count
        ):
            raise DocumentInterpretationError(
                f"{field_name} evidence page is invalid."
            )
        pages.append(page)
    if len(pages) != len(set(pages)):
        raise DocumentInterpretationError(
            f"{field_name} evidence pages must be unique."
        )
    return tuple(pages)

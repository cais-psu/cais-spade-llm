"""Author and persist one structural RobotAgent primitive-program draft."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingContractError,
    PAContextGroundingCompletionV3,
    load_pa_context_grounding_completion,
)
from cais_spade_llm.spec2primitives.agents.ra.context_handoff import (
    RAContextHandoffError,
    SelectedRAAssignmentEnvelope,
    SelectedRAContextSnapshot,
    load_selected_ra_context_snapshot,
    read_phase_5_1_diagnostic,
)

_DRAFT_DIRECTORY = Path("composition/primitive_program_drafts")
_DRAFT_PREFIX = "draft_"
_DRAFT_SUFFIX = ".json"
_MODEL_OUTPUT_KEYS = {"draft_status", "primitive_symbols", "unsupported_reason"}
_RECORD_KEYS = {
    "schema_version",
    "record_type",
    "draft_status",
    "pa_completion_ref",
    "pa_completion_sha256",
    "pa_completion_fingerprint",
    "assignment_ref",
    "assignment_sha256",
    "assignment_fingerprint",
    "robot_state_ref",
    "robot_state_sha256",
    "robot_state_fingerprint",
    "primitive_catalog_ref",
    "primitive_catalog_sha256",
    "primitive_catalog_fingerprint",
    "structural_steps",
    "unsupported_reason",
    "authored_at_ns",
    "fingerprint",
}


class PrimitiveDraftError(RuntimeError):
    """Raised when a structural primitive draft cannot be safely authored."""


class RobotAgentDraftRuntime(Protocol):
    """Expose the selected RobotAgent's structured composition call."""

    async def author_structural_draft(
        self,
        assignment: SelectedRAAssignmentEnvelope,
        *,
        prompt: str,
        response_format: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Return only structural primitive selection and ordering."""
        ...


@dataclass(frozen=True)
class PrimitiveProgramDraft:
    """Hold one validated append-only structural draft record."""

    path: Path
    record: Mapping[str, object]

    def to_record(self) -> dict[str, object]:
        """Return a JSON-safe copy of the persisted record."""
        return deepcopy(dict(self.record))


@dataclass(frozen=True)
class Phase52Diagnostic:
    """Describe the persisted structural-draft state for the active context."""

    status: str
    message: str
    draft_count: int = 0
    latest_draft_ref: str | None = None
    primitive_symbols: tuple[str, ...] = ()
    unsupported_reason: str | None = None
    draft: Mapping[str, object] | None = None
    failure: str | None = None

    def to_view(self) -> dict[str, object]:
        """Return the JSON-safe UI diagnostic."""
        return {
            "status": self.status,
            "message": self.message,
            "draft_count": self.draft_count,
            "latest_draft_ref": self.latest_draft_ref,
            "primitive_symbols": list(self.primitive_symbols),
            "unsupported_reason": self.unsupported_reason,
            "draft": deepcopy(dict(self.draft)) if self.draft is not None else None,
            "failure": self.failure,
        }


async def author_primitive_program_draft(
    runtime: RobotAgentDraftRuntime,
    interaction_root: Path,
) -> PrimitiveProgramDraft:
    """Ask the selected RA for one unbound structural primitive sequence."""
    root = Path(interaction_root).resolve()
    try:
        context = load_selected_ra_context_snapshot(root)
        completion, completion_path = _load_completion(root)
        existing = _load_draft_history(root)
    except (GroundingContractError, RAContextHandoffError, OSError) as exc:
        raise PrimitiveDraftError(str(exc)) from exc

    catalog_ref = context.primitive_catalog_path.relative_to(root).as_posix()
    if any(
        draft.record["primitive_catalog_ref"] == catalog_ref
        for draft in existing
    ):
        raise PrimitiveDraftError(
            "The latest Phase 5.1 context already has a PrimitiveProgramDraft."
        )

    composition_input = _composition_input(root, context, completion)
    catalog_symbols = tuple(
        str(entry["primitive_symbol"])
        for entry in context.primitive_catalog.primitive_catalog
    )
    try:
        response = await runtime.author_structural_draft(
            context.assignment,
            prompt=_composition_prompt(composition_input),
            response_format=_draft_response_format(catalog_symbols),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PrimitiveDraftError(
            f"Selected RobotAgent could not author a structural draft: {exc}"
        ) from exc
    model_output = _validate_model_output(response, catalog_symbols)

    draft_number = len(existing) + 1
    path = _DRAFT_DIRECTORY / f"{_DRAFT_PREFIX}{draft_number:04d}{_DRAFT_SUFFIX}"
    absolute_path = root / path
    state_ref = context.robot_state_path.relative_to(root).as_posix()
    assignment_ref = context.assignment_path.relative_to(root).as_posix()
    completion_ref = completion_path.relative_to(root).as_posix()
    structural_steps = [
        {"step_index": index, "primitive_symbol": symbol}
        for index, symbol in enumerate(model_output["primitive_symbols"], start=1)
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "PrimitiveProgramDraft",
        "draft_status": model_output["draft_status"],
        "pa_completion_ref": completion_ref,
        "pa_completion_sha256": _sha256_path(completion_path),
        "pa_completion_fingerprint": completion.to_record()["fingerprint"],
        "assignment_ref": assignment_ref,
        "assignment_sha256": _sha256_path(context.assignment_path),
        "assignment_fingerprint": context.assignment.fingerprint,
        "robot_state_ref": state_ref,
        "robot_state_sha256": _sha256_path(context.robot_state_path),
        "robot_state_fingerprint": context.robot_state.fingerprint,
        "primitive_catalog_ref": catalog_ref,
        "primitive_catalog_sha256": _sha256_path(context.primitive_catalog_path),
        "primitive_catalog_fingerprint": context.primitive_catalog.fingerprint,
        "structural_steps": structural_steps,
        "unsupported_reason": model_output["unsupported_reason"],
        "authored_at_ns": time.time_ns(),
    }
    payload["fingerprint"] = _fingerprint(payload)
    _write_json_exclusive(absolute_path, payload)
    return _load_draft(root, absolute_path)


def read_phase_5_2_diagnostic(interaction_root: Path) -> Phase52Diagnostic:
    """Read the structural-draft state without contacting the RobotAgent."""
    root = Path(interaction_root).resolve()
    phase_5_1 = read_phase_5_1_diagnostic(root)
    if phase_5_1.status != "context_captured":
        if phase_5_1.status == "blocked":
            return Phase52Diagnostic(
                status="blocked",
                message="Phase 5.1 evidence is blocked.",
                failure=phase_5_1.failure,
            )
        return Phase52Diagnostic(
            status="waiting_for_context",
            message="Capture one valid Phase 5.1 RobotAgent context first.",
        )
    try:
        context = load_selected_ra_context_snapshot(root)
        drafts = _load_draft_history(root)
    except (RAContextHandoffError, PrimitiveDraftError, OSError) as exc:
        return Phase52Diagnostic(
            status="blocked",
            message="PrimitiveProgramDraft evidence failed validation.",
            failure=str(exc),
        )
    latest_catalog_ref = context.primitive_catalog_path.relative_to(root).as_posix()
    matching = [
        draft
        for draft in drafts
        if draft.record["primitive_catalog_ref"] == latest_catalog_ref
    ]
    if not matching:
        return Phase52Diagnostic(
            status="ready_for_draft",
            message="The latest state/catalog pair is ready for RA structural composition.",
            draft_count=len(drafts),
        )
    draft = matching[0]
    record = draft.to_record()
    symbols = tuple(
        str(step["primitive_symbol"])
        for step in record["structural_steps"]
        if isinstance(step, Mapping)
    )
    draft_ref = draft.path.relative_to(root).as_posix()
    if record["draft_status"] == "unsupported":
        return Phase52Diagnostic(
            status="unsupported",
            message="The selected RA found no structural program in its current catalog.",
            draft_count=len(drafts),
            latest_draft_ref=draft_ref,
            unsupported_reason=str(record["unsupported_reason"]),
            draft=record,
        )
    return Phase52Diagnostic(
        status="draft_authored",
        message="The selected RA authored an unbound structural primitive sequence.",
        draft_count=len(drafts),
        latest_draft_ref=draft_ref,
        primitive_symbols=symbols,
        draft=record,
    )


def _composition_input(
    root: Path,
    context: SelectedRAContextSnapshot,
    completion: PAContextGroundingCompletionV3,
) -> dict[str, object]:
    assignment = context.assignment
    completion_record = completion.to_record()
    contract_path = _resolve_ref(root, str(completion_record["typed_grounding_contract_ref"]))
    contract = _read_json_mapping(contract_path, "TypedGroundingContract")
    typed_records = []
    for pinned in completion_record["typed_context_refs"]:
        if not isinstance(pinned, Mapping):
            raise PrimitiveDraftError("PA completion typed_context_refs are invalid.")
        record_ref = str(pinned.get("ref") or "")
        record = _read_json_mapping(_resolve_ref(root, record_ref), "typed context record")
        typed_records.append(
            {"record_type": str(record.get("record_type") or ""), "record_ref": record_ref}
        )
    return {
        "task": {
            "product_requirement": assignment.product_requirement,
            "specification_iri": assignment.specification_iri,
            "feature_iri": assignment.feature_iri,
            "process_iri": assignment.process_iri,
        },
        "selected_resource": {
            "resource_iri": assignment.selected_resource_iri,
            "resource_jid": assignment.selected_resource_jid,
            "execution_mode": assignment.selected_execution_mode,
        },
        "robot_state": deepcopy(dict(context.robot_state.robot_state)),
        "primitive_catalog": [
            deepcopy(dict(entry))
            for entry in context.primitive_catalog.primitive_catalog
        ],
        "grounded_context": {
            "context_summary": str(contract.get("context_summary") or ""),
            "known_context_limits": list(contract.get("missing_information") or []),
            "typed_records": typed_records,
        },
    }


def _composition_prompt(composition_input: Mapping[str, object]) -> str:
    return (
        "You are the exact selected RobotAgent. Author only the structural selection "
        "and order of primitive symbols needed for the requested outcome. Use symbols "
        "exactly as supplied in primitive_catalog. Repetition is allowed when "
        "semantically necessary. Do not bind parameters, invent a primitive, execute "
        "anything, or claim feasibility. Missing product or scene values are deferred "
        "to binding preflight and do not by themselves make the catalog unsupported. "
        "Return unsupported only when no ordered use of the supplied primitives can "
        "express the requested outcome.\n\nCOMPOSITION_INPUT\n"
        + json.dumps(composition_input, sort_keys=True, ensure_ascii=False)
    )


def _draft_response_format(catalog_symbols: tuple[str, ...]) -> dict[str, object]:
    if not catalog_symbols:
        raise PrimitiveDraftError("Primitive catalog must be non-empty.")
    return {
        "name": "spec2primitives_primitive_program_draft",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(_MODEL_OUTPUT_KEYS),
            "properties": {
                "draft_status": {
                    "type": "string",
                    "enum": ["proposed", "unsupported"],
                },
                "primitive_symbols": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(catalog_symbols),
                    },
                    "maxItems": 32,
                },
                "unsupported_reason": {
                    "type": ["string", "null"],
                    "minLength": 1,
                },
            },
        },
    }


def _validate_model_output(
    value: Mapping[str, object],
    catalog_symbols: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _MODEL_OUTPUT_KEYS:
        raise PrimitiveDraftError("RobotAgent structural draft fields are invalid.")
    status = value.get("draft_status")
    symbols_value = value.get("primitive_symbols")
    if not isinstance(symbols_value, list):
        raise PrimitiveDraftError("RobotAgent primitive_symbols must be an array.")
    symbols = []
    for symbol in symbols_value:
        if not isinstance(symbol, str) or symbol not in catalog_symbols:
            raise PrimitiveDraftError(
                "RobotAgent structural draft contains an unknown primitive symbol."
            )
        symbols.append(symbol)
    reason = value.get("unsupported_reason")
    if status == "proposed":
        if not symbols or len(symbols) > 32 or reason is not None:
            raise PrimitiveDraftError("Proposed structural draft is invalid.")
    elif status == "unsupported":
        if symbols or not isinstance(reason, str) or not reason.strip():
            raise PrimitiveDraftError("Unsupported structural draft is invalid.")
    else:
        raise PrimitiveDraftError("RobotAgent draft_status is invalid.")
    return {
        "draft_status": status,
        "primitive_symbols": symbols,
        "unsupported_reason": reason,
    }


def _load_completion(
    root: Path,
) -> tuple[PAContextGroundingCompletionV3, Path]:
    completion = load_pa_context_grounding_completion(root)
    if not isinstance(completion, PAContextGroundingCompletionV3):
        raise PrimitiveDraftError("Phase 5.2 requires PA completion version 3.")
    paths = sorted((root / "interaction_record").glob("context_completion_*.json"))
    if len(paths) != 1:
        raise PrimitiveDraftError("Phase 5.2 requires exactly one PA completion.")
    return completion, paths[0]


def _load_draft_history(root: Path) -> list[PrimitiveProgramDraft]:
    directory = root / _DRAFT_DIRECTORY
    if not directory.exists():
        return []
    paths = sorted(directory.glob(f"{_DRAFT_PREFIX}*{_DRAFT_SUFFIX}"))
    expected = [
        directory / f"{_DRAFT_PREFIX}{index:04d}{_DRAFT_SUFFIX}"
        for index in range(1, len(paths) + 1)
    ]
    if paths != expected:
        raise PrimitiveDraftError("PrimitiveProgramDraft history is not contiguous.")
    drafts = [_load_draft(root, path) for path in paths]
    catalog_refs = [str(draft.record["primitive_catalog_ref"]) for draft in drafts]
    if len(catalog_refs) != len(set(catalog_refs)):
        raise PrimitiveDraftError(
            "PrimitiveProgramDraft history contains duplicate context inputs."
        )
    return drafts


def _load_draft(root: Path, path: Path) -> PrimitiveProgramDraft:
    value = _read_json_mapping(path, "PrimitiveProgramDraft")
    if set(value) != _RECORD_KEYS:
        raise PrimitiveDraftError("PrimitiveProgramDraft fields are invalid.")
    payload = dict(value)
    fingerprint = payload.pop("fingerprint", None)
    if (
        value.get("schema_version") != 1
        or value.get("record_type") != "PrimitiveProgramDraft"
        or not isinstance(fingerprint, str)
        or fingerprint != _fingerprint(payload)
    ):
        raise PrimitiveDraftError("PrimitiveProgramDraft identity is invalid.")
    pinned = (
        ("pa_completion_ref", "pa_completion_sha256", "pa_completion_fingerprint"),
        ("assignment_ref", "assignment_sha256", "assignment_fingerprint"),
        ("robot_state_ref", "robot_state_sha256", "robot_state_fingerprint"),
        (
            "primitive_catalog_ref",
            "primitive_catalog_sha256",
            "primitive_catalog_fingerprint",
        ),
    )
    loaded_records: dict[str, Mapping[str, object]] = {}
    for ref_field, sha_field, fingerprint_field in pinned:
        record_path = _resolve_ref(root, str(value.get(ref_field) or ""))
        if _sha256_path(record_path) != value.get(sha_field):
            raise PrimitiveDraftError(f"PrimitiveProgramDraft {ref_field} hash is invalid.")
        record = _read_json_mapping(record_path, ref_field)
        if record.get("fingerprint") != value.get(fingerprint_field):
            raise PrimitiveDraftError(
                f"PrimitiveProgramDraft {ref_field} fingerprint is invalid."
            )
        loaded_records[ref_field] = record
    catalog = loaded_records["primitive_catalog_ref"]
    entries = catalog.get("primitive_catalog")
    if not isinstance(entries, list):
        raise PrimitiveDraftError("PrimitiveProgramDraft catalog reference is invalid.")
    catalog_symbols = tuple(
        str(entry.get("primitive_symbol") or "")
        for entry in entries
        if isinstance(entry, Mapping)
    )
    steps = value.get("structural_steps")
    if not isinstance(steps, list) or any(
        not isinstance(step, Mapping)
        or set(step) != {"step_index", "primitive_symbol"}
        or step.get("step_index") != index
        for index, step in enumerate(steps, start=1)
    ):
        raise PrimitiveDraftError("PrimitiveProgramDraft structural_steps are invalid.")
    output = {
        "draft_status": value.get("draft_status"),
        "primitive_symbols": [step["primitive_symbol"] for step in steps],
        "unsupported_reason": value.get("unsupported_reason"),
    }
    _validate_model_output(output, catalog_symbols)
    authored_at_ns = value.get("authored_at_ns")
    if (
        isinstance(authored_at_ns, bool)
        or not isinstance(authored_at_ns, int)
        or authored_at_ns < 0
    ):
        raise PrimitiveDraftError("PrimitiveProgramDraft authored_at_ns is invalid.")
    return PrimitiveProgramDraft(path=path, record=dict(value))


def _resolve_ref(root: Path, record_ref: str) -> Path:
    relative = Path(record_ref)
    if (
        relative.is_absolute()
        or not relative.parts
        or "." in relative.parts
        or ".." in relative.parts
    ):
        raise PrimitiveDraftError("PrimitiveProgramDraft record ref is invalid.")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PrimitiveDraftError(
            "PrimitiveProgramDraft record ref leaves its interaction."
        ) from exc
    if not path.is_file():
        raise PrimitiveDraftError("PrimitiveProgramDraft pinned record is unavailable.")
    return path


def _read_json_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrimitiveDraftError(f"{label} could not be read.") from exc
    if not isinstance(value, dict):
        raise PrimitiveDraftError(f"{label} must be a JSON object.")
    return value


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise PrimitiveDraftError(
            f"PrimitiveProgramDraft already exists: {path.name}."
        ) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise PrimitiveDraftError(
            f"PrimitiveProgramDraft could not be written: {path.name}."
        ) from exc


def _fingerprint(value: Mapping[str, object] | list[object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "Phase52Diagnostic",
    "PrimitiveDraftError",
    "PrimitiveProgramDraft",
    "RobotAgentDraftRuntime",
    "author_primitive_program_draft",
    "read_phase_5_2_diagnostic",
]

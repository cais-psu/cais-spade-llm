"""Persist host-only blinded presentation mappings for ProductAgent interactions."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PRESENTATION_ROOT = Path("products/grounding/presentation")
_EVIDENCE_RECORD_NAME = "evidence_presentation_record.json"
_ALLOCATION_RECORD_NAME = "allocation_presentation_record.json"
_HOST_AUTHORITY = "Spec2Primitives.presentation_host"
_LIVE_OBSERVATION_KEY = "__live_observation__"


class PresentationRecordError(ValueError):
    """Raised when a blinded presentation mapping cannot be trusted."""


@dataclass(frozen=True)
class EvidencePresentationEntry:
    """Map one opaque ProductAgent handle to host-only source identity."""

    pa_handle: str
    evidence_type: str
    canonical_ref: str | None
    source_revision: str

    @property
    def ordering_key(self) -> str:
        """Return the canonical key used only by host-side order injection."""
        return self.canonical_ref or _LIVE_OBSERVATION_KEY

    def to_record(self) -> dict[str, object]:
        """Return the exact host-only JSON representation."""
        return {
            "pa_handle": self.pa_handle,
            "evidence_type": self.evidence_type,
            "canonical_ref": self.canonical_ref,
            "source_revision": self.source_revision,
        }


@dataclass(frozen=True)
class EvidencePresentationRecord:
    """Pin one randomized, host-only evidence presentation for an interaction."""

    record_path: Path = field(repr=False, compare=False)
    record_ref: str
    entries: tuple[EvidencePresentationEntry, ...]
    presentation_order: tuple[str, ...]
    created_at_ns: int
    fingerprint: str

    def to_record(self) -> dict[str, object]:
        """Return the exact persisted record."""
        return {
            "schema_version": 1,
            "record_type": "EvidencePresentationRecord",
            "authority": _HOST_AUTHORITY,
            "entries": [entry.to_record() for entry in self.entries],
            "presentation_order": list(self.presentation_order),
            "created_at_ns": self.created_at_ns,
            "fingerprint": self.fingerprint,
        }

    def assert_unchanged(self) -> None:
        """Raise if the immutable presentation record changed on disk."""
        persisted = _read_mapping(self.record_path)
        if persisted != self.to_record():
            raise PresentationRecordError("EvidencePresentationRecord changed after creation.")
        payload = dict(persisted)
        payload.pop("fingerprint")
        if _fingerprint(payload) != self.fingerprint:
            raise PresentationRecordError("EvidencePresentationRecord fingerprint is invalid.")

    def entry_for_handle(self, pa_handle: str) -> EvidencePresentationEntry:
        """Resolve one opaque handle without exposing its canonical identity to PA."""
        matches = [entry for entry in self.entries if entry.pa_handle == pa_handle]
        if len(matches) != 1:
            raise PresentationRecordError("Evidence presentation handle is unauthorized.")
        return matches[0]

    def opaque_reference(self, canonical_ref: str, *, kind: str) -> str:
        """Derive a stable interaction-local reference for a canonical host ref."""
        if kind not in {"citation", "typed_record"}:
            raise PresentationRecordError("Opaque reference kind is invalid.")
        if not isinstance(canonical_ref, str) or not canonical_ref:
            raise PresentationRecordError("Canonical presentation reference is invalid.")
        digest = hashlib.sha256(
            f"{self.fingerprint}\0{kind}\0{canonical_ref}".encode()
        ).hexdigest()[:24]
        return f"{kind}_{digest}"


@dataclass(frozen=True)
class AllocationResourceEntry:
    """Pin one capable resource in its exact PA presentation order."""

    resource_symbol: str
    resource_iri: str
    resource_jid: str

    def to_record(self) -> dict[str, str]:
        """Return the exact host-side resource mapping."""
        return {
            "resource_symbol": self.resource_symbol,
            "resource_iri": self.resource_iri,
            "resource_jid": self.resource_jid,
        }


@dataclass(frozen=True)
class AllocationEvidenceSource:
    """Describe one neutral accepted record before randomized presentation."""

    canonical_key: str
    record_type: str
    record_ref: str
    record_sha256: str
    field_path: str
    observation_handle: str | None
    candidate_handle: str | None
    source_frame: str
    neutral_projection: Mapping[str, object]

    def identity(self) -> tuple[str, ...]:
        """Return the exact canonical identity used for resume validation."""
        return (
            self.canonical_key,
            self.record_type,
            self.record_ref,
            self.record_sha256,
            self.field_path,
            self.observation_handle or "",
            self.candidate_handle or "",
            self.source_frame,
            _fingerprint(self.neutral_projection),
        )


@dataclass(frozen=True)
class AllocationEvidenceEntry:
    """Map one opaque state-evidence handle to canonical verifier inputs."""

    pa_handle: str
    canonical_key: str
    record_type: str
    record_ref: str
    record_sha256: str
    field_path: str
    observation_handle: str | None
    candidate_handle: str | None
    source_frame: str
    neutral_projection: Mapping[str, object]

    def to_record(self) -> dict[str, object]:
        """Return the exact host-only candidate mapping."""
        return {
            "pa_handle": self.pa_handle,
            "canonical_key": self.canonical_key,
            "record_type": self.record_type,
            "record_ref": self.record_ref,
            "record_sha256": self.record_sha256,
            "field_path": self.field_path,
            "observation_handle": self.observation_handle,
            "candidate_handle": self.candidate_handle,
            "source_frame": self.source_frame,
            "neutral_projection": _json_clone(self.neutral_projection),
        }

    def prompt_projection(self) -> dict[str, object]:
        """Return only the opaque handle and neutral evidence measurements."""
        return {
            "evidence_handle": self.pa_handle,
            "record_type": self.record_type,
            **dict(_json_clone(self.neutral_projection)),
        }

    def identity(self) -> tuple[str, ...]:
        """Return the canonical identity independent of presentation handle."""
        return (
            self.canonical_key,
            self.record_type,
            self.record_ref,
            self.record_sha256,
            self.field_path,
            self.observation_handle or "",
            self.candidate_handle or "",
            self.source_frame,
            _fingerprint(self.neutral_projection),
        )


@dataclass(frozen=True)
class AllocationPresentationRecord:
    """Pin randomized capable-resource and neutral-evidence presentation orders."""

    record_path: Path = field(repr=False, compare=False)
    record_ref: str
    evidence_presentation_ref: str
    evidence_presentation_sha256: str
    evidence_presentation_fingerprint: str
    process_symbol: str
    process_iri: str
    feature_iri: str
    current_state_iri: str
    desired_state_iri: str
    resources: tuple[AllocationResourceEntry, ...]
    evidence_entries: tuple[AllocationEvidenceEntry, ...]
    created_at_ns: int
    fingerprint: str

    @property
    def resource_order(self) -> tuple[str, ...]:
        """Return the exact resource enum order shown to PA."""
        return tuple(entry.resource_symbol for entry in self.resources)

    @property
    def neutral_candidate_order(self) -> tuple[str, ...]:
        """Return the exact shared evidence enum order shown to PA."""
        return tuple(entry.pa_handle for entry in self.evidence_entries)

    def evidence_for_handle(self, pa_handle: str) -> AllocationEvidenceEntry:
        """Resolve one PA-authored neutral state-evidence handle."""
        matches = [entry for entry in self.evidence_entries if entry.pa_handle == pa_handle]
        if len(matches) != 1:
            raise PresentationRecordError(
                "Allocation state-evidence handle is not authorized."
            )
        return matches[0]

    def to_record(self) -> dict[str, object]:
        """Return the exact immutable allocation presentation record."""
        return {
            "schema_version": 1,
            "record_type": "AllocationPresentationRecord",
            "authority": _HOST_AUTHORITY,
            "evidence_presentation_ref": self.evidence_presentation_ref,
            "evidence_presentation_sha256": self.evidence_presentation_sha256,
            "evidence_presentation_fingerprint": self.evidence_presentation_fingerprint,
            "process_symbol": self.process_symbol,
            "process_iri": self.process_iri,
            "feature_iri": self.feature_iri,
            "current_state_iri": self.current_state_iri,
            "desired_state_iri": self.desired_state_iri,
            "resources": [entry.to_record() for entry in self.resources],
            "neutral_evidence": [entry.to_record() for entry in self.evidence_entries],
            "resource_presentation_order": list(self.resource_order),
            "neutral_candidate_presentation_order": list(self.neutral_candidate_order),
            "created_at_ns": self.created_at_ns,
            "fingerprint": self.fingerprint,
        }

    def assert_unchanged(self) -> None:
        """Raise if the allocation presentation or pinned evidence record changed."""
        persisted = _read_mapping(self.record_path)
        if persisted != self.to_record():
            raise PresentationRecordError("AllocationPresentationRecord changed.")
        payload = dict(persisted)
        payload.pop("fingerprint")
        if _fingerprint(payload) != self.fingerprint:
            raise PresentationRecordError("AllocationPresentationRecord fingerprint is invalid.")
        evidence_path = self.record_path.parents[3] / self.evidence_presentation_ref
        if _sha256_path(evidence_path) != self.evidence_presentation_sha256:
            raise PresentationRecordError("Pinned EvidencePresentationRecord changed.")


def load_or_create_allocation_presentation(  # noqa: PLR0913
    interaction_root: Path,
    *,
    evidence_presentation: EvidencePresentationRecord,
    process_symbol: str,
    process_iri: str,
    feature_iri: str,
    current_state_iri: str,
    desired_state_iri: str,
    resources: Sequence[tuple[str, str, str]],
    evidence_sources: Sequence[AllocationEvidenceSource],
    explicit_resource_order: Sequence[str] | None = None,
    explicit_candidate_order: Sequence[str] | None = None,
) -> AllocationPresentationRecord:
    """Load or create the exact randomized allocation presentation for one run."""
    root = Path(interaction_root).resolve()
    evidence_presentation.assert_unchanged()
    validated_resources = _validated_allocation_resources(resources)
    validated_evidence = _validated_allocation_evidence(evidence_sources)
    context = (process_symbol, process_iri, feature_iri, current_state_iri, desired_state_iri)
    if not all(isinstance(item, str) and item for item in context):
        raise PresentationRecordError("Allocation presentation context is invalid.")
    record_path = root / _PRESENTATION_ROOT / _ALLOCATION_RECORD_NAME
    if record_path.exists():
        record = _load_allocation_presentation(root, record_path)
        _assert_allocation_authority(
            record,
            evidence_presentation=evidence_presentation,
            context=context,
            resources=validated_resources,
            evidence_sources=validated_evidence,
            explicit_resource_order=explicit_resource_order,
            explicit_candidate_order=explicit_candidate_order,
        )
        record.assert_unchanged()
        return record

    ordered_resources = list(validated_resources)
    ordered_evidence = list(validated_evidence)
    _apply_explicit_or_secure_order(
        ordered_resources,
        keys=[item[0] for item in ordered_resources],
        explicit_order=explicit_resource_order,
        label="resource",
    )
    _apply_explicit_or_secure_order(
        ordered_evidence,
        keys=[item.canonical_key for item in ordered_evidence],
        explicit_order=explicit_candidate_order,
        label="neutral candidate",
    )
    resource_entries = tuple(
        AllocationResourceEntry(*resource) for resource in ordered_resources
    )
    evidence_entries = tuple(
        _allocation_evidence_entry(source, index)
        for index, source in enumerate(ordered_evidence, start=1)
    )
    evidence_sha256 = _sha256_path(evidence_presentation.record_path)
    created_at_ns = time.time_ns()
    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "AllocationPresentationRecord",
        "authority": _HOST_AUTHORITY,
        "evidence_presentation_ref": evidence_presentation.record_ref,
        "evidence_presentation_sha256": evidence_sha256,
        "evidence_presentation_fingerprint": evidence_presentation.fingerprint,
        "process_symbol": process_symbol,
        "process_iri": process_iri,
        "feature_iri": feature_iri,
        "current_state_iri": current_state_iri,
        "desired_state_iri": desired_state_iri,
        "resources": [entry.to_record() for entry in resource_entries],
        "neutral_evidence": [entry.to_record() for entry in evidence_entries],
        "resource_presentation_order": [entry.resource_symbol for entry in resource_entries],
        "neutral_candidate_presentation_order": [
            entry.pa_handle for entry in evidence_entries
        ],
        "created_at_ns": created_at_ns,
    }
    fingerprint = _fingerprint(payload)
    payload["fingerprint"] = fingerprint
    _write_json_exclusive(record_path, payload)
    return AllocationPresentationRecord(
        record_path=record_path,
        record_ref=record_path.relative_to(root).as_posix(),
        evidence_presentation_ref=evidence_presentation.record_ref,
        evidence_presentation_sha256=evidence_sha256,
        evidence_presentation_fingerprint=evidence_presentation.fingerprint,
        process_symbol=process_symbol,
        process_iri=process_iri,
        feature_iri=feature_iri,
        current_state_iri=current_state_iri,
        desired_state_iri=desired_state_iri,
        resources=resource_entries,
        evidence_entries=evidence_entries,
        created_at_ns=created_at_ns,
        fingerprint=fingerprint,
    )


def load_allocation_presentation(interaction_root: Path) -> AllocationPresentationRecord:
    """Load and verify the immutable allocation presentation for an interaction."""
    root = Path(interaction_root).resolve()
    record = _load_allocation_presentation(
        root,
        root / _PRESENTATION_ROOT / _ALLOCATION_RECORD_NAME,
    )
    record.assert_unchanged()
    return record


def load_evidence_presentation(interaction_root: Path) -> EvidencePresentationRecord:
    """Load and verify the immutable evidence presentation for an interaction."""
    root = Path(interaction_root).resolve()
    record = _load_evidence_presentation(
        root,
        root / _PRESENTATION_ROOT / _EVIDENCE_RECORD_NAME,
    )
    record.assert_unchanged()
    return record


def load_or_create_evidence_presentation(
    interaction_root: Path,
    *,
    sources: Sequence[tuple[str | None, str, str]],
    explicit_order: Sequence[str] | None = None,
) -> EvidencePresentationRecord:
    """Load or atomically create one securely randomized evidence presentation.

    `explicit_order` is a host-only reproducibility hook. Its values must be the
    exact canonical source refs plus ``__live_observation__`` for the live source.
    """
    root = Path(interaction_root).resolve()
    record_path = root / _PRESENTATION_ROOT / _EVIDENCE_RECORD_NAME
    expected = _validated_sources(sources)
    if record_path.exists():
        record = _load_evidence_presentation(root, record_path)
        if {
            (entry.canonical_ref, entry.evidence_type, entry.source_revision)
            for entry in record.entries
        } != set(expected):
            raise PresentationRecordError(
                "Evidence authority changed after presentation was pinned."
            )
        if explicit_order is not None and tuple(explicit_order) != tuple(
            entry.ordering_key for entry in record.entries
        ):
            raise PresentationRecordError(
                "Injected evidence order differs from the persisted presentation."
            )
        record.assert_unchanged()
        return record

    ordered = list(expected)
    canonical_keys = [canonical_ref or _LIVE_OBSERVATION_KEY for canonical_ref, _, _ in ordered]
    if explicit_order is None:
        secrets.SystemRandom().shuffle(ordered)
    else:
        requested = tuple(explicit_order)
        if len(requested) != len(set(requested)) or set(requested) != set(canonical_keys):
            raise PresentationRecordError(
                "Injected evidence order must be an exact source permutation."
            )
        by_key = {
            canonical_ref or _LIVE_OBSERVATION_KEY: (canonical_ref, evidence_type, revision)
            for canonical_ref, evidence_type, revision in ordered
        }
        ordered = [by_key[key] for key in requested]

    type_counts: dict[str, int] = {}
    entries: list[EvidencePresentationEntry] = []
    for canonical_ref, evidence_type, revision in ordered:
        type_counts[evidence_type] = type_counts.get(evidence_type, 0) + 1
        prefix = {
            "CAD": "cad_candidate",
            "document": "document_candidate",
            "observation": "observation_candidate",
        }.get(evidence_type)
        if prefix is None:
            raise PresentationRecordError("Evidence presentation type is unsupported.")
        entries.append(
            EvidencePresentationEntry(
                pa_handle=f"{prefix}_{type_counts[evidence_type]:04d}",
                evidence_type=evidence_type,
                canonical_ref=canonical_ref,
                source_revision=revision,
            )
        )
    created_at_ns = time.time_ns()
    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "EvidencePresentationRecord",
        "authority": _HOST_AUTHORITY,
        "entries": [entry.to_record() for entry in entries],
        "presentation_order": [entry.pa_handle for entry in entries],
        "created_at_ns": created_at_ns,
    }
    fingerprint = _fingerprint(payload)
    payload["fingerprint"] = fingerprint
    _write_json_exclusive(record_path, payload)
    return EvidencePresentationRecord(
        record_path=record_path,
        record_ref=record_path.relative_to(root).as_posix(),
        entries=tuple(entries),
        presentation_order=tuple(entry.pa_handle for entry in entries),
        created_at_ns=created_at_ns,
        fingerprint=fingerprint,
    )


def _load_allocation_presentation(
    root: Path,
    record_path: Path,
) -> AllocationPresentationRecord:
    payload = _read_mapping(record_path)
    expected_keys = {
        "schema_version",
        "record_type",
        "authority",
        "evidence_presentation_ref",
        "evidence_presentation_sha256",
        "evidence_presentation_fingerprint",
        "process_symbol",
        "process_iri",
        "feature_iri",
        "current_state_iri",
        "desired_state_iri",
        "resources",
        "neutral_evidence",
        "resource_presentation_order",
        "neutral_candidate_presentation_order",
        "created_at_ns",
        "fingerprint",
    }
    if set(payload) != expected_keys or (
        payload.get("schema_version") != 1
        or payload.get("record_type") != "AllocationPresentationRecord"
        or payload.get("authority") != _HOST_AUTHORITY
    ):
        raise PresentationRecordError("AllocationPresentationRecord fields are invalid.")
    scalar_fields = (
        "evidence_presentation_ref",
        "evidence_presentation_sha256",
        "evidence_presentation_fingerprint",
        "process_symbol",
        "process_iri",
        "feature_iri",
        "current_state_iri",
        "desired_state_iri",
        "fingerprint",
    )
    if not all(
        isinstance(payload.get(field_name), str) and payload[field_name]
        for field_name in scalar_fields
    ):
        raise PresentationRecordError("Allocation presentation identity is invalid.")
    resources = _load_allocation_resource_entries(payload.get("resources"))
    evidence_entries = _load_allocation_evidence_entries(payload.get("neutral_evidence"))
    resource_order = payload.get("resource_presentation_order")
    candidate_order = payload.get("neutral_candidate_presentation_order")
    created_at_ns = payload.get("created_at_ns")
    if (
        not isinstance(resource_order, list)
        or tuple(resource_order) != tuple(entry.resource_symbol for entry in resources)
        or not isinstance(candidate_order, list)
        or tuple(candidate_order) != tuple(entry.pa_handle for entry in evidence_entries)
        or isinstance(created_at_ns, bool)
        or not isinstance(created_at_ns, int)
        or created_at_ns <= 0
    ):
        raise PresentationRecordError("Allocation presentation order is invalid.")
    without_fingerprint = dict(payload)
    fingerprint = str(without_fingerprint.pop("fingerprint"))
    if _fingerprint(without_fingerprint) != fingerprint:
        raise PresentationRecordError("AllocationPresentationRecord fingerprint is invalid.")
    return AllocationPresentationRecord(
        record_path=record_path,
        record_ref=record_path.relative_to(root).as_posix(),
        evidence_presentation_ref=str(payload["evidence_presentation_ref"]),
        evidence_presentation_sha256=str(payload["evidence_presentation_sha256"]),
        evidence_presentation_fingerprint=str(payload["evidence_presentation_fingerprint"]),
        process_symbol=str(payload["process_symbol"]),
        process_iri=str(payload["process_iri"]),
        feature_iri=str(payload["feature_iri"]),
        current_state_iri=str(payload["current_state_iri"]),
        desired_state_iri=str(payload["desired_state_iri"]),
        resources=resources,
        evidence_entries=evidence_entries,
        created_at_ns=created_at_ns,
        fingerprint=fingerprint,
    )


def _load_allocation_resource_entries(value: object) -> tuple[AllocationResourceEntry, ...]:
    if not isinstance(value, list) or not value:
        raise PresentationRecordError("Allocation resources are invalid.")
    entries: list[AllocationResourceEntry] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "resource_symbol",
            "resource_iri",
            "resource_jid",
        }:
            raise PresentationRecordError("Allocation resource entry is invalid.")
        fields = tuple(item.get(key) for key in ("resource_symbol", "resource_iri", "resource_jid"))
        if not all(isinstance(field_value, str) and field_value for field_value in fields):
            raise PresentationRecordError("Allocation resource values are invalid.")
        entries.append(AllocationResourceEntry(*fields))  # type: ignore[arg-type]
    if len({entry.resource_symbol for entry in entries}) != len(entries):
        raise PresentationRecordError("Allocation resource symbols must be unique.")
    return tuple(entries)


def _load_allocation_evidence_entries(value: object) -> tuple[AllocationEvidenceEntry, ...]:
    if not isinstance(value, list) or not value:
        raise PresentationRecordError("Allocation neutral evidence is invalid.")
    entries: list[AllocationEvidenceEntry] = []
    expected_keys = {
        "pa_handle",
        "canonical_key",
        "record_type",
        "record_ref",
        "record_sha256",
        "field_path",
        "observation_handle",
        "candidate_handle",
        "source_frame",
        "neutral_projection",
    }
    for item in value:
        if not isinstance(item, Mapping) or set(item) != expected_keys:
            raise PresentationRecordError("Allocation evidence entry is invalid.")
        text_fields = (
            item.get("pa_handle"),
            item.get("canonical_key"),
            item.get("record_type"),
            item.get("record_ref"),
            item.get("record_sha256"),
            item.get("field_path"),
            item.get("source_frame"),
        )
        observation_handle = item.get("observation_handle")
        candidate_handle = item.get("candidate_handle")
        projection = item.get("neutral_projection")
        if (
            not all(isinstance(field_value, str) and field_value for field_value in text_fields)
            or (observation_handle is not None and not isinstance(observation_handle, str))
            or (candidate_handle is not None and not isinstance(candidate_handle, str))
            or not isinstance(projection, Mapping)
        ):
            raise PresentationRecordError("Allocation evidence values are invalid.")
        entries.append(
            AllocationEvidenceEntry(
                pa_handle=str(item["pa_handle"]),
                canonical_key=str(item["canonical_key"]),
                record_type=str(item["record_type"]),
                record_ref=str(item["record_ref"]),
                record_sha256=str(item["record_sha256"]),
                field_path=str(item["field_path"]),
                observation_handle=observation_handle,
                candidate_handle=candidate_handle,
                source_frame=str(item["source_frame"]),
                neutral_projection=_json_clone(projection),
            )
        )
    if (
        len({entry.pa_handle for entry in entries}) != len(entries)
        or len({entry.canonical_key for entry in entries}) != len(entries)
    ):
        raise PresentationRecordError("Allocation evidence handles must be unique.")
    return tuple(entries)


def _assert_allocation_authority(  # noqa: PLR0913
    record: AllocationPresentationRecord,
    *,
    evidence_presentation: EvidencePresentationRecord,
    context: tuple[str, str, str, str, str],
    resources: tuple[tuple[str, str, str], ...],
    evidence_sources: tuple[AllocationEvidenceSource, ...],
    explicit_resource_order: Sequence[str] | None,
    explicit_candidate_order: Sequence[str] | None,
) -> None:
    expected_context = (
        record.process_symbol,
        record.process_iri,
        record.feature_iri,
        record.current_state_iri,
        record.desired_state_iri,
    )
    if (
        expected_context != context
        or record.evidence_presentation_ref != evidence_presentation.record_ref
        or record.evidence_presentation_fingerprint != evidence_presentation.fingerprint
        or record.evidence_presentation_sha256
        != _sha256_path(evidence_presentation.record_path)
        or {tuple(entry.to_record().values()) for entry in record.resources} != set(resources)
        or {entry.identity() for entry in record.evidence_entries}
        != {source.identity() for source in evidence_sources}
    ):
        raise PresentationRecordError("Allocation authority changed after presentation.")
    if explicit_resource_order is not None and tuple(explicit_resource_order) != record.resource_order:
        raise PresentationRecordError("Injected resource order differs from persisted order.")
    expected_candidate_keys = tuple(entry.canonical_key for entry in record.evidence_entries)
    if (
        explicit_candidate_order is not None
        and tuple(explicit_candidate_order) != expected_candidate_keys
    ):
        raise PresentationRecordError("Injected candidate order differs from persisted order.")


def _validated_allocation_resources(
    resources: Sequence[tuple[str, str, str]],
) -> tuple[tuple[str, str, str], ...]:
    values = tuple(resources)
    if (
        not values
        or not all(
            isinstance(item, tuple)
            and len(item) == 3
            and all(isinstance(field_value, str) and field_value for field_value in item)
            for item in values
        )
        or len({item[0] for item in values}) != len(values)
    ):
        raise PresentationRecordError("Allocation capable-resource authority is invalid.")
    return values


def _validated_allocation_evidence(
    sources: Sequence[AllocationEvidenceSource],
) -> tuple[AllocationEvidenceSource, ...]:
    values = tuple(sources)
    if not values or len({source.canonical_key for source in values}) != len(values):
        raise PresentationRecordError("Allocation neutral evidence authority is invalid.")
    for source in values:
        if (
            not all(
                isinstance(item, str) and item
                for item in (
                    source.canonical_key,
                    source.record_type,
                    source.record_ref,
                    source.record_sha256,
                    source.field_path,
                    source.source_frame,
                )
            )
            or len(source.record_sha256) != 64
            or not isinstance(source.neutral_projection, Mapping)
            or (
                source.observation_handle is not None
                and not isinstance(source.observation_handle, str)
            )
            or source.candidate_handle is not None
            and not isinstance(source.candidate_handle, str)
        ):
            raise PresentationRecordError("Allocation neutral evidence source is invalid.")
    return values


def _apply_explicit_or_secure_order(
    values: list[Any],
    *,
    keys: Sequence[str],
    explicit_order: Sequence[str] | None,
    label: str,
) -> None:
    if explicit_order is None:
        secrets.SystemRandom().shuffle(values)
        return
    requested = tuple(explicit_order)
    if len(requested) != len(set(requested)) or set(requested) != set(keys):
        raise PresentationRecordError(
            f"Injected {label} order must be an exact authority permutation."
        )
    by_key = dict(zip(keys, values, strict=True))
    values[:] = [by_key[key] for key in requested]


def _allocation_evidence_entry(
    source: AllocationEvidenceSource,
    index: int,
) -> AllocationEvidenceEntry:
    return AllocationEvidenceEntry(
        pa_handle=f"state_evidence_{index:04d}",
        canonical_key=source.canonical_key,
        record_type=source.record_type,
        record_ref=source.record_ref,
        record_sha256=source.record_sha256,
        field_path=source.field_path,
        observation_handle=source.observation_handle,
        candidate_handle=source.candidate_handle,
        source_frame=source.source_frame,
        neutral_projection=_json_clone(source.neutral_projection),
    )


def _load_evidence_presentation(
    root: Path,
    record_path: Path,
) -> EvidencePresentationRecord:
    payload = _read_mapping(record_path)
    if set(payload) != {
        "schema_version",
        "record_type",
        "authority",
        "entries",
        "presentation_order",
        "created_at_ns",
        "fingerprint",
    } or (
        payload.get("schema_version") != 1
        or payload.get("record_type") != "EvidencePresentationRecord"
        or payload.get("authority") != _HOST_AUTHORITY
    ):
        raise PresentationRecordError("EvidencePresentationRecord fields are invalid.")
    raw_entries = payload.get("entries")
    raw_order = payload.get("presentation_order")
    created_at_ns = payload.get("created_at_ns")
    fingerprint = payload.get("fingerprint")
    if (
        not isinstance(raw_entries, list)
        or not raw_entries
        or not isinstance(raw_order, list)
        or isinstance(created_at_ns, bool)
        or not isinstance(created_at_ns, int)
        or created_at_ns <= 0
        or not isinstance(fingerprint, str)
        or len(fingerprint) != 64
    ):
        raise PresentationRecordError("EvidencePresentationRecord values are invalid.")
    entries: list[EvidencePresentationEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != {
            "pa_handle",
            "evidence_type",
            "canonical_ref",
            "source_revision",
        }:
            raise PresentationRecordError("Evidence presentation entry is invalid.")
        pa_handle = raw_entry.get("pa_handle")
        evidence_type = raw_entry.get("evidence_type")
        canonical_ref = raw_entry.get("canonical_ref")
        source_revision = raw_entry.get("source_revision")
        if (
            not isinstance(pa_handle, str)
            or not pa_handle
            or evidence_type not in {"CAD", "document", "observation"}
            or (canonical_ref is not None and not isinstance(canonical_ref, str))
            or not isinstance(source_revision, str)
            or not source_revision
        ):
            raise PresentationRecordError("Evidence presentation entry values are invalid.")
        entries.append(
            EvidencePresentationEntry(
                pa_handle=pa_handle,
                evidence_type=str(evidence_type),
                canonical_ref=canonical_ref,
                source_revision=source_revision,
            )
        )
    if (
        len({entry.pa_handle for entry in entries}) != len(entries)
        or tuple(raw_order) != tuple(entry.pa_handle for entry in entries)
    ):
        raise PresentationRecordError("Evidence presentation order is invalid.")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("fingerprint")
    if _fingerprint(without_fingerprint) != fingerprint:
        raise PresentationRecordError("EvidencePresentationRecord fingerprint is invalid.")
    return EvidencePresentationRecord(
        record_path=record_path,
        record_ref=record_path.relative_to(root).as_posix(),
        entries=tuple(entries),
        presentation_order=tuple(str(item) for item in raw_order),
        created_at_ns=created_at_ns,
        fingerprint=fingerprint,
    )


def _validated_sources(
    sources: Sequence[tuple[str | None, str, str]],
) -> tuple[tuple[str | None, str, str], ...]:
    values = tuple(sources)
    if not values or len({item[0] or _LIVE_OBSERVATION_KEY for item in values}) != len(values):
        raise PresentationRecordError("Evidence presentation sources are invalid.")
    for canonical_ref, evidence_type, revision in values:
        if (
            (canonical_ref is not None and (not isinstance(canonical_ref, str) or not canonical_ref))
            or evidence_type not in {"CAD", "document", "observation"}
            or not isinstance(revision, str)
            or not revision
            or (evidence_type == "observation") != (canonical_ref is None)
        ):
            raise PresentationRecordError("Evidence presentation source is invalid.")
    return values


def _read_mapping(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PresentationRecordError("Presentation record could not be read.") from exc
    if not isinstance(value, Mapping):
        raise PresentationRecordError("Presentation record must be a JSON object.")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise PresentationRecordError("Presentation record persistence failed.") from exc


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_path(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PresentationRecordError("Pinned presentation record could not be read.") from exc


def _json_clone(value: object) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise PresentationRecordError("Presentation value is not JSON-compatible.") from exc


__all__ = [
    "AllocationEvidenceEntry",
    "AllocationEvidenceSource",
    "AllocationPresentationRecord",
    "AllocationResourceEntry",
    "EvidencePresentationEntry",
    "EvidencePresentationRecord",
    "PresentationRecordError",
    "load_allocation_presentation",
    "load_evidence_presentation",
    "load_or_create_allocation_presentation",
    "load_or_create_evidence_presentation",
]

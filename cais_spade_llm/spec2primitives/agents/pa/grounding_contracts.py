"""Define and validate the PA-owned Phase 4.3 grounding contracts.

The contracts in this module describe robot-independent product context.  They
do not select primitive operations, contact a resource, or treat an ontology
graph as an operational completeness oracle.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdflib import Literal, URIRef

from cais_spade_llm.spec2primitives.agents.pa.product_context import ABoxSnapshot

_BINDING_STATUSES = frozenset(
    {"accepted", "ambiguous", "rejected", "stale", "unavailable"}
)
_GROUNDING_STATEMENT_STATUSES = frozenset({"directly_stated", "inferred"})
_INFORMATION_NEED_STATUSES = frozenset({"open", "resolved", "exhausted"})
_ACTION_ATTEMPT_STATUSES = frozenset(
    {"accepted", "no_change", "rejected", "unavailable"}
)
_GROUNDING_DECISION_TYPES = frozenset(
    {"request_evidence", "ready_for_ontology", "incomplete"}
)
_GROUNDING_SESSION_STATUSES = frozenset(
    {
        "grounding",
        "waiting_for_evidence",
        "waiting_for_user",
        "ready_for_ontology",
        "complete",
        "incomplete",
        "ontology_gap",
    }
)
_INFORMATION_STATUSES = frozenset({"enough", "partial", "not_enough"})
_SAFE_SYMBOL = re.compile(r"^[^\x00-\x1f\x7f]+$")
_VIEW_ROOT = Path("products/grounding/product_context")
_SESSION_ROOT = Path("products/grounding/session")
_DELTA_KEYS = frozenset(
    {
        "schema_version",
        "delta_number",
        "producer",
        "assertions",
        "uncertainty",
        "unresolved_evidence_needs",
        "typed_context_refs",
    }
)


class GroundingContractError(ValueError):
    """Raised when a Phase 4.3 grounding contract is invalid."""


@dataclass(frozen=True)
class GroundingStatement:
    """Record one source-cited statement without ontology commitments."""

    statement_id: str
    text: str
    status: str
    sources: tuple[str, ...]
    reason: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GroundingStatement:
        """Validate and construct one grounding statement."""
        _require_exact_keys(
            value,
            {"statement_id", "text", "status", "sources", "reason"},
            "GroundingStatement",
        )
        status = _required_string(value["status"], "GroundingStatement.status")
        if status not in _GROUNDING_STATEMENT_STATUSES:
            raise GroundingContractError("GroundingStatement.status is invalid.")
        sources = _string_tuple(value["sources"], "GroundingStatement.sources")
        if not sources:
            raise GroundingContractError(
                "GroundingStatement.sources must contain supporting sources."
            )
        return cls(
            statement_id=_required_symbol(
                value["statement_id"], "GroundingStatement.statement_id"
            ),
            text=_required_string(value["text"], "GroundingStatement.text"),
            status=status,
            sources=sources,
            reason=_required_string(value["reason"], "GroundingStatement.reason"),
        )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe statement record."""
        return {
            "statement_id": self.statement_id,
            "text": self.text,
            "status": self.status,
            "sources": list(self.sources),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class InformationNeed:
    """Describe one unresolved question and the records that may answer it."""

    need_id: str
    question: str
    required: bool
    sources: tuple[str, ...]
    accepted_record_types: tuple[str, ...]
    status: str
    answer_statement_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> InformationNeed:
        """Validate and construct one information need."""
        _require_exact_keys(
            value,
            {
                "need_id",
                "question",
                "required",
                "sources",
                "accepted_record_types",
                "status",
                "answer_statement_ids",
            },
            "InformationNeed",
        )
        required = value["required"]
        if not isinstance(required, bool):
            raise GroundingContractError("InformationNeed.required must be a boolean.")
        sources = _string_tuple(value["sources"], "InformationNeed.sources")
        if not sources:
            raise GroundingContractError(
                "InformationNeed.sources must identify why the question is open."
            )
        status = _required_string(value["status"], "InformationNeed.status")
        if status not in _INFORMATION_NEED_STATUSES:
            raise GroundingContractError("InformationNeed.status is invalid.")
        accepted_record_types = _string_tuple(
            value["accepted_record_types"],
            "InformationNeed.accepted_record_types",
        )
        if status == "open" and not accepted_record_types:
            raise GroundingContractError(
                "An open InformationNeed requires accepted_record_types."
            )
        answer_statement_ids = _string_tuple(
            value["answer_statement_ids"],
            "InformationNeed.answer_statement_ids",
        )
        if status == "resolved" and not answer_statement_ids:
            raise GroundingContractError(
                "A resolved InformationNeed requires answer_statement_ids."
            )
        if status != "resolved" and answer_statement_ids:
            raise GroundingContractError(
                "Only a resolved InformationNeed may have answer_statement_ids."
            )
        return cls(
            need_id=_required_symbol(value["need_id"], "InformationNeed.need_id"),
            question=_required_string(
                value["question"], "InformationNeed.question"
            ),
            required=required,
            sources=sources,
            accepted_record_types=accepted_record_types,
            status=status,
            answer_statement_ids=answer_statement_ids,
        )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe information-need record."""
        return {
            "need_id": self.need_id,
            "question": self.question,
            "required": self.required,
            "sources": list(self.sources),
            "accepted_record_types": list(self.accepted_record_types),
            "status": self.status,
            "answer_statement_ids": list(self.answer_statement_ids),
        }


@dataclass(frozen=True)
class GroundingActionAttempt:
    """Record one exact provider and source revision attempted for a need."""

    attempt_id: str
    need_id: str
    provider_id: str
    source_ref: str
    source_revision: str
    status: str
    record_refs: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GroundingActionAttempt:
        """Validate and construct one grounding action attempt."""
        _require_exact_keys(
            value,
            {
                "attempt_id",
                "need_id",
                "provider_id",
                "source_ref",
                "source_revision",
                "status",
                "record_refs",
            },
            "GroundingActionAttempt",
        )
        status = _required_string(value["status"], "GroundingActionAttempt.status")
        if status not in _ACTION_ATTEMPT_STATUSES:
            raise GroundingContractError("GroundingActionAttempt.status is invalid.")
        record_refs = _string_tuple(
            value["record_refs"], "GroundingActionAttempt.record_refs"
        )
        if status == "accepted" and not record_refs:
            raise GroundingContractError(
                "An accepted GroundingActionAttempt requires record_refs."
            )
        return cls(
            attempt_id=_required_symbol(
                value["attempt_id"], "GroundingActionAttempt.attempt_id"
            ),
            need_id=_required_symbol(
                value["need_id"], "GroundingActionAttempt.need_id"
            ),
            provider_id=_required_symbol(
                value["provider_id"], "GroundingActionAttempt.provider_id"
            ),
            source_ref=_required_symbol(
                value["source_ref"], "GroundingActionAttempt.source_ref"
            ),
            source_revision=_required_symbol(
                value["source_revision"], "GroundingActionAttempt.source_revision"
            ),
            status=status,
            record_refs=record_refs,
        )

    @property
    def action_key(self) -> tuple[str, str, str]:
        """Return the exact replay-protection key for this attempt."""
        return (self.need_id, self.provider_id, self.source_revision)

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe action-attempt record."""
        return {
            "attempt_id": self.attempt_id,
            "need_id": self.need_id,
            "provider_id": self.provider_id,
            "source_ref": self.source_ref,
            "source_revision": self.source_revision,
            "status": self.status,
            "record_refs": list(self.record_refs),
        }


@dataclass(frozen=True)
class GroundingDecision:
    """Hold one bounded next action or terminal PA grounding decision."""

    decision_type: str
    need_id: str | None
    provider_id: str | None
    source_ref: str | None
    source_revision: str | None
    query: str | None
    reason: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GroundingDecision:
        """Validate and construct one grounding decision."""
        _require_exact_keys(
            value,
            {
                "decision_type",
                "need_id",
                "provider_id",
                "source_ref",
                "source_revision",
                "query",
                "reason",
            },
            "GroundingDecision",
        )
        decision_type = _required_string(
            value["decision_type"], "GroundingDecision.decision_type"
        )
        if decision_type not in _GROUNDING_DECISION_TYPES:
            raise GroundingContractError("GroundingDecision.decision_type is invalid.")
        need_id = _optional_symbol(value["need_id"], "GroundingDecision.need_id")
        provider_id = _optional_symbol(
            value["provider_id"], "GroundingDecision.provider_id"
        )
        source_ref = _optional_symbol(
            value["source_ref"], "GroundingDecision.source_ref"
        )
        source_revision = _optional_symbol(
            value["source_revision"], "GroundingDecision.source_revision"
        )
        query = _optional_string(value["query"], "GroundingDecision.query")
        action_values = (need_id, provider_id, source_ref, source_revision, query)
        if decision_type == "request_evidence" and any(
            item is None for item in action_values
        ):
            raise GroundingContractError(
                "A request_evidence GroundingDecision requires one exact action."
            )
        if decision_type != "request_evidence" and any(
            item is not None for item in action_values
        ):
            raise GroundingContractError(
                "A terminal GroundingDecision must not contain an action."
            )
        return cls(
            decision_type=decision_type,
            need_id=need_id,
            provider_id=provider_id,
            source_ref=source_ref,
            source_revision=source_revision,
            query=query,
            reason=_required_string(value["reason"], "GroundingDecision.reason"),
        )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe decision record."""
        return {
            "decision_type": self.decision_type,
            "need_id": self.need_id,
            "provider_id": self.provider_id,
            "source_ref": self.source_ref,
            "source_revision": self.source_revision,
            "query": self.query,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GroundingSession:
    """Hold PA's ontology-neutral, source-cited grounding state."""

    revision: int
    requirement_text: str
    statements: tuple[GroundingStatement, ...]
    information_needs: tuple[InformationNeed, ...]
    attempted_actions: tuple[GroundingActionAttempt, ...]
    evidence_refs: tuple[str, ...]
    decision: GroundingDecision
    status: str
    missing_information: tuple[str, ...]
    information_status: str
    fingerprint: str

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        *,
        revision: int,
        requirement_text: str,
        statements: Sequence[GroundingStatement] = (),
        information_needs: Sequence[InformationNeed] = (),
        attempted_actions: Sequence[GroundingActionAttempt] = (),
        evidence_refs: Sequence[str] = (),
        decision: GroundingDecision,
        status: str,
        information_status: str,
    ) -> GroundingSession:
        """Create a fingerprinted session from already validated child records."""
        missing_information = tuple(
            need.question
            for need in information_needs
            if need.status in {"open", "exhausted"}
        )
        record = {
            "schema_version": 1,
            "record_type": "GroundingSession",
            "revision": revision,
            "requirement_text": requirement_text,
            "statements": [item.to_record() for item in statements],
            "information_needs": [item.to_record() for item in information_needs],
            "attempted_actions": [item.to_record() for item in attempted_actions],
            "evidence_refs": list(evidence_refs),
            "decision": decision.to_record(),
            "status": status,
            "missing_information": list(missing_information),
            "information_status": information_status,
        }
        record["fingerprint"] = _fingerprint(record)
        return cls.from_mapping(record)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GroundingSession:
        """Validate and reconstruct one persisted grounding session revision."""
        _require_exact_keys(
            value,
            {
                "schema_version",
                "record_type",
                "revision",
                "requirement_text",
                "statements",
                "information_needs",
                "attempted_actions",
                "evidence_refs",
                "decision",
                "status",
                "missing_information",
                "information_status",
                "fingerprint",
            },
            "GroundingSession",
        )
        if value["schema_version"] != 1 or value["record_type"] != "GroundingSession":
            raise GroundingContractError("GroundingSession identity is invalid.")
        statements = _record_tuple(
            value["statements"], GroundingStatement.from_mapping, "statements"
        )
        needs = _record_tuple(
            value["information_needs"],
            InformationNeed.from_mapping,
            "information_needs",
        )
        attempts = _record_tuple(
            value["attempted_actions"],
            GroundingActionAttempt.from_mapping,
            "attempted_actions",
        )
        statement_ids = [item.statement_id for item in statements]
        need_ids = [item.need_id for item in needs]
        attempt_ids = [item.attempt_id for item in attempts]
        if len(set(statement_ids)) != len(statement_ids):
            raise GroundingContractError("GroundingSession statement IDs must be unique.")
        if len(set(need_ids)) != len(need_ids):
            raise GroundingContractError("GroundingSession need IDs must be unique.")
        if len(set(attempt_ids)) != len(attempt_ids):
            raise GroundingContractError("GroundingSession attempt IDs must be unique.")
        if len({item.action_key for item in attempts}) != len(attempts):
            raise GroundingContractError(
                "GroundingSession must not repeat an action for one source revision."
            )
        known_statement_ids = set(statement_ids)
        known_need_ids = set(need_ids)
        for need in needs:
            if not set(need.answer_statement_ids).issubset(known_statement_ids):
                raise GroundingContractError(
                    "InformationNeed references an unknown answer statement."
                )
        for attempt in attempts:
            if attempt.need_id not in known_need_ids:
                raise GroundingContractError(
                    "GroundingActionAttempt references an unknown information need."
                )
        decision = GroundingDecision.from_mapping(
            _required_mapping(value["decision"], "GroundingSession.decision")
        )
        if decision.need_id is not None and decision.need_id not in known_need_ids:
            raise GroundingContractError(
                "GroundingDecision references an unknown information need."
            )
        if decision.decision_type == "request_evidence":
            selected_need = next(
                item for item in needs if item.need_id == decision.need_id
            )
            if selected_need.status != "open":
                raise GroundingContractError(
                    "GroundingDecision must request an open information need."
                )
            action_key = (
                decision.need_id,
                decision.provider_id,
                decision.source_revision,
            )
            if action_key in {item.action_key for item in attempts}:
                raise GroundingContractError(
                    "GroundingDecision repeats an attempted provider source revision."
                )
        status = _required_string(value["status"], "GroundingSession.status")
        if status not in _GROUNDING_SESSION_STATUSES:
            raise GroundingContractError("GroundingSession.status is invalid.")
        information_status = _required_string(
            value["information_status"], "GroundingSession.information_status"
        )
        if information_status not in _INFORMATION_STATUSES:
            raise GroundingContractError(
                "GroundingSession.information_status is invalid."
            )
        missing_information = _string_tuple(
            value["missing_information"], "GroundingSession.missing_information"
        )
        expected_missing = tuple(
            need.question
            for need in needs
            if need.status in {"open", "exhausted"}
        )
        if missing_information != expected_missing:
            raise GroundingContractError(
                "GroundingSession.missing_information does not match its needs."
            )
        _validate_session_decision_status(status, decision, needs)
        fingerprint = _sha256_string(
            value["fingerprint"], "GroundingSession.fingerprint"
        )
        payload = dict(value)
        payload.pop("fingerprint")
        if _fingerprint(payload) != fingerprint:
            raise GroundingContractError("GroundingSession fingerprint is invalid.")
        return cls(
            revision=_positive_integer(value["revision"], "GroundingSession.revision"),
            requirement_text=_required_string(
                value["requirement_text"], "GroundingSession.requirement_text"
            ),
            statements=statements,
            information_needs=needs,
            attempted_actions=attempts,
            evidence_refs=_string_tuple(
                value["evidence_refs"], "GroundingSession.evidence_refs"
            ),
            decision=decision,
            status=status,
            missing_information=missing_information,
            information_status=information_status,
            fingerprint=fingerprint,
        )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe grounding-session record."""
        return {
            "schema_version": 1,
            "record_type": "GroundingSession",
            "revision": self.revision,
            "requirement_text": self.requirement_text,
            "statements": [item.to_record() for item in self.statements],
            "information_needs": [item.to_record() for item in self.information_needs],
            "attempted_actions": [item.to_record() for item in self.attempted_actions],
            "evidence_refs": list(self.evidence_refs),
            "decision": self.decision.to_record(),
            "status": self.status,
            "missing_information": list(self.missing_information),
            "information_status": self.information_status,
            "fingerprint": self.fingerprint,
        }


def persist_grounding_session(
    interaction_root: Path,
    session: GroundingSession,
) -> Path:
    """Persist one append-only grounding-session revision."""
    root = Path(interaction_root).resolve()
    path = root / _SESSION_ROOT / f"revision_{session.revision:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(
                session.to_record(),
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
    except FileExistsError as exc:
        raise GroundingContractError(
            f"GroundingSession revision already exists: {session.revision}."
        ) from exc
    return path


def load_latest_grounding_session(interaction_root: Path) -> GroundingSession | None:
    """Load and validate the latest persisted grounding-session revision."""
    root = Path(interaction_root).resolve()
    paths = sorted((root / _SESSION_ROOT).glob("revision_*.json"))
    if not paths:
        return None
    session = GroundingSession.from_mapping(
        _read_json_mapping(paths[-1], "GroundingSession")
    )
    if paths[-1].stem != f"revision_{session.revision:04d}":
        raise GroundingContractError(
            "GroundingSession revision does not match its persisted filename."
        )
    return session


@dataclass(frozen=True)
class TypedContextBinding:
    """Summarize one validated non-RDF product-context record."""

    output_symbol: str
    subject_role: str
    record_type: str
    record_ref: str
    record_sha256: str
    status: str
    frame: str | None
    observed_at_ns: int | None
    valid_from_ns: int | None
    valid_until_ns: int | None
    producer: str
    evidence_refs: tuple[str, ...]
    provenance: Mapping[str, object]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TypedContextBinding:
        """Validate and construct one persisted typed binding summary."""
        expected = {
            "output_symbol",
            "subject_role",
            "record_type",
            "record_ref",
            "record_sha256",
            "status",
            "frame",
            "observed_at_ns",
            "valid_from_ns",
            "valid_until_ns",
            "producer",
            "evidence_refs",
            "provenance",
        }
        _require_exact_keys(value, expected, "TypedContextBinding")
        status = _required_string(value["status"], "TypedContextBinding.status")
        if status not in _BINDING_STATUSES:
            raise GroundingContractError("TypedContextBinding.status is invalid.")
        provenance = _required_mapping(
            value["provenance"], "TypedContextBinding.provenance"
        )
        return cls(
            output_symbol=_required_symbol(
                value["output_symbol"], "TypedContextBinding.output_symbol"
            ),
            subject_role=_required_symbol(
                value["subject_role"], "TypedContextBinding.subject_role"
            ),
            record_type=_required_symbol(
                value["record_type"], "TypedContextBinding.record_type"
            ),
            record_ref=_required_string(
                value["record_ref"], "TypedContextBinding.record_ref"
            ),
            record_sha256=_sha256_string(
                value["record_sha256"], "TypedContextBinding.record_sha256"
            ),
            status=status,
            frame=_optional_string(value["frame"], "TypedContextBinding.frame"),
            observed_at_ns=_optional_nonnegative_integer(
                value["observed_at_ns"], "TypedContextBinding.observed_at_ns"
            ),
            valid_from_ns=_optional_nonnegative_integer(
                value["valid_from_ns"], "TypedContextBinding.valid_from_ns"
            ),
            valid_until_ns=_optional_nonnegative_integer(
                value["valid_until_ns"], "TypedContextBinding.valid_until_ns"
            ),
            producer=_required_symbol(
                value["producer"], "TypedContextBinding.producer"
            ),
            evidence_refs=_string_tuple(
                value["evidence_refs"], "TypedContextBinding.evidence_refs"
            ),
            provenance=dict(provenance),
        )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe binding record."""
        return {
            "output_symbol": self.output_symbol,
            "subject_role": self.subject_role,
            "record_type": self.record_type,
            "record_ref": self.record_ref,
            "record_sha256": self.record_sha256,
            "status": self.status,
            "frame": self.frame,
            "observed_at_ns": self.observed_at_ns,
            "valid_from_ns": self.valid_from_ns,
            "valid_until_ns": self.valid_until_ns,
            "producer": self.producer,
            "evidence_refs": list(self.evidence_refs),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class ProductContextView:
    """Join the current PA ABox with validated typed-context bindings."""

    product_requirement: str
    interaction_namespace: str
    tbox_fingerprint: str
    abox_fingerprint: str
    delta_count: int
    assertions: tuple[Mapping[str, object], ...]
    typed_bindings: tuple[TypedContextBinding, ...]
    uncertainty: tuple[object, ...]
    unresolved_evidence_needs: tuple[object, ...]
    attempted_evidence: tuple[str, ...]
    assessed_at_ns: int
    fingerprint: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ProductContextView:
        """Validate and reconstruct one persisted view snapshot."""
        expected = {
            "schema_version",
            "record_type",
            "product_requirement",
            "interaction_namespace",
            "tbox_fingerprint",
            "abox_fingerprint",
            "delta_count",
            "assertions",
            "typed_bindings",
            "uncertainty",
            "unresolved_evidence_needs",
            "attempted_evidence",
            "assessed_at_ns",
            "fingerprint",
        }
        _require_exact_keys(value, expected, "ProductContextView")
        if value["schema_version"] != 1 or value["record_type"] != "ProductContextView":
            raise GroundingContractError("ProductContextView identity is invalid.")
        assertions = value["assertions"]
        typed_bindings = value["typed_bindings"]
        uncertainty = value["uncertainty"]
        unresolved = value["unresolved_evidence_needs"]
        if not all(isinstance(item, list) for item in (assertions, typed_bindings, uncertainty, unresolved)):
            raise GroundingContractError("ProductContextView collection fields are invalid.")
        result = cls(
            product_requirement=_required_string(
                value["product_requirement"], "ProductContextView.product_requirement"
            ),
            interaction_namespace=_required_string(
                value["interaction_namespace"], "ProductContextView.interaction_namespace"
            ),
            tbox_fingerprint=_sha256_string(
                value["tbox_fingerprint"], "ProductContextView.tbox_fingerprint"
            ),
            abox_fingerprint=_sha256_string(
                value["abox_fingerprint"], "ProductContextView.abox_fingerprint"
            ),
            delta_count=_nonnegative_integer(
                value["delta_count"], "ProductContextView.delta_count"
            ),
            assertions=tuple(
                dict(_required_mapping(item, "ProductContextView assertion"))
                for item in assertions
            ),
            typed_bindings=tuple(
                TypedContextBinding.from_mapping(
                    _required_mapping(item, "ProductContextView typed binding")
                )
                for item in typed_bindings
            ),
            uncertainty=tuple(uncertainty),
            unresolved_evidence_needs=tuple(unresolved),
            attempted_evidence=_string_tuple(
                value["attempted_evidence"], "ProductContextView.attempted_evidence"
            ),
            assessed_at_ns=_nonnegative_integer(
                value["assessed_at_ns"], "ProductContextView.assessed_at_ns"
            ),
            fingerprint=_sha256_string(
                value["fingerprint"], "ProductContextView.fingerprint"
            ),
        )
        payload = result.to_record()
        payload.pop("fingerprint")
        if _fingerprint(payload) != result.fingerprint:
            raise GroundingContractError("ProductContextView fingerprint is invalid.")
        return result

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe view snapshot."""
        return {
            "schema_version": 1,
            "record_type": "ProductContextView",
            "product_requirement": self.product_requirement,
            "interaction_namespace": self.interaction_namespace,
            "tbox_fingerprint": self.tbox_fingerprint,
            "abox_fingerprint": self.abox_fingerprint,
            "delta_count": self.delta_count,
            "assertions": [dict(item) for item in self.assertions],
            "typed_bindings": [item.to_record() for item in self.typed_bindings],
            "uncertainty": list(self.uncertainty),
            "unresolved_evidence_needs": list(self.unresolved_evidence_needs),
            "attempted_evidence": list(self.attempted_evidence),
            "assessed_at_ns": self.assessed_at_ns,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class TypedGroundingContract:
    """Carry grounded meaning that is not limited to the current ontology."""

    requirement_text: str
    session_ref: str
    session_fingerprint: str
    ontology_projection_ref: str
    statements: tuple[GroundingStatement, ...]
    missing_information: tuple[str, ...]
    information_status: str
    typed_record_refs: tuple[Mapping[str, str], ...]
    source_refs: tuple[Mapping[str, str], ...]
    clarification_refs: tuple[Mapping[str, str], ...]
    unrepresented_statement_ids: tuple[str, ...]
    fingerprint: str

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        *,
        requirement_text: str,
        session_ref: str,
        session_fingerprint: str,
        ontology_projection_ref: str,
        statements: Sequence[GroundingStatement],
        missing_information: Sequence[str],
        information_status: str,
        typed_record_refs: Sequence[Mapping[str, str]],
        source_refs: Sequence[Mapping[str, str]],
        clarification_refs: Sequence[Mapping[str, str]],
        unrepresented_statement_ids: Sequence[str],
    ) -> TypedGroundingContract:
        """Create one fingerprinted typed contract from validated completion inputs."""
        record: dict[str, object] = {
            "schema_version": 1,
            "record_type": "TypedGroundingContract",
            "requirement_text": requirement_text,
            "session_ref": session_ref,
            "session_fingerprint": session_fingerprint,
            "ontology_projection_ref": ontology_projection_ref,
            "statements": [item.to_record() for item in statements],
            "missing_information": list(missing_information),
            "information_status": information_status,
            "typed_record_refs": [dict(item) for item in typed_record_refs],
            "source_refs": [dict(item) for item in source_refs],
            "clarification_refs": [dict(item) for item in clarification_refs],
            "unrepresented_statement_ids": list(unrepresented_statement_ids),
        }
        record["fingerprint"] = _fingerprint(record)
        return cls.from_mapping(record)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TypedGroundingContract:
        """Validate and reconstruct one typed grounding contract."""
        _require_exact_keys(
            value,
            {
                "schema_version",
                "record_type",
                "requirement_text",
                "session_ref",
                "session_fingerprint",
                "ontology_projection_ref",
                "statements",
                "missing_information",
                "information_status",
                "typed_record_refs",
                "source_refs",
                "clarification_refs",
                "unrepresented_statement_ids",
                "fingerprint",
            },
            "TypedGroundingContract",
        )
        if value["schema_version"] != 1 or value["record_type"] != "TypedGroundingContract":
            raise GroundingContractError("TypedGroundingContract identity is invalid.")
        statements = _record_tuple(
            value["statements"], GroundingStatement.from_mapping, "statements"
        )
        statement_ids = {item.statement_id for item in statements}
        unrepresented = _string_tuple(
            value["unrepresented_statement_ids"], "unrepresented_statement_ids"
        )
        if not set(unrepresented).issubset(statement_ids):
            raise GroundingContractError(
                "TypedGroundingContract has an unknown unrepresented statement."
            )
        information_status = _required_string(
            value["information_status"], "information_status"
        )
        if information_status not in _INFORMATION_STATUSES:
            raise GroundingContractError(
                "TypedGroundingContract information_status is invalid."
            )
        fingerprint = _sha256_string(value["fingerprint"], "fingerprint")
        payload = dict(value)
        payload.pop("fingerprint")
        if _fingerprint(payload) != fingerprint:
            raise GroundingContractError("TypedGroundingContract fingerprint is invalid.")
        return cls(
            requirement_text=_required_string(
                value["requirement_text"], "requirement_text"
            ),
            session_ref=_required_string(value["session_ref"], "session_ref"),
            session_fingerprint=_sha256_string(
                value["session_fingerprint"], "session_fingerprint"
            ),
            ontology_projection_ref=_required_string(
                value["ontology_projection_ref"], "ontology_projection_ref"
            ),
            statements=statements,
            missing_information=_string_tuple(
                value["missing_information"], "missing_information"
            ),
            information_status=information_status,
            typed_record_refs=_hashed_ref_tuple(
                value["typed_record_refs"], "typed_record_refs"
            ),
            source_refs=_hashed_ref_tuple(value["source_refs"], "source_refs"),
            clarification_refs=_hashed_ref_tuple(
                value["clarification_refs"], "clarification_refs"
            ),
            unrepresented_statement_ids=unrepresented,
            fingerprint=fingerprint,
        )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe typed grounding contract."""
        return {
            "schema_version": 1,
            "record_type": "TypedGroundingContract",
            "requirement_text": self.requirement_text,
            "session_ref": self.session_ref,
            "session_fingerprint": self.session_fingerprint,
            "ontology_projection_ref": self.ontology_projection_ref,
            "statements": [item.to_record() for item in self.statements],
            "missing_information": list(self.missing_information),
            "information_status": self.information_status,
            "typed_record_refs": [dict(item) for item in self.typed_record_refs],
            "source_refs": [dict(item) for item in self.source_refs],
            "clarification_refs": [dict(item) for item in self.clarification_refs],
            "unrepresented_statement_ids": list(self.unrepresented_statement_ids),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class PAContextGroundingCompletionV2:
    """Pin the official ontology projection and typed grounding contract."""

    product_requirement: str
    completion_turn: int
    decision_ref: str
    grounding_session_ref: str
    grounding_session_sha256: str
    grounding_session_fingerprint: str
    ontology_projection_ref: str
    ontology_projection_sha256: str
    typed_grounding_contract_ref: str
    typed_grounding_contract_sha256: str
    typed_grounding_contract_fingerprint: str
    tbox_fingerprint: str
    abox_fingerprint: str
    typed_context_refs: tuple[Mapping[str, str], ...]
    source_refs: tuple[Mapping[str, str], ...]
    clarification_refs: tuple[Mapping[str, str], ...]
    completed_at_ns: int
    fingerprint: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> PAContextGroundingCompletionV2:
        """Validate and construct a version-2 completion bundle."""
        expected = {
            "schema_version",
            "record_type",
            "status",
            "product_requirement",
            "completion_turn",
            "decision_ref",
            "grounding_session_ref",
            "grounding_session_sha256",
            "grounding_session_fingerprint",
            "ontology_projection_ref",
            "ontology_projection_sha256",
            "typed_grounding_contract_ref",
            "typed_grounding_contract_sha256",
            "typed_grounding_contract_fingerprint",
            "tbox_fingerprint",
            "abox_fingerprint",
            "typed_context_refs",
            "source_refs",
            "clarification_refs",
            "completed_at_ns",
            "fingerprint",
        }
        _require_exact_keys(value, expected, "PAContextGroundingCompletion v2")
        if (
            value["schema_version"] != 2
            or value["record_type"] != "PAContextGroundingCompletion"
            or value["status"] != "context understanding complete"
        ):
            raise GroundingContractError(
                "PAContextGroundingCompletion v2 identity is invalid."
            )
        result = cls(
            product_requirement=_required_string(
                value["product_requirement"], "product_requirement"
            ),
            completion_turn=_positive_integer(
                value["completion_turn"], "completion_turn"
            ),
            decision_ref=_required_string(value["decision_ref"], "decision_ref"),
            grounding_session_ref=_required_string(
                value["grounding_session_ref"], "grounding_session_ref"
            ),
            grounding_session_sha256=_sha256_string(
                value["grounding_session_sha256"], "grounding_session_sha256"
            ),
            grounding_session_fingerprint=_sha256_string(
                value["grounding_session_fingerprint"],
                "grounding_session_fingerprint",
            ),
            ontology_projection_ref=_required_string(
                value["ontology_projection_ref"], "ontology_projection_ref"
            ),
            ontology_projection_sha256=_sha256_string(
                value["ontology_projection_sha256"], "ontology_projection_sha256"
            ),
            typed_grounding_contract_ref=_required_string(
                value["typed_grounding_contract_ref"],
                "typed_grounding_contract_ref",
            ),
            typed_grounding_contract_sha256=_sha256_string(
                value["typed_grounding_contract_sha256"],
                "typed_grounding_contract_sha256",
            ),
            typed_grounding_contract_fingerprint=_sha256_string(
                value["typed_grounding_contract_fingerprint"],
                "typed_grounding_contract_fingerprint",
            ),
            tbox_fingerprint=_sha256_string(
                value["tbox_fingerprint"], "tbox_fingerprint"
            ),
            abox_fingerprint=_sha256_string(
                value["abox_fingerprint"], "abox_fingerprint"
            ),
            typed_context_refs=_hashed_ref_tuple(
                value["typed_context_refs"], "typed_context_refs"
            ),
            source_refs=_hashed_ref_tuple(value["source_refs"], "source_refs"),
            clarification_refs=_hashed_ref_tuple(
                value["clarification_refs"], "clarification_refs"
            ),
            completed_at_ns=_nonnegative_integer(
                value["completed_at_ns"], "completed_at_ns"
            ),
            fingerprint=_sha256_string(value["fingerprint"], "fingerprint"),
        )
        payload = result.to_record()
        payload.pop("fingerprint")
        if _fingerprint(payload) != result.fingerprint:
            raise GroundingContractError(
                "PAContextGroundingCompletion v2 fingerprint is invalid."
            )
        return result

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe version-2 completion bundle."""
        return {
            "schema_version": 2,
            "record_type": "PAContextGroundingCompletion",
            "status": "context understanding complete",
            "product_requirement": self.product_requirement,
            "completion_turn": self.completion_turn,
            "decision_ref": self.decision_ref,
            "grounding_session_ref": self.grounding_session_ref,
            "grounding_session_sha256": self.grounding_session_sha256,
            "grounding_session_fingerprint": self.grounding_session_fingerprint,
            "ontology_projection_ref": self.ontology_projection_ref,
            "ontology_projection_sha256": self.ontology_projection_sha256,
            "typed_grounding_contract_ref": self.typed_grounding_contract_ref,
            "typed_grounding_contract_sha256": self.typed_grounding_contract_sha256,
            "typed_grounding_contract_fingerprint": self.typed_grounding_contract_fingerprint,
            "tbox_fingerprint": self.tbox_fingerprint,
            "abox_fingerprint": self.abox_fingerprint,
            "typed_context_refs": [dict(item) for item in self.typed_context_refs],
            "source_refs": [dict(item) for item in self.source_refs],
            "clarification_refs": [dict(item) for item in self.clarification_refs],
            "completed_at_ns": self.completed_at_ns,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class GroundingProducerDescriptor:
    """Describe one provider's evidence capabilities without routing priority."""

    provider_id: str
    description: str
    accepted_evidence_types: tuple[str, ...]
    produced_record_types: tuple[str, ...]
    prerequisites: Mapping[str, tuple[str, ...]]
    availability: bool
    estimated_cost: int

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> GroundingProducerDescriptor:
        """Validate and construct one producer descriptor."""
        expected = {
            "provider_id",
            "description",
            "accepted_evidence_types",
            "produced_record_types",
            "prerequisites",
            "availability",
            "estimated_cost",
        }
        _require_exact_keys(value, expected, "GroundingProducerDescriptor")
        produced_record_types = _string_tuple(
            value["produced_record_types"], "produced_record_types"
        )
        if not produced_record_types:
            raise GroundingContractError(
                "GroundingProducerDescriptor.produced_record_types must be non-empty."
            )
        prerequisites_value = _required_mapping(
            value["prerequisites"], "prerequisites"
        )
        if set(prerequisites_value) != set(produced_record_types):
            raise GroundingContractError(
                "GroundingProducerDescriptor.prerequisites must cover each output."
            )
        prerequisites = {
            record_type: _string_tuple(
                prerequisites_value[record_type],
                f"prerequisites[{record_type}]",
            )
            for record_type in produced_record_types
        }
        availability = value["availability"]
        if not isinstance(availability, bool):
            raise GroundingContractError(
                "GroundingProducerDescriptor.availability must be a boolean."
            )
        return cls(
            provider_id=_required_symbol(value["provider_id"], "provider_id"),
            description=_required_string(value["description"], "description"),
            accepted_evidence_types=_string_tuple(
                value["accepted_evidence_types"], "accepted_evidence_types"
            ),
            produced_record_types=produced_record_types,
            prerequisites=prerequisites,
            availability=availability,
            estimated_cost=_nonnegative_integer(
                value["estimated_cost"], "estimated_cost"
            ),
        )

    @property
    def producer(self) -> str:
        """Return the provider ID used by existing controlled-tool boundaries."""
        return self.provider_id

    @property
    def evidence_types(self) -> tuple[str, ...]:
        """Return accepted evidence types for the existing serving boundary."""
        return self.accepted_evidence_types

    def supports_record_type(self, record_type: str) -> bool:
        """Return whether this provider can produce the requested typed record."""
        return record_type in self.produced_record_types

    def prerequisites_for(self, record_type: str) -> tuple[str, ...]:
        """Return prerequisites for one advertised output record."""
        return self.prerequisites.get(record_type, ())

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe descriptor record."""
        return {
            "provider_id": self.provider_id,
            "description": self.description,
            "accepted_evidence_types": list(self.accepted_evidence_types),
            "produced_record_types": list(self.produced_record_types),
            "prerequisites": {
                key: list(items) for key, items in self.prerequisites.items()
            },
            "availability": self.availability,
            "estimated_cost": self.estimated_cost,
        }


def build_product_context_view(
    interaction_root: Path,
    abox: ABoxSnapshot,
    *,
    attempted_evidence: Iterable[str],
    assessed_at_ns: int,
) -> ProductContextView:
    """Build a tamper-evident view from the current ABox and accepted deltas."""
    if not isinstance(assessed_at_ns, int) or isinstance(assessed_at_ns, bool):
        raise GroundingContractError("assessed_at_ns must be an integer.")
    root = Path(interaction_root).resolve()
    if abox.interaction_root.resolve() != root:
        raise GroundingContractError("ABox interaction root does not match the view root.")

    assertion_records = tuple(_abox_assertion_records(abox))
    abox_fingerprint = _fingerprint(assertion_records)
    bindings: list[TypedContextBinding] = []
    uncertainty: list[object] = []
    unresolved: list[object] = []
    seen_refs: set[str] = set()
    for delta_number in range(1, abox.delta_count + 1):
        delta = _read_delta(abox, delta_number)
        uncertainty.extend(delta["uncertainty"])
        unresolved.extend(delta["unresolved_evidence_needs"])
        for record_ref in delta["typed_context_refs"]:
            if record_ref in seen_refs:
                continue
            seen_refs.add(record_ref)
            bindings.append(
                _typed_binding_from_ref(
                    root,
                    str(record_ref),
                    producer=str(delta["producer"]),
                )
            )

    attempted = tuple(_required_symbol(item, "attempted_evidence item") for item in attempted_evidence)
    payload = {
        "schema_version": 1,
        "record_type": "ProductContextView",
        "product_requirement": abox.product_requirement,
        "interaction_namespace": abox.namespace,
        "tbox_fingerprint": abox.tbox_fingerprint,
        "abox_fingerprint": abox_fingerprint,
        "delta_count": abox.delta_count,
        "assertions": list(assertion_records),
        "typed_bindings": [item.to_record() for item in bindings],
        "uncertainty": uncertainty,
        "unresolved_evidence_needs": unresolved,
        "attempted_evidence": list(attempted),
        "assessed_at_ns": assessed_at_ns,
    }
    fingerprint = _fingerprint(payload)
    return ProductContextView(
        product_requirement=abox.product_requirement,
        interaction_namespace=abox.namespace,
        tbox_fingerprint=abox.tbox_fingerprint,
        abox_fingerprint=abox_fingerprint,
        delta_count=abox.delta_count,
        assertions=assertion_records,
        typed_bindings=tuple(bindings),
        uncertainty=tuple(uncertainty),
        unresolved_evidence_needs=tuple(unresolved),
        attempted_evidence=attempted,
        assessed_at_ns=assessed_at_ns,
        fingerprint=fingerprint,
    )


def persist_product_context_view(
    interaction_root: Path,
    view: ProductContextView,
) -> Path:
    """Persist one append-only ProductContextView snapshot."""
    root = Path(interaction_root).resolve()
    view_root = root / _VIEW_ROOT
    numbers = [
        int(path.stem.removeprefix("view_"))
        for path in view_root.glob("view_*.json")
        if path.stem.removeprefix("view_").isdigit()
    ]
    view_number = max(numbers, default=-1) + 1
    path = view_root / f"view_{view_number:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(view.to_record(), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise GroundingContractError(
            f"ProductContextView already exists: {path.name}."
        ) from exc
    return path


def persist_pa_context_grounding_completion_v2(  # noqa: PLR0913
    interaction_root: Path,
    *,
    product_requirement: str,
    completion_turn: int,
    decision_ref: str,
    product_context: ProductContextView,
    clarification_refs: Sequence[str],
) -> Path:
    """Persist the official hash-pinned PA grounding completion bundle."""
    root = Path(interaction_root).resolve()
    session = load_latest_grounding_session(root)
    if (
        session is None
        or session.status != "complete"
        or session.requirement_text != product_requirement
    ):
        raise GroundingContractError(
            "PAContextGroundingCompletion v2 requires a complete GroundingSession."
        )
    session_path = root / _SESSION_ROOT / f"revision_{session.revision:04d}.json"
    proposal_paths = sorted(
        (root / "products/grounding/ontology_grounding").glob("proposal_*.json")
    )
    if not proposal_paths:
        raise GroundingContractError(
            "PAContextGroundingCompletion v2 requires an ontology projection."
        )
    proposal_path = proposal_paths[-1]
    proposal = _read_json_mapping(proposal_path, "OntologyGroundingProposal")
    if proposal.get("schema_version") != 2 or proposal.get("status") != "accepted":
        raise GroundingContractError(
            "PAContextGroundingCompletion v2 ontology projection is not accepted."
        )
    output = proposal.get("output")
    proposal_value = (
        output.get("ontology_grounding_proposal")
        if isinstance(output, Mapping)
        else None
    )
    unrepresented = (
        proposal_value.get("unrepresented_statement_ids")
        if isinstance(proposal_value, Mapping)
        else None
    )
    if not isinstance(unrepresented, list):
        raise GroundingContractError(
            "Ontology projection unrepresented statements are invalid."
        )
    typed_refs = tuple(
        {"ref": item.record_ref, "sha256": item.record_sha256}
        for item in product_context.typed_bindings
    )
    source_ids = sorted(
        {
            source
            for statement in session.statements
            for source in statement.sources
        }
    )
    source_refs = tuple(
        {"ref": source, "sha256": _grounding_source_hash(root, source, session)}
        for source in source_ids
    )
    clarification_records = tuple(
        {
            "ref": ref,
            "sha256": _sha256_path(
                _completion_ref_path(root, ref, prefix=("interaction_record",))
            ),
        }
        for ref in clarification_refs
    )
    session_ref = str(session_path.relative_to(root))
    proposal_ref = str(proposal_path.relative_to(root))
    contract = TypedGroundingContract.create(
        requirement_text=product_requirement,
        session_ref=session_ref,
        session_fingerprint=session.fingerprint,
        ontology_projection_ref=proposal_ref,
        statements=session.statements,
        missing_information=session.missing_information,
        information_status=session.information_status,
        typed_record_refs=typed_refs,
        source_refs=source_refs,
        clarification_refs=clarification_records,
        unrepresented_statement_ids=[str(item) for item in unrepresented],
    )
    contract_path = (
        root
        / "products/grounding/completion/typed_grounding_contract_0001.json"
    )
    _write_json_mapping_exclusive(contract_path, contract.to_record())
    payload: dict[str, object] = {
        "schema_version": 2,
        "record_type": "PAContextGroundingCompletion",
        "status": "context understanding complete",
        "product_requirement": product_requirement,
        "completion_turn": completion_turn,
        "decision_ref": decision_ref,
        "grounding_session_ref": session_ref,
        "grounding_session_sha256": _sha256_path(session_path),
        "grounding_session_fingerprint": session.fingerprint,
        "ontology_projection_ref": proposal_ref,
        "ontology_projection_sha256": _sha256_path(proposal_path),
        "typed_grounding_contract_ref": str(contract_path.relative_to(root)),
        "typed_grounding_contract_sha256": _sha256_path(contract_path),
        "typed_grounding_contract_fingerprint": contract.fingerprint,
        "tbox_fingerprint": product_context.tbox_fingerprint,
        "abox_fingerprint": product_context.abox_fingerprint,
        "typed_context_refs": [dict(item) for item in typed_refs],
        "source_refs": [dict(item) for item in source_refs],
        "clarification_refs": [dict(item) for item in clarification_records],
        "completed_at_ns": product_context.assessed_at_ns,
    }
    payload["fingerprint"] = _fingerprint(payload)
    completion = PAContextGroundingCompletionV2.from_mapping(payload)
    completion_path = root / "interaction_record/context_completion_0001.json"
    _write_json_mapping_exclusive(completion_path, completion.to_record())
    load_pa_context_grounding_completion(root)
    return completion_path


def load_pa_context_grounding_completion(
    interaction_root: Path,
) -> PAContextGroundingCompletionV2:
    """Load and verify one version-2 completion and all referenced inputs."""
    root = Path(interaction_root).resolve()
    paths = sorted((root / "interaction_record").glob("context_completion_*.json"))
    if len(paths) != 1:
        raise GroundingContractError(
            "Exactly one PAContextGroundingCompletion record is required."
        )
    completion_value = _read_json_mapping(paths[0], "PAContextGroundingCompletion")
    if completion_value.get("schema_version") != 2:
        raise GroundingContractError(
            "Only PAContextGroundingCompletion schema version 2 is supported."
        )
    return _load_pa_context_grounding_completion_v2(root, completion_value)


def _load_pa_context_grounding_completion_v2(
    root: Path,
    value: Mapping[str, object],
) -> PAContextGroundingCompletionV2:
    completion = PAContextGroundingCompletionV2.from_mapping(value)
    decision_path = _completion_ref_path(
        root, completion.decision_ref, prefix=("interaction_record",)
    )
    decision = _read_json_mapping(decision_path, "Phase 4.3 decision")
    output = decision.get("Phase_4_3_output")
    if not isinstance(output, Mapping):
        output = decision.get("PA_output")
    if (
        decision.get("turn") != completion.completion_turn
        or decision.get("product_requirement") != completion.product_requirement
        or decision.get("failure") is not None
        or not isinstance(output, Mapping)
        or output.get("context understanding complete") is not True
        or output.get("grounding_status") != "complete"
    ):
        raise GroundingContractError("Phase 3.5 v2 decision reference is invalid.")

    session_path = _completion_ref_path(
        root,
        completion.grounding_session_ref,
        prefix=("products", "grounding", "session"),
    )
    if _sha256_path(session_path) != completion.grounding_session_sha256:
        raise GroundingContractError("Phase 3.5 v2 GroundingSession hash is invalid.")
    session = GroundingSession.from_mapping(
        _read_json_mapping(session_path, "GroundingSession")
    )
    if (
        session.status != "complete"
        or session.requirement_text != completion.product_requirement
        or session.fingerprint != completion.grounding_session_fingerprint
    ):
        raise GroundingContractError("Phase 3.5 v2 GroundingSession is inconsistent.")

    projection_path = _completion_ref_path(
        root,
        completion.ontology_projection_ref,
        prefix=("products", "grounding", "ontology_grounding"),
    )
    if _sha256_path(projection_path) != completion.ontology_projection_sha256:
        raise GroundingContractError("Phase 3.5 v2 ontology projection hash is invalid.")
    projection = _read_json_mapping(projection_path, "OntologyGroundingProposal")
    if (
        projection.get("schema_version") != 2
        or projection.get("status") != "accepted"
        or projection.get("session_fingerprint")
        not in {session.fingerprint, _previous_session_fingerprint(root, session)}
    ):
        raise GroundingContractError("Phase 3.5 v2 ontology projection is invalid.")

    contract_path = _completion_ref_path(
        root,
        completion.typed_grounding_contract_ref,
        prefix=("products", "grounding", "completion"),
    )
    if _sha256_path(contract_path) != completion.typed_grounding_contract_sha256:
        raise GroundingContractError("Phase 3.5 v2 typed contract hash is invalid.")
    contract = TypedGroundingContract.from_mapping(
        _read_json_mapping(contract_path, "TypedGroundingContract")
    )
    if (
        contract.requirement_text != completion.product_requirement
        or contract.session_ref != completion.grounding_session_ref
        or contract.session_fingerprint != completion.grounding_session_fingerprint
        or contract.ontology_projection_ref != completion.ontology_projection_ref
        or contract.fingerprint != completion.typed_grounding_contract_fingerprint
        or contract.typed_record_refs != completion.typed_context_refs
        or contract.source_refs != completion.source_refs
        or contract.clarification_refs != completion.clarification_refs
    ):
        raise GroundingContractError("Phase 3.5 v2 typed contract is inconsistent.")
    for item in (*completion.typed_context_refs, *completion.clarification_refs):
        ref_path = _completion_ref_path(root, item["ref"], prefix=None)
        if _sha256_path(ref_path) != item["sha256"]:
            raise GroundingContractError("Phase 3.5 v2 pinned record hash is invalid.")
    return completion


def _previous_session_fingerprint(root: Path, session: GroundingSession) -> str:
    if session.revision <= 1:
        return ""
    path = root / _SESSION_ROOT / f"revision_{session.revision - 1:04d}.json"
    if not path.is_file():
        return ""
    return GroundingSession.from_mapping(
        _read_json_mapping(path, "previous GroundingSession")
    ).fingerprint


def _read_delta(abox: ABoxSnapshot, delta_number: int) -> Mapping[str, Any]:
    path = abox.ontology_root / f"delta_{delta_number:04d}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroundingContractError(
            f"Accepted ontology delta could not be read: {path.name}."
        ) from exc
    if (
        not isinstance(value, dict)
        or set(value) != _DELTA_KEYS
        or value["schema_version"] != 1
        or value["delta_number"] != delta_number
        or not isinstance(value["producer"], str)
        or not isinstance(value["uncertainty"], list)
        or not isinstance(value["unresolved_evidence_needs"], list)
        or not isinstance(value["typed_context_refs"], list)
        or not all(isinstance(item, str) for item in value["typed_context_refs"])
    ):
        raise GroundingContractError(f"Accepted ontology delta is invalid: {path.name}.")
    return value


def _typed_binding_from_ref(
    interaction_root: Path,
    record_ref: str,
    *,
    producer: str,
) -> TypedContextBinding:
    relative = Path(record_ref)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[:2] != ("products", "grounding")
    ):
        raise GroundingContractError("Typed context ref leaves products/grounding.")
    path = (interaction_root / relative).resolve()
    try:
        path.relative_to(interaction_root)
    except ValueError as exc:
        raise GroundingContractError("Typed context ref leaves its interaction.") from exc
    try:
        source = path.read_bytes()
        record = json.loads(source.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroundingContractError(
            f"Typed context record could not be read: {record_ref}."
        ) from exc
    if not isinstance(record, Mapping) or record.get("schema_version") != 1:
        raise GroundingContractError(f"Typed context record is invalid: {record_ref}.")
    _validate_embedded_hash_refs(record, interaction_root)

    record_type = _required_symbol(record.get("record_type"), "typed record_type")
    record_producer = _required_symbol(record.get("producer"), "typed producer")
    if record_producer != producer:
        raise GroundingContractError(
            f"Typed context producer mismatch for {record_ref}."
        )
    status = _binding_status(record_type, record)
    if status not in _BINDING_STATUSES:
        raise GroundingContractError(f"Typed context status is invalid: {record_ref}.")
    evidence_refs = _record_evidence_refs(record)
    provenance_value = record.get("provenance", {})
    provenance = (
        dict(provenance_value) if isinstance(provenance_value, Mapping) else {}
    )
    return TypedContextBinding(
        output_symbol=record_type,
        subject_role=_optional_symbol(record.get("subject_role"), "subject_role")
        or "product_context",
        record_type=record_type,
        record_ref=record_ref,
        record_sha256=hashlib.sha256(source).hexdigest(),
        status=status,
        frame=_record_frame(record),
        observed_at_ns=_record_observed_at_ns(record),
        valid_from_ns=_record_optional_integer(record, "valid_from_ns"),
        valid_until_ns=_record_optional_integer(record, "valid_until_ns"),
        producer=record_producer,
        evidence_refs=evidence_refs,
        provenance=provenance,
    )


def _binding_status(record_type: str, record: Mapping[str, object]) -> str:
    if record_type == "DocumentOverviewRecord":
        overview = record.get("overview")
        return "accepted" if isinstance(overview, Mapping) else "rejected"
    if record_type == "DocumentEvidenceRecord":
        observations = record.get("observations")
        return "accepted" if isinstance(observations, list) else "rejected"
    if record_type == "CADSizeCorrespondenceRecord":
        return str(record.get("CAD_correspondence"))
    if record_type == "CADPoseEstimationRecord":
        return str(record.get("pose"))
    if record_type == "RobotFramePoseRecord":
        return str(record.get("robot_frame_conversion"))
    if record_type in {
        "CADMeshRecord",
        "ColoredPointCloudSetRecord",
        "RGBDSegmentationRecord",
        "CameraToRobotCalibrationRecord",
    }:
        return "accepted"
    status = record.get("status")
    return str(status) if isinstance(status, str) else "unavailable"


def _record_frame(record: Mapping[str, object]) -> str | None:
    for key in ("coordinate_frame", "target_frame", "source_frame"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _record_observed_at_ns(record: Mapping[str, object]) -> int | None:
    value = record.get("observation_timestamp_ns")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    cameras = record.get("cameras")
    if isinstance(cameras, list):
        timestamps = [
            camera.get("timestamp_ns")
            for camera in cameras
            if isinstance(camera, Mapping)
            and isinstance(camera.get("timestamp_ns"), int)
            and not isinstance(camera.get("timestamp_ns"), bool)
        ]
        if timestamps:
            return max(int(item) for item in timestamps)
    return None


def _record_optional_integer(record: Mapping[str, object], key: str) -> int | None:
    value = record.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _record_evidence_refs(record: Mapping[str, object]) -> tuple[str, ...]:
    value = record.get("evidence_refs")
    if isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        return tuple(value)
    refs: list[str] = []
    for key in ("document_context_ref", "observation_ref"):
        item = record.get(key)
        if isinstance(item, str) and item:
            refs.append(item)
    cad = record.get("CAD")
    if isinstance(cad, Mapping):
        item = cad.get("context_ref")
        if isinstance(item, str) and item:
            refs.append(item)
    return tuple(refs)


def _validate_embedded_hash_refs(value: object, interaction_root: Path) -> None:
    if isinstance(value, Mapping):
        ref = value.get("ref")
        sha256 = value.get("sha256")
        if isinstance(ref, str) and isinstance(sha256, str):
            relative = Path(ref)
            if relative.is_absolute() or ".." in relative.parts:
                raise GroundingContractError("Typed record artifact ref is invalid.")
            path = (interaction_root / relative).resolve()
            try:
                path.relative_to(interaction_root)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, ValueError) as exc:
                raise GroundingContractError(
                    f"Typed record artifact could not be validated: {ref}."
                ) from exc
            if digest != sha256:
                raise GroundingContractError(
                    f"Typed record artifact hash does not match: {ref}."
                )
        for item in value.values():
            _validate_embedded_hash_refs(item, interaction_root)
    elif isinstance(value, list):
        for item in value:
            _validate_embedded_hash_refs(item, interaction_root)


def _abox_assertion_records(abox: ABoxSnapshot) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for subject, predicate, value in sorted(
        abox.graph, key=lambda triple: tuple(str(node) for node in triple)
    ):
        if isinstance(value, URIRef):
            object_record: dict[str, object] = {"kind": "iri", "value": str(value)}
        elif isinstance(value, Literal):
            object_record = {
                "kind": "literal",
                "value": value.toPython(),
                "datatype": None if value.datatype is None else str(value.datatype),
                "language": value.language,
            }
        else:
            object_record = {"kind": "node", "value": str(value)}
        records.append(
            {
                "subject": str(subject),
                "predicate": str(predicate),
                "object": object_record,
            }
        )
    return records


def _read_json_mapping(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroundingContractError(f"{label} could not be read.") from exc
    if not isinstance(value, Mapping):
        raise GroundingContractError(f"{label} must be an object.")
    return value


def _completion_ref_path(
    interaction_root: Path,
    record_ref: str,
    *,
    prefix: tuple[str, ...] | None,
) -> Path:
    relative = Path(record_ref)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or (prefix is not None and relative.parts[: len(prefix)] != prefix)
    ):
        raise GroundingContractError("Phase 3.5 record ref is outside its authority.")
    path = (interaction_root / relative).resolve()
    try:
        path.relative_to(interaction_root)
    except ValueError as exc:
        raise GroundingContractError("Phase 3.5 record ref leaves its interaction.") from exc
    return path


def _sha256_path(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise GroundingContractError("Phase 3.5 pinned record is unavailable.") from exc


def _grounding_source_hash(
    root: Path,
    source_ref: str,
    session: GroundingSession,
) -> str:
    if source_ref == "requirement_0001":
        return hashlib.sha256(session.requirement_text.encode("utf-8")).hexdigest()
    relative = Path(source_ref)
    if not relative.is_absolute() and ".." not in relative.parts:
        local_path = (root / relative).resolve()
        try:
            local_path.relative_to(root)
        except ValueError:
            local_path = root / "missing"
        if local_path.is_file():
            return _sha256_path(local_path)
    context_ref = source_ref.split("#page=", 1)[0]
    try:
        from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
            approved_cad_path,
            approved_document_metadata,
        )

        if context_ref.lower().endswith(".pdf"):
            metadata = approved_document_metadata(context_ref)
            digest = metadata.get("source_sha256")
            if isinstance(digest, str):
                return _sha256_string(digest, "approved document source hash")
        if context_ref.lower().endswith(".stl"):
            return _sha256_path(approved_cad_path(context_ref))
    except (OSError, TypeError, ValueError):
        pass
    for attempt in session.attempted_actions:
        if attempt.source_ref == source_ref:
            return hashlib.sha256(attempt.source_revision.encode("utf-8")).hexdigest()
    return hashlib.sha256(source_ref.encode("utf-8")).hexdigest()


def _write_json_mapping_exclusive(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise GroundingContractError(
            f"Grounding completion record already exists: {path.name}."
        ) from exc


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_tuple(
    value: object,
    constructor: Any,
    label: str,
) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise GroundingContractError(f"{label} must be a list.")
    return tuple(
        constructor(_required_mapping(item, f"{label} item")) for item in value
    )


def _hashed_ref_tuple(
    value: object,
    label: str,
) -> tuple[Mapping[str, str], ...]:
    if not isinstance(value, list):
        raise GroundingContractError(f"{label} must be a list.")
    result: list[Mapping[str, str]] = []
    for item in value:
        record = _required_mapping(item, f"{label} item")
        _require_exact_keys(record, {"ref", "sha256"}, f"{label} item")
        result.append(
            {
                "ref": _required_string(record["ref"], f"{label} ref"),
                "sha256": _sha256_string(
                    record["sha256"], f"{label} sha256"
                ),
            }
        )
    refs = [item["ref"] for item in result]
    if len(set(refs)) != len(refs):
        raise GroundingContractError(f"{label} must not contain duplicate refs.")
    return tuple(result)


def _validate_session_decision_status(
    status: str,
    decision: GroundingDecision,
    needs: Sequence[InformationNeed],
) -> None:
    open_required = any(item.required and item.status == "open" for item in needs)
    unresolved_required = any(
        item.required and item.status in {"open", "exhausted"} for item in needs
    )
    if decision.decision_type == "request_evidence":
        if status not in {"waiting_for_evidence", "waiting_for_user"}:
            raise GroundingContractError(
                "An evidence request requires a waiting GroundingSession status."
            )
        return
    if decision.decision_type == "ready_for_ontology":
        if status not in {"ready_for_ontology", "complete"} or unresolved_required:
            raise GroundingContractError(
                "ready_for_ontology requires every required need to be resolved."
            )
        return
    if status not in {"incomplete", "ontology_gap"}:
        raise GroundingContractError(
            "An incomplete decision requires an incomplete session status."
        )
    if status == "incomplete" and not unresolved_required:
        raise GroundingContractError(
            "An incomplete GroundingSession requires unresolved information."
        )
    if status == "ontology_gap" and open_required:
        raise GroundingContractError(
            "An ontology_gap cannot retain an open required information need."
        )


def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GroundingContractError(f"{label} must be an object.")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise GroundingContractError(f"{label} fields are invalid.")


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GroundingContractError(f"{label} must be a non-empty string.")
    return value


def _required_symbol(value: object, label: str) -> str:
    result = _required_string(value, label)
    if not _SAFE_SYMBOL.fullmatch(result):
        raise GroundingContractError(f"{label} contains control characters.")
    return result


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, label)


def _optional_symbol(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _required_symbol(value, label)


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise GroundingContractError(f"{label} must be a positive integer.")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise GroundingContractError(f"{label} must be a nonnegative integer.")
    return value


def _optional_nonnegative_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, label)


def _sha256_string(value: object, label: str) -> str:
    result = _required_string(value, label)
    if not re.fullmatch(r"[0-9a-f]{64}", result):
        raise GroundingContractError(f"{label} must be a lowercase SHA-256 value.")
    return result


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise GroundingContractError(f"{label} must be a list.")
    result = tuple(_required_symbol(item, f"{label} item") for item in value)
    if len(set(result)) != len(result):
        raise GroundingContractError(f"{label} must not contain duplicates.")
    return result

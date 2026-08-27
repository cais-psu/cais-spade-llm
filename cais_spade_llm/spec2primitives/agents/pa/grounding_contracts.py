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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rdflib import Literal, URIRef
from rdflib.namespace import RDF

from cais_spade_llm.spec2primitives.agents.pa.product_context import ABoxSnapshot

_CONTEXT_NEED_KINDS = frozenset(
    {"class", "property", "typed_context_record", "user_intent"}
)
_BINDING_STATUSES = frozenset(
    {"accepted", "ambiguous", "rejected", "stale", "unavailable"}
)
_SAFE_SYMBOL = re.compile(r"^[^\x00-\x1f\x7f]+$")
_VIEW_ROOT = Path("products/grounding/product_context")
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
class ContextNeed:
    """Describe one currently required semantic or typed PA-owned input."""

    kind: str
    symbol: str
    subject_role: str
    authority: str
    frame: str | None
    maximum_age_ns: int | None
    reason: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ContextNeed:
        """Validate and construct one exact context need."""
        expected = {
            "kind",
            "symbol",
            "subject_role",
            "authority",
            "frame",
            "maximum_age_ns",
            "reason",
        }
        _require_exact_keys(value, expected, "ContextNeed")
        kind = _required_string(value["kind"], "ContextNeed.kind")
        if kind not in _CONTEXT_NEED_KINDS:
            raise GroundingContractError("ContextNeed.kind is invalid.")
        authority = _required_string(value["authority"], "ContextNeed.authority")
        if authority != "PA":
            raise GroundingContractError("Pre-RA ContextNeed.authority must be PA.")
        frame = _optional_string(value["frame"], "ContextNeed.frame")
        maximum_age_ns = _optional_nonnegative_integer(
            value["maximum_age_ns"], "ContextNeed.maximum_age_ns"
        )
        if kind != "typed_context_record" and (
            frame is not None or maximum_age_ns is not None
        ):
            raise GroundingContractError(
                "Only a typed_context_record ContextNeed may constrain frame or age."
            )
        return cls(
            kind=kind,
            symbol=_required_symbol(value["symbol"], "ContextNeed.symbol"),
            subject_role=_required_symbol(
                value["subject_role"], "ContextNeed.subject_role"
            ),
            authority=authority,
            frame=frame,
            maximum_age_ns=maximum_age_ns,
            reason=_required_string(value["reason"], "ContextNeed.reason"),
        )

    @property
    def key(self) -> str:
        """Return a stable fingerprint used for routing and progress checks."""
        return _fingerprint(self.to_record())

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe contract record."""
        return {
            "kind": self.kind,
            "symbol": self.symbol,
            "subject_role": self.subject_role,
            "authority": self.authority,
            "frame": self.frame,
            "maximum_age_ns": self.maximum_age_ns,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TaskTransitionDraft:
    """Hold PA's evolving robot-independent task grounding requirements."""

    version: int
    product_requirement: str
    requested_process: str | None
    required_outcome: str | None
    required_inputs: tuple[ContextNeed, ...]
    unresolved_user_intent: str | None
    source_view_fingerprint: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TaskTransitionDraft:
        """Validate and construct one task-transition draft."""
        expected = {
            "version",
            "product_requirement",
            "requested_process",
            "required_outcome",
            "required_inputs",
            "unresolved_user_intent",
            "source_view_fingerprint",
        }
        _require_exact_keys(value, expected, "TaskTransitionDraft")
        inputs_value = value["required_inputs"]
        if not isinstance(inputs_value, list):
            raise GroundingContractError(
                "TaskTransitionDraft.required_inputs must be a list."
            )
        inputs = tuple(
            ContextNeed.from_mapping(_required_mapping(item, "required_inputs item"))
            for item in inputs_value
        )
        if len({item.key for item in inputs}) != len(inputs):
            raise GroundingContractError(
                "TaskTransitionDraft.required_inputs must not contain duplicates."
            )
        return cls(
            version=_positive_integer(value["version"], "TaskTransitionDraft.version"),
            product_requirement=_required_string(
                value["product_requirement"], "TaskTransitionDraft.product_requirement"
            ),
            requested_process=_optional_symbol(
                value["requested_process"], "TaskTransitionDraft.requested_process"
            ),
            required_outcome=_optional_string(
                value["required_outcome"], "TaskTransitionDraft.required_outcome"
            ),
            required_inputs=inputs,
            unresolved_user_intent=_optional_string(
                value["unresolved_user_intent"],
                "TaskTransitionDraft.unresolved_user_intent",
            ),
            source_view_fingerprint=_sha256_string(
                value["source_view_fingerprint"],
                "TaskTransitionDraft.source_view_fingerprint",
            ),
        )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe contract record."""
        return {
            "version": self.version,
            "product_requirement": self.product_requirement,
            "requested_process": self.requested_process,
            "required_outcome": self.required_outcome,
            "required_inputs": [item.to_record() for item in self.required_inputs],
            "unresolved_user_intent": self.unresolved_user_intent,
            "source_view_fingerprint": self.source_view_fingerprint,
        }


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
class PAContextGroundingCompletion:
    """Prove that the current pre-RA TaskTransitionDraft inputs are grounded."""

    product_requirement: str
    completion_turn: int
    decision_ref: str
    task_transition_draft_ref: str
    task_transition_draft_sha256: str
    product_context_ref: str
    product_context_fingerprint: str
    tbox_fingerprint: str
    abox_fingerprint: str
    attempted_evidence: tuple[str, ...]
    typed_context_refs: tuple[Mapping[str, str], ...]
    clarification_refs: tuple[str, ...]
    completed_at_ns: int
    fingerprint: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> PAContextGroundingCompletion:
        """Validate and construct one exact Phase 3.5 completion record."""
        expected = {
            "schema_version",
            "record_type",
            "status",
            "product_requirement",
            "completion_turn",
            "decision_ref",
            "task_transition_draft_ref",
            "task_transition_draft_sha256",
            "product_context_ref",
            "product_context_fingerprint",
            "tbox_fingerprint",
            "abox_fingerprint",
            "attempted_evidence",
            "typed_context_refs",
            "clarification_refs",
            "unresolved_context_needs",
            "completed_at_ns",
            "fingerprint",
        }
        _require_exact_keys(value, expected, "PAContextGroundingCompletion")
        if (
            value["schema_version"] != 1
            or value["record_type"] != "PAContextGroundingCompletion"
            or value["status"] != "context understanding complete"
        ):
            raise GroundingContractError(
                "PAContextGroundingCompletion identity is invalid."
            )
        unresolved = value["unresolved_context_needs"]
        if unresolved != []:
            raise GroundingContractError(
                "PAContextGroundingCompletion must have no unresolved context needs."
            )
        typed_value = value["typed_context_refs"]
        if not isinstance(typed_value, list):
            raise GroundingContractError(
                "PAContextGroundingCompletion.typed_context_refs must be a list."
            )
        typed_refs: list[Mapping[str, str]] = []
        for item in typed_value:
            binding = _required_mapping(item, "typed_context_refs item")
            _require_exact_keys(binding, {"ref", "sha256"}, "typed_context_refs item")
            typed_refs.append(
                {
                    "ref": _required_string(binding["ref"], "typed context ref"),
                    "sha256": _sha256_string(
                        binding["sha256"], "typed context sha256"
                    ),
                }
            )
        result = cls(
            product_requirement=_required_string(
                value["product_requirement"],
                "PAContextGroundingCompletion.product_requirement",
            ),
            completion_turn=_positive_integer(
                value["completion_turn"],
                "PAContextGroundingCompletion.completion_turn",
            ),
            decision_ref=_required_string(value["decision_ref"], "decision_ref"),
            task_transition_draft_ref=_required_string(
                value["task_transition_draft_ref"], "task_transition_draft_ref"
            ),
            task_transition_draft_sha256=_sha256_string(
                value["task_transition_draft_sha256"],
                "task_transition_draft_sha256",
            ),
            product_context_ref=_required_string(
                value["product_context_ref"], "product_context_ref"
            ),
            product_context_fingerprint=_sha256_string(
                value["product_context_fingerprint"],
                "product_context_fingerprint",
            ),
            tbox_fingerprint=_sha256_string(
                value["tbox_fingerprint"], "tbox_fingerprint"
            ),
            abox_fingerprint=_sha256_string(
                value["abox_fingerprint"], "abox_fingerprint"
            ),
            attempted_evidence=_string_tuple(
                value["attempted_evidence"], "attempted_evidence"
            ),
            typed_context_refs=tuple(typed_refs),
            clarification_refs=_string_tuple(
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
                "PAContextGroundingCompletion fingerprint is invalid."
            )
        return result

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe Phase 3.5 completion record."""
        return {
            "schema_version": 1,
            "record_type": "PAContextGroundingCompletion",
            "status": "context understanding complete",
            "product_requirement": self.product_requirement,
            "completion_turn": self.completion_turn,
            "decision_ref": self.decision_ref,
            "task_transition_draft_ref": self.task_transition_draft_ref,
            "task_transition_draft_sha256": self.task_transition_draft_sha256,
            "product_context_ref": self.product_context_ref,
            "product_context_fingerprint": self.product_context_fingerprint,
            "tbox_fingerprint": self.tbox_fingerprint,
            "abox_fingerprint": self.abox_fingerprint,
            "attempted_evidence": list(self.attempted_evidence),
            "typed_context_refs": [dict(item) for item in self.typed_context_refs],
            "clarification_refs": list(self.clarification_refs),
            "unresolved_context_needs": [],
            "completed_at_ns": self.completed_at_ns,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class GroundingProducerDescriptor:
    """Advertise which exact outputs one authorized producer can establish."""

    producer: str
    supported_outputs: tuple[tuple[str, str], ...]
    evidence_types: tuple[str, ...]
    required_record_types: tuple[str, ...]
    priority: int

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> GroundingProducerDescriptor:
        """Validate and construct one producer descriptor."""
        expected = {
            "producer",
            "supported_outputs",
            "evidence_types",
            "required_record_types",
            "priority",
        }
        _require_exact_keys(value, expected, "GroundingProducerDescriptor")
        outputs_value = value["supported_outputs"]
        if not isinstance(outputs_value, list) or not outputs_value:
            raise GroundingContractError(
                "GroundingProducerDescriptor.supported_outputs must be non-empty."
            )
        outputs: list[tuple[str, str]] = []
        for item in outputs_value:
            output = _required_mapping(item, "supported_outputs item")
            _require_exact_keys(output, {"kind", "symbol"}, "supported output")
            kind = _required_string(output["kind"], "supported output kind")
            if kind not in _CONTEXT_NEED_KINDS - {"user_intent"}:
                raise GroundingContractError("Supported output kind is invalid.")
            outputs.append(
                (kind, _required_symbol(output["symbol"], "supported output symbol"))
            )
        if len(set(outputs)) != len(outputs):
            raise GroundingContractError("Supported outputs must not contain duplicates.")
        return cls(
            producer=_required_symbol(value["producer"], "producer"),
            supported_outputs=tuple(outputs),
            evidence_types=_string_tuple(value["evidence_types"], "evidence_types"),
            required_record_types=_string_tuple(
                value["required_record_types"], "required_record_types"
            ),
            priority=_nonnegative_integer(value["priority"], "priority"),
        )

    def supports(self, need: ContextNeed) -> bool:
        """Return whether this descriptor advertises the need's exact output."""
        return (need.kind, need.symbol) in self.supported_outputs

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe descriptor record."""
        return {
            "producer": self.producer,
            "supported_outputs": [
                {"kind": kind, "symbol": symbol}
                for kind, symbol in self.supported_outputs
            ],
            "evidence_types": list(self.evidence_types),
            "required_record_types": list(self.required_record_types),
            "priority": self.priority,
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


def load_pa_context_grounding_completion(
    interaction_root: Path,
) -> PAContextGroundingCompletion:
    """Load and verify the complete Phase 3.5 record and all referenced inputs."""
    root = Path(interaction_root).resolve()
    paths = sorted((root / "interaction_record").glob("context_completion_*.json"))
    if len(paths) != 1:
        raise GroundingContractError(
            "Exactly one PAContextGroundingCompletion record is required."
        )
    completion_value = _read_json_mapping(paths[0], "PAContextGroundingCompletion")
    completion = PAContextGroundingCompletion.from_mapping(completion_value)

    decision_path = _completion_ref_path(
        root,
        completion.decision_ref,
        prefix=("interaction_record",),
    )
    decision = _read_json_mapping(decision_path, "Phase 4.3 decision")
    output = decision.get("Phase_4_3_output")
    if (
        decision.get("turn") != completion.completion_turn
        or decision.get("product_requirement") != completion.product_requirement
        or decision.get("failure") is not None
        or not isinstance(output, Mapping)
        or output.get("context understanding complete") is not True
        or output.get("needed_context") is not None
        or output.get("unresolved_semantic_need") is not None
    ):
        raise GroundingContractError("Phase 3.5 decision reference is invalid.")

    draft_path = _completion_ref_path(
        root,
        completion.task_transition_draft_ref,
        prefix=("products", "grounding", "task_transition"),
    )
    draft_source = draft_path.read_bytes()
    if hashlib.sha256(draft_source).hexdigest() != completion.task_transition_draft_sha256:
        raise GroundingContractError("Phase 3.5 TaskTransitionDraft hash is invalid.")
    draft = TaskTransitionDraft.from_mapping(
        _read_json_mapping(draft_path, "TaskTransitionDraft")
    )
    if draft.product_requirement != completion.product_requirement:
        raise GroundingContractError("Phase 3.5 TaskTransitionDraft is inconsistent.")

    view_path = _completion_ref_path(
        root,
        completion.product_context_ref,
        prefix=("products", "grounding", "product_context"),
    )
    view = ProductContextView.from_mapping(
        _read_json_mapping(view_path, "ProductContextView")
    )
    if (
        view.product_requirement != completion.product_requirement
        or view.fingerprint != completion.product_context_fingerprint
        or view.tbox_fingerprint != completion.tbox_fingerprint
        or view.abox_fingerprint != completion.abox_fingerprint
        or view.attempted_evidence != completion.attempted_evidence
    ):
        raise GroundingContractError("Phase 3.5 ProductContextView is inconsistent.")
    source_views = [
        ProductContextView.from_mapping(
            _read_json_mapping(path, "TaskTransitionDraft source ProductContextView")
        )
        for path in sorted(view_path.parent.glob("view_*.json"))
    ]
    if not any(item.fingerprint == draft.source_view_fingerprint for item in source_views):
        raise GroundingContractError(
            "Phase 3.5 TaskTransitionDraft source ProductContextView is missing."
        )
    completion_draft = replace(draft, source_view_fingerprint=view.fingerprint)
    if unresolved_context_needs(completion_draft, view):
        raise GroundingContractError("Phase 3.5 still has unresolved context needs.")

    binding_refs = {
        binding.record_ref: binding.record_sha256 for binding in view.typed_bindings
    }
    if binding_refs != {
        item["ref"]: item["sha256"] for item in completion.typed_context_refs
    }:
        raise GroundingContractError("Phase 3.5 typed context refs are inconsistent.")
    for record_ref, expected_hash in binding_refs.items():
        record_path = _completion_ref_path(
            root,
            record_ref,
            prefix=("products", "grounding"),
        )
        if hashlib.sha256(record_path.read_bytes()).hexdigest() != expected_hash:
            raise GroundingContractError("Phase 3.5 typed context hash is invalid.")

    for clarification_ref in completion.clarification_refs:
        clarification_path = _completion_ref_path(
            root,
            clarification_ref,
            prefix=("interaction_record",),
        )
        clarification = _read_json_mapping(clarification_path, "PAClarification")
        fingerprint = clarification.get("fingerprint")
        payload = {
            key: item for key, item in clarification.items() if key != "fingerprint"
        }
        if (
            clarification.get("record_type") != "PAClarification"
            or clarification.get("action") != "answered"
            or not isinstance(fingerprint, str)
            or fingerprint != _fingerprint(payload)
        ):
            raise GroundingContractError("Phase 3.5 clarification ref is invalid.")
    return completion


def unresolved_context_needs(
    draft: TaskTransitionDraft,
    view: ProductContextView,
) -> tuple[ContextNeed, ...]:
    """Return draft inputs not satisfied by the exact current view."""
    if draft.product_requirement != view.product_requirement:
        raise GroundingContractError(
            "TaskTransitionDraft and ProductContextView requirements do not match."
        )
    if draft.source_view_fingerprint != view.fingerprint:
        raise GroundingContractError(
            "TaskTransitionDraft was not authored from this ProductContextView."
        )
    return tuple(
        need for need in draft.required_inputs if not _need_is_satisfied(need, view)
    )


def select_grounding_producer(
    need: ContextNeed,
    descriptors: Sequence[GroundingProducerDescriptor],
    *,
    attempted_producers: Iterable[str] = (),
) -> GroundingProducerDescriptor:
    """Select the highest-priority untried producer for one exact output."""
    attempted = set(attempted_producers)
    candidates = sorted(
        (
            descriptor
            for descriptor in descriptors
            if descriptor.supports(need) and descriptor.producer not in attempted
        ),
        key=lambda descriptor: (descriptor.priority, descriptor.producer),
    )
    if not candidates:
        raise GroundingContractError(
            f"No untried grounding producer supports {need.kind}:{need.symbol}."
        )
    return candidates[0]


def _need_is_satisfied(need: ContextNeed, view: ProductContextView) -> bool:
    if need.kind == "class":
        return any(
            assertion["predicate"] == str(RDF.type)
            and assertion["object"] == {"kind": "iri", "value": need.symbol}
            for assertion in view.assertions
        )
    if need.kind == "property":
        return any(assertion["predicate"] == need.symbol for assertion in view.assertions)
    if need.kind == "user_intent":
        return False
    for binding in view.typed_bindings:
        if (
            binding.output_symbol != need.symbol
            and binding.record_type != need.symbol
        ):
            continue
        if binding.status != "accepted":
            continue
        if need.frame is not None and binding.frame != need.frame:
            continue
        if need.maximum_age_ns is not None:
            if binding.observed_at_ns is None:
                continue
            age = view.assessed_at_ns - binding.observed_at_ns
            if age < 0 or age > need.maximum_age_ns:
                continue
        if binding.valid_from_ns is not None and view.assessed_at_ns < binding.valid_from_ns:
            continue
        if binding.valid_until_ns is not None and view.assessed_at_ns > binding.valid_until_ns:
            continue
        return True
    return False


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

    record_type_value = record.get("record_type")
    if record_type_value is None and record.get("producer") == "document_evidence":
        record_type_value = "DocumentInterpretationRecord"
    record_type = _required_symbol(record_type_value, "typed record_type")
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
    if record_type == "DocumentInterpretationRecord":
        return "accepted" if record.get("failure") is None else "rejected"
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
    prefix: tuple[str, ...],
) -> Path:
    relative = Path(record_ref)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[: len(prefix)] != prefix:
        raise GroundingContractError("Phase 3.5 record ref is outside its authority.")
    path = (interaction_root / relative).resolve()
    try:
        path.relative_to(interaction_root)
    except ValueError as exc:
        raise GroundingContractError("Phase 3.5 record ref leaves its interaction.") from exc
    return path


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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

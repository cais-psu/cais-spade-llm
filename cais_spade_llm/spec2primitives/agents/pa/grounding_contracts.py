"""Define and validate the PA-owned Phase 4.3 grounding contracts.

The contracts in this module describe robot-independent product context.  They
do not select primitive operations, contact a resource, or treat an ontology
graph as an operational completeness oracle.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdflib import Literal, URIRef

from cais_spade_llm.spec2primitives.agents.pa.presentation_records import (
    load_allocation_presentation,
    load_evidence_presentation,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import ABoxSnapshot
from cais_spade_llm.spec2primitives.ontology import (
    PredefinedWorkcellSnapshot,
    ResourceRegistrySnapshot,
    TBoxSnapshot,
    load_predefined_resource_registry,
    load_predefined_workcell,
)
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_cad_path,
    approved_document_metadata,
)

_BINDING_STATUSES = frozenset({"accepted", "ambiguous", "rejected", "stale", "unavailable"})
_TYPED_RECORD_SCHEMA_VERSIONS = {
    "DocumentOverviewRecord": frozenset({1, 2, 3}),
    "ObservationCandidateReview": frozenset({1}),
    "RGBDSegmentationRecord": frozenset({1, 2}),
    "CADSizeCorrespondenceRecord": frozenset({1, 2}),
    "CADPoseEstimationRecord": frozenset({2, 3}),
    "RobotFramePoseRecord": frozenset({2, 3}),
    "RobotFrameLocationRecord": frozenset({1, 2}),
}
_ONTOLOGY_PROPOSAL_SCHEMA_VERSIONS = frozenset({3, 4, 5})
_ACTION_ATTEMPT_STATUSES = frozenset({"accepted", "no_change", "rejected", "unavailable"})
_GROUNDING_ACTIONS = frozenset(
    {"retrieve", "inspect", "propose_grounding", "ask_user", "incomplete"}
)
_GROUNDING_SESSION_STATUSES = frozenset(
    {
        "waiting_for_evidence",
        "waiting_for_user",
        "ready_for_ontology",
        "complete",
        "incomplete",
        "ontology_gap",
    }
)
_SAFE_SYMBOL = re.compile(r"^[^\x00-\x1f\x7f]+$")
_VIEW_ROOT = Path("products/grounding/product_context")
_SESSION_ROOT = Path("products/grounding/session")
_RESOURCE_SELECTION_PREFIX = (
    "products",
    "grounding",
    "resource_selection",
)
_RESOURCE_SELECTION_RECORD_NAME = "resource_selection_record.json"
_RESOURCE_SELECTION_POLICY = "first_reachable_in_predefined_registry_order"
_NATIVE_CONTEXT_SUMMARY_PREFIX = (
    "ProductAgent proposal summary (captured before deterministic typed grounding "
    "and resource assignment):"
)
_NATIVE_CONTEXT_SUMMARY_SUFFIX = (
    "Final typed context records and ResourceSelectionRecord are authoritative for "
    "completion state."
)
_ASSEMBLY_PROCESS_IRI = "https://cais-spade-llm.local/process/assembly"
_PREDEFINED_RESOURCES = (
    ("xarm6", "https://cais-spade-llm.local/resource/xarm6"),
    ("ur5e", "https://cais-spade-llm.local/resource/ur5e"),
)
_RESOURCE_SELECTION_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "selection_number",
        "specification_iri",
        "feature_iri",
        "process_iri",
        "candidate_resource_iris",
        "required_record_type",
        "target_frame",
        "pose_record_ref",
        "pose_record_sha256",
        "observation_timestamp_ns",
        "tbox_fingerprint",
        "registry_fingerprint",
        "workcell_fingerprint",
        "selection_policy",
        "candidate_reach_evidence",
        "selected_resource_symbol",
        "selected_resource_iri",
        "selected_resource_jid",
        "selected_execution_mode",
        "fingerprint",
    }
)
_CANDIDATE_REACH_KEYS = frozenset(
    {
        "resource_symbol",
        "resource_iri",
        "resource_jid",
        "manifest_ref",
        "manifest_sha256",
        "execution_mode",
        "reachable",
        "verdicts",
    }
)
_RESOURCE_SELECTION_V3_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "selection_number",
        "authority",
        "specification_iri",
        "feature_iri",
        "process_iri",
        "current_state_iri",
        "desired_state_iri",
        "candidate_resource_iris",
        "reachability_check_ref",
        "reachability_check_sha256",
        "reachability_check_fingerprint",
        "provisional_resource_symbol",
        "provisional_resource_iri",
        "provisional_resource_jid",
        "provisional_execution_mode",
        "robot_agent_validation_ref",
        "robot_agent_validation_sha256",
        "robot_agent_validation_fingerprint",
        "robot_agent_validation_status",
        "allocation_status",
        "selected_resource_symbol",
        "selected_resource_iri",
        "selected_resource_jid",
        "selected_execution_mode",
        "tbox_fingerprint",
        "registry_fingerprint",
        "workcell_fingerprint",
        "fingerprint",
    }
)
_RESOURCE_SELECTION_V4_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "selection_number",
        "authority",
        "specification_iri",
        "feature_iri",
        "process_symbol",
        "process_iri",
        "current_state_iri",
        "desired_state_iri",
        "candidate_resource_iris",
        "candidate_resource_symbols",
        "current_state_evidence",
        "desired_state_evidence",
        "evidence_presentation_ref",
        "evidence_presentation_sha256",
        "evidence_presentation_fingerprint",
        "allocation_presentation_ref",
        "allocation_presentation_sha256",
        "allocation_presentation_fingerprint",
        "reachability_check_ref",
        "reachability_check_sha256",
        "reachability_check_fingerprint",
        "provisional_resource_symbol",
        "provisional_resource_iri",
        "provisional_resource_jid",
        "provisional_execution_mode",
        "robot_agent_validation_ref",
        "robot_agent_validation_sha256",
        "robot_agent_validation_fingerprint",
        "robot_agent_validation_status",
        "allocation_status",
        "selected_resource_symbol",
        "selected_resource_iri",
        "selected_resource_jid",
        "selected_execution_mode",
        "tbox_fingerprint",
        "registry_fingerprint",
        "workcell_fingerprint",
        "fingerprint",
    }
)
_REACHABILITY_V3_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "check_number",
        "authority",
        "specification_iri",
        "feature_iri",
        "process_symbol",
        "process_iri",
        "resource_symbol",
        "resource_iri",
        "resource_jid",
        "execution_mode",
        "target_frame",
        "manifest_ref",
        "manifest_sha256",
        "allocation_presentation_ref",
        "allocation_presentation_sha256",
        "allocation_presentation_fingerprint",
        "current_state",
        "desired_state",
        "motion_mode",
        "cartesian_targets",
        "request_fingerprint",
        "robot_agent_validation_ref",
        "robot_agent_validation_sha256",
        "robot_agent_validation_fingerprint",
        "phase_results",
        "tbox_fingerprint",
        "registry_fingerprint",
        "workcell_fingerprint",
        "status",
        "fingerprint",
    }
)
_CARTESIAN_STATE_EVIDENCE_KEYS = frozenset(
    {
        "state_name",
        "state_iri",
        "evidence_handle",
        "source_record_type",
        "source_record_ref",
        "source_record_sha256",
        "source_field_path",
        "location_record_ref",
        "location_record_sha256",
        "observation_timestamp_ns",
        "translation_m",
        "cad_correspondence_record",
        "cad_geometry_record",
        "cad_dimensions_m",
        "segmentation_record",
        "calibration_record",
        "support_plane",
    }
)
_CARTESIAN_TARGET_KEYS = frozenset(
    {
        "pick_object_center_m",
        "pick_support_point_m",
        "pick_surface_normal",
        "place_support_point_m",
        "place_surface_normal",
        "place_object_center_m",
        "part_dimensions_m",
        "support_dimensions_m",
        "part_height_m",
        "motion_offsets",
    }
)
_CARTESIAN_MOTION_OFFSET_KEYS = frozenset(
    {
        "pick_approach_height_m",
        "pick_surface_clearance_m",
        "pick_tcp_z_bias_min_m",
        "pick_tcp_z_bias_max_m",
        "transfer_clearance_m",
        "place_approach_height_m",
    }
)
_PLAN_VALIDATION_V3_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "validation_number",
        "validator_authority",
        "process_symbol",
        "process_iri",
        "feature_iri",
        "current_state_iri",
        "desired_state_iri",
        "resource_symbol",
        "resource_iri",
        "resource_jid",
        "execution_mode",
        "motion_mode",
        "moveit_group",
        "end_effector_link",
        "tcp_link",
        "target_frame",
        "cartesian_path_service",
        "validation_scope",
        "checked_constraints",
        "unvalidated_constraints",
        "cartesian_parameters",
        "live_start_pose",
        "ee_to_tcp_transform",
        "waypoints",
        "phases",
        "mode",
        "motion_executed",
        "status",
        "feedback",
        "validated_at_ns",
        "request_fingerprint",
        "fingerprint",
    }
)
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


class _EmbeddedEvidenceStateError(GroundingContractError):
    """Mark an unavailable or hash-changed embedded source as stale evidence."""


@dataclass(frozen=True)
class GroundingNextAction:
    """Hold the one semantic action selected by ProductAgent."""

    action: str
    source_ref: str | None
    question: str | None
    reason: str | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GroundingNextAction:
        """Validate one minimal action variant without controller metadata."""
        action = _required_string(value.get("action"), "GroundingNextAction.action")
        if action not in _GROUNDING_ACTIONS:
            raise GroundingContractError("GroundingNextAction.action is invalid.")
        expected = {
            "retrieve": {"action", "source_ref"},
            "inspect": {"action", "source_ref", "question"},
            "propose_grounding": {"action"},
            "ask_user": {"action", "question"},
            "incomplete": {"action", "reason"},
        }[action]
        _require_exact_keys(value, expected, "GroundingNextAction")
        return cls(
            action=action,
            source_ref=(
                _required_symbol(value["source_ref"], "GroundingNextAction.source_ref")
                if "source_ref" in value
                else None
            ),
            question=(
                _required_string(value["question"], "GroundingNextAction.question")
                if "question" in value
                else None
            ),
            reason=(
                _required_string(value["reason"], "GroundingNextAction.reason")
                if "reason" in value
                else None
            ),
        )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe minimal action variant."""
        record: dict[str, object] = {"action": self.action}
        if self.source_ref is not None:
            record["source_ref"] = self.source_ref
        if self.question is not None:
            record["question"] = self.question
        if self.reason is not None:
            record["reason"] = self.reason
        return record


@dataclass(frozen=True)
class GroundingActionAttempt:
    """Record one host-resolved provider action and its result."""

    attempt_id: str
    action: str
    provider_id: str
    source_ref: str
    source_revision: str
    question: str | None
    status: str
    record_refs: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GroundingActionAttempt:
        """Validate and construct one grounding action attempt."""
        _require_exact_keys(
            value,
            {
                "attempt_id",
                "action",
                "provider_id",
                "source_ref",
                "source_revision",
                "question",
                "status",
                "record_refs",
            },
            "GroundingActionAttempt",
        )
        status = _required_string(value["status"], "GroundingActionAttempt.status")
        if status not in _ACTION_ATTEMPT_STATUSES:
            raise GroundingContractError("GroundingActionAttempt.status is invalid.")
        record_refs = _string_tuple(value["record_refs"], "GroundingActionAttempt.record_refs")
        if status == "accepted" and not record_refs:
            raise GroundingContractError("An accepted GroundingActionAttempt requires record_refs.")
        action = _required_string(value["action"], "GroundingActionAttempt.action")
        if action not in {"retrieve", "inspect", "ask_user"}:
            raise GroundingContractError("GroundingActionAttempt.action is invalid.")
        question = _optional_string(value["question"], "GroundingActionAttempt.question")
        if action in {"inspect", "ask_user"} and question is None:
            raise GroundingContractError(
                "An inspect or ask_user attempt requires its exact question."
            )
        if action == "retrieve" and question is not None:
            raise GroundingContractError("A retrieve attempt must not contain a question.")
        return cls(
            attempt_id=_required_symbol(value["attempt_id"], "GroundingActionAttempt.attempt_id"),
            action=action,
            provider_id=_required_symbol(
                value["provider_id"], "GroundingActionAttempt.provider_id"
            ),
            source_ref=_required_symbol(value["source_ref"], "GroundingActionAttempt.source_ref"),
            source_revision=_required_symbol(
                value["source_revision"], "GroundingActionAttempt.source_revision"
            ),
            question=question,
            status=status,
            record_refs=record_refs,
        )

    @property
    def action_key(self) -> tuple[str, str, str, str, str | None]:
        """Return the exact replay-protection key for this attempt."""
        return (
            self.action,
            self.provider_id,
            self.source_ref,
            self.source_revision,
            self.question,
        )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe action-attempt record."""
        return {
            "attempt_id": self.attempt_id,
            "action": self.action,
            "provider_id": self.provider_id,
            "source_ref": self.source_ref,
            "source_revision": self.source_revision,
            "question": self.question,
            "status": self.status,
            "record_refs": list(self.record_refs),
        }


@dataclass(frozen=True)
class GroundingSession:
    """Hold only the requirement and controller-owned action journal."""

    revision: int
    requirement_text: str
    attempted_actions: tuple[GroundingActionAttempt, ...]
    next_action: GroundingNextAction
    selected_provider_id: str | None
    selected_source_revision: str | None
    status: str
    fingerprint: str

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        *,
        revision: int,
        requirement_text: str,
        attempted_actions: Sequence[GroundingActionAttempt] = (),
        next_action: GroundingNextAction,
        selected_provider_id: str | None = None,
        selected_source_revision: str | None = None,
        status: str,
    ) -> GroundingSession:
        """Create a fingerprinted session from one action and prior attempts."""
        record = {
            "schema_version": 2,
            "record_type": "GroundingSession",
            "revision": revision,
            "requirement_text": requirement_text,
            "attempted_actions": [item.to_record() for item in attempted_actions],
            "next_action": next_action.to_record(),
            "selected_provider_id": selected_provider_id,
            "selected_source_revision": selected_source_revision,
            "status": status,
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
                "attempted_actions",
                "next_action",
                "selected_provider_id",
                "selected_source_revision",
                "status",
                "fingerprint",
            },
            "GroundingSession",
        )
        if value["schema_version"] != 2 or value["record_type"] != "GroundingSession":
            raise GroundingContractError("GroundingSession identity is invalid.")
        attempts = _record_tuple(
            value["attempted_actions"],
            GroundingActionAttempt.from_mapping,
            "attempted_actions",
        )
        attempt_ids = [item.attempt_id for item in attempts]
        if len(set(attempt_ids)) != len(attempt_ids):
            raise GroundingContractError("GroundingSession attempt IDs must be unique.")
        if len({item.action_key for item in attempts}) != len(attempts):
            raise GroundingContractError(
                "GroundingSession must not repeat an action for one source revision."
            )
        next_action = GroundingNextAction.from_mapping(
            _required_mapping(value["next_action"], "GroundingSession.next_action")
        )
        selected_provider_id = _optional_symbol(
            value["selected_provider_id"], "GroundingSession.selected_provider_id"
        )
        selected_source_revision = _optional_symbol(
            value["selected_source_revision"],
            "GroundingSession.selected_source_revision",
        )
        selects_provider = next_action.action in {"retrieve", "inspect", "ask_user"}
        if selects_provider != (
            selected_provider_id is not None and selected_source_revision is not None
        ):
            raise GroundingContractError(
                "GroundingSession provider metadata does not match its next action."
            )
        status = _required_string(value["status"], "GroundingSession.status")
        if status not in _GROUNDING_SESSION_STATUSES:
            raise GroundingContractError("GroundingSession.status is invalid.")
        _validate_session_action_status(status, next_action)
        fingerprint = _sha256_string(value["fingerprint"], "GroundingSession.fingerprint")
        payload = dict(value)
        payload.pop("fingerprint")
        if _fingerprint(payload) != fingerprint:
            raise GroundingContractError("GroundingSession fingerprint is invalid.")
        return cls(
            revision=_positive_integer(value["revision"], "GroundingSession.revision"),
            requirement_text=_required_string(
                value["requirement_text"], "GroundingSession.requirement_text"
            ),
            attempted_actions=attempts,
            next_action=next_action,
            selected_provider_id=selected_provider_id,
            selected_source_revision=selected_source_revision,
            status=status,
            fingerprint=fingerprint,
        )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe grounding-session record."""
        return {
            "schema_version": 2,
            "record_type": "GroundingSession",
            "revision": self.revision,
            "requirement_text": self.requirement_text,
            "attempted_actions": [item.to_record() for item in self.attempted_actions],
            "next_action": self.next_action.to_record(),
            "selected_provider_id": self.selected_provider_id,
            "selected_source_revision": self.selected_source_revision,
            "status": self.status,
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
    session = GroundingSession.from_mapping(_read_json_mapping(paths[-1], "GroundingSession"))
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
        provenance = _required_mapping(value["provenance"], "TypedContextBinding.provenance")
        return cls(
            output_symbol=_required_symbol(
                value["output_symbol"], "TypedContextBinding.output_symbol"
            ),
            subject_role=_required_symbol(
                value["subject_role"], "TypedContextBinding.subject_role"
            ),
            record_type=_required_symbol(value["record_type"], "TypedContextBinding.record_type"),
            record_ref=_required_string(value["record_ref"], "TypedContextBinding.record_ref"),
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
            producer=_required_symbol(value["producer"], "TypedContextBinding.producer"),
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
        if not all(
            isinstance(item, list) for item in (assertions, typed_bindings, uncertainty, unresolved)
        ):
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
                dict(_required_mapping(item, "ProductContextView assertion")) for item in assertions
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
            fingerprint=_sha256_string(value["fingerprint"], "ProductContextView.fingerprint"),
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
    """Carry one final cited context summary beside the ontology projection."""

    requirement_text: str
    session_ref: str
    session_fingerprint: str
    ontology_projection_ref: str
    context_summary: str
    context_evidence_refs: tuple[str, ...]
    missing_information: tuple[str, ...]
    typed_record_refs: tuple[Mapping[str, str], ...]
    source_refs: tuple[Mapping[str, str], ...]
    clarification_refs: tuple[Mapping[str, str], ...]
    fingerprint: str

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        *,
        requirement_text: str,
        session_ref: str,
        session_fingerprint: str,
        ontology_projection_ref: str,
        context_summary: str,
        context_evidence_refs: Sequence[str],
        missing_information: Sequence[str],
        typed_record_refs: Sequence[Mapping[str, str]],
        source_refs: Sequence[Mapping[str, str]],
        clarification_refs: Sequence[Mapping[str, str]],
    ) -> TypedGroundingContract:
        """Create one fingerprinted typed contract from validated completion inputs."""
        record: dict[str, object] = {
            "schema_version": 2,
            "record_type": "TypedGroundingContract",
            "requirement_text": requirement_text,
            "session_ref": session_ref,
            "session_fingerprint": session_fingerprint,
            "ontology_projection_ref": ontology_projection_ref,
            "context_summary": context_summary,
            "context_evidence_refs": list(context_evidence_refs),
            "missing_information": list(missing_information),
            "typed_record_refs": [dict(item) for item in typed_record_refs],
            "source_refs": [dict(item) for item in source_refs],
            "clarification_refs": [dict(item) for item in clarification_refs],
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
                "context_summary",
                "context_evidence_refs",
                "missing_information",
                "typed_record_refs",
                "source_refs",
                "clarification_refs",
                "fingerprint",
            },
            "TypedGroundingContract",
        )
        if value["schema_version"] != 2 or value["record_type"] != "TypedGroundingContract":
            raise GroundingContractError("TypedGroundingContract identity is invalid.")
        context_evidence_refs = _string_tuple(
            value["context_evidence_refs"], "context_evidence_refs"
        )
        if not context_evidence_refs:
            raise GroundingContractError("TypedGroundingContract requires cited context evidence.")
        fingerprint = _sha256_string(value["fingerprint"], "fingerprint")
        payload = dict(value)
        payload.pop("fingerprint")
        if _fingerprint(payload) != fingerprint:
            raise GroundingContractError("TypedGroundingContract fingerprint is invalid.")
        return cls(
            requirement_text=_required_string(value["requirement_text"], "requirement_text"),
            session_ref=_required_string(value["session_ref"], "session_ref"),
            session_fingerprint=_sha256_string(value["session_fingerprint"], "session_fingerprint"),
            ontology_projection_ref=_required_string(
                value["ontology_projection_ref"], "ontology_projection_ref"
            ),
            context_summary=_required_string(value["context_summary"], "context_summary"),
            context_evidence_refs=context_evidence_refs,
            missing_information=_string_tuple(value["missing_information"], "missing_information"),
            typed_record_refs=_hashed_ref_tuple(value["typed_record_refs"], "typed_record_refs"),
            source_refs=_hashed_ref_tuple(value["source_refs"], "source_refs"),
            clarification_refs=_hashed_ref_tuple(value["clarification_refs"], "clarification_refs"),
            fingerprint=fingerprint,
        )

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe typed grounding contract."""
        return {
            "schema_version": 2,
            "record_type": "TypedGroundingContract",
            "requirement_text": self.requirement_text,
            "session_ref": self.session_ref,
            "session_fingerprint": self.session_fingerprint,
            "ontology_projection_ref": self.ontology_projection_ref,
            "context_summary": self.context_summary,
            "context_evidence_refs": list(self.context_evidence_refs),
            "missing_information": list(self.missing_information),
            "typed_record_refs": [dict(item) for item in self.typed_record_refs],
            "source_refs": [dict(item) for item in self.source_refs],
            "clarification_refs": [dict(item) for item in self.clarification_refs],
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
    resource_selection_ref: str
    resource_selection_sha256: str
    resource_assignment_delta_ref: str
    resource_assignment_delta_sha256: str
    tbox_fingerprint: str
    abox_fingerprint: str
    typed_context_refs: tuple[Mapping[str, str], ...]
    source_refs: tuple[Mapping[str, str], ...]
    clarification_refs: tuple[Mapping[str, str], ...]
    completed_at_ns: int
    fingerprint: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PAContextGroundingCompletionV2:
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
            "resource_selection_ref",
            "resource_selection_sha256",
            "resource_assignment_delta_ref",
            "resource_assignment_delta_sha256",
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
            raise GroundingContractError("PAContextGroundingCompletion v2 identity is invalid.")
        result = cls(
            product_requirement=_required_string(
                value["product_requirement"], "product_requirement"
            ),
            completion_turn=_positive_integer(value["completion_turn"], "completion_turn"),
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
            resource_selection_ref=_required_string(
                value["resource_selection_ref"], "resource_selection_ref"
            ),
            resource_selection_sha256=_sha256_string(
                value["resource_selection_sha256"], "resource_selection_sha256"
            ),
            resource_assignment_delta_ref=_required_string(
                value["resource_assignment_delta_ref"],
                "resource_assignment_delta_ref",
            ),
            resource_assignment_delta_sha256=_sha256_string(
                value["resource_assignment_delta_sha256"],
                "resource_assignment_delta_sha256",
            ),
            tbox_fingerprint=_sha256_string(value["tbox_fingerprint"], "tbox_fingerprint"),
            abox_fingerprint=_sha256_string(value["abox_fingerprint"], "abox_fingerprint"),
            typed_context_refs=_hashed_ref_tuple(value["typed_context_refs"], "typed_context_refs"),
            source_refs=_hashed_ref_tuple(value["source_refs"], "source_refs"),
            clarification_refs=_hashed_ref_tuple(value["clarification_refs"], "clarification_refs"),
            completed_at_ns=_nonnegative_integer(value["completed_at_ns"], "completed_at_ns"),
            fingerprint=_sha256_string(value["fingerprint"], "fingerprint"),
        )
        payload = result.to_record()
        payload.pop("fingerprint")
        if _fingerprint(payload) != result.fingerprint:
            raise GroundingContractError("PAContextGroundingCompletion v2 fingerprint is invalid.")
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
            "resource_selection_ref": self.resource_selection_ref,
            "resource_selection_sha256": self.resource_selection_sha256,
            "resource_assignment_delta_ref": self.resource_assignment_delta_ref,
            "resource_assignment_delta_sha256": self.resource_assignment_delta_sha256,
            "tbox_fingerprint": self.tbox_fingerprint,
            "abox_fingerprint": self.abox_fingerprint,
            "typed_context_refs": [dict(item) for item in self.typed_context_refs],
            "source_refs": [dict(item) for item in self.source_refs],
            "clarification_refs": [dict(item) for item in self.clarification_refs],
            "completed_at_ns": self.completed_at_ns,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class PAContextGroundingCompletionV3:
    """Hold one verified native-tool grounding completion record."""

    record: Mapping[str, object]

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe version-3 completion bundle."""
        return dict(self.record)


@dataclass(frozen=True)
class PAContextGroundingCompletionV4:
    """Hold one verified target-feature grounding completion record."""

    record: Mapping[str, object]

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe version-4 completion bundle."""
        return dict(self.record)


@dataclass(frozen=True)
class PAContextGroundingCompletionV5:
    """Hold one verified two-state, PA-allocated completion record."""

    record: Mapping[str, object]

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe version-5 completion bundle."""
        return dict(self.record)


@dataclass(frozen=True)
class PAContextGroundingCompletionV6:
    """Hold one verified process-aware plan-only allocation completion."""

    record: Mapping[str, object]

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe version-6 completion bundle."""
        return dict(self.record)


@dataclass(frozen=True)
class _CompletionAllocationAuthority:
    """Hold candidate resources reconstructed from pinned registry/workcell records."""

    process_symbol: str
    process_iri: str
    candidate_resources: tuple[tuple[str, str, str], ...]
    tbox_fingerprint: str
    registry_fingerprint: str
    workcell_fingerprint: str


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
    def from_mapping(cls, value: Mapping[str, object]) -> GroundingProducerDescriptor:
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
        prerequisites_value = _required_mapping(value["prerequisites"], "prerequisites")
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
            estimated_cost=_nonnegative_integer(value["estimated_cost"], "estimated_cost"),
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
            "prerequisites": {key: list(items) for key, items in self.prerequisites.items()},
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

    attempted = tuple(
        _required_symbol(item, "attempted_evidence item") for item in attempted_evidence
    )
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
        raise GroundingContractError(f"ProductContextView already exists: {path.name}.") from exc
    return path


def _validated_target_feature(  # noqa: C901
    root: Path,
    projection: Mapping[str, object],
    *,
    product_context: ProductContextView,
    selection: Mapping[str, object],
    selection_schema_version: int = 2,
    allocation_authority: _CompletionAllocationAuthority | None = None,
) -> tuple[Mapping[str, object], tuple[str, ...], Mapping[str, object]]:
    """Validate one accepted proposal, review, values, and resource join."""
    proposal_schema_version = projection.get("schema_version")
    if proposal_schema_version not in {6, 7, 8}:
        raise GroundingContractError("OntologyGroundingProposal schema_version is unsupported.")
    proposal_label = f"OntologyGroundingProposal v{proposal_schema_version}"
    proposal_expected = {
        "schema_version",
        "record_type",
        "proposal_number",
        "initialized_specification_iri",
        "feature_iri",
        "output",
        "compiled_delta",
        "semantic_review_ref",
        "semantic_review_sha256",
        "semantic_review_fingerprint",
        "status",
        "failure",
    }
    _require_exact_keys(projection, proposal_expected, proposal_label)
    proposal_number = _positive_integer(projection["proposal_number"], "proposal_number")
    feature_iri = _required_string(projection["feature_iri"], "feature_iri")
    specification_iri = _required_string(
        projection["initialized_specification_iri"],
        "initialized_specification_iri",
    )
    if (
        projection["record_type"] != "OntologyGroundingProposal"
        or projection["status"] != "accepted"
        or projection["failure"] is not None
        or feature_iri != f"{product_context.interaction_namespace}feature_0001"
        or specification_iri != f"{product_context.interaction_namespace}specification_1"
    ):
        raise GroundingContractError(f"{proposal_label} identity is invalid.")
    output = _required_mapping(projection["output"], "proposal output")
    _require_exact_keys(output, {"target_feature"}, "proposal output")
    target_feature = _required_mapping(output["target_feature"], "target_feature")
    state_names = (
        ("desired_state",) if proposal_schema_version == 6 else ("current_state", "desired_state")
    )
    _require_exact_keys(
        target_feature,
        {"required_process", *state_names},
        "target_feature",
    )
    required_process = _required_mapping(target_feature["required_process"], "required_process")
    _require_exact_keys(
        required_process,
        {"process_iri", "evidence_refs"},
        "required_process",
    )
    process_iri = _required_string(required_process["process_iri"], "process_iri")
    process_refs = _direct_evidence_refs(
        required_process["evidence_refs"], "required_process.evidence_refs"
    )
    state_statement_refs: dict[str, tuple[str, ...]] = {}
    nested_evidence = list(process_refs)
    for state_name in state_names:
        statement_refs, state_evidence_refs = _validated_authored_state(
            root,
            state_name,
            target_feature[state_name],
            product_context=product_context,
        )
        state_statement_refs[state_name] = statement_refs
        nested_evidence.extend(state_evidence_refs)
    evidence_refs = tuple(dict.fromkeys(nested_evidence))

    compiled_delta = _required_mapping(projection["compiled_delta"], "compiled_delta")
    _validate_compiled_delta(
        compiled_delta,
        proposal_schema_version=proposal_schema_version,
        feature_iri=feature_iri,
        specification_iri=specification_iri,
        process_iri=process_iri,
        state_statement_refs=state_statement_refs,
        process_refs=process_refs,
        product_context=product_context,
    )
    if selection_schema_version == 2:
        _validate_target_selection_v2(
            selection,
            specification_iri=specification_iri,
            feature_iri=feature_iri,
            process_iri=process_iri,
            product_context=product_context,
        )
    elif selection_schema_version == 3:
        if proposal_schema_version != 7:
            raise GroundingContractError("PA allocation requires an OntologyGroundingProposal v7.")
        _validate_target_selection_v3(
            root,
            selection,
            specification_iri=specification_iri,
            feature_iri=feature_iri,
            process_iri=process_iri,
            product_context=product_context,
        )
    elif selection_schema_version == 4:
        if proposal_schema_version != 8 or allocation_authority is None:
            raise GroundingContractError(
                "Endpoint-motion allocation requires proposal v8 and pinned authorities."
            )
        _validate_target_selection_v4(
            root,
            selection,
            specification_iri=specification_iri,
            feature_iri=feature_iri,
            process_iri=process_iri,
            product_context=product_context,
            allocation_authority=allocation_authority,
        )
    else:
        raise GroundingContractError("Target-feature resource selection version is unsupported.")

    review_ref = _required_string(projection["semantic_review_ref"], "semantic_review_ref")
    review_path = _completion_ref_path(
        root,
        review_ref,
        prefix=("products", "grounding", "target_feature_review"),
    )
    review = _read_json_mapping(review_path, "TargetFeatureSemanticReview")
    review_expected = {
        "schema_version",
        "record_type",
        "review_number",
        "proposal_number",
        "target_feature_fingerprint",
        "evidence_refs",
        "verdict",
        "gap",
        "reviewed_at_ns",
        "fingerprint",
    }
    review_schema_version = 1 if proposal_schema_version == 6 else 2
    review_label = f"TargetFeatureSemanticReview v{review_schema_version}"
    _require_exact_keys(review, review_expected, review_label)
    review_payload = dict(review)
    review_persisted_fingerprint = review_payload.pop("fingerprint", None)
    review_sha256 = _sha256_path(review_path)
    if (
        review.get("schema_version") != review_schema_version
        or review.get("record_type") != "TargetFeatureSemanticReview"
        or review.get("proposal_number") != proposal_number
        or review.get("target_feature_fingerprint") != _fingerprint(target_feature)
        or review.get("evidence_refs") != list(evidence_refs)
        or review.get("verdict") != "complete"
        or review.get("gap") is not None
        or not _is_sha256_value(review_persisted_fingerprint)
        or _fingerprint(review_payload) != review_persisted_fingerprint
        or projection.get("semantic_review_sha256") != review_sha256
        or projection.get("semantic_review_fingerprint") != review_persisted_fingerprint
    ):
        raise GroundingContractError(f"{review_label} is inconsistent.")
    return target_feature, evidence_refs, review


def _validate_target_selection_v2(
    selection: Mapping[str, object],
    *,
    specification_iri: str,
    feature_iri: str,
    process_iri: str,
    product_context: ProductContextView,
) -> None:
    """Validate the read-only legacy location-based selection contract."""
    selection_expected = {
        "schema_version",
        "record_type",
        "selection_number",
        "specification_iri",
        "feature_iri",
        "process_iri",
        "candidate_resource_iris",
        "required_record_type",
        "target_frame",
        "grounding_record_ref",
        "grounding_record_sha256",
        "observation_timestamp_ns",
        "tbox_fingerprint",
        "registry_fingerprint",
        "workcell_fingerprint",
        "selection_policy",
        "candidate_reach_evidence",
        "selected_resource_symbol",
        "selected_resource_iri",
        "selected_resource_jid",
        "selected_execution_mode",
        "fingerprint",
    }
    _require_exact_keys(selection, selection_expected, "ResourceSelectionRecord v2")
    selection_payload = dict(selection)
    selection_fingerprint = selection_payload.pop("fingerprint", None)
    if (
        selection.get("schema_version") != 2
        or selection.get("record_type") != "ResourceSelectionRecord"
        or selection.get("specification_iri") != specification_iri
        or selection.get("feature_iri") != feature_iri
        or selection.get("process_iri") != process_iri
        or selection.get("required_record_type") != "RobotFrameLocationRecord"
        or selection.get("tbox_fingerprint") != product_context.tbox_fingerprint
        or not _is_sha256_value(selection_fingerprint)
        or _fingerprint(selection_payload) != selection_fingerprint
        or not isinstance(selection.get("selected_resource_iri"), str)
        or not selection.get("selected_resource_iri")
    ):
        raise GroundingContractError("Target-feature resource selection is inconsistent.")
    location_matches = [
        binding
        for binding in product_context.typed_bindings
        if binding.record_type == "RobotFrameLocationRecord"
        and binding.status == "accepted"
        and binding.record_ref == selection.get("grounding_record_ref")
        and binding.record_sha256 == selection.get("grounding_record_sha256")
    ]
    if len(location_matches) != 1:
        raise GroundingContractError(
            "Resource selection is not pinned to one accepted current location."
        )


def _validate_target_selection_v3(  # noqa: C901
    root: Path,
    selection: Mapping[str, object],
    *,
    specification_iri: str,
    feature_iri: str,
    process_iri: str,
    product_context: ProductContextView,
) -> None:
    """Validate a PA choice, both reach locations, and exact RA acceptance."""
    _require_exact_keys(
        selection,
        _RESOURCE_SELECTION_V3_KEYS,
        "ResourceSelectionRecord v3",
    )
    payload = dict(selection)
    fingerprint = payload.pop("fingerprint", None)
    current_state_iri = f"{product_context.interaction_namespace}currentstate_0001"
    desired_state_iri = f"{product_context.interaction_namespace}desiredstate_0001"
    selected_fields = (
        selection.get("selected_resource_symbol"),
        selection.get("selected_resource_iri"),
        selection.get("selected_resource_jid"),
        selection.get("selected_execution_mode"),
    )
    provisional_fields = (
        selection.get("provisional_resource_symbol"),
        selection.get("provisional_resource_iri"),
        selection.get("provisional_resource_jid"),
        selection.get("provisional_execution_mode"),
    )
    candidate_resources = selection.get("candidate_resource_iris")
    if (
        selection.get("schema_version") != 3
        or selection.get("record_type") != "ResourceSelectionRecord"
        or selection.get("authority") != "ProductAgent"
        or selection.get("specification_iri") != specification_iri
        or selection.get("feature_iri") != feature_iri
        or selection.get("process_iri") != process_iri
        or selection.get("current_state_iri") != current_state_iri
        or selection.get("desired_state_iri") != desired_state_iri
        or not isinstance(candidate_resources, list)
        or not all(isinstance(item, str) and item for item in candidate_resources)
        or sorted(candidate_resources)
        != sorted(resource_iri for _symbol, resource_iri in _PREDEFINED_RESOURCES)
        or selection.get("robot_agent_validation_status") != "accepted"
        or selection.get("allocation_status") != "accepted"
        or selected_fields != provisional_fields
        or not all(isinstance(item, str) and item for item in selected_fields)
        or selection.get("tbox_fingerprint") != product_context.tbox_fingerprint
        or not _is_sha256_value(selection.get("registry_fingerprint"))
        or not _is_sha256_value(selection.get("workcell_fingerprint"))
        or not _is_sha256_value(fingerprint)
        or _fingerprint(payload) != fingerprint
    ):
        raise GroundingContractError("PA ResourceSelectionRecord v3 is inconsistent.")

    reach_ref = _required_string(
        selection["reachability_check_ref"],
        "reachability_check_ref",
    )
    reach_path = _completion_ref_path(
        root,
        reach_ref,
        prefix=("products", "grounding", "reachability"),
    )
    reach = _read_json_mapping(reach_path, "ReachabilityCheckRecord")
    reach_payload = dict(reach)
    reach_fingerprint = reach_payload.pop("fingerprint", None)
    if (
        _sha256_path(reach_path) != selection.get("reachability_check_sha256")
        or reach_fingerprint != selection.get("reachability_check_fingerprint")
        or not _is_sha256_value(reach_fingerprint)
        or _fingerprint(reach_payload) != reach_fingerprint
        or reach.get("schema_version") != 1
        or reach.get("record_type") != "ReachabilityCheckRecord"
        or reach.get("authority") != "ProductAgent.check_reachability"
        or reach.get("specification_iri") != specification_iri
        or reach.get("feature_iri") != feature_iri
        or reach.get("process_iri") != process_iri
        or reach.get("resource_symbol") != selected_fields[0]
        or reach.get("resource_iri") != selected_fields[1]
        or reach.get("resource_jid") != selected_fields[2]
        or reach.get("execution_mode") != selected_fields[3]
        or reach.get("status") != "accepted"
    ):
        raise GroundingContractError("PA allocation reachability evidence is inconsistent.")
    for state_name, state_iri in (
        ("current_state", current_state_iri),
        ("desired_state", desired_state_iri),
    ):
        state = reach.get(state_name)
        if not isinstance(state, Mapping):
            raise GroundingContractError("PA allocation requires both reachability state records.")
        location_ref = state.get("location_record_ref")
        location_sha256 = state.get("location_record_sha256")
        if (
            state.get("state_name") != state_name
            or state.get("state_iri") != state_iri
            or state.get("reachable") is not True
            or not isinstance(location_ref, str)
            or not _is_sha256_value(location_sha256)
        ):
            raise GroundingContractError("PA allocation state reachability is inconsistent.")
        matches = [
            binding
            for binding in product_context.typed_bindings
            if binding.record_type == "RobotFrameLocationRecord"
            and binding.status == "accepted"
            and binding.record_ref == location_ref
            and binding.record_sha256 == location_sha256
        ]
        if len(matches) != 1:
            raise GroundingContractError(
                "PA allocation is not pinned to both accepted state locations."
            )

    validation_ref = _required_string(
        selection["robot_agent_validation_ref"],
        "robot_agent_validation_ref",
    )
    validation_path = _completion_ref_path(
        root,
        validation_ref,
        prefix=("resources", str(selected_fields[2]), "validation"),
    )
    validation = _read_json_mapping(
        validation_path,
        "PlanOnlyFeasibilityValidationRecord",
    )
    validation_payload = dict(validation)
    validation_fingerprint = validation_payload.pop("fingerprint", None)
    if (
        _sha256_path(validation_path) != selection.get("robot_agent_validation_sha256")
        or validation_fingerprint != selection.get("robot_agent_validation_fingerprint")
        or not _is_sha256_value(validation_fingerprint)
        or _fingerprint(validation_payload) != validation_fingerprint
        or validation.get("schema_version") != 1
        or validation.get("record_type") != "PlanOnlyFeasibilityValidationRecord"
        or validation.get("validator_authority") != selected_fields[2]
        or validation.get("resource_symbol") != selected_fields[0]
        or validation.get("resource_iri") != selected_fields[1]
        or validation.get("resource_jid") != selected_fields[2]
        or validation.get("execution_mode") != selected_fields[3]
        or validation.get("mode") != "plan_only"
        or validation.get("motion_executed") is not False
        or validation.get("status") != "accepted"
        or validation.get("request_fingerprint") != reach_fingerprint
    ):
        raise GroundingContractError("PA allocation RobotAgent validation is inconsistent.")


def _validate_target_selection_v4(  # noqa: C901, PLR0913
    root: Path,
    selection: Mapping[str, object],
    *,
    specification_iri: str,
    feature_iri: str,
    process_iri: str,
    product_context: ProductContextView,
    allocation_authority: _CompletionAllocationAuthority,
) -> None:
    """Validate process-aware selection from pinned dynamic authorities and presentations."""
    _require_exact_keys(selection, _RESOURCE_SELECTION_V4_KEYS, "ResourceSelectionRecord v4")
    payload = dict(selection)
    fingerprint = payload.pop("fingerprint", None)
    current_state_iri = f"{product_context.interaction_namespace}currentstate_0001"
    desired_state_iri = f"{product_context.interaction_namespace}desiredstate_0001"
    selected_fields = tuple(
        selection.get(field_name)
        for field_name in (
            "selected_resource_symbol",
            "selected_resource_iri",
            "selected_resource_jid",
            "selected_execution_mode",
        )
    )
    provisional_fields = tuple(
        selection.get(field_name)
        for field_name in (
            "provisional_resource_symbol",
            "provisional_resource_iri",
            "provisional_resource_jid",
            "provisional_execution_mode",
        )
    )
    expected_symbols = [item[0] for item in allocation_authority.candidate_resources]
    expected_iris = [item[1] for item in allocation_authority.candidate_resources]
    if (
        selection.get("schema_version") != 4
        or selection.get("record_type") != "ResourceSelectionRecord"
        or selection.get("authority") != "ProductAgent"
        or selection.get("specification_iri") != specification_iri
        or selection.get("feature_iri") != feature_iri
        or selection.get("process_symbol") != allocation_authority.process_symbol
        or selection.get("process_iri") != process_iri
        or process_iri != allocation_authority.process_iri
        or selection.get("current_state_iri") != current_state_iri
        or selection.get("desired_state_iri") != desired_state_iri
        or selection.get("candidate_resource_symbols") != expected_symbols
        or selection.get("candidate_resource_iris") != expected_iris
        or selection.get("robot_agent_validation_status") != "accepted"
        or selection.get("allocation_status") != "accepted"
        or selected_fields != provisional_fields
        or selected_fields[:3] not in allocation_authority.candidate_resources
        or not all(isinstance(item, str) and item for item in selected_fields)
        or selection.get("tbox_fingerprint") != allocation_authority.tbox_fingerprint
        or selection.get("tbox_fingerprint") != product_context.tbox_fingerprint
        or selection.get("registry_fingerprint") != allocation_authority.registry_fingerprint
        or selection.get("workcell_fingerprint") != allocation_authority.workcell_fingerprint
        or not _is_sha256_value(fingerprint)
        or _fingerprint(payload) != fingerprint
    ):
        raise GroundingContractError("PA ResourceSelectionRecord v4 is inconsistent.")

    assignments = {
        "current_state": _validated_state_evidence_assignment(
            selection.get("current_state_evidence"),
            state_iri=current_state_iri,
            label="current_state_evidence",
        ),
        "desired_state": _validated_state_evidence_assignment(
            selection.get("desired_state_evidence"),
            state_iri=desired_state_iri,
            label="desired_state_evidence",
        ),
    }
    evidence_presentation = load_evidence_presentation(root)
    allocation_presentation = load_allocation_presentation(root)
    if (
        selection.get("evidence_presentation_ref") != evidence_presentation.record_ref
        or selection.get("evidence_presentation_sha256")
        != _sha256_path(evidence_presentation.record_path)
        or selection.get("evidence_presentation_fingerprint") != evidence_presentation.fingerprint
        or selection.get("allocation_presentation_ref") != allocation_presentation.record_ref
        or selection.get("allocation_presentation_sha256")
        != _sha256_path(allocation_presentation.record_path)
        or selection.get("allocation_presentation_fingerprint")
        != allocation_presentation.fingerprint
        or allocation_presentation.process_symbol != allocation_authority.process_symbol
        or allocation_presentation.process_iri != process_iri
        or allocation_presentation.feature_iri != feature_iri
        or allocation_presentation.current_state_iri != current_state_iri
        or allocation_presentation.desired_state_iri != desired_state_iri
        or {
            (entry.resource_symbol, entry.resource_iri, entry.resource_jid)
            for entry in allocation_presentation.resources
        }
        != set(allocation_authority.candidate_resources)
    ):
        raise GroundingContractError("PA allocation presentation lineage is inconsistent.")
    for assignment in assignments.values():
        entry = allocation_presentation.evidence_for_handle(str(assignment["evidence_handle"]))
        if (
            entry.record_type != assignment["source_record_type"]
            or entry.record_ref != assignment["source_record_ref"]
            or entry.record_sha256 != assignment["source_record_sha256"]
            or entry.field_path != assignment["source_field_path"]
            or not _accepted_typed_binding_matches(
                product_context,
                record_type=entry.record_type,
                record_ref=entry.record_ref,
                record_sha256=entry.record_sha256,
            )
        ):
            raise GroundingContractError("PA state-evidence presentation mapping is inconsistent.")

    reach_ref = _required_string(selection["reachability_check_ref"], "reachability_check_ref")
    reach_path = _completion_ref_path(
        root,
        reach_ref,
        prefix=("products", "grounding", "reachability"),
    )
    reach = _read_json_mapping(reach_path, "ReachabilityCheckRecord")
    reach_schema = reach.get("schema_version")
    if reach_schema not in {2, 3}:
        raise GroundingContractError("ReachabilityCheckRecord schema is unsupported.")
    reach_fingerprint = _validated_fingerprinted_record(
        reach,
        expected_schema=int(reach_schema),
        expected_type="ReachabilityCheckRecord",
        label=f"ReachabilityCheckRecord v{reach_schema}",
    )
    if (
        _sha256_path(reach_path) != selection.get("reachability_check_sha256")
        or reach_fingerprint != selection.get("reachability_check_fingerprint")
        or reach.get("authority") != "ProductAgent.check_reachability"
        or reach.get("specification_iri") != specification_iri
        or reach.get("feature_iri") != feature_iri
        or reach.get("process_symbol") != allocation_authority.process_symbol
        or reach.get("process_iri") != process_iri
        or tuple(
            reach.get(field_name)
            for field_name in (
                "resource_symbol",
                "resource_iri",
                "resource_jid",
                "execution_mode",
            )
        )
        != selected_fields
        or reach.get("allocation_presentation_ref") != allocation_presentation.record_ref
        or reach.get("allocation_presentation_sha256")
        != _sha256_path(allocation_presentation.record_path)
        or reach.get("allocation_presentation_fingerprint") != allocation_presentation.fingerprint
        or reach.get("registry_fingerprint") != allocation_authority.registry_fingerprint
        or reach.get("workcell_fingerprint") != allocation_authority.workcell_fingerprint
        or reach.get("status") != "accepted"
    ):
        raise GroundingContractError("PA reachability lineage is inconsistent.")
    for state_name, assignment in assignments.items():
        if reach_schema == 2:
            _validate_state_reach_v2(
                root,
                state_name=state_name,
                state=reach.get(state_name),
                assignment=assignment,
                product_context=product_context,
            )
        else:
            _validate_state_reach_v3(
                root,
                state_name=state_name,
                state=reach.get(state_name),
                assignment=assignment,
                product_context=product_context,
            )
    if reach_schema == 3:
        _require_exact_keys(reach, _REACHABILITY_V3_KEYS, "ReachabilityCheckRecord v3")
        if (
            reach.get("execution_mode") != "simulation"
            or reach.get("motion_mode") != "cartesian_pick_place"
            or not _is_sha256_value(reach.get("request_fingerprint"))
            or not _valid_cartesian_targets(
                reach.get("cartesian_targets"),
                current_state=reach.get("current_state"),
                desired_state=reach.get("desired_state"),
            )
            or reach.get("robot_agent_validation_ref")
            != selection.get("robot_agent_validation_ref")
            or reach.get("robot_agent_validation_sha256")
            != selection.get("robot_agent_validation_sha256")
            or reach.get("robot_agent_validation_fingerprint")
            != selection.get("robot_agent_validation_fingerprint")
        ):
            raise GroundingContractError("PA reachability v3 lineage is inconsistent.")

    validation_ref = _required_string(
        selection["robot_agent_validation_ref"],
        "robot_agent_validation_ref",
    )
    validation_path = _completion_ref_path(
        root,
        validation_ref,
        prefix=("resources", str(selected_fields[2]), "validation"),
    )
    validation = _read_json_mapping(validation_path, "PlanOnlyFeasibilityValidationRecord")
    expected_validation_schema = 3 if reach_schema == 3 else 2
    validation_fingerprint = _validated_fingerprinted_record(
        validation,
        expected_schema=expected_validation_schema,
        expected_type="PlanOnlyFeasibilityValidationRecord",
        label=f"PlanOnlyFeasibilityValidationRecord v{expected_validation_schema}",
    )
    if (
        _sha256_path(validation_path) != selection.get("robot_agent_validation_sha256")
        or validation_fingerprint != selection.get("robot_agent_validation_fingerprint")
        or validation.get("validator_authority") != selected_fields[2]
        or validation.get("process_symbol") != allocation_authority.process_symbol
        or validation.get("process_iri") != process_iri
        or validation.get("feature_iri") != feature_iri
        or validation.get("current_state_iri") != current_state_iri
        or validation.get("desired_state_iri") != desired_state_iri
        or tuple(
            validation.get(field_name)
            for field_name in (
                "resource_symbol",
                "resource_iri",
                "resource_jid",
                "execution_mode",
            )
        )
        != selected_fields
        or validation.get("mode") != "plan_only"
        or validation.get("motion_executed") is not False
        or validation.get("status") != "accepted"
    ):
        raise GroundingContractError("Plan-only validation lineage is inconsistent.")
    if reach_schema == 2:
        if (
            validation.get("validation_scope") != "endpoint_motion"
            or validation.get("checked_constraints")
            != ["positional_ik", "collision_aware_endpoints", "path_between_endpoints"]
            or validation.get("unvalidated_constraints")
            != [
                "grasping",
                "end_effector_orientation",
                "attached_object_geometry",
                f"{allocation_authority.process_symbol}_tolerance",
                "force_contact",
                "insertion_constraints",
            ]
            or validation.get("request_fingerprint") != reach_fingerprint
            or not _accepted_endpoint_result(validation.get("current_state"))
            or not _accepted_endpoint_result(validation.get("desired_state"))
        ):
            raise GroundingContractError("Endpoint-motion validation v2 is inconsistent.")
    else:
        _validate_cartesian_plan_record_v3(
            validation,
            reach=reach,
            process_symbol=allocation_authority.process_symbol,
        )


def _validated_state_evidence_assignment(
    value: object,
    *,
    state_iri: str,
    label: str,
) -> Mapping[str, str]:
    assignment = _required_mapping(value, label)
    expected = {
        "state_iri",
        "evidence_handle",
        "source_record_type",
        "source_record_ref",
        "source_record_sha256",
        "source_field_path",
    }
    _require_exact_keys(assignment, expected, label)
    result = {key: _required_string(assignment[key], f"{label}.{key}") for key in expected}
    if result["state_iri"] != state_iri or not _is_sha256_value(result["source_record_sha256"]):
        raise GroundingContractError(f"{label} is inconsistent.")
    return result


def _validate_state_reach_v2(
    root: Path,
    *,
    state_name: str,
    state: object,
    assignment: Mapping[str, str],
    product_context: ProductContextView,
) -> None:
    value = _required_mapping(state, f"{state_name} reachability")
    expected = {
        "state_name",
        "state_iri",
        "evidence_handle",
        "source_record_type",
        "source_record_ref",
        "source_record_sha256",
        "source_field_path",
        "location_record_ref",
        "location_record_sha256",
        "observation_timestamp_ns",
        "translation_m",
        "planar_distance_from_reach_origin_m",
        "distance_from_reach_origin_m",
        "in_workspace",
        "in_gripper_reach",
        "reachable",
        "verdicts",
    }
    _require_exact_keys(value, expected, f"{state_name} reachability")
    direct_pairs = {
        "state_iri": "state_iri",
        "evidence_handle": "evidence_handle",
        "source_record_type": "source_record_type",
        "source_record_ref": "source_record_ref",
        "source_record_sha256": "source_record_sha256",
        "source_field_path": "source_field_path",
    }
    location_ref = value.get("location_record_ref")
    location_sha256 = value.get("location_record_sha256")
    if (
        value.get("state_name") != state_name
        or any(value.get(left) != assignment[right] for left, right in direct_pairs.items())
        or value.get("reachable") is not True
        or value.get("in_workspace") is not True
        or value.get("in_gripper_reach") is not True
        or not isinstance(location_ref, str)
        or not _is_sha256_value(location_sha256)
        or _sha256_path(_completion_ref_path(root, location_ref, prefix=None)) != location_sha256
        or not _accepted_typed_binding_matches(
            product_context,
            record_type="RobotFrameLocationRecord",
            record_ref=location_ref,
            record_sha256=str(location_sha256),
        )
    ):
        raise GroundingContractError(f"{state_name} reachability v2 is inconsistent.")


def _validate_state_reach_v3(
    root: Path,
    *,
    state_name: str,
    state: object,
    assignment: Mapping[str, str],
    product_context: ProductContextView,
) -> None:
    value = _required_mapping(state, f"{state_name} Cartesian reachability")
    _require_exact_keys(
        value,
        _CARTESIAN_STATE_EVIDENCE_KEYS,
        f"{state_name} Cartesian reachability",
    )
    direct_pairs = {
        "state_iri": "state_iri",
        "evidence_handle": "evidence_handle",
        "source_record_type": "source_record_type",
        "source_record_ref": "source_record_ref",
        "source_record_sha256": "source_record_sha256",
        "source_field_path": "source_field_path",
    }
    location_ref = value.get("location_record_ref")
    location_sha256 = value.get("location_record_sha256")
    support_plane = value.get("support_plane")
    if (
        value.get("state_name") != state_name
        or any(value.get(left) != assignment[right] for left, right in direct_pairs.items())
        or value.get("source_record_type") != "RGBDSegmentationRecord"
        or not isinstance(location_ref, str)
        or not _is_sha256_value(location_sha256)
        or _sha256_path(_completion_ref_path(root, location_ref, prefix=None)) != location_sha256
        or not _accepted_typed_binding_matches(
            product_context,
            record_type="RobotFrameLocationRecord",
            record_ref=location_ref,
            record_sha256=str(location_sha256),
        )
        or not _finite_vector(value.get("translation_m"), 3)
        or not _finite_vector(value.get("cad_dimensions_m"), 3, positive=True)
        or not isinstance(support_plane, Mapping)
        or set(support_plane) != {"point_m", "normal", "rms_distance_m"}
        or not _finite_vector(support_plane.get("point_m"), 3)
        or not _unit_vector(support_plane.get("normal"))
        or not _finite_nonnegative(support_plane.get("rms_distance_m"))
        or _validate_embedded_hash_refs(value, root) != 4
    ):
        raise GroundingContractError(f"{state_name} Cartesian reachability v3 is inconsistent.")


def _validate_cartesian_plan_record_v3(
    validation: Mapping[str, object],
    *,
    reach: Mapping[str, object],
    process_symbol: str,
) -> None:
    _require_exact_keys(
        validation,
        _PLAN_VALIDATION_V3_KEYS,
        "PlanOnlyFeasibilityValidationRecord v3",
    )
    checked = [
        "live_tf",
        "collision_aware_cartesian_pick_path",
        "collision_aware_cartesian_transfer_place_path",
        "complete_path_fraction",
    ]
    unvalidated = [
        "grasp_contact",
        "gripper_actuation",
        "attached_part_collision_geometry",
        f"{process_symbol}_tolerance",
        "force_control",
        "final_constrained_insertion_stroke",
    ]
    parameters = validation.get("cartesian_parameters")
    phases = validation.get("phases")
    live_start_pose = validation.get("live_start_pose")
    ee_to_tcp = validation.get("ee_to_tcp_transform")
    if (
        validation.get("execution_mode") != "simulation"
        or validation.get("motion_mode") != "cartesian_pick_place"
        or validation.get("validation_scope") != "cartesian_pick_place"
        or validation.get("checked_constraints") != checked
        or validation.get("unvalidated_constraints") != unvalidated
        or validation.get("request_fingerprint") != reach.get("request_fingerprint")
        or validation.get("phases") != reach.get("phase_results")
        or not isinstance(parameters, Mapping)
        or parameters
        != {
            "max_step_m": 0.01,
            "jump_threshold": 0.0,
            "avoid_collisions": True,
            "minimum_fraction": 0.999,
        }
        or not _accepted_cartesian_phases(phases)
        or not _valid_live_pose(live_start_pose)
        or not _valid_ee_to_tcp(ee_to_tcp)
        or not _cartesian_configuration_matches(
            validation,
            reach=reach,
            live_start_pose=live_start_pose,
            ee_to_tcp=ee_to_tcp,
        )
        or not _valid_cartesian_waypoints(validation.get("waypoints"))
    ):
        raise GroundingContractError("Cartesian pick-and-place validation v3 is inconsistent.")


def _accepted_cartesian_phases(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"pick", "place"}:
        return False
    expected_roles = {
        "pick": ["pick_approach", "grasp", "pick_retreat"],
        "place": ["transfer", "place_approach", "placement", "place_retreat"],
    }
    for phase, roles in expected_roles.items():
        result = value.get(phase)
        if (
            not isinstance(result, Mapping)
            or set(result)
            != {
                "phase",
                "status",
                "waypoint_roles",
                "fraction",
                "moveit_error_code",
                "terminal_state_available",
                "message",
            }
            or result.get("phase") != phase
            or result.get("status") != "accepted"
            or result.get("waypoint_roles") != roles
            or not _finite_nonnegative(result.get("fraction"))
            or float(result["fraction"]) < 0.999
            or float(result["fraction"]) > 1.0
            or result.get("moveit_error_code") != 1
            or result.get("terminal_state_available") is not True
            or not isinstance(result.get("message"), str)
            or not result.get("message")
        ):
            return False
    return True


def _valid_cartesian_targets(
    value: object,
    *,
    current_state: object,
    desired_state: object,
) -> bool:
    if (
        not isinstance(value, Mapping)
        or set(value) != _CARTESIAN_TARGET_KEYS
        or not isinstance(current_state, Mapping)
        or not isinstance(desired_state, Mapping)
    ):
        return False
    current_support = current_state.get("support_plane")
    desired_support = desired_state.get("support_plane")
    offsets = value.get("motion_offsets")
    part_height = value.get("part_height_m")
    if (
        not isinstance(current_support, Mapping)
        or not isinstance(desired_support, Mapping)
        or not isinstance(offsets, Mapping)
        or set(offsets) != _CARTESIAN_MOTION_OFFSET_KEYS
        or not _finite_nonnegative(part_height)
        or float(part_height) <= 0.0
        or not all(_finite_nonnegative(item) for item in offsets.values())
        or float(offsets["pick_tcp_z_bias_min_m"]) > float(offsets["pick_tcp_z_bias_max_m"])
    ):
        return False
    current_dimensions = current_state.get("cad_dimensions_m")
    desired_dimensions = desired_state.get("cad_dimensions_m")
    if (
        not _vectors_close(value.get("pick_object_center_m"), current_state.get("translation_m"))
        or not _vectors_close(value.get("pick_support_point_m"), current_support.get("point_m"))
        or not _vectors_close(value.get("pick_surface_normal"), current_support.get("normal"))
        or not _vectors_close(value.get("place_support_point_m"), desired_support.get("point_m"))
        or not _vectors_close(value.get("place_surface_normal"), desired_support.get("normal"))
        or not _vectors_close(value.get("part_dimensions_m"), current_dimensions)
        or not _vectors_close(value.get("support_dimensions_m"), desired_dimensions)
        or not _finite_vector(current_dimensions, 3, positive=True)
        or not math.isclose(
            float(part_height),
            min(float(item) for item in current_dimensions),  # type: ignore[union-attr]
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        return False
    support_point = desired_support.get("point_m")
    support_normal = desired_support.get("normal")
    if not _finite_vector(support_point, 3) or not _unit_vector(support_normal):
        return False
    expected_place_center = [
        float(support_point[index])  # type: ignore[index]
        + float(support_normal[index]) * float(part_height) * 0.5  # type: ignore[index]
        for index in range(3)
    ]
    return _vectors_close(value.get("place_object_center_m"), expected_place_center)


def _cartesian_configuration_matches(
    validation: Mapping[str, object],
    *,
    reach: Mapping[str, object],
    live_start_pose: object,
    ee_to_tcp: object,
) -> bool:
    configured = (
        "moveit_group",
        "end_effector_link",
        "tcp_link",
        "target_frame",
        "cartesian_path_service",
    )
    return (
        all(
            isinstance(validation.get(field), str) and validation.get(field) for field in configured
        )
        and validation.get("target_frame") == reach.get("target_frame")
        and isinstance(live_start_pose, Mapping)
        and live_start_pose.get("frame_id") == validation.get("target_frame")
        and live_start_pose.get("link_name") == validation.get("end_effector_link")
        and isinstance(ee_to_tcp, Mapping)
        and ee_to_tcp.get("parent_link") == validation.get("end_effector_link")
        and ee_to_tcp.get("child_link") == validation.get("tcp_link")
    )


def _vectors_close(left: object, right: object) -> bool:
    return (
        _finite_vector(left, 3)
        and _finite_vector(right, 3)
        and all(
            math.isclose(float(left[index]), float(right[index]), rel_tol=1e-9, abs_tol=1e-9)  # type: ignore[index]
            for index in range(3)
        )
    )


def _valid_live_pose(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"frame_id", "link_name", "position_m", "orientation_xyzw"}
        and isinstance(value.get("frame_id"), str)
        and bool(value.get("frame_id"))
        and isinstance(value.get("link_name"), str)
        and bool(value.get("link_name"))
        and _finite_vector(value.get("position_m"), 3)
        and _unit_quaternion(value.get("orientation_xyzw"))
    )


def _valid_ee_to_tcp(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"parent_link", "child_link", "translation_m", "rotation_xyzw"}
        and isinstance(value.get("parent_link"), str)
        and bool(value.get("parent_link"))
        and isinstance(value.get("child_link"), str)
        and bool(value.get("child_link"))
        and _finite_vector(value.get("translation_m"), 3)
        and _unit_quaternion(value.get("rotation_xyzw"))
    )


def _valid_cartesian_waypoints(value: object) -> bool:
    expected = [
        ("pick", "pick_approach"),
        ("pick", "grasp"),
        ("pick", "pick_retreat"),
        ("place", "transfer"),
        ("place", "place_approach"),
        ("place", "placement"),
        ("place", "place_retreat"),
    ]
    if not isinstance(value, list) or len(value) != len(expected):
        return False
    for item, (phase, role) in zip(value, expected, strict=True):
        pose = item.get("pose") if isinstance(item, Mapping) else None
        if (
            not isinstance(item, Mapping)
            or set(item) != {"phase", "role", "pose"}
            or item.get("phase") != phase
            or item.get("role") != role
            or not isinstance(pose, Mapping)
            or set(pose) != {"position_m", "orientation_xyzw"}
            or not _finite_vector(pose.get("position_m"), 3)
            or not _unit_quaternion(pose.get("orientation_xyzw"))
        ):
            return False
    return True


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _finite_vector(value: object, length: int, *, positive: bool = False) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == length
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            and (not positive or float(item) > 0.0)
            for item in value
        )
    )


def _unit_vector(value: object) -> bool:
    return (
        _finite_vector(value, 3)
        and abs(
            math.sqrt(sum(float(item) ** 2 for item in value)) - 1.0  # type: ignore[union-attr]
        )
        <= 1e-5
    )


def _unit_quaternion(value: object) -> bool:
    return (
        _finite_vector(value, 4)
        and abs(
            math.sqrt(sum(float(item) ** 2 for item in value)) - 1.0  # type: ignore[union-attr]
        )
        <= 1e-5
    )


def _accepted_typed_binding_matches(
    product_context: ProductContextView,
    *,
    record_type: str,
    record_ref: str,
    record_sha256: str,
) -> bool:
    return (
        sum(
            binding.record_type == record_type
            and binding.status == "accepted"
            and binding.record_ref == record_ref
            and binding.record_sha256 == record_sha256
            for binding in product_context.typed_bindings
        )
        == 1
    )


def _validated_fingerprinted_record(
    value: Mapping[str, object],
    *,
    expected_schema: int,
    expected_type: str,
    label: str,
) -> str:
    payload = dict(value)
    fingerprint = payload.pop("fingerprint", None)
    if (
        value.get("schema_version") != expected_schema
        or value.get("record_type") != expected_type
        or not _is_sha256_value(fingerprint)
        or _fingerprint(payload) != fingerprint
    ):
        raise GroundingContractError(f"{label} fingerprint is invalid.")
    return str(fingerprint)


def _accepted_endpoint_result(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"status", "message", "error_code"}
        and value.get("status") == "accepted"
        and isinstance(value.get("message"), str)
        and bool(value.get("message"))
        and (
            value.get("error_code") is None
            or (
                isinstance(value.get("error_code"), int)
                and not isinstance(value.get("error_code"), bool)
            )
        )
    )


def _completion_allocation_authority(
    registry: Mapping[str, object],
    workcell: Mapping[str, object],
    *,
    process_iri: str,
) -> _CompletionAllocationAuthority:
    """Reconstruct capable candidates from hash-pinned registry/workcell snapshots."""
    registry_expected = {
        "schema_version",
        "record_type",
        "ppr_namespace",
        "resource_namespace",
        "tbox_fingerprint",
        "workcell_profile_sha256",
        "graph_fingerprint",
        "resources",
        "fingerprint",
    }
    workcell_expected = {
        "schema_version",
        "record_type",
        "processes",
        "resource_iris",
        "resource_capabilities",
        "ppr_namespace",
        "resource_namespace",
        "tbox_fingerprint",
        "registry_fingerprint",
        "workcell_profile_sha256",
        "graph_fingerprint",
        "fingerprint",
    }
    _require_exact_keys(registry, registry_expected, "ResourceRegistrySnapshot v1")
    _require_exact_keys(workcell, workcell_expected, "PredefinedWorkcellSnapshot v2")
    registry_fingerprint = _validated_fingerprinted_record(
        registry,
        expected_schema=1,
        expected_type="ResourceRegistrySnapshot",
        label="ResourceRegistrySnapshot v1",
    )
    workcell_fingerprint = _validated_fingerprinted_record(
        workcell,
        expected_schema=2,
        expected_type="PredefinedWorkcellSnapshot",
        label="PredefinedWorkcellSnapshot v2",
    )
    tbox_fingerprint = _sha256_string(
        registry.get("tbox_fingerprint"),
        "registry tbox_fingerprint",
    )
    if (
        workcell.get("tbox_fingerprint") != tbox_fingerprint
        or workcell.get("registry_fingerprint") != registry_fingerprint
        or workcell.get("ppr_namespace") != registry.get("ppr_namespace")
        or workcell.get("resource_namespace") != registry.get("resource_namespace")
        or workcell.get("workcell_profile_sha256") != registry.get("workcell_profile_sha256")
    ):
        raise GroundingContractError("Pinned registry/workcell lineage is inconsistent.")

    raw_resources = registry.get("resources")
    if not isinstance(raw_resources, list) or not raw_resources:
        raise GroundingContractError("Pinned registry resources are invalid.")
    resources: list[tuple[str, str, str]] = []
    for item in raw_resources:
        resource = _required_mapping(item, "registry resource")
        _require_exact_keys(
            resource,
            {
                "resource_symbol",
                "resource_iri",
                "resource_jid",
                "resource_type",
                "source_ref",
                "source_sha256",
            },
            "registry resource",
        )
        if resource.get("resource_type") != "robot" or not _is_sha256_value(
            resource.get("source_sha256")
        ):
            raise GroundingContractError("Pinned registry resource is invalid.")
        resources.append(
            (
                _required_string(resource["resource_symbol"], "resource_symbol"),
                _required_string(resource["resource_iri"], "resource_iri"),
                _required_string(resource["resource_jid"], "resource_jid"),
            )
        )
    if len({item[0] for item in resources}) != len(resources) or len(
        {item[1] for item in resources}
    ) != len(resources):
        raise GroundingContractError("Pinned registry resource identities are not unique.")
    if workcell.get("resource_iris") != [item[1] for item in resources]:
        raise GroundingContractError("Pinned workcell resources do not match the registry.")

    raw_processes = workcell.get("processes")
    if not isinstance(raw_processes, list) or not raw_processes:
        raise GroundingContractError("Pinned workcell processes are invalid.")
    processes: list[tuple[str, str]] = []
    for item in raw_processes:
        process = _required_mapping(item, "workcell process")
        _require_exact_keys(
            process,
            {"process_symbol", "process_iri"},
            "workcell process",
        )
        processes.append(
            (
                _required_string(process["process_symbol"], "process_symbol"),
                _required_string(process["process_iri"], "process_iri"),
            )
        )
    process_matches = [item for item in processes if item[1] == process_iri]
    if len(process_matches) != 1:
        raise GroundingContractError("Selected process is not uniquely pinned by the workcell.")

    raw_capabilities = workcell.get("resource_capabilities")
    if not isinstance(raw_capabilities, list):
        raise GroundingContractError("Pinned workcell capabilities are invalid.")
    capabilities: dict[str, tuple[str, ...]] = {}
    for item in raw_capabilities:
        capability = _required_mapping(item, "resource capability")
        _require_exact_keys(
            capability,
            {"resource_iri", "capable_process_iris"},
            "resource capability",
        )
        resource_iri = _required_string(capability["resource_iri"], "resource_iri")
        process_iris = _string_tuple(
            capability["capable_process_iris"],
            "capable_process_iris",
        )
        if resource_iri in capabilities:
            raise GroundingContractError("Pinned workcell capability is duplicated.")
        capabilities[resource_iri] = process_iris
    if set(capabilities) != {item[1] for item in resources}:
        raise GroundingContractError("Pinned workcell capability coverage is invalid.")
    candidates = tuple(
        resource for resource in resources if process_iri in capabilities[resource[1]]
    )
    if not candidates:
        raise GroundingContractError("Pinned workcell has no capable resource for the process.")
    return _CompletionAllocationAuthority(
        process_symbol=process_matches[0][0],
        process_iri=process_iri,
        candidate_resources=candidates,
        tbox_fingerprint=tbox_fingerprint,
        registry_fingerprint=registry_fingerprint,
        workcell_fingerprint=workcell_fingerprint,
    )


def _validated_authored_state(
    root: Path,
    state_name: str,
    value: object,
    *,
    product_context: ProductContextView,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate one authored feature state and its accepted typed values."""
    state = _required_mapping(value, state_name)
    _require_exact_keys(state, {"statement", "state_values"}, state_name)
    statement = _required_mapping(state["statement"], f"{state_name}.statement")
    _require_exact_keys(
        statement,
        {"text", "evidence_refs"},
        f"{state_name}.statement",
    )
    _required_string(statement["text"], f"{state_name}.statement.text")
    statement_refs = _direct_evidence_refs(
        statement["evidence_refs"],
        f"{state_name}.statement.evidence_refs",
    )
    state_values = state["state_values"]
    if not isinstance(state_values, list):
        raise GroundingContractError(f"{state_name}.state_values must be a list.")
    names: set[str] = set()
    evidence_refs = list(statement_refs)
    for index, value_item in enumerate(state_values):
        label = f"{state_name}.state_values[{index}]"
        item = _required_mapping(value_item, label)
        _require_exact_keys(
            item,
            {"name", "value_ref", "evidence_refs"},
            label,
        )
        name = _required_string(item["name"], f"{label}.name")
        if name in names:
            raise GroundingContractError(f"{state_name} state value names must be unique.")
        names.add(name)
        evidence_refs.extend(_direct_evidence_refs(item["evidence_refs"], f"{label}.evidence_refs"))
        value_ref = _required_mapping(item["value_ref"], f"{label}.value_ref")
        _require_exact_keys(
            value_ref,
            {"record_ref", "field_path"},
            f"{label}.value_ref",
        )
        record_ref = _required_string(
            value_ref["record_ref"],
            f"{label}.value_ref.record_ref",
        )
        field_path = _required_string(
            value_ref["field_path"],
            f"{label}.value_ref.field_path",
        )
        bindings = [
            binding
            for binding in product_context.typed_bindings
            if binding.record_ref == record_ref and binding.status == "accepted"
        ]
        if len(bindings) != 1:
            raise GroundingContractError(
                "Every state value must reference one accepted typed binding."
            )
        record_path = _completion_ref_path(root, record_ref, prefix=None)
        if _sha256_path(record_path) != bindings[0].record_sha256:
            raise GroundingContractError("A target-feature typed record changed.")
        record = _read_json_mapping(record_path, "target-feature typed record")
        resolved = _resolve_json_pointer(record, field_path)
        if _empty_state_value(resolved):
            raise GroundingContractError("A target-feature state value is empty.")
    return statement_refs, tuple(evidence_refs)


def _direct_evidence_refs(value: object, label: str) -> tuple[str, ...]:
    """Validate one nonempty set of direct evidence references."""
    refs = _string_tuple(value, label)
    if not refs:
        raise GroundingContractError(f"{label} must be non-empty.")
    return refs


def _resolve_json_pointer(document: object, field_path: str) -> object:
    """Resolve one non-root RFC 6901 pointer for completion validation."""
    if not field_path.startswith("/"):
        raise GroundingContractError("state value field_path is not a JSON Pointer.")
    current = document
    for raw_token in field_path.split("/")[1:]:
        token = _decode_json_pointer_token(raw_token)
        if isinstance(current, Mapping):
            if token not in current:
                raise GroundingContractError("state value field_path does not exist.")
            current = current[token]
            continue
        if isinstance(current, list):
            if token == "-" or not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise GroundingContractError("state value field_path has an invalid array index.")
            index = int(token)
            if index >= len(current):
                raise GroundingContractError("state value field_path does not exist.")
            current = current[index]
            continue
        raise GroundingContractError("state value field_path traverses a scalar.")
    return current


def _decode_json_pointer_token(token: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(token):
        if token[index] != "~":
            result.append(token[index])
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            raise GroundingContractError("state value field_path escaping is invalid.")
        result.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(result)


def _empty_state_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (Mapping, list, tuple)):
        return not value
    return False


def _validate_compiled_delta(  # noqa: PLR0913
    delta: Mapping[str, object],
    *,
    proposal_schema_version: int,
    feature_iri: str,
    specification_iri: str,
    process_iri: str,
    state_statement_refs: Mapping[str, tuple[str, ...]],
    process_refs: tuple[str, ...],
    product_context: ProductContextView,
) -> None:
    """Require the exact host-owned target-feature RDF assertions."""
    if (
        set(delta)
        != {
            "assertions",
            "uncertainty",
            "unresolved_evidence_needs",
            "typed_context_refs",
        }
        or delta.get("uncertainty") != []
        or delta.get("unresolved_evidence_needs") != []
        or delta.get("typed_context_refs") != []
    ):
        raise GroundingContractError("Target-feature compiled delta fields are invalid.")
    desired_refs = list(state_statement_refs["desired_state"])
    expected: tuple[tuple[str, str, str, list[str]], ...] = (
        (feature_iri, "type", "feature", desired_refs),
        (specification_iri, "defines", feature_iri, desired_refs),
        (process_iri, "realizes", feature_iri, list(process_refs)),
    )
    if proposal_schema_version in {7, 8}:
        namespace = feature_iri.removesuffix("feature_0001")
        current_state_iri = f"{namespace}currentstate_0001"
        desired_state_iri = f"{namespace}desiredstate_0001"
        current_refs = list(state_statement_refs["current_state"])
        expected += (
            (current_state_iri, "type", "state", current_refs),
            (desired_state_iri, "type", "state", desired_refs),
            (feature_iri, "hascurrentstate", current_state_iri, current_refs),
            (feature_iri, "hasdesiredstate", desired_state_iri, desired_refs),
        )
    assertions = delta.get("assertions")
    if not isinstance(assertions, list) or len(assertions) != len(expected):
        raise GroundingContractError(
            "Target-feature compiled delta has an invalid assertion count."
        )
    graph_triples: set[tuple[str, str, str]] = set()
    for item in product_context.assertions:
        subject = item.get("subject")
        predicate = item.get("predicate")
        object_value = item.get("object")
        if (
            isinstance(subject, str)
            and isinstance(predicate, str)
            and isinstance(object_value, Mapping)
            and object_value.get("kind") == "iri"
            and isinstance(object_value.get("value"), str)
        ):
            graph_triples.add((subject, _iri_local_name(predicate), str(object_value["value"])))
    for assertion, expected_item in zip(assertions, expected, strict=True):
        if not isinstance(assertion, Mapping) or set(assertion) != {
            "subject",
            "predicate",
            "object",
            "evidence_refs",
        }:
            raise GroundingContractError("Target-feature assertion fields are invalid.")
        object_value = assertion["object"]
        if (
            not isinstance(object_value, Mapping)
            or set(object_value)
            != {
                "kind",
                "value",
            }
            or object_value.get("kind") != "iri"
        ):
            raise GroundingContractError("Target-feature assertion object is invalid.")
        expected_subject, expected_predicate, expected_object, expected_refs = expected_item
        actual_object = str(object_value.get("value"))
        object_matches = (
            _iri_local_name(actual_object) == expected_object
            if expected_predicate == "type"
            else actual_object == expected_object
        )
        if (
            assertion.get("subject") != expected_subject
            or _iri_local_name(str(assertion.get("predicate"))) != expected_predicate
            or not object_matches
            or assertion.get("evidence_refs") != expected_refs
            or (
                expected_subject,
                expected_predicate,
                actual_object,
            )
            not in graph_triples
        ):
            raise GroundingContractError("Target-feature compiled assertion is inconsistent.")


def persist_pa_context_grounding_completion_v6(  # noqa: PLR0913
    interaction_root: Path,
    *,
    tbox: TBoxSnapshot,
    product_requirement: str,
    completion_turn: int,
    decision_ref: str,
    product_context: ProductContextView,
    ontology_projection_ref: str,
    resource_selection_ref: str,
    tool_call_refs: Sequence[str],
    registry: ResourceRegistrySnapshot | None = None,
    workcell: PredefinedWorkcellSnapshot | None = None,
) -> Path:
    """Persist one process-aware, no-motion PA allocation."""
    root = Path(interaction_root).resolve()
    if product_context.product_requirement != product_requirement:
        raise GroundingContractError(
            "Completion requirement does not match the final ProductContextView."
        )
    tbox.assert_unchanged()
    selected_registry = registry
    if selected_registry is None and workcell is not None:
        selected_registry = workcell._registry
    if selected_registry is None:
        selected_registry = load_predefined_resource_registry(tbox)
    selected_workcell = workcell or load_predefined_workcell(tbox, selected_registry)
    selected_registry.assert_unchanged()
    selected_workcell.assert_unchanged()

    projection_path = _completion_ref_path(
        root,
        ontology_projection_ref,
        prefix=("products", "grounding", "ontology_grounding"),
    )
    selection_path = _completion_ref_path(
        root,
        resource_selection_ref,
        prefix=_RESOURCE_SELECTION_PREFIX,
    )
    projection = _read_json_mapping(projection_path, "OntologyGroundingProposal v8")
    selection = _read_json_mapping(selection_path, "ResourceSelectionRecord v4")
    process_iri = _required_string(selection.get("process_iri"), "process_iri")
    registry_record = selected_registry.to_record()
    workcell_record = selected_workcell.to_record()
    allocation_authority = _completion_allocation_authority(
        registry_record,
        workcell_record,
        process_iri=process_iri,
    )
    _target_feature, evidence_refs, review = _validated_target_feature(
        root,
        projection,
        product_context=product_context,
        selection=selection,
        selection_schema_version=4,
        allocation_authority=allocation_authority,
    )
    process_symbol = allocation_authority.process_symbol
    feature_iri = _required_string(selection["feature_iri"], "feature_iri")
    current_state_iri = _required_string(selection["current_state_iri"], "current_state_iri")
    desired_state_iri = _required_string(selection["desired_state_iri"], "desired_state_iri")
    review_ref = _required_string(projection.get("semantic_review_ref"), "semantic_review_ref")
    review_path = _completion_ref_path(
        root,
        review_ref,
        prefix=("products", "grounding", "target_feature_review"),
    )
    reachability_ref = _required_string(
        selection["reachability_check_ref"],
        "reachability_check_ref",
    )
    reachability_path = _completion_ref_path(
        root,
        reachability_ref,
        prefix=("products", "grounding", "reachability"),
    )
    validation_ref = _required_string(
        selection["robot_agent_validation_ref"],
        "robot_agent_validation_ref",
    )
    validation_path = _completion_ref_path(
        root,
        validation_ref,
        prefix=(
            "resources",
            _required_string(selection["selected_resource_jid"], "selected_resource_jid"),
            "validation",
        ),
    )
    validation = _read_json_mapping(
        validation_path,
        "PlanOnlyFeasibilityValidationRecord",
    )
    evidence_presentation_ref = _required_string(
        selection["evidence_presentation_ref"],
        "evidence_presentation_ref",
    )
    evidence_presentation_path = _completion_ref_path(
        root,
        evidence_presentation_ref,
        prefix=("products", "grounding", "presentation"),
    )
    allocation_presentation_ref = _required_string(
        selection["allocation_presentation_ref"],
        "allocation_presentation_ref",
    )
    allocation_presentation_path = _completion_ref_path(
        root,
        allocation_presentation_ref,
        prefix=("products", "grounding", "presentation"),
    )
    assignment_delta_ref = _native_assignment_delta_ref_v5(
        root,
        resource_selection_ref=resource_selection_ref,
        reachability_ref=reachability_ref,
        validation_ref=validation_ref,
        selected_resource_iri=_required_string(
            selection["selected_resource_iri"],
            "selected_resource_iri",
        ),
    )
    assignment_delta_path = _completion_ref_path(
        root,
        assignment_delta_ref,
        prefix=("products", "grounding", "ontology"),
    )
    typed_refs = tuple(
        {"ref": item.record_ref, "sha256": item.record_sha256}
        for item in product_context.typed_bindings
    )
    source_refs = tuple(
        {
            "ref": source_ref,
            "sha256": _native_source_hash(
                root,
                source_ref,
                product_requirement=product_requirement,
            ),
        }
        for source_ref in evidence_refs
    )
    tool_refs = tuple(
        {
            "ref": ref,
            "sha256": _sha256_path(_completion_ref_path(root, ref, prefix=("interaction_record",))),
        }
        for ref in tool_call_refs
    )
    completion_root = root / "products/grounding/completion"
    registry_path = completion_root / "resource_registry_snapshot_0001.json"
    workcell_path = completion_root / "predefined_workcell_snapshot_0001.json"
    _write_json_mapping_exclusive(registry_path, registry_record)
    _write_json_mapping_exclusive(workcell_path, workcell_record)
    registry_ref = registry_path.relative_to(root).as_posix()
    workcell_ref = workcell_path.relative_to(root).as_posix()

    selection_fingerprint = _sha256_string(
        selection["fingerprint"],
        "resource_selection_fingerprint",
    )
    reachability_fingerprint = _sha256_string(
        selection["reachability_check_fingerprint"],
        "reachability_check_fingerprint",
    )
    validation_fingerprint = _sha256_string(
        selection["robot_agent_validation_fingerprint"],
        "robot_agent_validation_fingerprint",
    )
    review_fingerprint = _sha256_string(
        review["fingerprint"],
        "semantic_review_fingerprint",
    )
    validation_scope = _required_string(
        validation.get("validation_scope"),
        "validation_scope",
    )
    allocation_label = (
        "validated Cartesian pick-place allocation"
        if validation_scope == "cartesian_pick_place"
        else "validated endpoint-motion allocation"
    )
    common: dict[str, object] = {
        "process_symbol": process_symbol,
        "process_iri": process_iri,
        "feature_iri": feature_iri,
        "ontology_projection_ref": ontology_projection_ref,
        "semantic_review_ref": review_ref,
        "semantic_review_sha256": _sha256_path(review_path),
        "semantic_review_fingerprint": review_fingerprint,
        "current_state_iri": current_state_iri,
        "desired_state_iri": desired_state_iri,
        "registry_snapshot_ref": registry_ref,
        "registry_snapshot_sha256": _sha256_path(registry_path),
        "registry_snapshot_fingerprint": allocation_authority.registry_fingerprint,
        "workcell_snapshot_ref": workcell_ref,
        "workcell_snapshot_sha256": _sha256_path(workcell_path),
        "workcell_snapshot_fingerprint": allocation_authority.workcell_fingerprint,
        "evidence_presentation_ref": evidence_presentation_ref,
        "evidence_presentation_sha256": _sha256_path(evidence_presentation_path),
        "evidence_presentation_fingerprint": selection["evidence_presentation_fingerprint"],
        "allocation_presentation_ref": allocation_presentation_ref,
        "allocation_presentation_sha256": _sha256_path(allocation_presentation_path),
        "allocation_presentation_fingerprint": selection["allocation_presentation_fingerprint"],
        "resource_selection_ref": resource_selection_ref,
        "resource_selection_sha256": _sha256_path(selection_path),
        "resource_selection_fingerprint": selection_fingerprint,
        "reachability_check_ref": reachability_ref,
        "reachability_check_sha256": _sha256_path(reachability_path),
        "reachability_check_fingerprint": reachability_fingerprint,
        "robot_agent_validation_ref": validation_ref,
        "robot_agent_validation_sha256": _sha256_path(validation_path),
        "robot_agent_validation_fingerprint": validation_fingerprint,
        "validation_scope": validation_scope,
        "checked_constraints": validation["checked_constraints"],
        "unvalidated_constraints": validation["unvalidated_constraints"],
        "allocation_label": allocation_label,
        "motion_executed": False,
    }
    contract: dict[str, object] = {
        "schema_version": 6,
        "record_type": "TypedGroundingContract",
        "requirement_text": product_requirement,
        **common,
        "typed_record_refs": [dict(item) for item in typed_refs],
        "source_refs": [dict(item) for item in source_refs],
        "tool_call_refs": [dict(item) for item in tool_refs],
    }
    contract["fingerprint"] = _fingerprint(contract)
    contract_path = completion_root / "typed_grounding_contract_0001.json"
    _write_json_mapping_exclusive(contract_path, contract)
    payload: dict[str, object] = {
        "schema_version": 6,
        "record_type": "PAContextGroundingCompletion",
        "status": "grounding complete",
        "product_requirement": product_requirement,
        "completion_turn": completion_turn,
        "decision_ref": decision_ref,
        **common,
        "ontology_projection_sha256": _sha256_path(projection_path),
        "typed_grounding_contract_ref": contract_path.relative_to(root).as_posix(),
        "typed_grounding_contract_sha256": _sha256_path(contract_path),
        "typed_grounding_contract_fingerprint": contract["fingerprint"],
        "resource_assignment_delta_ref": assignment_delta_ref,
        "resource_assignment_delta_sha256": _sha256_path(assignment_delta_path),
        "tbox_fingerprint": product_context.tbox_fingerprint,
        "abox_fingerprint": product_context.abox_fingerprint,
        "typed_context_refs": [dict(item) for item in typed_refs],
        "source_refs": [dict(item) for item in source_refs],
        "tool_call_refs": [dict(item) for item in tool_refs],
        "completed_at_ns": product_context.assessed_at_ns,
    }
    payload["fingerprint"] = _fingerprint(payload)
    completion_path = root / "interaction_record/context_completion_0001.json"
    _write_json_mapping_exclusive(completion_path, payload)
    load_pa_context_grounding_completion(root)
    return completion_path


def persist_pa_context_grounding_completion_v5(  # noqa: PLR0913
    interaction_root: Path,
    *,
    product_requirement: str,
    completion_turn: int,
    decision_ref: str,
    product_context: ProductContextView,
    ontology_projection_ref: str,
    resource_selection_ref: str,
    tool_call_refs: Sequence[str],
) -> Path:
    """Persist one accepted PA allocation with two state and verifier lineages."""
    root = Path(interaction_root).resolve()
    if product_context.product_requirement != product_requirement:
        raise GroundingContractError(
            "Completion requirement does not match the final ProductContextView."
        )
    projection_path = _completion_ref_path(
        root,
        ontology_projection_ref,
        prefix=("products", "grounding", "ontology_grounding"),
    )
    projection = _read_json_mapping(projection_path, "OntologyGroundingProposal")
    selection_path = _completion_ref_path(
        root,
        resource_selection_ref,
        prefix=_RESOURCE_SELECTION_PREFIX,
    )
    selection = _read_json_mapping(selection_path, "ResourceSelectionRecord")
    _target_feature, evidence_refs, review = _validated_target_feature(
        root,
        projection,
        product_context=product_context,
        selection=selection,
        selection_schema_version=3,
    )
    current_state_iri = _required_string(
        selection["current_state_iri"],
        "current_state_iri",
    )
    desired_state_iri = _required_string(
        selection["desired_state_iri"],
        "desired_state_iri",
    )
    review_ref = _required_string(
        projection.get("semantic_review_ref"),
        "semantic_review_ref",
    )
    review_path = _completion_ref_path(
        root,
        review_ref,
        prefix=("products", "grounding", "target_feature_review"),
    )
    reachability_ref = _required_string(
        selection["reachability_check_ref"],
        "reachability_check_ref",
    )
    reachability_path = _completion_ref_path(
        root,
        reachability_ref,
        prefix=("products", "grounding", "reachability"),
    )
    validation_ref = _required_string(
        selection["robot_agent_validation_ref"],
        "robot_agent_validation_ref",
    )
    validation_path = _completion_ref_path(
        root,
        validation_ref,
        prefix=(
            "resources",
            _required_string(selection["selected_resource_jid"], "selected_resource_jid"),
            "validation",
        ),
    )
    assignment_delta_ref = _native_assignment_delta_ref_v5(
        root,
        resource_selection_ref=resource_selection_ref,
        reachability_ref=reachability_ref,
        validation_ref=validation_ref,
        selected_resource_iri=_required_string(
            selection["selected_resource_iri"],
            "selected_resource_iri",
        ),
    )
    assignment_delta_path = _completion_ref_path(
        root,
        assignment_delta_ref,
        prefix=("products", "grounding", "ontology"),
    )
    typed_refs = tuple(
        {"ref": item.record_ref, "sha256": item.record_sha256}
        for item in product_context.typed_bindings
    )
    source_refs = tuple(
        {
            "ref": source_ref,
            "sha256": _native_source_hash(
                root,
                source_ref,
                product_requirement=product_requirement,
            ),
        }
        for source_ref in evidence_refs
    )
    tool_refs = tuple(
        {
            "ref": ref,
            "sha256": _sha256_path(_completion_ref_path(root, ref, prefix=("interaction_record",))),
        }
        for ref in tool_call_refs
    )
    selection_fingerprint = _sha256_string(
        selection["fingerprint"],
        "resource_selection_fingerprint",
    )
    reachability_fingerprint = _sha256_string(
        selection["reachability_check_fingerprint"],
        "reachability_check_fingerprint",
    )
    validation_fingerprint = _sha256_string(
        selection["robot_agent_validation_fingerprint"],
        "robot_agent_validation_fingerprint",
    )
    review_fingerprint = _sha256_string(
        review["fingerprint"],
        "semantic_review_fingerprint",
    )
    contract: dict[str, object] = {
        "schema_version": 5,
        "record_type": "TypedGroundingContract",
        "requirement_text": product_requirement,
        "ontology_projection_ref": ontology_projection_ref,
        "semantic_review_ref": review_ref,
        "semantic_review_sha256": _sha256_path(review_path),
        "semantic_review_fingerprint": review_fingerprint,
        "current_state_iri": current_state_iri,
        "desired_state_iri": desired_state_iri,
        "resource_selection_ref": resource_selection_ref,
        "resource_selection_sha256": _sha256_path(selection_path),
        "resource_selection_fingerprint": selection_fingerprint,
        "reachability_check_ref": reachability_ref,
        "reachability_check_sha256": _sha256_path(reachability_path),
        "reachability_check_fingerprint": reachability_fingerprint,
        "robot_agent_validation_ref": validation_ref,
        "robot_agent_validation_sha256": _sha256_path(validation_path),
        "robot_agent_validation_fingerprint": validation_fingerprint,
        "allocation_label": "validated allocation",
        "motion_executed": False,
        "typed_record_refs": [dict(item) for item in typed_refs],
        "source_refs": [dict(item) for item in source_refs],
        "tool_call_refs": [dict(item) for item in tool_refs],
    }
    contract["fingerprint"] = _fingerprint(contract)
    contract_path = root / "products/grounding/completion/typed_grounding_contract_0001.json"
    _write_json_mapping_exclusive(contract_path, contract)
    payload: dict[str, object] = {
        "schema_version": 5,
        "record_type": "PAContextGroundingCompletion",
        "status": "grounding complete",
        "allocation_label": "validated allocation",
        "motion_executed": False,
        "product_requirement": product_requirement,
        "completion_turn": completion_turn,
        "decision_ref": decision_ref,
        "ontology_projection_ref": ontology_projection_ref,
        "ontology_projection_sha256": _sha256_path(projection_path),
        "semantic_review_ref": review_ref,
        "semantic_review_sha256": _sha256_path(review_path),
        "semantic_review_fingerprint": review_fingerprint,
        "current_state_iri": current_state_iri,
        "desired_state_iri": desired_state_iri,
        "typed_grounding_contract_ref": contract_path.relative_to(root).as_posix(),
        "typed_grounding_contract_sha256": _sha256_path(contract_path),
        "typed_grounding_contract_fingerprint": contract["fingerprint"],
        "resource_selection_ref": resource_selection_ref,
        "resource_selection_sha256": _sha256_path(selection_path),
        "resource_selection_fingerprint": selection_fingerprint,
        "reachability_check_ref": reachability_ref,
        "reachability_check_sha256": _sha256_path(reachability_path),
        "reachability_check_fingerprint": reachability_fingerprint,
        "robot_agent_validation_ref": validation_ref,
        "robot_agent_validation_sha256": _sha256_path(validation_path),
        "robot_agent_validation_fingerprint": validation_fingerprint,
        "resource_assignment_delta_ref": assignment_delta_ref,
        "resource_assignment_delta_sha256": _sha256_path(assignment_delta_path),
        "tbox_fingerprint": product_context.tbox_fingerprint,
        "abox_fingerprint": product_context.abox_fingerprint,
        "typed_context_refs": [dict(item) for item in typed_refs],
        "source_refs": [dict(item) for item in source_refs],
        "tool_call_refs": [dict(item) for item in tool_refs],
        "completed_at_ns": product_context.assessed_at_ns,
    }
    payload["fingerprint"] = _fingerprint(payload)
    completion_path = root / "interaction_record/context_completion_0001.json"
    _write_json_mapping_exclusive(completion_path, payload)
    load_pa_context_grounding_completion(root)
    return completion_path


def persist_pa_context_grounding_completion_v4(  # noqa: PLR0913
    interaction_root: Path,
    *,
    product_requirement: str,
    completion_turn: int,
    decision_ref: str,
    product_context: ProductContextView,
    ontology_projection_ref: str,
    resource_selection_ref: str,
    tool_call_refs: Sequence[str],
) -> Path:
    """Persist the reviewed target feature and its exact grounding lineage."""
    root = Path(interaction_root).resolve()
    if product_context.product_requirement != product_requirement:
        raise GroundingContractError(
            "Completion requirement does not match the final ProductContextView."
        )
    projection_path = _completion_ref_path(
        root,
        ontology_projection_ref,
        prefix=("products", "grounding", "ontology_grounding"),
    )
    projection = _read_json_mapping(projection_path, "OntologyGroundingProposal")
    selection_path = _completion_ref_path(
        root,
        resource_selection_ref,
        prefix=_RESOURCE_SELECTION_PREFIX,
    )
    selection = _read_json_mapping(selection_path, "ResourceSelectionRecord")
    target_feature, evidence_refs, review = _validated_target_feature(
        root,
        projection,
        product_context=product_context,
        selection=selection,
    )
    del target_feature
    review_ref = _required_string(projection.get("semantic_review_ref"), "semantic_review_ref")
    review_path = _completion_ref_path(
        root,
        review_ref,
        prefix=("products", "grounding", "target_feature_review"),
    )
    review_sha256 = _sha256_path(review_path)
    review_fingerprint = _sha256_string(review["fingerprint"], "semantic_review_fingerprint")
    assignment_delta_ref = _native_assignment_delta_ref(
        root,
        resource_selection_ref=resource_selection_ref,
        selected_resource_iri=_required_string(
            selection.get("selected_resource_iri"), "selected_resource_iri"
        ),
    )
    assignment_delta_path = _completion_ref_path(
        root,
        assignment_delta_ref,
        prefix=("products", "grounding", "ontology"),
    )
    typed_refs = tuple(
        {"ref": item.record_ref, "sha256": item.record_sha256}
        for item in product_context.typed_bindings
    )
    source_refs = tuple(
        {
            "ref": source_ref,
            "sha256": _native_source_hash(
                root,
                source_ref,
                product_requirement=product_requirement,
            ),
        }
        for source_ref in evidence_refs
    )
    tool_refs = tuple(
        {
            "ref": ref,
            "sha256": _sha256_path(_completion_ref_path(root, ref, prefix=("interaction_record",))),
        }
        for ref in tool_call_refs
    )
    contract: dict[str, object] = {
        "schema_version": 4,
        "record_type": "TypedGroundingContract",
        "requirement_text": product_requirement,
        "ontology_projection_ref": ontology_projection_ref,
        "semantic_review_ref": review_ref,
        "semantic_review_sha256": review_sha256,
        "semantic_review_fingerprint": review_fingerprint,
        "typed_record_refs": [dict(item) for item in typed_refs],
        "source_refs": [dict(item) for item in source_refs],
        "tool_call_refs": [dict(item) for item in tool_refs],
    }
    contract["fingerprint"] = _fingerprint(contract)
    contract_path = root / "products/grounding/completion/typed_grounding_contract_0001.json"
    _write_json_mapping_exclusive(contract_path, contract)
    payload: dict[str, object] = {
        "schema_version": 4,
        "record_type": "PAContextGroundingCompletion",
        "status": "grounding complete",
        "product_requirement": product_requirement,
        "completion_turn": completion_turn,
        "decision_ref": decision_ref,
        "ontology_projection_ref": ontology_projection_ref,
        "ontology_projection_sha256": _sha256_path(projection_path),
        "semantic_review_ref": review_ref,
        "semantic_review_sha256": review_sha256,
        "semantic_review_fingerprint": review_fingerprint,
        "typed_grounding_contract_ref": contract_path.relative_to(root).as_posix(),
        "typed_grounding_contract_sha256": _sha256_path(contract_path),
        "typed_grounding_contract_fingerprint": contract["fingerprint"],
        "resource_selection_ref": resource_selection_ref,
        "resource_selection_sha256": _sha256_path(selection_path),
        "resource_assignment_delta_ref": assignment_delta_ref,
        "resource_assignment_delta_sha256": _sha256_path(assignment_delta_path),
        "tbox_fingerprint": product_context.tbox_fingerprint,
        "abox_fingerprint": product_context.abox_fingerprint,
        "typed_context_refs": [dict(item) for item in typed_refs],
        "source_refs": [dict(item) for item in source_refs],
        "tool_call_refs": [dict(item) for item in tool_refs],
        "completed_at_ns": product_context.assessed_at_ns,
    }
    payload["fingerprint"] = _fingerprint(payload)
    completion_path = root / "interaction_record/context_completion_0001.json"
    _write_json_mapping_exclusive(completion_path, payload)
    load_pa_context_grounding_completion(root)
    return completion_path


def persist_pa_context_grounding_completion_v3(  # noqa: PLR0913
    interaction_root: Path,
    *,
    product_requirement: str,
    completion_turn: int,
    decision_ref: str,
    product_context: ProductContextView,
    ontology_projection_ref: str,
    resource_selection_ref: str,
    tool_call_refs: Sequence[str],
) -> Path:
    """Persist a session-free, hash-pinned native-tool completion bundle."""
    root = Path(interaction_root).resolve()
    projection_path = _completion_ref_path(
        root,
        ontology_projection_ref,
        prefix=("products", "grounding", "ontology_grounding"),
    )
    projection = _read_json_mapping(projection_path, "OntologyGroundingProposal")
    output = projection.get("output")
    if (
        projection.get("schema_version") != 5
        or projection.get("status") != "accepted"
        or not isinstance(output, Mapping)
    ):
        raise GroundingContractError(
            "Native completion requires one accepted version-5 ontology proposal."
        )
    context_summary = output.get("context_summary")
    context_evidence_refs = output.get("evidence_refs")
    missing_information = output.get("missing_information")
    if (
        not isinstance(context_summary, str)
        or not context_summary.strip()
        or not isinstance(context_evidence_refs, list)
        or not context_evidence_refs
        or not all(isinstance(item, str) and item for item in context_evidence_refs)
        or not isinstance(missing_information, list)
        or not all(isinstance(item, str) and item for item in missing_information)
    ):
        raise GroundingContractError("Native ontology proposal context is invalid.")
    selection_path = _completion_ref_path(
        root,
        resource_selection_ref,
        prefix=_RESOURCE_SELECTION_PREFIX,
    )
    selection = _read_json_mapping(selection_path, "ResourceSelectionRecord")
    if (
        selection.get("schema_version") != 2
        or selection.get("record_type") != "ResourceSelectionRecord"
        or not isinstance(selection.get("selected_resource_iri"), str)
        or selection.get("required_record_type") != "RobotFrameLocationRecord"
    ):
        raise GroundingContractError(
            "Native completion requires an accepted location-based resource selection."
        )
    assignment_delta_ref = _native_assignment_delta_ref(
        root,
        resource_selection_ref=resource_selection_ref,
        selected_resource_iri=str(selection["selected_resource_iri"]),
    )
    assignment_delta_path = _completion_ref_path(
        root,
        assignment_delta_ref,
        prefix=("products", "grounding", "ontology"),
    )
    typed_refs = tuple(
        {"ref": item.record_ref, "sha256": item.record_sha256}
        for item in product_context.typed_bindings
    )
    location_binding = next(
        (
            item
            for item in product_context.typed_bindings
            if item.record_type == "RobotFrameLocationRecord"
            and item.status == "accepted"
            and item.record_ref == selection.get("grounding_record_ref")
            and item.record_sha256 == selection.get("grounding_record_sha256")
        ),
        None,
    )
    if location_binding is None:
        raise GroundingContractError(
            "Native completion resource selection is not pinned to the final location."
        )
    source_refs = tuple(
        {
            "ref": source_ref,
            "sha256": _native_source_hash(
                root,
                source_ref,
                product_requirement=product_requirement,
            ),
        }
        for source_ref in sorted(set(context_evidence_refs))
    )
    tool_refs = tuple(
        {
            "ref": ref,
            "sha256": _sha256_path(_completion_ref_path(root, ref, prefix=("interaction_record",))),
        }
        for ref in tool_call_refs
    )
    contract: dict[str, object] = {
        "schema_version": 3,
        "record_type": "TypedGroundingContract",
        "requirement_text": product_requirement,
        "ontology_projection_ref": ontology_projection_ref,
        "context_summary": _scoped_native_context_summary(context_summary),
        "context_evidence_refs": list(context_evidence_refs),
        "missing_information": list(missing_information),
        "typed_record_refs": [dict(item) for item in typed_refs],
        "source_refs": [dict(item) for item in source_refs],
        "tool_call_refs": [dict(item) for item in tool_refs],
    }
    contract["fingerprint"] = _fingerprint(contract)
    contract_path = root / "products/grounding/completion/typed_grounding_contract_0001.json"
    _write_json_mapping_exclusive(contract_path, contract)
    payload: dict[str, object] = {
        "schema_version": 3,
        "record_type": "PAContextGroundingCompletion",
        "status": "grounding complete",
        "product_requirement": product_requirement,
        "completion_turn": completion_turn,
        "decision_ref": decision_ref,
        "ontology_projection_ref": ontology_projection_ref,
        "ontology_projection_sha256": _sha256_path(projection_path),
        "typed_grounding_contract_ref": contract_path.relative_to(root).as_posix(),
        "typed_grounding_contract_sha256": _sha256_path(contract_path),
        "typed_grounding_contract_fingerprint": contract["fingerprint"],
        "resource_selection_ref": resource_selection_ref,
        "resource_selection_sha256": _sha256_path(selection_path),
        "resource_assignment_delta_ref": assignment_delta_ref,
        "resource_assignment_delta_sha256": _sha256_path(assignment_delta_path),
        "tbox_fingerprint": product_context.tbox_fingerprint,
        "abox_fingerprint": product_context.abox_fingerprint,
        "typed_context_refs": [dict(item) for item in typed_refs],
        "source_refs": [dict(item) for item in source_refs],
        "tool_call_refs": [dict(item) for item in tool_refs],
        "completed_at_ns": product_context.assessed_at_ns,
    }
    payload["fingerprint"] = _fingerprint(payload)
    completion_path = root / "interaction_record/context_completion_0001.json"
    _write_json_mapping_exclusive(completion_path, payload)
    load_pa_context_grounding_completion(root)
    return completion_path


def _scoped_native_context_summary(context_summary: str) -> str:
    """Mark the PA narrative as preceding deterministic completion state."""
    return (
        f"{_NATIVE_CONTEXT_SUMMARY_PREFIX}\n{context_summary}\n\n{_NATIVE_CONTEXT_SUMMARY_SUFFIX}"
    )


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
    proposal_candidates = [
        (path, _read_json_mapping(path, "OntologyGroundingProposal"))
        for path in sorted((root / "products/grounding/ontology_grounding").glob("proposal_*.json"))
    ]
    accepted_proposals = [
        (path, proposal)
        for path, proposal in proposal_candidates
        if proposal.get("schema_version") in _ONTOLOGY_PROPOSAL_SCHEMA_VERSIONS
        and proposal.get("status") == "accepted"
    ]
    if len(accepted_proposals) != 1:
        raise GroundingContractError(
            "PAContextGroundingCompletion v2 requires exactly one accepted ontology projection."
        )
    proposal_path, proposal = accepted_proposals[0]
    proposal_session = _proposal_grounding_session(root, proposal)
    _validate_completion_lineage(session, proposal_session)
    resource_selection_ref, resource_assignment_delta_ref = _validate_completion_assignment(
        root, product_context
    )
    resource_selection_path = _completion_ref_path(
        root,
        resource_selection_ref,
        prefix=_RESOURCE_SELECTION_PREFIX,
    )
    resource_assignment_delta_path = _completion_ref_path(
        root,
        resource_assignment_delta_ref,
        prefix=("products", "grounding", "ontology"),
    )
    output = proposal.get("output")
    proposal_value = (
        output.get("ontology_grounding_proposal") if isinstance(output, Mapping) else None
    )
    context_summary = (
        proposal_value.get("context_summary") if isinstance(proposal_value, Mapping) else None
    )
    context_evidence_refs = (
        proposal_value.get("evidence_refs") if isinstance(proposal_value, Mapping) else None
    )
    missing_information = (
        proposal_value.get("missing_information") if isinstance(proposal_value, Mapping) else None
    )
    if (
        not isinstance(context_summary, str)
        or not context_summary.strip()
        or not isinstance(context_evidence_refs, list)
        or not context_evidence_refs
        or not all(isinstance(item, str) and item for item in context_evidence_refs)
        or not isinstance(missing_information, list)
        or not all(isinstance(item, str) and item for item in missing_information)
    ):
        raise GroundingContractError("Ontology projection final context is invalid.")
    typed_refs = tuple(
        {"ref": item.record_ref, "sha256": item.record_sha256}
        for item in product_context.typed_bindings
    )
    source_ids = sorted(set(context_evidence_refs))
    source_refs = tuple(
        {"ref": source, "sha256": _grounding_source_hash(root, source, session)}
        for source in source_ids
    )
    clarification_records = tuple(
        {
            "ref": ref,
            "sha256": _sha256_path(_completion_ref_path(root, ref, prefix=("interaction_record",))),
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
        context_summary=context_summary,
        context_evidence_refs=context_evidence_refs,
        missing_information=missing_information,
        typed_record_refs=typed_refs,
        source_refs=source_refs,
        clarification_refs=clarification_records,
    )
    contract_path = root / "products/grounding/completion/typed_grounding_contract_0001.json"
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
        "resource_selection_ref": resource_selection_ref,
        "resource_selection_sha256": _sha256_path(resource_selection_path),
        "resource_assignment_delta_ref": resource_assignment_delta_ref,
        "resource_assignment_delta_sha256": _sha256_path(resource_assignment_delta_path),
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
) -> (
    PAContextGroundingCompletionV2
    | PAContextGroundingCompletionV3
    | PAContextGroundingCompletionV4
    | PAContextGroundingCompletionV5
    | PAContextGroundingCompletionV6
):
    """Load and verify one current or recovered completion bundle."""
    root = Path(interaction_root).resolve()
    paths = sorted((root / "interaction_record").glob("context_completion_*.json"))
    if len(paths) != 1:
        raise GroundingContractError("Exactly one PAContextGroundingCompletion record is required.")
    completion_value = _read_json_mapping(paths[0], "PAContextGroundingCompletion")
    if completion_value.get("schema_version") == 6:
        return _load_pa_context_grounding_completion_v6(root, completion_value)
    if completion_value.get("schema_version") == 5:
        return _load_pa_context_grounding_completion_v5(root, completion_value)
    if completion_value.get("schema_version") == 4:
        return _load_pa_context_grounding_completion_v4(root, completion_value)
    if completion_value.get("schema_version") == 3:
        return _load_pa_context_grounding_completion_v3(root, completion_value)
    if completion_value.get("schema_version") != 2:
        raise GroundingContractError("PAContextGroundingCompletion version is unsupported.")
    return _load_pa_context_grounding_completion_v2(root, completion_value)


def load_completed_product_context_view(
    interaction_root: Path,
) -> ProductContextView:
    """Load the exact final ProductContextView pinned by current PA completion."""
    root = Path(interaction_root).resolve()
    completion = load_pa_context_grounding_completion(root)
    if not isinstance(
        completion,
        (
            PAContextGroundingCompletionV3,
            PAContextGroundingCompletionV4,
            PAContextGroundingCompletionV5,
            PAContextGroundingCompletionV6,
        ),
    ):
        raise GroundingContractError(
            "Completed ProductContextView retrieval requires native completion version 3, 4, 5, or 6."
        )
    return _latest_product_context_view(root)


def _load_pa_context_grounding_completion_v6(  # noqa: C901
    root: Path,
    value: Mapping[str, object],
) -> PAContextGroundingCompletionV6:
    """Validate the process-aware plan-only allocation completion."""
    common_fields = {
        "process_symbol",
        "process_iri",
        "feature_iri",
        "ontology_projection_ref",
        "semantic_review_ref",
        "semantic_review_sha256",
        "semantic_review_fingerprint",
        "current_state_iri",
        "desired_state_iri",
        "registry_snapshot_ref",
        "registry_snapshot_sha256",
        "registry_snapshot_fingerprint",
        "workcell_snapshot_ref",
        "workcell_snapshot_sha256",
        "workcell_snapshot_fingerprint",
        "evidence_presentation_ref",
        "evidence_presentation_sha256",
        "evidence_presentation_fingerprint",
        "allocation_presentation_ref",
        "allocation_presentation_sha256",
        "allocation_presentation_fingerprint",
        "resource_selection_ref",
        "resource_selection_sha256",
        "resource_selection_fingerprint",
        "reachability_check_ref",
        "reachability_check_sha256",
        "reachability_check_fingerprint",
        "robot_agent_validation_ref",
        "robot_agent_validation_sha256",
        "robot_agent_validation_fingerprint",
        "validation_scope",
        "checked_constraints",
        "unvalidated_constraints",
        "allocation_label",
        "motion_executed",
    }
    expected = {
        "schema_version",
        "record_type",
        "status",
        "product_requirement",
        "completion_turn",
        "decision_ref",
        *common_fields,
        "ontology_projection_sha256",
        "typed_grounding_contract_ref",
        "typed_grounding_contract_sha256",
        "typed_grounding_contract_fingerprint",
        "resource_assignment_delta_ref",
        "resource_assignment_delta_sha256",
        "tbox_fingerprint",
        "abox_fingerprint",
        "typed_context_refs",
        "source_refs",
        "tool_call_refs",
        "completed_at_ns",
        "fingerprint",
    }
    _require_exact_keys(value, expected, "PAContextGroundingCompletion v6")
    payload = dict(value)
    fingerprint = payload.pop("fingerprint", None)
    validation_scope = value.get("validation_scope")
    if validation_scope == "cartesian_pick_place":
        allocation_label = "validated Cartesian pick-place allocation"
        checked_constraints = [
            "live_tf",
            "collision_aware_cartesian_pick_path",
            "collision_aware_cartesian_transfer_place_path",
            "complete_path_fraction",
        ]
        unvalidated_constraints = [
            "grasp_contact",
            "gripper_actuation",
            "attached_part_collision_geometry",
            f"{value['process_symbol']}_tolerance",
            "force_control",
            "final_constrained_insertion_stroke",
        ]
    elif validation_scope == "endpoint_motion":
        allocation_label = "validated endpoint-motion allocation"
        checked_constraints = [
            "positional_ik",
            "collision_aware_endpoints",
            "path_between_endpoints",
        ]
        unvalidated_constraints = [
            "grasping",
            "end_effector_orientation",
            "attached_object_geometry",
            f"{value['process_symbol']}_tolerance",
            "force_contact",
            "insertion_constraints",
        ]
    else:
        raise GroundingContractError("PAContextGroundingCompletion v6 validation scope is invalid.")
    if (
        value.get("schema_version") != 6
        or value.get("record_type") != "PAContextGroundingCompletion"
        or value.get("status") != "grounding complete"
        or value.get("allocation_label") != allocation_label
        or value.get("checked_constraints") != checked_constraints
        or value.get("unvalidated_constraints") != unvalidated_constraints
        or value.get("motion_executed") is not False
        or not _is_sha256_value(fingerprint)
        or _fingerprint(payload) != fingerprint
    ):
        raise GroundingContractError("PAContextGroundingCompletion v6 is invalid.")

    requirement = _required_string(value["product_requirement"], "product_requirement")
    completion_turn = _positive_integer(value["completion_turn"], "completion_turn")
    decision_path = _completion_ref_path(
        root,
        _required_string(value["decision_ref"], "decision_ref"),
        prefix=("interaction_record",),
    )
    decision = _read_json_mapping(decision_path, "native grounding decision")
    output = decision.get("PA_output")
    if (
        decision.get("turn") != completion_turn
        or decision.get("product_requirement") != requirement
        or decision.get("failure") is not None
        or not isinstance(output, Mapping)
        or output.get("grounding_status") != "complete"
        or output.get("allocation_label") != allocation_label
        or output.get("ontology_projection_ref") != value["ontology_projection_ref"]
        or output.get("resource_selection_ref") != value["resource_selection_ref"]
    ):
        raise GroundingContractError("Native v6 grounding decision reference is invalid.")

    hashed_fields = (
        ("ontology_projection_ref", "ontology_projection_sha256"),
        ("semantic_review_ref", "semantic_review_sha256"),
        ("registry_snapshot_ref", "registry_snapshot_sha256"),
        ("workcell_snapshot_ref", "workcell_snapshot_sha256"),
        ("evidence_presentation_ref", "evidence_presentation_sha256"),
        ("allocation_presentation_ref", "allocation_presentation_sha256"),
        ("typed_grounding_contract_ref", "typed_grounding_contract_sha256"),
        ("resource_selection_ref", "resource_selection_sha256"),
        ("reachability_check_ref", "reachability_check_sha256"),
        ("robot_agent_validation_ref", "robot_agent_validation_sha256"),
        ("resource_assignment_delta_ref", "resource_assignment_delta_sha256"),
    )
    for ref_field, hash_field in hashed_fields:
        path = _completion_ref_path(
            root,
            _required_string(value[ref_field], ref_field),
            prefix=None,
        )
        if _sha256_path(path) != _sha256_string(value[hash_field], hash_field):
            raise GroundingContractError(f"Native v6 {ref_field} hash is invalid.")

    typed_refs = _hashed_ref_tuple(value["typed_context_refs"], "typed_context_refs")
    source_refs = _hashed_ref_tuple(value["source_refs"], "source_refs")
    tool_refs = _hashed_ref_tuple(value["tool_call_refs"], "tool_call_refs")
    for item in (*typed_refs, *tool_refs):
        path = _completion_ref_path(root, item["ref"], prefix=None)
        if _sha256_path(path) != item["sha256"]:
            raise GroundingContractError("Native v6 pinned record changed.")
    for item in source_refs:
        if (
            _native_source_hash(root, item["ref"], product_requirement=requirement)
            != item["sha256"]
        ):
            raise GroundingContractError("Native v6 source hash is invalid.")

    latest_view = _latest_product_context_view(root)
    completed_at_ns = _nonnegative_integer(value["completed_at_ns"], "completed_at_ns")
    if (
        latest_view.product_requirement != requirement
        or latest_view.tbox_fingerprint != value["tbox_fingerprint"]
        or latest_view.abox_fingerprint != value["abox_fingerprint"]
        or latest_view.assessed_at_ns != completed_at_ns
        or tuple(
            {"ref": item.record_ref, "sha256": item.record_sha256}
            for item in latest_view.typed_bindings
        )
        != typed_refs
    ):
        raise GroundingContractError("Native v6 final ProductContextView is inconsistent.")

    registry_path = _completion_ref_path(
        root,
        str(value["registry_snapshot_ref"]),
        prefix=("products", "grounding", "completion"),
    )
    workcell_path = _completion_ref_path(
        root,
        str(value["workcell_snapshot_ref"]),
        prefix=("products", "grounding", "completion"),
    )
    registry_record = _read_json_mapping(registry_path, "ResourceRegistrySnapshot v1")
    workcell_record = _read_json_mapping(workcell_path, "PredefinedWorkcellSnapshot v2")
    process_iri = _required_string(value["process_iri"], "process_iri")
    allocation_authority = _completion_allocation_authority(
        registry_record,
        workcell_record,
        process_iri=process_iri,
    )
    if (
        allocation_authority.process_symbol != value["process_symbol"]
        or allocation_authority.tbox_fingerprint != value["tbox_fingerprint"]
        or allocation_authority.registry_fingerprint != value["registry_snapshot_fingerprint"]
        or allocation_authority.workcell_fingerprint != value["workcell_snapshot_fingerprint"]
    ):
        raise GroundingContractError("Native v6 allocation authority is inconsistent.")

    projection_path = _completion_ref_path(
        root,
        str(value["ontology_projection_ref"]),
        prefix=("products", "grounding", "ontology_grounding"),
    )
    selection_path = _completion_ref_path(
        root,
        str(value["resource_selection_ref"]),
        prefix=_RESOURCE_SELECTION_PREFIX,
    )
    projection = _read_json_mapping(projection_path, "OntologyGroundingProposal v8")
    selection = _read_json_mapping(selection_path, "ResourceSelectionRecord v4")
    _target_feature, evidence_refs, review = _validated_target_feature(
        root,
        projection,
        product_context=latest_view,
        selection=selection,
        selection_schema_version=4,
        allocation_authority=allocation_authority,
    )
    if tuple(item["ref"] for item in source_refs) != evidence_refs:
        raise GroundingContractError(
            "Native v6 sources do not match nested target-feature evidence."
        )
    direct_pairs = (
        ("process_symbol", "process_symbol"),
        ("process_iri", "process_iri"),
        ("feature_iri", "feature_iri"),
        ("current_state_iri", "current_state_iri"),
        ("desired_state_iri", "desired_state_iri"),
        ("evidence_presentation_ref", "evidence_presentation_ref"),
        ("evidence_presentation_sha256", "evidence_presentation_sha256"),
        ("evidence_presentation_fingerprint", "evidence_presentation_fingerprint"),
        ("allocation_presentation_ref", "allocation_presentation_ref"),
        ("allocation_presentation_sha256", "allocation_presentation_sha256"),
        ("allocation_presentation_fingerprint", "allocation_presentation_fingerprint"),
        ("resource_selection_fingerprint", "fingerprint"),
        ("reachability_check_ref", "reachability_check_ref"),
        ("reachability_check_sha256", "reachability_check_sha256"),
        ("reachability_check_fingerprint", "reachability_check_fingerprint"),
        ("robot_agent_validation_ref", "robot_agent_validation_ref"),
        ("robot_agent_validation_sha256", "robot_agent_validation_sha256"),
        (
            "robot_agent_validation_fingerprint",
            "robot_agent_validation_fingerprint",
        ),
    )
    if any(value[left] != selection[right] for left, right in direct_pairs):
        raise GroundingContractError("Native v6 allocation lineage is inconsistent.")
    if (
        projection.get("semantic_review_ref") != value["semantic_review_ref"]
        or projection.get("semantic_review_sha256") != value["semantic_review_sha256"]
        or review.get("fingerprint") != value["semantic_review_fingerprint"]
    ):
        raise GroundingContractError("Native v6 semantic review lineage is invalid.")

    validation_path = _completion_ref_path(
        root,
        str(value["robot_agent_validation_ref"]),
        prefix=(
            "resources",
            _required_string(selection["selected_resource_jid"], "selected_resource_jid"),
            "validation",
        ),
    )
    validation = _read_json_mapping(
        validation_path,
        "PlanOnlyFeasibilityValidationRecord v2",
    )
    if (
        validation.get("validation_scope") != value["validation_scope"]
        or validation.get("checked_constraints") != value["checked_constraints"]
        or validation.get("unvalidated_constraints") != value["unvalidated_constraints"]
        or validation.get("motion_executed") is not False
        or validation.get("status") != "accepted"
    ):
        raise GroundingContractError("Native v6 validation scope is inconsistent.")
    assignment_ref = _native_assignment_delta_ref_v5(
        root,
        resource_selection_ref=str(value["resource_selection_ref"]),
        reachability_ref=str(value["reachability_check_ref"]),
        validation_ref=str(value["robot_agent_validation_ref"]),
        selected_resource_iri=_required_string(
            selection["selected_resource_iri"],
            "selected_resource_iri",
        ),
    )
    if assignment_ref != value["resource_assignment_delta_ref"]:
        raise GroundingContractError("Native v6 assignment lineage is invalid.")

    contract_path = _completion_ref_path(
        root,
        str(value["typed_grounding_contract_ref"]),
        prefix=("products", "grounding", "completion"),
    )
    contract = _read_json_mapping(contract_path, "TypedGroundingContract v6")
    contract_expected = {
        "schema_version",
        "record_type",
        "requirement_text",
        *common_fields,
        "typed_record_refs",
        "source_refs",
        "tool_call_refs",
        "fingerprint",
    }
    _require_exact_keys(contract, contract_expected, "TypedGroundingContract v6")
    contract_payload = dict(contract)
    contract_fingerprint = contract_payload.pop("fingerprint", None)
    if (
        contract.get("schema_version") != 6
        or contract.get("record_type") != "TypedGroundingContract"
        or contract.get("requirement_text") != requirement
        or contract_fingerprint != value["typed_grounding_contract_fingerprint"]
        or not _is_sha256_value(contract_fingerprint)
        or _fingerprint(contract_payload) != contract_fingerprint
        or any(contract[field] != value[field] for field in common_fields)
        or contract.get("typed_record_refs") != list(typed_refs)
        or contract.get("source_refs") != list(source_refs)
        or contract.get("tool_call_refs") != list(tool_refs)
    ):
        raise GroundingContractError("TypedGroundingContract v6 is inconsistent.")
    return PAContextGroundingCompletionV6(record=dict(value))


def _load_pa_context_grounding_completion_v5(
    root: Path,
    value: Mapping[str, object],
) -> PAContextGroundingCompletionV5:
    """Validate the current two-state, PA-authored allocation completion."""
    expected = {
        "schema_version",
        "record_type",
        "status",
        "allocation_label",
        "motion_executed",
        "product_requirement",
        "completion_turn",
        "decision_ref",
        "ontology_projection_ref",
        "ontology_projection_sha256",
        "semantic_review_ref",
        "semantic_review_sha256",
        "semantic_review_fingerprint",
        "current_state_iri",
        "desired_state_iri",
        "typed_grounding_contract_ref",
        "typed_grounding_contract_sha256",
        "typed_grounding_contract_fingerprint",
        "resource_selection_ref",
        "resource_selection_sha256",
        "resource_selection_fingerprint",
        "reachability_check_ref",
        "reachability_check_sha256",
        "reachability_check_fingerprint",
        "robot_agent_validation_ref",
        "robot_agent_validation_sha256",
        "robot_agent_validation_fingerprint",
        "resource_assignment_delta_ref",
        "resource_assignment_delta_sha256",
        "tbox_fingerprint",
        "abox_fingerprint",
        "typed_context_refs",
        "source_refs",
        "tool_call_refs",
        "completed_at_ns",
        "fingerprint",
    }
    _require_exact_keys(value, expected, "PAContextGroundingCompletion v5")
    payload = dict(value)
    fingerprint = payload.pop("fingerprint", None)
    if (
        value.get("schema_version") != 5
        or value.get("record_type") != "PAContextGroundingCompletion"
        or value.get("status") != "grounding complete"
        or value.get("allocation_label") != "validated allocation"
        or value.get("motion_executed") is not False
        or not _is_sha256_value(fingerprint)
        or _fingerprint(payload) != fingerprint
    ):
        raise GroundingContractError("PAContextGroundingCompletion v5 is invalid.")
    requirement = _required_string(
        value["product_requirement"],
        "product_requirement",
    )
    completion_turn = _positive_integer(value["completion_turn"], "completion_turn")
    decision_path = _completion_ref_path(
        root,
        _required_string(value["decision_ref"], "decision_ref"),
        prefix=("interaction_record",),
    )
    decision = _read_json_mapping(decision_path, "native grounding decision")
    output = decision.get("PA_output")
    if (
        decision.get("turn") != completion_turn
        or decision.get("product_requirement") != requirement
        or decision.get("failure") is not None
        or not isinstance(output, Mapping)
        or output.get("grounding_status") != "complete"
        or output.get("allocation_label") != "validated allocation"
        or output.get("ontology_projection_ref") != value["ontology_projection_ref"]
        or output.get("resource_selection_ref") != value["resource_selection_ref"]
    ):
        raise GroundingContractError("Native v5 grounding decision reference is invalid.")

    hashed_fields = (
        ("ontology_projection_ref", "ontology_projection_sha256"),
        ("semantic_review_ref", "semantic_review_sha256"),
        ("typed_grounding_contract_ref", "typed_grounding_contract_sha256"),
        ("resource_selection_ref", "resource_selection_sha256"),
        ("reachability_check_ref", "reachability_check_sha256"),
        ("robot_agent_validation_ref", "robot_agent_validation_sha256"),
        ("resource_assignment_delta_ref", "resource_assignment_delta_sha256"),
    )
    for ref_field, hash_field in hashed_fields:
        path = _completion_ref_path(
            root,
            _required_string(value[ref_field], ref_field),
            prefix=None,
        )
        if _sha256_path(path) != _sha256_string(value[hash_field], hash_field):
            raise GroundingContractError(f"Native v5 {ref_field} hash is invalid.")

    typed_refs = _hashed_ref_tuple(value["typed_context_refs"], "typed_context_refs")
    source_refs = _hashed_ref_tuple(value["source_refs"], "source_refs")
    tool_refs = _hashed_ref_tuple(value["tool_call_refs"], "tool_call_refs")
    for item in (*typed_refs, *tool_refs):
        path = _completion_ref_path(root, item["ref"], prefix=None)
        if _sha256_path(path) != item["sha256"]:
            raise GroundingContractError("Native v5 pinned record changed.")
    for item in source_refs:
        if (
            _native_source_hash(
                root,
                item["ref"],
                product_requirement=requirement,
            )
            != item["sha256"]
        ):
            raise GroundingContractError("Native v5 source hash is invalid.")

    latest_view = _latest_product_context_view(root)
    completed_at_ns = _nonnegative_integer(value["completed_at_ns"], "completed_at_ns")
    if (
        latest_view.product_requirement != requirement
        or latest_view.tbox_fingerprint != value["tbox_fingerprint"]
        or latest_view.abox_fingerprint != value["abox_fingerprint"]
        or latest_view.assessed_at_ns != completed_at_ns
        or tuple(
            {"ref": item.record_ref, "sha256": item.record_sha256}
            for item in latest_view.typed_bindings
        )
        != typed_refs
    ):
        raise GroundingContractError("Native v5 final ProductContextView is inconsistent.")
    projection_path = _completion_ref_path(
        root,
        str(value["ontology_projection_ref"]),
        prefix=("products", "grounding", "ontology_grounding"),
    )
    selection_path = _completion_ref_path(
        root,
        str(value["resource_selection_ref"]),
        prefix=_RESOURCE_SELECTION_PREFIX,
    )
    projection = _read_json_mapping(projection_path, "OntologyGroundingProposal")
    selection = _read_json_mapping(selection_path, "ResourceSelectionRecord")
    _target_feature, evidence_refs, review = _validated_target_feature(
        root,
        projection,
        product_context=latest_view,
        selection=selection,
        selection_schema_version=3,
    )
    if tuple(item["ref"] for item in source_refs) != evidence_refs:
        raise GroundingContractError(
            "Native v5 sources do not match nested target-feature evidence."
        )
    direct_pairs = (
        ("current_state_iri", "current_state_iri"),
        ("desired_state_iri", "desired_state_iri"),
        ("resource_selection_fingerprint", "fingerprint"),
        ("reachability_check_ref", "reachability_check_ref"),
        ("reachability_check_sha256", "reachability_check_sha256"),
        ("reachability_check_fingerprint", "reachability_check_fingerprint"),
        ("robot_agent_validation_ref", "robot_agent_validation_ref"),
        ("robot_agent_validation_sha256", "robot_agent_validation_sha256"),
        (
            "robot_agent_validation_fingerprint",
            "robot_agent_validation_fingerprint",
        ),
    )
    if any(value[left] != selection[right] for left, right in direct_pairs):
        raise GroundingContractError("Native v5 allocation lineage is inconsistent.")
    if (
        projection.get("semantic_review_ref") != value["semantic_review_ref"]
        or projection.get("semantic_review_sha256") != value["semantic_review_sha256"]
        or review.get("fingerprint") != value["semantic_review_fingerprint"]
    ):
        raise GroundingContractError("Native v5 semantic review lineage is invalid.")
    assignment_ref = _native_assignment_delta_ref_v5(
        root,
        resource_selection_ref=str(value["resource_selection_ref"]),
        reachability_ref=str(value["reachability_check_ref"]),
        validation_ref=str(value["robot_agent_validation_ref"]),
        selected_resource_iri=_required_string(
            selection["selected_resource_iri"],
            "selected_resource_iri",
        ),
    )
    if assignment_ref != value["resource_assignment_delta_ref"]:
        raise GroundingContractError("Native v5 assignment lineage is invalid.")

    contract_path = _completion_ref_path(
        root,
        str(value["typed_grounding_contract_ref"]),
        prefix=("products", "grounding", "completion"),
    )
    contract = _read_json_mapping(contract_path, "TypedGroundingContract v5")
    contract_expected = {
        "schema_version",
        "record_type",
        "requirement_text",
        "ontology_projection_ref",
        "semantic_review_ref",
        "semantic_review_sha256",
        "semantic_review_fingerprint",
        "current_state_iri",
        "desired_state_iri",
        "resource_selection_ref",
        "resource_selection_sha256",
        "resource_selection_fingerprint",
        "reachability_check_ref",
        "reachability_check_sha256",
        "reachability_check_fingerprint",
        "robot_agent_validation_ref",
        "robot_agent_validation_sha256",
        "robot_agent_validation_fingerprint",
        "allocation_label",
        "motion_executed",
        "typed_record_refs",
        "source_refs",
        "tool_call_refs",
        "fingerprint",
    }
    _require_exact_keys(contract, contract_expected, "TypedGroundingContract v5")
    contract_payload = dict(contract)
    contract_fingerprint = contract_payload.pop("fingerprint", None)
    shared_fields = (
        "ontology_projection_ref",
        "semantic_review_ref",
        "semantic_review_sha256",
        "semantic_review_fingerprint",
        "current_state_iri",
        "desired_state_iri",
        "resource_selection_ref",
        "resource_selection_sha256",
        "resource_selection_fingerprint",
        "reachability_check_ref",
        "reachability_check_sha256",
        "reachability_check_fingerprint",
        "robot_agent_validation_ref",
        "robot_agent_validation_sha256",
        "robot_agent_validation_fingerprint",
        "allocation_label",
        "motion_executed",
    )
    if (
        contract.get("schema_version") != 5
        or contract.get("record_type") != "TypedGroundingContract"
        or contract.get("requirement_text") != requirement
        or contract_fingerprint != value["typed_grounding_contract_fingerprint"]
        or not _is_sha256_value(contract_fingerprint)
        or _fingerprint(contract_payload) != contract_fingerprint
        or any(contract[field] != value[field] for field in shared_fields)
        or contract.get("typed_record_refs") != list(typed_refs)
        or contract.get("source_refs") != list(source_refs)
        or contract.get("tool_call_refs") != list(tool_refs)
    ):
        raise GroundingContractError("TypedGroundingContract v5 is inconsistent.")
    return PAContextGroundingCompletionV5(record=dict(value))


def _load_pa_context_grounding_completion_v4(
    root: Path,
    value: Mapping[str, object],
) -> PAContextGroundingCompletionV4:
    expected = {
        "schema_version",
        "record_type",
        "status",
        "product_requirement",
        "completion_turn",
        "decision_ref",
        "ontology_projection_ref",
        "ontology_projection_sha256",
        "semantic_review_ref",
        "semantic_review_sha256",
        "semantic_review_fingerprint",
        "typed_grounding_contract_ref",
        "typed_grounding_contract_sha256",
        "typed_grounding_contract_fingerprint",
        "resource_selection_ref",
        "resource_selection_sha256",
        "resource_assignment_delta_ref",
        "resource_assignment_delta_sha256",
        "tbox_fingerprint",
        "abox_fingerprint",
        "typed_context_refs",
        "source_refs",
        "tool_call_refs",
        "completed_at_ns",
        "fingerprint",
    }
    _require_exact_keys(value, expected, "PAContextGroundingCompletion v4")
    fingerprint = _sha256_string(value["fingerprint"], "fingerprint")
    payload = dict(value)
    payload.pop("fingerprint")
    if (
        value["schema_version"] != 4
        or value["record_type"] != "PAContextGroundingCompletion"
        or value["status"] != "grounding complete"
        or _fingerprint(payload) != fingerprint
    ):
        raise GroundingContractError("PAContextGroundingCompletion v4 is invalid.")
    requirement = _required_string(value["product_requirement"], "product_requirement")
    completion_turn = _positive_integer(value["completion_turn"], "completion_turn")
    decision_path = _completion_ref_path(
        root,
        _required_string(value["decision_ref"], "decision_ref"),
        prefix=("interaction_record",),
    )
    decision = _read_json_mapping(decision_path, "native grounding decision")
    output = decision.get("PA_output")
    if (
        decision.get("turn") != completion_turn
        or decision.get("product_requirement") != requirement
        or decision.get("failure") is not None
        or not isinstance(output, Mapping)
        or output.get("grounding_status") != "complete"
        or output.get("ontology_projection_ref") != value["ontology_projection_ref"]
        or output.get("resource_selection_ref") != value["resource_selection_ref"]
    ):
        raise GroundingContractError("Native v4 grounding decision reference is invalid.")

    hashed_fields = (
        ("ontology_projection_ref", "ontology_projection_sha256"),
        ("semantic_review_ref", "semantic_review_sha256"),
        ("typed_grounding_contract_ref", "typed_grounding_contract_sha256"),
        ("resource_selection_ref", "resource_selection_sha256"),
        ("resource_assignment_delta_ref", "resource_assignment_delta_sha256"),
    )
    for ref_field, hash_field in hashed_fields:
        path = _completion_ref_path(
            root,
            _required_string(value[ref_field], ref_field),
            prefix=None,
        )
        if _sha256_path(path) != _sha256_string(value[hash_field], hash_field):
            raise GroundingContractError(f"Native v4 {ref_field} hash is invalid.")

    typed_refs = _hashed_ref_tuple(value["typed_context_refs"], "typed_context_refs")
    source_refs = _hashed_ref_tuple(value["source_refs"], "source_refs")
    tool_refs = _hashed_ref_tuple(value["tool_call_refs"], "tool_call_refs")
    for item in (*typed_refs, *tool_refs):
        path = _completion_ref_path(root, item["ref"], prefix=None)
        if _sha256_path(path) != item["sha256"]:
            raise GroundingContractError("Native v4 pinned record changed.")
    for item in source_refs:
        if (
            _native_source_hash(
                root,
                item["ref"],
                product_requirement=requirement,
            )
            != item["sha256"]
        ):
            raise GroundingContractError("Native v4 source hash is invalid.")

    latest_view = _latest_product_context_view(root)
    completed_at_ns = _nonnegative_integer(value["completed_at_ns"], "completed_at_ns")
    if (
        latest_view.product_requirement != requirement
        or latest_view.tbox_fingerprint != value["tbox_fingerprint"]
        or latest_view.abox_fingerprint != value["abox_fingerprint"]
        or latest_view.assessed_at_ns != completed_at_ns
        or tuple(
            {"ref": item.record_ref, "sha256": item.record_sha256}
            for item in latest_view.typed_bindings
        )
        != typed_refs
    ):
        raise GroundingContractError("Native v4 final ProductContextView is inconsistent.")
    projection_path = _completion_ref_path(
        root,
        str(value["ontology_projection_ref"]),
        prefix=("products", "grounding", "ontology_grounding"),
    )
    selection_path = _completion_ref_path(
        root,
        str(value["resource_selection_ref"]),
        prefix=_RESOURCE_SELECTION_PREFIX,
    )
    projection = _read_json_mapping(projection_path, "OntologyGroundingProposal")
    selection = _read_json_mapping(selection_path, "ResourceSelectionRecord")
    _target_feature, evidence_refs, review = _validated_target_feature(
        root,
        projection,
        product_context=latest_view,
        selection=selection,
    )
    if tuple(item["ref"] for item in source_refs) != evidence_refs:
        raise GroundingContractError(
            "Native v4 sources do not match nested target-feature evidence."
        )
    if (
        projection.get("semantic_review_ref") != value["semantic_review_ref"]
        or projection.get("semantic_review_sha256") != value["semantic_review_sha256"]
        or projection.get("semantic_review_fingerprint") != value["semantic_review_fingerprint"]
        or review.get("fingerprint") != value["semantic_review_fingerprint"]
    ):
        raise GroundingContractError("Native v4 semantic review lineage is invalid.")

    contract_path = _completion_ref_path(
        root,
        str(value["typed_grounding_contract_ref"]),
        prefix=("products", "grounding", "completion"),
    )
    contract = _read_json_mapping(contract_path, "TypedGroundingContract v4")
    contract_expected = {
        "schema_version",
        "record_type",
        "requirement_text",
        "ontology_projection_ref",
        "semantic_review_ref",
        "semantic_review_sha256",
        "semantic_review_fingerprint",
        "typed_record_refs",
        "source_refs",
        "tool_call_refs",
        "fingerprint",
    }
    _require_exact_keys(contract, contract_expected, "TypedGroundingContract v4")
    contract_fingerprint = _sha256_string(
        value["typed_grounding_contract_fingerprint"],
        "typed_grounding_contract_fingerprint",
    )
    contract_payload = dict(contract)
    persisted_contract_fingerprint = contract_payload.pop("fingerprint", None)
    if (
        contract.get("schema_version") != 4
        or contract.get("record_type") != "TypedGroundingContract"
        or persisted_contract_fingerprint != contract_fingerprint
        or _fingerprint(contract_payload) != contract_fingerprint
        or contract.get("requirement_text") != requirement
        or contract.get("ontology_projection_ref") != value["ontology_projection_ref"]
        or contract.get("semantic_review_ref") != value["semantic_review_ref"]
        or contract.get("semantic_review_sha256") != value["semantic_review_sha256"]
        or contract.get("semantic_review_fingerprint") != value["semantic_review_fingerprint"]
        or contract.get("typed_record_refs") != list(typed_refs)
        or contract.get("source_refs") != list(source_refs)
        or contract.get("tool_call_refs") != list(tool_refs)
    ):
        raise GroundingContractError("TypedGroundingContract v4 is inconsistent.")
    expected_assignment_ref = _native_assignment_delta_ref(
        root,
        resource_selection_ref=str(value["resource_selection_ref"]),
        selected_resource_iri=_required_string(
            selection.get("selected_resource_iri"), "selected_resource_iri"
        ),
    )
    if expected_assignment_ref != value["resource_assignment_delta_ref"]:
        raise GroundingContractError("Native v4 resource assignment lineage is invalid.")
    return PAContextGroundingCompletionV4(record=dict(value))


def _load_pa_context_grounding_completion_v3(
    root: Path,
    value: Mapping[str, object],
) -> PAContextGroundingCompletionV3:
    expected = {
        "schema_version",
        "record_type",
        "status",
        "product_requirement",
        "completion_turn",
        "decision_ref",
        "ontology_projection_ref",
        "ontology_projection_sha256",
        "typed_grounding_contract_ref",
        "typed_grounding_contract_sha256",
        "typed_grounding_contract_fingerprint",
        "resource_selection_ref",
        "resource_selection_sha256",
        "resource_assignment_delta_ref",
        "resource_assignment_delta_sha256",
        "tbox_fingerprint",
        "abox_fingerprint",
        "typed_context_refs",
        "source_refs",
        "tool_call_refs",
        "completed_at_ns",
        "fingerprint",
    }
    _require_exact_keys(value, expected, "PAContextGroundingCompletion v3")
    fingerprint = _sha256_string(value["fingerprint"], "fingerprint")
    payload = dict(value)
    payload.pop("fingerprint")
    if (
        value["schema_version"] != 3
        or value["record_type"] != "PAContextGroundingCompletion"
        or value["status"] != "grounding complete"
        or _fingerprint(payload) != fingerprint
    ):
        raise GroundingContractError("PAContextGroundingCompletion v3 is invalid.")
    requirement = _required_string(value["product_requirement"], "product_requirement")
    completion_turn = _positive_integer(value["completion_turn"], "completion_turn")
    decision_path = _completion_ref_path(
        root,
        _required_string(value["decision_ref"], "decision_ref"),
        prefix=("interaction_record",),
    )
    decision = _read_json_mapping(decision_path, "native grounding decision")
    output = decision.get("PA_output")
    if (
        decision.get("turn") != completion_turn
        or decision.get("product_requirement") != requirement
        or decision.get("failure") is not None
        or not isinstance(output, Mapping)
        or output.get("grounding_status") != "complete"
    ):
        raise GroundingContractError("Native grounding decision reference is invalid.")

    hashed_fields = (
        ("ontology_projection_ref", "ontology_projection_sha256"),
        ("typed_grounding_contract_ref", "typed_grounding_contract_sha256"),
        ("resource_selection_ref", "resource_selection_sha256"),
        ("resource_assignment_delta_ref", "resource_assignment_delta_sha256"),
    )
    for ref_field, hash_field in hashed_fields:
        path = _completion_ref_path(
            root,
            _required_string(value[ref_field], ref_field),
            prefix=None,
        )
        if _sha256_path(path) != _sha256_string(value[hash_field], hash_field):
            raise GroundingContractError(f"Native completion {ref_field} hash is invalid.")

    typed_refs = _hashed_ref_tuple(value["typed_context_refs"], "typed_context_refs")
    source_refs = _hashed_ref_tuple(value["source_refs"], "source_refs")
    tool_refs = _hashed_ref_tuple(value["tool_call_refs"], "tool_call_refs")
    for item in (*typed_refs, *tool_refs):
        path = _completion_ref_path(root, item["ref"], prefix=None)
        if _sha256_path(path) != item["sha256"]:
            raise GroundingContractError("Native completion pinned record changed.")
    for item in source_refs:
        if (
            _native_source_hash(root, item["ref"], product_requirement=requirement)
            != item["sha256"]
        ):
            raise GroundingContractError("Native completion source hash is invalid.")

    contract_path = _completion_ref_path(
        root,
        str(value["typed_grounding_contract_ref"]),
        prefix=("products", "grounding", "completion"),
    )
    contract = _read_json_mapping(contract_path, "TypedGroundingContract v3")
    contract_fingerprint = _sha256_string(
        value["typed_grounding_contract_fingerprint"],
        "typed_grounding_contract_fingerprint",
    )
    contract_payload = dict(contract)
    persisted_contract_fingerprint = contract_payload.pop("fingerprint", None)
    if (
        contract.get("schema_version") != 3
        or contract.get("record_type") != "TypedGroundingContract"
        or persisted_contract_fingerprint != contract_fingerprint
        or _fingerprint(contract_payload) != contract_fingerprint
        or contract.get("requirement_text") != requirement
        or contract.get("ontology_projection_ref") != value["ontology_projection_ref"]
        or contract.get("typed_record_refs") != list(typed_refs)
        or contract.get("source_refs") != list(source_refs)
        or contract.get("tool_call_refs") != list(tool_refs)
    ):
        raise GroundingContractError("TypedGroundingContract v3 is inconsistent.")
    latest_view = _latest_product_context_view(root)
    completed_at_ns = _nonnegative_integer(value["completed_at_ns"], "completed_at_ns")
    if (
        latest_view.product_requirement != requirement
        or latest_view.tbox_fingerprint != value["tbox_fingerprint"]
        or latest_view.abox_fingerprint != value["abox_fingerprint"]
        or latest_view.assessed_at_ns != completed_at_ns
        or tuple(
            {"ref": item.record_ref, "sha256": item.record_sha256}
            for item in latest_view.typed_bindings
        )
        != typed_refs
    ):
        raise GroundingContractError("Native completion final ProductContextView is inconsistent.")
    return PAContextGroundingCompletionV3(record=dict(value))


def _load_pa_context_grounding_completion_v2(  # noqa: C901
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
    session = GroundingSession.from_mapping(_read_json_mapping(session_path, "GroundingSession"))
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
    proposal_session = _proposal_grounding_session(root, projection)
    if (
        projection.get("schema_version") not in _ONTOLOGY_PROPOSAL_SCHEMA_VERSIONS
        or projection.get("status") != "accepted"
        or proposal_session.requirement_text != completion.product_requirement
        or proposal_session.next_action.action != "propose_grounding"
        or proposal_session.status != "ready_for_ontology"
    ):
        raise GroundingContractError("Phase 3.5 v2 ontology projection is invalid.")
    _validate_completion_lineage(session, proposal_session)

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
    for item in completion.source_refs:
        if _grounding_source_hash(root, item["ref"], session) != item["sha256"]:
            raise GroundingContractError("Phase 3.5 v2 source hash is invalid.")
    resource_selection_path = _completion_ref_path(
        root,
        completion.resource_selection_ref,
        prefix=_RESOURCE_SELECTION_PREFIX,
    )
    if _sha256_path(resource_selection_path) != completion.resource_selection_sha256:
        raise GroundingContractError("Phase 3.5 v2 ResourceSelectionRecord hash is invalid.")
    resource_assignment_delta_path = _completion_ref_path(
        root,
        completion.resource_assignment_delta_ref,
        prefix=("products", "grounding", "ontology"),
    )
    if _sha256_path(resource_assignment_delta_path) != completion.resource_assignment_delta_sha256:
        raise GroundingContractError("Phase 3.5 v2 resource assignment delta hash is invalid.")
    latest_view = _latest_product_context_view(root)
    if (
        latest_view.product_requirement != completion.product_requirement
        or latest_view.tbox_fingerprint != completion.tbox_fingerprint
        or latest_view.abox_fingerprint != completion.abox_fingerprint
        or latest_view.assessed_at_ns != completion.completed_at_ns
        or tuple(
            {"ref": item.record_ref, "sha256": item.record_sha256}
            for item in latest_view.typed_bindings
        )
        != completion.typed_context_refs
    ):
        raise GroundingContractError("Phase 3.5 v2 final ProductContextView is inconsistent.")
    validated_selection_ref, validated_assignment_delta_ref = _validate_completion_assignment(
        root, latest_view
    )
    if (
        validated_selection_ref != completion.resource_selection_ref
        or validated_assignment_delta_ref != completion.resource_assignment_delta_ref
    ):
        raise GroundingContractError("Phase 3.5 v2 resource assignment lineage is inconsistent.")
    return completion


def _validate_completion_lineage(
    session: GroundingSession,
    proposal_session: GroundingSession,
) -> None:
    """Require a complete host session to descend from its exact proposal session."""
    if (
        session.next_action.action != "propose_grounding"
        or proposal_session.next_action.action != "propose_grounding"
        or proposal_session.status != "ready_for_ontology"
        or proposal_session.revision >= session.revision
        or session.attempted_actions[: len(proposal_session.attempted_actions)]
        != proposal_session.attempted_actions
    ):
        raise GroundingContractError(
            "PAContextGroundingCompletion proposal/session lineage is invalid."
        )


def _validate_completion_assignment(
    root: Path,
    product_context: ProductContextView,
) -> tuple[str, str]:
    """Require the exact post-selection process-execution slice in the final ABox."""
    execution_iri = f"{product_context.interaction_namespace}process_execution_0001"
    specification_iri = f"{product_context.interaction_namespace}specification_1"
    expected = {
        (
            specification_iri,
            "hasProcessExecution",
            execution_iri,
        ),
        (
            execution_iri,
            "type",
            "processExecution",
        ),
        (
            execution_iri,
            "runsProcess",
            _ASSEMBLY_PROCESS_IRI,
        ),
    }
    actual: set[tuple[str, str, str]] = set()
    assigned_resource: str | None = None
    execution_assertion_count = 0
    for assertion in product_context.assertions:
        subject = assertion.get("subject")
        predicate = assertion.get("predicate")
        object_value = assertion.get("object")
        if (
            not isinstance(subject, str)
            or not isinstance(predicate, str)
            or not isinstance(object_value, Mapping)
            or object_value.get("kind") != "iri"
            or not isinstance(object_value.get("value"), str)
        ):
            continue
        local_predicate = _iri_local_name(predicate)
        object_iri = str(object_value["value"])
        if local_predicate in {
            "hasProcessExecution",
            "runsProcess",
            "runsOnResource",
        } or (
            subject == execution_iri
            and local_predicate == "type"
            and _iri_local_name(object_iri) == "processExecution"
        ):
            execution_assertion_count += 1
        actual.add(
            (subject, local_predicate, _iri_local_name(object_iri))
            if local_predicate == "type"
            else (subject, local_predicate, object_iri)
        )
        if subject == execution_iri and local_predicate == "runsOnResource":
            assigned_resource = object_iri
    if (
        not expected.issubset(actual)
        or assigned_resource
        not in {resource_iri for _symbol, resource_iri in _PREDEFINED_RESOURCES}
        or execution_assertion_count != 4
    ):
        raise GroundingContractError(
            "PAContextGroundingCompletion requires the exact host resource assignment."
        )
    selection_ref = _assignment_selection_ref(root, product_context)
    _validate_completion_resource_selection(
        root,
        product_context,
        selection_ref=selection_ref,
        assigned_resource=str(assigned_resource),
    )
    assignment_delta_ref = (
        f"products/grounding/ontology/delta_{product_context.delta_count:04d}.json"
    )
    return selection_ref, assignment_delta_ref


def _assignment_selection_ref(root: Path, product_context: ProductContextView) -> str:
    """Return the sole selection record cited by the host assignment delta."""
    delta_path = (
        root / "products/grounding/ontology" / f"delta_{product_context.delta_count:04d}.json"
    )
    delta = _read_json_mapping(delta_path, "resource assignment delta")
    assertions = delta.get("assertions")
    if (
        set(delta) != _DELTA_KEYS
        or delta.get("schema_version") != 1
        or delta.get("delta_number") != product_context.delta_count
        or delta.get("producer") != "resource_grounding_host"
        or not isinstance(assertions, list)
        or len(assertions) != 4
        or delta.get("uncertainty") != []
        or delta.get("unresolved_evidence_needs") != []
        or delta.get("typed_context_refs") != []
    ):
        raise GroundingContractError(
            "PAContextGroundingCompletion requires the host resource-selection delta."
        )
    selection_refs: set[str] = set()
    for assertion in assertions:
        evidence_refs = assertion.get("evidence_refs") if isinstance(assertion, Mapping) else None
        if (
            not isinstance(evidence_refs, list)
            or len(evidence_refs) != 1
            or not isinstance(evidence_refs[0], str)
        ):
            raise GroundingContractError(
                "Every assignment assertion must cite one ResourceSelectionRecord."
            )
        selection_refs.add(evidence_refs[0])
    if len(selection_refs) != 1:
        raise GroundingContractError(
            "Assignment assertions do not share one ResourceSelectionRecord."
        )
    return next(iter(selection_refs))


def _semantic_feature_iri(product_context: ProductContextView) -> str:
    """Return the sole feature joining specification and predefined process."""
    specification_iri = f"{product_context.interaction_namespace}specification_1"
    defined: set[str] = set()
    realized: set[str] = set()
    typed: set[str] = set()
    for assertion in product_context.assertions:
        subject = assertion.get("subject")
        predicate = assertion.get("predicate")
        object_value = assertion.get("object")
        if (
            not isinstance(subject, str)
            or not isinstance(predicate, str)
            or not isinstance(object_value, Mapping)
            or object_value.get("kind") != "iri"
            or not isinstance(object_value.get("value"), str)
        ):
            continue
        object_iri = str(object_value["value"])
        local_predicate = _iri_local_name(predicate)
        if subject == specification_iri and local_predicate == "defines":
            defined.add(object_iri)
        elif subject == _ASSEMBLY_PROCESS_IRI and local_predicate == "realizes":
            realized.add(object_iri)
        elif local_predicate == "type" and _iri_local_name(object_iri) == "feature":
            typed.add(subject)
    joined = defined & realized & typed
    if len(joined) != 1:
        raise GroundingContractError(
            "PAContextGroundingCompletion requires one semantic feature join."
        )
    return next(iter(joined))


def _validate_completion_resource_selection(  # noqa: C901
    root: Path,
    product_context: ProductContextView,
    *,
    selection_ref: str,
    assigned_resource: str,
) -> None:
    """Validate the selection decision and its exact accepted world-pose input."""
    selection_path = _completion_ref_path(
        root,
        selection_ref,
        prefix=_RESOURCE_SELECTION_PREFIX,
    )
    selection = _read_json_mapping(selection_path, "ResourceSelectionRecord")
    selection_number = selection.get("selection_number")
    feature_iri = _semantic_feature_iri(product_context)
    if (
        set(selection) != _RESOURCE_SELECTION_KEYS
        or selection.get("schema_version") != 1
        or selection.get("record_type") != "ResourceSelectionRecord"
        or isinstance(selection_number, bool)
        or not isinstance(selection_number, int)
        or selection_number < 1
        or Path(selection_ref).parts
        != (
            *_RESOURCE_SELECTION_PREFIX,
            f"selection_{selection_number:04d}",
            _RESOURCE_SELECTION_RECORD_NAME,
        )
    ):
        raise GroundingContractError(
            "PAContextGroundingCompletion ResourceSelectionRecord identity is invalid."
        )
    fingerprint = selection.get("fingerprint")
    fingerprint_payload = dict(selection)
    fingerprint_payload.pop("fingerprint")
    if (
        not isinstance(fingerprint, str)
        or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        or _fingerprint(fingerprint_payload) != fingerprint
    ):
        raise GroundingContractError(
            "PAContextGroundingCompletion ResourceSelectionRecord fingerprint is invalid."
        )

    specification_iri = f"{product_context.interaction_namespace}specification_1"
    candidate_resources = [item[1] for item in _PREDEFINED_RESOURCES]
    if (
        selection.get("specification_iri") != specification_iri
        or selection.get("feature_iri") != feature_iri
        or selection.get("process_iri") != _ASSEMBLY_PROCESS_IRI
        or selection.get("candidate_resource_iris") != candidate_resources
        or selection.get("required_record_type") != "RobotFramePoseRecord"
        or selection.get("target_frame") != "world"
        or selection.get("tbox_fingerprint") != product_context.tbox_fingerprint
        or not _is_sha256_value(selection.get("registry_fingerprint"))
        or not _is_sha256_value(selection.get("workcell_fingerprint"))
        or selection.get("selection_policy") != _RESOURCE_SELECTION_POLICY
        or selection.get("selected_resource_iri") != assigned_resource
    ):
        raise GroundingContractError(
            "PAContextGroundingCompletion ResourceSelectionRecord does not match "
            "the final assignment."
        )
    _validate_completion_candidate_verdicts(selection, assigned_resource)

    pose_ref = selection.get("pose_record_ref")
    pose_sha256 = selection.get("pose_record_sha256")
    observation_timestamp_ns = selection.get("observation_timestamp_ns")
    if (
        not isinstance(pose_ref, str)
        or not _is_sha256_value(pose_sha256)
        or isinstance(observation_timestamp_ns, bool)
        or not isinstance(observation_timestamp_ns, int)
        or observation_timestamp_ns < 0
    ):
        raise GroundingContractError("ResourceSelectionRecord world-pose reference is invalid.")
    pose_path = _completion_ref_path(
        root,
        pose_ref,
        prefix=("products", "grounding"),
    )
    if _sha256_path(pose_path) != pose_sha256:
        raise GroundingContractError("ResourceSelectionRecord world-pose hash is invalid.")
    pose_bindings = [
        binding
        for binding in product_context.typed_bindings
        if binding.record_type == "RobotFramePoseRecord"
        and binding.record_ref == pose_ref
        and binding.record_sha256 == pose_sha256
        and binding.status == "accepted"
        and binding.frame == "world"
        and binding.observed_at_ns == observation_timestamp_ns
    ]
    if len(pose_bindings) != 1:
        raise GroundingContractError(
            "ResourceSelectionRecord is not pinned to the final accepted world pose."
        )
    pose_record = _read_json_mapping(pose_path, "RobotFramePoseRecord")
    if _validate_embedded_hash_refs(pose_record, root) < 1:
        raise GroundingContractError(
            "RobotFramePoseRecord requires a pinned source-evidence chain."
        )


def _validate_completion_candidate_verdicts(
    selection: Mapping[str, object],
    assigned_resource: str,
) -> None:
    """Require exact ordered candidate verdicts and first-reachable selection."""
    candidates = selection.get("candidate_reach_evidence")
    if not isinstance(candidates, list) or len(candidates) != len(_PREDEFINED_RESOURCES):
        raise GroundingContractError("ResourceSelectionRecord candidate verdicts are invalid.")
    reachable: list[Mapping[str, object]] = []
    for candidate, (symbol, resource_iri) in zip(
        candidates,
        _PREDEFINED_RESOURCES,
        strict=True,
    ):
        if (
            not isinstance(candidate, Mapping)
            or set(candidate) != _CANDIDATE_REACH_KEYS
            or candidate.get("resource_symbol") != symbol
            or candidate.get("resource_iri") != resource_iri
            or not isinstance(candidate.get("resource_jid"), str)
            or not candidate.get("resource_jid")
            or not isinstance(candidate.get("manifest_ref"), str)
            or not candidate.get("manifest_ref")
            or not _is_sha256_value(candidate.get("manifest_sha256"))
            or not isinstance(candidate.get("execution_mode"), str)
            or not candidate.get("execution_mode")
            or not isinstance(candidate.get("reachable"), bool)
            or not isinstance(candidate.get("verdicts"), list)
            or not all(isinstance(item, str) and item for item in candidate["verdicts"])
        ):
            raise GroundingContractError("ResourceSelectionRecord candidate verdict is invalid.")
        if candidate["reachable"] is True:
            reachable.append(candidate)
    if not reachable or reachable[0].get("resource_iri") != assigned_resource:
        raise GroundingContractError(
            "ResourceSelectionRecord violates first-reachable registry order."
        )
    selected = reachable[0]
    if (
        selection.get("selected_resource_symbol") != selected.get("resource_symbol")
        or selection.get("selected_resource_jid") != selected.get("resource_jid")
        or selection.get("selected_execution_mode") != selected.get("execution_mode")
    ):
        raise GroundingContractError(
            "ResourceSelectionRecord selected fields do not match its verdict."
        )


def _is_sha256_value(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _latest_product_context_view(root: Path) -> ProductContextView:
    paths = sorted((root / _VIEW_ROOT).glob("view_*.json"))
    if not paths:
        raise GroundingContractError(
            "PAContextGroundingCompletion requires a final ProductContextView."
        )
    return ProductContextView.from_mapping(
        _read_json_mapping(paths[-1], "final ProductContextView")
    )


def _iri_local_name(iri: str) -> str:
    return iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _proposal_grounding_session(
    root: Path,
    proposal: Mapping[str, object],
) -> GroundingSession:
    """Load the exact session revision that authored an ontology proposal."""
    revision = proposal.get("session_revision")
    fingerprint = proposal.get("session_fingerprint")
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision < 1
        or not isinstance(fingerprint, str)
    ):
        raise GroundingContractError("Ontology projection session identity is invalid.")
    path = root / _SESSION_ROOT / f"revision_{revision:04d}.json"
    session = GroundingSession.from_mapping(
        _read_json_mapping(path, "ontology proposal GroundingSession")
    )
    if session.revision != revision or session.fingerprint != fingerprint:
        raise GroundingContractError(
            "Ontology projection does not match its exact GroundingSession."
        )
    return session


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
    if not isinstance(record, Mapping):
        raise GroundingContractError(f"Typed context record is invalid: {record_ref}.")
    record_type = _required_symbol(record.get("record_type"), "typed record_type")
    expected_schema_versions = _TYPED_RECORD_SCHEMA_VERSIONS.get(
        record_type,
        frozenset({1}),
    )
    schema_version = record.get("schema_version")
    if schema_version not in expected_schema_versions:
        raise GroundingContractError(f"Typed context record is invalid: {record_ref}.")
    embedded_evidence_stale = False
    try:
        embedded_hash_ref_count = _validate_embedded_hash_refs(
            record,
            interaction_root,
        )
    except _EmbeddedEvidenceStateError:
        embedded_evidence_stale = True
        embedded_hash_ref_count = 1
    if (
        record_type
        in {
            "RobotFramePoseRecord",
            "RobotFrameLocationRecord",
        }
        and embedded_hash_ref_count < 1
    ):
        raise GroundingContractError(
            "RobotFramePoseRecord requires a pinned source-evidence chain."
        )
    record_producer = _required_symbol(record.get("producer"), "typed producer")
    if record_producer != producer:
        raise GroundingContractError(f"Typed context producer mismatch for {record_ref}.")
    status = "stale" if embedded_evidence_stale else _binding_status(record_type, record)
    if status not in _BINDING_STATUSES:
        raise GroundingContractError(f"Typed context status is invalid: {record_ref}.")
    evidence_refs = _record_evidence_refs(record)
    provenance_value = record.get("provenance", {})
    provenance = dict(provenance_value) if isinstance(provenance_value, Mapping) else {}
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
    if record_type == "RobotFrameLocationRecord":
        return str(record.get("robot_frame_conversion"))
    if record_type in {
        "CADMeshRecord",
        "ColoredPointCloudSetRecord",
        "ObservationCandidateReview",
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


def _validate_embedded_hash_refs(value: object, interaction_root: Path) -> int:
    """Validate nested local hash references and return their exact count."""
    count = 0
    if isinstance(value, Mapping):
        ref = value.get("ref")
        sha256 = value.get("sha256")
        if isinstance(ref, str) or isinstance(sha256, str):
            if not isinstance(ref, str) or not isinstance(sha256, str):
                raise GroundingContractError("Typed record artifact hash reference is incomplete.")
            relative = Path(ref)
            if relative.is_absolute() or ".." in relative.parts:
                raise GroundingContractError("Typed record artifact ref is invalid.")
            path = (interaction_root / relative).resolve()
            try:
                path.relative_to(interaction_root)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise _EmbeddedEvidenceStateError(
                    f"Typed record artifact could not be validated: {ref}."
                ) from exc
            except ValueError as exc:
                raise GroundingContractError(
                    f"Typed record artifact ref is invalid: {ref}."
                ) from exc
            if digest != sha256:
                raise _EmbeddedEvidenceStateError(
                    f"Typed record artifact hash does not match: {ref}."
                )
            count += 1
        for item in value.values():
            count += _validate_embedded_hash_refs(item, interaction_root)
    elif isinstance(value, list):
        for item in value:
            count += _validate_embedded_hash_refs(item, interaction_root)
    return count


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


def _native_assignment_delta_ref(
    root: Path,
    *,
    resource_selection_ref: str,
    selected_resource_iri: str,
) -> str:
    """Find the one host assignment delta pinned to the native selection."""
    matches: list[Path] = []
    for path in sorted((root / "products/grounding/ontology").glob("delta_*.json")):
        delta = _read_json_mapping(path, "resource assignment delta")
        if delta.get("producer") != "resource_grounding_host":
            continue
        assertions = delta.get("assertions")
        if not isinstance(assertions, list):
            continue
        has_selected_resource = any(
            isinstance(assertion, Mapping)
            and _iri_local_name(str(assertion.get("predicate"))) == "runsOnResource"
            and isinstance(assertion.get("object"), Mapping)
            and assertion["object"].get("value") == selected_resource_iri
            and assertion.get("evidence_refs") == [resource_selection_ref]
            for assertion in assertions
        )
        if has_selected_resource:
            matches.append(path)
    if len(matches) != 1:
        raise GroundingContractError(
            "Native completion requires one exact resource assignment delta."
        )
    return matches[0].relative_to(root).as_posix()


def _native_assignment_delta_ref_v5(
    root: Path,
    *,
    resource_selection_ref: str,
    reachability_ref: str,
    validation_ref: str,
    selected_resource_iri: str,
) -> str:
    """Find the one host delta citing the complete accepted allocation lineage."""
    expected_refs = [
        resource_selection_ref,
        reachability_ref,
        validation_ref,
    ]
    matches: list[Path] = []
    for path in sorted((root / "products/grounding/ontology").glob("delta_*.json")):
        delta = _read_json_mapping(path, "resource assignment delta")
        assertions = delta.get("assertions")
        if (
            delta.get("producer") != "resource_grounding_host"
            or not isinstance(assertions, list)
            or len(assertions) != 4
            or not all(
                isinstance(assertion, Mapping) and assertion.get("evidence_refs") == expected_refs
                for assertion in assertions
            )
        ):
            continue
        has_selected_resource = any(
            _iri_local_name(str(assertion.get("predicate"))) == "runsOnResource"
            and isinstance(assertion.get("object"), Mapping)
            and assertion["object"].get("value") == selected_resource_iri
            for assertion in assertions
            if isinstance(assertion, Mapping)
        )
        if has_selected_resource:
            matches.append(path)
    if len(matches) != 1:
        raise GroundingContractError(
            "PA allocation completion requires one exact assignment delta."
        )
    return matches[0].relative_to(root).as_posix()


def _native_source_hash(
    root: Path,
    source_ref: str,
    *,
    product_requirement: str,
) -> str:
    """Hash one citation from the native PA evidence boundary."""
    if source_ref == "requirement_0001":
        return hashlib.sha256(product_requirement.encode("utf-8")).hexdigest()
    observation_manifest = root / "products" / "observations" / source_ref / "manifest.json"
    if observation_manifest.is_file():
        return _sha256_path(observation_manifest)
    relative = Path(source_ref)
    if not relative.is_absolute() and ".." not in relative.parts:
        local_path = (root / relative).resolve()
        try:
            local_path.relative_to(root)
        except ValueError:
            local_path = Path()
        if local_path.is_file():
            return _sha256_path(local_path)
    context_ref = source_ref.split("#page=", 1)[0]
    try:
        return str(approved_document_metadata(context_ref)["source_sha256"])
    except (OSError, TypeError, ValueError):
        pass
    try:
        return hashlib.sha256(approved_cad_path(context_ref).read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError) as exc:
        raise GroundingContractError(
            f"Native completion source is not approved: {source_ref}."
        ) from exc


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
    return tuple(constructor(_required_mapping(item, f"{label} item")) for item in value)


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
                "sha256": _sha256_string(record["sha256"], f"{label} sha256"),
            }
        )
    refs = [item["ref"] for item in result]
    if len(set(refs)) != len(refs):
        raise GroundingContractError(f"{label} must not contain duplicate refs.")
    return tuple(result)


def _validate_session_action_status(
    status: str,
    next_action: GroundingNextAction,
) -> None:
    expected = {
        "retrieve": {"waiting_for_evidence"},
        "inspect": {"waiting_for_evidence"},
        "ask_user": {"waiting_for_user"},
        "propose_grounding": {"ready_for_ontology", "complete"},
        "incomplete": {"incomplete", "ontology_gap"},
    }[next_action.action]
    if status not in expected:
        raise GroundingContractError("GroundingSession status does not match its next action.")


def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GroundingContractError(f"{label} must be an object.")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
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

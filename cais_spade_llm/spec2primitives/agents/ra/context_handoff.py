"""Activate one selected RobotAgent and persist its current composition context."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingContractError,
    PAContextGroundingCompletionV4,
    PAContextGroundingCompletionV5,
    PAContextGroundingCompletionV6,
    load_pa_context_grounding_completion,
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SNAPSHOT_NAME_PATTERN = re.compile(r"snapshot_(\d{4})\.json")
_SELECTION_NAME_PATTERN = re.compile(r"selection_(\d{4})")
_DELTA_NAME_PATTERN = re.compile(r"delta_(\d{4})\.json")

_ASSIGNMENT_ROOT = ("composition", "selected_ra_assignments")
_ASSIGNMENT_NAME = "assignment_0001.json"
_RESOURCE_SELECTION_PREFIX = ("products", "grounding", "resource_selection")
_ONTOLOGY_PREFIX = ("products", "grounding", "ontology")
_ROBOT_STATE_DIRECTORY = "robot_state"
_PRIMITIVE_CATALOG_DIRECTORY = "primitive_catalog_snapshot"

_PPR_HAS_PROCESS_EXECUTION = "http://PAonto.com#hasProcessExecution"
_PPR_PROCESS_EXECUTION = "http://PAonto.com#processExecution"
_PPR_RUNS_PROCESS = "http://PAonto.com#runsProcess"
_PPR_RUNS_ON_RESOURCE = "http://PAonto.com#runsOnResource"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

_SELECTION_KEYS = frozenset(
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
)
_CANDIDATE_KEYS = frozenset(
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
_SELECTION_V3_KEYS = frozenset(
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
_SELECTION_V4_KEYS = frozenset(
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
_ASSERTION_KEYS = frozenset({"subject", "predicate", "object", "evidence_refs"})
_CATALOG_REQUIRED_KEYS = frozenset(
    {
        "primitive_symbol",
        "operation_description",
        "typed_parameters",
        "typed_results",
        "invocation_binding",
        "truthful_limits",
        "direct_evidence",
        "evaluator_endpoints",
    }
)
_CATALOG_OPTIONAL_KEYS = frozenset({"conditions", "effects"})


class RAContextHandoffError(ValueError):
    """Raised when the Phase 5.1 assignment or RA context is invalid."""


@dataclass(frozen=True)
class SelectedRAAssignmentEnvelope:
    """Hold the minimum verified PA assignment delivered to one selected RA."""

    product_requirement: str
    specification_iri: str
    feature_iri: str
    process_iri: str
    process_symbol: str | None
    selected_resource_iri: str
    selected_resource_jid: str
    selected_execution_mode: str
    selected_resource_symbol: str | None
    pa_context_grounding_completion_ref: str
    pa_context_grounding_completion_sha256: str
    pa_context_grounding_completion_fingerprint: str
    resource_selection_ref: str
    resource_selection_sha256: str
    resource_selection_fingerprint: str
    resource_assignment_delta_ref: str
    resource_assignment_delta_sha256: str
    current_state_iri: str | None
    desired_state_iri: str | None
    reachability_check_ref: str | None
    reachability_check_sha256: str | None
    reachability_check_fingerprint: str | None
    robot_agent_validation_ref: str | None
    robot_agent_validation_sha256: str | None
    robot_agent_validation_fingerprint: str | None
    allocation_label: str | None
    motion_executed: bool | None
    current_state_evidence: Mapping[str, object] | None
    desired_state_evidence: Mapping[str, object] | None
    registry_snapshot_ref: str | None
    registry_snapshot_sha256: str | None
    registry_snapshot_fingerprint: str | None
    workcell_snapshot_ref: str | None
    workcell_snapshot_sha256: str | None
    workcell_snapshot_fingerprint: str | None
    evidence_presentation_ref: str | None
    evidence_presentation_sha256: str | None
    evidence_presentation_fingerprint: str | None
    allocation_presentation_ref: str | None
    allocation_presentation_sha256: str | None
    allocation_presentation_fingerprint: str | None
    validation_scope: str | None
    checked_constraints: tuple[str, ...] | None
    unvalidated_constraints: tuple[str, ...] | None
    fingerprint: str

    def to_record(self) -> dict[str, object]:
        """Return the exact JSON-safe assignment record."""
        schema_version = (
            3
            if self.process_symbol is not None
            else (2 if self.current_state_iri is not None else 1)
        )
        record: dict[str, object] = {
            "schema_version": schema_version,
            "record_type": "SelectedRAAssignmentEnvelope",
            "product_requirement": self.product_requirement,
            "specification_iri": self.specification_iri,
            "feature_iri": self.feature_iri,
            "process_iri": self.process_iri,
            "selected_resource_iri": self.selected_resource_iri,
            "selected_resource_jid": self.selected_resource_jid,
            "selected_execution_mode": self.selected_execution_mode,
            "pa_context_grounding_completion_ref": (self.pa_context_grounding_completion_ref),
            "pa_context_grounding_completion_sha256": (self.pa_context_grounding_completion_sha256),
            "pa_context_grounding_completion_fingerprint": (
                self.pa_context_grounding_completion_fingerprint
            ),
            "resource_selection_ref": self.resource_selection_ref,
            "resource_selection_sha256": self.resource_selection_sha256,
            "resource_selection_fingerprint": self.resource_selection_fingerprint,
            "resource_assignment_delta_ref": self.resource_assignment_delta_ref,
            "resource_assignment_delta_sha256": self.resource_assignment_delta_sha256,
        }
        if self.current_state_iri is not None:
            record.update(
                {
                    "current_state_iri": self.current_state_iri,
                    "desired_state_iri": self.desired_state_iri,
                    "reachability_check_ref": self.reachability_check_ref,
                    "reachability_check_sha256": self.reachability_check_sha256,
                    "reachability_check_fingerprint": (self.reachability_check_fingerprint),
                    "robot_agent_validation_ref": self.robot_agent_validation_ref,
                    "robot_agent_validation_sha256": (self.robot_agent_validation_sha256),
                    "robot_agent_validation_fingerprint": (self.robot_agent_validation_fingerprint),
                    "allocation_label": self.allocation_label,
                    "motion_executed": self.motion_executed,
                }
            )
        if schema_version == 3:
            record.update(
                {
                    "process_symbol": self.process_symbol,
                    "selected_resource_symbol": self.selected_resource_symbol,
                    "current_state_evidence": deepcopy(dict(self.current_state_evidence or {})),
                    "desired_state_evidence": deepcopy(dict(self.desired_state_evidence or {})),
                    "registry_snapshot_ref": self.registry_snapshot_ref,
                    "registry_snapshot_sha256": self.registry_snapshot_sha256,
                    "registry_snapshot_fingerprint": self.registry_snapshot_fingerprint,
                    "workcell_snapshot_ref": self.workcell_snapshot_ref,
                    "workcell_snapshot_sha256": self.workcell_snapshot_sha256,
                    "workcell_snapshot_fingerprint": self.workcell_snapshot_fingerprint,
                    "evidence_presentation_ref": self.evidence_presentation_ref,
                    "evidence_presentation_sha256": self.evidence_presentation_sha256,
                    "evidence_presentation_fingerprint": self.evidence_presentation_fingerprint,
                    "allocation_presentation_ref": self.allocation_presentation_ref,
                    "allocation_presentation_sha256": self.allocation_presentation_sha256,
                    "allocation_presentation_fingerprint": (
                        self.allocation_presentation_fingerprint
                    ),
                    "validation_scope": self.validation_scope,
                    "checked_constraints": list(self.checked_constraints or ()),
                    "unvalidated_constraints": list(self.unvalidated_constraints or ()),
                }
            )
        record["fingerprint"] = self.fingerprint
        return record

    def assert_addressed_to(self, resource_jid: str) -> None:
        """Raise unless this envelope is addressed to the exact receiving RA."""
        receiver = _required_text(resource_jid, "receiving resource_jid")
        if receiver != self.selected_resource_jid:
            raise RAContextHandoffError(
                "SelectedRAAssignmentEnvelope is addressed to a different RA."
            )


@dataclass(frozen=True)
class RobotStateSnapshot:
    """Hold one immutable fresh state response from the selected RA."""

    resource_jid: str
    assignment_fingerprint: str
    retrieved_at_ns: int
    robot_state: Mapping[str, object]
    state_fingerprint: str
    fingerprint: str

    def to_record(self) -> dict[str, object]:
        """Return the exact JSON-safe state snapshot."""
        return {
            "schema_version": 1,
            "record_type": "RobotStateSnapshot",
            "resource_jid": self.resource_jid,
            "assignment_fingerprint": self.assignment_fingerprint,
            "retrieved_at_ns": self.retrieved_at_ns,
            "robot_state": deepcopy(dict(self.robot_state)),
            "state_fingerprint": self.state_fingerprint,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class PrimitiveCatalogSnapshot:
    """Hold one immutable complete primitive catalog from the selected RA."""

    resource_jid: str
    assignment_ref: str
    assignment_sha256: str
    assignment_fingerprint: str
    resource_selection_ref: str
    resource_selection_sha256: str
    resource_selection_fingerprint: str
    robot_state_ref: str
    robot_state_sha256: str
    robot_state_fingerprint: str
    retrieved_at_ns: int
    primitive_catalog: tuple[Mapping[str, object], ...]
    catalog_fingerprint: str
    fingerprint: str

    def to_record(self) -> dict[str, object]:
        """Return the exact JSON-safe primitive-catalog snapshot."""
        return {
            "schema_version": 1,
            "record_type": "PrimitiveCatalogSnapshot",
            "resource_jid": self.resource_jid,
            "assignment_ref": self.assignment_ref,
            "assignment_sha256": self.assignment_sha256,
            "assignment_fingerprint": self.assignment_fingerprint,
            "resource_selection_ref": self.resource_selection_ref,
            "resource_selection_sha256": self.resource_selection_sha256,
            "resource_selection_fingerprint": self.resource_selection_fingerprint,
            "robot_state_ref": self.robot_state_ref,
            "robot_state_sha256": self.robot_state_sha256,
            "robot_state_fingerprint": self.robot_state_fingerprint,
            "retrieved_at_ns": self.retrieved_at_ns,
            "primitive_catalog": [deepcopy(dict(item)) for item in self.primitive_catalog],
            "catalog_fingerprint": self.catalog_fingerprint,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class SelectedRAContextSnapshot:
    """Return the verified Phase 5.1 assignment and paired snapshot artifacts."""

    assignment: SelectedRAAssignmentEnvelope
    assignment_path: Path
    robot_state: RobotStateSnapshot
    robot_state_path: Path
    primitive_catalog: PrimitiveCatalogSnapshot
    primitive_catalog_path: Path


@dataclass(frozen=True)
class Phase51Diagnostic:
    """Describe the read-only persisted status of one Phase 5.1 interaction."""

    status: str
    message: str
    product_requirement: str | None = None
    selected_resource_jid: str | None = None
    selected_execution_mode: str | None = None
    assignment_ref: str | None = None
    state_snapshot_count: int = 0
    latest_state_ref: str | None = None
    catalog_snapshot_count: int = 0
    latest_catalog_ref: str | None = None
    catalog_fingerprint: str | None = None
    robot_state: Mapping[str, object] | None = None
    primitive_catalog: tuple[Mapping[str, object], ...] = ()
    failure: str | None = None

    def to_view(self) -> dict[str, object]:
        """Return a JSON-safe view for the read-only Spec2Primitives UI card."""
        return {
            "status": self.status,
            "message": self.message,
            "product_requirement": self.product_requirement,
            "selected_resource_jid": self.selected_resource_jid,
            "selected_execution_mode": self.selected_execution_mode,
            "assignment_ref": self.assignment_ref,
            "state_snapshot_count": self.state_snapshot_count,
            "latest_state_ref": self.latest_state_ref,
            "catalog_snapshot_count": self.catalog_snapshot_count,
            "latest_catalog_ref": self.latest_catalog_ref,
            "catalog_fingerprint": self.catalog_fingerprint,
            "robot_state": (
                deepcopy(dict(self.robot_state)) if self.robot_state is not None else None
            ),
            "primitive_count": len(self.primitive_catalog),
            "primitive_symbols": [
                str(entry["primitive_symbol"]) for entry in self.primitive_catalog
            ],
            "primitive_catalog": [deepcopy(dict(entry)) for entry in self.primitive_catalog],
            "failure": self.failure,
        }


class RobotAgentCompositionRuntime(Protocol):
    """Read-only selected-RA operation used by the Phase 5.1 adapter."""

    async def request_assigned_context(
        self,
        assignment: SelectedRAAssignmentEnvelope,
    ) -> Mapping[str, object]:
        """Return fresh state and the complete catalog for this assignment."""
        ...


async def activate_selected_ra_context(
    runtime: RobotAgentCompositionRuntime,
    interaction_root: Path,
) -> SelectedRAContextSnapshot:
    """Deliver the Phase 4 assignment and persist one fresh selected-RA context."""
    root = Path(interaction_root).resolve()
    assignment, selection = _build_assignment_envelope(root)
    assignment_path = _persist_or_load_assignment(root, assignment)

    # Validate the existing pair history before contacting the RA so a corrupt
    # audit trail cannot be extended with another otherwise-valid response.
    snapshot_number = _next_snapshot_number(
        root,
        assignment=assignment,
        assignment_path=assignment_path,
        selection=selection,
    )
    response = await runtime.request_assigned_context(assignment)
    robot_state, primitive_catalog = _validate_runtime_response(response, assignment)
    retrieved_at_ns = time.time_ns()

    resource_root = root / "resources" / assignment.selected_resource_jid
    state_path = resource_root / _ROBOT_STATE_DIRECTORY / f"snapshot_{snapshot_number:04d}.json"
    catalog_path = (
        resource_root / _PRIMITIVE_CATALOG_DIRECTORY / f"snapshot_{snapshot_number:04d}.json"
    )
    state_payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "RobotStateSnapshot",
        "resource_jid": assignment.selected_resource_jid,
        "assignment_fingerprint": assignment.fingerprint,
        "retrieved_at_ns": retrieved_at_ns,
        "robot_state": robot_state,
        "state_fingerprint": _fingerprint(robot_state),
    }
    state_payload["fingerprint"] = _fingerprint(state_payload)
    _write_json_exclusive(state_path, state_payload)

    state_ref = state_path.relative_to(root).as_posix()
    assignment_ref = assignment_path.relative_to(root).as_posix()
    catalog_payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "PrimitiveCatalogSnapshot",
        "resource_jid": assignment.selected_resource_jid,
        "assignment_ref": assignment_ref,
        "assignment_sha256": _sha256_path(assignment_path),
        "assignment_fingerprint": assignment.fingerprint,
        "resource_selection_ref": assignment.resource_selection_ref,
        "resource_selection_sha256": assignment.resource_selection_sha256,
        "resource_selection_fingerprint": assignment.resource_selection_fingerprint,
        "robot_state_ref": state_ref,
        "robot_state_sha256": _sha256_path(state_path),
        "robot_state_fingerprint": state_payload["fingerprint"],
        "retrieved_at_ns": retrieved_at_ns,
        "primitive_catalog": primitive_catalog,
        "catalog_fingerprint": _fingerprint(primitive_catalog),
    }
    catalog_payload["fingerprint"] = _fingerprint(catalog_payload)
    _write_json_exclusive(catalog_path, catalog_payload)

    state_snapshot = _load_robot_state_snapshot(state_path)
    catalog_snapshot = _load_primitive_catalog_snapshot(catalog_path)
    _validate_snapshot_linkage(
        root,
        assignment=assignment,
        assignment_path=assignment_path,
        selection=selection,
        state_path=state_path,
        state=state_snapshot,
        catalog=catalog_snapshot,
    )
    return SelectedRAContextSnapshot(
        assignment=assignment,
        assignment_path=assignment_path,
        robot_state=state_snapshot,
        robot_state_path=state_path,
        primitive_catalog=catalog_snapshot,
        primitive_catalog_path=catalog_path,
    )


def read_phase_5_1_diagnostic(interaction_root: Path) -> Phase51Diagnostic:
    """Read and validate Phase 5.1 artifacts without activating or contacting an RA."""
    root = Path(interaction_root).resolve()
    assignment_directory = root.joinpath(*_ASSIGNMENT_ROOT)
    assignment_paths = sorted(assignment_directory.glob("*.json"))
    completion_paths = sorted((root / "interaction_record").glob("context_completion_*.json"))
    if not completion_paths:
        if assignment_paths:
            return Phase51Diagnostic(
                status="blocked",
                message="Phase 5.1 artifacts exist without a Phase 4 completion.",
                failure="SelectedRAAssignmentEnvelope has no Phase 4 authority.",
            )
        return Phase51Diagnostic(
            status="waiting_for_phase_4",
            message="Waiting for one validated Phase 4 completion.",
        )

    try:
        assignment, selection = _build_assignment_envelope(root)
    except RAContextHandoffError as exc:
        return Phase51Diagnostic(
            status="blocked",
            message="Phase 4 cannot authorize the Phase 5.1 handoff.",
            failure=str(exc),
        )

    common = {
        "product_requirement": assignment.product_requirement,
        "selected_resource_jid": assignment.selected_resource_jid,
        "selected_execution_mode": assignment.selected_execution_mode,
    }
    expected_assignment_path = assignment_directory / _ASSIGNMENT_NAME
    resource_root = root / "resources" / assignment.selected_resource_jid
    state_directory = resource_root / _ROBOT_STATE_DIRECTORY
    catalog_directory = resource_root / _PRIMITIVE_CATALOG_DIRECTORY
    try:
        state_paths = _numbered_snapshot_paths(state_directory)
        catalog_paths = _numbered_snapshot_paths(catalog_directory)
        if not assignment_paths:
            if state_paths or catalog_paths:
                raise RAContextHandoffError(
                    "RA snapshots exist without SelectedRAAssignmentEnvelope."
                )
            return Phase51Diagnostic(
                status="ready_for_assignment",
                message=("Phase 4 selected this RA; Phase 5.1 activation has not been requested."),
                **common,
            )
        if assignment_paths != [expected_assignment_path]:
            raise RAContextHandoffError("Phase 5.1 requires exactly assignment_0001.json.")
        persisted_assignment = _assignment_from_mapping(
            _read_json_mapping(
                expected_assignment_path,
                "SelectedRAAssignmentEnvelope",
            )
        )
        if persisted_assignment != assignment:
            raise RAContextHandoffError(
                "Persisted SelectedRAAssignmentEnvelope does not match Phase 4."
            )
        assignment_ref = expected_assignment_path.relative_to(root).as_posix()
        if set(state_paths) != set(catalog_paths):
            raise RAContextHandoffError(
                "Robot state and primitive catalog snapshot revisions are unpaired."
            )
        if not state_paths:
            return Phase51Diagnostic(
                status="waiting_for_ra",
                message=(
                    "The assignment is recorded; no valid RA state/catalog response "
                    "has been persisted."
                ),
                assignment_ref=assignment_ref,
                **common,
            )
        latest = _load_latest_selected_ra_context(
            root,
            assignment=assignment,
            assignment_path=expected_assignment_path,
            selection=selection,
        )
        return Phase51Diagnostic(
            status="context_captured",
            message=(
                "The selected RA assignment, current state, and complete catalog "
                "snapshot are valid."
            ),
            assignment_ref=assignment_ref,
            state_snapshot_count=len(state_paths),
            latest_state_ref=latest.robot_state_path.relative_to(root).as_posix(),
            catalog_snapshot_count=len(catalog_paths),
            latest_catalog_ref=latest.primitive_catalog_path.relative_to(root).as_posix(),
            catalog_fingerprint=latest.primitive_catalog.catalog_fingerprint,
            robot_state=latest.robot_state.robot_state,
            primitive_catalog=latest.primitive_catalog.primitive_catalog,
            **common,
        )
    except RAContextHandoffError as exc:
        return Phase51Diagnostic(
            status="blocked",
            message="Phase 5.1 persisted evidence failed validation.",
            failure=str(exc),
            **common,
        )


def load_selected_ra_context_snapshot(
    interaction_root: Path,
) -> SelectedRAContextSnapshot:
    """Load the latest fully validated Phase 5.1 assignment/state/catalog set."""
    root = Path(interaction_root).resolve()
    assignment, selection = _build_assignment_envelope(root)
    assignment_path = root.joinpath(*_ASSIGNMENT_ROOT, _ASSIGNMENT_NAME)
    if not assignment_path.is_file():
        raise RAContextHandoffError("Phase 5.2 requires one completed Phase 5.1 context capture.")
    persisted_assignment = _assignment_from_mapping(
        _read_json_mapping(assignment_path, "SelectedRAAssignmentEnvelope")
    )
    if persisted_assignment != assignment:
        raise RAContextHandoffError(
            "Persisted SelectedRAAssignmentEnvelope does not match Phase 4."
        )
    return _load_latest_selected_ra_context(
        root,
        assignment=assignment,
        assignment_path=assignment_path,
        selection=selection,
    )


def _load_latest_selected_ra_context(
    root: Path,
    *,
    assignment: SelectedRAAssignmentEnvelope,
    assignment_path: Path,
    selection: Mapping[str, object],
) -> SelectedRAContextSnapshot:
    """Load the latest pair after validating the complete append-only history."""
    resource_root = root / "resources" / assignment.selected_resource_jid
    state_paths = _numbered_snapshot_paths(resource_root / _ROBOT_STATE_DIRECTORY)
    catalog_paths = _numbered_snapshot_paths(resource_root / _PRIMITIVE_CATALOG_DIRECTORY)
    if set(state_paths) != set(catalog_paths):
        raise RAContextHandoffError(
            "Robot state and primitive catalog snapshot revisions are unpaired."
        )
    if not state_paths:
        raise RAContextHandoffError("Phase 5.2 requires one completed Phase 5.1 context capture.")
    _next_snapshot_number(
        root,
        assignment=assignment,
        assignment_path=assignment_path,
        selection=selection,
    )
    latest_number = max(state_paths)
    state_path = state_paths[latest_number]
    catalog_path = catalog_paths[latest_number]
    state = _load_robot_state_snapshot(state_path)
    catalog = _load_primitive_catalog_snapshot(catalog_path)
    _validate_snapshot_linkage(
        root,
        assignment=assignment,
        assignment_path=assignment_path,
        selection=selection,
        state_path=state_path,
        state=state,
        catalog=catalog,
    )
    return SelectedRAContextSnapshot(
        assignment=assignment,
        assignment_path=assignment_path,
        robot_state=state,
        robot_state_path=state_path,
        primitive_catalog=catalog,
        primitive_catalog_path=catalog_path,
    )


def _build_assignment_envelope(
    root: Path,
) -> tuple[SelectedRAAssignmentEnvelope, Mapping[str, object]]:
    try:
        completion = load_pa_context_grounding_completion(root)
    except GroundingContractError as exc:
        raise RAContextHandoffError(
            "Phase 5.1 requires one unchanged PAContextGroundingCompletion."
        ) from exc
    if not isinstance(
        completion,
        (
            PAContextGroundingCompletionV4,
            PAContextGroundingCompletionV5,
            PAContextGroundingCompletionV6,
        ),
    ):
        raise RAContextHandoffError(
            "Phase 5.1 requires PAContextGroundingCompletion version 4, 5, or 6."
        )
    completion_record = completion.to_record()
    completion_paths = sorted((root / "interaction_record").glob("context_completion_*.json"))
    if len(completion_paths) != 1:
        raise RAContextHandoffError("Phase 5.1 requires exactly one completion record.")
    completion_path = completion_paths[0]
    selection = _load_validated_selection(root, completion_record)
    _validate_assignment_delta(root, completion_record, selection)

    payload: dict[str, object] = {
        "schema_version": (
            3
            if isinstance(completion, PAContextGroundingCompletionV6)
            else (2 if isinstance(completion, PAContextGroundingCompletionV5) else 1)
        ),
        "record_type": "SelectedRAAssignmentEnvelope",
        "product_requirement": _required_text(
            completion_record.get("product_requirement"),
            "PAContextGroundingCompletion.product_requirement",
        ),
        "specification_iri": selection["specification_iri"],
        "feature_iri": selection["feature_iri"],
        "process_iri": selection["process_iri"],
        "selected_resource_iri": selection["selected_resource_iri"],
        "selected_resource_jid": selection["selected_resource_jid"],
        "selected_execution_mode": selection["selected_execution_mode"],
        "pa_context_grounding_completion_ref": completion_path.relative_to(root).as_posix(),
        "pa_context_grounding_completion_sha256": _sha256_path(completion_path),
        "pa_context_grounding_completion_fingerprint": _sha256_text(
            completion_record.get("fingerprint"),
            "PAContextGroundingCompletion.fingerprint",
        ),
        "resource_selection_ref": selection["record_ref"],
        "resource_selection_sha256": completion_record["resource_selection_sha256"],
        "resource_selection_fingerprint": selection["fingerprint"],
        "resource_assignment_delta_ref": completion_record["resource_assignment_delta_ref"],
        "resource_assignment_delta_sha256": completion_record["resource_assignment_delta_sha256"],
    }
    if isinstance(
        completion,
        (PAContextGroundingCompletionV5, PAContextGroundingCompletionV6),
    ):
        payload.update(
            {
                "current_state_iri": selection["current_state_iri"],
                "desired_state_iri": selection["desired_state_iri"],
                "reachability_check_ref": selection["reachability_check_ref"],
                "reachability_check_sha256": selection["reachability_check_sha256"],
                "reachability_check_fingerprint": selection["reachability_check_fingerprint"],
                "robot_agent_validation_ref": selection["robot_agent_validation_ref"],
                "robot_agent_validation_sha256": selection["robot_agent_validation_sha256"],
                "robot_agent_validation_fingerprint": selection[
                    "robot_agent_validation_fingerprint"
                ],
                "allocation_label": completion_record["allocation_label"],
                "motion_executed": False,
            }
        )
    if isinstance(completion, PAContextGroundingCompletionV6):
        payload.update(
            {
                "process_symbol": selection["process_symbol"],
                "selected_resource_symbol": selection["selected_resource_symbol"],
                "current_state_evidence": selection["current_state_evidence"],
                "desired_state_evidence": selection["desired_state_evidence"],
                "registry_snapshot_ref": completion_record["registry_snapshot_ref"],
                "registry_snapshot_sha256": completion_record["registry_snapshot_sha256"],
                "registry_snapshot_fingerprint": completion_record["registry_snapshot_fingerprint"],
                "workcell_snapshot_ref": completion_record["workcell_snapshot_ref"],
                "workcell_snapshot_sha256": completion_record["workcell_snapshot_sha256"],
                "workcell_snapshot_fingerprint": completion_record["workcell_snapshot_fingerprint"],
                "evidence_presentation_ref": selection["evidence_presentation_ref"],
                "evidence_presentation_sha256": selection["evidence_presentation_sha256"],
                "evidence_presentation_fingerprint": selection["evidence_presentation_fingerprint"],
                "allocation_presentation_ref": selection["allocation_presentation_ref"],
                "allocation_presentation_sha256": selection["allocation_presentation_sha256"],
                "allocation_presentation_fingerprint": selection[
                    "allocation_presentation_fingerprint"
                ],
                "validation_scope": completion_record["validation_scope"],
                "checked_constraints": completion_record["checked_constraints"],
                "unvalidated_constraints": completion_record["unvalidated_constraints"],
            }
        )
    payload["fingerprint"] = _fingerprint(payload)
    return _assignment_from_mapping(payload), selection


def _load_validated_selection(  # noqa: C901
    root: Path,
    completion: Mapping[str, object],
) -> dict[str, object]:
    if completion.get("schema_version") == 6:
        return _load_validated_selection_v4(root, completion)
    if completion.get("schema_version") == 5:
        return _load_validated_selection_v3(root, completion)
    selection_ref = _required_text(
        completion.get("resource_selection_ref"),
        "PAContextGroundingCompletion.resource_selection_ref",
    )
    selection_path = _resolve_ref(
        root,
        selection_ref,
        prefix=_RESOURCE_SELECTION_PREFIX,
    )
    selection = _read_json_mapping(selection_path, "ResourceSelectionRecord")
    _require_exact_keys(selection, _SELECTION_KEYS, "ResourceSelectionRecord")
    selection_number = selection.get("selection_number")
    if (
        selection.get("schema_version") != 2
        or selection.get("record_type") != "ResourceSelectionRecord"
        or isinstance(selection_number, bool)
        or not isinstance(selection_number, int)
        or selection_number < 1
        or len(selection_path.relative_to(root).parts) != 5
        or _SELECTION_NAME_PATTERN.fullmatch(selection_path.parent.name) is None
        or selection_path.parent.name != f"selection_{selection_number:04d}"
        or selection_path.name != "resource_selection_record.json"
    ):
        raise RAContextHandoffError("ResourceSelectionRecord identity is invalid.")
    if _sha256_path(selection_path) != _sha256_text(
        completion.get("resource_selection_sha256"),
        "PAContextGroundingCompletion.resource_selection_sha256",
    ):
        raise RAContextHandoffError("ResourceSelectionRecord hash is invalid.")
    selection_fingerprint = _record_fingerprint(
        selection,
        "ResourceSelectionRecord",
    )
    if selection.get("tbox_fingerprint") != completion.get("tbox_fingerprint"):
        raise RAContextHandoffError(
            "ResourceSelectionRecord TBox fingerprint does not match completion."
        )
    if (
        selection.get("required_record_type") != "RobotFrameLocationRecord"
        or selection.get("target_frame") != "world"
        or selection.get("selection_policy") != "first_reachable_in_predefined_registry_order"
    ):
        raise RAContextHandoffError(
            "ResourceSelectionRecord is not the current location-based selection."
        )
    for field in (
        "specification_iri",
        "feature_iri",
        "process_iri",
        "selected_resource_iri",
        "selected_resource_jid",
        "selected_execution_mode",
    ):
        _required_text(selection.get(field), f"ResourceSelectionRecord.{field}")
    _validate_resource_jid(str(selection["selected_resource_jid"]))

    candidates_value = selection.get("candidate_reach_evidence")
    candidate_iris = selection.get("candidate_resource_iris")
    if (
        not isinstance(candidates_value, list)
        or not candidates_value
        or not isinstance(candidate_iris, list)
        or len(candidate_iris) != len(candidates_value)
    ):
        raise RAContextHandoffError("ResourceSelectionRecord candidate evidence is invalid.")
    candidates: list[Mapping[str, object]] = []
    for index, candidate in enumerate(candidates_value):
        if not isinstance(candidate, Mapping):
            raise RAContextHandoffError(
                "ResourceSelectionRecord candidate evidence must be an object."
            )
        _require_exact_keys(candidate, _CANDIDATE_KEYS, "candidate reach evidence")
        for field in (
            "resource_symbol",
            "resource_iri",
            "resource_jid",
            "manifest_ref",
            "execution_mode",
        ):
            _required_text(candidate.get(field), f"candidate[{index}].{field}")
        _sha256_text(candidate.get("manifest_sha256"), "candidate manifest_sha256")
        if not isinstance(candidate.get("reachable"), bool):
            raise RAContextHandoffError("Candidate reachable must be a boolean.")
        _text_list(candidate.get("verdicts"), "candidate verdicts")
        if candidate_iris[index] != candidate.get("resource_iri"):
            raise RAContextHandoffError(
                "Candidate resource order does not match candidate evidence."
            )
        candidates.append(candidate)
    selected = next((item for item in candidates if item["reachable"] is True), None)
    if selected is None:
        raise RAContextHandoffError("ResourceSelectionRecord has no reachable RA.")
    selected_fields = (
        ("selected_resource_symbol", "resource_symbol"),
        ("selected_resource_iri", "resource_iri"),
        ("selected_resource_jid", "resource_jid"),
        ("selected_execution_mode", "execution_mode"),
    )
    if any(selection[left] != selected[right] for left, right in selected_fields):
        raise RAContextHandoffError(
            "ResourceSelectionRecord does not select its first reachable RA."
        )

    grounding_ref = _required_text(
        selection.get("grounding_record_ref"),
        "ResourceSelectionRecord.grounding_record_ref",
    )
    grounding_sha256 = _sha256_text(
        selection.get("grounding_record_sha256"),
        "ResourceSelectionRecord.grounding_record_sha256",
    )
    typed_context_refs = completion.get("typed_context_refs")
    if (
        not isinstance(typed_context_refs, list)
        or {"ref": grounding_ref, "sha256": grounding_sha256} not in typed_context_refs
        or _sha256_path(_resolve_ref(root, grounding_ref, prefix=("products", "grounding")))
        != grounding_sha256
    ):
        raise RAContextHandoffError(
            "ResourceSelectionRecord is not pinned to accepted typed context."
        )
    result = deepcopy(selection)
    result["record_ref"] = selection_ref
    result["fingerprint"] = selection_fingerprint
    return result


def _load_validated_selection_v3(
    root: Path,
    completion: Mapping[str, object],
) -> dict[str, object]:
    """Reconstruct the exact accepted PA choice without registry-order logic."""
    selection_ref = _required_text(
        completion.get("resource_selection_ref"),
        "PAContextGroundingCompletion.resource_selection_ref",
    )
    selection_path = _resolve_ref(
        root,
        selection_ref,
        prefix=_RESOURCE_SELECTION_PREFIX,
    )
    selection = _read_json_mapping(selection_path, "ResourceSelectionRecord")
    _require_exact_keys(selection, _SELECTION_V3_KEYS, "ResourceSelectionRecord v3")
    selection_number = selection.get("selection_number")
    if (
        selection.get("schema_version") != 3
        or selection.get("record_type") != "ResourceSelectionRecord"
        or selection.get("authority") != "ProductAgent"
        or isinstance(selection_number, bool)
        or not isinstance(selection_number, int)
        or selection_number < 1
        or selection_path.parent.name != f"selection_{selection_number:04d}"
        or selection_path.name != "resource_selection_record.json"
        or selection.get("allocation_status") != "accepted"
        or selection.get("robot_agent_validation_status") != "accepted"
    ):
        raise RAContextHandoffError("ResourceSelectionRecord v3 identity is invalid.")
    if _sha256_path(selection_path) != _sha256_text(
        completion.get("resource_selection_sha256"),
        "PAContextGroundingCompletion.resource_selection_sha256",
    ):
        raise RAContextHandoffError("ResourceSelectionRecord v3 hash is invalid.")
    selection_fingerprint = _record_fingerprint(
        selection,
        "ResourceSelectionRecord v3",
    )
    if selection_fingerprint != completion.get("resource_selection_fingerprint") or selection.get(
        "tbox_fingerprint"
    ) != completion.get("tbox_fingerprint"):
        raise RAContextHandoffError("ResourceSelectionRecord v3 completion lineage is invalid.")
    selected = tuple(
        selection.get(field)
        for field in (
            "selected_resource_symbol",
            "selected_resource_iri",
            "selected_resource_jid",
            "selected_execution_mode",
        )
    )
    provisional = tuple(
        selection.get(field)
        for field in (
            "provisional_resource_symbol",
            "provisional_resource_iri",
            "provisional_resource_jid",
            "provisional_execution_mode",
        )
    )
    if selected != provisional or not all(isinstance(item, str) and item for item in selected):
        raise RAContextHandoffError(
            "ResourceSelectionRecord v3 substituted the PA provisional resource."
        )
    for field in (
        "specification_iri",
        "feature_iri",
        "process_iri",
        "current_state_iri",
        "desired_state_iri",
    ):
        _required_text(selection.get(field), f"ResourceSelectionRecord.{field}")
    _validate_resource_jid(str(selection["selected_resource_jid"]))
    pinned_pairs = (
        (
            "reachability_check_ref",
            "reachability_check_sha256",
            ("products", "grounding", "reachability"),
        ),
        (
            "robot_agent_validation_ref",
            "robot_agent_validation_sha256",
            ("resources", str(selection["selected_resource_jid"]), "validation"),
        ),
    )
    for ref_field, sha_field, prefix in pinned_pairs:
        if (
            selection.get(ref_field) != completion.get(ref_field)
            or selection.get(sha_field) != completion.get(sha_field)
            or _sha256_path(_resolve_ref(root, str(selection[ref_field]), prefix=prefix))
            != _sha256_text(selection.get(sha_field), sha_field)
        ):
            raise RAContextHandoffError("ResourceSelectionRecord v3 verifier lineage is invalid.")
    for field in (
        "reachability_check_fingerprint",
        "robot_agent_validation_fingerprint",
    ):
        if selection.get(field) != completion.get(field):
            raise RAContextHandoffError(
                "ResourceSelectionRecord v3 verifier fingerprint is invalid."
            )
    result = deepcopy(selection)
    result["record_ref"] = selection_ref
    result["fingerprint"] = selection_fingerprint
    return result


def _load_validated_selection_v4(  # noqa: C901
    root: Path,
    completion: Mapping[str, object],
) -> dict[str, object]:
    """Reconstruct the exact process-aware PA choice for envelope v3."""
    selection_ref = _required_text(
        completion.get("resource_selection_ref"),
        "PAContextGroundingCompletion.resource_selection_ref",
    )
    selection_path = _resolve_ref(
        root,
        selection_ref,
        prefix=_RESOURCE_SELECTION_PREFIX,
    )
    selection = _read_json_mapping(selection_path, "ResourceSelectionRecord v4")
    _require_exact_keys(selection, _SELECTION_V4_KEYS, "ResourceSelectionRecord v4")
    selection_number = selection.get("selection_number")
    if (
        selection.get("schema_version") != 4
        or selection.get("record_type") != "ResourceSelectionRecord"
        or selection.get("authority") != "ProductAgent"
        or isinstance(selection_number, bool)
        or not isinstance(selection_number, int)
        or selection_number < 1
        or selection_path.parent.name != f"selection_{selection_number:04d}"
        or selection_path.name != "resource_selection_record.json"
        or selection.get("allocation_status") != "accepted"
        or selection.get("robot_agent_validation_status") != "accepted"
    ):
        raise RAContextHandoffError("ResourceSelectionRecord v4 identity is invalid.")
    if _sha256_path(selection_path) != _sha256_text(
        completion.get("resource_selection_sha256"),
        "PAContextGroundingCompletion.resource_selection_sha256",
    ):
        raise RAContextHandoffError("ResourceSelectionRecord v4 hash is invalid.")
    selection_fingerprint = _record_fingerprint(
        selection,
        "ResourceSelectionRecord v4",
    )
    if (
        selection_fingerprint != completion.get("resource_selection_fingerprint")
        or selection.get("tbox_fingerprint") != completion.get("tbox_fingerprint")
        or selection.get("registry_fingerprint") != completion.get("registry_snapshot_fingerprint")
        or selection.get("workcell_fingerprint") != completion.get("workcell_snapshot_fingerprint")
    ):
        raise RAContextHandoffError("ResourceSelectionRecord v4 authority is inconsistent.")

    selected = tuple(
        selection.get(field)
        for field in (
            "selected_resource_symbol",
            "selected_resource_iri",
            "selected_resource_jid",
            "selected_execution_mode",
        )
    )
    provisional = tuple(
        selection.get(field)
        for field in (
            "provisional_resource_symbol",
            "provisional_resource_iri",
            "provisional_resource_jid",
            "provisional_execution_mode",
        )
    )
    if selected != provisional or not all(isinstance(item, str) and item for item in selected):
        raise RAContextHandoffError(
            "ResourceSelectionRecord v4 substituted the PA provisional resource."
        )
    _validate_resource_jid(str(selection["selected_resource_jid"]))
    for field in (
        "specification_iri",
        "feature_iri",
        "process_symbol",
        "process_iri",
        "current_state_iri",
        "desired_state_iri",
    ):
        _required_text(selection.get(field), f"ResourceSelectionRecord.{field}")
    for field in (
        "process_symbol",
        "process_iri",
        "feature_iri",
        "current_state_iri",
        "desired_state_iri",
    ):
        if selection.get(field) != completion.get(field):
            raise RAContextHandoffError(
                "ResourceSelectionRecord v4 process/state lineage is invalid."
            )
    candidate_symbols = selection.get("candidate_resource_symbols")
    candidate_iris = selection.get("candidate_resource_iris")
    if (
        not isinstance(candidate_symbols, list)
        or not candidate_symbols
        or not all(isinstance(item, str) and item for item in candidate_symbols)
        or not isinstance(candidate_iris, list)
        or len(candidate_iris) != len(candidate_symbols)
        or not all(isinstance(item, str) and item for item in candidate_iris)
        or selection["selected_resource_symbol"] not in candidate_symbols
        or selection["selected_resource_iri"] not in candidate_iris
    ):
        raise RAContextHandoffError("ResourceSelectionRecord v4 candidate set is invalid.")

    state_evidence_keys = {
        "state_iri",
        "evidence_handle",
        "source_record_type",
        "source_record_ref",
        "source_record_sha256",
        "source_field_path",
    }
    for state_name in ("current_state", "desired_state"):
        evidence = selection.get(f"{state_name}_evidence")
        if not isinstance(evidence, Mapping):
            raise RAContextHandoffError("ResourceSelectionRecord state evidence is invalid.")
        _require_exact_keys(
            evidence,
            state_evidence_keys,
            f"ResourceSelectionRecord.{state_name}_evidence",
        )
        if evidence.get("state_iri") != selection[f"{state_name}_iri"]:
            raise RAContextHandoffError(
                "ResourceSelectionRecord state-evidence assignment is invalid."
            )
        for field in state_evidence_keys - {"source_record_sha256"}:
            _required_text(evidence.get(field), f"{state_name}_evidence.{field}")
        _sha256_text(
            evidence.get("source_record_sha256"),
            f"{state_name}_evidence.source_record_sha256",
        )

    pinned_pairs = (
        (
            "evidence_presentation_ref",
            "evidence_presentation_sha256",
            ("products", "grounding", "presentation"),
        ),
        (
            "allocation_presentation_ref",
            "allocation_presentation_sha256",
            ("products", "grounding", "presentation"),
        ),
        (
            "reachability_check_ref",
            "reachability_check_sha256",
            ("products", "grounding", "reachability"),
        ),
        (
            "robot_agent_validation_ref",
            "robot_agent_validation_sha256",
            ("resources", str(selection["selected_resource_jid"]), "validation"),
        ),
    )
    for ref_field, sha_field, prefix in pinned_pairs:
        if (
            selection.get(ref_field) != completion.get(ref_field)
            or selection.get(sha_field) != completion.get(sha_field)
            or _sha256_path(_resolve_ref(root, str(selection[ref_field]), prefix=prefix))
            != _sha256_text(selection.get(sha_field), sha_field)
        ):
            raise RAContextHandoffError(
                "ResourceSelectionRecord v4 presentation/verifier lineage is invalid."
            )
    for field in (
        "evidence_presentation_fingerprint",
        "allocation_presentation_fingerprint",
        "reachability_check_fingerprint",
        "robot_agent_validation_fingerprint",
    ):
        if selection.get(field) != completion.get(field):
            raise RAContextHandoffError(
                "ResourceSelectionRecord v4 presentation/verifier fingerprint is invalid."
            )
    result = deepcopy(selection)
    result["record_ref"] = selection_ref
    result["fingerprint"] = selection_fingerprint
    return result


def _validate_assignment_delta(
    root: Path,
    completion: Mapping[str, object],
    selection: Mapping[str, object],
) -> None:
    delta_ref = _required_text(
        completion.get("resource_assignment_delta_ref"),
        "PAContextGroundingCompletion.resource_assignment_delta_ref",
    )
    delta_path = _resolve_ref(root, delta_ref, prefix=_ONTOLOGY_PREFIX)
    delta = _read_json_mapping(delta_path, "resource assignment delta")
    _require_exact_keys(delta, _DELTA_KEYS, "resource assignment delta")
    delta_match = _DELTA_NAME_PATTERN.fullmatch(delta_path.name)
    if (
        delta.get("schema_version") != 1
        or delta.get("producer") != "resource_grounding_host"
        or delta_match is None
        or delta.get("delta_number") != int(delta_match.group(1))
        or delta.get("uncertainty") != []
        or delta.get("unresolved_evidence_needs") != []
        or delta.get("typed_context_refs") != []
        or _sha256_path(delta_path)
        != _sha256_text(
            completion.get("resource_assignment_delta_sha256"),
            "PAContextGroundingCompletion.resource_assignment_delta_sha256",
        )
    ):
        raise RAContextHandoffError("Resource assignment delta is invalid.")
    assertions = delta.get("assertions")
    if not isinstance(assertions, list) or len(assertions) != 4:
        raise RAContextHandoffError(
            "Resource assignment delta must contain exactly four assertions."
        )
    by_predicate: dict[str, Mapping[str, object]] = {}
    expected_evidence_refs = [selection["record_ref"]]
    if selection.get("schema_version") in {3, 4}:
        expected_evidence_refs.extend(
            [
                selection["reachability_check_ref"],
                selection["robot_agent_validation_ref"],
            ]
        )
    for assertion in assertions:
        if not isinstance(assertion, Mapping):
            raise RAContextHandoffError("Resource assignment assertion is invalid.")
        _require_exact_keys(assertion, _ASSERTION_KEYS, "resource assignment assertion")
        predicate = _required_text(assertion.get("predicate"), "assignment predicate")
        if predicate in by_predicate:
            raise RAContextHandoffError("Resource assignment predicate is duplicated.")
        if assertion.get("evidence_refs") != expected_evidence_refs:
            raise RAContextHandoffError(
                "Every resource assignment assertion must cite its allocation lineage."
            )
        object_value = assertion.get("object")
        if (
            not isinstance(object_value, Mapping)
            or set(object_value) != {"kind", "value"}
            or object_value.get("kind") != "iri"
        ):
            raise RAContextHandoffError("Resource assignment object is invalid.")
        _required_text(object_value.get("value"), "assignment object IRI")
        _required_text(assertion.get("subject"), "assignment subject IRI")
        by_predicate[predicate] = assertion
    required_predicates = {
        _PPR_HAS_PROCESS_EXECUTION,
        _RDF_TYPE,
        _PPR_RUNS_PROCESS,
        _PPR_RUNS_ON_RESOURCE,
    }
    if set(by_predicate) != required_predicates:
        raise RAContextHandoffError("Resource assignment delta has an unexpected assertion set.")
    has_execution = by_predicate[_PPR_HAS_PROCESS_EXECUTION]
    execution_iri = has_execution["object"]["value"]
    if (
        has_execution["subject"] != selection["specification_iri"]
        or by_predicate[_RDF_TYPE]["subject"] != execution_iri
        or by_predicate[_RDF_TYPE]["object"]["value"] != _PPR_PROCESS_EXECUTION
        or by_predicate[_PPR_RUNS_PROCESS]["subject"] != execution_iri
        or by_predicate[_PPR_RUNS_PROCESS]["object"]["value"] != selection["process_iri"]
        or by_predicate[_PPR_RUNS_ON_RESOURCE]["subject"] != execution_iri
        or by_predicate[_PPR_RUNS_ON_RESOURCE]["object"]["value"]
        != selection["selected_resource_iri"]
    ):
        raise RAContextHandoffError("Resource assignment delta does not match the selected RA.")


def _persist_or_load_assignment(
    root: Path,
    assignment: SelectedRAAssignmentEnvelope,
) -> Path:
    directory = root.joinpath(*_ASSIGNMENT_ROOT)
    paths = sorted(directory.glob("*.json"))
    path = directory / _ASSIGNMENT_NAME
    if not paths:
        _write_json_exclusive(path, assignment.to_record())
        return path
    if paths != [path]:
        raise RAContextHandoffError("Phase 5.1 requires one immutable assignment_0001.json record.")
    persisted = _assignment_from_mapping(_read_json_mapping(path, "SelectedRAAssignmentEnvelope"))
    if persisted != assignment:
        raise RAContextHandoffError(
            "Persisted SelectedRAAssignmentEnvelope does not match Phase 4."
        )
    return path


def _assignment_from_mapping(
    value: Mapping[str, object],
) -> SelectedRAAssignmentEnvelope:
    base_expected = {
        "schema_version",
        "record_type",
        "product_requirement",
        "specification_iri",
        "feature_iri",
        "process_iri",
        "selected_resource_iri",
        "selected_resource_jid",
        "selected_execution_mode",
        "pa_context_grounding_completion_ref",
        "pa_context_grounding_completion_sha256",
        "pa_context_grounding_completion_fingerprint",
        "resource_selection_ref",
        "resource_selection_sha256",
        "resource_selection_fingerprint",
        "resource_assignment_delta_ref",
        "resource_assignment_delta_sha256",
        "fingerprint",
    }
    version_two_fields = {
        "current_state_iri",
        "desired_state_iri",
        "reachability_check_ref",
        "reachability_check_sha256",
        "reachability_check_fingerprint",
        "robot_agent_validation_ref",
        "robot_agent_validation_sha256",
        "robot_agent_validation_fingerprint",
        "allocation_label",
        "motion_executed",
    }
    version_three_fields = {
        "process_symbol",
        "selected_resource_symbol",
        "current_state_evidence",
        "desired_state_evidence",
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
        "validation_scope",
        "checked_constraints",
        "unvalidated_constraints",
    }
    schema_version = value.get("schema_version")
    if schema_version == 3:
        expected = base_expected | version_two_fields | version_three_fields
    elif schema_version == 2:
        expected = base_expected | version_two_fields
    else:
        expected = base_expected
    _require_exact_keys(value, expected, "SelectedRAAssignmentEnvelope")
    if (
        schema_version not in {1, 2, 3}
        or value.get("record_type") != "SelectedRAAssignmentEnvelope"
    ):
        raise RAContextHandoffError("SelectedRAAssignmentEnvelope identity is invalid.")
    fingerprint = _record_fingerprint(value, "SelectedRAAssignmentEnvelope")
    text_fields = (
        "product_requirement",
        "specification_iri",
        "feature_iri",
        "process_iri",
        "selected_resource_iri",
        "selected_resource_jid",
        "selected_execution_mode",
        "pa_context_grounding_completion_ref",
        "resource_selection_ref",
        "resource_assignment_delta_ref",
    )
    texts = {field: _required_text(value.get(field), field) for field in text_fields}
    _validate_resource_jid(texts["selected_resource_jid"])
    sha_fields = (
        "pa_context_grounding_completion_sha256",
        "pa_context_grounding_completion_fingerprint",
        "resource_selection_sha256",
        "resource_selection_fingerprint",
        "resource_assignment_delta_sha256",
    )
    shas = {field: _sha256_text(value.get(field), field) for field in sha_fields}
    version_two_texts: dict[str, str] = {}
    version_two_shas: dict[str, str] = {}
    if schema_version in {2, 3}:
        version_two_texts = {
            field: _required_text(value.get(field), field)
            for field in (
                "current_state_iri",
                "desired_state_iri",
                "reachability_check_ref",
                "robot_agent_validation_ref",
                "allocation_label",
            )
        }
        validation_scope = value.get("validation_scope")
        expected_label = (
            "validated Cartesian pick-place allocation"
            if schema_version == 3 and validation_scope == "cartesian_pick_place"
            else (
                "validated endpoint-motion allocation"
                if schema_version == 3
                else "validated allocation"
            )
        )
        if (
            version_two_texts["allocation_label"] != expected_label
            or value.get("motion_executed") is not False
        ):
            raise RAContextHandoffError(
                "SelectedRAAssignmentEnvelope allocation status is invalid."
            )
        version_two_shas = {
            field: _sha256_text(value.get(field), field)
            for field in (
                "reachability_check_sha256",
                "reachability_check_fingerprint",
                "robot_agent_validation_sha256",
                "robot_agent_validation_fingerprint",
            )
        }
    version_three_texts: dict[str, str] = {}
    version_three_shas: dict[str, str] = {}
    current_state_evidence: dict[str, object] | None = None
    desired_state_evidence: dict[str, object] | None = None
    checked_constraints: tuple[str, ...] | None = None
    unvalidated_constraints: tuple[str, ...] | None = None
    if schema_version == 3:
        version_three_texts = {
            field: _required_text(value.get(field), field)
            for field in (
                "process_symbol",
                "selected_resource_symbol",
                "registry_snapshot_ref",
                "workcell_snapshot_ref",
                "evidence_presentation_ref",
                "allocation_presentation_ref",
                "validation_scope",
            )
        }
        validation_scope = version_three_texts["validation_scope"]
        if validation_scope not in {"endpoint_motion", "cartesian_pick_place"}:
            raise RAContextHandoffError("SelectedRAAssignmentEnvelope validation scope is invalid.")
        version_three_shas = {
            field: _sha256_text(value.get(field), field)
            for field in (
                "registry_snapshot_sha256",
                "registry_snapshot_fingerprint",
                "workcell_snapshot_sha256",
                "workcell_snapshot_fingerprint",
                "evidence_presentation_sha256",
                "evidence_presentation_fingerprint",
                "allocation_presentation_sha256",
                "allocation_presentation_fingerprint",
            )
        }
        current_state_evidence = _json_mapping(
            value.get("current_state_evidence"),
            "current_state_evidence",
        )
        desired_state_evidence = _json_mapping(
            value.get("desired_state_evidence"),
            "desired_state_evidence",
        )
        checked_constraints = tuple(
            _text_list(value.get("checked_constraints"), "checked_constraints")
        )
        unvalidated_constraints = tuple(
            _text_list(value.get("unvalidated_constraints"), "unvalidated_constraints")
        )
        if validation_scope == "cartesian_pick_place":
            expected_checked = (
                "live_tf",
                "collision_aware_cartesian_pick_path",
                "collision_aware_cartesian_transfer_place_path",
                "complete_path_fraction",
            )
            expected_unvalidated = (
                "grasp_contact",
                "gripper_actuation",
                "attached_part_collision_geometry",
                f"{version_three_texts['process_symbol']}_tolerance",
                "force_control",
                "final_constrained_insertion_stroke",
            )
        else:
            expected_checked = (
                "positional_ik",
                "collision_aware_endpoints",
                "path_between_endpoints",
            )
            expected_unvalidated = (
                "grasping",
                "end_effector_orientation",
                "attached_object_geometry",
                f"{version_three_texts['process_symbol']}_tolerance",
                "force_contact",
                "insertion_constraints",
            )
        if (
            checked_constraints != expected_checked
            or unvalidated_constraints != expected_unvalidated
        ):
            raise RAContextHandoffError(
                "SelectedRAAssignmentEnvelope plan-only constraints are invalid."
            )
    return SelectedRAAssignmentEnvelope(
        product_requirement=texts["product_requirement"],
        specification_iri=texts["specification_iri"],
        feature_iri=texts["feature_iri"],
        process_iri=texts["process_iri"],
        process_symbol=version_three_texts.get("process_symbol"),
        selected_resource_iri=texts["selected_resource_iri"],
        selected_resource_jid=texts["selected_resource_jid"],
        selected_execution_mode=texts["selected_execution_mode"],
        selected_resource_symbol=version_three_texts.get("selected_resource_symbol"),
        pa_context_grounding_completion_ref=texts["pa_context_grounding_completion_ref"],
        pa_context_grounding_completion_sha256=shas["pa_context_grounding_completion_sha256"],
        pa_context_grounding_completion_fingerprint=shas[
            "pa_context_grounding_completion_fingerprint"
        ],
        resource_selection_ref=texts["resource_selection_ref"],
        resource_selection_sha256=shas["resource_selection_sha256"],
        resource_selection_fingerprint=shas["resource_selection_fingerprint"],
        resource_assignment_delta_ref=texts["resource_assignment_delta_ref"],
        resource_assignment_delta_sha256=shas["resource_assignment_delta_sha256"],
        current_state_iri=version_two_texts.get("current_state_iri"),
        desired_state_iri=version_two_texts.get("desired_state_iri"),
        reachability_check_ref=version_two_texts.get("reachability_check_ref"),
        reachability_check_sha256=version_two_shas.get("reachability_check_sha256"),
        reachability_check_fingerprint=version_two_shas.get("reachability_check_fingerprint"),
        robot_agent_validation_ref=version_two_texts.get("robot_agent_validation_ref"),
        robot_agent_validation_sha256=version_two_shas.get("robot_agent_validation_sha256"),
        robot_agent_validation_fingerprint=version_two_shas.get(
            "robot_agent_validation_fingerprint"
        ),
        allocation_label=version_two_texts.get("allocation_label"),
        motion_executed=False if schema_version in {2, 3} else None,
        current_state_evidence=current_state_evidence,
        desired_state_evidence=desired_state_evidence,
        registry_snapshot_ref=version_three_texts.get("registry_snapshot_ref"),
        registry_snapshot_sha256=version_three_shas.get("registry_snapshot_sha256"),
        registry_snapshot_fingerprint=version_three_shas.get("registry_snapshot_fingerprint"),
        workcell_snapshot_ref=version_three_texts.get("workcell_snapshot_ref"),
        workcell_snapshot_sha256=version_three_shas.get("workcell_snapshot_sha256"),
        workcell_snapshot_fingerprint=version_three_shas.get("workcell_snapshot_fingerprint"),
        evidence_presentation_ref=version_three_texts.get("evidence_presentation_ref"),
        evidence_presentation_sha256=version_three_shas.get("evidence_presentation_sha256"),
        evidence_presentation_fingerprint=version_three_shas.get(
            "evidence_presentation_fingerprint"
        ),
        allocation_presentation_ref=version_three_texts.get("allocation_presentation_ref"),
        allocation_presentation_sha256=version_three_shas.get("allocation_presentation_sha256"),
        allocation_presentation_fingerprint=version_three_shas.get(
            "allocation_presentation_fingerprint"
        ),
        validation_scope=version_three_texts.get("validation_scope"),
        checked_constraints=checked_constraints,
        unvalidated_constraints=unvalidated_constraints,
        fingerprint=fingerprint,
    )


def _validate_runtime_response(
    response: Mapping[str, object],
    assignment: SelectedRAAssignmentEnvelope,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not isinstance(response, Mapping):
        raise RAContextHandoffError("Selected RA context response must be an object.")
    _require_exact_keys(
        response,
        {"resource_jid", "assignment_fingerprint", "robot_state", "primitive_catalog"},
        "selected RA context response",
    )
    if response.get("resource_jid") != assignment.selected_resource_jid:
        raise RAContextHandoffError("Selected RA response JID does not match assignment.")
    if response.get("assignment_fingerprint") != assignment.fingerprint:
        raise RAContextHandoffError("Selected RA response assignment fingerprint does not match.")
    state_value = response.get("robot_state")
    if not isinstance(state_value, Mapping) or not state_value:
        raise RAContextHandoffError("Selected RA robot_state must be a non-empty object.")
    robot_state = _json_mapping(state_value, "robot_state")
    primitive_catalog = _validate_primitive_catalog(response.get("primitive_catalog"))
    return robot_state, primitive_catalog


def _validate_primitive_catalog(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise RAContextHandoffError("Selected RA primitive_catalog must be non-empty.")
    result: list[dict[str, object]] = []
    symbols: set[str] = set()
    for index, raw_entry in enumerate(value):
        if not isinstance(raw_entry, Mapping):
            raise RAContextHandoffError("Primitive catalog entry must be an object.")
        keys = set(raw_entry)
        if not _CATALOG_REQUIRED_KEYS.issubset(keys) or not keys.issubset(
            _CATALOG_REQUIRED_KEYS | _CATALOG_OPTIONAL_KEYS
        ):
            raise RAContextHandoffError(f"Primitive catalog entry {index} has unexpected fields.")
        symbol = _required_text(
            raw_entry.get("primitive_symbol"),
            f"primitive_catalog[{index}].primitive_symbol",
        )
        if symbol in symbols:
            raise RAContextHandoffError("Primitive catalog symbols must be unique.")
        symbols.add(symbol)
        _required_text(
            raw_entry.get("operation_description"),
            f"primitive_catalog[{index}].operation_description",
        )
        _required_text(
            raw_entry.get("invocation_binding"),
            f"primitive_catalog[{index}].invocation_binding",
        )
        _validate_typed_parameters(raw_entry.get("typed_parameters"), index)
        _validate_typed_results(raw_entry.get("typed_results"), index)
        for field in ("truthful_limits", "direct_evidence", "evaluator_endpoints"):
            _text_list(raw_entry.get(field), f"primitive_catalog[{index}].{field}")
        for field in _CATALOG_OPTIONAL_KEYS & keys:
            if not isinstance(raw_entry.get(field), Mapping):
                raise RAContextHandoffError(
                    f"primitive_catalog[{index}].{field} must be an object."
                )
        result.append(_json_mapping(raw_entry, f"primitive_catalog[{index}]"))
    return result


def _validate_typed_parameters(value: object, entry_index: int) -> None:
    if not isinstance(value, list):
        raise RAContextHandoffError("typed_parameters must be a list.")
    names: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"name", "type", "required"}:
            raise RAContextHandoffError("Typed parameter entry is invalid.")
        name = _required_text(
            item.get("name"),
            f"primitive_catalog[{entry_index}].typed_parameters[{index}].name",
        )
        _required_text(
            item.get("type"),
            f"primitive_catalog[{entry_index}].typed_parameters[{index}].type",
        )
        if not isinstance(item.get("required"), bool):
            raise RAContextHandoffError("Typed parameter required must be a boolean.")
        if name in names:
            raise RAContextHandoffError("Typed parameter names must be unique.")
        names.add(name)


def _validate_typed_results(value: object, entry_index: int) -> None:
    if not isinstance(value, list):
        raise RAContextHandoffError("typed_results must be a list.")
    names: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"name", "type"}:
            raise RAContextHandoffError("Typed result entry is invalid.")
        name = _required_text(
            item.get("name"),
            f"primitive_catalog[{entry_index}].typed_results[{index}].name",
        )
        _required_text(
            item.get("type"),
            f"primitive_catalog[{entry_index}].typed_results[{index}].type",
        )
        if name in names:
            raise RAContextHandoffError("Typed result names must be unique.")
        names.add(name)


def _next_snapshot_number(
    root: Path,
    *,
    assignment: SelectedRAAssignmentEnvelope,
    assignment_path: Path,
    selection: Mapping[str, object],
) -> int:
    resource_root = root / "resources" / assignment.selected_resource_jid
    state_directory = resource_root / _ROBOT_STATE_DIRECTORY
    catalog_directory = resource_root / _PRIMITIVE_CATALOG_DIRECTORY
    state_paths = _numbered_snapshot_paths(state_directory)
    catalog_paths = _numbered_snapshot_paths(catalog_directory)
    if set(state_paths) != set(catalog_paths):
        raise RAContextHandoffError(
            "Robot state and primitive catalog snapshot revisions are unpaired."
        )
    numbers = sorted(state_paths)
    if numbers and numbers != list(range(1, numbers[-1] + 1)):
        raise RAContextHandoffError("Selected RA snapshot revisions contain a gap.")
    for number in numbers:
        state_path = state_paths[number]
        state = _load_robot_state_snapshot(state_path)
        catalog = _load_primitive_catalog_snapshot(catalog_paths[number])
        _validate_snapshot_linkage(
            root,
            assignment=assignment,
            assignment_path=assignment_path,
            selection=selection,
            state_path=state_path,
            state=state,
            catalog=catalog,
        )
    next_number = numbers[-1] + 1 if numbers else 1
    if next_number > 9999:
        raise RAContextHandoffError("Selected RA snapshot revision limit is reached.")
    return next_number


def _numbered_snapshot_paths(directory: Path) -> dict[int, Path]:
    if not directory.exists():
        return {}
    paths: dict[int, Path] = {}
    for path in sorted(directory.glob("*.json")):
        match = _SNAPSHOT_NAME_PATTERN.fullmatch(path.name)
        if match is None:
            raise RAContextHandoffError(f"Unexpected snapshot record name: {path.name}.")
        number = int(match.group(1))
        if number < 1 or number in paths:
            raise RAContextHandoffError("Selected RA snapshot revision is invalid.")
        paths[number] = path
    return paths


def _load_robot_state_snapshot(path: Path) -> RobotStateSnapshot:
    value = _read_json_mapping(path, "RobotStateSnapshot")
    expected = {
        "schema_version",
        "record_type",
        "resource_jid",
        "assignment_fingerprint",
        "retrieved_at_ns",
        "robot_state",
        "state_fingerprint",
        "fingerprint",
    }
    _require_exact_keys(value, expected, "RobotStateSnapshot")
    if value.get("schema_version") != 1 or value.get("record_type") != "RobotStateSnapshot":
        raise RAContextHandoffError("RobotStateSnapshot identity is invalid.")
    resource_jid = _required_text(value.get("resource_jid"), "resource_jid")
    _validate_resource_jid(resource_jid)
    assignment_fingerprint = _sha256_text(
        value.get("assignment_fingerprint"), "assignment_fingerprint"
    )
    retrieved_at_ns = _nonnegative_integer(value.get("retrieved_at_ns"), "retrieved_at_ns")
    robot_state_value = value.get("robot_state")
    if not isinstance(robot_state_value, Mapping) or not robot_state_value:
        raise RAContextHandoffError("RobotStateSnapshot.robot_state is invalid.")
    robot_state = _json_mapping(robot_state_value, "RobotStateSnapshot.robot_state")
    state_fingerprint = _sha256_text(value.get("state_fingerprint"), "state_fingerprint")
    if state_fingerprint != _fingerprint(robot_state):
        raise RAContextHandoffError("RobotStateSnapshot state fingerprint is invalid.")
    fingerprint = _record_fingerprint(value, "RobotStateSnapshot")
    return RobotStateSnapshot(
        resource_jid=resource_jid,
        assignment_fingerprint=assignment_fingerprint,
        retrieved_at_ns=retrieved_at_ns,
        robot_state=robot_state,
        state_fingerprint=state_fingerprint,
        fingerprint=fingerprint,
    )


def _load_primitive_catalog_snapshot(path: Path) -> PrimitiveCatalogSnapshot:
    value = _read_json_mapping(path, "PrimitiveCatalogSnapshot")
    expected = {
        "schema_version",
        "record_type",
        "resource_jid",
        "assignment_ref",
        "assignment_sha256",
        "assignment_fingerprint",
        "resource_selection_ref",
        "resource_selection_sha256",
        "resource_selection_fingerprint",
        "robot_state_ref",
        "robot_state_sha256",
        "robot_state_fingerprint",
        "retrieved_at_ns",
        "primitive_catalog",
        "catalog_fingerprint",
        "fingerprint",
    }
    _require_exact_keys(value, expected, "PrimitiveCatalogSnapshot")
    if value.get("schema_version") != 1 or value.get("record_type") != "PrimitiveCatalogSnapshot":
        raise RAContextHandoffError("PrimitiveCatalogSnapshot identity is invalid.")
    resource_jid = _required_text(value.get("resource_jid"), "resource_jid")
    _validate_resource_jid(resource_jid)
    primitive_catalog = _validate_primitive_catalog(value.get("primitive_catalog"))
    catalog_fingerprint = _sha256_text(value.get("catalog_fingerprint"), "catalog_fingerprint")
    if catalog_fingerprint != _fingerprint(primitive_catalog):
        raise RAContextHandoffError("PrimitiveCatalogSnapshot catalog fingerprint is invalid.")
    fingerprint = _record_fingerprint(value, "PrimitiveCatalogSnapshot")
    return PrimitiveCatalogSnapshot(
        resource_jid=resource_jid,
        assignment_ref=_required_text(value.get("assignment_ref"), "assignment_ref"),
        assignment_sha256=_sha256_text(value.get("assignment_sha256"), "assignment_sha256"),
        assignment_fingerprint=_sha256_text(
            value.get("assignment_fingerprint"), "assignment_fingerprint"
        ),
        resource_selection_ref=_required_text(
            value.get("resource_selection_ref"), "resource_selection_ref"
        ),
        resource_selection_sha256=_sha256_text(
            value.get("resource_selection_sha256"), "resource_selection_sha256"
        ),
        resource_selection_fingerprint=_sha256_text(
            value.get("resource_selection_fingerprint"),
            "resource_selection_fingerprint",
        ),
        robot_state_ref=_required_text(value.get("robot_state_ref"), "robot_state_ref"),
        robot_state_sha256=_sha256_text(value.get("robot_state_sha256"), "robot_state_sha256"),
        robot_state_fingerprint=_sha256_text(
            value.get("robot_state_fingerprint"), "robot_state_fingerprint"
        ),
        retrieved_at_ns=_nonnegative_integer(value.get("retrieved_at_ns"), "retrieved_at_ns"),
        primitive_catalog=tuple(primitive_catalog),
        catalog_fingerprint=catalog_fingerprint,
        fingerprint=fingerprint,
    )


def _validate_snapshot_linkage(
    root: Path,
    *,
    assignment: SelectedRAAssignmentEnvelope,
    assignment_path: Path,
    selection: Mapping[str, object],
    state_path: Path,
    state: RobotStateSnapshot,
    catalog: PrimitiveCatalogSnapshot,
) -> None:
    expected_assignment_ref = assignment_path.relative_to(root).as_posix()
    expected_state_ref = state_path.relative_to(root).as_posix()
    if (
        state.resource_jid != assignment.selected_resource_jid
        or state.assignment_fingerprint != assignment.fingerprint
        or catalog.resource_jid != assignment.selected_resource_jid
        or catalog.assignment_ref != expected_assignment_ref
        or catalog.assignment_sha256 != _sha256_path(assignment_path)
        or catalog.assignment_fingerprint != assignment.fingerprint
        or catalog.resource_selection_ref != assignment.resource_selection_ref
        or catalog.resource_selection_sha256 != assignment.resource_selection_sha256
        or catalog.resource_selection_fingerprint != selection["fingerprint"]
        or catalog.robot_state_ref != expected_state_ref
        or catalog.robot_state_sha256 != _sha256_path(state_path)
        or catalog.robot_state_fingerprint != state.fingerprint
        or catalog.retrieved_at_ns != state.retrieved_at_ns
    ):
        raise RAContextHandoffError(
            "Selected RA state and primitive catalog snapshots are not linked."
        )


def _resolve_ref(root: Path, record_ref: str, *, prefix: tuple[str, ...]) -> Path:
    relative = Path(record_ref)
    if (
        relative.is_absolute()
        or not relative.parts
        or "." in relative.parts
        or ".." in relative.parts
        or relative.parts[: len(prefix)] != prefix
    ):
        raise RAContextHandoffError("Phase 5.1 record ref is outside its authority.")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RAContextHandoffError("Phase 5.1 record ref leaves its interaction.") from exc
    return path


def _validate_resource_jid(value: str) -> None:
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in value
        )
        or Path(value).name != value
    ):
        raise RAContextHandoffError("Selected resource_jid is not one exact safe path component.")


def _read_json_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RAContextHandoffError(f"{label} is unavailable or malformed.") from exc
    if not isinstance(value, dict):
        raise RAContextHandoffError(f"{label} must be a JSON object.")
    _validate_json_value(value, label)
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    _validate_json_value(value, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise RAContextHandoffError(f"Phase 5.1 record already exists: {path.name}.") from exc
    except OSError as exc:
        raise RAContextHandoffError(f"Phase 5.1 record could not be written: {path.name}.") from exc


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str] | frozenset[str],
    label: str,
) -> None:
    if set(value) != set(expected):
        raise RAContextHandoffError(f"{label} fields are invalid.")


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RAContextHandoffError(f"{label} must be one exact non-empty string.")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise RAContextHandoffError(f"{label} contains a control character.")
    return value


def _sha256_text(value: object, label: str) -> str:
    text = _required_text(value, label)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise RAContextHandoffError(f"{label} must be a lowercase SHA-256 value.")
    return text


def _text_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise RAContextHandoffError(f"{label} must be a list.")
    return [_required_text(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RAContextHandoffError(f"{label} must be a non-negative integer.")
    return value


def _json_mapping(value: Mapping[str, object], label: str) -> dict[str, object]:
    _validate_json_value(value, label)
    return deepcopy(dict(value))


def _validate_json_value(value: object, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RAContextHandoffError(f"{label} contains a non-finite number.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{label}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RAContextHandoffError(f"{label} contains a non-string key.")
            _validate_json_value(item, f"{label}.{key}")
        return
    raise RAContextHandoffError(f"{label} contains a non-JSON value.")


def _record_fingerprint(value: Mapping[str, object], label: str) -> str:
    fingerprint = _sha256_text(value.get("fingerprint"), f"{label}.fingerprint")
    payload = dict(value)
    payload.pop("fingerprint")
    if _fingerprint(payload) != fingerprint:
        raise RAContextHandoffError(f"{label} fingerprint is invalid.")
    return fingerprint


def _fingerprint(value: object) -> str:
    _validate_json_value(value, "fingerprint payload")
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
        raise RAContextHandoffError("Phase 5.1 pinned record is unavailable.") from exc

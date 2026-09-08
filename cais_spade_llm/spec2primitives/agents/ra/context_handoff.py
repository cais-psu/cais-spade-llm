from __future__ import annotations

"""Activate one selected RobotAgent and persist its current composition context."""


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
    load_pa_context_grounding_completion,
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SNAPSHOT_NAME_PATTERN = re.compile(r"snapshot_(\d{4})\.json")
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
        "state_locations",
        "evidence_presentation_ref",
        "evidence_presentation_sha256",
        "evidence_presentation_fingerprint",
        "allocation_presentation_ref",
        "allocation_presentation_sha256",
        "allocation_presentation_fingerprint",
        "reachability_check_ref",
        "reachability_check_sha256",
        "reachability_check_fingerprint",
        "selected_resource_symbol",
        "selected_resource_iri",
        "selected_resource_jid",
        "selected_execution_mode",
        "allocation_status",
        "tbox_fingerprint",
        "registry_fingerprint",
        "workcell_fingerprint",
        "fingerprint",
    }
)
_DELTA_KEYS = frozenset(
    {
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
_CATALOG_OPTIONAL_KEYS = frozenset({"conditions", "effects", "parameter_schemas", "result_schemas"})


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
        """Return the exact current assignment record."""
        return {
            "record_type": "SelectedRAAssignmentEnvelope",
            **deepcopy(self.__dict__),
            "current_state_evidence": deepcopy(dict(self.current_state_evidence)),
            "desired_state_evidence": deepcopy(dict(self.desired_state_evidence)),
            "checked_constraints": list(self.checked_constraints),
            "unvalidated_constraints": list(self.unvalidated_constraints),
            "motion_validation_performed": True,
        }

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
    current_assignment, _ = _build_assignment_envelope(root)
    if current_assignment.fingerprint != assignment.fingerprint:
        raise RAContextHandoffError("Resource assignment changed during RobotAgent context capture.")
    retrieved_at_ns = time.time_ns()

    resource_root = root / "resources" / assignment.selected_resource_jid
    state_path = resource_root / _ROBOT_STATE_DIRECTORY / f"snapshot_{snapshot_number:04d}.json"
    catalog_path = (
        resource_root / _PRIMITIVE_CATALOG_DIRECTORY / f"snapshot_{snapshot_number:04d}.json"
    )
    state_payload: dict[str, object] = {
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
                message="RobotAgent context artifacts exist without a grounding completion.",
                failure="SelectedRAAssignmentEnvelope has no grounding completion authority.",
            )
        return Phase51Diagnostic(
            status="waiting_for_phase_4",
            message="Waiting for validated product grounding and arm assignment.",
        )

    try:
        assignment, selection = _build_assignment_envelope(root)
    except RAContextHandoffError as exc:
        return Phase51Diagnostic(
            status="blocked",
            message="The grounding completion cannot authorize the selected RobotAgent handoff.",
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
                message=("ProductAgent selected this RA; context capture has not been requested."),
                **common,
            )
        if assignment_paths != [expected_assignment_path]:
            raise RAContextHandoffError(
                "RobotAgent context capture requires exactly assignment_0001.json."
            )
        persisted_assignment = _assignment_from_mapping(
            _read_json_mapping(
                expected_assignment_path,
                "SelectedRAAssignmentEnvelope",
            )
        )
        if persisted_assignment != assignment:
            raise RAContextHandoffError(
                "Persisted SelectedRAAssignmentEnvelope does not match the grounding completion."
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
            message="Persisted RobotAgent context evidence failed validation.",
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
        raise RAContextHandoffError(
            "Primitive composition requires one completed RobotAgent context capture."
        )
    persisted_assignment = _assignment_from_mapping(
        _read_json_mapping(assignment_path, "SelectedRAAssignmentEnvelope")
    )
    if persisted_assignment != assignment:
        raise RAContextHandoffError(
            "Persisted SelectedRAAssignmentEnvelope does not match the grounding completion."
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
        raise RAContextHandoffError(
            "Primitive composition requires one completed RobotAgent context capture."
        )
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
            "RobotAgent context capture requires one unchanged PAContextGroundingCompletion. "
            "Start a fresh interaction."
        ) from exc
    completion_record = completion.to_record()
    completion_paths = sorted((root / "interaction_record").glob("context_completion_*.json"))
    if len(completion_paths) != 1:
        raise RAContextHandoffError(
            "RobotAgent context capture requires exactly one grounding completion record."
        )
    completion_path = completion_paths[0]
    selection = _load_validated_selection(root, completion_record)
    _validate_assignment_delta(root, completion_record, selection)

    payload: dict[str, object] = {
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
    payload.update(
        {
            "current_state_iri": selection["current_state_iri"],
            "desired_state_iri": selection["desired_state_iri"],
            "reachability_check_ref": selection["reachability_check_ref"],
            "reachability_check_sha256": selection["reachability_check_sha256"],
            "reachability_check_fingerprint": selection["reachability_check_fingerprint"],
            "allocation_label": completion_record["allocation_label"],
            "motion_executed": False,
        }
    )
    state_locations = selection.get("state_locations")
    current_evidence = {
        "location_handles": (
            state_locations.get("current_state") if isinstance(state_locations, Mapping) else None
        )
    }
    desired_evidence = {
        "location_handles": (
            state_locations.get("desired_state") if isinstance(state_locations, Mapping) else None
        )
    }
    payload.update(
        {
            "process_symbol": selection["process_symbol"],
            "selected_resource_symbol": selection["selected_resource_symbol"],
            "current_state_evidence": current_evidence,
            "desired_state_evidence": desired_evidence,
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
            "allocation_presentation_fingerprint": selection["allocation_presentation_fingerprint"],
            "validation_scope": completion_record["validation_scope"],
            "checked_constraints": completion_record["checked_constraints"],
            "unvalidated_constraints": completion_record["unvalidated_constraints"],
        }
    )
    payload["motion_validation_performed"] = True
    payload["fingerprint"] = _fingerprint(payload)
    return _assignment_from_mapping(payload), selection


def _load_validated_selection(
    root: Path,
    completion: Mapping[str, object],
) -> dict[str, object]:
    """Reconstruct the PA resource and exact location-handle choice."""
    selection_ref = _required_text(
        completion.get("resource_selection_ref"),
        "PAContextGroundingCompletion.resource_selection_ref",
    )
    selection_path = _resolve_ref(
        root,
        selection_ref,
        prefix=_RESOURCE_SELECTION_PREFIX,
    )
    selection = _read_json_mapping(selection_path, "ResourceSelectionRecord ")
    expected = set(_SELECTION_KEYS)
    expected.update(
        {
            "ontology_projection_ref",
            "ontology_projection_sha256",
            "validation_scope",
            "motion_validation_performed",
        }
    )
    _require_exact_keys(selection, expected, "ResourceSelectionRecord")
    state_locations = selection.get("state_locations")
    if (
        (selection.get("record_type") != "ResourceSelectionRecord")
        or (selection.get("authority") != "ProductAgent")
        or (selection.get("allocation_status") != "accepted")
        or (not isinstance(state_locations, Mapping))
        or (set(state_locations) != {"current_state", "desired_state"})
        or (
            any(
                not isinstance(state_locations[state_name], list) or not state_locations[state_name]
                for state_name in ("current_state", "desired_state")
            )
        )
    ):
        raise RAContextHandoffError("ResourceSelectionRecord  identity is invalid.")
    if _sha256_path(selection_path) != _sha256_text(
        completion.get("resource_selection_sha256"),
        "PAContextGroundingCompletion.resource_selection_sha256",
    ):
        raise RAContextHandoffError("ResourceSelectionRecord  hash is invalid.")
    selection_fingerprint = _record_fingerprint(
        selection,
        "ResourceSelectionRecord ",
    )
    if (
        selection_fingerprint != completion.get("resource_selection_fingerprint")
        or selection.get("tbox_fingerprint") != completion.get("tbox_fingerprint")
        or selection.get("registry_fingerprint") != completion.get("registry_snapshot_fingerprint")
        or selection.get("workcell_fingerprint") != completion.get("workcell_snapshot_fingerprint")
    ):
        raise RAContextHandoffError("ResourceSelectionRecord  authority is inconsistent.")
    for field in (
        "specification_iri",
        "feature_iri",
        "process_symbol",
        "process_iri",
        "current_state_iri",
        "desired_state_iri",
        "selected_resource_symbol",
        "selected_resource_iri",
        "selected_resource_jid",
        "selected_execution_mode",
    ):
        _required_text(selection.get(field), f"ResourceSelectionRecord.{field}")
    _validate_resource_jid(str(selection["selected_resource_jid"]))
    for ref_field, sha_field, prefix in (
        (
            "reachability_check_ref",
            "reachability_check_sha256",
            ("products", "grounding", "reachability"),
        ),
    ):
        if (
            selection.get(ref_field) != completion.get(ref_field)
            or selection.get(sha_field) != completion.get(sha_field)
            or _sha256_path(_resolve_ref(root, str(selection[ref_field]), prefix=prefix))
            != _sha256_text(selection.get(sha_field), sha_field)
        ):
            raise RAContextHandoffError("ResourceSelectionRecord  verifier lineage is invalid.")
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
        (delta.get("producer") != "resource_grounding_host")
        or (delta_match is None)
        or (delta.get("delta_number") != int(delta_match.group(1)))
        or (delta.get("uncertainty") != [])
        or (delta.get("unresolved_evidence_needs") != [])
        or (delta.get("typed_context_refs") != [])
        or (
            _sha256_path(delta_path)
            != _sha256_text(
                completion.get("resource_assignment_delta_sha256"),
                "PAContextGroundingCompletion.resource_assignment_delta_sha256",
            )
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
    expected_evidence_refs.extend(
        [selection["reachability_check_ref"], selection["ontology_projection_ref"]]
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
        raise RAContextHandoffError(
            "RobotAgent context capture requires one immutable assignment_0001.json record."
        )
    persisted = _assignment_from_mapping(_read_json_mapping(path, "SelectedRAAssignmentEnvelope"))
    if persisted != assignment:
        raise RAContextHandoffError(
            "Persisted SelectedRAAssignmentEnvelope does not match the grounding completion."
        )
    return path


def _assignment_from_mapping(value: Mapping[str, object]) -> SelectedRAAssignmentEnvelope:
    """Read only the current exact assignment shape and its provenance fields."""
    from ..pa.grounding_contracts import _live_location_constraints

    _require_exact_keys(
        value,
        {
            "allocation_presentation_sha256",
            "workcell_snapshot_ref",
            "validation_scope",
            "pa_context_grounding_completion_ref",
            "resource_selection_ref",
            "fingerprint",
            "resource_assignment_delta_ref",
            "evidence_presentation_ref",
            "motion_executed",
            "process_symbol",
            "allocation_presentation_ref",
            "registry_snapshot_ref",
            "process_iri",
            "workcell_snapshot_fingerprint",
            "registry_snapshot_sha256",
            "allocation_presentation_fingerprint",
            "reachability_check_ref",
            "checked_constraints",
            "resource_selection_sha256",
            "current_state_iri",
            "current_state_evidence",
            "selected_execution_mode",
            "record_type",
            "workcell_snapshot_sha256",
            "product_requirement",
            "resource_selection_fingerprint",
            "specification_iri",
            "unvalidated_constraints",
            "selected_resource_jid",
            "motion_validation_performed",
            "reachability_check_fingerprint",
            "evidence_presentation_sha256",
            "desired_state_evidence",
            "feature_iri",
            "selected_resource_symbol",
            "desired_state_iri",
            "reachability_check_sha256",
            "selected_resource_iri",
            "allocation_label",
            "evidence_presentation_fingerprint",
            "pa_context_grounding_completion_sha256",
            "pa_context_grounding_completion_fingerprint",
            "registry_snapshot_fingerprint",
            "resource_assignment_delta_sha256",
        },
        "SelectedRAAssignmentEnvelope",
    )
    if (
        value["record_type"] != "SelectedRAAssignmentEnvelope"
        or value["motion_validation_performed"] is not True
        or value["motion_executed"] is not False
        or value["validation_scope"] != "moveit_state_location_reachability"
        or value["allocation_label"] != "resource assignment validated by MoveIt"
    ):
        raise RAContextHandoffError(
            "SelectedRAAssignmentEnvelope validation scope is invalid. Start a fresh interaction."
        )
    _record_fingerprint(value, "SelectedRAAssignmentEnvelope")
    texts = {
        field: _required_text(value[field], field)
        for field in (
            "product_requirement",
            "specification_iri",
            "feature_iri",
            "process_iri",
            "process_symbol",
            "selected_resource_iri",
            "selected_resource_jid",
            "selected_execution_mode",
            "selected_resource_symbol",
            "pa_context_grounding_completion_ref",
            "resource_selection_ref",
            "resource_assignment_delta_ref",
            "current_state_iri",
            "desired_state_iri",
            "reachability_check_ref",
            "allocation_label",
            "registry_snapshot_ref",
            "workcell_snapshot_ref",
            "evidence_presentation_ref",
            "allocation_presentation_ref",
            "validation_scope",
        )
    }
    shas = {
        field: _sha256_text(value[field], field)
        for field in (
            "pa_context_grounding_completion_sha256",
            "pa_context_grounding_completion_fingerprint",
            "resource_selection_sha256",
            "resource_selection_fingerprint",
            "resource_assignment_delta_sha256",
            "reachability_check_sha256",
            "reachability_check_fingerprint",
            "registry_snapshot_sha256",
            "registry_snapshot_fingerprint",
            "workcell_snapshot_sha256",
            "workcell_snapshot_fingerprint",
            "evidence_presentation_sha256",
            "evidence_presentation_fingerprint",
            "allocation_presentation_sha256",
            "allocation_presentation_fingerprint",
            "fingerprint",
        )
    }
    _validate_resource_jid(texts["selected_resource_jid"])
    states = {}
    for state in ("current_state_evidence", "desired_state_evidence"):
        evidence = _json_mapping(value[state], state)
        _require_exact_keys(evidence, {"location_handles"}, state)
        handles = _text_list(evidence["location_handles"], state)
        if not handles or len(handles) != len(set(handles)):
            raise RAContextHandoffError("Assignment locations must be distinct nonempty handles.")
        states[state] = evidence
    constraints = _live_location_constraints()
    for field, expected in constraints.items():
        if value[field] != expected:
            raise RAContextHandoffError(
                "SelectedRAAssignmentEnvelope plan-only constraints are invalid."
            )
    return SelectedRAAssignmentEnvelope(
        **texts,
        **shas,
        **states,
        motion_executed=False,
        checked_constraints=tuple(value["checked_constraints"]),
        unvalidated_constraints=tuple(value["unvalidated_constraints"]),
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
        _validate_catalog_schemas(raw_entry)
        result.append(_json_mapping(raw_entry, f"primitive_catalog[{index}]"))
    return result


def _validate_catalog_schemas(entry: Mapping[str, object]) -> None:
    """Keep complete declarations consistent with the captured typed catalog."""
    for field, typed_field in (
        ("parameter_schemas", "typed_parameters"),
        ("result_schemas", "typed_results"),
    ):
        if field not in entry:
            continue
        schemas = entry[field]
        typed = entry[typed_field]
        assert isinstance(schemas, Mapping) and isinstance(typed, list)
        if set(schemas) != {item["name"] for item in typed}:
            raise RAContextHandoffError(f"{field} names disagree with {typed_field}.")
        for item in typed:
            schema = schemas[item["name"]]
            declared_type = schema
            if isinstance(schema, Mapping):
                declared_type = schema.get("type")
                if declared_type is None and field == "result_schemas" and schema:
                    declared_type = "object"
            if declared_type != item["type"]:
                raise RAContextHandoffError(f"{field} type disagrees with {typed_field}.")


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
        "record_type",
        "resource_jid",
        "assignment_fingerprint",
        "retrieved_at_ns",
        "robot_state",
        "state_fingerprint",
        "fingerprint",
    }
    _require_exact_keys(value, expected, "RobotStateSnapshot")
    if value.get("record_type") != "RobotStateSnapshot":
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
    if value.get("record_type") != "PrimitiveCatalogSnapshot":
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
        raise RAContextHandoffError("RobotAgent context record ref is outside its authority.")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RAContextHandoffError(
            "RobotAgent context record ref leaves its interaction."
        ) from exc
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
        raise RAContextHandoffError(
            f"RobotAgent context record already exists: {path.name}."
        ) from exc
    except OSError as exc:
        raise RAContextHandoffError(
            f"RobotAgent context record could not be written: {path.name}."
        ) from exc


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str] | frozenset[str],
    label: str,
) -> None:
    if set(value) != set(expected):
        raise RAContextHandoffError(f"{label} fields are invalid. Start a fresh interaction.")


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
        raise RAContextHandoffError("Pinned RobotAgent context record is unavailable.") from exc

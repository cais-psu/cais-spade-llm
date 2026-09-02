"""Ground PA-chosen feature-state evidence to one validated robot allocation."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import Namespace, URIRef
from rdflib.namespace import RDF

from cais_spade_llm.spec2primitives.agents.pa.presentation_records import (
    AllocationEvidenceEntry,
    AllocationPresentationRecord,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    ABoxSnapshot,
    MergeResult,
    commit_host_resource_assignment,
    load_interaction_abox,
)
from cais_spade_llm.spec2primitives.ontology.ppr_tbox import (
    OntologyContextError,
    TBoxSnapshot,
)
from cais_spade_llm.spec2primitives.ontology.resource_registry import (
    ResourceRegistryEntry,
    ResourceRegistrySnapshot,
)
from cais_spade_llm.spec2primitives.ontology.workcell import (
    PredefinedWorkcellSnapshot,
)

_REACHABILITY_ROOT = Path("products/grounding/reachability")
_SELECTION_ROOT = Path("products/grounding/resource_selection")
_REACHABILITY_RECORD_NAME = "reachability_check_record.json"
_SELECTION_RECORD_NAME = "resource_selection_record.json"
_HOST_PRODUCER = "resource_grounding_host"
_PA_AUTHORITY = "ProductAgent"
_TOOL_AUTHORITY = "ProductAgent.check_reachability"
_EXECUTION_ENVIRONMENT = {"simulation": "gazebo", "physical": "real"}
_VALIDATION_STATUSES = frozenset({"accepted", "rejected", "needs_context"})
_PLAN_VALIDATION_KEYS = {
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
    "moveit_group",
    "end_effector_link",
    "target_frame",
    "validation_scope",
    "checked_constraints",
    "unvalidated_constraints",
    "current_state",
    "desired_state",
    "mode",
    "motion_executed",
    "status",
    "feedback",
    "validated_at_ns",
    "request_fingerprint",
    "fingerprint",
}


class ResourceGroundingError(OntologyContextError):
    """Raised when PA-driven resource grounding cannot be trusted."""


class RobotFrameLocationEvidenceError(ResourceGroundingError):
    """Raised when neutral robot-frame location evidence is invalid."""


@dataclass(frozen=True)
class StateReachEvidence:
    """Hold one state location and its manifest-backed reach verdict."""

    state_name: str
    state_iri: str
    evidence_handle: str
    source_record_type: str
    source_record_ref: str
    source_record_sha256: str
    source_field_path: str
    location_record_ref: str
    location_record_sha256: str
    observation_timestamp_ns: int
    translation_m: tuple[float, float, float]
    planar_distance_from_reach_origin_m: float
    distance_from_reach_origin_m: float
    in_workspace: bool
    in_gripper_reach: bool
    reachable: bool
    verdicts: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        """Return the exact JSON-safe state verdict."""
        return {
            "state_name": self.state_name,
            "state_iri": self.state_iri,
            "evidence_handle": self.evidence_handle,
            "source_record_type": self.source_record_type,
            "source_record_ref": self.source_record_ref,
            "source_record_sha256": self.source_record_sha256,
            "source_field_path": self.source_field_path,
            "location_record_ref": self.location_record_ref,
            "location_record_sha256": self.location_record_sha256,
            "observation_timestamp_ns": self.observation_timestamp_ns,
            "translation_m": list(self.translation_m),
            "planar_distance_from_reach_origin_m": (self.planar_distance_from_reach_origin_m),
            "distance_from_reach_origin_m": self.distance_from_reach_origin_m,
            "in_workspace": self.in_workspace,
            "in_gripper_reach": self.in_gripper_reach,
            "reachable": self.reachable,
            "verdicts": list(self.verdicts),
        }


@dataclass(frozen=True)
class StateEvidenceAssignment:
    """Pin one PA-authored semantic-state to neutral-evidence mapping."""

    state_iri: str
    evidence_handle: str
    source_record_type: str
    source_record_ref: str
    source_record_sha256: str
    source_field_path: str

    @classmethod
    def from_reach_evidence(
        cls,
        state: StateReachEvidence,
    ) -> StateEvidenceAssignment:
        """Copy exactly the state mapping accepted by reachability."""
        return cls(
            state_iri=state.state_iri,
            evidence_handle=state.evidence_handle,
            source_record_type=state.source_record_type,
            source_record_ref=state.source_record_ref,
            source_record_sha256=state.source_record_sha256,
            source_field_path=state.source_field_path,
        )

    def to_record(self) -> dict[str, str]:
        """Return the exact JSON-safe state-evidence assignment."""
        return {
            "state_iri": self.state_iri,
            "evidence_handle": self.evidence_handle,
            "source_record_type": self.source_record_type,
            "source_record_ref": self.source_record_ref,
            "source_record_sha256": self.source_record_sha256,
            "source_field_path": self.source_field_path,
        }


@dataclass(frozen=True)
class ReachabilityCheckRecord:
    """Hold one PA-requested two-state reachability check."""

    record_path: Path = field(repr=False, compare=False)
    record_ref: str
    check_number: int
    specification_iri: str
    feature_iri: str
    process_symbol: str
    process_iri: str
    resource_symbol: str
    resource_iri: str
    resource_jid: str
    execution_mode: str
    target_frame: str
    manifest_ref: str
    manifest_sha256: str
    allocation_presentation_ref: str
    allocation_presentation_sha256: str
    allocation_presentation_fingerprint: str
    current_state: StateReachEvidence
    desired_state: StateReachEvidence
    tbox_fingerprint: str
    registry_fingerprint: str
    workcell_fingerprint: str
    status: str
    fingerprint: str
    _interaction_root: Path = field(repr=False, compare=False)

    def to_record(self) -> dict[str, object]:
        """Return the exact persisted reachability record."""
        return {
            "schema_version": 2,
            "record_type": "ReachabilityCheckRecord",
            "check_number": self.check_number,
            "authority": _TOOL_AUTHORITY,
            "specification_iri": self.specification_iri,
            "feature_iri": self.feature_iri,
            "process_symbol": self.process_symbol,
            "process_iri": self.process_iri,
            "resource_symbol": self.resource_symbol,
            "resource_iri": self.resource_iri,
            "resource_jid": self.resource_jid,
            "execution_mode": self.execution_mode,
            "target_frame": self.target_frame,
            "manifest_ref": self.manifest_ref,
            "manifest_sha256": self.manifest_sha256,
            "allocation_presentation_ref": self.allocation_presentation_ref,
            "allocation_presentation_sha256": self.allocation_presentation_sha256,
            "allocation_presentation_fingerprint": self.allocation_presentation_fingerprint,
            "current_state": self.current_state.to_record(),
            "desired_state": self.desired_state.to_record(),
            "tbox_fingerprint": self.tbox_fingerprint,
            "registry_fingerprint": self.registry_fingerprint,
            "workcell_fingerprint": self.workcell_fingerprint,
            "status": self.status,
            "fingerprint": self.fingerprint,
        }

    def assert_unchanged(self) -> None:
        """Raise if the record or either pinned state location changed."""
        persisted = _read_json_mapping(self.record_path, "reachability check record")
        if persisted != self.to_record():
            raise ResourceGroundingError("ReachabilityCheckRecord changed after validation.")
        payload = self.to_record()
        payload.pop("fingerprint")
        if _record_fingerprint(payload) != self.fingerprint:
            raise ResourceGroundingError(
                "ReachabilityCheckRecord fingerprint changed after validation."
            )
        for state in (self.current_state, self.desired_state):
            source_path = _resolve_interaction_ref(
                self._interaction_root,
                state.source_record_ref,
            )
            if _sha256_path(source_path) != state.source_record_sha256:
                raise ResourceGroundingError(
                    f"{state.state_name} source evidence changed after reachability."
                )
            path = _resolve_interaction_ref(
                self._interaction_root,
                state.location_record_ref,
            )
            if _sha256_path(path) != state.location_record_sha256:
                raise ResourceGroundingError(
                    f"{state.state_name} location evidence changed after reachability."
                )
        presentation_path = _resolve_interaction_ref(
            self._interaction_root,
            self.allocation_presentation_ref,
        )
        if _sha256_path(presentation_path) != self.allocation_presentation_sha256:
            raise ResourceGroundingError(
                "AllocationPresentationRecord changed after reachability."
            )


@dataclass(frozen=True)
class ResourceSelectionRecord:
    """Hold one PA-authored provisional choice and RobotAgent verdict."""

    record_path: Path = field(repr=False, compare=False)
    record_ref: str
    selection_number: int
    specification_iri: str
    feature_iri: str
    process_symbol: str
    process_iri: str
    current_state_iri: str
    desired_state_iri: str
    candidate_resource_iris: tuple[str, ...]
    candidate_resource_symbols: tuple[str, ...]
    current_state_evidence: StateEvidenceAssignment
    desired_state_evidence: StateEvidenceAssignment
    evidence_presentation_ref: str
    evidence_presentation_sha256: str
    evidence_presentation_fingerprint: str
    allocation_presentation_ref: str
    allocation_presentation_sha256: str
    allocation_presentation_fingerprint: str
    reachability_check_ref: str
    reachability_check_sha256: str
    reachability_check_fingerprint: str
    provisional_resource_symbol: str
    provisional_resource_iri: str
    provisional_resource_jid: str
    provisional_execution_mode: str
    robot_agent_validation_ref: str
    robot_agent_validation_sha256: str
    robot_agent_validation_fingerprint: str
    robot_agent_validation_status: str
    allocation_status: str
    selected_resource_symbol: str | None
    selected_resource_iri: str | None
    selected_resource_jid: str | None
    selected_execution_mode: str | None
    tbox_fingerprint: str
    registry_fingerprint: str
    workcell_fingerprint: str
    fingerprint: str
    _interaction_root: Path = field(repr=False, compare=False)

    def to_record(self) -> dict[str, object]:
        """Return the exact JSON-safe selection record."""
        return {
            "schema_version": 4,
            "record_type": "ResourceSelectionRecord",
            "selection_number": self.selection_number,
            "authority": _PA_AUTHORITY,
            "specification_iri": self.specification_iri,
            "feature_iri": self.feature_iri,
            "process_symbol": self.process_symbol,
            "process_iri": self.process_iri,
            "current_state_iri": self.current_state_iri,
            "desired_state_iri": self.desired_state_iri,
            "candidate_resource_iris": list(self.candidate_resource_iris),
            "candidate_resource_symbols": list(self.candidate_resource_symbols),
            "current_state_evidence": self.current_state_evidence.to_record(),
            "desired_state_evidence": self.desired_state_evidence.to_record(),
            "evidence_presentation_ref": self.evidence_presentation_ref,
            "evidence_presentation_sha256": self.evidence_presentation_sha256,
            "evidence_presentation_fingerprint": self.evidence_presentation_fingerprint,
            "allocation_presentation_ref": self.allocation_presentation_ref,
            "allocation_presentation_sha256": self.allocation_presentation_sha256,
            "allocation_presentation_fingerprint": self.allocation_presentation_fingerprint,
            "reachability_check_ref": self.reachability_check_ref,
            "reachability_check_sha256": self.reachability_check_sha256,
            "reachability_check_fingerprint": self.reachability_check_fingerprint,
            "provisional_resource_symbol": self.provisional_resource_symbol,
            "provisional_resource_iri": self.provisional_resource_iri,
            "provisional_resource_jid": self.provisional_resource_jid,
            "provisional_execution_mode": self.provisional_execution_mode,
            "robot_agent_validation_ref": self.robot_agent_validation_ref,
            "robot_agent_validation_sha256": self.robot_agent_validation_sha256,
            "robot_agent_validation_fingerprint": (self.robot_agent_validation_fingerprint),
            "robot_agent_validation_status": self.robot_agent_validation_status,
            "allocation_status": self.allocation_status,
            "selected_resource_symbol": self.selected_resource_symbol,
            "selected_resource_iri": self.selected_resource_iri,
            "selected_resource_jid": self.selected_resource_jid,
            "selected_execution_mode": self.selected_execution_mode,
            "tbox_fingerprint": self.tbox_fingerprint,
            "registry_fingerprint": self.registry_fingerprint,
            "workcell_fingerprint": self.workcell_fingerprint,
            "fingerprint": self.fingerprint,
        }

    def assert_unchanged(self) -> None:
        """Raise if the selection or either pinned decision record changed."""
        persisted = _read_json_mapping(self.record_path, "resource selection record")
        if persisted != self.to_record():
            raise ResourceGroundingError("ResourceSelectionRecord changed after validation.")
        pinned = (
            (self.reachability_check_ref, self.reachability_check_sha256),
            (self.robot_agent_validation_ref, self.robot_agent_validation_sha256),
            (self.evidence_presentation_ref, self.evidence_presentation_sha256),
            (self.allocation_presentation_ref, self.allocation_presentation_sha256),
        )
        for record_ref, expected_sha256 in pinned:
            path = _resolve_interaction_ref(self._interaction_root, record_ref)
            if _sha256_path(path) != expected_sha256:
                raise ResourceGroundingError("ResourceSelectionRecord pinned evidence changed.")
        for assignment in (self.current_state_evidence, self.desired_state_evidence):
            source_path = _resolve_interaction_ref(
                self._interaction_root,
                assignment.source_record_ref,
            )
            if _sha256_path(source_path) != assignment.source_record_sha256:
                raise ResourceGroundingError(
                    "ResourceSelectionRecord state evidence changed."
                )
        payload = self.to_record()
        payload.pop("fingerprint")
        if _record_fingerprint(payload) != self.fingerprint:
            raise ResourceGroundingError(
                "ResourceSelectionRecord fingerprint changed after validation."
            )


@dataclass(frozen=True)
class _AllocationContext:
    specification_iri: str
    feature_iri: str
    process_symbol: str
    process_iri: str
    current_state_iri: str
    desired_state_iri: str
    candidate_resource_iris: tuple[str, ...]
    target_frame: str


@dataclass(frozen=True)
class _RobotFrameLocationEvidence:
    path: Path
    record_ref: str
    sha256: str
    observation_timestamp_ns: int
    translation_m: tuple[float, float, float]


def candidate_resource_catalog(
    abox: ABoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
) -> Mapping[str, Mapping[str, str]]:
    """Return capable resources keyed by symbol without a selection priority."""
    _assert_authorities(workcell, registry)
    context = _allocation_context(abox, workcell)
    entries = {entry.resource_iri: entry for entry in registry.resources}
    catalog: dict[str, Mapping[str, str]] = {}
    for resource_iri in context.candidate_resource_iris:
        entry = entries[resource_iri]
        manifest = _load_resource_manifest(entry, _manifest_path(workcell, entry))
        catalog[entry.resource_symbol] = {
            "resource_iri": entry.resource_iri,
            "resource_jid": entry.resource_jid,
            "execution_mode": str(manifest["execution_mode"]),
        }
    return catalog


def check_resource_reachability(  # noqa: PLR0913
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
    resource_symbol: str,
    allocation_presentation: AllocationPresentationRecord,
    current_state_evidence_handle: str,
    desired_state_evidence_handle: str,
    current_location_record_path: Path,
    desired_location_record_path: Path,
    check_number: int = 1,
) -> ReachabilityCheckRecord:
    """Check both feature states for one explicit PA-chosen resource."""
    _validate_positive_integer(check_number, "check_number")
    root = Path(interaction_root).resolve()
    _assert_authorities(workcell, registry, tbox=tbox)
    abox = load_interaction_abox(root, tbox)
    context = _allocation_context(abox, workcell)
    allocation_presentation.assert_unchanged()
    if (
        allocation_presentation.process_symbol != context.process_symbol
        or allocation_presentation.process_iri != context.process_iri
        or allocation_presentation.feature_iri != context.feature_iri
        or allocation_presentation.current_state_iri != context.current_state_iri
        or allocation_presentation.desired_state_iri != context.desired_state_iri
        or resource_symbol not in allocation_presentation.resource_order
    ):
        raise ResourceGroundingError(
            "check_reachability does not match the pinned allocation presentation."
        )
    current_evidence = allocation_presentation.evidence_for_handle(
        current_state_evidence_handle
    )
    desired_evidence = allocation_presentation.evidence_for_handle(
        desired_state_evidence_handle
    )
    _assert_presented_evidence_unchanged(root, current_evidence)
    _assert_presented_evidence_unchanged(root, desired_evidence)
    entries = {
        entry.resource_symbol: entry
        for entry in registry.resources
        if entry.resource_iri in context.candidate_resource_iris
    }
    entry = entries.get(resource_symbol)
    if entry is None:
        raise ResourceGroundingError(
            "check_reachability resource_symbol is not a capable resource."
        )
    manifest_path = _manifest_path(workcell, entry)
    manifest = _load_resource_manifest(entry, manifest_path)
    environment = _resource_environment(manifest, entry.resource_symbol)
    static_capabilities = environment.get("static_capabilities")
    if not isinstance(static_capabilities, Mapping):
        raise ResourceGroundingError(
            f"Resource static_capabilities are missing: {entry.resource_symbol}."
        )
    workspace = _workspace_bounds(static_capabilities.get("workspace_bounds"))
    reach = _gripper_reach(
        static_capabilities.get("gripper_reach"),
        target_frame=context.target_frame,
    )
    current_location = _load_robot_frame_location(
        current_location_record_path,
        root,
        target_frame=context.target_frame,
    )
    desired_location = _load_robot_frame_location(
        desired_location_record_path,
        root,
        target_frame=context.target_frame,
    )
    current_state = _state_reach_evidence(
        "current_state",
        context.current_state_iri,
        current_location,
        evidence=current_evidence,
        workspace=workspace,
        reach=reach,
    )
    desired_state = _state_reach_evidence(
        "desired_state",
        context.desired_state_iri,
        desired_location,
        evidence=desired_evidence,
        workspace=workspace,
        reach=reach,
    )
    status = "accepted" if current_state.reachable and desired_state.reachable else "rejected"
    execution_mode = str(manifest["execution_mode"])
    payload: dict[str, object] = {
        "schema_version": 2,
        "record_type": "ReachabilityCheckRecord",
        "check_number": check_number,
        "authority": _TOOL_AUTHORITY,
        "specification_iri": context.specification_iri,
        "feature_iri": context.feature_iri,
        "process_symbol": context.process_symbol,
        "process_iri": context.process_iri,
        "resource_symbol": entry.resource_symbol,
        "resource_iri": entry.resource_iri,
        "resource_jid": entry.resource_jid,
        "execution_mode": execution_mode,
        "target_frame": context.target_frame,
        "manifest_ref": entry.source_ref,
        "manifest_sha256": entry.source_sha256,
        "allocation_presentation_ref": allocation_presentation.record_ref,
        "allocation_presentation_sha256": _sha256_path(
            allocation_presentation.record_path
        ),
        "allocation_presentation_fingerprint": allocation_presentation.fingerprint,
        "current_state": current_state.to_record(),
        "desired_state": desired_state.to_record(),
        "tbox_fingerprint": tbox.fingerprint,
        "registry_fingerprint": registry.fingerprint,
        "workcell_fingerprint": workcell.fingerprint,
        "status": status,
    }
    payload["fingerprint"] = _record_fingerprint(payload)
    destination = root / _REACHABILITY_ROOT / f"check_{check_number:04d}"
    record_path = destination / _REACHABILITY_RECORD_NAME
    _persist_record(destination, _REACHABILITY_RECORD_NAME, payload)
    result = ReachabilityCheckRecord(
        record_path=record_path,
        record_ref=record_path.relative_to(root).as_posix(),
        check_number=check_number,
        specification_iri=context.specification_iri,
        feature_iri=context.feature_iri,
        process_symbol=context.process_symbol,
        process_iri=context.process_iri,
        resource_symbol=entry.resource_symbol,
        resource_iri=entry.resource_iri,
        resource_jid=entry.resource_jid,
        execution_mode=execution_mode,
        target_frame=context.target_frame,
        manifest_ref=entry.source_ref,
        manifest_sha256=entry.source_sha256,
        allocation_presentation_ref=allocation_presentation.record_ref,
        allocation_presentation_sha256=str(payload["allocation_presentation_sha256"]),
        allocation_presentation_fingerprint=allocation_presentation.fingerprint,
        current_state=current_state,
        desired_state=desired_state,
        tbox_fingerprint=tbox.fingerprint,
        registry_fingerprint=registry.fingerprint,
        workcell_fingerprint=workcell.fingerprint,
        status=status,
        fingerprint=str(payload["fingerprint"]),
        _interaction_root=root,
    )
    result.assert_unchanged()
    return result


def persist_pa_resource_selection(  # noqa: PLR0913
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
    reachability: ReachabilityCheckRecord,
    allocation_presentation: AllocationPresentationRecord,
    robot_agent_validation_path: Path,
    selection_number: int = 1,
) -> ResourceSelectionRecord:
    """Pin one PA choice to accepted reach evidence and one RA verdict."""
    _validate_positive_integer(selection_number, "selection_number")
    root = Path(interaction_root).resolve()
    _assert_authorities(workcell, registry, tbox=tbox)
    abox = load_interaction_abox(root, tbox)
    context = _allocation_context(abox, workcell)
    reachability.assert_unchanged()
    allocation_presentation.assert_unchanged()
    if (
        reachability.status != "accepted"
        or reachability.specification_iri != context.specification_iri
        or reachability.feature_iri != context.feature_iri
        or reachability.process_symbol != context.process_symbol
        or reachability.process_iri != context.process_iri
        or reachability.current_state.state_iri != context.current_state_iri
        or reachability.desired_state.state_iri != context.desired_state_iri
        or reachability.resource_iri not in context.candidate_resource_iris
        or reachability.tbox_fingerprint != tbox.fingerprint
        or reachability.registry_fingerprint != registry.fingerprint
        or reachability.workcell_fingerprint != workcell.fingerprint
        or reachability.allocation_presentation_ref
        != allocation_presentation.record_ref
        or reachability.allocation_presentation_sha256
        != _sha256_path(allocation_presentation.record_path)
        or reachability.allocation_presentation_fingerprint
        != allocation_presentation.fingerprint
    ):
        raise ResourceGroundingError(
            "PA resource choice requires one cited accepted reachability check."
        )
    validation_path = Path(robot_agent_validation_path).resolve()
    try:
        validation_ref = validation_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ResourceGroundingError("RobotAgent validation is outside the interaction.") from exc
    validation = _read_json_mapping(
        validation_path,
        "plan-only RobotAgent validation",
    )
    _validate_plan_only_record(validation, reachability)
    validation_status = str(validation["status"])
    selected_fields: tuple[str | None, str | None, str | None, str | None]
    if validation_status == "accepted":
        selected_fields = (
            reachability.resource_symbol,
            reachability.resource_iri,
            reachability.resource_jid,
            reachability.execution_mode,
        )
    else:
        selected_fields = (None, None, None, None)
    validation_sha256 = _sha256_path(validation_path)
    validation_fingerprint = str(validation["fingerprint"])
    candidate_resource_iris = context.candidate_resource_iris
    candidate_resource_symbols = tuple(
        entry.resource_symbol
        for entry in registry.resources
        if entry.resource_iri in candidate_resource_iris
    )
    presented_resources = {
        (entry.resource_symbol, entry.resource_iri, entry.resource_jid)
        for entry in allocation_presentation.resources
    }
    registry_resources = {
        (entry.resource_symbol, entry.resource_iri, entry.resource_jid)
        for entry in registry.resources
        if entry.resource_iri in candidate_resource_iris
    }
    if presented_resources != registry_resources:
        raise ResourceGroundingError(
            "Allocation presentation candidate resources changed before selection."
        )
    current_state_evidence = StateEvidenceAssignment.from_reach_evidence(
        reachability.current_state
    )
    desired_state_evidence = StateEvidenceAssignment.from_reach_evidence(
        reachability.desired_state
    )
    evidence_presentation_path = _resolve_interaction_ref(
        root,
        allocation_presentation.evidence_presentation_ref,
    )
    payload: dict[str, object] = {
        "schema_version": 4,
        "record_type": "ResourceSelectionRecord",
        "selection_number": selection_number,
        "authority": _PA_AUTHORITY,
        "specification_iri": context.specification_iri,
        "feature_iri": context.feature_iri,
        "process_symbol": context.process_symbol,
        "process_iri": context.process_iri,
        "current_state_iri": context.current_state_iri,
        "desired_state_iri": context.desired_state_iri,
        "candidate_resource_iris": list(candidate_resource_iris),
        "candidate_resource_symbols": list(candidate_resource_symbols),
        "current_state_evidence": current_state_evidence.to_record(),
        "desired_state_evidence": desired_state_evidence.to_record(),
        "evidence_presentation_ref": allocation_presentation.evidence_presentation_ref,
        "evidence_presentation_sha256": _sha256_path(evidence_presentation_path),
        "evidence_presentation_fingerprint": (
            allocation_presentation.evidence_presentation_fingerprint
        ),
        "allocation_presentation_ref": allocation_presentation.record_ref,
        "allocation_presentation_sha256": _sha256_path(
            allocation_presentation.record_path
        ),
        "allocation_presentation_fingerprint": allocation_presentation.fingerprint,
        "reachability_check_ref": reachability.record_ref,
        "reachability_check_sha256": _sha256_path(reachability.record_path),
        "reachability_check_fingerprint": reachability.fingerprint,
        "provisional_resource_symbol": reachability.resource_symbol,
        "provisional_resource_iri": reachability.resource_iri,
        "provisional_resource_jid": reachability.resource_jid,
        "provisional_execution_mode": reachability.execution_mode,
        "robot_agent_validation_ref": validation_ref,
        "robot_agent_validation_sha256": validation_sha256,
        "robot_agent_validation_fingerprint": validation_fingerprint,
        "robot_agent_validation_status": validation_status,
        "allocation_status": validation_status,
        "selected_resource_symbol": selected_fields[0],
        "selected_resource_iri": selected_fields[1],
        "selected_resource_jid": selected_fields[2],
        "selected_execution_mode": selected_fields[3],
        "tbox_fingerprint": tbox.fingerprint,
        "registry_fingerprint": registry.fingerprint,
        "workcell_fingerprint": workcell.fingerprint,
    }
    payload["fingerprint"] = _record_fingerprint(payload)
    destination = root / _SELECTION_ROOT / f"selection_{selection_number:04d}"
    record_path = destination / _SELECTION_RECORD_NAME
    _persist_record(destination, _SELECTION_RECORD_NAME, payload)
    result = ResourceSelectionRecord(
        record_path=record_path,
        record_ref=record_path.relative_to(root).as_posix(),
        selection_number=selection_number,
        specification_iri=context.specification_iri,
        feature_iri=context.feature_iri,
        process_symbol=context.process_symbol,
        process_iri=context.process_iri,
        current_state_iri=context.current_state_iri,
        desired_state_iri=context.desired_state_iri,
        candidate_resource_iris=candidate_resource_iris,
        candidate_resource_symbols=candidate_resource_symbols,
        current_state_evidence=current_state_evidence,
        desired_state_evidence=desired_state_evidence,
        evidence_presentation_ref=allocation_presentation.evidence_presentation_ref,
        evidence_presentation_sha256=str(payload["evidence_presentation_sha256"]),
        evidence_presentation_fingerprint=(
            allocation_presentation.evidence_presentation_fingerprint
        ),
        allocation_presentation_ref=allocation_presentation.record_ref,
        allocation_presentation_sha256=str(payload["allocation_presentation_sha256"]),
        allocation_presentation_fingerprint=allocation_presentation.fingerprint,
        reachability_check_ref=reachability.record_ref,
        reachability_check_sha256=str(payload["reachability_check_sha256"]),
        reachability_check_fingerprint=reachability.fingerprint,
        provisional_resource_symbol=reachability.resource_symbol,
        provisional_resource_iri=reachability.resource_iri,
        provisional_resource_jid=reachability.resource_jid,
        provisional_execution_mode=reachability.execution_mode,
        robot_agent_validation_ref=validation_ref,
        robot_agent_validation_sha256=validation_sha256,
        robot_agent_validation_fingerprint=validation_fingerprint,
        robot_agent_validation_status=validation_status,
        allocation_status=validation_status,
        selected_resource_symbol=selected_fields[0],
        selected_resource_iri=selected_fields[1],
        selected_resource_jid=selected_fields[2],
        selected_execution_mode=selected_fields[3],
        tbox_fingerprint=tbox.fingerprint,
        registry_fingerprint=registry.fingerprint,
        workcell_fingerprint=workcell.fingerprint,
        fingerprint=str(payload["fingerprint"]),
        _interaction_root=root,
    )
    result.assert_unchanged()
    return result


def commit_resource_assignment(
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
    selection: ResourceSelectionRecord,
) -> MergeResult:
    """Commit exactly one RobotAgent-accepted PA resource assignment."""
    root = Path(interaction_root).resolve()
    _assert_authorities(workcell, registry, tbox=tbox)
    abox = load_interaction_abox(root, tbox)
    context = _allocation_context(abox, workcell)
    selection.assert_unchanged()
    if (
        selection.allocation_status != "accepted"
        or selection.robot_agent_validation_status != "accepted"
        or selection.selected_resource_iri is None
        or selection.selected_resource_iri != selection.provisional_resource_iri
        or selection.specification_iri != context.specification_iri
        or selection.feature_iri != context.feature_iri
        or selection.process_symbol != context.process_symbol
        or selection.process_iri != context.process_iri
        or selection.current_state_iri != context.current_state_iri
        or selection.desired_state_iri != context.desired_state_iri
        or selection.selected_resource_iri not in context.candidate_resource_iris
        or set(selection.candidate_resource_iris) != set(context.candidate_resource_iris)
        or selection.current_state_evidence.state_iri != context.current_state_iri
        or selection.desired_state_evidence.state_iri != context.desired_state_iri
    ):
        raise ResourceGroundingError(
            "Resource assignment requires the accepted PA choice without substitution."
        )
    ppr = Namespace(tbox.ppr_namespace)
    execution_iri = f"{abox.namespace}process_execution_0001"
    evidence_refs = [
        selection.record_ref,
        selection.reachability_check_ref,
        selection.robot_agent_validation_ref,
    ]
    assertions = [
        _iri_assertion(
            context.specification_iri,
            str(ppr.hasProcessExecution),
            execution_iri,
            evidence_refs,
        ),
        _iri_assertion(
            execution_iri,
            str(RDF.type),
            str(ppr.processExecution),
            evidence_refs,
        ),
        _iri_assertion(
            execution_iri,
            str(ppr.runsProcess),
            context.process_iri,
            evidence_refs,
        ),
        _iri_assertion(
            execution_iri,
            str(ppr.runsOnResource),
            selection.selected_resource_iri,
            evidence_refs,
        ),
    ]
    return commit_host_resource_assignment(
        root,
        tbox,
        producer=_HOST_PRODUCER,
        process_iri=context.process_iri,
        resource_iris=workcell.resource_iris,
        assertions=assertions,
        authorized_evidence_refs=evidence_refs,
    )


def _allocation_context(
    abox: ABoxSnapshot,
    workcell: PredefinedWorkcellSnapshot,
) -> _AllocationContext:
    if not isinstance(abox, ABoxSnapshot):
        raise ResourceGroundingError("Allocation requires an ABoxSnapshot.")
    if not isinstance(workcell, PredefinedWorkcellSnapshot):
        raise ResourceGroundingError("Allocation requires a PredefinedWorkcellSnapshot.")
    workcell.assert_unchanged()
    if abox.tbox_fingerprint != workcell.tbox_fingerprint:
        raise ResourceGroundingError("Allocation ABox and Workcell TBox fingerprints differ.")
    ppr = Namespace(workcell.ppr_namespace)
    specification = URIRef(abox.specification_iri)
    if any(abox.graph.objects(specification, ppr.hasProcessExecution)):
        raise ResourceGroundingError("The feature transformation already has a processExecution.")
    defined_features = {
        feature
        for feature in abox.graph.objects(specification, ppr.defines)
        if isinstance(feature, URIRef) and (feature, RDF.type, ppr.feature) in abox.graph
    }
    if len(defined_features) != 1:
        raise ResourceGroundingError("Allocation must identify exactly one grounded feature.")
    feature = next(iter(defined_features))
    authorized_process_iris = {iri for _symbol, iri in workcell.processes}
    realizing_processes = {
        process
        for process in abox.graph.subjects(ppr.realizes, feature)
        if isinstance(process, URIRef)
        and str(process) in authorized_process_iris
        and (process, RDF.type, ppr.process) in workcell.graph
    }
    if len(realizing_processes) != 1:
        raise ResourceGroundingError(
            "Allocation must identify exactly one authorized process for the feature."
        )
    process = next(iter(realizing_processes))
    process_iri = str(process)
    process_symbol = workcell.process_symbol_for_iri(process_iri)
    current_states = {
        state
        for state in abox.graph.objects(feature, ppr.hascurrentstate)
        if isinstance(state, URIRef) and (state, RDF.type, ppr.state) in abox.graph
    }
    desired_states = {
        state
        for state in abox.graph.objects(feature, ppr.hasdesiredstate)
        if isinstance(state, URIRef) and (state, RDF.type, ppr.state) in abox.graph
    }
    if len(current_states) != 1 or len(desired_states) != 1:
        raise ResourceGroundingError(
            "Allocation requires exactly one currentstate and desiredstate."
        )
    candidate_resource_iris = tuple(
        resource_iri
        for resource_iri in workcell.capable_resource_iris(process_iri)
        if (URIRef(resource_iri), ppr.capableOf, process) in workcell.graph
        and (URIRef(resource_iri), RDF.type, ppr.resource) in workcell.graph
    )
    if not candidate_resource_iris:
        raise ResourceGroundingError("No predefined resource is capable of the required process.")
    return _AllocationContext(
        specification_iri=abox.specification_iri,
        feature_iri=str(feature),
        process_symbol=process_symbol,
        process_iri=process_iri,
        current_state_iri=str(next(iter(current_states))),
        desired_state_iri=str(next(iter(desired_states))),
        candidate_resource_iris=candidate_resource_iris,
        target_frame=_candidate_target_frame(workcell, candidate_resource_iris),
    )


def _state_reach_evidence(
    state_name: str,
    state_iri: str,
    location: _RobotFrameLocationEvidence,
    *,
    evidence: AllocationEvidenceEntry,
    workspace: tuple[float, float, float, float, float, float],
    reach: tuple[float, float, float, float, float, float, float],
) -> StateReachEvidence:
    x, y, z = location.translation_m
    origin_x, origin_y, origin_z, max_radius, z_min, z_max, tolerance = reach
    in_workspace = (
        workspace[0] - tolerance <= x <= workspace[1] + tolerance
        and workspace[2] - tolerance <= y <= workspace[3] + tolerance
        and workspace[4] - tolerance <= z <= workspace[5] + tolerance
    )
    planar_distance = math.hypot(x - origin_x, y - origin_y)
    distance = math.sqrt((x - origin_x) ** 2 + (y - origin_y) ** 2 + (z - origin_z) ** 2)
    in_gripper_reach = (
        planar_distance <= max_radius + tolerance and z_min - tolerance <= z <= z_max + tolerance
    )
    verdicts: list[str] = []
    if not in_workspace:
        verdicts.append("outside_workspace_bounds")
    if not in_gripper_reach:
        verdicts.append("outside_gripper_reach")
    reachable = not verdicts
    if reachable:
        verdicts.append("coarsely_reachable")
    return StateReachEvidence(
        state_name=state_name,
        state_iri=state_iri,
        evidence_handle=evidence.pa_handle,
        source_record_type=evidence.record_type,
        source_record_ref=evidence.record_ref,
        source_record_sha256=evidence.record_sha256,
        source_field_path=evidence.field_path,
        location_record_ref=location.record_ref,
        location_record_sha256=location.sha256,
        observation_timestamp_ns=location.observation_timestamp_ns,
        translation_m=location.translation_m,
        planar_distance_from_reach_origin_m=planar_distance,
        distance_from_reach_origin_m=distance,
        in_workspace=in_workspace,
        in_gripper_reach=in_gripper_reach,
        reachable=reachable,
        verdicts=tuple(verdicts),
    )


def _assert_presented_evidence_unchanged(
    interaction_root: Path,
    evidence: AllocationEvidenceEntry,
) -> None:
    path = _resolve_interaction_ref(interaction_root, evidence.record_ref)
    if _sha256_path(path) != evidence.record_sha256:
        raise ResourceGroundingError(
            "PA-selected state evidence changed before reachability."
        )


def _load_robot_frame_location(
    path: Path,
    interaction_root: Path,
    *,
    target_frame: str,
) -> _RobotFrameLocationEvidence:
    root = Path(interaction_root).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
        source = resolved.read_bytes()
        record = json.loads(source.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RobotFrameLocationEvidenceError(
            "RobotFrameLocationRecord could not be read from this interaction."
        ) from exc
    if not isinstance(record, Mapping):
        raise RobotFrameLocationEvidenceError("RobotFrameLocationRecord must be a JSON object.")
    schema_version = record.get("schema_version")
    if (
        schema_version not in {1, 2}
        or record.get("record_type") != "RobotFrameLocationRecord"
        or record.get("target_frame") != target_frame
        or record.get("robot_frame_conversion") != "accepted"
        or record.get("location") != "available"
        or (schema_version == 1 and record.get("CAD_correspondence") != "accepted")
    ):
        raise RobotFrameLocationEvidenceError(
            "Reachability requires accepted neutral robot-frame location evidence."
        )
    observation_timestamp_ns = record.get("observation_timestamp_ns")
    if (
        isinstance(observation_timestamp_ns, bool)
        or not isinstance(observation_timestamp_ns, int)
        or observation_timestamp_ns < 0
    ):
        raise RobotFrameLocationEvidenceError(
            "RobotFrameLocationRecord observation timestamp is invalid."
        )
    translation = _finite_vector3(
        record.get("translated_location_m"),
        "RobotFrameLocationRecord translated_location_m",
    )
    if _validate_embedded_hash_refs(record, root) < 1:
        raise RobotFrameLocationEvidenceError(
            "RobotFrameLocationRecord requires embedded hash references."
        )
    return _RobotFrameLocationEvidence(
        path=resolved,
        record_ref=relative.as_posix(),
        sha256=hashlib.sha256(source).hexdigest(),
        observation_timestamp_ns=observation_timestamp_ns,
        translation_m=translation,
    )


def _validate_plan_only_record(
    validation: Mapping[str, object],
    reachability: ReachabilityCheckRecord,
) -> None:
    payload = dict(validation)
    fingerprint = payload.pop("fingerprint", None)
    if (
        set(validation) != _PLAN_VALIDATION_KEYS
        or validation.get("schema_version") != 2
        or validation.get("record_type") != "PlanOnlyFeasibilityValidationRecord"
        or validation.get("validator_authority") != reachability.resource_jid
        or validation.get("process_symbol") != reachability.process_symbol
        or validation.get("process_iri") != reachability.process_iri
        or validation.get("feature_iri") != reachability.feature_iri
        or validation.get("current_state_iri") != reachability.current_state.state_iri
        or validation.get("desired_state_iri") != reachability.desired_state.state_iri
        or validation.get("resource_symbol") != reachability.resource_symbol
        or validation.get("resource_iri") != reachability.resource_iri
        or validation.get("resource_jid") != reachability.resource_jid
        or validation.get("execution_mode") != reachability.execution_mode
        or validation.get("target_frame") != reachability.target_frame
        or validation.get("validation_scope") != "endpoint_motion"
        or validation.get("checked_constraints")
        != ["positional_ik", "collision_aware_endpoints", "path_between_endpoints"]
        or validation.get("unvalidated_constraints")
        != [
            "grasping",
            "end_effector_orientation",
            "attached_object_geometry",
            f"{reachability.process_symbol}_tolerance",
            "force_contact",
            "insertion_constraints",
        ]
        or validation.get("mode") != "plan_only"
        or validation.get("motion_executed") is not False
        or validation.get("status") not in _VALIDATION_STATUSES
        or validation.get("request_fingerprint") != reachability.fingerprint
        or not _is_sha256(fingerprint)
        or _record_fingerprint(payload) != fingerprint
    ):
        raise ResourceGroundingError("RobotAgent plan-only validation record is inconsistent.")


def _assert_authorities(
    workcell: PredefinedWorkcellSnapshot,
    registry: ResourceRegistrySnapshot,
    *,
    tbox: TBoxSnapshot | None = None,
) -> None:
    if not isinstance(registry, ResourceRegistrySnapshot):
        raise ResourceGroundingError("Resource grounding requires a ResourceRegistrySnapshot.")
    if not isinstance(workcell, PredefinedWorkcellSnapshot):
        raise ResourceGroundingError("Resource grounding requires a PredefinedWorkcellSnapshot.")
    if tbox is not None and not isinstance(tbox, TBoxSnapshot):
        raise ResourceGroundingError("Resource grounding requires a TBoxSnapshot.")
    try:
        registry.assert_unchanged()
        workcell.assert_unchanged()
        if tbox is not None:
            tbox.assert_unchanged()
    except OntologyContextError as exc:
        raise ResourceGroundingError(
            "Resource-grounding authority changed after validation."
        ) from exc
    if (
        workcell.registry_fingerprint != registry.fingerprint
        or workcell.tbox_fingerprint != registry.tbox_fingerprint
        or (tbox is not None and workcell.tbox_fingerprint != tbox.fingerprint)
    ):
        raise ResourceGroundingError("Resource-grounding authorities do not share fingerprints.")


def _candidate_target_frame(
    workcell: PredefinedWorkcellSnapshot,
    candidate_resource_iris: Sequence[str],
) -> str:
    entries = {entry.resource_iri: entry for entry in workcell._registry.resources}
    frames = {
        str(
            _resource_environment(
                _load_resource_manifest(
                    entries[resource_iri],
                    _manifest_path(workcell, entries[resource_iri]),
                ),
                entries[resource_iri].resource_symbol,
            )["static_capabilities"]["gripper_reach"]["frame"]
        )
        for resource_iri in candidate_resource_iris
    }
    if len(frames) != 1 or not next(iter(frames), ""):
        raise ResourceGroundingError("Capable resources declare inconsistent gripper_reach frames.")
    return next(iter(frames))


def _manifest_path(
    workcell: PredefinedWorkcellSnapshot,
    entry: ResourceRegistryEntry,
) -> Path:
    matches = [
        item.manifest_path.resolve()
        for item in workcell._profile.resources
        if item.symbol == entry.resource_symbol
    ]
    if len(matches) != 1:
        raise ResourceGroundingError(f"Resource manifest is unavailable: {entry.resource_symbol}.")
    return matches[0]


def _load_resource_manifest(
    entry: ResourceRegistryEntry,
    manifest_path: Path,
) -> Mapping[str, object]:
    source = _read_bytes(manifest_path, f"{entry.resource_symbol} manifest")
    if hashlib.sha256(source).hexdigest() != entry.source_sha256:
        raise ResourceGroundingError(
            f"Resource manifest hash does not match registry: {entry.resource_symbol}."
        )
    try:
        payload = json.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResourceGroundingError(
            f"Resource manifest is malformed: {entry.resource_symbol}."
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != {entry.resource_symbol}:
        raise ResourceGroundingError(f"Resource manifest symbol changed: {entry.resource_symbol}.")
    resource = payload[entry.resource_symbol]
    if (
        not isinstance(resource, Mapping)
        or resource.get("type") != "robot"
        or resource.get("jid") != entry.resource_jid
        or resource.get("execution_mode") not in _EXECUTION_ENVIRONMENT
    ):
        raise ResourceGroundingError(
            f"Resource manifest identity changed: {entry.resource_symbol}."
        )
    return resource


def _resource_environment(
    resource: Mapping[str, object],
    resource_symbol: str,
) -> Mapping[str, object]:
    execution_mode = str(resource.get("execution_mode"))
    environment = resource.get(_EXECUTION_ENVIRONMENT.get(execution_mode, ""))
    if not isinstance(environment, Mapping):
        raise ResourceGroundingError(
            f"Resource execution environment is missing: {resource_symbol}."
        )
    return environment


def _workspace_bounds(
    value: object,
) -> tuple[float, float, float, float, float, float]:
    if not isinstance(value, Mapping):
        raise ResourceGroundingError("Resource workspace_bounds are invalid.")
    bounds = tuple(
        _finite_number(value.get(key), f"workspace_bounds.{key}")
        for key in (
            "x_min_m",
            "x_max_m",
            "y_min_m",
            "y_max_m",
            "z_min_m",
            "z_max_m",
        )
    )
    if bounds[0] > bounds[1] or bounds[2] > bounds[3] or bounds[4] > bounds[5]:
        raise ResourceGroundingError("Resource workspace_bounds are inconsistent.")
    return bounds  # type: ignore[return-value]


def _gripper_reach(
    value: object,
    *,
    target_frame: str,
) -> tuple[float, float, float, float, float, float, float]:
    if not isinstance(value, Mapping) or value.get("frame") != target_frame:
        raise ResourceGroundingError(
            "Resource gripper_reach frame does not match the grounding frame."
        )
    origin = value.get("origin_pose")
    if not isinstance(origin, Mapping):
        raise ResourceGroundingError("Resource gripper_reach origin_pose is invalid.")
    result = (
        _finite_number(origin.get("x"), "gripper_reach.origin_pose.x"),
        _finite_number(origin.get("y"), "gripper_reach.origin_pose.y"),
        _finite_number(origin.get("z"), "gripper_reach.origin_pose.z"),
        _finite_number(value.get("max_xy_radius_m"), "gripper_reach.max_xy_radius_m"),
        _finite_number(value.get("z_min_m"), "gripper_reach.z_min_m"),
        _finite_number(value.get("z_max_m"), "gripper_reach.z_max_m"),
        _finite_number(value.get("tolerance_m"), "gripper_reach.tolerance_m"),
    )
    if result[3] < 0 or result[4] > result[5] or result[6] < 0:
        raise ResourceGroundingError("Resource gripper_reach is inconsistent.")
    return result


def _validate_embedded_hash_refs(value: object, interaction_root: Path) -> int:
    validated = 0
    if isinstance(value, Mapping):
        if "ref" in value or "sha256" in value:
            ref = value.get("ref")
            sha256 = value.get("sha256")
            if not isinstance(ref, str) or not ref or not _is_sha256(sha256):
                raise RobotFrameLocationEvidenceError(
                    "RobotFrameLocationRecord embedded hash reference is incomplete."
                )
            source_path = _resolve_interaction_ref(interaction_root, ref)
            if _sha256_path(source_path) != sha256:
                raise RobotFrameLocationEvidenceError(
                    f"RobotFrameLocationRecord embedded hash does not match: {ref}."
                )
            validated += 1
        for item in value.values():
            validated += _validate_embedded_hash_refs(item, interaction_root)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            validated += _validate_embedded_hash_refs(item, interaction_root)
    return validated


def _iri_assertion(
    subject: str,
    predicate: str,
    object_iri: str,
    evidence_refs: Sequence[str],
) -> dict[str, object]:
    return {
        "subject": subject,
        "predicate": predicate,
        "object": {"kind": "iri", "value": object_iri},
        "evidence_refs": list(evidence_refs),
    }


def _persist_record(
    destination: Path,
    record_name: str,
    payload: Mapping[str, object],
) -> None:
    if destination.exists():
        raise ResourceGroundingError(f"Grounding record already exists: {destination.name}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    except OSError as exc:
        raise ResourceGroundingError(
            "Grounding record staging directory could not be created."
        ) from exc
    try:
        with (temporary / record_name).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temporary.rename(destination)
    except (OSError, TypeError, ValueError) as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise ResourceGroundingError("Grounding record persistence failed.") from exc


def _resolve_interaction_ref(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ResourceGroundingError("Interaction evidence ref is invalid.")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ResourceGroundingError("Interaction evidence ref leaves its interaction.") from exc
    return path


def _validate_positive_integer(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ResourceGroundingError(f"{label} must be a positive integer.")


def _finite_vector3(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ResourceGroundingError(f"{label} must contain exactly three values.")
    result = tuple(_finite_number(item, label) for item in value)
    return result  # type: ignore[return-value]


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResourceGroundingError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ResourceGroundingError(f"{label} must be finite.")
    return result


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise ResourceGroundingError(f"{label} could not be read.") from exc


def _read_json_mapping(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResourceGroundingError(f"{label} could not be read.") from exc
    if not isinstance(value, Mapping):
        raise ResourceGroundingError(f"{label} must be a JSON object.")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _record_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ReachabilityCheckRecord",
    "ResourceGroundingError",
    "ResourceSelectionRecord",
    "RobotFrameLocationEvidenceError",
    "StateReachEvidence",
    "candidate_resource_catalog",
    "check_resource_reachability",
    "commit_resource_assignment",
    "persist_pa_resource_selection",
]

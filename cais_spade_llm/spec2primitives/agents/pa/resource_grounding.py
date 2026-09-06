from __future__ import annotations

"""Ground PA-chosen feature-state evidence to one validated robot allocation."""


import hashlib
import json
import math
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ra.feasibility_validation import RobotAgentFeasibilityRuntime

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


class ResourceGroundingError(OntologyContextError):
    """Raised when PA-driven resource grounding cannot be trusted."""


class RobotFrameLocationEvidenceError(ResourceGroundingError):
    """Raised when neutral robot-frame location evidence is invalid."""


@dataclass(frozen=True)
class StateLocationEvidence:
    """Pin one bound position and its live MoveIt planning verdict."""

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
    reachable: bool

    def to_record(self) -> dict[str, object]:
        """Return the source binding without example workspace metadata."""
        return {**self.__dict__, "translation_m": list(self.translation_m)}


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
    tbox_fingerprint: str
    registry_fingerprint: str
    workcell_fingerprint: str
    status: str
    fingerprint: str
    _interaction_root: Path = field(repr=False, compare=False)
    state_locations: Mapping[str, tuple[StateLocationEvidence, ...]] | None = None
    validation: Mapping[str, object] | None = None

    def to_record(self) -> dict[str, object]:
        """Return the exact persisted reachability record."""
        payload: dict[str, object] = {
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
        }
        if self.state_locations is None:
            raise ResourceGroundingError("ReachabilityCheckRecord  has no state locations.")
        payload["state_locations"] = {
            state_name: [item.to_record() for item in locations]
            for state_name, locations in self.state_locations.items()
        }
        payload["validation"] = self.validation
        payload.update(
            {
                "tbox_fingerprint": self.tbox_fingerprint,
                "registry_fingerprint": self.registry_fingerprint,
                "workcell_fingerprint": self.workcell_fingerprint,
                "status": self.status,
                "fingerprint": self.fingerprint,
            }
        )
        return payload

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
        state_evidence = tuple(
            item for locations in (self.state_locations or {}).values() for item in locations
        )
        for state in state_evidence:
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
            raise ResourceGroundingError("AllocationPresentationRecord changed after reachability.")


@dataclass(frozen=True)
class ResourceSelectionRecord:
    """Hold one PA choice and its pinned validation evidence."""

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
    evidence_presentation_ref: str
    evidence_presentation_sha256: str
    evidence_presentation_fingerprint: str
    allocation_presentation_ref: str
    allocation_presentation_sha256: str
    allocation_presentation_fingerprint: str
    reachability_check_ref: str
    reachability_check_sha256: str
    reachability_check_fingerprint: str
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
    state_location_handles: Mapping[str, tuple[str, ...]] | None = None
    ontology_projection_ref: str | None = None
    ontology_projection_sha256: str | None = None

    def to_record(self) -> dict[str, object]:
        """Return the exact JSON-safe selection record."""
        if self.state_location_handles is None:
            raise ResourceGroundingError("ResourceSelectionRecord  has no state locations.")
        record = {
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
            "state_locations": {
                state_name: list(handles)
                for state_name, handles in self.state_location_handles.items()
            },
            "evidence_presentation_ref": self.evidence_presentation_ref,
            "evidence_presentation_sha256": self.evidence_presentation_sha256,
            "evidence_presentation_fingerprint": self.evidence_presentation_fingerprint,
            "allocation_presentation_ref": self.allocation_presentation_ref,
            "allocation_presentation_sha256": self.allocation_presentation_sha256,
            "allocation_presentation_fingerprint": self.allocation_presentation_fingerprint,
            "reachability_check_ref": self.reachability_check_ref,
            "reachability_check_sha256": self.reachability_check_sha256,
            "reachability_check_fingerprint": self.reachability_check_fingerprint,
            "selected_resource_symbol": self.selected_resource_symbol,
            "selected_resource_iri": self.selected_resource_iri,
            "selected_resource_jid": self.selected_resource_jid,
            "selected_execution_mode": self.selected_execution_mode,
            "allocation_status": self.allocation_status,
            "tbox_fingerprint": self.tbox_fingerprint,
            "registry_fingerprint": self.registry_fingerprint,
            "workcell_fingerprint": self.workcell_fingerprint,
            "fingerprint": self.fingerprint,
        }
        record.update(
            ontology_projection_ref=self.ontology_projection_ref,
            ontology_projection_sha256=self.ontology_projection_sha256,
            validation_scope=("moveit_state_location_reachability"),
            motion_validation_performed=True,
        )
        return record

    def assert_unchanged(self) -> None:
        """Raise if the selection or either pinned decision record changed."""
        persisted = _read_json_mapping(self.record_path, "resource selection record")
        if persisted != self.to_record():
            raise ResourceGroundingError("ResourceSelectionRecord changed after validation.")
        pinned = [
            (self.reachability_check_ref, self.reachability_check_sha256),
            (self.evidence_presentation_ref, self.evidence_presentation_sha256),
            (self.allocation_presentation_ref, self.allocation_presentation_sha256),
        ]
        pinned.append((self.ontology_projection_ref, self.ontology_projection_sha256))
        for record_ref, expected_sha256 in pinned:
            path = _resolve_interaction_ref(self._interaction_root, record_ref)
            if _sha256_path(path) != expected_sha256:
                raise ResourceGroundingError("ResourceSelectionRecord pinned evidence changed.")
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


async def check_live_resource_reachability(  # noqa: PLR0913
    *,
    runtime: RobotAgentFeasibilityRuntime,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
    resource_symbol: str,
    allocation_presentation: AllocationPresentationRecord,
    state_location_record_paths: Mapping[str, Sequence[tuple[str, Path]]],
    check_number: int = 1,
) -> ReachabilityCheckRecord:
    """Evaluate every PA-submitted location independently for one resource."""
    _validate_positive_integer(check_number, "check_number")
    if set(state_location_record_paths) != {"current_state", "desired_state"}:
        raise ResourceGroundingError("Reachability state-location fields are invalid.")
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
    manifest = _load_resource_manifest(entry, _manifest_path(workcell, entry))
    environment = _resource_environment(manifest, entry.resource_symbol)
    profile = _location_planning_profile(environment, context.target_frame)
    state_iris = {
        "current_state": context.current_state_iri,
        "desired_state": context.desired_state_iri,
    }
    state_locations: dict[str, tuple[StateLocationEvidence, ...]] = {}
    for state_name in ("current_state", "desired_state"):
        submitted = state_location_record_paths[state_name]
        if (
            not isinstance(submitted, Sequence)
            or isinstance(submitted, (str, bytes))
            or not submitted
        ):
            raise ResourceGroundingError(
                f"Reachability requires one or more {state_name} locations."
            )
        seen_handles: set[str] = set()
        results: list[StateLocationEvidence] = []
        for evidence_handle, location_path in submitted:
            if evidence_handle in seen_handles:
                raise ResourceGroundingError(
                    f"Reachability {state_name} location handles must be unique."
                )
            seen_handles.add(evidence_handle)
            evidence = allocation_presentation.evidence_for_handle(evidence_handle)
            _assert_presented_evidence_unchanged(root, evidence)
            location = _load_robot_frame_location(
                location_path,
                root,
                target_frame=context.target_frame,
            )
            results.append(
                _state_location_evidence(
                    state_name, state_iris[state_name], location, evidence=evidence, reachable=False
                )
            )
        state_locations[state_name] = tuple(results)

    request = {
        "resource_symbol": entry.resource_symbol,
        "resource_iri": entry.resource_iri,
        "resource_jid": entry.resource_jid,
        "execution_mode": str(manifest["execution_mode"]),
        "process_symbol": context.process_symbol,
        "process_iri": context.process_iri,
        "feature_iri": context.feature_iri,
        "target_frame": context.target_frame,
        **profile,
        "validation_scope": "moveit_state_location_reachability",
        "mode": "plan_only",
        "motion_executed": False,
        "state_locations": _location_request_values(state_locations),
    }
    validate_location_planning_request(request)
    try:
        response = await runtime.validate_plan_only_allocation(request)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        from ...adapters.moveit_plan_only import _locations_unavailable

        response = _locations_unavailable(request, "Live MoveIt validation is unavailable.")
    validate_location_planning_response(request, response)
    state_locations = {
        state: tuple(
            replace(item, reachable=result["status"] == "accepted")
            for item, result in zip(locations, response["state_locations"][state], strict=True)
        )
        for state, locations in state_locations.items()
    }
    status = response["status"]
    validation = {"request": request, "response": response, "validated_at_ns": time.time_ns()}
    execution_mode = str(manifest["execution_mode"])
    payload: dict[str, object] = {
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
        "allocation_presentation_sha256": _sha256_path(allocation_presentation.record_path),
        "allocation_presentation_fingerprint": allocation_presentation.fingerprint,
        "state_locations": {
            state_name: [item.to_record() for item in values]
            for state_name, values in state_locations.items()
        },
        "tbox_fingerprint": tbox.fingerprint,
        "registry_fingerprint": registry.fingerprint,
        "workcell_fingerprint": workcell.fingerprint,
        "status": status,
        "validation": validation,
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
        tbox_fingerprint=tbox.fingerprint,
        registry_fingerprint=registry.fingerprint,
        workcell_fingerprint=workcell.fingerprint,
        status=status,
        fingerprint=str(payload["fingerprint"]),
        _interaction_root=root,
        validation=validation,
        state_locations=state_locations,
    )
    result.assert_unchanged()
    return result


def _validate_pa_grounding_evidence(root: Path, feature_iri: str) -> Mapping[str, object] | None:
    """Gate PA-owned allocation while retaining independent geometry checks."""
    from .grounding_contracts import validate_grounding_evidence

    paths = tuple((root / "products/grounding/ontology_grounding").glob("proposal_*.json"))
    if not paths:
        # Geometry-only callers have no PA claims. Persisted PA runs must keep
        # their proposal and evidence snapshot even when called below orchestration.
        if tuple((root / "interaction_record").glob("context_completion_*.json")):
            raise ResourceGroundingError("PA allocation requires its accepted proposal.")
        return
    proposals = [_read_json_mapping(path, "OntologyGroundingProposal") for path in paths]
    accepted = [
        proposal
        for proposal in proposals
        if proposal.get("status") == "accepted" and proposal.get("feature_iri") == feature_iri
    ]
    if len(accepted) != 1:
        raise ResourceGroundingError("PA allocation requires one accepted proposal.")
    try:
        validate_grounding_evidence(root, accepted[0])
        return accepted[0]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ResourceGroundingError(
            "PA allocation requires unchanged grounding evidence."
        ) from exc


def persist_pa_resource_selection(  # noqa: PLR0913
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
    reachability: ReachabilityCheckRecord,
    allocation_presentation: AllocationPresentationRecord,
    selection_number: int = 1,
    state_location_handles: Mapping[str, tuple[str, ...]] | None = None,
) -> ResourceSelectionRecord:
    """Pin one PA choice to its live reachability evidence."""
    _validate_positive_integer(selection_number, "selection_number")
    root = Path(interaction_root).resolve()
    _assert_authorities(workcell, registry, tbox=tbox)
    abox = load_interaction_abox(root, tbox)
    context = _allocation_context(abox, workcell)
    proposal = _validate_pa_grounding_evidence(root, context.feature_iri)
    reachability.assert_unchanged()
    allocation_presentation.assert_unchanged()
    return _persist_reachable_resource_selection(
        root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        context=context,
        proposal=proposal,
        reachability=reachability,
        presentation=allocation_presentation,
        selection_number=selection_number,
        state_location_handles=state_location_handles,
    )


def _persist_reachable_resource_selection(
    *,
    root: Path,
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
    context: _AllocationContext,
    proposal: Mapping[str, object] | None,
    reachability: ReachabilityCheckRecord,
    presentation: AllocationPresentationRecord,
    selection_number: int,
    state_location_handles: Mapping[str, tuple[str, ...]] | None,
) -> ResourceSelectionRecord:
    """Record the grounded goal and live MoveIt evidence for the exact PA-selected arm."""
    if proposal is None:
        raise ResourceGroundingError("Arm assignment requires the current validated proposal.")
    if (
        (reachability.status != "accepted")
        or (reachability.state_locations is None)
        or (reachability.feature_iri != context.feature_iri)
        or (reachability.process_iri != context.process_iri)
        or (reachability.specification_iri != context.specification_iri)
        or (reachability.allocation_presentation_fingerprint != presentation.fingerprint)
        or (reachability.registry_fingerprint != registry.fingerprint)
        or (reachability.workcell_fingerprint != workcell.fingerprint)
        or (reachability.resource_iri not in context.candidate_resource_iris)
    ):
        raise ResourceGroundingError(
            "Arm assignment requires an accepted capability and reachability check."
        )
    handles = {
        state: tuple(item.evidence_handle for item in locations)
        for state, locations in reachability.state_locations.items()
    }
    if handles != state_location_handles:
        raise ResourceGroundingError("Selected locations differ from the checked locations.")
    target = proposal["output"]["target_feature"]
    for state, state_iri in (
        ("current_state", context.current_state_iri),
        ("desired_state", context.desired_state_iri),
    ):
        expected = {
            (value["value_ref"]["record_ref"], value["value_ref"]["field_path"])
            for value in target[state]["state_values"]
            if _read_json_mapping(root / value["value_ref"]["record_ref"], "state evidence").get(
                "record_type"
            )
            in {"RGBDSegmentationRecord", "RobotFrameLocationRecord"}
        }
        checked = reachability.state_locations[state]
        actual = {(item.source_record_ref, item.source_field_path) for item in checked}
        if (
            not expected
            or expected != actual
            or any(item.state_iri != state_iri or not item.reachable for item in checked)
        ):
            raise ResourceGroundingError(
                "Reachability must cover every bound state location exactly."
            )
    proposal_ref = (
        f"products/grounding/ontology_grounding/proposal_{proposal['proposal_number']:04d}.json"
    )
    destination = root / _SELECTION_ROOT / f"selection_{selection_number:04d}"
    record_path = destination / _SELECTION_RECORD_NAME
    selection = ResourceSelectionRecord(
        record_path=record_path,
        record_ref=record_path.relative_to(root).as_posix(),
        selection_number=selection_number,
        specification_iri=context.specification_iri,
        feature_iri=context.feature_iri,
        process_symbol=context.process_symbol,
        process_iri=context.process_iri,
        current_state_iri=context.current_state_iri,
        desired_state_iri=context.desired_state_iri,
        candidate_resource_iris=context.candidate_resource_iris,
        candidate_resource_symbols=tuple(
            entry.resource_symbol
            for entry in registry.resources
            if entry.resource_iri in context.candidate_resource_iris
        ),
        evidence_presentation_ref=presentation.evidence_presentation_ref,
        evidence_presentation_sha256=_sha256_path(root / presentation.evidence_presentation_ref),
        evidence_presentation_fingerprint=presentation.evidence_presentation_fingerprint,
        allocation_presentation_ref=presentation.record_ref,
        allocation_presentation_sha256=_sha256_path(presentation.record_path),
        allocation_presentation_fingerprint=presentation.fingerprint,
        reachability_check_ref=reachability.record_ref,
        reachability_check_sha256=_sha256_path(reachability.record_path),
        reachability_check_fingerprint=reachability.fingerprint,
        allocation_status="accepted",
        selected_resource_symbol=reachability.resource_symbol,
        selected_resource_iri=reachability.resource_iri,
        selected_resource_jid=reachability.resource_jid,
        selected_execution_mode=reachability.execution_mode,
        tbox_fingerprint=tbox.fingerprint,
        registry_fingerprint=registry.fingerprint,
        workcell_fingerprint=workcell.fingerprint,
        fingerprint="",
        _interaction_root=root,
        state_location_handles=handles,
        ontology_projection_ref=proposal_ref,
        ontology_projection_sha256=_sha256_path(root / proposal_ref),
    )
    payload = selection.to_record()
    payload.pop("fingerprint")
    selection = replace(selection, fingerprint=_record_fingerprint(payload))
    _persist_record(destination, _SELECTION_RECORD_NAME, selection.to_record())
    selection.assert_unchanged()
    return selection


def commit_resource_assignment(
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
    selection: ResourceSelectionRecord,
) -> MergeResult:
    """Commit the PA-selected arm after revalidating its required evidence."""
    root = Path(interaction_root).resolve()
    _assert_authorities(workcell, registry, tbox=tbox)
    abox = load_interaction_abox(root, tbox)
    context = _allocation_context(abox, workcell)
    proposal = _validate_pa_grounding_evidence(root, context.feature_iri)
    selection.assert_unchanged()
    from .grounding_contracts import (
        GroundingContractError,
        _validate_reviewed_location_selection,
        build_product_context_view,
    )

    view = build_product_context_view(
        root, abox, attempted_evidence=(), assessed_at_ns=time.time_ns()
    )
    try:
        _validate_reviewed_location_selection(
            root,
            proposal,
            view,
            selection.to_record(),
            _read_json_mapping(root / selection.reachability_check_ref, "reachability"),
        )
    except GroundingContractError as exc:
        raise ResourceGroundingError("Resource assignment evidence is invalid.") from exc
    state_evidence_valid = (
        selection.state_location_handles is not None
        and set(selection.state_location_handles) == {"current_state", "desired_state"}
        and bool(selection.state_location_handles["current_state"])
        and bool(selection.state_location_handles["desired_state"])
    )
    if (
        (selection.allocation_status != "accepted")
        or (selection.selected_resource_iri is None)
        or (selection.specification_iri != context.specification_iri)
        or (selection.feature_iri != context.feature_iri)
        or (selection.process_symbol != context.process_symbol)
        or (selection.process_iri != context.process_iri)
        or (selection.current_state_iri != context.current_state_iri)
        or (selection.desired_state_iri != context.desired_state_iri)
        or (selection.selected_resource_iri not in context.candidate_resource_iris)
        or (set(selection.candidate_resource_iris) != set(context.candidate_resource_iris))
        or (not state_evidence_valid)
    ):
        raise ResourceGroundingError(
            "Resource assignment requires the accepted PA choice without substitution."
        )
    ppr = Namespace(tbox.ppr_namespace)
    execution_iri = f"{abox.namespace}process_execution_0001"
    evidence_refs = [
        selection.record_ref,
        selection.reachability_check_ref,
        selection.ontology_projection_ref,
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
        if isinstance(feature, URIRef)
        and any(
            isinstance(class_iri, URIRef)
            and workcell._tbox.is_class_or_subclass(class_iri, ppr.feature)
            for class_iri in abox.graph.objects(feature, RDF.type)
        )
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


def _state_location_evidence(
    state_name: str,
    state_iri: str,
    location: _RobotFrameLocationEvidence,
    *,
    evidence: AllocationEvidenceEntry,
    reachable: bool,
) -> StateLocationEvidence:
    return StateLocationEvidence(
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
        reachable=reachable,
    )


def _location_planning_profile(
    environment: Mapping[str, object], target_frame: str
) -> dict[str, object]:
    controller = environment["controller"]
    group = controller["move_group"]
    if group["frame_id"] != target_frame:
        raise ResourceGroundingError("MoveIt frame differs from the calibrated location frame.")
    return {
        "moveit_group": group["group_name"],
        "end_effector_link": group.get("tcp_link") or group["ee_link"],
        "motion_plan_service": controller["services"]["motion_plan"],
        "position_tolerance_m": controller["move_group"]["position_tolerance_m"],
    }


def _location_request_values(
    state_locations: Mapping[str, tuple[StateLocationEvidence, ...]],
) -> dict[str, object]:
    return {
        state: [
            {
                key: item.to_record()[key]
                for key in (
                    "state_iri",
                    "evidence_handle",
                    "translation_m",
                    "location_record_ref",
                    "location_record_sha256",
                )
            }
            for item in locations
        ]
        for state, locations in state_locations.items()
    }


def validate_location_planning_request(request: Mapping[str, object]) -> None:
    """Reject missing identities, invalid positions, or any request to execute motion."""
    expected = {
        "resource_symbol",
        "resource_iri",
        "resource_jid",
        "execution_mode",
        "process_symbol",
        "process_iri",
        "feature_iri",
        "target_frame",
        "moveit_group",
        "end_effector_link",
        "motion_plan_service",
        "position_tolerance_m",
        "validation_scope",
        "mode",
        "motion_executed",
        "state_locations",
    }
    if (
        set(request) != expected
        or request["execution_mode"] != "simulation"
        or request["mode"] != "plan_only"
        or request["motion_executed"] is not False
        or request["validation_scope"] != "moveit_state_location_reachability"
    ):
        raise ResourceGroundingError("Live location planning request is invalid.")
    for key in expected - {"position_tolerance_m", "motion_executed", "state_locations"}:
        if not isinstance(request[key], str) or not request[key]:
            raise ResourceGroundingError("Live planning identity is invalid.")
    tolerance = _finite_number(request["position_tolerance_m"], "position_tolerance_m")
    if not 0 < tolerance <= 0.01:
        raise ResourceGroundingError("Live planning position tolerance is invalid.")
    groups = request["state_locations"]
    if not isinstance(groups, Mapping) or set(groups) != {"current_state", "desired_state"}:
        raise ResourceGroundingError("Live planning must cover both bound states.")
    for locations in groups.values():
        if not isinstance(locations, list) or not locations:
            raise ResourceGroundingError("Live planning state locations are missing.")
        for item in locations:
            if not isinstance(item, Mapping) or set(item) != {
                "state_iri",
                "evidence_handle",
                "translation_m",
                "location_record_ref",
                "location_record_sha256",
            }:
                raise ResourceGroundingError("Live planning location binding is invalid.")
            for key in set(item) - {"translation_m"}:
                if not isinstance(item[key], str) or not item[key]:
                    raise ResourceGroundingError("Live planning location identity is invalid.")
            coordinates = item["translation_m"]
            if not isinstance(coordinates, list) or len(coordinates) != 3:
                raise ResourceGroundingError("Live planning coordinates are invalid.")
            for coordinate in coordinates:
                _finite_number(coordinate, "coordinate")


def validate_location_planning_response(
    request: Mapping[str, object], response: Mapping[str, object]
) -> None:
    """Require complete exact-location coverage and concrete successful planning evidence."""
    if not isinstance(response, Mapping) or set(response) != {
        "status",
        "state_locations",
        "feedback",
    }:
        raise ResourceGroundingError("Live planning response is invalid.")
    groups = response["state_locations"]
    if not isinstance(groups, Mapping) or set(groups) != set(request["state_locations"]):
        raise ResourceGroundingError("Live planning response coverage is invalid.")
    statuses = []
    for state, locations in request["state_locations"].items():
        results = groups[state]
        if not isinstance(results, list) or len(results) != len(locations):
            raise ResourceGroundingError("Live planning omitted a bound location.")
        for location, result in zip(locations, results, strict=True):
            if (
                not isinstance(result, Mapping)
                or set(result) != {"evidence_handle", "status", "message", "error_code", "plan"}
                or result["evidence_handle"] != location["evidence_handle"]
                or result["status"] not in _VALIDATION_STATUSES
            ):
                raise ResourceGroundingError(
                    "Live planning substituted a location or returned an invalid verdict."
                )
            statuses.append(result["status"])
            if result["status"] != "accepted":
                if result["plan"] is not None or result["error_code"] == 1:
                    raise ResourceGroundingError("Rejected planning evidence is inconsistent.")
                continue
            plan = result["plan"]
            if (
                type(result["error_code"]) is not int
                or result["error_code"] != 1
                or not isinstance(plan, Mapping)
                or set(plan)
                != {"joint_names", "points", "start_joint_names", "start_joint_positions"}
            ):
                raise ResourceGroundingError("Accepted reachability has no MoveIt plan.")
            for names in (plan["joint_names"], plan["start_joint_names"]):
                if (
                    not isinstance(names, list)
                    or not names
                    or any(not isinstance(name, str) or not name for name in names)
                    or len(set(names)) != len(names)
                ):
                    raise ResourceGroundingError("MoveIt joint identities are invalid.")
            if (
                not set(plan["joint_names"]).issubset(plan["start_joint_names"])
                or not isinstance(plan["points"], list)
                or not plan["points"]
            ):
                raise ResourceGroundingError("MoveIt trajectory or live start state is missing.")
            for names, points in (
                (plan["joint_names"], plan["points"]),
                (plan["start_joint_names"], [plan["start_joint_positions"]]),
            ):
                for point in points:
                    if not isinstance(point, list) or len(point) != len(names):
                        raise ResourceGroundingError("MoveIt trajectory dimensions are invalid.")
                    for coordinate in point:
                        _finite_number(coordinate, "joint position")
    expected = (
        "needs_context"
        if "needs_context" in statuses
        else "rejected"
        if "rejected" in statuses
        else "accepted"
    )
    if response["status"] != expected:
        raise ResourceGroundingError("MoveIt aggregate verdict differs from its location results.")


def _assert_presented_evidence_unchanged(
    interaction_root: Path,
    evidence: AllocationEvidenceEntry,
) -> None:
    path = _resolve_interaction_ref(interaction_root, evidence.record_ref)
    if _sha256_path(path) != evidence.record_sha256:
        raise ResourceGroundingError("PA-selected state evidence changed before reachability.")


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
    if not isinstance(record, Mapping) or "schema_version" in record:
        raise RobotFrameLocationEvidenceError(
            "RobotFrameLocationRecord fields are incompatible. Start a fresh interaction."
        )
    if (
        (record.get("record_type") != "RobotFrameLocationRecord")
        or (record.get("target_frame") != target_frame)
        or (record.get("robot_frame_conversion") != "accepted")
        or (record.get("location") != "available")
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
    frames: set[str] = set()
    for resource_iri in candidate_resource_iris:
        entry = entries[resource_iri]
        resource = _load_resource_manifest(
            entry,
            _manifest_path(workcell, entry),
        )
        environment = _resource_environment(resource, entry.resource_symbol)
        if resource.get("execution_mode") == "simulation":
            controller = environment.get("controller")
            move_group = controller.get("move_group") if isinstance(controller, Mapping) else None
            frame = move_group.get("frame_id") if isinstance(move_group, Mapping) else None
        else:
            static = environment.get("static_capabilities")
            reach = static.get("gripper_reach") if isinstance(static, Mapping) else None
            frame = reach.get("frame") if isinstance(reach, Mapping) else None
        if not isinstance(frame, str) or not frame:
            raise ResourceGroundingError(
                f"Resource target frame is unavailable: {entry.resource_symbol}."
            )
        frames.add(frame)
    if len(frames) != 1 or not next(iter(frames), ""):
        raise ResourceGroundingError("Capable resources declare inconsistent target frames.")
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

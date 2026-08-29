"""Ground one semantic assembly process to a predefined robot resource.

The PA supplies only product semantics.  This module owns the deterministic
join against the predefined Workcell ABox, validates one accepted world-frame
pose against the manifest-pinned coarse reach envelopes, and delegates the
exact process-execution write to the host-only ABox boundary.
"""

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

_RESOURCE_SELECTION_ROOT = Path("products/grounding/resource_selection")
_SELECTION_RECORD_NAME = "resource_selection_record.json"
_REQUIRED_RECORD_TYPE = "RobotFramePoseRecord"
_TARGET_FRAME = "world"
_SELECTION_POLICY = "first_reachable_in_predefined_registry_order"
_HOST_PRODUCER = "resource_grounding_host"
_EXPECTED_RESOURCE_SYMBOLS = ("xarm6", "ur5e")
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MANIFEST_PATHS = {
    "xarm6": _PACKAGE_ROOT / "initialization/resources/robot_xarm6.json",
    "ur5e": _PACKAGE_ROOT / "initialization/resources/robot_ur5e.json",
}
_EXECUTION_ENVIRONMENT = {
    "simulation": "gazebo",
    "physical": "real",
}


class ResourceGroundingError(OntologyContextError):
    """Raised when semantic resource grounding cannot be trusted."""


class RobotFramePoseEvidenceError(ResourceGroundingError):
    """Raised when a RobotFramePoseRecord cannot support resource selection."""


@dataclass(frozen=True)
class ResourceAssignmentNeed:
    """Describe the transient typed evidence needed for resource assignment."""

    specification_iri: str
    feature_iri: str
    process_iri: str
    candidate_resource_iris: tuple[str, ...]
    required_record_type: str
    target_frame: str
    tbox_fingerprint: str
    workcell_fingerprint: str


@dataclass(frozen=True)
class CandidateReachEvidence:
    """Record one manifest-backed coarse reach verdict without configuration."""

    resource_symbol: str
    resource_iri: str
    resource_jid: str
    manifest_ref: str
    manifest_sha256: str
    execution_mode: str
    reachable: bool
    verdicts: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        """Return the JSON-safe candidate verdict."""
        return {
            "resource_symbol": self.resource_symbol,
            "resource_iri": self.resource_iri,
            "resource_jid": self.resource_jid,
            "manifest_ref": self.manifest_ref,
            "manifest_sha256": self.manifest_sha256,
            "execution_mode": self.execution_mode,
            "reachable": self.reachable,
            "verdicts": list(self.verdicts),
        }


@dataclass(frozen=True)
class ResourceSelectionRecord:
    """Hold one persisted deterministic resource-selection decision."""

    record_path: Path = field(repr=False, compare=False)
    record_ref: str
    selection_number: int
    specification_iri: str
    feature_iri: str
    process_iri: str
    candidate_resource_iris: tuple[str, ...]
    required_record_type: str
    target_frame: str
    pose_record_ref: str
    pose_record_sha256: str
    observation_timestamp_ns: int
    tbox_fingerprint: str
    registry_fingerprint: str
    workcell_fingerprint: str
    selection_policy: str
    candidate_reach_evidence: tuple[CandidateReachEvidence, ...]
    selected_resource_symbol: str | None
    selected_resource_iri: str | None
    selected_resource_jid: str | None
    selected_execution_mode: str | None
    fingerprint: str
    _interaction_root: Path = field(repr=False, compare=False)
    _pose_path: Path = field(repr=False, compare=False)
    _manifest_paths: tuple[tuple[str, Path], ...] = field(repr=False, compare=False)
    _registry: ResourceRegistrySnapshot = field(repr=False, compare=False)
    _workcell: PredefinedWorkcellSnapshot = field(repr=False, compare=False)

    def to_record(self) -> dict[str, object]:
        """Return the exact JSON-safe persisted selection record."""
        return {
            "schema_version": 1,
            "record_type": "ResourceSelectionRecord",
            "selection_number": self.selection_number,
            "specification_iri": self.specification_iri,
            "feature_iri": self.feature_iri,
            "process_iri": self.process_iri,
            "candidate_resource_iris": list(self.candidate_resource_iris),
            "required_record_type": self.required_record_type,
            "target_frame": self.target_frame,
            "pose_record_ref": self.pose_record_ref,
            "pose_record_sha256": self.pose_record_sha256,
            "observation_timestamp_ns": self.observation_timestamp_ns,
            "tbox_fingerprint": self.tbox_fingerprint,
            "registry_fingerprint": self.registry_fingerprint,
            "workcell_fingerprint": self.workcell_fingerprint,
            "selection_policy": self.selection_policy,
            "candidate_reach_evidence": [
                item.to_record() for item in self.candidate_reach_evidence
            ],
            "selected_resource_symbol": self.selected_resource_symbol,
            "selected_resource_iri": self.selected_resource_iri,
            "selected_resource_jid": self.selected_resource_jid,
            "selected_execution_mode": self.selected_execution_mode,
            "fingerprint": self.fingerprint,
        }

    def assert_unchanged(self) -> None:
        """Raise if the decision record or any pinned selection input changed."""
        _assert_authorities(self._workcell, self._registry)
        pose = _load_world_pose(self._pose_path, self._interaction_root)
        if pose.record_ref != self.pose_record_ref or pose.sha256 != self.pose_record_sha256:
            raise ResourceGroundingError(
                "Resource selection world-frame pose changed after validation."
            )
        if pose.observation_timestamp_ns != self.observation_timestamp_ns:
            raise ResourceGroundingError(
                "Resource selection observation timestamp changed after validation."
            )

        manifest_paths = dict(self._manifest_paths)
        entries = {entry.resource_iri: entry for entry in self._registry.resources}
        expected_evidence = tuple(
            _candidate_reach_evidence(
                entries[resource_iri],
                manifest_paths[entries[resource_iri].resource_symbol],
                pose.translation_m,
            )
            for resource_iri in self.candidate_resource_iris
        )
        if expected_evidence != self.candidate_reach_evidence:
            raise ResourceGroundingError(
                "Resource selection reach verdict changed after validation."
            )
        expected_selected = next(
            (candidate for candidate in expected_evidence if candidate.reachable),
            None,
        )
        if _selected_fields(expected_selected) != (
            self.selected_resource_symbol,
            self.selected_resource_iri,
            self.selected_resource_jid,
            self.selected_execution_mode,
        ):
            raise ResourceGroundingError(
                "Resource selection decision changed after validation."
            )

        persisted = _read_json_mapping(self.record_path, "Resource selection record")
        if persisted != self.to_record():
            raise ResourceGroundingError(
                "Resource selection record changed after validation."
            )
        payload = self.to_record()
        payload.pop("fingerprint")
        if _record_fingerprint(payload) != self.fingerprint:
            raise ResourceGroundingError(
                "Resource selection fingerprint changed after validation."
            )


@dataclass(frozen=True)
class _WorldPoseEvidence:
    path: Path
    record_ref: str
    sha256: str
    observation_timestamp_ns: int
    translation_m: tuple[float, float, float]


def derive_resource_assignment_need(
    abox: ABoxSnapshot,
    workcell: PredefinedWorkcellSnapshot,
) -> ResourceAssignmentNeed | None:
    """Derive a world-pose need only from an unresolved semantic Workcell join.

    The result is intentionally transient.  Provider routing consumes its
    required record type and frame; it is not another task-transition record.
    """
    if not isinstance(abox, ABoxSnapshot):
        raise ResourceGroundingError(
            "Resource assignment need requires an ABoxSnapshot."
        )
    if not isinstance(workcell, PredefinedWorkcellSnapshot):
        raise ResourceGroundingError(
            "Resource assignment need requires a PredefinedWorkcellSnapshot."
        )
    try:
        workcell.assert_unchanged()
    except OntologyContextError as exc:
        raise ResourceGroundingError(
            "Resource assignment Workcell snapshot is not immutable."
        ) from exc
    if abox.tbox_fingerprint != workcell.tbox_fingerprint:
        raise ResourceGroundingError(
            "Resource assignment ABox and Workcell TBox fingerprints differ."
        )

    ppr = Namespace(workcell.ppr_namespace)
    specification = URIRef(abox.specification_iri)
    if any(abox.graph.objects(specification, ppr.hasProcessExecution)):
        return None

    process = URIRef(workcell.process_iri)
    joined_features = {
        feature
        for feature in abox.graph.objects(specification, ppr.defines)
        if isinstance(feature, URIRef)
        and (feature, RDF.type, ppr.feature) in abox.graph
        and (process, ppr.realizes, feature) in abox.graph
    }
    if not joined_features:
        return None
    if len(joined_features) != 1:
        raise ResourceGroundingError(
            "Resource assignment semantic join must identify exactly one feature."
        )

    candidates = tuple(
        resource_iri
        for resource_iri in workcell.resource_iris
        if (
            URIRef(resource_iri),
            ppr.capableOf,
            process,
        )
        in workcell.graph
        and (URIRef(resource_iri), RDF.type, ppr.resource) in workcell.graph
    )
    if not candidates:
        return None
    return ResourceAssignmentNeed(
        specification_iri=abox.specification_iri,
        feature_iri=str(next(iter(joined_features))),
        process_iri=workcell.process_iri,
        candidate_resource_iris=candidates,
        required_record_type=_REQUIRED_RECORD_TYPE,
        target_frame=_TARGET_FRAME,
        tbox_fingerprint=abox.tbox_fingerprint,
        workcell_fingerprint=workcell.fingerprint,
    )


def select_predefined_resource(  # noqa: PLR0913
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
    need: ResourceAssignmentNeed,
    robot_frame_pose_path: Path,
    manifest_paths: Mapping[str, Path] | None = None,
    selection_number: int = 1,
) -> ResourceSelectionRecord:
    """Select the first coarsely reachable predefined resource and persist it."""
    root = Path(interaction_root).resolve()
    _validate_selection_number(selection_number)
    _assert_authorities(workcell, registry, tbox=tbox)
    abox = load_interaction_abox(root, tbox)
    derived_need = derive_resource_assignment_need(abox, workcell)
    if derived_need is None or derived_need != need:
        raise ResourceGroundingError(
            "Resource assignment need is not the current derived semantic need."
        )
    _validate_need(need, tbox, registry, workcell)

    pose = _load_world_pose(Path(robot_frame_pose_path), root)
    configured_paths = _validated_manifest_paths(manifest_paths)
    registry_entries = {entry.resource_iri: entry for entry in registry.resources}
    evidence = tuple(
        _candidate_reach_evidence(
            registry_entries[resource_iri],
            configured_paths[registry_entries[resource_iri].resource_symbol],
            pose.translation_m,
        )
        for resource_iri in need.candidate_resource_iris
    )
    selected = next((candidate for candidate in evidence if candidate.reachable), None)
    selected_symbol, selected_iri, selected_jid, selected_mode = _selected_fields(
        selected
    )

    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "ResourceSelectionRecord",
        "selection_number": selection_number,
        "specification_iri": need.specification_iri,
        "feature_iri": need.feature_iri,
        "process_iri": need.process_iri,
        "candidate_resource_iris": list(need.candidate_resource_iris),
        "required_record_type": need.required_record_type,
        "target_frame": need.target_frame,
        "pose_record_ref": pose.record_ref,
        "pose_record_sha256": pose.sha256,
        "observation_timestamp_ns": pose.observation_timestamp_ns,
        "tbox_fingerprint": tbox.fingerprint,
        "registry_fingerprint": registry.fingerprint,
        "workcell_fingerprint": workcell.fingerprint,
        "selection_policy": _SELECTION_POLICY,
        "candidate_reach_evidence": [item.to_record() for item in evidence],
        "selected_resource_symbol": selected_symbol,
        "selected_resource_iri": selected_iri,
        "selected_resource_jid": selected_jid,
        "selected_execution_mode": selected_mode,
    }
    payload["fingerprint"] = _record_fingerprint(payload)
    destination = (
        root
        / _RESOURCE_SELECTION_ROOT
        / f"selection_{selection_number:04d}"
    )
    record_path = destination / _SELECTION_RECORD_NAME
    _persist_selection_record(destination, payload)
    result = ResourceSelectionRecord(
        record_path=record_path,
        record_ref=record_path.relative_to(root).as_posix(),
        selection_number=selection_number,
        specification_iri=need.specification_iri,
        feature_iri=need.feature_iri,
        process_iri=need.process_iri,
        candidate_resource_iris=need.candidate_resource_iris,
        required_record_type=need.required_record_type,
        target_frame=need.target_frame,
        pose_record_ref=pose.record_ref,
        pose_record_sha256=pose.sha256,
        observation_timestamp_ns=pose.observation_timestamp_ns,
        tbox_fingerprint=tbox.fingerprint,
        registry_fingerprint=registry.fingerprint,
        workcell_fingerprint=workcell.fingerprint,
        selection_policy=_SELECTION_POLICY,
        candidate_reach_evidence=evidence,
        selected_resource_symbol=selected_symbol,
        selected_resource_iri=selected_iri,
        selected_resource_jid=selected_jid,
        selected_execution_mode=selected_mode,
        fingerprint=str(payload["fingerprint"]),
        _interaction_root=root,
        _pose_path=pose.path,
        _manifest_paths=tuple(configured_paths.items()),
        _registry=registry,
        _workcell=workcell,
    )
    result.assert_unchanged()
    return result


def commit_resource_assignment(
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
    need: ResourceAssignmentNeed,
    selection: ResourceSelectionRecord,
) -> MergeResult:
    """Commit the exact host-derived process execution for one selection."""
    root = Path(interaction_root).resolve()
    _assert_authorities(workcell, registry, tbox=tbox)
    abox = load_interaction_abox(root, tbox)
    if derive_resource_assignment_need(abox, workcell) != need:
        raise ResourceGroundingError(
            "Resource assignment need is no longer current."
        )
    selection.assert_unchanged()
    _validate_selection_matches_need(
        selection,
        need,
        tbox,
        registry,
        workcell,
        interaction_root=root,
    )
    if selection.selected_resource_iri is None:
        raise ResourceGroundingError(
            "No predefined resource is coarsely reachable; assignment was not committed."
        )

    ppr = Namespace(tbox.ppr_namespace)
    execution_iri = f"{abox.namespace}process_execution_0001"
    evidence_refs = [selection.record_ref]
    assertions = [
        _iri_assertion(
            need.specification_iri,
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
            need.process_iri,
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
        assertions=assertions,
        authorized_evidence_refs=evidence_refs,
    )


def _assert_authorities(
    workcell: PredefinedWorkcellSnapshot,
    registry: ResourceRegistrySnapshot,
    *,
    tbox: TBoxSnapshot | None = None,
) -> None:
    if not isinstance(registry, ResourceRegistrySnapshot):
        raise ResourceGroundingError(
            "Resource grounding requires a ResourceRegistrySnapshot."
        )
    if not isinstance(workcell, PredefinedWorkcellSnapshot):
        raise ResourceGroundingError(
            "Resource grounding requires a PredefinedWorkcellSnapshot."
        )
    if tbox is not None and not isinstance(tbox, TBoxSnapshot):
        raise ResourceGroundingError("Resource grounding requires a TBoxSnapshot.")
    try:
        registry.assert_unchanged()
        workcell.assert_unchanged()
        if tbox is not None:
            tbox.assert_unchanged()
    except OntologyContextError as exc:
        raise ResourceGroundingError(
            "Resource-grounding ontology authority changed after validation."
        ) from exc
    if (
        workcell.registry_fingerprint != registry.fingerprint
        or workcell.tbox_fingerprint != registry.tbox_fingerprint
        or (tbox is not None and workcell.tbox_fingerprint != tbox.fingerprint)
    ):
        raise ResourceGroundingError(
            "Resource-grounding ontology authorities do not share fingerprints."
        )


def _validate_need(
    need: ResourceAssignmentNeed,
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
) -> None:
    if not isinstance(need, ResourceAssignmentNeed):
        raise ResourceGroundingError(
            "Resource selection requires a ResourceAssignmentNeed."
        )
    if (
        need.process_iri != workcell.process_iri
        or need.candidate_resource_iris != workcell.resource_iris
        or need.required_record_type != _REQUIRED_RECORD_TYPE
        or need.target_frame != _TARGET_FRAME
        or need.tbox_fingerprint != tbox.fingerprint
        or need.workcell_fingerprint != workcell.fingerprint
        or tuple(entry.resource_iri for entry in registry.resources)
        != need.candidate_resource_iris
    ):
        raise ResourceGroundingError(
            "Resource assignment need does not match the predefined Workcell."
        )


def _validate_selection_matches_need(
    selection: ResourceSelectionRecord,
    need: ResourceAssignmentNeed,
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
    *,
    interaction_root: Path,
) -> None:
    if not isinstance(selection, ResourceSelectionRecord):
        raise ResourceGroundingError(
            "Resource assignment requires a ResourceSelectionRecord."
        )
    if (
        selection.specification_iri != need.specification_iri
        or selection.feature_iri != need.feature_iri
        or selection.process_iri != need.process_iri
        or selection.candidate_resource_iris != need.candidate_resource_iris
        or selection.required_record_type != need.required_record_type
        or selection.target_frame != need.target_frame
        or selection.tbox_fingerprint != tbox.fingerprint
        or selection.registry_fingerprint != registry.fingerprint
        or selection.workcell_fingerprint != workcell.fingerprint
        or selection.selection_policy != _SELECTION_POLICY
        or selection._interaction_root != interaction_root
        or selection.record_path
        != interaction_root
        / _RESOURCE_SELECTION_ROOT
        / f"selection_{selection.selection_number:04d}"
        / _SELECTION_RECORD_NAME
    ):
        raise ResourceGroundingError(
            "Resource selection does not match the current assignment need."
        )


def _load_world_pose(path: Path, interaction_root: Path) -> _WorldPoseEvidence:
    root = Path(interaction_root).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
        source = resolved.read_bytes()
        record = json.loads(source.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RobotFramePoseEvidenceError(
            "RobotFramePoseRecord could not be read from this interaction."
        ) from exc
    if not isinstance(record, Mapping):
        raise RobotFramePoseEvidenceError(
            "RobotFramePoseRecord must be a JSON object."
        )
    if (
        record.get("schema_version") != 1
        or record.get("record_type") != _REQUIRED_RECORD_TYPE
        or record.get("target_frame") != _TARGET_FRAME
    ):
        raise RobotFramePoseEvidenceError(
            "Resource selection requires an exact world-frame RobotFramePoseRecord."
        )
    status = record.get("robot_frame_conversion", record.get("status"))
    if status != "accepted":
        raise RobotFramePoseEvidenceError(
            "Resource selection requires an accepted RobotFramePoseRecord."
        )
    for key, accepted in (
        ("CAD_correspondence", "accepted"),
        ("location", "available"),
        ("pose", "accepted"),
    ):
        if key in record and record[key] != accepted:
            raise RobotFramePoseEvidenceError(
                f"RobotFramePoseRecord {key} is not accepted."
            )
    observation_timestamp_ns = record.get("observation_timestamp_ns")
    if (
        isinstance(observation_timestamp_ns, bool)
        or not isinstance(observation_timestamp_ns, int)
        or observation_timestamp_ns < 0
    ):
        raise RobotFramePoseEvidenceError(
            "RobotFramePoseRecord observation timestamp is invalid."
        )
    robot_pose = record.get("robot_frame_pose")
    if not isinstance(robot_pose, Mapping):
        raise RobotFramePoseEvidenceError(
            "Accepted RobotFramePoseRecord has no robot_frame_pose."
        )
    try:
        translation = _finite_vector3(
            robot_pose.get("CAD_origin_translation_m"),
            "RobotFramePoseRecord CAD_origin_translation_m",
        )
        validated_hash_refs = _validate_embedded_hash_refs(record, root)
    except RobotFramePoseEvidenceError:
        raise
    except ResourceGroundingError as exc:
        raise RobotFramePoseEvidenceError(str(exc)) from exc
    if validated_hash_refs < 1:
        # A pose without an intact source chain cannot be distinguished from an
        # ungrounded coordinate supplied directly in the terminal record.
        raise RobotFramePoseEvidenceError(
            "RobotFramePoseRecord requires at least one embedded hash reference."
        )
    return _WorldPoseEvidence(
        path=resolved,
        record_ref=relative.as_posix(),
        sha256=hashlib.sha256(source).hexdigest(),
        observation_timestamp_ns=observation_timestamp_ns,
        translation_m=translation,
    )


def _validated_manifest_paths(
    manifest_paths: Mapping[str, Path] | None,
) -> dict[str, Path]:
    values = dict(_DEFAULT_MANIFEST_PATHS if manifest_paths is None else manifest_paths)
    if set(values) != set(_EXPECTED_RESOURCE_SYMBOLS):
        raise ResourceGroundingError(
            "Resource selection requires exact xarm6 then ur5e manifest paths."
        )
    return {symbol: Path(values[symbol]).resolve() for symbol in _EXPECTED_RESOURCE_SYMBOLS}


def _candidate_reach_evidence(
    entry: ResourceRegistryEntry,
    manifest_path: Path,
    translation_m: tuple[float, float, float],
) -> CandidateReachEvidence:
    source = _read_bytes(manifest_path, f"{entry.resource_symbol} manifest")
    digest = hashlib.sha256(source).hexdigest()
    if digest != entry.source_sha256:
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
        raise ResourceGroundingError(
            f"Resource manifest symbol changed: {entry.resource_symbol}."
        )
    resource = payload[entry.resource_symbol]
    if not isinstance(resource, Mapping):
        raise ResourceGroundingError(
            f"Resource manifest entry is invalid: {entry.resource_symbol}."
        )
    if resource.get("type") != "robot" or resource.get("jid") != entry.resource_jid:
        raise ResourceGroundingError(
            f"Resource manifest identity changed: {entry.resource_symbol}."
        )
    execution_mode = resource.get("execution_mode")
    if execution_mode not in _EXECUTION_ENVIRONMENT:
        raise ResourceGroundingError(
            f"Resource execution_mode is unsupported: {entry.resource_symbol}."
        )
    environment = resource.get(_EXECUTION_ENVIRONMENT[str(execution_mode)])
    if not isinstance(environment, Mapping):
        raise ResourceGroundingError(
            f"Resource execution environment is missing: {entry.resource_symbol}."
        )
    static_capabilities = environment.get("static_capabilities")
    if not isinstance(static_capabilities, Mapping):
        raise ResourceGroundingError(
            f"Resource static_capabilities are missing: {entry.resource_symbol}."
        )

    supports_pick_place = static_capabilities.get("supports_manipulator_pick_place")
    workspace = _workspace_bounds(static_capabilities.get("workspace_bounds"))
    reach = _gripper_reach(static_capabilities.get("gripper_reach"))
    x, y, z = translation_m
    tolerance = reach[5]
    in_workspace = (
        workspace[0] - tolerance <= x <= workspace[1] + tolerance
        and workspace[2] - tolerance <= y <= workspace[3] + tolerance
        and workspace[4] - tolerance <= z <= workspace[5] + tolerance
    )
    radial_distance = math.hypot(x - reach[0], y - reach[1])
    in_gripper_reach = (
        radial_distance <= reach[2] + tolerance
        and reach[3] - tolerance <= z <= reach[4] + tolerance
    )
    verdicts: list[str] = []
    if supports_pick_place is not True:
        verdicts.append("supports_manipulator_pick_place_false")
    if not in_workspace:
        verdicts.append("outside_workspace_bounds")
    if not in_gripper_reach:
        verdicts.append("outside_gripper_reach")
    reachable = not verdicts
    if reachable:
        verdicts.append("coarsely_reachable")
    return CandidateReachEvidence(
        resource_symbol=entry.resource_symbol,
        resource_iri=entry.resource_iri,
        resource_jid=entry.resource_jid,
        manifest_ref=entry.source_ref,
        manifest_sha256=entry.source_sha256,
        execution_mode=str(execution_mode),
        reachable=reachable,
        verdicts=tuple(verdicts),
    )


def _workspace_bounds(value: object) -> tuple[float, float, float, float, float, float]:
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
    return bounds


def _gripper_reach(
    value: object,
) -> tuple[float, float, float, float, float, float]:
    if not isinstance(value, Mapping) or value.get("frame") != _TARGET_FRAME:
        raise ResourceGroundingError("Resource gripper_reach frame must be world.")
    origin = value.get("origin_pose")
    if not isinstance(origin, Mapping):
        raise ResourceGroundingError("Resource gripper_reach origin_pose is invalid.")
    result = (
        _finite_number(origin.get("x"), "gripper_reach.origin_pose.x"),
        _finite_number(origin.get("y"), "gripper_reach.origin_pose.y"),
        _finite_number(value.get("max_xy_radius_m"), "gripper_reach.max_xy_radius_m"),
        _finite_number(value.get("z_min_m"), "gripper_reach.z_min_m"),
        _finite_number(value.get("z_max_m"), "gripper_reach.z_max_m"),
        _finite_number(value.get("tolerance_m"), "gripper_reach.tolerance_m"),
    )
    if result[2] < 0 or result[3] > result[4] or result[5] < 0:
        raise ResourceGroundingError("Resource gripper_reach is inconsistent.")
    return result


def _selected_fields(
    selected: CandidateReachEvidence | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    if selected is None:
        return None, None, None, None
    return (
        selected.resource_symbol,
        selected.resource_iri,
        selected.resource_jid,
        selected.execution_mode,
    )


def _validate_embedded_hash_refs(value: object, interaction_root: Path) -> int:
    validated = 0
    if isinstance(value, Mapping):
        ref = value.get("ref")
        sha256 = value.get("sha256")
        if "ref" in value or "sha256" in value:
            if (
                not isinstance(ref, str)
                or not ref
                or not isinstance(sha256, str)
                or not sha256
            ):
                raise RobotFramePoseEvidenceError(
                    "RobotFramePoseRecord embedded hash reference is incomplete."
                )
            relative = Path(ref)
            if relative.is_absolute() or ".." in relative.parts:
                raise RobotFramePoseEvidenceError(
                    "RobotFramePoseRecord embedded reference leaves its interaction."
                )
            source_path = (interaction_root / relative).resolve()
            try:
                source_path.relative_to(interaction_root)
            except ValueError as exc:
                raise RobotFramePoseEvidenceError(
                    "RobotFramePoseRecord embedded reference leaves its interaction."
                ) from exc
            if hashlib.sha256(_read_bytes(source_path, ref)).hexdigest() != sha256:
                raise RobotFramePoseEvidenceError(
                    f"RobotFramePoseRecord embedded hash does not match: {ref}."
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


def _persist_selection_record(
    destination: Path,
    payload: Mapping[str, object],
) -> None:
    if destination.exists():
        raise ResourceGroundingError(
            f"Resource selection already exists: {destination.name}."
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=".resource-selection-", dir=destination.parent)
        )
    except OSError as exc:
        raise ResourceGroundingError(
            "Resource selection staging directory could not be created."
        ) from exc
    try:
        serialized = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        (temporary / _SELECTION_RECORD_NAME).write_text(
            serialized + "\n", encoding="utf-8"
        )
        if destination.exists():
            raise ResourceGroundingError(
                f"Resource selection already exists: {destination.name}."
            )
        temporary.rename(destination)
    except ResourceGroundingError:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    except (OSError, TypeError, ValueError) as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise ResourceGroundingError("Resource selection persistence failed.") from exc


def _validate_selection_number(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ResourceGroundingError("selection_number must be a positive integer.")


def _finite_vector3(value: object, label: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ResourceGroundingError(f"{label} must contain exactly three values.")
    return tuple(_finite_number(item, label) for item in value)  # type: ignore[return-value]


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


def _record_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

"""Run generalized evidence-first PA grounding through authorized providers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingActionAttempt,
    GroundingContractError,
    GroundingNextAction,
    GroundingProducerDescriptor,
    GroundingSession,
    ProductContextView,
    TypedContextBinding,
    build_product_context_view,
    load_latest_grounding_session,
    persist_grounding_session,
    persist_product_context_view,
)
from cais_spade_llm.spec2primitives.agents.pa.ontology_grounding import (
    OntologyGroundingError,
    propose_and_validate_ontology_grounding,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    ABoxSnapshot,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
    ResourceAssignmentNeed,
    ResourceGroundingError,
    RobotFramePoseEvidenceError,
    commit_resource_assignment,
    derive_resource_assignment_need,
    select_predefined_resource,
)
from cais_spade_llm.spec2primitives.config import DocumentVLMConfig
from cais_spade_llm.spec2primitives.ontology import (
    TBoxSnapshot,
    load_predefined_resource_registry,
    load_predefined_workcell,
)
from cais_spade_llm.spec2primitives.tools.document_evidence import (
    DocumentEvidenceRecord,
    DocumentInterpretationError,
    DocumentVisionRuntime,
    document_overview_cache_status,
    inspect_document_evidence,
    interpret_document_evidence,
)
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_cad_path,
    approved_context_ref_evidence_types,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    CameraToRobotCalibrationResult,
    RobotFrameConversionError,
    associate_segmented_candidate_by_size,
    estimate_camera_frame_pose,
    preprocess_served_geometry,
    segment_preprocessed_observation,
    transform_camera_pose_to_robot_frame,
)

_DOCUMENT_PRODUCER = "document_evidence"
_GEOMETRY_PRODUCER = "rgb_d_cad_grounding"
_CALIBRATION_PRODUCER = "camera_to_world_calibration"
_FRAME_CONVERSION_PRODUCER = "camera_pose_to_world"
_WORLD_FRAME = "world"
_TYPED_OUTPUTS = (
    "DocumentOverviewRecord",
    "DocumentEvidenceRecord",
    "CADMeshRecord",
    "ColoredPointCloudSetRecord",
    "RGBDSegmentationRecord",
    "CADSizeCorrespondenceRecord",
    "CADPoseEstimationRecord",
    "PAClarification",
)


@dataclass(frozen=True)
class _EligibleProviderAction:
    """Describe one exact provider action available to the current session."""

    action: str
    provider_id: str
    source_ref: str
    source_revision: str
    evidence_type: str
    produced_record_types: tuple[str, ...]
    description: str
    estimated_cost: int
    automatic: bool = False
    prerequisite_bindings: tuple[tuple[str, TypedContextBinding], ...] = ()

    def to_record(self) -> dict[str, object]:
        """Return the ontology-neutral action record exposed to PA."""
        return {
            "action": self.action,
            "provider_id": self.provider_id,
            "source_ref": self.source_ref,
            "source_revision": self.source_revision,
            "evidence_type": self.evidence_type,
            "produced_record_types": list(self.produced_record_types),
            "description": self.description,
            "estimated_cost": self.estimated_cost,
        }


@dataclass(frozen=True)
class _CurrentRecordState:
    """Hold the exact active CAD hypothesis and its derived record lineage."""

    bindings: Mapping[str, TypedContextBinding]
    cad_bindings: tuple[TypedContextBinding, ...]
    current_segmentation: TypedContextBinding | None
    active_cad: TypedContextBinding | None
    selected_correspondence: TypedContextBinding | None
    attempted_derived: frozenset[tuple[str, str]]


class ProductionGroundingError(RuntimeError):
    """Raised when production grounding cannot make safe progress."""


class CameraToWorldCalibrationRuntime(Protocol):
    """Provide one approved camera-to-world calibration on explicit demand."""

    def materialize_camera_to_world_calibration(
        self,
        *,
        interaction_root: Path,
        camera_pose_record_path: Path,
        source_frame: str,
        target_frame: str,
        calibration_number: int,
    ) -> CameraToRobotCalibrationResult:
        """Persist and return an approved calibration valid for the observation."""
        ...


class ProductionProductContextGroundingRuntime:
    """Resolve pre-RA product context with real controlled tools."""

    def __init__(
        self,
        *,
        tbox: TBoxSnapshot,
        document_config: DocumentVLMConfig,
        document_vision_runtime: DocumentVisionRuntime,
        camera_to_world_calibration_runtime: CameraToWorldCalibrationRuntime
        | None = None,
    ) -> None:
        """Create a runtime pinned to one authoritative TBox snapshot."""
        tbox.assert_unchanged()
        self._tbox = tbox
        self._document_config = document_config
        self._document_vision_runtime = document_vision_runtime
        self._camera_to_world_calibration_runtime = (
            camera_to_world_calibration_runtime
        )
        self._registry = load_predefined_resource_registry(tbox)
        self._workcell = load_predefined_workcell(tbox, self._registry)
        self._descriptors = _producer_descriptors(
            tbox,
            calibration_available=camera_to_world_calibration_runtime is not None,
        )

    def grounding_producer_descriptors(
        self,
    ) -> Sequence[GroundingProducerDescriptor]:
        """Return the exact output-capable production registry."""
        self._tbox.assert_unchanged()
        return self._descriptors

    async def initial_product_context_decision(
        self,
        product_agent: ProductAgentContextRuntime,
        *,
        interaction_root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        product_context: Mapping[str, object],
        max_pa_turns: int,
    ) -> Mapping[str, object]:
        """Create the first evidence-first grounding decision."""
        return await self._assess(
            product_agent,
            interaction_root=interaction_root,
            tbox=tbox,
            abox=abox,
            product_context=product_context,
            attempted_evidence=(),
            clarification_history=(),
            turn_number=1,
            max_pa_turns=max_pa_turns,
        )

    async def interpret_served_context(
        self,
        *,
        producer: str,
        interaction_root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        served_context: Mapping[str, object],
        operation_number: int,
    ) -> Mapping[str, object]:
        """Run the producer selected for one exact served source."""
        self._validate_tbox(tbox, abox)
        selected_output = _selected_output_for_served_context(
            Path(interaction_root), served_context
        )
        if producer == _DOCUMENT_PRODUCER:
            if served_context.get("evidence_type") != "document":
                raise ProductionGroundingError(
                    "Document producer received non-document evidence."
                )
            session = load_latest_grounding_session(Path(interaction_root))
            if session is not None and session.next_action.action == "inspect":
                context_ref = served_context.get("context_ref")
                if not isinstance(context_ref, str):
                    raise ProductionGroundingError(
                        "Targeted document action requires an exact context_ref."
                    )
                cache_status = document_overview_cache_status(
                    context_ref,
                    cache_root=Path(interaction_root).resolve().parent
                    / "source_cache",
                    config=self._document_config,
                )
                record_path = cache_status.get("record_path")
                if (
                    cache_status.get("overview_status") != "prepared"
                    or not isinstance(record_path, str)
                ):
                    raise ProductionGroundingError(
                        "Targeted document action requires a prepared overview."
                    )
                overview = _read_json_mapping(
                    Path(record_path), "DocumentOverviewRecord cache"
                )
                snapshot = {
                    "schema_version": 1,
                    "record_type": "DocumentOverviewRecord",
                    "producer": _DOCUMENT_PRODUCER,
                    "operation_number": operation_number,
                    "cache_status": "hit",
                    "cache_record_ref": record_path,
                    "evidence_refs": sorted(_record_source_ids(overview)),
                    "overview": dict(overview),
                }
                evidence = await self._inspect_targeted_document_evidence(
                    root=Path(interaction_root).resolve(),
                    overview_snapshot=snapshot,
                    evidence_question=str(session.next_action.question),
                )
                return {
                    "assertions": [],
                    "uncertainty": list(evidence.record.get("uncertainty", [])),
                    "unresolved_evidence_needs": [],
                    "typed_context_refs": [
                        _relative_ref(Path(interaction_root), evidence.record_path)
                    ],
                }
            result = await interpret_document_evidence(
                interaction_root=interaction_root,
                tbox=tbox,
                abox=abox,
                served_context=served_context,
                operation_number=operation_number,
                config=self._document_config,
                vision_runtime=self._document_vision_runtime,
            )
            return result.delta
        if producer != _GEOMETRY_PRODUCER:
            raise ProductionGroundingError(f"Unknown production producer: {producer}.")

        preprocessing = await asyncio.to_thread(
            preprocess_served_geometry,
            interaction_root=interaction_root,
            served_context=served_context,
            operation_number=operation_number,
        )
        delta = dict(preprocessing.delta)
        if served_context.get("evidence_type") != "observation":
            return delta
        if selected_output not in {
            "RGBDSegmentationRecord",
            "CADSizeCorrespondenceRecord",
            "CADPoseEstimationRecord",
        }:
            return delta
        segmentation = await asyncio.to_thread(
            segment_preprocessed_observation,
            interaction_root=interaction_root,
            observation_record_path=preprocessing.record_path,
            segmentation_number=_next_number(
                Path(interaction_root),
                "products/grounding/rgb_d_cad_grounding/segmentation_*",
            ),
        )
        typed_refs = list(delta.get("typed_context_refs", []))
        typed_refs.append(_relative_ref(Path(interaction_root), segmentation.record_path))
        delta["typed_context_refs"] = typed_refs
        return delta

    async def assess_product_context(  # noqa: PLR0913
        self,
        product_agent: ProductAgentContextRuntime,
        *,
        interaction_root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        abox_view: Mapping[str, object],
        attempted_evidence: tuple[str, ...],
        clarification_history: tuple[Mapping[str, object], ...] = (),
        turn_number: int,
        max_pa_turns: int,
    ) -> Mapping[str, object]:
        """Update the grounding session from newly accepted evidence."""
        return await self._assess(
            product_agent,
            interaction_root=interaction_root,
            tbox=tbox,
            abox=abox,
            product_context=abox_view,
            attempted_evidence=attempted_evidence,
            clarification_history=clarification_history,
            turn_number=turn_number,
            max_pa_turns=max_pa_turns,
        )

    async def _assess(  # noqa: C901, PLR0913
        self,
        product_agent: ProductAgentContextRuntime,
        *,
        interaction_root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        product_context: Mapping[str, object],
        attempted_evidence: tuple[str, ...],
        clarification_history: tuple[Mapping[str, object], ...],
        turn_number: int,
        max_pa_turns: int,
    ) -> Mapping[str, object]:
        self._validate_tbox(tbox, abox)
        root = Path(interaction_root).resolve()
        view = ProductContextView.from_mapping(product_context)
        if view.tbox_fingerprint != tbox.fingerprint:
            raise ProductionGroundingError(
                "ProductContextView does not match the authoritative TBox."
            )
        previous = load_latest_grounding_session(root)
        if previous is not None and previous.requirement_text != view.product_requirement:
            raise ProductionGroundingError(
                "GroundingSession requirement does not match ProductContextView."
            )
        pending_attempt = _pending_action_attempt(
            previous,
            view,
            clarification_history=clarification_history,
        )
        try:
            need = derive_resource_assignment_need(abox, self._workcell)
        except ResourceGroundingError as exc:
            raise ProductionGroundingError(
                f"Resource-assignment need could not be derived: {exc}"
            ) from exc

        if need is not None:
            world_pose = _fresh_required_pose_binding(root, view, need)
            if world_pose is not None:
                try:
                    selection = select_predefined_resource(
                        interaction_root=root,
                        tbox=tbox,
                        registry=self._registry,
                        workcell=self._workcell,
                        need=need,
                        robot_frame_pose_path=root / world_pose.record_ref,
                        selection_number=_next_number(
                            root,
                            "products/grounding/resource_selection/selection_*",
                        ),
                    )
                except RobotFramePoseEvidenceError:
                    # A changed or invalid pose is no longer eligible. Provider
                    # closure below exposes a new evidence revision instead.
                    world_pose = None
                except ResourceGroundingError as exc:
                    raise ProductionGroundingError(
                        f"Predefined resource selection failed: {exc}"
                    ) from exc
                else:
                    if selection.selected_resource_iri is None:
                        unresolved = _resource_assignment_incomplete_session(
                            previous,
                            requirement_text=view.product_requirement,
                            pending_attempt=pending_attempt,
                            reason=(
                                "No predefined resource is coarsely reachable from "
                                "the accepted world-frame pose."
                            ),
                        )
                        persist_grounding_session(root, unresolved)
                        return _session_assessment(unresolved, ())
                    try:
                        assignment = commit_resource_assignment(
                            interaction_root=root,
                            tbox=tbox,
                            registry=self._registry,
                            workcell=self._workcell,
                            need=need,
                            selection=selection,
                        )
                    except ResourceGroundingError as exc:
                        raise ProductionGroundingError(
                            f"Host resource assignment failed: {exc}"
                        ) from exc
                    final_view = build_product_context_view(
                        root,
                        assignment.abox,
                        attempted_evidence=attempted_evidence,
                        assessed_at_ns=time.time_ns(),
                    )
                    persist_product_context_view(root, final_view)
                    completed = _resource_assignment_complete_session(
                        previous,
                        requirement_text=view.product_requirement,
                        pending_attempt=pending_attempt,
                    )
                    persist_grounding_session(root, completed)
                    return _session_assessment(completed, ())

        required_record_type = None if need is None else need.required_record_type
        required_target_frame = None if need is None else need.target_frame
        previews, authorized_sources = _collect_provider_previews(
            root,
            view,
            clarification_history=clarification_history,
            config=self._document_config,
            include_live_observation=True,
        )
        actions = _discover_provider_actions(
            root,
            self._descriptors,
            previews=previews,
            view=view,
            previous=previous,
            pending_attempt=pending_attempt,
            required_record_type=required_record_type,
            required_target_frame=required_target_frame,
        )
        automatic_action = next(
            (
                action
                for action in actions
                if action.automatic
            ),
            None,
        )
        if automatic_action is not None:
            if previous is None:
                raise ProductionGroundingError(
                    "A derived resource-grounding operation requires a provisional "
                    "grounding session."
                )
            updated_abox = self._run_session_derived_action(
                root=root,
                tbox=tbox,
                abox=abox,
                view=view,
                session=previous,
                action=automatic_action,
                assignment_need=need,
            )
            updated_view = build_product_context_view(
                root,
                updated_abox,
                attempted_evidence=attempted_evidence,
                assessed_at_ns=time.time_ns(),
            )
            persist_product_context_view(root, updated_view)
            return await self._assess(
                product_agent,
                interaction_root=root,
                tbox=tbox,
                abox=updated_abox,
                product_context=updated_view.to_record(),
                attempted_evidence=attempted_evidence,
                clarification_history=clarification_history,
                turn_number=turn_number,
                max_pa_turns=max_pa_turns,
            )
        if previous is not None and previous.revision >= max_pa_turns:
            session = _emergency_incomplete_session(previous, pending_attempt)
        else:
            session = await _author_grounding_session(
                product_agent,
                tbox=tbox,
                requirement_text=view.product_requirement,
                previous=previous,
                pending_attempt=pending_attempt,
                previews=previews,
                actions=actions,
                turn_number=turn_number,
                max_pa_turns=max_pa_turns,
                allow_proposal=need is None,
            )
        persist_grounding_session(root, session)
        selected_action = _selected_session_action(session, actions)
        if (
            selected_action is not None
            and selected_action.evidence_type == "existing_record"
        ):
            updated_abox = self._run_session_derived_action(
                root=root,
                tbox=tbox,
                abox=abox,
                view=view,
                session=session,
                action=selected_action,
                assignment_need=need,
            )
            updated_view = build_product_context_view(
                root,
                updated_abox,
                attempted_evidence=attempted_evidence,
                assessed_at_ns=time.time_ns(),
            )
            persist_product_context_view(root, updated_view)
            return await self._assess(
                product_agent,
                interaction_root=root,
                tbox=tbox,
                abox=updated_abox,
                product_context=updated_view.to_record(),
                attempted_evidence=attempted_evidence,
                clarification_history=clarification_history,
                turn_number=turn_number,
                max_pa_turns=max_pa_turns,
            )
        if session.status == "ready_for_ontology":
            try:
                mapping = await propose_and_validate_ontology_grounding(
                    product_agent,
                    interaction_root=root,
                    tbox=tbox,
                    abox=abox,
                    workcell=self._workcell,
                    session=session,
                    evidence_previews=previews,
                    authorized_evidence_refs=authorized_sources,
                )
            except OntologyGroundingError as exc:
                gap = _incomplete_session(
                    revision=session.revision + 1,
                    requirement_text=session.requirement_text,
                    attempted_actions=session.attempted_actions,
                    reason=f"Late ontology mapping was rejected: {exc}",
                    status="ontology_gap",
                )
                persist_grounding_session(root, gap)
                return _session_assessment(gap, actions)
            provisional_view = build_product_context_view(
                root,
                mapping.merge.abox,
                attempted_evidence=attempted_evidence,
                assessed_at_ns=time.time_ns(),
            )
            persist_product_context_view(root, provisional_view)
            return await self._assess(
                product_agent,
                interaction_root=root,
                tbox=tbox,
                abox=mapping.merge.abox,
                product_context=provisional_view.to_record(),
                attempted_evidence=attempted_evidence,
                clarification_history=clarification_history,
                turn_number=turn_number,
                max_pa_turns=max_pa_turns,
            )
        return _session_assessment(session, actions)

    async def _inspect_targeted_document_evidence(
        self,
        *,
        root: Path,
        overview_snapshot: Mapping[str, object],
        evidence_question: str,
    ) -> DocumentEvidenceRecord:
        try:
            return await inspect_document_evidence(
                interaction_root=root,
                overview_snapshot=overview_snapshot,
                evidence_question=evidence_question,
                config=self._document_config,
                vision_runtime=self._document_vision_runtime,
            )
        except DocumentInterpretationError as exc:
            raise ProductionGroundingError(
                f"Targeted document evidence failed: {exc}"
            ) from exc

    def _run_session_derived_action(  # noqa: C901, PLR0913
        self,
        *,
        root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        view: ProductContextView,
        session: GroundingSession,
        action: _EligibleProviderAction,
        assignment_need: ResourceAssignmentNeed | None,
    ) -> ABoxSnapshot:
        """Execute one selected zero-retrieval provider action from typed records."""
        del session
        output_types = action.produced_record_types
        if len(output_types) != 1:
            raise ProductionGroundingError(
                "Derived provider action does not select one exact output record."
            )
        output_type = output_types[0]
        if output_type == "CADMeshRecord":
            _exact_action_bindings(
                root,
                view,
                action,
                ("CADMeshRecord",),
            )
            return abox
        if output_type == "CADSizeCorrespondenceRecord":
            required = ("CADMeshRecord", "RGBDSegmentationRecord")
            bindings = _exact_action_bindings(
                root,
                view,
                action,
                required,
            )
            result = associate_segmented_candidate_by_size(
                interaction_root=root,
                cad_record_path=root / bindings["CADMeshRecord"].record_ref,
                segmentation_record_path=(
                    root / bindings["RGBDSegmentationRecord"].record_ref
                ),
                correspondence_number=_next_number(
                    root,
                    "products/grounding/rgb_d_cad_grounding/correspondence_*",
                ),
            )
            status = result.CAD_correspondence
            record_path = result.record_path
        elif output_type == "CADPoseEstimationRecord":
            required = ("CADSizeCorrespondenceRecord",)
            bindings = _exact_action_bindings(
                root,
                view,
                action,
                required,
            )
            result = estimate_camera_frame_pose(
                interaction_root=root,
                correspondence_record_path=(
                    root / bindings["CADSizeCorrespondenceRecord"].record_ref
                ),
                pose_number=_next_number(
                    root,
                    "products/grounding/rgb_d_cad_grounding/pose_*",
                ),
            )
            status = result.pose
            record_path = result.record_path
        elif output_type == "CameraToRobotCalibrationRecord":
            required = ("CADPoseEstimationRecord",)
            if (
                assignment_need is None
                or self._camera_to_world_calibration_runtime is None
            ):
                raise ProductionGroundingError(
                    "No approved CameraToWorldCalibrationRuntime is available."
                )
            bindings = _exact_action_bindings(
                root,
                view,
                action,
                required,
            )
            pose_binding = bindings["CADPoseEstimationRecord"]
            pose_record = _read_json_mapping(
                root / pose_binding.record_ref,
                "CADPoseEstimationRecord",
            )
            source_frame = pose_record.get("coordinate_frame")
            if not isinstance(source_frame, str) or not source_frame:
                raise ProductionGroundingError(
                    "Accepted camera pose has no exact source frame."
                )
            result = (
                self._camera_to_world_calibration_runtime
                .materialize_camera_to_world_calibration(
                    interaction_root=root,
                    camera_pose_record_path=root / pose_binding.record_ref,
                    source_frame=source_frame,
                    target_frame=assignment_need.target_frame,
                    calibration_number=_next_number(
                        root,
                        "products/grounding/rgb_d_cad_grounding/calibration_*",
                    ),
                )
            )
            if (
                result.source_frame != source_frame
                or result.target_frame != assignment_need.target_frame
            ):
                raise ProductionGroundingError(
                    "Injected calibration did not provide the requested frame pair."
                )
            status = "accepted"
            record_path = result.record_path
        elif output_type == "RobotFramePoseRecord":
            required = (
                "CADPoseEstimationRecord",
                "CameraToRobotCalibrationRecord",
            )
            if assignment_need is None:
                raise ProductionGroundingError(
                    "World-frame pose conversion requires a resource-assignment need."
                )
            bindings = _exact_action_bindings(
                root,
                view,
                action,
                required,
            )
            try:
                result = transform_camera_pose_to_robot_frame(
                    interaction_root=root,
                    pose_record_path=(
                        root / bindings["CADPoseEstimationRecord"].record_ref
                    ),
                    calibration_record_path=(
                        root
                        / bindings["CameraToRobotCalibrationRecord"].record_ref
                    ),
                    target_frame=assignment_need.target_frame,
                    conversion_number=_next_number(
                        root,
                        "products/grounding/rgb_d_cad_grounding/robot_pose_*",
                    ),
                )
            except RobotFrameConversionError as exc:
                raise ProductionGroundingError(
                    f"World-frame pose conversion failed: {exc}"
                ) from exc
            status = result.robot_frame_conversion
            record_path = result.record_path
        else:
            raise ProductionGroundingError(
                f"No internal provider action produces {output_type}."
            )
        record_ref = _relative_ref(root, record_path)
        evidence_refs = sorted(
            {
                evidence_ref
                for record_type in required
                for evidence_ref in (
                    bindings[record_type].record_ref,
                    *bindings[record_type].evidence_refs,
                )
            }
        )
        unresolved = []
        if status != "accepted":
            unresolved.append(
                {
                    "description": f"{output_type} is {status}.",
                    "evidence_refs": evidence_refs,
                }
            )
        merge = validate_and_merge_triple_delta(
            root,
            tbox,
            _GEOMETRY_PRODUCER,
            {
                "assertions": [],
                "uncertainty": [],
                "unresolved_evidence_needs": unresolved,
                "typed_context_refs": [record_ref],
            },
            authorized_evidence_refs=evidence_refs,
        )
        return merge.abox

    def _validate_tbox(self, tbox: TBoxSnapshot, abox: ABoxSnapshot) -> None:
        self._tbox.assert_unchanged()
        self._registry.assert_unchanged()
        self._workcell.assert_unchanged()
        tbox.assert_unchanged()
        if (
            tbox.fingerprint != self._tbox.fingerprint
            or abox.tbox_fingerprint != tbox.fingerprint
        ):
            raise ProductionGroundingError(
                "Production grounding inputs do not match the pinned TBox."
            )


def _collect_provider_previews(  # noqa: C901
    root: Path,
    view: ProductContextView,
    *,
    clarification_history: Sequence[Mapping[str, object]],
    config: DocumentVLMConfig,
    include_live_observation: bool = False,
) -> tuple[list[dict[str, object]], set[str]]:
    """Collect inference-free, authorized previews for one PA session update."""
    previews: list[dict[str, object]] = []
    authorized_sources = {"requirement_0001"}
    approved = approved_context_ref_evidence_types()
    for context_ref, evidence_type in sorted(approved.items()):
        if evidence_type == "document":
            status = document_overview_cache_status(
                context_ref,
                cache_root=root.parent / "source_cache",
                config=config,
            )
            preview: dict[str, object] = {
                "provider_id": _DOCUMENT_PRODUCER,
                "evidence_type": "document",
                "source_ref": context_ref,
                "source_revision": status.get("source_sha256"),
                "availability": status.get("source_status") == "valid",
                "overview_status": status.get("overview_status"),
                "summary": None,
                "observations": [],
                "uncertainty": [],
            }
            record_path = status.get("record_path")
            if status.get("overview_status") == "prepared" and isinstance(
                record_path, str
            ):
                record = _read_json_mapping(
                    Path(record_path), "DocumentOverviewRecord cache"
                )
                preview.update(
                    {
                        "cache_fingerprint": record.get("cache_fingerprint"),
                        "summary": record.get("summary"),
                        "observations": record.get("observations", []),
                        "uncertainty": record.get("uncertainty", []),
                    }
                )
                authorized_sources.add(context_ref)
                authorized_sources.update(_record_source_ids(record))
            preview["produced_record_types"] = (
                ["DocumentEvidenceRecord"]
                if preview["overview_status"] == "prepared"
                else ["DocumentOverviewRecord"]
            )
            previews.append(preview)
            continue
        try:
            source_path = approved_cad_path(context_ref)
            source_revision = hashlib.sha256(source_path.read_bytes()).hexdigest()
            availability = True
        except (OSError, TypeError, ValueError):
            source_revision = _fingerprint_value(
                {"context_ref": context_ref, "status": "unavailable"}
            )
            availability = False
        previews.append(
            {
                "provider_id": _GEOMETRY_PRODUCER,
                "evidence_type": "CAD",
                "source_ref": context_ref,
                "source_revision": source_revision,
                "availability": availability,
                "produced_record_types": ["CADMeshRecord"],
                "metadata": {"context_ref": context_ref, "registered": True},
            }
        )

    accepted_bindings = [
        binding
        for binding in view.typed_bindings
        if binding.status == "accepted" and _binding_file_is_intact(root, binding)
    ]
    for binding in sorted(accepted_bindings, key=lambda item: item.record_ref):
        record_path = root / binding.record_ref
        record = _read_json_mapping(record_path, binding.record_type)
        source_ids = {binding.record_ref, *binding.evidence_refs}
        source_ids.update(_record_source_ids(record))
        authorized_sources.update(source_ids)
        previews.append(
            {
                "provider_id": binding.producer,
                "evidence_type": "accepted_typed_record",
                "source_ref": binding.record_ref,
                "source_revision": binding.record_sha256,
                "availability": True,
                "record_type": binding.record_type,
                "source_ids": sorted(source_ids),
                "content": record,
            }
        )

    if include_live_observation:
        previews.append(
            {
                "provider_id": _GEOMETRY_PRODUCER,
                "evidence_type": "observation",
                "source_ref": "live_RGB-D",
                "source_revision": _observation_frontier_revision(root, view),
                "availability": True,
                "produced_record_types": [
                    "ColoredPointCloudSetRecord",
                    "RGBDSegmentationRecord",
                ],
                "metadata": {"fresh_capture_available": True},
            }
        )
    for index, clarification in enumerate(clarification_history, start=1):
        source_id = f"clarification_{index:04d}"
        authorized_sources.add(source_id)
        previews.append(
            {
                "provider_id": "user_clarification",
                "evidence_type": "user",
                "source_ref": source_id,
                "source_revision": str(
                    clarification.get("fingerprint")
                    or _fingerprint_value(dict(clarification))
                ),
                "availability": True,
                "reply": clarification.get("reply"),
            }
        )
    return previews, authorized_sources


def _pending_action_attempt(
    previous: GroundingSession | None,
    view: ProductContextView,
    *,
    clarification_history: Sequence[Mapping[str, object]],
) -> GroundingActionAttempt | None:
    """Describe the controller-owned result of the previously selected action."""
    if previous is None or previous.next_action.action not in {
        "retrieve",
        "inspect",
        "ask_user",
    }:
        return None
    previous_refs = {
        ref for attempt in previous.attempted_actions for ref in attempt.record_refs
    }
    record_refs: list[str] = []
    if previous.next_action.action == "ask_user":
        for index, _clarification in enumerate(clarification_history, start=1):
            source_id = f"clarification_{index:04d}"
            if source_id not in previous_refs:
                record_refs.append(source_id)
    else:
        for binding in view.typed_bindings:
            if (
                binding.status == "accepted"
                and binding.producer == previous.selected_provider_id
                and binding.record_ref not in previous_refs
            ):
                record_refs.append(binding.record_ref)
    attempt = GroundingActionAttempt.from_mapping(
        {
            "attempt_id": f"attempt_{len(previous.attempted_actions) + 1:04d}",
            "action": previous.next_action.action,
            "provider_id": previous.selected_provider_id,
            "source_ref": previous.next_action.source_ref or "user_intent",
            "source_revision": previous.selected_source_revision,
            "question": previous.next_action.question,
            "status": "accepted" if record_refs else "no_change",
            "record_refs": sorted(record_refs),
        }
    )
    return attempt


def _discover_provider_actions(  # noqa: C901, PLR0912
    root: Path,
    descriptors: Sequence[GroundingProducerDescriptor],
    *,
    previews: Sequence[Mapping[str, object]],
    view: ProductContextView,
    previous: GroundingSession | None,
    pending_attempt: GroundingActionAttempt | None,
    required_record_type: str | None = None,
    required_target_frame: str | None = None,
) -> tuple[_EligibleProviderAction, ...]:
    """Build actions from provider-owned capabilities and current prerequisites."""
    descriptor_by_id = {item.provider_id: item for item in descriptors}
    required_types = (
        None
        if required_record_type is None
        else _required_record_closure(descriptors, required_record_type)
    )
    record_state = _current_record_state(
        root,
        view,
        descriptors,
        required_target_frame=required_target_frame,
        previous=previous,
        pending_attempt=pending_attempt,
    )
    actions: list[_EligibleProviderAction] = []
    geometry_descriptor = descriptor_by_id.get(_GEOMETRY_PRODUCER)
    if (
        geometry_descriptor is not None
        and record_state.active_cad is not None
        and record_state.current_segmentation is not None
    ):
        prerequisites = {
            "CADMeshRecord": record_state.active_cad,
            "RGBDSegmentationRecord": record_state.current_segmentation,
        }
        revision = _cad_segmentation_revision(
            root,
            record_state.active_cad,
            record_state.current_segmentation,
        )
        if (
            "CADSizeCorrespondenceRecord",
            revision,
        ) not in record_state.attempted_derived:
            actions.append(
                _EligibleProviderAction(
                    action="retrieve",
                    provider_id=geometry_descriptor.provider_id,
                    source_ref=_cad_context_ref(root, record_state.active_cad),
                    source_revision=revision,
                    evidence_type="existing_record",
                    produced_record_types=("CADSizeCorrespondenceRecord",),
                    description=(
                        "Compare the one PA-selected CAD hypothesis against the "
                        "current RGB-D segmentation."
                    ),
                    estimated_cost=geometry_descriptor.estimated_cost,
                    automatic=True,
                    prerequisite_bindings=tuple(prerequisites.items()),
                )
            )

    for preview in previews:
        if preview.get("availability") is not True:
            continue
        provider_id = preview.get("provider_id")
        evidence_type = preview.get("evidence_type")
        source_ref = preview.get("source_ref")
        source_revision = preview.get("source_revision")
        descriptor = descriptor_by_id.get(provider_id)
        if (
            descriptor is None
            or not isinstance(evidence_type, str)
            or evidence_type not in descriptor.accepted_evidence_types
            or not isinstance(source_ref, str)
            or not isinstance(source_revision, str)
        ):
            continue
        if evidence_type == "CAD":
            if (
                record_state.active_cad is not None
                or record_state.selected_correspondence is not None
            ):
                continue
            cached = _matching_cad_binding(
                root,
                record_state.cad_bindings,
                source_ref=source_ref,
                source_revision=source_revision,
            )
            if cached is not None:
                if record_state.current_segmentation is None:
                    actions.append(
                        _EligibleProviderAction(
                            action="retrieve",
                            provider_id=descriptor.provider_id,
                            source_ref=source_ref,
                            source_revision=source_revision,
                            evidence_type="existing_record",
                            produced_record_types=("CADMeshRecord",),
                            description=(
                                "Select this cached approved CAD as the one active "
                                "hypothesis."
                            ),
                            estimated_cost=descriptor.estimated_cost,
                            prerequisite_bindings=(("CADMeshRecord", cached),),
                        )
                    )
                    continue
                prerequisites = {
                    "CADMeshRecord": cached,
                    "RGBDSegmentationRecord": record_state.current_segmentation,
                }
                revision = _cad_segmentation_revision(
                    root,
                    cached,
                    record_state.current_segmentation,
                )
                if (
                    "CADSizeCorrespondenceRecord",
                    revision,
                ) in record_state.attempted_derived:
                    continue
                actions.append(
                    _EligibleProviderAction(
                        action="retrieve",
                        provider_id=descriptor.provider_id,
                        source_ref=source_ref,
                        source_revision=revision,
                        evidence_type="existing_record",
                        produced_record_types=("CADSizeCorrespondenceRecord",),
                        description=(
                            "Select this approved CAD as the next hypothesis and "
                            "compare only it against the current RGB-D segmentation."
                        ),
                        estimated_cost=descriptor.estimated_cost,
                        prerequisite_bindings=tuple(prerequisites.items()),
                    )
                )
                continue
        preview_outputs = preview.get("produced_record_types")
        produced = tuple(
            item
            for item in (
                preview_outputs if isinstance(preview_outputs, list) else []
            )
            if isinstance(item, str)
            and descriptor.supports_record_type(item)
        )
        if not produced:
            continue
        if evidence_type == "document":
            revision = _fingerprint_value(
                {
                    "source_revision": source_revision,
                    "overview_status": preview.get("overview_status"),
                    "produced_record_types": produced,
                }
            )
        elif evidence_type in {"CAD", "observation"}:
            revision = source_revision
        else:
            continue
        actions.append(
            _EligibleProviderAction(
                action=(
                    "inspect"
                    if evidence_type == "document"
                    and "DocumentEvidenceRecord" in produced
                    else "retrieve"
                ),
                provider_id=descriptor.provider_id,
                source_ref=source_ref,
                source_revision=revision,
                evidence_type=evidence_type,
                produced_record_types=produced,
                description=descriptor.description,
                estimated_cost=descriptor.estimated_cost,
            )
        )

    for descriptor in descriptor_by_id.values():
        if (
            descriptor.availability
            and "existing_record" in descriptor.accepted_evidence_types
        ):
            for record_type in descriptor.produced_record_types:
                if record_type == "CADSizeCorrespondenceRecord":
                    continue
                if required_types is not None and record_type not in required_types:
                    continue
                if (
                    record_type
                    in {"CameraToRobotCalibrationRecord", "RobotFramePoseRecord"}
                    and required_target_frame is None
                ):
                    continue
                if record_type in record_state.bindings:
                    continue
                prerequisites = descriptor.prerequisites_for(record_type)
                if not prerequisites or not set(prerequisites).issubset(
                    record_state.bindings
                ):
                    continue
                prerequisite_revision = _prerequisite_revision(
                    prerequisites,
                    record_state.bindings,
                )
                if (
                    record_type,
                    prerequisite_revision,
                ) in record_state.attempted_derived:
                    continue
                actions.append(
                    _EligibleProviderAction(
                        action="retrieve",
                        provider_id=descriptor.provider_id,
                        source_ref=f"accepted_typed_records:{record_type}",
                        source_revision=prerequisite_revision,
                        evidence_type="existing_record",
                        produced_record_types=(record_type,),
                        description=descriptor.description,
                        estimated_cost=descriptor.estimated_cost,
                        automatic=True,
                        prerequisite_bindings=tuple(
                            (item, record_state.bindings[item])
                            for item in prerequisites
                        ),
                    )
                )
    clarification = descriptor_by_id.get("user_clarification")
    if clarification is not None and clarification.availability:
        actions.append(
            _EligibleProviderAction(
                action="ask_user",
                provider_id=clarification.provider_id,
                source_ref="user_intent",
                source_revision=_fingerprint_value(
                    {
                        "clarification_count": len(
                            [
                                item
                                for item in previews
                                if item.get("evidence_type") == "user"
                            ]
                        )
                    }
                ),
                evidence_type="user",
                produced_record_types=("PAClarification",),
                description=clarification.description,
                estimated_cost=clarification.estimated_cost,
            )
        )

    cad_choices = [
        action
        for action in actions
        if not action.automatic
        and (
            action.evidence_type == "CAD"
            or action.produced_record_types
            in {("CADMeshRecord",), ("CADSizeCorrespondenceRecord",)}
        )
    ]
    pose_revision = None
    if record_state.selected_correspondence is not None:
        pose_revision = _prerequisite_revision(
            ("CADSizeCorrespondenceRecord",),
            {
                "CADSizeCorrespondenceRecord": (
                    record_state.selected_correspondence
                )
            },
        )
    pose_failed = bool(
        pose_revision is not None
        and (
            "CADPoseEstimationRecord",
            pose_revision,
        )
        in record_state.attempted_derived
        and "CADPoseEstimationRecord" not in record_state.bindings
    )
    if record_state.current_segmentation is not None and (
        record_state.active_cad is not None
        or cad_choices
        or (
            record_state.selected_correspondence is not None
            and not pose_failed
        )
    ):
        actions = [
            action for action in actions if action.evidence_type != "observation"
        ]

    attempts = list(previous.attempted_actions) if previous is not None else []
    if pending_attempt is not None:
        attempts.append(pending_attempt)
    attempted_keys = {item.action_key for item in attempts}
    actions = [
        action
        for action in actions
        if action.automatic
        or action.action != "retrieve"
        or (
            action.action,
            action.provider_id,
            action.source_ref,
            action.source_revision,
            None,
        )
        not in attempted_keys
    ]
    return tuple(
        sorted(
            actions,
            key=lambda item: (
                item.provider_id,
                item.source_ref,
                item.source_revision,
                item.produced_record_types,
            ),
        )
    )


def _required_record_closure(
    descriptors: Sequence[GroundingProducerDescriptor],
    required_record_type: str,
) -> frozenset[str]:
    """Return every typed output that can contribute to one required output."""
    producers: dict[str, list[GroundingProducerDescriptor]] = {}
    for descriptor in descriptors:
        for record_type in descriptor.produced_record_types:
            producers.setdefault(record_type, []).append(descriptor)

    closure: set[str] = set()
    pending = [required_record_type]
    while pending:
        record_type = pending.pop()
        if record_type in closure:
            continue
        closure.add(record_type)
        for descriptor in producers.get(record_type, []):
            pending.extend(descriptor.prerequisites_for(record_type))
    return frozenset(closure)


def _current_record_state(  # noqa: C901, PLR0912, PLR0915
    root: Path,
    view: ProductContextView,
    descriptors: Sequence[GroundingProducerDescriptor],
    *,
    required_target_frame: str | None,
    previous: GroundingSession | None = None,
    pending_attempt: GroundingActionAttempt | None = None,
) -> _CurrentRecordState:
    """Resolve one PA-selected CAD hypothesis and its exact derived lineage."""
    grouped: dict[str, list[TypedContextBinding]] = {}
    for binding in view.typed_bindings:
        if not _binding_file_is_intact(root, binding):
            continue
        if (
            binding.record_type
            in {"CameraToRobotCalibrationRecord", "RobotFramePoseRecord"}
            and required_target_frame is not None
            and binding.frame != required_target_frame
        ):
            continue
        grouped.setdefault(binding.record_type, []).append(binding)

    prerequisite_options: dict[str, list[tuple[str, ...]]] = {}
    for descriptor in descriptors:
        for record_type in descriptor.produced_record_types:
            prerequisites = descriptor.prerequisites_for(record_type)
            options = prerequisite_options.setdefault(record_type, [])
            if prerequisites not in options:
                options.append(prerequisites)

    cad_bindings = tuple(
        sorted(
            (
                binding
                for binding in grouped.get("CADMeshRecord", [])
                if binding.status == "accepted"
            ),
            key=lambda binding: binding.record_ref,
        )
    )
    segmentations = [
        binding
        for binding in grouped.get("RGBDSegmentationRecord", [])
        if binding.status == "accepted"
    ]
    current_segmentation = (
        None if not segmentations else _newest_binding(segmentations)
    )

    current: dict[str, TypedContextBinding] = {}
    for record_type, options in prerequisite_options.items():
        if not options or any(options):
            continue
        if record_type in {"CADMeshRecord", "RGBDSegmentationRecord"}:
            continue
        accepted = [
            binding
            for binding in grouped.get(record_type, [])
            if binding.status == "accepted"
        ]
        if accepted:
            current[record_type] = _newest_binding(accepted)
    if current_segmentation is not None:
        current["RGBDSegmentationRecord"] = current_segmentation

    attempted: set[tuple[str, str]] = set()
    pair_outputs: dict[str, list[TypedContextBinding]] = {}
    for binding in grouped.get("CADSizeCorrespondenceRecord", []):
        revision = _correspondence_pair_revision(root, binding)
        if revision is not None:
            pair_outputs.setdefault(revision, []).append(binding)
    if current_segmentation is not None:
        for cad in cad_bindings:
            revision = _cad_segmentation_revision(
                root,
                cad,
                current_segmentation,
            )
            matches = pair_outputs.get(revision, [])
            if matches:
                attempted.add(("CADSizeCorrespondenceRecord", revision))

    selected_events: list[tuple[str, str, tuple[str, ...]]] = []
    if previous is not None:
        selected_events.extend(
            (
                attempt.source_ref,
                attempt.source_revision,
                attempt.record_refs,
            )
            for attempt in previous.attempted_actions
            if attempt.action == "retrieve"
            and attempt.provider_id == _GEOMETRY_PRODUCER
        )
    if (
        pending_attempt is not None
        and pending_attempt.action == "retrieve"
        and pending_attempt.provider_id == _GEOMETRY_PRODUCER
    ):
        selected_events.append(
            (
                pending_attempt.source_ref,
                pending_attempt.source_revision,
                pending_attempt.record_refs,
            )
        )
    if (
        previous is not None
        and previous.next_action.action == "retrieve"
        and previous.selected_provider_id == _GEOMETRY_PRODUCER
        and previous.next_action.source_ref is not None
        and previous.selected_source_revision is not None
    ):
        selected_events.append(
            (
                previous.next_action.source_ref,
                previous.selected_source_revision,
                (),
            )
        )

    active_cad: TypedContextBinding | None = None
    selected_cad: TypedContextBinding | None = None
    selected_correspondence: TypedContextBinding | None = None
    for source_ref, source_revision, record_refs in reversed(selected_events):
        cad = _selected_cad_binding(
            root,
            cad_bindings,
            source_ref=source_ref,
            source_revision=source_revision,
            record_refs=record_refs,
            segmentation=current_segmentation,
        )
        if cad is None:
            continue
        if current_segmentation is None:
            active_cad = cad
            break
        revision = _cad_segmentation_revision(root, cad, current_segmentation)
        matches = pair_outputs.get(revision, [])
        if not matches:
            active_cad = cad
            break
        accepted = [item for item in matches if item.status == "accepted"]
        if accepted:
            selected_cad = cad
            selected_correspondence = _newest_binding(accepted)
            break

    if active_cad is not None:
        current["CADMeshRecord"] = active_cad
    elif selected_cad is not None and selected_correspondence is not None:
        current["CADMeshRecord"] = selected_cad
        current["CADSizeCorrespondenceRecord"] = selected_correspondence

    changed = True
    while changed:
        changed = False
        for record_type, options in prerequisite_options.items():
            if record_type in {
                "CADMeshRecord",
                "RGBDSegmentationRecord",
                "CADSizeCorrespondenceRecord",
            }:
                continue
            for prerequisites in options:
                if not prerequisites or not set(prerequisites).issubset(current):
                    continue
                prerequisite_bindings = {
                    item: current[item] for item in prerequisites
                }
                revision = _prerequisite_revision(prerequisites, current)
                matches = [
                    binding
                    for binding in grouped.get(record_type, [])
                    if _binding_matches_prerequisites(
                        root,
                        binding,
                        prerequisite_bindings,
                        required_target_frame=required_target_frame,
                    )
                ]
                if not matches:
                    continue
                attempted.add((record_type, revision))
                accepted = [item for item in matches if item.status == "accepted"]
                if not accepted:
                    continue
                selected = _newest_binding(accepted)
                if current.get(record_type) != selected:
                    current[record_type] = selected
                    changed = True
    return _CurrentRecordState(
        bindings=current,
        cad_bindings=cad_bindings,
        current_segmentation=current_segmentation,
        active_cad=active_cad,
        selected_correspondence=selected_correspondence,
        attempted_derived=frozenset(attempted),
    )


def _prerequisite_revision(
    prerequisites: Sequence[str],
    bindings: Mapping[str, TypedContextBinding],
) -> str:
    """Fingerprint the exact record refs and hashes selected for one operation."""
    return _fingerprint_value(
        {
            record_type: {
                "record_ref": bindings[record_type].record_ref,
                "record_sha256": bindings[record_type].record_sha256,
            }
            for record_type in sorted(prerequisites)
        }
    )


def _cad_segmentation_revision(
    root: Path,
    cad: TypedContextBinding,
    segmentation: TypedContextBinding,
) -> str:
    """Key one comparison by CAD source hash and segmentation record hash."""
    # Reprocessing an unchanged STL is not a new semantic hypothesis, so its
    # operation-record hash must not reopen comparison for the same scene.
    return _fingerprint_value(
        {
            "CAD_source_sha256": _cad_source_sha256(root, cad),
            "segmentation_record_sha256": segmentation.record_sha256,
        }
    )


def _correspondence_pair_revision(
    root: Path,
    binding: TypedContextBinding,
) -> str | None:
    """Recover an intact correspondence's source-level comparison key."""
    record = _read_json_mapping(root / binding.record_ref, binding.record_type)
    cad = record.get("CAD")
    segmentation = record.get("segmentation")
    if not isinstance(cad, Mapping) or not isinstance(segmentation, Mapping):
        return None
    cad_record = _read_embedded_hashed_mapping(root, cad.get("record"))
    segmentation_ref = segmentation.get("record")
    segmentation_record = _read_embedded_hashed_mapping(root, segmentation_ref)
    if cad_record is None or segmentation_record is None:
        return None
    cad_source = cad_record.get("source")
    embedded_cad_source_sha256 = (
        cad_source.get("source_sha256")
        if isinstance(cad_source, Mapping)
        else None
    )
    declared_cad_source_sha256 = cad.get("source_sha256")
    segmentation_sha256 = (
        segmentation_ref.get("sha256")
        if isinstance(segmentation_ref, Mapping)
        else None
    )
    if (
        not isinstance(embedded_cad_source_sha256, str)
        or (
            declared_cad_source_sha256 is not None
            and declared_cad_source_sha256 != embedded_cad_source_sha256
        )
        or not isinstance(segmentation_sha256, str)
    ):
        return None
    return _fingerprint_value(
        {
            "CAD_source_sha256": embedded_cad_source_sha256,
            "segmentation_record_sha256": segmentation_sha256,
        }
    )


def _binding_matches_prerequisites(
    root: Path,
    binding: TypedContextBinding,
    prerequisites: Mapping[str, TypedContextBinding],
    *,
    required_target_frame: str | None,
) -> bool:
    """Return whether one output was produced for the selected exact inputs."""
    record = _read_json_mapping(root / binding.record_ref, binding.record_type)
    if binding.record_type == "CameraToRobotCalibrationRecord":
        pose = prerequisites.get("CADPoseEstimationRecord")
        validity = record.get("validity")
        if pose is None or not isinstance(validity, Mapping):
            return False
        valid_from_ns = validity.get("valid_from_ns")
        valid_until_ns = validity.get("valid_until_ns")
        observed_at_ns = _camera_pose_observation_timestamp(root, pose)
        return bool(
            pose.frame is not None
            and observed_at_ns is not None
            and record.get("source_frame") == pose.frame
            and (
                required_target_frame is None
                or record.get("target_frame") == required_target_frame
            )
            and isinstance(valid_from_ns, int)
            and not isinstance(valid_from_ns, bool)
            and valid_from_ns <= observed_at_ns
            and (
                valid_until_ns is None
                or (
                    isinstance(valid_until_ns, int)
                    and not isinstance(valid_until_ns, bool)
                    and observed_at_ns <= valid_until_ns
                )
            )
        )
    hashed_refs = _embedded_record_hash_refs(record)
    return all(
        (prerequisite.record_ref, prerequisite.record_sha256) in hashed_refs
        for prerequisite in prerequisites.values()
    )


def _camera_pose_observation_timestamp(
    root: Path,
    pose: TypedContextBinding,
) -> int | None:
    """Resolve the capture timestamp pinned through a camera-pose source chain."""
    if pose.observed_at_ns is not None:
        return pose.observed_at_ns
    pose_record = _read_json_mapping(root / pose.record_ref, pose.record_type)
    segmentation = pose_record.get("segmentation")
    if not isinstance(segmentation, Mapping):
        return None
    segmentation_record = _read_embedded_hashed_mapping(
        root,
        segmentation.get("record"),
    )
    if segmentation_record is None:
        return None
    preprocessing_record = _read_embedded_hashed_mapping(
        root,
        segmentation_record.get("source_record"),
    )
    if preprocessing_record is None:
        return None
    manifest = _read_embedded_hashed_mapping(
        root,
        preprocessing_record.get("source_manifest"),
    )
    if manifest is None:
        return None
    captured_at_ns = manifest.get("captured_at_ns")
    if isinstance(captured_at_ns, bool) or not isinstance(captured_at_ns, int):
        return None
    return captured_at_ns


def _read_embedded_hashed_mapping(
    root: Path,
    value: object,
) -> Mapping[str, object] | None:
    """Read one in-interaction mapping only when its embedded hash is intact."""
    if not isinstance(value, Mapping):
        return None
    ref = value.get("ref")
    sha256 = value.get("sha256")
    if not isinstance(ref, str) or not isinstance(sha256, str):
        return None
    path = (root / ref).resolve()
    try:
        path.relative_to(root.resolve())
        source = path.read_bytes()
        record = json.loads(source.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if hashlib.sha256(source).hexdigest() != sha256 or not isinstance(record, Mapping):
        return None
    return record


def _embedded_record_hash_refs(value: object) -> set[tuple[str, str]]:
    """Collect complete nested local record references without interpreting labels."""
    result: set[tuple[str, str]] = set()
    if isinstance(value, Mapping):
        ref = value.get("ref")
        sha256 = value.get("sha256")
        if isinstance(ref, str) and isinstance(sha256, str):
            result.add((ref, sha256))
        for item in value.values():
            result.update(_embedded_record_hash_refs(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_embedded_record_hash_refs(item))
    return result


def _newest_binding(
    bindings: Sequence[TypedContextBinding],
) -> TypedContextBinding:
    """Return the deterministic newest binding for one exact record type."""
    return max(
        bindings,
        key=lambda binding: (
            -1 if binding.observed_at_ns is None else binding.observed_at_ns,
            binding.record_ref,
        ),
    )


def _selected_cad_binding(
    root: Path,
    bindings: Sequence[TypedContextBinding],
    *,
    source_ref: str,
    source_revision: str,
    record_refs: Sequence[str],
    segmentation: TypedContextBinding | None,
) -> TypedContextBinding | None:
    """Resolve the exact CAD source represented by one PA-selected action."""
    candidates = [
        binding
        for binding in bindings
        if _cad_source_matches(
            root,
            binding,
            source_ref=source_ref,
            source_revision=None,
        )
    ]
    exact_records = [
        binding for binding in candidates if binding.record_ref in record_refs
    ]
    if exact_records:
        return _newest_binding(exact_records)

    selected_source_sha256 = _selected_cad_source_sha256(
        root,
        record_refs,
        source_ref=source_ref,
    )
    matches = [
        binding
        for binding in candidates
        if (
            selected_source_sha256 is not None
            and _cad_source_sha256(root, binding) == selected_source_sha256
        )
        or _cad_source_sha256(root, binding) == source_revision
        or (
            segmentation is not None
            and _cad_segmentation_revision(root, binding, segmentation)
            == source_revision
        )
    ]
    return None if not matches else _newest_binding(matches)


def _selected_cad_source_sha256(
    root: Path,
    record_refs: Sequence[str],
    *,
    source_ref: str,
) -> str | None:
    """Recover a selected CAD source hash from its accepted correspondence."""
    for record_ref in record_refs:
        try:
            record = _read_json_mapping(root / record_ref, "selected CAD lineage")
        except ProductionGroundingError:
            continue
        if record.get("record_type") != "CADSizeCorrespondenceRecord":
            continue
        cad = record.get("CAD")
        if not isinstance(cad, Mapping):
            continue
        cad_record = _read_embedded_hashed_mapping(root, cad.get("record"))
        cad_source = cad_record.get("source") if cad_record is not None else None
        if not isinstance(cad_source, Mapping):
            continue
        embedded_context_ref = cad_source.get("context_ref")
        embedded_source_sha256 = cad_source.get("source_sha256")
        declared_context_ref = cad.get("context_ref")
        declared_source_sha256 = cad.get("source_sha256")
        if (
            embedded_context_ref != source_ref
            or not isinstance(embedded_source_sha256, str)
            or (
                declared_context_ref is not None
                and declared_context_ref != embedded_context_ref
            )
            or (
                declared_source_sha256 is not None
                and declared_source_sha256 != embedded_source_sha256
            )
        ):
            continue
        return embedded_source_sha256
    return None


def _matching_cad_binding(
    root: Path,
    bindings: Sequence[TypedContextBinding],
    *,
    source_ref: str,
    source_revision: str | None,
) -> TypedContextBinding | None:
    """Return the newest intact CAD binding for one approved source revision."""
    matches = [
        binding
        for binding in bindings
        if _cad_source_matches(
            root,
            binding,
            source_ref=source_ref,
            source_revision=source_revision,
        )
    ]
    return None if not matches else _newest_binding(matches)


def _cad_source_matches(
    root: Path,
    binding: TypedContextBinding,
    *,
    source_ref: str,
    source_revision: str | None,
) -> bool:
    """Return whether one CAD binding resolves to the exact approved source."""
    record = _read_json_mapping(root / binding.record_ref, "CADMeshRecord")
    source = record.get("source")
    return bool(
        isinstance(source, Mapping)
        and source.get("context_ref") == source_ref
        and (
            source_revision is None
            or source.get("source_sha256") == source_revision
        )
    )


def _cad_source_sha256(root: Path, binding: TypedContextBinding) -> str:
    """Return the approved source hash pinned by one intact CAD record."""
    record = _read_json_mapping(root / binding.record_ref, "CADMeshRecord")
    source = record.get("source")
    source_sha256 = (
        source.get("source_sha256") if isinstance(source, Mapping) else None
    )
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ProductionGroundingError("CADMeshRecord has no approved source hash.")
    return source_sha256


def _cad_context_ref(root: Path, binding: TypedContextBinding) -> str:
    """Return the exact approved context ref pinned by one CAD binding."""
    record = _read_json_mapping(root / binding.record_ref, "CADMeshRecord")
    source = record.get("source")
    context_ref = source.get("context_ref") if isinstance(source, Mapping) else None
    if not isinstance(context_ref, str) or not context_ref:
        raise ProductionGroundingError("CADMeshRecord has no approved context ref.")
    return context_ref


def _fresh_required_pose_binding(
    root: Path,
    view: ProductContextView,
    need: ResourceAssignmentNeed,
) -> TypedContextBinding | None:
    """Return the newest intact accepted required pose for the current scene."""
    relevant = [
        binding
        for binding in view.typed_bindings
        if binding.record_type
        in {
            need.required_record_type,
            "ColoredPointCloudSetRecord",
            "RGBDSegmentationRecord",
        }
        and binding.observed_at_ns is not None
    ]
    newest_observation = max(
        (int(binding.observed_at_ns) for binding in relevant),
        default=None,
    )
    candidates = [
        binding
        for binding in view.typed_bindings
        if binding.record_type == need.required_record_type
        and binding.status == "accepted"
        and binding.frame == need.target_frame
        and binding.observed_at_ns is not None
        and _binding_file_is_intact(root, binding)
    ]
    if not candidates:
        return None
    selected = max(
        candidates,
        key=lambda binding: (int(binding.observed_at_ns or -1), binding.record_ref),
    )
    if (
        newest_observation is not None
        and selected.observed_at_ns != newest_observation
    ):
        return None
    return selected


def _exact_action_bindings(
    root: Path,
    view: ProductContextView,
    action: _EligibleProviderAction,
    record_types: Sequence[str],
) -> dict[str, TypedContextBinding]:
    """Validate and return the exact bindings pinned to one eligible action."""
    bindings = dict(action.prerequisite_bindings)
    if set(bindings) != set(record_types):
        raise ProductionGroundingError(
            "Derived provider action does not pin its exact prerequisites."
        )
    current = {
        (binding.record_type, binding.record_ref, binding.record_sha256)
        for binding in view.typed_bindings
    }
    for record_type in record_types:
        binding = bindings[record_type]
        if (
            binding.record_type != record_type
            or (
                binding.record_type,
                binding.record_ref,
                binding.record_sha256,
            )
            not in current
            or not _binding_file_is_intact(root, binding)
        ):
            raise ProductionGroundingError(
                "Derived provider action prerequisites changed before execution."
            )
    return bindings


def _binding_file_is_intact(root: Path, binding: TypedContextBinding) -> bool:
    """Return whether a typed binding still resolves to its pinned source hash."""
    path = (root / binding.record_ref).resolve()
    try:
        path.relative_to(root.resolve())
        source = path.read_bytes()
    except (OSError, ValueError):
        return False
    return hashlib.sha256(source).hexdigest() == binding.record_sha256


def _observation_frontier_revision(
    root: Path,
    view: ProductContextView,
) -> str:
    """Fingerprint decision progress without record IDs or capture timestamps."""
    cad_sources: set[str] = set()
    outcomes: set[str] = set()
    for binding in view.typed_bindings:
        if not _binding_file_is_intact(root, binding):
            continue
        record = _read_json_mapping(root / binding.record_ref, binding.record_type)
        if binding.record_type == "CADMeshRecord":
            source = record.get("source")
            if isinstance(source, Mapping):
                cad_sources.add(
                    json.dumps(
                        {
                            "context_ref": source.get("context_ref"),
                            "source_sha256": source.get("source_sha256"),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            continue
        if binding.record_type != "CADSizeCorrespondenceRecord":
            continue
        cad = record.get("CAD")
        ranked = record.get("ranked_candidates")
        plausible = record.get("plausible_candidates")
        best = ranked[0] if isinstance(ranked, list) and ranked else None
        signature = {
            "CAD_source_sha256": (
                cad.get("source_sha256") if isinstance(cad, Mapping) else None
            ),
            "verdict": record.get("CAD_correspondence"),
            "selected": _candidate_decision_identity(
                record.get("selected_candidate")
            ),
            "plausible": sorted(
                (
                    _candidate_decision_identity(item)
                    for item in plausible
                    if isinstance(item, Mapping)
                ),
                key=lambda item: json.dumps(item, sort_keys=True),
            )
            if isinstance(plausible, list)
            else [],
            "best": _candidate_decision_identity(best),
        }
        outcomes.add(
            json.dumps(signature, sort_keys=True, separators=(",", ":"))
        )
    return _fingerprint_value(
        {
            "CAD_sources": sorted(cad_sources),
            "CAD_decision_outcomes": sorted(outcomes),
        }
    )


def _candidate_decision_identity(value: object) -> dict[str, object] | None:
    """Return only fields that can change a correspondence decision branch."""
    if not isinstance(value, Mapping):
        return None
    return {
        "camera_id": value.get("camera_id"),
        "role": value.get("role"),
        "candidate_id": value.get("candidate_id"),
        "within_size_tolerance": value.get("within_size_tolerance"),
    }


async def _author_grounding_session(  # noqa: PLR0913
    product_agent: ProductAgentContextRuntime,
    *,
    tbox: TBoxSnapshot,
    requirement_text: str,
    previous: GroundingSession | None,
    pending_attempt: GroundingActionAttempt | None,
    previews: Sequence[Mapping[str, object]],
    actions: Sequence[_EligibleProviderAction],
    turn_number: int,
    max_pa_turns: int,
    allow_proposal: bool = True,
) -> GroundingSession:
    """Ask PA for one minimal semantic action and resolve it in the host."""
    revision = 1 if previous is None else previous.revision + 1
    attempts = () if previous is None else previous.attempted_actions
    if pending_attempt is not None:
        attempts = (*attempts, pending_attempt)
    if revision >= max_pa_turns:
        return _incomplete_session(
            revision=revision,
            requirement_text=requirement_text,
            attempted_actions=attempts,
            reason="The emergency PA call ceiling was reached.",
        )

    prompt_value = {
        "exact_requirement": requirement_text,
        "ontology": {
            "tbox_fingerprint": tbox.fingerprint,
            "classes": sorted(tbox.classes),
            "object_properties": sorted(tbox.object_properties),
            "datatype_properties": sorted(tbox.datatype_properties),
        },
        "available_sources_and_retrieved_records": [dict(item) for item in previews],
        "previous_action_attempts": [item.to_record() for item in attempts],
        "available_actions": _provider_action_prompt_records(
            actions=actions,
            attempted_actions=attempts,
        ),
        "operational_turn": turn_number,
        "emergency_max_pa_turns": max_pa_turns,
    }
    proposal_guidance = (
        "Use propose_grounding when the available evidence supports the "
        "ontology-level meaning. "
        if allow_proposal
        else "The provisional semantic ABox already exists; do not propose it again. "
    )
    base_prompt = (
        "Understand the exact requirement and ontology together, then use all "
        "currently retrieved evidence. Return exactly one next semantic action. "
        "Use retrieve for an available document, CAD file, observation, or derived "
        "record. Use inspect only for a prepared document when a focused follow-up "
        "over the complete ordered document is useful. "
        f"{proposal_guidance}"
        "Use ask_user only "
        "when user intent is needed. Use incomplete when no safe progress is possible. "
        "Do not return analysis, IDs, provider metadata, source revisions, citations, "
        "information needs, statements, transitions, or ontology facts in this step.\n\n"
        f"Grounding input:\n{json.dumps(prompt_value, indent=2, ensure_ascii=False)}"
    )

    validation_failure = "unknown action validation failure"
    response_format = _grounding_action_response_format(
        actions,
        allow_proposal=allow_proposal,
    )
    for response_number in range(2):
        prompt = base_prompt
        if response_number:
            prompt += (
                "\n\nThe prior response was rejected. Return one corrected action "
                f"only. Validation error: {validation_failure}"
            )
        output = await product_agent.ask_llm_structured(
            prompt,
            response_format=response_format,
        )
        try:
            if not isinstance(output, Mapping):
                raise GroundingContractError(
                    "ProductAgent action response must be an object."
                )
            if set(output) != {"next_action"}:
                raise GroundingContractError(
                    "ProductAgent action response must contain only next_action."
                )
            action_value = output["next_action"]
            if not isinstance(action_value, Mapping):
                raise GroundingContractError(
                    "ProductAgent next_action must be an object."
                )
            next_action = GroundingNextAction.from_mapping(action_value)
            if next_action.action == "propose_grounding" and not allow_proposal:
                raise GroundingContractError(
                    "The semantic task ABox is already provisional; another "
                    "propose_grounding action is not available."
                )
            selected = _resolve_provider_action(
                next_action=next_action,
                actions=actions,
                attempted_actions=attempts,
            )
        except (GroundingContractError, ProductionGroundingError) as exc:
            validation_failure = str(exc)
            continue

        if next_action.action == "propose_grounding":
            status = "ready_for_ontology"
        elif next_action.action == "incomplete":
            status = "incomplete"
        elif next_action.action == "ask_user":
            status = "waiting_for_user"
        else:
            status = "waiting_for_evidence"
        return GroundingSession.create(
            revision=revision,
            requirement_text=requirement_text,
            attempted_actions=attempts,
            next_action=next_action,
            selected_provider_id=None if selected is None else selected.provider_id,
            selected_source_revision=(
                None if selected is None else selected.source_revision
            ),
            status=status,
        )

    return _incomplete_session(
        revision=revision,
        requirement_text=requirement_text,
        attempted_actions=attempts,
        reason=(
            "ProductAgent could not return a valid next action after one repair: "
            f"{validation_failure}"
        ),
    )


def _provider_action_prompt_records(
    *,
    actions: Sequence[_EligibleProviderAction],
    attempted_actions: Sequence[GroundingActionAttempt],
) -> list[dict[str, object]]:
    """Expose available source operations without controller IDs in PA output."""
    attempted_keys = {item.action_key for item in attempted_actions}
    records: list[dict[str, object]] = []
    for action in actions:
        if action.automatic:
            continue
        key = (
            action.action,
            action.provider_id,
            action.source_ref,
            action.source_revision,
            None,
        )
        if action.action == "retrieve" and key in attempted_keys:
            continue
        record = action.to_record()
        record.pop("provider_id")
        record.pop("source_revision")
        records.append(record)
    return records


def _resolve_provider_action(
    *,
    next_action: GroundingNextAction,
    actions: Sequence[_EligibleProviderAction],
    attempted_actions: Sequence[GroundingActionAttempt],
) -> _EligibleProviderAction | None:
    """Resolve a minimal PA action and reject unavailable or exact replays."""
    if next_action.action in {"propose_grounding", "incomplete"}:
        return None
    matches = [
        item
        for item in actions
        if item.action == next_action.action
        and (
            next_action.action == "ask_user"
            or item.source_ref == next_action.source_ref
        )
    ]
    if len(matches) != 1:
        raise ProductionGroundingError(
            "The selected action does not identify one available source operation."
        )
    selected = matches[0]
    key = (
        next_action.action,
        selected.provider_id,
        selected.source_ref,
        selected.source_revision,
        next_action.question,
    )
    if key in {item.action_key for item in attempted_actions}:
        raise ProductionGroundingError(
            "The selected action exactly repeats an attempted source operation."
        )
    return selected


def _grounding_action_response_format(
    actions: Sequence[_EligibleProviderAction],
    *,
    allow_proposal: bool = True,
) -> dict[str, Any]:
    """Return the strict object-rooted five-variant semantic action schema."""
    variants: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for item in actions:
        if item.automatic:
            continue
        key = (item.action, item.source_ref)
        if key in seen or item.action == "ask_user":
            continue
        seen.add(key)
        properties: dict[str, object] = {
            "action": {"type": "string", "const": item.action},
            "source_ref": {"type": "string", "const": item.source_ref},
        }
        required = ["action", "source_ref"]
        if item.action == "inspect":
            properties["question"] = {"type": "string", "minLength": 1}
            required.append("question")
        variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": required,
                "properties": properties,
            }
        )
    if any(item.action == "ask_user" for item in actions):
        variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "question"],
                "properties": {
                    "action": {"type": "string", "const": "ask_user"},
                    "question": {"type": "string", "minLength": 1},
                },
            }
        )
    if allow_proposal:
        variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["action"],
                "properties": {
                    "action": {"type": "string", "const": "propose_grounding"}
                },
            }
        )
    variants.append(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "reason"],
            "properties": {
                "action": {"type": "string", "const": "incomplete"},
                "reason": {"type": "string", "minLength": 1},
            },
        }
    )
    return {
        "name": "spec2primitives_grounding_action",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["next_action"],
            "properties": {
                "next_action": {"anyOf": variants},
            },
        },
    }


def _incomplete_session(
    *,
    revision: int,
    requirement_text: str,
    attempted_actions: Sequence[GroundingActionAttempt],
    reason: str,
    status: str = "incomplete",
) -> GroundingSession:
    """Create a conservative terminal session without model bookkeeping."""
    return GroundingSession.create(
        revision=revision,
        requirement_text=requirement_text,
        attempted_actions=attempted_actions,
        next_action=GroundingNextAction.from_mapping(
            {"action": "incomplete", "reason": reason}
        ),
        status=status,
    )


def _resource_assignment_complete_session(
    previous: GroundingSession | None,
    *,
    requirement_text: str,
    pending_attempt: GroundingActionAttempt | None,
) -> GroundingSession:
    """Create the host terminal session only after assignment was committed."""
    if previous is None:
        raise ProductionGroundingError(
            "Resource assignment cannot complete without its grounding session."
        )
    attempts = previous.attempted_actions
    if pending_attempt is not None:
        attempts = (*attempts, pending_attempt)
    return GroundingSession.create(
        revision=previous.revision + 1,
        requirement_text=requirement_text,
        attempted_actions=attempts,
        next_action=GroundingNextAction.from_mapping(
            {"action": "propose_grounding"}
        ),
        status="complete",
    )


def _resource_assignment_incomplete_session(
    previous: GroundingSession | None,
    *,
    requirement_text: str,
    pending_attempt: GroundingActionAttempt | None,
    reason: str,
) -> GroundingSession:
    """Create a fail-closed terminal result when no candidate is reachable."""
    attempts = () if previous is None else previous.attempted_actions
    revision = 1 if previous is None else previous.revision + 1
    if pending_attempt is not None:
        attempts = (*attempts, pending_attempt)
    return _incomplete_session(
        revision=revision,
        requirement_text=requirement_text,
        attempted_actions=attempts,
        reason=reason,
    )


def _emergency_incomplete_session(
    previous: GroundingSession,
    pending_attempt: GroundingActionAttempt | None,
) -> GroundingSession:
    """Persist an incomplete state when the operational call ceiling is reached."""
    attempts = previous.attempted_actions
    if pending_attempt is not None:
        attempts = (*attempts, pending_attempt)
    return _incomplete_session(
        revision=previous.revision + 1,
        requirement_text=previous.requirement_text,
        attempted_actions=attempts,
        reason="The emergency PA call ceiling was reached.",
    )


def _session_assessment(
    session: GroundingSession,
    actions: Sequence[_EligibleProviderAction],
) -> dict[str, object]:
    """Translate a session decision to the existing public needed_context shape."""
    next_action = session.next_action
    if next_action.action == "propose_grounding":
        if session.status != "complete":
            raise ProductionGroundingError(
                "A provisional propose_grounding session cannot escape as completion."
            )
        return {
            "unresolved_semantic_need": None,
            "needed_context": None,
            "context understanding complete": True,
            "grounding_status": session.status,
        }
    if next_action.action == "incomplete":
        return {
            "unresolved_semantic_need": None,
            "needed_context": None,
            "context understanding complete": False,
            "grounding_status": session.status,
        }
    action = next(
        item
        for item in actions
        if item.action == next_action.action
        and item.provider_id == session.selected_provider_id
        and item.source_revision == session.selected_source_revision
        and (
            next_action.action == "ask_user"
            or item.source_ref == next_action.source_ref
        )
    )
    produced = action.produced_record_types[0]
    if action.evidence_type == "user":
        needed_context = {
            "context_ref": None,
            "request_live_observation": False,
            "clarification_question": next_action.question,
        }
        need_kind = "user_intent"
        symbol = "product_requirement"
        description = next_action.question or "Clarify the product requirement."
    elif action.evidence_type == "observation":
        needed_context = {
            "context_ref": None,
            "request_live_observation": True,
            "clarification_question": None,
        }
        need_kind = "typed_context_record"
        symbol = produced
        description = (
            next_action.question
            or "Retrieve a fresh observation contributing to "
            f"{produced}; evidence revision {action.source_revision}."
        )
    elif action.evidence_type in {"document", "CAD"}:
        needed_context = {
            "context_ref": action.source_ref,
            "request_live_observation": False,
            "clarification_question": None,
        }
        need_kind = "typed_context_record"
        symbol = produced
        description = (
            next_action.question
            or f"Retrieve available context from {action.source_ref}."
        )
    else:
        raise ProductionGroundingError(
            "Selected provider action must execute inside the grounding runtime."
        )
    return {
        "unresolved_semantic_need": {
            "kind": need_kind,
            "symbol": symbol,
            "description": description,
        },
        "needed_context": needed_context,
        "context understanding complete": False,
        "grounding_status": session.status,
    }


def _selected_session_action(
    session: GroundingSession,
    actions: Sequence[_EligibleProviderAction],
) -> _EligibleProviderAction | None:
    if session.next_action.action not in {"retrieve", "inspect", "ask_user"}:
        return None
    return next(
        (
            item
            for item in actions
            if item.action == session.next_action.action
            and item.provider_id == session.selected_provider_id
            and item.source_revision == session.selected_source_revision
            and (
                session.next_action.action == "ask_user"
                or item.source_ref == session.next_action.source_ref
            )
        ),
        None,
    )


def _record_source_ids(record: Mapping[str, object]) -> set[str]:
    """Collect explicit source IDs from a validated typed evidence record."""
    result: set[str] = set()
    context_ref = record.get("context_ref")
    if isinstance(context_ref, str):
        result.add(context_ref)
    evidence_refs = record.get("evidence_refs")
    if isinstance(evidence_refs, list):
        result.update(item for item in evidence_refs if isinstance(item, str))
    for field in ("observations", "uncertainty"):
        items = record.get(field)
        if not isinstance(items, list):
            continue
        for item in items:
            refs = item.get("evidence_refs") if isinstance(item, Mapping) else None
            if isinstance(refs, list):
                result.update(ref for ref in refs if isinstance(ref, str))
    overview = record.get("overview")
    if isinstance(overview, Mapping):
        result.update(_record_source_ids(overview))
    return result


def _fingerprint_value(value: object) -> str:
    """Return a stable fingerprint for action revisions and preview state."""
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _read_json_mapping(path: Path, label: str) -> Mapping[str, object]:
    """Read one required JSON object without accepting malformed records."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionGroundingError(f"{label} could not be read.") from exc
    if not isinstance(value, Mapping):
        raise ProductionGroundingError(f"{label} must be a JSON object.")
    return value


def _producer_descriptors(
    tbox: TBoxSnapshot,
    *,
    calibration_available: bool = False,
) -> tuple[GroundingProducerDescriptor, ...]:
    del tbox
    records = [
        {
            "provider_id": _DOCUMENT_PRODUCER,
            "description": (
                "Read an approved PDF overview or inspect selected document pages."
            ),
            "accepted_evidence_types": ["document"],
            "produced_record_types": [
                "DocumentOverviewRecord",
                "DocumentEvidenceRecord",
            ],
            "prerequisites": {
                "DocumentOverviewRecord": [],
                "DocumentEvidenceRecord": ["DocumentOverviewRecord"],
            },
            "availability": True,
            "estimated_cost": 1,
        },
        {
            "provider_id": _GEOMETRY_PRODUCER,
            "description": (
                "Create typed CAD, RGB-D, correspondence, or camera-pose records."
            ),
            "accepted_evidence_types": ["CAD", "observation", "existing_record"],
            "produced_record_types": [
                "CADMeshRecord",
                "ColoredPointCloudSetRecord",
                "RGBDSegmentationRecord",
                "CADSizeCorrespondenceRecord",
                "CADPoseEstimationRecord",
            ],
            "prerequisites": {
                "CADMeshRecord": [],
                "ColoredPointCloudSetRecord": [],
                "RGBDSegmentationRecord": [],
                "CADSizeCorrespondenceRecord": [
                    "CADMeshRecord",
                    "RGBDSegmentationRecord",
                ],
                "CADPoseEstimationRecord": ["CADSizeCorrespondenceRecord"],
            },
            "availability": True,
            "estimated_cost": 1,
        },
        {
            "provider_id": _CALIBRATION_PRODUCER,
            "description": (
                "Materialize an approved camera-to-world calibration for an "
                "accepted camera-frame pose."
            ),
            "accepted_evidence_types": ["existing_record"],
            "produced_record_types": ["CameraToRobotCalibrationRecord"],
            "prerequisites": {
                "CameraToRobotCalibrationRecord": ["CADPoseEstimationRecord"]
            },
            "availability": calibration_available,
            "estimated_cost": 0,
        },
        {
            "provider_id": _FRAME_CONVERSION_PRODUCER,
            "description": (
                "Convert an accepted camera-frame CAD pose through an approved "
                "calibration into the required world frame."
            ),
            "accepted_evidence_types": ["existing_record"],
            "produced_record_types": ["RobotFramePoseRecord"],
            "prerequisites": {
                "RobotFramePoseRecord": [
                    "CADPoseEstimationRecord",
                    "CameraToRobotCalibrationRecord",
                ]
            },
            "availability": True,
            "estimated_cost": 0,
        },
        {
            "provider_id": "user_clarification",
            "description": "Ask the user for intent that approved evidence cannot establish.",
            "accepted_evidence_types": ["user"],
            "produced_record_types": ["PAClarification"],
            "prerequisites": {"PAClarification": []},
            "availability": True,
            "estimated_cost": 1,
        },
    ]
    return tuple(GroundingProducerDescriptor.from_mapping(record) for record in records)


def _selected_output_for_served_context(
    root: Path,
    served_context: Mapping[str, object],
) -> str | None:
    del root
    return {
        "CAD": "CADMeshRecord",
        "observation": "RGBDSegmentationRecord",
    }.get(str(served_context.get("evidence_type")))


def _next_number(root: Path, pattern: str) -> int:
    return len(tuple(root.glob(pattern))) + 1


def _relative_ref(root: Path, path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ProductionGroundingError("Grounding record left its interaction.") from exc


def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProductionGroundingError(f"{label} must be an object.")
    return value

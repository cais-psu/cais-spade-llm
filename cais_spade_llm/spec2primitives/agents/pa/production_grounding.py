"""Run generalized evidence-first PA grounding through authorized providers."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingActionAttempt,
    GroundingContractError,
    GroundingDecision,
    GroundingProducerDescriptor,
    GroundingSession,
    GroundingStatement,
    InformationNeed,
    ProductContextView,
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
from cais_spade_llm.spec2primitives.config import DocumentVLMConfig
from cais_spade_llm.spec2primitives.ontology import TBoxSnapshot
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
    associate_segmented_candidate_by_size,
    estimate_camera_frame_pose,
    preprocess_served_geometry,
    segment_preprocessed_observation,
)

_DOCUMENT_PRODUCER = "document_evidence"
_GEOMETRY_PRODUCER = "rgb_d_cad_grounding"
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

    provider_id: str
    source_ref: str
    source_revision: str
    evidence_type: str
    produced_record_types: tuple[str, ...]
    description: str
    estimated_cost: int

    def to_record(self) -> dict[str, object]:
        """Return the ontology-neutral action record exposed to PA."""
        return {
            "provider_id": self.provider_id,
            "source_ref": self.source_ref,
            "source_revision": self.source_revision,
            "evidence_type": self.evidence_type,
            "produced_record_types": list(self.produced_record_types),
            "description": self.description,
            "estimated_cost": self.estimated_cost,
        }


class ProductionGroundingError(RuntimeError):
    """Raised when production grounding cannot make safe progress."""


class ProductionProductContextGroundingRuntime:
    """Resolve pre-RA product context with real controlled tools."""

    def __init__(
        self,
        *,
        tbox: TBoxSnapshot,
        document_config: DocumentVLMConfig,
        document_vision_runtime: DocumentVisionRuntime,
    ) -> None:
        """Create a runtime pinned to one authoritative TBox snapshot."""
        tbox.assert_unchanged()
        self._tbox = tbox
        self._document_config = document_config
        self._document_vision_runtime = document_vision_runtime
        self._descriptors = _producer_descriptors(tbox)

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
            if _session_requests_record_type(session, "DocumentEvidenceRecord"):
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
                    evidence_question=str(session.decision.query),
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

        preprocessing = preprocess_served_geometry(
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
        segmentation = segment_preprocessed_observation(
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

    async def _assess(  # noqa: PLR0913
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
        previews, authorized_sources = _collect_provider_previews(
            root,
            view,
            clarification_history=clarification_history,
            config=self._document_config,
        )
        pending_attempt, newly_accepted_sources = _pending_action_attempt(
            previous,
            view,
            clarification_history=clarification_history,
        )
        actions = _discover_provider_actions(
            root,
            self._descriptors,
            previews=previews,
            view=view,
            previous=previous,
            pending_attempt=pending_attempt,
        )
        if previous is not None and previous.revision >= max_pa_turns:
            session = _emergency_incomplete_session(previous, pending_attempt)
        else:
            session = await _author_grounding_session(
                product_agent,
                requirement_text=view.product_requirement,
                previous=previous,
                pending_attempt=pending_attempt,
                previews=previews,
                authorized_sources=authorized_sources,
                newly_accepted_sources=newly_accepted_sources,
                actions=actions,
                turn_number=turn_number,
                max_pa_turns=max_pa_turns,
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
                    session=session,
                )
            except OntologyGroundingError as exc:
                gap_decision = GroundingDecision.from_mapping(
                    {
                        "decision_type": "incomplete",
                        "need_id": None,
                        "provider_id": None,
                        "source_ref": None,
                        "source_revision": None,
                        "query": None,
                        "reason": f"Late ontology mapping was rejected: {exc}",
                    }
                )
                gap = GroundingSession.create(
                    revision=session.revision + 1,
                    requirement_text=session.requirement_text,
                    statements=session.statements,
                    information_needs=session.information_needs,
                    attempted_actions=session.attempted_actions,
                    evidence_refs=session.evidence_refs,
                    decision=gap_decision,
                    status="ontology_gap",
                    information_status=session.information_status,
                )
                persist_grounding_session(root, gap)
                return _session_assessment(gap, actions)
            final_view = build_product_context_view(
                root,
                mapping.merge.abox,
                attempted_evidence=attempted_evidence,
                assessed_at_ns=time.time_ns(),
            )
            persist_product_context_view(root, final_view)
            completed = GroundingSession.create(
                revision=session.revision + 1,
                requirement_text=session.requirement_text,
                statements=session.statements,
                information_needs=session.information_needs,
                attempted_actions=session.attempted_actions,
                evidence_refs=session.evidence_refs,
                decision=session.decision,
                status="complete",
                information_status=session.information_status,
            )
            persist_grounding_session(root, completed)
            return _session_assessment(completed, actions)
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

    def _run_session_derived_action(  # noqa: PLR0913
        self,
        *,
        root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        view: ProductContextView,
        session: GroundingSession,
        action: _EligibleProviderAction,
    ) -> ABoxSnapshot:
        """Execute one selected zero-retrieval provider action from typed records."""
        need = next(
            item
            for item in session.information_needs
            if item.need_id == session.decision.need_id
        )
        output_types = sorted(
            set(action.produced_record_types).intersection(
                need.accepted_record_types
            )
        )
        if len(output_types) != 1:
            raise ProductionGroundingError(
                "Derived provider action does not select one exact output record."
            )
        output_type = output_types[0]
        bindings = {
            binding.record_type: binding
            for binding in view.typed_bindings
            if binding.status == "accepted"
        }
        if output_type == "CADSizeCorrespondenceRecord":
            required = {"CADMeshRecord", "RGBDSegmentationRecord"}
            if not required.issubset(bindings):
                raise ProductionGroundingError(
                    "CAD correspondence prerequisites changed before execution."
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
            required = {"CADSizeCorrespondenceRecord"}
            if not required.issubset(bindings):
                raise ProductionGroundingError(
                    "CAD pose prerequisites changed before execution."
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
        else:
            raise ProductionGroundingError(
                f"No internal provider action produces {output_type}."
            )
        record_ref = _relative_ref(root, record_path)
        evidence_refs = sorted(
            {
                evidence_ref
                for record_type in required
                for evidence_ref in bindings[record_type].evidence_refs
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
            action.provider_id,
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
        authorized_sources.add(context_ref)

    accepted_bindings = [
        binding for binding in view.typed_bindings if binding.status == "accepted"
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

    observation_count = sum(
        binding.record_type in {"ColoredPointCloudSetRecord", "RGBDSegmentationRecord"}
        for binding in accepted_bindings
    )
    previews.append(
        {
            "provider_id": _GEOMETRY_PRODUCER,
            "evidence_type": "observation",
            "source_ref": "live_RGB-D",
            "source_revision": _fingerprint_value(
                {"accepted_observation_records": observation_count}
            ),
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
) -> tuple[GroundingActionAttempt | None, set[str]]:
    """Describe the controller-owned result of the previously selected action."""
    if previous is None or previous.decision.decision_type != "request_evidence":
        return None, set()
    decision = previous.decision
    previous_refs = set(previous.evidence_refs)
    record_refs: list[str] = []
    accepted_sources: set[str] = set()
    if decision.provider_id == "user_clarification":
        for index, clarification in enumerate(clarification_history, start=1):
            source_id = f"clarification_{index:04d}"
            if source_id not in previous_refs:
                record_refs.append(source_id)
                accepted_sources.add(source_id)
                fingerprint = clarification.get("fingerprint")
                if isinstance(fingerprint, str):
                    accepted_sources.add(fingerprint)
    else:
        for binding in view.typed_bindings:
            if (
                binding.status == "accepted"
                and binding.producer == decision.provider_id
                and binding.record_ref not in previous_refs
            ):
                record_refs.append(binding.record_ref)
                accepted_sources.add(binding.record_ref)
                accepted_sources.update(binding.evidence_refs)
    attempt = GroundingActionAttempt.from_mapping(
        {
            "attempt_id": f"attempt_{len(previous.attempted_actions) + 1:04d}",
            "need_id": decision.need_id,
            "provider_id": decision.provider_id,
            "source_ref": decision.source_ref,
            "source_revision": decision.source_revision,
            "status": "accepted" if record_refs else "no_change",
            "record_refs": sorted(record_refs),
        }
    )
    return attempt, accepted_sources


def _discover_provider_actions(
    root: Path,
    descriptors: Sequence[GroundingProducerDescriptor],
    *,
    previews: Sequence[Mapping[str, object]],
    view: ProductContextView,
    previous: GroundingSession | None,
    pending_attempt: GroundingActionAttempt | None,
) -> tuple[_EligibleProviderAction, ...]:
    """Build actions from provider-owned capabilities and current prerequisites."""
    del root, previous, pending_attempt
    descriptor_by_id = {item.provider_id: item for item in descriptors}
    accepted = {
        binding.record_type: binding.record_sha256
        for binding in view.typed_bindings
        if binding.status == "accepted"
    }
    actions: list[_EligibleProviderAction] = []
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
                prerequisites = descriptor.prerequisites_for(record_type)
                if not prerequisites or not set(prerequisites).issubset(accepted):
                    continue
                actions.append(
                    _EligibleProviderAction(
                        provider_id=descriptor.provider_id,
                        source_ref="accepted_typed_records",
                        source_revision=_fingerprint_value(
                            {
                                key: accepted[key]
                                for key in sorted(prerequisites)
                            }
                        ),
                        evidence_type="existing_record",
                        produced_record_types=(record_type,),
                        description=descriptor.description,
                        estimated_cost=descriptor.estimated_cost,
                    )
                )
    clarification = descriptor_by_id.get("user_clarification")
    if clarification is not None and clarification.availability:
        actions.append(
            _EligibleProviderAction(
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


async def _author_grounding_session(  # noqa: PLR0913
    product_agent: ProductAgentContextRuntime,
    *,
    requirement_text: str,
    previous: GroundingSession | None,
    pending_attempt: GroundingActionAttempt | None,
    previews: Sequence[Mapping[str, object]],
    authorized_sources: set[str],
    newly_accepted_sources: set[str],
    actions: Sequence[_EligibleProviderAction],
    turn_number: int,
    max_pa_turns: int,
) -> GroundingSession:
    """Ask PA for one ontology-neutral update and validate it deterministically."""
    revision = 1 if previous is None else previous.revision + 1
    prompt_value = {
        "exact_requirement": requirement_text,
        "requirement_source_id": "requirement_0001",
        "prior_session": None if previous is None else previous.to_record(),
        "last_action_result": (
            None if pending_attempt is None else pending_attempt.to_record()
        ),
        "authorized_provider_previews": [dict(item) for item in previews],
        "eligible_provider_actions": [item.to_record() for item in actions],
        "allowed_source_ids": sorted(authorized_sources),
        "session_revision": revision,
        "operational_turn": turn_number,
        "emergency_max_pa_turns": max_pa_turns,
    }
    prompt = (
        "Update one generalized product-grounding session from the exact requirement "
        "and authorized evidence shown below. First state what is directly supported, "
        "then distinguish any useful inference, then identify only information that is "
        "actually required to understand the requested outcome. Cite every statement "
        "and information need using allowed_source_ids. Preserve prior IDs and records "
        "exactly. Select at most one exact eligible provider action. If no untried "
        "eligible action can resolve required missing information, return incomplete. "
        "Do not create ontology entities or relations, use an ontology vocabulary, "
        "select resources or primitives, or assume simulator truth. Source-derived "
        "wording is allowed only when its cited source contains it.\n\n"
        f"Grounding input:\n{json.dumps(prompt_value, indent=2, ensure_ascii=False)}"
    )
    output = await product_agent.ask_llm_structured(
        prompt,
        response_format=_grounding_session_response_format(
            authorized_sources=sorted(authorized_sources),
            actions=actions,
        ),
    )
    if not isinstance(output, Mapping) or set(output) != {"grounding_update"}:
        raise ProductionGroundingError(
            "ProductAgent returned an invalid GroundingSession envelope."
        )
    update = _required_mapping(output["grounding_update"], "grounding_update")
    if set(update) != {
        "statements",
        "information_needs",
        "decision",
        "information_status",
    }:
        raise ProductionGroundingError("GroundingSession update fields are invalid.")
    try:
        statements = tuple(
            GroundingStatement.from_mapping(
                _required_mapping(item, "grounding statement")
            )
            for item in _required_list(update["statements"], "statements")
        )
        needs = tuple(
            InformationNeed.from_mapping(
                _required_mapping(item, "information need")
            )
            for item in _required_list(
                update["information_needs"], "information_needs"
            )
        )
        decision = GroundingDecision.from_mapping(
            _required_mapping(update["decision"], "decision")
        )
    except (GroundingContractError, TypeError) as exc:
        raise ProductionGroundingError(f"GroundingSession update is invalid: {exc}") from exc
    information_status = update["information_status"]
    if information_status not in {"enough", "partial", "not_enough"}:
        raise ProductionGroundingError(
            "GroundingSession information_status is invalid."
        )
    max_reached = revision >= max_pa_turns
    if max_reached and decision.decision_type == "request_evidence":
        decision = GroundingDecision.from_mapping(
            {
                "decision_type": "incomplete",
                "need_id": None,
                "provider_id": None,
                "source_ref": None,
                "source_revision": None,
                "query": None,
                "reason": "The emergency PA call ceiling was reached.",
            }
        )
        information_status = "partial" if statements else "not_enough"
    attempts = (*(() if previous is None else previous.attempted_actions),)
    if pending_attempt is not None:
        attempts = (*attempts, pending_attempt)
    _validate_grounding_session_update(
        previous=previous,
        pending_attempt=pending_attempt,
        statements=statements,
        needs=needs,
        decision=decision,
        information_status=str(information_status),
        authorized_sources=authorized_sources,
        newly_accepted_sources=newly_accepted_sources,
        actions=actions,
        max_reached=max_reached,
    )
    if decision.decision_type == "incomplete":
        needs = tuple(
            replace(item, status="exhausted")
            if item.required and item.status == "open"
            else item
            for item in needs
        )
        status = "incomplete"
    elif decision.decision_type == "ready_for_ontology":
        status = "ready_for_ontology"
    else:
        status = (
            "waiting_for_user"
            if decision.provider_id == "user_clarification"
            else "waiting_for_evidence"
        )
    evidence_refs = {
        source for statement in statements for source in statement.sources
    }
    evidence_refs.update(
        source for need in needs for source in need.sources
    )
    evidence_refs.update(
        record_ref for attempt in attempts for record_ref in attempt.record_refs
    )
    if previous is not None:
        evidence_refs.update(previous.evidence_refs)
    return GroundingSession.create(
        revision=revision,
        requirement_text=requirement_text,
        statements=statements,
        information_needs=needs,
        attempted_actions=attempts,
        evidence_refs=sorted(evidence_refs),
        decision=decision,
        status=status,
        information_status=str(information_status),
    )


def _validate_grounding_session_update(  # noqa: C901, PLR0913
    *,
    previous: GroundingSession | None,
    pending_attempt: GroundingActionAttempt | None,
    statements: Sequence[GroundingStatement],
    needs: Sequence[InformationNeed],
    decision: GroundingDecision,
    information_status: str,
    authorized_sources: set[str],
    newly_accepted_sources: set[str],
    actions: Sequence[_EligibleProviderAction],
    max_reached: bool,
) -> None:
    """Reject uncited, rewritten, replayed, or ineligible PA session updates."""
    statement_ids = [item.statement_id for item in statements]
    need_ids = [item.need_id for item in needs]
    if len(set(statement_ids)) != len(statement_ids):
        raise ProductionGroundingError("Grounding statement IDs must be unique.")
    if len(set(need_ids)) != len(need_ids):
        raise ProductionGroundingError("Information need IDs must be unique.")
    for statement in statements:
        if not set(statement.sources).issubset(authorized_sources):
            raise ProductionGroundingError(
                f"Grounding statement cites an unauthorized source: {statement.statement_id}."
            )
    for need in needs:
        if not set(need.sources).issubset(authorized_sources):
            raise ProductionGroundingError(
                f"Information need cites an unauthorized source: {need.need_id}."
            )
    if previous is not None:
        current_statements = {item.statement_id: item for item in statements}
        for old in previous.statements:
            if current_statements.get(old.statement_id) != old:
                raise ProductionGroundingError(
                    "A GroundingSession update cannot remove or rewrite a statement."
                )
        current_needs = {item.need_id: item for item in needs}
        old_need_ids = {item.need_id for item in previous.information_needs}
        for old in previous.information_needs:
            current = current_needs.get(old.need_id)
            if current is None or (
                current.need_id,
                current.question,
                current.required,
                current.sources,
                current.accepted_record_types,
            ) != (
                old.need_id,
                old.question,
                old.required,
                old.sources,
                old.accepted_record_types,
            ):
                raise ProductionGroundingError(
                    "A GroundingSession update cannot remove or rewrite an information need."
                )
            if old.status != "open" and current != old:
                raise ProductionGroundingError(
                    "A resolved or exhausted information need is immutable."
                )
        for need in needs:
            if (
                need.need_id not in old_need_ids
                and need.required
                and not set(need.sources).intersection(newly_accepted_sources)
            ):
                raise ProductionGroundingError(
                    "A new required information need must cite newly accepted evidence."
                )
    statement_by_id = {item.statement_id: item for item in statements}
    for need in needs:
        for statement_id in need.answer_statement_ids:
            statement = statement_by_id.get(statement_id)
            if statement is None or statement.status != "directly_stated":
                raise ProductionGroundingError(
                    "Only a directly stated statement may resolve information."
                )
    attempts = [] if previous is None else list(previous.attempted_actions)
    if pending_attempt is not None:
        attempts.append(pending_attempt)
    attempted_keys = {item.action_key for item in attempts}
    open_required = [item for item in needs if item.required and item.status == "open"]
    eligible_by_need = {
        need.need_id: [
            action
            for action in actions
            if set(action.produced_record_types).intersection(
                need.accepted_record_types
            )
            and (need.need_id, action.provider_id, action.source_revision)
            not in attempted_keys
        ]
        for need in open_required
    }
    if decision.decision_type == "request_evidence":
        selected = next(
            (item for item in needs if item.need_id == decision.need_id), None
        )
        matches = [
            action
            for action in eligible_by_need.get(str(decision.need_id), [])
            if (
                action.provider_id,
                action.source_ref,
                action.source_revision,
            )
            == (
                decision.provider_id,
                decision.source_ref,
                decision.source_revision,
            )
        ]
        if selected is None or selected.status != "open" or len(matches) != 1:
            raise ProductionGroundingError(
                "GroundingDecision selected an ineligible or repeated provider action."
            )
        if max_reached:
            raise ProductionGroundingError(
                "GroundingDecision cannot request evidence at the emergency ceiling."
            )
        return
    if decision.decision_type == "ready_for_ontology":
        if information_status != "enough" or open_required:
            raise ProductionGroundingError(
                "ready_for_ontology requires enough information and no open required need."
            )
        if not any(item.status == "directly_stated" for item in statements):
            raise ProductionGroundingError(
                "ready_for_ontology requires directly supported statements."
            )
        return
    if any(eligible_by_need.values()) and not max_reached:
        raise ProductionGroundingError(
            "Grounding cannot become incomplete while an eligible action remains."
        )


def _grounding_session_response_format(
    *,
    authorized_sources: Sequence[str],
    actions: Sequence[_EligibleProviderAction],
) -> dict[str, Any]:
    """Return the strict ontology-neutral PA grounding response schema."""
    record_types = sorted(
        {
            *_TYPED_OUTPUTS,
            *(
                record_type
                for action in actions
                for record_type in action.produced_record_types
            ),
        }
    )
    source_schema: dict[str, object] = {
        "type": "string",
        "enum": list(authorized_sources),
    }
    statement = {
        "type": "object",
        "additionalProperties": False,
        "required": ["statement_id", "text", "status", "sources", "reason"],
        "properties": {
            "statement_id": {"type": "string", "minLength": 1},
            "text": {"type": "string", "minLength": 1},
            "status": {
                "type": "string",
                "enum": ["directly_stated", "inferred"],
            },
            "sources": {
                "type": "array",
                "items": source_schema,
                "minItems": 1,
            },
            "reason": {"type": "string", "minLength": 1},
        },
    }
    need = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "need_id",
            "question",
            "required",
            "sources",
            "accepted_record_types",
            "status",
            "answer_statement_ids",
        ],
        "properties": {
            "need_id": {"type": "string", "minLength": 1},
            "question": {"type": "string", "minLength": 1},
            "required": {"type": "boolean"},
            "sources": {
                "type": "array",
                "items": source_schema,
                "minItems": 1,
            },
            "accepted_record_types": {
                "type": "array",
                "items": {"type": "string", "enum": record_types},
                "minItems": 1,
            },
            "status": {
                "type": "string",
                "enum": ["open", "resolved", "exhausted"],
            },
            "answer_statement_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
    }
    action_variants = [
        _decision_response_schema(
            decision_type="request_evidence",
            provider_id=action.provider_id,
            source_ref=action.source_ref,
            source_revision=action.source_revision,
        )
        for action in actions
    ]
    action_variants.extend(
        [
            _decision_response_schema(decision_type="ready_for_ontology"),
            _decision_response_schema(decision_type="incomplete"),
        ]
    )
    return {
        "name": "spec2primitives_grounding_session_update",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["grounding_update"],
            "properties": {
                "grounding_update": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "statements",
                        "information_needs",
                        "decision",
                        "information_status",
                    ],
                    "properties": {
                        "statements": {"type": "array", "items": statement},
                        "information_needs": {"type": "array", "items": need},
                        "decision": {"anyOf": action_variants},
                        "information_status": {
                            "type": "string",
                            "enum": ["enough", "partial", "not_enough"],
                        },
                    },
                }
            },
        },
    }


def _decision_response_schema(
    *,
    decision_type: str,
    provider_id: str | None = None,
    source_ref: str | None = None,
    source_revision: str | None = None,
) -> dict[str, object]:
    request = decision_type == "request_evidence"
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "decision_type",
            "need_id",
            "provider_id",
            "source_ref",
            "source_revision",
            "query",
            "reason",
        ],
        "properties": {
            "decision_type": {"type": "string", "const": decision_type},
            "need_id": (
                {"type": "string", "minLength": 1}
                if request
                else {"type": "null"}
            ),
            "provider_id": (
                {"type": "string", "const": provider_id}
                if request
                else {"type": "null"}
            ),
            "source_ref": (
                {"type": "string", "const": source_ref}
                if request
                else {"type": "null"}
            ),
            "source_revision": (
                {"type": "string", "const": source_revision}
                if request
                else {"type": "null"}
            ),
            "query": (
                {"type": "string", "minLength": 1}
                if request
                else {"type": "null"}
            ),
            "reason": {"type": "string", "minLength": 1},
        },
    }


def _emergency_incomplete_session(
    previous: GroundingSession,
    pending_attempt: GroundingActionAttempt | None,
) -> GroundingSession:
    """Persist an incomplete state when the operational call ceiling is reached."""
    attempts = previous.attempted_actions
    if pending_attempt is not None:
        attempts = (*attempts, pending_attempt)
    needs = tuple(
        replace(item, status="exhausted")
        if item.required and item.status == "open"
        else item
        for item in previous.information_needs
    )
    decision = GroundingDecision.from_mapping(
        {
            "decision_type": "incomplete",
            "need_id": None,
            "provider_id": None,
            "source_ref": None,
            "source_revision": None,
            "query": None,
            "reason": "The emergency PA call ceiling was reached.",
        }
    )
    return GroundingSession.create(
        revision=previous.revision + 1,
        requirement_text=previous.requirement_text,
        statements=previous.statements,
        information_needs=needs,
        attempted_actions=attempts,
        evidence_refs=previous.evidence_refs,
        decision=decision,
        status="incomplete",
        information_status="partial" if previous.statements else "not_enough",
    )


def _session_assessment(
    session: GroundingSession,
    actions: Sequence[_EligibleProviderAction],
) -> dict[str, object]:
    """Translate a session decision to the existing public needed_context shape."""
    decision = session.decision
    if decision.decision_type == "ready_for_ontology":
        return {
            "unresolved_semantic_need": None,
            "needed_context": None,
            "context understanding complete": True,
            "grounding_status": session.status,
        }
    if decision.decision_type == "incomplete":
        return {
            "unresolved_semantic_need": None,
            "needed_context": None,
            "context understanding complete": False,
            "grounding_status": session.status,
        }
    action = next(
        item
        for item in actions
        if (
            item.provider_id,
            item.source_ref,
            item.source_revision,
        )
        == (
            decision.provider_id,
            decision.source_ref,
            decision.source_revision,
        )
    )
    selected_need = next(
        item for item in session.information_needs if item.need_id == decision.need_id
    )
    produced = sorted(
        set(action.produced_record_types).intersection(
            selected_need.accepted_record_types
        )
    )[0]
    if action.evidence_type == "user":
        needed_context = {
            "context_ref": None,
            "request_live_observation": False,
            "clarification_question": decision.query,
        }
        need_kind = "user_intent"
        symbol = "product_requirement"
    elif action.evidence_type == "observation":
        needed_context = {
            "context_ref": None,
            "request_live_observation": True,
            "clarification_question": None,
        }
        need_kind = "typed_context_record"
        symbol = produced
    elif action.evidence_type in {"document", "CAD"}:
        needed_context = {
            "context_ref": action.source_ref,
            "request_live_observation": False,
            "clarification_question": None,
        }
        need_kind = "typed_context_record"
        symbol = produced
    else:
        raise ProductionGroundingError(
            "Selected provider action must execute inside the grounding runtime."
        )
    return {
        "unresolved_semantic_need": {
            "kind": need_kind,
            "symbol": symbol,
            "description": selected_need.question,
        },
        "needed_context": needed_context,
        "context understanding complete": False,
        "grounding_status": session.status,
    }


def _selected_session_action(
    session: GroundingSession,
    actions: Sequence[_EligibleProviderAction],
) -> _EligibleProviderAction | None:
    if session.decision.decision_type != "request_evidence":
        return None
    return next(
        (
            item
            for item in actions
            if (
                item.provider_id,
                item.source_ref,
                item.source_revision,
            )
            == (
                session.decision.provider_id,
                session.decision.source_ref,
                session.decision.source_revision,
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


def _session_requests_record_type(
    session: GroundingSession | None,
    record_type: str,
) -> bool:
    """Return whether the current exact action is intended to produce a record type."""
    if session is None or session.decision.decision_type != "request_evidence":
        return False
    need = next(
        (
            item
            for item in session.information_needs
            if item.need_id == session.decision.need_id
        ),
        None,
    )
    return need is not None and record_type in need.accepted_record_types


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


def _required_list(value: object, label: str) -> list[object]:
    """Return one required list from an untrusted model response."""
    if not isinstance(value, list):
        raise ProductionGroundingError(f"{label} must be a list.")
    return value


def _producer_descriptors(
    tbox: TBoxSnapshot,
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
    session = load_latest_grounding_session(root)
    if session is not None and session.decision.decision_type == "request_evidence":
        need = next(
            (
                item
                for item in session.information_needs
                if item.need_id == session.decision.need_id
            ),
            None,
        )
        if need is not None:
            evidence_type = served_context.get("evidence_type")
            candidates = {
                "CAD": {"CADMeshRecord"},
                "observation": {
                    "ColoredPointCloudSetRecord",
                    "RGBDSegmentationRecord",
                },
            }.get(str(evidence_type), set())
            selected = sorted(candidates.intersection(need.accepted_record_types))
            if selected:
                return selected[0]
    return None


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

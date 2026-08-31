"""Run native tool-using ProductAgent grounding through approved evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from cais_spade_llm.spec2primitives.agents.pa import context_serving
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import ProductAgentContextRuntime
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingProducerDescriptor,
    ProductContextView,
    TypedContextBinding,
    build_product_context_view,
    persist_product_context_view,
)
from cais_spade_llm.spec2primitives.agents.pa.ontology_grounding import (
    OntologyGroundingCandidate,
    OntologyGroundingError,
    OntologyGroundingInterruption,
    OntologyGroundingProposal,
    commit_ontology_grounding_candidate,
    propose_and_validate_ontology_grounding,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    ABoxSnapshot,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
    ResourceAssignmentNeed,
    RobotFrameLocationEvidenceError,
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
    DocumentVisionRuntime,
    interpret_document_evidence,
)
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_cad_path,
    approved_context_ref_evidence_types,
    approved_document_metadata,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    CameraToRobotCalibrationError,
    CameraToRobotCalibrationResult,
    RobotFrameConversionError,
    associate_segmented_candidate_by_size,
    preprocess_served_geometry,
    segment_preprocessed_observation,
    transform_correspondence_location_to_robot_frame,
)

_DOCUMENT_PRODUCER = "document_evidence"
_GEOMETRY_PRODUCER = "rgb_d_cad_grounding"
_CALIBRATION_PRODUCER = "camera_to_world_calibration"
_RETRIEVE_TOOL_NAME = "retrieve"
_MAX_TOOL_ROUNDS = 12
_LIVE_OBSERVATION_TIMEOUT_SEC = 5.0


class ProductionGroundingError(RuntimeError):
    """Raised when native production grounding cannot progress safely."""


class CameraToWorldCalibrationRuntime(Protocol):
    """Provide one approved camera-to-target calibration on explicit demand."""

    def materialize_camera_to_world_calibration(
        self,
        *,
        interaction_root: Path,
        grounding_record_path: Path,
        source_frame: str,
        target_frame: str,
        calibration_number: int,
    ) -> CameraToRobotCalibrationResult:
        """Persist an approved transform for the selected evidence and frame pair."""
        ...


@dataclass(frozen=True)
class _EvidenceHandle:
    evidence_id: str
    evidence_type: str
    display_name: str
    context_ref: str | None
    source_revision: str

    def discovery_record(self) -> dict[str, object]:
        """Return only metadata safe to expose before retrieval."""
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "display_name": self.display_name,
            "availability": "available",
        }


@dataclass(frozen=True)
class _GroundingGap:
    """Describe one descriptor-derived readiness gap for the next PA round."""

    required_record_type: str
    missing_record_types: tuple[str, ...]
    eligible_evidence_ids: tuple[str, ...]
    target_evidence_revision_required: bool = False
    provider_failure: str | None = None

    def to_prompt_projection(self) -> dict[str, object]:
        """Return the prompt-only projection exposed to PA."""
        return {
            "required_record_type": self.required_record_type,
            "missing_record_types": list(self.missing_record_types),
            "eligible_evidence_ids": list(self.eligible_evidence_ids),
            "target_evidence_revision_required": (
                self.target_evidence_revision_required
            ),
            "provider_failure": self.provider_failure,
        }

    def incomplete_message(self) -> str:
        """Return one generic diagnostic derived from the typed gap."""
        missing = ", ".join(self.missing_record_types) or self.required_record_type
        detail = f" Required grounding records remain unavailable: {missing}."
        if self.provider_failure:
            detail += f" Provider result: {self.provider_failure}"
        return detail.strip()


@dataclass(frozen=True)
class _PreparedGrounding:
    """Hold accepted physical context before semantic ABox commit."""

    abox: ABoxSnapshot
    view: ProductContextView
    need: ResourceAssignmentNeed
    location_binding: TypedContextBinding


class _NativeEvidenceInvestigation:
    """Resolve native retrieve calls and retain validated evidence state."""

    def __init__(
        self,
        *,
        runtime: ProductionProductContextGroundingRuntime,
        interaction_root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        requirement: str,
        handles: Sequence[_EvidenceHandle],
    ) -> None:
        self.runtime = runtime
        self.root = Path(interaction_root).resolve()
        self.tbox = tbox
        self.abox = abox
        self.requirement = requirement
        self.handles = {handle.evidence_id: handle for handle in handles}
        self.authorized_evidence_refs: set[str] = {"requirement_0001"}
        self.tool_call_refs: list[str] = []
        self.retrieved_results: dict[str, Mapping[str, object]] = {}
        self.retrieved_handle_ids: list[str] = []
        self.prior_evidence: list[Mapping[str, object]] = []
        self._tool_call_number = _latest_number(
            self.root / "interaction_record", "tool_call_*.json", "tool_call_"
        )
        self._retrieval_number = _latest_number(
            self.root / "interaction_record", "retrieval_*.json", "retrieval_"
        )
        self._load_prior_evidence()

    def _load_prior_evidence(self) -> None:
        """Restore hash-verified evidence state for a clarification resume."""
        for path in sorted(
            (self.root / "interaction_record").glob("tool_call_*.json")
        ):
            audit = _read_json(path)
            if audit.get("failure") is not None:
                continue
            resolved = audit.get("resolved_evidence")
            result_refs = audit.get("result_refs")
            if not isinstance(resolved, Mapping) or not isinstance(result_refs, list):
                raise ProductionGroundingError(
                    f"Successful tool audit is invalid: {path.name}."
                )
            evidence_id = resolved.get("evidence_id")
            handle = self.handles.get(evidence_id) if isinstance(evidence_id, str) else None
            if handle is None or resolved.get("source_revision") != handle.source_revision:
                raise ProductionGroundingError(
                    f"Previously retrieved evidence is stale: {path.name}."
                )
            record_refs: list[str] = []
            for item in result_refs:
                ref = item.get("ref") if isinstance(item, Mapping) else None
                sha256 = item.get("sha256") if isinstance(item, Mapping) else None
                if not isinstance(ref, str) or not isinstance(sha256, str):
                    raise ProductionGroundingError(
                        f"Tool audit result reference is invalid: {path.name}."
                    )
                record_path = (self.root / ref).resolve()
                try:
                    record_path.relative_to(self.root)
                except ValueError as exc:
                    raise ProductionGroundingError(
                        f"Tool audit result escapes the interaction: {path.name}."
                    ) from exc
                if _sha256_path(record_path) != sha256:
                    raise ProductionGroundingError(
                        f"Previously retrieved evidence changed: {ref}."
                    )
                record_refs.append(ref)
            if not record_refs:
                raise ProductionGroundingError(
                    f"Successful tool audit has no typed result: {path.name}."
                )
            source_refs = _restored_source_refs(self.root, handle, record_refs)
            result = _compact_retrieval_result(
                self.root,
                handle,
                {"typed_context_refs": record_refs},
                source_refs=source_refs,
            )
            returned_refs = result.get("evidence_refs")
            if isinstance(returned_refs, list):
                self.authorized_evidence_refs.update(
                    ref for ref in returned_refs if isinstance(ref, str) and ref
                )
            if handle.evidence_id not in self.retrieved_handle_ids:
                self.retrieved_handle_ids.append(handle.evidence_id)
                self.prior_evidence.append(
                    {"retrieval_state": "already_retrieved", **dict(result)}
                )
            if handle.evidence_type != "observation":
                self.retrieved_results[handle.evidence_id] = result

    async def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Execute one fail-closed ProductAgent tool call."""
        self._tool_call_number += 1
        call_id = f"tool_call_{self._tool_call_number:04d}"
        if tool_name != _RETRIEVE_TOOL_NAME or set(arguments) != {"evidence_id"}:
            return self._record_failure(
                call_id, tool_name, arguments, "malformed_tool_call",
                "Only retrieve with one evidence_id is authorized.",
            )
        evidence_id = arguments.get("evidence_id")
        handle = self.handles.get(evidence_id) if isinstance(evidence_id, str) else None
        if handle is None:
            return self._record_failure(
                call_id, tool_name, arguments, "unauthorized_evidence_id",
                "The evidence_id is unknown or no longer eligible.",
            )
        if handle.evidence_type != "observation" and evidence_id in self.retrieved_results:
            result = self.retrieved_results[evidence_id]
            self._record_success(call_id, handle, arguments, result, reused=True)
            return result
        if not _handle_revision_is_current(handle):
            return self._record_failure(
                call_id, tool_name, arguments, "stale_evidence_id",
                "The approved source changed after the evidence catalog was issued.",
                handle=handle,
            )

        self._retrieval_number += 1
        served = await asyncio.to_thread(
            context_serving.retrieve_pa_evidence,
            self.root,
            product_requirement=self.requirement,
            evidence_type=handle.evidence_type,
            context_ref=handle.context_ref,
            retrieval_number=self._retrieval_number,
            live_observation_timeout_sec=_LIVE_OBSERVATION_TIMEOUT_SEC,
        )
        served_context = served.get("served_context") if isinstance(served, Mapping) else None
        if not isinstance(served_context, Mapping):
            failure = served.get("failure") if isinstance(served, Mapping) else None
            return self._record_failure(
                call_id, tool_name, arguments, "retrieval_failed",
                _failure_message(failure), handle=handle,
            )
        try:
            delta = await self.runtime.interpret_retrieved_evidence(
                interaction_root=self.root,
                tbox=self.tbox,
                abox=self.abox,
                served_context=served_context,
                operation_number=self._retrieval_number,
            )
            source_refs = _served_source_refs(served_context)
            merge = validate_and_merge_triple_delta(
                self.root,
                self.tbox,
                _producer_for_type(handle.evidence_type),
                delta,
                authorized_evidence_refs=sorted(source_refs),
            )
            self.abox = merge.abox
            view = build_product_context_view(
                self.root,
                self.abox,
                attempted_evidence=tuple(self.retrieved_handle_ids) + (handle.evidence_id,),
                assessed_at_ns=time.time_ns(),
            )
            persist_product_context_view(self.root, view)
            result = _compact_retrieval_result(
                self.root, handle, delta, source_refs=source_refs,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return self._record_failure(
                call_id, tool_name, arguments, "evidence_processing_failed",
                f"{type(exc).__name__}: {exc}", handle=handle,
            )
        returned_refs = result.get("evidence_refs")
        if isinstance(returned_refs, list):
            self.authorized_evidence_refs.update(
                ref for ref in returned_refs if isinstance(ref, str) and ref
            )
        self.retrieved_results[handle.evidence_id] = result
        self.retrieved_handle_ids.append(handle.evidence_id)
        self._record_success(call_id, handle, arguments, result, reused=False)
        return result

    def _record_success(
        self,
        call_id: str,
        handle: _EvidenceHandle,
        arguments: Mapping[str, object],
        result: Mapping[str, object],
        *,
        reused: bool,
    ) -> None:
        refs = result.get("record_refs")
        record_refs = [item for item in refs if isinstance(item, str)] if isinstance(refs, list) else []
        self._persist_tool_call(
            call_id,
            {
                "schema_version": 1,
                "record_type": "ProductAgentToolCall",
                "tool_call_id": call_id,
                "tool_name": _RETRIEVE_TOOL_NAME,
                "arguments": dict(arguments),
                "resolved_evidence": {
                    "evidence_id": handle.evidence_id,
                    "evidence_type": handle.evidence_type,
                    "context_ref": handle.context_ref,
                    "source_revision": handle.source_revision,
                },
                "reused": reused,
                "result_refs": [
                    {"ref": ref, "sha256": _sha256_path(self.root / ref)}
                    for ref in record_refs
                ],
                "failure": None,
            },
        )

    def _record_failure(
        self,
        call_id: str,
        tool_name: str,
        arguments: Mapping[str, object],
        reason: str,
        message: str,
        *,
        handle: _EvidenceHandle | None = None,
    ) -> Mapping[str, object]:
        self._persist_tool_call(
            call_id,
            {
                "schema_version": 1,
                "record_type": "ProductAgentToolCall",
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "arguments": dict(arguments),
                "resolved_evidence": None if handle is None else {
                    "evidence_id": handle.evidence_id,
                    "evidence_type": handle.evidence_type,
                    "context_ref": handle.context_ref,
                    "source_revision": handle.source_revision,
                },
                "reused": False,
                "result_refs": [],
                "failure": {"reason": reason, "message": message},
            },
        )
        return {"error": {"reason": reason, "message": message}}

    def _persist_tool_call(self, call_id: str, record: Mapping[str, object]) -> None:
        path = self.root / "interaction_record" / f"{call_id}.json"
        _write_json_exclusive(path, record)
        self.tool_call_refs.append(path.relative_to(self.root).as_posix())


class ProductionProductContextGroundingRuntime:
    """Resolve PA context using one native retrieve tool and deterministic providers."""

    def __init__(
        self,
        *,
        tbox: TBoxSnapshot,
        document_config: DocumentVLMConfig,
        document_vision_runtime: DocumentVisionRuntime,
        camera_to_world_calibration_runtime: CameraToWorldCalibrationRuntime | None = None,
        camera_to_world_calibration_unavailable_reason: str | None = None,
    ) -> None:
        """Create a runtime pinned to ontology, workcell, and provider authorities."""
        tbox.assert_unchanged()
        self._tbox = tbox
        self._document_config = document_config
        self._document_vision_runtime = document_vision_runtime
        self._camera_to_world_calibration_runtime = camera_to_world_calibration_runtime
        self._camera_to_world_calibration_unavailable_reason = camera_to_world_calibration_unavailable_reason
        self._registry = load_predefined_resource_registry(tbox)
        self._workcell = load_predefined_workcell(tbox, self._registry)
        self._descriptors = _producer_descriptors(
            calibration_available=camera_to_world_calibration_runtime is not None
        )

    def grounding_producer_descriptors(self) -> Sequence[GroundingProducerDescriptor]:
        """Return provider-owned typed-output descriptors."""
        self._validate_authorities(self._tbox)
        return self._descriptors

    async def ground_product_context(
        self,
        product_agent: ProductAgentContextRuntime,
        *,
        interaction_root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        product_context: Mapping[str, object],
        max_pa_turns: int,
        clarification_history: tuple[Mapping[str, object], ...] = (),
    ) -> Mapping[str, object]:
        """Run one evidence-gated PA investigation and deterministic completion."""
        del product_context
        self._validate_authorities(tbox)
        if abox.tbox_fingerprint != tbox.fingerprint:
            raise ProductionGroundingError("ABox does not match the pinned TBox.")
        handles = _approved_evidence_handles()
        investigation = _NativeEvidenceInvestigation(
            runtime=self,
            interaction_root=interaction_root,
            tbox=tbox,
            abox=abox,
            requirement=abox.product_requirement,
            handles=handles,
        )
        clarification_evidence = _answered_clarification_evidence(
            investigation.root,
            clarification_history,
        )
        investigation.authorized_evidence_refs.update(
            str(item["evidence_ref"]) for item in clarification_evidence
        )
        validation_feedback: Mapping[str, object] | None = None
        evidence_gap: _GroundingGap | None = None
        target_frame = _configured_target_frame(self._workcell)
        for pa_round in range(1, max_pa_turns + 1):
            catalog = _current_evidence_catalog(handles, investigation)
            catalog.extend(clarification_evidence)
            try:
                outcome = await propose_and_validate_ontology_grounding(
                    product_agent,
                    interaction_root=interaction_root,
                    tbox=tbox,
                    abox=investigation.abox,
                    workcell=self._workcell,
                    evidence_catalog=catalog,
                    authorized_evidence_refs=investigation.authorized_evidence_refs,
                    tools=[_retrieve_tool(handles)],
                    tool_executor=investigation.execute,
                    max_tool_rounds=min(
                        max_pa_turns - pa_round + 1,
                        _MAX_TOOL_ROUNDS,
                    ),
                    required_output_projection={
                        "record_type": "RobotFrameLocationRecord",
                        "target_frame": target_frame,
                        "purpose": "deterministic coarse resource reachability",
                    },
                    validation_gap=(
                        None
                        if validation_feedback is None
                        else validation_feedback
                    ),
                )
            except OntologyGroundingError as exc:
                if pa_round < max_pa_turns:
                    validation_feedback = {
                        "kind": "ontology_proposal_validation_error",
                        "message": str(exc),
                        "expected_revision": (
                            "Return a corrected proposal that satisfies every supplied "
                            "ontology invariant. Do not invent or silently add an "
                            "unsupported assertion."
                        ),
                    }
                    evidence_gap = None
                    continue
                raise ProductionGroundingError(str(exc)) from exc
            if isinstance(outcome, OntologyGroundingInterruption):
                if (
                    outcome.kind == "clarification_question"
                    and _clarification_requests_system_choice(
                        outcome.message,
                        required_record_type="RobotFrameLocationRecord",
                        handles=handles,
                    )
                ):
                    if pa_round < max_pa_turns:
                        validation_feedback = {
                            "kind": "system_owned_grounding_choice",
                            "message": (
                                "A user clarification cannot decide whether a supplied "
                                "required output should be satisfied or which approved "
                                "evidence category should be used."
                            ),
                            "expected_revision": (
                                "Use any relevant approved retrieve handle and return a "
                                "proposal, or return insufficient_evidence if the "
                                "required output cannot be supported."
                            ),
                        }
                        continue
                    return {
                        "grounding_status": "incomplete",
                        "insufficient_evidence": (
                            "PA repeatedly delegated a system-owned grounding choice "
                            "to the user."
                        ),
                        "tool_call_refs": list(investigation.tool_call_refs),
                    }
                if (
                    outcome.kind == "clarification_question"
                    and not investigation.retrieved_handle_ids
                ):
                    if pa_round < max_pa_turns:
                        validation_feedback = {
                            "kind": "evidence_first_clarification",
                            "message": (
                                "A user clarification is permitted only after relevant "
                                "approved evidence has been retrieved and considered."
                            ),
                            "expected_revision": (
                                "Use the controlled retrieve tool for approved evidence "
                                "plausibly relevant to the ambiguity, then return a "
                                "proposal, a remaining requirement-meaning clarification, "
                                "or insufficient_evidence. Do not assume or supply an "
                                "interpretation on the user's behalf."
                            ),
                        }
                        continue
                    return {
                        "grounding_status": "incomplete",
                        "insufficient_evidence": (
                            "PA requested user clarification before retrieving and "
                            "considering approved evidence."
                        ),
                        "tool_call_refs": list(investigation.tool_call_refs),
                    }
                if (
                    outcome.kind == "insufficient_evidence"
                    and evidence_gap is not None
                    and _has_unattempted_evidence(evidence_gap, investigation)
                    and pa_round < max_pa_turns
                ):
                    continue
                key = outcome.kind
                return {
                    "grounding_status": (
                        "clarification_required"
                        if key == "clarification_question"
                        else "incomplete"
                    ),
                    key: outcome.message,
                    "tool_call_refs": list(investigation.tool_call_refs),
                }

            prepared = await self._prepare_required_context(
                root=investigation.root,
                tbox=tbox,
                abox=investigation.abox,
                candidate=outcome,
                handles=handles,
                attempted_evidence=tuple(investigation.retrieved_handle_ids),
            )
            if isinstance(prepared, _GroundingGap):
                evidence_gap = prepared
                validation_feedback = prepared.to_prompt_projection()
                if (
                    not prepared.eligible_evidence_ids
                    and not prepared.target_evidence_revision_required
                ):
                    return {
                        "grounding_status": "incomplete",
                        "insufficient_evidence": prepared.incomplete_message(),
                        "tool_call_refs": list(investigation.tool_call_refs),
                    }
                continue

            investigation.abox = prepared.abox
            accepted = commit_ontology_grounding_candidate(
                outcome,
                interaction_root=investigation.root,
                tbox=tbox,
                abox=investigation.abox,
                workcell=self._workcell,
                authorized_evidence_refs=investigation.authorized_evidence_refs,
            )
            investigation.abox = accepted.merge.abox
            accepted_view = build_product_context_view(
                investigation.root,
                investigation.abox,
                attempted_evidence=tuple(investigation.retrieved_handle_ids),
                assessed_at_ns=time.time_ns(),
            )
            persist_product_context_view(investigation.root, accepted_view)
            completion = await self._complete_resource_assignment(
                root=investigation.root,
                tbox=tbox,
                abox=investigation.abox,
                view=accepted_view,
                location_binding=prepared.location_binding,
            )
            return {
                **completion,
                "ontology_projection_ref": accepted.proposal_path.relative_to(
                    investigation.root
                ).as_posix(),
                "tool_call_refs": list(investigation.tool_call_refs),
            }

        final_gap = evidence_gap or _GroundingGap(
            required_record_type="RobotFrameLocationRecord",
            missing_record_types=("RobotFrameLocationRecord",),
            eligible_evidence_ids=(),
            provider_failure="The bounded PA investigation was exhausted.",
        )
        return {
            "grounding_status": "incomplete",
            "insufficient_evidence": final_gap.incomplete_message(),
            "tool_call_refs": list(investigation.tool_call_refs),
        }

    async def interpret_retrieved_evidence(
        self,
        *,
        interaction_root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        served_context: Mapping[str, object],
        operation_number: int,
    ) -> Mapping[str, object]:
        """Convert one approved high-level source into typed evidence records."""
        evidence_type = served_context.get("evidence_type")
        if evidence_type == "document":
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
        if evidence_type not in {"CAD", "observation"}:
            raise ProductionGroundingError("Retrieved evidence type is unsupported.")
        preprocessing = await asyncio.to_thread(
            preprocess_served_geometry,
            interaction_root=interaction_root,
            served_context=served_context,
            operation_number=operation_number,
        )
        delta = dict(preprocessing.delta)
        if evidence_type == "observation":
            segmentation = await asyncio.to_thread(
                segment_preprocessed_observation,
                interaction_root=interaction_root,
                observation_record_path=preprocessing.record_path,
                segmentation_number=_next_number(
                    Path(interaction_root),
                    "products/grounding/rgb_d_cad_grounding/segmentation_*",
                ),
            )
            refs = list(delta.get("typed_context_refs", []))
            refs.append(segmentation.record_path.relative_to(interaction_root).as_posix())
            delta["typed_context_refs"] = refs
        return delta

    async def _prepare_required_context(
        self,
        *,
        root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        candidate: OntologyGroundingCandidate,
        handles: Sequence[_EvidenceHandle],
        attempted_evidence: Sequence[str],
    ) -> _PreparedGrounding | _GroundingGap:
        """Resolve the required physical context before semantic ABox commit."""
        need = derive_resource_assignment_need(
            candidate.provisional_abox,
            self._workcell,
        )
        if need is None:
            return _GroundingGap(
                required_record_type="RobotFrameLocationRecord",
                missing_record_types=("ResourceAssignmentNeed",),
                eligible_evidence_ids=(),
                provider_failure="No resource-assignment semantic join was derived.",
            )
        try:
            required_plan = _required_record_plan(
                self._descriptors,
                need.required_record_type,
            )
        except ProductionGroundingError as exc:
            return _GroundingGap(
                required_record_type=need.required_record_type,
                missing_record_types=(need.required_record_type,),
                eligible_evidence_ids=(),
                provider_failure=str(exc),
            )
        view = build_product_context_view(
            root,
            abox,
            attempted_evidence=attempted_evidence,
            assessed_at_ns=time.time_ns(),
        )
        ready_binding = _target_required_binding(view, candidate.proposal, need)
        if ready_binding is not None:
            return _PreparedGrounding(
                abox=abox,
                view=view,
                need=need,
                location_binding=ready_binding,
            )
        cad_bindings = _proposal_cad_bindings(root, view, candidate.proposal)
        segmentation = _newest_binding(view, "RGBDSegmentationRecord")
        if not cad_bindings or segmentation is None:
            return _grounding_gap(
                required_record_type=need.required_record_type,
                required_plan=required_plan,
                descriptors=self._descriptors,
                handles=handles,
                view=view,
                target_cad_bindings=cad_bindings,
                target_evidence_revision_required=(
                    not cad_bindings
                    and _newest_binding(view, "CADMeshRecord") is not None
                ),
            )
        accepted_correspondences: list[TypedContextBinding] = []
        current_abox = abox
        for cad in cad_bindings:
            result = await asyncio.to_thread(
                associate_segmented_candidate_by_size,
                interaction_root=root,
                cad_record_path=root / cad.record_ref,
                segmentation_record_path=root / segmentation.record_ref,
                correspondence_number=_next_number(root, "products/grounding/rgb_d_cad_grounding/correspondence_*"),
            )
            current_abox, binding = _merge_derived_record(
                root, tbox, current_abox, result.record_path,
                status=result.CAD_correspondence,
                prerequisite_bindings=(cad, segmentation),
            )
            if result.CAD_correspondence == "accepted":
                accepted_correspondences.append(binding)
        if len(accepted_correspondences) != 1:
            current_view = build_product_context_view(
                root,
                current_abox,
                attempted_evidence=attempted_evidence,
                assessed_at_ns=time.time_ns(),
            )
            return _grounding_gap(
                required_record_type=need.required_record_type,
                required_plan=required_plan,
                descriptors=self._descriptors,
                handles=handles,
                view=current_view,
                target_cad_bindings=cad_bindings,
                provider_failure=(
                    "Physical correspondence did not identify exactly one target."
                ),
            )
        accepted_correspondence = accepted_correspondences[0]
        correspondence_record = _read_json(root / accepted_correspondence.record_ref)
        selected = correspondence_record.get("selected_candidate")
        source_frame = selected.get("frame") if isinstance(selected, Mapping) else None
        if not isinstance(source_frame, str) or not source_frame:
            raise ProductionGroundingError("Accepted correspondence has no source frame.")
        calibration_runtime = self._camera_to_world_calibration_runtime
        if calibration_runtime is None:
            return _GroundingGap(
                required_record_type=need.required_record_type,
                missing_record_types=("CameraToRobotCalibrationRecord",),
                eligible_evidence_ids=(),
                provider_failure=(
                    self._camera_to_world_calibration_unavailable_reason
                    or "No approved camera calibration is available."
                ),
            )
        try:
            calibration = await asyncio.to_thread(
                calibration_runtime.materialize_camera_to_world_calibration,
                interaction_root=root,
                grounding_record_path=root / accepted_correspondence.record_ref,
                source_frame=source_frame,
                target_frame=need.target_frame,
                calibration_number=_next_number(root, "products/grounding/rgb_d_cad_grounding/calibration_*"),
            )
            current_abox, calibration_binding = _merge_derived_record(
                root, tbox, current_abox, calibration.record_path,
                status="accepted", prerequisite_bindings=(accepted_correspondence,),
            )
            location = await asyncio.to_thread(
                transform_correspondence_location_to_robot_frame,
                interaction_root=root,
                correspondence_record_path=root / accepted_correspondence.record_ref,
                calibration_record_path=calibration.record_path,
                target_frame=need.target_frame,
                location_number=_next_number(root, "products/grounding/rgb_d_cad_grounding/robot_location_*"),
            )
            current_abox, location_binding = _merge_derived_record(
                root, tbox, current_abox, location.record_path,
                status="accepted",
                prerequisite_bindings=(accepted_correspondence, calibration_binding),
            )
        except (CameraToRobotCalibrationError, RobotFrameConversionError, RobotFrameLocationEvidenceError) as exc:
            return _GroundingGap(
                required_record_type=need.required_record_type,
                missing_record_types=(need.required_record_type,),
                eligible_evidence_ids=(),
                provider_failure=str(exc),
            )
        final_view = build_product_context_view(
            root,
            current_abox,
            attempted_evidence=attempted_evidence,
            assessed_at_ns=time.time_ns(),
        )
        return _PreparedGrounding(
            abox=current_abox,
            view=final_view,
            need=need,
            location_binding=location_binding,
        )

    async def _complete_resource_assignment(
        self,
        *,
        root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        view: ProductContextView,
        location_binding: TypedContextBinding,
    ) -> Mapping[str, object]:
        """Select and commit a resource after semantic and physical grounding."""
        need = derive_resource_assignment_need(abox, self._workcell)
        if need is None:
            raise ProductionGroundingError(
                "Committed ontology did not preserve the resource-assignment need."
            )
        selection = select_predefined_resource(
            interaction_root=root,
            tbox=tbox,
            registry=self._registry,
            workcell=self._workcell,
            need=need,
            grounding_record_path=root / location_binding.record_ref,
            selection_number=_next_number(
                root,
                "products/grounding/resource_selection/selection_*",
            ),
        )
        if selection.selected_resource_iri is None:
            return {
                "grounding_status": "incomplete",
                "insufficient_evidence": "No configured resource is coarsely reachable from the accepted robot-frame location.",
                "resource_selection_ref": selection.record_ref,
            }
        assignment = commit_resource_assignment(
            interaction_root=root,
            tbox=tbox,
            registry=self._registry,
            workcell=self._workcell,
            need=need,
            selection=selection,
        )
        final_view = build_product_context_view(
            root,
            assignment.abox,
            attempted_evidence=view.attempted_evidence,
            assessed_at_ns=time.time_ns(),
        )
        persist_product_context_view(root, final_view)
        return {
            "grounding_status": "complete",
            "resource_selection_ref": selection.record_ref,
            "resource_assignment_delta_count": assignment.abox.delta_count,
        }

    def _validate_authorities(self, tbox: TBoxSnapshot) -> None:
        self._tbox.assert_unchanged()
        self._registry.assert_unchanged()
        self._workcell.assert_unchanged()
        tbox.assert_unchanged()
        if tbox.fingerprint != self._tbox.fingerprint:
            raise ProductionGroundingError("Production TBox authority changed.")


def _producer_descriptors(*, calibration_available: bool) -> tuple[GroundingProducerDescriptor, ...]:
    return (
        GroundingProducerDescriptor.from_mapping({
            "provider_id": _DOCUMENT_PRODUCER,
            "description": "Read every page of one approved document.",
            "accepted_evidence_types": ["document"],
            "produced_record_types": ["DocumentOverviewRecord"],
            "prerequisites": {"DocumentOverviewRecord": []},
            "availability": True,
            "estimated_cost": 2,
        }),
        GroundingProducerDescriptor.from_mapping({
            "provider_id": _GEOMETRY_PRODUCER,
            "description": "Measure approved CAD evidence.",
            "accepted_evidence_types": ["CAD"],
            "produced_record_types": ["CADMeshRecord"],
            "prerequisites": {"CADMeshRecord": []},
            "availability": True,
            "estimated_cost": 2,
        }),
        GroundingProducerDescriptor.from_mapping({
            "provider_id": _GEOMETRY_PRODUCER,
            "description": "Measure approved live observation evidence.",
            "accepted_evidence_types": ["observation"],
            "produced_record_types": [
                "ColoredPointCloudSetRecord", "RGBDSegmentationRecord",
            ],
            "prerequisites": {
                "ColoredPointCloudSetRecord": [],
                "RGBDSegmentationRecord": ["ColoredPointCloudSetRecord"],
            },
            "availability": True,
            "estimated_cost": 2,
        }),
        GroundingProducerDescriptor.from_mapping({
            "provider_id": _GEOMETRY_PRODUCER,
            "description": "Derive physical context from accepted typed records.",
            "accepted_evidence_types": ["existing_record"],
            "produced_record_types": [
                "CADSizeCorrespondenceRecord", "CADPoseEstimationRecord",
                "RobotFrameLocationRecord",
            ],
            "prerequisites": {
                "CADSizeCorrespondenceRecord": ["CADMeshRecord", "RGBDSegmentationRecord"],
                "CADPoseEstimationRecord": ["CADSizeCorrespondenceRecord"],
                "RobotFrameLocationRecord": ["CADSizeCorrespondenceRecord", "CameraToRobotCalibrationRecord"],
            },
            "availability": True,
            "estimated_cost": 2,
        }),
        GroundingProducerDescriptor.from_mapping({
            "provider_id": _CALIBRATION_PRODUCER,
            "description": "Materialize approved camera calibration.",
            "accepted_evidence_types": ["existing_record"],
            "produced_record_types": ["CameraToRobotCalibrationRecord"],
            "prerequisites": {"CameraToRobotCalibrationRecord": ["CADSizeCorrespondenceRecord"]},
            "availability": calibration_available,
            "estimated_cost": 0,
        }),
    )


def _approved_evidence_handles() -> tuple[_EvidenceHandle, ...]:
    handles: list[_EvidenceHandle] = []
    for index, (context_ref, evidence_type) in enumerate(
        sorted(approved_context_ref_evidence_types().items()), start=1,
    ):
        handles.append(_EvidenceHandle(
            evidence_id=f"evidence_{index:04d}",
            evidence_type=evidence_type,
            display_name=context_ref,
            context_ref=context_ref,
            source_revision=_static_source_revision(context_ref, evidence_type),
        ))
    handles.append(_EvidenceHandle(
        evidence_id=f"evidence_{len(handles) + 1:04d}",
        evidence_type="observation",
        display_name="fresh live RGB-D observation",
        context_ref=None,
        source_revision="fresh_on_call",
    ))
    return tuple(handles)


def _retrieve_tool(handles: Sequence[_EvidenceHandle]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _RETRIEVE_TOOL_NAME,
            "description": "Retrieve and analyze one approved evidence catalog entry.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_id"],
                "properties": {"evidence_id": {"type": "string", "enum": [handle.evidence_id for handle in handles]}},
            },
        },
    }


def _static_source_revision(context_ref: str, evidence_type: str) -> str:
    if evidence_type == "document":
        return str(approved_document_metadata(context_ref)["source_sha256"])
    if evidence_type == "CAD":
        return hashlib.sha256(approved_cad_path(context_ref).read_bytes()).hexdigest()
    raise ProductionGroundingError("Static evidence type is unsupported.")


def _handle_revision_is_current(handle: _EvidenceHandle) -> bool:
    if handle.evidence_type == "observation":
        return True
    if handle.context_ref is None:
        return False
    try:
        return handle.source_revision == _static_source_revision(handle.context_ref, handle.evidence_type)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _producer_for_type(evidence_type: str) -> str:
    return _DOCUMENT_PRODUCER if evidence_type == "document" else _GEOMETRY_PRODUCER


def _served_source_refs(served: Mapping[str, object]) -> set[str]:
    evidence_type = served.get("evidence_type")
    if evidence_type in {"document", "CAD"}:
        context_ref = served.get("context_ref")
        if not isinstance(context_ref, str):
            raise ProductionGroundingError("Served static evidence has no context ref.")
        refs = {context_ref}
        if evidence_type == "document":
            document = served.get("document_evidence")
            pages = document.get("pages") if isinstance(document, Mapping) else None
            if not isinstance(pages, list):
                raise ProductionGroundingError("Served document pages are invalid.")
            refs.update(f"{context_ref}#page={index}" for index in range(1, len(pages) + 1))
        return refs
    observation_ref = served.get("observation_ref")
    if evidence_type == "observation" and isinstance(observation_ref, str):
        return {observation_ref}
    raise ProductionGroundingError("Served evidence identity is invalid.")


def _restored_source_refs(
    root: Path,
    handle: _EvidenceHandle,
    record_refs: Sequence[str],
) -> set[str]:
    """Recover citation handles from previously accepted typed records."""
    records = [_read_json(root / ref) for ref in record_refs]
    if handle.evidence_type == "document":
        if handle.context_ref is None:
            raise ProductionGroundingError("Restored document has no context ref.")
        refs = {handle.context_ref}
        overview = records[-1].get("overview")
        pages = overview.get("pages") if isinstance(overview, Mapping) else None
        if isinstance(pages, list):
            refs.update(
                evidence_ref
                for page in pages
                if isinstance(page, Mapping)
                for evidence_ref in [page.get("evidence_ref")]
                if isinstance(evidence_ref, str) and evidence_ref
            )
        return refs
    if handle.evidence_type == "CAD":
        if handle.context_ref is None:
            raise ProductionGroundingError("Restored CAD has no context ref.")
        return {handle.context_ref}
    observation_ref = records[0].get("observation_ref")
    if handle.evidence_type == "observation" and isinstance(observation_ref, str):
        return {observation_ref}
    raise ProductionGroundingError("Restored evidence identity is invalid.")


def _compact_retrieval_result(
    root: Path,
    handle: _EvidenceHandle,
    delta: Mapping[str, object],
    *,
    source_refs: set[str],
) -> Mapping[str, object]:
    refs = delta.get("typed_context_refs")
    if not isinstance(refs, list) or not refs or not all(isinstance(ref, str) for ref in refs):
        raise ProductionGroundingError("Evidence producer returned no typed record.")
    records = [_read_json(root / ref) for ref in refs]
    record_refs = [str(ref) for ref in refs]
    evidence_refs = sorted({*source_refs, *record_refs})
    if handle.evidence_type == "document":
        overview = records[-1].get("overview")
        if not isinstance(overview, Mapping):
            raise ProductionGroundingError("Document overview projection is invalid.")
        return {
            "evidence_type": "document", "source": handle.display_name,
            "evidence_refs": evidence_refs, "record_refs": record_refs,
            "summary": overview.get("summary"), "pages": overview.get("pages"),
            "visual_observations": overview.get("observations"), "uncertainty": overview.get("uncertainty"),
        }
    if handle.evidence_type == "CAD":
        record = records[0]
        return {
            "evidence_type": "CAD", "source": handle.display_name,
            "evidence_refs": evidence_refs, "record_refs": record_refs,
            "coordinate_frame": record.get("coordinate_frame"), "units": record.get("stored_units"),
            "triangle_count": record.get("triangle_count"), "vertex_count": record.get("vertex_count"),
            "dimensions": record.get("bounds_m"), "centroid": record.get("vertex_centroid_m"),
        }
    point_cloud = records[0]
    segmentation = records[-1]
    return {
        "evidence_type": "observation", "source": handle.display_name,
        "evidence_refs": evidence_refs, "record_refs": record_refs,
        "observation_ref": point_cloud.get("observation_ref"), "cameras": point_cloud.get("cameras"),
        "segmentation": {
            "candidate_state": segmentation.get("candidate_state"),
            "assembly_candidate_count": segmentation.get("assembly_candidate_count"),
            "cameras": segmentation.get("cameras"),
        },
    }


def _proposal_cad_bindings(
    root: Path,
    view: ProductContextView,
    proposal: OntologyGroundingProposal,
) -> tuple[TypedContextBinding, ...]:
    cited = _proposal_target_evidence_refs(proposal)
    if not cited:
        return ()
    candidates: list[TypedContextBinding] = []
    for binding in view.typed_bindings:
        if binding.record_type != "CADMeshRecord" or binding.status != "accepted":
            continue
        record = _read_json(root / binding.record_ref)
        source = record.get("source")
        context_ref = source.get("context_ref") if isinstance(source, Mapping) else None
        binding_refs = {binding.record_ref, *binding.evidence_refs}
        if isinstance(context_ref, str):
            binding_refs.add(context_ref)
        if cited.intersection(binding_refs):
            candidates.append(binding)
    return tuple(candidates)


def _proposal_target_evidence_refs(
    proposal: OntologyGroundingProposal,
) -> frozenset[str]:
    """Return citations attached to the unique process-realized feature."""
    realized_indices = {
        relation.get("object_individual_index")
        for relation in proposal.relations
        if relation.get("predicate_iri", "").endswith("realizes")
    }
    if len(realized_indices) != 1:
        return ()
    realized_index = next(iter(realized_indices))
    target = next(
        (
            individual
            for individual in proposal.individuals
            if individual.get("individual_index") == realized_index
        ),
        None,
    )
    if not isinstance(target, Mapping):
        return frozenset()
    refs = target.get("evidence_refs", ())
    return frozenset(ref for ref in refs if isinstance(ref, str) and ref)


def _target_required_binding(
    view: ProductContextView,
    proposal: OntologyGroundingProposal,
    need: ResourceAssignmentNeed,
) -> TypedContextBinding | None:
    """Return the newest accepted required record tied to the primary target."""
    target_refs = _proposal_target_evidence_refs(proposal)
    candidates = [
        binding
        for binding in view.typed_bindings
        if binding.record_type == need.required_record_type
        and binding.status == "accepted"
        and binding.frame == need.target_frame
        and target_refs.intersection(binding.evidence_refs)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            -1 if item.observed_at_ns is None else item.observed_at_ns,
            item.record_ref,
        ),
    )


def _newest_binding(view: ProductContextView, record_type: str) -> TypedContextBinding | None:
    candidates = [binding for binding in view.typed_bindings if binding.record_type == record_type and binding.status == "accepted"]
    return None if not candidates else max(
        candidates,
        key=lambda item: (-1 if item.observed_at_ns is None else item.observed_at_ns, item.record_ref),
    )


def _merge_derived_record(
    root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    record_path: Path,
    *,
    status: str,
    prerequisite_bindings: Sequence[TypedContextBinding],
) -> tuple[ABoxSnapshot, TypedContextBinding]:
    record_ref = record_path.relative_to(root).as_posix()
    evidence_refs = sorted({ref for binding in prerequisite_bindings for ref in (binding.record_ref, *binding.evidence_refs)})
    merge = validate_and_merge_triple_delta(
        root, tbox, _GEOMETRY_PRODUCER,
        {"assertions": [], "uncertainty": [], "unresolved_evidence_needs": [], "typed_context_refs": [record_ref]},
        authorized_evidence_refs=evidence_refs,
    )
    view = build_product_context_view(root, merge.abox, attempted_evidence=(), assessed_at_ns=time.time_ns())
    persist_product_context_view(root, view)
    binding = next(item for item in view.typed_bindings if item.record_ref == record_ref)
    if binding.status != status:
        raise ProductionGroundingError(f"Derived {binding.record_type} status does not match its provider result.")
    return merge.abox, binding


def _required_record_plan(
    descriptors: Sequence[GroundingProducerDescriptor],
    required_record_type: str,
) -> tuple[str, ...]:
    """Return a descriptor-derived prerequisite closure in dependency order."""
    providers = _provider_by_output(descriptors)
    ordered: list[str] = []
    visiting: set[str] = set()

    def visit(record_type: str) -> None:
        if record_type in ordered:
            return
        if record_type in visiting:
            raise ProductionGroundingError("Provider prerequisites contain a cycle.")
        descriptor = providers.get(record_type)
        if descriptor is None:
            raise ProductionGroundingError(f"No available provider produces {record_type}.")
        visiting.add(record_type)
        for prerequisite in descriptor.prerequisites_for(record_type):
            visit(prerequisite)
        visiting.remove(record_type)
        ordered.append(record_type)

    visit(required_record_type)
    return tuple(ordered)


def _provider_by_output(
    descriptors: Sequence[GroundingProducerDescriptor],
) -> dict[str, GroundingProducerDescriptor]:
    """Return one unambiguous available descriptor for every output symbol."""
    providers: dict[str, GroundingProducerDescriptor] = {}
    for descriptor in descriptors:
        if not descriptor.availability:
            continue
        for output in descriptor.produced_record_types:
            if output in providers:
                raise ProductionGroundingError(
                    f"Multiple available providers produce {output}."
                )
            providers[output] = descriptor
    return providers


def _grounding_gap(
    *,
    required_record_type: str,
    required_plan: Sequence[str],
    descriptors: Sequence[GroundingProducerDescriptor],
    handles: Sequence[_EvidenceHandle],
    view: ProductContextView,
    target_cad_bindings: Sequence[TypedContextBinding],
    target_evidence_revision_required: bool = False,
    provider_failure: str | None = None,
) -> _GroundingGap:
    """Build one generic gap from required records and accepted bindings."""
    accepted = {
        binding.record_type
        for binding in view.typed_bindings
        if binding.status == "accepted"
    }
    if not target_cad_bindings:
        accepted.discard("CADMeshRecord")
    missing = tuple(record for record in required_plan if record not in accepted)
    raw_types = _raw_evidence_types_for_gap(
        descriptors,
        missing,
    )
    eligible_ids = tuple(
        handle.evidence_id
        for handle in handles
        if handle.evidence_type in raw_types
    )
    return _GroundingGap(
        required_record_type=required_record_type,
        missing_record_types=missing or (required_record_type,),
        eligible_evidence_ids=eligible_ids,
        target_evidence_revision_required=target_evidence_revision_required,
        provider_failure=provider_failure,
    )


def _raw_evidence_types_for_gap(
    descriptors: Sequence[GroundingProducerDescriptor],
    missing_record_types: Sequence[str],
) -> frozenset[str]:
    """Resolve raw evidence types capable of reopening a missing record chain."""
    providers = _provider_by_output(descriptors)
    raw_types = {"document", "CAD", "observation"}
    missing = set(missing_record_types)
    resolved: set[str] = set()

    def collect(record_type: str, *, include_satisfied_inputs: bool) -> None:
        descriptor = providers.get(record_type)
        if descriptor is None:
            return
        resolved.update(raw_types.intersection(descriptor.accepted_evidence_types))
        prerequisites = descriptor.prerequisites_for(record_type)
        unresolved_prerequisites = [
            prerequisite for prerequisite in prerequisites if prerequisite in missing
        ]
        for prerequisite in (
            prerequisites
            if include_satisfied_inputs and not unresolved_prerequisites
            else unresolved_prerequisites
        ):
            collect(prerequisite, include_satisfied_inputs=include_satisfied_inputs)

    for record_type in missing_record_types:
        collect(record_type, include_satisfied_inputs=False)
    if not resolved:
        for record_type in missing_record_types:
            collect(record_type, include_satisfied_inputs=True)
    return frozenset(resolved)


def _current_evidence_catalog(
    handles: Sequence[_EvidenceHandle],
    investigation: _NativeEvidenceInvestigation,
) -> list[Mapping[str, object]]:
    """Return discovery metadata plus evidence retrieved in prior PA rounds."""
    catalog: list[Mapping[str, object]] = [
        handle.discovery_record() for handle in handles
    ]
    seen_record_refs: set[tuple[str, ...]] = set()
    for item in (
        *investigation.prior_evidence,
        *(
            {"retrieval_state": "already_retrieved", **dict(result)}
            for result in investigation.retrieved_results.values()
        ),
    ):
        refs = item.get("record_refs")
        key = tuple(str(ref) for ref in refs) if isinstance(refs, list) else ()
        if key and key in seen_record_refs:
            continue
        if key:
            seen_record_refs.add(key)
        catalog.append(dict(item))
    return catalog


def _has_unattempted_evidence(
    gap: _GroundingGap,
    investigation: _NativeEvidenceInvestigation,
) -> bool:
    """Return whether an eligible source can still change current evidence state."""
    for evidence_id in gap.eligible_evidence_ids:
        handle = investigation.handles.get(evidence_id)
        if handle is None:
            continue
        if handle.evidence_type == "observation":
            return True
        if evidence_id not in investigation.retrieved_handle_ids:
            return True
    return gap.target_evidence_revision_required


def _clarification_requests_system_choice(
    message: str,
    *,
    required_record_type: str,
    handles: Sequence[_EvidenceHandle],
) -> bool:
    """Reject clarification about system-owned output or evidence selection."""
    normalized = message.casefold()
    if required_record_type.casefold() in normalized:
        return True
    words = {
        token.strip(".,:;!?()[]{}\"'")
        for token in normalized.split()
    }
    system_terms = {
        _RETRIEVE_TOOL_NAME.casefold(),
        "evidence_id",
        *(handle.evidence_id.casefold() for handle in handles),
        *(handle.evidence_type.casefold() for handle in handles),
    }
    return bool(words.intersection(system_terms))


def _answered_clarification_evidence(
    root: Path,
    clarification_history: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Build evidence entries from exact persisted clarification records."""
    evidence: list[Mapping[str, object]] = []
    for clarification in clarification_history:
        question_turn = clarification.get("question_turn")
        reply = clarification.get("reply")
        if (
            not isinstance(question_turn, int)
            or isinstance(question_turn, bool)
            or question_turn < 1
            or not isinstance(reply, str)
            or not reply
        ):
            raise ProductionGroundingError(
                "Answered clarification history is invalid."
            )
        evidence_ref = (
            Path("interaction_record") / f"clarification_{question_turn:04d}.json"
        ).as_posix()
        clarification_path = root / evidence_ref
        if (
            not clarification_path.is_file()
            or _read_json(clarification_path) != dict(clarification)
        ):
            raise ProductionGroundingError(
                "Answered clarification record is unavailable or changed."
            )
        evidence.append({
            "evidence_type": "user_clarification",
            "evidence_ref": evidence_ref,
            "reply": reply,
        })
    return evidence


def _configured_target_frame(workcell: Any) -> str:
    """Derive and validate the common target frame without task-label flags."""
    frames: set[str] = set()
    for resource in workcell._profile.resources:
        record = _read_json(resource.manifest_path)
        entry = record.get(resource.symbol)
        mode = entry.get("execution_mode") if isinstance(entry, Mapping) else None
        environment_name = {"simulation": "gazebo", "physical": "real"}.get(mode)
        environment = entry.get(environment_name) if isinstance(entry, Mapping) and environment_name else None
        static = environment.get("static_capabilities") if isinstance(environment, Mapping) else None
        reach = static.get("gripper_reach") if isinstance(static, Mapping) else None
        frame = reach.get("frame") if isinstance(reach, Mapping) else None
        if not isinstance(frame, str) or not frame:
            raise ProductionGroundingError("A configured resource has no reach frame.")
        frames.add(frame)
    if len(frames) != 1:
        raise ProductionGroundingError("Configured resources have inconsistent reach frames.")
    return next(iter(frames))


def _next_number(root: Path, pattern: str) -> int:
    return len(tuple(root.glob(pattern))) + 1


def _latest_number(root: Path, pattern: str, prefix: str) -> int:
    """Return the largest append-only numeric suffix under one record root."""
    numbers = [
        int(path.stem.removeprefix(prefix))
        for path in root.glob(pattern)
        if path.stem.removeprefix(prefix).isdigit()
    ]
    return max(numbers, default=0)


def _failure_message(value: object) -> str:
    if isinstance(value, Mapping) and isinstance(value.get("message"), str):
        return str(value["message"])
    return "Approved evidence retrieval failed."


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionGroundingError(f"Grounding record could not be read: {path}.") from exc
    if not isinstance(value, dict):
        raise ProductionGroundingError(f"Grounding record is not an object: {path}.")
    return value


def _sha256_path(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ProductionGroundingError(f"Grounding record is unavailable: {path}.") from exc


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise ProductionGroundingError(f"Audit record already exists: {path.name}.") from exc

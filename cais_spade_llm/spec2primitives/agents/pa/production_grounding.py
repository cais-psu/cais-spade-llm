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

from rdflib import URIRef

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
    OntologyGroundingError,
    OntologyGroundingInterruption,
    OntologyGroundingProposal,
    commit_ontology_grounding_candidate,
    propose_and_validate_ontology_grounding,
    reject_ontology_grounding_candidate,
    review_target_feature_semantics,
)
from cais_spade_llm.spec2primitives.agents.pa.presentation_records import (
    AllocationEvidenceEntry,
    AllocationEvidenceSource,
    AllocationPresentationRecord,
    EvidencePresentationRecord,
    load_or_create_allocation_presentation,
    load_or_create_evidence_presentation,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    ABoxSnapshot,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
    ReachabilityCheckRecord,
    StateReachEvidence,
    candidate_resource_catalog,
    check_resource_reachability,
    commit_resource_assignment,
    persist_cartesian_reachability,
    persist_pa_resource_selection,
    prepare_cartesian_reachability,
)
from cais_spade_llm.spec2primitives.agents.ra.feasibility_validation import (
    PlanOnlyFeasibilityValidation,
    RobotAgentFeasibilityRuntime,
    validate_provisional_allocation,
)
from cais_spade_llm.spec2primitives.config import DocumentVLMConfig, ObservationVLMConfig
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
    CameraToRobotCalibrationResult,
    ObservationVisionRuntime,
    associate_segmented_candidate_by_size,
    preprocess_served_geometry,
    review_observation_candidates,
    segment_preprocessed_observation,
    transform_segmentation_candidate_location_to_robot_frame,
)

_DOCUMENT_PRODUCER = "document_evidence"
_GEOMETRY_PRODUCER = "rgb_d_cad_grounding"
_CALIBRATION_PRODUCER = "camera_to_world_calibration"
_RETRIEVE_TOOL_NAME = "retrieve"
_COMPARE_CAD_SIZE_TOOL_NAME = "compare_cad_size"
_MAX_TOOL_ROUNDS = 12
_LIVE_OBSERVATION_TIMEOUT_SEC = 5.0
_TOOL_ROUND_LIMIT_MESSAGE = "Exceeded max tool rounds"


class ProductionGroundingError(RuntimeError):
    """Raised when native production grounding cannot progress safely."""


def _grounding_readiness_projection() -> Mapping[str, object]:
    """Return requirement-neutral completion constraints for PA grounding."""
    return {
        "completion_rule": (
            "A complete result needs exactly one uniquely supported observed current "
            "object and exactly one uniquely supported observed destination or support. "
            "If approved evidence does not uniquely support both state assignments, "
            "return insufficient_evidence."
        ),
        "selection_authority": (
            "The ProductAgent chooses which approved evidence and controlled tools are "
            "relevant; the controller does not prescribe a source, component, or candidate."
        ),
        "citation_rule": (
            "Each selected state value cites every retrieved record that directly supports "
            "its identity and semantic role."
        ),
    }


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
        record: dict[str, object] = {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "availability": "available",
        }
        if self.evidence_type == "CAD" and self.context_ref is not None:
            record["context_ref"] = self.context_ref
        return record


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
            "target_evidence_revision_required": (self.target_evidence_revision_required),
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
class _StateCandidateBinding:
    """Identify one PA-authored state value that selects a neutral candidate."""

    state: str
    name: str
    record_ref: str
    field_path: str


@dataclass(frozen=True)
class _CADComparisonBinding:
    """Pin one PA-requested CAD comparison to its exact typed inputs."""

    record_ref: str
    cad_record_ref: str
    segmentation_record_ref: str


class _NativeEvidenceInvestigation:
    """Resolve native retrieve calls and retain validated evidence state."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        runtime: ProductionProductContextGroundingRuntime,
        interaction_root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        requirement: str,
        handles: Sequence[_EvidenceHandle],
        presentation: EvidencePresentationRecord,
    ) -> None:
        self.runtime = runtime
        self.root = Path(interaction_root).resolve()
        self.tbox = tbox
        self.abox = abox
        self.requirement = requirement
        self.presentation = presentation
        self.presentation.assert_unchanged()
        self.handles = {handle.evidence_id: handle for handle in handles}
        self.authorized_evidence_refs: set[str] = {"requirement_0001"}
        self._canonical_by_pa_ref: dict[str, str] = {"requirement_0001": "requirement_0001"}
        self._pa_by_canonical_ref: dict[str, str] = {"requirement_0001": "requirement_0001"}
        self.tool_call_refs: list[str] = []
        self.retrieved_results: dict[str, Mapping[str, object]] = {}
        self.comparison_results: dict[tuple[str, str], Mapping[str, object]] = {}
        self.comparison_bindings: dict[str, _CADComparisonBinding] = {}
        self.retrieved_handle_ids: list[str] = []
        self.prior_evidence: list[Mapping[str, object]] = []
        self._served_observation_context: Mapping[str, object] | None = None
        self.run_tool_call_count = 0
        self._tool_call_number = _latest_number(
            self.root / "interaction_record", "tool_call_*.json", "tool_call_"
        )
        self._retrieval_number = _latest_number(
            self.root / "interaction_record", "retrieval_*.json", "retrieval_"
        )
        self._load_prior_evidence()

    def _load_prior_evidence(self) -> None:
        """Restore hash-verified evidence state for a clarification resume."""
        for path in sorted((self.root / "interaction_record").glob("tool_call_*.json")):
            audit = _read_json(path)
            if audit.get("failure") is not None:
                continue
            if audit.get("record_type") == "ProductAgentCADComparisonToolCall":
                # A resumed interaction must compare against its new live observation.
                continue
            resolved = audit.get("resolved_evidence")
            result_refs = audit.get("result_refs")
            if not isinstance(resolved, Mapping) or not isinstance(result_refs, list):
                raise ProductionGroundingError(f"Successful tool audit is invalid: {path.name}.")
            evidence_id = resolved.get("evidence_id")
            handle = self.handles.get(evidence_id) if isinstance(evidence_id, str) else None
            if handle is None or resolved.get("source_revision") != handle.source_revision:
                raise ProductionGroundingError(
                    f"Previously retrieved evidence is stale: {path.name}."
                )
            if handle.evidence_type == "observation":
                # Every resumed grounding run receives one new live observation.
                continue
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
                    raise ProductionGroundingError(f"Previously retrieved evidence changed: {ref}.")
                record_refs.append(ref)
            if not record_refs:
                raise ProductionGroundingError(
                    f"Successful tool audit has no typed result: {path.name}."
                )
            source_refs = _restored_source_refs(self.root, handle, record_refs)
            self._register_references(source_refs=source_refs, record_refs=record_refs)
            result = _compact_retrieval_result(
                self.root,
                handle,
                {"typed_context_refs": record_refs},
                source_refs=source_refs,
                presentation=self.presentation,
            )
            self.authorized_evidence_refs.update(source_refs)
            self.authorized_evidence_refs.update(record_refs)
            if handle.evidence_id not in self.retrieved_handle_ids:
                self.retrieved_handle_ids.append(handle.evidence_id)
                self.prior_evidence.append({"retrieval_state": "already_retrieved", **dict(result)})
            self.retrieved_results[handle.evidence_id] = result

    async def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Execute one fail-closed ProductAgent tool call."""
        self._tool_call_number += 1
        call_id = f"tool_call_{self._tool_call_number:04d}"
        if self.run_tool_call_count >= _MAX_TOOL_ROUNDS:
            return self._record_failure(
                call_id,
                tool_name,
                arguments,
                "tool_budget_exhausted",
                "The bounded grounding tool budget is exhausted; return a final result.",
            )
        self.run_tool_call_count += 1
        if tool_name == _RETRIEVE_TOOL_NAME:
            return await self._execute_retrieve(call_id, arguments)
        if tool_name == _COMPARE_CAD_SIZE_TOOL_NAME:
            return await self._execute_cad_size_comparison(call_id, arguments)
        return self._record_failure(
            call_id,
            tool_name,
            arguments,
            "malformed_tool_call",
            "Only retrieve and compare_cad_size are authorized.",
        )

    async def _execute_retrieve(
        self,
        call_id: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Retrieve one approved source and retain its neutral typed projection."""
        if set(arguments) != {"evidence_id"}:
            return self._record_failure(
                call_id,
                _RETRIEVE_TOOL_NAME,
                arguments,
                "malformed_tool_call",
                "retrieve requires exactly one evidence_id.",
            )
        evidence_id = arguments.get("evidence_id")
        handle = self.handles.get(evidence_id) if isinstance(evidence_id, str) else None
        if handle is None:
            return self._record_failure(
                call_id,
                _RETRIEVE_TOOL_NAME,
                arguments,
                "unauthorized_evidence_id",
                "The evidence_id is unknown or no longer eligible.",
            )
        if evidence_id in self.retrieved_results:
            result = self.retrieved_results[evidence_id]
            self._record_success(call_id, handle, arguments, result, reused=True)
            return result
        if not _handle_revision_is_current(handle):
            return self._record_failure(
                call_id,
                _RETRIEVE_TOOL_NAME,
                arguments,
                "stale_evidence_id",
                "The approved source changed after the evidence catalog was issued.",
                handle=handle,
            )

        self._retrieval_number += 1
        served_context = (
            self._served_observation_context if handle.evidence_type == "observation" else None
        )
        served: Mapping[str, object] | None = None
        if served_context is None:
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
                call_id,
                _RETRIEVE_TOOL_NAME,
                arguments,
                "retrieval_failed",
                _failure_message(failure),
                handle=handle,
            )
        if handle.evidence_type == "observation":
            self._served_observation_context = dict(served_context)
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
            record_refs = _typed_context_refs(delta)
            self._register_references(source_refs=source_refs, record_refs=record_refs)
            result = _compact_retrieval_result(
                self.root,
                handle,
                delta,
                source_refs=source_refs,
                presentation=self.presentation,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return self._record_failure(
                call_id,
                _RETRIEVE_TOOL_NAME,
                arguments,
                "evidence_processing_failed",
                f"{type(exc).__name__}: {exc}",
                handle=handle,
            )
        self.authorized_evidence_refs.update(source_refs)
        self.authorized_evidence_refs.update(record_refs)
        self.retrieved_results[handle.evidence_id] = result
        if handle.evidence_id not in self.retrieved_handle_ids:
            self.retrieved_handle_ids.append(handle.evidence_id)
        self._record_success(call_id, handle, arguments, result, reused=False)
        return result

    async def _execute_cad_size_comparison(
        self,
        call_id: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Run one PA-requested neutral CAD-to-observation size comparison."""
        required = {"cad_evidence_id", "observation_evidence_id"}
        if set(arguments) != required:
            return self._record_failure(
                call_id,
                _COMPARE_CAD_SIZE_TOOL_NAME,
                arguments,
                "malformed_tool_call",
                "compare_cad_size requires cad_evidence_id and observation_evidence_id.",
            )
        cad_id = arguments.get("cad_evidence_id")
        observation_id = arguments.get("observation_evidence_id")
        cad_handle = self.handles.get(cad_id) if isinstance(cad_id, str) else None
        observation_handle = (
            self.handles.get(observation_id) if isinstance(observation_id, str) else None
        )
        if cad_handle is None or cad_handle.evidence_type != "CAD":
            return self._record_failure(
                call_id,
                _COMPARE_CAD_SIZE_TOOL_NAME,
                arguments,
                "unauthorized_cad_evidence_id",
                "cad_evidence_id must identify an approved CAD catalog entry.",
            )
        if observation_handle is None or observation_handle.evidence_type != "observation":
            return self._record_failure(
                call_id,
                _COMPARE_CAD_SIZE_TOOL_NAME,
                arguments,
                "unauthorized_observation_evidence_id",
                "observation_evidence_id must identify approved observation evidence.",
            )
        if not _handle_revision_is_current(cad_handle) or not _handle_revision_is_current(
            observation_handle
        ):
            return self._record_failure(
                call_id,
                _COMPARE_CAD_SIZE_TOOL_NAME,
                arguments,
                "stale_comparison_input",
                "A comparison input changed after the evidence catalog was issued.",
            )
        cad_result = self.retrieved_results.get(cad_handle.evidence_id)
        observation_result = self.retrieved_results.get(observation_handle.evidence_id)
        if cad_result is None or observation_result is None:
            return self._record_failure(
                call_id,
                _COMPARE_CAD_SIZE_TOOL_NAME,
                arguments,
                "comparison_prerequisite_not_retrieved",
                "Retrieve both comparison inputs before calling compare_cad_size.",
            )
        key = (cad_handle.evidence_id, observation_handle.evidence_id)
        reused = self.comparison_results.get(key)
        if reused is not None:
            self._record_comparison_success(call_id, arguments, reused, reused=True)
            return reused

        try:
            cad_record_ref = _retrieved_record_ref(
                self,
                cad_result,
                expected_record_type="CADMeshRecord",
            )
            segmentation_record_ref = _retrieved_record_ref(
                self,
                observation_result,
                expected_record_type="RGBDSegmentationRecord",
            )
            view = build_product_context_view(
                self.root,
                self.abox,
                attempted_evidence=tuple(self.retrieved_handle_ids),
                assessed_at_ns=time.time_ns(),
            )
            cad_binding = _accepted_binding(
                view,
                cad_record_ref,
                expected_record_type="CADMeshRecord",
            )
            segmentation_binding = _accepted_binding(
                view,
                segmentation_record_ref,
                expected_record_type="RGBDSegmentationRecord",
            )
            correspondence = await asyncio.to_thread(
                associate_segmented_candidate_by_size,
                interaction_root=self.root,
                segmentation_record_path=self.root / segmentation_record_ref,
                cad_record_path=self.root / cad_record_ref,
                correspondence_number=_next_number(
                    self.root,
                    "products/grounding/rgb_d_cad_grounding/correspondence_*",
                ),
            )
            self.abox, _binding = _merge_derived_record(
                self.root,
                self.tbox,
                self.abox,
                correspondence.record_path,
                status=correspondence.CAD_correspondence,
                prerequisite_bindings=(cad_binding, segmentation_binding),
            )
            comparison_ref = correspondence.record_path.relative_to(self.root).as_posix()
            self._register_references(source_refs=(), record_refs=(comparison_ref,))
            self.authorized_evidence_refs.add(comparison_ref)
            result = _cad_size_comparison_projection(
                self,
                correspondence.record,
                comparison_ref=comparison_ref,
                cad_evidence_id=cad_handle.evidence_id,
                observation_evidence_id=observation_handle.evidence_id,
                cad_result=cad_result,
                observation_result=observation_result,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return self._record_failure(
                call_id,
                _COMPARE_CAD_SIZE_TOOL_NAME,
                arguments,
                "cad_size_comparison_failed",
                f"{type(exc).__name__}: {exc}",
            )

        self.comparison_results[key] = result
        self.comparison_bindings[comparison_ref] = _CADComparisonBinding(
            record_ref=comparison_ref,
            cad_record_ref=cad_record_ref,
            segmentation_record_ref=segmentation_record_ref,
        )
        self._record_comparison_success(call_id, arguments, result, reused=False)
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
        record_refs = []
        if isinstance(refs, list):
            record_refs = [
                self.resolve_pa_reference(item) for item in refs if isinstance(item, str)
            ]
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
                    {"ref": ref, "sha256": _sha256_path(self.root / ref)} for ref in record_refs
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
                "resolved_evidence": None
                if handle is None
                else {
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

    def _record_comparison_success(
        self,
        call_id: str,
        arguments: Mapping[str, object],
        result: Mapping[str, object],
        *,
        reused: bool,
    ) -> None:
        """Persist one exact PA-requested comparison without changing its result."""
        comparison_ref = self.resolve_pa_reference(str(result["comparison_ref"]))
        self._persist_tool_call(
            call_id,
            {
                "schema_version": 1,
                "record_type": "ProductAgentCADComparisonToolCall",
                "tool_call_id": call_id,
                "tool_name": _COMPARE_CAD_SIZE_TOOL_NAME,
                "arguments": dict(arguments),
                "reused": reused,
                "result_ref": comparison_ref,
                "result_sha256": _sha256_path(self.root / comparison_ref),
                "result": dict(result),
                "failure": None,
            },
        )

    def _persist_tool_call(self, call_id: str, record: Mapping[str, object]) -> None:
        path = self.root / "interaction_record" / f"{call_id}.json"
        _write_json_exclusive(path, record)
        self.tool_call_refs.append(path.relative_to(self.root).as_posix())

    def resolve_typed_record(self, record_ref: str) -> Mapping[str, object]:
        """Resolve one exact accepted typed binding for target-feature validation."""
        view = build_product_context_view(
            self.root,
            self.abox,
            attempted_evidence=tuple(self.retrieved_handle_ids),
            assessed_at_ns=time.time_ns(),
        )
        matches = [
            binding
            for binding in view.typed_bindings
            if binding.record_ref == record_ref and binding.status == "accepted"
        ]
        if len(matches) != 1:
            raise ProductionGroundingError(
                "record_ref must identify exactly one accepted typed binding."
            )
        binding = matches[0]
        path = (self.root / record_ref).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ProductionGroundingError(
                "Typed target-feature record escapes the interaction."
            ) from exc
        if _sha256_path(path) != binding.record_sha256:
            raise ProductionGroundingError("Typed target-feature record changed after acceptance.")
        return {
            "record_type": binding.record_type,
            "record_sha256": binding.record_sha256,
            "record": _read_json(path),
        }

    def resolve_pa_reference(self, pa_ref: str) -> str:
        """Resolve one interaction-local opaque ref to its canonical host ref."""
        canonical = self._canonical_by_pa_ref.get(pa_ref)
        if canonical is None:
            raise ProductionGroundingError("PA reference is not authorized for this interaction.")
        return canonical

    def project_canonical_reference(self, canonical_ref: str) -> str:
        """Return the already-pinned opaque projection for one canonical ref."""
        pa_ref = self._pa_by_canonical_ref.get(canonical_ref)
        if pa_ref is None:
            raise ProductionGroundingError(
                "Canonical reference has no ProductAgent presentation handle."
            )
        return pa_ref

    def register_canonical_reference(self, canonical_ref: str, *, kind: str) -> None:
        """Authorize one host-created canonical ref for opaque PA citation."""
        self._register_reference(canonical_ref, kind=kind)

    def _register_references(
        self,
        *,
        source_refs: Sequence[str] | set[str],
        record_refs: Sequence[str],
    ) -> None:
        """Register deterministic opaque references for one retrieved result."""
        for canonical_ref in sorted(source_refs):
            self._register_reference(canonical_ref, kind="citation")
        for canonical_ref in record_refs:
            self._register_reference(canonical_ref, kind="typed_record")

    def _register_reference(self, canonical_ref: str, *, kind: str) -> None:
        existing = self._pa_by_canonical_ref.get(canonical_ref)
        if existing is not None:
            if self._canonical_by_pa_ref.get(existing) != canonical_ref:
                raise ProductionGroundingError("Opaque presentation reference collision detected.")
            return
        pa_ref = self.presentation.opaque_reference(canonical_ref, kind=kind)
        previous_canonical = self._canonical_by_pa_ref.setdefault(pa_ref, canonical_ref)
        previous_pa_ref = self._pa_by_canonical_ref.setdefault(canonical_ref, pa_ref)
        if previous_canonical != canonical_ref or previous_pa_ref != pa_ref:
            raise ProductionGroundingError("Opaque presentation reference collision detected.")


class _PAAllocationInvestigation:
    """Execute only PA-requested two-state reachability checks."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        runtime: ProductionProductContextGroundingRuntime,
        investigation: _NativeEvidenceInvestigation,
        interaction_root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        view: ProductContextView,
        allocation_presentation: AllocationPresentationRecord,
        current_state_evidence: AllocationEvidenceEntry,
        desired_state_evidence: AllocationEvidenceEntry,
        current_cad_correspondence_path: Path | None,
        desired_cad_correspondence_path: Path | None,
        target_frame: str,
    ) -> None:
        self.runtime = runtime
        self.investigation = investigation
        self.root = Path(interaction_root).resolve()
        self.tbox = tbox
        self.abox = abox
        self.view = view
        self.allocation_presentation = allocation_presentation
        self.allocation_presentation.assert_unchanged()
        self.current_state_evidence = current_state_evidence
        self.desired_state_evidence = desired_state_evidence
        self.current_cad_correspondence_path = (
            Path(current_cad_correspondence_path).resolve()
            if current_cad_correspondence_path is not None
            else None
        )
        self.desired_cad_correspondence_path = (
            Path(desired_cad_correspondence_path).resolve()
            if desired_cad_correspondence_path is not None
            else None
        )
        self.target_frame = target_frame
        self.reachability_checks: dict[str, ReachabilityCheckRecord] = {}
        self.validations: dict[str, PlanOnlyFeasibilityValidation] = {}
        self.pa_results: list[Mapping[str, object]] = []
        self._location_paths: dict[str, Path] = {}

    async def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Derive selected geometry on demand and check exactly one resource."""
        call_number = _next_number(
            self.root,
            "interaction_record/allocation_tool_call_*.json",
        )
        call_id = f"allocation_tool_call_{call_number:04d}"
        try:
            if tool_name != "check_reachability" or set(arguments) != {
                "resource_symbol",
            }:
                raise ProductionGroundingError("PA allocation tool call fields are invalid.")
            resource_symbol = _allocation_text(
                arguments["resource_symbol"],
                "resource_symbol",
            )
            current_option = self.current_state_evidence
            desired_option = self.desired_state_evidence
            current_handle = current_option.pa_handle
            desired_handle = desired_option.pa_handle
            current_location = await self._location_for(current_option)
            desired_location = await self._location_for(desired_option)
            catalog = candidate_resource_catalog(
                self.abox,
                self.runtime._registry,
                self.runtime._workcell,
            )
            resource = catalog.get(resource_symbol)
            if resource is None:
                raise ProductionGroundingError(
                    "PA allocation resource is not an authorized candidate."
                )
            if resource.get("execution_mode") != "simulation":
                reachability = check_resource_reachability(
                    interaction_root=self.root,
                    tbox=self.tbox,
                    registry=self.runtime._registry,
                    workcell=self.runtime._workcell,
                    resource_symbol=resource_symbol,
                    allocation_presentation=self.allocation_presentation,
                    current_state_evidence_handle=current_handle,
                    desired_state_evidence_handle=desired_handle,
                    current_location_record_path=current_location,
                    desired_location_record_path=desired_location,
                    check_number=_next_number(
                        self.root,
                        "products/grounding/reachability/check_*",
                    ),
                )
                reachability_handle = f"reachability_check_{reachability.check_number:04d}"
                self.reachability_checks[reachability_handle] = reachability
                result = {
                    "reachability_check_ref": reachability_handle,
                    "resource_symbol": reachability.resource_symbol,
                    "status": reachability.status,
                    "current_state_evidence_handle": current_handle,
                    "desired_state_evidence_handle": desired_handle,
                    "current_state": _legacy_reachability_pa_projection(reachability.current_state),
                    "desired_state": _legacy_reachability_pa_projection(reachability.desired_state),
                    "selection_made_by_tool": False,
                }
                _assert_blinded_pa_projection(result, self.investigation.presentation)
                self.pa_results.append(dict(result))
                self._persist_call(
                    call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    result_ref=reachability.record_ref,
                    result_sha256=_sha256_path(reachability.record_path),
                    result=result,
                    failure=None,
                )
                return result
            if (
                self.current_cad_correspondence_path is None
                or self.desired_cad_correspondence_path is None
            ):
                raise ProductionGroundingError(
                    "Cartesian reachability requires both PA-requested CAD comparison records."
                )
            prepared = prepare_cartesian_reachability(
                interaction_root=self.root,
                tbox=self.tbox,
                registry=self.runtime._registry,
                workcell=self.runtime._workcell,
                resource_symbol=resource_symbol,
                allocation_presentation=self.allocation_presentation,
                current_state_evidence_handle=current_handle,
                desired_state_evidence_handle=desired_handle,
                current_location_record_path=current_location,
                desired_location_record_path=desired_location,
                current_cad_correspondence_record_path=(self.current_cad_correspondence_path),
                desired_cad_correspondence_record_path=(self.desired_cad_correspondence_path),
            )
            feasibility_runtime = self.runtime._robot_agent_feasibility_runtime
            if feasibility_runtime is None:
                feasibility_runtime = _UnavailableRobotAgentFeasibilityRuntime()
            validation = await validate_provisional_allocation(
                feasibility_runtime,
                interaction_root=self.root,
                workcell=self.runtime._workcell,
                reachability=prepared,
                validation_number=_next_number(
                    self.root,
                    "resources/*/validation/plan_only_validation_*",
                ),
            )
            reachability = persist_cartesian_reachability(
                interaction_root=self.root,
                prepared=prepared,
                robot_agent_validation_path=validation.record_path,
                check_number=_next_number(
                    self.root,
                    "products/grounding/reachability/check_*",
                ),
            )
            reachability_handle = f"reachability_check_{reachability.check_number:04d}"
            self.reachability_checks[reachability_handle] = reachability
            self.validations[reachability_handle] = validation
            result = {
                "reachability_check_ref": reachability_handle,
                "resource_symbol": reachability.resource_symbol,
                "status": reachability.status,
                "current_state_evidence_handle": current_handle,
                "desired_state_evidence_handle": desired_handle,
                "phase_results": _cartesian_phase_pa_projection(reachability.phase_results),
                "robot_agent_validation_ref": (
                    f"robot_agent_validation_{validation.validation_number:04d}"
                ),
                "feedback": validation.feedback,
                "selection_made_by_tool": False,
            }
            _assert_blinded_pa_projection(result, self.investigation.presentation)
            self.pa_results.append(dict(result))
            self._persist_call(
                call_id,
                tool_name=tool_name,
                arguments=arguments,
                result_ref=reachability.record_ref,
                result_sha256=_sha256_path(reachability.record_path),
                result=result,
                failure=None,
            )
            return result
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            audit_failure = f"{type(exc).__name__}: {exc}"
            message = _allocation_context_feedback(exc)
            result = {
                "resource_symbol": (
                    arguments.get("resource_symbol")
                    if isinstance(arguments.get("resource_symbol"), str)
                    else None
                ),
                "status": "needs_context",
                "phase_results": {
                    "pick": {
                        "status": "needs_context",
                        "fraction": None,
                        "moveit_error_code": None,
                    },
                    "place": {
                        "status": "needs_context",
                        "fraction": None,
                        "moveit_error_code": None,
                    },
                },
                "feedback": message,
                "selection_made_by_tool": False,
            }
            _assert_blinded_pa_projection(result, self.investigation.presentation)
            self.pa_results.append(dict(result))
            self._persist_call(
                call_id,
                tool_name=tool_name,
                arguments=arguments,
                result_ref=None,
                result_sha256=None,
                result=result,
                failure=audit_failure,
            )
            return result

    def pa_result_history(self) -> list[Mapping[str, object]]:
        """Return prior neutral tool results for the next bounded PA attempt."""
        return [dict(result) for result in self.pa_results]

    async def _location_for(self, option: AllocationEvidenceEntry) -> Path:
        key = option.pa_handle
        existing = self._location_paths.get(key)
        if existing is not None:
            return existing
        record_path = self.root / option.record_ref
        if option.record_type == "RobotFrameLocationRecord":
            self._location_paths[key] = record_path
            return record_path
        if (
            option.record_type != "RGBDSegmentationRecord"
            or option.observation_handle is None
            or option.candidate_handle is None
            or option.source_frame is None
        ):
            raise ProductionGroundingError(
                "State evidence cannot provide a neutral candidate location."
            )
        calibration_runtime = self.runtime._camera_to_world_calibration_runtime
        if calibration_runtime is None:
            raise ProductionGroundingError(
                self.runtime._camera_to_world_calibration_unavailable_reason
                or "No approved camera calibration is available."
            )
        segmentation_binding = _accepted_binding(
            self.view,
            option.record_ref,
            expected_record_type="RGBDSegmentationRecord",
        )
        calibration = await asyncio.to_thread(
            calibration_runtime.materialize_camera_to_world_calibration,
            interaction_root=self.root,
            grounding_record_path=record_path,
            source_frame=option.source_frame,
            target_frame=self.target_frame,
            calibration_number=_next_number(
                self.root,
                "products/grounding/rgb_d_cad_grounding/calibration_*",
            ),
        )
        self.abox, calibration_binding = _merge_derived_record(
            self.root,
            self.tbox,
            self.abox,
            calibration.record_path,
            status="accepted",
            prerequisite_bindings=(segmentation_binding,),
        )
        location = await asyncio.to_thread(
            transform_segmentation_candidate_location_to_robot_frame,
            interaction_root=self.root,
            segmentation_record_path=record_path,
            calibration_record_path=calibration.record_path,
            observation_handle=option.observation_handle,
            candidate_handle=option.candidate_handle,
            target_frame=self.target_frame,
            location_number=_next_number(
                self.root,
                "products/grounding/rgb_d_cad_grounding/robot_location_*",
            ),
        )
        self.abox, _location_binding = _merge_derived_record(
            self.root,
            self.tbox,
            self.abox,
            location.record_path,
            status="accepted",
            prerequisite_bindings=(segmentation_binding, calibration_binding),
        )
        self.view = build_product_context_view(
            self.root,
            self.abox,
            attempted_evidence=self.view.attempted_evidence,
            assessed_at_ns=time.time_ns(),
        )
        persist_product_context_view(self.root, self.view)
        self._location_paths[key] = location.record_path
        return location.record_path

    def _persist_call(  # noqa: PLR0913
        self,
        call_id: str,
        *,
        tool_name: str,
        arguments: Mapping[str, object],
        result_ref: str | None,
        result_sha256: str | None,
        result: Mapping[str, object] | None,
        failure: str | None,
    ) -> None:
        path = self.root / "interaction_record" / f"{call_id}.json"
        _write_json_exclusive(
            path,
            {
                "schema_version": 1,
                "record_type": "ProductAgentAllocationToolCall",
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "arguments": dict(arguments),
                "result_ref": result_ref,
                "result_sha256": result_sha256,
                "result": None if result is None else dict(result),
                "failure": failure,
            },
        )
        self.investigation.tool_call_refs.append(path.relative_to(self.root).as_posix())


class _UnavailableRobotAgentFeasibilityRuntime:
    """Return a closed needs-context verdict through the validation boundary."""

    async def validate_plan_only_allocation(
        self,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        del request
        raise RuntimeError("Exact RobotAgent feasibility validation is unavailable.")


class ProductionProductContextGroundingRuntime:
    """Resolve PA context using one native retrieve tool and deterministic providers."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        tbox: TBoxSnapshot,
        document_config: DocumentVLMConfig,
        document_vision_runtime: DocumentVisionRuntime,
        observation_config: ObservationVLMConfig | None = None,
        observation_vision_runtime: ObservationVisionRuntime | None = None,
        camera_to_world_calibration_runtime: CameraToWorldCalibrationRuntime | None = None,
        camera_to_world_calibration_unavailable_reason: str | None = None,
        robot_agent_feasibility_runtime: RobotAgentFeasibilityRuntime | None = None,
        evidence_presentation_order: Sequence[str] | None = None,
        allocation_resource_order: Sequence[str] | None = None,
        allocation_candidate_order: Sequence[str] | None = None,
    ) -> None:
        """Create a runtime pinned to ontology, workcell, and provider authorities."""
        tbox.assert_unchanged()
        self._tbox = tbox
        self._document_config = document_config
        self._document_vision_runtime = document_vision_runtime
        self._observation_config = observation_config
        self._observation_vision_runtime = observation_vision_runtime
        self._camera_to_world_calibration_runtime = camera_to_world_calibration_runtime
        self._camera_to_world_calibration_unavailable_reason = (
            camera_to_world_calibration_unavailable_reason
        )
        self._robot_agent_feasibility_runtime = robot_agent_feasibility_runtime
        self._evidence_presentation_order = (
            None if evidence_presentation_order is None else tuple(evidence_presentation_order)
        )
        self._allocation_resource_order = (
            None if allocation_resource_order is None else tuple(allocation_resource_order)
        )
        self._allocation_candidate_order = (
            None if allocation_candidate_order is None else tuple(allocation_candidate_order)
        )
        self._registry = load_predefined_resource_registry(tbox)
        self._workcell = load_predefined_workcell(tbox, self._registry)
        self._descriptors = _producer_descriptors(
            calibration_available=camera_to_world_calibration_runtime is not None
        )

    def grounding_producer_descriptors(self) -> Sequence[GroundingProducerDescriptor]:
        """Return provider-owned typed-output descriptors."""
        self._validate_authorities(self._tbox)
        return self._descriptors

    async def ground_product_context(  # noqa: C901, PLR0912, PLR0915
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
        evidence_presentation = load_or_create_evidence_presentation(
            interaction_root,
            sources=_approved_evidence_sources(),
            explicit_order=self._evidence_presentation_order,
        )
        handles = _approved_evidence_handles(evidence_presentation)
        investigation = _NativeEvidenceInvestigation(
            runtime=self,
            interaction_root=interaction_root,
            tbox=tbox,
            abox=abox,
            requirement=abox.product_requirement,
            handles=handles,
            presentation=evidence_presentation,
        )
        clarification_evidence = _present_answered_clarification_evidence(
            investigation.root,
            clarification_history,
            investigation,
        )
        validation_feedback_history: list[Mapping[str, object]] = []
        evidence_gap: _GroundingGap | None = None
        required_output_projection = _grounding_readiness_projection()
        final_no_tools_pending = False
        last_failure_message: str | None = None
        for pa_round in range(1, max_pa_turns + 2):
            if pa_round > max_pa_turns and not final_no_tools_pending:
                break
            catalog = _current_evidence_catalog(handles, investigation)
            catalog.extend(clarification_evidence)
            remaining_tool_rounds = max(
                0,
                _MAX_TOOL_ROUNDS - investigation.run_tool_call_count,
            )
            final_no_tools_attempt = final_no_tools_pending or remaining_tool_rounds == 0
            final_no_tools_pending = False
            grounding_tools = (
                []
                if final_no_tools_attempt
                else [_retrieve_tool(handles), _compare_cad_size_tool(handles)]
            )
            try:
                outcome = await propose_and_validate_ontology_grounding(
                    product_agent,
                    interaction_root=interaction_root,
                    tbox=tbox,
                    abox=investigation.abox,
                    workcell=self._workcell,
                    evidence_catalog=catalog,
                    authorized_evidence_refs=investigation.authorized_evidence_refs,
                    tools=grounding_tools,
                    tool_executor=investigation.execute,
                    max_tool_rounds=(0 if final_no_tools_attempt else remaining_tool_rounds),
                    required_output_projection=required_output_projection,
                    validation_feedback_history=tuple(validation_feedback_history),
                    typed_record_resolver=investigation.resolve_typed_record,
                    pa_reference_resolver=investigation.resolve_pa_reference,
                )
            except RuntimeError as exc:
                if not _is_tool_round_exhaustion(exc):
                    raise
                feedback = {
                    "kind": "tool_budget_exhausted",
                    "message": (
                        "The bounded grounding tool budget was exhausted before a final "
                        "structured result was returned."
                    ),
                    "expected_revision": (
                        "Return the best final proposal, clarification, or "
                        "insufficient_evidence result using accumulated evidence and no "
                        "further tool calls."
                    ),
                }
                _append_validation_feedback(validation_feedback_history, feedback)
                last_failure_message = str(feedback["message"])
                if final_no_tools_attempt:
                    return {
                        "grounding_status": "incomplete",
                        "insufficient_evidence": last_failure_message,
                        "tool_call_refs": list(investigation.tool_call_refs),
                    }
                final_no_tools_pending = True
                continue
            except OntologyGroundingError as exc:
                can_retry = not final_no_tools_attempt and (
                    pa_round < max_pa_turns or investigation.run_tool_call_count >= _MAX_TOOL_ROUNDS
                )
                if can_retry:
                    feedback = {
                        "kind": "ontology_proposal_validation_error",
                        "message": str(exc),
                        "expected_revision": (
                            "Return a corrected proposal that satisfies every supplied "
                            "ontology invariant. Do not invent or silently add an "
                            "unsupported assertion."
                        ),
                    }
                    _append_validation_feedback(validation_feedback_history, feedback)
                    last_failure_message = str(feedback["message"])
                    if pa_round >= max_pa_turns:
                        final_no_tools_pending = True
                    evidence_gap = None
                    continue
                if final_no_tools_attempt:
                    return {
                        "grounding_status": "incomplete",
                        "insufficient_evidence": str(exc),
                        "tool_call_refs": list(investigation.tool_call_refs),
                    }
                raise ProductionGroundingError(str(exc)) from exc
            can_retry = not final_no_tools_attempt and (
                pa_round < max_pa_turns or investigation.run_tool_call_count >= _MAX_TOOL_ROUNDS
            )
            if can_retry and pa_round >= max_pa_turns:
                final_no_tools_pending = True
            if isinstance(outcome, OntologyGroundingInterruption):
                if (
                    outcome.kind == "clarification_question"
                    and _clarification_requests_system_choice(
                        outcome.message,
                        required_record_type="RGBDSegmentationRecord",
                        handles=handles,
                    )
                ):
                    if can_retry:
                        feedback = {
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
                        _append_validation_feedback(validation_feedback_history, feedback)
                        last_failure_message = str(feedback["message"])
                        continue
                    return {
                        "grounding_status": "incomplete",
                        "insufficient_evidence": (
                            "PA repeatedly delegated a system-owned grounding choice to the user."
                        ),
                        "tool_call_refs": list(investigation.tool_call_refs),
                    }
                if (
                    outcome.kind == "clarification_question"
                    and not investigation.retrieved_handle_ids
                ):
                    if can_retry:
                        feedback = {
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
                        _append_validation_feedback(validation_feedback_history, feedback)
                        last_failure_message = str(feedback["message"])
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
                    and can_retry
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

            try:
                _proposal_state_candidate_bindings(outcome.proposal)
            except ProductionGroundingError as exc:
                if can_retry:
                    feedback = {
                        "kind": "state_candidate_binding",
                        "message": str(exc),
                        "expected_revision": (
                            "Return exactly one uniquely supported observed candidate in "
                            "each state's state_values, or return insufficient_evidence if "
                            "either state assignment cannot be uniquely supported."
                        ),
                    }
                    _append_validation_feedback(validation_feedback_history, feedback)
                    last_failure_message = str(feedback["message"])
                    evidence_gap = None
                    continue
                return {
                    "grounding_status": "incomplete",
                    "insufficient_evidence": str(exc),
                    "tool_call_refs": list(investigation.tool_call_refs),
                }
            try:
                correspondence_gap = self._validate_pa_state_cad_comparisons(
                    investigation=investigation,
                    proposal=outcome.proposal,
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                correspondence_gap = (
                    "The submitted state assignments are not uniquely supported by "
                    "complete, internally consistent, hash-pinned approved evidence."
                )
            if correspondence_gap is not None:
                if can_retry:
                    feedback = {
                        "kind": "state_evidence_uniqueness",
                        "message": correspondence_gap,
                        "expected_revision": (
                            "Reconsider each state assignment using only PA-selected "
                            "approved evidence and preserve complete citations. If either "
                            "assignment remains non-unique, return insufficient_evidence. "
                            "The validator does not choose or reveal an alternative."
                        ),
                    }
                    _append_validation_feedback(validation_feedback_history, feedback)
                    last_failure_message = str(feedback["message"])
                    evidence_gap = None
                    continue
                return {
                    "grounding_status": "incomplete",
                    "insufficient_evidence": correspondence_gap,
                    "tool_call_refs": list(investigation.tool_call_refs),
                }
            review_catalog = _current_evidence_catalog(handles, investigation)
            review_catalog.extend(clarification_evidence)
            try:
                semantic_review = await review_target_feature_semantics(
                    product_agent,
                    interaction_root=investigation.root,
                    candidate=outcome,
                    product_requirement=investigation.abox.product_requirement,
                    evidence_catalog=review_catalog,
                    pa_reference_projector=investigation.project_canonical_reference,
                )
            except OntologyGroundingError as exc:
                if can_retry:
                    feedback = {
                        "kind": "target_feature_semantic_review_error",
                        "message": str(exc),
                        "expected_revision": (
                            "Return a target feature that can receive a valid separate "
                            "structured semantic consistency review."
                        ),
                    }
                    _append_validation_feedback(validation_feedback_history, feedback)
                    last_failure_message = str(feedback["message"])
                    evidence_gap = None
                    continue
                if final_no_tools_attempt:
                    return {
                        "grounding_status": "incomplete",
                        "insufficient_evidence": str(exc),
                        "tool_call_refs": list(investigation.tool_call_refs),
                    }
                raise ProductionGroundingError(str(exc)) from exc
            if semantic_review.verdict == "incomplete":
                reject_ontology_grounding_candidate(
                    outcome,
                    interaction_root=investigation.root,
                    abox=investigation.abox,
                    semantic_review=semantic_review,
                )
                gap = semantic_review.gap or "Target feature is semantically inconsistent."
                if can_retry:
                    feedback = {
                        "kind": "target_feature_semantic_review",
                        "message": gap,
                        "expected_revision": (
                            "Revise the target feature using only currently retrieved "
                            "evidence. Any revision will pass through deterministic "
                            "validation again; do not infer or repair unsupported facts."
                        ),
                    }
                    _append_validation_feedback(validation_feedback_history, feedback)
                    last_failure_message = str(feedback["message"])
                    evidence_gap = None
                    continue
                return {
                    "grounding_status": "incomplete",
                    "insufficient_evidence": gap,
                    "tool_call_refs": list(investigation.tool_call_refs),
                }
            accepted = commit_ontology_grounding_candidate(
                outcome,
                interaction_root=investigation.root,
                tbox=tbox,
                abox=investigation.abox,
                workcell=self._workcell,
                authorized_evidence_refs=investigation.authorized_evidence_refs,
                semantic_review=semantic_review,
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
                product_agent=product_agent,
                investigation=investigation,
                root=investigation.root,
                tbox=tbox,
                abox=investigation.abox,
                view=accepted_view,
                proposal=accepted.proposal,
                max_pa_turns=max_pa_turns,
            )
            return {
                **completion,
                "ontology_projection_ref": accepted.proposal_path.relative_to(
                    investigation.root
                ).as_posix(),
                "tool_call_refs": list(investigation.tool_call_refs),
            }

        final_gap = evidence_gap or _GroundingGap(
            required_record_type="grounded feature states",
            missing_record_types=("grounded feature states",),
            eligible_evidence_ids=(),
            provider_failure=(
                last_failure_message or "The bounded PA investigation was exhausted."
            ),
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
            if self._observation_config is None or self._observation_vision_runtime is None:
                raise ProductionGroundingError(
                    "Observation candidate review runtime is unavailable."
                )
            segmentation = await asyncio.to_thread(
                segment_preprocessed_observation,
                interaction_root=interaction_root,
                observation_record_path=preprocessing.record_path,
                segmentation_number=_next_number(
                    Path(interaction_root),
                    "products/grounding/rgb_d_cad_grounding/segmentation_*",
                ),
            )
            review = await review_observation_candidates(
                interaction_root=interaction_root,
                segmentation_record_path=segmentation.record_path,
                review_number=_next_number(
                    Path(interaction_root),
                    "products/grounding/rgb_d_cad_grounding/observation_review_*",
                ),
                config=self._observation_config,
                vision_runtime=self._observation_vision_runtime,
            )
            refs = list(delta.get("typed_context_refs", []))
            refs.append(segmentation.record_path.relative_to(interaction_root).as_posix())
            refs.append(review.record_path.relative_to(interaction_root).as_posix())
            delta["typed_context_refs"] = refs
        return delta

    def _validate_pa_state_cad_comparisons(
        self,
        *,
        investigation: _NativeEvidenceInvestigation,
        proposal: OntologyGroundingProposal,
    ) -> str | None:
        """Verify both PA selections against comparison records the PA requested."""
        current_binding, desired_binding = _proposal_state_candidate_bindings(proposal)
        view = build_product_context_view(
            investigation.root,
            investigation.abox,
            attempted_evidence=tuple(investigation.retrieved_handle_ids),
            assessed_at_ns=time.time_ns(),
        )
        for state_name, state_binding in (
            ("current_state", current_binding),
            ("desired_state", desired_binding),
        ):
            gap = _validate_pa_state_cad_comparison(
                investigation,
                view,
                proposal,
                state_name=state_name,
                state_binding=state_binding,
            )
            if gap is not None:
                return gap
        return None

    async def _complete_resource_assignment(  # noqa: C901
        self,
        *,
        product_agent: ProductAgentContextRuntime,
        investigation: _NativeEvidenceInvestigation,
        root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        view: ProductContextView,
        proposal: OntologyGroundingProposal,
        max_pa_turns: int,
    ) -> Mapping[str, object]:
        """Reuse PA-authored state evidence while PA chooses a validated resource."""
        evidence_sources = _allocation_evidence_sources(root, view)
        if not evidence_sources:
            return {
                "grounding_status": "incomplete",
                "insufficient_evidence": (
                    "PA allocation requires at least one approved neutral observation "
                    "candidate or robot-frame location."
                ),
            }
        _register_allocation_references(
            investigation,
            view=view,
            evidence_sources=evidence_sources,
        )
        resource_catalog = candidate_resource_catalog(
            abox,
            self._registry,
            self._workcell,
        )
        process = proposal.target_feature.get("required_process")
        process_iri = process.get("process_iri") if isinstance(process, Mapping) else None
        if not isinstance(process_iri, str):
            raise ProductionGroundingError("Accepted target feature has no process IRI.")
        process_symbol = self._workcell.process_symbol_for_iri(process_iri)
        current_state_iri, desired_state_iri = _feature_state_iris(
            abox,
            tbox=tbox,
            feature_iri=proposal.feature_iri,
        )
        allocation_presentation = load_or_create_allocation_presentation(
            root,
            evidence_presentation=investigation.presentation,
            process_symbol=process_symbol,
            process_iri=process_iri,
            feature_iri=proposal.feature_iri,
            current_state_iri=current_state_iri,
            desired_state_iri=desired_state_iri,
            resources=tuple(
                (
                    symbol,
                    str(resource["resource_iri"]),
                    str(resource["resource_jid"]),
                )
                for symbol, resource in resource_catalog.items()
            ),
            evidence_sources=evidence_sources,
            explicit_resource_order=self._allocation_resource_order,
            explicit_candidate_order=self._allocation_candidate_order,
        )
        current_binding, desired_binding = _proposal_state_candidate_bindings(proposal)
        current_state_evidence = _allocation_entry_for_binding(
            allocation_presentation,
            current_binding,
        )
        desired_state_evidence = _allocation_entry_for_binding(
            allocation_presentation,
            desired_binding,
        )
        current_cad_correspondence_path = _proposal_state_comparison_path(
            investigation,
            proposal,
            state_name="current_state",
        )
        desired_cad_correspondence_path = _proposal_state_comparison_path(
            investigation,
            proposal,
            state_name="desired_state",
        )
        allocation = _PAAllocationInvestigation(
            runtime=self,
            investigation=investigation,
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            view=view,
            allocation_presentation=allocation_presentation,
            current_state_evidence=current_state_evidence,
            desired_state_evidence=desired_state_evidence,
            current_cad_correspondence_path=current_cad_correspondence_path,
            desired_cad_correspondence_path=desired_cad_correspondence_path,
            target_frame=_configured_target_frame(self._workcell),
        )
        validation_feedback: Mapping[str, object] | None = None
        last_selection_ref: str | None = None
        rounds = max(1, max_pa_turns)
        for allocation_round in range(1, rounds + 1):
            allocation_prompt = _pa_allocation_prompt(
                requirement=abox.product_requirement,
                target_feature=_pa_target_feature_projection(
                    proposal.target_feature,
                    investigation,
                ),
                resource_catalog=resource_catalog,
                allocation_presentation=allocation_presentation,
                investigation=investigation,
                current_state_evidence=current_state_evidence,
                desired_state_evidence=desired_state_evidence,
                validation_feedback=validation_feedback,
                reachability_history=allocation.pa_result_history(),
            )
            final_no_tools_attempt = False
            try:
                response = await product_agent.ask_llm_structured(
                    allocation_prompt,
                    response_format=_pa_allocation_response_format(
                        allocation_presentation.resource_order
                    ),
                    tools=[
                        _check_reachability_tool(
                            allocation_presentation.resource_order,
                        )
                    ],
                    tool_executor=allocation.execute,
                    max_tool_rounds=min(rounds - allocation_round + 1, _MAX_TOOL_ROUNDS),
                )
            except RuntimeError as exc:
                if not _is_tool_round_exhaustion(exc):
                    raise
                final_no_tools_attempt = True
                validation_feedback = {
                    "kind": "allocation_tool_budget_exhausted",
                    "message": (
                        "The bounded allocation tool budget was exhausted before a "
                        "final structured result was returned."
                    ),
                    "expected_revision": (
                        "Using the accumulated reachability results, cite an existing "
                        "accepted result or return insufficient_evidence without tools."
                    ),
                }
                try:
                    response = await product_agent.ask_llm_structured(
                        _pa_allocation_prompt(
                            requirement=abox.product_requirement,
                            target_feature=_pa_target_feature_projection(
                                proposal.target_feature,
                                investigation,
                            ),
                            resource_catalog=resource_catalog,
                            allocation_presentation=allocation_presentation,
                            investigation=investigation,
                            current_state_evidence=current_state_evidence,
                            desired_state_evidence=desired_state_evidence,
                            validation_feedback=validation_feedback,
                            reachability_history=allocation.pa_result_history(),
                        ),
                        response_format=_pa_allocation_response_format(
                            allocation_presentation.resource_order
                        ),
                        tools=[],
                        tool_executor=allocation.execute,
                        max_tool_rounds=0,
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as final_exc:
                    return {
                        "grounding_status": "incomplete",
                        "insufficient_evidence": (
                            "Allocation tool exhaustion prevented a final PA result: "
                            f"{type(final_exc).__name__}: {final_exc}"
                        ),
                    }
            try:
                choice, insufficient = _validated_pa_allocation_response(response)
            except ProductionGroundingError as exc:
                if not final_no_tools_attempt:
                    raise
                return {
                    "grounding_status": "incomplete",
                    "insufficient_evidence": (
                        f"Allocation tool exhaustion produced no valid final PA result: {exc}"
                    ),
                }
            if insufficient is not None:
                return {
                    "grounding_status": "incomplete",
                    "insufficient_evidence": insufficient,
                    **(
                        {"resource_selection_ref": last_selection_ref}
                        if last_selection_ref is not None
                        else {}
                    ),
                }
            assert choice is not None
            reachability = allocation.reachability_checks.get(str(choice["reachability_check_ref"]))
            if (
                reachability is None
                or reachability.resource_symbol != choice["provisional_resource_symbol"]
                or reachability.status != "accepted"
            ):
                validation_feedback = {
                    "kind": "uncited_or_rejected_reachability",
                    "message": (
                        "The provisional resource must cite an accepted check_reachability "
                        "result created in this allocation investigation."
                    ),
                    "expected_revision": (
                        "Freely choose a resource, call check_reachability for the "
                        "already-grounded state evidence, and cite one accepted result."
                    ),
                }
                if allocation_round < rounds and not final_no_tools_attempt:
                    continue
                return {
                    "grounding_status": "incomplete",
                    "insufficient_evidence": str(validation_feedback["message"]),
                }

            validation = allocation.validations.get(str(choice["reachability_check_ref"]))
            if reachability.schema_version == 2:
                feasibility_runtime = self._robot_agent_feasibility_runtime
                if feasibility_runtime is None:
                    feasibility_runtime = _UnavailableRobotAgentFeasibilityRuntime()
                validation = await validate_provisional_allocation(
                    feasibility_runtime,
                    interaction_root=root,
                    workcell=self._workcell,
                    reachability=reachability,
                    validation_number=_next_number(
                        root,
                        "resources/*/validation/plan_only_validation_*",
                    ),
                )
            if validation is None or (
                reachability.schema_version == 3 and validation.status != "accepted"
            ):
                validation_feedback = {
                    "kind": "uncited_or_rejected_cartesian_validation",
                    "message": (
                        "The cited reachability result has no accepted live Cartesian "
                        "RobotAgent validation."
                    ),
                    "expected_revision": (
                        "Call check_reachability for another freely chosen resource, "
                        "or report insufficient_evidence."
                    ),
                }
                if allocation_round < rounds and not final_no_tools_attempt:
                    continue
                return {
                    "grounding_status": "incomplete",
                    "insufficient_evidence": str(validation_feedback["message"]),
                }
            selection = persist_pa_resource_selection(
                interaction_root=root,
                tbox=tbox,
                registry=self._registry,
                workcell=self._workcell,
                reachability=reachability,
                allocation_presentation=allocation_presentation,
                robot_agent_validation_path=validation.record_path,
                selection_number=_next_number(
                    root,
                    "products/grounding/resource_selection/selection_*",
                ),
            )
            last_selection_ref = selection.record_ref
            if selection.allocation_status != "accepted":
                validation_feedback = {
                    "kind": "robot_agent_plan_only_validation",
                    "provisional_resource_symbol": (selection.provisional_resource_symbol),
                    "status": selection.robot_agent_validation_status,
                    "feedback": validation.feedback,
                    "expected_revision": (
                        "The validator cannot substitute a robot or target. Use this "
                        "evidence to make another free provisional choice, or report "
                        "insufficient_evidence."
                    ),
                }
                if allocation_round < rounds and not final_no_tools_attempt:
                    continue
                return {
                    "grounding_status": "incomplete",
                    "insufficient_evidence": (
                        validation.feedback or "The PA provisional allocation was not accepted."
                    ),
                    "resource_selection_ref": selection.record_ref,
                }

            assignment = commit_resource_assignment(
                interaction_root=root,
                tbox=tbox,
                registry=self._registry,
                workcell=self._workcell,
                selection=selection,
            )
            final_view = build_product_context_view(
                root,
                assignment.abox,
                attempted_evidence=allocation.view.attempted_evidence,
                assessed_at_ns=time.time_ns(),
            )
            persist_product_context_view(root, final_view)
            return {
                "grounding_status": "complete",
                "resource_selection_ref": selection.record_ref,
                "resource_assignment_delta_count": assignment.abox.delta_count,
                "allocation_label": (
                    "validated Cartesian pick-place allocation"
                    if reachability.schema_version == 3
                    else "validated endpoint-motion allocation"
                ),
            }
        return {
            "grounding_status": "incomplete",
            "insufficient_evidence": "The bounded PA allocation was exhausted.",
            **(
                {"resource_selection_ref": last_selection_ref}
                if last_selection_ref is not None
                else {}
            ),
        }

    def _validate_authorities(self, tbox: TBoxSnapshot) -> None:
        self._tbox.assert_unchanged()
        self._registry.assert_unchanged()
        self._workcell.assert_unchanged()
        tbox.assert_unchanged()
        if tbox.fingerprint != self._tbox.fingerprint:
            raise ProductionGroundingError("Production TBox authority changed.")


def _allocation_evidence_sources(
    root: Path,
    view: ProductContextView,
) -> tuple[AllocationEvidenceSource, ...]:
    """Build one unpartitioned pool from every accepted neutral location source."""
    interaction_root = Path(root).resolve()
    sources: list[AllocationEvidenceSource] = []
    for binding in view.typed_bindings:
        if binding.status != "accepted" or binding.record_type not in {
            "RGBDSegmentationRecord",
            "RobotFrameLocationRecord",
        }:
            continue
        record_path = (interaction_root / binding.record_ref).resolve()
        try:
            record_path.relative_to(interaction_root)
        except ValueError as exc:
            raise ProductionGroundingError(
                "Neutral allocation evidence escapes the interaction."
            ) from exc
        if _sha256_path(record_path) != binding.record_sha256:
            raise ProductionGroundingError(
                "Neutral allocation evidence changed before presentation."
            )
        record = _read_json(record_path)
        if binding.record_type == "RGBDSegmentationRecord":
            sources.extend(_segmentation_evidence_sources(binding, record))
        else:
            sources.append(_robot_location_evidence_source(binding, record))
    return tuple(sources)


def _proposal_state_candidate_bindings(
    proposal: OntologyGroundingProposal,
) -> tuple[_StateCandidateBinding, _StateCandidateBinding]:
    """Resolve the one neutral candidate the PA authored for each feature state."""
    by_state: dict[str, list[_StateCandidateBinding]] = {
        "current_state": [],
        "desired_state": [],
    }
    for value in proposal.resolved_state_values:
        state = value.get("state")
        name = value.get("name")
        value_ref = value.get("value_ref")
        if (
            state not in by_state
            or value.get("record_type") != "RGBDSegmentationRecord"
            or not isinstance(name, str)
            or not name
            or not isinstance(value_ref, Mapping)
        ):
            continue
        record_ref = value_ref.get("record_ref")
        field_path = value_ref.get("field_path")
        parts = field_path.split("/") if isinstance(field_path, str) else []
        if (
            not isinstance(record_ref, str)
            or not record_ref
            or len(parts) != 5
            or parts[0] != ""
            or parts[1] != "cameras"
            or not parts[2].isdigit()
            or parts[3] != "candidates"
            or not parts[4].isdigit()
        ):
            continue
        by_state[str(state)].append(
            _StateCandidateBinding(
                state=str(state),
                name=name,
                record_ref=record_ref,
                field_path=field_path,
            )
        )
    for state, bindings in by_state.items():
        if len(bindings) != 1:
            raise ProductionGroundingError(
                f"{state} must select exactly one supplied neutral candidate."
            )
    current = by_state["current_state"][0]
    desired = by_state["desired_state"][0]
    if (
        current.record_ref == desired.record_ref
        and current.field_path == desired.field_path
        and current.name != desired.name
    ):
        raise ProductionGroundingError(
            "One candidate cannot support incompatible current_state and desired_state value names."
        )
    return current, desired


def _allocation_entry_for_binding(
    presentation: AllocationPresentationRecord,
    binding: _StateCandidateBinding,
) -> AllocationEvidenceEntry:
    """Map one accepted PA state binding into the host-only allocation record."""
    matches = [
        entry
        for entry in presentation.evidence_entries
        if entry.record_ref == binding.record_ref
        and entry.field_path == binding.field_path
        and entry.record_type == "RGBDSegmentationRecord"
    ]
    if len(matches) != 1:
        raise ProductionGroundingError(f"{binding.state} candidate is unavailable to allocation.")
    return matches[0]


def _register_allocation_references(
    investigation: _NativeEvidenceInvestigation,
    *,
    view: ProductContextView,
    evidence_sources: Sequence[AllocationEvidenceSource],
) -> None:
    """Make accepted neutral records re-projectable without exposing their refs."""
    source_record_refs = {source.record_ref for source in evidence_sources}
    for record_ref in source_record_refs:
        investigation.register_canonical_reference(record_ref, kind="typed_record")
    for binding in view.typed_bindings:
        if binding.record_ref not in source_record_refs:
            continue
        for evidence_ref in binding.evidence_refs:
            investigation.register_canonical_reference(evidence_ref, kind="citation")


def _segmentation_evidence_sources(
    binding: TypedContextBinding,
    record: Mapping[str, object],
) -> tuple[AllocationEvidenceSource, ...]:
    """Project all candidates from one neutral segmentation record."""
    cameras = record.get("cameras")
    if (
        record.get("schema_version") != 2
        or record.get("record_type") != "RGBDSegmentationRecord"
        or not isinstance(cameras, list)
    ):
        raise ProductionGroundingError("Neutral segmentation record is invalid.")
    sources: list[AllocationEvidenceSource] = []
    for camera_index, camera in enumerate(cameras):
        if not isinstance(camera, Mapping):
            raise ProductionGroundingError("Neutral segmentation view is invalid.")
        observation_handle = camera.get("observation_handle")
        source_frame = camera.get("frame")
        candidates = camera.get("candidates")
        if (
            not isinstance(observation_handle, str)
            or not observation_handle
            or not isinstance(source_frame, str)
            or not source_frame
            or not isinstance(candidates, list)
        ):
            raise ProductionGroundingError("Neutral segmentation view is invalid.")
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ProductionGroundingError("Neutral segmentation candidate is invalid.")
            candidate_handle = candidate.get("candidate_handle")
            if not isinstance(candidate_handle, str) or not candidate_handle:
                raise ProductionGroundingError("Neutral segmentation candidate handle is invalid.")
            field_path = f"/cameras/{camera_index}/candidates/{candidate_index}"
            canonical_key = f"{binding.record_ref}#{field_path}"
            sources.append(
                AllocationEvidenceSource(
                    canonical_key=canonical_key,
                    record_type="RGBDSegmentationRecord",
                    record_ref=binding.record_ref,
                    record_sha256=binding.record_sha256,
                    field_path=field_path,
                    observation_handle=observation_handle,
                    candidate_handle=candidate_handle,
                    source_frame=source_frame,
                    neutral_projection={
                        "visual_region_available": True,
                        "point_count": candidate.get("point_count"),
                        "pixel_bounds_uv": candidate.get("pixel_bounds_uv"),
                        "bounds_m": candidate.get("bounds_m"),
                        "centroid_m": candidate.get("centroid_m"),
                        "depth_range_m": candidate.get("depth_range_m"),
                    },
                )
            )
    return tuple(sources)


def _robot_location_evidence_source(
    binding: TypedContextBinding,
    record: Mapping[str, object],
) -> AllocationEvidenceSource:
    """Expose an existing numeric location only as a neutral opaque choice."""
    candidate_reference = record.get("candidate_reference")
    observation_handle = (
        candidate_reference.get("observation_handle")
        if isinstance(candidate_reference, Mapping)
        else None
    )
    candidate_handle = (
        candidate_reference.get("candidate_handle")
        if isinstance(candidate_reference, Mapping)
        else None
    )
    source_frame = record.get("source_frame")
    if (
        record.get("schema_version") != 2
        or record.get("record_type") != "RobotFrameLocationRecord"
        or not isinstance(source_frame, str)
        or not source_frame
        or (observation_handle is not None and not isinstance(observation_handle, str))
        or (candidate_handle is not None and not isinstance(candidate_handle, str))
    ):
        raise ProductionGroundingError("Neutral robot-frame location record is invalid.")
    field_path = "/translated_location_m"
    return AllocationEvidenceSource(
        canonical_key=f"{binding.record_ref}#{field_path}",
        record_type="RobotFrameLocationRecord",
        record_ref=binding.record_ref,
        record_sha256=binding.record_sha256,
        field_path=field_path,
        observation_handle=observation_handle,
        candidate_handle=candidate_handle,
        source_frame=source_frame,
        neutral_projection={
            "visual_region_available": bool(observation_handle and candidate_handle),
            "location_record_available": True,
        },
    )


def _accepted_binding(
    view: ProductContextView,
    record_ref: str,
    *,
    expected_record_type: str,
) -> TypedContextBinding:
    matches = [
        binding
        for binding in view.typed_bindings
        if binding.record_ref == record_ref
        and binding.record_type == expected_record_type
        and binding.status == "accepted"
    ]
    if len(matches) != 1:
        raise ProductionGroundingError("PA state value is not one exact accepted typed binding.")
    return matches[0]


def _feature_state_iris(
    abox: ABoxSnapshot,
    *,
    tbox: TBoxSnapshot,
    feature_iri: str,
) -> tuple[str, str]:
    """Resolve both stable semantic state IRIs from accepted ontology assertions."""
    feature = URIRef(feature_iri)
    current = tuple(
        str(value)
        for value in abox.graph.objects(
            feature,
            URIRef(f"{tbox.ppr_namespace}hascurrentstate"),
        )
    )
    desired = tuple(
        str(value)
        for value in abox.graph.objects(
            feature,
            URIRef(f"{tbox.ppr_namespace}hasdesiredstate"),
        )
    )
    if len(current) != 1 or len(desired) != 1:
        raise ProductionGroundingError(
            "Accepted feature must identify exactly one currentstate and desiredstate."
        )
    return current[0], desired[0]


def _pa_target_feature_projection(
    target_feature: Mapping[str, object],
    investigation: _NativeEvidenceInvestigation,
) -> Mapping[str, object]:
    """Re-project canonical target-feature citations before allocation review."""

    def project(value: object, *, parent_key: str | None = None) -> object:
        if isinstance(value, Mapping):
            result: dict[str, object] = {}
            for raw_key, item in value.items():
                key = str(raw_key)
                if key == "evidence_refs" and isinstance(item, list):
                    result[key] = [
                        investigation.project_canonical_reference(str(ref)) for ref in item
                    ]
                elif key == "record_ref" and parent_key == "value_ref" and isinstance(item, str):
                    result[key] = investigation.project_canonical_reference(item)
                else:
                    result[key] = project(item, parent_key=key)
            return result
        if isinstance(value, list):
            return [project(item, parent_key=parent_key) for item in value]
        return value

    projected = project(target_feature)
    if not isinstance(projected, Mapping):
        raise ProductionGroundingError("Target-feature PA projection is invalid.")
    _assert_blinded_pa_projection(projected, investigation.presentation)
    return projected


def _legacy_reachability_pa_projection(
    state: StateReachEvidence,
) -> Mapping[str, object]:
    """Return the unchanged physical-mode coarse reach evidence."""
    return {
        "state_name": state.state_name,
        "evidence_handle": state.evidence_handle,
        "translation_m": list(state.translation_m),
        "planar_distance_from_reach_origin_m": (state.planar_distance_from_reach_origin_m),
        "distance_from_reach_origin_m": state.distance_from_reach_origin_m,
        "in_workspace": state.in_workspace,
        "in_gripper_reach": state.in_gripper_reach,
        "reachable": state.reachable,
        "verdicts": list(state.verdicts),
    }


def _cartesian_phase_pa_projection(value: object) -> Mapping[str, object]:
    """Expose neutral Cartesian phase evidence without runtime identities."""
    if not isinstance(value, Mapping):
        raise ProductionGroundingError("Cartesian phase evidence is unavailable.")
    result: dict[str, object] = {}
    for phase in ("pick", "place"):
        item = value.get(phase)
        if not isinstance(item, Mapping):
            raise ProductionGroundingError(f"Cartesian {phase} phase evidence is unavailable.")
        result[phase] = {
            "status": item.get("status"),
            "fraction": item.get("fraction"),
            "moveit_error_code": item.get("moveit_error_code"),
            "waypoint_roles": item.get("waypoint_roles"),
            "message": item.get("message"),
        }
    return result


def _allocation_context_feedback(exc: Exception) -> str:
    """Return generic PA feedback without evidence identities or modalities."""
    del exc
    return (
        "The approved evidence needed to evaluate the submitted resource choice is "
        "unavailable, stale, or internally inconsistent."
    )


def _pa_allocation_prompt(  # noqa: PLR0913
    *,
    requirement: str,
    target_feature: Mapping[str, object],
    resource_catalog: Mapping[str, Mapping[str, str]],
    allocation_presentation: AllocationPresentationRecord,
    investigation: _NativeEvidenceInvestigation,
    current_state_evidence: AllocationEvidenceEntry,
    desired_state_evidence: AllocationEvidenceEntry,
    validation_feedback: Mapping[str, object] | None,
    reachability_history: Sequence[Mapping[str, object]] = (),
) -> str:
    prompt_input: dict[str, object] = {
        "exact_requirement": requirement,
        "grounded_target_feature": dict(target_feature),
        "approved_retrieved_evidence": _current_evidence_catalog((), investigation),
        "grounded_state_evidence": {
            "current_state": _allocation_evidence_prompt_projection(
                current_state_evidence,
                investigation,
            ),
            "desired_state": _allocation_evidence_prompt_projection(
                desired_state_evidence,
                investigation,
            ),
        },
        "capable_resource_catalog": [
            {"resource_symbol": symbol, **dict(resource_catalog[symbol])}
            for symbol in allocation_presentation.resource_order
        ],
        "presentation_order_has_no_priority": True,
    }
    if reachability_history:
        prompt_input["reachability_results_so_far"] = [
            dict(result) for result in reachability_history
        ]
    if validation_feedback is not None:
        prompt_input["previous_validation_feedback"] = dict(validation_feedback)
    return (
        "Act as the ProductAgent allocation authority for the already-grounded "
        "feature transformation. The current_state and desired_state evidence was "
        "already authored and accepted in grounded_target_feature; do not reselect or "
        "replace either value. Select one provisional resource according to the "
        "requirement, grounded state evidence, and reachability results. "
        "Presentation order is randomized and is never priority. Use check_reachability "
        "for any resource you consider. In simulation, that tool asks the exact "
        "RobotAgent and live MoveIt model for strict, collision-aware Cartesian "
        "paths using only the already-grounded state evidence; it never changes a state "
        "assignment or chooses a robot. Return a "
        "provisional_resource_symbol only with the exact ref of an accepted check "
        "created during this investigation. RobotAgent feedback is evidence for a "
        "new free choice: do not ask the validator to substitute a resource or "
        "target. Return insufficient_evidence if no supported choice remains. Do "
        "not claim that manufacturing or motion has executed.\n\n"
        f"Allocation input:\n{json.dumps(prompt_input, indent=2, ensure_ascii=False)}"
    )


def _allocation_evidence_prompt_projection(
    entry: AllocationEvidenceEntry,
    investigation: _NativeEvidenceInvestigation,
) -> dict[str, object]:
    """Link one neutral segmentation choice to its prior blinded evidence."""
    projection = entry.prompt_projection()
    if entry.record_type == "RGBDSegmentationRecord":
        if entry.observation_handle is None or entry.candidate_handle is None:
            raise ProductionGroundingError(
                "RGBDSegmentationRecord allocation evidence has no candidate identity."
            )
        projection.update(
            {
                "observation_handle": entry.observation_handle,
                "candidate_handle": entry.candidate_handle,
                # Reuse the earlier retrieval projection so PA can join evidence
                # without receiving the canonical interaction path.
                "candidate_value_ref": {
                    "record_ref": investigation.project_canonical_reference(entry.record_ref),
                    "field_path": entry.field_path,
                },
            }
        )
    _assert_blinded_pa_projection(projection, investigation.presentation)
    return projection


def _check_reachability_tool(
    resource_symbols: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "check_reachability",
            "description": (
                "Evaluate the complete grounded pick-and-place path for one "
                "explicitly chosen resource without selecting or commanding it."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["resource_symbol"],
                "properties": {
                    "resource_symbol": {
                        "type": "string",
                        "enum": list(resource_symbols),
                    },
                },
            },
        },
    }


def _pa_allocation_response_format(
    resource_symbols: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "name": "spec2primitives_pa_resource_allocation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["result"],
            "properties": {
                "result": {
                    "anyOf": [
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "provisional_resource_symbol",
                                "reachability_check_ref",
                            ],
                            "properties": {
                                "provisional_resource_symbol": {
                                    "type": "string",
                                    "enum": list(resource_symbols),
                                },
                                "reachability_check_ref": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                            },
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["insufficient_evidence"],
                            "properties": {
                                "insufficient_evidence": {
                                    "type": "string",
                                    "minLength": 1,
                                }
                            },
                        },
                    ]
                }
            },
        },
    }


def _validated_pa_allocation_response(
    value: object,
) -> tuple[Mapping[str, str] | None, str | None]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"result"}
        or not isinstance(value["result"], Mapping)
    ):
        raise ProductionGroundingError("PA allocation result wrapper is invalid.")
    result = value["result"]
    if set(result) == {"insufficient_evidence"}:
        message = _allocation_text(
            result["insufficient_evidence"],
            "insufficient_evidence",
        )
        return None, message
    if set(result) != {
        "provisional_resource_symbol",
        "reachability_check_ref",
    }:
        raise ProductionGroundingError("PA allocation result fields are invalid.")
    return (
        {
            "provisional_resource_symbol": _allocation_text(
                result["provisional_resource_symbol"],
                "provisional_resource_symbol",
            ),
            "reachability_check_ref": _allocation_text(
                result["reachability_check_ref"],
                "reachability_check_ref",
            ),
        },
        None,
    )


def _allocation_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionGroundingError(f"PA allocation {field} is invalid.")
    return value


def _producer_descriptors(
    *, calibration_available: bool
) -> tuple[GroundingProducerDescriptor, ...]:
    return (
        GroundingProducerDescriptor.from_mapping(
            {
                "provider_id": _DOCUMENT_PRODUCER,
                "description": "Read every page of one approved document.",
                "accepted_evidence_types": ["document"],
                "produced_record_types": ["DocumentOverviewRecord"],
                "prerequisites": {"DocumentOverviewRecord": []},
                "availability": True,
                "estimated_cost": 2,
            }
        ),
        GroundingProducerDescriptor.from_mapping(
            {
                "provider_id": _GEOMETRY_PRODUCER,
                "description": "Measure approved CAD evidence.",
                "accepted_evidence_types": ["CAD"],
                "produced_record_types": ["CADMeshRecord"],
                "prerequisites": {"CADMeshRecord": []},
                "availability": True,
                "estimated_cost": 2,
            }
        ),
        GroundingProducerDescriptor.from_mapping(
            {
                "provider_id": _GEOMETRY_PRODUCER,
                "description": "Measure approved live observation evidence.",
                "accepted_evidence_types": ["observation"],
                "produced_record_types": [
                    "ColoredPointCloudSetRecord",
                    "RGBDSegmentationRecord",
                    "ObservationCandidateReview",
                ],
                "prerequisites": {
                    "ColoredPointCloudSetRecord": [],
                    "RGBDSegmentationRecord": ["ColoredPointCloudSetRecord"],
                    "ObservationCandidateReview": ["RGBDSegmentationRecord"],
                },
                "availability": True,
                "estimated_cost": 2,
            }
        ),
        GroundingProducerDescriptor.from_mapping(
            {
                "provider_id": _GEOMETRY_PRODUCER,
                "description": "Derive physical context from accepted typed records.",
                "accepted_evidence_types": ["existing_record"],
                "produced_record_types": [
                    "CADSizeCorrespondenceRecord",
                    "CADPoseEstimationRecord",
                    "RobotFrameLocationRecord",
                ],
                "prerequisites": {
                    "CADSizeCorrespondenceRecord": ["CADMeshRecord", "RGBDSegmentationRecord"],
                    "CADPoseEstimationRecord": ["CADSizeCorrespondenceRecord"],
                    "RobotFrameLocationRecord": [
                        "RGBDSegmentationRecord",
                        "CameraToRobotCalibrationRecord",
                    ],
                },
                "availability": True,
                "estimated_cost": 2,
            }
        ),
        GroundingProducerDescriptor.from_mapping(
            {
                "provider_id": _CALIBRATION_PRODUCER,
                "description": "Materialize approved camera calibration.",
                "accepted_evidence_types": ["existing_record"],
                "produced_record_types": ["CameraToRobotCalibrationRecord"],
                "prerequisites": {"CameraToRobotCalibrationRecord": ["RGBDSegmentationRecord"]},
                "availability": calibration_available,
                "estimated_cost": 0,
            }
        ),
    )


def _approved_evidence_sources() -> tuple[tuple[str | None, str, str], ...]:
    """Return canonical host-only evidence authorities before presentation."""
    sources = [
        (
            context_ref,
            evidence_type,
            _static_source_revision(context_ref, evidence_type),
        )
        for context_ref, evidence_type in approved_context_ref_evidence_types().items()
    ]
    sources.append((None, "observation", "fresh_on_call"))
    return tuple(sources)


def _approved_evidence_handles(
    presentation: EvidencePresentationRecord,
) -> tuple[_EvidenceHandle, ...]:
    """Build only opaque PA handles from the immutable host presentation."""
    presentation.assert_unchanged()
    return tuple(
        _EvidenceHandle(
            evidence_id=entry.pa_handle,
            evidence_type=entry.evidence_type,
            display_name=entry.pa_handle,
            context_ref=entry.canonical_ref,
            source_revision=entry.source_revision,
        )
        for entry in presentation.entries
    )


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
                "properties": {
                    "evidence_id": {
                        "type": "string",
                        "enum": [handle.evidence_id for handle in handles],
                    }
                },
            },
        },
    }


def _compare_cad_size_tool(handles: Sequence[_EvidenceHandle]) -> dict[str, Any]:
    """Describe the neutral CAD-size comparison available to the ProductAgent."""
    cad_ids = [handle.evidence_id for handle in handles if handle.evidence_type == "CAD"]
    observation_ids = [
        handle.evidence_id for handle in handles if handle.evidence_type == "observation"
    ]
    return {
        "type": "function",
        "function": {
            "name": _COMPARE_CAD_SIZE_TOOL_NAME,
            "description": (
                "Compare one previously retrieved CAD mesh with candidates in one "
                "previously retrieved RGB-D observation. Returns measurements and "
                "tolerance results only; it does not identify or select a candidate."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cad_evidence_id", "observation_evidence_id"],
                "properties": {
                    "cad_evidence_id": {
                        "type": "string",
                        "enum": cad_ids,
                    },
                    "observation_evidence_id": {
                        "type": "string",
                        "enum": observation_ids,
                    },
                },
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
        return handle.source_revision == _static_source_revision(
            handle.context_ref, handle.evidence_type
        )
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
    presentation: EvidencePresentationRecord,
) -> Mapping[str, object]:
    canonical_record_refs = _typed_context_refs(delta)
    records = [_read_json(root / ref) for ref in canonical_record_refs]
    record_refs = [
        presentation.opaque_reference(ref, kind="typed_record") for ref in canonical_record_refs
    ]
    evidence_refs = [
        presentation.opaque_reference(ref, kind="citation") for ref in sorted(source_refs)
    ]
    evidence_refs.extend(record_refs)
    if handle.evidence_type == "document":
        overview = records[-1].get("overview")
        if not isinstance(overview, Mapping):
            raise ProductionGroundingError("Document overview projection is invalid.")
        result = {
            "evidence_type": "document",
            "evidence_handle": handle.evidence_id,
            "evidence_refs": evidence_refs,
            "record_refs": record_refs,
            "summary": _blind_source_identities(overview.get("summary"), presentation),
            "pages": _document_page_projection(overview, presentation),
            "visual_observations": _blind_source_identities(
                overview.get("observations"), presentation
            ),
            "uncertainty": _blind_source_identities(overview.get("uncertainty"), presentation),
        }
    elif handle.evidence_type == "CAD":
        record = records[0]
        result = {
            "evidence_type": "CAD",
            "evidence_handle": handle.evidence_id,
            "context_ref": handle.context_ref,
            "evidence_refs": evidence_refs,
            "record_refs": record_refs,
            "coordinate_frame": record.get("coordinate_frame"),
            "units": record.get("stored_units"),
            "triangle_count": record.get("triangle_count"),
            "vertex_count": record.get("vertex_count"),
            "dimensions": record.get("bounds_m"),
            "centroid": record.get("vertex_centroid_m"),
        }
    else:
        segmentations = [
            record for record in records if record.get("record_type") == "RGBDSegmentationRecord"
        ]
        reviews = [
            record
            for record in records
            if record.get("record_type") == "ObservationCandidateReview"
        ]
        if len(segmentations) != 1 or len(reviews) != 1:
            raise ProductionGroundingError(
                "Observation retrieval requires one segmentation and one candidate review."
            )
        segmentation = segmentations[0]
        review = reviews[0]
        segmentation_index = records.index(segmentation)
        result = {
            "evidence_type": "observation",
            "evidence_handle": handle.evidence_id,
            "evidence_refs": evidence_refs,
            "record_refs": record_refs,
            "segmentation": {
                "candidate_state": segmentation.get("candidate_state"),
                "candidate_count": segmentation.get("candidate_count"),
                "views": _neutral_candidate_views(
                    segmentation,
                    segmentation_record_ref=record_refs[segmentation_index],
                    review=review,
                ),
            },
        }
    _assert_blinded_pa_projection(result, presentation)
    return result


def _retrieved_record_ref(
    investigation: _NativeEvidenceInvestigation,
    result: Mapping[str, object],
    *,
    expected_record_type: str,
) -> str:
    """Resolve one exact typed record from a retained PA retrieval result."""
    refs = result.get("record_refs")
    if not isinstance(refs, list):
        raise ProductionGroundingError("Retrieved evidence has no typed record refs.")
    matches: list[str] = []
    for pa_ref in refs:
        if not isinstance(pa_ref, str):
            continue
        canonical_ref = investigation.resolve_pa_reference(pa_ref)
        record = _read_json(investigation.root / canonical_ref)
        if record.get("record_type") == expected_record_type:
            matches.append(canonical_ref)
    if len(matches) != 1:
        raise ProductionGroundingError(
            f"Retrieved evidence must contain exactly one {expected_record_type}."
        )
    return matches[0]


def _comparison_candidate_index(
    views: Sequence[object],
) -> tuple[
    dict[tuple[str, str], Mapping[str, object]],
    dict[tuple[str, str], int],
]:
    """Index PA-facing candidate refs in neutral observation order."""
    candidate_refs: dict[tuple[str, str], Mapping[str, object]] = {}
    candidate_order: dict[tuple[str, str], int] = {}
    for view in views:
        if not isinstance(view, Mapping):
            raise ProductionGroundingError("Observation comparison view is invalid.")
        observation_handle = view.get("observation_handle")
        candidates = view.get("candidates")
        if not isinstance(observation_handle, str) or not isinstance(candidates, list):
            raise ProductionGroundingError("Observation comparison candidates are invalid.")
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ProductionGroundingError("Observation comparison candidate is invalid.")
            candidate_handle = candidate.get("candidate_handle")
            candidate_value_ref = candidate.get("candidate_value_ref")
            if not isinstance(candidate_handle, str) or not isinstance(
                candidate_value_ref, Mapping
            ):
                raise ProductionGroundingError("Observation candidate reference is invalid.")
            candidate_key = (observation_handle, candidate_handle)
            candidate_order[candidate_key] = len(candidate_order)
            candidate_refs[candidate_key] = candidate_value_ref
    return candidate_refs, candidate_order


def _project_comparison_candidates(
    ranked: Sequence[object],
    *,
    candidate_refs: Mapping[tuple[str, str], Mapping[str, object]],
    candidate_order: Mapping[tuple[str, str], int],
    comparison_status: object,
) -> list[dict[str, object]]:
    """Project candidates while hiding an ambiguous comparison's audit ranking."""
    candidates_to_project: list[Mapping[str, object]] = []
    for candidate in ranked:
        if not isinstance(candidate, Mapping):
            raise ProductionGroundingError("Ranked CAD comparison candidate is invalid.")
        candidate_key = (
            str(candidate.get("observation_handle")),
            str(candidate.get("candidate_handle")),
        )
        if candidate_key not in candidate_refs:
            raise ProductionGroundingError(
                "Ranked CAD comparison candidate is absent from observation evidence."
            )
        if comparison_status != "ambiguous" or candidate.get("within_size_tolerance") is True:
            candidates_to_project.append(candidate)
    if comparison_status == "ambiguous":
        candidates_to_project.sort(
            key=lambda candidate: candidate_order[
                (
                    str(candidate.get("observation_handle")),
                    str(candidate.get("candidate_handle")),
                )
            ]
        )

    projected_candidates: list[dict[str, object]] = []
    for candidate in candidates_to_project:
        observation_handle = candidate.get("observation_handle")
        candidate_handle = candidate.get("candidate_handle")
        value_ref = candidate_refs[(str(observation_handle), str(candidate_handle))]
        projected_candidate = {
            "observation_handle": observation_handle,
            "candidate_handle": candidate_handle,
            "candidate_value_ref": dict(value_ref),
            "observed_dimensions_m": candidate.get("observed_dimensions_m"),
            "within_size_tolerance": candidate.get("within_size_tolerance"),
            "candidate_center_m": candidate.get("candidate_center_m"),
        }
        if comparison_status != "ambiguous":
            projected_candidate.update(
                {
                    "rank": candidate.get("rank"),
                    "dimension_errors": candidate.get("dimension_errors"),
                    "mean_dimension_error": candidate.get("mean_dimension_error"),
                }
            )
        projected_candidates.append(projected_candidate)
    return projected_candidates


def _cad_size_comparison_projection(  # noqa: PLR0913
    investigation: _NativeEvidenceInvestigation,
    record: Mapping[str, object],
    *,
    comparison_ref: str,
    cad_evidence_id: str,
    observation_evidence_id: str,
    cad_result: Mapping[str, object],
    observation_result: Mapping[str, object],
) -> Mapping[str, object]:
    """Project objective candidate measurements without making a semantic choice."""
    cad = record.get("CAD")
    ranked = record.get("ranked_candidates")
    segmentation = observation_result.get("segmentation")
    views = segmentation.get("views") if isinstance(segmentation, Mapping) else None
    if not isinstance(cad, Mapping) or not isinstance(ranked, list) or not isinstance(views, list):
        raise ProductionGroundingError("CAD size comparison projection is invalid.")

    candidate_refs, candidate_order = _comparison_candidate_index(views)
    comparison_status = record.get("CAD_correspondence")
    projected_candidates = _project_comparison_candidates(
        ranked,
        candidate_refs=candidate_refs,
        candidate_order=candidate_order,
        comparison_status=comparison_status,
    )

    comparison_pa_ref = investigation.project_canonical_reference(comparison_ref)
    evidence_refs: list[str] = []
    for source in (cad_result, observation_result):
        refs = source.get("evidence_refs")
        if isinstance(refs, list):
            evidence_refs.extend(ref for ref in refs if isinstance(ref, str))
    evidence_refs.append(comparison_pa_ref)
    parameters = record.get("parameters")
    result = {
        "evidence_type": "CAD_size_comparison",
        "cad_evidence_handle": cad_evidence_id,
        "observation_evidence_handle": observation_evidence_id,
        "comparison_ref": comparison_pa_ref,
        "evidence_refs": list(dict.fromkeys(evidence_refs)),
        "record_refs": [comparison_pa_ref],
        "compared_dimensions_m": cad.get("compared_dimensions_m"),
        "dimension_error_limit": (
            parameters.get("dimension_error_limit") if isinstance(parameters, Mapping) else None
        ),
        "CAD_correspondence": comparison_status,
        "plausible_candidate_count": sum(
            candidate.get("within_size_tolerance") is True for candidate in projected_candidates
        ),
        "selection_made_by_tool": False,
    }
    result["plausible_candidates" if comparison_status == "ambiguous" else "ranked_candidates"] = (
        projected_candidates
    )
    _assert_blinded_pa_projection(result, investigation.presentation)
    return result


def _typed_context_refs(delta: Mapping[str, object]) -> list[str]:
    refs = delta.get("typed_context_refs")
    if (
        not isinstance(refs, list)
        or not refs
        or not all(isinstance(ref, str) and ref for ref in refs)
    ):
        raise ProductionGroundingError("Evidence producer returned no typed record.")
    return [str(ref) for ref in refs]


def _document_page_projection(
    overview: Mapping[str, object],
    presentation: EvidencePresentationRecord,
) -> list[dict[str, object]]:
    """Return document pages without filenames, cache paths, or canonical refs."""
    pages = overview.get("pages")
    if not isinstance(pages, list):
        raise ProductionGroundingError("Document page projection is invalid.")
    projected: list[dict[str, object]] = []
    for page in pages:
        if not isinstance(page, Mapping):
            raise ProductionGroundingError("Document page projection is invalid.")
        evidence_ref = page.get("evidence_ref")
        if not isinstance(evidence_ref, str) or not evidence_ref:
            raise ProductionGroundingError("Document page evidence ref is invalid.")
        projected.append(
            {
                "page": page.get("page"),
                "evidence_ref": presentation.opaque_reference(
                    evidence_ref,
                    kind="citation",
                ),
                "extracted_text": _blind_source_identities(
                    page.get("extracted_text"), presentation
                ),
            }
        )
    return projected


def _blind_source_identities(
    value: object,
    presentation: EvidencePresentationRecord,
) -> object:
    """Replace exact approved source identities inside bounded PA content."""
    replacements = {
        entry.canonical_ref: entry.pa_handle
        for entry in presentation.entries
        if entry.canonical_ref is not None
    }
    if isinstance(value, str):
        projected = value
        for canonical_ref, pa_handle in replacements.items():
            projected = projected.replace(canonical_ref, pa_handle)
        return projected
    if isinstance(value, list):
        return [_blind_source_identities(item, presentation) for item in value]
    if isinstance(value, Mapping):
        projected: dict[str, object] = {}
        for key, item in value.items():
            projected_key = str(key)
            if projected_key == "evidence_ref" and isinstance(item, str):
                projected[projected_key] = presentation.opaque_reference(
                    item,
                    kind="citation",
                )
            elif projected_key == "evidence_refs" and isinstance(item, list):
                if not all(isinstance(ref, str) and ref for ref in item):
                    raise ProductionGroundingError("Document evidence references are invalid.")
                projected[projected_key] = [
                    presentation.opaque_reference(ref, kind="citation") for ref in item
                ]
            else:
                projected[projected_key] = _blind_source_identities(item, presentation)
        return projected
    return value


def _assert_blinded_pa_projection(
    value: object,
    presentation: EvidencePresentationRecord,
) -> None:
    """Reject PA-facing data that still contains a semantic shortcut."""
    forbidden_keys = {
        "assembly_target",
        "cad_identity",
        "camera_name",
        "display_name",
        "document_name",
        "evaluator_label",
        "ground_truth",
        "repository_path",
        "source_url",
    }
    canonical_refs = tuple(
        entry.canonical_ref for entry in presentation.entries if entry.canonical_ref
    )
    allowed_cad_refs = frozenset(
        entry.canonical_ref
        for entry in presentation.entries
        if entry.evidence_type == "CAD" and entry.canonical_ref is not None
    )

    def inspect(item: object) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if str(key).casefold() == "context_ref":
                    if not isinstance(nested, str) or nested not in allowed_cad_refs:
                        raise ProductionGroundingError(
                            "PA-facing evidence contains an unauthorized context_ref."
                        )
                    continue
                if str(key).casefold() in forbidden_keys:
                    raise ProductionGroundingError(
                        "PA-facing evidence contains a forbidden semantic shortcut."
                    )
                inspect(nested)
            return
        if isinstance(item, list | tuple):
            for nested in item:
                inspect(nested)
            return
        if not isinstance(item, str):
            return
        if item in allowed_cad_refs:
            return
        folded = item.casefold()
        if (
            any(ref in item for ref in canonical_refs)
            or "/home/" in folded
            or "file://" in folded
            or ".stl" in folded
            or ".pdf" in folded
            or "gazebo_ground_truth" in folded
        ):
            raise ProductionGroundingError(
                "PA-facing evidence contains a forbidden canonical identity or path."
            )

    inspect(value)


def _neutral_candidate_views(
    segmentation: Mapping[str, object],
    *,
    segmentation_record_ref: str,
    review: Mapping[str, object],
) -> list[dict[str, object]]:
    """Project segmentation candidates without semantic sensor-name shortcuts."""
    cameras = segmentation.get("cameras")
    if not isinstance(cameras, list):
        raise ProductionGroundingError("Segmentation candidate views are invalid.")
    review_candidates = review.get("candidates")
    if (
        review.get("record_type") != "ObservationCandidateReview"
        or review.get("status") != "accepted"
        or not isinstance(review_candidates, list)
    ):
        raise ProductionGroundingError("Observation candidate review is invalid.")
    review_by_handle = {
        (item.get("observation_handle"), item.get("candidate_handle")): item
        for item in review_candidates
        if isinstance(item, Mapping)
    }
    views: list[dict[str, object]] = []
    for camera_index, camera_value in enumerate(cameras):
        if not isinstance(camera_value, Mapping):
            raise ProductionGroundingError("Segmentation candidate view is invalid.")
        observation_handle = camera_value.get("observation_handle")
        candidates = camera_value.get("candidates")
        if not isinstance(observation_handle, str) or not isinstance(candidates, list):
            raise ProductionGroundingError("Segmentation candidate view is invalid.")
        projected_candidates: list[dict[str, object]] = []
        for candidate_index, candidate_value in enumerate(candidates):
            if not isinstance(candidate_value, Mapping):
                raise ProductionGroundingError("Segmentation candidate is invalid.")
            candidate_handle = candidate_value.get("candidate_handle")
            if not isinstance(candidate_handle, str) or not candidate_handle:
                raise ProductionGroundingError("Segmentation candidate handle is invalid.")
            field_root = f"/cameras/{camera_index}/candidates/{candidate_index}"
            reviewed = review_by_handle.get((observation_handle, candidate_handle))
            if not isinstance(reviewed, Mapping):
                raise ProductionGroundingError(
                    "Observation review does not cover every segmentation candidate."
                )
            projected_candidates.append(
                {
                    "candidate_handle": candidate_handle,
                    "point_count": candidate_value.get("point_count"),
                    "pixel_bounds_uv": candidate_value.get("pixel_bounds_uv"),
                    "bounds_m": candidate_value.get("bounds_m"),
                    "centroid_m": candidate_value.get("centroid_m"),
                    "depth_range_m": candidate_value.get("depth_range_m"),
                    "visual_description": reviewed.get("description"),
                    "visual_uncertainty": reviewed.get("uncertainty"),
                    "candidate_value_ref": {
                        "record_ref": segmentation_record_ref,
                        "field_path": field_root,
                    },
                    "centroid_value_ref": {
                        "record_ref": segmentation_record_ref,
                        "field_path": f"{field_root}/centroid_m",
                    },
                }
            )
        views.append(
            {
                "observation_handle": observation_handle,
                "rgb_evidence_handle": f"rgb_{camera_index + 1:04d}",
                "mask_evidence_handle": f"mask_{camera_index + 1:04d}",
                "candidate_count": camera_value.get("candidate_count"),
                "candidates": projected_candidates,
            }
        )
    return views


def _proposal_cad_bindings(
    root: Path,
    view: ProductContextView,
    proposal: OntologyGroundingProposal,
    *,
    state_name: str,
) -> tuple[TypedContextBinding, ...]:
    cited = _proposal_state_evidence_refs(proposal, state_name)
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


def _validate_pa_state_cad_comparison(  # noqa: PLR0911
    investigation: _NativeEvidenceInvestigation,
    view: ProductContextView,
    proposal: OntologyGroundingProposal,
    *,
    state_name: str,
    state_binding: _StateCandidateBinding,
) -> str | None:
    """Require one uniquely supported candidate without revealing failure details."""
    uniqueness_gap = (
        f"The submitted {state_name} assignment is not uniquely supported by complete, "
        "internally consistent, hash-pinned approved evidence."
    )
    cad_bindings = _proposal_cad_bindings(
        investigation.root,
        view,
        proposal,
        state_name=state_name,
    )
    if len(cad_bindings) != 1:
        return uniqueness_gap

    value_refs = _proposal_state_value_evidence_refs(proposal, state_name)
    comparisons = [
        binding
        for record_ref, binding in investigation.comparison_bindings.items()
        if record_ref in value_refs
    ]
    if len(comparisons) != 1:
        return uniqueness_gap
    comparison = comparisons[0]
    if comparison.cad_record_ref != cad_bindings[0].record_ref:
        return uniqueness_gap
    if comparison.segmentation_record_ref != state_binding.record_ref:
        return uniqueness_gap

    typed_bindings = [
        binding
        for binding in view.typed_bindings
        if binding.record_ref == comparison.record_ref
        and binding.record_type == "CADSizeCorrespondenceRecord"
    ]
    if len(typed_bindings) != 1:
        return uniqueness_gap
    record_path = investigation.root / comparison.record_ref
    if _sha256_path(record_path) != typed_bindings[0].record_sha256:
        return uniqueness_gap
    record = _read_json(record_path)
    cad = record.get("CAD")
    cad_record = cad.get("record") if isinstance(cad, Mapping) else None
    segmentation = record.get("segmentation")
    segmentation_record = segmentation.get("record") if isinstance(segmentation, Mapping) else None
    if (
        record.get("record_type") != "CADSizeCorrespondenceRecord"
        or not isinstance(cad_record, Mapping)
        or cad_record.get("ref") != comparison.cad_record_ref
        or not isinstance(segmentation_record, Mapping)
        or segmentation_record.get("ref") != comparison.segmentation_record_ref
        or typed_bindings[0].status != record.get("CAD_correspondence")
    ):
        return uniqueness_gap

    observation_handle = _candidate_observation_handle(
        investigation.root / state_binding.record_ref,
        state_binding.field_path,
    )
    candidate_handle = _candidate_handle(
        investigation.root / state_binding.record_ref,
        state_binding.field_path,
    )
    selected = record.get("selected_candidate")
    if (
        record.get("CAD_correspondence") != "accepted"
        or not isinstance(selected, Mapping)
        or selected.get("observation_handle") != observation_handle
        or selected.get("candidate_handle") != candidate_handle
    ):
        return uniqueness_gap
    return None


def _proposal_state_evidence_refs(
    proposal: OntologyGroundingProposal,
    state_name: str,
) -> frozenset[str]:
    """Return direct citations attached to one PA-authored feature state."""
    state = proposal.target_feature.get(state_name)
    if not isinstance(state, Mapping):
        raise ProductionGroundingError(f"{state_name} is unavailable for CAD validation.")
    refs: list[str] = []
    statement = state.get("statement")
    statement_refs = statement.get("evidence_refs") if isinstance(statement, Mapping) else None
    if isinstance(statement_refs, list):
        refs.extend(str(ref) for ref in statement_refs if isinstance(ref, str))
    values = state.get("state_values")
    if isinstance(values, list):
        for value in values:
            evidence_refs = value.get("evidence_refs") if isinstance(value, Mapping) else None
            if isinstance(evidence_refs, list):
                refs.extend(str(ref) for ref in evidence_refs if isinstance(ref, str))
    return frozenset(refs)


def _proposal_state_value_evidence_refs(
    proposal: OntologyGroundingProposal,
    state_name: str,
) -> frozenset[str]:
    """Return citations attached to the state's exactly one selected value."""
    state = proposal.target_feature.get(state_name)
    values = state.get("state_values") if isinstance(state, Mapping) else None
    if not isinstance(values, list) or len(values) != 1:
        return frozenset()
    value = values[0]
    refs = value.get("evidence_refs") if isinstance(value, Mapping) else None
    if not isinstance(refs, list):
        return frozenset()
    return frozenset(str(ref) for ref in refs if isinstance(ref, str))


def _proposal_state_comparison_path(
    investigation: _NativeEvidenceInvestigation,
    proposal: OntologyGroundingProposal,
    *,
    state_name: str,
) -> Path | None:
    """Resolve the one PA-requested CAD comparison cited by a state value."""
    refs = _proposal_state_value_evidence_refs(proposal, state_name)
    matches = [
        investigation.root / record_ref
        for record_ref in investigation.comparison_bindings
        if record_ref in refs
    ]
    return matches[0] if len(matches) == 1 else None


def _candidate_observation_handle(record_path: Path, field_path: str) -> str:
    candidate, camera = _candidate_at_field_path(record_path, field_path)
    del candidate
    observation_handle = camera.get("observation_handle")
    if not isinstance(observation_handle, str) or not observation_handle:
        raise ProductionGroundingError("Selected candidate observation handle is invalid.")
    return observation_handle


def _candidate_handle(record_path: Path, field_path: str) -> str:
    candidate, _camera = _candidate_at_field_path(record_path, field_path)
    candidate_handle = candidate.get("candidate_handle")
    if not isinstance(candidate_handle, str) or not candidate_handle:
        raise ProductionGroundingError("Selected candidate handle is invalid.")
    return candidate_handle


def _candidate_at_field_path(
    record_path: Path,
    field_path: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    parts = field_path.strip("/").split("/")
    if (
        len(parts) != 4
        or parts[0] != "cameras"
        or not parts[1].isdigit()
        or parts[2] != "candidates"
        or not parts[3].isdigit()
    ):
        raise ProductionGroundingError("Selected candidate field path is invalid.")
    record = _read_json(record_path)
    cameras = record.get("cameras")
    try:
        camera = cameras[int(parts[1])]  # type: ignore[index]
        candidate = camera["candidates"][int(parts[3])]
    except (IndexError, KeyError, TypeError) as exc:
        raise ProductionGroundingError("Selected candidate field path is unavailable.") from exc
    if not isinstance(camera, Mapping) or not isinstance(candidate, Mapping):
        raise ProductionGroundingError("Selected candidate field path is invalid.")
    return candidate, camera


def _newest_binding(view: ProductContextView, record_type: str) -> TypedContextBinding | None:
    candidates = [
        binding
        for binding in view.typed_bindings
        if binding.record_type == record_type and binding.status == "accepted"
    ]
    return (
        None
        if not candidates
        else max(
            candidates,
            key=lambda item: (
                -1 if item.observed_at_ns is None else item.observed_at_ns,
                item.record_ref,
            ),
        )
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
    evidence_refs = sorted(
        {
            ref
            for binding in prerequisite_bindings
            for ref in (binding.record_ref, *binding.evidence_refs)
        }
    )
    merge = validate_and_merge_triple_delta(
        root,
        tbox,
        _GEOMETRY_PRODUCER,
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [record_ref],
        },
        authorized_evidence_refs=evidence_refs,
    )
    view = build_product_context_view(
        root, merge.abox, attempted_evidence=(), assessed_at_ns=time.time_ns()
    )
    persist_product_context_view(root, view)
    binding = next(item for item in view.typed_bindings if item.record_ref == record_ref)
    if binding.status != status:
        raise ProductionGroundingError(
            f"Derived {binding.record_type} status does not match its provider result."
        )
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
                raise ProductionGroundingError(f"Multiple available providers produce {output}.")
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
        binding.record_type for binding in view.typed_bindings if binding.status == "accepted"
    }
    if not target_cad_bindings:
        accepted.discard("CADMeshRecord")
    missing = tuple(record for record in required_plan if record not in accepted)
    raw_types = _raw_evidence_types_for_gap(
        descriptors,
        missing,
    )
    eligible_ids = tuple(
        handle.evidence_id for handle in handles if handle.evidence_type in raw_types
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
    catalog: list[Mapping[str, object]] = [handle.discovery_record() for handle in handles]
    seen_record_refs: set[tuple[str, ...]] = set()
    for item in (
        *investigation.prior_evidence,
        *(
            {"retrieval_state": "already_retrieved", **dict(result)}
            for result in investigation.retrieved_results.values()
        ),
        *(
            {"comparison_state": "already_compared", **dict(result)}
            for result in getattr(investigation, "comparison_results", {}).values()
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


def _append_validation_feedback(
    history: list[Mapping[str, object]],
    feedback: Mapping[str, object],
) -> None:
    """Append one distinct correction while preserving its first-seen order."""
    fingerprint = json.dumps(feedback, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if any(
        json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False) == fingerprint
        for item in history
    ):
        return
    history.append(dict(feedback))


def _is_tool_round_exhaustion(error: RuntimeError) -> bool:
    """Recognize only the shared agent runtime's bounded-tool terminal error."""
    cause: BaseException | None = error
    while cause is not None:
        if str(cause) == _TOOL_ROUND_LIMIT_MESSAGE:
            return True
        cause = cause.__cause__
    return False


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
    words = {token.strip(".,:;!?()[]{}\"'") for token in normalized.split()}
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
            raise ProductionGroundingError("Answered clarification history is invalid.")
        evidence_ref = (
            Path("interaction_record") / f"clarification_{question_turn:04d}.json"
        ).as_posix()
        clarification_path = root / evidence_ref
        if not clarification_path.is_file() or _read_json(clarification_path) != dict(
            clarification
        ):
            raise ProductionGroundingError(
                "Answered clarification record is unavailable or changed."
            )
        evidence.append(
            {
                "evidence_type": "user_clarification",
                "evidence_ref": evidence_ref,
                "reply": reply,
            }
        )
    return evidence


def _present_answered_clarification_evidence(
    root: Path,
    clarification_history: Sequence[Mapping[str, object]],
    investigation: _NativeEvidenceInvestigation,
) -> list[Mapping[str, object]]:
    """Project persisted user clarifications through opaque citation handles."""
    projected: list[Mapping[str, object]] = []
    for item in _answered_clarification_evidence(root, clarification_history):
        canonical_ref = str(item["evidence_ref"])
        investigation.register_canonical_reference(canonical_ref, kind="citation")
        investigation.authorized_evidence_refs.add(canonical_ref)
        projected.append(
            {
                **dict(item),
                "evidence_ref": investigation.project_canonical_reference(canonical_ref),
            }
        )
    return projected


def _configured_target_frame(workcell: Any) -> str:
    """Derive the common controller frame without static simulation geometry."""
    frames: set[str] = set()
    for resource in workcell._profile.resources:
        record = _read_json(resource.manifest_path)
        entry = record.get(resource.symbol)
        mode = entry.get("execution_mode") if isinstance(entry, Mapping) else None
        environment_name = {"simulation": "gazebo", "physical": "real"}.get(mode)
        environment = (
            entry.get(environment_name) if isinstance(entry, Mapping) and environment_name else None
        )
        if mode == "simulation":
            controller = environment.get("controller") if isinstance(environment, Mapping) else None
            move_group = controller.get("move_group") if isinstance(controller, Mapping) else None
            frame = move_group.get("frame_id") if isinstance(move_group, Mapping) else None
        else:
            static = (
                environment.get("static_capabilities") if isinstance(environment, Mapping) else None
            )
            reach = static.get("gripper_reach") if isinstance(static, Mapping) else None
            frame = reach.get("frame") if isinstance(reach, Mapping) else None
        if not isinstance(frame, str) or not frame:
            raise ProductionGroundingError("A configured resource has no target frame.")
        frames.add(frame)
    if len(frames) != 1:
        raise ProductionGroundingError("Configured resources have inconsistent target frames.")
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

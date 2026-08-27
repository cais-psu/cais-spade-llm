"""Run production pre-RA PA grounding through authorized producers.

The runtime in this module evolves a robot-independent ``TaskTransitionDraft``
and resolves only its current PA-owned ``ContextNeed`` values.  It never
contacts RA, selects primitives, or converts a pose into a robot frame.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    ContextNeed,
    GroundingContractError,
    GroundingProducerDescriptor,
    ProductContextView,
    TaskTransitionDraft,
    build_product_context_view,
    persist_product_context_view,
    select_grounding_producer,
    unresolved_context_needs,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    ABoxSnapshot,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.config import DocumentVLMConfig
from cais_spade_llm.spec2primitives.ontology import TBoxSnapshot
from cais_spade_llm.spec2primitives.tools.document_evidence import (
    DocumentVisionRuntime,
    interpret_document_evidence,
)
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_context_ref_evidence_types,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    associate_segmented_candidate_by_size,
    estimate_camera_frame_pose,
    preprocess_served_geometry,
    segment_preprocessed_observation,
)

_PROHIBITED_DRAFT_TERMS = (
    "primitive_steps",
    "move_to_pick_location",
    "pick_part",
    "move_loaded_to_destination",
    "place_part",
)
_TASK_DRAFT_ROOT = Path("products/grounding/task_transition")
_SELECTION_ROOT = Path("interaction_record")
_DOCUMENT_PRODUCER = "document_evidence"
_GEOMETRY_PRODUCER = "rgb_d_cad_grounding"
_TYPED_OUTPUTS = (
    "CADMeshRecord",
    "ColoredPointCloudSetRecord",
    "RGBDSegmentationRecord",
    "CADSizeCorrespondenceRecord",
    "CADPoseEstimationRecord",
)


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
        """Create the first draft-driven evidence request before retrieval."""
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
            initial=True,
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
        """Evolve the task draft and return one validated Phase 4.3 decision."""
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
            initial=False,
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
        initial: bool,
    ) -> Mapping[str, object]:
        self._validate_tbox(tbox, abox)
        root = Path(interaction_root).resolve()
        view = ProductContextView.from_mapping(product_context)
        if view.tbox_fingerprint != tbox.fingerprint:
            raise ProductionGroundingError(
                "ProductContextView does not match the authoritative TBox."
            )
        for _derived_step in range(3):
            draft = await self._author_task_transition_draft(
                product_agent,
                root=root,
                view=view,
                clarification_history=clarification_history,
                turn_number=turn_number,
                max_pa_turns=max_pa_turns,
            )
            needs = unresolved_context_needs(draft, view)
            if not needs:
                if initial:
                    raise ProductionGroundingError(
                        "The initial PA decision must retrieve evidence before completion."
                    )
                return {
                    "unresolved_semantic_need": None,
                    "needed_context": None,
                    "context understanding complete": True,
                }
            need = needs[0]
            if need.kind == "user_intent":
                if initial:
                    raise ProductionGroundingError(
                        "The first PA decision cannot request user clarification."
                    )
                return _clarification_decision(need)

            attempted_producers = _attempted_producers(root, need)
            descriptor = select_grounding_producer(
                need,
                self._descriptors,
                attempted_producers=attempted_producers,
            )
            if "existing_record" in descriptor.evidence_types:
                abox = self._run_derived_producer(
                    root=root,
                    tbox=tbox,
                    abox=abox,
                    view=view,
                    need=need,
                    descriptor=descriptor,
                )
                view = build_product_context_view(
                    root,
                    abox,
                    attempted_evidence=attempted_evidence,
                    assessed_at_ns=time.time_ns(),
                )
                persist_product_context_view(root, view)
                continue

            request = await self._select_evidence_request(
                product_agent,
                need=need,
                descriptor=descriptor,
                view=view,
            )
            _persist_producer_selection(root, need, descriptor, request, view)
            return _needed_context_decision(need, request)
        raise ProductionGroundingError(
            "Derived grounding exceeded its bounded pre-RA progress loop."
        )

    async def _author_task_transition_draft(
        self,
        product_agent: ProductAgentContextRuntime,
        *,
        root: Path,
        view: ProductContextView,
        clarification_history: tuple[Mapping[str, object], ...],
        turn_number: int,
        max_pa_turns: int,
    ) -> TaskTransitionDraft:
        version = _next_number(root, f"{_TASK_DRAFT_ROOT}/draft_*.json")
        prompt_value = {
            "product_requirement": view.product_requirement,
            "ProductContextView": view.to_record(),
            "clarification_history": [dict(item) for item in clarification_history],
            "allowed_semantic_classes": sorted(self._tbox.classes),
            "allowed_semantic_properties": sorted(
                self._tbox.object_properties | self._tbox.datatype_properties
            ),
            "allowed_typed_context_records": list(_TYPED_OUTPUTS),
            "draft_version": version,
            "turn": turn_number,
            "max_pa_turns": max_pa_turns,
        }
        prompt = (
            "Evolve only one robot-independent Spec2Primitives TaskTransitionDraft. "
            "Declare only the currently blocking PA-owned semantic or typed inputs. "
            "Treat clarification replies only as user-owned intent. Never use a user "
            "reply as factual product, scene, geometry, or ontology evidence. "
            "Use exact supplied symbols. Do not select evidence, a resource, a primitive, "
            "primitive_steps, a predefined composite function, or robot behavior.\n\n"
            f"PA grounding input:\n{json.dumps(prompt_value, indent=2, ensure_ascii=False)}"
        )
        output = await product_agent.ask_llm_structured(
            prompt,
            response_format=_task_draft_response_format(
                view,
                version=version,
                classes=sorted(self._tbox.classes),
                properties=sorted(
                    self._tbox.object_properties | self._tbox.datatype_properties
                ),
            ),
        )
        if not isinstance(output, Mapping) or set(output) != {"task_transition_draft"}:
            raise ProductionGroundingError(
                "ProductAgent returned an invalid TaskTransitionDraft envelope."
            )
        try:
            draft = TaskTransitionDraft.from_mapping(
                _required_mapping(output["task_transition_draft"], "task_transition_draft")
            )
        except (GroundingContractError, TypeError) as exc:
            raise ProductionGroundingError(f"TaskTransitionDraft is invalid: {exc}") from exc
        _validate_task_draft(draft, view, self._tbox)
        _write_json_exclusive(
            root / _TASK_DRAFT_ROOT / f"draft_{version:04d}.json",
            draft.to_record(),
        )
        return draft

    async def _select_evidence_request(
        self,
        product_agent: ProductAgentContextRuntime,
        *,
        need: ContextNeed,
        descriptor: GroundingProducerDescriptor,
        view: ProductContextView,
    ) -> dict[str, object]:
        evidence_types = set(descriptor.evidence_types)
        if evidence_types == {"observation"}:
            return {
                "context_ref": None,
                "request_live_observation": True,
                "clarification_question": None,
            }
        approved = approved_context_ref_evidence_types()
        candidates = sorted(
            context_ref
            for context_ref, evidence_type in approved.items()
            if evidence_type in evidence_types
            and context_ref not in view.attempted_evidence
        )
        if not candidates:
            raise ProductionGroundingError(
                f"No untried approved evidence is available for {need.symbol}."
            )
        if len(candidates) == 1:
            context_ref = candidates[0]
        else:
            output = await product_agent.ask_llm_structured(
                (
                    "Select one exact approved evidence ref for the current PA ContextNeed. "
                    "Do not infer simulator identity, pose, primitive_steps, or robot behavior.\n\n"
                    f"Selection input:\n{json.dumps({'ContextNeed': need.to_record(), 'approved_context_refs': candidates}, indent=2, ensure_ascii=False)}"
                ),
                response_format=_evidence_selection_response_format(candidates),
            )
            if (
                not isinstance(output, Mapping)
                or set(output) != {"context_ref"}
                or output["context_ref"] not in candidates
            ):
                raise ProductionGroundingError(
                    "ProductAgent selected evidence outside the approved candidates."
                )
            context_ref = str(output["context_ref"])
        return {
            "context_ref": context_ref,
            "request_live_observation": False,
            "clarification_question": None,
        }

    def _run_derived_producer(  # noqa: PLR0913
        self,
        *,
        root: Path,
        tbox: TBoxSnapshot,
        abox: ABoxSnapshot,
        view: ProductContextView,
        need: ContextNeed,
        descriptor: GroundingProducerDescriptor,
    ) -> ABoxSnapshot:
        bindings = {
            binding.record_type: binding
            for binding in view.typed_bindings
            if binding.status == "accepted"
        }
        missing = [
            record_type
            for record_type in descriptor.required_record_types
            if record_type not in bindings
        ]
        if missing:
            raise ProductionGroundingError(
                f"{descriptor.producer} lacks prerequisite records: {missing}."
            )
        if need.symbol == "CADSizeCorrespondenceRecord":
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
        elif need.symbol == "CADPoseEstimationRecord":
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
                f"No derived pre-RA producer is implemented for {need.symbol}."
            )
        record_ref = _relative_ref(root, record_path)
        unresolved = []
        if status != "accepted":
            unresolved.append(
                {
                    "description": f"{need.symbol} is {status}.",
                    "evidence_refs": sorted(
                        {
                            evidence_ref
                            for binding in bindings.values()
                            for evidence_ref in binding.evidence_refs
                        }
                    ),
                }
            )
        merge = validate_and_merge_triple_delta(
            root,
            tbox,
            descriptor.producer,
            {
                "assertions": [],
                "uncertainty": [],
                "unresolved_evidence_needs": unresolved,
                "typed_context_refs": [record_ref],
            },
            authorized_evidence_refs=sorted(
                {
                    evidence_ref
                    for binding in bindings.values()
                    for evidence_ref in binding.evidence_refs
                }
            ),
        )
        _persist_producer_selection(
            root,
            need,
            descriptor,
            {"existing_record": list(descriptor.required_record_types)},
            view,
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


def _producer_descriptors(
    tbox: TBoxSnapshot,
) -> tuple[GroundingProducerDescriptor, ...]:
    prohibited_properties = {
        f"{tbox.ppr_namespace}{name}"
        for name in ("capableOf", "requires", "precedes", "provides")
    }
    semantic_outputs = [
        {"kind": "class", "symbol": symbol} for symbol in sorted(tbox.classes)
    ]
    semantic_outputs.extend(
        {"kind": "property", "symbol": symbol}
        for symbol in sorted(
            (tbox.object_properties | tbox.datatype_properties) - prohibited_properties
        )
    )
    records = [
        {
            "producer": _DOCUMENT_PRODUCER,
            "supported_outputs": semantic_outputs,
            "evidence_types": ["document"],
            "required_record_types": [],
            "priority": 0,
        },
        {
            "producer": _GEOMETRY_PRODUCER,
            "supported_outputs": [
                {"kind": "typed_context_record", "symbol": "CADMeshRecord"}
            ],
            "evidence_types": ["CAD"],
            "required_record_types": [],
            "priority": 0,
        },
        {
            "producer": _GEOMETRY_PRODUCER,
            "supported_outputs": [
                {
                    "kind": "typed_context_record",
                    "symbol": "ColoredPointCloudSetRecord",
                },
                {
                    "kind": "typed_context_record",
                    "symbol": "RGBDSegmentationRecord",
                },
            ],
            "evidence_types": ["observation"],
            "required_record_types": [],
            "priority": 0,
        },
        {
            "producer": _GEOMETRY_PRODUCER,
            "supported_outputs": [
                {
                    "kind": "typed_context_record",
                    "symbol": "CADSizeCorrespondenceRecord",
                }
            ],
            "evidence_types": ["existing_record"],
            "required_record_types": ["CADMeshRecord", "RGBDSegmentationRecord"],
            "priority": 0,
        },
        {
            "producer": _GEOMETRY_PRODUCER,
            "supported_outputs": [
                {
                    "kind": "typed_context_record",
                    "symbol": "CADPoseEstimationRecord",
                }
            ],
            "evidence_types": ["existing_record"],
            "required_record_types": ["CADSizeCorrespondenceRecord"],
            "priority": 0,
        },
    ]
    return tuple(GroundingProducerDescriptor.from_mapping(record) for record in records)


def _task_draft_response_format(
    view: ProductContextView,
    *,
    version: int,
    classes: list[str],
    properties: list[str],
) -> dict[str, Any]:
    symbols = [*classes, *properties, *_TYPED_OUTPUTS, "product_requirement"]
    return {
        "name": "spec2primitives_task_transition_draft",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["task_transition_draft"],
            "properties": {
                "task_transition_draft": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "version",
                        "product_requirement",
                        "requested_process",
                        "required_outcome",
                        "required_inputs",
                        "unresolved_user_intent",
                        "source_view_fingerprint",
                    ],
                    "properties": {
                        "version": {"const": version},
                        "product_requirement": {"const": view.product_requirement},
                        "requested_process": {"enum": [None, *classes]},
                        "required_outcome": {"type": ["string", "null"]},
                        "required_inputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "kind",
                                    "symbol",
                                    "subject_role",
                                    "authority",
                                    "frame",
                                    "maximum_age_ns",
                                    "reason",
                                ],
                                "properties": {
                                    "kind": {
                                        "enum": [
                                            "class",
                                            "property",
                                            "typed_context_record",
                                            "user_intent",
                                        ]
                                    },
                                    "symbol": {"enum": symbols},
                                    "subject_role": {"type": "string", "minLength": 1},
                                    "authority": {"const": "PA"},
                                    "frame": {"type": ["string", "null"]},
                                    "maximum_age_ns": {
                                        "type": ["integer", "null"],
                                        "minimum": 0,
                                    },
                                    "reason": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                        "unresolved_user_intent": {"type": ["string", "null"]},
                        "source_view_fingerprint": {"const": view.fingerprint},
                    },
                }
            },
        },
    }


def _evidence_selection_response_format(candidates: list[str]) -> dict[str, Any]:
    return {
        "name": "spec2primitives_grounding_evidence_selection",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["context_ref"],
            "properties": {"context_ref": {"enum": candidates}},
        },
    }


def _validate_task_draft(
    draft: TaskTransitionDraft,
    view: ProductContextView,
    tbox: TBoxSnapshot,
) -> None:
    if (
        draft.product_requirement != view.product_requirement
        or draft.source_view_fingerprint != view.fingerprint
    ):
        raise ProductionGroundingError(
            "TaskTransitionDraft does not match its ProductContextView."
        )
    for need in draft.required_inputs:
        if need.kind == "class" and need.symbol not in tbox.classes:
            raise ProductionGroundingError(
                f"TaskTransitionDraft uses undeclared ontology symbol: {need.symbol}."
            )
        if need.kind == "property" and need.symbol not in (
            tbox.object_properties | tbox.datatype_properties
        ):
            raise ProductionGroundingError(
                f"TaskTransitionDraft uses undeclared ontology symbol: {need.symbol}."
            )
        if need.kind == "typed_context_record" and need.symbol not in _TYPED_OUTPUTS:
            raise ProductionGroundingError(
                f"TaskTransitionDraft uses unsupported typed output: {need.symbol}."
            )
    serialized = json.dumps(draft.to_record(), ensure_ascii=False).lower()
    for term in _PROHIBITED_DRAFT_TERMS:
        if term.lower() in serialized:
            raise ProductionGroundingError(
                f"TaskTransitionDraft contains prohibited composition term: {term}."
            )


def _needed_context_decision(
    need: ContextNeed,
    request: Mapping[str, object],
) -> dict[str, object]:
    return {
        "unresolved_semantic_need": {
            "kind": need.kind,
            "symbol": need.symbol,
            "description": need.reason,
        },
        "needed_context": dict(request),
        "context understanding complete": False,
    }


def _clarification_decision(need: ContextNeed) -> dict[str, object]:
    return {
        "unresolved_semantic_need": {
            "kind": "user_intent",
            "symbol": need.symbol,
            "description": need.reason,
        },
        "needed_context": {
            "context_ref": None,
            "request_live_observation": False,
            "clarification_question": need.reason,
        },
        "context understanding complete": False,
    }


def _persist_producer_selection(
    root: Path,
    need: ContextNeed,
    descriptor: GroundingProducerDescriptor,
    request: Mapping[str, object],
    view: ProductContextView,
) -> Path:
    number = _next_number(root, f"{_SELECTION_ROOT}/producer_selection_*.json")
    path = root / _SELECTION_ROOT / f"producer_selection_{number:04d}.json"
    _write_json_exclusive(
        path,
        {
            "schema_version": 1,
            "selection": number,
            "ContextNeed": need.to_record(),
            "GroundingProducerDescriptor": descriptor.to_record(),
            "request": dict(request),
            "product_context_fingerprint": view.fingerprint,
        },
    )
    return path


def _attempted_producers(root: Path, need: ContextNeed) -> tuple[str, ...]:
    producers: list[str] = []
    for path in sorted((root / _SELECTION_ROOT).glob("producer_selection_*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProductionGroundingError(
                f"Producer selection record could not be read: {path.name}."
            ) from exc
        if not isinstance(value, Mapping) or value.get("ContextNeed") != need.to_record():
            continue
        descriptor = value.get("GroundingProducerDescriptor")
        producer = descriptor.get("producer") if isinstance(descriptor, Mapping) else None
        if isinstance(producer, str):
            producers.append(producer)
    return tuple(producers)


def _selected_output_for_served_context(
    root: Path,
    served_context: Mapping[str, object],
) -> str | None:
    context_ref = served_context.get("context_ref")
    evidence_type = served_context.get("evidence_type")
    for path in reversed(
        sorted((root / _SELECTION_ROOT).glob("producer_selection_*.json"))
    ):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        request = value.get("request") if isinstance(value, Mapping) else None
        need = value.get("ContextNeed") if isinstance(value, Mapping) else None
        if not isinstance(request, Mapping) or not isinstance(need, Mapping):
            continue
        if evidence_type == "observation" and request.get("request_live_observation") is True:
            symbol = need.get("symbol")
            return symbol if isinstance(symbol, str) else None
        if isinstance(context_ref, str) and request.get("context_ref") == context_ref:
            symbol = need.get("symbol")
            return symbol if isinstance(symbol, str) else None
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


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise ProductionGroundingError(f"Grounding record exists: {path.name}.") from exc

"""Controlled ontology and grounding doubles for PA workflow tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from rdflib import RDF

from cais_spade_llm.spec2primitives.agents.pa.context_grounding import (
    PAOntologyConfig,
)
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingNextAction,
    GroundingProducerDescriptor,
    GroundingSession,
    persist_grounding_session,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    ABoxSnapshot,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
    commit_resource_assignment,
    derive_resource_assignment_need,
    select_predefined_resource,
)
from cais_spade_llm.spec2primitives.ontology import (
    TBoxSnapshot,
    load_predefined_resource_registry,
    load_predefined_workcell,
)

PPR_NAMESPACE = "http://PAonto.com#"
MINIMAL_TBOX_PATH = Path(__file__).parent / "fixtures/ontology/minimal_ppr_tbox.owl"


def ontology_config() -> PAOntologyConfig:
    """Return the schema-only test fixture configuration."""
    return PAOntologyConfig(
        tbox_path=MINIMAL_TBOX_PATH,
        ppr_namespace=PPR_NAMESPACE,
    )


class ControlledGroundingRuntime:
    """Produce deterministic evidence deltas and Phase 4.3 decisions."""

    def __init__(
        self,
        *,
        assessments: list[Mapping[str, object]] | None = None,
        interpretation_error: Exception | None = None,
        assessment_error: Exception | None = None,
        interpretation_override: Mapping[str, object] | None = None,
        descriptors: list[GroundingProducerDescriptor] | None = None,
    ) -> None:
        self.assessments = None if assessments is None else list(assessments)
        self.interpretation_error = interpretation_error
        self.assessment_error = assessment_error
        self.interpretation_override = interpretation_override
        self.descriptors = list(descriptors or _default_descriptors())
        self.interpretation_calls: list[dict[str, object]] = []
        self.assessment_calls: list[dict[str, object]] = []

    def grounding_producer_descriptors(self) -> list[GroundingProducerDescriptor]:
        """Return the controlled output-capable producer registry."""
        return list(self.descriptors)

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
        """Return one valid dynamic assertion for the served evidence."""
        self.interpretation_calls.append(
            {
                "producer": producer,
                "interaction_root": interaction_root,
                "served_context": served_context,
                "operation_number": operation_number,
                "delta_count": abox.delta_count,
            }
        )
        if self.interpretation_error is not None:
            raise self.interpretation_error
        if self.interpretation_override is not None:
            return self.interpretation_override

        evidence_ref = served_context.get("context_ref") or served_context.get("observation_ref")
        if not isinstance(evidence_ref, str):
            raise ValueError("Controlled served evidence has no exact ref.")
        class_iri = (
            f"{PPR_NAMESPACE}feature"
            if producer == "document_evidence"
            else f"{PPR_NAMESPACE}product"
        )
        subject = f"{abox.namespace}{producer}_{operation_number}"
        return {
            "assertions": [
                {
                    "subject": subject,
                    "predicate": str(RDF.type),
                    "object": {"kind": "iri", "value": class_iri},
                    "evidence_refs": [evidence_ref],
                }
            ],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [],
        }

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
        """Return a scripted decision or delegate through the PA protocol."""
        call = {
            "interaction_root": interaction_root,
            "tbox_fingerprint": tbox.fingerprint,
            "delta_count": abox.delta_count,
            "abox_view": abox_view,
            "attempted_evidence": attempted_evidence,
            "clarification_history": clarification_history,
            "turn_number": turn_number,
            "max_pa_turns": max_pa_turns,
        }
        self.assessment_calls.append(call)
        if self.assessment_error is not None:
            raise self.assessment_error
        if self.assessments is not None:
            result = self.assessments.pop(0)
        else:
            prompt = (
                "Assess only this ontology-backed product context. Do not inspect raw "
                "served evidence.\n"
                f"{json.dumps(call, default=str, ensure_ascii=False)}"
            )
            result = await product_agent.ask_llm_structured(
                prompt,
                response_format={"name": "controlled_phase_4_3_test"},
            )
        if (
            set(result)
            == {
                "unresolved_semantic_need",
                "needed_context",
                "context understanding complete",
                "grounding_status",
            }
            and result.get("unresolved_semantic_need") is None
            and result.get("needed_context") is None
            and result.get("context understanding complete") is True
            and result.get("grounding_status") == "complete"
        ):
            _persist_controlled_completion_session(
                interaction_root,
                tbox=tbox,
                abox=abox,
                product_requirement=str(abox_view["product_requirement"]),
                attempted_evidence=attempted_evidence,
            )
        return result


def _persist_controlled_completion_session(
    interaction_root: Path,
    *,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    product_requirement: str,
    attempted_evidence: tuple[str, ...],
) -> None:
    """Persist minimal generalized completion inputs for controlled tests."""
    source_ref = attempted_evidence[-1] if attempted_evidence else "requirement_0001"
    next_action = GroundingNextAction.from_mapping(
        {"action": "propose_grounding"}
    )
    ready = GroundingSession.create(
        revision=1,
        requirement_text=product_requirement,
        attempted_actions=[],
        next_action=next_action,
        status="ready_for_ontology",
    )
    persist_grounding_session(interaction_root, ready)
    feature_iri = f"{abox.namespace}medium_gear_feature"
    task_assertions = [
        {
            "subject": feature_iri,
            "predicate": str(RDF.type),
            "object": {"kind": "iri", "value": f"{PPR_NAMESPACE}feature"},
            "evidence_refs": [source_ref],
        },
        {
            "subject": abox.specification_iri,
            "predicate": f"{PPR_NAMESPACE}defines",
            "object": {"kind": "iri", "value": feature_iri},
            "evidence_refs": [source_ref],
        },
        {
            "subject": "https://cais-spade-llm.local/process/assembly",
            "predicate": f"{PPR_NAMESPACE}realizes",
            "object": {"kind": "iri", "value": feature_iri},
            "evidence_refs": [source_ref],
        },
    ]
    task_delta = {
        "assertions": task_assertions,
        "uncertainty": [],
        "unresolved_evidence_needs": [],
        "typed_context_refs": [],
    }
    validate_and_merge_triple_delta(
        interaction_root,
        tbox,
        "ontology_grounding",
        task_delta,
        authorized_evidence_refs=[source_ref],
    )
    proposal_path = (
        Path(interaction_root)
        / "products/grounding/ontology_grounding/proposal_0001.json"
    )
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal = {
        "schema_version": 3,
        "record_type": "OntologyGroundingProposal",
        "proposal_number": 1,
        "session_revision": ready.revision,
        "session_fingerprint": ready.fingerprint,
        "initialized_specification_iri": abox.specification_iri,
        "output": {
            "ontology_grounding_proposal": {
                "individuals": [
                    {"individual_index": 1, "class_iri": f"{PPR_NAMESPACE}feature"}
                ],
                "relations": [
                    {
                        "subject_kind": "specification",
                        "subject_individual_index": None,
                        "subject_iri": None,
                        "predicate_iri": f"{PPR_NAMESPACE}defines",
                        "object_kind": "new_individual",
                        "object_individual_index": 1,
                        "object_iri": None,
                    },
                    {
                        "subject_kind": "existing_individual",
                        "subject_individual_index": None,
                        "subject_iri": "https://cais-spade-llm.local/process/assembly",
                        "predicate_iri": f"{PPR_NAMESPACE}realizes",
                        "object_kind": "new_individual",
                        "object_individual_index": 1,
                        "object_iri": None,
                    },
                ],
                "literal_facts": [],
                "context_summary": (
                    "The controlled interaction has enough cited product context."
                ),
                "evidence_refs": [source_ref],
                "missing_information": [],
            }
        },
        "compiled_delta": task_delta,
        "status": "accepted",
        "failure": None,
    }
    with proposal_path.open("x", encoding="utf-8") as stream:
        json.dump(proposal, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    pose_ref = _write_controlled_world_pose(interaction_root)
    pose_merge = validate_and_merge_triple_delta(
        interaction_root,
        tbox,
        "controlled_world_pose_provider",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [pose_ref],
        },
        authorized_evidence_refs=[source_ref],
    )
    registry = load_predefined_resource_registry(tbox)
    workcell = load_predefined_workcell(tbox, registry)
    need = derive_resource_assignment_need(pose_merge.abox, workcell)
    if need is None:
        raise AssertionError("Controlled completion has no resource-assignment need.")
    selection = select_predefined_resource(
        interaction_root=interaction_root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        need=need,
        robot_frame_pose_path=Path(interaction_root) / pose_ref,
    )
    commit_resource_assignment(
        interaction_root=interaction_root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        need=need,
        selection=selection,
    )
    complete = GroundingSession.create(
        revision=2,
        requirement_text=product_requirement,
        attempted_actions=[],
        next_action=next_action,
        status="complete",
    )
    persist_grounding_session(interaction_root, complete)


def _write_controlled_world_pose(interaction_root: Path) -> str:
    """Persist a direct world-pose record with one hash-pinned source record."""
    root = Path(interaction_root)
    destination = root / "products/grounding/controlled_world_pose"
    destination.mkdir(parents=True, exist_ok=True)
    source_path = destination / "source_evidence_0001.json"
    source_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "ControlledWorldPoseEvidence",
                "producer": "controlled_world_pose_provider",
                "status": "accepted",
                "observation_timestamp_ns": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source_ref = source_path.relative_to(root).as_posix()
    pose_path = destination / "world_pose_0001.json"
    pose_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "RobotFramePoseRecord",
                "producer": "controlled_world_pose_provider",
                "target_frame": "world",
                "observation_timestamp_ns": 1,
                "robot_frame_conversion": "accepted",
                "CAD_correspondence": "accepted",
                "location": "available",
                "pose": "accepted",
                "source_evidence": {
                    "ref": source_ref,
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                },
                "robot_frame_pose": {
                    "CAD_origin_translation_m": [0.0, -0.5, 1.1]
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return pose_path.relative_to(root).as_posix()


def request_context(
    context_ref: str,
    *,
    semantic_need: str = "additional product evidence",
) -> dict[str, object]:
    """Return one valid Phase 4.3 static-evidence request."""
    evidence_type = "document" if context_ref.endswith(".pdf") else "CAD"
    return {
        "unresolved_semantic_need": {
            "kind": "class",
            "symbol": (
                f"{PPR_NAMESPACE}feature"
                if evidence_type == "document"
                else f"{PPR_NAMESPACE}product"
            ),
            "description": semantic_need,
        },
        "needed_context": {
            "context_ref": context_ref,
            "request_live_observation": False,
            "clarification_question": None,
        },
        "context understanding complete": False,
        "grounding_status": "waiting_for_evidence",
    }


def request_live_observation(
    *,
    semantic_need: str = "fresh scene arrangement",
) -> dict[str, object]:
    """Return one valid Phase 4.3 live-observation request."""
    return {
        "unresolved_semantic_need": {
            "kind": "class",
            "symbol": f"{PPR_NAMESPACE}product",
            "description": semantic_need,
        },
        "needed_context": {
            "context_ref": None,
            "request_live_observation": True,
            "clarification_question": None,
        },
        "context understanding complete": False,
        "grounding_status": "waiting_for_evidence",
    }


def request_clarification(question: str) -> dict[str, object]:
    """Return one valid Phase 4.3 clarification decision."""
    return {
        "unresolved_semantic_need": {
            "kind": "user_intent",
            "symbol": "product_requirement",
            "description": "unresolved user intent",
        },
        "needed_context": {
            "context_ref": None,
            "request_live_observation": False,
            "clarification_question": question,
        },
        "context understanding complete": False,
        "grounding_status": "waiting_for_user",
    }


def complete_context() -> dict[str, object]:
    """Return one valid Phase 4.3 completion decision."""
    return {
        "unresolved_semantic_need": None,
        "needed_context": None,
        "context understanding complete": True,
        "grounding_status": "complete",
    }


def _default_descriptors() -> list[GroundingProducerDescriptor]:
    return [
        GroundingProducerDescriptor.from_mapping(
            {
                "provider_id": "document_evidence",
                "description": "Read controlled document evidence.",
                "accepted_evidence_types": ["document"],
                "produced_record_types": ["DocumentOverviewRecord"],
                "prerequisites": {"DocumentOverviewRecord": []},
                "availability": True,
                "estimated_cost": 1,
            }
        ),
        GroundingProducerDescriptor.from_mapping(
            {
                "provider_id": "rgb_d_cad_grounding",
                "description": "Read controlled CAD or observation evidence.",
                "accepted_evidence_types": ["CAD", "observation"],
                "produced_record_types": [
                    "CADMeshRecord",
                    "RGBDSegmentationRecord",
                ],
                "prerequisites": {
                    "CADMeshRecord": [],
                    "RGBDSegmentationRecord": [],
                },
                "availability": True,
                "estimated_cost": 1,
            }
        ),
    ]

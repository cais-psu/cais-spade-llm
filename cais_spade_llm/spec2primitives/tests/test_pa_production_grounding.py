"""Tests for the simple requirement-to-ontology production grounding flow."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from rdflib import Namespace, URIRef
from rdflib.namespace import RDF

from cais_spade_llm.spec2primitives.agents.pa import (
    context_serving,
    production_grounding,
)
from cais_spade_llm.spec2primitives.agents.pa.context_assessment import (
    continue_pa_context_interaction,
)
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    start_pa_context_interaction,
)
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingActionAttempt,
    GroundingNextAction,
    GroundingProducerDescriptor,
    GroundingSession,
    PAContextGroundingCompletionV2,
    TypedGroundingContract,
    build_product_context_view,
    load_latest_grounding_session,
    load_pa_context_grounding_completion,
    persist_grounding_session,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    ABoxSnapshot,
    initialize_interaction_abox,
    load_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.agents.pa.production_grounding import (
    CameraToWorldCalibrationRuntime,
    ProductionGroundingError,
    ProductionProductContextGroundingRuntime,
)
from cais_spade_llm.spec2primitives.config import load_model_runtime_config
from cais_spade_llm.spec2primitives.ontology import TBoxSnapshot
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    PPR_NAMESPACE,
    ontology_config,
)
from cais_spade_llm.spec2primitives.tests.test_cad_pose_estimation import (
    _accepted_registration,
)
from cais_spade_llm.spec2primitives.tests.test_cad_size_correspondence import (
    _prepare_inputs,
    _size_bundle,
)
from cais_spade_llm.spec2primitives.tests.test_document_interpretation import (
    ControlledVisionRuntime,
)
from cais_spade_llm.spec2primitives.tools.document_evidence import (
    prepare_document_overview,
)
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    resolve_context_ref,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    CameraToRobotCalibrationResult,
    record_camera_to_robot_calibration,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    pose_estimation as pose_module,
)

PPR = Namespace(PPR_NAMESPACE)
_TEST_DOCUMENT_REF = "NIST_assembly_instructions.pdf"
ActionValue = Mapping[str, object] | Callable[[Mapping[str, object]], Mapping[str, object]]
OntologyValue = Mapping[str, object] | Callable[[Mapping[str, object]], Mapping[str, object]]


class SimpleProductAgent:
    """Return controlled minimal actions and final ontology proposals."""

    def __init__(
        self,
        actions: Sequence[ActionValue],
        *,
        ontology_outputs: Sequence[OntologyValue] | None = None,
    ) -> None:
        self.actions = list(actions)
        self.ontology_outputs = list(ontology_outputs or [_valid_ontology_output])
        self.calls: list[dict[str, object]] = []

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        """Return one controlled response and retain the exact prompt."""
        payload = _prompt_payload(prompt)
        self.calls.append(
            {
                "prompt": prompt,
                "response_format": response_format,
                "payload": payload,
            }
        )
        if response_format["name"] == "spec2primitives_grounding_action":
            if not self.actions:
                raise AssertionError("No controlled grounding action remains.")
            action = self.actions.pop(0)
            value = action(payload) if callable(action) else action
            return {"next_action": dict(value)}
        if response_format["name"] == "spec2primitives_ontology_grounding_proposal":
            if not self.ontology_outputs:
                raise AssertionError("No controlled ontology output remains.")
            output = self.ontology_outputs.pop(0)
            value = output(payload) if callable(output) else output
            return {"ontology_grounding_proposal": dict(value)}
        raise AssertionError(f"Unexpected structured request: {response_format['name']}")


class _SyntheticWorldPoseProductionRuntime(
    ProductionProductContextGroundingRuntime
):
    """Add one alternative accepted world-pose output to a served CAD delta."""

    async def interpret_served_context(
        self,
        *,
        producer: str,
        interaction_root: Path,
        tbox: object,
        abox: object,
        served_context: Mapping[str, object],
        operation_number: int,
    ) -> Mapping[str, object]:
        delta = dict(
            await super().interpret_served_context(  # type: ignore[arg-type]
                producer=producer,
                interaction_root=interaction_root,
                tbox=tbox,
                abox=abox,
                served_context=served_context,
                operation_number=operation_number,
            )
        )
        if served_context.get("evidence_type") == "CAD":
            refs = list(delta.get("typed_context_refs", []))
            refs.append(_write_synthetic_world_pose(interaction_root, producer))
            delta["typed_context_refs"] = refs
        return delta


class _ControlledCameraToWorldCalibrationRuntime:
    """Persist one approved test calibration for the requested exact frames."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def materialize_camera_to_world_calibration(
        self,
        *,
        interaction_root: Path,
        camera_pose_record_path: Path,
        source_frame: str,
        target_frame: str,
        calibration_number: int,
    ) -> CameraToRobotCalibrationResult:
        """Return a pinned transform that places the test gear in xarm6 reach."""
        self.calls.append(
            {
                "camera_pose_record_path": camera_pose_record_path,
                "source_frame": source_frame,
                "target_frame": target_frame,
                "calibration_number": calibration_number,
            }
        )
        world_from_camera = np.eye(4)
        world_from_camera[:3, 3] = [-0.12, -0.46, 0.35]
        return record_camera_to_robot_calibration(
            interaction_root=interaction_root,
            calibration_id="approved-world-v1",
            source_frame=source_frame,
            target_frame=target_frame,
            target_from_camera_transform=world_from_camera,
            valid_from_ns=0,
            valid_until_ns=None,
            provenance_source="approved_test_calibration",
            provenance_sha256="a" * 64,
            calibration_number=calibration_number,
        )


def _runtime(
    vision: ControlledVisionRuntime | None = None,
    *,
    synthetic_world_pose: bool = False,
    calibration_runtime: CameraToWorldCalibrationRuntime | None = None,
) -> ProductionProductContextGroundingRuntime:
    runtime_type = (
        _SyntheticWorldPoseProductionRuntime
        if synthetic_world_pose
        else ProductionProductContextGroundingRuntime
    )
    return runtime_type(
        tbox=ontology_config().load_tbox(),
        document_config=load_model_runtime_config().document_vlm,
        document_vision_runtime=vision or ControlledVisionRuntime(),
        camera_to_world_calibration_runtime=calibration_runtime,
    )


def _valid_ontology_output(
    payload: Mapping[str, object],
    *,
    context_summary: str = (
        "The NIST document describes gear assembly and the approved catalog contains "
        "Gear_Medium.STL. The exact shaft and meshing partner are not specified."
    ),
    missing_information: Sequence[str] = (
        "The exact shaft and meshing partner are not specified.",
    ),
) -> Mapping[str, object]:
    allowed = [str(item) for item in payload["allowed_evidence_refs"]]  # type: ignore[index]
    evidence_ref = next(
        (item for item in allowed if item.endswith("#page=4")),
        "requirement_0001",
    )
    predefined_process = payload.get(
        "predefined_process",
        {"process_iri": "https://cais-spade-llm.local/process/assembly"},
    )
    assert isinstance(predefined_process, Mapping)
    return {
        "individuals": [
            {"individual_index": 1, "class_iri": str(PPR.feature)},
        ],
        "relations": [
            {
                "subject_kind": "specification",
                "subject_individual_index": None,
                "subject_iri": None,
                "predicate_iri": str(PPR.defines),
                "object_kind": "new_individual",
                "object_individual_index": 1,
                "object_iri": None,
            },
            {
                "subject_kind": "existing_individual",
                "subject_individual_index": None,
                "subject_iri": predefined_process["process_iri"],
                "predicate_iri": str(PPR.realizes),
                "object_kind": "new_individual",
                "object_individual_index": 1,
                "object_iri": None,
            },
        ],
        "literal_facts": [],
        "context_summary": context_summary,
        "evidence_refs": [evidence_ref],
        "missing_information": list(missing_information),
    }


def _prompt_payload(prompt: str) -> Mapping[str, object]:
    marker = "Grounding input:\n"
    if marker not in prompt:
        marker = "Late mapping input:\n"
    raw = prompt.split(marker, 1)[1]
    value, _ = json.JSONDecoder().raw_decode(raw)
    assert isinstance(value, Mapping)
    return value


def _prepare_overview(cache_root: Path, vision: ControlledVisionRuntime) -> None:
    served = resolve_context_ref({"context_ref": _TEST_DOCUMENT_REF})["served_context"]
    asyncio.run(
        prepare_document_overview(
            served_context=served,
            cache_root=cache_root,
            config=load_model_runtime_config().document_vlm,
            vision_runtime=vision,
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_revision_record(
    interaction_root: Path,
    name: str,
    value: Mapping[str, object],
) -> str:
    """Persist one minimal typed record used by revision-routing tests."""
    path = interaction_root / "products/grounding/revision_state" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path.relative_to(interaction_root).as_posix()


def _hashed_record_ref(interaction_root: Path, record_ref: str) -> dict[str, str]:
    """Return one exact local record ref and current SHA-256."""
    return {
        "ref": record_ref,
        "sha256": hashlib.sha256(
            (interaction_root / record_ref).read_bytes()
        ).hexdigest(),
    }


def _write_synthetic_world_pose(interaction_root: Path, producer: str) -> str:
    record_root = interaction_root / "products/grounding/synthetic_world_pose"
    record_root.mkdir(parents=True, exist_ok=True)
    source_path = record_root / "source_evidence_0001.json"
    source_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "SyntheticWorldPoseEvidence",
                "producer": producer,
                "status": "accepted",
                "observation_timestamp_ns": 1,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    source_ref = source_path.relative_to(interaction_root).as_posix()
    record_path = record_root / "world_pose_0001.json"
    record_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "RobotFramePoseRecord",
                "producer": producer,
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
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(record_path.relative_to(interaction_root))


def _prepare_resource_grounding_context(
    interaction_root: Path,
    *,
    candidate_diameters_m: tuple[float, ...],
) -> tuple[TBoxSnapshot, ABoxSnapshot, Mapping[str, object]]:
    """Persist provisional semantics plus accepted CAD and RGB-D source records."""
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(
        interaction_root,
        "assemble medium gear",
        tbox,
    )
    feature_iri = f"{abox.namespace}medium_gear_feature"
    validate_and_merge_triple_delta(
        interaction_root,
        tbox,
        "ontology_grounding_host",
        {
            "assertions": [
                {
                    "subject": feature_iri,
                    "predicate": str(RDF.type),
                    "object": {"kind": "iri", "value": str(PPR.feature)},
                    "evidence_refs": ["requirement_0001"],
                },
                {
                    "subject": abox.specification_iri,
                    "predicate": str(PPR.defines),
                    "object": {"kind": "iri", "value": feature_iri},
                    "evidence_refs": ["requirement_0001"],
                },
                {
                    "subject": "https://cais-spade-llm.local/process/assembly",
                    "predicate": str(PPR.realizes),
                    "object": {"kind": "iri", "value": feature_iri},
                    "evidence_refs": ["requirement_0001"],
                },
            ],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [],
        },
        authorized_evidence_refs=["requirement_0001"],
    )
    segmentation_path, cad_path = _prepare_inputs(
        interaction_root,
        _size_bundle(candidate_diameters_m),
    )
    external_merge = validate_and_merge_triple_delta(
        interaction_root,
        tbox,
        "rgb_d_cad_grounding",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [
                segmentation_path.relative_to(interaction_root).as_posix(),
                cad_path.relative_to(interaction_root).as_posix(),
            ],
        },
        authorized_evidence_refs=["Gear_Medium.STL", "observation_0001"],
    )
    persist_grounding_session(
        interaction_root,
        GroundingSession.create(
            revision=1,
            requirement_text=external_merge.abox.product_requirement,
            next_action=GroundingNextAction.from_mapping(
                {"action": "propose_grounding"}
            ),
            status="ready_for_ontology",
        ),
    )
    view = build_product_context_view(
        interaction_root,
        external_merge.abox,
        attempted_evidence=("Gear_Medium.STL", "observation_0001"),
        assessed_at_ns=3_000_004_000,
    )
    return tbox, external_merge.abox, view.to_record()


def _append_approved_cad(
    interaction_root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    *,
    context_ref: str,
    operation_number: int,
) -> ABoxSnapshot:
    """Append one approved CAD record without making it the active hypothesis."""
    resolution = resolve_context_ref({"context_ref": context_ref})
    served_context = resolution.get("served_context")
    assert isinstance(served_context, Mapping)
    preprocessing = production_grounding.preprocess_served_geometry(
        interaction_root=interaction_root,
        served_context=served_context,
        operation_number=operation_number,
    )
    merge = validate_and_merge_triple_delta(
        interaction_root,
        tbox,
        "rgb_d_cad_grounding",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [
                preprocessing.record_path.relative_to(interaction_root).as_posix()
            ],
        },
        authorized_evidence_refs=[context_ref],
    )
    return merge.abox


@pytest.mark.parametrize(
    "requirement",
    [
        "assemble medium gear",
        "drill the mounting hole",
        "weld the frame joint",
        "inspect the finished surface",
    ],
)
def test_requirements_and_ontology_share_one_minimal_action_schema(
    tmp_path: Path,
    requirement: str,
) -> None:
    agent = SimpleProductAgent(
        [
            {"action": "propose_grounding"},
            {"action": "retrieve", "source_ref": "Gear_Medium.STL"},
        ]
    )

    result = asyncio.run(
        start_pa_context_interaction(
            agent,
            tmp_path,
            requirement,
            ontology_config=ontology_config(),
            grounding_runtime=_runtime(),
        )
    )

    assert result["needed_context"]["context_ref"] == "Gear_Medium.STL"
    first_call = agent.calls[0]
    assert first_call["response_format"]["name"] == "spec2primitives_grounding_action"
    action_schema = first_call["response_format"]["schema"]
    assert action_schema["type"] == "object"
    assert action_schema["required"] == ["next_action"]
    assert "anyOf" in action_schema["properties"]["next_action"]
    assert first_call["payload"]["exact_requirement"] == requirement  # type: ignore[index]
    assert first_call["payload"]["ontology"]["classes"]  # type: ignore[index]
    ontology_call = agent.calls[1]
    assert (
        ontology_call["response_format"]["name"]
        == "spec2primitives_ontology_grounding_proposal"
    )
    ontology_schema = ontology_call["response_format"]["schema"]
    assert ontology_schema["type"] == "object"
    assert "uniqueItems" not in json.dumps(ontology_schema)
    serialized = json.dumps(agent.calls, ensure_ascii=False)
    for legacy in (
        "new_statements",
        "new_information_needs",
        "information_need_transitions",
        "answer_statement_ids",
        "directly_stated",
        '"understanding"',
    ):
        assert legacy not in serialized


def test_generic_overview_then_turn_two_fills_medium_gear_ontology(
    tmp_path: Path,
) -> None:
    interaction_root = tmp_path / "interaction"
    vision = ControlledVisionRuntime()
    agent = SimpleProductAgent(
        [
            {"action": "retrieve", "source_ref": _TEST_DOCUMENT_REF},
            {"action": "propose_grounding"},
            {"action": "retrieve", "source_ref": "Gear_Medium.STL"},
            {"action": "incomplete", "reason": "World pose is still missing."},
        ]
    )
    runtime = _runtime(vision)

    first = asyncio.run(
        start_pa_context_interaction(
            agent,
            interaction_root,
            "assemble medium gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )
    assert first["needed_context"]["context_ref"] == _TEST_DOCUMENT_REF
    assert vision.requests == []

    context_serving.serve_pa_requested_context(interaction_root)
    result = asyncio.run(
        continue_pa_context_interaction(
            agent,
            interaction_root,
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
            max_pa_turns=6,
        )
    )

    assert result["grounding_status"] == "incomplete"
    assert len(vision.requests) == 1
    assert vision.targeted_requests == []
    turn_2 = _read_json(interaction_root / "interaction_record/turn_0002.json")
    assert turn_2["PA_output"] is not None
    assert turn_2["failure"] is None

    session = load_latest_grounding_session(interaction_root)
    assert session is not None and session.status == "incomplete"
    session_record = session.to_record()
    for legacy in (
        "statements",
        "information_needs",
        "information_need_transitions",
        "answer_statement_ids",
        "understanding",
    ):
        assert legacy not in json.dumps(session_record)

    proposal = _read_json(
        interaction_root
        / "products/grounding/ontology_grounding/proposal_0001.json"
    )
    proposal_value = proposal["output"]["ontology_grounding_proposal"]
    assert proposal_value["context_summary"] == (
        "The NIST document describes gear assembly and the approved catalog contains "
        "Gear_Medium.STL. The exact shaft and meshing partner are not specified."
    )
    assert proposal_value["missing_information"] == [
        "The exact shaft and meshing partner are not specified."
    ]
    assert proposal_value["evidence_refs"] == [f"{_TEST_DOCUMENT_REF}#page=4"]
    manifest = _read_json(
        interaction_root / "products/grounding/ontology/abox_manifest.json"
    )
    assert manifest["accepted_assertion_count"] == 3


def test_turn_two_can_request_and_serve_another_source(
    tmp_path: Path,
) -> None:
    interaction_root = tmp_path / "interaction"
    agent = SimpleProductAgent(
        [
            {"action": "retrieve", "source_ref": _TEST_DOCUMENT_REF},
            {"action": "retrieve", "source_ref": "Gear_Medium.STL"},
            {"action": "propose_grounding"},
            {"action": "incomplete", "reason": "World pose is still missing."},
        ]
    )
    runtime = _runtime(ControlledVisionRuntime())

    first = asyncio.run(
        start_pa_context_interaction(
            agent,
            interaction_root,
            "assemble medium gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )
    assert first["needed_context"]["context_ref"] == _TEST_DOCUMENT_REF
    context_serving.serve_pa_requested_context(interaction_root)

    result = asyncio.run(
        continue_pa_context_interaction(
            agent,
            interaction_root,
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
            max_pa_turns=6,
        )
    )

    assert result["grounding_status"] == "incomplete"
    turn_2 = _read_json(interaction_root / "interaction_record/turn_0002.json")
    assert turn_2["PA_output"] == {
        "needed_context": {
            "context_ref": "Gear_Medium.STL",
            "request_live_observation": False,
            "clarification_question": None,
        },
        "context understanding complete": False,
        "grounding_status": "waiting_for_evidence",
    }
    retrieval_2 = _read_json(
        interaction_root / "interaction_record/retrieval_0002.json"
    )
    assert retrieval_2["served_context"]["context_ref"] == "Gear_Medium.STL"


def test_optional_inspect_uses_all_six_pages_and_preserves_attempt_provenance(
    tmp_path: Path,
) -> None:
    interaction_root = tmp_path / "interaction"
    vision = ControlledVisionRuntime(
        targeted_output={
            "observations": [
                {
                    "description": "Page 4 describes gear plate and shaft assembly.",
                    "evidence_pages": [4],
                }
            ],
            "uncertainty": [],
        }
    )
    _prepare_overview(tmp_path / "source_cache", vision)
    vision.requests.clear()
    question = "What assembly guidance applies to Medium Gear?"
    agent = SimpleProductAgent(
        [
            {
                "action": "inspect",
                "source_ref": _TEST_DOCUMENT_REF,
                "question": question,
            },
            {"action": "propose_grounding"},
            {"action": "incomplete", "reason": "World pose is still missing."},
        ]
    )
    runtime = _runtime(vision)

    first = asyncio.run(
        start_pa_context_interaction(
            agent,
            interaction_root,
            "assemble medium gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )
    assert first["needed_context"]["context_ref"] == _TEST_DOCUMENT_REF
    context_serving.serve_pa_requested_context(interaction_root)
    result = asyncio.run(
        continue_pa_context_interaction(
            agent,
            interaction_root,
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
            max_pa_turns=6,
        )
    )

    assert result["grounding_status"] == "incomplete"
    assert len(vision.targeted_requests) == 1
    request = vision.targeted_requests[0]
    assert request.evidence_question == question
    assert [page.page_number for page in request.pages] == [1, 2, 3, 4, 5, 6]
    evidence_ref = "products/grounding/document_evidence/evidence_0001.json"
    evidence = _read_json(interaction_root / evidence_ref)
    assert evidence["selected_pages"] == [1, 2, 3, 4, 5, 6]
    assert evidence["observations"][0]["evidence_refs"] == [
        f"{_TEST_DOCUMENT_REF}#page=4"
    ]
    session = load_latest_grounding_session(interaction_root)
    assert session is not None
    assert session.attempted_actions[0].action == "inspect"
    assert session.attempted_actions[0].question == question
    assert session.attempted_actions[0].record_refs == (evidence_ref,)


@pytest.mark.parametrize(
    ("action", "expected_status"),
    [
        (
            {"action": "ask_user", "question": "Which gear should be assembled?"},
            "waiting_for_user",
        ),
        (
            {"action": "incomplete", "reason": "No safe evidence is available."},
            "incomplete",
        ),
    ],
)
def test_terminal_and_clarification_actions_keep_public_output_simple(
    tmp_path: Path,
    action: Mapping[str, object],
    expected_status: str,
) -> None:
    agent = SimpleProductAgent([action])

    result = asyncio.run(
        start_pa_context_interaction(
            agent,
            tmp_path,
            "assemble gear",
            ontology_config=ontology_config(),
            grounding_runtime=_runtime(),
        )
    )

    session = load_latest_grounding_session(tmp_path)
    assert session is not None and session.status == expected_status
    if expected_status == "waiting_for_user":
        assert session.next_action.question == "Which gear should be assembled?"
        assert result.get("failure") is None
    else:
        assert result["grounding_status"] == expected_status
        assert result["needed_context"] is None


def test_invalid_or_unauthorized_action_gets_one_repair_then_incomplete(
    tmp_path: Path,
) -> None:
    invalid = {"action": "retrieve", "source_ref": "unauthorized.pdf"}
    agent = SimpleProductAgent([invalid, invalid])

    result = asyncio.run(
        start_pa_context_interaction(
            agent,
            tmp_path,
            "assemble medium gear",
            ontology_config=ontology_config(),
            grounding_runtime=_runtime(),
        )
    )

    assert result["grounding_status"] == "incomplete"
    assert result["needed_context"] is None
    assert len(agent.calls) == 2
    assert "prior response was rejected" in str(agent.calls[1]["prompt"])
    session = load_latest_grounding_session(tmp_path)
    assert session is not None
    assert "after one repair" in str(session.next_action.reason)
    assert session.attempted_actions == ()


def test_exact_replay_gets_one_repair_then_persists_incomplete(
    tmp_path: Path,
) -> None:
    cad_action = {"action": "retrieve", "source_ref": "Gear_Medium.STL"}
    agent = SimpleProductAgent([cad_action, cad_action, cad_action])
    runtime = _runtime()

    first = asyncio.run(
        start_pa_context_interaction(
            agent,
            tmp_path,
            "assemble medium gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )
    assert first["needed_context"]["context_ref"] == "Gear_Medium.STL"
    context_serving.serve_pa_requested_context(tmp_path)
    result = asyncio.run(
        continue_pa_context_interaction(
            agent,
            tmp_path,
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
            max_pa_turns=4,
        )
    )

    assert result["grounding_status"] == "incomplete"
    session = load_latest_grounding_session(tmp_path)
    assert session is not None and session.status == "incomplete"
    assert len(session.attempted_actions) == 1
    assert session.attempted_actions[0].source_ref == "Gear_Medium.STL"
    assert len(
        [
            call
            for call in agent.calls
            if call["response_format"]["name"] == "spec2primitives_grounding_action"  # type: ignore[index]
        ]
    ) == 3


def test_unresolved_selected_cad_withholds_other_cad_choices(
    tmp_path: Path,
) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble medium gear", tbox)
    abox = _append_approved_cad(
        tmp_path,
        tbox,
        abox,
        context_ref="Gear_Medium.STL",
        operation_number=1,
    )
    runtime = _runtime()
    view = build_product_context_view(
        tmp_path,
        abox,
        attempted_evidence=("Gear_Medium.STL",),
        assessed_at_ns=1,
    )
    cad = next(
        binding
        for binding in view.typed_bindings
        if binding.record_type == "CADMeshRecord"
    )
    source = _read_json(tmp_path / cad.record_ref)["source"]
    previous = GroundingSession.create(
        revision=1,
        requirement_text=abox.product_requirement,
        next_action=GroundingNextAction.from_mapping(
            {"action": "retrieve", "source_ref": "Gear_Medium.STL"}
        ),
        selected_provider_id="rgb_d_cad_grounding",
        selected_source_revision=source["source_sha256"],
        status="waiting_for_evidence",
    )
    actions = production_grounding._discover_provider_actions(
        tmp_path,
        runtime.grounding_producer_descriptors(),
        previews=[
            {
                "provider_id": "rgb_d_cad_grounding",
                "evidence_type": "CAD",
                "source_ref": "Gear_Medium.STL",
                "source_revision": source["source_sha256"],
                "availability": True,
                "produced_record_types": ["CADMeshRecord"],
            },
            {
                "provider_id": "rgb_d_cad_grounding",
                "evidence_type": "CAD",
                "source_ref": "Gear_Large.STL",
                "source_revision": "b" * 64,
                "availability": True,
                "produced_record_types": ["CADMeshRecord"],
            },
            {
                "provider_id": "rgb_d_cad_grounding",
                "evidence_type": "observation",
                "source_ref": "live_RGB-D",
                "source_revision": "c" * 64,
                "availability": True,
                "produced_record_types": [
                    "ColoredPointCloudSetRecord",
                    "RGBDSegmentationRecord",
                ],
            },
        ],
        view=view,
        previous=previous,
        pending_attempt=None,
        required_record_type="RobotFramePoseRecord",
        required_target_frame="world",
    )

    assert any(
        action.source_ref == "live_RGB-D" and not action.automatic
        for action in actions
    )
    assert not any(
        action.source_ref == "Gear_Large.STL"
        for action in actions
    )
    assert not any(
        action.source_ref == "Gear_Medium.STL"
        for action in actions
    )
    segmentation_ref = _write_revision_record(
        tmp_path,
        "selected_before_rgbd_segmentation",
        {
            "schema_version": 1,
            "record_type": "RGBDSegmentationRecord",
            "producer": "rgb_d_cad_grounding",
            "cameras": [{"timestamp_ns": 2}],
        },
    )
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        "rgb_d_cad_grounding",
        {"assertions": [], "typed_context_refs": [segmentation_ref]},
        authorized_evidence_refs=[],
    )
    view_after_observation = build_product_context_view(
        tmp_path,
        merge.abox,
        attempted_evidence=("Gear_Medium.STL", "observation_0001"),
        assessed_at_ns=2,
    )
    cad_attempt = GroundingActionAttempt.from_mapping(
        {
            "attempt_id": "attempt_0001",
            "action": "retrieve",
            "provider_id": "rgb_d_cad_grounding",
            "source_ref": "Gear_Medium.STL",
            "source_revision": source["source_sha256"],
            "question": None,
            "status": "accepted",
            "record_refs": [cad.record_ref],
        }
    )
    observation_session = GroundingSession.create(
        revision=2,
        requirement_text=merge.abox.product_requirement,
        attempted_actions=(cad_attempt,),
        next_action=GroundingNextAction.from_mapping(
            {"action": "retrieve", "source_ref": "live_RGB-D"}
        ),
        selected_provider_id="rgb_d_cad_grounding",
        selected_source_revision="c" * 64,
        status="waiting_for_evidence",
    )
    actions_after_observation = production_grounding._discover_provider_actions(
        tmp_path,
        runtime.grounding_producer_descriptors(),
        previews=[],
        view=view_after_observation,
        previous=observation_session,
        pending_attempt=None,
        required_record_type="RobotFramePoseRecord",
        required_target_frame="world",
    )
    correspondence_action = next(
        action
        for action in actions_after_observation
        if action.produced_record_types == ("CADSizeCorrespondenceRecord",)
    )
    exact_bindings = dict(correspondence_action.prerequisite_bindings)
    assert correspondence_action.automatic is True
    assert exact_bindings["CADMeshRecord"].record_ref == cad.record_ref
    assert (
        exact_bindings["RGBDSegmentationRecord"].record_ref
        == segmentation_ref
    )


def test_pose_failure_recaptures_for_the_same_selected_cad(
    tmp_path: Path,
) -> None:
    producer = "rgb_d_cad_grounding"
    cad_ref = _write_revision_record(
        tmp_path,
        "pose_retry_cad",
        {
            "schema_version": 1,
            "record_type": "CADMeshRecord",
            "producer": producer,
            "source": {
                "context_ref": "Gear_Medium.STL",
                "source_sha256": "a" * 64,
            },
        },
    )
    segmentation_1_ref = _write_revision_record(
        tmp_path,
        "pose_retry_segmentation_1",
        {
            "schema_version": 1,
            "record_type": "RGBDSegmentationRecord",
            "producer": producer,
            "cameras": [{"timestamp_ns": 100}],
        },
    )
    correspondence_ref = _write_revision_record(
        tmp_path,
        "pose_retry_correspondence",
        {
            "schema_version": 1,
            "record_type": "CADSizeCorrespondenceRecord",
            "producer": producer,
            "CAD_correspondence": "accepted",
            "CAD": {
                "context_ref": "Gear_Medium.STL",
                "source_sha256": "a" * 64,
                "record": _hashed_record_ref(tmp_path, cad_ref),
            },
            "segmentation": {
                "record": _hashed_record_ref(tmp_path, segmentation_1_ref)
            },
        },
    )
    failed_pose_ref = _write_revision_record(
        tmp_path,
        "pose_retry_failed_pose",
        {
            "schema_version": 1,
            "record_type": "CADPoseEstimationRecord",
            "producer": producer,
            "pose": "rejected",
            "source_correspondence": _hashed_record_ref(
                tmp_path,
                correspondence_ref,
            ),
        },
    )
    tbox = ontology_config().load_tbox()
    initialize_interaction_abox(tmp_path, "assemble medium gear", tbox)
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        producer,
        {
            "assertions": [],
            "typed_context_refs": [
                cad_ref,
                segmentation_1_ref,
                correspondence_ref,
                failed_pose_ref,
            ],
        },
        authorized_evidence_refs=[],
    )
    runtime = _runtime()
    view = build_product_context_view(
        tmp_path,
        merge.abox,
        attempted_evidence=(),
        assessed_at_ns=100,
    )
    cad = next(
        binding
        for binding in view.typed_bindings
        if binding.record_ref == cad_ref
    )
    segmentation_1 = next(
        binding
        for binding in view.typed_bindings
        if binding.record_ref == segmentation_1_ref
    )
    pair_revision = production_grounding._cad_segmentation_revision(
        tmp_path,
        cad,
        segmentation_1,
    )
    selected_session = GroundingSession.create(
        revision=2,
        requirement_text=merge.abox.product_requirement,
        next_action=GroundingNextAction.from_mapping(
            {"action": "retrieve", "source_ref": "Gear_Medium.STL"}
        ),
        selected_provider_id=producer,
        selected_source_revision=pair_revision,
        status="waiting_for_evidence",
    )
    actions = production_grounding._discover_provider_actions(
        tmp_path,
        runtime.grounding_producer_descriptors(),
        previews=[
            {
                "provider_id": producer,
                "evidence_type": "observation",
                "source_ref": "live_RGB-D",
                "source_revision": "b" * 64,
                "availability": True,
                "produced_record_types": [
                    "ColoredPointCloudSetRecord",
                    "RGBDSegmentationRecord",
                ],
            }
        ],
        view=view,
        previous=selected_session,
        pending_attempt=None,
        required_record_type="RobotFramePoseRecord",
        required_target_frame="world",
    )
    assert any(
        action.evidence_type == "observation" and not action.automatic
        for action in actions
    )

    segmentation_2_ref = _write_revision_record(
        tmp_path,
        "pose_retry_segmentation_2",
        {
            "schema_version": 1,
            "record_type": "RGBDSegmentationRecord",
            "producer": producer,
            "cameras": [{"timestamp_ns": 200}],
        },
    )
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        producer,
        {"assertions": [], "typed_context_refs": [segmentation_2_ref]},
        authorized_evidence_refs=[],
    )
    cad_attempt = GroundingActionAttempt.from_mapping(
        {
            "attempt_id": "attempt_0001",
            "action": "retrieve",
            "provider_id": producer,
            "source_ref": "Gear_Medium.STL",
            "source_revision": pair_revision,
            "question": None,
            "status": "accepted",
            "record_refs": [correspondence_ref],
        }
    )
    recapture_session = GroundingSession.create(
        revision=3,
        requirement_text=merge.abox.product_requirement,
        attempted_actions=(cad_attempt,),
        next_action=GroundingNextAction.from_mapping(
            {"action": "retrieve", "source_ref": "live_RGB-D"}
        ),
        selected_provider_id=producer,
        selected_source_revision="b" * 64,
        status="waiting_for_evidence",
    )
    recaptured_view = build_product_context_view(
        tmp_path,
        merge.abox,
        attempted_evidence=(),
        assessed_at_ns=200,
    )
    recaptured_actions = production_grounding._discover_provider_actions(
        tmp_path,
        runtime.grounding_producer_descriptors(),
        previews=[],
        view=recaptured_view,
        previous=recapture_session,
        pending_attempt=None,
        required_record_type="RobotFramePoseRecord",
        required_target_frame="world",
    )
    correspondence_action = next(
        action
        for action in recaptured_actions
        if action.produced_record_types == ("CADSizeCorrespondenceRecord",)
    )
    exact_bindings = dict(correspondence_action.prerequisite_bindings)
    assert correspondence_action.automatic is True
    assert exact_bindings["CADMeshRecord"].record_ref == cad_ref
    assert exact_bindings["RGBDSegmentationRecord"].record_ref == segmentation_2_ref


def test_unauthorized_final_evidence_repairs_then_becomes_ontology_gap(
    tmp_path: Path,
) -> None:
    invalid_ontology = {
        **_valid_ontology_output(
            {"allowed_evidence_refs": ["requirement_0001"]}
        ),
        "evidence_refs": ["unauthorized_source"],
    }
    agent = SimpleProductAgent(
        [{"action": "propose_grounding"}],
        ontology_outputs=[invalid_ontology, invalid_ontology],
    )

    result = asyncio.run(
        start_pa_context_interaction(
            agent,
            tmp_path,
            "assemble medium gear",
            ontology_config=ontology_config(),
            grounding_runtime=_runtime(),
        )
    )

    assert result["grounding_status"] == "ontology_gap"
    assert result["needed_context"] is None
    proposal = _read_json(
        tmp_path / "products/grounding/ontology_grounding/proposal_0001.json"
    )
    assert proposal["status"] == "rejected"
    assert "after one repair" in str(proposal["failure"]) or "unauthorized" in str(
        proposal["failure"]
    )
    manifest = _read_json(tmp_path / "products/grounding/ontology/abox_manifest.json")
    assert manifest["delta_count"] == 0
    assert manifest["accepted_assertion_count"] == 0


def test_duplicate_final_evidence_is_rejected_by_host(
    tmp_path: Path,
) -> None:
    invalid_ontology = {
        **_valid_ontology_output(
            {"allowed_evidence_refs": ["requirement_0001"]}
        ),
        "evidence_refs": ["requirement_0001", "requirement_0001"],
    }
    agent = SimpleProductAgent(
        [{"action": "propose_grounding"}],
        ontology_outputs=[invalid_ontology, invalid_ontology],
    )

    result = asyncio.run(
        start_pa_context_interaction(
            agent,
            tmp_path,
            "assemble medium gear",
            ontology_config=ontology_config(),
            grounding_runtime=_runtime(),
        )
    )

    assert result["grounding_status"] == "ontology_gap"
    proposal = _read_json(
        tmp_path / "products/grounding/ontology_grounding/proposal_0001.json"
    )
    assert proposal["status"] == "rejected"
    assert "duplicate" in str(proposal["failure"])
    manifest = _read_json(tmp_path / "products/grounding/ontology/abox_manifest.json")
    assert manifest["delta_count"] == 0


def test_provider_actions_are_order_independent_and_accept_synthetic_provider(
    tmp_path: Path,
) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "inspect item", tbox)
    view = build_product_context_view(
        tmp_path,
        abox,
        attempted_evidence=(),
        assessed_at_ns=1,
    )
    synthetic = GroundingProducerDescriptor.from_mapping(
        {
            "provider_id": "synthetic_provider",
            "description": "Produce a synthetic approved record.",
            "accepted_evidence_types": ["CAD"],
            "produced_record_types": ["SyntheticRecord"],
            "prerequisites": {"SyntheticRecord": []},
            "availability": True,
            "estimated_cost": 2,
        }
    )
    geometry = _runtime().grounding_producer_descriptors()[1]
    previews = [
        {
            "provider_id": "synthetic_provider",
            "evidence_type": "CAD",
            "source_ref": "synthetic.source",
            "source_revision": "a" * 64,
            "availability": True,
            "produced_record_types": ["SyntheticRecord"],
        }
    ]

    forward = production_grounding._discover_provider_actions(
        tmp_path,
        [synthetic, geometry],
        previews=previews,
        view=view,
        previous=None,
        pending_attempt=None,
    )
    reverse = production_grounding._discover_provider_actions(
        tmp_path,
        [geometry, synthetic],
        previews=previews,
        view=view,
        previous=None,
        pending_attempt=None,
    )

    assert [item.to_record() for item in forward] == [
        item.to_record() for item in reverse
    ]
    assert forward[0].action == "retrieve"
    assert forward[0].produced_record_types == ("SyntheticRecord",)


def test_direct_world_pose_provider_can_parallel_the_camera_conversion_path(
    tmp_path: Path,
) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "inspect item", tbox)
    view = build_product_context_view(
        tmp_path,
        abox,
        attempted_evidence=(),
        assessed_at_ns=1,
    )
    direct = GroundingProducerDescriptor.from_mapping(
        {
            "provider_id": "synthetic_world_pose_provider",
            "description": "Produce one accepted world-frame pose record.",
            "accepted_evidence_types": ["observation"],
            "produced_record_types": ["RobotFramePoseRecord"],
            "prerequisites": {"RobotFramePoseRecord": []},
            "availability": True,
            "estimated_cost": 1,
        }
    )
    frame_conversion = next(
        descriptor
        for descriptor in _runtime().grounding_producer_descriptors()
        if descriptor.provider_id == "camera_pose_to_world"
    )

    actions = production_grounding._discover_provider_actions(
        tmp_path,
        [frame_conversion, direct],
        previews=[
            {
                "provider_id": "synthetic_world_pose_provider",
                "evidence_type": "observation",
                "source_ref": "synthetic_world_pose",
                "source_revision": "a" * 64,
                "availability": True,
                "produced_record_types": ["RobotFramePoseRecord"],
            }
        ],
        view=view,
        previous=None,
        pending_attempt=None,
        required_record_type="RobotFramePoseRecord",
        required_target_frame="world",
    )

    assert len(actions) == 1
    assert actions[0].provider_id == "synthetic_world_pose_provider"
    assert actions[0].produced_record_types == ("RobotFramePoseRecord",)


def test_observation_preprocessing_and_segmentation_run_off_event_loop_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(
        tmp_path,
        "assemble medium gear",
        tbox,
    )
    event_loop_thread = threading.get_ident()
    operation_threads: list[int] = []

    def preprocess(**_kwargs: Any) -> Any:
        operation_threads.append(threading.get_ident())
        return SimpleNamespace(
            delta={"typed_context_refs": []},
            record_path=tmp_path / "preprocessing_record.json",
        )

    def segment(**_kwargs: Any) -> Any:
        operation_threads.append(threading.get_ident())
        return SimpleNamespace(record_path=tmp_path / "segmentation_record.json")

    monkeypatch.setattr(
        production_grounding,
        "preprocess_served_geometry",
        preprocess,
    )
    monkeypatch.setattr(
        production_grounding,
        "segment_preprocessed_observation",
        segment,
    )

    delta = asyncio.run(
        _runtime().interpret_served_context(
            producer="rgb_d_cad_grounding",
            interaction_root=tmp_path,
            tbox=tbox,
            abox=abox,
            served_context={"evidence_type": "observation"},
            operation_number=1,
        )
    )

    assert delta["typed_context_refs"] == ["segmentation_record.json"]
    assert len(operation_threads) == 2
    assert all(thread_id != event_loop_thread for thread_id in operation_threads)


def test_pa_selected_cached_cad_runs_exact_derived_resource_grounding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tbox, abox, _view = _prepare_resource_grounding_context(
        tmp_path,
        candidate_diameters_m=(0.042,),
    )
    abox = _append_approved_cad(
        tmp_path,
        tbox,
        abox,
        context_ref="Gear_Large.STL",
        operation_number=3,
    )
    abox = _append_approved_cad(
        tmp_path,
        tbox,
        abox,
        context_ref="Gear_Shaft.STL",
        operation_number=4,
    )
    view = build_product_context_view(
        tmp_path,
        abox,
        attempted_evidence=(
            "Gear_Medium.STL",
            "Gear_Large.STL",
            "Gear_Shaft.STL",
            "observation_0001",
        ),
        assessed_at_ns=3_000_004_000,
    ).to_record()
    monkeypatch.setattr(pose_module, "_register_candidate", _accepted_registration)
    calibration_runtime = _ControlledCameraToWorldCalibrationRuntime()
    runtime = _runtime(calibration_runtime=calibration_runtime)
    agent = SimpleProductAgent(
        [{"action": "retrieve", "source_ref": "Gear_Medium.STL"}]
    )

    result = asyncio.run(
        runtime.assess_product_context(
            agent,
            interaction_root=tmp_path,
            tbox=tbox,
            abox=abox,
            abox_view=view,
            attempted_evidence=("Gear_Medium.STL", "observation_0001"),
            turn_number=2,
            max_pa_turns=12,
        )
    )

    assert result["grounding_status"] == "complete"
    assert result["context understanding complete"] is True
    assert len(agent.calls) == 1
    assert len(calibration_runtime.calls) == 1
    assert calibration_runtime.calls[0]["target_frame"] == "world"
    grounding_root = tmp_path / "products/grounding/rgb_d_cad_grounding"
    correspondence_path = (
        grounding_root / "correspondence_0001/correspondence_record.json"
    )
    assert correspondence_path.is_file()
    assert _read_json(correspondence_path)["CAD"]["context_ref"] == "Gear_Medium.STL"
    assert len(list(grounding_root.glob("correspondence_*"))) == 1
    assert (grounding_root / "pose_0001/pose_record.json").is_file()
    assert (grounding_root / "calibration_0001/calibration_record.json").is_file()
    assert (
        grounding_root / "robot_pose_0001/robot_frame_pose_record.json"
    ).is_file()
    selection = _read_json(
        tmp_path
        / "products/grounding/resource_selection/selection_0001"
        / "resource_selection_record.json"
    )
    assert selection["selected_resource_symbol"] == "xarm6"
    final_abox = load_interaction_abox(tmp_path, tbox)
    execution = URIRef(f"{final_abox.namespace}process_execution_0001")
    assert (
        execution,
        PPR.runsOnResource,
        URIRef("https://cais-spade-llm.local/resource/xarm6"),
    ) in final_abox.graph


def test_rejected_large_then_selected_medium_reuses_the_same_segmentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tbox, abox, _view = _prepare_resource_grounding_context(
        tmp_path,
        candidate_diameters_m=(0.042,),
    )
    abox = _append_approved_cad(
        tmp_path,
        tbox,
        abox,
        context_ref="Gear_Large.STL",
        operation_number=3,
    )
    view = build_product_context_view(
        tmp_path,
        abox,
        attempted_evidence=(
            "Gear_Medium.STL",
            "Gear_Large.STL",
            "observation_0001",
        ),
        assessed_at_ns=3_000_004_000,
    )
    monkeypatch.setattr(pose_module, "_register_candidate", _accepted_registration)
    agent = SimpleProductAgent(
        [
            {"action": "retrieve", "source_ref": "Gear_Large.STL"},
            {"action": "retrieve", "source_ref": "Gear_Medium.STL"},
        ]
    )

    result = asyncio.run(
        _runtime(
            calibration_runtime=_ControlledCameraToWorldCalibrationRuntime()
        ).assess_product_context(
            agent,
            interaction_root=tmp_path,
            tbox=tbox,
            abox=abox,
            abox_view=view.to_record(),
            attempted_evidence=(
                "Gear_Medium.STL",
                "Gear_Large.STL",
                "observation_0001",
            ),
            turn_number=2,
            max_pa_turns=12,
        )
    )

    assert result["grounding_status"] == "complete"
    assert len(agent.calls) == 2
    second_actions = agent.calls[1]["payload"]["available_actions"]  # type: ignore[index]
    assert not any(
        isinstance(action, Mapping) and action.get("source_ref") == "live_RGB-D"
        for action in second_actions
    )
    grounding_root = tmp_path / "products/grounding/rgb_d_cad_grounding"
    records = [
        _read_json(path)
        for path in sorted(grounding_root.glob("correspondence_*/correspondence_record.json"))
    ]
    assert [record["CAD"]["context_ref"] for record in records] == [
        "Gear_Large.STL",
        "Gear_Medium.STL",
    ]
    assert [record["CAD_correspondence"] for record in records] == [
        "rejected",
        "accepted",
    ]
    assert records[0]["segmentation"] == records[1]["segmentation"]


def test_ambiguous_selected_cad_runs_once_then_exposes_another_cad(
    tmp_path: Path,
) -> None:
    tbox, abox, view = _prepare_resource_grounding_context(
        tmp_path,
        candidate_diameters_m=(0.042, 0.042),
    )
    agent = SimpleProductAgent(
        [
            {"action": "retrieve", "source_ref": "Gear_Medium.STL"},
            {"action": "incomplete", "reason": "controlled test stop"},
        ]
    )

    result = asyncio.run(
        _runtime().assess_product_context(
            agent,
            interaction_root=tmp_path,
            tbox=tbox,
            abox=abox,
            abox_view=view,
            attempted_evidence=("Gear_Medium.STL", "observation_0001"),
            turn_number=2,
            max_pa_turns=12,
        )
    )

    assert result["grounding_status"] == "incomplete"
    assert len(agent.calls) == 2
    first_actions = agent.calls[0]["payload"]["available_actions"]  # type: ignore[index]
    assert [
        action.get("source_ref")
        for action in first_actions
        if isinstance(action, Mapping)
        and action.get("produced_record_types")
        == ["CADSizeCorrespondenceRecord"]
    ] == ["Gear_Medium.STL"]
    next_actions = agent.calls[1]["payload"]["available_actions"]  # type: ignore[index]
    assert not any(
        isinstance(action, Mapping) and action.get("source_ref") == "live_RGB-D"
        for action in next_actions
    )
    assert any(
        isinstance(action, Mapping)
        and action.get("source_ref") != "Gear_Medium.STL"
        and action.get("evidence_type") == "CAD"
        for action in next_actions
    )
    grounding_root = tmp_path / "products/grounding/rgb_d_cad_grounding"
    assert len(list(grounding_root.glob("correspondence_*"))) == 1
    correspondence = _read_json(
        grounding_root / "correspondence_0001/correspondence_record.json"
    )
    assert correspondence["CAD_correspondence"] == "ambiguous"
    assert list(grounding_root.glob("pose_*")) == []


def test_newer_rgbd_revision_invalidates_only_downstream_derived_records(
    tmp_path: Path,
) -> None:
    producer = "rgb_d_cad_grounding"
    cad_ref = _write_revision_record(
        tmp_path,
        "cad_1",
        {
            "schema_version": 1,
            "record_type": "CADMeshRecord",
            "producer": producer,
            "source": {
                "context_ref": "Gear_Medium.STL",
                "source_sha256": "a" * 64,
            },
        },
    )
    segmentation_1_ref = _write_revision_record(
        tmp_path,
        "segmentation_1",
        {
            "schema_version": 1,
            "record_type": "RGBDSegmentationRecord",
            "producer": producer,
            "cameras": [{"timestamp_ns": 100}],
        },
    )
    correspondence_1_ref = _write_revision_record(
        tmp_path,
        "correspondence_1",
        {
            "schema_version": 1,
            "record_type": "CADSizeCorrespondenceRecord",
            "producer": producer,
            "CAD_correspondence": "accepted",
            "CAD": {"record": _hashed_record_ref(tmp_path, cad_ref)},
            "segmentation": {
                "record": _hashed_record_ref(tmp_path, segmentation_1_ref)
            },
        },
    )
    pose_1_ref = _write_revision_record(
        tmp_path,
        "pose_1",
        {
            "schema_version": 1,
            "record_type": "CADPoseEstimationRecord",
            "producer": producer,
            "pose": "accepted",
            "coordinate_frame": "cam_mk3_optical_frame",
            "observation_timestamp_ns": 100,
            "source_correspondence": _hashed_record_ref(
                tmp_path,
                correspondence_1_ref,
            ),
        },
    )
    calibration_ref = _write_revision_record(
        tmp_path,
        "calibration_1",
        {
            "schema_version": 1,
            "record_type": "CameraToRobotCalibrationRecord",
            "producer": producer,
            "source_frame": "cam_mk3_optical_frame",
            "target_frame": "world",
            "validity": {"valid_from_ns": 0, "valid_until_ns": None},
        },
    )
    world_pose_1_ref = _write_revision_record(
        tmp_path,
        "world_pose_1",
        {
            "schema_version": 1,
            "record_type": "RobotFramePoseRecord",
            "producer": producer,
            "target_frame": "world",
            "observation_timestamp_ns": 100,
            "robot_frame_conversion": "accepted",
            "source_pose": _hashed_record_ref(tmp_path, pose_1_ref),
            "source_calibration": _hashed_record_ref(tmp_path, calibration_ref),
            "robot_frame_pose": {
                "CAD_origin_translation_m": [0.0, -0.5, 1.1]
            },
        },
    )
    segmentation_2_ref = _write_revision_record(
        tmp_path,
        "segmentation_2",
        {
            "schema_version": 1,
            "record_type": "RGBDSegmentationRecord",
            "producer": producer,
            "cameras": [{"timestamp_ns": 200}],
        },
    )
    tbox = ontology_config().load_tbox()
    initialize_interaction_abox(tmp_path, "assemble medium gear", tbox)
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        producer,
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [
                cad_ref,
                segmentation_1_ref,
                correspondence_1_ref,
                pose_1_ref,
                calibration_ref,
                world_pose_1_ref,
                segmentation_2_ref,
            ],
        },
        authorized_evidence_refs=[],
    )
    calibration_runtime = _ControlledCameraToWorldCalibrationRuntime()
    runtime = _runtime(calibration_runtime=calibration_runtime)
    descriptors = runtime.grounding_producer_descriptors()
    previews = [
        {
            "provider_id": producer,
            "evidence_type": "CAD",
            "source_ref": "Gear_Medium.STL",
            "source_revision": "a" * 64,
            "availability": True,
            "produced_record_types": ["CADMeshRecord"],
        },
        {
            "provider_id": producer,
            "evidence_type": "observation",
            "source_ref": "live_RGB-D",
            "source_revision": "b" * 64,
            "availability": True,
            "produced_record_types": [
                "ColoredPointCloudSetRecord",
                "RGBDSegmentationRecord",
            ],
        }
    ]

    def derived_outputs(
        abox: ABoxSnapshot,
        *,
        selected_by_pa: bool = False,
    ) -> list[tuple[str, ...]]:
        view = build_product_context_view(
            tmp_path,
            abox,
            attempted_evidence=(),
            assessed_at_ns=200,
        )
        previous = None
        if selected_by_pa:
            cad = next(
                binding
                for binding in view.typed_bindings
                if binding.record_type == "CADMeshRecord"
            )
            segmentation = max(
                (
                    binding
                    for binding in view.typed_bindings
                    if binding.record_type == "RGBDSegmentationRecord"
                ),
                key=lambda binding: int(binding.observed_at_ns or -1),
            )
            source_revision = production_grounding._cad_segmentation_revision(
                tmp_path,
                cad,
                segmentation,
            )
            previous = GroundingSession.create(
                revision=2,
                requirement_text=abox.product_requirement,
                next_action=GroundingNextAction.from_mapping(
                    {"action": "retrieve", "source_ref": "Gear_Medium.STL"}
                ),
                selected_provider_id=producer,
                selected_source_revision=source_revision,
                status="waiting_for_evidence",
            )
        actions = production_grounding._discover_provider_actions(
            tmp_path,
            descriptors,
            previews=previews,
            view=view,
            previous=previous,
            pending_attempt=None,
            required_record_type="RobotFramePoseRecord",
            required_target_frame="world",
        )
        return [
            action.produced_record_types
            for action in actions
            if action.evidence_type == "existing_record"
        ]

    assert derived_outputs(merge.abox) == [("CADSizeCorrespondenceRecord",)]
    correspondence_2_ref = _write_revision_record(
        tmp_path,
        "correspondence_2",
        {
            "schema_version": 1,
            "record_type": "CADSizeCorrespondenceRecord",
            "producer": producer,
            "CAD_correspondence": "accepted",
            "CAD": {"record": _hashed_record_ref(tmp_path, cad_ref)},
            "segmentation": {
                "record": _hashed_record_ref(tmp_path, segmentation_2_ref)
            },
        },
    )
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        producer,
        {"assertions": [], "typed_context_refs": [correspondence_2_ref]},
        authorized_evidence_refs=[],
    )
    assert derived_outputs(
        merge.abox,
        selected_by_pa=True,
    ) == [("CADPoseEstimationRecord",)]
    pose_2_ref = _write_revision_record(
        tmp_path,
        "pose_2",
        {
            "schema_version": 1,
            "record_type": "CADPoseEstimationRecord",
            "producer": producer,
            "pose": "accepted",
            "coordinate_frame": "cam_mk3_optical_frame",
            "observation_timestamp_ns": 200,
            "source_correspondence": _hashed_record_ref(
                tmp_path,
                correspondence_2_ref,
            ),
        },
    )
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        producer,
        {"assertions": [], "typed_context_refs": [pose_2_ref]},
        authorized_evidence_refs=[],
    )
    assert derived_outputs(
        merge.abox,
        selected_by_pa=True,
    ) == [("RobotFramePoseRecord",)]


def test_repeated_decision_frontier_does_not_reopen_live_observation(
    tmp_path: Path,
) -> None:
    producer = "rgb_d_cad_grounding"
    cad_ref = _write_revision_record(
        tmp_path,
        "frontier_cad",
        {
            "schema_version": 1,
            "record_type": "CADMeshRecord",
            "producer": producer,
            "source": {
                "context_ref": "Gear_Small.STL",
                "source_sha256": "a" * 64,
            },
        },
    )

    def correspondence(name: str, candidate_id: int) -> str:
        candidate = {
            "camera_id": "cam_mk3",
            "role": "source",
            "candidate_id": candidate_id,
            "within_size_tolerance": False,
        }
        return _write_revision_record(
            tmp_path,
            name,
            {
                "schema_version": 1,
                "record_type": "CADSizeCorrespondenceRecord",
                "producer": producer,
                "CAD_correspondence": "rejected",
                "CAD": {
                    "context_ref": "Gear_Small.STL",
                    "source_sha256": "a" * 64,
                },
                "ranked_candidates": [candidate],
                "plausible_candidates": [],
                "selected_candidate": None,
            },
        )

    first_ref = correspondence("frontier_correspondence_1", 1)
    tbox = ontology_config().load_tbox()
    initialize_interaction_abox(tmp_path, "assemble medium gear", tbox)
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        producer,
        {"assertions": [], "typed_context_refs": [cad_ref, first_ref]},
        authorized_evidence_refs=[],
    )

    def frontier(abox_value: ABoxSnapshot) -> str:
        view = build_product_context_view(
            tmp_path,
            abox_value,
            attempted_evidence=(),
            assessed_at_ns=1,
        )
        return production_grounding._observation_frontier_revision(tmp_path, view)

    first_frontier = frontier(merge.abox)
    duplicate_ref = correspondence("frontier_correspondence_2", 1)
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        producer,
        {"assertions": [], "typed_context_refs": [duplicate_ref]},
        authorized_evidence_refs=[],
    )
    assert frontier(merge.abox) == first_frontier
    unchanged_view = build_product_context_view(
        tmp_path,
        merge.abox,
        attempted_evidence=(),
        assessed_at_ns=2,
    )
    live_attempt = GroundingActionAttempt.from_mapping(
        {
            "attempt_id": "attempt_0001",
            "action": "retrieve",
            "provider_id": producer,
            "source_ref": "live_RGB-D",
            "source_revision": first_frontier,
            "question": None,
            "status": "no_change",
            "record_refs": [],
        }
    )
    previous = GroundingSession.create(
        revision=2,
        requirement_text=merge.abox.product_requirement,
        attempted_actions=(live_attempt,),
        next_action=GroundingNextAction.from_mapping(
            {"action": "incomplete", "reason": "controlled no progress"}
        ),
        status="incomplete",
    )
    actions = production_grounding._discover_provider_actions(
        tmp_path,
        _runtime().grounding_producer_descriptors(),
        previews=[
            {
                "provider_id": producer,
                "evidence_type": "observation",
                "source_ref": "live_RGB-D",
                "source_revision": first_frontier,
                "availability": True,
                "produced_record_types": [
                    "ColoredPointCloudSetRecord",
                    "RGBDSegmentationRecord",
                ],
            }
        ],
        view=unchanged_view,
        previous=previous,
        pending_attempt=None,
        required_record_type="RobotFramePoseRecord",
        required_target_frame="world",
    )
    assert not any(action.evidence_type == "observation" for action in actions)

    changed_ref = correspondence("frontier_correspondence_3", 2)
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        producer,
        {"assertions": [], "typed_context_refs": [changed_ref]},
        authorized_evidence_refs=[],
    )
    assert frontier(merge.abox) != first_frontier


def test_exact_source_scene_pair_is_not_reopened_by_duplicate_cad_artifact(
    tmp_path: Path,
) -> None:
    producer = "rgb_d_cad_grounding"
    source_sha256 = "a" * 64
    cad_record = {
        "schema_version": 1,
        "record_type": "CADMeshRecord",
        "producer": producer,
        "source": {
            "context_ref": "Gear_Medium.STL",
            "source_sha256": source_sha256,
        },
    }
    cad_1_ref = _write_revision_record(tmp_path, "pair_cad_1", cad_record)
    cad_2_ref = _write_revision_record(
        tmp_path,
        "pair_cad_2",
        {**cad_record, "operation_number": 2},
    )
    changed_cad_ref = _write_revision_record(
        tmp_path,
        "pair_cad_changed",
        {
            **cad_record,
            "source": {
                "context_ref": "Gear_Medium.STL",
                "source_sha256": "b" * 64,
            },
        },
    )
    segmentation_1_ref = _write_revision_record(
        tmp_path,
        "pair_segmentation_1",
        {
            "schema_version": 1,
            "record_type": "RGBDSegmentationRecord",
            "producer": producer,
            "cameras": [{"timestamp_ns": 100}],
        },
    )
    correspondence_ref = _write_revision_record(
        tmp_path,
        "pair_correspondence",
        {
            "schema_version": 1,
            "record_type": "CADSizeCorrespondenceRecord",
            "producer": producer,
            "CAD_correspondence": "rejected",
            "CAD": {
                "context_ref": "Gear_Medium.STL",
                "source_sha256": source_sha256,
                "record": _hashed_record_ref(tmp_path, cad_1_ref),
            },
            "segmentation": {
                "record": _hashed_record_ref(tmp_path, segmentation_1_ref)
            },
        },
    )
    tbox = ontology_config().load_tbox()
    initialize_interaction_abox(tmp_path, "assemble medium gear", tbox)
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        producer,
        {
            "assertions": [],
            "typed_context_refs": [
                cad_1_ref,
                cad_2_ref,
                changed_cad_ref,
                segmentation_1_ref,
                correspondence_ref,
            ],
        },
        authorized_evidence_refs=[],
    )
    descriptors = _runtime().grounding_producer_descriptors()

    def actions_for(
        abox: ABoxSnapshot,
        *,
        preview_source_sha256: str,
    ) -> tuple[production_grounding._EligibleProviderAction, ...]:
        view = build_product_context_view(
            tmp_path,
            abox,
            attempted_evidence=(),
            assessed_at_ns=200,
        )
        cad = next(
            binding
            for binding in view.typed_bindings
            if binding.record_ref == cad_2_ref
        )
        segmentation = max(
            (
                binding
                for binding in view.typed_bindings
                if binding.record_type == "RGBDSegmentationRecord"
            ),
            key=lambda binding: int(binding.observed_at_ns or -1),
        )
        previous = GroundingSession.create(
            revision=2,
            requirement_text=abox.product_requirement,
            next_action=GroundingNextAction.from_mapping(
                {"action": "retrieve", "source_ref": "Gear_Medium.STL"}
            ),
            selected_provider_id=producer,
            selected_source_revision=(
                production_grounding._cad_segmentation_revision(
                    tmp_path,
                    cad,
                    segmentation,
                )
            ),
            status="waiting_for_evidence",
        )
        return production_grounding._discover_provider_actions(
            tmp_path,
            descriptors,
            previews=[
                {
                    "provider_id": producer,
                    "evidence_type": "CAD",
                    "source_ref": "Gear_Medium.STL",
                    "source_revision": preview_source_sha256,
                    "availability": True,
                    "produced_record_types": ["CADMeshRecord"],
                }
            ],
            view=view,
            previous=previous,
            pending_attempt=None,
            required_record_type="RobotFramePoseRecord",
            required_target_frame="world",
        )

    same_pair_actions = actions_for(
        merge.abox,
        preview_source_sha256=source_sha256,
    )
    assert not any(
        action.produced_record_types == ("CADSizeCorrespondenceRecord",)
        for action in same_pair_actions
    )
    changed_cad_actions = actions_for(
        merge.abox,
        preview_source_sha256="b" * 64,
    )
    assert any(
        action.source_ref == "Gear_Medium.STL"
        and action.produced_record_types == ("CADSizeCorrespondenceRecord",)
        for action in changed_cad_actions
    )

    segmentation_2_ref = _write_revision_record(
        tmp_path,
        "pair_segmentation_2",
        {
            "schema_version": 1,
            "record_type": "RGBDSegmentationRecord",
            "producer": producer,
            "cameras": [{"timestamp_ns": 200}],
        },
    )
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        producer,
        {"assertions": [], "typed_context_refs": [segmentation_2_ref]},
        authorized_evidence_refs=[],
    )
    changed_scene_actions = actions_for(
        merge.abox,
        preview_source_sha256=source_sha256,
    )
    assert any(
        action.source_ref == "Gear_Medium.STL"
        and action.produced_record_types == ("CADSizeCorrespondenceRecord",)
        for action in changed_scene_actions
    )


def test_non_pose_resource_selection_error_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tbox, abox, _view = _prepare_resource_grounding_context(
        tmp_path,
        candidate_diameters_m=(0.042,),
    )
    pose_ref = _write_synthetic_world_pose(tmp_path, "rgb_d_cad_grounding")
    pose_path = tmp_path / pose_ref
    pose_record = _read_json(pose_path)
    pose_record["observation_timestamp_ns"] = 3_000_004_000
    pose_path.write_text(json.dumps(pose_record) + "\n", encoding="utf-8")
    merge = validate_and_merge_triple_delta(
        tmp_path,
        tbox,
        "rgb_d_cad_grounding",
        {"assertions": [], "typed_context_refs": [pose_ref]},
        authorized_evidence_refs=[],
    )
    view = build_product_context_view(
        tmp_path,
        merge.abox,
        attempted_evidence=("Gear_Medium.STL", "observation_0001"),
        assessed_at_ns=3_000_004_000,
    )

    def fail_selection(**_kwargs: object) -> None:
        raise production_grounding.ResourceGroundingError(
            "controlled manifest failure"
        )

    monkeypatch.setattr(
        production_grounding,
        "select_predefined_resource",
        fail_selection,
    )

    with pytest.raises(ProductionGroundingError, match="controlled manifest failure"):
        asyncio.run(
            _runtime().assess_product_context(
                SimpleProductAgent([]),
                interaction_root=tmp_path,
                tbox=tbox,
                abox=abox,
                abox_view=view.to_record(),
                attempted_evidence=("Gear_Medium.STL", "observation_0001"),
                turn_number=2,
                max_pa_turns=12,
            )
        )


def test_emergency_pa_ceiling_persists_incomplete_without_model_call(
    tmp_path: Path,
) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "inspect current geometry", tbox)
    view = build_product_context_view(
        tmp_path,
        abox,
        attempted_evidence=(),
        assessed_at_ns=1,
    )
    agent = SimpleProductAgent([])

    result = asyncio.run(
        _runtime().initial_product_context_decision(
            agent,
            interaction_root=tmp_path,
            tbox=tbox,
            abox=abox,
            product_context=view.to_record(),
            max_pa_turns=1,
        )
    )

    assert result["grounding_status"] == "incomplete"
    assert agent.calls == []
    session = load_latest_grounding_session(tmp_path)
    assert session is not None and session.status == "incomplete"


def test_completion_v2_pins_new_context_contract_and_session_hash(
    tmp_path: Path,
) -> None:
    agent = SimpleProductAgent(
        [
            {"action": "propose_grounding"},
            {"action": "retrieve", "source_ref": "Gear_Medium.STL"},
        ]
    )
    runtime = _runtime(synthetic_world_pose=True)

    first = asyncio.run(
        start_pa_context_interaction(
            agent,
            tmp_path,
            "assemble medium gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )
    assert first["needed_context"]["context_ref"] == "Gear_Medium.STL"
    context_serving.serve_pa_requested_context(tmp_path)
    result = asyncio.run(
        continue_pa_context_interaction(
            agent,
            tmp_path,
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )

    assert result.get("failure") is None, result
    assert result["grounding_status"] == "complete"
    completion = load_pa_context_grounding_completion(tmp_path)
    assert isinstance(completion, PAContextGroundingCompletionV2)
    contract = TypedGroundingContract.from_mapping(
        _read_json(tmp_path / completion.typed_grounding_contract_ref)
    )
    assert contract.context_summary
    assert contract.context_evidence_refs == ("requirement_0001",)
    assert not hasattr(contract, "statements")
    session_path = tmp_path / completion.grounding_session_ref
    session_path.write_bytes(session_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="GroundingSession hash"):
        load_pa_context_grounding_completion(tmp_path)


def test_document_free_fill_never_calls_document_vlm(tmp_path: Path) -> None:
    vision = ControlledVisionRuntime()
    agent = SimpleProductAgent(
        [
            {"action": "propose_grounding"},
            {"action": "retrieve", "source_ref": "Gear_Medium.STL"},
        ]
    )

    result = asyncio.run(
        start_pa_context_interaction(
            agent,
            tmp_path,
            "inspect current geometry",
            ontology_config=ontology_config(),
            grounding_runtime=_runtime(vision),
        )
    )

    assert result["needed_context"]["context_ref"] == "Gear_Medium.STL"
    assert vision.requests == []
    assert vision.targeted_requests == []

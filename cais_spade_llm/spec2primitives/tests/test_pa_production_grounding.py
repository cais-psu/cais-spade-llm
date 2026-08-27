"""Tests for the production pre-RA PA grounding runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rdflib import Namespace

from cais_spade_llm.spec2primitives import spec2primitives_ui
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
    GroundingContractError,
    build_product_context_view,
    load_pa_context_grounding_completion,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    initialize_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.agents.pa.production_grounding import (
    ProductionProductContextGroundingRuntime,
)
from cais_spade_llm.spec2primitives.config import load_model_runtime_config
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    PPR_NAMESPACE,
    ontology_config,
)
from cais_spade_llm.spec2primitives.tests.test_document_interpretation import (
    ControlledVisionRuntime,
)
from cais_spade_llm.spec2primitives.tests.test_rgbd_cad_preprocessing import (
    _observation_bundle,
)
from cais_spade_llm.spec2primitives.tools.observation_context import (
    write_observation_bundle,
)

PPR = Namespace(PPR_NAMESPACE)


class DraftProductAgent:
    """Author controlled drafts using the exact runtime-supplied version and view."""

    def __init__(
        self,
        needs: list[dict[str, object]],
        *,
        selected_cad: str = "Gear_Medium.STL",
    ) -> None:
        self.needs = needs
        self.selected_cad = selected_cad
        self.calls: list[dict[str, object]] = []

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append({"prompt": prompt, "response_format": response_format})
        name = response_format["name"]
        if name == "spec2primitives_grounding_evidence_selection":
            return {"context_ref": self.selected_cad}
        if name != "spec2primitives_task_transition_draft":
            raise AssertionError(f"Unexpected structured request: {name}")
        properties = response_format["schema"]["properties"]["task_transition_draft"][
            "properties"
        ]
        return {
            "task_transition_draft": {
                "version": properties["version"]["const"],
                "product_requirement": properties["product_requirement"]["const"],
                "requested_process": str(PPR.process),
                "required_outcome": "Medium Gear assembled",
                "required_inputs": self.needs,
                "unresolved_user_intent": None,
                "source_view_fingerprint": properties["source_view_fingerprint"][
                    "const"
                ],
            }
        }


def _need(kind: str, symbol: str) -> dict[str, object]:
    return {
        "kind": kind,
        "symbol": symbol,
        "subject_role": "requested_product",
        "authority": "PA",
        "frame": None,
        "maximum_age_ns": None,
        "reason": f"the task draft requires {symbol}",
    }


def _runtime() -> ProductionProductContextGroundingRuntime:
    return ProductionProductContextGroundingRuntime(
        tbox=ontology_config().load_tbox(),
        document_config=load_model_runtime_config().document_vlm,
        document_vision_runtime=ControlledVisionRuntime(),
    )


def test_openai_response_schema_constraints_have_explicit_types(
    tmp_path: Path,
) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble Medium Gear", tbox)
    view = build_product_context_view(
        tmp_path,
        abox,
        attempted_evidence=(),
        assessed_at_ns=1_000,
    )
    task_format = production_grounding._task_draft_response_format(
        view,
        version=1,
        classes=sorted(tbox.classes),
        properties=sorted(tbox.object_properties | tbox.datatype_properties),
    )
    selection_format = production_grounding._evidence_selection_response_format(
        ["Gear_Medium.STL"]
    )

    def assert_constrained_nodes_are_typed(value: object) -> None:
        if isinstance(value, dict):
            if "const" in value or "enum" in value:
                assert "type" in value
            for nested in value.values():
                assert_constrained_nodes_are_typed(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_constrained_nodes_are_typed(nested)

    assert_constrained_nodes_are_typed(task_format["schema"])
    assert_constrained_nodes_are_typed(selection_format["schema"])
    properties = task_format["schema"]["properties"]["task_transition_draft"][
        "properties"
    ]
    assert properties["version"]["type"] == "integer"
    assert properties["product_requirement"]["type"] == "string"
    assert properties["requested_process"]["type"] == ["string", "null"]
    assert properties["source_view_fingerprint"]["type"] == "string"
    assert selection_format["schema"]["properties"]["context_ref"]["type"] == (
        "string"
    )


def test_document_grounding_runs_in_main_pa_loop_and_completes(tmp_path: Path) -> None:
    agent = DraftProductAgent([_need("class", str(PPR.feature))])
    runtime = _runtime()

    first = asyncio.run(
        start_pa_context_interaction(
            agent,
            tmp_path,
            "assemble Medium Gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )
    assert first["needed_context"]["context_ref"] == (
        "NIST_assembly_instructions.pdf"
    )
    served = context_serving.serve_pa_requested_context(tmp_path)
    assert served["served_context"]["evidence_type"] == "document"

    result = asyncio.run(
        continue_pa_context_interaction(
            agent,
            tmp_path,
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
            max_pa_turns=4,
        )
    )

    assert result == {
        "needed_context": None,
        "context understanding complete": True,
    }
    assert (tmp_path / "products/grounding/product_context/view_0001.json").is_file()
    assert (tmp_path / "products/grounding/task_transition/draft_0002.json").is_file()
    assert not list(tmp_path.glob("resources/**/*"))
    ui_view = spec2primitives_ui._pa_ui_view(
        {
            "interaction_identifier": "interaction_controlled",
            "interaction_root": tmp_path,
            "product_requirement": "assemble Medium Gear",
            "phase_3_1": first,
            "phase_3_2": served,
            "phase_3_3": result,
            "max_pa_turns": 4,
        }
    )
    assert str(PPR.feature) in ui_view["Ontology Grounding"]
    assert "DocumentInterpretationRecord" in ui_view["Typed Runtime Context"]
    assert "TaskTransitionDraft" in ui_view["Grounding Decisions"]
    assert "producer_selection_0001" in ui_view["interaction_record"]
    assert load_pa_context_grounding_completion(tmp_path).typed_context_refs
    assert "Ready for Phase 5" in ui_view["PA Grounding Completion"]


def test_exact_cad_is_selected_and_no_unused_geometry_stage_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = DraftProductAgent(
        [_need("typed_context_record", "CADMeshRecord")],
        selected_cad="Gear_Medium.STL",
    )
    runtime = _runtime()

    monkeypatch.setattr(
        production_grounding,
        "associate_segmented_candidate_by_size",
        lambda **_: (_ for _ in ()).throw(AssertionError("unexpected correspondence")),
    )
    monkeypatch.setattr(
        production_grounding,
        "estimate_camera_frame_pose",
        lambda **_: (_ for _ in ()).throw(AssertionError("unexpected pose")),
    )
    first = asyncio.run(
        start_pa_context_interaction(
            agent,
            tmp_path,
            "assemble Medium Gear",
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

    assert result["context understanding complete"] is True
    view = _read_json(tmp_path / "products/grounding/product_context/view_0001.json")
    assert [item["record_type"] for item in view["typed_bindings"]] == [
        "CADMeshRecord"
    ]
    assert not list(
        (tmp_path / "products/grounding/rgb_d_cad_grounding").glob(
            "correspondence_*"
        )
    )
    completion = load_pa_context_grounding_completion(tmp_path)
    assert [item["ref"] for item in completion.typed_context_refs] == [
        view["typed_bindings"][0]["record_ref"]
    ]
    typed_path = tmp_path / str(completion.typed_context_refs[0]["ref"])
    typed_path.write_bytes(typed_path.read_bytes() + b"\n")
    with pytest.raises(GroundingContractError, match="typed context hash"):
        load_pa_context_grounding_completion(tmp_path)


def test_observation_need_runs_preprocessing_and_segmentation_but_not_pose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = DraftProductAgent(
        [_need("typed_context_record", "RGBDSegmentationRecord")]
    )
    runtime = _runtime()

    def capture(
        observations_root: Path,
        observation_ref: str,
        *,
        timeout_sec: float,
    ) -> Path:
        assert timeout_sec == 5.0
        assert observation_ref == "observation_0001"
        return write_observation_bundle(
            observations_root,
            _observation_bundle(),
        )

    monkeypatch.setattr(context_serving, "capture_gazebo_observation", capture)
    monkeypatch.setattr(
        production_grounding,
        "estimate_camera_frame_pose",
        lambda **_: (_ for _ in ()).throw(AssertionError("unexpected pose")),
    )
    first = asyncio.run(
        start_pa_context_interaction(
            agent,
            tmp_path,
            "assemble Medium Gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )
    assert first["needed_context"]["request_live_observation"] is True
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

    assert result["context understanding complete"] is True
    view = _read_json(tmp_path / "products/grounding/product_context/view_0001.json")
    assert [item["record_type"] for item in view["typed_bindings"]] == [
        "ColoredPointCloudSetRecord",
        "RGBDSegmentationRecord",
    ]


def test_existing_records_drive_correspondence_and_camera_pose_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble Medium Gear", tbox)
    abox = _merge_minimal_binding(
        tmp_path,
        tbox,
        abox,
        record_type="CADMeshRecord",
        record_name="cad.json",
        evidence_refs=["Gear_Medium.STL"],
    )
    abox = _merge_minimal_binding(
        tmp_path,
        tbox,
        abox,
        record_type="RGBDSegmentationRecord",
        record_name="segmentation.json",
        evidence_refs=["observation_0001"],
    )
    view = build_product_context_view(
        tmp_path,
        abox,
        attempted_evidence=("Gear_Medium.STL", "observation_0001"),
        assessed_at_ns=1_000,
    )

    def correspondence(**kwargs: object) -> object:
        path = Path(str(kwargs["interaction_root"])) / (
            "products/grounding/rgb_d_cad_grounding/correspondence_0001/"
            "correspondence_record.json"
        )
        _write_minimal_record(
            path,
            "CADSizeCorrespondenceRecord",
            {"CAD_correspondence": "accepted"},
        )
        return SimpleNamespace(
            CAD_correspondence="accepted",
            record_path=path,
        )

    def pose(**kwargs: object) -> object:
        path = Path(str(kwargs["interaction_root"])) / (
            "products/grounding/rgb_d_cad_grounding/pose_0001/pose_record.json"
        )
        _write_minimal_record(
            path,
            "CADPoseEstimationRecord",
            {
                "pose": "accepted",
                "coordinate_frame": "cam_mk4_2_optical_frame",
            },
        )
        return SimpleNamespace(pose="accepted", record_path=path)

    monkeypatch.setattr(
        production_grounding,
        "associate_segmented_candidate_by_size",
        correspondence,
    )
    monkeypatch.setattr(production_grounding, "estimate_camera_frame_pose", pose)
    agent = DraftProductAgent(
        [
            _need("typed_context_record", "CADMeshRecord"),
            _need("typed_context_record", "RGBDSegmentationRecord"),
            _need("typed_context_record", "CADSizeCorrespondenceRecord"),
            _need("typed_context_record", "CADPoseEstimationRecord"),
        ]
    )

    result = asyncio.run(
        runtime.assess_product_context(
            agent,
            interaction_root=tmp_path,
            tbox=tbox,
            abox=abox,
            abox_view=view.to_record(),
            attempted_evidence=("Gear_Medium.STL", "observation_0001"),
            turn_number=3,
            max_pa_turns=12,
        )
    )

    assert result["context understanding complete"] is True
    final_view = _read_json(
        tmp_path / "products/grounding/product_context/view_0001.json"
    )
    assert final_view["typed_bindings"][-1]["record_type"] == (
        "CADPoseEstimationRecord"
    )
    assert final_view["typed_bindings"][-1]["frame"] == (
        "cam_mk4_2_optical_frame"
    )
    assert not list(
        (tmp_path / "products/grounding/rgb_d_cad_grounding").glob("robot_pose_*")
    )


def test_ambiguous_pose_stops_as_system_grounding_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble Medium Gear", tbox)
    abox = _merge_minimal_binding(
        tmp_path,
        tbox,
        abox,
        record_type="CADSizeCorrespondenceRecord",
        record_name="correspondence.json",
        evidence_refs=["Gear_Medium.STL", "observation_0001"],
    )
    view = build_product_context_view(
        tmp_path,
        abox,
        attempted_evidence=("Gear_Medium.STL", "observation_0001"),
        assessed_at_ns=1_000,
    )

    def ambiguous_pose(**kwargs: object) -> object:
        path = Path(str(kwargs["interaction_root"])) / (
            "products/grounding/rgb_d_cad_grounding/pose_0001/pose_record.json"
        )
        _write_minimal_record(path, "CADPoseEstimationRecord", {"pose": "ambiguous"})
        return SimpleNamespace(pose="ambiguous", record_path=path)

    monkeypatch.setattr(
        production_grounding,
        "estimate_camera_frame_pose",
        ambiguous_pose,
    )
    agent = DraftProductAgent(
        [_need("typed_context_record", "CADPoseEstimationRecord")]
    )

    with pytest.raises(GroundingContractError, match="No untried"):
        asyncio.run(
            runtime.assess_product_context(
                agent,
                interaction_root=tmp_path,
                tbox=tbox,
                abox=abox,
                abox_view=view.to_record(),
                attempted_evidence=("Gear_Medium.STL", "observation_0001"),
                turn_number=3,
                max_pa_turns=12,
            )
        )
    selection = _read_json(tmp_path / "interaction_record/producer_selection_0001.json")
    assert selection["request"] == {
        "existing_record": ["CADSizeCorrespondenceRecord"]
    }
    assert "clarification_question" not in selection["request"]


def _merge_minimal_binding(
    root: Path,
    tbox: object,
    abox: object,
    *,
    record_type: str,
    record_name: str,
    evidence_refs: list[str],
) -> Any:
    record_path = root / f"products/grounding/test_records/{record_name}"
    extra: dict[str, object] = {}
    if record_type == "CADSizeCorrespondenceRecord":
        extra["CAD_correspondence"] = "accepted"
    _write_minimal_record(record_path, record_type, extra)
    merge = validate_and_merge_triple_delta(
        root,
        tbox,  # type: ignore[arg-type]
        "rgb_d_cad_grounding",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [str(record_path.relative_to(root))],
        },
        authorized_evidence_refs=evidence_refs,
    )
    return merge.abox


def _write_minimal_record(
    path: Path,
    record_type: str,
    extra: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": record_type,
                "producer": "rgb_d_cad_grounding",
                "evidence_refs": ["Gear_Medium.STL", "observation_0001"],
                **extra,
            }
        ),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

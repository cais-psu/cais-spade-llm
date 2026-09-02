"""Focused tests for the native ProductAgent UI connection."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.spec2primitives import spec2primitives_ui
from cais_spade_llm.spec2primitives.agents.pa import product_agent_runtime
from cais_spade_llm.spec2primitives.tests.test_pa_completion import (
    persist_native_completion_fixture,
)


def test_connected_ui_starts_one_native_grounding_call(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, object]] = []
    output = {
        "grounding_status": "incomplete",
        "insufficient_evidence": "No approved observation is available.",
        "tool_call_refs": [],
    }

    async def _start(
        product_agent: object,
        interaction_root: Path,
        product_requirement: str,
        *,
        ontology_config: object,
        grounding_runtime: object,
    ) -> dict[str, object]:
        calls.append(
            {
                "product_agent": product_agent,
                "interaction_root": interaction_root,
                "product_requirement": product_requirement,
                "ontology_config": ontology_config,
                "grounding_runtime": grounding_runtime,
            }
        )
        return output

    monkeypatch.setattr(spec2primitives_ui, "start_pa_context_interaction", _start)
    runtime = SimpleNamespace(
        contexts_root=tmp_path,
        product_agent=object(),
        ontology_config=object(),
        grounding_runtime=object(),
    )

    interaction = asyncio.run(
        spec2primitives_ui._run_pa_ui_interaction(
            runtime,
            "assemble medium gear",
        )
    )

    assert len(calls) == 1
    assert interaction["phase_3_1"] == output
    assert interaction["phase_3_2"] is None
    assert interaction["phase_3_3"] is None
    assert Path(interaction["interaction_root"]).parent == tmp_path


def test_ui_observer_forwards_native_tools_without_changing_them() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "retrieve",
            "parameters": {"type": "object"},
        },
    }
    tool_results: list[tuple[str, Mapping[str, object]]] = []
    stages: list[str] = []

    async def _retrieve(
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        tool_results.append((tool_name, arguments))
        return {"record_type": "CADMeshRecord"}

    class _Agent:
        async def ask_llm_structured(
            self,
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            tool_executor: Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]]
            | None = None,
            max_tool_rounds: int = 3,
        ) -> dict[str, Any]:
            del prompt, response_format
            assert tools == [tool]
            assert tool_executor is _retrieve
            assert max_tool_rounds == 7
            await tool_executor("retrieve", {"evidence_id": "evidence_0001"})
            return {"insufficient_evidence": "More evidence is required."}

    observer = spec2primitives_ui._PAUIRuntimeObserver(
        _Agent(),
        on_pa_stage=stages.append,
    )
    result = asyncio.run(
        observer.ask_llm_structured(
            "ground context",
            response_format={"name": "spec2primitives_grounding_result"},
            tools=[tool],
            tool_executor=_retrieve,
            max_tool_rounds=7,
        )
    )

    assert result == {"insufficient_evidence": "More evidence is required."}
    assert tool_results == [("retrieve", {"evidence_id": "evidence_0001"})]
    assert stages == ["Investigating approved evidence and authoring the target feature."]


def test_shared_product_agent_bridge_wraps_async_retrieve_callback(
    monkeypatch: Any,
) -> None:
    scheduled: list[tuple[object, object]] = []

    class _Future:
        def result(self, *, timeout: float) -> Mapping[str, object]:
            assert timeout == 120.0
            return {
                "tool_name": "retrieve",
                "evidence_id": "evidence_0001",
            }

    def _schedule(coroutine: object, event_loop: object) -> _Future:
        scheduled.append((coroutine, event_loop))
        close = getattr(coroutine, "close", None)
        if callable(close):
            close()
        return _Future()

    monkeypatch.setattr(
        product_agent_runtime.asyncio,
        "run_coroutine_threadsafe",
        _schedule,
    )

    class _SharedAgent:
        async def ask_llm_structured(
            self,
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None,
            tool_executor: Callable[[str, dict[str, Any]], Mapping[str, object]],
            max_tool_rounds: int,
        ) -> dict[str, Any]:
            del prompt, response_format, tools, max_tool_rounds
            result = tool_executor(
                "retrieve",
                {"evidence_id": "evidence_0001"},
            )
            return dict(result)

    async def _retrieve(
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        return {"tool_name": tool_name, **arguments}

    runtime = object.__new__(product_agent_runtime._SharedProductAgentContextRuntime)
    runtime._product_agent = _SharedAgent()
    result = asyncio.run(
        runtime.ask_llm_structured(
            "ground context",
            response_format={"name": "spec2primitives_grounding_result"},
            tools=[{"type": "function"}],
            tool_executor=_retrieve,
            max_tool_rounds=4,
        )
    )

    assert result == {
        "tool_name": "retrieve",
        "evidence_id": "evidence_0001",
    }
    assert len(scheduled) == 1


def test_shared_product_agent_bridge_cancels_timed_out_retrieve_callback(
    monkeypatch: Any,
) -> None:
    cancelled: list[bool] = []

    class _Future:
        def result(self, *, timeout: float) -> Mapping[str, object]:
            assert timeout == 120.0
            raise product_agent_runtime.FutureTimeoutError

        def cancel(self) -> bool:
            cancelled.append(True)
            return True

    def _schedule(coroutine: object, event_loop: object) -> _Future:
        del event_loop
        close = getattr(coroutine, "close", None)
        if callable(close):
            close()
        return _Future()

    monkeypatch.setattr(
        product_agent_runtime.asyncio,
        "run_coroutine_threadsafe",
        _schedule,
    )

    class _SharedAgent:
        async def ask_llm_structured(
            self,
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None,
            tool_executor: Callable[[str, dict[str, Any]], Mapping[str, object]],
            max_tool_rounds: int,
        ) -> dict[str, Any]:
            del prompt, response_format, tools, max_tool_rounds
            return dict(
                tool_executor(
                    "retrieve",
                    {"evidence_id": "evidence_0001"},
                )
            )

    async def _retrieve(
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        return {"tool_name": tool_name, **arguments}

    runtime = object.__new__(product_agent_runtime._SharedProductAgentContextRuntime)
    runtime._product_agent = _SharedAgent()
    with pytest.raises(RuntimeError) as error:
        asyncio.run(
            runtime.ask_llm_structured(
                "ground context",
                response_format={"name": "spec2primitives_grounding_result"},
                tools=[{"type": "function"}],
                tool_executor=_retrieve,
                max_tool_rounds=4,
            )
        )

    assert str(error.value) == (
        "Controlled evidence retrieval and processing timed out after 120 seconds."
    )
    assert cancelled == [True]


def test_shared_product_agent_bridge_applies_configured_reasoning_effort(
    monkeypatch: Any,
) -> None:
    created: list[Any] = []

    class _SharedAgent:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs
            self.reasoning_effort = "medium"
            created.append(self)

    monkeypatch.setattr(product_agent_runtime, "ProductAgent", _SharedAgent)

    runtime = product_agent_runtime.create_product_agent_context_runtime(
        model="gpt-5.4",
        reasoning_effort="none",
    )

    assert runtime is not None
    assert len(created) == 1
    assert created[0].reasoning_effort == "none"
    assert created[0].kwargs["model"] == "gpt-5.4"


def test_completed_view_has_five_simple_stages_and_location(
    tmp_path: Path,
) -> None:
    interaction_root = tmp_path / "interaction_native"
    persist_native_completion_fixture(interaction_root)
    turn = _read_json(interaction_root / "interaction_record/turn_0001.json")
    output = turn["PA_output"]
    interaction = {
        "interaction_identifier": interaction_root.name,
        "interaction_root": interaction_root,
        "product_requirement": "assemble medium gear",
        "phase_3_1": output,
        "phase_3_2": None,
        "phase_3_3": output,
        "max_pa_turns": 12,
    }

    view = spec2primitives_ui._pa_ui_view(interaction)

    assert view["activity_state"] == "grounding complete"
    assert [event["title"] for event in view["timeline"]] == [
        "Requirement received",
        "Evidence investigated",
        "Target feature grounded",
        "Endpoint-motion allocation validated",
        "Grounding complete",
    ]
    final_result = view["final_result"]
    assert isinstance(final_result, dict)
    assert final_result["location"] == "available"
    assert final_result["target_frame"] == "world"
    assert final_result["pose"] is None
    assert final_result["selected_resource"] == "xarm6"
    assert final_result["allocation_label"] == "validated endpoint-motion allocation"
    assert final_result["motion_executed"] is False
    assert final_result["limitations"] == []
    assert final_result["reachability"]["status"] == "accepted"
    assert final_result["reachability"]["resource_symbol"] == "xarm6"
    assert final_result["reachability"]["process_symbol"] == "assembly"
    assert final_result["reachability"]["target_frame"] == "world"
    assert final_result["robot_agent_validation"]["status"] == "accepted"
    assert final_result["robot_agent_validation"]["mode"] == "plan_only"
    assert final_result["robot_agent_validation"]["validation_scope"] == (
        "endpoint_motion"
    )
    assert final_result["robot_agent_validation"]["motion_executed"] is False
    assert final_result["target_feature"]["desired_state"]["statement"]["text"] == (
        "The medium gear is assembled as requested."
    )
    assert final_result["target_feature"]["current_state"]["state_values"] == []
    assert final_result["target_feature"]["desired_state"]["state_values"] == []
    current = final_result["current_state_evidence"]
    desired = final_result["desired_state_evidence"]
    assert current["state_iri"].endswith("currentstate_0001")
    assert desired["state_iri"].endswith("desiredstate_0001")
    assert current["evidence_handle"].startswith("state_evidence_")
    assert desired["evidence_handle"].startswith("state_evidence_")
    assert current["evidence_handle"] != desired["evidence_handle"]
    assert current["translated_location_m"] == [0.0, -0.7, 1.1]
    assert desired["translated_location_m"] == [0.0, -0.2, 1.1]
    assert current["annotated_rgb"]["data_uri"].startswith("data:image/svg+xml;base64,")
    assert desired["annotated_rgb"]["data_uri"].startswith("data:image/svg+xml;base64,")
    ontology_rows = final_result["ontology"]["rows"]
    assert any(
        row["subject"] == "ctx:feature_0001"
        and row["predicate"] == "ppr:hascurrentstate"
        and row["object"] == "ctx:currentstate_0001"
        for row in ontology_rows
    )
    assert any(
        row["subject"] == "ctx:feature_0001"
        and row["predicate"] == "ppr:hasdesiredstate"
        and row["object"] == "ctx:desiredstate_0001"
        for row in ontology_rows
    )
    phase_5_1 = view["phase_5_1"]
    assert isinstance(phase_5_1, dict)
    assert phase_5_1["status"] == "ready_for_assignment"
    assert phase_5_1["product_requirement"] == "assemble medium gear"
    assert phase_5_1["selected_resource_jid"] == "xarm6@localhost"
    assert phase_5_1["selected_execution_mode"] == "simulation"
    assert phase_5_1["assignment_ref"] is None
    assert phase_5_1["robot_state"] is None
    assert phase_5_1["primitive_catalog"] == []
    phase_5_2 = view["phase_5_2"]
    assert isinstance(phase_5_2, dict)
    assert phase_5_2["status"] == "waiting_for_context"
    assert phase_5_2["draft"] is None
    serialized = json.dumps(view)
    for removed_wording in (
        "Target grounded",
        "Context understanding complete",
        "next_action",
        "Focused inspection",
        "ready to execute",
    ):
        assert removed_wording not in serialized


def test_completed_limitations_hide_document_uncertainty() -> None:
    document_notices = [
        (
            'The purchasing section refers to design files available from "LINK" '
            "and an example file name, but the actual download link is not provided "
            "in the extracted text."
        ),
        (
            "One part of the text mentions testing procedures described in a "
            'separate document marked "TBD," so the full testing procedure is not '
            "included here."
        ),
        (
            "Some part numbers and vendor details appear split across lines in the "
            "extracted text, making a few entries hard to read cleanly."
        ),
        (
            'The document uses both "course" and "coarse" in the context of thread '
            "descriptions; the exact intended spelling is unclear from the page "
            "images and text alone."
        ),
    ]
    product_context = {
        "typed_bindings": [],
        "uncertainty": [
            {"description": notice, "evidence_refs": ["document.pdf#page=1"]}
            for notice in document_notices
        ],
    }
    destination_gap = "The available evidence does not identify the destination shaft."

    limitations = spec2primitives_ui._final_result_limitations(
        product_context,
        {"missing_information": [destination_gap]},
    )

    assert limitations == [destination_gap]
    assert [item["description"] for item in product_context["uncertainty"]] == document_notices


def test_completed_limitations_hide_superseded_typed_record_claim() -> None:
    stale_location_gap = (
        "A RobotFrameLocationRecord in target_frame 'world' for the medium gear "
        "is not derivable from the retrieved evidence."
    )
    destination_gap = "The available evidence does not identify the destination shaft."
    product_context = {
        "typed_bindings": [
            {
                "output_symbol": "RobotFrameLocationRecord",
                "record_type": "RobotFrameLocationRecord",
                "status": "accepted",
            }
        ],
        "uncertainty": [],
    }

    limitations = spec2primitives_ui._final_result_limitations(
        product_context,
        {"missing_information": [stale_location_gap, destination_gap]},
    )

    assert limitations == [destination_gap]
    assert (
        spec2primitives_ui._final_result_limitations(
            product_context,
            {"missing_information": [stale_location_gap]},
        )
        == []
    )


def test_completed_view_recovers_from_persisted_native_records(
    tmp_path: Path,
) -> None:
    interaction_root = tmp_path / "interaction_native"
    persist_native_completion_fixture(interaction_root)

    recovered = spec2primitives_ui._latest_pa_ui_interaction(tmp_path)

    assert recovered is not None
    assert recovered["recovered"] is True
    assert recovered["interaction_identifier"] == "interaction_native"
    assert spec2primitives_ui._pa_ui_view(recovered)["activity_state"] == ("grounding complete")


def test_timeline_distinguishes_clarification_from_incomplete_grounding() -> None:
    clarification = spec2primitives_ui._persisted_pa_timeline(
        "assemble medium gear",
        turns=[
            {
                "PA_output": {
                    "grounding_status": "clarification_required",
                    "clarification_question": "Which product variant is intended?",
                }
            }
        ],
        retrievals=[],
        clarifications=[],
        records={},
        final_result=None,
        terminal_failure=None,
    )
    incomplete = spec2primitives_ui._persisted_pa_timeline(
        "assemble medium gear",
        turns=[
            {
                "PA_output": {
                    "grounding_status": "incomplete",
                    "insufficient_evidence": "No approved evidence resolved the need.",
                }
            }
        ],
        retrievals=[],
        clarifications=[],
        records={},
        final_result=None,
        terminal_failure=None,
    )

    assert clarification[-1]["title"] == "Clarification requested"
    assert incomplete[-1]["title"] == "Grounding incomplete"


def test_cad_identity_reports_all_cited_candidates_as_ambiguous(
    tmp_path: Path,
) -> None:
    record_refs = [
        "products/grounding/cad/candidate_a.json",
        "products/grounding/cad/candidate_b.json",
    ]
    context_refs = ["approved-cad-a", "approved-cad-b"]
    typed_bindings: list[dict[str, object]] = []
    for index, (record_ref, context_ref) in enumerate(
        zip(record_refs, context_refs, strict=True),
        start=1,
    ):
        record_path = tmp_path / record_ref
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "source": {"context_ref": context_ref},
                    "coordinate_frame": "cad_local",
                    "bounds_m": {"x": [0.0, float(index)]},
                }
            ),
            encoding="utf-8",
        )
        typed_bindings.append(
            {
                "output_symbol": "CADMeshRecord",
                "status": "accepted",
                "record_ref": record_ref,
                "evidence_refs": [context_ref],
            }
        )

    identity = spec2primitives_ui._cad_identity(
        tmp_path,
        typed_bindings=typed_bindings,
        target_feature={"evidence_refs": context_refs},
    )

    assert identity is not None
    assert identity["status"] == "ambiguous"
    assert [
        candidate["context_ref"] for candidate in identity["candidates"]
    ] == context_refs


def test_calibration_readiness_uses_actionable_runtime_state() -> None:
    ready = SimpleNamespace(
        camera_to_world_calibration_runtime=object(),
        camera_to_world_calibration_unavailable_reason=None,
    )
    unavailable = SimpleNamespace(
        camera_to_world_calibration_runtime=None,
        camera_to_world_calibration_unavailable_reason="Calibration hash changed.",
    )

    assert spec2primitives_ui._calibration_readiness(ready)[:2] == (
        "calibration ready",
        "green",
    )
    assert spec2primitives_ui._calibration_readiness(unavailable) == (
        "calibration unavailable",
        "amber",
        "Calibration hash changed.",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

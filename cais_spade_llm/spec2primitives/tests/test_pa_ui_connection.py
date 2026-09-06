from __future__ import annotations

"""Focused tests for the native ProductAgent UI connection."""


import asyncio
import base64
import hashlib
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.spec2primitives import spec2primitives_ui
from cais_spade_llm.spec2primitives.agents.pa import product_agent_runtime
from cais_spade_llm.spec2primitives.tests.test_pa_completion import (
    _write_location,
    persist_native_completion_fixture,
)


def test_connected_ui_starts_one_native_grounding_call(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, object]] = []
    output = {
        "grounding_status": "incomplete",
        "insufficient_evidence": (
            "The active grounding contract has no valid value for current_state.state_values."
        ),
        "grounding_validation_code": "location_evidence_unavailable",
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
            return {"result": {"clarification_question": "Which variant is intended?"}}

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

    assert result == {"result": {"clarification_question": "Which variant is intended?"}}
    assert tool_results == [("retrieve", {"evidence_id": "evidence_0001"})]
    assert stages == ["Investigating approved evidence and authoring the target feature."]


def test_ui_observer_reports_revision_without_overwriting_allocation() -> None:
    """Label a second proposal as revision while preserving subsequent semantic stages."""
    stages: list[str] = []
    events: list[dict[str, str]] = []

    class Agent:
        async def ask_llm_structured(self, *args: Any, **kwargs: Any) -> dict[str, object]:
            return {}

    observer = spec2primitives_ui._PAUIRuntimeObserver(
        Agent(), on_pa_stage=stages.append, on_pa_event=events.append
    )

    async def run() -> None:
        for name in (
            "spec2primitives_grounding_result",
            "spec2primitives_grounding_result",
            "spec2primitives_resource_selection",
        ):
            await observer.ask_llm_structured("unchanged", response_format={"name": name})

    asyncio.run(run())
    assert stages[1] == "Revising the target feature · attempt 2."
    assert stages[2] == "Grounding the validated product context."
    assert [event["title"] for event in events] == ["ProductAgent revision · attempt 2"]


def _progress_snapshot(
    used: int = 3, attempt: int = 1, reason: str | None = None
) -> dict[str, object]:
    return {
        "evidence_operations_used": used,
        "evidence_operations_limit": 48,
        "proposals_used": attempt,
        "proposals_limit": 8,
        "last_feedback": [
            {
                "validation_code": "evidence_reference_invalid",
                "message": "An evidence reference is invalid.",
                "authorized_evidence_refs": ["opaque_reference_for_diagnostics"],
                "missing_states": ["desired_state"],
            }
        ]
        if attempt > 1
        else [],
        "stop_reason": reason,
    }


@pytest.mark.parametrize("outcome", ["accepted", "detached", "runtime_error", "cancelled"])
def test_grounding_progress_preserves_producer_and_tears_down_monitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """Show host counters without letting page lifetime alter producer outcomes."""
    monkeypatch.setattr(spec2primitives_ui, "_GROUNDING_PROGRESS_INTERVAL_SECONDS", 0.001)
    request_root = tmp_path / "products/grounding/ontology_grounding"
    request_root.mkdir(parents=True)
    record_root = tmp_path / "interaction_record"
    record_root.mkdir()
    (record_root / "tool_call_0000.json").write_text(
        json.dumps({"grounding_progress": _progress_snapshot(48, 8, "clarification_required")})
    )
    stages: list[str] = []
    detached_updates: list[str] = []
    with spec2primitives_ui.ui.column() as container:
        spec2primitives_ui.ui.label("Grounding progress")
    observer = spec2primitives_ui._PAUIRuntimeObserver(object(), on_pa_stage=lambda _: None)

    async def scenario() -> None:
        first_seen = asyncio.Event()
        revision_seen = asyncio.Event()

        def stage(message: str) -> None:
            if container.is_deleted:
                detached_updates.append(message)
            stages.append(message)
            if "3/48" in message:
                first_seen.set()
            if "7/48" in message:
                revision_seen.set()

        async def produce() -> dict[str, object]:
            (request_root / "request_0001_0001.json").write_text(
                json.dumps({"grounding_progress": _progress_snapshot()})
            )
            await asyncio.wait_for(first_seen.wait(), timeout=2)
            if outcome == "runtime_error":
                raise RuntimeError("Original producer failure.")
            if outcome == "cancelled":
                await asyncio.get_running_loop().create_future()
            if outcome == "detached":
                container.delete()
            (record_root / "tool_call_0007.json").write_text(
                json.dumps({"grounding_progress": _progress_snapshot(7, 2)})
            )
            if outcome == "accepted":
                await asyncio.wait_for(revision_seen.wait(), timeout=2)
                observer.activity = "Assigning an arm."
                await asyncio.sleep(0.01)
                assert stages[-1].startswith("Assigning an arm.")
                assert "Validation feedback" in stages[-1]
            return {"grounding_status": "complete"}

        task = asyncio.create_task(
            spec2primitives_ui._await_with_grounding_progress(
                produce(), tmp_path, stage, lambda: not container.is_deleted, observer
            )
        )
        if outcome == "cancelled":
            await asyncio.wait_for(first_seen.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome == "runtime_error":
            with pytest.raises(RuntimeError, match="Original producer failure"):
                await task
        else:
            assert await task == {"grounding_status": "complete"}
        assert not any(
            pending.get_coro().__name__ == "_monitor_grounding_progress"
            for pending in asyncio.all_tasks()
        )

    try:
        asyncio.run(scenario())
        assert "Proposal attempt: 1/8" in stages[0] and "Elapsed:" in stages[0]
        assert all("48/48" not in stage for stage in stages)
        assert detached_updates == []
    finally:
        if not container.is_deleted:
            container.delete()


def test_grounding_progress_preserves_caller_cancellation_during_monitor_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation during monitor cleanup cannot turn the producer result into success."""

    async def scenario() -> None:
        monitor_started = asyncio.Event()
        cleanup_started = asyncio.Event()

        async def monitor(*args: Any) -> None:
            monitor_started.set()
            try:
                await asyncio.get_running_loop().create_future()
            finally:
                cleanup_started.set()
                await asyncio.get_running_loop().create_future()

        async def produce() -> dict[str, object]:
            await monitor_started.wait()
            return {"grounding_status": "complete"}

        monkeypatch.setattr(spec2primitives_ui, "_monitor_grounding_progress", monitor)
        task = asyncio.create_task(
            spec2primitives_ui._await_with_grounding_progress(
                produce(), tmp_path, lambda _: None, None
            )
        )
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


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
            self.reasoning_effort = "low"
            created.append(self)

    monkeypatch.setattr(product_agent_runtime, "ProductAgent", _SharedAgent)

    runtime = product_agent_runtime.create_product_agent_context_runtime(
        model="gpt-5.6",
        reasoning_effort="none",
    )

    assert runtime is not None
    assert len(created) == 1
    assert created[0].reasoning_effort == "none"
    assert created[0].kwargs["model"] == "gpt-5.6"


@pytest.fixture
def persisted_ui_interaction(tmp_path: Path) -> dict[str, object]:
    """Provide a minimal persisted interaction for operator-status checks."""
    interaction_root = tmp_path / "interaction_native"
    requirement_path = interaction_root / "products/user_requirement/product_requirement.json"
    requirement_path.parent.mkdir(parents=True)
    requirement_path.write_text(json.dumps({"product_requirement": "assemble medium gear."}))
    (interaction_root / "interaction_record").mkdir()
    return {
        "interaction_identifier": interaction_root.name,
        "interaction_root": interaction_root,
        "product_requirement": "assemble medium gear.",
        "phase_3_1": {},
        "phase_3_2": None,
        "phase_3_3": None,
        "max_pa_turns": 12,
    }


@pytest.mark.parametrize("source", ["empty_live_results", "earlier_live_failure", "recovered"])
def test_ui_uses_latest_persisted_failure_consistently(
    persisted_ui_interaction: dict[str, object], source: str
) -> None:
    """Keep the saved runtime failure visible in the activity, timeline and diagnostics."""
    interaction = persisted_ui_interaction
    interaction_root = interaction["interaction_root"]
    failure = {
        "reason": "pa_call_failed",
        "message": (
            "ProductAgent grounding failed: RuntimeError: "
            "The client this element belongs to has been deleted."
        ),
        "diagnostic": {"stage": "native ProductAgent grounding", "exception": "RuntimeError"},
    }
    earlier_failure = {"reason": "earlier_failure", "message": "An earlier request failed."}
    records = interaction_root / "interaction_record"
    records.joinpath("turn_0001.json").write_text(
        json.dumps({"turn": 1, "PA_output": None, "failure": earlier_failure})
    )
    records.joinpath("turn_0002.json").write_text(
        json.dumps({"turn": 2, "PA_output": None, "failure": failure})
    )
    records.joinpath("tool_call_0001.json").write_text(json.dumps({"failure": None}))
    if source == "earlier_live_failure":
        interaction["phase_3_1"] = {"failure": earlier_failure}
    elif source == "recovered":
        interaction = spec2primitives_ui._latest_pa_ui_interaction(interaction_root.parent)
        assert interaction is not None
        assert interaction["recovered"] is True

    view = spec2primitives_ui._pa_ui_view(interaction)

    assert view["activity_state"] == "failed"
    assert view["activity_color"] == "red"
    assert view["activity_message"] == failure["message"]
    assert view["timeline"][-1]["state"] == "failed"
    assert view["timeline"][-1]["detail"] == failure["message"]
    assert failure["reason"] in view["diagnostics"]["failure"]
    assert failure["message"] in view["diagnostics"]["failure"]
    assert earlier_failure["message"] not in view["diagnostics"]["failure"]


def test_ui_keeps_persisted_incomplete_grounding_distinct_from_runtime_failure(
    persisted_ui_interaction: dict[str, object],
) -> None:
    """Show a valid grounding diagnostic as amber without inventing a runtime error."""
    interaction = persisted_ui_interaction
    output = {
        "grounding_status": "incomplete",
        "insufficient_evidence": "Reviewed desired_state locations are unavailable.",
        "grounding_validation_code": "location_evidence_unavailable",
        "tool_call_refs": [],
    }
    turn_path = interaction["interaction_root"] / "interaction_record/turn_0001.json"
    turn_path.write_text(json.dumps({"turn": 1, "PA_output": output, "failure": None}))

    view = spec2primitives_ui._pa_ui_view(interaction)

    assert view["activity_state"] == "grounding incomplete"
    assert view["activity_color"] == "amber"
    assert output["insufficient_evidence"] in view["activity_message"]
    assert "location_evidence_unavailable" in view["activity_message"]
    assert view["timeline"][-1]["title"] == "Grounding incomplete"
    assert view["diagnostics"]["failure"] == ""


def test_persisted_grounding_progress_shows_budget_feedback_and_stop_reason(
    persisted_ui_interaction: dict[str, object],
) -> None:
    interaction = persisted_ui_interaction
    root = interaction["interaction_root"]
    output = {
        "grounding_status": "incomplete",
        "grounding_stage": "ontology_grounding",
        "grounding_validation_code": "grounding_budget_exhausted",
        "insufficient_evidence": "The grounding investigation exhausted its configured budget.",
        "grounding_progress": _progress_snapshot(48, 8, "grounding_budget_exhausted"),
        "tool_call_refs": [],
    }
    turn_path = root / "interaction_record/turn_0001.json"
    turn_path.write_text(json.dumps({"turn": 1, "PA_output": output, "failure": None}))
    for index, tool in enumerate(("retrieve", "query_document", "compare_cad_size"), start=1):
        (root / f"interaction_record/tool_call_{index:04d}.json").write_text(
            json.dumps(
                {
                    "tool_name": tool,
                    "failure": None,
                    "grounding_progress": _progress_snapshot(index),
                }
            )
        )
    before = turn_path.read_bytes()

    view = spec2primitives_ui._pa_ui_view(interaction)

    assert view["activity_state"] == "grounding incomplete"
    assert "Evidence operations: 48/48" in view["activity_message"]
    assert "Proposal attempt: 8/8" in view["activity_message"]
    assert "An evidence reference is invalid." in view["activity_message"]
    assert "Affected states: desired_state." in view["activity_message"]
    assert "opaque_reference_for_diagnostics" not in view["activity_message"]
    assert "grounding_budget_exhausted" in view["activity_message"]
    assert [event["title"] for event in view["timeline"]] == [
        "Requirement received",
        "Evidence investigated",
        "Grounding investigation",
        "Grounding incomplete",
    ]
    assert view["timeline"][1]["detail"] == "3 approved evidence operations completed."
    assert view["timeline"][2]["state"] == "waiting"
    assert view["diagnostics"]["grounding_progress"] == output["grounding_progress"]
    assert turn_path.read_bytes() == before


def test_removed_semantic_result_is_incompatible_and_cannot_claim_grounded(
    persisted_ui_interaction: dict[str, object],
) -> None:
    interaction = persisted_ui_interaction
    root = interaction["interaction_root"]
    proposal = root / "products/grounding/ontology_grounding/proposal_0001.json"
    proposal.parent.mkdir(parents=True)
    proposal.write_text(
        json.dumps({"status": "accepted", "semantic_review_ref": "old/review.json"})
    )
    turn = root / "interaction_record/turn_0001.json"
    turn.write_text(
        json.dumps(
            {
                "turn": 1,
                "failure": None,
                "PA_output": {
                    "grounding_status": "incomplete",
                    "grounding_validation_code": "semantic_review_unavailable",
                    "insufficient_evidence": "Independent source review could not be completed.",
                },
            }
        )
    )
    originals = {path: path.read_bytes() for path in (proposal, turn)}

    view = spec2primitives_ui._pa_ui_view(interaction)

    assert view["activity_state"] == "failed"
    assert "Start a fresh interaction" in view["activity_message"]
    assert not any(event["title"] == "Target feature grounded" for event in view["timeline"])
    assert view["final_result"] is None
    assert all(path.read_bytes() == before for path, before in originals.items())


def test_persisted_progress_uses_latest_turn_after_clarification() -> None:
    previous = _progress_snapshot(40, 6, "clarification_required")
    current = _progress_snapshot(3, 1, "complete")
    assert (
        spec2primitives_ui._persisted_grounding_progress(
            {
                "turn_0001": {"PA_output": {"grounding_progress": previous}},
                "turn_0002": {"PA_output": {"grounding_progress": current}},
                "grounding_request_0006": {"grounding_progress": previous},
            }
        )
        == current
    )


def _reject_deleted_page_updates(monkeypatch: pytest.MonkeyPatch, container: Any) -> None:
    """Fail if detached callbacks update deleted elements or notify their old page."""
    from nicegui.element import Element

    original_update = Element.update

    def update_live_element(element: Element) -> None:
        if element.is_deleted:
            pytest.fail("Grounding updated an element after its owning page was deleted.")
        original_update(element)

    def notify(*args: Any, **kwargs: Any) -> None:
        if container.is_deleted:
            pytest.fail("Grounding notified a deleted page.")

    monkeypatch.setattr(Element, "update", update_live_element)
    monkeypatch.setattr(spec2primitives_ui.ui, "notify", notify)


@pytest.mark.parametrize(
    ("action", "outcome"),
    [
        ("start", "returned"),
        ("clarification", "returned"),
        ("start", "runtime_error"),
        ("clarification", "cancelled"),
    ],
)
def test_rendered_interaction_detaches_without_interrupting_grounding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    outcome: str,
) -> None:
    """Dead page callbacks cannot prevent persistence or replace a real exception."""
    interaction_root = tmp_path / "interaction_detached"
    records = interaction_root / "interaction_record"
    requirement = "assemble medium gear."
    interaction = {
        "interaction_identifier": interaction_root.name,
        "interaction_root": interaction_root,
        "product_requirement": requirement,
        "phase_3_1": {},
        "phase_3_2": None,
        "phase_3_3": None,
        "max_pa_turns": 12,
    }
    requirement_path = interaction_root / "products/user_requirement/product_requirement.json"
    if action == "clarification":
        requirement_path.parent.mkdir(parents=True)
        requirement_path.write_text(json.dumps({"product_requirement": requirement}))
        records.mkdir()
        records.joinpath("turn_0001.json").write_text(
            json.dumps(
                {
                    "turn": 1,
                    "PA_output": {
                        "grounding_status": "clarification_required",
                        "clarification_question": "Which product variant is intended?",
                        "tool_call_refs": [],
                    },
                    "failure": None,
                }
            )
        )

    buttons: dict[str, Callable[..., Any]] = {}
    inputs: dict[str, Any] = {}
    original_on_click = spec2primitives_ui.ui.button.on_click
    original_input = spec2primitives_ui.ui.input
    callback_events: list[str] = []

    def capture_on_click(button: Any, callback: Callable[..., Any]) -> Any:
        buttons[button.text] = callback
        return original_on_click(button, callback)

    def capture_input(*args: Any, **kwargs: Any) -> Any:
        element = original_input(*args, **kwargs)
        inputs[kwargs["label"]] = element
        return element

    async def finish_grounding(
        *args: Any,
        on_pa_stage: Callable[[str], None],
        on_pa_event: Callable[[dict[str, str]], None],
        **kwargs: Any,
    ) -> dict[str, object]:
        await asyncio.sleep(0)
        container.delete()
        _reject_deleted_page_updates(monkeypatch, container)
        on_pa_stage("Reviewing retrieved evidence.")
        callback_events.append("stage")
        on_pa_event(
            {"state": "running", "title": "Evidence reviewed", "detail": "Review finished."}
        )
        callback_events.append("event")
        if outcome == "cancelled":
            raise asyncio.CancelledError
        output = {
            "grounding_status": "incomplete",
            "insufficient_evidence": "Reviewed desired_state locations are unavailable.",
            "grounding_validation_code": "location_evidence_unavailable",
            "tool_call_refs": [],
        }
        requirement_path.parent.mkdir(parents=True, exist_ok=True)
        requirement_path.write_text(json.dumps({"product_requirement": requirement}))
        records.mkdir(exist_ok=True)
        turn_number = 1 if action == "start" else 2
        failure = (
            {"reason": "pa_call_failed", "message": "Independent grounding failure."}
            if outcome == "runtime_error"
            else None
        )
        records.joinpath(f"turn_{turn_number:04d}.json").write_text(
            json.dumps(
                {
                    "turn": turn_number,
                    "PA_output": output if failure is None else None,
                    "failure": failure,
                }
            )
        )
        if outcome == "runtime_error":
            raise RuntimeError("Independent grounding failure.")
        interaction["phase_3_1" if action == "start" else "phase_3_3"] = output
        return interaction

    monkeypatch.setattr(spec2primitives_ui.ui.button, "on_click", capture_on_click)
    monkeypatch.setattr(spec2primitives_ui.ui, "input", capture_input)
    monkeypatch.setattr(spec2primitives_ui, "_run_pa_ui_interaction", finish_grounding)
    monkeypatch.setattr(spec2primitives_ui, "_submit_pa_ui_clarification", finish_grounding)
    runtime = SimpleNamespace(
        contexts_root=tmp_path,
        product_agent=object(),
        ontology_config=object(),
        grounding_runtime=object(),
        robot_agent_context_runtime=None,
        robot_agent_draft_runtime=None,
        document_diagnostic_unavailable_reason=None,
        camera_to_world_calibration_runtime=None,
        camera_to_world_calibration_unavailable_reason=None,
        model_config=None,
    )
    with spec2primitives_ui.ui.column() as container:
        spec2primitives_ui._render_pa_interaction(runtime)
    try:
        inputs["product_requirement"].value = requirement
        if action == "clarification":
            inputs["User reply"].value = "The supplied medium gear."
        callback = buttons["Start ProductAgent" if action == "start" else "Submit Reply"]
        if outcome == "runtime_error":
            with pytest.raises(RuntimeError, match="Independent grounding failure"):
                asyncio.run(callback())
        elif outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(callback())
        else:
            asyncio.run(callback())

        assert container.is_deleted
        assert callback_events == ["stage", "event"]
        turn_number = 1 if action == "start" else 2
        turn_path = records / f"turn_{turn_number:04d}.json"
        if outcome == "cancelled":
            assert not turn_path.exists()
        else:
            turn = _read_json(turn_path)
            if outcome == "returned":
                assert turn["PA_output"]["grounding_status"] == "incomplete"
            else:
                assert turn["PA_output"] is None
            assert (turn["failure"] is None) is (outcome == "returned")
    finally:
        if not container.is_deleted:
            container.delete()


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
        "Arm assigned",
        "Product grounding and arm assignment complete",
    ]
    final_result = view["final_result"]
    assert isinstance(final_result, dict)
    assert final_result["target_frame"] == "world"
    assert final_result["selected_resource"] == "xarm6"
    assert final_result["validation_scope"] == "moveit_state_location_reachability"
    assert final_result["motion_executed"] is False
    assert "grasp" in final_result["limitations"]
    assert "joint_limits" in final_result["checked_constraints"]
    assert final_result["reachability"]["status"] == "accepted"
    assert final_result["reachability"]["resource_symbol"] == "xarm6"
    assert final_result["reachability"]["process_symbol"] == "assembly"
    assert final_result["reachability"]["target_frame"] == "world"
    assert final_result["robot_agent_validation"]["response"]["status"] == "accepted"
    assert final_result["motion_validation_performed"] is True
    assert final_result["reachability"]["state_locations"] is not None
    assert final_result["target_feature"]["desired_state"]["statement"]["text"] == (
        "The medium gear is assembled as requested."
    )
    assert final_result["target_feature"]["current_state"]["state_values"][0]["name"] == (
        "medium_gear_location"
    )
    assert final_result["target_feature"]["desired_state"]["state_values"][0]["name"] == (
        "assembly_board_shaft_location"
    )
    assert (
        len(final_result["target_feature"]["assembly_feature_association"][0]["assembly_features"])
        == 2
    )
    current = final_result["current_state_evidence"]
    desired = final_result["desired_state_evidence"]
    assert len(current["reachability_locations"]) == 2
    assert len(desired["reachability_locations"]) == 1
    assert len(current["candidate_visuals"]) == 2
    assert len(desired["candidate_visuals"]) == 1
    assert desired["candidate_visuals"][0]["state_value_name"] == "assembly_board_shaft_location"
    assert any(
        visual["state_value_name"] == "medium_gear_location"
        and visual["source_state"] == "current_state"
        for visual in desired["relationship_visuals"]
    )
    assert len(desired["relationships"]) == 1
    assert "do not establish that assembly is complete" in desired["reference_note"]
    assert all(
        item["data_uri"].startswith("data:image/svg+xml;base64,")
        for item in desired["candidate_visuals"]
    )
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


def test_timeline_distinguishes_grounded_deferred_clarification_and_incomplete() -> None:
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
                    "insufficient_evidence": (
                        "The active grounding contract has no valid value for "
                        "desired_state.state_values."
                    ),
                    "grounding_validation_code": "location_evidence_unavailable",
                }
            }
        ],
        retrievals=[],
        clarifications=[],
        records={},
        final_result=None,
        terminal_failure=None,
    )
    grounded = spec2primitives_ui._persisted_pa_timeline(
        "assemble medium gear",
        turns=[
            {
                "PA_output": {
                    "grounding_status": "complete",
                    "resource_assignment_status": "deferred",
                    "ontology_projection_ref": "proposal.json",
                    "tool_call_refs": [],
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
    assert incomplete[-1]["detail"] == (
        "The active grounding contract has no valid value for "
        "desired_state.state_values. Validation code: location_evidence_unavailable."
    )
    assert grounded[-1] == {
        "state": "accepted",
        "title": "Product-state grounding complete",
        "detail": (
            "The current and desired states are validated; resource assignment is deferred."
        ),
    }


def test_product_state_grounding_view_is_complete_when_assignment_is_deferred(
    tmp_path: Path,
) -> None:
    interaction_root = tmp_path / "interaction"
    completion = persist_native_completion_fixture(interaction_root).to_record()
    (interaction_root / "interaction_record/context_completion_0001.json").unlink()
    turn_path = interaction_root / "interaction_record/turn_0001.json"
    output = {
        "grounding_status": "complete",
        "resource_assignment_status": "deferred",
        "ontology_projection_ref": completion["ontology_projection_ref"],
        "deferred_reason": "Reviewed grounding does not identify exactly one supported Cartesian coordinate pair.",
        "tool_call_refs": [],
    }
    turn_path.write_text(json.dumps({"PA_output": output}), encoding="utf-8")
    interaction = {
        "interaction_identifier": interaction_root.name,
        "interaction_root": interaction_root,
        "product_requirement": "assemble medium gear",
        "phase_3_1": output,
        "phase_3_2": None,
        "phase_3_3": output,
    }

    view = spec2primitives_ui._pa_ui_view(interaction)

    assert view["activity_state"] == "product state grounded"
    assert view["activity_color"] == "green"
    assert "review_status" not in view["final_result"]
    assert view["final_result"]["allocation_label"] == "Product grounding; allocation deferred"
    result = view["final_result"]
    assert result["ontology"]["rows"]
    assert "hascurrentstate" in result["ontology"]["raw_turtle"]
    assert "hasdesiredstate" in result["ontology"]["mermaid"]
    for state in ("current_state", "desired_state"):
        visuals = result[f"{state}_evidence"]["candidate_visuals"]
        assert len(visuals) == (2 if state == "current_state" else 1)
        assert visuals[0]["data_uri"].startswith("data:image/")
        assert visuals[0]["source_view_data_uri"].startswith("data:image/svg+xml;base64,")
    assert "reachability" not in result
    assert "robot_agent_validation" not in result
    assert "selected_resource" not in result
    summary = spec2primitives_ui._concise_final_grounding_summary(result)
    assert output["deferred_reason"] in summary["target"]
    assert "still incomplete" in summary["target"]
    assert summary["validation"] == "Not checked: resource assignment is deferred."

    elements = spec2primitives_ui._render_final_grounding_result()
    try:
        spec2primitives_ui._apply_final_grounding_result(elements, result)
        assert elements["card"].visible
        assert elements["ontology_table"].rows
        assert elements["current_images"].default_slot.children
        assert elements["desired_images"].default_slot.children
        assert elements["resource"].text == "Deferred"
    finally:
        elements["card"].delete()

    assert all("allocation validated" not in event["title"] for event in view["timeline"])


@pytest.mark.parametrize("candidate_count", [0, 1, 2])
def test_deferred_state_gallery_preserves_multiple_current_candidates_and_empty_desired_state(
    tmp_path: Path,
    candidate_count: int,
) -> None:
    values = []
    artifact_hashes = {}
    for name in ("medium_gear_location", "gear_shaft_location")[:candidate_count]:
        location_path, _ = _write_location(tmp_path, name, (0.0, 0.0, 0.0))
        location = json.loads(location_path.read_text())
        segmentation_ref = location["source_segmentation"]["ref"]
        segmentation_path = tmp_path / segmentation_ref
        segmentation = json.loads(segmentation_path.read_text())
        candidates = segmentation["cameras"][0]["candidates"]
        candidates.append({**candidates[0], "candidate_handle": "unselected_candidate"})
        segmentation_path.write_text(json.dumps(segmentation))
        artifact_hashes[segmentation_ref] = hashlib.sha256(
            segmentation_path.read_bytes()
        ).hexdigest()
        values.append(
            {
                "name": name,
                "value_ref": {
                    "record_ref": segmentation_ref,
                    "field_path": "/cameras/0/candidates/0",
                },
            }
        )
    current = {
        "candidate_visuals": spec2primitives_ui._state_visuals(
            tmp_path, {"state_values": values}, artifact_hashes, {}
        )
    }
    desired = {
        "candidate_visuals": spec2primitives_ui._state_visuals(
            tmp_path, {"state_values": []}, artifact_hashes, {}
        )
    }
    assert [visual["state_value_name"] for visual in current["candidate_visuals"]] == [
        value["name"] for value in values
    ]
    assert all(
        visual["candidate_reference"]["candidate_handle"] != "unselected_candidate"
        for visual in current["candidate_visuals"]
    )
    assert desired["candidate_visuals"] == []
    container = spec2primitives_ui.ui.column()
    try:
        spec2primitives_ui._render_state_visuals(container, current)
        images = [
            child
            for child in container.default_slot.children
            if isinstance(child, spec2primitives_ui.ui.image)
        ]
        assert len(images) == candidate_count
        assert [image.source for image in images] == [
            visual["data_uri"] for visual in current["candidate_visuals"]
        ]
        spec2primitives_ui._render_state_visuals(container, desired)
        assert len(container.default_slot.children) == 1
        assert container.default_slot.children[0].text == (
            "No image is directly bound to this state."
        )
    finally:
        container.delete()


def test_ui_does_not_render_uncoded_legacy_failure_prose() -> None:
    detail = spec2primitives_ui._grounding_incomplete_detail(
        {
            "grounding_status": "incomplete",
            "insufficient_evidence": "PA-authored legacy mounting and meshing prose.",
        }
    )

    assert detail == ("This record has no valid grounding result. Start a fresh interaction.")
    assert "mounting" not in detail


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
    assert [candidate["context_ref"] for candidate in identity["candidates"]] == context_refs
    summary = spec2primitives_ui._concise_final_grounding_summary(
        {
            "cad_identity": identity,
            "target_frame": "world",
            "current_state_evidence": {"translated_location_m": [0.1, 0.2, 0.3]},
            "desired_state_evidence": {"translated_location_m": [0.4, 0.5, 0.6]},
            "reachability": {"status": "accepted"},
            "robot_agent_validation": {},
        }
    )
    assert str(summary["target"]).splitlines()[0] == "CAD identity unresolved"


def test_single_uncited_cad_identity_remains_unresolved(tmp_path: Path) -> None:
    record_ref = "products/grounding/cad/candidate.json"
    record_path = tmp_path / record_ref
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps({"source": {"context_ref": "approved-cad"}}),
        encoding="utf-8",
    )

    identity = spec2primitives_ui._cad_identity(
        tmp_path,
        typed_bindings=[
            {
                "output_symbol": "CADMeshRecord",
                "status": "accepted",
                "record_ref": record_ref,
                "evidence_refs": ["approved-cad"],
            }
        ],
        target_feature={"evidence_refs": ["requirement_0001"]},
    )

    assert identity is None


def test_reviewed_crop_requires_exact_accepted_binding_hash(tmp_path: Path) -> None:
    segmentation_ref = "products/grounding/segmentation.json"
    crop_ref = "products/grounding/review/candidate.png"
    review_ref = "products/grounding/review/observation_candidate_review.json"
    crop_bytes = b"stable reviewed crop"
    crop_path = tmp_path / crop_ref
    crop_path.parent.mkdir(parents=True)
    crop_path.write_bytes(crop_bytes)
    review_path = tmp_path / review_ref
    review_path.write_text(
        json.dumps(
            {
                "record_type": "ObservationCandidateReview",
                "status": "accepted",
                "source_segmentation": {
                    "ref": segmentation_ref,
                    "sha256": "a" * 64,
                },
                "candidates": [
                    {
                        "observation_handle": "view_0001",
                        "candidate_handle": "candidate_0001_0001",
                        "description": "visible medium circular part",
                        "uncertainty": "identity not assigned",
                        "crop_artifact": {
                            "ref": crop_ref,
                            "sha256": hashlib.sha256(crop_bytes).hexdigest(),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    review_sha256 = hashlib.sha256(review_path.read_bytes()).hexdigest()

    assert (
        spec2primitives_ui._reviewed_candidate_crop(
            tmp_path,
            segmentation_ref=segmentation_ref,
            segmentation_sha256="a" * 64,
            observation_handle="view_0001",
            candidate_handle="candidate_0001_0001",
            observation_review_bindings={},
        )
        is None
    )
    reviewed = spec2primitives_ui._reviewed_candidate_crop(
        tmp_path,
        segmentation_ref=segmentation_ref,
        segmentation_sha256="a" * 64,
        observation_handle="view_0001",
        candidate_handle="candidate_0001_0001",
        observation_review_bindings={review_ref: review_sha256},
    )
    assert reviewed is not None
    assert base64.b64decode(str(reviewed["data_uri"]).split(",", maxsplit=1)[1]) == crop_bytes
    assert reviewed["observation_review_ref"] == review_ref


def test_concise_grounding_summary_keeps_raw_details_out_of_top_level() -> None:
    summary = spec2primitives_ui._concise_final_grounding_summary(
        {
            "cad_identity": {
                "status": "accepted",
                "context_ref": "approved/CAD/Gear_Medium.STL",
                "record_ref": "products/grounding/cad/record.json",
                "bounds_m": {"minimum": [0.0, 0.0, 0.0]},
            },
            "target_frame": "world",
            "current_state_evidence": {
                "statement": "The medium gear is separate.",
                "state_value_name": "medium_gear",
                "translated_location_m": [0.1, 0.2, 0.3],
                "reachable": True,
            },
            "desired_state_evidence": {
                "statement": "The medium gear is at the supported destination.",
                "state_value_name": "supported_destination",
                "translated_location_m": [0.4, 0.5, 0.6],
                "reachable": True,
            },
            "reachability": {
                "status": "accepted",
                "resource_symbol": "selected_resource",
                "target_frame": "world",
                "record_ref": "products/grounding/reachability/check.json",
            },
            "robot_agent_validation": {
                "status": "accepted",
                "mode": "plan_only",
                "validation_scope": "endpoint_motion",
                "motion_executed": False,
            },
        }
    )

    target = str(summary["target"])
    assert target.splitlines() == [
        "CAD: Gear_Medium.STL",
        "target frame: world",
        "current references checked: 0",
        "destination references checked: 0",
    ]
    assert "bounds_m" not in target
    assert "record_ref" not in target
    assert "products/grounding" not in target


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


def test_source_caveats_survive_completion_and_ui_reload(tmp_path):
    completion = persist_native_completion_fixture(
        tmp_path, include_target_state_values=True
    ).to_record()
    result = spec2primitives_ui._final_grounding_result(tmp_path, completion)
    assert "review_status" not in result
    assert "inferred from a figure" in result["limitations"][0]
    assert result["desired_state_evidence"]["source_uncertainty"]
    assert (
        spec2primitives_ui._concise_state_summary(result["desired_state_evidence"])
        == result["desired_state_evidence"]["statement"]
    )


@pytest.mark.parametrize("width", [320, 1280])
def test_statement_preview_and_expanded_evidence_use_contained_cards(tmp_path, width):
    """Check offline rendering; actual pixel overflow requires an available browser."""
    completion = persist_native_completion_fixture(
        tmp_path, include_target_state_values=True
    ).to_record()
    result = spec2primitives_ui._final_grounding_result(tmp_path, completion)
    statement = "An exact reviewed statement.\n" * 12
    source = "products/" + "long_source_path_" * 80 + "/record.json#field"
    result["desired_state_evidence"]["statement"] = statement
    result["desired_state_evidence"]["source_uncertainty"] = [
        {"description": "Exact caveat.", "evidence_refs": [source]}
    ]
    elements = spec2primitives_ui._render_final_grounding_result()
    try:
        elements["card"].style(f"width: {width}px")
        spec2primitives_ui._apply_final_grounding_result(elements, result)
        assert elements["desired_state"].text == statement
        assert "s2p-state-statement" in elements["desired_state"].classes
        assert elements["desired_caveat_count"].text == "1 caveats"
        assert elements["desired_show_more"].text == "Show more"
        assert "min-w-0" in elements["card"].classes
        assert "max-w-full" in elements["desired_state_details"].classes
        assert source in [child.text for child in elements["desired_caveats"].default_slot.children]
        assert source not in elements["desired_state"].text
        rows = elements["resource_comparison"].rows
        assert {row["resource"] for row in rows} == {"xarm6", "ur5e"}
        assert all(row["current_state"] == row["desired_state"] == "Reachable" for row in rows)
        assert [row["resource"] for row in rows if row["selected"]] == ["xarm6"]
    finally:
        elements["card"].delete()


@pytest.fixture
def statement_preview(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Create real preview elements with controlled browser and timer callbacks."""
    callbacks: dict[str, Any] = {}
    original_button = spec2primitives_ui.ui.button

    def button(*args: Any, **kwargs: Any) -> Any:
        callbacks["toggle"] = kwargs["on_click"]
        return original_button(*args, **kwargs)

    def timer(interval: float, callback: Callable[..., Any], **kwargs: Any) -> None:
        callbacks["measure"] = callback

    monkeypatch.setattr(spec2primitives_ui.ui, "button", button)
    monkeypatch.setattr(spec2primitives_ui.ui, "timer", timer)
    with spec2primitives_ui.ui.column() as container:
        widget = spec2primitives_ui._render_state_statement()
    client = widget["statement"].client
    connected = [True]
    monkeypatch.setattr(type(client), "has_socket_connection", property(lambda self: connected[0]))
    widget["statement"].set_text("The exact reviewed state statement.\n" * 12)
    widget["reset"]()
    try:
        yield {
            **widget,
            "callbacks": callbacks,
            "client": client,
            "connected": connected,
            "container": container,
        }
    finally:
        if not container.is_deleted:
            container.delete()


@pytest.mark.parametrize(
    ("reply", "statement", "visible"),
    [
        (True, "Long statement", True),
        (False, "Short statement", False),
        ("timeout", "Long statement", True),
        ("timeout", "", False),
    ],
)
def test_statement_measurement_timeout_preserves_access(
    statement_preview: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    reply: bool | str,
    statement: str,
    visible: bool,
) -> None:
    """Keep expansion accessible when the browser fails to measure nonempty text."""
    widget = statement_preview

    async def run_javascript(code: str) -> bool:
        if reply == "timeout":
            raise TimeoutError("JavaScript did not respond within 1.0 s")
        return bool(reply)

    monkeypatch.setattr(widget["client"], "run_javascript", run_javascript)
    widget["statement"].set_text(statement)
    widget["reset"]()
    asyncio.run(widget["callbacks"]["measure"]())
    assert widget["show_more"].visible is visible
    assert widget["statement"].text == statement
    if visible:
        widget["callbacks"]["toggle"]()
        assert widget["show_more"].text == "Show less"
        assert "s2p-state-expanded" in widget["statement"].classes


def test_statement_measurement_skips_disconnected_browser(
    statement_preview: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave text expandable without sending a request to a disconnected browser."""
    widget = statement_preview
    widget["connected"][0] = False

    async def run_javascript(code: str) -> bool:
        pytest.fail("Disconnected preview requested a browser measurement.")

    monkeypatch.setattr(widget["client"], "run_javascript", run_javascript)
    asyncio.run(widget["callbacks"]["measure"]())
    assert widget["show_more"].visible is True


@pytest.mark.parametrize("change", ["reset", "expand", "collapse", "delete", "disconnect"])
def test_statement_measurement_ignores_overlap_and_stale_responses(
    statement_preview: dict[str, Any], monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """A delayed browser response cannot override a newer or deleted preview."""
    widget = statement_preview

    async def scenario() -> None:
        reply = asyncio.get_running_loop().create_future()
        started = asyncio.Event()
        calls = 0

        async def run_javascript(code: str) -> bool:
            nonlocal calls
            calls += 1
            started.set()
            return await reply

        monkeypatch.setattr(widget["client"], "run_javascript", run_javascript)
        measure = widget["callbacks"]["measure"]
        task = asyncio.create_task(measure())
        await started.wait()
        await measure()
        assert calls == 1
        if change == "reset":
            widget["statement"].set_text("A new reviewed statement.")
            widget["reset"]()
        elif change in {"expand", "collapse"}:
            widget["callbacks"]["toggle"]()
            if change == "collapse":
                widget["callbacks"]["toggle"]()
        elif change == "delete":
            widget["container"].delete()

            def reject_visibility_update(visible: bool) -> None:
                pytest.fail("Deleted preview received a visibility update.")

            monkeypatch.setattr(widget["show_more"], "set_visibility", reject_visibility_update)
        else:
            widget["connected"][0] = False
        reply.set_result(False)
        await task
        if change == "delete":
            assert widget["statement"].is_deleted
        else:
            assert widget["show_more"].visible is True

    asyncio.run(scenario())


def test_statement_measurement_preserves_cancellation_and_can_measure_again(
    statement_preview: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation propagates and releases the in-flight measurement guard."""
    widget = statement_preview

    async def scenario() -> None:
        started = asyncio.Event()

        async def run_javascript(code: str) -> bool:
            started.set()
            await asyncio.get_running_loop().create_future()
            return False

        monkeypatch.setattr(widget["client"], "run_javascript", run_javascript)
        measure = widget["callbacks"]["measure"]
        task = asyncio.create_task(measure())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert widget["show_more"].visible is True

        async def completed_measurement(code: str) -> bool:
            return False

        monkeypatch.setattr(widget["client"], "run_javascript", completed_measurement)
        await measure()
        assert widget["show_more"].visible is False

    asyncio.run(scenario())


def test_robot_context_waiting_label_preserves_diagnostic_status() -> None:
    """Display operation names while keeping persisted status symbols unchanged."""
    diagnostic = spec2primitives_ui._phase_5_1_waiting_view("Waiting for grounded states.")
    with spec2primitives_ui.ui.column() as container:
        elements = spec2primitives_ui._render_phase_5_diagnostics()
    try:
        spec2primitives_ui._apply_phase_5_1_diagnostic(elements, diagnostic)
        assert elements["status_badge"].text == "Waiting for grounding"
        assert elements["start_button"].text == "Capture RobotAgent context"
        assert diagnostic["status"] == "waiting_for_phase_4"
    finally:
        container.delete()


def test_incompatible_saved_completion_is_rejected_without_rewriting(tmp_path):
    persist_native_completion_fixture(tmp_path)
    path = tmp_path / "interaction_record/context_completion_0001.json"
    record = _read_json(path)
    record["schema_version"] = 11
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    turn = _read_json(tmp_path / "interaction_record/turn_0001.json")
    view = spec2primitives_ui._pa_ui_view(
        {
            "interaction_identifier": tmp_path.name,
            "interaction_root": tmp_path,
            "product_requirement": "assemble medium gear",
            "phase_3_1": turn["PA_output"],
            "phase_3_2": None,
            "phase_3_3": turn["PA_output"],
        }
    )
    assert view["final_result"] is None
    assert "Start a fresh interaction" in view["activity_message"]
    assert path.read_bytes() == before

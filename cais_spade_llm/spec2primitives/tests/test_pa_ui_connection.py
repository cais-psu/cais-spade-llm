from __future__ import annotations

"""Focused tests for the native ProductAgent UI connection."""


import asyncio
import base64
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.spec2primitives import spec2primitives_ui
from cais_spade_llm.spec2primitives.agents.ra.validation_scope import (
    GAZEBO_PICK_PLACE_SCOPE, read_validation_scope,
)
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
        robot_agent_program_runtime=None,
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
    composition = view["primitive_composition"]
    assert isinstance(composition, dict)
    assert composition["status"] == "waiting_for_context"
    assert composition["candidate"] is None
    assert "phase_5_2" not in view
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


@pytest.mark.parametrize("detach", ["never", "validating", "authoring", "refreshing"])
@pytest.mark.parametrize("refinement", [False, True])
def test_compose_button_records_candidate_when_page_detaches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    detach: str,
    refinement: bool,
) -> None:
    """Compose once without a draft, keeping clicks responsive after page detachment."""
    from cais_spade_llm.spec2primitives.adapters.ui_runtime import Spec2PrimitivesUIRuntime
    from cais_spade_llm.spec2primitives.agents.ra import primitive_composition
    from cais_spade_llm.spec2primitives.tests.test_ra_context_handoff import (
        _prepare_composition,
        _program_action,
        _ProgramRuntime,
    )

    root = tmp_path / "interaction_composition"
    adapter, _, _ = _prepare_composition(root)
    buttons: dict[str, Callable] = {}
    phase_elements: dict[str, Any] = {}
    original_render = spec2primitives_ui._render_phase_5_diagnostics

    def capture_elements() -> dict[str, Any]:
        elements = original_render()
        assert elements["diagnostic_deadline"].value is False
        if refinement:
            elements["diagnostic_deadline"].set_value(True)
        phase_elements.update(elements)
        return elements

    monkeypatch.setattr(spec2primitives_ui, "_render_phase_5_diagnostics", capture_elements)
    original_on_click = spec2primitives_ui.ui.button.on_click

    def capture_on_click(button: Any, callback: Callable) -> Any:
        buttons[button.text] = callback
        return original_on_click(button, callback)

    def detach_page(stage: str) -> None:
        if detach == stage:
            container.delete()
            _reject_deleted_page_updates(monkeypatch, container)

    program_runtime = _ProgramRuntime(
        [
            _program_action([("move_cartesian", {"z": 0.07})]),
        ],
        on_call=lambda: detach_page("authoring"),
    )
    runtime = Spec2PrimitivesUIRuntime(
        dual_gazebo=object(),
        product_agent=object(),
        contexts_root=tmp_path,
        robot_agent_context_runtime=adapter,
        robot_agent_program_runtime=program_runtime,
    )
    if refinement:
        from dataclasses import replace
        from cais_spade_llm.spec2primitives.agents.ra.refinement import PrimitiveRefinementRuntime, load_refinement_profile

        class MeasuredContext:
            async def capture_validation_context(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Controlled UI fixture has no measured robot context.")

        async def validate_fixture(**kwargs: Any) -> dict[str, Any]:
            return {"scope": read_validation_scope(kwargs["profile"]), "status": "unknown", "findings": [{"step_index": None, "check": "motion", "status": "unknown", "message": "No live motion validator in this controlled UI test."}], "calculation_refs": []}

        runtime = replace(runtime, primitive_refinement_runtime=PrimitiveRefinementRuntime(program_runtime=program_runtime, robot_runtime=MeasuredContext(), validator=validate_fixture, profile={**load_refinement_profile(), "max_candidates": 1}))
    monkeypatch.setattr(spec2primitives_ui.ui.button, "on_click", capture_on_click)
    with spec2primitives_ui.ui.column() as container:
        spec2primitives_ui._render_pa_interaction(runtime)
    assert "Create Primitive Draft" not in buttons

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        validating = asyncio.Event()
        release = Event()
        blocked: list[str] = []
        input_reads = 0

        def slow_check(reader: Callable) -> Callable:
            def read(*args: Any) -> Any:
                nonlocal input_reads
                if reader.__name__ == "_load_inputs":
                    input_reads += 1
                    if input_reads == 1:
                        loop.call_soon_threadsafe(validating.set)
                        if not release.wait(timeout=0.5):
                            blocked.append("initial validation")
                elif reader.__name__ == "read_primitive_composition_diagnostic":
                    loop.call_soon_threadsafe(detach_page, "refreshing")
                heartbeat = Event()
                loop.call_soon_threadsafe(heartbeat.set)
                if not heartbeat.wait(timeout=0.5):
                    blocked.append(reader.__name__)
                return reader(*args)

            return read

        with monkeypatch.context() as patch:
            patch.setattr(
                primitive_composition,
                "_load_inputs",
                slow_check(primitive_composition._load_inputs),
            )
            for name in ("read_primitive_composition_diagnostic", "read_phase_5_1_diagnostic"):
                patch.setattr(
                    spec2primitives_ui, name, slow_check(getattr(spec2primitives_ui, name))
                )
            task = asyncio.create_task(buttons["Compose Primitive Program"]())
            try:
                await asyncio.wait_for(validating.wait(), timeout=5)
                assert not task.done(), "Validation blocked the UI until composition finished."
                await buttons["Compose Primitive Program"]()
                detach_page("validating")
            finally:
                release.set()
                await task
        assert blocked == [], f"UI validation blocked connection heartbeats: {blocked}"

    try:
        asyncio.run(scenario())
        diagnostic = spec2primitives_ui.read_primitive_composition_diagnostic(root)
        assert diagnostic["status"] == ("budget_exhausted" if refinement else "proposed")
        assert len(program_runtime.calls) == 1
        assert diagnostic["candidate"]["primitive_steps"][0]["params"] == {"z": 0.07}
        if refinement:
            request = json.loads(next((root / "composition/refinement_runs").glob("run_*/request.json")).read_text())
            assert request["profile"]["deadline_sec"] == 900
            assert runtime.primitive_refinement_runtime.profile["deadline_sec"] == 300
            if detach == "never":
                assert phase_elements["diagnostic_deadline"].value is False
        assert not (root / "composition/primitive_program_drafts").exists()
    finally:
        if not container.is_deleted:
            container.delete()


@pytest.mark.parametrize(
    "status", ["waiting_for_context", "blocked", "ready_for_composition", "proposed"]
)
def test_primitive_composition_ui_gates_and_displays_candidate(status: str) -> None:
    """Show RA parameters while distinguishing proposals from physical validation."""
    candidate = {
        "primitive_steps": [
            {"primitive_symbol": "move_cartesian", "params": {"z": 0.07}},
        ]
    }
    diagnostic = {
        "status": status,
        "message": "Motion remains unvalidated.",
        "candidate": candidate if status == "proposed" else None,
        "trace": [{"response": "recorded"}] if status == "proposed" else [],
        "attempt_count": 1,
        "composition_input": {
            "primitive_catalog": [
                {
                    "primitive_symbol": "move_cartesian",
                    "typed_parameters": [
                        {"name": name, "required": True} for name in ("x", "y", "z")
                    ],
                }
            ]
        },
    }
    with spec2primitives_ui.ui.column() as container:
        elements = spec2primitives_ui._render_phase_5_diagnostics()
    try:
        spec2primitives_ui._apply_primitive_composition_diagnostic(
            elements,
            diagnostic,
            authoring_available=True,
        )
        assert elements["candidate_status_badge"].text == status
        assert (not elements["create_candidate_button"]._props.get("disable", False)) == (
            status in {"ready_for_composition", "proposed"}
        )
        assert elements["diagnostic_deadline"].value is False
        assert elements["diagnostic_deadline"]._props.get("disable", False) == elements["create_candidate_button"]._props.get("disable", False)
        assert elements["candidate_steps"].visible == (status == "proposed")
        if status == "proposed":
            assert elements["candidate_steps"].content == (
                "1. move_cartesian(z=0.07, x=<unbound>, y=<unbound>)"
            )
            assert json.loads(elements["candidate_trace"].content)["candidate"] == candidate
            assert candidate["primitive_steps"][0]["params"] == {"z": 0.07}
            assert "unvalidated" in elements["candidate_message"].text
        spec2primitives_ui._apply_primitive_composition_diagnostic(
            elements,
            diagnostic,
            authoring_available=True,
            authoring_busy=True,
        )
        assert elements["create_candidate_button"]._props.get("disable") is True
        assert elements["diagnostic_deadline"]._props.get("disable") is True
        assert elements["candidate_status_badge"].text == "composing"
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


@pytest.mark.parametrize(
    "composition_status,gazebo_running,busy,available,enabled",
    [
        ("validated_for_declared_scope", True, False, True, True),
        ("proposed", True, False, True, False),
        ("validated_for_declared_scope", False, False, True, False),
        ("validated_for_declared_scope", True, True, True, False),
        ("validated_for_declared_scope", True, False, False, False),
    ],
)
def test_gazebo_execution_button_requires_validated_program_and_idle_simulation(
    composition_status: str, gazebo_running: bool, busy: bool, available: bool, enabled: bool
) -> None:
    """Expose execution only after the displayed program and simulator are ready."""
    with spec2primitives_ui.ui.column() as container:
        elements = spec2primitives_ui._render_phase_5_diagnostics()
    try:
        spec2primitives_ui._apply_primitive_execution_diagnostic(
            elements,
            {"status": "idle", "message": "Ready"},
            available=available,
            gazebo_running=gazebo_running,
            composition_status=composition_status,
            busy=busy,
        )
        assert elements["run_program_button"].text == "Run in Gazebo"
        assert (not elements["run_program_button"]._props.get("disable", False)) is enabled
        assert not elements["stop_execution_button"].visible
        if composition_status == "validated_for_declared_scope":
            assert "Saved program validated" in elements["execution_message"].text
    finally:
        container.delete()


def test_gazebo_execution_progress_stop_and_reconnect_preserve_composition() -> None:
    """Restored execution progress cannot become composition or duplicate a run."""
    with spec2primitives_ui.ui.column() as container:
        elements = spec2primitives_ui._render_phase_5_diagnostics()
    try:
        elements["candidate_status_badge"].set_text("validated_for_declared_scope")
        elements["execution_candidate_ref"] = (
            "composition/primitive_program_candidates/attempt_0001/candidate.json"
        )
        view = {
            "status": "running",
            "message": "Running step 3 of 10: move_cartesian",
            "step_index": 3,
            "events": [{"status": "running", "step_index": 3}],
            "candidate_ref": {"ref": elements["execution_candidate_ref"], "sha256": "a" * 64},
        }
        spec2primitives_ui._apply_primitive_execution_diagnostic(
            elements,
            view,
            available=True,
            gazebo_running=True,
            composition_status="validated_for_declared_scope",
            busy=True,
        )
        assert elements["stop_execution_button"].visible
        assert elements["execution_message"].text == view["message"]
        assert elements["execution_expansion"].visible
        spec2primitives_ui._apply_primitive_execution_diagnostic(
            elements,
            {**view, "status": "completed", "message": "Commands completed."},
            available=True,
            gazebo_running=True,
            composition_status="validated_for_declared_scope",
        )
        assert not elements["stop_execution_button"].visible
        assert elements["run_program_button"]._props.get("disable", False)
        assert elements["candidate_status_badge"].text == "validated_for_declared_scope"
        spec2primitives_ui._apply_primitive_execution_diagnostic(
            elements,
            {**view, "status": "blocked", "result": {"command_dispatched": False}},
            available=True,
            gazebo_running=True,
            composition_status="validated_for_declared_scope",
        )
        assert not elements["run_program_button"]._props.get("disable", False)
    finally:
        container.delete()


@pytest.mark.parametrize("status", ["unknown", "interrupted", "reset_required", "reset_completed"])
def test_interrupted_gazebo_history_allows_run_without_a_reset_button(status):
    """Let execution own a clean restart instead of exposing a manual reset action."""
    with spec2primitives_ui.ui.column() as container:
        elements = spec2primitives_ui._render_phase_5_diagnostics()
    try:
        assert "reset_execution_button" not in elements
        for _ in range(2):
            spec2primitives_ui._apply_primitive_execution_diagnostic(
                elements, {"status": status, "message": "Reset verification evidence."},
                available=True, gazebo_running=False, composition_status="validated_for_declared_scope",
                busy=False,
            )
            assert not elements["run_program_button"]._props.get("disable", False)
            assert "Run in Gazebo will start a clean shared scene" in elements["execution_message"].text
            assert "Reset verification evidence." in elements["execution_message"].text
    finally:
        container.delete()


@pytest.mark.parametrize("timeout", [False, True])
@pytest.mark.parametrize("previous_status", ["idle", "blocked"])
def test_gazebo_execution_preparation_survives_polling_and_allows_safe_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout: bool, previous_status: str,
) -> None:
    """Show preparation before records exist and retry only an undispatched run."""
    from cais_spade_llm.spec2primitives.adapters.ui_runtime import Spec2PrimitivesUIRuntime
    from cais_spade_llm.spec2primitives.agents.ra import RAContextHandoffError
    from cais_spade_llm.spec2primitives.tests.test_ra_context_handoff import _prepare_composition

    root = tmp_path / "interaction_execution"
    _prepare_composition(root)
    candidate_ref = "composition/primitive_program_candidates/attempt_0001/candidate.json"
    binding_ref = "composition/refinement_runs/run_0001/binding_0001.json"
    composition = {
        "status": "validated_for_declared_scope",
        "latest_candidate_ref": candidate_ref,
        "binding_ref": {"ref": binding_ref, "sha256": "b" * 64},
    }
    diagnostic = {
        "status": previous_status,
        "message": "Validate a program before running it in Gazebo.",
        "candidate_ref": {"ref": candidate_ref, "sha256": "a" * 64},
        "result": {"command_dispatched": False},
    }
    buttons, elements, timers = {}, {}, []
    original_on_click = spec2primitives_ui.ui.button.on_click
    original_render = spec2primitives_ui._render_phase_5_diagnostics

    def on_click(button: Any, callback: Callable) -> Any:
        buttons[button.text] = callback
        return original_on_click(button, callback)

    def render() -> dict[str, Any]:
        elements.update(original_render())
        return elements

    def timer(interval: float, callback: Callable, **kwargs: Any) -> None:
        if not kwargs.get("once", False):
            timers.append(callback)

    monkeypatch.setattr(spec2primitives_ui.ui.button, "on_click", on_click)
    monkeypatch.setattr(spec2primitives_ui.ui, "timer", timer)
    monkeypatch.setattr(spec2primitives_ui, "_render_phase_5_diagnostics", render)
    monkeypatch.setattr(spec2primitives_ui, "read_primitive_composition_diagnostic", lambda path: composition)
    monkeypatch.setattr(spec2primitives_ui, "read_primitive_execution_diagnostic", lambda path: dict(diagnostic))
    monkeypatch.setattr(
        spec2primitives_ui, "read_dual_gazebo_status",
        lambda runtime: SimpleNamespace(state="running", blocked_reason=None),
    )
    with spec2primitives_ui.ui.column() as container:
        pass

    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()
        calls, commands = [], []
        blocker = "Spec2Primitives Dual Gazebo did not become ready within 90 seconds: Waiting for core services: /compute_cartesian_path"

        class Executor:
            async def run(self, path: Path, **kwargs: Any) -> dict[str, Any]:
                assert path == root
                calls.append((kwargs["candidate_ref"], kwargs["binding_ref"]))
                entered.set()
                await release.wait()
                if timeout and len(calls) == 1:
                    raise RAContextHandoffError(blocker)
                commands.append("move_cartesian")
                diagnostic.update(status="completed", message="Commands completed.",
                                  result={"command_dispatched": True})
                await kwargs["progress"](dict(diagnostic))
                return diagnostic

        runtime = Spec2PrimitivesUIRuntime(
            dual_gazebo=object(), product_agent=object(), contexts_root=tmp_path,
            primitive_execution_runtime=Executor(),
        )
        with container:
            spec2primitives_ui._render_pa_interaction(runtime)

        async def click_run() -> None:
            with container:
                await buttons["Run in Gazebo"]()

        task = asyncio.create_task(click_run())
        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
            assert elements["execution_status"].text == "preparing"
            assert "waiting for Gazebo readiness" in elements["execution_message"].text
            await timers[0]()
            assert elements["execution_status"].text == "preparing"
            assert "waiting for Gazebo readiness" in elements["execution_message"].text
            assert elements["run_program_button"]._props["disable"]
            await click_run()
            assert calls == [(candidate_ref, binding_ref)]
            assert commands == []
            release.set()
            await asyncio.wait_for(task, timeout=3)
            if timeout:
                await timers[0]()
                assert elements["execution_status"].text == "blocked"
                assert elements["execution_message"].text == blocker
                assert not elements["stop_execution_button"].visible
                assert not elements["run_program_button"]._props.get("disable", False)
                assert commands == []
                await click_run()
            assert commands == ["move_cartesian"]
            assert elements["execution_status"].text == "completed"
            assert elements["run_program_button"]._props["disable"]
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        container.delete()


def test_gazebo_execution_callbacks_reconnect_and_stop_without_restarting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disconnect the first page during motion, then stop from the recovered page."""
    from types import SimpleNamespace
    from cais_spade_llm.spec2primitives.adapters.ui_runtime import Spec2PrimitivesUIRuntime
    from cais_spade_llm.spec2primitives.agents.ra.program_execution import PrimitiveExecutionRuntime
    from cais_spade_llm.spec2primitives.tests.test_primitive_execution import (
        _SHARE,
        _Transport,
        _validated,
        _validator,
    )

    root = tmp_path / "interaction_execution"
    robot, part, _ = _validated(root)
    callbacks: dict[str, Callable] = {}
    original_on_click = spec2primitives_ui.ui.button.on_click

    def capture_on_click(button: Any, callback: Callable) -> Any:
        callbacks[button.text] = callback
        return original_on_click(button, callback)

    monkeypatch.setattr(spec2primitives_ui.ui.button, "on_click", capture_on_click)
    monkeypatch.setattr(
        spec2primitives_ui,
        "read_dual_gazebo_status",
        lambda runtime: SimpleNamespace(state="running", blocked_reason=None),
    )
    with spec2primitives_ui.ui.column() as page_host:
        pass

    async def scenario() -> None:
        started = asyncio.Event()
        transport = _Transport(part, pause=started)
        executor = PrimitiveExecutionRuntime(
            robot_runtime=robot, validator=_validator, session_factory=transport, share=_SHARE
        )
        runtime = Spec2PrimitivesUIRuntime(
            dual_gazebo=object(),
            product_agent=object(),
            contexts_root=tmp_path,
            primitive_execution_runtime=executor,
        )
        with page_host:
            with spec2primitives_ui.ui.column() as first:
                spec2primitives_ui._render_pa_interaction(runtime)
        running = asyncio.create_task(callbacks["Run in Gazebo"]())
        try:
            await asyncio.wait_for(started.wait(), 15)
            await callbacks["Run in Gazebo"]()
            assert transport.calls == ["move_cartesian"]
            first.delete()
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            assert executor.diagnostic(root)["status"] == "running"
            with page_host:
                with spec2primitives_ui.ui.column() as recovered:
                    spec2primitives_ui._render_pa_interaction(runtime)
            try:
                joined = asyncio.create_task(executor.run(root))
                callbacks["Stop execution"]()
                result = await joined
                assert result["status"] == "stopped"
                assert transport.calls == ["move_cartesian"]
                assert len(list((root / "execution").glob("run_*"))) == 1
            finally:
                recovered.delete()
        finally:
            executor.stop(root)
            if not first.is_deleted:
                first.delete()
            if not running.done():
                await running

    try:
        asyncio.run(scenario())
    finally:
        page_host.delete()


def test_primitive_program_displays_nested_gaps_without_filling_parameters() -> None:
    steps = [
        {
            "primitive_symbol": "compute_place_targets",
            "params": {"product_geometry": {"board_center": {}}},
        }
    ]
    original = json.dumps(steps)
    catalog = [
        {
            "primitive_symbol": "compute_place_targets",
            "typed_parameters": [
                {"name": "product_geometry", "required": False},
                {"name": "pick_ctx", "required": False},
            ],
            "parameter_schemas": {
                "product_geometry": {
                    "type": "object",
                    "x-grounding-fields": ["board_center", "slot_xy"],
                    "properties": {
                        "board_center": {"type": "object", "x-grounding-fields": ["x", "y"]},
                    },
                },
                "pick_ctx": {"type": "object", "x-grounding-required": True},
            },
        }
    ]
    rendered = spec2primitives_ui._format_primitive_program(steps, catalog)
    assert rendered == (
        '1. compute_place_targets(product_geometry={"board_center": {"x": <unbound>, "y": <unbound>}, '
        '"slot_xy": <unbound>}, pick_ctx=<unbound>)'
    )
    assert json.dumps(steps) == original


@pytest.mark.parametrize("cancel", [False, True])
def test_composition_progress_is_visible_before_proposal_and_after_reconnect(
    tmp_path: Path, cancel: bool
) -> None:
    """Restore persisted request progress without duplicating authoring or counting reads as proposals."""
    from cais_spade_llm.spec2primitives.agents.ra.primitive_composition import read_primitive_composition_diagnostic
    from cais_spade_llm.spec2primitives.agents.ra.refinement import (
        PrimitiveRefinementRuntime, cancel_primitive_refinement, load_refinement_profile,
    )
    from cais_spade_llm.spec2primitives.tests.test_primitive_refinement import _setup
    from cais_spade_llm.spec2primitives.tests.test_ra_context_handoff import _program_action

    inputs, _, _, _ = _setup(tmp_path)
    current_ref = inputs.composition_input["target_feature"]["current_state"]["state_values"][0]["value_ref"]["record_ref"]
    waiting = asyncio.Event()
    release = asyncio.Event()
    calls = []
    events = []

    class Program:
        async def author_composition_action(self, *args: Any, **kwargs: Any) -> Any:
            calls.append(kwargs)
            waiting.set()
            await release.wait()
            return {"action": _program_action([("move_cartesian", {"x": 0.2})])}

    class Robot:
        async def capture_validation_context(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("No measured robot context in this UI fixture.")

    async def validator(**kwargs: Any) -> dict[str, Any]:
        return {"scope": read_validation_scope(kwargs["profile"]), "status": "unknown", "findings": [], "calculation_refs": []}

    runtime = PrimitiveRefinementRuntime(
        program_runtime=Program(), robot_runtime=Robot(), validator=validator,
        profile={**load_refinement_profile(), "max_candidates": 1},
    )
    with spec2primitives_ui.ui.column() as container:
        elements = spec2primitives_ui._render_phase_5_diagnostics()
        reconnected = spec2primitives_ui._render_phase_5_diagnostics()
    view = {"candidate": None, "attempt_count": 0, "trace": [], "refinement": {"events": events}}

    async def progress(event: Mapping[str, Any]) -> None:
        events.append(event)
        view.update(status=event.get("status", event["stage"]), message=event["message"])
        if "candidate" in event:
            view.update(candidate=event["candidate"], attempt_count=event["candidate_count"])
        spec2primitives_ui._apply_primitive_composition_diagnostic(elements, view)

    async def scenario() -> None:
        first = asyncio.create_task(runtime.compose(tmp_path, progress=progress))
        second = None
        try:
            await asyncio.wait_for(waiting.wait(), timeout=10)
            assert elements["candidate_attempts"].text == "Attempts: 0"
            assert "RA is authoring the primitive program" in elements["candidate_message"].text
            assert elements["candidate_progress"].visible
            assert elements["candidate_progress"].text == f"Elapsed at last update: {events[-1]['elapsed_sec']:.1f} s"
            assert f"Prompt: {len(calls[-1]['prompt'])} characters." in elements["candidate_message"].text
            restored = await asyncio.to_thread(read_primitive_composition_diagnostic, tmp_path)
            spec2primitives_ui._apply_primitive_composition_diagnostic(reconnected, restored)
            assert reconnected["candidate_message"].text == elements["candidate_message"].text
            assert reconnected["candidate_progress"].text == elements["candidate_progress"].text
            assert reconnected["candidate_attempts"].text == "Attempts: 0"
            second = asyncio.create_task(runtime.compose(tmp_path))
            await asyncio.sleep(0)
            assert len(calls) == 1
            if cancel:
                assert cancel_primitive_refinement(tmp_path)
            else:
                release.set()
            outcomes = await asyncio.gather(first, second)
            assert all(outcome["status"] == ("cancelled" if cancel else "needs_context") for outcome in outcomes)
            assert elements["candidate_attempts"].text == ("Attempts: 0" if cancel else "Attempts: 1")
            assert len(calls) == 1
            assert len(list((tmp_path / "composition/refinement_runs").glob("run_*"))) == 1
        finally:
            release.set()
            await first
            if second is not None:
                await second

    try:
        asyncio.run(scenario())
    finally:
        container.delete()





@pytest.mark.parametrize("outcome", ["validated", "cancelled", "deadline", "calculation_cancelled"])
def test_direct_answers_restore_numerical_bindings_and_pending_calculations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    from cais_spade_llm.spec2primitives.agents.ra.refinement import PrimitiveRefinementRuntime, cancel_primitive_refinement, load_refinement_profile
    from cais_spade_llm.spec2primitives.agents.ra.primitive_composition import read_primitive_composition_diagnostic
    from cais_spade_llm.spec2primitives.agents.ra.program_validation import validate_program
    from cais_spade_llm.spec2primitives.tests.test_primitive_refinement import _binding_fixture, _MessageProgramRuntime, _observed_program, _PlanningSession
    from cais_spade_llm.spec2primitives.tests.test_ra_context_handoff import _program_action
    _, robot, refs, product, _ = _binding_fixture(tmp_path, monkeypatch)
    program = _MessageProgramRuntime([_program_action([(s["primitive_symbol"], s["params"]) for s in _observed_program(refs, bound=False)])])
    waiting, release = asyncio.Event(), asyncio.Event()
    captures = []
    class Robot:
        async def capture_validation_context(self, *args: Any, **kwargs: Any) -> Any:
            captures.append(True)
            return {**robot, "captured_at_ns": time.time_ns()}
    async def validator(**kwargs: Any) -> Any:
        return await validate_program(**kwargs, session_factory=_PlanningSession)
    runtime = PrimitiveRefinementRuntime(program_runtime=program, robot_runtime=Robot(), product_runtime=product, validator=validator,
        profile={**load_refinement_profile(), "deadline_sec": 2 if outcome == "deadline" else 300})
    async def progress(event: Any) -> None:
        if ((outcome == "calculation_cancelled" and "calculation_ref" in event)
                or (outcome != "calculation_cancelled" and event["stage"] == "binding")):
            waiting.set()
            await release.wait()
    with spec2primitives_ui.ui.column() as container:
        elements = spec2primitives_ui._render_phase_5_diagnostics()
    async def scenario() -> None:
        task = asyncio.create_task(runtime.compose(tmp_path, progress=progress))
        try:
            await asyncio.wait_for(waiting.wait(), timeout=10)
            restored = await asyncio.to_thread(read_primitive_composition_diagnostic, tmp_path)
            spec2primitives_ui._apply_primitive_composition_diagnostic(elements, restored)
            assert elements["candidate_attempts"].text == "Attempts: 1"
            assert '"part_height_m": 0.02' in elements["candidate_steps"].content
            assert ("<pending: step 1" in elements["candidate_steps"].content) is (outcome != "calculation_cancelled")
            assert "<pending: step 6" in elements["candidate_steps"].content
            assert "<unbound>" not in elements["candidate_steps"].content
            assert elements["execution_binding_ref"] == restored["binding_ref"]["ref"]
            assert "value_ref" in elements["candidate_trace"].content
            assert len(captures) == (1 if outcome == "calculation_cancelled" else 0)
            if outcome in {"cancelled", "calculation_cancelled"}:
                assert cancel_primitive_refinement(tmp_path)
            elif outcome == "validated":
                release.set()
            result = await asyncio.wait_for(task, timeout=10)
            assert result["status"] == {"validated": "validated_for_declared_scope", "cancelled": "cancelled", "calculation_cancelled": "cancelled", "deadline": "budget_exhausted"}[outcome]
            final = read_primitive_composition_diagnostic(tmp_path)
            spec2primitives_ui._apply_primitive_composition_diagnostic(elements, final)
            assert len(program.calls) == 1 and product.product_agent.model_calls == 0
            if outcome == "validated":
                assert "<pending:" not in elements["candidate_steps"].content
                assert elements["execution_binding_ref"] == final["validation"]["binding_ref"]["ref"]
            else:
                assert "<pending:" in elements["candidate_steps"].content
                assert len(captures) == (1 if outcome == "calculation_cancelled" else 0)
                if outcome == "calculation_cancelled":
                    assert "<pending: step 1" not in elements["candidate_steps"].content
                    assert any("calculation_ref" in event for event in final["refinement"]["events"])
                assert "no completed validation" in elements["candidate_bindings"].text
            if outcome == "deadline":
                assert "during binding" in elements["candidate_message"].text
        finally:
            release.set()
            await task
    try:
        asyncio.run(scenario())
    finally:
        container.delete()



def test_pick_place_scope_wording_survives_ui_restoration() -> None:
    """Reopened composition and execution retain the simulation claim and real missing roles."""
    with spec2primitives_ui.ui.column() as container:
        first = spec2primitives_ui._render_phase_5_diagnostics()
        restored = spec2primitives_ui._render_phase_5_diagnostics()
    try:
        validation = {
            "scope": GAZEBO_PICK_PLACE_SCOPE, "status": "unknown",
            "findings": [{"step_index": None, "check": role, "status": "unknown", "authority": "PA",
                          "message": f"Required {role} evidence is missing: observed geometry is required."}
                         for role in ("part", "scene")],
        }
        message = spec2primitives_ui._format_validation_findings(validation, None)
        assert message == "Pick-and-place validation incomplete: part, scene."
        view = {
            "status": "validated_for_declared_scope", "validation_scope": GAZEBO_PICK_PLACE_SCOPE,
            "message": "Validated for Gazebo pick-and-place. No motion was executed.",
            "attempt_count": 2, "candidate": None, "trace": [],
            "validation": {"scope": GAZEBO_PICK_PLACE_SCOPE, "status": "passed", "findings": []},
        }
        for elements in (first, restored):
            spec2primitives_ui._apply_primitive_composition_diagnostic(elements, view)
            assert elements["candidate_message"].text == view["message"]
            assert elements["candidate_attempts"].text == "Attempts: 2"
            spec2primitives_ui._apply_primitive_execution_diagnostic(
                elements,
                {"status": "completed", "message": "Pick-and-place completed.",
                 "result": {"assembly_success": None}},
                available=True, gazebo_running=True, composition_status=view["status"],
            )
            assert elements["execution_message"].text == "Pick-and-place completed."
            assert elements["candidate_message"].text == view["message"]
    finally:
        container.delete()


@pytest.mark.parametrize("deadline", [300, 900])
def test_saved_deadline_is_separate_from_next_run_option_and_progress_is_lightweight(deadline: int) -> None:
    """Restore the actual request deadline without rebuilding the trace for every wait."""
    from cais_spade_llm.spec2primitives.agents.ra.validation_scope import GAZEBO_OBSERVED_SCOPE

    events = [{"stage": "composing", "message": "Waiting for RA", "elapsed_sec": 2.0, "deadline_sec": deadline}]
    view = {"status": "composing", "message": "Waiting for RA", "attempt_count": 0,
            "refinement": {"events": events, "request": {"profile": {"deadline_sec": deadline}}},
            "validation": {"scope": GAZEBO_OBSERVED_SCOPE, "status": "unknown", "findings": [
                {"check": "scene", "status": "unknown", "authority": "PA", "message": "Required scene evidence is missing: observed coverage is needed."},
            ]}}
    with spec2primitives_ui.ui.column() as container:
        elements = spec2primitives_ui._render_phase_5_diagnostics()
    try:
        elements["diagnostic_deadline"].set_value(False)
        spec2primitives_ui._apply_primitive_composition_diagnostic(elements, view)
        assert elements["candidate_deadline"].text == f"This run's deadline: {deadline} seconds"
        assert elements["diagnostic_deadline"].value is False
        assert elements["candidate_bindings"].text.startswith("Pick-and-place validation incomplete")
        trace = elements["candidate_trace"].content
        events.append({"stage": "evidence", "message": "PA measurement completed", "elapsed_sec": 8.0})
        view["message"] = events[-1]["message"]
        spec2primitives_ui._apply_primitive_composition_diagnostic(elements, view, progress_only=True)
        assert elements["candidate_message"].text == "PA measurement completed"
        assert elements["candidate_progress"].text == "Elapsed at last update: 8.0 s"
        assert elements["candidate_attempts"].text == "Attempts: 0"
        assert elements["candidate_trace"].content == trace
        spec2primitives_ui._apply_primitive_composition_diagnostic(elements, view)
        assert json.loads(elements["candidate_trace"].content)["refinement"]["events"] == events
    finally:
        container.delete()


@pytest.mark.parametrize("status", ["invalid", "unsupported", "cancelled", "proposal", "composing"])
def test_rejected_proposal_does_not_display_validation_as_pending(status: str) -> None:
    """Keep the invalid reference visible while distinguishing stopped and active runs."""
    candidate = {"status": "invalid" if status == "invalid" else "proposed", "primitive_steps": [
        {"primitive_symbol": "compute_pick_targets", "params": {"part_name": "medium gear"}},
        {"primitive_symbol": "move_cartesian", "params": {
            "x": {"result_ref": {"step_index": 1, "field_path": "/declared_output/approach_pose/x"}},
        }},
    ]}
    message = "result_ref selects an undeclared result field."
    view = {"status": status, "message": message, "candidate": candidate, "attempt_count": 0,
            "refinement": {"events": [{"stage": "composing", "message": "RA proposal received."}]}}
    with spec2primitives_ui.ui.column() as container:
        elements = spec2primitives_ui._render_phase_5_diagnostics()
    try:
        spec2primitives_ui._apply_primitive_composition_diagnostic(elements, view, authoring_available=True)
        summary = elements["candidate_bindings"].text
        assert "This proposal has no completed validation result." in summary
        ended = status in {"invalid", "unsupported", "cancelled"}
        assert ("The run ended before validation completed." in summary) is ended
        assert ("Validation is pending." in summary) is not ended
        assert elements["candidate_message"].text == message
        assert "/declared_output/approach_pose/x" in elements["candidate_steps"].content
        assert json.loads(elements["candidate_trace"].content)["candidate"] == candidate
    finally:
        container.delete()


def test_primitive_binding_summary_is_short_and_full_report_remains_expandable() -> None:
    issues = [
        {
            "step_index": n,
            "parameter_path": "/product_geometry",
            "status": "missing",
            "message": "Required input is unbound.",
        }
        for n in range(1, 6)
    ]
    issues.append(
        {
            "step_index": 6,
            "parameter_path": "/results",
            "status": "deferred",
            "message": "Not executed.",
        }
    )
    with spec2primitives_ui.ui.column() as container:
        elements = spec2primitives_ui._render_phase_5_diagnostics()
    try:
        spec2primitives_ui._apply_primitive_composition_diagnostic(
            elements,
            {
                "status": "proposed",
                "message": "Unvalidated proposal.",
                "candidate": {"primitive_steps": []},
                "binding_issues": issues,
                "composition_input": {"primitive_catalog": []},
                "trace": [],
            },
            authoring_available=True,
        )
        text = elements["candidate_bindings"].text
        assert elements["candidate_bindings"].visible
        assert "Deferred results: 1" in text
        assert "Step 3 /product_geometry" in text and "Step 4 /product_geometry" not in text
        assert len(text.splitlines()) == 5
        assert json.loads(elements["candidate_trace"].content)["binding_issues"] == issues
    finally:
        container.delete()


@pytest.mark.parametrize("complete_validation", [False, True])
def test_revised_proposal_does_not_inherit_previous_validation_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete_validation: bool,
) -> None:
    """Live and restored views keep the first report separate while revision checks wait."""
    from copy import deepcopy

    from cais_spade_llm.spec2primitives.adapters.ui_runtime import Spec2PrimitivesUIRuntime
    from cais_spade_llm.spec2primitives.agents.ra.refinement import (
        PrimitiveRefinementRuntime, cancel_primitive_refinement, load_refinement_profile,
    )
    from cais_spade_llm.spec2primitives.tests.test_primitive_refinement import (
        _observed_program, _binding_fixture, _MessageProgramRuntime,
    )
    from cais_spade_llm.spec2primitives.tests.test_ra_context_handoff import _program_action

    root = tmp_path / "interaction_composition"
    _, robot, refs, product, _ = _binding_fixture(root, monkeypatch)
    old_message = "An intermediate motion needs revision."
    current_message = "The revised motion could not be fully checked."
    revised = _observed_program(refs)
    program = _MessageProgramRuntime([
        _program_action([(step["primitive_symbol"], step["params"]) for step in _observed_program(refs, bound=False)]),
        _program_action([(step["primitive_symbol"], step["params"]) for step in revised]),
    ])
    waiting, release = asyncio.Event(), asyncio.Event()
    validations = []

    class Robot:
        captures = 0

        async def capture_validation_context(self, *args: Any, **kwargs: Any) -> Any:
            self.captures += 1
            if self.captures == 2:
                waiting.set()
                await release.wait()
            return {**deepcopy(robot), "captured_at_ns": time.time_ns()}

    async def validator(**kwargs: Any) -> Any:
        validations.append(deepcopy(kwargs["steps"]))
        from cais_spade_llm.spec2primitives.agents.ra.program_validation import _report
        return _report(kwargs["steps"], [{"step_index": 1, "check": "motion", "status": "failed", "authority": "RA",
                                         "message": old_message if len(validations) == 1 else current_message}], [], [], None,
                       scope=read_validation_scope(kwargs["profile"]))

    runtime = Spec2PrimitivesUIRuntime(
        dual_gazebo=object(), product_agent=object(), contexts_root=tmp_path,
        robot_agent_context_runtime=object(), robot_agent_program_runtime=program,
        primitive_refinement_runtime=PrimitiveRefinementRuntime(
            program_runtime=program, robot_runtime=Robot(), product_runtime=product, validator=validator,
            profile={**load_refinement_profile(), "max_candidates": 2},
        ),
    )
    buttons, elements = {}, {}
    original_on_click = spec2primitives_ui.ui.button.on_click
    original_render = spec2primitives_ui._render_phase_5_diagnostics

    def on_click(button: Any, callback: Callable) -> Any:
        buttons[button.text] = callback
        return original_on_click(button, callback)

    def render() -> dict[str, Any]:
        elements.update(original_render())
        return elements

    monkeypatch.setattr(spec2primitives_ui.ui.button, "on_click", on_click)
    monkeypatch.setattr(spec2primitives_ui, "_render_phase_5_diagnostics", render)
    with spec2primitives_ui.ui.column() as container:
        spec2primitives_ui._render_pa_interaction(runtime)

    def check_latest(*, finished: bool) -> None:
        assert elements["candidate_attempts"].text == "Attempts: 2"
        summary = elements["candidate_bindings"].text
        assert old_message not in summary
        if finished and complete_validation:
            assert current_message in summary
            assert "no completed validation" not in summary
        else:
            assert "This proposal has no completed validation result." in summary
            assert ("The run ended before validation completed." in summary) is finished
        trace = json.loads(elements["candidate_trace"].content)
        reports = [event["validation"] for event in trace["refinement"]["events"]
                   if event["stage"] == "validation_result"]
        assert reports[0]["findings"][0]["message"] == old_message
        assert trace["candidate"]["primitive_steps"] == revised

    async def scenario() -> None:
        task = asyncio.create_task(buttons["Compose Primitive Program"]())
        try:
            await asyncio.wait_for(waiting.wait(), timeout=20)
            check_latest(finished=False)
            restored = await asyncio.to_thread(spec2primitives_ui.read_primitive_composition_diagnostic, root)
            assert "validation" not in restored
            spec2primitives_ui._apply_primitive_composition_diagnostic(elements, restored)
            check_latest(finished=False)
            if complete_validation:
                release.set()
            else:
                assert cancel_primitive_refinement(root)
            await task
            check_latest(finished=True)
            restored = await asyncio.to_thread(spec2primitives_ui.read_primitive_composition_diagnostic, root)
            assert restored["attempt_count"] == 2
            if complete_validation:
                assert restored["validation"]["candidate_ref"]["ref"] == restored["latest_candidate_ref"]
            else:
                assert restored["status"] == "cancelled" and "validation" not in restored
                # Deadline termination has the same unvalidated revision and report history.
                restored.update(status="budget_exhausted", message="The configured refinement deadline (300 seconds) was reached.")
            spec2primitives_ui._apply_primitive_composition_diagnostic(elements, restored)
            check_latest(finished=True)
        finally:
            release.set()
            await task

    try:
        asyncio.run(scenario())
    finally:
        container.delete()


def test_repeated_unbound_findings_leave_destination_blocker_visible() -> None:
    """Repeated placement coordinates leave room for run_0008's recorded cause."""
    from copy import deepcopy

    blocker = "Grounding must establish one accepted destination feature for 'medium gear'."
    unresolved = [
        f"step_index 6, quantity /product_geometry/placement_surface_point/{axis}: blocked: {blocker}"
        for axis in ("x", "y", "z")
    ]
    validation = {
        "status": "unknown",
        "findings": [
            *[{"step_index": 6, "check": f"/product_geometry/placement_surface_point/{axis}",
               "status": "unknown", "authority": "PA", "message": "Required input is unbound."}
              for axis in ("x", "y", "z")],
            {"step_index": None, "check": "scene", "status": "unknown", "authority": "PA",
             "message": "Required scene evidence is missing."},
            *[{"step_index": None, "status": "unknown", "authority": "PA", "message": message}
              for message in unresolved],
        ],
    }
    original = deepcopy(validation)
    text = spec2primitives_ui._format_validation_findings(validation, {"unresolved": unresolved})
    assert text.splitlines() == [
        "Step 6: Required input is unbound.",
        "Required scene evidence is missing.",
        unresolved[0],
        "Open the program records for all findings.",
    ]
    assert validation == original


def test_current_validation_blockers_precede_older_pa_explanations() -> None:
    """A newer calculation failure must not be hidden by earlier missing-height feedback."""
    current = "The selected target calculation cannot resolve the supplied tool transform."
    old = "Earlier investigation did not derive part height."
    validation = {
        "status": "unknown",
        "findings": [
            {"step_index": None, "check": "part", "status": "unknown", "authority": "PA",
             "message": "Required part evidence is missing: observed geometry is needed."},
            {"step_index": 1, "check": "part_height_m", "status": "warning",
             "message": "Height is an estimate."},
            {"step_index": 1, "check": "calculation", "status": "unknown", "message": current},
            {"step_index": 2, "check": "bindings", "status": "unknown",
             "message": "Selected result from step 1 is not available."},
        ],
    }
    text = spec2primitives_ui._format_validation_findings(validation, {"unresolved": [old]})
    assert text.index(current) < text.index(old)
    assert "Open the program records for all findings." in text
    assert "Selected result from step 1" not in text
    assert len(validation["findings"]) == 4


def test_fitting_failure_remains_visible_once_with_calculated_coordinates() -> None:
    from copy import deepcopy

    failure = ("The checked through-bore radius (0.004987318 m) does not clear the shaft envelope "
               "(0.005006046 m). Entry chamfers do not establish the through-bore clearance.")
    validation = {"status": "failed", "scope": spec2primitives_ui.GAZEBO_OBSERVED_SCOPE, "findings": [
        *[{"step_index": index, "status": "unknown", "authority": "PA", "pa_response_ref": {"ref": "pa_0001/response.json"},
           "message": "not_investigated: no checked answer was selected."} for index in (1, 6, 2)],
        *[{"step_index": None, "check": "mating_geometry", "status": "failed", "authority": "PA", "message": failure}] * 2,
        {"step_index": 9, "check": "motion", "status": "failed", "message": "Insertion collides with the shaft."},
        {"step_index": None, "check": "assembly_outcome", "status": "unknown", "message": "The complete predicted shaft fitting could not be established."},
    ]}
    previous = {"unresolved": ["Earlier investigation did not derive part height."]}
    original = deepcopy(validation)
    text = spec2primitives_ui._format_validation_findings(validation, previous)
    assert text.count(failure) == 1
    assert text.index(failure) < text.index("Insertion collides") < text.index("Earlier investigation")
    assert "Open the program records for all findings." in text
    assert validation == original
    steps = [{"primitive_symbol": "move_cartesian", "params": {"x": .4004289837365861, "y": -.29956979509443044, "z": 1.4}}]
    rendered = spec2primitives_ui._format_primitive_program(steps, [], pending_results=True)
    assert "x=0.4004289837365861" in rendered and "<unbound>" not in rendered and "<pending:" not in rendered
    missing = {"scope": spec2primitives_ui.GAZEBO_LINK_ATTACHER_SCOPE, "status": "unknown", "findings": [
        {"check": "goal", "status": "unknown", "authority": "PA", "message": "PA has not supplied the required goal evidence."},
    ]}
    assert spec2primitives_ui._format_validation_findings(missing, None) == "Simulation validation incomplete: goal."


def test_unsupported_geometry_is_visible_once_before_older_measurement_requests() -> None:
    from copy import deepcopy

    failure = "The approved CAD has no supported pair of coaxial circular end faces."
    validation = {"status": "unknown", "scope": spec2primitives_ui.GAZEBO_OBSERVED_SCOPE, "findings": [
        *[{"step_index": None, "check": "coverage", "status": "unknown", "message": failure}] * 2,
        {"step_index": 6, "check": "bindings", "status": "unknown", "message": "Checked supported goal geometry is required for assembly targets."},
        {"step_index": None, "check": "assembly_outcome", "status": "unknown", "message": "The complete predicted assembly could not be established."},
    ]}
    original = deepcopy(validation)
    text = spec2primitives_ui._format_validation_findings(validation, {"unresolved": ["Earlier request for shaft measurements."]})
    assert text.count(failure) == 1 and text.index(failure) < text.index("Earlier request")
    assert validation == original and len(validation["findings"]) == 4


@pytest.mark.parametrize("robot_change", [False, True])
def test_robot_revision_feedback_is_visible_after_refresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, robot_change: bool) -> None:
    from copy import deepcopy
    from cais_spade_llm.spec2primitives.agents.ra.refinement import PrimitiveRefinementRuntime
    from cais_spade_llm.spec2primitives.agents.ra.program_validation import _report
    from cais_spade_llm.spec2primitives.tests.test_primitive_refinement import _binding_fixture, _MessageProgramRuntime, _observed_program
    from cais_spade_llm.spec2primitives.tests.test_ra_context_handoff import _program_action
    _, robot, refs, product, _ = _binding_fixture(tmp_path, monkeypatch)
    steps = _observed_program(refs, bound=False)
    program = _MessageProgramRuntime([_program_action([(s["primitive_symbol"], s["params"]) for s in steps])])
    class Robot:
        captures = 0
        async def capture_validation_context(self, *args: Any, **kwargs: Any) -> Any:
            self.captures += 1
            context = {**deepcopy(robot), "captured_at_ns": time.time_ns()}
            if robot_change and self.captures == 2:
                context["joint_state"]["positions"][0] = 0.004
            return context
    message = "The intermediate motion needs revision."
    async def validator(**kwargs: Any) -> Any:
        return _report(kwargs["steps"], [{"step_index": 2, "check": "motion", "status": "failed", "authority": "RA", "message": message}], [], [], None,
                       scope=read_validation_scope(kwargs["profile"]))
    result = asyncio.run(PrimitiveRefinementRuntime(program_runtime=program, robot_runtime=Robot(), product_runtime=product, validator=validator).compose(tmp_path))
    assert result["status"] == ("stale" if robot_change else "no_progress"), result
    assert len(program.calls) == 2
    assert "EXCHANGES" not in program.calls[1]["prompt"] and len(program.calls[1]["prompt"]) <= 32000
    with spec2primitives_ui.ui.column() as container:
        elements = spec2primitives_ui._render_phase_5_diagnostics()
    try:
        diagnostic = spec2primitives_ui.read_primitive_composition_diagnostic(tmp_path)
        spec2primitives_ui._apply_primitive_composition_diagnostic(elements, diagnostic)
        assert elements["candidate_status_badge"].text == result["status"]
        if robot_change:
            comparison = next(event for event in diagnostic["refinement"]["events"] if "differences" in event)
            assert "joint_state.positions['joint1']" in elements["candidate_message"].text
            assert "threshold 0.001" in comparison["message"]
            assert comparison["previous_robot_context_ref"]["ref"].endswith("robot_context_0001.json")
            assert comparison["robot_context_ref"]["ref"].endswith("robot_context_0002.json")
        else:
            assert message in elements["candidate_bindings"].text
    finally:
        container.delete()



@pytest.mark.parametrize("historical", [False, True])
def test_primitive_program_uses_attempt_catalog_for_execution_bindings(historical: bool) -> None:
    """New calls omit simulator fields; an older request keeps its recorded interface."""
    from cais_spade_llm.spec2primitives.agents.ra.composition_context import _composition_catalog_view

    entry = {
        "primitive_symbol": "grasp_part",
        "typed_parameters": [
            {"name": "model_name", "type": "string", "required": True},
            {"name": "part_name", "type": "string", "required": False},
        ],
        "parameter_schemas": {"model_name": {"type": "string"}, "part_name": {"type": "string"}},
        "conditions": {"held_part": {"equals": None}},
        "effects": {"held_part": {"set_from_param_any_of": ["part_name", "model_name"]}},
    }
    catalog = [entry] if historical else _composition_catalog_view((entry,))
    candidate = {
        "primitive_steps": [{"primitive_symbol": "grasp_part", "params": {"part_name": "medium gear"}}]
    }
    original = json.dumps(candidate)
    with spec2primitives_ui.ui.column() as container:
        elements = spec2primitives_ui._render_phase_5_diagnostics()
    try:
        spec2primitives_ui._apply_primitive_composition_diagnostic(
            elements,
            {
                "status": "proposed",
                "message": "Unvalidated proposal.",
                "candidate": candidate,
                "composition_input": {"primitive_catalog": catalog},
                "trace": [],
            },
            authoring_available=True,
        )
        suffix = ", model_name=<unbound>" if historical else ""
        assert elements["candidate_steps"].content == f'1. grasp_part(part_name="medium gear"{suffix})'
        assert json.dumps(candidate) == original
        assert json.loads(elements["candidate_trace"].content)["candidate"] == candidate
    finally:
        container.delete()

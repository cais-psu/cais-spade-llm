from __future__ import annotations

"""Provider transport parity, immutable attempts, tool rounds, and UI capture."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import jsonschema
import pytest

from cais_spade_llm.agents.shared_information import llm_agent
from cais_spade_llm.agents.shared_information.llm_request_records import capture_requests
from cais_spade_llm.ui.recovery_evidence import read_stage_record


def _response(content, tools=None):
    return SimpleNamespace(
        id="response-1",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tools),
            )
        ],
    )


def _agent():
    agent = object.__new__(llm_agent.LlmAgent)
    agent.model, agent.reasoning_effort, agent.instructions = "test-model", "medium", "instructions"
    return agent


def test_single_attempt_disables_sdk_and_application_retries(tmp_path, monkeypatch):
    create = Mock(side_effect=RuntimeError("transport unavailable"))
    limited = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    options = Mock(return_value=limited)
    monkeypatch.setattr(llm_agent, "_client", SimpleNamespace(with_options=options))
    monkeypatch.setattr(llm_agent.time, "sleep", lambda _: None)
    async def run():
        with capture_requests(directory=tmp_path, run_id="draft", stage="outline_draft", turn=1):
            with pytest.raises(RuntimeError, match="transport unavailable"):
                await _agent().ask_llm_structured("draft", response_format={"name": "draft", "schema": {}},
                    include_agent_instructions=False, max_tool_rounds=0, max_attempts=1)
    asyncio.run(run())
    options.assert_called_once_with(max_retries=0)
    create.assert_called_once()
    assert len(list(tmp_path.glob("requests/*/*_request.json"))) == 1


def test_primitive_subturn_preserves_records_without_copying_the_saved_scene(tmp_path):
    from pathlib import Path
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import _write_primitive_subturn_artifact

    class RecordedScene(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("A retrieval record must not duplicate the saved scene")

    scene = RecordedScene({"resource_models": {"owner": {"source": "recorded"}}})
    prepared = {"recovery_debug": {"per_turn_debug_dir": str(tmp_path)}, "physical_evidence": scene}
    event = {"outline_id": "RecordedEvent", "resource_jid": "owner@localhost"}
    response = {"decision": "need_context", "context_requests": ["/primitive_contracts"]}
    turn = {**event, "turn_index": 1, "primitive_local_turn_index": 1,
            "decision": "need_context", "prompt_text": "Exact primitive request",
            "llm_raw_response": response, "raw_response": response}
    paths = _write_primitive_subturn_artifact(prepared_recovery_request=prepared,
        session_state={"turn_index": 1}, assigned_outline_events=[event],
        recovery_session_id="recorded-session", turn_entry=turn)
    assert "Exact primitive request" in Path(paths["prompt_artifact_path"]).read_text()
    saved = json.loads(Path(paths["response_artifact_path"]).read_text())
    assert saved["response"] == response
    assert prepared["physical_evidence"] is scene
    assert scene == {"resource_models": {"owner": {"source": "recorded"}}}


@pytest.mark.parametrize("cancel", [False, True])
def test_composition_records_retrieval_rejection_revision_and_cancellation(tmp_path, monkeypatch, cancel):
    import threading
    from test_recovery_composition import contract, _prepared, feasible
    from cais_spade_llm.resources.recovery_composition import validate_recovery_primitive_composition
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import generate_primitive_batch_with_llm_agent

    case = contract.__wrapped__()
    prepared = _prepared(case)
    prepared["recovery_debug"] = {"artifact_directory": str(tmp_path), "per_turn_debug_dir": str(tmp_path)}
    steps = deepcopy(case["primitive_steps"])
    steps[1]["params"]["count"] = 3
    event = case["task"]
    reply = {"thought": "Contract fixture", "outline_id": event["outline_id"], "resource_jid": event["resource_jid"]}
    replies = [{**reply, "decision": "primitive_steps_ready", "primitive_steps": list(reversed(steps))},
               {**reply, "decision": "need_context", "context_requests": ["/resources/owner/recovery_des_model"]},
               {**reply, "decision": "primitive_steps_ready", "primitive_steps": steps}]
    agent = _agent()
    agent._composition_cancel_event = threading.Event()
    calls = []
    def create(**kwargs):
        calls.append(deepcopy(kwargs))
        if cancel:
            agent._composition_cancel_event.set()
        return _response(json.dumps(replies[len(calls) - 1]))
    monkeypatch.setattr(llm_agent, "_client", SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    def validate(**kwargs):
        kwargs.pop("prepared")
        return validate_recovery_primitive_composition(**kwargs, recovery_des_model=case["recovery_des_model"],
            primitive_catalog=case["primitive_catalog"], validate_primitive=feasible)
    run = generate_primitive_batch_with_llm_agent(llm_agent=agent,
        composition_owner=SimpleNamespace(validate_recovery_primitive_composition=validate),
        prepared_recovery_request=prepared, assigned_outline_events=[event], recovery_session_id="composition-test", max_turns=3)
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run)
        assert len(calls) == 1
    else:
        result = asyncio.run(run)
        assert result["decision"] == "draft_ready", result
        assert len(calls) == 3
        assert result["primitive_events"][0]["composition_validation"]["feasibility_status"] == "FEASIBLE"
        assert "preconditions" in calls[1]["messages"][0]["content"]
    requests = list(tmp_path.rglob("*_request.json"))
    provider = [json.loads(path.read_text()) for path in requests if '"kind": "provider_request"' in path.read_text()]
    assert len(provider) == len(calls)
    assert all(record["payload"] in calls for record in provider)


@pytest.mark.parametrize("mode", ["single_pass", "incremental", "incremental_validated", "incremental_candidates_validated"])
@pytest.mark.parametrize("action_horizon", ["1", "k", "full"])
def test_outline_schema_at_provider_boundary_accepts_retrieval(tmp_path, monkeypatch, mode, action_horizon):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        multi_turn_phase_response_schema,
    )

    calls = []
    reply = {"thought": "Read recorded part facts", "decision": "need_context", "context_requests": ["/parts"]}
    def create(**kwargs):
        calls.append(deepcopy(kwargs))
        response_format = kwargs["response_format"]["json_schema"]
        body = response_format["schema"]
        assert body["type"] == "object"
        assert not {"anyOf", "oneOf", "allOf", "enum", "const", "not"} & body.keys()
        assert response_format["strict"] is False
        return _response(json.dumps(reply))

    monkeypatch.setattr(llm_agent, "_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    schema = multi_turn_phase_response_schema("outline", outline_mode=mode,
        outline_context_retrieval=True, action_horizon=action_horizon)
    with capture_requests(directory=tmp_path, run_id="schema-regression", stage="outline", turn=2):
        response = asyncio.run(_agent().ask_llm_structured("Read recorded facts", response_format=schema))
    assert response == reply
    capture = json.loads(next(tmp_path.glob("requests/*/*_request.json")).read_text())
    assert capture["payload"] == calls[0]
    raw = json.loads(next(tmp_path.glob("requests/*/*_response.json")).read_text())
    assert json.loads(raw["raw_response"]["choices"][0]["message"]["content"]) == reply


@pytest.mark.parametrize("part_name", [None, "RGOCG4-50_Round_4mm"])
@pytest.mark.parametrize("include_active_job", [False, True])
def test_outline_provider_schema_keeps_declared_null_and_omission(
    tmp_path, monkeypatch, part_name, include_active_job,
):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import multi_turn

    jid = "recovery-resource-6@localhost"
    declaration = {"scope": "resource", "type": ["string", "null"],
                   "domain": ["environment_23"], "domain_complete": False}
    session = {"validation_contract_version": 2, "outline_mode": "incremental_validated",
               "outline_context_retrieval": True, "recovery_des_models": {jid: {"state_variables": {
                   "active_job": declaration,
                   "job_state": {"scope": "resource", "domain": ["running", "paused"]}}}}}
    before = deepcopy(session)
    end_state = {"job_state": "paused"}
    if include_active_job:
        end_state["active_job"] = None
    reply = {"thought": "Recorded declaration fixture", "next_transition": {
        "outline_id": "local_fixture", "event_name": "pause_job", "resource_jid": jid,
        "part_name": part_name, "expected_end_state": end_state, "rationale": "Local schema check"}}
    original_reply = deepcopy(reply)
    schema = multi_turn._get_response_schema("outline", session)
    calls = []

    def create(**kwargs):
        calls.append(deepcopy(kwargs))
        actual_schema = kwargs["response_format"]["json_schema"]
        assert actual_schema == schema
        assert actual_schema["strict"] is False
        assert actual_schema["schema"]["type"] == "object"
        assert not {"anyOf", "oneOf", "allOf", "enum", "const", "not"} & actual_schema["schema"].keys()
        jsonschema.validate(reply, actual_schema["schema"])
        jsonschema.validate({"thought": "Read recorded facts", "decision": "need_context",
                             "context_requests": [f"/resources/{jid}/state_variables"]}, actual_schema["schema"])
        return _response(json.dumps(reply))

    monkeypatch.setattr(llm_agent, "_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    with capture_requests(directory=tmp_path, run_id="nullable-field-regression", stage="outline", turn=2):
        response = asyncio.run(_agent().ask_llm_structured("Use recorded declarations", response_format=schema))
    request = json.loads(next(tmp_path.glob("requests/*/*_request.json")).read_text())
    captured_response = json.loads(next(tmp_path.glob("requests/*/*_response.json")).read_text())
    assert request["payload"] == calls[0]
    assert json.loads(captured_response["raw_response"]["choices"][0]["message"]["content"]) == original_reply
    assert response == original_reply and reply == original_reply
    assert ("active_job" in response["next_transition"]["expected_end_state"]) is include_active_job
    assert session == before


def test_outline_schema_at_provider_boundary_accepts_explicit_candidate_markers(tmp_path, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        multi_turn_phase_response_schema,
    )

    reply = {"thought": "Use recorded resource facts", "decision": None, "context_requests": [],
             "candidate_events": [{"outline_id": "recorded-1", "event_name": "recorded_event",
                                    "resource_jid": "recorded-resource", "rationale": "Recorded support",
                                    "expected_end_state": {"resource_state": "idle"}}]}
    schema = multi_turn_phase_response_schema("outline", outline_mode="incremental_candidates_validated",
        outline_context_retrieval=True, recovery_selection_mode="neurosymbolic")
    jsonschema.validate(reply, schema["schema"])
    def create(**kwargs):
        assert kwargs["response_format"]["json_schema"] == schema
        return _response(json.dumps(reply))
    monkeypatch.setattr(llm_agent, "_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    with capture_requests(directory=tmp_path, run_id="candidate-markers", stage="outline", turn=2):
        response = asyncio.run(_agent().ask_llm_structured("Read recorded facts", response_format=schema))
    assert response == reply
    request = json.loads(next(tmp_path.glob("requests/*/*_request.json")).read_text())
    assert request["payload"]["response_format"]["json_schema"] == schema
    capture = json.loads(next(tmp_path.glob("requests/*/*_response.json")).read_text())
    assert json.loads(capture["raw_response"]["choices"][0]["message"]["content"]) == reply


@pytest.mark.parametrize("markers", [{}, {"decision": None}, {"context_requests": []},
                                     {"decision": None, "context_requests": []}])
def test_candidate_markers_keep_candidate_validation_and_response_intact(monkeypatch, markers):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_outline_generation,
    )

    handler = AsyncMock(return_value=("need_revision", {}))
    monkeypatch.setattr(multi_turn_outline_generation, "_handle_outline_incremental_candidates_validated", handler)
    session = {"outline_context_retrieval": True, "outline_mode": "incremental_candidates_validated",
               "recovery_selection_mode": "neurosymbolic"}
    response = {"thought": "Use recorded facts", "candidate_events": [{"outline_id": "review"}], **markers}
    before = deepcopy(response)
    asyncio.run(multi_turn_outline_generation._handle_outline_phase(
        session_state=session, prepared_recovery_request={}, planner=object(), parsed_response=response))
    handler.assert_awaited_once()
    assert handler.call_args.kwargs["parsed_response"] == before
    assert response == before


@pytest.mark.parametrize("markers", [{"decision": "candidate"}, {"decision": False},
                                     {"context_requests": ["/parts"]}, {"context_requests": None},
                                     {"decision": None, "context_requests": ["/parts"]}])
def test_invalid_candidate_markers_cannot_reach_candidate_validation(monkeypatch, markers):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_outline_generation,
    )

    handler = AsyncMock(side_effect=AssertionError("Invalid format reached candidate validation"))
    monkeypatch.setattr(multi_turn_outline_generation, "_handle_outline_incremental_candidates_validated", handler)
    session = {"outline_context_retrieval": True, "outline_mode": "incremental_candidates_validated",
               "recovery_selection_mode": "neurosymbolic"}
    response = {"thought": "Use recorded facts", "candidate_events": [{"outline_id": "review"}], **markers}
    decision, turn = asyncio.run(multi_turn_outline_generation._handle_outline_phase(
        session_state=session, prepared_recovery_request={}, planner=object(), parsed_response=response))
    assert decision == "need_revision" and turn["validation_findings"]
    handler.assert_not_called()


@pytest.mark.parametrize("candidates", [[], [{"outline_id": "review"}]])
def test_actual_retrieval_cannot_include_candidate_fields(monkeypatch, candidates):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_context,
        multi_turn_outline_generation,
    )

    resolve = Mock(side_effect=AssertionError("Mixed response retrieved facts"))
    handler = AsyncMock(side_effect=AssertionError("Mixed response reached candidate validation"))
    monkeypatch.setattr(multi_turn_context, "resolve_outline_context", resolve)
    monkeypatch.setattr(multi_turn_outline_generation, "_handle_outline_incremental_candidates_validated", handler)
    response = {"thought": "Read facts", "decision": "need_context", "context_requests": ["/parts"],
                "candidate_events": candidates}
    session = {"outline_context_retrieval": True}
    _, turn = asyncio.run(multi_turn_outline_generation._handle_outline_phase(
        session_state=session, prepared_recovery_request={}, planner=object(), parsed_response=response))
    assert turn["context_errors"] and turn["served_context"] == {}
    handler.assert_not_called()
    resolve.assert_not_called()


@pytest.mark.parametrize("model,effort", [("test-model", "medium"), ("gpt-4o", "high"),
                                         ("gpt-5.6-sol", "low"), ("gpt-4o-2024-11-20", "medium")])
def test_capture_equals_every_call_including_retry_tools_and_raw_response(tmp_path, monkeypatch, model, effort):
    calls = []
    responses = [
        ConnectionError("offline"),
        _response(
            None,
            [
                SimpleNamespace(
                    id="tool-1",
                    function=SimpleNamespace(name="observe", arguments='{"part":"gear_large"}'),
                )
            ],
        ),
        _response('{"accepted":true}'),
    ]

    def create(**kwargs):
        calls.append(deepcopy(kwargs))
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(
        llm_agent,
        "_client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    monkeypatch.setattr(llm_agent.time, "sleep", lambda _: None)
    agent = _agent()
    agent.model, agent.reasoning_effort = model, effort
    evaluator_only = {"golden_answer": "EVALUATOR_CANARY_82"}
    with capture_requests(directory=tmp_path, run_id="run-1", stage="outline", turn=3):
        result = asyncio.run(
            agent.ask_llm_structured(
                "Recover the observed part",
                response_format={"name": "response", "schema": {"type": "object"}},
                tools=[{"type": "function", "function": {"name": "observe"}}],
                tool_executor=lambda name, params: {
                    "observed_pose": {"x": 0.1, "y": 0.2, "z": 0.3}
                },
                include_agent_instructions=False,
            )
        )
    assert result == {"accepted": True}
    paths = sorted(tmp_path.glob("requests/*/*_request.json"))
    recorded = [json.loads(path.read_text()) for path in paths]
    assert [row["payload"] for row in recorded] == calls
    assert all(row["model"] == model for row in calls)
    if model == "gpt-4o":
        assert all("reasoning_effort" not in row for row in calls)
        assert "reasoning_effort" not in agent._last_structured_request
    else:
        assert all(row["reasoning_effort"] == effort for row in calls)
        assert agent._last_structured_request["reasoning_effort"] == effort
    assert [(row["tool_round"], row["attempt"]) for row in recorded] == [(0, 1), (0, 2), (1, 1)]
    assert all(
        (row["run_id"], row["stage"], row["turn"]) == ("run-1", "outline", 3) for row in recorded
    )
    assert calls[-1]["messages"][-1]["role"] == "tool"
    assert evaluator_only["golden_answer"] not in json.dumps(calls)
    assert len(list(tmp_path.glob("requests/*/*_response.json"))) == 2
    assert len(list(tmp_path.glob("requests/*/*_error.json"))) == 1
    assert len(list(tmp_path.glob("requests/*/*_tool*.json"))) == 1
    ui_record = read_stage_record(tmp_path, paths[-1])
    assert ui_record["captures"]
    assert any(row["record"].get("payload") == calls[-1] for row in ui_record["captures"])


def test_malformed_response_is_recorded_before_parsing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        llm_agent,
        "_client",
        SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: _response("invalid JSON"))
            )
        ),
    )
    with capture_requests(directory=tmp_path, run_id="run", stage="primitive_generation", turn=1):
        with pytest.raises(json.JSONDecodeError):
            asyncio.run(_agent().ask_llm_structured("prompt", response_format={}))
    raw = json.loads(next(tmp_path.glob("requests/*/*_response.json")).read_text())
    assert raw["raw_response"]["choices"][0]["message"]["content"] == "invalid JSON"
    assert list(tmp_path.glob("requests/*/*_parse_error.json"))


def _claude_message(text, stop_reason="end_turn"):
    return SimpleNamespace(type="message", role="assistant", stop_reason=stop_reason,
                           content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=120, output_tokens=30, cache_read_input_tokens=0,
                                                 cache_creation_input_tokens=0))


def _claude_client(message, calls):
    class Stream:
        def __enter__(self):
            return SimpleNamespace(get_final_message=lambda: message)

        def __exit__(self, *exc):
            return False

    def stream(**kwargs):
        calls.append(kwargs)
        return Stream()

    client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    client.with_options = Mock(return_value=client)
    return client


def test_claude_model_gets_the_same_prompt_and_schema_as_a_hint_and_is_recorded(tmp_path, monkeypatch):
    from cais_spade_llm.ui.components.recovery_turns import _response_choices

    calls = []
    reply = {"thought": "Need facts", "decision": "need_context", "context_requests": ["/resources"]}
    client = _claude_client(_claude_message(json.dumps(reply)), calls)
    monkeypatch.setattr(llm_agent, "_claude_client", lambda: client)
    # Drafts borrow the method on an object that is not an LlmAgent (SavedRecoveryProduct).
    borrower = SimpleNamespace(model="claude-sonnet-5-5", reasoning_effort="medium", instructions="")
    schema = {"type": "object", "properties": {"thought": {"type": "string"}}, "required": ["thought"]}
    with capture_requests(directory=tmp_path, run_id="draft", stage="outline_draft", turn=1):
        response = asyncio.run(llm_agent.LlmAgent.ask_llm_structured(borrower, "Read recorded facts",
            include_agent_instructions=False, response_format={"name": "outline", "schema": schema, "strict": False},
            max_attempts=1))
    assert response == reply
    client.with_options.assert_called_once_with(max_retries=0)
    sent = calls[0]
    # The prompt is unchanged; the schema is stated in the system text, not enforced.
    assert sent["messages"] == [{"role": "user", "content": "Read recorded facts"}]
    assert sent["system"] == [{"type": "text", "text": "Reply with only one JSON object that follows this JSON schema:\n"
                                                       + json.dumps(schema)}]
    assert sent["output_config"] == {"effort": "medium"} and "response_format" not in sent
    # Thinking is asked for explicitly: Opus 4.7 and 4.8 would otherwise answer without thinking.
    assert sent["thinking"] == {"type": "adaptive"}
    request = json.loads(next(tmp_path.glob("requests/*/*_request.json")).read_text())
    raw = json.loads(next(tmp_path.glob("requests/*/*_response.json")).read_text())["raw_response"]
    assert request["payload"] == sent
    assert raw["usage"]["input_tokens"] == 120
    assert json.loads(_response_choices(raw)[0]["message"]["content"]) == reply


def test_claude_prompt_is_split_before_its_first_changing_section_for_the_cache(monkeypatch, tmp_path):
    calls = []
    client = _claude_client(_claude_message(json.dumps({"thought": "t"})), calls)
    monkeypatch.setattr(llm_agent, "_claude_client", lambda: client)
    borrower = SimpleNamespace(model="claude-opus-4-7", reasoning_effort="medium", instructions="")
    prompt = "Task and Role\nPlan.\n\nKnowledge Base\n- /parts\n\nValidation Feedback\n- refused\n\nResources\nM2 idle"
    with capture_requests(directory=tmp_path, run_id="draft", stage="outline_draft", turn=1):
        asyncio.run(llm_agent.LlmAgent.ask_llm_structured(borrower, prompt, include_agent_instructions=False,
            response_format={"name": "outline", "schema": {}}, max_attempts=1))
    content = calls[0]["messages"][0]["content"]
    # The same text, in two blocks: what stays the same is marked for reuse, the changing sections follow.
    assert "".join(block["text"] for block in content) == prompt
    assert content[0] == {"type": "text", "text": "Task and Role\nPlan.\n\nKnowledge Base\n- /parts",
                          "cache_control": {"type": "ephemeral"}}
    assert content[1]["text"].startswith("\n\nValidation Feedback\n") and "cache_control" not in content[1]


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_claude_reply_cut_short_is_recorded_and_ends_the_call(tmp_path, monkeypatch, stop_reason):
    monkeypatch.setattr(llm_agent, "_claude_client", lambda: _claude_client(_claude_message("{", stop_reason), []))
    agent = _agent()
    agent.model = "claude-haiku-5-5"
    with capture_requests(directory=tmp_path, run_id="draft", stage="outline_draft", turn=1):
        with pytest.raises(RuntimeError, match=stop_reason):
            asyncio.run(agent.ask_llm_structured("prompt", response_format={"name": "outline", "schema": {}}))
    assert len(list(tmp_path.glob("requests/*/*_response.json"))) == 1


def test_ui_cannot_promote_guessed_or_missing_request_to_exact_capture(tmp_path):
    record = tmp_path / "multi_turn_turn01_outline_result_20260101T010101.json"
    record.write_text(json.dumps({"request_record_paths": [str(tmp_path / "missing.json")]}))
    record.with_name(
        record.name.replace("_result_", "_request_").replace(".json", ".txt")
    ).write_text("reconstructed")
    result = read_stage_record(tmp_path, record)
    assert result["captures"] == [
        {"path": str(tmp_path / "missing.json"), "status": "not captured", "record": {}}
    ]


def test_capture_write_failure_never_retries_a_returned_response(tmp_path, monkeypatch):
    from cais_spade_llm.agents.shared_information.llm_request_records import (
        RequestCaptureError,
        RequestRecords,
    )

    calls = []
    monkeypatch.setattr(
        llm_agent,
        "_client",
        SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: calls.append(deepcopy(kwargs))
                    or _response('{"ok":true}'),
                )
            )
        ),
    )
    original = RequestRecords.write

    def fail_response(self, name, payload):
        if payload["kind"] == "provider_response":
            raise RequestCaptureError("disk unavailable")
        return original(self, name, payload)

    monkeypatch.setattr(RequestRecords, "write", fail_response)
    with capture_requests(directory=tmp_path, run_id="run", stage="outline", turn=1):
        with pytest.raises(RequestCaptureError):
            asyncio.run(_agent().ask_llm_structured("prompt", response_format={}))
    assert len(calls) == 1
    assert len(list(tmp_path.glob("requests/*/*_request.json"))) == 1


def test_tool_failure_is_recorded_without_a_fabricated_response(tmp_path, monkeypatch):
    tool = SimpleNamespace(
        id="tool-bad", function=SimpleNamespace(name="observe", arguments="malformed")
    )
    monkeypatch.setattr(
        llm_agent,
        "_client",
        SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: _response(None, [tool]),
                )
            )
        ),
    )
    with capture_requests(directory=tmp_path, run_id="run", stage="outline", turn=1):
        with pytest.raises(json.JSONDecodeError):
            asyncio.run(
                _agent().ask_llm_structured(
                    "prompt", response_format={}, tool_executor=lambda *_: {}
                )
            )
    assert len(list(tmp_path.glob("requests/*/*_request.json"))) == 1
    error = json.loads(next(tmp_path.glob("requests/*/*_tool*_error.json")).read_text())
    assert error["kind"] == "tool_error"
    assert error["error_type"] == "JSONDecodeError"


@pytest.mark.parametrize("scenario", ["Conveyor breakdown", "ur5e-1 breakdown",
    "Machining breakdown during part processing", "Part slippage"])
def test_runtime_prompt_and_context_tool_exclude_evaluator_and_private_models(
    tmp_path, monkeypatch, scenario
):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _serve_context_requests,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        build_multi_turn_phase_prompt_input,
        render_multi_turn_phase_prompt,
    )

    canary = "EVALUATOR_PRIVATE_SEQUENCE_79831"
    session = {
        "recovery_des_models": {
            "ur5e@localhost": {
                "events": [{"event_name": canary}],
                "state_variables": {},
            }
        },
        "evaluator_data": {"expected_recovery": canary},
    }
    prepared = {
        "llm_input": {"scenario": scenario, "evaluator_data": {"golden_answer": canary},
                      "operator_metadata": {"preferred_recovery": canary},
                      "nominal_sequence": [canary], "route_hints": [canary]},
        "evaluator_data": {"scoring": canary},
    }
    prompt = render_multi_turn_phase_prompt(
        build_multi_turn_phase_prompt_input(
            phase="outline",
            llm_input=prepared["llm_input"],
            session_state=session,
        )
    )
    calls = []
    returned = [
        _response(
            None,
            [
                SimpleNamespace(
                    id="try-private",
                    function=SimpleNamespace(
                        name="context",
                        arguments='{"refs":["/evaluator_data","/recovery_des_models"]}',
                    ),
                )
            ],
        ),
        _response('{"ok":true}'),
    ]

    def create(**kwargs):
        calls.append(deepcopy(kwargs))
        return returned.pop(0)

    def context_tool(name, args):
        served, errors = _serve_context_requests(
            session_state=session,
            prepared_recovery_request=prepared,
            outline_event={},
            context_requests=args["refs"],
        )
        return {"context": served, "errors": errors}

    monkeypatch.setattr(
        llm_agent,
        "_client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    with capture_requests(directory=tmp_path, run_id="run", stage="outline", turn=1):
        asyncio.run(
            _agent().ask_llm_structured(prompt, response_format={}, tool_executor=context_tool)
        )
    assert canary not in json.dumps(calls)
    tool_result = json.loads(calls[-1]["messages"][-1]["content"])
    assert tool_result["context"] == {}
    assert len(tool_result["errors"]) == 2


def test_turn_index_groups_attempts_and_reads_payloads_only_on_expansion(tmp_path, monkeypatch):
    from cais_spade_llm.ui import recovery_evidence

    directory = tmp_path / "recovery_outline"
    call = directory / "requests" / "call-one"
    call.mkdir(parents=True)
    messages = [{"role": "user", "content": "Current DES State\nExact observed facts"}]
    for attempt in (1, 2):
        identity = {"run_id": "one", "stage": "outline", "turn": 3, "call_id": "call-one", "tool_round": 0, "attempt": attempt}
        (call / f"round00_attempt{attempt:02d}_request.json").write_text(json.dumps({**identity,
            "kind": "provider_request", "payload": {"messages": messages}}, indent=2))
    (call / "round00_attempt01_error.json").write_text(json.dumps({**identity, "attempt": 1, "kind": "provider_error"}))
    (call / "round00_attempt02_response.json").write_text(json.dumps({**identity, "kind": "provider_response",
        "raw_response": {"choices": [{"message": {"content": '{"decision":"need_context"}'}}]}}))
    runtime = directory / "turns" / "turn_00003_outline.json"
    runtime.parent.mkdir()
    runtime.write_text(json.dumps({"turn_index": 3, "phase": "outline", "decision": "need_context",
                                  "served_context": {"/parts": {"value": [{"part_name": "gear_small"}]},
                                                     "/resources": {"value": [{"resource_jid": "ur5e-1@localhost"}]}}},
                                 indent=2))
    original = recovery_evidence.read_json_cached
    reader = Mock(side_effect=AssertionError("turn indexing loaded a payload"))
    monkeypatch.setattr(recovery_evidence, "read_json_cached", reader)
    turns = recovery_evidence.recovery_turns(directory, "recovery_outline")
    assert len(turns) == 1
    assert turns[0]["turn"] == 3 and turns[0]["status"] == "need_context"
    # An outline turn has no single resource, so a listed one never labels it.
    assert "resource_jid" not in turns[0]
    reader.assert_not_called()
    monkeypatch.setattr(recovery_evidence, "read_json_cached", original)
    expanded = recovery_evidence.read_recovery_turn(directory, turns[0])
    requests = [row["record"] for row in expanded["captures"] if row["record"]["kind"] == "provider_request"]
    assert len(requests) == 2
    assert all(row["payload"]["messages"] == messages for row in requests)
    assert len(expanded["results"]) == 1


def test_primitive_turns_with_reused_numbers_keep_their_own_captures(tmp_path):
    from cais_spade_llm.ui import recovery_evidence

    for number, jid in enumerate(("ur5e-3@localhost", "ur5e-4@localhost"), 1):
        call = tmp_path / "requests" / f"call-{number}"
        call.mkdir(parents=True)
        request = call / "round00_attempt01_request.json"
        request.write_text(json.dumps({"turn": 1, "stage": "primitive_generation", "call_id": call.name,
            "kind": "provider_request", "payload": {"messages": [{"role": "user", "content": jid}]}}, indent=2))
        response = tmp_path / f"multi_turn_RECOVERY_SEQ{number}_primitive_generation_turn01_response_20261008T10000{number}.txt"
        response.write_text(json.dumps({"outline_id": f"RECOVERY_SEQ{number}", "resource_jid": jid,
            "decision": "need_context", "response": {"context_requests": [f"/resources/{jid}/snapshot"]},
            "validation_evidence": {"long_context": "x" * 10000}, "request_record_paths": [str(request)]}, indent=2))
    turns = recovery_evidence.recovery_turns(tmp_path, "recovery_primitves")
    assert len(turns) == 2
    assert {turn["owner"] for turn in turns} == {"call-1", "call-2"}
    for turn in turns:
        expanded = recovery_evidence.read_recovery_turn(tmp_path, turn)
        assert len(expanded["results"]) == 1
        assert len(expanded["captures"]) == 1
        assert turn["resource_jid"] in expanded["captures"][0]["record"]["payload"]["messages"][0]["content"]
        assert expanded["results"][0]["response"]["context_requests"]


def test_legacy_primitive_response_is_output_and_preserves_its_prompt(tmp_path):
    from cais_spade_llm.ui import recovery_evidence

    response = tmp_path / "multi_turn_RECOVERY_SEQ1_primitive_generation_turn01_response_20261008T100001.txt"
    response.write_text(json.dumps({"outline_id": "RECOVERY_SEQ1", "decision": "need_context",
                                   "response": {"context_requests": ["/parts/gear_small/observed_pose"]}}))
    response.with_name(response.name.replace("_response_", "_prompt_")).write_text("Legacy prepared input")
    response.with_name(response.name.replace("turn01", "turn02")).write_text(response.read_text())
    turns = recovery_evidence.recovery_turns(tmp_path, "recovery_primitves")
    assert len(turns) == 2
    expanded = recovery_evidence.read_recovery_turn(tmp_path, turns[0])
    assert expanded["results"][0]["response"]["context_requests"] == ["/parts/gear_small/observed_pose"]
    assert expanded["prompts"][0]["text"] == "Legacy prepared input"
    assert not expanded["captures"]


@pytest.fixture
def saved_source_context():
    jid, part = "recovery-resource-4@localhost", "gear_small"
    pose = {"x": 0.0, "y": 0.3, "z": 1.025, "frame_id": "world"}
    resource = {"resource_jid": jid, "resource_state": "failed", "held_part": None,
                "resource_location": "home", "current_pose": {"x": 9, "y": 9, "z": 9}}
    recorded_part = {"part_name": part, "part_state": "misplaced", "part_location": None,
                     "current_location": None, "part_holder_resource_jid": None,
                     "observed_pose": pose, "source": "saved_observations", "captured_at": 123.0}
    task = {"outline_id": "recorded-1", "event_name": "recorded_event", "resource_jid": jid, "part_name": part,
            "expected_start_state": {**deepcopy(resource), "part_state": "misplaced", "part_location": None},
            "expected_end_state": {"resource_state": "picked", "resource_location": "assembly_board-v1",
                                   "held_part": part, "part_state": "in_gripper", "part_location": jid},
            "rationale": "Recorded facts"}
    prepared = {"llm_input": {"observed_runtime_state": {"resources": [resource]}, "part_facts": [recorded_part]}}
    return task, prepared, jid, part, pose


@pytest.mark.parametrize("event_name", ["recorded_event", "move_home", "print_part"])
def test_acquisition_uses_part_observation_not_resource_station_or_pose(saved_source_context, event_name):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import (
        recovery_validation_service as validation,
    )

    task, prepared, jid, part, pose = saved_source_context
    task["event_name"] = event_name
    original = deepcopy((task, prepared))
    resources, parts = validation.projected_outline_validation_context(session_state={}, prepared_recovery_request=prepared)
    result = validation.compile_grounded_recovery_outline_task(task, resources_by_jid=resources,
        parts_by_name=parts, validate_binding_tokens=False)
    grounded = result["grounded_action"]
    assert grounded["preconditions"]["part"]["requires_acquisition"] is True
    assert grounded["preconditions"]["source_ref"] == {"kind": "observed_pose", "location": "observed_pose", "pose": pose}
    assert "source_location" not in grounded["target"]
    assert "pose" not in grounded["target"]
    assert grounded["expected_effect"]["resource"]["location"] == "assembly_board-v1"
    assert grounded["expected_effect"]["part"]["location"] == jid
    physical = validation.build_recovery_physical_validation_input(task=task, grounded_action=grounded,
        session_state={}, prepared_recovery_request=prepared)
    assert physical["part_context"]["observed_pose"] == pose
    assert physical["part_context"]["source"] == "saved_observations"
    assert physical["part_context"]["captured_at"] == 123.0
    assert physical["part_context"]["current_location"] is None
    assert (task, prepared) == original


def test_missing_part_pose_cannot_be_filled_from_resource_pose(saved_source_context):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import (
        recovery_validation_service as validation,
    )

    task, prepared, _, _, _ = saved_source_context
    prepared["llm_input"]["part_facts"][0]["observed_pose"] = None
    resources, parts = validation.projected_outline_validation_context(session_state={}, prepared_recovery_request=prepared)
    grounded = validation.compile_grounded_recovery_outline_task(task, resources_by_jid=resources,
        parts_by_name=parts, validate_binding_tokens=False)["grounded_action"]
    assert grounded["preconditions"]["part"]["requires_acquisition"] is True
    assert "source_ref" not in grounded["preconditions"]
    assert "source_location" not in grounded["target"]
    assert "pose" not in grounded["target"]


def test_projected_unknown_part_location_and_released_custody_override_old_aliases(saved_source_context):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import (
        recovery_validation_service as validation,
    )

    task, prepared, jid, part, pose = saved_source_context
    recorded = prepared["llm_input"]["part_facts"][0]
    recorded.update(current_location="home", current_holder_resource_jid=jid)
    resources, parts = validation.projected_outline_validation_context(
        session_state={}, prepared_recovery_request=prepared)
    grounded = validation.compile_grounded_recovery_outline_task(task, resources_by_jid=resources,
        parts_by_name=parts, validate_binding_tokens=False)["grounded_action"]
    assert grounded["preconditions"]["part"]["requires_acquisition"] is True
    assert "holder" not in grounded["preconditions"]["part"]
    assert grounded["preconditions"]["source_ref"] == {
        "kind": "observed_pose", "location": "observed_pose", "pose": pose,
    }
    assert parts[part]["current_location"] == "home"
    assert parts[part]["part_location"] is None
    physical = validation.build_recovery_physical_validation_input(task=task, grounded_action=grounded,
        session_state={}, prepared_recovery_request=prepared)
    assert physical["part_context"]["part_location"] is None
    assert physical["part_context"]["part_holder_resource_jid"] is None
    assert physical["part_context"]["current_location"] == "home"
    assert physical["part_context"]["current_holder_resource_jid"] == jid
    assert physical["grounded_action"]["preconditions"]["source_ref"]["pose"] == pose


@pytest.mark.parametrize("value", [None, "recorded-location"])
@pytest.mark.parametrize("source", ["current", "part"])
def test_physical_input_fills_only_missing_location_and_custody_aliases(saved_source_context, value, source):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import (
        recovery_validation_service as validation,
    )

    task, prepared, _, part, _ = saved_source_context
    row = {"part_name": part, f"{source}_location": value, f"{source}_holder_resource_jid": value}
    prepared["llm_input"]["part_facts"] = [deepcopy(row)]
    physical = validation.build_recovery_physical_validation_input(task=task, grounded_action={},
        session_state={}, prepared_recovery_request=prepared)
    context = physical["part_context"]
    for key in ("part_location", "current_location", "part_holder_resource_jid", "current_holder_resource_jid"):
        assert context[key] == value
    assert prepared["llm_input"]["part_facts"] == [row]


def test_physical_input_retains_explicit_unknown_current_aliases(saved_source_context):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import (
        recovery_validation_service as validation,
    )

    task, prepared, jid, _, _ = saved_source_context
    row = prepared["llm_input"]["part_facts"][0]
    row.update(part_location=jid, part_holder_resource_jid=jid,
               current_location=None, current_holder_resource_jid=None)
    physical = validation.build_recovery_physical_validation_input(task=task, grounded_action={},
        session_state={}, prepared_recovery_request=prepared)
    assert physical["part_context"]["current_location"] is None
    assert physical["part_context"]["current_holder_resource_jid"] is None
    assert physical["part_context"]["part_location"] == jid
    assert physical["part_context"]["part_holder_resource_jid"] == jid


def test_resource_only_transition_keeps_resource_location_target():
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import (
        recovery_validation_service as validation,
    )

    jid = "recovery-resource-4@localhost"
    task = {"outline_id": "resource-only", "event_name": "recorded_event", "resource_jid": jid,
            "expected_start_state": {"resource_state": "idle", "resource_location": "assembly_board-v1"},
            "expected_end_state": {"resource_state": "idle", "resource_location": "home"}}
    grounded = validation.compile_grounded_recovery_outline_task(task,
        resources_by_jid={jid: {"resource_jid": jid, "resource_state": "idle"}},
        parts_by_name={}, validate_binding_tokens=False)["grounded_action"]
    assert grounded["target"]["target_location"] == "home"
    assert grounded["expected_effect"]["resource"]["location"] == "home"
    assert grounded["expected_effect"]["part"] == {}


@pytest.mark.parametrize("already_held", [False, True])
def test_compound_part_relocation_requires_acquisition_only_before_custody(saved_source_context, already_held):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import (
        recovery_validation_service as validation,
    )

    task, prepared, jid, part, pose = saved_source_context
    task["expected_end_state"].update(held_part=None, resource_state="placed", part_state="assembled",
                                       part_location="assembly_board-v1", resource_location="home")
    if already_held:
        prepared["llm_input"]["observed_runtime_state"]["resources"][0]["held_part"] = part
        prepared["llm_input"]["part_facts"][0].update(part_holder_resource_jid=jid, part_location=jid)
    resources, parts = validation.projected_outline_validation_context(session_state={}, prepared_recovery_request=prepared)
    grounded = validation.compile_grounded_recovery_outline_task(task, resources_by_jid=resources,
        parts_by_name=parts, validate_binding_tokens=False)["grounded_action"]
    assert grounded["preconditions"]["part"]["requires_acquisition"] is (not already_held)
    assert grounded["expected_effect"]["resource"]["location"] == "home"
    assert grounded["expected_effect"]["part"]["location"] == "assembly_board-v1"
    assert grounded["expected_effect"]["part"]["holder"] is None
    if not already_held:
        assert grounded["preconditions"]["source_ref"]["pose"] == pose
    else:
        assert "source_ref" not in grounded["preconditions"]


@pytest.mark.parametrize(("resource_jid", "part_state"), [("M1", "machined"), ("3D Printing Station", "ready")])
def test_nonrobot_processing_state_change_does_not_infer_acquisition(resource_jid, part_state):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import (
        recovery_validation_service as validation,
    )

    task = {"outline_id": "recorded-processing", "event_name": "recorded_event", "resource_jid": resource_jid,
            "part_name": "gear_small", "expected_end_state": {"resource_state": "idle", "part_state": part_state,
                                                                "part_location": resource_jid}}
    resource = {"resource_jid": resource_jid, "resource_state": "failed", "resource_location": resource_jid}
    part = {"part_name": "gear_small", "part_state": "ready", "part_location": resource_jid}
    grounded = validation.compile_grounded_recovery_outline_task(task, resources_by_jid={resource_jid: resource},
        parts_by_name={"gear_small": part}, validate_binding_tokens=False)["grounded_action"]
    assert grounded["preconditions"]["part"]["requires_acquisition"] is False
    assert "source_ref" not in grounded["preconditions"]


def test_explicit_grounded_acquisition_requirement_is_retained(saved_source_context):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import (
        recovery_validation_service as validation,
    )

    task, prepared, _, _, _ = saved_source_context
    task["expected_end_state"] = {"resource_state": "idle", "part_state": "ready"}
    task["preconditions"] = {"part": {"requires_acquisition": True}}
    resources, parts = validation.projected_outline_validation_context(session_state={}, prepared_recovery_request=prepared)
    grounded = validation.compile_grounded_recovery_outline_task(task, resources_by_jid=resources,
        parts_by_name=parts, validate_binding_tokens=False)["grounded_action"]
    assert grounded["preconditions"]["part"]["requires_acquisition"] is True

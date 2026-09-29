"""Provider transport parity, immutable attempts, tool rounds, and UI capture."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

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


def test_capture_equals_every_call_including_retry_tools_and_raw_response(tmp_path, monkeypatch):
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


def test_runtime_prompt_and_context_tool_exclude_evaluator_and_private_models(
    tmp_path, monkeypatch
):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_prompts import (
        build_multi_turn_phase_prompt_input,
        render_multi_turn_phase_prompt,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _serve_context_requests,
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
        "llm_input": {"evaluator_data": {"golden_answer": canary}},
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

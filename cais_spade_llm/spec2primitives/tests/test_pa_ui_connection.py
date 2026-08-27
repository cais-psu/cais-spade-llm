"""Tests for the PA UI connection through Phase 3.3."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import BadRequestError

from cais_spade_llm.agents.shared_information import llm_agent
from cais_spade_llm.spec2primitives import spec2primitives_ui
from cais_spade_llm.spec2primitives.adapters import ui_runtime
from cais_spade_llm.spec2primitives.adapters.ui_runtime import (
    Spec2PrimitivesUIRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa import product_agent_runtime
from cais_spade_llm.spec2primitives.agents.pa.context_grounding import (
    PAOntologyConfig,
)
from cais_spade_llm.spec2primitives.agents.pa.production_grounding import (
    ProductionProductContextGroundingRuntime,
)
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    ControlledGroundingRuntime,
    complete_context,
    ontology_config,
    request_clarification,
)


class FakeProductAgent:
    """Return controlled structured PA decisions without lifecycle behavior."""

    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[dict[str, object]] = []
        self.setup_calls = 0

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "prompt": prompt,
                "response_format": response_format,
            }
        )
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)

    def setup(self) -> None:
        self.setup_calls += 1
        raise AssertionError("ProductAgent.setup() must not run.")


def test_connected_ui_runs_through_phase_3_3_completion(
    tmp_path: Path,
) -> None:
    product_requirement = "  assemble Medium Gear exactly  "
    product_agent = FakeProductAgent(
        responses=[
            _needed_context_response(context_ref="Gear_Medium.STL"),
            complete_context(),
        ]
    )
    runtime = _runtime(tmp_path, product_agent)

    interaction = asyncio.run(
        spec2primitives_ui._run_pa_ui_interaction(runtime, product_requirement)
    )

    assert len(product_agent.calls) == 2
    assert product_agent.setup_calls == 0
    assert interaction["product_requirement"] == product_requirement
    interaction_identifier = interaction["interaction_identifier"]
    interaction_root = interaction["interaction_root"]
    assert isinstance(interaction_identifier, str)
    assert interaction_identifier.startswith("interaction_")
    assert interaction_root == tmp_path / interaction_identifier
    assert interaction["phase_3_1"] == _needed_context_response(context_ref="Gear_Medium.STL")
    assert interaction["phase_3_3"] == _completion_response()
    assert interaction["max_pa_turns"] == 12
    served_context = interaction["phase_3_2"]["served_context"]
    assert served_context["context_ref"] == "Gear_Medium.STL"
    assert served_context["evidence_type"] == "CAD"
    assert served_context["provenance"]["repository_path"] == (
        "ros2/cais_lab_robotics/cad_models/Gear_Medium.STL"
    )
    assert _read_json(interaction_root / "products/user_requirement/product_requirement.json") == {
        "product_requirement": product_requirement
    }
    assert (interaction_root / "products/served_references/Gear_Medium.STL.json").is_file()
    assert (interaction_root / "interaction_record/retrieval_0001.json").is_file()

    view = spec2primitives_ui._pa_ui_view(interaction)
    assert view["activity_state"] == "context understanding complete"
    assert "validated Phase 3.5 record" in view["activity_message"]
    assert "Phase 4.0 ontology initialized" in view["activity_message"]
    assert "Ready for Phase 5" in view["activity_message"]
    assert "Phase 5 remains unavailable" in view["activity_message"]
    assert product_requirement in view["messages"]
    assert "context understanding complete" in view["messages"]
    assert "Gear_Medium.STL" in view["needed_context"]
    assert "Gear_Medium.STL" in view["served context"]
    assert "repository_path" in view["Evidence Sources"]
    assert "turn_0001" in view["interaction_record"]
    assert "retrieval_0001" in view["interaction_record"]
    assert "pa_context_settings" in view["interaction_record"]
    assert "ontology_initialization" in view["interaction_record"]
    assert "ProductContextView" in view["Ontology Grounding"]
    assert f"product_requirement: {json.dumps(product_requirement)}" in view[
        "Ontology Grounding"
    ]
    assert "delta_count: 1" in view["Ontology Grounding"]
    assert str(PAOntologyConfig) not in view["Ontology Grounding"]
    assert "http://PAonto.com#specification" in view["Ontology Grounding"]
    assert "assertion_provenance" in view["Ontology Grounding"]
    assert "producer: \"interaction_initializer\"" in view["Ontology Grounding"]
    assert view["Typed Runtime Context"] == "[]"
    assert view["ProductAgent Request Failure"] == ""
    assert "Ready for Phase 5" in view["PA Grounding Completion"]
    assert '"max_pa_turns": 12' in view["interaction_record"]
    assert all(
        title in view for title, _ in spec2primitives_ui._PHASE_2_RESULT_AREAS
    )


def test_connected_ui_passes_and_displays_maximum_50_unchanged(
    tmp_path: Path,
) -> None:
    product_agent = FakeProductAgent(
        responses=[
            _needed_context_response(context_ref="Gear_Medium.STL"),
            complete_context(),
        ]
    )
    observed_turns: list[tuple[int, int]] = []

    interaction = asyncio.run(
        spec2primitives_ui._run_pa_ui_interaction(
            _runtime(tmp_path, product_agent),
            "assemble Medium Gear",
            max_pa_turns=50,
            on_pa_turn=lambda turn, limit: observed_turns.append((turn, limit)),
        )
    )

    assert observed_turns == [(1, 50), (2, 50)]
    assert interaction["max_pa_turns"] == 50
    assert '"max_pa_turns": 50' in spec2primitives_ui._pa_ui_view(interaction)["interaction_record"]
    settings = _read_json(
        interaction["interaction_root"] / "interaction_record/pa_context_settings.json"
    )
    assert settings["max_pa_turns"] == 50


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2, 2),
        (12.0, 12),
        (50, 50),
        (True, None),
        (1, None),
        (50.5, None),
        (51, None),
        ("12", None),
        (None, None),
    ],
)
def test_ui_maximum_pa_turns_validation(value: object, expected: int | None) -> None:
    assert spec2primitives_ui._validated_max_pa_turns(value) == expected


def test_connected_ui_stops_for_clarification_without_user_reply(
    tmp_path: Path,
) -> None:
    clarification_question = "Which Medium Gear should be assembled?"
    product_agent = FakeProductAgent(
        responses=[
            _needed_context_response(context_ref="NIST_assembly_instructions.pdf"),
            request_clarification(clarification_question),
        ]
    )
    interaction = asyncio.run(
        spec2primitives_ui._run_pa_ui_interaction(
            _runtime(tmp_path, product_agent),
            "assemble Medium Gear",
        )
    )

    assert interaction["phase_3_3"] == {
        "needed_context": request_clarification(clarification_question)["needed_context"],
        "context understanding complete": False,
    }
    view = spec2primitives_ui._pa_ui_view(interaction)
    assert view["activity_state"] == "clarification needed"
    assert view["clarification"] == clarification_question
    assert clarification_question in view["messages"]
    assert "Not available until Phase 3.4 is implemented." not in view["messages"]
    assert view["pending_clarification_turn"] == "2"


def test_connected_ui_submits_reply_in_same_interaction(tmp_path: Path) -> None:
    grounding = ControlledGroundingRuntime(
        assessments=[
            request_clarification("Which gear size should be assembled?"),
            complete_context(),
        ]
    )
    product_agent = FakeProductAgent(
        responses=[_needed_context_response(context_ref="NIST_assembly_instructions.pdf")]
    )
    runtime = _runtime(tmp_path, product_agent, grounding=grounding)
    interaction = asyncio.run(
        spec2primitives_ui._run_pa_ui_interaction(
            runtime,
            "assemble gear",
        )
    )
    interaction_root = interaction["interaction_root"]

    resumed = asyncio.run(
        spec2primitives_ui._submit_pa_ui_clarification(
            runtime,
            interaction,
            "Medium Gear",
        )
    )

    assert resumed["interaction_root"] == interaction_root
    view = spec2primitives_ui._pa_ui_view(resumed)
    assert view["activity_state"] == "context understanding complete"
    assert "User clarification reply" in view["messages"]
    assert "Medium Gear" in view["messages"]
    assert "clarification_0002" in view["interaction_record"]


def test_connected_ui_cancels_without_product_agent_call(tmp_path: Path) -> None:
    grounding = ControlledGroundingRuntime(
        assessments=[request_clarification("Which gear size should be assembled?")]
    )
    product_agent = FakeProductAgent(
        responses=[_needed_context_response(context_ref="NIST_assembly_instructions.pdf")]
    )
    runtime = _runtime(tmp_path, product_agent, grounding=grounding)
    interaction = asyncio.run(
        spec2primitives_ui._run_pa_ui_interaction(runtime, "assemble gear")
    )
    calls_before = len(product_agent.calls)

    cancelled = spec2primitives_ui._cancel_pa_ui_interaction(interaction)

    assert len(product_agent.calls) == calls_before
    view = spec2primitives_ui._pa_ui_view(cancelled)
    assert view["activity_state"] == "cancelled"
    assert '"action": "cancelled"' in view["clarification"]


def test_connected_ui_records_phase_3_1_failure_and_does_not_serve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api_error = RuntimeError("schema must have a 'type' key")
    api_error.status_code = 400  # type: ignore[attr-defined]
    api_error.request_id = "req_controlled"  # type: ignore[attr-defined]
    api_error.type = "invalid_request_error"  # type: ignore[attr-defined]
    api_error.param = "response_format"  # type: ignore[attr-defined]
    api_error.code = None  # type: ignore[attr-defined]
    api_error.body = {  # type: ignore[attr-defined]
        "message": "schema must have a 'type' key",
        "private_body_field": "must not render",
    }
    wrapped = RuntimeError("LLM call failed")
    wrapped.__cause__ = api_error
    product_agent = FakeProductAgent(error=wrapped)
    serving_calls: list[Path] = []

    def unexpected_serving(interaction_root: Path) -> dict[str, object]:
        serving_calls.append(interaction_root)
        raise AssertionError("Phase 3.2 must not run after a Phase 3.1 failure.")

    monkeypatch.setattr(
        spec2primitives_ui,
        "serve_pa_requested_context",
        unexpected_serving,
    )
    interaction = asyncio.run(
        spec2primitives_ui._run_pa_ui_interaction(
            _runtime(tmp_path, product_agent),
            "assemble Medium Gear",
        )
    )

    assert serving_calls == []
    assert interaction["phase_3_1"]["failure"]["reason"] == "pa_call_failed"
    assert interaction["phase_3_2"] is None
    view = spec2primitives_ui._pa_ui_view(interaction)
    assert view["activity_state"] == "failed"
    assert "status_code: 400" in view["ProductAgent Request Failure"]
    assert 'param: "response_format"' in view["ProductAgent Request Failure"]
    assert "schema must have a 'type' key" in view["ProductAgent Request Failure"]
    assert "private_body_field" not in view["ProductAgent Request Failure"]


def test_product_agent_failure_text_supports_older_records() -> None:
    text = spec2primitives_ui._product_agent_request_failure_text(
        {
            "reason": "pa_call_failed",
            "message": "controlled legacy failure",
        }
    )

    assert 'reason: "pa_call_failed"' in text
    assert 'message: "controlled legacy failure"' in text
    assert "diagnostic: unavailable for this record" in text


def test_production_ui_fails_closed_when_grounding_is_unconfigured(
    tmp_path: Path,
) -> None:
    product_agent = FakeProductAgent(
        responses=[_needed_context_response(context_ref="Gear_Medium.STL")]
    )

    interaction = asyncio.run(
        spec2primitives_ui._run_pa_ui_interaction(
            _runtime(tmp_path, product_agent, with_grounding=False),
            "assemble Medium Gear",
        )
    )
    view = spec2primitives_ui._pa_ui_view(interaction)

    assert product_agent.calls == []
    assert interaction["phase_3_1"]["failure"]["reason"] == "grounding_unavailable"
    assert interaction["phase_3_2"] is None
    assert interaction["phase_3_3"] is None
    assert view["activity_state"] == "grounding unavailable"
    assert "No PA evidence decision was requested" in view["activity_message"]
    assert "context understanding complete" not in view["messages"]


def test_ui_ignores_unbacked_completion_turn(tmp_path: Path) -> None:
    interaction_root = tmp_path / "interaction_controlled"
    record_root = interaction_root / "interaction_record"
    record_root.mkdir(parents=True)
    (record_root / "turn_0002.json").write_text(
        json.dumps(
            {
                "turn": 2,
                "product_requirement": "assemble Medium Gear",
                "PA_input": {"assessment_ref": "missing_decision.json"},
                "PA_output": _completion_response(),
                "failure": None,
            }
        ),
        encoding="utf-8",
    )
    interaction = {
        "interaction_identifier": "interaction_controlled",
        "interaction_root": interaction_root,
        "product_requirement": "assemble Medium Gear",
        "phase_3_1": {"needed_context": {}},
        "phase_3_2": None,
        "phase_3_3": None,
        "max_pa_turns": 12,
    }

    view = spec2primitives_ui._pa_ui_view(interaction)

    assert view["activity_state"] != "context understanding complete"
    assert "context understanding complete" not in view["messages"]


def test_connected_ui_uses_unique_no_overwrite_interaction_roots(
    tmp_path: Path,
) -> None:
    request = _needed_context_response(context_ref="Gear_Medium.STL")
    product_agent = FakeProductAgent(
        responses=[request, complete_context(), request, complete_context()]
    )
    runtime = _runtime(tmp_path, product_agent)

    first = asyncio.run(
        spec2primitives_ui._run_pa_ui_interaction(
            runtime,
            "assemble Medium Gear",
        )
    )
    second = asyncio.run(
        spec2primitives_ui._run_pa_ui_interaction(
            runtime,
            "assemble Medium Gear",
        )
    )

    assert first["interaction_identifier"] != second["interaction_identifier"]
    assert first["interaction_root"] != second["interaction_root"]
    assert first["interaction_root"].is_dir()
    assert second["interaction_root"].is_dir()
    assert len(product_agent.calls) == 4
    assert product_agent.setup_calls == 0


def test_connected_ui_routes_live_request_to_phase_3_2_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    product_agent = FakeProductAgent(
        responses=[
            _needed_context_response(request_live_observation=True),
            complete_context(),
        ]
    )
    serving_calls: list[Path] = []

    def controlled_serving(interaction_root: Path) -> dict[str, object]:
        serving_calls.append(interaction_root)
        return {
            "served_context": {
                "context_ref": None,
                "observation_ref": "observation_0001",
                "evidence_type": "observation",
                "evidence_label": "live",
                "provenance": {
                    "manifest_path": ("products/observations/observation_0001/manifest.json")
                },
                "observation_evidence": {},
            }
        }

    monkeypatch.setattr(
        spec2primitives_ui,
        "serve_pa_requested_context",
        controlled_serving,
    )
    interaction = asyncio.run(
        spec2primitives_ui._run_pa_ui_interaction(
            _runtime(tmp_path, product_agent),
            "assemble Medium Gear",
        )
    )

    assert serving_calls == [interaction["interaction_root"]]
    assert interaction["phase_3_2"]["served_context"]["observation_ref"] == ("observation_0001")


def test_product_agent_runtime_delegates_only_the_structured_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[dict[str, object]] = []

    class ControlledSharedProductAgent:
        def __init__(
            self,
            jid: str,
            password: str,
            *,
            name: str,
            instruction_override: str,
            model: str,
        ) -> None:
            constructed.append(
                {
                    "jid": jid,
                    "password": password,
                    "name": name,
                    "instruction_override": instruction_override,
                    "model": model,
                }
            )
            self.calls: list[dict[str, object]] = []

        async def ask_llm_structured(
            self,
            prompt: str,
            *,
            response_format: dict[str, Any],
        ) -> dict[str, Any]:
            self.calls.append({"prompt": prompt, "response_format": response_format})
            return _needed_context_response(context_ref="Gear_Medium.STL")

    monkeypatch.setattr(
        product_agent_runtime,
        "ProductAgent",
        ControlledSharedProductAgent,
    )
    runtime = product_agent_runtime.create_product_agent_context_runtime(model="gpt-5.4")
    response_format = {"strict": True}

    result = asyncio.run(
        runtime.ask_llm_structured(
            "controlled prompt",
            response_format=response_format,
        )
    )

    assert result == _needed_context_response(context_ref="Gear_Medium.STL")
    assert constructed == [
        {
            "jid": "spec2primitives_pa@localhost",
            "password": "",
            "name": "spec2primitives_pa",
            "instruction_override": (
                "For Spec2Primitives, follow the supplied structured "
                "needed_context prompt exactly. Do not plan, contact another "
                "agent, or execute robot behavior."
            ),
            "model": "gpt-5.4",
        }
    ]
    assert not hasattr(runtime, "setup")


def test_shared_structured_call_reports_bad_request_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        headers={"x-request-id": "req_controlled"},
    )
    api_error = BadRequestError(
        "Error code: 400",
        response=response,
        body={
            "message": "schema must have a 'type' key",
            "type": "invalid_request_error",
            "param": "response_format",
            "code": None,
            "private_body_field": "must not surface",
        },
    )
    calls: list[dict[str, object]] = []

    class ControlledCompletions:
        def create(self, **kwargs: object) -> object:
            calls.append(kwargs)
            raise api_error

    monkeypatch.setattr(
        llm_agent,
        "_client",
        SimpleNamespace(
            chat=SimpleNamespace(completions=ControlledCompletions())
        ),
    )

    async def controlled_to_thread(function: object, *args: object) -> object:
        assert callable(function)
        return function(*args)

    monkeypatch.setattr(llm_agent.asyncio, "to_thread", controlled_to_thread)
    runtime = SimpleNamespace(
        instructions="",
        model="gpt-5.4",
        reasoning_effort="medium",
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(
            llm_agent.LlmAgent.ask_llm_structured(
                runtime,
                "controlled prompt that must not surface",
                response_format={
                    "name": "controlled_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [],
                        "properties": {},
                    },
                },
            )
        )

    assert len(calls) == 1
    assert exc_info.value.__cause__ is api_error
    message = str(exc_info.value)
    assert "BadRequestError" in message
    assert "status_code=400" in message
    assert "request_id=req_controlled" in message
    assert "param=response_format" in message
    assert "schema must have a 'type' key" in message
    assert "controlled prompt" not in message
    assert "private_body_field" not in message


def test_ui_runtime_factory_uses_project_tbox_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dual_gazebo = object()
    product_agent = FakeProductAgent()
    selected_models: list[str] = []
    monkeypatch.delenv("SPEC2PRIMITIVES_PPR_TBOX_PATH", raising=False)
    monkeypatch.delenv("SPEC2PRIMITIVES_PPR_NAMESPACE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "controlled-test-key")
    monkeypatch.setattr(
        product_agent_runtime,
        "create_product_agent_context_runtime",
        lambda *, model: selected_models.append(model) or product_agent,
    )

    runtime = ui_runtime.create_spec2primitives_ui_runtime(dual_gazebo)

    assert runtime.dual_gazebo is dual_gazebo
    assert runtime.product_agent is product_agent
    assert runtime.contexts_root == ui_runtime.SPEC2PRIMITIVES_CONTEXTS_ROOT
    assert selected_models == ["gpt-5.4"]
    assert runtime.model_config is not None
    assert runtime.ontology_config == PAOntologyConfig(
        ui_runtime.DEFAULT_SPEC2PRIMITIVES_PPR_TBOX_PATH,
        ui_runtime.DEFAULT_SPEC2PRIMITIVES_PPR_NAMESPACE,
    )
    assert isinstance(
        runtime.grounding_runtime,
        ProductionProductContextGroundingRuntime,
    )
    assert runtime.tbox is not None
    assert runtime.tbox.source_path == (
        ui_runtime.DEFAULT_SPEC2PRIMITIVES_PPR_TBOX_PATH.resolve()
    )
    assert runtime.document_vision_runtime is not None
    assert runtime.observation_capture_runtime is not None
    assert runtime.document_diagnostic_unavailable_reason is None


def test_ui_runtime_factory_honors_paired_tbox_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product_agent = FakeProductAgent()
    config = ontology_config()
    monkeypatch.setenv("SPEC2PRIMITIVES_PPR_TBOX_PATH", str(config.tbox_path))
    monkeypatch.setenv("SPEC2PRIMITIVES_PPR_NAMESPACE", config.ppr_namespace)
    monkeypatch.setenv("OPENAI_API_KEY", "controlled-test-key")
    monkeypatch.setattr(
        product_agent_runtime,
        "create_product_agent_context_runtime",
        lambda *, model: product_agent,
    )

    runtime = ui_runtime.create_spec2primitives_ui_runtime(object())

    assert runtime.ontology_config == config
    assert isinstance(
        runtime.grounding_runtime,
        ProductionProductContextGroundingRuntime,
    )
    assert runtime.tbox is not None
    assert runtime.tbox.source_path == config.tbox_path.resolve()
    assert runtime.document_vision_runtime is not None
    assert runtime.document_diagnostic_unavailable_reason is None


@pytest.mark.parametrize(
    ("tbox_path", "ppr_namespace"),
    [
        ("/tmp/override.owl", None),
        (None, "http://PAonto.com#"),
    ],
)
def test_ui_runtime_factory_rejects_partial_tbox_override(
    monkeypatch: pytest.MonkeyPatch,
    tbox_path: str | None,
    ppr_namespace: str | None,
) -> None:
    if tbox_path is None:
        monkeypatch.delenv("SPEC2PRIMITIVES_PPR_TBOX_PATH", raising=False)
    else:
        monkeypatch.setenv("SPEC2PRIMITIVES_PPR_TBOX_PATH", tbox_path)
    if ppr_namespace is None:
        monkeypatch.delenv("SPEC2PRIMITIVES_PPR_NAMESPACE", raising=False)
    else:
        monkeypatch.setenv("SPEC2PRIMITIVES_PPR_NAMESPACE", ppr_namespace)
    monkeypatch.setenv("OPENAI_API_KEY", "controlled-test-key")
    monkeypatch.setattr(
        product_agent_runtime,
        "create_product_agent_context_runtime",
        lambda *, model: FakeProductAgent(),
    )

    runtime = ui_runtime.create_spec2primitives_ui_runtime(object())

    assert runtime.ontology_config is None
    assert runtime.grounding_runtime is None
    assert runtime.tbox is None
    assert runtime.document_vision_runtime is None
    assert "Set both SPEC2PRIMITIVES_PPR_TBOX_PATH" in (
        runtime.document_diagnostic_unavailable_reason or ""
    )


def test_ui_runtime_factory_rejects_invalid_paired_tbox_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "SPEC2PRIMITIVES_PPR_TBOX_PATH",
        str(tmp_path / "missing.owl"),
    )
    monkeypatch.setenv("SPEC2PRIMITIVES_PPR_NAMESPACE", "http://PAonto.com#")
    monkeypatch.setenv("OPENAI_API_KEY", "controlled-test-key")
    monkeypatch.setattr(
        product_agent_runtime,
        "create_product_agent_context_runtime",
        lambda *, model: FakeProductAgent(),
    )

    runtime = ui_runtime.create_spec2primitives_ui_runtime(object())

    assert runtime.ontology_config is not None
    assert runtime.grounding_runtime is None
    assert runtime.tbox is None
    assert runtime.document_vision_runtime is None
    assert "Authoritative TBox is invalid" in (
        runtime.document_diagnostic_unavailable_reason or ""
    )


def test_ui_runtime_factory_requires_openai_key_with_project_tbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPEC2PRIMITIVES_PPR_TBOX_PATH", raising=False)
    monkeypatch.delenv("SPEC2PRIMITIVES_PPR_NAMESPACE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        product_agent_runtime,
        "create_product_agent_context_runtime",
        lambda *, model: FakeProductAgent(),
    )

    runtime = ui_runtime.create_spec2primitives_ui_runtime(object())

    assert runtime.ontology_config == PAOntologyConfig(
        ui_runtime.DEFAULT_SPEC2PRIMITIVES_PPR_TBOX_PATH,
        ui_runtime.DEFAULT_SPEC2PRIMITIVES_PPR_NAMESPACE,
    )
    assert runtime.grounding_runtime is None
    assert runtime.tbox is not None
    assert runtime.tbox.source_path == (
        ui_runtime.DEFAULT_SPEC2PRIMITIVES_PPR_TBOX_PATH.resolve()
    )
    assert runtime.document_vision_runtime is None
    assert "Set OPENAI_API_KEY" in (
        runtime.document_diagnostic_unavailable_reason or ""
    )


def test_product_agent_runtime_has_no_setup_planning_or_execution_call() -> None:
    source = Path(product_agent_runtime.__file__).read_text(encoding="utf-8")

    assert "ProductAgent(" in source
    assert ".ask_llm_structured(" in source
    for forbidden_source in (
        ".setup(",
        "ProcessPlanner",
        "ResourceAgent",
        "RobotAgent",
        "primitive_steps",
        ".execute(",
    ):
        assert forbidden_source not in source


def _runtime(
    contexts_root: Path,
    product_agent: FakeProductAgent,
    *,
    with_grounding: bool = True,
    grounding: ControlledGroundingRuntime | None = None,
) -> Spec2PrimitivesUIRuntime:
    return Spec2PrimitivesUIRuntime(
        dual_gazebo=object(),
        product_agent=product_agent,
        contexts_root=contexts_root,
        ontology_config=ontology_config() if with_grounding else None,
        grounding_runtime=(
            grounding or ControlledGroundingRuntime() if with_grounding else None
        ),
    )


def _needed_context_response(
    *,
    context_ref: str | None = None,
    request_live_observation: bool = False,
    clarification_question: str | None = None,
) -> dict[str, Any]:
    return {
        "needed_context": {
            "context_ref": context_ref,
            "request_live_observation": request_live_observation,
            "clarification_question": clarification_question,
        }
    }


def _completion_response() -> dict[str, Any]:
    return {
        "needed_context": None,
        "context understanding complete": True,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

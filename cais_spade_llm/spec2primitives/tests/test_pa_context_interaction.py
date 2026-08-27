"""Tests for the first Spec2Primitives PA needed-context decision."""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from rdflib import OWL, RDF, URIRef

from cais_spade_llm.spec2primitives.agents.pa import context_interaction
from cais_spade_llm.spec2primitives.agents.pa.context_grounding import (
    PAOntologyConfig,
)
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    start_pa_context_interaction,
)
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    PPR_NAMESPACE,
    ControlledGroundingRuntime,
    ontology_config,
)
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_context_refs,
)


class FakeProductAgent:
    """Provide one controlled structured response without agent lifecycle behavior."""

    def __init__(
        self,
        response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

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
        return self.response  # type: ignore[return-value]


@pytest.mark.parametrize(
    "needed_context",
    [
        {
            "context_ref": "NIST_assembly_instructions.pdf",
            "request_live_observation": False,
            "clarification_question": None,
        },
        {
            "context_ref": None,
            "request_live_observation": True,
            "clarification_question": None,
        },
    ],
)
def test_each_valid_first_decision_is_returned_and_recorded(
    tmp_path: Path,
    needed_context: dict[str, object],
) -> None:
    product_requirement = "  assemble Medium Gear exactly  "
    response = {"needed_context": needed_context}
    product_agent = FakeProductAgent(response=response)

    result = _start(
        product_agent,
        tmp_path,
        product_requirement,
    )

    assert result == response
    assert len(product_agent.calls) == 1
    call = product_agent.calls[0]
    assert json.dumps(product_requirement) in call["prompt"]
    response_format = call["response_format"]
    assert response_format["strict"] is True
    context_ref_schema = response_format["schema"]["properties"]["needed_context"]["properties"][
        "context_ref"
    ]
    assert context_ref_schema["type"] == ["string", "null"]
    assert context_ref_schema["enum"] == [None, *approved_context_refs()]
    assert response_format["schema"]["properties"]["needed_context"]["properties"][
        "clarification_question"
    ] == {"type": "null"}
    assert "approved_context_ref_evidence_types" in call["prompt"]
    assert PPR_NAMESPACE in call["prompt"]
    assert 'status": "unresolved' in call["prompt"]

    requirement_record = _read_json(tmp_path / "products/user_requirement/product_requirement.json")
    assert requirement_record == {"product_requirement": product_requirement}
    turn_record = _read_json(tmp_path / "interaction_record/turn_0001.json")
    assert turn_record["turn"] == 1
    assert turn_record["product_requirement"] == product_requirement
    assert turn_record["PA_input"] == call
    assert turn_record["PA_output"] == response
    assert turn_record["failure"] is None
    assert (tmp_path / "products/grounding/ontology/interaction_abox.ttl").is_file()


@pytest.mark.parametrize("product_requirement", ["", " \t\n"])
def test_empty_product_requirement_is_rejected_without_call_or_records(
    tmp_path: Path,
    product_requirement: str,
) -> None:
    product_agent = FakeProductAgent(
        response={
            "needed_context": {
                "context_ref": "NIST_assembly_instructions.pdf",
                "request_live_observation": False,
                "clarification_question": None,
            }
        }
    )

    result = asyncio.run(start_pa_context_interaction(product_agent, tmp_path, product_requirement))

    assert result["failure"]["reason"] == "invalid_product_requirement"
    assert product_agent.calls == []
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "pa_output",
    [
        {},
        {"needed_context": None},
        {
            "needed_context": {
                "context_ref": None,
                "request_live_observation": False,
            }
        },
        {
            "needed_context": {
                "context_ref": None,
                "request_live_observation": False,
                "clarification_question": "Question?",
                "extra": True,
            }
        },
        {
            "needed_context": {
                "context_ref": "NIST_assembly_instructions.pdf",
                "request_live_observation": True,
                "clarification_question": None,
            }
        },
        {
            "needed_context": {
                "context_ref": "Not_Approved.STL",
                "request_live_observation": False,
                "clarification_question": None,
            }
        },
        {
            "needed_context": {
                "context_ref": None,
                "request_live_observation": False,
                "clarification_question": None,
            }
        },
        {
            "needed_context": {
                "context_ref": None,
                "request_live_observation": False,
                "clarification_question": "   ",
            }
        },
        {
            "needed_context": {
                "context_ref": None,
                "request_live_observation": 1,
                "clarification_question": None,
            }
        },
        {
            "context understanding complete": True,
        },
        {
            "needed_context": {
                "context_ref": None,
                "request_live_observation": True,
                "clarification_question": None,
            },
            "extra": True,
        },
    ],
)
def test_invalid_pa_responses_are_rejected_and_recorded(
    tmp_path: Path,
    pa_output: object,
) -> None:
    product_agent = FakeProductAgent(response=pa_output)

    result = _start(
        product_agent,
        tmp_path,
        "assemble Medium Gear",
    )

    assert result["failure"]["reason"] == "invalid_pa_response"
    assert len(product_agent.calls) == 1
    turn_record = _read_json(tmp_path / "interaction_record/turn_0001.json")
    assert turn_record["PA_output"] == pa_output
    assert turn_record["failure"] == result["failure"]


def test_pa_call_failure_is_returned_and_recorded(tmp_path: Path) -> None:
    product_agent = FakeProductAgent(error=RuntimeError("controlled PA failure"))

    result = _start(
        product_agent,
        tmp_path,
        "assemble Medium Gear",
    )

    assert result["failure"]["reason"] == "pa_call_failed"
    assert "RuntimeError: controlled PA failure" in result["failure"]["message"]
    assert result["failure"]["diagnostic"] == {
        "stage": "Phase 3.1 ProductAgent structured call",
        "exception": "RuntimeError",
        "status_code": None,
        "request_id": None,
        "error_type": None,
        "param": None,
        "code": None,
        "message": "controlled PA failure",
    }
    turn_record = _read_json(tmp_path / "interaction_record/turn_0001.json")
    assert turn_record["PA_output"] is None
    assert turn_record["failure"] == result["failure"]


def test_pa_call_failure_persists_only_sanitized_api_diagnostic(
    tmp_path: Path,
) -> None:
    class ControlledBadRequestError(Exception):
        pass

    api_error = ControlledBadRequestError("raw exception text")
    api_error.status_code = 400  # type: ignore[attr-defined]
    api_error.request_id = "req_controlled"  # type: ignore[attr-defined]
    api_error.type = "invalid_request_error"  # type: ignore[attr-defined]
    api_error.param = "response_format"  # type: ignore[attr-defined]
    api_error.code = None  # type: ignore[attr-defined]
    api_error.body = {  # type: ignore[attr-defined]
        "message": "schema must have a 'type' key",
        "private_body_field": "must not persist",
    }
    wrapped = RuntimeError("LLM call failed")
    wrapped.__cause__ = api_error
    product_agent = FakeProductAgent(error=wrapped)

    result = _start(product_agent, tmp_path, "assemble Medium Gear")

    assert result["failure"]["diagnostic"] == {
        "stage": "Phase 3.1 ProductAgent structured call",
        "exception": "ControlledBadRequestError",
        "status_code": 400,
        "request_id": "req_controlled",
        "error_type": "invalid_request_error",
        "param": "response_format",
        "code": None,
        "message": "schema must have a 'type' key",
    }
    turn_text = (tmp_path / "interaction_record/turn_0001.json").read_text(
        encoding="utf-8"
    )
    assert "private_body_field" not in turn_text
    assert "raw exception text" not in turn_text


@pytest.mark.parametrize(
    "existing_relative_path",
    [
        "products/user_requirement/product_requirement.json",
        "interaction_record/turn_0001.json",
    ],
)
def test_existing_phase_3_1_record_is_never_overwritten(
    tmp_path: Path,
    existing_relative_path: str,
) -> None:
    existing_path = tmp_path / existing_relative_path
    existing_path.parent.mkdir(parents=True)
    existing_path.write_text("controlled existing record", encoding="utf-8")
    product_agent = FakeProductAgent(response={})

    result = _start(
        product_agent,
        tmp_path,
        "assemble Medium Gear",
    )

    assert result["failure"]["reason"] == "interaction_exists"
    assert product_agent.calls == []
    assert existing_path.read_text(encoding="utf-8") == "controlled existing record"


def test_pa_context_interaction_has_only_the_approved_dependencies() -> None:
    source_path = Path(context_interaction.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    imported_modules = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module
    }
    imported_modules.update(
        alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    )

    assert not any(
        forbidden in module
        for module in imported_modules
        for forbidden in (
            "intelligent_product",
            "llm_agent",
            "rgb_d_cad_grounding",
            ".ui",
            ".agents.ra",
        )
    )
    assert "resolve_context_ref" not in source
    assert "capture_gazebo_observation" not in source
    assert ".setup(" not in source
    assert "primitive_steps" in source


def test_first_turn_clarification_is_rejected_after_abox_initialization(
    tmp_path: Path,
) -> None:
    product_agent = FakeProductAgent(
        response={
            "needed_context": {
                "context_ref": None,
                "request_live_observation": False,
                "clarification_question": "Which gear shaft?",
            }
        }
    )

    result = _start(product_agent, tmp_path, "assemble Medium Gear")

    assert result["failure"]["reason"] == "invalid_pa_response"
    assert "must remain null" in result["failure"]["message"]
    assert (tmp_path / "products/grounding/ontology/interaction_abox.ttl").is_file()


def test_missing_ontology_dependencies_fail_before_pa_or_records(tmp_path: Path) -> None:
    product_agent = FakeProductAgent(response={})

    result = asyncio.run(
        start_pa_context_interaction(
            product_agent,
            tmp_path,
            "assemble Medium Gear",
        )
    )

    assert result["failure"]["reason"] == "grounding_unavailable"
    assert product_agent.calls == []
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("source_name", ["missing.owl", "mixed_paonto_sample.owl"])
def test_invalid_runtime_tbox_fails_before_pa_evidence_selection(
    tmp_path: Path,
    source_name: str,
) -> None:
    tbox_path = (
        tmp_path / source_name
        if source_name == "missing.owl"
        else Path(__file__).parent / "fixtures/ontology" / source_name
    )
    product_agent = FakeProductAgent(response={})

    result = asyncio.run(
        start_pa_context_interaction(
            product_agent,
            tmp_path / "interaction",
            "assemble Medium Gear",
            ontology_config=PAOntologyConfig(tbox_path, PPR_NAMESPACE),
            grounding_runtime=ControlledGroundingRuntime(),
        )
    )

    assert result["failure"]["reason"] == "ontology_initialization_failed"
    assert product_agent.calls == []
    assert _read_json(
        tmp_path / "interaction/products/user_requirement/product_requirement.json"
    ) == {"product_requirement": "assemble Medium Gear"}
    assert (
        _read_json(tmp_path / "interaction/interaction_record/turn_0001.json")["failure"]
        == result["failure"]
    )


def test_mutated_tbox_snapshot_is_rejected_before_pa_call(tmp_path: Path) -> None:
    class MutatedOntologyConfig:
        def load_tbox(self) -> object:
            tbox = ontology_config().load_tbox()
            tbox.graph.add(
                (
                    URIRef(f"{PPR_NAMESPACE}mutated"),
                    RDF.type,
                    OWL.Class,
                )
            )
            return tbox

    product_agent = FakeProductAgent(response={})

    result = asyncio.run(
        start_pa_context_interaction(
            product_agent,
            tmp_path,
            "assemble Medium Gear",
            ontology_config=MutatedOntologyConfig(),  # type: ignore[arg-type]
            grounding_runtime=ControlledGroundingRuntime(),
        )
    )

    assert result["failure"]["reason"] == "ontology_initialization_failed"
    assert "changed after validation" in result["failure"]["message"]
    assert product_agent.calls == []


def _start(
    product_agent: FakeProductAgent,
    interaction_root: Path,
    product_requirement: str,
) -> dict[str, object]:
    return asyncio.run(
        start_pa_context_interaction(
            product_agent,
            interaction_root,
            product_requirement,
            ontology_config=ontology_config(),
            grounding_runtime=ControlledGroundingRuntime(),
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

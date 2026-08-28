"""Tests for generalized evidence-first production PA grounding."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from rdflib import Namespace

from cais_spade_llm.spec2primitives.agents.pa import context_serving
from cais_spade_llm.spec2primitives.agents.pa.context_assessment import (
    continue_pa_context_interaction,
)
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    start_pa_context_interaction,
)
from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingDecision,
    GroundingProducerDescriptor,
    GroundingSession,
    GroundingStatement,
    PAContextGroundingCompletionV2,
    TypedGroundingContract,
    build_product_context_view,
    load_latest_grounding_session,
    load_pa_context_grounding_completion,
)
from cais_spade_llm.spec2primitives.agents.pa.ontology_grounding import (
    OntologyGroundingError,
    propose_and_validate_ontology_grounding,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    initialize_interaction_abox,
)
from cais_spade_llm.spec2primitives.agents.pa.production_grounding import (
    ProductionGroundingError,
    ProductionProductContextGroundingRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa import production_grounding
from cais_spade_llm.spec2primitives.config import load_model_runtime_config
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    PPR_NAMESPACE,
    ontology_config,
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

PPR = Namespace(PPR_NAMESPACE)
_TEST_DOCUMENT_REF = "NIST_assembly_instructions.pdf"
GroundingUpdate = Mapping[str, object] | Callable[[Mapping[str, object]], Mapping[str, object]]


class GeneralizedProductAgent:
    """Return controlled grounding updates and late ontology proposals."""

    def __init__(
        self,
        updates: Sequence[GroundingUpdate],
        *,
        ontology_output: Mapping[str, object]
        | Callable[[Mapping[str, object]], Mapping[str, object]]
        | None = None,
    ) -> None:
        self.updates = list(updates)
        self.ontology_output = ontology_output
        self.calls: list[dict[str, object]] = []

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        """Return one response while retaining exact prompts for boundary tests."""
        name = response_format["name"]
        payload = _prompt_payload(prompt)
        self.calls.append(
            {
                "prompt": prompt,
                "response_format": response_format,
                "payload": payload,
            }
        )
        if name == "spec2primitives_grounding_session_update":
            if not self.updates:
                raise AssertionError("No controlled grounding update remains.")
            update = self.updates.pop(0)
            value = update(payload) if callable(update) else update
            return {"grounding_update": dict(value)}
        if name == "spec2primitives_ontology_grounding_proposal":
            if callable(self.ontology_output):
                value = self.ontology_output(payload)
            elif self.ontology_output is not None:
                value = self.ontology_output
            else:
                ids = [
                    str(item["statement_id"])
                    for item in payload["directly_supported_statements"]  # type: ignore[index]
                ]
                value = {
                    "individuals": [],
                    "relations": [],
                    "literal_facts": [],
                    "unrepresented_statement_ids": ids,
                }
            return {"ontology_grounding_proposal": dict(value)}
        raise AssertionError(f"Unexpected structured request: {name}")


def _runtime(
    vision: ControlledVisionRuntime | None = None,
) -> ProductionProductContextGroundingRuntime:
    return ProductionProductContextGroundingRuntime(
        tbox=ontology_config().load_tbox(),
        document_config=load_model_runtime_config().document_vlm,
        document_vision_runtime=vision or ControlledVisionRuntime(),
    )


def _statement(
    statement_id: str,
    text: str,
    *,
    status: str = "directly_stated",
    sources: Sequence[str] = ("requirement_0001",),
) -> dict[str, object]:
    return {
        "statement_id": statement_id,
        "text": text,
        "status": status,
        "sources": list(sources),
        "reason": "The cited source supports this statement.",
    }


def _terminal_decision(decision_type: str = "ready_for_ontology") -> dict[str, object]:
    return {
        "decision_type": decision_type,
        "need_id": None,
        "provider_id": None,
        "source_ref": None,
        "source_revision": None,
        "query": None,
        "reason": "No further evidence action is required.",
    }


def _ready_update(requirement: str) -> dict[str, object]:
    return {
        "statements": [
            _statement("statement_0001", f"The exact request is {requirement}."),
            _statement(
                "statement_0002",
                "The request may require additional physical context.",
                status="inferred",
            ),
        ],
        "information_needs": [],
        "decision": _terminal_decision(),
        "information_status": "enough",
    }


def _request_update(record_type: str) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
    def create(payload: Mapping[str, object]) -> Mapping[str, object]:
        action = next(
            item
            for item in payload["eligible_provider_actions"]  # type: ignore[index]
            if record_type in item["produced_record_types"]
        )
        return {
            "statements": [
                _statement("statement_0001", "The named item is Medium Gear.")
            ],
            "information_needs": [
                {
                    "need_id": "need_0001",
                    "question": "What approved evidence describes Medium Gear?",
                    "required": True,
                    "sources": ["requirement_0001"],
                    "accepted_record_types": [record_type],
                    "status": "open",
                    "answer_statement_ids": [],
                }
            ],
            "decision": {
                "decision_type": "request_evidence",
                "need_id": "need_0001",
                "provider_id": action["provider_id"],
                "source_ref": action["source_ref"],
                "source_revision": action["source_revision"],
                "query": "Describe the named item using the approved source.",
                "reason": "The source can provide the required record.",
            },
            "information_status": "partial",
        }

    return create


def _resolved_update(payload: Mapping[str, object]) -> Mapping[str, object]:
    prior = payload["prior_session"]
    page_ref = next(
        item
        for item in payload["allowed_source_ids"]  # type: ignore[index]
        if "#page=" in item
    )
    need = dict(prior["information_needs"][0])  # type: ignore[index]
    need.update(
        {
            "status": "resolved",
            "answer_statement_ids": ["statement_0002"],
        }
    )
    return {
        "statements": [
            *prior["statements"],  # type: ignore[index]
            _statement(
                "statement_0002",
                "The approved document contains information about Medium Gear.",
                sources=[page_ref],
            ),
        ],
        "information_needs": [need],
        "decision": _terminal_decision(),
        "information_status": "enough",
    }


def _incomplete_after_attempt(payload: Mapping[str, object]) -> Mapping[str, object]:
    prior = payload["prior_session"]
    return {
        "statements": prior["statements"],  # type: ignore[index]
        "information_needs": prior["information_needs"],  # type: ignore[index]
        "decision": _terminal_decision("incomplete"),
        "information_status": "partial",
    }


def _prompt_payload(prompt: str) -> Mapping[str, object]:
    marker = "Grounding input:\n"
    if marker not in prompt:
        marker = "Late mapping input:\n"
    return json.loads(prompt.split(marker, 1)[1])


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


@pytest.mark.parametrize(
    "requirement",
    [
        "assemble medium gear",
        "drill the mounting hole",
        "weld the frame joint",
        "inspect the finished surface",
    ],
)
def test_manufacturing_requirements_use_the_same_answer_neutral_schema(
    tmp_path: Path,
    requirement: str,
) -> None:
    agent = GeneralizedProductAgent([_ready_update(requirement)])

    result = asyncio.run(
        start_pa_context_interaction(
            agent,
            tmp_path,
            requirement,
            ontology_config=ontology_config(),
            grounding_runtime=_runtime(),
        )
    )

    assert result["grounding_status"] == "complete"
    grounding_call = agent.calls[0]
    assert grounding_call["response_format"]["name"] == (
        "spec2primitives_grounding_session_update"
    )
    serialized = json.dumps(grounding_call, ensure_ascii=False)
    serialized_response_formats = json.dumps(
        [call["response_format"] for call in agent.calls],
        ensure_ascii=False,
    )
    assert "uniqueItems" not in serialized_response_formats
    for leaked in (
        "assembly_context",
        "AssemblyGoalRecord",
        "expected destination",
        "evaluator label",
        "Gazebo truth",
        "TaskTransitionDraft",
        "ContextNeed",
    ):
        assert leaked not in serialized
    assert requirement in grounding_call["prompt"]


def test_cached_preview_supports_first_call_without_document_vlm(tmp_path: Path) -> None:
    interaction_root = tmp_path / "interaction"
    vision = ControlledVisionRuntime()
    _prepare_overview(tmp_path / "source_cache", vision)
    vision.requests.clear()

    def ready_from_preview(payload: Mapping[str, object]) -> Mapping[str, object]:
        page_ref = next(
            item
            for item in payload["allowed_source_ids"]  # type: ignore[index]
            if "#page=" in item
        )
        return {
            "statements": [
                _statement(
                    "statement_0001",
                    "The approved manual lists Medium Gear.",
                    sources=[page_ref],
                )
            ],
            "information_needs": [],
            "decision": _terminal_decision(),
            "information_status": "enough",
        }

    agent = GeneralizedProductAgent([ready_from_preview])
    result = asyncio.run(
        start_pa_context_interaction(
            agent,
            interaction_root,
            "assemble medium gear",
            ontology_config=ontology_config(),
            grounding_runtime=_runtime(vision),
        )
    )

    assert result["grounding_status"] == "complete"
    assert vision.requests == []
    assert vision.targeted_requests == []
    assert [call["response_format"]["name"] for call in agent.calls] == [
        "spec2primitives_grounding_session_update",
        "spec2primitives_ontology_grounding_proposal",
    ]


def test_missing_overview_runs_vlm_only_after_pa_selects_document(
    tmp_path: Path,
) -> None:
    interaction_root = tmp_path / "interaction"
    vision = ControlledVisionRuntime()
    agent = GeneralizedProductAgent(
        [_request_update("DocumentOverviewRecord"), _resolved_update]
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
            max_pa_turns=4,
        )
    )

    assert result["grounding_status"] == "complete"
    assert len(vision.requests) == 1
    assert vision.targeted_requests == []


def test_targeted_document_inspection_runs_once_for_selected_revision(
    tmp_path: Path,
) -> None:
    interaction_root = tmp_path / "interaction"
    vision = ControlledVisionRuntime()
    _prepare_overview(tmp_path / "source_cache", vision)
    vision.requests.clear()
    agent = GeneralizedProductAgent(
        [_request_update("DocumentEvidenceRecord"), _resolved_update]
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
            max_pa_turns=4,
        )
    )

    assert result["grounding_status"] == "complete"
    assert vision.requests == []
    assert len(vision.targeted_requests) == 1
    session = load_latest_grounding_session(interaction_root)
    assert session is not None
    assert len(session.attempted_actions) == 1


def test_exhausted_provider_produces_persisted_incomplete_not_exception(
    tmp_path: Path,
) -> None:
    interaction_root = tmp_path / "interaction"
    agent = GeneralizedProductAgent(
        [_request_update("DocumentOverviewRecord"), _incomplete_after_attempt]
    )
    runtime = _runtime()
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
            max_pa_turns=4,
        )
    )

    assert result == {
        "needed_context": None,
        "context understanding complete": False,
        "grounding_status": "incomplete",
    }
    session = load_latest_grounding_session(interaction_root)
    assert session is not None and session.status == "incomplete"
    assert session.information_needs[0].status == "exhausted"


def test_new_required_need_must_cite_newly_accepted_evidence(tmp_path: Path) -> None:
    interaction_root = tmp_path / "interaction"
    def invalid_new_need(payload: Mapping[str, object]) -> Mapping[str, object]:
        prior = payload["prior_session"]
        page_ref = next(
            item
            for item in payload["allowed_source_ids"]  # type: ignore[index]
            if "#page=" in item
        )
        return {
            "statements": [
                *prior["statements"],  # type: ignore[index]
                _statement("statement_0002", "New document fact.", sources=[page_ref]),
            ],
            "information_needs": [
                *prior["information_needs"],  # type: ignore[index]
                {
                    "need_id": "need_0002",
                    "question": "What newly discovered detail is required?",
                    "required": True,
                    "sources": ["requirement_0001"],
                    "accepted_record_types": ["DocumentEvidenceRecord"],
                    "status": "open",
                    "answer_statement_ids": [],
                },
            ],
            "decision": _terminal_decision("incomplete"),
            "information_status": "partial",
        }

    agent = GeneralizedProductAgent(
        [_request_update("DocumentOverviewRecord"), invalid_new_need]
    )
    runtime = _runtime()
    asyncio.run(
        start_pa_context_interaction(
            agent,
            interaction_root,
            "assemble medium gear",
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
        )
    )
    context_serving.serve_pa_requested_context(interaction_root)

    result = asyncio.run(
        continue_pa_context_interaction(
            agent,
            interaction_root,
            ontology_config=ontology_config(),
            grounding_runtime=runtime,
            max_pa_turns=4,
        )
    )
    assert result["failure"]["reason"] == "assessment_failed"
    assert "new required information need" in result["failure"]["message"]


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
    assert forward[0].produced_record_types == ("SyntheticRecord",)


def test_unchanged_action_cannot_count_as_progress(tmp_path: Path) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "inspect current geometry", tbox)
    view = build_product_context_view(
        tmp_path,
        abox,
        attempted_evidence=(),
        assessed_at_ns=1,
    )
    agent = GeneralizedProductAgent(
        [_request_update("CADMeshRecord"), _request_update("CADMeshRecord")]
    )
    runtime = _runtime()
    first = asyncio.run(
        runtime.initial_product_context_decision(
            agent,
            interaction_root=tmp_path,
            tbox=tbox,
            abox=abox,
            product_context=view.to_record(),
            max_pa_turns=12,
        )
    )
    assert first["grounding_status"] == "waiting_for_evidence"

    with pytest.raises(ProductionGroundingError, match="repeated provider action"):
        asyncio.run(
            runtime.assess_product_context(
                agent,
                interaction_root=tmp_path,
                tbox=tbox,
                abox=abox,
                abox_view=view.to_record(),
                attempted_evidence=(),
                turn_number=2,
                max_pa_turns=12,
            )
        )


def test_emergency_pa_ceiling_persists_incomplete(tmp_path: Path) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "inspect current geometry", tbox)
    view = build_product_context_view(
        tmp_path,
        abox,
        attempted_evidence=(),
        assessed_at_ns=1,
    )
    agent = GeneralizedProductAgent([_request_update("CADMeshRecord")])

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
    session = load_latest_grounding_session(tmp_path)
    assert session is not None and session.status == "incomplete"
    assert session.information_needs[0].status == "exhausted"


def test_inferred_statement_cannot_enter_late_ontology_mapping(tmp_path: Path) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble medium gear", tbox)
    session = _ready_session(
        "assemble medium gear",
        [
            GroundingStatement.from_mapping(
                _statement("statement_0001", "The request names Medium Gear.")
            ),
            GroundingStatement.from_mapping(
                _statement(
                    "statement_0002",
                    "Medium Gear may go on a shaft.",
                    status="inferred",
                )
            ),
        ],
    )
    agent = GeneralizedProductAgent(
        [],
        ontology_output={
            "individuals": [
                {
                    "individual_index": 1,
                    "class_iri": str(PPR.product),
                    "statement_ids": ["statement_0002"],
                }
            ],
            "relations": [],
            "literal_facts": [],
            "unrepresented_statement_ids": ["statement_0001"],
        },
    )

    with pytest.raises(OntologyGroundingError, match="unknown statement"):
        asyncio.run(
            propose_and_validate_ontology_grounding(
                agent,
                interaction_root=tmp_path,
                tbox=tbox,
                abox=abox,
                session=session,
            )
        )
    manifest = _read_json(tmp_path / "products/grounding/ontology/abox_manifest.json")
    assert manifest["delta_count"] == 0


def test_ontology_mapping_rejects_duplicate_unrepresented_statement_ids(
    tmp_path: Path,
) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "inspect medium gear", tbox)
    session = _ready_session(
        "inspect medium gear",
        [
            GroundingStatement.from_mapping(
                _statement("statement_0001", "The request names Medium Gear.")
            )
        ],
    )
    agent = GeneralizedProductAgent(
        [],
        ontology_output={
            "individuals": [],
            "relations": [],
            "literal_facts": [],
            "unrepresented_statement_ids": [
                "statement_0001",
                "statement_0001",
            ],
        },
    )

    with pytest.raises(OntologyGroundingError, match="must be unique"):
        asyncio.run(
            propose_and_validate_ontology_grounding(
                agent,
                interaction_root=tmp_path,
                tbox=tbox,
                abox=abox,
                session=session,
            )
        )

    manifest = _read_json(tmp_path / "products/grounding/ontology/abox_manifest.json")
    assert manifest["delta_count"] == 0


@pytest.mark.parametrize(
    "ontology_output",
    [
        {
            "entities": [
                {
                    "key": "specification",
                    "class_iri": str(PPR.feature),
                    "statement_ids": ["statement_0001"],
                }
            ],
            "relations": [],
            "literal_facts": [],
            "unrepresented_statement_ids": [],
        },
        {
            "individuals": [
                {
                    "individual_index": 1,
                    "class_iri": str(PPR.process),
                    "statement_ids": ["statement_0001"],
                }
            ],
            "relations": [
                {
                    "subject_kind": "specification",
                    "subject_individual_index": None,
                    "subject_iri": None,
                    "predicate_iri": str(PPR.realizes),
                    "object_kind": "new_individual",
                    "object_individual_index": 1,
                    "object_iri": None,
                    "statement_ids": ["statement_0001"],
                }
            ],
            "literal_facts": [],
            "unrepresented_statement_ids": [],
        },
    ],
)
def test_reserved_specification_and_domain_range_violations_do_not_mutate_abox(
    tmp_path: Path,
    ontology_output: Mapping[str, object],
) -> None:
    tbox = ontology_config().load_tbox()
    abox = initialize_interaction_abox(tmp_path, "assemble medium gear", tbox)
    session = _ready_session(
        "assemble medium gear",
        [
            GroundingStatement.from_mapping(
                _statement("statement_0001", "The request says assemble.")
            )
        ],
    )
    agent = GeneralizedProductAgent([], ontology_output=ontology_output)

    with pytest.raises(OntologyGroundingError):
        asyncio.run(
            propose_and_validate_ontology_grounding(
                agent,
                interaction_root=tmp_path,
                tbox=tbox,
                abox=abox,
                session=session,
            )
        )
    manifest = _read_json(tmp_path / "products/grounding/ontology/abox_manifest.json")
    assert manifest["delta_count"] == 0
    assert manifest["accepted_assertion_count"] == 0
    proposal = _read_json(
        tmp_path / "products/grounding/ontology_grounding/proposal_0001.json"
    )
    assert proposal["status"] == "rejected"


def test_completion_v2_pins_projection_contract_session_and_sources(
    tmp_path: Path,
) -> None:
    requirement = "assemble medium gear"
    ontology_output = {
        "individuals": [
            {
                "individual_index": 1,
                "class_iri": str(PPR.process),
                "statement_ids": ["statement_0001"],
            }
        ],
        "relations": [],
        "literal_facts": [],
        "unrepresented_statement_ids": ["statement_0002"],
    }
    ready = _ready_update(requirement)
    ready["statements"][1]["status"] = "directly_stated"  # type: ignore[index]
    agent = GeneralizedProductAgent(
        [ready], ontology_output=ontology_output
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

    assert result["grounding_status"] == "complete"
    completion = load_pa_context_grounding_completion(tmp_path)
    assert isinstance(completion, PAContextGroundingCompletionV2)
    contract = TypedGroundingContract.from_mapping(
        _read_json(tmp_path / completion.typed_grounding_contract_ref)
    )
    assert contract.unrepresented_statement_ids == ("statement_0002",)
    assert contract.source_refs
    assert contract.typed_record_refs == ()
    session_path = tmp_path / completion.grounding_session_ref
    session_path.write_bytes(session_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="GroundingSession hash"):
        load_pa_context_grounding_completion(tmp_path)


def test_document_free_ready_path_never_calls_document_vlm(tmp_path: Path) -> None:
    vision = ControlledVisionRuntime()
    agent = GeneralizedProductAgent([_ready_update("inspect current geometry")])

    result = asyncio.run(
        start_pa_context_interaction(
            agent,
            tmp_path,
            "inspect current geometry",
            ontology_config=ontology_config(),
            grounding_runtime=_runtime(vision),
        )
    )

    assert result["grounding_status"] == "complete"
    assert vision.requests == []
    assert vision.targeted_requests == []


def _ready_session(
    requirement: str,
    statements: Sequence[GroundingStatement],
) -> GroundingSession:
    decision = GroundingDecision.from_mapping(_terminal_decision())
    return GroundingSession.create(
        revision=1,
        requirement_text=requirement,
        statements=statements,
        information_needs=(),
        attempted_actions=(),
        evidence_refs=sorted(
            {source for statement in statements for source in statement.sources}
        ),
        decision=decision,
        status="ready_for_ontology",
        information_status="enough",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

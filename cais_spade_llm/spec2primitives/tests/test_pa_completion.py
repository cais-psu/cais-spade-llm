from __future__ import annotations

"""Tests for the native PA grounding-completion contract."""


import asyncio
import base64
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingContractError,
    PAContextGroundingCompletion,
    build_grounding_evidence,
    build_product_context_view,
    load_pa_context_grounding_completion,
    persist_pa_context_grounding_completion,
    persist_product_context_view,
    validate_grounding_evidence,
)
from cais_spade_llm.spec2primitives.agents.pa.ontology_grounding import (
    OntologyGroundingCandidate,
    OntologyGroundingResult,
    commit_ontology_grounding_candidate,
    propose_and_validate_ontology_grounding,
)
from cais_spade_llm.spec2primitives.agents.pa.presentation_records import (
    AllocationEvidenceSource,
    load_or_create_allocation_presentation,
    load_or_create_evidence_presentation,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    ABoxSnapshot,
    initialize_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
    candidate_resource_catalog,
    check_live_resource_reachability,
    commit_resource_assignment,
    persist_pa_resource_selection,
)

from cais_spade_llm.spec2primitives.ontology import (
    TBoxSnapshot,
    load_predefined_resource_registry,
    load_predefined_workcell,
)
from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import (
    ontology_config,
)


class _ProposalAgent:
    def __init__(
        self,
        evidence_refs: tuple[str, ...] = ("requirement_0001",),
        current_location_ref: str | None = None,
        desired_location_ref: str | None = None,
        target_state_record_ref: str | None = None,
    ) -> None:
        self.evidence_refs = evidence_refs
        self.current_location_ref = current_location_ref
        self.desired_location_ref = desired_location_ref
        self.target_state_record_ref = target_state_record_ref

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
        del prompt, tools, tool_executor, max_tool_rounds
        assert response_format["name"] == "spec2primitives_grounding_result"
        assert self.current_location_ref is not None
        assert self.desired_location_ref is not None
        current_location_value = {
            "name": "medium_gear_location",
            "value_ref": {
                "record_ref": self.current_location_ref,
                "field_path": "/translated_location_m",
            },
            "evidence_refs": [self.current_location_ref],
        }
        desired_location_value = {
            "name": "assembly_board_shaft_location",
            "value_ref": {
                "record_ref": self.desired_location_ref,
                "field_path": "/translated_location_m",
            },
            "evidence_refs": [self.desired_location_ref],
        }
        desired_state_values = [desired_location_value]
        if self.target_state_record_ref is not None:
            desired_state_values.extend(
                [
                    {
                        "name": "specified_finish",
                        "value_ref": {
                            "record_ref": self.target_state_record_ref,
                            "field_path": "/overview/summary",
                        },
                        "evidence_refs": [self.target_state_record_ref],
                    },
                    {
                        "name": "specified_coating",
                        "value_ref": {
                            "record_ref": self.target_state_record_ref,
                            "field_path": "/overview/observations/0",
                        },
                        "evidence_refs": [self.target_state_record_ref],
                    },
                ]
            )
        proposal = {
            "target_feature": {
                "required_process": {
                    "process_iri": "https://cais-spade-llm.local/process/assembly",
                    "evidence_refs": list(self.evidence_refs),
                },
                "current_state": {
                    "statement": {
                        "text": "The medium gear is currently separate from the assembly.",
                        "evidence_refs": list(self.evidence_refs),
                    },
                    "state_values": [current_location_value],
                },
                "desired_state": {
                    "statement": {
                        "text": "The medium gear is assembled as requested.",
                        "evidence_refs": list(self.evidence_refs),
                    },
                    "state_values": desired_state_values,
                },
                "assembly_feature_association": {
                    "assembly": {
                        "name": "assembly_board_with_medium_gear",
                        "evidence_refs": list(self.evidence_refs),
                    },
                    "assembly_features": [
                        {
                            "name": "medium_gear_bore",
                            "owner": {
                                "name": "medium_gear",
                                "type": "Part",
                                "evidence_refs": list(self.evidence_refs),
                            },
                            "state_name": "current_state",
                            "state_value_name": "medium_gear_location",
                            "evidence_refs": list(self.evidence_refs),
                        },
                        {
                            "name": "medium_gear_shaft",
                            "owner": {
                                "name": "assembly_board",
                                "type": "Assembly",
                                "evidence_refs": list(self.evidence_refs),
                            },
                            "state_name": "desired_state",
                            "state_value_name": "assembly_board_shaft_location",
                            "evidence_refs": list(self.evidence_refs),
                        },
                    ],
                    "evidence_refs": list(self.evidence_refs),
                },
            }
        }
        target = proposal["target_feature"]
        target["assembly_feature_association"] = [target["assembly_feature_association"]]
        association = target["assembly_feature_association"][0]
        association["state_names"] = ["desired_state"]
        target["current_state"]["state_values"].append(desired_location_value)
        association["assembly_features"][1]["state_name"] = "current_state"
        return {"result": proposal}


def test_completion_pins_two_states_and_state_location_allocation(
    tmp_path: Path,
) -> None:
    completion = persist_native_completion_fixture(tmp_path)

    assert isinstance(completion, PAContextGroundingCompletion)
    record = completion.to_record()
    assert "schema_version" not in record
    assert record["status"] == "grounding complete"
    assert record["allocation_label"] == "resource assignment validated by MoveIt"
    assert record["validation_scope"] == "moveit_state_location_reachability"
    assert record["motion_executed"] is False
    assert "grounding_session_ref" not in record
    assert "typed_grounding_contract_ref" not in record
    assert "semantic_review_ref" not in record
    assert len(record["typed_context_refs"]) == 2
    assert record["source_refs"][0]["ref"] == "requirement_0001"
    assert len(record["tool_call_refs"]) == 2
    assert record["current_state_iri"].endswith("currentstate_0001")
    assert record["desired_state_iri"].endswith("desiredstate_0001")
    assert "target_feature" not in record
    assert all(
        _read_json(tmp_path / item["ref"])["record_type"] == "RobotFrameLocationRecord"
        for item in record["typed_context_refs"]
    )
    proposal = _read_json(tmp_path / str(record["ontology_projection_ref"]))
    assert "schema_version" not in proposal
    assert proposal["feature_iri"].endswith("feature_0001")
    target_feature = proposal["output"]["target_feature"]
    assert target_feature["required_process"]["process_iri"] == (
        "https://cais-spade-llm.local/process/assembly"
    )
    assert target_feature["current_state"]["state_values"][0]["name"] == ("medium_gear_location")
    assert target_feature["desired_state"]["state_values"][0]["name"] == (
        "assembly_board_shaft_location"
    )
    assert len(target_feature["assembly_feature_association"][0]["assembly_features"]) == 2
    selection = _read_json(tmp_path / str(record["resource_selection_ref"]))
    reachability = _read_json(tmp_path / str(record["reachability_check_ref"]))
    assert "schema_version" not in selection
    assert "schema_version" not in reachability
    assert selection["state_locations"] == {
        state_name: [
            item["evidence_handle"] for item in reachability["state_locations"][state_name]
        ]
        for state_name in ("current_state", "desired_state")
    }
    assert not (
        tmp_path / "products/grounding/completion/typed_grounding_contract_0001.json"
    ).exists()
    assert not (tmp_path / "products/grounding/target_feature_review").exists()
    serialized = json.dumps(record)
    for forbidden in (
        "GroundingSession",
        "TaskTransitionContract",
        "primitive_steps",
        "RobotFramePoseRecord",
        "TargetFeatureGeometryRecord",
        "context_summary",
        "missing_information",
    ):
        assert forbidden not in serialized


def test_completion_pins_multiple_values_from_one_typed_record(
    tmp_path: Path,
) -> None:
    completion = persist_native_completion_fixture(
        tmp_path,
        include_target_state_values=True,
    ).to_record()
    proposal = _read_json(tmp_path / str(completion["ontology_projection_ref"]))
    state_values = proposal["output"]["target_feature"]["desired_state"]["state_values"]
    assert [item["name"] for item in state_values[1:]] == [
        "specified_finish",
        "specified_coating",
    ]
    assert len({item["value_ref"]["record_ref"] for item in state_values[1:]}) == 1
    assert [item["value_ref"]["field_path"] for item in state_values[1:]] == [
        "/overview/summary",
        "/overview/observations/0",
    ]
    state_ref = state_values[1]["value_ref"]["record_ref"]
    assert any(item["ref"] == state_ref for item in completion["typed_context_refs"])

    state_path = tmp_path / state_ref
    changed = _read_json(state_path)
    changed["overview"]["summary"] = "gloss"
    _write_json(state_path, changed)
    with pytest.raises(GroundingContractError, match="typed_context_refs changed"):
        load_pa_context_grounding_completion(tmp_path)


def test_completion_generically_pins_dynamic_query_and_layout_records(
    tmp_path: Path,
) -> None:
    completion = persist_native_completion_fixture(
        tmp_path,
        include_dynamic_grounding_records=True,
    ).to_record()
    record_types = {
        _read_json(tmp_path / item["ref"])["record_type"]
        for item in completion["typed_context_refs"]
    }

    assert "schema_version" not in completion
    assert {
        "DocumentSourceIndexRecord",
        "DocumentQueryRecord",
        "CandidateSpatialRelationRecord",
    }.issubset(record_types)
    assert isinstance(load_pa_context_grounding_completion(tmp_path), PAContextGroundingCompletion)


@pytest.mark.parametrize("opaque", [False, True])
def test_completion_pins_answered_clarification_source(tmp_path: Path, opaque: bool) -> None:
    clarification_ref = "interaction_record/clarification_0001.json"
    clarification_path = tmp_path / clarification_ref
    _write_json(
        clarification_path,
        {
            "record_type": "PAClarification",
            "product_requirement": "assemble medium gear",
            "question_turn": 1,
            "question": "Which product variant is intended?",
            "action": "answered",
            "reply": "Medium Gear",
            "recorded_at_ns": 1,
            "fingerprint": "0" * 64,
        },
    )
    completion = persist_native_completion_fixture(
        tmp_path,
        evidence_refs=("requirement_0001", clarification_ref),
        opaque_clarifications=opaque,
    ).to_record()

    pinned = next(item for item in completion["source_refs"] if item["ref"] == clarification_ref)
    assert pinned["sha256"] == hashlib.sha256(clarification_path.read_bytes()).hexdigest()

    clarification = _read_json(clarification_path)
    clarification["reply"] = "Small Gear"
    _write_json(clarification_path, clarification)
    with pytest.raises(GroundingContractError, match="source evidence changed"):
        load_pa_context_grounding_completion(tmp_path)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("selection", "pinned resource_selection_ref changed"),
        ("assignment", "pinned resource_assignment_delta_ref changed"),
        ("location", "typed_context_refs changed"),
        ("proposal", "pinned ontology_projection_ref changed"),
    ],
)
def test_completion_loader_rejects_changed_native_inputs(
    tmp_path: Path,
    target: str,
    message: str,
) -> None:
    completion = persist_native_completion_fixture(tmp_path).to_record()
    ref_field = {
        "selection": "resource_selection_ref",
        "assignment": "resource_assignment_delta_ref",
        "proposal": "ontology_projection_ref",
    }.get(target)
    if ref_field is not None:
        path = tmp_path / str(completion[ref_field])
    else:
        path = tmp_path / str(completion["typed_context_refs"][0]["ref"])
    value = _read_json(path)
    value["tampered"] = True
    _write_json(path, value)

    with pytest.raises(GroundingContractError, match=message):
        load_pa_context_grounding_completion(tmp_path)


def test_completion_loader_rejects_changed_decision(
    tmp_path: Path,
) -> None:
    decision_root = tmp_path / "decision"
    decision_completion = persist_native_completion_fixture(decision_root).to_record()
    decision_path = decision_root / str(decision_completion["decision_ref"])
    decision = _read_json(decision_path)
    decision["PA_output"]["grounding_status"] = "incomplete"
    _write_json(decision_path, decision)
    with pytest.raises(GroundingContractError, match="pinned decision_ref changed"):
        load_pa_context_grounding_completion(decision_root)


@pytest.mark.parametrize(
    "ref_field",
    [
        "registry_snapshot_ref",
        "workcell_snapshot_ref",
        "evidence_presentation_ref",
        "allocation_presentation_ref",
        "reachability_check_ref",
    ],
)
def test_completion_rejects_changed_allocation_lineage(
    tmp_path: Path,
    ref_field: str,
) -> None:
    completion = persist_native_completion_fixture(tmp_path).to_record()
    record_path = tmp_path / str(completion[ref_field])
    record = _read_json(record_path)
    record["tampered"] = True
    _write_json(record_path, record)

    with pytest.raises(GroundingContractError, match=f"pinned {ref_field} changed"):
        load_pa_context_grounding_completion(tmp_path)


def test_unsupported_completion_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "interaction_record/context_completion_0001.json"
    _write_json(path, {"schema_version": 1})

    with pytest.raises(GroundingContractError, match="Start a fresh interaction"):
        load_pa_context_grounding_completion(tmp_path)


async def prepare_native_completion_fixture(
    root: Path,
    *,
    evidence_refs: tuple[str, ...] = ("requirement_0001",),
    include_target_state_values: bool = False,
    include_dynamic_grounding_records: bool = False,
    resource_symbol: str = "xarm6",
    planning_statuses: dict[str, str] | None = None,
    opaque_clarifications: bool = False,
) -> dict[str, object]:
    """Create a current record chain for offline boundary tests."""
    requirement = "assemble medium gear"
    _write_json(
        root / "products/user_requirement/product_requirement.json",
        {"product_requirement": requirement},
    )
    tbox = ontology_config().load_tbox()
    if not (root / "products/grounding/ontology/abox_manifest.json").exists():
        initialize_interaction_abox(root, requirement, tbox)
    registry = load_predefined_resource_registry(tbox)
    workcell = load_predefined_workcell(tbox, registry)
    current_location_path, current_source_ref = _write_location(
        root,
        "current",
        (0.0, 0.7 if resource_symbol == "ur5e" else -0.7, 1.1),
    )
    desired_location_path, desired_source_ref = _write_location(
        root,
        "desired",
        (0.0, 0.2 if resource_symbol == "ur5e" else -0.2, 1.1),
    )
    current_location_ref = current_location_path.relative_to(root).as_posix()
    desired_location_ref = desired_location_path.relative_to(root).as_posix()
    validate_and_merge_triple_delta(
        root,
        tbox,
        "synthetic_world_location_provider",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [current_location_ref],
        },
        authorized_evidence_refs={current_source_ref},
    )
    desired_merge = validate_and_merge_triple_delta(
        root,
        tbox,
        "synthetic_world_location_provider",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [desired_location_ref],
        },
        authorized_evidence_refs={desired_source_ref},
    )
    target_state_record_ref: str | None = None
    proposal_abox = desired_merge.abox
    authorized_evidence_refs = {
        *evidence_refs,
        current_location_ref,
        desired_location_ref,
    }
    if include_target_state_values:
        target_state_path = _write_target_state_record(root)
        target_state_record_ref = target_state_path.relative_to(root).as_posix()
        authorized_evidence_refs.add(target_state_record_ref)
        target_state_merge = validate_and_merge_triple_delta(
            root,
            tbox,
            "document_evidence",
            {
                "assertions": [],
                "uncertainty": [],
                "unresolved_evidence_needs": [],
                "typed_context_refs": [target_state_record_ref],
            },
            authorized_evidence_refs={"requirement_0001"},
        )
        proposal_abox = target_state_merge.abox
    if include_dynamic_grounding_records:
        proposal_abox = await _merge_dynamic_grounding_records(root, tbox)

    def resolve_typed_record(record_ref: str) -> Mapping[str, object]:
        record_types = {
            current_location_ref: "RobotFrameLocationRecord",
            desired_location_ref: "RobotFrameLocationRecord",
        }
        if target_state_record_ref is not None:
            record_types[target_state_record_ref] = "DocumentOverviewRecord"
        if record_ref not in record_types:
            raise ValueError("record is not accepted")
        record_path = root / record_ref
        return {
            "record_type": record_types[record_ref],
            "record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
            "record": _read_json(record_path),
        }

    proposal_agent = _ProposalAgent(
        evidence_refs,
        current_location_ref,
        desired_location_ref,
        target_state_record_ref,
    )
    evidence_presentation = load_or_create_evidence_presentation(
        root,
        sources=((None, "observation", "fresh_on_call"),),
        explicit_order=("__live_observation__",),
    )
    clarification_catalog = [
        {
            "evidence_type": "user_clarification",
            "evidence_ref": (
                evidence_presentation.opaque_reference(ref, kind="citation")
                if opaque_clarifications
                else ref
            ),
            "reply": _read_json(root / ref)["reply"],
        }
        for ref in evidence_refs
        if ref.startswith("interaction_record/clarification_")
    ]
    candidate = await propose_and_validate_ontology_grounding(
        proposal_agent,
        interaction_root=root,
        tbox=tbox,
        abox=proposal_abox,
        workcell=workcell,
        evidence_catalog=clarification_catalog,
        authorized_evidence_refs=authorized_evidence_refs,
        tools=[],
        tool_executor=_unused_tool,
        max_tool_rounds=1,
        typed_record_resolver=resolve_typed_record,
    )
    assert isinstance(candidate, OntologyGroundingCandidate)
    context_path = persist_product_context_view(
        root,
        build_product_context_view(
            root, proposal_abox, attempted_evidence=(), assessed_at_ns=time.time_ns()
        ),
    )
    proposal = commit_ontology_grounding_candidate(
        candidate,
        context_view_ref=context_path.relative_to(root).as_posix(),
        interaction_root=root,
        tbox=tbox,
        abox=proposal_abox,
        workcell=workcell,
        authorized_evidence_refs=authorized_evidence_refs,
    )
    assert isinstance(proposal, OntologyGroundingResult)

    evidence_sources = tuple(
        _allocation_evidence_source(root, path)
        for path in (current_location_path, desired_location_path)
    )
    resource_catalog = candidate_resource_catalog(proposal.merge.abox, registry, workcell)
    process_iri = str(proposal.proposal.target_feature["required_process"]["process_iri"])
    allocation_presentation = load_or_create_allocation_presentation(
        root,
        evidence_presentation=evidence_presentation,
        process_symbol=workcell.process_symbol_for_iri(process_iri),
        process_iri=process_iri,
        feature_iri=proposal.proposal.feature_iri,
        current_state_iri=f"{proposal.merge.abox.namespace}currentstate_0001",
        desired_state_iri=f"{proposal.merge.abox.namespace}desiredstate_0001",
        resources=tuple(
            (symbol, str(entry["resource_iri"]), str(entry["resource_jid"]))
            for symbol, entry in resource_catalog.items()
        ),
        evidence_sources=evidence_sources,
        explicit_resource_order=tuple(resource_catalog),
        explicit_candidate_order=tuple(source.canonical_key for source in evidence_sources),
    )
    evidence_handles = {
        entry.canonical_key: entry.pa_handle for entry in allocation_presentation.evidence_entries
    }
    reachability_inputs = dict(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        resource_symbol=resource_symbol,
        allocation_presentation=allocation_presentation,
        state_location_record_paths={
            "current_state": [
                (
                    evidence_handles[evidence_sources[0].canonical_key],
                    current_location_path,
                ),
                *([(evidence_handles[evidence_sources[1].canonical_key], desired_location_path)]),
            ],
            "desired_state": [
                (
                    evidence_handles[evidence_sources[1].canonical_key],
                    desired_location_path,
                )
            ],
        },
    )
    from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import OfflineMoveItPlanning

    checks = {}
    tool_call_refs = []
    for check_number, symbol in enumerate(resource_catalog, start=1):
        checks[symbol] = await check_live_resource_reachability(
            runtime=OfflineMoveItPlanning(planning_statuses),
            **{**reachability_inputs, "resource_symbol": symbol, "check_number": check_number},
        )
        check = checks[symbol]
        call_id = f"allocation_tool_call_{check_number:04d}"
        call_ref = f"interaction_record/{call_id}.json"
        _write_json(
            root / call_ref,
            {
                "record_type": "ProductAgentAllocationToolCall",
                "tool_call_id": call_id,
                "tool_name": "check_reachability",
                "arguments": {"resource_symbol": symbol},
                "result_ref": check.record_ref,
                "result_sha256": hashlib.sha256(check.record_path.read_bytes()).hexdigest(),
                "result": {
                    "reachability_check_ref": f"reachability_check_{check_number:04d}",
                    "resource_symbol": symbol,
                    "status": check.status,
                },
                "failure": None,
            },
        )
        tool_call_refs.append(call_ref)
    reachability = checks[resource_symbol]
    selection = persist_pa_resource_selection(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        reachability=reachability,
        allocation_presentation=allocation_presentation,
        state_location_handles=(
            {
                state: tuple(item.evidence_handle for item in values)
                for state, values in reachability.state_locations.items()
            }
        ),
    )
    assignment = commit_resource_assignment(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        selection=selection,
    )
    final_view = build_product_context_view(
        root,
        assignment.abox,
        attempted_evidence=(),
        assessed_at_ns=time.time_ns(),
    )
    persist_product_context_view(root, final_view)
    proposal_ref = proposal.proposal_path.relative_to(root).as_posix()
    return {
        "tbox": tbox,
        "registry": registry,
        "workcell": workcell,
        "product_context": final_view,
        "output": {
            "grounding_status": "complete",
            "resource_assignment_status": "complete",
            "ontology_projection_ref": proposal_ref,
            "resource_selection_ref": selection.record_ref,
            "tool_call_refs": tool_call_refs,
            "allocation_label": "resource assignment validated by MoveIt",
        },
    }


def persist_native_completion_fixture(root: Path, **kwargs) -> PAContextGroundingCompletion:
    """Complete and reload the current offline evidence chain with the production writer."""
    prepared = asyncio.run(prepare_native_completion_fixture(root, **kwargs))
    output = prepared["output"]
    turn_path = root / "interaction_record/turn_0001.json"
    _write_json(
        turn_path,
        {
            "turn": 1,
            "product_requirement": "assemble medium gear",
            "PA_input": {"mode": "native_tool_grounding"},
            "PA_output": output,
            "failure": None,
        },
    )
    persist_pa_context_grounding_completion(
        root,
        tbox=prepared["tbox"],
        product_requirement="assemble medium gear",
        completion_turn=1,
        decision_ref=turn_path.relative_to(root).as_posix(),
        product_context=prepared["product_context"],
        ontology_projection_ref=output["ontology_projection_ref"],
        resource_selection_ref=output["resource_selection_ref"],
        tool_call_refs=tuple(output["tool_call_refs"]),
        registry=prepared["registry"],
        workcell=prepared["workcell"],
    )
    return load_pa_context_grounding_completion(root)


async def _unused_tool(
    tool_name: str,
    arguments: Mapping[str, object],
) -> Mapping[str, object]:
    del tool_name, arguments
    raise AssertionError("No tool call was expected.")


def _write_target_state_record(root: Path) -> Path:
    path = root / "products/grounding/test_target_state/document_overview.json"
    _write_json(
        path,
        {
            "record_type": "DocumentOverviewRecord",
            "producer": "document_evidence",
            "overview": {
                "summary": "matte",
                "observations": ["primer"],
                "uncertainty": [
                    {
                        "description": "The specified finish is inferred from a figure.",
                        "evidence_refs": [path.relative_to(root).as_posix()],
                    }
                ],
            },
        },
    )
    return path


async def _merge_dynamic_grounding_records(
    root: Path,
    tbox: TBoxSnapshot,
) -> ABoxSnapshot:
    """Include current document and measured layout records in completion provenance."""
    document_root = root / "products/grounding/document_evidence"
    geometry_root = root / "products/grounding/rgb_d_cad_grounding"
    document_root.mkdir(parents=True, exist_ok=True)
    geometry_root.mkdir(parents=True, exist_ok=True)
    from cais_spade_llm.spec2primitives.tests.test_document_interpretation import (
        ControlledVisionRuntime,
        _served_document,
    )
    from cais_spade_llm.spec2primitives.config import load_model_runtime_config
    from cais_spade_llm.spec2primitives.tools.document_evidence import (
        index_document_evidence,
        query_document_evidence,
    )
    from cais_spade_llm.spec2primitives.agents.pa.product_context import load_interaction_abox

    indexed = index_document_evidence(
        interaction_root=root,
        tbox=tbox,
        abox=load_interaction_abox(root, tbox),
        served_context=_served_document("NIST_assembly_instructions.pdf"),
        operation_number=1,
    )
    query = await query_document_evidence(
        interaction_root=root,
        source_index_record_path=indexed.source_index_record_path,
        operation_number=1,
        question="What is stated?",
        config=load_model_runtime_config().document_vlm,
        vision_runtime=ControlledVisionRuntime(),
    )
    validate_and_merge_triple_delta(
        root,
        tbox,
        "document_evidence",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [
                indexed.source_index_record_path.relative_to(root).as_posix(),
                query.record_path.relative_to(root).as_posix(),
            ],
        },
        authorized_evidence_refs={
            "NIST_assembly_instructions.pdf",
            *(f"NIST_assembly_instructions.pdf#page={page}" for page in range(1, 7)),
        },
    )

    from cais_spade_llm.spec2primitives.tests.test_cad_size_correspondence import (
        _write_layout_inputs,
    )
    from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import analyze_candidate_layout

    segmentation_path, candidate_paths = _write_layout_inputs(
        root, ((0.0, 0.0, 0.5), (0.1, 0.0, 0.5), (0.2, 0.0, 0.5))
    )
    relation = analyze_candidate_layout(
        interaction_root=root,
        segmentation_record_path=segmentation_path,
        candidate_field_paths=candidate_paths,
    )
    relation_ref = relation.record_path.relative_to(root).as_posix()
    merge = validate_and_merge_triple_delta(
        root,
        tbox,
        "rgb_d_cad_grounding",
        {
            "assertions": [],
            "uncertainty": [],
            "unresolved_evidence_needs": [],
            "typed_context_refs": [relation_ref],
        },
        authorized_evidence_refs=(),
    )
    return merge.abox


def _write_location(
    root: Path,
    name: str,
    translation: tuple[float, float, float],
) -> tuple[Path, str]:
    destination = root / "products/grounding/synthetic_world_location" / name
    destination.mkdir(parents=True, exist_ok=True)
    source_path = destination / "source_evidence.json"
    _write_json(source_path, {"source": "synthetic observed center"})
    source_ref = source_path.relative_to(root).as_posix()
    rgb_path = destination / "selected_rgb.png"
    rgb_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
            "AAAAC0lEQVR42mP8/x8AAusB9Y9Zl4sAAAAASUVORK5CYII="
        )
    )
    rgb_ref = rgb_path.relative_to(root).as_posix()
    segmentation_path = destination / "segmentation_record.json"
    _write_json(
        segmentation_path,
        {
            "record_type": "RGBDSegmentationRecord",
            "producer": "synthetic_neutral_segmentation_provider",
            "cameras": [
                {
                    "observation_handle": f"observation_{name}",
                    "source_artifacts": {
                        "rgb": {
                            "ref": rgb_ref,
                            "sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
                        }
                    },
                    "candidates": [
                        {
                            "candidate_handle": f"candidate_{name}",
                            "pixel_bounds_uv": {
                                "minimum": [10, 10],
                                "maximum": [45, 45],
                            },
                        }
                    ],
                    "label_mask_artifact": {"shape": [64, 64]},
                }
            ],
        },
    )
    segmentation_ref = segmentation_path.relative_to(root).as_posix()
    path = destination / "robot_frame_location_record.json"
    _write_json(
        path,
        {
            "record_type": "RobotFrameLocationRecord",
            "producer": "synthetic_world_location_provider",
            "method": "neutral_fixture",
            "source_segmentation": {
                "ref": segmentation_ref,
                "sha256": hashlib.sha256(segmentation_path.read_bytes()).hexdigest(),
            },
            "robot_frame_conversion": "accepted",
            "location": "available",
            "source_frame": "camera_optical_frame",
            "target_frame": "world",
            "observation_timestamp_ns": 11,
            "translated_location_m": list(translation),
            "candidate_reference": {
                "observation_handle": f"observation_{name}",
                "candidate_handle": f"candidate_{name}",
            },
            "source_evidence": {
                "ref": source_ref,
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            },
        },
    )
    return path, source_ref


def _allocation_evidence_source(
    root: Path,
    location_path: Path,
) -> AllocationEvidenceSource:
    record = _read_json(location_path)
    record_ref = location_path.relative_to(root).as_posix()
    candidate_reference = record["candidate_reference"]
    return AllocationEvidenceSource(
        canonical_key=f"{record_ref}#/translated_location_m",
        record_type="RobotFrameLocationRecord",
        record_ref=record_ref,
        record_sha256=hashlib.sha256(location_path.read_bytes()).hexdigest(),
        field_path="/translated_location_m",
        observation_handle=str(candidate_reference["observation_handle"]),
        candidate_handle=str(candidate_reference["candidate_handle"]),
        source_frame=str(record["source_frame"]),
        neutral_projection={"location_record_available": True},
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint_without_fingerprint(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("resource_symbol", ["xarm6", "ur5e"])
def test_current_completion_uses_reviewed_destinations_and_reachability(tmp_path, resource_symbol):
    completion = persist_native_completion_fixture(
        tmp_path, resource_symbol=resource_symbol
    ).to_record()
    assert "schema_version" not in completion
    assert completion["validation_scope"] == "moveit_state_location_reachability"
    assert completion["motion_validation_performed"] is True
    assert "robot_agent_validation_ref" not in completion
    selection = _read_json(tmp_path / completion["resource_selection_ref"])
    assert "schema_version" not in selection
    assert selection["selected_resource_symbol"] == resource_symbol
    assert len(selection["state_locations"]["current_state"]) == 2
    assert len(selection["state_locations"]["desired_state"]) == 1
    proposal = _read_json(tmp_path / completion["ontology_projection_ref"])
    association = proposal["output"]["target_feature"]["assembly_feature_association"][0]
    assert association["state_names"] == ["desired_state"]
    assert {item["state_name"] for item in association["assembly_features"]} == {"current_state"}
    snapshot = validate_grounding_evidence(tmp_path, proposal)
    assert snapshot.product_requirement == completion["product_requirement"]
    assert set(proposal["grounding_evidence"]) == {"source_refs", "input_artifacts"}



@pytest.mark.parametrize(
    "artifact",
    ["proposal", "snapshot", "evidence", "request", "reachability", "capability"],
)
def test_current_completion_rejects_changed_lineage(tmp_path, artifact):
    completion = persist_native_completion_fixture(tmp_path).to_record()
    proposal = _read_json(tmp_path / completion["ontology_projection_ref"])
    snapshot_ref = next(
        item["ref"] for item in proposal["grounding_evidence"]["input_artifacts"]
        if item["ref"].startswith("products/grounding/product_context/")
    )
    refs = {
        "proposal": completion["ontology_projection_ref"],
        "snapshot": snapshot_ref,
        "evidence": completion["typed_context_refs"][0]["ref"],
        "request": "products/grounding/ontology_grounding/request_0001_0001.json",
        "reachability": completion["reachability_check_ref"],
        "capability": completion["workcell_snapshot_ref"],
    }
    path = tmp_path / refs[artifact]
    payload = _read_json(path)
    payload["altered"] = True
    _write_json(path, payload)
    with pytest.raises(GroundingContractError):
        load_pa_context_grounding_completion(tmp_path)


@pytest.mark.parametrize("change", ["omit_arm", "other_locations", "check_hash", "old_shape"])
def test_completion_reload_rechecks_all_arm_evidence(tmp_path, change):
    completion = persist_native_completion_fixture(tmp_path).to_record()
    completion_path = tmp_path / "interaction_record/context_completion_0001.json"
    other = next(
        item
        for item in completion["tool_call_refs"]
        if _read_json(tmp_path / item["ref"])["arguments"]["resource_symbol"] == "ur5e"
    )
    call_path = tmp_path / other["ref"]
    call = _read_json(call_path)
    if change == "omit_arm":
        completion["tool_call_refs"].remove(other)
    else:
        check_path = tmp_path / call["result_ref"]
        check = _read_json(check_path)
        if change == "check_hash":
            check["status"] = "rejected"
        elif change == "other_locations":
            check["state_locations"]["desired_state"] = check["state_locations"]["current_state"]
        else:
            check["schema_version"] = 5
        check["fingerprint"] = _fingerprint_without_fingerprint(check)
        _write_json(check_path, check)
        call["result_sha256"] = hashlib.sha256(check_path.read_bytes()).hexdigest()
        _write_json(call_path, call)
        other["sha256"] = hashlib.sha256(call_path.read_bytes()).hexdigest()
    completion["fingerprint"] = _fingerprint_without_fingerprint(completion)
    _write_json(completion_path, completion)
    with pytest.raises(GroundingContractError):
        load_pa_context_grounding_completion(tmp_path)


@pytest.mark.parametrize("other_status", ["accepted", "rejected", "needs_context"])
def test_completion_reloads_all_arm_results_without_reclassifying_unavailable(
    tmp_path, other_status
):
    from cais_spade_llm.spec2primitives import spec2primitives_ui

    completion = persist_native_completion_fixture(
        tmp_path, planning_statuses={"ur5e": other_status}
    )
    reloaded = load_pa_context_grounding_completion(tmp_path)
    assert reloaded == completion
    result = spec2primitives_ui._final_grounding_result(tmp_path, reloaded.to_record())
    rows = {row["resource"]: row for row in result["resource_comparison"]}
    expected = {
        "accepted": "Reachable",
        "rejected": "Planning rejected",
        "needs_context": "Unavailable",
    }[other_status]
    assert rows["ur5e"]["current_state"] == rows["ur5e"]["desired_state"] == expected
    assert rows["ur5e"]["selected"] == ""
    assert rows["xarm6"]["selected"] == "Selected"
    # A reviewed-only UI has no completion authority and reports an omitted check explicitly.
    call_refs = [
        item["ref"]
        for item in reloaded.to_record()["tool_call_refs"]
        if json.loads((tmp_path / item["ref"]).read_text())["arguments"]["resource_symbol"]
        == "xarm6"
    ]
    pending = {
        row["resource"]: row
        for row in spec2primitives_ui._resource_comparison(tmp_path, tool_refs=call_refs)
    }
    assert pending["ur5e"]["current_state"] == pending["ur5e"]["desired_state"] == "Unchecked"


def test_new_record_chain_contains_no_format_version_fields(tmp_path):
    persist_native_completion_fixture(tmp_path)

    def check(value):
        if isinstance(value, dict):
            assert "schema_version" not in value
            for nested in value.values():
                check(nested)
        elif isinstance(value, list):
            for nested in value:
                check(nested)

    for path in tmp_path.rglob("*.json"):
        check(json.loads(path.read_text()))
    for path in (Path(__file__).parents[1] / "config").glob("*.json"):
        check(json.loads(path.read_text()))


@pytest.mark.parametrize(
    "change",
    [
        "empty_statement",
        "extra_state_field",
        "feature_identity",
        "process",
        "compiled_delta",
        "uncertainty",
        "source_coverage",
        "artifact_coverage",
        "snapshot_coverage",
        "old_review",
    ],
)
def test_direct_grounding_gate_rejects_changed_contract_without_completion_hash(tmp_path, change):
    completion = persist_native_completion_fixture(tmp_path).to_record()
    proposal = _read_json(tmp_path / completion["ontology_projection_ref"])
    target = proposal["output"]["target_feature"]
    if change == "empty_statement":
        target["desired_state"]["statement"]["text"] = ""
    elif change == "extra_state_field":
        target["desired_state"]["extra"] = True
    elif change == "feature_identity":
        proposal["feature_iri"] += "_changed"
    elif change == "process":
        target["required_process"]["process_iri"] += "_changed"
    elif change == "compiled_delta":
        proposal["compiled_delta"]["assertions"].pop()
    elif change == "uncertainty":
        proposal["compiled_delta"]["uncertainty"] = [{"description": "Invented certainty."}]
    elif change == "source_coverage":
        proposal["grounding_evidence"]["source_refs"].pop()
    elif change == "artifact_coverage":
        artifacts = proposal["grounding_evidence"]["input_artifacts"]
        artifacts.remove(
            next(item for item in artifacts if item["ref"].endswith("selected_rgb.png"))
        )
    elif change == "snapshot_coverage":
        artifacts = proposal["grounding_evidence"]["input_artifacts"]
        artifacts[:] = [
            item
            for item in artifacts
            if not item["ref"].startswith("products/grounding/product_context/")
        ]
    else:
        proposal.pop("grounding_evidence")
        proposal["semantic_review_ref"] = (
            "products/grounding/target_feature_review/review_0001.json"
        )
    with pytest.raises(GroundingContractError):
        validate_grounding_evidence(tmp_path, proposal)


def test_grounding_snapshot_cannot_drop_original_binding_even_with_updated_hash(tmp_path):
    completion = persist_native_completion_fixture(
        tmp_path, include_dynamic_grounding_records=True
    ).to_record()
    proposal = _read_json(tmp_path / completion["ontology_projection_ref"])
    entry = next(
        item
        for item in proposal["grounding_evidence"]["input_artifacts"]
        if item["ref"].startswith("products/grounding/product_context/")
    )
    path = tmp_path / entry["ref"]
    snapshot = _read_json(path)
    snapshot["typed_bindings"].pop()
    snapshot["fingerprint"] = _fingerprint_without_fingerprint(snapshot)
    _write_json(path, snapshot)
    proposal["grounding_evidence"], _ = build_grounding_evidence(
        tmp_path,
        context_view_ref=entry["ref"],
        target_feature=proposal["output"]["target_feature"],
        proposal_number=proposal["proposal_number"],
    )
    with pytest.raises(GroundingContractError, match="dropped original typed evidence"):
        validate_grounding_evidence(tmp_path, proposal)


def test_grounding_provenance_pins_recursive_rgb_artifacts(tmp_path):
    completion = persist_native_completion_fixture(tmp_path).to_record()
    proposal = _read_json(tmp_path / completion["ontology_projection_ref"])
    entry = next(
        item
        for item in proposal["grounding_evidence"]["input_artifacts"]
        if item["ref"].endswith("selected_rgb.png")
    )
    path = tmp_path / entry["ref"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(GroundingContractError, match="artifact changed"):
        validate_grounding_evidence(tmp_path, proposal)


def test_source_caveat_survives_acceptance_without_semantic_review(tmp_path, monkeypatch):
    import sys

    original = _write_location
    caveat = {
        "description": "Fixture observed position remains uncertain.",
        "evidence_refs": ["requirement_0001"],
    }

    def with_caveat(*args, **kwargs):
        path, ref = original(*args, **kwargs)
        record = _read_json(path)
        record["evidence_refs"] = ["requirement_0001"]
        record["uncertainty"] = [caveat]
        _write_json(path, record)
        return path, ref

    monkeypatch.setattr(sys.modules[__name__], "_write_location", with_caveat)
    completion = persist_native_completion_fixture(tmp_path).to_record()
    proposal = _read_json(tmp_path / completion["ontology_projection_ref"])
    assert proposal["compiled_delta"]["uncertainty"] == [caveat]
    validate_grounding_evidence(tmp_path, proposal)
    assert not (tmp_path / "products/grounding/target_feature_review").exists()
    proposal["compiled_delta"]["uncertainty"] = []
    with pytest.raises(GroundingContractError, match="source uncertainty changed"):
        validate_grounding_evidence(tmp_path, proposal)

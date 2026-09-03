"""Validate one PA-authored target feature and compile its minimal ABox view."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdflib import RDF

from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    ABoxSnapshot,
    MergeResult,
    validate_and_merge_triple_delta,
    validate_triple_delta,
)
from cais_spade_llm.spec2primitives.ontology import (
    PredefinedWorkcellSnapshot,
    TBoxSnapshot,
)

_PRODUCER = "ontology_grounding"
_PROPOSAL_ROOT = Path("products/grounding/ontology_grounding")
_REVIEW_ROOT = Path("products/grounding/target_feature_review")
_PROPOSAL_KEYS = {"target_feature"}
_TARGET_FEATURE_KEYS = {"required_process", "current_state", "desired_state"}
_REQUIRED_PROCESS_KEYS = {"process_iri", "evidence_refs"}
_STATE_KEYS = {"statement", "state_values"}
_STATEMENT_KEYS = {"text", "evidence_refs"}
_STATE_VALUE_KEYS = {"name", "value_ref", "evidence_refs"}
_VALUE_REF_KEYS = {"record_ref", "field_path"}

TypedRecordResolver = Callable[[str], Mapping[str, object]]
PAReferenceTranslator = Callable[[str], str]


class OntologyGroundingError(ValueError):
    """Raised when an untrusted target-feature proposal cannot be accepted."""


@dataclass(frozen=True)
class OntologyGroundingProposal:
    """Hold one validated PA-authored target feature and transient value projection."""

    target_feature: Mapping[str, object]
    feature_iri: str
    resolved_state_values: tuple[Mapping[str, object], ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class TargetFeatureSemanticReview:
    """Pin one separate semantic sufficiency review without private reasoning."""

    review_path: Path
    review_number: int
    proposal_number: int
    target_feature_fingerprint: str
    evidence_refs: tuple[str, ...]
    verdict: str
    gap: str | None
    fingerprint: str


@dataclass(frozen=True)
class OntologyGroundingResult:
    """Return accepted assertions and the persisted target-feature proposal."""

    merge: MergeResult
    proposal_path: Path
    proposal: OntologyGroundingProposal
    semantic_review: TargetFeatureSemanticReview


@dataclass(frozen=True)
class OntologyGroundingCandidate:
    """Hold one valid but uncommitted PA target-feature proposal."""

    provisional_abox: ABoxSnapshot
    proposal_path: Path
    proposal_number: int
    output: Mapping[str, object]
    compiled_delta: Mapping[str, object]
    proposal: OntologyGroundingProposal


@dataclass(frozen=True)
class OntologyGroundingInterruption:
    """Return a direct ProductAgent clarification or insufficiency result."""

    kind: str
    message: str


async def propose_and_validate_ontology_grounding(  # noqa: PLR0913
    product_agent: ProductAgentContextRuntime,
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    workcell: PredefinedWorkcellSnapshot,
    evidence_catalog: Sequence[Mapping[str, object]],
    authorized_evidence_refs: set[str],
    tools: list[dict[str, Any]],
    tool_executor: Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]],
    max_tool_rounds: int,
    required_output_projection: Mapping[str, object],
    validation_feedback_history: Sequence[Mapping[str, object]] = (),
    typed_record_resolver: TypedRecordResolver | None = None,
    pa_reference_resolver: PAReferenceTranslator | None = None,
) -> OntologyGroundingCandidate | OntologyGroundingInterruption:
    """Let PA investigate and return one valid, uncommitted target feature."""
    root = Path(interaction_root).resolve()
    _validate_authorities(root, tbox, abox, workcell)
    feature_class_iri = f"{tbox.ppr_namespace}feature"
    defines_iri = f"{tbox.ppr_namespace}defines"
    realizes_iri = f"{tbox.ppr_namespace}realizes"
    if feature_class_iri not in tbox.classes:
        raise OntologyGroundingError(
            "The authoritative TBox does not expose the required feature class."
        )
    if not {defines_iri, realizes_iri}.issubset(tbox.object_properties):
        raise OntologyGroundingError(
            "The authoritative TBox does not expose the task-grounding properties."
        )

    prompt_input: dict[str, object] = {
        "exact_requirement": abox.product_requirement,
        "approved_evidence_catalog": [dict(item) for item in evidence_catalog],
        "required_output_projection": dict(required_output_projection),
        "initialized_specification_iri": abox.specification_iri,
        "authorized_processes": [
            {
                "process_symbol": process_symbol,
                "process_iri": process_iri,
            }
            for process_symbol, process_iri in workcell.processes
        ],
        "target_feature_contract": {
            "cardinality": "exactly_one",
            "state_value_cardinality": "follow_required_output_projection",
            "value_names": "PA_authored_without_host_enum",
        },
    }
    if validation_feedback_history:
        prompt_input["validation_feedback_history"] = [
            dict(item) for item in validation_feedback_history
        ]
    prompt = (
        "Investigate the exact product requirement using only the requirement and "
        "evidence returned by the controlled tools. Catalog metadata is "
        "discovery-only. Never use hidden case knowledge, evaluator information, or "
        "an interpretation supplied by these instructions. Before asking a "
        "requirement-meaning clarification, retrieve "
        "and consider every approved entry plausibly relevant to that ambiguity. "
        "Return exactly one result: one target_feature proposal, one "
        "clarification_question, or one insufficient_evidence result.\n\n"
        "For a proposal, author exactly one semantic target_feature. Choose exactly "
        "one process_iri from authorized_processes and cite its direct evidence. "
        "Write complete current and desired product-state statements and cite their "
        "direct evidence. The current_state must describe the evidenced state now; "
        "the desired_state must describe the requested product outcome. Follow only "
        "the generic cardinality and readiness constraints in required_output_projection; "
        "they do not identify a source, component, candidate, or expected answer. Add "
        "state_values only when accepted typed records returned by controlled tools "
        "uniquely support values belonging to those states. For "
        "each state value, author a unique semantic name within that state, exact "
        "record_ref, JSON Pointer field_path, and direct evidence_refs. The same "
        "record may supply multiple values through different paths. Do not invent a "
        "record, path, value, citation, provider, retrieval method, target pose, or "
        "host-defined value-name vocabulary. Do not infer a current/desired role "
        "from sensor names, camera identity, candidate order, or provider metadata; "
        "assign state values only from the approved evidence. The required_output_projection "
        "is separate controller-owned grounding readiness; it does not automatically "
        "belong in target_feature. Do not output ontology individuals, relations, "
        "literal facts, context_summary, global evidence_refs, missing_information, "
        "primitive choices, resource choices, or execution details. The controller "
        "creates feature_0001, both state individuals, and the exact ontology "
        "assertions after validation. "
        "Cite only requirement_0001 or evidence refs actually returned by a controlled "
        "tool. Preserve and address every distinct item in validation_feedback_history.\n\n"
        f"Grounding input:\n{json.dumps(prompt_input, indent=2, ensure_ascii=False)}"
    )
    proposal_number = _next_number(root / _PROPOSAL_ROOT, "proposal_")
    proposal_path = root / _PROPOSAL_ROOT / f"proposal_{proposal_number:04d}.json"
    transport_output = await product_agent.ask_llm_structured(
        prompt,
        response_format=_proposal_response_format(
            tuple(process_iri for _process_symbol, process_iri in workcell.processes)
        ),
        tools=tools,
        tool_executor=tool_executor,
        max_tool_rounds=max_tool_rounds,
    )
    if (
        not isinstance(transport_output, Mapping)
        or set(transport_output) != {"result"}
        or not isinstance(transport_output["result"], Mapping)
    ):
        raise OntologyGroundingError("ProductAgent grounding result wrapper is invalid.")
    output = transport_output["result"]
    interruption = _validated_interruption(output)
    if interruption is not None:
        return interruption

    canonical_output = _json_clone(output)
    try:
        if pa_reference_resolver is not None:
            canonical_output = _translate_pa_references(
                output,
                pa_reference_resolver,
            )
        proposal = _validated_proposal(
            canonical_output,
            abox=abox,
            workcell=workcell,
            authorized_evidence_refs=authorized_evidence_refs,
            typed_record_resolver=typed_record_resolver,
        )
        delta = _compile_proposal_delta(proposal, abox=abox, tbox=tbox)
        validated_delta = validate_triple_delta(
            root,
            tbox,
            delta,
            authorized_evidence_refs=sorted(authorized_evidence_refs),
            authorized_external_process_iris={
                process_iri for _process_symbol, process_iri in workcell.processes
            },
        )
    except (KeyError, OntologyGroundingError, TypeError, ValueError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
        _write_proposal_record(
            proposal_path,
            proposal_number=proposal_number,
            specification_iri=abox.specification_iri,
            feature_iri=f"{abox.namespace}feature_0001",
            output=canonical_output,
            compiled_delta=None,
            semantic_review=None,
            status="rejected",
            failure=failure,
        )
        raise OntologyGroundingError(f"OntologyGroundingProposal is invalid: {failure}") from exc

    return OntologyGroundingCandidate(
        provisional_abox=validated_delta.abox,
        proposal_path=proposal_path,
        proposal_number=proposal_number,
        output=_json_clone(canonical_output),
        compiled_delta=_json_clone(delta),
        proposal=proposal,
    )


async def review_target_feature_semantics(
    product_agent: ProductAgentContextRuntime,
    *,
    interaction_root: Path,
    candidate: OntologyGroundingCandidate,
    product_requirement: str,
    evidence_catalog: Sequence[Mapping[str, object]],
    pa_reference_projector: PAReferenceTranslator | None = None,
) -> TargetFeatureSemanticReview:
    """Run and persist a scaffold-free semantic consistency review."""
    root = Path(interaction_root).resolve()
    if (
        candidate.proposal_path
        != root / _PROPOSAL_ROOT / f"proposal_{candidate.proposal_number:04d}.json"
        or candidate.proposal_path.exists()
        or not isinstance(product_requirement, str)
        or not product_requirement.strip()
    ):
        raise OntologyGroundingError(
            "Target-feature review inputs do not match the uncommitted proposal."
        )
    target_feature_projection = (
        _translate_pa_references(
            candidate.proposal.target_feature,
            pa_reference_projector,
        )
        if pa_reference_projector is not None
        else _json_clone(candidate.proposal.target_feature)
    )
    state_value_projection = (
        _translate_pa_references(
            list(candidate.proposal.resolved_state_values),
            pa_reference_projector,
        )
        if pa_reference_projector is not None
        else list(candidate.proposal.resolved_state_values)
    )
    prompt_input = {
        "exact_requirement": product_requirement,
        "target_feature": target_feature_projection,
        "resolved_state_values": state_value_projection,
        "currently_retrieved_evidence": [dict(item) for item in evidence_catalog],
    }
    prompt = (
        "Perform a scaffold-free consistency review of the PA-authored target_feature "
        "against the exact product requirement and currently retrieved evidence. The "
        "candidate bindings and evidence uniqueness belong to deterministic validation; "
        "do not repeat that gate, infer missing facts, repair the proposal, replace a "
        "candidate, or apply an expected answer. Judge only whether the "
        "current-state statement, desired-state statement, process, and included values "
        "are mutually consistent with the requirement and cited evidence. Do not demand "
        "target geometry, target_pose, tolerance, primitive parameters, resource state, "
        "or other execution information. Return only a verdict and a concise actionable "
        "consistency gap; do not return reasoning.\n\n"
        f"Semantic review input:\n{json.dumps(prompt_input, indent=2, ensure_ascii=False)}"
    )
    output = await product_agent.ask_llm_structured(
        prompt,
        response_format=_semantic_review_response_format(),
        tools=None,
        tool_executor=None,
        max_tool_rounds=1,
    )
    verdict, gap = _validated_review_output(output)
    review_number = _next_number(root / _REVIEW_ROOT, "review_")
    review_path = root / _REVIEW_ROOT / f"review_{review_number:04d}.json"
    target_fingerprint = target_feature_fingerprint(candidate.proposal.target_feature)
    record: dict[str, object] = {
        "schema_version": 2,
        "record_type": "TargetFeatureSemanticReview",
        "review_number": review_number,
        "proposal_number": candidate.proposal_number,
        "target_feature_fingerprint": target_fingerprint,
        "evidence_refs": list(candidate.proposal.evidence_refs),
        "verdict": verdict,
        "gap": gap,
        "reviewed_at_ns": time.time_ns(),
    }
    fingerprint = _fingerprint(record)
    record["fingerprint"] = fingerprint
    _write_json_exclusive(review_path, record)
    return TargetFeatureSemanticReview(
        review_path=review_path,
        review_number=review_number,
        proposal_number=candidate.proposal_number,
        target_feature_fingerprint=target_fingerprint,
        evidence_refs=candidate.proposal.evidence_refs,
        verdict=verdict,
        gap=gap,
        fingerprint=fingerprint,
    )


def reject_ontology_grounding_candidate(
    candidate: OntologyGroundingCandidate,
    *,
    interaction_root: Path,
    abox: ABoxSnapshot,
    semantic_review: TargetFeatureSemanticReview,
) -> Path:
    """Persist an uncommitted proposal rejected by its semantic review."""
    root = Path(interaction_root).resolve()
    _validate_candidate_and_review(candidate, semantic_review, root, abox)
    if semantic_review.verdict != "incomplete" or not semantic_review.gap:
        raise OntologyGroundingError("Only an incomplete semantic review can reject a candidate.")
    _write_proposal_record(
        candidate.proposal_path,
        proposal_number=candidate.proposal_number,
        specification_iri=abox.specification_iri,
        feature_iri=candidate.proposal.feature_iri,
        output=candidate.output,
        compiled_delta=candidate.compiled_delta,
        semantic_review=semantic_review,
        status="rejected",
        failure=f"TargetFeatureSemanticReview: {semantic_review.gap}",
    )
    return candidate.proposal_path


def commit_ontology_grounding_candidate(
    candidate: OntologyGroundingCandidate,
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    workcell: PredefinedWorkcellSnapshot,
    authorized_evidence_refs: set[str],
    semantic_review: TargetFeatureSemanticReview,
) -> OntologyGroundingResult:
    """Revalidate and persist one semantically reviewed target feature."""
    root = Path(interaction_root).resolve()
    _validate_authorities(root, tbox, abox, workcell)
    _validate_candidate_and_review(candidate, semantic_review, root, abox)
    if semantic_review.verdict != "complete" or semantic_review.gap is not None:
        raise OntologyGroundingError(
            "Ontology commit requires a complete target-feature semantic review."
        )
    if not set(candidate.proposal.evidence_refs).issubset(authorized_evidence_refs):
        raise OntologyGroundingError("Ontology grounding evidence authority changed before commit.")
    merge = validate_and_merge_triple_delta(
        root,
        tbox,
        _PRODUCER,
        candidate.compiled_delta,
        authorized_evidence_refs=sorted(authorized_evidence_refs),
        authorized_external_process_iris={
            process_iri for _process_symbol, process_iri in workcell.processes
        },
    )
    _write_proposal_record(
        candidate.proposal_path,
        proposal_number=candidate.proposal_number,
        specification_iri=abox.specification_iri,
        feature_iri=candidate.proposal.feature_iri,
        output=candidate.output,
        compiled_delta=candidate.compiled_delta,
        semantic_review=semantic_review,
        status="accepted",
        failure=None,
    )
    return OntologyGroundingResult(
        merge=merge,
        proposal_path=candidate.proposal_path,
        proposal=candidate.proposal,
        semantic_review=semantic_review,
    )


def resolve_json_pointer(document: object, field_path: str) -> object:
    """Resolve one non-root RFC 6901 JSON Pointer against an exact record."""
    if not isinstance(field_path, str) or not field_path.startswith("/"):
        raise OntologyGroundingError("state value field_path must be a non-root JSON Pointer.")
    current = document
    for raw_token in field_path.split("/")[1:]:
        token = _decode_pointer_token(raw_token)
        if isinstance(current, Mapping):
            if token not in current:
                raise OntologyGroundingError(
                    f"state value field_path does not exist: {field_path}."
                )
            current = current[token]
            continue
        if isinstance(current, list):
            if token == "-" or not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise OntologyGroundingError(
                    f"state value field_path has an invalid array index: {field_path}."
                )
            index = int(token)
            if index >= len(current):
                raise OntologyGroundingError(
                    f"state value field_path does not exist: {field_path}."
                )
            current = current[index]
            continue
        raise OntologyGroundingError(f"state value field_path traverses a scalar: {field_path}.")
    return current


def bounded_value_projection(value: object, *, depth: int = 0) -> object:
    """Return a deterministic bounded JSON projection for PA review and RA input."""
    if depth >= 5:
        return {"projection_truncated": True}
    if isinstance(value, str):
        return value if len(value) <= 2048 else f"{value[:2048]}…"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        projection = {
            str(key): bounded_value_projection(item, depth=depth + 1) for key, item in items[:32]
        }
        if len(items) > 32:
            projection["projection_truncated"] = True
        return projection
    if isinstance(value, (list, tuple)):
        projection = [bounded_value_projection(item, depth=depth + 1) for item in value[:32]]
        if len(value) > 32:
            projection.append({"projection_truncated": True})
        return projection
    raise OntologyGroundingError("Resolved state value is not JSON-compatible.")


def target_feature_fingerprint(target_feature: Mapping[str, object]) -> str:
    """Return the canonical fingerprint of one PA-authored target feature."""
    return _fingerprint(target_feature)


def target_feature_evidence_refs(target_feature: Mapping[str, object]) -> tuple[str, ...]:
    """Return all direct nested citations in deterministic first-use order."""
    refs: list[str] = []
    required_process = target_feature.get("required_process")
    if isinstance(required_process, Mapping):
        _append_string_refs(refs, required_process.get("evidence_refs"))
    for state_name in ("current_state", "desired_state"):
        state = target_feature.get(state_name)
        if not isinstance(state, Mapping):
            continue
        statement = state.get("statement")
        if isinstance(statement, Mapping):
            _append_string_refs(refs, statement.get("evidence_refs"))
        state_values = state.get("state_values")
        if isinstance(state_values, list):
            for item in state_values:
                if isinstance(item, Mapping):
                    _append_string_refs(refs, item.get("evidence_refs"))
    return tuple(refs)


def _validated_proposal(
    value: object,
    *,
    abox: ABoxSnapshot,
    workcell: PredefinedWorkcellSnapshot,
    authorized_evidence_refs: set[str],
    typed_record_resolver: TypedRecordResolver | None,
) -> OntologyGroundingProposal:
    if not isinstance(value, Mapping) or set(value) != _PROPOSAL_KEYS:
        raise OntologyGroundingError("OntologyGroundingProposal fields are invalid.")
    target_feature = value["target_feature"]
    if not isinstance(target_feature, Mapping) or set(target_feature) != _TARGET_FEATURE_KEYS:
        raise OntologyGroundingError("target_feature fields are invalid.")
    required_process = target_feature["required_process"]
    authorized_process_iris = {process_iri for _process_symbol, process_iri in workcell.processes}
    if (
        not isinstance(required_process, Mapping)
        or set(required_process) != _REQUIRED_PROCESS_KEYS
        or required_process.get("process_iri") not in authorized_process_iris
    ):
        raise OntologyGroundingError(
            "target_feature required_process is not an authorized process."
        )
    _validated_evidence_refs(
        required_process.get("evidence_refs"),
        "required_process.evidence_refs",
        authorized_evidence_refs,
    )
    resolved_values: list[Mapping[str, object]] = []
    for state_name in ("current_state", "desired_state"):
        resolved_values.extend(
            _validated_state(
                state_name,
                target_feature[state_name],
                authorized_evidence_refs=authorized_evidence_refs,
                typed_record_resolver=typed_record_resolver,
            )
        )
    cloned_target = _json_clone(target_feature)
    evidence_refs = tuple(dict.fromkeys(target_feature_evidence_refs(cloned_target)))
    return OntologyGroundingProposal(
        target_feature=cloned_target,
        feature_iri=f"{abox.namespace}feature_0001",
        resolved_state_values=tuple(resolved_values),
        evidence_refs=evidence_refs,
    )


def _validated_state(  # noqa: C901
    state_name: str,
    value: object,
    *,
    authorized_evidence_refs: set[str],
    typed_record_resolver: TypedRecordResolver | None,
) -> list[Mapping[str, object]]:
    """Validate one PA-authored feature state and resolve its cited values."""
    if not isinstance(value, Mapping) or set(value) != _STATE_KEYS:
        raise OntologyGroundingError(f"target_feature {state_name} fields are invalid.")
    statement = value["statement"]
    if not isinstance(statement, Mapping) or set(statement) != _STATEMENT_KEYS:
        raise OntologyGroundingError(f"{state_name} statement fields are invalid.")
    text = statement.get("text")
    if not isinstance(text, str) or not text.strip():
        raise OntologyGroundingError(f"{state_name} statement text is empty.")
    _validated_evidence_refs(
        statement.get("evidence_refs"),
        f"{state_name}.statement.evidence_refs",
        authorized_evidence_refs,
    )
    state_values = value["state_values"]
    if not isinstance(state_values, list) or not all(
        isinstance(item, Mapping) for item in state_values
    ):
        raise OntologyGroundingError(f"{state_name} state_values must be a list.")
    names: set[str] = set()
    resolved_values: list[Mapping[str, object]] = []
    for index, item in enumerate(state_values):
        if set(item) != _STATE_VALUE_KEYS:
            raise OntologyGroundingError(f"{state_name}.state_values[{index}] fields are invalid.")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise OntologyGroundingError(f"{state_name}.state_values[{index}].name is empty.")
        if name in names:
            raise OntologyGroundingError(f"{state_name} state value names must be unique.")
        names.add(name)
        _validated_evidence_refs(
            item.get("evidence_refs"),
            f"{state_name}.state_values[{index}].evidence_refs",
            authorized_evidence_refs,
        )
        value_ref = item.get("value_ref")
        if not isinstance(value_ref, Mapping) or set(value_ref) != _VALUE_REF_KEYS:
            raise OntologyGroundingError(
                f"{state_name}.state_values[{index}].value_ref fields are invalid."
            )
        record_ref = value_ref.get("record_ref")
        field_path = value_ref.get("field_path")
        if not isinstance(record_ref, str) or not record_ref:
            raise OntologyGroundingError(
                f"{state_name}.state_values[{index}].value_ref.record_ref is invalid."
            )
        if typed_record_resolver is None:
            raise OntologyGroundingError("state values require an accepted typed-record resolver.")
        try:
            resolved_record = typed_record_resolver(record_ref)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise OntologyGroundingError(
                f"{state_name}.state_values[{index}] record_ref is not an accepted typed binding."
            ) from exc
        if not isinstance(resolved_record, Mapping) or set(resolved_record) != {
            "record_type",
            "record_sha256",
            "record",
        }:
            raise OntologyGroundingError("Typed-record resolver result is invalid.")
        record_type = resolved_record["record_type"]
        record_sha256 = resolved_record["record_sha256"]
        record = resolved_record["record"]
        if (
            not isinstance(record_type, str)
            or not record_type
            or not isinstance(record_sha256, str)
            or len(record_sha256) != 64
            or not isinstance(record, Mapping)
        ):
            raise OntologyGroundingError("Typed-record resolver result is invalid.")
        resolved_value = resolve_json_pointer(record, field_path)  # type: ignore[arg-type]
        if _is_empty_resolved_value(resolved_value):
            raise OntologyGroundingError(
                f"{state_name}.state_values[{index}] resolves to an empty value."
            )
        resolved_values.append(
            {
                "state": state_name,
                "name": name,
                "value_ref": _json_clone(value_ref),
                "record_type": record_type,
                "record_sha256": record_sha256,
                "resolved_value": bounded_value_projection(resolved_value),
            }
        )
    return resolved_values


def _compile_proposal_delta(
    proposal: OntologyGroundingProposal,
    *,
    abox: ABoxSnapshot,
    tbox: TBoxSnapshot,
) -> dict[str, object]:
    target = proposal.target_feature
    process = target["required_process"]
    current_state = target["current_state"]
    desired_state = target["desired_state"]
    assert isinstance(process, Mapping)
    assert isinstance(current_state, Mapping)
    assert isinstance(desired_state, Mapping)
    current_statement = current_state["statement"]
    desired_statement = desired_state["statement"]
    assert isinstance(current_statement, Mapping)
    assert isinstance(desired_statement, Mapping)
    current_refs = list(current_statement["evidence_refs"])
    desired_refs = list(desired_statement["evidence_refs"])
    process_refs = list(process["evidence_refs"])
    current_state_iri = f"{abox.namespace}currentstate_0001"
    desired_state_iri = f"{abox.namespace}desiredstate_0001"
    return {
        "assertions": [
            {
                "subject": proposal.feature_iri,
                "predicate": str(RDF.type),
                "object": {
                    "kind": "iri",
                    "value": f"{tbox.ppr_namespace}feature",
                },
                "evidence_refs": desired_refs,
            },
            {
                "subject": abox.specification_iri,
                "predicate": f"{tbox.ppr_namespace}defines",
                "object": {"kind": "iri", "value": proposal.feature_iri},
                "evidence_refs": desired_refs,
            },
            {
                "subject": str(process["process_iri"]),
                "predicate": f"{tbox.ppr_namespace}realizes",
                "object": {"kind": "iri", "value": proposal.feature_iri},
                "evidence_refs": process_refs,
            },
            {
                "subject": current_state_iri,
                "predicate": str(RDF.type),
                "object": {
                    "kind": "iri",
                    "value": f"{tbox.ppr_namespace}state",
                },
                "evidence_refs": current_refs,
            },
            {
                "subject": desired_state_iri,
                "predicate": str(RDF.type),
                "object": {
                    "kind": "iri",
                    "value": f"{tbox.ppr_namespace}state",
                },
                "evidence_refs": desired_refs,
            },
            {
                "subject": proposal.feature_iri,
                "predicate": f"{tbox.ppr_namespace}hascurrentstate",
                "object": {"kind": "iri", "value": current_state_iri},
                "evidence_refs": current_refs,
            },
            {
                "subject": proposal.feature_iri,
                "predicate": f"{tbox.ppr_namespace}hasdesiredstate",
                "object": {"kind": "iri", "value": desired_state_iri},
                "evidence_refs": desired_refs,
            },
        ],
        "uncertainty": [],
        "unresolved_evidence_needs": [],
        "typed_context_refs": [],
    }


def _proposal_response_format(process_iris: tuple[str, ...]) -> dict[str, Any]:
    evidence_refs = {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "minItems": 1,
    }
    state = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_STATE_KEYS),
        "properties": {
            "statement": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_STATEMENT_KEYS),
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "evidence_refs": evidence_refs,
                },
            },
            "state_values": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(_STATE_VALUE_KEYS),
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "value_ref": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": sorted(_VALUE_REF_KEYS),
                            "properties": {
                                "record_ref": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "field_path": {
                                    "type": "string",
                                    "pattern": "^/",
                                },
                            },
                        },
                        "evidence_refs": evidence_refs,
                    },
                },
            },
        },
    }
    target_feature = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_TARGET_FEATURE_KEYS),
        "properties": {
            "required_process": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_REQUIRED_PROCESS_KEYS),
                "properties": {
                    "process_iri": {"type": "string", "enum": list(process_iris)},
                    "evidence_refs": evidence_refs,
                },
            },
            "current_state": state,
            "desired_state": state,
        },
    }
    return {
        "name": "spec2primitives_grounding_result",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["result"],
            "properties": {
                "result": {
                    "anyOf": [
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["target_feature"],
                            "properties": {"target_feature": target_feature},
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["clarification_question"],
                            "properties": {
                                "clarification_question": {
                                    "type": "string",
                                    "minLength": 1,
                                }
                            },
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["insufficient_evidence"],
                            "properties": {
                                "insufficient_evidence": {
                                    "type": "string",
                                    "minLength": 1,
                                }
                            },
                        },
                    ]
                }
            },
        },
    }


def _semantic_review_response_format() -> dict[str, Any]:
    return {
        "name": "spec2primitives_target_feature_review",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["verdict", "gap"],
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["complete", "incomplete"],
                },
                "gap": {"type": ["string", "null"]},
            },
        },
    }


def _validated_interruption(
    output: Mapping[str, object],
) -> OntologyGroundingInterruption | None:
    for kind in ("clarification_question", "insufficient_evidence"):
        if set(output) == {kind}:
            message = output[kind]
            if not isinstance(message, str) or not message.strip():
                raise OntologyGroundingError(f"{kind} is invalid.")
            return OntologyGroundingInterruption(kind, message)
    return None


def _validated_review_output(output: object) -> tuple[str, str | None]:
    if not isinstance(output, Mapping) or set(output) != {"verdict", "gap"}:
        raise OntologyGroundingError("TargetFeatureSemanticReview output is invalid.")
    verdict = output["verdict"]
    gap = output["gap"]
    if verdict == "complete" and gap is None:
        return "complete", None
    if verdict == "incomplete" and isinstance(gap, str) and gap.strip():
        return "incomplete", gap
    raise OntologyGroundingError("TargetFeatureSemanticReview verdict and gap are inconsistent.")


def _validate_authorities(
    root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    workcell: PredefinedWorkcellSnapshot,
) -> None:
    if abox.interaction_root != root or abox.tbox_fingerprint != tbox.fingerprint:
        raise OntologyGroundingError(
            "Ontology grounding inputs do not share one interaction and TBox."
        )
    if not isinstance(workcell, PredefinedWorkcellSnapshot):
        raise OntologyGroundingError("Ontology grounding requires a PredefinedWorkcellSnapshot.")
    try:
        workcell.assert_unchanged()
    except (TypeError, ValueError) as exc:
        raise OntologyGroundingError(
            "Ontology grounding requires an immutable predefined Workcell."
        ) from exc
    if workcell.tbox_fingerprint != tbox.fingerprint:
        raise OntologyGroundingError(
            "Ontology grounding Workcell does not match the authoritative TBox."
        )


def _validate_candidate_and_review(
    candidate: OntologyGroundingCandidate,
    semantic_review: TargetFeatureSemanticReview,
    root: Path,
    abox: ABoxSnapshot,
) -> None:
    if (
        candidate.proposal_path
        != root / _PROPOSAL_ROOT / f"proposal_{candidate.proposal_number:04d}.json"
        or candidate.proposal_path.exists()
        or candidate.provisional_abox.interaction_root != root
        or candidate.provisional_abox.namespace != abox.namespace
        or candidate.provisional_abox.product_requirement != abox.product_requirement
        or candidate.provisional_abox.tbox_fingerprint != abox.tbox_fingerprint
        or semantic_review.proposal_number != candidate.proposal_number
        or semantic_review.target_feature_fingerprint
        != target_feature_fingerprint(candidate.proposal.target_feature)
        or semantic_review.evidence_refs != candidate.proposal.evidence_refs
    ):
        raise OntologyGroundingError(
            "Ontology grounding candidate or semantic review no longer matches."
        )
    expected_review_path = root / _REVIEW_ROOT / f"review_{semantic_review.review_number:04d}.json"
    if semantic_review.review_path != expected_review_path:
        raise OntologyGroundingError("Target-feature semantic review path is invalid.")
    record = _read_json_mapping(expected_review_path, "TargetFeatureSemanticReview")
    fingerprint = record.pop("fingerprint", None)
    if (
        fingerprint != semantic_review.fingerprint
        or _fingerprint(record) != semantic_review.fingerprint
        or record.get("schema_version") != 2
        or record.get("proposal_number") != candidate.proposal_number
        or record.get("target_feature_fingerprint") != semantic_review.target_feature_fingerprint
        or record.get("verdict") != semantic_review.verdict
        or record.get("gap") != semantic_review.gap
        or record.get("evidence_refs") != list(semantic_review.evidence_refs)
    ):
        raise OntologyGroundingError("Target-feature semantic review changed.")


def _validated_evidence_refs(
    value: object,
    field: str,
    authorized_evidence_refs: set[str],
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
        or not set(value).issubset(authorized_evidence_refs)
    ):
        raise OntologyGroundingError(
            f"{field} must contain unique authorized direct evidence refs."
        )
    return tuple(value)


def _translate_pa_references(
    value: object,
    translator: PAReferenceTranslator,
    *,
    parent_key: str | None = None,
) -> Any:
    """Translate only citation and typed-record fields across the PA boundary."""
    if isinstance(value, Mapping):
        translated: dict[str, object] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if key == "evidence_refs" and isinstance(item, list):
                refs: list[str] = []
                for ref in item:
                    if not isinstance(ref, str):
                        refs.append(ref)  # type: ignore[arg-type]
                        continue
                    try:
                        refs.append(translator(ref))
                    except (OSError, RuntimeError, TypeError, ValueError) as exc:
                        raise OntologyGroundingError(
                            "PA evidence reference is not authorized."
                        ) from exc
                translated[key] = refs
            elif key == "record_ref" and parent_key == "value_ref" and isinstance(item, str):
                try:
                    translated[key] = translator(item)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    raise OntologyGroundingError(
                        "PA typed-record reference is not authorized."
                    ) from exc
            else:
                translated[key] = _translate_pa_references(
                    item,
                    translator,
                    parent_key=key,
                )
        return translated
    if isinstance(value, list):
        return [_translate_pa_references(item, translator, parent_key=parent_key) for item in value]
    return _json_clone(value)


def _append_string_refs(target: list[str], value: object) -> None:
    if isinstance(value, list):
        target.extend(item for item in value if isinstance(item, str) and item)


def _decode_pointer_token(token: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(token):
        if token[index] != "~":
            result.append(token[index])
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            raise OntologyGroundingError("state value field_path has invalid escaping.")
        result.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(result)


def _is_empty_resolved_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (Mapping, list, tuple)):
        return not value
    return False


def _write_proposal_record(  # noqa: PLR0913
    path: Path,
    *,
    proposal_number: int,
    specification_iri: str,
    feature_iri: str,
    output: Mapping[str, object],
    compiled_delta: Mapping[str, object] | None,
    semantic_review: TargetFeatureSemanticReview | None,
    status: str,
    failure: str | None,
) -> None:
    root = path.parents[3]
    review_ref = None
    review_sha256 = None
    review_fingerprint = None
    if semantic_review is not None:
        review_ref = semantic_review.review_path.relative_to(root).as_posix()
        review_sha256 = _sha256_path(semantic_review.review_path)
        review_fingerprint = semantic_review.fingerprint
    record = {
        "schema_version": 8,
        "record_type": "OntologyGroundingProposal",
        "proposal_number": proposal_number,
        "initialized_specification_iri": specification_iri,
        "feature_iri": feature_iri,
        "output": _json_clone(output),
        "compiled_delta": (None if compiled_delta is None else _json_clone(compiled_delta)),
        "semantic_review_ref": review_ref,
        "semantic_review_sha256": review_sha256,
        "semantic_review_fingerprint": review_fingerprint,
        "status": status,
        "failure": failure,
    }
    _write_json_exclusive(path, record)


def _next_number(root: Path, prefix: str) -> int:
    return len(tuple(root.glob(f"{prefix}*.json"))) + 1


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise OntologyGroundingError(f"Cannot read pinned record: {path.name}.") from exc


def _json_clone(value: object) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise OntologyGroundingError(f"Record already exists: {path.name}.") from exc


def _read_json_mapping(path: Path, expected_type: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OntologyGroundingError(f"Cannot read {expected_type}: {path.name}.") from exc
    if not isinstance(value, dict) or value.get("record_type") != expected_type:
        raise OntologyGroundingError(f"{expected_type} record is invalid.")
    return value

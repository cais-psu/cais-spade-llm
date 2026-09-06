from __future__ import annotations

"""Validate one PA-authored target feature and compile its minimal ABox view."""

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
_PROPOSAL_KEYS = {"target_feature"}
_TARGET_FEATURE_BASE_KEYS = {"required_process", "current_state", "desired_state"}
_ASSEMBLY_TARGET_FEATURE_KEYS = _TARGET_FEATURE_BASE_KEYS | {"assembly_feature_association"}
_REQUIRED_PROCESS_KEYS = {"process_iri", "evidence_refs"}
_STATE_KEYS = {"statement", "state_values"}
_STATEMENT_KEYS = {"text", "evidence_refs"}
_STATE_VALUE_KEYS = {"name", "value_ref", "evidence_refs"}
_VALUE_REF_KEYS = {"record_ref", "field_path"}
_ASSEMBLY_ASSOCIATION_KEYS = {"assembly", "assembly_features", "evidence_refs"}
_STATE_ASSOCIATION_KEYS = _ASSEMBLY_ASSOCIATION_KEYS | {"state_names"}
_ASSEMBLY_KEYS = {"name", "evidence_refs"}
_ASSEMBLY_FEATURE_KEYS = {
    "name",
    "owner",
    "state_name",
    "state_value_name",
    "evidence_refs",
}
_ASSEMBLY_FEATURE_OWNER_KEYS = {"name", "type", "evidence_refs"}
_ASSEMBLY_FEATURE_OWNER_TYPES = ("Part", "Assembly")
_LOCATION_RECORD_TYPES = frozenset({"RGBDSegmentationRecord", "RobotFrameLocationRecord"})
_SEGMENTATION_CANDIDATE_PATH = re.compile(r"/cameras/[0-9]+/candidates/[0-9]+")

TypedRecordResolver = Callable[[str], Mapping[str, object]]
PAReferenceTranslator = Callable[[str], str]


class OntologyGroundingError(ValueError):
    """Raised when an untrusted target-feature proposal cannot be accepted."""

    def __init__(
        self,
        message: str,
        *,
        validation_code: str = "invalid_target_feature",
        unissued_reference: bool = False,
    ) -> None:
        """Distinguish a correctable citation handle from invalid source provenance."""
        super().__init__(message)
        self.validation_code = validation_code
        self.unissued_reference = unissued_reference


@dataclass(frozen=True)
class OntologyGroundingProposal:
    """Hold one validated PA-authored target feature and transient value projection."""

    target_feature: Mapping[str, object]
    feature_iri: str
    resolved_state_values: tuple[Mapping[str, object], ...]
    evidence_refs: tuple[str, ...]
    assembly_feature_association: Mapping[str, object] | list[Mapping[str, object]] | None = None
    assembly_state_value_refs: Mapping[str, Mapping[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class OntologyGroundingResult:
    """Return accepted assertions and the persisted target-feature proposal."""

    merge: MergeResult
    proposal_path: Path
    proposal: OntologyGroundingProposal


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
class OntologyGroundingAttempt:
    """Hold one PA target-feature candidate before strict validation."""

    proposal_path: Path
    proposal_number: int
    output: Mapping[str, object]


@dataclass(frozen=True)
class OntologyGroundingInterruption:
    """Return one direct ProductAgent requirement clarification."""

    kind: str
    message: str


async def propose_ontology_grounding(  # noqa: PLR0913
    product_agent: ProductAgentContextRuntime,
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    workcell: PredefinedWorkcellSnapshot,
    evidence_catalog: Sequence[Mapping[str, object]],
    tools: list[dict[str, Any]],
    tool_executor: Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]],
    max_tool_rounds: int,
    validation_feedback: Sequence[Mapping[str, object]] = (),
    previous_proposal: Mapping[str, object] | None = None,
    grounding_progress: Mapping[str, object] | None = None,
) -> OntologyGroundingAttempt | OntologyGroundingInterruption:
    """Let PA investigate and return one canonical candidate or clarification."""
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
        "validation_feedback": list(validation_feedback),
        "grounding_progress": dict(grounding_progress or {}),
        "previous_proposal": previous_proposal,
        "approved_evidence_catalog": [dict(item) for item in evidence_catalog],
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
            "state_value_cardinality": "zero_or_more",
            "value_names": "PA_authored_without_host_enum",
            "assembly_process": {
                "process_symbol": "assembly",
                "association_cardinality": "zero_or_more",
                "assembly_feature_cardinality": "exactly_two",
                "state_location_binding": "evidence_supported_or_both_null",
            },
        },
    }
    prompt = (
        "Investigate the exact product requirement using only the requirement and "
        "evidence returned by the controlled tools. Catalog metadata is discovery-only. "
        "Choose evidence and tools at your discretion; their presentation order has no "
        "priority. This investigation also supplies the neutral evidence pool for the "
        "later independent resource decision, so retrieve any approved "
        "context you judge necessary for grounding and resource assignment before returning. No source type or "
        "tool is mandatory. Never use hidden case knowledge, evaluator information, or an "
        "interpretation supplied by these instructions. Return exactly one result: one "
        "complete target_feature, one genuine requirement-meaning clarification_question, "
        "or unsupported_process when none of the authorized processes represents the "
        "requirement. Do not return a failure reason or validation decision.\n\n"
        "Author the target_feature directly from the evidence you selected. Use an empty "
        "or multi-valued state_values list when evidence supports zero or multiple values. "
        "Choose a process_iri only from authorized_processes and cite its direct evidence. "
        "Write evidence-supported current and desired product-state statements and cite "
        "their direct evidence. Unlisted relations are unknown rather than false. The "
        "current_state describes the evidenced state now and the desired_state describes "
        "the requested outcome. Add "
        "state_values only when accepted typed records returned by controlled tools "
        "support values belonging to those states. Current values locate the product now. "
        "Desired values may reference observed assembly destinations: explain their role "
        "in the desired statement and cite evidence connecting them to the required outcome. "
        "An observed destination is a reference for the goal, not a measured final product pose. "
        "Product identity alone does not support a desired location. Keep observations used "
        "only to identify the product in current_state or relationship evidence. Use the same "
        "exact observation in both states only when evidence supports its location role in "
        "the requested outcome, including a location that should remain unchanged. "
        "For each state value, author a "
        "unique semantic name within that state, exact "
        "record_ref, JSON Pointer field_path, and direct evidence_refs. The same "
        "record may supply multiple values through different paths. Do not invent a "
        "record, path, value, citation, provider, location, target pose, or "
        "host-defined value-name vocabulary. Do not infer a current/desired role "
        "from sensor names, camera identity, candidate order, or provider metadata; "
        "assign state values only from the approved evidence. When the selected process "
        "symbol is assembly, author a collection of zero or more assembly_feature_association relationships. "
        "Name its Assembly and exactly two mating AssemblyFeature endpoints, name each "
        "endpoint's owning Part or Assembly. Set association state_names to current_state, "
        "desired_state, or both according to when that relationship is supported. Endpoint "
        "bindings identify observations independently of the relationship's state membership. "
        "A desired relationship may use two endpoints observed in the current state. "
        "Support each exact candidate's role in this task, not just its part type. A small "
        "dimensional-error advantage cannot establish task role among similar instances. "
        "Choose an interchangeable instance only when requirement and evidence establish "
        "interchangeability; otherwise preserve ambiguity. "
        "Distinctive visual features, CAD measurements and source relationships may jointly "
        "support identity; measurement tools do not need to select a winner for you. "
        "Bind an endpoint only when its identity and coordinates are supported; otherwise "
        "set both state_name and state_value_name to null. A document figure can support "
        "an intended relationship but cannot establish current physical attachment. "
        "Preserve the source's qualifications without upgrading certainty. Each bound state value "
        "must cite an accepted coordinate-bearing observation or robot-frame location. "
        "For RGBDSegmentationRecord, bind the complete candidate path "
        "using its exact presented opaque field_path; for RobotFrameLocationRecord, bind "
        "/translated_location_m. "
        "You choose the endpoint meanings and exact evidence; the controller does not. "
        "Do not output ontology IRIs, RDF assertions, "
        "literal facts, context_summary, global evidence_refs, missing_information, "
        "primitive choices, resource choices, or execution details. The controller "
        "creates all interaction-local individuals and the exact ontology "
        "assertions after validation. Cite only requirement_0001 or evidence refs actually "
        "returned by a controlled tool or supplied as exact clarification evidence. "
        "Copy reference strings exactly; do not reconstruct them from similar handles.\n\n"
        "If validation_feedback is nonempty, use the remaining interaction budget to "
        "address its deterministic contract findings. The feedback identifies structural "
        "or evidence-reference problems and does not supply semantic answers or new evidence. "
        "grounding_progress reports cumulative evidence operations and proposal limits; "
        "remaining operations and proposals are their limits minus the used counts. "
        "A revised proposal does not reset either budget. The pinned observation is unchanged. "
        "Keep supported claims and bindings; deleting required bindings does not resolve "
        "missing goal evidence. For an unissued citation, select the exact reference from "
        "the supplied authorized references and its source content. Never guess a replacement "
        "from spelling similarity. User clarifications define the requested scope; a document's "
        "broader final assembly does not by itself add tasks to that scope.\n\n"
        f"Grounding input:\n{json.dumps(prompt_input, indent=2, ensure_ascii=False)}"
    )
    proposal_number = _next_number(root / _PROPOSAL_ROOT, "proposal_")
    proposal_path = root / _PROPOSAL_ROOT / f"proposal_{proposal_number:04d}.json"
    response_format = _proposal_response_format(workcell.processes)
    request_path = (
        root
        / _PROPOSAL_ROOT
        / f"request_{proposal_number:04d}_{len(tuple((root / _PROPOSAL_ROOT).glob('request_*.json'))) + 1:04d}.json"
    )
    _write_json_exclusive(
        request_path,
        {
            "record_type": "ProductAgentGroundingRequest",
            "prompt": prompt,
            "response_format": response_format,
            "tools": tools,
            "max_tool_rounds": max_tool_rounds,
            "grounding_progress": dict(grounding_progress or {}),
        },
    )
    transport_output = await product_agent.ask_llm_structured(
        prompt,
        response_format=response_format,
        tools=tools or None,
        tool_executor=tool_executor if tools else None,
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

    attempt_output = _json_clone(output)
    if set(attempt_output) != _PROPOSAL_KEYS or not isinstance(
        attempt_output.get("target_feature"), Mapping
    ):
        raise OntologyGroundingError("ProductAgent target-feature candidate is invalid.")
    return OntologyGroundingAttempt(
        proposal_path=proposal_path,
        proposal_number=proposal_number,
        output=attempt_output,
    )


def validate_ontology_grounding_attempt(
    attempt: OntologyGroundingAttempt,
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    workcell: PredefinedWorkcellSnapshot,
    authorized_evidence_refs: set[str],
    typed_record_resolver: TypedRecordResolver | None = None,
    pa_reference_resolver: PAReferenceTranslator | None = None,
    observation_reference_resolver: Callable[[object], Any] | None = None,
) -> OntologyGroundingCandidate:
    """Strictly validate one complete attempt without committing its ABox delta."""
    root = Path(interaction_root).resolve()
    _validate_authorities(root, tbox, abox, workcell)
    proposal_number = attempt.proposal_number
    proposal_path = root / _PROPOSAL_ROOT / f"proposal_{proposal_number:04d}.json"
    if attempt.proposal_path != proposal_path or proposal_path.exists():
        raise OntologyGroundingError("Ontology grounding attempt path is invalid.")
    canonical_output = _json_clone(attempt.output)
    try:
        if observation_reference_resolver is not None:
            try:
                canonical_output = observation_reference_resolver(canonical_output)
            except ValueError as exc:
                raise OntologyGroundingError(
                    "PA observation reference is invalid.",
                    validation_code="evidence_reference_invalid",
                ) from exc
        if pa_reference_resolver is not None:
            canonical_output = _translate_pa_references(
                canonical_output,
                pa_reference_resolver,
            )
        proposal = _validated_proposal(
            canonical_output,
            abox=abox,
            workcell=workcell,
            authorized_evidence_refs=authorized_evidence_refs,
            typed_record_resolver=typed_record_resolver,
        )
        delta = _compile_proposal_delta(
            proposal, namespace=abox.namespace, specification_iri=abox.specification_iri,
            ppr_namespace=tbox.ppr_namespace,
        )
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
            status="rejected",
            failure=failure,
        )
        validation_code = (
            exc.validation_code
            if isinstance(exc, OntologyGroundingError)
            else "invalid_target_feature"
        )
        raise OntologyGroundingError(
            f"OntologyGroundingProposal is invalid: {failure}",
            validation_code=validation_code,
            unissued_reference=isinstance(exc, OntologyGroundingError) and exc.unissued_reference,
        ) from exc

    return OntologyGroundingCandidate(
        provisional_abox=validated_delta.abox,
        proposal_path=proposal_path,
        proposal_number=proposal_number,
        output=_json_clone(canonical_output),
        compiled_delta=_json_clone(delta),
        proposal=proposal,
    )


async def propose_and_validate_ontology_grounding(
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
    typed_record_resolver: TypedRecordResolver | None = None,
    pa_reference_resolver: PAReferenceTranslator | None = None,
) -> OntologyGroundingCandidate | OntologyGroundingInterruption:
    """Acquire and strictly validate one complete target-feature candidate."""
    attempt = await propose_ontology_grounding(
        product_agent,
        interaction_root=interaction_root,
        tbox=tbox,
        abox=abox,
        workcell=workcell,
        evidence_catalog=evidence_catalog,
        tools=tools,
        tool_executor=tool_executor,
        max_tool_rounds=max_tool_rounds,
    )
    if isinstance(attempt, OntologyGroundingInterruption):
        return attempt
    return validate_ontology_grounding_attempt(
        attempt,
        interaction_root=interaction_root,
        tbox=tbox,
        abox=abox,
        workcell=workcell,
        authorized_evidence_refs=authorized_evidence_refs,
        typed_record_resolver=typed_record_resolver,
        pa_reference_resolver=pa_reference_resolver,
    )


def commit_ontology_grounding_candidate(
    candidate: OntologyGroundingCandidate,
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    workcell: PredefinedWorkcellSnapshot,
    authorized_evidence_refs: set[str],
    context_view_ref: str,
) -> OntologyGroundingResult:
    """Accept a PA proposal only while its original evidence snapshot remains intact."""
    from .grounding_contracts import (
        GroundingContractError,
        ProductContextView,
        build_grounding_evidence,
        build_product_context_view,
        validate_grounding_evidence,
    )

    root = Path(interaction_root).resolve()
    _validate_authorities(root, tbox, abox, workcell)
    _validate_candidate(candidate, root, abox)
    if not set(candidate.proposal.evidence_refs).issubset(authorized_evidence_refs):
        raise OntologyGroundingError("Ontology grounding evidence authority changed before commit.")
    try:
        evidence, caveats = build_grounding_evidence(
            root,
            context_view_ref=context_view_ref,
            target_feature=candidate.proposal.target_feature,
            proposal_number=candidate.proposal_number,
        )
        snapshot = ProductContextView.from_mapping(
            json.loads((root / context_view_ref).read_text())
        )
        current = build_product_context_view(
            root,
            abox,
            attempted_evidence=snapshot.attempted_evidence,
            assessed_at_ns=snapshot.assessed_at_ns,
        )
        if snapshot.to_record() != current.to_record():
            raise GroundingContractError(
                "Grounding context is not the current pre-commit snapshot."
            )
        candidate = replace(
            candidate, compiled_delta={**candidate.compiled_delta, "uncertainty": caveats}
        )
        projection = {
            "record_type": "OntologyGroundingProposal",
            "proposal_number": candidate.proposal_number,
            "initialized_specification_iri": abox.specification_iri,
            "feature_iri": candidate.proposal.feature_iri,
            "output": candidate.output,
            "compiled_delta": candidate.compiled_delta,
            "status": "accepted",
            "failure": None,
            "grounding_evidence": evidence,
        }
        validate_grounding_evidence(root, projection, require_committed=False)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise OntologyGroundingError(
            "Grounding source provenance is invalid.", validation_code="evidence_reference_invalid"
        ) from exc
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
        status="accepted",
        failure=None,
        grounding_evidence=evidence,
    )
    return OntologyGroundingResult(
        merge=merge, proposal_path=candidate.proposal_path, proposal=candidate.proposal
    )


def reject_ontology_grounding_candidate(
    candidate: OntologyGroundingCandidate,
    *,
    failure: str,
) -> None:
    """Retain a rejected attempt for diagnosis without merging any assertions."""
    _write_proposal_record(
        candidate.proposal_path,
        proposal_number=candidate.proposal_number,
        specification_iri=candidate.provisional_abox.specification_iri,
        feature_iri=candidate.proposal.feature_iri,
        output=candidate.output,
        compiled_delta=candidate.compiled_delta,
        status="rejected",
        failure=failure,
    )


def resolve_json_pointer(document: object, field_path: str) -> object:
    """Resolve one non-root RFC 6901 JSON Pointer against an exact record."""
    if not isinstance(field_path, str) or not field_path.startswith("/"):
        raise OntologyGroundingError(
            "state value field_path must be a non-root JSON Pointer.",
            validation_code="evidence_reference_invalid",
        )
    current = document
    for raw_token in field_path.split("/")[1:]:
        token = _decode_pointer_token(raw_token)
        if isinstance(current, Mapping):
            if token not in current:
                raise OntologyGroundingError(
                    f"state value field_path does not exist: {field_path}.",
                    validation_code="evidence_reference_invalid",
                )
            current = current[token]
            continue
        if isinstance(current, list):
            if token == "-" or not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise OntologyGroundingError(
                    f"state value field_path has an invalid array index: {field_path}.",
                    validation_code="evidence_reference_invalid",
                )
            index = int(token)
            if index >= len(current):
                raise OntologyGroundingError(
                    f"state value field_path does not exist: {field_path}.",
                    validation_code="evidence_reference_invalid",
                )
            current = current[index]
            continue
        raise OntologyGroundingError(
            f"state value field_path traverses a scalar: {field_path}.",
            validation_code="evidence_reference_invalid",
        )
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
    raise OntologyGroundingError(
        "Resolved state value is not JSON-compatible.",
        validation_code="evidence_reference_invalid",
    )


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
    associations = target_feature.get("assembly_feature_association", [])
    if isinstance(associations, Mapping):
        associations = [associations]
    for association in associations or []:
        _append_string_refs(refs, association.get("evidence_refs"))
        assembly = association.get("assembly")
        if isinstance(assembly, Mapping):
            _append_string_refs(refs, assembly.get("evidence_refs"))
        assembly_features = association.get("assembly_features")
        if isinstance(assembly_features, list):
            for assembly_feature in assembly_features:
                if not isinstance(assembly_feature, Mapping):
                    continue
                _append_string_refs(refs, assembly_feature.get("evidence_refs"))
                owner = assembly_feature.get("owner")
                if isinstance(owner, Mapping):
                    _append_string_refs(refs, owner.get("evidence_refs"))
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
    if not isinstance(target_feature, Mapping):
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
    process_symbol = workcell.process_symbol_for_iri(str(required_process["process_iri"]))
    association: Mapping[str, object] | list[Mapping[str, object]] | None = None
    assembly_state_value_refs: Mapping[str, Mapping[str, str]] = {}
    if process_symbol == "assembly":
        if set(target_feature) != _ASSEMBLY_TARGET_FEATURE_KEYS:
            raise OntologyGroundingError(
                "assembly target_feature must contain one assembly_feature_association."
            )
        raw_associations = target_feature["assembly_feature_association"]
        if not isinstance(raw_associations, list):
            raise OntologyGroundingError("assembly_feature_association must be a collection.")
        association = [
            _validated_assembly_feature_association(
                item,
                resolved_state_values=resolved_values,
                authorized_evidence_refs=authorized_evidence_refs,
            )[0]
            for item in raw_associations
        ]
    elif set(target_feature) != _TARGET_FEATURE_BASE_KEYS:
        raise OntologyGroundingError("target_feature fields are invalid.")
    cloned_target = _json_clone(target_feature)
    evidence_refs = tuple(dict.fromkeys(target_feature_evidence_refs(cloned_target)))
    return OntologyGroundingProposal(
        target_feature=cloned_target,
        feature_iri=f"{abox.namespace}feature_0001",
        resolved_state_values=tuple(resolved_values),
        assembly_feature_association=(None if association is None else _json_clone(association)),
        assembly_state_value_refs=_json_clone(assembly_state_value_refs),
        evidence_refs=evidence_refs,
    )


def _validated_assembly_feature_association(  # noqa: C901
    value: object,
    *,
    resolved_state_values: Sequence[Mapping[str, object]],
    authorized_evidence_refs: set[str],
) -> tuple[Mapping[str, object], Mapping[str, Mapping[str, str]]]:
    """Validate one PA-authored neutral assembly-feature association."""
    expected_keys = _STATE_ASSOCIATION_KEYS
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise OntologyGroundingError("assembly_feature_association fields are invalid.")
    membership = value["state_names"]
    if (
        not isinstance(membership, list)
        or not membership
        or any(item not in ("current_state", "desired_state") for item in membership)
        or len(membership) != len(set(membership))
    ):
        raise OntologyGroundingError("Association state_names must identify distinct valid states.")
    _validated_evidence_refs(
        value.get("evidence_refs"),
        "assembly_feature_association.evidence_refs",
        authorized_evidence_refs,
    )
    assembly = value.get("assembly")
    if not isinstance(assembly, Mapping) or set(assembly) != _ASSEMBLY_KEYS:
        raise OntologyGroundingError("assembly_feature_association assembly is invalid.")
    _validated_name(assembly.get("name"), "assembly_feature_association.assembly.name")
    _validated_evidence_refs(
        assembly.get("evidence_refs"),
        "assembly_feature_association.assembly.evidence_refs",
        authorized_evidence_refs,
    )
    features = value.get("assembly_features")
    if (
        not isinstance(features, list)
        or len(features) != 2
        or not all(isinstance(item, Mapping) for item in features)
    ):
        raise OntologyGroundingError(
            "assembly_feature_association must contain exactly two assembly_features."
        )
    resolved_by_state_and_name = {
        (item.get("state"), item.get("name")): item for item in resolved_state_values
    }
    feature_names: set[str] = set()
    feature_identities: set[tuple[str, str, str]] = set()
    owner_names: set[str] = set()
    state_names: set[str] = set()
    state_refs: dict[str, Mapping[str, str]] = {}
    for index, feature_value in enumerate(features):
        assert isinstance(feature_value, Mapping)
        label = f"assembly_feature_association.assembly_features[{index}]"
        if set(feature_value) != _ASSEMBLY_FEATURE_KEYS:
            raise OntologyGroundingError(f"{label} fields are invalid.")
        feature_name = _validated_name(feature_value.get("name"), f"{label}.name")
        feature_names.add(feature_name)
        _validated_evidence_refs(
            feature_value.get("evidence_refs"),
            f"{label}.evidence_refs",
            authorized_evidence_refs,
        )
        owner = feature_value.get("owner")
        if not isinstance(owner, Mapping) or set(owner) != _ASSEMBLY_FEATURE_OWNER_KEYS:
            raise OntologyGroundingError(f"{label}.owner fields are invalid.")
        owner_name = _validated_name(owner.get("name"), f"{label}.owner.name")
        owner_names.add(owner_name)
        if owner.get("type") not in _ASSEMBLY_FEATURE_OWNER_TYPES:
            raise OntologyGroundingError(f"{label}.owner.type is invalid.")
        identity = (str(owner["type"]), owner_name, feature_name)
        if identity in feature_identities:
            raise OntologyGroundingError(
                "AssemblyFeature endpoints must be distinct exact identities."
            )
        feature_identities.add(identity)
        _validated_evidence_refs(
            owner.get("evidence_refs"),
            f"{label}.owner.evidence_refs",
            authorized_evidence_refs,
        )
        state_name = feature_value.get("state_name")
        state_value_name = feature_value.get("state_value_name")
        if (state_name is None) and (state_value_name is None):
            continue
        if state_name not in {"current_state", "desired_state"}:
            raise OntologyGroundingError(f"{label}.state_name is invalid.")
        state_names.add(str(state_name))
        state_value_name = _validated_name(
            state_value_name,
            f"{label}.state_value_name",
        )
        resolved = resolved_by_state_and_name.get((state_name, state_value_name))
        if resolved is None:
            raise OntologyGroundingError(f"{label} does not reference an accepted state value.")
        if resolved.get("record_type") not in _LOCATION_RECORD_TYPES:
            raise OntologyGroundingError(
                f"{label} state value is not coordinate-bearing location evidence."
            )
        value_ref = resolved.get("value_ref")
        if not isinstance(value_ref, Mapping):
            raise OntologyGroundingError(f"{label} state value reference is invalid.")
        field_path = value_ref.get("field_path")
        if (
            resolved.get("record_type") == "RGBDSegmentationRecord"
            and (
                not isinstance(field_path, str)
                or _SEGMENTATION_CANDIDATE_PATH.fullmatch(field_path) is None
            )
        ) or (
            resolved.get("record_type") == "RobotFrameLocationRecord"
            and field_path != "/translated_location_m"
        ):
            raise OntologyGroundingError(
                f"{label} state value does not identify one allocatable location."
            )
        state_refs[str(state_name)] = {
            "record_ref": str(value_ref["record_ref"]),
            "field_path": str(field_path),
            "record_sha256": str(resolved["record_sha256"]),
        }
    return _json_clone(value), state_refs


def eligible_allocation_pairs(
    associations: Sequence[Mapping[str, object]],
    resolved_state_values: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, Mapping[str, str]], ...]:
    """Return distinct exact coordinate pairs without selecting by presentation order."""
    resolved = {(item["state"], item["name"]): item for item in resolved_state_values}
    pairs: dict[str, Mapping[str, Mapping[str, str]]] = {}
    for association in associations:
        endpoints = association["assembly_features"]
        if {item["state_name"] for item in endpoints} != {"current_state", "desired_state"}:
            continue
        pair = {}
        for endpoint in endpoints:
            value = resolved[(endpoint["state_name"], endpoint["state_value_name"])]
            pair[endpoint["state_name"]] = {
                **value["value_ref"],
                "record_sha256": value["record_sha256"],
            }
        pairs.setdefault(_fingerprint(pair), pair)
    return tuple(pairs.values())


def _compile_associations(
    associations: Sequence[Mapping[str, object]],
    *,
    namespace: str,
    ppr_namespace: str,
) -> list[dict[str, object]]:
    assertions: list[dict[str, object]] = []
    owners: dict[tuple[str, str], str] = {}
    features: dict[tuple[str, str, str], str] = {}
    counts = {"Part": 0, "Assembly": 0}

    def triple(subject: str, predicate: str, value: str, refs: object) -> None:
        entry = {
            "subject": subject,
            "predicate": predicate,
            "object": {"kind": "iri", "value": value},
            "evidence_refs": list(refs),
        }
        if entry not in assertions:
            assertions.append(entry)

    def owner_iri(owner: Mapping[str, object]) -> str:
        key = (str(owner["type"]), str(owner["name"]))
        if key not in owners:
            counts[key[0]] += 1
            prefix = {"Part": "part", "Assembly": "assembly"}[key[0]]
            owners[key] = f"{namespace}{prefix}_{counts[key[0]]:04d}"
        iri = owners[key]
        triple(iri, str(RDF.type), f"{ppr_namespace}{key[0]}", owner["evidence_refs"])
        return iri

    for index, association in enumerate(associations, 1):
        assembly = association["assembly"]
        assembly_iri = owner_iri({**assembly, "type": "Assembly"})
        iri = f"{namespace}assemblyfeatureassociation_{index:04d}"
        refs = association["evidence_refs"]
        triple(iri, str(RDF.type), f"{ppr_namespace}AssemblyFeatureAssociation", refs)
        triple(assembly_iri, f"{ppr_namespace}hasAssemblyFeatureAssociation", iri, refs)
        for state_name in association.get("state_names", []):
            predicate, state = {
                "current_state": ("hascurrentstate", "currentstate_0001"),
                "desired_state": ("hasdesiredstate", "desiredstate_0001"),
            }[state_name]
            triple(iri, f"{ppr_namespace}{predicate}", f"{namespace}{state}", refs)
        for endpoint in association["assembly_features"]:
            owner = endpoint["owner"]
            owning_iri = owner_iri(owner)
            key = (str(owner["type"]), str(owner["name"]), str(endpoint["name"]))
            if key not in features:
                features[key] = f"{namespace}assemblyfeature_{len(features) + 1:04d}"
            feature_iri = features[key]
            if owning_iri != assembly_iri:
                triple(
                    assembly_iri, f"{ppr_namespace}hasPart", owning_iri, owner["evidence_refs"]
                )
            triple(
                feature_iri,
                str(RDF.type),
                f"{ppr_namespace}AssemblyFeature",
                endpoint["evidence_refs"],
            )
            triple(
                owning_iri,
                f"{ppr_namespace}hasAssemblyFeature",
                feature_iri,
                endpoint["evidence_refs"],
            )
            triple(iri, f"{ppr_namespace}relatesAssemblyFeature", feature_iri, refs)
    return assertions


def _validated_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OntologyGroundingError(f"{label} is empty.")
    return value


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
                f"{state_name}.state_values[{index}].value_ref.record_ref is invalid.",
                validation_code="evidence_reference_invalid",
            )
        if typed_record_resolver is None:
            raise OntologyGroundingError(
                "state values require an accepted typed-record resolver.",
                validation_code="evidence_reference_invalid",
            )
        try:
            resolved_record = typed_record_resolver(record_ref)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise OntologyGroundingError(
                f"{state_name}.state_values[{index}] record_ref is not an accepted typed binding.",
                validation_code="evidence_reference_invalid",
            ) from exc
        if not isinstance(resolved_record, Mapping) or set(resolved_record) != {
            "record_type",
            "record_sha256",
            "record",
        }:
            raise OntologyGroundingError(
                "Typed-record resolver result is invalid.",
                validation_code="evidence_reference_invalid",
            )
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
            raise OntologyGroundingError(
                "Typed-record resolver result is invalid.",
                validation_code="evidence_reference_invalid",
            )
        resolved_value = resolve_json_pointer(record, field_path)  # type: ignore[arg-type]
        if _is_empty_resolved_value(resolved_value):
            raise OntologyGroundingError(
                f"{state_name}.state_values[{index}] resolves to an empty value.",
                validation_code="evidence_reference_invalid",
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
    namespace: str,
    specification_iri: str,
    ppr_namespace: str,
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
    current_state_iri = f"{namespace}currentstate_0001"
    desired_state_iri = f"{namespace}desiredstate_0001"
    target_class = "feature"
    assertions: list[dict[str, object]] = [
        {
            "subject": proposal.feature_iri,
            "predicate": str(RDF.type),
            "object": {
                "kind": "iri",
                "value": f"{ppr_namespace}{target_class}",
            },
            "evidence_refs": desired_refs,
        },
        {
            "subject": specification_iri,
            "predicate": f"{ppr_namespace}defines",
            "object": {"kind": "iri", "value": proposal.feature_iri},
            "evidence_refs": desired_refs,
        },
        {
            "subject": str(process["process_iri"]),
            "predicate": f"{ppr_namespace}realizes",
            "object": {"kind": "iri", "value": proposal.feature_iri},
            "evidence_refs": process_refs,
        },
        {
            "subject": current_state_iri,
            "predicate": str(RDF.type),
            "object": {
                "kind": "iri",
                "value": f"{ppr_namespace}state",
            },
            "evidence_refs": current_refs,
        },
        {
            "subject": desired_state_iri,
            "predicate": str(RDF.type),
            "object": {
                "kind": "iri",
                "value": f"{ppr_namespace}state",
            },
            "evidence_refs": desired_refs,
        },
        {
            "subject": proposal.feature_iri,
            "predicate": f"{ppr_namespace}hascurrentstate",
            "object": {"kind": "iri", "value": current_state_iri},
            "evidence_refs": current_refs,
        },
        {
            "subject": proposal.feature_iri,
            "predicate": f"{ppr_namespace}hasdesiredstate",
            "object": {"kind": "iri", "value": desired_state_iri},
            "evidence_refs": desired_refs,
        },
    ]
    association = proposal.assembly_feature_association
    if isinstance(association, list):
        assertions.extend(_compile_associations(association, namespace=namespace, ppr_namespace=ppr_namespace))
    elif association is not None:
        association_refs = list(association["evidence_refs"])
        assembly = association["assembly"]
        assembly_features = association["assembly_features"]
        assert isinstance(assembly, Mapping)
        assert isinstance(assembly_features, list)
        assembly_iri = f"{namespace}assembly_0001"
        assertions.append(
            {
                "subject": assembly_iri,
                "predicate": str(RDF.type),
                "object": {"kind": "iri", "value": f"{ppr_namespace}Assembly"},
                "evidence_refs": list(assembly["evidence_refs"]),
            }
        )
        owner_type_counts = {"Part": 0, "Assembly": 1}
        ordered_features = sorted(
            assembly_features,
            key=lambda item: 0 if item["state_name"] == "current_state" else 1,
        )
        for feature_index, feature_value in enumerate(ordered_features, start=1):
            owner = feature_value["owner"]
            assert isinstance(owner, Mapping)
            owner_type = str(owner["type"])
            owner_type_counts[owner_type] += 1
            owner_iri = (
                f"{namespace}{owner_type.casefold()}_{owner_type_counts[owner_type]:04d}"
            )
            assembly_feature_iri = f"{namespace}assemblyfeature_{feature_index:04d}"
            owner_refs = list(owner["evidence_refs"])
            feature_refs = list(feature_value["evidence_refs"])
            assertions.extend(
                [
                    {
                        "subject": owner_iri,
                        "predicate": str(RDF.type),
                        "object": {
                            "kind": "iri",
                            "value": f"{ppr_namespace}{owner_type}",
                        },
                        "evidence_refs": owner_refs,
                    },
                    {
                        "subject": assembly_iri,
                        "predicate": f"{ppr_namespace}hasPart",
                        "object": {"kind": "iri", "value": owner_iri},
                        "evidence_refs": owner_refs,
                    },
                    {
                        "subject": assembly_feature_iri,
                        "predicate": str(RDF.type),
                        "object": {
                            "kind": "iri",
                            "value": f"{ppr_namespace}AssemblyFeature",
                        },
                        "evidence_refs": feature_refs,
                    },
                    {
                        "subject": owner_iri,
                        "predicate": f"{ppr_namespace}hasAssemblyFeature",
                        "object": {"kind": "iri", "value": assembly_feature_iri},
                        "evidence_refs": feature_refs,
                    },
                    {
                        "subject": proposal.feature_iri,
                        "predicate": f"{ppr_namespace}relatesAssemblyFeature",
                        "object": {"kind": "iri", "value": assembly_feature_iri},
                        "evidence_refs": list(dict.fromkeys((*association_refs, *feature_refs))),
                    },
                ]
            )
        assertions.append(
            {
                "subject": assembly_iri,
                "predicate": f"{ppr_namespace}hasAssemblyFeatureAssociation",
                "object": {"kind": "iri", "value": proposal.feature_iri},
                "evidence_refs": association_refs,
            }
        )
    return {
        "assertions": assertions,
        "uncertainty": [],
        "unresolved_evidence_needs": [],
        "typed_context_refs": [],
    }


def _proposal_response_format(
    processes: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    assembly_process_iris = tuple(
        process_iri for process_symbol, process_iri in processes if process_symbol == "assembly"
    )
    other_process_iris = tuple(
        process_iri for process_symbol, process_iri in processes if process_symbol != "assembly"
    )
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

    def target_feature_schema(
        allowed_process_iris: tuple[str, ...],
        *,
        require_assembly_association: bool,
    ) -> dict[str, Any]:
        required_keys = set(_TARGET_FEATURE_BASE_KEYS)
        properties: dict[str, Any] = {
            "required_process": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_REQUIRED_PROCESS_KEYS),
                "properties": {
                    "process_iri": {
                        "type": "string",
                        "enum": list(allowed_process_iris),
                    },
                    "evidence_refs": evidence_refs,
                },
            },
            "current_state": state,
            "desired_state": state,
        }
        if require_assembly_association:
            required_keys.add("assembly_feature_association")
            owner = {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_ASSEMBLY_FEATURE_OWNER_KEYS),
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "type": {
                        "type": "string",
                        "enum": list(_ASSEMBLY_FEATURE_OWNER_TYPES),
                    },
                    "evidence_refs": evidence_refs,
                },
            }
            assembly_feature = {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_ASSEMBLY_FEATURE_KEYS),
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "owner": owner,
                    "state_name": {
                        "type": ["string", "null"],
                        "enum": ["current_state", "desired_state", None],
                    },
                    "state_value_name": {"type": ["string", "null"]},
                    "evidence_refs": evidence_refs,
                },
            }
            properties["assembly_feature_association"] = {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_STATE_ASSOCIATION_KEYS),
                "properties": {
                    "state_names": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["current_state", "desired_state"]},
                        "minItems": 1,
                        "maxItems": 2,
                    },
                    "assembly": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": sorted(_ASSEMBLY_KEYS),
                        "properties": {
                            "name": {"type": "string", "minLength": 1},
                            "evidence_refs": evidence_refs,
                        },
                    },
                    "assembly_features": {
                        "type": "array",
                        "items": assembly_feature,
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "evidence_refs": evidence_refs,
                },
            }
            properties["assembly_feature_association"] = {
                "type": "array",
                "items": properties["assembly_feature_association"],
            }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(required_keys),
            "properties": properties,
        }

    target_feature_variants = []
    if assembly_process_iris:
        target_feature_variants.append(
            target_feature_schema(
                assembly_process_iris,
                require_assembly_association=True,
            )
        )
    if other_process_iris:
        target_feature_variants.append(
            target_feature_schema(
                other_process_iris,
                require_assembly_association=False,
            )
        )
    target_feature = (
        target_feature_variants[0]
        if len(target_feature_variants) == 1
        else {"anyOf": target_feature_variants}
    )
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
                            "required": ["unsupported_process"],
                            "properties": {"unsupported_process": {"type": "boolean"}},
                        },
                    ]
                }
            },
        },
    }


def _validated_interruption(
    output: Mapping[str, object],
) -> OntologyGroundingInterruption | None:
    if set(output) == {"clarification_question"}:
        message = output["clarification_question"]
        if not isinstance(message, str) or not message.strip():
            raise OntologyGroundingError("clarification_question is invalid.")
        return OntologyGroundingInterruption("clarification_question", message)
    if set(output) == {"unsupported_process"} and output["unsupported_process"] is True:
        return OntologyGroundingInterruption(
            "unsupported_process",
            "No authorized process represents the submitted requirement.",
        )
    return None


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


def _validate_candidate(
    candidate: OntologyGroundingCandidate,
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
    ):
        raise OntologyGroundingError("Ontology grounding candidate no longer matches.")


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
            f"{field} must contain unique authorized direct evidence refs.",
            validation_code="evidence_reference_invalid",
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
                            "PA evidence reference is not authorized.",
                            validation_code="evidence_reference_invalid",
                            unissued_reference=True,
                        ) from exc
                translated[key] = refs
            elif key == "record_ref" and parent_key == "value_ref" and isinstance(item, str):
                try:
                    translated[key] = translator(item)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    raise OntologyGroundingError(
                        "PA typed-record reference is not authorized.",
                        validation_code="evidence_reference_invalid",
                        unissued_reference=True,
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
            raise OntologyGroundingError(
                "state value field_path has invalid escaping.",
                validation_code="evidence_reference_invalid",
            )
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


def _write_proposal_record(
    path: Path,
    *,
    proposal_number: int,
    specification_iri: str,
    feature_iri: str,
    output: Mapping[str, object],
    compiled_delta: Mapping[str, object] | None,
    status: str,
    failure: str | None,
    grounding_evidence: Mapping[str, object] | None = None,
) -> None:
    record = {
        "record_type": "OntologyGroundingProposal",
        "proposal_number": proposal_number,
        "initialized_specification_iri": specification_iri,
        "feature_iri": feature_iri,
        "output": _json_clone(output),
        "compiled_delta": (None if compiled_delta is None else _json_clone(compiled_delta)),
        "status": status,
        "failure": failure,
    }
    if grounding_evidence is not None:
        record["grounding_evidence"] = _json_clone(grounding_evidence)
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

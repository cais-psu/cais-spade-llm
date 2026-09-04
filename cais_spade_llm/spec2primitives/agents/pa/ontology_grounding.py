"""Validate one PA-authored target feature and compile its minimal ABox view."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
_ASSEMBLY_TARGET_FEATURE_KEYS = _TARGET_FEATURE_BASE_KEYS | {
    "assembly_feature_association"
}
_REQUIRED_PROCESS_KEYS = {"process_iri", "evidence_refs"}
_STATE_KEYS = {"statement", "state_values"}
_STATEMENT_KEYS = {"text", "evidence_refs"}
_STATE_VALUE_KEYS = {"name", "value_ref", "evidence_refs"}
_VALUE_REF_KEYS = {"record_ref", "field_path"}
_ASSEMBLY_ASSOCIATION_KEYS = {"assembly", "assembly_features", "evidence_refs"}
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
_LOCATION_RECORD_TYPES = frozenset(
    {"RGBDSegmentationRecord", "RobotFrameLocationRecord"}
)
_SEGMENTATION_CANDIDATE_PATH = re.compile(
    r"/cameras/[0-9]+/candidates/[0-9]+"
)

TypedRecordResolver = Callable[[str], Mapping[str, object]]
PAReferenceTranslator = Callable[[str], str]


class OntologyGroundingError(ValueError):
    """Raised when an untrusted target-feature proposal cannot be accepted."""

    def __init__(
        self,
        message: str,
        *,
        validation_code: str = "invalid_target_feature",
    ) -> None:
        super().__init__(message)
        self.validation_code = validation_code


@dataclass(frozen=True)
class OntologyGroundingProposal:
    """Hold one validated PA-authored target feature and transient value projection."""

    target_feature: Mapping[str, object]
    feature_iri: str
    resolved_state_values: tuple[Mapping[str, object], ...]
    evidence_refs: tuple[str, ...]
    assembly_feature_association: Mapping[str, object] | None = None
    assembly_state_value_refs: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )


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
                "association_cardinality": "exactly_one",
                "assembly_feature_cardinality": "exactly_two",
                "state_location_binding": "one_current_and_one_desired",
            },
        },
    }
    prompt = (
        "Investigate the exact product requirement using only the requirement and "
        "evidence returned by the controlled tools. Catalog metadata is discovery-only. "
        "Choose evidence and tools at your discretion; their presentation order has no "
        "priority. This investigation also supplies the neutral evidence pool for the "
        "later independent resource decision, so retrieve any approved "
        "context you judge necessary for Phase 4 before returning. No source type or "
        "tool is mandatory. Never use hidden case knowledge, evaluator information, or an "
        "interpretation supplied by these instructions. Return exactly one result: one "
        "complete target_feature, one genuine requirement-meaning clarification_question, "
        "or unsupported_process when none of the authorized processes represents the "
        "requirement. Do not return a failure reason or validation decision.\n\n"
        "Author the target_feature directly from the evidence you selected. Use an empty "
        "or multi-valued state_values list when evidence supports zero or multiple values. "
        "Choose a process_iri only from authorized_processes and cite its direct evidence. "
        "Write positive, open-world current and desired product-state statements and cite "
        "their direct evidence. Unlisted relations are unknown rather than false. The "
        "current_state describes the evidenced state now and the desired_state describes "
        "the requested outcome. Add "
        "state_values only when accepted typed records returned by controlled tools "
        "support values belonging to those states. For each state value, author a "
        "unique semantic name within that state, exact "
        "record_ref, JSON Pointer field_path, and direct evidence_refs. The same "
        "record may supply multiple values through different paths. Do not invent a "
        "record, path, value, citation, provider, location, target pose, or "
        "host-defined value-name vocabulary. Do not infer a current/desired role "
        "from sensor names, camera identity, candidate order, or provider metadata; "
        "assign state values only from the approved evidence. When the selected process "
        "symbol is assembly, author exactly one neutral assembly_feature_association. "
        "Name its Assembly and exactly two mating AssemblyFeature endpoints, name each "
        "endpoint's owning Part or Assembly, and bind one endpoint to a current_state "
        "state value and the other to a desired_state state value. Each bound state value "
        "must cite an accepted coordinate-bearing observation or robot-frame location. "
        "For RGBDSegmentationRecord, bind the complete candidate path "
        "/cameras/<index>/candidates/<index>; for RobotFrameLocationRecord, bind "
        "/translated_location_m. "
        "You choose the endpoint meanings and exact evidence; the controller does not. "
        "Do not output ontology IRIs, RDF assertions, "
        "literal facts, context_summary, global evidence_refs, missing_information, "
        "primitive choices, resource choices, or execution details. The controller "
        "creates all interaction-local individuals and the exact ontology "
        "assertions after validation. Cite only requirement_0001 or evidence refs actually "
        "returned by a controlled tool.\n\n"
        f"Grounding input:\n{json.dumps(prompt_input, indent=2, ensure_ascii=False)}"
    )
    proposal_number = _next_number(root / _PROPOSAL_ROOT, "proposal_")
    proposal_path = root / _PROPOSAL_ROOT / f"proposal_{proposal_number:04d}.json"
    transport_output = await product_agent.ask_llm_structured(
        prompt,
        response_format=_proposal_response_format(workcell.processes),
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


def validate_ontology_grounding_attempt(  # noqa: PLR0913
    attempt: OntologyGroundingAttempt,
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    workcell: PredefinedWorkcellSnapshot,
    authorized_evidence_refs: set[str],
    typed_record_resolver: TypedRecordResolver | None = None,
    pa_reference_resolver: PAReferenceTranslator | None = None,
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
        if pa_reference_resolver is not None:
            canonical_output = _translate_pa_references(
                attempt.output,
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
        ) from exc

    return OntologyGroundingCandidate(
        provisional_abox=validated_delta.abox,
        proposal_path=proposal_path,
        proposal_number=proposal_number,
        output=_json_clone(canonical_output),
        compiled_delta=_json_clone(delta),
        proposal=proposal,
    )


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
) -> OntologyGroundingResult:
    """Revalidate and persist one PA-authored target feature without repair."""
    root = Path(interaction_root).resolve()
    _validate_authorities(root, tbox, abox, workcell)
    _validate_candidate(candidate, root, abox)
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
        status="accepted",
        failure=None,
    )
    return OntologyGroundingResult(
        merge=merge,
        proposal_path=candidate.proposal_path,
        proposal=candidate.proposal,
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
    association = target_feature.get("assembly_feature_association")
    if isinstance(association, Mapping):
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
    association: Mapping[str, object] | None = None
    assembly_state_value_refs: Mapping[str, Mapping[str, str]] = {}
    if process_symbol == "assembly":
        if set(target_feature) != _ASSEMBLY_TARGET_FEATURE_KEYS:
            raise OntologyGroundingError(
                "assembly target_feature must contain one assembly_feature_association."
            )
        association, assembly_state_value_refs = _validated_assembly_feature_association(
            target_feature["assembly_feature_association"],
            resolved_state_values=resolved_values,
            authorized_evidence_refs=authorized_evidence_refs,
        )
    elif set(target_feature) != _TARGET_FEATURE_BASE_KEYS:
        raise OntologyGroundingError("target_feature fields are invalid.")
    cloned_target = _json_clone(target_feature)
    evidence_refs = tuple(dict.fromkeys(target_feature_evidence_refs(cloned_target)))
    return OntologyGroundingProposal(
        target_feature=cloned_target,
        feature_iri=f"{abox.namespace}feature_0001",
        resolved_state_values=tuple(resolved_values),
        assembly_feature_association=(
            None if association is None else _json_clone(association)
        ),
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
    if not isinstance(value, Mapping) or set(value) != _ASSEMBLY_ASSOCIATION_KEYS:
        raise OntologyGroundingError("assembly_feature_association fields are invalid.")
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
        (item.get("state"), item.get("name")): item
        for item in resolved_state_values
    }
    feature_names: set[str] = set()
    owner_names: set[str] = set()
    state_names: set[str] = set()
    state_refs: dict[str, Mapping[str, str]] = {}
    for index, feature_value in enumerate(features):
        assert isinstance(feature_value, Mapping)
        label = f"assembly_feature_association.assembly_features[{index}]"
        if set(feature_value) != _ASSEMBLY_FEATURE_KEYS:
            raise OntologyGroundingError(f"{label} fields are invalid.")
        feature_name = _validated_name(feature_value.get("name"), f"{label}.name")
        if feature_name in feature_names:
            raise OntologyGroundingError("AssemblyFeature names must be distinct.")
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
        if owner_name in owner_names:
            raise OntologyGroundingError("AssemblyFeature owners must be distinct.")
        owner_names.add(owner_name)
        if owner.get("type") not in _ASSEMBLY_FEATURE_OWNER_TYPES:
            raise OntologyGroundingError(f"{label}.owner.type is invalid.")
        _validated_evidence_refs(
            owner.get("evidence_refs"),
            f"{label}.owner.evidence_refs",
            authorized_evidence_refs,
        )
        state_name = feature_value.get("state_name")
        state_value_name = feature_value.get("state_value_name")
        if state_name not in {"current_state", "desired_state"}:
            raise OntologyGroundingError(f"{label}.state_name is invalid.")
        if state_name in state_names:
            raise OntologyGroundingError(
                "AssemblyFeature endpoints must bind one current_state and one desired_state."
            )
        state_names.add(str(state_name))
        state_value_name = _validated_name(
            state_value_name,
            f"{label}.state_value_name",
        )
        resolved = resolved_by_state_and_name.get((state_name, state_value_name))
        if resolved is None:
            raise OntologyGroundingError(
                f"{label} does not reference an accepted state value."
            )
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
    if state_names != {"current_state", "desired_state"}:
        raise OntologyGroundingError(
            "AssemblyFeature endpoints must bind one current_state and one desired_state."
        )
    return _json_clone(value), state_refs


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
    target_class = (
        "AssemblyFeatureAssociation"
        if proposal.assembly_feature_association is not None
        else "feature"
    )
    assertions: list[dict[str, object]] = [
        {
            "subject": proposal.feature_iri,
            "predicate": str(RDF.type),
            "object": {
                "kind": "iri",
                "value": f"{tbox.ppr_namespace}{target_class}",
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
    ]
    association = proposal.assembly_feature_association
    if association is not None:
        association_refs = list(association["evidence_refs"])
        assembly = association["assembly"]
        assembly_features = association["assembly_features"]
        assert isinstance(assembly, Mapping)
        assert isinstance(assembly_features, list)
        assembly_iri = f"{abox.namespace}assembly_0001"
        assertions.append(
            {
                "subject": assembly_iri,
                "predicate": str(RDF.type),
                "object": {"kind": "iri", "value": f"{tbox.ppr_namespace}Assembly"},
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
                f"{abox.namespace}{owner_type.casefold()}_"
                f"{owner_type_counts[owner_type]:04d}"
            )
            assembly_feature_iri = f"{abox.namespace}assemblyfeature_{feature_index:04d}"
            owner_refs = list(owner["evidence_refs"])
            feature_refs = list(feature_value["evidence_refs"])
            assertions.extend(
                [
                    {
                        "subject": owner_iri,
                        "predicate": str(RDF.type),
                        "object": {
                            "kind": "iri",
                            "value": f"{tbox.ppr_namespace}{owner_type}",
                        },
                        "evidence_refs": owner_refs,
                    },
                    {
                        "subject": assembly_iri,
                        "predicate": f"{tbox.ppr_namespace}hasPart",
                        "object": {"kind": "iri", "value": owner_iri},
                        "evidence_refs": owner_refs,
                    },
                    {
                        "subject": assembly_feature_iri,
                        "predicate": str(RDF.type),
                        "object": {
                            "kind": "iri",
                            "value": f"{tbox.ppr_namespace}AssemblyFeature",
                        },
                        "evidence_refs": feature_refs,
                    },
                    {
                        "subject": owner_iri,
                        "predicate": f"{tbox.ppr_namespace}hasAssemblyFeature",
                        "object": {"kind": "iri", "value": assembly_feature_iri},
                        "evidence_refs": feature_refs,
                    },
                    {
                        "subject": proposal.feature_iri,
                        "predicate": f"{tbox.ppr_namespace}relatesAssemblyFeature",
                        "object": {"kind": "iri", "value": assembly_feature_iri},
                        "evidence_refs": list(
                            dict.fromkeys((*association_refs, *feature_refs))
                        ),
                    },
                ]
            )
        assertions.append(
            {
                "subject": assembly_iri,
                "predicate": f"{tbox.ppr_namespace}hasAssemblyFeatureAssociation",
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
                        "type": "string",
                        "enum": ["current_state", "desired_state"],
                    },
                    "state_value_name": {"type": "string", "minLength": 1},
                    "evidence_refs": evidence_refs,
                },
            }
            properties["assembly_feature_association"] = {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_ASSEMBLY_ASSOCIATION_KEYS),
                "properties": {
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
                            "properties": {
                                "unsupported_process": {"type": "boolean"}
                            },
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
                        ) from exc
                translated[key] = refs
            elif key == "record_ref" and parent_key == "value_ref" and isinstance(item, str):
                try:
                    translated[key] = translator(item)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    raise OntologyGroundingError(
                        "PA typed-record reference is not authorized.",
                        validation_code="evidence_reference_invalid",
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


def _write_proposal_record(  # noqa: PLR0913
    path: Path,
    *,
    proposal_number: int,
    specification_iri: str,
    feature_iri: str,
    output: Mapping[str, object],
    compiled_delta: Mapping[str, object] | None,
    status: str,
    failure: str | None,
) -> None:
    record = {
        "schema_version": 10,
        "record_type": "OntologyGroundingProposal",
        "proposal_number": proposal_number,
        "initialized_specification_iri": specification_iri,
        "feature_iri": feature_iri,
        "output": _json_clone(output),
        "compiled_delta": (None if compiled_delta is None else _json_clone(compiled_delta)),
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

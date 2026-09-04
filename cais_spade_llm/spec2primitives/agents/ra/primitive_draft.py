"""Author and persist one structural RobotAgent primitive-program draft."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingContractError,
    PAContextGroundingCompletionV4,
    PAContextGroundingCompletionV5,
    PAContextGroundingCompletionV6,
    PAContextGroundingCompletionV8,
    ProductContextView,
    load_completed_product_context_view,
    load_pa_context_grounding_completion,
)
from cais_spade_llm.spec2primitives.agents.ra.context_handoff import (
    RAContextHandoffError,
    SelectedRAAssignmentEnvelope,
    SelectedRAContextSnapshot,
    load_selected_ra_context_snapshot,
    read_phase_5_1_diagnostic,
)

_DRAFT_DIRECTORY = Path("composition/primitive_program_drafts")
_DRAFT_PREFIX = "draft_"
_DRAFT_SUFFIX = ".json"
_MODEL_OUTPUT_KEYS = {"draft_status", "primitive_symbols", "unsupported_reason"}
_PPR_DEFINES = "http://PAonto.com#defines"
_PPR_ASSEMBLY_FEATURE_ASSOCIATION = "http://PAonto.com#AssemblyFeatureAssociation"
_PPR_FEATURE = "http://PAonto.com#feature"
_PPR_HAS_PROCESS_EXECUTION = "http://PAonto.com#hasProcessExecution"
_PPR_PROCESS_EXECUTION = "http://PAonto.com#processExecution"
_PPR_REALIZES = "http://PAonto.com#realizes"
_PPR_RUNS_ON_RESOURCE = "http://PAonto.com#runsOnResource"
_PPR_RUNS_PROCESS = "http://PAonto.com#runsProcess"
_PPR_SPECIFICATION = "http://PAonto.com#specification"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_RDF_VALUE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#value"
_RECORD_KEYS = {
    "schema_version",
    "record_type",
    "draft_status",
    "pa_completion_ref",
    "pa_completion_sha256",
    "pa_completion_fingerprint",
    "assignment_ref",
    "assignment_sha256",
    "assignment_fingerprint",
    "robot_state_ref",
    "robot_state_sha256",
    "robot_state_fingerprint",
    "primitive_catalog_ref",
    "primitive_catalog_sha256",
    "primitive_catalog_fingerprint",
    "structural_steps",
    "unsupported_reason",
    "authored_at_ns",
    "fingerprint",
}


class PrimitiveDraftError(RuntimeError):
    """Raised when a structural primitive draft cannot be safely authored."""


class RobotAgentDraftRuntime(Protocol):
    """Expose the selected RobotAgent's structured composition call."""

    async def author_structural_draft(
        self,
        assignment: SelectedRAAssignmentEnvelope,
        *,
        prompt: str,
        response_format: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Return only structural primitive selection and ordering."""
        ...


@dataclass(frozen=True)
class PrimitiveProgramDraft:
    """Hold one validated append-only structural draft record."""

    path: Path
    record: Mapping[str, object]

    def to_record(self) -> dict[str, object]:
        """Return a JSON-safe copy of the persisted record."""
        return deepcopy(dict(self.record))


@dataclass(frozen=True)
class Phase52Diagnostic:
    """Describe the persisted structural-draft state for the active context."""

    status: str
    message: str
    draft_count: int = 0
    latest_draft_ref: str | None = None
    primitive_symbols: tuple[str, ...] = ()
    unsupported_reason: str | None = None
    draft: Mapping[str, object] | None = None
    composition_input: Mapping[str, object] | None = None
    failure: str | None = None

    def to_view(self) -> dict[str, object]:
        """Return the JSON-safe UI diagnostic."""
        return {
            "status": self.status,
            "message": self.message,
            "draft_count": self.draft_count,
            "latest_draft_ref": self.latest_draft_ref,
            "primitive_symbols": list(self.primitive_symbols),
            "unsupported_reason": self.unsupported_reason,
            "draft": deepcopy(dict(self.draft)) if self.draft is not None else None,
            "composition_input": (
                deepcopy(dict(self.composition_input))
                if self.composition_input is not None
                else None
            ),
            "failure": self.failure,
        }


async def author_primitive_program_draft(
    runtime: RobotAgentDraftRuntime,
    interaction_root: Path,
) -> PrimitiveProgramDraft:
    """Ask the selected RA for one unbound structural primitive sequence."""
    root = Path(interaction_root).resolve()
    try:
        context = load_selected_ra_context_snapshot(root)
        completion, completion_path = _load_completion(root)
        existing = _load_draft_history(root)
    except (GroundingContractError, RAContextHandoffError, OSError) as exc:
        raise PrimitiveDraftError(str(exc)) from exc

    catalog_ref = context.primitive_catalog_path.relative_to(root).as_posix()
    if any(draft.record["primitive_catalog_ref"] == catalog_ref for draft in existing):
        raise PrimitiveDraftError(
            "The latest Phase 5.1 context already has a PrimitiveProgramDraft."
        )

    composition_input = _composition_input(root, context, completion)
    catalog_symbols = tuple(
        str(entry["primitive_symbol"]) for entry in context.primitive_catalog.primitive_catalog
    )
    try:
        response = await runtime.author_structural_draft(
            context.assignment,
            prompt=_composition_prompt(composition_input),
            response_format=_draft_response_format(catalog_symbols),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PrimitiveDraftError(
            f"Selected RobotAgent could not author a structural draft: {exc}"
        ) from exc
    model_output = _validate_model_output(response, catalog_symbols)

    draft_number = len(existing) + 1
    path = _DRAFT_DIRECTORY / f"{_DRAFT_PREFIX}{draft_number:04d}{_DRAFT_SUFFIX}"
    absolute_path = root / path
    state_ref = context.robot_state_path.relative_to(root).as_posix()
    assignment_ref = context.assignment_path.relative_to(root).as_posix()
    completion_ref = completion_path.relative_to(root).as_posix()
    structural_steps = [
        {"step_index": index, "primitive_symbol": symbol}
        for index, symbol in enumerate(model_output["primitive_symbols"], start=1)
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "PrimitiveProgramDraft",
        "draft_status": model_output["draft_status"],
        "pa_completion_ref": completion_ref,
        "pa_completion_sha256": _sha256_path(completion_path),
        "pa_completion_fingerprint": completion.to_record()["fingerprint"],
        "assignment_ref": assignment_ref,
        "assignment_sha256": _sha256_path(context.assignment_path),
        "assignment_fingerprint": context.assignment.fingerprint,
        "robot_state_ref": state_ref,
        "robot_state_sha256": _sha256_path(context.robot_state_path),
        "robot_state_fingerprint": context.robot_state.fingerprint,
        "primitive_catalog_ref": catalog_ref,
        "primitive_catalog_sha256": _sha256_path(context.primitive_catalog_path),
        "primitive_catalog_fingerprint": context.primitive_catalog.fingerprint,
        "structural_steps": structural_steps,
        "unsupported_reason": model_output["unsupported_reason"],
        "authored_at_ns": time.time_ns(),
    }
    payload["fingerprint"] = _fingerprint(payload)
    _write_json_exclusive(absolute_path, payload)
    return _load_draft(root, absolute_path)


def read_phase_5_2_diagnostic(interaction_root: Path) -> Phase52Diagnostic:
    """Read the structural-draft state without contacting the RobotAgent."""
    root = Path(interaction_root).resolve()
    phase_5_1 = read_phase_5_1_diagnostic(root)
    if phase_5_1.status != "context_captured":
        if phase_5_1.status == "blocked":
            return Phase52Diagnostic(
                status="blocked",
                message="Phase 5.1 evidence is blocked.",
                failure=phase_5_1.failure,
            )
        return Phase52Diagnostic(
            status="waiting_for_context",
            message="Capture one valid Phase 5.1 RobotAgent context first.",
        )
    try:
        context = load_selected_ra_context_snapshot(root)
        drafts = _load_draft_history(root)
    except (RAContextHandoffError, PrimitiveDraftError, OSError) as exc:
        return Phase52Diagnostic(
            status="blocked",
            message="PrimitiveProgramDraft evidence failed validation.",
            failure=str(exc),
        )
    latest_catalog_ref = context.primitive_catalog_path.relative_to(root).as_posix()
    matching = [
        draft for draft in drafts if draft.record["primitive_catalog_ref"] == latest_catalog_ref
    ]
    if not matching:
        return Phase52Diagnostic(
            status="ready_for_draft",
            message="The latest state/catalog pair is ready for RA structural composition.",
            draft_count=len(drafts),
        )
    draft = matching[0]
    record = draft.to_record()
    draft_ref = draft.path.relative_to(root).as_posix()
    try:
        composition_input = _composition_input_for_draft(root, context, draft)
    except (GroundingContractError, PrimitiveDraftError, OSError) as exc:
        return Phase52Diagnostic(
            status="blocked",
            message="PrimitiveProgramDraft composition evidence failed validation.",
            draft_count=len(drafts),
            latest_draft_ref=draft_ref,
            failure=str(exc),
        )
    symbols = tuple(
        str(step["primitive_symbol"])
        for step in record["structural_steps"]
        if isinstance(step, Mapping)
    )
    if record["draft_status"] == "unsupported":
        return Phase52Diagnostic(
            status="unsupported",
            message="The selected RA found no structural program in its current catalog.",
            draft_count=len(drafts),
            latest_draft_ref=draft_ref,
            unsupported_reason=str(record["unsupported_reason"]),
            draft=record,
            composition_input=composition_input,
        )
    return Phase52Diagnostic(
        status="draft_authored",
        message="The selected RA authored an unbound structural primitive sequence.",
        draft_count=len(drafts),
        latest_draft_ref=draft_ref,
        primitive_symbols=symbols,
        draft=record,
        composition_input=composition_input,
    )


def _composition_input_for_draft(
    root: Path,
    context: SelectedRAContextSnapshot,
    draft: PrimitiveProgramDraft,
) -> dict[str, object]:
    """Reconstruct the exact input from the records pinned by one draft."""
    completion, completion_path = _load_completion(root)
    expected_refs = {
        "pa_completion_ref": completion_path,
        "assignment_ref": context.assignment_path,
        "robot_state_ref": context.robot_state_path,
        "primitive_catalog_ref": context.primitive_catalog_path,
    }
    for ref_field, path in expected_refs.items():
        expected_ref = path.relative_to(root).as_posix()
        if draft.record.get(ref_field) != expected_ref:
            raise PrimitiveDraftError(
                f"PrimitiveProgramDraft {ref_field} does not match its active context."
            )
    return _composition_input(root, context, completion)


def _composition_input(
    root: Path,
    context: SelectedRAContextSnapshot,
    completion: (
        PAContextGroundingCompletionV4
        | PAContextGroundingCompletionV5
        | PAContextGroundingCompletionV6
        | PAContextGroundingCompletionV8
    ),
) -> dict[str, object]:
    assignment = context.assignment
    completion_record = completion.to_record()
    try:
        product_context = load_completed_product_context_view(root)
    except GroundingContractError as exc:
        raise PrimitiveDraftError(str(exc)) from exc
    ontology_projection = _validated_ontology_projection(
        product_context,
        assignment,
        completion_record,
    )
    target_feature = _reconstructed_target_feature(
        root,
        assignment,
        completion_record,
        product_context,
    )
    typed_records = []
    for pinned in completion_record["typed_context_refs"]:
        if not isinstance(pinned, Mapping):
            raise PrimitiveDraftError("PA completion typed_context_refs are invalid.")
        record_ref = str(pinned.get("ref") or "")
        record = _read_json_mapping(_resolve_ref(root, record_ref), "typed context record")
        typed_records.append(
            {"record_type": str(record.get("record_type") or ""), "record_ref": record_ref}
        )
    return {
        "target_feature": target_feature,
        "selected_resource": {
            "resource_iri": assignment.selected_resource_iri,
            "resource_jid": assignment.selected_resource_jid,
            "execution_mode": assignment.selected_execution_mode,
        },
        "ontology_projection": ontology_projection,
        "robot_state": deepcopy(dict(context.robot_state.robot_state)),
        "primitive_catalog": [
            deepcopy(dict(entry)) for entry in context.primitive_catalog.primitive_catalog
        ],
        "grounded_context": {
            "typed_records": typed_records,
        },
    }


def _reconstructed_target_feature(  # noqa: C901
    root: Path,
    assignment: SelectedRAAssignmentEnvelope,
    completion: Mapping[str, object],
    product_context: ProductContextView,
) -> dict[str, object]:
    """Reconstruct and resolve the PA target feature from completion-pinned records."""
    proposal_path = _resolve_ref(root, str(completion["ontology_projection_ref"]))
    proposal = _read_json_mapping(proposal_path, "OntologyGroundingProposal")
    proposal_schema_version = proposal.get("schema_version")
    output = proposal.get("output")
    if (
        proposal_schema_version not in {6, 7, 8, 9, 10}
        or proposal.get("status") != "accepted"
        or proposal.get("initialized_specification_iri") != assignment.specification_iri
        or proposal.get("feature_iri") != assignment.feature_iri
        or not isinstance(output, Mapping)
        or set(output) != {"target_feature"}
    ):
        raise PrimitiveDraftError(
            "The completion-pinned target feature does not match the RA assignment."
        )
    state_names = (
        ("desired_state",) if proposal_schema_version == 6 else ("current_state", "desired_state")
    )
    authored = output["target_feature"]
    target_feature_keys = {"required_process", *state_names}
    if proposal_schema_version == 10 and completion.get("process_symbol") == "assembly":
        target_feature_keys.add("assembly_feature_association")
    if not isinstance(authored, Mapping) or set(authored) != target_feature_keys:
        raise PrimitiveDraftError("The PA-authored target_feature is invalid.")
    required_process = authored["required_process"]
    if (
        not isinstance(required_process, Mapping)
        or set(required_process) != {"process_iri", "evidence_refs"}
        or required_process.get("process_iri") != assignment.process_iri
    ):
        raise PrimitiveDraftError("The PA-authored target_feature process is invalid.")
    pinned_refs = completion.get("typed_context_refs")
    if not isinstance(pinned_refs, list):
        raise PrimitiveDraftError("PA completion typed_context_refs are invalid.")
    pinned_hashes = {
        str(item.get("ref")): str(item.get("sha256"))
        for item in pinned_refs
        if isinstance(item, Mapping)
    }
    binding_types = {
        binding.record_ref: binding.record_type
        for binding in product_context.typed_bindings
        if binding.status == "accepted"
    }
    resolved_values: list[dict[str, object]] = []
    reconstructed_states: dict[str, object] = {}
    for state_name in state_names:
        state = authored[state_name]
        resolved_values.extend(
            _resolved_authored_state_values(
                root,
                state_name,
                state,
                pinned_hashes=pinned_hashes,
                binding_types=binding_types,
                include_state_role=proposal_schema_version in {7, 8, 9, 10},
            )
        )
        assert isinstance(state, Mapping)
        reconstructed_states[state_name] = deepcopy(dict(state))
    reconstructed = {
        "product_requirement": assignment.product_requirement,
        "specification_iri": assignment.specification_iri,
        "feature_iri": assignment.feature_iri,
        "required_process": deepcopy(dict(required_process)),
        **reconstructed_states,
        "resolved_state_values": resolved_values,
    }
    if "assembly_feature_association" in authored:
        reconstructed["assembly_feature_association"] = deepcopy(
            authored["assembly_feature_association"]
        )
    return reconstructed


def _resolved_authored_state_values(  # noqa: C901
    root: Path,
    state_name: str,
    value: object,
    *,
    pinned_hashes: Mapping[str, str],
    binding_types: Mapping[str, str],
    include_state_role: bool,
) -> list[dict[str, object]]:
    """Validate and resolve all completion-pinned values for one feature state."""
    if not isinstance(value, Mapping) or set(value) != {"statement", "state_values"}:
        raise PrimitiveDraftError(f"The PA-authored {state_name} is invalid.")
    statement = value["statement"]
    state_values = value["state_values"]
    if (
        not isinstance(statement, Mapping)
        or set(statement) != {"text", "evidence_refs"}
        or not isinstance(statement.get("text"), str)
        or not str(statement["text"]).strip()
        or not isinstance(state_values, list)
    ):
        raise PrimitiveDraftError(f"The PA-authored {state_name} is invalid.")
    resolved_values: list[dict[str, object]] = []
    names: set[str] = set()
    for index, raw_value in enumerate(state_values):
        value_label = f"target_feature {state_name}.state_values[{index}]"
        if not isinstance(raw_value, Mapping) or set(raw_value) != {
            "name",
            "value_ref",
            "evidence_refs",
        }:
            raise PrimitiveDraftError(f"{value_label} is invalid.")
        name = raw_value.get("name")
        value_ref = raw_value.get("value_ref")
        evidence_refs = raw_value.get("evidence_refs")
        if (
            not isinstance(name, str)
            or not name.strip()
            or name in names
            or not isinstance(value_ref, Mapping)
            or set(value_ref) != {"record_ref", "field_path"}
            or not isinstance(evidence_refs, list)
            or not evidence_refs
            or not all(isinstance(item, str) and item for item in evidence_refs)
        ):
            raise PrimitiveDraftError(f"{value_label} is invalid.")
        names.add(name)
        record_ref = value_ref.get("record_ref")
        field_path = value_ref.get("field_path")
        if (
            not isinstance(record_ref, str)
            or record_ref not in pinned_hashes
            or record_ref not in binding_types
            or not isinstance(field_path, str)
        ):
            raise PrimitiveDraftError(f"{value_label} is not completion-pinned.")
        record_path = _resolve_ref(root, record_ref)
        if _sha256_path(record_path) != pinned_hashes[record_ref]:
            raise PrimitiveDraftError("A target-feature state record changed.")
        record = _read_json_mapping(record_path, "target-feature typed record")
        resolved = _resolve_json_pointer(record, field_path)
        if _empty_resolved_value(resolved):
            raise PrimitiveDraftError("A target-feature state value is empty.")
        resolved_item: dict[str, object] = {
            "name": name,
            "value_ref": deepcopy(dict(value_ref)),
            "evidence_refs": list(evidence_refs),
            "record_type": binding_types[record_ref],
            "record_sha256": pinned_hashes[record_ref],
            "resolved_value": _bounded_value_projection(resolved),
        }
        if include_state_role:
            resolved_item["state"] = state_name
        resolved_values.append(resolved_item)
    return resolved_values


def _validated_ontology_projection(
    product_context: ProductContextView,
    assignment: SelectedRAAssignmentEnvelope,
    completion: Mapping[str, object],
) -> dict[str, object]:
    """Return accepted assertions after proving they express the selected assignment."""
    if (
        product_context.product_requirement != assignment.product_requirement
        or product_context.tbox_fingerprint != completion.get("tbox_fingerprint")
        or product_context.abox_fingerprint != completion.get("abox_fingerprint")
    ):
        raise PrimitiveDraftError(
            "Completed ontology projection does not match the selected assignment authority."
        )

    assertions = tuple(product_context.assertions)
    _validate_projection_assertions(assertions)
    _require_iri_relation(
        assertions,
        subject=assignment.specification_iri,
        predicate=_RDF_TYPE,
        expected_object=_PPR_SPECIFICATION,
        label="specification type",
    )
    _require_literal_relation(
        assertions,
        subject=assignment.specification_iri,
        predicate=_RDF_VALUE,
        expected_object=assignment.product_requirement,
        label="specification value",
    )
    _require_iri_relation(
        assertions,
        subject=assignment.specification_iri,
        predicate=_PPR_DEFINES,
        expected_object=assignment.feature_iri,
        label="defines",
    )
    _require_iri_relation(
        assertions,
        subject=assignment.feature_iri,
        predicate=_RDF_TYPE,
        expected_object=(
            _PPR_ASSEMBLY_FEATURE_ASSOCIATION
            if completion.get("schema_version") == 8
            and completion.get("process_symbol") == "assembly"
            else _PPR_FEATURE
        ),
        label="feature type",
    )
    _require_iri_relation(
        assertions,
        subject=assignment.process_iri,
        predicate=_PPR_REALIZES,
        expected_object=assignment.feature_iri,
        label="realizes",
    )
    execution_iri = _single_iri_object(
        assertions,
        subject=assignment.specification_iri,
        predicate=_PPR_HAS_PROCESS_EXECUTION,
        label="hasProcessExecution",
    )
    _require_iri_relation(
        assertions,
        subject=execution_iri,
        predicate=_RDF_TYPE,
        expected_object=_PPR_PROCESS_EXECUTION,
        label="processExecution type",
    )
    _require_iri_relation(
        assertions,
        subject=execution_iri,
        predicate=_PPR_RUNS_PROCESS,
        expected_object=assignment.process_iri,
        label="runsProcess",
    )
    _require_iri_relation(
        assertions,
        subject=execution_iri,
        predicate=_PPR_RUNS_ON_RESOURCE,
        expected_object=assignment.selected_resource_iri,
        label="runsOnResource",
    )
    return {
        "tbox_fingerprint": product_context.tbox_fingerprint,
        "abox_fingerprint": product_context.abox_fingerprint,
        "assertions": [deepcopy(dict(assertion)) for assertion in assertions],
    }


def _validate_projection_assertions(assertions: tuple[Mapping[str, object], ...]) -> None:
    if not assertions:
        raise PrimitiveDraftError("Completed ontology projection has no assertions.")
    seen: set[str] = set()
    for assertion in assertions:
        if set(assertion) != {"subject", "predicate", "object"}:
            raise PrimitiveDraftError("Completed ontology projection assertion is invalid.")
        subject = assertion.get("subject")
        predicate = assertion.get("predicate")
        object_value = assertion.get("object")
        if (
            not isinstance(subject, str)
            or not subject
            or not isinstance(predicate, str)
            or not predicate
            or not isinstance(object_value, Mapping)
            or not isinstance(object_value.get("kind"), str)
            or not isinstance(object_value.get("value"), str)
            or not object_value.get("value")
        ):
            raise PrimitiveDraftError("Completed ontology projection assertion is invalid.")
        encoded = json.dumps(assertion, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if encoded in seen:
            raise PrimitiveDraftError(
                "Completed ontology projection contains a duplicate assertion."
            )
        seen.add(encoded)


def _require_iri_relation(
    assertions: tuple[Mapping[str, object], ...],
    *,
    subject: str,
    predicate: str,
    expected_object: str,
    label: str,
) -> None:
    objects = _iri_objects(assertions, subject=subject, predicate=predicate)
    if objects != [expected_object]:
        raise PrimitiveDraftError(
            f"Completed ontology projection {label} does not match the selected assignment."
        )


def _require_literal_relation(
    assertions: tuple[Mapping[str, object], ...],
    *,
    subject: str,
    predicate: str,
    expected_object: str,
    label: str,
) -> None:
    objects = [
        str(assertion["object"]["value"])
        for assertion in assertions
        if assertion["subject"] == subject
        and assertion["predicate"] == predicate
        and assertion["object"]["kind"] == "literal"
    ]
    if objects != [expected_object]:
        raise PrimitiveDraftError(
            f"Completed ontology projection {label} does not match the selected assignment."
        )


def _single_iri_object(
    assertions: tuple[Mapping[str, object], ...],
    *,
    subject: str,
    predicate: str,
    label: str,
) -> str:
    objects = _iri_objects(assertions, subject=subject, predicate=predicate)
    if len(objects) != 1:
        raise PrimitiveDraftError(
            f"Completed ontology projection {label} must identify exactly one processExecution."
        )
    return objects[0]


def _iri_objects(
    assertions: tuple[Mapping[str, object], ...],
    *,
    subject: str,
    predicate: str,
) -> list[str]:
    return [
        str(assertion["object"]["value"])
        for assertion in assertions
        if assertion["subject"] == subject
        and assertion["predicate"] == predicate
        and assertion["object"]["kind"] == "iri"
    ]


def _resolve_json_pointer(document: object, field_path: str) -> object:
    """Resolve one non-root RFC 6901 pointer against a pinned typed record."""
    if not field_path.startswith("/"):
        raise PrimitiveDraftError("target_feature field_path is not a JSON Pointer.")
    current = document
    for raw_token in field_path.split("/")[1:]:
        token = _decode_json_pointer_token(raw_token)
        if isinstance(current, Mapping):
            if token not in current:
                raise PrimitiveDraftError("target_feature field_path does not exist.")
            current = current[token]
            continue
        if isinstance(current, list):
            if token == "-" or not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise PrimitiveDraftError("target_feature field_path array index is invalid.")
            index = int(token)
            if index >= len(current):
                raise PrimitiveDraftError("target_feature field_path does not exist.")
            current = current[index]
            continue
        raise PrimitiveDraftError("target_feature field_path traverses a scalar.")
    return current


def _decode_json_pointer_token(token: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(token):
        if token[index] != "~":
            result.append(token[index])
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            raise PrimitiveDraftError("target_feature field_path escaping is invalid.")
        result.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(result)


def _empty_resolved_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (Mapping, list, tuple)):
        return not value
    return False


def _bounded_value_projection(value: object, *, depth: int = 0) -> object:
    """Return a deterministic bounded JSON projection for the RA prompt."""
    if depth >= 5:
        return {"projection_truncated": True}
    if isinstance(value, str):
        return value if len(value) <= 2048 else f"{value[:2048]}…"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        projection = {
            str(key): _bounded_value_projection(item, depth=depth + 1) for key, item in items[:32]
        }
        if len(items) > 32:
            projection["projection_truncated"] = True
        return projection
    if isinstance(value, (list, tuple)):
        projection = [_bounded_value_projection(item, depth=depth + 1) for item in value[:32]]
        if len(value) > 32:
            projection.append({"projection_truncated": True})
        return projection
    raise PrimitiveDraftError("target_feature resolved value is not JSON-compatible.")


def _composition_prompt(composition_input: Mapping[str, object]) -> str:
    return (
        "You are the exact selected RobotAgent. Treat target_feature as the "
        "authoritative semantic product outcome. Semantically select and order only "
        "exact primitive symbols supplied in primitive_catalog to express that target "
        "feature. Repetition is allowed when semantically necessary. Deterministic "
        "validation checks authority, lineage, resolved values, and exact symbols; it "
        "does not choose the primitive sequence. Do not bind parameters, invent a "
        "primitive, execute anything, or claim feasibility. Missing product or scene "
        "values are deferred to binding preflight and do not by themselves make the "
        "catalog unsupported. Return unsupported only when no ordered use of supplied "
        "primitives can express the target feature.\n\nCOMPOSITION_INPUT\n"
        + json.dumps(composition_input, sort_keys=True, ensure_ascii=False)
    )


def _draft_response_format(catalog_symbols: tuple[str, ...]) -> dict[str, object]:
    if not catalog_symbols:
        raise PrimitiveDraftError("Primitive catalog must be non-empty.")
    return {
        "name": "spec2primitives_primitive_program_draft",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(_MODEL_OUTPUT_KEYS),
            "properties": {
                "draft_status": {
                    "type": "string",
                    "enum": ["proposed", "unsupported"],
                },
                "primitive_symbols": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(catalog_symbols),
                    },
                    "maxItems": 32,
                },
                "unsupported_reason": {
                    "type": ["string", "null"],
                    "minLength": 1,
                },
            },
        },
    }


def _validate_model_output(
    value: Mapping[str, object],
    catalog_symbols: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _MODEL_OUTPUT_KEYS:
        raise PrimitiveDraftError("RobotAgent structural draft fields are invalid.")
    status = value.get("draft_status")
    symbols_value = value.get("primitive_symbols")
    if not isinstance(symbols_value, list):
        raise PrimitiveDraftError("RobotAgent primitive_symbols must be an array.")
    symbols = []
    for symbol in symbols_value:
        if not isinstance(symbol, str) or symbol not in catalog_symbols:
            raise PrimitiveDraftError(
                "RobotAgent structural draft contains an unknown primitive symbol."
            )
        symbols.append(symbol)
    reason = value.get("unsupported_reason")
    if status == "proposed":
        if not symbols or len(symbols) > 32 or reason is not None:
            raise PrimitiveDraftError("Proposed structural draft is invalid.")
    elif status == "unsupported":
        if symbols or not isinstance(reason, str) or not reason.strip():
            raise PrimitiveDraftError("Unsupported structural draft is invalid.")
    else:
        raise PrimitiveDraftError("RobotAgent draft_status is invalid.")
    return {
        "draft_status": status,
        "primitive_symbols": symbols,
        "unsupported_reason": reason,
    }


def _load_completion(
    root: Path,
) -> tuple[
    PAContextGroundingCompletionV4
    | PAContextGroundingCompletionV5
    | PAContextGroundingCompletionV6
    | PAContextGroundingCompletionV8,
    Path,
]:
    completion = load_pa_context_grounding_completion(root)
    if not isinstance(
        completion,
        (
            PAContextGroundingCompletionV4,
            PAContextGroundingCompletionV5,
            PAContextGroundingCompletionV6,
            PAContextGroundingCompletionV8,
        ),
    ):
        raise PrimitiveDraftError(
            "Phase 5.2 requires a supported PA completion; version 7 is audit-only."
        )
    paths = sorted((root / "interaction_record").glob("context_completion_*.json"))
    if len(paths) != 1:
        raise PrimitiveDraftError("Phase 5.2 requires exactly one PA completion.")
    return completion, paths[0]


def _load_draft_history(root: Path) -> list[PrimitiveProgramDraft]:
    directory = root / _DRAFT_DIRECTORY
    if not directory.exists():
        return []
    paths = sorted(directory.glob(f"{_DRAFT_PREFIX}*{_DRAFT_SUFFIX}"))
    expected = [
        directory / f"{_DRAFT_PREFIX}{index:04d}{_DRAFT_SUFFIX}"
        for index in range(1, len(paths) + 1)
    ]
    if paths != expected:
        raise PrimitiveDraftError("PrimitiveProgramDraft history is not contiguous.")
    drafts = [_load_draft(root, path) for path in paths]
    catalog_refs = [str(draft.record["primitive_catalog_ref"]) for draft in drafts]
    if len(catalog_refs) != len(set(catalog_refs)):
        raise PrimitiveDraftError(
            "PrimitiveProgramDraft history contains duplicate context inputs."
        )
    return drafts


def _load_draft(root: Path, path: Path) -> PrimitiveProgramDraft:
    value = _read_json_mapping(path, "PrimitiveProgramDraft")
    if set(value) != _RECORD_KEYS:
        raise PrimitiveDraftError("PrimitiveProgramDraft fields are invalid.")
    payload = dict(value)
    fingerprint = payload.pop("fingerprint", None)
    if (
        value.get("schema_version") != 1
        or value.get("record_type") != "PrimitiveProgramDraft"
        or not isinstance(fingerprint, str)
        or fingerprint != _fingerprint(payload)
    ):
        raise PrimitiveDraftError("PrimitiveProgramDraft identity is invalid.")
    pinned = (
        ("pa_completion_ref", "pa_completion_sha256", "pa_completion_fingerprint"),
        ("assignment_ref", "assignment_sha256", "assignment_fingerprint"),
        ("robot_state_ref", "robot_state_sha256", "robot_state_fingerprint"),
        (
            "primitive_catalog_ref",
            "primitive_catalog_sha256",
            "primitive_catalog_fingerprint",
        ),
    )
    loaded_records: dict[str, Mapping[str, object]] = {}
    for ref_field, sha_field, fingerprint_field in pinned:
        record_path = _resolve_ref(root, str(value.get(ref_field) or ""))
        if _sha256_path(record_path) != value.get(sha_field):
            raise PrimitiveDraftError(f"PrimitiveProgramDraft {ref_field} hash is invalid.")
        record = _read_json_mapping(record_path, ref_field)
        if record.get("fingerprint") != value.get(fingerprint_field):
            raise PrimitiveDraftError(f"PrimitiveProgramDraft {ref_field} fingerprint is invalid.")
        loaded_records[ref_field] = record
    catalog = loaded_records["primitive_catalog_ref"]
    entries = catalog.get("primitive_catalog")
    if not isinstance(entries, list):
        raise PrimitiveDraftError("PrimitiveProgramDraft catalog reference is invalid.")
    catalog_symbols = tuple(
        str(entry.get("primitive_symbol") or "") for entry in entries if isinstance(entry, Mapping)
    )
    steps = value.get("structural_steps")
    if not isinstance(steps, list) or any(
        not isinstance(step, Mapping)
        or set(step) != {"step_index", "primitive_symbol"}
        or step.get("step_index") != index
        for index, step in enumerate(steps, start=1)
    ):
        raise PrimitiveDraftError("PrimitiveProgramDraft structural_steps are invalid.")
    output = {
        "draft_status": value.get("draft_status"),
        "primitive_symbols": [step["primitive_symbol"] for step in steps],
        "unsupported_reason": value.get("unsupported_reason"),
    }
    _validate_model_output(output, catalog_symbols)
    authored_at_ns = value.get("authored_at_ns")
    if (
        isinstance(authored_at_ns, bool)
        or not isinstance(authored_at_ns, int)
        or authored_at_ns < 0
    ):
        raise PrimitiveDraftError("PrimitiveProgramDraft authored_at_ns is invalid.")
    return PrimitiveProgramDraft(path=path, record=dict(value))


def _resolve_ref(root: Path, record_ref: str) -> Path:
    relative = Path(record_ref)
    if (
        relative.is_absolute()
        or not relative.parts
        or "." in relative.parts
        or ".." in relative.parts
    ):
        raise PrimitiveDraftError("PrimitiveProgramDraft record ref is invalid.")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PrimitiveDraftError(
            "PrimitiveProgramDraft record ref leaves its interaction."
        ) from exc
    if not path.is_file():
        raise PrimitiveDraftError("PrimitiveProgramDraft pinned record is unavailable.")
    return path


def _read_json_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrimitiveDraftError(f"{label} could not be read.") from exc
    if not isinstance(value, dict):
        raise PrimitiveDraftError(f"{label} must be a JSON object.")
    return value


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise PrimitiveDraftError(f"PrimitiveProgramDraft already exists: {path.name}.") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise PrimitiveDraftError(
            f"PrimitiveProgramDraft could not be written: {path.name}."
        ) from exc


def _fingerprint(value: Mapping[str, object] | list[object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "Phase52Diagnostic",
    "PrimitiveDraftError",
    "PrimitiveProgramDraft",
    "RobotAgentDraftRuntime",
    "author_primitive_program_draft",
    "read_phase_5_2_diagnostic",
]

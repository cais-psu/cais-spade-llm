from __future__ import annotations

"""Reconstruct grounded composition inputs for the exact selected RobotAgent."""

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from cais_spade_llm.spec2primitives.agents.pa.grounding_contracts import (
    GroundingContractError,
    PAContextGroundingCompletion,
    ProductContextView,
    load_completed_product_context_view,
    load_pa_context_grounding_completion,
)

from .context_handoff import (
    RAContextHandoffError,
    SelectedRAAssignmentEnvelope,
    SelectedRAContextSnapshot,
)

_PPR_DEFINES = "http://PAonto.com#defines"
_PPR_FEATURE = "http://PAonto.com#feature"
_PPR_HAS_PROCESS_EXECUTION = "http://PAonto.com#hasProcessExecution"
_PPR_PROCESS_EXECUTION = "http://PAonto.com#processExecution"
_PPR_REALIZES = "http://PAonto.com#realizes"
_PPR_RUNS_ON_RESOURCE = "http://PAonto.com#runsOnResource"
_PPR_RUNS_PROCESS = "http://PAonto.com#runsProcess"
_PPR_SPECIFICATION = "http://PAonto.com#specification"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_RDF_VALUE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#value"


_EXCLUDED_STATE_FIELDS = frozenset(
    {
        "recovery_adapter", "function_names", "capability_decompositions",
        "primitive_steps", "composite_expansion", "example_families",
        "examples", "recovery_examples", "recovery_instructions",
    }
)


def _composition_state_view(value: object) -> Any:
    """Exclude recovery recipes from every model-facing state projection."""
    if isinstance(value, Mapping):
        return {
            name: _composition_state_view(item)
            for name, item in value.items()
            if name not in _EXCLUDED_STATE_FIELDS
        }
    if isinstance(value, list):
        return [_composition_state_view(item) for item in value]
    return deepcopy(value)


def _composition_catalog_view(entries: tuple[Mapping[str, object], ...]) -> list[dict[str, Any]]:
    """Project composition contracts while retaining full runtime snapshots."""
    result = [_without_model_name(entry) for entry in entries]
    for entry in result:
        symbol = entry["primitive_symbol"]
        if symbol in {"compute_pick_targets", "compute_place_targets"}:
            schemas = entry.get("parameter_schemas", {})
            geometry = schemas.get("product_geometry")
            if isinstance(geometry, dict):
                geometry["x-grounding-required"] = True
                geometry["description"] = (
                    "Explicit world geometry for the selected calculation; missing measurements "
                    "remain unbound and no controller geometry defaults are used."
                )
                if symbol == "compute_place_targets":
                    fields = geometry.setdefault("x-grounding-fields", [])
                    properties = geometry.setdefault("properties", {})
                    for name, required in (
                        ("target_reference", ["target_point"]),
                        ("target_origin_pose", ["x", "y", "z"]),
                    ):
                        if name not in fields:
                            fields.append(name)
                        declaration = properties.setdefault(name, {"type": "object"})
                        required_fields = declaration.setdefault("x-grounding-fields", [])
                        for field in required:
                            if field not in required_fields:
                                required_fields.append(field)
                    properties["target_reference"].setdefault("properties", {}).setdefault(
                        "target_point", {"type": "string"}
                    )["description"] = (
                        "part_origin or inserted_part_origin; the final part origin must be established."
                    )
                    origin = properties["target_origin_pose"].setdefault("properties", {})
                    for axis in ("x", "y", "z"):
                        origin.setdefault(axis, {"type": "number", "x-frame-source": "world"})
            if symbol == "compute_place_targets" and isinstance(schemas.get("pick_ctx"), dict):
                schemas["pick_ctx"]["description"] = (
                    "Selected pick result or measured held-part context; grasp/tool offsets "
                    "must be explicitly bound and no controller fallbacks are used."
                )
            if symbol == "compute_pick_targets" and isinstance(schemas.get("target_pose"), dict):
                schemas["target_pose"]["description"] = (
                    "Observed product location in world metres, not an end-effector target. "
                    "The current geometry validator requires an established CAD_origin "
                    "reference to check the grasp offset; an observed candidate center alone "
                    "does not establish that reference."
                )
        if entry["primitive_symbol"] in {"grasp_part", "release_part"}:
            for field in ("conditions", "effects"):
                entry[field] = {
                    name: value for name, value in entry[field].items() if name == "held_part"
                }
    return result


def _without_model_name(value: Any) -> Any:
    """Keep the simulator binding out of nested schemas and custody effects."""
    if isinstance(value, Mapping):
        return {
            name: _without_model_name(item)
            for name, item in value.items()
            if name != "model_name"
        }
    if isinstance(value, (list, tuple)):
        # Declarations also name fields in required lists, typed rows and effects.
        items = [
            _without_model_name(item)
            for item in value
            if item != "model_name"
            and not (isinstance(item, Mapping) and item.get("name") == "model_name")
        ]
        return tuple(items) if isinstance(value, tuple) else items
    return deepcopy(value)


def _composition_input(
    root: Path,
    context: SelectedRAContextSnapshot,
    completion: PAContextGroundingCompletion,
) -> dict[str, object]:
    assignment = context.assignment
    completion_record = completion.to_record()
    try:
        product_context = load_completed_product_context_view(root)
    except GroundingContractError as exc:
        raise RAContextHandoffError(str(exc)) from exc
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
            raise RAContextHandoffError("PA completion typed_context_refs are invalid.")
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
        "robot_state": _composition_state_view(context.robot_state.robot_state),
        "primitive_catalog": _composition_catalog_view(context.primitive_catalog.primitive_catalog),
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
    output = proposal.get("output")
    if (
        (proposal.get("status") != "accepted")
        or (proposal.get("initialized_specification_iri") != assignment.specification_iri)
        or (proposal.get("feature_iri") != assignment.feature_iri)
        or (not isinstance(output, Mapping))
        or (set(output) != {"target_feature"})
    ):
        raise RAContextHandoffError(
            "The completion-pinned target feature does not match the RA assignment."
        )
    state_names = ("current_state", "desired_state")
    authored = output["target_feature"]
    target_feature_keys = {"required_process", *state_names}
    if completion.get("process_symbol") == "assembly":
        target_feature_keys.add("assembly_feature_association")
    if not isinstance(authored, Mapping) or set(authored) != target_feature_keys:
        raise RAContextHandoffError("The PA-authored target_feature is invalid.")
    required_process = authored["required_process"]
    if (
        not isinstance(required_process, Mapping)
        or set(required_process) != {"process_iri", "evidence_refs"}
        or required_process.get("process_iri") != assignment.process_iri
    ):
        raise RAContextHandoffError("The PA-authored target_feature process is invalid.")
    pinned_refs = completion.get("typed_context_refs")
    if not isinstance(pinned_refs, list):
        raise RAContextHandoffError("PA completion typed_context_refs are invalid.")
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
                include_state_role=True,
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
        raise RAContextHandoffError(f"The PA-authored {state_name} is invalid.")
    statement = value["statement"]
    state_values = value["state_values"]
    if (
        not isinstance(statement, Mapping)
        or set(statement) != {"text", "evidence_refs"}
        or not isinstance(statement.get("text"), str)
        or not str(statement["text"]).strip()
        or not isinstance(state_values, list)
    ):
        raise RAContextHandoffError(f"The PA-authored {state_name} is invalid.")
    resolved_values: list[dict[str, object]] = []
    names: set[str] = set()
    for index, raw_value in enumerate(state_values):
        value_label = f"target_feature {state_name}.state_values[{index}]"
        if not isinstance(raw_value, Mapping) or set(raw_value) != {
            "name",
            "value_ref",
            "evidence_refs",
        }:
            raise RAContextHandoffError(f"{value_label} is invalid.")
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
            raise RAContextHandoffError(f"{value_label} is invalid.")
        names.add(name)
        record_ref = value_ref.get("record_ref")
        field_path = value_ref.get("field_path")
        if (
            not isinstance(record_ref, str)
            or record_ref not in pinned_hashes
            or record_ref not in binding_types
            or not isinstance(field_path, str)
        ):
            raise RAContextHandoffError(f"{value_label} is not completion-pinned.")
        record_path = _resolve_ref(root, record_ref)
        if _sha256_path(record_path) != pinned_hashes[record_ref]:
            raise RAContextHandoffError("A target-feature state record changed.")
        record = _read_json_mapping(record_path, "target-feature typed record")
        resolved = _resolve_json_pointer(record, field_path)
        if _empty_resolved_value(resolved):
            raise RAContextHandoffError("A target-feature state value is empty.")
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
        raise RAContextHandoffError(
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
        expected_object=(_PPR_FEATURE),
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
        raise RAContextHandoffError("Completed ontology projection has no assertions.")
    seen: set[str] = set()
    for assertion in assertions:
        if set(assertion) != {"subject", "predicate", "object"}:
            raise RAContextHandoffError("Completed ontology projection assertion is invalid.")
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
            raise RAContextHandoffError("Completed ontology projection assertion is invalid.")
        encoded = json.dumps(assertion, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if encoded in seen:
            raise RAContextHandoffError(
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
        raise RAContextHandoffError(
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
        raise RAContextHandoffError(
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
        raise RAContextHandoffError(
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
        raise RAContextHandoffError("target_feature field_path is not a JSON Pointer.")
    current = document
    for raw_token in field_path.split("/")[1:]:
        token = _decode_json_pointer_token(raw_token)
        if isinstance(current, Mapping):
            if token not in current:
                raise RAContextHandoffError("target_feature field_path does not exist.")
            current = current[token]
            continue
        if isinstance(current, list):
            if token == "-" or not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise RAContextHandoffError("target_feature field_path array index is invalid.")
            index = int(token)
            if index >= len(current):
                raise RAContextHandoffError("target_feature field_path does not exist.")
            current = current[index]
            continue
        raise RAContextHandoffError("target_feature field_path traverses a scalar.")
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
            raise RAContextHandoffError("target_feature field_path escaping is invalid.")
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
    raise RAContextHandoffError("target_feature resolved value is not JSON-compatible.")


def _load_completion(root: Path) -> tuple[PAContextGroundingCompletion, Path]:
    """Require current validated completion before primitive composition."""
    completion = load_pa_context_grounding_completion(root)
    if not isinstance(completion, PAContextGroundingCompletion):
        raise RAContextHandoffError(
            "Historical grounding has an incompatible completion contract. "
            "Start a fresh interaction before primitive composition."
        )
    paths = sorted((root / "interaction_record").glob("context_completion_*.json"))
    if len(paths) != 1:
        raise RAContextHandoffError(
            "Structural primitive composition requires exactly one PA completion."
        )
    return completion, paths[0]


def _resolve_ref(root: Path, record_ref: str) -> Path:
    relative = Path(record_ref)
    if (
        relative.is_absolute()
        or not relative.parts
        or "." in relative.parts
        or ".." in relative.parts
    ):
        raise RAContextHandoffError("Composition context record ref is invalid.")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RAContextHandoffError(
            "Composition context record ref leaves its interaction."
        ) from exc
    if not path.is_file():
        raise RAContextHandoffError("Composition context pinned record is unavailable.")
    return path


def _read_json_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RAContextHandoffError(f"{label} could not be read.") from exc
    if not isinstance(value, dict):
        raise RAContextHandoffError(f"{label} must be a JSON object.")
    return value


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

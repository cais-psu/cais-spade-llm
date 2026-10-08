"""Validate and compile given safety definitions without interpreting their text."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
    _catalog,
    _digest,
    _list,
    _object,
    _symbol,
)

from cais_spade_llm.agents.central_controller.ppr_ap import physical_ap_kind, physical_binding_fields

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Predefined safety contains a duplicate field: {key}")
        result[key] = value
    return result


def parse_predefined_safety(text: str) -> dict[str, Any] | None:
    """Read a predefined document, or return None for legacy requirement text.

    JSON-looking input is never sent to the natural-language generation path
    when its mode, version, or structure is invalid.
    """
    if not text.lstrip().startswith("{"):
        return None
    document = json.loads(text, object_pairs_hook=_unique_object)
    _object(document, "predefined safety")
    required = {"mode", "version", "catalog", "requirement_scopes"}
    if set(document) - required - {"product_geometry"} or not required <= set(document):
        raise ValueError("Predefined safety requires mode, version, catalog and requirement_scopes")
    if document["mode"] != "predefined" or type(document["version"]) is not int or document["version"] != 2:
        raise ValueError("Unsupported predefined safety mode or version; recompile required")
    _object(document["catalog"], "predefined catalog")
    _list(document["requirement_scopes"], "predefined requirement_scopes")
    if "product_geometry" in document:
        source = _object(document["product_geometry"], "product_geometry")
        if set(source) != {"path", "environment"}:
            raise ValueError("product_geometry requires its owner path and environment")
        for key, value in source.items():
            _symbol(value, f"product_geometry.{key}")
    return deepcopy(document)


def _geometry(document: dict[str, Any]) -> tuple[dict[str, Any], str]:
    source = document.get("product_geometry")
    if source is None:
        return {}, ""
    path = Path(source["path"])
    path = path if path.is_absolute() else _PROJECT_ROOT / path
    raw = path.read_bytes()
    geometry = _object(json.loads(raw), "product geometry document")
    selected = _object(geometry.get(source["environment"]), "product geometry environment")
    return selected, hashlib.sha256(raw).hexdigest()


def _validate_scopes(document: dict[str, Any], definitions: dict[str, Any]) -> None:
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
        _physical_bindings,
    )

    seen, used = set(), set()
    geometry, _ = _geometry(document)
    for scope in document["requirement_scopes"]:
        scope = _object(scope, "requirement scope")
        identifier = _symbol(scope.get("specification"), "scope.specification")
        if identifier not in definitions:
            raise ValueError("Predefined scope references an undeclared specification")
        definition = definitions[identifier]
        physical = [ap for ap in definition["aps"] if physical_ap_kind(ap) is not None]
        structured = [ap for ap in definition["aps"] if physical_ap_kind(ap) is None]
        expected = {"specification"}
        if "physical_ap_bindings" in scope:
            bindings = _physical_bindings(definition, scope["physical_ap_bindings"])
            expected.add("physical_ap_bindings")
        else:
            fields = set().union(*(physical_binding_fields(ap) for ap in physical))
            fields.discard("resources")
            expected.update(fields)
            bindings = {ap["label"]: {key: scope.get(key) for key in fields} for ap in physical}
            for key in fields:
                _symbol(scope.get(key), f"scope.{key}")
        if structured:
            expected.add("ap_groundings")
            groundings = _object(scope.get("ap_groundings"), "scope.ap_groundings")
            if set(groundings) != {ap["label"] for ap in structured}:
                raise ValueError("Every structured AP needs its explicit grounding")
        if set(scope) != expected:
            raise ValueError("Predefined scope fields do not match its AP bindings")
        for binding in bindings.values():
            if "target" not in binding:
                continue
            targets = _object(_object(geometry.get("parts"), "product geometry parts").get(
                "assembly_target_map"), "assembly_target_map")
            if targets.get(binding["part"]) != binding["target"]:
                raise ValueError("Predefined assembly target disagrees with its configured owner")
        identity = _digest(scope)
        if identity in seen:
            raise ValueError("Duplicate predefined requirement scope")
        seen.add(identity)
        used.add(identifier)
    if used != set(definitions):
        raise ValueError("Every predefined specification requires a declared scope")


def compile_predefined_safety(document: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Compile every given formula with its original rule and AP identities.

    Returns:
        Existing safety-artifact rule records and one strict DFA per rule.
        Compilation proves no physical grounding or execution feasibility.
    """
    parsed = parse_predefined_safety(json.dumps(document, allow_nan=False))
    if parsed is None:
        raise ValueError("A predefined safety document is required")
    definitions = _catalog(parsed["catalog"], structured_groundings=True)
    _validate_scopes(parsed, definitions)
    rules, dfas = [], {}
    for identifier, definition in definitions.items():
        if Path(identifier).name != identifier or identifier in {".", ".."}:
            raise ValueError("A predefined rule identifier cannot contain path components")
        rules.append({
            "id": identifier,
            "raw_text": definition["requirement"],
            "aps": deepcopy(definition["aps"]),
            "ltlf": definition["formula"],
            "requirement_scopes": [deepcopy(scope) for scope in parsed["requirement_scopes"]
                                   if scope["specification"] == identifier],
        })
        dfas[identifier] = definition["dfa_dot"]
    return rules, dfas


def predefined_safety_metadata(document: dict[str, Any], source_text: str) -> dict[str, Any]:
    """Bind saved rules to the entire given document and configured geometry."""
    if parse_predefined_safety(source_text) != document:
        raise ValueError("Predefined source text disagrees with the supplied document")
    return {
        "mode": "predefined",
        "ap_schema_version": 2,
        "predefined_safety": deepcopy(document),
        "predefined_source_sha256": hashlib.sha256(source_text.strip().encode("utf-8")).hexdigest(),
        "predefined_semantics_sha256": _digest(document),
        "predefined_geometry_sha256": _geometry(document)[1],
    }


def validate_predefined_safety_artifact(
    payload: dict[str, Any], *, source_text: str | None = None,
) -> dict[str, Any] | None:
    """Reject changed given definitions, bindings, formulas, or source evidence.

    Returns:
        The unchanged predefined document, or None for a legacy artifact.
    """
    payload = _object(payload, "predefined artifact")
    fields = {"predefined_safety", "predefined_source_sha256", "predefined_semantics_sha256",
              "predefined_geometry_sha256"}
    if payload.get("mode") != "predefined" and not fields.intersection(payload):
        if source_text is not None and parse_predefined_safety(source_text) is not None:
            raise ValueError("Predefined source cannot load an unmarked legacy safety artifact; recompile required")
        return None
    if payload.get("ap_schema_version") != 2:
        raise ValueError("Legacy AP artifacts are unsupported; recompile required")
    if payload.get("mode") != "predefined" or not fields <= set(payload):
        raise ValueError("Predefined safety artifact metadata is incomplete")
    document = payload["predefined_safety"]
    rules, _ = compile_predefined_safety(document)
    if payload["predefined_semantics_sha256"] != _digest(document):
        raise ValueError("Predefined definitions or bindings changed after compilation")
    if payload["predefined_geometry_sha256"] != _geometry(document)[1]:
        raise ValueError("Predefined product geometry changed after compilation")
    source_hash = _symbol(payload["predefined_source_sha256"], "predefined_source_sha256")
    if source_hash != payload.get("safety_text_sha256"):
        raise ValueError("Predefined source fingerprint is inconsistent")
    if source_text is not None and (
        parse_predefined_safety(source_text) != document or hashlib.sha256(
            source_text.strip().encode("utf-8")).hexdigest() != source_hash
    ):
        raise ValueError("Predefined safety source changed after compilation")
    supplied = _list(payload.get("rules"), "predefined artifact.rules")
    if len(supplied) != len(rules):
        raise ValueError("Predefined artifact has missing or additional rules")
    for expected, actual in zip(rules, supplied, strict=True):
        actual = _object(actual, "predefined artifact rule")
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("Predefined artifact changes a given rule, formula, AP, or scope")
    return deepcopy(document)

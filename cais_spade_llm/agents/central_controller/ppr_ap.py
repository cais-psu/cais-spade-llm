"""Context-free, typed Product-Process-Resource atomic propositions.

Definitions are authoritative structured data. Canonical JSON keys are stable
monitor identities, never slash-separated fields.
"""

from __future__ import annotations

import json
from copy import deepcopy
from math import isfinite
from typing import Any


def validate_ap_definition(definition: dict[str, Any]) -> dict[str, Any]:
    """Validate an AP without changing any supplied manufacturing symbol."""
    if not isinstance(definition, dict):
        raise ValueError("An AP definition must be a structured PPR object")
    kind = definition.get("kind")
    if kind not in {"ap_state", "ap_event"}:
        raise ValueError("An AP kind must be ap_state or ap_event")
    condition_key = "state" if kind == "ap_state" else "event"
    if set(definition) != {"kind", "product", "process", "resource", condition_key}:
        raise ValueError("PPR APs require product, process, resource and a typed condition; context is unsupported")
    for field in ("product", "process", "resource"):
        if not isinstance(definition[field], str) or not definition[field]:
            raise ValueError(f"AP {field} must be an exact nonempty symbol")
    condition = definition[condition_key]
    if not isinstance(condition, dict) or set(condition) != {"symbol", "arguments"}:
        raise ValueError("An AP condition requires symbol and arguments")
    if not isinstance(condition["symbol"], str) or not condition["symbol"]:
        raise ValueError("An AP condition needs an exact nonempty symbol")
    arguments = condition["arguments"]
    if not isinstance(arguments, dict):
        raise ValueError("AP condition arguments must be an object")
    for key, value in arguments.items():
        if not isinstance(key, str) or not key or key == "context":
            raise ValueError("AP condition arguments require named fields, never context")
        if value is not None and type(value) not in (str, bool, int, float):
            raise ValueError("AP condition arguments must have scalar values")
        if type(value) is float and not isfinite(value):
            raise ValueError("AP condition arguments must be finite")
    return deepcopy(definition)


def canonical_ap_key(definition: dict[str, Any]) -> str:
    """Return an identity derived only from a validated definition."""
    return json.dumps(
        validate_ap_definition(definition), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )


def parse_ap_definition(value: dict[str, Any] | str) -> dict[str, Any]:
    """Read a PPR definition, rejecting retired slash descriptors explicitly."""
    if isinstance(value, dict):
        return validate_ap_definition(value)
    if not isinstance(value, str) or not value.startswith("{"):
        raise ValueError("Legacy AP descriptors are unsupported; recompile safety with context-free PPR definitions")
    def unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError("Duplicate AP definition field")
            result[key] = item
        return result
    definition = validate_ap_definition(json.loads(value, object_pairs_hook=unique_object))
    if value != canonical_ap_key(definition):
        raise ValueError("AP identity is not its canonical structured definition")
    return definition


def parse_ap_record(ap: dict[str, Any]) -> dict[str, Any]:
    """Read a record and reject a key that disagrees with its definition."""
    if not isinstance(ap, dict):
        raise ValueError("An AP record must be an object")
    if "context" in ap:
        raise ValueError("AP context is unsupported; use typed condition arguments")
    definition = parse_ap_definition(ap.get("definition", ap.get("full")))
    if "full" in ap and ap["full"] != canonical_ap_key(definition):
        raise ValueError("AP key disagrees with its typed definition")
    return definition


def ap_record(label: str, definition: dict[str, Any], meaning: str, **metadata: Any) -> dict[str, Any]:
    """Build a monitor-facing record from an authoritative typed definition."""
    if set(metadata) & {"label", "full", "definition", "meaning", "context"}:
        raise ValueError("AP metadata cannot replace its definition or add context")
    validated = validate_ap_definition(definition)
    return {"label": label, "full": canonical_ap_key(validated), "definition": validated,
            "meaning": meaning, **deepcopy(metadata)}


def make_ap_definition(
    kind: str, product: str, process: str, resource: str,
    symbol: str, arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a typed condition preserving exact supplied symbols."""
    return validate_ap_definition({
        "kind": kind, "product": product, "process": process, "resource": resource,
        "state" if kind == "ap_state" else "event": {
            "symbol": symbol, "arguments": {} if arguments is None else arguments,
        },
    })


def ap_condition(definition: dict[str, Any]) -> dict[str, Any]:
    """Return the validated event or state condition of an AP."""
    parsed = validate_ap_definition(definition)
    return parsed["state" if parsed["kind"] == "ap_state" else "event"]


def physical_ap_kind(ap: dict[str, Any] | str) -> str | None:
    """Identify observation predicates independently of AP labels."""
    definition = (
        parse_ap_record(ap)
        if isinstance(ap, dict) and ("full" in ap or "definition" in ap)
        else parse_ap_definition(ap)
    )
    condition = ap_condition(definition)
    signature = (definition["kind"], condition["symbol"], frozenset(condition["arguments"]))
    registry = {
        ("ap_state", "any", frozenset({"region"})): "resource_region",
        ("ap_event", "part_region_entry", frozenset({"region"})): "part_region_entry",
        ("ap_state", "processCompleted", frozenset({"target"})): "process_target_completed",
        ("ap_state", "processCompleted", frozenset({"result"})): "process_result_completed",
        ("ap_event", "receiving_region_entry", frozenset({"region"})): "receiving_region_entry",
        ("ap_state", "contains_other_part", frozenset()): "contains_other_part",
    }
    return registry.get(signature)



def physical_binding_fields(ap: dict[str, Any] | str) -> set[str]:
    """Return explicit scope fields required by a physical AP template."""
    definition = parse_ap_record(ap) if isinstance(ap, dict) and ("full" in ap or "definition" in ap) else parse_ap_definition(ap)
    if physical_ap_kind(definition) is None:
        return set()
    values = [definition[key] for key in ("product", "process", "resource")]
    values.extend(ap_condition(definition)["arguments"].values())
    return {
        "resources" if value in {"$first_resource", "$second_resource"} else value[1:]
        for value in values if isinstance(value, str) and value.startswith("$")
    }


def bind_ap_record(ap: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    """Ground declared scope references without changing literal identifiers."""
    definition = parse_ap_record(ap)
    def resolve(value: Any) -> Any:
        if not isinstance(value, str) or not value.startswith("$"):
            return value
        if value in {"$first_resource", "$second_resource"}:
            resources = binding.get("resources")
            if not isinstance(resources, list) or len(resources) != 2 or resources[0] == resources[1]:
                raise ValueError("A mutex template requires two distinct configured resources")
            return resources[0 if value == "$first_resource" else 1]
        key = value[1:]
        if key not in binding:
            raise ValueError(f"AP scope is missing {key!r}")
        return binding[key]
    for field in ("product", "process", "resource"):
        definition[field] = resolve(definition[field])
    condition = definition["state" if definition["kind"] == "ap_state" else "event"]
    condition["arguments"] = {key: resolve(value) for key, value in condition["arguments"].items()}
    return ap_record(ap["label"], definition, ap["meaning"],
                     **{key: deepcopy(value) for key, value in ap.items()
                        if key not in {"label", "full", "definition", "meaning"}})


def physical_ap_binding(ap: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    """Resolve a physical condition into the observation fields it requires."""
    bound = bind_ap_record(ap, binding)
    definition = bound["definition"]
    kind = physical_ap_kind(definition)
    result = deepcopy(binding)
    result.update(ap_condition(definition)["arguments"])
    if kind == "resource_region":
        result["resource"] = definition["resource"]
    elif kind in {"part_region_entry", "process_target_completed", "process_result_completed",
                  "receiving_region_entry", "contains_other_part"}:
        result["part"] = definition["product"]
        if kind.startswith("process_"):
            result["process"] = definition["process"]
        if kind == "receiving_region_entry":
            result["resource"] = definition["resource"]
        if kind == "contains_other_part":
            result["receiving_resource"] = definition["resource"]
    return result


def build_mutex_specification(
    identifier: str, requirement: str, template: dict[str, Any],
) -> dict[str, Any]:
    """Compile a wildcard resource-region schema into pairwise monitor operands.

    The grounding layer instantiates both slots over every unordered distinct
    pair from the complete configured scene. This helper selects no participants.
    """
    definition = validate_ap_definition(template)
    if physical_ap_kind(definition) != "resource_region" or definition["resource"] != "*":
        raise ValueError("Mutex authoring requires ap_state with resource '*' and any(region)")
    aps = []
    for label, role, position in (
        ("ap001", "$first_resource", "first"), ("ap002", "$second_resource", "second"),
    ):
        instance = deepcopy(definition)
        instance["resource"] = role
        meaning = (
            f"The {position} bound resource's configured robot/tool geometry, including any carried part, "
            "touches or overlaps the bound shared area. A deposited part does not retain the resource's occupancy."
        )
        aps.append(ap_record(label, instance, meaning))
    return {"id": identifier, "requirement": requirement, "formula": "G !(ap001 & ap002)", "aps": aps}

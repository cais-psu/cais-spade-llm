"""Check supplied primitive models offline; passing never authorizes execution."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
    validate_reviewed_primitive_program_safety as validate_reviewed_primitive_program_safety,
)
from cais_spade_llm.resources.primitive_observations import model_primitive_observations

_DEFINITIONS = (
    Path(__file__).resolve().parents[2]
    / "specification"
    / "safety"
    / "primitive_observation_safety.json"
)
_BINDING_FIELDS = {
    "receiving_region_entry": {
        "rule_id",
        "specification",
        "region",
        "resource",
        "receiving_resource",
        "part",
    },
    "shared_area_mutex": {"rule_id", "specification", "region", "resources"},
}
_SPECIFICATION_MEANINGS = {
    "receiving_region_entry": {
        "requirement": "An incoming part must not begin entering an applicable resource's receiving region while that resource contains another part.",
        "aps": {
            "ap_event/physical_observation/receiving_region_entry": "The bound resource's incoming tool/part begins occupying the bound receiving region while carrying the bound part. Boundary contact counts as occupancy; initial occupancy does not create an entry.",
            "ap_state/physical_observation/receiving_resource_contains_other_part": "The complete modeled inventory of the bound receiving_resource contains a part other than the bound incoming part.",
        },
    },
    "shared_area_mutex": {
        "requirement": "Resources declared mutually exclusive must never occupy the same configured shared area simultaneously.",
        "aps": {
            "ap_state/physical_observation/shared_area_first_resource": "The first bound resource's configured robot/tool geometry, including any carried part, touches or overlaps the bound shared area. A deposited part does not retain the resource's occupancy.",
            "ap_state/physical_observation/shared_area_second_resource": "The second bound resource's configured robot/tool geometry, including any carried part, touches or overlaps the bound shared area. A deposited part does not retain the resource's occupancy.",
        },
    },
}


def _definition_labels(specification: str, aps: list[dict[str, Any]]) -> tuple[str, str]:
    by_full = {ap["full"]: ap["label"] for ap in aps}
    first, second = _SPECIFICATION_MEANINGS[specification]["aps"]
    return by_full[first], by_full[second]


def _read_definitions() -> dict[str, dict[str, Any]]:
    document = json.loads(_DEFINITIONS.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or set(document) != {"version", "specifications"}
        or type(document["version"]) is not int
        or document["version"] != 1
        or not isinstance(document["specifications"], list)
    ):
        raise ValueError("Unsupported primitive observation safety definition document")
    definitions: dict[str, dict[str, Any]] = {}
    for row in document["specifications"]:
        if not isinstance(row, dict) or set(row) != {"id", "requirement", "formula", "aps"}:
            raise ValueError("Malformed primitive observation safety specification")
        specification = row["id"]
        if (
            not isinstance(specification, str)
            or specification not in _SPECIFICATION_MEANINGS
            or specification in definitions
        ):
            raise ValueError("Unknown or duplicate primitive observation safety specification")
        expected = _SPECIFICATION_MEANINGS[specification]
        if row["requirement"] != expected["requirement"] or not isinstance(row["aps"], list):
            raise ValueError("Unsupported primitive observation safety requirement or APs")
        labels: set[str] = set()
        fulls: set[str] = set()
        for ap in row["aps"]:
            if not isinstance(ap, dict) or set(ap) != {"label", "full", "meaning"}:
                raise ValueError("Malformed primitive observation safety AP")
            label, full = ap["label"], ap["full"]
            if (
                not isinstance(label, str)
                or re.fullmatch(r"ap[0-9]+", label) is None
                or label in labels
                or not isinstance(full, str)
                or full not in expected["aps"]
                or full in fulls
                or ap["meaning"] != expected["aps"][full]
            ):
                raise ValueError(
                    "Unsupported or duplicated primitive observation AP meaning or label"
                )
            labels.add(label)
            fulls.add(full)
        if fulls != set(expected["aps"]):
            raise ValueError("Primitive observation safety specification is missing AP definitions")
        first, second = _definition_labels(specification, row["aps"])
        if row["formula"] != f"G !({first} & {second})":
            raise ValueError(
                "Primitive observation safety formula does not match its fixed invariant"
            )
        definitions[specification] = row
    if set(definitions) != set(_SPECIFICATION_MEANINGS):
        raise ValueError("Primitive observation safety definitions are incomplete")
    return definitions


def _dfa_dot(first: str, second: str) -> str:
    return f"""digraph DFA {{
    init [shape=point];
    1 [shape=doublecircle];
    2 [shape=circle];
    init -> 1;
    1 -> 1 [label="!{first} | !{second}"];
    1 -> 2 [label="{first} & {second}"];
    2 -> 2 [label="true"];
}}"""


def instantiate_primitive_safety_rules(bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bind the two fixed specifications without changing AP identifiers.

    Args:
        bindings: Exact rule identifiers and resource/region/part bindings.

    Returns:
        Independent definitions ordered by the exact rule identifiers.

    Raises:
        ValueError: Bindings are missing, malformed, duplicated, or unsupported.
        OSError: The checked-in specification definitions cannot be read.
    """
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("Explicit nonempty safety bindings are required")
    definitions = _read_definitions()
    rules = []
    rule_ids: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("Every safety binding must be a dictionary")
        specification = binding.get("specification")
        if not isinstance(specification, str) or specification not in _BINDING_FIELDS:
            raise ValueError("Safety binding has an unsupported specification")
        if set(binding) != _BINDING_FIELDS[specification]:
            raise ValueError(f"Safety binding fields do not match {specification!r}")
        for field in _BINDING_FIELDS[specification] - {"resources"}:
            if not isinstance(binding[field], str) or not binding[field]:
                raise ValueError(f"Safety binding {field!r} must be an exact nonempty identifier")
        rule_id = binding["rule_id"]
        if rule_id in rule_ids:
            raise ValueError(f"Duplicate safety rule_id: {rule_id!r}")
        rule_ids.add(rule_id)
        if specification == "shared_area_mutex":
            resources = binding["resources"]
            if (
                not isinstance(resources, list)
                or len(resources) != 2
                or any(not isinstance(resource, str) or not resource for resource in resources)
                or resources[0] == resources[1]
            ):
                raise ValueError("shared_area_mutex requires two distinct exact resources")
        rule = deepcopy(definitions[specification])
        labels = _definition_labels(specification, rule["aps"])
        rule.update(
            {
                "id": rule_id,
                "rule_id": rule_id,
                "binding": deepcopy(binding),
                "dfa_dot": _dfa_dot(*labels),
            }
        )
        rules.append(rule)
    return sorted(rules, key=lambda rule: rule["rule_id"])


def _occupies(observation: dict[str, Any], region: str, resource: str) -> bool:
    value = observation["region_occupancy"][region][resource]
    if type(value) is not bool:
        raise ValueError("Region occupancy must be an observed Boolean")
    return value


def _valuation(
    rule: dict[str, Any], observation: dict[str, Any], previous: dict[str, Any] | None
) -> dict[str, bool]:
    binding = rule["binding"]
    region = binding["region"]
    first_label, second_label = _definition_labels(binding["specification"], rule["aps"])
    if binding["specification"] == "shared_area_mutex":
        first, second = binding["resources"]
        return {
            first_label: _occupies(observation, region, first),
            second_label: _occupies(observation, region, second),
        }
    resource = binding["resource"]
    inside = _occupies(observation, region, resource)
    previously_inside = inside if previous is None else _occupies(previous, region, resource)
    carried_parts = observation["carried_parts"][resource]
    inventory = observation["resources"][binding["receiving_resource"]]["contained_parts"]
    for parts in (carried_parts, inventory):
        if not isinstance(parts, list) or any(
            not isinstance(part, str) or not part for part in parts
        ):
            raise ValueError(
                "Carried parts and complete containment evidence must list exact part identifiers"
            )
    return {
        first_label: inside and not previously_inside and binding["part"] in carried_parts,
        second_label: any(part != binding["part"] for part in inventory),
    }


def _result(
    *,
    status: str,
    reason: str,
    states_before: dict[str, str],
    states_after: dict[str, str],
    rules: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    model: dict[str, Any],
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    is_safe = status == "safe"
    code = "safety_rule_violation" if status == "violated" else "safety_validation_unavailable"
    findings = []
    if not is_safe:
        findings.append(
            {
                "constraint_owner": "cca",
                "constraint_family": "safety",
                "constraint_code": code,
                "failed_axes": [code],
                "rule_id": (failure or {}).get("rule_id"),
                "status": status,
                "reason": reason,
                "evidence": deepcopy(failure or model.get("evidence") or {}),
            }
        )
    return {
        "is_safe": is_safe,
        "feasibility_status": "FEASIBLE"
        if is_safe
        else "INFEASIBLE"
        if status == "violated"
        else "NEEDS_CONTEXT",
        "safety_ctx": {
            "status": status,
            "reason": reason,
            "offline_only": True,
            "rule_ids": [rule["rule_id"] for rule in rules],
            "safety_rules": deepcopy(rules),
            "rule_checks": deepcopy(checks),
        },
        "findings": findings,
        "safety_dfa_states_before": deepcopy(states_before),
        "safety_dfa_states_after": deepcopy(states_after if is_safe else states_before),
        "observations": deepcopy(model.get("observations", [])),
        "projected_snapshot": deepcopy(model.get("projected_snapshot")),
        "evidence": deepcopy(model.get("evidence", {})),
    }


def validate_primitive_program_safety(
    *,
    programs: list[dict[str, Any]],
    snapshot: dict[str, Any],
    geometry: dict[str, Any],
    horizon: list[float],
    stationary: dict[str, Any],
    bindings: list[dict[str, Any]],
    safety_dfa_states_before: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate joint modeled primitive consequences against fixed safety APs.

    Args:
        programs: Bound primitive programs with resource-supplied step results.
        snapshot: Joint initial resources, custody, and containment evidence.
        geometry: Configured regions and projected robot/tool/part geometry.
        horizon: Start and end time of the proposed joint model.
        stationary: Explicit stationary coverage for relevant resources.
        bindings: Concrete receiving-region and shared-area rule instances.
        safety_dfa_states_before: Exact monitor state for every binding, or None
            to initialize a fresh offline monitor.

    Returns:
        Safety findings, per-rule valuations/transitions, observations, and a
        projected snapshot. Rejected or unresolved checks preserve input DFA
        states. A projected snapshot describes the proposal, never execution.
    """
    before = deepcopy(safety_dfa_states_before) if safety_dfa_states_before is not None else {}
    rules: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    model: dict[str, Any] = {}
    try:
        rules = instantiate_primitive_safety_rules(bindings)
        checker = BaseSafetyChecker({rule["rule_id"]: rule["dfa_dot"] for rule in rules}, rules)
        if safety_dfa_states_before is None:
            before = {rule_id: dfa["initial"] for rule_id, dfa in checker.dfas.items()}
        elif not isinstance(before, dict) or set(before) != set(checker.dfas):
            raise ValueError("Initial DFA states must match the exact safety rule_ids")
        for rule_id, state in before.items():
            if not isinstance(state, str) or state not in checker.dfas[rule_id]["transitions"]:
                raise ValueError(f"Unknown initial DFA state for {rule_id!r}")
        model = model_primitive_observations(
            programs=programs,
            snapshot=snapshot,
            geometry=geometry,
            horizon=horizon,
            stationary=stationary,
            bindings=bindings,
        )
        if model.get("valid") is not True:
            return _result(
                status="unavailable",
                reason=model.get("reason", "Primitive observations are unavailable"),
                states_before=before,
                states_after=before,
                rules=rules,
                checks=checks,
                model=model,
            )
        observations = model["observations"]
        if not isinstance(observations, list) or not observations:
            raise ValueError("A complete nonempty observation trace is required")
        after = deepcopy(before)
        previous = None
        for observation in observations:
            for rule in rules:
                rule_id = rule["rule_id"]
                values = _valuation(rule, observation, previous)
                transition = checker.transition_evidence(
                    rule_id,
                    after[rule_id],
                    frozenset(label for label, active in values.items() if active),
                )
                check = {
                    "rule_id": rule_id,
                    "time": observation["time"],
                    "phase": observation["phase"],
                    "ap_values": values,
                    "aps": deepcopy(rule["aps"]),
                    "transition": transition,
                    "active_steps": deepcopy(observation["active_steps"]),
                    "resources": deepcopy(observation["resources"]),
                    "parts": deepcopy(observation["parts"]),
                }
                checks.append(check)
                if transition["status"] != "passed":
                    violated = transition["reason"] == "accepting_state_unreachable"
                    return _result(
                        status="violated" if violated else "unavailable",
                        reason=rule["requirement"] if violated else transition["reason"],
                        states_before=before,
                        states_after=before,
                        rules=rules,
                        checks=checks,
                        model=model,
                        failure=check,
                    )
                after[rule_id] = transition["to"]
            previous = observation
        return _result(
            status="safe",
            reason="",
            states_before=before,
            states_after=after,
            rules=rules,
            checks=checks,
            model=model,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _result(
            status="unavailable",
            reason=str(exc),
            states_before=before,
            states_after=before,
            rules=rules,
            checks=checks,
            model=model,
        )

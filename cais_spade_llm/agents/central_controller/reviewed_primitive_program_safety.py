"""Check reviewed LTLf requirements against a frozen, resource-owned trace.

This module has no execution authority. Acceptance is conditional on the
supplied physical model and declared applicability population. DFA reachability
for one rule does not establish a physically possible joint continuation.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from copy import deepcopy
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
from cais_spade_llm.resources.primitive_observations import model_primitive_observations

from cais_spade_llm.agents.central_controller.ppr_ap import (
    ap_record, bind_ap_record, canonical_ap_key, make_ap_definition,
    parse_ap_record, physical_ap_binding, physical_ap_kind, physical_binding_fields,
)

_CLOCK_VERSION = "primitive_observations_joint_trace_v1"
_FIRST = canonical_ap_key(make_ap_definition("ap_state", "*", "*", "$first_resource", "any", {"region": "$region"}))
_SECOND = canonical_ap_key(make_ap_definition("ap_state", "*", "*", "$second_resource", "any", {"region": "$region"}))
_ENTRY = canonical_ap_key(make_ap_definition("ap_event", "$part", "*", "*", "part_region_entry", {"region": "$region"}))
_COMPLETED = canonical_ap_key(make_ap_definition("ap_state", "$part", "$process", "*", "processCompleted", {"result": "$result"}))
_TARGET_COMPLETED = canonical_ap_key(make_ap_definition("ap_state", "$part", "$process", "*", "processCompleted", {"target": "$target"}))
_RECEIVING_ENTRY = canonical_ap_key(make_ap_definition("ap_event", "$part", "*", "$resource", "receiving_region_entry", {"region": "$region"}))
_CONTAINS_OTHER = canonical_ap_key(make_ap_definition("ap_state", "$part", "*", "$receiving_resource", "contains_other_part"))
_MEANINGS = {
    _FIRST: "The first bound resource's configured robot/tool geometry, including any carried part, touches or overlaps the bound shared area. A deposited part does not retain the resource's occupancy.",
    _SECOND: "The second bound resource's configured robot/tool geometry, including any carried part, touches or overlaps the bound shared area. A deposited part does not retain the resource's occupancy.",
    _ENTRY: "The bound part begins touching or overlapping the bound region. Initial occupancy is not an entry; custody changes alone do not create entry.",
    _COMPLETED: "The bound part's explicitly complete processCompleted ledger contains the exact bound process and result record.",
    _TARGET_COMPLETED: "The bound part's explicitly complete processCompleted ledger contains the exact bound process and target record.",
    _RECEIVING_ENTRY: "The bound resource's incoming tool/part begins occupying the bound receiving region while carrying the bound part. Boundary contact counts as occupancy; initial occupancy does not create an entry.",
    _CONTAINS_OTHER: "The complete modeled inventory of the bound receiving_resource contains a part other than the bound incoming part.",
}
_PHYSICAL_MEANINGS = {
    physical_ap_kind(full): meaning for full, meaning in _MEANINGS.items()
}


def _ground_rule_aps(rule: dict[str, Any]) -> dict[str, Any]:
    """Replace template references with exact PPR identities for one rule."""
    rule = deepcopy(rule)
    rule["aps"] = [
        bind_ap_record(ap, _ap_binding(rule, ap)) if physical_ap_kind(ap) else deepcopy(ap)
        for ap in rule["aps"]
    ]
    return rule



def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _symbol(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} requires an exact nonempty identifier")
    return value


def _symbols(value: Any, name: str) -> list[str]:
    symbols = [_symbol(item, name) for item in _list(value, name)]
    if len(set(symbols)) != len(symbols):
        raise ValueError(f"{name} contains duplicate identifiers")
    return symbols


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compile_formula(formula: str, labels: set[str]) -> str:
    """Compile with isolated MONA input instead of ltlf2dfa's package file."""
    # Check support even when an identical formula has previously been compiled.
    from ltlf2dfa.parser.ltlf import LTLfParser

    executable = shutil.which("mona")
    if executable is None or LTLfParser is None:
        raise ValueError("MONA or LTLfParser is unavailable; reviewed formulas cannot be compiled")
    return _compile_formula_cached(formula, frozenset(labels), executable)


@lru_cache(maxsize=128)
def _compile_formula_cached(formula: str, labels: frozenset[str], executable: str) -> str:
    from lark.exceptions import LarkError
    from ltlf2dfa.base import MonaProgram
    from ltlf2dfa.ltlf2dfa import output2dot
    from ltlf2dfa.parser.ltlf import LTLfParser

    try:
        parsed = LTLfParser()(formula)
    except (LarkError, ValueError) as exc:
        raise ValueError(f"Invalid reviewed LTLf formula: {exc}") from exc
    if not set(parsed.find_labels()) <= labels:
        raise ValueError("Reviewed formula refers to an undefined AP label")
    with TemporaryDirectory(prefix="reviewed-primitive-safety-") as directory:
        path = Path(directory) / "formula.mona"
        path.write_text(MonaProgram(parsed).mona_program(), encoding="utf-8")
        try:
            compiled = subprocess.run(
                [executable, "-q", "-u", "-w", str(path)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("MONA formula compilation exceeded 30 seconds") from exc
    output = compiled.stdout
    if compiled.returncode != 0 or compiled.stderr.strip():
        raise ValueError("MONA formula compilation failed: " + compiled.stderr.strip())
    required = (
        r"^DFA for formula with free variables:.*$",
        r"^Initial state:\s*0\s*$",
        r"^Accepting states:.*$",
        r"^Automaton has \d+ states? and \d+ BDD-nodes?\s*$",
        r"^State \d+: [01X]* -> state \d+\s*$",
    )
    if not all(re.search(pattern, output, re.MULTILINE) for pattern in required):
        raise ValueError("MONA returned incomplete or unsupported DFA evidence")
    # ltlf2dfa consumes MONA's structural initial position when constructing the
    # DFA. Check that convention instead of silently accepting arbitrary output.
    if "Formula is unsatisfiable" not in output:
        initial = re.findall(r"^State 0: ([01X]*) -> state (\d+)\s*$", output, re.MULTILINE)
        if len(initial) != 1 or set(initial[0][0]) - {"X"} or initial[0][1] != "1":
            raise ValueError("Unsupported MONA initial-position convention")
    dot = output2dot(output)
    checker = BaseSafetyChecker({"compiled": dot}, [])
    dfa = checker.dfas["compiled"]
    if not dfa["transitions"] or dfa["initial"] not in dfa["transitions"]:
        raise ValueError("MONA produced no usable initial DFA transition")
    states = set(dfa["transitions"])
    if not set(dfa["accepting_states"]) <= states or any(
        target not in states for edges in dfa["transitions"].values() for _, target in edges
    ):
        raise ValueError("MONA DFA references an undefined state")
    if not set(dfa["ap_symbols"]) <= labels:
        raise ValueError("MONA DFA contains an undefined AP label")
    return dot


def _catalog(catalog: Any, *, structured_groundings: bool = False) -> dict[str, dict[str, Any]]:
    document = _object(catalog, "catalog")
    if (
        set(document) != {"version", "specifications"}
        or type(document["version"]) is not int
        or document["version"] != 2
    ):
        raise ValueError("Unsupported reviewed catalog version or fields; recompile required")
    definitions = {}
    for raw in _list(document["specifications"], "catalog.specifications"):
        row = _object(raw, "specification")
        if set(row) != {"id", "requirement", "formula", "aps"}:
            raise ValueError("Reviewed specifications require id, requirement, formula, and aps")
        identifier = _symbol(row["id"], "specification.id")
        if identifier in definitions:
            raise ValueError("Duplicate reviewed specification identifier")
        _symbol(row["requirement"], "specification.requirement")
        _symbol(row["formula"], "specification.formula")
        labels = set()
        fulls = set()
        for ap in _list(row["aps"], "specification.aps"):
            ap = _object(ap, "AP")
            if set(ap) not in ({"label", "full", "meaning"}, {"label", "full", "definition", "meaning"}):
                raise ValueError("APs require exact label, full, and meaning fields")
            label = _symbol(ap["label"], "AP.label")
            full = _symbol(ap["full"], "AP.full")
            if re.fullmatch(r"ap[0-9]+", label) is None or label in labels or full in fulls:
                raise ValueError("Unsupported or duplicate reviewed AP identifier")
            definition = parse_ap_record(ap)
            kind = physical_ap_kind(definition)
            if kind is not None:
                expected = _MEANINGS.get(full, _PHYSICAL_MEANINGS[kind])
                if kind == "resource_region" and definition["resource"] in {"$first_resource", "$second_resource"}:
                    expected = _MEANINGS[_FIRST if definition["resource"] == "$first_resource" else _SECOND]
                if ap["meaning"] != expected:
                    raise ValueError("Unsupported predicate or changed fixed AP meaning")
            elif not structured_groundings:
                raise ValueError("Unsupported typed predicate: complete structured grounding is required")
            else:
                _symbol(ap["meaning"], "structured AP meaning")
            ap["definition"] = definition
            labels.add(label)
            fulls.add(full)
        if not labels:
            raise ValueError("A reviewed specification must declare supported APs")
        definitions[identifier] = deepcopy(row)
        definitions[identifier]["dfa_dot"] = _compile_formula(row["formula"], labels)
    if not definitions:
        raise ValueError("The complete reviewed catalog must be nonempty")
    return definitions


def _excluded_resources(population: dict[str, Any]) -> set[str]:
    excluded = set()
    for row in _list(population["excluded_resources"], "excluded_resources"):
        row = _object(row, "excluded resource")
        if set(row) != {"resource_id", "reason"}:
            raise ValueError("Every excluded resource requires identity and evidence assumptions")
        identifier = _symbol(row["resource_id"], "excluded resource_id")
        _symbol(row["reason"], "excluded resource reason")
        if identifier in excluded:
            raise ValueError("Duplicate excluded resource")
        excluded.add(identifier)
    return excluded


def _population(scope: dict[str, Any], geometry: dict[str, Any]) -> tuple[set[str], dict[str, Any]]:
    resources = {}
    for row in _list(scope["scene_resources"], "scene_resources"):
        row = _object(row, "scene resource")
        if set(row) != {"resource_id", "resource_type"}:
            raise ValueError("Each scene resource requires exact identity and type")
        identifier = _symbol(row["resource_id"], "resource_id")
        if identifier in resources:
            raise ValueError("Duplicate scene resource")
        resources[identifier] = _symbol(row["resource_type"], "resource_type")
    types = set(_symbols(scope["participant_resource_types"], "participant_resource_types"))
    if not resources or not types or not types <= set(resources.values()):
        raise ValueError("Applicability needs a populated, declared resource population")
    participants = {identifier for identifier, kind in resources.items() if kind in types}
    if not set(_object(geometry.get("resources"), "geometry.resources")) <= set(resources):
        raise ValueError("Physical geometry contains a resource omitted from the scene population")
    regions = _object(scope["regions"], "applicability.regions")
    if not regions:
        raise ValueError("Applicability requires an explicit region population")
    coverage = {"population": deepcopy(scope["scene_resources"]), "regions": {}, "rule_ids": []}
    for region, population in regions.items():
        population = _object(population, "region population")
        if set(population) != {"resources", "excluded_resources"}:
            raise ValueError("Region population requires included and excluded resources")
        if region not in _object(geometry.get("regions"), "geometry.regions"):
            raise ValueError("Applicable region has no configured geometry")
        included = set(_symbols(population["resources"], "region.resources"))
        if included != participants:
            raise ValueError(
                "Region participant coverage differs from the declared scene population"
            )
        if not included <= set(_object(geometry.get("resources"), "geometry.resources")):
            raise ValueError("An applicable participant has no physical geometry")
        excluded = _excluded_resources(population)
        if excluded != set(resources) - included:
            raise ValueError("Resource exclusion coverage is incomplete or contradictory")
        coverage["regions"][region] = deepcopy(population)
    return participants, coverage


def _scope_geometry(binding: dict, geometry: dict, regions: Any, resources: Any) -> None:
    if "region" in binding and binding["region"] not in regions:
        raise ValueError("Scoped region has no configured geometry")
    if "part" in binding and binding["part"] not in _object(geometry.get("parts"), "geometry.parts"):
        raise ValueError("Scoped part has no configured geometry")
    if "target" in binding and _object(geometry["parts"][binding["part"]], "part geometry").get("target") != binding["target"]:
        raise ValueError("Scoped assembly target differs from the part's configured geometry")
    for field in ("resource", "receiving_resource"):
        if field in binding and binding[field] not in resources:
            raise ValueError("Scoped resource is absent from the frozen scene")


def _reviewed_physical_bindings(definition: dict, raw: dict, geometry: dict, scope: dict) -> dict | None:
    if "physical_ap_bindings" not in raw:
        return None
    bindings = _physical_bindings(definition, raw["physical_ap_bindings"])
    if set(raw) != {"specification", "physical_ap_bindings"}:
        raise ValueError("Per-AP applicability cannot also supply shared binding fields")
    resources = {row["resource_id"] for row in scope["scene_resources"]}
    for binding in bindings.values():
        _scope_geometry(binding, geometry, scope["regions"], resources)
    return bindings


def _rules(  # noqa: C901
    definitions: dict[str, dict[str, Any]], applicability: Any, geometry: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scope = _object(applicability, "applicability")
    if (
        set(scope)
        != {"version", "scene_resources", "participant_resource_types", "regions", "rules"}
        or type(scope["version"]) is not int
        or scope["version"] != 1
    ):
        raise ValueError("Unsupported applicability fields or version")
    participants, coverage = _population(scope, geometry)
    regions = scope["regions"]
    used = set()
    rules = []
    identifiers = set()
    expansions = set()
    for raw in _list(scope["rules"], "applicability.rules"):
        raw = _object(raw, "applicable rule")
        specification = _symbol(raw.get("specification"), "rule.specification")
        if specification not in definitions:
            raise ValueError("Applicable requirement is absent from the reviewed catalog")
        definition = definitions[specification]
        required = set().union(*(physical_binding_fields(ap) for ap in definition["aps"]))
        physical_bindings = _reviewed_physical_bindings(definition, raw, geometry, scope)
        if physical_bindings is not None:
            required = {"physical_ap_bindings"}
        if "region" in required and raw.get("region") not in regions:
            raise ValueError("Applicable rule has no declared region population")
        expanded = []
        if raw.get("bindings") == "all_pairs":
            if required != {"region", "resources"} or set(raw) != {
                "specification",
                "region",
                "bindings",
            }:
                raise ValueError("all_pairs requires region/resource AP bindings")
            expansion_key = (specification, raw["region"])
            expansions.add(expansion_key)
            pairs = list(combinations(sorted(participants), 2))
            if not pairs:
                raise ValueError("Pair applicability requires at least two participants")
            expanded = [
                {"specification": specification, "region": raw["region"], "resources": list(pair)}
                for pair in pairs
            ]
        else:
            if "resources" in required:
                raise ValueError(
                    "Resource-pair requirements must declare complete all_pairs coverage"
                )
            if set(raw) != required | {"specification"}:
                raise ValueError("Applicability fields do not match the declared AP bindings")
            for field in required - {"physical_ap_bindings"}:
                _symbol(raw[field], f"binding.{field}")
            expanded = [deepcopy(raw)]
        used.add(specification)
        for binding in expanded:
            # JSON quoting preserves exact symbols and prevents delimiter aliases.
            identifier = json.dumps(binding, sort_keys=True, separators=(",", ":"))
            if identifier in identifiers:
                raise ValueError("Duplicate applicable rule instance")
            identifiers.add(identifier)
            rules.append(_ground_rule_aps({**deepcopy(definition), "rule_id": identifier, "binding": binding}))
    if used != set(definitions):
        raise ValueError("Reviewed catalog contains an applicable requirement without bindings")
    required_pairs = {
        (identifier, region)
        for identifier, definition in definitions.items()
        if any("resources" in physical_binding_fields(ap) for ap in definition["aps"])
        for region in regions
    }
    if expansions != required_pairs:
        raise ValueError("Resource-pair applicability must cover every declared region")
    rules.sort(key=lambda rule: rule["rule_id"])
    coverage["rule_ids"] = [rule["rule_id"] for rule in rules]
    return rules, coverage


def _boolean(value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError("Every physical AP requires explicit Boolean observation evidence")
    return value


def _physical_bindings(definition: dict[str, Any], raw: Any) -> dict[str, Any]:
    """Validate explicit per-AP physical bindings without selecting participants."""
    bindings = _object(raw, "physical_ap_bindings")
    physical = {ap["label"]: ap for ap in definition["aps"] if physical_ap_kind(ap) is not None}
    if not physical or set(bindings) != set(physical):
        raise ValueError("Every physical AP needs exactly one explicit binding")
    for label, ap in physical.items():
        fields = physical_binding_fields(ap)
        binding = _object(bindings[label], "physical AP binding")
        if "resources" in fields or set(binding) != fields:
            raise ValueError("Physical AP bindings require exact fields and no caller-selected pairs")
        for field in fields:
            _symbol(binding[field], f"physical AP binding.{field}")
    return deepcopy(bindings)


def _ap_binding(rule: dict, ap: dict) -> dict:
    bindings = rule.get("physical_ap_bindings", rule["binding"].get("physical_ap_bindings", {}))
    binding = bindings.get(ap["label"], rule["binding"])
    return physical_ap_binding(ap, binding) if physical_ap_kind(ap) else binding


def _complete_target_ledger(part: dict[str, Any]) -> list[dict[str, Any]]:
    provenance = _object(part.get("processCompleted_evidence"), "processCompleted_evidence")
    if part.get("processCompleted_complete") is not True or provenance.get("complete") is not True:
        raise ValueError("processCompleted requires explicitly complete checkpoint evidence")
    for field in ("source_kind", "checkpoint"):
        _symbol(provenance.get(field), f"processCompleted_evidence.{field}")
    ledger = _list(part.get("processCompleted"), "processCompleted")
    for record in ledger:
        record = _object(record, "processCompleted record")
        if set(record) not in ({"process"}, {"process", "result"}, {"process", "target"}):
            raise ValueError("Unsupported processCompleted record fields")
        for field, value in record.items():
            _symbol(value, f"processCompleted.{field}")
    return ledger


def _valuation(
    rule: dict[str, Any], observation: dict[str, Any], previous: dict[str, Any] | None
) -> dict[str, bool]:
    values = {}
    for ap in rule["aps"]:
        binding = _ap_binding(rule, ap)
        kind = physical_ap_kind(ap)
        if kind == "resource_region":
            resource = binding["resource"]
            value = _boolean(observation["region_occupancy"][binding["region"]][resource])
        elif kind == "part_region_entry":
            inside = _boolean(
                observation["part_region_occupancy"][binding["region"]][binding["part"]]
            )
            before = (
                inside
                if previous is None
                else _boolean(previous["part_region_occupancy"][binding["region"]][binding["part"]])
            )
            value = inside and not before
        elif kind == "process_target_completed":
            ledger = _complete_target_ledger(observation["parts"][binding["part"]])
            value = {"process": binding["process"], "target": binding["target"]} in ledger
        elif kind == "process_result_completed":
            part = observation["parts"][binding["part"]]
            provenance = _object(part.get("processCompleted_evidence"), "processCompleted_evidence")
            if (
                part.get("processCompleted_complete") is not True
                or provenance.get("complete") is not True
            ):
                raise ValueError(
                    "processCompleted requires explicitly complete checkpoint evidence"
                )
            _symbol(provenance.get("source_kind"), "processCompleted_evidence.source_kind")
            _symbol(provenance.get("checkpoint"), "processCompleted_evidence.checkpoint")
            ledger = _list(part.get("processCompleted"), "processCompleted")
            for item in ledger:
                record = _object(item, "processCompleted record")
                _symbol(record.get("process"), "processCompleted.process")
                _symbol(record.get("result"), "processCompleted.result")
            value = {"process": binding["process"], "result": binding["result"]} in ledger
        elif kind == "receiving_region_entry":
            inside = _boolean(
                observation["region_occupancy"][binding["region"]][binding["resource"]]
            )
            before = (
                inside
                if previous is None
                else _boolean(previous["region_occupancy"][binding["region"]][binding["resource"]])
            )
            carried = _symbols(observation["carried_parts"][binding["resource"]], "carried_parts")
            value = inside and not before and binding["part"] in carried
        elif kind == "contains_other_part":
            inventory = _symbols(
                observation["resources"][binding["receiving_resource"]]["contained_parts"],
                "contained_parts",
            )
            value = any(part != binding["part"] for part in inventory)
        else:
            raise ValueError("Unsupported physical AP condition")
        values[ap["label"]] = value
    return values


def _restore_continuation(
    checker: BaseSafetyChecker,
    rules: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    continuation: dict[str, Any] | None,
    trace_id: str,
    clock_version: str = _CLOCK_VERSION,
    valuations: list[dict[str, dict[str, bool]]] | None = None,
) -> tuple[dict[str, str], int]:
    initial = {identifier: dfa["initial"] for identifier, dfa in checker.dfas.items()}
    before = deepcopy(initial)
    offset = 0
    if continuation is not None:
        continuation = _object(continuation, "continuation")
        required = {
            "clock_version",
            "trace_id",
            "next_observation",
            "states",
            "previous_observation",
            "trace_complete",
        }
        if (
            set(continuation) != required
            or continuation["clock_version"] != clock_version
            or continuation["trace_id"] != trace_id
        ):
            raise ValueError("Continuation belongs to a different frozen trace or clock")
        offset = continuation["next_observation"]
        if (
            type(offset) is not int
            or not 0 < offset <= len(observations)
            or continuation["trace_complete"] is not False
        ):
            raise ValueError("Continuation has an invalid offset or already completed trace")
        if continuation["previous_observation"] != observations[offset - 1]:
            raise ValueError("Continuation preceding observation differs from the frozen trace")
        before = deepcopy(_object(continuation["states"], "continuation.states"))
        # Replaying the accepted prefix prevents a caller from replacing a
        # monitor state with an unrelated but syntactically valid state.
        replay = deepcopy(initial)
        for index, observation in enumerate(observations[:offset]):
            previous = observations[index - 1] if index else None
            for rule in rules:
                identifier = rule["rule_id"]
                values = (
                    valuations[index][identifier]
                    if valuations is not None
                    else _valuation(rule, observation, previous)
                )
                transition = checker.transition_evidence(
                    identifier,
                    replay[identifier],
                    frozenset(label for label, active in values.items() if active),
                )
                if transition["status"] != "passed":
                    raise ValueError("Continuation prefix is not an accepted monitor history")
                replay[identifier] = transition["to"]
        if before != replay:
            raise ValueError("Continuation monitor states do not match the accepted prefix")
    return before, offset


def _slice_end(
    offset: int, length: int, observation_slice: list[int] | None, trace_complete: bool
) -> int:
    end = length
    if observation_slice is not None:
        selection = _list(observation_slice, "observation_slice")
        if len(selection) != 2 or any(type(index) is not int for index in selection):
            raise ValueError("observation_slice must be two integer indices")
        start, end = selection
        if start != offset:
            raise ValueError(
                "Trace slices must be contiguous without gaps or duplicate observations"
            )
    if not offset <= end <= length or (end == offset and not (trace_complete and offset == length)):
        raise ValueError("Invalid or empty observation slice")
    if trace_complete and end != length:
        raise ValueError("trace_complete requires the actual frozen trace end")
    return end


def _new_result(
    continuation: dict[str, Any] | None, clock_version: str = _CLOCK_VERSION
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "unavailable",
        "is_safe": False,
        "reason": "",
        "reason_code": "NEEDS_CONTEXT",
        "feasibility_status": "NEEDS_CONTEXT",
        "pending_rule_ids": [],
        "observations": [],
        "bindings": [],
        "coverage": {},
        "rule_checks": [],
        "counterexample": None,
        "projected_snapshot": None,
        "safety_dfa_states_before": deepcopy(continuation.get("states", {}))
        if isinstance(continuation, dict)
        else {},
        "safety_dfa_states_after": deepcopy(continuation.get("states", {}))
        if isinstance(continuation, dict)
        else {},
        "continuation": deepcopy(continuation),
        "candidate_continuation": None,
        "trace_id": None,
        "trace_length": 0,
        "clock_version": clock_version,
        "observation_slice": None,
    }
    return result


def _validate_immutable_result_ledgers(rules: list, observations: list, snapshot: dict) -> None:
    for rule in rules:
        for ap in rule["aps"]:
            if ap["full"] != _COMPLETED:
                continue
            part = _ap_binding(rule, ap)["part"]
            for observation in observations:
                expected = deepcopy(snapshot["parts"][part].get("processCompleted"))
                previous = snapshot["parts"][part].get("product_effect_evidence", [])
                for update in observation["parts"][part].get("product_effect_evidence", []):
                    if update not in previous:
                        if update.get("kind") not in {"acknowledged", "predicted"} or not isinstance(expected, list):
                            raise ValueError("Completion ledger lacks validated task effects")
                        expected.extend(deepcopy(update["product_effects"].get(part, {}).get("processCompleted", [])))
                for field in (
                    "processCompleted", "processCompleted_complete", "processCompleted_evidence",
                ):
                    initial = expected if field == "processCompleted" else snapshot["parts"][part].get(field)
                    if observation["parts"][part].get(field) != initial:
                        raise ValueError("A modeled primitive changed checkpoint processCompleted evidence")


def _evaluate_frozen_trace(  # noqa: PLR0913
    *,
    result: dict[str, Any],
    model: dict[str, Any],
    snapshot: dict[str, Any],
    rules: list[dict[str, Any]],
    fingerprint: dict[str, Any],
    continuation: dict[str, Any] | None,
    trace_complete: bool,
    observation_slice: list[int] | None,
    clock_version: str = _CLOCK_VERSION,
    valuations: list[dict[str, dict[str, bool]]] | None = None,
    ap_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if model.get("valid") is not True:
        raise ValueError(model.get("reason", "Physical observations are unavailable"))
    if model.get('evidence', {}).get('continuous_motion'):
        raise ValueError('Continuous occupancy alternatives require local composition')
    observations = _list(model.get("observations"), "observations")
    if not observations:
        raise ValueError("The frozen trace must include an initial observation")
    result["evidence"] = deepcopy(model.get("evidence", {}))
    result["trace_length"] = len(observations)
    fingerprint = {
        **fingerprint,
        "clock_version": clock_version,
        "rules": rules,
        "observations": observations,
        **({"primitive_models": model["evidence"]["primitive_models"],
            "owner_effects": model["evidence"].get("owner_effects", [])}
           if model.get("evidence", {}).get("primitive_models") else {}),
    }
    result["trace_id"] = _digest(fingerprint)
    checker = BaseSafetyChecker({rule["rule_id"]: rule["dfa_dot"] for rule in rules}, rules)
    before, offset = _restore_continuation(
        checker, rules, observations, continuation, result["trace_id"], clock_version, valuations
    )
    result["safety_dfa_states_before"] = deepcopy(before)
    result["safety_dfa_states_after"] = deepcopy(before)
    end = _slice_end(offset, len(observations), observation_slice, trace_complete)
    result["observation_slice"] = [offset, end]
    result["observations"] = deepcopy(observations[offset:end])
    if ap_evidence is not None:
        result["ap_evidence"] = deepcopy(
            [row for row in ap_evidence if offset <= row["observation_index"] < end]
        )
    # Ground every required AP before deciding any formula. Missing physical
    # evidence never becomes a false proposition or an ignored requirement.
    valuations = (
        valuations
        if valuations is not None
        else [
            {
                rule["rule_id"]: _valuation(
                    rule, observation, observations[index - 1] if index else None
                )
                for rule in rules
            }
            for index, observation in enumerate(observations)
        ]
    )
    _validate_immutable_result_ledgers(rules, observations, snapshot)
    after = deepcopy(before)
    for index in range(offset, end):
        observation = observations[index]
        for rule in rules:
            identifier = rule["rule_id"]
            values = valuations[index][identifier]
            transition = checker.transition_evidence(
                identifier,
                after[identifier],
                frozenset(label for label, active in values.items() if active),
            )
            check = {
                "rule_id": identifier,
                "specification": rule["id"],
                "binding": deepcopy(rule["binding"]),
                "observation_index": index,
                "time": observation["time"],
                "phase": observation["phase"],
                "ap_values": values,
                "transition": transition,
                "active_steps": deepcopy(observation.get("active_steps", [])),
            }
            result["rule_checks"].append(check)
            if transition["status"] != "passed":
                if transition["reason"] != "accepting_state_unreachable":
                    raise ValueError(transition["reason"])
                result.update(
                    status="violated",
                    feasibility_status="INFEASIBLE",
                    reason=rule["requirement"],
                    reason_code="safety_rule_violation",
                    counterexample=check,
                )
                return result
            after[identifier] = transition["to"]
    pending = sorted(
        identifier
        for identifier, state in after.items()
        if state not in checker.dfas[identifier]["accepting_states"]
    )
    result["pending_rule_ids"] = pending
    if trace_complete and pending:
        identifier = pending[0]
        rule = next(row for row in rules if row["rule_id"] == identifier)
        result.update(
            status="violated",
            feasibility_status="INFEASIBLE",
            reason=rule["requirement"],
            reason_code="safety_rule_violation",
            counterexample={
                "rule_id": identifier,
                "specification": rule["id"],
                "binding": deepcopy(rule["binding"]),
                "observation_index": end - 1,
                "time": observations[-1]["time"],
                "phase": observations[-1]["phase"],
                "ap_values": deepcopy(valuations[-1][identifier]),
                "active_steps": deepcopy(observations[-1].get("active_steps", [])),
                "reason": "nonaccepting_at_trace_completion",
            },
        )
        return result
    candidate = {
        "clock_version": clock_version,
        "trace_id": result["trace_id"],
        "next_observation": end,
        "states": deepcopy(after),
        "previous_observation": deepcopy(observations[end - 1]),
        "trace_complete": trace_complete,
    }
    result.update(
        status="satisfied" if trace_complete else "prefix_checked",
        is_safe=trace_complete,
        reason="",
        reason_code="",
        feasibility_status=None,
        safety_dfa_states_after=after,
        continuation=deepcopy(candidate),
        candidate_continuation=candidate,
    )
    result["projected_snapshot"] = deepcopy(snapshot)
    result["projected_snapshot"].update(
        resources=deepcopy(observations[end - 1]["resources"]),
        parts=deepcopy(observations[end - 1]["parts"]),
    )
    return result


def validate_reviewed_primitive_program_safety(  # noqa: PLR0913
    *,
    programs: list[dict[str, Any]],
    snapshot: dict[str, Any],
    geometry: dict[str, Any],
    horizon: list[float],
    stationary: dict[str, Any],
    catalog: dict[str, Any],
    applicability: dict[str, Any],
    continuation: dict[str, Any] | None = None,
    trace_complete: bool = False,
    observation_slice: list[int] | None = None,
    primitive_models: dict | None = None,
) -> dict[str, Any]:
    """Evaluate reviewed formulas over contiguous slices of one frozen trace.

    Args:
        programs: Exact bound primitive programs and resource-model evidence.
        snapshot: Joint initial physical state and complete process ledgers.
        geometry: Configured physical envelopes and region bounds.
        horizon: Complete joint trace horizon, in seconds.
        stationary: Explicit stationary coverage for modeled resources.
        catalog: Version 1 reviewed requirements with fixed AP meanings.
        applicability: Complete declared population, exclusions, and bindings.
        continuation: Previously accepted continuation from this exact trace.
        trace_complete: Whether this call declares the actual trace end.
        observation_slice: Optional half-open [start, end] observation indices.
        primitive_models: Trusted pure resource models, never serialized code.

    Returns:
        Status, coverage, AP evidence, transitions, and candidate continuation.
        Prefix checking is not final satisfaction or joint feasibility. Rejected
        and unavailable checks retain the input continuation and monitor states.
    """
    result = _new_result(continuation)
    try:
        if type(trace_complete) is not bool:
            raise ValueError("trace_complete must be an explicit Boolean")
        definitions = _catalog(catalog)
        rules, result["coverage"] = _rules(definitions, applicability, geometry)
        result["bindings"] = [
            {"rule_id": rule["rule_id"], **deepcopy(rule["binding"])} for rule in rules
        ]
        physical_geometry = deepcopy(geometry)
        for part in _object(physical_geometry.get("parts"), "geometry.parts").values():
            if "target" in part:
                _symbol(part.pop("target"), "part geometry.target")
        model = model_primitive_observations(
            programs=programs,
            snapshot=snapshot,
            geometry=physical_geometry,
            horizon=horizon,
            stationary=stationary,
            bindings=[],
            primitive_models=primitive_models,
        )
        return _evaluate_frozen_trace(
            result=result,
            model=model,
            snapshot=snapshot,
            rules=rules,
            fingerprint={
                "programs": programs,
                "snapshot": snapshot,
                "geometry": geometry,
                "horizon": horizon,
                "stationary": stationary,
                "catalog": catalog,
                "applicability": applicability,
            },
            continuation=continuation,
            trace_complete=trace_complete,
            observation_slice=observation_slice,
        )
    except (ImportError, OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        result["reason"] = str(exc)
        return result

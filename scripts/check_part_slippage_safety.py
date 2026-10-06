"""Check supplied mock part_slippage traces against the predefined specifications.

Only the existing offline observation/checking API is called. The fixture programs
use that API's physical-model contract; they are not controller dispatch programs.
Synthetic task acknowledgements are assumptions of these traces, not live history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from copy import deepcopy
from pathlib import Path

from cais_spade_llm.agents.central_controller.offline_safety_grounding import (
    validate_grounded_primitive_program_safety,
)
from cais_spade_llm.agents.central_controller.predefined_safety import (
    compile_predefined_safety,
    parse_predefined_safety,
)
from cais_spade_llm.recovery_framework import fingerprint
from cais_spade_llm.resources.environment_models import build_environment_models

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "test/fixtures/part_slippage/predefined_safety"
CASES = ("mutex", "precedence", "safe")


def load_candidate(name: str, directory: Path = FIXTURES) -> tuple[dict, dict, dict]:
    """Load a frozen mock candidate and the unchanged supplied safety document."""
    if name not in CASES:
        raise ValueError("Unknown mock candidate: " + name)
    context = json.loads((directory / "initial_context.json").read_text(encoding="utf-8"))
    candidate = json.loads((directory / (name + ".json")).read_text(encoding="utf-8"))
    source = context["sources"]["predefined_safety"]
    raw = (ROOT / source["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("Predefined specifications changed; review this fixture before replay")
    document = parse_predefined_safety(raw.decode("utf-8"))
    if document is None or context.get("synthetic") is not True or candidate.get("synthetic") is not True:
        raise ValueError("This demonstration requires predefined rules and explicitly synthetic inputs")
    if candidate["initial_context_file"] != "initial_context.json":
        raise ValueError("Candidate references a different checkpoint")
    inputs = {field: deepcopy(context[field]) for field in ("scene", "geometry", "horizon")}
    inputs.update(deepcopy(candidate["inputs"]))
    inputs.update(catalog=deepcopy(document["catalog"]),
                  requirement_scopes=deepcopy(document["requirement_scopes"]))
    validate_event_records(candidate, inputs)
    return candidate, inputs, document


def validate_event_records(candidate: dict, inputs: dict) -> None:
    """Require exact event/step associations and chronological predecessor records."""
    events = candidate["outline_events"]
    by_id = {event["outline_id"]: event for event in events}
    if len(by_id) != len(events) or len({event["des_event_id"] for event in events}) != len(events):
        raise ValueError("Recovery event identities must be unique")
    programs = {program["resource_id"]: program for program in inputs["programs"]}
    if len(programs) != len(inputs["programs"]):
        raise ValueError("Supply one ordered observation program per resource")
    covered = set()
    for event in events:
        resource = event["resource_id"]
        program = programs[resource]
        indices = event["primitive_step_indices"]
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("An event must contain contiguous ordered primitives")
        if (program["step_results"][indices[0]]["start_time"] != event["start_time"]
                or program["step_results"][indices[-1]]["end_time"] != event["end_time"]):
            raise ValueError("Event and primitive times disagree")
        for previous in event["predecessors"]:
            if previous not in by_id or by_id[previous]["end_time"] > event["start_time"]:
                raise ValueError("Predecessor has not completed before the event starts")
        for local_index, index in enumerate(indices):
            key = (resource, index)
            if key in covered:
                raise ValueError("A primitive cannot belong to two recovery events")
            covered.add(key)
            step, trace = program["primitive_steps"][index], program["step_results"][index]
            source = step["source"]
            if source != trace["source"] or source["primitive_index"] != local_index:
                raise ValueError("Primitive provenance differs from its authored event")
            for field in ("outline_id", "des_event_id", "event_name", "resource_id", "resource_jid"):
                if source[field] != event[field]:
                    raise ValueError("Primitive provenance differs from " + field)
            if step["primitive"] != trace["primitive"] or step["params"] != trace["resolved_params"]:
                raise ValueError("Primitive parameters differ from their bound evidence")
    expected = {(resource, index) for resource, program in programs.items()
                for index in range(len(program["primitive_steps"]))}
    if covered != expected:
        raise ValueError("Recovery records omit primitive references")
    _validate_task_records(inputs["task_evidence"]["events"], by_id)


def _validate_task_records(tasks: list[dict], by_id: dict) -> None:
    for task in tasks:
        event = by_id[task["task_id"]]
        if (task["function"] != event["event_name"] or task["resource_id"] != event["resource_id"]
                or task["start_time"] != event["start_time"] or task["end_time"] != event["end_time"]
                or task["declared_task"] != event["declared_task"]):
            raise ValueError("Synthetic completion must refer to the exact declared recovery event")


def validate_observed_states(candidate: dict, result: dict) -> None:
    """Compare saved boundary/final expectations to observations, including rejected traces."""
    observations = result["observations"]
    if not observations:
        raise ValueError("No complete physical trace is available")
    boundaries = {row["time"]: row for row in observations if row["phase"] == "at"}
    for event in candidate["outline_events"]:
        for expected_field, time_field in (("expected_start_state", "start_time"),
                                           ("expected_end_state", "end_time")):
            actual = boundaries[event[time_field]]["resources"][event["resource_id"]]
            if any(actual.get(key) != value for key, value in event[expected_field].items()):
                raise ValueError("Recovery boundary expectation differs from modeled custody/pose")
    last = observations[-1]
    for kind, entries in candidate["expected_final"].items():
        for identity, fields in entries.items():
            if any(last[kind][identity].get(key) != value for key, value in fields.items()):
                raise ValueError("Final observation differs from the saved recovery endpoint")


def _evidence_changes(result: dict, actors: set[str]) -> list[dict]:
    previous, changes = {}, []
    for row in result["rule_checks"]:
        pair = row["binding"].get("resources")
        if pair and set(pair) != actors:
            continue
        identity = row["rule_id"]
        if previous.get(identity) == row["ap_values"]:
            continue
        previous[identity] = row["ap_values"]
        changes.append({field: deepcopy(row[field]) for field in (
            "specification", "rule_id", "binding", "observation_index", "time", "phase",
            "ap_values", "transition", "active_steps")})
    return changes


def _counterexample(result: dict) -> dict | None:
    counterexample = deepcopy(result["counterexample"])
    if counterexample is None:
        return None
    index = counterexample["observation_index"]
    observation = result["observations"][index]
    counterexample["time_exact"] = observation["time_exact"]
    counterexample["aps"] = [deepcopy(row) for row in result["ap_evidence"]
                             if row["rule_id"] == counterexample["rule_id"]
                             and row["observation_index"] == index]
    counterexample["region_occupancy"] = deepcopy(observation["region_occupancy"])
    counterexample["carried_parts"] = deepcopy(observation["carried_parts"])
    return counterexample


def check_candidates(directory: Path = FIXTURES) -> dict:
    """Evaluate all three traces, including each unchanged specification separately."""
    cases = []
    definitions = None
    for name in CASES:
        candidate, inputs, document = load_candidate(name, directory)
        before = deepcopy(inputs)
        compiled, dfas = compile_predefined_safety(document)
        definitions = [{**deepcopy(row), "dfa_dot": dfas[row["id"]]} for row in compiled]
        result = validate_grounded_primitive_program_safety(**inputs, trace_complete=True)
        if result["status"] in {"satisfied", "violated"}:
            validate_observed_states(candidate, result)
        actors = {program["resource_id"] for program in inputs["programs"]}
        per_specification, changes = {}, []
        for definition in document["catalog"]["specifications"]:
            identifier = definition["id"]
            separate = deepcopy(inputs)
            separate["catalog"]["specifications"] = [deepcopy(definition)]
            separate["requirement_scopes"] = [deepcopy(row) for row in inputs["requirement_scopes"]
                                              if row["specification"] == identifier]
            checked = validate_grounded_primitive_program_safety(**separate, trace_complete=True)
            per_specification[identifier] = {
                "status": checked["status"], "reason": checked["reason"],
                "counterexample": _counterexample(checked),
                "concrete_rule_count": len(checked["bindings"]),
            }
            changes.extend(_evidence_changes(checked, actors))
        if inputs != before:
            raise AssertionError("Offline checking mutated its supplied checkpoint")
        final = result["observations"][-1] if result["observations"] else None
        cases.append({
            "case_id": candidate["case_id"], "file": name + ".json",
            "candidate_origin": "mock", "evidence_origin": "synthetic",
            "status": result["status"], "reason": result["reason"],
            "trace_id": result["trace_id"], "clock_version": result["clock_version"],
            "input_fingerprint": fingerprint(inputs), "dispatch_authorized": False,
            "configured_resources": sorted(build_environment_models(inputs["scene"])),
            "concrete_rule_count": len(result["bindings"]),
            "specifications": per_specification, "counterexample": _counterexample(result),
            "ap_changes": sorted(changes, key=lambda row: (row["time"], row["observation_index"], row["rule_id"])),
            "events": deepcopy(candidate["outline_events"]),
            "expected": deepcopy(candidate["expected"]),
            "expectations_met": (result["status"] == candidate["expected"]["combined"]
                                 and all(row["status"] == candidate["expected"][key]
                                         for key, row in per_specification.items())),
            "modeled_final_state": None if final is None else {
                "resources": {rid: {field: final["resources"][rid].get(field)
                                     for field in ("held_part", "gripper_state", "current_pose")}
                              for rid in sorted(actors)},
                "parts": {part: {field: final["parts"][part].get(field)
                                  for field in ("current_pose", "contained_by", "processCompleted")}
                          for part in inputs["snapshot"]["parts"]},
                "pending_tasks": deepcopy(inputs["snapshot"]["nominal_tasks"]),
            },
        })
    context = json.loads((directory / "initial_context.json").read_text(encoding="utf-8"))
    return {"version": 1, "experiment": "part_slippage", "dispatch_authorized": False,
            "candidate_origin": "mock", "evidence_origin": "synthetic", "llm_calls": 0,
            "sources": context["sources"], "modeling_assumptions": context["modeling_assumptions"],
            "fixture_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                               for name in ("initial_context.json", *(case + ".json" for case in CASES))},
            "definitions": definitions, "cases": cases,
            "expectations_met": all(row["expectations_met"] for row in cases)}


def render_report(report: dict, output_directory: Path = FIXTURES) -> str:
    """Render the computed verdicts and their event/primitive counterexamples."""
    lines = ["# Mock part_slippage recovery: predefined safety results", "",
             "Mock plans and synthetic physical evidence; no LLM calls or robot execution.", "",
             "The existing offline checker derives AP values and evaluates the unchanged predefined LTLf/DFA definitions.", "",
             "| Candidate | Mutex | Precedence | Combined |", "|---|---|---|---|"]
    for case in report["cases"]:
        verdicts = [row["status"] for row in case["specifications"].values()]
        candidate_path = os.path.relpath(FIXTURES / case["file"], output_directory)
        lines.append(f"| [{case['case_id']}]({candidate_path}) | {verdicts[0]} | {verdicts[1]} | {case['status']} |")
    population = len(report["cases"][0]["configured_resources"])
    lines += ["", f"The frozen scene contains {population} configured resources. Each complete trace requires "
              f"{population * (population - 1) // 2} mutex instances and one precedence instance.",
              "The initial checkpoint has `gear_small` displaced near `ur5e-3`, and `KET4_Square_4mm` in `Buffer For Machined parts`.", ""]
    for case in report["cases"]:
        lines += ["## " + case["case_id"], ""]
        for event in sorted(case["events"], key=lambda row: (row["start_time"], row["resource_id"])):
            lines.append(f"- `{event['resource_id']}` {event['start_time']}–{event['end_time']} s: `{event['event_name']}`.")
        counterexample = case["counterexample"]
        if counterexample:
            lines += ["", f"**Rejected by `{counterexample['specification']}` at t={counterexample['time']:.6f} s** "
                      f"(exact time `{counterexample['time_exact']}`).",
                      f"AP values: `{json.dumps(counterexample['ap_values'], sort_keys=True)}`.", ""]
            for step in counterexample["active_steps"]:
                lines.append(f"- Event `{step['outline_id']}`, primitive index `{step['primitive_index']}` "
                             f"(zero-based): `{step['primitive']}` on `{step['resource_id']}`.")
            for ap in counterexample["aps"]:
                lines.append(f"- `{ap['descriptor']['label']}`: `{ap['descriptor']['full']}`, "
                             f"binding `{json.dumps(ap['binding'], sort_keys=True)}`, value `{ap['value']}`.")
        else:
            lines += ["", f"**Combined result: `{case['status']}`.**"]
        lines.append("")
    findings = ("The mutex-only candidate fails mutex and passes precedence. The precedence-only candidate passes mutex and fails precedence. The safe candidate satisfies both over its complete supplied trace."
                if report["expectations_met"] else
                "The expected outcomes were not all established. Consult the recorded statuses and reasons; no passing result is inferred.")
    lines += ["## What this establishes", "", findings, "",
              "`ur5e-4` recovers `gear_small`; `ur5e-3` clears the board and retrieves `KET4_Square_4mm`. Each part has retrieve, return, and place/retreat events, with multiple ordered primitives. Only timing/order differ between candidates.", "",
              "The safe candidate has a separately supplied synthetic `gear_small` assembly acknowledgement at t=9 s. `KET4_Square_4mm` subsequently enters the board: precedence is checked non-vacuously. Its placement preserves its trim record and does not complete its pending assembly task.", "",
              "## Model boundary", "", *["- " + row for row in report["modeling_assumptions"]], "",
              "Re-run from the project root with `poetry run python scripts/check_part_slippage_safety.py`. "
              "The detailed [report.json](report.json) includes exact formulas, AP descriptors, DFA transitions, bindings, fingerprints, and modeled final states.", ""]
    return "\n".join(lines)


def main() -> int:
    """Save the three computed results without changing any runtime state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=FIXTURES)
    args = parser.parse_args()
    report = check_candidates()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_directory / "REPORT.md").write_text(render_report(report, args.output_directory), encoding="utf-8")
    for case in report["cases"]:
        logger.info("%s: %s", case["case_id"], case["status"])
    logger.info("Report saved to %s", args.output_directory / "REPORT.md")
    return 0 if report["expectations_met"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())

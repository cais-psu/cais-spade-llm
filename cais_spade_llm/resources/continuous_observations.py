"""Joint physical evidence with bounded motion and explicit custody boundaries.

Versioned owner contracts can supply grasp/release effects and carried-part
geometry. Deposited parts need stationary coverage until the next grasp.
Unknown occupancy remains an alternative, never an authoritative false AP.
"""

from __future__ import annotations

import math
from copy import deepcopy
from fractions import Fraction
from functools import partial
from types import SimpleNamespace

from cais_spade_llm.recovery_framework import fingerprint
from cais_spade_llm.resources.continuous_motion import (
    ContinuousMotion,
    interval,
    occupancy,
    quaternion_transform,
    transformed_box,
)
from cais_spade_llm.resources.primitive_observations import _coverage, _interval, _number
from cais_spade_llm.resources.primitive_observations_3d import _compose, _Model, _plain, _unit_pose
from cais_spade_llm.resources.resource_safety_preparation import primitive_model_descriptors

CLOCK_VERSION = "continuous_physical_boundaries_v1"


def has_continuous_motion(programs):
    """Detect an explicitly declared evidence contract, never a resource name."""
    return any(
        "continuous_motion" in row.get("model_evidence", {})
        for program in programs
        for row in program.get("step_results", [])
    )


def _prepare_step(command, row, rid, previous_steps, snapshot, horizon, primitive_models, budget):
    if budget:
        budget.check()
    if (
        row.get("success") is not True
        or row["primitive"] != command["primitive"]
        or row["resolved_params"] != command["params"]
    ):
        raise ValueError("Prepared native primitive differs from its authored command")
    if row.get("source") != command.get("source"):
        raise ValueError("Continuous primitive provenance changed")
    start, end = _number(row["start_time"], "start"), _number(row["end_time"], "end")
    if (
        not horizon[0] <= start < end <= horizon[1]
        or previous_steps
        and start < previous_steps[-1]["end"]
    ):
        raise ValueError("Invalid continuous primitive ordering")
    effect = primitive_models[rid].effects(
        primitive=row["primitive"],
        params=row["resolved_params"],
        evidence=row["model_evidence"],
        start_time=start,
        end_time=end,
    )
    if (
        effect["trajectory"] is not None
        or effect["base_trajectory"] is not None
        or any(effect[key] for key in ("part_trajectories", "transfers", "resource_updates"))
    ):
        raise ValueError("Continuous motion cannot silently include unresolved additional effects")
    raw = effect["continuous_motion"]
    motion = ContinuousMotion(raw["joint_trajectory"], raw["configuration"])
    if motion.exact_times[-1] != end - start:
        raise ValueError("Prepared motion duration changed")
    if (
        previous_steps
        and previous_steps[-1]["motion"].points[-1]["positions"] != motion.points[0]["positions"]
    ):
        raise ValueError("Discontinuous prepared joints")
    if not previous_steps:
        joints = snapshot["resources"][rid].get("joint_positions")
        if joints != motion.points[0]["positions"]:
            raise ValueError("Continuous motion does not start at observed joints")
    return start, end, effect, motion


def _prepare_movements(
    programs,
    snapshot,
    horizon,
    stationary,
    primitive_models,
    descriptors,
    observation_boundaries,
    budget,
):
    movements, effects, critical = {}, [], set(horizon)
    for raw in observation_boundaries or []:
        value = _number(raw, "boundary")
        if not horizon[0] <= value <= horizon[1]:
            raise ValueError("Observation boundary outside continuous horizon")
        critical.add(value)
    for program in programs:
        rid = program["resource_id"]
        if rid in movements or rid not in descriptors:
            raise ValueError("Continuous programs need one registered owner per resource")
        authored, results = program["primitive_steps"], program["step_results"]
        if not authored or len(authored) != len(results) or program.get("validation_error"):
            raise ValueError("Incomplete continuous primitive program")
        movements[rid] = []
        for index, (command, row) in enumerate(zip(authored, results, strict=True)):
            start, end, effect, motion = _prepare_step(
                command, row, rid, movements[rid], snapshot, horizon, primitive_models, budget
            )
            movements[rid].append(
                {
                    "start": start,
                    "end": end,
                    "motion": motion,
                    "source": deepcopy(row["source"]),
                    "primitive": row["primitive"],
                    "step_index": index,
                    "custody_effects": deepcopy(effect.get("custody_effects", [])),
                }
            )
            critical.update((start, end))
            effects.append(
                {
                    "source": row["source"],
                    "resource_id": rid,
                    "primitive": row["primitive"],
                    "params": row["resolved_params"],
                    "start_time": str(start),
                    "end_time": str(end),
                    "effects": effect,
                }
            )
    if set(movements) - set(snapshot["resources"]) or set(stationary) - set(snapshot["resources"]):
        raise ValueError("Continuous evidence refers to unconfigured resources")
    for rid in snapshot["resources"]:
        steps = movements.get(rid, [])
        _coverage(
            rid,
            [SimpleNamespace(start=s["start"], end=s["end"], trajectory=[]) for s in steps],
            stationary,
            horizon,
        )
        for a, b in (_interval(raw, "stationary") for raw in stationary.get(rid, [])):
            if any(max(a, s["start"]) < min(b, s["end"]) for s in steps):
                raise ValueError("Stationary evidence overlaps commanded motion")
    return movements, effects, critical


def _chosen_step(rid, time, movements):
    steps = movements.get(rid, [])
    if not steps:
        return None, None
    chosen = steps[0]
    for step in steps:
        if step["start"] <= time:
            chosen = step
    local = max(0, min(chosen["motion"].exact_times[-1], time - chosen["start"]))
    return chosen, local


def _continuous_state(snapshot, movements, time, custody):
    state = deepcopy(snapshot)
    for end, rid, effect in custody:
        if end > time:
            break
        resource, part = state["resources"][rid], state["parts"][effect["part"]]
        if effect["kind"] == "grasp":
            receiver = part["contained_by"]
            if receiver is not None:
                state["resources"][receiver]["contained_parts"].remove(effect["part"])
            resource.update(held_part=effect["part"], gripper_state="closed",
                            grasp_transform=deepcopy(effect["grasp_transform"]))
            part["contained_by"] = None
        else:
            resource.update(held_part=None, gripper_state="open", grasp_transform=None)
            part.update(current_pose=deepcopy(effect["pose"]), contained_by=effect["contained_by"],
                        stationary_until=effect["stationary_until"])
            if effect["contained_by"] is not None:
                state["resources"][effect["contained_by"]]["contained_parts"].append(effect["part"])
    for rid, resource in state["resources"].items():
        step, local = _chosen_step(rid, time, movements)
        if step is not None:
            resource["current_pose"] = step["motion"].reference_pose(local)
            resource["joint_positions"] = [sum(pair) / 2 for pair in step["motion"].joints(local, local).values()]
        if resource.get("held_part") is not None:
            part = state["parts"][resource["held_part"]]
            part["current_pose"] = _plain(_compose(tuple(resource["current_pose"]), tuple(resource["grasp_transform"])))
    return state


def _validate_custody(snapshot, movements, geometry, horizon):
    custody = sorted(
        [(step["end"], rid, effect) for rid, steps in movements.items()
         for step in steps for effect in step["custody_effects"]],
        key=lambda row: (row[0], row[2].get("kind") != "release", row[1]),
    )
    accepted = []
    simultaneous = set()
    for end, rid, effect in custody:
        if not isinstance(effect, dict) or effect.get("kind") not in {"grasp", "release"}:
            raise ValueError("Unsupported continuous custody effect")
        kind, part = effect["kind"], effect.get("part")
        required = ({"kind", "part", "pose", "grasp_transform"} if kind == "grasp" else
                    {"kind", "part", "pose", "contained_by", "stationary_until"})
        if set(effect) != required or part not in snapshot["parts"] or (end, part) in simultaneous:
            raise ValueError("Incomplete or simultaneous conflicting custody evidence")
        simultaneous.add((end, part))
        state = _continuous_state(snapshot, movements, end, accepted)
        resource, actual = state["resources"][rid], state["parts"][part]
        pose = _unit_pose(effect["pose"], "continuous custody pose")
        if math.dist(pose[:3], actual["current_pose"][:3]) > 1e-9:
            raise ValueError("Custody effect would move the part without motion evidence")
        if any(abs(a-b) > 1e-9 for a,b in zip(pose[3:], actual["current_pose"][3:], strict=True)):
            raise ValueError("Custody effect changes an unmodeled part orientation")
        if kind == "grasp":
            if resource.get("held_part") is not None or any(row.get("held_part") == part for row in state["resources"].values()):
                raise ValueError("Conflicting continuous grasp custody")
            grasp = _unit_pose(effect["grasp_transform"], "continuous grasp transform")
            derived = _compose(tuple(resource["current_pose"]), grasp)
            if any(abs(a-b) > 1e-9 for a,b in zip(derived, pose, strict=True)):
                raise ValueError("Continuous grasp transform disagrees with the part pose")
        else:
            if resource.get("held_part") != part:
                raise ValueError("Continuous release does not match current custody")
            receiver = effect["contained_by"]
            if receiver is not None and receiver not in state["resources"]:
                raise ValueError("Continuous release containment owner is unavailable")
            until = _number(effect["stationary_until"], "release stationary coverage")
            if not end <= until <= horizon[1]:
                raise ValueError("Invalid continuous release stationary coverage")
        bounds = geometry["parts"][part].get("local_bounds")
        if bounds is None:
            raise ValueError("Carried part requires configured local geometry")
        transformed_box(quaternion_transform(pose[:3], pose[3:]), bounds, part)
        accepted.append((end, rid, effect))
    _Model._inventories(_continuous_state(snapshot, movements, horizon[1], accepted))
    _validate_stationary_parts(snapshot, accepted, horizon)
    return accepted


def _validate_stationary_parts(snapshot, custody, horizon):
    # Explicit deposited coverage must reach the next acquisition; it does not
    # assert that an initially stationary part stays stationary while carried.
    for part, initial in snapshot["parts"].items():
        relevant = [row for row in custody if row[2]["part"] == part]
        segments, start, row = [], horizon[0], initial
        held = any(r.get("held_part") == part for r in snapshot["resources"].values())
        for end, _, effect in relevant:
            if not held:
                segments.append((start, end, row))
            held = effect["kind"] == "grasp"
            start, row = end, effect
        if not held:
            segments.append((start, horizon[1], row))
        for start, end, source in segments:
            rows = deepcopy(source.get("stationary_intervals", []))
            if "stationary_until" in source:
                rows.append([float(start), source["stationary_until"]])
            clipped = []
            for raw in rows:
                a, b = _interval(raw, "stationary part coverage")
                if not horizon[0] <= a <= b <= horizon[1]:
                    raise ValueError("Stationary part evidence exceeds the horizon")
                if max(a, start) <= min(b, end):
                    clipped.append([float(max(a, start)), float(min(b, end))])
            _coverage(part, [], {part: clipped}, (start, end))


def _refine(start, end, values, budget):
    cells = [(start, end, 0)]
    proof = []
    while cells:
        if budget:
            budget.check()
        a, b, depth = cells.pop()
        v = values(a, b)
        ambiguous = any(len(x) > 1 for resource in v.values() for x in resource.values())
        if ambiguous and depth < 4:
            mid = (a + b) / 2
            cells.extend(((mid, b, depth + 1), (a, mid, depth + 1)))
        else:
            proof.append({"interval": [str(a), str(b)], "occupancy_possibilities": v})
    return proof


def _at(rid, start, end, movements, cache):
    steps = movements.get(rid, [])
    if not steps:
        return None
    chosen = steps[0]
    for step in steps:
        if step["start"] <= start:
            chosen = step
    local_start = max(0, min(chosen["motion"].exact_times[-1], start - chosen["start"]))
    local_end = max(0, min(chosen["motion"].exact_times[-1], end - chosen["start"]))
    key = (rid, chosen["step_index"], local_start, local_end)
    if key not in cache:
        cache[key] = chosen["motion"].boxes(local_start, local_end)
    return cache[key]


def _part_values(start, end, *, movements, cache, geometry, snapshot, custody, initial_observation):
    key = ("parts", start, end)
    if key in cache:
        return cache[key]
    state = _continuous_state(snapshot, movements, start, custody)
    boxes = {}
    for part, row in state["parts"].items():
        holder = next((rid for rid, resource in state["resources"].items() if resource.get("held_part") == part), None)
        bounds = geometry["parts"][part].get("local_bounds")
        if holder is not None:
            if bounds is None:
                raise ValueError("Carried part requires configured local geometry")
            step, a = _chosen_step(holder, start, movements)
            if step is None:
                matrix = quaternion_transform(row["current_pose"][:3], row["current_pose"][3:])
                boxes[part] = [transformed_box(matrix, bounds, part)]
            else:
                b = max(0, min(step["motion"].exact_times[-1], end - step["start"]))
                boxes[part] = [step["motion"].carried_part_box(
                    a, b, state["resources"][holder]["grasp_transform"], bounds, part)]
        elif bounds is not None:
            pose = row["current_pose"]
            boxes[part] = [transformed_box(quaternion_transform(pose[:3], pose[3:]), bounds, part)]
    result = {region: {part: occupancy(boxes[part], shape["bounds"]) if part in boxes else
                       [initial_observation["part_region_occupancy"][region][part]]
                       for part in snapshot["parts"]}
              for region, shape in geometry["regions"].items()}
    cache[key] = result
    return result


def _values(start, end, *, movements, cache, geometry, snapshot, initial_observation, custody, part_values):  # noqa: PLR0913
    result = {}
    parts = part_values(start, end)
    state = _continuous_state(snapshot, movements, start, custody)
    for region, shape in geometry["regions"].items():
        result[region] = {}
        for rid in snapshot["resources"]:
            boxes = _at(rid, start, end, movements, cache)
            if boxes is None and "component_bounds" in geometry["resources"][rid]:
                boxes = []
                components = geometry["resources"][rid]["component_bounds"]
                if not components:
                    raise ValueError("Complete stationary resource component bounds are required")
                for component in components:
                    bounds = [interval(*pair) for pair in component["bounds"]]
                    if len(bounds) != 3 or not component["id"]:
                        raise ValueError("Invalid stationary resource component bounds")
                    boxes.append(
                        {
                            "id": component["id"],
                            "minimum": [[lo, lo] for lo, _ in bounds],
                            "maximum": [[hi, hi] for _, hi in bounds],
                        }
                    )
            result[region][rid] = (
                occupancy(boxes, shape["bounds"])
                if boxes is not None
                else [initial_observation["region_occupancy"][region][rid]]
            )
            carried = [state["resources"][rid].get("held_part")]
            carried.extend(effect["part"] for time, owner, effect in custody
                           if time == start == end and owner == rid and effect["kind"] == "release")
            for part in carried:
                if part is not None:
                    result[region][rid] = sorted({a or b for a in result[region][rid] for b in parts[region][part]})
    return result


def _observe(start, end, phase, initial_observation, movements, values, budget,  # noqa: PLR0913
             snapshot, custody, part_values):
    if budget:
        budget.check()
    time = (start + end) / 2
    row = deepcopy(initial_observation)
    possibilities = values(start, end)
    row.update(
        time=float(time),
        time_exact=str(time),
        phase=phase,
        occupancy_possibilities=possibilities,
        continuous_interval=[str(start), str(end)],
        active_steps=[],
    )
    state = _continuous_state(snapshot, movements, time, custody)
    row.update(resources=state["resources"], parts=state["parts"],
               part_occupancy_possibilities=part_values(start, end),
               carried_parts={rid: [] if resource.get("held_part") is None else [resource["held_part"]]
                              for rid, resource in state["resources"].items()})
    for region, parts in row["part_occupancy_possibilities"].items():
        row["part_region_occupancy"][region] = {name: outcomes[0] if len(outcomes) == 1 else None
                                              for name, outcomes in parts.items()}
    for region, resources in possibilities.items():
        row["region_occupancy"][region] = {
            rid: outcomes[0] if len(outcomes) == 1 else None for rid, outcomes in resources.items()
        }
    for rid, steps in movements.items():
        previous = None
        for step in steps:
            if step["start"] <= time:
                previous = step
            if step["start"] <= time <= step["end"]:
                row["active_steps"].append(
                    {
                        "resource_id": rid,
                        "primitive": step["primitive"],
                        "step_index": step["step_index"],
                        "source": step["source"],
                    }
                )
        if previous:
            local = min(previous["motion"].duration, float(time - previous["start"]))
            row["resources"][rid]["current_pose"] = previous["motion"].reference_pose(local)
            if time >= previous["end"]:
                row["resources"][rid]["joint_positions"] = deepcopy(
                    previous["motion"].points[-1]["positions"]
                )
    # Refinement bounds tighten the set of possible words, without emitting
    # numerical subdivision points as semantic observations.
    if start < end:
        proof = _refine(start, end, values, budget)
        for cell in proof:
            a, b = (Fraction(value) for value in cell["interval"])
            cell["part_occupancy_possibilities"] = part_values(a, b)
        row["continuous_cells"] = proof
    return row


def model_continuous_observations(
    *,
    programs,
    snapshot,
    geometry,
    horizon,
    stationary,
    primitive_models,
    observation_boundaries=None,
    budget=None,
):
    """Prepare all physical alternatives without evaluating a safety formula."""
    horizon = _interval(horizon, "horizon")
    if horizon[0] >= horizon[1]:
        raise ValueError("Continuous checking needs a positive horizon")
    descriptors = primitive_model_descriptors(primitive_models)
    # Reuse the complete initial custody, inventory and geometry validator at a
    # zero-duration checkpoint. This supplies no future stationary assertion.
    initial = deepcopy(snapshot)
    for part in initial["parts"].values():
        part["stationary_until"] = float(horizon[0])
        part["stationary_intervals"] = [[float(horizon[0]), float(horizon[0])]]
    baseline_geometry = deepcopy(geometry)
    for shape in baseline_geometry["resources"].values():
        shape.pop("component_bounds", None)
    for shape in baseline_geometry["parts"].values():
        shape.pop("local_bounds", None)
    baseline = _Model(
        initial,
        baseline_geometry,
        (horizon[0], horizon[0]),
        [],
        {rid: [[float(horizon[0]), float(horizon[0])]] for rid in initial["resources"]},
        primitive_models,
    )
    movements, effects, critical = _prepare_movements(
        programs,
        snapshot,
        horizon,
        stationary,
        primitive_models,
        descriptors,
        observation_boundaries,
        budget,
    )
    initial_observation = baseline.observe(horizon[0])
    initial_observation["parts"] = deepcopy(snapshot["parts"])
    cache = {}
    custody = _validate_custody(snapshot, movements, geometry, horizon)
    part_values = partial(_part_values, movements=movements, cache=cache, geometry=geometry,
                          snapshot=snapshot, custody=custody, initial_observation=initial_observation)

    values = partial(
        _values,
        movements=movements,
        cache=cache,
        geometry=geometry,
        snapshot=snapshot,
        initial_observation=initial_observation,
        custody=custody, part_values=part_values,
    )

    observations, certificates = [], []

    ordered = sorted(critical)
    for index, time in enumerate(ordered):
        observations.append(
            _observe(time, time, "at", initial_observation, movements, values, budget,
                     snapshot, custody, part_values)
        )
        if index + 1 < len(ordered):
            row = _observe(
                time, ordered[index + 1], "between", initial_observation, movements, values, budget,
                snapshot, custody, part_values,
            )
            observations.append(row)
            certificates.append(
                {"interval": row["continuous_interval"], "cells": row["continuous_cells"]}
            )
    preceding = None
    for row in observations:
        row["part_entry_possibilities"] = {}
        for region, parts in row["part_occupancy_possibilities"].items():
            entries = row["part_entry_possibilities"][region] = {}
            for part, possible in parts.items():
                if preceding is None:
                    entries[part] = [False]
                elif row["phase"] == "between":
                    entries[part] = [False] if possible == [False] or possible == [True] and preceding[region][part] == [True] else [False, True]
                else:
                    entries[part] = sorted({current and not prior for current in possible for prior in preceding[region][part]})
        if row["phase"] == "at":
            preceding = row["part_occupancy_possibilities"]
    final = deepcopy(snapshot)
    final.update(
        resources=deepcopy(observations[-1]["resources"]), parts=deepcopy(observations[-1]["parts"])
    )
    return {
        "valid": True,
        "feasibility_status": "FEASIBLE",
        "reason": "",
        "observations": _plain(observations),
        "projected_snapshot": _plain(final),
        "evidence": {
            "offline": True,
            "dimension": 3,
            "clock_version": ("continuous_physical_boundaries_v2" if custody or any(
                row.get("held_part") is not None for row in snapshot["resources"].values()) else CLOCK_VERSION),
            "continuous_motion": True,
            "primitive_models": descriptors,
            "owner_effects": effects,
            "certificates": certificates,
            "certificate_fingerprint": fingerprint(certificates),
        },
    }

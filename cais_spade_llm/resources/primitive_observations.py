"""Resource-owned, offline 1D observations of bound primitive programs.

The caller supplies resource-model evidence, not LLM-authored AP valuations.
Poses retain their Cartesian xyz/xyzw interface; ``geometry.axis`` selects the
one modeled translation axis. Distances are metres and all times share the
supplied horizon in seconds. Fixed projected footprints and explicit stationary
intervals are model assumptions, not evidence of actual controller execution.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Any


class _Unavailable(ValueError):
    """Required resource-model evidence is missing or inconsistent."""


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise _Unavailable(f"{name} must be an object")
    return value


def _items(value: Any, name: str) -> list:
    if not isinstance(value, list):
        raise _Unavailable(f"{name} must be a list")
    return value


def _symbol(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _Unavailable(f"{name} must be an explicit identifier")
    return value


def _number(value: Any, name: str) -> Fraction:
    if type(value) not in (int, float):
        raise _Unavailable(f"{name} must be a finite number")
    try:
        finite = isfinite(value)
    except OverflowError as exc:
        raise _Unavailable(f"{name} is outside the supported numeric range") from exc
    if not finite:
        raise _Unavailable(f"{name} must be a finite number")
    # Decimal input arithmetic makes exact contact independent of sampling and
    # floating-point interpolation error at a region boundary.
    return Fraction(str(value))


def _interval(value: Any, name: str) -> tuple[Fraction, Fraction]:
    values = _items(value, name)
    if len(values) != 2:
        raise _Unavailable(f"{name} must contain two bounds")
    low, high = (_number(item, name) for item in values)
    if low > high:
        raise _Unavailable(f"{name} has reversed bounds")
    return low, high


def _observation_boundaries(value: Any, horizon: tuple) -> set[Fraction]:
    if value is None:
        return set()
    times = {_number(item, "observation boundary")
             for item in _items(value, "observation_boundaries")}
    if any(time < horizon[0] or time > horizon[1] for time in times):
        raise _Unavailable("Observation boundaries must lie within the supplied horizon")
    return times


def _pose(value: Any, name: str) -> tuple[Fraction, ...]:
    values = _items(value, name)
    if len(values) != 7:
        raise _Unavailable(f"{name} requires the complete xyz/xyzw pose")
    pose = tuple(_number(item, name) for item in values)
    if not any(pose[3:]):
        raise _Unavailable(f"{name} has no orientation")
    return pose


def _frame(row: dict, frame: str, name: str) -> None:
    if row.get("frame") != frame:
        raise _Unavailable(f"{name} has a missing or incompatible frame")


def _unresolved(value: Any) -> bool:
    if isinstance(value, dict):
        return "context_ref" in value or any(
            isinstance(key, str) and key.startswith("$") or _unresolved(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_unresolved(item) for item in value)
    return False


@dataclass
class _Step:
    resource: str
    primitive: str
    params: dict
    start: Fraction
    end: Fraction
    evidence: dict
    trajectory: list[tuple[Fraction, tuple[Fraction, ...]]]
    source: dict
    trace: dict


def _motion_target(params: dict) -> tuple[Fraction, ...]:
    fields = ("x", "y", "z", "qx", "qy", "qz", "qw")
    if "target" in params:
        if any(field in params for field in fields):
            raise _Unavailable("move_cartesian has ambiguous target interfaces")
        return _pose(params["target"], "move_cartesian.target")
    if not all(field in params for field in fields):
        raise _Unavailable("move_cartesian requires an explicitly bound complete target")
    return _pose([params[field] for field in fields], "move_cartesian target")


def _computed_output(outputs: Any) -> None:
    outputs = _object(outputs, "compute_place_targets.outputs")
    if _unresolved(outputs):
        raise _Unavailable("compute_place_targets output is unresolved")
    if "target" in outputs:
        _pose(outputs["target"], "compute_place_targets.outputs.target")
    elif "target_pose" in outputs:
        target = _object(outputs["target_pose"], "compute_place_targets.outputs.target_pose")
        for coordinate in ("x", "y", "z"):
            _number(target.get(coordinate), f"target_pose.{coordinate}")
        if any(coordinate in target for coordinate in ("qx", "qy", "qz", "qw")):
            _pose([target.get(coordinate) for coordinate in ("x", "y", "z", "qx", "qy", "qz", "qw")],
                  "compute_place_targets.outputs.target_pose")
    else:
        raise _Unavailable("compute_place_targets requires resource-model target output evidence")


def _trajectory(step: _Step, frame: str) -> None:
    _frame(step.evidence, frame, f"{step.resource}/{step.primitive}")
    if "primitive_contract" in step.evidence:
        raise _Unavailable("Owner observation contracts require dimension=3")
    if step.primitive == "compute_place_targets":
        _computed_output(step.evidence.get("outputs"))
        return
    if step.primitive == "release_part":
        if step.params.get("assembly_slot") is not None:
            raise _Unavailable("release_part assembly_slot motion is outside the supplied 1D model")
        _object(step.evidence.get("released_part"), "release_part.released_part")
        return
    if step.primitive != "move_cartesian":
        raise _Unavailable(f"No physical observation model for primitive {step.primitive!r}")
    for raw in _items(step.evidence.get("trajectory"), "move_cartesian.trajectory"):
        row = _object(raw, "trajectory point")
        time = _number(row.get("time"), "trajectory time")
        pose = _pose(row.get("pose"), "trajectory pose")
        if step.trajectory and time <= step.trajectory[-1][0]:
            raise _Unavailable("Trajectory times must strictly increase")
        step.trajectory.append((time, pose))
    if (len(step.trajectory) < 2 or step.trajectory[0][0] != step.start
            or step.trajectory[-1][0] != step.end):
        raise _Unavailable("Trajectory must cover the complete move_cartesian interval")
    if step.trajectory[-1][1] != _motion_target(step.params):
        raise _Unavailable("Motion evidence does not reach the bound move_cartesian target")
    waypoints = step.params.get("waypoints")
    if waypoints is not None:
        pending = [_pose(row, "move_cartesian waypoint")
                   for row in _items(waypoints, "move_cartesian.waypoints")]
        for _, pose in step.trajectory:
            if pending and pending[0] == pose:
                pending.pop(0)
        if pending:
            raise _Unavailable("Motion evidence does not cover the supplied waypoints in order")


def _steps(programs: Any, horizon: tuple, frame: str) -> dict[str, list[_Step]]:
    result: dict[str, list[_Step]] = {}
    for raw in _items(programs, "programs"):
        program = _object(raw, "program")
        if program.get("validation_error") or ("valid" in program and program["valid"] is not True):
            raise _Unavailable("The supplied primitive program is not valid")
        resource = _symbol(program.get("resource_id"), "program.resource_id")
        if resource in result:
            raise _Unavailable("Supply one ordered primitive program per resource")
        authored = _items(program.get("primitive_steps"), "primitive_steps")
        traced = _items(program.get("step_results"), "step_results")
        if not authored or len(authored) != len(traced):
            raise _Unavailable("Every authored primitive needs a matching step_result")
        result[resource] = []
        previous_end = horizon[0]
        for index, (raw_step, raw_trace) in enumerate(zip(authored, traced, strict=True)):
            row, trace = _object(raw_step, "primitive step"), _object(raw_trace, "step_result")
            primitive = _symbol(row.get("primitive"), "primitive")
            params = _object(row.get("params"), "primitive params")
            if (type(trace.get("step_index")) is not int or trace["step_index"] != index
                    or trace.get("primitive") != primitive
                    or trace.get("resolved_params") != params or _unresolved(params)
                    or trace.get("validation_error")
                    or ("valid" in trace and trace["valid"] is not True)):
                raise _Unavailable("step_results do not establish the exact bound primitive steps")
            start = _number(trace.get("start_time"), "start_time")
            end = _number(trace.get("end_time"), "end_time")
            if not previous_end <= start <= end <= horizon[1]:
                raise _Unavailable("Primitive times overlap, reverse order, or leave the horizon")
            previous_end = end
            source = {key: deepcopy(value) for key, value in program.items()
                      if key not in {"primitive_steps", "step_results"}}
            source.update(step_index=index, primitive=primitive, resolved_params=deepcopy(params))
            step = _Step(resource, primitive, params, start, end,
                         _object(trace.get("model_evidence"), "model_evidence"), [], source, trace)
            if "valid" in step.evidence and step.evidence["valid"] is not True:
                raise _Unavailable("The supplied resource-model evidence is not valid")
            _trajectory(step, frame)
            result[resource].append(step)
    return result


def _scope(bindings: Any) -> tuple[set[str], set[str], set[str]]:
    spatial, receiving, regions = set(), set(), set()
    rows = _items(bindings, "bindings")
    if not rows:
        raise _Unavailable("Explicit applicable safety bindings are required")
    for raw in rows:
        row = _object(raw, "binding")
        regions.add(_symbol(row.get("region"), "binding.region"))
        if row.get("specification") == "receiving_region_entry":
            spatial.add(_symbol(row.get("resource"), "binding.resource"))
            receiving.add(_symbol(row.get("receiving_resource"), "binding.receiving_resource"))
            _symbol(row.get("part"), "binding.part")
        elif row.get("specification") == "shared_area_mutex":
            pair = _items(row.get("resources"), "binding.resources")
            if len(pair) != 2 or pair[0] == pair[1]:
                raise _Unavailable("A mutex binding requires two distinct resources")
            spatial.update(_symbol(rid, "mutex resource") for rid in pair)
        else:
            raise _Unavailable("No observation model for the bound specification")
    return spatial, receiving, regions


def _coverage(resource: str, steps: list[_Step], stationary: dict, horizon: tuple) -> None:
    intervals = [(step.start, step.end) for step in steps]
    for raw in _items(stationary.get(resource, []), f"stationary[{resource!r}]"):
        interval = _interval(raw, "stationary interval")
        if interval[0] < horizon[0] or interval[1] > horizon[1]:
            raise _Unavailable("Stationary evidence extends outside the supplied horizon")
        for step in steps:
            if (step.trajectory and max(interval[0], step.start) < min(interval[1], step.end)
                    and any(pose != step.trajectory[0][1] for _, pose in step.trajectory)):
                raise _Unavailable(f"Stationary evidence contradicts motion for {resource!r}")
        intervals.append(interval)
    end = horizon[0]
    for start, stop in sorted(intervals):
        if start > end:
            raise _Unavailable(f"Motion or stationary coverage is missing for {resource!r}")
        end = max(end, stop)
    if not intervals or end < horizon[1]:
        raise _Unavailable(f"Motion or stationary coverage is missing for {resource!r}")


class _Model:
    def __init__(self, snapshot: dict, geometry: dict, horizon: tuple,
                 steps: dict[str, list[_Step]], bindings: list[dict], stationary: dict) -> None:
        self.snapshot = deepcopy(snapshot)
        self.resources = _object(snapshot.get("resources"), "snapshot.resources")
        self.parts = _object(snapshot.get("parts"), "snapshot.parts")
        self.frame = _symbol(geometry.get("frame"), "geometry.frame")
        self.axis = geometry.get("axis")
        if type(self.axis) is not int or self.axis not in (0, 1, 2):
            raise _Unavailable("geometry.axis must explicitly select x, y, or z by index")
        self.horizon, self.steps = horizon, steps
        spatial, receiving, regions = _scope(bindings)
        self.spatial = spatial | set(steps)
        self.receiving = receiving
        self.regions = {}
        for name in regions:
            row = _object(_object(geometry.get("regions"), "geometry.regions").get(name), name)
            _frame(row, self.frame, name)
            self.regions[name] = _interval(row.get("interval"), f"{name}.interval")
        self.poses, self.footprints, self.attachments = {}, {}, {}
        self.part_poses, self.part_footprints = {}, {}
        self.initial_held, self.releases = {}, []
        for resource in sorted(self.spatial):
            self._actor(resource, geometry)
        held = [part for part in self.initial_held.values() if part is not None]
        if len(held) != len(set(held)):
            raise _Unavailable("The same part cannot be held by two resources")
        for resource, program in steps.items():
            self._program(resource, program)
        for row in bindings:
            if row["specification"] == "receiving_region_entry":
                part = row["part"]
                if self.initial_held[row["resource"]] not in (None, part):
                    raise _Unavailable("Incoming-part binding disagrees with resource custody")
                part_row = _object(self.parts.get(part), f"bound part {part!r}")
                part_shape = _object(_object(geometry.get("parts"), "geometry.parts").get(part),
                                     f"geometry for {part!r}")
                _frame(part_shape, self.frame, part)
                _pose(part_row.get("current_pose"), f"{part}.current_pose")
                _interval(part_shape.get("footprint"), f"{part}.footprint")
                if "contained_by" not in part_row:
                    raise _Unavailable(f"Missing containment evidence for {part!r}")
        for resource in sorted(self.spatial | self.receiving):
            _object(self.resources.get(resource), f"resource {resource!r}")
            _coverage(resource, steps.get(resource, []), stationary, horizon)
        self._inventories()

    def _actor(self, resource: str, geometry: dict) -> None:
        row = _object(self.resources.get(resource), f"resource {resource!r}")
        shape = _object(_object(geometry.get("resources"), "geometry.resources").get(resource),
                        f"geometry for {resource!r}")
        _frame(shape, self.frame, resource)
        if "frame" in row:
            _frame(row, self.frame, resource)
        self.poses[resource] = _pose(row.get("current_pose"), f"{resource}.current_pose")
        self.footprints[resource] = _interval(shape.get("footprint"), f"{resource}.footprint")
        if "held_part" not in row:
            raise _Unavailable(f"Missing custody evidence for {resource!r}")
        part = row["held_part"]
        self.initial_held[resource] = part
        if part is None:
            return
        _symbol(part, "held_part")
        self.attachments[resource] = _number(shape.get("attachment_offset"), "attachment_offset")
        part_row = _object(self.parts.get(part), f"part {part!r}")
        part_shape = _object(_object(geometry.get("parts"), "geometry.parts").get(part),
                             f"geometry for {part!r}")
        _frame(part_shape, self.frame, part)
        if "frame" in part_row:
            _frame(part_row, self.frame, part)
        pose = _pose(part_row.get("current_pose"), f"{part}.current_pose")
        if ("contained_by" not in part_row or part_row["contained_by"] is not None
                or pose[self.axis] != self.poses[resource][self.axis] + self.attachments[resource]):
            raise _Unavailable(f"Part custody, position, or attachment is inconsistent for {part!r}")
        self.part_poses[part] = pose
        self.part_footprints[part] = _interval(part_shape.get("footprint"), f"{part}.footprint")

    def _program(self, resource: str, program: list[_Step]) -> None:
        pose = self.poses[resource]
        part = self.initial_held[resource]
        for step in program:
            self._step_snapshot(step, "start_snapshot", pose, part)
            if step.trajectory:
                if step.trajectory[0][1] != pose:
                    raise _Unavailable(f"Motion is discontinuous for {resource!r}")
                for _, point in step.trajectory:
                    if any(point[i] != pose[i] for i in range(7) if i != self.axis):
                        raise _Unavailable("The supplied motion exceeds the fixed-footprint 1D model")
                pose = step.trajectory[-1][1]
            elif step.primitive == "release_part":
                if part is None:
                    raise _Unavailable("release_part has no modeled held part")
                if "part_name" in step.params and step.params["part_name"] != part:
                    raise _Unavailable("release_part parameters disagree with custody")
                evidence = step.evidence["released_part"]
                _frame(evidence, self.frame, "released_part")
                expected = list(self.part_poses[part])
                expected[self.axis] = pose[self.axis] + self.attachments[resource]
                release_pose = _pose(evidence.get("pose"), "released_part.pose")
                if evidence.get("part_name") != part or release_pose != tuple(expected):
                    raise _Unavailable("Release position or part identity disagrees with the attachment")
                if ("contained_by" not in evidence
                        or _number(evidence.get("stationary_until"), "released_part.stationary_until")
                        < self.horizon[1]):
                    raise _Unavailable("Released-part position and containment need full horizon evidence")
                receiver = evidence["contained_by"]
                if receiver is not None:
                    self.receiving.add(_symbol(receiver, "released_part.contained_by"))
                self.releases.append((step.end, resource, part, release_pose, receiver))
                part = None
            self._step_snapshot(step, "projected_snapshot", pose, part)

    def _step_snapshot(self, step: _Step, field: str,
                       pose: tuple[Fraction, ...], part: str | None) -> None:
        if field not in step.trace:
            return
        row = _object(step.trace[field], field)
        if "frame" in row:
            _frame(row, self.frame, field)
        if "current_pose" in row and _pose(row["current_pose"], field) != pose:
            raise _Unavailable(f"{field}.current_pose contradicts the modeled primitive")
        if "held_part" in row and row["held_part"] != part:
            raise _Unavailable(f"{field}.held_part contradicts the modeled primitive")

    def _inventories(self) -> None:
        inventories: dict[str, str] = {}
        for resource in self.receiving:
            values = _items(self.resources[resource].get("contained_parts"),
                            f"{resource}.contained_parts")
            for part in values:
                _symbol(part, "contained part")
                row = _object(self.parts.get(part), f"contained part {part!r}")
                if part in inventories or row.get("contained_by") != resource:
                    raise _Unavailable("Contained-part inventories disagree with part evidence")
                inventories[part] = resource
        for part, row in self.parts.items():
            _symbol(part, "part identifier")
            row = _object(row, f"part {part!r}")
            receiver = row.get("contained_by")
            if receiver is not None:
                _symbol(receiver, "contained_by")
            if receiver in self.receiving and inventories.get(part) != receiver:
                raise _Unavailable("Contained-part inventory is incomplete")

    def _pose_at(self, resource: str, time: Fraction) -> tuple[Fraction, ...]:
        pose = self.poses[resource]
        for step in self.steps.get(resource, []):
            if not step.trajectory or time < step.start:
                continue
            for (start, first), (end, last) in zip(step.trajectory, step.trajectory[1:], strict=False):
                if start <= time <= end:
                    ratio = (time - start) / (end - start)
                    return tuple(a + ratio * (b - a) for a, b in zip(first, last, strict=True))
            pose = step.trajectory[-1][1]
        return pose

    def times(self) -> list[Fraction]:
        times = set(self.horizon)
        for resource, program in self.steps.items():
            offsets = [self.footprints[resource]]
            part = self.initial_held[resource]
            if part is not None:
                offsets.append(tuple(value + self.attachments[resource]
                                     for value in self.part_footprints[part]))
            for step in program:
                times.update((step.start, step.end))
                times.update(time for time, _ in step.trajectory)
                for (start, first), (end, last) in zip(step.trajectory, step.trajectory[1:], strict=False):
                    x0, x1 = first[self.axis], last[self.axis]
                    if x0 == x1:
                        continue
                    for low, high in self.regions.values():
                        for left, right in offsets:
                            for boundary in (low - right, high - left):
                                ratio = (boundary - x0) / (x1 - x0)
                                if 0 <= ratio <= 1:
                                    times.add(start + ratio * (end - start))
        return sorted(times)

    def observe(self, time: Fraction) -> dict:
        state = deepcopy(self.snapshot)
        resources, parts = state["resources"], state["parts"]
        poses = {resource: self._pose_at(resource, time) for resource in self.spatial}
        carried = {resource: ([] if part is None else [part])
                   for resource, part in self.initial_held.items()}
        for resource, pose in poses.items():
            resources[resource]["current_pose"] = [float(value) for value in pose]
            part = self.initial_held[resource]
            if part is not None:
                part_pose = list(self.part_poses[part])
                part_pose[self.axis] = pose[self.axis] + self.attachments[resource]
                parts[part]["current_pose"] = [float(value) for value in part_pose]
        for released, resource, part, pose, receiver in self.releases:
            if released > time:
                continue
            resources[resource]["held_part"] = None
            parts[part].update(current_pose=[float(value) for value in pose], contained_by=receiver)
            if receiver is not None:
                resources[receiver]["contained_parts"].append(part)
            if released < time:
                carried[resource] = []
        occupancy = {}
        for region, (low, high) in self.regions.items():
            occupancy[region] = {}
            for resource, pose in poses.items():
                offsets = [self.footprints[resource]]
                offsets.extend(tuple(value + self.attachments[resource]
                                     for value in self.part_footprints[part])
                               for part in carried[resource])
                occupancy[region][resource] = any(
                    pose[self.axis] + right >= low and pose[self.axis] + left <= high
                    for left, right in offsets
                )
        active_steps = [deepcopy(step.source) for program in self.steps.values() for step in program
                        if step.start <= time <= step.end]
        return {"time": float(time), "time_exact": str(time), "resources": resources,
                "parts": parts, "region_occupancy": occupancy, "carried_parts": carried,
                "active_steps": active_steps}


def model_primitive_observations(
    *, programs: list[dict], snapshot: dict, geometry: dict, horizon: list[float],
    stationary: dict, bindings: list[dict], observation_boundaries: list[float] | None = None,
    primitive_models: dict | None = None,
    motion_budget=None,
) -> dict[str, Any]:
    """Observe bound programs under an explicitly supplied offline 1D model.

    Args:
        programs: One program per moving resource, containing ``primitive_steps``
            and matching ``step_results`` with resolved parameters, times, and
            resource-owned ``model_evidence``. No primitive is executed.
        snapshot: Resource poses/custody/inventories and part poses/containment.
        geometry: Exact frame, axis, region intervals and projected footprints.
        horizon: Common start/end times in seconds for every participant.
        stationary: Explicit intervals covering unchanged modeled resource state
            outside authored steps. Modeled incoming releases update inventories.
        bindings: Concrete receiving-region entry or shared-area mutex bindings.
        observation_boundaries: Additional exact decimal times on the joint
            evidence clock, included before interval midpoints are generated.
        primitive_models: Optional pure owner models for the supported 3D path.

    Returns:
        Complete joint observations and a post-program snapshot, or
        ``NEEDS_CONTEXT``. Inputs are never changed. At a release instant, the
        closed robot envelope includes the just-released part and inventories
        include its supplied containment, covering simultaneous entry/release.
    """
    if isinstance(geometry, dict) and geometry.get("dimension") == 3:
        from cais_spade_llm.resources.continuous_observations import (
            has_continuous_motion, model_continuous_observations,
        )
        if has_continuous_motion(programs):
            return model_continuous_observations(
                programs=programs, snapshot=snapshot, geometry=geometry, horizon=horizon,
                stationary=stationary, observation_boundaries=observation_boundaries,
                primitive_models=primitive_models, budget=motion_budget)
        from cais_spade_llm.resources.primitive_observations_3d import (
            model_primitive_observations_3d,
        )

        return model_primitive_observations_3d(
            programs=programs, snapshot=snapshot, geometry=geometry,
            horizon=horizon, stationary=stationary,
            observation_boundaries=observation_boundaries,
            primitive_models=primitive_models,
        )
    try:
        if primitive_models is not None:
            raise _Unavailable("Owner models require the supported 3D observation contract")
        snapshot = _object(snapshot, "snapshot")
        geometry = _object(geometry, "geometry")
        stationary = _object(stationary, "stationary")
        interval = _interval(horizon, "horizon")
        boundaries = _observation_boundaries(observation_boundaries, interval)
        frame = _symbol(geometry.get("frame"), "geometry.frame")
        steps = _steps(programs, interval, frame)
        model = _Model(snapshot, geometry, interval, steps, bindings, stationary)
        critical = sorted(set(model.times()) | boundaries)
        times = sorted(set(critical) | {
            (a + b) / 2 for a, b in zip(critical, critical[1:], strict=False)
        })
        observations = [model.observe(time) for time in times]
        # Every interval between critical instants is represented by its midpoint;
        # state and AP truth are constant there under the declared linear model.
        for row, time in zip(observations, times, strict=True):
            row["phase"] = "at" if time in critical else "between"
        final = deepcopy(snapshot)
        final.update(resources=deepcopy(observations[-1]["resources"]),
                     parts=deepcopy(observations[-1]["parts"]))
        return {"valid": True, "feasibility_status": "FEASIBLE", "reason": "",
                "observations": observations, "projected_snapshot": final,
                "evidence": {"offline": True, "dimension": 1, "frame": frame,
                             "axis": model.axis, "closed_intervals": True,
                             "horizon": deepcopy(horizon),
                             **({"observation_boundaries": [str(time) for time in sorted(boundaries)]}
                                if observation_boundaries is not None else {})}}
    except _Unavailable as exc:
        return {"valid": False, "feasibility_status": "NEEDS_CONTEXT", "reason": str(exc),
                "observations": [], "projected_snapshot": deepcopy(snapshot),
                "evidence": {"offline": True, "dimension": 1}}

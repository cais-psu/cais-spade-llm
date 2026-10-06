"""Offline fixed-orientation 3D primitive observations, independent of safety rules.

The supplied trajectories, fixed world-axis envelopes and custody evidence are
model assumptions. No robot executes and no AP or admission decision is produced.
All motion is piecewise linear; exact decimal arithmetic determines the shared
boundary clock. Orientation changes and unmodeled primitive effects fail closed.
"""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from fractions import Fraction
from typing import Any

from cais_spade_llm.resources.primitive_observations import (
    _coverage,
    _frame,
    _interval,
    _items,
    _motion_target,
    _number,
    _object,
    _observation_boundaries,
    _pose,
    _symbol,
    _Unavailable,
    _unresolved,
)
from cais_spade_llm.resources.resource_safety_preparation import (
    legacy_primitive_model,
    primitive_model_descriptors,
)


def _bounds(value: Any, name: str) -> tuple:
    rows = _items(value, name)
    if len(rows) != 3:
        raise _Unavailable(f"{name} requires three world-axis intervals")
    return tuple(_interval(row, name) for row in rows)


def _base_pose(value: Any, name: str) -> tuple:
    values = _items(value, name)
    if len(values) != 3:
        raise _Unavailable(f"{name} requires x, y and yaw")
    return tuple(_number(item, name) for item in values)


def _unit_pose(value: Any, name: str) -> tuple:
    pose = _pose(value, name)
    if abs(sum(item * item for item in pose[3:]) - 1) > Fraction(1, 10**9):
        raise _Unavailable(f"{name} requires a unit quaternion")
    return pose


def _quaternion_product(a: tuple, b: tuple) -> tuple:
    x, y, z, w = a
    u, v, s, t = b
    return (w*u + x*t + y*s - z*v, w*v - x*s + y*t + z*u,
            w*s + x*v - y*u + z*t, w*t - x*u - y*v - z*s)


def _compose(pose: tuple, transform: tuple) -> tuple:
    q = pose[3:]
    norm = sum(value * value for value in q)
    inverse = tuple(-value / norm for value in q[:3]) + (q[3] / norm,)
    offset = _quaternion_product(_quaternion_product(q, transform[:3] + (Fraction(0),)), inverse)
    return tuple(pose[i] + offset[i] for i in range(3)) + _quaternion_product(q, transform[3:])


def _same_pose(first: tuple, second: tuple, name: str) -> None:
    # Position equality is exact. Quaternion representation is retained unchanged;
    # tolerance only accommodates declared decimal quaternion multiplication.
    if first[:3] != second[:3] or any(abs(a - b) > Fraction(1, 10**9)
                                    for a, b in zip(first[3:], second[3:], strict=True)):
        raise _Unavailable(f"{name} contradicts the modeled pose")


@dataclass
class _Step:
    resource: str
    primitive: str
    params: dict
    start: Fraction
    end: Fraction
    evidence: dict
    source: dict
    trace: dict
    trajectory: list = dataclass_field(default_factory=list)
    base_trajectory: list = dataclass_field(default_factory=list)
    part_trajectories: dict = dataclass_field(default_factory=dict)
    effects: dict | None = None
    model_descriptor: dict | None = None


def _points(value: Any, step: _Step, name: str, *, base: bool = False) -> list:
    result = []
    parser = _base_pose if base else _unit_pose
    for raw in _items(value, name):
        row = _object(raw, name)
        time = _number(row.get("time"), name + ".time")
        pose = parser(row.get("pose"), name + ".pose")
        if result and time <= result[-1][0]:
            raise _Unavailable(f"{name} times must strictly increase")
        if result and pose[2 if base else 3:] != result[0][1][2 if base else 3:]:
            raise _Unavailable(f"{name} changes orientation")
        result.append((time, pose))
    if len(result) < 2 or result[0][0] != step.start or result[-1][0] != step.end:
        raise _Unavailable(f"{name} must cover the complete primitive interval")
    return result


def _waypoints(points: list, values: Any, name: str, *, base: bool = False) -> None:
    parser = _base_pose if base else _unit_pose
    pending = [parser(row, name) for row in _items(values, name)]
    for _, pose in points:
        if pending and pending[0] == pose:
            pending.pop(0)
    if pending:
        raise _Unavailable(f"{name} is not covered in order by the trajectory")


def _parameters(step: _Step) -> None:
    allowed = {
        "move_cartesian": {"target", "waypoints", "seed"},
        "move_to_named_pose": {"pose_name"},
        "move_base": {"target_pose", "waypoints", "transform"},
        "compute_pick_targets": {"initial"},
        "grasp_part": {"part_name", "initial"},
        "release_part": {"transform"},
    }
    if step.primitive not in allowed or set(step.params) - allowed[step.primitive]:
        raise _Unavailable(f"Unsupported parameters or primitive {step.primitive!r}")
    for key in ("target", "initial", "transform"):
        if key in step.params and step.params[key] is not None:
            _unit_pose(step.params[key], key)
    if step.params.get("waypoints") is not None:
        _items(step.params["waypoints"], "waypoints")
    if step.params.get("seed") is not None:
        seed = _items(step.params["seed"], "seed")
        if len(seed) != 7:
            raise _Unavailable("move_cartesian seed requires seven joint values")
        for value in seed:
            _number(value, "seed")


def _pick_outputs(evidence: dict) -> None:
    outputs = _object(evidence.get("outputs"), "compute_pick_targets.outputs")
    for name in ("target", "approach", "lift"):
        _unit_pose(outputs.get(name), "compute_pick_targets.outputs." + name)
    if "retreat" in outputs:
        _unit_pose(outputs["retreat"], "compute_pick_targets.outputs.retreat")
    for pose in _items(outputs.get("lift_waypoints"), "compute_pick_targets.outputs.lift_waypoints"):
        _unit_pose(pose, "lift_waypoints")
    for name in ("seed", "carrying"):
        if name == "carrying" and name not in outputs:
            continue
        values = _items(outputs.get(name), "compute_pick_targets.outputs." + name)
        if len(values) != 7:
            raise _Unavailable(f"compute_pick_targets.{name} requires seven joint values")
        for value in values:
            _number(value, name)


def _base_evidence(step: _Step) -> None:
    step.base_trajectory = _points(step.evidence.get("base_trajectory"), step,
                                   "base_trajectory", base=True)
    if [time for time, _ in step.trajectory] != [time for time, _ in step.base_trajectory]:
        raise _Unavailable("move_base TCP and base trajectories need identical times")
    if step.base_trajectory[-1][1] != _base_pose(step.params.get("target_pose"), "target_pose"):
        raise _Unavailable("move_base evidence does not reach target_pose")
    _waypoints(step.base_trajectory, step.params.get("waypoints") or [], "move_base.waypoints", base=True)
    first_tcp, first_base = step.trajectory[0][1], step.base_trajectory[0][1]
    for (_, tcp), (_, base) in zip(step.trajectory, step.base_trajectory, strict=True):
        if (tcp[2] != first_tcp[2] or any(tcp[i] - first_tcp[i] != base[i] - first_base[i]
                                         for i in (0, 1))):
            raise _Unavailable("Empty fixed-yaw base motion must translate TCP rigidly")


def _named_evidence(step: _Step) -> None:
    _symbol(step.params.get("pose_name"), "move_to_named_pose.pose_name")
    if step.params["pose_name"] != "transport":
        raise _Unavailable("Only transport is modeled; home has unmodeled gripper effects")
    outputs = _object(step.evidence.get("outputs"), "move_to_named_pose.outputs")
    if outputs.get("pose_name") != step.params["pose_name"]:
        raise _Unavailable("Named-pose evidence does not bind the exact pose_name")
    if step.trajectory[-1][1] != _unit_pose(outputs.get("tcp_pose"), "outputs.tcp_pose"):
        raise _Unavailable("Named-pose output and trajectory disagree")


def _owner_evidence(step: _Step, model) -> None:
    step.model_descriptor = model.descriptor()
    step.effects = model.effects(primitive=step.primitive, params=step.params, evidence=step.evidence,
                                 start_time=step.start, end_time=step.end)
    if step.model_descriptor != model.descriptor():
        raise _Unavailable("Primitive model configuration changed during evaluation")
    if step.effects["trajectory"] is not None:
        step.trajectory = _points(step.effects["trajectory"], step, "trajectory")
    if step.effects["base_trajectory"] is not None:
        step.base_trajectory = _points(step.effects["base_trajectory"], step, "base_trajectory", base=True)
    for part, points in _object(step.effects["part_trajectories"], "part_trajectories").items():
        _symbol(part, "part_trajectories.part")
        step.part_trajectories[part] = _points(points, step, "part_trajectories." + part)
    _items(step.effects["transfers"], "transfers")
    _object(step.effects["resource_updates"], "resource_updates")
    if step.start == step.end and (step.trajectory or step.base_trajectory or step.part_trajectories or step.effects["transfers"]):
        raise _Unavailable("Physical effects require a positive primitive interval")


def _evidence(step: _Step, frame: str, primitive_models: dict | None) -> None:
    evidence = step.evidence
    _frame(evidence, frame, "model_evidence")
    if evidence.get("validation_error") or ("valid" in evidence and evidence["valid"] is not True):
        raise _Unavailable("The supplied resource-model evidence is not valid")
    if _unresolved(evidence):
        raise _Unavailable("Resource-model evidence is unresolved")
    if primitive_models is None:
        model = legacy_primitive_model(evidence)
    else:
        model = primitive_models.get(step.resource)
        if model is None:
            raise _Unavailable("Missing primitive model for " + step.resource)
    if model is not None:
        _owner_evidence(step, model)
        return
    _legacy_evidence(step, frame)


def _legacy_evidence(step: _Step, frame: str) -> None:
    evidence = step.evidence
    _parameters(step)
    if step.primitive in {"compute_pick_targets", "grasp_part", "release_part"} and any(
            key in evidence for key in ("trajectory", "base_trajectory")):
        raise _Unavailable("A no-motion primitive has contradictory trajectory evidence")
    if step.primitive != "move_base" and "base_trajectory" in evidence:
        raise _Unavailable("Only move_base may change the base pose")
    if step.primitive == "compute_pick_targets":
        _pick_outputs(evidence)
        return
    if step.primitive in {"grasp_part", "release_part"}:
        if step.start == step.end:
            raise _Unavailable("Custody primitives require a positive duration")
        if step.params.get("assembly_slot") is not None:
            raise _Unavailable("release_part assembly_slot motion is unsupported")
        name = "grasped_part" if step.primitive == "grasp_part" else "released_part"
        row = _object(evidence.get(name), name)
        _frame(row, frame, name)
        _symbol(row.get("part_name"), name + ".part_name")
        _unit_pose(row.get("pose"), name + ".pose")
        return
    if step.primitive not in {"move_cartesian", "move_to_named_pose", "move_base"}:
        raise _Unavailable(f"No 3D observation model for primitive {step.primitive!r}")
    step.trajectory = _points(evidence.get("trajectory"), step, "trajectory")
    if step.primitive == "move_cartesian":
        target = _motion_target(step.params)
        if step.trajectory[-1][1] != target:
            raise _Unavailable("Motion does not reach the bound move_cartesian target")
        _waypoints(step.trajectory, step.params.get("waypoints") or [], "move_cartesian.waypoints")
    elif step.primitive == "move_to_named_pose":
        _named_evidence(step)
    else:
        _base_evidence(step)


def _steps(programs: Any, horizon: tuple, frame: str, primitive_models: dict | None = None) -> dict:
    result = {}
    for raw in _items(programs, "programs"):
        program = _object(raw, "program")
        resource = _symbol(program.get("resource_id"), "program.resource_id")
        if resource in result:
            raise _Unavailable("Supply one ordered primitive program per resource")
        if program.get("validation_error") or ("valid" in program and program["valid"] is not True):
            raise _Unavailable("The supplied primitive program is not valid")
        authored = _items(program.get("primitive_steps"), "primitive_steps")
        traces = _items(program.get("step_results"), "step_results")
        if not authored or len(authored) != len(traces):
            raise _Unavailable("Every authored primitive needs a matching step_result")
        result[resource] = []
        previous_end = horizon[0]
        for index, (raw_step, raw_trace) in enumerate(zip(authored, traces, strict=True)):
            row, trace = _object(raw_step, "primitive_step"), _object(raw_trace, "step_result")
            primitive = _symbol(row.get("primitive"), "primitive")
            params = _object(row.get("params"), "params")
            if (type(trace.get("step_index")) is not int or trace["step_index"] != index
                    or trace.get("primitive") != primitive or trace.get("resolved_params") != params
                    or _unresolved(params) or trace.get("validation_error")
                    or ("valid" in trace and trace["valid"] is not True)):
                raise _Unavailable("step_results do not establish the exact bound primitive steps")
            start, end = (_number(trace.get(key), key) for key in ("start_time", "end_time"))
            if not previous_end <= start <= end <= horizon[1]:
                raise _Unavailable("Primitive times overlap, reverse order, or leave the horizon")
            previous_end = end
            metadata = deepcopy(_object(trace.get("source", row.get("source", {})), "source"))
            source = {**metadata, "source": metadata, "resource_id": resource,
                      "step_index": index, "global_step_index": index,
                      "primitive": primitive, "resolved_params": deepcopy(params)}
            step = _Step(resource, primitive, params, start, end,
                         _object(trace.get("model_evidence"), "model_evidence"), source, trace)
            _evidence(step, frame, primitive_models)
            for param, reference in _object(step.evidence.get("param_sources", {}), "param_sources").items():
                reference = _object(reference, "param_source")
                prior = reference.get("step_index")
                if (set(reference) != {"step_index", "output"} or type(prior) is not int
                        or not 0 <= prior < index or param not in params):
                    raise _Unavailable("Invalid helper output source")
                helper = result[resource][prior]
                outputs = helper.evidence.get("outputs", {})
                if ((helper.effects is None and helper.primitive != "compute_pick_targets") or reference["output"] not in outputs
                        or params[param] != outputs[reference["output"]]):
                    raise _Unavailable("Primitive parameter contradicts its helper output")
            result[resource].append(step)
    return result


def _at(points: list, time: Fraction, initial: tuple) -> tuple:
    pose = initial
    for (start, first), (end, last) in zip(points, points[1:], strict=False):
        if time < start:
            break
        if time <= end:
            ratio = (time - start) / (end - start)
            return tuple(a + ratio * (b - a) for a, b in zip(first, last, strict=True))
        pose = last
    return pose


def _overlaps(pose: tuple, footprint: tuple, region: tuple) -> bool:
    return all(pose[axis] + high >= region[axis][0]
               and pose[axis] + low <= region[axis][1]
               for axis, (low, high) in enumerate(footprint))


def _plain(value: Any) -> Any:
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


class _Model:
    def __init__(self, snapshot: dict, geometry: dict, horizon: tuple,
                 programs: Any, stationary: dict, primitive_models: dict | None = None) -> None:
        self.frame = _symbol(geometry.get("frame"), "geometry.frame")
        if type(geometry.get("dimension")) is not int or geometry["dimension"] != 3 or "axis" in geometry:
            raise _Unavailable("3D geometry requires dimension=3 and no legacy axis")
        self.horizon = horizon
        self.snapshot = deepcopy(snapshot)
        self.resources = _object(self.snapshot.get("resources"), "snapshot.resources")
        self.parts = _object(self.snapshot.get("parts"), "snapshot.parts")
        if set(geometry) - {"dimension", "frame", "regions", "resources", "parts"}:
            raise _Unavailable("Unsupported 3D geometry fields")
        self.stationary_only = set()
        self.regions = self._shapes(geometry.get("regions"), "bounds")
        self.footprints = self._shapes(geometry.get("resources"), "footprint", base=True)
        self.part_footprints = self._shapes(geometry.get("parts"), "footprint")
        if not self.regions or not self.footprints:
            raise _Unavailable("3D geometry needs explicit regions and spatial resources")
        if set(self.part_footprints) != set(self.parts):
            raise _Unavailable("Every declared part requires snapshot and geometry evidence")
        self.base_footprints = {}
        self.model_descriptors = primitive_model_descriptors(primitive_models)
        if set(self.model_descriptors) - set(self.resources):
            raise _Unavailable("Primitive model refers to an unconfigured resource")
        self.steps = _steps(programs, horizon, self.frame, primitive_models)
        for resource, steps in self.steps.items():
            for step in steps:
                if step.model_descriptor is not None:
                    if resource in self.model_descriptors and self.model_descriptors[resource] != step.model_descriptor:
                        raise _Unavailable("Resource program mixes incompatible observation contracts")
                    self.model_descriptors[resource] = step.model_descriptor
        self.owned = set(self.model_descriptors)
        if set(self.steps) - set(self.footprints) or set(stationary) - set(self.resources):
            raise _Unavailable("Programs or stationary coverage refer to an undeclared resource")
        self._initial(geometry, stationary)
        self.flat = [step for name in sorted(self.steps) for step in self.steps[name]]
        self.initial = deepcopy(self.snapshot)
        self._motion_continuity()
        self._timeline()
        self._stationary_parts()

    def _shapes(self, value: Any, field_name: str, *, base: bool = False) -> dict:
        result = {}
        for name, raw in _object(value, "geometry").items():
            _symbol(name, "geometry identifier")
            row = _object(raw, name)
            _frame(row, self.frame, name)
            allowed = {"frame", field_name}
            if base:
                allowed.update({"base_footprint", "stationary_only"})
                if "stationary_only" in row and type(row["stationary_only"]) is not bool:
                    raise _Unavailable("stationary_only must be Boolean")
                if row.get("stationary_only") is True:
                    if "base_footprint" in row:
                        raise _Unavailable("A stationary_only resource cannot declare base geometry")
                    self.stationary_only.add(name)
            if set(row) - allowed:
                raise _Unavailable("Unsupported or legacy fields in 3D geometry")
            result[name] = _bounds(row.get(field_name), name + "." + field_name)
        return result

    def _initial_parts(self) -> None:
        for name, row in self.parts.items():
            _object(row, "part")
            row["current_pose"] = _unit_pose(row.get("current_pose"), name + ".current_pose")
            if "frame" in row:
                _frame(row, self.frame, name)
            if "contained_by" not in row:
                raise _Unavailable("Every part needs explicit containment")
            if row["contained_by"] is not None:
                _symbol(row["contained_by"], "contained_by")
            if "processCompleted" in row:
                _items(row["processCompleted"], "processCompleted")

    def _initial(self, geometry: dict, stationary: dict) -> None:  # noqa: C901 - complete joint checkpoint
        self._initial_parts()
        for resource, row in self.resources.items():
            _symbol(resource, "resource")
            _object(row, "resource snapshot")
            _coverage(resource, self.steps.get(resource, []), stationary, self.horizon)
            if resource not in self.footprints:
                if "current_pose" in row or "base_pose" in row or row.get("held_part") is not None:
                    raise _Unavailable("A spatial resource is missing configured 3D geometry")
                continue
            row["current_pose"] = _unit_pose(row.get("current_pose"), resource + ".current_pose")
            if "frame" in row:
                _frame(row, self.frame, resource)
            if resource in self.stationary_only:
                if (resource in self.steps and resource not in self.owned) or set(row) & {
                    "base_pose", "held_part", "gripper_state", "grasp_transform",
                }:
                    raise _Unavailable("A stationary_only resource cannot declare programs, base or gripper custody")
                continue
            if resource in self.owned and "held_part" not in row:
                if set(row) & {"gripper_state", "grasp_transform"}:
                    raise _Unavailable("Gripper evidence has no declared custody")
                if "base_pose" in row or "base_footprint" in geometry["resources"][resource]:
                    raise _Unavailable("This owner model requires explicit supported base/custody evidence")
                continue
            if "held_part" not in row or row.get("gripper_state") not in {"open", "closed"}:
                raise _Unavailable("A spatial resource needs custody and gripper evidence")
            if "base_pose" in row:
                row["base_pose"] = _base_pose(row["base_pose"], "base_pose")
                self.base_footprints[resource] = _bounds(
                    geometry["resources"][resource].get("base_footprint"), "base_footprint")
            elif "base_footprint" in geometry["resources"][resource]:
                raise _Unavailable("base_footprint requires a base_pose")
            if row["held_part"] is not None:
                part = _symbol(row["held_part"], "held_part")
                if part not in self.parts or row["gripper_state"] != "closed":
                    raise _Unavailable("Held-part or gripper evidence is inconsistent")
                row["grasp_transform"] = _unit_pose(row.get("grasp_transform"), "grasp_transform")
                _same_pose(_compose(row["current_pose"], row["grasp_transform"]),
                           self.parts[part]["current_pose"], "Initial grasp transform")
            elif row.get("grasp_transform") is not None:
                raise _Unavailable("An empty resource cannot retain a grasp_transform")
        if set(self.footprints) - set(self.resources):
            raise _Unavailable("Configured spatial resource is missing its snapshot")
        self._inventories(self.snapshot)

    @staticmethod
    def _inventories(state: dict) -> None:
        owners = {}
        contained = {}
        for resource, row in state["resources"].items():
            part = row.get("held_part")
            if part is not None:
                if part in owners or part not in state["parts"]:
                    raise _Unavailable("The same part cannot be held by two resources")
                owners[part] = resource
            if "contained_parts" in row:
                for item in _items(row["contained_parts"], "contained_parts"):
                    _symbol(item, "contained part")
                    if item in contained or item not in state["parts"]:
                        raise _Unavailable("Contained-part inventory is inconsistent")
                    contained[item] = resource
        for part, row in state["parts"].items():
            receiver = row["contained_by"]
            if receiver != contained.get(part) or (part in owners and receiver is not None):
                raise _Unavailable("Custody and complete containment evidence disagree")

    def _motion_continuity(self) -> None:
        for resource, steps in self.steps.items():
            pose = self.resources[resource]["current_pose"]
            base = self.resources[resource].get("base_pose")
            for step in steps:
                if step.trajectory:
                    if step.trajectory[0][1] != pose or step.trajectory[-1][1][3:] != pose[3:]:
                        raise _Unavailable("TCP trajectory is discontinuous or changes orientation")
                    pose = step.trajectory[-1][1]
                    if resource in self.stationary_only and any(point != self.resources[resource]["current_pose"] for _, point in step.trajectory):
                        raise _Unavailable("Stationary resource geometry contradicts its motion")
                if step.base_trajectory:
                    if base is None or step.base_trajectory[0][1] != base:
                        raise _Unavailable("Base trajectory is discontinuous or has no initial evidence")
                    base = step.base_trajectory[-1][1]

    def _resource_pose(self, resource: str, time: Fraction, *, base: bool = False) -> tuple:
        pose = self.initial["resources"][resource]["base_pose" if base else "current_pose"]
        for step in self.steps.get(resource, []):
            points = step.base_trajectory if base else step.trajectory
            if points and time >= step.start:
                pose = _at(points, time, pose)
        return pose

    def _refresh(self, state: dict, time: Fraction) -> None:
        for resource in self.footprints:
            row = state["resources"][resource]
            row["current_pose"] = self._resource_pose(resource, time)
            if resource in self.base_footprints:
                row["base_pose"] = self._resource_pose(resource, time, base=True)
            part = row.get("held_part")
            if part is not None:
                pose = _compose(row["current_pose"], row["grasp_transform"])
                # A constant orientation retains the supplied part quaternion.
                state["parts"][part]["current_pose"] = pose[:3] + state["parts"][part]["current_pose"][3:]
        for step in self.flat:
            if step.start <= time <= step.end:
                for part, points in step.part_trajectories.items():
                    if part not in state["parts"]:
                        raise _Unavailable("Part trajectory has no configured part")
                    state["parts"][part]["current_pose"] = _at(points, time, points[0][1])

    def _part_snapshot(self, name: str, row: dict, state: dict, field_name: str) -> None:
        row = _object(row, "part snapshot")
        actual = _object(state["parts"].get(name), "snapshot part")
        for key in ("contained_by", "processCompleted", "processCompleted_complete", "processCompleted_evidence"):
            if key in row and row[key] != actual.get(key):
                raise _Unavailable(f"{field_name}.parts contradicts {key}")
        for key in ("current_pose", "pose"):
            if key in row:
                _same_pose(_unit_pose(row[key], "part." + key), actual["current_pose"], field_name)
        if "frame" in row:
            _frame(row, self.frame, field_name)
        if "location" in row:
            owners = [owner for owner in state["resources"].values() if owner.get("held_part") == name]
            location = owners[0].get("resource_jid") if owners else actual["contained_by"]
            if (owners and location is None) or row["location"] != location:
                raise _Unavailable(f"{field_name}.part_tracker contradicts custody or containment")

    def _resource_snapshot(self, supplied: dict, expected: dict, field_name: str,
                           *, stationary_only: bool = False, owner_model: bool = False) -> None:
        allowed = {"frame", "current_pose", "base_pose", "held_part", "gripper_state",
                   "grasp_transform", "contained_parts", "current_state", "resource_location"}
        native = set(expected) - allowed if owner_model else set()
        if set(supplied) - allowed - native:
            raise _Unavailable(f"{field_name} contains an unsupported resource postcondition")
        if stationary_only and set(supplied) & {
            "base_pose", "held_part", "gripper_state", "grasp_transform",
        }:
            raise _Unavailable(f"{field_name} declares base or gripper custody for a stationary_only resource")
        if "frame" in supplied:
            _frame(supplied, self.frame, field_name)
        if any(supplied[key] != expected[key] for key in native & set(supplied)):
            raise _Unavailable("Resource snapshot contradicts its declared owner effects")
        for key in ("held_part", "gripper_state", "contained_parts"):
            if key in supplied and supplied[key] != expected.get(key):
                raise _Unavailable(f"{field_name}.{key} contradicts the modeled primitive")
        for key, parser in (("current_pose", _unit_pose), ("base_pose", _base_pose),
                            ("grasp_transform", _unit_pose)):
            if key not in supplied:
                continue
            value = supplied[key]
            if value is None:
                if expected.get(key) is not None:
                    raise _Unavailable(f"{field_name}.{key} contradicts the modeled primitive")
            elif parser(value, key) != expected.get(key):
                raise _Unavailable(f"{field_name}.{key} contradicts the modeled primitive")

    def _check_snapshot(self, step: _Step, field_name: str, state: dict) -> None:
        if field_name not in step.trace:
            return
        supplied = _object(step.trace[field_name], field_name)
        for key in ("parts", "part_tracker"):
            for name, row in _object(supplied.get(key, {}), "snapshot." + key).items():
                self._part_snapshot(name, row, state, field_name)
        for name, row in _object(supplied.get("resources", {}), "snapshot.resources").items():
            expected = _object(state["resources"].get(name), "snapshot resource")
            self._resource_snapshot(_object(row, "resource snapshot"), expected, field_name,
                                    stationary_only=name in self.stationary_only, owner_model=name in self.owned)
        resource_fields = {key: value for key, value in supplied.items()
                           if key not in {"resources", "parts", "part_tracker"}}
        self._resource_snapshot(resource_fields, state["resources"][step.resource], field_name,
                                owner_model=step.effects is not None)

    def _start(self, step: _Step, state: dict) -> None:
        self._check_snapshot(step, "start_snapshot", state)
        row = state["resources"][step.resource]
        if step.effects is not None:
            for part, points in step.part_trajectories.items():
                initial = self._state_before_motion(step, part)
                if (part not in state["parts"] or state["parts"][part]["contained_by"] != step.resource
                        or any(owner.get("held_part") == part for owner in state["resources"].values())):
                    raise _Unavailable("Part motion is not owned by its declared containment resource")
                _same_pose(points[0][1], initial, "Part motion start")
            return
        if step.primitive == "move_base" and (row["held_part"] is not None or step.params.get("transform") is not None):
            raise _Unavailable("Only empty fixed-yaw move_base is supported")
        if step.primitive == "release_part" and row["held_part"] is None:
            raise _Unavailable("release_part starts without a held part")
        if step.primitive == "grasp_part":
            name = step.evidence["grasped_part"]["part_name"]
            if row["held_part"] is not None or any(r.get("held_part") == name for r in state["resources"].values()):
                raise _Unavailable("grasp_part starts with unavailable custody")
            if step.params.get("part_name") != name or name not in state["parts"]:
                raise _Unavailable("grasp_part identity disagrees with the bound primitive")
            _same_pose(_unit_pose(step.params.get("initial"), "grasp_part.initial"),
                       state["parts"][name]["current_pose"], "grasp_part.initial")
        if step.primitive == "compute_pick_targets":
            initial = _unit_pose(step.params.get("initial"), "compute_pick_targets.initial")
            names = [name for name, part in state["parts"].items() if part["current_pose"] == initial]
            source_part = step.source.get("part_name") or step.evidence.get("part_name")
            if source_part is not None:
                names = [name for name in names if name == source_part]
            if len(names) != 1 or any(r.get("held_part") == names[0] for r in state["resources"].values()):
                raise _Unavailable("compute_pick_targets initial pose lacks an available exact part")

    def _end_outputs(self, step: _Step, state: dict) -> None:
        if "outputs" not in step.evidence:
            return
        outputs = _object(step.evidence["outputs"], "outputs")
        if "success" in outputs and outputs["success"] is not True:
            raise _Unavailable("Primitive output reports unsuccessful execution")
        if "processCompleted" in outputs or "processCompleted_complete" in outputs:
            raise _Unavailable("Primitive output invents an unsupported process completion")
        resource = state["resources"][step.resource]
        if "attached" in outputs and (type(outputs["attached"]) is not bool
                                      or outputs["attached"] != (resource.get("held_part") is not None)):
            raise _Unavailable("Primitive attached output contradicts modeled custody")
        if "tcp_pose" in outputs:
            _same_pose(_unit_pose(outputs["tcp_pose"], "outputs.tcp_pose"),
                       resource["current_pose"], "outputs.tcp_pose")
        fields = {key: outputs[key] for key in ("current_pose", "base_pose", "held_part",
                  "gripper_state", "grasp_transform", "contained_parts") if key in outputs}
        self._resource_snapshot(fields, resource, "outputs", owner_model=step.effects is not None)
        for key in ("parts", "part_tracker"):
            for name, row in _object(outputs.get(key, {}), "outputs." + key).items():
                self._part_snapshot(name, row, state, "outputs")
        if (step.primitive == "move_cartesian" and "fraction" in outputs
                and _number(outputs["fraction"], "outputs.fraction") != 1):
            raise _Unavailable("Cartesian output reports an incomplete path")

    def _custody(self, steps: list, state: dict, time: Fraction) -> None:
        changes = [step for step in steps if step.effects is None and step.primitive in {"grasp_part", "release_part"}]
        if len({step.resource for step in changes}) != len(changes):
            raise _Unavailable("Simultaneous custody changes for one resource are ambiguous")
        grasps = [step for step in changes if step.primitive == "grasp_part"]
        names = [step.evidence["grasped_part"]["part_name"] for step in grasps]
        if len(names) != len(set(names)):
            raise _Unavailable("Simultaneous grasps claim the same part")
        for step in changes:
            if step.primitive != "release_part":
                continue
            row = state["resources"][step.resource]
            evidence = step.evidence["released_part"]
            part = row["held_part"]
            if part is None or evidence["part_name"] != part:
                raise _Unavailable("release_part identity disagrees with custody")
            if "part_name" in step.params and step.params["part_name"] != part:
                raise _Unavailable("release_part parameters disagree with custody")
            if step.params.get("transform") is not None and _unit_pose(step.params["transform"], "transform") != row["grasp_transform"]:
                raise _Unavailable("release_part transform disagrees with custody")
            _same_pose(_unit_pose(evidence["pose"], "released_part.pose"),
                       state["parts"][part]["current_pose"], "released_part.pose")
            if "contained_by" not in evidence:
                raise _Unavailable("Release evidence needs explicit containment")
            receiver = evidence["contained_by"]
            if receiver is not None:
                _symbol(receiver, "released_part.contained_by")
                if receiver not in state["resources"] or "contained_parts" not in state["resources"][receiver]:
                    raise _Unavailable("Release receiver has no complete inventory evidence")
                state["resources"][receiver]["contained_parts"].append(part)
            state["parts"][part]["contained_by"] = receiver
            self._part_snapshot(part, evidence, state, "released_part")
            state["parts"][part]["stationary_until"] = _number(
                evidence.get("stationary_until"), "released_part.stationary_until")
            row.update(held_part=None, grasp_transform=None, gripper_state="open")
            self.released_at.setdefault(time, []).append((step.resource, part))
        for step in grasps:
            row = state["resources"][step.resource]
            evidence = step.evidence["grasped_part"]
            part = evidence["part_name"]
            if part not in state["parts"] or row["held_part"] is not None or any(
                    item.get("held_part") == part for item in state["resources"].values()):
                raise _Unavailable("grasp_part has conflicting custody")
            transform = _unit_pose(evidence.get("grasp_transform"), "grasp_transform")
            actual = state["parts"][part]
            self._part_snapshot(part, evidence, state, "grasped_part")
            _same_pose(_unit_pose(evidence["pose"], "grasped_part.pose"), actual["current_pose"], "grasped_part.pose")
            _same_pose(_compose(row["current_pose"], transform), actual["current_pose"], "grasp transform")
            receiver = actual["contained_by"]
            if receiver is not None:
                state["resources"][receiver]["contained_parts"].remove(part)
            actual["contained_by"] = None
            actual.pop("stationary_until", None)
            row.update(held_part=part, grasp_transform=transform, gripper_state="closed")
        self._inventories(state)

    def _state_before_motion(self, step: _Step, part: str) -> tuple:
        pose = self.initial["parts"][part]["current_pose"]
        preceding = []
        for other in self.flat:
            if other is step:
                continue
            if part in other.part_trajectories:
                if max(step.start, other.start) < min(step.end, other.end):
                    raise _Unavailable("Simultaneous resource programs move the same part")
                if other.end <= step.start:
                    preceding.append((other.end, other.part_trajectories[part][-1][1]))
            if (other.effects is None and other.primitive == "release_part" and other.end <= step.start
                    and other.evidence["released_part"]["part_name"] == part):
                preceding.append((other.end, _unit_pose(other.evidence["released_part"]["pose"], "release pose")))
        return max(preceding, key=lambda row: row[0])[1] if preceding else pose

    def _owner_updates(self, steps: list, state: dict) -> None:
        transfers = {}
        protected = {"current_pose", "base_pose", "contained_parts", "held_part", "grasp_transform",
                     "gripper_state", "resource_id", "resource_jid", "processCompleted", "frame"}
        for step in steps:
            if step.effects is None:
                continue
            updates = step.effects["resource_updates"]
            declarations = step.model_descriptor["configuration"].get("state_variables", {})
            for field, value in updates.items():
                if field in protected or field not in state["resources"][step.resource]:
                    raise _Unavailable("Owner update lacks a declared native state field")
                from cais_spade_llm.resources.environment_models import check_value, declaration_for

                declaration = declaration_for({"state_variables": declarations}, field)
                if not check_value(declaration, value, state["parts"]):
                    raise _Unavailable("Owner update violates its native state declaration")
            state["resources"][step.resource].update(deepcopy(updates))
            for transfer in step.effects["transfers"]:
                if (not isinstance(transfer, dict) or set(transfer) != {"part_name", "from_resource", "to_resource"}
                        or any(not isinstance(value, str) or not value for value in transfer.values())):
                    raise _Unavailable("Containment transfer requires exact part and participant identities")
                key = tuple(transfer[field] for field in ("part_name", "from_resource", "to_resource"))
                transfers.setdefault(key, []).append(step.resource)
        seen = set()
        for (part, source, target), participants in transfers.items():
            if (part in seen or source == target or sorted(participants) != sorted([source, target])
                    or source not in state["resources"] or target not in state["resources"]
                    or part not in state["parts"] or state["parts"][part]["contained_by"] != source
                    or any(row.get("held_part") == part for row in state["resources"].values())):
                raise _Unavailable("Containment transfer participants or custody disagree")
            seen.add(part)
            origin, receiver = state["resources"][source], state["resources"][target]
            if part not in origin.get("contained_parts", []) or "contained_parts" not in receiver or part in receiver["contained_parts"]:
                raise _Unavailable("Containment transfer lacks complete inventories")
            origin["contained_parts"].remove(part)
            receiver["contained_parts"].append(part)
            state["parts"][part]["contained_by"] = target
        self._inventories(state)

    def _timeline(self) -> None:
        times = set(self.horizon)
        for step in self.flat:
            times.update((step.start, step.end))
            times.update(time for time, _ in step.trajectory)
            times.update(time for time, _ in step.base_trajectory)
            for points in step.part_trajectories.values():
                times.update(time for time, _ in points)
        self.timeline_times = sorted(times)
        self.timeline_states = []
        self.released_at = {}
        state = deepcopy(self.initial)
        for time in self.timeline_times:
            self._refresh(state, time)
            ending = [step for step in self.flat if step.end == time]
            self._custody(ending, state, time)
            self._owner_updates(ending, state)
            for step in ending:
                self._end_outputs(step, state)
                self._check_snapshot(step, "projected_snapshot", state)
            for step in self.flat:
                if step.start == time:
                    self._start(step, state)
            self.timeline_states.append(deepcopy(state))

    def _state(self, time: Fraction) -> dict:
        index = bisect_right(self.timeline_times, time) - 1
        state = deepcopy(self.timeline_states[index])
        self._refresh(state, time)
        return state

    def _stationary_parts(self) -> None:  # noqa: C901 - custody and complete motion interval coverage
        intervals = []
        transported = {part for step in self.flat for part in step.part_trajectories}
        for part in transported:
            row = self.initial["parts"][part]
            motion = [(step.start, step.end) for step in self.flat if part in step.part_trajectories]
            stationary = [_interval(value, "part.stationary_intervals")
                          for value in _items(row.get("stationary_intervals", []), "part.stationary_intervals")]
            if "stationary_until" in row:
                stationary.append((self.horizon[0], _number(row["stationary_until"], "stationary_until")))
            cursor = self.horizon[0]
            for start, end in sorted(motion + stationary):
                if start != cursor or end < start or end > self.horizon[1]:
                    raise _Unavailable("Transported part evidence has a gap or overlapping coverage")
                cursor = end
            if cursor != self.horizon[1]:
                raise _Unavailable("Transported part evidence does not cover the horizon")
            for start, end in stationary:
                pose = self._state(start)["parts"][part]["current_pose"]
                if any(self._state(time)["parts"][part]["current_pose"] != pose
                       for time in {start, end} | {t for t in self.timeline_times if start <= t <= end}):
                    raise _Unavailable("Part stationary evidence contradicts transported motion")
        held = {row.get("held_part") for row in self.initial["resources"].values()}
        for part, row in self.initial["parts"].items():
            if part not in held and part not in transported:
                intervals.append((part, self.horizon[0], row.get("stationary_until")))
        for step in self.flat:
            if step.effects is None and step.primitive == "release_part":
                row = step.evidence["released_part"]
                if row["part_name"] not in transported:
                    intervals.append((row["part_name"], step.end, row.get("stationary_until")))
        for part, start, raw_end in intervals:
            end = _number(raw_end, "part.stationary_until")
            next_grasp = min((step.end for step in self.flat if step.primitive == "grasp_part"
                              and step.evidence["grasped_part"]["part_name"] == part and step.end >= start),
                             default=self.horizon[1])
            if end < next_grasp or end > self.horizon[1] or end < start:
                raise _Unavailable("Part stationary evidence must reach its next grasp or the horizon")
            pose = self._state(start)["parts"][part]["current_pose"]
            for time in {start, end} | {t for t in self.timeline_times if start <= t <= end}:
                if self._state(time)["parts"][part]["current_pose"] != pose:
                    raise _Unavailable("Part stationary evidence contradicts modeled motion")

    def times(self) -> list:
        times = set(self.timeline_times)
        for start, end in zip(self.timeline_times, self.timeline_times[1:], strict=False):
            state = self._state(start)
            last = deepcopy(state)
            self._refresh(last, end)
            envelopes = []
            for resource, footprint in self.footprints.items():
                envelopes.append((state["resources"][resource]["current_pose"],
                                  last["resources"][resource]["current_pose"], footprint))
            for resource, footprint in self.base_footprints.items():
                first_base = state["resources"][resource]["base_pose"]
                last_base = last["resources"][resource]["base_pose"]
                envelopes.append((first_base[:2] + (Fraction(0),),
                                  last_base[:2] + (Fraction(0),), footprint))
            for part, footprint in self.part_footprints.items():
                envelopes.append((state["parts"][part]["current_pose"],
                                  last["parts"][part]["current_pose"], footprint))
            for first, final, footprint in envelopes:
                for region in self.regions.values():
                    for axis in range(3):
                        if first[axis] == final[axis]:
                            continue
                        for boundary in (region[axis][0] - footprint[axis][1],
                                         region[axis][1] - footprint[axis][0]):
                            ratio = (boundary - first[axis]) / (final[axis] - first[axis])
                            if 0 <= ratio <= 1:
                                times.add(start + ratio * (end - start))
        return sorted(times)

    def observe(self, time: Fraction) -> dict:
        state = self._state(time)
        carried = {resource: ([] if state["resources"][resource].get("held_part") is None
                              else [state["resources"][resource]["held_part"]])
                   for resource in self.footprints}
        for resource, part in self.released_at.get(time, []):
            if part not in carried[resource]:
                carried[resource].append(part)
        occupancy, part_occupancy = {}, {}
        for name, region in self.regions.items():
            part_occupancy[name] = {
                part: _overlaps(state["parts"][part]["current_pose"], footprint, region)
                for part, footprint in self.part_footprints.items()
            }
            occupancy[name] = {}
            for resource, footprint in self.footprints.items():
                row = state["resources"][resource]
                inside = _overlaps(row["current_pose"], footprint, region)
                if resource in self.base_footprints:
                    inside |= _overlaps(row["base_pose"][:2] + (Fraction(0),),
                                        self.base_footprints[resource], region)
                occupancy[name][resource] = inside or any(part_occupancy[name][part] for part in carried[resource])
        return {"time": float(time), "time_exact": str(time),
                "resources": _plain(state["resources"]), "parts": _plain(state["parts"]),
                "region_occupancy": occupancy, "part_region_occupancy": part_occupancy,
                "carried_parts": carried,
                "active_steps": [deepcopy(step.source) for step in self.flat if step.start <= time <= step.end]}


def model_primitive_observations_3d(
    *, programs: list[dict], snapshot: dict, geometry: dict,
    horizon: list[float], stationary: dict,
    observation_boundaries: list[float] | None = None,
    primitive_models: dict | None = None,
) -> dict[str, Any]:
    """Build one complete joint 3D observation trace without evaluating safety.

    Args:
        programs: One ordered, fully bound program per moving resource.
        snapshot: Complete declared resource, part, custody and inventory facts.
        geometry: Fixed world-axis region bounds and resource/part footprints.
            A resource with ``stationary_only: true`` has a fixed pose and no
            program, base or gripper custody; its containment inventory may vary.
        horizon: Common start and end times on the supplied evidence clock.
        stationary: Full resource coverage outside modeled primitive intervals.
        observation_boundaries: Additional exact decimal times on the joint
            evidence clock, included before interval midpoints are generated.
        primitive_models: Trusted pure owner models, bound by exact resource ID.

    Returns:
        Observations, a projected snapshot and offline model metadata, or a
        ``NEEDS_CONTEXT`` result explaining missing or contradictory evidence.
        Inputs are never mutated and a valid model never authorizes execution.
    """
    try:
        interval = _interval(horizon, "horizon")
        boundaries = _observation_boundaries(observation_boundaries, interval)
        model = _Model(_object(snapshot, "snapshot"), _object(geometry, "geometry"),
                       interval, programs, _object(stationary, "stationary"), primitive_models)
        critical = sorted(set(model.times()) | boundaries)
        times = sorted(set(critical) | {(a + b) / 2 for a, b in zip(critical, critical[1:], strict=False)})
        observations = []
        for time in times:
            row = model.observe(time)
            row["phase"] = "at" if time in critical else "between"
            observations.append(row)
        final = deepcopy(snapshot)
        final.update(resources=deepcopy(observations[-1]["resources"]),
                     parts=deepcopy(observations[-1]["parts"]))
        return {"valid": True, "feasibility_status": "FEASIBLE", "reason": "",
                "observations": observations, "projected_snapshot": final,
                "evidence": {"offline": True, "dimension": 3, "frame": model.frame,
                             "closed_intervals": True, "fixed_orientation": True,
                             "horizon": deepcopy(horizon), "clock_version": 1,
                             **({"primitive_models": model.model_descriptors,
                                 "owner_effects": [{"source": step.source, "resource_id": step.resource,
                                                    "primitive": step.primitive, "params": step.params,
                                                    "start_time": str(step.start), "end_time": str(step.end),
                                                    "effects": step.effects}
                                                   for step in model.flat if step.effects is not None]}
                                if model.model_descriptors else {}),
                             **({"observation_boundaries": [str(time) for time in sorted(boundaries)]}
                                if observation_boundaries is not None else {})}}
    except (_Unavailable, ValueError, KeyError, TypeError, IndexError) as exc:
        return {"valid": False, "feasibility_status": "NEEDS_CONTEXT", "reason": str(exc),
                "observations": [], "projected_snapshot": deepcopy(snapshot),
                "evidence": {"offline": True, "dimension": 3}}

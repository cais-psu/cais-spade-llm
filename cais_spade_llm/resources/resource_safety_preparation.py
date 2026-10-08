from __future__ import annotations

"""Resource-owned preparation contracts, command records and execution evidence.

Providers are registered by trusted application code. Scene documents contain
only identifiers and data; neither a document nor a verdict can install code.
"""

import json
import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from threading import RLock
from typing import Any
from uuid import uuid4

from cais_spade_llm.recovery_framework import fingerprint


@dataclass(frozen=True)
class PrimitiveModel:
    """A pure owner model with a serializable, versioned evidence contract.

    ``evaluate`` accepts exact primitive/params/evidence and start/end times. It
    returns trajectories, transfers and declared resource updates, never APs.
    Configuration must contain every fact on which the evaluator depends.
    """

    id: str
    version: int
    configuration: dict
    evaluate: Callable[..., dict]

    def descriptor(self) -> dict:
        """Return the frozen contract identity without executable objects."""
        if (not isinstance(self.id, str) or not self.id or type(self.version) is not int
                or self.version < 1 or not isinstance(self.configuration, dict)
                or not callable(self.evaluate)):
            raise ValueError("Invalid primitive model contract")
        result = {"id": self.id, "version": self.version,
                  "configuration": deepcopy(self.configuration)}
        json.dumps(result, allow_nan=False)
        return result

    def effects(self, *, primitive: str, params: dict, evidence: dict,
                start_time: Any, end_time: Any) -> dict:
        """Evaluate copied inputs and reject verdicts or unknown effect fields."""
        self.descriptor()
        if "primitive_contract" in evidence and evidence["primitive_contract"] != self.id:
            raise ValueError("Primitive evidence does not match its owner contract")
        result = self.evaluate(primitive=primitive, params=deepcopy(params),
                               evidence=deepcopy(evidence), start_time=start_time,
                               end_time=end_time, configuration=deepcopy(self.configuration))
        fields = {"trajectory", "base_trajectory", "part_trajectories", "transfers", "resource_updates"}
        if (not isinstance(result, dict) or not fields <= set(result)
                or set(result) - fields - {"continuous_motion", "custody_effects"}
                or "continuous_motion" in result and self.version < 2
                or "custody_effects" in result and self.version < 3):
            raise ValueError("Owner model must supply the complete supported effect record")
        if "custody_effects" in result and (not isinstance(result["custody_effects"], list)
                or any(not isinstance(row, dict) for row in result["custody_effects"])):
            raise ValueError("Custody effects require a list of explicit observation records")
        json.dumps(result, allow_nan=False)
        return deepcopy(result)


def primitive_model_descriptors(primitive_models: dict | None) -> dict:
    """Validate exact resource bindings and return only serializable metadata."""
    if primitive_models is None:
        return {}
    if not isinstance(primitive_models, dict):
        raise ValueError("primitive_models must bind exact resource identities")
    result = {}
    for resource, model in primitive_models.items():
        if not isinstance(resource, str) or not resource or not isinstance(model, PrimitiveModel):
            raise ValueError("Invalid resource-owned primitive model")
        result[resource] = model.descriptor()
    return result


_RESOURCE_PROVIDERS: dict[str, Any] = {}
_LEGACY_MODELS: dict[str, Callable[[], PrimitiveModel]] = {}


def register_resource_provider(identifier: str, provider: Any) -> None:
    """Register a trusted owner factory and its immutable program validator."""
    if (not isinstance(identifier, str) or not identifier
            or type(getattr(provider, "version", None)) is not int or provider.version < 1
            or not callable(getattr(provider, "create_owner", None))
            or not callable(getattr(provider, "validate_programs", None))):
        raise ValueError("Resource provider needs a version, owner factory and program validator")
    if identifier in _RESOURCE_PROVIDERS and _RESOURCE_PROVIDERS[identifier] is not provider:
        raise ValueError("Resource provider already registered: " + identifier)
    _RESOURCE_PROVIDERS[identifier] = provider


def resource_provider(identifier: str) -> Any:
    """Resolve an exact trusted provider reference without dynamic imports."""
    if not isinstance(identifier, str) or identifier not in _RESOURCE_PROVIDERS:
        raise ValueError("Resource owner provider unavailable: " + str(identifier))
    return _RESOURCE_PROVIDERS[identifier]


def resource_model_declarations(scene: dict) -> dict:
    """Read additive scene models without replacing any existing resource."""
    entries = scene.get("resource_models", {})
    if not isinstance(entries, dict):
        raise ValueError("scene.resource_models must be a mapping")
    identities = [*(row["resource_id"] for row in scene["robots"]),
                *(row["resource_id"] for row in scene["machines"]),
                "KMR", "Conveyor", "Buffer For Machined parts", "3D Printing Station", "Storage", "Exit"]
    existing = set(identities)
    if len(existing) != len(identities):
        raise ValueError("Duplicate configured resource identities")
    for rid, entry in entries.items():
        if (not isinstance(rid, str) or not rid or rid in existing
                or not isinstance(entry, dict) or set(entry) != {"model", "owner_provider"}
                or not isinstance(entry["owner_provider"], str) or not entry["owner_provider"]):
            raise ValueError("Invalid or duplicate additional resource declaration: " + str(rid))
        model = entry["model"]
        if (not isinstance(model, dict) or model.get("resource_id") != rid
                or not isinstance(model.get("events"), list)
                or not isinstance(model.get("state_variables"), dict)
                or not isinstance(model.get("current_valuation"), dict)
                or not isinstance(model.get("assignments"), dict)
                or not isinstance(model.get("marked_state_conditions"), list)):
            raise ValueError("Incomplete environment model for " + rid)
        if not {field for field in model["state_variables"] if "{part_name}" not in field} <= set(model["current_valuation"]):
            raise ValueError("Initial resource valuation must cover its declared fields")
        json.dumps(entry, allow_nan=False)
    return deepcopy(entries)


def legacy_primitive_model(evidence: dict) -> PrimitiveModel | None:
    """Resolve explicit historical contracts; absent tags retain legacy models."""
    contract = evidence.get("primitive_contract")
    if contract is None:
        return None
    if not _LEGACY_MODELS:
        # Built-in registration is separate from resource or action dispatch.
        from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
            register_preparation_contracts,
        )

        register_preparation_contracts(_LEGACY_MODELS)
    if not isinstance(contract, str) or contract not in _LEGACY_MODELS:
        raise ValueError("Unknown primitive observation contract: " + str(contract))
    return _LEGACY_MODELS[contract]()


def pose_values(pose) -> list[float]:
    """Copy an observed pose without changing its frame or quaternion."""
    values = [getattr(pose.position, key) for key in ("x", "y", "z")]
    values += [getattr(pose.orientation, key) for key in ("x", "y", "z", "w")]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite observed pose")
    if not math.isclose(sum(value * value for value in values[3:]), 1., abs_tol=1e-5):
        raise ValueError("Observed pose requires a unit quaternion")
    return values


def validate_joint_trajectory(trajectory: dict, joint_names: list[str], duration: float | None = None) -> None:
    """Check exact configured joints, values, point order and nanosecond timing."""
    if (not joint_names or any(not isinstance(name, str) or not name for name in joint_names)
            or len(set(joint_names)) != len(joint_names) or trajectory["joint_names"] != joint_names
            or len(trajectory["points"]) < 2):
        raise ValueError("Joint trajectory does not cover the configured joint identities")
    previous = -1
    for index, row in enumerate(trajectory["points"]):
        stamp = row["time_from_start"]
        if (type(stamp["sec"]) is not int or stamp["sec"] < 0
                or type(stamp["nanosec"]) is not int or not 0 <= stamp["nanosec"] < 1_000_000_000):
            raise ValueError("Invalid joint trajectory timing")
        nanos = stamp["sec"] * 1_000_000_000 + stamp["nanosec"]
        if (nanos <= previous or previous == -1 and nanos != 0
                or len(row["positions"]) != len(joint_names)):
            raise ValueError(
                "Joint trajectory point order or coverage is invalid: "
                f"point_index={index}, time_from_start_ns={nanos}, previous_time_ns={previous}, "
                f"positions={len(row['positions'])}, joint_names={len(joint_names)}"
            )
        for field in ("positions", "velocities", "accelerations", "effort"):
            values = row.get(field, [])
            if len(values) not in (0, len(joint_names)) or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError("Joint trajectory values are invalid")
        previous = nanos
    if (previous <= 0 or type(trajectory["duration_ns"]) is not int or trajectory["duration_ns"] != previous
            or duration is not None and not math.isclose(previous / 1e9, duration, abs_tol=1e-9)):
        raise ValueError("Joint trajectory duration disagrees with its interval")


def trajectory_record(trajectory, joint_names: list[str] | None = None, *, validate: bool = True) -> dict:
    """Preserve native values, allowing raw diagnostics before strict validation.

    Args:
        trajectory: Native ROS joint trajectory to copy.
        joint_names: Exact configured joint ordering, when available.
        validate: Require a complete trajectory beginning at zero. False only
            captures diagnostics; callers must validate before preparation.

    Returns:
        The unchanged native points, derivatives, header and timing.
    """
    names = list(trajectory.joint_names)
    rows = []
    for point in trajectory.points:
        stamp = point.time_from_start
        rows.append({**{key: list(getattr(point, key, [])) for key in (
            "positions", "velocities", "accelerations", "effort")},
            "time_from_start": {"sec": stamp.sec, "nanosec": stamp.nanosec}})
    if not rows:
        raise ValueError("Prepared trajectory has no points")
    header, last = trajectory.header, rows[-1]["time_from_start"]
    result = {"header": {"frame_id": header.frame_id,
                         "stamp": {"sec": header.stamp.sec, "nanosec": header.stamp.nanosec}},
              "joint_names": names, "points": rows,
              "duration_ns": last["sec"] * 1_000_000_000 + last["nanosec"]}
    if validate:
        validate_joint_trajectory(result, names if joint_names is None else joint_names)
    return result


def cartesian_coverage_fingerprint(trajectory: list, start_time) -> str:
    """Bind continuous coverage to relative time without inventing a schedule."""
    return fingerprint([{**row, "time": str(Fraction(str(row["time"])) - Fraction(str(start_time)))}
                        for row in trajectory])


def observation_identity(observation: dict) -> dict:
    """Exclude receipt times, retaining all owner-declared physical facts."""
    if observation.get("status") == "NEEDS_CONTEXT":
        raise ValueError(observation.get("reason", "Resource observation unavailable"))
    fields = {"observed_monotonic", "simulation_time", "tool_stamp", "joint_stamps",
              "joint_received_monotonic"}
    result = {key: deepcopy(value) for key, value in observation.items() if key not in fields}
    if isinstance(result.get("gazebo_observation"), dict):
        result["gazebo_observation"] = {key: value for key, value in result["gazebo_observation"].items()
                                        if key not in {"simulation_stamp", "observed_monotonic"}}
    return result


class LiveCommandLedger:
    """Retain pending commands before dispatch and outcomes until clearance.

    Args:
        lock: The runtime's shared admission transaction lock.
    """

    def __init__(self, lock: Any | None = None) -> None:
        self.lock = lock if lock is not None else RLock()
        self.revision = 0
        self.execution_epoch = 0
        self.commands: dict[str, dict[str, Any]] = {}

    def prepare(self, *, resource_id: str, task_id: str | None,
                command: dict, owner_identity: dict) -> str:
        """Record the exact owner command before its controller can accept it."""
        if not resource_id or not isinstance(command, dict) or not owner_identity:
            raise ValueError("Live command requires its resource, command, and owner identity")
        with self.lock:
            identifier = uuid4().hex
            self.commands[identifier] = {
                "command_id": identifier, "resource_id": resource_id,
                "task_id": task_id, "command": deepcopy(command),
                "command_fingerprint": fingerprint(command),
                "owner_identity": deepcopy(owner_identity), "status": "pending",
                "controller_goal_id": None, "observations": None,
            }
            self.revision += 1
            return identifier

    def authorize(self, identifier: str, *, reservation_token: str,
                  expected_revision: int) -> None:
        """Attach an atomic CCA reservation to an unchanged pending command."""
        with self.lock:
            row = self.commands[identifier]
            if (self.revision != expected_revision or row["status"] != "pending"
                    or row.get("reservation_token") or not isinstance(reservation_token, str) or not reservation_token):
                raise ValueError("Physical command reservation is stale or missing")
            row["reservation_token"] = reservation_token
            self.revision += 1
            self.execution_epoch += 1

    def authorize_many(self, identifiers: list[str], *, reservation_token: str,
                       expected_revision: int) -> None:
        """Authorize a complete program without exposing partial reservations."""
        with self.lock:
            if (self.revision != expected_revision or not identifiers
                    or len(identifiers) != len(set(identifiers))
                    or not isinstance(reservation_token, str) or not reservation_token):
                raise ValueError("Physical program reservation is stale or missing")
            rows = [self.commands[identifier] for identifier in identifiers]
            if any(row["status"] != "pending" or row.get("reservation_token") for row in rows):
                raise ValueError("Physical program has already started")
            for row in rows:
                row["reservation_token"] = reservation_token
            self.revision += 1
            self.execution_epoch += 1

    def require_authorized(self, identifier: str, *, command: dict) -> str:
        """Check the exact pending command immediately before dispatch."""
        with self.lock:
            row = self.commands[identifier]
            if (row["status"] != "pending" or row["command"] != command
                    or not row.get("reservation_token")):
                raise ValueError("Live command has no matching physical reservation")
            return row["reservation_token"]

    def start(self, identifier: str, *, command: dict, controller_goal_id: Any) -> None:
        """Bind controller acceptance to the unchanged pending command."""
        with self.lock:
            row = self.commands[identifier]
            if (row["status"] != "pending" or row["command"] != command
                    or not row.get("reservation_token")):
                raise ValueError("Controller command differs from its pending physical evidence")
            row.update(status="active", controller_goal_id=deepcopy(controller_goal_id))
            self.revision += 1
            self.execution_epoch += 1

    def finish(self, identifier: str, *, success: bool, observations: dict) -> None:
        """Record an outcome without inferring physical clearance from completion."""
        if type(success) is not bool or not isinstance(observations, dict):
            raise ValueError("Command completion requires observed outcome evidence")
        with self.lock:
            row = self.commands[identifier]
            if row["status"] not in {"pending", "active"}:
                raise ValueError("Live command already has a terminal outcome")
            row.update(status="completed" if success else "failed",
                       observations=deepcopy(observations))
            self.revision += 1
            self.execution_epoch += 1

    def bind_goal(self, identifier: str, controller_goal_id: list[int]) -> None:
        """Attach actual controller acceptance to the command already being sent."""
        with self.lock:
            row = self.commands[identifier]
            if row["status"] != "active" or row["controller_goal_id"] is not None:
                raise ValueError("Controller goal cannot be rebound")
            row["controller_goal_id"] = list(controller_goal_id)
            self.revision += 1
            self.execution_epoch += 1

    def snapshot(self) -> dict:
        """Return detached command history, including pending and failed motions."""
        with self.lock:
            return {"revision": self.revision, "execution_epoch": self.execution_epoch,
                    "commands": deepcopy(self.commands)}


def _validate_controller_contract(record: dict, planned: dict) -> None:
    """Check native provenance and distinguish ownership from physical bounds."""
    if (not isinstance(record, dict) or record.get("operation") != "prepare"
            or type(record.get("accepted")) is not bool):
        raise ValueError("Native motion contract response is malformed")
    if not record["accepted"]:
        raise ValueError("Native motion contract rejected preparation: " + str(record.get("reason", "")))
    if record.get("reason") != "":
        raise ValueError("Native motion contract accepted preparation with a conflicting reason")
    for field in ("instance_id", "expected_instance_id", "binding_fingerprint",
                  "requested_binding_fingerprint"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError("Native motion contract identity is unavailable: " + field)
    for field in ("command_revision", "expected_command_revision", "contract_revision",
                  "expected_contract_revision", "rejected_commands"):
        if type(record.get(field)) is not int or record[field] < 0:
            raise ValueError("Native motion contract revision or count is unavailable: " + field)
    if (record.get("instance_id") != record["expected_instance_id"]
            or record.get("command_revision") != record["expected_command_revision"]
            or record.get("contract_revision") != record["expected_contract_revision"]
            or record.get("binding_fingerprint") != record["requested_binding_fingerprint"]):
        raise ValueError("Native motion contract identity changed during preparation")
    names = planned["joint_trajectory"]["joint_names"]
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or record.get("joint_names") != names):
        raise ValueError("Native motion contract names another set of joints")
    for field in ("observed_positions", "observed_velocities"):
        values = record.get(field)
        if (not isinstance(values, list) or len(values) != len(names)
                or any(type(value) not in (float, int) or not math.isfinite(value) for value in values)):
            raise ValueError("Native motion contract has incomplete joint observations")
    if record["observed_positions"] != planned["start"]["joint_positions"]:
        raise ValueError("Native controller moved from the prepared initial positions")
    if record.get("stationary") is not False or record.get("reservation_token") != "":
        raise ValueError("Read-only motion preparation unexpectedly acquired execution authority")
    if (record.get("contract_state") != "unlocked" or record.get("observed_stationary") is not True
            or type(record.get("stationary_samples")) is not int or record["stationary_samples"] < 2
            or any(value != 0 for value in record["observed_velocities"])):
        raise ValueError("Native motion contract has no observed stationary unlocked owner")
    for field in ("simulation_time", "checkpoint_simulation_time", "elapsed_wall_s"):
        if type(record.get(field)) not in (int, float) or not math.isfinite(record[field]):
            raise ValueError("Native motion contract timing is unavailable")
    if (not 0 <= record["simulation_time"] - record["checkpoint_simulation_time"] <= 2.0
            or not 0 <= record["elapsed_wall_s"] <= 2.0):
        raise ValueError("Native motion contract observation is stale")
    if type(record.get("physical_execution_verified")) is not bool:
        raise ValueError("Native physical execution status is unavailable")
    if record["physical_execution_verified"]:
        raise ValueError("Unsupported physical containment assertion from the position controller")
    if (not isinstance(record.get("physical_execution_reason"), str)
            or not record["physical_execution_reason"]):
        raise ValueError("Native physical execution unavailability reason is missing")
    # These fields describe past samples only. Even zero observed error or
    # regular updates cannot establish future containment or stopping bounds.
    period = record.get("maximum_observed_update_period")
    errors = record.get("maximum_observed_position_error")
    if (type(period) not in (int, float) or not math.isfinite(period) or period < 0
            or not isinstance(errors, list) or len(errors) != len(names)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in errors)):
        raise ValueError("Native motion contract sample diagnostics are unavailable")


def validate_prepared_start(observed: dict, expected: dict, *, continuous_motion: dict | None = None) -> None:
    """Require unchanged owner state and every supplied physical geometry field.

    Detached preparation requires exact observations. A CCA-owned modeled start
    may use the retained joint_position_error assumption and its enclosing FK
    geometry. Neither path certifies future physical tracking or stopping.
    """
    for field in ("joint_names", "frame", "launch_id", "attachment"):
        if observed.get(field) != expected.get(field) or field not in observed or field not in expected:
            raise ValueError("Prepared motion owner or attachment changed: " + field)
    positions = observed["joint_positions"]
    if expected.get("model_execution") is True:
        from cais_spade_llm.resources.continuous_motion import ContinuousMotion, occupancy, quaternion_transform

        if continuous_motion is None:
            raise ValueError("Modeled start requires its retained continuous motion configuration")
        motion = ContinuousMotion(continuous_motion["joint_trajectory"], continuous_motion["configuration"])
        errors = motion.joint_position_error
        names = expected["joint_names"]
        if (motion.names != names or set(errors) != set(names) or len(positions) != len(names)
                or len(expected["joint_positions"]) != len(names)
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or abs(value - target) > errors[name]
                       for name, value, target in zip(names, positions, expected["joint_positions"], strict=True))):
            raise ValueError("Observed joints violate the configured joint_position_error at the prepared start")
        for field in ("base_pose", "root_pose", "geometry_source", "custody_complete", "model_static"):
            if field in expected and observed.get(field) != expected[field]:
                raise ValueError("Prepared physical geometry or custody changed: " + field)
        old_state, new_state = deepcopy(expected.get("observation_state", {})), deepcopy(observed.get("observation_state", {}))
        for state in (old_state, new_state):
            state.pop("current_pose", None)
            state.pop("joint_positions", None)
        if old_state != new_state:
            raise ValueError("Prepared physical geometry or custody changed: observation_state")
        pose = observed["current_pose"]
        if (not isinstance(pose, list) or len(pose) != 7
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in pose)):
            raise ValueError("Observed current_pose is incomplete at the modeled start")
        measured = quaternion_transform(pose[:3], pose[3:])
        reference = motion.transforms(Fraction(0), Fraction(0))[motion.configuration["reference_link"]]
        for i in range(3):
            for j in range(4):
                if measured[i][j][0] < reference[i][j][0] or measured[i][j][1] > reference[i][j][1]:
                    raise ValueError("Observed current_pose is outside the configured initial FK enclosure")
        components = observed.get("component_bounds")
        if not isinstance(components, list) or not components:
            raise ValueError("Observed component_bounds are unavailable at the modeled start")
        boxes = motion.boxes(Fraction(0), Fraction(0))
        fixed_links, known_links = {motion.configuration["root"]}, {motion.configuration["root"]}
        for joint in motion.configuration["joints"]:
            known_links.add(joint["child"])
            if joint["type"] == "fixed" and joint["parent"] in fixed_links:
                fixed_links.add(joint["child"])
        fixed_ids = {row["id"] for row in motion.configuration["components"] if row["link"] in fixed_links}
        fixed_boxes = [box for box in boxes if box["id"] in fixed_ids]
        moving_boxes = [box for box in boxes if box["id"] not in fixed_ids]
        captured_components = expected.get("component_bounds")
        if not isinstance(captured_components, list) or not captured_components:
            raise ValueError("Captured component_bounds are unavailable at the modeled start")
        component_maps, fixed_observations = [], []
        for rows in (captured_components, components):
            identifiers, fixed = {}, []
            for component in rows:
                bounds, identifier, link = component.get("bounds"), component.get("id"), component.get("link")
                if (not isinstance(identifier, str) or not identifier or identifier in identifiers
                        or link not in known_links):
                    raise ValueError("Observed component_bounds require unchanged literal native link identities: "
                                     + str(identifier))
                identifiers[identifier] = link
                if (not isinstance(bounds, list) or len(bounds) != 3
                        or any(not isinstance(pair, list) or len(pair) != 2
                               or any(type(value) not in (int, float) or not math.isfinite(value) for value in pair)
                               or pair[0] > pair[1] for pair in bounds)):
                    raise ValueError("Observed component_bounds are incomplete: " + str(identifier))
                if link in fixed_links:
                    fixed.append({"id": identifier, "minimum": [[pair[0], pair[0]] for pair in bounds],
                                  "maximum": [[pair[1], pair[1]] for pair in bounds]})
            component_maps.append(identifiers)
            fixed_observations.append(fixed)
        if component_maps[0] != component_maps[1]:
            raise ValueError("Observed component_bounds native link population changed")
        if bool(fixed_boxes) != bool(fixed_observations[0]):
            raise ValueError("Observed fixed component_bounds do not cover the configured root-fixed links")
        if fixed_boxes:
            stationary = {"kind": "idle_commanded_hold", "requires_no_running_tasks": True,
                          "requires_no_active_goals": True, "future_execution_tracking": "not_established"}
            if expected.get("stationary_contract") != stationary or observed.get("stationary_contract") != stationary:
                raise ValueError("Fixed component occupancy requires the declared commanded stationary model")
            regions = expected.get("model_execution_regions")
            if not isinstance(regions, dict) or not regions:
                raise ValueError("Fixed component occupancy requires the retained complete model_execution_regions")
            # Only the root-fixed contribution is compared: a moving arm in a
            # region must never conceal a different base occupancy in that region.
            for name, region in regions.items():
                bounds = region.get("bounds") if isinstance(region, dict) else None
                if (not isinstance(region, dict) or region.get("frame") != expected["frame"]
                        or not isinstance(bounds, list) or len(bounds) != 3
                        or any(not isinstance(pair, list) or len(pair) != 2
                               or any(type(value) not in (int, float) or not math.isfinite(value) for value in pair)
                               or pair[0] > pair[1] for pair in bounds)):
                    raise ValueError("Fixed component occupancy requires a complete configured region: " + str(name))
                modeled = occupancy(fixed_boxes, bounds)
                if len(modeled) != 1 or any(occupancy(group, bounds) != modeled for group in fixed_observations):
                    raise ValueError("Fixed component occupancy changed or is unknown in configured region: " + str(name))
        for component in components:
            if component["link"] in fixed_links:
                continue
            bounds = component["bounds"]
            if not any(all(box["minimum"][i][0] <= bounds[i][0] <= bounds[i][1] <= box["maximum"][i][1]
                           for i in range(3)) for box in moving_boxes):
                raise ValueError("Observed component_bounds are outside the configured initial enclosure: "
                                 + str(component.get("id")))
        return
    if (positions != expected["joint_positions"]
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in positions)):
        raise ValueError("Resource moved after physical motion preparation")
    for field in ("current_pose", "base_pose", "root_pose", "component_bounds", "footprint",
                  "geometry_source", "custody_complete", "model_static", "observation_state"):
        if field in expected and observed.get(field) != expected[field]:
            raise ValueError("Prepared physical geometry or custody changed: " + field)


class PreparedRobotEvidence:
    """Bind CCA requests to stored owner preparations and exact command execution.

    Args:
        owner: The initialized ResourceAgent.
        ledger: The run's shared pending/active command ledger.
    """

    def __init__(self, owner, ledger) -> None:
        self.owner = owner
        self.ledger = ledger
        self.steps: dict[str, dict] = {}
        self.preparations: dict[str, dict] = {}
        self.execution_observer = None
        self.dispatch_command = None
        self.execution_coverage_records: dict[str, dict] = {}

    def prepare_execution_coverage(self, *, prepared: dict, checkpoint: dict) -> dict:
        """Bind modeled Gazebo execution without promoting samples to bounds.

        The native controller owns command authorization and observation. Its
        position interface does not constrain Gazebo's intervening physics step,
        so this evidence never establishes a live physical admission certificate.

        Args:
            prepared: Exact program retained by this resource owner.
            checkpoint: Complete observed scene used for native preparation.

        Returns:
            Model coverage and separate physical-bound diagnostics. A prepared
            result remains conditional on its exact interpolation and stationary
            contracts; it grants no command or reservation by itself.
        """
        result = {
            "status": "NEEDS_CONTEXT", "resource_id": self.owner.agent_name,
            "checkpoint_id": checkpoint.get("checkpoint_id"), "dispatch_authorized": False,
            "command_sent": False, "native_contract": None,
            "physical_execution_verified": False,
            "physical_execution_reason": None,
            "native_preparation_verified": False,
            "native_preparation_reason": None,
            "model_execution_verified": False,
            "model_execution_assumptions": None,
            "missing_physical_evidence": [
                "verified tracking bounds", "future stationary containment bounds",
                "failure stopping bounds",
            ],
            "stationary_resources": [],
        }
        identity = fingerprint(prepared)
        result["prepared_fingerprint"] = identity
        try:
            checkpoint_identity = fingerprint(checkpoint)
            result["checkpoint_fingerprint"] = checkpoint_identity
            if prepared.get("status") != "prepared" or prepared.get("resource_id") != self.owner.agent_name:
                raise ValueError("Live coverage requires the responsible owner's prepared program")
            steps = prepared.get("steps")
            if not isinstance(steps, list) or len(steps) != 1 or steps[0].get("primitive") != "move_cartesian":
                raise ValueError("Live motion coverage is limited to one prepared move_cartesian")
            planned = steps[0]
            result["preparation_id"] = planned["preparation_id"]
            with self.ledger.lock:
                if self.steps.get(planned.get("preparation_id")) != planned:
                    raise ValueError("Live coverage differs from the retained owner preparation")
                if any(row["status"] == "active"
                       or row["status"] == "pending" and row.get("reservation_token")
                       for row in self.ledger.commands.values()):
                    raise ValueError("Outstanding physical commands prevent stationary scene coverage")
                ledger_revision = self.ledger.revision
                execution_epoch = self.ledger.execution_epoch
                result.update(command_ledger_revision=ledger_revision, execution_epoch=execution_epoch)
            if (not isinstance(checkpoint.get("checkpoint_id"), str) or not checkpoint["checkpoint_id"]
                    or checkpoint.get("unresolved") != []
                    or not isinstance(checkpoint.get("observations"), dict)
                    or self.owner.agent_name not in checkpoint["observations"]):
                raise ValueError("Live coverage requires a complete physical checkpoint")
            model_execution = checkpoint.get("model_execution") is True
            for resource, row in checkpoint["observations"].items():
                physical = row["physical"]
                if (not model_execution and physical.get("idle") is not True
                        or physical.get("custody_complete") is not True
                        or not isinstance(physical.get("attachment"), dict)
                        or "model_name" not in physical["attachment"]
                        or physical["attachment"]["model_name"] is not None
                        or not isinstance(physical.get("observation_state"), dict)
                        or physical["observation_state"].get("held_part") is not None):
                    raise ValueError("Live coverage requires observed stationary empty custody: " + resource)
            result["stationary_resources"] = sorted(checkpoint["observations"])
            controller = self.owner._controller
            if controller.execution_mode != "simulation":
                raise ValueError("The native motion contract supports Gazebo only")
            result["reason"] = (
                "Native command ownership is available, but the Gazebo position controller has no "
                "verified tracking, future stationary containment, or failure stopping bounds"
            )
            if model_execution:
                from cais_spade_llm.resources.continuous_motion import ContinuousMotion

                continuous = planned["continuous_motion"]
                if (continuous["joint_trajectory"] != planned["joint_trajectory"]
                        or planned["configuration_fingerprint"] != fingerprint(controller.controller_config)):
                    raise ValueError("Modeled execution differs from the retained controller configuration")
                ContinuousMotion(continuous["joint_trajectory"], continuous["configuration"])
                observer = controller.recovery_safety_observer
                if continuous["configuration"] != observer.motion_configuration:
                    raise ValueError("Modeled execution changed its registered kinematic configuration")
                stationary = {}
                for resource, row in checkpoint["observations"].items():
                    declaration = row["physical"].get("stationary_contract")
                    if not isinstance(declaration, dict) or not declaration.get("kind"):
                        raise ValueError("Modeled stationary contract is missing: " + resource)
                    stationary[resource] = deepcopy(declaration)
                goals = checkpoint.get("controller_goals")
                service = observer.configuration["controller_node"].rstrip("/") + "/recovery_state"
                if not isinstance(goals, dict) or service not in goals:
                    raise ValueError("Modeled execution has no observed controller identity")
                controller_goals = {}
                for name, state in goals.items():
                    if (not isinstance(state, dict) or not isinstance(state.get("instance_id"), str)
                            or not state["instance_id"] or state.get("holding") is not True
                            or state.get("has_active_goal") is not False or state.get("has_pending_goal") is not False
                            or state.get("contract_state", "unlocked") != "unlocked"
                            or state.get("reservation_token", "") != ""):
                        raise ValueError("Modeled stationary controller is not in an observed commanded hold: " + name)
                    for field in ("command_revision", "contract_revision"):
                        if field in state and (type(state[field]) is not int or state[field] < 0):
                            raise ValueError("Modeled controller revision is unavailable: " + name)
                    controller_goals[name] = {key: deepcopy(state[key]) for key in (
                        "instance_id", "controller", "command_revision", "contract_revision", "holding",
                        "has_active_goal", "has_pending_goal", "navigation_enabled", "contract_state",
                        "reservation_token", "binding_fingerprint",
                    ) if key in state}
                result.update(
                    status="prepared", model_execution_verified=True,
                    model_execution_assumptions={"continuous_motion": deepcopy(continuous),
                                                 "stationary_contracts": stationary,
                                                 "controller_goals": controller_goals},
                    reason="Gazebo execution is conditional on the retained interpolation, joint_position_error "
                           "and stationary contracts; physical bounds are not certified",
                )
            from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
                _prepare_controller_contract,
            )

            try:
                record = _prepare_controller_contract(controller, planned, checkpoint)
                result["native_contract"] = deepcopy(record)
                _validate_controller_contract(record, planned)
                result["native_preparation_verified"] = True
                result["physical_execution_reason"] = record["physical_execution_reason"]
            except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, ImportError) as exc:
                result["native_preparation_reason"] = str(exc)
                if not model_execution:
                    raise
            if model_execution and isinstance(result["native_contract"], dict):
                # A rejection for sampled drift is diagnostic. A changed owner
                # incarnation or command revision invalidates this model binding.
                record = result["native_contract"]
                for field in ("instance_id", "command_revision", "contract_revision"):
                    if record.get(field) != goals[service].get(field):
                        raise ValueError("Modeled controller identity changed during preparation: " + field)
            with self.ledger.lock:
                if (self.ledger.revision != ledger_revision or self.ledger.execution_epoch != execution_epoch
                        or self.steps.get(planned["preparation_id"]) != planned
                        or fingerprint(prepared) != identity or fingerprint(checkpoint) != checkpoint_identity):
                    raise ValueError("Owner preparation, checkpoint or command ledger changed during native preparation")
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, ImportError) as exc:
            result.update(status="NEEDS_CONTEXT", model_execution_verified=False,
                          native_preparation_verified=False, reason=str(exc))
        with self.ledger.lock:
            self.execution_coverage_records[identity] = deepcopy(result)
        return result

    def register_program(self, prepared: dict) -> None:
        """Retain only this owner's original continuously modeled preparation."""
        if prepared.get("status") != "prepared" or prepared.get("resource_id") != self.owner.agent_name:
            raise ValueError("Cannot register unavailable physical preparation")
        with self.ledger.lock:
            for step in prepared["steps"]:
                identity = step.get("preparation_id")
                if (step.get("observation_status") != "prepared" or not identity
                        or identity != fingerprint({key: value for key, value in step.items()
                                                    if key != "preparation_id"})):
                    raise ValueError("Physical preparation identity is unavailable")
                previous = self.steps.get(identity)
                if previous is not None and previous != step:
                    raise ValueError("Owner preparation changed")
                self.steps[identity] = deepcopy(step)

    def prepare(self, *, resource_agent, request: dict, physical_snapshot: dict) -> dict:
        """Verify every schedule against original native trajectory evidence."""
        if resource_agent is not self.owner or request["resource_jid"] != str(self.owner.jid):
            raise ValueError("Preparation belongs to another resource owner")
        identity = fingerprint(request)
        with self.ledger.lock:
            if identity in self.preparations:
                result = deepcopy(self.preparations[identity]["result"])
                result["command_ledger_revision"] = self.ledger.revision
                return result
            programs = request["event_programs"]
            if not programs:
                raise ValueError("No complete physical schedule supplied")
            selected = None
            for program in programs:
                rows = []
                for result in program["step_results"]:
                    evidence = result["model_evidence"]
                    planned = self.steps.get(evidence.get("preparation_id"))
                    if planned is None:
                        raise ValueError("CCA schedule has no original owner preparation")
                    if (planned["primitive"] != result["primitive"]
                            or planned["params"] != result["resolved_params"]
                            or planned["joint_trajectory"] != evidence["joint_trajectory"]
                            or planned["continuous_motion"] != evidence["continuous_motion"]
                            or planned.get("resolved_motion") != evidence.get("resolved_motion")):
                        raise ValueError("CCA schedule changed the owner motion")
                    rows.append(deepcopy(planned))
                if selected is not None and selected != rows:
                    raise ValueError("Schedules change the prepared primitive program")
                selected = rows
            if len(selected) != len(request["primitive_steps"]):
                raise ValueError("Prepared commands do not cover the complete event")
            command_ids = [self.ledger.prepare(
                resource_id=self.owner.agent_name, task_id=request["task_id"],
                command={"primitive": row["primitive"], "params": row["params"],
                         "joint_trajectory": row["joint_trajectory"]},
                owner_identity={"resource_jid": str(self.owner.jid),
                                "launch_id": row["start"]["launch_id"],
                                "preparation_id": row["preparation_id"]}) for row in selected]
            result = {"status": "prepared", "resource_jid": str(self.owner.jid),
                      "program_hash": request["program_hash"], "execution_mode": "live",
                      "preparation_id": identity, "command_ids": command_ids,
                      "command_ledger_revision": self.ledger.revision}
            self.preparations[identity] = {"request": deepcopy(request), "steps": selected,
                                           "result": result, "executed": set()}
            return deepcopy(result)

    def authorize_preparation(self, *, preparation_id: str, reservation_token: str,
                              command_ids: list[str], expected_revision: int) -> bool:
        """Atomically bind a whole primitive program to the CCA's region grant."""
        with self.ledger.lock:
            prepared = self.preparations[preparation_id]
            if prepared["result"]["command_ids"] != command_ids:
                raise ValueError("Physical reservation names another prepared program")
            self.ledger.authorize_many(command_ids, reservation_token=reservation_token,
                                       expected_revision=expected_revision)
            return True

    def execute_step(self, *, resource_agent, task_id: str, step_index: int,
                     primitive: str, params: dict, grant: dict) -> dict:
        """Consume a granted command once and dispatch its original trajectory."""
        if resource_agent is not self.owner:
            raise ValueError("Prepared executor owner changed")
        with self.ledger.lock:
            prepared = self.preparations[grant["preparation_id"]]
            if task_id != prepared["request"]["task_id"] or step_index in prepared["executed"]:
                raise ValueError("Prepared command task changed or command was consumed")
            planned = deepcopy(prepared["steps"][step_index])
            if planned["primitive"] != primitive or planned["params"] != params:
                raise ValueError("Dispatch parameters differ from the physical preparation")
            command_id = prepared["result"]["command_ids"][step_index]
            command = {"primitive": primitive, "params": params,
                       "joint_trajectory": planned["joint_trajectory"]}
            self.ledger.require_authorized(command_id, command=command)
            prepared["executed"].add(step_index)
            self.ledger.start(command_id, command=command, controller_goal_id=None)
            self.dispatch_command = command_id
        try:
            result = self.owner._controller.execute_prepared_recovery_step(planned)
        except (ValueError, KeyError, RuntimeError, TimeoutError) as exc:
            result = {"success": False, "command_id": command_id, "error": str(exc)}
            self.ledger.finish(command_id, success=False, observations={"error": str(exc)})
            self._publish(task_id, step_index, result)
            raise
        finally:
            self.dispatch_command = None
        self.ledger.finish(command_id, success=result.get("success") is True,
                           observations=result.get("observations", {}))
        self._publish(task_id, step_index, result)
        return result

    def _publish(self, task_id: str, step_index: int, result: dict) -> None:
        if not callable(self.execution_observer):
            raise ValueError("Live execution has no registered monitor observer")
        self.execution_observer(task_id=task_id, step_index=step_index, result=deepcopy(result))

    def dispatch_guard(self, topic: str, trajectory) -> str:
        """Reject physical sends that bypass an admitted original trajectory."""
        with self.ledger.lock:
            identifier = self.dispatch_command
            row = self.ledger.commands.get(identifier)
            if (row is None or row["status"] != "active" or not row.get("reservation_token")
                    or row["command"]["joint_trajectory"] != trajectory_record(trajectory)):
                raise ValueError("Controller dispatch lacks its exact live physical grant")
            if topic != self.owner._controller.arm_trajectory_topic:
                raise ValueError("Prepared trajectory belongs to another controller channel")
            return identifier

    def dispatch_accepted(self, identifier: str, goal_id: list[int]) -> None:
        """Retain the action server's actual accepted goal identity."""
        self.ledger.bind_goal(identifier, goal_id)

"""Resource-owned preparation contracts and pure physical-evidence validation.

Providers are registered by trusted application code. Scene documents contain
only identifiers and data; neither a document nor a verdict can install code.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

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
    for row in trajectory["points"]:
        stamp = row["time_from_start"]
        if (type(stamp["sec"]) is not int or stamp["sec"] < 0
                or type(stamp["nanosec"]) is not int or not 0 <= stamp["nanosec"] < 1_000_000_000):
            raise ValueError("Invalid joint trajectory timing")
        nanos = stamp["sec"] * 1_000_000_000 + stamp["nanosec"]
        if (nanos <= previous or previous == -1 and nanos != 0
                or len(row["positions"]) != len(joint_names)):
            raise ValueError("Joint trajectory point order or coverage is invalid")
        for field in ("positions", "velocities", "accelerations", "effort"):
            values = row.get(field, [])
            if len(values) not in (0, len(joint_names)) or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError("Joint trajectory values are invalid")
        previous = nanos
    if (previous <= 0 or type(trajectory["duration_ns"]) is not int or trajectory["duration_ns"] != previous
            or duration is not None and not math.isclose(previous / 1e9, duration, abs_tol=1e-9)):
        raise ValueError("Joint trajectory duration disagrees with its interval")


def trajectory_record(trajectory, joint_names: list[str] | None = None) -> dict:
    """Preserve every native trajectory point, derivative and nanosecond."""
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

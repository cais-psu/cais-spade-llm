"""MoveIt checks of composed robot motions on a capture's recorded model and scene.

Composition-only. A check runs the installed native ``preview_recovery_motion``
in its file mode with the capture's own ``native_preview_model/parameters.yaml``
on a private planning scene projected from the capture: every robot at its
carried joints, held parts attached at their grasp, other parts at their
projected poses, the fingers of every robot at the recorded custody joint values
of its declared gripper state, and the scene's declared mounting contacts.
Contact between two robots is always checked: no allowance between two robots is
ever added.

The waypoints of a move are the ones the recovery controllers fly from the
projected start: ``GazeboPickPlaceController`` for the ur5e arms and the
``kmr_gazebo`` worker for the KMR arm. A KMR result checks one IK path (MoveIt's
``computeCartesianPath``), not the worker's own ``/compute_ik`` path.

Nothing here writes a value of a recovery event. A check accepts a motion with
the end joints MoveIt reached, or refuses it with the measured fact.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import struct
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# MoveIt's one-argument MaxEEFStep(step), which the ur5e controller's GetCartesianPath call uses, turns at
# most 3.5 x the linear step per sample (the live move_group logs count 0.035 rad steps for 0.01 m).
_UR5E_ROTATION_PER_LINEAR_STEP = 3.5
# GazeboPickPlaceController._move_xy_direct keeps the observed orientation for these robot names when the
# stated one is within 0.02 rad. The recovery runs use prewarmed controllers whose robot_name is the
# resource id.
_ORIENTATION_SNAP_ROBOTS = ("ur5e-1", "ur5e-2")
_ORIENTATION_SNAP_RAD = 0.02
# GazeboPickPlaceController.move_cartesian flies one straight segment when the start XY equals the
# target XY within 1e-6 m (_move_pose_direct).
_SAME_XY_M = 1e-6
# move_to_named_pose ends only when every arm joint is within 0.02 rad (wrapped) of the named joints.
_NAMED_POSE_TOLERANCE_RAD = 0.02
# kmr_gazebo.part_collision attaches a KMR payload to the TCP link with these links besides the TCP.
_KMR_PAYLOAD_TOUCH_LINKS = ("rg2_base_link", "rg2_left_inner_finger", "rg2_right_inner_finger")
# Reasons preview_recovery_motion.cpp gives when it refuses the motion itself, as opposed to a request
# it could not read.
_NATIVE_MOTION_REFUSALS = (
    "Cartesian path is incomplete",
    "Cartesian endpoint cannot be refined",
    "Cartesian endpoint refinement did not converge",
    "Refined Cartesian path exceeds the declared joint step",
    "Starting configuration is in collision",
    "Starting joint configuration violates model bounds",
    "Completed trajectory collision check failed",
    "Collision contact inventory is incomplete or truncated",
)
# The native refinement's stop when it cannot bring MoveIt's IK state onto a waypoint to 1e-13 m and rad.
_REFINEMENT_STOP = "Cartesian endpoint cannot be refined within the declared bounds and collision checks"
KMR_IK_PATH_LABEL = (
    "a check of one IK path: MoveIt computeCartesianPath on the captured model; the KMR worker solves "
    "its own path with /compute_ik within 0.35 rad joint windows")
PLACE_ARRIVAL_STATEMENT = (
    "allowed for this request by the planning scene's allowed collision matrix; the allowance does not "
    "bound the contact depth, which is bounded by the grounded target pose")
_STATE_RESOURCE_FIELDS = frozenset(("joint_positions", "gripper_state", "held_part", "grasp_transform",
                                    "base_pose"))
_NATIVE_TIMEOUT_S = 120.0
_MODEL_LIMIT = 2
_RESULT_LIMIT = 512
_MODELS: OrderedDict[tuple[str, str], CapturedMotionModel] = OrderedDict()
_MODEL_KEYS: dict[tuple[str, int, int], tuple[str, str]] = {}
_RESULTS: OrderedDict[tuple[str, str], dict] = OrderedDict()


class MissingMotionContext(ValueError):
    """A recorded input the motion check needs is absent from the capture."""


class MotionCheckError(RuntimeError):
    """The native check could not run on the request it was given."""


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _pose_message(pose: Sequence[float]) -> dict:
    return {"position": dict(zip("xyz", map(float, pose[:3]), strict=True)),
            "orientation": dict(zip("xyzw", map(float, pose[3:]), strict=True))}


def _normalized(pose: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in pose[3:]))
    return [*map(float, pose[:3]), *(float(value) / norm for value in pose[3:])]


def _wrapped(angle: float) -> float:
    return abs(math.atan2(math.sin(angle), math.cos(angle)))


def _dot(first: Sequence[float], second: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(first, second, strict=True))


def _down_angle(quaternion: Sequence[float]) -> float:
    """Angle between the tool z axis and straight down."""
    from cais_spade_llm.recovery_framework.geometry import rotate

    return math.acos(max(-1.0, min(1.0, -rotate(list(quaternion), [0., 0., 1.])[2])))


def _xyz(values: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(value):.4f}" for value in values) + ")"


class CapturedMotionModel:
    """One capture's recorded model, scene and robot settings, loaded once for many checks.

    Every part is represented by its recorded collision: the SDF producer's identity when the captured
    MoveIt scene holds it, the KMR's released-object identity when a recorded completed KMR release put
    it there (``kmr_recovery.recorded_scene_payloads``), otherwise the recorded physics collision under
    the recorded identity. A part without a recorded collision raises ``MissingMotionContext``.

    Attributes:
        observation_path: The capture's ``observations.json``.
        run_id: The capture's run id.
        observations: The capture as ``read_capture`` merges it.
        archive: The verified native model companion (``read_native_preview_model``).
        scene_capture: The captured MoveIt scene with an identity object for each part it lacks.
        physical: Physical evidence with scene object and support collision bindings.
        mounted_acm: The captured collision matrix with the declared mounting contacts.
        roots: Recorded root transforms of every robot.
        arms: Settings of each ur5e arm of ``inputs.scene.robots``, keyed by resource id.
        kmr: Settings of the KMR of ``inputs.scene.KMR``, or None.
        identity_evidence: How each part the captured scene lacks under its recorded identity was found.
    """

    def __init__(self, observation_path: str | Path) -> None:
        """Load and verify the capture, its native model and its scene bindings.

        Args:
            observation_path: The capture's ``observations.json``.
        """
        from cais_spade_llm.recovery_framework.capture_files import read_capture
        from cais_spade_llm.recovery_framework.recovery_motion_preview import (
            _native_preview_executable,
            apply_declared_mounting_contacts,
            build_scene_object_bindings,
            prepare_recorded_root_transforms,
        )
        from cais_spade_llm.recovery_framework.recovery_observations import (
            _file_sha256,
            read_native_preview_model,
        )

        self.observation_path = Path(observation_path).resolve()
        self.observations = read_capture(self.observation_path)
        self.run_id = self.observations.get("run_id")
        self.archive = read_native_preview_model(self.observation_path, self.observations)
        captured = self.observations["planning_scene_capture"]
        self.model_fingerprint = captured["model_fingerprint"]
        self.capture_fingerprint = self.archive["binding"]["observation_fingerprint"]
        self.parameters_path = self.archive["parameters_path"]
        self.executable = _native_preview_executable()
        self.executable_sha256 = _file_sha256(self.executable)
        physical = self.observations["physical_evidence"]
        _check_geometry_sources(physical)
        scene_inputs = self.observations["inputs"]["scene"]
        acm_names = captured["planning_scene"]["allowed_collision_matrix"]["entry_names"]
        description = ET.fromstring(captured["robot_description"])
        self.robot_links = frozenset(link.get("name") for link in description.findall("link"))
        self.arms = _arm_settings(physical, scene_inputs, acm_names)
        self.kmr = _kmr_settings(physical, scene_inputs, description)
        payload, templates, self.identity_evidence = self._part_identities(physical, captured)
        self.scene_capture = _with_world_objects(captured, templates)
        targets = {rid: arm["attachment_target"] for rid, arm in self.arms.items()}
        if self.kmr is not None:
            targets[self.kmr["resource_id"]] = self.kmr["attachment_target"]
        bindings = build_scene_object_bindings(self.scene_capture, physical, payload_evidence=payload,
                                               attachment_targets=targets)
        self.physical = {**physical, "scene_object_bindings": bindings,
                         "support_collision_object_bindings": self._support_bindings(physical)}
        declarations = (scene_inputs.get("planning_scene_mounting_contacts")
                        or self.archive.get("supplementary_model_declarations"))
        if declarations is None:
            raise MissingMotionContext("The capture records no planning_scene_mounting_contacts")
        # Mounting contacts change only the collision matrix, between robot bases and fixed plates that no
        # projection moves, so the matrix is computed once for every check scene of this capture.
        mounted = apply_declared_mounting_contacts(
            captured["planning_scene"], robot_description=captured["robot_description"],
            robots=scene_inputs["robots"], physical_evidence=physical, declarations=declarations)
        self.mounted_acm = mounted["planning_scene"]["allowed_collision_matrix"]
        self.mounting_contacts = [row["pair"] for row in mounted["evidence"]["contacts"]]
        self.roots = prepare_recorded_root_transforms(captured, physical)["recorded_root_transforms"]
        joint_state = captured["planning_scene"]["robot_state"]["joint_state"]
        self.captured_joints = dict(zip(joint_state["name"], joint_state["position"], strict=True))
        scene = self.scene_capture["planning_scene"]
        self.scene_objects = {row["id"]: row for row in scene["world"]["collision_objects"]}
        self.scene_objects.update((row["object"]["id"], row["object"])
                                  for row in scene["robot_state"]["attached_collision_objects"])
        # Converted messages by the id of the dict (or mesh list) they came from, and the messages whose
        # encoding the fingerprint may reuse; both hold their objects, so the ids stay unique.
        self._messages: dict[int, tuple[Any, Any]] = {}
        self._kept: dict[int, Any] = {}
        self._bytes: dict[int, bytes] = {}
        for row in scene["world"]["collision_objects"]:
            message = self._converted_object(row)
            self._messages[id(row)] = (row, message)
            self._kept[id(message)] = message
        self._check_lean_scene()

    def object_message(self, row: Mapping) -> Any:
        """The ROS message of one collision object; captured fixtures and part meshes are converted once.

        Args:
            row: A collision object of a check scene.

        Returns:
            Its ``moveit_msgs/CollisionObject``.
        """
        cached = self._messages.get(id(row))
        if cached is not None and cached[0] is row:
            return cached[1]
        return self._converted_object(row)

    def message_bytes(self, message: Any) -> bytes:
        """The bytes ``_scene_message_fingerprint`` hashes for a message, reusing those of converted fixtures.

        Args:
            message: A ROS message of a check scene.

        Returns:
            The field-value encoding the native check fingerprints.
        """
        return _encoded_message(message, self._bytes, self._kept)

    def _converted_object(self, row: Mapping) -> Any:
        from moveit_msgs.msg import CollisionObject

        from cais_spade_llm.recovery_framework.recovery_motion_preview import (
            _set_recorded_message_fields,
        )

        message = CollisionObject()
        _set_recorded_message_fields(message, {name: value for name, value in row.items() if name != "meshes"})
        meshes = row.get("meshes") or []
        if meshes:
            cached = self._messages.get(id(meshes))
            if cached is None or cached[0] is not meshes:
                converted = CollisionObject()
                _set_recorded_message_fields(converted, {"meshes": meshes})
                cached = (meshes, list(converted.meshes))
                self._messages[id(meshes)] = cached
                self._kept.update((id(mesh), mesh) for mesh in cached[1])
            message.meshes = cached[1]
        return message

    def _check_lean_scene(self) -> None:
        """Refuse the model unless its lean scene is project_preview_scene's, with the native fingerprint."""
        from cais_spade_llm.recovery_framework.recovery_motion_preview import (
            _scene_message_fingerprint,
            project_preview_scene,
        )

        snapshot = projected_snapshot(self, {})
        expected = project_preview_scene(self.scene_capture, {**self.physical, "snapshot": snapshot})
        actual = _project_scene(self, snapshot)
        if actual != expected["planning_scene"]:
            raise MotionCheckError("The lean scene projection differs from project_preview_scene")
        message, fingerprint = _scene_message(self, actual)
        if fingerprint != _scene_message_fingerprint(message):
            raise MotionCheckError("The scene fingerprint differs from _scene_message_fingerprint")

    @property
    def robot_ids(self) -> list[str]:
        """Resource ids of every robot whose joints the check scene carries."""
        return [*self.arms, *([self.kmr["resource_id"]] if self.kmr else [])]

    def configuration(self, resource_id: str) -> dict:
        """Return the recorded kinematic chain of one robot.

        Args:
            resource_id: The robot.

        Returns:
            ``physical_evidence.resource_models[resource_id].configuration``.
        """
        return self.physical["resource_models"][resource_id]["configuration"]

    def _part_identities(self, physical: Mapping, captured: Mapping) -> tuple[dict, list, list]:
        """Payload evidence for every part, identity objects for absent ones, and how they were found."""
        from cais_spade_llm.recovery_framework.kmr_recovery import recorded_scene_payloads

        scene = captured["planning_scene"]
        present = {row["id"] for row in scene["world"]["collision_objects"]}
        present.update(row["object"]["id"] for row in scene["robot_state"]["attached_collision_objects"])
        released = recorded_scene_payloads(self.observations, captured)
        frame = physical["snapshot"]["frame"]
        collisions = {collision["id"] for body in physical["physics_evidence"]["models"].values()
                      for link in body.get("links", {}).values() for collision in link.get("collisions", [])}
        payload, templates, evidence = {}, [], []
        for part, row in physical["parts"].items():
            identity = row.get("collision_scene_identity") or {}
            objects = identity.get("collision_objects") or []
            if not objects:
                raise MissingMotionContext(f"{part} records no collision_scene_identity")
            missing = [item for item in objects if item["id"] not in present]
            payload[f"parts/{part}"] = identity
            if not missing:
                continue
            if part in released:
                payload[f"parts/{part}"] = released[part]
                evidence.append({"part": part, "source": released[part]["source"], "scene_object_ids": [
                    item["id"] for item in released[part]["collision_objects"]]})
                continue
            for item in missing:
                collision_id = "::".join(str(item.get(key)) for key in ("model", "link", "collision"))
                if collision_id not in collisions:
                    raise MissingMotionContext(
                        f"{part}: the captured MoveIt scene has no object {item['id']} and the capture records "
                        f"no physics collision {collision_id}")
                # Identity holder only: build_scene_object_bindings takes the shape from the recorded physics
                # collision, and project_preview_scene replaces this object with the projected part.
                templates.append({"header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": frame},
                                  "pose": _pose_message([0., 0., 0., 0., 0., 0., 1.]), "id": item["id"],
                                  "type": {"key": "", "db": ""}, "primitives": [], "primitive_poses": [],
                                  "meshes": [], "mesh_poses": [], "planes": [], "plane_poses": [],
                                  "subframe_names": [], "subframe_poses": [], "operation": 0})
                evidence.append({"part": part, "scene_object_ids": [item["id"]],
                                 "source": f"recorded physics collision {collision_id}; the captured MoveIt "
                                           f"scene has no object {item['id']}"})
        return payload, templates, evidence

    def _support_bindings(self, physical: Mapping) -> dict[str, dict]:
        """MoveIt object of each recorded static collision, by the SDF producer's identity.

        ``geometry.collision_boxes`` names the object of collision ``model::link::collision``
        ``model/link/collision``. Only objects the captured scene holds exactly once are bound.
        """
        world = [row["id"] for row in self.scene_capture["planning_scene"]["world"]["collision_objects"]]
        bindings = {}
        for name, body in physical["physics_evidence"]["models"].items():
            if body.get("static") is not True:
                continue
            for link in body.get("links", {}).values():
                for collision in link.get("collisions", []):
                    object_id = "/".join(collision["id"].split("::"))
                    if world.count(object_id) == 1:
                        bindings[collision["id"]] = {"object_id": object_id, "source": {
                            "calculation": "SDF collision producer identity (geometry.collision_boxes)",
                            "physics_model": name, "physics_collision_id": collision["id"]}}
        return bindings


def _check_geometry_sources(physical: Mapping) -> None:
    """Refuse a capture whose robot collision meshes changed on disk since it was recorded."""
    from cais_spade_llm.recovery_framework.recovery_observations import _file_sha256

    sources: dict[str, str] = {}
    for row in physical.get("resources", {}).values():
        recorded = (row.get("continuous_motion_configuration") or {}).get("geometry_sources", {})
        for name, digest in recorded.items():
            if sources.setdefault(name, digest) != digest:
                raise MotionCheckError("Captured geometry source fingerprints conflict: " + name)
    for name, digest in sources.items():
        if _file_sha256(Path(name)) != digest:
            raise MotionCheckError("Native preview geometry source changed: " + name)


def _arm_settings(physical: Mapping, scene_inputs: Mapping, acm_names: Sequence[str]) -> dict[str, dict]:
    """Recorded controller settings of each ur5e arm.

    The attachment mirrors GazeboPickPlaceController._sync_part_collision: the payload is attached to
    ``payload_collision.observation_link`` with the collision matrix names that start with the robot's
    ``rg2`` prefix plus ``attach.attach_link_candidates`` as touch links.
    """
    arms = {}
    for row in scene_inputs["robots"]:
        rid = row["resource_id"]
        model = physical["resource_models"][rid]
        primitive = model["primitive_model"]["configuration"]
        controller = primitive["controller_configuration"]
        payload = controller["payload_collision"]
        touch = [name for name in acm_names if name.startswith(payload["robot_prefix"] + "rg2")]
        touch.extend(controller["attach"]["attach_link_candidates"])
        arms[rid] = {
            "move_group": copy.deepcopy(controller["move_group"]),
            "cartesian_motion": copy.deepcopy(controller["cartesian_motion"]),
            "named_joint_poses": copy.deepcopy(model.get("named_joint_poses") or {}),
            "custody": copy.deepcopy((primitive.get("custody") or {}).get("joint_positions") or {}),
            "attachment_target": {"link_name": payload["observation_link"], "touch_links": touch},
        }
    return arms


def _kmr_settings(physical: Mapping, scene_inputs: Mapping, description: ET.Element) -> dict | None:
    """Recorded worker settings of the KMR arm, or None when the scene has no KMR."""
    declaration = scene_inputs.get("KMR")
    model = physical.get("resource_models", {}).get("KMR")
    if not isinstance(declaration, dict) or not model or "configuration" not in model:
        return None
    execution = declaration["task_execution"]
    primitive = model["primitive_model"]["configuration"]
    tcp = execution["tcp_link"]
    return {
        "resource_id": "KMR",
        "task_execution": copy.deepcopy(execution),
        "custody_by_part": copy.deepcopy(primitive.get("custody_by_part") or {}),
        "attachment_target": {"link_name": tcp, "touch_links": [tcp, *_KMR_PAYLOAD_TOUCH_LINKS]},
        "base_joints": _base_joints(description, model["configuration"]["root"]),
    }


def _base_joints(description: ET.Element, root: str) -> list[str]:
    """The movable URDF joints between the world and a mobile robot's root link: x, y and yaw, in order."""
    parents = {joint.find("child").get("link"): joint for joint in description.findall("joint")}
    chain, link = [], root
    while link in parents:
        joint = parents[link]
        if joint.get("type") != "fixed":
            chain.insert(0, joint)
        link = joint.find("parent").get("link")
    shape = [(joint.get("type"), [float(value) for value in joint.find("axis").get("xyz", "1 0 0").split()])
             for joint in chain]
    planar = (len(shape) == 3 and shape[0] == ("prismatic", [1., 0., 0.])
              and shape[1] == ("prismatic", [0., 1., 0.])
              and shape[2][0] in ("continuous", "revolute") and shape[2][1] == [0., 0., 1.])
    if not planar:
        raise MotionCheckError(f"The mobile base of {root} is not a planar x, y and yaw joint chain: {shape}")
    return [joint.get("name") for joint in chain]


def _with_world_objects(captured: Mapping, objects: list[dict]) -> dict:
    if not objects:
        return dict(captured)
    scene = captured["planning_scene"]
    world = {**scene["world"], "collision_objects": [*scene["world"]["collision_objects"], *objects]}
    return {**captured, "planning_scene": {**scene, "world": world}}


def load_captured_motion_model(observation_path: str | Path) -> CapturedMotionModel:
    """Return the loaded model of one capture, from a small cache keyed by run id and model fingerprint.

    Args:
        observation_path: The capture's ``observations.json``.

    Returns:
        The verified model shared by every check of that capture.
    """
    path = Path(observation_path).resolve()
    stat = path.stat()
    file_key = (str(path), stat.st_mtime_ns, stat.st_size)
    key = _MODEL_KEYS.get(file_key)
    model = _MODELS.get(key) if key else None
    if model is None:
        started = time.monotonic()
        model = CapturedMotionModel(path)
        logger.info("Loaded the motion check model of run %s in %.1f s", model.run_id,
                    time.monotonic() - started)
        key = (model.run_id, model.model_fingerprint)
        _MODEL_KEYS[file_key] = key
        _MODELS[key] = model
        while len(_MODELS) > _MODEL_LIMIT:
            _MODELS.popitem(last=False)
    _MODELS.move_to_end(key)
    return model


def clear_result_cache() -> None:
    """Forget every cached native result, so the next checks run MoveIt again."""
    _RESULTS.clear()


def robot_pose(model: CapturedMotionModel, resource_id: str, joints: Sequence[float], *,
               base_pose: Sequence[float] | None = None) -> list[float]:
    """Return the reference link pose of a robot's recorded chain at the given joints.

    Args:
        model: The capture's model.
        resource_id: The robot.
        joints: Its arm joint values, in the chain's joint order.
        base_pose: The KMR base pose ``[x, y, z, qx, qy, qz, qw]`` when it differs from the capture.

    Returns:
        ``[x, y, z, qx, qy, qz, qw]`` of the reference link (tool0, or the KMR TCP) in the world frame.
    """
    from scipy.spatial.transform import Rotation

    from cais_spade_llm.resources.kinematic_reach import chain_frames

    configuration = model.configuration(resource_id)
    if base_pose is not None:
        configuration = {**configuration, "root_xyz": list(base_pose[:3]),
                         "root_quaternion": list(base_pose[3:])}
    names = configuration["joint_names"]
    if len(joints) != len(names):
        raise ValueError(f"{resource_id} needs {len(names)} joint values, not {len(joints)}")
    frame = chain_frames(configuration, {name: float(value) for name, value in zip(names, joints, strict=True)})[
        configuration["reference_link"]]
    return [*map(float, frame[:3, 3]), *map(float, Rotation.from_matrix(frame[:3, :3]).as_quat())]


def projected_snapshot(model: CapturedMotionModel, state: Mapping) -> dict:
    """Apply a composition's projected state to the captured snapshot.

    Args:
        model: The capture's model.
        state: ``{"resources": {rid: {...}}, "parts": {part: {"current_pose": [...]}}}``. A robot may state
            ``joint_positions``, ``gripper_state``, ``held_part``, ``grasp_transform`` and, for the KMR,
            ``base_pose`` as ``[x, y, z, qx, qy, qz, qw]``. Anything left out keeps its captured value.

    Returns:
        A snapshot whose robot poses are FK of their joints and whose held parts sit at pose x grasp.
    """
    from cais_spade_llm.resources.primitive_observations_3d import _compose, _unit_pose

    captured = model.physical["snapshot"]
    resources = dict(captured["resources"])
    parts = {name: dict(row) for name, row in captured["parts"].items()}
    given = state.get("resources") or {}
    unknown = sorted(set(given) - set(model.robot_ids))
    if unknown:
        raise ValueError(f"The check scene carries no robot joints for {unknown}")
    kmr_id = model.kmr["resource_id"] if model.kmr else None
    for rid in model.robot_ids:
        stated = dict(given.get(rid) or {})
        if set(stated) - _STATE_RESOURCE_FIELDS or ("base_pose" in stated and rid != kmr_id):
            raise ValueError(f"Unsupported projected fields for {rid}: {sorted(stated)}")
        row = {**captured["resources"][rid], **copy.deepcopy(stated)}
        if row.get("held_part") is None:
            row["grasp_transform"] = None
        row["joint_positions"] = [float(value) for value in row["joint_positions"]]
        base = _planar_base_pose(row["base_pose"]) if rid == kmr_id else None
        row["current_pose"] = robot_pose(model, rid, row["joint_positions"], base_pose=base)
        resources[rid] = row
    holders: dict[str, str] = {}
    for rid in model.robot_ids:
        part = resources[rid].get("held_part")
        if part is None:
            continue
        if part in holders or part not in parts:
            raise ValueError(f"{part} cannot be held by {rid}: held by {holders.get(part)} or unknown")
        if resources[rid].get("grasp_transform") is None:
            raise MissingMotionContext(f"{rid} holds {part} without a grasp_transform")
        holders[part] = rid
        pose = _compose(_unit_pose(resources[rid]["current_pose"], f"{rid} pose"),
                        _unit_pose(resources[rid]["grasp_transform"], f"{rid} grasp_transform"))
        parts[part].update(current_pose=[float(value) for value in pose], contained_by=None)
    for part, row in (state.get("parts") or {}).items():
        if part not in parts or set(row) != {"current_pose"}:
            raise ValueError(f"Unsupported projected part {part}: {sorted(row)}")
        if part in holders:
            raise ValueError(f"{part} is held by {holders[part]}; its pose follows the grasp")
        parts[part]["current_pose"] = [float(value) for value in row["current_pose"]]
    return {**captured, "resources": resources, "parts": parts}


def composition_scene_state(model: CapturedMotionModel, views: Mapping[str, Mapping],
                            observed_parts: Mapping[str, Mapping]) -> dict:
    """The state of one composition step, as the accepted composition prefix and the program leave the cell.

    Each robot stands at its carried joints (``resource_evidence.snapshot.joint_positions``) with its declared
    ``held_part`` and ``gripper_state``, holding its part by the grasp its ``resource_evidence.attachment``
    records, and the KMR on its ``base_pose``; whatever a view leaves out keeps the value the capture recorded,
    so a robot no program has moved stands at its captured joints. Each part a program put down (an
    ``observed_parts`` row of source kind ``projected``) and no robot holds stands at its projected pose.

    Args:
        model: The capture's model.
        views: Per robot resource id, its composition view: ``held_part``, ``gripper_state``, ``base_pose`` and
            ``resource_evidence``.
        observed_parts: The parts as the acting owner's evidence records them.

    Returns:
        The state ``projected_snapshot`` and ``check_robot_moves`` take.

    Raises:
        MissingMotionContext: A projected part pose is not a complete pose.
    """
    kmr_id = model.kmr["resource_id"] if model.kmr else None
    resources = {}
    for rid, view in views.items():
        if rid not in model.robot_ids:
            continue
        evidence = view.get("resource_evidence") or {}
        snapshot = evidence.get("snapshot") or {}
        row = {field: copy.deepcopy(view[field]) for field in ("held_part", "gripper_state") if field in view}
        if isinstance(snapshot.get("joint_positions"), list):
            row["joint_positions"] = [float(value) for value in snapshot["joint_positions"]]
        attachment = evidence.get("attachment") or {}
        # A robot's attachment names its part as part_name; the KMR's recorded one as held_part.
        if row.get("held_part") is not None and row["held_part"] in (attachment.get("part_name"),
                                                                    attachment.get("held_part")) \
                and attachment.get("grasp_transform"):
            row["grasp_transform"] = [float(value) for value in attachment["grasp_transform"]]
        base = snapshot.get("base_pose") or view.get("base_pose")
        if rid == kmr_id and base is not None:
            row["base_pose"] = [float(value) for value in base]
        resources[rid] = row
    held = {row.get("held_part") for row in projected_snapshot(model, {"resources": resources})["resources"].values()}
    parts = {}
    for name, row in observed_parts.items():
        if (row.get("source") or {}).get("kind") != "projected" or name in held:
            continue
        pose = row.get("observed_pose", row.get("current_pose"))
        if isinstance(pose, Mapping):
            pose = [pose.get(axis) for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")]
        if (not isinstance(pose, list) or len(pose) != 7 or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in pose)):
            raise MissingMotionContext(f"{name}'s projected pose is not a complete pose: {pose}")
        parts[name] = {"current_pose": [float(value) for value in pose]}
    return {"resources": resources, "parts": parts}


def motion_end_pose(model: CapturedMotionModel, state: Mapping, resource_id: str, move: Mapping) -> list[float] | None:
    """The tool pose a robot's motion ends at, as its controller routes it from the state.

    Args:
        model: The capture's model.
        state: The projected state at the motion's start.
        resource_id: The moving robot.
        move: ``{"primitive", "params"}`` with resolved parameters.

    Returns:
        ``[x, y, z, qx, qy, qz, qw]`` of the reference link at the route's last waypoint (the start when the
        controller sends no motion), or None when the controller refuses the motion.
    """
    row = projected_snapshot(model, state)["resources"][resource_id]
    if model.kmr and resource_id == model.kmr["resource_id"]:
        route = kmr_motion(model, move["primitive"], move["params"], start_pose=row["current_pose"],
                           base_pose=row["base_pose"], held_part=row.get("held_part"))
    else:
        route = arm_motion(model, resource_id, move["primitive"], move["params"], start_pose=row["current_pose"],
                           start_joints=row["joint_positions"], held_part=row.get("held_part"))
    if route.get("refusal"):
        return None
    return [float(value) for value in (route["waypoints"][-1] if route["waypoints"] else row["current_pose"])]


def part_rest_support(model: CapturedMotionModel, part_name: str, part_pose: Sequence[float]) -> dict:
    """Where a part set down at a pose rests on the capture's recorded static geometry.

    Args:
        model: The capture's model.
        part_name: The part.
        part_pose: Its pose ``[x, y, z, qx, qy, qz, qw]``.

    Returns:
        ``projected_part_support``'s result: ``status`` ``supported`` with its ``support_collisions``, or the reason
        no recorded support holds the part there.
    """
    from cais_spade_llm.resources.recovery_geometry import projected_part_support

    model_name = model.physical["snapshot"]["parts"][part_name]["model_name"]
    state = {"parts": {part_name: {"model_name": model_name, "current_pose": [float(value) for value in part_pose]}}}
    return projected_part_support(part_name=part_name, state=state, evidence=model.physical, contact_layer_tolerance=True)


def _planar_base_pose(pose: Sequence[float]) -> list[float]:
    values = [float(value) for value in pose]
    if len(values) != 7 or abs(values[2]) > 1e-9 or abs(values[3]) > 1e-9 or abs(values[4]) > 1e-9:
        raise ValueError(f"The KMR base pose must be planar [x, y, 0, 0, 0, qz, qw]: {values}")
    return values


def _finger_values(model: CapturedMotionModel, resource_id: str, row: Mapping) -> tuple[dict, str]:
    """Recorded finger joint values of a robot's declared gripper state, and their source."""
    state = row.get("gripper_state")
    if resource_id in model.arms:
        values = model.arms[resource_id]["custody"].get(state)
        source = f"custody.joint_positions.{state}"
    else:
        custody = model.kmr["custody_by_part"]
        part = row.get("held_part") if row.get("held_part") in custody else "default"
        values = (custody.get(part) or {}).get("joint_positions", {}).get(state)
        source = f"custody_by_part.{part}.joint_positions.{state}"
    if not values:
        return {}, f"captured finger joints (gripper_state {state!r} records no custody joint values)"
    return values, source


def _base_joint_values(model: CapturedMotionModel, row: Mapping) -> list[float] | None:
    """KMR base joints of a projected base pose, or None when the base stands where it was captured."""
    base = _planar_base_pose(row["base_pose"])
    names = model.kmr["base_joints"]
    captured = [model.captured_joints[name] for name in names]
    yaw = math.atan2(2 * base[5] * base[6], 1 - 2 * base[5] ** 2)
    yaw += 2 * math.pi * round((captured[2] - yaw) / (2 * math.pi))
    values = [base[0], base[1], yaw]
    if all(abs(value - recorded) <= 1e-12 for value, recorded in zip(values, captured, strict=True)):
        return None
    return values


def _with_allowed_pairs(matrix: Mapping, pairs: Sequence[Sequence[str]]) -> dict:
    """Copy a collision matrix and allow the given pairs, adding names it lacks."""
    result = copy.deepcopy(dict(matrix))
    names, rows = result["entry_names"], result["entry_values"]
    for pair in pairs:
        for name in pair:
            if name not in names:
                for row in rows:
                    row["enabled"].append(False)
                names.append(name)
                rows.append({"enabled": [False] * len(names)})
        first, second = (names.index(name) for name in pair)
        rows[first]["enabled"][second] = rows[second]["enabled"][first] = True
    return result


def motion_scene(model: CapturedMotionModel, state: Mapping, *,
                 allowed_contacts: Sequence[Mapping] = ()) -> dict:
    """Build the private planning scene of one check.

    Args:
        model: The capture's model.
        state: The composition's projected state (see ``projected_snapshot``).
        allowed_contacts: Contact records (``place_arrival_contact``) whose pairs this scene allows.

    Returns:
        ``planning_scene``, ``start_state``, ``scene_fingerprint``, ``recorded_root_transforms``, the
        projected ``snapshot``, the ROS ``message`` and the scene ``evidence``.
    """
    snapshot = projected_snapshot(model, state)
    scene = _project_scene(model, snapshot)
    pairs = [list(pair) for record in allowed_contacts for pair in record["pairs"]]
    if model.robot_links.intersection(name for pair in pairs for name in pair):
        raise ValueError(f"A contact allowance never names a robot link: {pairs}")
    scene["allowed_collision_matrix"] = _with_allowed_pairs(model.mounted_acm, pairs)
    joint_state = scene["robot_state"]["joint_state"]
    joint_state["position"] = list(joint_state["position"])
    index = {name: position for position, name in enumerate(joint_state["name"])}
    fingers = {}
    for rid in model.robot_ids:
        values, fingers[rid] = _finger_values(model, rid, snapshot["resources"][rid])
        for name, value in values.items():
            joint_state["position"][index[name]] = float(value)
    roots, base_joints = model.roots, None
    if model.kmr is not None:
        base_joints = _base_joint_values(model, snapshot["resources"][model.kmr["resource_id"]])
        if base_joints is not None:
            for name, value in zip(model.kmr["base_joints"], base_joints, strict=True):
                joint_state["position"][index[name]] = value
            # The recorded KMR root is a moving root: the native check only compares it with the FK of the
            # base joints, so a moved base leaves it out instead of stating a root no one recorded.
            root = model.configuration(model.kmr["resource_id"])["root"]
            roots = [row for row in roots if row["child_frame_id"] != root]
    message, fingerprint = _scene_message(model, scene)
    return {"planning_scene": scene, "start_state": scene["robot_state"],
            "scene_fingerprint": fingerprint, "recorded_root_transforms": roots, "snapshot": snapshot,
            "message": message,
            "evidence": {"fingers": fingers, "kmr_base_joints": base_joints, "allowed_pairs": pairs,
                         "mounting_contacts": model.mounting_contacts,
                         "identity_evidence": model.identity_evidence}}


def _project_scene(model: CapturedMotionModel, snapshot: Mapping) -> dict:
    """``project_preview_scene`` of a snapshot, sharing captured fixtures and part shapes instead of copying them.

    Each step is the repository builder's: the robots' configured and frozen joints, then every part's
    recorded objects at the part pose, or attached at ``link_to_reference`` x grasp when a robot holds it.
    Copying the 4 MB of captured meshes for every check is what this avoids; the model checks at load that
    both builders give the same scene.
    """
    from cais_spade_llm.resources.primitive_observations_3d import _compose, _unit_pose

    captured = model.scene_capture["planning_scene"]
    joint_state = captured["robot_state"]["joint_state"]
    positions = list(joint_state["position"])
    index = {name: position for position, name in enumerate(joint_state["name"])}
    resources = snapshot["resources"]
    for name, value in _projected_joint_values(model, resources).items():
        positions[index[name]] = value
    bindings = model.physical["scene_object_bindings"]
    replaced, world, attached = set(), [], []
    for part, part_state in snapshot["parts"].items():
        binding = bindings.get(part)
        if not isinstance(binding, dict) or not binding.get("source") or not binding.get("objects"):
            raise ValueError(f"Missing scene_object_bindings for {part}")
        holders = [rid for rid, row in resources.items() if row.get("held_part") == part]
        if len(holders) > 1 or (holders and part_state.get("contained_by") is not None):
            raise ValueError(f"Conflicting projected custody for {part}")
        attachment = binding.get("attachments", {}).get(holders[0]) if holders else None
        if holders:
            row = resources[holders[0]]
            if not isinstance(attachment, dict) or not attachment.get("link_name"):
                raise ValueError(f"Missing attachment declaration for {holders[0]} and {part}")
            carried = _compose(_unit_pose(row["current_pose"], "projected resource current_pose"),
                               _unit_pose(row["grasp_transform"], "projected grasp_transform"))
            pose = _unit_pose(part_state["current_pose"], "projected part current_pose")
            if any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(carried, pose, strict=True)):
                raise ValueError(f"Projected part pose conflicts with its attachment for {part}")
            parent = _compose(_unit_pose(attachment["link_to_reference"], "attachment link_to_reference"),
                              _unit_pose(row["grasp_transform"], "projected grasp_transform"))
            parent_frame = attachment["link_name"]
        else:
            parent = _unit_pose(part_state["current_pose"], "projected part current_pose")
            parent_frame = snapshot["frame"]
        for object_binding in binding["objects"]:
            projected = _projected_object(model, object_binding, parent, parent_frame)
            replaced.add(projected["id"])
            if attachment is None:
                world.append(projected)
            else:
                attached.append({"link_name": parent_frame, "touch_links": list(attachment["touch_links"]),
                                 "object": projected})
    robot_state = {**captured["robot_state"], "joint_state": {**joint_state, "position": positions},
                   "attached_collision_objects": [
                       row for row in captured["robot_state"]["attached_collision_objects"]
                       if row["object"]["id"] not in replaced] + attached}
    objects = [row for row in captured["world"]["collision_objects"] if row["id"] not in replaced] + world
    return {**captured, "robot_state": robot_state, "world": {**captured["world"], "collision_objects": objects}}


def _projected_joint_values(model: CapturedMotionModel, resources: Mapping) -> dict[str, float]:
    """Configured and frozen joint values of every recorded chain, as project_preview_scene assigns them."""
    assignments: dict[str, float] = {}
    for rid, recorded in model.physical.get("resource_models", {}).items():
        configuration = recorded.get("configuration")
        if configuration is None:
            continue
        row = resources.get(rid, {})
        names, values = row.get("joint_names"), row.get("joint_positions")
        if names != configuration.get("joint_names") or not isinstance(values, list) or len(names) != len(values):
            raise ValueError(f"Projected joint identities differ from the owner model for {rid}")
        pairs = list(zip(names, values, strict=True))
        for joint in configuration.get("joints", []):
            if "frozen_position" in joint:
                pairs.append((joint["name"], joint["frozen_position"]))
            elif "frozen_interval" in joint:
                low, high = joint["frozen_interval"]
                if low != high:
                    raise ValueError(f"Projected joint {joint['name']} has no exact configuration")
                pairs.append((joint["name"], low))
        for name, value in pairs:
            if (name not in model.captured_joints or isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"Projected joint is missing or invalid: {name}")
            if assignments.setdefault(name, value) != value:
                raise ValueError(f"Conflicting projected joint evidence: {name}")
    return assignments


def _projected_object(model: CapturedMotionModel, object_binding: Mapping, parent: tuple,
                      parent_frame: str) -> dict:
    """One part-local collision object composed onto its parent, its shapes shared with the binding."""
    from cais_spade_llm.resources.primitive_observations_3d import _compose, _unit_pose

    template = object_binding["part_local_object"]
    object_id = object_binding["object_id"]
    if template.get("id") != object_id or template.get("subframe_names") or template.get("subframe_poses"):
        raise ValueError(f"Missing or conflicting collision object mapping for {object_id}")
    pose = template.get("pose")
    local = (0., 0., 0., 0., 0., 0., 1.) if pose is None else _unit_pose(
        [*(pose["position"][axis] for axis in "xyz"), *(pose["orientation"][axis] for axis in "xyzw")],
        "collision object pose")
    component_parent = _compose(parent, local)
    projected = dict(template)
    for poses in ("primitive_poses", "mesh_poses", "plane_poses"):
        result = []
        for row in template.get(poses, []):
            composed = _compose(component_parent, _unit_pose(
                [*(row["position"][axis] for axis in "xyz"), *(row["orientation"][axis] for axis in "xyzw")],
                "part-local collision pose"))
            result.append(_pose_message([float(value) for value in composed]))
        projected[poses] = result
    projected["header"] = {**copy.deepcopy(model.scene_objects[object_id].get("header", {})),
                           "frame_id": parent_frame}
    if "pose" in projected:
        projected["pose"] = _pose_message([0., 0., 0., 0., 0., 0., 1.])
    projected["operation"] = 0
    return projected


def _scene_message(model: CapturedMotionModel, scene: Mapping) -> tuple[Any, str]:
    """Convert a check scene to its ROS message, reusing converted fixtures and meshes, and fingerprint it."""
    from moveit_msgs.msg import (
        AttachedCollisionObject,
        PlanningScene,
        PlanningSceneWorld,
        RobotState,
    )

    from cais_spade_llm.recovery_framework.recovery_motion_preview import (
        _set_recorded_message_fields,
    )

    message = PlanningScene()
    _set_recorded_message_fields(message, {name: value for name, value in scene.items()
                                           if name not in ("robot_state", "world")})
    state = RobotState()
    _set_recorded_message_fields(state, {name: value for name, value in scene["robot_state"].items()
                                         if name != "attached_collision_objects"})
    rows = []
    for row in scene["robot_state"]["attached_collision_objects"]:
        attached = AttachedCollisionObject()
        _set_recorded_message_fields(attached, {name: value for name, value in row.items() if name != "object"})
        attached.object = model.object_message(row["object"])
        rows.append(attached)
    state.attached_collision_objects = rows
    message.robot_state = state
    world = PlanningSceneWorld()
    _set_recorded_message_fields(world, {name: value for name, value in scene["world"].items()
                                         if name != "collision_objects"})
    world.collision_objects = [model.object_message(row) for row in scene["world"]["collision_objects"]]
    message.world = world
    return message, hashlib.sha256(model.message_bytes(message)).hexdigest()


_SCALAR_FORMATS = {"boolean": "?", "octet": "B", "char": "B", "uint8": "B", "int8": "b", "uint16": "H",
                   "int16": "h", "uint32": "I", "int32": "i", "uint64": "Q", "int64": "q", "float": "f",
                   "double": "d"}


def _encoded_message(message: Any, memo: dict[int, bytes], kept: Mapping[int, Any]) -> bytes:
    """_scene_message_fingerprint's field encoding of one message, memoized for messages the model keeps."""
    cached = memo.get(id(message))
    if cached is not None:
        return cached
    out = bytearray()
    for name, descriptor in zip(message.get_fields_and_field_types(), message.SLOT_TYPES, strict=True):
        _encode_field(getattr(message, name), descriptor, memo, kept, out)
    data = bytes(out)
    if id(message) in kept:
        memo[id(message)] = data
    return data


def _encode_field(value: Any, descriptor: Any, memo: dict[int, bytes], kept: Mapping[int, Any],
                  out: bytearray) -> None:
    from rosidl_parser.definition import (
        AbstractSequence,
        AbstractString,
        Array,
        BasicType,
        NamespacedType,
    )

    if isinstance(descriptor, (AbstractSequence, Array)):
        out += struct.pack("<Q", len(value))
        item = descriptor.value_type
        if isinstance(item, BasicType) and item.typename not in {"char", "octet"}:
            out += struct.pack(f"<{len(value)}{_SCALAR_FORMATS[item.typename]}", *value)
            return
        for element in value:
            _encode_field(element, item, memo, kept, out)
    elif isinstance(descriptor, BasicType):
        if descriptor.typename in {"char", "octet"} and isinstance(value, (str, bytes)):
            value = ord(value)
        out += struct.pack("<" + _SCALAR_FORMATS[descriptor.typename], value)
    elif isinstance(descriptor, AbstractString):
        data = value.encode("utf-8")
        out += struct.pack("<Q", len(data))
        out += data
    elif isinstance(descriptor, NamespacedType):
        out += _encoded_message(value, memo, kept)
    else:
        raise MotionCheckError("Unsupported ROS field type in the scene fingerprint")


def _stated_quaternion(params: Mapping) -> tuple[list[float] | None, str | None]:
    """The stated orientation normalized as GazeboPickPlaceController._normalized_optional_quaternion does."""
    values = [params.get(key) for key in ("qx", "qy", "qz", "qw")]
    supplied = [value is not None for value in values]
    if not any(supplied):
        return None, None
    if not all(supplied):
        return None, "orientation requires qx, qy, qz, and qw together"
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in values):
        return None, "orientation quaternion must contain finite numeric values"
    norm = math.hypot(*values)
    if norm <= 1e-12:
        return None, "orientation quaternion must be non-zero"
    return [float(value) / norm for value in values], None


def _arm_route(model: CapturedMotionModel, resource_id: str, start: list[float],
               target: list[float]) -> list[list[float]]:
    """Poses GazeboPickPlaceController._move_xy_direct sends under cartesian_motion.only."""
    from cais_spade_llm.resources.robot.cartesian_waypoints import fixed_orientation_waypoints

    settings = model.arms[resource_id]["cartesian_motion"]
    target = list(target)
    if (resource_id in _ORIENTATION_SNAP_ROBOTS
            and abs(_dot(start[3:], target[3:])) >= math.cos(_ORIENTATION_SNAP_RAD / 2)):
        target[3:] = start[3:]
    return fixed_orientation_waypoints(start, target, [],
                                       rotation_point=settings.get("rotation_waypoint", start[:3]),
                                       angular_step=float(settings.get("angular_step_rad", .05)))


def _nearest_named_joints(model: CapturedMotionModel, resource_id: str, positions: Sequence[float],
                          start: Sequence[float]) -> list[float]:
    """Named joints at the 2*pi turns nearest the start, within the chain's bounds.

    Mirrors GazeboPickPlaceController._nearest_simulation_joint_targets.
    """
    configuration = model.configuration(resource_id)
    bounds = {row["name"]: row["position_bounds"] for row in configuration["joints"] if "position_bounds" in row}
    targets = []
    for name, nominal, observed in zip(configuration["joint_names"], positions, start, strict=True):
        lower, upper = bounds[name]
        if not lower <= nominal <= upper:
            raise ValueError(f"Target exceeds joint limits: {name}")
        turns = round((observed - nominal) / (2 * math.pi))
        turns = max(turns, math.ceil((lower - nominal) / (2 * math.pi)))
        turns = min(turns, math.floor((upper - nominal) / (2 * math.pi)))
        targets.append(float(nominal + turns * 2 * math.pi))
    return targets


def arm_motion(model: CapturedMotionModel, resource_id: str, primitive: str, params: Mapping, *,
               start_pose: Sequence[float], start_joints: Sequence[float] | None = None,
               held_part: str | None = None) -> dict:
    """Waypoints a ur5e controller flies for one recovery motion, from the given joints.

    The controller's tests run on FK of the joints: same XY within 1e-6 m flies ``[target]``; otherwise
    ``fixed_orientation_waypoints`` turns at ``cartesian_motion.rotation_waypoint`` after the ur5e-1/ur5e-2
    0.02 rad orientation snap. ``move_relative`` is one segment at the start orientation. A named pose flies
    to FK of the named joints nearest the start (ur5e-4's preserved-orientation home keeps the start
    orientation) and must end within 0.02 rad of those joints.

    Args:
        model: The capture's model.
        resource_id: The ur5e arm.
        primitive: ``move_cartesian``, ``move_relative`` or ``move_to_named_pose``.
        params: The step's resolved parameters.
        start_pose: The tool pose the motion starts from (FK of the carried joints).
        start_joints: The joints the motion starts from; a named pose needs them.
        held_part: The part the arm holds, if any.

    Returns:
        ``waypoints`` (``[x, y, z, qx, qy, qz, qw]`` poses, empty when the controller sends no motion),
        ``named_joints`` for a named pose's end check, and ``refusal`` when the controller refuses the step.
    """
    start = [float(value) for value in start_pose]
    if primitive == "move_cartesian":
        quaternion, error = _stated_quaternion(params)
        if error:
            return {"waypoints": [], "refusal": f"move_cartesian {error}"}
        target = [*(float(params[axis]) for axis in "xyz"), *(quaternion or start[3:])]
        if all(abs(start[axis] - target[axis]) <= _SAME_XY_M for axis in (0, 1)):
            return {"waypoints": [target], "route": "same XY: one segment (_move_pose_direct)"}
        return {"waypoints": _arm_route(model, resource_id, start, target),
                "route": "_move_xy_direct with the rotation waypoint"}
    if primitive == "move_relative":
        if any(isinstance(params.get(key), bool) or not isinstance(params.get(key), (int, float))
               for key in ("dx", "dy", "dz")):
            return {"waypoints": [], "refusal": "move_relative requires numeric dx, dy and dz"}
        target = [start[0] + params["dx"], start[1] + params["dy"], start[2] + params["dz"], *start[3:]]
        return {"waypoints": [[float(value) for value in target]], "route": "one segment at the start orientation"}
    if primitive == "move_to_named_pose":
        if start_joints is None:
            raise ValueError("A named pose's route depends on the joints it starts from")
        return _named_arm_motion(model, resource_id, params.get("pose_name"), start, start_joints, held_part)
    raise ValueError(f"{primitive} is not a ur5e motion primitive")


def _named_arm_motion(model: CapturedMotionModel, resource_id: str, pose_name: Any, start: list[float],
                      start_joints: Sequence[float], held_part: str | None) -> dict:
    arm = model.arms[resource_id]
    named = arm["named_joint_poses"].get(pose_name)
    if not named:
        return {"waypoints": [], "refusal": f"unknown pose '{pose_name}'; "
                                            f"available={sorted(arm['named_joint_poses'])}"}
    configuration = model.configuration(resource_id)
    if named.get("joint_names", configuration["joint_names"]) != configuration["joint_names"]:
        raise MissingMotionContext(f"{resource_id} named pose {pose_name} names other joints")
    positions = [float(value) for value in named["positions"]]
    preserve = pose_name == "home" and arm["cartesian_motion"].get("home_preserve_orientation") is True
    if not preserve and all(_wrapped(value - target) <= _NAMED_POSE_TOLERANCE_RAD
                            for value, target in zip(start_joints, positions, strict=True)):
        return {"waypoints": [], "route": f"already within {_NAMED_POSE_TOLERANCE_RAD} rad of {pose_name}"}
    targets = _nearest_named_joints(model, resource_id, positions, start_joints)
    target = robot_pose(model, resource_id, targets)
    if preserve:
        if held_part is not None:
            return {"waypoints": [], "refusal": f"Cartesian home requires an empty gripper; {resource_id} "
                                                f"holds {held_part}"}
        angle = _down_angle(start[3:])
        if angle > _NAMED_POSE_TOLERANCE_RAD:
            return {"waypoints": [], "refusal": f"Cartesian home requires a downward gripper; the tool axis "
                                                f"is {angle:.4f} rad from facing down"}
        target = [*target[:3], *start[3:]]
    waypoints = _arm_route(model, resource_id, start, target)
    return {"waypoints": waypoints, "named_joints": None if preserve else targets,
            "route": f"_move_xy_direct to FK of {pose_name}" + (" at the start orientation" if preserve else "")}


def kmr_motion(model: CapturedMotionModel, primitive: str, params: Mapping, *, start_pose: Sequence[float],
               base_pose: Sequence[float], held_part: str | None = None) -> dict:
    """Waypoints the kmr_gazebo worker flies for one KMR arm motion, from the given joints.

    ``move_cartesian`` with ``waypoints`` (or a seed) flies ``[*waypoints, target]``; without them it must
    keep the TCP orientation within the worker's tolerance and turns, if at all, at the transport TCP.
    ``move_relative`` adds its offset at the current orientation. ``transport`` follows the worker's
    transfer arc and ``home`` (gripper opened first) goes straight to the transport pose.

    Args:
        model: The capture's model.
        primitive: ``move_cartesian``, ``move_relative`` or ``move_to_named_pose``.
        params: The step's resolved parameters.
        start_pose: The TCP pose the motion starts from (FK of the carried joints on the base).
        base_pose: The KMR base pose ``[x, y, z, qx, qy, qz, qw]``.
        held_part: The part the KMR holds, if any.

    Returns:
        ``waypoints``, the worker's ``polyline`` for its path check, ``gripper_state`` when the motion
        opens the gripper first, and ``refusal`` when the worker refuses the step.
    """
    from cais_spade_llm.recovery_framework.kmr_recovery import transfer_waypoints, transport_pose

    execution = model.kmr["task_execution"]
    tolerance = execution["orientation_tolerance_rad"]
    scene = model.observations["inputs"]["scene"]
    base = _planar_base_pose(base_pose)
    start = [float(value) for value in start_pose]
    if primitive == "move_relative":
        offset = params.get("offset")
        if (not isinstance(offset, list) or len(offset) != 3
                or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in offset)):
            return {"waypoints": [], "refusal": "KMR move_relative requires three finite distances"}
        primitive, params = "move_cartesian", {"target": [*(a + b for a, b in zip(start[:3], offset, strict=True)),
                                                          *start[3:]]}
    if primitive == "move_cartesian":
        target = [float(value) for value in params["target"]]
        waypoints = params.get("waypoints")
        if waypoints is None and params.get("seed") is None:
            if abs(_dot(start[3:], target[3:])) < math.cos(tolerance / 2):
                return {"waypoints": [], "refusal": "Cartesian segment must preserve the observed TCP orientation"}
            return _kmr_path(model, base, start, target, None, shortcut=False)
        return _kmr_path(model, base, start, target, waypoints, shortcut=True)
    if primitive == "move_to_named_pose":
        pose_name = params.get("pose_name")
        if pose_name == "home":
            if held_part is not None:
                return {"waypoints": [], "refusal": "KMR downward home requires an empty gripper"}
            route = _kmr_path(model, base, start, transport_pose(scene, base, start), None, shortcut=True)
            return {**route, "gripper_state": "open"}
        if pose_name == "transport":
            transport = transport_pose(scene, base, start)
            return _kmr_path(model, base, start, transport, transfer_waypoints(scene, base, start, transport),
                             shortcut=True)
        return {"waypoints": [], "refusal": "Unknown KMR arm pose"}
    raise ValueError(f"{primitive} is not a KMR arm motion primitive")


def _kmr_path(model: CapturedMotionModel, base: list[float], start: list[float], target: list[float],
              waypoints: Sequence | None, *, shortcut: bool) -> dict:
    """Targets of kmr_gazebo plan_motion (and move_to_pose's no-motion shortcut)."""
    from cais_spade_llm.recovery_framework.geometry import compose
    from cais_spade_llm.recovery_framework.kmr_recovery import transport_pose
    from cais_spade_llm.resources.robot.cartesian_waypoints import fixed_orientation_waypoints

    execution = model.kmr["task_execution"]
    tolerance = execution["orientation_tolerance_rad"]
    settings = execution["cartesian_waypoints"]
    if (shortcut and not waypoints and math.dist(start[:3], target[:3]) <= execution["position_tolerance_m"]
            and abs(_dot(start[3:], target[3:])) >= math.cos(tolerance / 2)):
        return {"waypoints": [], "route": "TCP already at the target within the worker's tolerance"}
    middle = settings.get("transfer_orientation_waypoint_xyzw_in_base_frame")
    if middle is not None:
        middle = compose(base, [0., 0., 0., *middle])[3:]
    scene = model.observations["inputs"]["scene"]
    if waypoints:
        targets = [[float(value) for value in pose] for pose in waypoints] + [list(target)]
    else:
        targets = fixed_orientation_waypoints(start, target, [],
                                              rotation_point=transport_pose(scene, base, start)[:3],
                                              angular_step=settings.get("angular_step_rad", .05),
                                              rotation_midpoint=middle) or [list(target)]
    for pose in [start, *targets]:
        if (len(pose) != 7 or not all(math.isfinite(value) for value in pose)
                or not math.isclose(sum(value * value for value in pose[3:]), 1., abs_tol=1e-5)):
            return {"waypoints": [], "refusal": "KMR Cartesian waypoints require finite XYZ and a unit quaternion"}
        angle = _down_angle(pose[3:])
        if angle > tolerance:
            return {"waypoints": [], "refusal": f"KMR Cartesian waypoints must keep the gripper downward; "
                                                f"{_xyz(pose[:3])} turns the tool {angle:.4f} rad from down"}
    return {"waypoints": [_normalized(pose) for pose in targets], "polyline": [start, *targets],
            "route": "[*waypoints, target]" if waypoints else "fixed orientation, turning at the transport TCP"}


def _motion_request(model: CapturedMotionModel, scene: Mapping, resource_id: str,
                    waypoints: Sequence[Sequence[float]], *, support_departure: Mapping | None = None,
                    mating: Mapping | None = None) -> dict:
    """One native Cartesian request with the controller's own group, link and step settings."""
    from cais_spade_llm.recovery_framework.recovery_motion_preview import (
        bind_mating_contact,
        bind_support_departure,
    )

    if resource_id in model.arms:
        arm = model.arms[resource_id]
        group, settings = arm["move_group"], arm["cartesian_motion"]
        linear = float(settings["linear_step_m"])
        motion = {"group_name": group["group_name"], "link_name": group["ee_link"],
                  "frame_id": group["frame_id"], "max_translation_step": linear,
                  "max_rotation_step": _UR5E_ROTATION_PER_LINEAR_STEP * linear,
                  "max_joint_step": float(settings["max_joint_step_rad"])}
    else:
        execution = model.kmr["task_execution"]
        settings = execution["cartesian_waypoints"]
        motion = {"group_name": execution["planning_group"], "link_name": execution["tcp_link"],
                  "frame_id": "world", "max_translation_step": float(settings["linear_step_m"]),
                  "max_rotation_step": float(settings.get("angular_step_rad", .05)),
                  "max_joint_step": float(settings["max_joint_step_rad"])}
    motion.update(resource_id=resource_id, mode="cartesian",
                  waypoints=[_pose_message(_normalized(pose)) for pose in waypoints])
    # The proposal binding names the checked motion itself, so an unchanged motion in an unchanged scene is
    # one request, whichever program it belongs to.
    proposal = _fingerprint({**motion, "scene_fingerprint": scene["scene_fingerprint"],
                             "recorded_root_transforms": scene["recorded_root_transforms"],
                             "support_departure": support_departure, "mating": mating})
    request = {**motion, "planning_scene": scene["planning_scene"], "start_state": scene["start_state"],
               "joint_target": {}, "recorded_root_transforms": scene["recorded_root_transforms"],
               "capture_fingerprint": model.capture_fingerprint, "proposal_fingerprint": proposal,
               "scene_fingerprint": scene["scene_fingerprint"], "model_fingerprint": model.model_fingerprint,
               "velocity_scaling": 1.0, "acceleration_scaling": 1.0, "timeout_seconds": _NATIVE_TIMEOUT_S}
    # The binders validate a copy of the request; the robot state is all of the scene they read.
    light = {**request, "planning_scene": {"is_diff": scene["planning_scene"]["is_diff"],
                                           "robot_state": scene["start_state"]}}
    if support_departure is not None:
        bound = bind_support_departure(light, support_departure)
        request.update({key: bound[key] for key in ("support_departure_json", "support_departure_fingerprint")})
    if mating is not None:
        bound = bind_mating_contact(light, mating)
        request.update({key: bound[key] for key in ("mating_contact_json", "mating_contact_fingerprint")})
    return request


def _run_request(model: CapturedMotionModel, scene: Mapping, request: Mapping, joint_names: Sequence[str]) -> dict:
    """Run one request through the native file mode, check it with preview_result_record and summarize it."""
    from cais_lab_robotics.action import PreviewRecoveryMotion
    from rclpy.serialization import deserialize_message, serialize_message
    from rosidl_runtime_py.convert import message_to_ordereddict

    from cais_spade_llm.recovery_framework.recovery_motion_preview import (
        _set_recorded_message_fields,
        preview_result_record,
        validate_preview_request,
    )

    motion = {name: value for name, value in request.items() if name not in ("planning_scene", "start_state")}
    key = (scene["scene_fingerprint"], _fingerprint(motion))
    cached = _RESULTS.get(key)
    if cached is not None:
        _RESULTS.move_to_end(key)
        return {**copy.deepcopy(cached), "cached": True}
    began = time.monotonic()
    if request.get("mode") != "cartesian" or request.get("scene_endpoint_json"):
        raise MotionCheckError("The composed motion runner sends Cartesian requests only")
    # In Cartesian mode validate_preview_request and preview_result_record read no world object. They get
    # the scene's robot state only, so they do not copy the captured meshes; the native check receives the
    # complete scene message, which scene_fingerprint binds.
    prepared = validate_preview_request({**request, "planning_scene": {
        "is_diff": request["planning_scene"]["is_diff"], "robot_state": request["start_state"]}})
    goal = PreviewRecoveryMotion.Goal()
    _set_recorded_message_fields(goal, {name: value for name, value in prepared.items()
                                        if name not in ("planning_scene", "start_state")})
    goal.planning_scene = scene["message"]
    goal.start_state = scene["message"].robot_state
    with tempfile.TemporaryDirectory(prefix="composed_motion_") as directory:
        goal_path, result_path = Path(directory) / "goal.cdr", Path(directory) / "result.cdr"
        log_path = Path(directory) / "native.log"
        goal_path.write_bytes(serialize_message(goal))
        prepared_at = time.monotonic()
        with log_path.open("w") as log:
            try:
                completed = subprocess.run(
                    [str(model.executable), "--input-goal", str(goal_path), "--output-result", str(result_path),
                     "--ros-args", "--params-file", model.parameters_path],
                    stdout=log, stderr=subprocess.STDOUT, timeout=_NATIVE_TIMEOUT_S + 30., check=False)
            except subprocess.TimeoutExpired as exc:
                raise MotionCheckError(f"The native check exceeded {_NATIVE_TIMEOUT_S + 30.} s") from exc
        process_seconds = time.monotonic() - prepared_at
        if completed.returncode != 0 or not result_path.is_file():
            raise MotionCheckError("Native preview failed: " + log_path.read_text()[-2000:])
        raw = message_to_ordereddict(deserialize_message(result_path.read_bytes(), PreviewRecoveryMotion.Result))
    record = preview_result_record(prepared, raw)
    summary = _result_summary(record, joint_names)
    summary["timing"] = {"prepare_seconds": prepared_at - began, "process_seconds": process_seconds,
                         "native_seconds": float(record.get("elapsed_seconds") or 0.),
                         "total_seconds": time.monotonic() - began}
    _RESULTS[key] = copy.deepcopy(summary)
    while len(_RESULTS) > _RESULT_LIMIT:
        _RESULTS.popitem(last=False)
    return {**summary, "cached": False}


def _joints(joint_state: Mapping, names: Sequence[str]) -> list[float]:
    values = dict(zip(joint_state["name"], joint_state["position"], strict=True))
    return [float(values[name]) for name in names]


def _result_summary(record: Mapping, joint_names: Sequence[str]) -> dict:
    """The parts of a checked native result that the composition check reads."""
    diagnostics = record.get("cartesian_diagnostics") or {}
    waypoints = []
    for row in diagnostics.get("waypoints", []):
        waypoint = {key: row.get(key) for key in ("index", "fraction", "returned_sample_count")}
        for key in ("starting_state", "last_accepted_state"):
            if row.get(key):
                pose = row[key]["link_pose"]
                waypoint[key] = {"joints": _joints(row[key]["joint_state"], joint_names),
                                 "link_pose": [*pose["position"], *pose["orientation"]]}
        waypoints.append(waypoint)
    queries = diagnostics.get("ik_queries") or []
    trajectory = record.get("joint_trajectory") or {}
    names = list(trajectory.get("joint_names") or [])
    positions = [[float(value) for value in point["positions"]] for point in trajectory.get("points", [])]
    if names and names != list(joint_names):
        positions = [[row[names.index(name)] for name in joint_names] for row in positions]
    end_state = (record.get("end_state") or {}).get("joint_state") or {}
    collision = record.get("collision_rejection_evidence")
    return {
        "status": record.get("status"), "reason": record.get("reason"), "fraction": record.get("fraction"),
        "checked_states": record.get("checked_states"), "path_complete": record.get("path_complete"),
        "positions": positions, "end_joints": _joints(end_state, joint_names) if end_state.get("name") else None,
        "waypoints": waypoints, "failure_kind": diagnostics.get("failure_kind"),
        "last_ik_query": {key: queries[-1].get(key) for key in ("success", "error_code")} if queries else None,
        "collision": None if collision is None else {
            key: copy.deepcopy(collision.get(key))
            for key in ("phase", "state_index", "kind", "reason", "contacts")},
        "support_departure_evidence": copy.deepcopy(record.get("support_departure_evidence")),
    }


def check_robot_moves(model: CapturedMotionModel, state: Mapping, resource_id: str, moves: Sequence[Mapping], *,
                      allowed_contacts: Sequence[Mapping] = (), support_departure: Mapping | None = None,
                      mating: Mapping | None = None) -> dict:
    """Check consecutive motions of one robot, whose custody and gripper do not change, with MoveIt.

    The motions fly in one request, except that a named pose starts a new request, because its route
    depends on the joints it starts from. Every other robot stands at its joints in ``state``. Where only the
    native endpoint refinement stops at a joint bound, the path goes on from MoveIt's IK state
    (``_continued_path``).

    Args:
        model: The capture's model.
        state: The composition's projected state at the first motion (see ``projected_snapshot``).
        resource_id: The moving robot.
        moves: ``{"step_index", "primitive", "params"}`` rows in program order, with resolved parameters.
        allowed_contacts: ``place_arrival_contact`` records allowed in every request of this call.
        support_departure: A ``support_departure_record`` bound to the first request.
        mating: A ``mating_contact_record`` bound to the request; the call then holds the insertion only.

    Returns:
        ``status`` (FEASIBLE, INFEASIBLE or NEEDS_CONTEXT), ``reason``, the refused ``step_index``, the
        completed ``steps`` with their end joints, the robot's ``end_joints``, one row per native request and
        the ``refinement_stops`` the path went on from.
    """
    try:
        return _check_moves(model, state, resource_id, list(moves), {
            "allowed_contacts": list(allowed_contacts), "support_departure": support_departure, "mating": mating})
    except MissingMotionContext as exc:
        return {"status": "NEEDS_CONTEXT", "reason": str(exc), "step_index": None, "steps": [],
                "end_joints": None, "requests": []}


def _check_moves(model: CapturedMotionModel, state: Mapping, resource_id: str, moves: list[Mapping],
                 contacts: dict) -> dict:
    snapshot = projected_snapshot(model, state)
    row = snapshot["resources"][resource_id]
    joints, held, gripper = list(row["joint_positions"]), row.get("held_part"), row.get("gripper_state")
    base = _planar_base_pose(row["base_pose"]) if model.kmr and resource_id == model.kmr["resource_id"] else None
    result = {"status": "FEASIBLE", "reason": "", "step_index": None, "steps": [], "end_joints": joints,
              "requests": []}
    pending: list[dict] = []
    pose = robot_pose(model, resource_id, joints, base_pose=base)

    def flush() -> bool:
        nonlocal joints, pose, pending
        if not pending:
            return True
        resources = {**(state.get("resources") or {})}
        resources[resource_id] = {**(resources.get(resource_id) or {}), "joint_positions": joints,
                                  "gripper_state": gripper}
        refusal, joints = _run_batch(model, {**state, "resources": resources}, resource_id, pending, joints,
                                     contacts, result)
        contacts["support_departure"] = None
        pending = []
        pose = robot_pose(model, resource_id, joints, base_pose=base)
        if refusal is not None:
            result.update(status="INFEASIBLE", reason=refusal["reason"], step_index=refusal["step_index"])
        result["end_joints"] = joints
        return refusal is None

    for move in moves:
        named = move["primitive"] == "move_to_named_pose"
        if named and pending and not flush():
            return result
        if base is None:
            route = arm_motion(model, resource_id, move["primitive"], move["params"], start_pose=pose,
                               start_joints=joints if named else None, held_part=held)
        else:
            route = kmr_motion(model, move["primitive"], move["params"], start_pose=pose, base_pose=base,
                               held_part=held)
        if route.get("refusal"):
            if flush():
                prefix = f"{move['primitive']} step {move['step_index']} ({resource_id})"
                result.update(status="INFEASIBLE", step_index=move["step_index"],
                              reason=f"{prefix}: {route['refusal']}")
            return result
        if route.get("gripper_state"):
            if pending and not flush():
                return result
            gripper = route["gripper_state"]
        pending.append({**move, **route, "pose_name": move["params"].get("pose_name")})
        if route["waypoints"]:
            pose = route["waypoints"][-1]
    flush()
    return result


def _run_batch(model: CapturedMotionModel, state: Mapping, resource_id: str, pending: list[dict],
               joints: list[float], contacts: dict, result: dict) -> tuple[dict | None, list[float]]:
    """Fly one request of consecutive motions; record each completed step; return a refusal and end joints."""
    names = model.configuration(resource_id)["joint_names"]
    base = None
    if model.kmr and resource_id == model.kmr["resource_id"]:
        base = _planar_base_pose(projected_snapshot(model, state)["resources"][resource_id]["base_pose"])
    ranges, count = [], 0
    for move in pending:
        ranges.append((count, count + len(move["waypoints"])))
        count += len(move["waypoints"])
    if count == 0:
        for move in pending:
            result["steps"].append(_step_row(model, resource_id, move, joints, base))
            refusal = _named_pose_refusal(model, resource_id, move, joints)
            if refusal is not None:
                return {"step_index": move["step_index"], "reason": refusal}, joints
        return None, joints
    waypoints = [pose for move in pending for pose in move["waypoints"]]
    if contacts["allowed_contacts"] and len(waypoints) > 1:
        refusal = _approach_refusal(model, state, resource_id, pending, (joints, base), contacts, result)
        if refusal is not None:
            return refusal
    scene = motion_scene(model, state, allowed_contacts=contacts["allowed_contacts"])
    request = _motion_request(model, scene, resource_id, waypoints,
                              support_departure=contacts["support_departure"], mating=contacts["mating"])
    summary = _run_request(model, scene, request, names)
    result["requests"].append(_request_row(model, resource_id, pending, scene, request, summary))
    summary = _continued_path(model, state, resource_id, waypoints, summary, joints=joints, contacts=contacts,
                              rows=result["requests"], pending=pending)
    result.setdefault("refinement_stops", []).extend(summary.get("refinement_stops") or [])
    if summary["status"] != "PREVIEW_COMPLETE":
        return _native_refusal(model, resource_id, pending, ranges, (joints, base), summary, result)
    ends = [row["starting_state"]["joints"] for row in summary["waypoints"][1:]] + [summary["end_joints"]]
    start_index, end = 0, joints
    for move, (first, last) in zip(pending, ranges, strict=True):
        start = end
        end = ends[last - 1] if last > first else end
        refusal = _named_pose_refusal(model, resource_id, move, end)
        if refusal is None and base is not None and last > first:
            stop = _state_index(summary["positions"], end, start_index)
            refusal = _kmr_path_refusal(model, move, summary["positions"][start_index:stop + 1], base)
            start_index = stop
        if refusal is not None:
            return {"step_index": move["step_index"], "reason": refusal}, start
        result["steps"].append(_step_row(model, resource_id, move, end, base))
    return None, end


def _approach_refusal(model: CapturedMotionModel, state: Mapping, resource_id: str, pending: list[dict],
                      start: tuple, contacts: dict, result: dict) -> tuple[dict, list[float]] | None:
    """Check a place arrival's path up to its last waypoint with no contact allowed.

    A place arrival's allowed contact (``place_arrival_contact``) is the part on its support at the motion's end.
    It covers only the segment that arrives there: a waypoint before it that takes the part into the support, or
    into anything else, is refused by that contact here.

    Returns:
        The refusal and the joints the path reached, or None when the path to the last waypoint is free.
    """
    joints, _ = start
    names = model.configuration(resource_id)["joint_names"]
    approach = [dict(move) for move in pending]
    last = max(index for index, move in enumerate(approach) if move["waypoints"])
    approach[last]["waypoints"] = approach[last]["waypoints"][:-1]
    ranges, count = [], 0
    for move in approach:
        ranges.append((count, count + len(move["waypoints"])))
        count += len(move["waypoints"])
    waypoints = [pose for move in approach for pose in move["waypoints"]]
    plain = {**contacts, "allowed_contacts": []}
    scene = motion_scene(model, state)
    request = _motion_request(model, scene, resource_id, waypoints,
                              support_departure=contacts["support_departure"], mating=contacts["mating"])
    summary = _run_request(model, scene, request, names)
    row = _request_row(model, resource_id, approach, scene, request, summary)
    row["before_place_arrival"] = True
    result["requests"].append(row)
    summary = _continued_path(model, state, resource_id, waypoints, summary, joints=joints, contacts=plain,
                              rows=result["requests"], pending=approach)
    result.setdefault("refinement_stops", []).extend(summary.get("refinement_stops") or [])
    if summary["status"] == "PREVIEW_COMPLETE":
        return None
    return _native_refusal(model, resource_id, approach, ranges, start, summary, result)


def _continued_path(model: CapturedMotionModel, state: Mapping, resource_id: str, waypoints: list, summary: dict, *,
                    joints: list[float], contacts: dict, rows: list, pending: list[dict]) -> dict:
    """Take MoveIt's IK state where only the native endpoint refinement stops at a joint bound, and go on from it.

    ``preview_recovery_motion`` refines MoveIt's IK state onto each waypoint to 1e-13 m and rad. Where MoveIt's
    Cartesian path reached a waypoint (fraction 1) with a joint on its bound, that refinement cannot move the joint
    further and the request stops, although every state of the path was checked within the bounds and free of
    collision, and the state lies within MoveIt's IK tolerance of the waypoint. The path then reaches the waypoint
    at that state and goes on from it, in a new request, through the remaining waypoints. A request with a mating
    record is not continued: its insertion corridor is judged over the whole insertion. Every native request stays
    in ``rows`` and each stop is listed with its residual.

    Args:
        model: The capture's model.
        state: The projected state at the batch's start.
        resource_id: The moving robot.
        waypoints: Every waypoint of the batch, in order.
        summary: The batch's first native result.
        joints: The robot's joints at the batch's start.
        contacts: The batch's allowed contacts and the records bound to its first request.
        rows: The call's request rows; each continuation adds its own.
        pending: The batch's motions.

    Returns:
        ``summary`` when no refinement stop was taken; otherwise one summary of the whole path, with
        ``refinement_stops``: each stopped request's known states (every waypoint's start and the state MoveIt reached
        at the stop; the states between lie on its Cartesian interpolation and were checked by the native check),
        then the last request's own, or the last stop's state when it was the batch's last waypoint.
    """
    names = model.configuration(resource_id)["joint_names"]
    positions, done, stops, checked, offset, start, current = [], [], [], 0, 0, list(joints), summary
    while True:
        here = [{**row, "index": row["index"] + offset} for row in current["waypoints"]]
        stop = here[-1] if here else None
        reached = (stop or {}).get("last_accepted_state")
        bound = _joints_at_bounds(model, resource_id, reached["joints"]) if reached else []
        if not (bound and current["status"] != "PREVIEW_COMPLETE"
                and (current["reason"] or "").startswith(_REFINEMENT_STOP) and contacts["mating"] is None
                and float(stop.get("fraction") or 0.) >= 1. - 1e-9):
            break
        stops.append({"waypoint": stop["index"] + 1, "joints_at_bounds": bound,
                      "position_residual_m": math.dist(reached["link_pose"][:3], waypoints[stop["index"]][:3])})
        positions += [start, *(row["starting_state"]["joints"] for row in here[1:])]
        done += here
        checked += int(current.get("checked_states") or 0)
        start = list(reached["joints"])
        if stop["index"] + 1 == len(waypoints):
            return {**current, "status": "PREVIEW_COMPLETE", "waypoints": done, "positions": [*positions, start],
                    "end_joints": start, "checked_states": checked, "refinement_stops": stops}
        resources = {**(state.get("resources") or {})}
        resources[resource_id] = {**(resources.get(resource_id) or {}), "joint_positions": start}
        scene = motion_scene(model, {**state, "resources": resources}, allowed_contacts=contacts["allowed_contacts"])
        request = _motion_request(model, scene, resource_id, waypoints[stop["index"] + 1:])
        current = _run_request(model, scene, request, names)
        row = _request_row(model, resource_id, pending, scene, request, current)
        row["continued_from_waypoint"] = stop["index"] + 1
        rows.append(row)
        offset = stop["index"] + 1
    if not stops:
        return summary
    return {**current, "waypoints": done + here, "positions": positions + list(current["positions"]),
            "checked_states": checked + int(current.get("checked_states") or 0), "refinement_stops": stops}


def _state_index(positions: Sequence[Sequence[float]], joints: Sequence[float], start: int) -> int:
    """Index of the trajectory state at the given joints, searching forward from ``start``."""
    for index in range(start, len(positions)):
        if all(abs(a - b) <= 1e-12 for a, b in zip(positions[index], joints, strict=True)):
            return index
    raise MotionCheckError("A waypoint's end state is missing from the native trajectory")


def _step_row(model: CapturedMotionModel, resource_id: str, move: Mapping, end: Sequence[float],
              base: Sequence[float] | None) -> dict:
    return {"step_index": move["step_index"], "primitive": move["primitive"], "waypoints": len(move["waypoints"]),
            "route": move.get("route"), "end_joints": list(end),
            "end_pose": robot_pose(model, resource_id, end, base_pose=base)}


def _request_row(model: CapturedMotionModel, resource_id: str, pending: list[dict], scene: Mapping,
                 request: Mapping, summary: Mapping) -> dict:
    row = {"steps": [move["step_index"] for move in pending], "waypoints": len(request["waypoints"]),
           "status": summary["status"], "reason": summary["reason"], "fraction": summary["fraction"],
           "checked_states": summary["checked_states"], "cached": summary["cached"], **summary["timing"],
           "scene_fingerprint": scene["scene_fingerprint"],
           "proposal_fingerprint": request["proposal_fingerprint"],
           "allowed_pairs": scene["evidence"]["allowed_pairs"], "fingers": scene["evidence"]["fingers"],
           "kmr_base_joints": scene["evidence"]["kmr_base_joints"],
           "support_departure": bool(request.get("support_departure_json")),
           "mating": bool(request.get("mating_contact_json"))}
    if model.kmr and resource_id == model.kmr["resource_id"]:
        row["label"] = KMR_IK_PATH_LABEL
    return row


def _native_refusal(model: CapturedMotionModel, resource_id: str, pending: list[dict], ranges: list[tuple],
                    start: tuple, summary: Mapping, result: dict) -> tuple[dict, list[float]]:
    """Name the step where MoveIt refused the request, with the measured facts."""
    joints, base = start
    reason = summary["reason"] or ""
    if not any(reason.startswith(text) for text in _NATIVE_MOTION_REFUSALS):
        raise MotionCheckError("The native check refused the request itself: " + reason)
    # Interpolation stops and endpoint refinement failures happen at the last waypoint the diagnostics list;
    # a start collision or a whole-path check names no waypoint.
    stop = None
    if summary["waypoints"] and reason.startswith(("Cartesian path is incomplete", "Cartesian endpoint")):
        stop = summary["waypoints"][-1]["index"]
    end = joints
    for move, (first, last) in zip(pending, ranges, strict=True):
        if last == first or (stop is not None and last <= stop):
            if last > first:
                end = summary["waypoints"][last]["starting_state"]["joints"]
            result["steps"].append(_step_row(model, resource_id, move, end, base))
            continue
        local = stop - first if stop is not None else None
        text = _native_refusal_text(model, resource_id, move, summary, local)
        return {"step_index": move["step_index"], "reason": text}, end
    raise MotionCheckError("A native refusal names no motion of the request")


def _max_joint_step(model: CapturedMotionModel, resource_id: str) -> float:
    if resource_id in model.arms:
        return float(model.arms[resource_id]["cartesian_motion"]["max_joint_step_rad"])
    return float(model.kmr["task_execution"]["cartesian_waypoints"]["max_joint_step_rad"])


def _contacts_text(collision: Mapping) -> str:
    rows = []
    for pair in collision.get("contacts") or []:
        deepest = max(pair["contacts"], key=lambda contact: contact["depth"])
        rows.append(f"{pair['body_1']} <> {pair['body_2']} {deepest['depth'] * 1000:.4f} mm deep at "
                    f"{_xyz(deepest['position'])}")
    text = "; ".join(rows)
    if collision.get("kind") != "unrelated_collision":
        text = f"{collision.get('reason')}: {text}" if text else str(collision.get("reason"))
    return text


def _native_refusal_text(model: CapturedMotionModel, resource_id: str, move: Mapping, summary: Mapping,
                         local: int | None) -> str:
    prefix = f"{move['primitive']} step {move['step_index']} ({resource_id}): "
    collision = summary["collision"]
    native = f"Native reason: {summary['reason']}"
    if collision is not None and collision.get("phase") == "initial":
        return (prefix + "the start of this motion is in collision on the captured model: "
                + _contacts_text(collision) + ". " + native)
    facts = []
    if local is not None:
        waypoint = summary["waypoints"][-1]
        target = move["waypoints"][local]
        where = f"waypoint {local + 1} of {len(move['waypoints'])}"
        if summary["reason"].startswith("Cartesian endpoint"):
            facts.append(f"MoveIt's Cartesian path on the captured model reached {where} of this step, "
                         f"{_xyz(target[:3])}, but could not refine the joints onto it within the joint bounds "
                         "and step limits")
        else:
            fraction = 100 * float(waypoint.get("fraction") or 0.)
            facts.append(f"MoveIt's Cartesian path on the captured model stopped {fraction:.0f}% of the way "
                         f"along {where} toward {_xyz(target[:3])}")
        reached = waypoint.get("last_accepted_state") or waypoint.get("starting_state")
        if reached is not None:
            facts.extend(_joints_at_bounds(model, resource_id, reached["joints"]))
            position = reached["link_pose"][:3]
            text = f"the last reached tool pose is {_xyz(position)}"
            if resource_id in model.arms and collision is None:
                root = model.configuration(resource_id)["root_xyz"]
                axis = math.hypot(position[0] - root[0], position[1] - root[1])
                text += f", {axis:.3f} m from {resource_id}'s base axis"
            facts.append(text)
    if collision is not None:
        where = ("at the rejected state" if collision.get("phase") != "trajectory"
                 else f"at state {collision.get('state_index')} of the completed path")
        facts.append(f"collisions {where}: {_contacts_text(collision)}")
    elif summary["failure_kind"] == "joint_jump":
        facts.append(f"the next IK solution jumps more than {_max_joint_step(model, resource_id)} rad")
    elif summary["failure_kind"] == "ik_search_failed":
        query = summary["last_ik_query"] or {}
        facts.append(f"the IK search found no joint solution there (error code {query.get('error_code')})")
    if model.kmr and resource_id == model.kmr["resource_id"]:
        facts.append(KMR_IK_PATH_LABEL)
    return prefix + "; ".join(facts) + ". " + native


def _joints_at_bounds(model: CapturedMotionModel, resource_id: str, joints: Sequence[float]) -> list[str]:
    """Facts naming each joint of the reached state that sits on its recorded position bound."""
    configuration = model.configuration(resource_id)
    bounds = {row["name"]: row["position_bounds"] for row in configuration["joints"] if "position_bounds" in row}
    facts = []
    for name, value in zip(configuration["joint_names"], joints, strict=True):
        bound = min(bounds[name], key=lambda limit: abs(value - limit))
        if abs(value - bound) <= 1e-6:
            facts.append(f"{name} is at its bound {bound}")
    return facts


def _named_pose_refusal(model: CapturedMotionModel, resource_id: str, move: Mapping,
                        end: Sequence[float]) -> str | None:
    targets = move.get("named_joints")
    if not targets:
        return None
    errors = [_wrapped(value - target) for value, target in zip(end, targets, strict=True)]
    worst = max(range(len(errors)), key=errors.__getitem__)
    if errors[worst] <= _NAMED_POSE_TOLERANCE_RAD:
        return None
    name = model.configuration(resource_id)["joint_names"][worst]
    return (f"move_to_named_pose step {move['step_index']} ({resource_id}): the controller's Cartesian path "
            f"to FK of {move.get('pose_name')}'s joints ends at joints {[round(value, 4) for value in end]}, "
            f"{errors[worst]:.4f} rad from the named joints at {name}; the controller requires every joint "
            f"within {_NAMED_POSE_TOLERANCE_RAD} rad (Cartesian home did not reach the configured joint posture)")


def _kmr_path_refusal(model: CapturedMotionModel, move: Mapping, states: Sequence[Sequence[float]],
                      base: Sequence[float]) -> str | None:
    """The kmr_gazebo worker's own limits, checked on the states MoveIt returned for one step."""
    execution = model.kmr["task_execution"]
    polyline = move["polyline"]
    prefix = f"{move['primitive']} step {move['step_index']} (KMR): at MoveIt state {{}} of this step "
    for index, joints in enumerate(states):
        pose = robot_pose(model, model.kmr["resource_id"], joints, base_pose=base)
        angle = _down_angle(pose[3:])
        if angle > execution["orientation_tolerance_rad"]:
            return (prefix.format(index) + f"the tool axis is {angle:.4f} rad from facing down; the KMR worker "
                    f"allows {execution['orientation_tolerance_rad']} rad (KMR controller interpolation leaves "
                    f"the downward Cartesian orientation); {KMR_IK_PATH_LABEL}")
        distances = []
        for left, right in zip(polyline[:-1], polyline[1:], strict=True):
            direction = [b - a for a, b in zip(left[:3], right[:3], strict=True)]
            length = sum(value * value for value in direction)
            fraction = (max(0., min(1., sum((pose[i] - left[i]) * direction[i] for i in range(3)) / length))
                        if length else 0.)
            distances.append(math.dist(pose[:3], [left[i] + fraction * direction[i] for i in range(3)]))
        if min(distances) > execution["position_tolerance_m"]:
            return (prefix.format(index) + f"the TCP is {min(distances):.4f} m from the step's Cartesian path; "
                    f"the KMR worker allows {execution['position_tolerance_m']} m (KMR controller interpolation "
                    f"leaves the declared Cartesian path); {KMR_IK_PATH_LABEL}")
    return None


def place_arrival_contact(model: CapturedMotionModel, part_name: str,
                          support_collision_ids: Sequence[str]) -> dict:
    """The allowed contact of a place descent's final pose: the released part on its destination support.

    Args:
        model: The capture's model.
        part_name: The part the descent carries.
        support_collision_ids: The recorded physics collisions of the stated destination's support, as
            𝒦 gives them (compute_place_targets ``support_placement.support_collision_id``).

    Returns:
        ``pairs`` of MoveIt objects to allow, with their sources and the allowance's stated limit.
    """
    from cais_spade_llm.resources.recovery_support_departure import _object_id

    binding = model.physical["scene_object_bindings"].get(part_name)
    if binding is None:
        raise MissingMotionContext(f"{part_name} has no recorded collision object")
    parts = [row["object_id"] for row in binding["objects"]]
    supports = []
    for collision_id in support_collision_ids:
        try:
            supports.append(_object_id(collision_id, model.physical))
        except ValueError as exc:
            raise MissingMotionContext(f"The destination support {collision_id} of {part_name} has no object in "
                                       "the captured MoveIt scene") from exc
    if not supports or set(parts) & set(supports) or model.robot_links & {*parts, *supports}:
        raise ValueError(f"A place arrival pairs {part_name} with a recorded support, never itself or a "
                         "robot link")
    return {"pairs": [[part, support] for part in parts for support in supports], "part_name": part_name,
            "support_collision_ids": list(support_collision_ids), "statement": PLACE_ARRIVAL_STATEMENT}


def _evidence_on_base(model: CapturedMotionModel, snapshot: Mapping) -> dict:
    """Physical evidence whose KMR chain stands on the snapshot's base pose (the recorded chain, moved)."""
    if model.kmr is None:
        return model.physical
    rid = model.kmr["resource_id"]
    base = _planar_base_pose(snapshot["resources"][rid]["base_pose"])
    configuration = model.configuration(rid)
    if [*configuration["root_xyz"], *configuration["root_quaternion"]] == base:
        return model.physical
    moved = {**configuration, "root_xyz": base[:3], "root_quaternion": base[3:]}
    models = {**model.physical["resource_models"],
              rid: {**model.physical["resource_models"][rid], "configuration": moved}}
    return {**model.physical, "resource_models": models}


def support_departure_record(model: CapturedMotionModel, state: Mapping, resource_id: str) -> dict | None:
    """The recorded support contact of a part a robot has just grasped, for its first motion.

    ``prepare_support_departure`` bounds the part's contact with the support it rests on by the initial
    penetration; the native check then requires the part to separate and never re-enter. A part still at
    its recorded pose rests on its recorded support; a part an earlier event moved rests on the support
    ``projected_part_support`` finds from the recorded geometry at its projected pose.

    Args:
        model: The capture's model.
        state: The projected state right after ``grasp_part`` (the robot holds the part at its grasp).
        resource_id: The robot that grasped.

    Returns:
        The prepared record, or None when the part does not touch the support face it rests on.
    """
    from cais_spade_llm.resources.recovery_geometry import (
        _cartesian_endpoint_residual,
        owner_cartesian_endpoint_precision,
        projected_part_support,
    )
    from cais_spade_llm.resources.recovery_support_departure import prepare_support_departure

    snapshot = projected_snapshot(model, state)
    part = snapshot["resources"][resource_id].get("held_part")
    if part is None:
        raise ValueError(f"{resource_id} holds no part")
    evidence = _evidence_on_base(model, snapshot)
    recorded = evidence["physics_evidence"]["models"][snapshot["parts"][part]["model_name"]]["pose"]
    residual = _cartesian_endpoint_residual(snapshot["parts"][part]["current_pose"], [float(v) for v in recorded])
    precision = owner_cartesian_endpoint_precision()
    if (residual["position_residual"] > precision["position_precision"]
            or residual["orientation_residual"] > precision["orientation_precision"]):
        derived = projected_part_support(part_name=part, state=snapshot, evidence=evidence, contact_layer_tolerance=True)
        if derived.get("status") != "supported":
            raise MissingMotionContext(f"No recorded support holds {part} at its projected pose: "
                                       f"{derived.get('reason')}")
        support = {key: value for key, value in derived.items() if key != "physics_evidence"}
        evidence = {**evidence, "stationary_part_evidence": {**evidence.get("stationary_part_evidence", {}),
                                                             part: support}}
    record = prepare_support_departure(resource_id=resource_id, state=snapshot, evidence=evidence,
                                       exact_cylinders=True)
    if record.get("status") == "prepared":
        return record
    if record.get("reason") == "Recorded part is already separated from the requested supporting face":
        return None
    raise MissingMotionContext(f"The support departure of {part} cannot be prepared: {record.get('reason')}")


def mating_contact_record(model: CapturedMotionModel, state: Mapping, resource_id: str,
                          owner_evidence: Mapping, *, part_name: str, target_origin_pose: Mapping,
                          target_tool_pose: Sequence[float]) -> dict:
    """The recorded insertion corridor of a part, for its one aligned insertion motion.

    Args:
        model: The capture's model.
        state: The projected state at the insertion's start (the robot holds the part).
        resource_id: The inserting robot.
        owner_evidence: The robot's ``execution_mode``, ``controller_config``, ``inputs.geometry`` and ``source``.
        part_name: The inserted part.
        target_origin_pose: The calculated part-origin destination ``{"x", "y", "z"}``.
        target_tool_pose: The insertion's tool target ``[x, y, z, qx, qy, qz, qw]``.

    Returns:
        The prepared record from ``prepare_mating_contact``.
    """
    from cais_spade_llm.resources.recovery_mating_contact import prepare_mating_contact

    corridors = owner_evidence["inputs"]["geometry"]["assembly_board"].get("mating_contact_by_part") or {}
    if part_name not in corridors:
        raise MissingMotionContext(f"inputs.geometry.assembly_board.mating_contact_by_part records no insertion "
                                   f"corridor for {part_name}")
    snapshot = projected_snapshot(model, state)
    record = prepare_mating_contact(resource_id=resource_id, part_name=part_name, state=snapshot,
                                    evidence=_evidence_on_base(model, snapshot),
                                    owner_evidence=dict(owner_evidence),
                                    target_origin_pose=dict(target_origin_pose),
                                    target_tool_pose=[float(value) for value in target_tool_pose])
    if record.get("status") != "prepared":
        raise MissingMotionContext(f"The recorded insertion corridor of {part_name} cannot be prepared: "
                                   f"{record.get('reason')}")
    return record

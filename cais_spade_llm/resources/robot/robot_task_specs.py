"""Shared task-spec registry for robot-level assembly functions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable


def _step_ref(fact_path: str, path: str) -> dict[str, str]:
    return {"$ref": f"event_facts.{fact_path}.{path}"}


def _resource_token(resource_jid: str) -> str:
    token = str(resource_jid or "<RESOURCE_JID>").strip()
    return token or "<RESOURCE_JID>"


@dataclass(frozen=True)
class RobotTaskSpec:
    function_name: str
    docstring: str
    modeled_transition: str
    task_preconditions_builder: Callable[[str], list[str]]
    bridge_visible_steps: tuple[dict[str, Any], ...]
    execution_notes: tuple[str, ...]
    source: str = "robot_agent.py"

    def capability_decomposition(self, *, resource_jid: str = "") -> dict[str, Any]:
        return {
            "function_name": self.function_name,
            "source": self.source,
            "modeled_transition": self.modeled_transition,
            "task_preconditions": list(self.task_preconditions_builder(resource_jid)),
            "bridge_visible_steps": deepcopy(list(self.bridge_visible_steps)),
            "execution_notes": list(self.execution_notes),
        }


_ROBOT_TASK_SPECS: tuple[RobotTaskSpec, ...] = (
    RobotTaskSpec(
        function_name="pick_approach",
        docstring="""
---
process: assembly
resource_type: robot

in_state: idle
out_state: at_pick

required_context_keys: [origin]
context_mapping:
  location_param: origin_resource_location
  location_type: part_location

params:
  origin_resource_location:
    type: string
    description: Target origin location to approach for picking.
  part_name:
    type: string
    description: Name of the part intended to be picked (for tracking).
  speed:
    type: number
    description: Optional motion speed.
  product_geometry:
    type: object
    description: Product geometry payload containing part poses in world frame.
  product_jid:
    type: string
    description: JID of the ProductAgent that owns this task.
  task_id:
    type: string

description: Approach the part's origin location with empty gripper.
---
Approach the part's origin location with empty gripper.
""".strip(),
        modeled_transition="idle -> at_pick",
        task_preconditions_builder=lambda _resource_jid: [
            "held_part is empty",
            "part pose is available from detect_parts, product_geometry, or served context",
        ],
        bridge_visible_steps=(
            {
                "primitive": "detect_parts",
                "params": {"part_name": "<PART>"},
                "note": "May be skipped only when equivalent observed pose was retrieved.",
            },
            {
                "primitive": "compute_pick_targets",
                "params": {"part_name": "<PART>"},
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": _step_ref("pick_targets.<PART>", "approach_pose.x"),
                    "y": _step_ref("pick_targets.<PART>", "approach_pose.y"),
                    "z": _step_ref("pick_targets.<PART>", "approach_pose.z"),
                },
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": _step_ref("pick_targets.<PART>", "target_pose.x"),
                    "y": _step_ref("pick_targets.<PART>", "target_pose.y"),
                    "z": _step_ref("pick_targets.<PART>", "target_pose.z"),
                },
            },
        ),
        execution_notes=(
            "RobotAgent.pick_approach computes pick geometry, opens the gripper, moves above the part, then descends to the pick pose.",
            "open_gripper and direct controller pose helpers are hidden from synthesis; use compute_pick_targets plus move_cartesian approach/target poses.",
        ),
    ),
    RobotTaskSpec(
        function_name="pick_grasp",
        docstring="""
---
process: assembly
resource_type: robot

in_state: at_pick
out_state: picked
part_in_state: ready

required_context_keys: [origin]
context_mapping:
  location_param: origin_resource_location
  location_type: current_location

part_transition:
  completed:
    state: in_gripper
    location_template: "{resource_jid}_gripper"

params:
  part_name:
    type: string
    description: Name of the part to pick.
  origin_resource_location:
    type: string
    description: Origin location of the part (printer or fixture).
  gripper:
    type: string
    description: Optional gripper configuration.
  product_geometry:
    type: object
    description: Product geometry payload containing part poses in world frame.
  product_jid:
    type: string
    description: JID of the ProductAgent that owns this task.
  task_id:
    type: string

description: Pick a ready part from an origin location.
---
Pick a ready part from an origin location.
""".strip(),
        modeled_transition="at_pick -> picked",
        task_preconditions_builder=lambda _resource_jid: [
            "resource is already at the pick pose from pick_approach",
            "held_part is empty",
            "pick target was grounded for the active part",
        ],
        bridge_visible_steps=(
            {
                "primitive": "grasp_part",
                "params": {
                    "model_name": "<MODEL_NAME_FROM_PART_TARGET>",
                    "part_name": "<PART>",
                },
            },
            {
                "primitive": "move_relative",
                "params": {"dx": 0.0, "dy": 0.0, "dz": 0.05, "speed": 0.8},
                "note": "Positive dz lift/retreat after grasp.",
            },
        ),
        execution_notes=(
            "RobotAgent.pick_grasp closes the gripper, attaches the part in simulation, then lifts to travel height.",
            "close_gripper and attach_part are hidden from synthesis; use grasp_part as the visible composite.",
        ),
    ),
    RobotTaskSpec(
        function_name="place_approach",
        docstring="""
---
process: assembly
resource_type: robot

in_state: picked
out_state: positioned
part_in_state: in_gripper

required_context_keys: [destination]
context_mapping:
  location_param: destination_location
  location_type: reachable_location

part_transition:
  completed:
    state: in_transit
    location_template: "{resource_jid}_gripper"

params:
  destination_location:
    type: string
    description: Destination location to carry the loaded part.
  part_name:
    type: string
    description: Name of the part being moved.
  speed:
    type: number
    description: Optional motion speed while loaded.
  product_geometry:
    type: object
    description: Product geometry payload containing target placement poses.
  product_jid:
    type: string
    description: JID of the ProductAgent that owns this task.
  task_id:
    type: string

description: Move the loaded part to its destination location.
---
Move the loaded part to its destination location.
""".strip(),
        modeled_transition="picked -> positioned",
        task_preconditions_builder=lambda _resource_jid: [
            "held_part exists",
            "destination geometry is available from destination_location, product_geometry, or served context",
        ],
        bridge_visible_steps=(
            {
                "primitive": "compute_place_targets",
                "params": {
                    "part_name": "<PART>",
                    "destination_location": "<DESTINATION_LOCATION>",
                },
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": _step_ref("place_targets.<PART>", "approach_pose.x"),
                    "y": _step_ref("place_targets.<PART>", "approach_pose.y"),
                    "z": _step_ref("place_targets.<PART>", "approach_pose.z"),
                },
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": _step_ref("place_targets.<PART>", "target_pose.x"),
                    "y": _step_ref("place_targets.<PART>", "target_pose.y"),
                    "z": _step_ref("place_targets.<PART>", "target_pose.z"),
                },
            },
        ),
        execution_notes=(
            "RobotAgent.place_approach computes destination geometry, moves above the destination, then descends to the place pose.",
            "Direct controller pose helpers are hidden from synthesis; use compute_place_targets plus move_cartesian approach/target poses.",
        ),
    ),
    RobotTaskSpec(
        function_name="place_insert",
        docstring="""
---
process: assembly
resource_type: robot

in_state: positioned
out_state: placed
part_in_state: in_transit

required_context_keys: [destination]
context_mapping:
  location_param: destination_location
  location_type: current_location

part_transition:
  completed:
    state: assembled
    verify_camera: true
    location_param: destination_location

params:
  destination_location:
    type: string
    description: Final assembly location for the part.
  part_name:
    type: string
    description: Name of the part being assembled.
  orientation:
    type: string
    description: Optional placement orientation.
  product_geometry:
    type: object
    description: Product geometry payload containing insertion target pose.
  product_jid:
    type: string
    description: JID of the ProductAgent that owns this task.
  task_id:
    type: string

description: Assemble the currently held part at its final destination.
---
Assemble the currently held part at its final destination.
""".strip(),
        modeled_transition="positioned -> placed",
        task_preconditions_builder=lambda _resource_jid: [
            "held_part exists",
            "place_approach already positioned the robot at the target pose",
        ],
        bridge_visible_steps=(
            {
                "primitive": "release_part",
                "params": {
                    "model_name": "<MODEL_NAME_FROM_PART_TARGET>",
                    "part_name": "<PART>",
                },
            },
            {
                "primitive": "move_relative",
                "params": {"dx": 0.0, "dy": 0.0, "dz": 0.08, "speed": 1.0},
                "note": "Positive dz retreat after release.",
            },
        ),
        execution_notes=(
            "RobotAgent.place_insert releases the part at the pose established by place_approach, detaches in simulation, snaps/settles as needed, then lifts away.",
            "open_gripper and detach_part are hidden from synthesis; use release_part as the visible composite.",
        ),
    ),
    RobotTaskSpec(
        function_name="move_home",
        docstring="""
---
process: assembly
resource_type: robot

in_state: any
out_state: idle

context: []

params:
  product_jid:
    type: string
    description: JID of the ProductAgent that owns this task.
  task_id:
    type: string

description: Robot arm move to its home position.
---
Robot arm move to its home position.
""".strip(),
        modeled_transition="any -> idle",
        task_preconditions_builder=lambda resource_jid: [
            f"resource {_resource_token(resource_jid)} exposes a named pose called 'home'",
        ],
        bridge_visible_steps=(
            {
                "primitive": "move_to_named_pose",
                "params": {"pose_name": "home", "speed": 0.8},
            },
        ),
        execution_notes=(
            "RobotAgent.move_home maps to the controller's home/named-pose motion when that named pose is available.",
        ),
    ),
)


def robot_task_spec_registry() -> dict[str, RobotTaskSpec]:
    return {spec.function_name: spec for spec in _ROBOT_TASK_SPECS}


def robot_task_spec_names() -> tuple[str, ...]:
    preferred_order = (
        "pick_approach",
        "pick_grasp",
        "place_approach",
        "move_home",
        "place_insert",
    )
    registry = robot_task_spec_registry()
    ordered = [name for name in preferred_order if name in registry]
    ordered.extend(
        spec.function_name
        for spec in _ROBOT_TASK_SPECS
        if spec.function_name not in ordered
    )
    return tuple(ordered)


def robot_task_docstring(function_name: str) -> str:
    spec = robot_task_spec_registry().get(str(function_name or "").strip())
    return spec.docstring if spec is not None else ""


def robot_task_capability_decompositions(
    *,
    function_name: str = "",
    resource_jid: str = "",
) -> dict[str, Any]:
    registry = robot_task_spec_registry()
    token = str(function_name or "").strip()
    if token:
        spec = registry.get(token)
        return spec.capability_decomposition(resource_jid=resource_jid) if spec is not None else {}
    return {
        name: spec.capability_decomposition(resource_jid=resource_jid)
        for name, spec in registry.items()
    }


__all__ = [
    "RobotTaskSpec",
    "robot_task_capability_decompositions",
    "robot_task_docstring",
    "robot_task_spec_names",
    "robot_task_spec_registry",
]

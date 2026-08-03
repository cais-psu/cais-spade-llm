"""Declarative definition for the pick_grasp robot task."""

from __future__ import annotations

from ..robot_task_model import (
    RobotTaskArgument,
    RobotTaskDefinition,
    RobotTaskEffect,
    RobotTaskGuard,
    RobotTaskProgram,
    RobotTaskStep,
    _arg,
    _first,
    _format,
    _state,
    _sub,
)

ROBOT_TASK_DEFINITION = RobotTaskDefinition(
    name="pick_grasp",
    description="Pick a ready part from an origin location.",
    arguments=(
        RobotTaskArgument(
            name="part_name",
            type="string",
            description="Name of the part to pick.",
            required=True,
        ),
        RobotTaskArgument(
            name="origin_resource_location",
            type="string",
            description="Origin location of the part (printer or fixture).",
            required=True,
        ),
        RobotTaskArgument(
            name="gripper",
            type="string",
            description="Optional gripper configuration.",
        ),
        RobotTaskArgument(
            name="product_geometry",
            type="object",
            description="Product geometry payload containing part poses in world frame.",
        ),
        RobotTaskArgument(
            name="product_jid",
            type="string",
            description="JID of the ProductAgent that owns this task.",
        ),
        RobotTaskArgument(name="task_id", type="string"),
    ),
    program=RobotTaskProgram(
        entry_state="at_pick",
        success_state="picked",
        part_in_state="ready",
        required_context_keys=("origin",),
        context_mapping={
            "location_param": "origin_resource_location",
            "location_type": "current_location",
        },
        part_transition={
            "completed": {
                "state": "in_gripper",
                "location_template": "{resource_jid}",
            }
        },
        entry_guards=(
            RobotTaskGuard(
                predicate="always",
                message="pick_grasp requires pick_approach to finish at at_pick.",
                description="resource is at_pick after pick_approach",
                condition={
                    "field": "resource_state",
                    "operator": "equals",
                    "value": "at_pick",
                },
            ),
            RobotTaskGuard(
                predicate="held_part_empty",
                message=_format(
                    "Already holding {held_part}; assemble it before picking a new part.",
                    held_part=_state("_held_part"),
                ),
                description="resource is already at the pick pose from pick_approach",
                condition={
                    "field": "held_part",
                    "operator": "equals",
                    "value": None,
                },
            ),
            RobotTaskGuard(
                predicate="always",
                message="pick_approach has not established the requested origin context.",
                description="pick target was grounded for the requested origin",
                condition={
                    "field": "task_ctx.origin_resource_location",
                    "operator": "equals",
                    "value": _arg("origin_resource_location"),
                },
            ),
            RobotTaskGuard(
                predicate="always",
                message="pick_approach has not established the requested part context.",
                description="pick target was grounded for the active part",
                condition={
                    "field": "task_ctx.part_name",
                    "operator": "equals",
                    "value": _arg("part_name"),
                },
            ),
        ),
        steps=(
            RobotTaskStep(
                id="grasp_part",
                op="grasp_part",
                executor="primitive",
                exposed=True,
                params={
                    "model_name": _state("_task_ctx", "model_name"),
                    "part_name": _arg("part_name"),
                    "position": _state("_task_ctx", "gripper_close_position"),
                },
                public_params={
                    "model_name": "<MODEL_NAME_FROM_PART_TARGET>",
                    "part_name": _arg("part_name"),
                },
                failure_observations={
                    "part_name": _arg("part_name"),
                    "model_name": _state("_task_ctx", "model_name"),
                },
            ),
            RobotTaskStep(
                id="delay_after_grasp",
                op="delay",
                executor="primitive",
                exposed=False,
                params={"duration_sec": 0.25},
            ),
            RobotTaskStep(
                id="lift",
                op="move_relative",
                executor="primitive",
                exposed=True,
                params={
                    "dx": 0.0,
                    "dy": 0.0,
                    "dz": _sub(_state("_task_ctx", "travel_z"), _state("_position", "z")),
                    "speed": 0.45,
                },
                public_params={
                    "dx": 0.0,
                    "dy": 0.0,
                    "dz": 0.05,
                    "speed": 0.45,
                },
                note="Positive dz lift/retreat after grasp.",
                failure_observations={"part_name": _arg("part_name")},
            ),
        ),
        effects=(
            RobotTaskEffect(target="held_part", action="set", value=_arg("part_name")),
            RobotTaskEffect(target="current_state", action="set", value="picked"),
            RobotTaskEffect(target="gripper_state", action="set", value="closed"),
        ),
        success_response={
            "status": "completed",
            "content": _format("Picked {part_name}.", part_name=_arg("part_name")),
            "observations": {
                "part_name": _arg("part_name"),
                "origin_pose": {
                    "x": _state("_task_ctx", "tx"),
                    "y": _state("_task_ctx", "ty"),
                    "z": _state("_task_ctx", "tz"),
                },
            },
        },
        dry_run_description=_format(
            "Picking {part_name} from {origin_resource_location} (gripper={gripper})",
            part_name=_arg("part_name"),
            origin_resource_location=_arg("origin_resource_location"),
            gripper=_first(_arg("gripper"), "default"),
        ),
        dry_run_duration=5.0,
        failure_part=_arg("part_name"),
        notes=(
            "RobotAgent.pick_grasp closes the gripper, attaches the part in simulation, then lifts to travel height.",
            "close_gripper and attach_part are hidden from synthesis; use grasp_part as the visible composite.",
        ),
    ),
)

__all__ = ["ROBOT_TASK_DEFINITION"]

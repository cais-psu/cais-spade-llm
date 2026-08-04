"""Declarative definition for the place_insert robot task."""

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
    name="place_insert",
    description="Assemble the currently held part at its final destination.",
    arguments=(
        RobotTaskArgument(
            name="destination_location",
            type="string",
            description="Final assembly location for the part.",
            required=True,
        ),
        RobotTaskArgument(
            name="part_name",
            type="string",
            description="Name of the part being assembled.",
            required=True,
        ),
        RobotTaskArgument(
            name="orientation",
            type="string",
            description="Optional placement orientation.",
        ),
        RobotTaskArgument(
            name="product_geometry",
            type="object",
            description="Product geometry payload containing insertion target pose.",
        ),
        RobotTaskArgument(
            name="product_jid",
            type="string",
            description="JID of the ProductAgent that owns this task.",
        ),
        RobotTaskArgument(name="task_id", type="string"),
    ),
    program=RobotTaskProgram(
        entry_state="positioned",
        success_state="placed",
        part_in_state="in_transit",
        required_context_keys=("destination",),
        context_mapping={
            "location_param": "destination_location",
            "location_type": "current_location",
        },
        part_transition={
            "completed": {
                "state": "assembled",
                "verify_camera": True,
                "location_param": "destination_location",
            }
        },
        entry_guards=(
            RobotTaskGuard(
                predicate="always",
                message="place_insert requires place_approach to finish at positioned.",
                description="resource is positioned after place_approach",
                condition={
                    "field": "resource_state",
                    "operator": "equals",
                    "value": "positioned",
                },
            ),
            RobotTaskGuard(
                predicate="held_part_exists",
                message="No part currently held; run pick_grasp first.",
                description="held_part exists",
                condition={
                    "field": "held_part",
                    "operator": "equals",
                    "value": _arg("part_name"),
                },
            ),
            RobotTaskGuard(
                predicate="always",
                message="place_approach has not established this destination context.",
                description="place_approach already positioned the robot at the target pose",
                condition={
                    "field": "task_ctx.destination_location",
                    "operator": "equals",
                    "value": _arg("destination_location"),
                },
            ),
        ),
        steps=(
            RobotTaskStep(
                id="delay_before_release",
                op="delay",
                executor="primitive",
                exposed=False,
                params={"duration_sec": 0.25},
            ),
            RobotTaskStep(
                id="release_part",
                op="release_part",
                executor="primitive",
                exposed=True,
                params={
                    "model_name": _state("_task_ctx", "model_name"),
                    "part_name": _state("_held_part"),
                },
                public_params={
                    "model_name": "<MODEL_NAME_FROM_PART_TARGET>",
                    "part_name": _arg("part_name"),
                },
                failure_observations={
                    "part_name": _state("_held_part"),
                    "destination_location": _arg("destination_location"),
                },
            ),
            RobotTaskStep(
                id="delay_after_release",
                op="delay",
                executor="primitive",
                exposed=False,
                params={"duration_sec": 0.25},
            ),
            RobotTaskStep(
                id="snap_part_to_slot",
                op="snap_part_to_slot",
                executor="primitive",
                exposed=False,
                params={
                    "model_name": _state("_task_ctx", "model_name"),
                    "slot_x": _state("_task_ctx", "slot_x"),
                    "slot_y": _state("_task_ctx", "slot_y"),
                    "part_height": _state("_task_ctx", "part_height"),
                    "board_top_z": _state("_task_ctx", "board_top_z"),
                    "part_origin_z": _state("_task_ctx", "place_part_origin_z"),
                    "destination_location": _arg("destination_location"),
                },
                when=(
                    RobotTaskGuard(
                        predicate="execution_mode",
                        args={"mode": "simulation"},
                    ),
                    RobotTaskGuard(
                        predicate="task_ctx_key_truthy",
                        args={"key": "model_name"},
                    ),
                ),
                continue_on_failure=True,
                failure_observations={
                    "part_name": _state("_held_part"),
                    "destination_location": _arg("destination_location"),
                },
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
                    "dz": 0.08,
                    "speed": 0.45,
                },
                note="Positive dz retreat after release.",
                failure_observations={
                    "part_name": _state("_held_part"),
                    "destination_location": _arg("destination_location"),
                },
            ),
        ),
        effects=(
            RobotTaskEffect(target="held_part", action="clear"),
            RobotTaskEffect(target="current_state", action="set", value="placed"),
            RobotTaskEffect(target="gripper_state", action="set", value="open"),
            RobotTaskEffect(target="task_ctx", action="clear"),
        ),
        success_response={
            "status": "completed",
            "content": _format(
                "Assembled {placed} at {destination_location}.",
                placed=_first(_arg("part_name"), _state("_held_part")),
                destination_location=_arg("destination_location"),
            ),
            "placed_location": _arg("destination_location"),
        },
        dry_run_description=_format(
            "Assembling {held_part} at {destination_location} (orientation={orientation})",
            held_part=_state("_held_part"),
            destination_location=_arg("destination_location"),
            orientation=_first(_arg("orientation"), "default"),
        ),
        dry_run_duration=5.0,
        failure_part=_first(_arg("part_name"), _state("_held_part")),
        notes=(
            "RobotAgent.place_insert releases the part at the pose established by place_approach, detaches and snaps/settles in simulation, then lifts away.",
            "open_gripper and detach_part are hidden from synthesis; use release_part as the visible composite.",
        ),
    ),
)

__all__ = ["ROBOT_TASK_DEFINITION"]

"""Declarative definition for the place_approach robot task."""

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
    _step_output,
    _step_ref,
)

ROBOT_TASK_DEFINITION = RobotTaskDefinition(
    name="place_approach",
    description="Move the loaded part to its destination location.",
    arguments=(
        RobotTaskArgument(
            name="destination_location",
            type="string",
            description="Destination location to carry the loaded part.",
            required=True,
        ),
        RobotTaskArgument(
            name="part_name",
            type="string",
            description="Name of the part being moved.",
            required=True,
        ),
        RobotTaskArgument(
            name="speed",
            type="number",
            description="Optional motion speed while loaded.",
        ),
        RobotTaskArgument(
            name="product_geometry",
            type="object",
            description="Product geometry payload containing target placement poses.",
        ),
        RobotTaskArgument(
            name="product_jid",
            type="string",
            description="JID of the ProductAgent that owns this task.",
        ),
        RobotTaskArgument(name="task_id", type="string"),
    ),
    program=RobotTaskProgram(
        entry_state="picked",
        success_state="positioned",
        part_in_state="in_gripper",
        required_context_keys=("destination",),
        context_mapping={
            "location_param": "destination_location",
            "location_type": "reachable_location",
        },
        part_transition={
            "completed": {
                "state": "in_transit",
                "location_template": "{resource_jid}",
            }
        },
        entry_guards=(
            RobotTaskGuard(
                predicate="always",
                message="place_approach requires pick_grasp to finish at picked.",
                description="resource is picked after pick_grasp",
                condition={
                    "field": "resource_state",
                    "operator": "equals",
                    "value": "picked",
                },
            ),
            RobotTaskGuard(
                predicate="held_part_exists",
                message="Cannot move-loaded without holding a part.",
                description="held_part exists",
                condition={
                    "field": "held_part",
                    "operator": "equals",
                    "value": _arg("part_name"),
                },
            ),
            RobotTaskGuard(
                predicate="always",
                description="destination geometry is available from destination_location, product_geometry, or served context",
            ),
        ),
        steps=(
            RobotTaskStep(
                id="move_to_destination_location",
                op="move_to_named_pose",
                executor="primitive",
                exposed=False,
                params={
                    "pose_name": _arg("destination_location"),
                    "speed": _arg("speed"),
                },
                when=(
                    RobotTaskGuard(
                        predicate="execution_mode",
                        args={"mode": "physical"},
                    ),
                ),
                note=(
                    "Automatic physical staging from destination_location; "
                    "no recording required."
                ),
            ),
            RobotTaskStep(
                id="localize_assembly_board_v1",
                op="localize_assembly_board_v1",
                executor="primitive",
                exposed=False,
                store_as="assembly_board_v1_aruco",
                params={
                    "destination_location": _arg("destination_location"),
                    "part_name": _arg("part_name"),
                },
                when=(
                    RobotTaskGuard(
                        predicate="execution_mode",
                        args={"mode": "physical"},
                    ),
                ),
                failure_observations={
                    "destination_location": _arg("destination_location"),
                    "part_name": _state("_held_part"),
                },
                note="Freeze one fresh role-specific assembly_board-v1 ArUco pose.",
            ),
            RobotTaskStep(
                id="compute_place_targets",
                op="compute_place_targets",
                executor="primitive",
                exposed=True,
                store_as="place_targets",
                params={
                    "pick_ctx": _state("_task_ctx"),
                    "product_geometry": _arg("product_geometry"),
                    "part_name": _arg("part_name"),
                    "z_adjustment_m": 0.0,
                    "destination_location": _arg("destination_location"),
                    "assembly_board_v1_aruco": _step_output(
                        "assembly_board_v1_aruco"
                    ),
                },
                public_params={
                    "part_name": _arg("part_name"),
                    "destination_location": _arg("destination_location"),
                },
                dry_run_output={
                    "slot_x": 0.0,
                    "slot_y": 0.0,
                    "board_top_z": 1.025,
                    "place_z": 1.1,
                    "part_height": 0.08,
                    "destination_location": _arg("destination_location"),
                },
                failure_observations={"part_name": _state("_held_part")},
            ),
            RobotTaskStep(
                id="move_above_destination",
                op="move_cartesian",
                executor="primitive",
                exposed=True,
                physical_position_required=True,
                params={
                    "x": _step_output("place_targets", "approach_pose", "x"),
                    "y": _step_output("place_targets", "approach_pose", "y"),
                    "z": _step_output("place_targets", "approach_pose", "z"),
                    "speed": _arg("speed"),
                },
                public_params={
                    "x": _step_ref("place_targets.<PART>", "approach_pose.x"),
                    "y": _step_ref("place_targets.<PART>", "approach_pose.y"),
                    "z": _step_ref("place_targets.<PART>", "approach_pose.z"),
                },
                failure_observations={"part_name": _state("_held_part")},
            ),
            RobotTaskStep(
                id="descend",
                op="move_cartesian",
                executor="primitive",
                exposed=True,
                physical_position_required=True,
                params={
                    "x": _step_output("place_targets", "target_pose", "x"),
                    "y": _step_output("place_targets", "target_pose", "y"),
                    "z": _step_output("place_targets", "target_pose", "z"),
                },
                public_params={
                    "x": _step_ref("place_targets.<PART>", "target_pose.x"),
                    "y": _step_ref("place_targets.<PART>", "target_pose.y"),
                    "z": _step_ref("place_targets.<PART>", "target_pose.z"),
                },
                failure_observations={"part_name": _state("_held_part")},
            ),
        ),
        effects=(
            RobotTaskEffect(
                target="task_ctx",
                action="merge",
                value={
                    "slot_x": _step_output("place_targets", "slot_x"),
                    "slot_y": _step_output("place_targets", "slot_y"),
                    "board_top_z": _step_output("place_targets", "board_top_z"),
                    "place_z": _step_output("place_targets", "place_z"),
                    "place_part_origin_z": _step_output("place_targets", "place_part_origin_z"),
                    "part_height": _step_output("place_targets", "part_height"),
                    "destination_location": _arg("destination_location"),
                    "model_name": _step_output("place_targets", "model_name"),
                },
                skip_empty_values=True,
            ),
            RobotTaskEffect(target="current_state", action="set", value="positioned"),
            RobotTaskEffect(
                target="position",
                action="set",
                value={
                    "x": _step_output("place_targets", "slot_x"),
                    "y": _step_output("place_targets", "slot_y"),
                    "z": _step_output("place_targets", "place_z"),
                },
            ),
            RobotTaskEffect(target="recovery_pose_ref", action="set", value=None),
        ),
        success_response={
            "status": "completed",
            "content": _format(
                "Reached {destination_location} with {held_part}.",
                destination_location=_arg("destination_location"),
                held_part=_state("_held_part"),
            ),
        },
        dry_run_description=_format(
            "Move loaded part {held_part} to {destination_location} (speed={speed})",
            held_part=_state("_held_part"),
            destination_location=_arg("destination_location"),
            speed=_first(_arg("speed"), "default"),
        ),
        dry_run_duration=5.0,
        failure_part=_first(_arg("part_name"), _state("_held_part")),
        notes=(
            "Physical RobotAgent.place_approach stages at destination_location, freezes assembly_board-v1 localization, computes destination geometry, moves above the destination, then descends to the place pose.",
            "Direct controller pose helpers are hidden from synthesis; use compute_place_targets plus move_cartesian approach/target poses.",
        ),
    ),
)

__all__ = ["ROBOT_TASK_DEFINITION"]

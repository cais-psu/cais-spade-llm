"""Declarative definition for the pick_approach robot task."""

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
    _step_output,
    _step_ref,
)

ROBOT_TASK_DEFINITION = RobotTaskDefinition(
    name="pick_approach",
    description="Approach the part's origin location with empty gripper.",
    arguments=(
        RobotTaskArgument(
            name="origin_resource_location",
            type="string",
            description="Target origin location to approach for picking.",
            required=True,
        ),
        RobotTaskArgument(
            name="part_name",
            type="string",
            description="Name of the part intended to be picked (for tracking).",
            required=True,
        ),
        RobotTaskArgument(
            name="speed",
            type="number",
            description="Optional motion speed.",
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
        entry_state="idle",
        success_state="at_pick",
        required_context_keys=("origin",),
        context_mapping={
            "location_param": "origin_resource_location",
            "location_type": "part_location",
        },
        entry_guards=(
            RobotTaskGuard(
                predicate="always",
                message="pick_approach requires the robot to be idle.",
                description="resource is idle before approaching a part",
                condition={
                    "field": "resource_state",
                    "operator": "equals",
                    "value": "idle",
                },
            ),
            RobotTaskGuard(
                predicate="held_part_empty",
                message="Cannot move-to-pick while already holding a part.",
                description="held_part is empty",
                condition={
                    "field": "held_part",
                    "operator": "equals",
                    "value": None,
                },
            ),
        ),
        steps=(
            RobotTaskStep(
                id="move_to_origin_resource_location",
                op="move_to_named_pose",
                executor="primitive",
                exposed=False,
                params={
                    "pose_name": _arg("origin_resource_location"),
                    "speed": _arg("speed"),
                },
                when=(
                    RobotTaskGuard(
                        predicate="execution_mode",
                        args={"mode": "physical"},
                    ),
                ),
                note=(
                    "Automatic physical staging from origin_resource_location; "
                    "no recording required."
                ),
            ),
            RobotTaskStep(
                id="detect_parts",
                op="detect_parts",
                executor="primitive",
                exposed=True,
                store_as="detected_parts",
                params={"part_name": _arg("part_name")},
                note="May be skipped only when equivalent observed pose was retrieved.",
            ),
            RobotTaskStep(
                id="compute_pick_targets",
                op="compute_pick_targets",
                executor="primitive",
                exposed=True,
                store_as="pick_targets",
                params={
                    "part_name": _arg("part_name"),
                    "product_geometry": _arg("product_geometry"),
                    "detected_parts": _step_output("detected_parts"),
                },
                public_params={
                    "part_name": _arg("part_name"),
                    "product_geometry": _arg("product_geometry"),
                },
                dry_run_output={
                    "part_name": _arg("part_name"),
                    "model_name": "",
                    "tx": 0.0,
                    "ty": 0.0,
                    "tz": 0.0,
                    "pick_z": 0.0,
                    "travel_z": 1.2,
                    "part_height": 0.08,
                    "tcp_offset_z": -0.17,
                    "pick_tcp_z": 0.0,
                    "pick_tcp_z_offset_from_table_m": 0.0,
                    "source_stl": "",
                    "source_stl_sha256": "",
                    "hub_up": False,
                    "hub_diameter_m": 0.0,
                    "hub_height_m": 0.0,
                    "tooth_diameter_m": 0.0,
                    "tooth_height_m": 0.0,
                    "grasp_width_m": 0.0,
                    "tooth_clearance_m": 0.0,
                    "minimum_hub_overlap_m": 0.0,
                    "finger_tooth_clearance_m": 0.0,
                    "finger_hub_overlap_m": 0.0,
                    "gripper_close_position": None,
                    "start_x": 0.0,
                    "start_y": 0.0,
                    "start_z": 0.0,
                },
                failure_observations={"part_name": _arg("part_name")},
            ),
            RobotTaskStep(
                id="open_gripper",
                op="open_gripper",
                executor="primitive",
                exposed=False,
            ),
            RobotTaskStep(
                id="move_above_part",
                op="move_cartesian",
                executor="primitive",
                exposed=True,
                params={
                    "x": _step_output("pick_targets", "approach_pose", "x"),
                    "y": _step_output("pick_targets", "approach_pose", "y"),
                    "z": _step_output("pick_targets", "approach_pose", "z"),
                    "speed": _arg("speed"),
                },
                public_params={
                    "x": _step_ref("pick_targets.<PART>", "approach_pose.x"),
                    "y": _step_ref("pick_targets.<PART>", "approach_pose.y"),
                    "z": _step_ref("pick_targets.<PART>", "approach_pose.z"),
                },
                failure_observations={"part_name": _arg("part_name")},
            ),
            RobotTaskStep(
                id="descend",
                op="move_cartesian",
                executor="primitive",
                exposed=True,
                params={
                    "x": _step_output("pick_targets", "target_pose", "x"),
                    "y": _step_output("pick_targets", "target_pose", "y"),
                    "z": _step_output("pick_targets", "target_pose", "z"),
                },
                public_params={
                    "x": _step_ref("pick_targets.<PART>", "target_pose.x"),
                    "y": _step_ref("pick_targets.<PART>", "target_pose.y"),
                    "z": _step_ref("pick_targets.<PART>", "target_pose.z"),
                },
                failure_observations={"part_name": _arg("part_name")},
            ),
        ),
        effects=(
            RobotTaskEffect(
                target="task_ctx",
                action="set",
                value={
                    "part_name": _step_output("pick_targets", "part_name"),
                    "model_name": _step_output("pick_targets", "model_name"),
                    "tx": _step_output("pick_targets", "tx"),
                    "ty": _step_output("pick_targets", "ty"),
                    "tz": _step_output("pick_targets", "tz"),
                    "pick_z": _step_output("pick_targets", "pick_z"),
                    "travel_z": _step_output("pick_targets", "travel_z"),
                    "part_height": _step_output("pick_targets", "part_height"),
                    "tcp_offset_z": _step_output("pick_targets", "tcp_offset_z"),
                    "pick_tcp_z": _step_output("pick_targets", "pick_tcp_z"),
                    "pick_tcp_z_offset_from_table_m": _step_output(
                        "pick_targets", "pick_tcp_z_offset_from_table_m"
                    ),
                    "source_stl": _step_output("pick_targets", "source_stl"),
                    "source_stl_sha256": _step_output(
                        "pick_targets", "source_stl_sha256"
                    ),
                    "hub_up": _step_output("pick_targets", "hub_up"),
                    "hub_diameter_m": _step_output("pick_targets", "hub_diameter_m"),
                    "hub_height_m": _step_output("pick_targets", "hub_height_m"),
                    "tooth_diameter_m": _step_output(
                        "pick_targets", "tooth_diameter_m"
                    ),
                    "tooth_height_m": _step_output("pick_targets", "tooth_height_m"),
                    "grasp_width_m": _step_output("pick_targets", "grasp_width_m"),
                    "tooth_clearance_m": _step_output(
                        "pick_targets", "tooth_clearance_m"
                    ),
                    "minimum_hub_overlap_m": _step_output(
                        "pick_targets", "minimum_hub_overlap_m"
                    ),
                    "finger_tooth_clearance_m": _step_output(
                        "pick_targets", "finger_tooth_clearance_m"
                    ),
                    "finger_hub_overlap_m": _step_output(
                        "pick_targets", "finger_hub_overlap_m"
                    ),
                    "gripper_close_position": _step_output(
                        "pick_targets", "gripper_close_position"
                    ),
                    "origin_resource_location": _arg("origin_resource_location"),
                    "origin_pose": _step_output("pick_targets", "origin_pose"),
                    "start_x": _step_output("pick_targets", "start_x"),
                    "start_y": _step_output("pick_targets", "start_y"),
                    "start_z": _step_output("pick_targets", "start_z"),
                },
            ),
            RobotTaskEffect(target="current_state", action="set", value="at_pick"),
            RobotTaskEffect(
                target="position",
                action="set",
                value={
                    "x": _step_output("pick_targets", "tx"),
                    "y": _step_output("pick_targets", "ty"),
                    "z": _step_output("pick_targets", "pick_z"),
                },
            ),
            RobotTaskEffect(target="recovery_pose_ref", action="set", value=None),
        ),
        success_response={
            "status": "completed",
            "content": _format(
                "Arrived at {origin_resource_location} ready to pick {part_name}.",
                origin_resource_location=_arg("origin_resource_location"),
                part_name=_arg("part_name"),
            ),
        },
        dry_run_description=_format(
            "Travel empty to pick location {origin_resource_location} for {part_name} (speed={speed})",
            origin_resource_location=_arg("origin_resource_location"),
            part_name=_arg("part_name"),
            speed=_first(_arg("speed"), "default"),
        ),
        dry_run_duration=5.0,
        failure_part=_arg("part_name"),
        notes=(
            "RobotAgent.pick_approach computes pick geometry, opens the gripper, moves above the part, then descends to the pick pose.",
            "open_gripper and direct controller pose helpers are hidden from synthesis; use compute_pick_targets plus move_cartesian approach/target poses.",
            "Physical MG geometry is derived from the actual source_stl and leaves the gripper open at the smooth raised hub target.",
        ),
    ),
)

__all__ = ["ROBOT_TASK_DEFINITION"]

"""Declarative definition for the move_home robot task."""

from __future__ import annotations

from ..robot_task_model import (
    RobotTaskArgument,
    RobotTaskDefinition,
    RobotTaskEffect,
    RobotTaskExposure,
    RobotTaskGuard,
    RobotTaskProgram,
    RobotTaskStep,
)

ROBOT_TASK_DEFINITION = RobotTaskDefinition(
    name="move_home",
    description="Robot arm move to its home position.",
    arguments=(
        RobotTaskArgument(
            name="product_jid",
            type="string",
            description="JID of the ProductAgent that owns this task.",
        ),
        RobotTaskArgument(name="task_id", type="string"),
    ),
    exposure=RobotTaskExposure(
        predicates=(
            RobotTaskGuard(
                predicate="named_pose_available",
                args={"pose_name": "home"},
            ),
        ),
    ),
    program=RobotTaskProgram(
        entry_state="any",
        success_state="idle",
        entry_guards=(
            RobotTaskGuard(
                predicate="held_part_empty",
                message="Cannot move home while still holding a part; assemble it first.",
                description="held_part is empty",
                condition={
                    "field": "held_part",
                    "operator": "equals",
                    "value": None,
                },
            ),
            RobotTaskGuard(
                predicate="named_pose_available",
                args={"pose_name": "home"},
                description="resource exposes a named pose called 'home'",
            ),
        ),
        steps=(
            RobotTaskStep(
                id="move_home",
                op="move_to_named_pose",
                executor="primitive",
                exposed=True,
                params={
                    "pose_name": "home",
                    "speed": 0.25,
                },
                public_params={"pose_name": "home", "speed": 0.25},
                dry_run_output={
                    "absolute_position": {"x": 0.0, "y": 0.0, "z": 445.0},
                },
            ),
        ),
        effects=(
            RobotTaskEffect(target="current_state", action="set", value="idle"),
            RobotTaskEffect(
                target="position",
                action="set",
                value={"x": 0.0, "y": 0.0, "z": 445.0},
            ),
            RobotTaskEffect(target="recovery_pose_ref", action="set", value="home"),
            RobotTaskEffect(target="task_ctx", action="clear"),
        ),
        success_response={
            "status": "completed",
            "content": "At home position.",
        },
        dry_run_description="Moving arm to home position",
        dry_run_duration=3.0,
        failure_part="",
        notes=(
            "RobotAgent.move_home maps to the controller's home/named-pose motion when that named pose is available.",
        ),
    ),
)

__all__ = ["ROBOT_TASK_DEFINITION"]

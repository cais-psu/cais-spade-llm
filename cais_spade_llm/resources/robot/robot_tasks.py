"""Compatibility facade for the declarative robot task registry."""

from __future__ import annotations

from .robot_task_model import (
    RobotTaskArgument,
    RobotTaskDefinition,
    RobotTaskEffect,
    RobotTaskExposure,
    RobotTaskGuard,
    RobotTaskProgram,
    RobotTaskStep,
)
from .robot_task_recovery import robot_recovery_des_descriptor  # noqa: F401
from .robot_task_registry import (
    resolve_robot_task_names,
    robot_task_capability_context,
    robot_task_capability_decompositions,
    robot_task_docstring,
    robot_task_names,
    robot_task_registry,
)
from .robot_task_runtime import execute_robot_task

__all__ = [
    "RobotTaskArgument",
    "RobotTaskDefinition",
    "RobotTaskEffect",
    "RobotTaskExposure",
    "RobotTaskGuard",
    "RobotTaskProgram",
    "RobotTaskStep",
    "execute_robot_task",
    "resolve_robot_task_names",
    "robot_task_capability_context",
    "robot_task_capability_decompositions",
    "robot_task_docstring",
    "robot_task_names",
    "robot_task_registry",
]

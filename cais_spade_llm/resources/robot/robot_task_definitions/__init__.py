"""Exact declarative definitions for the five robot tasks."""

from __future__ import annotations

from .move_home import ROBOT_TASK_DEFINITION as MOVE_HOME_TASK
from .pick_approach import ROBOT_TASK_DEFINITION as PICK_APPROACH_TASK
from .pick_grasp import ROBOT_TASK_DEFINITION as PICK_GRASP_TASK
from .place_approach import ROBOT_TASK_DEFINITION as PLACE_APPROACH_TASK
from .place_insert import ROBOT_TASK_DEFINITION as PLACE_INSERT_TASK

__all__ = [
    "MOVE_HOME_TASK",
    "PICK_APPROACH_TASK",
    "PICK_GRASP_TASK",
    "PLACE_APPROACH_TASK",
    "PLACE_INSERT_TASK",
]

"""Registry and capability views for the five robot task definitions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any

from .robot_task_definitions import (
    MOVE_HOME_TASK,
    PICK_APPROACH_TASK,
    PICK_GRASP_TASK,
    PLACE_APPROACH_TASK,
    PLACE_INSERT_TASK,
)
from .robot_task_model import RobotTaskDefinition


def _make_robot_task_handler(function_name: str) -> Callable[..., Awaitable[dict[str, Any]]]:
    async def _handler(self: Any, **kwargs: Any) -> dict[str, Any]:
        return await self._execute_registered_robot_task(function_name, **kwargs)

    _handler.__name__ = function_name
    _handler.__qualname__ = f"RobotAgent.{function_name}"
    return _handler


def _register_robot_task(task: RobotTaskDefinition) -> RobotTaskDefinition:
    handler = _make_robot_task_handler(task.name)
    object.__setattr__(task, "handler", handler)
    handler.__doc__ = task.rendered_docstring()
    handler.__tool_spec__ = task
    return task


_TASK_ORDER = (
    "pick_approach",
    "pick_grasp",
    "place_approach",
    "move_home",
    "place_insert",
)


_ROBOT_TASKS: tuple[RobotTaskDefinition, ...] = (
    _register_robot_task(PICK_APPROACH_TASK),
    _register_robot_task(PICK_GRASP_TASK),
    _register_robot_task(PLACE_APPROACH_TASK),
    _register_robot_task(PLACE_INSERT_TASK),
    _register_robot_task(MOVE_HOME_TASK),
)


def robot_task_registry() -> dict[str, RobotTaskDefinition]:
    """Return all registered robot task definitions by exact function name."""
    return {task.name: task for task in _ROBOT_TASKS}


def robot_task_names() -> tuple[str, ...]:
    """Return registered robot task names in their public order."""
    registry = robot_task_registry()
    ordered = [name for name in _TASK_ORDER if name in registry]
    ordered.extend(task.name for task in _ROBOT_TASKS if task.name not in ordered)
    return tuple(ordered)


def robot_task_capability_context(
    *,
    static_capabilities: dict[str, Any] | None = None,
    named_positions: dict[str, Any] | None = None,
    controller_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the context used to decide which task definitions are exposed."""
    return {
        "static_capabilities": deepcopy(static_capabilities or {}),
        "named_positions": deepcopy(named_positions or {}),
        "controller_config": deepcopy(controller_config or {}),
    }


def resolve_robot_task_names(
    *,
    static_capabilities: dict[str, Any] | None = None,
    named_positions: dict[str, Any] | None = None,
    controller_config: dict[str, Any] | None = None,
    requested_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> tuple[str, ...]:
    """Return the task names enabled by the supplied robot configuration."""
    capability_context = robot_task_capability_context(
        static_capabilities=static_capabilities,
        named_positions=named_positions,
        controller_config=controller_config,
    )
    registry = robot_task_registry()
    enabled = [
        name
        for name in robot_task_names()
        if name in registry and registry[name].is_enabled(capability_context)
    ]
    if requested_names is None:
        return tuple(enabled)
    requested = [str(name or "").strip() for name in requested_names if str(name or "").strip()]
    if not requested:
        return tuple(enabled)
    requested_set = set(requested)
    return tuple(name for name in enabled if name in requested_set)


def robot_task_docstring(function_name: str) -> str:
    """Render the tool docstring for one exact robot task name."""
    task = robot_task_registry().get(str(function_name or "").strip())
    return task.rendered_docstring() if task is not None else ""


def robot_task_capability_decompositions(
    *,
    function_name: str = "",
    resource_jid: str = "",
) -> dict[str, Any]:
    """Render recovery-visible primitive composition from the task registry."""
    registry = robot_task_registry()
    token = str(function_name or "").strip()
    if token:
        task = registry.get(token)
        return task.capability_decomposition(resource_jid=resource_jid) if task is not None else {}
    return {
        name: task.capability_decomposition(resource_jid=resource_jid)
        for name, task in registry.items()
    }


__all__ = [
    "resolve_robot_task_names",
    "robot_task_capability_context",
    "robot_task_capability_decompositions",
    "robot_task_docstring",
    "robot_task_names",
    "robot_task_registry",
]

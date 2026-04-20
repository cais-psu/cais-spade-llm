"""Shared Gazebo cleanup flags for UI/debug lifecycle paths."""

from __future__ import annotations

import os


_TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in _TRUE_VALUES


def keep_gazebo_on_exit() -> bool:
    return env_flag_enabled("CAIS_KEEP_GAZEBO_ON_EXIT")

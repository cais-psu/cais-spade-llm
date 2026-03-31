"""Shared part-state semantics for bridge and repair logic.

This centralizes the product/bridge part-state vocabulary so prompt shaping,
validation, and recovery heuristics do not each maintain their own ad hoc
state buckets.

The state labels here reflect the vocabulary already used across:
  - ``ProductAgent.part_tracker`` updates
  - robot bridge event projection
  - bridge/test harness part_tracker fixtures
"""

from __future__ import annotations

from typing import Any

_POSE_UNCERTAIN_PART_STATES = frozenset(
    {
        "misplaced",
        "displaced",
        "unknown",
        "untracked",
    }
)

_STABLY_GROUNDED_PART_STATES = frozenset(
    {
        "in_gripper",
        "in_transit",
        "ready",
        "assembled",
    }
)

_CARRIED_PART_STATES = frozenset(
    {
        "in_gripper",
        "in_transit",
    }
)


def normalize_part_state(state: Any) -> str:
    """Return the canonical lowercase token for a part state value."""
    return str(state or "").strip().lower()


def part_state_requires_external_localization(state: Any) -> bool:
    """Whether the state implies external observation is still needed."""
    return normalize_part_state(state) in _POSE_UNCERTAIN_PART_STATES


def part_state_is_stably_grounded(state: Any) -> bool:
    """Whether the state is normally execution-grounded without fresh observe."""
    return normalize_part_state(state) in _STABLY_GROUNDED_PART_STATES


def part_state_is_carried(state: Any) -> bool:
    """Whether the part is currently carried by a resource rather than free."""
    return normalize_part_state(state) in _CARRIED_PART_STATES

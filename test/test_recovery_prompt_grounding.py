"""Focused tests for the grounding surfaces the outline prompt publishes.

A named location and an `action_target` pose are alternative ways to state the
same target, so the prompt must not hand out both. The published geometry stays
available to safety validation and to the response schema even though the
prompt withholds it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.central_controller.outline_macro_safety import (
    _location_area_rows,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
    multi_turn,
    multi_turn_prompts,
)

BOARD = "assembly_board-v1"


def _public_locations() -> list[dict[str, Any]]:
    return [
        {
            "resource_jid": "ur5e@localhost",
            "locations": [
                {"location": "home"},
                {
                    "location": "prusa-mk4-2",
                    "pose": {
                        "frame": "world",
                        "units": "m",
                        "x": 0.4,
                        "y": 0.3,
                        "z": 1.04,
                    },
                },
                {
                    "location": BOARD,
                    "area": {
                        "frame": "world",
                        "units": "m",
                        "bounds": {
                            "x": {"min": -0.175, "max": 0.175},
                            "y": {"min": -0.15, "max": 0.15},
                        },
                    },
                },
            ],
        }
    ]


def test_slimmed_locations_publish_names_without_geometry() -> None:
    slimmed = multi_turn_prompts._slim_public_locations(_public_locations())
    rows = slimmed[0]["locations"]

    assert rows == [
        {"location": "home"},
        {"location": "prusa-mk4-2"},
        {"location": BOARD},
    ]


def test_slimming_does_not_mutate_the_published_locations() -> None:
    """Safety and schema builders keep reading the unslimmed source."""
    public_locations = _public_locations()
    original = deepcopy(public_locations)

    multi_turn_prompts._slim_public_locations(public_locations)

    assert public_locations == original
    assert public_locations[0]["locations"][1]["pose"]["x"] == 0.4


def test_action_target_schema_enum_survives_slimming() -> None:
    """The response schema's location enum comes from names, not poses."""
    public_action_targets = [
        {
            "resource_jid": "ur5e@localhost",
            "action_target": {
                "fields": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                },
                "required": ["x", "y", "z"],
                "frame": "world",
                "units": "m",
            },
        }
    ]
    from_source = multi_turn_prompts._resource_action_target_schemas(
        public_action_targets=public_action_targets,
        public_locations=_public_locations(),
    )
    from_slimmed = multi_turn_prompts._resource_action_target_schemas(
        public_action_targets=public_action_targets,
        public_locations=multi_turn_prompts._slim_public_locations(
            _public_locations()
        ),
    )
    assert from_source == from_slimmed


def test_location_areas_still_resolve_for_safety() -> None:
    rows = _location_area_rows({"public_locations": _public_locations()})
    assert [row["location"] for row in rows] == [BOARD]
    assert rows[0]["area"]["bounds"]["y"] == {"min": -0.15, "max": 0.15}


def test_action_targets_render_from_the_declared_contract() -> None:
    rendered = multi_turn_prompts._compact_action_targets(
        [
            {
                "resource_jid": "ur5e@localhost",
                "action_target": {
                    "fields": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                    },
                    "required": ["x", "y", "z"],
                    "frame": "world",
                    "units": "m",
                },
            }
        ],
        _public_locations(),
    )
    assert rendered == (
        '- ur5e@localhost: {"location": one of home, prusa-mk4-2, '
        'assembly_board-v1} or {"x": number, "y": number, "z": number} '
        "[frame=world units=m]"
    )


def test_action_targets_generalize_beyond_poses() -> None:
    """A resource that declares non-spatial fields publishes them unchanged."""
    rendered = multi_turn_prompts._compact_action_targets(
        [
            {
                "resource_jid": "prusa-mk4-2@localhost",
                "action_target": {
                    "fields": {
                        "material": {"type": "string"},
                        "temperature": {"type": "number"},
                    },
                    "required": ["material"],
                },
            }
        ],
        [],
    )
    assert rendered == (
        '- prusa-mk4-2@localhost: {"material": string, "temperature": number}'
    )


def test_all_null_part_state_is_normalized_away() -> None:
    """A no-part candidate must not be read as authoring a part state."""
    normalized = multi_turn._without_unasserted_state_blocks(
        {
            "resource_state": {"condition": "idle", "location": "home"},
            "part_state": {"condition": None, "location": None},
        }
    )
    assert normalized == {"resource_state": {"condition": "idle", "location": "home"}}


def test_partially_null_state_blocks_still_assert() -> None:
    state = {
        "resource_state": {"condition": "idle", "location": None},
        "part_state": {"condition": "in_gripper", "location": None},
    }
    assert multi_turn._without_unasserted_state_blocks(state) == state


def test_normalization_keeps_non_compound_fields() -> None:
    state = {"held_part": None, "part_state": {"condition": None, "location": None}}
    assert multi_turn._without_unasserted_state_blocks(state) == {"held_part": None}

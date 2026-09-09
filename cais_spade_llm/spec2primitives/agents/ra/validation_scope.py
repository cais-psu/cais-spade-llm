from __future__ import annotations

"""Keep recorded validation scope consistent across authoring and execution."""

from collections.abc import Mapping
from typing import Any

VALIDATION_SCOPE = "rigid_vertical_gear_assembly_direct_cartesian"
GAZEBO_PICK_PLACE_SCOPE = "gazebo_pick_place_direct_cartesian"
GAZEBO_OBSERVED_SCOPE = "gazebo_pick_place_observed_geometry_direct_cartesian"

_SUPPORTED = frozenset({
    "compute_pick_targets",
    "compute_place_targets",
    "move_cartesian",
    "grasp_part",
    "release_part",
})


def read_validation_scope(record: Mapping[str, Any]) -> str:
    """Read an explicit scope, preserving the assembly scope of historical records."""
    # Older saved profiles and authoring requests predate this field. A changed
    # application default must never reinterpret their validation authority.
    scope = record.get("validation_scope", VALIDATION_SCOPE)
    if not isinstance(scope, str) or scope not in {VALIDATION_SCOPE, GAZEBO_PICK_PLACE_SCOPE, GAZEBO_OBSERVED_SCOPE}:
        raise ValueError(f"Unknown validation scope: {scope!r}.")
    return scope


def required_validation_roles(scope: str) -> tuple[str, ...]:
    """Return validation evidence roles separately from selected primitive inputs."""
    scope = read_validation_scope({"validation_scope": scope})
    if is_pick_place_scope(scope):
        return ("part", "scene")
    return ("part", "goal", "scene", "specification")


def is_pick_place_scope(scope: str) -> bool:
    """Identify scopes that check simulated custody rather than assembly seating."""
    return read_validation_scope({"validation_scope": scope}) in {
        GAZEBO_PICK_PLACE_SCOPE, GAZEBO_OBSERVED_SCOPE,
    }


def supported_primitive_symbols(scope: str) -> frozenset[str]:
    """Return primitives with both an owned validator and an execution handler."""
    read_validation_scope({"validation_scope": scope})
    return _SUPPORTED


def validation_scope_instruction(scope: str) -> str:
    """Describe the recorded acceptance criteria without prescribing a program."""
    scope = read_validation_scope({"validation_scope": scope})
    if scope == GAZEBO_OBSERVED_SCOPE:
        return (
            f"Validation scope: {scope}. Use approved observed bounds and support surfaces "
            "for Gazebo pick-and-place. ObservedGeometryEvidence establishes an approximate "
            "box and its observed_bounds_center reference, not a unique CAD pose. "
            "compute_pick_targets uses its center, measured part_height_m and board_center.z. "
            "compute_place_targets uses product_geometry.placement_surface_point (world XYZ "
            "on a PA-selected destination surface), the part height and the held reference offset. "
            "No slot, shaft seating, CAD-origin pose or physical gripper-fit proof is required. "
            "Only part and scene validation roles are required; goal and specification stay null. "
            "Motion, observed collision coverage, part proximity, custody and execution "
            "acknowledgments remain required. RA chooses all operations and bindings. "
        )
    if scope == GAZEBO_PICK_PLACE_SCOPE:
        return (
            f"Validation scope: {scope}. Check grounded pick-and-place calculations, "
            "direct motion, observed collision geometry and grasp/release custody. "
            "Execution requires gripper and link-attacher acknowledgments. "
            "Precise seating and assembly tolerances are not acceptance requirements. "
            "Only part and scene are required validation roles; goal and specification "
            "remain null. Measurements needed by selected primitive inputs are still required. "
        )
    return (
        f"Validation scope: {scope}. Part, goal, scene and specification evidence are "
        "required to check rigid vertical assembly, including seating and approved tolerances. "
    )

from __future__ import annotations

"""Keep recorded validation scope consistent across authoring and execution."""

from collections.abc import Mapping
from typing import Any

VALIDATION_SCOPE = "rigid_vertical_gear_assembly_direct_cartesian"
GAZEBO_PICK_PLACE_SCOPE = "gazebo_pick_place_direct_cartesian"
GAZEBO_OBSERVED_SCOPE = "gazebo_pick_place_observed_geometry_direct_cartesian"
GAZEBO_LINK_ATTACHER_SCOPE = "gazebo_link_attacher_direct_cartesian"

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
    if not isinstance(scope, str) or scope not in {
        VALIDATION_SCOPE, GAZEBO_PICK_PLACE_SCOPE, GAZEBO_OBSERVED_SCOPE, GAZEBO_LINK_ATTACHER_SCOPE,
    }:
        raise ValueError(f"Unknown validation scope: {scope!r}.")
    return scope


def required_validation_roles(scope: str, target_feature: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Return validation evidence roles separately from selected primitive inputs."""
    scope = read_validation_scope({"validation_scope": scope})
    if is_observed_scope(scope) and any(
        "desired_state" in association.get("state_names", [])
        and len(association.get("assembly_features", [])) == 2
        for association in (target_feature or {}).get("assembly_feature_association", [])
    ):
        return ("part", "goal", "scene")
    if is_pick_place_scope(scope):
        return ("part", "scene")
    return ("part", "goal", "scene", "specification")


def is_pick_place_scope(scope: str) -> bool:
    """Identify scopes using simulated pick-and-place custody checks."""
    return read_validation_scope({"validation_scope": scope}) in {
        GAZEBO_PICK_PLACE_SCOPE, GAZEBO_OBSERVED_SCOPE, GAZEBO_LINK_ATTACHER_SCOPE,
    }


def is_observed_scope(scope: str) -> bool:
    """Identify scopes grounded in observed bounds and measured destination geometry."""
    return read_validation_scope({"validation_scope": scope}) in {
        GAZEBO_OBSERVED_SCOPE, GAZEBO_LINK_ATTACHER_SCOPE,
    }


def supported_primitive_symbols(scope: str) -> frozenset[str]:
    """Return primitives with both an owned validator and an execution handler."""
    read_validation_scope({"validation_scope": scope})
    return _SUPPORTED


def validation_scope_instruction(scope: str) -> str:
    """Describe the recorded acceptance criteria without prescribing a program."""
    scope = read_validation_scope({"validation_scope": scope})
    if scope == GAZEBO_LINK_ATTACHER_SCOPE:
        return (
            f"Validation scope: {scope}. Evaluate primitive composition using Gazebo link attachment. "
            "Use PA-qualified observed part bounds, support surfaces and the accepted destination relationship. "
            "compute_pick_targets uses observed_bounds_center, measured part_height_m and board_center.z. "
            "compute_place_targets uses the measured placement_surface_point and held part-to-tool transform. "
            "An accepted desired assembly relationship requires checked goal geometry: target_origin_pose, "
            "insertion_axis, insertion_distance_m and part_height_m from the same accepted pair. "
            "Reach that final part reference before release; pre_insert_pose alone is not placement. "
            "Check position and insertion-axis alignment against robot motion tolerances. "
            "Intended contact between the selected part and its checked destination/seat is allowed. "
            "Bore clearance, press-fit mechanics and micrometre seating are not acceptance criteria. "
            "Other collisions, complete Cartesian motion, correct part identity, grasp proximity and custody remain checked. "
            "Execution acknowledges gripper attachment, detachment and attachment to the configured assembly board at release. "
            "No placement snap repairs an incorrect program. This validates simulated primitive composition, not physical assembly. "
            "Missing input paths and required evidence roles are derived automatically; return context_requests=[] "
            "instead of prose requests repeating these checks. Motion and execution checks belong to RA, not PA. "
            "RA chooses every primitive, its order and bindings. "
        )
    if scope == GAZEBO_OBSERVED_SCOPE:
        return (
            f"Validation scope: {scope}. Use approved observed bounds and support surfaces "
            "for Gazebo pick-and-place. ObservedGeometryEvidence establishes an approximate "
            "box and its observed_bounds_center reference, not a unique CAD pose. "
            "compute_pick_targets uses its center, measured part_height_m and board_center.z. "
            "compute_place_targets uses product_geometry.placement_surface_point (world XYZ "
            "on a PA-selected destination surface), the part height and the held reference offset. "
            "An accepted desired assembly relationship additionally requires checked goal geometry, including an Assembly destination. "
            "Endpoint owner types do not establish the mating shape. PA must qualify the accepted moving and destination bindings "
            "against approved CAD and observed features. A through bore on the moving component, a solid or hollow circular shaft "
            "and an actual seating surface support nominal straight circular insertion. Those checked features supply "
            "target_origin_pose, insertion_axis, insertion_distance_m and part_height_m; CAD yaw remains unresolved. "
            "pre_insert_pose is above shaft engagement; insert_pose establishes nominal seating within measured surface resolution. "
            "A release above the seat or nonpositive measured clearance cannot satisfy that goal. "
            "Threading, press fits, snap fits and other mating shapes are unsupported; report the specific coverage gap "
            "instead of requesting shaft measurements or substituting pick-and-place success. "
            "Missing manufacturing tolerances do not prevent nominal simulation checks. Robustness to measurement errors "
            "and physical assembly success remain unproven. "
            "Other observed pick-and-place requests require only part and scene. "
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

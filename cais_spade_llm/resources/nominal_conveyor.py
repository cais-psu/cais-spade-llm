"""Pure, task-boundary occupancy projections for the nominal Conveyor DES.

These functions describe symbolic outcomes, never command a belt. Regions denote
order and loading clearance, not equal travel distances or independent drives.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

CONVEYOR_LOCATIONS = (
    "loading_position_1",
    "after loading_position_1",
    "loading_position_2",
    "after loading_position_2",
    "output_nest",
)


def conveyor_known_parts(valuation: dict[str, Any]) -> list[str]:
    """Read registered occupancy records without using resource eligibility lists."""
    return [field.split(".", 1)[1] for field in valuation if field.startswith("part_location.")]


def _location_interval(model: dict, location: str) -> tuple[Fraction, Fraction, bool]:
    loading = {
        row["loading_position"]: Fraction(str(row["pose"][0]))
        for row in model["assignments"]["loading_positions"].values()
    }
    points = [
        loading["loading_position_1"],
        loading["loading_position_2"],
        Fraction(str(model["assignments"]["output_nest_pose"][0])),
    ]
    if location == "Buffer For Machined parts":
        position = Fraction(str(model["assignments"]["buffer_zone_1_pose"][0]))
        return position, position, False
    index = CONVEYOR_LOCATIONS.index(location)
    if index % 2 == 0:
        return points[index // 2], points[index // 2], False
    return points[index // 2], points[index // 2 + 1], True


def _check_shared_displacement(
    model: dict,
    valuation: dict,
    next_locations: dict,
    ordered: list[str],
    delivered_part: str | None,
) -> None:
    # Intersect possible displacements of all occupied regions. This checks a
    # common belt without selecting positions inside regions or a distance step.
    lower, lower_open = Fraction(0), True
    upper, upper_open = None, False
    for part in ordered:
        before = _location_interval(model, valuation[f"part_location.{part}"])
        after = _location_interval(
            model, "Buffer For Machined parts" if part == delivered_part else next_locations[part]
        )
        minimum, maximum = after[0] - before[1], after[1] - before[0]
        boundary_open = before[2] or after[2]
        if minimum > lower:
            lower, lower_open = minimum, boundary_open
        elif minimum == lower:
            lower_open = lower_open or boundary_open
        if upper is None or maximum < upper:
            upper, upper_open = maximum, boundary_open
        elif maximum == upper:
            upper_open = upper_open or boundary_open
    if upper is None or lower > upper or (lower == upper and (lower_open or upper_open)):
        raise ValueError("Conveyor outcome requires inconsistent movement of the shared belt")


def conveyor_parts(model: dict[str, Any], valuation: dict[str, Any]) -> list[str]:
    """Return resident part identities in downstream-to-upstream order.

    Args:
        model: The Conveyor descriptor.
        valuation: One symbolic Conveyor valuation.

    Returns:
        Exact part identifiers, with the leading part first.

    Raises:
        ValueError: Location and order disagree or an exclusive area is occupied twice.
    """
    parts = conveyor_known_parts(valuation)
    resident = [part for part in parts if valuation[f"part_location.{part}"] is not None]
    for part in parts:
        if part not in resident and valuation[f"part_order.{part}"] is not None:
            raise ValueError(f"Conveyor order without custody: {part}")
    ranks = [valuation[f"part_order.{part}"] for part in resident]
    if any(type(rank) is not int for rank in ranks) or sorted(ranks) != list(range(len(resident))):
        raise ValueError("Conveyor part_order must be a contiguous, unique downstream order")
    ordered = sorted(resident, key=lambda part: valuation[f"part_order.{part}"])
    locations = [valuation[f"part_location.{part}"] for part in ordered]
    if any(location not in CONVEYOR_LOCATIONS for location in locations):
        raise ValueError("Unknown Conveyor location")
    positions = [CONVEYOR_LOCATIONS.index(location) for location in locations]
    if positions != sorted(positions, reverse=True):
        raise ValueError("Conveyor part_order disagrees with part_location")
    for area in ("loading_position_1", "loading_position_2", "output_nest"):
        if locations.count(area) > 1:
            raise ValueError(f"Conveyor {area} is occupied more than once")
    return ordered


def conveyor_load_parameters(
    model: dict[str, Any], valuation: dict[str, Any], part_name: str, loading_position: str
) -> dict[str, Any]:
    """Calculate an insertion into the spatial order for a completed placement.

    Args:
        model: The Conveyor descriptor.
        valuation: The pre-placement symbolic valuation.
        part_name: Exact part identifier.
        loading_position: One configured loading area.

    Returns:
        Parameters for the descriptor's declared order updates.

    Raises:
        ValueError: The part is already resident or the loading area is occupied.
    """
    ordered = conveyor_parts(model, valuation)
    if part_name not in conveyor_known_parts(valuation) or part_name in ordered:
        raise ValueError("Conveyor cannot load an unknown or already resident part")
    if loading_position not in ("loading_position_1", "loading_position_2"):
        raise ValueError("Unknown Conveyor loading position")
    if any(valuation[f"part_location.{part}"] == loading_position for part in ordered):
        raise ValueError(f"Conveyor {loading_position} is occupied")
    position = CONVEYOR_LOCATIONS.index(loading_position)
    ahead = [
        part
        for part in ordered
        if CONVEYOR_LOCATIONS.index(valuation[f"part_location.{part}"]) > position
    ]
    ordered.insert(len(ahead), part_name)
    return {
        f"next_part_order.{part}": ordered.index(part) if part in ordered else None
        for part in conveyor_known_parts(valuation)
    }


def conveyor_advance_parameters(
    model: dict[str, Any],
    valuation: dict[str, Any],
    next_locations: dict[str, str],
    delivered_part: str | None,
) -> dict[str, Any]:
    """Validate one coupled movement and bind its complete symbolic outcome.

    Args:
        model: The Conveyor descriptor.
        valuation: The pre-movement symbolic valuation.
        next_locations: A location for every part remaining on the belt.
        delivered_part: The leading part acknowledged by buffer zone 1, or None.

    Returns:
        Parameters for all declared Conveyor location and order updates.

    Raises:
        ValueError: The outcome loses, duplicates, reverses, or overtakes a part,
            leaves a part in an exclusive loading/output area during movement,
            or attempts to deliver a part that has not reached the output.
    """
    ordered = conveyor_parts(model, valuation)
    if not ordered:
        raise ValueError("Conveyor has no part to advance")
    if not isinstance(next_locations, dict):
        raise ValueError("next_locations must bind every remaining Conveyor part")
    if delivered_part is not None:
        if delivered_part != ordered[0]:
            raise ValueError("Only the leading Conveyor part may enter the buffer")
        if valuation[f"part_location.{delivered_part}"] != "output_nest":
            raise ValueError("The delivered part must first reach output_nest")
    remaining = [part for part in ordered if part != delivered_part]
    if set(next_locations) != set(remaining):
        raise ValueError("Conveyor movement must account for every resident part exactly once")
    candidate = dict(valuation)
    changed = delivered_part is not None
    for part in conveyor_known_parts(valuation):
        before = valuation[f"part_location.{part}"]
        after = next_locations.get(part)
        if after is not None:
            if after not in CONVEYOR_LOCATIONS:
                raise ValueError(f"Unknown Conveyor location: {after}")
            old_position = CONVEYOR_LOCATIONS.index(before)
            new_position = CONVEYOR_LOCATIONS.index(after)
            if new_position < old_position:
                raise ValueError("Conveyor movement cannot move a part upstream")
            # Parts may remain within a broad intervening region while moving.
            # A stationary part at an entry/output point would require another drive.
            if before == after and before in (
                "loading_position_1",
                "loading_position_2",
                "output_nest",
            ):
                raise ValueError("Shared belt movement must clear occupied loading/output points")
            changed = changed or after != before
        candidate[f"part_location.{part}"] = after
        candidate[f"part_order.{part}"] = remaining.index(part) if part in remaining else None
    if not changed:
        raise ValueError("advance_conveyor requires a changed occupancy boundary")
    conveyor_parts(model, candidate)
    _check_shared_displacement(model, valuation, next_locations, ordered, delivered_part)
    return {
        f"next_{field}": value
        for field, value in candidate.items()
        if field.startswith(("part_location.", "part_order."))
    }

"""Actual STL geometry used to ground the physical MG grasp."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BINARY_STL_HEADER_SIZE = 84
_BINARY_STL_TRIANGLE_SIZE = 50


def _resolved_source_stl(source_stl: str) -> Path:
    path = Path(str(source_stl or "").strip()).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _read_binary_stl_triangles(path: Path) -> tuple[bytes, tuple[tuple[float, ...], ...]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read actual MG STL {path}: {exc}") from exc
    if len(payload) < _BINARY_STL_HEADER_SIZE:
        raise ValueError(f"actual MG STL is incomplete: {path}")

    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    expected_size = _BINARY_STL_HEADER_SIZE + triangle_count * _BINARY_STL_TRIANGLE_SIZE
    if triangle_count <= 0 or len(payload) != expected_size:
        raise ValueError(
            f"actual MG STL is not a valid binary STL: {path} "
            f"(triangles={triangle_count}, bytes={len(payload)}, expected={expected_size})"
        )

    triangles: list[tuple[float, ...]] = []
    for triangle_index in range(triangle_count):
        offset = _BINARY_STL_HEADER_SIZE + triangle_index * _BINARY_STL_TRIANGLE_SIZE + 12
        vertices = struct.unpack_from("<9f", payload, offset)
        if not all(math.isfinite(value) for value in vertices):
            raise ValueError(
                f"actual MG STL contains non-finite vertex data at triangle {triangle_index}"
            )
        triangles.append(vertices)
    return payload, tuple(triangles)


def _outer_radius_at_z(
    triangles: tuple[tuple[float, ...], ...],
    *,
    center_x: float,
    center_y: float,
    z: float,
) -> float:
    maximum_radius = 0.0
    intersections = 0
    for values in triangles:
        vertices = (
            values[0:3],
            values[3:6],
            values[6:9],
        )
        for start, end in (
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]),
            (vertices[2], vertices[0]),
        ):
            start_z = start[2]
            end_z = end[2]
            if abs(end_z - start_z) <= 1e-12:
                continue
            if not (min(start_z, end_z) <= z <= max(start_z, end_z)):
                continue
            fraction = (z - start_z) / (end_z - start_z)
            x = start[0] + fraction * (end[0] - start[0])
            y = start[1] + fraction * (end[1] - start[1])
            maximum_radius = max(maximum_radius, math.hypot(x - center_x, y - center_y))
            intersections += 1
    if intersections == 0 or maximum_radius <= 0.0:
        raise ValueError(f"actual MG STL has no closed cross-section at z={z:.6f}")
    return maximum_radius


@lru_cache(maxsize=8)
def _load_actual_mg_stl_geometry_cached(
    source_stl: str,
    modified_time_ns: int,
    file_size: int,
    scale_m_per_unit: float,
    hub_up: bool,
) -> dict[str, Any]:
    _ = (modified_time_ns, file_size)
    path = Path(source_stl)
    payload, triangles = _read_binary_stl_triangles(path)
    coordinates = [
        (values[index], values[index + 1], values[index + 2])
        for values in triangles
        for index in (0, 3, 6)
    ]
    min_x = min(vertex[0] for vertex in coordinates)
    max_x = max(vertex[0] for vertex in coordinates)
    min_y = min(vertex[1] for vertex in coordinates)
    max_y = max(vertex[1] for vertex in coordinates)
    min_z = min(vertex[2] for vertex in coordinates)
    max_z = max(vertex[2] for vertex in coordinates)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    height_units = max_z - min_z
    if height_units <= 0.0:
        raise ValueError(f"actual MG STL has zero height: {path}")

    lower_radius = _outer_radius_at_z(
        triangles,
        center_x=center_x,
        center_y=center_y,
        z=min_z + height_units * 0.25,
    )
    upper_radius = _outer_radius_at_z(
        triangles,
        center_x=center_x,
        center_y=center_y,
        z=min_z + height_units * 0.75,
    )
    hub_on_lower_side = lower_radius < upper_radius
    hub_radius_units = min(lower_radius, upper_radius)
    tooth_radius_units = max(lower_radius, upper_radius)
    if tooth_radius_units - hub_radius_units <= 0.5:
        raise ValueError(f"actual MG STL has no distinct smooth raised hub: {path}")

    transition_radius = (hub_radius_units + tooth_radius_units) * 0.5
    lower_bound = min_z + height_units * 0.25
    upper_bound = min_z + height_units * 0.75
    for _index in range(24):
        midpoint = (lower_bound + upper_bound) * 0.5
        midpoint_radius = _outer_radius_at_z(
            triangles,
            center_x=center_x,
            center_y=center_y,
            z=midpoint,
        )
        midpoint_is_hub = midpoint_radius < transition_radius
        if midpoint_is_hub == hub_on_lower_side:
            if hub_on_lower_side:
                lower_bound = midpoint
            else:
                upper_bound = midpoint
        elif hub_on_lower_side:
            upper_bound = midpoint
        else:
            lower_bound = midpoint
    transition_z = (lower_bound + upper_bound) * 0.5
    hub_height_units = transition_z - min_z if hub_on_lower_side else max_z - transition_z

    part_height_m = height_units * scale_m_per_unit
    hub_height_m = hub_height_units * scale_m_per_unit
    tooth_height_m = part_height_m - hub_height_m
    hub_diameter_m = hub_radius_units * 2.0 * scale_m_per_unit
    tooth_diameter_m = tooth_radius_units * 2.0 * scale_m_per_unit
    expected_ranges = {
        "part height": (part_height_m, 0.018, 0.022),
        "smooth hub diameter": (hub_diameter_m, 0.028, 0.032),
        "smooth hub height": (hub_height_m, 0.008, 0.012),
        "tooth diameter": (tooth_diameter_m, 0.040, 0.044),
    }
    for label, (value, minimum, maximum) in expected_ranges.items():
        if not minimum <= value <= maximum:
            raise ValueError(
                f"actual MG STL {label} is outside the physical MG range: {value:.6f} m from {path}"
            )
    if not hub_up:
        raise ValueError("physical MG requires hub_up=true for the smooth raised hub grasp")

    return {
        "source_stl": str(path),
        "source_stl_sha256": hashlib.sha256(payload).hexdigest(),
        "hub_up": True,
        "part_height_m": part_height_m,
        "hub_diameter_m": hub_diameter_m,
        "hub_height_m": hub_height_m,
        "tooth_diameter_m": tooth_diameter_m,
        "tooth_height_m": tooth_height_m,
    }


def actual_mg_stl_geometry_for_part(
    part_name: str,
    parts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return actual STL-derived geometry when the real MG config requests it."""
    part_key = str(part_name or "").strip()
    part_config = dict(parts or {})
    source_stl = str(dict(part_config.get("source_stl_map") or {}).get(part_key) or "").strip()
    if not source_stl:
        return {}
    if part_key != "MG":
        raise ValueError(f"actual STL physical grasp geometry is not configured for {part_key}")

    hub_up_value = dict(part_config.get("hub_up") or {}).get(part_key)
    if not isinstance(hub_up_value, bool):
        raise ValueError("actual MG STL geometry requires boolean parts.hub_up.MG")
    try:
        scale_m_per_unit = float(
            dict(part_config.get("source_stl_scale_m_per_unit") or {}).get(part_key)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "actual MG STL geometry requires numeric parts.source_stl_scale_m_per_unit.MG"
        ) from exc
    if not math.isfinite(scale_m_per_unit) or scale_m_per_unit <= 0.0:
        raise ValueError("actual MG STL scale must be finite and positive")

    path = _resolved_source_stl(source_stl)
    try:
        stat = path.stat()
    except OSError as exc:
        raise ValueError(f"actual MG STL is unavailable: {path}: {exc}") from exc
    geometry = dict(
        _load_actual_mg_stl_geometry_cached(
            str(path),
            stat.st_mtime_ns,
            stat.st_size,
            scale_m_per_unit,
            hub_up_value,
        )
    )

    def configured_distance(field_name: str) -> float:
        raw_value = dict(part_config.get(field_name) or {}).get(part_key)
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"actual MG STL geometry requires numeric parts.{field_name}.MG"
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"parts.{field_name}.MG must be finite and non-negative")
        return value

    grasp_width_preload_m = configured_distance("grasp_width_preload_m")
    tooth_clearance_m = configured_distance("tooth_clearance_m")
    minimum_hub_overlap_m = configured_distance("minimum_hub_overlap_m")
    grasp_width_m = geometry["hub_diameter_m"] - grasp_width_preload_m
    if grasp_width_m <= 0.0:
        raise ValueError("actual MG STL grasp width is not positive after closing preload")
    geometry.update(
        {
            "grasp_width_preload_m": grasp_width_preload_m,
            "grasp_width_m": grasp_width_m,
            "tooth_clearance_m": tooth_clearance_m,
            "minimum_hub_overlap_m": minimum_hub_overlap_m,
        }
    )
    return geometry


__all__ = ["actual_mg_stl_geometry_for_part"]

#!/usr/bin/env python3
"""Generate the fixed occupancy map for recovery-framework KMR navigation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


RESOLUTION_M = 0.05
ORIGIN_X_M = -10.5
ORIGIN_Y_M = -1.5
WIDTH = 240
HEIGHT = 126
FREE = 254
OCCUPIED = 0

# Ground-plane envelopes from the accepted recovery layout. Operational blue
# docking markers are intentionally omitted from the occupied geometry.
OBSTACLE_BOUNDS = {
    "Storage": (-9.65, -8.65, 1.675, 2.925),
    "M1": (-6.85, -5.15, 1.475, 3.125),
    "M2": (-3.45, -1.75, 1.475, 3.125),
    "ur5e-1 work area": (-6.80, -5.65, 0.72, 1.47),
    "ur5e-2 work area": (-3.40, -2.25, 0.72, 1.47),
    "Conveyor": (-6.75, -0.74, 0.32, 0.68),
    "Buffer For Machined parts": (-0.74, -0.26, 0.32, 0.68),
    "Assembly Station": (-0.75, 0.75, -0.95, 1.05),
}


def pixel_center(row: int, column: int) -> tuple[float, float]:
    """Return the world position at the center of a PGM pixel."""

    x = ORIGIN_X_M + (column + 0.5) * RESOLUTION_M
    y = ORIGIN_Y_M + (HEIGHT - row - 0.5) * RESOLUTION_M
    return x, y


def occupied_at(x: float, y: float) -> bool:
    """Return whether the fixed cell map marks a world coordinate occupied."""

    maximum_x = ORIGIN_X_M + WIDTH * RESOLUTION_M
    maximum_y = ORIGIN_Y_M + HEIGHT * RESOLUTION_M
    if not (ORIGIN_X_M < x < maximum_x and ORIGIN_Y_M < y < maximum_y):
        return True
    return any(
        minimum_x <= x <= maximum_x and minimum_y <= y <= maximum_y
        for minimum_x, maximum_x, minimum_y, maximum_y in OBSTACLE_BOUNDS.values()
    )


def map_pixels() -> bytes:
    """Build raw PGM pixels from the accepted fixed obstacle envelopes."""

    pixels = bytearray(WIDTH * HEIGHT)
    for row in range(HEIGHT):
        for column in range(WIDTH):
            x, y = pixel_center(row, column)
            border = row in (0, HEIGHT - 1) or column in (0, WIDTH - 1)
            pixels[row * WIDTH + column] = OCCUPIED if border or occupied_at(x, y) else FREE
    return bytes(pixels)


def footprint_is_free(x: float, y: float, yaw: float, padding: float = 0.03) -> bool:
    """Check the configured KMR rectangle against occupied pixel centers."""

    half_length = 1.19 / 2.0 + padding
    half_width = 0.72 / 2.0 + padding
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    for row in range(HEIGHT):
        for column in range(WIDTH):
            world_x, world_y = pixel_center(row, column)
            if not occupied_at(world_x, world_y):
                continue
            dx, dy = world_x - x, world_y - y
            local_x = cosine * dx + sine * dy
            local_y = -sine * dx + cosine * dy
            if abs(local_x) <= half_length and abs(local_y) <= half_width:
                return False
    return True


def write_map(path: Path) -> None:
    """Write the deterministic binary PGM used by Nav2 map_server."""

    header = (
        "P5\n"
        "# CAIS recovery framework fixed cell, 0.050 m/pixel\n"
        f"{WIDTH} {HEIGHT}\n"
        "255\n"
    ).encode("ascii")
    path.write_bytes(header + map_pixels())


def main() -> None:
    """Generate the map at an explicit output path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_map(args.output)


if __name__ == "__main__":
    main()

"""Outward-rounded bounds for configured joint interpolation and rigid geometry.

The model describes commanded trajectories and explicitly configured joint-error
envelopes. Those envelopes require separate owner execution validation.
Refinement cells are numerical evidence, never additional LTLf observations.
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

from cais_spade_llm.resources.resource_safety_preparation import validate_joint_trajectory


def interval(low, high=None) -> tuple[float, float]:
    """Validate a finite closed real interval."""
    high = low if high is None else high
    low, high = float(low), float(high)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError("Invalid continuous-motion interval")
    return low, high


def add(a, b):
    return math.nextafter(a[0] + b[0], -math.inf), math.nextafter(a[1] + b[1], math.inf)


def mul(a, b):
    values = [x * y for x in a for y in b]
    return math.nextafter(min(values), -math.inf), math.nextafter(max(values), math.inf)


def neg(a):
    return -a[1], -a[0]


@lru_cache(maxsize=16384)
def _sin_point(value: float) -> tuple:
    # Rational Taylor arithmetic avoids relying on a libm rounding guarantee.
    # On [-8, 8], Taylor's theorem bounds the remainder by |x|**81/81!.
    x = Fraction(value)
    total = sum(
        ((-1) ** i * x ** (2 * i + 1) / math.factorial(2 * i + 1) for i in range(40)), Fraction()
    )
    error = abs(x) ** 81 / math.factorial(81)
    return math.nextafter(float(total - error), -math.inf), math.nextafter(
        float(total + error), math.inf
    )


def sine(a):
    """Enclose sin over an interval, including every interior extremum."""
    if a[0] < -8 or a[1] > 8:
        return (-1.0, 1.0)
    values = [*_sin_point(a[0]), *_sin_point(a[1])]
    pi = (math.nextafter(math.pi, -math.inf), math.nextafter(math.pi, math.inf))
    for k in range(-3, 4):
        point = mul(interval(0.5 + k), pi)
        if point[0] <= a[1] and point[1] >= a[0]:
            values.append(1.0 if k % 2 == 0 else -1.0)
    return max(-1.0, min(values)), min(1.0, max(values))


def cosine(a):
    pi_half = (math.nextafter(math.pi / 2, -math.inf), math.nextafter(math.pi / 2, math.inf))
    return sine(add(a, pi_half))


def polynomial_bounds(coefficients, start=0.0, end=1.0) -> tuple:
    """Bound a power polynomial using exact rational Bernstein coefficients."""
    a, width = Fraction(start), Fraction(end) - Fraction(start)
    if width < 0:
        raise ValueError("Reversed polynomial interval")
    power = [Fraction(v) for v in coefficients]
    n = len(power) - 1
    shifted = [
        sum(
            (power[k] * math.comb(k, j) * a ** (k - j) * width**j for k in range(j, n + 1)),
            Fraction(),
        )
        for j in range(n + 1)
    ]
    bernstein = [
        sum(
            (shifted[k] * Fraction(math.comb(i, k), math.comb(n, k)) for k in range(i + 1)),
            Fraction(),
        )
        for i in range(n + 1)
    ]
    return math.nextafter(float(min(bernstein)), -math.inf), math.nextafter(
        float(max(bernstein)), math.inf
    )


def joint_coefficients(first, last, index, duration) -> list:
    """Match position, cubic, or quintic joint_trajectory_controller splines."""
    q0, q1 = Fraction(first["positions"][index]), Fraction(last["positions"][index])
    h = Fraction(duration)
    if h <= 0:
        raise ValueError("Joint trajectory needs strictly increasing times")
    if any(bool(first.get(key)) != bool(last.get(key)) for key in ("velocities", "accelerations")):
        raise ValueError("Mixed derivative availability requires another interpolation contract")
    velocity = bool(first.get("velocities")) and bool(last.get("velocities"))
    acceleration = bool(first.get("accelerations")) and bool(last.get("accelerations"))
    if acceleration and not velocity:
        raise ValueError("Acceleration without velocity interpolation is unsupported")
    d = q1 - q0
    if not velocity:
        return [q0, d]
    v0, v1 = Fraction(first["velocities"][index]) * h, Fraction(last["velocities"][index]) * h
    if not acceleration:
        return [q0, v0, 3 * d - 2 * v0 - v1, -2 * d + v0 + v1]
    a0 = Fraction(first["accelerations"][index]) * h * h
    a1 = Fraction(last["accelerations"][index]) * h * h
    return [
        q0,
        v0,
        a0 / 2,
        10 * d - 6 * v0 - 4 * v1 - 3 * a0 / 2 + a1 / 2,
        -15 * d + 8 * v0 + 7 * v1 + 3 * a0 / 2 - a1,
        6 * d - 3 * v0 - 3 * v1 - a0 / 2 + a1 / 2,
    ]


def _identity():
    return [[interval(int(i == j)) for j in range(4)] for i in range(4)]


def _matrix_product(a, b):
    return [[_sum(mul(a[i][k], b[k][j]) for k in range(4)) for j in range(4)] for i in range(4)]


def _sum(values):
    result = interval(0)
    for value in values:
        result = add(result, value)
    return result


def rotation(axis, angle):
    """Enclose an axis-angle homogeneous rotation using Rodrigues' formula."""
    axis = [float(v) for v in axis]
    if len(axis) != 3 or not math.isclose(sum(v * v for v in axis), 1.0, abs_tol=1e-12):
        raise ValueError("Joint axis must be a configured unit vector")
    c, s = cosine(angle), sine(angle)
    one_minus_c = add(interval(1), neg(c))
    norm = sum(Fraction(v) ** 2 for v in axis)
    low = high = math.sqrt(norm)
    while Fraction(low) ** 2 > norm:
        low = math.nextafter(low, -math.inf)
    while Fraction(high) ** 2 < norm:
        high = math.nextafter(high, math.inf)
    inverse = (
        math.nextafter(float(1 / Fraction(high)), -math.inf),
        math.nextafter(float(1 / Fraction(low)), math.inf),
    )
    axis = [mul(interval(v), inverse) for v in axis]
    x, y, z = axis
    zero = interval(0)
    cross = [[zero, neg(z), y], [z, zero, neg(x)], [neg(y), x, zero]]
    result = _identity()
    for i in range(3):
        for j in range(3):
            result[i][j] = _sum(
                (
                    mul(mul(axis[i], axis[j]), one_minus_c),
                    mul(cross[i][j], s),
                    c if i == j else interval(0),
                )
            )
    return result


def transform(xyz, rpy):
    """Construct an outward-rounded URDF origin transform."""
    result = _identity()
    for axis, value in zip(((0, 0, 1), (0, 1, 0), (1, 0, 0)), reversed(rpy), strict=True):
        result = _matrix_product(result, rotation(axis, interval(value)))
    for i, value in enumerate(xyz):
        result[i][3] = interval(value)
    return result


def quaternion_transform(xyz, quaternion):
    """Enclose a recorded quaternion transform without a rounded Euler conversion."""
    if len(quaternion) != 4:
        raise ValueError("A complete root quaternion is required")
    q = [Fraction(v) for v in quaternion]
    norm = sum(v * v for v in q)
    if norm <= 0:
        raise ValueError("Root quaternion has zero norm")
    x, y, z, w = q
    values = [
        [1 - 2 * (y * y + z * z) / norm, 2 * (x * y - z * w) / norm, 2 * (x * z + y * w) / norm],
        [2 * (x * y + z * w) / norm, 1 - 2 * (x * x + z * z) / norm, 2 * (y * z - x * w) / norm],
        [2 * (x * z - y * w) / norm, 2 * (y * z + x * w) / norm, 1 - 2 * (x * x + y * y) / norm],
    ]
    result = _identity()
    for i in range(3):
        for j in range(3):
            v = float(values[i][j])
            result[i][j] = (math.nextafter(v, -math.inf), math.nextafter(v, math.inf))
        result[i][3] = interval(xyz[i])
    return result


def kinematic_transforms(configuration: dict, positions: dict, origins: dict | None = None) -> dict:
    """Enclose configured FK for supplied joint intervals, without assuming motion."""
    origins = origins or {
        row["child"]: transform(row["xyz"], row["rpy"]) for row in configuration["joints"]
    }
    root = (
        quaternion_transform(configuration["root_xyz"], configuration["root_quaternion"])
        if "root_quaternion" in configuration
        else transform(configuration["root_xyz"], configuration["root_rpy"])
    )
    result = {configuration["root"]: root}
    for row in configuration["joints"]:
        matrix = _matrix_product(result[row["parent"]], origins[row["child"]])
        position = positions.get(row["name"], interval(*row["frozen_interval"])
                                 if "frozen_interval" in row else interval(row.get("frozen_position", 0)))
        if row["type"] in {"revolute", "continuous"}:
            matrix = _matrix_product(matrix, rotation(row["axis"], position))
        elif row["type"] == "prismatic":
            change = _identity()
            for i, value in enumerate(row["axis"]):
                change[i][3] = mul(interval(value), position)
            matrix = _matrix_product(matrix, change)
        result[row["child"]] = matrix
    return result


def representative_pose(matrix: list) -> list:
    """Return a display pose from enclosing FK; occupancy uses the bounds themselves."""
    from scipy.spatial.transform import Rotation

    return [sum(matrix[i][3]) / 2 for i in range(3)] + Rotation.from_matrix(
        [[sum(matrix[i][j]) / 2 for j in range(3)] for i in range(3)]
    ).as_quat().tolist()


class ContinuousMotion:
    """Pure configured kinematics over one unchanged joint trajectory.

    Configuration lists ordered URDF joints and bounded geometry components.
    Every movable joint must have an exact trajectory identity. Unsupported
    mimic, floating and planar joints are rejected by configuration loading.
    """

    def __init__(self, trajectory: dict, configuration: dict):
        if configuration.get("interpolation") != "splines":
            raise ValueError("Unsupported controller interpolation")
        self.configuration, self.trajectory = configuration, trajectory
        self.names = configuration["joint_names"]
        self.joint_position_error = configuration.get("joint_position_error", {})
        if self.joint_position_error and (
            set(self.joint_position_error) != set(self.names)
            or any(type(value) not in (float, int) or not math.isfinite(value) or value < 0
                   for value in self.joint_position_error.values())
        ):
            raise ValueError("Tracking bounds must cover every configured joint")
        validate_joint_trajectory(trajectory, self.names)
        self.points = trajectory["points"]
        self.exact_times = [
            Fraction(point["time_from_start"]["sec"])
            + Fraction(point["time_from_start"]["nanosec"], 1_000_000_000)
            for point in self.points
        ]
        self.times = [float(value) for value in self.exact_times]
        if self.times[0] != 0 or len(self.points) < 2:
            raise ValueError("Continuous trajectory must begin at zero and have a duration")
        self.duration = self.times[-1]
        self.coefficients = [
            [joint_coefficients(a, b, i, t1 - t0) for i in range(len(self.names))]
            for a, b, t0, t1 in zip(
                self.points, self.points[1:], self.exact_times, self.exact_times[1:], strict=False
            )
        ]
        self.origins = {
            row["child"]: transform(row["xyz"], row["rpy"]) for row in configuration["joints"]
        }
        known = {configuration["root"]}
        used = set()
        for row in configuration["joints"]:
            if row["parent"] not in known or row["child"] in known:
                raise ValueError("Kinematics must be an ordered tree")
            known.add(row["child"])
            if row["type"] != "fixed":
                if (
                    row["type"] not in {"revolute", "continuous", "prismatic"}
                    or row["name"] not in self.names
                    and "frozen_position" not in row
                ):
                    raise ValueError("Kinematics has an unresolved joint")
                if row["name"] in self.names:
                    used.add(row["name"])
        if used != set(self.names) or configuration["reference_link"] not in known:
            raise ValueError("Kinematics does not cover the configured joint identities")
        if not configuration["components"]:
            raise ValueError("Resource geometry components are required")
        for row in configuration["components"]:
            if row["link"] not in known or len(row["bounds"]) != 3:
                raise ValueError("Geometry refers to an unknown link")
            for pair in row["bounds"]:
                interval(*pair)

    def joints(self, start: float, end: float) -> dict:
        """Enclose joint positions throughout the requested closed interval."""
        if start < 0 or start > end or (end > self.exact_times[-1] and end != self.duration):
            raise ValueError("Continuous interval is outside its prepared trajectory")
        start = max(
            Fraction(0),
            start if isinstance(start, Fraction) else Fraction(math.nextafter(start, -math.inf)),
        )
        end = min(
            self.exact_times[-1],
            end if isinstance(end, Fraction) else Fraction(math.nextafter(end, math.inf)),
        )
        if start > end:
            raise ValueError("Continuous interval is reversed")
        result = {}
        for i, name in enumerate(self.names):
            values = []
            for k, (a, b) in enumerate(zip(self.exact_times, self.exact_times[1:], strict=False)):
                if max(a, start) <= min(b, end):
                    values.append(
                        polynomial_bounds(
                            self.coefficients[k][i],
                            (max(a, start) - a) / (b - a),
                            (min(b, end) - a) / (b - a),
                        )
                    )
            result[name] = add(
                (min(v[0] for v in values), max(v[1] for v in values)),
                interval(-self.joint_position_error.get(name, 0), self.joint_position_error.get(name, 0)),
            )
        return result

    def transforms(self, start: float, end: float) -> dict:
        """Enclose every configured link transform over the same interval."""
        return kinematic_transforms(self.configuration, self.joints(start, end), self.origins)

    def boxes(self, start: float, end: float) -> list:
        """Return bounds on every configured box's minimum and maximum coordinates."""
        matrices = self.transforms(start, end)
        result = []
        for component in self.configuration["components"]:
            matrix = matrices[component["link"]]
            result.append(transformed_box(matrix, component["bounds"], component["id"]))
        return result

    def carried_part_box(self, start: float, end: float, grasp_transform: list,
                         bounds: list, part: str) -> dict:
        """Enclose a rigidly attached part using the same continuous link bounds."""
        matrix = self.transforms(start, end)[self.configuration["reference_link"]]
        attached = _matrix_product(matrix, quaternion_transform(grasp_transform[:3], grasp_transform[3:]))
        return transformed_box(attached, bounds, part)

    def reference_pose(self, time: float) -> list:
        """Return a representative endpoint; safety always uses enclosing bounds."""
        return representative_pose(
            self.transforms(time, time)[self.configuration["reference_link"]]
        )


def occupancy(boxes: list, region: list) -> list[bool]:
    """Return all possible occupancy values; envelope overlap alone is uncertain."""
    unknown = False
    for box in boxes:
        low, high = box["minimum"], box["maximum"]
        if any(high[i][1] < region[i][0] or low[i][0] > region[i][1] for i in range(3)):
            continue
        if all(high[i][0] >= region[i][0] and low[i][1] <= region[i][1] for i in range(3)):
            return [True]
        unknown = True
    return [False, True] if unknown else [False]


def transformed_box(matrix: list, bounds: list, identifier: str) -> dict:
    """Enclose a configured local box under an interval rigid transform."""
    from itertools import product

    if len(bounds) != 3:
        raise ValueError("A spatial envelope requires three configured bounds")
    for pair in bounds:
        interval(*pair)
    corners = [
        [_sum([matrix[i][3], *(mul(matrix[i][j], interval(corner[j])) for j in range(3))])
         for i in range(3)]
        for corner in product(*bounds)
    ]
    return {"id": identifier,
            "minimum": [[min(c[i][0] for c in corners), min(c[i][1] for c in corners)] for i in range(3)],
            "maximum": [[max(c[i][0] for c in corners), max(c[i][1] for c in corners)] for i in range(3)]}

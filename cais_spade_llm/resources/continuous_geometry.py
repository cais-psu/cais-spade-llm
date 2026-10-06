"""Load configured URDF collision envelopes for pure continuous-motion models."""

from __future__ import annotations

import hashlib
import itertools
import xml.etree.ElementTree as ET
from pathlib import Path

from cais_spade_llm.resources.continuous_motion import add, interval, mul, transform


def _values(element, key, default):
    return [
        float(v) for v in (element.get(key, default) if element is not None else default).split()
    ]


def _joint_tree(tree, root, joint_names, observed_joints):
    joints, links = [], {root}
    pending = list(tree.findall("joint"))
    while True:
        added = False
        for row in list(pending):
            if row.find("parent").get("link") not in links:
                continue
            pending.remove(row)
            child, name, kind = row.find("child").get("link"), row.get("name"), row.get("type")
            axis = _values(row.find("axis"), "xyz", "1 0 0")
            frozen = {}
            if kind != "fixed" and name not in joint_names:
                mimic = row.find("mimic")
                if mimic is not None:
                    value = observed_joints[mimic.get("joint")]["position"] * float(
                        mimic.get("multiplier", "1")
                    ) + float(mimic.get("offset", "0"))
                else:
                    value = observed_joints[name]["position"]
                if kind not in {"revolute", "continuous", "prismatic"}:
                    raise ValueError("Unsupported non-commanded URDF joint: " + name)
                frozen = {"frozen_position": value}
            if row.find("mimic") is not None and name in joint_names:
                raise ValueError("Commanded mimic joints require another interpolation contract")
            joints.append(
                {
                    "name": name,
                    "parent": row.find("parent").get("link"),
                    "child": child,
                    "type": kind,
                    "axis": axis,
                    "xyz": _values(row.find("origin"), "xyz", "0 0 0"),
                    "rpy": _values(row.find("origin"), "rpy", "0 0 0"),
                    **frozen,
                }
            )
            links.add(child)
            added = True
        if not added:
            break
    return joints, links


def _shape_bounds(collision, sources):
    from ament_index_python.packages import get_package_share_directory

    from cais_spade_llm.recovery_framework.geometry import _mesh_vertices

    shape = collision.find("geometry")
    if shape.find("box") is not None:
        size = _values(shape.find("box"), "size", "")
        bounds = [[-v / 2, v / 2] for v in size]
    elif shape.find("cylinder") is not None:
        item = shape.find("cylinder")
        r, h = float(item.get("radius")), float(item.get("length"))
        bounds = [[-r, r], [-r, r], [-h / 2, h / 2]]
    elif shape.find("sphere") is not None:
        r = float(shape.find("sphere").get("radius"))
        bounds = [[-r, r]] * 3
    elif shape.find("mesh") is not None:
        mesh = shape.find("mesh")
        name = mesh.get("filename")
        if name.startswith("package://"):
            package, relative = name[len("package://") :].split("/", 1)
            path = Path(get_package_share_directory(package)) / relative
        elif name.startswith("file://"):
            path = Path(name[len("file://") :])
        else:
            path = Path(name)
        if not path.is_absolute() or path.suffix.lower() != ".stl":
            raise ValueError("Unsupported URDF collision mesh: " + name)
        stamp = path.stat()
        low, high = _mesh_vertices(path, stamp.st_mtime_ns, stamp.st_size)
        scale = _values(mesh, "scale", "1 1 1")
        if any(v <= 0 for v in scale):
            raise ValueError("Mesh scales must be positive")
        bounds = [
            list(mul(interval(a, b), interval(s))) for a, b, s in zip(low, high, scale, strict=True)
        ]
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        raise ValueError("Unsupported configured collision geometry")
    return bounds


def load_continuous_geometry(
    description: str,
    *,
    root: str,
    reference_link: str,
    joint_names: list,
    observed_joints: dict,
    root_pose: list,
    interpolation: str,
) -> dict:
    """Bind a URDF subtree and freeze every non-commanded joint explicitly.

    Collision meshes use their actual configured vertices. Unsupported geometry
    fails instead of substituting a guessed radius or tool-only envelope.
    """
    tree = ET.fromstring(description)
    joints, links = _joint_tree(tree, root, joint_names, observed_joints)
    sources = {}
    components = []
    for link in tree.findall("link"):
        if link.get("name") not in links:
            continue
        for index, collision in enumerate(link.findall("collision")):
            bounds = _shape_bounds(collision, sources)
            matrix = transform(
                _values(collision.find("origin"), "xyz", "0 0 0"),
                _values(collision.find("origin"), "rpy", "0 0 0"),
            )
            corners = []
            for corner in itertools.product(*bounds):
                point = []
                for i in range(3):
                    coordinate = matrix[i][3]
                    for j in range(3):
                        coordinate = add(coordinate, mul(matrix[i][j], interval(corner[j])))
                    point.append(coordinate)
                corners.append(point)
            # Serialized geometry is an enclosing AABB, not an exact mesh.
            import math

            bounds = [
                [
                    math.nextafter(min(v[i][0] for v in corners), -math.inf),
                    math.nextafter(max(v[i][1] for v in corners), math.inf),
                ]
                for i in range(3)
            ]
            components.append(
                {
                    "id": f"{link.get('name')}/{collision.get('name', str(index))}",
                    "link": link.get("name"),
                    "bounds": bounds,
                }
            )
    return {
        "root": root,
        "reference_link": reference_link,
        "joint_names": list(joint_names),
        "root_xyz": root_pose[:3],
        "root_quaternion": list(root_pose[3:]),
        "joints": joints,
        "components": components,
        "interpolation": interpolation,
        "description_sha256": hashlib.sha256(description.encode()).hexdigest(),
        "geometry_sources": sources,
        "fixed_joint_positions": {
            name: row["position"]
            for name, row in observed_joints.items()
            if name not in joint_names
        },
    }

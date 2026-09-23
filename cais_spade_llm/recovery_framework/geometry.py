"""Read the same fixed collision geometry used by recovery-framework Gazebo."""

from __future__ import annotations

import hashlib
import math
import struct
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=128)
def _xml_root(path: Path, modified: int, size: int) -> ET.Element:
    return ET.parse(path).getroot()


def _read_model(path: Path) -> ET.Element:
    stamp = path.stat()
    return _xml_root(path, stamp.st_mtime_ns, stamp.st_size)


@lru_cache(maxsize=128)
def _mesh_vertices(path: Path, modified: int, size: int) -> tuple:
    data = path.read_bytes()
    count = struct.unpack_from('<I', data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + 50*count:
        vertices = tuple(struct.unpack_from('<3f', data, 84+50*i+12+12*j)
                         for i in range(count) for j in range(3))
    else:
        vertices = tuple(tuple(float(v) for v in fields[1:])
                         for line in data.decode('ascii').splitlines()
                         if (fields := line.split()) and fields[0] == 'vertex' and len(fields) == 4)
    if not vertices:
        raise ValueError(f'Empty STL collision mesh: {path}')
    return tuple(min(v[i] for v in vertices) for i in range(3)), tuple(
        max(v[i] for v in vertices) for i in range(3))


@lru_cache(maxsize=16)
def _mesh_shape(path: Path, modified: int, size: int, scale: tuple) -> dict:
    """Keep CAD triangles for mating parts whose bore cannot use a solid bound."""
    data = path.read_bytes()
    count = struct.unpack_from('<I', data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + 50 * count:
        vertices = [struct.unpack_from('<3f', data, 84 + 50 * i + 12 + 12 * j)
                    for i in range(count) for j in range(3)]
    else:
        vertices = [tuple(float(v) for v in fields[1:])
                    for line in data.decode('ascii').splitlines()
                    if (fields := line.split()) and fields[0] == 'vertex']
    low, high = _mesh_vertices(path, modified, size)
    center = [(low[i] + high[i]) / 2 for i in range(3)]
    points, indices, lookup = [], [], {}
    for vertex in vertices:
        point = tuple((vertex[i] - center[i]) * scale[i] for i in range(3))
        if point not in lookup:
            lookup[point] = len(points)
            points.append(list(point))
        indices.append(lookup[point])
    return {'vertices': points, 'triangles': [indices[i:i + 3] for i in range(0, len(indices), 3)],
            'source': str(path), 'source_sha256': hashlib.sha256(data).hexdigest(),
            'scale': list(scale), 'center': center}


def quaternion(rpy: list[float]) -> list[float]:
    """Convert configured roll, pitch, yaw to an xyzw quaternion."""
    r, p, y = (v / 2 for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]


def multiply(a: list[float], b: list[float]) -> list[float]:
    """Compose xyzw rotations in parent-to-child order."""
    x, y, z, w = a
    u, v, t, s = b
    return [w*u+x*s+y*t-z*v, w*v-x*t+y*s+z*u, w*t+x*v-y*u+z*s, w*s-x*u-y*v-z*t]


def rotate(q: list[float], xyz: list[float]) -> list[float]:
    """Rotate a translation by a unit xyzw quaternion."""
    return multiply(multiply(q, [*xyz, 0.]), [-q[0], -q[1], -q[2], q[3]])[:3]


def compose(parent: list[float], child: list[float]) -> list[float]:
    """Compose xyz/xyzw poses."""
    offset = rotate(parent[3:], child[:3])
    return [*(parent[i]+offset[i] for i in range(3)), *multiply(parent[3:], child[3:])]


def _pose(element: ET.Element) -> list[float]:
    values = [float(v) for v in element.findtext('pose', '0 0 0 0 0 0').split()]
    return [*values[:3], *quaternion(values[3:])]


def collision_boxes(world_path: Path, models_path: Path, part_poses: dict | None = None, *,
                    exact_models: set[str] | None = None) -> list[dict]:
    """Extract static SDF boxes and conservative cylinder bounds in world coordinates.

    Open robot-loading windows stay open because their surrounding panels are
    imported individually. Floor support is represented below world z=0.
    """
    part_poses = part_poses or {}
    world = _read_model(world_path).find('world')
    boxes = [{'id': 'ground_plane', 'pose': [0., 0., -.055, 0., 0., 0., 1.], 'size': [40., 40., .1]}]
    models = [(model.get('name'), model, _pose(model)) for model in world.findall('model')]
    for include in world.findall('include'):
        uri = include.findtext('uri', '')
        path = models_path / uri.removeprefix('model://') / 'model.sdf'
        if not path.is_file():
            continue
        model = _read_model(path).find('model')
        models.append((include.findtext('name', model.get('name')), model, _pose(include)))
    for name, model, world_pose in models:
        if (model.findtext('static') != 'true' and name not in part_poses) or name == 'KMR':
            continue
        world_pose = part_poses.get(name, world_pose)
        for link in model.findall('link'):
            link_pose = compose(world_pose, _pose(link))
            for collision in link.findall('collision'):
                geometry = collision.find('geometry')
                if geometry is None:
                    continue
                box = geometry.find('box')
                cylinder = geometry.find('cylinder')
                mesh = geometry.find('mesh')
                local_pose = _pose(collision)
                shape = {}
                if box is not None:
                    size = [float(v) for v in box.findtext('size').split()]
                elif cylinder is not None:
                    radius = float(cylinder.findtext('radius'))
                    size = [2*radius, 2*radius, float(cylinder.findtext('length'))]
                    if name in (exact_models or set()):
                        shape['cylinder'] = [size[2], radius]
                elif mesh is not None:
                    relative = mesh.findtext('uri').removeprefix('model://')
                    path = models_path / relative
                    if not path.is_file():
                        path = models_path.parent / relative
                    scale = [float(v) for v in mesh.findtext('scale', '1 1 1').split()]
                    stamp = path.stat()
                    minimum, maximum = _mesh_vertices(path, stamp.st_mtime_ns, stamp.st_size)
                    low = [minimum[i]*scale[i] for i in range(3)]
                    high = [maximum[i]*scale[i] for i in range(3)]
                    size = [high[i]-low[i] for i in range(3)]
                    if name in (exact_models or set()):
                        shape['mesh'] = _mesh_shape(path, stamp.st_mtime_ns, stamp.st_size, tuple(scale))
                    local_pose = compose(local_pose, [*((high[i]+low[i])/2 for i in range(3)), 0., 0., 0., 1.])
                else:
                    raise ValueError(f'Unmodeled fixed collision geometry: {name}/{link.get("name")}')
                boxes.append({'id': f'{name}/{link.get("name")}/{collision.get("name")}',
                              'pose': compose(link_pose, local_pose), 'size': size, **shape})
    return boxes

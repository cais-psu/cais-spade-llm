"""Configuration and model checks for the recovery framework Gazebo scene."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from cais_spade_llm.ui import ros2_processes

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = ROOT / 'ros2/cais_lab_robotics/launch/recovery_framework_gazebo.launch.py'
ROBOTS_PATH = ROOT / 'cais_spade_llm/initialization/recovery_framework_gazebo.json'
WORLD_PATH = ROOT / 'ros2/cais_lab_robotics/worlds/table_recovery_framework.world'
MACHINE_MODEL_PATH = ROOT / 'ros2/cais_lab_robotics/models/haas_mini_mill/model.sdf'
PRINTER_MODEL_PATH = ROOT / 'ros2/cais_lab_robotics/models/prusa_mk4_2/model.sdf'
KMR_MODEL_PATH = ROOT / 'ros2/cais_lab_robotics/models/KMR/model.sdf'
KMR_URDF_PATH = ROOT / 'ros2/cais_lab_robotics/urdf/KMR_recovery.urdf.xacro'
KMR_CONTROLLERS_PATH = ROOT / 'ros2/cais_lab_robotics/config/recovery_framework_kmr_controllers.yaml'
DOCK_KMR_ACTION_PATH = ROOT / 'ros2/cais_lab_robotics/action/DockKMR.action'
KMR_BASE_CONTROLLER_PATH = ROOT / 'ros2/cais_lab_robotics/scripts/kmr_base_controller.py'
RECOVERY_MARKERS_PATH = ROOT / 'ros2/cais_lab_robotics/scripts/recovery_drag_markers.py'
RECOVERY_RVIZ_PATH = ROOT / 'ros2/cais_lab_robotics/rviz/recovery_framework.rviz'
RECOVERY_NAV2_PATH = ROOT / 'ros2/cais_lab_robotics/config/recovery_framework_nav2.yaml'
RECOVERY_NAV_THROUGH_PATH = (
    ROOT / 'ros2/cais_lab_robotics/config/recovery_framework_navigate_through_poses.xml'
)
RECOVERY_NAV_TO_PATH = (
    ROOT / 'ros2/cais_lab_robotics/config/recovery_framework_navigate_to_pose.xml'
)
RECOVERY_MAP_YAML_PATH = ROOT / 'ros2/cais_lab_robotics/config/recovery_framework_map.yaml'
RECOVERY_MAP_PATH = ROOT / 'ros2/cais_lab_robotics/config/recovery_framework_map.pgm'
RECOVERY_MAP_GENERATOR_PATH = ROOT / 'ros2/cais_lab_robotics/scripts/generate_recovery_framework_map.py'
KMR_NOTICE_PATH = ROOT / 'ros2/cais_lab_robotics/models/KMR/THIRD_PARTY_NOTICES.md'
KMR_LICENSES_PATH = ROOT / 'ros2/cais_lab_robotics/models/KMR/LICENSES'
spec = importlib.util.spec_from_file_location('recovery_framework_gazebo', LAUNCH_PATH)
assert spec is not None and spec.loader is not None
scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene)
base_spec = importlib.util.spec_from_file_location('kmr_base_controller', KMR_BASE_CONTROLLER_PATH)
assert base_spec is not None and base_spec.loader is not None
kmr_base = importlib.util.module_from_spec(base_spec)
base_spec.loader.exec_module(kmr_base)
map_spec = importlib.util.spec_from_file_location(
    'generate_recovery_framework_map', RECOVERY_MAP_GENERATOR_PATH,
)
assert map_spec is not None and map_spec.loader is not None
recovery_map = importlib.util.module_from_spec(map_spec)
map_spec.loader.exec_module(recovery_map)
marker_spec = importlib.util.spec_from_file_location(
    'recovery_drag_markers', RECOVERY_MARKERS_PATH,
)
assert marker_spec is not None and marker_spec.loader is not None
recovery_markers = importlib.util.module_from_spec(marker_spec)
marker_spec.loader.exec_module(recovery_markers)


def test_robot_bindings_match_machine_and_preserved_assembly_positions() -> None:
    robots = scene._load_robots(ROBOTS_PATH)
    assert [(robot['resource_id'], robot['prefix']) for robot in robots] == [
        ('ur5e-1', 'ur5e_1_'), ('ur5e-2', 'ur5e_2_'),
        ('ur5e-3', 'ur5e_3_'), ('ur5e-4', 'ur5e_4_'),
    ]
    assert [robot['base_xyz'] for robot in robots] == [
        [-6.0, 1.1, 0.8], [-2.6, 1.1, 0.8],
        [0.0, 0.5, 1.021], [0.0, -0.5, 1.021],
    ]


def test_m1_and_m2_use_the_same_machine_with_separate_assignments() -> None:
    payload = json.loads(ROBOTS_PATH.read_text())
    machines = payload['machines']
    assert [machine['resource_id'] for machine in machines] == ['M1', 'M2']
    assert {machine['gazebo_model'] for machine in machines} == {'haas_mini_mill'}
    assert machines[0]['nominal_parts'] == [
        'KET4_Square_4mm', 'KET8_Square_8mm',
        'KET12_Square_12mm', 'KET16_Square_16mm',
    ]
    assert machines[1]['nominal_parts'] == [
        'RGOCG4-50_Round_4mm', 'RGOCG8-50_8mm',
        'RGOCG12-50_12mm', 'RGOCG16-50_16mm',
    ]


def test_machining_station_matches_the_side_by_side_material_flow() -> None:
    machines = json.loads(ROBOTS_PATH.read_text())['machines']
    assert [machine['world_pose'] for machine in machines] == [
        [-6.0, 2.3, 0.0, 0.0, 0.0, -1.57079632679],
        [-2.6, 2.3, 0.0, 0.0, 0.0, -1.57079632679],
    ]
    assert [machine['KMR_docking_pose'] for machine in machines] == [
        [-7.25, 2.3, 0.0, 0.0, 0.0, -1.57079632679],
        [-3.85, 2.3, 0.0, 0.0, 0.0, -1.57079632679],
    ]
    assert {machine['front_access_link'] for machine in machines} == {'front_access'}
    assert {machine['side_access_link'] for machine in machines} == {'side_access'}


def test_machine_world_poses_and_kmr_docking_poses_match_configuration() -> None:
    payload = json.loads(ROBOTS_PATH.read_text())
    world = ET.parse(WORLD_PATH)
    includes = {item.findtext('name'): item for item in world.findall('.//include')}
    models = {item.attrib['name']: item for item in world.findall('.//model')}
    for machine in payload['machines']:
        resource_id = machine['resource_id']
        include = includes[resource_id]
        assert include.findtext('uri') == f"model://{machine['gazebo_model']}"
        assert [float(value) for value in include.findtext('pose').split()] == pytest.approx(machine['world_pose'])
        docking = models[f'{resource_id}_KMR_docking_pose']
        actual = [float(value) for value in docking.findtext('pose').split()]
        expected = machine['KMR_docking_pose']
        assert actual[:2] + actual[3:] == expected[:2] + expected[3:]
        assert actual[2] == pytest.approx(0.004)


def test_machine_enclosures_have_one_metre_horizontal_separation_from_assembly() -> None:
    payload = json.loads(ROBOTS_PATH.read_text())
    enclosure = ET.parse(MACHINE_MODEL_PATH).find("./model/link[@name='enclosure']")
    east_edges = []
    for machine in payload['machines']:
        x, _, _, _, _, yaw = machine['world_pose']
        for collision in enclosure.findall('collision'):
            cx, cy, *_ = map(float, collision.findtext('pose').split())
            sx, sy, _ = map(float, collision.findtext('geometry/box/size').split())
            east_edges.append(x + math.cos(yaw) * cx - math.sin(yaw) * cy
                              + abs(math.cos(yaw)) * sx / 2 + abs(math.sin(yaw)) * sy / 2)
    tables = [item for item in ET.parse(WORLD_PATH).findall('./world/include')
              if item.findtext('uri') == 'model://table']
    # The existing table model has a 1.5 m wide top; its original pose is preserved.
    assembly_west_edge = min(float(item.findtext('pose').split()[0]) - 0.75 for item in tables)
    gap = assembly_west_edge - max(east_edges)
    assert payload['minimum_station_gap'] == 1.0
    assert gap == pytest.approx(payload['minimum_station_gap'])


def test_compact_layout_stays_inside_the_intended_horizontal_range() -> None:
    payload = json.loads(ROBOTS_PATH.read_text())
    storage_min_x = payload['Storage']['world_pose'][0] - payload['Storage']['size'][1] / 2
    assembly_max_x = max(
        float(item.findtext('pose').split()[0]) + 0.75
        for item in ET.parse(WORLD_PATH).findall('./world/include')
        if item.findtext('uri') == 'model://table'
    )
    assert storage_min_x == pytest.approx(-9.65)
    assert assembly_max_x == pytest.approx(0.75)
    assert storage_min_x >= -9.8
    assert assembly_max_x <= 0.8


def test_passive_layout_supports_match_manifest_and_both_loading_positions() -> None:
    payload = json.loads(ROBOTS_PATH.read_text())
    world = ET.parse(WORLD_PATH)
    models = {item.attrib['name']: item for item in world.findall('./world/model')}
    for name in ('Storage', 'Conveyor', 'Buffer For Machined parts', 'Exit'):
        model = models[name]
        assert model.findtext('static') == 'true'
        assert model.find('plugin') is None
        assert model.find('.//collision') is not None
        assert list(map(float, model.findtext('pose').split())) == pytest.approx(payload[name]['world_pose'])
    conveyor = payload['Conveyor']
    belt = models['Conveyor'].find("./link/collision[@name='belt']")
    sx, sy, sz = map(float, belt.findtext('geometry/box/size').split())
    assert [sx, sy] == [conveyor['length'], conveyor['width']]
    assert float(belt.findtext('pose').split()[2]) + sz / 2 == pytest.approx(conveyor['surface_height'])
    assert conveyor['transport_enabled'] is False
    output_nest = models['Conveyor'].find("./link/visual[@name='output_nest']")
    relative = list(map(float, output_nest.findtext('pose').split()))
    output_world = [
        conveyor['world_pose'][index] + relative[index] for index in range(3)
    ]
    assert output_world == pytest.approx(conveyor['output_nest_pose'][:3])
    assert conveyor['output_nest_capacity'] == 1
    robots = {robot['resource_id']: robot for robot in payload['robots']}
    for machine in payload['machines']:
        robot = robots[machine['handling_robot']]
        pedestal = models[robot['resource_id'] + ' pedestal']
        assert list(map(float, pedestal.findtext('pose').split()))[:2] == robot['base_xyz'][:2]
        plate = pedestal.find("./link/collision[@name='mounting_plate']")
        assert float(plate.findtext('pose').split()[2]) + float(plate.findtext('geometry/box/size').split()[2]) / 2 == pytest.approx(robot['base_xyz'][2])
        loading = machine['conveyor_loading_pose']
        assert abs(loading[0] - conveyor['world_pose'][0]) < sx / 2
        assert loading[1] == conveyor['world_pose'][1]
        assert loading[2] == conveyor['surface_height']
        tray = models[machine['resource_id'] + ' staging tray']
        assert list(map(float, tray.findtext('pose').split()))[:2] == machine['staging_pose'][:2]
        assert machine['staging_capacity'] == 1
        collisions = {item.attrib['name'] for item in tray.findall('link/collision')}
        assert collisions == {'pedestal', 'surface', 'rear_edge', 'left_edge', 'right_edge'}
        assert tray.find("link/visual[@name='capacity_one_nest']") is not None
    assert 'Conveyor output pusher' not in models
    assert 'Conveyor output pusher' not in payload
    assert models['Conveyor'].find("link/collision[@name='end_stop']") is None

    buffer = payload['Buffer For Machined parts']
    assert len(buffer['slot_poses']) == buffer['capacity'] == 4
    assert buffer['initial_state'] == 'empty'
    assert buffer['transport_enabled'] is False
    bx, by, bz = buffer['world_pose'][:3]
    model = models['Buffer For Machined parts']
    assert model.find('joint') is None
    length, width = buffer['length'], buffer['width']
    assert [length, width] == [0.48, 0.24]
    assert model.find("link/collision[@name='belt']") is None
    assert buffer['pickup_pose'] == buffer['slot_poses'][-1]
    assert [pose[0] for pose in buffer['slot_poses']] == [-0.68, -0.56, -0.44, -0.32]
    for index, (x, y, z, *_) in enumerate(buffer['slot_poses'], 1):
        assert bx - length / 2 < x < bx + length / 2
        assert y == by and z == buffer['surface_height']
        assert -0.75 <= x <= 0.75 and 0.0 <= y <= 0.8
        marker = model.find(f"link/visual[@name='waiting_position_{index}']/pose")
        relative = list(map(float, marker.text.split()))
        assert [bx + relative[0], by + relative[1]] == pytest.approx([x, y])
    collision_names = {item.attrib['name'] for item in model.findall('link/collision')}
    assert {'guide_left', 'guide_right', 'downstream_stop', 'transfer_plate',
            'infeed_guide_left', 'infeed_guide_right'} <= collision_names
    assert {f'zone_{index}_belt' for index in range(1, 5)} <= collision_names
    assert {f'zone_{index}_stop' for index in range(1, 5)} <= collision_names


def test_transfer_plate_continuously_joins_an_unobstructed_buffer_inlet() -> None:
    payload = json.loads(ROBOTS_PATH.read_text())
    world = ET.parse(WORLD_PATH)
    model = world.find("./world/model[@name='Buffer For Machined parts']")
    buffer = payload['Buffer For Machined parts']
    conveyor = payload['Conveyor']
    bx, by, bz = buffer['world_pose'][:3]
    plate = model.find("link/collision[@name='transfer_plate']")
    px, py, pz, roll, pitch, yaw = map(float, plate.findtext('pose').split())
    length, width, thickness = map(float, plate.findtext('geometry/box/size').split())
    assert roll == yaw == 0.0
    assert width == buffer['width']
    endpoints = []
    for x in (-length / 2, length / 2):
        endpoints.append([
            bx + px + math.cos(pitch) * x + math.sin(pitch) * thickness / 2,
            by + py,
            bz + pz - math.sin(pitch) * x + math.cos(pitch) * thickness / 2,
        ])
    assert endpoints[0] == pytest.approx([
        conveyor['world_pose'][0] + conveyor['length'] / 2,
        conveyor['world_pose'][1], conveyor['surface_height'],
    ])
    assert endpoints[1] == pytest.approx([
        bx - buffer['length'] / 2, by, buffer['surface_height'],
    ])
    # Tapered guides begin on the Conveyor and converge on the shallow channel.
    for name, expected_yaw in (
        ('infeed_guide_left', -0.375912), ('infeed_guide_right', 0.375912),
    ):
        collision = model.find(f"link/collision[@name='{name}']")
        x, y, z, roll, pitch, yaw = map(float, collision.findtext('pose').split())
        sx, sy, sz = map(float, collision.findtext('geometry/box/size').split())
        assert x - sx / 2 < -buffer['length'] / 2
        assert abs(y) > buffer['guide_clear_width_m'] / 2
        assert z - sz / 2 >= 0.002 - 1e-9
        assert [roll, pitch, yaw] == pytest.approx([0, 0, expected_yaw])
    stop = model.find("link/collision[@name='downstream_stop']")
    stop_x = bx + float(stop.findtext('pose').split()[0])
    stop_half = float(stop.findtext('geometry/box/size').split()[0]) / 2
    assert stop_x - stop_half > buffer['pickup_pose'][0] + buffer['part_length_m'] / 2


def test_storage_contains_all_eight_nist_pegs_on_configured_shelves() -> None:
    payload = json.loads(ROBOTS_PATH.read_text())
    world = ET.parse(WORLD_PATH)
    models = {item.attrib['name']: item for item in world.findall('./world/model')}
    slots = payload['Storage']['slots']
    assert set(slots) == {
        'KET4_Square_4mm', 'KET8_Square_8mm',
        'KET12_Square_12mm', 'KET16_Square_16mm',
        'RGOCG4-50_Round_4mm', 'RGOCG8-50_8mm',
        'RGOCG12-50_12mm', 'RGOCG16-50_16mm',
    }
    storage = models['Storage']
    trays = payload['Storage']['kitting_trays']
    assert trays == [
        {'parts': 'KET', 'surface_z': 1.165, 'capacity': 4},
        {'parts': 'RGOCG', 'surface_z': 0.645, 'capacity': 4},
    ]
    for tray in trays:
        collision = storage.find(f"link/collision[@name='{tray['parts']}_kitting_tray']")
        assert collision is not None
        relative_z = float(collision.findtext('pose').split()[2])
        thickness = float(collision.findtext('geometry/box/size').split()[2])
        assert relative_z + thickness / 2 == pytest.approx(tray['surface_z'])
        slot_visuals = [
            item for item in storage.findall('link/visual')
            if item.attrib['name'].startswith(f"{tray['parts']}_slot_")
        ]
        dividers = [
            item for item in storage.findall('link/collision')
            if item.attrib['name'].startswith(f"{tray['parts']}_divider_")
        ]
        assert len(slot_visuals) == tray['capacity'] == 4
        assert len(dividers) == 5

    for name, expected_pose in slots.items():
        model = models[name]
        actual_pose = list(map(float, model.findtext('pose').split()))
        assert actual_pose == pytest.approx(expected_pose)
        assert actual_pose[0] == pytest.approx(-8.95)
        assert actual_pose[5] == pytest.approx(math.pi / 2)
        mesh = model.findtext('link/visual/geometry/mesh/uri')
        assert mesh == f'model://cad_models/{name}.STL'
        if name.startswith('KET'):
            assert actual_pose[2] == pytest.approx(1.165)
        else:
            assert actual_pose[2] == pytest.approx(0.645)
        storage_x, storage_y, *_, storage_yaw = payload['Storage']['world_pose']
        dx, dy = actual_pose[0] - storage_x, actual_pose[1] - storage_y
        local_x = math.cos(storage_yaw) * dx + math.sin(storage_yaw) * dy
        assert -0.55 < local_x < 0.55


def test_storage_kitting_trays_leave_each_part_pickable() -> None:
    payload = json.loads(ROBOTS_PATH.read_text())
    models = {item.attrib['name']: item for item in ET.parse(WORLD_PATH).findall('./world/model')}
    storage_x, storage_y, *_, storage_yaw = payload['Storage']['world_pose']
    boundaries = [-0.55, -0.28, 0.0, 0.28, 0.55]
    for name, pose in payload['Storage']['slots'].items():
        dx, dy = pose[0] - storage_x, pose[1] - storage_y
        local_x = math.cos(storage_yaw) * dx + math.sin(storage_yaw) * dy
        local_y = -math.sin(storage_yaw) * dx + math.cos(storage_yaw) * dy
        left, right = next(
            (left, right) for left, right in zip(boundaries, boundaries[1:])
            if left < local_x < right
        )
        collision = models[name].find('link/collision/geometry')
        size = collision.findtext('box/size') or collision.findtext('cylinder/radius')
        half_width = float(size.split()[0]) / 2 if ' ' in size else float(size)
        assert local_x - half_width > left + 0.0075
        assert local_x + half_width < right - 0.0075
        assert -0.3325 < local_y - half_width
        assert local_y + half_width < -0.0675


def test_single_printing_station_starts_with_three_gears_and_empty_exit() -> None:
    payload = json.loads(ROBOTS_PATH.read_text())
    world = ET.parse(WORLD_PATH)
    models = {item.attrib['name']: item for item in world.findall('./world/model')}
    includes = {item.findtext('name'): item for item in world.findall('./world/include')}
    assert {'prusa_mk3', 'prusa_mk4_1', 'cam_mk3', 'cam_mk4_1'}.isdisjoint(models)
    assert includes['prusa_mk4_2'].findtext('uri') == 'model://prusa_mk4_2'
    printer = payload['3D Printing Station']
    assert printer['world_pose'] == [0.50, -0.50, 1.04, 0.0, 0.0, -1.57079632679]
    assert printer['initial_products'] == printer['supported_products'] == [
        'gear_small', 'gear_medium', 'gear_large',
    ]
    assert printer['initial_state'] == 'completed'
    assert printer['handling_robot'] == 'ur5e-4'
    assert 'initial_product' not in printer and 'output_pose' not in printer
    assert list(map(float, includes['prusa_mk4_2'].findtext('pose').split())) == printer['world_pose']
    for name, pose in printer['output_poses'].items():
        assert includes[name].findtext('uri') == f'model://{name}'
        assert list(map(float, includes[name].findtext('pose').split())) == pose
        assert len([item for item in world.findall('./world/include') if item.findtext('name') == name]) == 1
    printer_model = ET.parse(PRINTER_MODEL_PATH).getroot().find('model')
    assert printer_model is not None and printer_model.findtext('static') == 'true'
    components = {item.attrib['name'] for item in printer_model.findall('link/visual')}
    assert {'base', 'bed', 'bed_print_area', 'left_frame', 'right_frame',
            'top_frame', 'gantry', 'extruder', 'nozzle', 'spool', 'display'} <= components
    assert {'cam_storage', 'cam_mk4_2', 'cam_assembly'} <= set(models)
    assert list(map(float, models['cam_storage'].findtext('pose').split())) == pytest.approx(
        [-8.20, 0.75, 2.35, 0.0, 0.694, 2.047]
    )
    assert list(map(float, models['cam_mk4_2'].findtext('pose').split())) == pytest.approx(
        [0.50, -0.50, 1.75, 0.0, 1.5708, 0.0]
    )
    exit_contract = payload['Exit']
    assert exit_contract['capacity'] == 1 and exit_contract['initial_state'] == 'empty'
    assert exit_contract['handling_robot'] == 'ur5e-3'
    assert exit_contract['completed_product'] == 'assembly_board-v1'
    assert list(map(float, models['Exit'].findtext('pose').split())) == exit_contract['world_pose']
    assert exit_contract['completed_product_pose'] == [0.50, 0.58, 1.075, 0.0, 0.0, 0.0]
    robots = {robot['resource_id']: robot for robot in payload['robots']}
    assert robots['ur5e-3']['base_rpy'] == [0.0, 0.0, 3.142]
    assert robots['ur5e-4']['base_rpy'] == [0.0, 0.0, 0.0]
    assert math.dist(printer['world_pose'][:2], robots['ur5e-4']['base_xyz'][:2]) == pytest.approx(0.50)
    assert math.dist(exit_contract['world_pose'][:2], robots['ur5e-3']['base_xyz'][:2]) == pytest.approx(0.51, abs=0.01)
    assert exit_contract['world_pose'][1] - 0.20 > printer['world_pose'][1] + 0.20
    assert printer['world_pose'][0] + 0.20 < 0.75
    assert printer['world_pose'][1] - 0.20 > -0.80


def test_three_gears_rest_on_print_area_without_overlapping_printer_or_each_other() -> None:
    printer = json.loads(ROBOTS_PATH.read_text())['3D Printing Station']
    model = ET.parse(PRINTER_MODEL_PATH).find('./model/link')
    bed = model.find("collision[@name='bed']")
    bed_top = float(bed.findtext('pose').split()[2]) + float(bed.findtext('geometry/box/size').split()[2]) / 2
    area = model.find("visual[@name='bed_print_area']")
    ax, ay, *_ = map(float, area.findtext('pose').split())
    width, depth, _ = map(float, area.findtext('geometry/box/size').split())
    px, py, pz, _, _, yaw = printer['world_pose']
    placed = []
    for name, (x, y, z, *_) in printer['output_poses'].items():
        gear_path = PRINTER_MODEL_PATH.parent.parent / name / 'model.sdf'
        gear = ET.parse(gear_path).find('./model/link')
        cylinder = gear.find('collision/geometry/cylinder')
        if cylinder is not None:
            radius = float(cylinder.findtext('radius'))
            lower = -float(cylinder.findtext('length')) / 2
        else:
            uri = gear.findtext('collision/geometry/mesh/uri')
            mesh = PRINTER_MODEL_PATH.parent.parent / uri.removeprefix('model://')
            vertices = [tuple(map(float, line.split()[1:])) for line in mesh.read_text().splitlines()
                        if line.strip().startswith('vertex ')]
            radius = max(math.hypot(v[0], v[1]) for v in vertices)
            lower = min(v[2] for v in vertices)
        assert z + lower == pytest.approx(pz + bed_top, abs=1e-8)
        lx = math.cos(yaw) * (x - px) + math.sin(yaw) * (y - py)
        ly = -math.sin(yaw) * (x - px) + math.cos(yaw) * (y - py)
        assert abs(lx - ax) + radius < width / 2
        assert abs(ly - ay) + radius < depth / 2
        for obstacle in model.findall('collision'):
            if obstacle.attrib['name'] == 'bed':
                continue
            ox, oy, oz, *_ = map(float, obstacle.findtext('pose').split())
            sx, sy, sz = map(float, obstacle.findtext('geometry/box/size').split())
            if pz + oz - sz / 2 >= z + 0.01 or pz + oz + sz / 2 <= z + lower:
                continue
            dx, dy = max(abs(lx - ox) - sx / 2, 0), max(abs(ly - oy) - sy / 2, 0)
            assert math.hypot(dx, dy) > radius
        for other_x, other_y, other_radius in placed:
            assert math.hypot(x - other_x, y - other_y) > radius + other_radius + 0.02
        placed.append((x, y, radius))


def test_kmr_source_layout_stays_at_storage_and_runtime_control_is_configured() -> None:
    payload = json.loads(ROBOTS_PATH.read_text())
    kmr = payload['KMR']
    expected = {
        'spawned': True,
        'gazebo_model': 'KMR',
        'initial_pose': [-8.15, 2.3, 0.0, 0.0, 0.0, -1.57079632679],
        'integrated': False,
        'simulation_control_integrated': True,
        'manufacturer': 'KUKA',
        'platform_model': 'KMR iiwa',
        'mobile_platform_model': 'KMP omniMove 400',
        'platform_body_dimensions_m': [1.08, 0.63, 0.70],
        'platform_overall_dimensions_m': [1.19, 0.72, 0.70],
        'wheel_diameter_m': 0.25,
        'arm_model': 'LBR iiwa 14 R820',
        'arm_mount_xyz': [-0.25, 0.0, 0.70],
        'arm_mount_rpy': [0.0, 0.0, 1.57079632679],
        'end_effector': 'OnRobot RG2',
        'arm_reach_m': 0.82,
        'arm_payload_kg': 14,
        'gripper_stroke_m': 0.11,
        'mount_verified': False,
        'parked_arm_configuration': [0, 0, 0, 0, 0, 0, 0],
        'predefined_routes': [['Storage', 'M1'], ['Storage', 'M2']],
    }
    assert {key: kmr[key] for key in expected} == expected
    assert kmr['arm_joint_names'] == list(scene.KMR_ARM_JOINTS)
    assert kmr['arm_controller'] == '/KMR/KMR_iiwa_joint_trajectory_controller'
    assert kmr['gripper_joint'] == 'KMR_rg2_finger_width'
    assert kmr['gripper_controller'] == '/KMR/KMR_rg2_gripper_traj_controller'
    assert kmr['base_command_topic'] == '/KMR/cmd_vel'
    assert kmr['base_odometry_topic'] == '/KMR/odom'
    assert kmr['docking_action'] == '/KMR/dock'
    storage_east_edge = payload['Storage']['world_pose'][0] + payload['Storage']['size'][1] / 2
    storage_dock_west_edge = (
        payload['Storage']['KMR_docking_pose'][0]
        - kmr['platform_overall_dimensions_m'][1] / 2
    )
    assert storage_dock_west_edge - storage_east_edge == pytest.approx(0.14)
    assert kmr['initial_pose'] == payload['Storage']['KMR_docking_pose']
    world = ET.parse(WORLD_PATH)
    marker = world.find("./world/model[@name='Storage_KMR_docking_pose']/pose")
    actual = list(map(float, marker.text.split()))
    expected = payload['Storage']['KMR_docking_pose']
    assert actual[:2] + actual[3:] == expected[:2] + expected[3:]
    assert actual[2] == pytest.approx(0.004)
    for name in ('M1_KMR_docking_pose', 'M2_KMR_docking_pose', 'Storage_KMR_docking_pose'):
        size = world.find(
            f"./world/model[@name='{name}']/link/visual/geometry/box/size"
        )
        assert list(map(float, size.text.split()))[:2] == kmr['platform_overall_dimensions_m'][:2]
    m1_marker_pose = list(map(float, world.findtext(
        "./world/model[@name='M1_KMR_docking_pose']/pose"
    ).split()))
    assert m1_marker_pose[-1] == pytest.approx(-math.pi / 2)
    assert payload['machines'][0]['KMR_docking_pose'] == pytest.approx(
        [-7.25, 2.3, 0.0, 0.0, 0.0, -math.pi / 2]
    )
    assert payload['machines'][1]['KMR_docking_pose'] == pytest.approx(
        [-3.85, 2.3, 0.0, 0.0, 0.0, -math.pi / 2]
    )
    assert payload['Storage']['KMR_docking_pose'][-1] == pytest.approx(-math.pi / 2)
    kmr_include = next(item for item in world.findall('./world/include')
                       if item.findtext('name') == 'KMR')
    assert kmr_include.findtext('uri') == 'model://KMR'
    assert list(map(float, kmr_include.findtext('pose').split())) == kmr['initial_pose']

    model = ET.parse(KMR_MODEL_PATH).getroot().find('model')
    assert model is not None and model.findtext('static') == 'true'
    assert model.find('.//plugin') is None
    assert model.find('joint') is None
    assert model.find('.//ros2_control') is None
    platform = model.find("link[@name='mobile_platform_link']/collision[@name='mobile_platform']")
    assert list(map(float, platform.findtext('geometry/box/size').split())) == kmr['platform_body_dimensions_m']
    platform_link = model.find("link[@name='mobile_platform_link']")
    collisions = {item.attrib['name']: item for item in platform_link.findall('collision')}
    visuals = {item.attrib['name']: item for item in platform_link.findall('visual')}
    assert {'mobile_platform', 'front_safety_scanner', 'rear_safety_scanner',
            'wheel_front_left', 'wheel_front_right', 'wheel_rear_left',
            'wheel_rear_right'} <= set(collisions)
    scanner_x_extents = []
    for name in ('front_safety_scanner', 'rear_safety_scanner'):
        scanner = collisions[name]
        x = float(scanner.findtext('pose').split()[0])
        length = float(scanner.findtext('geometry/cylinder/length'))
        scanner_x_extents.extend((x - length / 2, x + length / 2))
    assert max(scanner_x_extents) - min(scanner_x_extents) == pytest.approx(
        kmr['platform_overall_dimensions_m'][0]
    )
    wheel_y_extents = []
    for name in ('wheel_front_left', 'wheel_front_right',
                 'wheel_rear_left', 'wheel_rear_right'):
        wheel = collisions[name]
        y = float(wheel.findtext('pose').split()[1])
        length = float(wheel.findtext('geometry/cylinder/length'))
        radius = float(wheel.findtext('geometry/cylinder/radius'))
        assert 2 * radius == pytest.approx(kmr['wheel_diameter_m'])
        wheel_y_extents.extend((y - length / 2, y + length / 2))
    assert max(wheel_y_extents) - min(wheel_y_extents) == pytest.approx(
        kmr['platform_overall_dimensions_m'][1]
    )
    assert len([name for name in visuals if '_roller_' in name]) == 16
    assert {'front_safety_scanner_window', 'rear_safety_scanner_window',
            'left_RGB_LED_band', 'right_RGB_LED_band', 'emergency_stop_left',
            'emergency_stop_right', 'led_sound_buzzer', 'clear_deck',
            'iiwa_mounting_plate'} <= set(visuals)
    assert {f'ultrasonic_{index}' for index in range(1, 9)} <= set(visuals)
    clear_deck = visuals['clear_deck']
    deck_x = float(clear_deck.findtext('pose').split()[0])
    deck_length = float(clear_deck.findtext('geometry/box/size').split()[0])
    mount = visuals['iiwa_mounting_plate']
    mount_x = float(mount.findtext('pose').split()[0])
    mount_radius = float(mount.findtext('geometry/cylinder/radius'))
    assert mount_x + mount_radius < deck_x - deck_length / 2
    links = {item.attrib['name']: item for item in model.findall('link')}
    assert {f'iiwa_link_{index}' for index in range(8)} <= set(links)
    assert {'rg2_adapter', 'rg2_base_link', 'rg2_left_outer_knuckle',
            'rg2_right_outer_knuckle', 'rg2_left_inner_knuckle',
            'rg2_right_inner_knuckle', 'rg2_left_inner_finger',
            'rg2_right_inner_finger'} <= set(links)
    for index in range(8):
        link = links[f'iiwa_link_{index}']
        assert 'model://KMR/meshes/lbr_iiwa_14_r820/visual/' in link.findtext('visual/geometry/mesh/uri')
        assert 'model://KMR/meshes/lbr_iiwa_14_r820/collision/' in link.findtext('collision/geometry/mesh/uri')
    expected_y = [0.0, 0.0, -0.000436240, -0.000436240,
                  0.0, 0.0, 0.0, 0.0]
    for index, (expected_link_y, expected_z) in enumerate(zip(
            expected_y, [0.70, 0.70, 1.06, 1.06, 1.48, 1.48, 1.88, 1.88])):
        x, y, z, roll, pitch, yaw = map(float, links[f'iiwa_link_{index}'].findtext('pose').split())
        assert x == pytest.approx(-0.25) and y == pytest.approx(expected_link_y)
        assert z == pytest.approx(expected_z)
        assert [roll, pitch, yaw] == pytest.approx([0.0, 0.0, math.pi / 2])
    assert list(map(float, links['rg2_adapter'].findtext('pose').split())) == pytest.approx(
        [-0.25, 0, 2.044, 0, 0, math.pi / 2]
    )
    assert list(map(float, links['rg2_base_link'].findtext('pose').split())) == pytest.approx(
        [-0.25, 0, 2.054, 0, 0, math.pi / 2]
    )
    assert float(links['rg2_adapter'].findtext('collision/geometry/cylinder/length')) == pytest.approx(0.020)
    for name in ('rg2_base_link', 'rg2_left_outer_knuckle', 'rg2_right_outer_knuckle',
                 'rg2_left_inner_knuckle', 'rg2_right_inner_knuckle',
                 'rg2_left_inner_finger', 'rg2_right_inner_finger'):
        pose = list(map(float, links[name].findtext('pose').split()))
        assert pose[1] == pytest.approx(0.0)
        expected_yaw = -math.pi / 2 if 'right_' in name else math.pi / 2
        assert pose[-1] == pytest.approx(expected_yaw)
        assert 'model://KMR/meshes/rg2/visual/' in links[name].findtext('visual/geometry/mesh/uri')
        assert 'model://KMR/meshes/rg2/collision/' in links[name].findtext('collision/geometry/mesh/uri')
    left_finger_x = float(links['rg2_left_inner_finger'].findtext('pose').split()[0])
    right_finger_x = float(links['rg2_right_inner_finger'].findtext('pose').split()[0])
    assert right_finger_x + left_finger_x == pytest.approx(-0.5)
    assert left_finger_x - right_finger_x > kmr['gripper_stroke_m']

    base_x, base_y, _, _, _, base_yaw = kmr['initial_pose']
    mount_x, mount_y, mount_z = kmr['arm_mount_xyz']
    world_mount_x = base_x + math.cos(base_yaw) * mount_x - math.sin(base_yaw) * mount_y
    world_mount_y = base_y + math.sin(base_yaw) * mount_x + math.cos(base_yaw) * mount_y
    assert [world_mount_x, world_mount_y, mount_z] == pytest.approx([-8.15, 2.55, 0.70])
    assert base_yaw + kmr['arm_mount_rpy'][2] == pytest.approx(0.0)
    assert (KMR_LICENSES_PATH / 'iiwa_ros2-Apache-2.0.txt').is_file()
    assert (KMR_LICENSES_PATH / 'OnRobot_ROS2_Description-MIT.txt').is_file()
    notice = KMR_NOTICE_PATH.read_text()
    assert '9d048b901f8fc4acaa9a0ad3f52067bd4476093a' in notice
    assert '29180b3fa9cba6555f3e515e789b8ccd34252fab' in notice
    for uri in {item.text for item in model.findall('.//mesh/uri')}:
        assert uri is not None and uri.startswith('model://KMR/')
        assert (KMR_MODEL_PATH.parent / uri.removeprefix('model://KMR/')).is_file()
    assert len(list((KMR_MODEL_PATH.parent / 'meshes/lbr_iiwa_14_r820/visual').iterdir())) == 8
    assert len(list((KMR_MODEL_PATH.parent / 'meshes/lbr_iiwa_14_r820/collision').iterdir())) == 8
    assert len(list((KMR_MODEL_PATH.parent / 'meshes/rg2/visual').iterdir())) == 4
    assert len(list((KMR_MODEL_PATH.parent / 'meshes/rg2/collision').iterdir())) == 4
    west_edge = kmr['initial_pose'][0] - kmr['platform_overall_dimensions_m'][1] / 2
    east_edge = kmr['initial_pose'][0] + kmr['platform_overall_dimensions_m'][1] / 2
    assert -9.8 <= west_edge < east_edge <= 0.8
    south_edge = kmr['initial_pose'][1] - kmr['platform_overall_dimensions_m'][0] / 2
    north_edge = kmr['initial_pose'][1] + kmr['platform_overall_dimensions_m'][0] / 2
    assert south_edge < north_edge
    storage_east_edge_at_m1 = (
        payload['Storage']['world_pose'][0] + payload['Storage']['size'][1] / 2
    )
    m1_west_edge = payload['machines'][0]['world_pose'][0] - 1.70 / 2
    assert west_edge > storage_east_edge_at_m1
    assert east_edge < m1_west_edge
    m1_dock_x = payload['machines'][0]['KMR_docking_pose'][0]
    assert m1_west_edge - (
        m1_dock_x + kmr['platform_overall_dimensions_m'][1] / 2
    ) == pytest.approx(0.04)
    m2_west_edge = payload['machines'][1]['world_pose'][0] - 1.70 / 2
    m2_dock_x = payload['machines'][1]['KMR_docking_pose'][0]
    assert m2_west_edge - (
        m2_dock_x + kmr['platform_overall_dimensions_m'][1] / 2
    ) == pytest.approx(0.04)


def test_ur5e_3_owns_conveyor_buffer_and_exit_while_ur5e_4_owns_printer() -> None:
    payload = json.loads(ROBOTS_PATH.read_text())
    assert payload['Conveyor']['output_handling_robot'] == 'ur5e-3'
    assert payload['Conveyor']['normal_output_transfer'] == 'Buffer For Machined parts'
    assert payload['Conveyor']['full_buffer_behavior'] == 'apply backpressure'
    assert payload['Buffer For Machined parts']['handling_robot'] == 'ur5e-3'
    assert payload['Exit']['handling_robot'] == 'ur5e-3'
    assert payload['3D Printing Station']['handling_robot'] == 'ur5e-4'


def test_nominal_handoff_moves_pegs_directly_without_removable_carriers() -> None:
    payload = json.loads(ROBOTS_PATH.read_text())
    buffer = payload['Buffer For Machined parts']
    assert buffer['normal_input_resource'] == 'Conveyor'
    assert {'diameter', 'index_direction', 'index_angle_rad', 'intake_pocket_pose',
            'presentation_pocket_pose'}.isdisjoint(buffer)
    assert buffer['nominal_grasps_per_machined_part'] == 1
    sequence = buffer['nominal_sequence']
    assert len(sequence) == 6
    assert sum('ur5e-3 picks' in step for step in sequence) == 1
    assert 'horizontally' in sequence[0]
    assert 'directly' in sequence[1]
    assert 'Zone 4 stops' in sequence[3]
    assert 'clear zone 4' in sequence[4]
    assert 'backpressure to Conveyor' in sequence[-1]
    assert 'staging nests' in sequence[-1]
    assert payload['Conveyor']['part_transport'] == buffer['part_transport'] == 'direct_on_belt'
    assert payload['Conveyor']['part_orientation_rpy'] == buffer['part_orientation_rpy']
    assert 'part_carriers' not in payload
    assert 'Empty carrier collection' not in payload
    assert all('carrier_supply' not in machine for machine in payload['machines'])


def test_direct_part_zones_fit_every_peg_without_contact_between_parts() -> None:
    """The fixed channel supports one horizontal 50 mm peg in each 120 mm zone."""
    payload = json.loads(ROBOTS_PATH.read_text())
    buffer = payload['Buffer For Machined parts']
    model = ET.parse(WORLD_PATH).find("world/model[@name='Buffer For Machined parts']")
    models = {item.get('name') for item in ET.parse(WORLD_PATH).findall('world/model')}
    assert not any(name and (name.startswith('carrier_') or name.endswith('carrier supply'))
                   for name in models)
    assert len(buffer['zones']) == buffer['capacity'] == 4
    assert buffer['zone_pitch'] - buffer['part_length_m'] == pytest.approx(.07)
    assert buffer['guide_clear_width_m'] > max(buffer['supported_part_cross_section_m'])
    assert buffer['guide_type'] == 'fixed shallow V-channel'
    assert buffer['control_implemented'] is buffer['sensor_feedback_enabled'] is False
    assert buffer['transfer_requirements'][0] == 'downstream zone empty and reserved'
    assert 'destination sensor acknowledges arrival' in buffer['transfer_requirements']
    assert buffer['release_requirements'] == [
        'part pickup acknowledged', 'zone 4 sensor clear', 'robot clear of transfer corridor',
    ]

    bx, _, bz = buffer['world_pose'][:3]
    for index, zone in enumerate(buffer['zones'], 1):
        assert zone['pose'] == buffer['slot_poses'][index - 1]
        assert zone['initial_part'] is None and zone['capacity'] == 1
        assert zone['control_implemented'] is False
        belt = model.find(f"link/collision[@name='{zone['belt_collision']}']")
        x, y, z, *_ = map(float, belt.findtext('pose').split())
        length, width, height = map(float, belt.findtext('geometry/box/size').split())
        assert length > buffer['part_length_m']
        assert buffer['zone_pitch'] - length <= .001 + 1e-9
        assert bx + x == pytest.approx(zone['pose'][0])
        assert y == 0 and width == buffer['width']
        assert bz + z + height / 2 == pytest.approx(buffer['surface_height'])
        stop = model.find(f"link/collision[@name='{zone['stop_collision']}']")
        stop_x, _, stop_z, *_ = map(float, stop.findtext('pose').split())
        stop_length, stop_width, stop_height = map(
            float, stop.findtext('geometry/box/size').split()
        )
        assert stop_x - stop_length / 2 > x + buffer['part_length_m'] / 2
        assert stop_width == buffer['guide_clear_width_m']
        assert stop_z + stop_height / 2 < height / 2
        assert model.find(f"link/visual[@name='{zone['drive_visual']}']") is not None
        eye = model.find(f"link/visual[@name='{zone['sensor_visual']}']")
        eye_z = float(eye.findtext('pose').split()[2])
        assert height / 2 < eye_z < height / 2 + min(buffer['supported_part_cross_section_m'])

    for name, roll in (('guide_left', .25), ('guide_right', -.25)):
        guide = model.find(f"link/collision[@name='{name}']")
        values = list(map(float, guide.findtext('pose').split()))
        assert values[3] == pytest.approx(roll)
        assert abs(values[1]) - .012 / 2 == pytest.approx(buffer['guide_clear_width_m'] / 2)


def test_opening_view_shows_machines_horizontally_left_of_assembly() -> None:
    world = ET.parse(WORLD_PATH)
    camera = world.find('./world/gui/camera')
    assert camera is not None
    x, y, z, roll, pitch, yaw = map(float, camera.findtext('pose').split())
    assert roll == 0.0
    assert camera.findtext('view_controller') == 'orbit'

    def project(point: list[float]) -> tuple[float, float]:
        dx, dy, dz = point[0] - x, point[1] - y, point[2] - z
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        left = -math.sin(yaw) * dx + math.cos(yaw) * dy
        depth = math.cos(pitch) * forward - math.sin(pitch) * dz
        up = math.sin(pitch) * forward + math.cos(pitch) * dz
        assert depth > 0.0
        return -left / depth, up / depth

    machines = json.loads(ROBOTS_PATH.read_text())['machines']
    m1, m2 = [project(machine['world_pose'][:3]) for machine in machines]
    assembly = project([0.0, 0.0, 0.0])
    assert m1[0] < m2[0] < assembly[0]
    assert m1[1] == pytest.approx(m2[1])
    assert m1[1] > assembly[1]
    for machine in machines:
        mx, my, _, _, _, machine_yaw = machine['world_pose']
        # Local +X is front_access; the opening view must face the front doors.
        assert math.cos(machine_yaw) * (x - mx) + math.sin(machine_yaw) * (y - my) > 0.0


def test_haas_mini_mill_has_collision_aware_robot_and_kmr_access() -> None:
    model = ET.parse(MACHINE_MODEL_PATH).getroot().find('model')
    assert model is not None and model.findtext('static') == 'true'
    links = {link.attrib['name']: link for link in model.findall('link')}
    assert {'enclosure', 'workholding', 'front_access', 'side_access'} <= set(links)
    assert links['workholding'].find('collision') is not None
    assert links['front_access'].find('collision') is None
    assert links['side_access'].find('collision') is None
    enclosure_collisions = {item.attrib['name'] for item in links['enclosure'].findall('collision')}
    assert {'base_collision', 'rear_collision', 'roof_collision',
            'front_lower_collision', 'front_upper_collision',
            'side_window_lower_collision', 'side_window_upper_collision'} <= enclosure_collisions
    visuals = {item.attrib['name'] for item in links['enclosure'].findall('visual')}
    assert {'front_door_left_visual', 'front_door_right_visual',
            'side_window_visual', 'control_panel_visual'} <= visuals


@pytest.mark.parametrize('fault', ['duplicate_prefix', 'duplicate_id', 'missing_joint', 'nonfinite_pose', 'missing_robot'])
def test_invalid_robot_bindings_are_rejected_before_launch(tmp_path: Path, fault: str) -> None:
    payload = json.loads(ROBOTS_PATH.read_text())
    first, second = payload['robots'][:2]
    if fault == 'duplicate_prefix':
        second['prefix'] = first['prefix']
    elif fault == 'duplicate_id':
        second['resource_id'] = first['resource_id']
    elif fault == 'missing_joint':
        del second['initial_joint_positions']['wrist_3_joint']
    elif fault == 'nonfinite_pose':
        first['base_xyz'][0] = float('nan')
    else:
        payload['robots'].pop()
    path = tmp_path / 'robots.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        scene._load_robots(path)


def test_each_controller_owns_only_its_ur5e_joints() -> None:
    robots = scene._load_robots(ROBOTS_PATH)
    config = scene._controller_config(robots)
    joint_sets = []
    for robot in robots:
        prefix = robot['prefix']
        for controller, expected in (
            (f'{prefix}joint_trajectory_controller', {prefix + name for name in scene.UR5E_JOINTS}),
            (f'{prefix}rg2_gripper_traj_controller', {f'{prefix}rg2_finger_width'}),
        ):
            params = config[controller]['ros__parameters']
            assert set(params['joints']) == expected
            assert params['allow_partial_joints_goal'] is False
            joint_sets.append(expected)
    assert len(set.union(*joint_sets)) == 28
    assert sum(map(len, joint_sets)) == 28


def test_kmr_routes_are_reversible_and_reject_cross_machine_motion() -> None:
    kmr, routes = kmr_base.load_kmr_config(ROBOTS_PATH)
    assert kmr['simulation_control_integrated'] is True
    assert len(routes) == 2
    storage_m1 = kmr_base.route_for(routes, 'Storage', 'M1')
    storage_m2 = kmr_base.route_for(routes, 'Storage', 'M2')
    assert [value for pose in storage_m1 for value in pose] == pytest.approx([
        value for pose in (
        (-8.15, 2.3, -math.pi / 2), (-7.25, 2.3, -math.pi / 2),
        ) for value in pose
    ])
    assert [value for pose in storage_m2 for value in pose] == pytest.approx([
        value for pose in (
        (-8.15, 2.3, -math.pi / 2),
        (-8.15, 3.85, -math.pi / 2),
        (-3.95, 3.85, -math.pi / 2),
        (-3.95, 2.3, -math.pi / 2),
        (-3.85, 2.3, -math.pi / 2),
        ) for value in pose
    ])
    assert kmr_base.route_for(routes, 'M2', 'Storage') == tuple(reversed(storage_m2))
    assert kmr_base.route_for(routes, 'M1', 'M2') is None
    assert kmr['base_control'] == {
        'linear_speed_mps': 1.2,
        'angular_speed_radps': 0.5,
        'linear_acceleration_mps2': 1.0,
        'angular_acceleration_radps2': 1.0,
        'docking_linear_speed_mps': 0.4,
        'docking_slow_distance_m': 0.2,
        'simulation_speed_override': True,
        'arm_parking_duration_sec': 1.0,
        'arm_hold_duration_sec': 3600.0,
        'arm_parked_tolerance_rad': 0.075,
        'minimum_in_place_angular_speed_radps': 0.1,
        'position_tolerance_m': 0.03,
        'yaw_tolerance_rad': 0.035,
        'control_rate_hz': 20.0,
        'odometry_timeout_sec': 0.5,
        'arm_state_timeout_sec': 2.0,
        'command_timeout_sec': 0.25,
        'waypoint_timeout_sec': 60.0,
    }


def test_kmr_nav2_routes_and_velocity_gate_preserve_safety_boundaries() -> None:
    kmr, routes = kmr_base.load_kmr_config(ROBOTS_PATH)
    endpoints = {
        'Storage': tuple(kmr['initial_pose'][index] for index in (0, 1, 5)),
        'M1': (-7.25, 2.3, -math.pi / 2),
        'M2': (-3.85, 2.3, -math.pi / 2),
    }
    assert kmr_base.docking_poses(routes, endpoints, 'Storage', 'M1') == (
        (-7.25, 2.3, pytest.approx(-math.pi / 2)),
    )
    assert kmr_base.docking_poses(routes, endpoints, 'M1', 'M2') is None
    assert kmr_base.docking_poses(routes, endpoints, None, 'M1') is None
    assert kmr_base.docking_poses(routes, endpoints, None, 'Storage') == (
        (-7.70, 2.3, pytest.approx(-math.pi / 2)),
        endpoints['Storage'],
    )
    vx, vy, angular = kmr_base.clamp_planar_velocity(0.3, 0.4, 0.8, 0.2, 0.35)
    assert math.hypot(vx, vy) == pytest.approx(0.2)
    assert (vx, vy, angular) == pytest.approx((0.12, 0.16, 0.35))
    assert kmr_base.clamp_planar_velocity(
        0.0, 0.0, 0.055, 0.2, 0.35, 0.1,
    ) == pytest.approx((0.0, 0.0, 0.1))
    assert kmr_base.clamp_planar_velocity(
        0.0, 0.0, 0.0, 0.2, 0.35, 0.1,
    ) == pytest.approx((0.0, 0.0, 0.0))
    assert kmr_base.slew_planar_velocity(
        (0.0, 0.0, 0.0), (0.6, 0.8, 0.3), 0.2, 0.1,
    ) == pytest.approx((0.12, 0.16, 0.1))
    assert kmr_base.slew_planar_velocity(
        (0.12, 0.16, 0.1), (0.0, 0.0, 0.0), 0.2, 0.1,
    ) == pytest.approx((0.0, 0.0, 0.0))

    x, y, angular, arrived = kmr_base.docking_velocity(
        (-7.25, 2.3, -math.pi / 2),
        (-8.15, 2.3, -math.pi / 2),
        0.3,
        0.35,
        0.03,
        0.035,
    )
    assert (x, y, angular) == pytest.approx((0.0, -0.3, 0.0))
    assert arrived is False
    assert kmr_base.docking_velocity(
        (-7.25, 2.3, 0.0),
        (-8.15, 2.3, -math.pi / 2),
        0.3,
        0.35,
        0.03,
        0.035,
    ) == pytest.approx((0.0, 0.0, -0.35, False))
    assert kmr_base.docking_velocity(
        (-8.14, 2.30, -math.pi / 2 + 0.01),
        (-8.15, 2.30, -math.pi / 2),
        0.3,
        0.35,
        0.03,
        0.035,
    ) == pytest.approx((0.0, 0.0, 0.0, True))


def test_kmr_fixed_routes_are_clear_for_the_padded_footprint() -> None:
    kmr, routes = kmr_base.load_kmr_config(ROBOTS_PATH)
    endpoints = {
        'Storage': tuple(kmr['initial_pose'][index] for index in (0, 1, 5)),
        'M1': (-7.25, 2.3, -math.pi / 2),
        'M2': (-3.85, 2.3, -math.pi / 2),
    }
    for source, target in (
        ('Storage', 'M1'),
        ('M1', 'Storage'),
        ('Storage', 'M2'),
        ('M2', 'Storage'),
    ):
        goals = kmr_base.docking_poses(routes, endpoints, source, target)
        assert goals is not None
        start = endpoints[source]
        for finish in goals:
            steps = max(1, math.ceil(math.dist(start[:2], finish[:2]) / 0.05))
            for index in range(steps + 1):
                fraction = index / steps
                x = start[0] + fraction * (finish[0] - start[0])
                y = start[1] + fraction * (finish[1] - start[1])
                assert recovery_map.footprint_is_free(x, y, finish[2])
            start = finish


def test_dock_kmr_action_and_controller_contracts_are_exact() -> None:
    action_lines = [
        line for line in DOCK_KMR_ACTION_PATH.read_text(encoding='utf-8').splitlines()
        if not line.startswith('#')
    ]
    assert action_lines == [
        'string target_resource', '---', 'bool success',
        'string final_resource', 'string message', '---',
        'string active_route_from', 'string active_route_to',
        'uint32 waypoint_index', 'geometry_msgs/Pose2D current_pose',
        'float64 remaining_distance_m', 'float64 remaining_yaw_rad',
    ]
    config = scene._kmr_controller_config(KMR_CONTROLLERS_PATH)
    manager = config['/KMR/controller_manager']['ros__parameters']
    assert set(manager) - {'update_rate', 'use_sim_time'} == {
        'joint_state_broadcaster', 'KMR_iiwa_joint_trajectory_controller',
        'KMR_rg2_gripper_traj_controller',
    }
    assert config['/KMR/KMR_iiwa_joint_trajectory_controller']['ros__parameters']['joints'] == list(
        scene.KMR_ARM_JOINTS
    )
    assert config['/KMR/KMR_rg2_gripper_traj_controller']['ros__parameters']['joints'] == [
        'KMR_rg2_finger_width'
    ]


def test_runtime_world_replaces_static_kmr_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    temporary_file = scene.tempfile.NamedTemporaryFile
    monkeypatch.setattr(
        scene.tempfile,
        'NamedTemporaryFile',
        lambda **kwargs: temporary_file(dir=tmp_path, **kwargs),
    )
    runtime_world = scene._runtime_world_without_static_kmr(WORLD_PATH)
    source_names = [item.findtext('name') for item in ET.parse(WORLD_PATH).findall('./world/include')]
    runtime_names = [item.findtext('name') for item in ET.parse(runtime_world).findall('./world/include')]
    assert source_names.count('KMR') == 1
    assert 'KMR' not in runtime_names
    assert set(runtime_names) == set(source_names) - {'KMR'}


def test_recovery_rviz_uses_all_robots_and_current_state_markers() -> None:
    text = RECOVERY_RVIZ_PATH.read_text(encoding='utf-8')
    config = yaml.safe_load(text)
    displays = config['Visualization Manager']['Displays']
    by_name = {display['Name']: display for display in displays}
    assert by_name['RobotModel']['Enabled'] is False
    assert by_name['KMR Fixed Occupancy Map']['Enabled'] is False
    assert config['Visualization Manager']['Global Options']['Frame Rate'] == 15
    assert {'Class': 'nav2_rviz_plugins/Navigation 2', 'Name': 'Navigation 2'} in config['Panels']
    assert 'Planning Group: all_robots' in text
    assert 'Velocity_Scaling_Factor: 0.4' in text
    assert 'Acceleration_Scaling_Factor: 0.3' in text
    assert 'Query Goal State: true' in text
    assert 'Interactive Markers Namespace: /recovery_drag_markers' in text
    assert 'Class: nav2_rviz_plugins/GoalTool' in text
    assert 'Name: KMR Nav2 Path' in text
    assert 'Value: /KMR/plan' in text
    marker_source = RECOVERY_MARKERS_PATH.read_text(encoding='utf-8')
    assert '/recovery_drag_markers/resync' in marker_source
    assert 'Reset all targets to current' in marker_source
    assert '/KMR/dock' in marker_source
    assert '/KMR/compute_path_to_pose' in marker_source
    assert '/KMR/validated_follow_path' in marker_source
    assert '/KMR/cancel_base_motion' in marker_source
    assert 'Plan KMR base (path only)' in marker_source
    assert 'Execute stored KMR base plan (moves)' in marker_source
    assert 'Plan+Execute KMR base (moves)' in marker_source
    assert 'if command in {"plan_base", "plan_execute_base"}' in marker_source
    assert 'self._stage_base_target(feedback.pose)' in marker_source
    assert 'self.last_base_plan_start = plan_start' in marker_source
    assert 'start = self.last_base_plan_start' in marker_source
    assert 'start = self.last_base_path.poses[0].pose' not in marker_source
    assert 'Reset KMR base target to current' in marker_source
    assert 'Cancel KMR base motion' in marker_source
    assert 'InteractiveMarkerControl.MOVE_PLANE' in marker_source
    assert 'InteractiveMarkerControl.ROTATE_AXIS' in marker_source
    assert 'control.name = "KMR_move_xy"' in marker_source
    assert 'control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE' in marker_source
    assert 'control.interaction_mode = InteractiveMarkerControl.MENU' not in marker_source
    assert 'marker.scale = 0.7' in marker_source
    assert 'goal.request.max_velocity_scaling_factor = 0.4' in marker_source
    assert 'goal.request.max_acceleration_scaling_factor = 0.3' in marker_source
    assert 'if not self._state_ready() or (self.markers_ready and not force):' in marker_source
    assert 'marker.pose.position.z = 0.76' in marker_source
    assert 'body.pose.position.x = 0.25' in marker_source
    assert 'body.scale.x, body.scale.y, body.scale.z = 0.35, 0.25, 0.06' in marker_source
    assert marker_source.count('node = RecoveryDragMarkers()') == 1


def test_fixed_kmr_map_and_nav2_parameters_match_the_accepted_cell() -> None:
    metadata = scene.yaml.safe_load(RECOVERY_MAP_YAML_PATH.read_text(encoding='utf-8'))
    assert metadata == {
        'image': 'recovery_framework_map.pgm',
        'mode': 'trinary',
        'resolution': 0.05,
        'origin': [-10.5, -1.5, 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
    }
    content = RECOVERY_MAP_PATH.read_bytes()
    header_end = content.index(b'255\n') + len(b'255\n')
    assert content[header_end:] == recovery_map.map_pixels()
    assert (recovery_map.WIDTH, recovery_map.HEIGHT) == (240, 126)
    assert recovery_map.occupied_at(-6.0, 2.3)
    assert recovery_map.occupied_at(-3.75, 0.5)
    assert not recovery_map.occupied_at(-5.0, 3.85)

    payload = json.loads(ROBOTS_PATH.read_text(encoding='utf-8'))
    docks = [
        payload['KMR']['initial_pose'],
        *(machine['KMR_docking_pose'] for machine in payload['machines']),
    ]
    assert all(
        recovery_map.footprint_is_free(pose[0], pose[1], pose[5]) for pose in docks
    )

    nav2 = scene.yaml.safe_load(RECOVERY_NAV2_PATH.read_text(encoding='utf-8'))
    nav_to = ET.parse(RECOVERY_NAV_TO_PATH)
    nav_through = ET.parse(RECOVERY_NAV_THROUGH_PATH)
    remove_passed = nav_through.find('.//RemovePassedGoals')
    assert remove_passed is not None
    assert float(remove_passed.attrib['radius']) == pytest.approx(0.10)
    assert remove_passed.attrib['global_frame'] == 'world'
    assert remove_passed.attrib['robot_base_frame'] == 'KMR_base_link'
    assert nav2['bt_navigator']['ros__parameters']['default_nav_through_poses_bt_xml'] == (
        'recovery_framework_navigate_through_poses.xml'
    )
    assert nav2['bt_navigator']['ros__parameters']['default_nav_to_pose_bt_xml'] == (
        'recovery_framework_navigate_to_pose.xml'
    )
    assert nav2['bt_navigator']['ros__parameters']['wait_for_service_timeout'] == 10000
    for tree in (nav_to, nav_through):
        assert tree.find('.//FollowPath') is not None
        assert tree.find('.//Spin') is None
        assert tree.find('.//BackUp') is None
    planner = nav2['planner_server']['ros__parameters']['GridBased']
    assert planner['plugin'] == 'nav2_smac_planner/SmacPlanner2D'
    controller = nav2['controller_server']['ros__parameters']['FollowPath']
    assert controller['plugin'] == 'dwb_core::DWBLocalPlanner'
    assert (controller['max_speed_xy'], controller['max_vel_theta']) == (1.2, 0.5)
    assert (controller['max_vel_x'], controller['max_vel_y']) == (1.2, 1.2)
    assert (controller['min_vel_x'], controller['min_vel_y']) == (-1.2, -1.2)
    assert (controller['acc_lim_x'], controller['acc_lim_y']) == (1.0, 1.0)
    assert (controller['decel_lim_x'], controller['decel_lim_y']) == (-1.0, -1.0)
    assert controller['acc_lim_theta'] == 1.0
    assert controller['decel_lim_theta'] == -1.0
    assert controller['sim_time'] == 0.6
    assert controller['linear_granularity'] == 0.02
    assert controller['Twirling.scale'] == 5.0
    goal_checker = nav2['controller_server']['ros__parameters']['precise_goal_checker']
    assert goal_checker['xy_goal_tolerance'] == 0.06
    assert goal_checker['yaw_goal_tolerance'] == 0.05
    assert controller['min_speed_theta'] == 0.10
    assert controller['PathAlign.scale'] == 0.0
    assert controller['GoalAlign.scale'] == 0.0
    assert controller['RotateToGoal.scale'] == 24.0
    progress = nav2['controller_server']['ros__parameters']['progress_checker']
    assert progress == {
        'plugin': 'nav2_controller::PoseProgressChecker',
        'required_movement_radius': 0.05,
        'required_movement_angle': 0.01,
        'movement_time_allowance': 20.0,
    }
    for name in (
        'bt_navigator_navigate_through_poses_rclcpp_node',
        'bt_navigator_navigate_to_pose_rclcpp_node',
    ):
        parameters = nav2[name]['ros__parameters']
        assert parameters['global_frame'] == 'world'
        assert parameters['robot_base_frame'] == 'KMR_base_link'
        assert parameters['odom_topic'] == '/KMR/odom'
    for name in ('local_costmap', 'global_costmap'):
        parameters = nav2[name][name]['ros__parameters']
        assert parameters['footprint'] == (
            '[[0.595, 0.36], [0.595, -0.36], [-0.595, -0.36], [-0.595, 0.36]]'
        )
        assert parameters['footprint_padding'] == 0.03
        assert 'obstacle_layer' not in parameters['plugins']


def test_dragged_kmr_targets_are_checked_before_nav2_planning() -> None:
    data = [
        100
        if recovery_map.occupied_at(
            recovery_map.ORIGIN_X_M + (column + 0.5) * recovery_map.RESOLUTION_M,
            recovery_map.ORIGIN_Y_M + (row + 0.5) * recovery_map.RESOLUTION_M,
        )
        else 0
        for row in range(recovery_map.HEIGHT)
        for column in range(recovery_map.WIDTH)
    ]

    def is_clear(x: float, y: float, yaw: float = -math.pi / 2.0) -> bool:
        return recovery_markers.occupancy_grid_footprint_is_clear(
            data,
            recovery_map.WIDTH,
            recovery_map.HEIGHT,
            recovery_map.RESOLUTION_M,
            recovery_map.ORIGIN_X_M,
            recovery_map.ORIGIN_Y_M,
            x,
            y,
            yaw,
        )

    def controller_is_clear(x: float, y: float, yaw: float = -math.pi / 2.0) -> bool:
        return kmr_base.occupancy_grid_footprint_is_clear(
            data,
            recovery_map.WIDTH,
            recovery_map.HEIGHT,
            recovery_map.RESOLUTION_M,
            recovery_map.ORIGIN_X_M,
            recovery_map.ORIGIN_Y_M,
            x,
            y,
            yaw,
        )

    assert is_clear(-8.15, 2.30)
    assert is_clear(-5.0, 3.85)
    assert not is_clear(-6.0, 2.30)
    assert not is_clear(-3.75, 0.50)
    assert not is_clear(-10.4, 3.0)
    assert controller_is_clear(-8.15, 2.30)
    assert controller_is_clear(-5.0, 3.85)
    assert not controller_is_clear(-6.0, 2.30)
    assert not controller_is_clear(-3.75, 0.50)
    assert not controller_is_clear(-10.4, 3.0)

    marker_source = RECOVERY_MARKERS_PATH.read_text(encoding='utf-8')
    precheck = marker_source.index('if not occupancy_grid_footprint_is_clear(')
    send_goal = marker_source.index('self.compute_path_client.send_goal_async(goal)')
    assert precheck < send_goal
    assert '"/KMR/map"' in marker_source
    assert 'KMR base target is outside the map or overlaps a fixed obstacle' in marker_source


def test_kmr_base_motion_is_nav2_gated_and_fail_closed() -> None:
    controller = KMR_BASE_CONTROLLER_PATH.read_text(encoding='utf-8')
    assert '"/KMR/nav_cmd_vel"' in controller
    assert '"/KMR/cmd_vel"' in controller
    assert 'NavigateToPose' in controller
    assert '"/KMR/navigate_to_pose"' in controller
    assert '"/KMR/validated_navigate_to_pose"' in controller
    assert '"/navigate_to_pose"' in controller
    assert 'self.rviz_navigation_action_server.destroy()' in controller
    assert '"/KMR/validated_follow_path"' in controller
    assert '"/KMR/cancel_base_motion"' in controller
    assert '"/KMR/map"' in controller
    assert 'self._navigation_target_is_clear(request)' in controller
    assert 'self._active_goal or self._follow_path_active' in controller
    assert 'remaining_distance <= self.docking_slow_distance' in controller
    assert 'and self._arm_is_parked()' in controller
    assert '"/KMR/follow_path/_action/status"' in controller
    assert '(self._navigation_active or self._follow_path_active)' in controller
    assert 'now - command_updated <= self.command_timeout' in controller
    assert 'self._stop()' in controller
    assert 'body_velocity(' not in controller
    assert 'asyncio.sleep' not in controller

    launch = LAUNCH_PATH.read_text(encoding='utf-8')
    assert "('cmd_vel', 'nav_cmd_vel')" in launch
    assert "('navigate_to_pose', '/KMR/validated_navigate_to_pose')" in launch
    assert "{'LIBGL_ALWAYS_SOFTWARE': '1'}" in launch
    assert "os.environ.get('WSL_DISTRO_NAME')" in launch
    assert "package='nav2_map_server'" not in launch
    assert "('nav2_map_server', 'map_server', 'map_server', [])" in launch
    assert "'launch_nav2': 'true'" in launch


def test_recovery_controller_startup_does_not_wait_for_spawn_client_response() -> None:
    launch_source = LAUNCH_PATH.read_text(encoding='utf-8')
    assert 'target_action=ur_spawn, on_exit=[ur_spawner]' not in launch_source
    assert 'ur_spawn,\n            # Gazebo can finish inserting' in launch_source
    assert 'ur_spawner,\n            RegisterEventHandler' in launch_source
    assert 'target_action=ur_spawner, on_exit=[kmr_spawn, kmr_spawner]' in launch_source


def test_recovery_rviz_waits_for_live_state_before_showing_goal_models() -> None:
    launch_source = LAUNCH_PATH.read_text(encoding='utf-8')
    assert 'OnProcessIO' in launch_source
    assert 'target_action=recovery_markers' in launch_source
    assert 'on_stderr=start_rviz_from_live_state' in launch_source
    assert 'Recovery markers initialized from live joint and KMR odometry state' in launch_source
    assert "return [rviz]" in launch_source


@pytest.fixture(scope='module')
def models(tmp_path_factory: pytest.TempPathFactory):
    pytest.importorskip('ament_index_python')
    from ament_index_python import get_package_share_directory

    robots = scene._load_robots(ROBOTS_PATH)
    controllers = tmp_path_factory.mktemp('recovery') / 'controllers.yaml'
    controllers.write_text(scene.yaml.safe_dump(scene._controller_config(robots)))
    description = scene._build_description(robots, str(controllers))
    kmr_description = scene._build_kmr_description(
        ROOT / 'ros2/cais_lab_robotics', KMR_CONTROLLERS_PATH,
    )
    planning_description = scene._build_planning_description(description, kmr_description)
    ur_share = Path(get_package_share_directory('ur_moveit_config'))
    srdf = scene._build_srdf(robots, planning_description, ur_share)
    return robots, description, kmr_description, planning_description, srdf, ur_share


def test_urdf_has_four_complete_independent_ur5e_rg2_chains(models) -> None:
    robots, description, _, _, _, _ = models
    root = ET.fromstring(description)
    assert 'xarm' not in description.lower()
    for tag in ('link', 'joint', 'ros2_control'):
        names = [item.attrib['name'] for item in root.findall(tag)]
        assert len(names) == len(set(names))
    plugins = root.findall("gazebo/plugin[@filename='libgazebo_ros2_control.so']")
    assert len(plugins) == 1
    assert len(root.findall('ros2_control')) == 1
    assert root.findtext('ros2_control/hardware/plugin') == 'gazebo_ros2_control/GazeboSystem'
    control_joints = {joint.attrib['name']: joint for joint in root.findall('ros2_control/joint')}
    assert len(control_joints) == 28
    links = {link.attrib['name'] for link in root.findall('link')}
    for robot in robots:
        prefix = robot['prefix']
        assert {prefix + name for name in ('base_link', 'tool0', 'rg2_gripper_tcp')} <= links
        origin = root.find(f"joint[@name='{prefix}world_joint']/origin")
        assert [float(value) for value in origin.attrib['xyz'].split()] == robot['base_xyz']
        assert [float(value) for value in origin.attrib['rpy'].split()] == robot['base_rpy']
        initial = {**robot['initial_joint_positions'], 'rg2_finger_width': 0.11}
        for name, expected in initial.items():
            state = control_joints[prefix + name].find("state_interface[@name='position']/param[@name='initial_value']")
            assert float(state.text) == expected
    for joint in root.findall('joint'):
        assert joint.find('parent').attrib['link'] in links
        assert joint.find('child').attrib['link'] in links
        mimic = joint.find('mimic')
        if mimic is not None:
            assert mimic.attrib['joint'] in {item.attrib['name'] for item in root.findall('joint')}


def test_kmr_runtime_urdf_has_dynamic_base_arm_gripper_and_control(models) -> None:
    _, ur_description, kmr_description, planning_description, _, _ = models
    root = ET.fromstring(kmr_description)
    base = root.find("link[@name='KMR_base_link']")
    assert base is not None
    assert float(base.find('inertial/mass').attrib['value']) == 400.0
    assert all(visual.find('material/color') is not None for visual in base.findall('visual'))
    root_joint = root.find("joint[@name='iiwa_root_joint']/origin")
    assert root_joint.attrib == {
        'xyz': '-0.25 0 0.70', 'rpy': '0 0 1.57079632679',
    }
    controlled = root.findall('ros2_control/joint')
    assert {joint.attrib['name'] for joint in controlled} == {
        *scene.KMR_ARM_JOINTS, 'KMR_rg2_finger_width',
    }
    plugins = {plugin.attrib['filename']: plugin for plugin in root.findall('gazebo/plugin')}
    assert 'libgazebo_ros2_control.so' in plugins
    planar = plugins['libgazebo_ros_planar_move.so']
    assert planar.findtext('odometry_frame') == 'world'
    assert planar.findtext('robot_base_frame') == 'KMR_base_link'
    gravity_free_links = {
        element.attrib['reference']
        for element in root.findall("gazebo")
        if element.findtext('gravity') == 'false'
    }
    kinematic_links = {
        element.attrib['reference']
        for element in root.findall("gazebo")
        if element.findtext('kinematic') == 'true'
    }
    assert {f'iiwa_link_{index}' for index in range(1, 8)} <= gravity_free_links
    assert {
        'rg2_base_link', 'KMR_rg2_finger_width_mock_link',
        'rg2_left_outer_knuckle', 'rg2_right_outer_knuckle',
        'rg2_left_inner_knuckle', 'rg2_right_inner_knuckle',
        'rg2_left_inner_finger', 'rg2_right_inner_finger',
    } <= gravity_free_links
    assert gravity_free_links == kinematic_links
    assert 'KMR_base_link' in kinematic_links

    planning = ET.fromstring(planning_description)
    planning_joints = {joint.attrib['name'] for joint in planning.findall('joint')}
    assert set(scene.KMR_BASE_STATE_JOINTS) <= planning_joints
    assert len(ET.fromstring(ur_description).findall('ros2_control/joint')) + len(controlled) == 36


def test_kmr_runtime_sdf_assigns_explicit_gazebo_colors() -> None:
    names = [token for token, _ in scene.KMR_SDF_VISUAL_COLORS]
    names += [f'wheel_extra_{index}' for index in range(30)]
    visuals = ''.join(
        f'<visual name="KMR_base_link_fixed_joint_lump__{name}_visual_{index}"/>'
        for index, name in enumerate(names)
    )
    root = ET.fromstring(f'<sdf><model><link>{visuals}</link></model></sdf>')
    colored = ET.fromstring(scene._apply_kmr_sdf_materials(ET.tostring(root, encoding='unicode')))
    assert len(colored.findall('.//visual/material/ambient')) == scene.KMR_SDF_EXPECTED_COLOR_COUNT
    assert len(colored.findall('.//visual/material/diffuse')) == scene.KMR_SDF_EXPECTED_COLOR_COUNT


def test_moveit_groups_and_controllers_match_gazebo_without_cross_arm_exclusions(models) -> None:
    robots, description, kmr_description, planning_description, srdf, ur_share = models
    semantic = ET.fromstring(srdf)
    groups = {group.attrib['name'] for group in semantic.findall('group')}
    assert groups == {'all_robots', 'KMR_iiwa_arm', 'KMR_rg2_gripper'} | {
        robot['prefix'] + suffix for robot in robots
        for suffix in ('ur_manipulator', 'rg2_gripper')
    }
    all_robots = semantic.find("group[@name='all_robots']")
    assert [group.attrib['name'] for group in all_robots.findall('group')] == [
        'ur5e_1_ur_manipulator', 'ur5e_2_ur_manipulator',
        'ur5e_3_ur_manipulator', 'ur5e_4_ur_manipulator', 'KMR_iiwa_arm',
    ]
    assert semantic.find("group[@name='dual_robots']") is None
    for pair in semantic.findall('disable_collisions'):
        owners = {robot['prefix'] for robot in robots for link in pair.attrib.values()
                  if link.startswith(robot['prefix'])}
        assert len(owners) <= 1
    parameters = scene._moveit_parameters(robots, planning_description, srdf, ur_share)
    assert parameters['robot_description'] == planning_description
    controller_config = scene._controller_config(robots)
    controllers = parameters['moveit_simple_controller_manager']
    expected_ur_controllers = set(controller_config) - {'controller_manager'}
    assert set(controllers['controller_names']) == expected_ur_controllers | {
        'KMR/KMR_iiwa_joint_trajectory_controller',
        'KMR/KMR_rg2_gripper_traj_controller',
    }
    for name in expected_ur_controllers:
        assert controllers[name]['joints'] == controller_config[name]['ros__parameters']['joints']
    assert controllers['KMR/KMR_iiwa_joint_trajectory_controller']['joints'] == list(
        scene.KMR_ARM_JOINTS
    )
    assert controllers['KMR/KMR_rg2_gripper_traj_controller']['joints'] == [
        'KMR_rg2_finger_width'
    ]
    assert controllers['KMR/KMR_iiwa_joint_trajectory_controller']['action_ns'] == (
        'follow_joint_trajectory'
    )


def test_single_ur5e_builder_retains_its_original_defaults(models) -> None:
    module = scene._load_launch_module('ur5e_rg2_gazebo.launch.py')
    root = ET.fromstring(module._build_ur5e_rg2_description('/tmp/single-ur5e-test.yaml'))
    origin = root.find("joint[@name='ur5e_world_joint']/origin")
    assert origin.attrib == {'xyz': '0.0 0.0 1.021', 'rpy': '0 0 3.142'}
    assert root.find("link[@name='ur5e_tool0']") is not None


def test_launch_resolves_the_nist_cad_meshes(models, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path / 'ros_logs'))
    from launch import LaunchContext
    from launch.actions import AppendEnvironmentVariable

    _, description, kmr_description, planning_description, srdf, _ = models
    monkeypatch.setattr(scene, '_build_description', lambda *args: description)
    monkeypatch.setattr(scene, '_build_kmr_description', lambda *args: kmr_description)
    monkeypatch.setattr(scene, '_build_planning_description', lambda *args: planning_description)
    monkeypatch.setattr(scene, '_kmr_controller_config', lambda *args: {})
    monkeypatch.setattr(scene, '_build_srdf', lambda *args: srdf)
    monkeypatch.setattr(scene, '_write_kmr_runtime_sdf', lambda *args: tmp_path / 'KMR.sdf')
    temporary_file = scene.tempfile.NamedTemporaryFile
    monkeypatch.setattr(scene.tempfile, 'NamedTemporaryFile', lambda **kwargs: temporary_file(dir=tmp_path, **kwargs))
    context = LaunchContext()
    context.environment.pop('GAZEBO_MODEL_PATH', None)
    context.launch_configurations.update({
        'robots_file': str(ROBOTS_PATH), 'world_file': 'table_recovery_framework.world',
        'run_perception': 'false', 'include_assembly_parts': 'true', 'include_loose_parts': 'true',
        'launch_gazebo': 'true', 'launch_gazebo_gui': 'false',
        'launch_moveit': 'false', 'launch_rviz': 'false',
        'launch_nav2': 'false',
    })
    for action in scene.launch_setup(context):
        if isinstance(action, AppendEnvironmentVariable):
            action.execute(context)
    model_paths = [Path(value) for value in context.environment['GAZEBO_MODEL_PATH'].split(os.pathsep)]
    world = ET.parse(ROOT / 'ros2/cais_lab_robotics/worlds/table_recovery_framework.world')
    meshes = {item.text.removeprefix('model://') for item in world.findall('.//mesh/uri')}
    assert len(meshes) == 11
    for mesh in meshes:
        assert any((path / mesh).is_file() for path in model_paths), mesh
    assert any((path / 'cais_lab_robotics/models/KMR').is_dir() for path in model_paths)


def test_dashboard_requires_recovery_robot_assets_without_changing_spec2primitives() -> None:
    arguments = {
        'venv_python': ROOT / '.venv/bin/python',
        'ur5e_rg2_gripper_script': ROOT / 'ros2/cais_lab_robotics/scripts/ur5e_rg2_rtde_gripper.py',
        'ur5e_rtde_trajectory_script': ROOT / 'ros2/cais_lab_robotics/scripts/ur5e_rtde_trajectory_server.py',
    }
    required = ros2_processes.ros2_launch_required_paths('gazebo_dual', **arguments)
    names = {path.name for path, _ in required}
    assert {'recovery_framework_gazebo.launch.py', 'recovery_framework_gazebo.json',
            'recovery_framework.rviz', 'ur5e_rg2_gazebo.launch.py',
            'recovery_framework_kmr_controllers.yaml', 'KMR_recovery.urdf.xacro',
            'kmr_base_controller.py', 'recovery_drag_markers.py',
            'recovery_framework_nav2.yaml', 'recovery_framework_map.yaml',
            'recovery_framework_map.pgm',
            'recovery_framework_navigate_to_pose.xml',
            'recovery_framework_navigate_through_poses.xml'} <= names
    spec_required = ros2_processes.ros2_launch_required_paths('gazebo_dual_spec2primitives', **arguments)
    assert 'recovery_framework_gazebo.json' not in {path.name for path, _ in spec_required}

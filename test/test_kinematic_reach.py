"""Reach from each resource's own kinematic chain, checked against recorded robot states."""

from __future__ import annotations

import math

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cais_spade_llm.resources.kinematic_reach import chain_frames, chain_limits, reach

ARMS = json.loads((Path(__file__).parent / "fixtures/recorded_arm_chains.json").read_text())["arms"]


def _pose(frame):
    return [*frame[:3, 3], *Rotation.from_matrix(frame[:3, :3]).as_quat()]


def _matches(frame, pose):
    target = np.asarray(pose)
    turn = Rotation.from_matrix(frame[:3, :3]).inv() * Rotation.from_quat(target[3:])
    return np.linalg.norm(frame[:3, 3] - target[:3]) < 1e-4 and turn.magnitude() < 1e-3


@pytest.mark.parametrize("rid", sorted(ARMS))
def test_a_recorded_chain_reproduces_and_reaches_its_recorded_pose(rid):
    arm = ARMS[rid]
    configuration = arm["configuration"]
    names, reference = configuration["joint_names"], configuration["reference_link"]
    recorded = chain_frames(configuration, dict(zip(names, arm["joint_positions"], strict=True)))[reference]
    assert _matches(recorded, arm["reference_pose"])
    checked = reach(configuration, arm["reference_pose"])
    assert checked["feasibility_status"] == "FEASIBLE"
    limits = chain_limits(configuration)
    assert all(limits[name]["lower"] <= value <= limits[name]["upper"]
               for name, value in zip(names, checked["joint_positions"], strict=True))
    assert _matches(chain_frames(configuration, dict(zip(names, checked["joint_positions"], strict=True)))[reference],
                    arm["reference_pose"])


@pytest.mark.parametrize("rid", sorted(ARMS))
def test_reach_finds_poses_the_chain_takes_proves_far_ones_and_leaves_the_rest_unresolved(rid):
    arm = ARMS[rid]
    configuration = arm["configuration"]
    names, reference, limits = configuration["joint_names"], configuration["reference_link"], chain_limits(configuration)
    rng = np.random.default_rng(3)
    for _ in range(4):
        positions = {name: 0.9 * rng.uniform(limits[name]["lower"], limits[name]["upper"]) for name in names}
        assert reach(configuration, _pose(chain_frames(configuration, positions)[reference]))["feasibility_status"] \
            == "FEASIBLE"
    far = list(arm["reference_pose"])
    far[0] += 3.0
    assert reach(configuration, far)["feasibility_status"] == "INFEASIBLE"
    unbounded = {**configuration, "joints": [{key: value for key, value in row.items() if key != "position_bounds"}
                                             for row in configuration["joints"]]}
    assert reach(unbounded, arm["reference_pose"])["feasibility_status"] == "NEEDS_CONTEXT"


@pytest.mark.parametrize("rid, start, target, status", [
    # Machining event 7: the Conveyor advances KET4 from loading position 1 (M1) to loading position 2 (M2).
    ("Conveyor", -4.05, -2.05, "FEASIBLE"),
    # Upstream beyond the whole downstream travel is proven impossible; nearer upstream stays unresolved.
    ("Conveyor", -2.05, -4.05, "INFEASIBLE"),
    ("Conveyor", -2.05, -3.0, "NEEDS_CONTEXT"),
    # It hands a part over onto Buffer zone 1, the first slot of its output transfer, and no farther.
    ("Conveyor", -2.05, -0.68, "FEASIBLE"),
    ("Conveyor", -2.05, -0.56, "INFEASIBLE"),
    ("Buffer For Machined parts", -0.68, -0.32, "FEASIBLE"),
    ("Buffer For Machined parts", -0.32, -0.56, "INFEASIBLE"),
])
def test_a_transport_reaches_downstream_stops_only(rid, start, target, status):
    from cais_spade_llm.recovery_framework import SCENE_PATH
    from cais_spade_llm.recovery_framework.workflow_gazebo import transport_chain

    configuration = transport_chain(json.loads(SCENE_PATH.read_text()), rid, [start, 0.5, 1.015])
    assert reach(configuration, [target, 0.5, 1.015, 0.0, 0.0, 0.0, 1.0])["feasibility_status"] == status


@pytest.mark.parametrize("target, status", [
    # The KMR's routes never reach the assembly station; from a free spot beside it the arm reaches its edge.
    ([-0.65, 0.0, 1.05], "FEASIBLE"),
    ([0.0, 0.0, 1.05], "NEEDS_CONTEXT"),
    # M2's staging tray lies inside the ur5e-2 work area; the free spots from which the arm reaches it are
    # narrower than a 0.25 m grid.
    ([-2.6, 1.1, 1.0], "FEASIBLE"),
])
def test_a_mobile_base_reaches_from_a_free_base_pose_its_routes_never_use(target, status):
    from cais_spade_llm.recovery_framework import SCENE_PATH
    from cais_spade_llm.recovery_framework.outline_draft import _navigation_floor
    from cais_spade_llm.resources.kinematic_reach import free_base_poses, reach_from_base

    arm, kmr = ARMS["KMR"], json.loads(SCENE_PATH.read_text())["KMR"]
    candidates = free_base_poses(_navigation_floor(), kmr["platform_overall_dimensions_m"], kmr["initial_pose"][:2],
                                 target[:2])
    checked = reach_from_base(arm["configuration"], [*target, *arm["reference_pose"][3:]], candidates)
    assert checked["feasibility_status"] == status
    if status == "FEASIBLE":
        assert checked["base_pose"] in candidates


def test_a_base_pose_needs_a_free_footprint_and_a_way_from_the_start():
    from cais_spade_llm.resources.kinematic_reach import free_base_poses

    # A 4 x 4 m floor split by a wall at x = 2: from the left half, the right half is out of reach.
    floor = {"floor_extent_xy": [0.0, 4.0, 0.0, 4.0], "floor_obstacles_xy": {"wall": [1.9, 2.1, 0.0, 4.0]}}
    left = free_base_poses(floor, [1.0, 0.6], [1.0, 2.0], [1.0, 2.0], radius_m=0.5)
    right = free_base_poses(floor, [1.0, 0.6], [1.0, 2.0], [3.0, 2.0], radius_m=0.5)
    assert left and not right
    assert all(abs(x - 2.0) >= 0.4 for x, _, _ in left)


def test_the_floor_check_gives_the_base_controllers_answer_on_the_real_map():
    from importlib.util import module_from_spec, spec_from_file_location

    from cais_spade_llm.recovery_framework import SCENE_PATH
    from cais_spade_llm.recovery_framework.outline_draft import ROOT, _navigation_floor
    from cais_spade_llm.resources.kinematic_reach import _FLOOR_CELL_M, _footprint_hits, _occupied_cells

    spec = spec_from_file_location("kmr_base_controller_floor", ROOT / "ros2/cais_lab_robotics/scripts/kmr_base_controller.py")
    controller = module_from_spec(spec)
    spec.loader.exec_module(controller)
    content = (ROOT / "ros2/cais_lab_robotics/config/recovery_framework_map.pgm").read_bytes()
    header = content.index(b"255\n") + 4
    width, height = map(int, content.split(b"\n")[2].split())
    # The file's top row is the map's far edge, as map_server loads it.
    data = [100 if content[header + (height - 1 - row) * width + column] < 100 else 0
            for row in range(height) for column in range(width)]
    floor = _navigation_floor()
    x0, y0 = floor["floor_extent_xy"][0], floor["floor_extent_xy"][2]
    footprint = json.loads(SCENE_PATH.read_text())["KMR"]["platform_overall_dimensions_m"]
    cells = _occupied_cells(floor)
    assert len({(round(x, 6), round(y, 6)) for x, y in cells.tolist()}) == sum(value > 0 for value in data)
    around = [(-0.68, 1.43), (-1.12, 1.3), (-1.15, -0.33), (-5.47, 1.1), (-3.275, 1.4), (-2.7, 2.2), (-5.54, 2.13)]
    poses = [(x + dx, y + dy, yaw) for x, y in around for dx in (-0.06, -0.025, 0.0, 0.02, 0.05)
             for dy in (-0.06, -0.025, 0.0, 0.02, 0.05) for yaw in (0.0, 0.4, math.pi / 2, 2.2, math.pi, -math.pi / 2)]
    # Exact ties, where a cell lies exactly on the padded outline.
    poses += [(x, 1.2, yaw) for x in (-1.075, -3.175) for yaw in (0.0, math.pi, -math.pi)]
    poses += [(x, 1.2, math.pi / 2) for x in (-1.14, -1.31, -1.41, -2.69, -3.31, -3.41)]
    for pose in poses:
        assert controller.occupancy_grid_footprint_is_clear(
            data, width, height, _FLOOR_CELL_M, x0, y0, *pose) != _footprint_hits(cells, footprint, pose), pose


def test_a_footprint_names_the_floor_obstacles_it_overlaps():
    from cais_spade_llm.resources.kinematic_reach import footprint_obstacles

    floor = {"floor_extent_xy": [0.0, 4.0, 0.0, 4.0], "floor_obstacles_xy": {"wall": [1.9, 2.1, 0.0, 4.0]}}
    assert footprint_obstacles(floor, [1.0, 0.6], [1.6, 2.0, 0.0]) == ["wall"]
    # Turned a quarter, the same base is 0.6 m long in x and clears the wall.
    assert footprint_obstacles(floor, [1.0, 0.6], [1.55, 2.0, 1.5707963267948966]) == []
    # The padding counts: the plain outline is 2.5 cm from the wall's first cell, and the padded one is not.
    assert footprint_obstacles(floor, [1.0, 0.6], [1.6, 2.0, 1.5707963267948966]) == ["wall"]

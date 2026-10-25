"""Reach from each resource's own kinematic chain, checked against recorded robot states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cais_spade_llm.resources.kinematic_reach import chain_frames, reach, urdf_joint_limits

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
    configuration, limits = arm["configuration"], arm["limits"]
    names, reference = configuration["joint_names"], configuration["reference_link"]
    recorded = chain_frames(configuration, dict(zip(names, arm["joint_positions"], strict=True)))[reference]
    assert _matches(recorded, arm["reference_pose"])
    joints = reach(configuration, limits, arm["reference_pose"])
    assert joints is not None
    assert all(limits[name][0] <= value <= limits[name][1] for name, value in zip(names, joints, strict=True))
    assert _matches(chain_frames(configuration, dict(zip(names, joints, strict=True)))[reference],
                    arm["reference_pose"])


@pytest.mark.parametrize("rid", sorted(ARMS))
def test_reach_finds_poses_the_chain_can_take_and_refuses_one_beyond_it(rid):
    arm = ARMS[rid]
    configuration, limits = arm["configuration"], arm["limits"]
    names, reference = configuration["joint_names"], configuration["reference_link"]
    rng = np.random.default_rng(3)
    for _ in range(4):
        positions = {name: 0.9 * rng.uniform(*limits[name]) for name in names}
        pose = _pose(chain_frames(configuration, positions)[reference])
        joints = reach(configuration, limits, pose)
        assert joints is not None
        assert _matches(chain_frames(configuration, dict(zip(names, joints, strict=True)))[reference], pose)
    far = list(arm["reference_pose"])
    far[0] += 3.0
    assert reach(configuration, limits, far) is None
    assert reach(configuration, {}, arm["reference_pose"]) is None


@pytest.mark.parametrize("rid, start, target, reachable", [
    # Machining event 7: the Conveyor advances KET4 from loading position 1 (M1) to loading position 2 (M2).
    ("Conveyor", -4.05, -2.05, True),
    ("Conveyor", -2.05, -4.05, False),
    # It hands a part over onto Buffer zone 1, the first slot of its output transfer, and no farther.
    ("Conveyor", -2.05, -0.68, True),
    ("Conveyor", -2.05, -0.56, False),
    ("Buffer For Machined parts", -0.68, -0.32, True),
    ("Buffer For Machined parts", -0.32, -0.56, False),
])
def test_a_transport_reaches_downstream_stops_only(rid, start, target, reachable):
    from cais_spade_llm.recovery_framework import SCENE_PATH
    from cais_spade_llm.recovery_framework.workflow_gazebo import transport_chain

    scene = json.loads(SCENE_PATH.read_text())
    configuration, limits = transport_chain(scene, rid, [start, 0.5, 1.015])
    assert (reach(configuration, limits, [target, 0.5, 1.015, 0.0, 0.0, 0.0, 1.0]) is not None) is reachable


@pytest.mark.parametrize("target, reachable", [
    # The KMR's routes never reach the assembly station; from a free spot beside it the arm reaches its edge.
    ([-0.65, 0.0, 1.05], True),
    ([0.0, 0.0, 1.05], False),
])
def test_a_mobile_base_reaches_from_a_free_base_pose_its_routes_never_use(target, reachable):
    from cais_spade_llm.recovery_framework import SCENE_PATH
    from cais_spade_llm.recovery_framework.outline_draft import _navigation_floor
    from cais_spade_llm.resources.kinematic_reach import free_base_poses, reach_from_base

    arm, kmr = ARMS["KMR"], json.loads(SCENE_PATH.read_text())["KMR"]
    floor, footprint = _navigation_floor(), kmr["platform_overall_dimensions_m"]
    pose = [*target, *arm["reference_pose"][3:]]
    candidates = free_base_poses(floor, footprint, kmr["initial_pose"][:2], target[:2])
    found = reach_from_base(arm["configuration"], arm["limits"], pose, candidates)
    assert (found is not None) is reachable
    if found is not None:
        base, joints = found
        assert base in candidates
        assert all(arm["limits"][name][0] <= value <= arm["limits"][name][1]
                   for name, value in zip(arm["configuration"]["joint_names"], joints, strict=True))


def test_a_base_pose_needs_a_free_footprint_and_a_way_from_the_start():
    from cais_spade_llm.resources.kinematic_reach import free_base_poses

    # A 4 x 4 m floor split by a wall at x = 2: from the left half, the right half is out of reach.
    floor = {"floor_extent_xy": [0.0, 4.0, 0.0, 4.0], "floor_obstacles_xy": {"wall": [1.9, 2.1, 0.0, 4.0]}}
    left = free_base_poses(floor, [1.0, 0.6], [1.0, 2.0], [1.0, 2.0], radius_m=0.5)
    right = free_base_poses(floor, [1.0, 0.6], [1.0, 2.0], [3.0, 2.0], radius_m=0.5)
    assert left and not right
    assert all(abs(x - 2.0) >= 0.4 for x, _, _ in left)


def test_urdf_limits_are_read_per_moving_joint():
    urdf = ('<robot><joint name="a" type="revolute"><limit lower="-1.5" upper="2" effort="1"/></joint>'
            '<joint name="b" type="fixed"/><joint name="c" type="prismatic"><limit upper="0.4" lower="0"/></joint></robot>')
    assert urdf_joint_limits(urdf) == {"a": (-1.5, 2.0), "c": (0.0, 0.4)}

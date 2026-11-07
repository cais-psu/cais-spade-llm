"""MoveIt checks of composed motions on the recorded v7 captures and the v7b compositions.

The native checks need ROS sourced (``preview_recovery_motion`` and its message types) and the recorded
captures; without them these tests skip. Each native request takes about 1.5 s.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from cais_spade_llm.recovery_framework import composed_motion_check as cmc

RECORDS = Path(__file__).resolve().parents[1] / "cais_spade_llm/monitor/environment_runs/recovery"
# The flown v7b programs these tests replay, kept as fixtures after the v7b records were retired (README there).
FIXTURES = Path(__file__).resolve().parents[1] / "cais_spade_llm/monitor/environment_runs/regression_fixtures/composed_motion"
MACHINING = "capture_6cd8cd6940fa4691b67fa68a0099ed34"
PART_SLIPPAGE = "capture_be3e6112cf09475fa6a2d0fb5073af3a"
UR5E1_BREAKDOWN = "capture_4f9097aa46bf450ca0717840262806a4"
COMPOSITIONS = {MACHINING: "composition_aa802336a7af49c3b671b7c46a672a96",
                PART_SLIPPAGE: "composition_3456b6403a184dc38484fb88df5bcfed",
                UR5E1_BREAKDOWN: "composition_f29ddbc3ef164eaa8bc3a95e56a82117"}
MOTIONS = {"move_cartesian", "move_relative", "move_to_named_pose"}


def _model(capture: str) -> cmc.CapturedMotionModel:
    pytest.importorskip("rclpy")
    pytest.importorskip("moveit_msgs.msg")
    pytest.importorskip("cais_lab_robotics.action")
    path = RECORDS / capture / "observations.json"
    if not path.exists() or not (FIXTURES / COMPOSITIONS[capture] / "composition.json").exists():
        pytest.skip("the recorded capture or composition is not on this machine")
    return cmc.load_captured_motion_model(path)


def _rows(capture: str) -> dict[str, dict]:
    path = FIXTURES / COMPOSITIONS[capture] / "composition.json"
    return {row["outline_id"]: row for row in json.loads(path.read_text())["accepted_primitive_program"]}


def _moves(row: dict, steps: set[int]) -> list[dict]:
    return [{"step_index": step["step_index"], "primitive": step["primitive"], "params": step["resolved_params"]}
            for step in row["composition_validation"]["trace"] if step["step_index"] in steps]


def _inverse(pose: list[float]) -> list[float]:
    from cais_spade_llm.recovery_framework.geometry import rotate

    quaternion = [-pose[3], -pose[4], -pose[5], pose[6]]
    return [*rotate(quaternion, [-value for value in pose[:3]]), *quaternion]


def _compose(first: list[float], second: list[float]) -> list[float]:
    from cais_spade_llm.resources.primitive_observations_3d import _compose as compose
    from cais_spade_llm.resources.primitive_observations_3d import _unit_pose

    return [float(value) for value in compose(_unit_pose(first, "first"), _unit_pose(second, "second"))]


class _Replay:
    """Walk a saved composition's robot events through the engine, carrying joints and custody.

    A grasp holds the part where it rests (grasp = inverse(tool) x part); a release leaves it at tool x
    grasp; the last motion before a release is its own request with the place-arrival contact of the
    support the event's compute_place_targets names; the first motion after a grasp carries a support
    departure record. Parts other events move take the pose the saved record projects.
    """

    def __init__(self, model: cmc.CapturedMotionModel):
        self.model = model
        self.state: dict = {"resources": {}, "parts": {}}
        self.rid_of = {jid: rid for rid, jid in model.observations["jids"].items()}
        self.requests: list[dict] = []

    def run(self, rows: list[dict]) -> None:
        for row in rows:
            rid = self.rid_of.get(row["resource_jid"])
            if rid in self.model.robot_ids:
                self.robot_event(rid, row)
                continue
            pose = (row["composition_validation"].get("projected_part_context") or {}).get("observed_pose")
            if pose and row.get("part_name"):
                self.state["parts"][row["part_name"]] = {
                    "current_pose": [pose[key] for key in ("x", "y", "z", "qx", "qy", "qz", "qw")]}

    def robot(self, rid: str) -> dict:
        return self.state["resources"].setdefault(rid, {})

    def current(self, rid: str) -> dict:
        return cmc.projected_snapshot(self.model, self.state)["resources"][rid]

    def robot_event(self, rid: str, row: dict) -> None:
        moves, supports, departing = [], None, False
        for step in row["composition_validation"]["trace"]:
            primitive, params = step["primitive"], step["resolved_params"]
            if primitive in MOTIONS:
                moves.append({"step_index": step["step_index"], "primitive": primitive, "params": params})
            elif primitive == "move_base":
                self.check(rid, row, moves, departing)
                moves, departing = [], False
                x, y, yaw = params["target_pose"]
                self.robot(rid)["base_pose"] = [x, y, 0., 0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]
            elif primitive == "compute_place_targets":
                placement = (step.get("preview_output") or {}).get("support_placement") or {}
                supports = [placement["support_collision_id"]] if placement.get("support_collision_id") else None
            elif primitive == "grasp_part":
                self.check(rid, row, moves, departing)
                moves, departing = [], True
                part = params["part_name"]
                rest = cmc.projected_snapshot(self.model, self.state)["parts"][part]["current_pose"]
                self.state["parts"].pop(part, None)
                self.robot(rid).update(held_part=part, gripper_state="closed",
                                       grasp_transform=_compose(_inverse(self.current(rid)["current_pose"]), rest))
            elif primitive == "release_part":
                self.check(rid, row, moves, departing, supports=supports)
                moves, departing = [], False
                holder = self.current(rid)
                rest = _compose(holder["current_pose"], holder["grasp_transform"])
                self.robot(rid).update(held_part=None, gripper_state="open", grasp_transform=None)
                self.state["parts"][holder["held_part"]] = {"current_pose": rest}
        self.check(rid, row, moves, departing)

    def check(self, rid: str, row: dict, moves: list[dict], departing: bool, *, supports=None) -> None:
        groups = [(moves, ())]
        if supports and moves:
            arrival = cmc.place_arrival_contact(self.model, self.current(rid)["held_part"], supports)
            groups = [(moves[:-1], ()), (moves[-1:], (arrival,))]
        for group, contacts in groups:
            if not group:
                continue
            departure = None
            if departing:
                departure = cmc.support_departure_record(self.model, self.state, rid)
                departing = False
            result = cmc.check_robot_moves(self.model, self.state, rid, group, allowed_contacts=contacts,
                                           support_departure=departure)
            self.requests.extend(result["requests"])
            assert result["status"] == "FEASIBLE", f"{row['outline_id']} {rid}: {result['reason']}"
            self.robot(rid)["joint_positions"] = result["end_joints"]



@pytest.fixture(scope="module")
def machining() -> cmc.CapturedMotionModel:
    return _model(MACHINING)


@pytest.fixture(scope="module")
def part_slippage() -> cmc.CapturedMotionModel:
    return _model(PART_SLIPPAGE)


@pytest.fixture(scope="module")
def ur5e1_breakdown() -> cmc.CapturedMotionModel:
    return _model(UR5E1_BREAKDOWN)


@pytest.fixture(scope="module")
def machining_replay(machining: cmc.CapturedMotionModel) -> _Replay:
    rows = _rows(MACHINING)
    replay = _Replay(machining)
    replay.run([rows[f"RECOVERY_SEQ{index}"] for index in range(1, 8)])
    return replay


@pytest.fixture(scope="module")
def machining_seq8_start(machining_replay: _Replay) -> dict:
    return machining_replay.state


@pytest.fixture(scope="module")
def part_slippage_seq2_start(part_slippage: cmc.CapturedMotionModel) -> dict:
    replay = _Replay(part_slippage)
    replay.run([_rows(PART_SLIPPAGE)["RECOVERY_SEQ1"]])
    return replay.state


def test_the_lean_scene_is_project_preview_scene_with_its_native_fingerprint(part_slippage):
    from cais_spade_llm.recovery_framework.recovery_motion_preview import (
        preview_scene_fingerprint,
        project_preview_scene,
    )

    model = part_slippage
    gear = cmc.projected_snapshot(model, {})["parts"]["gear_small"]["current_pose"]
    tool = cmc.projected_snapshot(model, {})["resources"]["ur5e-3"]["current_pose"]
    home = model.arms["ur5e-4"]["named_joint_poses"]["home"]["positions"]
    state = {"resources": {"ur5e-3": {"held_part": "gear_small", "grasp_transform": _compose(_inverse(tool), gear),
                                      "gripper_state": "closed"},
                           "ur5e-4": {"joint_positions": home}},
             "parts": {"KET4_Square_4mm": {"current_pose": [-0.68, 0.5, 1.017, 0., 0., math.sqrt(.5), math.sqrt(.5)]}}}
    snapshot = cmc.projected_snapshot(model, state)
    lean = cmc._project_scene(model, snapshot)
    assert lean == project_preview_scene(model.scene_capture, {**model.physical, "snapshot": snapshot})["planning_scene"]
    scene = cmc.motion_scene(model, state)
    assert scene["scene_fingerprint"] == preview_scene_fingerprint(scene["planning_scene"])
    attached = {row["object"]["id"]: row["link_name"] for row in scene["planning_scene"]["robot_state"]["attached_collision_objects"]}
    assert attached == {"gear_small/link/collision": "ur5e_3_wrist_3_link"}


def test_parts_missing_from_the_captured_scene_take_their_recorded_collision(machining):
    sources = {row["part"]: row["source"] for row in machining.identity_evidence}
    assert sources["KET4_Square_4mm"].startswith("recorded physics collision KET4_Square_4mm::link::collision")
    assert sources["RGOCG4-50_Round_4mm"].startswith("KMR.part_collision released object identity")
    world = {row["id"]: row for row in cmc.motion_scene(machining, {})["planning_scene"]["world"]["collision_objects"]}
    assert world["KET4_Square_4mm/link/collision"]["primitives"][0]["dimensions"] == [0.004, 0.004, 0.05]
    # The KMR's released box under the bare model id now holds RGOCG4's recorded cylinder.
    assert world["RGOCG4-50_Round_4mm"]["primitives"][0] == {"type": 3, "dimensions": [0.05, 0.002]}


def test_fingers_take_the_custody_values_of_the_declared_gripper_state(part_slippage):
    model = part_slippage
    for gripper in ("open", "closed"):
        state = {"resources": {"ur5e-3": {"held_part": None, "gripper_state": gripper}}}
        scene = cmc.motion_scene(model, state)
        joints = dict(zip(*(scene["planning_scene"]["robot_state"]["joint_state"][key] for key in ("name", "position")),
                          strict=True))
        assert all(joints[name] == value for name, value in model.arms["ur5e-3"]["custody"][gripper].items())
        assert scene["evidence"]["fingers"]["ur5e-3"] == f"custody.joint_positions.{gripper}"
    assert scene["evidence"]["fingers"]["KMR"] == "custody_by_part.default.joint_positions.open"


def test_a_moved_kmr_base_writes_its_joints_and_leaves_out_the_recorded_root(ur5e1_breakdown):
    model = ur5e1_breakdown
    captured = cmc.motion_scene(model, {})
    assert captured["evidence"]["kmr_base_joints"] is None
    assert "KMR_base_link" in [row["child_frame_id"] for row in captured["recorded_root_transforms"]]
    dock = [-4.7, 2.2, 0., 0., 0., math.sin(math.pi / 4), math.cos(math.pi / 4)]
    moved = cmc.motion_scene(model, {"resources": {"KMR": {"base_pose": dock}}})
    assert moved["evidence"]["kmr_base_joints"] == pytest.approx([-4.7, 2.2, math.pi / 2])
    assert "KMR_base_link" not in [row["child_frame_id"] for row in moved["recorded_root_transforms"]]
    assert cmc.robot_pose(model, "KMR", cmc.projected_snapshot(model, {})["resources"]["KMR"]["joint_positions"],
                          base_pose=dock)[:3] == pytest.approx([-5.2, 1.95, 1.16], abs=1e-6)


def test_a_contact_allowance_pairs_a_part_with_its_recorded_support_never_a_robot(machining):
    contact = cmc.place_arrival_contact(machining, "KET4_Square_4mm", ["M2::workholding::workholding_collision"])
    assert contact["pairs"] == [["KET4_Square_4mm/link/collision", "M2/workholding/workholding_collision"]]
    assert "does not bound the contact depth" in contact["statement"]
    with pytest.raises(ValueError, match="never names a robot link"):
        cmc.motion_scene(machining, {}, allowed_contacts=[{"pairs": [["ur5e_1_wrist_3_link", "ur5e_2_wrist_3_link"]]}])
    with pytest.raises(cmc.MissingMotionContext, match="no object in the captured MoveIt scene"):
        cmc.place_arrival_contact(machining, "KET4_Square_4mm", ["M2::workholding::no_such_collision"])


def test_ur5e_routes_follow_the_controller_from_the_projected_start(part_slippage, machining):
    start = [0.2, 0.45, 1.35, math.sqrt(.5), math.sqrt(.5), 0., 0.]
    vertical = cmc.arm_motion(part_slippage, "ur5e-3", "move_cartesian", {"x": 0.2, "y": 0.45, "z": 1.3},
                              start_pose=start)
    assert vertical["waypoints"] == [[0.2, 0.45, 1.3, *start[3:]]]
    turned = cmc.arm_motion(part_slippage, "ur5e-3", "move_cartesian",
                            {"x": 0.4, "y": 0.15, "z": 1.3, "qx": 1., "qy": 0., "qz": 0., "qw": 0.}, start_pose=start)
    rotation_point = part_slippage.arms["ur5e-3"]["cartesian_motion"]["rotation_waypoint"]
    assert turned["waypoints"][0][:3] == rotation_point and len(turned["waypoints"]) > 3
    # ur5e-2 keeps its start orientation when the stated one is within 0.02 rad.
    snapped = cmc.arm_motion(machining, "ur5e-2", "move_cartesian",
                             {"x": -2.2, "y": 0.6, "z": 1.3, "qx": 0., "qy": 1., "qz": 0.005, "qw": 0.},
                             start_pose=[-2.05, 0.5, 1.3, 0., 1., 0., 0.])
    assert snapped["waypoints"] == [[-2.2, 0.6, 1.3, 0., 1., 0., 0.]]
    relative = cmc.arm_motion(part_slippage, "ur5e-3", "move_relative", {"dx": 0., "dy": 0., "dz": 0.05},
                              start_pose=start)
    assert relative["waypoints"] == [pytest.approx([0.2, 0.45, 1.4, *start[3:]])]
    partial = cmc.arm_motion(part_slippage, "ur5e-3", "move_cartesian", {"x": 0.3, "y": 0.4, "z": 1.3, "qx": 1.},
                             start_pose=start)
    assert partial["refusal"] == "move_cartesian orientation requires qx, qy, qz, and qw together"


def test_named_poses_follow_the_controller(part_slippage):
    model = part_slippage
    home = model.arms["ur5e-3"]["named_joint_poses"]["home"]["positions"]
    there = cmc.arm_motion(model, "ur5e-3", "move_to_named_pose", {"pose_name": "home"},
                           start_pose=cmc.robot_pose(model, "ur5e-3", home), start_joints=home)
    assert there["waypoints"] == []
    ur5e4 = cmc.projected_snapshot(model, {})["resources"]["ur5e-4"]
    preserved = cmc.arm_motion(model, "ur5e-4", "move_to_named_pose", {"pose_name": "home"},
                               start_pose=ur5e4["current_pose"], start_joints=ur5e4["joint_positions"])
    target = cmc.robot_pose(model, "ur5e-4", model.arms["ur5e-4"]["named_joint_poses"]["home"]["positions"])
    assert preserved["named_joints"] is None
    assert preserved["waypoints"][-1][:3] == pytest.approx(target[:3])
    assert preserved["waypoints"][-1][3:] == pytest.approx(ur5e4["current_pose"][3:])
    unknown = cmc.arm_motion(model, "ur5e-3", "move_to_named_pose", {"pose_name": "park"},
                             start_pose=cmc.robot_pose(model, "ur5e-3", home), start_joints=home)
    assert unknown["refusal"] == "unknown pose 'park'; available=['home']"


def test_kmr_routes_follow_the_worker(ur5e1_breakdown):
    model = ur5e1_breakdown
    dock = [-4.7, 2.2, 0., 0., 0., math.sin(math.pi / 4), math.cos(math.pi / 4)]
    joints = cmc.projected_snapshot(model, {})["resources"]["KMR"]["joint_positions"]
    start = cmc.robot_pose(model, "KMR", joints, base_pose=dock)
    tilted = [*start[:3], 0.9998, 0., 0.02, 0.]
    refused = cmc.kmr_motion(model, "move_cartesian", {"target": [-4.9, 1.8, 1.1, *tilted[3:]]}, start_pose=start,
                             base_pose=dock)
    assert refused["refusal"] == "Cartesian segment must preserve the observed TCP orientation"
    transport = cmc.kmr_motion(model, "move_to_named_pose", {"pose_name": "transport"}, start_pose=start,
                               base_pose=dock)
    assert transport["waypoints"] == [] and transport["route"].startswith("TCP already at the target")
    home = cmc.kmr_motion(model, "move_to_named_pose", {"pose_name": "home"}, start_pose=start, base_pose=dock)
    assert home["gripper_state"] == "open"
    holding = cmc.kmr_motion(model, "move_to_named_pose", {"pose_name": "home"}, start_pose=start, base_pose=dock,
                             held_part="KET4_Square_4mm")
    assert holding["refusal"] == "KMR downward home requires an empty gripper"
    lowered = cmc.kmr_motion(model, "move_relative", {"offset": [0., 0., -0.05]}, start_pose=start, base_pose=dock)
    assert lowered["waypoints"] == [pytest.approx([start[0], start[1], start[2] - 0.05, *start[3:]])]


def test_the_replay_to_seq8_lifts_each_grasped_part_off_its_support(machining_replay):
    lifts = [request for request in machining_replay.requests if request["steps"] == [5]]
    assert [request["status"] for request in lifts] == ["PREVIEW_COMPLETE"] * 3
    # ur5e-1 lifts KET4 from M1, ur5e-2 lifts RGOCG4 (a cylinder) from M2 and KET4 from the conveyor, each with a
    # support departure record and no contact allowance.
    assert [request["support_departure"] for request in lifts] == [True, True, True]
    assert [request["allowed_pairs"] for request in lifts] == [[], [], []]


def test_machining_seq8_routed_program_completes_five_times(machining, machining_seq8_start):
    """Gate (1): the routed SEQ8 program completes, the place arrival allowed, five times over."""
    row = _rows(MACHINING)["RECOVERY_SEQ8"]
    start = machining_seq8_start["resources"]["ur5e-2"]["joint_positions"]
    assert cmc.robot_pose(machining, "ur5e-2", start)[:3] == pytest.approx([-2.05, 0.5, 1.3005], abs=1e-4)
    for _ in range(5):
        cmc.clear_result_cache()
        replay = _Replay(machining)
        replay.state = copy.deepcopy(machining_seq8_start)
        replay.run([row])
        assert [request["steps"] for request in replay.requests] == [[1, 2, 3], [4], [6]]
        assert [request["status"] for request in replay.requests] == ["PREVIEW_COMPLETE"] * 3
        assert replay.requests[1]["allowed_pairs"] == [["KET4_Square_4mm/link/collision",
                                                        "M2/workholding/workholding_collision"]]
        assert replay.requests[2]["fingers"]["ur5e-2"] == "custody.joint_positions.open"


def test_machining_seq8_straight_move_is_refused_by_ik_near_the_base_axis(machining, machining_seq8_start):
    """Gate (1): straight from (-2.05, 0.5, 1.3005) to the M2 approach stops where IK fails by the base axis."""
    cmc.clear_result_cache()
    row = _rows(MACHINING)["RECOVERY_SEQ8"]
    result = cmc.check_robot_moves(machining, machining_seq8_start, "ur5e-2", _moves(row, {3}))
    assert result["status"] == "INFEASIBLE" and result["step_index"] == 3
    assert "stopped 35% of the way along waypoint 1 of 1 toward (-2.1300, 1.8200, 1.3455)" in result["reason"]
    assert "the IK search found no joint solution there" in result["reason"]
    reached = result["reason"].split("the last reached tool pose is (")[1].split(")")[0]
    x, y, _ = (float(value) for value in reached.split(", "))
    assert math.hypot(x + 2.05, y - 1.1) < 0.15
    assert result["requests"][0]["fraction"] == pytest.approx(47 / 133)


def test_part_slippage_seq2_meets_ur5e4_at_its_failure_joints(part_slippage, part_slippage_seq2_start):
    """Gate (2): with ur5e-4 where it stopped, ur5e-3's route to the gear runs into ur5e-4."""
    cmc.clear_result_cache()
    row = _rows(PART_SLIPPAGE)["RECOVERY_SEQ2"]
    result = cmc.check_robot_moves(part_slippage, part_slippage_seq2_start, "ur5e-3", _moves(row, {1, 2, 3, 4, 5}))
    assert result["status"] == "INFEASIBLE" and result["step_index"] == 2
    assert [step["step_index"] for step in result["steps"]] == [1]
    assert "ur5e_3_rg2_onrobot_base_link <> ur5e_4_wrist_1_link" in result["reason"]


def test_part_slippage_seq2_steps_one_to_three_complete_with_ur5e4_home(part_slippage, part_slippage_seq2_start):
    """Gate (2): with ur5e-4 at its named home, steps 1-3 complete; the open finger then meets the Exit tray."""
    cmc.clear_result_cache()
    row = _rows(PART_SLIPPAGE)["RECOVERY_SEQ2"]
    state = copy.deepcopy(part_slippage_seq2_start)
    state["resources"]["ur5e-4"] = {"joint_positions": part_slippage.arms["ur5e-4"]["named_joint_poses"]["home"]["positions"]}
    result = cmc.check_robot_moves(part_slippage, state, "ur5e-3", _moves(row, {1, 2, 3, 4, 5}))
    assert [step["step_index"] for step in result["steps"]][:3] == [1, 2, 3]
    assert result["status"] == "INFEASIBLE" and result["step_index"] == 4
    assert "Exit/link/side_edge_1 <> ur5e_3_rg2_right_inner_finger" in result["reason"]


def test_kmr_arc_and_descent_onto_the_m2_tray_complete(ur5e1_breakdown):
    """Gate (3): the KMR's SEQ5 arc from the transport pose to the M2 tray, and its descent, complete."""
    from cais_spade_llm.resources.recovery_geometry import projected_part_support

    model = ur5e1_breakdown
    rows = _rows(UR5E1_BREAKDOWN)
    replay = _Replay(model)
    replay.run([rows["RECOVERY_SEQ2"]])
    trace = rows["RECOVERY_SEQ5"]["composition_validation"]["trace"]
    dock = [-2.7, 2.2, 0., 0., 0., math.sin(math.pi / 4), math.cos(math.pi / 4)]
    # SEQ4 drove to the M2 dock holding KET4; the arm stands at its captured transport joints.
    state = {"resources": {"ur5e-1": replay.state["resources"]["ur5e-1"],
                           "KMR": {"base_pose": dock, "held_part": "KET4_Square_4mm", "gripper_state": "closed",
                                   "grasp_transform": trace[0]["resolved_params"]["transform"]}}}
    cmc.clear_result_cache()
    arc = cmc.check_robot_moves(model, state, "KMR", _moves(rows["RECOVERY_SEQ5"], {1}))
    assert arc["status"] == "FEASIBLE", arc["reason"]
    assert arc["steps"][0]["end_pose"][:3] == pytest.approx(trace[1]["resolved_params"]["target"][:3], abs=1e-9)
    assert arc["requests"][0]["waypoints"] == 14 and arc["requests"][0]["label"] == cmc.KMR_IK_PATH_LABEL
    probe = cmc.projected_snapshot(model, state)
    probe["resources"]["KMR"]["held_part"] = None
    probe["parts"]["KET4_Square_4mm"]["current_pose"] = trace[0]["preview_output"]["destination"]
    support = projected_part_support(part_name="KET4_Square_4mm", state=probe, evidence=model.physical)
    arrival = cmc.place_arrival_contact(model, "KET4_Square_4mm", support["support_collisions"])
    state["resources"]["KMR"]["joint_positions"] = arc["end_joints"]
    descent = cmc.check_robot_moves(model, state, "KMR", _moves(rows["RECOVERY_SEQ5"], {2}), allowed_contacts=[arrival])
    assert descent["status"] == "FEASIBLE", descent["reason"]
    assert descent["requests"][0]["allowed_pairs"] == [["KET4_Square_4mm/link/collision",
                                                        "M2 staging tray/link/surface"]]


def _kmr_at_the_m2_tray(model: cmc.CapturedMotionModel) -> tuple[dict, dict, dict, dict]:
    """The KMR at the M2 dock with KET4, its SEQ5 arc flown, and the tray's arrival allowance for KET4."""
    from cais_spade_llm.resources.recovery_geometry import projected_part_support

    rows = _rows(UR5E1_BREAKDOWN)
    replay = _Replay(model)
    replay.run([rows["RECOVERY_SEQ2"]])
    trace = rows["RECOVERY_SEQ5"]["composition_validation"]["trace"]
    dock = [-2.7, 2.2, 0., 0., 0., math.sin(math.pi / 4), math.cos(math.pi / 4)]
    state = {"resources": {"ur5e-1": replay.state["resources"]["ur5e-1"],
                           "KMR": {"base_pose": dock, "held_part": "KET4_Square_4mm", "gripper_state": "closed",
                                   "grasp_transform": trace[0]["resolved_params"]["transform"]}}}
    arc = cmc.check_robot_moves(model, state, "KMR", _moves(rows["RECOVERY_SEQ5"], {1}))
    assert arc["status"] == "FEASIBLE", arc["reason"]
    state["resources"]["KMR"]["joint_positions"] = arc["end_joints"]
    probe = cmc.projected_snapshot(model, state)
    probe["resources"]["KMR"]["held_part"] = None
    probe["parts"]["KET4_Square_4mm"]["current_pose"] = trace[0]["preview_output"]["destination"]
    support = projected_part_support(part_name="KET4_Square_4mm", state=probe, evidence=model.physical)
    arrival = cmc.place_arrival_contact(model, "KET4_Square_4mm", support["support_collisions"])
    return state, rows, probe, arrival


def test_a_set_down_rests_within_the_recorded_contact_layer_only_where_composition_asks(ur5e1_breakdown):
    """F1: KET4's recorded contact_min_depth is 0, so the RA's exact test refuses a part 0.5 µm off the tray face.

    Composition counts a vertex within the recorded contact_surface_layer (1 mm) of the face, on either side, as
    touching it; a part 2 mm inside, or 2 mm above, still rests on no support.
    """
    from cais_spade_llm.resources.recovery_geometry import projected_part_support

    model = ur5e1_breakdown
    _state, _rows_unused, probe, _arrival = _kmr_at_the_m2_tray(model)
    resting = probe["parts"]["KET4_Square_4mm"]["current_pose"]

    def support(dz: float, tolerance: bool) -> dict:
        moved = copy.deepcopy(probe)
        moved["parts"]["KET4_Square_4mm"]["current_pose"] = [resting[0], resting[1], resting[2] + dz, *resting[3:]]
        return projected_part_support(part_name="KET4_Square_4mm", state=moved, evidence=model.physical,
                                      contact_layer_tolerance=tolerance)

    assert support(0., False)["status"] == "supported"
    for dz in (-5e-7, 5e-7):
        assert support(dz, False)["status"] != "supported"
        found = support(dz, True)
        assert found["status"] == "supported", found.get("reason")
        assert found["source"]["contact_layer_tolerance"] is True
        assert {row["penetration_allowance"] for row in found["source"]["contact_margins"]} == {0.001}
    for dz in (-0.002, 0.002):
        assert support(dz, True)["status"] != "supported"


def test_a_kmr_place_arrival_allows_the_tray_contact_on_its_final_segment_only(ur5e1_breakdown):
    """F1: a descent that states a waypoint halfway down arrives on the tray; one that dips below it is refused.

    The path up to the last waypoint is checked first with no contact allowed; the arrival allowance covers the
    final segment only, so a stated waypoint inside the support meets the support there.
    """
    model = ur5e1_breakdown
    state, rows, _probe, arrival = _kmr_at_the_m2_tray(model)
    descent = _moves(rows["RECOVERY_SEQ5"], {2})
    start = cmc.robot_pose(model, "KMR", state["resources"]["KMR"]["joint_positions"],
                           base_pose=cmc._planar_base_pose(state["resources"]["KMR"]["base_pose"]))
    target = list(descent[0]["params"]["target"])
    halfway = [(a + b) / 2 for a, b in zip(start[:3], target[:3], strict=True)] + target[3:]

    def with_waypoint(waypoint: list[float]) -> list[dict]:
        return [{**descent[0], "params": {**descent[0]["params"], "waypoints": [waypoint]}}]

    cmc.clear_result_cache()
    stated = cmc.check_robot_moves(model, state, "KMR", with_waypoint(halfway), allowed_contacts=[arrival])
    assert stated["status"] == "FEASIBLE", stated["reason"]
    before, arriving = stated["requests"][0], stated["requests"][-1]
    assert before["before_place_arrival"] is True and before["allowed_pairs"] == []
    assert arriving["allowed_pairs"] == [["KET4_Square_4mm/link/collision", "M2 staging tray/link/surface"]]
    dip = cmc.check_robot_moves(model, state, "KMR", with_waypoint([*target[:2], target[2] - 0.005, *target[3:]]),
                                allowed_contacts=[arrival])
    assert dip["status"] == "INFEASIBLE" and dip["step_index"] == 2
    assert "M2 staging tray/link/surface" in dip["reason"] and "KET4_Square_4mm" in dip["reason"]
    assert dip["requests"][-1].get("before_place_arrival") is True


def test_kmr_seq3_arc_goes_on_from_moveit_s_state_where_the_refinement_stops_at_a7_s_bound(ur5e1_breakdown):
    """The v7b ur5e-1 SEQ3 arc to M1 turns the KMR wrist onto joint_a7's bound in MoveIt's IK path.

    MoveIt's path reaches waypoint 21 of the arc and then the grasp pose with joint_a7 on its bound 3.0541, where
    the native refinement to 1e-13 m cannot move it further; the path goes on from the state MoveIt reached.
    """
    model = ur5e1_breakdown
    rows = _rows(UR5E1_BREAKDOWN)
    replay = _Replay(model)
    replay.run([rows["RECOVERY_SEQ1"], rows["RECOVERY_SEQ2"]])
    cmc.clear_result_cache()
    result = cmc.check_robot_moves(model, replay.state, "KMR", _moves(rows["RECOVERY_SEQ3"], {0, 3, 4}))
    assert result["status"] == "FEASIBLE", result["reason"]
    stops = result["refinement_stops"]
    assert [stop["waypoint"] for stop in stops] == [21, 23]
    assert all(stop["joints_at_bounds"] == ["joint_a7 is at its bound 3.0541"] for stop in stops)
    assert all(stop["position_residual_m"] < 1e-5 for stop in stops)
    # Every native request stays in the evidence, labelled as one IK path; the second goes on from waypoint 21.
    assert [(row["status"], row.get("continued_from_waypoint")) for row in result["requests"][-2:]] == [
        ("NEEDS_CONTEXT", None), ("NEEDS_CONTEXT", 21)]
    assert all(row["label"] == cmc.KMR_IK_PATH_LABEL for row in result["requests"])
    # The approach ends inside a7's bound, and the grasp pose on it, at MoveIt's own states.
    approach, grasp = result["steps"][-2:]
    assert approach["end_joints"][6] < 3.0541 and grasp["end_joints"][6] == pytest.approx(3.0541, abs=1e-6)
    target = rows["RECOVERY_SEQ3"]["composition_validation"]["trace"][4]["resolved_params"]["target"]
    assert grasp["end_pose"][:3] == pytest.approx(target[:3], abs=1e-5)


def _bound_to_model(program: list[dict], jid: str, request: dict) -> list[dict]:
    """The saved program with each number it writes bound to the same number of the robot's nominal steps.

    Every number in a program is a reference into the served context (D1). The flown programs typed theirs, so each is
    bound to the nominal step parameter of that name and value in the resource's model, as a program the typed-number
    check accepts would bind it.
    """
    from cais_spade_llm.resources.recovery_composition import typed_number_refusal

    events = request["recovery_resources"][jid]["recovery_des_model"]["events"]
    sources = {(name, value): f"/resources/{jid}/recovery_des_model/events/{event}/composition_contract/steps/{step}/"
                              f"params/{name}"
               for event, row in reversed(list(enumerate(events)))
               for step, nominal in reversed(list(enumerate((row.get("composition_contract") or {}).get("steps") or [])))
               for name, value in (nominal.get("params") or {}).items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)}
    bound = copy.deepcopy(program)
    for step in bound:
        for name, value in step["params"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and (name, value) in sources:
                step["params"][name] = {"context_ref": sources[name, value]}
    assert typed_number_refusal(bound) is None
    return bound


UR5E1_COMPOSITION = COMPOSITIONS[UR5E1_BREAKDOWN]


def test_a_kmr_program_is_checked_from_the_owner_drive_and_moves_alike(ur5e1_breakdown):
    """ur5e-1 breakdown v7b through the owners: the KMR's drive to M1, ur5e-1's home, then the KMR's pick at M1."""
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _accepted_program_row,
        _validate_single_event_primitive_steps,
    )
    from cais_spade_llm.recovery_framework.offline_recovery import SavedRecoveryResource
    from cais_spade_llm.recovery_framework.recovery_observations import read_observations

    folder = FIXTURES / UR5E1_COMPOSITION
    source = json.loads((folder / "outline_request.json").read_text())
    composition = json.loads((folder / "composition.json").read_text())
    request = {key: value for key, value in source.items() if key not in {"multi_turn_session_state", "recovery_debug"}}
    session = {key: value for key, value in source["multi_turn_session_state"].items()
               if key not in {"turns", "projected_physical_evidence", "outline_monitor_context"}}
    session.update(composition_validation_version=1, current_phase="primitive_generation", accepted_primitive_program=[])
    request.update(composition_validation_version=1, multi_turn_session_state=session)
    observations = read_observations(RECORDS / UR5E1_BREAKDOWN / "observations.json")
    owners = {jid: SavedRecoveryResource(jid, entry, observations, _copy_observations=False)
              for jid, entry in observations["prepared_recovery_request"]["recovery_resources"].items()}
    programs = {row["outline_id"]: _bound_to_model(row["primitive_steps"], row["resource_jid"], request)
                for row in composition["accepted_primitive_program"]}
    events = {task["outline_id"]: task for task in session["accepted_outline_prefix"]}

    def accept(outline_id: str) -> dict:
        task = events[outline_id]
        checked, _ = _validate_single_event_primitive_steps(session_state=session, prepared_recovery_request=request,
            outline_event=task, primitive_steps=programs[outline_id], composition_owner=owners[task["resource_jid"]])
        evidence = checked["composition_validation"]
        assert evidence["feasibility_status"] == "FEASIBLE", evidence.get("reason")
        accepted = _accepted_program_row(outline_event=task, primitive_steps=programs[outline_id],
                                         projected_snapshot=checked["projected_snapshot"],
                                         projected_outline_state=checked["projected_outline_state"])
        session["accepted_primitive_program"] = [*session["accepted_primitive_program"],
                                                 {**accepted, "composition_validation": evidence}]
        return evidence

    cmc.clear_result_cache()
    drive = accept("RECOVERY_SEQ1")
    route = drive["trace"][0]["capability_evidence"]["evidence"]["drive"]
    assert route["route"][-1] == pytest.approx([-4.7, 2.2, math.pi / 2]) and route["held_part"] is None
    # D2: the KMR is at the place whose dock its base reached, written into its mirrors too.
    assert drive["projected_snapshot"]["resource_location"] == "M1"
    assert drive["projected_snapshot"]["resource_core"]["current_location"] == "M1"
    accept("RECOVERY_SEQ2")
    pick = accept("RECOVERY_SEQ3")
    trace = {row["step_index"]: row["capability_evidence"]["evidence"] for row in pick["trace"]}
    assert [stop["waypoint"] for stop in trace[3]["composed_motion"]["refinement_stops"]] == [21]
    assert trace[4]["composed_motion"]["refinement_stops"][0]["joints_at_bounds"] == ["joint_a7 is at its bound 3.0541"]
    assert trace[5]["grasp"]["tcp_to_part_m"] == pytest.approx(0.025, abs=1e-4)
    # The lift leaves M1's workholding by the part's recorded support contact.
    assert trace[6]["composed_motion"]["support_departure"] is True
    # The KMR's next event starts from the joints its lift reached, holding the part by the grasp found at its TCP.
    assert pick["resource_evidence"]["snapshot"]["joint_positions"] == trace[6]["composed_motion"]["end_joints"]
    assert pick["resource_evidence"]["attachment"]["grasp_transform"] == \
        trace[5]["grasp"]["attachment"]["grasp_transform"]
    assert pick["projected_part_context"]["part_location"] == events["RECOVERY_SEQ3"]["resource_jid"]


def test_an_unchanged_request_is_answered_from_the_result_cache(machining, machining_seq8_start):
    cmc.clear_result_cache()
    moves = _moves(_rows(MACHINING)["RECOVERY_SEQ8"], {1, 2, 3})
    first = cmc.check_robot_moves(machining, machining_seq8_start, "ur5e-2", moves)
    again = cmc.check_robot_moves(machining, machining_seq8_start, "ur5e-2", moves)
    assert first["requests"][0]["cached"] is False and again["requests"][0]["cached"] is True
    assert again["end_joints"] == first["end_joints"]


def test_a_cylinder_part_leaves_its_support_by_its_exact_surface(machining):
    rows = _rows(MACHINING)
    replay = _Replay(machining)
    replay.run([rows["RECOVERY_SEQ1"]])
    trace = rows["RECOVERY_SEQ3"]["composition_validation"]["trace"]
    replay.robot_event("ur5e-2", {**rows["RECOVERY_SEQ3"], "composition_validation": {
        "trace": [step for step in trace if step["step_index"] < 5]}})
    record = cmc.support_departure_record(machining, replay.state, "ur5e-2")
    pair = record["pairs"][0]
    # The cylinder's own surface bounds its recorded penetration of the workholding, as the native check measures it.
    assert pair["support_object_id"] == "M2/workholding/workholding_collision"
    assert set(pair["part_cylinder"]) == {"center", "axis", "radius", "length"} and "part_vertices" not in pair
    assert 0 < pair["initial_penetration_bound"] < 1e-5
    lift = cmc.check_robot_moves(machining, replay.state, "ur5e-2", _moves(rows["RECOVERY_SEQ3"], {5}),
                                 support_departure=record)
    assert lift["status"] == "FEASIBLE", lift["reason"]


def test_a_recorded_insertion_corridor_binds_the_gear_insertion(part_slippage, part_slippage_seq2_start):
    rows = _rows(PART_SLIPPAGE)
    trace = rows["RECOVERY_SEQ4"]["composition_validation"]["trace"]
    targets = trace[0]["preview_output"]
    state = copy.deepcopy(part_slippage_seq2_start)
    state["resources"]["ur5e-4"] = {"joint_positions": part_slippage.arms["ur5e-4"]["named_joint_poses"]["home"]["positions"]}
    state["resources"]["ur5e-3"].update(held_part="gear_small", gripper_state="closed",
                                        grasp_transform=targets["grasp_transform"])
    state["parts"].pop("gear_small", None)
    approach = cmc.check_robot_moves(part_slippage, state, "ur5e-3", _moves(rows["RECOVERY_SEQ4"], {1}))
    assert approach["status"] == "FEASIBLE", approach["reason"]
    state["resources"]["ur5e-3"]["joint_positions"] = approach["end_joints"]
    owner = rows["RECOVERY_SEQ2"]["composition_validation"]["resource_evidence"]
    insert = [targets["insert_pose"][key] for key in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    record = cmc.mating_contact_record(part_slippage, state, "ur5e-3", owner, part_name="gear_small",
                                       target_origin_pose=targets["target_origin_pose"], target_tool_pose=insert)
    assert record["simulation_mating_contact"]["target_collision_object"] == "Gear_Plate/Gear_Shaft_1/collision"
    inserted = cmc.check_robot_moves(part_slippage, state, "ur5e-3", _moves(rows["RECOVERY_SEQ4"], {2}), mating=record)
    assert inserted["status"] == "FEASIBLE", inserted["reason"]
    assert inserted["requests"][0]["mating"] is True
    with pytest.raises(cmc.MissingMotionContext, match="records no insertion corridor for KET4_Square_4mm"):
        cmc.mating_contact_record(part_slippage, state, "ur5e-3", owner, part_name="KET4_Square_4mm",
                                  target_origin_pose=targets["target_origin_pose"], target_tool_pose=insert)


def _composition_step(model: cmc.CapturedMotionModel, state: dict, rid: str, row: dict):
    """A robot's composition step check standing in the engine state ``state``, as its evidence carries it.

    The robot's own evidence carries its joints, the part it holds and the parts programs put down; the other
    robots stand in the grounding context as the accepted prefix leaves them.
    """
    from cais_spade_llm.resources.robot.recovery_composition import RobotEventScene

    snapshot = cmc.projected_snapshot(model, state)
    jids = model.observations["jids"]

    def evidence_of(robot: str) -> dict:
        current = snapshot["resources"][robot]
        held = current.get("held_part")
        return {"snapshot": {"joint_positions": list(current["joint_positions"])},
                "attachment": {"part_name": held, "grasp_transform": list(current["grasp_transform"])} if held else {},
                "observed_parts": {part: {"current_pose": row["current_pose"], "source": {"kind": "projected"}}
                                   for part, row in state["parts"].items()}}

    others = {jids[robot]: {"held_part": snapshot["resources"][robot].get("held_part"),
                            "gripper_state": snapshot["resources"][robot].get("gripper_state"),
                            "resource_evidence": evidence_of(robot)}
              for robot in model.arms if robot != rid}
    scene = RobotEventScene(task={"resource_jid": jids[rid], "part_name": row.get("part_name")},
                            program=row["primitive_steps"], resource_id=rid, grounding_context={"resources": others},
                            physical_evidence=model.physical, recovery_des_model={"events": []},
                            motion_model=lambda: model)
    declared = {"held_part": snapshot["resources"][rid].get("held_part"),
                "gripper_state": snapshot["resources"][rid].get("gripper_state")}
    return scene, evidence_of(rid), declared


def test_a_straight_move_past_the_base_axis_is_refused_at_its_step(machining, machining_seq8_start):
    """The composition's step check flies the straight move MoveIt cannot follow near ur5e-2's base axis."""
    from cais_spade_llm.resources.robot.recovery_composition import check_robot_step

    cmc.clear_result_cache()
    row = _rows(MACHINING)["RECOVERY_SEQ8"]
    scene, evidence, declared = _composition_step(machining, machining_seq8_start, "ur5e-2", row)
    straight = _moves(row, {3})[0]
    result = check_robot_step(evidence, {}, scene, index=3, primitive="move_cartesian", params=straight["params"],
                              recovery_snapshot=declared, checked={"feasibility_status": "FEASIBLE", "allowed": True})
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["reason"].startswith("move_cartesian step 3 (ur5e-2): MoveIt's Cartesian path on the captured model "
                                       "stopped 35% of the way along waypoint 1 of 1")
    assert "the IK search found no joint solution there" in result["reason"]


def test_each_routed_step_starts_where_the_last_one_left_the_arm(machining, machining_seq8_start):
    """Step by step, the composition carries the joints MoveIt reached into the robot's evidence."""
    from cais_spade_llm.resources.robot.recovery_composition import advance_robot_program, check_robot_step

    row = _rows(MACHINING)["RECOVERY_SEQ8"]
    scene, evidence, declared = _composition_step(machining, machining_seq8_start, "ur5e-2", row)
    for move in _moves(row, {1, 2, 3}):
        checked = check_robot_step(evidence, {}, scene, index=move["step_index"], primitive=move["primitive"],
                                   params=move["params"], recovery_snapshot=declared,
                                   checked={"feasibility_status": "FEASIBLE", "allowed": True})
        assert checked["feasibility_status"] == "FEASIBLE", checked["reason"]
        motion = checked["evidence"]["composed_motion"]
        assert motion["requests"][0]["status"] == "PREVIEW_COMPLETE" and not motion["allowed_contacts"]
        advance_robot_program(evidence, {}, task=scene.task, primitive=move["primitive"], params=move["params"],
                              recovery_snapshot=declared, checked=checked)
        assert evidence["snapshot"]["joint_positions"] == motion["end_joints"]
        assert motion["end_pose"][:3] == pytest.approx([move["params"][axis] for axis in "xyz"], abs=1e-4)
    # The descent before the release ends with the part on the M2 workholding: that pair alone is allowed.
    descent = _moves(row, {4})[0]
    checked = check_robot_step(evidence, {}, scene, index=4, primitive="move_cartesian", params=descent["params"],
                               recovery_snapshot=declared, checked={"feasibility_status": "FEASIBLE", "allowed": True})
    assert checked["feasibility_status"] == "FEASIBLE", checked["reason"]
    assert checked["evidence"]["composed_motion"]["allowed_contacts"][0]["pairs"] == [
        ["KET4_Square_4mm/link/collision", "M2/workholding/workholding_collision"]]


PART_SLIPPAGE_COMPOSITION = COMPOSITIONS[PART_SLIPPAGE]


def test_a_composed_program_is_checked_by_moveit_from_the_owner(part_slippage):
    """Part slippage v7b through the robot owner's composition check: SEQ1 rests KET4 in buffer zone 1, and SEQ2's
    route meets the stopped ur5e-4."""
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
        _accepted_program_row,
        _validate_single_event_primitive_steps,
    )
    from cais_spade_llm.recovery_framework.offline_recovery import SavedRecoveryResource
    from cais_spade_llm.recovery_framework.recovery_observations import read_observations

    folder = FIXTURES / PART_SLIPPAGE_COMPOSITION
    source = json.loads((folder / "outline_request.json").read_text())
    composition = json.loads((folder / "composition.json").read_text())
    request = {key: value for key, value in source.items() if key not in {"multi_turn_session_state", "recovery_debug"}}
    session = {key: value for key, value in source["multi_turn_session_state"].items()
               if key not in {"turns", "projected_physical_evidence", "outline_monitor_context"}}
    session.update(composition_validation_version=1, current_phase="primitive_generation", accepted_primitive_program=[])
    request.update(composition_validation_version=1, multi_turn_session_state=session)
    observations = read_observations(RECORDS / PART_SLIPPAGE / "observations.json")
    owners = {jid: SavedRecoveryResource(jid, entry, observations, _copy_observations=False)
              for jid, entry in observations["prepared_recovery_request"]["recovery_resources"].items()}
    programs = {row["outline_id"]: _bound_to_model(row["primitive_steps"], row["resource_jid"], request)
                for row in composition["accepted_primitive_program"]}
    first, second = session["accepted_outline_prefix"][:2]
    cmc.clear_result_cache()
    placed, _ = _validate_single_event_primitive_steps(session_state=session, prepared_recovery_request=request,
        outline_event=first, primitive_steps=programs["RECOVERY_SEQ1"], composition_owner=owners[first["resource_jid"]])
    evidence = placed["composition_validation"]
    assert evidence["feasibility_status"] == "FEASIBLE", evidence.get("reason")
    trace = {row["step_index"]: row for row in evidence["trace"]}
    # The descent before the release ends with KET4 on the zone's belt: that one pair is allowed for that one motion.
    assert trace[2]["capability_evidence"]["evidence"]["composed_motion"]["allowed_contacts"][0]["pairs"] == [
        ["KET4_Square_4mm/link/collision", "Buffer For Machined parts/link/zone_1_belt"]]
    release = trace[3]["capability_evidence"]["evidence"]["release"]
    assert (release["place"], release["area"]) == ("Buffer For Machined parts", "Buffer For Machined parts.zone_1")
    assert evidence["projected_part_context"]["part_location"] == "Buffer For Machined parts"
    # The robot's next event starts from the joints its last motion reached.
    assert evidence["resource_evidence"]["snapshot"]["joint_positions"] == \
        trace[4]["capability_evidence"]["evidence"]["composed_motion"]["end_joints"]
    accepted = _accepted_program_row(outline_event=first, primitive_steps=programs["RECOVERY_SEQ1"],
                                     projected_snapshot=placed["projected_snapshot"],
                                     projected_outline_state=placed["projected_outline_state"])
    session["accepted_primitive_program"] = [{**accepted, "composition_validation": evidence}]
    refused, feedback = _validate_single_event_primitive_steps(session_state=session, prepared_recovery_request=request,
        outline_event=second, primitive_steps=programs["RECOVERY_SEQ2"], composition_owner=owners[second["resource_jid"]])
    finding = refused["composition_validation"]["findings"][0]
    assert refused["composition_validation"]["feasibility_status"] == "INFEASIBLE"
    assert finding["step_index"] == 2
    assert "ur5e_3_rg2_onrobot_base_link <> ur5e_4_wrist_1_link" in finding["reason"]
    assert feedback[0]["reason"] == finding["reason"]

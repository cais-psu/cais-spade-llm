"""Tests for strict no-motion Cartesian MoveIt validation."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from cais_spade_llm.spec2primitives.adapters import moveit_plan_only


def _grounded_targets() -> dict[str, object]:
    return {
        "pick_object_center_m": [0.4, -0.3, 1.06],
        "pick_support_point_m": [0.4, -0.3, 1.04],
        "pick_surface_normal": [0.0, 0.0, 1.0],
        "place_support_point_m": [0.0, 0.144, 1.04],
        "place_surface_normal": [0.0, 0.0, 1.0],
        "place_object_center_m": [0.0, 0.144, 1.05],
        "part_dimensions_m": [0.042, 0.042, 0.02],
        "support_dimensions_m": [0.01, 0.01, 0.02],
        "part_height_m": 0.02,
        "motion_offsets": {
            "pick_approach_height_m": 0.06,
            "pick_surface_clearance_m": 0.002,
            "pick_tcp_z_bias_min_m": 0.004,
            "pick_tcp_z_bias_max_m": 0.008,
            "transfer_clearance_m": 0.10,
            "place_approach_height_m": 0.04,
        },
    }


def _request() -> dict[str, object]:
    return {
        "process_symbol": "assembly",
        "process_iri": "https://example.local/process/assembly",
        "feature_iri": "https://example.local/feature/1",
        "resource_symbol": "xarm6",
        "resource_iri": "https://example.local/resource/xarm6",
        "resource_jid": "xarm6@localhost",
        "execution_mode": "simulation",
        "motion_mode": "cartesian_pick_place",
        "moveit_group": "xarm6",
        "end_effector_link": "xarm6_link_eef",
        "tcp_link": "xarm6_link_tcp",
        "target_frame": "world",
        "cartesian_path_service": "/compute_cartesian_path",
        "validation_scope": "cartesian_pick_place",
        "checked_constraints": [
            "live_tf",
            "collision_aware_cartesian_pick_path",
            "collision_aware_cartesian_transfer_place_path",
            "complete_path_fraction",
        ],
        "unvalidated_constraints": [
            "grasp_contact",
            "gripper_actuation",
            "attached_part_collision_geometry",
            "assembly_tolerance",
            "force_control",
            "final_constrained_insertion_stroke",
        ],
        "cartesian_parameters": {
            "max_step_m": 0.01,
            "jump_threshold": 0.0,
            "avoid_collisions": True,
            "minimum_fraction": 0.999,
        },
        "current_state": {
            "state_iri": "https://example.local/state/current",
            "evidence_handle": "state_candidate_1",
            "translation_m": [0.4, -0.3, 1.06],
            "location_record_ref": "current.json",
            "location_record_sha256": "0" * 64,
        },
        "desired_state": {
            "state_iri": "https://example.local/state/desired",
            "evidence_handle": "state_candidate_2",
            "translation_m": [0.0, 0.144, 1.06],
            "location_record_ref": "desired.json",
            "location_record_sha256": "1" * 64,
        },
        "grounded_targets": _grounded_targets(),
        "mode": "plan_only",
        "motion_executed": False,
        "request_fingerprint": "2" * 64,
    }


def _live_start_pose() -> dict[str, object]:
    return {
        "frame_id": "world",
        "link_name": "xarm6_link_eef",
        "position_m": [0.0, -0.5, 1.3],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def _ee_to_tcp() -> dict[str, object]:
    return {
        "parent_link": "xarm6_link_eef",
        "child_link": "xarm6_link_tcp",
        "translation_m": [0.0, 0.0, -0.17],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def test_waypoints_use_grounded_gear_centers_and_live_tcp_transform() -> None:
    request = _request()
    waypoints = moveit_plan_only._cartesian_waypoints(
        request,
        live_start_pose=_live_start_pose(),
        ee_to_tcp=_ee_to_tcp(),
    )

    assert [(item["phase"], item["role"]) for item in waypoints] == [
        ("pick", "pick_approach"),
        ("pick", "grasp"),
        ("pick", "pick_retreat"),
        ("place", "transfer"),
        ("place", "place_approach"),
        ("place", "placement"),
        ("place", "place_retreat"),
    ]
    placement = next(item for item in waypoints if item["role"] == "placement")
    placement_position = placement["pose"]["position_m"]
    assert placement_position == pytest.approx([0.0, 0.144, 1.227])
    assert placement_position != request["desired_state"]["translation_m"]
    assert all(
        item["pose"]["orientation_xyzw"] == _live_start_pose()["orientation_xyzw"]
        for item in waypoints
    )


def test_cartesian_phases_are_strict_and_second_phase_uses_pick_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = moveit_plan_only._validated_request(_request())
    waypoints = moveit_plan_only._cartesian_waypoints(
        request,
        live_start_pose=_live_start_pose(),
        ee_to_tcp=_ee_to_tcp(),
    )
    pick_terminal = object()
    place_terminal = object()
    responses = [
        SimpleNamespace(
            fraction=1.0,
            error_code=SimpleNamespace(val=1),
            solution=SimpleNamespace(terminal_state=pick_terminal),
        ),
        SimpleNamespace(
            fraction=0.999,
            error_code=SimpleNamespace(val=1),
            solution=SimpleNamespace(terminal_state=place_terminal),
        ),
    ]

    class Future:
        def __init__(self, response: object) -> None:
            self._response = response

        def done(self) -> bool:
            return True

        def result(self) -> object:
            return self._response

    class Client:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def call_async(self, service_request: object) -> Future:
            self.requests.append(service_request)
            return Future(responses[len(self.requests) - 1])

    class Request:
        def __init__(self) -> None:
            self.header = SimpleNamespace(frame_id=None, stamp=None)
            self.start_state = SimpleNamespace(is_diff=False)

    service_type = SimpleNamespace(Request=Request)

    node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp"))
    )
    rclpy = SimpleNamespace(ok=lambda: True, spin_once=lambda *_args, **_kwargs: None)
    client = Client()
    monkeypatch.setattr(moveit_plan_only, "_geometry_pose", lambda value: value)
    monkeypatch.setattr(
        moveit_plan_only,
        "_trajectory_terminal_state",
        lambda solution: solution.terminal_state,
    )

    pick, returned_pick_terminal = moveit_plan_only._plan_cartesian_phase(
        rclpy,
        node,
        client,
        service_type,
        request,
        phase="pick",
        waypoints=waypoints,
        start_state=None,
        timeout_sec=1.0,
    )
    place, returned_place_terminal = moveit_plan_only._plan_cartesian_phase(
        rclpy,
        node,
        client,
        service_type,
        request,
        phase="place",
        waypoints=waypoints,
        start_state=returned_pick_terminal,
        timeout_sec=1.0,
    )

    assert pick["status"] == "accepted"
    assert place["status"] == "accepted"
    assert returned_pick_terminal is pick_terminal
    assert returned_place_terminal is place_terminal
    assert client.requests[0].start_state.is_diff is True
    assert client.requests[1].start_state is pick_terminal
    assert [len(item.waypoints) for item in client.requests] == [3, 4]
    for service_request in client.requests:
        assert service_request.header.frame_id == "world"
        assert service_request.group_name == "xarm6"
        assert service_request.link_name == "xarm6_link_eef"
        assert service_request.max_step == pytest.approx(0.01)
        assert service_request.jump_threshold == pytest.approx(0.0)
        assert service_request.avoid_collisions is True


@pytest.mark.parametrize(
    ("fraction", "error_code"),
    [(0.998, 1), (1.0, -1)],
)
def test_partial_or_moveit_rejected_path_is_never_accepted(
    monkeypatch: pytest.MonkeyPatch,
    fraction: float,
    error_code: int,
) -> None:
    request = moveit_plan_only._validated_request(_request())
    waypoints = moveit_plan_only._cartesian_waypoints(
        request,
        live_start_pose=_live_start_pose(),
        ee_to_tcp=_ee_to_tcp(),
    )
    response = SimpleNamespace(
        fraction=fraction,
        error_code=SimpleNamespace(val=error_code),
        solution=SimpleNamespace(terminal_state=object()),
    )
    future = SimpleNamespace(done=lambda: True, result=lambda: response)
    client = SimpleNamespace(call_async=lambda _request: future)
    node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp"))
    )
    rclpy = SimpleNamespace(ok=lambda: True, spin_once=lambda *_args, **_kwargs: None)

    class Request:
        def __init__(self) -> None:
            self.header = SimpleNamespace(frame_id=None, stamp=None)
            self.start_state = SimpleNamespace(is_diff=False)

    service_type = SimpleNamespace(Request=Request)

    monkeypatch.setattr(moveit_plan_only, "_geometry_pose", lambda value: value)
    monkeypatch.setattr(
        moveit_plan_only,
        "_trajectory_terminal_state",
        lambda solution: solution.terminal_state,
    )

    result, terminal = moveit_plan_only._plan_cartesian_phase(
        rclpy,
        node,
        client,
        service_type,
        request,
        phase="pick",
        waypoints=waypoints,
        start_state=None,
        timeout_sec=1.0,
    )

    assert result["status"] == "rejected"
    assert terminal is None


def test_runtime_contains_no_trajectory_execution_client() -> None:
    source = inspect.getsource(moveit_plan_only)

    assert "ExecuteTrajectory" not in source
    assert "execute_trajectory" not in source


def test_live_tf_timeout_returns_no_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter((0.0, 0.01, 0.02, 0.11))
    spins: list[float] = []
    monkeypatch.setattr(moveit_plan_only.time, "monotonic", lambda: next(times))
    rclpy = SimpleNamespace(
        ok=lambda: True,
        time=SimpleNamespace(Time=lambda: object()),
        spin_once=lambda _node, *, timeout_sec: spins.append(timeout_sec),
    )
    tf_buffer = SimpleNamespace(
        lookup_transform=lambda *_args: (_ for _ in ()).throw(RuntimeError("missing TF"))
    )

    result = moveit_plan_only._lookup_live_transform(
        rclpy,
        object(),
        tf_buffer,
        target_frame="world",
        source_frame="xarm6_link_eef",
        timeout_sec=0.1,
    )

    assert result is None
    assert spins == [pytest.approx(0.08)]


def test_cartesian_service_timeout_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = moveit_plan_only._validated_request(_request())
    waypoints = moveit_plan_only._cartesian_waypoints(
        request,
        live_start_pose=_live_start_pose(),
        ee_to_tcp=_ee_to_tcp(),
    )

    class Request:
        def __init__(self) -> None:
            self.header = SimpleNamespace(frame_id=None, stamp=None)
            self.start_state = SimpleNamespace(is_diff=False)

    service_type = SimpleNamespace(Request=Request)
    node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp"))
    )
    client = SimpleNamespace(call_async=lambda _request: object())
    monkeypatch.setattr(moveit_plan_only, "_geometry_pose", lambda value: value)
    monkeypatch.setattr(moveit_plan_only, "_wait_future", lambda *_args, **_kwargs: False)

    result, terminal = moveit_plan_only._plan_cartesian_phase(
        object(),
        node,
        client,
        service_type,
        request,
        phase="pick",
        waypoints=waypoints,
        start_state=None,
        timeout_sec=0.1,
    )

    assert result["status"] == "needs_context"
    assert result["fraction"] is None
    assert terminal is None

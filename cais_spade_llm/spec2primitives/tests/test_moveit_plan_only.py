from __future__ import annotations

"""Tests for strict no-motion Cartesian MoveIt validation."""


import inspect
import sys
from types import SimpleNamespace

import pytest

from cais_spade_llm.spec2primitives.adapters import moveit_plan_only


def _location_request():
    return {
        "process_symbol": "assembly",
        "process_iri": "https://example.local/process/assembly",
        "feature_iri": "https://example.local/feature/1",
        "resource_symbol": "xarm6",
        "resource_iri": "https://example.local/resource/xarm6",
        "resource_jid": "xarm6@localhost",
        "execution_mode": "simulation",
        "moveit_group": "xarm6",
        "end_effector_link": "xarm6_link_eef",
        "target_frame": "world",
        "mode": "plan_only",
        "motion_executed": False,
        "motion_plan_service": "/plan_kinematic_path",
        "position_tolerance_m": 0.005,
        "validation_scope": "moveit_state_location_reachability",
        "state_locations": {
            state: [
                {
                    "state_iri": f"https://example.local/state/{state}",
                    "evidence_handle": f"state_candidate_{index}",
                    "translation_m": point,
                    "location_record_ref": f"{state}.json",
                    "location_record_sha256": str(index) * 64,
                }
            ]
            for index, (state, point) in enumerate(
                (("current_state", [0.4, -0.3, 1.06]), ("desired_state", [0.0, 0.144, 1.06])),
                start=1,
            )
        },
    }


def test_location_goal_uses_live_state_and_position_only_constraints(monkeypatch):
    class Pose:
        def __init__(self):
            self.position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.orientation = SimpleNamespace(w=0.0)

    class PositionConstraint:
        def __init__(self):
            self.header = SimpleNamespace(frame_id="")
            self.constraint_region = SimpleNamespace(primitives=[], primitive_poses=[])

    class Constraints:
        def __init__(self):
            self.position_constraints = []
            self.orientation_constraints = []

    class Primitive:
        SPHERE = 2

    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", SimpleNamespace(Pose=Pose))
    monkeypatch.setitem(
        sys.modules,
        "moveit_msgs.msg",
        SimpleNamespace(Constraints=Constraints, PositionConstraint=PositionConstraint),
    )
    monkeypatch.setitem(sys.modules, "shape_msgs.msg", SimpleNamespace(SolidPrimitive=Primitive))
    service = SimpleNamespace(
        Request=lambda: SimpleNamespace(
            motion_plan_request=SimpleNamespace(start_state=SimpleNamespace(is_diff=False))
        )
    )
    request = _location_request()
    goal = moveit_plan_only._location_motion_plan_request(
        service, request, request["state_locations"]["desired_state"][0]
    ).motion_plan_request

    assert goal.group_name == request["moveit_group"]
    assert goal.start_state.is_diff is True
    assert goal.goal_constraints[0].orientation_constraints == []
    position = goal.goal_constraints[0].position_constraints[0]
    assert position.link_name == request["end_effector_link"]
    assert position.header.frame_id == "world"
    assert position.constraint_region.primitive_poses[0].position.y == 0.144
    assert position.constraint_region.primitives[0].dimensions == [0.005]


@pytest.mark.parametrize("available,code", [(True, 1), (True, -1), (False, 1)])
def test_location_runtime_queries_moveit_without_static_fallback(monkeypatch, available, code):
    from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
        validate_location_planning_response,
    )

    request = _location_request()
    calls = []
    response = SimpleNamespace(
        group_name=request["moveit_group"],
        error_code=SimpleNamespace(val=code),
        trajectory=SimpleNamespace(
            joint_trajectory=SimpleNamespace(
                joint_names=["joint"], points=[SimpleNamespace(positions=[0.2])]
            )
        ),
        trajectory_start=SimpleNamespace(
            joint_state=SimpleNamespace(name=["joint"], position=[0.0])
        ),
    )

    def call(goal):
        calls.append(goal)
        return SimpleNamespace(
            done=lambda: True, result=lambda: SimpleNamespace(motion_plan_response=response)
        )

    client = SimpleNamespace(wait_for_service=lambda **_: available, call_async=call)

    class Node:
        def __init__(self, name):
            self.name = name

        def create_client(self, service_type, service_name):
            assert service_name == "/plan_kinematic_path"
            return client

        def destroy_node(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "rclpy",
        SimpleNamespace(ok=lambda: True, spin_until_future_complete=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(sys.modules, "rclpy.node", SimpleNamespace(Node=Node))
    monkeypatch.setitem(sys.modules, "moveit_msgs.srv", SimpleNamespace(GetMotionPlan=object()))
    monkeypatch.setattr(
        moveit_plan_only,
        "_location_motion_plan_request",
        lambda service, req, location: dict(location),
    )
    result = moveit_plan_only.MoveItPlanOnlyRuntime()._validate_state_locations_sync(request)
    validate_location_planning_response(request, result)
    assert result["status"] == (
        "needs_context" if not available else "accepted" if code == 1 else "rejected"
    )
    assert len(calls) == (2 if available else 0)
    assert all("workspace_bounds" not in call for call in calls)


def test_runtime_contains_no_trajectory_execution_client() -> None:
    source = inspect.getsource(moveit_plan_only)

    assert "ExecuteTrajectory" not in source
    assert "execute_trajectory" not in source

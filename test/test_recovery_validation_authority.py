"""Focused tests for PA/RA/CCA recovery-validation authority boundaries."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from cais_spade_llm.agents.central_controller.outline_macro_safety import (
    validate_outline_macro_recovery_safety,
)
from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
    multi_turn_outline_generation,
    multi_turn_prompts,
)
from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent
from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    RECOVERY_OUTLINE_PHYSICAL_VALIDATE,
    RECOVERY_OUTLINE_PHYSICAL_VALIDATED,
    recovery_validation_fingerprint,
    recovery_validation_reply_matches,
)
from cais_spade_llm.resources.resource_primitives import (
    get_resource_recovery_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("robot", ["xarm6", "ur5e"])
def test_robot_manifest_exposes_manipulator_pick_place_capability(robot: str) -> None:
    payload = json.loads(
        (
            ROOT
            / "cais_spade_llm"
            / "initialization"
            / "resources"
            / f"robot_{robot}.json"
        ).read_text(encoding="utf-8")
    )

    for execution_environment in ("gazebo", "real"):
        static_capabilities = payload[robot][execution_environment][
            "static_capabilities"
        ]
        assert static_capabilities["supports_manipulator_pick_place"] is True


def _product_transport_double() -> SimpleNamespace:
    product = SimpleNamespace(
        jid="product@localhost",
        cca_jid="cca@localhost",
        logger=logging.getLogger("test.recovery_validation_authority"),
        _recovery_outline_validation_waiters={},
        _recovery_outline_validation_batches={},
    )
    product._bare_jid = ProductAgent._bare_jid
    product._request_recovery_outline_validation = (
        ProductAgent._request_recovery_outline_validation.__get__(product)
    )
    product._flush_recovery_outline_validation_batch = (
        ProductAgent._flush_recovery_outline_validation_batch.__get__(product)
    )
    return product


def _correlated_payload(*, candidate_index: int = 0) -> dict[str, Any]:
    return {
        "recovery_session_id": "session-1",
        "turn_index": 4,
        "state_fingerprint": "pa-fingerprint",
        "candidate_task": {"resource_jid": "ra@localhost"},
        "grounded_action": {"resource_jid": "ra@localhost"},
        "candidates": [
            {
                "candidate_index": candidate_index,
                "task": {"resource_jid": "ra@localhost"},
                "physical_input": {},
            }
        ],
    }


def test_reply_correlation_requires_request_session_turn_and_fingerprint() -> None:
    payload = {
        "request_id": "request-1",
        "recovery_session_id": "session-1",
        "turn_index": 4,
        "state_fingerprint": "pa-fingerprint",
    }
    assert recovery_validation_reply_matches(
        payload,
        request_id="request-1",
        recovery_session_id="session-1",
        turn_index=4,
        state_fingerprint="pa-fingerprint",
    )
    for field, value in (
        ("request_id", "stale-request"),
        ("recovery_session_id", "stale-session"),
        ("turn_index", 3),
        ("state_fingerprint", "stale-fingerprint"),
    ):
        stale = deepcopy(payload)
        stale[field] = value
        assert not recovery_validation_reply_matches(
            stale,
            request_id="request-1",
            recovery_session_id="session-1",
            turn_index=4,
            state_fingerprint="pa-fingerprint",
        )


def test_product_transport_accepts_only_correlated_expected_sender_reply() -> None:
    async def _run() -> None:
        product = _product_transport_double()
        loop = asyncio.get_running_loop()

        def _send(_agent: Any, message: Any, **_kwargs: Any) -> str:
            request = json.loads(message.body)
            reply = {
                **request,
                "validator_jid": "ra@localhost",
                "results": [],
            }
            loop.call_soon(
                lambda: ProductAgent._resolve_recovery_outline_validation_reply(
                    product,
                    response_type=RECOVERY_OUTLINE_PHYSICAL_VALIDATED,
                    sender="ra@localhost",
                    payload=reply,
                )
            )
            return "mocked"

        with patch(
            "cais_spade_llm.agents.intelligent_product.product_agent.send_agent_message_sync",
            side_effect=_send,
        ):
            reply = await ProductAgent._request_recovery_outline_validation(
                product,
                target_jid="ra@localhost",
                request_type=RECOVERY_OUTLINE_PHYSICAL_VALIDATE,
                response_type=RECOVERY_OUTLINE_PHYSICAL_VALIDATED,
                payload=_correlated_payload(),
                timeout_s=0.2,
            )
        assert reply["validator_jid"] == "ra@localhost"

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("sender", "reply_update"),
    [
        ("wrong-ra@localhost", {}),
        ("ra@localhost", {"recovery_session_id": "stale-session"}),
        ("ra@localhost", {"turn_index": 3}),
        ("ra@localhost", {"state_fingerprint": "stale-fingerprint"}),
    ],
)
def test_wrong_sender_and_stale_reply_fail_closed(
    sender: str,
    reply_update: dict[str, Any],
) -> None:
    async def _run() -> None:
        product = _product_transport_double()
        loop = asyncio.get_running_loop()

        def _send(_agent: Any, message: Any, **_kwargs: Any) -> str:
            reply = json.loads(message.body)
            reply.update(reply_update)
            loop.call_soon(
                lambda: ProductAgent._resolve_recovery_outline_validation_reply(
                    product,
                    response_type=RECOVERY_OUTLINE_PHYSICAL_VALIDATED,
                    sender=sender,
                    payload=reply,
                )
            )
            return "mocked"

        with patch(
            "cais_spade_llm.agents.intelligent_product.product_agent.send_agent_message_sync",
            side_effect=_send,
        ), pytest.raises(RuntimeError):
            await ProductAgent._request_recovery_outline_validation(
                product,
                target_jid="ra@localhost",
                request_type=RECOVERY_OUTLINE_PHYSICAL_VALIDATE,
                response_type=RECOVERY_OUTLINE_PHYSICAL_VALIDATED,
                payload=_correlated_payload(),
                timeout_s=0.05,
            )

    asyncio.run(_run())


def test_timeout_and_malformed_reply_fail_closed() -> None:
    async def _timeout() -> None:
        product = _product_transport_double()
        with patch(
            "cais_spade_llm.agents.intelligent_product.product_agent.send_agent_message_sync",
            return_value="mocked",
        ), pytest.raises(RuntimeError, match="timed out"):
            await ProductAgent._request_recovery_outline_validation(
                product,
                target_jid="ra@localhost",
                request_type=RECOVERY_OUTLINE_PHYSICAL_VALIDATE,
                response_type=RECOVERY_OUTLINE_PHYSICAL_VALIDATED,
                payload=_correlated_payload(),
                timeout_s=0.01,
            )

    async def _malformed() -> None:
        product = _product_transport_double()
        loop = asyncio.get_running_loop()

        def _send(_agent: Any, message: Any, **_kwargs: Any) -> str:
            request = json.loads(message.body)
            loop.call_soon(
                lambda: ProductAgent._resolve_recovery_outline_validation_reply(
                    product,
                    response_type=RECOVERY_OUTLINE_PHYSICAL_VALIDATED,
                    sender="ra@localhost",
                    payload={"request_id": request["request_id"]},
                )
            )
            return "mocked"

        with patch(
            "cais_spade_llm.agents.intelligent_product.product_agent.send_agent_message_sync",
            side_effect=_send,
        ), pytest.raises(RuntimeError, match="did not match"):
            await ProductAgent._request_recovery_outline_validation(
                product,
                target_jid="ra@localhost",
                request_type=RECOVERY_OUTLINE_PHYSICAL_VALIDATE,
                response_type=RECOVERY_OUTLINE_PHYSICAL_VALIDATED,
                payload=_correlated_payload(),
                timeout_s=0.05,
            )

    asyncio.run(_timeout())
    asyncio.run(_malformed())


def test_same_resource_candidate_requests_are_grouped_into_one_ra_batch() -> None:
    async def _run() -> None:
        product = _product_transport_double()
        direct_requests: list[dict[str, Any]] = []

        async def _direct_request(**kwargs: Any) -> dict[str, Any]:
            payload = deepcopy(kwargs["payload"])
            direct_requests.append(payload)
            return {
                "request_id": "grouped-request",
                "recovery_session_id": payload["recovery_session_id"],
                "turn_index": payload["turn_index"],
                "state_fingerprint": payload["state_fingerprint"],
                "results": [
                    {
                        "candidate_index": row["candidate_index"],
                        "allowed": True,
                        "findings": [],
                    }
                    for row in payload["candidates"]
                ],
            }

        product._request_recovery_outline_validation = _direct_request
        replies = await asyncio.gather(
            *[
                ProductAgent._request_recovery_outline_validation_batched(
                    product,
                    target_jid="ra@localhost",
                    request_type=RECOVERY_OUTLINE_PHYSICAL_VALIDATE,
                    response_type=RECOVERY_OUTLINE_PHYSICAL_VALIDATED,
                    payload=_correlated_payload(candidate_index=index),
                    timeout_s=0.2,
                )
                for index in range(3)
            ]
        )
        assert len(direct_requests) == 1
        assert [
            row["candidate_index"] for row in direct_requests[0]["candidates"]
        ] == [0, 1, 2]
        assert all(len(reply["results"]) == 3 for reply in replies)

    asyncio.run(_run())


def test_resource_base_validator_and_missing_robot_capabilities_fail_closed() -> None:
    base_result = ResourceAgent.check_recovery_physical_feasibility(
        object(),
        part_context={},
        recovery_snapshot={},
        grounded_action={},
    )
    assert base_result == {
        "allowed": False,
        "constraint_code": "resource_validation_unavailable",
        "reason": "resource_validation_unavailable",
    }

    robot = SimpleNamespace(static_capabilities={})
    inside, reason = RobotAgent._is_pose_in_workspace(robot, {"x": 0.0, "y": 0.0, "z": 0.0})
    assert inside is False
    assert "workspace_bounds" in reason


def test_ra_rejects_wrong_resource_and_robot_specific_physical_conflicts() -> None:
    resource = SimpleNamespace(jid="xarm6@localhost")
    assert ResourceAgent.recovery_validation_resource_matches(
        resource, "xarm6@localhost"
    )
    assert not ResourceAgent.recovery_validation_resource_matches(
        resource, "ur5e@localhost"
    )

    missing_named_pose_data = SimpleNamespace(
        jid="xarm6@localhost",
        static_capabilities={},
    )
    result = RobotAgent.check_recovery_physical_feasibility(
        missing_named_pose_data,
        part_context={},
        recovery_snapshot={},
        grounded_action={
            "effect_scope": "resource_only",
            "task_kind": "resource_only",
            "target": {"named_pose": "home"},
            "expected_effect": {"resource": {"current_state": "clear_state"}},
        },
    )
    assert result["allowed"] is False
    assert result["constraint_code"] == "resource_validation_unavailable"

    unavailable_named_pose = SimpleNamespace(
        jid="xarm6@localhost",
        static_capabilities={"named_poses": ["home"]},
    )
    result = RobotAgent.check_recovery_physical_feasibility(
        unavailable_named_pose,
        part_context={},
        recovery_snapshot={},
        grounded_action={
            "effect_scope": "resource_only",
            "task_kind": "resource_only",
            "target": {"named_pose": "station"},
            "expected_effect": {"resource": {"current_state": "clear_state"}},
        },
    )
    assert result["allowed"] is False
    assert result["constraint_code"] == "named_pose_unavailable"

    bounded_robot = SimpleNamespace(
        jid="xarm6@localhost",
        static_capabilities={
            "workspace_bounds": {
                "x_min_m": -0.5,
                "x_max_m": 0.5,
                "y_min_m": -0.5,
                "y_max_m": 0.5,
                "z_min_m": 0.0,
                "z_max_m": 1.5,
            }
        },
    )
    bounded_robot._is_pose_in_workspace = lambda pose: RobotAgent._is_pose_in_workspace(
        bounded_robot, pose
    )
    result = RobotAgent.check_recovery_physical_feasibility(
        bounded_robot,
        part_context={},
        recovery_snapshot={},
        grounded_action={
            "effect_scope": "resource_only",
            "task_kind": "resource_only",
            "target": {"pose": {"x": 1.0, "y": 0.0, "z": 1.0}},
            "expected_effect": {"resource": {"current_state": "clear_state"}},
        },
    )
    assert result["allowed"] is False
    assert result["constraint_code"] == "workspace_unreachable"

    result = RobotAgent.check_recovery_physical_feasibility(
        bounded_robot,
        part_context={"current_holder_resource_jid": ""},
        recovery_snapshot={"held_part": "MCP", "gripper_state": "closed"},
        part_name="LG",
        grounded_action={
            "part_name": "LG",
            "effect_scope": "resource_and_part",
            "task_kind": "part_handling",
            "preconditions": {
                "part": {"requires_acquisition": True},
                "source_ref": {"location": "observed_pose", "pose": {"x": 0.0}},
            },
            "expected_effect": {"part": {"state": "secured"}},
        },
    )
    assert result["allowed"] is False
    assert result["constraint_code"] == "holder_conflict"

    result = RobotAgent.check_recovery_physical_feasibility(
        bounded_robot,
        part_context={"current_holder_resource_jid": ""},
        recovery_snapshot={"held_part": None, "gripper_state": "closed"},
        part_name="LG",
        grounded_action={
            "part_name": "LG",
            "effect_scope": "resource_and_part",
            "task_kind": "part_handling",
            "preconditions": {
                "part": {"requires_acquisition": True},
                "source_ref": {
                    "location": "observed_pose",
                    "pose": {"x": 0.0, "y": 0.0, "z": 1.0},
                },
            },
            "expected_effect": {"part": {"state": "secured"}},
        },
    )
    assert result["allowed"] is False
    assert result["constraint_code"] == "gripper_occupancy_conflict"


def _configured_recovery_robot() -> SimpleNamespace:
    robot = SimpleNamespace(
        jid="xarm6@localhost",
        static_capabilities={
            "supports_manipulator_pick_place": True,
            "reachability": ["prusa-mk4-1", "assembly_board-v1"],
            "workspace_bounds": {
                "x_min_m": -0.6,
                "x_max_m": 0.6,
                "y_min_m": -1.0,
                "y_max_m": 0.2,
                "z_min_m": 0.9,
                "z_max_m": 1.6,
            },
            "gripper_reach": {
                "frame": "world",
                "origin_pose": {"x": 0.0, "y": -0.5, "z": 1.02},
                "max_xy_radius_m": 0.8,
                "z_min_m": 0.9,
                "z_max_m": 1.6,
                "tolerance_m": 0.01,
            },
        },
        controller_config={
            "hardware_cartesian_service": "/xarm6/xarm/set_position",
            "joint_state_topics": ["/joint_states"],
            "move_group": {
                "frame_id": "world",
                "ee_link": "link_eef",
                "tcp_link": "link_tcp",
            },
            "gripper": {
                "hardware_action": "/xarm6/xarm_gripper/gripper_action",
                "open_width_mm": 85.0,
            },
            "services": {"detect_all": "/perception/xarm6/detect_all"},
        },
        executables={
            "pick_approach": lambda: None,
            "place_approach": lambda: None,
        },
        motion_config={"recovery_observed_pick_approach_height_m": 0.06},
        _controller=None,
    )
    robot._is_pose_in_workspace = lambda pose: RobotAgent._is_pose_in_workspace(
        robot,
        pose,
    )
    return robot


def test_configured_robot_recovery_snapshot_exposes_current_runtime_evidence() -> None:
    controller = SimpleNamespace(
        frame_id="world",
        ee_link="future_ee",
        tcp_link="future_tcp",
        is_usable=lambda: True,
        get_current_pose=lambda: {
            "success": True,
            "pose": {"x": 0.1, "y": -0.2, "z": 1.1},
        },
    )
    agent = SimpleNamespace(
        jid="future_robot@localhost",
        agent_name="future_robot",
        execution_mode="physical",
        static_capabilities={"resource_type": "robot"},
        controller_config={
            "move_group": {
                "frame_id": "world",
                "ee_link": "future_ee",
                "tcp_link": "future_tcp",
            },
            "services": {"detect_all": "/future_robot/detect_all"},
        },
        _controller=controller,
        _current_state="idle",
        _held_part=None,
        _gripper_state="open",
        _recovery_pose_ref=None,
        _position={"x": 0.0, "y": 0.0, "z": 1.0},
        named_positions={"home": [0.0] * 6},
        executables={"pick_approach": lambda: None},
    )

    snapshot = get_resource_recovery_snapshot(agent)

    assert snapshot["controller_ready"] is True
    assert snapshot["tf_ready"] is True
    assert snapshot["tcp_ready"] is True
    assert snapshot["perception_ready"] is True
    assert snapshot["destination_localization_ready"] is True
    assert snapshot["function_names"] == ["pick_approach"]
    assert snapshot["current_pose"] == pytest.approx(
        {"x": 0.1, "y": -0.2, "z": 1.1}
    )
    assert snapshot["current_pose_captured_at"] > 0.0


def _configured_pick_recovery_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    observed_pose = {"x": 0.4, "y": -0.3, "z": 1.04}
    part_context = {
        "current_holder_resource_jid": "",
        "observed_pose": deepcopy(observed_pose),
    }
    snapshot = {
        "held_part": None,
        "gripper_state": "open",
        "controller_ready": True,
        "tf_ready": True,
        "tcp_ready": True,
        "perception_ready": True,
        "current_pose": {"x": 0.0, "y": -0.4, "z": 1.2},
    }
    action = {
        "function_name": "pick_approach",
        "part_name": "MG",
        "effect_scope": "resource_and_part",
        "task_kind": "part_handling",
        "product_geometry": {"grasp_width_m": 0.03},
        "target": {"named_pose": "prusa-mk4-1"},
        "preconditions": {
            "part": {"requires_acquisition": True},
            "source_ref": {
                "location": "prusa-mk4-1",
                "pose": deepcopy(observed_pose),
                "captured_at": time.time(),
            },
        },
        "poses": {
            "source_pose": deepcopy(observed_pose),
            "approach_pose": {"x": 0.4, "y": -0.3, "z": 1.10},
            "target_pose": deepcopy(observed_pose),
            "retreat_pose": {"x": 0.4, "y": -0.3, "z": 1.10},
        },
        "expected_effect": {"part": {"state": "secured"}},
    }
    return part_context, snapshot, action


def test_configured_pick_recovery_uses_geometry_and_not_recording_history() -> None:
    robot = _configured_recovery_robot()
    part_context, snapshot, action = _configured_pick_recovery_inputs()

    result = RobotAgent.check_recovery_physical_feasibility(
        robot,
        part_context=part_context,
        recovery_snapshot=snapshot,
        grounded_action=action,
    )

    assert result["allowed"] is True
    assert result["evidence"]["recording_history_used"] is False
    assert set(result["evidence"]["checked_poses"]) == {
        "source_pose",
        "approach_pose",
        "target_pose",
        "retreat_pose",
    }


@pytest.mark.parametrize(
    ("mutation", "expected_status", "expected_guard"),
    [
        (
            lambda robot, _context, _snapshot, _action: robot.controller_config.pop(
                "move_group"
            ),
            "INFEASIBLE",
            "controller_behavior_unavailable",
        ),
        (
            lambda robot, _context, _snapshot, _action: robot.static_capabilities.pop(
                "supports_manipulator_pick_place"
            ),
            "INFEASIBLE",
            "manipulator_pick_place_unavailable",
        ),
        (
            lambda _robot, _context, snapshot, _action: snapshot.pop("tf_ready"),
            "NEEDS_CONTEXT",
            "runtime_behavior_evidence_unavailable",
        ),
        (
            lambda robot, _context, _snapshot, _action: robot.controller_config[
                "services"
            ].clear(),
            "INFEASIBLE",
            "perception_unavailable",
        ),
        (
            lambda _robot, _context, _snapshot, action: action.pop(
                "product_geometry"
            ),
            "NEEDS_CONTEXT",
            "product_geometry_unavailable",
        ),
        (
            lambda _robot, _context, _snapshot, action: action[
                "product_geometry"
            ].update({"grasp_width_m": 0.2}),
            "INFEASIBLE",
            "gripper_incompatible",
        ),
        (
            lambda robot, _context, _snapshot, _action: robot.static_capabilities.update(
                {"reachability": ["assembly_board-v1"]}
            ),
            "INFEASIBLE",
            "location_unreachable",
        ),
        (
            lambda _robot, context, _snapshot, action: (
                context["observed_pose"].update({"x": 2.0}),
                action["preconditions"]["source_ref"]["pose"].update({"x": 2.0}),
                action["poses"]["source_pose"].update({"x": 2.0}),
            ),
            "INFEASIBLE",
            "pose_unreachable",
        ),
        (
            lambda _robot, _context, _snapshot, action: action["preconditions"][
                "source_ref"
            ].update({"captured_at": time.time() - 30.0}),
            "NEEDS_CONTEXT",
            "pose_evidence_stale",
        ),
    ],
)
def test_configured_pick_recovery_classifies_missing_and_incompatible_evidence(
    mutation: Any,
    expected_status: str,
    expected_guard: str,
) -> None:
    robot = _configured_recovery_robot()
    part_context, snapshot, action = _configured_pick_recovery_inputs()
    mutation(robot, part_context, snapshot, action)

    result = RobotAgent.check_recovery_physical_feasibility(
        robot,
        part_context=part_context,
        recovery_snapshot=snapshot,
        grounded_action=action,
    )

    assert result["allowed"] is False
    assert result["feasibility_status"] == expected_status
    assert result["guard"]["kind"] == expected_guard


def test_configured_pick_recovery_requires_handoff_when_other_robot_holds_part() -> None:
    robot = _configured_recovery_robot()
    part_context, snapshot, action = _configured_pick_recovery_inputs()
    part_context["current_holder_resource_jid"] = "ur5e@localhost"

    result = RobotAgent.check_recovery_physical_feasibility(
        robot,
        part_context=part_context,
        recovery_snapshot=snapshot,
        grounded_action=action,
    )

    assert result["allowed"] is False
    assert result["feasibility_status"] == "INFEASIBLE"
    assert result["constraint_code"] == "holder_conflict"


def test_configured_place_recovery_requires_destination_support_and_occupancy() -> None:
    robot = _configured_recovery_robot()
    pose = {"x": 0.0, "y": -0.08, "z": 1.03}
    action = {
        "function_name": "place_approach",
        "part_name": "MG",
        "effect_scope": "resource_and_part",
        "task_kind": "part_handling",
        "product_geometry": {"grasp_width_m": 0.03},
        "target": {
            "destination_location": "assembly_board-v1",
            "destination_pose": deepcopy(pose),
            "captured_at": time.time(),
        },
        "poses": {
            "source_pose": {"x": 0.1, "y": -0.3, "z": 1.2},
            "approach_pose": {"x": 0.0, "y": -0.08, "z": 1.09},
            "target_pose": deepcopy(pose),
            "retreat_pose": {"x": 0.0, "y": -0.08, "z": 1.09},
        },
        "expected_effect": {"part": {"state": "assembled"}},
    }
    snapshot = {
        "held_part": "MG",
        "gripper_state": "closed",
        "controller_ready": True,
        "tf_ready": True,
        "tcp_ready": True,
        "destination_localization_ready": True,
    }

    missing = RobotAgent.check_recovery_physical_feasibility(
        robot,
        part_context={"current_holder_resource_jid": "xarm6@localhost"},
        recovery_snapshot=snapshot,
        grounded_action=action,
    )
    assert missing["feasibility_status"] == "NEEDS_CONTEXT"
    assert missing["guard"]["kind"] == "destination_support_unavailable"

    snapshot.update(
        {"destination_support_valid": True, "destination_occupied": True}
    )
    occupied = RobotAgent.check_recovery_physical_feasibility(
        robot,
        part_context={"current_holder_resource_jid": "xarm6@localhost"},
        recovery_snapshot=snapshot,
        grounded_action=action,
    )
    assert occupied["feasibility_status"] == "INFEASIBLE"
    assert occupied["guard"]["kind"] == "destination_occupied"

    snapshot["destination_occupied"] = False
    allowed = RobotAgent.check_recovery_physical_feasibility(
        robot,
        part_context={"current_holder_resource_jid": "xarm6@localhost"},
        recovery_snapshot=snapshot,
        grounded_action=action,
    )
    assert allowed["allowed"] is True


def test_ra_uses_fresh_field_specific_location_domains() -> None:
    class _Owner:
        jid = "ur5e@localhost"
        static_capabilities = {"resource_type": "robot"}

        def __init__(self) -> None:
            self.snapshot_calls = 0
            self.model_calls = 0
            self.physical_calls = 0
            self.part_location_domain = [
                None,
                "prusa-mk4-2",
                "ur5e@localhost",
            ]

        def get_recovery_snapshot(self) -> dict[str, Any]:
            self.snapshot_calls += 1
            return {
                "resource_state": "picked",
                "resource_location": "prusa-mk4-2",
                "held_part": "MCP",
                "named_poses": {"home": "home"},
            }

        def recovery_des_model(
            self,
            *,
            snapshot: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            self.model_calls += 1
            return {
                "state_variables": {
                    "resource_state": {
                        "scope": "resource",
                        "domain": [dict(snapshot or {}).get("resource_state")],
                    },
                    "resource_location": {
                        "scope": "resource",
                        "domain": ["prusa-mk4-2", "home"],
                    },
                    "held_part": {
                        "scope": "resource",
                        "domain": [None, "MCP"],
                    },
                    "part_state": {
                        "scope": "part",
                        "domain": ["in_gripper", "ready"],
                    },
                    "part_location": {
                        "scope": "part",
                        "domain": deepcopy(self.part_location_domain),
                    },
                }
            }

        def recovery_validation_resource_matches(self, resource_jid: str) -> bool:
            return ResourceAgent.recovery_validation_resource_matches(
                self, resource_jid
            )

        def check_recovery_physical_feasibility(
            self,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            self.physical_calls += 1
            return {"allowed": True, "reason": "physical feasibility passed"}

    owner = _Owner()
    home_release = {
        "candidate_index": 0,
        "task": {
            "resource_jid": "ur5e@localhost",
            "part_name": "MCP",
            "expected_start_state": {
                "resource_state": "picked",
                "resource_location": "prusa-mk4-2",
                "held_part": "MCP",
                "part_state": "in_gripper",
                "part_location": "ur5e@localhost",
            },
            "expected_end_state": {
                "resource_state": "placed",
                "resource_location": "home",
                "held_part": None,
                "part_state": "ready",
                "part_location": "home",
            },
        },
        "physical_input": {
            "part_context": {
                "part_state": "in_gripper",
                "part_location": "ur5e@localhost",
                "current_holder_resource_jid": "ur5e@localhost",
            }
        },
    }
    reachable_release = deepcopy(home_release)
    reachable_release["candidate_index"] = 1
    reachable_release["task"]["expected_end_state"]["resource_location"] = (
        "prusa-mk4-2"
    )
    reachable_release["task"]["expected_end_state"]["part_location"] = (
        "prusa-mk4-2"
    )
    payload = {
        "candidates": [
            home_release,
            reachable_release,
        ]
    }

    first = ResourceAgent.validate_recovery_outline_physical_candidates(
        owner, payload
    )
    rejected_candidate = first["results"][0]
    assert rejected_candidate["transition_feasibility"]["constraint_code"] == (
        "unknown_location_token"
    )
    assert rejected_candidate["transition_feasibility"]["evidence"] == {
        "field": "expected_end_state.part_location",
        "location_token": "home",
    }
    assert rejected_candidate["physical_feasibility"]["skipped"] is True
    assert first["results"][1]["allowed"] is True
    assert owner.physical_calls == 1

    owner.part_location_domain.append("home")
    second = ResourceAgent.validate_recovery_outline_physical_candidates(
        owner, {"candidates": [home_release]}
    )
    assert second["results"][0]["allowed"] is True
    assert owner.snapshot_calls == 2
    assert owner.model_calls == 2
    assert owner.physical_calls == 2

    resource_only_task = {
        "resource_jid": "ur5e@localhost",
        "expected_start_state": {
            "resource_state": "picked",
            "resource_location": "prusa-mk4-2",
            "held_part": "MCP",
        },
        "expected_end_state": {
            "resource_state": "idle",
            "resource_location": "home",
            "held_part": "MCP",
        },
    }
    resource_only = ResourceAgent.check_recovery_transition_feasibility(
        owner,
        task=resource_only_task,
        recovery_snapshot=owner.get_recovery_snapshot(),
        part_context={},
        recovery_des_model=owner.recovery_des_model(),
    )
    assert resource_only["allowed"] is True

    null_resource_location = deepcopy(resource_only_task)
    null_resource_location["expected_end_state"]["resource_location"] = None
    snapshot = owner.get_recovery_snapshot()
    descriptor = owner.recovery_des_model(snapshot=snapshot)
    rejected_null = ResourceAgent.check_recovery_transition_feasibility(
        owner,
        task=null_resource_location,
        recovery_snapshot=snapshot,
        part_context={},
        recovery_des_model=descriptor,
    )
    assert rejected_null["allowed"] is False
    assert rejected_null["constraint_code"] == "unknown_location_token"

    descriptor_with_null = deepcopy(descriptor)
    descriptor_with_null["state_variables"]["resource_location"]["domain"].append(
        None
    )
    accepted_null = ResourceAgent.check_recovery_transition_feasibility(
        owner,
        task=null_resource_location,
        recovery_snapshot=snapshot,
        part_context={},
        recovery_des_model=descriptor_with_null,
    )
    assert accepted_null["allowed"] is True

    custody_change_without_part_name = ResourceAgent.check_recovery_transition_feasibility(
        owner,
        task={
            "resource_jid": "ur5e@localhost",
            "expected_start_state": {
                "resource_state": "picked",
                "resource_location": "prusa-mk4-2",
                "held_part": "MCP",
            },
            "expected_end_state": {
                "resource_state": "idle",
                "resource_location": "home",
                "held_part": None,
            },
        },
        recovery_snapshot=owner.get_recovery_snapshot(),
        part_context={},
        recovery_des_model=owner.recovery_des_model(),
    )
    assert custody_change_without_part_name["allowed"] is False
    assert custody_change_without_part_name["constraint_code"] == (
        "part_traceability_violation"
    )


def test_ra_preserves_observed_pose_without_weakening_custody() -> None:
    descriptor = {
        "state_variables": {
            "resource_state": {"scope": "resource", "domain": ["idle"]},
            "resource_location": {
                "scope": "resource",
                "domain": ["assembly_board-v1", "home"],
            },
            "held_part": {"scope": "resource", "domain": [None, "LG"]},
            "part_state": {
                "scope": "part",
                "domain": ["unknown", "in_gripper"],
            },
            "part_location": {
                "scope": "part",
                "domain": [
                    None,
                    "assembly_board-v1",
                    "ur5e@localhost",
                ],
            },
        }
    }
    owner = SimpleNamespace(
        jid="ur5e@localhost",
        static_capabilities={"resource_type": "robot"},
    )
    snapshot = {
        "resource_state": "idle",
        "resource_location": "assembly_board-v1",
        "held_part": None,
        "named_poses": {"home": "home"},
    }
    part_context = {
        "part_state": "unknown",
        "observed_pose": {"x": 0.1, "y": 0.2, "z": 0.3},
        "current_holder_resource_jid": None,
    }
    acquisition = ResourceAgent.check_recovery_transition_feasibility(
        owner,
        task={
            "resource_jid": "ur5e@localhost",
            "part_name": "LG",
            "expected_start_state": {
                "resource_state": "idle",
                "resource_location": "assembly_board-v1",
                "held_part": None,
                "part_state": "unknown",
                "part_location": "observed_pose",
            },
            "expected_end_state": {
                "resource_state": "picked",
                "resource_location": "assembly_board-v1",
                "held_part": "LG",
                "part_state": "in_gripper",
                "part_location": "ur5e@localhost",
            },
        },
        recovery_snapshot=snapshot,
        part_context=part_context,
        recovery_des_model=descriptor,
    )
    unchanged = ResourceAgent.check_recovery_transition_feasibility(
        owner,
        task={
            "resource_jid": "ur5e@localhost",
            "part_name": "LG",
            "expected_start_state": {
                "resource_state": "idle",
                "resource_location": "assembly_board-v1",
                "held_part": None,
                "part_state": "unknown",
                "part_location": "observed_pose",
            },
            "expected_end_state": {
                "resource_state": "idle",
                "resource_location": "assembly_board-v1",
                "held_part": None,
                "part_state": "misplaced",
                "part_location": "observed_pose",
            },
        },
        recovery_snapshot=snapshot,
        part_context=part_context,
        recovery_des_model=descriptor,
    )

    invalid_release = {
        "resource_jid": "ur5e@localhost",
        "part_name": "LG",
        "expected_start_state": {
            "resource_state": "picked",
            "held_part": "LG",
            "part_state": "in_gripper",
            "part_location": "ur5e@localhost",
        },
        "expected_end_state": {
            "resource_state": "placed",
            "held_part": None,
            "part_state": "ready",
            "part_location": "ur5e@localhost",
        },
    }
    custody_mismatch = ResourceAgent.check_recovery_transition_feasibility(
        owner,
        task=invalid_release,
        recovery_snapshot={"resource_state": "picked", "held_part": "LG"},
        part_context={
            "part_state": "in_gripper",
            "part_location": "ur5e@localhost",
            "current_holder_resource_jid": "ur5e@localhost",
        },
        recovery_des_model=descriptor,
    )

    assert acquisition["allowed"] is True
    assert unchanged["allowed"] is True
    assert custody_mismatch["constraint_code"] == "held_part_location_mismatch"


def test_ra_rejects_empty_or_wrong_bound_custody_and_preserves_omitted_custody() -> None:
    descriptor = {
        "state_variables": {
            "resource_state": {
                "scope": "resource",
                "domain": ["picked", "inspected", "idle"],
            },
            "held_part": {
                "scope": "resource",
                "domain": [None, "LG", "MCP"],
            },
            "part_state": {
                "scope": "part",
                "domain": ["in_gripper", "verified", "ready"],
            },
            "part_location": {
                "scope": "part",
                "domain": [None, "station", "ur5e@localhost"],
            },
        }
    }
    owner = SimpleNamespace(
        jid="ur5e@localhost",
        static_capabilities={"resource_type": "robot"},
    )
    held_snapshot = {
        "resource_state": "picked",
        "held_part": "LG",
    }
    held_part_context = {
        "part_state": "in_gripper",
        "part_location": "ur5e@localhost",
        "current_holder_resource_jid": "ur5e@localhost",
    }
    inspection = {
        "resource_jid": "ur5e@localhost",
        "part_name": "LG",
        "expected_start_state": {
            "resource_state": "picked",
            "held_part": "LG",
            "part_state": "in_gripper",
            "part_location": "ur5e@localhost",
        },
        "expected_end_state": {
            "resource_state": "inspected",
            "part_state": "verified",
        },
    }
    preserved = ResourceAgent.check_recovery_transition_feasibility(
        owner,
        task=inspection,
        recovery_snapshot=held_snapshot,
        part_context=held_part_context,
        recovery_des_model=descriptor,
    )
    assert preserved["allowed"] is True

    empty_held_part = deepcopy(inspection)
    empty_held_part["expected_end_state"].update(
        {
            "held_part": "",
            "part_location": "station",
        }
    )
    rejected_empty = ResourceAgent.check_recovery_transition_feasibility(
        owner,
        task=empty_held_part,
        recovery_snapshot=held_snapshot,
        part_context=held_part_context,
        recovery_des_model=descriptor,
    )
    assert rejected_empty["allowed"] is False
    assert rejected_empty["constraint_code"] == "part_traceability_violation"

    wrong_bound_release = deepcopy(inspection)
    wrong_bound_release["part_name"] = "LG"
    wrong_bound_release["expected_start_state"]["held_part"] = "MCP"
    wrong_bound_release["expected_end_state"].update(
        {
            "held_part": None,
            "part_location": "station",
        }
    )
    rejected_wrong_part = ResourceAgent.check_recovery_transition_feasibility(
        owner,
        task=wrong_bound_release,
        recovery_snapshot={"resource_state": "picked", "held_part": "MCP"},
        part_context=held_part_context,
        recovery_des_model=descriptor,
    )
    assert rejected_wrong_part["allowed"] is False
    assert rejected_wrong_part["constraint_code"] == (
        "part_traceability_violation"
    )
    assert rejected_wrong_part["evidence"]["field"] == (
        "expected_start_state.held_part"
    )


def test_ra_batch_refreshes_one_snapshot_and_rejects_wrong_resource_request() -> None:
    class _Owner:
        jid = "xarm6@localhost"

        def __init__(self) -> None:
            self.snapshot_calls = 0
            self.validation_snapshots: list[dict[str, Any]] = []

        def get_recovery_snapshot(self) -> dict[str, Any]:
            self.snapshot_calls += 1
            return {
                "resource_state": "failed",
                "resource_location": "assembly_board-v1",
                "held_part": None,
            }

        def recovery_des_model(
            self,
            *,
            snapshot: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            del snapshot
            return {
                "state_variables": {
                    "resource_state": {
                        "scope": "resource",
                        "domain": ["failed", "idle"],
                    },
                    "resource_location": {
                        "scope": "resource",
                        "domain": ["assembly_board-v1", "home"],
                    },
                    "held_part": {
                        "scope": "resource",
                        "domain": [None],
                    },
                },
                "descriptor_fingerprint": "test-descriptor",
            }

        def recovery_validation_resource_matches(self, resource_jid: str) -> bool:
            return ResourceAgent.recovery_validation_resource_matches(
                self, resource_jid
            )

        def check_recovery_physical_feasibility(self, **kwargs: Any) -> dict[str, Any]:
            self.validation_snapshots.append(deepcopy(kwargs["recovery_snapshot"]))
            return {"allowed": True, "reason": "owner accepted"}

    owner = _Owner()
    payload = {
        "candidates": [
            {
                "candidate_index": 0,
                "task": {
                    "resource_jid": "xarm6@localhost",
                    "expected_start_state": {
                        "resource_state": "failed",
                        "resource_location": "assembly_board-v1",
                        "held_part": None,
                    },
                    "expected_end_state": {
                        "resource_state": "generated_clear_state",
                        "resource_location": "assembly_board-v1",
                        "held_part": None,
                    },
                },
                "physical_input": {
                    "use_projected_recovery_snapshot": False,
                    "projected_recovery_snapshot": {
                        "resource_state": "projected_but_not_accepted",
                        "resource_location": "home",
                        "held_part": "LG",
                    }
                },
            },
            {
                "candidate_index": 1,
                "task": {
                    "resource_jid": "xarm6@localhost",
                    "expected_start_state": {
                        "resource_state": "idle",
                        "resource_location": "home",
                        "held_part": "LG",
                    },
                    "expected_end_state": {
                        "resource_state": "generated_transport_state",
                        "resource_location": "home",
                        "held_part": "LG",
                    },
                },
                "physical_input": {
                    "use_projected_recovery_snapshot": True,
                    "projected_recovery_snapshot": {
                        "resource_state": "idle",
                        "resource_location": "home",
                        "held_part": "LG",
                        "gripper_state": "closed",
                    }
                },
            },
            {
                "candidate_index": 2,
                "task": {"resource_jid": "ur5e@localhost"},
                "physical_input": {},
            },
        ]
    }
    result = ResourceAgent.validate_recovery_outline_physical_candidates(
        owner, payload
    )
    assert owner.snapshot_calls == 1
    assert [row["allowed"] for row in result["results"]] == [True, True, False]
    assert owner.validation_snapshots == [
        {
            "resource_state": "failed",
            "resource_location": "assembly_board-v1",
            "held_part": None,
        },
        {
            "resource_state": "idle",
            "resource_location": "home",
            "held_part": "LG",
        },
    ]
    assert result["snapshot"] == {
        "resource_state": "failed",
        "resource_location": "assembly_board-v1",
        "held_part": None,
    }
    assert result["results"][2]["findings"][0]["constraint_code"] == (
        "wrong_resource_validator"
    )
    assert result["results"][2]["findings"][0]["validation_category"] == (
        "transition_feasibility"
    )
    assert result["snapshot_fingerprint"] == recovery_validation_fingerprint(
        result["snapshot"]
    )


def test_ra_retrieves_dynamic_capabilities_and_snapshot_on_every_request() -> None:
    class _Owner:
        jid = "xarm6@localhost"

        def __init__(self) -> None:
            self.snapshot_calls = 0
            self.model_calls = 0
            self.current_state = "failed"
            self.location_domain = ["home"]

        def get_recovery_snapshot(self) -> dict[str, Any]:
            self.snapshot_calls += 1
            return {
                "resource_state": self.current_state,
                "resource_location": "home",
                "held_part": None,
                "named_poses": {"home": "home"},
            }

        def recovery_des_model(
            self,
            *,
            snapshot: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            self.model_calls += 1
            descriptor = {
                "state_variables": {
                    "resource_state": {
                        "scope": "resource",
                        "domain": [dict(snapshot or {}).get("resource_state")],
                    },
                    "resource_location": {
                        "scope": "resource",
                        "domain": deepcopy(self.location_domain),
                    },
                    "held_part": {"scope": "resource", "domain": [None]},
                }
            }
            descriptor["descriptor_fingerprint"] = recovery_validation_fingerprint(
                descriptor
            )
            return descriptor

        def recovery_validation_resource_matches(self, resource_jid: str) -> bool:
            return ResourceAgent.recovery_validation_resource_matches(
                self, resource_jid
            )

        def check_recovery_physical_feasibility(self, **_kwargs: Any) -> dict[str, Any]:
            return {"allowed": True, "reason": "physical feasibility passed"}

    owner = _Owner()
    payload = {
        "candidates": [
            {
                "candidate_index": 0,
                "task": {
                    "outline_id": "dynamic-capability",
                    "event_name": "generated_transition_name",
                    "resource_jid": "xarm6@localhost",
                    "expected_start_state": {
                        "resource_state": "failed",
                        "resource_location": "home",
                        "held_part": None,
                    },
                    "expected_end_state": {
                        "resource_state": "generated_resource_state",
                        "resource_location": "station",
                        "held_part": None,
                    },
                },
                "physical_input": {},
            }
        ]
    }

    first = ResourceAgent.validate_recovery_outline_physical_candidates(
        owner, payload
    )
    assert first["results"][0]["transition_feasibility"]["allowed"] is False
    assert first["results"][0]["findings"][0]["constraint_code"] == (
        "unknown_location_token"
    )

    owner.location_domain = ["home", "station"]
    second = ResourceAgent.validate_recovery_outline_physical_candidates(
        owner, payload
    )
    assert second["results"][0]["allowed"] is True

    owner.current_state = "idle"
    third = ResourceAgent.validate_recovery_outline_physical_candidates(
        owner, payload
    )
    assert third["results"][0]["transition_feasibility"]["allowed"] is False
    assert third["results"][0]["findings"][0]["constraint_code"] == (
        "validation_state_stale"
    )
    assert owner.snapshot_calls == 3
    assert owner.model_calls == 3


def test_ra_capability_and_malformed_result_fail_closed() -> None:
    class _Owner:
        jid = "xarm6@localhost"

        def get_recovery_snapshot(self) -> dict[str, Any]:
            return {
                "resource_state": "failed",
                "resource_location": "home",
                "held_part": None,
            }

        def recovery_des_model(
            self,
            *,
            snapshot: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            del snapshot
            return {}

        def recovery_validation_resource_matches(self, resource_jid: str) -> bool:
            return ResourceAgent.recovery_validation_resource_matches(
                self, resource_jid
            )

    payload = {
        "candidates": [
            {
                "candidate_index": 0,
                "task": {
                    "resource_jid": "xarm6@localhost",
                    "expected_start_state": {"resource_state": "failed"},
                    "expected_end_state": {"resource_state": "generated_state"},
                },
                "physical_input": {},
            }
        ]
    }
    result = ResourceAgent.validate_recovery_outline_physical_candidates(
        _Owner(), payload
    )
    candidate = result["results"][0]
    assert candidate["allowed"] is False
    assert candidate["transition_feasibility"]["constraint_code"] == (
        "resource_validation_unavailable"
    )
    assert candidate["physical_feasibility"]["skipped"] is True

    with pytest.raises(RuntimeError, match="unavailable or invalid"):
        multi_turn_outline_generation._verified_recovery_des_model(
            ra_reply={
                "recovery_des_model": {"state_variables": {}},
                "recovery_des_model_fingerprint": "malformed-fingerprint",
            }
        )


def test_robotagent_owns_projected_gripper_evidence_from_held_part() -> None:
    released = RobotAgent.recovery_physical_validation_snapshot(
        live_snapshot={"held_part": "MCP", "gripper_state": "closed"},
        physical_input={
            "use_projected_recovery_snapshot": True,
            "projected_recovery_snapshot": {"held_part": None},
        },
        recovery_des_model={
            "state_variables": {
                "held_part": {"scope": "resource"},
            }
        },
    )
    acquired = RobotAgent.recovery_physical_validation_snapshot(
        live_snapshot={"held_part": None, "gripper_state": "open"},
        physical_input={
            "use_projected_recovery_snapshot": True,
            "projected_recovery_snapshot": {"held_part": "LG"},
        },
        recovery_des_model={
            "state_variables": {
                "held_part": {"scope": "resource"},
            }
        },
    )

    assert released["gripper_state"] == "open"
    assert acquired["gripper_state"] == "closed"


def test_production_ra_reply_includes_recovery_des_descriptor() -> None:
    async def _run() -> dict[str, Any]:
        descriptor = {
            "model_type": "extended_finite_automaton",
            "resource_jid": "xarm6@localhost",
            "state_variables": {
                "resource_state": {
                    "scope": "resource",
                    "domain": ["idle"],
                }
            },
            "current_valuation": {"resource_state": "idle"},
            "events": [],
            "descriptor_fingerprint": "descriptor-fingerprint",
        }
        validation = {
            "validator_jid": "xarm6@localhost",
            "snapshot": {"resource_state": "idle"},
            "snapshot_fingerprint": "snapshot-fingerprint",
            "recovery_des_model": deepcopy(descriptor),
            "recovery_des_model_fingerprint": "descriptor-fingerprint",
            "results": [],
        }
        owner = SimpleNamespace(
            jid="xarm6@localhost",
            logger=logging.getLogger("test.ra.reply"),
            loop=asyncio.get_running_loop(),
            presence=None,
            web=None,
            validate_recovery_outline_physical_candidates=lambda _payload: deepcopy(
                validation
            ),
        )
        behaviour = ResourceAgent._RecoveryOutlinePhysicalValidationInbox()
        behaviour.set_agent(owner)
        message = SimpleNamespace(
            body=json.dumps(
                {
                    "request_id": "request-1",
                    "recovery_session_id": "session-1",
                    "turn_index": 2,
                    "state_fingerprint": "state-fingerprint",
                    "product_jid": "product@localhost",
                }
            ),
            sender="product@localhost",
        )

        async def _receive(*, timeout: float) -> Any:
            del timeout
            return message

        sent_messages: list[Any] = []

        async def _send(_behaviour: Any, reply: Any, **_kwargs: Any) -> str:
            sent_messages.append(reply)
            return "sent"

        behaviour.receive = _receive
        with patch(
            "cais_spade_llm.agents.resource_agent.resource_agent.send_agent_message",
            side_effect=_send,
        ):
            await behaviour.run()
        assert len(sent_messages) == 1
        return json.loads(sent_messages[0].body)

    response = asyncio.run(_run())
    assert response["recovery_des_model"]["resource_jid"] == "xarm6@localhost"
    assert response["recovery_des_model_fingerprint"] == "descriptor-fingerprint"


def test_ra_transition_uses_its_snapshot_for_start_state_consistency() -> None:
    task = {
        "outline_id": "candidate",
        "resource_jid": "xarm6@localhost",
        "expected_start_state": {
            "resource_state": "idle",
            "resource_location": "home",
            "held_part": None,
        },
        "expected_end_state": {
            "resource_state": "generated_recovery_state",
            "resource_location": "home",
            "held_part": None,
        },
    }
    snapshot = {
        "resource_state": "failed",
        "resource_location": "assembly_board-v1",
        "held_part": None,
    }
    descriptor = {
        "state_variables": {
            "resource_state": {"scope": "resource", "domain": ["failed", "idle"]},
            "resource_location": {
                "scope": "resource",
                "domain": ["assembly_board-v1", "home"],
            },
            "held_part": {"scope": "resource", "domain": [None]},
        }
    }
    result = ResourceAgent.check_recovery_transition_feasibility(
        SimpleNamespace(jid="xarm6@localhost", static_capabilities={}),
        task=task,
        recovery_snapshot=snapshot,
        part_context={},
        recovery_des_model=descriptor,
    )
    assert result["allowed"] is False
    assert result["constraint_code"] == "validation_state_stale"

    result = ResourceAgent.check_recovery_transition_feasibility(
        SimpleNamespace(jid="xarm6@localhost", static_capabilities={}),
        task=task,
        recovery_snapshot={
            "resource_state": "idle",
            "resource_location": "home",
            "held_part": None,
        },
        part_context={},
        recovery_des_model=descriptor,
    )
    assert result["allowed"] is True


def test_outline_state_schema_matches_pa_candidate_validator() -> None:
    schema = multi_turn_prompts._outline_candidates_response_schema(
        candidate_count=3,
        action_horizon="1",
    )
    definitions = schema["schema"]["$defs"]
    resource_state_schema = definitions["resource_only_outline_state"]
    part_state_schema = definitions["part_outline_state"]
    custody_state_schema = definitions["custody_outline_state"]

    for state_schema in (
        resource_state_schema,
        part_state_schema,
        custody_state_schema,
    ):
        assert state_schema["additionalProperties"] is False
        assert "resource_state" in state_schema["required"]

    assert set(resource_state_schema["properties"]) == {
        "resource_state",
        "resource_location",
    }
    assert set(part_state_schema["properties"]) == {
        "resource_state",
        "resource_location",
        "part_state",
        "part_location",
    }
    assert set(custody_state_schema["properties"]) == {
        "resource_state",
        "resource_location",
        "held_part",
        "part_state",
        "part_location",
    }
    assert custody_state_schema["required"] == [
        "resource_state",
        "held_part",
        "part_state",
        "part_location",
    ]


def test_cca_candidate_validation_allows_when_no_safety_rules_are_active() -> None:
    result = validate_outline_macro_recovery_safety(
        task={
            "outline_id": "candidate",
            "resource_jid": "xarm6@localhost",
            "expected_start_state": {"resource_state": "failed"},
            "expected_end_state": {
                "resource_state": "clear_state",
                "resource_location": "home",
            },
        },
        signature={"task_kind": "resource_only"},
        pre_resources={
            "xarm6@localhost": {
                "resource_jid": "xarm6@localhost",
                "current_state": "failed",
                "current_location": "assembly_board-v1",
            }
        },
        pre_parts={},
        projected_resources={
            "xarm6@localhost": {
                "resource_jid": "xarm6@localhost",
                "current_state": "clear_state",
                "current_location": "home",
            }
        },
        projected_parts={},
        llm_input={"loaded_safety_rules": []},
    )
    assert result["is_safe"] is True
    assert result["findings"] == []


def test_validation_fingerprint_is_stable_without_changing_tokens() -> None:
    first = {
        "resources": {"xarm6@localhost": {"resource_state": "failed"}},
        "parts": {"LG": {"part_state": "misplaced"}},
    }
    second = {
        "parts": {"LG": {"part_state": "misplaced"}},
        "resources": {"xarm6@localhost": {"resource_state": "failed"}},
    }
    assert recovery_validation_fingerprint(first) == recovery_validation_fingerprint(second)

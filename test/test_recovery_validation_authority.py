"""Focused tests for PA/RA/CCA recovery-validation authority boundaries."""

from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from cais_spade_llm.agents.central_controller.outline_macro_safety import (
    validate_outline_macro_recovery_safety,
)
from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
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
from cais_spade_llm.resources.capability_engine import (
    bind_configured_capabilities,
)


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


def test_ra_batch_refreshes_one_snapshot_and_rejects_wrong_resource_request() -> None:
    class _Owner:
        jid = "xarm6@localhost"
        static_capabilities = {
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
                    "domain": [None, "LG"],
                },
            },
            "events": [
                {
                    "event_name": "validate",
                    "guards": {},
                    "updates": {},
                }
            ],
        }
        capability_runtime_fact_names = ResourceAgent.capability_runtime_fact_names
        capability_runtime_facts = ResourceAgent.capability_runtime_facts
        capability_state_valuation = ResourceAgent.capability_state_valuation
        _executable_required_runtime_facts = staticmethod(
            ResourceAgent._executable_required_runtime_facts
        )
        _executable_runtime_fact_names = staticmethod(
            ResourceAgent._executable_runtime_fact_names
        )
        _configured_event_required_runtime_facts = (
            ResourceAgent._configured_event_required_runtime_facts
        )
        _capability_execution_arguments = (
            ResourceAgent._capability_execution_arguments
        )
        _capability_atomic_transition = (
            ResourceAgent._capability_atomic_transition
        )
        _evaluate_capability_transition = (
            ResourceAgent._evaluate_capability_transition
        )
        _resolve_generated_capability = (
            ResourceAgent._resolve_generated_capability
        )
        _generated_successor_from_action_target = (
            ResourceAgent._generated_successor_from_action_target
        )
        _grounded_action_from_atomic_transitions = (
            ResourceAgent._grounded_action_from_atomic_transitions
        )

        def __init__(self) -> None:
            self.snapshot_calls = 0
            self.validation_snapshots: list[dict[str, Any]] = []
            self._configured_capability_declarations = deepcopy(
                self.static_capabilities
            )
            self.executables = {
                "validate": lambda: {"success": True},
            }

        def get_recovery_snapshot(self) -> dict[str, Any]:
            self.snapshot_calls += 1
            return {
                "resource_state": "failed",
                "resource_location": "assembly_board-v1",
                "held_part": None,
            }

        def recovery_validation_resource_matches(self, resource_jid: str) -> bool:
            return ResourceAgent.recovery_validation_resource_matches(
                self, resource_jid
            )

        def _bind_capabilities(
            self,
            *,
            snapshot: dict[str, Any],
        ) -> dict[str, Any]:
            del snapshot
            return bind_configured_capabilities(
                SimpleNamespace(
                    jid=self.jid,
                    static_capabilities=self.static_capabilities,
                ),
                capabilities=self._configured_capability_declarations,
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
                    "event_name": "validate",
                    "resource_jid": "xarm6@localhost",
                    "expected_start_state": {
                        "resource_state": "failed",
                        "resource_location": "assembly_board-v1",
                        "held_part": None,
                    },
                    "expected_end_state": {
                        "resource_state": "failed",
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
                    "event_name": "validate",
                    "resource_jid": "xarm6@localhost",
                    "expected_start_state": {
                        "resource_state": "idle",
                        "resource_location": "home",
                        "held_part": "LG",
                    },
                    "expected_end_state": {
                        "resource_state": "idle",
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


def test_robotagent_owns_projected_gripper_evidence_from_held_part() -> None:
    released = RobotAgent.recovery_physical_validation_snapshot(
        live_snapshot={"held_part": "MCP", "gripper_state": "closed"},
        physical_input={
            "use_projected_recovery_snapshot": True,
            "projected_recovery_snapshot": {"held_part": None},
        },
        bound_capabilities={
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
        bound_capabilities={
            "state_variables": {
                "held_part": {"scope": "resource"},
            }
        },
    )

    assert released["gripper_state"] == "open"
    assert acquired["gripper_state"] == "closed"


def test_production_ra_reply_exposes_results_without_private_capabilities() -> None:
    async def _run() -> dict[str, Any]:
        validation = {
            "validator_jid": "xarm6@localhost",
            "snapshot": {"resource_state": "idle"},
            "results": [],
            "enabled_capability_results": [],
            "enabled_event_ids": [],
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
    assert response["results"] == []
    assert response["enabled_capability_results"] == []
    assert "snapshot" not in response
    assert "snapshot_fingerprint" not in response
    assert "configured_capabilities_fingerprint" not in response
    assert "state_variables" not in response
    assert "events" not in response


def test_outline_state_schema_matches_pa_candidate_validator() -> None:
    schema = multi_turn_prompts._outline_candidates_response_schema(
        candidate_count=3,
        action_horizon="1",
    )
    state_schema = schema["schema"]["$defs"]["outline_state"]
    assert state_schema["additionalProperties"] == {
        "type": ["string", "number", "boolean", "null"],
    }
    assert state_schema["minProperties"] == 1
    assert "required" not in state_schema
    assert set(state_schema["properties"]) == {
        "resource_state",
        "part_state",
    }
    assert (
        state_schema["properties"]["resource_state"]["properties"][
            "condition"
        ]["type"]
        == ["string", "null"]
    )
    assert "enum" not in json.dumps(state_schema)


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

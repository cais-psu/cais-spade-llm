from __future__ import annotations

from copy import deepcopy

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_validation_service import (
    validate_bridge_candidate_task,
)


def _prepared_bridge_request() -> dict[str, object]:
    return {
        "bridge_resources": {
            "ur5e@localhost": {
                "resource_type": "robot",
                "bridge_snapshot": {
                    "resource_type": "robot",
                    "reachable_locations": ["assembly_board-v1", "prusa-mk4-2"],
                    "available_named_poses": ["home"],
                },
                "static_capabilities": {
                    "resource_type": "robot",
                    "reachability": ["assembly_board-v1", "prusa-mk4-2"],
                },
            },
        },
        "grounding_context": {
            "parts": {
                "LG": {
                    "target": {"location": "assembly_board-v1"},
                }
            }
        },
        "llm_input": {
            "fault_event": {"affected_part_names": ["LG"]},
        },
    }


def _turn6_state() -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "symbolic_resources": {
                "ur5e@localhost": {
                    "resource_jid": "ur5e@localhost",
                    "resource_type": "robot",
                    "current_state": "idle",
                    "held_part": None,
                    "gripper_state": "open",
                }
            },
            "symbolic_parts": {
                "LG": {
                    "part_name": "LG",
                    "current_state": "misplaced",
                    "current_location": None,
                    "observed_pose": {"x": 0.0, "y": 0.2, "z": 1.035},
                    "current_holder_resource_jid": None,
                    "goal_location": "assembly_board-v1",
                }
            },
            "observation_store": {},
        },
        _prepared_bridge_request(),
    )


class _Planner:
    def __init__(self, *resource_agents: object) -> None:
        self.resource_agents = list(resource_agents)


class _ResourceWithoutOracle:
    def __init__(self, jid: str) -> None:
        self.jid = jid


class _AllowingResource:
    def __init__(self, jid: str) -> None:
        self.jid = jid

    def bridge_feasibility_oracle(self, **_kwargs: object) -> dict[str, object]:
        return {"allowed": True}


def test_bridge_validation_service_fails_closed_without_resource_oracle() -> None:
    session_state, prepared_bridge_request = _turn6_state()
    planner = _Planner(_ResourceWithoutOracle("ur5e@localhost"))

    result = validate_bridge_candidate_task(
        planner=planner,
        candidate_task={
            "outline_id": "RECOVERY_SEQ3",
            "event_schema_id": "pick_part",
            "resource_binding": "ur5e@localhost",
            "object_bindings": {
                "part": "LG",
                "source_location": "observed_pose",
            },
            "parameters": {},
            "depends_on": [],
            "rationale": "Pick LG from the observed pose.",
        },
        session_state=deepcopy(session_state),
        prepared_bridge_request=prepared_bridge_request,
    )

    assert not result.ok
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.stage == "resource_realizability"
    assert finding.code == "resource_validation_unavailable"
    assert "fails closed" in finding.reason


def test_bridge_validation_service_surfaces_cca_findings(monkeypatch) -> None:
    session_state, prepared_bridge_request = _turn6_state()
    planner = _Planner(_AllowingResource("ur5e@localhost"))

    from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge import bridge_validation_service

    def _fake_cca(**_kwargs: object) -> dict[str, object]:
        return {
            "findings": [
                {
                    "constraint_code": "order_violation",
                    "reason": "candidate would violate continuation ordering",
                    "resource_jid": "ur5e@localhost",
                    "part_name": "LG",
                }
            ]
        }

    monkeypatch.setattr(
        bridge_validation_service,
        "validate_outline_macro_cca_constraints",
        _fake_cca,
    )

    result = validate_bridge_candidate_task(
        planner=planner,
        candidate_task={
            "outline_id": "RECOVERY_SEQ3",
            "event_schema_id": "pick_part",
            "resource_binding": "ur5e@localhost",
            "object_bindings": {
                "part": "LG",
                "source_location": "observed_pose",
            },
            "parameters": {},
            "depends_on": [],
            "rationale": "Pick LG from the observed pose.",
        },
        session_state=deepcopy(session_state),
        prepared_bridge_request=prepared_bridge_request,
    )

    assert not result.ok
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.stage == "supervisor_admissibility"
    assert finding.code == "order_violation"
    assert "continuation ordering" in finding.reason

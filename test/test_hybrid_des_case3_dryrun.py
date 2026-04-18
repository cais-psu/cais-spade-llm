"""Dry-run scenario for Case 3 hybrid DES bridge recovery.

Run directly:
    python test/test_hybrid_des_case3_dryrun.py
    python test/test_hybrid_des_case3_dryrun.py --model gpt-5
    python test/test_hybrid_des_case3_dryrun.py --scripted
    python test/test_hybrid_des_case3_dryrun.py --scripted --mode procedural

Run as pytest:
    pytest test/test_hybrid_des_case3_dryrun.py -v
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bootstrap_repo_site_packages(root: Path) -> None:
    venv_lib = root / ".venv" / "lib"
    if not venv_lib.exists():
        return
    for site_packages in sorted(venv_lib.glob("python*/site-packages")):
        site_path = str(site_packages.resolve())
        if site_path not in sys.path:
            sys.path.insert(0, site_path)


_bootstrap_repo_site_packages(ROOT)

import pytest

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge import bridge_session as bridge_session_module
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
    build_hybrid_session_seed,
    build_procedural_session_seed,
    execute_hybrid_des_bridge,
    execute_procedural_des_bridge,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import des_recovery_common as des_common_module


def _workspace_bounds(x_min: float, x_max: float, y_min: float, y_max: float, z_min: float = 0.8, z_max: float = 1.4) -> dict[str, float]:
    return {
        "x_min_m": x_min,
        "x_max_m": x_max,
        "y_min_m": y_min,
        "y_max_m": y_max,
        "z_min_m": z_min,
        "z_max_m": z_max,
    }


def _pose_in_bounds(pose: dict[str, Any], bounds: dict[str, float]) -> bool:
    for axis in ("x", "y", "z"):
        value = float(pose.get(axis, 0.0))
        if value < float(bounds[f"{axis}_min_m"]) or value > float(bounds[f"{axis}_max_m"]):
            return False
    return True


def _case3_tools_catalog() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for owner in ("xarm6", "ur5e"):
        for function_name in ("move_home", "pick_approach", "pick_grasp", "place_approach", "place_insert"):
            tools.append({
                "function": function_name,
                "function_owner_agent": owner,
            })
    return tools


def build_case3_prepared_bridge_request(
    *,
    reasoning_mode: str,
    max_turns: int = 6,
) -> dict[str, Any]:
    bridge_resources = {
        "xarm6@localhost": {
            "resource_jid": "xarm6@localhost",
            "current_state": "idle",
            "current_location": "xarm6_home",
            "workspace_bounds": _workspace_bounds(-0.6, 0.6, -1.0, 0.1),
            "named_poses": {"xarm6_home": {}, "Assembly Station": {}},
        },
        "ur5e@localhost": {
            "resource_jid": "ur5e@localhost",
            "current_state": "idle",
            "current_location": "ur5e_home",
            "workspace_bounds": _workspace_bounds(-0.7, 0.7, -0.15, 1.1),
            "named_poses": {"ur5e_home": {}, "Assembly Station": {}},
        },
    }
    llm_input = {
        "observed_runtime_state": {
            "resources": [
                {
                    "resource_jid": "xarm6@localhost",
                    "current_state": "idle",
                    "current_location": "xarm6_home",
                },
                {
                    "resource_jid": "ur5e@localhost",
                    "current_state": "idle",
                    "current_location": "ur5e_home",
                },
            ],
        },
        "part_facts": [
            {
                "part_name": "LG",
                "current_location": "recovery_surface",
                "goal_location": "Assembly Station",
                "assigned_resource_jid": "xarm6@localhost",
                "goal_resource_jid": "xarm6@localhost",
                "needs_observation": True,
                "location_basis": "symbolic_inference",
                "observed_pose": None,
                "current_holder_resource_jid": None,
            },
            {
                "part_name": "MCP",
                "current_location": "Assembly Station",
                "goal_location": "Assembly Station",
                "assigned_resource_jid": "ur5e@localhost",
                "goal_resource_jid": "ur5e@localhost",
                "location_basis": "symbolic_inference",
                "observed_pose": None,
                "current_holder_resource_jid": None,
            },
        ],
        "loaded_safety_rules": [],
        "relevant_assembly_requirements": [
            {"requirement_id": "REQ_1", "summary": "restore MCP", "status": "completed"},
            {"requirement_id": "REQ_2", "summary": "restore LG", "status": "failed"},
        ],
    }
    return {
        "bridge_session": {
            "reasoning_mode": reasoning_mode,
            "max_turns": max_turns,
            "solver_max_explored_states": 1000,
            "validator_timeout_s": 1.0,
        },
        "bridge_debug": {"reasoning_mode": reasoning_mode},
        "bridge_resources": deepcopy(bridge_resources),
        "llm_input": deepcopy(llm_input),
        "ra_jid": "ur5e@localhost",
        "tools_catalog": _case3_tools_catalog(),
    }


def hybrid_case3_llm_responses() -> list[dict[str, Any]]:
    return [
        {
            "thought": "First draft is too shallow and should be rejected.",
            "plant": {
                "states": ["s0"],
                "initial": "s0",
                "events": [],
            },
        },
        {
            "thought": (
                "LG was displaced into the UR5e region while still assigned to xArm6. "
                "Recover by using UR5e to reacquire and complete placement so execution can resume."
            ),
            "plant": {
                "states": ["s0", "s1", "s2", "s3", "s_goal"],
                "initial": "s0",
                "marked": ["s_goal"],
                "state_metadata": {
                    "s0": {
                        "atomic_bindings": {
                            "PoseKnown(LG)": True,
                            "LGNeedsRecovery": True,
                        },
                    },
                    "s1": {
                        "atomic_bindings": {
                            "PoseKnown(LG)": True,
                            "UR5AlignedToLG": True,
                        },
                    },
                    "s2": {
                        "atomic_bindings": {"HeldBy(LG, ur5e@localhost)": True},
                    },
                    "s3": {
                        "atomic_bindings": {"AtGoal(LG, Assembly Station)": False},
                    },
                },
                "marked_state_metadata": {
                    "s_goal": {
                        "marking_predicate": "LG is at Assembly Station and the disrupted suffix can resume"
                    }
                },
                "events": [
                    {
                        "name": "ur5e_pick_approach_lg",
                        "from": "s0",
                        "to": "s1",
                        "resource_jid": "ur5e@localhost",
                        "part_name": "LG",
                        "location_ref": "observed_pose",
                        "description": "UR5e approaches the grounded LG pose in its reachable workspace",
                    },
                    {
                        "name": "ur5e_pick_grasp_lg",
                        "from": "s1",
                        "to": "s2",
                        "resource_jid": "ur5e@localhost",
                        "part_name": "LG",
                        "location_ref": "observed_pose",
                        "description": "UR5e grasps LG from the observed displaced pose",
                    },
                    {
                        "name": "ur5e_place_approach_lg",
                        "from": "s2",
                        "to": "s3",
                        "resource_jid": "ur5e@localhost",
                        "part_name": "LG",
                        "location_ref": "Assembly Station",
                        "description": "UR5e carries LG back toward the assembly station",
                    },
                    {
                        "name": "ur5e_place_insert_lg",
                        "from": "s3",
                        "to": "s_goal",
                        "resource_jid": "ur5e@localhost",
                        "part_name": "LG",
                        "location_ref": "Assembly Station",
                        "description": "UR5e completes LG placement at the assembly station",
                    },
                ],
            },
        },
    ]


class ScriptedProductAgent:
    def __init__(self, scripted_responses: list[dict[str, Any]]) -> None:
        self._responses = list(scripted_responses)
        self.prompts: list[str] = []

    async def ask_llm_structured(
        self,
        *,
        prompt: str,
        response_format: dict[str, Any],
        **_: Any,
    ) -> dict[str, Any]:
        del response_format
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("No scripted LLM responses left for hybrid test.")
        return deepcopy(self._responses.pop(0))


class ResourceAgentStub:
    def __init__(self, resource_jid: str, workspace_bounds: dict[str, float]) -> None:
        self.resource_jid = resource_jid
        self.workspace_bounds = deepcopy(workspace_bounds)

    def get_bridge_snapshot(self) -> dict[str, Any]:
        return {"workspace_bounds": deepcopy(self.workspace_bounds)}

    def bridge_feasibility_oracle(
        self,
        *,
        operation_kind: str,
        part_name: str | None,
        part_context: dict[str, Any],
        bridge_snapshot: dict[str, Any],
        grounded_action: dict[str, Any],
    ) -> dict[str, Any]:
        del grounded_action
        normalized = str(operation_kind or "").strip().lower()
        if normalized in {"pick_grasp", "pick_part", "pick", "grasp", "acquire"} and part_name == "LG":
            pose = dict(part_context.get("observed_pose") or {})
            bounds = dict(bridge_snapshot.get("workspace_bounds") or self.workspace_bounds)
            if pose and not _pose_in_bounds(pose, bounds):
                return {
                    "allowed": False,
                    "constraint_code": "workspace_unreachable",
                    "reason": (
                        f"{self.resource_jid} cannot reach observed LG pose "
                        f"{pose} within workspace {bounds}"
                    ),
                    "evidence": {
                        "checked_pose": deepcopy(pose),
                        "workspace_bounds": deepcopy(bounds),
                    },
                }
        return {"allowed": True}


class PlannerStub:
    def __init__(
        self,
        *,
        product_agent: Any,
        bridge_resources: dict[str, Any],
    ) -> None:
        self.product_agent = product_agent
        self._resource_agents = {
            resource_jid: ResourceAgentStub(resource_jid, dict(resource_entry.get("workspace_bounds") or {}))
            for resource_jid, resource_entry in bridge_resources.items()
        }
        self._last_bridge_debug: dict[str, Any] = {}

    def _resource_by_jid(self, resource_jid: str) -> Any:
        return self._resource_agents.get(resource_jid)

    def _set_last_bridge_debug(self, bridge_debug: dict[str, Any]) -> None:
        self._last_bridge_debug = deepcopy(bridge_debug)


def _allow_all_cca(*_: Any, **__: Any) -> dict[str, Any]:
    return {"findings": []}


def run_hybrid_case3_dryrun(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(des_common_module, "validate_outline_macro_cca_constraints", _allow_all_cca)
    prepared_bridge_request = build_case3_prepared_bridge_request(reasoning_mode="hybrid")
    prepared_bridge_request["hybrid_session_seed"] = build_hybrid_session_seed(prepared_bridge_request)
    product_agent = ScriptedProductAgent(hybrid_case3_llm_responses())
    planner = PlannerStub(
        product_agent=product_agent,
        bridge_resources=dict(prepared_bridge_request.get("bridge_resources") or {}),
    )

    first_result = asyncio.run(execute_hybrid_des_bridge(planner, prepared_bridge_request))
    assert first_result is None
    paused_state = deepcopy(prepared_bridge_request.get("hybrid_session_state") or {})
    assert paused_state.get("status") == "paused_after_grounding"
    assert "multi_turn_session_result" not in prepared_bridge_request
    assert "hybrid_pending_observation_tasks" in prepared_bridge_request
    assert "multi_turn_session" not in dict(prepared_bridge_request.get("bridge_debug") or {})
    grounding_turn = dict((paused_state.get("turns") or [])[-1] or {})
    assert "Current Resource Facts" in str(grounding_turn.get("prompt_text") or "")
    assert "Current Part Facts" in str(grounding_turn.get("prompt_text") or "")
    grounding_response = dict(grounding_turn.get("raw_response") or {})
    assert grounding_response.get("decision") == "observe"
    assert grounding_response.get("observe_requests")

    checkpoint = bridge_session_module.record_grounding_checkpoint(paused_state)
    assert checkpoint["current_phase"] == "evaluate_grounding"

    resumed_state = bridge_session_module.resume_from_grounding(
        paused_state,
        {"LG": {"x": 0.0, "y": 0.2, "z": 1.05}},
        prepared_bridge_request=prepared_bridge_request,
    )
    prepared_bridge_request["hybrid_session_state"] = deepcopy(resumed_state)
    proposal = asyncio.run(
        execute_hybrid_des_bridge(
            planner,
            prepared_bridge_request,
            session_state=resumed_state,
        )
    )
    session_state = deepcopy(prepared_bridge_request.get("hybrid_session_state") or {})
    return {
        "proposal": proposal,
        "session_state": session_state,
        "prepared_bridge_request": prepared_bridge_request,
        "planner": planner,
        "product_agent": product_agent,
    }


def run_procedural_case3_dryrun(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(des_common_module, "validate_outline_macro_cca_constraints", _allow_all_cca)
    prepared_bridge_request = build_case3_prepared_bridge_request(reasoning_mode="procedural_des_v1", max_turns=4)
    prepared_bridge_request["procedural_session_seed"] = build_procedural_session_seed(prepared_bridge_request)
    planner = PlannerStub(
        product_agent=object(),
        bridge_resources=dict(prepared_bridge_request.get("bridge_resources") or {}),
    )

    first_result = asyncio.run(execute_procedural_des_bridge(planner, prepared_bridge_request))
    assert first_result is None
    paused_state = deepcopy(prepared_bridge_request.get("procedural_session_state") or {})
    assert paused_state.get("status") == "paused_after_grounding"

    resumed_state = bridge_session_module.resume_from_grounding(
        paused_state,
        {"LG": {"x": 0.0, "y": 0.2, "z": 1.05}},
        prepared_bridge_request=prepared_bridge_request,
    )
    prepared_bridge_request["procedural_session_state"] = deepcopy(resumed_state)
    proposal = asyncio.run(
        execute_procedural_des_bridge(
            planner,
            prepared_bridge_request,
            session_state=resumed_state,
        )
    )
    session_state = deepcopy(prepared_bridge_request.get("procedural_session_state") or {})
    return {
        "proposal": proposal,
        "session_state": session_state,
        "prepared_bridge_request": prepared_bridge_request,
        "planner": planner,
    }


def test_hybrid_des_case3_dryrun(monkeypatch: pytest.MonkeyPatch) -> None:
    result = run_hybrid_case3_dryrun(monkeypatch)
    proposal = dict(result.get("proposal") or {})
    session_state = dict(result.get("session_state") or {})
    prepared_bridge_request = dict(result.get("prepared_bridge_request") or {})
    product_agent = result.get("product_agent")

    assert proposal
    assert proposal.get("engine") == "hybrid_des_v1"
    assert len(proposal.get("accepted_trace") or []) >= 4
    assert len(proposal.get("outline_tasks") or []) >= 4
    assert proposal.get("declared_composite_states")
    assert proposal.get("declared_marking_predicates")
    assert proposal.get("revision_history_summary") != "(none)"

    assert session_state.get("status") == "completed"
    assert session_state.get("domain_revision_count", 0) >= 1
    assert session_state.get("revision_history")
    assert prepared_bridge_request.get("hybrid_session_state")
    assert "multi_turn_session_result" not in prepared_bridge_request
    assert "multi_turn_session" not in dict(prepared_bridge_request.get("bridge_debug") or {})
    assert "Open Recovery Conditions" in product_agent.prompts[-1]
    assert "Allowed Location References" in product_agent.prompts[-1]
    assert "Output Constraints" in product_agent.prompts[-1]
    assert "Deadlock / Transfer Context" not in product_agent.prompts[-1]
    assert "Dynamic Predicate Vocabulary" not in product_agent.prompts[-1]


def _run_with_monkeypatch(mode: str) -> dict[str, Any]:
    monkeypatch = pytest.MonkeyPatch()
    try:
        if mode == "procedural":
            return run_procedural_case3_dryrun(monkeypatch)
        return run_hybrid_case3_dryrun(monkeypatch)
    finally:
        monkeypatch.undo()


def _print_dryrun_summary(result: dict[str, Any]) -> None:
    proposal = dict(result.get("proposal") or {})
    session_state = dict(result.get("session_state") or {})
    print()
    print(f"Scripted DES dry-run status: {session_state.get('status') or '-'}")
    print(f"Engine: {proposal.get('engine') or '-'}")
    print(f"Accepted trace events: {len(proposal.get('accepted_trace') or [])}")
    print(f"Outline tasks: {len(proposal.get('outline_tasks') or [])}")
    print(f"Domain revisions: {session_state.get('domain_revision_count', 0)}")


async def _run_full_case3_hybrid_dryrun(
    *,
    llm_model: str | None,
    write_debug: bool,
) -> None:
    from test_case3_bridge_dryrun import (
        _print_debug_artifact_paths,
        _print_prepare_context_summary,
        run_case3_bridge_dryrun,
    )

    result = await run_case3_bridge_dryrun(
        write_debug=write_debug,
        llm_model=llm_model,
        reasoning_mode="hybrid",
        stop_before_primitive_generation=False,
    )

    print()
    print(f"Bridge status: {result.get('status') or '-'}")
    _print_prepare_context_summary(result.get("context_summary") or {})
    _print_debug_artifact_paths(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Case 3 DES bridge dry-run harness"
    )
    parser.add_argument("--model", default=None, help="OpenAI model name for the full hybrid harness")
    parser.add_argument("--no-debug", action="store_true", help="Skip final debug artifact write in the full harness")
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Run the compact scripted unit dry-run instead of the full live hybrid harness",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=("hybrid", "procedural"),
        help="Scripted DES bridge mode to execute with --scripted",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if args.scripted:
        _print_dryrun_summary(_run_with_monkeypatch(args.mode))
    else:
        asyncio.run(
            _run_full_case3_hybrid_dryrun(
                llm_model=args.model,
                write_debug=not args.no_debug,
            )
        )

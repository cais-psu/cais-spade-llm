"""
End-to-end test: SG/MCP deadlock recovery scenario.

Scenario:
  - xArm6 fails to place SG (slippage) → recovery_required, unavailable
  - SG is misplaced at UR5e workspace region
  - UR5e is holding MCP, positioned at assembly board
  - Safety: SG must be placed before MCP → deadlock

Expected recovery (what PDDL should find):
  1. UR5e moves MCP back to prusa-mk4-2
  2. UR5e places MCP at prusa-mk4-2
  3. UR5e moves to pick SG from failed location
  4. UR5e picks SG
  5. UR5e moves SG to assembly board
  6. UR5e places SG on assembly board  ← ordering unlocked
  7. UR5e moves to pick MCP from prusa-mk4-2
  8. UR5e picks MCP
  9. UR5e moves MCP to assembly board
  10. UR5e places MCP on assembly board ← done

Run with:
    python -m cais_spade_llm.agents.intelligent_product.pddl.test_deadlock_scenario
"""

import asyncio
import json

from cais_spade_llm.agents.intelligent_product.pddl.replanner import _parse_pddl_blocks, _translate_plan, replan
from cais_spade_llm.agents.intelligent_product.pddl.solver import solve


# ---------------------------------------------------------------------------
# This is the PDDL the LLM should generate for the deadlock scenario.
# We test it directly to verify the solver finds the correct recovery.
# ---------------------------------------------------------------------------

DEADLOCK_DOMAIN = """\
(define (domain recovery)
  (:requirements :strips :typing :negative-preconditions)
  (:types resource part location - object)
  (:predicates
    (idle ?r - resource)
    (at-source ?r - resource)
    (carrying ?r - resource ?p - part)
    (resource-positioned ?r - resource ?p - part)
    (part-at ?p - part ?l - location)
    (part-placed ?p - part ?l - location)
    (reachable ?r - resource ?l - location)
    (resource-available ?r - resource)
    (placement-allowed ?p - part)
    (placed-flag ?p - part)
  )

  (:action move-to-pick-location
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and (resource-available ?r) (idle ?r) (reachable ?r ?l))
    :effect (and (at-source ?r) (not (idle ?r)))
  )

  (:action pick-part
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and (resource-available ?r) (at-source ?r) (part-at ?p ?l) (reachable ?r ?l))
    :effect (and (carrying ?r ?p) (not (at-source ?r)) (not (part-at ?p ?l)))
  )

  (:action move-loaded-to-destination
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and (resource-available ?r) (carrying ?r ?p) (reachable ?r ?l))
    :effect (and (resource-positioned ?r ?p) (not (carrying ?r ?p)))
  )

  (:action place-part
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and (resource-available ?r) (resource-positioned ?r ?p) (reachable ?r ?l) (placement-allowed ?p))
    :effect (and (part-placed ?p ?l) (placed-flag ?p) (idle ?r) (not (resource-positioned ?r ?p)))
  )

  (:action stage-part
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and (resource-available ?r) (resource-positioned ?r ?p) (reachable ?r ?l))
    :effect (and (part-at ?p ?l) (idle ?r) (not (resource-positioned ?r ?p)))
  )

  (:action unlock-placement
    :parameters (?p1 - part ?p2 - part)
    :precondition (placed-flag ?p1)
    :effect (placement-allowed ?p2)
  )
)
"""

DEADLOCK_PROBLEM = """\
(define (problem deadlock-recovery)
  (:domain recovery)
  (:objects
    ur5e-localhost - resource
    sg mcp - part
    prusa-mk4-2 failed-loc-sg assembly-board-v1 - location
  )
  (:init
    (resource-available ur5e-localhost)

    ; UR5e is positioned with MCP (about to place, but can't because SG first)
    (resource-positioned ur5e-localhost mcp)

    ; SG is misplaced at the failed location (dropped by xArm6)
    (part-at sg failed-loc-sg)

    ; SG must be placed before MCP (ordering constraint)
    (placement-allowed sg)
    ; MCP is NOT placement-allowed yet (needs unlock-placement after SG placed)

    ; Reachability: UR5e can reach all relevant locations
    (reachable ur5e-localhost prusa-mk4-2)
    (reachable ur5e-localhost failed-loc-sg)
    (reachable ur5e-localhost assembly-board-v1)

    ; xArm6 is NOT listed — it's unavailable (no resource-available fact)
  )
  (:goal
    (and
      (part-placed sg assembly-board-v1)
      (part-placed mcp assembly-board-v1)
    )
  )
)
"""

# Tools catalog (what's in tools.json)
TOOLS_CATALOG = [
    {
        "function": "move_to_pick_location",
        "params": {
            "origin_resource_location": {"type": "string"},
            "part_name": {"type": "string"},
            "product_jid": {"type": "string"},
            "task_id": {"type": "string"},
        },
    },
    {
        "function": "pick_part",
        "params": {
            "origin_resource_location": {"type": "string"},
            "part_name": {"type": "string"},
            "product_jid": {"type": "string"},
            "task_id": {"type": "string"},
        },
    },
    {
        "function": "move_loaded_to_destination",
        "params": {
            "destination_location": {"type": "string"},
            "part_name": {"type": "string"},
            "product_jid": {"type": "string"},
            "task_id": {"type": "string"},
        },
    },
    {
        "function": "place_part",
        "params": {
            "destination_location": {"type": "string"},
            "part_name": {"type": "string"},
            "product_jid": {"type": "string"},
            "task_id": {"type": "string"},
        },
    },
]

RESOURCE_INFOS = [
    {
        "jid": "ur5e@localhost",
        "static_capabilities": {
            "reachability": ["prusa-mk4-2", "assembly_board-v1"],
            "staging_areas": {},
        },
    },
    {
        "jid": "xarm6@localhost",
        "static_capabilities": {
            "reachability": ["prusa-mk4-1", "assembly_board-v1"],
            "staging_areas": {},
        },
    },
]

FAILED_PLAN_NODES = [
    {"id": "REQ_1_T1", "status": "completed", "function_name": "move_to_pick_location",
     "params": {"part_name": "SG"}, "resource_jid": "xarm6@localhost",
     "predecessors": [], "successors": ["REQ_1_T2"]},
    {"id": "REQ_1_T2", "status": "completed", "function_name": "pick_part",
     "params": {"part_name": "SG"}, "resource_jid": "xarm6@localhost",
     "predecessors": ["REQ_1_T1"], "successors": ["REQ_1_T3"]},
    {"id": "REQ_1_T3", "status": "completed", "function_name": "move_loaded_to_destination",
     "params": {"part_name": "SG"}, "resource_jid": "xarm6@localhost",
     "predecessors": ["REQ_1_T2"], "successors": ["REQ_1_T4"]},
    {"id": "REQ_1_T4", "status": "failed:misplaced", "function_name": "place_part",
     "params": {"part_name": "SG"}, "resource_jid": "xarm6@localhost",
     "predecessors": ["REQ_1_T3"], "successors": []},
    {"id": "REQ_2_T1", "status": "completed", "function_name": "move_to_pick_location",
     "params": {"part_name": "MCP"}, "resource_jid": "ur5e@localhost",
     "predecessors": [], "successors": ["REQ_2_T2"]},
    {"id": "REQ_2_T2", "status": "completed", "function_name": "pick_part",
     "params": {"part_name": "MCP"}, "resource_jid": "ur5e@localhost",
     "predecessors": ["REQ_2_T1"], "successors": ["REQ_2_T3"]},
    {"id": "REQ_2_T3", "status": "completed", "function_name": "move_loaded_to_destination",
     "params": {"part_name": "MCP"}, "resource_jid": "ur5e@localhost",
     "predecessors": ["REQ_2_T2"], "successors": ["REQ_2_T4"]},
    {"id": "REQ_2_T4", "status": "blocked", "function_name": "place_part",
     "params": {"part_name": "MCP"}, "resource_jid": "ur5e@localhost",
     "predecessors": ["REQ_2_T3", "REQ_1_T4"], "successors": []},
]


def test_solver_finds_recovery():
    """Verify pyperplan finds the correct deadlock recovery sequence."""
    plan = solve(DEADLOCK_DOMAIN, DEADLOCK_PROBLEM)

    print(f"\n  Plan ({len(plan)} steps):")
    for i, (action, params) in enumerate(plan):
        print(f"    {i+1}. {action}({', '.join(params)})")

    # The plan must:
    # 1. First stage/place MCP somewhere (UR5e is holding it, can't pick SG)
    # 2. Then pick SG, place SG on assembly board
    # 3. Then unlock MCP placement
    # 4. Then pick MCP again, place MCP on assembly board

    # Verify critical ordering constraints
    action_names = [a for a, _ in plan]
    action_parts = [(a, p) for a, p in plan]

    # UR5e must free its gripper first (stage MCP)
    first_action = action_names[0]
    assert first_action == "stage-part", (
        f"First action should be stage-part (free gripper), got {first_action}"
    )
    # And it should be staging MCP
    assert "mcp" in plan[0][1], f"First action should involve MCP: {plan[0][1]}"

    # SG must be placed before MCP
    sg_placed_idx = None
    mcp_placed_idx = None
    for i, (action, params) in enumerate(plan):
        if action == "place-part" and "sg" in params:
            sg_placed_idx = i
        if action == "place-part" and "mcp" in params:
            mcp_placed_idx = i

    assert sg_placed_idx is not None, "SG should be placed"
    assert mcp_placed_idx is not None, "MCP should be placed"
    assert sg_placed_idx < mcp_placed_idx, (
        f"SG (step {sg_placed_idx+1}) must be placed before MCP (step {mcp_placed_idx+1})"
    )

    # Final goals should be achieved: both parts placed at assembly board
    # Check that place-part for SG targets assembly-board-v1
    sg_place = plan[sg_placed_idx]
    assert "assembly-board-v1" in sg_place[1], f"SG should be placed at assembly board: {sg_place}"

    mcp_place = plan[mcp_placed_idx]
    assert "assembly-board-v1" in mcp_place[1], f"MCP should be placed at assembly board: {mcp_place}"

    # Only UR5e should be used (xArm6 is unavailable)
    for action, params in plan:
        if params:
            assert "ur5e-localhost" in params[0] or len(params) < 1 or action == "unlock-placement", (
                f"Only UR5e should act, but got: {action}({params})"
            )

    print("  PASS: test_solver_finds_recovery")
    return plan


def test_translate_deadlock_recovery():
    """Verify the translator produces correct DAG patches for the recovery."""
    plan = solve(DEADLOCK_DOMAIN, DEADLOCK_PROBLEM)

    patch = _translate_plan(
        pddl_plan=plan,
        tools_catalog=TOOLS_CATALOG,
        resource_infos=RESOURCE_INFOS,
        product_jid="assembly_board-v1@localhost",
        existing_task_ids={n["id"] for n in FAILED_PLAN_NODES},
        failed_plan_nodes=FAILED_PLAN_NODES,
    )

    tasks = patch["tasks"]
    print(f"\n  DAG patches ({len(tasks)} tasks):")
    for t in tasks:
        fn = t.get("function_name", "(modifier)")
        rid = t.get("resource_jid", "")
        reason = t.get("change_reason", "")
        print(f"    {t['id']}: {fn} on {rid}")
        print(f"      reason: {reason}")
        print(f"      preds: {t.get('predecessors', [])}")

    # All recovery tasks should target ur5e@localhost
    recovery_tasks = [t for t in tasks if t.get("function_name")]
    for t in recovery_tasks:
        assert t["resource_jid"] == "ur5e@localhost", (
            f"Expected ur5e@localhost, got {t['resource_jid']}"
        )

    # Predecessor chain should be sequential
    for i in range(1, len(recovery_tasks)):
        assert recovery_tasks[i]["predecessors"] == [recovery_tasks[i-1]["id"]], (
            f"Task {i} should have task {i-1} as predecessor"
        )

    # All task IDs should be unique and not collide with existing
    existing_ids = {n["id"] for n in FAILED_PLAN_NODES}
    for t in tasks:
        if t["id"].startswith("RECOVERY_"):
            assert t["id"] not in existing_ids, f"ID collision: {t['id']}"

    # Check that blocked task REQ_2_T4 gets rewired
    rewire_tasks = [t for t in tasks if t["id"] == "REQ_2_T4"]
    if rewire_tasks:
        rewire = rewire_tasks[0]
        assert "REQ_1_T4" not in rewire["predecessors"], (
            "REQ_2_T4 should no longer depend on failed REQ_1_T4"
        )
        print(f"\n  REQ_2_T4 rewired: predecessors={rewire['predecessors']}")

    print("  PASS: test_translate_deadlock_recovery")


async def test_full_pipeline_with_mock_llm():
    """
    Test the full replan() function with a mock LLM that returns
    pre-built PDDL for the deadlock scenario.
    """
    mock_response = f"```pddl\n{DEADLOCK_DOMAIN}\n```\n\n```pddl\n{DEADLOCK_PROBLEM}\n```"

    async def mock_ask_llm(prompt, with_functions=False, temperature=0.0):
        return mock_response

    violations = [
        {
            "type": "task_failed",
            "failed_task_id": "REQ_1_T4",
            "failure_context": {
                "failure_mode": "slippage",
                "severity": "high",
                "retryable": False,
                "affected_entities": [
                    {"entity_type": "part", "entity_id": "SG", "state": "misplaced"}
                ],
            },
        }
    ]

    system_state = {
        "robots": {
            "xarm6@localhost": {
                "held_part": None,
                "current_state": "recovery_required",
            },
            "ur5e@localhost": {
                "held_part": "MCP",
                "current_state": "positioned",
            },
        },
        "parts": {
            "SG": {
                "state": "misplaced",
                "position": {"x": 750.0, "y": -200.0, "z": 50.0},
            },
            "MCP": {
                "state": "in_transit",
                "location": "ur5e@localhost_gripper",
            },
        },
    }

    result = await replan(
        ask_llm=mock_ask_llm,
        violations=violations,
        system_state=system_state,
        tools_catalog=TOOLS_CATALOG,
        resource_infos=RESOURCE_INFOS,
        safety_text="SG must be placed before MCP",
        failed_plan_nodes=FAILED_PLAN_NODES,
        product_jid="assembly_board-v1@localhost",
        existing_task_ids={n["id"] for n in FAILED_PLAN_NODES},
    )

    assert result is not None, "replan() should return a patch"
    tasks = result["tasks"]
    assert len(tasks) > 0, "Should have recovery tasks"

    print(f"\n  Full pipeline produced {len(tasks)} tasks:")
    for t in tasks:
        fn = t.get("function_name", "(modifier)")
        print(f"    {t['id']}: {fn} — {t.get('change_reason', '')}")

    print("  PASS: test_full_pipeline_with_mock_llm")


if __name__ == "__main__":
    print("Running deadlock scenario tests...")
    test_solver_finds_recovery()
    test_translate_deadlock_recovery()
    asyncio.run(test_full_pipeline_with_mock_llm())
    print("\nAll deadlock scenario tests passed.")

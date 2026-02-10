"""
Smoke test: feed the exact system_state from the debug file and verify
the planner produces the correct recovery sequence.

Run with:
    python -m cais_spade_llm.pddl.test_pddl_smoke
"""

import json
from cais_spade_llm.pddl.pddl_domain import get_domain
from cais_spade_llm.pddl.pddl_problem import build_problem
from cais_spade_llm.pddl.pddl_planner import solve
from cais_spade_llm.pddl.pddl_translator import translate

# ── Exact state from the debug file ────────────────────────────────────────
SYSTEM_STATE = {
    "robots": {
        "xarm6@localhost": {
            "held_part": None,
            "current_state": "recovery_required",
            "position": {"x": 400.0, "y": -200.0, "z": 200.0},
            "gripper_state": "open",
        },
        "ur5e@localhost": {
            "held_part": "MCP",
            "current_state": "positioned",
            "position": {"x": 400.0, "y": -200.0, "z": 200.0},
            "gripper_state": "closed",
        },
    },
    "parts": {
        "SG": {
            "state": "misplaced",
            "position": {"x": 750.0, "y": -200.0, "z": 50.0},
            "last_known_location": "assembly_board-v1",
        },
        "MCP": {
            "state": "in_transit",
            "location": "ur5e@localhost_gripper",
        },
    },
}

RESOURCE_INFOS = [
    {
        "jid": "xarm6@localhost",
        "static_capabilities": {
            "reachability": ["prusa-mk4-1", "prusa-mk3", "assembly_board-v1"],
            "workspace_boundaries": {
                "x_range": [-150, 650], "y_range": [-550, 100], "z_range": [0, 600]
            },
            "staging_areas": {
                "staging_zone_1": {"x": 200, "y": -250, "z": 172, "accessible_by": ["xarm6", "ur5e"]},
                "staging_zone_neutral": {"x": 300, "y": -150, "z": 172, "accessible_by": ["xarm6", "ur5e"]},
            },
        },
    },
    {
        "jid": "ur5e@localhost",
        "static_capabilities": {
            "reachability": ["prusa-mk4-2", "prusa-mk3", "assembly_board-v1"],
            "workspace_boundaries": {
                "x_range": [100, 900], "y_range": [-450, 200], "z_range": [0, 600]
            },
            "staging_areas": {
                "staging_zone_2": {"x": 500, "y": -100, "z": 172, "accessible_by": ["ur5e", "xarm6"]},
                "staging_zone_neutral": {"x": 300, "y": -150, "z": 172, "accessible_by": ["xarm6", "ur5e"]},
            },
        },
    },
]

PENDING_GOALS = [
    {"part": "SG",  "destination": "assembly_board-v1"},
    {"part": "MCP", "destination": "assembly_board-v1"},
]

SAFETY_RULES = [
    {
        "id": "SAFE_1",
        "constraint_type": "ordering_place_before",
        "product": ["sg", "mcp"],
        "event": "place_part",
    }
]


if __name__ == "__main__":
    domain = get_domain()

    problem = build_problem(
        system_state=SYSTEM_STATE,
        resource_infos=RESOURCE_INFOS,
        pending_goals=PENDING_GOALS,
        safety_rules=SAFETY_RULES,
    )

    print("=== DOMAIN ===")
    print(domain)
    print("=== PROBLEM ===")
    print(problem)

    plan = solve(domain, problem)

    print("=== PLAN ===")
    for i, (action, params) in enumerate(plan):
        print(f"  {i+1}. {action}({', '.join(params)})")

    robot_jid_map = {
        "xarm6-localhost": "xarm6@localhost",
        "ur5e-localhost":  "ur5e@localhost",
    }
    part_name_map = {"sg": "SG", "mcp": "MCP"}
    location_map  = {
        "assembly-board-v1":   "assembly_board-v1",
        "failed-loc-sg":       "assembly_board-v1_failed",
        "staging-zone-neutral": "staging_zone_neutral",
        "staging-zone-1":      "staging_zone_1",
        "staging-zone-2":      "staging_zone_2",
    }

    result = translate(
        plan,
        robot_jid_map=robot_jid_map,
        part_name_map=part_name_map,
        location_map=location_map,
        product_jid="assembly_board-v1@localhost",
        failed_task_id="REQ_1_T4",
        existing_task_ids={"REQ_1_T1","REQ_1_T2","REQ_1_T3","REQ_1_T4",
                           "REQ_2_T1","REQ_2_T2","REQ_2_T3","REQ_2_T4"},
    )

    print("\n=== TASK DAG PATCHES ===")
    print(json.dumps(result, indent=2))

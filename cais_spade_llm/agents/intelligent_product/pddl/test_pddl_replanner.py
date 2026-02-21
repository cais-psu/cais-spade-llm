"""
Tests for the generic PDDL replanner.

Deterministic tests — no LLM needed. Tests the solver, parser, and translator.

Run with:
    python -m cais_spade_llm.agents.intelligent_product.pddl.test_pddl_replanner
"""

from cais_spade_llm.agents.intelligent_product.pddl.replanner import _parse_pddl_blocks, _translate_plan
from cais_spade_llm.agents.intelligent_product.pddl.solver import solve


# ---------------------------------------------------------------------------
# Test: PDDL block parsing from mock LLM response
# ---------------------------------------------------------------------------

def test_parse_pddl_blocks():
    mock_response = '''```pddl
(define (domain recovery)
  (:requirements :strips :typing :negative-preconditions)
  (:types resource part location - object)
  (:predicates
    (idle ?r - resource)
    (carrying ?r - resource ?p - part)
    (part-at ?p - part ?l - location)
    (reachable ?r - resource ?l - location)
    (resource-available ?r - resource)
    (at-source ?r - resource)
    (part-placed ?p - part ?l - location)
  )
  (:action move-to-pick-location
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and (resource-available ?r) (idle ?r) (reachable ?r ?l))
    :effect (and (at-source ?r) (not (idle ?r)))
  )
)
```

```pddl
(define (problem test-problem)
  (:domain recovery)
  (:objects ur5e-localhost - resource sg - part loc1 - location)
  (:init (idle ur5e-localhost) (resource-available ur5e-localhost))
  (:goal (part-placed sg loc1))
)
```'''

    domain, problem = _parse_pddl_blocks(mock_response)

    assert domain is not None, "Domain should be extracted"
    assert "(define (domain recovery)" in domain
    assert problem is not None, "Problem should be extracted"
    assert "(define (problem test-problem)" in problem
    print("  PASS: test_parse_pddl_blocks")


# ---------------------------------------------------------------------------
# Test: Solve known PDDL + translate to DAG patches
# ---------------------------------------------------------------------------

DOMAIN = """\
(define (domain recovery)
  (:requirements :strips :typing :negative-preconditions)
  (:types resource part location - object)
  (:predicates
    (idle ?r - resource)
    (carrying ?r - resource ?p - part)
    (part-at ?p - part ?l - location)
    (reachable ?r - resource ?l - location)
    (resource-available ?r - resource)
    (at-source ?r - resource)
    (resource-positioned ?r - resource ?p - part)
    (part-placed ?p - part ?l - location)
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
  (:action assemble-part
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and (resource-available ?r) (resource-positioned ?r ?p) (reachable ?r ?l))
    :effect (and (part-placed ?p ?l) (idle ?r) (not (resource-positioned ?r ?p)))
  )
)
"""

PROBLEM = """\
(define (problem recovery-instance)
  (:domain recovery)
  (:objects
    ur5e-localhost - resource
    sg - part
    failed-loc-sg assembly-board-v1 - location
  )
  (:init
    (resource-available ur5e-localhost)
    (idle ur5e-localhost)
    (part-at sg failed-loc-sg)
    (reachable ur5e-localhost failed-loc-sg)
    (reachable ur5e-localhost assembly-board-v1)
  )
  (:goal (part-placed sg assembly-board-v1))
)
"""


def test_solve():
    plan = solve(DOMAIN, PROBLEM)

    assert len(plan) == 4, f"Expected 4 actions, got {len(plan)}"
    assert plan[0][0] == "move-to-pick-location"
    assert plan[1][0] == "pick-part"
    assert plan[2][0] == "move-loaded-to-destination"
    assert plan[3][0] == "assemble-part"

    # Verify parameters
    for _, params in plan:
        assert "ur5e-localhost" in params, f"Robot should be in params: {params}"
        assert "sg" in params, f"Part should be in params: {params}"

    print("  PASS: test_solve")
    return plan


def test_translate():
    plan = solve(DOMAIN, PROBLEM)

    tools_catalog = [
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
            "function": "assemble_part",
            "params": {
                "destination_location": {"type": "string"},
                "part_name": {"type": "string"},
                "product_jid": {"type": "string"},
                "task_id": {"type": "string"},
            },
        },
    ]

    resource_infos = [
        {
            "jid": "ur5e@localhost",
            "static_capabilities": {
                "reachability": ["assembly_board-v1"],
                "staging_areas": {},
            },
        },
    ]

    patch = _translate_plan(
        pddl_plan=plan,
        tools_catalog=tools_catalog,
        resource_infos=resource_infos,
        product_jid="assembly_board-v1@localhost",
        existing_task_ids={"REQ_1_T1", "REQ_1_T2", "REQ_1_T3", "REQ_1_T4"},
        failed_plan_nodes=[
            {
                "id": "REQ_1_T4",
                "status": "failed",
                "function_name": "assemble_part",
                "params": {"part_name": "SG"},
                "predecessors": ["REQ_1_T3"],
                "successors": [],
            },
        ],
    )

    tasks = patch["tasks"]
    assert len(tasks) >= 4, f"Expected at least 4 tasks, got {len(tasks)}"

    # Verify function names
    assert tasks[0]["function_name"] == "move_to_pick_location"
    assert tasks[1]["function_name"] == "pick_part"
    assert tasks[2]["function_name"] == "move_loaded_to_destination"
    assert tasks[3]["function_name"] == "assemble_part"

    # Verify resource JID mapping
    for t in tasks[:4]:
        assert t["resource_jid"] == "ur5e@localhost", (
            f"Expected ur5e@localhost, got {t['resource_jid']}"
        )

    # Verify predecessor chain
    assert tasks[0]["predecessors"] == []
    assert tasks[1]["predecessors"] == [tasks[0]["id"]]
    assert tasks[2]["predecessors"] == [tasks[1]["id"]]
    assert tasks[3]["predecessors"] == [tasks[2]["id"]]

    # Verify successor chain
    assert tasks[0]["successors"] == [tasks[1]["id"]]
    assert tasks[1]["successors"] == [tasks[2]["id"]]
    assert tasks[2]["successors"] == [tasks[3]["id"]]

    # Verify params
    assert tasks[0]["params"]["part_name"] == "SG"
    assert tasks[0]["params"]["product_jid"] == "assembly_board-v1@localhost"
    assert "origin_resource_location" in tasks[0]["params"]
    assert "destination_location" in tasks[3]["params"]

    # Verify unique IDs
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "Task IDs should be unique"

    print("  PASS: test_translate")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running PDDL replanner tests...")
    test_parse_pddl_blocks()
    test_solve()
    test_translate()
    print("All tests passed.")

from cais_spade_llm.agents.intelligent_product.replanner.des_search.resource_bidding import (
    _tool_signature,
    compute_bid,
)


def _recovery_tools() -> list[dict]:
    return [
        {
            "function": "move_home",
            "function_owner_agent": "ur5e",
            "process": "assembly",
            "resource_type": "robot",
            "in_state": "any",
            "out_state": "idle",
            "context": [],
            "params": {},
            "description": "Robot arm move to its home position.",
        },
        {
            "function": "place_approach",
            "function_owner_agent": "ur5e",
            "process": "assembly",
            "resource_type": "robot",
            "in_state": "picked",
            "out_state": "positioned",
            "part_in_state": "in_gripper",
            "context_mapping": {
                "location_param": "destination_location",
                "location_type": "reachable_location",
            },
            "part_transition": {
                "completed": {
                    "state": "in_transit",
                    "location_template": "{resource_jid}_gripper",
                }
            },
            "params": {},
            "description": "Move the loaded part to its destination location.",
        },
        {
            "function": "place_insert",
            "function_owner_agent": "ur5e",
            "process": "assembly",
            "resource_type": "robot",
            "in_state": "positioned",
            "out_state": "placed",
            "part_in_state": "in_transit",
            "context_mapping": {
                "location_param": "destination_location",
                "location_type": "current_location",
            },
            "part_transition": {
                "completed": {
                    "state": "assembled",
                    "location_param": "destination_location",
                }
            },
            "params": {},
            "description": "Assemble the currently held part at its final destination.",
        },
    ]


def test_compute_bid_reaches_catalog_recovery_from_projected_suffix_state():
    tools = _recovery_tools()
    move_home = next(tool for tool in tools if tool["function"] == "move_home")
    bid = compute_bid(
        x_c={
            "resource_state": "placed",
            "current_part": None,
            "current_location": "Assembly Station",
            "part_states": {"MCP": "assembled"},
            "part_locations": {"MCP": "Assembly Station"},
        },
        P_id=[],
        goal_state="assembled",
        tools=tools,
        reachability=["Assembly Station"],
        staging_areas={},
        resource_jid="ur5e@localhost",
        goal_event_signatures={_tool_signature(move_home)},
    )

    assert bid is not None
    assert bid.complete is True
    assert [event["function_name"] for event in bid.str_e] == ["move_home"]
    assert bid.str_x[-1]["resource_state"] == "idle"
    assert bid.str_x[-1]["current_part"] is None

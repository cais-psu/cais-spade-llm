import json
import asyncio
from cais_spade_llm.agents.intelligent_product.replanner.des_search.resource_bidding import (
    compute_bid,
)

def run_test():
    with open("cais_spade_llm/initialization/tools.json") as f:
        tools = json.load(f)

    # Mimic a simple valid start state
    x_c = {
        "resource_state": "idle",
        "current_part": None,
        "current_location": None,
        "part_states": {"sg": "ready", "mcp": "ready"},
        "part_locations": {"sg": "prusa-mk4-1", "mcp": "prusa-mk4-2"}
    }
    
    P_id = ["sg"]
    reachability = ["prusa-mk4-1", "assembly_board-v1"]
    staging_areas = {}
    resource_jid = "xarm6@localhost"

    bid = compute_bid(
        x_c=x_c,
        P_id=P_id,
        goal_state="assembled",
        tools=tools,
        reachability=reachability,
        staging_areas=staging_areas,
        resource_jid=resource_jid
    )

    if bid and bid.complete:
        print("SUCCESS! Found complete bid sequence:")
        for e in bid.str_e:
            print(f"  {e['function_name']}: {e['params']}")
    else:
        print("FAILED to find valid sequence")

if __name__ == "__main__":
    run_test()

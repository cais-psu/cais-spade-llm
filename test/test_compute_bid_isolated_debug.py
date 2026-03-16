import json
import asyncio
from cais_spade_llm.agents.intelligent_product.replanner.des_search.resource_bidding import (
    _expand,
    compute_bid,
)

def run_test():
    with open("cais_spade_llm/initialization/tools.json") as f:
        tools = json.load(f)

    # Mimic a simple valid start state
    P_id = ["sg"]
    reachability = ["prusa-mk4-1", "assembly_board-v1"]
    staging_areas = {}
    resource_jid = "xarm6@localhost"

    robot_tools = [t for t in tools if t.get("function_owner_agent") == "xarm6"]
    
    print("DEBUGGING IDLE MATCHES:")
    for tool in robot_tools:
        if tool.get("in_state") == "idle":
            print(f"Tool {tool['function']} matches idle")
            ctx_map = tool.get("context_mapping", {})
            loc_param = ctx_map.get("location_param")
            loc_type = ctx_map.get("location_type")
            part_in = tool.get("part_in_state")
            print(f"  ctx_map: {ctx_map}, loc_type: {loc_type}, part_in: {part_in}")
            
if __name__ == "__main__":
    run_test()

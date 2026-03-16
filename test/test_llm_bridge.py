import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

async def mock_ask_llm(prompt: str, **kwargs) -> str:
    from openai import OpenAI
    client = OpenAI()
    
    def _call():
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="low"
        )
        return response.choices[0].message.content or ""
        
    return await asyncio.to_thread(_call)

async def run_test():
    from cais_spade_llm.agents.intelligent_product.replanner.des_search.resource_bidding import (
        Bid,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.des_search.environment_model import (
        compile_environment_model,
    )
    from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_generation import (
        llm_explore_states_and_events,
    )

    # 1. State setup based on the deadlock
    #    UR5e is holding MCP, but SG slipped and is misplaced at a known XYZ coordinate.
    stuck_state = {
        "resource_state": "holding",
        "current_part": "MCP",
        "current_location": "assembly_board-v1",
        "part_states": {"SG": "misplaced", "MCP": "picked"},
        "part_locations": {"SG": "ur5e_workspace", "MCP": "ur5e_gripper"}
    }
    P_id = ["SG", "MCP"]
    ra_jid = "ur5e@localhost"
    goal_state = "assembled"
    part_tracker = {
        "SG": {
            "state": "misplaced",
            "location": {
                "x": 150.5,
                "y": -20.0,
                "z": 5.0,
                "zone": "ur5e_workspace",
                "description": "Dropped on workspace surface after slippage"
            }
        },
        "MCP": {"state": "picked", "location": "ur5e_gripper"}
    }

    # Only the 4 REAL robot functions — no invented ones.
    # The LLM must figure out how to use these (or invent new ones) to recover.
    tools_catalog = [
        {
            "function_name": "pick_approach",
            "process": "assembly",
            "resource_type": "robot",
            "in_state": "idle",
            "out_state": "at_pick",
            "params": {
                "origin_resource_location": {"type": "string", "description": "Target origin location to approach for picking."},
                "part_name": {"type": "string", "description": "Name of the part intended to be picked."},
                "speed": {"type": "number", "description": "Optional motion speed."}
            },
            "description": "Move empty gripper to the part's origin location."
        },
        {
            "function_name": "pick_grasp",
            "process": "assembly",
            "resource_type": "robot",
            "in_state": "at_pick",
            "out_state": "picked",
            "part_in_state": "ready",
            "part_transition": {"completed": {"state": "in_gripper", "location_template": "{resource_jid}_gripper"}},
            "params": {
                "part_name": {"type": "string", "description": "Name of the part to pick."},
                "origin_resource_location": {"type": "string", "description": "Origin location of the part (printer or fixture)."},
                "gripper": {"type": "string", "description": "Optional gripper configuration."}
            },
            "description": "Pick a ready part from an origin location."
        },
        {
            "function_name": "place_approach",
            "process": "assembly",
            "resource_type": "robot",
            "in_state": "picked",
            "out_state": "positioned",
            "part_in_state": "in_gripper",
            "part_transition": {"completed": {"state": "in_transit", "location_template": "{resource_jid}_gripper"}},
            "params": {
                "destination_location": {"type": "string", "description": "Destination location to carry the loaded part."},
                "part_name": {"type": "string", "description": "Name of the part being moved."},
                "speed": {"type": "number", "description": "Optional motion speed while loaded."}
            },
            "description": "Move the loaded part to its destination location."
        },
        {
            "function_name": "place_insert",
            "process": "assembly",
            "resource_type": "robot",
            "in_state": "positioned",
            "out_state": "idle",
            "part_in_state": "in_transit",
            "part_transition": {"completed": {"state": "assembled", "location_param": "destination_location"}},
            "params": {
                "destination_location": {"type": "string", "description": "Final assembly location for the part."},
                "part_name": {"type": "string", "description": "Name of the part being assembled."},
                "orientation": {"type": "string", "description": "Optional placement orientation."}
            },
            "description": "Assemble the currently held part at its final destination."
        }
    ]

    resource_infos = [
        {
            "jid": "ur5e@localhost",
            "static_capabilities": {
                "reachability": ["assembly_board-v1", "prusa-mk4-1", "prusa-mk4-2"],
                "staging_areas": {
                    "prusa-mk4-2": {"x": 300.0, "y": 100.0, "z": 50.0, "accessible_by": ["ur5e"]}
                }
            }
        }
    ]
    
    # 2. Call llm_explore_states_and_events (hits real LLM)
    print("Sending prompt to LLM...")
    bridge_tools = await llm_explore_states_and_events(
        stuck_state=stuck_state,
        P_id=P_id,
        ra_jid=ra_jid,
        ask_llm=mock_ask_llm,
        goal_state=goal_state,
        tools_catalog=tools_catalog,
        resource_infos=resource_infos,
        part_tracker=part_tracker,
    )
    
    print(f"LLM returned {len(bridge_tools)} bridge tool(s):")
    print(json.dumps(bridge_tools, indent=2))
    
    # 3. Simulate ProcessPlanner's bridge injection
    bridge_bid = Bid(
        request_id="bridge",
        ra_jid=ra_jid,
        str_e=bridge_tools,
        str_x=[stuck_state] + [
            {
                "resource_state": t.get("out_state", stuck_state.get("resource_state")),
                "part_states": {
                    **stuck_state.get("part_states", {}),
                    **{k: v.get("state") for k, v in t.get("part_effect", {}).items()},
                },
                "part_locations": {
                    **stuck_state.get("part_locations", {}),
                    **{k: v.get("location") for k, v in t.get("part_effect", {}).items()},
                },
            }
            for t in bridge_tools
        ],
        prp_p_achieved=[],
        complete=False,
    )
    
    # 4. Compile the environment model from the synthetic bid
    M_e = compile_environment_model([bridge_bid])
    
    # 5. Output
    dump_data = {
        "01_Initial_Stuck_State": stuck_state,
        "02_LLM_Bridge_Events_Generated": bridge_tools,
        "03_Synthetic_Bid_States (str_x)": bridge_bid.str_x,
        "04_Compiled_Environment_Model (M_e)": M_e
    }
    
    out_dir = Path("cais_spade_llm/monitor/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "llm_bridge_test_output.json"
    
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(dump_data, f, indent=2)
        
    print(f"\nTest complete. Output written to {out_file.resolve()}")

if __name__ == "__main__":
    asyncio.run(run_test())

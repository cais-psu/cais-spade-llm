import json

def patch():
    with open("cais_spade_llm/initialization/tools.json", "r") as f:
        tools = json.load(f)
        
    for t in tools:
        fn = t.get("function")
        if fn == "pick_part":
            t["context_mapping"] = {"location_param": "origin_resource_location", "location_type": "current_location"}
        elif fn == "move_to_pick_location":
            t["context_mapping"] = {"location_param": "origin_resource_location", "location_type": "part_location"}
        elif fn == "move_loaded_to_destination":
            t["context_mapping"] = {"location_param": "destination_location", "location_type": "reachable_location"}
        elif fn == "assemble_part":
            t["context_mapping"] = {"location_param": "destination_location", "location_type": "current_location"}

    with open("cais_spade_llm/initialization/tools.json", "w") as f:
        json.dump(tools, f, indent=2)
        
    print("Patched tools.json successfully")

if __name__ == "__main__":
    patch()

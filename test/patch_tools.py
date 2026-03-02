import json

def patch():
    with open("cais_spade_llm/initialization/tools.json", "r") as f:
        tools = json.load(f)
        
    for t in tools:
        fn = t.get("function")
        if fn == "pick_grasp":
            t["context_mapping"] = {"location_param": "origin_resource_location", "location_type": "current_location"}
        elif fn == "pick_approach":
            t["context_mapping"] = {"location_param": "origin_resource_location", "location_type": "part_location"}
        elif fn == "place_approach":
            t["context_mapping"] = {"location_param": "destination_location", "location_type": "reachable_location"}
        elif fn == "place_insert":
            t["context_mapping"] = {"location_param": "destination_location", "location_type": "current_location"}

    with open("cais_spade_llm/initialization/tools.json", "w") as f:
        json.dump(tools, f, indent=2)
        
    print("Patched tools.json successfully")

if __name__ == "__main__":
    patch()

# TODO: ProtoTwin Multi-Robot Collaborative Assembly Demo

## Goal

Full pipeline demo with 2 robots doing collaborative assembly in ProtoTwin simulation, with camera integration, ready for lab deployment.

```
NL requirements → Plan → Execute (simulated) → Inject fault → DES+LLM replan → Recover
```

## Current Infrastructure (already built)

| Component | Status | Location |
|---|---|---|
| UR5e config (IP, workspace, cameras, grasp strategies) | ✅ | `initialization/resources/robot_ur5e.json` |
| xArm6 config (IP, workspace, cameras, grasp strategies) | ✅ | `initialization/resources/robot_xarm6.json` |
| Robot agent (pick/move/place/assemble) | ✅ | `agents/resource_agent/robot_agent.py` |
| Product agent + process planner | ✅ | `agents/intelligent_product/` |
| CCA safety controller | ✅ | CCA agent + `safety/` DFAs |
| Camera module (mock + real integration point) | ✅ | `sensors/camera_module.py` |
| DES bidding + environment model | ✅ | `replanner/resource_bidding.py`, `environment_model.py` |
| LLM bridge for recovery | ✅ | `environment_model.py` |
| Tools catalog (both robots) | ✅ | `initialization/tools.json` |
| SG slippage fault injection | ✅ | `robot_agent.py` (configurable per robot) |
| `spade_main.py` orchestrator | ✅ | Entry point, spawns all agents |

## Scenario Design

### Assembly task
- **Product**: Assembly board with SG + MCP parts
- **xArm6**: Picks SG from `prusa-mk4-1`, moves to `assembly_board-v1`, assembles
- **UR5e**: Picks MCP from `prusa-mk4-2`, moves to `assembly_board-v1`, assembles
- **Safety**: SG must be assembled before MCP (`SAFE_1`)

### Fault injection
- xArm6 has `sg_slippage_mode: "once"` → SG slips on first placement
- Camera detects SG at unexpected XYZ in xArm6 workspace
- DES BFS finds no path → LLM bridge generates recovery → re-pick SG from coordinates → retry assembly

## Implementation Steps

### Phase 1: ProtoTwin Simulation Setup
- [ ] Set up ProtoTwin environment with 2 robot models (UR5e + xArm6)
- [ ] Define assembly station and printer locations matching `static_capabilities` in configs
- [ ] Create simulated parts (SG, MCP) with pick/place physics
- [ ] Implement ProtoTwin ↔ SPADE communication interface (REST API or similar)

### Phase 2: Camera Integration
- [ ] Replace `CameraModule.observe()` mock with ProtoTwin vision query
- [ ] Map ProtoTwin world coordinates to robot base frame coordinates
- [ ] Test camera detection for normal placement and slippage scenarios

### Phase 3: End-to-End Pipeline
- [ ] Run full scenario: NL → plan → execute in ProtoTwin → verify via camera
- [ ] Trigger SG slippage → watch replanning → verify recovery in ProtoTwin
- [ ] Record demo video showing the full pipeline

### Phase 4: Lab Deployment Prep
- [ ] Replace ProtoTwin API calls with real robot SDK calls (UR5e RTDE, xArm Python SDK)
- [ ] Connect real cameras (cameraA, cameraB per robot config)
- [ ] Test with real hardware on physical assembly station

## Key Questions to Resolve
1. What is the ProtoTwin API for sending robot commands and reading positions?
2. Does ProtoTwin have a camera/vision simulation, or do we simulate camera data from object positions?
3. What format does ProtoTwin expect for robot movement commands (joint space vs Cartesian)?

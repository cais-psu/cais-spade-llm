# TODO: Connect ROS2/MoveIt to Robot Controllers & Agent

## Goal

Bridge the gap between the SPADE multi-agent system and the ROS2/MoveIt/Gazebo stack so that `RobotAgent` can drive real or simulated robots through MoveIt's collision-aware planning, instead of just sleeping.

```
RobotAgent (SPADE)
  → Backend abstraction (mock / ros2)
    → MockBackend: asyncio.sleep (current behavior, no ROS2)
    → MoveItBackend: MoveGroup action client → plan + execute (Gazebo or real hardware)
```

## Current State

| Component | Status | Location |
|---|---|---|
| UR5e controller (raw trajectory publishing) | ⚠️ Bypasses MoveIt | `resources/robot/ur5e_controller.py` |
| xArm6 controller | ⚠️ Placeholder | `resources/robot/xarm6_controller.py` |
| RobotAgent (pick/move/place/assemble) | ⚠️ All actions simulated with `asyncio.sleep` | `agents/resource_agent/robot_agent.py` |
| MoveIt action client for UR5e arm | ✅ Tested | `test/test_ur5e_moveit_action.py` |
| MoveIt action client for UR5e gripper | ✅ Tested | `test/test_ur5e_rg2_gripper_action.py` |
| Dual-robot Gazebo + MoveIt launch | ✅ Working | `ros2/xarm_gazebo/launch/dual_moveit_gazebo.launch.py` |
| Robot JSON configs (workspace, home, gripper) | ✅ | `initialization/resources/robot_*.json` |

## Design Decision: Two Modes, Not Three

| Mode | What | When to use | ROS2 needed? |
|------|------|-------------|-------------|
| **mock** | Pure-Python simulation (sleep + log) | Testing SPADE agent logic, LLM planning, safety validation | No |
| **ros2** | MoveIt action client → plan + execute | Controlling robots (Gazebo OR real hardware) | Yes |

Gazebo vs real hardware is **not** a Python code concern — it depends on which ROS2 launch file you start. The MoveIt action client sends identical goals either way. A "digital twin" (Gazebo mirroring real hardware simultaneously) can be added later as a ROS2 launch configuration without changing Python code.

---

## Phase 1: Backend Abstraction (no ROS2 needed)

- [ ] **Create `resources/robot/robot_backend.py`** — ABC + MockBackend
  - `RobotBackend` abstract class with methods:
    - `move_to_joint_positions(positions, velocity_scaling) → {success, message}`
    - `open_gripper() → {success}`
    - `close_gripper() → {success}`
    - `get_joint_positions() → list[float] | None`
    - `move_to_named_position(name) → {success}`
    - `shutdown()`
  - `MockBackend` implements all with `asyncio.sleep()` + logging (extracts current `_simulate_action` logic)

- [ ] **Create `resources/robot/position_registry.py`** — Named position mapping
  - Maps semantic location names (`"home"`, `"prusa-mk4-2"`) → joint-space values (radians)
  - Loaded from `named_positions` section in robot JSON configs
  - Needs actual joint values recorded from RViz (see Phase 3)

- [ ] **Create `resources/robot/backend_factory.py`** — Factory function
  - Reads `control_mode` from config, returns `MockBackend` or `MoveItBackend`
  - Lazy-imports `MoveItBackend` so mock mode works without ROS2 installed

- [ ] **Modify `agents/resource_agent/robot_agent.py`** — Use backend instead of `_simulate_action`
  - Add `_backend: RobotBackend` attribute, created via factory in `__init__`
  - Action method dispatch:
    - `move_to_pick_location` → `_backend.move_to_named_position()` or `move_to_joint_positions()`
    - `pick_part` → `_backend.close_gripper()`
    - `move_loaded_to_destination` → `_backend.move_to_joint_positions()`
    - `assemble_part` → `_backend.open_gripper()`
    - `move_home` → `_backend.move_to_named_position("home")`
  - Update `_position` from `_backend.get_joint_positions()` after each move

- [ ] **Modify `agent_creator.py`** — Pass `robot_config=meta` through to `RobotAgent`

- [ ] **Update robot JSON configs** — Add `control_mode`, `ros2` section, `named_positions`

  **robot_ur5e.json** additions:
  ```json
  "control_mode": "mock",
  "ros2": {
    "arm_group_name": "ur5e_ur_manipulator",
    "gripper_group_name": "ur5e_rg2_gripper",
    "arm_joint_names": ["ur5e_shoulder_pan_joint", "ur5e_shoulder_lift_joint",
                         "ur5e_elbow_joint", "ur5e_wrist_1_joint",
                         "ur5e_wrist_2_joint", "ur5e_wrist_3_joint"],
    "gripper_joint_name": "ur5e_rg2_finger_width",
    "gripper_open_value": 0.11,
    "gripper_close_value": 0.0
  },
  "named_positions": {
    "home": [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
  }
  ```

  **robot_xarm6.json** additions:
  ```json
  "control_mode": "mock",
  "ros2": {
    "arm_group_name": "xarm6_xarm6",
    "gripper_group_name": "xarm6_xarm_gripper",
    "arm_joint_names": ["xarm6_joint1", "xarm6_joint2", "xarm6_joint3",
                         "xarm6_joint4", "xarm6_joint5", "xarm6_joint6"],
    "gripper_joint_name": "xarm6_drive_joint",
    "gripper_open_value": 0.85,
    "gripper_close_value": 0.0
  },
  "named_positions": {
    "home": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  }
  ```

- [ ] **Verify**: Run `python -m cais_spade_llm.spade_main` with `control_mode: "mock"` — behavior identical to current

---

## Phase 2: MoveIt Backend (requires ROS2 environment)

- [ ] **Create `resources/robot/moveit_backend.py`** — Implements `RobotBackend` using MoveIt action client
  - Replicates pattern from `test/test_ur5e_moveit_action.py` and `test/test_ur5e_rg2_gripper_action.py`
  - Creates `rclpy.Node` + `ActionClient(MoveGroup, '/move_action')`
  - Spins ROS2 node in a **daemon thread** (SPADE asyncio + rclpy in same thread = deadlock)
  - Bridges async↔sync via `loop.run_in_executor()`
  - Handles action server readiness with timeout (MoveIt takes ~40s to start)
  - Subscribes to `/joint_states` for position feedback (filters by robot prefix in dual setup)

- [ ] **Test against Gazebo+MoveIt**: Launch dual robot stack, set `control_mode: "ros2"`, run SPADE, verify robot moves
- [ ] **Test gripper**: Verify open/close through pick-and-place sequence

---

## Phase 3: Position Teaching (follow-up)

- [ ] **Record named positions**: Jog each robot in RViz, read joint values from `/joint_states`, save to JSON
  - Minimum: `home`, approach positions for each printer/fixture, assembly board above
  - Optionally build a CLI tool to automate this

- [ ] **Cartesian pose support**: Add `move_to_cartesian_pose()` to backend using MoveIt pose constraints

---

## Files Summary

### New files
| File | Purpose |
|---|---|
| `resources/robot/robot_backend.py` | ABC + MockBackend |
| `resources/robot/moveit_backend.py` | MoveIt action client backend |
| `resources/robot/position_registry.py` | Location name → joint values |
| `resources/robot/backend_factory.py` | Factory function |

### Modified files
| File | Change |
|---|---|
| `agents/resource_agent/robot_agent.py` | Use backend instead of `_simulate_action` |
| `agent_creator.py` | Pass `robot_config=meta` to RobotAgent |
| `initialization/resources/robot_ur5e.json` | Add `control_mode`, `ros2`, `named_positions` |
| `initialization/resources/robot_xarm6.json` | Same |

### Reference files (reuse patterns from)
| What | Where |
|---|---|
| MoveGroup action client | `test/test_ur5e_moveit_action.py` |
| Gripper action client | `test/test_ur5e_rg2_gripper_action.py` |
| Prefixed joint names | `ros2/xarm_gazebo/config/xarm6_ur5e_controllers.yaml` |

---

## Technical Challenges

1. **ROS2 + asyncio threading**: SPADE runs asyncio, rclpy has its own executor. MoveItBackend spins ROS2 in a daemon thread, bridges with `run_in_executor()`.
2. **ROS2 import isolation**: Factory lazy-imports `moveit_backend.py` so mock mode works without ROS2.
3. **Joint value formats**: Current configs use vendor format (mm, degrees). `named_positions` uses radians for MoveIt.
4. **MoveIt startup time**: `move_group` takes ~40s. Backend must wait with timeout.
5. **Dual-robot joint filtering**: `/joint_states` publishes all joints from both robots. Filter by prefix.

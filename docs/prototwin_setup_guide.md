# ProtoTwin Setup Guide

Connecting the SPADE multi-agent framework (WSL Ubuntu) with ProtoTwin for manufacturing simulation.

## Architecture Overview

```
┌─────────────────────────────┐     ┌──────────────────────────────┐
│  WSL Ubuntu                 │     │  Windows                     │
│                             │     │                              │
│  SPADE Multi-Agent System   │     │  ProtoTwin (Browser)         │
│  ├─ ProductAgent            │     │  ├─ 3D Assembly Scene        │
│  ├─ RobotAgent(s)           │◄───►│  ├─ Robot Models (UR5e, etc) │
│  └─ CentralControllerAgent  │     │  └─ Signals (I/O)           │
│                             │     │                              │
│  ProtoTwin Python Client    │     │  ProtoTwin Connect (Bridge)  │
│  (prototwin + prototwin-    │     │  (runs on Windows, exposes   │
│   gymnasium packages)       │────►│   signal API over localhost) │
└─────────────────────────────┘     └──────────────────────────────┘
```

## Step 1: Install ProtoTwin Connect on Windows

ProtoTwin runs in the browser, but **ProtoTwin Connect** is required to bridge the browser simulation to external code (your Python agents). Download it from [prototwin.com](https://prototwin.com/) and install on your Windows host.

## Step 2: Set Up the ProtoTwin Scene

In ProtoTwin Simulate (browser), build your assembly workstation:

1. Import/create robot models (UR5e, xArm6 — matching your config)
2. Add an assembly board with part locations matching your `assembly_board-v1.json`
3. Define **signals** for each robot action:

| Signal Name | Type | Direction | Purpose |
|---|---|---|---|
| `ur5e_command` | Uint8 | Write | Command ID (0=idle, 1=move_to_pick, 2=pick, 3=move_loaded, 4=place, 5=home) |
| `ur5e_target_x/y/z` | Float | Write | Target position |
| `ur5e_part_name` | Uint8 | Write | Part identifier |
| `ur5e_status` | Uint8 | Read | Status (0=idle, 1=busy, 2=done, 3=error) |
| `ur5e_gripper_state` | Boolean | Read | Gripper open/closed |
| `xarm6_command` | Uint8 | Write | Same pattern for second robot |
| ... | ... | ... | ... |

## Step 3: WSL Networking Setup

ProtoTwin Connect runs on Windows. Your SPADE agents run in WSL. You need WSL to reach Windows localhost.

### Option A — Mirrored networking (recommended, WSL2 2.0+)

Add to `%UserProfile%\.wslconfig` on Windows:

```ini
[wsl2]
networkingMode=mirrored
```

Then restart WSL. After this, `localhost` from WSL reaches Windows services directly.

### Option B — Use Windows host IP

```bash
# From WSL, get Windows host IP:
export WINDOWS_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
echo $WINDOWS_HOST  # e.g., 172.x.x.x
```

## Step 4: Install Python Packages in WSL

```bash
pip install prototwin prototwin-gymnasium
```

## Step 5: Create a ProtoTwin Bridge Layer

Create a bridge module that translates your `RobotAgent`'s high-level commands into ProtoTwin signal read/writes. This sits between your `RobotAgent` and the simulation.

```python
# cais_spade_llm/simulation/prototwin_bridge.py

import asyncio
import prototwin
from typing import Dict, Any, Optional

# Signal address map — matches your ProtoTwin scene signals
ROBOT_SIGNALS = {
    "ur5e": {
        "command": 0,       # Write: command ID
        "target_x": 1,      # Write: target position
        "target_y": 2,
        "target_z": 3,
        "part_id": 4,       # Write: part identifier
        "status": 5,        # Read: execution status
        "gripper": 6,       # Read: gripper state
        "position_x": 7,    # Read: current position
        "position_y": 8,
        "position_z": 9,
    },
    "xarm6": {
        "command": 10,
        "target_x": 11,
        # ... same pattern, offset addresses
    },
}

# Command IDs
CMD_IDLE = 0
CMD_MOVE_TO_PICK = 1
CMD_PICK = 2
CMD_MOVE_LOADED = 3
CMD_PLACE = 4
CMD_HOME = 5

# Status codes
STATUS_IDLE = 0
STATUS_BUSY = 1
STATUS_DONE = 2
STATUS_ERROR = 3


class ProtoTwinBridge:
    """Bridge between SPADE RobotAgents and ProtoTwin simulation."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.client = None

    async def connect(self):
        """Connect to ProtoTwin Connect service."""
        self.client = await prototwin.start()
        await self.client.load(self.model_path)

    async def send_command(
        self, robot_name: str, command: int,
        target: Optional[Dict[str, float]] = None,
        part_id: int = 0,
    ) -> Dict[str, Any]:
        """Send a command to a robot and wait for completion."""
        sigs = ROBOT_SIGNALS[robot_name]

        # Write target position if provided
        if target:
            self.client.set(sigs["target_x"], target.get("x", 0.0))
            self.client.set(sigs["target_y"], target.get("y", 0.0))
            self.client.set(sigs["target_z"], target.get("z", 0.0))

        self.client.set(sigs["part_id"], part_id)
        self.client.set(sigs["command"], command)

        # Step simulation and poll until done
        result = await self._wait_for_completion(robot_name)
        return result

    async def _wait_for_completion(
        self, robot_name: str, timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Step simulation until robot reports done or error."""
        sigs = ROBOT_SIGNALS[robot_name]
        elapsed = 0.0
        dt = 0.01  # simulation step

        while elapsed < timeout:
            self.client.step()  # advance simulation by one step
            status = self.client.get(sigs["status"])

            if status == STATUS_DONE:
                return {
                    "status": "success",
                    "position": {
                        "x": self.client.get(sigs["position_x"]),
                        "y": self.client.get(sigs["position_y"]),
                        "z": self.client.get(sigs["position_z"]),
                    },
                    "gripper": "closed" if self.client.get(sigs["gripper"]) else "open",
                }
            elif status == STATUS_ERROR:
                return {"status": "error", "message": "Robot reported error"}

            elapsed += dt

        return {"status": "error", "message": "Timeout waiting for robot"}

    async def read_robot_state(self, robot_name: str) -> Dict[str, Any]:
        """Read current robot state from simulation."""
        sigs = ROBOT_SIGNALS[robot_name]
        return {
            "position": {
                "x": self.client.get(sigs["position_x"]),
                "y": self.client.get(sigs["position_y"]),
                "z": self.client.get(sigs["position_z"]),
            },
            "gripper": "closed" if self.client.get(sigs["gripper"]) else "open",
            "status": self.client.get(sigs["status"]),
        }
```

## Step 6: Integrate into RobotAgent

Modify your `RobotAgent` to use the bridge instead of simulated delays. The key change is in your action methods — instead of `asyncio.sleep()` and state toggling, you send commands to ProtoTwin and read back results:

```python
# In robot_agent.py — modify action methods

async def pick_part(self, part_name, origin_resource_location, **kw):
    # ... existing docstring/validation ...

    # Instead of simulated execution:
    if self._prototwin_bridge:
        result = await self._prototwin_bridge.send_command(
            robot_name=self.agent_name,
            command=CMD_PICK,
            part_id=PART_NAME_TO_ID[part_name],
        )
        if result["status"] == "error":
            return {"success": False, "error": result["message"]}

        # Update internal state from simulation
        self._position = result["position"]
        self._gripper_state = result["gripper"]
    else:
        # Fallback to existing simulated behavior
        await asyncio.sleep(0.5)

    # ... rest of existing state updates ...
```

## Step 7: ProtoTwin Scene Scripting

In ProtoTwin's TypeScript script editor, create a controller component that:

1. Reads command signals
2. Drives robot animations/physics
3. Writes status signals back

```typescript
// ProtoTwin script component (TypeScript)
import { Component, Signal, Robot } from "prototwin";

export class RobotController extends Component {
    @Signal() command: number = 0;
    @Signal() status: number = 0;
    @Signal() targetX: number = 0;
    @Signal() targetY: number = 0;
    @Signal() targetZ: number = 0;

    update(dt: number) {
        if (this.command !== 0 && this.status !== 1) {
            this.status = 1; // busy
            // Drive robot to target based on command type
            this.executeCommand(this.command);
        }
    }

    private executeCommand(cmd: number) {
        // Move robot joints, actuate gripper, etc.
        // When done: this.status = 2 (done)
        // On error: this.status = 3 (error)
    }
}
```

## Step 8: Run Everything

```bash
# Terminal 1 (Windows): Start ProtoTwin Connect
# (Launch ProtoTwin Connect application)

# Terminal 2 (WSL): Start your SPADE system
cd ~/projects/cais-spade-llm
python -m cais_spade_llm.main --simulation prototwin
```

## Key Considerations

1. **Signal address mapping** — The signal addresses (integers) must match between your Python bridge and your ProtoTwin scene. Document these carefully.

2. **Failure injection** — Your existing `sg_slippage_mode` can still work. Either inject failures in the bridge layer (before sending to ProtoTwin) or configure ProtoTwin physics to naturally produce failures (gripper slip, collision).

3. **Timing** — ProtoTwin can run faster-than-realtime when stepped programmatically. This is great for testing replanning scenarios quickly.

4. **Coordinate systems** — Map your workspace boundaries from robot configs to ProtoTwin's coordinate system. Ensure the staging areas in your config match the physical layout in the scene.

5. **Fallback mode** — Keep the existing simulated execution as a fallback (no ProtoTwin dependency for unit tests/CI).

## References

- [ProtoTwin Documentation](https://prototwin.com/docs)
- [ProtoTwin Features](https://prototwin.com/features)
- [prototwin-gymnasium on PyPI](https://pypi.org/project/prototwin-gymnasium/)
- [WSL Networking — Microsoft](https://learn.microsoft.com/en-us/windows/wsl/networking)

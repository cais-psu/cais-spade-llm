# LLM Bridge Construction: Primitive-Based Recovery Macros

## Motivation

The original bridge design (see `README.md` "Known Design Boundary") required bridge proposals to compile into existing catalog-backed task nodes. This limited recovery to recombinations of the five pre-defined robot task functions (`pick_approach`, `pick_grasp`, `place_approach`, `place_insert`, `move_home`).

That boundary is intentionally removed by this refactor. The LLM bridge can now compose recovery macros from **controller-level primitives**, enabling novel recovery sequences that were not anticipated at design time.

## Two-Layer Architecture

### Controller primitives (low-level)

`cais_spade_llm/resources/robot/ros2_pick_place_controller.py` exposes public atomic operations:

| Primitive | Description |
|---|---|
| `move_cartesian(x, y, z, speed=None)` | Move end-effector to absolute Cartesian position |
| `move_relative(dx, dy, dz, speed=None)` | Move relative to current end-effector position |
| `move_to_named_pose(pose_name)` | Move to a named joint configuration (e.g. "home") |
| `open_gripper()` | Open the gripper |
| `close_gripper()` | Close the gripper |
| `detect_parts(part_name=None)` | Call perception service, optionally filter by part name |
| `attach_part(model_name, link=None)` | Gazebo link attacher (simulation) |
| `detach_part(model_name="", link=None)` | Gazebo link detacher (simulation) |
| `get_current_pose()` | Return current end-effector pose via TF lookup |

These primitives are stateless atomic motion commands. They do not track task-level state (held part, assembly progress, etc.).

### Primitive semantics (`preconditions/effects`)

To make primitive composition recognizable to the bridge and safely validatable
before execution, each bridge-visible primitive now carries explicit YAML
semantics in its controller docstring:

- `preconditions`: when the primitive is allowed
- `effects`: what primitive-level state it changes

This is intentionally different from task-level `in_state/out_state` on
`RobotAgent`. Task methods still describe coarse workflow phases such as
`idle -> picked -> placed`, while primitive semantics describe low-level bridge
state such as:

- `held_part`
- `gripper_state`
- `current_pose`
- `current_pose_ref`

Examples:

- `attach_part(model_name)`
  - precondition: `held_part == null`
  - effect: `held_part = model_name`
- `detach_part(...)`
  - precondition: `held_part != null`
  - effect: `held_part = null`
- `open_gripper()`
  - effect: `gripper_state = open`
- `move_relative(dx,dy,dz)`
  - precondition: `current_pose` exists
  - effect: `current_pose = current_pose + (dx,dy,dz)`

These semantics are generic action meaning only. They are **not**
scenario-specific recovery rules.

### RobotAgent task methods (task-level)

`cais_spade_llm/agents/resource_agent/robot_agent.py` exposes the same five task methods used by DES planning. Each task method:

- Composes controller primitives internally
- Updates agent-owned logical state (`_current_state`, `_held_part`, `_gripper_state`)
- Carries YAML metadata (`in_state`, `out_state`, `part_transition`) for DES and safety validation
- Remains the canonical DES/safety/task-tracking surface

## Bridge Recovery Macros

### How the bridge sees primitives

The bridge does **not** use the shared `tools.json` catalog. Instead, a private primitive catalog is built in memory at bridge prompt time via `FunctionAnalyzer` introspection of the controller's public primitive methods. This catalog now includes:

- primitive name and description
- parameter schema
- primitive `preconditions`
- primitive `effects`
- derived `semantic_summary`

This private catalog is used for:

1. Constructing the bridge LLM prompt (so the LLM knows what primitives are available)
2. Validating bridge proposals (ensuring proposed primitives exist and are semantically valid)

The private catalog is never persisted to disk or shared with DES planning.

### Primitive-level bridge snapshot

Alongside the primitive catalog, the planner now builds a private primitive-level
snapshot for the target robot resource. This snapshot includes:

- `current_state`
- `held_part`
- `gripper_state`
- `current_pose`
- `current_pose_ref`
- available `named_poses`

The focused resource snapshot is shown to the LLM and is also used by the
deterministic semantic validator to project primitive effects across each
proposed macro task before it is approved.

The current bridge prompt is whole-system scoped. In addition to the focused
resource snapshot, the planner also provides:

- per-resource primitive catalogs for all bridge-capable robot resources
- per-resource bridge snapshots and modeled states
- pending tasks per resource
- part tracker state and observed locations
- active `obligation_targets`
- a grounded context tree that supports `{"context_ref": "/..."}` lookups

### Bridge proposal schema

The current bridge schema is:

- one top-level `primary_obligation`
- ordered `macro_tasks[]`
- one `macro_task` per recovery step in the bridge sequence

In v1, `macro_tasks[]` are executed serially in array order. Each task is
validated against the projected post-state of earlier tasks.

```json
{
  "primary_obligation": {
    "rule_id": "SAFE_2",
    "resource_jid": "xarm6@localhost"
  },
  "macro_tasks": [
    {
      "resource_jid": "ur5e@localhost",
      "macro_name": "stash_mcp",
      "description": "Temporarily clear the later-priority part from UR5e.",
      "rationale": "Free UR5e to recover the slipped prerequisite part.",
      "expected_start_state": "picked",
      "part_name": "MCP",
      "task_params": {
        "destination_location": "buffer_a"
      },
      "task_metadata": {
        "in_state": "picked",
        "out_state": "idle",
        "required_context_keys": ["destination_location"],
        "context_mapping": {
          "location_param": "destination_location",
          "location_type": "current_location"
        },
        "part_transition": {
          "completed": {
            "state": "ready",
            "location_param": "destination_location"
          }
        }
      },
      "primitive_steps": [
        {"primitive": "open_gripper", "params": {}},
        {"primitive": "move_to_named_pose", "params": {"pose_name": "home"}}
      ],
      "expected_snapshot": {
        "current_state": "picked",
        "held_part": "MCP",
        "gripper_state": "closed"
      },
      "projected_snapshot": {
        "current_state": "idle",
        "held_part": null,
        "gripper_state": "open",
        "current_pose_ref": "home"
      },
      "projected_part_entry": {
        "state": "ready",
        "location": "buffer_a"
      }
    },
    {
      "resource_jid": "xarm6@localhost",
      "macro_name": "recover_lcp",
      "description": "Restore XArm6 to a modeled idle state after the failed insert.",
      "rationale": "Discharge the primary obligation and allow DES to continue.",
      "expected_start_state": "recovery_required",
      "task_metadata": {
        "in_state": "recovery_required",
        "out_state": "idle",
        "required_context_keys": [],
        "context_mapping": {},
        "part_transition": null
      },
      "primitive_steps": [
        {"primitive": "move_to_named_pose", "params": {"pose_name": "home"}}
      ],
      "expected_snapshot": {
        "current_state": "recovery_required",
        "held_part": null,
        "gripper_state": "open"
      },
      "projected_snapshot": {
        "current_state": "idle",
        "held_part": null,
        "gripper_state": "open",
        "current_pose_ref": "home"
      }
    }
  ]
}
```

Before approval, each macro task is normalized and validated against the current
or projected bridge snapshot using the primitive `preconditions/effects`.
Examples of pre-approval rejection include:

- a primitive sequence that violates semantic preconditions
- unresolved `context_ref` values
- invalid `store_as` / step-output references
- missing `task_params` required by `task_metadata.required_context_keys`
- proposals whose final projected state does not restore a modeled continuation

### Compilation into runtime recovery tasks

`ProcessPlanner.apply_bridge_macro_proposal()` compiles each approved
`macro_task` into one `execute_recovery_macro` task node. Those nodes are
inserted as a serial runtime recovery chain.

Each compiled node carries:

- `function_name = execute_recovery_macro`
- `resource_jid`
- `params`: `macro_name`, `primitive_steps`, `expected_start_state`,
  `expected_snapshot`, `product_jid`, `task_id`, plus any bridge `task_params`
- bridge sequencing metadata:
  - `bridge_sequence_id`
  - `bridge_sequence_index`
  - `bridge_sequence_length`
- node-level metadata copied from `task_metadata`:
  - `in_state`
  - `out_state`
  - `required_context_keys`
  - `context_mapping`
  - `part_transition`
- bridge validation / tracking fields:
  - `part_name`
  - `primary_obligation`
  - `projected_snapshot`
  - `projected_part_entry`

### execute_recovery_macro

`RobotAgent.execute_recovery_macro` is registered in `self.executables` for runtime dispatch but is excluded from `function_names`, `ALLOWED_FUNCS`, and shared tool-catalog generation. This keeps it invisible to DES planning and the shared `tools.json`.

At runtime, `execute_recovery_macro`:

1. Validates `expected_start_state` against the robot's current `_current_state`
2. Validates `expected_snapshot` against the robot's current primitive-level
   bridge snapshot
3. Re-validates the primitive sequence against the latest runtime snapshot
4. Resolves runtime param refs, including supported `/step_outputs/...` lookups
5. Iterates `primitive_steps` sequentially, calling the corresponding
   controller primitive
6. Extracts supported observational outputs (`detect_parts`, `get_current_pose`)
   when `store_as` is used
7. Applies the same primitive `effects` to the live `RobotAgent` bridge state
   (`_held_part`, `_gripper_state`, pose snapshot)
8. On success: returns a normal task-style status payload
9. On failure: stops at the failing step and returns failure context that
   includes the step index / primitive plus semantic or snapshot details when
   available

## Safety and Tracking Integration

### Node-level metadata

Bridge macro tasks carry their own `in_state`/`out_state`/`part_transition` on
the plan node because `execute_recovery_macro` does not exist in the shared
catalog.

The following systems prefer node-level metadata when present, and fall back to shared-catalog lookup for normal tasks:

- **Global FSA compilation**: uses node `in_state`/`out_state` to construct state transitions
- **Runtime plan safety validation**: validates node-level states against safety rules
- **Product part-tracker**: applies node-level `part_transition` after task
  completion, using `part_name` / `touched_part` from the bridge node

### Approval gate

Bridge proposals remain approval-gated. The operator reviews the ordered bridge
macro tasks before they are compiled into the plan. Rejection feedback is fed
back into bridge regeneration as before.

## Runtime Handoff and Fail-Closed Checks

After an approved bridge macro task completes, Product does not blindly run the
rest of the bridge tail. It performs a runtime handoff check:

1. Refresh the affected resource's bridge snapshot
2. Compare the refreshed runtime snapshot against `projected_snapshot`
3. Compare the tracked part entry against `projected_part_entry` when present
4. If the live state diverged, stop and fail closed to `human_required`
5. If the original failed task is directly executable again, validate the
   updated plan and resume normal execution
6. Otherwise, re-run DES from the refreshed runtime state
7. If DES can now continue, trim the remaining bridge tail and validate the
   repaired plan
8. If DES still cannot continue but more bridge tasks remain, continue the
   approved bridge tail
9. If the final approved bridge task completes and DES still has no
   continuation, escalate to `human_required`

## Relationship to Existing Recovery Flow

This design supersedes the older legacy bridge flow described in
`README.md`. Previously, bridge proposals compiled into existing
catalog-backed task nodes. Now, bridge proposals embed primitive sequences that
execute through ordered `execute_recovery_macro` tasks.

The rest of the recovery flow is unchanged:

- DES search remains the primary recovery planner
- Bridge is invoked only when DES search cannot find a catalog-valid obligation-satisfying path
- Approval/rejection/regeneration workflow is preserved
- Terminal failure (`human_required`) conditions are unchanged

## Why primitive semantics are needed

The primitive catalog alone is not enough for safe bridge generation. Primitive
names and parameter schemas tell the LLM **what** can be called, but not **when**
those calls are valid or **what** state they change.

The bridge therefore uses explicit primitive `preconditions/effects` so that:

- the LLM sees the low-level action meaning directly in the prompt
- semantically impossible primitive sequences are rejected before approval
- runtime macro execution updates `RobotAgent` state consistently with the same
  semantics used during validation

This avoids hard-coded recovery rules such as "must release before recover"
while still making it possible for the bridge to infer that, for example, a
robot already holding one part cannot validly `attach_part(...)` on another
part until it has first executed a release sequence.

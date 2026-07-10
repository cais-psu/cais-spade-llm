# Classical Planning Limits vs LLM + ReAct + DES for Recovery

## Problem Statement

The current recovery stack is intentionally split across two levels of abstraction.

At the task level, runtime recovery is modeled through the five predefined
`RobotAgent` task functions:

- `pick_approach`
- `pick_grasp`
- `place_approach`
- `place_insert`
- `move_home`

Those task functions are the surface used by DES recovery in
`resource_bidding.py`. They provide a compact, discrete, and catalog-backed
planning interface, which is exactly why the current DES layer is reliable for
modeled recovery.

At the low level, however, the robot also exposes controller primitives in
`gazebo_pick_place_controller.py` and `hardware_pick_place_controller.py`, such as:

- `move_cartesian`
- `move_relative`
- `move_to_named_pose`
- `open_gripper`
- `close_gripper`
- `detect_parts`
- `attach_part`
- `detach_part`
- `get_current_pose`
- `compute_pick_targets`
- `compute_place_targets`

These primitives are paired with generic primitive `preconditions` and
`effects`, extracted and validated in `recovery_primitives.py`. That primitive
layer is expressive enough to describe many recovery actions that were not
anticipated when the five task-level functions were authored.

The core design boundary of the current architecture is that it intentionally
does **not** model every recovery state, sensing branch, or recovery operator as a
symbolic planner action. The system keeps the task-level DES model compact and
uses generic low-level primitives to recover from novel disturbances below that
task surface.

That design choice is important. It reduces manual recovery-domain engineering,
but it also means the recovery layer is not already available as a finite
classical planning domain.

## What a Classical Planner Would Need

A classical planner could solve the same recovery class if the recovery layer were
converted into an explicitly modeled symbolic domain. In practice, that would
require at least the following additional abstractions:

- finite symbolic recovery regions rather than open-ended Cartesian targets
- explicit observation outcome models for sensing actions such as
  `detect_parts`
- a searchable primitive-level operator space, not just primitive-level
  validation
- recovery-specific re-entry predicates that define when the system is back in a
  modeled DES continuation
- abstractions that bind continuous pose parameters into a finite search space

Those ingredients are not theoretical luxuries. They are what make classical
search tractable and auditable at the recovery layer.

The current repository does not provide that full planner-ready abstraction.
Instead, it provides:

- a task-level DES search space over the five catalog functions
- a low-level primitive space with generic semantics
- deterministic validation and projection of proposed primitive sequences

That is enough to **check** recovery candidates, but it is not yet the same thing
as a finite symbolic recovery planner.

## Why the Current Architecture Is Not Classically Plannable at the Recovery Layer

The main issue is not that classical planning is weak. The issue is that the
current recovery layer is intentionally under-modeled relative to what a
classical planner would require.

### Task-level DES is discrete and searchable

`resource_bidding.py` performs DES-style forward search over catalog-backed task
transitions. This works because the search space is already discrete:

- resource states are symbolic
- part transitions are symbolic
- tool selection is finite
- parameterization is limited to catalog-defined task parameters

That is exactly the right representation for modeled recovery, but it is only
available at the task level.

### Primitive semantics support validation, not full search

`recovery_primitives.py` provides the pieces needed to normalize, resolve, and
project primitive sequences:

- `build_primitive_catalog(...)`
- `validate_and_project_steps(...)`
- `preview_step_output(...)`
- `extract_step_output(...)`
- `get_robot_recovery_snapshot(...)`

This is already a strong deterministic layer. It can tell whether a candidate
primitive sequence is semantically valid and what state it would project to.

However, this is not yet a classical planner over primitives. The current code
does not search over all enabled primitive sequences, all observation branches,
or all feasible parameter bindings. It validates a sequence after some upstream
mechanism proposes one.

### Controller primitives are expressive but open-ended

The low-level primitives in `gazebo_pick_place_controller.py` and
`hardware_pick_place_controller.py` include open-ended
numeric and observation-driven actions:

- `move_cartesian(x, y, z, ...)`
- `move_relative(dx, dy, dz, ...)`
- `move_pose(...)`
- `detect_parts(...)`
- `get_current_pose()`
- `compute_pick_targets(...)`
- `compute_place_targets(...)`

These primitives are meaningful enough to compose into recovery behavior, but
they are not already expressed as a finite classical planning domain. Several of
them depend on runtime observations, return continuous values, or generate
parameters that are only known after sensing.

That creates a practical search gap:

- the action vocabulary is generic and reusable
- the semantic validator can check a candidate sequence
- but the repository does not currently define a finite symbolic recovery search
  domain over those primitives

Under these modeling assumptions, a pure classical planner would first require a
new symbolic abstraction layer for recovery. The current architecture
deliberately avoids building that layer by hand for every recovery family.

## Why LLM + ReAct + DES Fits This Architecture

The proposed `LLM + multi-shot ReAct + DES` architecture fits the current design
because it respects the division of labor already present in the repository.

### DES provides the formal task-level backbone

DES remains responsible for:

- the nominal task structure
- obligation-driven modeled recovery
- identifying when no catalog-valid continuation exists
- defining the marked re-entry condition for normal execution

In other words, DES still anchors the recovery problem in a discrete workflow
that is formally meaningful at the task level.

### ReAct handles the under-modeled recovery layer

When DES cannot continue, the missing problem is not "plan the whole factory
again." The missing problem is: compose a small recovery sequence out of existing
controller primitives that restores the system to a modeled continuation.

That is exactly where a multi-shot ReAct loop is useful:

- it can decide what to observe next
- it can use live outputs from `detect_parts`, `get_current_pose`, and the
  target-computation helpers
- it can synthesize a recovery sequence from generic primitives without requiring
  a pre-authored symbolic recovery operator library

### Deterministic code still owns correctness boundaries

The LLM is not the source of safety or truth in this design.

`recovery_primitives.py` still provides deterministic semantic checks.
`robot_agent.py` still owns task-level and runtime execution semantics.
DES still defines the discrete continuation target.
The proposed safety shield still decides whether a candidate recovery action is
allowed.

This means the LLM is not replacing DES. It is filling the recovery layer that
would otherwise require extensive manual symbolic recovery modeling.

That is the practical novelty of the architecture: the system keeps a compact
formal task-level model, but uses an LLM as a bounded recovery-synthesis mechanism
over generic primitives when recovery falls below that modeled surface.

## Concrete LG Slippage Example

The LG slippage path illustrates the gap clearly.

In `robot_agent.py`, the LG fault injection causes `place_insert` to fail and
drops the part into the UR5e-side recovery lane. The failure observations
include that the part rolled out of the original XArm6 insertion path and into a
recovery situation that is not represented as one of the original five task
operators.

With only the five predefined task-level functions, task-level DES does not have
enough expressive power to synthesize the needed recovery. The recovery
requires a combination such as:

1. clear the original robot from the shared zone
2. temporarily unload the part currently held by the other robot
3. observe the displaced LG again
4. recover LG using low-level pick primitives
5. insert LG using low-level place primitives
6. restore the suffix state needed for the nominal continuation

The low-level primitive layer already contains enough building blocks to carry
out that recovery:

- motion primitives
- gripper primitives
- attach/detach primitives
- sensing primitives
- target-generation helpers

What is missing is not low-level capability. What is missing is a generic,
pre-authored symbolic recovery operator library that says exactly how to compose
those low-level pieces for every recovery case.

That is why the LLM is practically useful in the current architecture. It can
synthesize a recovery sequence from those generic primitives without requiring the
developer to model LG slippage, MCP unloading, cross-robot recovery transfer,
and suffix restoration as explicit classical planner operators ahead of time.

The resulting recovery is still bounded by deterministic checks:

- primitive preconditions and effects
- runtime snapshot validation
- modeled continuation checks
- safety constraints and obligation targets

So the value of the LLM is not unrestricted autonomy. The value is bounded
recovery synthesis over an intentionally under-modeled primitive space.

## Novelty Claim and Non-Claim

**Claim**

Under the present architecture, `LLM + multi-shot ReAct + DES` solves an
under-modeled recovery problem without requiring the designer to
exhaustively engineer symbolic recovery operators for every disturbance pattern.
DES preserves the formal task-level structure, while the LLM synthesizes recovery
sequences from existing low-level primitives and deterministic primitive
semantics.

**Non-claim**

This is not a proof that classical planning can never solve the same recovery
scenario in principle. A sufficiently enriched symbolic recovery model could
make the recovery layer classically plannable. The point of this architecture is
that the current system deliberately avoids building that full symbolic recovery
model by hand and instead uses the LLM as a bounded model-completion mechanism.

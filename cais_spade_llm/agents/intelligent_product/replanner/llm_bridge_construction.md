# LLM Bridge Construction

This note is a current map of how the `llm_bridge` code is organized and how
the end-to-end bridge workflow runs today.

The most important mental model is:

- DES remains the nominal planner.
- The bridge runs only after DES cannot find a modeled continuation.
- The bridge does not own resource-specific behavior directly.
- Resource-specific behavior is provided by `ResourceProfile` plugins.
- The bridge tries to return the system to a state where the pending DES suffix
  can resume.

## 1. What The `bridge_*.py` Files Do

### `llm_bridge/__init__.py`

This is only the package entry point.

- It aggregates:
  - `BridgeSessionMixin`
  - `BridgeCompilerMixin`
  - `BridgeSafetyMixin`
- `LlmBridgeReplannerMixin` is the mixin that `ProcessPlanner` uses.

### `bridge_adapters.py`

This is the canonicalization layer between raw runtime state and bridge state.

- Detects the resource type for a resource/snapshot.
- Builds a canonical bridge snapshot with:
  - `resource_core`
  - `resource_facets`
  - compatibility fields
- Normalizes bridge events into canonical fields such as:
  - `operation_family`
  - `targets`
  - `projected_effects`
- Normalizes bridge constraints from safety metadata into bridge-consumable
  constraint objects.

Think of this file as: "turn raw agent state and raw bridge events into bridge
objects with a stable shape."

### `bridge_safety.py`

This is the state, safety, and bridge-event validation layer.

- Builds the grounding context exposed to prompts.
- Builds the primitive context for each bridge-capable resource.
- Derives marked re-entry conditions from the current disrupted state and the
  resumable DES suffix.
- Derives bridge safety constraints from runtime safety rules.
- Validates candidate `bridge_events`:
  - feasibility
  - state-delta consistency
  - safety constraints
  - coverage of the current marked re-entry gap
- Projects bridge events forward to check what they would do to resources and
  parts.

Think of this file as: "what must be true for a bridge to be allowed."

### `bridge_generation.py`

This is the JSON parsing and primitive-plan normalization layer.

- Parses raw LLM JSON into:
  - `observe`
  - `bridge_events`
  - `final_plan`
- Normalizes primitive-based final plans into validated `macro_tasks`.
- Resolves `context_ref` references through the current grounding context.
- Projects macro-task outputs to produce:
  - projected resource snapshots
  - projected part states/locations

Think of this file as: "turn LLM output into a validated internal bridge plan."

### `bridge_session.py`

This is the orchestration layer for the whole bridge session.

- `prepare_bridge_session()` builds the full `prepared_bridge_request`.
- `_refresh_bridge_grounding_context()` refreshes:
  - grounding context
  - marked re-entry context
  - bridge safety context
- `run_bridge_react_session()` runs the multi-turn bridge loop.
- Handles phase control:
  - `observe_required`
  - `bridge_events`
  - `final_plan`
- Appends validation feedback after each rejection.
- Accepts or rejects the final bridge proposal.
- `validate_preprogrammed_bridge_proposal()` validates a prebuilt bridge plan
  through the same normalization and safety/re-entry gates.

Think of this file as: "the runtime state machine for the bridge."

### `bridge_compiler.py`

This is the deterministic compilation layer.

- Compiles approved bridge events into executable `macro_tasks`.
- Uses resource profiles to choose the right compiler hook per resource type
  and operation family.
- Checks whether a final plan realizes the approved bridge events.
- Synthesizes the `plan_rewrite` section that tells DES which suffix task IDs
  to resume after the bridge.

Think of this file as: "turn approved bridge events into executable bridge
macros and resume instructions."

### `primitive_semantics.py`

This is the primitive semantics layer.

- Builds a private bridge primitive catalog from resource primitives.
- Extracts primitive parameter schemas.
- Reads YAML frontmatter preconditions/effects.
- Validates primitive sequences.
- Applies primitive effects to projected snapshots.
- Resolves and previews observation outputs.
- Syncs accepted bridge snapshots back to runtime agent state when needed.

Think of this file as: "the semantic model of controller primitives used by the
bridge."

## 2. What The `*_profile.py` Files Do

The bridge is now profile-driven. The profiles are the main place where
resource-specific behavior lives.

### `resources/resource_profile.py`

This is the shared plugin contract.

- Defines the `ResourceProfile` dataclass.
- Owns the profile registry.
- Provides helper accessors for snapshot fields:
  - `resource_snapshot_field_value(...)`
  - `resource_snapshot_set_field(...)`
  - `resource_snapshot_fields_map(...)`
  - `resource_snapshot_availability(...)`
  - `resource_snapshot_carried_entity(...)`
- Boots built-in profiles lazily on first access.

Important idea:

- `ResourceProfile` is the contract that tells the bridge how to interpret a
  resource type.
- The bridge should ask the profile, not hardcode resource behavior in the
  bridge core.

### `resources/robot/robot_profile.py`

This is the robot/manipulator profile.

- Defines the manipulator facet:
  - `held_part`
  - `gripper_state`
  - `current_pose`
  - `current_pose_ref`
  - `named_poses`
- Defines robot occupancy.
- Defines robot event-family resolution such as:
  - `pick`
  - `place`
  - `stage`
  - `clear`
  - `pick_place`
- Defines robot state projection and validation after bridge events.
- Defines compiler hooks for deterministic macro compilation.
- Defines manipulator prompt addenda and repair examples.

Think of it as: "everything the bridge needs to know about manipulator-style
resources."

### `resources/machine/printer_profile.py`

This is the printer profile.

- Defines the printer facet:
  - `active_job`
  - `job_state`
  - `material_state`
  - `bed_state`
- Defines printer occupancy and availability.
- Defines printer event families such as:
  - `pause_job`
  - `resume_job`
  - `cancel_job`
- Defines printer-specific prompt addenda and repair examples.

At the moment the printer is structurally visible to the bridge, but the main
recovery execution path is still much more mature for robots than for printers.

## 3. How The Bridge Starts

At a high level:

1. A runtime disruption happens.
2. DES cannot find a modeled continuation.
3. `ProcessPlanner` falls back to the bridge mixin.
4. The bridge prepares a `prepared_bridge_request`.
5. The ReAct bridge loop runs until it accepts a bridge or exhausts retries.

The bridge request contains:

- the disrupted search state
- the pending parts
- the focused disrupted resource
- tools catalog
- part tracker
- resource states
- part states and locations
- bridge resource snapshots
- primitive catalogs
- marked re-entry context
- bridge safety context
- bridge session state

## 4. End-To-End Workflow For The Bridge

### Step 1: Prepare the bridge request

`prepare_bridge_session()` does the heavy setup:

- build primitive catalogs for bridge-capable resources
- snapshot each resource into canonical bridge state
- build grounding context for prompts
- derive marked re-entry conditions
- derive bridge safety constraints
- initialize the bridge session state

The output is `prepared_bridge_request`, which becomes the working state for
the rest of the bridge loop.

### Step 2: Decide the current phase

The bridge can be in three phases:

- `observe_required`
  - there is not enough grounded state to bridge safely
- `bridge_events`
  - the model must propose high-level bridge events
- `final_plan`
  - the model or planner must produce executable `macro_tasks`

This phase is recomputed every turn from the current prepared state, not
hardcoded once at the start.

### Step 3: Observation turn, if needed

In `observe_required`, the LLM can only request a bounded observation action.

Typical observation primitives:

- `detect_parts`
- `get_current_pose`

The observation result is stored in `bridge_session["observation_store"]`, the
grounding context is refreshed, and the bridge re-checks whether it now has
enough information to move to `bridge_events`.

### Step 4: `bridge_events` turn

In `bridge_events`, the LLM proposes high-level bridge events such as:

- clear a blocked resource
- stage a carried part away
- pick a misplaced part
- place/assemble a part at its goal
- reacquire a resume-suffix part

The bridge then validates those events through:

- feasibility checks
- bridge safety constraints
- event ordering checks
- state-delta consistency
- coverage of all unmet marked re-entry conditions

If validation fails, the session appends validation feedback and asks the LLM
to try again.

If validation succeeds:

- the approved events are stored
- the bridge moves to `final_plan`
- the planner immediately tries to build a deterministic preview final plan
  from the approved events

### Step 5: `final_plan` turn

This is where an important distinction exists:

#### Normal live path

After `bridge_events` are accepted, the planner first tries
`_compile_bridge_events_to_macro_tasks(...)`.

If that deterministic compilation succeeds and preview validation succeeds, the
bridge already has a planner-generated draft final plan.

In that case:

- the session may emit a `planner_compiler` response directly
- or, if the draft later needs repair, it will prompt the LLM in `llm_repair`
  mode with the draft included

So in the current live design, the LLM does not always infer the final plan
from scratch. Often it infers:

- the needed observations
- the approved bridge events

and the planner compiles the first final plan draft deterministically.

#### LLM repair path

If deterministic compilation cannot give an accepted draft, the bridge asks the
LLM for a `final_plan` directly.

The LLM is constrained by:

- approved bridge events
- marked re-entry conditions
- bridge safety context
- primitive reference card
- current grounding context
- validation feedback from prior failures
- optional draft final plan to repair

The `final_plan` is then normalized and checked.

### Step 6: Final-plan validation

A candidate `final_plan` is accepted only if it passes all of these gates:

- normalization into valid `macro_tasks`
- realization of the approved bridge events
- feasibility
- bridge safety constraints
- marked re-entry closure
- modeled continuation / resume viability

If any gate fails, the bridge appends feedback and retries until the retry
budget is exhausted.

### Step 7: Acceptance and merge

When a `final_plan` is accepted:

- bridge macros are compiled/validated
- `plan_rewrite` is synthesized
- the failed DES branch is replaced
- the resumable suffix task IDs are merged back into the planner

At that point the bridge is done and DES can continue from the repaired state.

## 5. End-To-End Workflow For The Preprogrammed Bridge Recovery Scenario

The preprogrammed bridge scenario is useful because it shows exactly what the
bridge validates, without relying on live LLM creativity.

The key file is:

- `agents/intelligent_product/replanner/preprogrammed_bridge_scenarios.py`

That file currently contains `build_preprogrammed_bridge_proposal(...)` for the
`recover_lg_v1` scenario.

### What happens in the scripted test harness

In the scripted harness:

1. The bridge session is still prepared normally.
2. The scripted turns provide:
   - an `observe` turn
   - a `bridge_events` turn
   - a `final_plan` turn
3. For the `final_plan` turn, the fake harness can inject:
   - a preprogrammed plan from
     `build_preprogrammed_bridge_proposal(prepared_bridge_request)`

So in the preprogrammed scripted path, the final plan is not inferred by the
LLM. It is built by the scenario helper and then validated by the same bridge
gates used for live plans.

### Why this still matters

Even though the preprogrammed final plan is not LLM-generated, the bridge still
checks the same things:

- primitive normalization
- safety
- feasibility
- marked re-entry closure
- approved-event realization
- resume viability

That makes the scripted scenario a good "reference execution" for debugging the
bridge runtime.

## 6. How The LLM Actually Infers The Final Plan Today

The short answer is:

- in the scripted preprogrammed scenario, it usually does not
- in the live path, it often infers `observe` and `bridge_events`, while the
  planner compiles the first final-plan draft
- the LLM mainly infers the final plan when the deterministic draft is missing
  or needs repair

So the current architecture is best described as:

- LLM-guided bridge-event planning
- planner-compiled first final-plan draft
- LLM repair only when needed

That is why `bridge_events` are the key semantic output of the LLM bridge. The
accepted bridge events are the bridge-level intent, and the final plan is the
executable realization of that intent.

## 7. Reading Order For A New AI Session

If you are new to this code and want the shortest useful path, read in this
order:

1. `resources/resource_profile.py`
2. `resources/robot/robot_profile.py`
3. `resources/machine/printer_profile.py`
4. `llm_bridge/bridge_adapters.py`
5. `llm_bridge/bridge_safety.py`
6. `llm_bridge/bridge_session.py`
7. `llm_bridge/bridge_compiler.py`
8. `llm_bridge/bridge_generation.py`
9. `llm_bridge/primitive_semantics.py`
10. `agents/intelligent_product/replanner/preprogrammed_bridge_scenarios.py`
11. `test/test_case3_recovery_main.py`

If you only want the runtime control flow, start with:

1. `bridge_session.py`
2. `bridge_safety.py`
3. `bridge_compiler.py`
4. `preprogrammed_bridge_scenarios.py`

## 8. Current Practical Takeaway

The bridge is no longer a single prompt-to-plan file. It is a profile-driven
runtime made of:

- canonical state adapters
- safety and re-entry validation
- JSON/primitive-plan normalization
- deterministic event compilation
- a multi-turn session orchestrator

If you remember only one thing, remember this:

The bridge's main semantic output is the approved `bridge_events`; the final
plan is the validated executable realization of those events.

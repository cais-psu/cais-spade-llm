# LLM Bridge Construction

This note describes the current active `llm_bridge` package.

The current mental model is:

- DES remains the nominal replanner.
- The bridge starts only after DES has no modeled continuation.
- The active bridge is now a compact v4 path.
- The current v4 slice prepares a curated bridge state package, sends one
  single-shot prompt, and records the raw reply.

## 1. Active Files

### `llm_bridge/__init__.py`

This is the package entry point.

- It exposes `BridgeSessionMixin` through `LlmBridgeReplannerMixin`.
- `ProcessPlanner` uses this active mixin.

### `bridge_resource_normalization.py`

This is the bridge normalization layer.

- Resolves a resource type from raw snapshots and modeled state.
- Normalizes raw snapshots into a stable bridge resource shape with:
  - `resource_core`
  - `resource_facets`
  - top-level mirrored fields
  - occupancy
  - availability
- Normalizes event-like payloads into stable bridge event fields such as:
  - `operation_family`
  - `targets`
  - `projected_effects`
- Normalizes safety rules into generic bridge constraint payloads.

Think of this file as: "turn raw runtime/state payloads into stable bridge
objects."

### `primitive_semantics.py`

This is the primitive semantics layer for v4.

- Builds execution primitive catalogs from live `_BRIDGE_PRIMITIVES`.
- Builds the synthesis-visible primitive catalog used by prompt preparation.
- Exposes prompt-facing composite robot entries such as:
  - `grasp_part`
  - `release_part`
- Hides raw prompt-time gripper/attachment primitives from the LLM surface.
- Resolves `context_ref` values against grounding context and step outputs.
- Validates primitive step sequences against required params and semantic
  preconditions.
- Projects primitive effects onto bridge snapshots.
- Syncs accepted bridge snapshots back onto runtime agents when needed.

Think of this file as: "the semantic model of the controller primitives the
bridge can talk about."

### `bridge_prompts.py`

This is the active prompt-preparation module.

- Builds `single_shot_prompt_input`.
- Renders `single_shot_prompt_text`.
- Packages:
  - `llm_input`
  - proposal success criteria
  - the required JSON response contract
- Renders `Allowed Execution Surface` directly from:
  - `llm_input["allowed_execution_surface"]`
- Tells the model that macro tasks may span multiple listed resources when a
  coordinated recovery is needed.

Think of this file as: "prepare the final single-shot prompt package."

### `bridge_session.py`

This is the active bridge orchestration layer.

- Builds the prepared bridge request after DES failure.
- Loads requirement and safety artifacts from the verified bundle.
- Builds:
  - `context_summary`
  - `llm_input`
  - `primitive_catalog`
  - `bridge_snapshot`
  - `bridge_resources`
  - `single_shot_prompt_input`
  - `single_shot_prompt_text`
- Builds the prompt-facing multi-resource action surface inside:
  - `llm_input["allowed_execution_surface"]`
- Tracks the bridge session mode:
  - `single_shot`
  - `multi_turn`
- In the current slice, `execute_prepared_bridge_request()` now:
  - runs one `single_shot` LLM call
  - captures the raw response as `llm_output_recorded`
  - `unsupported_reasoning_mode` for `multi_turn`

Think of this file as: "prepare the bridge package, send the single-shot
prompt once, and capture the raw reply before proposal parsing/validation."

## 2. Resource Profiles

The bridge is profile-driven. Resource-specific behavior lives in
`ResourceProfile` implementations rather than in bridge core code.

### `resources/resource_profile.py`

This is the shared plugin contract.

- Defines the `ResourceProfile` dataclass.
- Owns the profile registry.
- Provides snapshot helpers such as:
  - `resource_snapshot_field_value(...)`
  - `resource_snapshot_set_field(...)`
  - `resource_snapshot_fields_map(...)`
  - `resource_snapshot_availability(...)`
  - `resource_snapshot_carried_entity(...)`

### `resources/robot/robot_profile.py`

This is the manipulator profile.

- Defines robot facets such as:
  - `held_part`
  - `gripper_state`
  - `current_pose`
  - `current_pose_ref`
  - `named_poses`
- Defines robot occupancy and availability behavior.
- Defines robot event-family resolution.
- Defines robot-specific state projection and sync behavior.

### `resources/machine/printer_profile.py`

This is the printer profile.

- Defines printer facets such as:
  - `active_job`
  - `job_state`
  - `material_state`
  - `bed_state`
- Defines printer occupancy and availability behavior.
- Defines printer event-family resolution for job control flows.

## 3. Current Active Bridge Flow

At a high level:

1. A runtime disruption happens.
2. DES cannot find a modeled continuation.
3. `ProcessPlanner` falls back to the active bridge mixin.
4. The bridge prepares `prepared_bridge_request`.
5. The bridge builds the final single-shot prompt artifacts.
6. The bridge sends one LLM request.
7. The bridge records the raw response text.

## 4. What `prepared_bridge_request` Contains

The prepared bridge request is the working handoff package for the active v4
bridge.

Important fields include:

- disruption context
- focused resource and failure anchor
- part tracker and resource states
- normalized bridge resources and focused bridge snapshot
- requirement inventory and requirement status
- loaded safety rules
- `context_summary`
- `llm_input`
- `llm_input["allowed_execution_surface"]`
- `single_shot_prompt_input`
- `single_shot_prompt_text`
- `bridge_debug`
- raw single-shot response text after execution

## 5. What `llm_input` Means

`llm_input` is the curated LLM handoff, not the whole debug artifact.

It intentionally mixes three kinds of information:

- observed/runtime-grounded facts
  - failure event
  - observed resources
  - observed parts and poses
- loaded/model-backed context
  - relevant safety rules
  - relevant assembly requirements
- deterministically derived bridge context
  - modeled continuation gap
  - unmet goal conditions
  - unmet continuation conditions

## 6. What The Current Single-Shot Prompt Adds

The single-shot prompt wraps `llm_input` with proposal success criteria and the
response contract.

It includes:

- task and role framing
- fault event
- observed runtime state
- loaded safety rules
- relevant assembly requirements
- modeled continuation gap
- proposal success criteria
- allowed execution surface
- required JSON response contract
- hard constraints

The active bridge now prepares this prompt, sends it once to the configured
LLM, and records the raw reply.

The model-facing execution surface is now:

- multi-resource
- runtime-derived
- prompt-facing composite-only

So the LLM sees coordinated recovery options across the listed resources rather
than only a focused-resource primitive menu.

## 7. Reading Order For A New Session

If you want the shortest useful path through the current bridge, read in this
order:

1. `resources/resource_profile.py`
2. `resources/robot/robot_profile.py`
3. `resources/machine/printer_profile.py`
4. `llm_bridge/bridge_resource_normalization.py`
5. `llm_bridge/primitive_semantics.py`
6. `llm_bridge/bridge_prompts.py`
7. `llm_bridge/bridge_session.py`
8. `test/test_case3_bridge_dryrun.py`

If you only want the runtime control flow, start with:

1. `bridge_session.py`
2. `bridge_prompts.py`
3. `primitive_semantics.py`
4. `test_case3_bridge_dryrun.py`

## 8. Why The LLM — Architectural Justification

### The core separation

| Layer | Owner | Role |
|---|---|---|
| Successor generation | LLM | Synthesize plausible recovery actions for novel/partially-modeled failures |
| State evolution | DES (deterministic) | Project symbolic state forward |
| Validation | Feasibility oracle + safety checker | Accept/reject each candidate |
| Selection | Search / scoring | Pick the best valid candidate |

### Why MCTS or graph search alone is not enough

MCTS and classical planners require a complete successor function `next_states(s)`.
In this system, the recovery successor set is **not fully enumerable** because:

- **Failure modes are open-ended.** The system handles novel disruptions that
  are not covered by pre-authored recovery templates.
- **Cross-resource reassignment is not trivially enumerable.** Recovering from a
  fault on one resource may require recruiting a different resource type with a
  different capability vocabulary.
- **Heterogeneous resources expose different state vocabularies.** A robot
  profile and a printer profile have different facets, primitives, and state
  projections. The combinatorial cross-product is too large and too sparse to
  enumerate manually for every failure/resource combination.
- **Recovery strategies require domain reasoning** that is not captured in the
  JSON manifests alone — for example, deciding that a part stuck in a failed
  gripper should be re-approached from a different angle rather than retried.

### What the LLM actually contributes

The LLM acts as a **learned approximate successor generator**: given the current
symbolic state, the failure context, and the allowed execution surface, it
proposes plausible recovery actions that were not explicitly hand-authored.

It does **not** own selection, validation, or state evolution. Those are
deterministic and must remain so.

A weak justification would be: "the LLM picks the best next step."

A strong justification is: **"the LLM expands the recovery successor set under
novel, heterogeneous, partially modeled failures; deterministic mechanisms then
filter and choose."**

### What the deterministic machinery owns

Once the LLM proposes candidate recovery actions:

1. The **grounding compiler** resolves symbolic references to concrete runtime
   values.
2. The **feasibility oracle** validates workspace reachability, gripper
   occupancy, part-holder compatibility, and resource availability.
3. The **safety checker** validates against LTL/FSA safety rules.
4. The **DES state projector** applies symbolic effects to advance the state.
5. The **selector** (scoring or search) commits the best valid candidate.

The LLM proposes; formal methods dispose.

## 9. Current Practical Takeaway

The active bridge is now a compact v4 path made of:

- state normalization
- primitive semantics
- prompt preparation
- single-shot bridge-session packaging
- one-shot raw LLM execution

If you remember only one thing, remember this:

The current active bridge prepares a structured `llm_input`, builds a
multi-resource composite prompt-facing action surface inside it, sends one
single-shot prompt to the LLM, and records the raw reply before proposal
parsing and validation.

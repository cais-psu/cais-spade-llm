# TSS-Based LLM Bridge v4: Design Intent

This note describes what the new bridge is trying to do, what is implemented
right now, and what still needs to be built. The v4 direction is a hybrid
bridge: DES remains the nominal planner, the bridge starts only after DES can
no longer find a modeled continuation, the bridge prepares a curated state
package for LLM reasoning, and symbolic checks remain responsible for deciding
whether a proposed recovery is acceptable.

There is no separate runtime `replan_mode="llm"` path in this v4 direction.
DES owns runtime replanning, and the LLM appears only inside the bridge
fallback after DES cannot produce a modeled continuation.

## 1. What Problem v4 Is Solving

The v4 bridge is for runtime disruptions that happen after nominal execution has
already started. In that situation, the system cannot safely recover by
inventing arbitrary tools or by guessing a missing step from raw logs alone.

Instead, the bridge needs to reason over:

- the current runtime state
- loaded safety rules
- relevant assembly requirements
- the modeled continuation gap

The goal is not to replace DES. The goal is to return the system to a state
where the pending nominal DES suffix can resume.

## 2. What Is Implemented Now

The active bridge now goes one step past prompt preparation: it sends one
single-shot prompt to the configured LLM and records the raw reply.

Implemented pieces:

- `prepare_bridge_request(...)` builds the prepared bridge package.
- The active bridge core files are now native top-level v4 files:
  - `llm_bridge/bridge_resource_normalization.py`
  - `llm_bridge/primitive_semantics.py`
  - `llm_bridge/bridge_prompts.py`
  - `llm_bridge/bridge_session.py`
- `prepared_bridge_request["llm_input"]` now exists and is the main
  model-facing structured payload.
- `prepared_bridge_request["single_shot_prompt_input"]` now exists.
- `prepared_bridge_request["single_shot_prompt_text"]` now exists.
- `execute_prepared_bridge_request()` now:
  - ensures the single-shot prompt artifacts exist
  - calls `ask_llm(...)` once
  - records the raw reply
  - sets bridge status to `llm_output_recorded`

The human-facing prepare trace now shows:

- `Fault Event`
- `Current Product State`
- `Loaded Safety Rules`
- `Relevant Assembly Requirements`
- `Modeled Continuation Gap`

The active bridge does **not** yet parse, normalize, validate, or compile the
returned proposal. The current v4 checkpoint is:

1. build the bridge package
2. build the final single-shot prompt
3. send one LLM request
4. record the raw output

The older direct online full-LLM replanner is no longer the active runtime
direction. Offline LLM plan repair still exists separately.

## 3. Important Meaning of the Current `llm_input`

The current `llm_input` is intentionally not "raw observation only." It is a
curated bridge context with three different kinds of information.

### Observed or runtime-grounded

- failure event facts
- resource snapshots
- resource state after failure
- part states
- observed part poses

### Loaded or model-backed

- safety rules loaded from bundle artifacts
- assembly requirements loaded from bundle and planner artifacts

### Deterministically derived

- goal predicates
- unmet goal conditions
- continuation gap
- resumability checks
- allowed execution surface for the current bridge turn

Important clarifications:

- this is **not** the entire debug artifact
- this is **not** raw observation only
- this is the curated bridge context intended for direct LLM use

## 4. What The Current Model-Facing Action Surface Looks Like

The prompt-facing primitive surface now lives **inside** `llm_input` rather
than as a separate prompt-only attachment.

Specifically:

- `llm_input["allowed_execution_surface"]` now exists
- it is **multi-resource**
- it is **runtime-derived**
- it is **prompt-facing composite-only**

That action surface contains:

- `focused_resource_jid`
- `resources[]`

Each resource entry includes:

- `resource_jid`
- `role`
- `current_state`
- `current_location`
- `availability`
- `held_part`
- `pending_task_ids`
- `prompt_bridge_snapshot`
- `prompt_primitive_catalog`

For robot resources, the prompt-facing primitive catalog now hides raw
gripper/attachment steps such as:

- `open_gripper`
- `close_gripper`
- `attach_part`
- `detach_part`

and instead exposes prompt-facing composite entries such as:

- `grasp_part`
- `release_part`

The raw execution primitive catalog still exists underneath for later
validation and execution work, but it is no longer what the LLM sees directly.

## 5. Prompt Shape Improvements Already Landed

The single-shot prompt is no longer the older focused-only surface.

It now includes:

- runtime-derived `blocked_at_task_id` and `blocked_at_function`
- `Proposal Success Criteria`
- `Allowed Execution Surface` rendered from
  `llm_input["allowed_execution_surface"]`
- explicit wording that macro tasks may span multiple listed resources
- guidance to prefer the smallest coordinated recovery that restores the
  protected suffix

It also removed several earlier prompt-time duplications:

- no full prompt-time `bridge_resources` dump
- no separate focused-only primitive catalog field
- no separate focused bridge snapshot field
- no separate other-resource summary field

## 6. Debug / Inspection Artifacts

The dry-run and live runtime path now persist prompt inspection artifacts for
the active single-shot flow.

Current practical artifacts include:

- the latest rendered prompt text
- timestamped prompt snapshots
- the latest raw LLM output text
- timestamped raw LLM output snapshots

This makes it possible to inspect both the exact prompt and the exact raw model
reply outside the terminal.

## 7. What v4 Still Needs

The next missing pieces are on the proposal and validation side.

- normalize returned `macro_tasks[]`
- validate:
  - bridge safety
  - marked re-entry
  - modeled continuation restoration
- compile accepted proposals into runtime recovery tasks

The intended authority split is:

- LLM for proposing a recovery
- deterministic validators for approval or rejection

## 8. Proposed v4 Execution Shape

The intended runtime flow is:

1. DES fails to find a modeled continuation.
2. The bridge prepares `llm_input`.
3. The bridge prepares the final single-shot prompt.
4. The bridge sends one single-shot LLM request.
5. The raw reply is recorded.
6. The proposal is normalized.
7. The proposal is validated against safety and continuation checks.
8. The accepted proposal is compiled into recovery macros.
9. Runtime resumes under approval gating.

This means the bridge may invent recovery structure, but it must still do so
inside the approved primitive and macro execution surface.

## 9. Relation to Existing Files

How to read the current bridge docs and code:

- `tss_based_llm_bridge_v3.md`
  Prior bridge framing and the older bridge-first architecture.
- `llm_bridge_construction.md`
  Code-organization and runtime construction note for the active bridge module
  layout.
- `llm_bridge/bridge_session.py`
  Active prepare-trace implementation, `llm_input` builder, single-shot prompt
  preparation, and one-shot raw LLM execution.
- `llm_bridge/bridge_prompts.py`
  Active v4 prompt builder for the single-shot prompt and its rendered text.
- `llm_bridge/primitive_semantics.py`
  Active v4 primitive layer, including prompt-facing composite primitives such
  as `grasp_part` and `release_part`.
- `test/test_case3_bridge_dryrun.py`
  Dry-run harness for inspecting the current bridge package, prompt, and raw
  LLM reply.

So the split is:

- `v3` explains the prior bridge framing
- `construction` explains how the active bridge code is organized
- `v4` explains what the new bridge is trying to achieve

## 10. Guardrails and Non-Goals

- The bridge must not invent new runtime execution surfaces.
- The bridge must not bypass safety or continuation validation.
- The debug artifact is not the prompt by default.
- Observed facts and modeled expectations should stay separate.
- The current v4 slice has **raw** LLM generation, but not accepted proposal
  execution yet.

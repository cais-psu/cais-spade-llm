# TODO: Unify Product Planning and Safety with a Shared Typed IR

## Goal

Replace the current partially shared, string-heavy planning/safety flow with one explicit typed layer that both sides compile against:

```
NL
  -> RequirementIR / SafetyRuleIR
  -> TaskIR / EventSpec / APSpec
  -> DAG/FSA and LTLf/DFA
  -> Runtime monitoring + replanning
```

The key design rule is:

- LLMs may propose structured JSON
- deterministic code must ground that JSON to canonical symbols before it is trusted

This is meant to eliminate mismatches such as:

- `MCP part` vs `MCP`
- planner task names that do not match safety APs
- geometry/safety grounding happening only after execution starts
- safety and product semantics meeting only through failure + feedback

## Shared IR to Introduce

Add a new package:

- `cais_spade_llm/domain/`

Files:

- `cais_spade_llm/domain/models.py`
- `cais_spade_llm/domain/schema.py`
- `cais_spade_llm/domain/grounding.py`
- `cais_spade_llm/domain/compilers.py`
- `cais_spade_llm/domain/validators.py`

### models.py

Add typed data models for:

- `DomainSchema`
- `RequirementIR`
- `TaskIR`
- `SafetyRuleIR`
- `EventSpec`
- `APSpec`
- `GroundingIssue`

Suggested fields:

```python
RequirementIR:
  id
  part
  source
  destination
  quantity
  goal_state
  resource_candidates
  raw_text

TaskIR:
  id
  requirement_id
  resource
  action
  part
  source
  destination
  zone
  preconditions
  effects
  raw_origin

SafetyRuleIR:
  id
  type
  resources
  parts
  zones
  trigger_event
  required_event
  forbidden_event
  raw_text

EventSpec:
  name
  action
  resource
  part
  zone
  phase

APSpec:
  label
  predicate
  args
  source_rule_id
```

### schema.py

Build one canonical vocabulary from existing repo state:

- product manifests in `cais_spade_llm/initialization/products/`
- resource manifests in `cais_spade_llm/initialization/resources/`
- actions/tools from `cais_spade_llm/initialization/tools.json`
- geometry files in `cais_spade_llm/specification/products/geometry/`

Suggested class:

- `DomainSchemaBuilder.build(product_meta, resource_meta, tools_catalog, geometry_payload) -> DomainSchema`

Canonical symbol sets:

- resources: `ur5e`, `xarm6`
- actions: `pick_approach`, `pick_grasp`, `place_approach`, `place_insert`, `move_home`
- parts: canonical product part names
- zones/locations: printers, assembly board, slots, staging areas
- event predicates: `task_started`, `task_completed`, `entered_zone`, `holding_part`, `returned_home`

### grounding.py

Add deterministic grounding and canonicalization:

- `ground_requirement_ir(raw_json, schema) -> RequirementIR, list[GroundingIssue]`
- `ground_safety_rule_ir(raw_json, schema) -> SafetyRuleIR, list[GroundingIssue]`
- `canonicalize_part_name(...)`
- `canonicalize_resource_name(...)`
- `canonicalize_zone_name(...)`
- `canonicalize_action_name(...)`

If a field cannot be grounded, do not silently guess. Return an explicit issue.

### validators.py

Add validation passes:

- `validate_requirement_ir`
- `validate_task_ir`
- `validate_safety_rule_ir`
- `validate_ap_specs`
- `validate_task_safety_alignment`

Reject or flag:

- unknown part/resource/action/zone
- missing source/destination
- missing geometry for required placement actions
- safety rules referencing symbols that do not exist in the task/event model

### compilers.py

Add deterministic compilers:

- `compile_requirement_ir_to_tasks(...) -> list[TaskIR]`
- `compile_task_ir_to_dag_nodes(...) -> list[dict]`
- `compile_task_ir_to_event_specs(...) -> list[EventSpec]`
- `compile_safety_rule_ir_to_ap_specs(...) -> list[APSpec]`
- `compile_safety_rule_ir_to_ltlf(...) -> str`

## Exact Files to Refactor

### 1. `cais_spade_llm/agents/intelligent_product/process_planner.py`

Current role:

- parses NL requirements
- expands them to tasks
- compiles global FSA

Target refactor:

- keep `_llm_parse_requirements(...)`, but treat its output as raw candidate JSON only
- immediately ground into `RequirementIR`
- store grounded requirements as the planner's source of truth
- replace string-based task expansion with:
  - `RequirementIR -> TaskIR`
  - `TaskIR -> DAG nodes`
- keep existing DAG/FSA JSON artifacts during transition, but derive them from `TaskIR`

Add methods:

- `_build_domain_schema(...)`
- `_ground_requirements(...)`
- `_compile_requirement_ir_to_tasks(...)`
- `_serialize_task_ir(...)`

Patch order inside this file:

1. add `RequirementIR` storage alongside existing `requirements`
2. add grounding immediately after `_llm_parse_requirements`
3. derive existing task node format from `TaskIR`
4. update `save()` payloads to persist both legacy and IR forms

### 2. `cais_spade_llm/agents/intelligent_product/product_agent.py`

Current role:

- loads product spec text
- builds plan
- enriches execution params with geometry
- dispatches tasks

Target refactor:

- load or build `DomainSchema` at startup
- emit runtime events using typed event names instead of ad hoc task metadata only
- use `TaskIR`-derived task messages as the source of execution fields

Add methods:

- `_build_domain_schema()`
- `_task_ir_to_execution_payload(task: TaskIR) -> dict`
- `_emit_runtime_event(...)`
- `_validate_task_payload_against_schema(...)`

### 3. `cais_spade_llm/agents/central_controller/safety_logic.py`

Current role:

- parse safety NL
- infer APs / rule family
- compile LTLf
- build DFA

Target refactor:

- keep `_llm_parse_safety_rules(...)`, but ground into `SafetyRuleIR`
- make `SafetyRuleIR` the semantic source of truth
- derive APs from `TaskIR/EventSpec` and schema symbols
- reduce reliance on free-text family inference as the main compiler input

Add methods:

- `_build_domain_schema(...)`
- `_ground_safety_rules(...)`
- `_compile_rule_ir_to_ap_specs(...)`
- `_compile_rule_ir_to_ltlf(...)`

Patch order inside this file:

1. persist `SafetyRuleIR` alongside current `rules`
2. generate APs from typed fields
3. keep current LTLf/DFA outputs but mark them as compiled artifacts from IR
4. gradually remove heuristic-only rule compilation paths

### 4. `cais_spade_llm/agents/central_controller/base_safety_checker.py`
### 5. `cais_spade_llm/agents/central_controller/offline_safety_validator.py`
### 6. `cais_spade_llm/agents/central_controller/online_safety_monitor.py`

Current role:

- consume task/runtime state and evaluate safety

Target refactor:

- consume a canonical event/AP stream
- use `EventSpec` / `APSpec` consistently for both offline and online checks

Add/adjust:

- one event normalization layer
- one AP evaluation layer shared by offline and online validation

### 7. `cais_spade_llm/agents/resource_agent/robot_agent.py`

Current role:

- executes actions and returns statuses

Target refactor:

- emit typed runtime events such as:
  - `task_started`
  - `task_completed`
  - `entered_zone`
  - `returned_home`
  - `grasp_succeeded`
  - `insert_completed`

These events should be the same ones safety compilers and monitors reason about.

### 8. `cais_spade_llm/bundles/bundle_compiler.py`

Current role:

- compile and store verified plan bundles

Target refactor:

Persist the new IR artifacts:

- `catalog/domain_schema.json`
- `plan/requirements_ir.json`
- `plan/tasks_ir.json`
- `safety/safety_rules_ir.json`
- `catalog/event_specs.json`

Keep these legacy artifacts during migration:

- `plan/*.json`
- `global_fsa.json`
- `cca_safety_logic.json`
- DFA `.dot` / `.png`

### 9. `cais_spade_llm/ui/bridge.py`

Current role:

- UI/runtime boundary
- bundle/source resolution
- preview generation

Target refactor:

- expose IR artifacts to UI pages
- surface grounding/validation errors
- stop using implicit fallbacks where possible

Add methods:

- `get_domain_schema_preview(...)`
- `get_requirement_ir_preview(...)`
- `get_safety_rule_ir_preview(...)`
- `get_grounding_issues(...)`

### 10. UI Pages

Files:

- `cais_spade_llm/ui/pages/plans.py`
- `cais_spade_llm/ui/pages/safety.py`

Target changes:

- show parsed IR before compiled outputs
- show unresolved symbols
- show which canonical symbols each requirement/rule grounded to
- show why generation was blocked if grounding failed

### 11. `cais_spade_llm/prompts.py`

Current role:

- prompt templates for requirement parsing, task expansion, safety parsing, interpretation

Target refactor:

- pass schema vocabulary into prompts
- require JSON fields that map to schema types
- tell the LLM to return unresolved fields explicitly instead of inventing unseen symbols

## Constraint Injection Before Execution

This is the key conceptual improvement.

Do not wait until runtime monitoring to make safety and product logic meet.

Apply `SafetyRuleIR` as planning constraints in `process_planner.py`:

- mutual exclusion in a zone
  - add no-overlap or ordering constraint
- precedence rule
  - add required DAG edge
- response rule
  - add successor task requirement, e.g. `move_home` after `place_insert`

Suggested implementation:

- add `apply_safety_constraints_to_task_ir(tasks, safety_rules, schema) -> tasks`

This should run before:

- `compile_global_fsa()`
- offline bundle validation

## Migration Strategy

### Phase 1: Add IR Without Breaking Existing Outputs

- add `domain/` package
- persist `RequirementIR` and `SafetyRuleIR`
- keep current JSON structures and execution path

### Phase 2: Make Product Planning IR-First

- `NL -> RequirementIR -> TaskIR -> legacy DAG`
- compare legacy and IR-derived outputs during transition

### Phase 3: Make Safety Compilation IR-First

- `NL -> SafetyRuleIR -> APSpec -> LTLf -> DFA`
- keep current rule preview UI, but source it from `SafetyRuleIR`

### Phase 4: Standardize Event Emission

- all runtime monitors consume the same typed event vocabulary

### Phase 5: Remove Heuristics and Fragile Fallbacks

Examples to remove or reduce:

- first-file-in-directory requirement fallback
- text-only safety family inference as semantic truth
- AP generation that is not tied to shared typed symbols

## Tests to Add

New tests:

- `test/test_domain_schema.py`
- `test/test_domain_grounding.py`
- `test/test_requirement_ir_compilation.py`
- `test/test_safety_rule_ir_compilation.py`
- `test/test_event_ap_alignment.py`

Extend existing tests:

- `test/test_bundle_bridge_compatibility.py`
- `test/test_product_precomputed_bundle.py`
- `test/test_safety_logic_grounding.py`

Key assertions:

- `MCP part` grounds to `MCP`
- missing geometry becomes a grounding/validation error before execution
- safety APs reference canonical task/event symbols
- plans and safety rules compile against the same `DomainSchema`
- bundle artifacts include IR payloads

## First Three Patches to Make

### Patch 1

Add the new `domain/` package with:

- `models.py`
- `schema.py`
- `grounding.py`

This is the foundation and can be added without breaking anything.

### Patch 2

Refactor `process_planner.py` to:

- store `RequirementIR`
- ground parsed requirement JSON
- derive legacy task nodes from `TaskIR`

### Patch 3

Refactor `safety_logic.py` to:

- store `SafetyRuleIR`
- ground parsed safety JSON
- derive APs/LTLf from typed symbols instead of only free-text heuristics

## Expected Payoff

- fewer planner/safety mismatches
- fewer runtime failures caused by naming drift
- better UI explainability
- cleaner bundle reproducibility
- stronger publishable story: formal safety + typed IR + human refinement + multi-agent execution

# Journal Paper 2: Recovery Framework

Start with [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), the single current
roadmap for formal validation, deterministic safety, local composition, the
four failure scenarios, UI dry runs, and experiments.

## Document index

| Document | Purpose |
| --- | --- |
| [Implementation plan](IMPLEMENTATION_PLAN.md) | Work order, implementation decisions, acceptance gates, and experimental design. |
| [Formal validation and selection reference](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md) | Model notation, PA/RA/CCA authority, current validator behavior, target contracts, and known gaps. |
| [Implementation history](IMPLEMENTATION_HISTORY.md) | Preserved dated narratives, original handoffs, verification, and evidence limitations. |
| [M1/M2 layout](MACHINING_STATION_LAYOUT.md) | Geometry, exact resource placement, access assumptions, and [layout drawing](MACHINING_STATION_LAYOUT.svg). |
| [Resource functions and primitives](RESOURCE_FUNCTIONS_AND_PRIMITIVES.md) | Saved function transitions, primitive catalogs, execution support, and simulation limits. |
| [Controller command reference](CONTROLLER_COMMAND_REFERENCE.md) | Actual controller bindings and distinctions from possible future device commands. |
| [Requirement/capability matching](ADAPTIVE_REQUIREMENT_CAPABILITY_MATCHING.md) | Product requirements, resource capabilities, and versioned processPlan design. |

## Current status

The retained nominal Gazebo baseline records 11 placed parts and 195 matching
function transitions. Its diagnostic CCA bypass and simulation assumptions remain
part of that evidence; it does not establish active-rule safety or recovery in
the four proposed failure scenarios. See [the retained run record](../../cais_spade_llm/monitor/recovery_gazebo_runs/attempt-f2090c8976d84acd90cf35353c5acc34/README.md).

The project UI has **run → setup → results → recovery** tabs. The existing
**recovery → Test a failure scenario** panel runs diagnostics from saved context.
It now uses a reusable runner, frozen inputs, shared request capture, and detailed
RA/CCA evidence. The supplied manuscript is preserved in the formal reference.
The [roadmap status table](IMPLEMENTATION_PLAN.md#current-status-and-dependencies)
separates the implemented validation/inspection slice from remaining work:
four-scenario fixtures and execution, deterministic safety construction, and
local composition. Historical status wording remains in the history file.

## Working Thesis

CAIS-SPADE-LLM combines formal recovery, LLM proposals, agent-owned validation,
and observed execution for manufacturing recovery. See the
[formal model and claim boundaries](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md#formal-model-and-authority).

## Experiment Plan

Follow the [experimental design](IMPLEMENTATION_PLAN.md#experimental-design),
including Gazebo-first recovery trials, composition scaling, deterministic safety,
and prompt-integrity checks. These are planned comparisons, not collected results.

## Neurosymbolic Selection Method

Use the [formal selection definition and current implementation gap](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md#neurosymbolic-selection-method).
The [roadmap](IMPLEMENTATION_PLAN.md#milestone-1-formal-validation-contract)
tracks the outstanding strict-expansion correction.

# Adaptive matching of product requirements to resource capabilities

**Status: processPlan matching and resource graph representation are implemented.**
Controller integration and live execution of these process plans are outside this change.
The separate KMR Storage-to-M1 delivery retains
the configured M1/square and M2/circular assignments. This document does not
enable machine reassignment, failure injection, or recovery.

## Process plans and resource graphs

The active Product Order requests ordered steps without assigning transport routes:

```json
{
  "processPlan": {
    "KET4_Square_4mm": [
      {"processesToComplete": [{"process": "trim", "result": "square"}]},
      {"processesToComplete": [{"process": "assembly"}]}
    ]
  }
}
```

Every process in a step must complete before the next step begins. Matching uses
declared `product_effects`; `assembly` binds the exact feature in
`parts.assembly_target_map`. The task model supplies `pick_approach`, `pick_grasp`,
`place_approach`, `place_insert`, and `move_home` where their guards permit them.
An acknowledged `place_insert` records `assembly` and its exact target internally;
ordinary `place_release` records placement. Feasibility, custody, CCA, execution
adapters, and completion validators retain their existing authority.

New processPlan runs and exported process descriptors use `schema_version: 3`.
Orders containing historical `requirements` retain version 2 effects and replay.
Version 1 delivery records retain their separate projector. Saved completion
records are never migrated or rewritten.

Resources displays a conditional part and task graph built from structured
transition conditions and effects. Machining progress survives later handoffs;
KMR movement retains custody; shared preparation and `move_home` events retain
their responsible resource. Every edge identifies its existing `event_id`, with
the full participant guards, updates, product effects, and collection constraints
available in the same panel. Graphs bind `part_name` at task time and do not create
a copy for every inventory item. They describe declared capabilities, not live
occupancy or an execution schedule. Conveyor advances still move all resident
parts together; buffer capacity and staging guards still apply.

The sections below preserve the delivery implementation's earlier design and
migration rationale.

## Current model and framework relationship

The implementation adapts the separation in the read-only
`/home/jongh/projects/SemiconductorSimulation_testing-master` repository:

| Responsibility | Semiconductor framework | CAIS-SPADE-LLM |
| --- | --- | --- |
| Requirements, identity, selected plan, history | `Agents/intelligentProduct/ProductAgent.java`: `partName`, `productionPlan`, `plan`, `productHistory` | ProductAgent and its nominal context own one `assembly_board-v1` order and each selected component's requirements, state, and history. |
| Location and process information | `Agents/sharedInformation/ProductState.java`: `location`, `processCompleted` | `product/state.py` retains `location`, `state`, `last_task`, and exact `processCompleted` identifiers. |
| Capabilities and constraints | `Agents/resourceAgent/ResourceAgent.java`: `actionsGraph`, `capabilitiesPTA`, `query` | ResourceAgent exposes its nominal descriptor and authoritative resource valuation; guards constrain parameterized task events. |

The Java implementation also contains timed constraints, costs, neighbor
queries, and scheduling. The CAIS nominal implementation uses deterministic
search and acknowledged task transitions. It does not reproduce those timed
or distributed algorithms. Describe the research relationship as an adaptation
of the agent architecture, with separately evaluated extensions.

The commissioned delivery path uses configured assignments and exact inventory.
Square pegs use M1 and circular pegs use M2. Parameterized capabilities avoid
one event per inventory item, but removing this duplication does not implement
requirement-to-capability matching. The delivery order specifies a terminal
valuation; ordinary `place_release` into M1 records `loaded`, with no machining
or assembly completion.

## Prospective matching

For example, **square pegs require `p1`; a machine supports `p1`**. Here `p1`
would be an explicitly configured process requirement, not a renamed task.
The executable task identifier remains `machine_part`. This delivery increment adds no `p1` token to its active requirements or
descriptors. Concurrent capability-model changes are preserved separately; they
do not establish adaptive Gazebo machine execution.

ProductAgent would request the next unmet process requirement independently
of a particular machine. ResourceAgent would advertise the supported process
and its applicable constraints. ProcessPlanner would bind an exact component
and candidate resource, obtain a feasible transport and processing sequence,
and validate the complete participating valuations. ProductState would record
the configured completed process only after matching successful execution
evidence. Transport preserves completed processes; `place_insert` alone
establishes component assembly.

Matching a requirement is necessary but does not establish execution
feasibility. Before selecting a machine, check:

- tooling and its configured process;
- workholding, component dimensions, and orientation;
- loading openings and collision-aware manipulator reach;
- available configured transport routes and docking poses;
- occupancy, custody, reservations, resource availability, and permitted resources;
- applicable CCA requirements and the responsible ResourceAgent's execution support.

Overlapping capabilities would allow M1 and M2 to be candidates for the same
requirement only after their configurations support it. A breakdown cannot
make an otherwise incompatible machine eligible. Partially machined WIP also
needs an explicit remaining-process requirement and a supported extraction
and loading operation. Preserve acknowledged progress when replanning.

Exact part identities remain necessary for inventory, CAD/Gazebo geometry,
occupancy, custody conservation, selected Product Orders, and execution history.
`KET4_Square_4mm` remains that exact component; `p1` would not replace its name.
Do not alias `LG` or any retired scenario to a NIST component.

## Migration and verification

1. Add explicit, versioned requirement and supported-process configuration.
   Keep existing orders and assignment-based descriptors valid during migration.
   Require an explicit migration; never infer process identity from a part name.
2. Define how each supported `machine_part` binding satisfies a requirement.
   Use the same declared effects for planning, projection, acknowledgement, and
   ProductState updates. Preserve existing `machine_part` histories and saved
   descriptor snapshots without relabeling them as `p1` executions.
3. Extend capability requests and deterministic planning to evaluate every
   compatible permitted resource. Record rejected candidates and their failed
   constraints. Retain search limits and explicit blocked outcomes.
4. Display requirements in Products and supported processes with constraints
   in Resources. Setup chooses permitted resources and failure settings; it
   must not duplicate either definition editor.
5. Test legacy orders, missing/unknown requirements, incompatible tooling and
   workholding, overlapping capabilities, unavailable resources, blocked routes,
   unreachable loading poses, and deterministic selection/replanning.
6. Re-run Conveyor ordering, shared movement, staging, buffer backpressure,
   complete assembly and Exit, custody conservation, and ordinary placement
   regressions. Reject failed, stale, mismatched, and duplicate acknowledgements
   without manufacturing process completion.
7. Verify saved-run replay against its recorded definitions and migration
   version. Compare assignment-based and adaptive experiments separately,
   reporting planning results, simulated execution, and hardware results as
   distinct evidence.

Failure injection, LLM recovery, selector corrections, and distributed timed
negotiation remain separately scoped work in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) and
[JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md).

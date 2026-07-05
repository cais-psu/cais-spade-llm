# Journal Paper 3: Ontology RAG

## Working Thesis

CAIS-SPADE-LLM can improve flexible and accurate execution by moving from
agent-requested context retrieval to PPR-grounded Ontology RAG. The graph
retrieves Product, Process, Resource, primitive, fact, and constraint context
before generation, then the existing validators check the generated primitive
program before execution.

This paper should focus on dynamic retrieval, GraphRAG, PPR ontology structure,
and resource-agent primitive composition.

## Core Idea

The current framework already supports dynamic retrieval. The LLM can request
specific refs such as resource snapshots, part targets, primitive catalogs,
primitive contracts, safety rules, and capability decompositions.

The proposed improvement is to make retrieval graph-directed:

1. Build a PPR execution graph from product state, process plans, resource
   profiles, primitive contracts, safety constraints, and runtime observations.
2. Retrieve the relevant subgraph for the active task or recovery event.
3. Use that subgraph to constrain which resources, tasks, primitives, facts,
   and symbols are visible to the LLM.
4. Generate primitive_steps inside that constrained context.
5. Validate the primitive program and write execution results back to the graph.

## PPR Mapping

- Product:
  - product id, part names, required final state, current part state,
    destination geometry, assembly board slots.
- Process:
  - task DAG, recovery task, bridge event, primitive sequence,
    preconditions, effects, expected start/end state, safety/DES constraints.
- Resource:
  - xArm6, UR5e, grippers, sensors, execution mode, current state,
    held part, capabilities, primitive catalog, workspace limits.

The graph relation of interest is:

`Product goal -> Process step -> Resource capability -> Primitive program -> event_facts -> validated execution`

## Main Contributions

- A PPR-grounded retrieval model for resource-agent primitive composition.
- A graph retrieval layer that proactively selects execution-relevant context
  instead of relying only on LLM-requested refs.
- A constrained generation surface that reduces invalid resource choices,
  primitive ordering errors, missing event_facts, and invalid symbols.
- A runtime feedback loop where execution results update the graph and improve
  later retrieval.

## Before And After

Before:

- The LLM receives a prepared context and can ask for refs.
- The LLM must know which refs are needed.
- Primitive catalogs are mostly filtered by resource visibility.
- Invalid sequences can be generated first and rejected later.

After:

- The PPR graph retrieves relevant Product, Process, Resource, primitive, fact,
  and constraint context before generation.
- The LLM sees a smaller and more correct control space.
- Resource/task/primitive choices are constrained by current graph state.
- Validation still runs, but fewer invalid proposals should reach validation.

## Example Retrieval Query

Input:

- Active recovery event: place MCP.
- Current state: xArm6 is failed, UR5e is idle or holding the part.
- Product goal: MCP must be placed at assembly_board-v1.

PPR Ontology RAG retrieves:

- Product node: MCP and target slot geometry.
- Process node: place or recover_place process.
- Resource node: UR5e if xArm6 is failed or not enabled.
- Primitive nodes: compute_place_targets, move_cartesian, release_part,
  move_relative.
- Fact nodes: event_facts.place_targets.MCP.
- Constraint nodes: expected_start_state, release_target_grounded,
  plant_enabledness, safety rules.

The generator then produces primitive_steps from a graph-filtered context instead
of from broad prompt context.

## Relationship To Journal Paper 2

Journal Paper 2 explains the recovery framework and its current validation
pipeline. Journal Paper 3 builds on that system and improves the retrieval layer.

Journal Paper 2 baseline:

`LLM asks -> bridge serves requested context -> LLM generates -> validator checks`

Journal Paper 3 proposed system:

`PPR graph retrieves -> LLM generates inside constrained context -> validator checks -> execution updates graph`

## Related Work Positioning

Use ontology-based feedback for runtime control in multi-agent manufacturing
systems as a foundation. The extension is not just storing runtime history in an
ontology. The extension is using PPR Ontology RAG to control retrieval for
primitive composition and execution validation.

Potential framing:

> Building on ontology-based runtime feedback for multi-agent manufacturing, we
> propose a PPR-grounded Ontology RAG framework that retrieves execution-relevant
> product, process, resource, primitive, fact, and constraint context for
> resource-agent primitive composition.

## Paper Outline

1. Introduction
   - Flexible manufacturing requires correct context retrieval for LLM-assisted
     control.
   - Current agentic retrieval helps but depends on what the LLM asks for.
   - PPR Ontology RAG can proactively retrieve execution-relevant context.

2. Related Work
   - PPR ontology models.
   - Ontology-based feedback in multi-agent manufacturing.
   - GraphRAG and retrieval-augmented generation.
   - LLM planning for robotics and manufacturing.
   - Resource-agent capability and primitive composition.

3. Current Baseline Framework
   - Agent-requested dynamic context retrieval.
   - Published refs and primitive catalogs.
   - Resource-agent primitive generation.
   - Validation and runtime execution.

4. PPR Ontology RAG Framework
   - Graph schema.
   - Runtime graph update.
   - Retrieval query types.
   - Context assembly for generation.
   - Feedback after validation and execution.

5. Primitive Composition With Graph Retrieval
   - Product-driven retrieval.
   - Process-driven retrieval.
   - Resource-driven retrieval.
   - Fact and constraint retrieval.
   - Generation and validation loop.

6. Experiments
   - Current dynamic retrieval vs PPR Ontology RAG.
   - Recovery scenarios with wrong resource, missing facts, invalid destination,
     failed resource, and safety constraints.

7. Discussion
   - Benefits and limits of graph-directed retrieval.
   - When the LLM should still request extra context.
   - Generalization beyond xArm6 and UR5e.

8. Conclusion
   - PPR Ontology RAG improves retrieval precision for flexible and accurate
     execution.

## Experiment Plan

Compare:

- Full context prompt.
- Current agentic dynamic retrieval.
- PPR Ontology RAG retrieval.
- PPR Ontology RAG plus validation feedback.

Measure:

- Correct resource selection rate.
- Correct primitive ordering rate.
- Missing event_facts rate.
- Invalid symbol rate.
- Number of context turns.
- Primitive program acceptance rate.
- Recovery execution success rate.

## Figures To Produce

- PPR execution graph schema.
- Current retrieval vs PPR Ontology RAG retrieval flow.
- Example retrieved subgraph for one recovery event.
- Primitive composition pipeline.
- Experiment comparison chart.

## Immediate Writing Tasks

- Define the minimum graph schema.
- Choose one running example from the recovery framework.
- Write the baseline vs proposed retrieval example.
- Decide whether the implementation uses JSON-backed graph, RDF/OWL, or a
  hybrid internal graph for the first experiments.

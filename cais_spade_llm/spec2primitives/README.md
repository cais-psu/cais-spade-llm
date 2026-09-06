# Spec2Primitives

Spec2Primitives is the isolated ICRA 2027 case study for product-specification-driven dynamic primitive composition. PA grounds instances under the supplied PPR TBox; it does not discover or modify the ontology schema.

## Current implementation

Product grounding and arm assignment complete when the PA-authored goal passes deterministic structure/evidence validation and PA assigns a configured capable arm whose grounded locations pass reachability checks. The production profile contains `assembly`, `xarm6`, and `ur5e`. Capability declarations alone do not establish reachability or a primitive sequence.

One `target_feature` represents the requirement. For assembly, `assembly_feature_association` is a collection of pairwise relationships. Each has exactly two `AssemblyFeature` endpoints and explicit `state_names`. A desired relationship can reference two currently observed objects. Document-only endpoints may remain unbound; unresolved required identities or destinations prevent completion.

For example, if a requirement specifies a gear over a shaft, current evidence may locate both objects. The desired state describes their required assembled relationship and references the observed shaft as the destination. An observation belongs in `desired_state.state_values` only when evidence supports its role in the required outcome; identifying the moving gear at its current location is insufficient. A source-supported observation may belong to both states when it independently supports both roles. The destination reference does not assert that assembly is already complete or that the shaft's center is the final insertion pose. A document figure alone cannot prove current installation. See [the ontology guide](ASSEMBLY_ONTOLOGY.md).

## Grounding flow

```text
exact requirement and operator clarification
→ PA investigates approved document, CAD, and pinned RGB-D evidence
→ PA proposes current/desired states and assembly relationships
→ host validates structure, citations, bindings, and hashes
→ return generic validation feedback when correction is needed
→ continue PA investigation within shared operation/proposal budgets
→ commit only the deterministically valid proposal
→ derive every grounded current and destination coordinate reference
→ PA checks every capable arm and submits its cited selection
→ revalidate capability and live MoveIt reachability
→ commit processExecution and runsOnResource
→ persist current completion
```

PA controls `retrieve`, `query_document`, `compare_cad_size`, and `analyze_candidate_layout`. Configured `grounding_limits` default to 24 evidence operations and 6 proposals shared across the investigation. The snapshot and observation mapping stay pinned across corrections. Deterministic feedback identifies invalid structure/references without supplying a replacement answer. Budget exhaustion, repeated invalid proposals without progress, changed evidence or unavailable prerequisites leave grounding incomplete. Rejected assertions never merge.

Document retrieval supplies a deterministic source index with page text/images; PA can author its own document question. CAD comparison reports all candidate measurements without selecting a winner. Same-view layout analysis reports geometry without a built-in task-role answer.

PA owns semantic interpretation, whole-goal coverage and task roles. Deterministic validators check structure, exact references, evidence lineage and ontology constraints; they cannot certify that a plausible interpretation is correct. Observation/document VLM tools remain evidence producers. A closer size match does not prove a unique task role. There is no mandatory second semantic reviewer or reviewer bypass flag.

## Arm assignment and limits

Resource assignment is part of the ontology. A supported PA choice adds specification `hasProcessExecution`, the `processExecution` type, `runsProcess`, and `runsOnResource`. All grounded coordinate-bearing state values are checked, with exact duplicates checked once per state. Multiple relationships do not require choosing one Cartesian pair or deferring allocation.

PA may choose either eligible arm when evidence gives no advantage. The host does not select by order, optimize distance, or substitute a resource. Missing locations or an unsupported/unreachable choice leave Phase 4 incomplete, while the grounded ontology and images remain visible.

Current reachability uses approved calibration and MoveIt `GetMotionPlan` for each exact checked simulated robot, without starting or contacting a RobotAgent. The request uses its configured planning group/TCP, live robot state, joint limits and collision scene, with a configured 5 mm position tolerance and unconstrained tool orientation. Example boxes and fixed-radius estimates cannot accept or reject allocation. Missing services, incomplete results or failed planning block assignment; a failed search does not prove mathematical unreachability. `motion_validation_performed` is true for accepted position plans, while `motion_executed` remains false. Grasping, attached-part collision geometry, orientation, insertion, force and tolerance remain unvalidated.

## Records and UI

Each record type has one current format, without version markers or compatibility readers. [Schemas](schemas/README.md) and [artifact paths](contexts/README.md) describe the lineage. The accepted proposal pins its evidence snapshot, source hashes and typed artifacts in `grounding_evidence`.

The UI shows evidence-operation usage, proposal attempts, current activity, elapsed time, validation feedback and stop reasons. It also shows ontology tables/diagram, both states, relationships, source caveats, current images, desired reference images, the selected arm, and validation limits. Its grounding timeline ends with “Product grounding and arm assignment complete”. Operator labels describe RobotAgent context capture and structural primitive drafting; implementation phase numbers remain in development documentation and internal identifiers. Destination images are labeled observed references, not completed manufacturing. Source uncertainty survives persistence and reload.

Recovery revalidates the current proposal/evidence contract, source hashes, capability authority, exact location coverage, and the exact MoveIt requests, position plans and controller profile. Old reviewer-bearing or otherwise incompatible records require “Start a fresh interaction”. Existing records remain unchanged and cannot authorize new RA work; they are not upgraded or read through a compatibility path.

## Phase 5 boundary

Phase 5.1 activates/reuses only the exact selected RobotAgent through the owned adapter and captures paired state/catalog snapshots. Phase 5.2A asks that RA for one unbound `PrimitiveProgramDraft` using exact catalog symbols. Activation, recovered context, and draft entrypoints require current completion and evidence lineage.

Live SPADE delivery, context/binding exchange, executable validation, execution, and observed outcomes remain future work. Shared agents and `SystemBridge` interfaces remain unchanged.

## Evidence boundary and bias checks

Recognition receives the exact requirement/clarifications, approved document content and CAD files, RGB, depth, and approved calibration. Real camera names, frames, paths, hashes, and canonical candidate pointers stay internal. Model-facing view/candidate handles and order are randomized per interaction and pinned for audit. Approved CAD filenames and document part names remain legitimate supplied evidence.

Gazebo identities, world/SDF contents, spawn poses, entity-state services, detector answers, evaluator labels, expected answers, and resource-selection results are excluded from recognition. Evaluation uses a separate evaluator after prediction finalization.

[BIAS_VALIDATION.md](BIAS_VALIDATION.md) explains inspection of actual model inputs and controlled experiments. Offline tests establish contracts, not model accuracy, generalization, or absence of bias.

## Reading and running

- [Implementation status](IMPLEMENTATION_PLAN.md), [research scope](ICRA_SCOPE.md), [research positioning](RESEARCH_POSITIONING.md)
- [PA](agents/pa/README.md), [RA](agents/ra/README.md), [tools](tools/README.md), [tests](tests/README.md)
- [Approved references](references/README.md)

The entrypoint remains `poetry run python -m cais_spade_llm.ui_main`; the page is `/spec2primitives`. Configuration remains in `config/model_runtime.json`, `config/workcell_profile.json`, and the approved calibration manifest. Runtime records belong under `contexts/`; evaluator-only data belongs under `evaluations/`.

Each owned record type uses one current format, with no format-version fields or compatibility dispatch. Start a fresh interaction after this format change; saved interactions are neither converted nor rewritten. Hashes, provenance, exact references, timestamps, operation identifiers and calibration values remain authoritative.

Before PA selects an arm, it must check every resource in the capable-resource catalog against the same grounded current and desired positions. One bounded correction lists missing checks and reuses completed results. PA may choose either passing arm; there is no preferred resource or proximity objective. Completion reload verifies coverage from the pinned tool-call records.

State cards show the exact grounded statement once in a four-line preview, with “Show more” when it exceeds the available space. Visible caveat counts lead to expandable caveats, sources, endpoints and raw records. Desired State displays only directly bound images; cross-state relationship images remain under evidence details and identify their source state. The resource comparison distinguishes reachable, rejected, unavailable and unchecked results. Responsive grids and wrapping contain long paths and expanded records.

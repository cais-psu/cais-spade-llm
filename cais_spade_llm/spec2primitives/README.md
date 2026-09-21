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

PA checks every capable arm and prefers the accepted arm with the smaller mean base-to-current-location distance. The host reads live TF using `config/resource_base_frames.json`, computes Euclidean distances to the existing grounded locations in the same frame, and supplies advisory evidence; PA makes the choice. Choosing a farther arm requires supporting task evidence. Equal distances or unavailable TF give no distance preference, and proximity never overrides a rejected or unavailable reachability check. Missing locations or an unsupported/unreachable choice leave Phase 4 incomplete, while the grounded ontology and images remain visible.

Current reachability uses approved calibration and MoveIt `GetMotionPlan` for each exact checked simulated robot, without starting or contacting a RobotAgent. The request uses its configured planning group/TCP, live robot state, joint limits and collision scene, with a configured 5 mm position tolerance and unconstrained tool orientation. Example boxes and fixed-radius estimates cannot accept or reject allocation. Missing services, incomplete results or failed planning block assignment; a failed search does not prove mathematical unreachability. `motion_validation_performed` is true for accepted position plans, while `motion_executed` remains false. Grasping, attached-part collision geometry, orientation, insertion, force and tolerance remain unvalidated.

## Records and UI

The project tabs are **run → setup → results**, with **run** open by default.
The existing Gazebo, PA, and RA controls are constructed once per page client;
opening the page does not start simulation or execution. **setup** shows
configuration, saved requirements, resource/catalog evidence, and the planned paper protocols.
**results** lists saved interactions, including incomplete and failed attempts,
with searchable recorded values, artifact inspection, and CSV export. Inspection
does not resume work or change the active configuration. Missing measurements are
shown as `not recorded`; validation, simulation execution, and physical outcomes
remain separate. Revisions are not independent trials. Batch experiments and
aggregate paper comparisons are deferred.

Each record type has one current format, without version markers or compatibility readers. [Schemas](schemas/README.md) and [artifact paths](contexts/README.md) describe the lineage. The accepted proposal pins its evidence snapshot, source hashes and typed artifacts in `grounding_evidence`.

The UI shows evidence-operation usage, proposal attempts, current activity, elapsed time, validation feedback and stop reasons. It also shows ontology tables/diagram, both states, relationships, source caveats, current images, desired reference images, the selected arm, and validation limits. Its grounding timeline ends with “Product grounding and arm assignment complete”. Operator labels describe RobotAgent context capture and primitive composition; implementation phase numbers remain in development documentation and internal identifiers. Destination images are labeled observed references, not completed manufacturing. Source uncertainty survives persistence and reload.

Recovery revalidates the current proposal/evidence contract, source hashes, capability authority, exact location coverage, and the exact MoveIt requests, position plans and controller profile. Old reviewer-bearing or otherwise incompatible records require “Start a fresh interaction”. Existing records remain unchanged and cannot authorize new RA work; they are not upgraded or read through a compatibility path.

For an explicitly authorized reassignment of a current valid interaction, `reassign_completed_interaction` archives the complete original outside its retrieval root, restores only the pre-assignment ontology in a staged copy, and reuses allocation with fresh checks for every arm. An optional `requested_resource_symbol` is pinned in the new turn and enforced against PA's accepted selection. The replacement becomes active only after completion validation; failures retain the original. Phase 4 recognition and calibrated location bytes stay unchanged. Old RA context/program artifacts remain in the archive, and fresh selected-RA context capture is required. `activate=False` supports a separate allocation trial without replacing the active run.

## Phase 5 boundary

Phase 4 supplies observed product information and desired assembly relationships. Phase 5 uses that evidence to compose primitives and, the bounded refinement process, obtain robot targets. PA can investigate missing product evidence; the selected RA adapter measures robot context. Calculations and modeled validation follow RA-selected bindings, without robot motion.

Phase 5.1 activates/reuses only the exact selected RobotAgent through the owned adapter and captures paired state/catalog snapshots. Activation, recovered context and composition require current completion and evidence lineage.

**Compose Primitive Program** starts directly from Phase 4 grounding and the captured RA context. The exact selected RA can read pinned records, query accepted ontology assertions, and author `primitive_steps`, choosing order, intermediate movements and available parameters together. There is no separate draft action or preliminary sequence call. Each attempt pins completion and context snapshots directly and records its prompts, reads, raw response and candidate under `composition/primitive_program_candidates/`. Historical drafts remain untouched and are excluded from new inputs. Full parameter/result declarations are required; older snapshots that lack them must be recaptured.

The UI displays one compact numbered program, with omitted required parameters shown as `<unbound>` and raw records/traces expandable. Blocking evidence checks run outside the UI event loop. JSON structure, catalog membership and supplied parameter names, types and references are checked; missing geometry permits a visible proposal with omitted parameters. A proposed candidate is not certified feasible or correct. A dependency report identifies missing evidence, blocked calculations and deferred outputs. The bounded flow obtains supplemental evidence, lets RA bind it, calculates selected helpers and validates ordered Cartesian motion in a private MoveIt scene. Findings return to RA for revision. `validated_for_declared_scope` means the recorded rigid vertical geometry/motion model passed; force/contact validity, execution and observed outcomes remain subsequent work. Shared agents and `SystemBridge` interfaces remain unchanged.

## Evidence boundary and bias checks

Recognition receives the exact requirement/clarifications, approved document content and CAD files, RGB, depth, and approved calibration. Real camera names, frames, paths, hashes, and canonical candidate pointers stay internal. Model-facing view/candidate handles and order are randomized per interaction and pinned for audit. Approved CAD filenames and document part names remain legitimate supplied evidence.

Gazebo identities, world/SDF contents, spawn poses, entity-state services, detector answers, evaluator labels, expected answers, and resource-selection results are excluded from recognition. Evaluation uses a separate evaluator after prediction finalization.

[BIAS_VALIDATION.md](BIAS_VALIDATION.md) explains inspection of actual model inputs and controlled experiments. Offline tests establish contracts, not model accuracy, generalization, or absence of bias.

## Reading and running

- [Implementation status](IMPLEMENTATION_PLAN.md), [research scope](ICRA_SCOPE.md), [research positioning](RESEARCH_POSITIONING.md)
- [PA](agents/pa/README.md), [RA](agents/ra/README.md), [tools](tools/README.md), [tests](tests/README.md)
- [Approved references](references/README.md)

The entrypoint remains `poetry run python -m cais_spade_llm.ui_main`; the page is `/spec2primitives`. Configuration remains in `config/model_runtime.json`, `config/workcell_profile.json`, and the approved calibration manifest. Runtime records belong under `contexts/`; evaluator-only data belongs under `evaluations/`.

Each owned record type uses one current format, with no format-version fields or compatibility dispatch. Incompatible saved records require a fresh interaction. The explicitly authorized reassignment path above operates only on valid current records and retains the original archive. Hashes, provenance, exact references, timestamps, operation identifiers and calibration values remain authoritative.

Before PA selects an arm, it must check every resource in the capable-resource catalog against the same grounded current and desired positions. One bounded correction lists missing checks and reuses completed results. Completion reload verifies coverage and recomputes advisory distances from pinned tool results. Reachability records retain their existing contract. Base distance is neither trajectory cost nor evidence of collision-free grasping or lifting; collision-scene synchronization remains outside this change.

State cards show the exact grounded statement once in a four-line preview, with “Show more” when it exceeds the available space. Visible caveat counts lead to expandable caveats, sources, endpoints and raw records. Desired State displays only directly bound images; cross-state relationship images remain under evidence details and identify their source state. The resource comparison distinguishes reachable, rejected, unavailable and unchecked results. Responsive grids and wrapping contain long paths and expanded records.

Fresh context captures retain the complete runtime contracts, nested geometry schemas, and configured `motion_context` (`frame_id`, `ee_link`, `tcp_link`). The model-facing grasp/release conditions and effects contain only `held_part`. Recovery examples and decomposition metadata are excluded from both initial state and subsequent record reads. The binding report checks only RA-selected references; it does not supply values or rewrite steps.

New composition catalogs omit `model_name` from parameters, nested helper inputs, outputs and custody effects. RA uses `part_name` and grounded product evidence; the catalog is a composition interface, not a directly executable Python signature. New submissions containing simulator identifier arguments are rejected without rewriting them. Missing geometry remains visible. The future execution adapter must establish the correct physical instance's Gazebo identifier; it cannot guess from a product label.

Saved attempts are validated and displayed against the composition catalog embedded in their own hash-checked request. Older programs may therefore still show `model_name`. Subsequent attempts use the current projection without receiving earlier programs as input. No saved record or program is automatically regenerated.

`move_cartesian(x, y, z, ...)` addresses the controlled end-effector in the configured planning frame. Matching frames do not turn an observed object point into a picking position. A raw CAD record is not placement geometry, a descriptive destination is not a resolved target, and `part_name` does not establish the execution identifier `model_name`. Helper outputs, including `approach_pose` and conditional insertion poses, describe available values without prescribing their use or order. `target_pose` can be pre-insertion rather than completed insertion.

Restart the running application before capturing corrected context if it still has the older catalog cached. Use the existing context-capture action to append fresh snapshots; saved records and programs are never automatically regenerated. See [bounded validation/revision](VALIDATION_AND_REVISION.md) and [composition experiments](COMPOSITION_EVALUATION.md).

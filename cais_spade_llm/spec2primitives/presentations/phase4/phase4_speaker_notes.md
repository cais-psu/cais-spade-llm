# How Phase 4 works — 10-minute technical presentation

Prepared from the current local implementation and one saved Phase 4 interaction on 2026-09-09. The saved trace is a worked example, not an independently scored recognition or assembly experiment.

- [Editable PowerPoint, including presenter notes](phase4_technical_overview.pptx)
- [Presentation PDF](phase4_technical_overview.pdf)

Open the PowerPoint in Presenter View to see the narration. The nine main slides total exactly 10 minutes at the suggested pacing. Diagram shapes and text are editable. The observation images are existing saved evidence, embedded without changing their source files.

## Timing

| Slide | Topic | Time | Cumulative |
| --- | --- | --- | --- |
| 1 | Phase 4: ground the product, then assign an arm | 45 s | 0:45 |
| 2 | Phase 4 builds a grounded product context | 60 s | 1:45 |
| 3 | PA decides what evidence to request next | 75 s | 3:00 |
| 4 | Documents, CAD and RGB-D contribute different facts | 75 s | 4:15 |
| 5 | The host checks contracts; PA owns meaning | 60 s | 5:15 |
| 6 | Saved example: “assemble medium gear.” | 75 s | 6:30 |
| 7 | One desired relationship links two grounded features | 60 s | 7:30 |
| 8 | PA checks every capable arm before selecting | 90 s | 9:00 |
| 9 | Completion hands grounded evidence to Phase 5 | 60 s | 10:00 |

## Slide 1 — Phase 4: ground the product, then assign an arm

**Time:** 45 seconds.

**Say:**

Phase 4 turns an assembly requirement into a grounded product goal and an arm assignment. PA means ProductAgent. It investigates what the user wants, which observed objects are relevant, how the product is currently configured, and what relationship should hold after assembly. The output links those claims to their evidence and records which arm passed the required position checks. I will explain the implemented flow using the saved requirement, “assemble medium gear.” In that run, PA selected xarm6. That selection is backed by modeled position planning. Primitive order, robot targets and later execution belong to subsequent stages. The central question for these ten minutes is how the system connects a short requirement to evidence that a downstream RobotAgent can use.

**Source checks:**

- [Implemented phase boundaries](../../IMPLEMENTATION_PLAN.md) — line 5
- [PA orchestration](../../agents/pa/production_grounding.py) — line 1481
- [Saved completion](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/interaction_record/context_completion_0001.json)

## Slide 2 — Phase 4 builds a grounded product context

**Time:** 60 seconds.

**Say:**

The implementation divides Phase 4 into five boundaries. Phase 4.0 supplies the PPR TBox and initializes an interaction ABox. The TBox is the supplied vocabulary and its constraints; the ABox contains facts for this interaction. PA creates instances under that vocabulary rather than discovering a new schema. Phase 4.1 makes approved document pages available and supports questions authored by PA. Phase 4.2 provides CAD and RGB-D evidence: segmentation, morphology, measurements, layout and calibrated locations. Phase 4.3 defines typed evidence, provenance and validation contracts across the investigation. Phase 4.4 combines a complete target_feature proposal, deterministic checks, pairwise assembly relationships and required arm assignment. These labels describe implementation boundaries, rather than a fixed sequence of model tool calls. Phase 5 begins from the completed grounding and captures context from the exact selected RA before composing primitive_steps.

**Source checks:**

- [Implemented phase table](../../IMPLEMENTATION_PLAN.md) — line 5
- [Interaction entry point](../../agents/pa/context_interaction.py) — line 64
- [Selected-RA handoff](../../agents/ra/context_handoff.py) — line 304

## Slide 3 — PA decides what evidence to request next

**Time:** 75 seconds.

**Say:**

The production entry point creates an evidence investigation and presents approved source handles to PA. PA chooses among four tools. retrieve obtains a document index, CAD measurements or an observation bundle. query_document asks a task-specific question about approved document pages. compare_cad_size reports candidate measurements against a selected CAD file. analyze_candidate_layout measures relationships among selected candidates in the same view. PA can use several operations before submitting one complete target_feature. The configured investigation limits are twenty-four evidence operations and six proposals. These are ceilings, not a required number of calls. Correctable unissued references or missing state locations can trigger feedback and another proposal while retaining the pinned observation. Some invalid outputs stop immediately; unchanged failures stop as no progress. Genuine requirement ambiguity can request user clarification. In the saved example PA used nine evidence operations and one proposal. The host does not prescribe that evidence order.

**Source checks:**

- [Investigation and bounded correction](../../agents/pa/production_grounding.py) — line 1481
- [Configured budgets](../../config/model_runtime.json) — line 2
- [Saved tool usage](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/interaction_record/turn_0001.json)

## Slide 4 — Documents, CAD and RGB-D contribute different facts

**Time:** 75 seconds.

**Say:**

Each evidence stream supplies a different part of the grounding problem. Document retrieval creates a source index with page text, rendered images and provenance. A document VLM answers PA's question with citations and uncertainty. CAD preprocessing measures the approved mesh, with units converted for geometry calculations. The active size comparison returns measurements for every evaluated candidate; it does not select a winner. RGB-D preprocessing deprojects valid depth, and neutral segmentation yields candidate bounds and centroids. An observation VLM describes candidate morphology and uncertainty without assigning task roles. The layout tool reports same-view positions, displacements, distances and collinearity. Approved calibration later maps selected observations into the frame used for arm checks. PA combines these facts to interpret identity and task role. Document relationships alone do not establish current installation. Source and candidate presentation handles are randomized and retained for audit; simulator object identities and evaluator answers are excluded from recognition.

**Source checks:**

- [Active evidence production](../../agents/pa/production_grounding.py) — line 1709
- [Document tools](../../tools/document_evidence/README.md) — line 3
- [Perception tools](../../tools/rgb_d_cad_grounding/README.md) — line 3
- [Recognition evidence boundary](../../AGENTS.md) — line 156

## Slide 5 — The host checks contracts; PA owns meaning

**Time:** 60 seconds.

**Say:**

There are two responsibilities here. PA interprets the evidence: which candidate is the required part, which observed reference has the destination role, what the current and desired states mean, and whether the requirement is covered. The host validates the proposal's structure, allowed ontology terms, exact references, typed values and evidence hashes. It also requires coordinate-bearing state values in both states for the planning consumer. A rejected target proposal never contributes its proposed assertions. Supported correction preserves the observation and returns generic feedback, without giving PA the task answer. Source uncertainty is derived from typed evidence and retained with accepted claims. The current production path has no second independent semantic reviewer. Consequently, a deterministic pass establishes the checked contracts and evidence lineage. Semantic correctness and model performance still require separate evaluation. A later arm-assignment failure can leave the grounded product visible while preventing completion.

**Source checks:**

- [Proposal validation](../../agents/pa/ontology_grounding.py) — line 305
- [Evidence-preserving commit](../../agents/pa/ontology_grounding.py) — line 433
- [Correction and incomplete outcomes](../../agents/pa/production_grounding.py) — line 1570
- [Current reviewer boundary](../../IMPLEMENTATION_PLAN.md) — line 43

## Slide 6 — Saved example: “assemble medium gear.”

**Time:** 75 seconds.

**Say:**

This example comes from a saved interaction, rather than a new experiment conducted for this presentation. On the left is the saved candidate crop that PA bound as the medium gear. The center image is the observed assembly workspace. PA's saved current-state statement says the medium gear is separate from its intended middle gear shaft on the gear plate. The desired state describes the required mounted relationship. Its coordinate-bearing value is named “observed middle gear shaft destination reference.” The saved statement explicitly distinguishes that observed shaft reference from a measured final gear pose. PA retrieved document, observation and CAD evidence; compared candidate sizes; asked which shaft should receive the medium gear; and requested same-view layout evidence. That trace contains nine evidence operations and one target proposal. These images show the observations associated with the saved interpretation. They do not show a completed assembly. The trace is useful for explaining behavior, while an independent evaluator would still need to assess whether the interpretation is correct.

**Source checks:**

- [Saved target_feature](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology_grounding/proposal_0001.json)
- [Saved investigation counts](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/interaction_record/turn_0001.json)
- [PA document question](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/interaction_record/tool_call_0008.json)
- [PA layout request](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/interaction_record/tool_call_0009.json)
- [Saved gear crop](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/rgb_d_cad_grounding/observation_review_0001/view_0003_candidate_0003_0002.png)
- [Saved workspace RGB](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/observations/observation_0002/cam_assembly_rgb.png)

## Slide 7 — One desired relationship links two grounded features

**Time:** 60 seconds.

**Say:**

The ontology preserves a pairwise representation. In this saved proposal, the owning Assembly is named “gear plate assembly.” It has an AssemblyFeatureAssociation relating exactly two AssemblyFeature endpoints: “medium gear central bore” and “middle gear shaft mounting surface.” The association's state_names contains desired_state, which says this is the intended relationship. Each endpoint has its own observation binding. In this saved run the gear endpoint binds to a current_state value, and the shaft endpoint binds to the desired_state destination reference. Relationship membership and endpoint bindings are separate concepts; the general contract also permits both endpoint observations to come from current_state. Multiple relationships create separate pairwise association individuals. The host preserves exact names and reuses owners only by exact identity. These assertions describe the product and the required relationship. They do not prescribe approach, grasp, lift or insertion order; RA determines primitive order later.

**Source checks:**

- [Fixed terms and state membership](../../ASSEMBLY_ONTOLOGY.md) — line 5
- [Saved pairwise association](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology_grounding/proposal_0001.json)
- [Pairwise contract regression](../../tests/test_pa_completion.py) — line 1179

## Slide 8 — PA checks every capable arm before selecting

**Time:** 90 seconds.

**Say:**

Once the target is committed, allocation gathers every grounded coordinate-bearing value in current_state and desired_state. PA must check every configured capable arm against those same locations. This saved example has two current references and one desired reference; the desired shaft is also a current reference, so the sets contain a repeated physical location. The owned MoveIt adapter calls GetMotionPlan using the configured planning group and end-effector link, live robot state and the available collision scene. The position tolerance is five millimeters and tool orientation is unconstrained. No RobotAgent is activated and no motion is executed during these checks. Both xarm6 and ur5e returned accepted checks. Their recorded mean base-to-current-location distances were 0.549 and 0.627 meters, respectively. PA selected xarm6. Distance is advisory straight-line evidence, not path length or execution time. A farther accepted arm needs supporting task evidence; equal or unavailable distances supply no preference. Rejected or unavailable planning cannot be overridden by proximity. Omitted arm checks get one correction attempt. A planning failure means no plan was validated by that search, rather than a proof of mathematical unreachability.

**Source checks:**

- [Allocation orchestration](../../agents/pa/production_grounding.py) — line 1790
- [PA selection policy](../../agents/pa/production_grounding.py) — line 2394
- [MoveIt request](../../adapters/moveit_plan_only.py) — line 146
- [Proximity calculation](../../agents/pa/resource_proximity.py) — line 75
- [xarm6 check and distance](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/interaction_record/allocation_tool_call_0001.json)
- [ur5e check and distance](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/interaction_record/allocation_tool_call_0002.json)
- [Saved arm selection](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/resource_selection/selection_0001/resource_selection_record.json)

## Slide 9 — Completion hands grounded evidence to Phase 5

**Time:** 60 seconds.

**Say:**

A successful Phase 4 result packages the accepted target proposal, source and typed evidence, selected arm, all-arm check lineage, assignment delta and final product context. The completion writer pins references and hashes so recovery can check that the evidence chain still matches. An incomplete assignment cannot authorize the next stage. In the saved completion, validation_scope is moveit_state_location_reachability, motion_validation_performed is true, and motion_executed is false. The next explicit boundary activates or reuses only the exact selected RA and captures its context. RA then authors primitive_steps and obtains the targets needed by those operations within Phase 5's bounded process. The defensible claim for Phase 4 is an evidence-linked product context with an arm assignment supported by modeled position planning. Grasping, attached-part motion, insertion, force and manufacturing tolerances are outside those position checks. Execution and independent outcome observation are needed before claiming assembly success.

**Source checks:**

- [Completion writer](../../agents/pa/grounding_contracts.py) — line 1705
- [Completion recovery](../../agents/pa/grounding_contracts.py) — line 2076
- [Saved completion fields](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/interaction_record/context_completion_0001.json)
- [Next boundary: selected RA](../../agents/ra/context_handoff.py) — line 304

## Questions to prepare for

**Does Phase 4 generate an ontology?** It grounds interaction-specific instances and assertions under the supplied PPR TBox. It does not discover or modify the ontology schema.

**Who chooses the part, destination and arm?** PA interprets the part and task roles, proposes the target and chooses the arm. Evidence tools supply measurements and interpretations; the host validates the contracts and permitted choice. The size-comparison tool supplies all evaluated measurements without selecting a winner.

**Is the process a fixed pipeline?** The software has staged boundaries, but PA chooses the evidence tools, source order and questions within the configured budget. The 9-operation example is one trace.

**Why can the desired state use a current observation?** An observed shaft can anchor an intended gear–shaft relationship. Association `state_names` is independent of endpoint observation bindings. An observed destination reference does not by itself establish a final part pose or tool pose.

**Is there an independent semantic reviewer?** No second mandatory semantic reviewer is present in the current Phase 4 runtime. Document VLM answers and observation morphology review remain evidence producers. Some older tool documentation still describes a retired independent target review; the current production call path and implementation status govern this presentation.

**What happens when evidence or planning is missing?** Exact-reference and missing-location corrections are bounded. Malformed proposals, changed evidence, budget exhaustion or repeated failures can stop grounding. Assignment requires accepted checks covering all grounded locations. A rejected plan and an unavailable planning service remain distinct. Grounded product evidence can remain visible while completion is blocked.

**Why xarm6 in this trace?** Both arms returned accepted checks. The recorded mean base-to-current-location distances were 0.5486871357469699 m for `xarm6` and 0.627242081056697 m for `ur5e`; PA selected `xarm6`. This is a distance preference, not a proof of optimal path length, execution time or assembly capability. The current_state set includes both the gear and the observed shaft. Distances are rounded to three decimals on the slide.

**What is actually validated by MoveIt?** Position-goal plans using the configured planning group and end-effector link, live state and available collision scene. Orientation is unconstrained. These independent position checks are not a validated ordered carrying or insertion program; they establish no grasp, attached-part collision, force, tolerance or physical assembly outcome.

**Does Phase 4 activate RobotAgent?** The required MoveIt checks run through the owned planning adapter without starting or contacting RobotAgent. Selected-RA context capture is the next explicit boundary.

**What does this saved run prove?** It documents what the system recorded for one input. The presentation checked the cited record/image hashes and reported fields, without rerunning recognition, planning, ROS2 or hardware and without independently evaluating task accuracy. Offline regression tests establish software contracts, not accuracy, generalization or absence of bias.

## Outcome

Created the [nine-slide PowerPoint](phase4_technical_overview.pptx), [PDF](phase4_technical_overview.pdf) and [this timed narration](phase4_speaker_notes.md). This is documentation/presentation work with no changed runtime entry point.

## Process flow

`start_pa_context_interaction → ground_product_context → PA evidence investigation → proposal validation → ontology commit → all-arm reachability and PA selection → assignment commit → context completion`

## Read these locations in order

1. [start_pa_context_interaction](../../agents/pa/context_interaction.py) — line 64: takes the exact requirement, initializes the ABox and invokes grounding; writes completion only after the complete result.
2. [ground_product_context](../../agents/pa/production_grounding.py) — line 1481: takes PA, the pinned ontology and interaction context; orchestrates evidence tools, bounded proposals, commit and allocation; returns complete, clarification or incomplete status.
3. [validate_ontology_grounding_attempt and commit_ontology_grounding_candidate](../../agents/pa/ontology_grounding.py) — lines 305 and 433: validate a proposal and its evidence before merging accepted assertions into the ABox.
4. [_complete_resource_assignment](../../agents/pa/production_grounding.py) — line 1790: derives the proposal-bound locations, requires all capable-arm checks, validates PA’s selection and commits the assignment. [MoveIt position request](../../adapters/moveit_plan_only.py) — line 146: constructs a position-only goal from an exact grounded location.
5. [persist_pa_context_grounding_completion](../../agents/pa/grounding_contracts.py) — line 1705: validates and pins the complete evidence and assignment chain; [activate_selected_ra_context](../../agents/ra/context_handoff.py) — line 304: shows the next boundary, which consumes accepted completion.

## Read this test

[test_current_completion_uses_reviewed_destinations_and_reachability](../../tests/test_pa_completion.py) — line 1179: a parametrized fixture covers `xarm6` and `ur5e`, two current values, one desired value, desired relationship membership and current-state endpoint bindings. It asserts the `moveit_state_location_reachability` scope.

[test_location_goal_uses_live_state_and_position_only_constraints](../../tests/test_moveit_plan_only.py) — line 50: checks the live-state request, configured link, 5 mm position region and absence of orientation constraints using mocked ROS message types. These tests were read for this presentation; they were not rerun.

## Runtime evidence

The saved interaction is [interaction_116f33c8cce64d33b0066891bf228572](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/). Read its [target proposal](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/ontology_grounding/proposal_0001.json), [xarm6 check](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/interaction_record/allocation_tool_call_0001.json), [ur5e check](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/interaction_record/allocation_tool_call_0002.json), [selection](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/products/grounding/resource_selection/selection_0001/resource_selection_record.json), and [completion](../../contexts/interaction_116f33c8cce64d33b0066891bf228572/interaction_record/context_completion_0001.json).

The cited completion and allocation record hashes matched the referenced files. The source RGB hashes matched the segmentation record. No evaluator labels were read.

## You can ignore

- Shared `SystemBridge`, ProductAgent and RobotAgent internals.
- Phase 5 primitive composition, refinement and execution implementation.
- Ground-truth evaluator files and unrelated saved runs.

## Refactoring performed

None. Only the three requested presentation artifacts were added.

## Verification and intentionally unchanged behavior

Presentation checks: generated and reopened all 9 PowerPoint slides with presenter notes; generated 9 PDF pages; checked text layout, local source links, timings and cited source hashes; visually inspected slide renders; ran `git diff --check`. The PDF is rendered from the same layout specification as the PowerPoint. The PowerPoint was structurally reopened, but was not rendered by Microsoft PowerPoint or LibreOffice in this environment.

No runtime tests, full Python checks, live model calls, ROS2/MoveIt runs or hardware validation were performed for this presentation-only change. Existing saved interactions and runtime code were not edited.

Pre-existing local changes were preserved in `VALIDATION_AND_REVISION.md`, `agents/pa/primitive_context.py`, `agents/ra/primitive_composition.py`, `agents/ra/refinement.py`, `spec2primitives_ui.py`, `tests/test_pa_ui_connection.py`, `tests/test_primitive_refinement.py`, `tools/assembly_geometry.py`, `tools/rgb_d_cad_grounding/size_correspondence.py`, and the untracked `agents/pa/primitive_input_resolution.py`.

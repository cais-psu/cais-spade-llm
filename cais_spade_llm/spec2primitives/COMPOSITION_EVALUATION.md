# Composition evaluation

This document specifies experiments to run. Current offline tests establish software contracts and proposal preservation; they do not establish reliable LLM composition, physical correctness, or a publishable result. Use [VALIDATION_AND_REVISION.md](VALIDATION_AND_REVISION.md) for the implemented bounded proposal-validation flow and subsequent execution work and [RESEARCH_POSITIONING.md](RESEARCH_POSITIONING.md) for the broader literature discussion.

## Hypothesis and unit of evaluation

Test whether the selected RA can infer useful intermediate-step dependencies that are absent from the supplied formal contracts, using the task evidence, current robot state, and truthful primitive descriptions. In the composition view, grasp/release expose only `held_part` conditions/effects. No ontology rule or supplied program prescribes approach, descent, grasp, lift, release, or retreat.

The claim is about composition under a partial formal model. It is not that symbolic planning cannot solve a fully specified assembly problem. Natural-language primitive semantics, model pretraining, helper outputs, controller policies, and supplied evidence are prior information; measure and report their influence.

Evaluate a task-and-state instance, not just a changed coordinate. Predefine what outcome is required and which conditions create a need for different operations, order, repetition, or output dependencies. Allow multiple valid programs; do not grade only against one reference sequence.

## Case matrix and held-out conditions

Separate a development set used for prompt/interface debugging from held-out evaluation conditions. Freeze the split before inspecting test results. Include multiple objects, configurations, and initial states supported by the actual primitive/controller library.

| Condition | What it tests |
| --- | --- |
| Same assembly, translated objects | Coordinate adaptation control; unchanged sequence alone is not structural generalization. |
| Part already held versus loose | Whether custody changes the selected operations and data dependencies. |
| Clearance or obstruction changed | Whether intermediate motion structure adapts when direct transport is insufficient. Use only a scene representation and controller able to assess that distinction. |
| Partially seated versus not inserted | Whether the proposed operations address the remaining assembly relationship. Pre-insertion arrival is not final seating. |
| Orientation or mating constraint changed | Whether the program uses an available orientation/insertion capability when needed, or reports unsupported capability. |
| Different selected robot or catalog | Whether composition respects the exact resource interfaces and their limits. |

Do not call a hex nut assembled merely because it is placed over a shaft. Admit a threaded case only when the controller offers the required rotation/contact operation and the evaluator can establish engagement and the required final condition. Otherwise score an honest unsupported result. Do not conceal a complete task sequence inside a newly named primitive.

Hold out combinations and, where supported, object families. Document what was unseen in development; model pretraining cannot generally be audited, so avoid claiming the model has never encountered the concept.

## Baselines and ablations

Use identical task evidence, robot state, primitive interfaces, available retrieval operations, and resource budgets for comparable composers. Record any extra assumptions explicitly.

Use the same composition projection for every method: simulator `model_name` is excluded from new program inputs and outputs, while full backend signatures remain archived. A future common execution adapter supplies verified instance bindings. Report unavailable execution bindings separately from sequence correctness; generating or guessing simulator names is not evidence of better composition.

- A fixed or parameterized template establishes a coverage boundary; let it abstain outside its admitted cases. Its failure alone does not prove LLM superiority.
- A single-pass LLM composer and the current bounded read-only composer isolate the benefit of evidence/ontology retrieval.
- A symbolic/interface-based composer receives the same represented conditions and outputs. Report what is unmodeled and its coverage rather than interpreting missing modeling as planner failure.
- An independently engineered richer symbolic model or task-and-motion baseline measures what additional modeling achieves. Report its extra predicates, geometry services, engineering effort and supervision separately.
- Compare first-pass composition with validation-feedback revision using equal proposal and planning budgets. The validator must not generate repair steps for any method.

Run retrieval, ontology-projection, current-state/freshness, validation-feedback, and description/output-declaration ablations. For output ablations, distinguish waypoint availability from sequence hints: helper-provided `approach_pose` and insertion poses contain domain knowledge even when RA selects their order. Keep actual callable behavior truthful and document which information each condition removes. Do not give one method privileged geometry, evaluator labels, or a larger hidden retry budget.

Include shuffled candidate presentation, distractors, ambiguous observations, and missing-input cases. Recovery examples and decomposition metadata must stay excluded from every nominal composition input and record read. Any experiment that deliberately adds demonstrations is a separately labeled condition with its examples and selection procedure disclosed.

## Independent evaluation and leakage controls

Before generation, define outcome tolerances, admissible operations, collision/contact constraints, and rejection/unsupported criteria. Assess program structure using independent reviewers or evaluators blind to the method where practical. Record disagreement and adjudication. The generating LLM's explanation is not an independent correctness label.

Capture every initial prompt, structured schema, tool request/result, accepted ontology projection, and follow-up prompt. Audit all model-input paths, including root record reads and nested references. Hash the original snapshots and the projected input. Verify that the full authoritative recovery metadata cannot be retrieved through the composition reader.

Recognition may use the approved requirement, documents, CAD, RGB-D and calibration. Simulator identity, world/SDF contents, configured object poses, detector answers, expected sequences and evaluator labels must not enter recognition. A future controller identifier binding occurs after recognition and remains an execution-only authority. Ground truth is accessed only by an isolated evaluator after the prediction is finalized.

Freeze evaluator access and splits before running trials. Never copy a successful test program into the composition prompt, ontology, retrieval corpus, or helper metadata. Do not rerun failed test cases until a desirable sample appears and report only that sample.

## Measurements and reproducibility

For every trial, retain first-pass output, invalid responses, binding gaps, unsupported results, validation findings, every revision, stopping reason and final outcome. Use all attempted trials as the denominator, with unsupported coverage reported separately.

Measure independently:

- Schema/reference validity and exact preservation of the submitted program.
- Intermediate-dependency adequacy and structural adaptation, separately from coordinate changes.
- Complete grounded bindings, including frame/TCP correctness; report future execution-identifier provenance separately.
- Modeled-condition validity, motion feasibility, and independently observed assembly success.
- First-pass versus post-revision success, number of reads/revisions/planning calls, latency by stage, and model usage/cost when available.

Use paired scene/state instances across methods and repeated runs. Set sample sizes and stopping rules prospectively. Report uncertainty intervals and variation across scenes; avoid treating many repeated samples from one scene as independent evidence of broad generalization. Include failure categories such as missing lift/clearance, premature release, wrong output dependency, wrong frame, unresolved identity, insufficient geometry, and unsupported assembly semantics without exposing these as per-case answer hints.

Archive commit and local-diff identifiers, prompt/schema versions or hashes, model identifier/settings, controller/catalog configuration hashes, corpus/split hashes, evidence and context refs, randomization seeds, calibration, scene initialization, hardware/software environment and evaluator configuration. Publish a replayable redacted artifact set where permitted. Mark unavailable model determinism explicitly. Timing must separate generation, evidence I/O, calculation, planning and execution.

## Claim limits

A schema-valid sequence is a proposal. A successful helper calculation is geometry evidence, not proof that RA chose a correct sequence. A position plan is not a grasp or assembly result. Simulation success is not physical robot success. Make only claims supported at each tested level.

The existing medium-gear program is a pilot proposal, not proof of reliable composition or successful assembly. New composition from a fixed primitive library alone is not sufficient novelty. A defensible contribution needs controlled evidence that the proposed information/authority boundaries and RA reasoning improve composition coverage or quality under the stated partial contracts, with honest baselines, failures, and limitations.

## Implemented recording boundary

The nominal treatment can now use the bounded refinement flow: three candidate versions, two PA investigation batches, twelve PA evidence operations and five minutes by default. Freeze the profile before comparison. Record first-pass and each revised version separately using `composition/refinement_runs/run_*/` candidate/report references and event elapsed times. Selected calculations and resolved values remain separate from RA's authored parameters. Unknown validation coverage is a non-pass, and no fallback controller movement counts as an authored intermediate step.

The controlled end-to-end fixture starts with unbound geometry, supplies PA-selected records, preserves the first candidate, and validates an RA-authored second version using actual numerical calculators and mocked planning. Separate fixtures reject missing clearance movement and pre-insertion-only release. These are software tests, not model-quality or physical-success measurements. Live no-motion results and failures are reported separately in [implementation status](IMPLEMENTATION_PLAN.md). Physical success remains unobserved until the independent execution experiments are run.

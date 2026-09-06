# Bias validation: audit inputs and test counterfactuals

The implementation removes specific answer-shaping paths; it does not prove that PA or the VLM is unbiased. You do not need to examine every source file yourself. Start with the actual inputs saved for a run, then test whether the result follows evidence when irrelevant presentation details or relevant physical facts change.

## What changed, and what remains uncertain

| Risk | Implemented control | Remaining empirical question |
| --- | --- | --- |
| Camera names reveal scene roles | Opaque randomized view/candidate metadata; real frames stay internal | Can visual background or capture order still cue a memorized answer? |
| Candidate rank/order drives selection | Randomized pinned presentation; all CAD measurements retained | Does PA still prefer a displayed position or a small dimension advantage? |
| One forced current/desired pair determines meaning | Pairwise relationship collection with independent state membership | Does PA cover the complete goal and distinguish current observations from intended relations? |
| PA interpretation contains a plausible unsupported claim | Exact evidence lineage and deterministic structure validation; no independent runtime semantic reviewer | Does PA interpret the supplied evidence correctly, including the whole goal? |
| Valid citations hide unsupported meaning | PA owns interpretation; separate post-prediction evaluation scores task roles and complete goals | How often does a structurally valid answer have unsupported meaning? |
| Uncertainty disappears | Source-derived caveats persisted/displayed | Is qualifying wording adequate, and is ambiguity correctly retained? |
| Host chooses an arm by convenience | PA choice, required capability/reachability, no host substitution | Does PA show unsupported arm-name/order preference when both qualify? |

Approved CAD filenames and document part names are legitimate supplied evidence. Their usefulness should still be measured through label/distractor ablations. Segmentation thresholds, visible backgrounds, incomplete documents, measurement error, prompt wording, model training priors and shared-model errors remain potential sources of systematic error.

## Inspect one run

Start from a fresh interaction. Record the code revision/diff, exact requirement and clarifications, model configuration, approved source hashes, observation and calibration revisions. Do not reuse a previous accepted target or allocation. Saved examples and evaluator answers must not enter the runtime.

Inspect these records in order under `contexts/<interaction_identifier>/`:

1. `products/user_requirement/product_requirement.json` and `interaction_record/clarification_*.json`: distinguish information supplied by the operator from facts inferred by PA. A clarification can legitimately narrow the task, but that changes the experimental condition.
2. `products/grounding/ontology_grounding/request_*.json`: the PA request, tool schemas and response contract. Look for a preferred candidate/shaft/arm, assumed installation, a task recipe or a prescribed current/desired answer.
3. `interaction_record/model_tool_exchange_*.json`: actual projected tool arguments/results, including retrieved descriptions and retry investigation. Inspect error responses too. Camera roles, simulator names, real frames, canonical candidate indexes, configured object poses and expected answers must be absent from model-facing metadata. Source document text and approved CAD names are allowed exceptions, not sensor metadata.
4. `products/grounding/ontology_grounding/proposal_*.json`: inspect rejected attempts too. Verify that the accepted desired state covers the requirement, relationship membership is explicit, and bindings refer to observed evidence without inventing final insertion poses.
5. Allocation presentation/tool calls, reachability and selection records: the chosen arm must match its accepted check and every grounded location must be covered. Check the retained MoveIt service/group/TCP, every exact location binding, joint-plan results and live start state. The scope is position planning, with no static box/radius fallback or claim of validated grasping/insertion.
6. `interaction_record/context_completion_0001.json` and the reloaded UI: compare ontology, both state/reference galleries, relationship caveats, selected arm and limits. Historical records cannot start RA work.

Request audits preserve application-level model inputs and artifact identities, not private model reasoning or a provider's internal processing. Source/artifact hashes detect changed evidence; they do not prove that the source is true or the model interpreted it correctly. Keep audit files out of subsequent recognition inputs.

## Controlled experiment matrix

Use paired trials: keep all factors fixed except the factor named below. Use new interaction roots and the same model/configuration within a comparison. For presentation-only tests, replay the exact approved RGB/depth/calibration bytes with new presentation mappings. Do not edit signed records in place. Compare physical/source identities in the separate evaluator rather than comparing randomized handle strings.

| Condition | Change | Expected evidence-sensitive behavior |
| --- | --- | --- |
| Presentation invariance | Permute source, view, crop, candidate, location and resource order; generate fresh opaque handles | Supported physical identities and required relationships remain consistent, allowing explicitly supported interchangeable alternatives |
| Physical destination | Move/rearrange candidate objects while keeping requirement and part types fixed | Coordinates and any layout-dependent role follow the new evidence, not a remembered second/middle candidate |
| CAD distractors | Add similarly sized candidates or change a small dimension-error advantage without adding role evidence | Type matching alone does not establish a unique destination; ambiguity persists unless other evidence resolves it |
| Missing role evidence | Remove the distinguishing page/crop/layout cue using a separately approved test corpus/capture | Correct abstention, further investigation or qualification; no confident missing role invented |
| Installation ambiguity | Occlude contact or present a loose/surface-contacting counterpart instead of verified attachment | Current attachment remains qualified/unresolved; manual intention is not promoted to observed installation |
| Contradictory evidence | Supply independently prepared approved evidence inconsistent with a proposed relation | Measure whether PA investigates the contradiction and corrects its proposal or stops incomplete within the shared budget |
| Whole-goal coverage | Require several pairwise relations or add a supported required relation | All required relationships/destinations appear; one easy subproblem is not reported as the whole goal |
| Arm reachability | Change calibrated location evidence so only one arm qualifies, neither qualifies, or both qualify | Select only a checked eligible arm; no completion when none qualifies; either eligible arm allowed when both qualify |
| Robot-model sensitivity | Change a robot joint configuration or collision-scene obstacle while keeping the same product evidence and former example limits | Replan and follow the real robot result; service failure must remain distinct from rejected planning |
| Arm presentation | Reverse resource order while capabilities and reach results are unchanged | No host fallback; measure any unsupported model preference rather than demanding an artificial 50/50 split |
| Wording/labels | Paraphrase requirements without changing meaning; use a separately registered neutral-name CAD corpus with documented exact mappings | Required relationships remain evidence-backed; report how much performance depends on semantic corpus labels |
| Holdout | Use unseen supported requirements, parts, layouts, candidate counts, backgrounds and evidence paths | Measure generalization separately from the development scene |

Do not rename formal project symbols or normalize identities in production to create an experiment. Label experiments require a separate approved test corpus/profile and evaluator mapping. Missing-document/capture experiments intentionally change source authority; do not bypass provenance checks to make them run.

## Protocol and scoring

Before collecting results, freeze the matrix, trial counts, stopping rule, annotation rules and success criteria. Use a pilot only to debug the harness, then exclude pilot runs from the frozen test set. An independent annotator should determine supported identities, relations, attachments and permissible alternatives from approved source evidence without seeing the model prediction. Keep those labels outside recognition and reveal them to the evaluator after prediction finalization.

Run multiple independent trials per condition, balancing condition order. As a practical exploratory starting point, use 20 paired trials per condition; this is not a statistical power justification or a generalization claim. Choose the final sample size for the error rate/change you need to estimate, before examining test outcomes. Save every failure, retry, clarification and abstention. Do not rerun until success and report only the successful run.

Score separately:

- Whole-goal and relationship/state-membership correctness.
- Part identity versus specific task-role correctness.
- Semantic-to-coordinate binding, with a declared measurement tolerance.
- Unsupported current-installation claims and lost caveats.
- Correct abstention on missing/ambiguous evidence and unnecessary abstention on sufficient evidence.
- Structurally accepted but semantically unsupported claims, and unnecessary abstention on supported claims.
- Arm eligibility/coverage and unsupported preference; no optimality claim when both arms qualify.
- Stage completion, tool/model calls, latency, service failures, operation/proposal usage, corrections and budget/no-progress stops.

Report outcome counts over all trials, accepted-result correctness separately from completion rate, paired changes across conditions, and uncertainty intervals appropriate to the sampling design. Do not treat many assertions from one scene as independent scene trials. Zero observed errors in a finite sample is not evidence of zero possible errors.

Useful ablations compare frozen runtime revisions or investigation budgets in an isolated evaluation harness. Historical reviewer-based runs may be evaluated as a separate frozen condition, without adding reviewer flags or compatibility paths to current production. Keep execution disabled and preserve deterministic provenance checks. A separate independent evaluator can score correlated errors after prediction finalization without becoming a runtime answer source.

## What you can claim

After passing contract tests: “The tested gates reject malformed/altered evidence, preserve caveats and prevent historical RA bypass.”

After live trials: report exactly which conditions, scenes, models and sample sizes were tested, with observed accuracy/abstention/error rates. “No answer leakage was found in the inspected inputs” is narrower and more defensible than “the system has no bias.” If a permutation changes a supported physical answer or removed evidence leaves an unsupported confident answer, preserve the trace, diagnose the source and rerun a new frozen evaluation after the fix.

No live counterfactual or hardware accuracy claim follows from the offline fixtures shipped with this change.

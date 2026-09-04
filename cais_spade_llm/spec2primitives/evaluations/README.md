# Evaluations

Post-prediction evaluation records belong here and remain unavailable to PA,
RA, retrieval tools, and recognition code. Generated evaluation contents are
ignored by Git.

An accepted PA `target_feature`, resource selection, or structural
`PrimitiveProgramDraft` is intent/provenance evidence, not proof that the
desired product state was realized. Reachability does not certify grasping,
contact, tolerance, insertion, execution, or outcome. Observation-backed
outcome evaluation are still deferred and must be reported separately when
implemented. Any later outcome verifier must derive the numeric geometry it
needs from PA-selected approved evidence; it must not receive a predetermined
target-geometry answer.

## Phase 4 blind protocol

Any interaction whose controller prompt prescribes a state answer, CAD choice,
candidate identity, resource, evidence order, or modality is excluded from
baseline results and may be reported only as a contaminated-prompt ablation.
`interaction_b4898e88d9a14c3b8e64ccd1f52184a6` is an evidence-only regression
source; its candidate identities, ordering, and expected result are not exposed
to PA.

Run fresh interactions for one blind regression requirement and at least one
held-out supported requirement with different wording, component identity,
candidate count/layout, source order, resource order, and valid evidence path.
Each run has a new interaction root and new presentation records; no proposal,
tool call, selection, completion, or evaluation record is copied between runs.
Record snapshot hashes and all actual presentation orders.

The allowed PA inputs remain the requirement, approved NIST documents, approved
candidate CAD files, RGB, depth, and camera calibration. Gazebo model or entity
names, world or SDF contents, spawn data, configured poses, detector answers,
evaluator labels, and ground truth remain unavailable until the PA prediction
is final.

| ID | Intervention | Five-run procedure |
| --- | --- | --- |
| 1 | Baseline | Start five independent interactions with independently pinned orders. Do not prescribe retrieval count, candidate choice, or resource choice. |
| 2 | Reversed order | For each corresponding baseline source set, inject the exact reverse evidence, neutral-candidate, and resource permutations through the existing reproducibility hooks. Change no evidence contents or validator behavior. |
| 3 | Selected-resource rejection | Make one PA-selected resource fail reachability while leaving other resources unranked. The stage must fail without host substitution or a correction prompt. |
| 4 | Candidate-count variation | Exercise 2, 3, and 5 neutral candidates without a hidden role, answer, expected relation, or order cue. |

For every run, audit the records in this order:

1. `EvidencePresentationRecord`: source set, fingerprint, and presentation order.
2. `ProductAgentToolCall`: PA-authored retrieval calls and hash-pinned results.
3. `OntologyGroundingProposal` v9: PA-authored target feature and evidence
   citations.
4. `AllocationPresentationRecord`: capable resources, neutral location handles,
   fingerprints, and both presentation orders.
5. `ProductAgentAllocationToolCall` and `ReachabilityCheckRecord`: exact
   PA-selected state handles and resource, with `selection_made_by_tool: false`.
6. `ResourceSelectionRecord` v5: PA choice, exact handles, and cited accepted
   reachability.
7. `PAContextGroundingCompletion` v7: final hash-pinned Phase 4 authority when
   the interaction completes.

Record the candidate and location handles PA used, selected resource, retrieval
count and order, clarification or deterministic stage code, validator status,
and whether selected items were first in their presented lists. Report all
runs, including failures and non-triggered interventions.

The autonomy evidence passes these checks when reversing order does not produce
a systematic first-item rule, every semantic choice originates in a PA response
or PA tool call, every controller decision is limited to integrity, ontology,
capability, or reachability validation, and rejection never causes host
substitution or answer-shaping feedback. Different evidence may produce
different PA decisions; identical outputs are not by themselves failures when
their cited evidence supports them.

`RGBDSegmentationRecord` production, reachability calculation, RobotAgent
validation, and Phase 4 stage order are fixed deterministic boundaries. These
experiments test PA choices inside those boundaries. Do not report them as proof
of dynamic perception-chain selection, primitive execution, or realized
assembly outcome.

Populate results only after the runs complete. Five repetitions are an initial
artifact audit and do not by themselves establish statistical generality.

# Evaluations

Post-prediction evaluation records belong here and remain unavailable to PA,
RA, retrieval tools, and recognition code. Generated evaluation contents are
ignored by Git.

An accepted PA `target_feature`, `TargetFeatureSemanticReview`, or structural
`PrimitiveProgramDraft` is intent/provenance evidence, not proof that the
desired product state was realized. Grasp/contact/tolerance/insertion
validation, execution, and observation-backed
outcome evaluation are still deferred and must be reported separately when
implemented. Any later outcome verifier must derive the numeric geometry it
needs from PA-selected approved evidence; it must not receive a predetermined
target-geometry answer.

## Phase 4 Medium Gear protocol

`interaction_d93d297a1ddc4bdc85cbbe60579a2b57` is immutable and excluded from
all baseline results because its controller prompt prescribed the state answer,
CAD choices, and evidence modalities. It may be reported only as a
contaminated-prompt ablation. Gate 1 intentionally leaves the present desired
state ambiguous and must stop before semantic review, allocation, or completion;
fresh baseline runs begin only after the separately authorized relational-
grounding gate and its smoke run pass.

Run five fresh interactions for each condition. A fresh interaction has a new
interaction root and new presentation records; no proposal, tool call,
selection, completion, or evaluation record is copied between runs. Use the
exact `product requirement: assemble Medium Gear`, the same approved source set,
and the same pinned workcell and registry snapshots. Record snapshot hashes and
all actual presentation orders for every run.

The allowed PA inputs remain the requirement, approved NIST documents, approved
candidate CAD files, RGB, depth, and camera calibration. Gazebo model or entity
names, world or SDF contents, spawn data, configured poses, detector answers,
evaluator labels, and ground truth remain unavailable until the PA prediction
is final.

| ID | Intervention | Five-run procedure |
| --- | --- | --- |
| 1 | Baseline | Start five independent interactions with independently pinned orders. Do not prescribe retrieval count, candidate choice, or resource choice. |
| 2 | Reversed order | For each corresponding baseline source set, inject the exact reverse evidence, neutral-candidate, and resource permutations through the existing reproducibility hooks. Change no evidence contents or validator behavior. |
| 3 | `xarm6` rejection | Configure plan-only validation to reject a PA-authored provisional `xarm6` and leave other results unchanged. Do not force the PA to choose `xarm6`; record non-triggered runs explicitly. |
| 4 | Ambiguous candidates | Supply two neutral segmentation candidates for which the allowed evidence does not establish a unique state assignment. Do not include a hidden role, answer, or order cue. |

For every run, audit the records in this order:

1. `EvidencePresentationRecord`: source set, fingerprint, and presentation order.
2. `ProductAgentToolCall`: PA-authored retrieval calls and hash-pinned results.
3. `OntologyGroundingProposal` and `TargetFeatureSemanticReview`: PA-authored
   target feature, evidence citations, and any revision.
4. `AllocationPresentationRecord`: capable resources, neutral candidates,
   fingerprints, and both presentation orders.
5. `ProductAgentAllocationToolCall` and `ReachabilityCheckRecord`: exact
   PA-selected state handles and resource, with `selection_made_by_tool: false`.
6. `ResourceSelectionRecord`: provisional choice, RobotAgent verdict, and either
   the identical accepted choice or no selected resource.
7. `PAContextGroundingCompletion`: final hash-pinned Phase 4 authority when the
   interaction completes.

Record the initial and final candidate handles, initial and final provisional
resource symbols, retrieval count and order, revision count, clarification or
`insufficient_evidence`, validator status, and whether the selected item was
first in its presented list. Report all five runs, including failures and
non-triggered Experiment 3 runs.

The bounded-autonomy evidence passes these checks when reversing order does not
produce a systematic first-item rule, every accepted candidate and resource was
authored by PA before validation, rejection never causes host substitution, and
the ambiguous condition ends in admissible evidence seeking, clarification, or
`insufficient_evidence` rather than an unsupported forced choice. Different
evidence or validation feedback may produce different PA decisions; identical
outputs are not by themselves failures when their cited evidence supports them.

`RGBDSegmentationRecord` production, reachability calculation, RobotAgent
validation, and Phase 4 stage order are fixed deterministic boundaries. These
experiments test PA choices inside those boundaries. Do not report them as proof
of dynamic perception-chain selection, primitive execution, or realized
assembly outcome.

Populate results only after the runs complete. Five repetitions are an initial
artifact audit and do not by themselves establish statistical generality.

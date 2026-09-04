# Phase 4 reference answer: assemble Medium Gear

This is the evaluator-only reference answer for:

`product requirement: assemble Medium Gear`

It preserves the result that was confirmed as correct. The source result came
from the biased framework and is therefore a contaminated-prompt ablation, not
evidence that the unbiased method independently found the answer.

## Expected Phase 4 answer

- `required_process`: `assembly`
- `current_state`: the Medium Gear is separate from the task board gear assembly;
  the selected observation must depict the Medium Gear.
- `desired_state`: the Medium Gear is mounted at the supported destination on the
  task board gear assembly; the selected observation must depict that supported
  destination.
- CAD identity: `Gear_Medium.STL`
- `selected_resource_symbol`: `xarm6`
- `selected_resource_jid`: `xarm6@localhost`
- `selected_execution_mode`: `simulation`

For this case, the `current_state` observation and `desired_state` observation
refer to distinct physical roles. Fresh interaction handles, presentation order,
record numbers, fingerprints, and hashes are not part of the expected answer and
must be evaluated through their referenced evidence.

## Provenance and use boundary

- Source interaction: `interaction_d93d297a1ddc4bdc85cbbe60579a2b57`
- Classification: immutable contaminated-prompt ablation
- The original interaction directory is not present under `contexts/` in the
  current workspace. This file preserves the confirmed answer, not a
  hash-identical copy of the original runtime records.
- A blind run must finalize its Phase 4 prediction before a separate evaluator
  reads this file.
- PA, RA, retrieval tools, recognition code, prompts, validation feedback, and
  candidate or resource ordering must never receive this answer.
- Agreement with this answer is an evaluation result. It must not be reported as
  proof of independence unless the blind protocol and artifact audit also pass.
- `grounding complete` does not prove primitive composition, execution, or an
  observation-backed realized assembly outcome.

# Evaluator-only ground truth

Frozen reference answers and annotations are available only to a separate evaluator after the prediction is finalized. PA, retrieval, recognition and runtime prompts cannot read this directory's answers or simulator identity/pose information.

Reference answers are historical evaluation artifacts, not current framework instructions. They are not rewritten when schemas change. Score new proposal/evidence/completion runs under preregistered criteria, including uncertainty and correct abstention, and label old runs separately.

See [the evaluation protocol](../README.md) and [bias experiments](../../BIAS_VALIDATION.md). No ground-truth file is an approved recognition source.

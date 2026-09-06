# Evaluation

Evaluation runs after the prediction is finalized, with separate evaluator authority. PA, retrieval and recognition must not import or read evaluator labels, answer files, simulator identities, configured poses or entity-state services.

Use [BIAS_VALIDATION.md](../BIAS_VALIDATION.md) as the Phase 4 audit/experiment protocol. Freeze requirements, clarifications, approved source revisions, scene captures, calibration, configuration and scoring rules before trials. Preserve all runs, including rejection, ambiguity, service failure and retry. Do not count a failed run only after selecting a successful rerun.

Current predictions use proposal, evidence, selection and completion. Finalization can also be incomplete; correct abstention matters when evidence is absent or ambiguous. Current arm assignment checks capability, joint limits and collision-aware position planning; it does not validate execution. Historical runs remain separately labeled under their original rules.

Audit actual PA prompts/tool results and deterministic validation feedback. Compare candidate identity in a separate evaluator through canonical source references, not randomized display handles. Check whole-goal completeness, relationship membership, current attachment claims, destination bindings, uncertainty and arm choice separately.

Use paired presentation permutations, physical counterfactuals, evidence-removal/contradiction cases, ambiguity controls, reachability changes and held-out requirements/scenes. Predeclare repeated-trial counts and outcome criteria. A few successful examples and deterministic fixture tests do not establish unbiased behavior. Report sample sizes, all outcome counts and uncertainty; report unsupported PA interpretations and correlated model/tool errors explicitly.

Existing finalized evaluation records and frozen reference answers remain unchanged. Any run that crosses the forbidden recognition-input boundary is invalid and must be excluded from performance claims, with the reason retained.

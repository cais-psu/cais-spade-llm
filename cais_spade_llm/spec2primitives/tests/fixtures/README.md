# Offline fixtures

Fixtures supply deterministic source geometry and agent/tool responses for contract tests. They are not model predictions, expected-answer prompts, or evidence of unbiased scene understanding.

Current fixtures exercise proposal, deterministic evidence validation, live MoveIt reachability, selection and completion. Rejection fixtures inject incompatible old shapes; there are no compatibility fixtures that authorize new RA work. Both arm identities are exercised in locations with accepted offline planning fixtures.

Keep runtime recognition independent of fixtures and evaluator labels. Use fresh scenes/model requests for the separate [bias experiments](../../BIAS_VALIDATION.md).

Composition fixtures use actual shared catalog metadata with mocked RA responses and no robot actions. Refinement fixtures additionally run the real pure calculators, compare the existing runtime helper arithmetic without initializing a controller, and inject controlled planning responses. They test minimal model-facing contracts versus unchanged full snapshots, nested geometry gaps, conditional outputs, configured frames, filtered reads and exact proposal preservation. Simulator-identifier fixtures check rejection in new composition inputs and preservation of an older attempt's recorded interface; they do not resolve real simulator identities or authorize execution. They are not demonstrations for RA or independent labels for composition quality; use [COMPOSITION_EVALUATION.md](../../COMPOSITION_EVALUATION.md) for those experiments.

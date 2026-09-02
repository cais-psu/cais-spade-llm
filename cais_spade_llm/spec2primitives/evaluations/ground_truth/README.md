# Ground truth

Ground truth may be read only by the separate evaluator after a prediction is
finalized. It must never be exposed through product references, resource
references, runtime contexts, agent adapters, or controlled tools.

In particular, PA may not use evaluator data to author or semantically review
`target_feature`, and RA may not receive it while composing a structural
`PrimitiveProgramDraft`. Ground truth remains post-prediction evidence only.

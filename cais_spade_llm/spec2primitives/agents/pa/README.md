# PA

`context_interaction.py` implements Phase 3.1 through composition with the shared
ProductAgent. It accepts the exact `product_requirement`, exposes only
`ask_llm_structured(...)` through `ProductAgentContextRuntime`, and records one
validated `needed_context` decision under the caller-owned interaction root.

`context_serving.py` implements Phase 3.2. It reads the completed Phase 3.1
records, resolves exactly one approved document or CAD `context_ref`, or captures
one explicitly requested numbered fresh live RGB-D observation. It preserves
static evidence, live artifact references, internal `provenance`, and
`retrieval_error` under the same caller-owned interaction root without
substituting fallback evidence. The public Phase 3.2 entrypoint retains its exact
one-request behavior.

`context_assessment.py` implements the Phase 3.3 loop mechanics. It resumes a
strictly successful Phase 3.1-3.2 interaction, records the operator-selected
`max_pa_turns`, and runs the structured assess, request, serve, observe, and
reassess loop. The current implementation still accepts early clarification and
early `context understanding complete`. The planned retrieval-first correction
will instead transition to Phase 4 with `needed_context: null` and
`context understanding complete: false`; only the later Phase 4.3 decision may
enter clarification or completion.

`product_agent_runtime.py` composes one shared ProductAgent behind the narrow
`ProductAgentContextRuntime` interface used by the Phase 2.1 UI connection. It
delegates only `ask_llm_structured(...)` and never calls ProductAgent `setup()`.

No implemented PA step starts ProductAgent SPADE behaviours, calls a VLM,
retrieves the primitive catalog, builds an assembly plan, connects RA or CCA,
performs grounding, or executes robot behavior. Those remain later, separately
authorized phases.

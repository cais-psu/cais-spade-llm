# Cases

This directory will contain controlled Spec2Primitives case-study inputs.

The starting case is:

```text
product requirement: assemble Medium Gear
```

The implemented PA binds this requirement to reviewed evidence by authoring one
v8 `target_feature` with current and desired states. The host creates the feature
and both state individuals; semantic review and v6 completion pin the accepted
endpoint-motion allocation; and Phase 5.2A reconstructs it for the RA LLM. This
is one development case, not a hard-coded evidence sequence, primitive catalog
size, predetermined target answer, or expected `primitive_steps` recipe.

The production workcell profile intentionally configures only `assembly`,
`xarm6`, and `ur5e`, so this deployment is an assembly case study rather than a
runtime process-discovery claim. The target-feature and allocation contracts are
process-independent: additional configured processes and resources can use the
same PA/RA handoff without source-code changes once truthful evidence,
capabilities, and validators exist.

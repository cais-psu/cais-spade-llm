# Cases

This directory will contain controlled Spec2Primitives case-study inputs.

The starting case is:

```text
product requirement: assemble Medium Gear
```

The implemented PA binds this requirement to evidence by authoring one v9
`target_feature` with current and desired states, then independently choosing
location handles and a capable reachable resource. The host commits seven
feature/state assertions and four `processExecution`/resource assertions;
completion v7 pins the result, and Phase 5.2A reconstructs it for the RA LLM.
This is one blind regression case, not a hard-coded evidence sequence, primitive
catalog size, predetermined target answer, or expected `primitive_steps` recipe.

The production workcell profile intentionally configures only `assembly`,
`xarm6`, and `ur5e`, so this deployment is an assembly case study rather than a
runtime process-discovery claim. The target-feature and allocation contracts are
process-independent: additional configured processes and resources can use the
same PA/RA handoff without source-code changes once truthful evidence,
capabilities, and validators exist.

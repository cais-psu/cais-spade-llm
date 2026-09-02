# Agents

Only PA and RA are agents in the Spec2Primitives roadmap. Their shared ProductAgent
and RobotAgent implementations remain outside this package and read-only.

- `pa/` owns the implemented narrow ProductAgent context adapter, single
  evidence-grounded two-state `target_feature`, semantic review, PA-controlled
  allocation, exact RobotAgent endpoint-motion validation, v6 completion contracts,
  and production grounding workflow through Phase 4.4.
- `ra/` owns the implemented Phase 5.1 contract-first assignment and resource
  context boundary plus the implemented Phase 5.2A structural primitive draft.
  Primitive binding, primitive-level validation, and execution remain future work.

The PA boundary delegates only structured context calls and does not own the
shared ProductAgent lifecycle. The RA adapter can reuse or start only the exact
Phase 4-selected context-only RobotAgent under its documented simulation gate;
it does not change the shared RobotAgent implementation. The same adapter
performs exact-resource no-motion allocation validation. Live SPADE delivery,
binding, execution validation, and execution require later separately
authorized implementation phases.

# Agents

Only PA and RA are agents in the Spec2Primitives roadmap. Their shared ProductAgent
and RobotAgent implementations remain outside this package and read-only.

- `pa/` owns the implemented narrow ProductAgent context adapter, PA context
  contracts, and production grounding workflow through Phase 4.4.
- `ra/` owns the implemented Phase 5.1 contract-first assignment and resource
  context boundary plus future structural primitive drafts, composition, and
  robot-local validation.

The PA boundary delegates only structured context calls and does not own the
shared ProductAgent lifecycle. The Phase 5.1 RA boundary invokes only an
injected read-only runtime and does not start the shared RobotAgent lifecycle.
Live SPADE delivery and every later RA behavior require their separately
authorized implementation phases.

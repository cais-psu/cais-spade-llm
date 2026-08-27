# Agents

Only PA and RA are agents in the Spec2Primitives roadmap. Their shared ProductAgent
and RobotAgent implementations remain outside this package and read-only.

- `pa/` owns the implemented narrow ProductAgent context adapter, current PA
  context contracts, and future production grounding workflow.
- `ra/` owns the future Spec2Primitives RobotAgent adapter, structural primitive
  drafts, resource context, primitive composition, and robot-local validation.

The PA boundary delegates only structured context calls and does not own the
shared ProductAgent lifecycle. No RobotAgent connection or RA behavior exists
yet. Every later adapter or agent behavior requires its separately authorized
implementation phase.

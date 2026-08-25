# Agents

Only PA and RA are agents in the Spec2Primitives roadmap. Their shared ProductAgent
and RobotAgent implementations remain outside this package and read-only.

- `pa/` owns future Spec2Primitives PA adapter and PA-specific workflow code.
- `ra/` owns future Spec2Primitives RobotAgent adapter and RA-specific workflow code.

These directories do not connect either agent yet. Every adapter or agent
behavior requires its separately authorized implementation phase.

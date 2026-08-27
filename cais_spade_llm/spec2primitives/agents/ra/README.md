# RA

RA means RobotAgent. This directory is reserved for the future Spec2Primitives-owned
RobotAgent adapter, contracts, context retrieval, primitive composition, and
robot-local validation workflow. It contains no RobotAgent connection or
behavior yet.

The selected RA will retrieve its fresh state and complete current
primitive-only catalog without assuming a fixed number of entries. It will
preserve every catalog symbol exactly, pin the catalog fingerprint for one
composition attempt, and author a structural `PrimitiveProgramDraft`. A
deterministic binding preflight may identify missing inputs but cannot create or
repair primitive steps. RA-owned inputs remain local; product or scene inputs
are deduplicated into a `MissingContextBatch` for PA. After receiving a
versioned `CompositionContextBundle`, RA alone authors the fully bound
`primitive_steps` candidate. This paragraph describes planned behavior only.

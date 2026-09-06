# Primitive catalog references

Primitive catalogs are owned by the exact selected RobotAgent and captured as paired Phase 5.1 state/catalog snapshots after current completion passes review/reachability recovery checks. Preserve exact symbols, order, typed parameters, results, limits and provenance; no fixed catalog cardinality is assumed.

Phase 5.2A authors an unbound structural `PrimitiveProgramDraft` using only current catalog symbols. Phase 4 arm assignment validates robot position plans, while primitive coverage, attached-part motion, grasping and execution feasibility remain unvalidated. Parameter binding, executable validation and execution remain future work.

Do not supply completed task recipes or expected sequences through catalog metadata. See [RA behavior](../../../agents/ra/README.md) and [bias validation](../../../BIAS_VALIDATION.md).

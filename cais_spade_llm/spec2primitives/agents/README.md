# Agents

Spec2Primitives owns adapters for PA and RA; their shared ProductAgent and RobotAgent implementations remain read-only.

- `pa/`: evidence investigation, proposal with pairwise relationships, deterministic structure/provenance validation, required evidence-based arm choice, live MoveIt reachability and completion.
- `ra/`: selected-RA envelope, paired context snapshots and one unbound structural primitive draft per pair. Current completion/evidence gates apply to activation, recovery and drafting.

PA chooses semantics and the arm; host code checks evidence provenance, exact identities, capability and all grounded locations. Resource assignment is an ontology relation, not a motion-plan or execution result. Historical records remain readable but cannot authorize new RA work.

The context-only RA adapter retains its existing readiness/lifecycle boundary. Live SPADE delivery, parameter binding, primitive-level validation and execution remain future work. See [bias validation](../BIAS_VALIDATION.md).

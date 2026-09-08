# Agents

Spec2Primitives owns adapters for PA and RA; their shared ProductAgent and RobotAgent implementations remain read-only.

- `pa/`: evidence investigation, proposal with pairwise relationships, deterministic structure/provenance validation, required evidence-based arm choice, live MoveIt reachability and completion.
- `ra/`: selected-RA envelope, paired context snapshots and one concise program per composition attempt, with available parameters and a derived binding report. Current completion/evidence gates apply to activation, reload and composition.

PA chooses semantics and the arm; host code checks evidence provenance, exact identities, capability and all grounded locations. Resource assignment is an ontology relation, not a motion-plan or execution result. Historical records remain readable but cannot authorize new RA work.

The context-only RA adapter retains its existing readiness/lifecycle boundary. The bounded Phase 5 flow now adds supplemental PA investigation, measured RA geometry, strict target calculations and private MoveIt validation with findings returned to RA. Physical execution and observed outcomes remain future work. See [bias validation](../BIAS_VALIDATION.md).

Phase 4 supplies observed product facts and desired assembly relationships; Phase 5 chooses primitives and later computes robot targets. Only RA authors step order and parameters. See [the RA guide](ra/README.md), [bounded validation/revision](../VALIDATION_AND_REVISION.md) and [composition experiments](../COMPOSITION_EVALUATION.md).

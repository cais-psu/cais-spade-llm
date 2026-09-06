# ProductAgent grounding

This package owns the non-executing connection to the shared ProductAgent; shared lifecycle/interfaces remain unchanged.

## Call path

`start_pa_context_interaction` records the exact requirement and initializes the ABox. `ProductionProductContextGroundingRuntime.ground_product_context` runs investigation → deterministic validation and bounded correction → ontology commit → required arm assignment → completion.

PA receives the requirement, configured processes, PPR structure, source handles, and `retrieve`, `query_document`, `compare_cad_size`, `analyze_candidate_layout`. It chooses sources/order, within shared configured limits of 24 evidence operations and 6 proposals by default. Genuine requirement clarification remains available. Replies are exact append-only evidence. Clarification resume creates a fresh observation investigation; proposal correction reuses the pinned snapshot.

## Evidence and presentation

Document retrieval yields an ordered `DocumentSourceIndexRecord`; PA can ask a question for `DocumentQueryRecord`. CAD supplies measured mesh geometry. RGB-D supplies neutral segmentation/crops and morphological review. CAD comparison reports all candidates without selecting a winner. Layout reports same-view geometry without semantic role decisions.

`EvidencePresentationRecord` pins randomized source handles. `ObservationPresentationRecord` hides view/candidate indexes and randomizes order. `AllocationPresentationRecord` pins resource/location presentation. Real sensors, canonical pointers, paths, frames and hashes stay internal. Approved CAD filenames and source text remain allowed. Exact original references are restored for geometry/provenance, without name normalization.

PA requests and projected tool exchanges are persisted for audit. They are not new semantic evidence. Recognition/review exclude simulator identities, configured poses, detector answers, evaluator data, expected answers and resource-selection results.

## Proposal and review

`ontology_grounding.py` owns the proposal: one complete `target_feature`, cited statements and typed values, plus a collection of pairwise assembly relationships. Association `state_names` is independent of endpoint observation bindings. Both endpoint binding fields are valid or both null. Current observations can support desired relations; null bindings cannot hide unresolved required roles.

PA owns source interpretation, candidate/task roles and whole-goal completeness. Deterministic validation checks schema, ontology, exact references and provenance. Invalid proposals receive generic structural/reference feedback without a task-specific answer. The shared operation/proposal budget applies across corrections; budget exhaustion or repeated invalid proposals without progress returns explicit incomplete status. Rejected assertions never merge.

Observation references are opaque from the first retrieval, including layout inputs. PA may combine visual, CAD and document evidence without requiring a measurement-tool winner. Part type alone does not establish a destination, and a document cannot prove current installation. Source uncertainty is derived from typed evidence/selected candidates and retained with exact references. No independent runtime semantic reviewer certifies these interpretations.

The host compiles root feature/state individuals and separate pairwise associations under their exact owners. Count is variable; TBox symbols/cardinality remain unchanged.

## Resource assignment and recovery

`_complete_resource_assignment` derives all grounded coordinate-bearing values in both states. Several associations do not require a unique Cartesian pair. Missing locations leave Phase 4 incomplete with the grounded result visible.

PA calls `check_reachability` for every configured capable arm against the same grounded current and desired locations, then cites an accepted result for its choice. One bounded correction lists unchecked arms and existing results; completed checks are reused within the allocation attempt. Continued omission returns `invalid_resource_selection`. Unavailable planning remains distinct from rejected planning. Checks use calibrated locations and the selected simulated robot’s live MoveIt state, joint limits and collision scene for every grounded location. Either eligible arm may be selected; the host cannot prefer or substitute an arm. Commit revalidates selection and adds four `processExecution`/resource assertions. MoveIt plans to each grounded position without constraining tool orientation or executing motion.

Completion pins proposal, evidence, selection, reachability, evidence, presentations, capability snapshots, assignment delta and final context. Recovery rechecks hashes, deterministic evidence snapshot/bindings, exact location coverage and pinned MoveIt requests/results. It does not write a duplicate target or new `TypedGroundingContract`.

Each record type has one current unversioned format. Saved interactions are never converted, deleted or rewritten; incompatible records require “Start a fresh interaction”. Phase 5 remains context capture and unbound structural composition.

See [ontology examples](../../ASSEMBLY_ONTOLOGY.md), [records](../../contexts/README.md), [production tests](../../tests/test_pa_production_grounding.py), and [bias experiments](../../BIAS_VALIDATION.md). Offline contracts do not establish model accuracy or absence of bias.

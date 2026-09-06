# Contexts

Each caller-owned interaction root contains exact requirements, observations, typed sources, PA decisions, evidence, assignments and later RA snapshots. Saved runs are immutable evidence; do not rewrite them to fit a newer schema.

## Current artifact map

```text
contexts/<interaction_identifier>/
  products/user_requirement/product_requirement.json
  products/observations/<observation_ref>/
  products/served_references/
  products/grounding/
    document_evidence/
    rgb_d_cad_grounding/
    presentation/evidence_presentation_record.json
    presentation/observation_presentation_record.json
    presentation/allocation_presentation_record.json
    ontology_grounding/request_<proposal>_<request>.json
    ontology_grounding/proposal_<number>.json
    reachability/check_<number>/reachability_check_record.json
    resource_selection/selection_<number>/resource_selection_record.json
    ontology/
    product_context/view_<number>.json
    completion/
  interaction_record/
    model_tool_exchange_<number>.json
    allocation_tool_call_<number>.json
    clarification_<turn>.json
    context_completion_0001.json
  composition/selected_ra_assignments/
  composition/primitive_program_drafts/
  resources/<exact_RA_identifier>/robot_state/
  resources/<exact_RA_identifier>/primitive_catalog_snapshot/
```

Current completion pins proposal, selection, reachability, all source/typed evidence, presentations, capability snapshots, assignment delta and final context. Proposal `grounding_evidence` pins the precommit context view and source/artifact hashes. Audit requests preserve model-facing content; internal hashes and paths remain provenance rather than model hints.

Corrections reuse the pinned observation and mapping within the shared configured investigation budget. Rejected proposals never contribute assertions. Source uncertainty persists in accepted context. Existing requests, tool records and terminal output retain host-owned `grounding_progress` operation/proposal counts, generic validation feedback and stop reasons. A grounded result lacking accepted assignment remains inspectable without manufacturing completion.

RGB-D bundles retain lossless RGB, metric depth and `fixture`, `replay`, or `live` labels. Canonical sensor/candidate identities, calibration frames, hashes and geometry remain internal. Randomized model-facing handles resolve back to those exact records. Calibration must match the originating observation timestamp. The approved simulation calibration may be replaced through `SPEC2PRIMITIVES_CAMERA_TO_WORLD_CALIBRATION_PATH`; recognition never reads a world or spawn manifest.

Document source indexes and existing overview caches live under `contexts/source_cache/` outside an interaction. Interaction evidence pins its source revision. Active CAD comparison records all measurements; pose diagnostics consume the same measurement format.

Phase 5.1 requires current completion before writing envelope or reading recovered paired context. Phase 5.2A appends one unbound draft per context pair. It reconstructs the target from pinned authorities rather than copying it. Binding bundles, missing-context batches, bound candidates, execution validation and execution records remain future work.

Saved interactions remain untouched. There is one current format per record type. Incompatible records require “Start a fresh interaction” and cannot authorize new RA work; no conversion or compatibility reader is provided.

See [schemas](../schemas/README.md) and [bias audit paths and experiments](../BIAS_VALIDATION.md). Evaluator answers belong under `evaluations/` and never enter recognition inputs.

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
  composition/primitive_program_candidates/attempt_<number>/
    request.json
    exchange_<number>.json
    candidate.json
  resources/<exact_RA_identifier>/robot_state/
  resources/<exact_RA_identifier>/primitive_catalog_snapshot/
```

Current completion pins proposal, selection, reachability, all source/typed evidence, presentations, capability snapshots, assignment delta and final context. Proposal `grounding_evidence` pins the precommit context view and source/artifact hashes. Audit requests preserve model-facing content; internal hashes and paths remain provenance rather than model hints.

Corrections reuse the pinned observation and mapping within the shared configured investigation budget. Rejected proposals never contribute assertions. Source uncertainty persists in accepted context. Existing requests, tool records and terminal output retain host-owned `grounding_progress` operation/proposal counts, generic validation feedback and stop reasons. A grounded result lacking accepted assignment remains inspectable without manufacturing completion.

RGB-D bundles retain lossless RGB, metric depth and `fixture`, `replay`, or `live` labels. Canonical sensor/candidate identities, calibration frames, hashes and geometry remain internal. Randomized model-facing handles resolve back to those exact records. Calibration must match the originating observation timestamp. The approved simulation calibration may be replaced through `SPEC2PRIMITIVES_CAMERA_TO_WORLD_CALIBRATION_PATH`; recognition never reads a world or spawn manifest.

Document source indexes and existing overview caches live under `contexts/source_cache/` outside an interaction. Interaction evidence pins its source revision. Active CAD comparison records all measurements; pose diagnostics consume the same measurement format.

Phase 5.1 requires current completion before writing an envelope or reading recovered paired context. Composition reconstructs the target from pinned authorities and authors one program with available parameters. Each request directly pins completion, assignment, state and catalog snapshots. Historical draft files and draft-dependent attempts remain untouched and are excluded from new inputs. `composition/refinement_runs/run_*/` appends context requests, PA investigations, measured robot contexts, calculation/validation records, events and a stop result; candidate references preserve every RA version. Execution bindings and robot execution remain future work.

Saved interactions remain untouched. There is one current format per record type. Incompatible records require “Start a fresh interaction” and cannot authorize new RA work; no conversion or compatibility reader is provided.

See [schemas](../schemas/README.md) and [bias audit paths and experiments](../BIAS_VALIDATION.md). Evaluator answers belong under `evaluations/` and never enter recognition inputs.

The current `binding_issues` report is a derived diagnostic, not a new authoritative record or a rewritten candidate. Each item contains `step_index`, `parameter_path`, `status` (`missing`, `incompatible`, `unverified`, or `deferred`) and `message`. A deferred result refers to an unexecuted earlier step; it is distinct from missing source evidence.

Each attempt is validated and displayed using the catalog embedded in its saved `request.prompt` `COMPOSITION_INPUT` payload. The reader checks that catalog against the pinned context after projection. Older programs retain their original `model_name` arguments/findings; new attempts use the simplified composition interface with that execution binding excluded. This uses existing record fields and does not rewrite snapshots, requests, traces or candidates. Independent new runs exclude prior programs. Within one run, refinement explicitly pins and supplies its preceding candidate and findings.

Fresh state snapshots include configuration-sourced `motion_context` with frame/EE/TCP names. Full runtime state/contracts stay captured, while initial composition and record reads use a filtered state view and minimal grasp/release contracts. Hash checks apply to original bytes. Old snapshots retain their original declarations and saved programs remain unchanged; recapture context to use corrected metadata. See [RA input contracts](../agents/ra/README.md).

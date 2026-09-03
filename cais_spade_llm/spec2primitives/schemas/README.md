# Schemas

This directory remains reserved for shared data templates and contracts. Schemas
define the shape and required fields of stage inputs and outputs; they are not
experiment results.

Phase 4.0 ontology code is not stored as a schema in this directory. Shared
immutable PPR TBox loading and validation live in `ontology/ppr_tbox.py`, while
the project-authoritative schema-only vocabulary lives in
`ontology/spec2primitives_ppr_tbox.owl`, and
PA-owned writable interaction-ABox behavior lives in
`agents/pa/product_context.py`. The TBox supplies allowed classes and property
signatures; it does not require any particular relation for a requirement.
The validator rejects wrong-domain/range relations and RA,
primitive-offering, and `capableOf` assertions from the PA ABox.
The current framework adds no separate RA resource ABox. The selected RA's
primitive catalog remains a separately owned typed record; only the TBox is
shared.
Phase 4.0 is wired into Phase 3 through package-local Python contracts and adds
no ontology file to this directory. Retrieval, interpretation, ontology delta, and
Phase 4.3-style decision records share an aligned operation number.

Document handling uses a strict `DocumentOverviewRecord` version 3 contract,
owned by `tools/document_evidence/interpreter.py`. One retrieval contains every
ordered page's extracted text, rendered-page hash, neutral visual observations,
uncertainty, and exact page citations. New runs do not produce targeted
`DocumentEvidenceRecord` values. Older overview and targeted-evidence records
remain read-only recovery inputs where a loader still supports them.

`OntologyGroundingProposal`, owned by
`agents/pa/ontology_grounding.py`, is built from the requirement, ontology, and
retrieved typed records together. New proposals use schema version 6 and contain
exactly one model-authored `target_feature`:

```json
{
  "target_feature": {
    "required_process": {
      "process_iri": "<authorized process IRI>",
      "evidence_refs": ["<one or more direct evidence refs>"]
    },
    "desired_state": {
      "statement": {
        "text": "<complete desired product state>",
        "evidence_refs": ["<one or more direct evidence refs>"]
      },
      "state_values": [
        {
          "name": "<unique PA-authored semantic name>",
          "value_ref": {
            "record_ref": "<accepted typed-record path>",
            "field_path": "<JSON Pointer>"
          },
          "evidence_refs": ["<one or more direct evidence refs>"]
        }
      ]
    }
  }
}
```

`state_values` supports zero, one, or multiple entries. One typed record may
supply several values through different field paths. Names are PA-authored and
have no host enum. Each included value has exactly one `value_ref` and one or
more direct evidence refs. PA—not deterministic code—chooses the process,
statement, values, refs, paths, and citations from retrieved evidence.

Validation first produces a transient candidate and provisional ABox. The host
generates `feature_0001`, `currentstate_0001`, and `desiredstate_0001` and
compiles their exact seven type, state-link, `ppr:defines`, and `ppr:realizes`
assertions. It neither writes an accepted proposal nor merges RDF at that
boundary. Only after `TargetFeatureSemanticReview` v2 is complete and the active
consumer's typed prerequisite chain passes does the system persist the accepted
version-7 proposal and merge the compiled assertions. Versions 3, 4, 5, and 6
remain read-only recovery contracts.

Phase 4.2A typed geometry records are owned by
`tools/rgb_d_cad_grounding/preprocessor.py`. `CADMeshRecord` references complete
triangle and facet-normal arrays; `ColoredPointCloudSetRecord` references one
calibrated colored point cloud per camera optical frame. Their generic deltas
contain no RDF assertions and leave correspondence and pose unresolved.

Phase 4.2B1 records are owned by
`tools/rgb_d_cad_grounding/segmenter.py`, `observation_review.py`, and
`diagnostic.py`.
`RGBDSegmentationRecord` references one camera-local `uint16` label mask per
camera and stores neutral candidate handles, candidate count, bounds, centroid,
hashes, and fixed parameters. `RGBDSegmentationStatus` is the compact
status-only UI contract with `idle`, `running`, `ready`, or `failed`, a neutral
candidate count, and unevaluated identity, CAD correspondence, and pose. These
records contain no ontology assertions and express neither matching, context
completion, nor assembly readiness.

`ObservationCandidateReview` version 1 pins the segmentation record, source RGB
hashes, stable candidate crops, exact opaque-handle coverage, visible
descriptions, uncertainty, provider configuration, and response metadata. Its
strict output has no state, CAD, process, or resource assignment fields.

Phase 4.2B2A adds `CADSizeCorrespondenceRecord`, owned by
`tools/rgb_d_cad_grounding/size_correspondence.py`. It binds one exact validated
CAD record to one validated segmentation record and preserves CAD and artifact
hashes, two-dimensional principal-size comparisons, relative errors,
deterministic ranking, and any unique candidate's median center and optical
frame. Its `CAD_correspondence` state is `accepted`, `ambiguous`, or `rejected`;
its `location` state is `available`, `ambiguous`, or `unavailable`; pose remains
`not_evaluated`. The compact `RGBDSegmentationStatus` can show these two states
without exposing coordinates, scores, masks, or thresholds. The record is not
an ontology assertion, complete pose, context-completion assessment, or
assembly-readiness claim.

The implemented Phase 4.2B2 pose increment adds
`CADPoseEstimationRecord`, owned by
`tools/rgb_d_cad_grounding/pose_estimation.py`. It binds one validated size
correspondence to ranked camera-frame registration hypotheses and stores a
translation, rotation matrix, quaternion, and camera-from-CAD transform only
for a clear fit. `pose` is `accepted`, `ambiguous`, or `rejected`; qualified
competing rotations remain in the typed record. The compact status may display
the pose state but never exposes coordinates or transforms.

The frame-conversion increment adds `CameraToRobotCalibrationRecord`,
`RobotFrameLocationRecord`, and `RobotFramePoseRecord`, owned by
`tools/rgb_d_cad_grounding/frame_conversion.py`. The calibration record binds
one caller-approved rigid transform to exact source and target frames, a
validity window, provenance, and a deterministic payload hash.
`RobotFrameLocationRecord` version 2 can hash either neutral segmentation plus
calibration or a legacy correspondence chain and stores the translated selected
candidate center without assigning a semantic source/target role.
`RobotFramePoseRecord` stores the composed
translation and orientation only for an orientation-sensitive consumer. Pose
records remain available, but they are not an active prerequisite for coarse
reachability. A location record's meaning comes from PA assigning its neutral
candidate evidence to `current_state` or `desired_state`.

Implemented pre-RA grounding contracts in
`agents/pa/grounding_contracts.py` cover:

- `GroundingProducerDescriptor`, describing one provider's accepted evidence,
  produced records, prerequisites, availability, and estimated cost;
- `TypedContextBinding` and `ProductContextView`, joining compact ABox meaning
  with validated record refs, hashes, status, frames, validity, and source
  details;
- `EvidencePresentationRecord` version 1, pinning host-only opaque source
  handles to canonical source identities and hashes;
- `AllocationPresentationRecord` version 1, pinning randomized resource and
  shared neutral-candidate presentation orders while allocation reuses the two
  candidate bindings already accepted in `state_values`;
- `ReachabilityCheckRecord` version 3 for simulation, recording the selected
  process, PA-authored state-evidence mapping, explicit provisional resource,
  hash-pinned CAD/support provenance, derived Cartesian targets, request
  fingerprint, exact RobotAgent validation ref, phase results, and final status
  without static `in_workspace` or `in_gripper_reach` claims;
- `PlanOnlyFeasibilityValidationRecord` version 3 for simulation, recording the
  live start pose, EE-to-TCP transform, ordered waypoint roles, configured
  group/link/service, per-phase fractions and MoveIt error codes, and
  `motion_executed: false`;
- `ResourceSelectionRecord` version 4, pinning PA authority, process and
  candidate set, both presentations, both state-evidence assignments, the
  provisional choice, reach evidence, and exact RobotAgent plan-only
  verdict;
- `TargetFeatureSemanticReview` schema version 2, preserving only the structured
  `complete`/`incomplete` verdict and concise gap, never private reasoning;
- `TypedGroundingContract` schema version 6, pinning the accepted v8 proposal,
  selected process, both state IRIs, semantic review, nested evidence, typed
  values, both presentation records, registry/workcell snapshots, two-state
  reachability, endpoint-motion or Cartesian RobotAgent validation, resource selection,
  assignment delta, and final ABox without copying `target_feature`; and
- append-only native PA tool audits, `PAClarification` records, and
  `PAContextGroundingCompletion` version 6.

`agents/ra/context_handoff.py` now covers the Phase 5.1 contracts:

- `SelectedRAAssignmentEnvelope` version 3, the hash-pinned Phase 4 process,
  resource, state-evidence, presentation, workcell, reachability, and validation
  lineage delivered to the exact selected RA;
- `RobotStateSnapshot`, one fresh JSON state response correlated to that
  assignment; and
- `PrimitiveCatalogSnapshot`, the complete ordered primitive-only catalog
  linked to the assignment, resource selection, and matching state record.

Each catalog entry preserves its exact `primitive_symbol` and declares an
operation description, ordered typed parameters and results, invocation
binding, truthful limits, direct evidence, evaluator endpoints, and optional
modeled conditions and effects. Omitted conditions or effects remain
`unmodeled`. No fixed catalog cardinality, composite expansion, or
`primitive_steps` is accepted at this boundary.

`agents/ra/primitive_draft.py` owns the implemented Phase 5.2A
`PrimitiveProgramDraft`. The model-authored portion contains only `draft_status`,
an ordered list of exact catalog symbols, and an unsupported reason. The host
derives structural step indexes and pins the record to the exact completion,
assignment, robot-state snapshot, and primitive-catalog snapshot. It contains
no parameter values, `primitive_steps`, feasibility claim, or execution request.
Its transient model input includes the completion-consistent post-assignment
ontology assertions with the matching TBox/ABox fingerprints and a reconstructed
`target_feature` after the host validates the exact feature/process/resource
chain. The reconstruction includes the requirement and host identifiers plus a
bounded projection of each hash-pinned JSON-Pointer value. It has no top-level
`task` section. Neither the target feature nor those resolved values are copied
into the model-authored `PrimitiveProgramDraft`.

The active system exposes approved neutral evidence to PA. Once PA has grounded
both feature states, it explicitly chooses state values and a provisional
resource through `check_reachability`. In simulation that verifier derives the
gear pick and shaft-centered place targets from pinned geometry and requests
live Cartesian planning from the exact chosen RobotAgent; it never chooses or
substitutes a resource. Historical and physical-mode version 2 records remain
read-only/compatible.

`GroundingNextAction`, `GroundingActionAttempt`, `GroundingSession` version 2,
targeted document evidence, pose-linked `ResourceSelectionRecord` version 1,
older ontology proposals, `TypedGroundingContract` versions 2/3/4, resource
selection versions 1/2, and completion versions 2/3/4 describe historical
interactions only. Supported loaders may
verify them for read-only recovery; new production runs do not write or migrate
them. New Phase 5 handoff requires completion version 6. Proposal v7,
selection v1-v3, completion v2-v5, feasibility-validation v1, and prior RA
envelopes remain read-only.

Planned downstream contracts cover:

- versioned `CompositionContextBundle` records with the grounded product outcome,
  ABox, binding,
  selected-resource, and catalog fingerprints
- deduplicated `MissingContextBatch`, PA batch response, and progress or
  no-progress decision records
- fully bound `primitive_steps`, primitive-level state checks and validation,
  rejected candidate revision, and accepted candidate handoff
- post-process state observation and update

These downstream names remain planned boundaries. The current framework has no
separate `TaskTransitionContract`: Phase 5.2A consumes the validated
post-assignment ontology projection, `ResourceSelectionRecord`, Phase 5.1
snapshots, and minimum hash-pinned typed evidence directly. Phase 5.2A produces
only the unbound structural draft; it produces no RA resource ABox, composition
bundle, missing-context batch, fully bound candidate, validation result, or
execution command.

`TargetFeatureGeometryRecord` is not part of the architecture. PA assigns
neutral approved candidates to current and desired feature states. Numeric
locations are derived only for an activated verifier and remain pinned typed
evidence; absent pose, tolerance, or execution parameters are not inferred.

Phase 0 intentionally defines no JSON, YAML, or Python schema.

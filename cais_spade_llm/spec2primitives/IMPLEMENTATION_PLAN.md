# Spec2Primitives implementation plan

## Plan-maintenance rule

Every authorized increment updates this file in the same change. The update must
identify what is implemented, the current boundary, the next separately
authorized increment, and behavior that remains deferred. Status describes
verified repository behavior, not an aspirational architecture.

## Implemented through Phase 4.4

The active ontology-driven resource-grounding path is:

```text
requirement + ontology
→ retrieve only currently relevant approved evidence
→ propose_grounding commits a provisional semantic task ABox
→ join the task ABox with the predefined Workcell ABox
→ derive ResourceAssignmentNeed(RobotFramePoseRecord, world)
→ resolve the typed provider prerequisites dynamically
→ evaluate manifest-backed coarse reach in fixed registry order
→ host commits the selected processExecution assignment
→ persist completion against the post-assignment ABox
```

ProductAgent returns only one semantic action:

- `retrieve(source_ref)`
- `inspect(source_ref, question)`
- `propose_grounding`
- `ask_user(question)`
- `incomplete(reason)`

There is no `fill_ontology` compatibility alias. `propose_grounding` may commit
the evidence-backed specification, feature, and `assembly realizes feature`
facts without ending the interaction. The host, not ProductAgent, derives the
remaining resource-assignment need from the accepted ABox and current typed
records.

The predefined Workcell ABox records only that exact resources `xarm6` and
`ur5e` are broadly `capableOf` the exact process `assembly`. It does not claim
that either robot can reach the current Medium Gear, and it contains no
primitive-level `capableOf` assertions or primitive sequence.

When the semantic join has candidates but no accepted same-interaction
`RobotFramePoseRecord` in `world`, the host derives a non-persisted
`ResourceAssignmentNeed`. Provider descriptors map that typed output and its
prerequisites to approved sources. The path may therefore obtain
`Gear_Medium.STL`, a fresh RGB-D observation, a camera-frame pose, and injected
`CameraToWorldCalibrationRuntime`, but it never selects a camera through a
Medium Gear, `Gear_Medium`, `xarm6`, or modality-specific conditional. No
semantic candidates means no camera request; an already accepted world-frame
pose means no repeated capture. Missing, ambiguous, stale, wrong-frame, or
hash-invalid evidence reopens the unresolved need at a new evidence revision.
Unchanged derived inputs are evaluated only once: a rejected or ambiguous
derived result returns control to PA, while a newer CAD or RGB-D record
invalidates and regenerates only its downstream records. Manifest, authority,
configuration, and selection-persistence failures remain terminal fail-closed
errors rather than triggering another camera request.

The calibration boundary is injectable and production refuses conversion when
no approved calibration is supplied. The current default UI composition does
not install an approved `CameraToWorldCalibrationRuntime`, so that deployment
correctly stops before frame conversion until one is configured.

After revalidating the pinned manifests, coarse reach uses only
`supports_manipulator_pick_place`, `workspace_bounds`, and `gripper_reach`.
Candidate order is the immutable resource-registry order: `xarm6`, then `ur5e`.
The first reachable resource is recorded in `ResourceSelectionRecord`, including
the exact JID, source and evidence fingerprints, execution mode, per-candidate
verdicts, and selection policy. Numeric poses, workspace values, JIDs,
credentials, and controller details stay outside RDF.

Only the host may add the functional PPR execution slice:

```text
specification ppr:hasProcessExecution processExecution
processExecution ppr:runsProcess assembly
processExecution ppr:runsOnResource selected resource
```

ProductAgent-authored resource or execution assertions are rejected atomically.
Completion remains unavailable until the exact originating ontology proposal is
linked to a valid host assignment and the post-assignment ABox fingerprint. The
completion bundle also pins and reloads the exact `ResourceSelectionRecord` and
the four-assertion `resource_grounding_host` delta, whose assertions all cite
that one selection record.

## Intentionally unchanged and deferred

- ProductAgent and RobotAgent remain shared, read-only runtime authorities.
- `SystemBridge` and the public UI-to-runtime surface are unchanged.
- Detailed resource configuration remains authoritative in the shared JSON
  manifests and is not duplicated in OWL.
- The identity-only resource registry remains a two-triple projection.
- There is no separate `TaskTransitionContract`.
- Primitive implementations and catalogs remain outside the ontology; no
  primitive receives `ppr:capableOf`.
- Primitive composition, IK, collision checking, execution, and realized-outcome
  evaluation remain deferred.

## Next planned boundary

Phase 5 is the next separately authorized increment. It starts from the exact
resource selected in Phase 4.4 and adds selected-JID RA handoff, retrieval and
fingerprinting of that RA's complete typed primitive-only catalog, and
structured `PrimitiveProgramDraft`/`primitive_steps` composition. It must not
introduce a second task-transition contract, infer a primitive sequence from
the ontology, or expose primitive implementations through `ppr:capableOf`.

Robot-local validation and revision, simulation execution, and physical
execution remain later, separately authorized increments.

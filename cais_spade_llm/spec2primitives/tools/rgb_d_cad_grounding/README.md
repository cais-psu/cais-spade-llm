# RGB-D and CAD grounding

The demand-driven capture provider writes approved RGB/depth/calibration evidence into one pinned observation bundle. Preprocessing verifies source hashes, converts CAD millimetres to metres, and deprojects valid depth into calibrated camera-local colored points.

`segmenter.py` uses uniform fixed support-plane/foreground and connected-region rules for every view. It records neutral candidates, bounds, centroids, masks and provenance. These rules can miss or merge objects; no general segmentation accuracy is claimed.

`observation_review.py` creates hash-pinned candidate crops and submits all views/crops through configured observation VLM settings. `ObservationCandidateReview` 1 covers exact candidates with morphology and uncertainty, without state/CAD/process/resource assignments. `ObservationPresentationRecord` randomizes view/candidate handles and ordering before the VLM and PA receive them. Canonical metadata remains internal for exact geometry/provenance joins.

`size_correspondence.py` exposes active `measure_segmented_candidates_against_cad`, writing `CADSizeCorrespondenceRecord` 3 with every evaluated candidate's dimensions. It does not rank/select a winner or establish task role. Canonical record order is retained internally; model presentation order is independently randomized.

`candidate_layout.py` returns positions, displacements, distances and collinearity for any two or more PA-selected same-view candidates. It does not emit a built-in destination or relationship verdict. Cross-camera fusion is not implemented.

`frame_conversion.py` validates approved calibration against exact source/target frames and observation time, then writes a neutral `RobotFrameLocationRecord` 2 for the selected candidate. Real sensor names and transforms stay internal. A location's semantic role comes from reviewed PA state values, not its camera or tool name.

Historical selecting size-correspondence, `pose_estimation.py`, and orientation-sensitive frame-conversion diagnostics remain explicit separate consumers. A clear pose fit and ambiguous symmetric alternatives retain their existing statuses. They are not prerequisites for current arm assignment.

Phase 4 review checks part identities separately from task roles, destination bindings, current attachment and intended relations. Figures alone cannot prove installation, and a small dimension advantage cannot resolve several plausible destinations. Existing source uncertainty is preserved; missing meaning prevents completion.

PA's required arm check covers every reviewed current/destination coordinate using live MoveIt position planning. It does not call Cartesian planning, infer a final insertion pose, or execute motion. Tools never access world/SDF contents, spawn poses, entity state, detector answers or evaluator labels. Internal calibration is allowed authority, not a recognition answer.

Standalone diagnostics keep compact status and no operator segmentation controls. Production PA retrieval is connected to these evidence tools. See [bias experiments](../../BIAS_VALIDATION.md), [schemas](../../schemas/README.md), and [tests](../../tests/README.md).

An observed object location or CAD-local mesh is not a computed end-effector target or complete placement geometry. Phase 4 preserves these observed references and desired assembly relationships. Phase 5 reports gaps, then PA can request pose estimation, calibrated conversion and neutral feature/support measurements. Symmetric hypotheses preserve uncertainty; only demonstrably invariant quantities may be supplied separately. [Validation and revision](../../VALIDATION_AND_REVISION.md) defines the measured context, strict selected calculations and motion scope.

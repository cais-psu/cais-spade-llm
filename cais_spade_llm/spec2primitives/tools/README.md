# Tools

Controlled retrieval, observation, and grounding components belong here. They
are tools used by the Spec2Primitives workflow, not agents.

- `exact_ref_resolver.py` serves approved product document and CAD evidence;
  approved CAD summaries include a source hash for later integrity checks.
- `observation_context.py` validates and stores fixture, replay, and live RGB-D
  bundles.
- `document_evidence/` validates any registered approved PDF, prepares a
  content-addressed ontology-neutral overview cache, and returns one
  assertion-free version-2 record containing every ordered page. Historical
  targeted-evidence files are not produced by the active retrieval path. Its
  diagnostic keeps document evidence, a transient PA proposal, and accepted
  assertions separate.
- `rgb_d_cad_grounding/` captures one fresh live Gazebo RGB-D bundle only when
  invoked, implements Phase 4.2A preprocessing, and implements Phase 4.2B1
  minimal camera-local segmentation, Phase 4.2B2A size-only association, and
  location-oriented frame conversion for loose source candidates. Generalized
  pose estimation remains available for an orientation-sensitive consumer.
  It atomically persists complete approved CAD meshes, calibrated colored point
  clouds, compact candidate records, label masks, and deterministic
  `CADSizeCorrespondenceRecord`, `CADPoseEstimationRecord`,
  `CameraToRobotCalibrationRecord`, `RobotFrameLocationRecord`, and optional
  `RobotFramePoseRecord` results. Pose
  estimation compares only one exact caller-supplied CAD record, returns a
  complete camera-from-CAD transform only for a clear fit, and preserves
  candidate or rotation ambiguity. Frame conversion consumes only a separate
  caller-approved calibration record.

The live observation provider creates no background subscription. Its catalog
entry reports availability only; it supplies an observation only after PA
calls the approved `retrieve` handle. The automatic observation entrypoint performs
capture → validation → preprocessing → segmentation only when called. The UI
polls compact status and exposes no processing controls. CAD preprocessing stays
separate; association is invoked only after one exact approved CAD record and
the required typed records exist. Cross-camera fusion is outside the completed
Phase 4 boundary because the active location-based path selects and converts
one exact camera frame. RA-owned Phase 5 composition and validation require
later separately authorized work.

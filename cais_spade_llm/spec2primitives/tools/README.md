# Tools

Controlled retrieval, observation, and future perception components belong
here. They are tools used by the Spec2Primitives workflow, not agents.

- `exact_ref_resolver.py` serves approved product document and CAD evidence;
  approved CAD summaries include a source hash for later integrity checks.
- `observation_context.py` validates and stores fixture, replay, and live RGB-D
  bundles.
- `document_evidence/` renders approved PDF pages, invokes an injected OpenAI
  vision boundary, compiles a generic evidence-backed delta, and runs a separate
  diagnostic ABox.
- `rgb_d_cad_grounding/` captures one fresh live Gazebo RGB-D bundle only when
  invoked, implements Phase 4.2A preprocessing, and implements Phase 4.2B1
  minimal camera-local segmentation, Phase 4.2B2A size-only association, and
  simple generalized camera-frame pose estimation and frame conversion for
  loose source candidates.
  It atomically persists complete approved CAD meshes, calibrated colored point
  clouds, compact candidate records, label masks, and deterministic
  `CADSizeCorrespondenceRecord`, `CADPoseEstimationRecord`,
  `CameraToRobotCalibrationRecord`, and `RobotFramePoseRecord` results. Pose
  estimation compares only one exact caller-supplied CAD record, returns a
  complete camera-from-CAD transform only for a clear fit, and preserves
  candidate or rotation ambiguity. Frame conversion consumes only a separate
  caller-approved calibration record.

The live observation provider creates no background subscription and supplies
no observation to PA or RA. The automatic observation entrypoint performs
capture → validation → preprocessing → segmentation only when called. The UI
polls compact status and exposes no processing controls. CAD preprocessing stays
separate; the association entrypoint is invoked only by a future caller that
already selected one CAD record. Cross-camera transformation, PA integration,
and assessment still require later separately authorized work.

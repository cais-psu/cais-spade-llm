# Phase 4.2 preprocessing through location and pose frame conversion

`gazebo_observation_provider.py` implements the Phase 1.2 demand-driven live
capture boundary. `capture_gazebo_observation(...)` creates request-owned ROS2
subscriptions only for the duration of one explicit call, captures the exact
`cam_mk3`, `cam_mk4_1`, `cam_mk4_2`, and `cam_assembly` RGB-D evidence, and
writes one existing Phase 1.1 `ObservationBundle` with evidence label `live`.

Every successful bundle contains one lossless `<camera>_rgb.png` image and one
original metric `<camera>_depth_m.npy` array for each camera. The caller can
inspect the PNG files under
`contexts/<interaction_identifier>/products/observations/<observation_ref>/`.

`preprocessor.py` loads only exact approved binary STL refs, verifies the hash
bound to the served ref, converts every triangle vertex from millimetres to
metres, and atomically persists compressed triangles and facet normals with a
typed `CADMeshRecord`. It also reloads a complete validated four-camera bundle
and deprojects every finite positive metric-depth pixel with that camera's
intrinsics and supported distortion model. Each compressed point-cloud artifact
retains full valid-resolution `points_m`, `colors_rgb`, and `pixels_uv` arrays
in its independent optical frame.

`segmenter.py` accepts only an intact Phase 4.2A observation record. It verifies
every referenced point-cloud hash and array contract, then uses deterministic
fixed internal parameters. Every view applies the same neutral one-sided
dominant-plane filter and retains only camera-side foreground points before
depth-connected region extraction. If no reliable plane is available, that
view produces no candidates. Each successful call atomically persists one compact
`RGBDSegmentationRecord` and four `uint16` label masks. Zero candidates are
recorded as unresolved; identity, `CAD_correspondence`, pose, and cross-camera
fusion remain `not_evaluated`.

`observation_review.py` creates one stable hash-pinned crop for every valid
candidate and submits the source views and crops to the configured observation
VLM. The strict `ObservationCandidateReview` version 1 output covers the exact
opaque handles once and stores only visible descriptions and uncertainty. It
cannot add `current_state`, `desired_state`, CAD, process, or resource decisions.

`size_correspondence.py` accepts one validated segmentation record and one
already-preprocessed exact approved CAD record. It revalidates their paths,
hashes, arrays, bounds, masks, frames, summaries, and counts; reconstructs each
candidate's camera-local points; and compares the two largest principal
dimensions with the two largest CAD dimensions. Both relative errors must be at
most 15 percent, and the next reliable candidate must be at least ten percentage
points worse. Similar valid sizes are ambiguous, no valid size is rejected, and
an image-boundary candidate is treated as unreliable partial visibility.

Each successful call atomically persists one `CADSizeCorrespondenceRecord` with
the exact CAD and segmentation provenance, deterministic ranked candidates,
dimension errors, and any selected median center in its camera optical frame.
The result reports only `CAD_correspondence` and camera-frame `location` status.
Rotation, complete pose, cross-camera fusion, and robot-frame conversion remain
`not_evaluated`. The tool never searches the CAD inventory or loads more than
the exact CAD record supplied by its caller.

`pose_estimation.py` consumes one intact `CADSizeCorrespondenceRecord`,
revalidates its complete CAD, segmentation, point-cloud, mask, and hash chain,
and considers only size-plausible loose `source` candidates. It deterministically
samples the CAD surface, creates 24 principal-axis orientation hypotheses, and
refines each with trimmed point-to-point ICP using NumPy and SciPy. A clear fit
persists translation, rotation matrix, quaternion, and the complete
camera-from-CAD transform; competing candidates or rotations remain
`ambiguous`, and weak fits are `rejected`.

Each successful call atomically persists one `CADPoseEstimationRecord` with the
exact source-correspondence hash, complete upstream provenance, fixed
parameters, all ranked and qualified hypotheses, and any selected camera
optical frame. The result is not a robot-frame pose or pick point. It does not
search the CAD inventory, use a learned detector, or read simulator identity,
configured pose, detector response, or evaluator data.

`frame_conversion.py` atomically records one injected
`CameraToRobotCalibrationRecord` with exact source and target frames, validity,
approved-source provenance, derived rotation representations, and a
deterministic payload hash. For allocation, it checks calibration at the
originating observation timestamp, requires the declared target frame, and
translates only the PA-selected neutral candidate into that frame.

An accepted location conversion atomically persists one
`RobotFrameLocationRecord` version 2 with the translated 3D location and exact
input hashes. A separate orientation-sensitive consumer may activate the
retained `CADPoseEstimationRecord` and `RobotFramePoseRecord` path. Ambiguous or
rejected inputs preserve their state without accepted robot-frame coordinates.
`RobotFrameLocationRecord` and `RobotFramePoseRecord` have no embedded semantic
source/target role. PA's feature-state assignment provides their meaning. These
paths perform no robot selection, RA call, planning, or execution and
read no world, spawn, entity-state, detector, or evaluator input.

`diagnostic.py` exposes the injected `ObservationCaptureRuntime`, retains the
controlled Phase 4.2A preprocessing diagnostic, and adds
`run_automatic_rgbd_segmentation_pipeline(...)`. The automatic entrypoint has no
operator processing parameters and performs one capture → validation →
observation preprocessing → segmentation sequence. CAD is not loaded by this
path. The production capture adapter delegates only to
`capture_gazebo_observation(...)`. Every geometry operation is exclusive and
atomic under `products/grounding/rgb_d_cad_grounding/`; Phase 4.2A generic
deltas contain no RDF assertions. `run_cad_size_association_pipeline(...)`
updates only the compact read-only status after a controlled caller supplies
the required records. `run_cad_pose_estimation_pipeline(...)` similarly exposes
only `CAD_correspondence`, `location`, and `pose` states.
The frame-conversion diagnostics expose only compact status and never return
coordinates through the UI status boundary.

The tools have no background stream, general identity recognition,
cross-camera fusion, PA-loop connection, planning, or
robot execution. The UI reads only the compact automatic status
(`idle`, `running`, `ready`, or `failed`), candidate counts, CAD-correspondence
state, location state, pose state, and robot-frame conversion state; it exposes
no CAD, coordinate,
score, timeout, camera-role, threshold, mask, or artifact controls. Capture,
preprocessing, segmentation, size association, pose estimation, and frame
conversion are
supporting infrastructure, not the research contribution, and cannot authorize
context completion.

These components expose neutral candidates, not semantic source/target labels.
The architecture has no `TargetFeatureGeometryRecord`. PA decides which
approved evidence to retrieve and which candidate supports each feature state;
the frame converter derives a location only after a verifier requests that
selected handle. This README prescribes no document, RGB-D, CAD, camera, or
resource order.

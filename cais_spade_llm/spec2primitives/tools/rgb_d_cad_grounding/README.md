# Phase 4.2A preprocessing through Phase 4.2B2A size association

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
fixed internal parameters. `cam_mk3`, `cam_mk4_1`, and `cam_mk4_2` remove a
reliable dominant support plane before depth-connected loose-region extraction.
`cam_assembly` retains its dominant surface and segments the assembly-plate area
in its own optical frame. Each successful call atomically persists one compact
`RGBDSegmentationRecord` and four `uint16` label masks. Zero candidates are
recorded as unresolved; identity, `CAD_correspondence`, pose, and cross-camera
fusion remain `not_evaluated`.

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
the required records.

The tools have no background stream, general identity recognition,
cross-camera fusion, rotation or complete pose estimation, PA-loop connection,
planning, or robot execution. The UI reads only the compact automatic status
(`idle`, `running`, `ready`, or `failed`), candidate counts, CAD-correspondence
state, location state, and `pose: not_evaluated`; it exposes no CAD, coordinate,
score, timeout, camera-role, threshold, mask, or artifact controls. Capture,
preprocessing, segmentation, and size association are supporting
infrastructure, not the research contribution, and cannot authorize context
completion.

# RGB-D/CAD grounding tool

`gazebo_observation_provider.py` implements the Phase 1.2 demand-driven live
capture boundary. `capture_gazebo_observation(...)` creates request-owned ROS2
subscriptions only for the duration of one explicit call, captures the exact
`cam_mk3`, `cam_mk4_1`, `cam_mk4_2`, and `cam_assembly` RGB-D evidence, and
writes one existing Phase 1.1 `ObservationBundle` with evidence label `live`.

Every successful bundle contains one lossless `<camera>_rgb.png` image and one
original metric `<camera>_depth_m.npy` array for each camera. The caller can
inspect the PNG files under
`contexts/<interaction_identifier>/products/observations/<observation_ref>/`.

The provider has no background stream, cached latest observation, automatic
agent delivery, UI behavior, segmentation, recognition, grounding, PA, or RA
connection. Phase 2 may let PA request this operation dynamically through its
future Spec2Primitives-owned adapter. RGB segmentation, depth geometry, and CAD
registration remain future work.

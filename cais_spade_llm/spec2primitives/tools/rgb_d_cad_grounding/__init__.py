"""Phase 4.2 preprocessing, segmentation, correspondence, and pose boundaries."""

from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.diagnostic import (
    LiveGazeboObservationCaptureRuntime,
    ObservationCaptureRuntime,
    read_rgbd_segmentation_status,
    run_automatic_rgbd_segmentation_pipeline,
    run_cad_pose_estimation_pipeline,
    run_cad_size_association_pipeline,
    run_robot_frame_pose_conversion_pipeline,
    run_rgbd_cad_preprocessing_diagnostic,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.frame_conversion import (
    CameraToRobotCalibrationError,
    CameraToRobotCalibrationResult,
    RobotFrameConversionError,
    RobotFramePoseResult,
    record_camera_to_robot_calibration,
    transform_camera_pose_to_robot_frame,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.pose_estimation import (
    CADPoseEstimationError,
    CADPoseEstimationResult,
    estimate_camera_frame_pose,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.preprocessor import (
    GeometryPreprocessingError,
    GeometryPreprocessingResult,
    preprocess_served_geometry,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.segmenter import (
    RGBDSegmentationError,
    RGBDSegmentationResult,
    segment_preprocessed_observation,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.size_correspondence import (
    CADSizeAssociationError,
    CADSizeAssociationResult,
    associate_segmented_candidate_by_size,
)

__all__ = [
    "CADSizeAssociationError",
    "CADSizeAssociationResult",
    "CADPoseEstimationError",
    "CADPoseEstimationResult",
    "CameraToRobotCalibrationError",
    "CameraToRobotCalibrationResult",
    "GeometryPreprocessingError",
    "GeometryPreprocessingResult",
    "LiveGazeboObservationCaptureRuntime",
    "ObservationCaptureRuntime",
    "RGBDSegmentationError",
    "RGBDSegmentationResult",
    "RobotFrameConversionError",
    "RobotFramePoseResult",
    "associate_segmented_candidate_by_size",
    "estimate_camera_frame_pose",
    "preprocess_served_geometry",
    "read_rgbd_segmentation_status",
    "record_camera_to_robot_calibration",
    "run_automatic_rgbd_segmentation_pipeline",
    "run_cad_pose_estimation_pipeline",
    "run_cad_size_association_pipeline",
    "run_robot_frame_pose_conversion_pipeline",
    "run_rgbd_cad_preprocessing_diagnostic",
    "segment_preprocessed_observation",
    "transform_camera_pose_to_robot_frame",
]

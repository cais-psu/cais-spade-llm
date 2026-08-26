"""Phase 4.2 preprocessing, segmentation, and size-association boundaries."""

from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.diagnostic import (
    LiveGazeboObservationCaptureRuntime,
    ObservationCaptureRuntime,
    read_rgbd_segmentation_status,
    run_automatic_rgbd_segmentation_pipeline,
    run_cad_size_association_pipeline,
    run_rgbd_cad_preprocessing_diagnostic,
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
    "GeometryPreprocessingError",
    "GeometryPreprocessingResult",
    "LiveGazeboObservationCaptureRuntime",
    "ObservationCaptureRuntime",
    "RGBDSegmentationError",
    "RGBDSegmentationResult",
    "associate_segmented_candidate_by_size",
    "preprocess_served_geometry",
    "read_rgbd_segmentation_status",
    "run_automatic_rgbd_segmentation_pipeline",
    "run_cad_size_association_pipeline",
    "run_rgbd_cad_preprocessing_diagnostic",
    "segment_preprocessed_observation",
]

"""Compose the runtime values required by the Spec2Primitives PA UI."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cais_spade_llm.spec2primitives.adapters.dual_gazebo import DualGazeboRuntime
from cais_spade_llm.spec2primitives.agents.pa.context_grounding import (
    PAOntologyConfig,
    ProductContextGroundingRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa.production_grounding import (
    CameraToWorldCalibrationRuntime,
    ProductionProductContextGroundingRuntime,
)
from cais_spade_llm.spec2primitives.config import (
    DEFAULT_GAZEBO_CAMERA_TO_WORLD_CALIBRATION_PATH,
    ModelRuntimeConfig,
    load_camera_to_world_calibration_runtime,
    load_model_runtime_config,
)
from cais_spade_llm.spec2primitives.ontology import (
    OntologyContextError,
    TBoxSnapshot,
)
from cais_spade_llm.spec2primitives.tools.document_evidence import (
    DocumentVisionRuntime,
    OpenAIDocumentVisionRuntime,
    document_overview_cache_status,
)
from cais_spade_llm.spec2primitives.tools.exact_ref_resolver import (
    approved_document_metadata,
    approved_document_refs,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    LiveGazeboObservationCaptureRuntime,
    ObservationCaptureRuntime,
)

SPEC2PRIMITIVES_CONTEXTS_ROOT = Path(__file__).resolve().parents[1] / "contexts"
DEFAULT_SPEC2PRIMITIVES_PPR_TBOX_PATH = (
    Path(__file__).resolve().parents[1] / "ontology" / "spec2primitives_ppr_tbox.owl"
)
DEFAULT_SPEC2PRIMITIVES_PPR_NAMESPACE = "http://PAonto.com#"


@dataclass(frozen=True)
class Spec2PrimitivesUIRuntime:
    """Hold the isolated UI runtime authorities and interaction root."""

    dual_gazebo: DualGazeboRuntime
    product_agent: ProductAgentContextRuntime
    contexts_root: Path
    ontology_config: PAOntologyConfig | None = None
    grounding_runtime: ProductContextGroundingRuntime | None = None
    tbox: TBoxSnapshot | None = None
    model_config: ModelRuntimeConfig | None = None
    document_vision_runtime: DocumentVisionRuntime | None = None
    document_diagnostic_unavailable_reason: str | None = None
    document_source_status: tuple[dict[str, object], ...] = ()
    observation_capture_runtime: ObservationCaptureRuntime | None = None
    camera_to_world_calibration_runtime: CameraToWorldCalibrationRuntime | None = None
    camera_to_world_calibration_unavailable_reason: str | None = None


class _UnavailableProductAgentRuntime:
    """Fail closed when the package model configuration cannot be loaded."""

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[
            [str, Mapping[str, object]], Awaitable[Mapping[str, object]]
        ]
        | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        raise RuntimeError("Spec2Primitives model configuration is unavailable.")


def create_spec2primitives_ui_runtime(
    dual_gazebo: DualGazeboRuntime,
) -> Spec2PrimitivesUIRuntime:
    """Create one application-owned runtime for the Spec2Primitives page."""
    from cais_spade_llm.spec2primitives.agents.pa.product_agent_runtime import (
        create_product_agent_context_runtime,
    )

    model_config: ModelRuntimeConfig | None = None
    ontology_config: PAOntologyConfig | None = None
    grounding_runtime: ProductContextGroundingRuntime | None = None
    vision_runtime: DocumentVisionRuntime | None = None
    tbox: TBoxSnapshot | None = None
    unavailable_reason: str | None = None
    calibration_runtime: CameraToWorldCalibrationRuntime | None = None
    calibration_unavailable_reason: str | None = None
    product_agent: ProductAgentContextRuntime = _UnavailableProductAgentRuntime()

    calibration_path_override = os.environ.get(
        "SPEC2PRIMITIVES_CAMERA_TO_WORLD_CALIBRATION_PATH"
    )
    calibration_path = (
        Path(calibration_path_override)
        if calibration_path_override
        else DEFAULT_GAZEBO_CAMERA_TO_WORLD_CALIBRATION_PATH
    )
    try:
        calibration_runtime = load_camera_to_world_calibration_runtime(
            calibration_path
        )
    except (OSError, ValueError) as exc:
        calibration_unavailable_reason = (
            "Approved camera-to-world calibration manifest is invalid: "
            f"{type(exc).__name__}: {exc}"
        )

    try:
        model_config = load_model_runtime_config()
    except (OSError, ValueError) as exc:
        unavailable_reason = f"Model configuration is invalid: {type(exc).__name__}: {exc}"
    else:
        product_agent = create_product_agent_context_runtime(
            model=model_config.product_agent_llm.model
        )
    document_source_status = _document_startup_status(model_config)

    tbox_path_override = os.environ.get("SPEC2PRIMITIVES_PPR_TBOX_PATH")
    ppr_namespace_override = os.environ.get("SPEC2PRIMITIVES_PPR_NAMESPACE")
    if unavailable_reason is None:
        if bool(tbox_path_override) != bool(ppr_namespace_override):
            unavailable_reason = (
                "Set both SPEC2PRIMITIVES_PPR_TBOX_PATH and "
                "SPEC2PRIMITIVES_PPR_NAMESPACE to override the project TBox."
            )
        else:
            tbox_path = (
                Path(tbox_path_override)
                if tbox_path_override
                else DEFAULT_SPEC2PRIMITIVES_PPR_TBOX_PATH
            )
            ppr_namespace = (
                ppr_namespace_override
                if ppr_namespace_override
                else DEFAULT_SPEC2PRIMITIVES_PPR_NAMESPACE
            )
            ontology_config = PAOntologyConfig(Path(tbox_path), ppr_namespace)
            try:
                tbox = ontology_config.load_tbox()
            except (OSError, OntologyContextError) as exc:
                unavailable_reason = f"Authoritative TBox is invalid: {type(exc).__name__}: {exc}"

    if unavailable_reason is None and not os.environ.get("OPENAI_API_KEY"):
        unavailable_reason = (
            "Set OPENAI_API_KEY to enable production grounding and the "
            "OpenAI document diagnostic."
        )
    if unavailable_reason is None and model_config is not None and tbox is not None:
        vision_runtime = OpenAIDocumentVisionRuntime(model_config.document_vlm)
        grounding_runtime = ProductionProductContextGroundingRuntime(
            tbox=tbox,
            document_config=model_config.document_vlm,
            document_vision_runtime=vision_runtime,
            camera_to_world_calibration_runtime=calibration_runtime,
            camera_to_world_calibration_unavailable_reason=(
                calibration_unavailable_reason
            ),
        )

    return Spec2PrimitivesUIRuntime(
        dual_gazebo=dual_gazebo,
        product_agent=product_agent,
        contexts_root=SPEC2PRIMITIVES_CONTEXTS_ROOT,
        ontology_config=ontology_config,
        grounding_runtime=grounding_runtime,
        tbox=tbox,
        model_config=model_config,
        document_vision_runtime=vision_runtime,
        document_diagnostic_unavailable_reason=unavailable_reason,
        document_source_status=document_source_status,
        observation_capture_runtime=LiveGazeboObservationCaptureRuntime(),
        camera_to_world_calibration_runtime=calibration_runtime,
        camera_to_world_calibration_unavailable_reason=(
            calibration_unavailable_reason
        ),
    )


def _document_startup_status(
    model_config: ModelRuntimeConfig | None,
) -> tuple[dict[str, object], ...]:
    """Validate registered PDFs and inspect caches without model requests."""
    try:
        context_refs = approved_document_refs()
    except (OSError, ValueError) as exc:
        return (
            {
                "context_ref": None,
                "source_status": "invalid",
                "overview_status": "unavailable",
                "record_path": None,
                "rejection": f"{type(exc).__name__}: {exc}",
            },
        )
    statuses: list[dict[str, object]] = []
    for context_ref in context_refs:
        if model_config is not None:
            statuses.append(
                document_overview_cache_status(
                    context_ref,
                    cache_root=SPEC2PRIMITIVES_CONTEXTS_ROOT / "source_cache",
                    config=model_config.document_vlm,
                )
            )
            continue
        try:
            approved_document_metadata(context_ref)
        except (OSError, ValueError) as exc:
            rejection: object = f"{type(exc).__name__}: {exc}"
            source_status = "invalid"
        else:
            rejection = None
            source_status = "valid"
        statuses.append(
            {
                "context_ref": context_ref,
                "source_status": source_status,
                "overview_status": "unavailable",
                "record_path": None,
                "rejection": rejection,
            }
        )
    return tuple(statuses)

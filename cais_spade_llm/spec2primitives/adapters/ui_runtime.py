"""Compose the runtime values required by the Spec2Primitives PA UI."""

from __future__ import annotations

import os
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
from cais_spade_llm.spec2primitives.config import (
    ModelRuntimeConfig,
    load_model_runtime_config,
)
from cais_spade_llm.spec2primitives.ontology import OntologyContextError
from cais_spade_llm.spec2primitives.tools.document_evidence import (
    DocumentVisionRuntime,
    OpenAIDocumentVisionRuntime,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    LiveGazeboObservationCaptureRuntime,
    ObservationCaptureRuntime,
)

SPEC2PRIMITIVES_CONTEXTS_ROOT = Path(__file__).resolve().parents[1] / "contexts"


@dataclass(frozen=True)
class Spec2PrimitivesUIRuntime:
    """Hold the isolated UI runtime authorities and interaction root."""

    dual_gazebo: DualGazeboRuntime
    product_agent: ProductAgentContextRuntime
    contexts_root: Path
    ontology_config: PAOntologyConfig | None = None
    grounding_runtime: ProductContextGroundingRuntime | None = None
    model_config: ModelRuntimeConfig | None = None
    document_vision_runtime: DocumentVisionRuntime | None = None
    document_diagnostic_unavailable_reason: str | None = None
    observation_capture_runtime: ObservationCaptureRuntime | None = None


class _UnavailableProductAgentRuntime:
    """Fail closed when the package model configuration cannot be loaded."""

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        del prompt, response_format
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
    vision_runtime: DocumentVisionRuntime | None = None
    unavailable_reason: str | None = None
    product_agent: ProductAgentContextRuntime = _UnavailableProductAgentRuntime()

    try:
        model_config = load_model_runtime_config()
    except (OSError, ValueError) as exc:
        unavailable_reason = f"Model configuration is invalid: {type(exc).__name__}: {exc}"
    else:
        product_agent = create_product_agent_context_runtime(
            model=model_config.product_agent_llm.model
        )

    tbox_path = os.environ.get("SPEC2PRIMITIVES_PPR_TBOX_PATH")
    ppr_namespace = os.environ.get("SPEC2PRIMITIVES_PPR_NAMESPACE")
    if unavailable_reason is None:
        if not tbox_path or not ppr_namespace:
            unavailable_reason = (
                "Set SPEC2PRIMITIVES_PPR_TBOX_PATH and "
                "SPEC2PRIMITIVES_PPR_NAMESPACE to an authoritative schema-only TBox."
            )
        else:
            ontology_config = PAOntologyConfig(Path(tbox_path), ppr_namespace)
            try:
                ontology_config.load_tbox()
            except (OSError, OntologyContextError) as exc:
                unavailable_reason = f"Authoritative TBox is invalid: {type(exc).__name__}: {exc}"

    if unavailable_reason is None and not os.environ.get("OPENAI_API_KEY"):
        unavailable_reason = "Set OPENAI_API_KEY to enable the OpenAI document diagnostic."
    if unavailable_reason is None and model_config is not None:
        vision_runtime = OpenAIDocumentVisionRuntime(model_config.document_vlm)

    return Spec2PrimitivesUIRuntime(
        dual_gazebo=dual_gazebo,
        product_agent=product_agent,
        contexts_root=SPEC2PRIMITIVES_CONTEXTS_ROOT,
        ontology_config=ontology_config,
        model_config=model_config,
        document_vision_runtime=vision_runtime,
        document_diagnostic_unavailable_reason=unavailable_reason,
        observation_capture_runtime=LiveGazeboObservationCaptureRuntime(),
    )

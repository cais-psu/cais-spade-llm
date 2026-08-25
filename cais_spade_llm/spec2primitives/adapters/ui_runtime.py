"""Compose the runtime values required by the Spec2Primitives PA UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cais_spade_llm.spec2primitives.adapters.dual_gazebo import DualGazeboRuntime
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
)

SPEC2PRIMITIVES_CONTEXTS_ROOT = Path(__file__).resolve().parents[1] / "contexts"


@dataclass(frozen=True)
class Spec2PrimitivesUIRuntime:
    """Hold the isolated UI runtime authorities and interaction root."""

    dual_gazebo: DualGazeboRuntime
    product_agent: ProductAgentContextRuntime
    contexts_root: Path


def create_spec2primitives_ui_runtime(
    dual_gazebo: DualGazeboRuntime,
) -> Spec2PrimitivesUIRuntime:
    """Create one application-owned runtime for the Spec2Primitives page."""
    from cais_spade_llm.spec2primitives.agents.pa.product_agent_runtime import (
        create_product_agent_context_runtime,
    )

    return Spec2PrimitivesUIRuntime(
        dual_gazebo=dual_gazebo,
        product_agent=create_product_agent_context_runtime(),
        contexts_root=SPEC2PRIMITIVES_CONTEXTS_ROOT,
    )

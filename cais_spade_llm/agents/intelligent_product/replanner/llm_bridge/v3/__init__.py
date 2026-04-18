"""LLM bridge v3 — archived implementation."""

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.bridge_safety import (
    BridgeSafetyMixin,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.bridge_session import (
    BridgeSessionMixin,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.universal_repair_session import (
    UniversalRepairSessionMixin,
)


class LlmBridgeReplannerMixin(
    UniversalRepairSessionMixin,
    BridgeSessionMixin,
    BridgeSafetyMixin,
):
    """Aggregate mixin for DES-guided LLM bridge replanning (v3).

    Uses v3 UniversalRepairSessionMixin as the primary LLM bridge session.
    BridgeSessionMixin provides session preparation and preprogrammed
    proposal validation.  BridgeSafetyMixin provides safety constraint
    derivation and modeled-continuation checks.
    """

    pass

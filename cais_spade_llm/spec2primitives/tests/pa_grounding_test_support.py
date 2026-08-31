"""Shared ontology configuration for ProductAgent grounding tests."""

from __future__ import annotations

from pathlib import Path

from cais_spade_llm.spec2primitives.agents.pa.context_grounding import (
    PAOntologyConfig,
)

PPR_NAMESPACE = "http://PAonto.com#"
MINIMAL_TBOX_PATH = Path(__file__).parent / "fixtures/ontology/minimal_ppr_tbox.owl"


def ontology_config() -> PAOntologyConfig:
    """Return the schema-only test fixture configuration."""
    return PAOntologyConfig(
        tbox_path=MINIMAL_TBOX_PATH,
        ppr_namespace=PPR_NAMESPACE,
    )

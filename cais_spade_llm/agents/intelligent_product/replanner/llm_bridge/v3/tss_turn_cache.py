"""TSS v3 turn cache: prompt artifact caching and duplicate-turn suppression.

Caches rendered prompt sections (primitive catalogs, schema description) across
turns and detects when a new proposal is identical to a previously rejected one
so the bridge can skip redundant validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.tss_feedback import (
    BridgeFeedbackSummary,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.recovery_context_builder import (
    recovery_context_to_prompt_dict,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.mutation_types import (
    RepairProgram,
    repair_program_to_dict,
)


# ---------------------------------------------------------------------------
# Cache structure
# ---------------------------------------------------------------------------

@dataclass
class TurnCache:
    """Bridge-owned cache for prompt deltas and duplicate-turn reuse."""

    turn_index: int
    context_fingerprint: str
    program_fingerprint: str | None
    rendered_catalog_by_resource: dict[str, str]
    rendered_schema_section: str
    last_feedback: BridgeFeedbackSummary | None
    last_rejected_proposal: dict[str, Any] | None
    last_validated_result: dict[str, Any] | None

    def render_relevant_catalog_excerpt(
        self,
        recovery_context: Any,
        *,
        relevant_resource_ids: set[str] | None = None,
    ) -> str:
        """Return cached catalog excerpts for resources present in context.

        If the context includes a resource not in the cache, returns an empty
        string (caller should fall back to full catalog rendering).
        """
        lines: list[str] = []
        for jid in (recovery_context.resource_snapshots or {}):
            if relevant_resource_ids and jid not in relevant_resource_ids:
                continue
            excerpt = self.rendered_catalog_by_resource.get(jid)
            if excerpt:
                lines.append(excerpt)
        if not lines:
            return ""
        return (
            "## Available Primitives (cached excerpt)\n\n"
            + "\n\n".join(lines)
        )

    def render_schema_reminder(self) -> str:
        """Return a compact schema reminder from the cached full section."""
        if not self.rendered_schema_section:
            return ""
        # Return a truncated version — first 40 lines as a reminder.
        section_lines = self.rendered_schema_section.splitlines()
        if len(section_lines) > 40:
            section_lines = section_lines[:40]
            section_lines.append("... (see Turn 1 for full schema)")
        return "\n".join(section_lines)

    def requires_full_regrounding(self, recovery_context: Any) -> bool:
        """Check whether the context changed enough to require a full prompt.

        Returns True if the context fingerprint differs from the cached one,
        indicating material changes to the recovery context.
        """
        new_fp = compute_context_fingerprint(recovery_context)
        return new_fp != self.context_fingerprint


# ---------------------------------------------------------------------------
# Program delta
# ---------------------------------------------------------------------------

@dataclass
class ProgramDelta:
    """Diff between current and cached repair programs."""

    is_identical: bool = False
    changed_functions: list[str] = field(default_factory=list)
    changed_success_conditions: bool = False
    touched_resources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

def compute_context_fingerprint(recovery_context: Any) -> str:
    """SHA256 fingerprint of the serialized recovery context."""
    payload = recovery_context_to_prompt_dict(recovery_context)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def compute_outline_context_fingerprint(recovery_context: Any) -> str:
    """SHA256 fingerprint for task-level outline validity.

    Unlike the full prompt fingerprint, this intentionally ignores rejection-only
    metadata such as discovered constraints so an accepted task-level outline
    remains valid across purely low-level synthesis failures.
    """
    payload = recovery_context_to_prompt_dict(recovery_context)
    payload.pop("discovered_constraints", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def compute_program_fingerprint(program: RepairProgram) -> str:
    """SHA256 of the serialized program (excluding bridge-only narration)."""
    d = repair_program_to_dict(program)
    d.pop("rationale", None)
    d.pop("reasoning", None)
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Diff + reuse logic
# ---------------------------------------------------------------------------

def diff_programs(
    program: RepairProgram,
    cache: TurnCache,
) -> ProgramDelta:
    """Compute a bridge-level delta between a new proposal and the cached one."""
    new_fp = compute_program_fingerprint(program)
    if new_fp == cache.program_fingerprint:
        return ProgramDelta(is_identical=True)

    changed_functions = [fn.name for fn in program.function_defs]
    touched_resources: set[str] = set()
    for step in program.steps:
        payload = step.payload if isinstance(step.payload, dict) else {}
        rjid = str(payload.get("resource_jid", "")).strip()
        if rjid:
            touched_resources.add(rjid)

    return ProgramDelta(
        changed_functions=changed_functions,
        changed_success_conditions=True,
        touched_resources=sorted(touched_resources),
    )


def can_reuse_validation_result(
    *,
    delta: ProgramDelta,
    context_fingerprint: str,
    cache: TurnCache,
) -> bool:
    """Safe memoization rule for bridge v3.

    Reuse only when the proposal is identical AND the recovery context is
    identical.  This suppresses duplicate-turn work without weakening the
    validator's authority.
    """
    return (
        delta.is_identical
        and context_fingerprint == cache.context_fingerprint
        and cache.last_validated_result is not None
    )

"""TSS v3 bridge feedback: structured validation feedback for the LLM.

Partitions validator rejections into bridge-facing buckets (blocked steps,
unmet obligations, violated rules, continuation notes) so the LLM receives
targeted guidance instead of one undifferentiated failure blob.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BridgeFeedbackSummary:
    """Compact bridge-owned summary of validator outcome.

    TSS-inspired at the *feedback* layer: separate what is blocked,
    what is still missing, and what remains viable.
    """

    blocked_steps: list[str] = field(default_factory=list)
    """Local issues: bad primitive usage, bad step sequencing, invalid mutation."""

    unmet_obligations: list[str] = field(default_factory=list)
    """Goal/reentry obligations still not discharged by the proposal."""

    violated_rules: list[str] = field(default_factory=list)
    """Safety violations or rule-specific rejection messages."""

    continuation_notes: list[str] = field(default_factory=list)
    """Continuation viability warnings or errors."""

    validator_notes: list[str] = field(default_factory=list)
    """Other validator messages that don't fit the above categories."""

    suggested_adaptations: list[str] = field(default_factory=list)
    """Rule-based suggestions for how the LLM might revise its proposal."""

    open_witnesses: list[dict[str, str]] = field(default_factory=list)
    """Structured missing execution witnesses still required by validation."""


# ---------------------------------------------------------------------------
# Summarize validation feedback
# ---------------------------------------------------------------------------

def summarize_validation_feedback(
    validated: Any,
) -> BridgeFeedbackSummary:
    """Partition validator rejection reasons into bridge-facing buckets.

    Parameters
    ----------
    validated:
        A ``ValidatedRepairProgram`` instance (or any object with a
        ``rejection_reasons`` attribute that is a list of dicts with
        ``check`` and ``message`` keys).
    """
    blocked_steps: list[str] = []
    unmet_obligations: list[str] = []
    violated_rules: list[str] = []
    continuation_notes: list[str] = []
    validator_notes: list[str] = []
    open_witnesses: list[dict[str, str]] = []

    for reason in getattr(validated, "rejection_reasons", None) or []:
        if not isinstance(reason, dict):
            continue
        check = str(reason.get("check", "")).strip()
        msg = str(reason.get("message", "")).strip()
        if not msg:
            continue

        if check in (
            "schema",
            "function_synthesis",
            "step_sequencing",
            "direct_task_call_policy",
            "mutation",
            "reasoning_contract",
        ):
            blocked_steps.append(msg)
        elif check in ("obligation_discharge", "pre_resume_obligation"):
            unmet_obligations.append(msg)
        elif check in ("ltlf_safety", "fsa_compilation"):
            violated_rules.append(msg)
        elif check == "continuation_viability":
            continuation_notes.append(msg)
        else:
            validator_notes.append(msg)
        open_witnesses.extend(_derive_open_witnesses(check=check, message=msg))

    return BridgeFeedbackSummary(
        blocked_steps=blocked_steps,
        unmet_obligations=unmet_obligations,
        violated_rules=violated_rules,
        continuation_notes=continuation_notes,
        validator_notes=validator_notes,
        suggested_adaptations=_derive_suggested_adaptations(
            blocked_steps=blocked_steps,
            unmet_obligations=unmet_obligations,
            violated_rules=violated_rules,
        ),
        open_witnesses=_dedupe_open_witnesses(open_witnesses),
    )


# ---------------------------------------------------------------------------
# Suggested adaptations (rule-based)
# ---------------------------------------------------------------------------

def _derive_suggested_adaptations(
    *,
    blocked_steps: list[str],
    unmet_obligations: list[str],
    violated_rules: list[str],
) -> list[str]:
    """Derive simple rule-based suggestions from rejection buckets."""
    suggestions: list[str] = []

    if unmet_obligations:
        for ob in unmet_obligations:
            ob_lower = ob.lower()
            if "restored" in ob_lower or "gripper" in ob_lower:
                suggestions.append(
                    f"Add a step to restore the part to the gripper: {ob}"
                )
            elif "assembled" in ob_lower:
                suggestions.append(
                    f"Ensure the assembly step is included: {ob}"
                )
            else:
                suggestions.append(f"Add steps to satisfy: {ob}")

    if violated_rules:
        for rule in violated_rules:
            rule_lower = rule.lower()
            if "mutex" in rule_lower or "simultaneous" in rule_lower:
                suggestions.append(
                    "Ensure only one robot occupies the shared zone at a time. "
                    "Add a clear/home step before the next robot approaches."
                )
            else:
                suggestions.append(f"Revise to avoid: {rule}")

    if blocked_steps:
        for step in blocked_steps:
            step_lower = step.lower()
            if "could not be resolved to place geometry" in step_lower:
                suggestions.append(
                    "If the destination is only a named staging anchor, do not use "
                    "compute_place_targets from that token alone. Move to the grounded "
                    "staging destination, descend, release_part, and retreat, or "
                    "supply explicit product_geometry."
                )
        suggestions.append(
            "Fix blocked steps: check primitive names, parameter types, "
            "and precondition ordering."
        )

    return suggestions


_PART_REJECTION_RE = re.compile(r"repair for '([^']+)'")
_GROUNDED_PART_RE = re.compile(r"grounded part '([^']+)'")
_OBLIGATION_RE = re.compile(r"([A-Za-z0-9_@-]+)\.([A-Za-z0-9_@-]+)")
_STAGING_PART_RE = re.compile(r"staging of '([^']+)'")


def _entity_from_rejection_message(message: str) -> str:
    for pattern in (_PART_REJECTION_RE, _GROUNDED_PART_RE, _STAGING_PART_RE):
        match = pattern.search(message)
        if match:
            return str(match.group(1) or "").strip()
    match = _OBLIGATION_RE.search(message)
    if match:
        return str(match.group(1) or "").strip()
    return ""


def _append_open_witness(
    witnesses: list[dict[str, str]],
    *,
    entity: str,
    witness_type: str,
    message: str,
) -> None:
    token = str(witness_type or "").strip()
    if not token:
        return
    row = {
        "entity": str(entity or "").strip(),
        "witness_type": token,
        "message": str(message or "").strip(),
    }
    witnesses.append({k: v for k, v in row.items() if v})


def _derive_open_witnesses(
    *,
    check: str,
    message: str,
) -> list[dict[str, str]]:
    witnesses: list[dict[str, str]] = []
    entity = _entity_from_rejection_message(message)
    lower = message.lower()

    if check == "pre_resume_obligation":
        _append_open_witness(
            witnesses,
            entity=entity,
            witness_type="pre_resume_obligation",
            message=message,
        )
        return witnesses

    if check != "under_modeled_part_recovery" and check != "blocker_staging_witness":
        return witnesses

    if "must ground the pickup from the trusted observed pose" in lower:
        _append_open_witness(
            witnesses,
            entity=entity,
            witness_type="pickup_from_observed_pose",
            message=message,
        )
    elif "never reaches the computed place-approach pose" in lower:
        _append_open_witness(
            witnesses,
            entity=entity,
            witness_type="destination_approach",
            message=message,
        )
    elif "must ground compute_place_targets with destination geometry" in lower:
        _append_open_witness(
            witnesses,
            entity=entity,
            witness_type="destination_geometry",
            message=message,
        )
    elif "transports the part after pickup but never reaches the computed placement target" in lower:
        _append_open_witness(
            witnesses,
            entity=entity,
            witness_type="destination_release",
            message=message,
        )
    elif "releases the part away from the computed placement target" in lower:
        _append_open_witness(
            witnesses,
            entity=entity,
            witness_type="destination_release",
            message=message,
        )
    elif "only performs local motion before release" in lower:
        _append_open_witness(
            witnesses,
            entity=entity,
            witness_type="post_grasp_transport",
            message=message,
        )
    elif "grasps the part but never completes a placement step" in lower:
        _append_open_witness(
            witnesses,
            entity=entity,
            witness_type="post_grasp_transport",
            message=message,
        )
    elif "requires an explicit non-assembly staging destination" in lower:
        _append_open_witness(
            witnesses,
            entity=entity,
            witness_type="staging_destination",
            message=message,
        )
    elif "must move to explicit staging destination" in lower:
        _append_open_witness(
            witnesses,
            entity=entity,
            witness_type="staging_destination",
            message=message,
        )
    elif "must descend to the staging destination" in lower or "must retreat after release_part" in lower:
        _append_open_witness(
            witnesses,
            entity=entity,
            witness_type="staging_descend_release_retreat",
            message=message,
        )

    return witnesses


def _dedupe_open_witnesses(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("entity") or "").strip(),
            str(row.get("witness_type") or "").strip(),
            str(row.get("message") or "").strip(),
        )
        if not key[1] or key in seen:
            continue
        seen.add(key)
        deduped.append({
            "entity": key[0],
            "witness_type": key[1],
            "message": key[2],
        })
    return deduped


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def feedback_to_prompt_section(feedback: BridgeFeedbackSummary) -> str:
    """Render a :class:`BridgeFeedbackSummary` as a prompt section for the LLM."""
    lines = ["## Bridge Feedback Summary"]

    if feedback.blocked_steps:
        lines.append(f"- Blocked steps: {len(feedback.blocked_steps)}")
        for msg in feedback.blocked_steps[:3]:
            lines.append(f"  - {msg}")

    if feedback.unmet_obligations:
        lines.append(f"- Unmet obligations: {len(feedback.unmet_obligations)}")
        for msg in feedback.unmet_obligations[:3]:
            lines.append(f"  - {msg}")

    if feedback.violated_rules:
        lines.append(f"- Violated rules: {len(feedback.violated_rules)}")
        for msg in feedback.violated_rules[:3]:
            lines.append(f"  - {msg}")

    if feedback.continuation_notes:
        lines.append(
            f"- Continuation notes: {len(feedback.continuation_notes)}"
        )
        for msg in feedback.continuation_notes[:2]:
            lines.append(f"  - {msg}")

    if feedback.validator_notes:
        lines.append(
            f"- Other validator notes: {len(feedback.validator_notes)}"
        )

    if feedback.open_witnesses:
        lines.append(f"- Open witnesses: {len(feedback.open_witnesses)}")

    if feedback.suggested_adaptations:
        lines.append("\n### Suggested Revisions")
        for msg in feedback.suggested_adaptations:
            lines.append(f"- {msg}")

    return "\n".join(lines)


def open_witnesses_to_prompt_section(feedback: BridgeFeedbackSummary) -> str:
    """Render only the still-missing execution witnesses for retry prompts."""
    if not feedback.open_witnesses:
        return ""

    lines = ["## Open Repair Witnesses"]
    for row in feedback.open_witnesses:
        if not isinstance(row, dict):
            continue
        entity = str(row.get("entity") or "").strip()
        witness_type = str(row.get("witness_type") or "").strip()
        message = str(row.get("message") or "").strip()
        prefix = f"- `{witness_type}`"
        if entity:
            prefix = f"- `{entity}`: `{witness_type}`"
        if message:
            prefix += f" — {message}"
        lines.append(prefix)
    return "\n".join(lines)

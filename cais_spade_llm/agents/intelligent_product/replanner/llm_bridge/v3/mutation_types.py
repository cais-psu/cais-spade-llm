"""Core types for the universal mutation-synthesis LLM bridge (v2).

Defines the typed repair language: RecoveryContext, SynthesizedTaskFn,
RepairProgram, ValidatedRepairProgram, TaskMutationStep, RepairStep,
and RecoveryLibraryEntry.

The RepairProgram is the single universal repair artifact emitted by the
LLM.  Deterministic code (the validator) produces ValidatedRepairProgram
which owns risk classification, approval gating, and continuation
viability — the LLM never sets those fields.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskMutationType(str, enum.Enum):
    """v1 mutation surface — only these four ship initially."""

    INSERT = "insert"
    DELETE = "delete"
    REASSIGN = "reassign"
    REPLACE_SUFFIX = "replace_suffix"


class RepairStepKind(str, enum.Enum):
    """Valid step kinds inside a RepairProgram.

    ``observe`` is intentionally absent — it is a top-level *turn* output,
    not an in-program step.  In-execution observation is handled by
    observation primitives (e.g. ``detect_parts``) within
    ``SynthesizedTaskFn.primitive_program`` using ``store_as`` /
    ``context_ref``.
    """

    TASK_MUTATION = "task_mutation"
    CALL_FUNCTION = "call_function"
    WAIT = "wait"
    RESUME_SUFFIX = "resume_suffix"


class RiskLevel(str, enum.Enum):
    """Validator-owned risk classification."""

    LOW = "low"
    HIGH = "high"


# ---------------------------------------------------------------------------
# RecoveryContext — obligation-centric view of system state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecoveryContext:
    """Everything the LLM needs to reason about recovery.

    Obligation-centric: presents all resources and obligations equally,
    rather than focusing on one stuck resource.
    """

    resource_snapshots: dict[str, dict[str, Any]]
    """Mapping from resource JID to canonical snapshot dict."""

    part_states: dict[str, dict[str, Any]]
    """Mapping from part name to ``{state, location, ...}``."""

    pending_tasks: list[dict[str, Any]]
    """Remaining tasks from the active plan suffix."""

    active_obligations: list[dict[str, Any]]
    """Safety/goal obligations currently in force."""

    goal_state: str
    """Human-readable goal description."""

    priority_context: dict[str, Any]
    """Demand / priority metadata (may be empty)."""

    available_task_actions: list[dict[str, Any]]
    """Tools-catalog rows available for nominal plan mutations."""

    available_primitives: dict[str, list[dict[str, Any]]]
    """Resource JID → primitive catalog entries."""

    capability_degradations: list[dict[str, Any]]
    """Known infeasible assignments (e.g. resource offline)."""

    observation_store: dict[str, Any] = field(default_factory=dict)
    """Outputs captured from prior top-level ``observe`` turns."""

    discovered_constraints: list[dict[str, Any]] = field(default_factory=list)
    """Constraints extracted from prior validator rejections
    (fresh-prompt constraint accumulation)."""

    grounded_environment_facts: dict[str, Any] = field(default_factory=dict)
    """Derived neutral environment facts computed from observations,
    obligations, geometry, and capability state."""


# ---------------------------------------------------------------------------
# SynthesizedTaskFn — LLM-authored recovery function
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SynthesizedTaskFn:
    """A named task-level recovery function backed by known primitives.

    The LLM defines these inside ``RepairProgram.function_defs``.
    Each function is independently validatable.
    """

    name: str
    """Unique name within the RepairProgram (e.g. ``stow_mrp_to_printer``)."""

    intent: str
    """Natural-language purpose."""

    resource_constraints: dict[str, Any]
    """Required resource_type, capabilities, etc."""

    inputs: dict[str, Any]
    """Parameter schema for the function (may be empty)."""

    preconditions: dict[str, dict[str, Any]]
    """Snapshot field → semantic rule (e.g. ``{"held_part": {"equals": null}}``)."""

    effects: dict[str, dict[str, Any]]
    """Snapshot field → semantic effect (e.g. ``{"held_part": {"set": null}}``)."""

    primitive_program: list[dict[str, Any]]
    """Ordered primitive steps.  Each step has ``primitive``, ``params``,
    and optionally ``store_as``."""

    expected_post_state: dict[str, Any]
    """Projected snapshot fields after all primitives execute."""


# ---------------------------------------------------------------------------
# RepairStep — a single step inside a RepairProgram
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepairStep:
    """One step in a RepairProgram, discriminated by *kind*."""

    kind: RepairStepKind
    """Step type — determines the shape of *payload*."""

    payload: dict[str, Any]
    """Union-discriminated data:

    * ``task_mutation`` → ``{"mutation_type", "target_task_ids", "payload"}``
    * ``call_function`` → ``{"function_name", "resource_jid", "args"}``
    * ``wait``          → ``{"until": {entity_kind, entity, field, expected}}``
    * ``resume_suffix`` → ``{}``
    """


# ---------------------------------------------------------------------------
# TaskMutationStep — extracted / validated mutation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskMutationStep:
    """A validated task-graph mutation.

    ``reassign`` is a validation-enriched delete+insert: the LLM provides
    both the task to remove and its replacement already grounded for the
    new resource.  The validator checks portability.
    """

    mutation_type: TaskMutationType
    target_task_ids: list[str]
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# RepairProgram — the single universal repair artifact (LLM output)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepairProgram:
    """Universal repair artifact emitted by the LLM.

    Contains synthesized function definitions, an ordered step list, and
    success conditions.  Does **not** contain risk_level or approval
    flags — those belong to :class:`ValidatedRepairProgram`.
    """

    function_defs: list[SynthesizedTaskFn]
    """Named recovery functions the LLM has defined for this program."""

    steps: list[RepairStep]
    """Ordered execution steps (mutations, calls, waits, resume)."""

    success_conditions: list[dict[str, Any]]
    """Conditions that must hold after the program completes.

    Each entry: ``{entity_kind, entity, field, expected}``."""

    rationale: str = ""
    """Optional LLM-provided explanation."""

    reasoning: dict[str, Any] = field(default_factory=dict)
    """v3 structured planning analysis (current_state_analysis, goal_gap_analysis,
    transition_plan, safety_check).  Empty dict for v2 programs."""


# ---------------------------------------------------------------------------
# ValidatedRepairProgram — validator-owned wrapper
# ---------------------------------------------------------------------------

@dataclass
class ValidatedRepairProgram:
    """Validator output wrapping a :class:`RepairProgram`.

    All fields except *program* are **authoritative** — they override any
    LLM assertions.
    """

    program: RepairProgram
    """The original LLM-emitted program."""

    risk_level: RiskLevel = RiskLevel.HIGH
    """Computed risk classification."""

    requires_operator_approval: bool = True
    """Whether operator must approve before execution."""

    continuation_viable: bool = False
    """Whether the repaired FSA admits a path from current state to
    a marked (accepting) state."""

    rejection_reasons: list[dict[str, Any]] = field(default_factory=list)
    """Structured rejection reasons for the feedback loop.

    Each entry: ``{layer, check, message, ...extra_context}``.
    When non-empty the program is **rejected** and the reasons are
    condensed into the ``discovered_constraints`` list for the next
    fresh-prompt iteration.
    """

    @property
    def is_valid(self) -> bool:
        """A program is valid iff there are no rejection reasons."""
        return len(self.rejection_reasons) == 0


# ---------------------------------------------------------------------------
# RecoveryLibraryEntry — candidate memory
# ---------------------------------------------------------------------------

@dataclass
class RecoveryLibraryEntry:
    """A synthesized function stored in the recovery library.

    The library is **candidate memory** in v1 — the LLM may retrieve and
    adapt these, but every reused function still goes through full
    validation as part of a fresh ``RepairProgram``.
    """

    function_def: SynthesizedTaskFn
    """The validated synthesized function."""

    signature_hash: str
    """Cache key hash covering resource profile, primitive fingerprint,
    touched entities, preconditions, post-state, catalog version."""

    resource_profile_id: str
    """Resource type / profile identity."""

    primitive_fingerprint: str
    """Ordered hash of primitive names + param schemas."""

    touched_entities: list[str]
    """Part / resource entity names touched by this function."""

    catalog_version: str
    """Primitive catalog version at time of storage."""

    validation_count: int = 0
    """Number of times deterministic validation has passed."""

    runtime_success_count: int = 0
    """Number of successful runtime executions."""

    runtime_failure_count: int = 0
    """Number of failed runtime executions."""

    first_seen_utc: str = ""
    last_used_utc: str = ""

    source_context_summary: dict[str, Any] = field(default_factory=dict)
    """Compact summary of the RecoveryContext where the function was
    first synthesized (for debugging / analysis)."""


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _synthesized_fn_to_dict(fn: SynthesizedTaskFn) -> dict[str, Any]:
    """Serialize a :class:`SynthesizedTaskFn` to a plain dict."""
    return {
        "name": fn.name,
        "intent": fn.intent,
        "resource_constraints": fn.resource_constraints,
        "inputs": fn.inputs,
        "preconditions": fn.preconditions,
        "effects": fn.effects,
        "primitive_program": fn.primitive_program,
        "expected_post_state": fn.expected_post_state,
    }


def _synthesized_fn_from_dict(d: dict[str, Any]) -> SynthesizedTaskFn:
    """Deserialize a plain dict to :class:`SynthesizedTaskFn`."""
    resource_constraints = dict(d.get("resource_constraints") or {})
    resource_jid = str(d.get("resource_jid") or "").strip()
    if resource_jid and "resource_jid" not in resource_constraints:
        resource_constraints["resource_jid"] = resource_jid
    return SynthesizedTaskFn(
        name=str(d.get("name") or d.get("function_name") or ""),
        intent=str(d.get("intent") or d.get("description") or ""),
        resource_constraints=resource_constraints,
        inputs=dict(d.get("inputs") or {}),
        preconditions=dict(d.get("preconditions") or {}),
        effects=dict(d.get("effects") or {}),
        primitive_program=list(d.get("primitive_program") or d.get("primitives") or []),
        expected_post_state=dict(d.get("expected_post_state") or {}),
    )


def _repair_step_to_dict(step: RepairStep) -> dict[str, Any]:
    """Serialize a :class:`RepairStep` to a plain dict."""
    return {"kind": step.kind.value, "payload": step.payload}


# Map common LLM mistakes to valid RepairStepKind values.
_STEP_KIND_ALIASES: dict[str, str] = {
    "replace_suffix": "task_mutation",
    "insert": "task_mutation",
    "delete": "task_mutation",
    "reassign": "task_mutation",
    "repair": "call_function",
    "run_task_fn": "call_function",
    "execute_function": "call_function",
    "execute": "call_function",
}

_MUTATION_TYPE_ALIASES: dict[str, str] = {
    "append_action": "insert",
}


def _repair_step_from_dict(d: dict[str, Any]) -> RepairStep:
    """Deserialize a plain dict to :class:`RepairStep`."""
    if not isinstance(d, dict):
        raise ValueError("repair step must be an object")
    payload = dict(d.get("payload") or {})
    raw_kind = str(d.get("kind") or "").strip().lower()
    if not raw_kind:
        if d.get("resume_suffix") is True:
            raw_kind = "resume_suffix"
            payload = {}
        elif d.get("fn"):
            raw_kind = "call_function"
            payload = {
                "function_name": d.get("fn"),
                "resource_jid": d.get("resource_jid"),
                "args": dict(d.get("args") or {}),
            }
        else:
            raise ValueError("repair step missing kind")
    resolved_kind = _STEP_KIND_ALIASES.get(raw_kind, raw_kind)

    if resolved_kind == RepairStepKind.CALL_FUNCTION.value:
        if "function_name" not in payload and d.get("function_name"):
            payload["function_name"] = d.get("function_name")
        if "resource_jid" not in payload and d.get("resource_jid"):
            payload["resource_jid"] = d.get("resource_jid")
        if "args" not in payload:
            payload["args"] = dict(d.get("args") or {})
    elif resolved_kind == RepairStepKind.WAIT.value:
        if "until" not in payload and isinstance(d.get("until"), dict):
            payload["until"] = dict(d.get("until") or {})
    elif resolved_kind == RepairStepKind.RESUME_SUFFIX.value:
        payload = {}

    mutation_type = str(payload.get("mutation_type") or "").strip().lower()
    if mutation_type in _MUTATION_TYPE_ALIASES:
        payload["mutation_type"] = _MUTATION_TYPE_ALIASES[mutation_type]

    # If the LLM used a mutation type name as the step kind, wrap it
    # as a task_mutation step with the correct mutation_type in payload.
    if raw_kind in _STEP_KIND_ALIASES and raw_kind != "task_mutation":
        if "mutation_type" not in payload:
            payload["mutation_type"] = raw_kind

    return RepairStep(
        kind=RepairStepKind(resolved_kind),
        payload=payload,
    )


def repair_program_to_dict(program: RepairProgram) -> dict[str, Any]:
    """Serialize a :class:`RepairProgram` to a plain dict."""
    d: dict[str, Any] = {
        "function_defs": [_synthesized_fn_to_dict(f) for f in program.function_defs],
        "steps": [_repair_step_to_dict(s) for s in program.steps],
        "success_conditions": program.success_conditions,
        "rationale": program.rationale,
    }
    if program.reasoning:
        d["reasoning"] = program.reasoning
    return d


def repair_program_from_dict(d: dict[str, Any]) -> RepairProgram:
    """Deserialize a plain dict (from LLM JSON) to :class:`RepairProgram`."""
    return RepairProgram(
        function_defs=[
            _synthesized_fn_from_dict(fd)
            for fd in (d.get("function_defs") or [])
        ],
        steps=[
            _repair_step_from_dict(s)
            for s in (d.get("steps") or [])
        ],
        success_conditions=list(d.get("success_conditions") or []),
        rationale=str(d.get("rationale", "")),
        reasoning=dict(d.get("reasoning") or {}),
    )


def validated_program_to_dict(vp: ValidatedRepairProgram) -> dict[str, Any]:
    """Serialize a :class:`ValidatedRepairProgram` to a plain dict."""
    return {
        "program": repair_program_to_dict(vp.program),
        "risk_level": vp.risk_level.value,
        "requires_operator_approval": vp.requires_operator_approval,
        "continuation_viable": vp.continuation_viable,
        "rejection_reasons": vp.rejection_reasons,
        "is_valid": vp.is_valid,
    }


def validated_program_from_dict(d: dict[str, Any]) -> ValidatedRepairProgram:
    """Deserialize a plain dict to :class:`ValidatedRepairProgram`."""
    program_dict = d.get("program") or {}
    return ValidatedRepairProgram(
        program=repair_program_from_dict(program_dict),
        risk_level=RiskLevel(d.get("risk_level", "high")),
        requires_operator_approval=bool(d.get("requires_operator_approval", True)),
        continuation_viable=bool(d.get("continuation_viable", False)),
        rejection_reasons=list(d.get("rejection_reasons") or []),
    )


def library_entry_to_dict(entry: RecoveryLibraryEntry) -> dict[str, Any]:
    """Serialize a :class:`RecoveryLibraryEntry` to a plain dict."""
    return {
        "function_def": _synthesized_fn_to_dict(entry.function_def),
        "signature_hash": entry.signature_hash,
        "resource_profile_id": entry.resource_profile_id,
        "primitive_fingerprint": entry.primitive_fingerprint,
        "touched_entities": entry.touched_entities,
        "catalog_version": entry.catalog_version,
        "validation_count": entry.validation_count,
        "runtime_success_count": entry.runtime_success_count,
        "runtime_failure_count": entry.runtime_failure_count,
        "first_seen_utc": entry.first_seen_utc,
        "last_used_utc": entry.last_used_utc,
        "source_context_summary": entry.source_context_summary,
    }


def library_entry_from_dict(d: dict[str, Any]) -> RecoveryLibraryEntry:
    """Deserialize a plain dict to :class:`RecoveryLibraryEntry`."""
    return RecoveryLibraryEntry(
        function_def=_synthesized_fn_from_dict(d.get("function_def") or {}),
        signature_hash=str(d.get("signature_hash", "")),
        resource_profile_id=str(d.get("resource_profile_id", "")),
        primitive_fingerprint=str(d.get("primitive_fingerprint", "")),
        touched_entities=list(d.get("touched_entities") or []),
        catalog_version=str(d.get("catalog_version", "")),
        validation_count=int(d.get("validation_count", 0)),
        runtime_success_count=int(d.get("runtime_success_count", 0)),
        runtime_failure_count=int(d.get("runtime_failure_count", 0)),
        first_seen_utc=str(d.get("first_seen_utc", "")),
        last_used_utc=str(d.get("last_used_utc", "")),
        source_context_summary=dict(d.get("source_context_summary") or {}),
    )


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def compute_primitive_fingerprint(
    primitive_program: list[dict[str, Any]],
) -> str:
    """Ordered hash of primitive names and param key sets."""
    parts: list[str] = []
    for step in primitive_program:
        prim = str(step.get("primitive", ""))
        param_keys = sorted(str(k) for k in (step.get("params") or {}).keys())
        parts.append(f"{prim}({','.join(param_keys)})")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def compute_signature_hash(
    resource_profile_id: str,
    primitive_fingerprint: str,
    touched_entities: list[str],
    preconditions: dict[str, dict[str, Any]],
    expected_post_state: dict[str, Any],
    catalog_version: str,
) -> str:
    """Compute the full cache key hash for library matching."""
    blob = json.dumps(
        {
            "resource_profile_id": resource_profile_id,
            "primitive_fingerprint": primitive_fingerprint,
            "touched_entities": sorted(touched_entities),
            "preconditions": preconditions,
            "expected_post_state": expected_post_state,
            "catalog_version": catalog_version,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Constraint extraction helper
# ---------------------------------------------------------------------------

def extract_constraint_from_rejection(
    rejection: dict[str, Any],
) -> dict[str, Any]:
    """Condense a rejection reason into a discovered-constraint entry.

    Used by the session to build the ``discovered_constraints`` list for
    fresh-prompt constraint accumulation.
    """
    return {
        "layer": str(rejection.get("layer", "")),
        "check": str(rejection.get("check", "")),
        "constraint": str(rejection.get("message", "")),
        "rule_id": str(rejection.get("rule_id", "")),
    }

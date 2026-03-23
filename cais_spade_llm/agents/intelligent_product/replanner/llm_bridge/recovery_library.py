"""Persistent candidate-memory store for validated synthesized functions.

In v1 the library is **candidate memory** — the LLM may retrieve, reuse,
or adapt prior functions, but every reused function still goes through
full validation as part of a fresh ``RepairProgram``.  No direct execution
from library lookup.

Storage lives at ``store/recovery_library/library.json`` (durable across
runs, outside ``monitor/`` ephemeral state).
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.mutation_types import (
    RecoveryLibraryEntry,
    SynthesizedTaskFn,
    compute_primitive_fingerprint,
    compute_signature_hash,
    library_entry_from_dict,
    library_entry_to_dict,
    _synthesized_fn_to_dict,
)

logger = logging.getLogger(__name__)

_DEFAULT_STORE_PATH = Path("store/recovery_library/library.json")


class RecoveryLibrary:
    """Persistent candidate-memory store for synthesized functions."""

    def __init__(self, storage_path: Path | str | None = None) -> None:
        self._storage_path = Path(storage_path) if storage_path else _DEFAULT_STORE_PATH
        self._entries: dict[str, RecoveryLibraryEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def find_matching(
        self,
        *,
        resource_profile_id: str = "",
        intent_keywords: list[str] | None = None,
        precondition_fields: set[str] | None = None,
        effect_fields: set[str] | None = None,
        max_results: int = 5,
    ) -> list[RecoveryLibraryEntry]:
        """Find library entries matching the given criteria.

        Matching is approximate — the caller (LLM prompt builder) presents
        candidates, and full validation gates any actual reuse.
        """
        candidates: list[tuple[int, RecoveryLibraryEntry]] = []

        for entry in self._entries.values():
            score = 0

            # Resource type match.
            if resource_profile_id and entry.resource_profile_id == resource_profile_id:
                score += 3

            # Intent keyword overlap.
            if intent_keywords:
                entry_words = set(entry.function_def.intent.lower().split())
                for kw in intent_keywords:
                    if kw.lower() in entry_words:
                        score += 1

            # Precondition field overlap.
            if precondition_fields:
                entry_pre_fields = set(entry.function_def.preconditions.keys())
                overlap = precondition_fields & entry_pre_fields
                score += len(overlap)

            # Effect field overlap.
            if effect_fields:
                entry_eff_fields = set(entry.function_def.effects.keys())
                overlap = effect_fields & entry_eff_fields
                score += len(overlap)

            # Prefer runtime-proven functions.
            if entry.runtime_success_count > 0:
                score += 2

            if score > 0:
                candidates.append((score, entry))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in candidates[:max_results]]

    def get_by_hash(self, signature_hash: str) -> RecoveryLibraryEntry | None:
        """Look up an entry by its signature hash."""
        return self._entries.get(signature_hash)

    @property
    def entries(self) -> list[RecoveryLibraryEntry]:
        """All entries in the library."""
        return list(self._entries.values())

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_validated(
        self,
        fn_def: SynthesizedTaskFn,
        *,
        resource_profile_id: str,
        catalog_version: str = "",
        context_summary: dict[str, Any] | None = None,
    ) -> str:
        """Register a validated synthesized function.

        Called after deterministic validation passes.  The entry starts
        with ``runtime_success_count=0`` until runtime execution succeeds.

        Returns the signature hash.
        """
        touched = _extract_touched_entities(fn_def)
        fingerprint = compute_primitive_fingerprint(fn_def.primitive_program)
        sig_hash = compute_signature_hash(
            resource_profile_id=resource_profile_id,
            primitive_fingerprint=fingerprint,
            touched_entities=touched,
            preconditions=fn_def.preconditions,
            expected_post_state=fn_def.expected_post_state,
            catalog_version=catalog_version,
        )

        now = datetime.now(timezone.utc).isoformat()

        existing = self._entries.get(sig_hash)
        if existing is not None:
            existing.validation_count += 1
            existing.last_used_utc = now
            self._save()
            return sig_hash

        entry = RecoveryLibraryEntry(
            function_def=fn_def,
            signature_hash=sig_hash,
            resource_profile_id=resource_profile_id,
            primitive_fingerprint=fingerprint,
            touched_entities=touched,
            catalog_version=catalog_version,
            validation_count=1,
            runtime_success_count=0,
            runtime_failure_count=0,
            first_seen_utc=now,
            last_used_utc=now,
            source_context_summary=dict(context_summary or {}),
        )
        self._entries[sig_hash] = entry
        self._save()
        logger.info(
            "Recovery library: registered '%s' (hash=%s)",
            fn_def.name, sig_hash,
        )
        return sig_hash

    def record_runtime_success(self, signature_hash: str) -> None:
        """Record a successful runtime execution."""
        entry = self._entries.get(signature_hash)
        if entry is None:
            return
        entry.runtime_success_count += 1
        entry.last_used_utc = datetime.now(timezone.utc).isoformat()
        self._save()

    def record_runtime_failure(self, signature_hash: str) -> None:
        """Record a failed runtime execution."""
        entry = self._entries.get(signature_hash)
        if entry is None:
            return
        entry.runtime_failure_count += 1
        entry.last_used_utc = datetime.now(timezone.utc).isoformat()
        self._save()

    # ------------------------------------------------------------------
    # Seeding from compiler_map
    # ------------------------------------------------------------------

    def seed_from_compiler_map(
        self,
        seeds: list[dict[str, Any]],
        *,
        resource_profile_id: str,
        catalog_version: str = "",
    ) -> int:
        """Seed the library from existing compiler_map knowledge.

        Each seed dict should have::

            {
                "name": str,
                "intent": str,
                "preconditions": dict,
                "effects": dict,
                "primitive_program": list[dict],
                "expected_post_state": dict,
            }

        Returns the number of entries added.
        """
        added = 0
        for seed in seeds:
            if not isinstance(seed, dict):
                continue
            fn_def = SynthesizedTaskFn(
                name=str(seed.get("name", "")),
                intent=str(seed.get("intent", "")),
                resource_constraints={"resource_type": resource_profile_id},
                inputs={},
                preconditions=dict(seed.get("preconditions") or {}),
                effects=dict(seed.get("effects") or {}),
                primitive_program=list(seed.get("primitive_program") or []),
                expected_post_state=dict(seed.get("expected_post_state") or {}),
            )
            self.register_validated(
                fn_def,
                resource_profile_id=resource_profile_id,
                catalog_version=catalog_version,
                context_summary={"source": "compiler_map_seed"},
            )
            added += 1
        return added

    # ------------------------------------------------------------------
    # Low-risk check
    # ------------------------------------------------------------------

    def is_runtime_proven(self, signature_hash: str) -> bool:
        """Check if a function has been runtime-proven (success_count >= 1).

        Part of the unified low-risk definition.
        """
        entry = self._entries.get(signature_hash)
        if entry is None:
            return False
        return entry.runtime_success_count >= 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load entries from disk."""
        if not self._storage_path.exists():
            self._entries = {}
            return
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
            entries_list = raw.get("entries") or []
            self._entries = {}
            for d in entries_list:
                if isinstance(d, dict):
                    entry = library_entry_from_dict(d)
                    if entry.signature_hash:
                        self._entries[entry.signature_hash] = entry
            logger.info(
                "Recovery library: loaded %d entries from %s",
                len(self._entries), self._storage_path,
            )
        except Exception:
            logger.warning(
                "Recovery library: failed to load from %s",
                self._storage_path, exc_info=True,
            )
            self._entries = {}

    def _save(self) -> None:
        """Persist entries to disk."""
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "entries": [
                    library_entry_to_dict(e) for e in self._entries.values()
                ],
            }
            self._storage_path.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            logger.warning(
                "Recovery library: failed to save to %s",
                self._storage_path, exc_info=True,
            )

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def candidates_for_prompt(
        self,
        *,
        resource_profile_id: str = "",
        max_entries: int = 5,
    ) -> list[dict[str, Any]]:
        """Return compact summaries of library entries for the LLM prompt."""
        matching = self.find_matching(
            resource_profile_id=resource_profile_id,
            max_results=max_entries,
        )
        summaries: list[dict[str, Any]] = []
        for entry in matching:
            fn = entry.function_def
            summaries.append({
                "name": fn.name,
                "intent": fn.intent,
                "preconditions": fn.preconditions,
                "effects": fn.effects,
                "primitive_count": len(fn.primitive_program),
                "runtime_successes": entry.runtime_success_count,
                "signature_hash": entry.signature_hash,
            })
        return summaries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_touched_entities(fn_def: SynthesizedTaskFn) -> list[str]:
    """Extract part/resource entity names touched by a function."""
    entities: set[str] = set()
    for step in fn_def.primitive_program:
        params = step.get("params") or {}
        for key in ("part_name", "part_filter", "target_location"):
            val = params.get(key)
            if isinstance(val, str) and val.strip():
                entities.add(val.strip())
    return sorted(entities)

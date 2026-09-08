from __future__ import annotations

"""Rebuild a completed allocation while preserving its approved product evidence."""

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Mapping
from pathlib import Path

from rdflib import URIRef

from .context_interaction import ProductAgentContextRuntime
from .grounding_contracts import (
    ProductContextView,
    build_product_context_view,
    load_pa_context_grounding_completion,
    persist_pa_context_grounding_completion,
    validate_grounding_evidence,
)
from .ontology_grounding import _validated_proposal, target_feature_evidence_refs
from .presentation_records import load_evidence_presentation
from .product_context import load_interaction_abox
from .production_grounding import (
    ProductionGroundingError,
    ProductionProductContextGroundingRuntime,
    _approved_evidence_handles,
    _NativeEvidenceInvestigation,
)

logger = logging.getLogger(__name__)


async def reassign_completed_interaction(
    *,
    interaction_root: Path,
    runtime: ProductionProductContextGroundingRuntime,
    product_agent: ProductAgentContextRuntime,
    requested_resource_symbol: str | None,
    activate: bool = True,
) -> dict[str, object]:
    """Stage fresh PA allocation and optionally replace one validated saved interaction.

    Args:
        interaction_root: Completed interaction whose grounded evidence is reused.
        runtime: Owned production grounding runtime with live planning access.
        product_agent: PA model authority used for a new allocation decision.
        requested_resource_symbol: Explicit user constraint, or None for a preference trial.
        activate: Whether to install the validated candidate at the original path.

    Returns:
        Archive and candidate paths, selected robot, and activation status.

    Raises:
        ProductionGroundingError: If allocation fails or the original changes during staging.
        ValueError: If any saved evidence or replacement completion is invalid.
    """
    root = Path(interaction_root).resolve()
    completion = load_pa_context_grounding_completion(root).to_record()
    lock_path = root.parent / f".{root.name}.resource_reassignment.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ProductionGroundingError(
            "This interaction already has a reassignment in progress."
        ) from exc
    os.close(lock_fd)
    attempt_id = uuid.uuid4().hex
    attempt = root.parent / "_resource_reassignments" / root.name / attempt_id
    # Grounding pins absolute paths in the authorized sibling source_cache. Keep both
    # copies beside the original without matching the UI's interaction_* discovery.
    archive = root.parent / f"_resource_reassignment_{attempt_id}_original"
    candidate = root.parent / f"_resource_reassignment_{attempt_id}_candidate"
    try:
        attempt.mkdir(parents=True)
        before = _file_hashes(root)
        shutil.copytree(root, archive)
        if _file_hashes(archive) != before or _file_hashes(root) != before:
            raise ProductionGroundingError("Original interaction changed during archival.")
        load_pa_context_grounding_completion(archive)
        shutil.copytree(archive, candidate)
        _restore_before_assignment(candidate, completion, runtime)
        output = await _allocate_staged_interaction(
            candidate, completion, runtime, product_agent, requested_resource_symbol
        )
        _assert_grounding_preserved(archive, candidate, completion)
        rebuilt = load_pa_context_grounding_completion(candidate).to_record()
        selection = _read_json(candidate / rebuilt["resource_selection_ref"])
        if (
            requested_resource_symbol is not None
            and selection["selected_resource_symbol"] != requested_resource_symbol
        ):
            raise ProductionGroundingError(
                "Reassignment differs from the user's requested resource."
            )
        if _file_hashes(root) != before:
            raise ProductionGroundingError(
                "Original interaction changed; replacement was not activated."
            )
        if activate:
            _activate_candidate(
                root, candidate, root.parent / f"_resource_reassignment_{attempt_id}_retired"
            )
            candidate = root
        result = {
            "archive_root": str(archive),
            "candidate_root": str(candidate),
            "activated": activate,
            "requested_resource_symbol": requested_resource_symbol,
            "selected_resource_symbol": selection["selected_resource_symbol"],
            "completion_fingerprint": rebuilt["fingerprint"],
            "resource_selection_ref": output["resource_selection_ref"],
        }
        _write_json(attempt / "result.json", result)
        logger.info(
            "Resource reassignment prepared for %s: %s",
            root.name,
            result["selected_resource_symbol"],
        )
        return result
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if attempt.exists():
            _write_json(
                attempt / "failure.json",
                {
                    "requested_resource_symbol": requested_resource_symbol,
                    "failure": f"{type(exc).__name__}: {exc}",
                },
            )
        raise
    finally:
        lock_path.unlink()


def _restore_before_assignment(
    root: Path,
    completion: Mapping[str, object],
    runtime: ProductionProductContextGroundingRuntime,
) -> None:
    """Remove only the terminal assignment from a private staged copy."""
    abox = load_interaction_abox(root, runtime._tbox)
    delta_path = root / completion["resource_assignment_delta_ref"]
    delta = _read_json(delta_path)
    if (
        delta["delta_number"] != abox.delta_count
        or delta["producer"] != "resource_grounding_host"
        or len(delta["assertions"]) != 4
        or delta["typed_context_refs"]
    ):
        raise ProductionGroundingError(
            "Reassignment requires the terminal four-assertion resource assignment."
        )
    for assertion in delta["assertions"]:
        if assertion["object"]["kind"] != "iri":
            raise ProductionGroundingError("Resource assignment contains a non-IRI assertion.")
        triple = (
            URIRef(assertion["subject"]),
            URIRef(assertion["predicate"]),
            URIRef(assertion["object"]["value"]),
        )
        if triple not in abox.graph:
            raise ProductionGroundingError("Resource assignment is missing from its ontology.")
        abox.graph.remove(triple)
    manifest = _read_json(abox.manifest_path)
    provenance = _read_json(abox.provenance_path)
    remaining = [
        entry for entry in provenance["assertions"] if entry.get("delta_ref") != delta_path.name
    ]
    if len(provenance["assertions"]) - len(remaining) != 4:
        raise ProductionGroundingError("Resource assignment provenance is inconsistent.")
    provenance["assertions"] = remaining
    manifest["delta_count"] -= 1
    manifest["accepted_assertion_count"] -= 4
    abox.graph.serialize(destination=str(abox.abox_path), format="turtle")
    _write_json(abox.manifest_path, manifest)
    _write_json(abox.provenance_path, provenance)
    delta_path.unlink()
    load_interaction_abox(root, runtime._tbox)
    # The original completion and RA data remain byte-for-byte in the external archive.
    (root / "interaction_record/context_completion_0001.json").unlink()
    for directory in ("products/grounding/completion", "composition", "resources"):
        path = root / directory
        if path.exists():
            shutil.rmtree(path)


async def _allocate_staged_interaction(
    root: Path,
    previous: Mapping[str, object],
    runtime: ProductionProductContextGroundingRuntime,
    product_agent: ProductAgentContextRuntime,
    requested_resource_symbol: str | None,
) -> Mapping[str, object]:
    abox = load_interaction_abox(root, runtime._tbox)
    view = build_product_context_view(
        root, abox, attempted_evidence=(), assessed_at_ns=time.time_ns()
    )
    projection_ref = previous["ontology_projection_ref"]
    projection = _read_json(root / projection_ref)
    validate_grounding_evidence(root, projection)
    proposal = _validated_proposal(
        projection["output"],
        abox=abox,
        workcell=runtime._workcell,
        authorized_evidence_refs=set(
            target_feature_evidence_refs(projection["output"]["target_feature"])
        ),
        typed_record_resolver=lambda ref: _resolve_binding(root, view, ref),
    )
    presentation = load_evidence_presentation(root)
    investigation = _NativeEvidenceInvestigation(
        runtime=runtime,
        interaction_root=root,
        tbox=runtime._tbox,
        abox=abox,
        requirement=abox.product_requirement,
        handles=_approved_evidence_handles(presentation),
        presentation=presentation,
    )
    for ref in proposal.evidence_refs:
        investigation.register_canonical_reference(ref, kind="citation")
    for binding in view.typed_bindings:
        investigation.register_canonical_reference(binding.record_ref, kind="typed_record")
    # Reuse the pinned calibrated positions rather than capturing or relocalizing the part.
    prior_check = _read_json(root / previous["reachability_check_ref"])
    location_paths = {
        entry["evidence_handle"]: root / entry["location_record_ref"]
        for entries in prior_check["state_locations"].values()
        for entry in entries
    }
    output = dict(
        await runtime._complete_resource_assignment(
            product_agent=product_agent,
            investigation=investigation,
            root=root,
            tbox=runtime._tbox,
            abox=abox,
            view=view,
            proposal=proposal,
            ontology_projection_ref=projection_ref,
            requested_resource_symbol=requested_resource_symbol,
            location_paths=location_paths,
        )
    )
    # Historical recognition calls remain pinned, but superseded allocation calls must not
    # satisfy the new attempt's all-robot coverage checks.
    recognition_refs = [
        entry["ref"]
        for entry in previous["tool_call_refs"]
        if _read_json(root / entry["ref"]).get("record_type")
        not in {"ProductAgentAllocationToolCall", "ProductAgentAllocationExchange"}
    ]
    output["ontology_projection_ref"] = projection_ref
    output["tool_call_refs"] = recognition_refs + investigation.tool_call_refs
    turns = [
        int(path.stem.removeprefix("turn_"))
        for path in (root / "interaction_record").glob("turn_*.json")
    ]
    turn_number = max(turns, default=0) + 1
    turn_path = root / "interaction_record" / f"turn_{turn_number:04d}.json"
    _write_json(
        turn_path,
        {
            "turn": turn_number,
            "product_requirement": abox.product_requirement,
            "PA_input": {
                "mode": "resource_reassignment",
                "requested_resource_symbol": requested_resource_symbol,
                "previous_completion_fingerprint": previous["fingerprint"],
            },
            "PA_output": output,
            "failure": None if output.get("grounding_status") == "complete" else output,
        },
    )
    if output.get("grounding_status") != "complete":
        raise ProductionGroundingError(
            f"Resource reassignment did not complete: {output.get('grounding_validation_code')}"
        )
    # _complete_resource_assignment persisted the final view; reuse its exact timestamp/hash.
    from .grounding_contracts import _latest_product_context_view

    final_view = _latest_product_context_view(root)
    persist_pa_context_grounding_completion(
        root,
        tbox=runtime._tbox,
        product_requirement=abox.product_requirement,
        completion_turn=turn_number,
        decision_ref=turn_path.relative_to(root).as_posix(),
        product_context=final_view,
        ontology_projection_ref=projection_ref,
        resource_selection_ref=output["resource_selection_ref"],
        tool_call_refs=output["tool_call_refs"],
        registry=runtime._registry,
        workcell=runtime._workcell,
    )
    return output


def _resolve_binding(root: Path, view: ProductContextView, ref: str) -> dict[str, object]:
    matches = [binding for binding in view.typed_bindings if binding.record_ref == ref]
    if len(matches) != 1 or _sha256(root / ref) != matches[0].record_sha256:
        raise ProductionGroundingError("Saved allocation input is not one unchanged typed binding.")
    binding = matches[0]
    return {
        "record_type": binding.record_type,
        "record_sha256": binding.record_sha256,
        "record": _read_json(root / ref),
    }


def _assert_grounding_preserved(
    archive: Path, candidate: Path, completion: Mapping[str, object]
) -> None:
    refs = [entry["ref"] for entry in completion["typed_context_refs"]]
    refs.append(completion["ontology_projection_ref"])
    check = _read_json(archive / completion["reachability_check_ref"])
    refs.extend(
        entry["location_record_ref"]
        for entries in check["state_locations"].values()
        for entry in entries
    )
    for ref in refs:
        if _sha256(archive / ref) != _sha256(candidate / ref):
            raise ProductionGroundingError("Reassignment changed approved Phase 4 evidence.")


def _activate_candidate(root: Path, candidate: Path, retired: Path) -> None:
    root.rename(retired)
    try:
        candidate.rename(root)
        load_pa_context_grounding_completion(root)
    except (OSError, RuntimeError, TypeError, ValueError):
        if root.exists():
            root.rename(candidate)
        retired.rename(root)
        raise
    shutil.rmtree(retired)


def _file_hashes(root: Path) -> dict[str, str]:
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ProductionGroundingError("Reassignment does not follow interaction symlinks.")
    return {path.relative_to(root).as_posix(): _sha256(path) for path in paths if path.is_file()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path.name}.")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )

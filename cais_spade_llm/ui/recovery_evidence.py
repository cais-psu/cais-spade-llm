"""Read the recorded recovery stages without merging independent examples."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from cais_spade_llm.ui.evidence import contained_path, read_json_cached

DEBUG_ROOT = Path(__file__).resolve().parents[1] / "monitor/debug"
STAGES = {
    "recovery_outline": "recovery outline",
    "recovery_primitves": "primitive composition",
    "recovery_safety": "recovery safety",
}


def list_examples(root: Path = DEBUG_ROOT) -> list[dict[str, Any]]:
    """Index the requested stages and isolated tests without parsing sessions."""
    examples = []
    for stage in STAGES:
        base = root / stage
        folders = set()
        for pattern in ("*.json", "*.txt"):
            for path in base.rglob(pattern):
                parts = path.relative_to(base).parts
                folder = (
                    base.joinpath(*parts[: parts.index("requests")])
                    if "requests" in parts
                    else path.parent
                )
                folders.add(folder)
        for folder in sorted(folders):
            contained_path(root, folder)
            identifier = str(folder.relative_to(root))
            examples.append(
                {
                    "id": identifier,
                    "path": folder,
                    "stages": {stage: folder} if stage_records(folder, stage) else {},
                    "label": identifier,
                    "scenario": "not recorded",
                    "status": "saved example",
                }
            )
    for path in sorted((root / "test_runs").glob("*/run.json"), reverse=True):
        try:
            metadata = read_json_cached(contained_path(root, path))
            identifier = str(path.parent.relative_to(root))
            examples.append(
                {
                    "id": identifier,
                    "path": path.parent,
                    "stages": {stage: path.parent / stage for stage in STAGES},
                    "label": f"{path.parent.name} · {metadata.get('scenario', 'not recorded')} · {metadata.get('status', 'not recorded')}",
                    "scenario": metadata.get("scenario", "not recorded"),
                    "status": metadata.get("status", "not recorded"),
                    "response_source": metadata.get("response_source", "not recorded"),
                    "assisted": metadata.get("assisted"),
                    "evidence_source": metadata.get("evidence_source", "not recorded"),
                }
            )
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            examples.append(
                {
                    "id": str(path.parent.relative_to(root)),
                    "path": path.parent,
                    "stages": {},
                    "label": f"{path.parent.name} · unavailable",
                    "scenario": "not recorded",
                    "status": str(exc),
                }
            )
    return examples


def resolve_reference(directory: Path, reference: str) -> Path | None:
    """Resolve recorded paths inside this example, including uniquely moved files."""
    try:
        candidate = contained_path(directory, reference)
        if candidate.is_file():
            return candidate
    except (OSError, ValueError):
        pass
    # Saved examples were moved into numbered folders. An exact, unique basename
    # in this example can repair that pointer without reading another run.
    matches = [p for p in directory.rglob("*") if p.name == Path(reference).name and p.is_file()]
    if len(matches) == 1:
        try:
            return contained_path(directory, matches[0])
        except ValueError:
            pass
    return None


def stage_records(directory: Path, stage: str) -> list[Path]:
    """List records, preferring each outline audit over its summary copy."""
    paths = sorted(p for p in directory.glob("*") if p.suffix in {".txt", ".json"})
    if stage == "recovery_outline":
        chosen = [
            p
            for p in paths
            if p.name.startswith("multi_turn_turn")
            and any(
                t in p.name
                for t in (
                    "_result_",
                    "_audit_",
                    "_outline_response_",
                    "_grounding_response_",
                    "final_output_response_",
                )
            )
        ]
        chosen = [
            p
            for p in chosen
            if not (
                "_result_" in p.name and p.with_name(p.name.replace("_result_", "_audit_")).exists()
            )
        ]
    elif stage == "recovery_primitves":
        chosen = [
            p
            for p in paths
            if "_primitive_generation_" in p.name
            and "_response_" in p.name
            and "_llm_response_" not in p.name
        ]
    else:
        chosen = [
            p
            for p in paths
            if p.name == "recovery_safety_generation_result.json"
            or (p.name.startswith("recovery_safety_grounding_response_") and "latest" not in p.name)
        ]
    chosen.extend(sorted((directory / "requests").glob("*/*_request.json")))
    if not chosen and stage == "recovery_outline" and directory.name == "recovery_outline":
        checkpoint = directory.parent / "outline_checkpoint.json"
        if checkpoint.is_file() and (directory.parent / "run.json").is_file():
            return [contained_path(directory.parent, checkpoint)]
    names = {
        re.sub(r"_(?:latest|\d{8}T\d{6})(?=\.)", "", p.name)
        for p in chosen
        if "_latest." not in p.name
    }
    return [
        contained_path(directory, p)
        for p in chosen
        if "_latest." not in p.name or re.sub(r"_latest(?=\.)", "", p.name) not in names
    ]


def read_stage_record(  # noqa: C901 - retain legacy and exact evidence with separate provenance
    directory: Path, path: Path
) -> dict[str, Any]:
    """Read a record, exact prompt references, and recorded validation rows."""
    path = contained_path(directory, path)
    payload = read_json_cached(path)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    if path.name == "outline_checkpoint.json" and isinstance(
        payload.get("multi_turn_session"), dict
    ):
        payload = {**payload["multi_turn_session"], "outline_source": payload.get("outline_source")}
    references = dict(payload.get("artifact_paths") or {})
    references.update(
        {k: v for k, v in payload.items() if k.endswith("_artifact_path") and isinstance(v, str)}
    )
    prompt_refs = {
        k: v
        for k, v in references.items()
        if ("prompt" in k or "request" in k) and "latest" not in k
    }
    if not prompt_refs:
        for source, target in (
            ("_audit_", "_request_"),
            ("_result_", "_request_"),
            ("_response_", "_prompt_"),
        ):
            if source in path.name:
                prompt_refs["recorded prompt"] = path.name.replace(source, target).replace(
                    ".json", ".txt"
                )
                break
    prompts = [
        (reference, resolve_reference(directory, reference))
        for reference in dict.fromkeys(prompt_refs.values())
    ]
    response = payload.get("response") or payload.get("llm_response") or {}
    if not isinstance(response, dict):
        response = {"response": response}
    candidates = payload.get("candidate_evaluation_summary") or []
    if not isinstance(candidates, list) or any(not isinstance(row, dict) for row in candidates):
        raise ValueError("candidate_evaluation_summary must contain recorded candidate objects")
    detailed_candidates = payload.get("candidate_validation_records") or candidates
    validations = list(payload.get("validation_stages") or [])
    for candidate in detailed_candidates:
        for row in candidate.get("validation_stages") or []:
            validations.append(
                {
                    "candidate_id": candidate.get("candidate_id", candidate.get("candidate_index")),
                    **row,
                }
            )
    rules = payload.get("all_rule_results") or payload.get("rules") or []
    trace = (
        payload.get("transition_trace")
        or payload.get("accepted_outline_prefix")
        or payload.get("accepted_trace")
        or payload.get("outline_tasks")
        or []
    )
    if not trace and isinstance(payload.get("selected_transition"), dict):
        trace = [payload["selected_transition"]]
    linked = []
    for reference in payload.get("dfa_dot_files") or []:
        if isinstance(reference, str):
            linked.append((reference, resolve_reference(directory, reference)))
    captures = []
    record_refs = list(payload.get("request_record_paths") or [])
    if payload.get("kind") == "provider_request":
        record_refs = [str(p) for p in sorted(path.parent.glob("*.json"))]
    for reference in record_refs:
        # Exact records require an explicit contained reference, never a latest file
        # or the guessed basename repair used for legacy explanatory artifacts.
        try:
            recorded_path = contained_path(directory, reference)
            record = read_json_cached(recorded_path)
        except (OSError, ValueError):
            captures.append({"path": reference, "status": "not captured", "record": {}})
            continue
        captures.append({"path": str(recorded_path), "status": "captured", "record": record})
    return {
        "payload": payload,
        "response": response,
        "prompts": prompts,
        "trace": trace,
        "candidates": candidates,
        "validations": validations,
        "validation_details": detailed_candidates,
        "captures": captures,
        "rules": rules,
        "dfa": linked,
    }

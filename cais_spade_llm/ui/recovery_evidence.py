"""Read the recorded recovery stages without merging independent examples."""

from __future__ import annotations

import json
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


def _turn_header(path: Path) -> dict:
    with path.open("rb") as stream:
        head = stream.read(4096).decode("utf-8", errors="replace")
        stream.seek(max(0, path.stat().st_size - 4096))
        tail = stream.read(4096).decode("utf-8", errors="replace")
    fields = {}
    for name in ("turn", "turn_index", "phase", "stage", "decision", "status", "call_id", "outline_id", "resource_jid"):
        match = re.search(r'"' + name + r'"\s*:\s*("[^"\n]*"|\d+)', head)
        if match:
            fields[name] = json.loads(match[1])
    if not fields.get("call_id"):
        # Legacy primitive summaries place explicit capture references last.
        # Read only the bounded tail until the user expands this turn.
        references = re.findall(r'"request_record_paths"\s*:\s*(\[[^\]]*\])', tail)
        if references:
            paths = json.loads(references[-1])
            if paths:
                fields["call_id"] = Path(paths[0]).parent.name
    if re.search(r'"nominal_continuation_artifact_version"\s*:\s*1\b', tail):
        fields["nominal_continuation_artifact_version"] = 1
    return fields


def recovery_turns(directory: Path, stage: str) -> list[dict[str, Any]]:
    """Index turn headers without loading large provider or audit payloads."""
    records = set(stage_records(directory, stage))
    records.update(directory.glob("turns/turn_*.json"))
    records.update(directory.glob("multi_turn_turn*_request_*.txt"))
    grouped: dict[tuple, dict] = {}
    for raw_path in sorted(records):
        path = contained_path(directory, raw_path)
        fields = _turn_header(path)
        match = re.search(r"turn_?(\d+)", path.name)
        number = fields.get("turn", int(match[1]) if match else fields.get("turn_index"))
        if number is None:
            continue
        phase = fields.get("phase", fields.get("stage"))
        if phase is None:
            phase = "grounding" if "grounding" in path.name else "primitive_generation" if stage == "recovery_primitves" else "outline"
        owner = fields.get("call_id") or fields.get("outline_id", "")
        key = (number, phase)
        if phase == "primitive_generation" and owner:
            key = (phase, owner) if fields.get("call_id") else (number, phase, owner)
        row = grouped.setdefault(key, {"turn": number, "phase": phase, "status": "waiting for response", "files": [],
                                       "owner": owner if phase == "primitive_generation" else "",
                                       "call_id": fields.get("call_id") if phase == "primitive_generation" else None,
                                       "recorded_order": path.stat().st_mtime_ns})
        row["recorded_order"] = min(row["recorded_order"], path.stat().st_mtime_ns)
        if path.name.endswith("_request.json"):
            row["turn"] = number
        if fields.get("resource_jid") and phase == "primitive_generation":
            row["resource_jid"] = fields["resource_jid"]
        row["files"].append(str(path))
        if fields.get("decision"):
            row["status"] = fields["decision"]
        elif row["status"] == "waiting for response" and path.name.endswith("_request.json"):
            response = path.with_name(path.name.replace("_request.json", "_response.json"))
            error = path.with_name(path.name.replace("_request.json", "_error.json"))
            if response.exists():
                row["status"] = "response received"
            elif error.exists():
                row["status"] = "request failed"
        elif row["status"] == "waiting for response" and any(value in path.name for value in ("_result_", "_audit_", "_response_")):
            row["status"] = "recorded"
    for row in grouped.values():
        related = {Path(name) for name in row["files"]}
        for path in list(related):
            if path.name.endswith("_request.json"):
                related.update(path.parent.glob("*.json"))
        row["signature"] = [(str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(related)]
    return sorted(grouped.values(), key=lambda row: (row["recorded_order"], row["turn"]))


def read_recovery_turn(directory: Path, turn: dict) -> dict:
    """Read an expanded turn, preserving exact attempts and legacy provenance."""
    captures, results, prompts = {}, [], []
    nominal_proofs: dict[tuple[str, str], dict] = {}
    artifact_records: dict[Path, dict] = {}
    paths = [contained_path(directory, name) for name in turn["files"]]
    # Summaries verify and load their full records once before indexed copies
    # are displayed. Header inspection remains bounded for collapsed turns.
    paths.sort(key=lambda path: bool(_turn_header(path).get("nominal_continuation_artifact_version")))
    for path in paths:
        if path.suffix == ".txt" and any(token in path.name for token in ("_request_", "_prompt_")):
            prompts.append({"path": str(path), "text": path.read_text(encoding="utf-8")})
            continue
        if path.name.endswith("_request.json"):
            for raw_path in sorted(path.parent.glob("*.json")):
                capture = contained_path(directory, raw_path)
                captures[str(capture)] = {"path": str(capture), "record": read_json_cached(capture), "status": "captured"}
        else:
            payload = artifact_records.get(path)
            if payload is None:
                payload = read_json_cached(path)
            result = _stage_record(directory, path, payload)
            _nominal_continuation_record(directory, result, nominal_proofs, artifact_records)
            results.append({"path": str(path), **result})
            for _, prompt in result["prompts"]:
                if prompt and not any(row["path"] == str(prompt) for row in prompts):
                    prompts.append({"path": str(prompt), "text": prompt.read_text(encoding="utf-8")})
            for capture in result["captures"]:
                captures[capture["path"]] = capture
    ordered = sorted(captures.values(), key=lambda row: (
        row["record"].get("recorded_at", ""), row["record"].get("call_id", ""),
        row["record"].get("tool_round", 0), row["record"].get("attempt", 0), row["path"]))
    return {"captures": ordered, "results": results, "prompts": prompts}


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
    runtime_root = root.parent / "environment_runs"
    metadata_paths = [*runtime_root.glob("recovery/*/run.json"), *runtime_root.glob("*/recovery/run.json")]
    for metadata_path in sorted(metadata_paths, reverse=True):
        metadata = read_json_cached(metadata_path)
        for generation in sorted(metadata_path.parent.glob("generation_*"), reverse=True):
            identifier = str(generation.relative_to(root.parent))
            examples.append({
                "id": identifier, "path": generation,
                "stages": {stage: generation / stage for stage in STAGES if (generation / stage).is_dir()},
                "label": f"{metadata.get('run_id', metadata_path.parent.parent.name)} · {generation.name} · Part slippage",
                "scenario": "Part slippage", "status": metadata.get("status", "not recorded"),
                "response_source": "live LLM captures", "assisted": None,
                "evidence_source": "Gazebo — safety checks deferred",
            })
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
    result = _stage_record(directory, path, read_json_cached(path))
    _nominal_continuation_record(directory, result, {}, {})
    return result


def _stage_record(directory: Path, path: Path, payload: Any) -> dict[str, Any]:
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
        "path": str(path),
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


def _nominal_continuation_record(directory: Path, result: dict,
                                 proofs: dict[tuple[str, str], dict],
                                 artifact_records: dict[Path, dict]) -> None:
    """Resolve archived nominal evidence for display without changing saved data."""
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (
        read_nominal_continuation,
    )

    payload = result["payload"]
    nested = payload.get("validation_evidence")
    owner = nested if isinstance(nested, dict) and "nominal_continuation_evidence" in nested else payload
    nominal = owner.get("nominal_continuation_evidence")
    if not isinstance(nominal, dict):
        return
    reference = nominal.get("artifact_reference")
    if "artifact_reference" in nominal:
        key, identity = None, None
        try:
            if not isinstance(reference, dict):
                raise ValueError("Nominal continuation artifact reference must be an object")
            path = contained_path(directory, reference.get("path", ""))
            key = (str(path), str(reference.get("sha256", "")))
            identity = json.dumps(nominal, sort_keys=True, ensure_ascii=False)
            cached = proofs.get(key)
            if cached is not None:
                if cached["summary_identity"] != identity:
                    raise ValueError("Nominal continuation references disagree within this turn")
                display = cached
            else:
                proof = read_nominal_continuation(nominal, directory=directory,
                                                  artifact_records=artifact_records)
                display = {"path": str(path), "sha256": reference["sha256"], "status": "verified",
                           "proof": proof, "summary_identity": identity}
                proofs[key] = display
        except (OSError, ValueError, TypeError) as exc:
            display = {"path": str(reference.get("path", "")) if isinstance(reference, dict) else "",
                       "status": "unavailable", "reason": str(exc)}
            if key is not None and identity is not None:
                display.update(path=key[0], sha256=key[1], summary_identity=identity)
                proofs[key] = display
        result["nominal_continuation_artifact"] = display
    elif payload.get("nominal_continuation_artifact_version") == 1:
        path = str(Path(result["path"]).resolve())
        display = next((value for key, value in proofs.items() if key[0] == path), None)
        result["nominal_continuation_artifact"] = display or {
            "path": path, "status": "recorded", "proof": nominal,
        }
    else:
        return
    # Keep full records in the per-expansion result, but render a large proof
    # only through its dedicated expansion rather than every summary copy.
    summary = {key: nominal[key] for key in ("complete", "status", "reason", "execution_authorized") if key in nominal}
    if result["nominal_continuation_artifact"]["status"] == "unavailable":
        summary = {"status": "unavailable", "reason": result["nominal_continuation_artifact"]["reason"]}
    if owner is nested:
        result["payload"] = {**payload, "validation_evidence": {**nested, "nominal_continuation_evidence": summary}}
    else:
        result["payload"] = {**payload, "nominal_continuation_evidence": summary}

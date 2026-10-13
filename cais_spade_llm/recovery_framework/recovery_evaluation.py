from __future__ import annotations

"""Frozen, isolated outline evaluation using saved Gazebo observations."""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.diagnostics import read_object, write_record

logger = logging.getLogger(__name__)
CASES = (
    "Conveyor breakdown", "ur5e-1 breakdown",
    "Machining breakdown during part processing", "Part slippage",
)
MODEL = {"model": "gpt-5.6-sol", "reasoning_effort": "low"}
FORBIDDEN = (
    "spade.agent.Agent.start", "spade.behaviour.CyclicBehaviour.send",
    "cais_spade_llm.recovery_framework.recovery_motion_preview.preview_recovery_motion",
    "cais_spade_llm.recovery_framework.recovery_motion_preview.capture_preview_scene",
    "cais_spade_llm.recovery_framework.recovery_motion_preview._new_node",
    "cais_spade_llm.recovery_framework.failure_effects.observe_robot",
    "cais_spade_llm.recovery_framework.failure_effects._robot",
    "cais_spade_llm.recovery_framework.failure_effects._observe_part",
    "cais_spade_llm.recovery_framework.recovery_observations.capture_recovery_physical_evidence",
    "cais_spade_llm.recovery_framework.kmr_live_safety.capture_state",
    "cais_spade_llm.agents.resource_agent.robot_agent.RobotAgent._build_controller",
    "cais_spade_llm.resources.robot.gazebo_pick_place_controller.GazeboPickPlaceController.__init__",
    "cais_spade_llm.agents.resource_agent.robot_agent.RobotAgent._execute_primitive",
    "cais_spade_llm.agents.resource_agent.resource_agent.ResourceAgent.execute_recovery_composition_step",
    "cais_spade_llm.agents.resource_agent.resource_agent.ResourceAgent.execute_recovery_macro",
    "cais_spade_llm.recovery_framework.part_slippage_recovery.PartSlippageRecovery._execute",
    "cais_spade_llm.recovery_framework.part_slippage_recovery.PartSlippageRecovery._generate_primitives",
)


def file_hash(path: Path) -> str:
    """Hash original bytes using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_hash(value: object) -> str:
    """Fingerprint exact JSON values without changing identifiers."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _implementation_files(root: Path) -> set[Path]:
    files = {root / "pyproject.toml", root / "poetry.lock"}
    package = root / "cais_spade_llm"
    files.update(path for path in package.rglob("*.py") if "monitor" not in path.relative_to(package).parts)
    for directory in (package / "initialization", package / "specification", root / "ros2"):
        files.update(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower()
                     in {".py", ".json", ".txt", ".yaml", ".yml", ".cpp", ".hpp", ".h", ".action",
                         ".urdf", ".srdf", ".xacro", ".world", ".sdf", ".stl", ".dae", ".obj", ".xml", ".pgm"})
    return files


def frozen_files(captures: list[dict], *, root: Path = ROOT) -> dict[str, str]:
    """Freeze implementation, declarations, specifications, and saved model files."""
    files = _implementation_files(root)
    for capture in captures:
        source = Path(capture["source"]).resolve()
        files.add(source)
        files.update(path for path in (source.parent / "native_preview_model").rglob("*") if path.is_file())
        files.update(Path(path) for path in capture.get("evidence_files", []))
    return {str(path.relative_to(root) if path.is_relative_to(root) else path): file_hash(path)
            for path in sorted(files)}


def verify_frozen(manifest: dict, *, root: Path = ROOT) -> None:
    """Refuse subsequent requests when any frozen input changed."""
    if "implementation_files" in manifest:
        current = sorted(str(path.relative_to(root)) for path in _implementation_files(root))
        if current != manifest["implementation_files"]:
            raise ValueError("Frozen implementation file inventory changed")
    for name, digest in manifest["files"].items():
        path = root / name
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError("Frozen evaluation input changed: " + name)


def verify_batch_manifest(manifest: dict) -> None:
    """Require the complete unchanged reserved protocol before any attempt."""
    fingerprint = manifest.get("manifest_fingerprint")
    payload = {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    if not isinstance(fingerprint, str) or fingerprint != value_hash(payload):
        raise ValueError("Frozen evaluation manifest changed or has no fingerprint")
    settings = {"schema_version": 1, "model_settings": MODEL, "max_turns": 80,
                "attempt_deadline_seconds": None, "attempts_per_condition": 3,
                "planned_attempts": 24, "operator_guidance_allowed": False,
                "execution_authorized": False, "status": "frozen"}
    if any(key not in manifest or type(manifest[key]) is not type(value)
           or manifest[key] != value for key, value in settings.items()):
        raise ValueError("Frozen evaluation protocol differs from the reserved plan")
    captures = manifest.get("captures")
    expected = {(scenario, condition) for scenario in CASES for condition in ("normal", "changed")}
    if (not isinstance(captures, list) or len(captures) != 8
            or any(not isinstance(row, dict) for row in captures)
            or {(row.get("scenario"), row.get("condition")) for row in captures} != expected):
        raise ValueError("Evaluation requires normal and changed captures for all four exact scenarios")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or manifest.get("implementation_fingerprint") != value_hash(files):
        raise ValueError("Frozen evaluation file manifest differs")
    sources = set()
    for capture in captures:
        source = Path(capture["source"]).resolve()
        if source in sources:
            raise ValueError("Each condition requires its own immutable observation capture")
        sources.add(source)
        native = source.parent / "native_preview_model/manifest.json"
        for path, digest in ((source, capture.get("source_sha256")),
                             (native, capture.get("native_preview_model_manifest_sha256"))):
            key = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
            if not isinstance(digest, str) or files.get(key) != digest:
                raise ValueError("Reserved capture differs from its frozen source or native model: " + str(path))
        condition = Path(capture["condition_record"]).resolve()
        key = str(condition.relative_to(ROOT) if condition.is_relative_to(ROOT) else condition)
        if key not in files:
            raise ValueError("Reserved capture condition is absent from the frozen file manifest")
    verify_frozen(manifest)


def _verify_manifest_contents(path: Path, manifest: dict) -> None:
    if value_hash(read_object(path)) != value_hash(manifest):
        raise ValueError("Frozen evaluation manifest changed during this run")


def prepare_batch(captures: list[dict], output: Path) -> dict:
    """Reserve eight distinct, recorded conditions before any provider call."""
    from cais_spade_llm.recovery_framework.recovery_observations import (
        offline_preparation_missing, read_native_preview_model, read_observations,
    )

    expected = {(scenario, condition) for scenario in CASES for condition in ("normal", "changed")}
    if len(captures) != 8 or {(row["scenario"], row["condition"]) for row in captures} != expected:
        raise ValueError("Evaluation requires normal and changed captures for all four exact scenarios")
    rows, sources = [], set()
    for row in captures:
        source = Path(row["source"]).resolve()
        if source in sources:
            raise ValueError("Each condition requires its own immutable observation capture")
        sources.add(source)
        facts = read_observations(source)
        if facts.get("scenario") != row["scenario"] or facts.get("evidence_source") != "Gazebo":
            raise ValueError("Evaluation requires scenario-matched Gazebo observations")
        missing = offline_preparation_missing(facts)
        if missing:
            raise ValueError("NEEDS_CONTEXT: " + ", ".join(missing))
        prepared = facts.get("prepared_recovery_request") or {}
        if (facts.get("operator_assistance") or prepared.get("operator_assistance")
                or prepared.get("recovery_feedback")):
            raise ValueError("Operator-guided observations are excluded from unassisted evaluation")
        # This record is an evaluator artifact and is never copied into generation inputs.
        condition_record = read_object(Path(row["condition_record"]))
        if condition_record.get("purpose") != "evaluation":
            raise ValueError("Development captures cannot be promoted into the reserved evaluation")
        implementation_path = Path(condition_record["implementation_manifest"]).resolve()
        implementation = read_object(implementation_path)
        verify_frozen(implementation)
        if (condition_record.get("implementation_fingerprint") != value_hash(implementation["files"])
                or implementation["files"] != frozen_files([])):
            raise ValueError("Reserved capture was produced by a different implementation")
        if (condition_record.get("source_sha256") != file_hash(source)
                or condition_record.get("scenario") != row["scenario"]
                or condition_record.get("condition") != row["condition"]
                or condition_record.get("capture_verified") is not True):
            raise ValueError("Condition evidence is not bound to the actual captured facts")
        from cais_spade_llm.recovery_framework.evaluation_conditions import verify_capture_condition

        actual_condition = verify_capture_condition(facts, row["scenario"], row["condition"])
        if condition_record.get("condition_evidence") != actual_condition:
            raise ValueError("Condition verification differs from the recorded facts")
        native = read_native_preview_model(source, facts)
        evidence_files = [str(Path(row["condition_record"]).resolve()), str(implementation_path),
                          native["manifest_path"], *native["files"]]
        rows.append({**row, "source": str(source), "source_sha256": file_hash(source),
                     "captured_at": facts.get("captured_at"), "source_run_id": facts.get("run_id"),
                     "native_preview_model_manifest_sha256": native["manifest_sha256"],
                     "evidence_files": evidence_files})
    files = frozen_files(rows)
    from ament_index_python.packages import get_package_prefix

    installed = Path(get_package_prefix("cais_lab_robotics"))
    native = [installed / "lib/cais_lab_robotics/preview_recovery_motion"]
    native += list((installed / "lib").glob("*rosidl*.so"))
    native += list((installed / "local/lib/python3.10/dist-packages/cais_lab_robotics").glob("*.so"))
    files.update({str(path): file_hash(path) for path in native})
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "captures": rows, "files": files, "implementation_fingerprint": value_hash(files),
                "implementation_files": sorted(str(path.relative_to(ROOT)) for path in _implementation_files(ROOT)),
                "model_settings": MODEL, "max_turns": 80, "attempt_deadline_seconds": None,
                "attempts_per_condition": 3, "planned_attempts": 24, "operator_guidance_allowed": False,
                "execution_authorized": False, "status": "frozen"}
    manifest["manifest_fingerprint"] = value_hash(manifest)
    output.mkdir(parents=True, exist_ok=False)
    write_record(output / "manifest.json", manifest)
    return manifest


def forbidden(*args, **kwargs):
    """Reject transport, observation, controller and execution calls offline."""
    raise RuntimeError("Offline evaluation attempted a live interface or a later recovery stage")


def capture_report(session, elapsed: float, error: str | None) -> tuple[dict, dict]:
    """Index exact evidence without loading large validation histories again.

    The separate recovery_outline_audit process must verify every accepted event and
    the nominal continuation before this attempt can count as complete.
    """
    state = session.snapshot()
    workflow = session._prepared.get("multi_turn_session_state") or {}
    directory = Path(state["artifact_directory"])
    directories = {directory / "recovery_outline"}
    for turn in workflow.get("turns") or []:
        for key, value in turn.items():
            if key.endswith("artifact_path") and isinstance(value, str):
                directories.update(parent for parent in Path(value).parents if parent.name == "recovery_outline")
    records, validation_artifacts = [], []
    for evidence_directory in sorted(directories):
        for path in sorted(evidence_directory.glob("requests/*/*.json")):
            raw = path.read_bytes()
            row = json.loads(raw)
            records.append({**row, "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        for path in sorted(evidence_directory.glob("turns/turn_*.json")):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            validation_artifacts.append({"path": str(path), "sha256": digest.hexdigest()})
    records.sort(key=lambda row: (row.get("turn") or 0, row.get("recorded_at") or "", row["path"]))
    requests = [row for row in records if row.get("kind") == "provider_request"]
    responses = [row for row in records if row.get("kind") == "provider_response"]
    errors = [row for row in records if row.get("kind") in {"provider_error", "parse_error", "tool_error"}]
    stage_error_path = directory / "stage_error.json"
    stage_error = read_object(stage_error_path) if stage_error_path.is_file() else {}
    error = stage_error.get("error") or error
    if error is None and state["status"] == "blocked":
        error = state.get("message")
    workflow_status = workflow.get("status")
    provider_outcomes = [row for row in records if row.get("kind") in {"provider_response", "provider_error"}]
    if state["status"] == "outline_review" and error is None:
        outcome = "outline_review"
    elif error and provider_outcomes and provider_outcomes[-1]["kind"] == "provider_error":
        outcome = "provider_error"
    elif workflow_status == "NEEDS_CONTEXT" or any(isinstance(message, str)
            and message.startswith(("NEEDS_CONTEXT:", "ValueError: NEEDS_CONTEXT:"))
            for message in (stage_error.get("error"), error)):
        outcome = "NEEDS_CONTEXT"
    elif error:
        outcome = "stage_error"
    else:
        outcome = workflow_status or state["status"]
    proposals, retrievals, provider_records = [], [], []
    for row in responses:
        provenance = {key: row.get(key) for key in ("path", "sha256", "turn", "recorded_at", "attempt", "tool_round")}
        provider_records.append({**provenance, "model": row["raw_response"].get("model"),
                                 "usage": row["raw_response"].get("usage")})
        for choice in row["raw_response"].get("choices", []):
            content = choice.get("message", {}).get("content")
            if not isinstance(content, str):
                continue
            try:
                response = json.loads(content)
            except json.JSONDecodeError:
                continue
            if not isinstance(response, dict):
                continue
            if isinstance(response.get("next_transition"), dict):
                proposals.append({**provenance, "event": response["next_transition"]})
            if response.get("decision") == "need_context":
                retrievals.append({**provenance, "context_requests": response.get("context_requests")})
    usage = {key: sum((row.get("usage") or {}).get(key, 0) or 0 for row in provider_records)
             for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    nominal = workflow.get("nominal_continuation_evidence") or {}
    nominal_summary = {key: nominal[key] for key in ("status", "complete", "reason", "execution_authorized", "artifact_reference") if key in nominal}
    prefix = workflow.get("accepted_outline_prefix") or []
    validation = []
    for turn in workflow.get("turns") or []:
        stages = []
        for stage in turn.get("validation_stages") or []:
            evidence = stage.get("evidence") or {}
            physical = evidence.get("geometry_evidence") or {}
            stages.append({"validator_role": stage.get("validator_role"),
                "validator_jid": stage.get("validator_jid"),
                **{key: evidence[key] for key in ("feasibility_status", "allowed", "is_safe", "reason",
                    "validation_status", "missing_context", "candidate_fingerprint") if key in evidence},
                "physical": {key: physical[key] for key in ("status", "reason", "unresolved", "interval_complete") if key in physical}})
        validation.append({"turn_index": turn.get("turn_index", turn.get("turn")),
            "decision": turn.get("decision"), "stages": stages,
            "findings": deepcopy(turn.get("validation_findings", turn.get("outline_validation_findings", []))),
            "retrievals": deepcopy(turn.get("validation_context_requests", []))})
    lineage = {"complete": False, "audit_status": "pending independent saved-prefix audit",
        "provider_request_records": [{"path": row["path"], "sha256": row["sha256"]} for row in requests],
        "provider_records": provider_records, "validation_artifacts": validation_artifacts,
        "evidence_directories": [str(path) for path in sorted(directories)],
        "llm_proposals": proposals, "retrieval_requests": retrievals,
        "accepted_events": [{"event": event, "validation_status": "pending independent audit"} for event in prefix],
        "nominal_continuation_evidence": nominal_summary,
        "nominal_evidence_source": "session.json#/nominal_continuation_evidence",
        "usage": usage, "usage_complete": bool(responses) and all(row.get("usage") for row in provider_records),
        "execution_authorized": False}
    return {"run_id": state["run_id"], "artifact_directory": str(directory), "status": state["status"],
        "session_status": workflow_status, "workflow_status": workflow_status,
        "outcome": outcome, "message": state.get("message"), "error": error,
        "stage_error": ({"path": str(stage_error_path), "sha256": file_hash(stage_error_path),
                         "record": stage_error} if stage_error_path.is_file() else None),
        "elapsed_seconds": elapsed, "complete": False,
        "runtime_nominal_complete": nominal.get("complete") is True,
        "provider_requests": len(requests), "provider_responses": len(responses), "provider_errors": errors,
        "validation": validation,
        "llm_proposals": proposals, "retrieval_requests": retrievals, "accepted_prefix": prefix,
        "usage": usage, "usage_complete": lineage["usage_complete"],
        "nominal_continuation_evidence": nominal_summary,
        "prompt_sizes": [{"turn": row["turn"], "phase": row["stage"],
            "characters": sum(len(message.get("content") or "") for message in row["payload"]["messages"])} for row in requests],
        "validation_contract_version": session._prepared.get("validation_contract_version"),
        "primitive_support_semantics": session._prepared.get("primitive_support_semantics"),
        "public_safety_context_version": session._prepared.get("public_safety_context_version"),
        "physical_preview_contract_version": session._prepared.get("physical_evidence", {}).get("physical_preview_contract_version"),
        "execution_authorized": False}, lineage


async def run_attempt(manifest_path: Path, capture_index: int, trial: int, output: Path) -> dict:
    """Run one fresh production Offline Generate with no overall deadline."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report_path = output / 'report.json'
    initial = {'pid': os.getpid(), 'attempt_directory': str(output), 'capture_index': capture_index,
               'trial': trial, 'started_at': datetime.now(timezone.utc).isoformat(),
               'generation_enabled': False, 'execution_authorized': False, 'live_provider_calls': [],
               'result': {'status': 'initializing', 'complete': False, 'execution_authorized': False}}
    write_record(report_path, initial)
    try:
        return await _run_attempt(manifest_path, capture_index, trial, output)
    except (asyncio.CancelledError, OSError, ValueError, KeyError, TypeError, RuntimeError,
            ImportError, IndexError, AttributeError, AssertionError) as exc:
        report = read_object(report_path)
        error = f'{type(exc).__name__}: {exc}'
        report.update(error=error, elapsed_seconds=time.monotonic() - started,
                      terminal_generation_status='cancelled' if isinstance(exc, asyncio.CancelledError) else 'stage_error')
        report['result'] = {**report.get('result', {}), 'status': report['terminal_generation_status'],
                            'outcome': report['terminal_generation_status'],
                            'error': error, 'complete': False, 'execution_authorized': False}
        write_record(report_path, report)
        if isinstance(exc, asyncio.CancelledError):
            raise
        logger.exception('Evaluation attempt initialization or reporting failed')
        return report


async def _run_attempt(manifest_path: Path, capture_index: int, trial: int, output: Path) -> dict:
    """Initialize and generate inside the attempt's retained error boundary."""
    from cais_spade_llm.agents.shared_information import llm_agent
    from cais_spade_llm.agents.shared_information.llm_request_records import _REQUEST_CONTEXT, _response_payload
    from cais_spade_llm.recovery_framework.offline_recovery import create_offline_recovery
    from cais_spade_llm.recovery_framework.recovery_observations import (
        observation_fingerprint, read_native_preview_model, read_observations,
    )

    manifest = read_object(manifest_path)
    verify_batch_manifest(manifest)
    if manifest['model_settings'] != MODEL or manifest['max_turns'] != 80 or not 1 <= trial <= 3:
        raise ValueError('Evaluation settings differ from the frozen plan')
    if not 0 <= capture_index < len(manifest['captures']):
        raise ValueError('Evaluation capture index is outside the reserved captures')
    capture = manifest['captures'][capture_index]
    source = Path(capture['source'])
    initial = read_object(output / 'report.json')
    initial.update({key: capture[key] for key in ('source', 'source_sha256', 'scenario', 'condition')})
    write_record(output / 'report.json', initial)
    observations = read_observations(source)
    native = read_native_preview_model(source, observations)
    if native['manifest_sha256'] != capture['native_preview_model_manifest_sha256']:
        raise ValueError('Native preview companion differs from the reserved capture')
    files = manifest['files']
    native_files = {name: digest for name, digest in files.items()
                    if 'ros2/' in name or 'ros2_ws/' in name or 'native_preview_model/' in name}
    report = {
        'source': str(source), 'source_sha256': capture['source_sha256'],
        'parent_source': str(source), 'parent_source_sha256': capture['source_sha256'],
        'scenario': capture['scenario'], 'condition': capture['condition'], 'trial': trial,
        'fresh_attempt': True, 'generation_enabled': True, 'generation_path': 'production Offline Generate',
        'external_validation_retries': 0, 'pid': os.getpid(), 'attempt_directory': str(output),
        'model': MODEL['model'], 'reasoning_effort': MODEL['reasoning_effort'], 'max_turns': 80,
        'generation_deadline_seconds': None, 'timed_out': False,
        'source_observation_fingerprint': observation_fingerprint(observations),
        'native_preview_model_manifest_sha256': native['manifest_sha256'],
        'code_sha256': files, 'native_code_sha256': native_files,
        'implementation_fingerprint': manifest['implementation_fingerprint'],
        'evaluation_manifest': {'path': str(manifest_path), 'fingerprint': manifest.get('manifest_fingerprint')},
        'started_at': datetime.now(timezone.utc).isoformat(), 'live_provider_calls': [],
        'execution_authorized': False, 'unassisted': True,
    }
    started = time.monotonic()
    session, error = None, None
    client = llm_agent._client.with_options(timeout=120, max_retries=0)
    original_create = client.chat.completions.create

    def create(**kwargs):
        _verify_manifest_contents(manifest_path, manifest)
        verify_frozen(manifest)
        if any(kwargs.get(key) != value for key, value in MODEL.items()):
            raise ValueError('Provider request settings differ from the frozen evaluation')
        context = dict(_REQUEST_CONTEXT.get() or {})
        begin = time.monotonic() - started
        response = original_create(**kwargs)
        report['live_provider_calls'].append({
            'request_sent': True, 'response_source': 'live', 'pid': os.getpid(),
            'attempt_directory': str(output), **{key: context.get(key) for key in ('run_id', 'stage', 'turn')},
            'started_elapsed_seconds': begin, 'returned_elapsed_seconds': time.monotonic() - started,
            'request_fingerprint': value_hash(kwargs), 'response_fingerprint': value_hash(_response_payload(response)),
            'request_id': getattr(response, '_request_id', None),
        })
        write_record(output / 'provider_receipts.json', {'calls': report['live_provider_calls']})
        return response

    with ExitStack() as stack:
        guards = [stack.enter_context(patch(path, side_effect=forbidden)) for path in FORBIDDEN]
        stack.enter_context(patch.object(llm_agent, '_client', client))
        stack.enter_context(patch.object(client.chat.completions, 'create', side_effect=create))
        try:
            session = create_offline_recovery(observations, source_observation_path=source,
                outline_mode='manual', primitive_mode='manual', **MODEL)
            session.begin(observations['failure'], generate_outline=False)
            await session._task
            if session.snapshot()['status'] != 'outline_ready':
                raise ValueError(session.snapshot().get('message', 'Input preparation did not finish'))
            seed = session._prepared['multi_turn_session_seed']
            if seed.get('accepted_outline_prefix') or seed.get('turn_index') != 0:
                raise ValueError('Evaluation must begin with a fresh empty outline')
            report['fresh_session'] = {
                'run_id': session.snapshot()['run_id'], 'artifact_directory': session.snapshot()['artifact_directory'],
                'session_id': session._prepared['recovery_session']['session_id'],
            }
            write_record(output / 'report.json', report)
            session.action('generate_outline')
            await session._task
            report['terminal_generation_elapsed_seconds'] = time.monotonic() - started
            report['terminal_generation_status'] = session.snapshot()['status']
        except asyncio.CancelledError:
            error = 'CancelledError: evaluation stopped'
            raise
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as exc:
            error = f'{type(exc).__name__}: {exc}'
            logger.exception('Evaluation attempt failed')
        finally:
            if session is not None:
                if session._task is not None and not session._task.done():
                    session.stop(reason=error or 'evaluation cleanup')
                    try:
                        await session._task
                    except asyncio.CancelledError:
                        pass
                report['result'], report['lineage'] = capture_report(session, time.monotonic() - started, error)
                error = report['result']['error']
                write_record(output / 'session.json', session._prepared.get('multi_turn_session_state') or {})
                report['unassisted'] = not (session.snapshot().get('operator_assistance')
                    or session._prepared.get('operator_assistance') or session._prepared.get('recovery_feedback'))
            report.update(error=error, generation_stopped_elapsed_seconds=time.monotonic() - started,
                elapsed_seconds=time.monotonic() - started, forbidden_calls=sum(item.call_count for item in guards),
                source_unchanged=file_hash(source) == capture['source_sha256'],
                parent_source_unchanged=file_hash(source) == capture['source_sha256'])
            try:
                _verify_manifest_contents(manifest_path, manifest)
                verify_frozen(manifest)
                report.update(code_unchanged=True, native_code_unchanged=True)
            except ValueError as exc:
                report.update(code_unchanged=False, native_code_unchanged=False, freeze_error=str(exc))
            write_record(output / 'report.json', report)
    return report


@contextmanager
def _cancel_on_sigterm():
    """Route termination through owned-process cancellation and restore handlers."""
    previous = signal.getsignal(signal.SIGTERM)

    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt("Evaluation received SIGTERM")

    signal.signal(signal.SIGTERM, cancel)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_batch(manifest_path: Path) -> dict:
    """Run every reserved attempt sequentially and audit after process cleanup."""
    existing = manifest_path.parent / 'results.json'
    if existing.is_file():
        raise ValueError('This evaluation batch already started; retain it and use a new batch')
    try:
        with _cancel_on_sigterm():
            return _run_batch(manifest_path)
    except (KeyboardInterrupt, asyncio.CancelledError, SystemExit, OSError, ValueError,
            KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        retained = read_object(existing) if existing.is_file() else {'results': []}
        retained.update(status='cancelled' if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError, SystemExit)) else 'failed',
                        error=f'{type(exc).__name__}: {exc}',
                        attempts=len(retained.get('results', [])), execution_authorized=False)
        write_record(existing, retained)
        raise


def _run_batch(manifest_path: Path) -> dict:
    """Retain each reaped attempt before checking the next frozen input."""
    manifest = read_object(manifest_path)
    verify_batch_manifest(manifest)
    output = manifest_path.parent
    results = []
    existing = output / 'results.json'
    if existing.is_file():
        raise ValueError('This evaluation batch already started; retain it and use a new batch')
    write_record(existing, {'status': 'running', 'results': results})
    for index, capture in enumerate(manifest['captures']):
        for trial in range(1, 4):
            _verify_manifest_contents(manifest_path, manifest)
            verify_frozen(manifest)
            directory = output / f'condition_{index + 1:02d}_attempt_{trial}'
            directory.mkdir()
            attempt = directory / 'attempt'
            command = [sys.executable, '-m', 'cais_spade_llm.recovery_framework.recovery_evaluation', '--attempt', str(manifest_path),
                       '--capture-index', str(index), '--trial', str(trial), '--output', str(attempt)]
            start = time.monotonic()
            with (directory / 'generation.log').open('w') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                returncode, cancelled, hard_timeout = None, False, False
                try:
                    returncode = process.wait()
                except (KeyboardInterrupt, asyncio.CancelledError, SystemExit, OSError, subprocess.SubprocessError) as exc:
                    # Reap the owned group before a cancellation leaves this batch.
                    cancelled = isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError, SystemExit))
                    try:
                        os.killpg(process.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                    try:
                        returncode = process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        hard_timeout = True
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        returncode = process.wait()
                    raise
                finally:
                    write_record(directory / 'process_result.json', {
                        'pid': process.pid, 'attempt_directory': str(attempt), 'returncode': returncode,
                        'hard_timeout': hard_timeout, 'cancelled': cancelled,
                        'generation_deadline_seconds': None, 'cancellation_completed': returncode is not None,
                        'source': capture['source'], 'source_sha256': capture['source_sha256'],
                        'elapsed_seconds': time.monotonic() - start,
                    })
            audit_path = directory / 'audit.json'
            with (directory / 'audit.log').open('w') as stream:
                audit = subprocess.run([sys.executable, '-m',
                    'cais_spade_llm.recovery_framework.recovery_outline_audit', str(attempt),
                    '--output', str(audit_path)], stdout=stream, stderr=subprocess.STDOUT, check=False)
            report = read_object(attempt / 'report.json') if (attempt / 'report.json').is_file() else {}
            verified = read_object(audit_path) if audit_path.is_file() else {}
            result = report.get('result') or {}
            row = {key: capture[key] for key in ('scenario', 'condition', 'source', 'source_sha256')}
            row.update(trial=trial, attempt_directory=str(attempt), returncode=returncode,
                audit_returncode=audit.returncode, audit=str(audit_path),
                complete=(returncode == 0 and audit.returncode == 0 and verified.get('outline_complete') is True
                          and report.get('unassisted') is True and not verified.get('forbidden_calls')),
                status=result.get('status') or report.get('terminal_generation_status') or 'stage_error',
                workflow_status=result.get('workflow_status', result.get('session_status')),
                outcome=result.get('outcome') or result.get('status') or 'stage_error',
                error=report.get('error') or result.get('error'), message=result.get('message'),
                stage_error=result.get('stage_error'), proposed_events=result.get('llm_proposals', []),
                accepted_events=result.get('accepted_prefix', []), retrievals=result.get('retrieval_requests', []),
                provider_errors=result.get('provider_errors', []), usage=result.get('usage'),
                validation=result.get('validation', []),
                elapsed_seconds=report.get('elapsed_seconds'), audit_errors=verified.get('errors', []),
                nominal_continuation=result.get('nominal_continuation_evidence'), execution_authorized=False)
            results.append(row)
            write_record(existing, {'status': 'running', 'results': results})
            _verify_manifest_contents(manifest_path, manifest)
            verify_frozen(manifest)
    summary = {'status': 'completed', 'attempts': len(results), 'complete_outlines': sum(row['complete'] for row in results),
               'results': results, 'execution_authorized': False}
    write_record(existing, summary)
    return summary


def main() -> None:
    """Freeze a capture manifest, run a reserved batch, or run one owned child."""
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--freeze', type=Path, help='JSON object with the eight captures')
    operation.add_argument('--run', type=Path, help='Frozen manifest.json')
    operation.add_argument('--attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--capture-index', type=int)
    parser.add_argument('--trial', type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    if args.freeze:
        if args.output is None:
            parser.error('--freeze requires --output')
        prepare_batch(read_object(args.freeze)['captures'], args.output.resolve())
    elif args.run:
        run_batch(args.run.resolve())
    else:
        with _cancel_on_sigterm():
            asyncio.run(run_attempt(args.attempt.resolve(), args.capture_index, args.trial, args.output.resolve()))


if __name__ == '__main__':
    main()

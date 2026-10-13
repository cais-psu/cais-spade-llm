from __future__ import annotations

"""Frozen condition coverage, process records and independent evaluation audits."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from cais_spade_llm.recovery_framework import recovery_evaluation as evaluation
from cais_spade_llm.recovery_framework import recovery_outline_audit as audit
from cais_spade_llm.recovery_framework import evaluation_conditions as conditions


def test_freeze_detects_input_change_and_missing_files(tmp_path):
    path = tmp_path / "declaration.json"
    path.write_text('{"part_name": null}')
    manifest = {"files": {str(path): evaluation.file_hash(path)}}
    evaluation.verify_frozen(manifest)
    path.write_text('{"part_name": "KET4_Square_4mm"}')
    with pytest.raises(ValueError, match="Frozen evaluation input changed"):
        evaluation.verify_frozen(manifest)
    path.unlink()
    with pytest.raises(ValueError, match="Frozen evaluation input changed"):
        evaluation.verify_frozen(manifest)


def test_freeze_rejects_added_implementation_files(tmp_path):
    package = tmp_path / "cais_spade_llm"
    package.mkdir()
    (tmp_path / "pyproject.toml").write_text("configuration")
    (tmp_path / "poetry.lock").write_text("lock")
    (package / "owner.py").write_text("value = 1\n")
    paths = evaluation._implementation_files(tmp_path)
    manifest = {"files": {str(path.relative_to(tmp_path)): evaluation.file_hash(path) for path in paths},
                "implementation_files": sorted(str(path.relative_to(tmp_path)) for path in paths)}
    evaluation.verify_frozen(manifest, root=tmp_path)
    (package / "new_validator.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="inventory changed"):
        evaluation.verify_frozen(manifest, root=tmp_path)


@pytest.mark.parametrize("change", ["content", "added", "removed"])
def test_freeze_retains_nav2_map_pixels_and_inventory(tmp_path, change):
    (tmp_path / "pyproject.toml").write_text("configuration")
    (tmp_path / "poetry.lock").write_text("lock")
    directory = tmp_path / "ros2/cais_lab_robotics/config"
    directory.mkdir(parents=True)
    map_path = directory / "recovery_framework_map.pgm"
    map_path.write_bytes(b"P5\n2 1\n255\n\xff\x00")
    (directory / "recovery_framework_map.yaml").write_text("image: recovery_framework_map.pgm\n")
    paths = evaluation._implementation_files(tmp_path)
    assert map_path in paths
    manifest = {"files": evaluation.frozen_files([], root=tmp_path),
                "implementation_files": sorted(str(path.relative_to(tmp_path)) for path in paths)}
    evaluation.verify_frozen(manifest, root=tmp_path)
    if change == "content":
        map_path.write_bytes(b"P5\n2 1\n255\n\x00\x00")
    elif change == "added":
        (directory / "additional_map.pgm").write_bytes(map_path.read_bytes())
    else:
        map_path.unlink()
    expected = "Frozen evaluation input changed" if change == "content" else "inventory changed"
    with pytest.raises(ValueError, match=expected):
        evaluation.verify_frozen(manifest, root=tmp_path)


@pytest.mark.parametrize("count", [0, 1, 7, 9])
def test_batch_requires_all_eight_conditions_before_provider_access(tmp_path, count):
    rows = [{"scenario": scenario, "condition": condition, "source": "unused"}
            for scenario in evaluation.CASES for condition in ("normal", "changed")]
    rows = (rows + rows)[:count]
    with pytest.raises(ValueError, match="all four exact scenarios"):
        evaluation.prepare_batch(rows, tmp_path / "batch")
    assert not (tmp_path / "batch").exists()


def test_development_capture_cannot_be_relabelled_as_reserved_evaluation(tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import recovery_observations

    rows = []
    for index, (scenario, condition) in enumerate(
            (scenario, condition) for scenario in evaluation.CASES for condition in ("normal", "changed")):
        source = tmp_path / f"capture_{index}.json"
        source.write_text(json.dumps({"scenario": scenario, "evidence_source": "Gazebo"}))
        condition_path = tmp_path / f"condition_{index}.json"
        condition_path.write_text(json.dumps({"purpose": "development"}))
        rows.append({"scenario": scenario, "condition": condition, "source": str(source),
                     "condition_record": str(condition_path)})
    monkeypatch.setattr(recovery_observations, "read_observations", lambda source: json.loads(source.read_text()))
    monkeypatch.setattr(recovery_observations, "offline_preparation_missing", lambda facts: [])
    with pytest.raises(ValueError, match="Development captures"):
        evaluation.prepare_batch(rows, tmp_path / "batch")
    assert not (tmp_path / "batch").exists()


def test_fresh_completion_has_no_ten_minute_or_other_attempt_deadline():
    report = {"terminal_generation_elapsed_seconds": 7200., "generation_deadline_seconds": None,
              "terminal_generation_status": "outline_review", "timed_out": False, "error": None,
              "source": "capture", "source_sha256": "original"}
    process = {"returncode": 0, "hard_timeout": False, "cancellation_completed": True,
               "generation_deadline_seconds": None, "source": "capture", "source_sha256": "original"}
    assert audit.completion_within_deadline(report, process)
    for field, value in (("returncode", 1), ("hard_timeout", True), ("cancellation_completed", False)):
        assert not audit.completion_within_deadline(report, {**process, field: value})


def test_event_authorship_keeps_null_and_provider_identifiers():
    event = {"outline_id": "r1", "event_name": "owner_proposed_event", "resource_jid": "M1",
             "part_name": None, "expected_end_state": {"resource_state": "idle"}, "rationale": "Observed fact"}
    assert audit.authored_match(event, deepcopy(event))
    changed = deepcopy(event)
    changed.pop("part_name")
    assert not audit.authored_match(event, changed)
    changed["part_name"] = "KET4_Square_4mm"
    assert not audit.authored_match(event, changed)


def test_changed_machine_condition_cannot_be_claimed_from_missing_requirements():
    capture = {"scenario": "Machining breakdown during part processing", "evidence_source": "Gazebo",
               "setup": {"execution_mode": "simulation"},
               "failure": {"injection_status": "completed", "part_name": "KET4_Square_4mm"},
               "inputs": {"scene": {"machines": [{"resource_id": "M2", "current_configuration": {}}]}},
               "requirements": {"KET4_Square_4mm": []}}
    with pytest.raises(ValueError, match="NEEDS_CONTEXT"):
        conditions.verify_capture_condition(capture, capture["scenario"], "changed")


def test_recorded_kinematic_bound_includes_mobile_base_limits():
    robot = '''<robot name="recorded"><joint name="KMR_base_x_joint" type="prismatic">
      <parent link="world"/><child link="base"/><limit lower="-20" upper="20"/>
      </joint><joint name="arm" type="fixed"><parent link="base"/><child link="tcp"/>
      <origin xyz="1 0 0"/></joint></robot>'''
    assert conditions._kinematic_radius(robot) == 21.
    with pytest.raises(ValueError, match="unsupported mobile joint"):
        conditions._kinematic_radius(robot.replace('type="prismatic"', 'type="planar"'))


def test_isolated_configuration_allows_only_removing_declared_programs():
    from cais_spade_llm.ui.recovery_setup import _declared_program_subset

    original = {"machines": [{"resource_id": "M2", "current_configuration": {"program": "circle"},
                             "program_options": {"square": {"program": "square"}}}], "robots": ["ur5e-2"]}
    selected = deepcopy(original)
    selected["machines"][0]["program_options"] = {}
    assert _declared_program_subset(original, selected)
    selected["robots"] = ["ur5e-1"]
    assert not _declared_program_subset(original, selected)
    selected = deepcopy(original)
    selected["machines"][0]["program_options"]["square"] = {"program": "unknown"}
    assert not _declared_program_subset(original, selected)


def test_all_attempts_are_retained_even_when_every_attempt_is_unresolved(tmp_path, monkeypatch):
    captures = [{"scenario": scenario, "condition": condition, "source": "capture", "source_sha256": "original"}
                for scenario in evaluation.CASES for condition in ("normal", "changed")]
    manifest = tmp_path / "manifest.json"
    evaluation.write_record(manifest, {"captures": captures, "files": {}})
    calls = []

    class Process:
        pid = 123

        def __init__(self, command, **kwargs):
            calls.append(command)
            attempt = Path(command[command.index("--output") + 1])
            attempt.mkdir()
            evaluation.write_record(attempt / "report.json", {"result": {
                "status": "blocked", "outcome": "NEEDS_CONTEXT",
                "session_status": "NEEDS_CONTEXT", "accepted_prefix": [], "llm_proposals": [],
                "nominal_continuation_evidence": {"complete": False}}, "error": None, "unassisted": True})

        def wait(self):
            return 0

    def replay(command, **kwargs):
        destination = Path(command[command.index("--output") + 1])
        evaluation.write_record(destination, {"outline_complete": False, "errors": []})
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(evaluation.subprocess, "Popen", Process)
    monkeypatch.setattr(evaluation.subprocess, "run", replay)
    monkeypatch.setattr(evaluation, "verify_batch_manifest", evaluation.verify_frozen)
    result = evaluation.run_batch(manifest)
    assert result["attempts"] == len(calls) == 24
    assert result["complete_outlines"] == 0
    assert all(row["status"] == "blocked" and row["outcome"] == "NEEDS_CONTEXT"
               and row["workflow_status"] == "NEEDS_CONTEXT" for row in result["results"])
    assert all(command[2] == "cais_spade_llm.recovery_framework.recovery_evaluation" for command in calls)
    assert len(list(tmp_path.glob("condition_*/process_result.json"))) == 24
    with pytest.raises(ValueError, match="already started"):
        evaluation.run_batch(manifest)


def test_freeze_preflights_every_native_companion_before_publishing(tmp_path, monkeypatch):
    from cais_spade_llm.recovery_framework import recovery_observations

    implementation = tmp_path / "implementation.json"
    evaluation.write_record(implementation, {"files": {}})
    captures = []
    for index, (scenario, condition) in enumerate(
        (scenario, condition) for scenario in evaluation.CASES for condition in ("normal", "changed")
    ):
        source = tmp_path / str(index) / "observations.json"
        source.parent.mkdir()
        facts = {"scenario": scenario, "evidence_source": "Gazebo", "prepared_recovery_request": {}}
        evaluation.write_record(source, facts)
        record = source.parent / "condition.json"
        evaluation.write_record(record, {"purpose": "evaluation", "implementation_manifest": str(implementation),
            "implementation_fingerprint": evaluation.value_hash({}), "source_sha256": evaluation.file_hash(source),
            "scenario": scenario, "condition": condition, "capture_verified": True, "condition_evidence": {}})
        captures.append({"source": str(source), "scenario": scenario, "condition": condition,
                         "condition_record": str(record)})
    calls = []

    def native(source, facts):
        calls.append(source)
        if len(calls) == 8:
            raise ValueError("Native preview source/model binding differs")
        return {"manifest_path": str(source.parent / "native_preview_model/manifest.json"),
                "manifest_sha256": "recorded", "files": {}}

    monkeypatch.setattr(recovery_observations, "read_observations", evaluation.read_object)
    monkeypatch.setattr(recovery_observations, "offline_preparation_missing", lambda facts: [])
    monkeypatch.setattr(recovery_observations, "read_native_preview_model", native)
    monkeypatch.setattr(conditions, "verify_capture_condition", lambda *args: {})
    monkeypatch.setattr(evaluation, "frozen_files", lambda rows: {})
    with pytest.raises(ValueError, match="source/model binding"):
        evaluation.prepare_batch(captures, tmp_path / "batch")
    assert calls == [Path(row["source"]) for row in captures]
    assert not (tmp_path / "batch").exists()


def test_native_initialization_error_retains_exact_failure_without_provider_call(tmp_path, monkeypatch):
    import asyncio

    from cais_spade_llm.recovery_framework import recovery_observations
    from cais_spade_llm.agents.shared_information import llm_agent

    source = tmp_path / "observations.json"
    evaluation.write_record(source, {"failure": {}})
    capture = {"source": str(source), "source_sha256": evaluation.file_hash(source),
               "scenario": "Conveyor breakdown", "condition": "normal"}
    manifest = tmp_path / "manifest.json"
    evaluation.write_record(manifest, {"files": {}, "model_settings": evaluation.MODEL,
        "max_turns": 80, "captures": [capture]})
    monkeypatch.setattr(recovery_observations, "read_observations", evaluation.read_object)

    def invalid(*args):
        raise ValueError("Native preview captured model fingerprint changed")

    class NoProvider:
        def with_options(self, **kwargs):
            pytest.fail("Provider must not be initialized after missing native evidence")

    monkeypatch.setattr(recovery_observations, "read_native_preview_model", invalid)
    monkeypatch.setattr(llm_agent, "_client", NoProvider())
    monkeypatch.setattr(evaluation, "verify_batch_manifest", evaluation.verify_frozen)
    destination = tmp_path / "attempt"
    report = asyncio.run(evaluation.run_attempt(manifest, 0, 1, destination))
    assert report["error"] == "ValueError: Native preview captured model fingerprint changed"
    assert report["result"]["status"] == "stage_error"
    assert report["result"]["complete"] is False
    assert report["source"] == str(source)
    assert report["live_provider_calls"] == []
    assert evaluation.read_object(destination / "report.json") == report


@pytest.mark.parametrize("force_kill", [False, True])
@pytest.mark.parametrize("sigterm", [False, True])
def test_batch_cancellation_reaps_and_records_owned_process(tmp_path, monkeypatch, force_kill, sigterm):
    import signal
    import subprocess

    manifest = tmp_path / "manifest.json"
    evaluation.write_record(manifest, {"files": {}, "captures": [{"scenario": "Conveyor breakdown",
        "condition": "normal", "source": "recorded", "source_sha256": "original"}]})
    waits, signals = [], []
    original_sigterm = signal.getsignal(signal.SIGTERM)

    class Process:
        pid = 4321

        def __init__(self, command, **kwargs):
            self.command = command

        def wait(self, timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                if sigterm:
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                raise KeyboardInterrupt()
            if force_kill and len(waits) == 2:
                raise subprocess.TimeoutExpired(self.command, timeout)
            return -9 if force_kill else -2

    monkeypatch.setattr(evaluation.subprocess, "Popen", Process)
    monkeypatch.setattr(evaluation.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(evaluation, "verify_batch_manifest", evaluation.verify_frozen)
    with pytest.raises(KeyboardInterrupt):
        evaluation.run_batch(manifest)
    record = evaluation.read_object(tmp_path / "condition_01_attempt_1/process_result.json")
    assert record["cancellation_completed"] is True
    assert record["cancelled"] is True
    assert record["hard_timeout"] is force_kill
    assert record["returncode"] == (-9 if force_kill else -2)
    assert signals == [(4321, signal.SIGINT)] + ([(4321, signal.SIGKILL)] if force_kill else [])
    assert evaluation.read_object(tmp_path / "results.json")["status"] == "cancelled"
    assert not (tmp_path / "condition_01_attempt_2").exists()
    assert signal.getsignal(signal.SIGTERM) == original_sigterm


def test_batch_freeze_failure_is_terminal_and_recorded(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    evaluation.write_record(manifest, {"files": {"missing.py": "original"}, "captures": [
        {"scenario": "Part slippage", "condition": "normal", "source": "recorded", "source_sha256": "original"}]})
    monkeypatch.setattr(evaluation.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Changed code must stop the batch"))
    monkeypatch.setattr(evaluation, "verify_batch_manifest", evaluation.verify_frozen)
    with pytest.raises(ValueError, match="Frozen evaluation input changed"):
        evaluation.run_batch(manifest)
    result = evaluation.read_object(tmp_path / "results.json")
    assert result["status"] == "failed"
    assert result["attempts"] == 0
    assert "missing.py" in result["error"]


@pytest.mark.parametrize(("preparation_error", "expected_error", "expected_outcome"), [
    (False, "transport failed after prior turn", "provider_error"),
    (True, "missing declared snapshot", "stage_error"),
    (True, "NEEDS_CONTEXT: /resources/recorded_resource/recovery_snapshot", "NEEDS_CONTEXT"),
])
def test_runtime_caught_error_retains_terminal_status_and_original_details(
        tmp_path, monkeypatch, preparation_error, expected_error, expected_outcome):
    import asyncio
    from types import SimpleNamespace

    from openai import OpenAIError

    from cais_spade_llm.agents.shared_information import llm_agent
    from cais_spade_llm.agents.shared_information.llm_request_records import RequestRecords, capture_requests
    from cais_spade_llm.recovery_framework import offline_recovery, recovery_observations
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery

    source = tmp_path / "observations.json"
    evaluation.write_record(source, {"failure": {}})
    capture = {"source": str(source), "source_sha256": evaluation.file_hash(source),
               "scenario": "Conveyor breakdown", "condition": "normal",
               "native_preview_model_manifest_sha256": "recorded"}
    manifest = tmp_path / "manifest.json"
    evaluation.write_record(manifest, {"files": {}, "model_settings": evaluation.MODEL,
        "max_turns": 80, "captures": [capture], "implementation_fingerprint": "frozen"})
    artifact_directory = tmp_path / "generation_0001"
    artifact_directory.mkdir()
    provider_calls = []

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def with_options(self, **kwargs):
            return self

        def create(self, **kwargs):
            provider_calls.append(kwargs)
            if len(provider_calls) == 2:
                raise OpenAIError(expected_error)
            return {"model": evaluation.MODEL["model"], "choices": [{"message": {"content": json.dumps({
                "thought": "The recorded state needs more information.", "decision": "need_context",
                "context_requests": ["/resources"]})}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    class Session:
        _schedule = PartSlippageRecovery._schedule
        _task_finished = PartSlippageRecovery._task_finished
        _record_error = PartSlippageRecovery._record_error

        def __init__(self):
            self._task = None
            self._stopped = False
            self._prepared = {"multi_turn_session_seed": {"accepted_outline_prefix": [], "turn_index": 0},
                              "recovery_session": {"session_id": "fresh_0001"}}
            self._state = {"run_id": "fresh", "artifact_directory": str(artifact_directory),
                           "status": "preparing_outline", "outline_status": "preparing", "primitive_status": "waiting"}

        def snapshot(self):
            return deepcopy(self._state)

        def _publish(self, **kwargs):
            self._state.update(kwargs)

        def begin(self, failure, *, generate_outline):
            async def prepare():
                if preparation_error:
                    raise ValueError(expected_error)
                self._publish(status="outline_ready", outline_status="ready")

            self._schedule(prepare)

        def action(self, action):
            assert action == "generate_outline"

            async def generate():
                self._publish(status="generating_outline", outline_status="generating")
                for turn in (1, 2):
                    with capture_requests(directory=artifact_directory / "recovery_outline", run_id="fresh_0001",
                                          stage="outline", turn=turn):
                        records = RequestRecords({})
                        records.submit(llm_agent._client.chat.completions.create,
                            {**evaluation.MODEL, "messages": [{"role": "user", "content": "Recorded facts"}]},
                            tool_round=0, attempt=1)
                    self._prepared["multi_turn_session_state"] = {"status": "paused_after_outline_turn", "turns": []}

            self._schedule(generate)

    session = Session()
    monkeypatch.setattr(evaluation, "FORBIDDEN", ())
    monkeypatch.setattr(evaluation, "verify_batch_manifest", evaluation.verify_frozen)
    monkeypatch.setattr(offline_recovery, "create_offline_recovery", lambda *args, **kwargs: session)
    monkeypatch.setattr(recovery_observations, "read_observations", evaluation.read_object)
    monkeypatch.setattr(recovery_observations, "observation_fingerprint", lambda facts: "recorded")
    monkeypatch.setattr(recovery_observations, "read_native_preview_model", lambda *args: {"manifest_sha256": "recorded"})
    monkeypatch.setattr(llm_agent, "_client", Client())
    report = asyncio.run(evaluation.run_attempt(manifest, 0, 1, tmp_path / "attempt"))
    result = report["result"]
    assert result["status"] == "blocked"
    assert result["workflow_status"] == (None if preparation_error else "paused_after_outline_turn")
    assert result["outcome"] == expected_outcome
    recorded_error = ("ValueError: " if preparation_error else "OpenAIError: ") + expected_error
    assert report["error"] == result["error"] == recorded_error
    assert result["message"] == recorded_error
    assert result["stage_error"]["record"]["error"] == recorded_error
    assert result["stage_error"]["sha256"] == evaluation.file_hash(artifact_directory / "stage_error.json")
    assert result["complete"] is False
    if preparation_error:
        assert not provider_calls and not result["provider_errors"]
    else:
        assert len(provider_calls) == 2 and result["provider_responses"] == 1
        assert len(result["provider_errors"]) == 1
        assert result["provider_errors"][0]["error_type"] == "OpenAIError"
        assert result["retrieval_requests"][0]["context_requests"] == ["/resources"]


def test_failed_attempt_summary_preserves_terminal_and_workflow_status(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    evaluation.write_record(manifest, {"files": {}, "captures": [{"scenario": "Conveyor breakdown",
        "condition": "normal", "source": "recorded", "source_sha256": "original"}]})
    outcomes = [
        {"outcome": "provider_error", "session_status": "paused_after_outline_turn", "error": "provider failed"},
        {"outcome": "stage_error", "session_status": None, "error": "input preparation failed"},
        {"outcome": "NEEDS_CONTEXT", "session_status": "NEEDS_CONTEXT", "error": "NEEDS_CONTEXT: base motion evidence"},
    ]

    class Process:
        pid = 123

        def __init__(self, command, **kwargs):
            trial = int(command[command.index("--trial") + 1])
            attempt = Path(command[command.index("--output") + 1])
            attempt.mkdir()
            result = {"status": "blocked", "message": outcomes[trial - 1]["error"], **outcomes[trial - 1]}
            evaluation.write_record(attempt / "report.json", {"result": result, "error": result["error"], "unassisted": True})

        def wait(self):
            return 0

    def replay(command, **kwargs):
        destination = Path(command[command.index("--output") + 1])
        evaluation.write_record(destination, {"outline_complete": False, "errors": []})
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(evaluation.subprocess, "Popen", Process)
    monkeypatch.setattr(evaluation.subprocess, "run", replay)
    monkeypatch.setattr(evaluation, "verify_batch_manifest", evaluation.verify_frozen)
    rows = evaluation.run_batch(manifest)["results"]
    for row, expected in zip(rows, outcomes, strict=True):
        assert row["status"] == "blocked" and row["complete"] is False
        assert row["outcome"] == expected["outcome"]
        assert row["workflow_status"] == expected["session_status"]
        assert row["error"] == row["message"] == expected["error"]


def test_unresolved_attempt_keeps_rejected_proposal_separate(tmp_path):
    from types import SimpleNamespace

    evidence = {"feasibility_status": "INFEASIBLE", "allowed": False,
                "reason": "Recorded custody conflicts with the proposed start condition"}
    workflow = {"status": "NEEDS_CONTEXT", "turns": [{"turn_index": 2, "decision": "need_revision",
        "validation_stages": [{"validator_role": "ResourceAgent", "validator_jid": "recorded_resource",
                               "evidence": evidence}]}]}
    state = {"run_id": "fresh", "artifact_directory": str(tmp_path), "status": "blocked",
             "message": "NEEDS_CONTEXT: base motion evidence"}
    evaluation.write_record(tmp_path / "stage_error.json", {"error": state["message"]})
    session = SimpleNamespace(snapshot=lambda: deepcopy(state), _prepared={"multi_turn_session_state": workflow})
    result, _ = evaluation.capture_report(session, 2.0, None)
    assert result["outcome"] == "NEEDS_CONTEXT"
    assert result["validation"][0]["stages"][0]["feasibility_status"] == "INFEASIBLE"
    assert result["validation"][0]["stages"][0]["reason"] == evidence["reason"]
    assert result["complete"] is False


def _reserved_manifest(tmp_path):
    captures, files = [], {}
    for index, (scenario, condition) in enumerate(
        (scenario, condition) for scenario in evaluation.CASES for condition in ("normal", "changed")
    ):
        directory = tmp_path / f"capture_{index}"
        (directory / "native_preview_model").mkdir(parents=True)
        source = directory / "observations.json"
        native = directory / "native_preview_model/manifest.json"
        condition_path = directory / "condition_record.json"
        for path in (source, native, condition_path):
            path.write_text(json.dumps({"scenario": scenario, "condition": condition}))
            files[str(path)] = evaluation.file_hash(path)
        captures.append({"scenario": scenario, "condition": condition, "source": str(source),
            "source_sha256": files[str(source)], "native_preview_model_manifest_sha256": files[str(native)],
            "condition_record": str(condition_path)})
    manifest = {"schema_version": 1, "captures": captures, "files": files,
        "implementation_fingerprint": evaluation.value_hash(files), "model_settings": evaluation.MODEL,
        "max_turns": 80, "attempt_deadline_seconds": None, "attempts_per_condition": 3,
        "planned_attempts": 24, "operator_guidance_allowed": False,
        "execution_authorized": False, "status": "frozen"}
    manifest["manifest_fingerprint"] = evaluation.value_hash(manifest)
    return manifest


def test_run_boundary_requires_all_eight_frozen_conditions_and_exact_protocol(tmp_path):
    manifest = _reserved_manifest(tmp_path)
    evaluation.verify_batch_manifest(manifest)
    changed = deepcopy(manifest)
    changed["captures"].pop()
    with pytest.raises(ValueError, match="manifest changed"):
        evaluation.verify_batch_manifest(changed)
    changed.pop("manifest_fingerprint")
    changed["manifest_fingerprint"] = evaluation.value_hash(changed)
    with pytest.raises(ValueError, match="all four exact scenarios"):
        evaluation.verify_batch_manifest(changed)
    for field, value in (("max_turns", 16), ("attempt_deadline_seconds", 600),
                         ("operator_guidance_allowed", True), ("model_settings", {"model": "gpt-4o"})):
        changed = {**manifest, field: value}
        changed.pop("manifest_fingerprint")
        changed["manifest_fingerprint"] = evaluation.value_hash(changed)
        with pytest.raises(ValueError, match="protocol differs"):
            evaluation.verify_batch_manifest(changed)


def test_capture_metadata_cannot_select_another_frozen_source(tmp_path):
    manifest = _reserved_manifest(tmp_path)
    changed = deepcopy(manifest)
    changed["captures"][0]["source_sha256"] = "different source"
    changed.pop("manifest_fingerprint")
    changed["manifest_fingerprint"] = evaluation.value_hash(changed)
    with pytest.raises(ValueError, match="frozen source or native model"):
        evaluation.verify_batch_manifest(changed)


def test_partial_manifest_stops_batch_before_spawning_any_attempt(tmp_path, monkeypatch):
    manifest = _reserved_manifest(tmp_path)
    manifest["captures"].pop()
    manifest.pop("manifest_fingerprint")
    manifest["manifest_fingerprint"] = evaluation.value_hash(manifest)
    path = tmp_path / "manifest.json"
    evaluation.write_record(path, manifest)
    monkeypatch.setattr(evaluation.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Partial batch launched"))
    with pytest.raises(ValueError, match="all four exact scenarios"):
        evaluation.run_batch(path)
    assert evaluation.read_object(tmp_path / "results.json")["attempts"] == 0


def test_partial_manifest_stops_direct_child_before_provider_access(tmp_path, monkeypatch):
    import asyncio

    from cais_spade_llm.agents.shared_information import llm_agent
    from cais_spade_llm.recovery_framework import recovery_observations

    manifest = _reserved_manifest(tmp_path)
    manifest["captures"].pop()
    manifest.pop("manifest_fingerprint")
    manifest["manifest_fingerprint"] = evaluation.value_hash(manifest)
    path = tmp_path / "manifest.json"
    evaluation.write_record(path, manifest)

    class NoProvider:
        def with_options(self, **kwargs):
            pytest.fail("Partial batch initialized provider")

    monkeypatch.setattr(llm_agent, "_client", NoProvider())
    monkeypatch.setattr(recovery_observations, "read_native_preview_model",
                        lambda *args: pytest.fail("Partial batch prepared a native model"))
    report = asyncio.run(evaluation.run_attempt(path, 0, 1, tmp_path / "attempt"))
    assert "all four exact scenarios" in report["error"]
    assert report["generation_enabled"] is False
    assert report["live_provider_calls"] == []


def test_manifest_change_during_run_is_rejected_even_with_recomputed_fingerprint(tmp_path):
    manifest = _reserved_manifest(tmp_path)
    path = tmp_path / "manifest.json"
    evaluation.write_record(path, manifest)
    evaluation._verify_manifest_contents(path, manifest)
    changed = deepcopy(manifest)
    changed["captures"] = list(reversed(changed["captures"]))
    changed.pop("manifest_fingerprint")
    changed["manifest_fingerprint"] = evaluation.value_hash(changed)
    evaluation.write_record(path, changed)
    with pytest.raises(ValueError, match="manifest changed during"):
        evaluation._verify_manifest_contents(path, manifest)

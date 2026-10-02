"""Saved-context diagnostic inputs, outline reuse, and subprocess lifecycle."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from cais_spade_llm.recovery_framework import scenario_runner as runner

from cais_spade_llm.recovery_framework import diagnostics

CONTEXT = diagnostics.ROOT / "test/fixtures/part_slippage/runtime_context.json"
SCENARIO = diagnostics.ROOT / "cais_spade_llm/initialization/failure_scenarios/part_slippage.json"
TRACE = [
    {
        "outline_id": "RECOVERY_SEQ1",
        "event_name": "recover_to_home",
        "resource_jid": "recovery-resource-3@localhost",
    }
]


def _checkpoint(path: Path) -> dict:
    data = {
        "input_fingerprint": diagnostics.inspect_inputs(CONTEXT)["fingerprint"],
        "validator_fingerprint": diagnostics.validator_fingerprint(),
        "experiment_settings": {},
        "multi_turn_session": {
            "current_phase": "primitive_generation",
            "status": "ready_for_primitive_generation",
            "transition_trace": TRACE,
        },
    }
    diagnostics.write_record(path, data)
    return data


def test_inputs_validate_scenario_task_and_resource_identity(tmp_path):
    original = diagnostics.read_object(CONTEXT)
    path = tmp_path / "runtime_context.json"
    for change, message in (
        ({"failure_scenario_id": "move_home_failure"}, "does not match"),
        ({"failure_event": {"failed_task_id": "unknown"}}, "not in the selected plan"),
        ({"task_statuses": {"UNKNOWN": "failed"}}, "unknown task IDs"),
    ):
        diagnostics.write_record(path, {**original, **change})
        with pytest.raises(ValueError, match=message):
            diagnostics.inspect_inputs(path, SCENARIO)
    changed = deepcopy(original)
    changed["resource_snapshots"][0]["resource_jid"] = "unrelated@localhost"
    diagnostics.write_record(path, changed)
    with pytest.raises(ValueError, match="identity"):
        diagnostics.inspect_inputs(path)


def test_other_failure_context_uses_its_matching_task(tmp_path):
    data = diagnostics.read_object(CONTEXT)
    data["failure_scenario_id"] = "move_home_failure"
    data["failure_event"]["failed_task_id"] = "REQ_2_T5"
    data["task_statuses"]["REQ_2_T4"] = "completed"
    data["task_statuses"]["REQ_2_T5"] = "failed"
    path = tmp_path / "runtime_context.json"
    diagnostics.write_record(path, data)
    inspected = diagnostics.inspect_inputs(path)
    assert inspected["scenario"] == "move_home_failure"
    assert inspected["runtime_context"]["failure_event"]["failed_task_id"] == "REQ_2_T5"
    assert inspected["fingerprint"] != diagnostics.inspect_inputs(CONTEXT)["fingerprint"]


@pytest.mark.parametrize("mode", ["primitive", "safety", "full"])
def test_stage_reuses_accepted_outline_without_outline_generation(tmp_path, monkeypatch, mode):
    source = tmp_path / "source.json"
    _checkpoint(source)
    before = source.read_bytes()
    output = tmp_path / mode
    output.mkdir()
    prepared = {"recovery_outline_experiment_settings": {}}
    prepare = AsyncMock(
        return_value=({}, SimpleNamespace(turn_log=[]), SimpleNamespace(), prepared)
    )
    monkeypatch.setattr(runner, "_prepare_recovery_dryrun_harness", prepare)
    calls = []

    async def execute(planner, request, *, stop_after):
        assert stop_after == "primitive"
        assert request["multi_turn_session_state"]["transition_trace"] == TRACE
        calls.append(stop_after)
        request["multi_turn_session_state"]["accepted_primitive_program"] = [
            {
                "outline_id": "RECOVERY_SEQ1",
                "primitive_steps": [
                    {"primitive": "move_to_named_pose", "params": {"pose_name": "home"}}
                ],
            },
        ]
        return None

    safety = AsyncMock(
        return_value={
            "ok": True,
            "recovery_safety_status": "ready",
            "accepted_outline_prefix": TRACE,
            "rule_ids": ["SAFE_2"],
        }
    )
    monkeypatch.setattr(runner, "_execute_recovery_until", execute)
    monkeypatch.setattr(runner, "_run_safety_from_outline", safety)
    monkeypatch.setattr(runner, "_write_recovery_final_bundle", lambda **kwargs: {})
    result = asyncio.run(
        runner._run_actual_recovery(
            mode=mode,
            debug_root=output,
            runtime_context_path=CONTEXT,
            outline_checkpoint=source,
        )
    )
    assert calls == ([] if mode == "safety" else ["primitive"])
    assert safety.await_count == (mode in {"safety", "full"})
    assert result["transition_trace"] == TRACE
    assert result["runtime_context_source"] == str(CONTEXT)
    assert source.read_bytes() == before
    assert (output / "outline_checkpoint.json").is_file()


def test_outline_reuse_rejects_changed_inputs_and_validators(tmp_path, monkeypatch):
    source = tmp_path / "outline.json"
    _checkpoint(source)
    inputs = diagnostics.inspect_inputs(CONTEXT)
    with pytest.raises(ValueError, match="inputs differ"):
        diagnostics.load_outline_checkpoint(source, {**inputs, "fingerprint": "changed"})
    monkeypatch.setattr(diagnostics, "validator_fingerprint", lambda root: "changed")
    with pytest.raises(ValueError, match="validator code has changed"):
        diagnostics.load_outline_checkpoint(source, inputs)


@pytest.mark.parametrize(
    "returncode, expected", [(0, "completed"), (2, "failed"), (None, "cancelled")]
)
def test_jobs_are_explicit_isolated_single_and_preserve_history(
    tmp_path, monkeypatch, returncode, expected
):
    async def exercise():
        release = asyncio.Event()

        class Process:
            pid = 12345
            returncode = None

            async def wait(self):
                await release.wait()
                self.returncode = -15 if returncode is None else returncode
                return self.returncode

            def terminate(self):
                release.set()

            def kill(self):
                release.set()

        process = Process()
        monkeypatch.setattr(diagnostics.os, "killpg", lambda pid, sig: release.set())
        launch = AsyncMock(return_value=process)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
        jobs = diagnostics.DiagnosticJobs(runs=tmp_path / "runs")
        assert not jobs.runs.exists()
        output = await jobs.start("outline", CONTEXT, SCENARIO)
        command = launch.call_args.args
        assert command[1:3] == ("-m", "cais_spade_llm.recovery_framework.scenario_runner")
        assert "--runtime-context" in command and "--debug-root" in command
        assert launch.call_args.kwargs.get("shell") is None
        assert (output / "inputs/runtime_context.json").read_bytes() == CONTEXT.read_bytes()
        assert diagnostics.read_object(output / "run.json")["status"] == "running"
        with pytest.raises(ValueError, match="already running"):
            await jobs.start("outline", CONTEXT, SCENARIO)
        if returncode is None:
            await jobs.cancel()
        else:
            release.set()
            await jobs.task
        assert diagnostics.read_object(output / "run.json")["status"] == expected
        original = (output / "run.json").read_bytes()
        second = await jobs.start("outline", CONTEXT, SCENARIO)
        await jobs.task
        assert second != output
        assert (output / "run.json").read_bytes() == original

    asyncio.run(exercise())


def test_restart_marks_only_abandoned_jobs_interrupted(tmp_path):
    abandoned, active, completed = (
        tmp_path / name for name in ("abandoned", "active", "completed")
    )
    for directory in (abandoned, active, completed):
        directory.mkdir()
    diagnostics.write_record(abandoned / "run.json", {"status": "running", "owner_pid": -1})
    diagnostics.write_record(
        active / "run.json", {"status": "running", "owner_pid": diagnostics.os.getpid()}
    )
    diagnostics.write_record(completed / "run.json", {"status": "completed"})
    jobs = diagnostics.DiagnosticJobs(runs=tmp_path)
    asyncio.run(jobs.recover_interrupted())
    assert diagnostics.read_object(abandoned / "run.json")["status"] == "interrupted"
    assert diagnostics.read_object(active / "run.json")["status"] == "running"
    assert diagnostics.read_object(completed / "run.json")["status"] == "completed"
    assert jobs.process is None


def test_worker_rejects_inputs_changed_since_selection(tmp_path, monkeypatch):
    diagnostics.write_record(tmp_path / "inputs.json", {"fingerprint": "changed"})
    prepare = AsyncMock()
    monkeypatch.setattr(runner, "_prepare_recovery_dryrun_harness", prepare)
    with pytest.raises(ValueError, match="changed after test selection"):
        asyncio.run(
            runner._run_actual_recovery(
                mode="outline", debug_root=tmp_path, runtime_context_path=CONTEXT
            )
        )
    prepare.assert_not_called()


def test_job_cancels_an_actual_isolated_process(tmp_path, monkeypatch):
    async def exercise():
        original = asyncio.create_subprocess_exec

        async def inert_process(*args, **kwargs):
            return await original(
                diagnostics.sys.executable, "-c", "import time; time.sleep(60)", **kwargs
            )

        monkeypatch.setattr(asyncio, "create_subprocess_exec", inert_process)
        jobs = diagnostics.DiagnosticJobs(runs=tmp_path / "runs")
        directory = await jobs.start("outline", CONTEXT, SCENARIO)
        await jobs.cancel()
        assert jobs.process.returncode is not None
        assert diagnostics.read_object(directory / "run.json")["status"] == "cancelled"

    asyncio.run(exercise())


def test_runner_prepares_selected_failure_context_without_model_work(tmp_path):
    context = diagnostics.read_object(CONTEXT)
    context["failure_scenario_id"] = "move_home_failure"
    context["failure_event"]["failed_task_id"] = "REQ_2_T5"
    context["task_statuses"].update(REQ_2_T4="completed", REQ_2_T5="failed")
    path = tmp_path / "runtime_context.json"
    diagnostics.write_record(path, context)
    fixture, product_agent, planner, prepared = asyncio.run(
        runner._prepare_recovery_dryrun_harness(
            runtime_context_path=path,
            debug_root=tmp_path / "artifacts",
            scripted_responses=[],
        )
    )
    assert fixture["failed_task_id"] == "REQ_2_T5"
    assert prepared["failure_context_raw"]["failed_function_name"] == "move_home"
    assert product_agent.turn_log == []
    assert product_agent._scripted_responses == []


def test_frozen_worker_rebinds_absolute_references_and_uses_only_saved_files(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    context = diagnostics.read_object(CONTEXT)
    context["bundle_root"] = str(diagnostics.ROOT / context["bundle_root"])
    context["product_geometry"] = str(diagnostics.ROOT / context["product_geometry"])
    for snapshot in context["resource_snapshots"]:
        snapshot["resource_config"] = str(diagnostics.ROOT / snapshot["resource_config"])
    diagnostics.write_record(source, context)
    inputs = diagnostics.inspect_inputs(source)
    run = tmp_path / "run"
    diagnostics.snapshot_inputs(run, inputs)
    frozen = run / "inputs"
    assert diagnostics.inspect_inputs(frozen / "runtime_context.json", root=frozen)["fingerprint"] == inputs["fingerprint"]
    diagnostics.write_record(source, {**context, "goal_state": "changed after selection"})
    monkeypatch.setattr(runner, "DATA_ROOT", frozen)
    load = runner._load_json
    accessed = []

    def only_frozen(path):
        assert path.is_relative_to(frozen), path
        accessed.append(path)
        return load(path)

    monkeypatch.setattr(runner, "_load_json", only_frozen)
    fixture, product_agent, _, prepared = asyncio.run(runner._prepare_recovery_dryrun_harness(
        runtime_context_path=frozen / "runtime_context.json", debug_root=run / "artifacts",
    ))
    assert accessed
    assert fixture["goal_state"] == context["goal_state"]
    assert product_agent.turn_log == []
    assert prepared["recovery_resources"]
    for resource in product_agent._mock_recovery_validation_resources.values():
        assert isinstance(resource, runner.SnapshotRecoveryRobot)
        assert resource.get_recovery_snapshot().get("controller_ready") is None
        with pytest.raises(RuntimeError, match="dispatch is disabled"):
            resource.release_part()


def test_replay_outline_cannot_be_reused_as_live_evidence(tmp_path):
    source = tmp_path / "outline.json"
    checkpoint = _checkpoint(source)
    checkpoint["input_fingerprint"] = diagnostics.inspect_inputs(CONTEXT, response_source="fixture_response_replay")["fingerprint"]
    diagnostics.write_record(source, checkpoint)
    with pytest.raises(ValueError, match="inputs differ"):
        diagnostics.load_outline_checkpoint(source, diagnostics.inspect_inputs(CONTEXT))


def test_injection_target_is_not_promoted_to_an_observation():
    context = diagnostics.read_object(CONTEXT)
    plan = runner._load_json(runner._case3_paths(context)["plan"])
    event = runner._build_live_style_failure_payload(plan, runtime_context=context)
    assert "dropped_location" not in event["failure_context"].get("observations", {})
    assert "gripper_force" not in event["failure_context"].get("observations", {})
    context["failure_event"]["observations"] = {"dropped_location": {"x": 0.2, "y": 0.1, "z": 0.3}, "evidence_source": "observed"}
    event = runner._build_live_style_failure_payload(plan, runtime_context=context)
    assert event["failure_context"]["observations"]["dropped_location"] == context["failure_event"]["observations"]["dropped_location"]


def test_live_diagnostics_require_saved_cca_history_and_running_evidence():
    from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor

    monitor = OnlineSafetyMonitor({"SAFE_1": 'digraph DFA { node [shape = doublecircle]; 2; node [shape = circle]; 1; init -> 1; 1 -> 2 [label="true"]; 2 -> 2 [label="true"]; }'}, [])
    agent = runner.FakeProductAgent(tools_catalog=[], product_geometry={})
    with pytest.raises(ValueError, match="safety_dfa_states"):
        agent._saved_monitor_states(monitor)
    agent._saved_safety_context = {"safety_dfa_states": {"SAFE_1": "2"}}
    with pytest.raises(ValueError, match="running_aps"):
        agent._saved_monitor_states(monitor)
    agent._saved_safety_context["running_aps"] = []
    assert agent._saved_monitor_states(monitor) == {"SAFE_1": "2"}
    agent._saved_safety_context["history_error"] = {"reason": "malformed step"}
    with pytest.raises(ValueError, match="history"):
        agent._saved_monitor_states(monitor)


@pytest.mark.parametrize("suffix,slipping,held", [
    ("", "KET4_Square_4mm", "gear_large"),
    ("_reverse", "gear_large", "KET4_Square_4mm"),
])
def test_current_slippage_diagnostics_preserve_both_interrupted_tasks(suffix, slipping, held, tmp_path):
    path = CONTEXT.with_name("runtime_context" + suffix + ".json")
    inputs = diagnostics.inspect_inputs(path, SCENARIO)
    assert inputs["scenario"] == "part_slippage"
    context = inputs["runtime_context"]
    assert context["fixture_kind"] == "synthetic_checkpoint"
    assert context["task_statuses"]["REQ_1_T3"] == "pending"
    assert context["task_statuses"]["REQ_2_T3"] == "pending"
    assert context["part_tracker"][slipping]["state"] == "misplaced"
    assert context["part_tracker"][held]["state"] == "in_gripper"
    fixture, product, _, request = asyncio.run(runner._prepare_recovery_dryrun_harness(
        runtime_context_path=path, debug_root=tmp_path, scripted_responses=[],
    ))
    failure = request["failure_context_raw"]
    assert failure["failed_task_id"] == context["failure_event"]["failed_task_id"]
    affected = failure["failure_context"]["affected_entities"]
    assert any(row["entity_type"] == "part" and row["entity_id"] == slipping for row in affected)
    snapshots = {row["resource_jid"]: row for row in context["resource_snapshots"]}
    custodian = context["part_tracker"][held]["location"]
    assert snapshots[custodian]["held_part"] == held
    assert all(row["resource_id"] in {"ur5e-3", "ur5e-4"} for row in snapshots.values())
    assert not product.turn_log

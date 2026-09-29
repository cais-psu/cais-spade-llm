"""Saved-context diagnostic inputs, outline reuse, and subprocess lifecycle."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import test_case3_recovery_dryrun as runner

from cais_spade_llm.recovery_framework import diagnostics

CONTEXT = diagnostics.ROOT / "test/fixtures/case3_recovery/runtime_context.json"
SCENARIO = diagnostics.ROOT / "cais_spade_llm/initialization/failure_scenarios/lg_slippage.json"
TRACE = [
    {
        "outline_id": "RECOVERY_SEQ1",
        "event_name": "recover_to_home",
        "resource_jid": "xarm6@localhost",
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
        assert command[1].endswith("test/test_case3_recovery_dryrun.py")
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

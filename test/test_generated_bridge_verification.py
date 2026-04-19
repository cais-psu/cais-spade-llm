from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cais_spade_llm.agents.intelligent_product.product_agent as product_agent_module
from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn import (
    normalize_multi_turn_final_output_to_bridge_proposal,
)
from test.test_case3_bridge_dryrun import _prepare_bridge_dryrun_harness


ROOT = Path(__file__).resolve().parents[1]
WORKED_FINAL_OUTPUT = (
    ROOT
    / "cais_spade_llm"
    / "monitor"
    / "debug"
    / "worked"
    / "1"
    / "multi_turn_turn24_final_output_response_20260416T160605.txt"
)


def _load_worked_final_output() -> dict:
    return json.loads(WORKED_FINAL_OUTPUT.read_text(encoding="utf-8"))


def _prepare_request() -> dict:
    _, _, _, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(reasoning_mode="multi_turn")
    )
    return prepared_bridge_request


def _minimal_runtime_request(prepared_bridge_request: dict) -> dict:
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["reasoning_mode"] = "multi_turn"
    return {
        "ra_jid": deepcopy(prepared_bridge_request.get("ra_jid")),
        "bridge_session": bridge_session,
        "bridge_debug": deepcopy(prepared_bridge_request.get("bridge_debug") or {}),
        "primitive_catalog": deepcopy(prepared_bridge_request.get("primitive_catalog") or []),
        "bridge_snapshot": deepcopy(prepared_bridge_request.get("bridge_snapshot") or {}),
        "grounding_context": deepcopy(prepared_bridge_request.get("grounding_context") or {}),
        "bridge_resources": deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
        "obligation_targets": deepcopy(prepared_bridge_request.get("obligation_targets") or []),
    }


def _build_runtime_test_agent(prepared_bridge_request: dict) -> ProductAgent:
    async def _unexpected_live_generation(
        _prepared_bridge_request: dict,
    ) -> dict:
        raise AssertionError("live bridge generation should not run during fixture replay")

    class _StubPlanner:
        def __init__(self) -> None:
            self._last_bridge_debug: dict = {}
            self.replan_result: dict | None = None

        async def replan_with_feedback_online(
            self,
            *_args,
            **_kwargs,
        ) -> dict:
            return deepcopy(self.replan_result or {})

        async def execute_prepared_bridge_request(
            self,
            prepared_bridge_request: dict,
        ) -> dict:
            return await _unexpected_live_generation(prepared_bridge_request)

        def get_last_bridge_debug(self) -> dict:
            return deepcopy(self._last_bridge_debug)

        def _set_last_bridge_debug(self, bridge_debug: dict) -> None:
            self._last_bridge_debug = deepcopy(bridge_debug)

        @staticmethod
        def _bridge_summary(proposal: dict) -> list[str]:
            return [
                str(task.get("macro_name") or "").strip()
                for task in (proposal.get("macro_tasks") or [])
                if isinstance(task, dict) and str(task.get("macro_name") or "").strip()
            ]

    agent = ProductAgent.__new__(ProductAgent)
    agent.agent_name = "generated-bridge-runtime-test"
    agent.jid = "assembly_board-v1@localhost"
    agent.logger = logging.getLogger("generated-bridge-runtime-test")
    agent._runtime_repair_max_attempts = 3
    agent._runtime_repair_fail_streak = 0
    agent._runtime_repair_inflight = False
    agent._bridge_generation_mode = "auto"
    agent._bridge_reasoning_mode = "multi_turn"
    agent._generated_bridge_gazebo_verification_enabled = False
    agent.runtime_repair_state = "idle"
    agent.plan_safety_alert = None
    agent.process_planner = _StubPlanner()
    agent._runtime_recovery_context = {
        "prepared_bridge_request": deepcopy(prepared_bridge_request),
        "failed_task_id": "REQ_2_T4",
        "violations": [],
        "trigger": "runtime_failure",
        "bridge_feedback_history": [],
    }
    agent.runtime_recovery = agent._empty_runtime_recovery()
    agent._set_runtime_recovery(
        status="bridge_ready",
        resolution_class="none",
        trigger="runtime_failure",
        failed_task_id="REQ_2_T4",
        message="Runtime bridge ready.",
        attempts_used=0,
        attempts_max=3,
        used_llm_bridge=False,
        bridge_proposal=None,
        bridge_debug=deepcopy(prepared_bridge_request.get("bridge_debug") or {}),
        bridge_approval_state="none",
        active_bridge_sequence=None,
        bridge_feedback_history=[],
        violations=[],
    )
    agent._persist_product_state = lambda: None
    agent._record_runtime_bridge_artifacts = lambda **kwargs: {}
    agent._clear_plan_safety_alert = lambda: None
    agent._set_plan_safety_alert = lambda **kwargs: None
    agent._auto_runtime_preprogrammed_scenario_id = lambda: ""
    agent.load_preprogrammed_runtime_bridge_scenario_sync = (
        lambda scenario_id: {"scenario_id": scenario_id}
    )
    agent.approve_runtime_bridge_proposal_sync = lambda: {"auto_approved": False}
    return agent


async def _direct_to_thread(func, /, *args, **kwargs):
    return func(*args, **kwargs)


def test_worked_final_output_normalizes_to_primitive_bridge_proposal() -> None:
    prepared_bridge_request = _prepare_request()
    final_output = _load_worked_final_output()

    adapter_result = normalize_multi_turn_final_output_to_bridge_proposal(
        final_output_payload=final_output,
        prepared_bridge_request=prepared_bridge_request,
    )

    assert adapter_result["accepted"] is True, adapter_result.get("reason")
    proposal = dict(adapter_result.get("normalized_proposal") or {})
    macro_tasks = list(proposal.get("macro_tasks") or [])
    assert len(macro_tasks) == 4
    assert [task.get("resource_jid") for task in macro_tasks] == [
        "xarm6@localhost",
        "ur5e@localhost",
        "ur5e@localhost",
        "ur5e@localhost",
    ]
    assert [task.get("macro_name") for task in macro_tasks] == [
        "xarm6_clear_fault_to_idle",
        "ur5e_place_MCP_to_prusa-mk4-2",
        "ur5e_move_to_LG_observed_pose",
        "ur5e_move_with_LG_to_assembly_board-v1_place_approach",
    ]
    assert [len(task.get("primitive_steps") or []) for task in macro_tasks] == [1, 5, 5, 4]
    assert proposal.get("primary_obligation") in (None, {})


def test_worked_final_output_compiles_to_execute_recovery_macro_nodes(tmp_path: Path) -> None:
    prepared_bridge_request = _prepare_request()
    final_output = _load_worked_final_output()
    adapter_result = normalize_multi_turn_final_output_to_bridge_proposal(
        final_output_payload=final_output,
        prepared_bridge_request=prepared_bridge_request,
    )
    proposal = dict(adapter_result.get("normalized_proposal") or {})

    fake_product = SimpleNamespace(
        jid="assembly_board-v1@localhost",
        logger=logging.getLogger("generated-bridge-test"),
        global_fsa_path=str(tmp_path / "global_fsa.json"),
    )
    planner = ProcessPlanner(fake_product, [])
    compiled_tasks = planner.apply_bridge_macro_proposal(proposal, anchor_task_id="")

    assert len(compiled_tasks) == 4
    assert all(task.get("function_name") == "execute_recovery_macro" for task in compiled_tasks)
    assert all(task.get("bridge_sequence_id") for task in compiled_tasks)
    assert [int(task.get("bridge_sequence_index") or 0) for task in compiled_tasks] == [1, 2, 3, 4]
    assert [
        str(dict(task.get("params") or {}).get("macro_name") or "")
        for task in compiled_tasks
    ] == [
        "xarm6_clear_fault_to_idle",
        "ur5e_place_MCP_to_prusa-mk4-2",
        "ur5e_move_to_LG_observed_pose",
        "ur5e_move_with_LG_to_assembly_board-v1_place_approach",
    ]


def test_malformed_final_output_is_rejected() -> None:
    prepared_bridge_request = _prepare_request()
    final_output = deepcopy(_load_worked_final_output())
    final_output.pop("accepted_primitive_program", None)

    adapter_result = normalize_multi_turn_final_output_to_bridge_proposal(
        final_output_payload=final_output,
        prepared_bridge_request=prepared_bridge_request,
    )

    assert adapter_result["accepted"] is False
    assert adapter_result.get("reason") == "final output did not include accepted_primitive_program"


def test_fixture_replay_helper_matches_direct_adapter(monkeypatch) -> None:
    prepared_bridge_request = _prepare_request()
    direct = normalize_multi_turn_final_output_to_bridge_proposal(
        final_output_payload=_load_worked_final_output(),
        prepared_bridge_request=prepared_bridge_request,
    )
    monkeypatch.setenv(
        "CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT",
        str(WORKED_FINAL_OUTPUT),
    )
    agent = ProductAgent.__new__(ProductAgent)

    fixture_replay = agent._load_runtime_bridge_fixture_replay(
        deepcopy(prepared_bridge_request)
    )

    assert fixture_replay["enabled"] is True
    assert fixture_replay["load_status"] == "loaded"
    assert fixture_replay["normalization_status"] == "accepted"
    assert fixture_replay.get("proposal") == direct.get("normalized_proposal")


def test_fixture_replay_runtime_loads_without_auto_verification(monkeypatch) -> None:
    prepared_bridge_request = _minimal_runtime_request(_prepare_request())
    monkeypatch.setenv(
        "CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT",
        str(WORKED_FINAL_OUTPUT),
    )
    monkeypatch.setenv("EXECUTION_MODE", "dry_run")
    monkeypatch.setenv("ROBOT_ENV", "gazebo")
    monkeypatch.setattr(product_agent_module.asyncio, "to_thread", _direct_to_thread)
    agent = _build_runtime_test_agent(prepared_bridge_request)

    recovery = asyncio.run(agent.generate_runtime_bridge_proposal())

    assert recovery["status"] == "llm_bridge"
    assert recovery["bridge_approval_state"] == "pending"
    assert isinstance(recovery.get("bridge_proposal"), dict)
    assert dict(recovery.get("fixture_replay") or {}).get("enabled") is True
    assert dict(recovery.get("fixture_replay") or {}).get("load_status") == "loaded"
    assert dict(recovery.get("fixture_replay") or {}).get("normalization_status") == "accepted"
    verification = dict(recovery.get("generated_code_verification") or {})
    assert verification.get("enabled") is False
    assert verification.get("source_artifact_path") == str(WORKED_FINAL_OUTPUT.resolve())


def test_fixture_replay_runtime_auto_verifies_when_gated(monkeypatch) -> None:
    prepared_bridge_request = _minimal_runtime_request(_prepare_request())
    monkeypatch.setenv(
        "CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT",
        str(WORKED_FINAL_OUTPUT),
    )
    monkeypatch.setenv("EXECUTION_MODE", "simulation")
    monkeypatch.setenv("ROBOT_ENV", "gazebo")
    monkeypatch.setattr(product_agent_module.asyncio, "to_thread", _direct_to_thread)
    agent = _build_runtime_test_agent(prepared_bridge_request)
    agent._generated_bridge_gazebo_verification_enabled = True
    approve_calls: list[bool] = []

    def _approve() -> dict:
        approve_calls.append(True)
        return {"auto_approved": True}

    agent.approve_runtime_bridge_proposal_sync = _approve

    result = asyncio.run(agent.generate_runtime_bridge_proposal())

    assert approve_calls == [True]
    assert result == {"auto_approved": True}


def test_fixture_replay_overrides_preprogrammed_case3_route(monkeypatch) -> None:
    prepared_bridge_request = _minimal_runtime_request(_prepare_request())
    monkeypatch.setenv(
        "CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT",
        str(WORKED_FINAL_OUTPUT),
    )
    monkeypatch.setenv("EXECUTION_MODE", "dry_run")
    monkeypatch.setenv("ROBOT_ENV", "gazebo")
    monkeypatch.setattr(product_agent_module.asyncio, "to_thread", _direct_to_thread)
    agent = _build_runtime_test_agent(prepared_bridge_request)
    agent._auto_runtime_preprogrammed_scenario_id = lambda: "case3_llm_bridge"

    def _unexpected_preprogrammed(_scenario_id: str) -> dict:
        raise AssertionError("preprogrammed scenario should be bypassed when fixture replay is enabled")

    agent.load_preprogrammed_runtime_bridge_scenario_sync = _unexpected_preprogrammed

    recovery = asyncio.run(agent.generate_runtime_bridge_proposal())

    assert recovery["status"] == "llm_bridge"
    assert recovery["bridge_approval_state"] == "pending"
    assert dict(recovery.get("fixture_replay") or {}).get("enabled") is True
    assert isinstance(recovery.get("bridge_proposal"), dict)


def test_fixture_replay_autostarts_from_prepare_checkpoint_without_verification(
    monkeypatch,
) -> None:
    prepared_bridge_request = _minimal_runtime_request(_prepare_request())
    monkeypatch.setenv(
        "CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT",
        str(WORKED_FINAL_OUTPUT),
    )
    monkeypatch.setenv("EXECUTION_MODE", "dry_run")
    monkeypatch.setenv("ROBOT_ENV", "gazebo")
    monkeypatch.setattr(product_agent_module.asyncio, "to_thread", _direct_to_thread)
    agent = _build_runtime_test_agent(prepared_bridge_request)
    agent._auto_runtime_preprogrammed_scenario_id = lambda: "case3_llm_bridge"
    agent.process_planner.replan_result = {
        "awaiting_bridge_generation": True,
        "prepared_bridge_request": deepcopy(prepared_bridge_request),
        "bridge_debug": deepcopy(prepared_bridge_request.get("bridge_debug") or {}),
        "message": "Prepared fixture replay request.",
    }
    generate_calls: list[bool] = []

    async def _fake_generate_runtime_bridge_proposal() -> dict:
        generate_calls.append(True)
        return {"status": "llm_bridge", "bridge_approval_state": "pending"}

    agent.generate_runtime_bridge_proposal = _fake_generate_runtime_bridge_proposal
    log_messages: list[str] = []

    class _RecordingLogger:
        def info(self, message: str, *args) -> None:
            log_messages.append(message % args if args else message)

        def exception(self, message: str, *args) -> None:
            log_messages.append(message % args if args else message)

    agent.logger = _RecordingLogger()

    recovery = asyncio.run(
        agent._run_des_runtime_recovery_attempt(
            violations=[],
            trigger="runtime_failure",
            failed_task_id="REQ_2_T4",
            system_coordination_state={},
        )
    )

    assert generate_calls == [True]
    assert recovery == {"status": "llm_bridge", "bridge_approval_state": "pending"}
    log_text = "\n".join(log_messages)
    assert "Runtime bridge fixture replay gate: fixture_replay=True" in log_text
    assert str(WORKED_FINAL_OUTPUT.resolve()) in log_text
    assert "Auto-starting runtime bridge fixture replay from prepared request" in log_text


def test_fixture_replay_disabled_logs_env_missing_at_prepare_checkpoint(
    monkeypatch,
) -> None:
    prepared_bridge_request = _minimal_runtime_request(_prepare_request())
    monkeypatch.delenv("CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT", raising=False)
    monkeypatch.setenv("EXECUTION_MODE", "simulation")
    monkeypatch.setenv("ROBOT_ENV", "gazebo")
    monkeypatch.setattr(product_agent_module.asyncio, "to_thread", _direct_to_thread)
    agent = _build_runtime_test_agent(prepared_bridge_request)
    agent._auto_runtime_preprogrammed_scenario_id = lambda: "case3_llm_bridge"
    agent.process_planner.replan_result = {
        "awaiting_bridge_generation": True,
        "prepared_bridge_request": deepcopy(prepared_bridge_request),
        "bridge_debug": deepcopy(prepared_bridge_request.get("bridge_debug") or {}),
        "message": "Prepared request without fixture replay.",
    }
    log_messages: list[str] = []

    class _RecordingLogger:
        def info(self, message: str, *args) -> None:
            log_messages.append(message % args if args else message)

        def exception(self, message: str, *args) -> None:
            log_messages.append(message % args if args else message)

    agent.logger = _RecordingLogger()

    recovery = asyncio.run(
        agent._run_des_runtime_recovery_attempt(
            violations=[],
            trigger="runtime_failure",
            failed_task_id="REQ_2_T4",
            system_coordination_state={},
        )
    )

    assert recovery["status"] == "bridge_ready"
    assert recovery["bridge_approval_state"] == "ready"
    log_text = "\n".join(log_messages)
    assert "Runtime bridge fixture replay gate: fixture_replay=False" in log_text
    assert "source_path=<unset>" in log_text
    assert (
        "CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT is not set; staying at bridge_ready"
        in log_text
    )


def test_fixture_replay_missing_path_returns_structured_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prepared_bridge_request = _prepare_request()
    missing_path = tmp_path / "missing_final_output.json"
    monkeypatch.setenv(
        "CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT",
        str(missing_path),
    )
    agent = ProductAgent.__new__(ProductAgent)

    fixture_replay = agent._load_runtime_bridge_fixture_replay(
        deepcopy(prepared_bridge_request)
    )

    assert fixture_replay["enabled"] is True
    assert fixture_replay["load_status"] == "missing"
    assert fixture_replay["normalization_status"] == "skipped"
    assert "does not exist" in str(fixture_replay.get("reason") or "")


def test_fixture_replay_invalid_json_returns_structured_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prepared_bridge_request = _prepare_request()
    invalid_path = tmp_path / "invalid_final_output.json"
    invalid_path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setenv(
        "CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT",
        str(invalid_path),
    )
    agent = ProductAgent.__new__(ProductAgent)

    fixture_replay = agent._load_runtime_bridge_fixture_replay(
        deepcopy(prepared_bridge_request)
    )

    assert fixture_replay["enabled"] is True
    assert fixture_replay["load_status"] == "invalid_json"
    assert fixture_replay["normalization_status"] == "skipped"
    assert "failed to parse fixture final_output artifact" in str(
        fixture_replay.get("reason") or ""
    )


def test_fixture_replay_adapter_rejection_returns_structured_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prepared_bridge_request = _prepare_request()
    rejected_path = tmp_path / "rejected_final_output.json"
    rejected_payload = deepcopy(_load_worked_final_output())
    rejected_payload.pop("accepted_primitive_program", None)
    rejected_path.write_text(json.dumps(rejected_payload), encoding="utf-8")
    monkeypatch.setenv(
        "CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT",
        str(rejected_path),
    )
    agent = ProductAgent.__new__(ProductAgent)

    fixture_replay = agent._load_runtime_bridge_fixture_replay(
        deepcopy(prepared_bridge_request)
    )

    assert fixture_replay["enabled"] is True
    assert fixture_replay["load_status"] == "loaded"
    assert fixture_replay["normalization_status"] == "rejected"
    assert fixture_replay.get("reason") == "final output did not include accepted_primitive_program"


def test_generated_bridge_verification_gate_requires_multi_turn_gazebo(monkeypatch) -> None:
    agent = ProductAgent.__new__(ProductAgent)
    agent._generated_bridge_gazebo_verification_enabled = True

    monkeypatch.setenv("EXECUTION_MODE", "simulation")
    monkeypatch.setenv("ROBOT_ENV", "gazebo")
    assert agent._should_enable_generated_bridge_verification(
        bridge_debug={"reasoning_mode": "multi_turn"}
    )
    assert not agent._should_enable_generated_bridge_verification(
        bridge_debug={"reasoning_mode": "legacy_mode"}
    )

    monkeypatch.setenv("EXECUTION_MODE", "dry_run")
    assert not agent._should_enable_generated_bridge_verification(
        bridge_debug={"reasoning_mode": "multi_turn"}
    )

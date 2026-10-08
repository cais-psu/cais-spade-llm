from __future__ import annotations

"""Live command custody and retained physical monitor regression coverage."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from cais_spade_llm.agents.central_controller.online_safety_monitor import LivePhysicalMonitor
from cais_spade_llm.recovery_framework import fingerprint
from cais_spade_llm.resources.resource_safety_preparation import (
    LiveCommandLedger,
    PreparedRobotEvidence,
)

MUTEX = '''digraph DFA {
    node [shape = doublecircle]; 0; init -> 0;
    0 -> 0 [label="!ap001 | !ap002"];
    0 -> 1 [label="ap001 & ap002"];
    1 -> 1 [label="true"];
}'''
PRECEDENCE = '''digraph DFA {
    node [shape = doublecircle]; 0; 1; init -> 0;
    0 -> 2 [label="ap002"];
    0 -> 1 [label="ap001 & !ap002"];
    0 -> 0 [label="!ap001 & !ap002"];
    1 -> 1 [label="true"];
    2 -> 2 [label="true"];
}'''


def _physical_case(dot, phase, values):
    return {"rules": [{"rule_id": "physical", "dfa_dot": dot, "formula": "fixture", "aps": []}],
            "observations": [{"time_exact": "0", "phase": phase, "rule_cells": [{"physical": values}]}],
            "valuations": [{"physical": values}]}


def test_preparation_has_no_dispatch_authority_and_batch_authorization_is_atomic():
    ledger = LiveCommandLedger()
    first = ledger.prepare(resource_id="ur5e-3", task_id="task", command={"q": [0]},
                           owner_identity={"preparation_id": "first"})
    second = ledger.prepare(resource_id="ur5e-3", task_id="task", command={"q": [1]},
                            owner_identity={"preparation_id": "second"})
    assert ledger.execution_epoch == 0
    with pytest.raises(ValueError, match="pending physical evidence"):
        ledger.start(first, command={"q": [0]}, controller_goal_id=None)
    with pytest.raises(ValueError, match="matching physical reservation"):
        ledger.require_authorized(first, command={"q": [0]})
    with pytest.raises(ValueError, match="stale"):
        ledger.authorize_many([first, second], reservation_token="claim", expected_revision=1)
    assert all("reservation_token" not in row for row in ledger.commands.values())
    ledger.authorize_many([first, second], reservation_token="claim", expected_revision=2)
    assert ledger.execution_epoch == 1
    with pytest.raises(ValueError, match="already started"):
        ledger.authorize_many([first, second], reservation_token="changed", expected_revision=ledger.revision)
    with pytest.raises(ValueError, match="matching physical reservation"):
        ledger.require_authorized(first, command={"q": [9]})
    ledger.start(first, command={"q": [0]}, controller_goal_id=None)
    ledger.bind_goal(first, [1] * 16)
    ledger.finish(first, success=False, observations={"joint_positions": [0.3]})
    assert ledger.commands[first]["status"] == "failed"
    assert ledger.commands[first]["reservation_token"] == "claim"
    assert ledger.commands[first]["controller_goal_id"] == [1] * 16
    with pytest.raises(ValueError, match="matching physical reservation"):
        ledger.require_authorized(first, command={"q": [0]})


def test_monitor_checks_every_continuous_valuation_and_preserves_predictions():
    monitor = LivePhysicalMonitor()
    unsafe = _physical_case(MUTEX, "between", {"ap001": [False, True], "ap002": True})
    assert monitor.check(unsafe)["status"] == "held"
    assert monitor.revision == 0 and monitor.states == {} and monitor.history == []
    safe = _physical_case(MUTEX, "between", {"ap001": [False, True], "ap002": False})
    proof = monitor.check(safe)
    assert proof["status"] == "allowed"
    monitor.commit(proof, success=True, execution_evidence={"owner": "ur5e-3", "actual": True})
    assert monitor.states == {"physical": {"0"}}
    with pytest.raises(ValueError, match="history"):
        monitor.commit(proof, success=True, execution_evidence={"owner": "ur5e-3"})


def test_precedence_monitor_retains_accepted_history_across_new_programs():
    monitor = LivePhysicalMonitor()
    entry = _physical_case(PRECEDENCE, "point", {"ap001": False, "ap002": True})
    assert monitor.check(entry)["status"] == "held"
    assembly = _physical_case(PRECEDENCE, "point", {"ap001": True, "ap002": False})
    monitor.commit(monitor.check(assembly), success=True, execution_evidence={"assembly": "observed"})
    assert monitor.states == {"physical": {"1"}}
    assert monitor.check(entry)["status"] == "allowed"
    assert monitor.revision == 1


def test_failure_keeps_all_possible_prefix_states_without_resetting_monitor():
    monitor = LivePhysicalMonitor()
    assembly = _physical_case(PRECEDENCE, "point", {"ap001": True, "ap002": False})
    monitor.commit(monitor.check(assembly), success=False, execution_evidence={"controller": "failed"})
    assert monitor.states == {"physical": {"0", "1"}}
    entry = _physical_case(PRECEDENCE, "point", {"ap001": False, "ap002": True})
    with pytest.raises(ValueError, match="stopping_coverage_unverified"):
        monitor.check(entry)


def _owner_evidence():
    ledger = LiveCommandLedger()
    owner = SimpleNamespace(agent_name="ur5e-3", jid="ur5e-3@localhost")
    provider = PreparedRobotEvidence(owner, ledger)
    planned = {"status": "prepared", "observation_status": "prepared", "primitive": "move_cartesian",
               "params": {"x": 1}, "start": {"launch_id": "launch"},
               "joint_trajectory": {"native": "untouched"}, "continuous_motion": {"configuration": "registered"}}
    planned["preparation_id"] = fingerprint(planned)
    provider.register_program({"status": "prepared", "resource_id": owner.agent_name, "steps": [planned]})
    row = {"primitive": planned["primitive"], "resolved_params": deepcopy(planned["params"]),
           "model_evidence": {key: deepcopy(planned[key])
                              for key in ("preparation_id", "joint_trajectory", "continuous_motion")}}
    request = {"task_id": "task", "resource_jid": owner.jid, "program_hash": "program",
               "primitive_steps": [{"primitive": "move_cartesian", "params": {"x": 1}}],
               "event_programs": [{"step_results": [row]}]}
    return owner, ledger, provider, request


def test_owner_rejects_changed_native_evidence_and_executes_exact_grant_once():
    owner, ledger, provider, request = _owner_evidence()
    changed = deepcopy(request)
    changed["event_programs"][0]["step_results"][0]["model_evidence"]["joint_trajectory"] = {"native": "changed"}
    with pytest.raises(ValueError, match="changed the owner motion"):
        provider.prepare(resource_agent=owner, request=changed, physical_snapshot={})
    prepared = provider.prepare(resource_agent=owner, request=request, physical_snapshot={})
    results, calls = [], []
    provider.execution_observer = lambda **row: results.append(row)
    owner._controller = SimpleNamespace(execute_prepared_recovery_step=lambda planned:
        calls.append(deepcopy(planned)) or {"success": True, "observations": {"joint_positions": [1]}})
    assert provider.authorize_preparation(
        preparation_id=prepared["preparation_id"], reservation_token="claim",
        command_ids=prepared["command_ids"], expected_revision=ledger.revision) is True
    grant = {"preparation_id": prepared["preparation_id"]}
    assert provider.execute_step(resource_agent=owner, task_id="task", step_index=0,
                                 primitive="move_cartesian", params={"x": 1}, grant=grant)["success"]
    assert calls[0]["joint_trajectory"] == {"native": "untouched"}
    assert results[0]["task_id"] == "task" and results[0]["result"]["success"]
    with pytest.raises(ValueError, match="consumed"):
        provider.execute_step(resource_agent=owner, task_id="task", step_index=0,
                              primitive="move_cartesian", params={"x": 1}, grant=grant)
    assert len(calls) == 1


@pytest.mark.parametrize("asserted", [None, {"kind": "controller_error_bound", "error": 0.001},
    {"future_execution_tracking": "established", "certificate": "LLM-generated"}])
def test_idle_observer_cannot_promote_missing_or_asserted_tracking_to_live_authority(asserted):
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import LiveSafetyPreparation

    provider = LiveSafetyPreparation(SimpleNamespace(), SimpleNamespace(), {})
    provider.observers["ur5e-3"] = SimpleNamespace()
    checkpoint = {"unresolved": [], "tracking_certificate": asserted}
    with pytest.raises(ValueError, match="live_execution_tracking_unverified"):
        provider.require_execution_coverage(checkpoint, "ur5e-3")


def test_prefix_viability_does_not_discharge_a_pending_eventual_requirement():
    eventually = '''digraph DFA {
        node [shape = doublecircle]; 1; init -> 0;
        0 -> 0 [label="!ap001"]; 0 -> 1 [label="ap001"]; 1 -> 1 [label="true"];
    }'''
    monitor = LivePhysicalMonitor()
    pending = monitor.check(_physical_case(eventually, "point", {"ap001": False}))
    assert pending["status"] == "inconclusive" and pending["prefix_safe"] is True
    assert pending["pending_rule_ids"] == ["physical"] and pending["end_accepting"] is False
    complete = monitor.check(_physical_case(eventually, "point", {"ap001": True}))
    assert complete["status"] == "allowed" and complete["end_accepting"] is True


def _trajectory_record():
    return {"header": {"frame_id": "world", "stamp": {"sec": 0, "nanosec": 0}},
            "joint_names": ["slide"], "duration_ns": 1_000_000_000,
            "points": [{"positions": [q], "velocities": [], "accelerations": [], "effort": [],
                        "time_from_start": {"sec": i, "nanosec": 0}}
                       for i, q in enumerate([0., 1.])]}


def _native_coverage_case():
    owner, ledger, provider, _ = _owner_evidence()
    owner._controller = SimpleNamespace(execution_mode="simulation")
    planned = deepcopy(next(iter(provider.steps.values())))
    planned.pop("preparation_id")
    planned["joint_trajectory"] = _trajectory_record()
    planned["start"] = {"joint_positions": [0.], "launch_id": "launch"}
    planned["preparation_id"] = fingerprint(planned)
    prepared = {"status": "prepared", "resource_id": owner.agent_name, "steps": [planned]}
    provider.register_program(prepared)
    owner.recovery_composition_evidence_provider = provider
    checkpoint = {"checkpoint_id": "checkpoint", "unresolved": [], "observations": {
        owner.agent_name: {"physical": {"idle": True, "custody_complete": True,
            "attachment": {"model_name": None}, "observation_state": {"held_part": None}}}}}
    native = {"operation": "prepare", "accepted": True, "reason": "",
        "instance_id": "instance", "expected_instance_id": "instance",
        "command_revision": 3, "expected_command_revision": 3,
        "contract_revision": 0, "expected_contract_revision": 0,
        "binding_fingerprint": "binding", "requested_binding_fingerprint": "binding",
        "joint_names": ["slide"], "observed_positions": [0.], "observed_velocities": [0.],
        "stationary": False, "reservation_token": "", "physical_execution_verified": False,
        "physical_execution_reason": "gazebo_physics_containment_and_stopping_unverified",
        "maximum_observed_update_period": 0.001, "maximum_observed_position_error": [0.0],
        "rejected_commands": 0,
        "contract_state": "unlocked", "observed_stationary": True, "stationary_samples": 3,
        "simulation_time": 1.0, "checkpoint_simulation_time": 1.0, "elapsed_wall_s": 0.1}
    return owner, provider, prepared, checkpoint, native


def test_native_ownership_diagnostics_never_supply_physics_execution_bounds(monkeypatch):
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import LiveSafetyPreparation
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller

    owner, provider, prepared, checkpoint, native = _native_coverage_case()
    calls = []
    monkeypatch.setattr(gazebo_pick_place_controller, "_prepare_controller_contract",
                        lambda *args: calls.append(deepcopy(args[1])) or deepcopy(native))
    preparation = LiveSafetyPreparation(SimpleNamespace(), SimpleNamespace(), {})
    preparation.observers[owner.agent_name] = SimpleNamespace(owner=owner)
    before = deepcopy(prepared), deepcopy(checkpoint), provider.ledger.snapshot()
    with pytest.raises(ValueError, match="no verified tracking"):
        preparation.require_execution_coverage(checkpoint, owner.agent_name, prepared=prepared)
    result = preparation.last_execution_coverage
    assert result["status"] == "NEEDS_CONTEXT" and result["native_contract"] == native
    assert not result["dispatch_authorized"] and not result["command_sent"]
    assert calls == prepared["steps"]
    assert before == (prepared, checkpoint, provider.ledger.snapshot())
    assert len(provider.execution_coverage_records) == 1


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("instance_id", "restarted", "identity changed"),
    ("command_revision", 4, "identity changed"),
    ("contract_revision", 1, "identity changed"),
    ("binding_fingerprint", "another geometry", "identity changed"),
    ("observed_positions", [0.1], "moved from"),
    ("observed_velocities", [float("nan")], "incomplete joint"),
    ("joint_names", ["other"], "another set of joints"),
    ("physical_execution_verified", True, "Unsupported physical containment assertion"),
    ("physical_execution_verified", None, "status is unavailable"),
    ("reservation_token", "unexpected", "unexpectedly acquired execution authority"),
    ("simulation_time", 4.0, "observation is stale"),
    ("simulation_time", 0.5, "observation is stale"),
    ("elapsed_wall_s", 3.0, "observation is stale"),
    ("simulation_time", float("nan"), "timing is unavailable"),
    ("contract_state", "active", "no observed stationary unlocked owner"),
    ("observed_stationary", False, "no observed stationary unlocked owner"),
    ("stationary_samples", 1, "no observed stationary unlocked owner"),
    ("observed_velocities", [0.1], "no observed stationary unlocked owner"),
])
def test_native_execution_coverage_rejects_changed_or_asserted_evidence(monkeypatch, field, value, reason):
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller

    _, provider, prepared, checkpoint, native = _native_coverage_case()
    native[field] = value
    monkeypatch.setattr(gazebo_pick_place_controller, "_prepare_controller_contract", lambda *args: native)
    result = provider.prepare_execution_coverage(prepared=prepared, checkpoint=checkpoint)
    assert result["status"] == "NEEDS_CONTEXT" and reason in result["reason"]
    assert not provider.ledger.commands and not result["dispatch_authorized"]


def test_execution_coverage_rejects_trajectory_changes_before_native_query(monkeypatch):
    from unittest.mock import Mock

    from cais_spade_llm.resources.robot import gazebo_pick_place_controller

    _, provider, prepared, checkpoint, _ = _native_coverage_case()
    query = Mock()
    monkeypatch.setattr(gazebo_pick_place_controller, "_prepare_controller_contract", query)
    prepared["steps"][0]["joint_trajectory"]["points"][-1]["positions"] = [2.]
    result = provider.prepare_execution_coverage(prepared=prepared, checkpoint=checkpoint)
    assert "differs from the retained owner preparation" in result["reason"]
    query.assert_not_called()


@pytest.mark.parametrize("success", [True, False])
def test_native_coverage_retains_terminal_command_history_without_treating_it_as_active(monkeypatch, success):
    from cais_spade_llm.resources.robot import gazebo_pick_place_controller

    owner, provider, prepared, checkpoint, native = _native_coverage_case()
    ledger = provider.ledger
    identifier = ledger.prepare(resource_id=owner.agent_name, task_id="prior", command={"q": [0.]},
                                owner_identity={"preparation_id": "prior"})
    ledger.authorize(identifier, reservation_token="retained", expected_revision=ledger.revision)
    ledger.start(identifier, command={"q": [0.]}, controller_goal_id=None)
    ledger.finish(identifier, success=success, observations={"joint_positions": [0.]})
    before = ledger.snapshot()
    monkeypatch.setattr(gazebo_pick_place_controller, "_prepare_controller_contract", lambda *args: native)
    result = provider.prepare_execution_coverage(prepared=prepared, checkpoint=checkpoint)
    assert result["native_contract"] == native and "no verified tracking" in result["reason"]
    assert ledger.snapshot() == before


@pytest.mark.parametrize("change", ["moving", "held_part", "attachment", "missing_custody"])
def test_execution_coverage_requires_every_stationary_resource_and_empty_custody(monkeypatch, change):
    from unittest.mock import Mock

    from cais_spade_llm.resources.robot import gazebo_pick_place_controller

    _, provider, prepared, checkpoint, _ = _native_coverage_case()
    other = deepcopy(next(iter(checkpoint["observations"].values())))
    checkpoint["observations"]["other"] = other
    physical = other["physical"]
    if change == "moving":
        physical["idle"] = False
    elif change == "held_part":
        physical["observation_state"]["held_part"] = "gear_small"
    elif change == "attachment":
        physical["attachment"]["model_name"] = "gear_small"
    else:
        physical.pop("custody_complete")
    query = Mock()
    monkeypatch.setattr(gazebo_pick_place_controller, "_prepare_controller_contract", query)
    result = provider.prepare_execution_coverage(prepared=prepared, checkpoint=checkpoint)
    assert "observed stationary empty custody: other" in result["reason"]
    query.assert_not_called()


def test_ur_dispatch_roundtrips_native_trajectory_without_replanning_and_checks_start():
    pytest.importorskip("trajectory_msgs.msg")
    from cais_spade_llm.resources.resource_safety_preparation import trajectory_record
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
        execute_prepared_robot_step,
    )

    start = {"joint_names": ["slide"], "joint_positions": [0.], "frame": "world",
             "launch_id": "launch", "attachment": {"model_name": None}}
    observed = [deepcopy(start), {**deepcopy(start), "joint_positions": [1.]}]
    sends = []
    controller = SimpleNamespace(execution_mode="simulation", controller_config={"id": "owner"},
        arm_joint_names=["slide"], arm_trajectory_topic="/arm/joint_trajectory",
        capture_recovery_safety_state=lambda: observed.pop(0),
        _send_simulation_joint_trajectory=lambda topic, trajectory:
            sends.append((topic, trajectory_record(trajectory))) or True)
    planned = {"status": "prepared", "observation_status": "prepared",
               "configuration_fingerprint": fingerprint(controller.controller_config),
               "start": deepcopy(start), "joint_trajectory": _trajectory_record()}
    planned["preparation_id"] = fingerprint(planned)
    result = execute_prepared_robot_step(controller, planned)
    assert result["success"] and result["observations"]["joint_positions"] == [1.]
    assert sends == [(controller.arm_trajectory_topic, planned["joint_trajectory"])]
    observed.append({**deepcopy(start), "joint_positions": [.1]})
    with pytest.raises(ValueError, match="moved after"):
        execute_prepared_robot_step(controller, planned)
    assert len(sends) == 1


def test_kmr_prepares_without_dispatch_then_executes_the_retained_plan_once():
    from test_continuous_motion import configuration

    from cais_spade_llm.recovery_framework.kmr_safety_execution import worker_safety_request

    position, plans, sends = [0.], [], []
    def state():
        return SimpleNamespace(joint_state=SimpleNamespace(name=["slide"], position=list(position)))
    def project(state, names, values):
        return SimpleNamespace(joint_state=SimpleNamespace(name=list(names), position=list(values)))
    def plan(state, *, target, cartesian, waypoints):
        raw = _trajectory_record()
        native = SimpleNamespace(joint_names=raw["joint_names"],
            header=SimpleNamespace(frame_id="world", stamp=SimpleNamespace(sec=0, nanosec=0)),
            points=[SimpleNamespace(**{**row, "time_from_start": SimpleNamespace(**row["time_from_start"])})
                    for row in raw["points"]])
        result = (SimpleNamespace(joint_trajectory=native), {"target": target})
        plans.append(result)
        return result
    def execute(value):
        sends.append(value)
        position[:] = [1.]
    base = [0., 0., 0., 0., 0., 0., 1.]
    start = {"launch_id": "launch", "base_pose": base, "joint_names": ["slide"],
             "joint_positions": [0.], "current_pose": base}
    command = {"primitive": "move_cartesian", "params": {"target": [1., *base[1:]]}}
    request = {"mode": "safety_prepare", "start": start, "continuous_motion": configuration(),
        "program": {"resource_id": "KMR", "primitive_steps": [command]}, "resource_jid": "KMR@localhost",
        "binding": {"checkpoint_id": "checkpoint"}}
    session = {}
    callbacks = {"session": session, "probe": {"launch_id": "launch"}, "config": {"avoid_collisions": True},
        "kmr": {"arm_joint_names": ["slide"]}, "observed_state": state, "updated_state": project,
        "plan_motion": plan, "execute_plan": execute, "base_pose": base,
        "tcp_pose": lambda: [position[0], *base[1:]],
        "execution_evidence": lambda: [{"goal_id": "01" * 16, "success": True}]}
    prepared = worker_safety_request(request=request, **callbacks)["preparation"]
    assert not sends and len(plans) == 1
    original = prepared["steps"][0]
    dispatch = {"mode": "safety_execute", **command, "preparation_id": original["preparation_id"],
                "joint_trajectory": original["joint_trajectory"]}
    changed = deepcopy(dispatch)
    changed["params"]["target"][0] = 2.
    with pytest.raises(ValueError, match="differs"):
        worker_safety_request(request=changed, **callbacks)
    result = worker_safety_request(request=dispatch, **callbacks)["result"]
    assert sends[0] is plans[0] and len(plans) == 1
    assert result["observations"]["joint_positions"] == [1.]
    with pytest.raises(ValueError, match="already consumed"):
        worker_safety_request(request=dispatch, **callbacks)


@pytest.mark.parametrize("field", ["current_pose", "base_pose", "root_pose", "component_bounds",
                                   "footprint", "geometry_source", "attachment"])
def test_ur_refuses_changed_physical_geometry_before_native_dispatch(field):
    from unittest.mock import Mock

    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
        execute_prepared_robot_step,
    )

    start = {"joint_names": ["slide"], "joint_positions": [0.], "frame": "world", "launch_id": "launch",
             "attachment": {"model_name": None}, "current_pose": [0, 0, 0, 0, 0, 0, 1],
             "base_pose": [0, 0, 0, 0, 0, 0, 1], "root_pose": [0, 0, 0, 0, 0, 0, 1],
             "component_bounds": [{"bounds": [[-1, 1]] * 3}], "footprint": [[-1, 1]] * 3,
             "geometry_source": {"model": "registered", "configuration": "original"}}
    observed = deepcopy(start)
    observed[field] = "changed"
    controller = SimpleNamespace(execution_mode="simulation", controller_config={"id": "owner"},
        capture_recovery_safety_state=lambda: observed, _send_simulation_joint_trajectory=Mock())
    planned = {"status": "prepared", "observation_status": "prepared",
               "configuration_fingerprint": fingerprint(controller.controller_config),
               "start": start, "joint_trajectory": _trajectory_record()}
    planned["preparation_id"] = fingerprint(planned)
    with pytest.raises(ValueError, match="changed"):
        execute_prepared_robot_step(controller, planned)
    controller._send_simulation_joint_trajectory.assert_not_called()


@pytest.mark.parametrize("field", ["attachment", "component_bounds", "current_pose"])
def test_kmr_fresh_observer_rejects_changed_custody_or_geometry_without_worker_dispatch(monkeypatch, field):
    import asyncio
    from unittest.mock import AsyncMock

    from cais_spade_llm.recovery_framework import kmr_live_safety

    owner, ledger, stored, request = _owner_evidence()
    provider = kmr_live_safety.KMRPreparedEvidence(owner, ledger)
    provider.steps = stored.steps
    evidence = provider.prepare(resource_agent=owner, request=request, physical_snapshot={})
    preparation = provider.preparations[evidence["preparation_id"]]
    expected = {"joint_names": ["slide"], "joint_positions": [0.], "frame": "world", "launch_id": "launch",
                "attachment": {"model_name": None}, "current_pose": [0, 0, 0, 0, 0, 0, 1],
                "component_bounds": [{"bounds": [[-1, 1]] * 3}]}
    preparation["steps"][0]["start"] = deepcopy(expected)
    observed = deepcopy(expected)
    observed[field] = "changed"
    monkeypatch.setattr(kmr_live_safety, "capture_state", lambda owner, **kwargs: observed)
    owner.worker = SimpleNamespace(run=AsyncMock())
    reports = []
    provider.execution_observer = lambda **row: reports.append(row)
    provider.authorize_preparation(preparation_id=evidence["preparation_id"], reservation_token="claim",
        command_ids=evidence["command_ids"], expected_revision=ledger.revision)
    with pytest.raises(ValueError, match="changed"):
        asyncio.run(provider.execute_async_step(resource_agent=owner, task_id="task", step_index=0,
            primitive="move_cartesian", params={"x": 1}, grant={"preparation_id": evidence["preparation_id"]}))
    owner.worker.run.assert_not_called()
    assert reports[0]["result"]["success"] is False
    assert ledger.commands[evidence["command_ids"][0]]["reservation_token"] == "claim"


@pytest.mark.parametrize("executions", [{}, {0: {"success": False, "error": "no physical observation"}}])
def test_failure_without_owner_observation_retains_claim_and_does_not_commit_history(executions):
    from threading import RLock
    from unittest.mock import Mock

    from cais_spade_llm.agents.central_controller.recovery_composition_admission import (
        LiveSafetyRuntime,
    )

    live = object.__new__(LiveSafetyRuntime)
    live.lock = RLock()
    live.preparation = object()
    live.epoch = 0
    live.runtime = SimpleNamespace(context=SimpleNamespace(run_id="run"))
    live.monitor = SimpleNamespace(commit=Mock(), revision=7, invalid_reason="")
    live.regions = SimpleNamespace(finish=Mock())
    reference = {"recovery_id": "recovery", "task_id": "task"}
    task = {"task_id": "task", "resource_jid": "r1@localhost", "function_name": "execute_recovery_macro",
            "primitive_steps": [{}]}
    work = {"grant": {"recovery_composition_ref": reference, "region_reservation_token": "claim"},
            "executions": deepcopy(executions), "complete": False, "proof": {"predicted": True}}
    session = {"tasks": {"task": task}, "grants": {"task": work}, "product_jid": "product@localhost",
               "complete": False, "invalid_reason": ""}
    live.sessions = {"recovery": session}
    result = live.observe({"task_id": "task", "resource_jid": "r1@localhost",
        "function_name": task["function_name"], "run_id": "run", "status": "failed",
        "recovery_composition_ref": reference}, sender="r1@localhost")
    assert result["status"] == "inconclusive" and result["physical_history_revision"] == 7
    live.monitor.commit.assert_not_called()
    live.regions.finish.assert_not_called()
    assert not work["complete"] and session["invalid_reason"] and live.monitor.invalid_reason

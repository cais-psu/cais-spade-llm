"""Reuse real nominal acknowledgements while retaining recovery monitor history."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cais_spade_llm.agents.central_controller._recovery_monitor_history import TaskMonitorHistories
from cais_spade_llm.agents.central_controller.local_composition import Action
from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor
from cais_spade_llm.agents.central_controller.recovery_admission_runtime import (
    _context,
    recovery_admission,
)
from cais_spade_llm.agents.central_controller.recovery_composition_admission import (
    _monitor_identity,
)
from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import _digest
from cais_spade_llm.product.environment import EnvironmentProductContext
from cais_spade_llm.recovery_framework import PRODUCT_PATH, ROOT, SCENE_PATH, read_json
from cais_spade_llm.recovery_framework.environment_admission import EnvironmentAdmission
from cais_spade_llm.resources.environment_models import project_transition

_PRODUCT = "assembly_board-v1@localhost"
_RECOVERY = "KMR_storage_interruption_nominal_ack"
_DOT = '''digraph DFA {
node [shape = doublecircle]; 1; 2;
node [shape = circle]; 0;
init [shape = plaintext, label = ""]; init -> 0;
0 -> 0 [label="!ap001"];
0 -> 1 [label="ap001"];
1 -> 2 [label="true"];
2 -> 2 [label="true"];
}'''


def _environment_ack_case(*, self_loop=False):
    meta = next(iter(read_json(PRODUCT_PATH).values()))
    context = EnvironmentProductContext(
        scene=read_json(SCENE_PATH), product_order=read_json(ROOT / meta["product_order_file"]),
        geometry=read_json(ROOT / meta["product_geometry_file"])["gazebo"],
    )
    event = next(row for row in context.models["ur5e-3"]["events"] if row["event_name"] == "move_home")
    task = context.prepare({"resource_id": "ur5e-3", "event_id": event["event_id"],
                            "event_name": "move_home", "parameters": {
                                "resource_id": "ur5e-3", "home_available": True}}, simulated=True)
    jids = {resource: resource + "@localhost" for resource in context.resources}
    dot = _DOT.replace("doublecircle]; 1; 2;", "doublecircle]; 0; 1; 2;").replace('0 -> 1 [label="ap001"]', '0 -> 0 [label="ap001"]') if self_loop else _DOT
    formula = "G(ap001 | !ap001)" if self_loop else "F(ap001)"
    monitor = OnlineSafetyMonitor({"nominal_completion": dot}, [{"id": "nominal_completion", "ltlf": formula, "aps": [
        {"label": "ap001", "full": "ap/any/any/ur5e-3/move_home/any"}]}])
    runtime = SimpleNamespace(context=context, jids=jids, stopped=False, operation_goals={})
    admission = EnvironmentAdmission(runtime, monitor)
    labels = frozenset(monitor._map_task_to_aps(jids["ur5e-3"], "move_home", task["parameters"]))
    assert labels == {"ap001"}
    # The task was admitted before recovery registration; acknowledge still uses
    # the real pending task, reservation, projection, and observer implementation.
    admission.grants[task["task_id"]] = Action(task["task_id"], deepcopy(task), frozenset(), labels)
    monitor.running_aps = set(labels)
    after, products = project_transition(context.models, context.snapshot(), context.part_tracker,
                                         task, context.product_name, context.requirements)
    updates = {resource: {key: value for key, value in values.items()
                          if value != context.snapshot()[resource][key]}
               for resource, values in after.items() if values != context.snapshot()[resource]}
    native_task = {"task_id": task["task_id"], "resource_id": "ur5e-3", "event_name": "move_home",
                   "function_name": "move_home", "parameters": deepcopy(task["parameters"])}
    native = {"tasks": {task["task_id"]: {"task": native_task, "participants": ["ur5e-3"],
                                         "resource_updates": updates, "product_updates": {}}}}
    physical = {"resources": {"ur5e-3": {"current_pose": [0, 0, 1, 0, 0, 0, 1]}}, "parts": {}}
    inputs = {"running_work": [{"task_id": task["task_id"], "resource_id": "ur5e-3", "event_name": "move_home"}],
              "grounding_inputs": {"snapshot": deepcopy(physical)}}
    owner = {"revision": 0, "time_exact": "0", "current_snapshot": physical,
             "composition_inputs": inputs, "observations": [], "task_monitor_context": native,
             "execution_mode": "mock", "physical_rule_activation": "prospective"}
    agent = SimpleNamespace(
        resource_agents=[SimpleNamespace(jid=jid) for jid in jids.values()],
        safety_monitor=monitor, recovery_safety_scopes={}, allow_mock_recovery_execution=True,
        recovery_composition_context_provider=lambda *_: owner,
        _environment_runtime_for_sender=lambda _: runtime,
        _environment_admission=lambda _: admission, plan_fsa_monitor=None,
    )
    coordinator = recovery_admission(agent, _PRODUCT)
    captured = _context(agent, _PRODUCT, _RECOVERY)
    histories = TaskMonitorHistories(captured["task_monitor_context"], {}, {task["task_id"]: inputs["running_work"][0]})
    target_state, checks = histories.finish(histories.initial, task["task_id"])
    assert target_state["monitors"][0]["state"]["resources"] == after
    assert target_state["monitors"][0]["state"]["products"] == products
    edge = {"edge_id": "observed_nominal_completion", "source": "before", "target": "after",
            "kind": "task_completion", "controllable": False, "time_exact": "1",
            "task_id": task["task_id"], "event_id": task["task_id"],
            "task_monitor_state": target_state, "task_rule_checks": checks}
    nodes = {"before": {"id": "before", "state": deepcopy(physical), "task_monitor_state": histories.initial,
                        "winning": True, "marked": False},
             "after": {"id": "after", "state": deepcopy(physical), "task_monitor_state": target_state,
                       "winning": True, "marked": True}}
    session = {"request": {"recovery_id": _RECOVERY}, "product_jid": _PRODUCT, "inputs": inputs,
               "tasks": {}, "refs": {}, "grants": {}, "nodes": nodes, "edges": {"before": [edge]},
               "node": "before", "path": [], "completed_tasks": set(), "ledger": [], "record_ids": {},
               "invalid_reason": "", "complete": False, "execution_mode": "mock",
               "native_identity": _monitor_identity(captured), "revision": captured["revision"], "time_exact": "0",
               "physical_history": _digest([None, "prospective"]), "nominal_acknowledgements": [],
               "nominal_run_id": context.run_id, "owner_incarnations": captured["owner_incarnations"],
               "analysis": {"problem_id": "real_nominal_ack_test", "scope": {}}}
    coordinator.sessions[_RECOVERY] = session
    monitor.transition_evidence = Mock(wraps=monitor.transition_evidence)
    return SimpleNamespace(context=context, task=task, monitor=monitor, admission=admission,
                           coordinator=coordinator, session=session, owner=owner, agent=agent,
                           expected_states=target_state["monitors"][0]["state"]["states"])


def _acknowledge(case, *, physical=True):
    assert case.context.acknowledge({**case.task, "status": "completed"})
    assert case.admission.ack_cursor == 1
    assert case.monitor.current_states == case.expected_states
    case.owner["time_exact"] = "1"
    if physical:
        case.owner["observations"].append({"kind": "task_completion", "time_exact": "1",
                                           "task_id": case.task["task_id"], "status": "completed"})


def test_real_nominal_acknowledgement_is_reused_without_second_dfa_tick():
    case = _environment_ack_case()
    _acknowledge(case)
    result = case.coordinator.observe({"status": "owner_observation"}, sender=_PRODUCT)
    assert not case.session["invalid_reason"], result
    assert case.session["complete"]
    assert case.monitor.current_states == {"nominal_completion": "1"}
    assert case.monitor.transition_evidence.call_count == 1
    assert len(case.admission.acknowledgement_history) == 1
    assert case.session["nominal_acknowledgements"] == case.admission.acknowledgement_history
    assert case.context.snapshot()["ur5e-3"]["resource_location"] == "home"
    assert not case.context.acknowledge({**case.task, "status": "completed"})
    case.admission.synchronize()
    case.coordinator.observe({"status": "owner_observation"}, sender=_PRODUCT)
    assert case.monitor.transition_evidence.call_count == 1


def test_missing_physical_ack_holds_without_discarding_committed_nominal_history():
    case = _environment_ack_case()
    _acknowledge(case, physical=False)
    history = deepcopy(case.admission.acknowledgement_history)
    result = case.coordinator.observe({"status": "owner_observation"}, sender=_PRODUCT)
    assert result["results"][0]["status"] == "held"
    assert not case.session["invalid_reason"]
    assert case.coordinator.holds()
    assert case.session["node"] == "before"
    assert case.monitor.current_states == {"nominal_completion": "1"}
    assert case.monitor.transition_evidence.call_count == 1
    assert case.admission.acknowledgement_history == history
    case.owner["observations"].append({"kind": "task_completion", "time_exact": "1",
                                       "task_id": case.task["task_id"], "status": "completed"})
    case.coordinator.observe({"status": "owner_observation"}, sender=_PRODUCT)
    assert case.session["complete"]
    assert case.monitor.transition_evidence.call_count == 1


def test_self_loop_completion_still_requires_exact_owner_acknowledgement_audit():
    case = _environment_ack_case(self_loop=True)
    _acknowledge(case)
    assert case.monitor.current_states == {"nominal_completion": "0"}
    case.admission.acknowledgement_history.clear()
    case.coordinator.observe({"status": "owner_observation"}, sender=_PRODUCT)
    assert case.session["invalid_reason"] == "nominal_acknowledgement_history_unavailable"
    assert case.session["path"] == []
    assert case.monitor.transition_evidence.call_count == 1


@pytest.mark.parametrize("change", ["task_id", "parameters", "before", "after", "cursor", "duplicate", "checks"])
def test_changed_nominal_audit_cannot_be_reused(change):
    case = _environment_ack_case()
    _acknowledge(case)
    record = case.admission.acknowledgement_history[0]
    if change == "task_id":
        record["task_id"] = "other"
    elif change == "parameters":
        record["task"]["parameters"]["home_available"] = False
    elif change in {"before", "after"}:
        record[change + "_states"] = {"nominal_completion": "2"}
    elif change == "cursor":
        record["cursor"] = 2
    elif change == "duplicate":
        case.admission.acknowledgement_history.append(deepcopy(record))
    else:
        record["rule_checks"] = []
    case.coordinator.observe({"status": "owner_observation"}, sender=_PRODUCT)
    assert case.session["invalid_reason"].startswith("nominal_acknowledgement")
    assert case.monitor.current_states == {"nominal_completion": "1"}
    assert case.monitor.transition_evidence.call_count == 1
    assert case.session["path"] == []

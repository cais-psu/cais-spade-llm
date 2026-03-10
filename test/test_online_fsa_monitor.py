from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "cais_spade_llm"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from cais_spade_llm.agents.central_controller.online_fsa_monitor import OnlineFsaMonitor


def _sample_fsa() -> dict:
    return {
        "A": {
            "X": [
                "(ur5e@localhost=(k=0,idle),xarm6@localhost=(k=0,idle))",
                "(ur5e@localhost=(k=1,idle),xarm6@localhost=(k=0,run=REQ_2_T3:place_approach))",
                "(ur5e@localhost=(k=1,idle),xarm6@localhost=(k=1,idle))",
            ],
            "E": [
                "REQ_1_T5.done",
                "REQ_2_T3.start",
                "REQ_2_T3.done",
            ],
            "Tr": [
                {
                    "from": "(ur5e@localhost=(k=0,idle),xarm6@localhost=(k=0,idle))",
                    "event": "REQ_1_T5.done",
                    "to": "(ur5e@localhost=(k=1,idle),xarm6@localhost=(k=0,run=REQ_2_T3:place_approach))",
                    "task_id": "REQ_1_T5",
                    "function_name": "move_home",
                    "resource_jid": "ur5e@localhost",
                },
                {
                    "from": "(ur5e@localhost=(k=1,idle),xarm6@localhost=(k=0,run=REQ_2_T3:place_approach))",
                    "event": "REQ_2_T3.done",
                    "to": "(ur5e@localhost=(k=1,idle),xarm6@localhost=(k=1,idle))",
                    "task_id": "REQ_2_T3",
                    "function_name": "place_approach",
                    "resource_jid": "xarm6@localhost",
                },
            ],
            "x0": "(ur5e@localhost=(k=0,idle),xarm6@localhost=(k=0,idle))",
            "Xm": ["(ur5e@localhost=(k=1,idle),xarm6@localhost=(k=1,idle))"],
        },
        "meta": {},
    }


def _repairable_fsa() -> dict:
    s0 = "(ur5e@localhost=(k=2,idle),xarm6@localhost=(k=2,idle))"
    s1 = "(ur5e@localhost=(k=2,run=REQ_1_T3:place_approach),xarm6@localhost=(k=2,idle))"
    s2 = "(ur5e@localhost=(k=3,idle),xarm6@localhost=(k=2,idle))"
    s3 = "(ur5e@localhost=(k=3,run=REQ_1_T4:place_insert),xarm6@localhost=(k=2,idle))"
    s4 = "(ur5e@localhost=(k=4,idle),xarm6@localhost=(k=2,idle))"
    s5 = "(ur5e@localhost=(k=4,run=RECOVERY_1:move_home),xarm6@localhost=(k=2,idle))"
    s6 = "(ur5e@localhost=(k=5,idle),xarm6@localhost=(k=2,idle))"
    return {
        "A": {
            "X": [s0, s1, s2, s3, s4, s5, s6],
            "E": [
                "REQ_1_T3.start",
                "REQ_1_T3.done",
                "REQ_1_T4.start",
                "REQ_1_T4.done",
                "RECOVERY_1.start",
                "RECOVERY_1.done",
            ],
            "Tr": [
                {
                    "from": s0,
                    "event": "REQ_1_T3.start",
                    "to": s1,
                    "task_id": "REQ_1_T3",
                    "function_name": "place_approach",
                    "resource_jid": "ur5e@localhost",
                },
                {
                    "from": s1,
                    "event": "REQ_1_T3.done",
                    "to": s2,
                    "task_id": "REQ_1_T3",
                    "function_name": "place_approach",
                    "resource_jid": "ur5e@localhost",
                },
                {
                    "from": s2,
                    "event": "REQ_1_T4.start",
                    "to": s3,
                    "task_id": "REQ_1_T4",
                    "function_name": "place_insert",
                    "resource_jid": "ur5e@localhost",
                },
                {
                    "from": s3,
                    "event": "REQ_1_T4.done",
                    "to": s4,
                    "task_id": "REQ_1_T4",
                    "function_name": "place_insert",
                    "resource_jid": "ur5e@localhost",
                },
                {
                    "from": s4,
                    "event": "RECOVERY_1.start",
                    "to": s5,
                    "task_id": "RECOVERY_1",
                    "function_name": "move_home",
                    "resource_jid": "ur5e@localhost",
                },
                {
                    "from": s5,
                    "event": "RECOVERY_1.done",
                    "to": s6,
                    "task_id": "RECOVERY_1",
                    "function_name": "move_home",
                    "resource_jid": "ur5e@localhost",
                },
            ],
            "x0": s0,
            "Xm": [s6],
        },
        "meta": {},
    }


def test_online_fsa_monitor_detects_running_tasks_from_state():
    fsa = _sample_fsa()
    monitor = OnlineFsaMonitor(fsa)
    monitor.current_state = "(ur5e@localhost=(k=1,idle),xarm6@localhost=(k=0,run=REQ_2_T3:place_approach))"

    assert monitor.running_task_ids_from_state() == ["REQ_2_T3"]
    assert monitor._next_task_ids_from_state(monitor.current_state) == []


def test_online_fsa_monitor_matches_identical_fsa():
    fsa = _sample_fsa()
    monitor = OnlineFsaMonitor(fsa)

    assert monitor.matches_fsa(deepcopy(fsa)) is True

    changed = deepcopy(fsa)
    changed["A"]["Xm"] = [changed["A"]["x0"]]
    assert monitor.matches_fsa(changed) is False


def test_online_fsa_monitor_restores_live_running_state_into_repaired_fsa():
    fsa = _repairable_fsa()
    prior_monitor = OnlineFsaMonitor(fsa)
    prior_monitor.current_state = "(ur5e@localhost=(k=2,run=REQ_1_T3:place_approach),xarm6@localhost=(k=2,idle))"
    prior_monitor.completed_task_ids = {"REQ_1_T1", "REQ_1_T2", "REQ_2_T1", "REQ_2_T2"}

    repaired_monitor = OnlineFsaMonitor(deepcopy(fsa))
    assert repaired_monitor.restore_from_prior_monitor(prior_monitor) is True

    assert repaired_monitor.running_task_ids_from_state() == ["REQ_1_T3"]

    repaired_monitor.process_event(
        event_type="done",
        task_id="REQ_1_T3",
        function_name="place_approach",
        resource_jid="ur5e@localhost",
        status="completed",
    )

    assert repaired_monitor.current_state == "(ur5e@localhost=(k=3,idle),xarm6@localhost=(k=2,idle))"
    assert repaired_monitor._next_task_ids_from_state(repaired_monitor.current_state) == ["REQ_1_T4"]


def test_online_fsa_monitor_ignores_unmatched_done_event():
    monitor = OnlineFsaMonitor(_repairable_fsa())

    monitor.process_event(
        event_type="done",
        task_id="REQ_1_T3",
        function_name="place_approach",
        resource_jid="ur5e@localhost",
        status="completed",
    )

    assert monitor.current_state == "(ur5e@localhost=(k=2,idle),xarm6@localhost=(k=2,idle))"
    assert "REQ_1_T3" not in monitor.completed_task_ids

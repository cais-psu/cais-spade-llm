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

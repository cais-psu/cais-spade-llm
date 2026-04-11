from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cais_spade_llm.ui.components.dag_graph import (
    build_fsa_dag_overlay,
    nodes_to_mermaid,
)


def _task(
    task_id: str,
    resource_jid: str,
    *,
    predecessors: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": task_id,
        "type": "task",
        "resource_jid": resource_jid,
        "function_name": "pick_approach",
        "status": "pending",
        "predecessors": list(predecessors or []),
        "successors": [],
    }


def test_build_fsa_dag_overlay_marks_startable_and_resource_blocked_roots() -> None:
    nodes = [
        _task("REQ_1_T1", "xarm6@localhost"),
        _task("REQ_2_T1", "xarm6@localhost"),
        _task("REQ_3_T1", "ur5e@localhost"),
    ]
    global_fsa = {
        "A": {
            "x0": "x0",
            "Tr": [
                {
                    "from": "x0",
                    "event": "REQ_1_T1.start",
                    "task_id": "REQ_1_T1",
                    "resource_jid": "xarm6@localhost",
                },
                {
                    "from": "x0",
                    "event": "REQ_3_T1.start",
                    "task_id": "REQ_3_T1",
                    "resource_jid": "ur5e@localhost",
                },
            ],
        },
        "meta": {
            "resource_requirement_blocks": {
                "xarm6@localhost": [
                    {
                        "block_id": "requirement:REQ_1",
                        "label": "REQ_1",
                        "requirement_id": "REQ_1",
                        "task_ids": ["REQ_1_T1"],
                    },
                    {
                        "block_id": "requirement:REQ_2",
                        "label": "REQ_2",
                        "requirement_id": "REQ_2",
                        "task_ids": ["REQ_2_T1"],
                    },
                ],
                "ur5e@localhost": [
                    {
                        "block_id": "requirement:REQ_3",
                        "label": "REQ_3",
                        "requirement_id": "REQ_3",
                        "task_ids": ["REQ_3_T1"],
                    }
                ],
            }
        },
    }

    overlay = build_fsa_dag_overlay(nodes, global_fsa)

    assert overlay["REQ_1_T1"]["label_suffix"] == "FSA-startable"
    assert overlay["REQ_3_T1"]["label_suffix"] == "FSA-startable"
    assert overlay["REQ_2_T1"]["label_suffix"].startswith(
        "resource-blocked:xarm6@localhost"
    )
    assert "REQ_1 -> REQ_2" in overlay["REQ_2_T1"]["label_suffix"]

    mermaid = nodes_to_mermaid(nodes, overlay=overlay)

    assert "FSA-startable" in mermaid
    assert "resource-blocked:xarm6@localhost [REQ_1 -> REQ_2]" in mermaid


def test_nodes_to_mermaid_labels_part_and_robot() -> None:
    mermaid = nodes_to_mermaid(
        [
            {
                "id": "REQ_4_T1",
                "type": "task",
                "function_name": "pick_approach",
                "resource_jid": "xarm6-2@localhost",
                "params": {"part_name": "MG"},
                "status": "pending",
                "predecessors": [],
                "successors": [],
            }
        ]
    )

    assert "part: MG | robot: xarm6-2" in mermaid

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.agents.intelligent_product.product_recovery_controller import (
    ProductRecoveryController,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn_primitive_generation import (
    _active_primitive_outline_event,
    _missing_primitive_outline_events,
)


class _FakeResource:
    def __init__(self, jid: str) -> None:
        self.jid = jid
        self.static_capabilities = {"resource_type": "robot"}

    def get_bridge_snapshot(self) -> dict[str, object]:
        return {
            "resource_jid": self.jid,
            "resource_type": "robot",
            "resource_core": {
                "resource_type": "robot",
                "current_state": "idle",
                "current_location": "home",
            },
            "current_state": "idle",
            "current_location": "home",
        }


class _FakeProduct:
    def __init__(self) -> None:
        self.agent_name = "test-product"
        self.jid = "product@localhost"
        self.logger = logging.getLogger("test.parallel_bridge_recovery")
        self._runtime_repair_max_attempts = 3
        self._runtime_bridge_mode = "manual"
        self._runtime_bridge_validation_policy = "validated"
        self._runtime_bridge_start_safety_mode = ""
        self._runtime_bridge_execution_shape = ""
        self._runtime_bridge_archive_path = ""
        self._runtime_bridge_archive_label = ""
        self.global_fsa_path = str(Path("/tmp/test_parallel_bridge_global_fsa.json"))
        self._runtime_recovery_context: dict[str, object] = {}
        self.task_states: dict[str, str] = {}
        self.part_tracker: dict[str, dict[str, object]] = {}
        self.execution_timeline: list[dict[str, object]] = []
        self._orphaned_bridge_task_warning_ids: set[str] = set()
        self.resource_agents = [
            _FakeResource("xarm6@localhost"),
            _FakeResource("ur5e@localhost"),
        ]
        self.process_planner = ProcessPlanner(self, self.resource_agents)
        self.recovery_controller = ProductRecoveryController(self)
        self.recovery_controller.bind_methods()
        self.runtime_recovery = self._empty_runtime_recovery()

    @staticmethod
    def _violation_summary(_violations: list[dict[str, object]]) -> tuple[list[str], int]:
        return [], 0

    @staticmethod
    def _event_guard_violations(
        _task_node: dict[str, object],
        *,
        plant_state: dict[str, object],
        enforce_unknown: bool = False,
    ) -> list[dict[str, object]]:
        del plant_state, enforce_unknown
        return []

    def _record_runtime_des_trace(self, **_kwargs: object) -> None:
        return None

    @staticmethod
    def _try_compile_controllable_repair(**_kwargs: object) -> None:
        return None

    def _mark_runtime_des_human_required(self, **_kwargs: object) -> None:
        return None

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


def _task(
    task_id: str,
    *,
    resource_jid: str,
    status: str = "pending",
    predecessors: list[str] | None = None,
    sequence_index: int = 0,
    bridge_sequence_id: str = "",
    bridge_sequence_index: int = 0,
    bridge_outline_id: str = "",
) -> dict[str, object]:
    return {
        "id": task_id,
        "type": "task",
        "function_name": "noop",
        "params": {"task_id": task_id},
        "resource_jid": resource_jid,
        "status": status,
        "predecessors": list(predecessors or []),
        "successors": [],
        "sequence_index": sequence_index,
        "bridge_sequence_id": bridge_sequence_id,
        "bridge_sequence_index": bridge_sequence_index,
        "bridge_outline_id": bridge_outline_id,
    }


def test_active_primitive_outline_event_returns_first_missing_event() -> None:
    session_state = {
        "accepted_outline_prefix": [
            {"outline_id": "a", "resource_jid": "xarm6@localhost"},
            {"outline_id": "b", "resource_jid": "ur5e@localhost"},
            {"outline_id": "c", "resource_jid": "xarm6@localhost"},
        ],
        "accepted_primitive_program": [
            {"outline_id": "a", "resource_jid": "xarm6@localhost"},
            {"outline_id": "c", "resource_jid": "xarm6@localhost"},
        ],
        "primitive_generation_cursor": 0,
    }

    cursor, active_event, _accepted_prefix = _active_primitive_outline_event(session_state)

    assert cursor == 1
    assert active_event == {"outline_id": "b", "resource_jid": "ur5e@localhost"}
    assert _missing_primitive_outline_events(session_state) == [
        {"outline_id": "b", "resource_jid": "ur5e@localhost"}
    ]


def test_apply_bridge_macro_proposal_compiles_depends_on_as_parallel_dag() -> None:
    product = _FakeProduct()
    planner = product.process_planner
    planner.nodes = [
        _task(
            "ANCHOR",
            resource_jid="ur5e@localhost",
            status="completed",
            sequence_index=1,
        )
    ]
    proposal = {
        "start_safety_mode": "cca_check",
        "macro_tasks": [
            {
                "outline_id": "clear_xarm6_zone",
                "depends_on": [],
                "resource_jid": "xarm6@localhost",
                "macro_name": "clear_xarm6_zone",
                "primitive_steps": [{"primitive": "move_to_named_pose", "params": {}}],
            },
            {
                "outline_id": "return_mcp_to_printer",
                "depends_on": [],
                "resource_jid": "ur5e@localhost",
                "macro_name": "return_mcp_to_printer",
                "primitive_steps": [{"primitive": "open_gripper", "params": {}}],
            },
            {
                "outline_id": "pick_lg",
                "depends_on": ["clear_xarm6_zone", "return_mcp_to_printer"],
                "resource_jid": "ur5e@localhost",
                "macro_name": "pick_lg",
                "primitive_steps": [{"primitive": "close_gripper", "params": {}}],
            },
        ],
    }

    tasks = planner.apply_bridge_macro_proposal(proposal, anchor_task_id="ANCHOR")
    tasks_by_outline_id = {
        str(task.get("bridge_outline_id") or ""): task
        for task in tasks
    }

    assert tasks_by_outline_id["clear_xarm6_zone"]["predecessors"] == ["ANCHOR"]
    assert tasks_by_outline_id["return_mcp_to_printer"]["predecessors"] == ["ANCHOR"]
    assert set(tasks_by_outline_id["pick_lg"]["predecessors"]) == {
        str(tasks_by_outline_id["clear_xarm6_zone"]["id"]),
        str(tasks_by_outline_id["return_mcp_to_printer"]["id"]),
    }
    assert tasks_by_outline_id["pick_lg"]["params"]["start_safety_mode"] == "cca_check"


def test_apply_bridge_macro_proposal_preserves_legacy_serial_order() -> None:
    product = _FakeProduct()
    planner = product.process_planner
    planner.nodes = [
        _task(
            "ANCHOR",
            resource_jid="ur5e@localhost",
            status="completed",
            sequence_index=1,
        )
    ]
    proposal = {
        "macro_tasks": [
            {
                "resource_jid": "xarm6@localhost",
                "macro_name": "first",
                "primitive_steps": [{"primitive": "move_to_named_pose", "params": {}}],
            },
            {
                "resource_jid": "ur5e@localhost",
                "macro_name": "second",
                "primitive_steps": [{"primitive": "open_gripper", "params": {}}],
            },
        ],
    }

    tasks = planner.apply_bridge_macro_proposal(proposal, anchor_task_id="ANCHOR")

    assert tasks[0]["predecessors"] == ["ANCHOR"]
    assert tasks[1]["predecessors"] == [str(tasks[0]["id"])]


def test_select_runtime_event_allows_nominal_on_unclaimed_resource_during_bridge() -> None:
    product = _FakeProduct()
    planner = product.process_planner
    planner.nodes = [
        _task(
            "BRIDGE_X",
            resource_jid="xarm6@localhost",
            status="completed",
            sequence_index=1,
            bridge_sequence_id="SEQ_1",
            bridge_sequence_index=1,
            bridge_outline_id="clear_xarm6_zone",
        ),
        _task(
            "BRIDGE_U",
            resource_jid="ur5e@localhost",
            status="running",
            sequence_index=2,
            bridge_sequence_id="SEQ_1",
            bridge_sequence_index=2,
            bridge_outline_id="return_mcp_to_printer",
        ),
        _task(
            "NOMINAL_X",
            resource_jid="xarm6@localhost",
            status="pending",
            sequence_index=10,
        ),
        _task(
            "NOMINAL_U",
            resource_jid="ur5e@localhost",
            status="pending",
            sequence_index=11,
        ),
    ]
    product.runtime_recovery = product._set_runtime_recovery(
        reset=True,
        status="resolved",
        bridge_approval_state="approved",
        active_bridge_sequence={
            "bridge_sequence_id": "SEQ_1",
            "bridge_task_ids": ["BRIDGE_X", "BRIDGE_U"],
            "state": "executing",
            "execution_policy": {"complete_full_tail": True},
        },
    )

    selected = product._select_runtime_event()

    assert selected is not None
    assert str(selected.get("id") or "") == "NOMINAL_X"


def test_select_runtime_event_blocks_nominal_on_claimed_resource() -> None:
    product = _FakeProduct()
    planner = product.process_planner
    planner.nodes = [
        _task(
            "BRIDGE_U",
            resource_jid="ur5e@localhost",
            status="running",
            sequence_index=1,
            bridge_sequence_id="SEQ_2",
            bridge_sequence_index=1,
            bridge_outline_id="return_mcp_to_printer",
        ),
        _task(
            "NOMINAL_U",
            resource_jid="ur5e@localhost",
            status="pending",
            sequence_index=10,
        ),
    ]
    product.runtime_recovery = product._set_runtime_recovery(
        reset=True,
        status="resolved",
        bridge_approval_state="approved",
        active_bridge_sequence={
            "bridge_sequence_id": "SEQ_2",
            "bridge_task_ids": ["BRIDGE_U"],
            "state": "executing",
            "execution_policy": {"complete_full_tail": True},
        },
    )

    selected = product._select_runtime_event()

    assert selected is None

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("OPENAI_API_KEY", "sk-local-test")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent


class _FakeController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._last_failure_message = ""
        self.release_descend_time_scale = 1.35
        self.controller_config = {"motion": {}}
        self.pick_targets = {
            "success": True,
            "part_name": "MCP",
            "model_name": "circ_pin_medium",
            "tx": 0.24,
            "ty": -0.18,
            "tz": 1.02,
            "pick_z": 1.11,
            "travel_z": 1.28,
            "part_height": 0.08,
            "tcp_offset_z": -0.17,
            "pick_tcp_z": 0.94,
            "start_x": 0.02,
            "start_y": 0.03,
            "start_z": 1.31,
        }
        self.place_targets = {
            "success": True,
            "slot_x": 0.05,
            "slot_y": -0.08,
            "board_top_z": 1.025,
            "place_z": 1.12,
            "place_tcp_z": 0.95,
            "part_height": 0.08,
            "model_name": "circ_pin_medium",
        }
        self.fail_move_above_message: str | None = None
        self.home_message = "returned to remembered start pose"
        self.current_pose = {
            "x": 0.02,
            "y": 0.03,
            "z": 1.31,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
        self.detected_parts: list[dict[str, Any]] = [
            {
                "part_name": "SG",
                "model_name": "spur_gear",
                "x": 0.31,
                "y": -0.12,
                "z": 0.15,
            }
        ]

    def is_usable(self) -> bool:
        return True

    def compute_pick_targets(self, part_name: str, product_geometry: dict[str, Any] | None) -> dict[str, Any]:
        self.calls.append(
            (
                "compute_pick_targets",
                {
                    "part_name": part_name,
                    "product_geometry": product_geometry,
                },
            )
        )
        targets = dict(self.pick_targets)
        targets["part_name"] = part_name or targets["part_name"]
        return targets

    def open_gripper(self) -> bool:
        self.calls.append(("open_gripper", {}))
        self._last_failure_message = ""
        return True

    def close_gripper(self) -> bool:
        self.calls.append(("close_gripper", {}))
        self._last_failure_message = ""
        return True

    def attach_part(self, model_name: str) -> dict[str, Any]:
        self.calls.append(("attach_part", {"model_name": model_name}))
        self._last_failure_message = ""
        return {"success": True, "message": f"attached {model_name}"}

    def compute_place_targets(
        self,
        pick_ctx: dict[str, Any],
        product_geometry: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "compute_place_targets",
                {
                    "pick_ctx": dict(pick_ctx),
                    "product_geometry": product_geometry,
                },
            )
        )
        return dict(self.place_targets)

    def _move_xy_at_z(
        self,
        x: float,
        y: float,
        z: float,
        *,
        label: str,
        speed: float | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "_move_xy_at_z",
                {"x": x, "y": y, "z": z, "label": label, "speed": speed},
            )
        )
        if self.fail_move_above_message and label == "Move above part":
            self._last_failure_message = self.fail_move_above_message
            return {"success": False, "message": ""}
        self._last_failure_message = ""
        return {"success": True, "message": f"ok {label}"}

    def _move_pose_direct(
        self,
        x: float,
        y: float,
        z: float,
        *,
        label: str,
        speed: float | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "_move_pose_direct",
                {"x": x, "y": y, "z": z, "label": label, "speed": speed},
            )
        )
        self._last_failure_message = ""
        return {"success": True, "message": f"ok {label}"}

    def _release_part_sequence(
        self,
        *,
        model_name: str,
        slot_x: float,
        slot_y: float,
        part_height: float,
        board_top_z: float,
        place_z: float,
        travel_z: float,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "_release_part_sequence",
                {
                    "model_name": model_name,
                    "slot_x": slot_x,
                    "slot_y": slot_y,
                    "part_height": part_height,
                    "board_top_z": board_top_z,
                    "place_z": place_z,
                    "travel_z": travel_z,
                },
            )
        )
        self._last_failure_message = ""
        return {"success": True, "message": "released part and lifted clear"}

    def move_home(self) -> dict[str, Any]:
        self.calls.append(("move_home", {}))
        self._last_failure_message = ""
        return {"success": True, "message": self.home_message}

    def move_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        speed: float | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("move_cartesian", {"x": x, "y": y, "z": z, "speed": speed}))
        self.current_pose.update({"x": x, "y": y, "z": z})
        self._last_failure_message = ""
        return {"success": True, "message": f"moved to ({x}, {y}, {z})"}

    def detect_parts(self, part_name: str | None = None) -> list[dict[str, Any]]:
        self.calls.append(("detect_parts", {"part_name": part_name}))
        if not part_name:
            return [dict(item) for item in self.detected_parts]
        return [
            dict(item)
            for item in self.detected_parts
            if str(item.get("part_name") or "") == str(part_name)
        ]

    def get_current_pose(self) -> dict[str, Any]:
        self.calls.append(("get_current_pose", {}))
        return {"success": True, "message": "current pose", "pose": dict(self.current_pose)}


def _build_agent(controller: _FakeController | None = None) -> tuple[RobotAgent, _FakeController]:
    fake = controller or _FakeController()
    agent = RobotAgent(
        "ur5e@localhost",
        "pw",
        name="ur5e",
        instructions="test robot agent",
        execution_mode="simulation",
        controller_config={"motion": {}},
        named_positions={"home": [0, 1, 2, 3, 4, 5]},
        prewarmed_controller=fake,
        sg_slippage_mode="off",
        sg_slippage_scope="ur5e",
    )
    return agent, fake


def test_pick_approach_uses_controller_helpers_and_updates_state() -> None:
    agent, controller = _build_agent()

    result = asyncio.run(
        agent.pick_approach(
            origin_resource_location="prusa-mk4-2",
            part_name="MCP",
            speed=0.8,
            product_geometry={"model_name": "circ_pin_medium"},
        )
    )

    assert result["status"] == "completed"
    assert agent._current_state == "at_pick"
    assert agent._position == {"x": 0.24, "y": -0.18, "z": 1.11}
    assert agent._pick_ctx["travel_z"] == 1.28
    assert agent._pick_ctx["start_x"] == 0.02
    assert [name for name, _payload in controller.calls] == [
        "compute_pick_targets",
        "open_gripper",
        "_move_xy_at_z",
        "_move_pose_direct",
    ]
    assert controller.calls[2][1]["label"] == "Move above part"
    assert controller.calls[2][1]["speed"] == 0.8
    assert "Descend to pick" in controller.calls[3][1]["label"]


def test_pick_approach_returns_rich_failure_context() -> None:
    controller = _FakeController()
    controller.fail_move_above_message = "[Move above part] planning fraction too low: 0.200 < 0.900"
    agent, controller = _build_agent(controller)

    result = asyncio.run(
        agent.pick_approach(
            origin_resource_location="prusa-mk4-2",
            part_name="MCP",
            product_geometry={"model_name": "circ_pin_medium"},
        )
    )

    assert result["status"] == "failed"
    assert "planning fraction too low" in result["content"]
    assert result["failure_context"]["failure_mode"] == "unreachable"
    assert result["failure_context"]["observations"]["step"] == "pick_approach.move_above_part"
    assert [name for name, _payload in controller.calls] == [
        "compute_pick_targets",
        "open_gripper",
        "_move_xy_at_z",
    ]


def test_place_insert_and_move_home_preserve_split_task_path() -> None:
    agent, controller = _build_agent()
    agent._held_part = "MCP"
    agent._current_state = "picked"
    agent._gripper_state = "closed"
    agent._pick_ctx = {
        "part_name": "MCP",
        "model_name": "circ_pin_medium",
        "tx": 0.24,
        "ty": -0.18,
        "tz": 1.02,
        "pick_z": 1.11,
        "travel_z": 1.28,
        "part_height": 0.08,
        "tcp_offset_z": -0.17,
        "pick_tcp_z": 0.94,
        "start_x": 0.02,
        "start_y": 0.03,
        "start_z": 1.31,
    }

    approach = asyncio.run(
        agent.place_approach(
            destination_location="assembly-board-slot",
            part_name="MCP",
            speed=0.5,
            product_geometry={"slot_xy": [0.0, -0.08]},
        )
    )
    insert = asyncio.run(
        agent.place_insert(
            destination_location="assembly-board-slot",
            part_name="MCP",
        )
    )
    home = asyncio.run(agent.move_home())

    assert approach["status"] == "completed"
    assert insert["status"] == "completed"
    assert home["status"] == "completed"
    assert agent._current_state == "idle"
    assert agent._held_part is None
    assert agent._gripper_state == "open"
    assert agent._bridge_pose_ref is None
    assert agent._position == {"x": 0.02, "y": 0.03, "z": 1.31}

    call_names = [name for name, _payload in controller.calls]
    assert call_names == [
        "compute_place_targets",
        "_move_pose_direct",
        "_move_xy_at_z",
        "_move_pose_direct",
        "_release_part_sequence",
        "move_home",
        "get_current_pose",
    ]
    assert controller.calls[1][1]["label"] == "Lift with part"
    assert controller.calls[2][1]["label"] == "Move above destination"
    assert controller.calls[2][1]["speed"] == 0.5
    assert "Descend to place" in controller.calls[3][1]["label"]
    assert controller.calls[3][1]["speed"] == controller.release_descend_time_scale
    assert controller.calls[4][1]["model_name"] == "circ_pin_medium"


def test_execute_recovery_macro_binds_detected_part_pose_to_move_cartesian() -> None:
    agent, controller = _build_agent()
    executed: list[tuple[str, dict[str, Any]]] = []

    async def fake_execute_primitive(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        executed.append((primitive, dict(params)))
        if primitive == "detect_parts":
            part_name = str(params.get("part_name") or "")
            return {
                "success": True,
                "message": "detect_parts returned 1 items",
                "data": [
                    dict(item)
                    for item in controller.detected_parts
                    if str(item.get("part_name") or "") == part_name
                ],
            }
        if primitive == "move_cartesian":
            return {"success": True, "message": "moved to detected pose"}
        raise AssertionError(f"unexpected primitive {primitive}")

    agent._execute_primitive = fake_execute_primitive  # type: ignore[method-assign]

    result = asyncio.run(
        agent.execute_recovery_macro(
            "recover_slipped_part",
            [
                {
                    "primitive": "detect_parts",
                    "params": {"part_name": "SG"},
                    "store_as": "detected_sg",
                },
                {
                    "primitive": "move_cartesian",
                    "params": {
                        "x": {"context_ref": "/step_outputs/detected_sg/pose/x"},
                        "y": {"context_ref": "/step_outputs/detected_sg/pose/y"},
                        "z": {"context_ref": "/step_outputs/detected_sg/pose/z"},
                    },
                },
            ],
        )
    )

    assert result["status"] == "completed"
    assert [name for name, _payload in controller.calls] == [
        "get_current_pose",
    ]
    assert executed == [
        ("detect_parts", {"part_name": "SG"}),
        ("move_cartesian", {"x": 0.31, "y": -0.12, "z": 0.15}),
    ]


def test_execute_recovery_macro_rejects_ambiguous_detect_parts_store_as() -> None:
    controller = _FakeController()
    controller.detected_parts = [
        {
            "part_name": "SG",
            "model_name": "spur_gear",
            "x": 0.31,
            "y": -0.12,
            "z": 0.15,
        },
        {
            "part_name": "SG",
            "model_name": "spur_gear",
            "x": 0.33,
            "y": -0.14,
            "z": 0.16,
        },
    ]
    agent, controller = _build_agent(controller)
    executed: list[tuple[str, dict[str, Any]]] = []

    async def fake_execute_primitive(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        executed.append((primitive, dict(params)))
        if primitive == "detect_parts":
            part_name = str(params.get("part_name") or "")
            return {
                "success": True,
                "message": "detect_parts returned 2 items",
                "data": [
                    dict(item)
                    for item in controller.detected_parts
                    if str(item.get("part_name") or "") == part_name
                ],
            }
        raise AssertionError(f"unexpected primitive {primitive}")

    agent._execute_primitive = fake_execute_primitive  # type: ignore[method-assign]

    result = asyncio.run(
        agent.execute_recovery_macro(
            "recover_slipped_part",
            [
                {
                    "primitive": "detect_parts",
                    "params": {"part_name": "SG"},
                    "store_as": "detected_sg",
                },
                {
                    "primitive": "move_cartesian",
                    "params": {
                        "x": {"context_ref": "/step_outputs/detected_sg/pose/x"},
                        "y": {"context_ref": "/step_outputs/detected_sg/pose/y"},
                        "z": {"context_ref": "/step_outputs/detected_sg/pose/z"},
                    },
                },
            ],
        )
    )

    assert result["status"] == "failed"
    assert "exactly one result" in result["content"]
    assert [name for name, _payload in controller.calls] == [
        "get_current_pose",
    ]
    assert executed == [("detect_parts", {"part_name": "SG"})]


def test_execute_recovery_macro_fails_runtime_semantic_validation_before_first_step() -> None:
    agent, controller = _build_agent()
    executed: list[tuple[str, dict[str, Any]]] = []

    async def fake_execute_primitive(
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        executed.append((primitive, dict(params)))
        raise AssertionError("primitive execution should not start when semantic validation fails")

    agent._execute_primitive = fake_execute_primitive  # type: ignore[method-assign]

    result = asyncio.run(
        agent.execute_recovery_macro(
            "invalid_runtime_macro",
            [
                {
                    "primitive": "move_cartesian",
                    "params": {"x": 0.1, "y": 0.2},
                },
            ],
        )
    )

    assert result["status"] == "failed"
    assert "runtime semantic validation" in result["content"]
    assert "semantic_error" in result["observations"]
    assert executed == []
    assert [name for name, _payload in controller.calls] == [
        "get_current_pose",
    ]

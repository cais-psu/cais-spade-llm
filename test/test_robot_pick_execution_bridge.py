"""Focused tests for guarded physical UR5e robot-function execution."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.agents.resource_agent import robot_agent as robot_agent_module
from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent
from cais_spade_llm.resources.robot import robot_task_runtime
from cais_spade_llm.ui import bridge as bridge_module
from cais_spade_llm.ui.bridge import SystemBridge

ROOT = Path(__file__).resolve().parents[1]
ACTUAL_MG_STL = str(
    (ROOT / "ros2/cais_lab_robotics/cad_models/Gear_Medium.STL").resolve()
)
ACTUAL_MG_STL_SHA256 = "73d2c5d06497db2042ec6624a45558398ee47c9e55ccd87d49184c179a501507"
MANUAL_DESCEND_POSE = {
    "x": 0.3,
    "y": 0.4,
    "z": 1.1,
    "qx": 0.0,
    "qy": 1.0,
    "qz": 0.0,
    "qw": 0.0,
}
UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@pytest.fixture(autouse=True)
def _isolate_operator_insertion_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bridge_module,
        "_MOVE_INSERT_TRIALS_DIR",
        tmp_path / "operator_move_insert_trials",
    )
    monkeypatch.setattr(
        bridge_module,
        "_INSERTION_DEMONSTRATIONS_DIR",
        tmp_path / "operator_move_insert_demonstrations",
    )
    monkeypatch.setattr(
        bridge_module,
        "_HARDWARE_STATE_DIR",
        tmp_path / "operator_hardware_state",
    )
    monkeypatch.setattr(
        bridge_module,
        "_MOVE_INSERT_SERVER_TRACE_ROOT",
        tmp_path / "operator_move_insert_server_traces",
    )
    monkeypatch.setattr(
        bridge_module,
        "_INSERTION_DEMONSTRATION_TRACE_ROOT",
        tmp_path / "operator_insertion_demonstration_traces",
    )
    monkeypatch.setattr(
        bridge_module,
        "_UR5E_RTDE_TRAJECTORY_LAST_TERMINAL_STATUS",
        tmp_path / "operator_last_terminal_status.json",
    )
    monkeypatch.setattr(
        bridge_module,
        "_UR5E_RTDE_TRAJECTORY_STATUS",
        tmp_path / "operator_rtde_status.json",
    )


def _operator_confirmed_mg_handoff(
    part_name: str = "MG",
) -> dict[str, Any]:
    model_name = {
        "SG": "gear_small",
        "MG": "gear_medium",
        "LG": "gear_large",
        "SCP": "circ_pin_small",
        "MCP": "circ_pin_medium",
        "LCP": "circ_pin_large",
    }[part_name]
    current_tf_stamp_sec = time.time()
    return {
        "robot": "ur5e",
        "destination_location": "assembly_board-v1",
        "part_name": part_name,
        "model_name": model_name,
        "origin_resource_location": "prusa-mk4-2",
        "frame_id": "world",
        "tool_frame": "tool0",
        "part_frame": "held_part_origin",
        "source": "operator_confirmed_pick_approach_recording",
        "orientation_source": "realsense_roboflow_identity",
        "captured_at": current_tf_stamp_sec,
        "current_world_tool0_pose": deepcopy(MANUAL_DESCEND_POSE),
        "current_tf_stamp_sec": current_tf_stamp_sec,
        "pick_approach_recording_path": (
            "cais_spade_llm/resources/robot/taught_functions/"
            "pick_approach/default__hardware.json"
        ),
        "pick_approach_recording_sha256": "d" * 64,
        "pick_approach_recorded_at": 1_786_400_802.0,
        "world_tool0_pose_at_grasp": deepcopy(MANUAL_DESCEND_POSE),
        "world_held_part_pose_at_grasp": deepcopy(MANUAL_DESCEND_POSE),
        "tool0_to_held_part": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        "origin_pose_provenance": {
            "frame_id": "world",
            "part_name": part_name,
            "model_name": model_name,
            "source": "operator_confirmed_pick_approach_recording",
            "orientation_source": "realsense_roboflow_identity",
            "captured_at": 1_786_400_757.0,
            "pick_approach_recording_sha256": "d" * 64,
        },
    }


def _normal_mg_handoff() -> dict[str, Any]:
    """Return the immutable SE(3) handoff produced by a normal pick_grasp."""
    return {
        "part_name": "MG",
        "model_name": "gear_medium",
        "origin_resource_location": "prusa-mk4-2",
        "frame_id": "world",
        "tool_frame": "tool0",
        "part_frame": "held_part_origin",
        "source": "pick_grasp",
        "world_tool0_pose_at_grasp": deepcopy(MANUAL_DESCEND_POSE),
        "world_held_part_pose_at_grasp": deepcopy(MANUAL_DESCEND_POSE),
        "tool0_to_held_part": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        "origin_pose_provenance": {
            "frame_id": "world",
            "part_name": "MG",
            "model_name": "gear_medium",
        },
    }


def _move_insert_effective(part_name: str = "MG") -> dict[str, Any]:
    return {
        "part_name": part_name,
        "calibration_id": "move_insert-shared-v1",
        "shared_calibration_id": "move_insert-shared-v1",
        "override_calibration_id": None,
        "pre_insert_offset_m": 0.01,
        "contact_speed_m_s": 0.002,
        "contact_force_delta_n": 2.0,
        "engagement_progress_m": 0.002,
        "insertion_force_n": 8.0,
        "spiral_radius_m": 0.001,
        "spiral_pitch_m": 0.0005,
        "spiral_speed_m_s": 0.002,
        "spiral_acceleration_m_s2": 0.01,
        "max_axial_force_n": 20.0,
        "max_lateral_force_n": 10.0,
        "max_torque_nm": 2.0,
        "tilt_tolerance_rad": 0.1,
        "seated_depth_tolerance_m": 0.001,
        "settle_time_sec": 0.2,
    }


def _move_insert_settings(part_name: str = "MG") -> dict[str, Any]:
    hard_caps = _move_insert_test_hard_caps()
    return {
        "success": True,
        "validated": True,
        "profile_state": "ready",
        "profile_sha256": "a" * 64,
        "effective": _move_insert_effective(part_name),
        "derived_timeout_sec": 30.0,
        "hard_caps": hard_caps,
        "hard_caps_sha256": SystemBridge._move_insert_hard_caps_sha256(
            hard_caps
        ),
        "message": f"move_insert profile is ready for {part_name}.",
    }


def _move_insert_preinsert_hard_caps() -> dict[str, float]:
    return {
        "insert_max_travel_m": 0.02,
        "insert_start_position_tolerance_m": 0.003,
        "insert_start_orientation_tolerance_rad": 0.1,
        "insert_max_timeout_sec": 30.0,
    }


def _mg_place_product_geometry() -> dict[str, Any]:
    hard_caps = _move_insert_test_hard_caps()
    return {
        **_mg_product_geometry(),
        "move_insert_hard_caps": hard_caps,
        "move_insert_hard_caps_sha256": (
            SystemBridge._move_insert_hard_caps_sha256(hard_caps)
        ),
    }


def _move_insert_test_hard_caps() -> dict[str, float]:
    rtde_config = deepcopy(_move_insert_hardware_config()["ur5e"]["rtde"])
    mg_hard_caps = dict(rtde_config.pop("MG"))
    return {
        cap_name: float(
            mg_hard_caps[cap_name]
            if cap_name in mg_hard_caps
            else rtde_config[cap_name]
        )
        for cap_name in (
            *bridge_module._MOVE_INSERT_REQUIRED_HARD_CAPS,
            *bridge_module._MOVE_INSERT_MG_TACTILE_POLICY_CAPS,
        )
    }


def _mg_product_geometry(part_name: str = "MG") -> dict[str, Any]:
    model_name = {
        "SG": "gear_small",
        "MG": "gear_medium",
        "LG": "gear_large",
        "SCP": "circ_pin_small",
        "MCP": "circ_pin_medium",
        "LCP": "circ_pin_large",
    }[part_name]
    return {
        "part_name": part_name,
        "model_name": model_name,
        "part_height_m": 0.02,
        "source_stl": ACTUAL_MG_STL,
        "source_stl_sha256": ACTUAL_MG_STL_SHA256,
        "hub_up": True,
        "hub_diameter_m": 0.03,
        "hub_height_m": 0.01,
        "tooth_diameter_m": 0.042,
        "tooth_height_m": 0.01,
        "grasp_width_m": 0.028,
        "tooth_clearance_m": 0.002,
        "minimum_hub_overlap_m": 0.006,
        "slot_xy": [0.0, 0.08],
        "slot_floor_z_m": 1.025,
        "board_center": {"x": 0.0, "y": 0.0, "z": 1.02},
        "assembly_board-v1_aruco_to_assembly_board-v1": {
            "calibration_id": "assembly-board-registration-test-v1",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        "target_reference": {
            "source": "slot_geometry",
            "target_point": "inserted_part_origin",
            "surface_role": "assembly_slot",
        },
    }


def _mg_task_context() -> dict[str, Any]:
    return {
        **_mg_product_geometry(),
        "origin_resource_location": "prusa-mk4-2",
        "pick_tcp_z": 1.0323,
        "finger_tooth_clearance_m": 0.003,
        "finger_hub_overlap_m": 0.007,
        "pick_z_adjustment_m": 0.001,
        "pick_tool0_z_adjustment_m": 0.005,
        "open_gripper_position": 0.11,
        "mg_gripper_close_position": 0.047,
        "open_inner_pad_lower_z_from_tcp_m": 0.01751,
        "open_inner_pad_upper_z_from_tcp_m": 0.04726,
        "closed_inner_pad_lower_z_from_tcp_m": -0.00865,
        "closed_inner_pad_upper_z_from_tcp_m": 0.0211,
        "predicted_closing_z_displacement_m": -0.02616,
        "gripper_close_position": 0.047,
        "travel_z": 1.2,
        "resolved_cartesian_positions": {
            "descend": deepcopy(MANUAL_DESCEND_POSE),
        },
    }


class _PhysicalController:
    gripper_open = 0.11
    gripper_close = 0.02

    def __init__(self) -> None:
        self.gripper_calls: list[tuple[str, float | None]] = []
        self.close_success = True
        self.reopen_success = True
        self._last_failure_message = ""
        self._ur5e_hardware_insert_action = (
            "/cais_ur5e_rtde_cartesian_controller/move_insert"
        )
        self._ur5e_hardware_insert_client: Any | None = None
        self.move_insert_prepare_timeouts: list[float] = []

    def _ensure_move_insert_client_ready(
        self,
        *,
        timeout_sec: float = 2.0,
    ) -> tuple[bool, str]:
        self.move_insert_prepare_timeouts.append(timeout_sec)
        if self._ur5e_hardware_insert_client is None:
            self._ur5e_hardware_insert_client = SimpleNamespace(
                wait_for_server=lambda *, timeout_sec: True,
            )
        return True, ""

    def close_gripper(self, position: float | None = None) -> bool:
        self.gripper_calls.append(("close_gripper", position))
        if not self.close_success:
            self._last_failure_message = "close failed"
        return self.close_success

    def open_gripper(self) -> bool:
        self.gripper_calls.append(("open_gripper", None))
        if not self.reopen_success:
            self._last_failure_message = "reopen failed"
        return self.reopen_success

    @staticmethod
    def _physical_stl_pick_readiness(geometry: dict[str, Any]) -> dict[str, Any]:
        if not geometry.get("source_stl"):
            return {"success": False, "message": "actual source_stl is missing"}
        return {
            "success": True,
            **geometry,
            "gripper_close_position": 0.047,
            "finger_tooth_clearance_m": 0.003,
            "finger_hub_overlap_m": 0.007,
            "pick_z_adjustment_m": 0.001,
            "open_gripper_position": 0.11,
            "mg_gripper_close_position": 0.047,
            "predicted_closing_z_displacement_m": -0.02616,
        }


class _PhysicalUR5eAgent:
    def __init__(
        self,
        *,
        state: str = "idle",
        held_part: str | None = None,
        gripper_state: str = "open",
    ) -> None:
        self.agent_name = "ur5e"
        self.jid = "ur5e@localhost"
        self.execution_mode = "physical"
        self._controller = _PhysicalController()
        self.controller_config = {
            "arm_joint_names": list(UR5E_JOINT_NAMES),
            "move_group": {
                "frame_id": "world",
                "ee_link": "tool0",
                "tcp_link": "ur5e_rg2_gripper_tcp",
            },
        }
        self.named_positions = {
            "home": [0.0, -1.0, -2.0, -1.5, 1.5, 0.0],
            "prusa-mk4-2": [0.1, -0.8, -2.1, -1.6, 1.5, -3.1],
            "assembly_board-v1": [0.2, -0.9, -2.0, -1.4, 1.5, -3.0],
        }
        self._current_state = state
        self._held_part = held_part
        self._gripper_state = gripper_state
        self._position: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 1.0}
        self._task_ctx: dict[str, Any] = {}
        self._robot_motion_lock = threading.Lock()
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.operator_confirmed_held_part_calls: list[bool] = []
        self.operator_confirmed_held_part_handoff_calls: list[
            dict[str, Any] | None
        ] = []
        self.executables = {
            name: getattr(self, name)
            for name in (
                "pick_approach",
                "pick_grasp",
                "place_approach",
                "place_insert",
                "move_home",
            )
        }

    @staticmethod
    def is_alive() -> bool:
        return True

    async def pick_approach(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("pick_approach", kwargs))
        self._task_ctx = {
            **_mg_task_context(),
            "part_name": kwargs["part_name"],
            "origin_resource_location": kwargs["origin_resource_location"],
        }
        self._current_state = "at_pick"
        return {"status": "completed", "content": "Arrived at the live pick target."}

    async def pick_grasp(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("pick_grasp", kwargs))
        self._task_ctx["held_part_handoff"] = _normal_mg_handoff()
        self._held_part = kwargs["part_name"]
        self._gripper_state = "closed"
        self._current_state = "picked"
        return {"status": "completed", "content": "Picked MG."}

    async def place_approach(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("place_approach", kwargs))
        self._task_ctx["destination_location"] = kwargs["destination_location"]
        product_geometry = dict(kwargs.get("product_geometry") or {})
        if product_geometry.get("move_insert_profile"):
            self._task_ctx["move_insert_profile"] = deepcopy(
                product_geometry["move_insert_profile"]
            )
            self._task_ctx["move_insert_profile_sha256"] = str(
                product_geometry.get("move_insert_profile_sha256") or ""
            )
        self._task_ctx["resolved_cartesian_positions"] = {
            "descend": deepcopy(MANUAL_DESCEND_POSE),
        }
        if kwargs["destination_location"] == "assembly_board-v1":
            self._task_ctx["assembly_board_v1_aruco_generation"] = 1
            self._task_ctx["assembly_board_v1_aruco"] = {
                "destination_location": "assembly_board-v1",
                "camera_role": "ur5e",
                "generation": 1,
                "calibration_id": "ur5e-calibration",
            }
        self._current_state = "positioned"
        return {"status": "completed", "content": "Reached assembly_board-v1."}

    async def place_insert(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("place_insert", kwargs))
        self._held_part = None
        self._gripper_state = "open"
        self._current_state = "placed"
        self._task_ctx = {}
        return {"status": "completed", "content": "Assembled MG."}

    async def move_home(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("move_home", kwargs))
        self._current_state = "idle"
        self._task_ctx = {}
        return {"status": "completed", "content": "At home position."}

    async def _execute_registered_robot_task_for_manual_function_execution(
        self,
        function_name: str,
        pre_execute: Any = None,
        post_staging_acceptance: Any = None,
        /,
        *,
        operator_confirmed_held_part: bool = False,
        operator_confirmed_held_part_handoff: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.operator_confirmed_held_part_calls.append(
            operator_confirmed_held_part
        )
        self.operator_confirmed_held_part_handoff_calls.append(
            deepcopy(operator_confirmed_held_part_handoff)
        )
        if callable(pre_execute):
            pre_execute_error = str(pre_execute() or "").strip()
            if pre_execute_error:
                return {
                    "status": "blocked",
                    "content": pre_execute_error,
                    "manual_pre_execute_blocked": True,
                }
        callback_name = "_assembly_board_v1_post_staging_accept_callback"
        install_callback = bool(
            callable(post_staging_acceptance)
            and function_name == "place_approach"
            and kwargs.get("destination_location") == "assembly_board-v1"
        )
        if install_callback:
            setattr(self._controller, callback_name, post_staging_acceptance)
        if operator_confirmed_held_part:
            handoff = deepcopy(operator_confirmed_held_part_handoff or {})
            self._current_state = "picked"
            self._held_part = kwargs["part_name"]
            self._gripper_state = "closed"
            self._task_ctx = {
                "part_name": handoff.get("part_name"),
                "model_name": handoff.get("model_name"),
                "origin_resource_location": handoff.get(
                    "origin_resource_location"
                ),
                "origin_pose": deepcopy(
                    handoff.get("world_held_part_pose_at_grasp") or {}
                ),
                "origin_pose_provenance": deepcopy(
                    handoff.get("origin_pose_provenance") or {}
                ),
                "resolved_cartesian_positions": {
                    "descend": deepcopy(
                        handoff.get("world_tool0_pose_at_grasp") or {}
                    )
                },
                "held_part_handoff": handoff,
            }
        try:
            result = await getattr(self, function_name)(**kwargs)
            if operator_confirmed_held_part:
                result = {
                    **result,
                    "operator_confirmed_held_part": True,
                    "operator_held_part": kwargs["part_name"],
                    "held_part_handoff_adopted": True,
                    "move_insert_trial_context_ready": True,
                    "move_insert_authorized": False,
                }
            return result
        finally:
            if install_callback:
                delattr(self._controller, callback_name)


def _healthy_status(target: str) -> dict[str, Any]:
    return {
        "target": target,
        "repair_needed": False,
        "repair_reason": "",
        "gazebo": {"status": "running"},
        "hardware": {"overall": "running"},
        "sync/status": {"process_status": "running", "state": "mirroring"},
    }


def _ready_bridge(agent: _PhysicalUR5eAgent) -> SystemBridge:
    bridge = object.__new__(SystemBridge)
    bridge.system_running = True
    bridge.execution_mode = "physical"
    bridge.selected_product = "product.json"
    bridge.resource_agents = [agent]
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_preflight_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._ur5e_robot_function_execution_stage = ""
    bridge._ur5e_robot_function_execution_started_at = 0.0
    bridge._robot_function_execution_robot = ""
    bridge._robot_function_execution_active_step = ""
    bridge._assembly_move_insert_profile_sha256 = ""
    bridge._assembly_move_insert_effective = {}
    bridge._ur5e_robot_function_state_uncertain = False
    bridge._MG_CLOSE_TEST_HOLD_S = 0.0
    bridge._digital_twin_sim_mode = lambda _target: "monitor"
    bridge._robot_function_capture_source = lambda _target, _cfg: "hardware"
    bridge._digital_twin_direction = lambda _target: "hardware -> gazebo"
    bridge.digital_twin_statuses = lambda: {
        target: _healthy_status(target) for target in ("ur5e only", "dual robots")
    }
    bridge._digital_twin_ur5e_motion_readiness = lambda _target, _cfg, _agent=None: (
        {
            "hardware_domain_id": 42,
            "trajectory_action_ready": True,
            "rtde_receive_connected": True,
            "joint_states_fresh": True,
            "rtde_control_connected": True,
            "cartesian_position_tolerance_m": 0.002,
            "cartesian_orientation_tolerance_rad": 0.05,
        },
        "",
    )
    bridge._digital_twin_ur5e_gripper_readiness = lambda _target, _domain, _agent=None: (
        {"gripper_action_ready": True},
        "",
    )
    bridge._robot_function_capture_snapshot = lambda _target, _robot: {
        "success": True,
        "world_tool0_ready": True,
        "tf_stamp_sec": time.time(),
        "blocked_reason": "",
        "waypoint": {
            "source": "hardware",
            "pose": {
                **deepcopy(MANUAL_DESCEND_POSE),
                "frame_id": "world",
                "child_frame_id": "link_eef" if _robot == "xarm6" else "tool0",
            },
        },
    }
    bridge.physical_perception_ready = lambda: (True, "")
    bridge._robot_function_product_geometry_for_part = _mg_product_geometry
    bridge._operator_confirmed_mg_held_part_handoff = (
        lambda _target, _agent, *, origin_resource_location, part_name, product_geometry: (
            _operator_confirmed_mg_handoff(part_name),
            {
                "operator_confirmed_held_part": True,
                "operator_held_part": part_name,
                "held_part_handoff_ready": True,
                "move_insert_trial_context_ready": False,
                "move_insert_authorized": False,
            },
            "",
        )
    )
    bridge._digital_twin_place_approach_recording_error = lambda _agent, _destination, _part: (
        f"/tmp/assembly_board-v1__{_part}__hardware.json",
        "",
    )
    bridge.digital_twin_move_insert_settings = (
        lambda _target, _robot, *, destination_location, part_name: (
            _move_insert_settings(part_name)
        )
    )
    bridge._digital_twin_move_insert_live_readiness = (
        lambda _settings, _agent=None: (
            {
                "insert_action_ready": True,
                "tcp_force_feedback_ready": True,
                "insert_function_ready": True,
            },
            "",
        )
    )
    bridge._move_insert_preinsert_hard_caps_readiness = lambda _part_name="": (
        {
            "move_insert_hard_caps": _move_insert_test_hard_caps(),
            "move_insert_live_hard_caps": _move_insert_test_hard_caps(),
            "move_insert_hard_caps_sha256": (
                SystemBridge._move_insert_hard_caps_sha256(
                    _move_insert_test_hard_caps()
                )
            ),
        },
        "",
    )
    bridge._move_insert_normal_qualification_readiness = (
        lambda *, destination_location, part_name, settings, resource_agent=None: (
            {
                "qualified": True,
                "qualification": {
                    "robot": "ur5e",
                    "destination_location": destination_location,
                    "part_name": part_name,
                    "profile_sha256": str(settings.get("profile_sha256") or ""),
                },
                "move_insert_qualification_identities": {},
            },
            "",
        )
    )
    bridge._ur5e_rtde_trajectory_status = lambda: {}
    bridge.perception_manager = SimpleNamespace(
        assembly_board_v1_aruco_status=lambda role: {
            "camera_role": role,
            "accepted": True,
            "accepted_baseline_ready": True,
            "accepted_baseline_error": "",
            "accepted_generation": 1,
            "accepted_calibration_id": f"{role}-calibration",
            "active_calibration_id": f"{role}-calibration",
            "calibration_changed": False,
            "movement_evidence_valid": False,
            "movement_blocked": False,
            "frame_id": "world",
            "pose": deepcopy(MANUAL_DESCEND_POSE),
            "accepted_pose": deepcopy(MANUAL_DESCEND_POSE),
        }
    )

    async def _run_on_agent_runtime(coroutine: Any) -> Any:
        return await coroutine

    bridge._run_on_agent_runtime = _run_on_agent_runtime
    return bridge


def _ready_assembly_bridge(agent: _PhysicalUR5eAgent | None = None) -> SystemBridge:
    resource_agent = agent or _PhysicalUR5eAgent()
    bridge = _ready_bridge(resource_agent)
    bridge.system_running = False
    bridge._starting = False
    bridge._stopping = False
    bridge.last_error = None
    bridge._ur5e_robot_function_agent = resource_agent
    bridge._ur5e_robot_function_agent_domain_id = 42
    bridge._ui_process_ros_domain_id = lambda: 42
    bridge._robot_function_execution_robot = ""
    bridge._robot_function_execution_active_step = ""
    bridge._assembly_move_insert_profile_sha256 = ""
    bridge._assembly_move_insert_effective = {}
    bridge._ur5e_robot_function_state_uncertain_reason = ""
    bridge._digital_twin_function_steps = {}
    bridge._digital_twin_record_lock = threading.Lock()
    bridge.digital_twin_list_function_buffer_steps = lambda *_args, **_kwargs: []
    bridge.digital_twin_list_function_file_steps = lambda *_args, **_kwargs: []
    bridge.digital_twin_move_insert_settings = (
        lambda _target, _robot, *, destination_location, part_name: (
            _move_insert_settings(part_name)
        )
    )
    bridge._digital_twin_move_insert_live_readiness = (
        lambda _settings, _agent=None: (
            {
                "insert_action_ready": True,
                "tcp_force_feedback_ready": True,
                "insert_function_ready": True,
            },
            "",
        )
    )
    return bridge


def _agent_for(function_name: str) -> _PhysicalUR5eAgent:
    if function_name == "pick_approach":
        return _PhysicalUR5eAgent()
    if function_name == "pick_grasp":
        agent = _PhysicalUR5eAgent(state="at_pick")
        agent._task_ctx = _mg_task_context()
        return agent
    if function_name == "place_approach":
        agent = _PhysicalUR5eAgent(
            state="picked",
            held_part="MG",
            gripper_state="closed",
        )
        agent._task_ctx = {
            **_mg_task_context(),
            "held_part_handoff": _normal_mg_handoff(),
        }
        return agent
    if function_name == "place_insert":
        agent = _PhysicalUR5eAgent(state="positioned", held_part="MG", gripper_state="closed")
        settings = _move_insert_settings("MG")
        agent._task_ctx = {
            "destination_location": "assembly_board-v1",
            "travel_z": 1.2,
            "move_insert_profile": _move_insert_effective(),
            "move_insert_profile_sha256": "a" * 64,
            "move_insert_hard_caps_sha256": settings["hard_caps_sha256"],
            "move_insert_boundary_ready": True,
            "move_insert_boundary_error": "",
            "assembly_board_v1_aruco_generation": 1,
            "assembly_board_v1_aruco": {
                "destination_location": "assembly_board-v1",
                "camera_role": "ur5e",
                "generation": 1,
                "calibration_id": "ur5e-calibration",
            },
            "resolved_cartesian_positions": {
                "descend": deepcopy(MANUAL_DESCEND_POSE),
            },
        }
        return agent
    return _PhysicalUR5eAgent(state="placed")


def _use_real_operator_handoff_builder(
    bridge: SystemBridge,
    agent: _PhysicalUR5eAgent,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    confirmed: bool = True,
    tf_age_sec: float = 0.0,
    part_name: str = "MG",
) -> None:
    bridge.__dict__.pop("_operator_confirmed_mg_held_part_handoff", None)
    source_path = (
        ROOT
        / "cais_spade_llm"
        / "resources"
        / "robot"
        / "taught_functions"
        / "pick_approach"
        / "default__hardware.json"
    )
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    steps = payload["robots"]["ur5e"]["steps"]
    descend = next(step for step in steps if step["step_name"] == "descend")
    descend["confirmed"] = confirmed
    descend["relative_reference"]["name"] = part_name
    taught_root = tmp_path / "taught_functions"
    recording_path = taught_root / "pick_approach" / "default__hardware.json"
    recording_path.parent.mkdir(parents=True)
    recording_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", taught_root)

    current_tf_stamp_sec = time.time() - tf_age_sec
    bridge._robot_function_execution_pose_readiness = (
        lambda _target, _robot, _agent: {
            "success": True,
            "world_tool0_ready": True,
            "tf_stamp_sec": current_tf_stamp_sec,
            "blocked_reason": "",
            "waypoint": {
                "pose": {
                    **deepcopy(MANUAL_DESCEND_POSE),
                    "frame_id": "world",
                    "child_frame_id": "tool0",
                }
            },
        }
    )
    assert agent.calls == []


def _run_to_thread_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _inline(function: Any, *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _inline)


def _move_insert_resource() -> dict[str, Any]:
    resource = {
        "ur5e": {
            "real": {
                "controller": {
                    "parts_tuning": {
                        "move_insert": {
                            "calibration_id": "move_insert-shared-v1",
                            "validated_parts": ["MG"],
                            **{
                                key: value
                                for key, value in _move_insert_effective().items()
                                if key
                                not in {
                                    "part_name",
                                    "calibration_id",
                                    "shared_calibration_id",
                                    "override_calibration_id",
                                    "demonstration_recipe",
                                }
                            },
                            "part_overrides": {},
                            "qualifications": {},
                        },
                    },
                },
            },
        },
        "unrelated": {"preserved": True},
    }
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
        move_insert_profile_sha256,
    )

    profile = resource["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]
    profile_sha256, profile_error = move_insert_profile_sha256(profile, "MG")
    assert not profile_error
    qualification_identity = {
        "confirmed_at": "2026-08-19T12:00:00Z",
        "robot": "ur5e",
        "destination_location": "assembly_board-v1",
        "part_name": "MG",
        "tool_frame": "tool0",
        "profile_sha256": profile_sha256,
        "hard_caps_sha256": SystemBridge._move_insert_hard_caps_sha256(
            _move_insert_test_hard_caps()
        ),
        "place_approach_recording_sha256": "b" * 64,
        "board_calibration_id": "ur5e-calibration",
        "board_geometry_sha256": "c" * 64,
        "recording_id": "recording-test",
        "demonstration_sha256": "e" * 64,
    }
    qualification_identity.pop("confirmed_at")
    trial_ids = ["move-insert-test-1"]
    result_sha256s = ["1" * 64]
    trace_sha256s = ["4" * 64]
    qualification_policy_sha256 = (
        SystemBridge._move_insert_qualification_policy_sha256()
    )
    qualification_evidence_sha256 = (
        SystemBridge._move_insert_qualification_evidence_sha256(
            qualification_identity=qualification_identity,
            trial_ids=trial_ids,
            result_sha256s=result_sha256s,
            trace_sha256s=trace_sha256s,
        )
    )
    profile["qualifications"]["MG"] = {
        "trial_id": trial_ids[-1],
        "confirmed_at": "2026-08-19T12:00:00Z",
        **qualification_identity,
        "board_generation": 1,
        "generation": 1,
        "qualification_policy_version": 3,
        "qualification_policy_sha256": qualification_policy_sha256,
        "required_confirmed_trials": 1,
        "confirmed_trial_count": 1,
        "confirmed_trial_ids": trial_ids,
        "confirmed_trial_result_sha256s": result_sha256s,
        "confirmed_trial_trace_sha256s": trace_sha256s,
        "qualification_evidence_sha256": qualification_evidence_sha256,
    }
    return resource


def _move_insert_hardware_config() -> dict[str, Any]:
    return {
        "ur5e": {
            "rtde": {
                "insert_max_contact_speed_m_s": 0.01,
                "insert_max_contact_force_delta_n": 5.0,
                "insert_max_engagement_progress_m": 0.005,
                "insert_max_insertion_force_n": 15.0,
                "insert_max_spiral_radius_m": 0.003,
                "insert_max_spiral_pitch_m": 0.001,
                "insert_max_spiral_speed_m_s": 0.005,
                "insert_max_spiral_acceleration_m_s2": 0.02,
                "insert_max_axial_force_n": 30.0,
                "insert_max_lateral_force_n": 15.0,
                "insert_max_torque_nm": 3.0,
                "insert_max_tool_flange_torque_nm": 5.0,
                "MG": {
                    "insert_max_insertion_force_n": 15.0,
                    "insert_max_axial_force_n": 30.0,
                    "insert_max_lateral_force_n": 15.0,
                    "insert_max_torque_nm": 3.0,
                    "insert_max_tool_flange_torque_nm": 5.0,
                    "insert_max_relief_retreat_m": 0.0006,
                    "insert_max_contact_search_radius_m": 0.01,
                    "insert_max_disengagement_cycles": 6,
                    "insert_search_peck_retreat_m": 0.003,
                    "insert_search_peck_interval_sec": 0.75,
                },
                "insert_soft_filter_window_sec": 0.05,
                "insert_soft_overload_hold_sec": 0.1,
                "insert_relief_unload_dwell_sec": 0.1,
                "insert_relief_clear_dwell_sec": 0.1,
                "insert_relief_timeout_sec": 1.0,
                "insert_relief_axial_force_ratio": 0.5,
                "insert_relief_reverse_force_ratio": 0.25,
                "insert_relief_clear_hysteresis_ratio": 0.8,
                "insert_relief_resume_ramp_sec": 0.1,
                "insert_relief_search_force_ratio": 0.5,
                "insert_relief_search_speed_ratio": 0.5,
                "insert_relief_backoff_step_m": 0.0001,
                "insert_max_relief_retreat_m": 0.0003,
                "insert_relief_stationary_speed_m_s": 0.0005,
                "insert_relief_stationary_angular_speed_rad_s": 0.01,
                "insert_max_relief_cycles": 3,
                "insert_max_tilt_tolerance_rad": 0.2,
                "insert_max_seated_depth_tolerance_m": 0.002,
                "insert_max_settle_time_sec": 1.0,
                "insert_max_timeout_sec": 30.0,
                "insert_max_travel_m": 0.02,
                "insert_start_position_tolerance_m": 0.003,
                "insert_start_orientation_tolerance_rad": 0.1,
            },
            "hardware_insert_action": (
                "/cais_ur5e_rtde_cartesian_controller/move_insert"
            ),
        },
    }


def _move_insert_live_status(
    bridge: SystemBridge,
    hardware_config: dict[str, Any],
) -> dict[str, Any]:
    rtde_config = deepcopy(hardware_config["ur5e"]["rtde"])
    mg_hard_caps, hard_caps_error = bridge._move_insert_hard_caps("MG")
    assert hard_caps_error == ""
    return {
        "updated_at": time.time(),
        "insert_action_name": (
            "/cais_ur5e_rtde_cartesian_controller/move_insert"
        ),
        "insert_action_ready": True,
        "tcp_force_feedback_ready": True,
        "insert_function_ready": True,
        "insert_MG_hard_caps": mg_hard_caps,
        "insert_MG_hard_caps_sha256": (
            bridge._move_insert_hard_caps_sha256(mg_hard_caps)
        ),
        **rtde_config,
    }


def _move_insert_settings_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SystemBridge, Path, dict[str, Any]]:
    resource_path = tmp_path / "robot_ur5e.json"
    resource_path.write_text(json.dumps(_move_insert_resource(), indent=2), encoding="utf-8")
    hardware_config = _move_insert_hardware_config()
    monkeypatch.setattr(bridge_module, "_UR5E_RESOURCE", resource_path)
    monkeypatch.setattr(
        bridge_module.ros2_processes,
        "load_hardware_arms_config",
        lambda _project_root: deepcopy(hardware_config),
    )
    bridge = object.__new__(SystemBridge)
    bridge.system_running = False
    bridge._starting = False
    bridge._stopping = False
    bridge._move_insert_profile_edit_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_preflight_lock = threading.Lock()
    bridge._ur5e_rtde_reset_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._ur5e_robot_function_agent = None
    bridge._move_insert_trials_dir = tmp_path / "move_insert_trials"
    bridge._ur5e_robot_function_state_uncertain = False
    bridge._ur5e_move_insert_profile_reload_required = False
    bridge._teleop_smooth_session = None
    bridge._teleop_cartesian_modes = {"ur5e": "off", "xarm6": "off"}
    bridge._digital_twin_robot_function_request_error = (
        lambda _target, _robot, _function, _origin, _destination, _part: ({}, "")
    )
    bridge._ur5e_rtde_trajectory_status = lambda: {}
    return bridge, resource_path, hardware_config


def _legacy_quarantine_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    trace_bytes: bytes,
    expected_trace_sha256: str,
) -> tuple[SystemBridge, Path, Path, dict[str, Any]]:
    recording_id = "insertion-demonstration-test-legacy"
    metadata = {
        "part_name": "MG",
        "trial_id": "move-insert-test-legacy-terminal",
        "recording_id": recording_id,
        "demonstration_sha256": expected_trace_sha256,
        "profile_sha256": "a" * 64,
        "calibration_id": "insertion-demonstration-test-legacy",
        "reason": "legacy clipped-to-hard-caps test evidence",
    }
    monkeypatch.setattr(
        bridge_module,
        "_LEGACY_MG_INSERTION_DEMONSTRATION",
        metadata,
    )
    resource = _move_insert_resource()
    profile = resource["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]
    profile["demonstration_recipes"] = {
        "MG": {
            "recording_id": recording_id,
            "demonstration_sha256": expected_trace_sha256,
        }
    }
    resource_path = tmp_path / "robot_ur5e.json"
    resource_path.write_text(json.dumps(resource, indent=2), encoding="utf-8")
    monkeypatch.setattr(bridge_module, "_UR5E_RESOURCE", resource_path)
    demonstrations = tmp_path / "demonstrations"
    source_directory = demonstrations / recording_id
    source_directory.mkdir(parents=True)
    (source_directory / "trace.jsonl").write_bytes(trace_bytes)
    (source_directory / "summary.json").write_text("{}\n", encoding="utf-8")
    (source_directory / "analysis.json").write_text("{}\n", encoding="utf-8")
    bridge = object.__new__(SystemBridge)
    bridge._insertion_demonstrations_dir = demonstrations
    bridge._move_insert_trials_dir = tmp_path / "move_insert_trials"
    bridge._insertion_demonstration_lock = threading.RLock()
    bridge._insertion_demonstration = None
    return bridge, resource_path, source_directory, metadata


def test_legacy_quarantine_verifies_and_preserves_exact_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_bytes = b'{"sample":1}\n{"sample":2}\n'
    trace_file = tmp_path / "trace-for-hash.jsonl"
    trace_file.write_bytes(trace_bytes)
    trace_sha256 = bridge_module.sha256_file(trace_file)
    bridge, resource_path, source_directory, metadata = (
        _legacy_quarantine_bridge(
            tmp_path,
            monkeypatch,
            trace_bytes=trace_bytes,
            expected_trace_sha256=trace_sha256,
        )
    )

    result = bridge.digital_twin_quarantine_legacy_move_insert_recording(
        confirmed=True
    )

    assert result["success"] is True
    assert not source_directory.exists()
    quarantine_directory = Path(result["quarantine_path"])
    quarantined_trace = quarantine_directory / "trace.jsonl"
    assert quarantined_trace.read_bytes() == trace_bytes
    assert bridge_module.sha256_file(quarantined_trace) == trace_sha256
    quarantine = json.loads(
        (quarantine_directory / "quarantine.json").read_text(encoding="utf-8")
    )
    assert quarantine["verified_trace_sha256"] == trace_sha256
    assert quarantine["demonstration_sha256"] == metadata[
        "demonstration_sha256"
    ]
    assert quarantine["trace_rewritten"] is False
    with bridge_module.zipfile.ZipFile(result["download_path"]) as archive:
        assert archive.read("trace.jsonl") == trace_bytes
        assert json.loads(archive.read("quarantine.json"))[
            "verified_trace_sha256"
        ] == trace_sha256
    profile = json.loads(resource_path.read_text(encoding="utf-8"))["ur5e"][
        "real"
    ]["controller"]["parts_tuning"]["move_insert"]
    assert profile["demonstration_recipes"] == {}
    assert profile["validated_parts"] == []
    assert profile["qualifications"] == {}


def test_legacy_quarantine_preserves_exact_terminal_repair_latch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_bytes = b'{"sample":1}\n{"terminal":"failed"}\n'
    trace_file = tmp_path / "trace-for-terminal-repair-hash.jsonl"
    trace_file.write_bytes(trace_bytes)
    trace_sha256 = bridge_module.sha256_file(trace_file)
    bridge, _resource_path, source_directory, metadata = (
        _legacy_quarantine_bridge(
            tmp_path,
            monkeypatch,
            trace_bytes=trace_bytes,
            expected_trace_sha256=trace_sha256,
        )
    )
    trial = bridge._store_move_insert_trial(
        {
            "success": False,
            "ready": False,
            "target": "ur5e only",
            "robot": "ur5e",
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
            "trial_id": metadata["trial_id"],
            "profile_sha256": metadata["profile_sha256"],
            "move_insert_effective": {
                "calibration_id": metadata["calibration_id"],
                "demonstration_recipe": {
                    "recording_id": metadata["recording_id"],
                    "demonstration_sha256": metadata[
                        "demonstration_sha256"
                    ],
                },
            },
            "active": False,
            "completion_motion_active": False,
            "review_required": False,
            "recovery_required": False,
            "normal_repair_required": True,
            "hardware_stack_repair_required": True,
            "hardware_stack_repair_reason": "exact terminal force failure",
            "part_clamped": True,
            "released": False,
            "lifted": False,
            "completion_eligible": False,
            "qualified": False,
            "failure_recorded": True,
            "motion_settled": False,
            "message": (
                "Terminal supervised move_insert requires Repair Hardware Stack."
            ),
        }
    )
    assert trial["hardware_stack_repair_required"] is True
    pending_path = bridge._move_insert_pending_review_path()
    summary_path = (
        bridge._move_insert_trial_directory(str(metadata["trial_id"]))
        / "summary.json"
    )
    pending_before = pending_path.read_bytes()
    summary_before = summary_path.read_bytes()

    result = bridge.digital_twin_quarantine_legacy_move_insert_recording(
        confirmed=True
    )

    assert result["success"] is True
    assert result["terminal_repair_preserved"] is True
    assert "Repair Hardware Stack latch was not changed" in result["message"]
    assert not source_directory.exists()
    assert pending_path.read_bytes() == pending_before
    assert summary_path.read_bytes() == summary_before
    pending_trial = bridge._move_insert_pending_review()
    assert pending_trial is not None
    assert pending_trial["trial_id"] == metadata["trial_id"]
    assert pending_trial["hardware_stack_repair_required"] is True
    assert pending_trial["part_clamped"] is True
    assert pending_trial["qualified"] is False


def test_legacy_quarantine_rejects_nonexact_terminal_repair_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_bytes = b'{"sample":1}\n'
    trace_file = tmp_path / "trace-for-nonexact-terminal-hash.jsonl"
    trace_file.write_bytes(trace_bytes)
    trace_sha256 = bridge_module.sha256_file(trace_file)
    bridge, resource_path, source_directory, metadata = (
        _legacy_quarantine_bridge(
            tmp_path,
            monkeypatch,
            trace_bytes=trace_bytes,
            expected_trace_sha256=trace_sha256,
        )
    )
    pending_trial = {
        "target": "ur5e only",
        "robot": "ur5e",
        "destination_location": "assembly_board-v1",
        "part_name": "MG",
        "trial_id": metadata["trial_id"],
        "profile_sha256": "b" * 64,
        "move_insert_effective": {
            "calibration_id": metadata["calibration_id"],
            "demonstration_recipe": {
                "recording_id": metadata["recording_id"],
                "demonstration_sha256": metadata["demonstration_sha256"],
            },
        },
        "active": False,
        "completion_motion_active": False,
        "review_required": False,
        "recovery_required": False,
        "hardware_stack_repair_required": True,
        "part_clamped": True,
        "released": False,
        "lifted": False,
        "completion_eligible": False,
        "qualified": False,
        "failure_recorded": True,
    }
    bridge._move_insert_pending_review = (  # type: ignore[method-assign]
        lambda: deepcopy(pending_trial)
    )
    resource_before = resource_path.read_bytes()
    trace_before = (source_directory / "trace.jsonl").read_bytes()

    result = bridge.digital_twin_quarantine_legacy_move_insert_recording(
        confirmed=True
    )

    assert result["success"] is False
    assert "supervised move_insert result still owns durable custody" in result[
        "message"
    ]
    assert resource_path.read_bytes() == resource_before
    assert source_directory.is_dir()
    assert (source_directory / "trace.jsonl").read_bytes() == trace_before


def test_legacy_quarantine_trace_hash_mismatch_changes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, resource_path, source_directory, _metadata = (
        _legacy_quarantine_bridge(
            tmp_path,
            monkeypatch,
            trace_bytes=b"mismatched trace bytes\n",
            expected_trace_sha256="a" * 64,
        )
    )
    resource_before = resource_path.read_bytes()
    trace_before = (source_directory / "trace.jsonl").read_bytes()

    result = bridge.digital_twin_quarantine_legacy_move_insert_recording(
        confirmed=True
    )

    assert result["success"] is False
    assert "trace SHA-256 does not match" in result["message"]
    assert resource_path.read_bytes() == resource_before
    assert source_directory.is_dir()
    assert (source_directory / "trace.jsonl").read_bytes() == trace_before
    assert not (
        source_directory.parent / ".quarantine" / source_directory.name
    ).exists()


def test_preinsert_hard_caps_require_fresh_live_config_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _resource_path, hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    rtde_config = dict(hardware_config["ur5e"]["rtde"])
    bridge._ur5e_rtde_trajectory_status = lambda: {
        **rtde_config,
        "updated_at": time.time(),
    }

    readiness, error = bridge._move_insert_preinsert_hard_caps_readiness()

    assert error == ""
    assert readiness["move_insert_hard_caps"] == (
        _move_insert_preinsert_hard_caps()
    )
    assert readiness["move_insert_live_hard_caps"] == (
        _move_insert_preinsert_hard_caps()
    )

    bridge._ur5e_rtde_trajectory_status = lambda: {
        **rtde_config,
        "updated_at": time.time() - 3.1,
    }
    stale_readiness, stale_error = (
        bridge._move_insert_preinsert_hard_caps_readiness()
    )
    assert "not fresh" in stale_error
    assert stale_readiness["move_insert_live_hard_caps"] == {}


def test_preinsert_hard_caps_bind_complete_exact_mg_live_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _resource_path, hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    bridge._ur5e_rtde_trajectory_status = lambda: _move_insert_live_status(
        bridge,
        hardware_config,
    )
    expected_caps, caps_error = bridge._move_insert_hard_caps("MG")
    assert caps_error == ""

    readiness, error = bridge._move_insert_preinsert_hard_caps_readiness("MG")

    assert error == ""
    assert readiness["move_insert_hard_caps"] == expected_caps
    assert readiness["move_insert_live_hard_caps"] == expected_caps
    assert readiness["move_insert_hard_caps_sha256"] == (
        bridge._move_insert_hard_caps_sha256(expected_caps)
    )
    assert "insert_max_contact_search_radius_m" in readiness[
        "move_insert_hard_caps"
    ]


def test_move_insert_settings_reports_complete_shared_profile_and_certified_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _resource_path, _hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )

    result = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )

    assert result["success"] is True
    assert result["validated"] is True
    assert result["editable"] is True
    assert result["execution_mode"] == "physical"
    assert result["profile_state"] == "ready"
    assert len(result["profile_sha256"]) == 64
    assert result["override"] == {}
    assert result["shared"]["calibration_id"] == "move_insert-shared-v1"
    assert result["shared"]["contact_speed_m_s"] == pytest.approx(0.002)
    assert result["shared"]["spiral_pitch_m"] == pytest.approx(0.0005)
    assert {
        key: value
        for key, value in result["effective"].items()
        if key not in {"qualification", "demonstration_recipe"}
    } == {
        key: value
        for key, value in _move_insert_effective().items()
        if key != "demonstration_recipe"
    }
    assert result["effective"]["qualification"]["part_name"] == "MG"
    assert result["effective"]["qualification"]["profile_sha256"] == result[
        "profile_sha256"
    ]
    assert result["effective"]["shared_calibration_id"] == "move_insert-shared-v1"
    assert result["effective"]["override_calibration_id"] is None
    assert result["derived_timeout_sec"] == pytest.approx(30.0)
    assert result["adjustable_fields"] == {
        "insertion_force_n": {
            "minimum": 0.0,
            "exclusive_minimum": True,
            "maximum": 15.0,
            "unit": "N",
        },
        "spiral_radius_m": {
            "minimum": 0.0,
            "exclusive_minimum": False,
            "maximum": 0.003,
            "unit": "m",
        },
    }

    trial = bridge._digital_twin_move_insert_trial_settings("MG")
    assert trial["validated"] is True
    assert "qualification" not in trial["effective"]


def test_xarm6_move_insert_resource_seam_never_reads_or_patches_ur5e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ur5e_path = tmp_path / "robot_ur5e.json"
    xarm6_path = tmp_path / "robot_xarm6.json"
    ur5e_path.write_text(
        json.dumps(_move_insert_resource(), indent=2),
        encoding="utf-8",
    )
    xarm6_resource = {
        "xarm6": {
            "real": {
                "controller": {
                    "parts_tuning": {
                        "xarm6_only_marker": "robot_xarm6.json",
                    }
                }
            }
        }
    }
    xarm6_path.write_text(
        json.dumps(xarm6_resource, indent=2),
        encoding="utf-8",
    )
    monkeypatch.setattr(bridge_module, "_UR5E_RESOURCE", ur5e_path)
    monkeypatch.setattr(bridge_module, "_XARM6_RESOURCE", xarm6_path)
    bridge = object.__new__(SystemBridge)
    ur5e_before = ur5e_path.read_bytes()
    xarm6_before = xarm6_path.read_bytes()

    resource, digest, read_error = bridge._move_insert_resource_snapshot(
        "xarm6"
    )
    profile, profile_error = bridge._move_insert_profile_from_resource(
        resource,
        "xarm6",
    )
    qualification, validated = bridge._move_insert_qualification_from_resource(
        resource,
        "MG",
        "xarm6",
    )
    patch_result = bridge._patch_move_insert_qualification(
        part_name="MG",
        expected_profile_sha256="a" * 64,
        qualification=None,
        robot="xarm6",
    )

    assert read_error == ""
    assert len(digest) == 64
    assert resource == xarm6_resource
    assert profile == {}
    assert profile_error == (
        "robot_xarm6.json is missing "
        "xarm6.real.controller.parts_tuning.move_insert"
    )
    assert qualification == {}
    assert validated is False
    assert patch_result["success"] is False
    assert patch_result["message"] == profile_error
    assert ur5e_path.read_bytes() == ur5e_before
    assert xarm6_path.read_bytes() == xarm6_before


def test_public_xarm6_move_insert_gate_runs_before_resource_lookup() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._digital_twin_robot_function_request_error = (
        lambda *_args: ({}, "")
    )
    bridge._move_insert_resource_snapshot = lambda *_args: pytest.fail(
        "xarm6 must fail before any resource lookup"
    )

    result = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "xarm6",
        destination_location="assembly_board-v1",
        part_name="MG",
    )

    assert result["success"] is False
    assert result["robot"] == "xarm6"
    assert result["message"] == (
        "move_insert settings are available only for the exact robot ur5e."
    )


@pytest.mark.parametrize(
    ("field", "cap_name"),
    [
        ("max_axial_force_n", "insert_max_axial_force_n"),
        ("max_lateral_force_n", "insert_max_lateral_force_n"),
        ("max_torque_nm", "insert_max_torque_nm"),
    ],
)
def test_move_insert_soft_limits_must_remain_strictly_below_hard_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    cap_name: str,
) -> None:
    bridge, resource_path, hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
        move_insert_profile_sha256,
    )

    def _write_value(value: float) -> None:
        resource = json.loads(resource_path.read_text(encoding="utf-8"))
        profile = resource["ur5e"]["real"]["controller"]["parts_tuning"][
            "move_insert"
        ]
        profile[field] = value
        profile_sha256, profile_error = move_insert_profile_sha256(profile, "MG")
        assert profile_error == ""
        qualification = profile["qualifications"]["MG"]
        qualification["profile_sha256"] = profile_sha256
        qualification["qualification_evidence_sha256"] = (
            SystemBridge._move_insert_qualification_evidence_sha256(
                qualification_identity={
                    field_name: qualification[field_name]
                    for field_name in (
                        bridge_module._MOVE_INSERT_QUALIFICATION_IDENTITY_FIELDS
                    )
                },
                trial_ids=qualification["confirmed_trial_ids"],
                result_sha256s=qualification[
                    "confirmed_trial_result_sha256s"
                ],
                trace_sha256s=qualification[
                    "confirmed_trial_trace_sha256s"
                ],
            )
        )
        resource_path.write_text(json.dumps(resource, indent=2), encoding="utf-8")

    hard_cap = float(hardware_config["ur5e"]["rtde"][cap_name])
    _write_value(hard_cap)
    equal = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    assert equal["validated"] is False
    assert f"{field}={hard_cap:.9g} must remain strictly below" in equal[
        "message"
    ]

    _write_value(hard_cap - 1e-6)
    below = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    assert below["validated"] is True


def test_move_insert_save_requires_confirmation_and_replaces_complete_sparse_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, resource_path, _hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    initial = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    before = resource_path.read_bytes()

    unconfirmed = bridge.digital_twin_save_move_insert_override(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        override={"insertion_force_n": 9.0, "spiral_radius_m": 0.002},
        expected_profile_sha256=initial["profile_sha256"],
    )
    assert unconfirmed["success"] is False
    assert "confirmation" in unconfirmed["message"]
    assert resource_path.read_bytes() == before

    saved = bridge.digital_twin_save_move_insert_override(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        override={"insertion_force_n": 9.0, "spiral_radius_m": 0.002},
        expected_profile_sha256=initial["profile_sha256"],
        confirmed=True,
    )
    assert saved["success"] is True
    assert saved["changed"] is True
    assert saved["qualification_invalidated"] is True
    assert saved["override"] == {
        "insertion_force_n": 9.0,
        "spiral_radius_m": 0.002,
    }
    first_calibration_id = saved["calibration_id"]
    assert first_calibration_id.startswith("move_insert-MG-g1-")
    assert saved["generation"] == 1

    replaced = bridge.digital_twin_save_move_insert_override(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        override={"insertion_force_n": 10.0},
        expected_profile_sha256=saved["profile_sha256"],
        confirmed=True,
    )
    assert replaced["success"] is True
    assert replaced["override"] == {"insertion_force_n": 10.0}
    assert replaced["effective"]["spiral_radius_m"] == pytest.approx(0.001)
    assert replaced["generation"] == 2
    assert replaced["calibration_id"] != first_calibration_id
    assert replaced["effective"]["shared_calibration_id"] == "move_insert-shared-v1"
    assert replaced["effective"]["override_calibration_id"] == replaced["calibration_id"]
    assert replaced["effective"]["calibration_id"] == replaced["calibration_id"]

    persisted = json.loads(resource_path.read_text(encoding="utf-8"))
    selected = persisted["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]["part_overrides"]["MG"]
    persisted_profile = persisted["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]
    assert set(selected) == {
        "calibration_id",
        "generation",
        "updated_at",
        "profile_sha256",
        "insertion_force_n",
    }
    assert selected["profile_sha256"] == replaced["profile_sha256"]
    assert "MG" not in persisted_profile["validated_parts"]
    assert "MG" not in persisted_profile["qualifications"]
    assert persisted["unrelated"] == {"preserved": True}
    assert resource_path.read_bytes().endswith(b"\n")
    assert list(tmp_path.glob(".robot_ur5e.json.tmp.*")) == []


def test_move_insert_selected_hash_ignores_and_preserves_unrelated_part_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, resource_path, _hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    profile = resource["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]
    profile["validated_parts"] = ["MG", "SG"]
    resource_path.write_text(json.dumps(resource, indent=2) + "\n", encoding="utf-8")
    initial = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    sg_override = {
        "calibration_id": "move_insert-SG-g1-test",
        "generation": 1,
        "updated_at": "2026-08-18T12:00:00+00:00",
        "insertion_force_n": 7.0,
    }
    profile["part_overrides"]["SG"] = sg_override
    _resolver, profile_hash, _timeout = bridge._move_insert_profile_helpers()
    sg_hash, error = profile_hash(profile, "SG")
    assert error == ""
    sg_override["profile_sha256"] = sg_hash
    resource_path.write_text(json.dumps(resource, indent=2) + "\n", encoding="utf-8")

    refreshed = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    assert refreshed["profile_sha256"] == initial["profile_sha256"]
    mg_hash_after_unrelated_override = refreshed["profile_sha256"]
    sg_before = deepcopy(sg_override)

    saved = bridge.digital_twin_save_move_insert_override(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        override={"spiral_radius_m": 0.002},
        expected_profile_sha256=mg_hash_after_unrelated_override,
        confirmed=True,
    )

    assert saved["success"] is True
    persisted = json.loads(resource_path.read_text(encoding="utf-8"))
    overrides = persisted["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]["part_overrides"]
    assert overrides["SG"] == sg_before


@pytest.mark.parametrize(
    ("override", "message_fragment"),
    [
        ({}, "Clear Override"),
        ({"timeout_sec": 5.0}, "unsupported fields"),
        ({"insertion_force_n": True}, "finite number"),
        ({"insertion_force_n": float("nan")}, "finite number"),
        ({"insertion_force_n": 0.0}, "greater than 0"),
        ({"insertion_force_n": 16.0}, "hard cap"),
        ({"spiral_radius_m": -0.001}, "at least 0"),
        ({"spiral_radius_m": 0.004}, "hard cap"),
    ],
)
def test_move_insert_save_rejects_non_sparse_or_uncertified_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, Any],
    message_fragment: str,
) -> None:
    bridge, resource_path, _hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    settings = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    before = resource_path.read_bytes()

    result = bridge.digital_twin_save_move_insert_override(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        override=override,
        expected_profile_sha256=settings["profile_sha256"],
        confirmed=True,
    )

    assert result["success"] is False
    assert message_fragment in result["message"]
    assert resource_path.read_bytes() == before


def test_move_insert_save_rejects_selected_hash_staleness_and_whole_file_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, resource_path, _hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    settings = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    before = resource_path.read_bytes()

    stale = bridge.digital_twin_save_move_insert_override(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        override={"insertion_force_n": 9.0},
        expected_profile_sha256="0" * 64,
        confirmed=True,
    )
    assert stale["success"] is False
    assert "changed after this form was loaded" in stale["message"]
    assert resource_path.read_bytes() == before

    monkeypatch.setattr(bridge_module, "sha256_file", lambda _path: "f" * 64)
    raced = bridge.digital_twin_save_move_insert_override(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        override={"insertion_force_n": 9.0},
        expected_profile_sha256=settings["profile_sha256"],
        confirmed=True,
    )
    assert raced["success"] is False
    assert "changed during" in raced["message"]
    assert resource_path.read_bytes() == before


def test_move_insert_atomic_replace_failure_preserves_source_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, resource_path, _hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    settings = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    before = resource_path.read_bytes()

    def _replace_failure(_source: Path, _destination: Path) -> None:
        raise PermissionError("forced atomic replace failure")

    monkeypatch.setattr(bridge_module.os, "replace", _replace_failure)
    result = bridge.digital_twin_save_move_insert_override(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        override={"insertion_force_n": 9.0},
        expected_profile_sha256=settings["profile_sha256"],
        confirmed=True,
    )

    assert result["success"] is False
    assert "forced atomic replace failure" in result["message"]
    assert resource_path.read_bytes() == before
    assert list(tmp_path.glob(".robot_ur5e.json.tmp.*")) == []


def test_move_insert_clear_deletes_only_selected_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, resource_path, _hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    initial = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    saved = bridge.digital_twin_save_move_insert_override(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        override={"spiral_radius_m": 0.0},
        expected_profile_sha256=initial["profile_sha256"],
        confirmed=True,
    )
    before_clear = resource_path.read_bytes()

    unconfirmed = bridge.digital_twin_clear_move_insert_override(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        expected_profile_sha256=saved["profile_sha256"],
    )
    assert unconfirmed["success"] is False
    assert "confirmation" in unconfirmed["message"]
    assert resource_path.read_bytes() == before_clear

    cleared = bridge.digital_twin_clear_move_insert_override(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        expected_profile_sha256=saved["profile_sha256"],
        confirmed=True,
    )

    assert cleared["success"] is True
    assert cleared["changed"] is True
    assert cleared["override"] == {}
    persisted = json.loads(resource_path.read_text(encoding="utf-8"))
    overrides = persisted["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]["part_overrides"]
    assert overrides == {}
    assert persisted["unrelated"] == {"preserved": True}


def test_move_insert_settings_disable_editing_when_one_certified_cap_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _resource_path, hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    del hardware_config["ur5e"]["rtde"]["insert_max_spiral_radius_m"]

    settings = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )

    assert settings["success"] is True
    assert settings["validated"] is False
    assert settings["editable"] is False
    assert settings["adjustable_fields"]["spiral_radius_m"]["maximum"] is None
    assert "insert_max_spiral_radius_m" in settings["message"]


def test_move_insert_clear_removes_orphan_override_from_incomplete_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, resource_path, _hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    initial = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    saved = bridge.digital_twin_save_move_insert_override(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        override={"insertion_force_n": 9.0},
        expected_profile_sha256=initial["profile_sha256"],
        confirmed=True,
    )
    assert saved["success"] is True

    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    profile = resource["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]
    profile["validated_parts"] = []
    profile.pop("contact_speed_m_s")
    resource_path.write_text(json.dumps(resource, indent=2) + "\n", encoding="utf-8")
    orphan = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    assert orphan["validated"] is False
    assert orphan["override"] == {"insertion_force_n": 9.0}

    cleared = bridge.digital_twin_clear_move_insert_override(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        expected_profile_sha256=orphan["profile_sha256"],
        confirmed=True,
    )

    assert cleared["success"] is True
    assert cleared["changed"] is True
    persisted = json.loads(resource_path.read_text(encoding="utf-8"))
    persisted_profile = persisted["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]
    assert persisted_profile["part_overrides"] == {}
    assert persisted_profile["validated_parts"] == []
    assert "contact_speed_m_s" not in persisted_profile


@pytest.mark.parametrize("gate", ["system_running", "_starting", "_stopping", "lock"])
def test_move_insert_save_obeys_lifecycle_and_motion_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate: str,
) -> None:
    bridge, resource_path, _hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    settings = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    before = resource_path.read_bytes()
    if gate == "lock":
        bridge._ur5e_robot_function_execution_active = "Assembly"
        bridge._ur5e_robot_function_execution_lock.acquire()
    else:
        setattr(bridge, gate, True)
    try:
        result = bridge.digital_twin_save_move_insert_override(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            override={"insertion_force_n": 9.0},
            expected_profile_sha256=settings["profile_sha256"],
            confirmed=True,
        )
    finally:
        if gate == "lock":
            bridge._ur5e_robot_function_execution_lock.release()

    assert result["success"] is False
    assert "CAIS system" in result["message"] or "motion" in result["message"]
    assert resource_path.read_bytes() == before


@pytest.mark.parametrize(
    ("gate", "message_fragment"),
    [
        ("preflight", "preflight is active"),
        ("reset", "Reset UR5e RTDE"),
        ("teleop_hold", "Smooth Hold"),
        ("teleop_mode", "Cartesian mode"),
        ("rtde_motion", "is executing"),
        ("cached_agent", "Complete move_home"),
        ("uncertain", "state is uncertain"),
    ],
)
def test_move_insert_save_rejects_every_direct_control_busy_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate: str,
    message_fragment: str,
) -> None:
    bridge, resource_path, _hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    settings = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    before = resource_path.read_bytes()
    held_lock: threading.Lock | None = None
    if gate == "preflight":
        held_lock = bridge._ur5e_robot_function_preflight_lock
        held_lock.acquire()
    elif gate == "reset":
        held_lock = bridge._ur5e_rtde_reset_lock
        held_lock.acquire()
    elif gate == "teleop_hold":
        bridge._teleop_smooth_session = {"robot": "ur5e", "axis": "z"}
    elif gate == "teleop_mode":
        bridge._teleop_cartesian_modes["ur5e"] = "smooth"
    elif gate == "rtde_motion":
        bridge._ur5e_rtde_trajectory_status = lambda: {
            "state": "executing",
            "motion_kind": "insert",
        }
    elif gate == "cached_agent":
        bridge._ur5e_robot_function_agent = _PhysicalUR5eAgent(
            state="picked",
            held_part="MG",
            gripper_state="closed",
        )
    else:
        bridge._ur5e_robot_function_state_uncertain = True
    try:
        result = bridge.digital_twin_save_move_insert_override(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            override={"insertion_force_n": 9.0},
            expected_profile_sha256=settings["profile_sha256"],
            confirmed=True,
        )
    finally:
        if held_lock is not None:
            held_lock.release()

    assert result["success"] is False
    assert message_fragment in result["message"]
    assert resource_path.read_bytes() == before


def test_move_insert_save_invalidates_idle_cached_agent_for_next_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _resource_path, _hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    cached_agent = _PhysicalUR5eAgent()
    bridge._ur5e_robot_function_agent = cached_agent
    bridge._ur5e_robot_function_agent_lifecycle_lock = asyncio.Lock()
    settings = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    saved = bridge.digital_twin_save_move_insert_override(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        override={"insertion_force_n": 9.0},
        expected_profile_sha256=settings["profile_sha256"],
        confirmed=True,
    )
    assert saved["success"] is True
    assert bridge._ur5e_move_insert_profile_reload_required is True

    fresh_agent = object()
    disposed: list[Any] = []

    async def _target_ready(_target: str, _cfg: dict[str, Any]) -> str:
        return ""

    async def _dispose() -> None:
        disposed.append(bridge._ur5e_robot_function_agent)
        bridge._ur5e_robot_function_agent = None

    async def _ensure(_target: str, _robot: str) -> tuple[Any, str]:
        assert bridge._ur5e_robot_function_agent is None
        bridge._ur5e_robot_function_agent = fresh_agent
        return fresh_agent, ""

    async def _prepared(*_args: Any) -> tuple[Any, dict[str, Any], dict[str, Any], str]:
        return fresh_agent, {}, {}, ""

    bridge._wait_for_digital_twin_robot_function_target_error = _target_ready
    bridge._dispose_ur5e_robot_function_agent = _dispose
    bridge._ensure_ur5e_robot_function_agent_locked = _ensure
    bridge._digital_twin_robot_function_execution_preflight_prepared_async = _prepared

    agent, _kwargs, _readiness, error = asyncio.run(
        bridge._digital_twin_robot_function_execution_preflight_async(
            "dual robots",
            "ur5e",
            "pick_approach",
            "prusa-mk4-2",
            "",
            "MG",
        )
    )

    assert error == ""
    assert disposed == [cached_agent]
    assert agent is fresh_agent
    assert bridge._ur5e_move_insert_profile_reload_required is False


def test_move_insert_live_readiness_requires_matching_caps_and_force_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _resource_path, hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    settings = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    live_status = _move_insert_live_status(bridge, hardware_config)
    bridge._ur5e_rtde_trajectory_status = lambda: deepcopy(live_status)

    readiness, error = bridge._digital_twin_move_insert_live_readiness(settings)
    assert error == ""
    assert readiness["insert_function_ready"] is True
    assert readiness["move_insert_live_hard_caps"][
        "insert_max_insertion_force_n"
    ] == pytest.approx(15.0)

    live_status["tcp_force_feedback_ready"] = False
    live_status["insert_readiness_message"] = "TCP force unavailable"
    _readiness, error = bridge._digital_twin_move_insert_live_readiness(settings)
    assert error == "TCP force unavailable"

    live_status["tcp_force_feedback_ready"] = True
    live_status["insert_MG_hard_caps"][
        "insert_max_insertion_force_n"
    ] = 14.0
    _readiness, error = bridge._digital_twin_move_insert_live_readiness(settings)
    assert "does not match Hardware Stack YAML" in error


def test_exact_part_hard_caps_overlay_only_the_selected_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _resource_path, hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    sg_overlay = {
        **deepcopy(hardware_config["ur5e"]["rtde"]["MG"]),
        "insert_max_insertion_force_n": 16.0,
        "insert_max_axial_force_n": 31.0,
        "insert_max_lateral_force_n": 16.0,
        "insert_max_torque_nm": 3.1,
        "insert_max_tool_flange_torque_nm": 5.1,
        "insert_max_relief_retreat_m": 0.0007,
        "insert_max_contact_search_radius_m": 0.009,
        "insert_max_disengagement_cycles": 4,
        "insert_search_peck_retreat_m": 0.0025,
        "insert_search_peck_interval_sec": 0.6,
    }
    hardware_config["ur5e"]["rtde"]["SG"] = sg_overlay

    sg_caps, sg_error = bridge._move_insert_hard_caps("SG")
    mg_caps, mg_error = bridge._move_insert_hard_caps("MG")
    lg_caps, lg_error = bridge._move_insert_hard_caps("LG")
    sg_policy, sg_policy_error = bridge._move_insert_learning_policy(
        sg_caps,
        "SG",
    )
    lg_policy, lg_policy_error = bridge._move_insert_learning_policy(
        lg_caps,
        "LG",
    )

    assert sg_error == mg_error == lg_error == ""
    assert sg_policy_error == lg_policy_error == ""
    assert sg_caps["insert_max_axial_force_n"] == pytest.approx(31.0)
    assert mg_caps["insert_max_axial_force_n"] == pytest.approx(30.0)
    assert lg_caps["insert_max_axial_force_n"] == pytest.approx(30.0)
    assert sg_policy["insert_max_disengagement_cycles"] == pytest.approx(4.0)
    assert "insert_max_disengagement_cycles" not in lg_policy
    assert "tactile_center_policy" in sg_policy
    assert "tactile_center_policy" not in lg_policy


def test_incomplete_exact_part_hard_caps_require_that_parts_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _resource_path, hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    hardware_config["ur5e"]["rtde"]["SG"] = deepcopy(
        hardware_config["ur5e"]["rtde"]["MG"]
    )
    del hardware_config["ur5e"]["rtde"]["SG"][
        "insert_search_peck_interval_sec"
    ]

    sg_caps, sg_error = bridge._move_insert_hard_caps("SG")
    mg_caps, mg_error = bridge._move_insert_hard_caps("MG")

    assert "for SG" in sg_error
    assert "insert_search_peck_interval_sec" in sg_error
    assert sg_caps["insert_search_peck_interval_sec"] is None
    assert mg_error == ""
    assert mg_caps["insert_search_peck_interval_sec"] == pytest.approx(0.75)


def test_live_readiness_uses_exact_non_mg_hard_cap_status_without_shared_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _resource_path, hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    hardware_config["ur5e"]["rtde"]["SG"] = {
        "insert_max_axial_force_n": 31.0,
    }
    sg_caps, caps_error = bridge._move_insert_hard_caps("SG")
    sg_policy, policy_error = bridge._move_insert_learning_policy(
        sg_caps,
        "SG",
    )
    assert caps_error == ""
    assert policy_error == ""
    assert sg_caps["insert_max_axial_force_n"] == pytest.approx(31.0)
    assert "insert_max_disengagement_cycles" not in sg_caps
    assert "tactile_center_policy" not in sg_policy
    settings = {
        "validated": True,
        "effective": {"part_name": "SG"},
        "hard_caps": sg_caps,
        "hard_caps_sha256": bridge._move_insert_hard_caps_sha256(sg_caps),
        "derived_timeout_sec": 10.0,
    }
    status = {
        "updated_at": time.time(),
        "insert_action_name": (
            "/cais_ur5e_rtde_cartesian_controller/move_insert"
        ),
        "insert_action_ready": True,
        "tcp_force_feedback_ready": True,
        "insert_function_ready": True,
        "insert_exact_part_hard_caps": {"SG": deepcopy(sg_caps)},
        "insert_exact_part_hard_caps_error": {"SG": ""},
        "insert_exact_part_hard_caps_sha256": {
            "SG": bridge._move_insert_hard_caps_sha256(sg_caps)
        },
        **deepcopy(hardware_config["ur5e"]["rtde"]),
    }
    bridge._ur5e_rtde_trajectory_status = lambda: deepcopy(status)

    readiness, error = bridge._digital_twin_move_insert_live_readiness(settings)

    assert error == ""
    assert readiness["move_insert_live_hard_caps"] == sg_caps
    del status["insert_exact_part_hard_caps"]["SG"]
    _readiness, missing_error = bridge._digital_twin_move_insert_live_readiness(
        settings
    )
    assert "shared top-level caps cannot verify an exact override" in missing_error


def test_move_insert_live_readiness_requires_prepared_client_without_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _resource_path, hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    settings = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    live_status = _move_insert_live_status(bridge, hardware_config)
    bridge._ur5e_rtde_trajectory_status = lambda: deepcopy(live_status)
    controller = SimpleNamespace(
        _ur5e_hardware_insert_action=(
            "/cais_ur5e_rtde_cartesian_controller/move_insert"
        ),
        _ur5e_hardware_insert_client=None,
    )
    resource_agent = SimpleNamespace(_controller=controller)

    readiness, error = bridge._digital_twin_move_insert_live_readiness(
        settings,
        resource_agent,
    )
    assert readiness["move_insert_client_ready"] is False
    assert "move_insert client is unavailable" in error

    wait_timeouts: list[float] = []
    controller._ur5e_hardware_insert_client = SimpleNamespace(
        wait_for_server=lambda *, timeout_sec: (
            wait_timeouts.append(timeout_sec) or True
        )
    )
    readiness, error = bridge._digital_twin_move_insert_live_readiness(
        settings,
        resource_agent,
    )
    assert error == ""
    assert readiness["move_insert_client_ready"] is True
    assert wait_timeouts == [0.0]


def test_move_insert_status_fields_are_forwarded_without_renaming() -> None:
    rtde_config = deepcopy(_move_insert_hardware_config()["ur5e"]["rtde"])
    mg_hard_caps = rtde_config.pop("MG")
    insert_status = {
        "insert_action_name": "/cais_ur5e_rtde_cartesian_controller/move_insert",
        "insert_action_ready": True,
        "insert_supported_part_names": ["SG", "MG", "LG", "SCP", "MCP", "LCP"],
        "tcp_force_feedback_ready": True,
        "insert_function_ready": True,
        "insert_readiness_message": "UR5e insertion interface ready",
        "motion_kind": "insert",
        "insert_phase": "searching",
        "insert_insertion_depth_m": 0.004,
        "insert_depth_error_m": 0.001,
        "insert_lateral_offset_m": 0.0002,
        "insert_search_radius_m": 0.0006,
        "insert_axial_force_n": 7.5,
        "insert_lateral_force_n": 1.2,
        "insert_torque_nm": 0.3,
        "insert_contact_detected": True,
        "actual_tcp_force": [0.1, 0.2, 7.5, 0.0, 0.0, 0.3],
        "peak_axial_force_n": 8.0,
        "peak_lateral_force_n": 1.5,
        "peak_torque_nm": 0.4,
        "cartesian_world_base_ready": True,
        "cartesian_world_base_message": (
            "live world -> base matches protected "
            "ur5e.rtde.cartesian_world_base"
        ),
        "cartesian_world_base_expected": {
            "x": 0.0,
            "y": 0.5,
            "z": 1.021,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        "cartesian_world_base_observed": {
            "x": 0.0,
            "y": 0.5,
            "z": 1.021,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        "cartesian_world_base_position_error_m": 0.0,
        "cartesian_world_base_orientation_error_rad": 0.0,
        "insert_MG_hard_caps": mg_hard_caps,
        "insert_MG_hard_caps_error": "",
        "insert_MG_hard_caps_sha256": (
            SystemBridge._move_insert_hard_caps_sha256(
                {
                    **rtde_config,
                    **mg_hard_caps,
                }
            )
        ),
        **rtde_config,
    }

    status = SystemBridge._attach_ur5e_rtde_trajectory_status(
        {"driver": "running"},
        insert_status,
    )

    for key, value in insert_status.items():
        assert status[f"rtde_trajectory_{key}"] == value
    assert status["insert_action_ready"] is True
    assert status["tcp_force_feedback_ready"] is True
    assert status["insert_function_ready"] is True
    assert status["insert_readiness_message"] == "UR5e insertion interface ready"
    assert status["insert_phase"] == "searching"
    assert status["cartesian_world_base_ready"] is True
    assert status["cartesian_world_base_expected"]["y"] == pytest.approx(0.5)
    assert status["cartesian_world_base_observed"]["y"] == pytest.approx(0.5)
    assert status["cartesian_world_base_position_error_m"] == pytest.approx(0.0)
    assert status["cartesian_world_base_orientation_error_rad"] == pytest.approx(
        0.0
    )


@pytest.mark.parametrize(
    ("function_name", "arguments", "expected_kwargs"),
    [
        (
            "pick_approach",
            {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"},
            {
                "origin_resource_location": "prusa-mk4-2",
                "part_name": "MG",
                "product_geometry": _mg_product_geometry(),
            },
        ),
        (
            "pick_grasp",
            {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"},
            {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"},
        ),
        (
            "place_approach",
            {"destination_location": "assembly_board-v1", "part_name": "MG"},
            {
                "destination_location": "assembly_board-v1",
                "part_name": "MG",
                "product_geometry": _mg_place_product_geometry(),
            },
        ),
        (
            "place_insert",
            {"destination_location": "assembly_board-v1", "part_name": "MG"},
            {"destination_location": "assembly_board-v1", "part_name": "MG"},
        ),
        ("move_home", {}, {}),
    ],
)
def test_generic_execution_dispatches_each_exact_generated_method(
    function_name: str,
    arguments: dict[str, str],
    expected_kwargs: dict[str, Any],
) -> None:
    agent = _agent_for(function_name)
    bridge = _ready_bridge(agent)
    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            function_name,
            **arguments,
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert agent.calls == [(function_name, expected_kwargs)]


def test_robot_agent_manual_function_execution_uses_private_identity_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(RobotAgent)
    agent.agent_name = "ur5e"
    agent._robot_motion_lock = threading.Lock()
    observed: dict[str, Any] = {}

    async def _execute(
        actual_agent: Any,
        task_name: str,
        authority: object | None = None,
        /,
        **kwargs: Any,
    ) -> dict[str, Any]:
        observed.update(
            {
                "agent": actual_agent,
                "task_name": task_name,
                "authority": authority,
                "kwargs": kwargs,
            }
        )
        return {"status": "completed"}

    monkeypatch.setattr(robot_agent_module, "execute_robot_task", _execute)
    pre_execute_lock_states: list[bool] = []

    def _pre_execute() -> str:
        pre_execute_lock_states.append(agent._robot_motion_lock.locked())
        return ""

    result = asyncio.run(
        agent._execute_registered_robot_task_for_manual_function_execution(
            "pick_approach",
            _pre_execute,
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result == {"status": "completed"}
    assert observed == {
        "agent": agent,
        "task_name": "pick_approach",
        "authority": robot_task_runtime._MANUAL_FUNCTION_EXECUTION_AUTHORITY,
        "kwargs": {
            "origin_resource_location": "prusa-mk4-2",
            "part_name": "MG",
        },
    }
    assert pre_execute_lock_states == [True]
    assert agent._robot_motion_lock.locked() is False

    observed.clear()
    blocked = asyncio.run(
        agent._execute_registered_robot_task_for_manual_function_execution(
            "pick_grasp",
            lambda: "current TCP moved",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )
    assert blocked == {
        "status": "blocked",
        "content": "current TCP moved",
        "manual_pre_execute_blocked": True,
    }
    assert observed == {}


def test_robot_agent_scopes_post_staging_acceptance_to_its_motion_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(RobotAgent)
    agent.agent_name = "ur5e"
    agent._robot_motion_lock = threading.Lock()
    agent._controller = SimpleNamespace()

    def callback(_requested_at: float) -> None:
        pass

    async def _execute(
        actual_agent: Any,
        task_name: str,
        _authority: object | None = None,
        /,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        assert task_name == "place_approach"
        assert actual_agent._robot_motion_lock.locked() is True
        assert (
            actual_agent._controller._assembly_board_v1_post_staging_accept_callback
            is callback
        )
        return {"status": "completed"}

    monkeypatch.setattr(robot_agent_module, "execute_robot_task", _execute)

    result = asyncio.run(
        agent._execute_registered_robot_task_for_manual_function_execution(
            "place_approach",
            None,
            callback,
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result == {"status": "completed"}
    assert not hasattr(
        agent._controller,
        "_assembly_board_v1_post_staging_accept_callback",
    )
    assert agent._robot_motion_lock.locked() is False
    assert agent._robot_motion_lock.locked() is False


@pytest.mark.parametrize("target", ["ur5e only", "dual robots"])
def test_readiness_accepts_each_hardware_led_ur5e_monitor_target(target: str) -> None:
    agent = _agent_for("pick_grasp")
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            target,
            "ur5e",
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert result["target"] == target


def test_pick_approach_readiness_blocks_without_actual_mg_stl_geometry() -> None:
    agent = _agent_for("pick_approach")
    bridge = _ready_bridge(agent)
    bridge._robot_function_product_geometry_for_part = lambda _part: {
        "part_name": "MG",
        "model_name": "gear_medium",
        "part_height_m": 0.02,
    }

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "actual source_stl" in result["message"]
    assert agent.calls == []


def test_pick_approach_readiness_ignores_manual_sequence_state_without_mutation() -> None:
    agent = _PhysicalUR5eAgent(state="picked")
    agent._task_ctx = _mg_task_context()
    retained_context = deepcopy(agent._task_ctx)
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert "pick_approach_reset_to_idle" not in result
    assert "state 'idle'" not in result["message"]
    assert agent._current_state == "picked"
    assert agent._task_ctx == retained_context
    assert agent.calls == []


def test_confirmed_pick_approach_dispatches_manual_mode_without_state_rewrite() -> None:
    agent = _PhysicalUR5eAgent(state="picked")
    agent._task_ctx = _mg_task_context()
    retained_context = deepcopy(agent._task_ctx)
    agent._recovery_pose_ref = "stale-pick"
    observed: dict[str, Any] = {}

    async def _pick_approach_manual(**kwargs: Any) -> dict[str, Any]:
        observed["state"] = agent._current_state
        observed["task_ctx"] = deepcopy(agent._task_ctx)
        observed["recovery_pose_ref"] = agent._recovery_pose_ref
        agent.calls.append(("pick_approach", kwargs))
        agent._current_state = "at_pick"
        return {"status": "completed", "content": "Arrived at the live pick target."}

    agent.pick_approach = _pick_approach_manual  # type: ignore[method-assign]
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert observed == {
        "state": "picked",
        "task_ctx": retained_context,
        "recovery_pose_ref": "stale-pick",
    }
    assert agent._current_state == "at_pick"


def test_pick_approach_manual_run_still_requires_an_empty_gripper() -> None:
    agent = _PhysicalUR5eAgent(
        state="picked",
        held_part="MG",
        gripper_state="closed",
    )
    agent._task_ctx = _mg_task_context()
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "empty ur5e gripper" in result["message"]
    assert agent._current_state == "picked"
    assert agent._task_ctx
    assert agent.calls == []


def test_pick_approach_manual_readiness_does_not_rewrite_uncertain_sequence_state() -> None:
    agent = _PhysicalUR5eAgent(state="picked")
    agent._task_ctx = _mg_task_context()
    bridge = _ready_bridge(agent)
    bridge._ur5e_robot_function_state_uncertain = True

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert "state is uncertain" not in result["message"]
    assert "pick_approach_reset_to_idle" not in result
    assert agent._current_state == "picked"
    assert agent._task_ctx
    assert agent.calls == []


@pytest.mark.parametrize(
    ("function_name", "manual_state", "arguments"),
    [
        (
            "pick_approach",
            "positioned",
            {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"},
        ),
        (
            "pick_grasp",
            "idle",
            {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"},
        ),
        (
            "place_approach",
            "picked",
            {"destination_location": "assembly_board-v1", "part_name": "MG"},
        ),
        (
            "place_insert",
            "picked",
            {"destination_location": "assembly_board-v1", "part_name": "MG"},
        ),
    ],
)
def test_manual_function_readiness_ignores_only_sequence_state(
    function_name: str,
    manual_state: str,
    arguments: dict[str, str],
) -> None:
    agent = _agent_for(function_name)
    agent._current_state = manual_state
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            function_name,
            **arguments,
        )
    )

    assert result["ready"] is True, result
    assert agent._current_state == manual_state
    assert agent.calls == []


def test_manual_pick_grasp_requires_pick_approach_descend_pose() -> None:
    agent = _agent_for("pick_grasp")
    agent._current_state = "idle"
    agent._task_ctx.pop("resolved_cartesian_positions")
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "pick_approach.descend" in result["message"]
    assert "Run pick_approach before pick_grasp" in result["message"]
    assert result["expected_start_pose"] == {}
    assert result["current_world_tool0_pose"] == {}
    assert result["expected_start_source"] == "pick_approach.descend"
    assert result["dispatch_attempted"] is False
    assert agent.calls == []


def test_manual_place_insert_rejects_tcp_moved_from_place_approach_descend() -> None:
    agent = _agent_for("place_insert")
    agent._current_state = "picked"
    bridge = _ready_bridge(agent)
    moved_pose = {**deepcopy(MANUAL_DESCEND_POSE), "x": MANUAL_DESCEND_POSE["x"] + 0.003}
    bridge._robot_function_capture_snapshot = lambda _target, _robot: {
        "success": True,
        "world_tool0_ready": True,
        "tf_stamp_sec": time.time(),
        "blocked_reason": "",
        "waypoint": {
            "source": "hardware",
            "pose": {
                **moved_pose,
                "frame_id": "world",
                "child_frame_id": "tool0",
            },
        },
    }

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "place_approach.descend" in result["message"]
    assert "3.00 mm" in result["message"]
    assert "limit 3.00 mm" in result["message"]
    assert agent.calls == []


def test_manual_pick_grasp_rechecks_tcp_after_acquiring_agent_motion_lock() -> None:
    agent = _agent_for("pick_grasp")
    agent._current_state = "idle"
    bridge = _ready_bridge(agent)
    poses = iter(
        (
            deepcopy(MANUAL_DESCEND_POSE),
            {
                **deepcopy(MANUAL_DESCEND_POSE),
                "x": MANUAL_DESCEND_POSE["x"] + 0.003,
            },
        )
    )

    def _snapshot(_target: str, _robot: str) -> dict[str, Any]:
        return {
            "success": True,
            "world_tool0_ready": True,
            "blocked_reason": "",
            "waypoint": {
                "source": "hardware",
                "pose": {
                    **next(poses),
                    "frame_id": "world",
                    "child_frame_id": "tool0",
                },
            },
        }

    bridge._robot_function_capture_snapshot = _snapshot

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert "pick_approach.descend" in result["message"]
    assert agent.calls == []
    assert agent._robot_motion_lock.locked() is False
    assert bridge._ur5e_robot_function_state_uncertain is False


def test_place_insert_rechecks_same_calibration_board_generation_before_dispatch() -> None:
    agent = _agent_for("place_insert")
    bridge = _ready_bridge(agent)
    base_board_status = dict(
        bridge.perception_manager.assembly_board_v1_aruco_status("ur5e")
    )
    accepted_generation = 1
    bridge.perception_manager.assembly_board_v1_aruco_status = lambda _role: {
        **base_board_status,
        "accepted_generation": accepted_generation,
        "accepted_calibration_id": "ur5e-calibration",
    }
    original_manual_execution = (
        agent._execute_registered_robot_task_for_manual_function_execution
    )

    async def _reaccept_before_pre_execute(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal accepted_generation
        accepted_generation = 2
        return await original_manual_execution(*args, **kwargs)

    agent._execute_registered_robot_task_for_manual_function_execution = (  # type: ignore[method-assign]
        _reaccept_before_pre_execute
    )

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert "accepted board generation changed after place_approach" in result["message"]
    assert "Run place_approach again before move_insert dispatch" in result["message"]
    assert agent.calls == []
    assert agent._current_state == "positioned"
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"


def test_place_insert_rechecks_full_task_context_identity_before_dispatch() -> None:
    agent = _agent_for("place_insert")
    bridge = _ready_bridge(agent)

    def _qualification_readiness(
        *,
        destination_location: str,
        part_name: str,
        settings: dict[str, Any],
        resource_agent: Any = None,
    ) -> tuple[dict[str, Any], str]:
        runtime_identities, identity_error = (
            bridge._move_insert_current_identities(
                resource_agent,
                destination_location=destination_location,
                part_name=part_name,
                settings=settings,
            )
        )
        return {
            "qualified": not identity_error,
            "qualification": {},
            "move_insert_runtime_identities": {
                key: deepcopy(value)
                for key, value in runtime_identities.items()
                if key != "resource_agent_identity"
            },
        }, identity_error

    bridge._move_insert_normal_qualification_readiness = (
        _qualification_readiness
    )
    original_manual_execution = (
        agent._execute_registered_robot_task_for_manual_function_execution
    )

    async def _mutate_handoff_before_pre_execute(
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        agent._task_ctx["held_part_handoff"] = _normal_mg_handoff()
        return await original_manual_execution(*args, **kwargs)

    agent._execute_registered_robot_task_for_manual_function_execution = (  # type: ignore[method-assign]
        _mutate_handoff_before_pre_execute
    )

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert "move_insert runtime tool_frame changed before dispatch" in (
        result["message"]
    )
    assert agent.calls == []
    assert agent._current_state == "positioned"
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"


@pytest.mark.parametrize(
    ("robot", "child_frame_id", "position_tolerance_m", "orientation_tolerance_rad"),
    [
        ("ur5e", "tool0", 0.002, 0.0349065850),
        ("xarm6", "link_eef", 0.003, 0.0523598776),
    ],
)
def test_manual_dependent_pose_uses_role_tolerance_and_quaternion_sign(
    robot: str,
    child_frame_id: str,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> None:
    bridge = object.__new__(SystemBridge)
    expected = {
        "x": 0.0,
        "y": 0.1,
        "z": 1.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    current = {
        **expected,
        "x": expected["x"] + position_tolerance_m,
        "qw": -1.0,
        "frame_id": "world",
        "child_frame_id": child_frame_id,
    }
    bridge._robot_function_execution_pose_readiness = lambda *_args: {
        "success": True,
        "world_tool0_ready": True,
        "waypoint": {"source": "hardware", "pose": deepcopy(current)},
    }
    controller = SimpleNamespace(
        _xarm6_cartesian_position_tolerance_m=position_tolerance_m,
        _xarm6_cartesian_orientation_tolerance_rad=orientation_tolerance_rad,
    )
    resource_agent = SimpleNamespace(_controller=controller)
    task_context = {
        "resolved_cartesian_positions": {"descend": deepcopy(expected)}
    }
    motion_readiness = {
        "cartesian_position_tolerance_m": position_tolerance_m,
        "cartesian_orientation_tolerance_rad": orientation_tolerance_rad,
    }

    readiness, error = bridge._manual_dependent_function_pose_error(
        "dual robots",
        robot,
        "pick_grasp",
        resource_agent,
        task_context,
        motion_readiness,
    )

    assert error == ""
    assert readiness["manual_pose_translation_error_m"] == pytest.approx(
        position_tolerance_m
    )
    assert readiness["manual_pose_rotation_error_rad"] == pytest.approx(0.0)

    current["x"] += 1e-5
    _readiness, moved_error = bridge._manual_dependent_function_pose_error(
        "dual robots",
        robot,
        "pick_grasp",
        resource_agent,
        task_context,
        motion_readiness,
    )
    assert "position error" in moved_error

    current["x"] = expected["x"]
    half_angle = 0.5 * (orientation_tolerance_rad + 1e-4)
    current.update({"qz": math.sin(half_angle), "qw": math.cos(half_angle)})
    _readiness, rotated_error = bridge._manual_dependent_function_pose_error(
        "dual robots",
        robot,
        "pick_grasp",
        resource_agent,
        task_context,
        motion_readiness,
    )
    assert "rotation error" in rotated_error


def test_place_insert_expected_start_mismatch_reports_structured_diagnostics() -> None:
    bridge = object.__new__(SystemBridge)
    expected = deepcopy(MANUAL_DESCEND_POSE)
    position_delta_m = 0.51619
    current = {
        **expected,
        "x": expected["x"] + position_delta_m,
        "frame_id": "world",
        "child_frame_id": "tool0",
    }
    tf_stamp_sec = time.time()
    bridge._robot_function_execution_pose_readiness = lambda *_args: {
        "success": True,
        "world_tool0_ready": True,
        "tf_stamp_sec": tf_stamp_sec,
        "waypoint": {"source": "hardware", "pose": deepcopy(current)},
    }
    resource_agent = SimpleNamespace(_controller=SimpleNamespace())
    task_context = {
        "resolved_cartesian_positions": {"descend": deepcopy(expected)}
    }
    motion_readiness = {
        "cartesian_position_tolerance_m": 0.002,
        "cartesian_orientation_tolerance_rad": 0.05,
        "insert_start_position_tolerance_m": 0.003,
        "insert_start_orientation_tolerance_rad": 0.05,
    }

    readiness, error = bridge._manual_dependent_function_pose_error(
        "dual robots",
        "ur5e",
        "place_insert",
        resource_agent,
        task_context,
        motion_readiness,
    )

    assert "blocked before move_insert dispatch" in error
    assert "516.19 mm" in error
    assert "No move_insert goal" in error
    assert readiness["expected_start_pose"] == {
        **expected,
        "frame_id": "world",
        "child_frame_id": "tool0",
    }
    assert readiness["current_world_tool0_pose"] == current
    assert readiness["expected_start_delta_m"] == pytest.approx(
        {"x": position_delta_m, "y": 0.0, "z": 0.0}
    )
    assert readiness["expected_start_position_error_m"] == pytest.approx(
        position_delta_m
    )
    assert readiness["expected_start_rotation_error_rad"] == pytest.approx(0.0)
    assert readiness["expected_start_position_tolerance_m"] == pytest.approx(
        0.003
    )
    assert readiness["expected_start_orientation_tolerance_rad"] == pytest.approx(
        0.05
    )
    assert readiness["expected_start_tf_timestamp_sec"] == pytest.approx(
        tf_stamp_sec
    )
    assert 0.0 <= readiness["expected_start_tf_age_sec"] < 1.0
    assert readiness["expected_start_source"] == "place_approach.descend"
    assert readiness["dispatch_attempted"] is False


@pytest.mark.parametrize(
    ("tf_stamp", "message_fragment"),
    [
        ("fresh", ""),
        (None, "no source header timestamp"),
        ("stale", "source header is not fresh"),
    ],
)
def test_place_insert_expected_start_requires_fresh_tf_before_dispatch(
    tf_stamp: str | None,
    message_fragment: str,
) -> None:
    bridge = object.__new__(SystemBridge)
    expected = deepcopy(MANUAL_DESCEND_POSE)
    snapshot = {
        "success": True,
        "world_tool0_ready": True,
        "waypoint": {
            "source": "hardware",
            "pose": {
                **expected,
                "frame_id": "world",
                "child_frame_id": "tool0",
            },
        },
    }
    if tf_stamp == "fresh":
        snapshot["tf_stamp_sec"] = time.time()
    elif tf_stamp == "stale":
        snapshot["tf_stamp_sec"] = time.time() - 3.1
    bridge._robot_function_execution_pose_readiness = lambda *_args: deepcopy(
        snapshot
    )
    resource_agent = SimpleNamespace(_controller=SimpleNamespace())

    readiness, error = bridge._manual_dependent_function_pose_error(
        "dual robots",
        "ur5e",
        "place_insert",
        resource_agent,
        {"resolved_cartesian_positions": {"descend": expected}},
        {
            "cartesian_position_tolerance_m": 0.002,
            "cartesian_orientation_tolerance_rad": 0.05,
        },
    )

    if message_fragment:
        assert message_fragment in error
    else:
        assert error == ""
    assert readiness["dispatch_attempted"] is False
    assert readiness["expected_start_source"] == "place_approach.descend"


def test_pick_grasp_readiness_blocks_invalid_stl_grounded_context() -> None:
    agent = _agent_for("pick_grasp")
    agent._task_ctx["source_stl"] = "/tmp/not-the-actual-mg.STL"
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "actual MG geometry must come from" in result["message"]
    assert agent.calls == []


def test_pick_grasp_readiness_allows_zero_fingertip_hub_overlap() -> None:
    agent = _agent_for("pick_grasp")
    agent._task_ctx["finger_hub_overlap_m"] = 0.0
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert "overlap" not in result["message"]
    assert agent.calls == []


def test_mg_close_test_closes_once_holds_and_reopens_without_arm_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_to_thread_inline(monkeypatch)
    agent = _agent_for("pick_grasp")
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_execute_mg_close_test(
            "ur5e only",
            "ur5e",
            "prusa-mk4-2",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert result["gripper_close_position"] == pytest.approx(0.047)
    assert result["hold_sec"] == pytest.approx(0.0)
    assert agent._controller.gripper_calls == [
        ("close_gripper", pytest.approx(0.047)),
        ("open_gripper", None),
    ]
    assert agent.calls == []
    assert agent._current_state == "at_pick"
    assert agent._gripper_state == "open"
    assert bridge._ur5e_robot_function_state_uncertain is False


@pytest.mark.parametrize(
    ("part_name", "gripper_close_position"),
    [("SG", 0.061), ("MG", 0.047), ("LG", 0.033)],
)
def test_gripper_close_test_uses_each_retained_part_position_without_arm_motion(
    part_name: str,
    gripper_close_position: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_to_thread_inline(monkeypatch)
    agent = _agent_for("pick_grasp")
    agent._task_ctx.update(
        {
            "part_name": part_name,
            "gripper_close_position": gripper_close_position,
        }
    )
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_execute_gripper_close_test(
            "ur5e only",
            "ur5e",
            "prusa-mk4-2",
            part_name,
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert result["part_name"] == part_name
    assert result["gripper_close_position"] == pytest.approx(gripper_close_position)
    assert agent._controller.gripper_calls == [
        ("close_gripper", pytest.approx(gripper_close_position)),
        ("open_gripper", None),
    ]
    assert agent.calls == []
    assert agent._current_state == "at_pick"
    assert agent._gripper_state == "open"


@pytest.mark.parametrize(
    ("close_success", "reopen_success", "failed_action"),
    [
        (False, True, "close"),
        (True, False, "reopen"),
    ],
)
def test_mg_close_test_reopens_in_finally_and_failed_cycle_requires_move_home(
    close_success: bool,
    reopen_success: bool,
    failed_action: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_to_thread_inline(monkeypatch)
    agent = _agent_for("pick_grasp")
    agent._controller.close_success = close_success
    agent._controller.reopen_success = reopen_success
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_execute_mg_close_test(
            "dual robots",
            "ur5e",
            "prusa-mk4-2",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert failed_action in result["message"]
    assert "complete move_home" not in result["message"]
    assert "inspect the UR5e" in result["message"]
    assert agent._controller.gripper_calls[-1] == ("open_gripper", None)
    assert agent.calls == []
    assert bridge._ur5e_robot_function_state_uncertain is True

    readiness = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_grasp",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )
    assert readiness["ready"] is True
    assert "state is uncertain" not in readiness["message"]


def test_mg_close_test_requires_at_pick_empty_mg_context_and_motion_locks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_to_thread_inline(monkeypatch)
    idle_agent = _PhysicalUR5eAgent(state="idle")
    idle_bridge = _ready_bridge(idle_agent)

    idle = asyncio.run(
        idle_bridge.digital_twin_execute_mg_close_test(
            "ur5e only",
            "ur5e",
            "prusa-mk4-2",
            confirmed=True,
        )
    )

    assert idle["success"] is False
    assert "active pick context" in idle["message"]
    assert idle_agent._controller.gripper_calls == []

    closed_agent = _agent_for("pick_grasp")
    closed_agent._gripper_state = "closed"
    closed_bridge = _ready_bridge(closed_agent)
    closed = asyncio.run(
        closed_bridge.digital_twin_execute_mg_close_test(
            "ur5e only",
            "ur5e",
            "prusa-mk4-2",
            confirmed=True,
        )
    )
    assert closed["success"] is False
    assert "empty RG2 to be open" in closed["message"]
    assert closed_agent._controller.gripper_calls == []

    locked_agent = _agent_for("pick_grasp")
    locked_bridge = _ready_bridge(locked_agent)
    locked_agent._robot_motion_lock.acquire()
    try:
        locked = asyncio.run(
            locked_bridge.digital_twin_execute_mg_close_test(
                "ur5e only",
                "ur5e",
                "prusa-mk4-2",
                confirmed=True,
            )
        )
    finally:
        locked_agent._robot_motion_lock.release()

    assert locked["success"] is False
    assert "already executing" in locked["message"]
    assert locked_agent._controller.gripper_calls == []


def test_readiness_waits_for_a_running_ur5e_mirror_to_recover() -> None:
    bridge = _ready_bridge(_agent_for("move_home"))
    bridge._ROBOT_FUNCTION_MIRROR_RECOVERY_TIMEOUT_S = 1.0
    calls = [0]

    def _statuses() -> dict[str, dict[str, Any]]:
        calls[0] += 1
        if calls[0] == 1:
            return {
                "ur5e only": {
                    "target": "ur5e only",
                    "repair_needed": True,
                    "repair_reason": "ur5e mirror is waiting",
                    "gazebo": {"status": "running"},
                    "hardware": {"overall": "running"},
                    "sync/status": {
                        "process_status": "running",
                        "state": "waiting",
                    },
                }
            }
        return {"ur5e only": _healthy_status("ur5e only")}

    bridge.digital_twin_statuses = _statuses

    error = asyncio.run(
        bridge._wait_for_digital_twin_robot_function_target_error(
            "ur5e only",
            bridge._DIGITAL_TWIN_TARGETS["ur5e only"],
        )
    )

    assert error == ""
    assert calls[0] == 2


def test_readiness_does_not_wait_for_a_stopped_ur5e_mirror_process() -> None:
    bridge = _ready_bridge(_agent_for("move_home"))
    bridge._ROBOT_FUNCTION_MIRROR_RECOVERY_TIMEOUT_S = 1.0
    calls = [0]

    def _statuses() -> dict[str, dict[str, Any]]:
        calls[0] += 1
        return {
            "ur5e only": {
                "target": "ur5e only",
                "repair_needed": True,
                "repair_reason": "hardware -> gazebo mirror process is stopped",
                "gazebo": {"status": "running"},
                "hardware": {"overall": "running"},
                "sync/status": {
                    "process_status": "stopped",
                    "state": "waiting",
                },
            }
        }

    bridge.digital_twin_statuses = _statuses

    error = asyncio.run(
        bridge._wait_for_digital_twin_robot_function_target_error(
            "ur5e only",
            bridge._DIGITAL_TWIN_TARGETS["ur5e only"],
        )
    )

    assert "requires Repair Twin" in error
    assert calls[0] == 1


@pytest.mark.parametrize(
    ("target", "robot", "function_name", "arguments", "message"),
    [
        ("xarm only", "ur5e", "move_home", {}, "ur5e is not part"),
        ("unknown", "ur5e", "move_home", {}, "unknown digital twin target"),
        ("dual robots", "UR5E", "move_home", {}, "UR5E is not part"),
        ("dual robots", "ur5e", "Pick_Approach", {}, "unknown robot function"),
        (
            "dual robots",
            "ur5e",
            "pick_grasp",
            {
                "origin_resource_location": "prusa-mk4-2",
                "destination_location": "assembly_board-v1",
                "part_name": "MG",
            },
            "does not accept destination_location",
        ),
        (
            "dual robots",
            "ur5e",
            "move_home",
            {"part_name": "MG"},
            "does not accept location or part arguments",
        ),
    ],
)
def test_readiness_rejects_nonexact_or_irrelevant_requests(
    target: str,
    robot: str,
    function_name: str,
    arguments: dict[str, str],
    message: str,
) -> None:
    bridge = _ready_bridge(_PhysicalUR5eAgent())

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            target,
            robot,
            function_name,
            **arguments,
        )
    )

    assert result["ready"] is False
    assert message in result["message"]


@pytest.mark.parametrize("function_name", ["pick_approach", "place_approach"])
def test_approach_functions_use_only_configured_product_when_unselected(
    function_name: str,
) -> None:
    agent = _agent_for(function_name)
    bridge = _ready_bridge(agent)
    bridge.selected_product = ""
    product_file = "/tmp/assembly_board-v1.json"
    bridge.list_product_files = lambda: [product_file]
    arguments = (
        {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"}
        if function_name == "pick_approach"
        else {"destination_location": "assembly_board-v1", "part_name": "MG"}
    )

    readiness = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            function_name,
            **arguments,
        )
    )
    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            function_name,
            **arguments,
            confirmed=True,
        )
    )

    assert readiness["ready"] is True
    assert readiness["selected_product"] == product_file
    assert result["success"] is True
    assert agent.calls[0][0] == function_name
    assert agent.calls[0][1]["part_name"] == "MG"
    assert agent.calls[0][1]["product_geometry"]["part_name"] == "MG"


@pytest.mark.parametrize("function_name", ["pick_approach", "place_approach"])
def test_approach_functions_require_selection_when_multiple_products_are_configured(
    function_name: str,
) -> None:
    agent = _agent_for(function_name)
    bridge = _ready_bridge(agent)
    bridge.selected_product = ""
    bridge.list_product_files = lambda: ["product-a.json", "product-b.json"]
    bridge._robot_function_product_geometry_for_part = lambda _part: pytest.fail(
        "geometry lookup must not choose between multiple products"
    )
    arguments = (
        {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"}
        if function_name == "pick_approach"
        else {"destination_location": "assembly_board-v1", "part_name": "MG"}
    )

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            function_name,
            **arguments,
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert "Select a product" in result["message"]
    assert agent.calls == []


def test_preflight_requires_exact_executable_membership() -> None:
    agent = _agent_for("pick_grasp")
    agent.executables.pop("pick_grasp")
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_execute_pick_grasp(
            "dual robots",
            "ur5e",
            "prusa-mk4-2",
            "MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert "does not expose executable pick_grasp" in result["message"]
    assert agent.calls == []


@pytest.mark.parametrize(
    ("function_name", "pose_name", "pose", "arguments"),
    [
        (
            "pick_approach",
            "prusa-mk4-2",
            [0.0] * 5,
            {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"},
        ),
        (
            "move_home",
            "home",
            [0.0, 0.0, 0.0, 0.0, 0.0, float("nan")],
            {},
        ),
    ],
)
def test_named_position_gates_reject_incomplete_or_nonfinite_joints(
    function_name: str,
    pose_name: str,
    pose: list[float],
    arguments: dict[str, str],
) -> None:
    agent = _agent_for(function_name)
    agent.named_positions[pose_name] = pose
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            function_name,
            **arguments,
        )
    )

    assert result["ready"] is False
    assert "joint values" in result["message"]
    assert agent.calls == []


@pytest.mark.parametrize("function_name", ["place_approach", "place_insert"])
def test_place_functions_require_logically_closed_gripper(function_name: str) -> None:
    agent = _agent_for(function_name)
    agent._gripper_state = "open"
    bridge = _ready_bridge(agent)
    arguments = {"destination_location": "assembly_board-v1", "part_name": "MG"}

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            function_name,
            **arguments,
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert "gripper_state 'closed'" in result["message"]
    assert agent.calls == []


@pytest.mark.parametrize(
    "status",
    [
        {
            "accepted": False,
            "accepted_generation": 0,
            "accepted_baseline_ready": False,
            "accepted_baseline_error": (
                "Locate & Accept Board for ur5e before using assembly_board-v1."
            ),
            "post_staging_acceptance_allowed": True,
        },
        {
            "accepted": True,
            "accepted_generation": 5,
            "accepted_baseline_ready": False,
            "accepted_baseline_error": (
                "assembly_board-v1 moved more than 10 mm or 2 deg from the accepted "
                "ur5e pose."
            ),
            "movement_blocked": True,
            "post_staging_acceptance_allowed": True,
        },
    ],
)
def test_place_approach_allows_post_staging_board_acceptance(
    status: dict[str, Any],
) -> None:
    agent = _agent_for("place_approach")
    bridge = _ready_bridge(agent)
    bridge.perception_manager.assembly_board_v1_aruco_status = lambda _role: dict(status)

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert agent.calls == [
        (
            "place_approach",
            {
                "destination_location": "assembly_board-v1",
                "part_name": "MG",
                "product_geometry": _mg_place_product_geometry(),
            },
        )
    ]


def test_place_approach_still_blocks_post_staging_acceptance_after_calibration_change() -> None:
    agent = _agent_for("place_approach")
    bridge = _ready_bridge(agent)
    bridge.perception_manager.assembly_board_v1_aruco_status = lambda _role: {
        "accepted": True,
        "accepted_generation": 5,
        "accepted_baseline_ready": False,
        "accepted_baseline_error": "The active ur5e calibration identity changed.",
        "calibration_changed": True,
        "post_staging_acceptance_allowed": False,
    }

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert "calibration identity changed" in result["message"]
    assert agent.calls == []


def test_place_approach_post_staging_acceptance_is_scoped_to_confirmed_runtime() -> None:
    agent = _agent_for("place_approach")
    bridge = _ready_bridge(agent)
    calls: list[tuple[str, object]] = []

    bridge.perception_manager.assembly_board_v1_aruco_status = lambda role: {
        "camera_role": role,
        "accepted": True,
        "accepted_generation": 5,
        "accepted_baseline_ready": False,
        "accepted_baseline_error": "board movement requires post-staging acceptance",
        "movement_blocked": True,
        "calibration_changed": False,
        "post_staging_acceptance_allowed": True,
    }

    def _accept(role: str, *, minimum_sample_started_at: float | None = None) -> dict[str, Any]:
        calls.append((role, minimum_sample_started_at))
        return {"success": True, "accepted_generation": 6}

    bridge.perception_manager.locate_and_accept_assembly_board_v1 = _accept

    async def _place_approach(**kwargs: Any) -> dict[str, Any]:
        callback = agent._controller._assembly_board_v1_post_staging_accept_callback
        assert callable(callback)
        accepted = callback(123.5)
        assert accepted["accepted_generation"] == 6
        agent.calls.append(("place_approach", kwargs))
        return {"status": "completed", "content": "Reached assembly_board-v1."}

    agent.place_approach = _place_approach

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert calls == [("ur5e", 123.5)]
    assert not hasattr(
        agent._controller,
        "_assembly_board_v1_post_staging_accept_callback",
    )


def test_place_approach_reaccepts_fresh_post_staging_cross_view_movement() -> None:
    agent = _agent_for("place_approach")
    bridge = _ready_bridge(agent)
    calls: list[tuple[str, object]] = []

    bridge.perception_manager.assembly_board_v1_aruco_status = lambda role: {
        "camera_role": role,
        "accepted": True,
        "accepted_generation": 5,
        "accepted_baseline_ready": True,
        "accepted_baseline_error": "",
        "calibration_changed": False,
        "calibration_identity_matches": True,
        "ready_to_accept": True,
        "movement_blocked": False,
        "movement_evidence_valid": True,
        "excessive_movement": True,
        "post_staging_acceptance_allowed": False,
    }

    def _accept(
        role: str,
        *,
        minimum_sample_started_at: float | None = None,
    ) -> dict[str, Any]:
        calls.append((role, minimum_sample_started_at))
        return {"success": True, "accepted_generation": 6}

    bridge.perception_manager.locate_and_accept_assembly_board_v1 = _accept

    async def _place_approach(**kwargs: Any) -> dict[str, Any]:
        callback = agent._controller._assembly_board_v1_post_staging_accept_callback
        accepted = callback(456.75)
        assert accepted["accepted_generation"] == 6
        agent.calls.append(("place_approach", kwargs))
        return {"status": "completed", "content": "Reached assembly_board-v1."}

    agent.place_approach = _place_approach

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert calls == [("ur5e", 456.75)]
    assert not hasattr(
        agent._controller,
        "_assembly_board_v1_post_staging_accept_callback",
    )


def test_place_approach_accepts_occluded_but_usable_board_baseline() -> None:
    agent = _agent_for("place_approach")
    bridge = _ready_bridge(agent)
    bridge.perception_manager.assembly_board_v1_aruco_status = lambda _role: {
        "accepted": True,
        "accepted_baseline_ready": True,
        "accepted_baseline_error": "",
        "accepted_generation": 1,
        "accepted_calibration_id": "ur5e-calibration",
        "visible": False,
        "movement_evidence_valid": False,
        "movement_blocked": False,
    }

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is True


def test_place_insert_rejects_reaccepted_board_before_confirmation() -> None:
    agent = _agent_for("place_insert")
    bridge = _ready_bridge(agent)
    bridge.perception_manager.assembly_board_v1_aruco_status = lambda _role: {
        "accepted": True,
        "accepted_baseline_ready": True,
        "accepted_baseline_error": "",
        "accepted_generation": 2,
        "accepted_calibration_id": "ur5e-calibration",
    }

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "accepted generation changed after place_approach" in result["message"]
    assert agent.calls == []


@pytest.mark.parametrize(
    ("bridge_method", "manager_method", "expected"),
    [
        (
            "perception_locate_and_accept_assembly_board_v1",
            "locate_and_accept_assembly_board_v1",
            {"success": True, "accepted_generation": 2},
        ),
        (
            "perception_activate_calibration",
            "activate_calibration",
            Path("/tmp/ur5e_realsense_hand_eye.yaml"),
        ),
        (
            "perception_rollback_calibration",
            "rollback_calibration",
            Path("/tmp/ur5e_realsense_hand_eye.yaml"),
        ),
    ],
)
def test_board_acceptance_and_calibration_identity_changes_use_execution_lock(
    bridge_method: str,
    manager_method: str,
    expected: object,
) -> None:
    bridge = _ready_bridge(_agent_for("place_approach"))
    calls: list[str] = []

    def _change(role: str) -> object:
        assert bridge._ur5e_robot_function_execution_lock.locked() is True
        calls.append(role)
        return expected

    setattr(bridge.perception_manager, manager_method, _change)

    result = getattr(bridge, bridge_method)("ur5e")

    assert result == expected
    assert calls == ["ur5e"]
    assert bridge._ur5e_robot_function_execution_lock.acquire(blocking=False) is True
    bridge._ur5e_robot_function_execution_lock.release()


@pytest.mark.parametrize(
    ("bridge_method", "manager_method", "operation_name"),
    [
        (
            "perception_locate_and_accept_assembly_board_v1",
            "locate_and_accept_assembly_board_v1",
            "Locate & Accept Board (xarm6)",
        ),
        (
            "perception_activate_calibration",
            "activate_calibration",
            "Activate calibration (xarm6)",
        ),
        (
            "perception_rollback_calibration",
            "rollback_calibration",
            "Rollback calibration (xarm6)",
        ),
    ],
)
def test_board_acceptance_and_calibration_identity_changes_refuse_during_execution(
    bridge_method: str,
    manager_method: str,
    operation_name: str,
) -> None:
    bridge = _ready_bridge(_agent_for("place_approach"))
    calls: list[str] = []
    setattr(
        bridge.perception_manager,
        manager_method,
        lambda role: calls.append(role),
    )
    bridge._ur5e_robot_function_execution_active = "place_approach"
    bridge._ur5e_robot_function_execution_lock.acquire()
    try:
        with pytest.raises(RuntimeError) as error:
            getattr(bridge, bridge_method)("xarm6")
    finally:
        bridge._ur5e_robot_function_execution_lock.release()

    assert str(error.value) == (
        f"{operation_name} is unavailable while physical robot function execution is active: "
        "place_approach."
    )
    assert calls == []


def test_board_acceptance_releases_execution_lock_after_perception_failure() -> None:
    bridge = _ready_bridge(_agent_for("place_approach"))

    def _fail(_role: str) -> dict[str, Any]:
        raise RuntimeError("fresh stable ArUco observation is unavailable")

    bridge.perception_manager.locate_and_accept_assembly_board_v1 = _fail

    with pytest.raises(RuntimeError, match="fresh stable ArUco observation is unavailable"):
        bridge.perception_locate_and_accept_assembly_board_v1("ur5e")

    assert bridge._ur5e_robot_function_execution_lock.acquire(blocking=False) is True
    bridge._ur5e_robot_function_execution_lock.release()


def test_automatic_board_acceptance_is_idempotent_after_one_accepted_generation() -> None:
    bridge = _ready_bridge(_agent_for("place_approach"))
    calls: list[str] = []

    def _status(role: str) -> dict[str, Any]:
        assert bridge._ur5e_robot_function_execution_lock.locked() is True
        calls.append(f"status:{role}")
        return {
            "camera_role": role,
            "accepted": False,
            "accepted_generation": 4,
            "accepted_baseline_ready": True,
        }

    def _accept(role: str) -> dict[str, Any]:
        calls.append(f"accept:{role}")
        return {"success": True, "accepted_generation": 5}

    bridge.perception_manager.assembly_board_v1_aruco_status = _status
    bridge.perception_manager.locate_and_accept_assembly_board_v1 = _accept

    result = bridge.perception_locate_and_accept_assembly_board_v1(
        "ur5e",
        only_if_unaccepted=True,
    )

    assert result == {
        "camera_role": "ur5e",
        "accepted": False,
        "accepted_generation": 4,
        "accepted_baseline_ready": True,
        "success": True,
        "auto_accepted": False,
    }
    assert calls == ["status:ur5e"]


def test_automatic_board_acceptance_accepts_one_unaccepted_observation() -> None:
    bridge = _ready_bridge(_agent_for("place_approach"))
    calls: list[str] = []

    def _status(role: str) -> dict[str, Any]:
        assert bridge._ur5e_robot_function_execution_lock.locked() is True
        calls.append(f"status:{role}")
        return {
            "camera_role": role,
            "accepted": False,
            "accepted_generation": 0,
            "ready_to_accept": True,
        }

    def _accept(role: str) -> dict[str, Any]:
        assert bridge._ur5e_robot_function_execution_lock.locked() is True
        calls.append(f"accept:{role}")
        return {
            "success": True,
            "camera_role": role,
            "accepted": True,
            "accepted_generation": 1,
        }

    bridge.perception_manager.assembly_board_v1_aruco_status = _status
    bridge.perception_manager.locate_and_accept_assembly_board_v1 = _accept

    result = bridge.perception_locate_and_accept_assembly_board_v1(
        "xarm6",
        only_if_unaccepted=True,
    )

    assert result == {
        "success": True,
        "camera_role": "xarm6",
        "accepted": True,
        "accepted_generation": 1,
        "auto_accepted": True,
    }
    assert calls == ["status:xarm6", "accept:xarm6"]


def test_solving_calibration_candidate_remains_unlocked() -> None:
    bridge = _ready_bridge(_agent_for("place_approach"))
    expected = Path("/tmp/ur5e_realsense_hand_eye.candidate.yaml")
    bridge.perception_manager.solve_calibration = lambda role: expected
    bridge._ur5e_robot_function_execution_active = "place_insert"
    bridge._ur5e_robot_function_execution_lock.acquire()
    try:
        assert bridge.perception_solve_calibration("ur5e") == expected
    finally:
        bridge._ur5e_robot_function_execution_lock.release()


def test_gripper_functions_require_exact_target_domain_rg2_action() -> None:
    agent = _agent_for("pick_grasp")
    bridge = _ready_bridge(agent)
    bridge._digital_twin_ur5e_gripper_readiness = lambda target, domain, _agent=None: (
        {"gripper_action_ready": False},
        (
            "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory is unavailable "
            f"on ROS_DOMAIN_ID={domain} for {target}"
        ),
    )

    result = asyncio.run(
        bridge.digital_twin_execute_pick_grasp(
            "ur5e only",
            "ur5e",
            "prusa-mk4-2",
            "MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert "ROS_DOMAIN_ID=42" in result["message"]
    assert agent.calls == []


def test_motion_and_gripper_probes_use_exact_actions_and_target_domain() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._digital_twin_domain_ids = lambda: {
        "gazebo": 41,
        "hardware": 42,
        "hardware_xarm6": 42,
        "hardware_ur5e": 43,
    }
    bridge._digital_twin_hardware_domain_id = lambda _cfg, _robot, _domains: 43
    rtde_status = {
        "updated_at": time.time(),
        "ros_domain_id": 43,
        "action_name": "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory",
        "rtde_receive_connected": True,
        "joint_states_fresh": True,
        "rtde_control_connected": True,
        "joint_goal_tolerance_rad": 0.025,
        "actual_positions_rad": [0.1, -0.8, -2.1, -1.6, 1.5, -3.1],
    }
    bridge._ur5e_rtde_trajectory_status = lambda: dict(rtde_status)
    bridge._ur5e_rtde_result_timeout_requires_repair = lambda _status: False
    probes: list[tuple[str, int | None]] = []

    def _wait(action: str, **kwargs: Any) -> None:
        probes.append((action, kwargs.get("ros_domain_id")))
        return None

    bridge._wait_for_ros_action = _wait
    motion, motion_error = bridge._digital_twin_ur5e_motion_readiness(
        "dual robots", {"hardware": ("xarm6", "ur5e")}
    )
    gripper, gripper_error = bridge._digital_twin_ur5e_gripper_readiness("dual robots", 43)

    assert motion_error == ""
    assert gripper_error == ""
    assert motion["trajectory_action_ready"] is True
    assert motion["joint_goal_tolerance_rad"] == pytest.approx(0.025)
    assert motion["actual_positions_rad"] == pytest.approx(
        [0.1, -0.8, -2.1, -1.6, 1.5, -3.1]
    )
    assert gripper["gripper_action_ready"] is True
    assert probes == [
        ("/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory", 43),
        ("/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory", 43),
    ]
    rtde_status["ros_domain_id"] = 42
    _wrong_domain, wrong_domain_error = bridge._digital_twin_ur5e_motion_readiness(
        "dual robots", {"hardware": ("xarm6", "ur5e")}
    )
    assert "belongs to ROS_DOMAIN_ID=42" in wrong_domain_error
    assert "requested ROS_DOMAIN_ID=43" in wrong_domain_error
    assert len(probes) == 2


def _recorded_step(step_name: str, z: float) -> dict[str, Any]:
    pose = {
        "x": -0.11,
        "y": 0.42,
        "z": z,
        "qx": 0.0,
        "qy": 0.70710678,
        "qz": 0.0,
        "qw": 0.70710678,
    }
    return {
        "step_name": step_name,
        "primitive": "move_cartesian",
        "params": deepcopy(pose),
        "capture_source": "hardware",
        "confirmed": True,
        "position_sources": {
            "x": "captured_relative",
            "y": "captured_relative",
            "z": "captured_relative",
        },
        "relative_position_m": {
            "x": -0.61,
            "y": -0.18,
            "z": z - 0.3,
        },
        "computed_position_m": {"x": 0.5, "y": 0.6, "z": 0.3},
        "computed_pose": {
            "x": 0.5,
            "y": 0.6,
            "z": 0.3,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        "computed_source": "test",
        "computed_at": time.time(),
        "relative_pose": {
            "x": -0.61,
            "y": -0.18,
            "z": z - 0.3,
            "qx": 0.0,
            "qy": 0.70710678,
            "qz": 0.0,
            "qw": 0.70710678,
        },
        "relative_reference": {
            "kind": "destination_target",
            "frame_id": "world",
            "name": "assembly_board-v1",
            "position_m": {"x": 0.5, "y": 0.6, "z": 0.3},
            "pose": {
                "x": 0.5,
                "y": 0.6,
                "z": 0.3,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "source": "assembly_board-v1_aruco",
            "captured_at": time.time(),
            "camera_role": "ur5e",
            "generation": 1,
            "calibration_id": "ur5e-calibration",
        },
        "waypoint": {
            "pose": {
                "frame_id": "world",
                "child_frame_id": "tool0",
                **deepcopy(pose),
            },
            "joint_names": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            "joint_positions": [0.0] * 6,
            "source": "hardware",
        },
    }


def test_place_readiness_runs_complete_runtime_recording_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_task_runtime, "_TAUGHT_FUNCTIONS_ROOT", tmp_path)
    path = tmp_path / "place_approach" / "default__hardware.json"
    path.parent.mkdir(parents=True)
    payload = {
        "function_name": "place_approach",
        "capture_source": "hardware",
        "robots": {
            "ur5e": {
                "frame_id": "world",
                "ee_link": "tool0",
                "tcp_link": "ur5e_rg2_gripper_tcp",
                "steps": [
                    _recorded_step("move_above_destination", 0.69),
                    _recorded_step("descend", 0.36),
                ],
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    agent = _agent_for("place_approach")
    bridge = _ready_bridge(agent)
    del bridge._digital_twin_place_approach_recording_error

    async def _check_both_recordings() -> tuple[dict[str, Any], dict[str, Any]]:
        ready_result = await bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
        payload["robots"]["ur5e"]["steps"][1] = deepcopy(
            payload["robots"]["ur5e"]["steps"][0]
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        blocked_result = await bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
        return ready_result, blocked_result

    ready, blocked = asyncio.run(_check_both_recordings())
    assert ready["ready"] is True
    assert blocked["ready"] is False
    assert "duplicate physical position step_name" in blocked["message"]


def test_confirmation_and_generalized_motion_lock_block_before_preflight() -> None:
    agent = _agent_for("move_home")
    bridge = _ready_bridge(agent)
    unconfirmed = asyncio.run(
        bridge.digital_twin_execute_robot_function("dual robots", "ur5e", "move_home")
    )
    assert "Explicit operator confirmation" in unconfirmed["message"]

    bridge._ur5e_robot_function_execution_lock.acquire()
    bridge._ur5e_robot_function_execution_active = "place_insert"
    busy = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots", "ur5e", "move_home", confirmed=True
        )
    )
    bridge._ur5e_robot_function_execution_lock.release()
    assert busy["active_function"] == "place_insert"
    assert agent.calls == []


@pytest.mark.parametrize("failure_kind", ["result", "exception"])
def test_post_dispatch_failure_warns_that_physical_state_may_have_changed(
    failure_kind: str,
) -> None:
    agent = _agent_for("move_home")

    async def _fail(**_kwargs: Any) -> dict[str, Any]:
        if failure_kind == "exception":
            raise RuntimeError("RTDE transport ended")
        return {
            "status": "failed",
            "content": "move_to_named_pose failed",
            "failure_context": {"observations": {"step": "move_home.move_home"}},
        }

    agent.move_home = _fail
    bridge = _ready_bridge(agent)
    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots", "ur5e", "move_home", confirmed=True
        )
    )

    assert result["success"] is False
    assert "Physical state may have changed" in result["message"]
    assert "inspect the robot and recover before retrying" in result["message"]
    if failure_kind == "result":
        assert "Failed step: move_home.move_home" in result["message"]


def test_ui_cancellation_keeps_generalized_lock_until_runtime_finishes() -> None:
    async def _exercise() -> None:
        agent = _agent_for("pick_approach")
        bridge = _ready_bridge(agent)
        started = asyncio.Event()
        finish = asyncio.Event()

        async def _slow_pick_approach(**kwargs: Any) -> dict[str, Any]:
            agent.calls.append(("pick_approach", kwargs))
            started.set()
            await finish.wait()
            return {"status": "completed", "content": "motion complete"}

        agent.pick_approach = _slow_pick_approach
        execution = asyncio.create_task(
            bridge.digital_twin_execute_pick_approach(
                "dual robots",
                "ur5e",
                "prusa-mk4-2",
                "MG",
                confirmed=True,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=10.0)
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution

        assert bridge._ur5e_robot_function_execution_lock.acquire(blocking=False) is False
        assert bridge._ur5e_robot_function_execution_active == "pick_approach"

        finish.set()
        for _attempt in range(20):
            await asyncio.sleep(0)
            if bridge._ur5e_robot_function_execution_lock.acquire(blocking=False):
                bridge._ur5e_robot_function_execution_lock.release()
                break
        else:
            pytest.fail("UR5e lock was not released after the agent runtime completed")
        assert bridge._ur5e_robot_function_execution_active is None

    asyncio.run(_exercise())


def test_ui_cancellation_keeps_generalized_lock_until_preflight_finishes() -> None:
    async def _exercise() -> None:
        agent = _agent_for("move_home")
        bridge = _ready_bridge(agent)
        started = threading.Event()
        finish = threading.Event()
        original_preflight = bridge._digital_twin_robot_function_execution_preflight

        def _slow_preflight(*args: Any) -> tuple[Any | None, dict[str, Any], dict[str, Any], str]:
            started.set()
            assert finish.wait(timeout=2.0)
            return original_preflight(*args)

        bridge._digital_twin_robot_function_execution_preflight = _slow_preflight
        execution = asyncio.create_task(
            bridge.digital_twin_execute_robot_function(
                "dual robots",
                "ur5e",
                "move_home",
                confirmed=True,
            )
        )
        for _attempt in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("UR5e preflight worker did not start")

        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution
        assert bridge._ur5e_robot_function_execution_lock.acquire(blocking=False) is False
        assert bridge._ur5e_robot_function_execution_active == "move_home"

        finish.set()
        for _attempt in range(100):
            await asyncio.sleep(0.01)
            if bridge._ur5e_robot_function_execution_lock.acquire(blocking=False):
                bridge._ur5e_robot_function_execution_lock.release()
                break
        else:
            pytest.fail("UR5e lock was not released after preflight completed")
        assert bridge._ur5e_robot_function_execution_active is None
        assert agent.calls == []

    asyncio.run(_exercise())


def test_robot_function_preflights_are_serialized() -> None:
    async def _exercise() -> None:
        bridge = _ready_bridge(_agent_for("move_home"))
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        call_count = 0
        active_count = 0
        max_active = 0
        count_lock = threading.Lock()

        def _preflight(*_args: Any) -> tuple[Any | None, dict[str, Any], dict[str, Any], str]:
            nonlocal call_count, active_count, max_active
            with count_lock:
                call_count += 1
                call_number = call_count
                active_count += 1
                max_active = max(max_active, active_count)
            if call_number == 1:
                first_started.set()
                assert release_first.wait(timeout=2.0)
            else:
                second_started.set()
            with count_lock:
                active_count -= 1
            return None, {}, {}, "expected test stop"

        bridge._digital_twin_robot_function_execution_preflight = _preflight
        first = asyncio.create_task(
            bridge.digital_twin_robot_function_execution_readiness(
                "dual robots", "ur5e", "move_home"
            )
        )
        for _attempt in range(100):
            if first_started.is_set():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("first UR5e preflight did not start")
        second = asyncio.create_task(
            bridge.digital_twin_robot_function_execution_readiness(
                "dual robots", "ur5e", "move_home"
            )
        )
        await asyncio.sleep(0.05)
        assert second_started.is_set() is False
        release_first.set()
        await asyncio.gather(first, second)
        assert second_started.is_set() is True
        assert max_active == 1

    asyncio.run(_exercise())


def test_robot_function_preflight_has_a_bounded_wait() -> None:
    bridge = _ready_bridge(_agent_for("move_home"))
    bridge._ROBOT_FUNCTION_PREFLIGHT_TIMEOUT_S = 0.05
    finish = threading.Event()

    def _hung_preflight(
        *_args: Any,
    ) -> tuple[Any | None, dict[str, Any], dict[str, Any], str]:
        assert finish.wait(timeout=2.0)
        return None, {}, {}, "expected test stop"

    bridge._digital_twin_robot_function_execution_preflight = _hung_preflight
    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness("dual robots", "ur5e", "move_home")
    )
    finish.set()

    assert result["ready"] is False
    assert "readiness timed out after 0.05s" in result["message"]


@pytest.mark.parametrize(
    ("function_name", "location_kwargs"),
    [
        (
            "pick_grasp",
            {"origin_resource_location": "prusa-mk4-2", "part_name": "MG"},
        ),
        (
            "place_insert",
            {"destination_location": "assembly_board-v1", "part_name": "MG"},
        ),
    ],
)
def test_relative_lift_functions_require_finite_positive_return_height(
    function_name: str,
    location_kwargs: dict[str, str],
) -> None:
    agent = _agent_for(function_name)
    agent._task_ctx.pop("travel_z")
    bridge = _ready_bridge(agent)

    missing = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            function_name,
            **location_kwargs,
        )
    )
    agent._task_ctx["travel_z"] = agent._position["z"]
    nonpositive = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            function_name,
            **location_kwargs,
        )
    )

    assert missing["ready"] is False
    assert "task_ctx.travel_z and position.z" in missing["message"]
    assert nonpositive["ready"] is False
    assert "positive lift" in nonpositive["message"]
    assert agent.calls == []


def test_part_options_cover_all_four_exact_part_functions() -> None:
    bridge = object.__new__(SystemBridge)
    bridge.product_geometry_slots_for_product = lambda: ["MG", "SG"]

    for function_name in (
        "pick_approach",
        "pick_grasp",
        "place_approach",
        "place_insert",
    ):
        assert bridge.digital_twin_function_part_options(function_name) == ["MG", "SG"]
    assert bridge.digital_twin_function_part_options("move_home") == []


def test_function_held_part_reads_exact_selected_robot_agent_state() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._physical_ur5e_robot_agent = lambda: SimpleNamespace(_held_part="MCP")
    bridge._physical_xarm6_robot_agent = lambda: SimpleNamespace(_held_part="LG")

    assert bridge.digital_twin_function_held_part("ur5e") == "MCP"
    assert bridge.digital_twin_function_held_part("xarm6") == "LG"
    assert bridge.digital_twin_function_held_part("UR5E") == ""


def test_place_approach_first_run_requires_picked_with_valid_handoff() -> None:
    agent = _agent_for("place_approach")
    bridge = _ready_bridge(agent)

    ready = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    assert ready["ready"] is True
    assert ready["place_approach_start_state"] == "picked"
    assert ready["place_approach_repeat"] is False
    assert ready["held_part_handoff_ready"] is True

    agent._task_ctx.pop("held_part_handoff")
    blocked = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    assert blocked["ready"] is False
    assert "blocked before assembly_board-v1 staging" in blocked["message"]
    assert "held_part_handoff is missing" in blocked["message"]
    assert agent.calls == []


def test_place_approach_remains_ready_without_move_insert_hard_caps() -> None:
    agent = _agent_for("place_approach")
    bridge = _ready_bridge(agent)
    bridge._move_insert_preinsert_hard_caps_readiness = lambda _part_name="": (
        {
            "move_insert_hard_caps": {},
            "move_insert_live_hard_caps": {},
        },
        "move_insert hard caps are not commissioned",
    )

    readiness = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert readiness["ready"] is True
    assert readiness["move_insert_hard_caps"] == {}
    assert readiness["move_insert_hard_caps_error"] == (
        "move_insert hard caps are not commissioned"
    )
    assert agent.calls == []


def test_repeat_place_approach_allows_exact_preinsert_despite_generic_uncertainty() -> None:
    agent = _agent_for("place_approach")
    agent._current_state = "positioned"
    agent._task_ctx["destination_location"] = "assembly_board-v1"
    bridge = _ready_bridge(agent)
    bridge._ur5e_robot_function_state_uncertain = True
    capture_snapshot = bridge._robot_function_capture_snapshot
    bridge._robot_function_capture_snapshot = lambda target, robot: {
        **capture_snapshot(target, robot),
        "tf_stamp_sec": time.time() + 0.4,
    }

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert result["place_approach_repeat"] is True
    assert result["place_approach_pre_staging_ready"] is True
    assert result["manual_pose_translation_error_m"] == pytest.approx(0.0)
    assert -0.5 <= result["place_approach_tf_age_sec"] <= 0.0
    assert "move_home" not in result["message"]
    assert agent.calls == []


def test_repeat_place_approach_uses_protected_move_insert_start_tolerance() -> None:
    agent = _agent_for("place_approach")
    agent._current_state = "positioned"
    agent._task_ctx["destination_location"] = "assembly_board-v1"
    bridge = _ready_bridge(agent)
    current = {
        **deepcopy(MANUAL_DESCEND_POSE),
        "z": MANUAL_DESCEND_POSE["z"] + 0.00236,
        "frame_id": "world",
        "child_frame_id": "tool0",
    }
    bridge._robot_function_capture_snapshot = lambda _target, _robot: {
        "success": True,
        "world_tool0_ready": True,
        "tf_stamp_sec": time.time(),
        "blocked_reason": "",
        "waypoint": {"source": "hardware", "pose": deepcopy(current)},
    }

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert result["place_approach_pre_staging_ready"] is True
    assert result["manual_pose_translation_error_m"] == pytest.approx(0.00236)
    assert result["manual_pose_position_tolerance_m"] == pytest.approx(0.003)
    assert agent.calls == []


def test_repeat_place_approach_allows_exact_named_staging_joints() -> None:
    agent = _agent_for("place_approach")
    agent._current_state = "positioned"
    agent._task_ctx["destination_location"] = "assembly_board-v1"
    bridge = _ready_bridge(agent)
    actual_positions = [
        expected + delta
        for expected, delta in zip(
            agent.named_positions["assembly_board-v1"],
            (1e-5, -2e-5, 3e-6, -4e-6, 2e-6, 1.1e-5),
            strict=True,
        )
    ]
    bridge._digital_twin_ur5e_motion_readiness = (
        lambda _target, _cfg, _agent=None: (
            {
                "hardware_domain_id": 42,
                "trajectory_action_ready": True,
                "rtde_receive_connected": True,
                "joint_states_fresh": True,
                "rtde_control_connected": True,
                "cartesian_position_tolerance_m": 0.002,
                "cartesian_orientation_tolerance_rad": 0.05,
                "joint_goal_tolerance_rad": 0.025,
                "actual_positions_rad": actual_positions,
            },
            "",
        )
    )
    bridge._robot_function_capture_snapshot = lambda _target, _robot: {
        "success": True,
        "world_tool0_ready": True,
        "tf_stamp_sec": time.time(),
        "blocked_reason": "",
        "waypoint": {
            "source": "hardware",
            "pose": {
                "x": MANUAL_DESCEND_POSE["x"] + 0.149,
                "y": MANUAL_DESCEND_POSE["y"],
                "z": MANUAL_DESCEND_POSE["z"],
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
                "frame_id": "world",
                "child_frame_id": "tool0",
            },
        },
    }

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert result["place_approach_repeat"] is True
    assert result["place_approach_repeat_start"] == "assembly_board-v1"
    assert result["place_approach_pre_staging_ready"] is True
    assert result["place_approach_named_staging_max_joint_error_rad"] == (
        pytest.approx(2e-5)
    )
    assert result["place_approach_named_staging_joint_tolerance_rad"] == (
        pytest.approx(0.025)
    )
    assert "manual_pose_translation_error_m" not in result
    assert agent.calls == []


def test_repeat_place_approach_named_staging_is_rechecked_under_lock() -> None:
    agent = _agent_for("place_approach")
    agent._current_state = "positioned"
    agent._task_ctx["destination_location"] = "assembly_board-v1"
    bridge = _ready_bridge(agent)
    motion_readiness_calls = 0

    def _motion_readiness(
        _target: str,
        _cfg: dict[str, Any],
        _agent: Any = None,
    ) -> tuple[dict[str, Any], str]:
        nonlocal motion_readiness_calls
        motion_readiness_calls += 1
        return {
            "hardware_domain_id": 42,
            "trajectory_action_ready": True,
            "rtde_receive_connected": True,
            "joint_states_fresh": True,
            "rtde_control_connected": True,
            "cartesian_position_tolerance_m": 0.002,
            "cartesian_orientation_tolerance_rad": 0.05,
            "joint_goal_tolerance_rad": 0.025,
            "actual_positions_rad": list(
                agent.named_positions["assembly_board-v1"]
            ),
        }, ""

    bridge._digital_twin_ur5e_motion_readiness = _motion_readiness

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert motion_readiness_calls == 2
    assert [name for name, _kwargs in agent.calls] == ["place_approach"]


def test_repeat_place_approach_blocks_off_preinsert_before_staging() -> None:
    agent = _agent_for("place_approach")
    agent._current_state = "positioned"
    agent._task_ctx["destination_location"] = "assembly_board-v1"
    agent._task_ctx["move_insert_result"] = {
        "engagement_detected": True,
        "seated_detected": True,
        "state_uncertain": False,
    }
    bridge = _ready_bridge(agent)
    moved_pose = {
        **deepcopy(MANUAL_DESCEND_POSE),
        "z": MANUAL_DESCEND_POSE["z"] - 0.01,
    }
    bridge._robot_function_capture_snapshot = lambda _target, _robot: {
        "success": True,
        "world_tool0_ready": True,
        "tf_stamp_sec": time.time(),
        "blocked_reason": "",
        "waypoint": {
            "source": "hardware",
            "pose": {
                **moved_pose,
                "frame_id": "world",
                "child_frame_id": "tool0",
            },
        },
    }

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "blocked before assembly_board-v1 staging" in result["message"]
    assert "may still be seated or bound" in result["message"]
    assert "Cartesian Step or Smooth Hold" in result["message"]
    assert "reverse insertion_axis_world" in result["message"]
    assert "do not run place_approach or named-position motion" in result["message"]
    assert "pre-insertion pose" in result["message"]
    assert result["manual_pose_translation_error_m"] == pytest.approx(0.01)
    assert agent.calls == []


def test_repeat_place_approach_allows_clear_positioned_pose_after_alignment_edit() -> None:
    agent = _agent_for("place_approach")
    agent._current_state = "positioned"
    agent._task_ctx["destination_location"] = "assembly_board-v1"
    agent._task_ctx.pop("move_insert_result", None)
    agent._task_ctx.pop("move_insert_trial_result", None)
    bridge = _ready_bridge(agent)
    moved_pose = {
        **deepcopy(MANUAL_DESCEND_POSE),
        "x": MANUAL_DESCEND_POSE["x"] + 0.00714,
    }
    bridge._robot_function_capture_snapshot = lambda _target, _robot: {
        "success": True,
        "world_tool0_ready": True,
        "tf_stamp_sec": time.time(),
        "blocked_reason": "",
        "waypoint": {
            "source": "hardware",
            "pose": {
                **moved_pose,
                "frame_id": "world",
                "child_frame_id": "tool0",
            },
        },
    }

    readiness = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert readiness["ready"] is True
    assert readiness["place_approach_repeat"] is True
    assert readiness["place_approach_repeat_start"] == (
        "current_clear_positioned_pose"
    )
    assert readiness["place_approach_pre_staging_ready"] is True
    assert readiness["place_approach_repeat_pose_offset_accepted"] is True
    assert readiness["manual_pose_translation_error_m"] == pytest.approx(0.00714)
    assert agent.calls == []

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert [name for name, _kwargs in agent.calls] == ["place_approach"]


def test_repeat_place_approach_allows_settled_unengaged_move_insert_retry() -> None:
    agent = _agent_for("place_approach")
    agent._current_state = "positioned"
    agent._task_ctx["destination_location"] = "assembly_board-v1"
    agent._task_ctx["move_insert_result"] = {
        "trial_id": "move-insert-settled-failure",
        "success": False,
        "motion_settled": True,
        "state_uncertain": False,
        "final_tool0_pose_valid": True,
        "engagement_detected": False,
        "seated_detected": False,
        "hard_limit_detected": True,
    }
    bridge = _ready_bridge(agent)
    bridge._robot_function_capture_snapshot = lambda _target, _robot: {
        "success": True,
        "world_tool0_ready": True,
        "tf_stamp_sec": time.time(),
        "blocked_reason": "",
        "waypoint": {
            "source": "hardware",
            "pose": {
                **deepcopy(MANUAL_DESCEND_POSE),
                "x": MANUAL_DESCEND_POSE["x"] + 0.012,
                "z": MANUAL_DESCEND_POSE["z"] - 0.035,
                "frame_id": "world",
                "child_frame_id": "tool0",
            },
        },
    }

    readiness = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert readiness["ready"] is True
    assert readiness["place_approach_repeat"] is True
    assert readiness["place_approach_repeat_start"] == "assembly_board-v1"
    assert readiness["place_approach_pre_staging_ready"] is True
    assert readiness["move_insert_retry_motion_settled"] is True
    assert agent.calls == []

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert [name for name, _kwargs in agent.calls] == ["place_approach"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("motion_settled", False),
        ("state_uncertain", True),
        ("final_tool0_pose_valid", False),
        ("engagement_detected", True),
        ("seated_detected", True),
    ],
)
def test_repeat_place_approach_does_not_relax_nonretry_move_insert_state(
    field: str,
    value: bool,
) -> None:
    agent = _agent_for("place_approach")
    agent._current_state = "positioned"
    agent._task_ctx["destination_location"] = "assembly_board-v1"
    agent._task_ctx["move_insert_result"] = {
        "trial_id": "move-insert-not-retryable",
        "success": False,
        "motion_settled": True,
        "state_uncertain": False,
        "final_tool0_pose_valid": True,
        "engagement_detected": False,
        "seated_detected": False,
        field: value,
    }
    bridge = _ready_bridge(agent)
    bridge._robot_function_capture_snapshot = lambda _target, _robot: {
        "success": True,
        "world_tool0_ready": True,
        "tf_stamp_sec": time.time(),
        "blocked_reason": "",
        "waypoint": {
            "source": "hardware",
            "pose": {
                **deepcopy(MANUAL_DESCEND_POSE),
                "z": MANUAL_DESCEND_POSE["z"] - 0.035,
                "frame_id": "world",
                "child_frame_id": "tool0",
            },
        },
    }

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert result.get("move_insert_retry_motion_settled") is not True
    assert agent.calls == []


def test_repeat_place_approach_blocks_off_named_staging_and_off_preinsert() -> None:
    agent = _agent_for("place_approach")
    agent._current_state = "positioned"
    agent._task_ctx["destination_location"] = "assembly_board-v1"
    agent._task_ctx["move_insert_result"] = {
        "engagement_detected": True,
        "seated_detected": False,
        "state_uncertain": False,
    }
    bridge = _ready_bridge(agent)
    bridge._digital_twin_ur5e_motion_readiness = (
        lambda _target, _cfg, _agent=None: (
            {
                "hardware_domain_id": 42,
                "trajectory_action_ready": True,
                "rtde_receive_connected": True,
                "joint_states_fresh": True,
                "rtde_control_connected": True,
                "cartesian_position_tolerance_m": 0.002,
                "cartesian_orientation_tolerance_rad": 0.05,
                "joint_goal_tolerance_rad": 0.025,
                "actual_positions_rad": [
                    value + (0.05 if index == 0 else 0.0)
                    for index, value in enumerate(
                        agent.named_positions["assembly_board-v1"]
                    )
                ],
            },
            "",
        )
    )
    bridge._robot_function_capture_snapshot = lambda _target, _robot: {
        "success": True,
        "world_tool0_ready": True,
        "tf_stamp_sec": time.time(),
        "blocked_reason": "",
        "waypoint": {
            "source": "hardware",
            "pose": {
                **deepcopy(MANUAL_DESCEND_POSE),
                "z": MANUAL_DESCEND_POSE["z"] + 0.01,
                "frame_id": "world",
                "child_frame_id": "tool0",
            },
        },
    }

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "exact assembly_board-v1 named staging joints" in result["message"]
    assert result["place_approach_named_staging_max_joint_error_rad"] == (
        pytest.approx(0.05)
    )
    assert result["manual_pose_translation_error_m"] == pytest.approx(0.01)
    assert agent.calls == []


def test_repeat_place_approach_blocks_unknown_move_insert_acceptance() -> None:
    agent = _agent_for("place_approach")
    agent._current_state = "positioned"
    agent._task_ctx["destination_location"] = "assembly_board-v1"
    agent._task_ctx["move_insert_trial_result"] = {"state_uncertain": True}
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "move_insert acceptance is unknown" in result["message"]
    assert "before assembly_board-v1 staging" in result["message"]
    assert agent.calls == []


@pytest.mark.parametrize(
    ("tf_stamp_sec", "message_fragment"),
    [
        (None, "has no source header timestamp"),
        ("stale", "world -> tool0 TF is not fresh"),
    ],
)
def test_repeat_place_approach_requires_fresh_source_tf_timestamp(
    tf_stamp_sec: float | str | None,
    message_fragment: str,
) -> None:
    agent = _agent_for("place_approach")
    agent._current_state = "positioned"
    agent._task_ctx["destination_location"] = "assembly_board-v1"
    bridge = _ready_bridge(agent)

    def _snapshot(_target: str, _robot: str) -> dict[str, Any]:
        snapshot = {
            "success": True,
            "world_tool0_ready": True,
            "blocked_reason": "",
            "waypoint": {
                "source": "hardware",
                "pose": {
                    **deepcopy(MANUAL_DESCEND_POSE),
                    "frame_id": "world",
                    "child_frame_id": "tool0",
                },
            },
        }
        if tf_stamp_sec is not None:
            snapshot["tf_stamp_sec"] = (
                time.time() - 3.1 if tf_stamp_sec == "stale" else tf_stamp_sec
            )
        return snapshot

    bridge._robot_function_capture_snapshot = _snapshot
    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "blocked before assembly_board-v1 staging" in result["message"]
    assert message_fragment in result["message"]
    assert agent.calls == []


@pytest.mark.parametrize(
    ("blocker_name", "blocker_message"),
    [
        (
            "_insertion_demonstration_blocking_error",
            "Insertion demonstration for MG is active.",
        ),
        (
            "_move_insert_pending_review_error",
            "Supervised Test move_insert for MG awaits review.",
        ),
    ],
)
def test_place_approach_blocks_insertion_custody_before_staging(
    blocker_name: str,
    blocker_message: str,
) -> None:
    agent = _agent_for("place_approach")
    bridge = _ready_bridge(agent)
    setattr(bridge, blocker_name, lambda **_kwargs: blocker_message)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert result["message"] == blocker_message
    assert agent.calls == []


def test_place_approach_pre_execute_rechecks_repeat_pose_before_dispatch() -> None:
    agent = _agent_for("place_approach")
    agent._current_state = "positioned"
    agent._task_ctx["destination_location"] = "assembly_board-v1"
    bridge = _ready_bridge(agent)
    original_gate = bridge._physical_ur5e_place_approach_pre_staging_error
    gate_calls = 0

    def _gate(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], str]:
        nonlocal gate_calls
        gate_calls += 1
        if gate_calls == 2:
            return {}, (
                "Repeat place_approach is blocked before assembly_board-v1 staging: "
                "fresh pre-insertion pose changed immediately before dispatch."
            )
        return original_gate(*args, **kwargs)

    bridge._physical_ur5e_place_approach_pre_staging_error = _gate
    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["result"]["manual_pre_execute_blocked"] is True
    assert "immediately before dispatch" in result["message"]
    assert gate_calls == 2
    assert agent.calls == []


def test_non_assembly_place_approach_preserves_existing_preflight() -> None:
    agent = _agent_for("place_approach")
    agent._current_state = "idle"
    bridge = _ready_bridge(agent)
    bridge._physical_ur5e_place_approach_pre_staging_error = (
        lambda *_args, **_kwargs: pytest.fail(
            "non-assembly place_approach must not use the insertion pre-staging gate"
        )
    )

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert "place_approach_pre_staging_ready" not in result
    assert agent.calls == []


@pytest.mark.parametrize("held_part", [None, ""])
def test_place_approach_allows_independent_empty_held_part(
    held_part: str | None,
) -> None:
    agent = _PhysicalUR5eAgent(
        state="idle",
        held_part=held_part,
        gripper_state="open",
    )
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert result["independent_commissioning"] is True
    assert "held_part empty" in result["message"]
    assert "tests the freshly computed target" in result["message"]
    assert "optional confirmed robot correction without a part" in result["message"]


def test_place_approach_readiness_builds_exact_operator_confirmed_mg_handoff() -> None:
    agent = _PhysicalUR5eAgent(
        state="idle",
        held_part=None,
        gripper_state="open",
    )
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            operator_confirmed_held_part=True,
        )
    )

    assert result["ready"] is True
    assert result["independent_commissioning"] is False
    assert result["operator_confirmed_held_part"] is True
    assert result["operator_held_part"] == "MG"
    assert result["held_part_handoff_ready"] is True
    assert result["move_insert_trial_context_ready"] is False
    assert result["move_insert_authorized"] is False
    assert "gripper_action_ready" not in result
    assert "confirmed pick_approach.descend handoff" in result["message"]
    assert "no RG2 command-position or load-sensing evidence" in result["message"]
    assert "Supervised Test move_insert" in result["message"]
    assert agent._held_part is None
    assert agent.calls == []


@pytest.mark.parametrize(
    ("part_name", "model_name"),
    (
        ("SG", "gear_small"),
        ("MG", "gear_medium"),
        ("LG", "gear_large"),
        ("SCP", "circ_pin_small"),
        ("MCP", "circ_pin_medium"),
        ("LCP", "circ_pin_large"),
    ),
)
def test_place_approach_executes_each_exact_operator_confirmed_part(
    part_name: str,
    model_name: str,
) -> None:
    agent = _PhysicalUR5eAgent(
        state="idle",
        held_part=None,
        gripper_state="open",
    )
    bridge = _ready_bridge(agent)
    bridge._physical_ur5e_place_approach_pre_staging_error = (
        lambda *_args, **_kwargs: pytest.fail(
            "operator-confirmed handoff is adopted inside the runtime"
        )
    )

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_approach",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name=part_name,
            confirmed=True,
            operator_confirmed_held_part=True,
        )
    )

    assert result["success"] is True
    assert result["operator_held_part"] == part_name
    assert result["held_part_handoff_adopted"] is True
    assert agent._held_part == part_name
    assert agent._task_ctx["part_name"] == part_name
    assert agent._task_ctx["model_name"] == model_name
    assert agent._task_ctx["held_part_handoff"]["part_name"] == part_name
    assert agent._task_ctx["held_part_handoff"]["model_name"] == model_name


def test_place_approach_executes_operator_confirmed_mg_and_retains_custody() -> None:
    agent = _PhysicalUR5eAgent(
        state="idle",
        held_part=None,
        gripper_state="open",
    )
    bridge = _ready_bridge(agent)
    bridge._physical_ur5e_place_approach_pre_staging_error = (
        lambda *_args, **_kwargs: pytest.fail(
            "operator-confirmed handoff is adopted inside the runtime"
        )
    )

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_approach",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
            operator_confirmed_held_part=True,
        )
    )

    assert result["success"] is True
    assert result["operator_confirmed_held_part"] is True
    assert result["operator_held_part"] == "MG"
    assert result["held_part_handoff_adopted"] is True
    assert result["move_insert_trial_context_ready"] is False
    assert result["move_insert_authorized"] is False
    assert result["result"]["move_insert_authorized"] is False
    assert agent.operator_confirmed_held_part_calls == [True]
    assert (
        agent.operator_confirmed_held_part_handoff_calls[0]["source"]
        == "operator_confirmed_pick_approach_recording"
    )
    assert [call[0] for call in agent.calls] == ["place_approach"]
    assert agent._current_state == "positioned"
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"
    assert agent._task_ctx["destination_location"] == "assembly_board-v1"
    assert agent._task_ctx["held_part_handoff"]["part_name"] == "MG"


def test_operator_confirmed_place_approach_does_not_require_move_insert_profile() -> None:
    agent = _PhysicalUR5eAgent(
        state="idle",
        held_part=None,
        gripper_state="open",
    )
    bridge = _ready_bridge(agent)
    bridge.digital_twin_move_insert_settings = lambda *_args, **_kwargs: pytest.fail(
        "standalone place_approach must not read move_insert settings"
    )
    bridge._prepare_ur5e_move_insert_client = lambda _agent: pytest.fail(
        "standalone place_approach must not prepare move_insert"
    )

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            operator_confirmed_held_part=True,
        )
    )

    assert result["ready"] is True
    assert result["move_insert_trial_context_ready"] is False
    assert "calibration_id is missing" not in result["message"]
    assert agent.calls == []


def test_assembly_place_approach_still_requires_move_insert_profile() -> None:
    agent = _PhysicalUR5eAgent(
        state="picked",
        held_part="MG",
        gripper_state="closed",
    )
    bridge = _ready_bridge(agent)
    incomplete = {
        "validated": False,
        "profile_state": "incomplete",
        "profile_sha256": "",
        "effective": {},
        "hard_caps": {},
        "message": (
            "controller.parts_tuning.move_insert.calibration_id is missing "
            "or is not exact"
        ),
    }
    settings_calls: list[str] = []

    def _settings(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        settings_calls.append("settings")
        return deepcopy(incomplete)

    bridge.digital_twin_move_insert_settings = _settings
    bridge._digital_twin_move_insert_trial_settings = (  # type: ignore[method-assign]
        lambda _part_name: deepcopy(incomplete)
    )
    token = bridge_module._MOVE_INSERT_PREFLIGHT_REQUIRED_CONTEXT.set(True)
    try:
        _readiness, error = bridge._digital_twin_robot_function_move_insert_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            "assembly_board-v1",
            "MG",
            agent,
            {},
        )
    finally:
        bridge_module._MOVE_INSERT_PREFLIGHT_REQUIRED_CONTEXT.reset(token)

    assert settings_calls == ["settings"]
    assert error == (
        "controller.parts_tuning.move_insert.calibration_id is missing or is not exact"
    )


def test_operator_confirmed_place_approach_fails_if_runtime_drops_custody() -> None:
    agent = _PhysicalUR5eAgent()

    async def _drops_custody(
        _function_name: str,
        _pre_execute: Any = None,
        _post_staging_acceptance: Any = None,
        /,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {"status": "completed", "content": "Moved without custody."}

    agent._execute_registered_robot_task_for_manual_function_execution = (  # type: ignore[method-assign]
        _drops_custody
    )
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_approach",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
            operator_confirmed_held_part=True,
        )
    )

    assert result["success"] is False
    assert result["held_part_handoff_adopted"] is False
    assert result["move_insert_trial_context_ready"] is False
    assert "did not retain state positioned" in result["message"]
    assert "do not run place_insert" in result["message"]
    assert bridge._ur5e_robot_function_state_uncertain is True


def test_operator_confirmed_mg_handoff_uses_confirmed_recording_and_operator_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_bridge(agent)
    _use_real_operator_handoff_builder(
        bridge,
        agent,
        monkeypatch,
        tmp_path,
    )

    handoff, readiness, error = bridge._operator_confirmed_mg_held_part_handoff(
        "dual robots",
        agent,
        origin_resource_location="prusa-mk4-2",
        part_name="MG",
        product_geometry=_mg_product_geometry(),
    )

    assert error == ""
    assert readiness["held_part_handoff_ready"] is True
    assert readiness["move_insert_trial_context_ready"] is False
    assert readiness["move_insert_authorized"] is False
    assert handoff["source"] == "operator_confirmed_pick_approach_recording"
    assert handoff["orientation_source"] == "realsense_roboflow_identity"
    assert "gripper_command_position" not in handoff
    assert "gripper_status_updated_at" not in handoff
    assert set(handoff["origin_pose_provenance"]) == {
        "frame_id",
        "part_name",
        "model_name",
        "source",
        "orientation_source",
        "captured_at",
        "pick_approach_recording_sha256",
    }
    validated, validation_error = (
        robot_task_runtime._validated_operator_confirmed_held_part_handoff(
            handoff
        )
    )
    assert validation_error == ""
    assert validated["tool0_to_held_part"] == pytest.approx(
        handoff["tool0_to_held_part"]
    )
    assert agent.calls == []


@pytest.mark.parametrize(
    ("part_name", "model_name"),
    (
        ("SG", "gear_small"),
        ("MG", "gear_medium"),
        ("LG", "gear_large"),
        ("SCP", "circ_pin_small"),
        ("MCP", "circ_pin_medium"),
        ("LCP", "circ_pin_large"),
    ),
)
def test_operator_confirmed_handoff_uses_each_exact_supported_part_and_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    part_name: str,
    model_name: str,
) -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_bridge(agent)
    _use_real_operator_handoff_builder(
        bridge,
        agent,
        monkeypatch,
        tmp_path,
        part_name=part_name,
    )
    product_geometry = {
        **_mg_product_geometry(part_name),
        "model_name": model_name,
    }

    handoff, readiness, error = bridge._operator_confirmed_mg_held_part_handoff(
        "dual robots",
        agent,
        origin_resource_location="prusa-mk4-2",
        part_name=part_name,
        product_geometry=product_geometry,
    )

    assert error == ""
    assert readiness["operator_held_part"] == part_name
    assert readiness["held_part_handoff_ready"] is True
    assert handoff["part_name"] == part_name
    assert handoff["model_name"] == model_name
    assert handoff["origin_pose_provenance"]["part_name"] == part_name
    assert handoff["origin_pose_provenance"]["model_name"] == model_name


@pytest.mark.parametrize(
    ("part_name", "model_name"),
    (
        ("SG", "gear_small"),
        ("MG", "gear_medium"),
        ("LG", "gear_large"),
        ("SCP", "circ_pin_small"),
        ("MCP", "circ_pin_medium"),
        ("LCP", "circ_pin_large"),
    ),
)
def test_capture_pose_uses_each_exact_operator_confirmed_part_and_model(
    part_name: str,
    model_name: str,
) -> None:
    agent = _PhysicalUR5eAgent(
        state="idle",
        held_part=None,
        gripper_state="open",
    )
    bridge = _ready_bridge(agent)
    bridge._robot_function_validate_request = (
        lambda _target, _robot, _function_name: ({"robot": "ur5e"}, "")
    )
    bridge._robot_function_part_name_error = (
        lambda _function_name, _part_name: ""
    )
    bridge._robot_function_template_step = (
        lambda _function_name, _step_name: {
            "primitive": "move_cartesian",
            "recordable": True,
        }
    )
    captured_task_contexts: list[dict[str, Any]] = []

    def _resolve(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        captured_task_contexts.append(deepcopy(agent._task_ctx))
        return {
            "success": True,
            "computed_position": deepcopy(MANUAL_DESCEND_POSE),
            "computed_pose": deepcopy(MANUAL_DESCEND_POSE),
            "computed_reference": {
                "kind": "assembly_slot",
                "name": part_name,
                "frame_id": "world",
            },
            "computed_source": "capture",
            "computed_at": time.time(),
        }

    bridge._resolve_robot_function_position = _resolve
    bridge._digital_twin_record_lock = threading.RLock()
    bridge._digital_twin_function_steps = {}
    bridge.digital_twin_function_template = lambda _function_name: [
        {"step_name": "descend"}
    ]

    result = bridge.digital_twin_capture_function_step(
        "dual robots",
        "ur5e",
        "place_approach",
        "assembly_board-v1",
        "descend",
        part_name=part_name,
        operator_confirmed_held_part=True,
        operator_handoff_origin_resource_location="prusa-mk4-2",
    )

    assert result["success"] is True
    assert result["readiness"]["operator_held_part"] == part_name
    assert captured_task_contexts[0]["part_name"] == part_name
    assert captured_task_contexts[0]["model_name"] == model_name
    assert captured_task_contexts[0]["held_part_handoff"]["part_name"] == part_name
    assert captured_task_contexts[0]["held_part_handoff"]["model_name"] == model_name
    assert agent._task_ctx == {}


def test_operator_confirmed_mg_handoff_rejects_unconfirmed_pick_recording(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_bridge(agent)
    _use_real_operator_handoff_builder(
        bridge,
        agent,
        monkeypatch,
        tmp_path,
        confirmed=False,
    )

    handoff, readiness, error = bridge._operator_confirmed_mg_held_part_handoff(
        "dual robots",
        agent,
        origin_resource_location="prusa-mk4-2",
        part_name="MG",
        product_geometry=_mg_product_geometry(),
    )

    assert handoff == {}
    assert readiness["held_part_handoff_ready"] is False
    assert "requires confirmed pick_approach.descend" in error
    assert agent.calls == []


def test_operator_confirmed_mg_handoff_does_not_read_rg2_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_bridge(agent)
    _use_real_operator_handoff_builder(
        bridge,
        agent,
        monkeypatch,
        tmp_path,
    )
    bridge._ur5e_rg2_gripper_status = lambda: pytest.fail(
        "operator confirmation must not read RG2 status"
    )

    handoff, readiness, error = bridge._operator_confirmed_mg_held_part_handoff(
        "dual robots",
        agent,
        origin_resource_location="prusa-mk4-2",
        part_name="MG",
        product_geometry=_mg_product_geometry(),
    )

    assert error == ""
    assert readiness["held_part_handoff_ready"] is True
    assert readiness["operator_held_part_clamped_by_operator_confirmation"] is True
    assert handoff["part_name"] == "MG"
    assert agent.calls == []


def test_operator_confirmed_mg_handoff_does_not_require_gripper_action_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_bridge(agent)
    _use_real_operator_handoff_builder(
        bridge,
        agent,
        monkeypatch,
        tmp_path,
    )
    bridge._digital_twin_ur5e_gripper_readiness = lambda *_args: pytest.fail(
        "operator-confirmed place_approach must not require gripper action readiness"
    )

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            operator_confirmed_held_part=True,
        )
    )

    assert result["ready"] is True
    assert result["operator_held_part_clamped_by_operator_confirmation"] is True
    assert "gripper_action_ready" not in result
    assert agent.calls == []


def test_operator_confirmed_mg_handoff_rejects_stale_current_tf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_bridge(agent)
    _use_real_operator_handoff_builder(
        bridge,
        agent,
        monkeypatch,
        tmp_path,
        tf_age_sec=9.0,
    )

    handoff, readiness, error = bridge._operator_confirmed_mg_held_part_handoff(
        "dual robots",
        agent,
        origin_resource_location="prusa-mk4-2",
        part_name="MG",
        product_geometry=_mg_product_geometry(),
    )

    assert handoff == {}
    assert readiness["held_part_handoff_ready"] is False
    assert "requires fresh world -> tool0 TF evidence" in error
    assert agent.calls == []


@pytest.mark.parametrize(
    ("function_name", "destination_location", "part_name"),
    [
        ("pick_approach", "", "MG"),
        ("place_insert", "assembly_board-v1", "MG"),
        ("place_approach", "assembly_board-v1", "SRP"),
    ],
)
def test_operator_confirmed_held_part_is_exactly_scoped_to_ur5e_mg_place_approach(
    function_name: str,
    destination_location: str,
    part_name: str,
) -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            function_name,
            origin_resource_location=(
                "prusa-mk4-2" if function_name == "pick_approach" else ""
            ),
            destination_location=destination_location,
            part_name=part_name,
            operator_confirmed_held_part=True,
        )
    )

    assert result["ready"] is False
    assert "only for standalone ur5e place_approach" in result["message"]
    assert agent.calls == []


def test_operator_confirmed_held_part_rejects_existing_pick_context() -> None:
    agent = _PhysicalUR5eAgent(
        state="at_pick",
        held_part=None,
        gripper_state="open",
    )
    agent._task_ctx = {
        "part_name": "MG",
        "origin_resource_location": "prusa-mk4-2",
    }
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            operator_confirmed_held_part=True,
        )
    )

    assert result["ready"] is False
    assert "cannot replace an active pick context" in result["message"]
    assert agent.calls == []


def test_operator_confirmed_held_part_rejects_non_boolean_value() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
            operator_confirmed_held_part="true",  # type: ignore[arg-type]
        )
    )

    assert result["ready"] is False
    assert result["message"] == (
        "operator_confirmed_held_part must be an exact boolean."
    )
    assert agent.calls == []


def test_place_approach_requires_pick_grasp_when_pick_context_is_active() -> None:
    agent = _PhysicalUR5eAgent(
        state="at_pick",
        held_part=None,
        gripper_state="open",
    )
    agent._task_ctx = {
        "part_name": "MG",
        "origin_resource_location": "prusa-mk4-2",
    }
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "active pick context for part_name 'MG'" in result["message"]
    assert "ur5e held_part is empty" in result["message"]
    assert "Complete pick_grasp before place_approach" in result["message"]


def test_place_approach_reports_exact_held_part_selection() -> None:
    agent = _PhysicalUR5eAgent(
        state="picked",
        held_part="MCP",
        gripper_state="closed",
    )
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "selected part_name 'MG'" in result["message"]
    assert "ur5e held_part is 'MCP'" in result["message"]
    assert "Select part_name 'MCP'" in result["message"]


@pytest.mark.parametrize("held_part", [None, ""])
def test_place_insert_blocks_independent_empty_held_part_at_assembly_board(
    held_part: str | None,
) -> None:
    agent = _PhysicalUR5eAgent(
        state="idle",
        held_part=held_part,
        gripper_state="open",
    )
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "held part" in result["message"].lower()
    assert "pick_grasp" in result["message"]
    assert "place_approach" in result["message"]
    assert agent._controller.move_insert_prepare_timeouts == []


def test_non_independent_place_insert_readiness_prepares_optional_client() -> None:
    agent = _agent_for("place_insert")
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert result["move_insert_client_ready"] is True
    assert agent._controller.move_insert_prepare_timeouts == [2.0]
    assert agent._controller._ur5e_hardware_insert_client is not None


def test_non_independent_place_insert_blocks_before_dispatch_when_client_is_unavailable(
) -> None:
    agent = _agent_for("place_insert")
    bridge = _ready_bridge(agent)
    agent._controller._ensure_move_insert_client_ready = (
        lambda *, timeout_sec=2.0: (
            False,
            "/cais_ur5e_rtde_cartesian_controller/move_insert is unavailable",
        )
    )

    result = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["message"] == (
        "/cais_ur5e_rtde_cartesian_controller/move_insert is unavailable"
    )
    assert agent.calls == []


def test_place_insert_reports_exact_held_part_selection() -> None:
    agent = _agent_for("place_insert")
    agent._held_part = "MCP"
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "place_insert",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "selected part_name 'MG'" in result["message"]
    assert "ur5e held_part is 'MCP'" in result["message"]
    assert "Select part_name 'MCP'" in result["message"]


def test_dual_hardware_function_execution_uses_selected_robot_cartesian_readiness() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._selected_normal_hardware_stack = lambda: "dual robots"
    bridge._teleop_smooth_session = None
    bridge._hardware_cartesian_readiness = {
        "xarm6": {
            "cartesian_jog_ready": True,
            "cartesian_function_ready": True,
            "message": "xArm6 Cartesian frame validation ready",
        },
        "ur5e": {
            "cartesian_jog_ready": False,
            "cartesian_function_ready": False,
            "message": "world -> tool0 disagrees with live RTDE TCP",
        },
    }

    assert bridge._physical_robot_function_cartesian_error("xarm6") == ""
    assert bridge._physical_robot_function_cartesian_error("ur5e") == (
        "Cartesian frame validation failed: "
        "world -> tool0 disagrees with live RTDE TCP"
    )
    bridge._teleop_smooth_session = {"robot": "xarm6", "axis": "z"}
    assert bridge._physical_robot_function_cartesian_error("xarm6") == (
        "Release Cartesian Smooth Hold before Function Execution. "
        "Active robot=xarm6 axis=World Z."
    )


_ASSEMBLY_FUNCTIONS = (
    "pick_approach",
    "pick_grasp",
    "place_approach",
    "place_insert",
    "move_home",
)


@pytest.mark.parametrize(
    ("mutation", "message_fragment"),
    [
        ("registration", "assembly_board-v1_aruco_to_assembly_board-v1"),
        ("target_reference", "inserted_part_origin"),
        ("slot_xy", "assembly_slot geometry"),
    ],
)
def test_assembly_blocks_static_move_insert_geometry_before_confirmation_and_motion(
    mutation: str,
    message_fragment: str,
) -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    product_geometry = _mg_product_geometry()
    if mutation == "registration":
        product_geometry.pop("assembly_board-v1_aruco_to_assembly_board-v1")
    elif mutation == "target_reference":
        product_geometry["target_reference"]["target_point"] = "part_origin"
    else:
        product_geometry["slot_xy"] = []
    bridge._robot_function_product_geometry_for_part = (
        lambda _part_name: deepcopy(product_geometry)
    )
    arguments = {
        "origin_resource_location": "prusa-mk4-2",
        "destination_location": "assembly_board-v1",
        "part_name": "MG",
    }

    readiness = asyncio.run(
        bridge.digital_twin_assembly_readiness(
            "dual robots",
            "ur5e",
            **arguments,
        )
    )
    execution = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            **arguments,
            confirmed=True,
        )
    )

    assert readiness["ready"] is False
    assert message_fragment in readiness["message"]
    assert execution["success"] is False
    assert message_fragment in execution["message"]
    assert execution["completed_functions"] == []
    assert agent.calls == []


def test_assembly_requires_board_pose_for_static_geometry_before_confirmation() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    bridge.perception_manager.assembly_board_v1_aruco_status = lambda _role: {
        "accepted": True,
        "accepted_baseline_ready": True,
        "accepted_baseline_error": "",
        "accepted_generation": 1,
        "accepted_calibration_id": "ur5e-calibration",
    }

    readiness = asyncio.run(
        bridge.digital_twin_assembly_readiness(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert readiness["ready"] is False
    assert "accepted or current assembly_board-v1 ArUco pose" in readiness["message"]
    assert agent.calls == []


def test_assembly_requires_real_held_part_handoff_before_place_approach() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    original_pick_grasp = agent.pick_grasp

    async def _pick_grasp_without_handoff(**kwargs: Any) -> dict[str, Any]:
        result = await original_pick_grasp(**kwargs)
        agent._task_ctx.pop("held_part_handoff", None)
        return result

    agent.pick_grasp = _pick_grasp_without_handoff  # type: ignore[method-assign]

    result = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["failed_function"] == "place_approach"
    assert result["completed_functions"] == ["pick_approach", "pick_grasp"]
    assert "held_part_handoff is required after pick_grasp" in result["message"]
    assert [name for name, _kwargs in agent.calls] == [
        "pick_approach",
        "pick_grasp",
    ]


def test_assembly_rejects_incomplete_computed_pose_before_place_approach() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)

    def _geometry(**kwargs: Any) -> dict[str, Any]:
        if not kwargs["held_part_handoff"]:
            return {
                "success": False,
                "missing_held_part_handoff": True,
                "message": (
                    "move_insert learned geometry is valid; held_part_handoff is required "
                    "after pick_grasp to compute complete insertion poses"
                ),
            }
        return {
            "success": True,
            "part_name": "MG",
            "move_insert_profile_sha256": "a" * 64,
            "move_insert_profile": deepcopy(_move_insert_effective()),
            "approach_pose": deepcopy(MANUAL_DESCEND_POSE),
            "pre_insert_pose": {
                key: value for key, value in MANUAL_DESCEND_POSE.items() if key != "qw"
            },
            "insert_pose": deepcopy(MANUAL_DESCEND_POSE),
            "insertion_axis_world": {"x": 0.0, "y": 0.0, "z": -1.0},
            "move_insert_timeout_sec": 9.0,
        }

    bridge._move_insert_geometry_helper = lambda: _geometry

    result = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["failed_function"] == "place_approach"
    assert result["completed_functions"] == ["pick_approach", "pick_grasp"]
    assert "pre_insert_pose.qw must be finite" in result["message"]
    assert [name for name, _kwargs in agent.calls] == [
        "pick_approach",
        "pick_grasp",
    ]


def test_assembly_recheck_restores_recipe_derived_profile_fields() -> None:
    agent = _agent_for("place_insert")
    bridge = _ready_assembly_bridge(agent)
    settings = _move_insert_settings("MG")
    force_depth_profile = {
        "depth_fraction": [0.0, 1.0],
        "axial_upper_n": [10.0, 10.0],
        "lateral_upper_n": [5.0, 5.0],
        "torque_upper_nm": [1.0, 1.0],
    }
    settings["effective"] = {
        **dict(settings["effective"]),
        "demonstration_recipe": {
            "force_depth_profile": deepcopy(force_depth_profile),
            "hard_caps_sha256": "b" * 64,
        },
        "force_depth_profile": deepcopy(force_depth_profile),
        "hard_caps_sha256": "b" * 64,
    }
    bridge.digital_twin_move_insert_settings = (
        lambda *_args, **_kwargs: deepcopy(settings)
    )
    retained_profile = deepcopy(settings["effective"])
    retained_profile.pop("force_depth_profile")
    retained_profile.pop("hard_caps_sha256")
    agent._task_ctx["move_insert_profile"] = retained_profile
    agent._task_ctx["move_insert_profile_sha256"] = settings["profile_sha256"]

    _readiness, error = bridge._digital_twin_assembly_move_insert_recheck(
        {
            "target": "dual robots",
            "robot": "ur5e",
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
            "move_insert_profile_sha256": settings["profile_sha256"],
            "move_insert_effective": deepcopy(settings["effective"]),
        },
        agent,
    )

    assert error == ""
    assert agent._task_ctx["move_insert_profile"]["force_depth_profile"] == (
        force_depth_profile
    )
    assert agent._task_ctx["move_insert_profile"]["hard_caps_sha256"] == "b" * 64


def test_assembly_recheck_still_rejects_a_changed_profile_field() -> None:
    agent = _agent_for("place_insert")
    bridge = _ready_assembly_bridge(agent)
    settings = _move_insert_settings("MG")
    bridge.digital_twin_move_insert_settings = (
        lambda *_args, **_kwargs: deepcopy(settings)
    )
    agent._task_ctx["move_insert_profile"] = deepcopy(settings["effective"])
    agent._task_ctx["move_insert_profile"]["insertion_force_n"] = 9.0
    agent._task_ctx["move_insert_profile_sha256"] = settings["profile_sha256"]

    _readiness, error = bridge._digital_twin_assembly_move_insert_recheck(
        {
            "target": "dual robots",
            "robot": "ur5e",
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
            "move_insert_profile_sha256": settings["profile_sha256"],
            "move_insert_effective": deepcopy(settings["effective"]),
        },
        agent,
    )

    assert "Mismatched fields: insertion_force_n" in error


def test_assembly_stops_if_product_geometry_changes_after_pick_grasp() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)

    def _geometry(_part_name: str) -> dict[str, Any]:
        geometry = _mg_product_geometry()
        if agent._current_state == "picked":
            geometry["slot_xy"] = [0.001, 0.08]
        return geometry

    bridge._robot_function_product_geometry_for_part = _geometry

    result = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["failed_function"] == "place_approach"
    assert result["completed_functions"] == ["pick_approach", "pick_grasp"]
    assert "product geometry changed" in result["message"]
    assert [name for name, _kwargs in agent.calls] == [
        "pick_approach",
        "pick_grasp",
    ]


def test_assembly_runs_exact_order_with_fresh_preflight_and_one_agent() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    preflight_functions: list[str] = []
    preflight_agents: list[Any] = []
    insert_board_generations: list[int] = []
    original_preflight = bridge._digital_twin_robot_function_execution_preflight_async
    original_place_insert = agent.place_insert

    async def _observed_preflight(
        target: str,
        robot: str,
        function_name: str,
        origin_resource_location: str,
        destination_location: str,
        part_name: str,
    ) -> tuple[Any | None, dict[str, Any], dict[str, Any], str]:
        assert bridge._ur5e_robot_function_execution_lock.locked() is True
        preflight_functions.append(function_name)
        result = await original_preflight(
            target,
            robot,
            function_name,
            origin_resource_location,
            destination_location,
            part_name,
        )
        preflight_agents.append(result[0])
        return result

    async def _observed_place_insert(**kwargs: Any) -> dict[str, Any]:
        insert_board_generations.append(
            int(agent._task_ctx["assembly_board_v1_aruco_generation"])
        )
        return await original_place_insert(**kwargs)

    bridge._digital_twin_robot_function_execution_preflight_async = _observed_preflight
    agent.place_insert = _observed_place_insert  # type: ignore[method-assign]

    result = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert result["assembly_step_index"] == 5
    assert result["assembly_step_count"] == 5
    assert result["completed_functions"] == list(_ASSEMBLY_FUNCTIONS)
    assert result["failed_function"] == ""
    assert result["state"] == "idle"
    assert len(result["function_results"]) == 5
    assert preflight_functions == list(_ASSEMBLY_FUNCTIONS)
    assert preflight_agents == [agent] * 5
    assert [name for name, _kwargs in agent.calls] == list(_ASSEMBLY_FUNCTIONS)
    assert agent.calls == [
        (
            "pick_approach",
            {
                "origin_resource_location": "prusa-mk4-2",
                "part_name": "MG",
                "product_geometry": _mg_product_geometry(),
            },
        ),
        (
            "pick_grasp",
            {
                "origin_resource_location": "prusa-mk4-2",
                "part_name": "MG",
            },
        ),
        (
            "place_approach",
            {
                "destination_location": "assembly_board-v1",
                "part_name": "MG",
                "product_geometry": {
                    **_mg_place_product_geometry(),
                    "move_insert_profile": _move_insert_effective(),
                    "move_insert_profile_sha256": "a" * 64,
                },
            },
        ),
        (
            "place_insert",
            {
                "destination_location": "assembly_board-v1",
                "part_name": "MG",
            },
        ),
        ("move_home", {}),
    ]
    assert insert_board_generations == [1]
    assert agent._held_part is None
    assert agent._gripper_state == "open"
    assert agent._task_ctx == {}
    assert bridge._ur5e_robot_function_execution_lock.locked() is False


@pytest.mark.parametrize("failed_function", _ASSEMBLY_FUNCTIONS)
def test_assembly_stops_at_first_failed_function_without_recovery_motion(
    failed_function: str,
) -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    suspended_parts: list[str] = []
    bridge._suspend_move_insert_qualification = (
        lambda part_name: suspended_parts.append(part_name) or ""
    )

    async def _fail(**kwargs: Any) -> dict[str, Any]:
        agent.calls.append((failed_function, kwargs))
        return {
            "status": "failed",
            "content": f"forced {failed_function} failure",
        }

    setattr(agent, failed_function, _fail)
    result = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    failed_index = _ASSEMBLY_FUNCTIONS.index(failed_function)
    expected_calls = list(_ASSEMBLY_FUNCTIONS[: failed_index + 1])
    assert result["success"] is False
    assert result["assembly_step_count"] == 5
    assert result["completed_functions"] == list(_ASSEMBLY_FUNCTIONS[:failed_index])
    assert result["failed_function"] == failed_function
    assert len(result["function_results"]) == len(expected_calls)
    assert [name for name, _kwargs in agent.calls] == expected_calls
    if failed_function != "move_home":
        assert "move_home" not in expected_calls
    assert suspended_parts == (["MG"] if failed_function == "place_insert" else [])
    assert bridge._ur5e_robot_function_execution_lock.locked() is False


def test_assembly_stops_when_completed_function_has_wrong_state_transition() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)

    async def _pick_grasp_without_state_transition(**kwargs: Any) -> dict[str, Any]:
        agent.calls.append(("pick_grasp", kwargs))
        return {"status": "completed", "content": "did not update state"}

    agent.pick_grasp = _pick_grasp_without_state_transition  # type: ignore[method-assign]
    result = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["completed_functions"] == ["pick_approach"]
    assert result["failed_function"] == "pick_grasp"
    assert result["state"] == "at_pick"
    assert "picked" in result["message"]
    assert [name for name, _kwargs in agent.calls] == ["pick_approach", "pick_grasp"]


@pytest.mark.parametrize(
    ("function_name", "expected_calls", "completed_functions"),
    [
        ("pick_approach", ["pick_approach"], []),
        (
            "place_approach",
            ["pick_approach", "pick_grasp", "place_approach"],
            ["pick_approach", "pick_grasp"],
        ),
    ],
)
def test_assembly_stops_when_runtime_shaped_approach_descend_is_xyz_only(
    function_name: str,
    expected_calls: list[str],
    completed_functions: list[str],
) -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    original_approach = getattr(agent, function_name)

    async def _approach_with_xyz_only_descend(**kwargs: Any) -> dict[str, Any]:
        result = await original_approach(**kwargs)
        descend = dict(agent._task_ctx["resolved_cartesian_positions"]["descend"])
        agent._task_ctx["resolved_cartesian_positions"]["descend"] = {
            field: descend[field] for field in ("x", "y", "z")
        }
        return result

    setattr(agent, function_name, _approach_with_xyz_only_descend)
    result = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["failed_function"] == function_name
    assert result["completed_functions"] == completed_functions
    assert f"{function_name}.descend" in result["message"]
    assert "qx, qy, qz, and qw" in result["message"]
    assert [name for name, _kwargs in agent.calls] == expected_calls
    assert "move_home" not in expected_calls


@pytest.mark.parametrize(
    ("descend", "message_fragment"),
    [
        (
            {
                **MANUAL_DESCEND_POSE,
                "x": float("nan"),
            },
            "finite",
        ),
        (
            {
                **MANUAL_DESCEND_POSE,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 0.0,
            },
            "nonzero quaternion",
        ),
    ],
)
def test_assembly_resolved_descend_requires_finite_nonzero_se3(
    descend: dict[str, float],
    message_fragment: str,
) -> None:
    agent = _PhysicalUR5eAgent()
    agent._task_ctx = {
        "resolved_cartesian_positions": {"descend": descend},
    }

    error = SystemBridge._digital_twin_assembly_resolved_descend_error(
        agent,
        "pick_approach",
    )

    assert "pick_approach.descend" in error
    assert message_fragment in error


def test_assembly_stops_at_fresh_preflight_failure_without_dispatch_or_move_home() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    preflight_functions: list[str] = []
    original_preflight = bridge._digital_twin_robot_function_execution_preflight_async

    async def _fail_place_preflight(
        target: str,
        robot: str,
        function_name: str,
        origin_resource_location: str,
        destination_location: str,
        part_name: str,
    ) -> tuple[Any | None, dict[str, Any], dict[str, Any], str]:
        preflight_functions.append(function_name)
        if function_name == "place_approach":
            return None, {}, {"trajectory_action_ready": False}, "fresh TF is unavailable"
        return await original_preflight(
            target,
            robot,
            function_name,
            origin_resource_location,
            destination_location,
            part_name,
        )

    bridge._digital_twin_robot_function_execution_preflight_async = _fail_place_preflight
    result = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["completed_functions"] == ["pick_approach", "pick_grasp"]
    assert result["failed_function"] == "place_approach"
    assert "fresh TF is unavailable" in result["message"]
    assert preflight_functions == ["pick_approach", "pick_grasp", "place_approach"]
    assert [name for name, _kwargs in agent.calls] == ["pick_approach", "pick_grasp"]
    assert "move_home" not in [name for name, _kwargs in agent.calls]


@pytest.mark.parametrize("lifecycle_attribute", ["system_running", "_starting", "_stopping"])
def test_assembly_is_blocked_during_full_cais_lifecycle(
    lifecycle_attribute: str,
) -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    setattr(bridge, lifecycle_attribute, True)

    readiness = asyncio.run(
        bridge.digital_twin_assembly_readiness(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    result = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert readiness["ready"] is False
    assert "CAIS system" in readiness["message"]
    assert result["success"] is False
    assert "CAIS system" in result["message"]
    assert agent.calls == []


def test_assembly_rejects_xarm6_before_readiness_or_motion() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    arguments = {
        "origin_resource_location": "prusa-mk4-2",
        "destination_location": "assembly_board-v1",
        "part_name": "MG",
    }

    readiness = asyncio.run(
        bridge.digital_twin_assembly_readiness(
            "dual robots",
            "xarm6",
            **arguments,
        )
    )
    execution = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "xarm6",
            **arguments,
            confirmed=True,
        )
    )

    assert readiness["ready"] is False
    assert execution["success"] is False
    assert readiness["message"] == (
        "Assembly requires robot 'ur5e' because move_insert is only commissioned "
        "for ur5e."
    )
    assert execution["message"] == readiness["message"]
    assert agent.calls == []


@pytest.mark.parametrize(
    ("state", "held_part", "gripper_state", "message_fragment"),
    [
        ("at_pick", None, "open", "idle"),
        ("idle", "MG", "closed", "empty"),
        ("idle", None, "closed", "open"),
    ],
)
def test_assembly_readiness_requires_idle_empty_open_initial_state(
    state: str,
    held_part: str | None,
    gripper_state: str,
    message_fragment: str,
) -> None:
    bridge = _ready_assembly_bridge(
        _PhysicalUR5eAgent(
            state=state,
            held_part=held_part,
            gripper_state=gripper_state,
        )
    )

    result = asyncio.run(
        bridge.digital_twin_assembly_readiness(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert message_fragment in result["message"]


@pytest.mark.parametrize("function_name", ["pick_approach", "place_approach"])
@pytest.mark.parametrize("correction_state", ["buffered", "saved_unconfirmed"])
def test_assembly_blocks_unresolved_optional_cartesian_corrections(
    function_name: str,
    correction_state: str,
) -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    correction = {
        "step_name": "descend",
        "confirmed": False,
    }

    def _function_name(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        return str(args[2] if len(args) > 2 else kwargs["function_name"])

    def _buffered_steps(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        requested_function = _function_name(args, kwargs)
        if correction_state == "buffered" and requested_function == function_name:
            return [correction]
        return []

    def _saved_steps(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        requested_function = _function_name(args, kwargs)
        if (
            correction_state == "saved_unconfirmed"
            and requested_function == function_name
        ):
            return [correction]
        return []

    bridge.digital_twin_list_function_buffer_steps = _buffered_steps
    bridge.digital_twin_list_function_file_steps = _saved_steps
    result = asyncio.run(
        bridge.digital_twin_assembly_readiness(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    execution = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    expected_step = f"{function_name}.descend"
    assert result["ready"] is False
    assert expected_step in result["message"]
    assert "Save/Replace Pose or Clear Position" in result["message"]
    assert execution["success"] is False
    assert expected_step in execution["message"]
    assert "Save/Replace Pose or Clear Position" in execution["message"]
    assert agent.calls == []


@pytest.mark.parametrize(
    ("function_name", "expected_calls", "completed_functions"),
    [
        ("pick_approach", [], []),
        (
            "place_approach",
            ["pick_approach", "pick_grasp"],
            ["pick_approach", "pick_grasp"],
        ),
    ],
)
def test_assembly_rechecks_correction_after_preflight_before_approach_dispatch(
    function_name: str,
    expected_calls: list[str],
    completed_functions: list[str],
) -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    correction_visible = False
    original_preflight = bridge._digital_twin_robot_function_execution_preflight_async

    def _buffered_steps(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        requested_function = str(
            args[2] if len(args) > 2 else kwargs["function_name"]
        )
        if correction_visible and requested_function == function_name:
            return [{"step_name": "descend", "confirmed": True}]
        return []

    async def _make_correction_visible_after_preflight(
        target: str,
        robot: str,
        requested_function: str,
        origin_resource_location: str,
        destination_location: str,
        part_name: str,
    ) -> tuple[Any | None, dict[str, Any], dict[str, Any], str]:
        nonlocal correction_visible
        result = await original_preflight(
            target,
            robot,
            requested_function,
            origin_resource_location,
            destination_location,
            part_name,
        )
        if requested_function == function_name:
            correction_visible = True
        return result

    bridge.digital_twin_list_function_buffer_steps = _buffered_steps
    bridge._digital_twin_robot_function_execution_preflight_async = (
        _make_correction_visible_after_preflight
    )
    result = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["failed_function"] == function_name
    assert result["completed_functions"] == completed_functions
    assert f"{function_name}.descend" in result["message"]
    assert "Save/Replace Pose or Clear Position" in result["message"]
    assert [name for name, _kwargs in agent.calls] == expected_calls
    assert "move_home" not in expected_calls


def test_assembly_allows_missing_optional_cartesian_corrections() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_assembly_readiness(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert "fresh computed targets" in result["message"]
    assert "Missing optional robot corrections are allowed" in result["message"]
    assert "irreversibly" in result["message"]
    assert result["move_insert_client_ready"] is True
    assert agent._controller.move_insert_prepare_timeouts == [2.0]


def test_pick_only_readiness_does_not_prepare_optional_move_insert_client() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_bridge(agent)

    result = asyncio.run(
        bridge.digital_twin_robot_function_execution_readiness(
            "dual robots",
            "ur5e",
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
        )
    )

    assert result["ready"] is True
    assert agent._controller.move_insert_prepare_timeouts == []
    assert agent._controller._ur5e_hardware_insert_client is None


def test_generic_ur5e_controller_prewarm_does_not_prepare_optional_insert_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_to_thread_inline(monkeypatch)
    wait_timeouts: list[float] = []
    insert_preparation_calls: list[float] = []
    controller = SimpleNamespace(
        wait_for_services=lambda timeout_sec: (
            wait_timeouts.append(timeout_sec) or True
        ),
        _ensure_move_insert_client_ready=lambda *, timeout_sec=2.0: (
            insert_preparation_calls.append(timeout_sec) or (True, "")
        ),
    )
    agent = SimpleNamespace(
        _controller=controller,
        _controller_prewarm_done=False,
        controller_prewarm_timeout_s=4.0,
    )
    bridge = object.__new__(SystemBridge)

    error = asyncio.run(bridge._prewarm_ur5e_robot_function_agent(agent))

    assert error == ""
    assert wait_timeouts == [4.0]
    assert insert_preparation_calls == []
    assert agent._controller_prewarm_done is True


def test_confirmed_assembly_prepares_optional_client_before_pick_motion() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    preparation_call_counts: list[int] = []
    original_prepare = agent._controller._ensure_move_insert_client_ready

    def _prepare(*, timeout_sec: float = 2.0) -> tuple[bool, str]:
        preparation_call_counts.append(len(agent.calls))
        return original_prepare(timeout_sec=timeout_sec)

    agent._controller._ensure_move_insert_client_ready = _prepare
    result = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert preparation_call_counts == [0]
    assert agent._controller.move_insert_prepare_timeouts == [2.0]


def test_assembly_readiness_blocks_when_optional_client_is_unavailable() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    agent._controller._ensure_move_insert_client_ready = (
        lambda *, timeout_sec=2.0: (False, "move_insert discovery timed out")
    )

    result = asyncio.run(
        bridge.digital_twin_assembly_readiness(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert result["move_insert_client_ready"] is False
    assert result["message"] == "move_insert discovery timed out"
    assert agent.calls == []


def test_assembly_allows_saved_confirmed_optional_cartesian_corrections() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    bridge.digital_twin_list_function_file_steps = (
        lambda *_args, **_kwargs: [
            {"step_name": "descend", "confirmed": True},
        ]
    )

    readiness = asyncio.run(
        bridge.digital_twin_assembly_readiness(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    execution = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert readiness["ready"] is True
    assert "every present correction is confirmed" in readiness["message"]
    assert execution["success"] is True
    assert [name for name, _kwargs in agent.calls] == list(_ASSEMBLY_FUNCTIONS)


def test_assembly_readiness_rechecks_cais_lifecycle_before_confirmation() -> None:
    bridge = _ready_assembly_bridge()
    original_recording_error = bridge._digital_twin_place_approach_recording_error

    def _start_system_during_readiness(
        resource_agent: Any,
        destination_location: str,
        part_name: str,
    ) -> tuple[str, str]:
        result = original_recording_error(
            resource_agent,
            destination_location,
            part_name,
        )
        bridge.system_running = True
        return result

    bridge._digital_twin_place_approach_recording_error = (
        _start_system_during_readiness
    )
    result = asyncio.run(
        bridge.digital_twin_assembly_readiness(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )

    assert result["ready"] is False
    assert "Stop the CAIS system" in result["message"]


def test_assembly_requires_one_confirmation_and_respects_active_motion_lock() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    arguments = {
        "origin_resource_location": "prusa-mk4-2",
        "destination_location": "assembly_board-v1",
        "part_name": "MG",
    }

    unconfirmed = asyncio.run(
        bridge.digital_twin_execute_assembly("dual robots", "ur5e", **arguments)
    )
    assert unconfirmed["success"] is False
    assert "confirmation" in unconfirmed["message"]

    bridge._ur5e_robot_function_execution_active = "place_insert"
    bridge._ur5e_robot_function_execution_lock.acquire()
    try:
        busy = asyncio.run(
            bridge.digital_twin_execute_assembly(
                "dual robots",
                "ur5e",
                **arguments,
                confirmed=True,
            )
        )
    finally:
        bridge._ur5e_robot_function_execution_lock.release()

    assert busy["success"] is False
    assert busy["active_function"] == "place_insert"
    assert agent.calls == []


def test_assembly_cancellation_settles_only_in_flight_function_and_retains_lock() -> None:
    async def _exercise() -> None:
        agent = _PhysicalUR5eAgent()
        bridge = _ready_assembly_bridge(agent)
        started = asyncio.Event()
        finish = asyncio.Event()
        original_pick_grasp = agent.pick_grasp

        async def _slow_pick_grasp(**kwargs: Any) -> dict[str, Any]:
            started.set()
            await finish.wait()
            return await original_pick_grasp(**kwargs)

        agent.pick_grasp = _slow_pick_grasp  # type: ignore[method-assign]
        execution = asyncio.create_task(
            bridge.digital_twin_execute_assembly(
                "dual robots",
                "ur5e",
                origin_resource_location="prusa-mk4-2",
                destination_location="assembly_board-v1",
                part_name="MG",
                confirmed=True,
            )
        )
        await started.wait()

        progress = bridge.digital_twin_robot_function_execution_progress()
        assert progress["function_name"] == "Assembly"
        assert progress["assembly_step_index"] == 2
        assert progress["assembly_step_count"] == 5
        assert progress["completed_functions"] == ["pick_approach"]

        competing = await bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "move_home",
            confirmed=True,
        )
        assert competing["success"] is False
        assert competing["active_function"] == "Assembly"

        await bridge.start_system()
        assert bridge.last_error == (
            "Cannot start the CAIS system while UR5e motion is active: Assembly."
        )

        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution
        assert bridge._ur5e_robot_function_execution_lock.acquire(blocking=False) is False
        assert bridge._ur5e_robot_function_execution_active == "Assembly"

        finish.set()
        for _attempt in range(100):
            await asyncio.sleep(0.01)
            if bridge._ur5e_robot_function_execution_lock.acquire(blocking=False):
                bridge._ur5e_robot_function_execution_lock.release()
                break
        else:
            pytest.fail("Assembly lock was not released after the in-flight function settled")

        assert [name for name, _kwargs in agent.calls] == [
            "pick_approach",
            "pick_grasp",
        ]
        assert bridge._ur5e_robot_function_execution_active is None

    asyncio.run(_exercise())


def test_assembly_stops_before_place_insert_when_selected_profile_changes() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    profile_changed = False
    original_place_approach = agent.place_approach

    def _settings(
        _target: str,
        _robot: str,
        *,
        destination_location: str,
        part_name: str,
    ) -> dict[str, Any]:
        result = _move_insert_settings(part_name)
        if profile_changed:
            result["profile_sha256"] = "b" * 64
        return result

    async def _change_profile_after_place_approach(**kwargs: Any) -> dict[str, Any]:
        nonlocal profile_changed
        result = await original_place_approach(**kwargs)
        profile_changed = True
        return result

    bridge.digital_twin_move_insert_settings = _settings
    agent.place_approach = _change_profile_after_place_approach  # type: ignore[method-assign]

    result = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["failed_function"] == "place_insert"
    assert result["completed_functions"] == [
        "pick_approach",
        "pick_grasp",
        "place_approach",
    ]
    assert "profile changed" in result["message"]
    assert [name for name, _kwargs in agent.calls] == [
        "pick_approach",
        "pick_grasp",
        "place_approach",
    ]


def test_assembly_progress_reports_move_insert_step_and_frozen_profile() -> None:
    async def _exercise() -> None:
        agent = _PhysicalUR5eAgent()
        bridge = _ready_assembly_bridge(agent)
        bridge._ur5e_rtde_trajectory_status = lambda: {
            "updated_at": time.time(),
            "motion_kind": "insert",
            "insert_phase": "searching",
        }
        started = asyncio.Event()
        finish = asyncio.Event()
        original_place_insert = agent.place_insert

        async def _slow_move_insert(**kwargs: Any) -> dict[str, Any]:
            callback = agent._robot_task_progress_callback
            callback("place_insert", "move_insert")
            started.set()
            await finish.wait()
            return await original_place_insert(**kwargs)

        agent.place_insert = _slow_move_insert  # type: ignore[method-assign]
        execution = asyncio.create_task(
            bridge.digital_twin_execute_assembly(
                "dual robots",
                "ur5e",
                origin_resource_location="prusa-mk4-2",
                destination_location="assembly_board-v1",
                part_name="MG",
                confirmed=True,
            )
        )
        await started.wait()

        progress = bridge.digital_twin_robot_function_execution_progress()
        assert progress["function_name"] == "Assembly"
        assert progress["active_step"] == "place_insert.move_insert"
        assert progress["insert_phase"] == "searching"
        assert progress["move_insert_profile_sha256"] == "a" * 64
        assert progress["move_insert_effective"] == _move_insert_effective()

        finish.set()
        result = await execution
        assert result["success"] is True
        settled = bridge.digital_twin_robot_function_execution_progress()
        assert settled["active_step"] == ""
        assert settled["move_insert_profile_sha256"] == ""
        assert settled["move_insert_effective"] == {}

    asyncio.run(_exercise())


def _supervised_move_insert_trial_bridge(
    tmp_path: Path,
    *,
    automatic_checks_pass: bool = True,
    prior_confirmed_trials: int = 0,
) -> tuple[SystemBridge, _PhysicalUR5eAgent, dict[str, Any]]:
    agent = _agent_for("place_insert")
    agent._task_ctx.update(
        {
            "part_name": "MG",
            "move_insert_mode": "force_limited_trial",
            "held_part_handoff": {
                "part_name": "MG",
                "frame_id": "world",
                "tool_frame": "tool0",
                "part_frame": "held_part_origin",
                "world_tool0_pose_at_grasp": deepcopy(MANUAL_DESCEND_POSE),
                "world_held_part_pose_at_grasp": deepcopy(MANUAL_DESCEND_POSE),
                "tool0_to_held_part": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                },
            },
            "insert_pose": {**deepcopy(MANUAL_DESCEND_POSE), "z": 1.09},
            "insertion_axis_world": {"x": 0.0, "y": 0.0, "z": -1.0},
        }
    )
    bridge = _ready_assembly_bridge(agent)
    bridge._move_insert_trials_dir = tmp_path / "move_insert_trials"
    bridge._move_insert_server_trace_root = tmp_path / "server_traces"
    bridge._ur5e_rtde_reset_lock = threading.Lock()
    bridge._teleop_cartesian_modes = {"ur5e": "off", "xarm6": "off"}
    bridge._digital_twin_assembly_correction_error = lambda *_args, **_kwargs: ""
    trial_settings = _move_insert_settings("MG")
    trial_settings["effective"] = {
        **dict(trial_settings["effective"]),
        "demonstration_recipe": {
            "recording_id": "recording-test",
            "demonstration_sha256": "f" * 64,
            "place_approach_recording_sha256": "b" * 64,
        },
    }
    agent._task_ctx["move_insert_profile"] = deepcopy(
        trial_settings["effective"]
    )
    agent._task_ctx["move_insert_hard_caps"] = {
        name: float(trial_settings["hard_caps"][name])
        for name in (
            "insert_max_travel_m",
            "insert_start_position_tolerance_m",
            "insert_start_orientation_tolerance_rad",
            "insert_max_timeout_sec",
        )
    }
    agent._task_ctx["move_insert_hard_caps_sha256"] = str(
        trial_settings["hard_caps_sha256"]
    )
    agent._task_ctx["move_insert_boundary_ready"] = True
    agent._task_ctx["move_insert_boundary_error"] = ""
    bridge._digital_twin_move_insert_trial_settings = lambda _part_name: {
        **deepcopy(trial_settings),
        "profile_state": "trial_ready",
    }
    identities = {
        "robot": "ur5e",
        "destination_location": "assembly_board-v1",
        "part_name": "MG",
        "tool_frame": "tool0",
        "profile_sha256": "a" * 64,
        "hard_caps_sha256": str(trial_settings["hard_caps_sha256"]),
        "place_approach_recording_path": "/tmp/default__hardware.json",
        "place_approach_recording_sha256": "b" * 64,
        "board_calibration_id": "ur5e-calibration",
        "board_generation": 1,
        "board_geometry_sha256": "c" * 64,
        "task_context_sha256": "e" * 64,
        "resource_agent_identity": id(agent),
    }
    bridge._move_insert_current_identities = (
        lambda _resource_agent, *, destination_location, part_name, settings: (
            deepcopy(identities),
            "",
        )
    )
    bridge._move_insert_offline_qualification_identities = (
        lambda *, destination_location, part_name, settings: (
            {
                key: deepcopy(value)
                for key, value in identities.items()
                if key not in {
                    "place_approach_recording_path",
                    "task_context_sha256",
                    "resource_agent_identity",
                }
            }
            | {
                "recording_id": "recording-test",
                "demonstration_sha256": "f" * 64,
            },
            "",
        )
    )
    qualification_identity, qualification_identity_sha256, identity_error = (
        SystemBridge._move_insert_qualification_identity(
            identities=identities,
            move_insert_effective=trial_settings["effective"],
        )
    )
    assert identity_error == ""
    for index in range(prior_confirmed_trials):
        prior_trial_id = f"prior-confirmed-{index + 1}"
        prior_directory = bridge._move_insert_trials_dir / prior_trial_id
        prior_directory.mkdir(parents=True, exist_ok=True)
        (prior_directory / "summary.json").write_text(
            json.dumps(
                {
                    "trial_id": prior_trial_id,
                    "target": "dual robots",
                    "robot": "ur5e",
                    "destination_location": "assembly_board-v1",
                    "part_name": "MG",
                    "success": True,
                    "active": False,
                    "automatic_checks_passed": True,
                    "completion_motion_completed": True,
                    "released": True,
                    "lifted": True,
                    "confirmation_counted": True,
                    "confirmed_at": time.time() - (10 - index),
                    "qualification_policy_version": 3,
                    "qualification_policy_sha256": (
                        SystemBridge._move_insert_qualification_policy_sha256()
                    ),
                    "required_confirmed_trials": 1,
                    "qualification_identity": qualification_identity,
                    "qualification_identity_sha256": (
                        qualification_identity_sha256
                    ),
                    "move_insert_result": {"hard_limit_detected": False},
                    "move_insert_result_sha256": str(index + 1) * 64,
                    "server_trace_copied_sha256": str(index + 4) * 64,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    bridge._move_insert_qualification_view = (
        lambda *, part_name, identities=None: {
            "qualified": False,
            "qualification": {},
            "qualification_mismatch_fields": [],
            "qualification_error": f"move_insert is not confirmed for {part_name}.",
        }
    )
    release_feedback_sequence = 0

    def _release_feedback_snapshot(
        _resource_agent: Any,
        *,
        target: str,
    ) -> tuple[dict[str, float], str]:
        nonlocal release_feedback_sequence
        assert target == "dual robots"
        release_feedback_sequence += 1
        return {
            "tf_stamp_sec": float(release_feedback_sequence),
            "rtde_feedback_timestamp_sec": float(release_feedback_sequence),
        }, ""

    bridge._move_insert_release_feedback_snapshot = _release_feedback_snapshot  # type: ignore[method-assign]
    qualification_patches: list[dict[str, Any]] = []

    def _patch_qualification(**kwargs: Any) -> dict[str, Any]:
        qualification_patches.append(deepcopy(kwargs))
        return {
            "success": True,
            "changed": True,
            "profile_sha256": str(kwargs.get("expected_profile_sha256") or ""),
        }

    bridge._patch_move_insert_qualification = _patch_qualification
    suspended_parts: list[str] = []

    def _suspend(part_name: str) -> str:
        suspended_parts.append(part_name)
        return ""

    bridge._suspend_move_insert_qualification = _suspend
    calls = {
        "trial": 0,
        "move_insert_dispatched": 0,
        "completion": 0,
        "qualification_patches": qualification_patches,
        "suspended_parts": suspended_parts,
    }

    async def _execute_trial(
        pre_execute: Any = None,
        /,
        *,
        destination_location: str,
        part_name: str,
        trial_id: str,
    ) -> dict[str, Any]:
        calls["trial"] += 1
        pre_execute_error = str(pre_execute() or "") if callable(pre_execute) else ""
        if pre_execute_error:
            return {
                "status": "failed",
                "trial_id": trial_id,
                "motion_settled": True,
                "dispatch_attempted": False,
                "content": pre_execute_error,
                "move_insert_result": {
                    "success": False,
                    "trial_id": trial_id,
                    "motion_settled": True,
                    "state_uncertain": False,
                    "final_tool0_pose_valid": False,
                    "engagement_detected": False,
                    "seated_detected": False,
                },
            }
        calls["move_insert_dispatched"] += 1
        assert destination_location == "assembly_board-v1"
        assert part_name == "MG"
        trace_directory = bridge._move_insert_server_trace_root / trial_id
        trace_directory.mkdir(parents=True, exist_ok=True)
        trace_path = trace_directory / "trace.jsonl"
        trace_path.write_text(
            json.dumps(
                {
                    "sample_index": 0,
                    "phase": "seating",
                    "axial_force_n": 7.5,
                    "engagement_detected": automatic_checks_pass,
                    "seated_detected": automatic_checks_pass,
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        trace_sha256 = bridge_module.sha256_file(trace_path)
        move_insert_result = {
            "success": automatic_checks_pass,
            "state_uncertain": False,
            "final_tool0_pose_valid": True,
            "engagement_detected": automatic_checks_pass,
            "seated_detected": automatic_checks_pass,
            "trial_id": trial_id,
            "motion_settled": True,
            "server_trace_id": trial_id,
            "server_trace_path": str(trace_path),
            "server_trace_sha256": trace_sha256,
            "server_trace_status": "complete",
            "server_trace_complete": True,
            "server_trace_sample_count": 1,
            "feedback_trace": [
                {
                    "timestamp": 1.0,
                    "phase": "seating",
                    "axial_force_n": 7.5,
                    "engagement_detected": automatic_checks_pass,
                    "seated_detected": automatic_checks_pass,
                }
            ],
        }
        agent._task_ctx["move_insert_result"] = deepcopy(move_insert_result)
        agent._task_ctx["move_insert_trial_result_sha256"] = "d" * 64
        agent._task_ctx.setdefault("resolved_cartesian_positions", {})[
            "move_insert"
        ] = deepcopy(MANUAL_DESCEND_POSE)
        return {
            "status": "completed",
            "trial_id": trial_id,
            "motion_settled": True,
            "content": "move_insert trial settled",
            "completed_steps": ["move_insert"],
            "move_insert_result": move_insert_result,
            "move_insert_result_sha256": "d" * 64,
            "trial_ready_for_confirmation": automatic_checks_pass,
            "server_trace_id": trial_id,
            "server_trace_path": str(trace_path),
            "server_trace_sha256": trace_sha256,
            "server_trace_status": "complete",
            "server_trace_complete": True,
            "server_trace_sample_count": 1,
        }

    async def _complete_trial(
        pre_execute: Any = None,
        /,
        *,
        destination_location: str,
        part_name: str,
        expected_move_insert_result_sha256: str,
    ) -> dict[str, Any]:
        calls["completion"] += 1
        pre_execute_error = str(pre_execute() or "") if callable(pre_execute) else ""
        if pre_execute_error:
            return {"status": "failed", "content": pre_execute_error}
        assert destination_location == "assembly_board-v1"
        assert part_name == "MG"
        assert expected_move_insert_result_sha256 == "d" * 64
        agent._current_state = "placed"
        agent._held_part = None
        agent._gripper_state = "open"
        agent._task_ctx = {}
        return {
            "status": "completed",
            "content": "released and lifted once",
            "completed_steps": ["release_part", "lift"],
        }

    agent._execute_place_insert_move_insert_trial = _execute_trial  # type: ignore[attr-defined]
    agent._complete_place_insert_after_move_insert_trial = _complete_trial  # type: ignore[attr-defined]
    return bridge, agent, calls


def test_supervised_move_insert_trial_runs_only_insert_and_writes_diagnostics(
    tmp_path: Path,
) -> None:
    bridge, agent, calls = _supervised_move_insert_trial_bridge(tmp_path)

    unconfirmed = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    assert unconfirmed["success"] is False
    assert "confirmation" in unconfirmed["message"]
    assert calls["trial"] == 0

    readiness = asyncio.run(
        bridge.digital_twin_move_insert_trial_readiness(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    assert readiness["ready"] is True
    result = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert result["state"] == "awaiting_visual_confirmation"
    assert result["completion_eligible"] is True
    assert result["engagement_detected"] is True
    assert result["seated_detected"] is True
    assert calls["trial"] == 1
    assert calls["move_insert_dispatched"] == 1
    assert calls["completion"] == 0
    assert agent.calls == []
    assert agent._current_state == "positioned"
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"
    assert result["normal_repair_required"] is False
    assert result["hardware_stack_repair_required"] is False
    assert result["motion_settled"] is True
    assert Path(result["diagnostic_bundle_path"]).is_file()
    diagnostic_directory = Path(result["diagnostic_directory"])
    assert (diagnostic_directory / "summary.json").is_file()
    assert (diagnostic_directory / "trace.jsonl").is_file()
    assert Path(result["diagnostic_bundle_path"]).is_file()
    with bridge_module.zipfile.ZipFile(result["diagnostic_bundle_path"]) as archive:
        assert set(archive.namelist()) == {"summary.json", "trace.jsonl"}
    trace = (diagnostic_directory / "trace.jsonl").read_text(encoding="utf-8")
    assert '"phase":"seating"' in trace
    status = bridge.digital_twin_move_insert_trial_status(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=result["trial_id"],
    )
    assert status["trial_id"] == result["trial_id"]
    assert status["review_required"] is True


def test_authoritative_server_trace_is_not_duplicated_in_trial_summary(
    tmp_path: Path,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(tmp_path)

    result = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["move_insert_result"]["feedback_trace"]
    summary_path = Path(result["diagnostic_directory"]) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "feedback_trace" not in summary["move_insert_result"]
    assert "feedback_trace" not in summary["result"]["move_insert_result"]
    assert summary["move_insert_result_sha256"] == "d" * 64
    assert summary["server_trace_sample_count"] == 1
    assert summary["server_trace_copied"] is True
    assert summary["server_trace_copied_sha256"] == result[
        "server_trace_copied_sha256"
    ]
    with bridge_module.zipfile.ZipFile(result["diagnostic_bundle_path"]) as archive:
        bundled_summary = json.loads(archive.read("summary.json"))
    assert "feedback_trace" not in bundled_summary["move_insert_result"]
    assert "feedback_trace" not in bundled_summary["result"]["move_insert_result"]


def test_failed_server_trace_status_cannot_claim_complete_trace(
    tmp_path: Path,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(tmp_path)
    trial_id = "move-insert-impossible-trace-state"
    trace_directory = bridge._move_insert_server_trace_root / trial_id
    trace_directory.mkdir(parents=True)
    trace_path = trace_directory / "trace.jsonl"
    trace_path.write_text('{"sample_index":0}\n', encoding="utf-8")

    source, error = bridge._move_insert_server_trace_source(
        {
            "trial_id": trial_id,
            "move_insert_result": {
                "server_trace_id": trial_id,
                "server_trace_path": str(trace_path),
                "server_trace_sha256": bridge_module.sha256_file(trace_path),
                "server_trace_status": "failed",
                "server_trace_complete": True,
                "server_trace_sample_count": 1,
            },
        }
    )

    assert source is None
    assert error == "move_insert terminal server trace is incomplete"


def test_trial_diagnostics_retain_feedback_trace_without_server_trace(
    tmp_path: Path,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(tmp_path)
    trial_id = "move-insert-feedback-trace-fallback"
    feedback_trace = [
        {"phase": "seating", "insertion_depth_m": 0.001},
        {"phase": "settling", "insertion_depth_m": 0.002},
    ]

    persisted, diagnostic_error = bridge._write_move_insert_trial_diagnostics(
        {
            "trial_id": trial_id,
            "move_insert_result_sha256": "e" * 64,
            "move_insert_result": {
                "feedback_trace": deepcopy(feedback_trace),
            },
        }
    )

    assert diagnostic_error == ""
    assert persisted["server_trace_copied"] is False
    summary_path = Path(persisted["diagnostic_directory"]) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["move_insert_result"]["feedback_trace"] == feedback_trace
    trace_rows = [
        json.loads(line)
        for line in (
            Path(persisted["diagnostic_directory"]) / "trace.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["insertion_depth_m"] for row in trace_rows] == [0.001, 0.002]


def test_supervised_move_insert_reconciles_exact_saved_recording_context(
    tmp_path: Path,
) -> None:
    agent = _agent_for("place_insert")
    agent._task_ctx.update(
        {
            "part_name": "MG",
            "move_insert_mode": "force_limited_trial",
            "held_part_handoff": _normal_mg_handoff(),
        }
    )
    agent._task_ctx["assembly_board_v1_aruco"]["pose"] = deepcopy(
        MANUAL_DESCEND_POSE
    )
    bridge = _ready_assembly_bridge(agent)
    bridge._insertion_demonstrations_dir = tmp_path / "demonstrations"
    context, context_error = bridge._insertion_demonstration_context(
        agent,
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    assert context_error == ""

    recording_id = "insertion-demonstration-saved-test"
    directory = bridge._insertion_demonstration_directory(recording_id)
    directory.mkdir(parents=True)
    trace_path = directory / "trace.jsonl"
    trace_path.write_text('{"sample_index":0}\n', encoding="utf-8")
    demonstration_sha256 = bridge_module.sha256_file(trace_path)
    insert_pose = {**deepcopy(MANUAL_DESCEND_POSE), "z": 1.09}
    aruco_to_seated_held_part = deepcopy(MANUAL_DESCEND_POSE)
    seated_event = {
        "world_tool0_pose": insert_pose,
        "insertion_axis_world": {"x": 0.0, "y": 0.0, "z": -1.0},
        "aruco_to_seated_held_part": aruco_to_seated_held_part,
    }
    summary = {
        "recording_id": recording_id,
        "robot": "ur5e",
        "destination_location": "assembly_board-v1",
        "part_name": "MG",
        "context": context,
        "context_sha256": context["context_sha256"],
        "place_approach_recording_sha256": "b" * 64,
        "controller_result": {"trace_sha256": demonstration_sha256},
        "seated_event": seated_event,
    }
    (directory / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    settings = _move_insert_settings("MG")
    settings["profile_sha256"] = "c" * 64
    settings["effective"] = {
        **dict(settings["effective"]),
        "demonstration_recipe": {
            "robot": "ur5e",
            "part_name": "MG",
            "tool_frame": "tool0",
            "recording_id": recording_id,
            "demonstration_sha256": demonstration_sha256,
            "context_sha256": context["context_sha256"],
            "place_approach_recording_sha256": "b" * 64,
            "board_calibration_id": "ur5e-calibration",
            "aruco_to_seated_held_part": aruco_to_seated_held_part,
        },
    }
    learned_pre_insert_pose = {
        **deepcopy(MANUAL_DESCEND_POSE),
        "z": float(MANUAL_DESCEND_POSE["z"]) + 0.004,
    }
    bridge._move_insert_geometry_helper = lambda: (
        lambda **_kwargs: {
            "success": True,
            "pre_insert_pose": deepcopy(learned_pre_insert_pose),
            "insert_pose": deepcopy(insert_pose),
            "insertion_axis_world": {"x": 0.0, "y": 0.0, "z": -1.0},
        }
    )
    agent._task_ctx["move_insert_profile_sha256"] = "a" * 64
    agent._task_ctx["move_insert_profile"] = _move_insert_effective("MG")

    reconciled, reconciliation_error = (
        bridge._reconcile_saved_insertion_demonstration_task_context(
            agent,
            destination_location="assembly_board-v1",
            part_name="MG",
            settings=settings,
        )
    )

    assert reconciled is True
    assert reconciliation_error == ""
    assert agent._task_ctx["move_insert_profile_sha256"] == "c" * 64
    assert agent._task_ctx["pre_insert_pose"] == MANUAL_DESCEND_POSE
    assert agent._task_ctx["insert_pose"] == insert_pose
    assert agent._task_ctx["move_insert_boundary_ready"] is True
    assert agent._task_ctx["move_insert_boundary_error"] == ""
    assert agent._task_ctx["move_insert_boundary_metrics"][
        "learned_start_position_error_m"
    ] == pytest.approx(0.004)
    assert agent._task_ctx["move_insert_hard_caps_sha256"] == settings[
        "hard_caps_sha256"
    ]
    assert agent._task_ctx["insertion_demonstration_recording_id"] == recording_id
    assert (
        agent._task_ctx["insertion_demonstration_sha256"]
        == demonstration_sha256
    )


def test_supervised_place_insert_readiness_reconciles_before_preflight() -> None:
    agent = _agent_for("place_insert")
    bridge = _ready_assembly_bridge(agent)
    settings = _move_insert_settings("MG")
    settings["profile_sha256"] = "c" * 64
    settings["effective"] = {
        **dict(settings["effective"]),
        "demonstration_recipe": {
            "recording_id": "insertion-demonstration-saved-test",
            "demonstration_sha256": "d" * 64,
        },
    }
    agent._task_ctx["move_insert_profile_sha256"] = "c" * 64
    agent._task_ctx["move_insert_profile"] = deepcopy(settings["effective"])
    agent._task_ctx["move_insert_boundary_ready"] = False
    agent._task_ctx["move_insert_boundary_error"] = (
        "place_approach move_insert hard_caps_sha256 is missing"
    )
    agent._task_ctx.pop("move_insert_hard_caps_sha256", None)
    bridge._digital_twin_move_insert_trial_settings = lambda _part_name: deepcopy(
        settings
    )
    reconciliation_calls: list[dict[str, Any]] = []

    def _reconcile(
        resource_agent: Any,
        *,
        destination_location: str,
        part_name: str,
        settings: dict[str, Any],
    ) -> tuple[bool, str]:
        reconciliation_calls.append(deepcopy(settings))
        resource_agent._task_ctx["move_insert_profile_sha256"] = "c" * 64
        resource_agent._task_ctx["move_insert_profile"] = deepcopy(
            settings["effective"]
        )
        resource_agent._task_ctx["move_insert_hard_caps_sha256"] = str(
            settings["hard_caps_sha256"]
        )
        resource_agent._task_ctx["move_insert_boundary_ready"] = True
        resource_agent._task_ctx["move_insert_boundary_error"] = ""
        return True, ""

    bridge._reconcile_saved_insertion_demonstration_task_context = _reconcile
    token = bridge_module._MOVE_INSERT_PREFLIGHT_REQUIRED_CONTEXT.set(True)
    try:
        readiness, error = (
            bridge._digital_twin_robot_function_move_insert_readiness(
                "dual robots",
                "ur5e",
                "place_insert",
                "assembly_board-v1",
                "MG",
                agent,
                {},
            )
        )
    finally:
        bridge_module._MOVE_INSERT_PREFLIGHT_REQUIRED_CONTEXT.reset(token)

    assert error == ""
    assert readiness["move_insert_trial_preparation"] is True
    assert reconciliation_calls == [settings]
    assert agent._task_ctx["move_insert_profile_sha256"] == "c" * 64
    assert agent._task_ctx["move_insert_boundary_ready"] is True


def test_zero_sample_terminal_failure_keeps_complete_server_trace(
    tmp_path: Path,
) -> None:
    bridge, agent, calls = _supervised_move_insert_trial_bridge(tmp_path)
    execute_trial = agent._execute_place_insert_move_insert_trial

    async def _zero_sample_failure(
        pre_execute: Any = None,
        /,
        *,
        destination_location: str,
        part_name: str,
        trial_id: str,
    ) -> dict[str, Any]:
        result = await execute_trial(
            pre_execute,
            destination_location=destination_location,
            part_name=part_name,
            trial_id=trial_id,
        )
        trace_path = Path(result["server_trace_path"])
        trace_path.write_text(
            '{"event":"trace_started"}\n{"event":"terminal_failure"}\n',
            encoding="utf-8",
        )
        trace_sha256 = bridge_module.sha256_file(trace_path)
        move_insert_result = dict(result["move_insert_result"])
        move_insert_result.update(
            {
                "success": False,
                "engagement_detected": False,
                "seated_detected": False,
                "server_trace_sha256": trace_sha256,
                "server_trace_status": "complete",
                "server_trace_sample_count": 0,
            }
        )
        agent._task_ctx["move_insert_result"] = deepcopy(move_insert_result)
        return {
            **result,
            "status": "failed",
            "trial_ready_for_confirmation": False,
            "server_trace_sha256": trace_sha256,
            "server_trace_status": "complete",
            "server_trace_sample_count": 0,
            "move_insert_result": move_insert_result,
        }

    agent._execute_place_insert_move_insert_trial = _zero_sample_failure

    result = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["completion_eligible"] is False
    assert result["review_required"] is False
    assert result["server_trace_copied"] is True
    assert result["server_trace_copied_sha256"] == result["result"][
        "server_trace_sha256"
    ]
    persisted_trace = Path(result["diagnostic_directory"]) / "trace.jsonl"
    assert persisted_trace.read_text(encoding="utf-8") == (
        '{"event":"trace_started"}\n{"event":"terminal_failure"}\n'
    )
    assert "contains no RTDE samples" in result["message"]
    assert calls["completion"] == 0


def test_server_trace_copy_hash_mismatch_blocks_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(tmp_path)

    def _replace_copy_with_changed_bytes(_source: Any, destination: Any) -> None:
        Path(destination).write_text("changed during copy\n", encoding="utf-8")

    monkeypatch.setattr(
        bridge_module.shutil,
        "copyfile",
        _replace_copy_with_changed_bytes,
    )
    result = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["completion_eligible"] is False
    assert result["automatic_checks_passed"] is False
    assert "changed while it was copied" in result["message"]
    assert (
        Path(result["diagnostic_directory"]) / "trace.jsonl"
    ).read_text(encoding="utf-8") == ""


def test_uncertain_terminal_insert_failure_forbids_jog_until_repair(
    tmp_path: Path,
) -> None:
    bridge, agent, _calls = _supervised_move_insert_trial_bridge(
        tmp_path,
        automatic_checks_pass=False,
    )
    execute_trial = agent._execute_place_insert_move_insert_trial

    async def _uncertain_failure(
        pre_execute: Any = None,
        /,
        *,
        destination_location: str,
        part_name: str,
        trial_id: str,
    ) -> dict[str, Any]:
        result = await execute_trial(
            pre_execute,
            destination_location=destination_location,
            part_name=part_name,
            trial_id=trial_id,
        )
        move_insert_result = dict(result["move_insert_result"])
        move_insert_result["state_uncertain"] = True
        move_insert_result["motion_settled"] = False
        return {
            **result,
            "motion_settled": False,
            "move_insert_result": move_insert_result,
        }

    agent._execute_place_insert_move_insert_trial = _uncertain_failure
    result = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["normal_repair_required"] is True
    assert result["hardware_stack_repair_required"] is True
    assert "do not jog" in result["message"]
    assert "Repair Hardware Stack" in result["message"]
    assert "jog back to pre-insertion" not in result["message"]
    pending, pending_error = bridge._read_move_insert_pending_review()
    assert pending_error == ""
    assert pending is not None
    assert pending["hardware_stack_repair_required"] is True


def test_supervised_move_insert_rechecks_expected_start_under_lock(
    tmp_path: Path,
) -> None:
    bridge, agent, calls = _supervised_move_insert_trial_bridge(tmp_path)
    original_snapshot = bridge._robot_function_capture_snapshot
    snapshot_calls = 0

    def _snapshot(target: str, robot: str) -> dict[str, Any]:
        nonlocal snapshot_calls
        snapshot_calls += 1
        snapshot = deepcopy(original_snapshot(target, robot))
        if snapshot_calls >= 3:
            snapshot["tf_stamp_sec"] = time.time()
            snapshot["waypoint"]["pose"]["z"] -= 0.01
        return snapshot

    bridge._robot_function_capture_snapshot = _snapshot

    result = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert "blocked before move_insert dispatch" in result["message"]
    assert "No move_insert goal" in result["message"]
    assert "Complete trial-keyed server trace evidence is missing" not in result[
        "message"
    ]
    assert "jog back to pre-insertion" not in result["message"]
    assert result["dispatch_attempted"] is False
    assert snapshot_calls == 3
    assert calls["trial"] == 1
    assert calls["move_insert_dispatched"] == 0
    assert calls["completion"] == 0
    assert agent.calls == []
    assert agent._current_state == "positioned"
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"
    assert result["normal_repair_required"] is False
    assert result["hardware_stack_repair_required"] is False
    assert result["motion_settled"] is True
    assert calls["suspended_parts"] == []
    assert Path(result["diagnostic_bundle_path"]).is_file()


def test_supervised_move_insert_rechecks_board_generation_under_lock(
    tmp_path: Path,
) -> None:
    bridge, agent, calls = _supervised_move_insert_trial_bridge(tmp_path)
    base_board_status = dict(
        bridge.perception_manager.assembly_board_v1_aruco_status("ur5e")
    )
    accepted_generation = 1
    bridge.perception_manager.assembly_board_v1_aruco_status = lambda _role: {
        **base_board_status,
        "accepted_generation": accepted_generation,
        "accepted_calibration_id": "ur5e-calibration",
    }
    production_current_identities = (
        SystemBridge._move_insert_current_identities.__get__(bridge, SystemBridge)
    )
    identity_calls = 0

    def _current_identities(
        resource_agent: Any,
        *,
        destination_location: str,
        part_name: str,
        settings: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        nonlocal accepted_generation, identity_calls
        identity_calls += 1
        if identity_calls == 3:
            accepted_generation = 2
        identities, identity_error = production_current_identities(
            resource_agent,
            destination_location=destination_location,
            part_name=part_name,
            settings=settings,
        )
        identities["place_approach_recording_sha256"] = "b" * 64
        return identities, identity_error

    bridge._move_insert_current_identities = _current_identities  # type: ignore[method-assign]

    result = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert "accepted board generation changed after place_approach" in result["message"]
    assert "before move_insert dispatch" in result["message"]
    assert identity_calls == 3
    assert calls["trial"] == 1
    assert calls["move_insert_dispatched"] == 0
    assert calls["completion"] == 0
    assert agent.calls == []
    assert agent._current_state == "positioned"
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"


def test_pending_supervised_trial_recovers_fail_closed_after_bridge_restart(
    tmp_path: Path,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(tmp_path)
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )
    assert bridge._move_insert_pending_review_path().is_file()

    restarted, _replacement_agent, restarted_calls = (
        _supervised_move_insert_trial_bridge(tmp_path)
    )
    restarted._ur5e_robot_function_agent = None
    restarted.resource_agents = []
    restarted._hardware_stack_lifecycle_state = "stopped"
    restarted._hardware_stack_selected = ""
    status = restarted.digital_twin_move_insert_trial_status(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
    )

    assert status["durable_recovered"] is True
    assert status["review_required"] is False
    assert status["active"] is False
    assert status["completion_eligible"] is False
    assert status["recovery_required"] is False
    assert status["normal_repair_required"] is True
    assert status["hardware_stack_repair_required"] is True
    assert status["part_clamped"] is True
    assert status["released"] is False
    assert status["lifted"] is False
    assert "Confirm Completion is blocked" in status["message"]
    assert "Repair Hardware Stack" in status["message"]
    assert status["server_trace_copied"] is True
    assert '"phase":"seating"' in (
        Path(status["diagnostic_directory"]) / "trace.jsonl"
    ).read_text(encoding="utf-8")
    assert "requires Repair Hardware Stack" in (
        restarted._move_insert_pending_review_error()
    )
    restarted._ros2_procs = {}
    restarted._hardware_stack_lifecycle_generation = 0
    restarted._hardware_stack_last_error = ""
    restarted._hardware_stack_failed_process = ""
    restarted._hardware_stack_failed_return_code = None
    restarted._hardware_stack_process_log_path = ""
    restarted._hardware_cartesian_readiness = {}
    for stack_name in ("dual robots", "ur5e", "xarm6"):
        stack_status = restarted.hardware_stack_status(stack_name)
        assert stack_status["overall"] == "stopped"
        assert stack_status["lifecycle_state"] == "stopped"
        assert stack_status["hardware_stack_repair_required"] is False
        assert stack_status["hardware_stack_repair_reason"] == ""
        assert stack_status["hardware_stack_operation_blocked_reason"] == ""
        for operation in ("start", "stop", "repair"):
            assert (
                restarted._hardware_stack_lifecycle_motion_error(
                    operation,
                    stack_name,
                )
                == ""
            )
    blocked_confirm = asyncio.run(
        restarted.digital_twin_confirm_move_insert_completion(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id=trial["trial_id"],
            confirmed=True,
        )
    )
    assert blocked_confirm["success"] is False
    assert restarted_calls["completion"] == 0
    pending, pending_read_error = restarted._read_move_insert_pending_review()
    assert pending_read_error == ""
    assert pending is not None
    assert pending["trial_id"] == trial["trial_id"]
    assert pending["hardware_stack_repair_required"] is True

    restarted_again, _third_agent, _third_calls = (
        _supervised_move_insert_trial_bridge(tmp_path)
    )
    restarted_again._hardware_stack_lifecycle_state = "stopped"
    restarted_again._hardware_stack_selected = ""
    durable_error = restarted_again._move_insert_pending_review_error()
    assert "requires Repair Hardware Stack" in durable_error
    assert restarted_again._ur5e_robot_function_state_uncertain is True
    assert durable_error in (
        restarted_again._ur5e_robot_function_agent_handoff_error()
    )
    for stack_name in ("dual robots", "ur5e", "xarm6"):
        for operation in ("start", "stop", "repair"):
            assert (
                restarted_again._hardware_stack_lifecycle_motion_error(
                    operation,
                    stack_name,
                )
                == ""
            )


def test_stale_active_trial_copies_exact_terminal_server_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(tmp_path)
    trial_id = "move-insert-stale-terminal-trace"
    bridge._store_move_insert_trial(
        {
            "success": True,
            "ready": False,
            "target": "dual robots",
            "robot": "ur5e",
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
            "trial_id": trial_id,
            "active": True,
            "review_required": True,
            "recovery_required": False,
            "part_clamped": True,
            "profile_sha256": "a" * 64,
            "move_insert_effective": {"calibration_id": "insert-calibration"},
            "started_at": 10.0,
            "message": "Supervised Test move_insert is running.",
        }
    )
    server_directory = bridge._move_insert_server_trace_root / trial_id
    server_directory.mkdir(parents=True)
    server_trace = server_directory / "trace.jsonl"
    server_trace.write_text(
        '{"event":"trace_started"}\n'
        '{"sample_index":0,"axial_force_n":31.0}\n'
        '{"event":"terminal_hard_limit"}\n',
        encoding="utf-8",
    )
    server_trace_sha256 = bridge_module.sha256_file(server_trace)
    terminal_status = {
        "state": "failed",
        "motion_kind": "insert",
        "part_name": "MG",
        "trial_id": trial_id,
        "profile_sha256": "a" * 64,
        "calibration_id": "insert-calibration",
        "updated_at": 12.0,
        "insert_motion_settled": False,
        "rtde_reset_required": True,
        "blocked_reason": "axial force hard limit",
        "server_trace_id": trial_id,
        "server_trace_path": str(server_trace),
        "server_trace_sha256": server_trace_sha256,
        "server_trace_status": "complete",
        "server_trace_complete": True,
        "server_trace_sample_count": 1,
    }
    restarted, _replacement_agent, _replacement_calls = (
        _supervised_move_insert_trial_bridge(tmp_path)
    )
    restarted._ur5e_rtde_trajectory_status = lambda: deepcopy(terminal_status)
    restarted._suspend_move_insert_qualification = lambda _part_name: ""
    monkeypatch.setattr(
        bridge_module,
        "_UR5E_RTDE_TRAJECTORY_LAST_TERMINAL_STATUS",
        tmp_path / "missing-terminal.json",
    )

    status = restarted.digital_twin_move_insert_trial_status(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial_id,
    )

    assert status["success"] is False
    assert status["active"] is False
    assert status["completion_eligible"] is False
    assert status["hardware_stack_repair_required"] is True
    assert status["part_clamped"] is True
    assert status["server_trace_copied"] is True
    assert status["server_trace_copied_sha256"] == server_trace_sha256
    persisted_trace = Path(status["diagnostic_directory"]) / "trace.jsonl"
    assert persisted_trace.read_bytes() == server_trace.read_bytes()
    assert bridge_module.sha256_file(persisted_trace) == server_trace_sha256
    pending, pending_error = restarted._read_move_insert_pending_review()
    assert pending_error == ""
    assert pending is not None
    assert pending["hardware_stack_repair_required"] is True


@pytest.mark.parametrize(
    ("trial_id", "status_trial_id", "status_calibration_id", "matched"),
    [
        ("move-insert-new-trial", "", "insert-calibration", False),
        (
            "move-insert-new-trial",
            "move-insert-new-trial",
            "different-calibration",
            False,
        ),
        (
            bridge_module._LEGACY_MOVE_INSERT_PENDING_TRIAL_ID,
            "",
            "insert-calibration",
            True,
        ),
    ],
)
def test_stale_trial_reconciliation_requires_exact_trial_and_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trial_id: str,
    status_trial_id: str,
    status_calibration_id: str,
    matched: bool,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(tmp_path)
    terminal_path = tmp_path / "no-terminal-status.json"
    monkeypatch.setattr(
        bridge_module,
        "_UR5E_RTDE_TRAJECTORY_LAST_TERMINAL_STATUS",
        terminal_path,
    )
    terminal_reason = "exact terminal force failure"
    bridge._ur5e_rtde_trajectory_status = lambda: {
        "state": "failed",
        "motion_kind": "insert",
        "part_name": "MG",
        "trial_id": status_trial_id,
        "profile_sha256": "a" * 64,
        "calibration_id": status_calibration_id,
        "updated_at": 12.0,
        "insert_motion_settled": False,
        "rtde_reset_required": True,
        "blocked_reason": terminal_reason,
    }
    bridge._suspend_move_insert_qualification = lambda _part_name: ""
    trial = {
        "success": True,
        "ready": False,
        "target": "dual robots",
        "robot": "ur5e",
        "destination_location": "assembly_board-v1",
        "part_name": "MG",
        "trial_id": trial_id,
        "profile_sha256": "a" * 64,
        "move_insert_effective": {"calibration_id": "insert-calibration"},
        "started_at": 10.0,
        "active": False,
        "review_required": True,
        "part_clamped": True,
        "message": "original pending trial",
    }

    reconciled = bridge._reconcile_recovered_move_insert_trial(trial)

    assert reconciled["active"] is False
    assert reconciled["part_clamped"] is True
    assert reconciled["normal_repair_required"] is True
    assert (terminal_reason in reconciled["message"]) is matched
    if matched:
        assert reconciled["controller_status"]["calibration_id"] == (
            "insert-calibration"
        )
    else:
        assert reconciled["controller_status"] == {}


def test_confirm_move_insert_completion_runs_only_suffix_once_and_qualifies(
    tmp_path: Path,
) -> None:
    bridge, agent, calls = _supervised_move_insert_trial_bridge(tmp_path)
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )
    seated_pose = {**deepcopy(MANUAL_DESCEND_POSE), "z": 1.09}
    agent._task_ctx["resolved_cartesian_positions"]["move_insert"] = deepcopy(
        seated_pose
    )
    bridge._robot_function_capture_snapshot = lambda _target, _robot: {
        "success": True,
        "world_tool0_ready": True,
        "tf_stamp_sec": time.time(),
        "blocked_reason": "",
        "waypoint": {
            "source": "hardware",
            "pose": {
                **deepcopy(seated_pose),
                "frame_id": "world",
                "child_frame_id": "tool0",
            },
        },
    }

    unconfirmed = asyncio.run(
        bridge.digital_twin_confirm_move_insert_completion(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id=trial["trial_id"],
        )
    )
    assert unconfirmed["success"] is False
    assert calls["completion"] == 0
    bridge._latch_move_insert_suspension("MG", "test suspension")
    assert bridge._move_insert_suspension_error("MG") == "test suspension"

    confirmed = asyncio.run(
        bridge.digital_twin_confirm_move_insert_completion(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id=trial["trial_id"],
            confirmed=True,
        )
    )

    assert confirmed["success"] is True, confirmed["message"]
    assert confirmed["state"] == "confirmed"
    assert confirmed["released"] is True
    assert confirmed["lifted"] is True
    assert calls["trial"] == 1
    assert calls["completion"] == 1
    assert agent._current_state == "placed"
    assert agent._held_part is None
    assert agent._gripper_state == "open"
    assert agent.calls == []
    qualification_patch = calls["qualification_patches"][-1]
    qualification = qualification_patch["qualification"]
    assert qualification["trial_id"] == trial["trial_id"]
    assert qualification["robot"] == "ur5e"
    assert qualification["destination_location"] == "assembly_board-v1"
    assert qualification["part_name"] == "MG"
    assert qualification["profile_sha256"] == "a" * 64
    assert qualification["place_approach_recording_sha256"] == "b" * 64
    assert qualification["board_calibration_id"] == "ur5e-calibration"
    assert qualification["board_geometry_sha256"] == "c" * 64
    assert qualification["board_generation"] == 1
    assert qualification["generation"] >= 1
    assert qualification["confirmed_at"].endswith("Z")
    assert bridge._move_insert_suspension_error("MG") == ""
    assert bridge._ur5e_move_insert_profile_reload_required is True
    summary_path = Path(confirmed["diagnostic_directory"]) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["qualified"] is True
    assert summary["review_required"] is False
    assert summary["qualification"]["confirmed_at"] == qualification["confirmed_at"]
    assert float(summary["confirmed_at"]) > 0.0
    assert "future place_insert and Assembly runs" in summary["message"]
    with bridge_module.zipfile.ZipFile(
        confirmed["diagnostic_bundle_path"]
    ) as archive:
        bundled_summary = json.loads(archive.read("summary.json"))
    assert bundled_summary["qualified"] is True
    assert bundled_summary["review_required"] is False
    pending, pending_error = bridge._read_move_insert_pending_review()
    assert pending_error == ""
    assert pending is None

    blocked_pick = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "pick_approach",
            origin_resource_location="prusa-mk4-2",
            part_name="MG",
            confirmed=True,
        )
    )
    assert blocked_pick["success"] is False
    assert "Complete move_home" in blocked_pick["message"]
    home = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "move_home",
            confirmed=True,
        )
    )
    assert home["success"] is True
    assert agent._current_state == "idle"
    assert bridge._ur5e_move_insert_profile_reload_required is True

    replacement_agent = _PhysicalUR5eAgent()
    disposed: list[_PhysicalUR5eAgent] = []

    async def _dispose_confirmed_agent() -> None:
        disposed.append(agent)
        bridge._ur5e_robot_function_agent = None
        bridge._ur5e_robot_function_agent_domain_id = None

    async def _ensure_reloaded_agent(
        _target: str,
        _robot: str,
    ) -> tuple[_PhysicalUR5eAgent, str]:
        bridge._ur5e_robot_function_agent = replacement_agent
        bridge._ur5e_robot_function_agent_domain_id = 42
        return replacement_agent, ""

    bridge._dispose_ur5e_robot_function_agent = _dispose_confirmed_agent  # type: ignore[method-assign]
    bridge._ensure_ur5e_robot_function_agent_locked = _ensure_reloaded_agent  # type: ignore[method-assign]
    bridge.resource_agents = []
    reloaded_agent, _kwargs, _readiness, reload_error = asyncio.run(
        bridge._digital_twin_robot_function_execution_preflight_async(
            "dual robots",
            "ur5e",
            "pick_approach",
            "prusa-mk4-2",
            "",
            "MG",
        )
    )
    assert reload_error == ""
    assert disposed == [agent]
    assert reloaded_agent is replacement_agent
    assert bridge._ur5e_move_insert_profile_reload_required is False

    repeated = asyncio.run(
        bridge.digital_twin_confirm_move_insert_completion(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id=trial["trial_id"],
            confirmed=True,
        )
    )
    assert repeated["qualified"] is True
    assert calls["trial"] == 1
    assert calls["completion"] == 1


def test_move_insert_requires_one_confirmation(
    tmp_path: Path,
) -> None:
    def _run_confirmed_trial() -> tuple[dict[str, Any], dict[str, Any]]:
        bridge, agent, _calls = _supervised_move_insert_trial_bridge(
            tmp_path,
            prior_confirmed_trials=0,
        )
        trial = asyncio.run(
            bridge.digital_twin_execute_move_insert_trial(
                "dual robots",
                "ur5e",
                destination_location="assembly_board-v1",
                part_name="MG",
                confirmed=True,
            )
        )
        seated_pose = {**deepcopy(MANUAL_DESCEND_POSE), "z": 1.09}
        agent._task_ctx["resolved_cartesian_positions"]["move_insert"] = (
            deepcopy(seated_pose)
        )
        bridge._robot_function_capture_snapshot = lambda _target, _robot: {
            "success": True,
            "world_tool0_ready": True,
            "tf_stamp_sec": time.time(),
            "blocked_reason": "",
            "waypoint": {
                "source": "hardware",
                "pose": {
                    **deepcopy(seated_pose),
                    "frame_id": "world",
                    "child_frame_id": "tool0",
                },
            },
        }
        confirmed = asyncio.run(
            bridge.digital_twin_confirm_move_insert_completion(
                "dual robots",
                "ur5e",
                destination_location="assembly_board-v1",
                part_name="MG",
                trial_id=trial["trial_id"],
                confirmed=True,
            )
        )
        return confirmed, bridge.digital_twin_move_insert_trial_status(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id=trial["trial_id"],
        )

    confirmed, confirmed_status = _run_confirmed_trial()
    assert confirmed["qualified"] is True
    assert confirmed["state"] == "confirmed"
    assert confirmed["confirmed_trial_count"] == 1
    assert len(confirmed["qualification"]["confirmed_trial_ids"]) == 1
    assert len(
        confirmed["qualification"]["confirmed_trial_result_sha256s"]
    ) == 1
    assert len(
        confirmed["qualification"]["confirmed_trial_trace_sha256s"]
    ) == 1
    assert len(
        confirmed["qualification"]["qualification_evidence_sha256"]
    ) == 64
    assert confirmed_status["qualified"] is True
    assert confirmed_status["confirmed_trial_count"] == 1
    assert "Confirmed 1 of 1" in confirmed["message"]


def test_move_insert_terminal_failure_resets_durable_confirmation_streak(
    tmp_path: Path,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(
        tmp_path,
        automatic_checks_pass=False,
        prior_confirmed_trials=1,
    )
    failed = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )
    assert failed["automatic_checks_passed"] is False
    assert "rerun place_approach" in failed["message"]
    assert "stage at assembly_board-v1 again" in failed["message"]

    restarted, _agent, _calls = _supervised_move_insert_trial_bridge(
        tmp_path,
        prior_confirmed_trials=0,
    )
    status = restarted.digital_twin_move_insert_trial_status(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    assert status["qualified"] is False
    assert status["confirmed_trial_count"] == 0
    assert status["state"] == "not_confirmed"


def test_move_insert_depth_failure_is_terminal_and_reports_observed_shortfall(
    tmp_path: Path,
) -> None:
    bridge, agent, calls = _supervised_move_insert_trial_bridge(
        tmp_path,
        automatic_checks_pass=False,
        prior_confirmed_trials=0,
    )
    execute_trial = agent._execute_place_insert_move_insert_trial
    failure_message = (
        "UR5e insertion search exhausted: insertion timed out without engagement "
        "MG remains clamped; release_part, lift, and move_home were not commanded."
    )

    async def _depth_failure(
        pre_execute: Any = None,
        /,
        *,
        destination_location: str,
        part_name: str,
        trial_id: str,
    ) -> dict[str, Any]:
        result = await execute_trial(
            pre_execute,
            destination_location=destination_location,
            part_name=part_name,
            trial_id=trial_id,
        )
        move_insert_result = {
            **dict(result["move_insert_result"]),
            "message": "insertion timed out without engagement",
            "final_insertion_depth_m": 0.018905057234852887,
            "final_depth_error_m": 0.005371949282258315,
            "disengagement_cycle_count": 1,
            "last_disengagement_reason": (
                "MG disengagement timed out after 0.300 s before contact cleared"
            ),
            "disengagement_contact_cleared": False,
            "feedback_trace": [
                {"phase": "seating", "insertion_depth_m": 0.018768017343248095},
                {
                    "phase": "disengaging",
                    "insertion_depth_m": 0.019461623750438223,
                },
                {
                    "phase": "disengaging",
                    "insertion_depth_m": 0.018905057234852887,
                },
            ],
        }
        agent._task_ctx["move_insert_result"] = deepcopy(move_insert_result)
        return {
            **result,
            "status": "failed",
            "content": failure_message,
            "trial_ready_for_confirmation": False,
            "move_insert_result": move_insert_result,
        }

    agent._execute_place_insert_move_insert_trial = _depth_failure

    failed = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert failed["success"] is False
    assert failed["state"] == "not_confirmed"
    assert failed["qualification_state"] == "not_confirmed"
    assert failed["active"] is False
    assert failed["completion_eligible"] is False
    assert failed["recovery_required"] is True
    assert failed["normal_repair_required"] is False
    assert failed["part_clamped"] is True
    assert failed["released"] is False
    assert failed["lifted"] is False
    assert failed["insertion_depth_outside_seated_tolerance"] is True
    assert failed["max_insertion_depth_outside_seated_tolerance"] is True
    assert failed["final_insertion_depth_outside_seated_tolerance"] is True
    assert failed["max_insertion_depth_m"] == pytest.approx(
        0.019461623750438223
    )
    assert failed["final_insertion_depth_m"] == pytest.approx(
        0.018905057234852887
    )
    assert failed["target_insertion_depth_m"] == pytest.approx(
        0.0242770065171112
    )
    assert failed["required_seated_depth_m"] == pytest.approx(
        0.0232770065171112
    )
    assert failed["max_insertion_depth_shortfall_m"] == pytest.approx(
        0.003815382766672977
    )
    assert failed["message"].startswith(failure_message)
    assert "maximum observed 19.462 mm" in failed["message"]
    assert "final observed 18.905 mm" in failed["message"]
    assert "required at least 23.277 mm" in failed["message"]
    assert "Protected disengagement ended before contact was proven clear" in failed[
        "message"
    ]
    assert "do not jog, rerun place_approach, release the part" in failed["message"]
    assert "You can rerun place_approach" not in failed["message"]
    assert calls["suspended_parts"] == ["MG"]
    assert calls["completion"] == 0
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"

    summary_path = Path(failed["diagnostic_directory"]) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["state"] == "not_confirmed"
    assert summary["qualification_state"] == "not_confirmed"
    assert summary["active"] is False
    assert summary["recovery_required"] is True
    assert summary["insertion_depth_outside_seated_tolerance"] is True
    assert "feedback_trace" not in summary["move_insert_result"]
    status = bridge.digital_twin_move_insert_trial_status(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=failed["trial_id"],
    )
    assert status["state"] == "not_confirmed"
    assert status["qualification_state"] == "not_confirmed"
    assert status["active"] is False
    assert status["recovery_required"] is True


def test_confirm_completion_exposes_non_cancellable_suffix_motion(
    tmp_path: Path,
) -> None:
    async def _exercise() -> None:
        bridge, agent, calls = _supervised_move_insert_trial_bridge(tmp_path)
        trial = await bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
        started = asyncio.Event()
        finish = asyncio.Event()

        async def _slow_completion(
            pre_execute: Any = None,
            /,
            *,
            destination_location: str,
            part_name: str,
            expected_move_insert_result_sha256: str,
        ) -> dict[str, Any]:
            calls["completion"] += 1
            assert not (str(pre_execute() or "") if callable(pre_execute) else "")
            assert destination_location == "assembly_board-v1"
            assert part_name == "MG"
            assert expected_move_insert_result_sha256 == "d" * 64
            started.set()
            await finish.wait()
            agent._current_state = "placed"
            agent._held_part = None
            agent._gripper_state = "open"
            agent._task_ctx = {}
            return {
                "status": "completed",
                "completed_steps": ["release_part", "lift"],
            }

        agent._complete_place_insert_after_move_insert_trial = _slow_completion  # type: ignore[attr-defined]
        confirmation = asyncio.create_task(
            bridge.digital_twin_confirm_move_insert_completion(
                "dual robots",
                "ur5e",
                destination_location="assembly_board-v1",
                part_name="MG",
                trial_id=trial["trial_id"],
                confirmed=True,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=5.0)
        status = bridge.digital_twin_move_insert_trial_status(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id=trial["trial_id"],
        )
        assert status["active"] is False
        assert status["completion_motion_active"] is True
        assert "releasing and lifting" in bridge._move_insert_pending_review_error()
        stop = await bridge.digital_twin_cancel_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id=trial["trial_id"],
        )
        assert stop["success"] is False
        assert "cannot be cancelled with Stop Supervised move_insert" in stop["message"]
        finish.set()
        completed = await asyncio.wait_for(confirmation, timeout=5.0)
        assert completed["qualified"] is True
        assert completed["completion_motion_active"] is False

    asyncio.run(_exercise())


def test_confirm_completion_fails_closed_when_suspension_clear_cannot_persist(
    tmp_path: Path,
) -> None:
    bridge, _agent, calls = _supervised_move_insert_trial_bridge(tmp_path)
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )
    assert bridge._latch_move_insert_suspension("MG", "test suspension") == ""
    bridge._write_move_insert_suspensions = (  # type: ignore[method-assign]
        lambda _suspensions: "simulated durable suspension clear failure"
    )

    confirmed = asyncio.run(
        bridge.digital_twin_confirm_move_insert_completion(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id=trial["trial_id"],
            confirmed=True,
        )
    )

    assert confirmed["success"] is False
    assert confirmed["qualified"] is False
    assert confirmed["released"] is True
    assert confirmed["lifted"] is True
    assert calls["completion"] == 1
    assert "could not be cleared" in confirmed["message"]
    assert bridge._move_insert_suspension_error("MG") == "test suspension"
    assert bridge._ur5e_robot_function_state_uncertain is True


def test_move_insert_release_rejects_tf_that_does_not_advance(
    tmp_path: Path,
) -> None:
    bridge, agent, _calls = _supervised_move_insert_trial_bridge(tmp_path)
    bridge._move_insert_release_feedback_snapshot = (  # type: ignore[method-assign]
        lambda _agent, *, target: (
            {
                "tf_stamp_sec": 10.0,
                "rtde_feedback_timestamp_sec": 21.0,
            },
            "",
        )
    )

    error = bridge._move_insert_release_feedback_advanced(
        agent,
        {"tf_stamp_sec": 10.0, "rtde_feedback_timestamp_sec": 20.0},
        target="dual robots",
        part_name="SG",
        timeout_sec=0.0,
    )

    assert "did not advance" in error
    assert "world -> tool0 TF" in error
    assert "Keep SG clamped" in error


def test_record_move_insert_failure_never_moves_and_suspends_qualification(
    tmp_path: Path,
) -> None:
    bridge, agent, calls = _supervised_move_insert_trial_bridge(
        tmp_path,
        automatic_checks_pass=False,
    )
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )
    assert trial["completion_eligible"] is False
    assert calls["suspended_parts"] == ["MG"]

    recorded = bridge.digital_twin_record_move_insert_failure(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
        note="Gear remained above the pin.",
    )

    assert recorded["success"] is True
    assert recorded["state"] == "failure_recorded"
    assert recorded["failure_id"].startswith("move-insert-failure-")
    assert recorded["recovery_required"] is False
    assert recorded["normal_repair_required"] is False
    assert recorded["hardware_stack_repair_required"] is False
    assert "Confirm Physical Recovery" not in recorded["message"]
    assert "Jog back to pre-insertion" in recorded["message"]
    assert "Delete Previous Recording" in recorded["message"]
    assert Path(recorded["diagnostic_bundle_path"]).is_file()
    assert calls["trial"] == 1
    assert calls["completion"] == 0
    assert calls["qualification_patches"][-1]["qualification"] is None
    assert agent._current_state == "positioned"
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"
    assert agent.calls == []

    repeated = bridge.digital_twin_record_move_insert_failure(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
        note="A different note must not create another failure.",
    )
    assert repeated["success"] is True
    assert repeated["failure_id"] == recorded["failure_id"]
    assert len(calls["qualification_patches"]) == 1


def test_move_insert_recovery_required_survives_bridge_restart(
    tmp_path: Path,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(
        tmp_path,
        automatic_checks_pass=False,
    )
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )
    recorded = bridge.digital_twin_record_move_insert_failure(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
        note="Operator rejected insertion before restarting the UI.",
    )
    assert recorded["recovery_required"] is False
    legacy_recorded = bridge._find_move_insert_trial(
        target="dual robots",
        robot="ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
    )
    assert legacy_recorded is not None
    legacy_recorded = bridge._store_move_insert_trial(
        {
            **legacy_recorded,
            "success": False,
            "recovery_required": True,
            "normal_repair_required": False,
            "hardware_stack_repair_required": False,
            "motion_settled": True,
            "message": (
                "Legacy Record Failure requires Confirm Physical Recovery."
            ),
        }
    )
    assert legacy_recorded["recovery_required"] is True

    restarted, _replacement_agent, _replacement_calls = (
        _supervised_move_insert_trial_bridge(tmp_path)
    )
    restarted._ur5e_robot_function_agent = None
    restarted.resource_agents = []
    restarted._hardware_stack_lifecycle_state = "stopped"
    restarted._hardware_stack_selected = ""
    status = restarted.digital_twin_move_insert_trial_status(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
    )
    assert status["state"] == "failure_recorded"
    assert status["durable_recovered"] is True
    assert status["recovery_required"] is True
    assert status["normal_repair_required"] is False
    assert status["hardware_stack_repair_required"] is False
    assert status["part_clamped"] is True
    assert status["qualified"] is False
    assert "Confirm Physical Recovery" in status["message"]
    assert "requires inspected physical recovery" in (
        restarted._move_insert_pending_review_error()
    )
    pending, pending_error = restarted._read_move_insert_pending_review()
    assert pending_error == ""
    assert pending is not None
    assert pending["trial_id"] == trial["trial_id"]
    assert pending["recovery_required"] is True

    confirmed = restarted.digital_twin_confirm_move_insert_recovery(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
        confirmed=True,
    )
    assert confirmed["success"] is False
    assert confirmed["recovery_required"] is False
    assert confirmed["recovery_confirmed_at"].endswith("Z")
    assert confirmed["recovery_evidence"]["live_rtde_feedback_required"] is False
    assert "without commanding robot motion" in confirmed["message"]


def test_move_insert_restart_relatches_latest_auto_cleared_contact_failure(
    tmp_path: Path,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(
        tmp_path,
        automatic_checks_pass=False,
    )
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )
    current = bridge._find_move_insert_trial(
        target="dual robots",
        robot="ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
    )
    assert current is not None
    auto_cleared = bridge._store_move_insert_trial(
        {
            **current,
            "success": False,
            "active": False,
            "completion_motion_active": False,
            "review_required": False,
            "recovery_required": False,
            "part_clamped": True,
            "released": False,
            "lifted": False,
            "motion_settled": True,
            "automatic_checks_passed": False,
            "completion_eligible": False,
            "qualified": False,
            "recovery_confirmed_at": "",
            "recovery_evidence": {},
            "move_insert_result": {
                **dict(current.get("move_insert_result") or {}),
                "motion_settled": True,
                "disengagement_cycle_count": 1,
                "disengagement_contact_cleared": False,
            },
            "result": {
                **dict(current.get("result") or {}),
                "insert_disengagement_contact_cleared": True,
            },
            "message": (
                "Legacy migration incorrectly cleared recovery_required after "
                "protected disengagement."
            ),
        }
    )
    assert auto_cleared["recovery_required"] is False
    pending, pending_error = bridge._read_move_insert_pending_review()
    assert pending_error == ""
    assert pending is None

    restarted, _replacement_agent, _replacement_calls = (
        _supervised_move_insert_trial_bridge(tmp_path)
    )
    recovered = restarted.digital_twin_move_insert_trial_status(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
    )

    assert recovered["success"] is False
    assert recovered["active"] is False
    assert recovered["recovery_required"] is True
    assert recovered["part_clamped"] is True
    assert recovered["released"] is False
    assert recovered["lifted"] is False
    assert recovered["motion_settled"] is True
    assert recovered["qualified"] is False
    assert "Restart recovery restored recovery_required" in recovered["message"]
    pending, pending_error = restarted._read_move_insert_pending_review()
    assert pending_error == ""
    assert pending is not None
    assert pending["trial_id"] == trial["trial_id"]
    assert pending["recovery_required"] is True
    summary = json.loads(
        (Path(recovered["diagnostic_directory"]) / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["recovery_required"] is True
    with bridge_module.zipfile.ZipFile(
        recovered["diagnostic_bundle_path"]
    ) as archive:
        bundled_summary = json.loads(archive.read("summary.json"))
    assert bundled_summary["recovery_required"] is True


def test_record_move_insert_failure_cannot_clear_recovery_required(
    tmp_path: Path,
) -> None:
    bridge, agent, calls = _supervised_move_insert_trial_bridge(
        tmp_path,
        automatic_checks_pass=False,
    )
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )
    current = bridge._find_move_insert_trial(
        target="dual robots",
        robot="ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
    )
    assert current is not None
    latched = bridge._store_move_insert_trial(
        {
            **current,
            "recovery_required": True,
            "motion_settled": True,
            "part_clamped": True,
            "released": False,
            "lifted": False,
            "message": "Protected disengagement did not prove contact clear.",
        }
    )
    assert latched["recovery_required"] is True

    blocked = bridge.digital_twin_record_move_insert_failure(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
        note="This must not bypass physical recovery.",
    )

    assert blocked["success"] is False
    assert blocked["recovery_required"] is True
    assert blocked["failure_id"] == ""
    assert "cannot clear or replace recovery_required" in blocked["message"]
    status = bridge.digital_twin_move_insert_trial_status(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
    )
    assert status["recovery_required"] is True
    assert status["released"] is False
    assert status["lifted"] is False
    assert calls["completion"] == 0
    assert agent.calls == []


def test_confirm_move_insert_recovery_is_no_motion_and_keeps_suspension(
    tmp_path: Path,
) -> None:
    bridge, agent, calls = _supervised_move_insert_trial_bridge(
        tmp_path,
        automatic_checks_pass=False,
    )
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )
    current = bridge._find_move_insert_trial(
        target="dual robots",
        robot="ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
    )
    assert current is not None
    latched = bridge._store_move_insert_trial(
        {
            **current,
            "recovery_required": True,
            "motion_settled": True,
            "part_clamped": True,
            "released": False,
            "lifted": False,
            "message": "Protected disengagement did not prove contact clear.",
        }
    )
    assert latched["recovery_required"] is True
    recovery_block = bridge._move_insert_pending_review_error()
    assert "UR5e motion and release_part remain blocked" in recovery_block
    assert bridge._move_insert_pending_review_error(
        allow_terminal_recovery=True
    ) == ""
    assert bridge._digital_twin_move_insert_edit_lifecycle_error() == ""
    bridge._move_insert_physical_recovery_readiness = (  # type: ignore[method-assign]
        lambda _target, _agent: pytest.fail(
            "supervised recovery confirmation must not require live RTDE feedback"
        )
    )
    bridge._physical_ur5e_robot_agent = lambda: pytest.fail(  # type: ignore[method-assign]
        "supervised recovery confirmation must not require a reconstructed RobotAgent"
    )
    unconfirmed = bridge.digital_twin_confirm_move_insert_recovery(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
    )
    assert unconfirmed["success"] is False
    assert unconfirmed["recovery_required"] is True
    assert "Explicit operator confirmation" in unconfirmed["message"]

    confirmed = bridge.digital_twin_confirm_move_insert_recovery(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
        confirmed=True,
    )

    assert confirmed["success"] is False
    assert confirmed["state"] == "not_confirmed"
    assert confirmed["qualified"] is False
    assert confirmed["recovery_required"] is False
    assert confirmed["completion_eligible"] is False
    assert confirmed["released"] is False
    assert confirmed["lifted"] is False
    assert confirmed["recovery_confirmed_at"].endswith("Z")
    assert confirmed["recovery_evidence"][
        "operator_confirmed_physical_recovery"
    ] is True
    assert confirmed["recovery_evidence"]["target"] == "dual robots"
    assert confirmed["recovery_evidence"]["robot"] == "ur5e"
    assert confirmed["recovery_evidence"]["destination_location"] == (
        "assembly_board-v1"
    )
    assert confirmed["recovery_evidence"]["part_name"] == "MG"
    assert confirmed["recovery_evidence"]["trial_id"] == trial["trial_id"]
    assert confirmed["recovery_evidence"]["terminal_motion_settled"] is True
    assert confirmed["recovery_evidence"]["terminal_part_clamped"] is True
    assert confirmed["recovery_evidence"]["live_rtde_feedback_required"] is False
    assert "without commanding robot motion" in confirmed["message"]
    assert bridge._move_insert_pending_review_error() == ""
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"
    assert agent._current_state == "positioned"
    assert agent.calls == []
    assert calls["trial"] == 1
    assert calls["completion"] == 0
    assert calls["suspended_parts"] == ["MG"]
    assert bridge._move_insert_suspension_error("MG")
    summary = json.loads(
        (Path(confirmed["diagnostic_directory"]) / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["recovery_required"] is False
    assert summary["success"] is False
    assert summary["qualified"] is False
    assert summary["completion_eligible"] is False
    assert summary["released"] is False
    assert summary["lifted"] is False
    assert summary["recovery_confirmed_at"] == confirmed["recovery_confirmed_at"]
    assert summary["recovery_evidence"] == confirmed["recovery_evidence"]
    with bridge_module.zipfile.ZipFile(
        confirmed["diagnostic_bundle_path"]
    ) as archive:
        bundled_summary = json.loads(archive.read("summary.json"))
    assert bundled_summary["recovery_required"] is False
    assert bundled_summary["recovery_evidence"] == confirmed["recovery_evidence"]


def test_confirm_move_insert_recovery_rejects_a_stale_exact_trial(
    tmp_path: Path,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(
        tmp_path,
        automatic_checks_pass=False,
    )
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )
    current = bridge._find_move_insert_trial(
        target="dual robots",
        robot="ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
    )
    assert current is not None
    current = bridge._store_move_insert_trial(
        {
            **current,
            "recovery_required": True,
            "motion_settled": True,
            "part_clamped": True,
            "released": False,
            "lifted": False,
        }
    )
    newer_trial_id = "move-insert-newer-exact-trial"
    selection_key = bridge._move_insert_trial_selection_key(
        "dual robots",
        "ur5e",
        "assembly_board-v1",
        "MG",
    )
    with bridge._get_move_insert_trial_lock():
        trials, last_by_selection = bridge._move_insert_trial_stores()
        trials[newer_trial_id] = {
            **current,
            "trial_id": newer_trial_id,
            "recovery_required": False,
        }
        last_by_selection[selection_key] = newer_trial_id
    bridge._move_insert_physical_recovery_readiness = (  # type: ignore[method-assign]
        lambda _target, _agent: pytest.fail(
            "a stale trial must not collect recovery evidence"
        )
    )

    blocked = bridge.digital_twin_confirm_move_insert_recovery(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
        confirmed=True,
    )

    assert blocked["success"] is False
    assert blocked["recovery_required"] is True
    assert "exact inactive current supervised move_insert trial" in blocked["message"]
    with bridge._get_move_insert_trial_lock():
        trials, _last_by_selection = bridge._move_insert_trial_stores()
        assert trials[trial["trial_id"]]["recovery_required"] is True


def test_confirm_move_insert_recovery_relatches_when_evidence_cannot_persist(
    tmp_path: Path,
) -> None:
    bridge, agent, _calls = _supervised_move_insert_trial_bridge(
        tmp_path,
        automatic_checks_pass=False,
    )
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )
    current = bridge._find_move_insert_trial(
        target="dual robots",
        robot="ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
    )
    assert current is not None
    bridge._store_move_insert_trial(
        {
            **current,
            "recovery_required": True,
            "motion_settled": True,
            "part_clamped": True,
            "released": False,
            "lifted": False,
        }
    )
    bridge._move_insert_physical_recovery_readiness = (  # type: ignore[method-assign]
        lambda _target, _agent: pytest.fail(
            "supervised recovery confirmation must not require live RTDE feedback"
        )
    )
    bridge._write_move_insert_trial_diagnostics = (  # type: ignore[method-assign]
        lambda candidate: (dict(candidate), "simulated recovery evidence write failure")
    )

    blocked = bridge.digital_twin_confirm_move_insert_recovery(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
        confirmed=True,
    )

    assert blocked["success"] is False
    assert blocked["recovery_required"] is True
    assert blocked["recovery_confirmed_at"] == ""
    assert blocked["recovery_evidence"] == {}
    assert "durable recovery custody could not be cleared" in blocked["message"]
    pending, pending_error = bridge._read_move_insert_pending_review()
    assert pending_error == ""
    assert pending is not None
    assert pending["trial_id"] == trial["trial_id"]
    assert pending["recovery_required"] is True
    assert agent.calls == []


def test_confirm_move_insert_recovery_does_not_require_stationary_fresh_feedback(
    tmp_path: Path,
) -> None:
    bridge, agent, _calls = _supervised_move_insert_trial_bridge(
        tmp_path,
        automatic_checks_pass=False,
    )
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )
    current = bridge._find_move_insert_trial(
        target="dual robots",
        robot="ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
    )
    assert current is not None
    bridge._store_move_insert_trial(
        {
            **current,
            "recovery_required": True,
            "motion_settled": True,
            "normal_repair_required": True,
            "hardware_stack_repair_required": True,
            "part_clamped": True,
            "released": False,
            "lifted": False,
            "message": "Protected disengagement did not prove contact clear.",
        }
    )
    bridge._move_insert_physical_recovery_readiness = (  # type: ignore[method-assign]
        lambda _target, _agent: pytest.fail(
            "supervised recovery confirmation must not require live RTDE feedback"
        )
    )

    confirmed = bridge.digital_twin_confirm_move_insert_recovery(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
        confirmed=True,
    )
    assert confirmed["success"] is False
    assert confirmed["recovery_required"] is False
    assert confirmed["normal_repair_required"] is True
    assert confirmed["hardware_stack_repair_required"] is True
    assert "Repair Hardware Stack remains required" in confirmed["message"]
    assert confirmed["recovery_evidence"]["live_rtde_feedback_required"] is False
    status = bridge.digital_twin_move_insert_trial_status(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
    )
    assert status["recovery_required"] is False
    assert status["normal_repair_required"] is True
    assert status["hardware_stack_repair_required"] is True
    assert status["recovery_confirmed_at"].endswith("Z")
    assert status["recovery_evidence"] == confirmed["recovery_evidence"]
    assert "requires Repair Hardware Stack" in bridge._move_insert_pending_review_error()
    assert bridge._move_insert_pending_review_error(
        allow_terminal_recovery=True
    ) == ""
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"


def test_failed_confirm_after_release_can_be_recorded_without_more_motion(
    tmp_path: Path,
) -> None:
    bridge, agent, calls = _supervised_move_insert_trial_bridge(tmp_path)
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    async def _release_then_fail_lift(
        pre_execute: Any = None,
        /,
        *,
        destination_location: str,
        part_name: str,
        expected_move_insert_result_sha256: str,
    ) -> dict[str, Any]:
        calls["completion"] += 1
        assert not (str(pre_execute() or "") if callable(pre_execute) else "")
        agent._held_part = None
        agent._gripper_state = "open"
        return {
            "status": "failed",
            "content": "release succeeded; lift failed",
            "completed_steps": ["release_part"],
            "failed_step": "lift",
        }

    agent._complete_place_insert_after_move_insert_trial = _release_then_fail_lift  # type: ignore[attr-defined]
    failed = asyncio.run(
        bridge.digital_twin_confirm_move_insert_completion(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id=trial["trial_id"],
            confirmed=True,
        )
    )

    assert failed["success"] is False
    assert failed["released"] is True
    assert failed["lifted"] is False
    assert failed["completion_eligible"] is False
    assert failed["review_required"] is True
    assert "MG was released, but lift did not complete" in failed["message"]
    assert bridge._ur5e_robot_function_state_uncertain is True

    recorded = bridge.digital_twin_record_move_insert_failure(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial["trial_id"],
        note="Lift failed after the irreversible release.",
    )
    assert recorded["success"] is True
    assert recorded["state"] == "failure_recorded"
    assert recorded["recovery_required"] is False
    assert recorded["normal_repair_required"] is True
    assert recorded["hardware_stack_repair_required"] is True
    assert "already released" in recorded["message"]
    assert "Repair Hardware Stack" in recorded["message"]
    assert calls["trial"] == 1
    assert calls["completion"] == 1
    assert agent.calls == []


def test_terminal_diagnostic_failure_blocks_confirm_and_suspends_automatic_use(
    tmp_path: Path,
) -> None:
    bridge, agent, calls = _supervised_move_insert_trial_bridge(tmp_path)
    original_write = bridge._write_move_insert_trial_diagnostics
    writes = 0

    def _fail_terminal_diagnostics(
        trial: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        nonlocal writes
        writes += 1
        if writes >= 2:
            return trial, "simulated terminal diagnostic write failure"
        return original_write(trial)

    bridge._write_move_insert_trial_diagnostics = _fail_terminal_diagnostics  # type: ignore[method-assign]
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert trial["success"] is False
    assert trial["completion_eligible"] is False
    assert "Diagnostics error" in trial["message"]
    assert calls["suspended_parts"] == ["MG"]
    blocked = asyncio.run(
        bridge.digital_twin_confirm_move_insert_completion(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id=trial["trial_id"],
            confirmed=True,
        )
    )
    assert blocked["success"] is False
    assert calls["completion"] == 0
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"


def test_release_lift_evidence_failure_prevents_qualification_write(
    tmp_path: Path,
) -> None:
    bridge, _agent, calls = _supervised_move_insert_trial_bridge(tmp_path)
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )
    original_write = bridge._write_move_insert_trial_diagnostics
    confirm_writes = 0

    def _fail_release_lift_evidence(
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        nonlocal confirm_writes
        confirm_writes += 1
        if confirm_writes == 2:
            return payload, "simulated release/lift evidence persistence failure"
        return original_write(payload)

    bridge._write_move_insert_trial_diagnostics = _fail_release_lift_evidence  # type: ignore[method-assign]
    result = asyncio.run(
        bridge.digital_twin_confirm_move_insert_completion(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id=trial["trial_id"],
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["qualified"] is False
    assert result["released"] is True
    assert result["lifted"] is True
    assert "evidence was not durably committed" in result["message"]
    assert calls["qualification_patches"] == []
    assert calls["completion"] == 1


def test_final_qualification_diagnostic_failure_suspends_and_retains_custody(
    tmp_path: Path,
) -> None:
    bridge, _agent, calls = _supervised_move_insert_trial_bridge(tmp_path)
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )
    original_write = bridge._write_move_insert_trial_diagnostics
    confirm_writes = 0

    def _fail_final_qualification_diagnostics(
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        nonlocal confirm_writes
        confirm_writes += 1
        if confirm_writes == 4:
            return payload, "simulated final qualification diagnostic failure"
        return original_write(payload)

    bridge._write_move_insert_trial_diagnostics = (  # type: ignore[method-assign]
        _fail_final_qualification_diagnostics
    )
    result = asyncio.run(
        bridge.digital_twin_confirm_move_insert_completion(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id=trial["trial_id"],
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["qualified"] is False
    assert result["review_required"] is True
    assert result["released"] is True
    assert result["lifted"] is True
    assert "Final qualification diagnostics could not be persisted" in result["message"]
    assert calls["suspended_parts"] == ["MG"]
    pending, pending_error = bridge._read_move_insert_pending_review()
    assert pending_error == ""
    assert pending is not None
    assert pending["trial_id"] == trial["trial_id"]
    summary = json.loads(
        (Path(result["diagnostic_directory"]) / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["qualified"] is False
    assert summary["review_required"] is True


def test_cancel_move_insert_trial_retains_lock_until_runtime_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_to_thread_inline(monkeypatch)

    async def _exercise() -> None:
        bridge, agent, calls = _supervised_move_insert_trial_bridge(tmp_path)
        started = asyncio.Event()
        finish = asyncio.Event()
        loop = asyncio.get_running_loop()
        settings = bridge._digital_twin_move_insert_trial_settings("MG")
        identities, identities_error = bridge._move_insert_current_identities(
            agent,
            destination_location="assembly_board-v1",
            part_name="MG",
            settings=settings,
        )
        assert identities_error == ""

        async def _immediate_preflight(
            _target: str,
            _robot: str,
            *,
            destination_location: str,
            part_name: str,
            execution_lock_held: bool,
            ignore_pending_trial_id: str = "",
        ) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], str]:
            del execution_lock_held, ignore_pending_trial_id
            assert destination_location == "assembly_board-v1"
            assert part_name == "MG"
            return (
                agent,
                deepcopy(settings),
                deepcopy(identities),
                {"qualified": False},
                "",
            )

        bridge._digital_twin_move_insert_trial_preflight_async = _immediate_preflight  # type: ignore[method-assign]

        async def _slow_trial(
            pre_execute: Any = None,
            /,
            *,
            destination_location: str,
            part_name: str,
            trial_id: str,
        ) -> dict[str, Any]:
            calls["trial"] += 1
            assert not (str(pre_execute() or "") if callable(pre_execute) else "")
            started.set()
            await finish.wait()
            return {
                "status": "failed",
                "trial_id": trial_id,
                "motion_settled": True,
                "content": "move_insert cancelled and settled",
                "move_insert_result": {
                    "success": False,
                    "state_uncertain": False,
                    "final_tool0_pose_valid": True,
                    "engagement_detected": False,
                    "seated_detected": False,
                },
                "move_insert_result_sha256": "f" * 64,
                "trial_ready_for_confirmation": False,
            }

        agent._execute_place_insert_move_insert_trial = _slow_trial  # type: ignore[attr-defined]

        def _cancel_move_insert(*, timeout_sec: float = 8.0) -> dict[str, Any]:
            assert timeout_sec == pytest.approx(8.0)
            loop.call_soon_threadsafe(finish.set)
            return {"success": True, "message": "cancellation accepted"}

        agent._controller.cancel_move_insert = _cancel_move_insert  # type: ignore[attr-defined]
        execution = asyncio.create_task(
            bridge.digital_twin_execute_move_insert_trial(
                "dual robots",
                "ur5e",
                destination_location="assembly_board-v1",
                part_name="MG",
                confirmed=True,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=10.0)
        active = bridge.digital_twin_move_insert_trial_status(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
        assert active["active"] is True
        assert bridge._ur5e_robot_function_execution_lock.locked() is True

        cancelled = await asyncio.wait_for(
            bridge.digital_twin_cancel_move_insert_trial(
                "dual robots",
                "ur5e",
                destination_location="assembly_board-v1",
                part_name="MG",
                trial_id=active["trial_id"],
            ),
            timeout=10.0,
        )
        settled = await asyncio.wait_for(execution, timeout=10.0)

        assert cancelled["success"] is True
        assert settled["cancelled"] is True
        assert settled["active"] is False
        assert bridge._ur5e_robot_function_execution_lock.locked() is False
        assert calls["completion"] == 0
        assert agent._held_part == "MG"
        assert agent._gripper_state == "closed"

    asyncio.run(_exercise())


def test_trial_store_serializes_cancel_and_terminal_settlement(
    tmp_path: Path,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(tmp_path)
    initial = bridge._store_move_insert_trial(
        {
            "success": True,
            "target": "dual robots",
            "robot": "ur5e",
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
            "trial_id": "move-insert-trial-cancel-race",
            "active": True,
            "review_required": True,
            "completion_eligible": False,
            "message": "Testing.",
        }
    )
    cancel_copy = deepcopy(initial)
    cancel_copy.update(
        {"active": True, "cancellation_requested": True, "message": "Stopping."}
    )
    terminal_copy = deepcopy(initial)
    terminal_copy.update(
        {
            "active": False,
            "cancelled": True,
            "completion_eligible": False,
            "message": "Cancelled and settled.",
        }
    )
    original_write = bridge._write_move_insert_trial_diagnostics
    cancel_write_started = threading.Event()
    permit_cancel_write = threading.Event()

    def _barrier_write(
        trial: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        if trial.get("cancellation_requested") and trial.get("active"):
            cancel_write_started.set()
            assert permit_cancel_write.wait(timeout=5.0)
        return original_write(trial)

    bridge._write_move_insert_trial_diagnostics = _barrier_write  # type: ignore[method-assign]
    cancel_thread = threading.Thread(
        target=bridge._store_move_insert_trial,
        args=(cancel_copy,),
    )
    terminal_thread = threading.Thread(
        target=bridge._store_move_insert_trial,
        args=(terminal_copy,),
    )
    cancel_thread.start()
    assert cancel_write_started.wait(timeout=5.0)
    terminal_thread.start()
    assert terminal_thread.is_alive()
    permit_cancel_write.set()
    cancel_thread.join(timeout=5.0)
    terminal_thread.join(timeout=5.0)
    assert not cancel_thread.is_alive()
    assert not terminal_thread.is_alive()

    final = bridge._find_move_insert_trial(
        target="dual robots",
        robot="ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id="move-insert-trial-cancel-race",
    )
    assert final is not None
    assert final["active"] is False
    assert final["cancelled"] is True
    assert final["message"] == "Cancelled and settled."


def test_record_failure_wins_race_before_confirm_and_prevents_release(
    tmp_path: Path,
) -> None:
    bridge, agent, calls = _supervised_move_insert_trial_bridge(tmp_path)
    trial = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )
    record_patch_started = threading.Event()
    permit_record_patch = threading.Event()
    qualification_writes: list[Any] = []

    def _barrier_patch(**kwargs: Any) -> dict[str, Any]:
        qualification_writes.append(deepcopy(kwargs.get("qualification")))
        if kwargs.get("qualification") is None:
            record_patch_started.set()
            assert permit_record_patch.wait(timeout=5.0)
        return {
            "success": True,
            "changed": True,
            "profile_sha256": kwargs.get("expected_profile_sha256"),
        }

    bridge._patch_move_insert_qualification = _barrier_patch  # type: ignore[method-assign]
    record_results: list[dict[str, Any]] = []
    record_thread = threading.Thread(
        target=lambda: record_results.append(
            bridge.digital_twin_record_move_insert_failure(
                "dual robots",
                "ur5e",
                destination_location="assembly_board-v1",
                part_name="MG",
                trial_id=trial["trial_id"],
                note="Operator rejected seating.",
            )
        )
    )
    record_thread.start()
    assert record_patch_started.wait(timeout=5.0)

    blocked_confirm = asyncio.run(
        bridge.digital_twin_confirm_move_insert_completion(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            trial_id=trial["trial_id"],
            confirmed=True,
        )
    )
    permit_record_patch.set()
    record_thread.join(timeout=5.0)

    assert blocked_confirm["success"] is False
    assert calls["completion"] == 0
    assert qualification_writes == [None]
    assert record_results[0]["state"] == "failure_recorded"
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"


def test_pending_move_insert_review_blocks_competing_control_paths(
    tmp_path: Path,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(tmp_path)
    pending = bridge._store_move_insert_trial(
        {
            "success": True,
            "ready": False,
            "target": "dual robots",
            "robot": "ur5e",
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
            "trial_id": "move-insert-pending-review",
            "active": False,
            "review_required": True,
            "completion_eligible": True,
            "qualified": False,
            "message": "Awaiting visual confirmation.",
        }
    )
    assert pending["review_required"] is True
    expected = bridge._move_insert_pending_review_error()
    assert "awaits Confirm Completion" in expected
    assert "Delete Previous Recording" in expected
    assert expected in bridge._ur5e_robot_function_agent_handoff_error()
    assert bridge._digital_twin_assembly_lifecycle_error() == expected
    assert bridge._digital_twin_move_insert_edit_lifecycle_error() == expected

    assembly = asyncio.run(
        bridge.digital_twin_assembly_readiness(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
        )
    )
    assert assembly["ready"] is False
    assert assembly["message"] == expected
    function = asyncio.run(
        bridge.digital_twin_execute_robot_function(
            "dual robots",
            "ur5e",
            "move_home",
            confirmed=True,
        )
    )
    assert function["success"] is False
    assert function["message"] == expected

    bridge.teleop_target = lambda *_args, **_kwargs: {
        "environment": "real",
        "ready": True,
        "warning": "",
    }
    assert bridge._teleop_preflight("ur5e", "cartesian")["warning"] == expected
    mode_ok, mode_message = bridge.teleop_cartesian_mode("ur5e", "smooth")
    assert mode_ok is False
    assert mode_message == expected
    off_ok, _off_message = bridge.teleop_cartesian_mode("ur5e", "off")
    assert off_ok is True
    reset_ok, reset_message = bridge.teleop_reset_ur5e_rtde_connection()
    assert reset_ok is False
    assert reset_message == expected


@pytest.mark.parametrize(
    "active_operation",
    ["Supervised Test move_insert", "Confirm Completion"],
)
def test_active_supervised_move_insert_motion_blocks_hardware_stack_lifecycle(
    tmp_path: Path,
    active_operation: str,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(tmp_path)
    bridge._hardware_stack_for_robot = lambda _robot: ("hardware_ur5e_rtde",)
    bridge._ur5e_robot_function_execution_active = active_operation
    bridge._ur5e_robot_function_execution_lock.acquire()
    try:
        start_error = bridge.ros2_start_hardware_stack("ur5e")
        stop_error = bridge.ros2_stop_hardware_stack("ur5e")
        repair_error = bridge.ros2_repair_hardware_stack("ur5e")
    finally:
        bridge._ur5e_robot_function_execution_lock.release()

    for lifecycle_error in (start_error, stop_error, repair_error):
        assert "motion is active" in str(lifecycle_error).lower()
        assert active_operation in str(lifecycle_error)


@pytest.mark.parametrize(
    ("rtde_status", "message_fragment"),
    [
        (
            {
                "insertion_demonstration_active": True,
                "insertion_demonstration_recording_id": "recording-live",
            },
            "insertion demonstration recording-live is active",
        ),
        (
            {
                "state": "executing",
                "trial_id": "move-insert-live",
                "insert_phase": "searching",
            },
            "supervised move_insert move-insert-live is active",
        ),
    ],
)
def test_server_active_insertion_blocks_teardown_but_not_start_from_stopped(
    tmp_path: Path,
    rtde_status: dict[str, Any],
    message_fragment: str,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(tmp_path)
    bridge.ros2_proc_status = lambda process_name: (
        "running"
        if process_name == "hardware_ur5e_rtde_trajectory_server"
        else "stopped"
    )
    bridge._ur5e_rtde_trajectory_status = lambda: deepcopy(rtde_status)

    assert bridge._hardware_stack_lifecycle_motion_error("start", "ur5e") == ""
    for operation in ("stop", "repair"):
        error = bridge._hardware_stack_lifecycle_motion_error(operation, "ur5e")
        assert message_fragment in error


def test_hardware_stack_start_and_stop_hold_robot_execution_lock(
    tmp_path: Path,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(tmp_path)
    bridge._store_move_insert_trial(
        {
            "success": False,
            "target": "ur5e only",
            "robot": "ur5e",
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
            "trial_id": "move-insert-terminal-does-not-own-lifecycle",
            "active": False,
            "completion_motion_active": False,
            "review_required": True,
            "normal_repair_required": True,
            "hardware_stack_repair_required": True,
            "part_clamped": True,
            "qualified": False,
            "message": "Retained move_insert result.",
        }
    )
    bridge._hardware_stack_for_robot = lambda _robot: ("hardware_ur5e_rtde",)
    bridge._hardware_stack_lifecycle_lock = threading.Lock()
    bridge._hardware_stack_selected = ""
    bridge._hardware_stack_lifecycle_state = "stopped"
    bridge._hardware_stack_lifecycle_generation = 0
    bridge._hardware_stack_stationary_results = {}
    bridge._hardware_stack_cartesian_jog_reset_results = {}
    bridge._hardware_stack_last_error = ""
    bridge._hardware_stack_failed_process = ""
    bridge._hardware_stack_failed_return_code = None
    bridge._hardware_stack_process_log_path = ""
    observed: list[str] = []

    def _start_locked(_key: str) -> None:
        assert bridge._ur5e_robot_function_execution_lock.locked()
        observed.append("start")
        bridge._hardware_cartesian_readiness_states()["ur5e"] = {
            "cartesian_jog_ready": True,
            "cartesian_function_ready": True,
            "generation": bridge._hardware_stack_lifecycle_generation,
            "message": "fresh Cartesian readiness",
        }
        return None

    def _stop_locked(_key: str) -> None:
        assert bridge._ur5e_robot_function_execution_lock.locked()
        observed.append("stop")
        return None

    def _smooth_stop_locked(_key: str) -> str:
        assert bridge._ur5e_robot_function_execution_lock.locked()
        observed.append("smooth_stop")
        return "No Cartesian Smooth Hold session was active"

    bridge._ros2_start_hardware_stack_locked = _start_locked  # type: ignore[method-assign]

    def _validated_processes(_key: str) -> None:
        bridge._hardware_stack_validated_process_pids = {
            "hardware_ur5e_rtde": 4321
        }
        return None

    def _stationary(robot: str, *, ros_domain_id: int) -> None:
        bridge._hardware_stack_stationary_results[robot] = {
            "generation": bridge._hardware_stack_lifecycle_generation,
            "stationary_ready": True,
            "message": "stationary feedback ready",
            "diagnostics": {},
        }
        return None

    bridge._record_hardware_stack_validated_processes = _validated_processes
    bridge._validate_hardware_stationary = _stationary
    bridge._ros2_stop_hardware_stack_locked = _stop_locked  # type: ignore[method-assign]
    bridge._stop_cartesian_smooth_for_repair = _smooth_stop_locked  # type: ignore[method-assign]

    assert bridge.ros2_start_hardware_stack("ur5e") is None
    assert bridge.ros2_stop_hardware_stack("ur5e") is None
    assert observed == ["start", "smooth_stop", "stop"]
    assert bridge._ur5e_robot_function_execution_lock.locked() is False


@pytest.mark.parametrize("stationary_ready", [False, True])
def test_hardware_stack_repair_does_not_mutate_terminal_move_insert_trial(
    tmp_path: Path,
    stationary_ready: bool,
) -> None:
    bridge, _agent, _calls = _supervised_move_insert_trial_bridge(tmp_path)
    trial_id = "move-insert-terminal-repair"
    bridge._store_move_insert_trial(
        {
            "success": False,
            "ready": False,
            "target": "ur5e only",
            "robot": "ur5e",
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
            "trial_id": trial_id,
            "active": False,
            "completion_motion_active": False,
            "review_required": False,
            "recovery_required": False,
            "normal_repair_required": True,
            "hardware_stack_repair_required": True,
            "part_clamped": True,
            "released": False,
            "lifted": False,
            "automatic_checks_passed": False,
            "completion_eligible": False,
            "qualified": False,
            "failure_recorded": True,
            "status": "failed",
            "hardware_stack_repair_reason": "terminal force failure",
            "message": "Terminal force failure; Repair Hardware Stack required.",
        }
    )
    bridge._ur5e_robot_function_state_uncertain = True
    bridge._ur5e_robot_function_state_uncertain_reason = "terminal force failure"
    bridge._hardware_stack_for_robot = lambda _robot: ("hardware_ur5e_rtde",)
    bridge._hardware_stack_lifecycle_lock = threading.Lock()
    bridge._hardware_stack_selected = ""
    bridge._hardware_stack_lifecycle_state = "failed"
    bridge._hardware_stack_lifecycle_generation = 0
    bridge._hardware_stack_stationary_results = {}
    bridge._hardware_stack_cartesian_jog_reset_results = {}
    bridge._hardware_stack_last_error = "terminal force failure"
    bridge._hardware_stack_failed_process = "hardware_ur5e_rtde"
    bridge._hardware_stack_failed_return_code = 1
    bridge._hardware_stack_process_log_path = "/tmp/ur5e.log"
    bridge._default_ros_domain_id = lambda: 42
    bridge._stop_cartesian_smooth_for_repair = (
        lambda _robot: "No Cartesian Smooth Hold session was active"
    )
    bridge._ros2_stop_hardware_stack_locked = lambda _robot: None

    def _start(_robot: str) -> None:
        generation = bridge._hardware_stack_lifecycle_generation
        bridge._hardware_cartesian_readiness_states()["ur5e"] = {
            "cartesian_jog_ready": True,
            "cartesian_function_ready": True,
            "generation": generation,
            "message": "fresh Cartesian readiness",
        }
        return None

    bridge._ros2_start_hardware_stack_locked = _start

    def _validated_processes(_robot: str) -> None:
        bridge._hardware_stack_validated_process_pids = {
            "hardware_ur5e_rtde": 4321
        }
        return None

    bridge._record_hardware_stack_validated_processes = _validated_processes

    def _stationary(robot: str, *, ros_domain_id: int) -> str | None:
        assert ros_domain_id == 42
        bridge._hardware_stack_stationary_results[robot] = {
            "generation": bridge._hardware_stack_lifecycle_generation,
            "stationary_ready": stationary_ready,
            "message": (
                "stationary feedback ready"
                if stationary_ready
                else "stationary feedback unavailable"
            ),
            "diagnostics": {},
        }
        return None if stationary_ready else "stationary feedback unavailable"

    bridge._validate_hardware_stationary = _stationary

    repair_error = bridge.ros2_repair_hardware_stack("ur5e")
    status = bridge.digital_twin_move_insert_trial_status(
        "ur5e only",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=trial_id,
    )

    if not stationary_ready:
        assert "stationary feedback validation failed" in str(repair_error)
        assert status["normal_repair_required"] is True
        assert status["hardware_stack_repair_required"] is True
        assert bridge._ur5e_robot_function_state_uncertain is True
        assert "repair_evidence" not in status["result"]
        return
    assert repair_error is None
    assert status["normal_repair_required"] is False
    assert status["hardware_stack_repair_required"] is False
    assert status["qualified"] is False
    assert status["part_clamped"] is True
    assert status["released"] is False
    assert status["lifted"] is False
    assert bridge._ur5e_robot_function_state_uncertain is False
    assert status["result"]["repair_evidence"]["hardware_stack"] == "ur5e"
    assert status["result"]["repair_evidence"][
        "hardware_stack_generation"
    ] == bridge._hardware_stack_lifecycle_generation
    assert status["result"]["repair_evidence"]["stationary_feedback"][
        "stationary_ready"
    ] is True
    assert status["result"]["repair_evidence"]["cartesian_readiness"][
        "cartesian_jog_ready"
    ] is True
    assert bridge._move_insert_pending_review_error() == ""


@pytest.mark.parametrize(
    ("robot", "part_name", "message_fragment"),
    [
        ("xarm6", "MG", "exact robot ur5e"),
        ("ur5e", "MRP", "angular alignment"),
    ],
)
def test_move_insert_trial_rejects_xarm6_and_parts_not_enabled_first(
    robot: str,
    part_name: str,
    message_fragment: str,
) -> None:
    bridge = _ready_assembly_bridge()
    bridge._digital_twin_robot_function_request_error = (
        lambda *_args, **_kwargs: ({}, "")
    )

    result = asyncio.run(
        bridge.digital_twin_move_insert_trial_readiness(
            "dual robots",
            robot,
            destination_location="assembly_board-v1",
            part_name=part_name,
        )
    )

    assert result["ready"] is False
    assert message_fragment in result["message"]


@pytest.mark.parametrize("part_name", ("SG", "MG", "LG", "SCP", "MCP", "LCP"))
def test_move_insert_trial_accepts_each_exact_supported_ur5e_part(
    part_name: str,
) -> None:
    bridge = _ready_assembly_bridge()
    bridge._digital_twin_robot_function_request_error = (
        lambda *_args, **_kwargs: ({}, "")
    )

    assert (
        bridge._move_insert_trial_request_error(
            "dual robots",
            "ur5e",
            "assembly_board-v1",
            part_name,
        )
        == ""
    )


@pytest.mark.parametrize("part_name", ("SG", "MG", "LG", "SCP", "MCP", "LCP"))
def test_supervised_move_insert_keeps_changed_place_approach_sha_as_evidence(
    part_name: str,
) -> None:
    recipe_sha256 = "2" * 64
    current_sha256 = "7" * 64
    settings = _move_insert_settings(part_name)
    settings["effective"] = {
        **dict(settings["effective"]),
        "demonstration_recipe": {
            "recording_id": f"recording-{part_name}",
            "demonstration_sha256": "5" * 64,
            "place_approach_recording_sha256": recipe_sha256,
        },
    }
    identities = {
        "robot": "ur5e",
        "tool_frame": "tool0",
        "destination_location": "assembly_board-v1",
        "part_name": part_name,
        "profile_sha256": "1" * 64,
        "hard_caps_sha256": "3" * 64,
        "place_approach_recording_sha256": current_sha256,
        "board_calibration_id": "ur5e-calibration",
        "board_geometry_sha256": "4" * 64,
    }

    payload, identity_sha256, error = (
        SystemBridge._move_insert_qualification_identity(
            identities=identities,
            move_insert_effective=settings["effective"],
        )
    )

    assert error == ""
    assert identity_sha256
    assert payload["place_approach_recording_sha256"] == current_sha256
    assert settings["effective"]["demonstration_recipe"][
        "place_approach_recording_sha256"
    ] == recipe_sha256


def test_supervised_move_insert_changed_place_approach_sha_dispatches(
    tmp_path: Path,
) -> None:
    bridge, agent, calls = _supervised_move_insert_trial_bridge(tmp_path)
    settings = deepcopy(bridge._digital_twin_move_insert_trial_settings("MG"))
    settings["effective"]["demonstration_recipe"][
        "place_approach_recording_sha256"
    ] = "3" * 64
    bridge._digital_twin_move_insert_trial_settings = (
        lambda _part_name: deepcopy(settings)
    )
    agent._task_ctx["move_insert_profile"] = deepcopy(settings["effective"])

    result = asyncio.run(
        bridge.digital_twin_execute_move_insert_trial(
            "dual robots",
            "ur5e",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is True
    assert result["state"] == "awaiting_visual_confirmation"
    assert calls["trial"] == 1
    assert calls["move_insert_dispatched"] == 1
    assert agent.calls == []
    stored = bridge._find_move_insert_trial(
        target="dual robots",
        robot="ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        trial_id=result["trial_id"],
    )
    assert stored is not None
    assert stored["identities"]["place_approach_recording_sha256"] == "b" * 64
    assert stored["move_insert_effective"]["demonstration_recipe"][
        "place_approach_recording_sha256"
    ] == "3" * 64


def test_move_insert_qualification_rejects_identity_changes_but_not_board_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, resource_path, _hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    qualification = resource["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]["qualifications"]["MG"]
    identities = {
        key: deepcopy(qualification[key])
        for key in (
            "robot",
            "destination_location",
            "part_name",
            "tool_frame",
            "profile_sha256",
            "hard_caps_sha256",
            "place_approach_recording_sha256",
                "board_calibration_id",
                "board_geometry_sha256",
                "recording_id",
                "demonstration_sha256",
            )
    }
    identities["board_generation"] = 999

    current = bridge._move_insert_qualification_view(
        part_name="MG",
        identities=identities,
    )
    assert current["qualified"] is True
    assert current["qualification_mismatch_fields"] == []

    stale_identities = {**identities, "board_geometry_sha256": "d" * 64}
    stale = bridge._move_insert_qualification_view(
        part_name="MG",
        identities=stale_identities,
    )
    assert stale["qualified"] is False
    assert stale["qualification_mismatch_fields"] == [
        "board_geometry_sha256"
    ]

    changed_hard_caps = {
        **identities,
        "hard_caps_sha256": "e" * 64,
    }
    stale = bridge._move_insert_qualification_view(
        part_name="MG",
        identities=changed_hard_caps,
    )
    assert stale["qualified"] is False
    assert stale["qualification_mismatch_fields"] == ["hard_caps_sha256"]

    changed_profile = {
        **identities,
        "profile_sha256": "f" * 64,
    }
    stale = bridge._move_insert_qualification_view(
        part_name="MG",
        identities=changed_profile,
    )
    assert stale["qualified"] is False
    assert stale["qualification_mismatch_fields"] == ["profile_sha256"]

    bridge._move_insert_offline_qualification_identities = (
        lambda **_kwargs: (deepcopy(stale_identities), "")
    )
    readiness, error = bridge._move_insert_normal_qualification_readiness(
        destination_location="assembly_board-v1",
        part_name="MG",
        settings={"profile_sha256": qualification["profile_sha256"]},
    )
    assert readiness["qualified"] is False
    assert "board_geometry_sha256" in error


def test_move_insert_qualification_identity_binds_exact_tool_frame() -> None:
    identities = {
        "robot": "ur5e",
        "destination_location": "assembly_board-v1",
        "part_name": "MG",
        "tool_frame": "tool0",
        "profile_sha256": "a" * 64,
        "hard_caps_sha256": "b" * 64,
        "place_approach_recording_sha256": "c" * 64,
        "board_calibration_id": "ur5e-calibration",
        "board_geometry_sha256": "d" * 64,
    }
    effective = {
        "demonstration_recipe": {
            "recording_id": "insertion-demonstration-test",
            "demonstration_sha256": "e" * 64,
        }
    }

    payload, identity_sha256, error = (
        SystemBridge._move_insert_qualification_identity(
            identities=identities,
            move_insert_effective=effective,
        )
    )
    changed_payload, changed_sha256, changed_error = (
        SystemBridge._move_insert_qualification_identity(
            identities={**identities, "tool_frame": "xarm6_link6"},
            move_insert_effective=effective,
        )
    )
    _missing_payload, missing_sha256, missing_error = (
        SystemBridge._move_insert_qualification_identity(
            identities={**identities, "tool_frame": ""},
            move_insert_effective=effective,
        )
    )

    assert error == changed_error == ""
    assert payload["tool_frame"] == "tool0"
    assert changed_payload["tool_frame"] == "xarm6_link6"
    assert identity_sha256 != changed_sha256
    assert missing_sha256 == ""
    assert "tool_frame" in missing_error


def test_saved_insertion_demonstration_does_not_load_another_robots_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording_id = "insertion-demonstration-robot-isolation"
    resource = _move_insert_resource()
    profile = resource["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]
    profile["demonstration_recipes"] = {
        "MG": {
            "robot": "ur5e",
            "part_name": "MG",
            "recording_id": recording_id,
        }
    }
    resource_path = tmp_path / "robot_ur5e.json"
    resource_path.write_text(json.dumps(resource), encoding="utf-8")
    monkeypatch.setattr(bridge_module, "_UR5E_RESOURCE", resource_path)
    directory = tmp_path / "demonstrations" / recording_id
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "recording_id": recording_id,
                "robot": "xarm6",
                "part_name": "MG",
            }
        ),
        encoding="utf-8",
    )
    bridge = object.__new__(SystemBridge)
    bridge._insertion_demonstrations_dir = tmp_path / "demonstrations"

    assert bridge._saved_insertion_demonstration("MG") is None


def test_standalone_place_approach_skips_stale_move_insert_qualification() -> None:
    bridge = object.__new__(SystemBridge)
    agent = _agent_for("place_approach")
    normal_settings = _move_insert_settings("MG")
    trial_settings = {**_move_insert_settings("MG"), "profile_state": "trial_ready"}
    bridge.digital_twin_move_insert_settings = lambda *_args, **_kwargs: deepcopy(
        normal_settings
    )
    bridge._digital_twin_move_insert_trial_settings = lambda _part_name: deepcopy(
        trial_settings
    )
    bridge._move_insert_normal_qualification_readiness = (
        lambda **_kwargs: (
            {
                "qualified": False,
                "qualification_mismatch_fields": [
                    "place_approach_recording_sha256"
                ],
            },
            (
                "move_insert qualification for MG is stale: "
                "place_approach_recording_sha256"
            ),
        )
    )

    approach_readiness, approach_error = (
        bridge._digital_twin_robot_function_move_insert_readiness(
            "dual robots",
            "ur5e",
            "place_approach",
            "assembly_board-v1",
            "MG",
            agent,
            {},
        )
    )
    assert approach_error == ""
    assert approach_readiness == {}

    insert_readiness, insert_error = (
        bridge._digital_twin_robot_function_move_insert_readiness(
            "dual robots",
            "ur5e",
            "place_insert",
            "assembly_board-v1",
            "MG",
            agent,
            {},
        )
    )
    assert insert_readiness["qualified"] is False
    assert "place_approach_recording_sha256" in insert_error


def test_failed_qualification_removal_latches_normal_move_insert_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _resource_path, _hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )

    def _failed_write(_resource: dict[str, Any]) -> None:
        raise OSError("simulated atomic replacement failure")

    bridge._atomic_write_move_insert_resource = _failed_write  # type: ignore[method-assign]
    error = bridge._suspend_move_insert_qualification("MG")

    assert "simulated atomic replacement failure" in error
    suspension = bridge._move_insert_suspension_error("MG")
    assert "Automatic move_insert for MG is suspended" in suspension
    settings = bridge.digital_twin_move_insert_settings(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    assert settings["validated"] is False
    assert settings["profile_state"] == "suspended"
    assert settings["message"] == suspension
    readiness, readiness_error = bridge._move_insert_normal_qualification_readiness(
        destination_location="assembly_board-v1",
        part_name="MG",
        settings=settings,
    )
    assert readiness["qualified"] is False
    assert readiness_error == suspension
    durable_payload = json.loads(
        bridge._move_insert_suspensions_path().read_text(encoding="utf-8")
    )
    durable_record = durable_payload["suspensions"]["MG"]
    assert durable_record["robot"] == "ur5e"
    assert durable_record["destination_location"] == "assembly_board-v1"
    assert durable_record["part_name"] == "MG"
    assert durable_record["reason"] == suspension
    assert durable_record["suspended_at"].endswith("Z")

    durable_record["reason"] = (
        "Automatic move_insert for MG is suspended after a failed or rejected "
        "insertion. Complete another supervised trial."
    )
    bridge._move_insert_suspensions_path().write_text(
        json.dumps(durable_payload),
        encoding="utf-8",
    )
    restarted = object.__new__(SystemBridge)
    restarted._move_insert_trials_dir = bridge._move_insert_trials_dir
    restarted._move_insert_suspended_parts = {}
    assert restarted._move_insert_suspension_error("MG") == (
        "Automatic move_insert for MG is suspended after a failed or rejected "
        "insertion. Complete one successful supervised trial."
    )


def test_suspension_write_failure_keeps_process_latch_and_reports_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _resource_path, _hardware_config = _move_insert_settings_bridge(
        tmp_path,
        monkeypatch,
    )
    bridge._write_move_insert_suspensions = (  # type: ignore[method-assign]
        lambda _suspensions: "simulated suspension storage failure"
    )

    def _failed_manifest_write(_resource: dict[str, Any]) -> None:
        raise OSError("simulated qualification removal failure")

    bridge._atomic_write_move_insert_resource = _failed_manifest_write  # type: ignore[method-assign]
    error = bridge._suspend_move_insert_qualification("MG")

    assert "simulated suspension storage failure" in error
    assert "simulated qualification removal failure" in error
    assert "Automatic move_insert for MG is suspended" in (
        bridge._move_insert_suspension_error("MG")
    )


@pytest.mark.parametrize("assembly", [False, True])
def test_automatic_place_insert_failure_suspends_qualification(
    assembly: bool,
) -> None:
    agent = _PhysicalUR5eAgent() if assembly else _agent_for("place_insert")
    bridge = _ready_assembly_bridge(agent) if assembly else _ready_bridge(agent)
    suspended: list[str] = []
    bridge._suspend_move_insert_qualification = (
        lambda part_name: suspended.append(part_name) or ""
    )

    async def _failed_place_insert(**kwargs: Any) -> dict[str, Any]:
        agent.calls.append(("place_insert", kwargs))
        return {
            "status": "failed",
            "content": "move_insert did not engage",
            "failed_step": "move_insert",
        }

    agent.place_insert = _failed_place_insert  # type: ignore[method-assign]
    agent.executables["place_insert"] = _failed_place_insert
    if assembly:
        result = asyncio.run(
            bridge.digital_twin_execute_assembly(
                "dual robots",
                "ur5e",
                origin_resource_location="prusa-mk4-2",
                destination_location="assembly_board-v1",
                part_name="MG",
                confirmed=True,
            )
        )
        assert result["failed_function"] == "place_insert"
    else:
        result = asyncio.run(
            bridge.digital_twin_execute_robot_function(
                "dual robots",
                "ur5e",
                "place_insert",
                destination_location="assembly_board-v1",
                part_name="MG",
                confirmed=True,
            )
        )
    assert result["success"] is False
    assert suspended == ["MG"]


@pytest.mark.parametrize("assembly", [False, True])
def test_automatic_place_insert_pre_dispatch_failure_keeps_qualification(
    assembly: bool,
) -> None:
    agent = _PhysicalUR5eAgent() if assembly else _agent_for("place_insert")
    bridge = _ready_assembly_bridge(agent) if assembly else _ready_bridge(agent)
    suspended: list[str] = []
    bridge._suspend_move_insert_qualification = (
        lambda part_name: suspended.append(part_name) or ""
    )

    async def _blocked_before_move_insert(**kwargs: Any) -> dict[str, Any]:
        agent.calls.append(("place_insert", kwargs))
        return {
            "status": "failed",
            "content": (
                "place_insert move_insert boundary is not ready: place_approach "
                "move_insert hard_caps_sha256 is missing"
            ),
            "failure_context": {
                "observations": {
                    "step": "place_insert.assembly_board_v1_aruco_generation_lock",
                    "move_insert_dispatched": False,
                }
            },
        }

    agent.place_insert = _blocked_before_move_insert  # type: ignore[method-assign]
    agent.executables["place_insert"] = _blocked_before_move_insert
    if assembly:
        result = asyncio.run(
            bridge.digital_twin_execute_assembly(
                "dual robots",
                "ur5e",
                origin_resource_location="prusa-mk4-2",
                destination_location="assembly_board-v1",
                part_name="MG",
                confirmed=True,
            )
        )
        assert result["failed_function"] == "place_insert"
    else:
        result = asyncio.run(
            bridge.digital_twin_execute_robot_function(
                "dual robots",
                "ur5e",
                "place_insert",
                destination_location="assembly_board-v1",
                part_name="MG",
                confirmed=True,
            )
        )

    assert result["success"] is False
    assert result["move_insert_dispatched"] is False
    assert "place_insert was not dispatched" in result["message"]
    assert suspended == []
    assert bridge._ur5e_robot_function_state_uncertain is False


def test_assembly_place_insert_post_state_mismatch_suspends_qualification() -> None:
    agent = _PhysicalUR5eAgent()
    bridge = _ready_assembly_bridge(agent)
    suspended: list[str] = []
    bridge._suspend_move_insert_qualification = (
        lambda part_name: suspended.append(part_name) or ""
    )

    async def _completed_without_place_effects(**kwargs: Any) -> dict[str, Any]:
        agent.calls.append(("place_insert", kwargs))
        return {
            "status": "completed",
            "content": "reported completed without applying place effects",
            "completed_steps": ["move_insert", "release_part", "lift"],
        }

    agent.place_insert = _completed_without_place_effects  # type: ignore[method-assign]
    agent.executables["place_insert"] = _completed_without_place_effects
    result = asyncio.run(
        bridge.digital_twin_execute_assembly(
            "dual robots",
            "ur5e",
            origin_resource_location="prusa-mk4-2",
            destination_location="assembly_board-v1",
            part_name="MG",
            confirmed=True,
        )
    )

    assert result["success"] is False
    assert result["failed_function"] == "place_insert"
    assert suspended == ["MG"]
    assert [name for name, _kwargs in agent.calls][-1] == "place_insert"
    assert all(name != "move_home" for name, _kwargs in agent.calls)


class _InsertionDemonstrationController:
    def __init__(self) -> None:
        self.recording_id = ""
        self.sample_count = 0
        self.pose = {
            "x": 0.4,
            "y": 0.2,
            "z": 0.48,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
        self.trace_root: Path | None = None

    @staticmethod
    def _ensure_insertion_demonstration_client_ready(
        *, timeout_sec: float = 2.0
    ) -> tuple[bool, str]:
        assert timeout_sec == 2.0
        return True, ""

    def start_insertion_demonstration(self, **kwargs: Any) -> dict[str, Any]:
        self.recording_id = str(kwargs["recording_id"])
        return {
            "success": True,
            "active": True,
            "recording_id": self.recording_id,
            "phase": "starting",
            "sample_count": 0,
        }

    def insertion_demonstration_status(self) -> dict[str, Any]:
        self.sample_count += 8
        return {
            "success": True,
            "active": True,
            "recording_id": self.recording_id,
            "phase": "recording_insertion",
            "sample_count": self.sample_count,
            "actual_tool0_pose": deepcopy(self.pose),
            "actual_tcp_force": [0.0, 0.0, 8.0, 0.0, 0.0, 0.2],
            "actual_tcp_speed": [0.0] * 6,
            "baseline_valid": True,
            "force_bias": [0.0, 0.0, 1.0, 0.0, 0.0, 0.1],
            "updated_at": time.time(),
        }

    def stop_insertion_demonstration(self, *, timeout_sec: float) -> dict[str, Any]:
        assert timeout_sec == 12.0
        assert self.trace_root is not None
        trace_path = self.trace_root / f"{self.recording_id}.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as handle:
            for sample_index in range(max(self.sample_count + 8, 32)):
                baseline = sample_index < 6
                insertion_fraction = (
                    0.0
                    if baseline
                    else min(1.0, (sample_index - 5) / 18.0)
                )
                world_tool0_pose = {
                    **deepcopy(self.pose),
                    "z": 0.5 - 0.02 * insertion_fraction,
                }
                handle.write(
                    json.dumps(
                        {
                            "sample_index": sample_index,
                            "rtde_timestamp_sec": 100.0 + sample_index * 0.02,
                            "phase": (
                                "recording_baseline"
                                if baseline
                                else "recording_insertion"
                            ),
                            "actual_tcp_force": [
                                0.0,
                                0.0,
                                1.0 if baseline else 8.0,
                                0.0,
                                0.0,
                                0.1,
                            ],
                            "actual_tcp_speed": (
                                [0.0] * 6
                                if baseline
                                else [0.0, 0.0, -0.002, 0.0, 0.0, 0.0]
                            ),
                            "actual_base_tcp_pose": deepcopy(
                                world_tool0_pose
                            ),
                            "world_base_pose": {
                                "x": 0.0,
                                "y": 0.0,
                                "z": 0.0,
                                "qx": 0.0,
                                "qy": 0.0,
                                "qz": 0.0,
                                "qw": 1.0,
                            },
                            "tool0_tcp_pose": {
                                "x": 0.0,
                                "y": 0.0,
                                "z": 0.0,
                                "qx": 0.0,
                                "qy": 0.0,
                                "qz": 0.0,
                                "qw": 1.0,
                            },
                            "world_tool0_pose": world_tool0_pose,
                            "active_motion_kind": (
                                "" if baseline else "relative_cartesian"
                            ),
                            "active_motion_status": {
                                "message": "matching low-speed Cartesian jog",
                                "world_linear_velocity_m_s": [0.0, 0.0, -0.002],
                                "state": "executing",
                            },
                        }
                    )
                    + "\n"
                )
        return {
            "success": True,
            "active": False,
            "state_uncertain": False,
            "motion_settled": True,
            "recording_id": self.recording_id,
            "trace_path": str(trace_path),
            "trace_sha256": bridge_module.sha256_file(trace_path),
            "sample_count": max(self.sample_count + 8, 32),
            "baseline_valid": True,
            "force_bias": [0.0, 0.0, 1.0, 0.0, 0.0, 0.1],
            "baseline_force_span_n": 0.3,
            "baseline_torque_span_nm": 0.02,
        }


def _insertion_demonstration_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SystemBridge, SimpleNamespace, _InsertionDemonstrationController]:
    controller = _InsertionDemonstrationController()
    trace_root = tmp_path / "server-traces"
    controller.trace_root = trace_root
    monkeypatch.setattr(
        bridge_module,
        "_INSERTION_DEMONSTRATION_TRACE_ROOT",
        trace_root,
    )
    taught_root = tmp_path / "taught_functions"
    taught_recording = taught_root / "place_approach" / "default__hardware.json"
    taught_recording.parent.mkdir(parents=True)
    taught_recording.write_text(
        json.dumps({"recording": "must remain byte-for-byte unchanged"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bridge_module,
        "_ROBOT_TAUGHT_FUNCTIONS_DIR",
        taught_root,
    )
    robot_resource = tmp_path / "robot_ur5e.json"
    robot_resource.write_text(
        (ROOT / "cais_spade_llm/initialization/resources/robot_ur5e.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    resource_payload = json.loads(robot_resource.read_text(encoding="utf-8"))
    move_insert_profile = resource_payload["ur5e"]["real"]["controller"][
        "parts_tuning"
    ]["move_insert"]
    move_insert_profile["demonstration_recipes"] = {}
    move_insert_profile["qualifications"] = {}
    move_insert_profile["validated_parts"] = []
    robot_resource.write_text(
        json.dumps(resource_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bridge_module, "_UR5E_RESOURCE", robot_resource)
    start_pose = {
        "x": 0.4,
        "y": 0.2,
        "z": 0.5,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    agent = SimpleNamespace(
        _current_state="positioned",
        _held_part="MG",
        _gripper_state="closed",
        _controller=controller,
        _task_ctx={
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
            "resolved_cartesian_positions": {
                "move_above_destination": {**deepcopy(start_pose), "z": 0.55},
                "descend": deepcopy(start_pose),
            },
            "held_part_handoff": {
                "part_name": "MG",
                "tool_frame": "tool0",
                "tool0_to_held_part": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                },
            },
            "assembly_board_v1_aruco": {
                "destination_location": "assembly_board-v1",
                "camera_role": "ur5e",
                "generation": 12,
                "calibration_id": "camera-calibration",
                "pose": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                },
            },
            "assembly_board_v1_aruco_generation": 12,
        },
    )
    bridge = SystemBridge.__new__(SystemBridge)
    bridge._insertion_demonstrations_dir = tmp_path / "demonstrations"
    bridge._move_insert_trials_dir = tmp_path / "move_insert_trials"
    bridge._insertion_demonstration_lock = threading.RLock()
    bridge._insertion_demonstration = None
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._starting = False
    bridge._stopping = False
    bridge.system_running = False
    bridge._physical_ur5e_robot_agent = lambda: agent
    bridge._move_insert_pending_review_error = lambda: ""
    bridge._move_insert_hard_caps = lambda _part_name="": (
        {
            "insert_max_contact_speed_m_s": 0.005,
            "insert_max_contact_force_delta_n": 12.0,
            "insert_max_engagement_progress_m": 0.02,
            "insert_max_insertion_force_n": 15.0,
            "insert_max_spiral_radius_m": 0.002,
            "insert_max_spiral_pitch_m": 0.001,
            "insert_max_spiral_speed_m_s": 0.004,
            "insert_max_spiral_acceleration_m_s2": 0.03,
            "insert_max_axial_force_n": 25.0,
            "insert_max_lateral_force_n": 15.0,
            "insert_max_torque_nm": 1.5,
            "insert_max_tool_flange_torque_nm": 3.0,
            "insert_soft_filter_window_sec": 0.05,
            "insert_soft_overload_hold_sec": 0.10,
            "insert_relief_unload_dwell_sec": 0.10,
            "insert_relief_clear_dwell_sec": 0.10,
            "insert_relief_timeout_sec": 1.0,
            "insert_relief_axial_force_ratio": 0.5,
            "insert_relief_reverse_force_ratio": 0.25,
            "insert_relief_clear_hysteresis_ratio": 0.8,
            "insert_relief_resume_ramp_sec": 0.10,
            "insert_relief_search_force_ratio": 0.5,
            "insert_relief_search_speed_ratio": 0.5,
            "insert_relief_backoff_step_m": 0.0001,
            "insert_max_relief_retreat_m": 0.0003,
            "insert_relief_stationary_speed_m_s": 0.0005,
            "insert_relief_stationary_angular_speed_rad_s": 0.01,
            "insert_max_relief_cycles": 3.0,
            "insert_max_tilt_tolerance_rad": math.radians(3.0),
            "insert_max_seated_depth_tolerance_m": 0.003,
            "insert_max_settle_time_sec": 1.5,
            "insert_max_timeout_sec": 60.0,
            "insert_max_travel_m": 0.05,
            "insert_start_position_tolerance_m": 0.003,
            "insert_start_orientation_tolerance_rad": math.radians(2.0),
            "insert_max_contact_search_radius_m": 0.01,
            "insert_max_disengagement_cycles": 6.0,
            "insert_search_peck_retreat_m": 0.003,
            "insert_search_peck_interval_sec": 0.75,
        },
        "",
    )
    bridge._robot_function_execution_pose_readiness = lambda *_args, **_kwargs: {
        "success": True,
        "waypoint": {
            "pose": {
                **start_pose,
                "frame_id": "world",
                "child_frame_id": "tool0",
            }
        },
    }
    return bridge, agent, controller


def _rewrite_insertion_demonstration_trace_on_stop(
    controller: _InsertionDemonstrationController,
    transform: Any,
) -> None:
    original_stop = controller.stop_insertion_demonstration

    def _stop(*, timeout_sec: float) -> dict[str, Any]:
        result = dict(original_stop(timeout_sec=timeout_sec))
        trace_path = Path(result["trace_path"])
        rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]
        transform(rows)
        trace_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        result["trace_sha256"] = bridge_module.sha256_file(trace_path)
        return result

    controller.stop_insertion_demonstration = _stop  # type: ignore[method-assign]


def test_insertion_demonstration_active_tcp_torque_accepts_rtde_pose_vector() -> None:
    torque_norm = SystemBridge._insertion_demonstration_active_tcp_torque_norm(
        {
            "actual_base_tcp_pose": [0.1, -0.2, 0.3, 0.0, 0.0, math.pi / 2.0],
            "tool0_tcp_pose": {
                "x": 0.1,
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        },
        [0.0, 0.0, 10.0, 1.0, 0.0, 0.0],
    )

    assert torque_norm == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "actual_base_tcp_pose",
    (
        [0.0] * 5,
        [0.0, 0.0, 0.0, math.nan, 0.0, 0.0],
        {"qx": 0.0, "qy": 0.0, "qz": 0.0},
    ),
)
def test_insertion_demonstration_active_tcp_torque_rejects_malformed_pose(
    actual_base_tcp_pose: Any,
) -> None:
    with pytest.raises(ValueError, match="base -> TCP pose"):
        SystemBridge._insertion_demonstration_active_tcp_torque_norm(
            {
                "actual_base_tcp_pose": actual_base_tcp_pose,
                "tool0_tcp_pose": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                },
            },
            [0.0] * 6,
        )


def test_save_insertion_recording_accepts_rtde_pose_shape_before_hard_cap_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def _real_rtde_pose_with_unsafe_axial_force(
        rows: list[dict[str, Any]],
    ) -> None:
        for row in rows:
            pose = dict(row["actual_base_tcp_pose"])
            row["actual_base_tcp_pose"] = [
                pose["x"],
                pose["y"],
                pose["z"],
                0.0,
                0.0,
                0.0,
            ]
            if row["phase"] == "recording_insertion":
                row["actual_tcp_force"] = [0.0, 0.0, 30.0, 0.0, 0.0, 0.1]

    _rewrite_insertion_demonstration_trace_on_stop(
        controller,
        _real_rtde_pose_with_unsafe_axial_force,
    )
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    rejected = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )

    assert rejected["success"] is True
    assert rejected["recording_accepted"] is False
    assert rejected["state"] == "recording_saved_mg_hard_cap_review_required"
    assert rejected["candidate_recipe"] == {}
    assert rejected["analysis"]["hard_cap_review_required"] is True
    assert "demonstrated axial force 29.000 N" in rejected["analysis"][
        "hard_caps_error"
    ]
    assert "protected 25.000 N hard cap" in rejected["analysis"][
        "hard_caps_error"
    ]
    assert rejected["analysis"]["minimum_required_hard_caps"][
        "insert_max_axial_force_n"
    ] == pytest.approx(41.0)
    assert "active-TCP torque reference" not in rejected["analysis"][
        "hard_caps_error"
    ]


def test_save_insertion_recording_learns_recipe_without_return_to_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    taught_recording = (
        tmp_path
        / "taught_functions"
        / "place_approach"
        / "default__hardware.json"
    )
    taught_recording_before = taught_recording.read_bytes()
    resolved_before = deepcopy(
        _agent._task_ctx["resolved_cartesian_positions"]
    )

    readiness = bridge.digital_twin_insertion_demonstration_readiness(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    assert readiness["ready"] is True
    assert "recipe is not required" in readiness["message"]

    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    assert started["active"] is True
    completed = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )
    assert completed["success"] is True, completed.get("message")
    assert completed["state"] == "recording_saved_return_to_pre_insertion"
    assert completed["seated_event"]["aruco_to_seated_held_part"]["z"] == pytest.approx(
        0.48
    )
    assert completed["analysis"]["trajectory_replay_authorized"] is False
    assert completed["analysis"]["automatic_move_insert_qualified"] is False
    assert completed["candidate_recipe"]["part_name"] == "MG"
    assert completed["candidate_recipe"]["spiral_radius_m"] == pytest.approx(
        0.0015
    )
    assert completed["candidate_recipe"]["recipe_version"] == 9
    assert completed["candidate_recipe"]["learning_policy_version"] == 12
    assert completed["candidate_recipe"]["learning_policy"][
        "insert_soft_filter_window_sec"
    ] == pytest.approx(0.05)
    assert completed["candidate_recipe"]["learning_policy"][
        "insert_max_relief_cycles"
    ] == pytest.approx(3.0)
    assert len(completed["candidate_recipe"]["learning_policy_sha256"]) == 64
    assert len(completed["candidate_recipe"]["learning_evidence_sha256"]) == 64
    assert len(completed["candidate_recipe"]["hard_caps_sha256"]) == 64
    assert len(
        completed["candidate_recipe"]["force_depth_profile"][
            "depth_fraction"
        ]
    ) == 16
    assert len(
        completed["candidate_recipe"]["force_depth_profile"][
            "axial_upper_n"
        ]
    ) == 16
    assert len(
        completed["candidate_recipe"]["force_depth_profile"][
            "lateral_upper_n"
        ]
    ) == 16
    assert len(
        completed["candidate_recipe"]["force_depth_profile"][
            "torque_upper_nm"
        ]
    ) == 16
    assert len(
        completed["candidate_recipe"]["force_depth_profile_sha256"]
    ) == 64
    recipe = completed["candidate_recipe"]
    force_uncertainty_n = recipe["baseline_force_uncertainty_n"]
    torque_uncertainty_nm = recipe["baseline_torque_uncertainty_nm"]
    profile = recipe["force_depth_profile"]
    assert max(profile["axial_upper_n"]) == pytest.approx(
        recipe["observed_filtered_axial_force_n"]
        + (2.0 * force_uncertainty_n)
    )
    assert recipe["max_axial_force_n"] == pytest.approx(
        recipe["observed_filtered_axial_force_n"]
        + (2.0 * force_uncertainty_n)
    )
    assert all(
        upper + force_uncertainty_n < recipe["hard_caps"][cap_name]
        for field_name, cap_name in (
            ("axial_upper_n", "insert_max_axial_force_n"),
            ("lateral_upper_n", "insert_max_lateral_force_n"),
        )
        for upper in profile[field_name]
    )
    assert all(
        upper + torque_uncertainty_nm
        < recipe["hard_caps"]["insert_max_torque_nm"]
        for upper in profile["torque_upper_nm"]
    )
    assert recipe["tool_frame"] == "tool0"
    assert completed["candidate_recipe"][
        "observed_tool_flange_torque_nm"
    ] >= completed["candidate_recipe"][
        "seated_filtered_tool_flange_torque_nm"
    ]
    trial_settings = bridge._digital_twin_move_insert_trial_settings("MG")
    assert trial_settings["validated"] is True
    assert trial_settings["effective"]["demonstration_recipe"]["recording_id"] == (
        started["recording_id"]
    )
    assert _agent._task_ctx["move_insert_mode"] == "force_limited_trial"
    assert _agent._task_ctx["insert_pose"]["z"] == pytest.approx(0.48)
    assert _agent._task_ctx["resolved_cartesian_positions"] == resolved_before
    assert taught_recording.read_bytes() == taught_recording_before
    assert completed["analysis"]["maximum_tared_lateral_force_n"] == pytest.approx(
        0.0
    )
    assert completed["analysis"]["tared_axial_force_min_n"] < 0.0
    directory = Path(completed["diagnostic_directory"])
    assert (directory / "summary.json").is_file()
    assert (directory / "trace.jsonl").is_file()
    assert (directory / "analysis.json").is_file()
    assert (directory / "diagnostic_bundle.zip").is_file()
    assert json.loads(
        bridge._insertion_demonstration_pending_path().read_text(encoding="utf-8")
    )["pending"] is None


def test_insertion_demonstration_transient_speed_feedback_does_not_reject_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def _brief_feedback_overshoot(rows: list[dict[str, Any]]) -> None:
        insertion_rows = [
            row for row in rows if row["phase"] == "recording_insertion"
        ]
        for row in insertion_rows[8:11]:
            row["actual_tcp_speed"] = [0.0, 0.0, -0.007, 0.0, 0.0, 0.0]

    _rewrite_insertion_demonstration_trace_on_stop(
        controller,
        _brief_feedback_overshoot,
    )
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    saved = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )

    assert saved["success"] is True, saved.get("message")
    assert saved["candidate_recipe"]["recipe_version"] == 9


def test_single_demonstration_profile_rejects_without_extra_hard_cap_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    _agent._held_part = "SG"
    _agent._task_ctx["part_name"] = "SG"
    _agent._task_ctx["held_part_handoff"]["part_name"] = "SG"
    original_hard_caps = bridge._move_insert_hard_caps

    def _sg_shared_lateral_cap(
        part_name: str = "",
    ) -> tuple[dict[str, Any], str]:
        caps, caps_error = original_hard_caps(part_name)
        return {**caps, "insert_max_lateral_force_n": 12.0}, caps_error

    bridge._move_insert_hard_caps = _sg_shared_lateral_cap

    def _lateral_envelope_without_reserve(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["phase"] == "recording_insertion":
                row["actual_tcp_force"] = [3.0, 0.0, 8.0, 0.0, 0.0, 0.1]

    _rewrite_insertion_demonstration_trace_on_stop(
        controller,
        _lateral_envelope_without_reserve,
    )
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="SG",
        confirmed=True,
    )
    rejected = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="SG",
        recording_id=started["recording_id"],
    )

    assert rejected["success"] is True
    assert rejected["recording_accepted"] is False
    assert rejected["state"] == "recording_saved_mg_hard_cap_review_required"
    assert rejected["candidate_recipe"] == {}
    assert rejected["analysis"]["hard_cap_review_required"] is True
    assert "16-bin lateral force envelope" in rejected["analysis"][
        "hard_caps_error"
    ]
    assert "additional uncertainty reserve" in rejected["analysis"][
        "hard_caps_error"
    ]
    assert "rejected, not clipped" in rejected["analysis"][
        "hard_caps_error"
    ]
    assert rejected["analysis"]["minimum_required_hard_caps"][
        "insert_max_lateral_force_n"
    ] == pytest.approx(15.0)


def test_insertion_demonstration_sustained_speed_feedback_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def _sustained_feedback_overspeed(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["phase"] == "recording_insertion":
                row["actual_tcp_speed"] = [0.0, 0.0, -0.007, 0.0, 0.0, 0.0]

    _rewrite_insertion_demonstration_trace_on_stop(
        controller,
        _sustained_feedback_overspeed,
    )
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    saved = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )

    assert saved["success"] is True, saved.get("message")
    recipe = saved["candidate_recipe"]
    assert recipe["learning_policy_version"] == 12
    assert recipe["learning_policy"]["demonstration_speed_policy"] == (
        "diagnostic_only"
    )
    assert recipe["observed_advancing_speed_m_s"] == pytest.approx(0.007)
    assert recipe["contact_speed_m_s"] == pytest.approx(0.005)


def test_insertion_demonstration_precontact_backtracking_is_learning_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def _precontact_backtracking(rows: list[dict[str, Any]]) -> None:
        insertion_rows = [
            row for row in rows if row["phase"] == "recording_insertion"
        ]
        for index, row in enumerate(insertion_rows[:8]):
            row["actual_tcp_force"] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.1]
            z = 0.499 if index < 4 else 0.502
            row["world_tool0_pose"]["z"] = z
            row["actual_base_tcp_pose"]["z"] = z

    _rewrite_insertion_demonstration_trace_on_stop(
        controller,
        _precontact_backtracking,
    )
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    saved = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )

    assert saved["success"] is True, saved.get("message")
    assert saved["recording_accepted"] is True
    assert saved["candidate_recipe"]["learning_policy_version"] == 12
    assert saved["candidate_recipe"]["learning_policy"]["rebound_policy"] == (
        "final_saved_depth_within_tolerance_of_post_contact_maximum"
    )


def test_rejected_insertion_recording_remains_available_for_reanalysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def _final_rebound(rows: list[dict[str, Any]]) -> None:
        insertion_rows = [
            row
            for row in rows
            if row["phase"] == "recording_insertion"
            and int(row["sample_index"]) < controller.sample_count
        ]
        deepest_row = insertion_rows[-2]
        deepest_row["world_tool0_pose"]["z"] = 0.48
        deepest_row["actual_base_tcp_pose"]["z"] = 0.48
        final_row = insertion_rows[-1]
        final_row["world_tool0_pose"]["z"] = 0.482
        final_row["actual_base_tcp_pose"]["z"] = 0.482

    _rewrite_insertion_demonstration_trace_on_stop(controller, _final_rebound)
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    rejected = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )

    assert rejected["success"] is False
    assert rejected["reanalyze_available"] is True
    assert "rebounded before Save Recording" in rejected["message"]
    bridge._insertion_demonstration = None

    restarted_readiness = bridge.digital_twin_insertion_demonstration_readiness(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )

    assert restarted_readiness["success"] is True
    assert restarted_readiness["ready"] is True
    assert restarted_readiness["reanalyze_available"] is True
    assert restarted_readiness["recording_id"] == started["recording_id"]
    assert "Start Recording is also available" in restarted_readiness["message"]


def test_mg_review_bundle_reanalyzes_without_motion_or_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, agent, controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    reviewed_caps, reviewed_caps_error = bridge._move_insert_hard_caps("MG")
    assert reviewed_caps_error == ""
    mg_cap_fields = (
        "insert_max_insertion_force_n",
        "insert_max_axial_force_n",
        "insert_max_lateral_force_n",
        "insert_max_torque_nm",
        "insert_max_tool_flange_torque_nm",
    )
    review_caps = deepcopy(reviewed_caps)
    for field_name in mg_cap_fields:
        review_caps[field_name] = None

    def _review_caps(part_name: str = "") -> tuple[dict[str, Any], str]:
        if part_name == "MG":
            return deepcopy(review_caps), (
                "Insertion hard caps are missing or invalid: "
                + ", ".join(mg_cap_fields)
            )
        return deepcopy(reviewed_caps), ""

    bridge._move_insert_hard_caps = _review_caps
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    saved = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )

    assert saved["success"] is True
    assert saved["recording_accepted"] is False
    assert saved["state"] == "recording_saved_mg_hard_cap_review_required"
    assert saved["candidate_recipe"] == {}
    assert saved["analysis"]["hard_cap_review_required"] is True
    assert set(saved["analysis"]["minimum_required_hard_caps"]) == set(
        mg_cap_fields
    )
    directory = Path(saved["diagnostic_directory"])
    source_paths = tuple(
        directory / name
        for name in (
            "trace.jsonl",
            "analysis.json",
            "summary.json",
            "diagnostic_bundle.zip",
        )
    )
    source_bytes = {path: path.read_bytes() for path in source_paths}
    controller.start_insertion_demonstration = lambda **_kwargs: pytest.fail(
        "reanalysis must not start a recorder"
    )
    controller.stop_insertion_demonstration = lambda **_kwargs: pytest.fail(
        "reanalysis must not stop or command a recorder"
    )

    review = bridge.digital_twin_reanalyze_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=str(saved["recording_id"]),
    )

    assert review["success"] is True
    assert review["ready"] is False
    assert review["hard_cap_review_required"] is True
    assert Path(review["reanalysis_path"]).name == "analysis_v3.json"
    assert all(path.read_bytes() == source_bytes[path] for path in source_paths)
    resource_path = bridge_module._UR5E_RESOURCE
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    profile = resource["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]
    assert profile["demonstration_recipes"] == {}
    assert profile["validated_parts"] == []
    assert profile["qualifications"] == {}

    bridge._move_insert_hard_caps = lambda _part_name="": (
        deepcopy(reviewed_caps),
        "",
    )
    installed = bridge.digital_twin_reanalyze_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=str(saved["recording_id"]),
    )

    assert installed["success"] is True
    assert installed["ready"] is True
    assert installed["candidate_recipe"]["recipe_version"] == 9
    assert installed["candidate_recipe"]["hard_caps"] == reviewed_caps
    assert installed["candidate_recipe"]["hard_caps_sha256"] == (
        bridge._move_insert_hard_caps_sha256(reviewed_caps)
    )
    assert len(
        installed["candidate_recipe"]["force_depth_profile"][
            "depth_fraction"
        ]
    ) == 16
    assert all(path.read_bytes() == source_bytes[path] for path in source_paths)
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    profile = resource["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]
    assert profile["demonstration_recipes"]["MG"] == installed[
        "candidate_recipe"
    ]
    assert profile["validated_parts"] == []
    assert profile["qualifications"] == {}


def test_current_unsafe_mg_demonstration_is_rejected_not_clipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def _unsafe_envelope(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["phase"] == "recording_insertion":
                row["actual_tcp_force"] = [
                    21.857,
                    0.0,
                    116.004,
                    0.0,
                    0.0,
                    4.913,
                ]

    _rewrite_insertion_demonstration_trace_on_stop(
        controller,
        _unsafe_envelope,
    )
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    rejected = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )

    assert rejected["success"] is True
    assert rejected["recording_accepted"] is False
    assert rejected["state"] == "recording_saved_mg_hard_cap_review_required"
    assert rejected["part_clamped"] is True
    assert "axial force 115.004 N" in rejected["analysis"][
        "hard_caps_error"
    ]
    assert "rejected, not clipped" in rejected["analysis"][
        "hard_caps_error"
    ]
    assert rejected["candidate_recipe"] == {}
    assert rejected["analysis"]["hard_cap_review_required"] is True
    assert rejected["analysis"]["minimum_required_hard_caps"][
        "insert_max_axial_force_n"
    ] == pytest.approx(127.004)
    assert rejected["analysis"]["maximum_tared_lateral_force_n"] == (
        pytest.approx(21.857)
    )
    assert rejected["analysis"]["maximum_tared_tool_flange_torque_norm_nm"] == (
        pytest.approx(4.813)
    )
    resource = json.loads(
        (tmp_path / "robot_ur5e.json").read_text(encoding="utf-8")
    )
    profile = resource["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]
    assert profile["demonstration_recipes"] == {}
    assert profile["validated_parts"] == []
    assert profile["qualifications"] == {}
    assert Path(rejected["diagnostic_bundle_path"]).is_file()
    assert json.loads(
        bridge._insertion_demonstration_pending_path().read_text(
            encoding="utf-8"
        )
    )["pending"] is None


def test_tensile_only_demonstration_cannot_learn_compressive_contact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def _tensile_only(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["phase"] == "recording_insertion":
                row["actual_tcp_force"] = [0.0, 0.0, -6.0, 0.0, 0.0, 0.1]

    _rewrite_insertion_demonstration_trace_on_stop(controller, _tensile_only)
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    rejected = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )

    assert rejected["success"] is False
    assert rejected["recording_accepted"] is False
    assert "sustained compressive force change" in rejected["message"]
    assert rejected["candidate_recipe"] == {}


def test_tool_flange_hard_torque_rejects_low_active_tcp_torque_demo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def _high_flange_low_tcp(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["phase"] == "recording_insertion":
                row["tool0_tcp_pose"]["x"] = 0.5
                row["actual_tcp_force"] = [0.0, 0.0, 8.0, 0.0, -3.5, 0.1]

    _rewrite_insertion_demonstration_trace_on_stop(
        controller,
        _high_flange_low_tcp,
    )
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    rejected = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )

    assert rejected["success"] is True
    assert rejected["recording_accepted"] is False
    assert rejected["state"] == "recording_saved_mg_hard_cap_review_required"
    assert "tool-flange torque 3.500 Nm" in rejected["analysis"][
        "hard_caps_error"
    ]
    assert "rejected, not clipped" in rejected["analysis"][
        "hard_caps_error"
    ]
    assert rejected["analysis"]["minimum_required_hard_caps"][
        "insert_max_tool_flange_torque_nm"
    ] == pytest.approx(3.68)
    assert rejected["analysis"]["maximum_tared_torque_norm_nm"] == (
        pytest.approx(0.0, abs=1e-9)
    )
    assert rejected["analysis"]["maximum_tared_tool_flange_torque_norm_nm"] == (
        pytest.approx(3.5)
    )


def test_delete_previous_insertion_recording_trashes_exact_recipe_without_motion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, agent, _controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    saved = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )
    assert saved["success"] is True
    rejected_directory = (
        bridge._insertion_demonstrations_dir
        / "insertion-demonstration-rejected-diagnostic"
    )
    rejected_directory.mkdir(parents=True)
    (rejected_directory / "trace.jsonl").write_text("{}\n", encoding="utf-8")
    (rejected_directory / "analysis.json").write_text(
        json.dumps(
            {
                "accepted": False,
                "hard_cap_review_required": False,
                "rejection_reason": "Rejected diagnostic evidence only.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (rejected_directory / "summary.json").write_text(
        json.dumps(
            {
                "recording_id": rejected_directory.name,
                "robot": "ur5e",
                "destination_location": "assembly_board-v1",
                "part_name": "MG",
                "success": False,
                "ready": True,
                "active": False,
                "state": "not_recorded",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    deleted = bridge.digital_twin_delete_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )

    assert deleted["success"] is True
    assert deleted["state"] == "not_recorded"
    assert not Path(deleted["trashed_path"]).exists()
    assert deleted["permanently_deleted"] is True
    resource = json.loads(
        bridge_module._UR5E_RESOURCE.read_text(encoding="utf-8")
    )
    profile = resource["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]
    assert "MG" not in profile["demonstration_recipes"]
    assert "MG" not in profile["validated_parts"]
    assert "MG" not in profile["qualifications"]
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"
    readiness = bridge.digital_twin_insertion_demonstration_readiness(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
    )
    assert readiness["ready"] is True
    assert readiness["state"] == "not_recorded"


def test_delete_previous_insertion_recording_ignores_stopped_server_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, _controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    saved = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )
    assert saved["success"] is True
    bridge._ur5e_rtde_trajectory_status = lambda: {
        "updated_at": time.time(),
        "process_id": 4101,
        "insertion_demonstration_active": True,
    }
    bridge._ros2_procs = {}

    deleted = bridge.digital_twin_delete_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )

    assert deleted["success"] is True


def test_delete_previous_insertion_recording_blocks_fresh_owned_recorder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, _controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    saved = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )
    assert saved["success"] is True
    recording_directory = bridge._insertion_demonstration_directory(
        str(saved["recording_id"])
    )
    bridge._insertion_demonstration = None
    bridge._ur5e_rtde_trajectory_status = lambda: {
        "updated_at": time.time(),
        "process_id": 4102,
        "insertion_demonstration_active": True,
    }
    bridge._ros2_procs = {
        "hardware_ur5e_rtde_trajectory_server": SimpleNamespace(
            pid=4102,
            poll=lambda: None,
        )
    }

    blocked = bridge.digital_twin_delete_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )

    assert blocked["success"] is False
    assert blocked["message"] == (
        "Cancel Recording before deleting the previous recording."
    )
    assert recording_directory.is_dir()


def test_delete_previous_insertion_recording_blocks_fresh_owned_insert_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, _controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    saved = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )
    assert saved["success"] is True
    recording_directory = bridge._insertion_demonstration_directory(
        str(saved["recording_id"])
    )
    bridge._insertion_demonstration = None
    bridge._ur5e_rtde_trajectory_status = lambda: {
        "updated_at": time.time(),
        "process_id": 4103,
        "state": "executing",
        "trial_id": "move-insert-server-active",
        "insert_phase": "searching",
    }
    bridge._ros2_procs = {
        "hardware_ur5e_rtde_trajectory_server": SimpleNamespace(
            pid=4103,
            poll=lambda: None,
        )
    }

    blocked = bridge.digital_twin_delete_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )

    assert blocked["success"] is False
    assert "terminal settlement" in blocked["message"]
    assert recording_directory.is_dir()


def test_delete_previous_insertion_recording_blocks_stale_running_rtde_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, _controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    saved = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )
    assert saved["success"] is True
    recording_directory = bridge._insertion_demonstration_directory(
        str(saved["recording_id"])
    )
    bridge._insertion_demonstration = None
    bridge._ur5e_rtde_trajectory_status = lambda: {
        "updated_at": time.time() - 4.0,
        "process_id": 9999,
        "insertion_demonstration_active": False,
        "state": "ready",
    }
    bridge._ros2_procs = {
        "hardware_ur5e_rtde_trajectory_server": SimpleNamespace(
            pid=4104,
            poll=lambda: None,
        )
    }

    blocked = bridge.digital_twin_delete_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )

    assert blocked["success"] is False
    assert "fresh, tracked UR5e RTDE server status" in blocked["message"]
    assert recording_directory.is_dir()


def test_delete_previous_insertion_recording_ignores_inactive_other_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, _controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    sg_recording_id = "insertion-demonstration-sg-retained"
    sg_directory = bridge._insertion_demonstration_directory(sg_recording_id)
    sg_directory.mkdir(parents=True)
    (sg_directory / "summary.json").write_text(
        json.dumps(
            {
                "target": "dual robots",
                "robot": "ur5e",
                "destination_location": "assembly_board-v1",
                "part_name": "SG",
                "recording_id": sg_recording_id,
                "active": False,
                "recovery_required": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bridge._insertion_demonstration = {
        "target": "dual robots",
        "robot": "ur5e",
        "destination_location": "assembly_board-v1",
        "part_name": "SG",
        "recording_id": sg_recording_id,
        "active": False,
        "recovery_required": False,
    }

    deleted = bridge.digital_twin_delete_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )

    assert deleted["success"] is True
    assert deleted["trashed_recording_paths"] == []
    assert deleted["trashed_trial_paths"] == []
    assert sg_directory.is_dir()


def test_delete_previous_insertion_recording_rejects_disagreeing_exact_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, _controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    base = {
        "target": "dual robots",
        "robot": "ur5e",
        "destination_location": "assembly_board-v1",
        "part_name": "MG",
        "active": False,
        "recovery_required": False,
    }
    process_id = "insertion-demonstration-process"
    durable_id = "insertion-demonstration-durable"
    process_directory = bridge._insertion_demonstration_directory(process_id)
    durable_directory = bridge._insertion_demonstration_directory(durable_id)
    process_directory.mkdir(parents=True)
    durable_directory.mkdir(parents=True)
    bridge._insertion_demonstration = {**base, "recording_id": process_id}
    assert (
        bridge._write_insertion_demonstration_pending(
            {**base, "recording_id": durable_id}
        )
        == ""
    )

    blocked = bridge.digital_twin_delete_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )

    assert blocked["success"] is False
    assert "authority changed" in blocked["message"]
    assert process_directory.is_dir()
    assert durable_directory.is_dir()


def test_delete_previous_insertion_recording_trashes_orphan_trial_and_suspension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, agent, _controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    saved = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )
    assert saved["success"] is True
    recording_id = str(saved["recording_id"])

    resource_path = bridge_module._UR5E_RESOURCE
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    profile = resource["ur5e"]["real"]["controller"]["parts_tuning"][
        "move_insert"
    ]
    profile["demonstration_recipes"] = {}
    profile["validated_parts"] = []
    profile["qualifications"] = {}
    resource_path.write_text(json.dumps(resource, indent=2) + "\n", encoding="utf-8")
    resource_sha256 = bridge_module.sha256_file(resource_path)
    bridge._insertion_demonstration = None

    trial_id = "move-insert-orphan-recording"
    trial_directory = bridge._move_insert_trial_directory(trial_id)
    trial_directory.mkdir(parents=True)
    (trial_directory / "summary.json").write_text(
        json.dumps(
            {
                "success": False,
                "target": "dual robots",
                "robot": "ur5e",
                "destination_location": "assembly_board-v1",
                "part_name": "MG",
                "trial_id": trial_id,
                "active": False,
                "completion_motion_active": False,
                "review_required": False,
                "recovery_required": False,
                "normal_repair_required": False,
                "hardware_stack_repair_required": False,
                "part_clamped": True,
                "released": False,
                "lifted": False,
                "failure_recorded": True,
                "qualified": False,
                "move_insert_effective": {
                    "demonstration_recipe": {"recording_id": recording_id}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert bridge._latch_move_insert_suspension("MG", "old MG trial") == ""

    deleted = bridge.digital_twin_delete_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )

    assert deleted["success"] is True
    assert deleted["resource_changed"] is False
    assert bridge_module.sha256_file(resource_path) == resource_sha256
    assert len(deleted["trashed_recording_paths"]) == 1
    assert len(deleted["trashed_trial_paths"]) == 1
    assert not Path(deleted["trashed_recording_paths"][0]).exists()
    assert not Path(deleted["trashed_trial_paths"][0]).exists()
    assert not bridge._insertion_demonstration_directory(recording_id).exists()
    assert not trial_directory.exists()
    assert bridge._move_insert_suspension_error("MG") == ""
    pending, pending_error = bridge._read_move_insert_pending_review()
    assert pending_error == ""
    assert pending is None
    assert agent._held_part == "MG"
    assert agent._gripper_state == "closed"


def test_delete_previous_insertion_recording_transfers_uncertainty_before_trash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, _controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    bridge._hardware_state_dir = tmp_path / "hardware_state"
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    saved = bridge.digital_twin_save_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )
    assert saved["success"] is True
    recording_id = str(saved["recording_id"])
    bridge._insertion_demonstration = None

    trial_id = "move-insert-uncertain-delete"
    stored = bridge._store_move_insert_trial(
        {
            "success": False,
            "ready": False,
            "target": "dual robots",
            "robot": "ur5e",
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
            "trial_id": trial_id,
            "active": False,
            "completion_motion_active": False,
            "review_required": False,
            "recovery_required": False,
            "normal_repair_required": True,
            "hardware_stack_repair_required": True,
            "part_clamped": True,
            "released": False,
            "lifted": False,
            "automatic_checks_passed": False,
            "completion_eligible": False,
            "qualified": False,
            "failure_recorded": True,
            "hardware_stack_repair_reason": "terminal feedback is uncertain",
            "message": "terminal feedback is uncertain",
            "move_insert_effective": {
                "demonstration_recipe": {"recording_id": recording_id}
            },
        }
    )
    assert stored["hardware_stack_repair_required"] is True

    deleted = bridge.digital_twin_delete_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )

    assert deleted["success"] is True
    uncertainty, uncertainty_error = (
        bridge._read_ur5e_hardware_state_uncertainty()
    )
    assert uncertainty_error == ""
    assert uncertainty is not None
    assert uncertainty["robot"] == "ur5e"
    assert uncertainty["source"] == "move_insert"
    assert uncertainty["source_id"] == trial_id
    assert uncertainty["reason"] == "terminal feedback is uncertain"
    assert bridge._ur5e_robot_function_state_uncertain is True
    assert bridge._ur5e_cartesian_jog_state_uncertain is True
    pending, pending_error = bridge._read_move_insert_pending_review()
    assert pending_error == ""
    assert pending is None
    assert not Path(deleted["trashed_recording_paths"][0]).exists()
    assert not Path(deleted["trashed_trial_paths"][0]).exists()


@pytest.mark.parametrize("part_name", ["SG", "MG", "LG", "SCP", "MCP", "LCP"])
def test_insertion_demonstration_accepts_exact_translation_only_parts(
    part_name: str,
) -> None:
    assert (
        SystemBridge._insertion_demonstration_request_error(
            "dual robots",
            "ur5e",
            "assembly_board-v1",
            part_name,
        )
        == ""
    )


@pytest.mark.parametrize("part_name", ["SRP", "MRP", "LRP"])
def test_insertion_demonstration_keeps_angular_parts_blocked(part_name: str) -> None:
    assert "angular alignment" in SystemBridge._insertion_demonstration_request_error(
        "dual robots",
        "ur5e",
        "assembly_board-v1",
        part_name,
    )


def test_insertion_demonstration_allows_only_ur5e_cartesian_jog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, _controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    bridge._insertion_demonstration = {
        "recording_id": "insertion-demonstration-1",
        "robot": "ur5e",
        "destination_location": "assembly_board-v1",
        "part_name": "MG",
        "active": True,
        "recovery_required": False,
    }

    assert (
        bridge._insertion_demonstration_blocking_error(
            allow_ur5e_cartesian_jog=True
        )
        == ""
    )
    assert "Only matching UR5e Cartesian jog" in (
        bridge._insertion_demonstration_blocking_error()
    )


@pytest.mark.parametrize("op", ["cartesian", "cartesian_smooth"])
def test_recording_mode_cartesian_jog_bypasses_unrelated_move_insert_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    op: str,
) -> None:
    bridge, _agent, _controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    assert started["state"] == "recording_insertion"
    bridge.robot_env = "real"
    bridge.teleop_target = lambda _robot, _op: {
        "environment": "real",
        "ready": True,
        "warning": "",
        "hardware_cartesian_readiness": {
            "cartesian_jog_ready": True,
            "message": "ready",
        },
    }
    bridge.teleop_named_position_readiness = lambda _robot: (True, "")
    bridge._move_insert_pending_review_error = lambda: "unrelated review blocker"
    bridge._ur5e_robot_function_state_uncertain = False
    bridge._ur5e_cartesian_jog_state_uncertain = False
    bridge._teleop_cartesian_modes = {"ur5e": "off", "xarm6": "off"}
    bridge._teleop_smooth_session_lock = threading.RLock()
    bridge._teleop_smooth_session = None
    bridge._teleop_xarm6_cartesian_session_lock = threading.RLock()
    bridge._teleop_xarm6_cartesian_session = None
    bridge._ur5e_robot_function_execution_active = None
    payloads: list[dict[str, Any]] = []
    bridge._teleop_request_payload = (  # type: ignore[method-assign]
        lambda payload, timeout_sec, ros_domain_id=None: (
            payloads.append(deepcopy(payload))
            or (True, "matching UR5e Cartesian jog accepted", {"state_uncertain": False})
        )
    )

    readiness = bridge._teleop_preflight("ur5e", op)

    assert readiness["ready"] is True
    assert readiness["warning"] == ""
    assert bridge._insertion_demonstration_cartesian_blocker == ""
    mode = "smooth" if op == "cartesian_smooth" else "step"
    assert bridge.teleop_cartesian_mode("ur5e", mode)[0] is True
    if op == "cartesian_smooth":
        accepted, _message = bridge.teleop_cartesian_smooth(
            "ur5e",
            "z",
            -5.0,
            "start",
        )
        assert accepted is True
        assert bridge.teleop_cartesian_smooth("ur5e", "z", 0.0, "stop")[0] is True
    else:
        accepted, _message = bridge.teleop_jog("ur5e", "z", -1.0)
        assert accepted is True
    assert any(payload["op"] == op for payload in payloads)


def test_recording_mode_cartesian_preflight_keeps_joint_state_monitor_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, _controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    assert started["state"] == "recording_insertion"
    bridge.robot_env = "real"
    bridge.teleop_target = lambda _robot, _op: {
        "environment": "real",
        "source": "hardware",
        "ros_domain_id": 42,
        "ready": True,
        "warning": "",
        "hardware_cartesian_readiness": {
            "cartesian_jog_ready": True,
            "message": "ready",
        },
    }
    bridge._ur5e_rtde_trajectory_status = lambda: {
        "updated_at": time.time(),
        "ros_domain_id": 42,
        "rtde_control_connected": True,
        "state": "ready",
    }
    bridge._ros_action_readiness_error = lambda *_args, **_kwargs: ""
    bridge._teleop_request_payload = (  # type: ignore[method-assign]
        lambda payload, timeout_sec, ros_domain_id=None: (
            True,
            "fresh UR5e state",
            {
                "state": {
                    "joint_state_age_sec": 0.05,
                    "positions": [0.0] * 6,
                }
            },
        )
    )
    bridge._move_insert_pending_review_error = lambda: "unrelated review blocker"
    bridge._ur5e_robot_function_state_uncertain = False
    bridge._ur5e_cartesian_jog_state_uncertain = False

    ready, message = bridge.teleop_named_position_readiness("ur5e")
    preflight = bridge._teleop_preflight("ur5e", "cartesian")

    assert ready is True, message
    assert preflight["ready"] is True
    assert preflight["warning"] == ""
    assert bridge._insertion_demonstration_cartesian_blocker == ""


def test_cancelled_insertion_demonstration_discards_without_restart_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, _controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )

    cancelled = bridge.digital_twin_cancel_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
        note="operator stopped the demonstration",
    )

    assert cancelled["success"] is True
    assert cancelled["recovery_required"] is False
    assert cancelled["active"] is False
    restarted = SystemBridge.__new__(SystemBridge)
    restarted._insertion_demonstrations_dir = tmp_path / "demonstrations"
    restarted._insertion_demonstration_lock = threading.RLock()
    restarted._insertion_demonstration = None
    recovered = restarted._current_insertion_demonstration()
    assert recovered is None
    assert restarted._insertion_demonstration_blocking_error() == ""


def test_cancelling_insertion_recording_poll_discards_after_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _agent, controller = _insertion_demonstration_bridge(
        tmp_path,
        monkeypatch,
    )
    started = bridge.digital_twin_start_insertion_recording(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        confirmed=True,
    )
    trace_path = controller.trace_root / f"{started['recording_id']}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("temporary recording\n", encoding="utf-8")
    cancelling = {
        **started,
        "success": False,
        "active": True,
        "state": "cancelling",
        "cancelled": True,
    }
    bridge._insertion_demonstration = cancelling
    bridge._write_insertion_demonstration_pending(cancelling)

    def _settled_stop(*, timeout_sec: float) -> dict[str, Any]:
        assert timeout_sec == pytest.approx(1.0)
        return {
            "success": True,
            "active": False,
            "motion_settled": True,
            "trace_path": str(trace_path),
        }

    controller.stop_insertion_demonstration = _settled_stop  # type: ignore[method-assign]
    status = bridge.digital_twin_insertion_recording_status(
        "dual robots",
        "ur5e",
        destination_location="assembly_board-v1",
        part_name="MG",
        recording_id=started["recording_id"],
    )

    assert status["success"] is True
    assert status["state"] == "not_recorded"
    assert status["recovery_required"] is False
    assert not trace_path.exists()
    assert bridge._current_insertion_demonstration() is None


def test_restarted_insertion_recording_uses_normal_hardware_uncertainty(
    tmp_path: Path,
) -> None:
    bridge = SystemBridge.__new__(SystemBridge)
    bridge._insertion_demonstrations_dir = tmp_path / "demonstrations"
    bridge._insertion_demonstration_lock = threading.RLock()
    bridge._insertion_demonstration = None
    bridge._write_insertion_demonstration_pending(
        {
            "recording_id": "insertion-demonstration-interrupted",
            "robot": "ur5e",
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
            "active": True,
            "state": "cancelling",
            "message": "recorder settlement interrupted by UI restart",
        }
    )

    assert bridge._current_insertion_demonstration() is None
    assert bridge._ur5e_robot_function_state_uncertain is True
    assert bridge._ur5e_cartesian_jog_state_uncertain is True
    assert bridge._read_insertion_demonstration_pending() is None
    uncertainty, uncertainty_error = (
        bridge._read_ur5e_hardware_state_uncertainty()
    )
    assert uncertainty_error == ""
    assert uncertainty is not None
    assert uncertainty["source"] == "insertion_demonstration"
    assert uncertainty["source_id"] == "insertion-demonstration-interrupted"
    assert bridge._insertion_demonstration_blocking_error() == ""


def test_restarted_insertion_baseline_does_not_create_hardware_uncertainty(
    tmp_path: Path,
) -> None:
    bridge = SystemBridge.__new__(SystemBridge)
    bridge._insertion_demonstrations_dir = tmp_path / "demonstrations"
    bridge._hardware_state_dir = tmp_path / "hardware_state"
    bridge._insertion_demonstration_lock = threading.RLock()
    bridge._insertion_demonstration = None
    bridge._ur5e_robot_function_state_uncertain = False
    bridge._ur5e_robot_function_state_uncertain_reason = ""
    bridge._ur5e_cartesian_jog_state_uncertain = False
    bridge._ur5e_cartesian_jog_state_uncertain_reason = ""
    bridge._write_insertion_demonstration_pending(
        {
            "recording_id": "insertion-demonstration-passive-baseline",
            "robot": "ur5e",
            "destination_location": "assembly_board-v1",
            "part_name": "MG",
            "active": True,
            "state": "recording_baseline",
            "message": (
                "Recording stationary force baseline. Keep UR5e still until "
                "Recording insertion appears."
            ),
        }
    )

    assert bridge._current_insertion_demonstration() is None
    assert bridge._read_insertion_demonstration_pending() is None
    assert bridge._ur5e_robot_function_state_uncertain is False
    assert bridge._ur5e_cartesian_jog_state_uncertain is False

"""Dry-run test: Case 3 LG-slippage scenario → LLM bridge recovery.

Mimics what happens in Gazebo when xArm6 places LG, slippage is injected,
and the LLM bridge is triggered to produce a recovery plan.  The plan is then
compared against the verified preprogrammed plan (recover_lg_v1).

Run with F5 / python directly:
    python test/test_case3_bridge_dryrun.py
    python test/test_case3_bridge_dryrun.py --prepare-only
    python test/test_case3_bridge_dryrun.py --model gpt-4o
    python test/test_case3_bridge_dryrun.py --show-llm-input
    python test/test_case3_bridge_dryrun.py --show-single-shot-prompt

Run as pytest:
    pytest test/test_case3_bridge_dryrun.py -v
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bootstrap_repo_site_packages(root: Path) -> None:
    venv_lib = root / ".venv" / "lib"
    if not venv_lib.exists():
        return
    for site_packages in sorted(venv_lib.glob("python*/site-packages")):
        site_path = str(site_packages.resolve())
        if site_path not in sys.path:
            sys.path.insert(0, site_path)


_bootstrap_repo_site_packages(ROOT)

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def _load_local_env(path: Path) -> None:
    if load_dotenv is not None:
        load_dotenv(path)
        return
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


_load_local_env(ROOT / ".env")

import pytest

from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_artifacts import (
    write_bridge_artifacts,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
    get_resource_bridge_snapshot,
)
from cais_spade_llm.agents.intelligent_product.replanner.failure_context import (
    build_failure_event,
    failure_context_from_scenario_config,
    load_failure_scenario_config,
)


class ProcessPlannerPrepareTrace(ProcessPlanner):
    """ProcessPlanner using the active top-level bridge session wiring."""
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CASE_ID = "case3_llm_bridge"
FAILED_TASK_ID = "REQ_2_T4"
ANCHOR_TASK_ID = "REQ_2_T3"
SCENARIO_ID = "recover_lg_v1"
GOAL_STATE = "assembled"
DEFAULT_LIVE_MODEL = os.environ.get("CASE3_RECOVERY_MODEL", "gpt-4o")
DEBUG_DIR = Path("cais_spade_llm/monitor/debug")
CASE3_COMPLETED_TASK_IDS = (
    "REQ_1_T1",
    "REQ_1_T2",
    "REQ_2_T1",
    "REQ_2_T2",
    "REQ_2_T3",
)
MOCK_SINGLE_SHOT_RESPONSE = json.dumps(
    {
        "primary_obligation": {
            "rule_id": "SAFE_2-1",
            "resource_jid": "xarm6@localhost",
        },
        "macro_tasks": [],
    },
    indent=2,
)

MUTEX_RULE_FAMILIES = frozenset(
    {
        "mutual_exclusion_zone",
        "no_simultaneous_presence_in_destination_area",
    }
)

# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return ROOT


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case3_paths() -> dict[str, Path]:
    root = _repo_root()
    bundle_root = root / "cais_spade_llm" / "user_verified_plan" / "bundles" / CASE_ID
    return {
        "bundle_manifest": bundle_root / "bundle_manifest.json",
        "tools": bundle_root / "catalog" / "tools.json",
        "plan": bundle_root / "plan" / "case3_two_arm_llm_bridge_plan.json",
        "requirements": bundle_root / "plan" / "case3_two_arm_llm_bridge_requirements.json",
        "safety_logic": bundle_root / "safety" / "cca_safety_logic.json",
        "geometry": (
            root / "cais_spade_llm" / "specification" / "products"
            / "geometry" / "assembly_board-v1.json"
        ),
        "ur5e": root / "cais_spade_llm" / "initialization" / "resources" / "robot_ur5e.json",
        "xarm6": root / "cais_spade_llm" / "initialization" / "resources" / "robot_xarm6.json",
    }


def _case3_bundle_context(paths: dict[str, Path]) -> dict[str, Any]:
    manifest_payload = _load_json(paths["bundle_manifest"])
    if not isinstance(manifest_payload, dict):
        manifest_payload = {}
    manifest_payload["artifacts"] = {
        **dict(manifest_payload.get("artifacts") or {}),
        "requirements_json": str(paths["requirements"]),
        "plan_json": str(paths["plan"]),
        "tools_json": str(paths["tools"]),
        "safety_logic_json": str(paths["safety_logic"]),
    }
    manifest_payload["manifest_path"] = str(paths["bundle_manifest"])
    return manifest_payload


def _load_robot_config(path: Path, key: str) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"robot config {path} did not decode to an object")
    config = payload.get(key)
    if not isinstance(config, dict):
        raise KeyError(f"robot config {path} is missing top-level key '{key}'")
    return config


# ---------------------------------------------------------------------------
# Safety rule helpers (loaded from cca_safety_logic.json)
# ---------------------------------------------------------------------------


def _load_runtime_safety_rules() -> list[dict[str, Any]]:
    payload = _load_json(
        _repo_root() / "cais_spade_llm" / "safety" / "cca_safety_logic.json"
    )
    if isinstance(payload, dict):
        raw_rules = payload.get("rules") or []
    elif isinstance(payload, list):
        raw_rules = payload
    else:
        raw_rules = []
    return [dict(rule) for rule in raw_rules if isinstance(rule, dict)]


def _find_runtime_safety_rule(
    *,
    destination: str,
    required_resources: set[str],
    constraint_families: set[str] | None = None,
) -> dict[str, Any]:
    allowed_families = {
        str(item or "").strip().lower()
        for item in (constraint_families or MUTEX_RULE_FAMILIES)
        if str(item or "").strip()
    }
    normalized_resources = {
        str(r or "").strip().lower() for r in required_resources if str(r or "").strip()
    }
    for rule in _load_runtime_safety_rules():
        ct = str(rule.get("constraint_type", "") or "").strip().lower()
        ctx = dict(rule.get("context") or {})
        dest = str(ctx.get("destination", "") or "").strip()
        rule_res = {
            str(r or "").strip().lower()
            for r in (rule.get("resources") or [])
            if str(r or "").strip()
        }
        if ct in allowed_families and dest == destination and normalized_resources.issubset(rule_res):
            return deepcopy(rule)
    raise LookupError(
        f"No runtime safety rule for destination={destination!r} "
        f"resources={sorted(normalized_resources)!r}"
    )


# Module-level safety constants (evaluated once at import).
CASE3_BOARD_MUTEX_RULE = _find_runtime_safety_rule(
    destination="assembly_board-v1",
    required_resources={"ur5e", "xarm6"},
)
CASE3_BOARD_MUTEX_RULE_ID = str(CASE3_BOARD_MUTEX_RULE.get("id", "") or "").strip()

CASE3_LG_BEFORE_MCP_RULE_ID = "CASE3_LG_BEFORE_MCP_PRECEDENCE"
CASE3_LG_BEFORE_MCP_RULE: dict[str, Any] = {
    "id": CASE3_LG_BEFORE_MCP_RULE_ID,
    "constraint_type": "precedence",
    "text": "LG must be assembled at the assembly board before MCP may return to the assembly board.",
    "raw_text": "LG must be assembled at the assembly board before MCP may return to the assembly board.",
    "generated_interpretation": (
        "Treat LG completion as a safety-gated prerequisite before MCP resumes "
        "to the protected goal location."
    ),
    "resources": ["ur5e", "xarm6"],
    "context": {"before_part": "LG", "after_part": "MCP"},
}


# ---------------------------------------------------------------------------
# FakeProductAgent  (live LLM only — no scripted turns)
# ---------------------------------------------------------------------------


class FakeProductAgent:
    def __init__(
        self,
        *,
        tools_catalog: list[dict[str, Any]],
        product_geometry: dict[str, Any],
        llm_model: str | None = None,
        precomputed_bundle: dict[str, Any] | None = None,
    ) -> None:
        self.jid = "assembly_board-v1@localhost"
        self.logger = logging.getLogger("case3_bridge_dryrun")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.tools_catalog = deepcopy(tools_catalog)
        self.product_geometry = deepcopy(product_geometry)
        self.llm_model = str(llm_model or DEFAULT_LIVE_MODEL).strip()
        self.precomputed_bundle = deepcopy(precomputed_bundle or {})
        precomputed_policy = (
            self.precomputed_bundle.get("replan_policy", {})
            if isinstance(self.precomputed_bundle.get("replan_policy"), dict)
            else {}
        )
        self._bridge_reasoning_mode = str(
            precomputed_policy.get("bridge_reasoning_mode", "single_shot") or "single_shot"
        ).strip().lower()
        if self._bridge_reasoning_mode not in {"single_shot", "multi_turn"}:
            self._bridge_reasoning_mode = "single_shot"
        bundle_artifacts = dict(self.precomputed_bundle.get("artifacts") or {})
        self.structured_requirements_path = Path(
            str(bundle_artifacts.get("requirements_json") or "")
        ) if bundle_artifacts.get("requirements_json") else None
        self.prepared_bridge_request: dict[str, Any] | None = None
        self.turn_log: list[dict[str, Any]] = []
        self._turn_index = 0

    def _geometry_for_part(self, part_name: str) -> dict[str, Any]:
        """Mirror ProductAgent geometry lookup for bridge grounding context."""
        if not self.product_geometry:
            return {}
        board = self.product_geometry.get("assembly_board", {})
        parts = self.product_geometry.get("parts", {})
        slot_xy = board.get("slots", {}).get(part_name)
        if slot_xy is None:
            return {}
        return {
            "slot_xy": slot_xy,
            "part_height_m": parts.get("heights_m", {}).get(part_name),
            "model_name": parts.get("model_map", {}).get(part_name),
            "slot_floor_z_m": board.get("slot_floor_z_m"),
            "board_center": board.get("center", {}),
        }

    async def ask_llm(
        self,
        *,
        prompt: str,
        with_functions: bool = False,
        temperature: float = 0.0,
    ) -> str:
        del with_functions, temperature
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package required") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        client = OpenAI()

        def _call() -> str:
            r = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return (r.choices[0].message.content or "").strip()

        raw = await asyncio.to_thread(_call)
        self._turn_index += 1
        self.turn_log.append({"turn_index": self._turn_index, "prompt": prompt, "response": raw})
        return raw

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package required for live bridge") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        client = OpenAI()

        def _call() -> dict[str, Any]:
            msgs: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
            for _ in range(max_tool_rounds + 1):
                kwargs: dict[str, Any] = {
                    "model": self.llm_model,
                    "messages": msgs,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": response_format,
                    },
                }
                if tools:
                    kwargs["tools"] = tools
                r = client.chat.completions.create(**kwargs)
                choice = r.choices[0].message
                if getattr(choice, "tool_calls", None) and tool_executor:
                    msgs.append({
                        "role": "assistant",
                        "content": choice.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.tool_calls
                        ],
                    })
                    for tc in choice.tool_calls:
                        result = tool_executor(
                            tc.function.name,
                            json.loads(tc.function.arguments),
                        )
                        msgs.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, default=str),
                        })
                    continue
                return json.loads(choice.content or "{}")
            raise RuntimeError("Exceeded max tool rounds")

        parsed = await asyncio.to_thread(_call)
        self._turn_index += 1
        self.turn_log.append({
            "turn_index": self._turn_index,
            "model": self.llm_model,
            "prompt": prompt,
            "response": deepcopy(parsed),
        })
        return parsed


# ---------------------------------------------------------------------------
# FakeBridgeRobot
# ---------------------------------------------------------------------------


class FakeBridgeRobot:
    _BRIDGE_PRIMITIVES = frozenset(
        {
            "move_cartesian",
            "move_pose",
            "move_relative",
            "move_to_named_pose",
            "open_gripper",
            "close_gripper",
            "detect_parts",
            "compute_pick_targets",
            "compute_place_targets",
            "attach_part",
            "detach_part",
            "get_current_pose",
        }
    )

    def __init__(
        self,
        *,
        config: dict[str, Any],
        execution_env: str,
        current_state: str,
        held_part: str | None,
        gripper_state: str,
        pose_ref: str | None,
        position: dict[str, float],
        observations: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        env_block = dict(config.get(execution_env) or {})
        self.agent_name = str(config.get("jid") or "").strip()
        self.jid = self.agent_name
        self.execution_mode = "dry_run"
        self.static_capabilities = deepcopy(env_block.get("static_capabilities") or {})
        self.named_positions = deepcopy(env_block.get("named_positions") or {})
        self._current_state = str(current_state)
        self._held_part = held_part
        self._gripper_state = str(gripper_state)
        self._bridge_pose_ref = pose_ref
        self._position = deepcopy(position)
        self._observations = deepcopy(observations or {})

    def move_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        speed: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Move end-effector to an absolute Cartesian position.
        params:
          x: {type: number, description: "Target X coordinate in meters"}
          y: {type: number, description: "Target Y coordinate in meters"}
          z: {type: number, description: "Target Z coordinate in meters"}
          speed: {type: number, description: "Optional motion speed"}
        preconditions: {}
        effects:
          current_pose:
            pose_absolute_from_params: [x, y, z]
          current_pose_ref:
            set_unknown: true
        ---
        """
        self._position = {"x": float(x), "y": float(y), "z": float(z)}
        self._bridge_pose_ref = None
        return {"success": True, "message": "fake move_cartesian ok"}

    def move_pose(
        self,
        x: float,
        y: float,
        z: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
        speed: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Move end-effector to an absolute pose with orientation.
        params:
          x: {type: number, description: "Target X coordinate in meters"}
          y: {type: number, description: "Target Y coordinate in meters"}
          z: {type: number, description: "Target Z coordinate in meters"}
          qx: {type: number, description: "Quaternion X"}
          qy: {type: number, description: "Quaternion Y"}
          qz: {type: number, description: "Quaternion Z"}
          qw: {type: number, description: "Quaternion W"}
          speed: {type: number, description: "Optional motion speed"}
        preconditions: {}
        effects:
          current_pose:
            pose_absolute_from_params: [x, y, z]
          current_pose_ref:
            set_unknown: true
        ---
        """
        self._position = {"x": float(x), "y": float(y), "z": float(z)}
        self._bridge_pose_ref = None
        return {"success": True, "message": "fake move_pose ok"}

    def move_relative(
        self,
        dx: float,
        dy: float,
        dz: float,
        speed: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Move end-effector relative to its current position.
        params:
          dx: {type: number, description: "Delta X in meters"}
          dy: {type: number, description: "Delta Y in meters"}
          dz: {type: number, description: "Delta Z in meters"}
          speed: {type: number, description: "Optional motion speed"}
        preconditions:
          current_pose:
            exists: true
        effects:
          current_pose:
            pose_relative_from_params: [dx, dy, dz]
          current_pose_ref:
            set_unknown: true
        ---
        """
        self._position = {
            "x": float(self._position.get("x", 0.0)) + float(dx),
            "y": float(self._position.get("y", 0.0)) + float(dy),
            "z": float(self._position.get("z", 0.0)) + float(dz),
        }
        self._bridge_pose_ref = None
        return {"success": True, "message": "fake move_relative ok"}

    def move_to_named_pose(self, pose_name: str, speed: float | None = None) -> dict[str, Any]:
        """
        ---
        description: Move to a named joint configuration (e.g. 'home').
        params:
          pose_name: {type: string, description: "Named pose from robot manifest"}
          speed: {type: number, description: "Optional motion speed"}
        preconditions:
          held_part:
            equals: null
        effects:
          current_pose_ref:
            set_from_param: pose_name
          current_pose:
            set_unknown: true
          occupancy.location:
            set_from_param: pose_name
        ---
        """
        self._bridge_pose_ref = str(pose_name or "").strip() or None
        return {"success": True, "message": "fake move_to_named_pose ok"}

    def open_gripper(self) -> bool:
        """
        ---
        description: Open the robot gripper.
        params: {}
        preconditions: {}
        effects:
          gripper_state:
            set: open
        ---
        """
        self._gripper_state = "open"
        return True

    def close_gripper(self, position: float | None = None) -> bool:
        """
        ---
        description: Close the robot gripper.
        params:
          position: {type: number, description: "Optional gripper position override"}
        preconditions: {}
        effects:
          gripper_state:
            set: closed
        ---
        """
        self._gripper_state = "closed"
        return True

    def attach_part(
        self,
        model_name: str,
        link: str | None = None,
        part_name: str = "",
    ) -> dict[str, Any]:
        """
        ---
        description: Attach a part model to the robot gripper.
        params:
          model_name: {type: string, description: "Part model name"}
          link: {type: string, description: "Optional link override"}
          part_name: {type: string, description: "Optional canonical part name"}
        preconditions:
          held_part:
            equals: null
          gripper_state:
            equals: closed
        effects:
          held_part:
            set_from_param_any_of: [part_name, model_name]
        ---
        """
        self._held_part = str(part_name or model_name or "").strip() or None
        return {"success": True, "message": "fake attach_part ok"}

    def detach_part(
        self,
        model_name: str = "",
        link: str | None = None,
        assume_released_if_open: bool = False,
    ) -> dict[str, Any]:
        """
        ---
        description: Detach a part model from the robot gripper.
        params:
          model_name: {type: string, description: "Part model name"}
          link: {type: string, description: "Optional link override"}
          assume_released_if_open: {type: boolean, description: "Allow open-gripper release assumption"}
        preconditions:
          held_part:
            not_equals: null
          gripper_state:
            equals: open
        effects:
          held_part:
            set: null
        ---
        """
        self._held_part = None
        return {"success": True, "message": "fake detach_part ok"}

    def get_current_pose(self) -> dict[str, Any]:
        """
        ---
        description: Return the current end-effector pose in the base frame.
        params: {}
        preconditions: {}
        effects:
          current_pose_ref:
            set_unknown: true
        ---
        """
        return {
            "success": True,
            "message": "fake get_current_pose ok",
            "pose": {
                "x": float(self._position.get("x", 0.0)),
                "y": float(self._position.get("y", 0.0)),
                "z": float(self._position.get("z", 0.0)),
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        }

    def compute_pick_targets(
        self,
        part_name: str = "",
        product_geometry: dict[str, Any] | None = None,
        target_pose: dict[str, Any] | None = None,
        approach_height_override_m: float | None = None,
        ignore_current_height_for_travel_z: bool = False,
        min_pick_tcp_z_override_m: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Compute pick target positions from perception + geometry without moving.
        params:
          part_name: {type: string, description: "Name of the detected part to pick"}
          product_geometry: {type: object, description: "Optional geometry override dict"}
          target_pose: {type: object, description: "Optional known target pose"}
          approach_height_override_m: {type: number, description: "Optional vertical approach distance"}
          ignore_current_height_for_travel_z: {type: boolean}
          min_pick_tcp_z_override_m: {type: number}
        preconditions: {}
        effects: {}
        ---
        """
        target = dict(target_pose or {})
        pose = {
            "x": float(target.get("x", 0.0) or 0.0),
            "y": float(target.get("y", 0.0) or 0.0),
            "z": float(target.get("z", 1.0) or 1.0),
        }
        return {
            "success": True,
            "part_name": str(part_name or target.get("part_name") or ""),
            "model_name": "fake_model",
            "tx": pose["x"],
            "ty": pose["y"],
            "tz": pose["z"],
            "pick_z": pose["z"] + 0.02,
            "travel_z": pose["z"] + float(approach_height_override_m or 0.2),
            "approach_pose": {"x": pose["x"], "y": pose["y"], "z": pose["z"] + 0.2},
            "target_pose": {"x": pose["x"], "y": pose["y"], "z": pose["z"] + 0.02},
            "part_height": 0.08,
            "tcp_offset_z": -0.17,
            "pick_tcp_z": pose["z"] + 0.19,
            "start_x": float(self._position.get("x", 0.0)),
            "start_y": float(self._position.get("y", 0.0)),
            "start_z": float(self._position.get("z", 0.0)),
        }

    def compute_place_targets(
        self,
        pick_ctx: dict[str, Any] | None = None,
        product_geometry: dict[str, Any] | None = None,
        part_name: str = "",
        z_adjustment_m: float = 0.0,
        destination_location: str = "",
    ) -> dict[str, Any]:
        """
        ---
        description: Compute placement target positions from pick context + geometry without moving.
        params:
          part_name: {type: string, description: "Name of the held part to place"}
          pick_ctx: {type: object, description: "Optional output context from previous pick"}
          product_geometry: {type: object, description: "Optional geometry override dict"}
          z_adjustment_m: {type: number, description: "Extra Z vertical adjustment"}
          destination_location: {type: string, description: "Optional symbolic destination token"}
        preconditions: {}
        effects: {}
        ---
        """
        _ = destination_location
        pick = dict(pick_ctx or {})
        return {
            "success": True,
            "part_name": str(part_name or pick.get("part_name") or ""),
            "slot_x": float((pick.get("target_pose") or {}).get("x", 0.0) or 0.0),
            "slot_y": float((pick.get("target_pose") or {}).get("y", 0.0) or 0.0),
            "board_top_z": 1.02,
            "place_z": 1.05 + float(z_adjustment_m or 0.0),
            "place_tcp_z": 1.22 + float(z_adjustment_m or 0.0),
            "approach_pose": {"x": 0.0, "y": 0.0, "z": 1.10},
            "target_pose": {"x": 0.0, "y": 0.0, "z": 1.05 + float(z_adjustment_m or 0.0)},
            "part_height": 0.08,
            "tcp_offset_z": -0.17,
            "grasp_tcp_to_part_origin_z": 0.19,
            "model_name": "fake_model",
        }

    def detect_parts(self, part_name: str | None = None) -> list[dict[str, Any]]:
        """
        ---
        description: Detect parts via perception service. Optionally filter by part name.
        params:
          part_name: {type: string, description: "Filter results to this part name"}
        preconditions: {}
        effects: {}
        ---
        """
        if part_name:
            observation = deepcopy(self._observations.get(str(part_name).strip()) or {})
            if observation:
                return [observation]
            return []
        return [deepcopy(item) for item in self._observations.values()]

    def get_bridge_snapshot(self) -> dict[str, Any]:
        return get_resource_bridge_snapshot(self)

    def _is_pose_in_workspace(self, pose: dict[str, Any]) -> tuple[bool, str]:
        bounds = self.static_capabilities.get("workspace_bounds")
        if not bounds or not isinstance(bounds, dict):
            return True, "no workspace_bounds configured"
        violations: list[str] = []
        for axis in ("x", "y", "z"):
            val = pose.get(axis)
            if val is None:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            lo = bounds.get(f"{axis}_min_m")
            hi = bounds.get(f"{axis}_max_m")
            if lo is not None and val < float(lo):
                violations.append(f"{axis}={val:.4f} < {axis}_min_m={float(lo):.4f}")
            if hi is not None and val > float(hi):
                violations.append(f"{axis}={val:.4f} > {axis}_max_m={float(hi):.4f}")
        if violations:
            return False, f"pose outside workspace: {', '.join(violations)}"
        return True, "pose within workspace bounds"

    def bridge_feasibility_oracle(
        self,
        *,
        operation_kind: str,
        part_name: str | None,
        part_context: dict[str, Any],
        bridge_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        op = str(operation_kind or "").strip().lower()
        evidence = {
            "part_context": deepcopy(part_context),
            "bridge_snapshot": deepcopy(bridge_snapshot),
            "resource_jid": self.jid,
        }
        if op in {"clear", "home"}:
            return {"allowed": True, "reason": f"{op} always feasible", "evidence": evidence}
        target_pose: dict[str, Any] | None = None
        if op in {"pick", "pick_place"}:
            target_pose = part_context.get("observed_pose") or part_context.get("pose")
        elif op == "place":
            target_info = part_context.get("target") or {}
            target_pose = target_info.get("slot_pose") or target_info.get("pose")
        if target_pose is None:
            return {"allowed": True, "reason": f"no target pose for {op}", "evidence": evidence}
        inside, reason = self._is_pose_in_workspace(target_pose)
        evidence["checked_pose"] = deepcopy(target_pose)
        evidence["workspace_bounds"] = deepcopy(
            self.static_capabilities.get("workspace_bounds") or {}
        )
        return {"allowed": inside, "reason": reason, "evidence": evidence}

    async def execute_bridge_observation(
        self,
        primitive: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = deepcopy(params or {})
        primitive_name = str(primitive or "").strip()
        if primitive_name == "get_current_pose":
            return {
                "success": True,
                "message": "fake get_current_pose succeeded",
                "primitive": primitive_name,
                "params": payload,
                "observation": {"pose": deepcopy(self._position), "resource_jid": self.jid},
                "snapshot": self.get_bridge_snapshot(),
            }
        if primitive_name != "detect_parts":
            return {
                "success": False,
                "message": f"unsupported observation primitive '{primitive_name}'",
                "snapshot": self.get_bridge_snapshot(),
            }
        part_name = str(payload.get("part_name") or "").strip()
        if not part_name:
            for alt_key in ("part_names", "targets"):
                alt = payload.get(alt_key)
                if isinstance(alt, list) and len(alt) == 1:
                    candidate = str(alt[0] or "").strip()
                    if candidate:
                        part_name = candidate
                        break
        if not part_name and len(self._observations) == 1:
            part_name = next(iter(self._observations.keys()))
        observation = deepcopy(self._observations.get(part_name) or {})
        if not observation:
            return {
                "success": False,
                "message": f"no observation configured for part '{part_name}'",
                "snapshot": self.get_bridge_snapshot(),
            }
        return {
            "success": True,
            "message": f"fake detect_parts succeeded for {part_name}",
            "primitive": primitive_name,
            "params": payload,
            "observation": observation,
            "snapshot": self.get_bridge_snapshot(),
        }


# ---------------------------------------------------------------------------
# Slippage fixtures
# ---------------------------------------------------------------------------


def _task_node_by_id(plan_payload: dict[str, Any], task_id: str) -> dict[str, Any]:
    for node in (plan_payload.get("nodes") or []):
        if not isinstance(node, dict):
            continue
        if str(node.get("id") or "").strip() == str(task_id).strip():
            return deepcopy(node)
    return {}

def _origin_resource_location_for_part(plan_payload: dict[str, Any], part_name: str) -> str:
    for node in (plan_payload.get("nodes") or []):
        if not isinstance(node, dict):
            continue
        if str(node.get("type") or "").strip() != "task":
            continue
        if str(node.get("function_name") or "").strip() != "pick_grasp":
            continue
        params = dict(node.get("params") or {})
        if str(params.get("part_name") or "").strip() != str(part_name).strip():
            continue
        origin = str(params.get("origin_resource_location") or "").strip()
        if origin:
            return origin
    return ""


def _latest_completed_task_id_for_part(
    plan_payload: dict[str, Any],
    *,
    part_name: str,
    completed_task_ids: tuple[str, ...],
) -> str:
    best_task_id = ""
    best_sequence_index = -1
    completed = {str(task_id).strip() for task_id in completed_task_ids if str(task_id).strip()}
    for node in (plan_payload.get("nodes") or []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if node_id not in completed:
            continue
        params = dict(node.get("params") or {})
        if str(params.get("part_name") or "").strip() != str(part_name).strip():
            continue
        try:
            sequence_index = int(node.get("sequence_index"))
        except (TypeError, ValueError):
            sequence_index = -1
        if sequence_index >= best_sequence_index:
            best_sequence_index = sequence_index
            best_task_id = node_id
    return best_task_id


def _apply_runtime_status_snapshot(plan_nodes: list[dict[str, Any]]) -> None:
    completed = {str(task_id).strip() for task_id in CASE3_COMPLETED_TASK_IDS}
    for node in plan_nodes:
        if not isinstance(node, dict) or str(node.get("type") or "").strip() != "task":
            continue
        node_id = str(node.get("id") or "").strip()
        if node_id in completed:
            node["status"] = "completed"
        elif node_id == FAILED_TASK_ID:
            node["status"] = "failed"
        else:
            node["status"] = "pending"


def _merge_part_tracker(
    base_part_tracker: dict[str, Any],
    derived_part_tracker: dict[str, Any],
) -> dict[str, Any]:
    merged_part_tracker = deepcopy(base_part_tracker)
    for part_name, derived_entry in (derived_part_tracker or {}).items():
        if not isinstance(derived_entry, dict):
            continue
        current_entry = dict(merged_part_tracker.get(part_name) or {})
        force_keys = {
            str(key).strip()
            for key in (derived_entry.get("_force_keys") or [])
            if str(key).strip()
        }
        for key, value in derived_entry.items():
            if key == "_force_keys":
                continue
            if key not in force_keys and value in (None, "", [], {}):
                continue
            if (
                key in {"state", "location"}
                and key not in force_keys
                and current_entry.get(key) not in (None, "", "unknown")
            ):
                continue
            current_entry[key] = deepcopy(value)
        merged_part_tracker[str(part_name)] = current_entry
    return merged_part_tracker


def _contains_fixed_recovery_labels(value: Any) -> bool:
    fixed_tokens = (
        "ur5e_recovery_lane",
        "ur5e_base_area",
        "assembly_board_v1",
    )
    serialized = json.dumps(value, default=str)
    return any(token in serialized for token in fixed_tokens)


def _compact_prompt_condition_target(
    entry: dict[str, Any],
    *,
    include_entity: bool,
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    if include_entity:
        entity_kind = str(entry.get("entity_kind") or "").strip()
        entity = str(entry.get("entity") or "").strip()
        if entity_kind:
            compact["entity_kind"] = entity_kind
        if entity:
            compact["entity"] = entity
    field = str(entry.get("field") or "").strip()
    if field:
        compact["field"] = field
    if "expected" in entry:
        compact["expected"] = deepcopy(entry.get("expected"))
    return compact


def _expected_prompt_bridge_snapshot(
    resource_entry: dict[str, Any],
) -> dict[str, Any]:
    bridge_snapshot = dict(resource_entry.get("bridge_snapshot") or {})
    resource_core = dict(bridge_snapshot.get("resource_core") or {})
    resource_facets = dict(bridge_snapshot.get("resource_facets") or {})
    manipulator = dict(resource_facets.get("manipulator") or {})
    prompt_snapshot = {
        "resource_jid": str(
            resource_entry.get("resource_jid")
            or resource_core.get("resource_jid")
            or ""
        ).strip(),
        "resource_type": deepcopy(
            bridge_snapshot.get("resource_type")
            or resource_core.get("resource_type")
        ),
        "current_state": deepcopy(
            bridge_snapshot.get("current_state")
            if "current_state" in bridge_snapshot
            else resource_core.get("current_state")
        ),
        "current_location": deepcopy(
            bridge_snapshot.get("current_location")
            if "current_location" in bridge_snapshot
            else resource_core.get("current_location")
        ),
        "availability": deepcopy(
            bridge_snapshot.get("availability")
            if "availability" in bridge_snapshot
            else resource_core.get("availability")
        ),
        "active_work": deepcopy(
            bridge_snapshot.get("active_work")
            if "active_work" in bridge_snapshot
            else resource_core.get("active_work")
        ),
        "occupancy": deepcopy(
            bridge_snapshot.get("occupancy")
            or resource_core.get("occupancy")
            or {}
        ),
        "held_part": deepcopy(
            bridge_snapshot.get("held_part")
            if "held_part" in bridge_snapshot
            else manipulator.get("held_part")
        ),
        "gripper_state": deepcopy(
            bridge_snapshot.get("gripper_state")
            if "gripper_state" in bridge_snapshot
            else manipulator.get("gripper_state")
        ),
        "current_pose": deepcopy(
            bridge_snapshot.get("current_pose")
            if "current_pose" in bridge_snapshot
            else manipulator.get("current_pose")
        ),
        "current_pose_ref": deepcopy(
            bridge_snapshot.get("current_pose_ref")
            if "current_pose_ref" in bridge_snapshot
            else manipulator.get("current_pose_ref")
        ),
        "named_poses": deepcopy(
            bridge_snapshot.get("named_poses")
            if "named_poses" in bridge_snapshot
            else manipulator.get("named_poses")
        ),
    }
    return {
        key: value
        for key, value in prompt_snapshot.items()
        if value not in (None, "", [], {})
    }


def _expected_allowed_execution_surface(
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    context_summary = dict(prepared_bridge_request.get("context_summary") or {})
    fault_event = dict(context_summary.get("fault_event") or {})
    focused_resource_jid = str(fault_event.get("focused_resource_jid") or "").strip()
    current_product_state = dict(context_summary.get("current_product_state") or {})
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})

    observed_resources = [
        dict(row)
        for row in (current_product_state.get("resources") or [])
        if isinstance(row, dict)
    ]
    observed_by_jid = {
        str(row.get("resource_jid") or "").strip(): row
        for row in observed_resources
        if str(row.get("resource_jid") or "").strip()
    }

    ordered_resource_jids: list[str] = []
    seen_resource_jids: set[str] = set()

    def _append_resource_jid(raw_value: Any) -> None:
        resource_jid = str(raw_value or "").strip()
        if not resource_jid or resource_jid in seen_resource_jids:
            return
        seen_resource_jids.add(resource_jid)
        ordered_resource_jids.append(resource_jid)

    _append_resource_jid(focused_resource_jid)
    for row in observed_resources:
        _append_resource_jid(row.get("resource_jid"))
    for resource_jid in sorted(bridge_resources):
        _append_resource_jid(resource_jid)

    prompt_resources: list[dict[str, Any]] = []
    for resource_jid in ordered_resource_jids:
        bridge_entry = dict(bridge_resources.get(resource_jid) or {})
        adapter_capabilities = dict(bridge_entry.get("bridge_adapter") or {})
        if not adapter_capabilities.get("supports_executable_bridge"):
            continue
        observed_row = dict(observed_by_jid.get(resource_jid) or {})
        prompt_bridge_snapshot = _expected_prompt_bridge_snapshot(bridge_entry)
        pending_task_ids = [
            str(task.get("id") or "").strip()
            for task in (bridge_entry.get("pending_tasks") or [])
            if isinstance(task, dict) and str(task.get("id") or "").strip()
        ]
        prompt_primitive_catalog = [
            {
                key: deepcopy(value)
                for key, value in item.items()
                if key != "composite_expansion"
            }
            for item in (bridge_entry.get("primitive_catalog") or [])
            if isinstance(item, dict)
        ]
        prompt_resources.append(
            {
                "resource_jid": resource_jid,
                "role": deepcopy(
                    observed_row.get("role")
                    or ("focused" if resource_jid == focused_resource_jid else "supporting")
                ),
                "current_state": deepcopy(
                    observed_row.get("current_state")
                    if "current_state" in observed_row
                    else prompt_bridge_snapshot.get("current_state")
                ),
                "current_location": deepcopy(
                    observed_row.get("current_location")
                    if "current_location" in observed_row
                    else prompt_bridge_snapshot.get("current_location")
                ),
                "availability": deepcopy(
                    observed_row.get("availability")
                    if "availability" in observed_row
                    else prompt_bridge_snapshot.get("availability")
                ),
                "held_part": deepcopy(
                    observed_row.get("held_part")
                    if "held_part" in observed_row
                    else prompt_bridge_snapshot.get("held_part")
                ),
                "pending_task_ids": deepcopy(
                    observed_row.get("pending_task_ids")
                    if isinstance(observed_row.get("pending_task_ids"), list)
                    else pending_task_ids
                ),
                "prompt_bridge_snapshot": deepcopy(prompt_bridge_snapshot),
                "prompt_primitive_catalog": prompt_primitive_catalog,
            }
        )

    return {
        "focused_resource_jid": focused_resource_jid,
        "resources": prompt_resources,
    }


def _build_live_style_failure_payload(
    plan_payload: dict[str, Any],
    *,
    failed_resource_position: dict[str, float],
) -> dict[str, Any]:
    failed_task = _task_node_by_id(plan_payload, FAILED_TASK_ID)
    failed_resource_jid = str(failed_task.get("resource_jid") or "xarm6@localhost").strip()
    failed_function_name = str(failed_task.get("function_name") or "").strip()
    scenario_config = load_failure_scenario_config("lg_slippage")
    drop_pose = deepcopy((dict(scenario_config.get("injection") or {}).get("drop_pose") or {}))
    return build_failure_event(
        failed_task_id=FAILED_TASK_ID,
        failed_resource_jid=failed_resource_jid,
        failed_function_name=failed_function_name,
        final_status="failed",
        part_name="LG",
        base_failure_context=failure_context_from_scenario_config(scenario_config),
        observations={
            "last_commanded_location": str(
                dict(failed_task.get("params") or {}).get("destination_location") or ""
            ).strip(),
            "dropped_location": drop_pose,
        },
        state_before={
            "execution_mode": "simulation",
            "controller_ready": True,
            "held_part": "LG",
            "current_state": "positioned",
            "position": deepcopy(failed_resource_position),
            "gripper_state": "closed",
        },
        state_after={
            "execution_mode": "simulation",
            "controller_ready": True,
            "held_part": None,
            "current_state": "failed",
            "position": deepcopy(failed_resource_position),
            "gripper_state": "open",
        },
    )


def _live_style_part_order(plan_payload: dict[str, Any]) -> list[str]:
    ordered_parts: list[str] = []
    seen: set[str] = set()
    for task_id in CASE3_COMPLETED_TASK_IDS:
        node = _task_node_by_id(plan_payload, task_id)
        part_name = str((node.get("params") or {}).get("part_name") or "").strip()
        if not part_name or part_name in seen:
            continue
        seen.add(part_name)
        ordered_parts.append(part_name)
    for part_name in ("MCP", "LG"):
        if part_name not in seen:
            ordered_parts.append(part_name)
    return ordered_parts


def _build_live_style_slippage_fixture(
    *,
    planner: ProcessPlanner,
    plan_payload: dict[str, Any],
    xarm6_position: dict[str, float],
) -> dict[str, Any]:
    failure_payload = _build_live_style_failure_payload(
        plan_payload,
        failed_resource_position=xarm6_position,
    )
    part_defaults: dict[str, dict[str, Any]] = {
        "LG": {
            "state": "unknown",
            "location": "xarm6@localhost_gripper",
            "last_known_location": "xarm6@localhost_gripper",
            "last_successful_task": _latest_completed_task_id_for_part(
                plan_payload,
                part_name="LG",
                completed_task_ids=CASE3_COMPLETED_TASK_IDS,
            ),
            "origin_resource_location": _origin_resource_location_for_part(plan_payload, "LG"),
        },
        "MCP": {
            "state": "in_gripper",
            "location": "ur5e@localhost_gripper",
            "last_known_location": "ur5e@localhost_gripper",
            "last_successful_task": _latest_completed_task_id_for_part(
                plan_payload,
                part_name="MCP",
                completed_task_ids=CASE3_COMPLETED_TASK_IDS,
            ),
            "origin_resource_location": _origin_resource_location_for_part(plan_payload, "MCP"),
        },
    }
    base_part_tracker: dict[str, Any] = {
        part_name: deepcopy(part_defaults[part_name])
        for part_name in _live_style_part_order(plan_payload)
        if part_name in part_defaults
    }
    derived_part_tracker = planner._derive_part_tracker_from_violations([failure_payload])
    part_tracker = _merge_part_tracker(base_part_tracker, derived_part_tracker)
    part_states = {
        str(name): info.get("state")
        for name, info in part_tracker.items()
        if isinstance(info, dict)
    }
    part_locations = {
        str(name): info.get("location")
        for name, info in part_tracker.items()
        if isinstance(info, dict)
    }
    resource_states: dict[str, dict[str, Any]] = {
        "xarm6@localhost": {
            "current_state": "failed",
            "held_part": None,
            "current_location": None,
        },
        "ur5e@localhost": {
            "current_state": "picked",
            "held_part": "MCP",
            "current_location": None,
        },
    }
    stuck_state = planner._build_resource_search_state(
        resource_jid="xarm6@localhost",
        resource_states=resource_states,
        default_resource_state="idle",
        part_states=part_states,
        part_locations=part_locations,
    )
    return {
        "failed_task_id": FAILED_TASK_ID,
        "anchor_task_id": ANCHOR_TASK_ID,
        "goal_state": GOAL_STATE,
        "P_id": [name for name, state in part_states.items() if state != GOAL_STATE],
        "obligation_targets": [],
        "bridge_feedback": "",
        "default_resource_state": "idle",
        "part_tracker": part_tracker,
        "part_states": part_states,
        "part_locations": part_locations,
        "resource_states": resource_states,
        "stuck_state": stuck_state,
        "bridge_safety_context": {},
        "failure_context": failure_payload,
    }


def _build_synthetic_slippage_fixture() -> dict[str, Any]:
    """Build the richer synthetic case used only for optional full-run experiments."""
    scenario_config = load_failure_scenario_config("lg_slippage")
    lg_slippage_pose = deepcopy(
        (dict(scenario_config.get("injection") or {}).get("drop_pose") or {})
    )
    part_tracker: dict[str, Any] = {
        "LG": {
            "state": "misplaced",
            "location": None,
            "last_known_location": None,
            "observed_pose": deepcopy(lg_slippage_pose),
            "pose_source": "synthetic_fixture",
            "last_successful_task": ANCHOR_TASK_ID,
            "origin_resource_location": "prusa-mk4-1",
        },
        "MCP": {
            "state": "in_gripper",
            "location": "ur5e@localhost_gripper",
            "last_known_location": "ur5e@localhost_gripper",
            "observed_pose": None,
            "last_successful_task": "REQ_1_T2",
            "origin_resource_location": "prusa-mk4-2",
        },
    }
    part_states = {"LG": "misplaced", "MCP": "in_gripper"}
    part_locations = {
        "LG": None,
        "MCP": "ur5e@localhost_gripper",
    }
    resource_states: dict[str, dict[str, Any]] = {
        "xarm6@localhost": {
            "current_state": "failed",
            "held_part": None,
            "current_location": "assembly_board-v1",
        },
        "ur5e@localhost": {
            "current_state": "picked",
            "held_part": "MCP",
            "current_location": None,
        },
    }
    stuck_state: dict[str, Any] = {
        "resource_state": "failed",
        "current_part": None,
        "current_location": "assembly_board-v1",
        "part_states": deepcopy(part_states),
        "part_locations": deepcopy(part_locations),
    }
    return {
        "failed_task_id": FAILED_TASK_ID,
        "anchor_task_id": ANCHOR_TASK_ID,
        "goal_state": GOAL_STATE,
        "P_id": ["LG", "MCP"],
        "obligation_targets": [
            {"rule_id": CASE3_BOARD_MUTEX_RULE_ID, "resource_jid": "xarm6@localhost"}
        ],
        "bridge_feedback": "",
        "default_resource_state": "idle",
        "part_tracker": part_tracker,
        "part_states": part_states,
        "part_locations": part_locations,
        "resource_states": resource_states,
        "stuck_state": stuck_state,
        "bridge_safety_context": {
            "rule_ids": [CASE3_LG_BEFORE_MCP_RULE_ID, CASE3_BOARD_MUTEX_RULE_ID],
            "safe_next_task_ids": [],
            "running_aps": [],
            "candidate_aps": [],
            "predicted_state_aps": [],
            "status": "",
            "reason": "",
            "constraints": [
                {
                    "part_name": "MCP",
                    "forbidden_location": "assembly_board-v1",
                    "until_conditions": [
                        {"entity_kind": "part", "entity": "LG", "field": "state", "expected": "assembled"},
                        {"entity_kind": "part", "entity": "LG", "field": "location", "expected": "assembly_board-v1"},
                    ],
                    "reason": (
                        "resume-suffix parts may not be staged at their protected goal "
                        "location before bridge-replaced parts satisfy marked re-entry"
                    ),
                    "rule_id": CASE3_LG_BEFORE_MCP_RULE_ID,
                }
            ],
            "safety_rules": [
                deepcopy(CASE3_LG_BEFORE_MCP_RULE),
                deepcopy(CASE3_BOARD_MUTEX_RULE),
            ],
        },
    }


# ---------------------------------------------------------------------------
# Precondition helpers
# ---------------------------------------------------------------------------


def _relax_recovery_clear_precondition(prepared_bridge_request: dict[str, Any]) -> None:
    """Remove the not_equals:'recovery_required' precondition from move_to_named_pose.

    The bridge primitive catalog may block move_to_named_pose when the robot is in
    recovery_required state.  Since we're using 'failed', this is a no-op here but
    kept for consistency in case the catalog uses a different token.
    """
    def _rewrite_catalog(catalog: list[dict[str, Any]]) -> None:
        for row in catalog:
            if not isinstance(row, dict):
                continue
            if str(row.get("name") or "") != "move_to_named_pose":
                continue
            preconditions = dict(row.get("preconditions") or {})
            current_state = dict(preconditions.get("current_state") or {})
            if current_state.get("not_equals") in ("recovery_required", "failed"):
                current_state.pop("not_equals", None)
                if current_state:
                    preconditions["current_state"] = current_state
                else:
                    preconditions.pop("current_state", None)
                row["preconditions"] = preconditions

    bridge_resources = prepared_bridge_request.get("bridge_resources") or {}
    if isinstance(bridge_resources, dict):
        xarm_entry = bridge_resources.get("xarm6@localhost")
        if isinstance(xarm_entry, dict):
            _rewrite_catalog(list(xarm_entry.get("primitive_catalog") or []))


def _configure_live_bridge_session(prepared_bridge_request: dict[str, Any]) -> None:
    """Increase turn budget for live LLM runs."""
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["max_turns"] = max(int(bridge_session.get("max_turns", 6) or 6), 16)
    bridge_session["max_observations"] = max(int(bridge_session.get("max_observations", 3) or 3), 5)
    bridge_session["max_observe_batch"] = max(1, min(3, int(bridge_session.get("max_observe_batch", 3) or 3)))
    bridge_session["max_final_retries"] = max(int(bridge_session.get("max_final_retries", 2) or 2), 4)
    bridge_session["repair_mode"] = "recover"
    bridge_session["observation_backend"] = "mock_detect_parts_harness"
    prepared_bridge_request["bridge_session"] = bridge_session


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


async def _prepare_bridge_dryrun_harness(
    *,
    llm_model: str | None = None,
    configure_live: bool = True,
    fixture_mode: str = "live",
    bridge_reasoning_mode: str | None = None,
) -> tuple[dict[str, Any], FakeProductAgent, ProcessPlanner, dict[str, Any]]:
    """Load configs, build fake agents, prepare the bridge request."""
    paths = _case3_paths()
    tools_catalog = _load_json(paths["tools"])
    plan_payload = _load_json(paths["plan"])
    geometry_payload = _load_json(paths["geometry"])
    bundle_context = _case3_bundle_context(paths)
    if bridge_reasoning_mode is not None:
        bundle_context["replan_policy"] = {
            **dict(bundle_context.get("replan_policy") or {}),
            "bridge_reasoning_mode": str(bridge_reasoning_mode or "").strip(),
        }
    ur5e_config = _load_robot_config(paths["ur5e"], "ur5e")
    xarm6_config = _load_robot_config(paths["xarm6"], "xarm6")

    if not isinstance(tools_catalog, list):
        raise TypeError("tools catalog did not decode to a list")
    if not isinstance(plan_payload, dict):
        raise TypeError("case3 plan did not decode to an object")

    product_agent = FakeProductAgent(
        tools_catalog=tools_catalog,
        product_geometry=deepcopy(geometry_payload.get("gazebo") or {}),
        llm_model=llm_model,
        precomputed_bundle=bundle_context,
    )

    ur5e = FakeBridgeRobot(
        config=ur5e_config,
        execution_env="gazebo",
        current_state="picked",
        held_part="MCP",
        gripper_state="closed",
        pose_ref=None,
        position={"x": -0.25, "y": 0.22, "z": 1.18},
        observations={
            "MCP": {"part_name": "MCP", "pose": {"x": 0.0, "y": -0.08, "z": 1.025}},
        },
    )

    xarm6 = FakeBridgeRobot(
        config=xarm6_config,
        execution_env="gazebo",
        current_state="failed",   # slippage: place_insert() returned {"status": "failed"}
        held_part=None,
        gripper_state="open",
        pose_ref=None,
        position={"x": 0.1, "y": 0.08, "z": 1.1994999760206477},
        observations={},
    )

    planner = ProcessPlannerPrepareTrace(product_agent, [ur5e, xarm6])
    planner.nodes = deepcopy(plan_payload.get("nodes") or [])
    _apply_runtime_status_snapshot(planner.nodes)

    if str(fixture_mode or "").strip().lower() == "synthetic":
        fixture = _build_synthetic_slippage_fixture()
    else:
        fixture = _build_live_style_slippage_fixture(
            planner=planner,
            plan_payload=plan_payload,
            xarm6_position={"x": 0.1, "y": 0.08, "z": 1.1994999760206477},
        )

    async def _direct_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    with patch(
        "cais_spade_llm.agents.intelligent_product.process_planner.asyncio.to_thread",
        new=_direct_to_thread,
    ):
        prepared_bridge_request = await planner.prepare_bridge_request(
            stuck_state=deepcopy(fixture["stuck_state"]),
            P_id=deepcopy(fixture["P_id"]),
            ra_jid="xarm6@localhost",
            goal_state=str(fixture["goal_state"]),
            tools_catalog=deepcopy(tools_catalog),
            part_tracker=deepcopy(fixture["part_tracker"]),
            obligation_targets=deepcopy(fixture["obligation_targets"]),
            bridge_feedback=str(fixture["bridge_feedback"]),
            resource_states=deepcopy(fixture["resource_states"]),
            default_resource_state=str(fixture["default_resource_state"]),
            part_states=deepcopy(fixture["part_states"]),
            part_locations=deepcopy(fixture["part_locations"]),
            bridge_safety_context=deepcopy(fixture.get("bridge_safety_context") or {}),
            failure_context=deepcopy(fixture.get("failure_context") or {}),
        )

    product_agent.prepared_bridge_request = prepared_bridge_request
    if configure_live:
        _relax_recovery_clear_precondition(prepared_bridge_request)
        _configure_live_bridge_session(prepared_bridge_request)

    return fixture, product_agent, planner, prepared_bridge_request


# ---------------------------------------------------------------------------
# Main coroutine
# ---------------------------------------------------------------------------


async def run_case3_bridge_dryrun(
    write_debug: bool = True,
    *,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """Run the active bridge through the pre-LLM single-shot handoff."""
    fixture, product_agent, planner, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
        llm_model=llm_model,
        fixture_mode="live",
    )

    proposal = await planner.execute_prepared_bridge_request(prepared_bridge_request)
    bridge_debug = planner.get_last_bridge_debug()

    result: dict[str, Any] = {
        "scenario": "case3_lg_slippage",
        "status": str(bridge_debug.get("status") or ""),
        "proposal": proposal,
        "bridge_debug": bridge_debug,
        "prepared_bridge_request": prepared_bridge_request,
        "context_summary": deepcopy(prepared_bridge_request.get("context_summary") or {}),
        "llm_input": deepcopy(prepared_bridge_request.get("llm_input") or {}),
        "single_shot_prompt_input": deepcopy(
            prepared_bridge_request.get("single_shot_prompt_input") or {}
        ),
        "single_shot_prompt_text": str(
            prepared_bridge_request.get("single_shot_prompt_text") or ""
        ),
        "raw_response": str(
            ((bridge_debug.get("single_shot_turn") or {}).get("raw_response") or "")
        ),
        "turn_log": deepcopy(product_agent.turn_log),
        "prompt_artifact_path": None,
        "latest_prompt_artifact_path": None,
        "response_artifact_path": None,
        "latest_response_artifact_path": None,
    }

    if write_debug:
        artifact_paths = _write_debug_artifacts(result)
        result.update(artifact_paths)

    return result


async def run_case3_bridge_prepare_trace(
    write_debug: bool = True,
    *,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """Prepare the bridge request and stop before any LLM stage."""
    fixture, product_agent, planner, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
        llm_model=llm_model,
        configure_live=False,
        fixture_mode="live",
    )
    result: dict[str, Any] = {
        "scenario": "case3_lg_slippage",
        "status": "prepared_only",
        "fixture": fixture,
        "bridge_debug": planner.get_last_bridge_debug(),
        "prepared_bridge_request": prepared_bridge_request,
        "context_summary": deepcopy(prepared_bridge_request.get("context_summary") or {}),
        "llm_input": deepcopy(prepared_bridge_request.get("llm_input") or {}),
        "single_shot_prompt_input": deepcopy(
            prepared_bridge_request.get("single_shot_prompt_input") or {}
        ),
        "single_shot_prompt_text": str(
            prepared_bridge_request.get("single_shot_prompt_text") or ""
        ),
        "raw_response": "",
        "turn_log": deepcopy(product_agent.turn_log),
        "prompt_artifact_path": None,
        "latest_prompt_artifact_path": None,
        "response_artifact_path": None,
        "latest_response_artifact_path": None,
    }
    if write_debug:
        artifact_paths = _write_debug_artifacts(
            result,
            filename_prefix="bridge_case3_prepare",
        )
        result.update(artifact_paths)
    return result


def _write_debug_artifacts(
    payload: dict[str, Any],
    *,
    filename_prefix: str = "bridge_case3_slippage",
) -> dict[str, str]:
    debug_dir = Path(DEBUG_DIR)
    if not debug_dir.is_absolute():
        debug_dir = _repo_root() / debug_dir
    return write_bridge_artifacts(
        payload,
        phase_label=filename_prefix,
        debug_dir=debug_dir,
        write_latest=True,
        filename_prefix=filename_prefix,
    )


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def _assert_bridge_dryrun(result: dict[str, Any]) -> None:
    assert result.get("proposal") is None, "single-shot prepare slice should stop before any proposal"
    bridge_debug = result.get("bridge_debug") or {}
    assert bridge_debug.get("status") == "llm_output_recorded", (
        f"Bridge did not reach prompt-ready state; status={bridge_debug.get('status')!r}"
    )
    assert result.get("single_shot_prompt_input"), "single-shot prompt input is missing"
    assert str(result.get("single_shot_prompt_text") or "").strip(), "single-shot prompt text is missing"
    assert str(result.get("raw_response") or "").strip(), "raw LLM response is missing"


def _assert_bridge_prepare_trace(result: dict[str, Any]) -> None:
    assert result.get("status") == "prepared_only"
    prepared_bridge_request = result.get("prepared_bridge_request") or {}
    context_summary = result.get("context_summary") or {}
    llm_input = prepared_bridge_request.get("llm_input") or {}
    bridge_session = prepared_bridge_request.get("bridge_session") or {}
    single_shot_prompt_input = prepared_bridge_request.get("single_shot_prompt_input") or {}
    single_shot_prompt_text = str(prepared_bridge_request.get("single_shot_prompt_text") or "")
    assert prepared_bridge_request, "prepared bridge request is missing"
    assert context_summary, "context summary is missing"
    assert llm_input, "llm_input is missing"
    assert bridge_session.get("reasoning_mode") == "single_shot"
    primitive_catalog = list(prepared_bridge_request.get("primitive_catalog") or [])
    assert primitive_catalog, "focused primitive catalog is missing"
    assert isinstance(prepared_bridge_request.get("bridge_snapshot"), dict)
    assert single_shot_prompt_input, "single-shot prompt input is missing"
    assert single_shot_prompt_text.strip(), "single-shot prompt text is missing"

    fault_event = context_summary.get("fault_event") or {}
    assert fault_event.get("focused_resource_jid") == "xarm6@localhost"
    assert fault_event.get("blocked_at_task_id") == FAILED_TASK_ID
    assert fault_event.get("blocked_at_function") == "place_insert"
    assert "goal_state" not in fault_event
    assert "pending_parts" not in fault_event

    current_product_state = context_summary.get("current_product_state") or {}
    relevant_assembly_requirements = context_summary.get("relevant_assembly_requirements") or []
    resources = current_product_state.get("resources") or []
    parts = current_product_state.get("parts") or []
    active_safety_diagnosis = current_product_state.get("active_safety_diagnosis") or {}
    loaded_safety_rules = current_product_state.get("loaded_safety_rules") or []
    modeled_continuation_gap = context_summary.get("modeled_continuation_gap") or {}
    failure_context_raw = prepared_bridge_request.get("failure_context_raw") or {}
    failure_anchor = prepared_bridge_request.get("failure_anchor") or {}
    assert failure_context_raw, "failure context raw is missing"
    assert failure_anchor, "failure anchor is missing"
    assert resources, "resource summary is missing"
    assert parts, "part summary is missing"
    assert prepared_bridge_request.get("requirement_nodes"), "requirement inventory is missing"
    assert prepared_bridge_request.get("requirements_status"), "requirements status is missing"
    assert prepared_bridge_request.get("loaded_safety_rules"), "loaded safety rules are missing"
    assert modeled_continuation_gap.get("goal_state") == GOAL_STATE
    assert modeled_continuation_gap.get("pending_nominal_task_ids"), "pending continuation summary is missing"
    assert modeled_continuation_gap.get("blocking_reasons"), "blocking reasons are missing"
    assert FAILED_TASK_ID not in (modeled_continuation_gap.get("pending_nominal_task_ids") or [])
    assert "REQ_2_T5" in (modeled_continuation_gap.get("pending_nominal_task_ids") or [])

    parts_by_name = {
        str(row.get("part_name") or ""): row
        for row in parts
        if isinstance(row, dict) and str(row.get("part_name") or "")
    }
    lg_row = parts_by_name.get("LG") or {}
    assert lg_row.get("location") in (None, "")
    assert lg_row.get("observed_pose"), "LG observed pose should come from shared failure context"
    assert not (active_safety_diagnosis.get("obligation_targets") or []), "live-style default should not inject obligations"
    assert not (active_safety_diagnosis.get("rule_ids") or []), "live-style default should not inject safety rules"
    loaded_rule_ids = {
        str(rule.get("rule_id") or "")
        for rule in loaded_safety_rules
        if isinstance(rule, dict) and str(rule.get("rule_id") or "")
    }
    assert loaded_rule_ids >= {"SAFE_1", "SAFE_2"}, "loaded safety rules should include the verified bundle rules"

    requirement_map = {
        str(entry.get("requirement_id") or ""): entry
        for entry in relevant_assembly_requirements
        if isinstance(entry, dict) and str(entry.get("requirement_id") or "")
    }
    assert set(requirement_map) >= {"REQ_1", "REQ_2"}, "relevant assembly requirements should include MCP and LG goals"
    assert requirement_map["REQ_1"].get("status") in {"pending", "in_progress"}
    assert "MCP" in str(requirement_map["REQ_1"].get("summary") or "")
    assert requirement_map["REQ_2"].get("status") == "failed"
    assert "LG" in str(requirement_map["REQ_2"].get("summary") or "")
    goal_conditions = modeled_continuation_gap.get("goal_conditions") or []
    assert goal_conditions, "goal conditions are missing"
    goal_condition_keys = {
        (
            str(entry.get("condition_family") or ""),
            str(entry.get("entity") or ""),
            str(entry.get("field") or ""),
            entry.get("expected"),
        )
        for entry in goal_conditions
        if isinstance(entry, dict)
    }
    assert ("goal", "LG", "state", GOAL_STATE) in goal_condition_keys
    assert ("goal", "MCP", "state", GOAL_STATE) in goal_condition_keys

    continuation_requirements = modeled_continuation_gap.get("continuation_requirements") or []
    idle_requirement = next(
        (
            entry for entry in continuation_requirements
            if isinstance(entry, dict)
            and entry.get("entity") == "xarm6@localhost"
            and entry.get("field") == "current_state"
            and entry.get("expected") == "idle"
        ),
        None,
    )
    assert idle_requirement, "continuation idle requirement is missing"
    assert idle_requirement.get("source_task_id") == "REQ_2_T5"
    assert idle_requirement.get("source_function_name") == "move_home"

    unsatisfied_goal_conditions = modeled_continuation_gap.get("unsatisfied_goal_conditions") or []
    goal_unsatisfied_keys = {
        (
            str(entry.get("condition_family") or ""),
            str(entry.get("entity") or ""),
            str(entry.get("field") or ""),
            entry.get("expected"),
            entry.get("actual"),
        )
        for entry in unsatisfied_goal_conditions
        if isinstance(entry, dict)
    }
    assert ("goal", "LG", "state", GOAL_STATE, "unknown") in goal_unsatisfied_keys
    assert ("goal", "MCP", "state", GOAL_STATE, "in_gripper") in goal_unsatisfied_keys

    unsatisfied_conditions = modeled_continuation_gap.get("unsatisfied_conditions") or []
    assert unsatisfied_conditions, "unsatisfied conditions are missing"
    families = {
        str(entry.get("condition_family") or "")
        for entry in unsatisfied_conditions
        if isinstance(entry, dict)
    }
    assert "goal" in families, "goal-family unsatisfied conditions are missing"
    assert "continuation" in families, "continuation-family unsatisfied conditions are missing"
    assert modeled_continuation_gap.get("resume_ready") is False
    assert llm_input.get("fault_event", {}).get("blocked_at_task_id") == FAILED_TASK_ID

    observed_runtime_state = llm_input.get("observed_runtime_state") or {}
    llm_resources = observed_runtime_state.get("resources") or []
    llm_parts = observed_runtime_state.get("parts") or []
    assert llm_resources, "llm_input observed resources are missing"
    assert llm_parts, "llm_input observed parts are missing"

    llm_parts_by_name = {
        str(row.get("part_name") or ""): row
        for row in llm_parts
        if isinstance(row, dict) and str(row.get("part_name") or "")
    }
    llm_lg_row = llm_parts_by_name.get("LG") or {}
    assert llm_lg_row.get("observed_pose"), "llm_input should retain LG observed pose"
    assert "target_location" not in llm_lg_row, "llm_input parts should not include modeled target fields"

    llm_resource_by_jid = {
        str(row.get("resource_jid") or ""): row
        for row in llm_resources
        if isinstance(row, dict) and str(row.get("resource_jid") or "")
    }
    xarm6_runtime_row = llm_resource_by_jid.get("xarm6@localhost") or {}
    assert xarm6_runtime_row.get("current_state") == "failed"
    assert "pending_task_ids" not in xarm6_runtime_row, "llm_input observed resources should not include pending task ids"

    llm_loaded_rule_ids = {
        str(rule.get("rule_id") or "")
        for rule in (llm_input.get("loaded_safety_rules") or [])
        if isinstance(rule, dict) and str(rule.get("rule_id") or "")
    }
    assert llm_loaded_rule_ids >= {"SAFE_1", "SAFE_2"}
    assert isinstance(llm_input.get("obligation_targets"), list)

    llm_requirement_ids = {
        str(entry.get("requirement_id") or "")
        for entry in (llm_input.get("relevant_assembly_requirements") or [])
        if isinstance(entry, dict) and str(entry.get("requirement_id") or "")
    }
    assert llm_requirement_ids >= {"REQ_1", "REQ_2"}

    llm_gap = llm_input.get("modeled_continuation_gap") or {}
    assert llm_gap.get("goal_state") == GOAL_STATE
    continuation_gap_entries = llm_gap.get("unmet_continuation_conditions") or []
    idle_continuation_gap = next(
        (
            entry for entry in continuation_gap_entries
            if isinstance(entry, dict)
            and entry.get("entity") == "xarm6@localhost"
            and entry.get("field") == "current_state"
            and entry.get("expected") == "idle"
            and entry.get("actual") == "failed"
        ),
        None,
    )
    assert idle_continuation_gap, "llm_input should retain unmet continuation condition for xarm6"
    assert "bridge_debug" not in llm_input
    assert "data_flow_trace" not in llm_input
    assert "requirement_task_index" not in llm_input
    assert "task_requirement_map" not in llm_input

    allowed_execution_surface = llm_input.get("allowed_execution_surface") or {}
    assert allowed_execution_surface == _expected_allowed_execution_surface(
        prepared_bridge_request
    )
    allowed_surface_resources = allowed_execution_surface.get("resources") or []
    allowed_resource_by_jid = {
        str(row.get("resource_jid") or ""): row
        for row in allowed_surface_resources
        if isinstance(row, dict) and str(row.get("resource_jid") or "")
    }
    assert set(allowed_resource_by_jid) >= {"ur5e@localhost", "xarm6@localhost"}
    for resource_jid, resource_entry in allowed_resource_by_jid.items():
        prompt_bridge_snapshot = resource_entry.get("prompt_bridge_snapshot") or {}
        prompt_primitive_catalog = resource_entry.get("prompt_primitive_catalog") or []
        assert prompt_bridge_snapshot.get("resource_jid") == resource_jid
        primitive_names = {
            str(entry.get("name") or "")
            for entry in prompt_primitive_catalog
            if isinstance(entry, dict) and str(entry.get("name") or "")
        }
        assert {"grasp_part", "release_part"} <= primitive_names
        assert not (
            primitive_names
            & {"open_gripper", "close_gripper", "attach_part", "detach_part"}
        )
        for primitive_entry in prompt_primitive_catalog:
            assert "composite_expansion" not in primitive_entry

    assert single_shot_prompt_input.get("reasoning_mode") == "single_shot"
    assert single_shot_prompt_input.get("llm_input") == llm_input
    assert "obligation_targets" not in single_shot_prompt_input
    assert "primitive_catalog" not in single_shot_prompt_input
    assert "bridge_snapshot" not in single_shot_prompt_input
    assert "focused_bridge_snapshot" not in single_shot_prompt_input
    assert "bridge_resources" not in single_shot_prompt_input
    assert "other_resources_summary" not in single_shot_prompt_input
    proposal_success_criteria = single_shot_prompt_input.get("proposal_success_criteria") or {}
    assert proposal_success_criteria.get("focused_resource_jid") == llm_input.get("fault_event", {}).get(
        "focused_resource_jid"
    )
    assert proposal_success_criteria.get("resume_ready_now") == llm_gap.get("resume_ready")
    assert proposal_success_criteria.get("target_resume_ready") is True
    assert proposal_success_criteria.get("protected_nominal_task_suffix") == llm_gap.get(
        "pending_nominal_task_ids"
    )
    assert proposal_success_criteria.get("goal_targets_to_improve") == [
        _compact_prompt_condition_target(entry, include_entity=True)
        for entry in (llm_gap.get("unmet_goal_conditions") or [])
        if isinstance(entry, dict)
    ]
    must_satisfy = proposal_success_criteria.get("focused_resource_targets") or []
    assert must_satisfy == [
        _compact_prompt_condition_target(entry, include_entity=False)
        for entry in continuation_gap_entries
        if isinstance(entry, dict)
    ]
    current_product_state = context_summary.get("current_product_state") or {}
    active_safety_diagnosis = (current_product_state.get("active_safety_diagnosis") or {})
    focused_resource_jid = llm_input.get("fault_event", {}).get("focused_resource_jid")
    obligation_rule_ids = proposal_success_criteria.get("obligation_rule_ids_to_preserve") or []
    assert obligation_rule_ids == [
        str(target.get("rule_id") or "")
        for target in (active_safety_diagnosis.get("obligation_targets") or [])
        if isinstance(target, dict)
        and target.get("resource_jid") == focused_resource_jid
        and str(target.get("rule_id") or "")
    ]
    response_contract = single_shot_prompt_input.get("response_contract") or {}
    assert response_contract.get("top_level_required_fields") == ["primary_obligation", "macro_tasks"]
    assert "Loaded Safety Rules" in single_shot_prompt_text
    assert "Relevant Assembly Requirements" in single_shot_prompt_text
    assert "Modeled Continuation Gap" in single_shot_prompt_text
    assert "Proposal Success Criteria" in single_shot_prompt_text
    assert "Required JSON Response Contract" in single_shot_prompt_text
    assert "Hard Constraints" in single_shot_prompt_text
    assert "Allowed Execution Surface" in single_shot_prompt_text
    assert "prompt_bridge_snapshot" in single_shot_prompt_text
    assert "prompt_primitive_catalog" in single_shot_prompt_text
    assert "bridge_resources" not in single_shot_prompt_text
    assert "focused_primitive_catalog" not in single_shot_prompt_text
    assert "other_resources_summary" not in single_shot_prompt_text
    assert "grasp_part" in single_shot_prompt_text
    assert "release_part" in single_shot_prompt_text
    assert "open_gripper" not in single_shot_prompt_text
    assert "close_gripper" not in single_shot_prompt_text
    assert "attach_part" not in single_shot_prompt_text
    assert "detach_part" not in single_shot_prompt_text
    assert "ur5e@localhost" in single_shot_prompt_text
    assert "xarm6@localhost" in single_shot_prompt_text

    assert not _contains_fixed_recovery_labels(prepared_bridge_request), "prepared request still contains fixed recovery labels"

    bridge_debug = prepared_bridge_request.get("bridge_debug") or {}
    single_shot_turn = bridge_debug.get("single_shot_turn") or {}
    assert single_shot_turn.get("reasoning_mode") == "single_shot"
    assert single_shot_turn.get("status") == "prepared_for_llm"
    assert single_shot_turn.get("prompt_input") == single_shot_prompt_input
    assert single_shot_turn.get("prompt_text") == single_shot_prompt_text

    bridge_session_source = Path(
        "cais_spade_llm/agents/intelligent_product/replanner/llm_bridge/bridge_session.py"
    ).read_text(encoding="utf-8")
    bridge_prompts_source = Path(
        "cais_spade_llm/agents/intelligent_product/replanner/llm_bridge/bridge_prompts.py"
    ).read_text(encoding="utf-8")
    bridge_normalization_source = Path(
        "cais_spade_llm/agents/intelligent_product/replanner/llm_bridge/bridge_resource_normalization.py"
    ).read_text(encoding="utf-8")
    primitive_semantics_source = Path(
        "cais_spade_llm/agents/intelligent_product/replanner/llm_bridge/primitive_semantics.py"
    ).read_text(encoding="utf-8")
    assert "llm_bridge.v3" not in bridge_session_source
    assert "llm_bridge.v3" not in bridge_prompts_source
    assert "llm_bridge.v3" not in bridge_normalization_source
    assert "llm_bridge.v3.primitive_semantics" not in primitive_semantics_source

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _print_prepare_context_summary(context_summary)
    rendered = buffer.getvalue()
    fault_pos = rendered.find("Fault Event")
    current_pos = rendered.find("Current Product State")
    active_safety_pos = rendered.find("Active Safety Diagnosis")
    loaded_safety_pos = rendered.find("Loaded Safety Rules")
    assembly_req_pos = rendered.find("Relevant Assembly Requirements")
    gap_pos = rendered.find("Modeled Continuation Gap")
    assert fault_pos != -1, "Fault Event heading is missing"
    assert current_pos != -1, "Current Product State heading is missing"
    assert active_safety_pos == -1, "empty Active Safety Diagnosis should be hidden"
    assert loaded_safety_pos != -1, "Loaded Safety Rules heading is missing"
    assert assembly_req_pos != -1, "Relevant Assembly Requirements heading is missing"
    assert gap_pos != -1, "Modeled Continuation Gap heading is missing"
    assert fault_pos < current_pos < loaded_safety_pos < assembly_req_pos < gap_pos, "prepare-trace section order is incorrect"


def _format_pose_brief(value: Any) -> str:
    if not isinstance(value, dict):
        return "-"
    try:
        return "({x:.3f}, {y:.3f}, {z:.3f})".format(
            x=float(value.get("x", 0.0)),
            y=float(value.get("y", 0.0)),
            z=float(value.get("z", 0.0)),
        )
    except (TypeError, ValueError):
        return str(value)


def _print_prepare_context_summary(context_summary: dict[str, Any]) -> None:
    fault_event = dict(context_summary.get("fault_event") or {})
    current_product_state = dict(context_summary.get("current_product_state") or {})
    relevant_assembly_requirements = list(context_summary.get("relevant_assembly_requirements") or [])
    modeled_continuation_gap = dict(context_summary.get("modeled_continuation_gap") or {})
    resources = list(current_product_state.get("resources") or [])
    parts = list(current_product_state.get("parts") or [])
    active_safety_diagnosis = dict(current_product_state.get("active_safety_diagnosis") or {})
    loaded_safety_rules = list(current_product_state.get("loaded_safety_rules") or [])
    has_active_safety_diagnosis = bool(
        (active_safety_diagnosis.get("obligation_targets") or [])
        or (active_safety_diagnosis.get("rule_ids") or [])
        or (active_safety_diagnosis.get("active_rules") or [])
        or str(active_safety_diagnosis.get("status") or "").strip()
        or str(active_safety_diagnosis.get("reason") or "").strip()
    )

    def _derived_from(entry: dict[str, Any]) -> str:
        task_id = str(entry.get("source_task_id") or "").strip()
        function_name = str(entry.get("source_function_name") or "").strip()
        if not task_id and not function_name:
            return "-"
        return f"{task_id or '-'}" + (f"/{function_name}" if function_name else "")

    print()
    print("Fault Event")
    print(f"  focused_resource_jid: {fault_event.get('focused_resource_jid')}")
    print(f"  blocked_at_task_id:   {fault_event.get('blocked_at_task_id') or '-'}")
    print(f"  blocked_at_function:  {fault_event.get('blocked_at_function') or '-'}")
    print(f"  resource_state:       {fault_event.get('resource_state') or '-'}")

    print()
    print("Current Product State")
    print("Resources")
    for row in resources:
        if not isinstance(row, dict):
            continue
        print(
            "  {jid}: state={state}, held_part={held}, location={location} [{basis}], "
            "availability={availability}, pending={pending}".format(
                jid=row.get("resource_jid") or "-",
                state=row.get("current_state") or "-",
                held=row.get("held_part") or "-",
                location=row.get("current_location") or "-",
                basis=row.get("current_location_basis") or "-",
                availability=row.get("availability") or "-",
                pending=row.get("pending_task_ids") or [],
            )
        )

    print()
    print("Parts")
    for row in parts:
        if not isinstance(row, dict):
            continue
        print(
            "  {part}: state={state}, location={location} [{basis}], observed_pose={pose}, "
            "target_location={target}".format(
                part=row.get("part_name") or "-",
                state=row.get("state") or "-",
                location=row.get("location") or "-",
                basis=row.get("location_basis") or "-",
                pose=_format_pose_brief(row.get("observed_pose")),
                target=row.get("target_location") or "-",
            )
        )

    if has_active_safety_diagnosis:
        print()
        print("Active Safety Diagnosis")
        print(f"  obligation_targets: {active_safety_diagnosis.get('obligation_targets') or []}")
        print(f"  active_rule_ids:    {active_safety_diagnosis.get('rule_ids') or []}")
        if str(active_safety_diagnosis.get("status") or "").strip():
            print(f"  status:             {active_safety_diagnosis.get('status')}")
        if str(active_safety_diagnosis.get("reason") or "").strip():
            print(f"  reason:             {active_safety_diagnosis.get('reason')}")
        for rule in (active_safety_diagnosis.get("active_rules") or [])[:4]:
            if not isinstance(rule, dict):
                continue
            print(
                "  active_rule: {rule_id} [{constraint_type}] {summary}".format(
                    rule_id=rule.get("rule_id") or "-",
                    constraint_type=rule.get("constraint_type") or "-",
                    summary=rule.get("summary") or "-",
                )
            )

    print()
    print("Loaded Safety Rules")
    if not loaded_safety_rules:
        print("  rules: []")
    for rule in loaded_safety_rules[:6]:
        if not isinstance(rule, dict):
            continue
        print(
            "  loaded_rule: {rule_id} [{constraint_type}] {summary}".format(
                rule_id=rule.get("rule_id") or "-",
                constraint_type=rule.get("constraint_type") or "-",
                summary=rule.get("summary") or "-",
            )
        )

    print()
    print("Relevant Assembly Requirements")
    if not relevant_assembly_requirements:
        print("  requirements: []")
    for requirement in relevant_assembly_requirements:
        if not isinstance(requirement, dict):
            continue
        print(
            "  {requirement_id} [{status}] {summary}".format(
                requirement_id=requirement.get("requirement_id") or "-",
                status=requirement.get("status") or "unknown",
                summary=requirement.get("summary") or "-",
            )
        )

    print()
    print("Modeled Continuation Gap")
    print(f"  goal_state:               {modeled_continuation_gap.get('goal_state') or '-'}")
    print(f"  pending_nominal_task_ids: {modeled_continuation_gap.get('pending_nominal_task_ids') or []}")
    print(f"  resume_ready:             {modeled_continuation_gap.get('resume_ready')}")
    for entry in (modeled_continuation_gap.get("unsatisfied_conditions") or [])[:8]:
        if not isinstance(entry, dict):
            continue
        condition_family = str(entry.get("condition_family") or "").strip()
        if condition_family == "goal":
            print(
                "  unmet goal: {entity}.{field} expected={expected!r} actual={actual!r}".format(
                    entity=entry.get("entity") or "?",
                    field=entry.get("field") or "?",
                    expected=entry.get("expected"),
                    actual=entry.get("actual"),
                )
            )
            continue
        print(
            "  unmet continuation: {entity}.{field} expected={expected!r} actual={actual!r} "
            "derived_from={derived_from}".format(
                entity=entry.get("entity") or "?",
                field=entry.get("field") or "?",
                expected=entry.get("expected"),
                actual=entry.get("actual"),
                derived_from=_derived_from(entry),
            )
        )


def _print_llm_input(llm_input: dict[str, Any]) -> None:
    print()
    print("LLM Input")
    print(json.dumps(llm_input, indent=2, default=str))


def _print_single_shot_prompt(prompt_text: str) -> None:
    print()
    print("Single-Shot Prompt")
    print(prompt_text or "")


def _print_debug_artifact_paths(result: dict[str, Any]) -> None:
    prompt_artifact_path = result.get("prompt_artifact_path")
    latest_prompt_artifact_path = result.get("latest_prompt_artifact_path")
    response_artifact_path = result.get("response_artifact_path")
    latest_response_artifact_path = result.get("latest_response_artifact_path")
    if (
        not prompt_artifact_path
        and not latest_prompt_artifact_path
        and not response_artifact_path
        and not latest_response_artifact_path
    ):
        return
    print()
    if prompt_artifact_path:
        print("Prompt artifact:            ", prompt_artifact_path)
    if latest_prompt_artifact_path:
        print("Latest prompt artifact:     ", latest_prompt_artifact_path)
    if response_artifact_path:
        print("Response artifact:          ", response_artifact_path)
    if latest_response_artifact_path:
        print("Latest response artifact:   ", latest_response_artifact_path)


def _assert_completed_parts_drop_out_of_unsatisfied_goals(
    planner: ProcessPlannerPrepareTrace,
    fixture: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> None:
    variant_request = deepcopy(prepared_bridge_request)
    variant_request["part_tracker"]["MCP"] = {
        **dict(variant_request.get("part_tracker", {}).get("MCP") or {}),
        "state": GOAL_STATE,
        "location": "assembly_board-v1",
        "last_known_location": "assembly_board-v1",
    }
    variant_parts = dict((variant_request.get("grounding_context") or {}).get("parts") or {})
    variant_parts["MCP"] = {
        **dict(variant_parts.get("MCP") or {}),
        "state": GOAL_STATE,
        "location": "assembly_board-v1",
    }
    variant_request["grounding_context"] = {
        **dict(variant_request.get("grounding_context") or {}),
        "parts": variant_parts,
    }
    context_summary = planner._build_context_summary(
        variant_request,
        input_bridge_safety_context=deepcopy(fixture.get("bridge_safety_context") or {}),
    )
    modeled_gap = context_summary.get("modeled_continuation_gap") or {}
    unsatisfied_goal_conditions = modeled_gap.get("unsatisfied_goal_conditions") or []
    unsatisfied_mcp_conditions = [
        entry
        for entry in unsatisfied_goal_conditions
        if isinstance(entry, dict) and entry.get("entity") == "MCP"
    ]
    assert not unsatisfied_mcp_conditions, "completed MCP predicates should not remain unsatisfied"
    unsatisfied_lg_conditions = [
        entry
        for entry in unsatisfied_goal_conditions
        if isinstance(entry, dict) and entry.get("entity") == "LG"
    ]
    assert unsatisfied_lg_conditions, "unfinished LG predicates should remain unsatisfied"


def _assert_requirement_filtering(
    planner: ProcessPlannerPrepareTrace,
    fixture: dict[str, Any],
    prepared_bridge_request: dict[str, Any],
) -> None:
    variant_request = deepcopy(prepared_bridge_request)
    variant_request["P_id"] = ["LG"]
    variant_request["part_tracker"]["MCP"] = {
        **dict(variant_request.get("part_tracker", {}).get("MCP") or {}),
        "state": GOAL_STATE,
        "location": "assembly_board-v1",
        "last_known_location": "assembly_board-v1",
    }
    variant_parts = dict((variant_request.get("grounding_context") or {}).get("parts") or {})
    variant_parts["MCP"] = {
        **dict(variant_parts.get("MCP") or {}),
        "state": GOAL_STATE,
        "location": "assembly_board-v1",
    }
    variant_request["grounding_context"] = {
        **dict(variant_request.get("grounding_context") or {}),
        "parts": variant_parts,
    }
    variant_bridge_resources = dict(variant_request.get("bridge_resources") or {})
    variant_bridge_resources["ur5e@localhost"] = {
        **dict(variant_bridge_resources.get("ur5e@localhost") or {}),
        "pending_tasks": [],
    }
    variant_request["bridge_resources"] = variant_bridge_resources
    context_summary = planner._build_context_summary(
        variant_request,
        input_bridge_safety_context=deepcopy(fixture.get("bridge_safety_context") or {}),
    )
    relevant_assembly_requirements = context_summary.get("relevant_assembly_requirements") or []
    requirement_ids = [
        str(entry.get("requirement_id") or "")
        for entry in relevant_assembly_requirements
        if isinstance(entry, dict) and str(entry.get("requirement_id") or "")
    ]
    assert requirement_ids == ["REQ_2"], "only the LG requirement should remain relevant"


def _assert_synthetic_safety_rules_render() -> None:
    fixture = _build_synthetic_slippage_fixture()
    context_summary = {
        "fault_event": {},
        "current_product_state": {
            "resources": [],
            "parts": [],
            "active_safety_diagnosis": {
                "obligation_targets": deepcopy(fixture.get("obligation_targets") or []),
                "rule_ids": deepcopy(
                    (fixture.get("bridge_safety_context") or {}).get("rule_ids") or []
                ),
                "active_rules": [
                    {
                        "rule_id": str(rule.get("id") or "").strip(),
                        "constraint_type": str(rule.get("constraint_type") or "").strip(),
                        "summary": str(
                            rule.get("generated_interpretation")
                            or rule.get("text")
                            or rule.get("raw_text")
                            or ""
                        ).strip(),
                    }
                    for rule in ((fixture.get("bridge_safety_context") or {}).get("safety_rules") or [])
                    if isinstance(rule, dict)
                ],
            },
            "loaded_safety_rules": [
                {
                    "rule_id": str(CASE3_BOARD_MUTEX_RULE.get("id") or "").strip(),
                    "constraint_type": str(CASE3_BOARD_MUTEX_RULE.get("constraint_type") or "").strip(),
                    "summary": str(
                        CASE3_BOARD_MUTEX_RULE.get("generated_interpretation")
                        or CASE3_BOARD_MUTEX_RULE.get("text")
                        or CASE3_BOARD_MUTEX_RULE.get("raw_text")
                        or ""
                    ).strip(),
                }
            ],
            "formal_state": {},
        },
        "relevant_assembly_requirements": [
            {
                "requirement_id": "REQ_2",
                "summary": "xarm6 Assemble LG from prusa-mk4-1 to the Assembly Station.",
                "status": "pending",
            }
        ],
        "modeled_continuation_gap": {},
    }
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _print_prepare_context_summary(context_summary)
    rendered = buffer.getvalue()
    assert "active_rule:" in rendered, "active safety rule lines should render when rules are present"
    assert "loaded_rule:" in rendered, "loaded safety rule lines should render when rules are present"
    assert "Relevant Assembly Requirements" in rendered
    assert CASE3_LG_BEFORE_MCP_RULE_ID in rendered
    assert CASE3_BOARD_MUTEX_RULE_ID in rendered


# ---------------------------------------------------------------------------
# pytest entry
# ---------------------------------------------------------------------------


def test_case3_bridge_prepare_trace() -> None:
    result = asyncio.run(run_case3_bridge_prepare_trace(write_debug=False))
    _assert_bridge_prepare_trace(result)
    fixture, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
        )
    )
    _assert_completed_parts_drop_out_of_unsatisfied_goals(
        planner=planner,
        fixture=fixture,
        prepared_bridge_request=prepared_bridge_request,
    )
    _assert_requirement_filtering(
        planner=planner,
        fixture=fixture,
        prepared_bridge_request=prepared_bridge_request,
    )
    _assert_synthetic_safety_rules_render()


def test_case3_bridge_dryrun() -> None:
    async def _mock_ask_llm(
        self: FakeProductAgent,
        *,
        prompt: str,
        with_functions: bool = False,
        temperature: float = 0.0,
    ) -> str:
        del prompt, with_functions, temperature
        self._turn_index += 1
        self.turn_log.append(
            {
                "turn_index": self._turn_index,
                "prompt": "<mocked>",
                "response": MOCK_SINGLE_SHOT_RESPONSE,
            }
        )
        return MOCK_SINGLE_SHOT_RESPONSE

    with patch.object(FakeProductAgent, "ask_llm", new=_mock_ask_llm):
        result = asyncio.run(run_case3_bridge_dryrun(write_debug=False))
    _assert_bridge_dryrun(result)
    llm_input = result.get("llm_input") or {}
    fault_event = llm_input.get("fault_event") or {}
    assert fault_event.get("blocked_at_task_id") == FAILED_TASK_ID
    assert fault_event.get("blocked_at_function") == "place_insert"


def test_case3_bridge_debug_sidecars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.modules[__name__], "DEBUG_DIR", tmp_path)
    async def _mock_ask_llm(
        self: FakeProductAgent,
        *,
        prompt: str,
        with_functions: bool = False,
        temperature: float = 0.0,
    ) -> str:
        del prompt, with_functions, temperature
        self._turn_index += 1
        self.turn_log.append(
            {
                "turn_index": self._turn_index,
                "prompt": "<mocked>",
                "response": MOCK_SINGLE_SHOT_RESPONSE,
            }
        )
        return MOCK_SINGLE_SHOT_RESPONSE

    with patch.object(FakeProductAgent, "ask_llm", new=_mock_ask_llm):
        result = asyncio.run(run_case3_bridge_dryrun(write_debug=True))

    prompt_artifact_path = Path(str(result.get("prompt_artifact_path") or ""))
    latest_prompt_artifact_path = Path(str(result.get("latest_prompt_artifact_path") or ""))
    response_artifact_path = Path(str(result.get("response_artifact_path") or ""))
    latest_response_artifact_path = Path(str(result.get("latest_response_artifact_path") or ""))

    assert prompt_artifact_path.exists()
    assert prompt_artifact_path.name.startswith("single_shot_prompt_")
    assert prompt_artifact_path.suffix == ".txt"
    assert prompt_artifact_path.read_text(encoding="utf-8") == str(
        result.get("single_shot_prompt_text") or ""
    )
    assert latest_prompt_artifact_path.exists()
    assert latest_prompt_artifact_path.name == "single_shot_prompt_latest.txt"
    assert latest_prompt_artifact_path.read_text(encoding="utf-8") == str(
        result.get("single_shot_prompt_text") or ""
    )
    assert response_artifact_path.exists()
    assert response_artifact_path.name.startswith("single_shot_response_")
    assert response_artifact_path.suffix == ".txt"
    assert response_artifact_path.read_text(encoding="utf-8") == str(
        result.get("raw_response") or ""
    )
    assert latest_response_artifact_path.exists()
    assert latest_response_artifact_path.name == "single_shot_response_latest.txt"
    assert latest_response_artifact_path.read_text(encoding="utf-8") == str(
        result.get("raw_response") or ""
    )


def test_case3_bridge_multi_turn_override() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = prepared_bridge_request.get("bridge_session") or {}
    assert bridge_session.get("reasoning_mode") == "multi_turn"
    assert "single_shot_prompt_input" not in prepared_bridge_request
    assert "single_shot_prompt_text" not in prepared_bridge_request

    proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))
    assert proposal is None
    bridge_debug = planner.get_last_bridge_debug()
    assert bridge_debug.get("status") == "unsupported_reasoning_mode"


# ---------------------------------------------------------------------------
# CLI entry  (F5 / python test/test_case3_bridge_dryrun.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Case 3 LG-slippage bridge single-shot prompt preparation"
    )
    parser.add_argument("--model", default=DEFAULT_LIVE_MODEL, help="OpenAI model name")
    parser.add_argument("--no-debug", action="store_true", help="Skip writing debug artifact")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare and log the context trace, then exit before any LLM stage",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Compatibility flag; full single-shot prompt preparation is now the default",
    )
    parser.add_argument(
        "--show-llm-input",
        dest="show_llm_input",
        action="store_true",
        help="Print the full prepared llm_input JSON in the terminal",
    )
    parser.add_argument(
        "--hide-llm-input",
        dest="show_llm_input",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--show-single-shot-prompt",
        dest="show_single_shot_prompt",
        action="store_true",
        help="Print the rendered single-shot prompt text in the terminal",
    )
    parser.add_argument(
        "--hide-single-shot-prompt",
        dest="show_single_shot_prompt",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(show_llm_input=False, show_single_shot_prompt=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.prepare_only:
        result = asyncio.run(
            run_case3_bridge_prepare_trace(
                write_debug=not args.no_debug,
                llm_model=args.model,
            )
        )
        print()
        print("Prepare-trace context build completed.")
        _print_prepare_context_summary(result.get("context_summary") or {})
        if args.show_llm_input:
            _print_llm_input(result.get("llm_input") or {})
        if args.show_single_shot_prompt:
            _print_single_shot_prompt(result.get("single_shot_prompt_text") or "")
        _print_debug_artifact_paths(result)
        sys.exit(0)

    result = asyncio.run(
        run_case3_bridge_dryrun(
            write_debug=not args.no_debug,
            llm_model=args.model,
        )
    )

    print()
    print("Single-shot bridge status:", result.get("status") or "-")
    _print_prepare_context_summary(result.get("context_summary") or {})
    if args.show_single_shot_prompt:
        _print_single_shot_prompt(result.get("single_shot_prompt_text") or "")
    if args.show_llm_input:
        _print_llm_input(result.get("llm_input") or {})
    _print_debug_artifact_paths(result)

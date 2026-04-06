"""Dry-run scenario for Case 3 LG slippage -> LLM bridge recovery.

Run directly:
    python test/test_case3_bridge_dryrun.py
    python test/test_case3_bridge_dryrun.py --model gpt-4o
    python test/test_case3_bridge_dryrun.py --reasoning-mode hybrid
    python test/test_case3_bridge_dryrun.py --show-llm-input
    python test/test_case3_bridge_dryrun.py --show-prompt

Run as pytest:
    pytest test/test_case3_bridge_dryrun.py -v
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
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

from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
    build_hybrid_session_seed,
    multi_turn_v2 as multi_turn_v2_mode,
)
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
GOAL_STATE = "assembled"
DEFAULT_LIVE_MODEL = os.environ.get("CASE3_RECOVERY_MODEL", "gpt-4o")
DEBUG_DIR = Path("cais_spade_llm/monitor/debug")
_POST_VALIDATION_INSPECTION_TURNS = 3
CASE3_COMPLETED_TASK_IDS = (
    "REQ_1_T1",
    "REQ_1_T2",
    "REQ_2_T1",
    "REQ_2_T2",
    "REQ_2_T3",
)
MOCK_SINGLE_SHOT_RESPONSE = json.dumps(
    {
        "thought": (
            "xarm6 is failed after LG placement, and SAFE_1 keeps the protected MCP suffix blocked "
            "until LG is recovered. Using xarm6 to recover LG is the smallest proposal that restores "
            "the blocked suffix and returns the failed resource to a resumable state."
        ),
        "primary_obligation": {
            "rule_id": "SAFE_2-1",
            "resource_jid": "xarm6@localhost",
        },
        "macro_tasks": [],
    },
    indent=2,
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
            precomputed_policy.get("bridge_reasoning_mode", "hybrid") or "hybrid"
        ).strip().lower()
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
        self._shared_observations: dict[str, dict[str, Any]] = {}

    def set_shared_observations(self, observations: dict[str, dict[str, Any]] | None) -> None:
        self._shared_observations = deepcopy(observations or {})

    def _observation_catalog(self) -> dict[str, dict[str, Any]]:
        catalog = deepcopy(self._shared_observations or {})
        catalog.update(deepcopy(self._observations or {}))
        return catalog

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

    def detect_parts(self, part_name: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        """
        ---
        description: Detect parts via perception service. Optionally filter by part name.
        params:
          part_name: {type: string, description: "Filter results to this part name"}
        preconditions: {}
        effects: {}
        ---
        """
        if not part_name:
            part_name = kwargs.get("filter_part_name")
        catalog = self._observation_catalog()
        if part_name:
            observation = deepcopy(catalog.get(str(part_name).strip()) or {})
            if observation:
                return [observation]
            return []
        return [deepcopy(item) for item in catalog.values()]

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
        grounded_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del operation_kind
        bridge_snapshot = deepcopy(bridge_snapshot or {})
        part_context = deepcopy(part_context or {})
        grounded_action = deepcopy(grounded_action or {})
        evidence = {
            "part_context": deepcopy(part_context),
            "bridge_snapshot": deepcopy(bridge_snapshot),
            "resource_jid": self.jid,
            "grounded_action": deepcopy(grounded_action),
        }
        target_info = dict(grounded_action.get("target") or part_context.get("target") or {})
        expected_effect = dict(grounded_action.get("expected_effect") or {})
        preconditions = dict(grounded_action.get("preconditions") or {})
        resource_preconditions = dict(preconditions.get("resource") or {})
        part_preconditions = dict(preconditions.get("part") or {})
        source_ref = dict(preconditions.get("source_ref") or {})
        effect_scope = str(grounded_action.get("effect_scope") or "").strip().lower()
        task_kind = str(grounded_action.get("task_kind") or "").strip().lower()
        expected_resource = dict(expected_effect.get("resource") or {})
        expected_part = dict(expected_effect.get("part") or {})
        named_pose = str(target_info.get("named_pose") or part_context.get("named_pose") or "").strip()
        available_named_poses = {
            str(name).strip()
            for name in (
                bridge_snapshot.get("named_poses")
                or self.static_capabilities.get("named_poses")
                or []
            )
            if str(name).strip()
        }
        if named_pose and available_named_poses and named_pose not in available_named_poses:
            return {
                "allowed": False,
                "constraint_code": "named_pose_unavailable",
                "guard": {
                    "kind": "named_pose_unavailable",
                    "resource_jid": self.jid,
                    "named_pose": named_pose,
                },
                "reason": f"named pose '{named_pose}' is not available on this robot",
                "evidence": {**evidence, "named_pose": named_pose},
            }

        held_part = str(
            bridge_snapshot.get("held_part")
            or part_context.get("resource_held_part")
            or ""
        ).strip()
        gripper_state = str(
            bridge_snapshot.get("gripper_state")
            or part_context.get("resource_gripper_state")
            or ""
        ).strip().lower()
        current_holder = str(part_context.get("current_holder_resource_jid") or "").strip()
        desired_resource_state = str(expected_resource.get("current_state") or "").strip()
        desired_resource_location = str(expected_resource.get("location") or "").strip()
        supported_recovery_states = {
            str(token).strip()
            for token in (
                bridge_snapshot.get("supported_recovery_states")
                or self.static_capabilities.get("supported_recovery_states")
                or []
            )
            if str(token).strip()
        }
        allows_abstract_idle_recovery = (
            effect_scope == "resource_only"
            and desired_resource_state.lower() == "idle"
        )
        part_affecting = bool(
            effect_scope in {"part_only", "resource_and_part"}
            or any(
                key in expected_part and expected_part.get(key) not in (None, "", [], {})
                for key in ("state", "location", "pose", "holder")
            )
        )
        requires_part_acquisition = bool(
            part_name
            and part_affecting
            and bool(part_preconditions.get("requires_acquisition"))
        )
        if (
            effect_scope == "resource_only"
            and desired_resource_state
            and not allows_abstract_idle_recovery
            and not (
                named_pose
                or target_info.get("pose")
                or target_info.get("slot_pose")
                or desired_resource_location
            )
            and (not supported_recovery_states or desired_resource_state not in supported_recovery_states)
        ):
            return {
                "allowed": False,
                "constraint_code": "unsupported_resource_target",
                "guard": {
                    "kind": "unsupported_resource_target",
                    "resource_jid": self.jid,
                    "resource_state": desired_resource_state,
                },
                "reason": (
                    f"resource-only transition targets state '{desired_resource_state}' "
                    "without a concrete supported recovery pose or advertised recovery target"
                ),
                "evidence": {
                    **evidence,
                    "supported_recovery_states": sorted(supported_recovery_states),
                    "resource_preconditions": deepcopy(resource_preconditions),
                },
            }
        if requires_part_acquisition and part_name:
            if held_part and held_part != str(part_name).strip():
                return {
                    "allowed": False,
                    "constraint_code": "holder_conflict",
                    "guard": {
                        "kind": "resource_holds_part",
                        "resource_jid": self.jid,
                        "held_part": held_part,
                    },
                    "reason": f"resource already holds '{held_part}' and cannot acquire '{str(part_name).strip()}'",
                    "evidence": {**evidence, "conflicting_part": held_part},
                }
            if current_holder and current_holder != self.jid:
                return {
                    "allowed": False,
                    "constraint_code": "holder_conflict",
                    "guard": {
                        "kind": "part_held_by_other",
                        "part_name": str(part_name).strip(),
                        "current_holder_resource_jid": current_holder,
                    },
                    "reason": f"part '{str(part_name).strip()}' is currently held by '{current_holder}', not this robot",
                    "evidence": {**evidence, "current_holder_resource_jid": current_holder},
                }
            if not held_part and gripper_state == "closed":
                return {
                    "allowed": False,
                    "constraint_code": "gripper_occupancy_conflict",
                    "guard": {
                        "kind": "gripper_closed_without_target_part",
                        "resource_jid": self.jid,
                    },
                    "reason": "gripper is already closed without holding the target part",
                    "evidence": evidence,
                }
            if not source_ref:
                return {
                    "allowed": False,
                    "constraint_code": "source_reference_unavailable",
                    "guard": {
                        "kind": "source_reference_unavailable",
                        "resource_jid": self.jid,
                        "part_name": str(part_name).strip(),
                    },
                    "reason": (
                        f"task requires acquiring '{str(part_name).strip()}' first but no "
                        "grounded current source reference is available"
                    ),
                    "evidence": evidence,
                }
        elif part_affecting and part_name and task_kind != "continuation_resume":
            if held_part != str(part_name).strip() and current_holder != self.jid:
                return {
                    "allowed": False,
                    "constraint_code": "required_part_not_held",
                    "guard": {
                        "kind": "required_part_not_held",
                        "resource_jid": self.jid,
                        "part_name": str(part_name).strip(),
                    },
                    "reason": (
                        f"task changes part '{str(part_name).strip()}' but resource "
                        f"'{self.jid}' does not currently hold it"
                    ),
                    "evidence": evidence,
                }

        target_pose: dict[str, Any] | None = None
        if requires_part_acquisition or str(target_info.get("source_location") or "").strip() == "observed_pose":
            source_pose = dict(source_ref.get("pose") or {})
            target_pose = (
                source_pose
                or part_context.get("observed_pose")
                or target_info.get("source_pose")
                or part_context.get("pose")
            )
            if requires_part_acquisition and target_pose is None:
                return {
                    "allowed": False,
                    "constraint_code": "source_reference_unavailable",
                    "guard": {
                        "kind": "source_reference_unavailable",
                        "resource_jid": self.jid,
                        "part_name": str(part_name or "").strip() or None,
                    },
                    "reason": (
                        f"task requires acquiring '{str(part_name).strip()}' first but its "
                        "grounded source reference has no usable pose or location evidence"
                    ),
                    "evidence": {
                        **evidence,
                        "source_ref": deepcopy(source_ref),
                    },
                }
        if target_pose is None:
            target_pose = (
                target_info.get("slot_pose")
                or target_info.get("pose")
                or dict(expected_part.get("pose") or {})
                or None
            )
        if target_pose is None:
            return {
                "allowed": True,
                "reason": (
                    "grounded preconditions are satisfied and no pose-dependent "
                    "reachability check is required"
                ),
                "evidence": evidence,
            }
        inside, reason = self._is_pose_in_workspace(target_pose)
        evidence["checked_pose"] = deepcopy(target_pose)
        evidence["workspace_bounds"] = deepcopy(
            self.static_capabilities.get("workspace_bounds") or {}
        )
        return {
            "allowed": inside,
            "constraint_code": "workspace_unreachable" if not inside else None,
            "guard": (
                {
                    "kind": "observed_pose_unreachable",
                    "resource_jid": self.jid,
                    "part_name": str(part_name or "").strip() or None,
                    "pose": deepcopy(target_pose),
                }
                if not inside
                else None
            ),
            "reason": reason,
            "evidence": evidence,
        }

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
        catalog = self._observation_catalog()
        if not part_name and len(catalog) == 1:
            part_name = next(iter(catalog.keys()))
        observation = deepcopy(catalog.get(part_name) or {})
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
            "state": "misplaced",
            "location": None,
            "last_known_location": None,
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
    lg_entry = dict(part_tracker.get("LG") or {})
    if lg_entry:
        lg_entry["state"] = "misplaced"
        lg_entry["location"] = None
        lg_entry["last_known_location"] = None
        part_tracker["LG"] = lg_entry
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

def _normalized_observation_pose(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    pose: dict[str, Any] = {}
    nested_pose = payload.get("pose")
    if isinstance(nested_pose, dict):
        pose.update(deepcopy(nested_pose))
    for axis in ("x", "y", "z", "qx", "qy", "qz", "qw"):
        value = payload.get(axis)
        if value is not None:
            pose[axis] = deepcopy(value)
    if any(axis not in pose for axis in ("x", "y", "z")):
        return None
    normalized: dict[str, Any] = {}
    for axis in ("x", "y", "z"):
        try:
            normalized[axis] = float(pose[axis])
        except (TypeError, ValueError):
            return None
    for axis in ("qx", "qy", "qz", "qw"):
        value = pose.get(axis)
        if value is not None:
            normalized[axis] = deepcopy(value)
    return normalized


def _holder_resource_jid_for_part_row(part_row: dict[str, Any]) -> str:
    holder_resource_jid = str(part_row.get("current_holder_resource_jid") or "").strip()
    if holder_resource_jid:
        return holder_resource_jid
    current_location = str(part_row.get("current_location") or "").strip()
    if current_location.endswith("_gripper"):
        return current_location.rsplit("_gripper", 1)[0]
    return ""


def _build_shared_grounding_observation_catalog(
    *,
    prepared_bridge_request: dict[str, Any],
    robots: list[FakeBridgeRobot],
) -> dict[str, dict[str, Any]]:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    part_rows = [
        dict(row)
        for row in (llm_input.get("part_facts") or [])
        if isinstance(row, dict) and str(row.get("part_name") or "").strip()
    ]
    if not part_rows:
        return {}

    robots_by_jid = {
        str(robot.jid or "").strip(): robot
        for robot in robots
        if str(getattr(robot, "jid", "") or "").strip()
    }
    explicit_observations_by_part: dict[str, dict[str, Any]] = {}
    for robot in robots:
        for part_name, observation in dict(getattr(robot, "_observations", {}) or {}).items():
            token = str(part_name or "").strip()
            if not token or token in explicit_observations_by_part or not isinstance(observation, dict):
                continue
            explicit_observations_by_part[token] = deepcopy(observation)

    shared_catalog: dict[str, dict[str, Any]] = {}
    for part_row in part_rows:
        part_name = str(part_row.get("part_name") or "").strip()
        if not part_name:
            continue
        observation = deepcopy(explicit_observations_by_part.get(part_name) or {})
        normalized_pose = _normalized_observation_pose(observation)
        if normalized_pose is None:
            normalized_pose = _normalized_observation_pose(part_row.get("observed_pose"))
        holder_resource_jid = _holder_resource_jid_for_part_row(part_row)
        if normalized_pose is None and holder_resource_jid:
            holder_robot = robots_by_jid.get(holder_resource_jid)
            if holder_robot is not None:
                normalized_pose = _normalized_observation_pose(
                    dict(getattr(holder_robot, "_position", {}) or {})
                )
        if normalized_pose is None:
            continue
        observation["part_name"] = part_name
        observation["x"] = normalized_pose["x"]
        observation["y"] = normalized_pose["y"]
        observation["z"] = normalized_pose["z"]
        observation["pose"] = deepcopy(normalized_pose)
        current_location = str(part_row.get("current_location") or "").strip()
        if current_location:
            observation["current_location"] = current_location
        if holder_resource_jid:
            observation["current_holder_resource_jid"] = holder_resource_jid
        shared_catalog[part_name] = observation
    return shared_catalog


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


def _configure_live_bridge_session(
    prepared_bridge_request: dict[str, Any],
    *,
    reasoning_mode: str = "hybrid",
) -> None:
    """Tune the prepared bridge session for the direct dry-run harness."""
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    normalized_mode = str(reasoning_mode or "multi_turn").strip().lower() or "multi_turn"
    bridge_session["reasoning_mode"] = normalized_mode
    bridge_session["max_turns"] = max(int(bridge_session.get("max_turns", 6) or 6), 20)
    bridge_session["repair_mode"] = "recover"
    bridge_session["observation_backend"] = "mock_detect_parts_harness"
    if normalized_mode == "multi_turn":
        bridge_session["multi_turn_engine"] = "v2"
        bridge_session["outline_mode"] = "incremental_candidates_validated"
    else:
        bridge_session.pop("multi_turn_engine", None)
        bridge_session.pop("outline_mode", None)
    prepared_bridge_request["bridge_session"] = bridge_session


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


async def _prepare_bridge_dryrun_harness(
    *,
    llm_model: str | None = None,
    reasoning_mode: str = "multi_turn",
) -> tuple[dict[str, Any], FakeProductAgent, ProcessPlanner, dict[str, Any]]:
    """Load configs, build fake agents, prepare the bridge request."""
    paths = _case3_paths()
    tools_catalog = _load_json(paths["tools"])
    plan_payload = _load_json(paths["plan"])
    geometry_payload = _load_json(paths["geometry"])
    bundle_context = _case3_bundle_context(paths)
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
    product_agent._bridge_reasoning_mode = (
        str(reasoning_mode or "hybrid").strip().lower() or "hybrid"
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
            "MCP": {"part_name": "MCP", "x": 0.0, "y": -0.08, "z": 1.025, "pose": {"x": 0.0, "y": -0.08, "z": 1.025}},
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
        observations={
            "LG": {"part_name": "LG", "x": 0.0, "y": 0.2, "z": 1.035, "pose": {"x": 0.0, "y": 0.2, "z": 1.035}},
        },
    )

    planner = ProcessPlannerPrepareTrace(product_agent, [ur5e, xarm6])
    planner.nodes = deepcopy(plan_payload.get("nodes") or [])
    _apply_runtime_status_snapshot(planner.nodes)

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

    shared_grounding_observations = _build_shared_grounding_observation_catalog(
        prepared_bridge_request=prepared_bridge_request,
        robots=[ur5e, xarm6],
    )
    ur5e.set_shared_observations(shared_grounding_observations)
    xarm6.set_shared_observations(shared_grounding_observations)

    product_agent.prepared_bridge_request = prepared_bridge_request
    _relax_recovery_clear_precondition(prepared_bridge_request)
    _configure_live_bridge_session(
        prepared_bridge_request,
        reasoning_mode=product_agent._bridge_reasoning_mode,
    )
    if product_agent._bridge_reasoning_mode == "hybrid":
        prepared_bridge_request["hybrid_session_seed"] = build_hybrid_session_seed(
            prepared_bridge_request
        )
        prepared_bridge_request.pop("multi_turn_session_seed", None)
    else:
        prepared_bridge_request["multi_turn_session_seed"] = (
            multi_turn_v2_mode.build_multi_turn_session_seed(prepared_bridge_request)
        )
        prepared_bridge_request.pop("hybrid_session_seed", None)

    return fixture, product_agent, planner, prepared_bridge_request


# ---------------------------------------------------------------------------
# Main coroutine
# ---------------------------------------------------------------------------


async def run_case3_bridge_dryrun(
    write_debug: bool = True,
    *,
    llm_model: str | None = None,
    reasoning_mode: str = "hybrid",
    stop_before_primitive_generation: bool = True,
) -> dict[str, Any]:
    """Run the Case 3 dry-run scenario through the bridge once."""
    _, product_agent, planner, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
        llm_model=llm_model,
        reasoning_mode=reasoning_mode,
    )

    if write_debug:
        debug_dir = Path(DEBUG_DIR)
        if not debug_dir.is_absolute():
            debug_dir = _repo_root() / debug_dir
        bridge_debug_seed = dict(prepared_bridge_request.get("bridge_debug") or {})
        bridge_debug_seed["per_turn_debug_dir"] = str(debug_dir)
        prepared_bridge_request["bridge_debug"] = bridge_debug_seed

    # First run: grounding + first outline task
    proposal = await planner.execute_prepared_bridge_request(prepared_bridge_request)

    if str(reasoning_mode or "multi_turn").strip().lower() == "multi_turn":
        # Resume loop: keep running while paused after outline turns
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
            execute_multi_turn_bridge as _resume_bridge,
        )
        max_resume = max(
            10,
            int(
                dict(prepared_bridge_request.get("bridge_session") or {}).get("max_turns")
                or dict(prepared_bridge_request.get("multi_turn_session_seed") or {}).get("max_turns")
                or 0
            ),
        )
        post_validation_resume_budget = 0
        for _resume_i in range(max_resume):
            ss = prepared_bridge_request.get("multi_turn_session_state") or {}
            if ss.get("status") != "paused_after_outline_turn":
                break
            if stop_before_primitive_generation and str(ss.get("current_phase") or "").strip().lower() == "primitive_generation":
                logging.getLogger("case3_bridge_dryrun").info(
                    "[DryRun] Outline completed; stopping before primitive_generation for inspection"
                )
                break
            if stop_before_primitive_generation:
                logging.getLogger("case3_bridge_dryrun").info(
                    "[DryRun] Resuming outline loop until primitive_generation (round %d)", _resume_i + 1,
                )
                proposal = await _resume_bridge(
                    planner, prepared_bridge_request, session_state=ss,
                )
                continue
            findings = list(ss.get("outline_validation_findings") or [])
            if not findings and str(ss.get("outline_mode") or "").strip().lower() == "incremental_candidates_validated":
                findings = [
                    deepcopy(row)
                    for row in (ss.get("candidate_rejection_feedback") or [])
                    if isinstance(row, dict)
                ]
            if findings and post_validation_resume_budget <= 0:
                logging.getLogger("case3_bridge_dryrun").info(
                    "[DryRun] Validation rejection detected (%d findings) — resuming %d more turns",
                    len(findings),
                    _POST_VALIDATION_INSPECTION_TURNS,
                )
                post_validation_resume_budget = _POST_VALIDATION_INSPECTION_TURNS
            elif post_validation_resume_budget > 0:
                logging.getLogger("case3_bridge_dryrun").info(
                    "[DryRun] Resuming post-validation inspection turn (%d remaining after this resume)",
                    post_validation_resume_budget - 1,
                )
            else:
                logging.getLogger("case3_bridge_dryrun").info(
                    "[DryRun] Resuming outline loop (round %d)", _resume_i + 1,
                )
            proposal = await _resume_bridge(
                planner, prepared_bridge_request, session_state=ss,
            )
            if post_validation_resume_budget > 0:
                post_validation_resume_budget -= 1
                next_ss = prepared_bridge_request.get("multi_turn_session_state") or {}
                if (
                    post_validation_resume_budget == 0
                    and next_ss.get("status") == "paused_after_outline_turn"
                ):
                    logging.getLogger("case3_bridge_dryrun").info(
                        "[DryRun] Paused after final post-validation inspection turn — inspect debug artifacts"
                    )
                    break

    elif str(reasoning_mode or "hybrid").strip().lower() == "hybrid":
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
            execute_hybrid_des_bridge as _resume_hybrid,
        )
        for _resume_i in range(5):
            ss = prepared_bridge_request.get("hybrid_session_state") or prepared_bridge_request.get("multi_turn_session_result") or {}
            if ss.get("status") != "paused_after_grounding":
                break
                
            logging.getLogger("case3_bridge_dryrun").info(
                "[DryRun] Mocking actual sensor observation for hybrid grounding block..."
            )
            # Mock the camera response by directly mutating the symbolic_parts in the state
            parts = ss.get("symbolic_parts") or {}
            if "LG" in parts:
                parts["LG"]["observed_pose"] = {"x": 0.0, "y": 0.200, "z": 1.035, "q_x": 0, "q_y": 0, "q_z": 0, "q_w": 1}
                # Optional: specify location if testing boundary conditions
                parts["LG"]["current_location"] = "prusa-mk4-1"
                
            ss["status"] = "pending"
            prepared_bridge_request["hybrid_session_state"] = ss
            if "multi_turn_session_result" in prepared_bridge_request:
                del prepared_bridge_request["multi_turn_session_result"]
            
            logging.getLogger("case3_bridge_dryrun").info(
                "[DryRun] Resuming hybrid bridge (round %d) after grounding...", _resume_i + 1
            )
            proposal = await _resume_hybrid(
                planner, prepared_bridge_request, session_state=ss,
            )

    bridge_debug = planner.get_last_bridge_debug()
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    reasoning_mode = str(bridge_session.get("reasoning_mode") or "hybrid").strip()
    multi_turn_session = dict((bridge_debug or {}).get("multi_turn_session") or {})
    turns = list(multi_turn_session.get("turns") or [])

    result: dict[str, Any] = {
        "scenario": "case3_lg_slippage",
        "reasoning_mode": reasoning_mode,
        "status": str(bridge_debug.get("status") or ""),
        "proposal": proposal,
        "bridge_debug": bridge_debug,
        "prepared_bridge_request": prepared_bridge_request,
        "context_summary": deepcopy(prepared_bridge_request.get("context_summary") or {}),
        "llm_input": deepcopy(prepared_bridge_request.get("llm_input") or {}),
        "multi_turn_session": deepcopy(multi_turn_session),
        "turns": deepcopy(turns),
        "turn_log": deepcopy(product_agent.turn_log),
        "prompt_artifact_path": None,
        "latest_prompt_artifact_path": None,
        "response_artifact_path": None,
        "latest_response_artifact_path": None,
        "session_transcript_artifact_path": None,
        "latest_session_transcript_artifact_path": None,
    }

    if write_debug:
        artifact_paths = _write_debug_artifacts(result)
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
        write_latest=False,
        write_session_transcript=True,
        filename_prefix=filename_prefix,
    )


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def _assert_bridge_dryrun(
    result: dict[str, Any],
    *,
    expected_reasoning_mode: str = "hybrid",
) -> None:
    bridge_debug = result.get("bridge_debug") or {}
    status = str(bridge_debug.get("status") or "")
    reasoning_mode = str(result.get("reasoning_mode") or "")
    assert reasoning_mode == expected_reasoning_mode, (
        f"Expected {expected_reasoning_mode} reasoning mode; got {reasoning_mode!r}"
    )
    multi_turn_session = result.get("multi_turn_session") or {}
    session_status = str(multi_turn_session.get("status") or "")
    assert session_status in ("completed", "turn_budget_exhausted", "error"), (
        f"Unexpected multi-turn session status: {session_status!r}"
    )
    turns = result.get("turns") or []
    assert len(turns) >= 1, "Expected at least one turn in multi-turn session"


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
    relevant_assembly_requirements = list(
        context_summary.get("relevant_assembly_requirements") or []
    )
    modeled_continuation_gap = dict(context_summary.get("modeled_continuation_gap") or {})
    resources = list(current_product_state.get("resources") or [])
    parts = list(current_product_state.get("parts") or [])

    print()
    print("Fault Event")
    print(f"  focused_resource_jid: {fault_event.get('focused_resource_jid') or '-'}")
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
            "  {jid}: state={state}, held_part={held}, location={location}".format(
                jid=row.get("resource_jid") or "-",
                state=row.get("current_state") or "-",
                held=row.get("held_part") or "-",
                location=row.get("current_location") or "-",
            )
        )

    print()
    print("Parts")
    for row in parts:
        if not isinstance(row, dict):
            continue
        print(
            "  {part}: state={state}, location={location}, observed_pose={pose}".format(
                part=row.get("part_name") or "-",
                state=row.get("state") or "-",
                location=row.get("location") or "-",
                pose=_format_pose_brief(row.get("observed_pose")),
            )
        )

    print()
    print("Relevant Assembly Requirements")
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
    print(
        "  pending_nominal_task_ids: "
        f"{modeled_continuation_gap.get('pending_nominal_task_ids') or []}"
    )
    print(f"  resume_ready:             {modeled_continuation_gap.get('resume_ready')}")


def _print_llm_input(llm_input: dict[str, Any]) -> None:
    print()
    print("LLM Input")
    print(json.dumps(llm_input, indent=2, default=str))


def _print_prompt(prompt_text: str) -> None:
    print()
    print("Prompt")
    print(prompt_text or "")


def _print_debug_artifact_paths(result: dict[str, Any]) -> None:
    prompt_artifact_path = result.get("prompt_artifact_path")
    latest_prompt_artifact_path = result.get("latest_prompt_artifact_path")
    response_artifact_path = result.get("response_artifact_path")
    latest_response_artifact_path = result.get("latest_response_artifact_path")
    session_transcript_artifact_path = result.get("session_transcript_artifact_path")
    latest_session_transcript_artifact_path = result.get("latest_session_transcript_artifact_path")
    if not any(
        (
            prompt_artifact_path,
            latest_prompt_artifact_path,
            response_artifact_path,
            latest_response_artifact_path,
            session_transcript_artifact_path,
            latest_session_transcript_artifact_path,
        )
    ):
        return
    print()
    if prompt_artifact_path:
        print("Prompt artifact:          ", prompt_artifact_path)
    if latest_prompt_artifact_path:
        print("Latest prompt artifact:   ", latest_prompt_artifact_path)
    if response_artifact_path:
        print("Response artifact:        ", response_artifact_path)
    if latest_response_artifact_path:
        print("Latest response artifact: ", latest_response_artifact_path)
    if session_transcript_artifact_path:
        print("Session artifact:         ", session_transcript_artifact_path)
    if latest_session_transcript_artifact_path:
        print("Latest session artifact:  ", latest_session_transcript_artifact_path)


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


def test_v2_grounding_repeated_observe_request_becomes_grounded() -> None:
    async def _run() -> None:
        _, _, planner, prepared_bridge_request = await _prepare_bridge_dryrun_harness()
        session_state = multi_turn_v2_mode.build_multi_turn_session_seed(
            prepared_bridge_request
        )
        fact_key = json.dumps(
            {"entity": "LG", "fact_type": "part_pose"},
            sort_keys=True,
            ensure_ascii=True,
        )
        session_state["observation_store"] = {
            "observed_pose_LG": {
                "part_name": "LG",
                "x": 0.0,
                "y": 0.2,
                "z": 1.035,
                "pose": {"x": 0.0, "y": 0.2, "z": 1.035},
            }
        }
        session_state["observation_fact_ledger"] = {
            fact_key: {
                "fact_key": fact_key,
                "fact_type": "part_pose",
                "entity": "LG",
                "entity_kind": "part",
                "primitive": "detect_parts",
                "params": {"part_name": "LG"},
                "output": deepcopy(session_state["observation_store"]["observed_pose_LG"]),
                "turn_index": 1,
                "validity": "current",
                "freshness": "current_session",
                "aliases": ["observed_pose_LG"],
            }
        }

        async def _fail_if_called(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise AssertionError("_execute_observe_requests should not be called")

        with patch.object(
            multi_turn_v2_mode,
            "_execute_observe_requests",
            new=_fail_if_called,
        ):
            decision, turn_entry = await multi_turn_v2_mode._handle_grounding_phase(
                session_state=session_state,
                parsed_response={
                    "decision": "observe",
                    "observe_requests": [
                        {"fact_type": "part_pose", "entity": "LG", "reason": "repeat"}
                    ],
                },
                prepared_bridge_request=prepared_bridge_request,
                planner=planner,
            )

        assert decision == "grounded"
        assert turn_entry.get("grounding_override_reason") == (
            "all_observe_requests_resolved: already_fulfilled"
        )
        assert turn_entry.get("already_fulfilled_observe_requests")
        assert session_state.get("observation_count") == 0

    asyncio.run(_run())


def test_v2_grounding_unresolved_observe_request_still_executes() -> None:
    async def _run() -> None:
        _, _, planner, prepared_bridge_request = await _prepare_bridge_dryrun_harness()
        session_state = multi_turn_v2_mode.build_multi_turn_session_seed(
            prepared_bridge_request
        )
        captured_requests: list[dict[str, Any]] = []
        fact_key = json.dumps(
            {"entity": "LG", "fact_type": "part_pose"},
            sort_keys=True,
            ensure_ascii=True,
        )
        fake_result = {
            "fact_key": fact_key,
            "fact_type": "part_pose",
            "entity": "LG",
            "entity_kind": "part",
            "scope": None,
            "reason": "needed",
            "primitive": "detect_parts",
            "params": {"part_name": "LG"},
            "store_as": "observed_pose_LG",
            "output": {
                "part_name": "LG",
                "x": 0.0,
                "y": 0.2,
                "z": 1.035,
                "pose": {"x": 0.0, "y": 0.2, "z": 1.035},
            },
        }

        async def _fake_execute_observe_requests(
            planner: Any,
            prepared_bridge_request: dict[str, Any],
            session_state: dict[str, Any],
            observe_requests: list[dict[str, Any]],
        ) -> tuple[list[dict[str, Any]], str | None]:
            del planner, prepared_bridge_request, session_state
            captured_requests.extend(deepcopy(observe_requests))
            return [deepcopy(fake_result)], None

        with patch.object(
            multi_turn_v2_mode,
            "_execute_observe_requests",
            new=_fake_execute_observe_requests,
        ):
            decision, turn_entry = await multi_turn_v2_mode._handle_grounding_phase(
                session_state=session_state,
                parsed_response={
                    "decision": "observe",
                    "observe_requests": [
                        {"fact_type": "part_pose", "entity": "LG", "reason": "needed"}
                    ],
                },
                prepared_bridge_request=prepared_bridge_request,
                planner=planner,
            )

        assert decision == "observe"
        assert captured_requests == [
            {"fact_type": "part_pose", "entity": "LG", "reason": "needed"}
        ]
        assert turn_entry.get("observation_results") == [fake_result]
        assert session_state.get("observation_count") == 1
        assert session_state["observation_fact_ledger"][fact_key]["validity"] == "current"

    asyncio.run(_run())


def test_v2_outline_validation_findings_persist_until_resolved() -> None:
    async def _run() -> None:
        _, _, planner, prepared_bridge_request = await _prepare_bridge_dryrun_harness()
        session_state = multi_turn_v2_mode.build_multi_turn_session_seed(
            prepared_bridge_request
        )
        workspace_finding = {
            "constraint_owner": "resource",
            "constraint_code": "workspace_unreachable",
            "resource_jid": "xarm6@localhost",
            "part_name": "LG",
            "reason": (
                "Part 'LG' observed pose {'x': 0.0, 'y': 0.2, 'z': 1.035} is outside "
                "'xarm6@localhost' workspace bounds"
            ),
            "evidence": {
                "pose": {"x": 0.0, "y": 0.2, "z": 1.035},
                "bounds": {
                    "x_min_m": -0.6,
                    "x_max_m": 0.6,
                    "y_min_m": -1.0,
                    "y_max_m": 0.1,
                    "z_min_m": 0.9,
                    "z_max_m": 1.5,
                },
            },
        }
        session_state["outline_validation_findings"] = [deepcopy(workspace_finding)]
        session_state["observation_store"] = {
            "observed_pose_LG": {
                "part_name": "LG",
                "x": 0.0,
                "y": 0.2,
                "z": 1.035,
                "pose": {"x": 0.0, "y": 0.2, "z": 1.035},
            }
        }

        decision, _turn_entry = await multi_turn_v2_mode._handle_outline_incremental_validated(
            session_state=session_state,
            parsed_response={
                "next_task": {
                    "outline_id": "FIX_XARM6",
                    "resource_jid": "xarm6@localhost",
                    "description": "Diagnose and recover xarm6 from failed to idle.",
                    "expected_start_state": {"resource_state": "failed"},
                    "expected_end_state": {"resource_state": "idle"},
                },
                "lookahead_tasks": [],
            },
            prepared_bridge_request=prepared_bridge_request,
            planner=planner,
        )

        assert decision == "outline_ready"
        assert session_state.get("outline_validation_findings") == [workspace_finding]

    asyncio.run(_run())


def test_v2_outline_validation_findings_clear_once_resolved() -> None:
    async def _run() -> None:
        _, _, planner, prepared_bridge_request = await _prepare_bridge_dryrun_harness()
        session_state = multi_turn_v2_mode.build_multi_turn_session_seed(
            prepared_bridge_request
        )
        session_state["symbolic_resources"]["xarm6@localhost"]["current_state"] = "idle"
        finding = {
            "constraint_owner": "resource",
            "constraint_code": "required_part_not_held",
            "resource_jid": "xarm6@localhost",
            "part_name": "LG",
            "reason": "Task expects 'xarm6@localhost' to hold 'LG' but it currently holds nothing",
            "evidence": {"expected": "LG", "actual": ""},
        }
        session_state["outline_validation_findings"] = [deepcopy(finding)]
        session_state["observation_store"] = {
            "observed_pose_LG": {
                "part_name": "LG",
                "x": 0.0,
                "y": -0.1,
                "z": 1.0,
                "pose": {"x": 0.0, "y": -0.1, "z": 1.0},
            }
        }

        decision, _turn_entry = await multi_turn_v2_mode._handle_outline_incremental_validated(
            session_state=session_state,
            parsed_response={
                "next_task": {
                    "outline_id": "PICK_LG",
                    "resource_jid": "xarm6@localhost",
                    "description": "Acquire LG so later part-changing actions are feasible.",
                    "part_name": "LG",
                    "expected_start_state": {"resource_state": "idle"},
                    "expected_end_state": {
                        "resource_state": "idle",
                        "held_part": "LG",
                        "part_state": "in_gripper",
                        "part_location": "xarm6@localhost_gripper",
                        "part_holder_resource_jid": "xarm6@localhost",
                    },
                },
                "lookahead_tasks": [],
            },
            prepared_bridge_request=prepared_bridge_request,
            planner=planner,
        )

        assert decision == "outline_ready"
        assert session_state.get("outline_validation_findings") == []

    asyncio.run(_run())


def test_v2_dryrun_allows_turn_7_after_first_validation_rejection() -> None:
    async def _run() -> None:
        responses = [
            {
                "thought": "Need LG pose.",
                "decision": "observe",
                "blocking_reasons": ["LG pose unknown"],
                "grounded_facts": [],
                "recovery_implications": [],
                "observe_reason": "Need LG pose",
                "observe_requests": [
                    {"fact_type": "part_pose", "entity": "LG", "reason": "Need LG pose"}
                ],
            },
            {
                "thought": "Now grounded.",
                "decision": "grounded",
                "blocking_reasons": [],
                "grounded_facts": ["LG pose observed"],
                "recovery_implications": [],
                "observe_reason": "",
                "observe_requests": [],
            },
            {
                "thought": "Accept reset.",
                "candidate_tasks": [
                    {
                        "outline_id": "BAD_3A",
                        "resource_jid": "xarm6@localhost",
                        "action_name": "idle prep",
                        "description": "No-op prep step.",
                        "expected_start_state": {"resource_state": "failed"},
                        "expected_end_state": {"resource_state": "failed"},
                    },
                    {
                        "outline_id": "FIX_XARM6",
                        "resource_jid": "xarm6@localhost",
                        "action_name": "reset resource",
                        "description": "Reset xarm6 to idle.",
                        "expected_start_state": {"resource_state": "failed"},
                        "expected_end_state": {"resource_state": "idle"},
                    },
                    {
                        "outline_id": "BAD_3C",
                        "resource_jid": "xarm6@localhost",
                        "action_name": "relocate part",
                        "description": "Relocate LG without carrying it.",
                        "part_name": "LG",
                        "target_ref": "assembly_board-v1",
                        "expected_start_state": {"resource_state": "failed"},
                        "expected_end_state": {
                            "resource_state": "failed",
                            "part_location": "assembly_board-v1",
                        },
                    },
                ],
            },
            {
                "thought": "Rejected turn 4.",
                "candidate_tasks": [
                    {
                        "outline_id": "BAD_4A",
                        "resource_jid": "xarm6@localhost",
                        "action_name": "pick up",
                        "description": "Pick unreachable LG.",
                        "part_name": "LG",
                        "expected_start_state": {"resource_state": "idle"},
                        "expected_end_state": {
                            "resource_state": "holding",
                            "held_part": "LG",
                            "part_state": "picked",
                            "part_location": "xarm6@localhost_gripper",
                            "part_holder_resource_jid": "xarm6@localhost",
                        },
                    },
                    {
                        "outline_id": "BAD_4B",
                        "resource_jid": "xarm6@localhost",
                        "action_name": "idle prep",
                        "description": "No-op prep step.",
                        "expected_start_state": {"resource_state": "idle"},
                        "expected_end_state": {"resource_state": "idle"},
                    },
                    {
                        "outline_id": "BAD_4C",
                        "resource_jid": "xarm6@localhost",
                        "action_name": "place part",
                        "description": "Relocate LG without carrying it.",
                        "part_name": "LG",
                        "target_ref": "assembly_board-v1",
                        "expected_start_state": {"resource_state": "idle"},
                        "expected_end_state": {
                            "resource_state": "idle",
                            "part_state": "placed",
                            "part_location": "assembly_board-v1",
                        },
                    },
                ],
            },
            {
                "thought": "Rejected turn 5.",
                "candidate_tasks": [
                    {
                        "outline_id": "BAD_5A",
                        "resource_jid": "xarm6@localhost",
                        "action_name": "idle prep",
                        "description": "No-op prep step.",
                        "expected_start_state": {"resource_state": "idle"},
                        "expected_end_state": {"resource_state": "idle"},
                    },
                    {
                        "outline_id": "BAD_5B",
                        "resource_jid": "xarm6@localhost",
                        "action_name": "pick up",
                        "description": "Pick unreachable LG again.",
                        "part_name": "LG",
                        "expected_start_state": {"resource_state": "idle"},
                        "expected_end_state": {
                            "resource_state": "holding",
                            "held_part": "LG",
                            "part_state": "picked",
                            "part_location": "xarm6@localhost_gripper",
                            "part_holder_resource_jid": "xarm6@localhost",
                        },
                    },
                    {
                        "outline_id": "BAD_5C",
                        "resource_jid": "xarm6@localhost",
                        "action_name": "place part",
                        "description": "Relocate LG without carrying it.",
                        "part_name": "LG",
                        "target_ref": "assembly_board-v1",
                        "expected_start_state": {"resource_state": "idle"},
                        "expected_end_state": {
                            "resource_state": "idle",
                            "part_state": "placed",
                            "part_location": "assembly_board-v1",
                        },
                    },
                ],
            },
            {
                "thought": "Rejected turn 6.",
                "candidate_tasks": [
                    {
                        "outline_id": "BAD_6A",
                        "resource_jid": "xarm6@localhost",
                        "action_name": "place part",
                        "description": "Relocate LG without carrying it.",
                        "part_name": "LG",
                        "target_ref": "assembly_board-v1",
                        "expected_start_state": {"resource_state": "idle"},
                        "expected_end_state": {
                            "resource_state": "idle",
                            "part_state": "placed",
                            "part_location": "assembly_board-v1",
                        },
                    },
                    {
                        "outline_id": "BAD_6B",
                        "resource_jid": "xarm6@localhost",
                        "action_name": "idle prep",
                        "description": "No-op prep step.",
                        "expected_start_state": {"resource_state": "idle"},
                        "expected_end_state": {"resource_state": "idle"},
                    },
                    {
                        "outline_id": "BAD_6C",
                        "resource_jid": "xarm6@localhost",
                        "action_name": "pick up",
                        "description": "Pick unreachable LG.",
                        "part_name": "LG",
                        "expected_start_state": {"resource_state": "idle"},
                        "expected_end_state": {
                            "resource_state": "holding",
                            "held_part": "LG",
                            "part_state": "picked",
                            "part_location": "xarm6@localhost_gripper",
                            "part_holder_resource_jid": "xarm6@localhost",
                        },
                    },
                ],
            },
            {
                "thought": "Turn 7 arrives before stop.",
                "candidate_tasks": [
                    {
                        "outline_id": "BAD_7A",
                        "resource_jid": "xarm6@localhost",
                        "action_name": "pick up",
                        "description": "Pick unreachable LG again.",
                        "part_name": "LG",
                        "expected_start_state": {"resource_state": "idle"},
                        "expected_end_state": {
                            "resource_state": "holding",
                            "held_part": "LG",
                            "part_state": "picked",
                            "part_location": "xarm6@localhost_gripper",
                            "part_holder_resource_jid": "xarm6@localhost",
                        },
                    },
                    {
                        "outline_id": "BAD_7B",
                        "resource_jid": "ur5e@localhost",
                        "action_name": "hold position",
                        "description": "Hold position while xarm6 recovery remains unresolved.",
                        "expected_start_state": {"resource_state": "picked"},
                        "expected_end_state": {"resource_state": "picked"},
                    },
                    {
                        "outline_id": "BAD_7C",
                        "resource_jid": "xarm6@localhost",
                        "action_name": "place part",
                        "description": "Relocate LG without carrying it.",
                        "part_name": "LG",
                        "target_ref": "assembly_board-v1",
                        "expected_start_state": {"resource_state": "idle"},
                        "expected_end_state": {
                            "resource_state": "idle",
                            "part_state": "placed",
                            "part_location": "assembly_board-v1",
                        },
                    },
                ],
            },
        ]

        async def _fake_ask_llm_structured(
            self: FakeProductAgent,
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
            max_tool_rounds: int = 3,
        ) -> dict[str, Any]:
            del self, prompt, response_format, tools, tool_executor, max_tool_rounds
            if not responses:
                raise RuntimeError("ran out of mocked responses")
            return responses.pop(0)

        with patch.object(
            FakeProductAgent,
            "ask_llm_structured",
            new=_fake_ask_llm_structured,
        ):
            result = await run_case3_bridge_dryrun(
                write_debug=False,
                llm_model="mock",
                stop_before_primitive_generation=False,
            )

        session = result.get("multi_turn_session") or {}
        assert session.get("turn_index") == 7
        assert str(session.get("status") or "") == "paused_after_outline_turn"

    asyncio.run(_run())


def test_case3_dryrun_resume_loop_uses_session_max_turns() -> None:
    async def _run() -> None:
        prepared_bridge_request = {
            "bridge_session": {"max_turns": 20},
            "multi_turn_session_seed": {"max_turns": 20},
            "multi_turn_session_state": {
                "status": "paused_after_outline_turn",
                "current_phase": "outline",
                "turn_index": 4,
                "max_turns": 20,
            },
            "context_summary": {},
            "llm_input": {},
        }

        class _FakePlanner:
            async def execute_prepared_bridge_request(self, _prepared: dict[str, Any]) -> dict[str, Any]:
                return {"ok": True}

            def get_last_bridge_debug(self) -> dict[str, Any]:
                return {
                    "status": str(prepared_bridge_request["multi_turn_session_state"].get("status") or ""),
                    "multi_turn_session": deepcopy(prepared_bridge_request["multi_turn_session_state"]),
                }

        class _FakeProductAgent:
            turn_log: list[dict[str, Any]] = []

        async def _fake_prepare_bridge_dryrun_harness(
            llm_model: str | None = None,
            reasoning_mode: str = "multi_turn",
        ) -> tuple[None, _FakeProductAgent, _FakePlanner, dict[str, Any]]:
            del llm_model, reasoning_mode
            return None, _FakeProductAgent(), _FakePlanner(), prepared_bridge_request

        async def _fake_resume_bridge(
            planner: Any,
            prepared_request: dict[str, Any],
            *,
            session_state: dict[str, Any],
        ) -> dict[str, Any]:
            del planner, session_state
            state = dict(prepared_request.get("multi_turn_session_state") or {})
            state["turn_index"] = int(state.get("turn_index") or 0) + 1
            state["status"] = (
                "completed"
                if int(state.get("turn_index") or 0) >= int(state.get("max_turns") or 0)
                else "paused_after_outline_turn"
            )
            prepared_request["multi_turn_session_state"] = state
            return {"turn_index": state["turn_index"]}

        with patch.object(
            sys.modules[__name__],
            "_prepare_bridge_dryrun_harness",
            side_effect=_fake_prepare_bridge_dryrun_harness,
        ), patch(
            "cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.execute_multi_turn_bridge",
            side_effect=_fake_resume_bridge,
        ):
            result = await run_case3_bridge_dryrun(
                write_debug=False,
                stop_before_primitive_generation=False,
            )

        session_state = dict(result.get("prepared_bridge_request", {}).get("multi_turn_session_state") or {})
        assert int(session_state.get("turn_index") or 0) == 20
        assert str(session_state.get("status") or "") == "completed"

    asyncio.run(_run())


def test_case3_dryrun_hybrid_mode_skips_multi_turn_resume_loop() -> None:
    async def _run() -> None:
        prepared_bridge_request = {
            "bridge_session": {"reasoning_mode": "hybrid", "max_turns": 20},
            "hybrid_session_seed": {"max_turns": 20},
            "hybrid_session_state": {
                "status": "completed",
                "current_phase": "finalize",
                "turn_index": 3,
                "max_turns": 20,
            },
            "context_summary": {},
            "llm_input": {},
        }
        proposal = {"engine": "hybrid_des_v1", "outline_tasks": []}

        class _FakePlanner:
            async def execute_prepared_bridge_request(
                self,
                _prepared: dict[str, Any],
            ) -> dict[str, Any]:
                return deepcopy(proposal)

            def get_last_bridge_debug(self) -> dict[str, Any]:
                return {
                    "status": "completed",
                    "multi_turn_session": deepcopy(
                        prepared_bridge_request["hybrid_session_state"]
                    ),
                }

        class _FakeProductAgent:
            turn_log: list[dict[str, Any]] = []

        async def _fake_prepare_bridge_dryrun_harness(
            llm_model: str | None = None,
            reasoning_mode: str = "multi_turn",
        ) -> tuple[None, _FakeProductAgent, _FakePlanner, dict[str, Any]]:
            del llm_model
            assert reasoning_mode == "hybrid"
            return None, _FakeProductAgent(), _FakePlanner(), prepared_bridge_request

        with patch.object(
            sys.modules[__name__],
            "_prepare_bridge_dryrun_harness",
            side_effect=_fake_prepare_bridge_dryrun_harness,
        ), patch(
            "cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.execute_multi_turn_bridge",
            side_effect=AssertionError("multi-turn resume should not run in hybrid mode"),
        ):
            result = await run_case3_bridge_dryrun(
                write_debug=False,
                reasoning_mode="hybrid",
                stop_before_primitive_generation=False,
            )

        assert result.get("proposal") == proposal
        assert result.get("reasoning_mode") == "hybrid"
        assert dict(result.get("prepared_bridge_request") or {}).get("hybrid_session_seed")
        assert not dict(result.get("prepared_bridge_request") or {}).get("multi_turn_session_seed")

    asyncio.run(_run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Case 3 LG-slippage bridge dry-run harness"
    )
    parser.add_argument("--model", default=DEFAULT_LIVE_MODEL, help="OpenAI model name")
    parser.add_argument(
        "--reasoning-mode",
        default="hybrid",
        choices=("multi_turn", "hybrid"),
        help="Bridge reasoning mode to run in the dry-run harness",
    )
    parser.add_argument("--no-debug", action="store_true", help="Skip writing debug artifacts")
    parser.add_argument(
        "--show-llm-input",
        action="store_true",
        help="Print the prepared llm_input JSON",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the rendered single-shot prompt",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    result = asyncio.run(
        run_case3_bridge_dryrun(
            write_debug=not args.no_debug,
            llm_model=args.model,
            reasoning_mode=args.reasoning_mode,
        )
    )

    print()
    print(f"Bridge status: {result.get('status') or '-'}")
    _print_prepare_context_summary(result.get("context_summary") or {})
    if args.show_prompt:
        turns = result.get("turns") or []
        if turns:
            _print_prompt(str(turns[0].get("prompt_text") or ""))
    if args.show_llm_input:
        _print_llm_input(result.get("llm_input") or {})
    _print_debug_artifact_paths(result)

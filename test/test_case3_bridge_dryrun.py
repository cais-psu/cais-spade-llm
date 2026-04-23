"""Dry-run scenario for Case 3 LG slippage -> LLM bridge recovery.

Run directly:
    python test/test_case3_bridge_dryrun.py
    python test/test_case3_bridge_dryrun.py --model gpt-5
    python test/test_case3_bridge_dryrun.py --reasoning-mode multi_turn
    python test/test_case3_bridge_dryrun.py --focus primitive_generation
    python test/test_case3_bridge_dryrun.py --show-llm-input
    python test/test_case3_bridge_dryrun.py --show-prompt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
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


def _env_default(*env_names: str, fallback: str) -> str:
    for env_name in env_names:
        token = str(os.environ.get(env_name) or "").strip()
        if token:
            return token
    return str(fallback or "").strip()


def _normalize_reasoning_effort_for_model(model_name: str, effort: str) -> str:
    normalized_model = str(model_name or "").strip().lower()
    normalized_effort = str(effort or "").strip().lower()
    if normalized_model.startswith("gpt-5.4") and normalized_effort == "minimal":
        return "none"
    return normalized_effort

from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.agents.intelligent_product.product_agent import (
    ProductAgent,
    _ack_status_is_regression,
    _should_persist_ack_state,
)
from cais_spade_llm.agents.intelligent_product.product_recovery_controller import (
    ProductRecoveryController,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_primitives import (
    snapshot_matches_expected,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
    multi_turn as multi_turn_mode,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn_primitive_generation import (
    _resolve_context_ref,
    generate_primitive_batch_with_llm_agent,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_artifacts import (
    write_bridge_artifacts,
)
from cais_spade_llm.resources.resource_primitives import (
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
DEFAULT_LIVE_MODEL = _env_default(
    "CAIS_SPADE_LLM_MODEL",
    "OPENAI_MODEL",
    "CASE3_RECOVERY_MODEL",
    fallback="gpt-5.4",
)
DEFAULT_REASONING_EFFORT = _env_default(
    "CAIS_SPADE_REASONING_EFFORT",
    "OPENAI_REASONING_EFFORT",
    fallback="medium",
)
DEBUG_DIR = Path("cais_spade_llm/monitor/debug")
_POST_VALIDATION_INSPECTION_TURNS = 3
CASE3_COMPLETED_TASK_IDS = (
    "REQ_1_T1",
    "REQ_1_T2",
    "REQ_2_T1",
    "REQ_2_T2",
    "REQ_2_T3",
)
CASE3_ARCHIVED_FINAL_OUTPUT_PATH = (
    ROOT
    / "cais_spade_llm"
    / "agents"
    / "intelligent_product"
    / "replanner"
    / "llm_bridge"
    / "runtime_data"
    / "imported"
    / "worked"
    / "1"
    / "multi_turn_turn09_final_output_response_20260423T013259.txt"
)


def _configure_dryrun_logging() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return ROOT


def _resolve_debug_root() -> Path:
    debug_dir = Path(DEBUG_DIR)
    if not debug_dir.is_absolute():
        debug_dir = _repo_root() / debug_dir
    return debug_dir


def _allocate_dryrun_artifact_directory() -> Path:
    artifact_directory = _resolve_debug_root()
    artifact_directory.mkdir(parents=True, exist_ok=True)
    return artifact_directory


def _payload_artifact_directory(payload: dict[str, Any] | None = None) -> Path:
    if isinstance(payload, dict):
        bridge_debug = payload.get("bridge_debug")
        if not isinstance(bridge_debug, dict):
            prepared_bridge_request = payload.get("prepared_bridge_request")
            if isinstance(prepared_bridge_request, dict):
                bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
            else:
                bridge_debug = {}
        for key in ("artifact_directory", "per_turn_debug_dir"):
            candidate_raw = str(bridge_debug.get(key) or "").strip()
            if not candidate_raw:
                continue
            candidate = Path(candidate_raw)
            return candidate if candidate.is_absolute() else (_repo_root() / candidate)
    return _resolve_debug_root()


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_resume_checkpoint_path(path_raw: str | Path) -> Path:
    candidate = Path(path_raw)
    if not candidate.is_absolute():
        candidate = (_repo_root() / candidate).resolve()
    if candidate.suffix.lower() == ".json" and candidate.exists():
        return candidate

    candidate_name = candidate.name
    checkpoint_name_variants = [
        candidate_name.replace("_prompt_", "_resume_checkpoint_").replace(".txt", ".json"),
        candidate_name.replace("_response_", "_resume_checkpoint_").replace(".txt", ".json"),
        candidate_name.replace("_prompt_latest.txt", "_resume_checkpoint_latest.json"),
        candidate_name.replace("_response_latest.txt", "_resume_checkpoint_latest.json"),
    ]
    for checkpoint_name in checkpoint_name_variants:
        if checkpoint_name == candidate_name:
            continue
        checkpoint_path = candidate.with_name(checkpoint_name)
        if checkpoint_path.exists():
            return checkpoint_path
    raise FileNotFoundError(
        f"resume checkpoint not found for {str(candidate)}"
    )


def _load_resume_checkpoint(path_raw: str | Path) -> tuple[Path, dict[str, Any]]:
    checkpoint_path = _resolve_resume_checkpoint_path(path_raw)
    payload = _load_json(checkpoint_path)
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint payload must be a JSON object")
    checkpoint_kind = str(payload.get("kind") or "").strip()
    if checkpoint_kind not in {
        "multi_turn_resume_checkpoint",
        "primitive_batch_resume_checkpoint",
    }:
        raise ValueError(
            f"unsupported resume checkpoint kind: {checkpoint_kind or '<missing>'}"
        )
    return checkpoint_path, payload


def _configure_resume_bridge_debug(
    *,
    prepared_bridge_request: dict[str, Any],
    write_debug: bool,
    write_resume_checkpoints: bool = False,
    checkpoint_path: Path | None = None,
) -> None:
    bridge_debug_seed = dict(prepared_bridge_request.get("bridge_debug") or {})
    bridge_debug_seed["write_resume_checkpoints"] = bool(write_resume_checkpoints)
    if write_debug:
        checkpoint_dir = (
            checkpoint_path.parent
            if checkpoint_path is not None
            else _allocate_dryrun_artifact_directory()
        )
        if not str(bridge_debug_seed.get("artifact_directory") or "").strip():
            bridge_debug_seed["artifact_directory"] = str(checkpoint_dir)
        if not str(bridge_debug_seed.get("per_turn_debug_dir") or "").strip():
            bridge_debug_seed["per_turn_debug_dir"] = str(checkpoint_dir)
    else:
        bridge_debug_seed["artifact_directory"] = ""
        bridge_debug_seed["per_turn_debug_dir"] = ""
    prepared_bridge_request["bridge_debug"] = bridge_debug_seed


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
        llm_reasoning_effort: str | None = None,
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
        self.llm_reasoning_effort = _normalize_reasoning_effort_for_model(
            self.llm_model,
            str(llm_reasoning_effort or DEFAULT_REASONING_EFFORT).strip(),
        )
        self.precomputed_bundle = deepcopy(precomputed_bundle or {})
        precomputed_policy = (
            self.precomputed_bundle.get("replan_policy", {})
            if isinstance(self.precomputed_bundle.get("replan_policy"), dict)
            else {}
        )
        configured_reasoning_mode = str(
            precomputed_policy.get("bridge_reasoning_mode", "multi_turn") or "multi_turn"
        ).strip().lower()
        self._bridge_reasoning_mode = (
            configured_reasoning_mode
            if configured_reasoning_mode == "multi_turn"
            else "multi_turn"
        )
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
                reasoning_effort=self.llm_reasoning_effort,
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
                    "reasoning_effort": self.llm_reasoning_effort,
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
            "grasp_part",
            "release_part",
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
        self.static_capabilities.setdefault("resource_type", "robot")
        self.named_positions = deepcopy(env_block.get("named_positions") or {})
        self._current_state = str(current_state)
        self._held_part = held_part
        self._gripper_state = str(gripper_state)
        self._bridge_pose_ref = pose_ref
        self._position = deepcopy(position)
        self._observations = deepcopy(observations or {})
        self._shared_observations: dict[str, dict[str, Any]] = {}
        self._primitive_catalog_cache: list[dict[str, Any]] | None = None
        self.logger = logging.getLogger(f"FakeBridgeRobot.{self.agent_name or 'robot'}")

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
          current_state:
            set: idle
          current_pose_ref:
            set_from_param: pose_name
          current_pose:
            set_unknown: true
          occupancy.location:
            set_from_param: pose_name
        ---
        """
        self._current_state = "idle"
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

    def grasp_part(
        self,
        model_name: str,
        part_name: str = "",
        position: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Close the gripper and attach the target part as one high-level grasp primitive.
        params:
          model_name: {type: string, description: "Part model name"}
          part_name: {type: string, description: "Optional canonical part name"}
          position: {type: number, description: "Optional gripper position override"}
        preconditions:
          held_part:
            equals: null
        effects:
          current_state:
            set: picked
          gripper_state:
            set: closed
          held_part:
            set_from_param_any_of: [part_name, model_name]
        ---
        """
        if not self.close_gripper(position=position):
            return {"success": False, "message": "fake close_gripper failed"}
        attached = self.attach_part(model_name=model_name, part_name=part_name)
        if attached.get("success"):
            return {"success": True, "message": "fake grasp_part ok"}
        self.open_gripper()
        return {
            "success": False,
            "message": f"{str(attached.get('message') or 'fake attach failed')}; rollback: reopened gripper",
        }

    def release_part(
        self,
        model_name: str = "",
        part_name: str = "",
        assume_released_if_open: bool = False,
    ) -> dict[str, Any]:
        """
        ---
        description: Open the gripper and detach the currently held part as one high-level release primitive.
        params:
          model_name: {type: string, description: "Part model name"}
          part_name: {type: string, description: "Optional canonical part name"}
          assume_released_if_open: {type: boolean, description: "Allow open-gripper release assumption"}
        preconditions:
          held_part:
            not_equals: null
        effects:
          current_state:
            set: idle
          gripper_state:
            set: open
          held_part:
            set: null
        ---
        """
        if not self.open_gripper():
            return {"success": False, "message": "fake open_gripper failed"}
        detached = self.detach_part(
            model_name=model_name,
            assume_released_if_open=assume_released_if_open,
        )
        if detached.get("success"):
            return {"success": True, "message": f"fake release_part ok {part_name or model_name}".strip()}
        self.close_gripper()
        return {
            "success": False,
            "message": f"{str(detached.get('message') or 'fake detach failed')}; rollback: reclosed gripper",
        }

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
        target_pose_source: str = "",
        prefer_live_detection: bool = False,
        approach_height_override_m: float | None = None,
        ignore_current_height_for_travel_z: bool = False,
        min_pick_tcp_z_override_m: float | None = None,
        use_global_min_pick_tcp_z: bool = True,
        surface_clearance_override_m: float | None = None,
        apply_pick_z_adjustments: bool = True,
    ) -> dict[str, Any]:
        """
        ---
        description: Compute pick target positions from perception + geometry without moving.
        params:
          part_name: {type: string, description: "Name of the detected part to pick"}
          product_geometry: {type: object, description: "Optional geometry override dict"}
          target_pose: {type: object, description: "Optional known target pose"}
          target_pose_source: {type: string}
          prefer_live_detection: {type: boolean}
          approach_height_override_m: {type: number, description: "Optional vertical approach distance"}
          ignore_current_height_for_travel_z: {type: boolean}
          min_pick_tcp_z_override_m: {type: number}
          use_global_min_pick_tcp_z: {type: boolean}
          surface_clearance_override_m: {type: number}
          apply_pick_z_adjustments: {type: boolean}
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
        surface_clearance = float(surface_clearance_override_m or 0.0)
        return {
            "success": True,
            "part_name": str(part_name or target.get("part_name") or ""),
            "model_name": "fake_model",
            "tx": pose["x"],
            "ty": pose["y"],
            "tz": pose["z"],
            "pick_z": pose["z"] + 0.02 + surface_clearance,
            "travel_z": pose["z"] + float(approach_height_override_m or 0.2),
            "approach_pose": {"x": pose["x"], "y": pose["y"], "z": pose["z"] + 0.2},
            "target_pose": {"x": pose["x"], "y": pose["y"], "z": pose["z"] + 0.02 + surface_clearance},
            "part_height": 0.08,
            "tcp_offset_z": -0.17,
            "pick_tcp_z": pose["z"] + 0.19 + surface_clearance,
            "surface_clearance_m": surface_clearance,
            "pick_z_adjustment_m": 0.0,
            "apply_pick_z_adjustments": bool(apply_pick_z_adjustments),
            "target_pose_source": target_pose_source,
            "prefer_live_detection": bool(prefer_live_detection),
            "use_global_min_pick_tcp_z": bool(use_global_min_pick_tcp_z),
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

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "resource_jid": self.jid,
            "resource_type": "robot",
            "current_state": self._current_state,
            "current_pose": deepcopy(self._position),
            "current_pose_ref": self._bridge_pose_ref,
            "held_part": self._held_part,
            "gripper_state": self._gripper_state,
        }

    async def _ensure_controller_prewarmed(self) -> None:
        return None

    def _cached_primitive_catalog(self) -> list[dict[str, Any]]:
        if self._primitive_catalog_cache is None:
            from cais_spade_llm.resources.resource_primitives import (
                build_execution_primitive_catalog,
            )

            self._primitive_catalog_cache = build_execution_primitive_catalog(self)
        return deepcopy(self._primitive_catalog_cache)

    async def _execute_primitive(self, primitive: str, params: dict[str, Any]) -> dict[str, Any]:
        method = getattr(self, primitive, None)
        if not callable(method):
            return {"success": False, "message": f"fake robot missing primitive '{primitive}'"}
        result = method(**dict(params or {}))
        if isinstance(result, bool):
            return {"success": result, "message": f"fake {primitive} {'ok' if result else 'failed'}"}
        if isinstance(result, list):
            return {
                "success": True,
                "message": f"fake {primitive} returned {len(result)} items",
                "data": deepcopy(result),
            }
        if isinstance(result, dict):
            return deepcopy(result)
        return {"success": False, "message": f"fake {primitive} returned unexpected type"}

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
        event_instance: Any | None = None,
        schema: Any | None = None,
        projection: Any | None = None,
        part_context: dict[str, Any],
        bridge_snapshot: dict[str, Any],
        operation_kind: str = "",
        part_name: str | None = None,
        grounded_action: dict[str, Any] | None = None,
        **_compat_kwargs: Any,
    ) -> dict[str, Any]:
        bridge_snapshot = deepcopy(bridge_snapshot or {})
        part_context = deepcopy(part_context or {})
        if grounded_action is None and projection is not None:
            end_state = dict(getattr(projection, "end_state", {}) or {})
            part_name = part_name or str(getattr(projection, "part_name", "") or "").strip() or None
            expected_resource = {
                "current_state": str(end_state.get("resource_state") or "").strip() or None,
                "location": str(
                    end_state.get("resource_location")
                    or end_state.get("current_location")
                    or end_state.get("location")
                    or end_state.get("named_pose")
                    or ""
                ).strip() or None,
                "held_part": str(end_state.get("held_part") or "").strip() or None,
            }
            expected_part = {
                "state": str(end_state.get("part_state") or "").strip() or None,
                "location": str(end_state.get("part_location") or "").strip() or None,
                "holder": str(end_state.get("part_holder_resource_jid") or "").strip() or None,
            }
            part_affecting = bool(
                part_name
                and any(
                    expected_part.get(key) not in (None, "", [], {})
                    for key in ("state", "location", "holder")
                )
            )
            resource_affecting = bool(
                any(
                    expected_resource.get(key) not in (None, "", [], {})
                    for key in ("current_state", "location", "held_part")
                )
            )
            effect_scope = (
                "resource_and_part"
                if resource_affecting and part_affecting
                else "part_only"
                if part_affecting
                else "resource_only"
            )
            source_ref = {
                "location": str(
                    getattr(event_instance, "object_bindings", {}).get("source_location") or ""
                ).strip() or None,
            }
            if source_ref.get("location") == "observed_pose":
                observed_pose = dict(part_context.get("observed_pose") or {})
                if observed_pose:
                    source_ref["pose"] = deepcopy(observed_pose)
            grounded_action = {
                "resource_jid": self.jid,
                "part_name": part_name,
                "operation_kind": str(getattr(schema, "action_type", "") or operation_kind or "").strip(),
                "task_kind": str(getattr(schema, "action_type", "") or operation_kind or "").strip(),
                "target": deepcopy(getattr(projection, "action_target", {}) or {}),
                "expected_effect": {
                    "resource": expected_resource,
                    "part": expected_part,
                },
                "preconditions": {
                    "source_ref": source_ref,
                    "part": {
                        "requires_acquisition": str(
                            getattr(schema, "schema_id", "") or ""
                        ).strip().lower()
                        == "pick_part"
                    },
                },
                "effect_scope": effect_scope,
            }
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
    reasoning_mode: str = "multi_turn",
) -> None:
    """Tune the prepared bridge session for the direct dry-run harness."""
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    normalized_mode = str(reasoning_mode or "multi_turn").strip().lower() or "multi_turn"
    if normalized_mode != "multi_turn":
        normalized_mode = "multi_turn"
    bridge_session["reasoning_mode"] = normalized_mode
    bridge_session["max_turns"] = max(int(bridge_session.get("max_turns", 6) or 6), 1000)
    bridge_session["repair_mode"] = "recover"
    bridge_session["observation_backend"] = "mock_detect_parts_harness"
    bridge_session["outline_mode"] = "incremental_candidates_validated"
    prepared_bridge_request["bridge_session"] = bridge_session


def _case3_known_accepted_outline_prefix() -> list[dict[str, Any]]:
    return [
        {
            "outline_id": "RECOVERY_SEQ1",
            "event_name": "recover_to_home_idle",
            "resource_jid": "xarm6@localhost",
            "rationale": (
                "Enabled because xarm6 is currently failed and has a grounded named pose "
                "home. This directly advances the open guard requiring xarm6@localhost "
                "to reach idle and reduces coordination risk before LG/MCP recovery "
                "continues."
            ),
            "description": (
                "Enabled because xarm6 is currently failed and has a grounded named pose "
                "home. This directly advances the open guard requiring xarm6@localhost "
                "to reach idle and reduces coordination risk before LG/MCP recovery "
                "continues."
            ),
            "predecessors": [],
            "expected_start_state": {
                "resource_state": "failed",
            },
            "expected_end_state": {
                "resource_state": "idle",
                "resource_location": "home",
            },
        },
        {
            "outline_id": "RECOVERY_SEQ2",
            "event_name": "stage_mcp_to_prusa_mk4_2",
            "resource_jid": "ur5e@localhost",
            "part_name": "MCP",
            "target_ref": "prusa-mk4-2",
            "rationale": (
                "Enabled now because ur5e is holding MCP and prusa-mk4-2 is a grounded "
                "reachable location for ur5e. This clears ur5e's gripper so it can "
                "recover LG first, which is required before any MCP place approach at "
                "assembly_board-v1."
            ),
            "description": (
                "Enabled now because ur5e is holding MCP and prusa-mk4-2 is a grounded "
                "reachable location for ur5e. This clears ur5e's gripper so it can "
                "recover LG first, which is required before any MCP place approach at "
                "assembly_board-v1."
            ),
            "predecessors": [],
            "expected_start_state": {
                "resource_state": "picked",
                "held_part": "MCP",
                "part_state": "in_gripper",
                "part_location": "ur5e@localhost_gripper",
            },
            "expected_end_state": {
                "resource_state": "idle",
                "held_part": None,
                "part_state": "placed",
                "part_location": "prusa-mk4-2",
            },
        },
        {
            "outline_id": "RECOVERY_SEQ3",
            "event_name": "recover_pick_LG_from_observed_pose",
            "resource_jid": "ur5e@localhost",
            "part_name": "LG",
            "rationale": (
                "Enabled because ur5e@localhost is idle, not holding any part, and LG "
                "is misplaced and unheld at an observed pose within ur5e's reachable "
                "workspace. This is the necessary next recovery step to clear the "
                "blocker on LG and move toward satisfying REQ_2, which must be "
                "completed before MCP can safely proceed to the assembly station."
            ),
            "description": (
                "Enabled because ur5e@localhost is idle, not holding any part, and LG "
                "is misplaced and unheld at an observed pose within ur5e's reachable "
                "workspace. This is the necessary next recovery step to clear the "
                "blocker on LG and move toward satisfying REQ_2, which must be "
                "completed before MCP can safely proceed to the assembly station."
            ),
            "predecessors": ["RECOVERY_SEQ2"],
            "expected_start_state": {
                "resource_state": "idle",
                "held_part": None,
                "part_state": "misplaced",
            },
            "expected_end_state": {
                "resource_state": "idle",
                "held_part": "LG",
                "part_state": "held",
            },
        },
        {
            "outline_id": "RECOVERY_SEQ4",
            "event_name": "recover_place_LG_to_assembly_board-v1",
            "resource_jid": "ur5e@localhost",
            "part_name": "LG",
            "target_ref": "assembly_board-v1",
            "rationale": (
                "Enabled now because ur5e is idle, currently holding LG, and can reach "
                "assembly_board-v1. This directly satisfies REQ_2 and clears the "
                "explicit guard that LG must be placed at assembly_board-v1 before MCP "
                "can proceed toward assembly placement."
            ),
            "description": (
                "Enabled now because ur5e is idle, currently holding LG, and can reach "
                "assembly_board-v1. This directly satisfies REQ_2 and clears the "
                "explicit guard that LG must be placed at assembly_board-v1 before MCP "
                "can proceed toward assembly placement."
            ),
            "predecessors": ["RECOVERY_SEQ3"],
            "expected_start_state": {
                "resource_state": "idle",
                "held_part": "LG",
                "part_state": "held",
            },
            "expected_end_state": {
                "resource_state": "idle",
                "held_part": None,
                "part_state": "placed",
                "part_location": "assembly_board-v1",
            },
        },
    ]


def _seed_case3_primitive_generation_focus(session_state: dict[str, Any]) -> dict[str, Any]:
    seeded = deepcopy(session_state)
    seeded["current_phase"] = "primitive_generation"
    seeded["status"] = "pending"
    seeded["turn_index"] = 7
    seeded["accepted_outline_prefix"] = _case3_known_accepted_outline_prefix()
    seeded["primitive_generation_cursor"] = 0
    seeded["accepted_primitive_program"] = []
    seeded["primitive_rejection_feedback"] = []
    seeded["candidate_rejection_feedback"] = []
    seeded["outline_validation_findings"] = []
    seeded["observation_store"] = {
        "observed_pose_LG": {
            "part_name": "LG",
            "x": 0.0,
            "y": 0.2,
            "z": 1.035,
            "pose": {"x": 0.0, "y": 0.2, "z": 1.035},
        }
    }
    resources = seeded.setdefault("symbolic_resources", {})
    resources.setdefault("xarm6@localhost", {"resource_jid": "xarm6@localhost"})
    resources.setdefault("ur5e@localhost", {"resource_jid": "ur5e@localhost"})
    resources["xarm6@localhost"].update({
        "current_state": "failed",
        "held_part": None,
        "gripper_state": "open",
    })
    resources["ur5e@localhost"].update({
        "current_state": "picked",
        "current_location": "prusa-mk4-2",
        "held_part": "MCP",
        "gripper_state": "closed",
    })
    parts = seeded.setdefault("symbolic_parts", {})
    parts.setdefault("LG", {"part_name": "LG"})
    parts.setdefault("MCP", {"part_name": "MCP"})
    parts["LG"].update({
        "current_state": "misplaced",
        "current_location": None,
        "observed_pose": {"x": 0.0, "y": 0.2, "z": 1.035},
        "current_holder_resource_jid": None,
    })
    parts["MCP"].update({
        "current_state": "in_gripper",
        "current_location": "ur5e@localhost_gripper",
        "current_holder_resource_jid": "ur5e@localhost",
    })
    return seeded


def test_case3_seeded_primitive_focus_seed_starts_with_xarm6_seq1() -> None:
    session_state = _seed_case3_primitive_generation_focus({})
    first_event = deepcopy(session_state["accepted_outline_prefix"][0])
    second_event = deepcopy(session_state["accepted_outline_prefix"][1])

    assert first_event["outline_id"] == "RECOVERY_SEQ1"
    assert first_event["resource_jid"] == "xarm6@localhost"
    assert first_event["event_name"] == "recover_to_home_idle"
    assert second_event["outline_id"] == "RECOVERY_SEQ2"
    assert second_event["resource_jid"] == "ur5e@localhost"
    assert second_event["event_name"] == "stage_mcp_to_prusa_mk4_2"
    assert first_event["predecessors"] == []
    assert second_event["predecessors"] == []


def test_case3_seeded_primitive_context_uses_event_local_start_state_for_seq2() -> None:
    session_state = _seed_case3_primitive_generation_focus({})
    outline_event = deepcopy(session_state["accepted_outline_prefix"][1])
    prepared_bridge_request = {
        "bridge_resources": {
            "ur5e@localhost": {
                "resource_type": "resource",
                "bridge_snapshot": {
                    "resource_jid": "ur5e@localhost",
                    "resource_type": "resource",
                    "current_state": "idle",
                    "current_location": "prusa-mk4-2",
                    "held_part": None,
                    "gripper_state": "closed",
                },
            }
        },
        "grounding_context": {"parts": {}},
    }

    held_part, held_error = _resolve_context_ref(
        ref="/resources/ur5e@localhost/held_part",
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        outline_event=outline_event,
    )
    snapshot, snapshot_error = _resolve_context_ref(
        ref="/resources/ur5e@localhost/snapshot",
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        outline_event=outline_event,
    )

    assert held_error is None
    assert snapshot_error is None
    assert held_part == "MCP"
    assert snapshot["resource_state"] == "picked"
    assert snapshot["held_part"] == "MCP"
    assert "current_state" not in snapshot


def test_bridge_snapshot_mismatch_ignores_missing_current_location_for_part_projection() -> None:
    mismatch = ProductRecoveryController._bridge_snapshot_mismatch(
        actual_snapshot={
            "resource_type": "robot",
            "current_state": "idle",
            "current_location": None,
            "held_part": None,
            "gripper_state": "open",
        },
        projected_snapshot={
            "resource_type": "robot",
            "current_state": "idle",
            "current_location": "prusa-mk4-2",
            "held_part": None,
            "gripper_state": "open",
        },
        allow_missing_current_location=True,
    )

    assert mismatch == ""


def test_bridge_snapshot_mismatch_accepts_picked_alias_from_closed_gripper() -> None:
    mismatch = ProductRecoveryController._bridge_snapshot_mismatch(
        actual_snapshot={
            "resource_type": "robot",
            "current_state": "idle",
            "current_location": None,
            "held_part": "LG",
            "gripper_state": "closed",
        },
        projected_snapshot={
            "resource_type": "robot",
            "current_state": "picked",
            "current_location": "prusa-mk4-2",
            "held_part": "LG",
            "gripper_state": "closed",
        },
        allow_missing_current_location=True,
    )

    assert mismatch == ""


def test_snapshot_matches_expected_accepts_picked_alias_from_closed_gripper() -> None:
    matches, mismatch = snapshot_matches_expected(
        actual={
            "resource_type": "robot",
            "current_state": "idle",
            "held_part": "LG",
            "gripper_state": "closed",
        },
        expected={
            "resource_type": "robot",
            "current_state": "picked",
            "held_part": "LG",
            "gripper_state": "closed",
        },
    )

    assert matches is True
    assert mismatch is None


def test_ack_status_is_regression_for_late_bridge_updates() -> None:
    assert _ack_status_is_regression("completed", "running") is True
    assert _ack_status_is_regression("running", "accepted") is True
    assert _ack_status_is_regression("dispatched", "running") is False
    assert _ack_status_is_regression("running", "failed") is False


def test_should_persist_ack_state_skips_transient_bridge_updates() -> None:
    bridge_task = {"function_name": "execute_recovery_macro"}
    nominal_task = {"function_name": "move_home"}

    assert _should_persist_ack_state(bridge_task, "accepted") is False
    assert _should_persist_ack_state(bridge_task, "running") is False
    assert _should_persist_ack_state(bridge_task, "completed") is True
    assert _should_persist_ack_state(nominal_task, "running") is True


def test_case3_archived_bridge_exposes_two_ready_roots_for_batch_dispatch(
    tmp_path: Path,
) -> None:
    _, product_agent, planner, _, recovery = _approve_case3_archived_bridge(tmp_path)

    assert recovery["status"] == "resolved"
    ready_nodes = product_agent._active_bridge_ready_tasks(max_count=2)

    assert [str(node.get("bridge_outline_id") or "").strip() for node in ready_nodes] == [
        "RECOVERY_SEQ1",
        "RECOVERY_SEQ2",
    ]
    assert [str(node.get("resource_jid") or "").strip() for node in ready_nodes] == [
        "xarm6@localhost",
        "ur5e@localhost",
    ]
    assert all(str(node.get("status") or "").strip() == "pending" for node in ready_nodes)


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
    requested_reasoning_mode = (
        str(reasoning_mode or "multi_turn").strip().lower() or "multi_turn"
    )
    product_agent._bridge_reasoning_mode = (
        requested_reasoning_mode
        if requested_reasoning_mode == "multi_turn"
        else "multi_turn"
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
        "cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_session.asyncio.to_thread",
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
    prepared_bridge_request["multi_turn_session_seed"] = (
        multi_turn_mode.build_multi_turn_session_seed(prepared_bridge_request)
    )

    return fixture, product_agent, planner, prepared_bridge_request


class _ImmediateThread:
    def __init__(
        self,
        *,
        target: Callable[..., Any] | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        name: str | None = None,
        daemon: bool | None = None,
    ) -> None:
        self._target = target
        self._args = tuple(args or ())
        self._kwargs = dict(kwargs or {})
        self.name = name
        self.daemon = daemon

    def start(self) -> None:
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def join(self, timeout: float | None = None) -> None:
        del timeout
        return None


def _configure_runtime_bridge_approval_harness(
    *,
    product_agent: FakeProductAgent,
    planner: ProcessPlanner,
    fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    product_agent._utc_now_iso = staticmethod(
        lambda: datetime.now(timezone.utc).isoformat()
    )
    product_agent._direct_predecessors_from_nodes = staticmethod(
        ProductAgent._direct_predecessors_from_nodes
    )
    product_agent._collect_descendants_from_nodes = staticmethod(
        ProductAgent._collect_descendants_from_nodes
    )
    product_agent._violation_summary = staticmethod(
        lambda violations: (
            sorted(
                {
                    str(v.get("violated_rule_id"))
                    for v in (violations if isinstance(violations, list) else [])
                    if isinstance(v, dict) and v.get("violated_rule_id")
                }
            ),
            len(violations if isinstance(violations, list) else []),
        )
    )
    product_agent.agent_name = "assembly_board-v1"
    product_agent.process_planner = planner
    product_agent.resource_agents = list(getattr(planner, "resource_agents", []) or [])
    product_agent.task_states = {}
    product_agent.part_tracker = deepcopy(fixture.get("part_tracker") or {})
    product_agent.execution_timeline = []
    product_agent.runtime_repair_state = "idle"
    product_agent.plan_safety_alert = None
    product_agent._runtime_repair_inflight = False
    product_agent._runtime_repair_fail_streak = 0
    product_agent._runtime_repair_max_attempts = 3
    product_agent._bridge_generation_mode = "auto"
    product_agent._runtime_bridge_mode = "pre_ran"
    product_agent._runtime_bridge_validation_policy = "no_validation"
    product_agent._runtime_bridge_start_safety_mode = "cca_check"
    product_agent._runtime_bridge_execution_shape = "dag"
    product_agent._runtime_bridge_archive_path = str(CASE3_ARCHIVED_FINAL_OUTPUT_PATH)
    product_agent._runtime_bridge_archive_label = CASE3_ARCHIVED_FINAL_OUTPUT_PATH.name
    product_agent._orphaned_bridge_task_warning_ids = set()
    product_agent._generated_bridge_gazebo_verification_enabled = False
    product_agent.cca_jid = "cca@localhost"
    product_agent.plan_path = tmp_path / "case3_plan.json"
    product_agent.global_fsa_path = tmp_path / "case3_global_fsa.json"
    product_agent.product_state_path = tmp_path / "case3_product_state.json"
    product_agent.resource_state_path = tmp_path / "case3_resource_state.json"
    product_agent.recovery_controller = ProductRecoveryController(product_agent)
    product_agent.recovery_controller.bind_methods()
    product_agent._refresh_bridge_sequence_runtime_metadata = (
        product_agent.recovery_controller._refresh_bridge_sequence_runtime_metadata
    )
    product_agent.runtime_recovery = product_agent._empty_runtime_recovery()
    product_agent._runtime_recovery_context = {}


def _approve_case3_archived_bridge(
    tmp_path: Path,
) -> tuple[dict[str, Any], FakeProductAgent, ProcessPlanner, dict[str, Any], dict[str, Any]]:
    fixture, product_agent, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(reasoning_mode="multi_turn")
    )
    _configure_runtime_bridge_approval_harness(
        product_agent=product_agent,
        planner=planner,
        fixture=fixture,
        tmp_path=tmp_path,
    )

    final_output_payload = _load_json(CASE3_ARCHIVED_FINAL_OUTPUT_PATH)
    if not isinstance(final_output_payload, dict):
        raise TypeError("archived final output payload must decode to an object")
    proposal_result = multi_turn_mode.build_multi_turn_bridge_proposal(
        final_output_payload=deepcopy(final_output_payload),
        prepared_bridge_request=deepcopy(prepared_bridge_request),
    )
    assert proposal_result["accepted"] is True
    bridge_proposal = deepcopy(proposal_result.get("bridge_proposal") or {})
    bridge_debug = {
        "source": "archived_final_output",
        "archive_replay": {
            "source_path": str(CASE3_ARCHIVED_FINAL_OUTPUT_PATH),
            "source_label": CASE3_ARCHIVED_FINAL_OUTPUT_PATH.name,
        },
        "execution_policy": {
            "complete_full_tail": True,
            "execution_shape": "dag",
            "start_safety_mode": "cca_check",
        },
    }
    violations = [
        {
            "failed_task_id": FAILED_TASK_ID,
            "task_id": FAILED_TASK_ID,
            "resource_jid": "xarm6@localhost",
            "failure_context": deepcopy(fixture.get("failure_context") or {}),
        }
    ]
    product_agent._runtime_recovery_context = {
        "trigger": "runtime_des_replan",
        "failed_task_id": FAILED_TASK_ID,
        "violations": deepcopy(violations),
        "prepared_bridge_request": deepcopy(prepared_bridge_request),
        "system_coordination_state": {
            "resource_states": deepcopy(fixture.get("resource_states") or {})
        },
    }
    product_agent._set_runtime_recovery(
        reset=True,
        status="llm_bridge",
        resolution_class="none",
        trigger="runtime_des_replan",
        failed_task_id=FAILED_TASK_ID,
        message="Awaiting archived bridge approval.",
        used_llm_bridge=True,
        bridge_proposal=bridge_proposal,
        bridge_debug=bridge_debug,
        bridge_approval_state="pending",
        violations=violations,
    )
    with patch(
        "cais_spade_llm.agents.intelligent_product.product_recovery_controller.threading.Thread",
        new=_ImmediateThread,
    ):
        recovery = product_agent.approve_runtime_bridge_proposal_sync()
    return fixture, product_agent, planner, prepared_bridge_request, recovery


def test_case3_archived_bridge_approval_compiles_per_resource_concurrency(
    tmp_path: Path,
) -> None:
    _, product_agent, planner, _, recovery = _approve_case3_archived_bridge(tmp_path)

    assert recovery["status"] == "resolved"
    active_bridge_sequence = dict(product_agent.runtime_recovery.get("active_bridge_sequence") or {})
    assert active_bridge_sequence["execution_shape"] == "dag"
    assert active_bridge_sequence["start_safety_mode"] == "cca_check"

    bridge_nodes_by_outline_id = {
        str(node.get("bridge_outline_id") or node.get("params", {}).get("outline_id") or "").strip(): node
        for node in planner.nodes
        if isinstance(node, dict)
        and str(node.get("function_name") or "").strip() == "execute_recovery_macro"
    }
    seq1 = dict(bridge_nodes_by_outline_id["RECOVERY_SEQ1"])
    seq2 = dict(bridge_nodes_by_outline_id["RECOVERY_SEQ2"])
    seq3 = dict(bridge_nodes_by_outline_id["RECOVERY_SEQ3"])
    seq4 = dict(bridge_nodes_by_outline_id["RECOVERY_SEQ4"])
    req_1_t3 = dict(planner._find_node("REQ_1_T3") or {})
    req_1_t4 = dict(planner._find_node("REQ_1_T4") or {})
    req_1_t5 = dict(planner._find_node("REQ_1_T5") or {})
    req_2_t5 = dict(planner._find_node("REQ_2_T5") or {})

    assert max(
        int(seq2.get("sequence_index") or 0),
        int(seq3.get("sequence_index") or 0),
        int(seq4.get("sequence_index") or 0),
    ) < min(
        int(req_1_t3.get("sequence_index") or 0),
        int(req_1_t4.get("sequence_index") or 0),
        int(req_1_t5.get("sequence_index") or 0),
    )
    assert int(seq1.get("sequence_index") or 0) < int(req_2_t5.get("sequence_index") or 0)
    assert seq4["id"] in list(req_1_t3.get("predecessors") or [])
    assert seq1["id"] not in list(req_1_t3.get("predecessors") or [])
    assert seq4["id"] not in list(req_2_t5.get("predecessors") or [])
    assert seq1["id"] in list(req_2_t5.get("predecessors") or [])

    transitions = list((planner.global_fsa or {}).get("A", {}).get("Tr", []) or [])
    seq2_start_events = [
        tr
        for tr in transitions
        if isinstance(tr, dict) and str(tr.get("task_id") or "").strip() == str(seq2.get("id") or "").strip()
        and str(tr.get("event") or "").strip().endswith(".start")
    ]
    assert any(
        str(tr.get("from") or "")
        == "(ur5e@localhost=(k=2,idle),xarm6@localhost=(k=3,idle))"
        for tr in seq2_start_events
    )


def test_case3_archived_bridge_allows_xarm6_nominal_release_while_ur5e_bridge_active(
    tmp_path: Path,
) -> None:
    _, product_agent, planner, _, _ = _approve_case3_archived_bridge(tmp_path)

    bridge_nodes_by_outline_id = {
        str(node.get("bridge_outline_id") or node.get("params", {}).get("outline_id") or "").strip(): node
        for node in planner.nodes
        if isinstance(node, dict)
        and str(node.get("function_name") or "").strip() == "execute_recovery_macro"
    }
    seq1 = bridge_nodes_by_outline_id["RECOVERY_SEQ1"]
    seq2 = bridge_nodes_by_outline_id["RECOVERY_SEQ2"]
    seq3 = bridge_nodes_by_outline_id["RECOVERY_SEQ3"]
    seq4 = bridge_nodes_by_outline_id["RECOVERY_SEQ4"]

    seq1["status"] = "completed"
    seq2["status"] = "running"
    seq3["status"] = "pending"
    seq4["status"] = "pending"
    req_2_t5 = planner._find_node("REQ_2_T5")
    assert isinstance(req_2_t5, dict)
    req_2_t5["status"] = "pending"

    refreshed_sequence = product_agent._refresh_bridge_sequence_runtime_metadata(
        product_agent.runtime_recovery.get("active_bridge_sequence") or {}
    )
    product_agent._set_runtime_recovery(
        message=str(product_agent.runtime_recovery.get("message") or "").strip(),
        active_bridge_sequence=refreshed_sequence,
    )

    with patch.object(
        ProductRecoveryController,
        "_build_runtime_plant_state",
        return_value={},
    ), patch.object(
        ProductRecoveryController,
        "_event_guard_violations",
        return_value=[],
    ), patch.object(
        ProductRecoveryController,
        "_record_runtime_des_trace",
        return_value=None,
    ), patch.object(
        ProductRecoveryController,
        "_try_compile_controllable_repair",
        return_value=None,
    ), patch.object(
        ProductRecoveryController,
        "_mark_runtime_des_human_required",
        return_value=None,
    ):
        next_node = product_agent._select_runtime_event()

    assert isinstance(next_node, dict)
    assert str(next_node.get("id") or "").strip() == "REQ_2_T5"
    assert str(next_node.get("resource_jid") or "").strip() == "xarm6@localhost"


def test_case3_archived_bridge_place_macros_use_snap_and_cartesian_retreat() -> None:
    final_output_payload = _load_json(CASE3_ARCHIVED_FINAL_OUTPUT_PATH)
    accepted_program = list(final_output_payload.get("accepted_primitive_program") or [])
    event_rows = {
        str(row.get("event_name") or "").strip(): row
        for row in accepted_program
        if isinstance(row, dict)
    }

    stage_steps = [
        str(step.get("primitive") or "").strip()
        for step in list(event_rows["stage_mcp_to_prusa_mk4_2"].get("primitive_steps") or [])
        if isinstance(step, dict)
    ]
    place_steps = [
        str(step.get("primitive") or "").strip()
        for step in list(event_rows["recover_place_LG_to_assembly_board-v1"].get("primitive_steps") or [])
        if isinstance(step, dict)
    ]

    assert stage_steps[-3:] == ["release_part", "snap_part_to_slot", "move_cartesian"]
    assert place_steps[-3:] == ["release_part", "snap_part_to_slot", "move_cartesian"]


def test_case3_archived_bridge_does_not_inject_mcp_repick_step() -> None:
    final_output_payload = _load_json(CASE3_ARCHIVED_FINAL_OUTPUT_PATH)
    accepted_program = list(final_output_payload.get("accepted_primitive_program") or [])
    event_names = {
        str(row.get("event_name") or "").strip()
        for row in accepted_program
        if isinstance(row, dict)
    }

    assert int(final_output_payload.get("accepted_trace_length") or 0) == 4
    assert "recover_pick_MCP_from_prusa_mk4_2" not in event_names


# ---------------------------------------------------------------------------
# Main coroutine
# ---------------------------------------------------------------------------


async def run_case3_bridge_dryrun(
    write_debug: bool = True,
    *,
    llm_model: str | None = None,
    reasoning_mode: str = "multi_turn",
    stop_before_primitive_generation: bool = True,
    focus: str = "full",
    resume_checkpoint: str | Path | None = None,
    write_resume_checkpoints: bool = False,
) -> dict[str, Any]:
    """Run the Case 3 dry-run scenario through the bridge once."""
    _configure_dryrun_logging()
    normalized_reasoning_mode = (
        str(reasoning_mode or "multi_turn").strip().lower() or "multi_turn"
    )
    if normalized_reasoning_mode != "multi_turn":
        normalized_reasoning_mode = "multi_turn"
    normalized_focus = str(focus or "full").strip().lower()
    if normalized_focus not in {"full", "primitive_generation"}:
        raise ValueError("focus must be 'full' or 'primitive_generation'")
    checkpoint_path: Path | None = None
    resume_payload: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        checkpoint_path, resume_payload = _load_resume_checkpoint(resume_checkpoint)
    if normalized_focus == "primitive_generation":
        stop_before_primitive_generation = False
    _, product_agent, planner, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
        llm_model=llm_model,
        reasoning_mode=normalized_reasoning_mode,
    )

    if resume_payload is not None:
        prepared_bridge_request = deepcopy(
            resume_payload.get("prepared_bridge_request") or {}
        )
        if not isinstance(prepared_bridge_request, dict):
            raise ValueError("resume checkpoint prepared_bridge_request is invalid")
        _configure_resume_bridge_debug(
            prepared_bridge_request=prepared_bridge_request,
            write_debug=write_debug,
            write_resume_checkpoints=write_resume_checkpoints,
            checkpoint_path=checkpoint_path,
        )
    elif write_debug:
        debug_dir = _allocate_dryrun_artifact_directory()
        bridge_debug_seed = dict(prepared_bridge_request.get("bridge_debug") or {})
        bridge_debug_seed["artifact_directory"] = str(debug_dir)
        bridge_debug_seed["per_turn_debug_dir"] = str(debug_dir)
        bridge_debug_seed["write_resume_checkpoints"] = bool(write_resume_checkpoints)
        prepared_bridge_request["bridge_debug"] = bridge_debug_seed

    proposal: dict[str, Any] | None = None
    if resume_payload is not None and str(resume_payload.get("kind") or "").strip() == "primitive_batch_resume_checkpoint":
        resource_jid = str(resume_payload.get("resource_jid") or "").strip()
        resource_agents = multi_turn_mode._resource_agent_map(planner)
        llm_owner = resource_agents.get(resource_jid)
        if not callable(getattr(llm_owner, "ask_llm_structured", None)):
            llm_owner = product_agent
        primitive_result = await generate_primitive_batch_with_llm_agent(
            llm_agent=llm_owner,
            prepared_bridge_request=prepared_bridge_request,
            assigned_outline_events=[
                deepcopy(row)
                for row in (resume_payload.get("assigned_outline_events") or [])
                if isinstance(row, dict)
            ],
            bridge_session_id=str(resume_payload.get("bridge_session_id") or "").strip(),
            session_state=deepcopy(resume_payload.get("session_state") or {}),
        )
        primitive_session = deepcopy(primitive_result.get("session_state") or {})
        latest_turn = {}
        if isinstance(primitive_session.get("turns"), list) and primitive_session["turns"]:
            latest_turn = dict(primitive_session["turns"][-1] or {})
        primitive_bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        primitive_bridge_debug["status"] = str(primitive_result.get("decision") or "")
        primitive_bridge_debug["multi_turn_session"] = deepcopy(primitive_session)
        result = {
            "scenario": "case3_lg_slippage",
            "reasoning_mode": normalized_reasoning_mode,
            "status": str(primitive_result.get("decision") or ""),
            "proposal": deepcopy(primitive_result.get("bridge_proposal") or {}),
            "bridge_debug": primitive_bridge_debug,
            "prepared_bridge_request": prepared_bridge_request,
            "context_summary": deepcopy(prepared_bridge_request.get("context_summary") or {}),
            "llm_input": deepcopy(prepared_bridge_request.get("llm_input") or {}),
            "multi_turn_session": primitive_session,
            "turns": [
                deepcopy(row)
                for row in (primitive_session.get("turns") or [])
                if isinstance(row, dict)
            ],
            "turn_log": deepcopy(product_agent.turn_log),
            "resume_checkpoint_source_path": str(checkpoint_path or ""),
            "prompt_artifact_path": str(
                latest_turn.get("prompt_artifact_path") or ""
            ) or None,
            "latest_prompt_artifact_path": None,
            "response_artifact_path": str(
                latest_turn.get("response_artifact_path") or ""
            ) or None,
            "latest_response_artifact_path": None,
            "session_transcript_artifact_path": None,
            "latest_session_transcript_artifact_path": None,
            "resume_checkpoint_artifact_path": None,
            "latest_resume_checkpoint_artifact_path": None,
            "primitive_resume_checkpoint_artifact_path": str(
                latest_turn.get("primitive_resume_checkpoint_artifact_path") or ""
            ) or None,
            "latest_primitive_resume_checkpoint_artifact_path": str(
                latest_turn.get("latest_primitive_resume_checkpoint_artifact_path") or ""
            ) or None,
        }
        if write_debug:
            artifact_paths = _write_debug_artifacts(result)
            result.update(artifact_paths)
        return result
    if resume_payload is not None:
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
            execute_multi_turn_bridge as _resume_bridge,
        )
        proposal = await _resume_bridge(
            planner,
            prepared_bridge_request,
            session_state=deepcopy(resume_payload.get("session_state") or {}),
        )
    elif normalized_focus == "primitive_generation":
        seed = deepcopy(
            prepared_bridge_request.get("multi_turn_session_seed")
            or multi_turn_mode.build_multi_turn_session_seed(prepared_bridge_request)
        )
        seed = _seed_case3_primitive_generation_focus(seed)
        prepared_bridge_request["multi_turn_session_seed"] = deepcopy(seed)
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
            execute_multi_turn_bridge as _resume_bridge,
        )
        proposal = await _resume_bridge(
            planner,
            prepared_bridge_request,
            session_state=seed,
        )
    else:
        # First run: grounding + first outline task
        if stop_before_primitive_generation:
            prepared_bridge_request["_stop_after_multi_turn_phase"] = "outline"
        proposal = await planner.execute_prepared_bridge_request(prepared_bridge_request)
        prepared_bridge_request["_stop_after_multi_turn_phase"] = ""

    effective_reasoning_mode = str(
        dict(prepared_bridge_request.get("bridge_session") or {}).get("reasoning_mode")
        or normalized_reasoning_mode
        or "multi_turn"
    ).strip().lower()
    if effective_reasoning_mode == "multi_turn":
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
            current_status = str(ss.get("status") or "")
            if current_status in {
                "paused_after_primitive_stuck",
                "paused_after_primitive_blocked",
            }:
                diagnostics = [
                    deepcopy(row)
                    for row in (ss.get("primitive_escalation_diagnostics") or [])
                    if isinstance(row, dict)
                ]
                feedback = [
                    deepcopy(row)
                    for row in (ss.get("primitive_rejection_feedback") or [])
                    if isinstance(row, dict)
                ]
                summary = str((diagnostics[0] or {}).get("reason") or "").strip() if diagnostics else ""
                if not summary and feedback:
                    summary = str((feedback[0] or {}).get("reason") or "").strip()
                logging.getLogger("case3_bridge_dryrun").warning(
                    "[DryRun] Primitive generation paused%s%s%s",
                    (
                        " on blocked event"
                        if current_status == "paused_after_primitive_blocked"
                        else " on stuck event"
                    ),
                    (
                        f" {str((diagnostics[0] or {}).get('outline_id') or '').strip()}"
                        if diagnostics else ""
                    ),
                    (f": {summary}" if summary else ""),
                )
                break
            if current_status not in {
                "paused_after_outline_turn",
                "ready_for_primitive_generation",
                "paused_after_primitive_turn",
            }:
                break
            if stop_before_primitive_generation and str(ss.get("current_phase") or "").strip().lower() == "primitive_generation":
                logging.getLogger("case3_bridge_dryrun").info(
                    "[DryRun] Outline completed; stopping before primitive_generation for inspection"
                )
                break
            if stop_before_primitive_generation:
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
            proposal = await _resume_bridge(
                planner, prepared_bridge_request, session_state=ss,
            )
            if post_validation_resume_budget > 0:
                post_validation_resume_budget -= 1
                next_ss = prepared_bridge_request.get("multi_turn_session_state") or {}
                if (
                    post_validation_resume_budget == 0
                    and next_ss.get("status") in {
                        "paused_after_outline_turn",
                        "ready_for_primitive_generation",
                    }
                ):
                    logging.getLogger("case3_bridge_dryrun").info(
                        "[DryRun] Paused after final post-validation inspection turn — inspect debug artifacts"
                    )
                    break

    bridge_debug = planner.get_last_bridge_debug()
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    reasoning_mode = str(bridge_session.get("reasoning_mode") or "multi_turn").strip()
    multi_turn_session = dict((bridge_debug or {}).get("multi_turn_session") or {})
    turns = list(multi_turn_session.get("turns") or [])
    latest_turn = dict(turns[-1] or {}) if turns else {}

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
        "resume_checkpoint_source_path": str(checkpoint_path or ""),
        "prompt_artifact_path": str(latest_turn.get("prompt_artifact_path") or "") or None,
        "latest_prompt_artifact_path": None,
        "response_artifact_path": str(latest_turn.get("response_artifact_path") or "") or None,
        "latest_response_artifact_path": None,
        "session_transcript_artifact_path": str(
            latest_turn.get("session_transcript_artifact_path") or ""
        ) or None,
        "latest_session_transcript_artifact_path": None,
        "resume_checkpoint_artifact_path": str(
            latest_turn.get("resume_checkpoint_artifact_path") or ""
        ) or None,
        "latest_resume_checkpoint_artifact_path": str(
            latest_turn.get("latest_resume_checkpoint_artifact_path") or ""
        ) or None,
        "primitive_resume_checkpoint_artifact_path": None,
        "latest_primitive_resume_checkpoint_artifact_path": None,
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
    debug_dir = _payload_artifact_directory(payload)
    debug_dir.mkdir(parents=True, exist_ok=True)
    return write_bridge_artifacts(
        payload,
        phase_label=filename_prefix,
        debug_dir=debug_dir,
        write_latest=False,
        write_session_transcript=True,
        write_phase_prompt_response=False,
        filename_prefix=filename_prefix,
    )


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
    resume_checkpoint_source_path = result.get("resume_checkpoint_source_path")
    prompt_artifact_path = result.get("prompt_artifact_path")
    latest_prompt_artifact_path = result.get("latest_prompt_artifact_path")
    response_artifact_path = result.get("response_artifact_path")
    latest_response_artifact_path = result.get("latest_response_artifact_path")
    session_transcript_artifact_path = result.get("session_transcript_artifact_path")
    latest_session_transcript_artifact_path = result.get("latest_session_transcript_artifact_path")
    resume_checkpoint_artifact_path = result.get("resume_checkpoint_artifact_path")
    latest_resume_checkpoint_artifact_path = result.get("latest_resume_checkpoint_artifact_path")
    primitive_resume_checkpoint_artifact_path = result.get("primitive_resume_checkpoint_artifact_path")
    latest_primitive_resume_checkpoint_artifact_path = result.get("latest_primitive_resume_checkpoint_artifact_path")
    if not any(
        (
            resume_checkpoint_source_path,
            prompt_artifact_path,
            latest_prompt_artifact_path,
            response_artifact_path,
            latest_response_artifact_path,
            session_transcript_artifact_path,
            latest_session_transcript_artifact_path,
            resume_checkpoint_artifact_path,
            latest_resume_checkpoint_artifact_path,
            primitive_resume_checkpoint_artifact_path,
            latest_primitive_resume_checkpoint_artifact_path,
        )
    ):
        return
    print()
    if resume_checkpoint_source_path:
        print("Resumed from checkpoint:", resume_checkpoint_source_path)
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
    if resume_checkpoint_artifact_path:
        print("Resume checkpoint:        ", resume_checkpoint_artifact_path)
    if latest_resume_checkpoint_artifact_path:
        print("Latest resume checkpoint: ", latest_resume_checkpoint_artifact_path)
    if primitive_resume_checkpoint_artifact_path:
        print("Primitive checkpoint:     ", primitive_resume_checkpoint_artifact_path)
    if latest_primitive_resume_checkpoint_artifact_path:
        print("Latest primitive checkpoint:", latest_primitive_resume_checkpoint_artifact_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Case 3 LG-slippage bridge dry-run harness"
    )
    parser.add_argument("--model", default=DEFAULT_LIVE_MODEL, help="OpenAI model name")
    parser.add_argument(
        "--reasoning-mode",
        default="multi_turn",
        choices=("multi_turn",),
        help="Bridge reasoning mode to run in the dry-run harness",
    )
    parser.add_argument(
        "--focus",
        default="full",
        choices=("full", "primitive_generation"),
        help="Run the full harness or start directly from the known Case 3 primitive-generation outline",
    )
    parser.add_argument(
        "--stop-before-primitive-generation",
        action="store_true",
        help=(
            "Stop after outline completes so primitive_generation can be inspected. "
            "Only applies when --focus full."
        ),
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
        help="Print the first rendered multi-turn prompt",
    )
    parser.add_argument(
        "--resume-checkpoint",
        help=(
            "Resume from a saved multi_turn or primitive_generation checkpoint JSON. "
            "You can also pass the sibling prompt/response artifact path."
        ),
    )
    parser.add_argument(
        "--write-resume-checkpoints",
        action="store_true",
        help="Write resume checkpoint JSON artifacts alongside the prompt/response debug files",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    _configure_dryrun_logging()
    result = asyncio.run(
        run_case3_bridge_dryrun(
            write_debug=not args.no_debug,
            llm_model=args.model,
            reasoning_mode=args.reasoning_mode,
            stop_before_primitive_generation=args.stop_before_primitive_generation,
            focus=args.focus,
            resume_checkpoint=args.resume_checkpoint,
            write_resume_checkpoints=args.write_resume_checkpoints,
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

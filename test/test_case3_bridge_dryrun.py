"""Case 3 runtime failure-context dry-run for recovery bridge stages.

Run directly:
    python test/test_case3_bridge_dryrun.py --mode outline
    python test/test_case3_bridge_dryrun.py --mode primitive
    python test/test_case3_bridge_dryrun.py --mode safety
    python test/test_case3_bridge_dryrun.py --mode full

Each mode rebuilds the Case 3 runtime failure context from lg_slippage.json.
The archived Case 3 artifacts are only used by tests as mocked LLM responses.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
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

from cais_spade_llm.agents.central_controller.recovery_safety_generation import (
    generate_recovery_safety_bundle,
)  # noqa: E402
from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner  # noqa: E402
from cais_spade_llm.agents.intelligent_product.replanner.failure_context import (  # noqa: E402
    build_failure_event,
    failure_context_from_scenario_config,
    load_failure_scenario_config,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (  # noqa: E402
    multi_turn as multi_turn_mode,
)
from cais_spade_llm.resources.resource_primitives import get_resource_bridge_snapshot  # noqa: E402


CASE_ID = "case3_llm_bridge"
FAILED_TASK_ID = "REQ_2_T4"
ANCHOR_TASK_ID = "REQ_2_T3"
GOAL_STATE = "assembled"
CASE3_COMPLETED_TASK_IDS = (
    "REQ_1_T1",
    "REQ_1_T2",
    "REQ_2_T1",
    "REQ_2_T2",
    "REQ_2_T3",
)

DEBUG_ROOT = ROOT / "cais_spade_llm" / "monitor" / "debug"
CASE3_ARCHIVE = (
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
)
AUTO_CANDIDATE_COUNT_CAP = 8


def _load_local_env(path: Path) -> None:
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


def _env_default(*env_names: str, fallback: str) -> str:
    for env_name in env_names:
        token = str(os.environ.get(env_name) or "").strip()
        if token:
            return token
    return fallback


_load_local_env(ROOT / ".env")
DEFAULT_LIVE_MODEL = _env_default(
    "CASE3_RECOVERY_MODEL",
    "CAIS_SPADE_LLM_MODEL",
    "OPENAI_MODEL",
    fallback="gpt-5.4",
)
DEFAULT_REASONING_EFFORT = _env_default(
    "CAIS_SPADE_REASONING_EFFORT",
    "OPENAI_REASONING_EFFORT",
    fallback="medium",
)


def _normalize_reasoning_effort_for_model(model_name: str, effort: str) -> str:
    normalized_model = str(model_name or "").strip().lower()
    normalized_effort = str(effort or "").strip().lower()
    if normalized_model.startswith("gpt-5.4") and normalized_effort == "minimal":
        return "none"
    return normalized_effort


def _parse_structured_json_text(raw_text: str) -> Any:
    text = str(raw_text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        for start_idx, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                parsed, _end_idx = decoder.raw_decode(text[start_idx:])
            except json.JSONDecodeError:
                continue
            return parsed
        raise exc


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(path)


def _utc_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _debug_root(debug_root: Path | None = None) -> Path:
    root = debug_root or DEBUG_ROOT
    return root if root.is_absolute() else (ROOT / root)


def _case3_paths() -> dict[str, Path]:
    bundle_root = ROOT / "cais_spade_llm" / "user_verified_plan" / "bundles" / CASE_ID
    return {
        "bundle_manifest": bundle_root / "bundle_manifest.json",
        "tools": bundle_root / "catalog" / "tools.json",
        "plan": bundle_root / "plan" / "case3_two_arm_llm_bridge_plan.json",
        "requirements": bundle_root / "plan" / "case3_two_arm_llm_bridge_requirements.json",
        "safety_logic": bundle_root / "safety" / "cca_safety_logic.json",
        "geometry": (
            ROOT
            / "cais_spade_llm"
            / "specification"
            / "products"
            / "geometry"
            / "assembly_board-v1.json"
        ),
        "failure_scenario": (
            ROOT / "cais_spade_llm" / "initialization" / "failure_scenarios" / "lg_slippage.json"
        ),
        "recovery_outline_experiment_settings": (
            ROOT / "cais_spade_llm" / "initialization" / "recovery_outline_experiment_settings.json"
        ),
        "ur5e": ROOT / "cais_spade_llm" / "initialization" / "resources" / "robot_ur5e.json",
        "xarm6": ROOT / "cais_spade_llm" / "initialization" / "resources" / "robot_xarm6.json",
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


def _normalize_recovery_outline_experiment_settings(raw_settings: Any) -> dict[str, Any]:
    raw = dict(raw_settings or {}) if isinstance(raw_settings, dict) else {}
    enabled = bool(raw.get("enabled", True))
    recovery_selection_mode = str(raw.get("recovery_selection_mode") or "pure_llm").strip().lower()
    if recovery_selection_mode not in {"pure_llm", "neurosymbolic"}:
        recovery_selection_mode = "pure_llm"

    raw_action_horizon = raw.get("action_horizon", 1)
    action_horizon: int | str
    if isinstance(raw_action_horizon, str):
        horizon_token = raw_action_horizon.strip().lower()
        if horizon_token == "full":
            action_horizon = "full"
        else:
            try:
                action_horizon = max(1, int(horizon_token))
            except ValueError:
                action_horizon = 1
    else:
        try:
            action_horizon = max(1, int(raw_action_horizon))
        except (TypeError, ValueError):
            action_horizon = 1

    raw_candidate_count = raw.get("candidate_count", "auto")
    candidate_count: int | str
    if isinstance(raw_candidate_count, str):
        candidate_token = raw_candidate_count.strip().lower()
        if candidate_token in {"auto", "n"}:
            candidate_count = "auto"
        else:
            try:
                candidate_count = max(1, int(candidate_token))
            except ValueError:
                candidate_count = "auto"
    else:
        try:
            candidate_count = max(1, int(raw_candidate_count))
        except (TypeError, ValueError):
            candidate_count = "auto"

    return {
        "enabled": enabled,
        "recovery_selection_mode": recovery_selection_mode,
        "action_horizon": action_horizon,
        "candidate_count": candidate_count,
    }


def _bridge_action_horizon_fields(action_horizon: int | str) -> dict[str, Any]:
    if action_horizon == "full":
        return {
            "action_horizon": "full",
            "action_horizon_steps": "full",
            "action_horizon_k": 1,
        }
    steps = max(1, int(action_horizon))
    return {
        "action_horizon": "1" if steps == 1 else "k",
        "action_horizon_steps": steps,
        "action_horizon_k": steps,
    }


def _bridge_candidate_count_fields(candidate_count: int | str) -> dict[str, Any]:
    if candidate_count == "auto":
        return {
            "candidate_count": "auto",
            "candidate_bound": AUTO_CANDIDATE_COUNT_CAP,
            "candidate_bound_cap": AUTO_CANDIDATE_COUNT_CAP,
        }
    count = max(1, int(candidate_count))
    return {
        "candidate_count": count,
        "candidate_bound": count,
        "candidate_bound_cap": count,
    }


def _load_recovery_outline_experiment_settings(
    settings_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(settings_override, dict):
        return _normalize_recovery_outline_experiment_settings(settings_override)
    paths = _case3_paths()
    settings_path = paths["recovery_outline_experiment_settings"]
    if not settings_path.exists():
        return _normalize_recovery_outline_experiment_settings({})
    payload = _load_json(settings_path)
    return _normalize_recovery_outline_experiment_settings(payload)


def _load_robot_config(path: Path, key: str) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"robot config {path} did not decode to an object")
    config = payload.get(key)
    if not isinstance(config, dict):
        raise KeyError(f"robot config {path} is missing top-level key {key!r}")
    return config


class ProcessPlannerPrepareTrace(ProcessPlanner):
    """ProcessPlanner with the active bridge-session wiring."""


class FakeProductAgent:
    """Small product-agent surface used by the Case 3 dry-run harness."""

    def __init__(
        self,
        *,
        tools_catalog: list[dict[str, Any]],
        product_geometry: dict[str, Any],
        llm_model: str | None = None,
        llm_reasoning_effort: str | None = None,
        precomputed_bundle: dict[str, Any] | None = None,
        scripted_responses: list[dict[str, Any]] | None = None,
    ) -> None:
        self.jid = "assembly_board-v1@localhost"
        self.agent_name = "assembly_board-v1"
        self.instructions = "Case 3 runtime recovery dry-run product agent."
        self.logger = logging.getLogger("case3_bridge_dryrun")
        self.tools_catalog = deepcopy(tools_catalog)
        self.product_geometry = deepcopy(product_geometry)
        self.llm_model = str(llm_model or DEFAULT_LIVE_MODEL).strip()
        self.llm_reasoning_effort = _normalize_reasoning_effort_for_model(
            self.llm_model,
            str(llm_reasoning_effort or DEFAULT_REASONING_EFFORT).strip(),
        )
        self.precomputed_bundle = deepcopy(precomputed_bundle or {})
        self.structured_requirements_path = Path(
            str((self.precomputed_bundle.get("artifacts") or {}).get("requirements_json") or "")
        )
        self._bridge_reasoning_mode = "multi_turn"
        self.turn_log: list[dict[str, Any]] = []
        self._turn_index = 0
        self._scripted_responses = [deepcopy(row) for row in scripted_responses or []]

    def _geometry_for_part(self, part_name: str) -> dict[str, Any]:
        board = dict(self.product_geometry.get("assembly_board") or {})
        parts = dict(self.product_geometry.get("parts") or {})
        slot_xy = dict(board.get("slots") or {}).get(part_name)
        if slot_xy is None:
            return {}
        return {
            "slot_xy": slot_xy,
            "part_height_m": dict(parts.get("heights_m") or {}).get(part_name),
            "model_name": dict(parts.get("model_map") or {}).get(part_name),
            "slot_floor_z_m": board.get("slot_floor_z_m"),
            "board_center": deepcopy(board.get("center") or {}),
        }

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        if self._scripted_responses:
            parsed = deepcopy(self._scripted_responses.pop(0))
            self._record_turn(prompt, parsed)
            return parsed

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("openai package required for live bridge") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        client = OpenAI()

        def _call() -> dict[str, Any]:
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
            for _round_index in range(max_tool_rounds + 1):
                kwargs: dict[str, Any] = {
                    "model": self.llm_model,
                    "messages": messages,
                    "reasoning_effort": self.llm_reasoning_effort,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": response_format,
                    },
                }
                if tools:
                    kwargs["tools"] = tools
                response = client.chat.completions.create(**kwargs)
                choice = response.choices[0].message
                if getattr(choice, "tool_calls", None) and tool_executor:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": choice.content or "",
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                }
                                for tool_call in choice.tool_calls
                            ],
                        }
                    )
                    for tool_call in choice.tool_calls:
                        result = tool_executor(
                            tool_call.function.name,
                            json.loads(tool_call.function.arguments),
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result, default=str),
                            }
                        )
                    continue
                return _parse_structured_json_text(choice.content or "{}")
            raise RuntimeError("exceeded max tool rounds")

        parsed = await asyncio.to_thread(_call)
        self._record_turn(prompt, parsed)
        return parsed

    def _record_turn(self, prompt: str, response: dict[str, Any]) -> None:
        self._turn_index += 1
        self.turn_log.append(
            {
                "turn_index": self._turn_index,
                "model": self.llm_model,
                "prompt": prompt,
                "response": deepcopy(response),
            }
        )


class FakeBridgeRobot:
    """Small robot-agent surface used by the bridge dry-run."""

    _BRIDGE_PRIMITIVES = (
        "move_cartesian",
        "move_pose",
        "move_relative",
        "move_to_named_pose",
        "delay",
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
        self.logger = logging.getLogger(f"FakeBridgeRobot.{self.agent_name}")

    def set_shared_observations(self, observations: dict[str, dict[str, Any]] | None) -> None:
        self._shared_observations = deepcopy(observations or {})

    def _observation_catalog(self) -> dict[str, dict[str, Any]]:
        catalog = deepcopy(self._shared_observations)
        catalog.update(deepcopy(self._observations))
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
          x: {type: number}
          y: {type: number}
          z: {type: number}
          speed: {type: number}
        preconditions: {}
        effects:
          current_pose:
            pose_absolute_from_params: [x, y, z]
          current_pose_ref:
            set_unknown: true
        ---
        """
        del speed
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
          x: {type: number}
          y: {type: number}
          z: {type: number}
          qx: {type: number}
          qy: {type: number}
          qz: {type: number}
          qw: {type: number}
          speed: {type: number}
        preconditions: {}
        effects:
          current_pose:
            pose_absolute_from_params: [x, y, z]
          current_pose_ref:
            set_unknown: true
        ---
        """
        del qx, qy, qz, qw, speed
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
          dx: {type: number}
          dy: {type: number}
          dz: {type: number}
          speed: {type: number}
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
        del speed
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
        description: Move to a named joint configuration.
        params:
          pose_name: {type: string}
          speed: {type: number}
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
        del speed
        self._current_state = "idle"
        self._bridge_pose_ref = str(pose_name or "").strip() or None
        return {"success": True, "message": "fake move_to_named_pose ok"}

    def delay(self, duration_sec: float) -> dict[str, Any]:
        """
        ---
        description: Wait intentionally between robot task steps.
        params:
          duration_sec: {type: number}
        preconditions: {}
        effects: {}
        synthesis_hidden: true
        ---
        """
        return {"success": True, "message": f"fake delay ok {float(duration_sec):.3f}"}

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
          position: {type: number}
        preconditions: {}
        effects:
          gripper_state:
            set: closed
        ---
        """
        del position
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
          model_name: {type: string}
          link: {type: string}
          part_name: {type: string}
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
        del link
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
          model_name: {type: string}
          link: {type: string}
          assume_released_if_open: {type: boolean}
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
        del model_name, link, assume_released_if_open
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
        description: Close the gripper and attach the target part.
        params:
          model_name: {type: string}
          part_name: {type: string}
          position: {type: number}
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
        self.close_gripper(position=position)
        self.attach_part(model_name=model_name, part_name=part_name)
        self._current_state = "picked"
        return {"success": True, "message": "fake grasp_part ok"}

    def release_part(
        self,
        model_name: str = "",
        part_name: str = "",
        assume_released_if_open: bool = False,
    ) -> dict[str, Any]:
        """
        ---
        description: Open the gripper and detach the currently held part.
        params:
          model_name: {type: string}
          part_name: {type: string}
          assume_released_if_open: {type: boolean}
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
        del assume_released_if_open
        self.open_gripper()
        self.detach_part(model_name=model_name)
        self._current_state = "idle"
        return {"success": True, "message": f"fake release_part ok {part_name or model_name}"}

    def get_current_pose(self) -> dict[str, Any]:
        """
        ---
        description: Return the current end-effector pose.
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
        target_pose: dict[str, Any] | None = None,
        target_pose_source: str = "",
        approach_height_override_m: float | None = None,
        surface_clearance_override_m: float | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """
        ---
        description: Compute pick target positions from perception and geometry.
        params:
          part_name: {type: string}
          target_pose: {type: object}
          target_pose_source: {type: string}
          approach_height_override_m: {type: number}
          surface_clearance_override_m: {type: number}
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
        approach_height = float(approach_height_override_m or 0.2)
        return {
            "success": True,
            "part_name": str(part_name or target.get("part_name") or ""),
            "model_name": "fake_model",
            "approach_pose": {"x": pose["x"], "y": pose["y"], "z": pose["z"] + approach_height},
            "target_pose": {
                "x": pose["x"],
                "y": pose["y"],
                "z": pose["z"] + 0.02 + surface_clearance,
            },
            "target_pose_source": target_pose_source,
        }

    def compute_place_targets(
        self,
        pick_ctx: dict[str, Any] | None = None,
        part_name: str = "",
        z_adjustment_m: float = 0.0,
        destination_location: str = "",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """
        ---
        description: Compute placement target positions from context and geometry.
        params:
          pick_ctx: {type: object}
          part_name: {type: string}
          z_adjustment_m: {type: number}
          destination_location: {type: string}
        preconditions: {}
        effects: {}
        ---
        """
        del destination_location
        pick = dict(pick_ctx or {})
        x = float((pick.get("target_pose") or {}).get("x", 0.0) or 0.0)
        y = float((pick.get("target_pose") or {}).get("y", 0.0) or 0.0)
        z = 1.05 + float(z_adjustment_m or 0.0)
        return {
            "success": True,
            "part_name": str(part_name or pick.get("part_name") or ""),
            "model_name": "fake_model",
            "approach_pose": {"x": x, "y": y, "z": z + 0.1},
            "target_pose": {"x": x, "y": y, "z": z},
        }

    def detect_parts(self, part_name: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        """
        ---
        description: Detect parts via perception service.
        params:
          part_name: {type: string}
        preconditions: {}
        effects: {}
        ---
        """
        if not part_name:
            part_name = kwargs.get("filter_part_name")
        catalog = self._observation_catalog()
        if part_name:
            observation = deepcopy(catalog.get(str(part_name).strip()) or {})
            return [observation] if observation else []
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

    def _is_pose_in_workspace(self, pose: dict[str, Any]) -> tuple[bool, str]:
        bounds = self.static_capabilities.get("workspace_bounds")
        if not isinstance(bounds, dict):
            return True, "no workspace_bounds configured"
        violations: list[str] = []
        for axis in ("x", "y", "z"):
            if pose.get(axis) is None:
                continue
            value = float(pose[axis])
            low = bounds.get(f"{axis}_min_m")
            high = bounds.get(f"{axis}_max_m")
            if low is not None and value < float(low):
                violations.append(f"{axis}={value:.4f} < {axis}_min_m={float(low):.4f}")
            if high is not None and value > float(high):
                violations.append(f"{axis}={value:.4f} > {axis}_max_m={float(high):.4f}")
        if violations:
            return False, f"pose outside workspace: {', '.join(violations)}"
        return True, "pose within workspace bounds"

    def bridge_feasibility_oracle(
        self,
        *,
        part_context: dict[str, Any],
        bridge_snapshot: dict[str, Any],
        grounded_action: dict[str, Any] | None = None,
        part_name: str | None = None,
        **_compat_kwargs: Any,
    ) -> dict[str, Any]:
        target_info = dict(
            (grounded_action or {}).get("target") or part_context.get("target") or {}
        )
        target_pose = (
            target_info.get("slot_pose")
            or target_info.get("pose")
            or part_context.get("observed_pose")
            or part_context.get("pose")
        )
        evidence = {
            "part_context": deepcopy(part_context),
            "bridge_snapshot": deepcopy(bridge_snapshot),
            "resource_jid": self.jid,
            "grounded_action": deepcopy(grounded_action or {}),
        }
        if not isinstance(target_pose, dict):
            return {
                "allowed": True,
                "reason": "grounded preconditions are satisfied and no pose check is required",
                "evidence": evidence,
            }
        inside, reason = self._is_pose_in_workspace(target_pose)
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
            "evidence": {**evidence, "checked_pose": deepcopy(target_pose)},
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
                "message": f"unsupported observation primitive {primitive_name!r}",
                "snapshot": self.get_bridge_snapshot(),
            }
        part_name = str(payload.get("part_name") or "").strip()
        if not part_name:
            for alt_key in ("part_names", "targets"):
                alt = payload.get(alt_key)
                if isinstance(alt, list) and len(alt) == 1 and str(alt[0] or "").strip():
                    part_name = str(alt[0]).strip()
                    break
        catalog = self._observation_catalog()
        if not part_name and len(catalog) == 1:
            part_name = next(iter(catalog.keys()))
        observation = deepcopy(catalog.get(part_name) or {})
        if not observation:
            return {
                "success": False,
                "message": f"no observation configured for part {part_name!r}",
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


def _task_node_by_id(plan_payload: dict[str, Any], task_id: str) -> dict[str, Any]:
    for node in plan_payload.get("nodes") or []:
        if isinstance(node, dict) and str(node.get("id") or "").strip() == task_id:
            return deepcopy(node)
    return {}


def _origin_resource_location_for_part(plan_payload: dict[str, Any], part_name: str) -> str:
    for node in plan_payload.get("nodes") or []:
        if not isinstance(node, dict) or node.get("type") != "task":
            continue
        if str(node.get("function_name") or "").strip() != "pick_grasp":
            continue
        params = dict(node.get("params") or {})
        if str(params.get("part_name") or "").strip() == part_name:
            return str(params.get("origin_resource_location") or "").strip()
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
    for node in plan_payload.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if node_id not in completed:
            continue
        params = dict(node.get("params") or {})
        if str(params.get("part_name") or "").strip() != part_name:
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
            str(key).strip() for key in (derived_entry.get("_force_keys") or []) if str(key).strip()
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
    drop_pose = deepcopy(dict(scenario_config.get("injection") or {}).get("drop_pose") or {})
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
        if part_name and part_name not in seen:
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
    part_defaults = {
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
    base_part_tracker = {
        part_name: deepcopy(part_defaults[part_name])
        for part_name in _live_style_part_order(plan_payload)
        if part_name in part_defaults
    }
    derived_part_tracker = planner._derive_part_tracker_from_violations([failure_payload])
    part_tracker = _merge_part_tracker(base_part_tracker, derived_part_tracker)
    lg_entry = dict(part_tracker.get("LG") or {})
    lg_entry.update({"state": "misplaced", "location": None, "last_known_location": None})
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
    resource_states = {
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
        if payload.get(axis) is not None:
            pose[axis] = deepcopy(payload.get(axis))
    if any(axis not in pose for axis in ("x", "y", "z")):
        return None
    return {axis: float(pose[axis]) for axis in ("x", "y", "z")}


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
    robots_by_jid = {robot.jid: robot for robot in robots}
    explicit_observations_by_part: dict[str, dict[str, Any]] = {}
    for robot in robots:
        for part_name, observation in robot._observations.items():
            if part_name not in explicit_observations_by_part:
                explicit_observations_by_part[part_name] = deepcopy(observation)

    shared_catalog: dict[str, dict[str, Any]] = {}
    for part_row in llm_input.get("part_facts") or []:
        if not isinstance(part_row, dict):
            continue
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
                normalized_pose = _normalized_observation_pose(holder_robot._position)
        if normalized_pose is None:
            continue
        observation["part_name"] = part_name
        observation["x"] = normalized_pose["x"]
        observation["y"] = normalized_pose["y"]
        observation["z"] = normalized_pose["z"]
        observation["pose"] = deepcopy(normalized_pose)
        if part_row.get("current_location"):
            observation["current_location"] = deepcopy(part_row.get("current_location"))
        if holder_resource_jid:
            observation["current_holder_resource_jid"] = holder_resource_jid
        shared_catalog[part_name] = observation
    return shared_catalog


def _configure_live_bridge_session(
    prepared_bridge_request: dict[str, Any],
    *,
    experiment_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    settings = _load_recovery_outline_experiment_settings(experiment_settings)
    bridge_session["reasoning_mode"] = "multi_turn"
    bridge_session["max_turns"] = max(int(bridge_session.get("max_turns", 6) or 6), 1000)
    bridge_session["repair_mode"] = "recover"
    bridge_session["observation_backend"] = "mock_detect_parts_harness"
    bridge_session["outline_mode"] = "incremental_candidates_validated"
    if settings.get("enabled", True):
        bridge_session["recovery_selection_mode"] = settings["recovery_selection_mode"]
        bridge_session.update(_bridge_action_horizon_fields(settings["action_horizon"]))
        bridge_session.update(_bridge_candidate_count_fields(settings["candidate_count"]))
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["recovery_outline_experiment_settings"] = deepcopy(settings)
    return settings


async def _prepare_bridge_dryrun_harness(
    *,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], FakeProductAgent, ProcessPlanner, dict[str, Any]]:
    paths = _case3_paths()
    tools_catalog = _load_json(paths["tools"])
    plan_payload = _load_json(paths["plan"])
    geometry_payload = _load_json(paths["geometry"])
    ur5e_config = _load_robot_config(paths["ur5e"], "ur5e")
    xarm6_config = _load_robot_config(paths["xarm6"], "xarm6")
    if not isinstance(tools_catalog, list):
        raise TypeError("tools catalog did not decode to a list")
    if not isinstance(plan_payload, dict):
        raise TypeError("case3 plan did not decode to an object")
    if not isinstance(geometry_payload, dict):
        raise TypeError("case3 geometry did not decode to an object")

    product_agent = FakeProductAgent(
        tools_catalog=tools_catalog,
        product_geometry=deepcopy(geometry_payload.get("gazebo") or {}),
        precomputed_bundle=_case3_bundle_context(paths),
        scripted_responses=scripted_responses,
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
            "MCP": {
                "part_name": "MCP",
                "x": 0.0,
                "y": -0.08,
                "z": 1.025,
                "pose": {"x": 0.0, "y": -0.08, "z": 1.025},
            },
        },
    )
    xarm6 = FakeBridgeRobot(
        config=xarm6_config,
        execution_env="gazebo",
        current_state="failed",
        held_part=None,
        gripper_state="open",
        pose_ref=None,
        position={"x": 0.1, "y": 0.08, "z": 1.1994999760206477},
        observations={
            "LG": {
                "part_name": "LG",
                "x": 0.0,
                "y": 0.2,
                "z": 1.035,
                "pose": {"x": 0.0, "y": 0.2, "z": 1.035},
            },
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

    async def _direct_to_thread(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
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

    shared_observations = _build_shared_grounding_observation_catalog(
        prepared_bridge_request=prepared_bridge_request,
        robots=[ur5e, xarm6],
    )
    ur5e.set_shared_observations(shared_observations)
    xarm6.set_shared_observations(shared_observations)
    _configure_live_bridge_session(
        prepared_bridge_request,
        experiment_settings=experiment_settings,
    )
    prepared_bridge_request["multi_turn_session_seed"] = (
        multi_turn_mode.build_multi_turn_session_seed(prepared_bridge_request)
    )

    artifact_root = _debug_root(debug_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
    bridge_debug["artifact_directory"] = str(artifact_root)
    bridge_debug["per_turn_debug_dir"] = str(artifact_root)
    prepared_bridge_request["bridge_debug"] = bridge_debug
    return fixture, product_agent, planner, prepared_bridge_request


def _outline_trace_from_session_state(session_state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(session_state, dict):
        return []
    for rows in (
        session_state.get("accepted_outline_prefix"),
        session_state.get("transition_trace"),
        dict(session_state.get("final_output") or {}).get("transition_trace"),
    ):
        trace = [deepcopy(row) for row in (rows or []) if isinstance(row, dict)]
        if trace:
            return trace
    return []


def _latest_session(
    prepared_bridge_request: dict[str, Any], planner: ProcessPlanner
) -> dict[str, Any]:
    bridge_debug = (
        deepcopy(planner.get_last_bridge_debug())
        if callable(getattr(planner, "get_last_bridge_debug", None))
        else {}
    )
    for candidate in (
        bridge_debug.get("multi_turn_session"),
        dict(prepared_bridge_request.get("bridge_debug") or {}).get("multi_turn_session"),
        prepared_bridge_request.get("multi_turn_session_state"),
        prepared_bridge_request.get("multi_turn_session_seed"),
    ):
        if isinstance(candidate, dict):
            return deepcopy(candidate)
    return {}


def _primitive_program_from_session(session_state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(session_state, dict):
        return []
    for rows in (
        session_state.get("accepted_primitive_program"),
        dict(session_state.get("final_output") or {}).get("accepted_primitive_program"),
    ):
        program = [deepcopy(row) for row in (rows or []) if isinstance(row, dict)]
        if program:
            return program
    return []


async def _execute_bridge_until(
    planner: ProcessPlanner,
    prepared_bridge_request: dict[str, Any],
    *,
    stop_after: str,
) -> dict[str, Any] | None:
    target = str(stop_after or "").strip().lower()
    prepared_bridge_request["_stop_after_multi_turn_phase"] = (
        target if target in {"outline", "primitive"} else ""
    )
    session_state = dict(prepared_bridge_request.get("multi_turn_session_state") or {})
    if session_state:
        proposal = await multi_turn_mode.execute_multi_turn_bridge(
            planner,
            prepared_bridge_request,
            session_state=session_state,
        )
    else:
        proposal = await planner.execute_prepared_bridge_request(prepared_bridge_request)
    max_resume = max(
        10,
        int(
            dict(prepared_bridge_request.get("bridge_session") or {}).get("max_turns")
            or dict(prepared_bridge_request.get("multi_turn_session_seed") or {}).get("max_turns")
            or 0
        ),
    )
    for _resume_index in range(max_resume):
        session_state = _latest_session(prepared_bridge_request, planner)
        pause_status = str(session_state.get("status") or "").strip().lower()
        current_phase = str(session_state.get("current_phase") or "").strip().lower()
        if target == "outline":
            if _outline_trace_from_session_state(session_state) and current_phase in {
                "primitive_generation",
                "finalize",
            }:
                break
            resume_needed = pause_status == "paused_after_outline_turn"
        elif target == "primitive":
            if _primitive_ready_payload(prepared_bridge_request, session_state):
                break
            resume_needed = pause_status in {
                "paused_after_outline_turn",
                "ready_for_primitive_generation",
                "paused_after_primitive_turn",
            }
        else:
            resume_needed = pause_status in {
                "paused_after_outline_turn",
                "ready_for_primitive_generation",
                "paused_after_primitive_turn",
            }
        if not resume_needed:
            break
        proposal = await multi_turn_mode.execute_multi_turn_bridge(
            planner,
            prepared_bridge_request,
            session_state=session_state,
        )
    prepared_bridge_request["_stop_after_multi_turn_phase"] = ""
    return proposal


def _build_recovery_safety_payload(
    *,
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
    debug_root: Path | None = None,
) -> dict[str, Any]:
    def _nominal_candidate_tasks(
        *,
        pending_nominal_tasks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pending_by_id = {
            str(row.get("id") or row.get("task_id") or "").strip(): deepcopy(row)
            for row in pending_nominal_tasks
            if str(row.get("id") or row.get("task_id") or "").strip()
        }
        requirement_task_index = dict(prepared_bridge_request.get("requirement_task_index") or {})
        task_requirement_map = dict(prepared_bridge_request.get("task_requirement_map") or {})
        bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
        pending_task_rows_by_id: dict[str, dict[str, Any]] = {}
        active_requirement_ids: set[str] = set()

        for resource_entry in bridge_resources.values():
            if not isinstance(resource_entry, dict):
                continue
            for task in resource_entry.get("pending_tasks") or []:
                if not isinstance(task, dict):
                    continue
                task_id = str(task.get("id") or "").strip()
                if not task_id:
                    continue
                pending_task_rows_by_id[task_id] = deepcopy(task)
                requirement_id = str(
                    task.get("requirement_id") or task_requirement_map.get(task_id) or ""
                ).strip()
                if requirement_id:
                    active_requirement_ids.add(requirement_id)

        for task_id in pending_by_id:
            requirement_id = str(task_requirement_map.get(task_id) or "").strip()
            if requirement_id:
                active_requirement_ids.add(requirement_id)

        candidate_rows: list[dict[str, Any]] = []
        seen_task_ids: set[str] = set()
        for requirement_id in sorted(active_requirement_ids):
            for raw_task in requirement_task_index.get(requirement_id) or []:
                if not isinstance(raw_task, dict):
                    continue
                task_id = str(raw_task.get("task_id") or raw_task.get("id") or "").strip()
                if not task_id or task_id in seen_task_ids:
                    continue
                seen_task_ids.add(task_id)
                enriched = dict(pending_task_rows_by_id.get(task_id) or {})
                pending_row = dict(pending_by_id.get(task_id) or {})
                params = dict(enriched.get("params") or raw_task.get("params") or {})
                candidate_rows.append(
                    {
                        "id": task_id,
                        "function": str(
                            raw_task.get("function_name")
                            or enriched.get("function_name")
                            or pending_row.get("function")
                            or ""
                        ).strip(),
                        "resource": str(
                            raw_task.get("resource_jid")
                            or enriched.get("resource_jid")
                            or pending_row.get("resource")
                            or ""
                        ).strip(),
                        "part": str(
                            raw_task.get("part_name")
                            or params.get("part_name")
                            or pending_row.get("part")
                            or ""
                        ).strip(),
                        "status": str(
                            raw_task.get("status")
                            or enriched.get("status")
                            or pending_row.get("status")
                            or ""
                        ).strip(),
                        "in_state": str(
                            raw_task.get("in_state")
                            or enriched.get("in_state")
                            or pending_row.get("in_state")
                            or ""
                        ).strip(),
                        "out_state": str(
                            raw_task.get("out_state")
                            or enriched.get("out_state")
                            or pending_row.get("out_state")
                            or ""
                        ).strip(),
                        "requirement_id": requirement_id,
                        "sequence_index": int(raw_task.get("sequence_index") or 0),
                        "product_jid": str(params.get("product_jid") or "").strip(),
                        "destination_location": str(
                            params.get("destination_location") or ""
                        ).strip(),
                        "blocked_by_condition_ids": [
                            str(token).strip()
                            for token in (pending_row.get("blocked_by_condition_ids") or [])
                            if str(token).strip()
                        ],
                        "expected_start_state": deepcopy(
                            pending_row.get("expected_start_state") or {}
                        ),
                        "expected_end_state": deepcopy(pending_row.get("expected_end_state") or {}),
                        "projected_outline_state": deepcopy(
                            pending_row.get("projected_outline_state") or {}
                        ),
                    }
                )
        return candidate_rows

    accepted_outline_prefix = _outline_trace_from_session_state(session_state)
    if not accepted_outline_prefix:
        return {}
    projected_outline_state = deepcopy(
        accepted_outline_prefix[-1].get("projected_outline_state")
        or session_state.get("projected_outline_state")
        or {}
    )
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    modeled_continuation_gap = dict(llm_input.get("modeled_continuation_gap") or {})
    pending_nominal_tasks = [
        deepcopy(row)
        for row in (modeled_continuation_gap.get("pending_nominal_tasks") or [])
        if isinstance(row, dict)
    ]
    pending_nominal_task_ids = [
        str(task_id).strip()
        for task_id in (
            modeled_continuation_gap.get("pending_nominal_task_ids")
            or [row.get("task_id") or row.get("id") for row in pending_nominal_tasks]
        )
        if str(task_id).strip()
    ]
    nominal_candidate_tasks = _nominal_candidate_tasks(
        pending_nominal_tasks=pending_nominal_tasks,
    )
    recovery_safety_dir = _debug_root(debug_root) / "recovery_safety"
    return {
        "product_jid": "assembly_board-v1@localhost",
        "recovery_safety_scope_id": "dryrun_recovery_scope",
        "accepted_outline_prefix": accepted_outline_prefix,
        "projected_outline_state": projected_outline_state,
        "pending_nominal_tasks": pending_nominal_tasks,
        "pending_nominal_task_ids": pending_nominal_task_ids,
        "nominal_candidate_tasks": nominal_candidate_tasks,
        "nominal_candidate_task_ids": [
            str(row.get("id") or "").strip()
            for row in nominal_candidate_tasks
            if str(row.get("id") or "").strip()
        ],
        "loaded_safety_rules": [
            deepcopy(rule)
            for rule in (prepared_bridge_request.get("loaded_safety_rules") or [])
            if isinstance(rule, dict)
        ],
        "bridge_safety_context": deepcopy(
            prepared_bridge_request.get("bridge_safety_context") or {}
        ),
        "recovery_safety_dir": str(recovery_safety_dir),
        "recovery_plan_dir": str(recovery_safety_dir),
        "recovery_safery_dir": str(recovery_safety_dir),
        "tools_catalog": deepcopy(prepared_bridge_request.get("tools_catalog") or []),
    }


async def _run_safety_from_outline(
    *,
    product_agent: FakeProductAgent,
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
    debug_root: Path | None = None,
) -> dict[str, Any]:
    payload = _build_recovery_safety_payload(
        prepared_bridge_request=prepared_bridge_request,
        session_state=session_state,
        debug_root=debug_root,
    )
    if not payload:
        raise AssertionError("safety synthsis requires an accepted recovery outline")
    return await generate_recovery_safety_bundle(product_agent, payload)


def _primitive_ready_payload(
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
) -> dict[str, Any]:
    for candidate in (
        dict(session_state.get("final_output") or {}),
        dict(prepared_bridge_request.get("bridge_debug") or {}).get("final_output"),
    ):
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("final_output_stage") or "").strip() == "primitive_program_ready":
            return deepcopy(candidate)
    return {}


def _primitive_ready_source_path(session_state: dict[str, Any]) -> str:
    for turn in reversed(
        [row for row in (session_state.get("turns") or []) if isinstance(row, dict)]
    ):
        if str(turn.get("phase") or "").strip().lower() != "final_output":
            continue
        if str(turn.get("final_output_stage") or "").strip() != "primitive_program_ready":
            continue
        source = str(turn.get("response_artifact_path") or "").strip()
        if source:
            return source
    return ""


def _write_recovery_final_bundle(
    *,
    prepared_bridge_request: dict[str, Any],
    session_state: dict[str, Any],
    recovery_safety_generation: dict[str, Any],
    debug_root: Path | None = None,
) -> dict[str, str]:
    if not recovery_safety_generation.get("ok"):
        return {}
    primitive_payload = _primitive_ready_payload(prepared_bridge_request, session_state)
    if not primitive_payload:
        return {}
    recovery_safety_logic_json = str(
        recovery_safety_generation.get("recovery_safety_logic_json") or ""
    ).strip()
    if not recovery_safety_logic_json:
        return {}
    recovery_safety_logic_path = Path(recovery_safety_logic_json)
    if not recovery_safety_logic_path.exists():
        return {}

    recovery_final_dir = _debug_root(debug_root) / "recovery_final"
    recovery_final_dir.mkdir(parents=True, exist_ok=True)
    source_final_output_path = _primitive_ready_source_path(session_state)
    if source_final_output_path and Path(source_final_output_path).exists():
        source_path = Path(source_final_output_path)
        target_final_output_path = recovery_final_dir / source_path.name
        if source_path.resolve() != target_final_output_path.resolve():
            shutil.copy2(source_path, target_final_output_path)
    else:
        target_final_output_path = (
            recovery_final_dir / f"multi_turn_final_output_response_{_utc_token()}.txt"
        )
        _write_json(target_final_output_path, primitive_payload)

    target_logic_path = recovery_final_dir / "cca_safety_logic.json"
    shutil.copy2(recovery_safety_logic_path, target_logic_path)
    for source_dfa_path in sorted(recovery_safety_logic_path.parent.glob("*_dfa.dot")):
        shutil.copy2(source_dfa_path, recovery_final_dir / source_dfa_path.name)
    return {
        "recovery_safety_logic_json": str(target_logic_path.resolve()),
        "recovery_final_dir": str(recovery_final_dir),
        "recovery_final_output_path": str(target_final_output_path.resolve()),
    }


async def _run_actual_recovery(
    *,
    mode: str,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fixture, product_agent, planner, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
        debug_root=debug_root,
        scripted_responses=scripted_responses,
        experiment_settings=experiment_settings,
    )
    paths = _case3_paths()
    proposal: dict[str, Any] | None = None
    recovery_safety_generation: dict[str, Any] = {}
    recovery_final: dict[str, str] = {}

    proposal = await _execute_bridge_until(
        planner,
        prepared_bridge_request,
        stop_after="outline",
    )
    outline_session = _latest_session(prepared_bridge_request, planner)
    outline_trace = _outline_trace_from_session_state(outline_session)
    _validate_outline_trace(outline_trace, label="transition_trace")

    primitive_session: dict[str, Any] = {}
    primitive_program: list[dict[str, Any]] = []
    if mode in {"primitive", "full"}:
        proposal = await _execute_bridge_until(
            planner,
            prepared_bridge_request,
            stop_after="primitive",
        )
        primitive_session = _latest_session(prepared_bridge_request, planner)
        primitive_program = _primitive_program_from_session(primitive_session)
        _validate_primitive_program(
            primitive_program,
            outline_ids=_outline_ids(outline_trace),
        )

    if mode in {"safety", "full"}:
        recovery_safety_generation = await _run_safety_from_outline(
            product_agent=product_agent,
            prepared_bridge_request=prepared_bridge_request,
            session_state=outline_session,
            debug_root=debug_root,
        )
        _validate_safety_result(
            recovery_safety_generation,
            _outline_trace_from_safety(recovery_safety_generation),
        )

    if mode == "full":
        recovery_final = _write_recovery_final_bundle(
            prepared_bridge_request=prepared_bridge_request,
            session_state=primitive_session,
            recovery_safety_generation=recovery_safety_generation,
            debug_root=debug_root,
        )

    bridge_debug = (
        deepcopy(planner.get_last_bridge_debug())
        if callable(getattr(planner, "get_last_bridge_debug", None))
        else {}
    )
    final_session = primitive_session or outline_session
    return {
        "mode": mode,
        "scenario": "case3_lg_slippage",
        "failure_scenario_source": str(paths["failure_scenario"]),
        "failure_context": deepcopy(fixture.get("failure_context") or {}),
        "proposal": deepcopy(proposal),
        "prepared_bridge_request": prepared_bridge_request,
        "context_summary": deepcopy(prepared_bridge_request.get("context_summary") or {}),
        "llm_input": deepcopy(prepared_bridge_request.get("llm_input") or {}),
        "bridge_debug": bridge_debug,
        "experiment_settings": deepcopy(
            prepared_bridge_request.get("recovery_outline_experiment_settings") or {}
        ),
        "multi_turn_session": deepcopy(final_session),
        "turns": [
            deepcopy(row) for row in (final_session.get("turns") or []) if isinstance(row, dict)
        ],
        "turn_log": deepcopy(product_agent.turn_log),
        "transition_trace": outline_trace,
        "accepted_primitive_program": primitive_program,
        "recovery_safety_generation": recovery_safety_generation,
        "recovery_final": recovery_final,
    }


def run_recovery_outline_only(
    *,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Case 3 runtime failure context through accepted recovery outline."""
    return asyncio.run(
        _run_actual_recovery(
            mode="outline",
            debug_root=debug_root,
            scripted_responses=scripted_responses,
            experiment_settings=experiment_settings,
        )
    )


def run_primitive_composition(
    *,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Case 3 outline first, then actual primitive composition."""
    return asyncio.run(
        _run_actual_recovery(
            mode="primitive",
            debug_root=debug_root,
            scripted_responses=scripted_responses,
            experiment_settings=experiment_settings,
        )
    )


def run_safety_synthsis(
    *,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Case 3 outline first, then actual recovery safety synthsis."""
    return asyncio.run(
        _run_actual_recovery(
            mode="safety",
            debug_root=debug_root,
            scripted_responses=scripted_responses,
            experiment_settings=experiment_settings,
        )
    )


def run_full(
    *,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Case 3 outline, safety synthsis, primitive composition, and final bundle."""
    return asyncio.run(
        _run_actual_recovery(
            mode="full",
            debug_root=debug_root,
            scripted_responses=scripted_responses,
            experiment_settings=experiment_settings,
        )
    )


def _outline_ids(trace: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("outline_id") or "").strip() for row in trace]


def _outline_trace_from_safety(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        deepcopy(row)
        for row in (payload.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]


def _validate_outline_trace(trace: list[dict[str, Any]], *, label: str) -> None:
    if not trace:
        raise AssertionError(f"{label} is empty")
    seen_outline_ids: set[str] = set()
    for index, row in enumerate(trace):
        if not isinstance(row, dict):
            raise AssertionError(f"{label}[{index}] is not an object")
        for key in ("outline_id", "event_name", "resource_jid"):
            if not str(row.get(key) or "").strip():
                raise AssertionError(f"{label}[{index}] is missing {key}")
        for key in ("expected_start_state", "expected_end_state"):
            value = row.get(key)
            if value is not None and not isinstance(value, dict):
                raise AssertionError(f"{label}[{index}].{key} is not an object")
        outline_id = str(row.get("outline_id") or "").strip()
        if outline_id in seen_outline_ids:
            raise AssertionError(f"{label} has duplicate outline_id {outline_id!r}")
        seen_outline_ids.add(outline_id)


def _validate_primitive_program(
    primitive_program: list[dict[str, Any]],
    *,
    outline_ids: list[str],
) -> None:
    if not primitive_program:
        raise AssertionError("accepted_primitive_program is empty")
    known_outline_ids = set(outline_ids)
    primitive_outline_ids: set[str] = set()
    for index, row in enumerate(primitive_program):
        if not isinstance(row, dict):
            raise AssertionError(f"accepted_primitive_program[{index}] is not an object")
        outline_id = str(row.get("outline_id") or "").strip()
        if not outline_id:
            raise AssertionError(f"accepted_primitive_program[{index}] is missing outline_id")
        if known_outline_ids and outline_id not in known_outline_ids:
            raise AssertionError(
                f"accepted_primitive_program[{index}] references unknown outline_id {outline_id!r}"
            )
        primitive_outline_ids.add(outline_id)
        primitive_steps = row.get("primitive_steps")
        if not isinstance(primitive_steps, list) or not primitive_steps:
            raise AssertionError(f"accepted_primitive_program[{index}] has no primitive_steps")
    missing_outline_ids = [
        outline_id for outline_id in outline_ids if outline_id not in primitive_outline_ids
    ]
    if missing_outline_ids:
        raise AssertionError(
            f"accepted_primitive_program is missing produced outline_id(s): {missing_outline_ids}"
        )


def _validate_safety_result(
    payload: dict[str, Any],
    accepted_outline_prefix: list[dict[str, Any]],
) -> None:
    _validate_outline_trace(accepted_outline_prefix, label="accepted_outline_prefix")
    if "ok" not in payload:
        raise AssertionError("safety result is missing ok")
    if not str(payload.get("recovery_safety_status") or "").strip():
        raise AssertionError("safety result is missing recovery_safety_status")
    rule_ids = [
        str(rule_id).strip() for rule_id in (payload.get("rule_ids") or []) if str(rule_id).strip()
    ]
    if not rule_ids:
        raise AssertionError("safety result has no rule_ids")


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)


def _emit(line: str = "") -> None:
    sys.stdout.write(f"{line}\n")


def _print_result(result: dict[str, Any]) -> None:
    _emit(f"Case 3 actual recovery dry-run mode: {result.get('mode') or '-'}")
    _emit(f"Failure scenario source: {result.get('failure_scenario_source') or '-'}")
    _print_experiment_settings(result.get("experiment_settings") or {})
    _emit("Failure context payload:")
    _emit(_compact_json(result.get("failure_context") or {}))
    _print_fault_event(result.get("context_summary") or {})
    _print_turns(result.get("turns") or [])
    _print_artifact_paths(result)
    _emit()
    _print_outline(result.get("transition_trace") or [])
    if result.get("accepted_primitive_program"):
        _emit()
        _print_primitives(result.get("accepted_primitive_program") or [])
    if result.get("recovery_safety_generation"):
        _emit()
        _print_safety(result.get("recovery_safety_generation") or {})
    if result.get("recovery_final"):
        _emit()
        _emit("Recovery final")
        for key, value in (result.get("recovery_final") or {}).items():
            _emit(f"  {key}: {value}")


def _print_experiment_settings(settings: dict[str, Any]) -> None:
    if not settings:
        return
    _emit("Recovery outline experiment settings:")
    for key in (
        "enabled",
        "recovery_selection_mode",
        "action_horizon",
        "candidate_count",
    ):
        _emit(f"  {key}: {settings.get(key)}")


def _print_fault_event(context_summary: dict[str, Any]) -> None:
    fault_event = dict(context_summary.get("fault_event") or {})
    _emit()
    _emit("Fault event used in prompt")
    _emit(f"  focused_resource_jid: {fault_event.get('focused_resource_jid') or '-'}")
    _emit(f"  blocked_at_task_id: {fault_event.get('blocked_at_task_id') or '-'}")
    _emit(f"  blocked_at_function: {fault_event.get('blocked_at_function') or '-'}")
    _emit(f"  resource_state: {fault_event.get('resource_state') or '-'}")


def _print_turns(turns: list[dict[str, Any]]) -> None:
    _emit()
    _emit("Multi-turn trace")
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        _emit(
            "  turn={turn} phase={phase} decision={decision}".format(
                turn=turn.get("turn_index") or "-",
                phase=turn.get("phase") or "-",
                decision=turn.get("decision") or turn.get("final_output_stage") or "-",
            )
        )
        prompt_path = str(turn.get("prompt_artifact_path") or "").strip()
        response_path = str(turn.get("response_artifact_path") or "").strip()
        if prompt_path:
            _emit(f"    prompt: {prompt_path}")
        if response_path:
            _emit(f"    response: {response_path}")


def _print_artifact_paths(result: dict[str, Any]) -> None:
    safety = dict(result.get("recovery_safety_generation") or {})
    final = dict(result.get("recovery_final") or {})
    paths = [
        safety.get("snapshot_artifact_path"),
        safety.get("grounding_prompt_artifact_path"),
        safety.get("grounding_llm_response_artifact_path"),
        safety.get("grounding_response_artifact_path"),
        safety.get("recovery_safety_logic_json"),
        final.get("recovery_final_dir"),
        final.get("recovery_final_output_path"),
    ]
    paths = [str(path).strip() for path in paths if str(path or "").strip()]
    if not paths:
        return
    _emit()
    _emit("Stage artifact paths")
    for path in paths:
        _emit(f"  {path}")


def _print_outline(trace: list[dict[str, Any]]) -> None:
    _emit("Recovery outline")
    for row in trace:
        outline_id = str(row.get("outline_id") or "-")
        event_name = str(row.get("event_name") or "-")
        resource_jid = str(row.get("resource_jid") or "-")
        part_name = str(row.get("part_name") or "-")
        _emit(f"  {outline_id}: {event_name}")
        _emit(f"    resource_jid: {resource_jid}")
        _emit(f"    part_name: {part_name}")
        _emit(f"    expected_start_state: {_compact_json(row.get('expected_start_state') or {})}")
        _emit(f"    expected_end_state:   {_compact_json(row.get('expected_end_state') or {})}")


def _print_primitives(primitive_program: list[dict[str, Any]]) -> None:
    _emit("Primitive composition")
    for row in primitive_program:
        outline_id = str(row.get("outline_id") or "-")
        event_name = str(row.get("event_name") or "-")
        primitive_steps = [
            step for step in (row.get("primitive_steps") or []) if isinstance(step, dict)
        ]
        _emit(f"  {outline_id}: {event_name} ({len(primitive_steps)} step(s))")
        for index, step in enumerate(primitive_steps, start=1):
            primitive = str(step.get("primitive") or "-")
            params = step.get("params") if isinstance(step.get("params"), dict) else {}
            _emit(f"    {index}. {primitive} {_compact_json(params)}")


def _print_safety(result: dict[str, Any]) -> None:
    _emit("Safety synthsis")
    _emit(f"  ok: {result.get('ok')}")
    _emit(f"  recovery_safety_status: {result.get('recovery_safety_status') or '-'}")
    _emit(f"  recovery_safety_scope_id: {result.get('recovery_safety_scope_id') or '-'}")
    _emit(f"  recovery_safety_logic_json: {result.get('recovery_safety_logic_json') or '-'}")
    _emit(f"  rule_ids: {result.get('rule_ids') or []}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case 3 actual runtime recovery dry-run debugger")
    parser.add_argument(
        "--mode",
        choices=("outline", "primitive", "safety", "full"),
        default="outline",
        help="Recovery stage to run",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parse_args()
    run_by_mode = {
        "outline": run_recovery_outline_only,
        "primitive": run_primitive_composition,
        "safety": run_safety_synthsis,
        "full": run_full,
    }
    result = run_by_mode[args.mode]()
    _print_result(result)
    return 0


def _load_archive_json(relative_path: str) -> dict[str, Any]:
    payload = _load_json(CASE3_ARCHIVE / relative_path)
    if not isinstance(payload, dict):
        raise TypeError(f"archive response {relative_path} did not decode to an object")
    return payload


def _without_description_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_description_fields(item)
            for key, item in value.items()
            if key != "description"
        }
    if isinstance(value, list):
        return [_without_description_fields(item) for item in value]
    return deepcopy(value)


def _archive_outline_responses() -> list[dict[str, Any]]:
    return [
        _without_description_fields(payload)
        for payload in (
            _load_archive_json(
                "recovery_outline/multi_turn_turn01_grounding_response_20260426T202431.txt"
            ),
            _load_archive_json(
                "recovery_outline/multi_turn_turn02_grounding_response_20260426T202443.txt"
            ),
            _load_archive_json(
                "recovery_outline/multi_turn_turn03_outline_response_20260426T202502.txt"
            ),
            _load_archive_json(
                "recovery_outline/multi_turn_turn04_outline_response_20260426T202524.txt"
            ),
            _load_archive_json(
                "recovery_outline/multi_turn_turn05_outline_response_20260426T202541.txt"
            ),
            _load_archive_json(
                "recovery_outline/multi_turn_turn06_outline_response_20260426T202556.txt"
            ),
        )
    ]


def _candidate_event(event_name: str, *, outline_id: str | None = None) -> dict[str, Any]:
    return {
        "outline_id": outline_id or event_name,
        "event_name": event_name,
        "resource_jid": "xarm6@localhost",
        "part_name": "LG",
        "expected_start_state": {"resource_state": "idle", "held_part": None},
        "expected_end_state": {"resource_state": "picked", "held_part": "LG"},
        "rationale": f"{event_name} rationale",
    }


def _candidate_session(
    *,
    recovery_selection_mode: str,
    action_horizon: str,
    candidate_count: int | str = "auto",
) -> dict[str, Any]:
    action_horizon_setting: int | str = 3 if action_horizon == "k" else action_horizon
    session = {
        "accepted_outline_prefix": [],
        "candidate_rejection_feedback": [],
        "candidate_prune_history": {},
        "recovery_selection_mode": recovery_selection_mode,
        "outline_stagnation_count": 0,
        "outline_validation_findings": [],
        "pruned_actions": [],
        "symbolic_resources": {},
        "symbolic_parts": {},
    }
    session.update(_bridge_action_horizon_fields(action_horizon_setting))
    session.update(_bridge_candidate_count_fields(candidate_count))
    return session


async def _run_mocked_candidate_handler(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
        multi_turn_outline_generation,
    )

    def _mock_progress_score(
        *,
        task: dict[str, Any],
        session_state: dict[str, Any],
        prepared_bridge_request: dict[str, Any],
    ) -> tuple[int, dict[str, int]]:
        del session_state, prepared_bridge_request
        scores = {
            "low_score": 1,
            "better_score": 5,
            "step_a": 2,
            "step_b": 2,
        }
        score = scores.get(str(task.get("event_name") or ""), 1)
        return score, {"mock_score": score}

    def _mock_remaining_counts(
        *,
        session_state: dict[str, Any],
        prepared_bridge_request: dict[str, Any],
    ) -> tuple[int, int]:
        del prepared_bridge_request
        accepted_count = len(session_state.get("accepted_outline_prefix") or [])
        return (0, 0) if accepted_count >= 2 else (1, 0)

    with (
        patch.object(multi_turn_mode, "_active_pruned_actions", return_value=[]),
        patch.object(
            multi_turn_mode,
            "_derive_candidate_outline_task",
            side_effect=lambda candidate_task, **_kwargs: (deepcopy(candidate_task), []),
        ),
        patch.object(multi_turn_mode, "_matching_active_pruned_action", return_value=None),
        patch.object(
            multi_turn_mode,
            "_validate_single_outline_task",
            side_effect=lambda **_kwargs: ([], {}),
        ),
        patch.object(
            multi_turn_mode,
            "_candidate_progress_score",
            side_effect=_mock_progress_score,
        ),
        patch.object(
            multi_turn_mode,
            "_remaining_blocked_issue_counts",
            side_effect=_mock_remaining_counts,
        ),
        patch.object(multi_turn_mode, "_apply_task_effects_to_symbolic_state", return_value=None),
        patch.object(multi_turn_mode, "_promote_durable_candidate_rejections", return_value=None),
    ):
        return await multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response=parsed_response,
            prepared_bridge_request={},
            planner=object(),
        )


def test_case3_actual_outline_uses_runtime_failure_context(tmp_path: Path) -> None:
    result = run_recovery_outline_only(
        debug_root=tmp_path,
        scripted_responses=_archive_outline_responses(),
    )
    trace = result["transition_trace"]
    failure_context = result["prepared_bridge_request"]["failure_context_raw"]

    assert trace
    assert failure_context["failed_task_id"] == FAILED_TASK_ID
    assert failure_context["failed_function_name"] == "place_insert"
    assert (tmp_path / "recovery_outline").is_dir()
    assert list((tmp_path / "recovery_outline").glob("multi_turn_turn*_prompt_*.txt"))


def test_case3_actual_outline_validation_is_structural() -> None:
    trace = [
        {
            "outline_id": "A",
            "event_name": "event_a",
            "resource_jid": "resource@localhost",
            "expected_start_state": {},
            "expected_end_state": {},
        }
    ]
    _validate_outline_trace(trace, label="transition_trace")


def test_case3_experiment_settings_are_loaded_from_file() -> None:
    settings = _load_recovery_outline_experiment_settings()

    assert settings["recovery_selection_mode"] == "pure_llm"
    assert settings["action_horizon"] == 1
    assert settings["candidate_count"] == "auto"


def test_case3_neurosymbolic_selects_best_valid_one_step_candidate() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="neurosymbolic",
        action_horizon="1",
    )
    decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "propose alternatives",
                "selected_candidate_index": 0,
                "candidate_events": [
                    _candidate_event("low_score"),
                    _candidate_event("better_score"),
                ],
            },
        )
    )

    assert decision == "need_next_task"
    assert turn_entry["selected_by"] == "neurosymbolic"
    assert turn_entry["selected_candidate_index"] == 1
    assert session_state["accepted_outline_prefix"][0]["event_name"] == "better_score"


def test_case3_pure_llm_keeps_llm_selected_one_step_candidate() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="1",
    )
    _decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "select my preferred candidate",
                "selected_candidate_index": 0,
                "candidate_events": [
                    _candidate_event("low_score"),
                    _candidate_event("better_score"),
                ],
            },
        )
    )

    assert turn_entry["selected_by"] == "pure_llm"
    assert turn_entry["selected_candidate_index"] == 0
    assert session_state["accepted_outline_prefix"][0]["event_name"] == "low_score"


def test_case3_neurosymbolic_k_horizon_commits_selected_sequence() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="neurosymbolic",
        action_horizon="k",
    )
    _decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "propose short traces",
                "candidate_traces": [
                    {"events": [_candidate_event("low_score")]},
                    {
                        "events": [
                            _candidate_event("step_a"),
                            _candidate_event("step_b"),
                        ]
                    },
                ],
            },
        )
    )

    assert turn_entry["selected_by"] == "neurosymbolic"
    assert turn_entry["selected_candidate_index"] == 1
    assert len(turn_entry["selected_transition_sequence"]) == 2
    assert [row["event_name"] for row in session_state["accepted_outline_prefix"]] == [
        "step_a",
        "step_b",
    ]


def test_case3_neurosymbolic_full_horizon_requires_complete_trace() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="neurosymbolic",
        action_horizon="full",
    )
    decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "propose full traces",
                "candidate_traces": [
                    {"events": [_candidate_event("better_score")]},
                    {
                        "events": [
                            _candidate_event("step_a"),
                            _candidate_event("step_b"),
                        ]
                    },
                ],
            },
        )
    )

    assert decision == "outline_ready"
    assert turn_entry["selected_candidate_index"] == 1
    assert len(session_state["accepted_outline_prefix"]) == 2


def test_case3_candidate_count_integer_rejects_wrong_count() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="neurosymbolic",
        action_horizon="1",
        candidate_count=1,
    )
    decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "too many alternatives",
                "candidate_events": [
                    _candidate_event("low_score"),
                    _candidate_event("better_score"),
                ],
            },
        )
    )

    assert decision == "need_revision"
    assert "exactly 1 candidate" in turn_entry["error"]


if __name__ == "__main__":
    raise SystemExit(main())

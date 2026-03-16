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

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bootstrap_repo_site_packages(root: Path) -> None:
    """Allow plain `python` to reuse packages installed in the repo .venv."""
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
    """Load .env values without requiring python-dotenv."""
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

from cais_spade_llm.agents.central_controller.central_controller_agent import (
    CentralControllerAgent,
)
from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_adapters import (
    bridge_adapter_capabilities,
    canonical_bridge_resource,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_generation import (
    normalize_bridge_turn_response,
)
from cais_spade_llm.agents.intelligent_product.replanner.preprogrammed_bridge_scenarios import (
    build_preprogrammed_bridge_proposal,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
    build_primitive_catalog,
    get_resource_bridge_snapshot,
    preview_step_output,
    resolve_context_ref,
    resolve_param_refs,
    sync_agent_from_bridge_snapshot,
)
from cais_spade_llm.resources.resource_profile import resource_snapshot_set_field
from cais_spade_llm.prompts import build_bridge_turn_prompt


CASE_ID = "case3_llm_bridge"
MAIN_V1_VARIANT = "main-v1"
MAIN_V2_VARIANT = "main-v2"
BOTH_REACHABLE_VARIANT = "both-reachable"
MCP_SWAP_VARIANT = "mcp-swap"
NO_GRIPPER_CONFLICT_VARIANT = "no-gripper-conflict"
SCENARIO_ID = "recover_lg_v1"
MAIN_V2_SCENARIO_ID = "recover_lg_main_v2_mirror"
BOTH_REACHABLE_SCENARIO_ID = "recover_lg_both_reachable"
MCP_SWAP_SCENARIO_ID = "recover_mcp_swap"
NO_GRIPPER_CONFLICT_SCENARIO_ID = "recover_lg_no_gripper_conflict"
FAILED_TASK_ID = "REQ_2_T4"
ANCHOR_TASK_ID = "REQ_2_T3"
GOAL_STATE = "assembled"
LG_DROP_POSE = {"x": 0.002, "y": 0.198, "z": 1.034}
DEBUG_DIR = Path("cais_spade_llm/monitor/debug")
DEFAULT_LIVE_MODEL = os.environ.get("CASE3_RECOVERY_MODEL", "gpt-5")
BRIDGE_PRIMITIVES = frozenset(
    {
        "move_cartesian",
        "move_pose",
        "move_relative",
        "move_to_named_pose",
        "rotate_wrist",
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


def _repo_root() -> Path:
    return ROOT


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


MUTEX_RULE_FAMILIES = frozenset(
    {
        "mutual_exclusion_zone",
        "no_simultaneous_presence_in_destination_area",
    }
)


def _load_runtime_safety_rules() -> list[dict[str, Any]]:
    payload = _load_json(_repo_root() / "cais_spade_llm" / "safety" / "cca_safety_logic.json")
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
    normalized_destination = str(destination or "").strip()
    allowed_families = {
        str(item or "").strip().lower()
        for item in (constraint_families or MUTEX_RULE_FAMILIES)
        if str(item or "").strip()
    }
    normalized_resources = {
        str(resource or "").strip().lower()
        for resource in required_resources
        if str(resource or "").strip()
    }
    for rule in _load_runtime_safety_rules():
        constraint_type = str(rule.get("constraint_type", "") or "").strip().lower()
        context = dict(rule.get("context") or {})
        destination_value = str(context.get("destination", "") or "").strip()
        rule_resources = {
            str(resource or "").strip().lower()
            for resource in (rule.get("resources") or [])
            if str(resource or "").strip()
        }
        if (
            constraint_type in allowed_families
            and destination_value == normalized_destination
            and normalized_resources.issubset(rule_resources)
        ):
            return deepcopy(rule)
    raise LookupError(
        f"Could not find runtime safety rule for destination={normalized_destination!r} "
        f"resources={sorted(normalized_resources)!r}"
    )


def _synthetic_mutex_rule(
    *,
    rule_id: str,
    destination: str,
    resources: list[str],
    constraint_type: str = "mutual_exclusion_zone",
) -> dict[str, Any]:
    destination_token = str(destination or "").strip()
    resource_tokens = [str(resource or "").strip() for resource in resources if str(resource or "").strip()]
    return {
        "id": str(rule_id or "").strip(),
        "constraint_type": str(constraint_type or "").strip(),
        "raw_text": (
            f"{' and '.join(resource_tokens)} must not both occupy {destination_token} at the same time"
        ),
        "generated_interpretation": (
            f"Robots {', '.join(resource_tokens)} are mutually exclusive in {destination_token}."
        ),
        "resources": resource_tokens,
        "context": {"destination": destination_token},
    }


CASE3_BOARD_MUTEX_RULE = _find_runtime_safety_rule(
    destination="assembly_board-v1",
    required_resources={"ur5e", "xarm6"},
)
CASE3_BOARD_MUTEX_RULE_ID = str(CASE3_BOARD_MUTEX_RULE.get("id", "") or "").strip()


def _case3_paths() -> dict[str, Path]:
    root = _repo_root()
    bundle_root = root / "cais_spade_llm" / "user_verified_plan" / "bundles" / CASE_ID
    return {
        "tools": bundle_root / "catalog" / "tools.json",
        "plan": bundle_root / "plan" / "twopart_assembly_llm_bridge_plan.json",
        "geometry": root / "cais_spade_llm" / "specification" / "products" / "geometry" / "assembly_board-v1.json",
        "ur5e": root / "cais_spade_llm" / "initialization" / "resources" / "robot_ur5e.json",
        "xarm6": root / "cais_spade_llm" / "initialization" / "resources" / "robot_xarm6.json",
    }


class FakeProductAgent:
    def __init__(
        self,
        *,
        tools_catalog: list[dict[str, Any]],
        product_geometry: dict[str, Any],
        scripted_turns: list[dict[str, Any]] | None,
        llm_mode: str = "scripted",
        llm_model: str | None = None,
        scenario_id: str = SCENARIO_ID,
        scenario_variant: str = MAIN_V1_VARIANT,
        scripted_final_plan_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.jid = "assembly_board-v1@localhost"
        self.logger = logging.getLogger("case3_recovery_main")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.tools_catalog = deepcopy(tools_catalog)
        self.product_geometry = deepcopy(product_geometry)
        self.plan_path = Path("/tmp/case3_recovery_main_plan.json")
        self.global_fsa_path = Path("/tmp/case3_recovery_main_global_fsa.json")
        self.llm_mode = str(llm_mode or "scripted").strip().lower()
        self.llm_model = str(llm_model or DEFAULT_LIVE_MODEL).strip()
        self.scenario_id = str(scenario_id or SCENARIO_ID).strip() or SCENARIO_ID
        self.scenario_variant = str(scenario_variant or MAIN_V1_VARIANT).strip() or MAIN_V1_VARIANT
        self._scripted_final_plan_builder = scripted_final_plan_builder
        self._scripted_turns = [deepcopy(turn) for turn in (scripted_turns or [])]
        self._turn_index = 0
        self.prepared_bridge_request: dict[str, Any] | None = None
        self.turn_log: list[dict[str, Any]] = []

    def _geometry_for_part(self, part_name: str) -> dict[str, Any]:
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
    ) -> dict[str, Any] | str:
        del with_functions, temperature
        if self.llm_mode == "live":
            return await self._ask_llm_live(prompt)

        if self._turn_index >= len(self._scripted_turns):
            raise RuntimeError("FakeProductAgent received more bridge turns than scripted")
        turn = deepcopy(self._scripted_turns[self._turn_index])
        self._turn_index += 1
        if turn.get("type") == "final_plan" and not turn.get("plan"):
            if not isinstance(self.prepared_bridge_request, dict):
                raise RuntimeError("prepared_bridge_request must be set before final_plan turn")
            if callable(self._scripted_final_plan_builder):
                turn["plan"] = self._scripted_final_plan_builder(self.prepared_bridge_request)
        self.turn_log.append(
            {
                "turn_index": self._turn_index,
                "mode": self.llm_mode,
                "prompt": prompt,
                "response": deepcopy(turn),
            }
        )
        return turn

    async def _ask_llm_live(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - depends on local install
            raise RuntimeError(
                "openai package is required for --live mode"
            ) from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set; cannot run --live mode")

        client = OpenAI()

        def _call() -> str:
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="low",
            )
            return (response.choices[0].message.content or "").strip()

        raw = await asyncio.to_thread(_call)
        self._turn_index += 1
        self.turn_log.append(
            {
                "turn_index": self._turn_index,
                "mode": self.llm_mode,
                "model": self.llm_model,
                "prompt": prompt,
                "response": raw,
            }
        )
        return raw


class FakeBridgeRobot:
    _BRIDGE_PRIMITIVES = BRIDGE_PRIMITIVES

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

    def get_bridge_snapshot(self) -> dict[str, Any]:
        return get_resource_bridge_snapshot(self)

    def _is_pose_in_workspace(
        self,
        pose: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check if a Cartesian pose falls within this robot's workspace bounds."""
        bounds = self.static_capabilities.get("workspace_bounds")
        if not bounds or not isinstance(bounds, dict):
            return True, "no workspace_bounds configured; defaulting to allowed"

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

        # Clear/home: always feasible.
        if op in {"clear", "home"}:
            return {
                "allowed": True,
                "reason": f"{op} operation always feasible for own robot",
                "evidence": evidence,
            }

        # Pick/place/pick_place: check target pose against workspace bounds.
        target_pose: dict[str, Any] | None = None
        if op in {"pick", "pick_place"}:
            target_pose = part_context.get("observed_pose") or part_context.get("pose")
        elif op == "place":
            target_info = part_context.get("target") or {}
            target_pose = target_info.get("slot_pose") or target_info.get("pose")

        if target_pose is None:
            return {
                "allowed": True,
                "reason": f"no target pose available for {op}; defaulting to allowed",
                "evidence": evidence,
            }

        inside, reason = self._is_pose_in_workspace(target_pose)
        evidence["checked_pose"] = deepcopy(target_pose)
        evidence["workspace_bounds"] = deepcopy(
            self.static_capabilities.get("workspace_bounds") or {}
        )
        return {
            "allowed": inside,
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
                "observation": {
                    "pose": deepcopy(self._position),
                    "resource_jid": self.jid,
                },
                "snapshot": self.get_bridge_snapshot(),
            }
        if primitive_name != "detect_parts":
            return {
                "success": False,
                "message": f"unsupported fake observation primitive '{primitive_name}'",
                "snapshot": self.get_bridge_snapshot(),
            }
        part_name = str(payload.get("part_name") or "").strip()
        observation = deepcopy(self._observations.get(part_name) or {})
        if not observation:
            return {
                "success": False,
                "message": f"no fake observation configured for part '{part_name}'",
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


class FakeBridgePrinter:
    def __init__(
        self,
        *,
        jid: str = "printer@localhost",
        current_state: str = "idle",
        current_location: str | None = "printer_bay",
        active_job: str | None = None,
        job_state: str | None = None,
        bed_state: str | None = None,
        material_state: str | None = None,
    ) -> None:
        self.agent_name = str(jid)
        self.jid = str(jid)
        self.static_capabilities = {"resource_type": "printer"}
        self._current_state = str(current_state)
        self._current_location = current_location
        self._active_job = active_job
        self._job_state = job_state or current_state
        self._bed_state = bed_state
        self._material_state = material_state

    def get_bridge_snapshot(self) -> dict[str, Any]:
        return canonical_bridge_resource(
            resource_jid=self.jid,
            resource_type="printer",
            snapshot={
                "resource_type": "printer",
                "current_state": self._current_state,
                "current_location": self._current_location,
                "active_job": self._active_job,
                "job_state": self._job_state,
                "bed_state": self._bed_state,
                "material_state": self._material_state,
            },
            modeled_state={},
        )


def _load_robot_config(path: Path, key: str) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"robot config {path} did not decode to an object")
    config = payload.get(key)
    if not isinstance(config, dict):
        raise KeyError(f"robot config {path} is missing top-level key '{key}'")
    return config


def _scripted_turns() -> list[dict[str, Any]]:
    return [
        {
            "type": "observe",
            "resource_jid": "ur5e@localhost",
            "primitive": "detect_parts",
            "params": {"part_name": "LG"},
            "store_as": "detected_lg",
            "reason_summary": "Need the live LG pose before returning a final bridge plan.",
        },
        {
            "type": "bridge_events",
            "events": [
                {
                    "event_name": "clear_xarm6_zone",
                    "resource_jid": "xarm6@localhost",
                    "operation_family": "clear",
                    "expected_resource_delta": {"from": "recovery_required", "to": "idle"},
                    "closes_conditions": [
                        {
                            "entity_kind": "resource",
                            "entity": "xarm6@localhost",
                            "field": "current_state",
                            "expected": "idle",
                        }
                    ],
                    "rationale": "Clear the blocked assembly robot to satisfy the active board mutex safety rule.",
                },
                {
                    "event_name": "return_mcp_to_printer",
                    "resource_jid": "ur5e@localhost",
                    "operation_family": "stage",
                    "part_name": "MCP",
                    "expected_resource_delta": {"from": "picked", "to": "idle"},
                    "expected_part_delta": {
                        "part_name": "MCP",
                        "from": "in_gripper",
                        "to": "ready",
                        "location_to": "prusa-mk4-2",
                    },
                    "closes_conditions": [],
                    "rationale": "Temporarily unload MCP before recovering LG.",
                },
                {
                    "event_name": "pick_lg",
                    "resource_jid": "ur5e@localhost",
                    "operation_family": "pick",
                    "part_name": "LG",
                    "expected_resource_delta": {"from": "idle", "to": "picked"},
                    "expected_part_delta": {
                        "part_name": "LG",
                        "from": "misplaced",
                        "to": "in_gripper",
                    },
                    "closes_conditions": [],
                    "rationale": "UR5e performs the LG recovery pick.",
                },
                {
                    "event_name": "insert_lg",
                    "resource_jid": "ur5e@localhost",
                    "operation_family": "assemble",
                    "part_name": "LG",
                    "expected_resource_delta": {"from": "picked", "to": "idle"},
                    "expected_part_delta": {
                        "part_name": "LG",
                        "from": "in_gripper",
                        "to": "assembled",
                        "location_to": "assembly_board-v1",
                    },
                    "closes_conditions": [
                        {
                            "entity_kind": "part",
                            "entity": "LG",
                            "field": "state",
                            "expected": "assembled",
                        },
                        {
                            "entity_kind": "part",
                            "entity": "LG",
                            "field": "location",
                            "expected": "assembly_board-v1",
                        },
                    ],
                    "rationale": "Insert LG onto the product so DES can resume.",
                },
                {
                    "event_name": "repick_mcp_for_resume",
                    "resource_jid": "ur5e@localhost",
                    "operation_family": "pick",
                    "part_name": "MCP",
                    "expected_resource_delta": {"from": "idle", "to": "picked"},
                    "expected_part_delta": {
                        "part_name": "MCP",
                        "from": "ready",
                        "to": "in_gripper",
                    },
                    "closes_conditions": [],
                    "rationale": "Restore the MCP suffix after LG is reassembled.",
                },
            ],
            "reason_summary": "Close all marked re-entry conditions before primitive refinement.",
        },
        {
            "type": "final_plan",
            "plan": {},
            "reason_summary": (
                "UR5e should recover LG while xarm6 clears and MCP is temporarily unloaded."
            ),
        },
    ]


def _scripted_turns_main_v2() -> list[dict[str, Any]]:
    return [
        {
            "type": "observe",
            "resource_jid": "xarm6@localhost",
            "primitive": "detect_parts",
            "params": {"part_name": "LG"},
            "store_as": "detected_lg",
            "reason_summary": "Confirm the mirrored LG recovery pose before proposing bridge events.",
        },
        {
            "type": "bridge_events",
            "events": [
                {
                    "event_name": "xarm6_recover_to_idle",
                    "resource_jid": "xarm6@localhost",
                    "operation_family": "clear",
                    "expected_resource_delta": {"from": "recovery_required", "to": "idle"},
                    "closes_conditions": [
                        {
                            "entity_kind": "resource",
                            "entity": "xarm6@localhost",
                            "field": "current_state",
                            "expected": "idle",
                        }
                    ],
                    "rationale": "Clear the disrupted xarm6 so it can safely perform the mirrored LG recovery.",
                },
                {
                    "event_name": "xarm6_pick_place_LG",
                    "resource_jid": "xarm6@localhost",
                    "operation_family": "pick_place",
                    "part_name": "LG",
                    "expected_resource_delta": {"from": "idle", "to": "idle"},
                    "expected_part_delta": {
                        "part_name": "LG",
                        "from": "misplaced",
                        "to": "assembled",
                        "location_to": "assembly_board-v1",
                    },
                    "closes_conditions": [
                        {
                            "entity_kind": "part",
                            "entity": "LG",
                            "field": "state",
                            "expected": "assembled",
                        },
                        {
                            "entity_kind": "part",
                            "entity": "LG",
                            "field": "location",
                            "expected": "assembly_board-v1",
                        },
                    ],
                    "rationale": "In the mirrored case, xarm6 can recover and assemble LG directly.",
                },
            ],
            "reason_summary": "Mirror the original case by letting xarm6 complete the LG recovery itself.",
        },
        {
            "type": "final_plan",
            "plan": {},
            "reason_summary": "Planner should deterministically compile the mirrored bridge from approved events.",
        },
    ]


def _scripted_turns_both_reachable() -> list[dict[str, Any]]:
    """LG at y=0.05 — both robots can reach. xarm6 handles it alone (simpler)."""
    return [
        {
            "type": "observe",
            "resource_jid": "xarm6@localhost",
            "primitive": "detect_parts",
            "params": {"part_name": "LG"},
            "store_as": "detected_lg",
            "reason_summary": "Observe LG at shared-workspace pose.",
        },
        {
            "type": "bridge_events",
            "events": [
                {
                    "event_name": "xarm6_recover_to_idle",
                    "resource_jid": "xarm6@localhost",
                    "operation_family": "clear",
                    "expected_resource_delta": {"from": "recovery_required", "to": "idle"},
                    "closes_conditions": [
                        {"entity_kind": "resource", "entity": "xarm6@localhost", "field": "current_state", "expected": "idle"},
                    ],
                    "rationale": "Transition xarm6 out of recovery_required.",
                },
                {
                    "event_name": "xarm6_pick_place_LG",
                    "resource_jid": "xarm6@localhost",
                    "operation_family": "pick_place",
                    "part_name": "LG",
                    "expected_resource_delta": {"from": "idle", "to": "idle"},
                    "expected_part_delta": {"part_name": "LG", "from": "misplaced", "to": "assembled", "location_to": "assembly_board-v1"},
                    "closes_conditions": [
                        {"entity_kind": "part", "entity": "LG", "field": "state", "expected": "assembled"},
                        {"entity_kind": "part", "entity": "LG", "field": "location", "expected": "assembly_board-v1"},
                    ],
                    "rationale": "Both robots can reach LG; xarm6 is simpler (no MCP staging needed).",
                },
            ],
            "reason_summary": "LG reachable by both; xarm6 handles alone.",
        },
        {
            "type": "final_plan",
            "plan": {},
            "reason_summary": "Deterministic compilation from approved events.",
        },
    ]


def _both_reachable_scenario_overrides() -> dict[str, Any]:
    both_reachable_pose = {"x": 0.0, "y": 0.05, "z": 1.034}
    return {
        "fixture": {
            "part_tracker": {
                "LG": {
                    "location": "fixture_shared_recovery_zone",
                    "last_known_location": "fixture_shared_recovery_zone",
                    "observed_pose": deepcopy(both_reachable_pose),
                },
            },
            "part_locations": {"LG": "fixture_shared_recovery_zone"},
            "stuck_state": {"part_locations": {"LG": "fixture_shared_recovery_zone"}},
        },
        "robots": {
            "ur5e@localhost": {
                "observations": {"LG": {"part_name": "LG", "pose": deepcopy(both_reachable_pose)}},
            },
            "xarm6@localhost": {
                "observations": {"LG": {"part_name": "LG", "pose": deepcopy(both_reachable_pose)}},
            },
        },
        "scripted_turns": _scripted_turns_both_reachable(),
    }


def _scripted_turns_mcp_swap() -> list[dict[str, Any]]:
    """MCP is misplaced instead of LG. LG is in ur5e's gripper."""
    return [
        {
            "type": "observe",
            "resource_jid": "ur5e@localhost",
            "primitive": "detect_parts",
            "params": {"part_name": "MCP"},
            "store_as": "detected_mcp",
            "reason_summary": "Observe misplaced MCP location.",
        },
        {
            "type": "observe",
            "resource_jid": "ur5e@localhost",
            "primitive": "detect_parts",
            "params": {"part_name": "LG"},
            "store_as": "detected_lg",
            "reason_summary": "Confirm LG pose in gripper for staging.",
        },
        {
            "type": "bridge_events",
            "events": [
                {
                    "event_name": "xarm6_recover_to_idle",
                    "resource_jid": "xarm6@localhost",
                    "operation_family": "clear",
                    "expected_resource_delta": {"from": "recovery_required", "to": "idle"},
                    "closes_conditions": [
                        {"entity_kind": "resource", "entity": "xarm6@localhost", "field": "current_state", "expected": "idle"},
                    ],
                    "rationale": "Clear xarm6 from recovery state.",
                },
                {
                    "event_name": "ur5e_stage_LG_to_prusa_mk4_2",
                    "resource_jid": "ur5e@localhost",
                    "operation_family": "stage",
                    "part_name": "LG",
                    "expected_resource_delta": {"from": "picked", "to": "idle"},
                    "expected_part_delta": {"part_name": "LG", "from": "in_gripper", "to": "ready", "location_to": "prusa-mk4-2"},
                    "closes_conditions": [],
                    "rationale": "Free ur5e gripper so it can pick MCP.",
                },
                {
                    "event_name": "ur5e_pick_MCP_from_recovery_zone",
                    "resource_jid": "ur5e@localhost",
                    "operation_family": "pick",
                    "part_name": "MCP",
                    "expected_resource_delta": {"from": "idle", "to": "picked"},
                    "expected_part_delta": {"part_name": "MCP", "from": "misplaced", "to": "in_gripper"},
                    "closes_conditions": [],
                    "rationale": "Pick up misplaced MCP.",
                },
                {
                    "event_name": "ur5e_place_MCP_to_assembly",
                    "resource_jid": "ur5e@localhost",
                    "operation_family": "assemble",
                    "part_name": "MCP",
                    "expected_resource_delta": {"from": "picked", "to": "idle"},
                    "expected_part_delta": {"part_name": "MCP", "from": "in_gripper", "to": "assembled", "location_to": "assembly_board-v1"},
                    "closes_conditions": [
                        {"entity_kind": "part", "entity": "MCP", "field": "state", "expected": "assembled"},
                        {"entity_kind": "part", "entity": "MCP", "field": "location", "expected": "assembly_board-v1"},
                    ],
                    "rationale": "Place MCP at assembly destination.",
                },
                {
                    "event_name": "ur5e_repick_LG_for_resume",
                    "resource_jid": "ur5e@localhost",
                    "operation_family": "pick",
                    "part_name": "LG",
                    "expected_resource_delta": {"from": "idle", "to": "picked"},
                    "expected_part_delta": {"part_name": "LG", "from": "ready", "to": "in_gripper"},
                    "closes_conditions": [
                        {"entity_kind": "resource", "entity": "ur5e@localhost", "field": "held_part", "expected": "LG"},
                        {"entity_kind": "part", "entity": "LG", "field": "state", "expected": "in_gripper"},
                    ],
                    "rationale": "Repick LG so ur5e can resume its pending suffix.",
                },
            ],
            "reason_summary": "MCP-swap: recover misplaced MCP, stage LG, then repick LG for resume.",
        },
        {
            "type": "final_plan",
            "plan": {},
            "reason_summary": "Deterministic compilation from approved events.",
        },
    ]


def _mcp_swap_scenario_overrides() -> dict[str, Any]:
    mcp_drop_pose = {"x": 0.002, "y": 0.198, "z": 1.025}
    return {
        "fixture": {
            "part_tracker": {
                "LG": {
                    "state": "in_gripper",
                    "location": "ur5e@localhost_gripper",
                    "last_known_location": "ur5e@localhost_gripper",
                    "observed_pose": None,
                },
                "MCP": {
                    "state": "misplaced",
                    "location": "fixture_ur5e_recovery_pick_zone",
                    "last_known_location": "fixture_ur5e_recovery_pick_zone",
                    "observed_pose": deepcopy(mcp_drop_pose),
                },
            },
            "part_states": {"LG": "in_gripper", "MCP": "misplaced"},
            "part_locations": {"LG": "ur5e@localhost_gripper", "MCP": "fixture_ur5e_recovery_pick_zone"},
            "stuck_state": {
                "part_states": {"LG": "in_gripper", "MCP": "misplaced"},
                "part_locations": {"LG": "ur5e@localhost_gripper", "MCP": "fixture_ur5e_recovery_pick_zone"},
            },
            "resource_states": {
                "ur5e@localhost": {"held_part": "LG"},
            },
        },
        "robots": {
            "ur5e@localhost": {
                "held_part": "LG",
                "observations": {
                    "MCP": {"part_name": "MCP", "pose": deepcopy(mcp_drop_pose)},
                    "LG": {"part_name": "LG", "pose": {"x": -0.25, "y": 0.22, "z": 1.18}},
                },
            },
            "xarm6@localhost": {
                "observations": {
                    "MCP": {"part_name": "MCP", "pose": deepcopy(mcp_drop_pose)},
                },
            },
        },
        "scripted_turns": _scripted_turns_mcp_swap(),
    }


def _scripted_turns_no_gripper_conflict() -> list[dict[str, Any]]:
    """LG misplaced, ur5e gripper empty (MCP already assembled). Simpler recovery."""
    return [
        {
            "type": "observe",
            "resource_jid": "ur5e@localhost",
            "primitive": "detect_parts",
            "params": {"part_name": "LG"},
            "store_as": "detected_lg",
            "reason_summary": "Observe misplaced LG.",
        },
        {
            "type": "bridge_events",
            "events": [
                {
                    "event_name": "xarm6_recover_to_idle",
                    "resource_jid": "xarm6@localhost",
                    "operation_family": "clear",
                    "expected_resource_delta": {"from": "recovery_required", "to": "idle"},
                    "closes_conditions": [
                        {"entity_kind": "resource", "entity": "xarm6@localhost", "field": "current_state", "expected": "idle"},
                    ],
                    "rationale": "Clear xarm6 from recovery state.",
                },
                {
                    "event_name": "ur5e_pick_LG_from_recovery_zone",
                    "resource_jid": "ur5e@localhost",
                    "operation_family": "pick",
                    "part_name": "LG",
                    "expected_resource_delta": {"from": "idle", "to": "picked"},
                    "expected_part_delta": {"part_name": "LG", "from": "misplaced", "to": "in_gripper"},
                    "closes_conditions": [],
                    "rationale": "Pick up LG — no staging needed since gripper is empty.",
                },
                {
                    "event_name": "ur5e_place_LG_to_assembly",
                    "resource_jid": "ur5e@localhost",
                    "operation_family": "assemble",
                    "part_name": "LG",
                    "expected_resource_delta": {"from": "picked", "to": "idle"},
                    "expected_part_delta": {"part_name": "LG", "from": "in_gripper", "to": "assembled", "location_to": "assembly_board-v1"},
                    "closes_conditions": [
                        {"entity_kind": "part", "entity": "LG", "field": "state", "expected": "assembled"},
                        {"entity_kind": "part", "entity": "LG", "field": "location", "expected": "assembly_board-v1"},
                    ],
                    "rationale": "Place LG at assembly board.",
                },
            ],
            "reason_summary": "Simple recovery: no gripper conflict, pick-and-place LG directly.",
        },
        {
            "type": "final_plan",
            "plan": {},
            "reason_summary": "Deterministic compilation from approved events.",
        },
    ]


def _no_gripper_conflict_scenario_overrides() -> dict[str, Any]:
    return {
        "fixture": {
            "part_tracker": {
                "MCP": {
                    "state": "assembled",
                    "location": "assembly_board-v1",
                    "last_known_location": "assembly_board-v1",
                    "observed_pose": None,
                },
            },
            "part_states": {"MCP": "assembled"},
            "part_locations": {"MCP": "assembly_board-v1"},
            "stuck_state": {
                "part_states": {"LG": "misplaced", "MCP": "assembled"},
                "part_locations": {"LG": "fixture_ur5e_recovery_pick_zone", "MCP": "assembly_board-v1"},
            },
            "resource_states": {
                "ur5e@localhost": {
                    "current_state": "idle",
                    "held_part": None,
                    "current_location": "ur5e_home",
                },
            },
        },
        "robots": {
            "ur5e@localhost": {
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            },
        },
        "scripted_turns": _scripted_turns_no_gripper_conflict(),
    }


def _runtime_fixture() -> dict[str, Any]:
    part_tracker = {
        "LG": {
            "state": "misplaced",
            "location": "fixture_ur5e_recovery_pick_zone",
            "last_known_location": "fixture_ur5e_recovery_pick_zone",
            "last_successful_task": "REQ_2_T3",
            "observed_pose": deepcopy(LG_DROP_POSE),
            "origin_resource_location": "prusa-mk4-1",
        },
        "MCP": {
            "state": "in_gripper",
            "location": "ur5e@localhost_gripper",
            "last_known_location": "ur5e@localhost_gripper",
            "last_successful_task": "REQ_1_T2",
            "origin_resource_location": "prusa-mk4-2",
        },
    }
    part_states = {"LG": "misplaced", "MCP": "in_gripper"}
    part_locations = {
        "LG": "fixture_ur5e_recovery_pick_zone",
        "MCP": "ur5e@localhost_gripper",
    }
    resource_states = {
        "xarm6@localhost": {
            "current_state": "recovery_required",
            "held_part": None,
            "current_location": "assembly_board-v1",
        },
        "ur5e@localhost": {
            "current_state": "picked",
            "held_part": "MCP",
            "current_location": "ur5e_home",
        },
    }
    stuck_state = {
        "resource_state": "recovery_required",
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
        "obligation_targets": [{"rule_id": CASE3_BOARD_MUTEX_RULE_ID, "resource_jid": "xarm6@localhost"}],
        "bridge_feedback": "",
        "default_resource_state": "idle",
        "part_tracker": part_tracker,
        "part_states": part_states,
        "part_locations": part_locations,
        "resource_states": resource_states,
        "stuck_state": stuck_state,
        "bridge_safety_context": {
            "rule_ids": [CASE3_BOARD_MUTEX_RULE_ID],
            "safe_next_task_ids": [],
            "running_aps": [],
            "candidate_aps": [],
            "predicted_state_aps": [],
            "status": "",
            "reason": "",
            "constraints": [],
            "safety_rules": [deepcopy(CASE3_BOARD_MUTEX_RULE)],
        },
    }


def _deep_merge_dict(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge_dict(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def _case3_robot_specs() -> dict[str, dict[str, Any]]:
    return {
        "ur5e@localhost": {
            "current_state": "picked",
            "held_part": "MCP",
            "gripper_state": "closed",
            "pose_ref": None,
            "position": {"x": -0.25, "y": 0.22, "z": 1.18},
            "observations": {
                "LG": {"part_name": "LG", "pose": deepcopy(LG_DROP_POSE)},
                "MCP": {"part_name": "MCP", "pose": {"x": 0.0, "y": -0.08, "z": 1.025}},
            },
        },
        "xarm6@localhost": {
            "current_state": "recovery_required",
            "held_part": None,
            "gripper_state": "open",
            "pose_ref": None,
            "position": {"x": 0.1, "y": 0.08, "z": 1.05},
            "observations": {
                "LG": {"part_name": "LG", "pose": deepcopy(LG_DROP_POSE)},
            },
        },
    }


def _apply_case3_scenario_overrides(
    fixture: dict[str, Any],
    robot_specs: dict[str, dict[str, Any]],
    scenario_overrides: dict[str, Any] | None,
) -> tuple[str, list[dict[str, Any]] | None]:
    overrides = deepcopy(scenario_overrides or {})
    fixture_overrides = overrides.get("fixture") or {}
    if isinstance(fixture_overrides, dict):
        _deep_merge_dict(fixture, fixture_overrides)

    robot_overrides = overrides.get("robots") or {}
    for resource_jid, raw_robot_overrides in robot_overrides.items():
        jid = str(resource_jid or "").strip()
        if not jid or not isinstance(raw_robot_overrides, dict):
            continue
        if jid not in robot_specs:
            robot_specs[jid] = {}
        _deep_merge_dict(robot_specs[jid], raw_robot_overrides)

    ra_jid = str(overrides.get("ra_jid") or "xarm6@localhost").strip() or "xarm6@localhost"
    scripted_turns = overrides.get("scripted_turns")
    if scripted_turns is not None and not isinstance(scripted_turns, list):
        scripted_turns = None
    return ra_jid, deepcopy(scripted_turns)


def _apply_prepared_bridge_request_overrides(
    planner: ProcessPlanner,
    prepared_bridge_request: dict[str, Any],
    scenario_overrides: dict[str, Any] | None,
) -> None:
    overrides = deepcopy(scenario_overrides or {})
    prepared_overrides = overrides.get("prepared") or {}
    if "ra_jid" in overrides:
        prepared_bridge_request["ra_jid"] = str(overrides.get("ra_jid") or "").strip()

    if not prepared_overrides:
        if "ra_jid" in overrides:
            planner._refresh_bridge_grounding_context(prepared_bridge_request)
        return

    for key, value in prepared_overrides.items():
        if key == "bridge_resources" and isinstance(value, dict):
            bridge_resources = prepared_bridge_request.setdefault("bridge_resources", {})
            for resource_jid, raw_resource_updates in value.items():
                jid = str(resource_jid or "").strip()
                if not jid:
                    continue
                if not isinstance(raw_resource_updates, dict):
                    bridge_resources[jid] = deepcopy(raw_resource_updates)
                    continue
                resource_entry = bridge_resources.setdefault(jid, {})
                for field, field_value in raw_resource_updates.items():
                    if (
                        isinstance(field_value, dict)
                        and isinstance(resource_entry.get(field), dict)
                    ):
                        _deep_merge_dict(resource_entry[field], field_value)
                    else:
                        resource_entry[field] = deepcopy(field_value)
            continue

        if isinstance(value, dict) and isinstance(prepared_bridge_request.get(key), dict):
            _deep_merge_dict(prepared_bridge_request[key], value)
        else:
            prepared_bridge_request[key] = deepcopy(value)

    planner._refresh_bridge_grounding_context(prepared_bridge_request)


def _bridge_observation_entry(
    *,
    resource_jid: str,
    part_name: str,
    pose: dict[str, float],
    turn_index: int = 1,
    store_as: str | None = None,
) -> dict[str, Any]:
    return {
        "turn_index": int(turn_index),
        "resource_jid": str(resource_jid),
        "primitive": "detect_parts",
        "params": {"part_name": str(part_name)},
        "store_as": str(store_as or f"detected_{str(part_name).lower()}"),
        "observation": {"part_name": str(part_name), "pose": deepcopy(pose)},
    }


def _record_bridge_observation(
    planner: ProcessPlanner,
    prepared_bridge_request: dict[str, Any],
    *,
    resource_jid: str,
    part_name: str,
    pose: dict[str, float],
    turn_index: int = 1,
    replace_history: bool = True,
) -> None:
    part_tracker = prepared_bridge_request.setdefault("part_tracker", {})
    part_entry = part_tracker.setdefault(str(part_name), {})
    part_entry["observed_pose"] = deepcopy(pose)

    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    observation_history = [] if replace_history else list(bridge_session.get("observation_history") or [])
    observation_history.append(
        _bridge_observation_entry(
            resource_jid=resource_jid,
            part_name=part_name,
            pose=pose,
            turn_index=turn_index,
        )
    )
    bridge_session["observation_history"] = observation_history
    prepared_bridge_request["bridge_session"] = bridge_session
    planner._refresh_bridge_grounding_context(prepared_bridge_request)


def _bridge_clear_event(
    *,
    resource_jid: str,
    from_state: str = "recovery_required",
    to_state: str = "idle",
) -> dict[str, Any]:
    resource_token = str(resource_jid).split("@", 1)[0]
    return {
        "event_name": f"{resource_token}_recover_to_idle",
        "resource_jid": str(resource_jid),
        "operation_family": "clear",
        "expected_resource_delta": {"from": str(from_state), "to": str(to_state)},
        "closes_conditions": [
            {
                "entity_kind": "resource",
                "entity": str(resource_jid),
                "field": "current_state",
                "expected": str(to_state),
            }
        ],
    }


def _bridge_pick_place_event(
    *,
    resource_jid: str,
    part_name: str,
    from_state: str = "idle",
    to_state: str = "idle",
    location_to: str = "assembly_board-v1",
) -> dict[str, Any]:
    resource_token = str(resource_jid).split("@", 1)[0]
    return {
        "event_name": f"{resource_token}_pick_place_{str(part_name)}",
        "resource_jid": str(resource_jid),
        "operation_family": "pick_place",
        "part_name": str(part_name),
        "expected_resource_delta": {"from": str(from_state), "to": str(to_state)},
        "expected_part_delta": {
            "part_name": str(part_name),
            "from": "misplaced",
            "to": "assembled",
            "location_to": str(location_to),
        },
        "closes_conditions": [
            {
                "entity_kind": "part",
                "entity": str(part_name),
                "field": "state",
                "expected": "assembled",
            },
            {
                "entity_kind": "part",
                "entity": str(part_name),
                "field": "location",
                "expected": str(location_to),
            },
        ],
    }


def _bridge_stage_event(
    *,
    resource_jid: str,
    part_name: str,
    from_state: str = "picked",
    to_state: str = "idle",
    location_to: str = "prusa-mk4-2",
) -> dict[str, Any]:
    resource_token = str(resource_jid).split("@", 1)[0]
    return {
        "event_name": f"{resource_token}_stage_{str(part_name)}",
        "resource_jid": str(resource_jid),
        "operation_family": "stage",
        "part_name": str(part_name),
        "expected_resource_delta": {"from": str(from_state), "to": str(to_state)},
        "expected_part_delta": {
            "part_name": str(part_name),
            "from": "in_gripper",
            "to": "ready",
            "location_to": str(location_to),
        },
        "closes_conditions": [],
    }


def _both_robots_idle_overrides() -> dict[str, Any]:
    return {
        "fixture": {
            "part_tracker": {
                "MCP": {
                    "state": "ready",
                    "location": "prusa-mk4-2",
                    "last_known_location": "prusa-mk4-2",
                }
            },
            "part_states": {"MCP": "ready"},
            "part_locations": {"MCP": "prusa-mk4-2"},
            "resource_states": {
                "ur5e@localhost": {
                    "current_state": "idle",
                    "held_part": None,
                    "current_location": "ur5e_home",
                },
                "xarm6@localhost": {
                    "current_state": "idle",
                    "held_part": None,
                    "current_location": "xarm6_home",
                },
            },
            "stuck_state": {
                "resource_state": "idle",
                "current_location": "xarm6_home",
                "part_states": {"MCP": "ready"},
                "part_locations": {"MCP": "prusa-mk4-2"},
            },
        },
        "robots": {
            "ur5e@localhost": {
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            },
            "xarm6@localhost": {
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            },
        },
    }


def _main_v2_scenario_overrides() -> dict[str, Any]:
    mirror_pose = {"x": 0.0, "y": -0.50, "z": 1.034}
    return {
        "fixture": {
            "part_tracker": {
                "LG": {
                    "location": "fixture_xarm6_recovery_pick_zone",
                    "last_known_location": "fixture_xarm6_recovery_pick_zone",
                    "observed_pose": deepcopy(mirror_pose),
                }
            },
            "part_locations": {"LG": "fixture_xarm6_recovery_pick_zone"},
            "stuck_state": {"part_locations": {"LG": "fixture_xarm6_recovery_pick_zone"}},
        },
        "robots": {
            "ur5e@localhost": {
                "observations": {
                    "LG": {"part_name": "LG", "pose": deepcopy(mirror_pose)},
                }
            },
            "xarm6@localhost": {
                "observations": {
                    "LG": {"part_name": "LG", "pose": deepcopy(mirror_pose)},
                }
            },
        },
        "scripted_turns": _scripted_turns_main_v2(),
    }


def _case3_variant_config(variant: str) -> dict[str, Any]:
    token = str(variant or MAIN_V1_VARIANT).strip() or MAIN_V1_VARIANT
    if token == MAIN_V2_VARIANT:
        return {
            "variant": MAIN_V2_VARIANT,
            "scenario_id": MAIN_V2_SCENARIO_ID,
            "scenario_overrides": _main_v2_scenario_overrides(),
            "scripted_final_plan_builder": None,
        }
    if token == BOTH_REACHABLE_VARIANT:
        return {
            "variant": BOTH_REACHABLE_VARIANT,
            "scenario_id": BOTH_REACHABLE_SCENARIO_ID,
            "scenario_overrides": _both_reachable_scenario_overrides(),
            "scripted_final_plan_builder": None,
        }
    if token == MCP_SWAP_VARIANT:
        return {
            "variant": MCP_SWAP_VARIANT,
            "scenario_id": MCP_SWAP_SCENARIO_ID,
            "scenario_overrides": _mcp_swap_scenario_overrides(),
            "scripted_final_plan_builder": None,
        }
    if token == NO_GRIPPER_CONFLICT_VARIANT:
        return {
            "variant": NO_GRIPPER_CONFLICT_VARIANT,
            "scenario_id": NO_GRIPPER_CONFLICT_SCENARIO_ID,
            "scenario_overrides": _no_gripper_conflict_scenario_overrides(),
            "scripted_final_plan_builder": None,
        }
    return {
        "variant": MAIN_V1_VARIANT,
        "scenario_id": SCENARIO_ID,
        "scenario_overrides": {},
        "scripted_final_plan_builder": lambda prepared_bridge_request: build_preprogrammed_bridge_proposal(
            scenario_id=SCENARIO_ID,
            prepared_bridge_request=prepared_bridge_request,
        ),
    }


def _write_debug_artifact(payload: dict[str, Any], *, variant: str = MAIN_V1_VARIANT) -> Path:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    variant_token = str(variant or MAIN_V1_VARIANT).strip().replace("-", "_") or "main_v1"
    artifact_path = DEBUG_DIR / f"bridge_case3_recovery_{variant_token}_dry_run_{ts}.json"
    latest_path = DEBUG_DIR / f"bridge_case3_recovery_{variant_token}_latest.json"
    artifact_text = json.dumps(payload, indent=2, default=str)
    artifact_path.write_text(artifact_text, encoding="utf-8")
    latest_path.write_text(artifact_text, encoding="utf-8")
    return artifact_path


def _build_debug_payload(
    *,
    llm_mode: str,
    product_agent: FakeProductAgent,
    fixture: dict[str, Any],
    proposal: dict[str, Any] | None,
    compiled_tasks: list[dict[str, Any]] | None,
    bridge_debug: dict[str, Any] | None,
    prepared_bridge_request: dict[str, Any] | None,
    error: str = "",
) -> dict[str, Any]:
    prepared = prepared_bridge_request if isinstance(prepared_bridge_request, dict) else {}
    bridge_debug_dict = deepcopy(bridge_debug or {}) if isinstance(bridge_debug, dict) else {}
    turn_types = [
        str((turn.get("normalized_response") or {}).get("type") or "").strip()
        for turn in (bridge_debug_dict.get("turns") or [])
        if isinstance(turn, dict)
        and isinstance(turn.get("normalized_response"), dict)
        and str((turn.get("normalized_response") or {}).get("type") or "").strip()
    ]
    return {
        "case_id": CASE_ID,
        "scenario_id": product_agent.scenario_id,
        "scenario_variant": product_agent.scenario_variant,
        "llm_mode": llm_mode,
        "llm_model": product_agent.llm_model,
        "fixture": deepcopy(fixture),
        "scripted_turns": deepcopy(product_agent.turn_log),
        "raw_final_plan": deepcopy(
            (product_agent.turn_log[-1].get("response") or {}).get("plan")
            if product_agent.turn_log and isinstance(product_agent.turn_log[-1].get("response"), dict)
            else {}
        ),
        "proposal": deepcopy(proposal) if isinstance(proposal, dict) else None,
        "compiled_tasks": deepcopy(compiled_tasks or []),
        "bridge_debug": bridge_debug_dict,
        "marked_reentry_context": deepcopy(
            prepared.get("marked_reentry_context")
            or prepared.get("continuation_context")
            or {}
        ),
        "bridge_event_summary": deepcopy(
            (proposal or {}).get("bridge_event_summary") if isinstance(proposal, dict) else []
        ),
        "plan_rewrite": deepcopy(
            (proposal or {}).get("plan_rewrite") if isinstance(proposal, dict) else {}
        ),
        "approved_bridge_events": deepcopy(
            prepared.get("bridge_session", {}).get("approved_bridge_events") or []
        ),
        "feasibility_decisions": deepcopy(
            bridge_debug_dict.get("feasibility_decisions")
            or (
                (bridge_debug_dict.get("turns") or [])[-1].get("feasibility_decisions")
                if isinstance((bridge_debug_dict.get("turns") or [])[-1], dict)
                else []
            )
            if (bridge_debug_dict.get("turns") or [])
            else []
        ),
        "bridge_safety_context": deepcopy(
            prepared.get("bridge_safety_context") or {}
        ),
        "compile_path": str(
            bridge_debug_dict.get("compile_path")
            or (
                (bridge_debug_dict.get("turns") or [])[-1].get("compile_path")
                if isinstance((bridge_debug_dict.get("turns") or [])[-1], dict)
                else ""
            )
            or ""
        ),
        "turn_types": turn_types,
        "phase": str(prepared.get("bridge_session", {}).get("phase") or ""),
        "observation_history": deepcopy(
            prepared.get("bridge_session", {}).get("observation_history") or []
        ),
        "step_outputs": deepcopy(
            prepared.get("grounding_context", {}).get("step_outputs") or {}
        ),
        "error": str(error or ""),
    }


def _configure_live_bridge_session(prepared_bridge_request: dict[str, Any]) -> None:
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["max_turns"] = max(int(bridge_session.get("max_turns", 6) or 6), 10)
    bridge_session["max_observations"] = max(
        int(bridge_session.get("max_observations", 3) or 3), 4
    )
    bridge_session["max_final_retries"] = max(
        int(bridge_session.get("max_final_retries", 2) or 2), 4
    )
    feedback = [
        str(item).strip()
        for item in (bridge_session.get("operator_feedback_history") or [])
        if str(item).strip()
    ]
    guidance = (
        "Phase reminder: follow observe_required -> bridge_events -> final_plan. "
        "In observe_required, request one live observation for the misplaced bridge-critical part. "
        "In bridge_events, propose DES-style bridge events that close all marked re-entry conditions. "
        "In final_plan, realize the approved bridge events in order. "
        "Schema reminder: in final_plan primitive_steps, use store_as only on "
        "detect_parts, get_current_pose, compute_pick_targets, or compute_place_targets. "
        "Do not use store_as on motion, gripper, or attach/detach primitives. "
        "A valid final bridge must close all marked re-entry conditions and restore the resumable suffix, "
        "not only satisfy the immediate safety obligation."
    )
    if guidance not in feedback:
        feedback.append(guidance)
    bridge_session["operator_feedback_history"] = feedback[-12:]
    prepared_bridge_request["bridge_session"] = bridge_session


def _build_safety_only_bridge_plan() -> dict[str, Any]:
    return {
        "primary_obligation": {
            "rule_id": CASE3_BOARD_MUTEX_RULE_ID,
            "resource_jid": "xarm6@localhost",
        },
        "bridge_event_summary": [
            {
                "event_name": "clear_blocked_xarm6",
                "resource_jid": "xarm6@localhost",
                "closes_conditions": [
                    {
                        "entity_kind": "resource",
                        "entity": "xarm6@localhost",
                        "field": "current_state",
                        "expected": "idle",
                    }
                ],
                "rationale": "Clears the blocked robot but does not restore the displaced part.",
            }
        ],
        "macro_tasks": [
            {
                "resource_jid": "xarm6@localhost",
                "macro_name": "xarm6_recovery_clear_only",
                "description": "Move xarm6 to recovery_clear",
                "rationale": "Satisfy the immediate safety obligation.",
                "expected_start_state": "recovery_required",
                "task_params": {},
                "task_metadata": {
                    "in_state": "recovery_required",
                    "out_state": "idle",
                    "required_context_keys": [],
                    "context_mapping": {},
                    "part_transition": {},
                },
                "primitive_steps": [
                    {
                        "primitive": "move_to_named_pose",
                        "params": {"pose_name": "recovery_clear"},
                    }
                ],
            }
        ],
    }


def _relax_recovery_clear_precondition(prepared_bridge_request: dict[str, Any]) -> None:
    def _rewrite_catalog(catalog: list[dict[str, Any]]) -> None:
        for row in catalog:
            if not isinstance(row, dict):
                continue
            if str(row.get("name") or "") != "move_to_named_pose":
                continue
            preconditions = dict(row.get("preconditions") or {})
            current_state = dict(preconditions.get("current_state") or {})
            if current_state.get("not_equals") == "recovery_required":
                current_state.pop("not_equals", None)
                if current_state:
                    preconditions["current_state"] = current_state
                else:
                    preconditions.pop("current_state", None)
                row["preconditions"] = preconditions

    _rewrite_catalog(list(prepared_bridge_request.get("primitive_catalog") or []))
    bridge_resources = prepared_bridge_request.get("bridge_resources") or {}
    if isinstance(bridge_resources, dict):
        xarm_entry = bridge_resources.get("xarm6@localhost")
        if isinstance(xarm_entry, dict):
            _rewrite_catalog(list(xarm_entry.get("primitive_catalog") or []))


def _assert_expected_output(result: dict[str, Any]) -> None:
    variant = str(result.get("scenario_variant") or MAIN_V1_VARIANT)
    if variant == MAIN_V2_VARIANT:
        _assert_expected_output_main_v2(result)
        return
    if variant in (BOTH_REACHABLE_VARIANT, MCP_SWAP_VARIANT, NO_GRIPPER_CONFLICT_VARIANT):
        _assert_expected_output_generic(result)
        return
    proposal = result["proposal"]
    compiled_tasks = result["compiled_tasks"]
    bridge_debug = result["bridge_debug"]
    prepared_bridge_request = result["prepared_bridge_request"]
    marked_reentry_context = (
        prepared_bridge_request.get("marked_reentry_context")
        or prepared_bridge_request.get("continuation_context")
        or {}
    )

    macro_names = [
        str(task.get("macro_name") or "")
        for task in (proposal.get("macro_tasks") or [])
        if isinstance(task, dict)
    ]
    assert proposal.get("plan_rewrite", {}).get("replace_failed_branch") is True
    assert proposal.get("plan_rewrite", {}).get("resume_task_ids") == [
        "REQ_1_T3",
        "REQ_1_T4",
        "REQ_1_T5",
    ]
    assert proposal.get("primary_obligation", {}).get("rule_id") == CASE3_BOARD_MUTEX_RULE_ID
    assert proposal.get("primary_obligation", {}).get("resource_jid") == "xarm6@localhost"
    assert macro_names == [
        "clear_xarm6_zone",
        "return_mcp_to_printer",
        "pick_lg",
        "insert_lg",
        "repick_mcp_for_resume",
    ]
    assert bridge_debug.get("status") == "accepted"
    assert bridge_debug.get("marked_reentry_check", {}).get("accepted") is True
    assert bridge_debug.get("compile_path") in {"deterministic", "llm_repair"}
    assert len(bridge_debug.get("turns") or []) == 3
    assert (
        bridge_debug.get("turns", [{}])[0]
        .get("normalized_response", {})
        .get("type")
        == "observe"
    )
    assert (
        bridge_debug.get("turns", [{}, {}])[1]
        .get("normalized_response", {})
        .get("type")
        == "bridge_events"
    )
    assert bridge_debug.get("turns", [{}, {}, {}])[1].get("accepted") is True
    assert bridge_debug.get("turns", [{}, {}, {}])[2].get("accepted") is True
    assert (
        prepared_bridge_request["grounding_context"]["step_outputs"]["detected_lg"]["pose"]["x"]
        == 0.002
    )
    assert marked_reentry_context.get("marked_reentry_conditions")
    assert len(proposal.get("bridge_event_summary") or []) == 5
    assert len(prepared_bridge_request.get("bridge_session", {}).get("approved_bridge_events") or []) == 5
    assert bridge_debug.get("feasibility_decisions")
    assert prepared_bridge_request.get("bridge_safety_context")
    assert all(
        (event.get("part_name") or "") != "None"
        for event in (prepared_bridge_request.get("bridge_session", {}).get("approved_bridge_events") or [])
        if isinstance(event, dict)
    )
    assert prepared_bridge_request.get("bridge_session", {}).get("phase") == "review"
    assert len(compiled_tasks) == 5
    assert all(task.get("function_name") == "execute_recovery_macro" for task in compiled_tasks)
    assert compiled_tasks[0].get("predecessors") == [ANCHOR_TASK_ID]


def _assert_expected_output_generic(result: dict[str, Any]) -> None:
    """Lightweight assertions for diverse test variants — validates structure, not exact content."""
    proposal = result["proposal"]
    compiled_tasks = result["compiled_tasks"]
    bridge_debug = result["bridge_debug"]

    assert isinstance(proposal, dict) and proposal
    assert proposal.get("plan_rewrite", {}).get("replace_failed_branch") is True
    macro_tasks = proposal.get("macro_tasks") or []
    assert len(macro_tasks) >= 1, "Expected at least 1 macro task"
    assert len(compiled_tasks) >= 1, "Expected at least 1 compiled task"
    assert bridge_debug.get("status") == "accepted"
    assert bridge_debug.get("compile_path") in {"deterministic", "llm_repair"}
    turns = bridge_debug.get("turns") or []
    assert len(turns) >= 2, "Expected at least 2 turns (observe + bridge_events)"


def _assert_expected_output_main_v2(result: dict[str, Any]) -> None:
    proposal = result["proposal"]
    compiled_tasks = result["compiled_tasks"]
    bridge_debug = result["bridge_debug"]
    prepared_bridge_request = result["prepared_bridge_request"]

    assert isinstance(proposal, dict) and proposal
    assert proposal.get("plan_rewrite", {}).get("replace_failed_branch") is True
    assert proposal.get("plan_rewrite", {}).get("resume_task_ids") == [
        "REQ_1_T3",
        "REQ_1_T4",
        "REQ_1_T5",
    ]
    assert proposal.get("primary_obligation", {}).get("resource_jid") == "xarm6@localhost"
    bridge_events = proposal.get("bridge_event_summary") or []
    assert len(bridge_events) == 2
    lg_events = [
        event for event in bridge_events
        if isinstance(event, dict) and str(event.get("part_name") or "").strip() == "LG"
    ]
    assert len(lg_events) == 1
    assert lg_events[0].get("resource_jid") == "xarm6@localhost"
    assert not any(
        str(event.get("part_name") or "").strip() == "MCP"
        for event in bridge_events
        if isinstance(event, dict)
    )
    assert bridge_debug.get("status") == "accepted"
    assert bridge_debug.get("compile_path") in {"deterministic", "llm_repair"}
    assert len(bridge_debug.get("turns") or []) == 3
    assert (
        prepared_bridge_request["grounding_context"]["step_outputs"]["detected_lg"]["pose"]["y"]
        == -0.5
    )
    assert prepared_bridge_request.get("bridge_session", {}).get("phase") == "review"
    assert len(compiled_tasks) == 2
    assert all(task.get("function_name") == "execute_recovery_macro" for task in compiled_tasks)
    assert compiled_tasks[0].get("predecessors") == [ANCHOR_TASK_ID]
    macro_resources = [
        str(task.get("params", {}).get("part_name") or "")
        for task in compiled_tasks
        if str(task.get("params", {}).get("part_name") or "").strip()
    ]
    assert macro_resources == ["LG"]


def _assert_live_output(result: dict[str, Any]) -> None:
    proposal = result["proposal"]
    compiled_tasks = result["compiled_tasks"]
    bridge_debug = result["bridge_debug"]
    prepared_bridge_request = result["prepared_bridge_request"]

    assert isinstance(proposal, dict) and proposal
    assert bridge_debug.get("status") == "accepted"
    assert bridge_debug.get("marked_reentry_check", {}).get("accepted") is True
    assert bridge_debug.get("compile_path") in {"deterministic", "llm_repair"}
    turn_types = [
        str((turn.get("normalized_response") or {}).get("type") or "")
        for turn in (bridge_debug.get("turns") or [])
        if isinstance(turn, dict)
    ]
    assert len(turn_types) >= 3
    assert "bridge_events" in turn_types
    assert "observe" in turn_types
    assert turn_types[-1] == "final_plan"
    assert turn_types.index("observe") < len(turn_types) - 1
    assert turn_types.index("bridge_events") < len(turn_types) - 1
    assert len(compiled_tasks) >= 1
    assert proposal.get("bridge_event_summary")
    assert proposal.get("plan_rewrite", {}).get("replace_failed_branch") is True
    assert proposal.get("plan_rewrite", {}).get("resume_task_ids") == [
        "REQ_1_T3",
        "REQ_1_T4",
        "REQ_1_T5",
    ]
    assert all(task.get("function_name") == "execute_recovery_macro" for task in compiled_tasks)
    assert compiled_tasks[0].get("predecessors") == [ANCHOR_TASK_ID]
    assert prepared_bridge_request.get("bridge_session", {}).get("turn_index", 0) >= 1
    assert prepared_bridge_request.get("bridge_session", {}).get("phase") == "review"
    assert bridge_debug.get("feasibility_decisions")
    assert prepared_bridge_request.get("bridge_safety_context")
    assert all(
        (event.get("part_name") or "") != "None"
        for event in (prepared_bridge_request.get("bridge_session", {}).get("approved_bridge_events") or [])
        if isinstance(event, dict)
    )
    assert prepared_bridge_request.get("marked_reentry_context") or prepared_bridge_request.get(
        "continuation_context"
    )


def _prepare_case3_harness_state(
    *,
    llm_mode: str,
    llm_model: str | None,
    scenario_overrides: dict[str, Any] | None = None,
    variant: str = MAIN_V1_VARIANT,
    extra_resources: list[Any] | None = None,
) -> tuple[dict[str, Any], FakeProductAgent, ProcessPlanner, dict[str, Any], list[dict[str, Any]]]:
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
        raise TypeError("geometry payload did not decode to an object")

    fixture = _runtime_fixture()
    robot_specs = _case3_robot_specs()
    variant_config = _case3_variant_config(variant)
    combined_overrides = deepcopy(variant_config.get("scenario_overrides") or {})
    if scenario_overrides:
        _deep_merge_dict(combined_overrides, scenario_overrides)
    ra_jid, scripted_turns_override = _apply_case3_scenario_overrides(
        fixture,
        robot_specs,
        combined_overrides,
    )
    product_agent = FakeProductAgent(
        tools_catalog=tools_catalog,
        product_geometry=deepcopy(geometry_payload.get("gazebo") or {}),
        scripted_turns=(
            scripted_turns_override
            if llm_mode == "scripted" and scripted_turns_override is not None
            else _scripted_turns() if llm_mode == "scripted" else None
        ),
        llm_mode=llm_mode,
        llm_model=llm_model,
        scenario_id=str(variant_config.get("scenario_id") or SCENARIO_ID),
        scenario_variant=str(variant_config.get("variant") or MAIN_V1_VARIANT),
        scripted_final_plan_builder=variant_config.get("scripted_final_plan_builder"),
    )
    ur5e_spec = robot_specs["ur5e@localhost"]
    ur5e = FakeBridgeRobot(
        config=ur5e_config,
        execution_env="gazebo",
        current_state=str(ur5e_spec.get("current_state") or ""),
        held_part=ur5e_spec.get("held_part"),
        gripper_state=str(ur5e_spec.get("gripper_state") or ""),
        pose_ref=ur5e_spec.get("pose_ref"),
        position=deepcopy(ur5e_spec.get("position") or {}),
        observations=deepcopy(ur5e_spec.get("observations") or {}),
    )
    xarm6_spec = robot_specs["xarm6@localhost"]
    xarm6 = FakeBridgeRobot(
        config=xarm6_config,
        execution_env="gazebo",
        current_state=str(xarm6_spec.get("current_state") or ""),
        held_part=xarm6_spec.get("held_part"),
        gripper_state=str(xarm6_spec.get("gripper_state") or ""),
        pose_ref=xarm6_spec.get("pose_ref"),
        position=deepcopy(xarm6_spec.get("position") or {}),
        observations=deepcopy(xarm6_spec.get("observations") or {}),
    )

    planner = ProcessPlanner(product_agent, [ur5e, xarm6, *(extra_resources or [])])
    planner.nodes = deepcopy(plan_payload.get("nodes") or [])

    async def _direct_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    with patch(
        "cais_spade_llm.agents.intelligent_product.process_planner.asyncio.to_thread",
        new=_direct_to_thread,
    ):
        prepared_bridge_request = asyncio.run(
            planner.prepare_bridge_session(
                stuck_state=deepcopy(fixture["stuck_state"]),
                P_id=deepcopy(fixture["P_id"]),
                ra_jid=ra_jid,
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
            )
        )
    product_agent.prepared_bridge_request = prepared_bridge_request
    _relax_recovery_clear_precondition(prepared_bridge_request)
    _apply_prepared_bridge_request_overrides(
        planner,
        prepared_bridge_request,
        combined_overrides,
    )
    if llm_mode == "live":
        _configure_live_bridge_session(prepared_bridge_request)

    return fixture, product_agent, planner, prepared_bridge_request, deepcopy(tools_catalog)


def run_case3_recovery_dry_run(
    write_debug: bool = True,
    *,
    variant: str = MAIN_V1_VARIANT,
) -> dict[str, Any]:
    return _run_case3_recovery_harness(
        write_debug=write_debug,
        llm_mode="scripted",
        llm_model=None,
        variant=variant,
    )


def run_case3_recovery_live_react(
    write_debug: bool = True,
    *,
    llm_model: str | None = None,
    variant: str = MAIN_V1_VARIANT,
) -> dict[str, Any]:
    return _run_case3_recovery_harness(
        write_debug=write_debug,
        llm_mode="live",
        llm_model=llm_model,
        variant=variant,
    )


def run_case3_recovery_dry_run_main_v1(write_debug: bool = True) -> dict[str, Any]:
    return run_case3_recovery_dry_run(write_debug=write_debug, variant=MAIN_V1_VARIANT)


def run_case3_recovery_dry_run_main_v2(write_debug: bool = True) -> dict[str, Any]:
    return run_case3_recovery_dry_run(write_debug=write_debug, variant=MAIN_V2_VARIANT)


def _run_case3_recovery_harness(
    *,
    write_debug: bool,
    llm_mode: str,
    llm_model: str | None,
    variant: str = MAIN_V1_VARIANT,
) -> dict[str, Any]:
    fixture, product_agent, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode=llm_mode,
        llm_model=llm_model,
        variant=variant,
    )
    proposal: dict[str, Any] | None = None
    compiled_tasks: list[dict[str, Any]] = []
    artifact_path: Path | None = None
    bridge_debug: dict[str, Any] = {}
    try:
        proposal = asyncio.run(planner.run_bridge_react_session(prepared_bridge_request))
        if not isinstance(proposal, dict):
            bridge_debug = planner.get_last_bridge_debug()
            if write_debug:
                artifact_path = _write_debug_artifact(
                    _build_debug_payload(
                        llm_mode=llm_mode,
                        product_agent=product_agent,
                        fixture=fixture,
                        proposal=proposal,
                        compiled_tasks=compiled_tasks,
                        bridge_debug=bridge_debug,
                        prepared_bridge_request=prepared_bridge_request,
                        error="bridge session did not return a validated proposal",
                    ),
                    variant=product_agent.scenario_variant,
                )
            validation_feedback = (
                (bridge_debug.get("session") or {}).get("validation_feedback")
                if isinstance(bridge_debug, dict)
                else []
            )
            raise AssertionError(
                "bridge session did not return a validated proposal"
                + (
                    f"; debug_artifact={artifact_path}"
                    if artifact_path is not None
                    else ""
                )
                + (
                    f"; validation_feedback={validation_feedback}"
                    if validation_feedback
                    else ""
                )
            )
        compiled_tasks = planner.apply_bridge_macro_proposal(
            proposal,
            anchor_task_id=ANCHOR_TASK_ID,
        )
        bridge_debug = planner.get_last_bridge_debug()
        if write_debug:
            artifact_path = _write_debug_artifact(
                _build_debug_payload(
                    llm_mode=llm_mode,
                    product_agent=product_agent,
                    fixture=fixture,
                    proposal=proposal,
                    compiled_tasks=compiled_tasks,
                    bridge_debug=bridge_debug,
                    prepared_bridge_request=prepared_bridge_request,
                ),
                variant=product_agent.scenario_variant,
            )
    except Exception as exc:
        bridge_debug = planner.get_last_bridge_debug()
        if write_debug:
            artifact_path = _write_debug_artifact(
                _build_debug_payload(
                    llm_mode=llm_mode,
                    product_agent=product_agent,
                    fixture=fixture,
                    proposal=proposal,
                    compiled_tasks=compiled_tasks,
                    bridge_debug=bridge_debug,
                    prepared_bridge_request=prepared_bridge_request,
                    error=repr(exc),
                ),
                variant=product_agent.scenario_variant,
            )
        raise

    result = {
        "proposal": proposal,
        "raw_final_plan": deepcopy(
            (product_agent.turn_log[-1].get("response") or {}).get("plan")
            if product_agent.turn_log and isinstance(product_agent.turn_log[-1].get("response"), dict)
            else {}
        ),
        "compiled_tasks": compiled_tasks,
        "bridge_debug": bridge_debug,
        "prepared_bridge_request": prepared_bridge_request,
        "artifact_path": str(artifact_path) if artifact_path else "",
        "turn_log": deepcopy(product_agent.turn_log),
        "llm_mode": llm_mode,
        "llm_model": product_agent.llm_model,
        "scenario_id": product_agent.scenario_id,
        "scenario_variant": product_agent.scenario_variant,
    }
    if llm_mode == "scripted":
        _assert_expected_output(result)
    else:
        _assert_live_output(result)
    return result


def run_test(
    *,
    llm_mode: str = "scripted",
    llm_model: str | None = None,
    write_debug: bool = True,
    variant: str = MAIN_V1_VARIANT,
) -> dict[str, Any]:
    if llm_mode == "live":
        result = run_case3_recovery_live_react(
            write_debug=write_debug,
            llm_model=llm_model,
            variant=variant,
        )
    else:
        result = run_case3_recovery_dry_run(write_debug=write_debug, variant=variant)
    proposal = result["proposal"]
    compiled_tasks = result["compiled_tasks"]
    bridge_debug = result["bridge_debug"]
    macro_names = [
        str(task.get("macro_name") or "")
        for task in (proposal.get("macro_tasks") or [])
        if isinstance(task, dict)
    ]
    resume_ids = proposal.get("plan_rewrite", {}).get("resume_task_ids") or []
    compiled_task_ids = [str(task.get("id") or "") for task in compiled_tasks]
    turn_types = []
    for turn in bridge_debug.get("turns") or []:
        normalized = turn.get("normalized_response") or {}
        turn_types.append(
            normalized.get("type") or ("accepted_final_plan" if turn.get("accepted") else "unknown")
        )
    print("Variant:", result["scenario_variant"], f"(scenario_id={result['scenario_id']})")
    print("Mode:", llm_mode, f"(model={result['llm_model']})")
    print("ReAct turn types:", turn_types)
    print("Accepted macro order:", macro_names)
    print("Resume task ids:", resume_ids)
    print("Compiled bridge task ids:", compiled_task_ids)
    print("Debug artifact:", result["artifact_path"])
    return result


def test_case3_recovery_process_dry_run() -> None:
    result = run_case3_recovery_dry_run(write_debug=True)
    assert result["proposal"] is not None


def test_case3_recovery_process_dry_run_main_v2() -> None:
    result = run_case3_recovery_dry_run(write_debug=False, variant=MAIN_V2_VARIANT)
    assert result["proposal"] is not None
    assert result["scenario_variant"] == MAIN_V2_VARIANT


@pytest.mark.parametrize(
    "variant",
    [BOTH_REACHABLE_VARIANT, NO_GRIPPER_CONFLICT_VARIANT],
)
def test_case3_diverse_variants_compile_successfully(variant: str) -> None:
    result = run_case3_recovery_dry_run(write_debug=False, variant=variant)
    assert result["proposal"] is not None
    assert result["scenario_variant"] == variant
    compiled_tasks = result.get("compiled_tasks") or []
    assert len(compiled_tasks) >= 1, f"Expected at least 1 compiled task for {variant}"


def test_robot_bridge_snapshot_exposes_resource_core_and_manipulator_facet() -> None:
    robot_specs = _case3_robot_specs()["ur5e@localhost"]
    ur5e_config = _load_robot_config(_case3_paths()["ur5e"], "ur5e")
    robot = FakeBridgeRobot(
        config=ur5e_config,
        execution_env="gazebo",
        current_state=str(robot_specs.get("current_state") or ""),
        held_part=robot_specs.get("held_part"),
        gripper_state=str(robot_specs.get("gripper_state") or ""),
        pose_ref=robot_specs.get("pose_ref"),
        position=deepcopy(robot_specs.get("position") or {}),
        observations=deepcopy(robot_specs.get("observations") or {}),
    )
    snapshot = robot.get_bridge_snapshot()
    assert snapshot.get("resource_core", {}).get("resource_type") == "robot"
    assert snapshot.get("resource_core", {}).get("current_state") == "picked"
    assert snapshot.get("resource_facets", {}).get("manipulator", {}).get("held_part") == "MCP"
    assert snapshot.get("held_part") == "MCP"


def test_printer_bridge_snapshot_exposes_resource_core_and_printer_facet() -> None:
    printer = FakeBridgePrinter(
        jid="printer@localhost",
        current_state="printing",
        current_location="printer_bay",
        active_job="JOB_42",
        bed_state="loaded",
        material_state="pla_ready",
    )
    snapshot = printer.get_bridge_snapshot()
    assert snapshot.get("resource_core", {}).get("resource_type") == "printer"
    assert snapshot.get("resource_core", {}).get("current_state") == "printing"
    assert snapshot.get("resource_core", {}).get("active_work") == "JOB_42"
    assert snapshot.get("resource_facets", {}).get("printer", {}).get("active_job") == "JOB_42"
    assert snapshot.get("gripper_state") is None


def test_cca_replan_message_includes_filtered_bridge_safety_rules() -> None:
    secondary_rule = _synthetic_mutex_rule(
        rule_id="RULE_UNUSED",
        destination="shared_buffer-v3",
        resources=["ur5e", "xarm6"],
    )
    cca = CentralControllerAgent.__new__(CentralControllerAgent)
    cca.plan_fsa_monitor = None
    cca.resource_agents = []
    cca.safety_monitor = None
    cca.safety_rules = [deepcopy(CASE3_BOARD_MUTEX_RULE), secondary_rule]
    cca.logger = logging.getLogger("case3_recovery_main.cca_test")
    cca._collect_system_coordination_state = lambda: {"resource_states": {}}
    cca._build_obligation_targets = lambda **_: []

    message = CentralControllerAgent._build_replan_message(
        cca,
        product_jid="assembly_board-v1@localhost",
        reason="safety_violation",
        event={"task_id": FAILED_TASK_ID},
        safety_info={
            "rule_ids": [CASE3_BOARD_MUTEX_RULE_ID],
            "bridge_safety_context": {
                "rule_ids": [CASE3_BOARD_MUTEX_RULE_ID],
                "constraints": [],
            },
        },
    )
    payload = json.loads(message.body)
    bridge_safety_context = payload.get("safety_ctx", {}).get("bridge_safety_context", {})
    assert bridge_safety_context.get("rule_ids") == [CASE3_BOARD_MUTEX_RULE_ID]
    safety_rules = bridge_safety_context.get("safety_rules") or []
    assert [str(rule.get("id") or "") for rule in safety_rules] == [CASE3_BOARD_MUTEX_RULE_ID]
    assert safety_rules[0].get("context", {}).get("destination") == "assembly_board-v1"


def test_bridge_prompt_exposes_marked_reentry_gap() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    prompt = planner._build_bridge_turn_prompt_preview(prepared_bridge_request)
    assert "MARKED RE-ENTRY CONDITIONS" in prompt
    assert "Gamma(x_d, M_bridge)" in prompt
    assert "named_poses" not in prompt
    assert "side_insert_post" not in prompt
    assert "side_insert_pre" not in prompt
    assert "side_pick_approach" not in prompt
    assert "recovery_clear" not in prompt
    assert "/step_outputs/lg_pick_targets/approach_pose/x" not in prompt
    assert "parts.LG.observed_pose.x" not in prompt
    assert 'parts.<PART>.observed_pose.x' in prompt
    assert '"/step_outputs/<alias>/approach_pose/x"' in prompt
    assert "STRUCTURAL REPAIR EXAMPLE" not in prompt
    marked_reentry = prepared_bridge_request.get("marked_reentry_context") or {}
    assert any(
        cond.get("entity") == "LG"
        and cond.get("field") == "location"
        and cond.get("expected") == "assembly_board-v1"
        for cond in (marked_reentry.get("marked_reentry_conditions") or [])
        if isinstance(cond, dict)
    )


def test_bridge_events_prompt_uses_dynamic_state_safety_and_retry_hints() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    _record_bridge_observation(
        planner,
        prepared_bridge_request,
        resource_jid="ur5e@localhost",
        part_name="LG",
        pose=LG_DROP_POSE,
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["validation_feedback"] = [
        {
            "kind": "bridge_events_rejected",
            "message": (
                "bridge event 'xarm6_pick_LG_from_recovery_zone' is infeasible on "
                "xarm6@localhost: pose outside workspace: y=0.1980 > y_max_m=0.1000"
            ),
        }
    ]
    prepared_bridge_request["bridge_session"] = bridge_session
    planner._refresh_bridge_grounding_context(prepared_bridge_request)

    prompt = planner._build_bridge_turn_prompt_preview(prepared_bridge_request)

    assert "Current planner phase: bridge_events." in prompt
    assert "Constraints detected in current state:" in prompt
    assert "Resource xarm6@localhost is in state 'recovery_required'." in prompt
    assert "Resource ur5e@localhost currently holds 'MCP'." in prompt
    assert "Safety constraint: part 'MCP' must not be placed at 'assembly_board-v1'" in prompt
    assert "assembly_board-v1" in prompt
    assert "Prior proposal rejected:" in prompt
    assert "pose outside workspace" in prompt
    assert "Do not re-propose the same resource for that operation" in prompt
    assert 'A resource in state "recovery_required" cannot perform pick/place actions' not in prompt
    assert '"safety_rules"' in prompt
    assert CASE3_BOARD_MUTEX_RULE_ID in prompt
    assert '"bridge_constraint_family"' not in prompt
    assert '"constraint_type"' in prompt
    # Proactive reachability verdict
    assert "Part 'LG' at observed pose:" in prompt
    assert "reachable by ur5e@localhost" in prompt
    assert "NOT reachable by xarm6@localhost" in prompt


def test_bridge_prompt_repair_example_only_appears_in_repair_mode() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["observation_history"] = [
        {
            "turn_index": 1,
            "resource_jid": "ur5e@localhost",
            "primitive": "detect_parts",
            "params": {"part_name": "LG"},
            "store_as": "detected_lg",
            "observation": {"part_name": "LG", "pose": deepcopy(LG_DROP_POSE)},
        }
    ]
    bridge_session["approved_bridge_events"] = deepcopy(_scripted_turns()[1]["events"])
    bridge_session["draft_final_plan"] = {"macro_tasks": [{"macro_name": "draft_macro"}]}
    bridge_session["draft_final_plan_status"] = {
        "compile_path": "llm_repair",
        "status": "compile_failed",
        "error": "draft invalid",
    }
    prepared_bridge_request["bridge_session"] = bridge_session
    planner._refresh_bridge_grounding_context(prepared_bridge_request)
    prompt = planner._build_bridge_turn_prompt_preview(prepared_bridge_request)
    assert "GENERIC FINAL-PLAN SHAPE EXAMPLE" in prompt
    assert "MANIPULATOR PICK/PLACE REPAIR EXAMPLE" in prompt
    assert "<PART>" in prompt
    assert "<RESOURCE_JID>" in prompt
    assert "named_poses/home" not in prompt


def test_mixed_robot_printer_bridge_session_surfaces_printer_as_context_only() -> None:
    printer = FakeBridgePrinter(
        jid="printer@localhost",
        current_state="printing",
        current_location="printer_bay",
        active_job="JOB_42",
        bed_state="loaded",
        material_state="pla_ready",
    )
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
        extra_resources=[printer],
    )
    bridge_resources = prepared_bridge_request.get("bridge_resources") or {}
    printer_entry = dict(bridge_resources.get("printer@localhost") or {})
    assert printer_entry.get("resource_type") == "printer"
    assert printer_entry.get("bridge_adapter", {}).get("supports_executable_bridge") is False
    assert printer_entry.get("primitive_catalog") == []
    assert printer_entry.get("bridge_snapshot", {}).get("resource_core", {}).get("active_work") == "JOB_42"
    assert (
        prepared_bridge_request.get("grounding_context", {})
        .get("resources", {})
        .get("printer@localhost", {})
        .get("resource_facets", {})
        .get("printer", {})
        .get("active_job")
        == "JOB_42"
    )
    prompt = planner._build_bridge_turn_prompt_preview(prepared_bridge_request)
    assert "printer@localhost" in prompt
    assert '"resource_type": "printer"' in prompt
    assert '"active_job": "JOB_42"' in prompt


def test_mixed_robot_printer_repair_prompt_includes_manipulator_example_once() -> None:
    printer = FakeBridgePrinter(
        jid="printer@localhost",
        current_state="printing",
        current_location="printer_bay",
        active_job="JOB_42",
        bed_state="loaded",
        material_state="pla_ready",
    )
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
        extra_resources=[printer],
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["observation_history"] = [
        {
            "turn_index": 1,
            "resource_jid": "ur5e@localhost",
            "primitive": "detect_parts",
            "params": {"part_name": "LG"},
            "store_as": "detected_lg",
            "observation": {"part_name": "LG", "pose": deepcopy(LG_DROP_POSE)},
        }
    ]
    bridge_session["approved_bridge_events"] = deepcopy(_scripted_turns()[1]["events"])
    bridge_session["draft_final_plan"] = {"macro_tasks": [{"macro_name": "draft_macro"}]}
    bridge_session["draft_final_plan_status"] = {
        "compile_path": "llm_repair",
        "status": "compile_failed",
        "error": "draft invalid",
    }
    prepared_bridge_request["bridge_session"] = bridge_session
    planner._refresh_bridge_grounding_context(prepared_bridge_request)
    prompt = planner._build_bridge_turn_prompt_preview(prepared_bridge_request)
    assert "GENERIC FINAL-PLAN SHAPE EXAMPLE" in prompt
    assert prompt.count("MANIPULATOR PICK/PLACE REPAIR EXAMPLE") == 1


def test_printer_only_repair_prompt_omits_manipulator_example() -> None:
    printer = FakeBridgePrinter(
        jid="printer@localhost",
        current_state="printing",
        current_location="printer_bay",
        active_job="JOB_42",
        bed_state="loaded",
        material_state="pla_ready",
    )
    printer_snapshot = printer.get_bridge_snapshot()
    prompt = build_bridge_turn_prompt(
        session_id="bridge_test",
        turn_index=2,
        max_turns=6,
        phase="final_plan",
        focused_resource_jid="printer@localhost",
        stuck_state={"resource_state": "printing"},
        goal_state="ready",
        pending_parts=[],
        obligation_targets=[],
        bridge_resources={
            "printer@localhost": {
                "resource_jid": "printer@localhost",
                "resource_type": "printer",
                "primitive_catalog": [],
                "bridge_snapshot": deepcopy(printer_snapshot),
                "resource_core": deepcopy(printer_snapshot.get("resource_core") or {}),
                "resource_facets": deepcopy(printer_snapshot.get("resource_facets") or {}),
                "bridge_adapter": bridge_adapter_capabilities("printer"),
            }
        },
        grounding_context={
            "resource": {
                "jid": "printer@localhost",
                "resource_core": deepcopy(printer_snapshot.get("resource_core") or {}),
                "resource_facets": deepcopy(printer_snapshot.get("resource_facets") or {}),
            },
            "resources": {
                "printer@localhost": {
                    "jid": "printer@localhost",
                    "resource_core": deepcopy(printer_snapshot.get("resource_core") or {}),
                    "resource_facets": deepcopy(printer_snapshot.get("resource_facets") or {}),
                }
            },
            "parts": {},
        },
        observation_history=[],
        operator_feedback_history=[],
        validation_feedback=[],
        marked_reentry_conditions=[],
        unmet_reentry_conditions=[],
        pending_suffix_summary=[],
        last_plan_failure={},
        allowed_observation_primitives=[],
        approved_bridge_events=[
            {
                "event_name": "printer_clear_for_recovery",
                "resource_jid": "printer@localhost",
                "closes_conditions": [
                    {
                        "entity_kind": "resource",
                        "entity": "printer@localhost",
                        "field": "current_state",
                        "expected": "idle",
                    }
                ],
                "rationale": "Clear the printer work queue.",
            }
        ],
        bridge_safety_context={},
        draft_final_plan={"macro_tasks": [{"macro_name": "draft_macro"}]},
        draft_final_plan_status={
            "compile_path": "llm_repair",
            "status": "compile_failed",
            "error": "draft invalid",
        },
    )
    assert "GENERIC FINAL-PLAN SHAPE EXAMPLE" in prompt
    assert "MANIPULATOR PICK/PLACE REPAIR EXAMPLE" not in prompt
    assert "Acquire <PART> from its observed location." not in prompt



def test_context_ref_aliases_for_step_outputs_are_supported() -> None:
    grounding_context = {
        "parts": {"PART_A": {"observed_pose": {"x": 0.002, "y": 0.198, "z": 1.034}}},
        "step_outputs": {
            "part_pick_targets": {
                "approach_pose": {"x": 0.11, "y": 0.22, "z": 1.33},
            }
        }
    }
    assert resolve_context_ref("part_pick_targets.approach_pose.x", grounding_context) == 0.11
    assert resolve_context_ref("/step_outputs/PartPickTargets/approach_pose/x", grounding_context) == 0.11
    preserved = resolve_param_refs(
        {"x": {"context_ref": "part_pick_targets.approach_pose.x"}},
        {"parts": grounding_context["parts"]},
        preserve_step_output_refs=True,
    )
    assert preserved == {"x": {"context_ref": "part_pick_targets.approach_pose.x"}}
    assert resolve_param_refs(
        {"x": "/step_outputs/part_pick_targets/approach_pose/x"},
        {"parts": grounding_context["parts"]},
        step_outputs=grounding_context["step_outputs"],
    ) == {"x": 0.11}
    assert resolve_param_refs(
        {"model_name": "parts.PART_A.target.model_name"},
        {
            "parts": {
                "PART_A": {
                    "target": {"model_name": "test_part_model"},
                }
            }
        },
    ) == {"model_name": "test_part_model"}
    assert resolve_param_refs(
        {"pick_ctx": "/step_outputs/part_pick_targets"},
        {"parts": grounding_context["parts"]},
        step_outputs=grounding_context["step_outputs"],
    ) == {"pick_ctx": grounding_context["step_outputs"]["part_pick_targets"]}
    assert resolve_param_refs(
        {"pick_ctx": {"context_ref": "/step_outputs/part_pick_targets"}},
        {"parts": grounding_context["parts"]},
        step_outputs=grounding_context["step_outputs"],
    ) == {"pick_ctx": grounding_context["step_outputs"]["part_pick_targets"]}
    assert resolve_param_refs(
        {"x": {"context_ref": "/step_outputs/DetectedPart/pose/x"}},
        {"parts": grounding_context["parts"]},
        step_outputs={
            "detected_part": {"pose": {"x": 0.002, "y": 0.198, "z": 1.034}},
        },
    ) == {"x": 0.002}


def test_pick_and_place_preview_outputs_expose_pose_aliases() -> None:
    robot_bridge_snapshot = canonical_bridge_resource(
        resource_jid="robot@localhost",
        resource_type="robot",
        snapshot={
            "resource_type": "robot",
            "current_state": "idle",
            "current_pose": {"x": -0.25, "y": 0.22, "z": 1.18},
        },
        modeled_state={},
    )

    pick_output, pick_error = preview_step_output(
        primitive="compute_pick_targets",
        params={
            "part_name": "LG",
            "product_geometry": {
                "board_center": {"x": 0.0, "y": 0.0, "z": 1.025},
                "slot_floor_z_m": 1.025,
                "slot_xy": [0.1, 0.08],
                "part_height_m": 0.02,
                "model_name": "gear_large",
            },
        },
        snapshot=robot_bridge_snapshot,
        grounding_context={"parts": {"LG": {"observed_pose": deepcopy(LG_DROP_POSE)}}},
    )
    assert pick_error is None
    assert pick_output["approach_pose"]["x"] == pick_output["tx"]
    assert pick_output["target_pose"]["z"] == pick_output["pick_z"]

    place_output, place_error = preview_step_output(
        primitive="compute_place_targets",
        params={
            "part_name": "LG",
            "product_geometry": {
                "board_center": {"x": 0.0, "y": 0.0, "z": 1.025},
                "slot_floor_z_m": 1.025,
                "slot_xy": [0.1, 0.08],
                "part_height_m": 0.02,
                "model_name": "gear_large",
            },
            "pick_ctx": pick_output,
        },
        snapshot=robot_bridge_snapshot,
        grounding_context={},
    )
    assert place_error is None
    assert "approach_pose" in place_output
    assert "target_pose" in place_output


def test_safety_only_bridge_is_rejected_by_marked_reentry_gap() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    try:
        planner.validate_preprogrammed_bridge_proposal(
            proposal=_build_safety_only_bridge_plan(),
            prepared_bridge_request=prepared_bridge_request,
            source="test",
            scenario_id="safety_only_bridge",
        )
    except ValueError:
        bridge_debug = planner.get_last_bridge_debug()
        marked_check = bridge_debug.get("marked_reentry_check") or {}
        assert marked_check.get("accepted") is False
        assert "marked re-entry" in str(marked_check.get("reason") or "").lower()
        return

    raise AssertionError("safety-only bridge unexpectedly satisfied the marked re-entry gate")


def test_bridge_events_require_phase_order_and_feasible_resource() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    current_phase = planner._bridge_current_phase(prepared_bridge_request)
    assert current_phase == "observe_required"

    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["observation_history"] = [
        {
            "turn_index": 1,
            "resource_jid": "ur5e@localhost",
            "primitive": "detect_parts",
            "params": {"part_name": "LG"},
            "store_as": "detected_lg",
            "observation": {"part_name": "LG", "pose": deepcopy(LG_DROP_POSE)},
        }
    ]
    prepared_bridge_request["bridge_session"] = bridge_session
    planner._refresh_bridge_grounding_context(prepared_bridge_request)
    assert planner._bridge_current_phase(prepared_bridge_request) == "bridge_events"

    approved_events, feasibility_decisions, error = planner._bridge_validate_bridge_events(
        prepared_bridge_request,
        events=[
            {
                "event_name": "xarm6_pick_and_insert_lg",
                "resource_jid": "xarm6@localhost",
                "part_name": "LG",
                "closes_conditions": [
                    {
                        "entity_kind": "resource",
                        "entity": "xarm6@localhost",
                        "field": "current_state",
                        "expected": "idle",
                    },
                    {
                        "entity_kind": "part",
                        "entity": "LG",
                        "field": "state",
                        "expected": "assembled",
                    },
                    {
                        "entity_kind": "part",
                        "entity": "LG",
                        "field": "location",
                        "expected": "assembly_board-v1",
                    },
                ],
                "rationale": "Incorrectly assign LG recovery to xarm6.",
            }
        ],
    )
    assert approved_events is None
    assert error is not None
    assert "infeasible" in error.lower()
    assert any(
        decision.get("resource_jid") == "xarm6@localhost"
        and decision.get("allowed") is False
        for decision in feasibility_decisions
    )


def test_bridge_safety_constraints_reject_protected_goal_staging_before_bridge_complete() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    planner._refresh_bridge_grounding_context(prepared_bridge_request)
    safety_ctx = prepared_bridge_request.get("bridge_safety_context") or {}
    assert any(
        str(item.get("part_name") or "").strip() == "MCP"
        and str(item.get("forbidden_location") or "").strip() == "assembly_board-v1"
        and isinstance(item.get("until_conditions"), list)
        for item in (safety_ctx.get("constraints") or [])
        if isinstance(item, dict)
    )
    ok, error = planner._bridge_validate_safety_constraints(
        prepared_bridge_request,
        events=[
            {
                "event_name": "stage_resume_part_to_goal",
                "resource_jid": "ur5e@localhost",
                "part_name": "MCP",
                "expected_resource_delta": {"from": "picked", "to": "idle"},
                "expected_part_delta": {
                    "part_name": "MCP",
                    "from": "in_gripper",
                    "to": "ready",
                    "location_to": "assembly_board-v1",
                },
            }
        ],
    )
    assert ok is False
    assert any(
        marker in str(error or "")
        for marker in ["cannot be moved", "cannot enter 'assembly_board-v1'"]
    )


def test_collect_bridge_safety_context_preserves_and_deduplicates_safety_rules() -> None:
    _, _, planner, _, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    violation_rule = deepcopy(CASE3_BOARD_MUTEX_RULE)
    collected = planner._collect_bridge_safety_context(
        [
            {
                "safety_ctx": {
                    "bridge_safety_context": {
                        "rule_ids": [CASE3_BOARD_MUTEX_RULE_ID],
                        "constraints": [],
                        "safety_rules": [violation_rule, deepcopy(violation_rule)],
                    }
                }
            },
            {
                "safety_ctx": {
                    "bridge_safety_context": {
                        "rule_ids": [CASE3_BOARD_MUTEX_RULE_ID],
                        "safety_rules": [deepcopy(violation_rule)],
                    }
                }
            },
        ]
    )
    assert collected.get("rule_ids") == [CASE3_BOARD_MUTEX_RULE_ID]
    safety_rules = collected.get("safety_rules") or []
    assert len(safety_rules) == 1
    assert safety_rules[0].get("id") == CASE3_BOARD_MUTEX_RULE_ID
    assert safety_rules[0].get("context", {}).get("destination") == "assembly_board-v1"


def test_bridge_safety_constraints_include_runtime_mutex_rule() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    planner._refresh_bridge_grounding_context(prepared_bridge_request)
    safety_ctx = prepared_bridge_request.get("bridge_safety_context") or {}
    mutex_constraints = [
        item
        for item in (safety_ctx.get("constraints") or [])
        if isinstance(item, dict)
        and str(item.get("location") or "").strip() == "assembly_board-v1"
        and len(item.get("resource_jids") or []) >= 2
    ]
    assert any(
        str(item.get("rule_id") or "").strip() == CASE3_BOARD_MUTEX_RULE_ID
        and str(item.get("location") or "").strip() == "assembly_board-v1"
        and set(item.get("resource_jids") or []) == {"ur5e@localhost", "xarm6@localhost"}
        for item in mutex_constraints
    )


def test_bridge_mutex_translation_uses_runtime_rule_fields_without_hardcoding() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    synthetic_rule = _synthetic_mutex_rule(
        rule_id="RULE_MUTEX_CUSTOM_17",
        destination="shared_buffer-v3",
        resources=["ur5e", "xarm6"],
        constraint_type="no_simultaneous_presence_in_destination_area",
    )
    prepared_bridge_request["bridge_safety_context"] = {
        "rule_ids": [synthetic_rule["id"]],
        "constraints": [],
        "safety_rules": [deepcopy(synthetic_rule)],
    }
    derived = planner._derive_bridge_safety_constraints(
        prepared_bridge_request,
        marked_reentry_context=deepcopy(
            prepared_bridge_request.get("marked_reentry_context")
            or prepared_bridge_request.get("continuation_context")
            or {}
        ),
    )
    mutex_constraints = [
        item
        for item in (derived.get("constraints") or [])
        if isinstance(item, dict)
        and str(item.get("location") or "").strip() == synthetic_rule["context"]["destination"]
        and len(item.get("resource_jids") or []) >= 2
    ]
    assert len(mutex_constraints) == 1
    mutex_constraint = mutex_constraints[0]
    assert mutex_constraint.get("rule_id") == synthetic_rule["id"]
    assert mutex_constraint.get("constraint_type") == synthetic_rule["constraint_type"]
    assert mutex_constraint.get("location") == synthetic_rule["context"]["destination"]
    assert set(mutex_constraint.get("resource_jids") or []) == {
        "ur5e@localhost",
        "xarm6@localhost",
    }
    assert mutex_constraint.get("generated_interpretation") == synthetic_rule["generated_interpretation"]


def test_bridge_safety_constraints_reject_bad_mutex_ordering() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    planner._refresh_bridge_grounding_context(prepared_bridge_request)
    ok, error = planner._bridge_validate_safety_constraints(
        prepared_bridge_request,
        events=[
            _bridge_stage_event(resource_jid="ur5e@localhost", part_name="MCP"),
            _bridge_pick_place_event(resource_jid="ur5e@localhost", part_name="LG"),
            _bridge_clear_event(resource_jid="xarm6@localhost"),
        ],
    )
    assert ok is False
    assert CASE3_BOARD_MUTEX_RULE_ID in str(error or "")
    assert "assembly_board-v1" in str(error or "")
    assert "xarm6@localhost" in str(error or "")


def test_bridge_safety_constraints_accept_clear_before_conflicting_place() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    planner._refresh_bridge_grounding_context(prepared_bridge_request)
    ok, error = planner._bridge_validate_safety_constraints(
        prepared_bridge_request,
        events=[
            _bridge_clear_event(resource_jid="xarm6@localhost"),
            _bridge_stage_event(resource_jid="ur5e@localhost", part_name="MCP"),
            _bridge_pick_place_event(resource_jid="ur5e@localhost", part_name="LG"),
        ],
    )
    assert ok is True
    assert error is None


def test_bridge_events_reject_semantically_unrealizable_stage_before_final_plan() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    _record_bridge_observation(
        planner,
        prepared_bridge_request,
        resource_jid="ur5e@localhost",
        part_name="LG",
        pose=deepcopy(LG_DROP_POSE),
    )
    bridge_resources = deepcopy(prepared_bridge_request.get("bridge_resources") or {})
    ur5e_entry = dict(bridge_resources.get("ur5e@localhost") or {})
    ur5e_snapshot = resource_snapshot_set_field(
        ur5e_entry.get("bridge_snapshot") or {},
        "held_part",
        None,
    )
    ur5e_entry["bridge_snapshot"] = ur5e_snapshot
    bridge_resources["ur5e@localhost"] = ur5e_entry
    prepared_bridge_request["bridge_resources"] = bridge_resources
    planner._refresh_bridge_grounding_context(prepared_bridge_request)

    approved_events, _, error = planner._bridge_validate_bridge_events(
        prepared_bridge_request,
        events=[
            _bridge_clear_event(resource_jid="xarm6@localhost"),
            _bridge_stage_event(resource_jid="ur5e@localhost", part_name="MCP"),
            _bridge_pick_place_event(
                resource_jid="ur5e@localhost",
                part_name="LG",
                from_state="idle",
            ),
        ],
    )

    assert approved_events is None
    assert error is not None
    assert "must be carrying 'MCP' before it starts" in str(error)


def test_bridge_events_require_explicit_operation_family() -> None:
    response, error = normalize_bridge_turn_response(
        raw=json.dumps(
            {
                "type": "bridge_events",
                "events": [
                    {
                        "event_name": "ur5e_stage_MCP_to_prusa_mk4_2",
                        "resource_jid": "ur5e@localhost",
                        "part_name": "MCP",
                        "expected_resource_delta": {"from": "picked", "to": "idle"},
                        "expected_part_delta": {
                            "part_name": "MCP",
                            "from": "in_gripper",
                            "to": "ready",
                            "location_to": "prusa-mk4-2",
                        },
                    }
                ],
            }
        ),
        available_resource_jids=["ur5e@localhost"],
        allowed_observation_primitives=[],
    )

    assert response is None
    assert error == "bridge_events.events[].operation_family is required"


def test_bridge_events_reject_non_executable_printer_resource() -> None:
    printer = FakeBridgePrinter(
        jid="printer@localhost",
        current_state="printing",
        current_location="printer_bay",
        active_job="JOB_42",
    )
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
        extra_resources=[printer],
    )
    _record_bridge_observation(
        planner,
        prepared_bridge_request,
        resource_jid="ur5e@localhost",
        part_name="LG",
        pose=LG_DROP_POSE,
    )
    approved_events, feasibility_decisions, error = planner._bridge_validate_bridge_events(
        prepared_bridge_request,
        events=[
            {
                "event_name": "printer_clear_for_recovery",
                "resource_jid": "printer@localhost",
                "expected_resource_delta": {"from": "printing", "to": "idle"},
                "closes_conditions": [
                    {
                        "entity_kind": "resource",
                        "entity": "xarm6@localhost",
                        "field": "current_state",
                        "expected": "idle",
                    }
                ],
            }
        ],
    )
    assert approved_events is None
    assert "does not advertise executable bridge primitives" in str(error or "")
    assert any(
        decision.get("resource_jid") == "printer@localhost"
        and decision.get("allowed") is False
        for decision in feasibility_decisions
    )


# ---------------------------------------------------------------------------
# Printer bridge generalization tests (Step 5b)
# ---------------------------------------------------------------------------


class FakeBridgePrinterWithPrimitives:
    """Fake printer that advertises bridge primitives for catalog building."""

    _BRIDGE_PRIMITIVES: list[str] = ["pause_job", "resume_job", "cancel_job"]

    def __init__(
        self,
        *,
        jid: str = "printer@localhost",
        current_state: str = "idle",
        current_location: str | None = "printer_bay",
        active_job: str | None = None,
        job_state: str | None = None,
        bed_state: str | None = None,
        material_state: str | None = None,
    ) -> None:
        self.agent_name = str(jid)
        self.jid = str(jid)
        self.static_capabilities = {"resource_type": "printer"}
        self._current_state = str(current_state)
        self._current_location = current_location
        self._active_job = active_job
        self._job_state = job_state or current_state
        self._bed_state = bed_state
        self._material_state = material_state

    async def pause_job(self, *, job_id: str = "", **kwargs: Any) -> dict[str, Any]:
        """
        ---
        preconditions:
          current_state:
            equals: "printing"
        effects:
          current_state:
            set: "paused"
          job_state:
            set: "paused"
        ---
        Pause the active print job without cancelling it.
        """
        self._current_state = "paused"
        self._job_state = "paused"
        return {"success": True, "state": "paused"}

    async def resume_job(self, *, job_id: str = "", **kwargs: Any) -> dict[str, Any]:
        """
        ---
        preconditions:
          current_state:
            equals: "paused"
        effects:
          current_state:
            set: "printing"
          job_state:
            set: "printing"
        ---
        Resume a previously paused print job.
        """
        self._current_state = "printing"
        self._job_state = "printing"
        return {"success": True, "state": "printing"}

    async def cancel_job(self, *, job_id: str = "", **kwargs: Any) -> dict[str, Any]:
        """
        ---
        preconditions:
          current_state:
            not_equals: "idle"
        effects:
          current_state:
            set: "idle"
          active_job:
            set: null
          job_state:
            set: "idle"
        ---
        Cancel the active print job and return the printer to idle.
        """
        self._current_state = "idle"
        self._active_job = None
        self._job_state = "idle"
        return {"success": True, "state": "idle"}

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "resource_type": "printer",
            "agent_name": self.agent_name,
            "current_state": self._current_state,
            "current_location": self._current_location,
            "active_job": self._active_job,
            "job_state": self._job_state,
            "material_state": self._material_state,
            "bed_state": self._bed_state,
        }

    def get_bridge_snapshot(self) -> dict[str, Any]:
        return canonical_bridge_resource(
            resource_jid=self.jid,
            resource_type="printer",
            snapshot={
                "resource_type": "printer",
                "current_state": self._current_state,
                "current_location": self._current_location,
                "active_job": self._active_job,
                "job_state": self._job_state,
                "bed_state": self._bed_state,
                "material_state": self._material_state,
            },
            modeled_state={},
        )

    def bridge_feasibility_oracle(
        self,
        *,
        operation_kind: str = "",
        part_name: str | None = None,
        part_context: dict[str, Any] | None = None,
        bridge_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {"allowed": True, "reason": "fake oracle"}


def test_printer_with_primitives_has_nonempty_catalog() -> None:
    """Printer advertising _BRIDGE_PRIMITIVES yields a non-empty catalog."""
    printer = FakeBridgePrinterWithPrimitives(
        jid="printer@localhost",
        current_state="printing",
        active_job="JOB_42",
    )
    catalog = build_primitive_catalog(printer)
    assert len(catalog) > 0
    names = {entry["name"] for entry in catalog}
    assert names == {"pause_job", "resume_job", "cancel_job"}


def test_printer_with_primitives_capability_derived_executable() -> None:
    """Printer with non-empty catalog gets supports_executable_bridge=True."""
    printer = FakeBridgePrinterWithPrimitives(
        jid="printer@localhost",
        current_state="printing",
        active_job="JOB_42",
    )
    catalog = build_primitive_catalog(printer)
    caps = bridge_adapter_capabilities("printer", primitive_catalog=catalog)
    assert caps["supports_executable_bridge"] is True
    assert caps["supports_printer_job_control"] is True
    assert caps["supports_manipulator_pick_place"] is False


def test_snapshot_only_printer_not_executable() -> None:
    """Printer without _BRIDGE_PRIMITIVES has supports_executable_bridge=False."""
    printer = FakeBridgePrinter(
        jid="printer@localhost",
        current_state="printing",
        active_job="JOB_42",
    )
    catalog = build_primitive_catalog(printer)
    assert catalog == []
    caps = bridge_adapter_capabilities("printer", primitive_catalog=catalog)
    assert caps["supports_executable_bridge"] is False
    assert caps["supports_printer_job_control"] is False


def test_get_resource_bridge_snapshot_no_recursion_for_robot() -> None:
    """get_resource_bridge_snapshot for a robot builds snapshot directly, no infinite recursion."""
    robot_specs = _case3_robot_specs()["ur5e@localhost"]
    ur5e_config = _load_robot_config(_case3_paths()["ur5e"], "ur5e")
    robot = FakeBridgeRobot(
        config=ur5e_config,
        execution_env="gazebo",
        current_state=str(robot_specs.get("current_state") or ""),
        held_part=robot_specs.get("held_part"),
        gripper_state=str(robot_specs.get("gripper_state") or ""),
        pose_ref=robot_specs.get("pose_ref"),
        position=deepcopy(robot_specs.get("position") or {}),
        observations=deepcopy(robot_specs.get("observations") or {}),
    )
    snapshot = get_resource_bridge_snapshot(robot)
    assert isinstance(snapshot, dict)
    assert snapshot.get("resource_type") == "robot"
    assert "current_state" in snapshot.get("resource_core", {})


def test_sync_agent_from_bridge_snapshot_printer() -> None:
    """sync_agent_from_bridge_snapshot syncs printer-specific fields."""
    printer = FakeBridgePrinterWithPrimitives(
        jid="printer@localhost",
        current_state="printing",
        active_job="JOB_42",
        job_state="printing",
    )
    bridge_snapshot = {
        "current_state": "idle",
        "active_job": None,
        "job_state": "idle",
        "material_state": "pla_ready",
    }
    sync_agent_from_bridge_snapshot(printer, bridge_snapshot)
    assert printer._current_state == "idle"
    assert printer._active_job is None
    assert printer._job_state == "idle"
    assert printer._material_state == "pla_ready"


def test_base_class_execute_recovery_macro_dispatches_printer_primitive() -> None:
    """ResourceAgent.execute_recovery_macro dispatches to printer primitives."""
    from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent

    printer = FakeBridgePrinterWithPrimitives(
        jid="printer@localhost",
        current_state="printing",
        active_job="JOB_42",
        job_state="printing",
    )
    # Borrow the base-class executor (it uses getattr dispatch).
    result = asyncio.run(
        ResourceAgent.execute_recovery_macro(
            printer,
            macro_name="cancel_job_macro",
            primitive_steps=[
                {"primitive": "cancel_job", "params": {"job_id": "JOB_42"}},
            ],
        )
    )
    assert result.get("status") == "completed"
    assert printer._current_state == "idle"
    assert printer._active_job is None


def test_deterministic_bridge_compiler_supplies_place_part_name() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["approved_bridge_events"] = deepcopy(_scripted_turns()[1]["events"])
    bridge_session["observation_history"] = [
        {
            "turn_index": 1,
            "resource_jid": "ur5e@localhost",
            "primitive": "detect_parts",
            "params": {"part_name": "LG"},
            "store_as": "detected_lg",
            "observation": {"part_name": "LG", "pose": deepcopy(LG_DROP_POSE)},
        }
    ]
    prepared_bridge_request["bridge_session"] = bridge_session
    planner._refresh_bridge_grounding_context(prepared_bridge_request)
    compiled_plan, error = planner._compile_bridge_events_to_macro_tasks(
        prepared_bridge_request,
        approved_events=deepcopy(bridge_session["approved_bridge_events"]),
    )
    assert error is None
    assert isinstance(compiled_plan, dict)
    for task in compiled_plan.get("macro_tasks") or []:
        if not isinstance(task, dict):
            continue
        for step in task.get("primitive_steps") or []:
            if not isinstance(step, dict):
                continue
            if str(step.get("primitive") or "").strip() == "compute_place_targets":
                assert str((step.get("params") or {}).get("part_name") or "").strip()


def test_validate_preprogrammed_bridge_synthesizes_plan_rewrite() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    validated = planner.validate_preprogrammed_bridge_proposal(
        proposal=build_preprogrammed_bridge_proposal(
            scenario_id=SCENARIO_ID,
            prepared_bridge_request=prepared_bridge_request,
        ),
        prepared_bridge_request=prepared_bridge_request,
        source="test",
        scenario_id="plan_rewrite_synthesis",
    )
    assert validated.get("plan_rewrite", {}).get("replace_failed_branch") is True
    assert validated.get("plan_rewrite", {}).get("resume_task_ids") == [
        "REQ_1_T3",
        "REQ_1_T4",
        "REQ_1_T5",
    ]


def test_shorthand_part_transition_still_closes_marked_reentry_gap() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    proposal = build_preprogrammed_bridge_proposal(
        scenario_id=SCENARIO_ID,
        prepared_bridge_request=prepared_bridge_request,
    )
    macro_tasks = proposal.get("macro_tasks") or []
    insert_lg_macro = next(
        task for task in macro_tasks
        if isinstance(task, dict) and str(task.get("macro_name") or "").strip() == "insert_lg"
    )
    insert_lg_macro["task_metadata"]["part_transition"] = {
        "part_name": "LG",
        "from": "in_gripper",
        "to": "assembled",
    }

    validated = planner.validate_preprogrammed_bridge_proposal(
        proposal=proposal,
        prepared_bridge_request=prepared_bridge_request,
        source="test",
        scenario_id="shorthand_part_transition",
    )
    projected_parts = validated.get("projected_parts") or {}
    assert projected_parts.get("LG", {}).get("state") == "assembled"
    assert projected_parts.get("LG", {}).get("location") == "assembly_board-v1"


def test_scenario_matrix_m1_mirror_reachability_selects_xarm6_for_lg_recovery() -> None:
    mirror_pose = {"x": 0.0, "y": -0.50, "z": 1.034}
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
    )
    _record_bridge_observation(
        planner,
        prepared_bridge_request,
        resource_jid="ur5e@localhost",
        part_name="LG",
        pose=mirror_pose,
    )
    approved_events, _, error = planner._bridge_validate_bridge_events(
        prepared_bridge_request,
        events=[
            _bridge_clear_event(resource_jid="xarm6@localhost"),
            _bridge_pick_place_event(
                resource_jid="xarm6@localhost",
                part_name="LG",
                from_state="idle",
            ),
        ],
    )
    assert error is None
    assert approved_events is not None
    lg_events = [
        event for event in approved_events
        if isinstance(event, dict) and str(event.get("part_name") or "").strip() == "LG"
    ]
    assert len(lg_events) == 1
    assert lg_events[0].get("resource_jid") == "xarm6@localhost"

    compiled_plan, compile_error = planner._compile_bridge_events_to_macro_tasks(
        prepared_bridge_request,
        approved_events=deepcopy(approved_events),
    )
    assert compile_error is None
    assert isinstance(compiled_plan, dict)
    assert [
        str(task.get("resource_jid") or "")
        for task in (compiled_plan.get("macro_tasks") or [])
        if isinstance(task, dict) and str(task.get("part_name") or "").strip() == "LG"
    ] == ["xarm6@localhost"]


def test_scenario_matrix_m2_free_gripper_removes_mcp_stage_and_repick() -> None:
    scenario_overrides = {
        "fixture": {
            "part_tracker": {
                "MCP": {
                    "state": "ready",
                    "location": "prusa-mk4-2",
                    "last_known_location": "prusa-mk4-2",
                }
            },
            "part_states": {"MCP": "ready"},
            "part_locations": {"MCP": "prusa-mk4-2"},
            "resource_states": {
                "ur5e@localhost": {
                    "current_state": "idle",
                    "held_part": None,
                }
            },
            "stuck_state": {
                "part_states": {"MCP": "ready"},
                "part_locations": {"MCP": "prusa-mk4-2"},
            },
        },
        "robots": {
            "ur5e@localhost": {
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            }
        },
    }
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
        scenario_overrides=scenario_overrides,
    )
    _record_bridge_observation(
        planner,
        prepared_bridge_request,
        resource_jid="ur5e@localhost",
        part_name="LG",
        pose=deepcopy(LG_DROP_POSE),
    )
    approved_events, _, error = planner._bridge_validate_bridge_events(
        prepared_bridge_request,
        events=[
            _bridge_clear_event(resource_jid="xarm6@localhost"),
            _bridge_pick_place_event(
                resource_jid="ur5e@localhost",
                part_name="LG",
                from_state="idle",
            ),
        ],
    )
    assert error is None
    assert approved_events is not None
    assert not any(
        str(event.get("part_name") or "").strip() == "MCP"
        for event in approved_events
        if isinstance(event, dict)
    )
    assert planner._bridge_synthesize_plan_rewrite(prepared_bridge_request) == {
        "replace_failed_branch": True,
        "resume_task_ids": ["REQ_1_T1", "REQ_1_T2", "REQ_1_T3", "REQ_1_T4", "REQ_1_T5"],
    }
    pending_suffix = next(
        item
        for item in (prepared_bridge_request.get("marked_reentry_context") or {}).get("pending_suffix_summary") or []
        if isinstance(item, dict) and str(item.get("resource_jid") or "").strip() == "ur5e@localhost"
    )
    assert pending_suffix.get("entry_task_id") == "REQ_1_T1"


def test_scenario_matrix_m3_no_resumable_suffix_pressure_skips_restore_event() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
        scenario_overrides={
            "prepared": {
                "bridge_resources": {
                    "ur5e@localhost": {
                        "pending_tasks": [],
                    }
                }
            }
        },
    )
    _record_bridge_observation(
        planner,
        prepared_bridge_request,
        resource_jid="ur5e@localhost",
        part_name="LG",
        pose=deepcopy(LG_DROP_POSE),
    )
    approved_events, _, error = planner._bridge_validate_bridge_events(
        prepared_bridge_request,
        events=[
            _bridge_clear_event(resource_jid="xarm6@localhost"),
            _bridge_stage_event(resource_jid="ur5e@localhost", part_name="MCP"),
            _bridge_pick_place_event(
                resource_jid="ur5e@localhost",
                part_name="LG",
                from_state="idle",
            ),
        ],
    )
    assert error is None
    assert approved_events is not None
    assert planner._bridge_synthesize_plan_rewrite(prepared_bridge_request) == {
        "replace_failed_branch": True,
        "resume_task_ids": [],
    }
    assert any(
        str(event.get("part_name") or "").strip() == "MCP"
        and str((event.get("expected_part_delta") or {}).get("to") or "").strip() == "ready"
        for event in approved_events
        if isinstance(event, dict)
    )
    assert not any(
        str(event.get("part_name") or "").strip() == "MCP"
        and str((event.get("expected_part_delta") or {}).get("to") or "").strip() == "in_gripper"
        for event in approved_events
        if isinstance(event, dict)
    )


def test_scenario_matrix_m4_both_empty_resource_feasibility_follows_pose() -> None:
    cases = [
        ({"x": 0.0, "y": 0.25, "z": 1.034}, "ur5e@localhost", "xarm6@localhost"),
        ({"x": 0.0, "y": -0.50, "z": 1.034}, "xarm6@localhost", "ur5e@localhost"),
    ]
    for pose, expected_resource, rejected_resource in cases:
        _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
            llm_mode="scripted",
            llm_model=None,
            scenario_overrides=_both_robots_idle_overrides(),
        )
        _record_bridge_observation(
            planner,
            prepared_bridge_request,
            resource_jid=expected_resource,
            part_name="LG",
            pose=pose,
        )
        approved_events, _, error = planner._bridge_validate_bridge_events(
            prepared_bridge_request,
            events=[
                _bridge_pick_place_event(
                    resource_jid=expected_resource,
                    part_name="LG",
                    from_state="idle",
                )
            ],
        )
        assert error is None
        assert approved_events is not None

        rejected_events, _, rejected_error = planner._bridge_validate_bridge_events(
            prepared_bridge_request,
            events=[
                _bridge_pick_place_event(
                    resource_jid=rejected_resource,
                    part_name="LG",
                    from_state="idle",
                )
            ],
        )
        assert rejected_events is None
        assert rejected_error is not None
        assert "infeasible" in rejected_error.lower()


def test_scenario_matrix_m5_infeasible_pose_rejects_both_resources() -> None:
    unreachable_pose = {"x": 0.0, "y": 1.30, "z": 1.034}
    for resource_jid in ("ur5e@localhost", "xarm6@localhost"):
        _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
            llm_mode="scripted",
            llm_model=None,
            scenario_overrides=_both_robots_idle_overrides(),
        )
        _record_bridge_observation(
            planner,
            prepared_bridge_request,
            resource_jid="ur5e@localhost",
            part_name="LG",
            pose=unreachable_pose,
        )
        approved_events, _, error = planner._bridge_validate_bridge_events(
            prepared_bridge_request,
            events=[
                _bridge_pick_place_event(
                    resource_jid=resource_jid,
                    part_name="LG",
                    from_state="idle",
                )
            ],
        )
        assert approved_events is None
        assert error is not None
        assert "pose outside workspace" in error


def test_scenario_matrix_m6_focused_resource_switches_marked_reentry_roles() -> None:
    _, _, _, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
        scenario_overrides={"ra_jid": "ur5e@localhost"},
    )
    marked_reentry = prepared_bridge_request.get("marked_reentry_context") or {}
    assert marked_reentry.get("focused_resource_jid") == "ur5e@localhost"
    assert any(
        str(condition.get("entity") or "").strip() == "MCP"
        and str(condition.get("role") or "").strip() == "bridge_replaced"
        for condition in (marked_reentry.get("marked_reentry_conditions") or [])
        if isinstance(condition, dict)
    )
    assert not any(
        str(condition.get("entity") or "").strip() == "LG"
        and str(condition.get("role") or "").strip() == "bridge_replaced"
        for condition in (marked_reentry.get("marked_reentry_conditions") or [])
        if isinstance(condition, dict)
    )
    assert any(
        str(item.get("resource_jid") or "").strip() == "xarm6@localhost"
        and str(item.get("role") or "").strip() == "resume_suffix"
        for item in (marked_reentry.get("pending_suffix_summary") or [])
        if isinstance(item, dict)
    )


def test_scenario_matrix_m7_different_misplaced_part_tracks_mcp_without_lg() -> None:
    mcp_pose = {"x": 0.0, "y": -0.50, "z": 1.034}
    scenario_overrides = {
        "ra_jid": "ur5e@localhost",
        "fixture": {
            "part_tracker": {
                "MCP": {
                    "state": "misplaced",
                    "location": "fixture_xarm6_recovery_pick_zone",
                    "last_known_location": "fixture_xarm6_recovery_pick_zone",
                    "observed_pose": deepcopy(mcp_pose),
                }
            },
            "part_states": {"MCP": "misplaced"},
            "part_locations": {"MCP": "fixture_xarm6_recovery_pick_zone"},
            "resource_states": {
                "ur5e@localhost": {
                    "current_state": "recovery_required",
                    "held_part": None,
                },
                "xarm6@localhost": {
                    "current_state": "idle",
                    "held_part": None,
                },
            },
            "stuck_state": {
                "resource_state": "recovery_required",
                "part_states": {"MCP": "misplaced"},
                "part_locations": {"MCP": "fixture_xarm6_recovery_pick_zone"},
            },
        },
        "robots": {
            "ur5e@localhost": {
                "current_state": "recovery_required",
                "held_part": None,
                "gripper_state": "open",
            },
            "xarm6@localhost": {
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            },
        },
        "prepared": {
            "bridge_resources": {
                "xarm6@localhost": {
                    "pending_tasks": [],
                }
            }
        },
    }
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
        scenario_overrides=scenario_overrides,
    )
    _record_bridge_observation(
        planner,
        prepared_bridge_request,
        resource_jid="xarm6@localhost",
        part_name="MCP",
        pose=mcp_pose,
    )
    approved_events, _, error = planner._bridge_validate_bridge_events(
        prepared_bridge_request,
        events=[
            _bridge_clear_event(resource_jid="ur5e@localhost"),
            _bridge_pick_place_event(
                resource_jid="xarm6@localhost",
                part_name="MCP",
                from_state="idle",
            ),
        ],
    )
    assert error is None
    assert approved_events is not None
    assert not any(
        str(event.get("part_name") or "").strip() == "LG"
        for event in approved_events
        if isinstance(event, dict)
    )
    compiled_plan, compile_error = planner._compile_bridge_events_to_macro_tasks(
        prepared_bridge_request,
        approved_events=deepcopy(approved_events),
    )
    assert compile_error is None
    assert isinstance(compiled_plan, dict)
    compiled_part_names = {
        str(task.get("part_name") or "").strip()
        for task in (compiled_plan.get("macro_tasks") or [])
        if isinstance(task, dict) and str(task.get("part_name") or "").strip()
    }
    assert compiled_part_names == {"MCP"}


def test_scenario_matrix_m8_observation_required_without_fresh_pose() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
        scenario_overrides={
            "fixture": {
                "part_tracker": {
                    "LG": {
                        "observed_pose": None,
                    }
                }
            }
        },
    )
    assert planner._bridge_current_phase(prepared_bridge_request) == "observe_required"
    assert not (prepared_bridge_request.get("bridge_session", {}).get("observation_history") or [])
    assert not (prepared_bridge_request.get("grounding_context", {}).get("step_outputs") or {})


def test_scenario_matrix_m10_neutral_drop_pose_is_feasible_for_both_resources() -> None:
    neutral_pose = {"x": 0.0, "y": 0.0, "z": 1.034}
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
        scenario_overrides=_both_robots_idle_overrides(),
    )
    _record_bridge_observation(
        planner,
        prepared_bridge_request,
        resource_jid="ur5e@localhost",
        part_name="LG",
        pose=neutral_pose,
    )
    for resource_jid in ("ur5e@localhost", "xarm6@localhost"):
        decision = planner._bridge_feasibility_decision(
            prepared_bridge_request,
            resource_jid=resource_jid,
            operation_kind="pick",
            part_name="LG",
        )
        assert decision.get("allowed") is True
        assert "workspace" in str(decision.get("reason") or "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the case3 recovery harness.")
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Use the deterministic scripted baseline instead of the live ReAct loop.",
    )
    parser.add_argument(
        "--variant",
        choices=[MAIN_V1_VARIANT, MAIN_V2_VARIANT],
        default=MAIN_V1_VARIANT,
        help=(
            f"Scenario variant to run: {MAIN_V1_VARIANT} is the original baseline, "
            f"{MAIN_V2_VARIANT} mirrors the recovery to xarm6."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LIVE_MODEL,
        help=f"Model to use in live mode (default: {DEFAULT_LIVE_MODEL}).",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Skip writing the JSON debug artifact.",
    )
    args = parser.parse_args()
    # Always enable DEBUG logging for bridge modules when running directly.
    for _mod in (
        "cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_generation",
        "cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics",
    ):
        _logger = logging.getLogger(_mod)
        _logger.setLevel(logging.DEBUG)
        if not _logger.handlers:
            _handler = logging.StreamHandler()
            _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            _logger.addHandler(_handler)
    run_test(
        llm_mode="scripted" if args.scripted else "live",
        llm_model=args.model,
        write_debug=not args.no_debug,
        variant=args.variant,
    )

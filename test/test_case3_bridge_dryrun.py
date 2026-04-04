"""Dry-run test: Case 3 LG-slippage scenario → LLM bridge recovery.

Mimics what happens in Gazebo when xArm6 places LG, slippage is injected,
and the LLM bridge is triggered to produce a recovery plan.  The plan is then
compared against the verified preprogrammed plan (recover_lg_v1).

Run with F5 / python directly:
    python test/test_case3_bridge_dryrun.py
    python test/test_case3_bridge_dryrun.py --prepare-only
    python test/test_case3_bridge_dryrun.py --model gpt-4o
    python test/test_case3_bridge_dryrun.py --show-llm-input
    python test/test_case3_bridge_dryrun.py --show-prompt

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
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
    build_multi_turn_session_seed,
    transition_multi_turn_phase,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn import (
    _active_pruned_actions_for_state,
    _apply_outline_task_effects,
    _build_outline_validation_findings,
    _build_pruned_actions,
    _execute_observe_requests,
    _infer_outline_macro_signature,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_grounding_compiler import (
    compile_grounded_outline_task,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.prompts.multi_turn import (
    build_multi_turn_phase_prompt_input,
    multi_turn_phase_response_schema,
    render_multi_turn_phase_prompt,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
    get_resource_bridge_snapshot,
)
from cais_spade_llm.agents.central_controller.outline_macro_safety import (
    project_outline_macro_bridge_aps,
    validate_outline_macro_bridge_safety,
)
from cais_spade_llm.agents.intelligent_product.replanner.failure_context import (
    build_failure_event,
    failure_context_from_scenario_config,
    load_failure_scenario_config,
)
from cais_spade_llm.resources.resource_profile import (
    get_resource_profile,
    resource_store_as_contract,
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


def _outline_validation_ref(
    *,
    task_id: str,
    pose_source: str,
    failed_axes: list[str],
    resource_jid: str = "",
) -> dict[str, Any]:
    ref = {
        "task_id": task_id,
        "pose_source": pose_source,
        "failed_axes": list(failed_axes),
    }
    if resource_jid:
        ref["resource_jid"] = resource_jid
    return ref


def _case3_continuation_condition_ids(prepared_bridge_request: dict[str, Any]) -> dict[str, str]:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    modeled_gap = dict(llm_input.get("modeled_continuation_gap") or {})
    condition_ids: dict[str, str] = {}
    for raw_condition in (modeled_gap.get("unmet_continuation_conditions") or []):
        if not isinstance(raw_condition, dict):
            continue
        condition_id = str(raw_condition.get("condition_id") or "").strip()
        if not condition_id:
            continue
        kind = str(raw_condition.get("kind") or "").strip()
        entity = str(raw_condition.get("entity") or "").strip()
        source_task_id = str(raw_condition.get("source_task_id") or "").strip()
        if kind == "focused_resource_terminal_state" and entity == "xarm6@localhost":
            condition_ids["xarm6_idle"] = condition_id
        elif kind == "safety_blocked_suffix_task" and (
            source_task_id == "REQ_1_T3" or entity == "REQ_1_T3"
        ):
            condition_ids["mcp_safe1"] = condition_id
    return condition_ids


def _outline_bridge_validation_context(
    prepared_bridge_request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    resources_by_jid = {
        str(row.get("resource_jid") or "").strip(): deepcopy(row)
        for row in (dict(llm_input.get("observed_runtime_state") or {}).get("resources") or [])
        if isinstance(row, dict) and str(row.get("resource_jid") or "").strip()
    }
    parts_by_name = {
        str(row.get("part_name") or "").strip(): deepcopy(row)
        for row in (llm_input.get("part_facts") or [])
        if isinstance(row, dict) and str(row.get("part_name") or "").strip()
    }
    return llm_input, resources_by_jid, parts_by_name


MOCK_MULTI_TURN_OUTLINE_TASKS = [
    {
        "outline_id": "outline_clear_xarm6",
        "resource_jid": "xarm6@localhost",
        "description": "Move xarm6 from the failed placement posture back to home so it is no longer blocking recovery.",
        "rationale": "The focused failed resource must leave the failed placement state before the fallback sequence can continue safely.",
        "action_target": {
            "named_pose": "home",
            "requirement_id": "REQ_2",
        },
        "expected_start_state": {
            "current_state": "failed",
            "held_part": None,
            "gripper_state": "open",
            "position": {
                "x": 0.1,
                "y": 0.08,
                "z": 1.1994999760206477,
            },
        },
        "expected_end_state": {
            "current_state": "idle",
            "held_part": None,
            "gripper_state": "open",
        },
        "depends_on": [],
    },
    {
        "outline_id": "outline_return_mcp_to_printer",
        "resource_jid": "ur5e@localhost",
        "description": "Place MCP back at prusa-mk4-2 so ur5e can switch to LG recovery.",
        "rationale": "The supporting robot must free its gripper before it can recover the misplaced predecessor part.",
        "part_name": "MCP",
        "action_target": {
            "target_location": "prusa-mk4-2",
            "requirement_id": "REQ_1",
        },
        "expected_start_state": {
            "current_state": "picked",
            "gripper_state": "closed",
            "held_part": "MCP",
            "part_name": "MCP",
        },
        "expected_end_state": {
            "current_state": "idle",
            "gripper_state": "released",
            "held_part": None,
            "part_name": "MCP",
            "location": "prusa-mk4-2",
        },
        "depends_on": ["outline_clear_xarm6"],
    },
    {
        "outline_id": "outline_pick_lg_with_ur5e",
        "resource_jid": "ur5e@localhost",
        "description": "Move to the grounded LG pose and pick the misplaced LG part.",
        "rationale": "ur5e must secure LG from the grounded misplaced pose before it can restore the part to the assembly board.",
        "part_name": "LG",
        "action_target": {
            "source_location": "observed_pose",
            "requirement_id": "REQ_2",
        },
        "expected_start_state": {
            "part_name": "LG",
            "position": {"x": 0.0, "y": 0.2, "z": 1.035},
            "current_state": "misplaced",
            "held_part": None,
            "gripper_state": "open",
        },
        "expected_end_state": {
            "part_name": "LG",
            "held_part": "LG",
            "gripper_state": "closed",
            "current_state": "picked",
        },
        "depends_on": ["outline_return_mcp_to_printer"],
    },
    {
        "outline_id": "outline_place_lg_with_ur5e",
        "resource_jid": "ur5e@localhost",
        "description": "Place LG at assembly_board-v1 from the grounded recovery-side pose.",
        "rationale": "SAFE_1 keeps MCP blocked until LG is restored to the assembly board.",
        "part_name": "LG",
        "action_target": {
            "target_location": "assembly_board-v1",
            "requirement_id": "REQ_2",
        },
        "expected_start_state": {
            "part_name": "LG",
            "held_part": "LG",
            "gripper_state": "closed",
            "current_state": "picked",
        },
        "expected_end_state": {
            "part_name": "LG",
            "current_state": "assembled",
            "held_part": None,
            "gripper_state": "released",
            "location": "assembly_board-v1",
        },
        "depends_on": ["outline_pick_lg_with_ur5e"],
    },
    {
        "outline_id": "outline_resume_mcp_assembly",
        "resource_jid": "ur5e@localhost",
        "description": "Resume MCP assembly once LG has been restored.",
        "rationale": "With LG placed, the blocked MCP suffix can safely resume.",
        "part_name": "MCP",
        "action_target": {
            "target_location": "assembly_board-v1",
            "requirement_id": "REQ_1",
        },
        "expected_start_state": {
            "current_state": "idle",
            "held_part": None,
            "part_name": "MCP",
        },
        "expected_end_state": {
            "current_state": "idle",
            "gripper_state": "released",
            "held_part": None,
            "part_name": "MCP",
            "location": "assembly_board-v1",
        },
        "depends_on": ["outline_place_lg_with_ur5e"],
    },
]
MOCK_MULTI_TURN_INVALID_XARM_OUTLINE_TASKS = [
    {
        "outline_id": "outline_clear_xarm6",
        "resource_jid": "xarm6@localhost",
        "description": "Move xarm6 from the failed placement posture back to home.",
        "rationale": "The focused failed resource must clear the failed posture before the fallback sequence continues.",
        "action_target": {
            "named_pose": "home",
            "requirement_id": "REQ_2",
        },
        "expected_start_state": {
            "current_state": "failed",
            "held_part": None,
            "gripper_state": "open",
            "position": {
                "x": 0.1,
                "y": 0.08,
                "z": 1.1994999760206477,
            },
        },
        "expected_end_state": {
            "current_state": "idle",
            "held_part": None,
            "gripper_state": "open",
        },
        "depends_on": [],
    },
    {
        "outline_id": "outline_recover_lg_with_xarm6",
        "resource_jid": "xarm6@localhost",
        "description": "Recover LG from the observed slippage pose with xarm6 and place it on the board.",
        "rationale": "This invalid draft tries to keep the repair on the failed robot even though the grounded LG pose is outside xarm6 reach.",
        "part_name": "LG",
        "action_target": {
            "target_location": "assembly_board-v1",
            "requirement_id": "REQ_2",
        },
        "expected_start_state": {
            "part_name": "LG",
            "position": {"x": 0.0, "y": 0.2, "z": 1.035},
        },
        "expected_end_state": {
            "part_name": "LG",
            "current_state": "assembled",
            "location": "assembly_board-v1",
        },
        "depends_on": ["outline_clear_xarm6"],
    },
]
MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS: list[dict[str, Any]] = []
MOCK_INVALID_XARM_REQUIRED_FINDING_REFS = [
    _outline_validation_ref(
        task_id="outline_recover_lg_with_xarm6",
        resource_jid="xarm6@localhost",
        pose_source="resource_feasibility",
        failed_axes=["workspace_unreachable"],
    ),
]
MOCK_MISSING_PART_NAME_REQUIRED_FINDING_REFS = [
    _outline_validation_ref(
        task_id="OT2",
        resource_jid="xarm6@localhost",
        pose_source="task_contract",
        failed_axes=["missing_part_name"],
    ),
    _outline_validation_ref(
        task_id="OT2",
        resource_jid="xarm6@localhost",
        pose_source="observed_pose",
        failed_axes=["y=0.2000 > y_max_m=0.1000"],
    ),
    _outline_validation_ref(
        task_id="OT2",
        resource_jid="xarm6@localhost",
        pose_source="expected_start_state",
        failed_axes=["y=0.2000 > y_max_m=0.1000"],
    ),
]
MOCK_ABSTRACT_TASK_REQUIRED_FINDING_REFS = [
    _outline_validation_ref(
        task_id="OT2",
        resource_jid="ur5e@localhost",
        pose_source="task_contract",
        failed_axes=["task_not_projectable"],
    )
]
MOCK_UNEXPECTED_PART_REFERENCE_REQUIRED_FINDING_REFS = [
    _outline_validation_ref(
        task_id="OT1",
        resource_jid="xarm6@localhost",
        pose_source="task_contract",
        failed_axes=["resource_only_part_tag_ignored"],
    )
]
MOCK_HELD_PART_CONFLICT_REQUIRED_FINDING_REFS = [
    _outline_validation_ref(
        task_id="OT2",
        resource_jid="ur5e@localhost",
        pose_source="resource_feasibility",
        failed_axes=["holder_conflict"],
    ),
]
MOCK_ILLEGAL_HOLDER_SWAP_REQUIRED_FINDING_REFS = [
    _outline_validation_ref(
        task_id="OT2",
        resource_jid="ur5e@localhost",
        pose_source="resource_feasibility",
        failed_axes=["holder_conflict"],
    )
]
MOCK_NAMED_POSE_NOT_AVAILABLE_REQUIRED_FINDING_REFS = [
    _outline_validation_ref(
        task_id="OT1",
        resource_jid="xarm6@localhost",
        pose_source="resource_feasibility",
        failed_axes=["named_pose_unavailable"],
    )
]
MOCK_MISSING_POST_TASK_ANCHOR_REQUIRED_FINDING_REFS = [
    _outline_validation_ref(
        task_id="OT2",
        resource_jid="ur5e@localhost",
        pose_source="task_contract",
        failed_axes=["task_not_projectable"],
    )
]
MOCK_CONTINUATION_PREREQUISITE_REQUIRED_FINDING_REFS = [
    _outline_validation_ref(
        task_id="resume_mcp",
        resource_jid="ur5e@localhost",
        pose_source="task_contract",
        failed_axes=["dependency_unsatisfied"],
    ),
    _outline_validation_ref(
        task_id="resume_mcp",
        resource_jid="ur5e@localhost",
        pose_source="task_contract",
        failed_axes=["order_violation"],
    ),
    _outline_validation_ref(
        task_id="resume_mcp",
        resource_jid="ur5e@localhost",
        pose_source="task_contract",
        failed_axes=["blocker_open"],
    ),
]
MOCK_MULTI_TURN_MACRO_TASKS = [
    {
        "resource_jid": "xarm6@localhost",
        "macro_name": "clear_failed_robot",
        "description": "Clear xarm6 from the failed placement pose to home.",
        "rationale": "This restores the focused failed resource to a resumable state.",
        "expected_start_state": "xarm6 failed with empty gripper",
        "task_params": {"target_pose_name": "home"},
        "task_metadata": {"outline_id": "outline_clear_xarm6"},
        "primitive_steps": [
            {
                "primitive": "move_to_named_pose",
                "params": {"pose_name": "home"},
            }
        ],
    },
    {
        "resource_jid": "ur5e@localhost",
        "macro_name": "recover_lg_and_resume_mcp",
        "description": "Return MCP, recover LG to the board, and leave ur5e ready to continue.",
        "rationale": "LG must be assembled before MCP continuation is safe.",
        "part_name": "LG",
        "expected_start_state": "ur5e holds MCP while LG is observed away from the board",
        "task_params": {
            "mcp_origin": "prusa-mk4-2",
            "lg_destination": "assembly_board-v1",
        },
        "task_metadata": {"outline_id": "outline_recover_lg_with_ur5e"},
        "primitive_steps": [
            {
                "primitive": "compute_place_targets",
                "params": {"part_name": "MCP", "destination_location": "prusa-mk4-2"},
                "store_as": "mcp_return",
            },
            {
                "primitive": "move_cartesian",
                "params": {"context_ref": "/step_outputs/mcp_return/approach_pose"},
            },
            {
                "primitive": "move_cartesian",
                "params": {"context_ref": "/step_outputs/mcp_return/target_pose"},
            },
            {
                "primitive": "release_part",
                "params": {"part_name": "MCP"},
            },
            {
                "primitive": "detect_parts",
                "params": {"part_name": "LG"},
                "store_as": "lg_detection",
            },
            {
                "primitive": "compute_pick_targets",
                "params": {
                    "part_name": "LG",
                    "target_pose": {"context_ref": "/step_outputs/lg_detection/pose"},
                },
                "store_as": "lg_pick",
            },
            {
                "primitive": "compute_place_targets",
                "params": {
                    "part_name": "LG",
                    "pick_ctx": {"context_ref": "/step_outputs/lg_pick"},
                    "destination_location": "assembly_board-v1",
                },
                "store_as": "lg_place",
            },
        ],
    },
]
MOCK_MULTI_TURN_FINAL_PROPOSAL = {
    "thought": (
        "xarm6 first clears the failed placement state so the focused failed resource becomes resumable. "
        "ur5e then restores MCP to origin, recovers LG to the board, and leaves the blocked MCP suffix runnable again."
    ),
    "primary_obligation": {
        "rule_id": "SAFE_1",
        "resource_jid": "ur5e@localhost",
    },
    "macro_tasks": deepcopy(MOCK_MULTI_TURN_MACRO_TASKS),
}
MOCK_MULTI_TURN_RESPONSES = [
    {
        "thought": "LG placement remains unresolved, and the current failure facts do not include a trusted live LG pose.",
        "decision": "observe",
        "blocking_reasons": [
            "LG is not yet assembled on the board.",
            "SAFE_1 blocks MCP continuation until LG is assembled.",
        ],
        "grounded_facts": [
            "xarm6 is failed at the blocked LG suffix.",
            "LG still needs a runtime observation before recovery can be grounded.",
        ],
        "recovery_implications": [
            "Recovery planning still needs a concrete LG pose before choosing a replacement manipulator sequence.",
        ],
        "sufficient_grounding": False,
        "observe_reason": (
            "Grounding is not yet sufficient because LG still needs a runtime pose observation."
        ),
        "observe_requests": [
            {
                "fact_type": "part_pose",
                "entity": "LG",
                "store_as": "lg_detection_seed",
            }
        ],
    },
    {
        "thought": "The prompt facts plus the stored LG observation are enough to outline recovery.",
        "decision": "grounded",
        "blocking_reasons": [
            "xarm6 is failed.",
            "ur5e holds MCP while LG still needs assembly.",
        ],
        "grounded_facts": [
            "LG was observed at x=0.0, y=0.2, z=1.035 in the ur5e side of the cell.",
            "xarm6 remains failed and cannot finish the original LG assemble suffix.",
            "SAFE_1 still requires LG placement before MCP can approach the assembly board.",
        ],
        "recovery_implications": [
            "The replacement plan should recover LG from the ur5e-reachable side of the workspace.",
            "The assembly-board area must remain clear of xarm6 activity while the fallback sequence is prepared.",
        ],
        "sufficient_grounding": True,
        "observe_requests": [],
    },
    {
        "thought": "The recovery needs grounded task-level actions that clear xarm6, restore LG, and then resume MCP.",
        "addressed_validation_findings": deepcopy(MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS),
        "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
    },
    {
        "thought": "The accepted outline now resolves into primitive-connected macro tasks that preserve the LG-before-MCP dependency.",
        "decision": "draft_ready",
        "macro_tasks": deepcopy(MOCK_MULTI_TURN_MACRO_TASKS),
    },
    {
        "thought": "The draft is now packaged as the final bridge proposal.",
        "decision": "final_ready",
        "final_proposal": deepcopy(MOCK_MULTI_TURN_FINAL_PROPOSAL),
    },
]

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


def _contains_fixed_recovery_labels(value: Any) -> bool:
    fixed_tokens = (
        "ur5e_recovery_lane",
        "ur5e_base_area",
        "assembly_board_v1",
    )
    serialized = json.dumps(value, default=str)
    return any(token in serialized for token in fixed_tokens)


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
        "workspace_bounds": deepcopy(
            dict(resource_entry.get("static_capabilities") or {}).get("workspace_bounds")
        ),
    }
    return {
        key: value
        for key, value in prompt_snapshot.items()
        if value not in (None, "", [], {})
    }


def _expected_ordered_resource_jids(
    *,
    focused_resource_jid: str,
    observed_resources: list[dict[str, Any]],
    bridge_resources: dict[str, Any],
) -> list[str]:
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
    return ordered_resource_jids


def _expected_runtime_resource_facts(
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    context_summary = dict(prepared_bridge_request.get("context_summary") or {})
    fault_event = dict(context_summary.get("fault_event") or {})
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

    runtime_resources: list[dict[str, Any]] = []
    for resource_jid in _expected_ordered_resource_jids(
        focused_resource_jid=str(fault_event.get("focused_resource_jid") or "").strip(),
        observed_resources=observed_resources,
        bridge_resources=bridge_resources,
    ):
        observed_row = dict(observed_by_jid.get(resource_jid) or {})
        bridge_snapshot = _expected_prompt_bridge_snapshot(
            dict(bridge_resources.get(resource_jid) or {})
        )
        runtime_row = {
            "resource_jid": resource_jid,
            "current_state": deepcopy(
                observed_row.get("current_state")
                if "current_state" in observed_row
                else bridge_snapshot.get("current_state")
            ),
            "current_location": deepcopy(
                observed_row.get("current_location")
                if "current_location" in observed_row
                else bridge_snapshot.get("current_location")
            ),
            "availability": deepcopy(
                observed_row.get("availability")
                if "availability" in observed_row
                else bridge_snapshot.get("availability")
            ),
            "held_part": deepcopy(
                observed_row.get("held_part")
                if "held_part" in observed_row
                else bridge_snapshot.get("held_part")
            ),
        }
        for optional_field in (
            "gripper_state",
            "current_pose",
            "current_pose_ref",
            "named_poses",
            "workspace_bounds",
        ):
            optional_value = deepcopy(bridge_snapshot.get(optional_field))
            if optional_value not in (None, "", [], {}):
                runtime_row[optional_field] = optional_value
        runtime_resources.append(runtime_row)
    return runtime_resources


def _expected_part_facts(
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    context_summary = dict(prepared_bridge_request.get("context_summary") or {})
    current_product_state = dict(context_summary.get("current_product_state") or {})
    relevant_assembly_requirements = [
        deepcopy(entry)
        for entry in (context_summary.get("relevant_assembly_requirements") or [])
        if isinstance(entry, dict)
    ]
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})

    requirements_by_product: dict[str, list[dict[str, Any]]] = {}
    for requirement_entry in relevant_assembly_requirements:
        product = str(requirement_entry.get("product") or "").strip()
        if product:
            requirements_by_product.setdefault(product, []).append(requirement_entry)

    current_resources = [
        dict(row)
        for row in (current_product_state.get("resources") or [])
        if isinstance(row, dict)
    ]
    held_part_to_resource = {
        str(row.get("held_part") or "").strip(): str(row.get("resource_jid") or "").strip()
        for row in current_resources
        if str(row.get("held_part") or "").strip()
        and str(row.get("resource_jid") or "").strip()
    }

    pending_task_ids_by_part: dict[str, list[str]] = {}
    for resource_jid in sorted(bridge_resources):
        resource_entry = dict(bridge_resources.get(resource_jid) or {})
        for task in (resource_entry.get("pending_tasks") or []):
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("id") or "").strip()
            part_name = str((task.get("params") or {}).get("part_name") or "").strip()
            if task_id and part_name:
                pending_task_ids_by_part.setdefault(part_name, []).append(task_id)

    current_parts = [
        dict(row)
        for row in (current_product_state.get("parts") or [])
        if isinstance(row, dict)
    ]
    current_part_by_name = {
        str(row.get("part_name") or "").strip(): row
        for row in current_parts
        if str(row.get("part_name") or "").strip()
    }
    tracker_by_part = {
        str(part_name or "").strip(): dict(raw_entry or {})
        for part_name, raw_entry in dict(prepared_bridge_request.get("part_tracker") or {}).items()
        if str(part_name or "").strip() and isinstance(raw_entry, dict)
    }

    ordered_part_names: list[str] = []
    seen_part_names: set[str] = set()
    for row in current_parts:
        part_name = str(row.get("part_name") or "").strip()
        if part_name and part_name not in seen_part_names:
            seen_part_names.add(part_name)
            ordered_part_names.append(part_name)
    for part_name in sorted(tracker_by_part):
        if part_name not in seen_part_names:
            seen_part_names.add(part_name)
            ordered_part_names.append(part_name)

    part_facts: list[dict[str, Any]] = []
    for part_name in ordered_part_names:
        current_part_row = dict(current_part_by_name.get(part_name) or {})
        tracker_entry = dict(tracker_by_part.get(part_name) or {})
        requirement_entry = dict((requirements_by_product.get(part_name) or [None])[0] or {})
        current_location = deepcopy(current_part_row.get("location"))
        holder_resource_jid = str(held_part_to_resource.get(part_name) or "").strip()
        if (
            not holder_resource_jid
            and isinstance(current_location, str)
            and current_location.endswith("_gripper")
        ):
            holder_resource_jid = current_location.rsplit("_gripper", 1)[0]
        part_facts.append(
            {
                "part_name": part_name,
                "current_state": deepcopy(current_part_row.get("state")),
                "current_location": current_location,
                "location_basis": deepcopy(current_part_row.get("location_basis")),
                "observed_pose": deepcopy(current_part_row.get("observed_pose")),
                "current_holder_resource_jid": holder_resource_jid or None,
                "origin_location": deepcopy(tracker_entry.get("origin_resource_location")),
                "goal_location": deepcopy(current_part_row.get("target_location")),
                "goal_requirement_id": (
                    str(requirement_entry.get("requirement_id") or "").strip() or None
                ),
                "nominal_requirement_resource_jid": (
                    str(requirement_entry.get("resource_jid") or "").strip() or None
                ),
                "pending_nominal_task_ids": deepcopy(
                    pending_task_ids_by_part.get(part_name) or []
                ),
            }
        )
    return part_facts


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
    ordered_resource_jids = _expected_ordered_resource_jids(
        focused_resource_jid=focused_resource_jid,
        observed_resources=observed_resources,
        bridge_resources=bridge_resources,
    )

    prompt_resources: list[dict[str, Any]] = []

    def _expected_allowed_primitives(resource_entry: dict[str, Any]) -> list[dict[str, Any]]:
        resource_type = str(resource_entry.get("resource_type") or "resource").strip() or "resource"
        profile = get_resource_profile(resource_type)
        preview_output_map = dict(profile.preview_output_map or {})
        extract_output_map = dict(profile.extract_output_map or {})
        allowed_primitives: list[dict[str, Any]] = []
        for item in (resource_entry.get("primitive_catalog") or []):
            if not isinstance(item, dict):
                continue
            primitive_name = str(item.get("name") or "").strip()
            if not primitive_name:
                continue
            allowed_entry: dict[str, Any] = {"name": primitive_name}
            description = str(item.get("description") or item.get("semantic_summary") or "").strip()
            if description:
                allowed_entry["description"] = description
            allowed_entry["required_params"] = [
                str(param).strip()
                for param in (item.get("required_params") or [])
                if str(param).strip()
            ]
            primitive_kind = str(item.get("primitive_kind") or "").strip()
            if primitive_kind:
                allowed_entry["primitive_kind"] = primitive_kind
            output_fields = [
                str(field).strip()
                for field in dict(item.get("output_schema") or {}).keys()
                if str(field).strip()
            ]
            if output_fields:
                allowed_entry["output_fields"] = output_fields
            hard_preconditions = deepcopy(item.get("preconditions") or {})
            if hard_preconditions:
                allowed_entry["hard_preconditions"] = hard_preconditions
            supports_store_as = (
                primitive_name in preview_output_map or primitive_name in extract_output_map
            )
            allowed_entry["supports_store_as"] = supports_store_as
            if supports_store_as:
                store_as_contract = resource_store_as_contract(profile, primitive_name)
                store_as_required_params = [
                    str(param).strip()
                    for param in (store_as_contract.get("required_params") or [])
                    if str(param).strip()
                ]
                store_as_any_of_param_sets = [
                    [
                        str(param).strip()
                        for param in (param_set or [])
                        if str(param).strip()
                    ]
                    for param_set in (store_as_contract.get("any_of_param_sets") or [])
                    if isinstance(param_set, (list, tuple))
                ]
                if store_as_required_params:
                    allowed_entry["store_as_required_params"] = store_as_required_params
                if store_as_any_of_param_sets:
                    allowed_entry["store_as_any_of_param_sets"] = store_as_any_of_param_sets
            allowed_primitives.append(allowed_entry)
        return allowed_primitives

    for resource_jid in ordered_resource_jids:
        bridge_entry = dict(bridge_resources.get(resource_jid) or {})
        adapter_capabilities = dict(bridge_entry.get("bridge_adapter") or {})
        if not adapter_capabilities.get("supports_executable_bridge"):
            continue
        observed_row = dict(observed_by_jid.get(resource_jid) or {})
        prompt_resources.append(
            {
                "resource_jid": resource_jid,
                "resource_type": str(
                    bridge_entry.get("resource_type")
                    or dict((bridge_entry.get("bridge_snapshot") or {}).get("resource_core") or {}).get("resource_type")
                    or "unknown"
                ).strip(),
                "role": deepcopy(
                    observed_row.get("role")
                    or ("focused" if resource_jid == focused_resource_jid else "supporting")
                ),
                "allowed_primitives": _expected_allowed_primitives(bridge_entry),
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
    stop_after_phase: str | None = None,
) -> None:
    """Increase turn budget for live LLM runs."""
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    reasoning_mode = str(bridge_session.get("reasoning_mode") or "").strip().lower()
    bridge_session["max_turns"] = max(int(bridge_session.get("max_turns", 6) or 6), 16)
    bridge_session["max_observations"] = max(int(bridge_session.get("max_observations", 3) or 3), 5)
    bridge_session["max_observe_batch"] = max(1, min(3, int(bridge_session.get("max_observe_batch", 3) or 3)))
    bridge_session["max_final_retries"] = max(int(bridge_session.get("max_final_retries", 2) or 2), 4)
    normalized_stop_after = str(stop_after_phase or "").strip().lower()
    if stop_after_phase is None and reasoning_mode == "multi_turn":
        normalized_stop_after = "grounding"
    if normalized_stop_after:
        bridge_session["stop_after_phase"] = normalized_stop_after
    else:
        bridge_session.pop("stop_after_phase", None)
    bridge_session["repair_mode"] = "recover"
    bridge_session["observation_backend"] = "mock_detect_parts_harness"
    prepared_bridge_request["bridge_session"] = bridge_session
    if reasoning_mode == "multi_turn":
        prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
            prepared_bridge_request
        )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


async def _prepare_bridge_dryrun_harness(
    *,
    llm_model: str | None = None,
    configure_live: bool = True,
    fixture_mode: str = "live",
    bridge_reasoning_mode: str | None = None,
    multi_turn_stop_after_phase: str | None = None,
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

    shared_grounding_observations = _build_shared_grounding_observation_catalog(
        prepared_bridge_request=prepared_bridge_request,
        robots=[ur5e, xarm6],
    )
    ur5e.set_shared_observations(shared_grounding_observations)
    xarm6.set_shared_observations(shared_grounding_observations)

    product_agent.prepared_bridge_request = prepared_bridge_request
    if configure_live:
        _relax_recovery_clear_precondition(prepared_bridge_request)
        _configure_live_bridge_session(
            prepared_bridge_request,
            stop_after_phase=multi_turn_stop_after_phase,
        )

    return fixture, product_agent, planner, prepared_bridge_request


# ---------------------------------------------------------------------------
# Main coroutine
# ---------------------------------------------------------------------------


async def run_case3_bridge_dryrun(
    write_debug: bool = True,
    *,
    llm_model: str | None = None,
    bridge_reasoning_mode: str | None = None,
    multi_turn_stop_after_phase: str | None = None,
) -> dict[str, Any]:
    """Run the active bridge through the configured bridge reasoning mode."""
    effective_stop_after_phase = multi_turn_stop_after_phase
    if (
        effective_stop_after_phase is None
        and str(bridge_reasoning_mode or "").strip().lower() == "multi_turn"
    ):
        effective_stop_after_phase = "outline"
    fixture, product_agent, planner, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
        llm_model=llm_model,
        fixture_mode="live",
        bridge_reasoning_mode=bridge_reasoning_mode,
        multi_turn_stop_after_phase=effective_stop_after_phase,
    )

    if write_debug:
        debug_dir = Path(DEBUG_DIR)
        if not debug_dir.is_absolute():
            debug_dir = _repo_root() / debug_dir
        bridge_debug_seed = dict(prepared_bridge_request.get("bridge_debug") or {})
        bridge_debug_seed["per_turn_debug_dir"] = str(debug_dir)
        prepared_bridge_request["bridge_debug"] = bridge_debug_seed

    proposal = await planner.execute_prepared_bridge_request(prepared_bridge_request)
    bridge_debug = planner.get_last_bridge_debug()
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    reasoning_mode = str(bridge_session.get("reasoning_mode") or "single_shot").strip()
    multi_turn_session = deepcopy(bridge_debug.get("multi_turn_session") or {})
    multi_turn_turns = list(multi_turn_session.get("turns") or [])
    raw_response_value = ""
    if reasoning_mode == "multi_turn":
        latest_response = (
            dict(multi_turn_turns[-1] or {}).get("raw_response")
            if multi_turn_turns
            else None
        )
        if latest_response not in (None, "", [], {}):
            raw_response_value = json.dumps(
                latest_response,
                indent=2,
                default=str,
                ensure_ascii=True,
            )
    else:
        raw_response_value = str(
            ((bridge_debug.get("single_shot_turn") or {}).get("raw_response") or "")
        )

    result: dict[str, Any] = {
        "scenario": "case3_lg_slippage",
        "reasoning_mode": reasoning_mode,
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
        "multi_turn_session_seed": deepcopy(
            prepared_bridge_request.get("multi_turn_session_seed") or {}
        ),
        "multi_turn_session": multi_turn_session,
        "raw_response": raw_response_value,
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


async def run_case3_bridge_prepare_trace(
    write_debug: bool = True,
    *,
    llm_model: str | None = None,
    bridge_reasoning_mode: str | None = None,
) -> dict[str, Any]:
    """Prepare the bridge request and stop before any LLM stage."""
    fixture, product_agent, planner, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
        llm_model=llm_model,
        configure_live=False,
        fixture_mode="live",
        bridge_reasoning_mode=bridge_reasoning_mode,
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    result: dict[str, Any] = {
        "scenario": "case3_lg_slippage",
        "reasoning_mode": str(bridge_session.get("reasoning_mode") or "single_shot").strip(),
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
        "multi_turn_session_seed": deepcopy(
            prepared_bridge_request.get("multi_turn_session_seed") or {}
        ),
        "raw_response": "",
        "turn_log": deepcopy(product_agent.turn_log),
        "prompt_artifact_path": None,
        "latest_prompt_artifact_path": None,
        "response_artifact_path": None,
        "latest_response_artifact_path": None,
        "session_transcript_artifact_path": None,
        "latest_session_transcript_artifact_path": None,
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
    assert lg_row.get("state") == "misplaced"
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
    safe1_loaded_rule = next(
        (
            rule for rule in loaded_safety_rules
            if isinstance(rule, dict) and str(rule.get("rule_id") or "") == "SAFE_1"
        ),
        {},
    )
    assert safe1_loaded_rule.get("ap_scope") in {"both", "bridge"}
    assert safe1_loaded_rule.get("bridge_aps"), "bridge AP metadata should be loaded for SAFE_1"
    assert str(safe1_loaded_rule.get("dfa_dot") or "").strip(), "SAFE_1 DFA DOT should be loaded"

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
    assert ("goal", "LG", "state", GOAL_STATE, "misplaced") in goal_unsatisfied_keys
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
    assert llm_resources, "llm_input observed resources are missing"
    assert "parts" not in observed_runtime_state, "llm_input part facts should live in part_facts"
    assert llm_resources == _expected_runtime_resource_facts(prepared_bridge_request)

    llm_resource_by_jid = {
        str(row.get("resource_jid") or ""): row
        for row in llm_resources
        if isinstance(row, dict) and str(row.get("resource_jid") or "")
    }
    xarm6_runtime_row = llm_resource_by_jid.get("xarm6@localhost") or {}
    assert xarm6_runtime_row.get("current_state") == "failed"
    assert "pending_task_ids" not in xarm6_runtime_row, "llm_input observed resources should not include pending task ids"
    assert xarm6_runtime_row.get("current_pose"), "llm_input resource facts should retain grounded poses"
    assert xarm6_runtime_row.get("named_poses"), "llm_input resource facts should retain named poses"

    part_facts = llm_input.get("part_facts") or []
    assert part_facts == _expected_part_facts(prepared_bridge_request)
    llm_parts_by_name = {
        str(row.get("part_name") or ""): row
        for row in part_facts
        if isinstance(row, dict) and str(row.get("part_name") or "")
    }
    llm_lg_row = llm_parts_by_name.get("LG") or {}
    assert llm_lg_row.get("current_state") == "misplaced"
    assert llm_lg_row.get("observed_pose"), "llm_input should retain LG observed pose"
    assert llm_lg_row.get("goal_location") == "assembly_board-v1"
    assert llm_lg_row.get("goal_requirement_id") == "REQ_2"
    assert llm_lg_row.get("nominal_requirement_resource_jid") == "xarm6@localhost"
    assert llm_lg_row.get("current_holder_resource_jid") is None
    llm_mcp_row = llm_parts_by_name.get("MCP") or {}
    assert llm_mcp_row.get("current_holder_resource_jid") == "ur5e@localhost"
    assert llm_mcp_row.get("origin_location") == "prusa-mk4-2"
    assert "target_location" not in llm_lg_row, "llm_input part facts should not expose raw modeled target fields"

    llm_loaded_rule_ids = {
        str(rule.get("rule_id") or "")
        for rule in (llm_input.get("loaded_safety_rules") or [])
        if isinstance(rule, dict) and str(rule.get("rule_id") or "")
    }
    assert llm_loaded_rule_ids >= {"SAFE_1", "SAFE_2"}
    llm_safe1_rule = next(
        (
            rule for rule in (llm_input.get("loaded_safety_rules") or [])
            if isinstance(rule, dict) and str(rule.get("rule_id") or "") == "SAFE_1"
        ),
        {},
    )
    assert llm_safe1_rule.get("bridge_aps"), "llm_input should retain bridge AP metadata for SAFE_1"
    assert isinstance(llm_input.get("obligation_targets"), list)

    llm_requirement_ids = {
        str(entry.get("requirement_id") or "")
        for entry in (llm_input.get("relevant_assembly_requirements") or [])
        if isinstance(entry, dict) and str(entry.get("requirement_id") or "")
    }
    assert llm_requirement_ids >= {"REQ_1", "REQ_2"}

    llm_gap = llm_input.get("modeled_continuation_gap") or {}
    assert "goal_state" not in llm_gap
    assert "unmet_goal_conditions" not in llm_gap
    assert llm_gap.get("pending_nominal_tasks"), "llm_input should retain pending nominal tasks"
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
    assert str(idle_continuation_gap.get("condition_id") or "").startswith("cond_")
    pending_nominal_tasks = llm_gap.get("pending_nominal_tasks") or []
    move_home_task = next(
        (
            entry for entry in pending_nominal_tasks
            if isinstance(entry, dict) and entry.get("id") == "REQ_2_T5"
        ),
        None,
    )
    assert move_home_task, "llm_input should retain pending move_home task"
    assert idle_continuation_gap.get("condition_id") in (
        move_home_task.get("blocked_by_condition_ids") or []
    )
    assert "bridge_debug" not in llm_input
    assert "data_flow_trace" not in llm_input
    assert "requirement_task_index" not in llm_input
    assert "task_requirement_map" not in llm_input
    llm_fault_event = llm_input.get("fault_event") or {}
    assert llm_fault_event.get("affected_part_names") == ["LG"]
    assert "affected_parts" not in llm_fault_event

    allowed_execution_surface = llm_input.get("allowed_execution_surface") or {}
    assert allowed_execution_surface == _expected_allowed_execution_surface(
        prepared_bridge_request
    )
    assert "primitive_catalog_by_resource_type" not in allowed_execution_surface
    allowed_surface_resources = allowed_execution_surface.get("resources") or []
    allowed_resource_by_jid = {
        str(row.get("resource_jid") or ""): row
        for row in allowed_surface_resources
        if isinstance(row, dict) and str(row.get("resource_jid") or "")
    }
    assert set(allowed_resource_by_jid) >= {"ur5e@localhost", "xarm6@localhost"}
    for resource_jid, resource_entry in allowed_resource_by_jid.items():
        allowed_primitives = resource_entry.get("allowed_primitives") or []
        assert set(resource_entry) == {
            "resource_jid",
            "resource_type",
            "role",
            "allowed_primitives",
        }
        primitive_names = {
            str(entry.get("name") or "")
            for entry in allowed_primitives
            if isinstance(entry, dict) and str(entry.get("name") or "")
        }
        assert {"grasp_part", "release_part"} <= primitive_names
        assert not (
            primitive_names
            & {"open_gripper", "close_gripper", "attach_part", "detach_part"}
        )
        for primitive_entry in allowed_primitives:
            assert "composite_expansion" not in primitive_entry
            assert "params" not in primitive_entry
            assert "effects" not in primitive_entry
            assert "preconditions" not in primitive_entry
            assert "output_schema" not in primitive_entry
            assert isinstance(primitive_entry.get("required_params"), list)
            assert isinstance(primitive_entry.get("supports_store_as"), bool)
            if primitive_entry.get("name") in {
                "detect_parts",
                "get_current_pose",
                "compute_pick_targets",
                "compute_place_targets",
            }:
                assert primitive_entry.get("supports_store_as") is True
            if primitive_entry.get("name") in {"grasp_part", "release_part", "move_to_named_pose"}:
                assert primitive_entry.get("hard_preconditions")

    assert single_shot_prompt_input.get("reasoning_mode") == "single_shot"
    assert single_shot_prompt_input.get("llm_input") == llm_input
    assert "obligation_targets" not in single_shot_prompt_input
    assert "primitive_catalog" not in single_shot_prompt_input
    assert "bridge_snapshot" not in single_shot_prompt_input
    assert "focused_bridge_snapshot" not in single_shot_prompt_input
    assert "bridge_resources" not in single_shot_prompt_input
    assert "other_resources_summary" not in single_shot_prompt_input
    assert "proposal_success_criteria" not in single_shot_prompt_input
    response_contract = single_shot_prompt_input.get("response_contract") or {}
    assert response_contract.get("top_level_required_fields") == [
        "thought",
        "primary_obligation",
        "macro_tasks",
    ]
    thought_contract = response_contract.get("thought") or {}
    assert thought_contract.get("summary_style") == "2-4 factual sentences"
    assert "Loaded Safety Rules" in single_shot_prompt_text
    assert "Current Resource Facts" in single_shot_prompt_text
    assert "Current Part Facts" in single_shot_prompt_text
    assert "Relevant Assembly Requirements" in single_shot_prompt_text
    assert "Modeled Continuation Gap" in single_shot_prompt_text
    assert "Proposal Success Criteria" not in single_shot_prompt_text
    assert "Required JSON Response Contract" in single_shot_prompt_text
    assert "Hard Constraints" in single_shot_prompt_text
    assert "Allowed Execution Surface" in single_shot_prompt_text
    assert "Observed Runtime State" not in single_shot_prompt_text
    assert "prompt_bridge_snapshot" not in single_shot_prompt_text
    assert "allowed_primitives" in single_shot_prompt_text
    assert "primitive_catalog_by_resource_type" not in single_shot_prompt_text
    assert "bridge_resources" not in single_shot_prompt_text
    assert "focused_primitive_catalog" not in single_shot_prompt_text
    assert "other_resources_summary" not in single_shot_prompt_text
    assert "recovery_opportunities" not in single_shot_prompt_text
    assert "grasp_part" in single_shot_prompt_text
    assert "release_part" in single_shot_prompt_text
    assert "open_gripper" not in single_shot_prompt_text
    assert "close_gripper" not in single_shot_prompt_text
    assert "attach_part" not in single_shot_prompt_text
    assert "detach_part" not in single_shot_prompt_text
    assert "ur5e@localhost" in single_shot_prompt_text
    assert "xarm6@localhost" in single_shot_prompt_text
    assert "Do not contradict grounded runtime facts already present in the prompt." in single_shot_prompt_text
    assert "Return a recovery that restores a state where the blocked nominal tasks can run again" in single_shot_prompt_text
    assert "Make the recovery logically connected as a state progression" in single_shot_prompt_text
    assert "Reuse grounded facts and previously derived outputs." in single_shot_prompt_text
    assert "Do not invent concrete grounded values" in single_shot_prompt_text
    current_part_facts_block = (
        single_shot_prompt_text.split("Current Part Facts\n", 1)[1]
        .split("\n\nLoaded Safety Rules", 1)[0]
    )
    assert '"part_name": "LG"' in current_part_facts_block
    assert '"current_state": "misplaced"' in current_part_facts_block
    assert '"current_location":' not in current_part_facts_block
    assert '"current_state": "unknown"' not in current_part_facts_block

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


def _extract_latest_prompt_text(result: dict[str, Any]) -> str:
    reasoning_mode = str(result.get("reasoning_mode") or "").strip().lower()
    if reasoning_mode == "multi_turn":
        multi_turn_session = dict(result.get("multi_turn_session") or {})
        turns = list(multi_turn_session.get("turns") or [])
        if turns:
            latest_turn = dict(turns[-1] or {})
            return str(latest_turn.get("prompt_text") or "")
    return str(result.get("single_shot_prompt_text") or "")


def _print_prompt(prompt_text: str, *, reasoning_mode: str) -> None:
    print()
    print(f"Prompt ({reasoning_mode or 'single_shot'})")
    print(prompt_text or "")


def _print_debug_artifact_paths(result: dict[str, Any]) -> None:
    prompt_artifact_path = result.get("prompt_artifact_path")
    latest_prompt_artifact_path = result.get("latest_prompt_artifact_path")
    response_artifact_path = result.get("response_artifact_path")
    latest_response_artifact_path = result.get("latest_response_artifact_path")
    session_transcript_artifact_path = result.get("session_transcript_artifact_path")
    latest_session_transcript_artifact_path = result.get(
        "latest_session_transcript_artifact_path"
    )
    if (
        not prompt_artifact_path
        and not latest_prompt_artifact_path
        and not response_artifact_path
        and not latest_response_artifact_path
        and not session_transcript_artifact_path
        and not latest_session_transcript_artifact_path
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
    if session_transcript_artifact_path:
        print("Session artifact:           ", session_transcript_artifact_path)
    if latest_session_transcript_artifact_path:
        print("Latest session artifact:    ", latest_session_transcript_artifact_path)


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
    fixture, _, planner, prepared_bridge_request = asyncio.run(
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
    session_seed = prepared_bridge_request.get("multi_turn_session_seed") or {}
    assert session_seed.get("current_phase") == "grounding"
    assert session_seed.get("observation_store") == {}
    assert session_seed.get("accepted_outline") is None
    llm_input = prepared_bridge_request.get("llm_input") or {}
    assert llm_input.get("part_facts"), "multi-turn should reuse the shared single-shot grounding package"
    llm_parts_by_name = {
        str(row.get("part_name") or ""): row
        for row in (llm_input.get("part_facts") or [])
        if isinstance(row, dict) and str(row.get("part_name") or "")
    }
    assert (llm_parts_by_name.get("LG") or {}).get("observed_pose") is None

    scripted_responses = iter(deepcopy(MOCK_MULTI_TURN_RESPONSES))

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        response = deepcopy(next(scripted_responses))
        self._turn_index += 1
        self.turn_log.append(
            {
                "turn_index": self._turn_index,
                "prompt": "<mocked-structured>",
                "response": deepcopy(response),
            }
        )
        return response

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal == MOCK_MULTI_TURN_FINAL_PROPOSAL
    bridge_debug = planner.get_last_bridge_debug()
    assert bridge_debug.get("status") == "final_proposal_recorded"
    multi_turn_session = bridge_debug.get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "final_proposal_recorded"
    assert multi_turn_session.get("observation_count") == 1
    assert len(multi_turn_session.get("observation_history") or []) == 1
    assert multi_turn_session.get("accepted_outline") == MOCK_MULTI_TURN_OUTLINE_TASKS
    accepted_outline = list(multi_turn_session.get("accepted_outline") or [])
    assert len(accepted_outline) > 1
    outline_by_id = {
        str(row.get("outline_id") or ""): dict(row)
        for row in accepted_outline
        if isinstance(row, dict) and str(row.get("outline_id") or "")
    }
    assert outline_by_id["outline_return_mcp_to_printer"]["depends_on"] == [
        "outline_clear_xarm6"
    ]
    assert outline_by_id["outline_pick_lg_with_ur5e"]["resource_jid"] == "ur5e@localhost"
    assert outline_by_id["outline_place_lg_with_ur5e"]["depends_on"] == [
        "outline_pick_lg_with_ur5e"
    ]
    assert outline_by_id["outline_resume_mcp_assembly"]["depends_on"] == [
        "outline_place_lg_with_ur5e"
    ]
    assert not any(
        "diagnose failure" in str(dict(row).get("description") or "").lower()
        for row in accepted_outline
        if isinstance(row, dict)
    )
    assert not any(
        dict(row).get("resource_jid") == "xarm6@localhost"
        and dict(row).get("part_name") == "LG"
        for row in accepted_outline
        if isinstance(row, dict)
    )
    assert multi_turn_session.get("proposal_draft") == {
        "thought": MOCK_MULTI_TURN_RESPONSES[3]["thought"],
        "macro_tasks": MOCK_MULTI_TURN_MACRO_TASKS,
    }
    assert multi_turn_session.get("final_proposal") == MOCK_MULTI_TURN_FINAL_PROPOSAL
    turns = list(multi_turn_session.get("turns") or [])
    assert [
        turn.get("phase")
        for turn in turns
    ] == [
        "grounding",
        "grounding",
        "outline",
        "primitive_generation",
        "finalize",
    ]
    outline_turn = dict(turns[2] or {})
    outline_task_types = list(outline_turn.get("outline_task_types") or [])
    assert any(
        dict(row).get("outline_id") == "outline_clear_xarm6"
        and dict(row).get("task_type") == "resource_only"
        for row in outline_task_types
        if isinstance(row, dict)
    )
    first_turn = dict(turns[0] or {})
    first_prompt_input = dict(first_turn.get("prompt_input") or {})
    world_observation_surface = dict(first_prompt_input.get("world_observation_surface") or {})
    fact_rows = [
        dict(row)
        for row in (world_observation_surface.get("observation_facts") or [])
        if isinstance(row, dict)
    ]
    assert fact_rows, "grounding world observation surface should be present"
    fact_types = {str(row.get("fact_type") or "").strip() for row in fact_rows}
    assert fact_types == {"part_pose"}
    part_pose_entry = next(
        row for row in fact_rows if str(row.get("fact_type") or "").strip() == "part_pose"
    )
    assert part_pose_entry.get("entity_kind") == "part"
    assert part_pose_entry.get("request_fields") == ["fact_type", "entity"]
    assert part_pose_entry.get("optional_request_fields") == ["scope", "reason"]
    assert part_pose_entry.get("output_fields") == ["part_name", "x", "y", "z", "pose", "orientation"]
    first_prompt_text = str(first_turn.get("prompt_text") or "")
    assert "Session Observation Store" not in first_prompt_text
    assert "Observation Fact Ledger" not in first_prompt_text
    assert "Latest Session Delta" not in first_prompt_text
    assert "Session Turn History" not in first_prompt_text
    assert "World Observation Surface" in first_prompt_text
    assert "\"observed_pose\": null" in first_prompt_text
    assert "observed_pose_LG" not in first_prompt_text
    assert "\"store_as\"" not in first_prompt_text
    second_turn = dict(turns[1] or {})
    second_prompt_input = dict(second_turn.get("prompt_input") or {})
    second_llm_parts_by_name = {
        str(row.get("part_name") or ""): row
        for row in (dict(second_prompt_input.get("llm_input") or {}).get("part_facts") or [])
        if isinstance(row, dict) and str(row.get("part_name") or "")
    }
    second_lg_row = dict(second_llm_parts_by_name.get("LG") or {})
    assert second_lg_row.get("current_state") == "misplaced"
    assert second_lg_row.get("observed_pose") == {"x": 0.0, "y": 0.2, "z": 1.035}
    assert second_lg_row.get("location_basis") == "session_observation"
    assert second_lg_row.get("observation_status") == "observed"
    assert second_lg_row.get("observed_in_session") is True
    assert second_lg_row.get("observed_by") == "detect_parts"
    assert second_lg_row.get("observed_by_primitive") == "detect_parts"
    assert second_lg_row.get("observed_fact_type") == "part_pose"
    assert second_lg_row.get("observed_store_as") == "observed_pose_LG"
    assert second_lg_row.get("observed_aliases") == ["observed_pose_LG"]
    assert second_lg_row.get("observed_turn_index") == 1
    second_prompt_text = str(second_turn.get("prompt_text") or "")
    assert "Observation Fulfillment Status" in second_prompt_text
    assert "Latest Session Delta" in second_prompt_text
    assert "Session Observation Store" not in second_prompt_text
    assert "Observation Fact Ledger" not in second_prompt_text
    assert "Session Turn History" not in second_prompt_text
    assert "Phase Feedback" not in second_prompt_text
    assert '"observation_status": "observed"' in second_prompt_text
    assert '"fact_type": "part_pose"' in second_prompt_text
    assert "\"observed_pose\": {\n      \"x\": 0.0,\n      \"y\": 0.2,\n      \"z\": 1.035\n    }" in second_prompt_text
    assert "\"observed_store_as\"" not in second_prompt_text
    assert "\"observed_aliases\"" not in second_prompt_text
    assert "\"store_as\"" not in second_prompt_text
    observation_history = multi_turn_session.get("observation_history") or []
    assert observation_history[0].get("store_as") == "observed_pose_LG"
    assert observation_history[0].get("fact_type") == "part_pose"
    assert observation_history[0].get("entity") == "LG"
    assert "resource_jid" not in observation_history[0]
    assert (multi_turn_session.get("observation_store") or {}).get("observed_pose_LG")
    fact_ledger = dict(multi_turn_session.get("observation_fact_ledger") or {})
    assert fact_ledger
    ledger_values = [dict(row) for row in fact_ledger.values() if isinstance(row, dict)]
    assert any(
        row.get("fact_type") == "part_pose"
        and row.get("entity") == "LG"
        and row.get("aliases") == ["observed_pose_LG"]
        for row in ledger_values
    )
    assert fixture.get("part_tracker", {}).get("LG"), "fixture sanity check failed"


def test_case3_bridge_multi_turn_can_ground_without_observation() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )

    scripted_responses = iter(
        [
            {
                "thought": "The failure state and symbolic continuation gaps are already sufficient to move into outline planning.",
                "decision": "grounded",
                "blocking_reasons": [
                    "xarm6 is failed.",
                    "ur5e holds MCP while LG still needs assembly.",
                ],
                "grounded_facts": [
                    "xarm6 is failed and cannot complete its nominal tail.",
                    "SAFE_1 keeps MCP blocked until LG is placed.",
                ],
                "recovery_implications": [
                    "Grounding is already sufficient to begin outlining a replacement sequence without new observation.",
                ],
                "sufficient_grounding": True,
                "observe_requests": [],
            },
            {
                "thought": "The recovery still needs one macro to clear xarm6 and one macro to restore the blocked ordering.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
            },
            {
                "thought": "The accepted outline resolves into primitive-connected macro tasks.",
                "decision": "draft_ready",
                "macro_tasks": deepcopy(MOCK_MULTI_TURN_MACRO_TASKS),
            },
            {
                "thought": "The draft is now packaged as the final bridge proposal.",
                "decision": "final_ready",
                "final_proposal": deepcopy(MOCK_MULTI_TURN_FINAL_PROPOSAL),
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal == MOCK_MULTI_TURN_FINAL_PROPOSAL
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    assert multi_turn_session.get("observation_count") == 0
    assert [
        turn.get("phase")
        for turn in (multi_turn_session.get("turns") or [])
    ] == [
        "grounding",
        "outline",
        "primitive_generation",
        "finalize",
    ]


def test_case3_bridge_multi_turn_can_pause_after_grounding() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "grounding"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    scripted_responses = iter(deepcopy(MOCK_MULTI_TURN_RESPONSES))

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    bridge_debug = planner.get_last_bridge_debug() or {}
    assert bridge_debug.get("status") == "paused_after_grounding"
    multi_turn_session = bridge_debug.get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_grounding"
    assert multi_turn_session.get("current_phase") == "outline"
    turns = list(multi_turn_session.get("turns") or [])
    assert [turn.get("phase") for turn in turns] == ["grounding", "grounding"]
    last_response = dict(turns[-1].get("raw_response") or {})
    assert last_response.get("decision") == "grounded"
    assert last_response.get("grounded_facts")
    assert last_response.get("recovery_implications")


def test_case3_bridge_multi_turn_normalizes_contradictory_grounded_boolean() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "grounding"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    scripted_turns = deepcopy(MOCK_MULTI_TURN_RESPONSES[:2])
    scripted_turns[1]["sufficient_grounding"] = False
    scripted_responses = iter(scripted_turns)

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    bridge_debug = planner.get_last_bridge_debug() or {}
    assert bridge_debug.get("status") == "paused_after_grounding"
    multi_turn_session = bridge_debug.get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_grounding"
    assert multi_turn_session.get("current_phase") == "outline"
    turns = list(multi_turn_session.get("turns") or [])
    assert [turn.get("phase") for turn in turns] == ["grounding", "grounding"]

    last_turn = dict(turns[-1] or {})
    assert last_turn.get("decision") == "grounded"
    assert last_turn.get("sufficient_grounding") is True
    assert dict(last_turn.get("raw_response") or {}).get("sufficient_grounding") is False
    normalization = dict(last_turn.get("grounding_decision_normalization") or {})
    assert normalization.get("field") == "sufficient_grounding"
    assert normalization.get("from") is False
    assert normalization.get("to") is True


def test_case3_bridge_multi_turn_grounding_observation_surfaces_mcp_location_in_dry_run() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "grounding"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    scripted_responses = iter(
        [
            {
                "thought": "Grounding still needs the live LG pose and the currently held MCP pose before recovery can proceed.",
                "decision": "observe",
                "blocking_reasons": [
                    "LG is still misplaced.",
                    "MCP remains held by ur5e while recovery sequencing is unresolved.",
                ],
                "grounded_facts": [
                    "xarm6 is failed at the blocked LG suffix.",
                ],
                "recovery_implications": [
                    "Grounding needs both part poses so the next step can reason over the actual recovery geometry.",
                ],
                "sufficient_grounding": False,
                "observe_reason": "Observe both LG and MCP before finalizing the grounded recovery state.",
                "observe_requests": [
                    {
                        "fact_type": "part_pose",
                        "entity": "LG",
                        "store_as": "lg_detection_seed",
                    },
                    {
                        "fact_type": "part_pose",
                        "entity": "MCP",
                        "store_as": "mcp_detection_seed",
                    },
                ],
            },
            {
                "thought": "The stored LG and MCP observations now make grounding sufficient for the next outline turn.",
                "decision": "grounded",
                "blocking_reasons": [
                    "xarm6 is failed.",
                    "ur5e still holds MCP while LG needs recovery.",
                ],
                "grounded_facts": [
                    "LG has a grounded runtime pose.",
                    "MCP is observed in ur5e's gripper.",
                ],
                "recovery_implications": [
                    "The outline can now reason over both the misplaced LG and the held MCP.",
                ],
                "sufficient_grounding": True,
                "observe_requests": [],
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    bridge_debug = planner.get_last_bridge_debug() or {}
    assert bridge_debug.get("status") == "paused_after_grounding"
    multi_turn_session = bridge_debug.get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_grounding"
    assert multi_turn_session.get("observation_count") == 2
    turns = list(multi_turn_session.get("turns") or [])
    assert [turn.get("phase") for turn in turns] == ["grounding", "grounding"]

    observation_history = list(multi_turn_session.get("observation_history") or [])
    assert {row.get("entity") for row in observation_history} == {"LG", "MCP"}
    mcp_observation = next(
        dict(row)
        for row in observation_history
        if str(row.get("entity") or "").strip() == "MCP"
    )
    assert dict(mcp_observation.get("output") or {}) == {
        "part_name": "MCP",
        "x": 0.0,
        "y": -0.08,
        "z": 1.025,
        "pose": {"x": 0.0, "y": -0.08, "z": 1.025},
        "current_location": "ur5e@localhost_gripper",
        "current_holder_resource_jid": "ur5e@localhost",
    }

    second_turn = dict(turns[1] or {})
    second_prompt_input = dict(second_turn.get("prompt_input") or {})
    second_llm_parts_by_name = {
        str(row.get("part_name") or ""): row
        for row in (dict(second_prompt_input.get("llm_input") or {}).get("part_facts") or [])
        if isinstance(row, dict) and str(row.get("part_name") or "")
    }
    second_mcp_row = dict(second_llm_parts_by_name.get("MCP") or {})
    assert second_mcp_row.get("observed_pose") == {"x": 0.0, "y": -0.08, "z": 1.025}
    assert second_mcp_row.get("current_location") == "ur5e@localhost_gripper"
    assert second_mcp_row.get("current_holder_resource_jid") == "ur5e@localhost"
    assert second_mcp_row.get("location_basis") == "session_observation"
    assert second_mcp_row.get("observation_status") == "observed"
    assert second_mcp_row.get("observed_store_as") == "observed_pose_MCP"

    second_prompt_text = str(second_turn.get("prompt_text") or "")
    assert '"current_location": "ur5e@localhost_gripper"' in second_prompt_text
    assert '"current_holder_resource_jid": "ur5e@localhost"' in second_prompt_text


def test_case3_bridge_multi_turn_fact_request_resolves_to_detect_parts() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    session_state = build_multi_turn_session_seed(prepared_bridge_request)
    session_state["turn_index"] = 1

    observation_results, observe_error = asyncio.run(
        _execute_observe_requests(
            planner,
            prepared_bridge_request,
            session_state,
            [
                {
                    "fact_type": "part_pose",
                    "entity": "LG",
                    "store_as": "observed_lg_pose",
                }
            ],
        )
    )

    assert observe_error is None
    assert len(observation_results) == 1
    observation_row = dict(observation_results[0] or {})
    assert observation_row.get("fact_type") == "part_pose"
    assert observation_row.get("entity") == "LG"
    assert observation_row.get("primitive") == "detect_parts"
    assert observation_row.get("params") == {"part_name": "LG"}
    assert dict(observation_row.get("output") or {}).get("pose") == {
        "x": 0.0,
        "y": 0.2,
        "z": 1.035,
    }


def test_case3_bridge_multi_turn_observe_request_without_store_as_is_supported() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    session_state = build_multi_turn_session_seed(prepared_bridge_request)
    session_state["turn_index"] = 1

    observation_results, observe_error = asyncio.run(
        _execute_observe_requests(
            planner,
            prepared_bridge_request,
            session_state,
            [
                {
                    "fact_type": "part_pose",
                    "entity": "LG",
                }
            ],
        )
    )

    assert observe_error is None
    assert len(observation_results) == 1
    observation_row = dict(observation_results[0] or {})
    assert observation_row.get("store_as") == "observed_pose_LG"
    assert observation_row.get("fact_type") == "part_pose"
    assert observation_row.get("entity") == "LG"


def test_case3_bridge_multi_turn_fact_request_resolves_held_mcp_with_location() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    session_state = build_multi_turn_session_seed(prepared_bridge_request)
    session_state["turn_index"] = 1

    observation_results, observe_error = asyncio.run(
        _execute_observe_requests(
            planner,
            prepared_bridge_request,
            session_state,
            [
                {
                    "fact_type": "part_pose",
                    "entity": "MCP",
                    "store_as": "observed_mcp_pose",
                }
            ],
        )
    )

    assert observe_error is None
    assert len(observation_results) == 1
    observation_row = dict(observation_results[0] or {})
    assert observation_row.get("fact_type") == "part_pose"
    assert observation_row.get("entity") == "MCP"
    assert observation_row.get("primitive") == "detect_parts"
    assert observation_row.get("params") == {"part_name": "MCP"}
    assert dict(observation_row.get("output") or {}) == {
        "part_name": "MCP",
        "x": 0.0,
        "y": -0.08,
        "z": 1.025,
        "pose": {"x": 0.0, "y": -0.08, "z": 1.025},
        "current_location": "ur5e@localhost_gripper",
        "current_holder_resource_jid": "ur5e@localhost",
    }


def test_case3_bridge_multi_turn_repeated_observation_is_semantic_duplicate() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    session_state = build_multi_turn_session_seed(prepared_bridge_request)
    session_state["observation_store"] = {
        "observed_lg_pose": {
            "part_name": "LG",
            "x": 0.0,
            "y": 0.2,
            "z": 1.035,
            "pose": {"x": 0.0, "y": 0.2, "z": 1.035},
        }
    }
    session_state["observation_fact_ledger"] = {
        "{\"entity\": \"LG\", \"fact_type\": \"part_pose\"}": {
            "fact_key": "{\"entity\": \"LG\", \"fact_type\": \"part_pose\"}",
            "fact_type": "part_pose",
            "entity": "LG",
            "entity_kind": "part",
            "primitive": "detect_parts",
            "params": {"part_name": "LG"},
            "output": {
                "part_name": "LG",
                "x": 0.0,
                "y": 0.2,
                "z": 1.035,
                "pose": {"x": 0.0, "y": 0.2, "z": 1.035},
            },
            "turn_index": 1,
            "validity": "current",
            "freshness": "current_session",
            "aliases": ["observed_lg_pose"],
        }
    }
    session_state["observation_history"] = [
        {
            "turn_index": 1,
            "phase": "grounding",
            "fact_key": "{\"entity\": \"LG\", \"fact_type\": \"part_pose\"}",
            "fact_type": "part_pose",
            "entity": "LG",
            "entity_kind": "part",
            "primitive": "detect_parts",
            "params": {"part_name": "LG"},
            "store_as": "observed_lg_pose",
            "output": {
                "part_name": "LG",
                "x": 0.0,
                "y": 0.2,
                "z": 1.035,
                "pose": {"x": 0.0, "y": 0.2, "z": 1.035},
            },
        }
    ]

    observation_results, observe_error = asyncio.run(
        _execute_observe_requests(
            planner,
            prepared_bridge_request,
            session_state,
            [
                {
                    "fact_type": "part_pose",
                    "entity": "LG",
                    "store_as": "observed_lg_pose_v2",
                }
            ],
        )
    )

    assert observation_results == []
    assert observe_error
    assert "part_pose" in observe_error
    assert "LG" in observe_error


def test_case3_bridge_multi_turn_duplicate_observe_request_becomes_repair_feedback() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "grounding"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    scripted_responses = iter(
        [
            {
                "thought": "LG still needs a runtime pose observation before recovery can be grounded.",
                "decision": "observe",
                "blocking_reasons": [
                    "LG is not yet assembled on the board.",
                ],
                "grounded_facts": [
                    "xarm6 is failed at the blocked LG suffix.",
                ],
                "recovery_implications": [
                    "Recovery planning still needs a concrete LG pose.",
                ],
                "sufficient_grounding": False,
                "observe_reason": "Grounding is not yet sufficient because LG still needs a runtime pose observation.",
                "observe_requests": [
                    {
                        "fact_type": "part_pose",
                        "entity": "LG",
                        "scope": "environment",
                    }
                ],
            },
            {
                "thought": "The same LG observation is being requested again.",
                "decision": "observe",
                "blocking_reasons": [
                    "LG is still misplaced.",
                ],
                "grounded_facts": [
                    "LG was already observed once.",
                ],
                "recovery_implications": [
                    "The loop should repair the duplicate observation request instead of stopping.",
                ],
                "sufficient_grounding": False,
                "observe_reason": "The same fact is requested again.",
                "observe_requests": [
                    {
                        "fact_type": "part_pose",
                        "entity": "LG",
                        "scope": "environment",
                    }
                ],
            },
            {
                "thought": "The fulfilled LG observation is already in session state, so grounding can now move to outline.",
                "decision": "grounded",
                "blocking_reasons": [
                    "xarm6 remains failed.",
                ],
                "grounded_facts": [
                    "LG has already been observed in the session.",
                ],
                "recovery_implications": [
                    "The outline phase can use the fulfilled LG pose fact without another observation.",
                ],
                "observe_requests": [],
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    bridge_debug = planner.get_last_bridge_debug() or {}
    assert bridge_debug.get("status") == "paused_after_grounding"
    multi_turn_session = bridge_debug.get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_grounding"
    assert multi_turn_session.get("current_phase") == "outline"
    turns = list(multi_turn_session.get("turns") or [])
    assert [turn.get("phase") for turn in turns] == ["grounding", "grounding", "grounding"]
    repair_turn = dict(turns[1] or {})
    assert "already succeeded earlier" in str(repair_turn.get("error") or "")
    grounding_contract = dict(repair_turn.get("grounding_contract") or {})
    assert grounding_contract.get("status") == "failed"
    assert grounding_contract.get("reason") == "observe_already_fulfilled"
    fulfillment = dict(grounding_contract.get("observation_fulfillment") or {})
    assert fulfillment.get("fulfilled_count") == 1
    assert fulfillment.get("unfulfilled_count") == 0
    third_prompt_text = str(turns[2].get("prompt_text") or "")
    assert "Grounding Contract" in third_prompt_text
    assert "\"observe_already_fulfilled\"" in third_prompt_text
    phase_feedback = list(multi_turn_session.get("phase_feedback") or [])
    assert any(
        dict(dict(row).get("detail") or {}).get("grounding_contract", {}).get("reason")
        == "observe_already_fulfilled"
        for row in phase_feedback
        if isinstance(row, dict)
    )


def test_case3_bridge_multi_turn_invalid_part_pose_entity_stops_session() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )

    scripted_responses = iter(
        [
            {
                "thought": "Grounding still needs live confirmation of LG and xarm6 before recovery can proceed.",
                "decision": "observe",
                "blocking_reasons": [
                    "xarm6 is failed.",
                    "LG remains misplaced.",
                ],
                "grounded_facts": [
                    "xarm6 failed during LG placement.",
                ],
                "recovery_implications": [
                    "Recovery still needs a grounded LG pose.",
                ],
                "sufficient_grounding": False,
                "observe_reason": "Confirm LG pose and xarm6 positioning.",
                "observe_requests": [
                    {
                        "fact_type": "part_pose",
                        "entity": "LG",
                        "store_as": "lg_pose",
                    },
                    {
                        "fact_type": "part_pose",
                        "entity": "xarm6",
                        "store_as": "xarm6_pose",
                    },
                ],
            }
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    bridge_debug = planner.get_last_bridge_debug() or {}
    assert bridge_debug.get("status") == "invalid_observe_request"
    multi_turn_session = bridge_debug.get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "invalid_observe_request"
    turns = list(multi_turn_session.get("turns") or [])
    assert [turn.get("phase") for turn in turns] == ["grounding"]
    assert "expects a part entity" in str(turns[-1].get("error") or "")
    assert "'xarm6'" in str(turns[-1].get("error") or "")


def test_case3_bridge_multi_turn_empty_observe_after_fulfilled_fact_stays_in_grounding() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "grounding"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    scripted_responses = iter(
        [
            {
                "thought": "LG still needs a runtime pose observation before recovery can be grounded.",
                "decision": "observe",
                "blocking_reasons": [
                    "LG is not yet assembled on the board.",
                ],
                "grounded_facts": [
                    "xarm6 is failed at the blocked LG suffix.",
                ],
                "recovery_implications": [
                    "Recovery planning still needs a concrete LG pose.",
                ],
                "sufficient_grounding": False,
                "observe_reason": "Grounding is not yet sufficient because LG still needs a runtime pose observation.",
                "observe_requests": [
                    {
                        "fact_type": "part_pose",
                        "entity": "LG",
                        "scope": "environment",
                        "store_as": "actual_lg_position",
                    }
                ],
            },
            {
                "thought": "The LG observation is already available in the session facts, so grounding can advance.",
                "decision": "observe",
                "blocking_reasons": [
                    "xarm6 remains failed.",
                ],
                "grounded_facts": [
                    "LG has already been observed in the session.",
                ],
                "recovery_implications": [
                    "The outline phase can now use the fulfilled LG pose fact.",
                ],
                "sufficient_grounding": False,
                "observe_reason": "The needed observation has already been completed.",
                "observe_requests": [],
            },
            {
                "thought": "The fulfilled LG observation is already in the session store, so grounding is now sufficient to move to outline.",
                "decision": "grounded",
                "blocking_reasons": [
                    "xarm6 remains failed.",
                ],
                "grounded_facts": [
                    "LG has already been observed in the session.",
                ],
                "recovery_implications": [
                    "The outline phase can now use the fulfilled LG pose fact.",
                ],
                "sufficient_grounding": True,
                "observe_requests": [],
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    bridge_debug = planner.get_last_bridge_debug() or {}
    assert bridge_debug.get("status") == "paused_after_grounding"
    multi_turn_session = bridge_debug.get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_grounding"
    assert multi_turn_session.get("current_phase") == "outline"
    turns = list(multi_turn_session.get("turns") or [])
    assert [turn.get("phase") for turn in turns] == ["grounding", "grounding", "grounding"]
    contract_turn = dict(turns[1] or {})
    assert contract_turn.get("decision") == "observe"
    assert dict(contract_turn.get("raw_response") or {}).get("decision") == "observe"
    assert contract_turn.get("decision_compatibility") is None
    grounding_contract = dict(contract_turn.get("grounding_contract") or {})
    assert grounding_contract.get("status") == "failed"
    assert grounding_contract.get("reason") == "observe_without_requests"
    fulfillment = dict(grounding_contract.get("observation_fulfillment") or {})
    assert fulfillment.get("fulfilled_count") == 1
    assert fulfillment.get("unfulfilled_count") == 0
    third_prompt_text = str(turns[2].get("prompt_text") or "")
    assert "Grounding Contract" in third_prompt_text
    assert "\"observe_without_requests\"" in third_prompt_text
    phase_feedback = list(multi_turn_session.get("phase_feedback") or [])
    assert any(
        dict(dict(row).get("detail") or {}).get("grounding_contract", {}).get("reason")
        == "observe_without_requests"
        for row in phase_feedback
        if isinstance(row, dict)
    )


def test_case3_bridge_multi_turn_accepts_legacy_blocking_summary() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "grounding"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    scripted_responses = iter(
        [
            {
                "thought": "Legacy grounding response still uses blocking_summary.",
                "decision": "grounded",
                "blocking_summary": [
                    "xarm6 is failed.",
                    "LG still needs recovery planning.",
                ],
                "grounded_facts": [
                    "The symbolic failure context is already enough to begin outlining recovery.",
                ],
                "recovery_implications": [
                    "Grounding may proceed without additional observation in this scripted case.",
                ],
                "sufficient_grounding": True,
                "observe_requests": [],
            }
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    turns = list(((planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}).get("turns") or [])
    assert turns
    assert turns[0].get("blocking_reasons") == [
        "xarm6 is failed.",
        "LG still needs recovery planning.",
    ]
    assert "blocking_summary" not in dict(turns[0] or {})


def test_case3_bridge_multi_turn_transitions() -> None:
    assert transition_multi_turn_phase("grounding", "observe") == "grounding"
    assert transition_multi_turn_phase("grounding", "grounded") == "outline"
    assert transition_multi_turn_phase("outline", "need_revision") == "outline"
    assert transition_multi_turn_phase("outline", "outline_ready") == "primitive_generation"
    assert transition_multi_turn_phase("primitive_generation", "need_outline_revision") == "outline"
    assert transition_multi_turn_phase("primitive_generation", "draft_ready") == "finalize"
    assert transition_multi_turn_phase("finalize", "need_outline_revision") == "outline"
    assert (
        transition_multi_turn_phase("finalize", "need_primitive_revision")
        == "primitive_generation"
    )
    assert transition_multi_turn_phase("finalize", "final_ready") == "finalize"
    with pytest.raises(ValueError, match="unsupported multi-turn transition"):
        transition_multi_turn_phase("outline", "need_grounding")
    with pytest.raises(ValueError, match="unsupported multi-turn transition"):
        transition_multi_turn_phase("primitive_generation", "need_grounding")
    with pytest.raises(ValueError, match="unsupported multi-turn transition"):
        transition_multi_turn_phase("finalize", "need_grounding")


def test_case3_bridge_multi_turn_post_grounding_schemas_close_grounding() -> None:
    grounding_schema = ((multi_turn_phase_response_schema("grounding") or {}).get("schema") or {})
    outline_schema = ((multi_turn_phase_response_schema("outline") or {}).get("schema") or {})
    grounding_properties = dict(grounding_schema.get("properties") or {})
    grounding_required = list(grounding_schema.get("required") or [])
    grounding_observe_item_schema = dict(
        dict(grounding_properties.get("observe_requests") or {}).get("items") or {}
    )
    grounding_observe_properties = dict(grounding_observe_item_schema.get("properties") or {})
    grounding_observe_required = list(grounding_observe_item_schema.get("required") or [])
    primitive_enum = (
        ((multi_turn_phase_response_schema("primitive_generation") or {}).get("schema") or {})
        .get("properties", {})
        .get("decision", {})
        .get("enum", [])
    )
    finalize_enum = (
        ((multi_turn_phase_response_schema("finalize") or {}).get("schema") or {})
        .get("properties", {})
        .get("decision", {})
        .get("enum", [])
    )
    outline_properties = dict(outline_schema.get("properties") or {})
    outline_task_schema = dict(
        dict(outline_properties.get("outline_tasks") or {}).get("items") or {}
    )
    outline_task_properties = dict(outline_task_schema.get("properties") or {})
    outline_task_required = list(outline_task_schema.get("required") or [])
    outline_required = list(outline_schema.get("required") or [])
    assert "sufficient_grounding" not in grounding_properties
    assert "sufficient_grounding" not in grounding_required
    assert "store_as" not in grounding_observe_properties
    assert grounding_observe_required == ["fact_type", "entity"]
    assert "decision" not in outline_properties
    assert "addressed_validation_findings" in outline_properties
    assert "addressed_validation_findings" in outline_required
    assert "macro_name" not in outline_task_properties
    assert "task_action" not in outline_task_properties
    assert "task_action" not in outline_task_required
    assert "closes_condition_ids" not in outline_task_properties
    assert primitive_enum == ["need_outline_revision", "draft_ready"]
    assert finalize_enum == [
        "final_ready",
        "need_outline_revision",
        "need_primitive_revision",
    ]
    assert "need_grounding" not in primitive_enum
    assert "need_grounding" not in finalize_enum


def test_case3_bridge_multi_turn_outline_validation_reprompts_in_outline() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )
    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This draft incorrectly keeps LG recovery on xarm6.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_INVALID_XARM_OUTLINE_TASKS),
            },
            {
                "thought": "The outline should stay in outline and reassign LG recovery to ur5e using the existing grounded facts.",
                "addressed_validation_findings": deepcopy(
                    MOCK_INVALID_XARM_REQUIRED_FINDING_REFS
                ),
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    bridge_debug = planner.get_last_bridge_debug() or {}
    assert bridge_debug.get("status") == "paused_after_outline"
    multi_turn_session = bridge_debug.get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_outline"
    assert multi_turn_session.get("current_phase") == "outline"
    assert multi_turn_session.get("paused_before_phase_transition") == "primitive_generation"
    turns = list(multi_turn_session.get("turns") or [])
    assert [turn.get("phase") for turn in turns] == [
        "grounding",
        "grounding",
        "outline",
        "outline",
    ]
    assert turns[2].get("decision") == "need_revision"
    assert turns[2].get("validation_violations")
    assert dict(turns[2].get("outline_validation") or {}).get("status") == "failed"
    fourth_prompt_text = str(turns[3].get("prompt_text") or "")
    assert "Available Resource Task Actions" not in fourth_prompt_text
    assert "\"task_action\":" not in fourth_prompt_text
    assert "Repair Contract" in fourth_prompt_text
    assert "\"required_addressed_validation_findings\"" in fourth_prompt_text
    assert "Blocked Nominal Tasks" in fourth_prompt_text
    assert "Unmet Continuation Conditions" in fourth_prompt_text
    assert "Outline Validation Violations" not in fourth_prompt_text
    assert "ur5e@localhost" in fourth_prompt_text
    assert "Current Resource Facts" in fourth_prompt_text
    assert "Current Part Facts" in fourth_prompt_text
    assert "\"failed_axes\"" in fourth_prompt_text
    assert "\"failed_reason\"" in fourth_prompt_text
    assert "\"addressed_validation_findings\"" in fourth_prompt_text
    assert "\"workspace_contains_observed_pose\"" not in fourth_prompt_text
    assert "\"currently_ready_to_acquire\"" not in fourth_prompt_text
    assert "\"readiness_blockers\"" not in fourth_prompt_text
    assert "\"temporary_state_blockers\"" not in fourth_prompt_text
    assert "\"required_condition_ids_to_clear\"" not in fourth_prompt_text
    assert "\"condition_id\"" not in fourth_prompt_text
    assert "\"guidance\"" not in fourth_prompt_text
    assert "Part Reachability Matrix" not in fourth_prompt_text
    assert "\"reachable_resource_jids\"" not in fourth_prompt_text
    assert "\"workspace_reachable_resource_jids\"" not in fourth_prompt_text
    assert "grounding is closed" not in fourth_prompt_text.lower()
    assert "each outline macro must be state-consistent" in fourth_prompt_text.lower()
    assert (
        "resource-limit or safety failures are not resolved by only changing a resource's internal state, labels, or expected values"
        in fourth_prompt_text.lower()
    )
    assert "robot-only recovery tasks must not name a part" not in fourth_prompt_text.lower()
    assert "cannot be resolved by adjusting expected states" not in fourth_prompt_text.lower()
    assert "\"need_grounding\"" not in fourth_prompt_text
    assert "Grounded Feasibility Facts" not in fourth_prompt_text
    assert "Recovery Gap State" not in fourth_prompt_text
    assert "Previous Outline Attempt" not in fourth_prompt_text
    assert "Session Turn History" not in fourth_prompt_text
    assert "Session State" not in fourth_prompt_text
    assert "Session Observation Store" not in fourth_prompt_text
    assert fourth_prompt_text.index("Unmet Continuation Conditions") < fourth_prompt_text.index(
        "Repair Contract"
    )


def test_case3_bridge_multi_turn_outline_validation_runs_on_every_outline_turn() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This draft still incorrectly keeps LG recovery on xarm6.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_INVALID_XARM_OUTLINE_TASKS),
            },
            {
                "thought": "The outline now uses ur5e for LG recovery while keeping the same grounded facts.",
                "addressed_validation_findings": deepcopy(
                    MOCK_INVALID_XARM_REQUIRED_FINDING_REFS
                ),
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    bridge_debug = planner.get_last_bridge_debug() or {}
    assert bridge_debug.get("status") == "paused_after_outline"
    multi_turn_session = bridge_debug.get("multi_turn_session") or {}
    assert multi_turn_session.get("current_phase") == "outline"
    assert multi_turn_session.get("paused_before_phase_transition") == "primitive_generation"
    turns = list(multi_turn_session.get("turns") or [])
    assert [turn.get("phase") for turn in turns] == [
        "grounding",
        "grounding",
        "outline",
        "outline",
    ]
    third_outline_validation = dict(turns[2].get("outline_validation") or {})
    assert third_outline_validation.get("status") == "failed"
    fourth_prompt_text = str(turns[3].get("prompt_text") or "")
    assert "Repair Contract" in fourth_prompt_text
    assert "\"required_addressed_validation_findings\"" in fourth_prompt_text
    assert "Grounded Feasibility Facts" not in fourth_prompt_text
    assert "Recovery Gap State" not in fourth_prompt_text
    assert "Previous Outline Attempt" not in fourth_prompt_text
    assert "Session Turn History" not in fourth_prompt_text
    fourth_outline_validation = dict(turns[3].get("outline_validation") or {})
    assert fourth_outline_validation.get("status") == "passed"
    assert all(
        dict(turn.get("outline_validation") or {}).get("status") != "not_run"
        for turn in turns
        if turn.get("phase") == "outline"
    )


def test_case3_bridge_multi_turn_outline_revision_requires_addressed_validation_findings() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This draft still incorrectly keeps LG recovery on xarm6.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_INVALID_XARM_OUTLINE_TASKS),
            },
            {
                "thought": "This revision fixes the resource assignment but does not explicitly acknowledge the prior validator finding refs.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
            },
            {
                "thought": "This revision explicitly addresses the outstanding validator finding refs and keeps LG recovery on ur5e.",
                "addressed_validation_findings": deepcopy(
                    MOCK_INVALID_XARM_REQUIRED_FINDING_REFS
                ),
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_outline"
    assert multi_turn_session.get("current_phase") == "outline"
    turns = list(multi_turn_session.get("turns") or [])
    assert [turn.get("phase") for turn in turns] == [
        "grounding",
        "grounding",
        "outline",
        "outline",
        "outline",
    ]
    assert turns[2].get("decision") == "need_revision"
    assert turns[3].get("decision") == "need_revision"
    coverage = dict(turns[3].get("outline_revision_coverage") or {})
    assert coverage.get("status") == "failed"
    assert any(
        "missing required refs" in str(item)
        for item in (coverage.get("violations") or [])
    )
    fifth_prompt_text = str(turns[4].get("prompt_text") or "")
    assert "Repair Contract" in fifth_prompt_text
    assert "\"revision_feedback\"" in fifth_prompt_text
    assert "\"required_addressed_validation_findings\"" in fifth_prompt_text
    assert "outline_recover_lg_with_xarm6" in fifth_prompt_text
    assert "\"addressed_validation_findings\"" in fifth_prompt_text
    assert dict(turns[4].get("outline_validation") or {}).get("status") == "passed"


def test_case3_bridge_multi_turn_outline_validation_accepts_condition_closure_annotations() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )
    condition_ids = _case3_continuation_condition_ids(prepared_bridge_request)

    valid_outline = deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS)
    valid_outline[0]["closes_condition_ids"] = [condition_ids["xarm6_idle"]]
    valid_outline[3]["closes_condition_ids"] = [condition_ids["mcp_safe1"]]

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This outline explicitly labels which continuation blockers each recovery macro clears.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": valid_outline,
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_outline"
    accepted_outline = list(multi_turn_session.get("accepted_outline") or [])
    assert accepted_outline == valid_outline
    turns = list(multi_turn_session.get("turns") or [])
    outline_turn = dict(turns[2] or {})
    assert outline_turn.get("decision") == "outline_ready"
    assert not outline_turn.get("validation_violations")
    assert dict(outline_turn.get("outline_validation") or {}).get("status") == "passed"


def test_case3_bridge_multi_turn_outline_validation_ignores_extra_closes_condition_ids() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )
    condition_ids = _case3_continuation_condition_ids(prepared_bridge_request)

    invalid_outline = deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS)
    invalid_outline[0]["closes_condition_ids"] = [condition_ids["mcp_safe1"]]
    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This outline still adds legacy closes_condition_ids fields, but the runtime should derive continuation semantics from the rollout.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": invalid_outline,
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_outline"
    turns = list(multi_turn_session.get("turns") or [])
    outline_turn = dict(turns[2] or {})
    assert outline_turn.get("decision") == "outline_ready"
    assert dict(outline_turn.get("outline_validation") or {}).get("status") == "passed"
    findings = list(outline_turn.get("outline_validation_findings") or [])
    assert not any(
        "claimed_condition_not_cleared" in str(dict(item).get("failed_axes") or [])
        or "claimed_condition_not_currently_unmet" in str(dict(item).get("failed_axes") or [])
        for item in findings
        if isinstance(item, dict)
    )


def test_case3_bridge_multi_turn_outline_validation_accepts_uniquely_inferable_part_binding() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This outline omits task.part_name even though the task clearly references LG.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": [
                    {
                        "outline_id": "OT1",
                        "resource_jid": "xarm6@localhost",
                        "description": "Recover the xarm6 robot to a functioning state to resume operations.",
                        "rationale": "xarm6 is blocked in a failed state and must be restored first.",
                        "action_target": {
                            "named_pose": "home",
                        },
                        "expected_start_state": {
                            "current_state": "failed",
                            "held_part": None,
                            "position": {
                                "x": 0.1,
                                "y": 0.08,
                                "z": 1.1994999760206477,
                            },
                            "gripper_state": "open",
                        },
                        "expected_end_state": {
                            "current_state": "idle",
                            "held_part": None,
                            "gripper_state": "open",
                        },
                        "depends_on": [],
                    },
                    {
                        "outline_id": "OT2",
                        "resource_jid": "xarm6@localhost",
                        "description": "Pick LG from the grounded misplaced pose after xarm6 recovers.",
                        "rationale": "Positioning LG properly is required before downstream work can continue.",
                        "action_target": {
                            "source_location": "observed_pose",
                            "target_location": "assembly_board-v1",
                        },
                        "expected_start_state": {
                            "position": {"x": 0.0, "y": 0.2, "z": 1.035},
                            "current_state": "idle",
                            "held_part": None,
                            "gripper_state": "open",
                        },
                        "expected_end_state": {
                            "current_state": "busy",
                            "held_part": "LG",
                            "gripper_state": "closed",
                        },
                        "depends_on": ["OT1"],
                    },
                ],
            },
                {
                    "thought": "The revised outline keeps the semantically inferable LG task but reassigns the unreachable pickup away from xarm6.",
                    "addressed_validation_findings": [
                        _outline_validation_ref(
                            task_id="OT2",
                            resource_jid="xarm6@localhost",
                            pose_source="resource_feasibility",
                            failed_axes=["workspace_unreachable"],
                        ),
                    ],
                    "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
                },
            ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_outline"
    assert multi_turn_session.get("current_phase") == "outline"
    turns = list(multi_turn_session.get("turns") or [])
    assert [turn.get("phase") for turn in turns] == [
        "grounding",
        "grounding",
        "outline",
        "outline",
    ]
    assert turns[2].get("decision") == "need_revision"
    violations = [str(item) for item in (turns[2].get("validation_violations") or [])]
    assert any("pose outside workspace" in item for item in violations)
    findings = list(turns[2].get("outline_validation_findings") or [])
    assert not any(
        "ambiguous_part_reference" in str(dict(item).get("failed_axes") or [])
        or "unbound_part_reference" in str(dict(item).get("failed_axes") or [])
        for item in findings
        if isinstance(item, dict)
    )
    fourth_prompt_text = str(turns[3].get("prompt_text") or "")
    assert "\"ambiguous_part_reference\"" not in fourth_prompt_text
    assert "\"unbound_part_reference\"" not in fourth_prompt_text
    assert "Repair Contract" in fourth_prompt_text


def test_case3_bridge_multi_turn_outline_validation_accepts_resource_only_task_with_stray_part_tag() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This outline incorrectly tags the robot-only move-home recovery step as an LG manipulation task.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": [
                    {
                        "outline_id": "OT1",
                        "resource_jid": "xarm6@localhost",
                        "description": "Move xarm6 from failed to idle by returning it to home.",
                        "rationale": "The failed robot must clear its blocked posture before any downstream recovery can proceed.",
                        "part_name": "LG",
                        "action_target": {
                            "named_pose": "home",
                        },
                        "expected_start_state": {
                            "current_state": "failed",
                            "held_part": None,
                            "gripper_state": "open",
                            "position": {
                                "x": 0.1,
                                "y": 0.08,
                                "z": 1.1994999760206477,
                            },
                        },
                        "expected_end_state": {
                            "current_state": "idle",
                            "held_part": None,
                            "gripper_state": "open",
                        },
                        "depends_on": [],
                    }
                ],
            },
            ]
        )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_outline"
    turns = list(multi_turn_session.get("turns") or [])
    invalid_turn = dict(turns[2] or {})
    assert invalid_turn.get("decision") == "outline_ready"
    assert any(
        dict(row).get("outline_id") == "OT1" and dict(row).get("task_type") == "resource_only"
        for row in (invalid_turn.get("outline_task_types") or [])
        if isinstance(row, dict)
    )
    assert not (invalid_turn.get("validation_violations") or [])
    assert dict(invalid_turn.get("outline_validation") or {}).get("status") == "passed"


def test_case3_bridge_multi_turn_outline_validation_rejects_held_part_conflict() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    invalid_outline_tasks = [
        deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS[0]),
        {
            "outline_id": "OT2",
            "resource_jid": "ur5e@localhost",
            "description": "Use ur5e to pick LG immediately while still holding MCP.",
            "rationale": "This intentionally exercises held-part conflict validation.",
            "part_name": "LG",
            "action_target": {
                "source_location": "observed_pose",
                "requirement_id": "REQ_2",
            },
            "expected_start_state": {
                "current_state": "picked",
                "gripper_state": "closed",
                "held_part": "MCP",
                "part_name": "LG",
            },
            "expected_end_state": {
                "held_part": "LG",
                "gripper_state": "closed",
                "current_state": "picked",
            },
            "depends_on": ["outline_clear_xarm6"],
        },
    ]

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This outline incorrectly asks ur5e to manipulate LG while it is still holding MCP.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": invalid_outline_tasks,
            },
            {
                "thought": "The revised outline frees ur5e before assigning it LG recovery work.",
                "addressed_validation_findings": [
                    {
                        "task_id": "OT2",
                        "pose_source": "resource_feasibility",
                        "failed_axes": ["holder_conflict"],
                        "resource_jid": "ur5e@localhost",
                    }
                ],
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_outline"
    turns = list(multi_turn_session.get("turns") or [])
    invalid_turn = dict(turns[2] or {})
    assert invalid_turn.get("decision") == "need_revision"
    violations = [str(item) for item in (invalid_turn.get("validation_violations") or [])]
    assert any("already holds 'MCP'" in item for item in violations)
    findings = list(invalid_turn.get("outline_validation_findings") or [])
    assert any(
        dict(item).get("task_id") == "OT2"
        and "holder_conflict" in str(dict(item).get("failed_axes") or [])
        for item in findings
        if isinstance(item, dict)
    )
    fourth_prompt_text = str(turns[3].get("prompt_text") or "")
    assert "\"holder_conflict\"" in fourth_prompt_text
    assert "Repair Contract" in fourth_prompt_text


def test_case3_bridge_grounding_compiler_rejects_target_only_part_move_without_current_source() -> None:
    _, _, _, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    _, resources_by_jid, parts_by_name = _outline_bridge_validation_context(
        prepared_bridge_request
    )
    lg_row = dict(parts_by_name.get("LG") or {})
    lg_row["observed_pose"] = None
    lg_row["current_location"] = None
    lg_row["current_holder_resource_jid"] = None
    parts_by_name["LG"] = lg_row

    result = compile_grounded_outline_task(
        {
            "outline_id": "OT_missing_source",
            "resource_jid": "ur5e@localhost",
            "description": "Place LG on the board without first binding where it is now.",
            "rationale": "This intentionally omits the concrete current source reference.",
            "part_name": "LG",
            "action_target": {
                "target_location": "assembly_board-v1",
                "requirement_id": "REQ_2",
            },
            "expected_end_state": {
                "part_name": "LG",
                "location": "assembly_board-v1",
            },
            "depends_on": [],
        },
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )

    assert result.get("status") == "task_not_projectable"
    finding = dict(result.get("finding") or {})
    assert finding.get("constraint_code") == "task_not_projectable"
    assert "source reference" in str(finding.get("reason") or "").lower()


def test_case3_bridge_multi_turn_outline_validation_rejects_target_only_lg_move_while_ur5e_holds_mcp() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    invalid_outline_tasks = [
        deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS[0]),
        {
            "outline_id": "recovery_03",
            "resource_jid": "ur5e@localhost",
            "description": "Relocate LG to assembly_board-v1 directly.",
            "rationale": "This incorrectly assumes ur5e can move LG without first releasing MCP.",
            "part_name": "LG",
            "action_target": {
                "target_location": "assembly_board-v1",
                "requirement_id": "REQ_2",
            },
            "expected_start_state": {
                "part_location": "prusa-mk4-1",
            },
            "expected_end_state": {
                "part_location": "assembly_board-v1",
            },
            "depends_on": ["outline_clear_xarm6"],
        },
    ]

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This outline incorrectly treats LG relocation as a direct placement even though ur5e is still holding MCP.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": invalid_outline_tasks,
            },
            {
                "thought": "The revised outline frees ur5e before assigning it LG recovery work.",
                "addressed_validation_findings": [
                    {
                        "task_id": "recovery_03",
                        "pose_source": "resource_feasibility",
                        "failed_axes": ["holder_conflict"],
                        "resource_jid": "ur5e@localhost",
                    }
                ],
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    turns = list(multi_turn_session.get("turns") or [])
    invalid_turn = dict(turns[2] or {})
    assert invalid_turn.get("decision") == "need_revision"
    violations = [str(item) for item in (invalid_turn.get("validation_violations") or [])]
    assert any("already holds 'MCP'" in item for item in violations)
    findings = list(invalid_turn.get("outline_validation_findings") or [])
    assert any(
        dict(item).get("task_id") == "recovery_03"
        and "holder_conflict" in str(dict(item).get("failed_axes") or [])
        for item in findings
        if isinstance(item, dict)
    )


def test_case3_bridge_multi_turn_outline_validation_rejects_unsupported_resource_target_state() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    invalid_outline_tasks = [
        {
            "outline_id": "recovery_01",
            "resource_jid": "xarm6@localhost",
            "description": "Perform self-diagnosis to transition from failed to safe.",
            "rationale": "This intentionally uses an unsupported abstract resource target.",
            "expected_start_state": {
                "current_state": "failed",
            },
            "expected_end_state": {
                "current_state": "safe",
            },
            "depends_on": [],
        }
    ]

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This outline uses an invented resource-only target state without a grounded recovery target.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": invalid_outline_tasks,
            },
            {
                "thought": "The revised outline uses a concrete home pose instead of an invented state target.",
                "addressed_validation_findings": [
                    {
                        "task_id": "recovery_01",
                        "pose_source": "resource_feasibility",
                        "failed_axes": ["unsupported_resource_target"],
                        "resource_jid": "xarm6@localhost",
                    }
                ],
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    turns = list(multi_turn_session.get("turns") or [])
    invalid_turn = dict(turns[2] or {})
    assert invalid_turn.get("decision") == "need_revision"
    findings = list(invalid_turn.get("outline_validation_findings") or [])
    assert any(
        dict(item).get("task_id") == "recovery_01"
        and "unsupported_resource_target" in str(dict(item).get("failed_axes") or [])
        for item in findings
        if isinstance(item, dict)
    )


def test_case3_bridge_state_guarded_pruning_reenables_after_holder_state_changes() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    llm_input, resources_by_jid, parts_by_name = _outline_bridge_validation_context(
        prepared_bridge_request
    )
    task = {
        "outline_id": "OT2",
        "resource_jid": "ur5e@localhost",
        "description": "Use ur5e to pick LG immediately while still holding MCP.",
        "rationale": "This intentionally exercises held-part conflict validation.",
        "part_name": "LG",
        "action_target": {
            "source_location": "observed_pose",
            "requirement_id": "REQ_2",
        },
        "expected_start_state": {
            "current_state": "picked",
            "gripper_state": "closed",
            "held_part": "MCP",
            "part_name": "LG",
        },
        "expected_end_state": {
            "held_part": "LG",
            "gripper_state": "closed",
            "current_state": "picked",
        },
        "depends_on": ["outline_clear_xarm6"],
    }

    findings = _build_outline_validation_findings([task], llm_input, planner=planner)
    pruned_actions = _build_pruned_actions(
        existing_pruned_actions=[],
        outline_tasks=[task],
        validation_findings=findings,
        llm_input=llm_input,
    )

    assert any(
        "MCP" in str(dict(row).get("reason") or "")
        and str(dict(row).get("reason") or "").strip()
        for row in pruned_actions
    )

    released_llm_input = deepcopy(llm_input)
    for row in (dict(released_llm_input.get("observed_runtime_state") or {}).get("resources") or []):
        if not isinstance(row, dict):
            continue
        if str(row.get("resource_jid") or "").strip() != "ur5e@localhost":
            continue
        row["held_part"] = None
        row["gripper_state"] = "open"
        row["current_state"] = "idle"

    refreshed_resources, refreshed_parts = _outline_bridge_validation_context(
        {"llm_input": released_llm_input}
    )[1:]
    refreshed_pruned_actions = _active_pruned_actions_for_state(
        pruned_actions,
        resources_by_jid=refreshed_resources,
        parts_by_name=refreshed_parts,
        llm_input=released_llm_input,
    )
    assert refreshed_pruned_actions == []


def test_case3_bridge_state_guarded_pruning_skips_non_stateful_binding_cases() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    llm_input, _, _ = _outline_bridge_validation_context(prepared_bridge_request)
    task = {
        "outline_id": "OT2",
        "resource_jid": "xarm6@localhost",
        "description": "Pick LG from the grounded misplaced pose after xarm6 recovers.",
        "rationale": "Positioning LG properly is required before downstream work can continue.",
        "action_target": {
            "source_location": "observed_pose",
            "target_location": "assembly_board-v1",
        },
        "expected_start_state": {
            "position": {"x": 0.0, "y": 0.2, "z": 1.035},
            "current_state": "idle",
            "held_part": None,
            "gripper_state": "open",
        },
        "expected_end_state": {
            "current_state": "busy",
            "held_part": "LG",
            "gripper_state": "closed",
        },
        "depends_on": ["OT1"],
    }

    findings = _build_outline_validation_findings([task], llm_input, planner=planner)
    assert not any(
        "ambiguous_part_reference" in str(dict(row).get("failed_axes") or [])
        or "unbound_part_reference" in str(dict(row).get("failed_axes") or [])
        for row in findings
        if isinstance(row, dict)
    )
    pruned_actions = _build_pruned_actions(
        existing_pruned_actions=[],
        outline_tasks=[task],
        validation_findings=findings,
        llm_input=llm_input,
    )
    assert pruned_actions == []


def test_case3_bridge_state_guarded_pruning_repeated_blocked_action_gets_prune_finding() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    llm_input, _, _ = _outline_bridge_validation_context(prepared_bridge_request)
    task = {
        "outline_id": "OT2",
        "resource_jid": "ur5e@localhost",
        "description": "Use ur5e to pick LG immediately while still holding MCP.",
        "rationale": "This intentionally exercises held-part conflict validation.",
        "part_name": "LG",
        "action_target": {
            "source_location": "observed_pose",
            "requirement_id": "REQ_2",
        },
        "expected_start_state": {
            "current_state": "picked",
            "gripper_state": "closed",
            "held_part": "MCP",
            "part_name": "LG",
        },
        "expected_end_state": {
            "held_part": "LG",
            "gripper_state": "closed",
            "current_state": "picked",
        },
        "depends_on": ["outline_clear_xarm6"],
    }

    initial_findings = _build_outline_validation_findings([task], llm_input, planner=planner)
    pruned_actions = _build_pruned_actions(
        existing_pruned_actions=[],
        outline_tasks=[task],
        validation_findings=initial_findings,
        llm_input=llm_input,
    )
    repeated_findings = _build_outline_validation_findings(
        [task],
        llm_input,
        pruned_actions=pruned_actions,
        planner=planner,
    )

    assert any(
        dict(row).get("validation_status") == "state_infeasible"
        and list(dict(row).get("failed_axes") or []) == ["pruned_action"]
        for row in repeated_findings
        if isinstance(row, dict)
    )


def test_case3_bridge_outline_prompt_lists_currently_pruned_actions() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    llm_input, _, _ = _outline_bridge_validation_context(prepared_bridge_request)
    task = {
        "outline_id": "OT2",
        "resource_jid": "ur5e@localhost",
        "description": "Use ur5e to pick LG immediately while still holding MCP.",
        "rationale": "This intentionally exercises held-part conflict validation.",
        "part_name": "LG",
        "action_target": {
            "source_location": "observed_pose",
            "requirement_id": "REQ_2",
        },
        "expected_start_state": {
            "current_state": "picked",
            "gripper_state": "closed",
            "held_part": "MCP",
            "part_name": "LG",
        },
        "expected_end_state": {
            "held_part": "LG",
            "gripper_state": "closed",
            "current_state": "picked",
        },
        "depends_on": ["outline_clear_xarm6"],
    }
    findings = _build_outline_validation_findings([task], llm_input, planner=planner)
    pruned_actions = _build_pruned_actions(
        existing_pruned_actions=[],
        outline_tasks=[task],
        validation_findings=findings,
        llm_input=llm_input,
    )
    session_state = build_multi_turn_session_seed(prepared_bridge_request)
    session_state["current_phase"] = "outline"
    session_state["pruned_actions"] = deepcopy(pruned_actions)
    session_state["outline_validation_findings"] = deepcopy(findings)

    prompt_input = build_multi_turn_phase_prompt_input(
        phase="outline",
        llm_input=llm_input,
        session_state=session_state,
        pruned_actions=pruned_actions,
    )
    prompt_text = render_multi_turn_phase_prompt(prompt_input)

    assert "Repair Contract" in prompt_text
    assert "\"active_pruned_actions\"" in prompt_text
    assert "already holds 'MCP'" in prompt_text


def test_case3_bridge_ap_projection_skips_part_manipulation_for_resource_only_reset() -> None:
    _, _, _, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    llm_input, resources_by_jid, parts_by_name = _outline_bridge_validation_context(
        prepared_bridge_request
    )
    task = deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS[0])
    signature = _infer_outline_macro_signature(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )
    projected_resources = deepcopy(resources_by_jid)
    projected_parts = deepcopy(parts_by_name)
    _apply_outline_task_effects(
        task,
        resources_by_jid=projected_resources,
        parts_by_name=projected_parts,
        task_type=str(signature.get("task_kind") or ""),
    )
    projection = project_outline_macro_bridge_aps(
        task=task,
        signature=signature,
        pre_resources=resources_by_jid,
        pre_parts=parts_by_name,
        projected_resources=projected_resources,
        projected_parts=projected_parts,
        llm_input=llm_input,
    )
    assert projection.get("candidate_aps") == []
    assert "ap003" not in (projection.get("candidate_aps") or [])
    assert "ap004" not in (projection.get("candidate_aps") or [])


def test_case3_bridge_generic_cca_safety_violation_for_safe1_unload_to_board() -> None:
    _, _, _, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    llm_input, resources_by_jid, parts_by_name = _outline_bridge_validation_context(
        prepared_bridge_request
    )
    condition_ids = _case3_continuation_condition_ids(prepared_bridge_request)
    task = {
        "outline_id": "OT_SAFE1",
        "resource_jid": "ur5e@localhost",
        "description": "Unload MCP to the protected assembly board destination.",
        "part_name": "MCP",
        "action_target": {
            "target_location": "assembly_board-v1",
            "requirement_id": "REQ_1",
        },
        "expected_start_state": {
            "current_state": "picked",
            "gripper_state": "closed",
            "held_part": "MCP",
            "part_name": "MCP",
        },
        "expected_end_state": {
            "current_state": "idle",
            "gripper_state": "released",
            "held_part": None,
            "part_name": "MCP",
            "location": "assembly_board-v1",
        },
        "closes_condition_ids": [condition_ids["mcp_safe1"]],
    }
    signature = _infer_outline_macro_signature(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )
    projected_resources = deepcopy(resources_by_jid)
    projected_parts = deepcopy(parts_by_name)
    _apply_outline_task_effects(
        task,
        resources_by_jid=projected_resources,
        parts_by_name=projected_parts,
        task_type=str(signature.get("task_kind") or ""),
    )
    result = validate_outline_macro_bridge_safety(
        task=task,
        signature=signature,
        pre_resources=resources_by_jid,
        pre_parts=parts_by_name,
        projected_resources=projected_resources,
        projected_parts=projected_parts,
        llm_input=llm_input,
    )
    findings = list(result.get("findings") or [])
    assert result.get("is_safe") is False
    assert any(
        dict(item).get("rule_id") == "SAFE_1"
        and "safety_rule_violation" in str(dict(item).get("failed_axes") or [])
        for item in findings
        if isinstance(item, dict)
    )


def test_case3_bridge_generic_cca_safety_allows_return_mcp_to_printer() -> None:
    _, _, _, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    llm_input, resources_by_jid, parts_by_name = _outline_bridge_validation_context(
        prepared_bridge_request
    )
    task = deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS[1])
    signature = _infer_outline_macro_signature(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )
    projected_resources = deepcopy(resources_by_jid)
    projected_parts = deepcopy(parts_by_name)
    _apply_outline_task_effects(
        task,
        resources_by_jid=projected_resources,
        parts_by_name=projected_parts,
        task_type=str(signature.get("task_kind") or ""),
    )
    result = validate_outline_macro_bridge_safety(
        task=task,
        signature=signature,
        pre_resources=resources_by_jid,
        pre_parts=parts_by_name,
        projected_resources=projected_resources,
        projected_parts=projected_parts,
        llm_input=llm_input,
    )
    assert result.get("is_safe") is True
    assert not (result.get("findings") or [])


def test_case3_bridge_generic_cca_safety_violation_for_safe2_destination_overlap() -> None:
    _, _, _, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    llm_input, resources_by_jid, parts_by_name = _outline_bridge_validation_context(
        prepared_bridge_request
    )
    resources_by_jid["xarm6@localhost"]["current_location"] = "assembly_board-v1"
    task = {
        "outline_id": "OT_SAFE2",
        "resource_jid": "ur5e@localhost",
        "description": "Move ur5e into the assembly board destination zone while xarm6 is still there.",
        "action_target": {
            "target_location": "assembly_board-v1",
        },
        "expected_start_state": {
            "current_state": "picked",
            "held_part": "MCP",
        },
        "expected_end_state": {
            "current_state": "positioned",
            "location": "assembly_board-v1",
            "held_part": "MCP",
        },
    }
    signature = _infer_outline_macro_signature(
        task,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )
    projected_resources = deepcopy(resources_by_jid)
    projected_parts = deepcopy(parts_by_name)
    _apply_outline_task_effects(
        task,
        resources_by_jid=projected_resources,
        parts_by_name=projected_parts,
        task_type=str(signature.get("task_kind") or ""),
    )
    result = validate_outline_macro_bridge_safety(
        task=task,
        signature=signature,
        pre_resources=resources_by_jid,
        pre_parts=parts_by_name,
        projected_resources=projected_resources,
        projected_parts=projected_parts,
        llm_input=llm_input,
    )
    findings = list(result.get("findings") or [])
    assert result.get("is_safe") is False
    assert any(
        dict(item).get("rule_id") == "SAFE_2"
        and "safety_rule_violation" in str(dict(item).get("failed_axes") or [])
        for item in findings
        if isinstance(item, dict)
    )


def test_case3_bridge_multi_turn_invalid_task_does_not_get_condition_credit() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )
    condition_ids = _case3_continuation_condition_ids(prepared_bridge_request)

    invalid_outline_tasks = [
        deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS[0]),
        {
            "outline_id": "OT2",
            "resource_jid": "ur5e@localhost",
            "description": "Move MCP toward assembly_board-v1 and claim that this clears the safety blocker.",
            "rationale": "This intentionally mirrors the bad turn-3 shape where the blocked suffix interaction is reused and incorrectly credited with clearing SAFE_1.",
            "part_name": "MCP",
            "action_target": {
                "target_location": "assembly_board-v1",
                "source_location": "ur5e@localhost_gripper",
                "requirement_id": "REQ_1",
            },
            "expected_start_state": {
                "current_state": "picked",
                "gripper_state": "closed",
                "held_part": "MCP",
                "part_name": "MCP",
                "current_location": "prusa-mk4-2",
            },
            "expected_end_state": {
                "current_state": "idle",
                "gripper_state": "open",
                "held_part": None,
                "location": "assembly_board-v1",
                "part_name": "MCP",
            },
            "depends_on": ["outline_clear_xarm6"],
            "closes_condition_ids": [condition_ids["mcp_safe1"]],
        },
    ]

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This outline incorrectly assumes the still-blocked MCP suffix interaction can clear its own safety blocker.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": invalid_outline_tasks,
            },
                {
                    "thought": "The revised outline frees ur5e and restores LG before resuming the blocked MCP suffix interaction.",
                    "addressed_validation_findings": [
                        _outline_validation_ref(
                            task_id="OT2",
                            resource_jid="ur5e@localhost",
                            pose_source="task_contract",
                            failed_axes=["blocker_open"],
                        ),
                        _outline_validation_ref(
                            task_id="OT2",
                            resource_jid="ur5e@localhost",
                            pose_source="bridge_safety_rule",
                            failed_axes=["safety_rule_violation"],
                    ),
                ],
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_outline"
    turns = list(multi_turn_session.get("turns") or [])
    invalid_turn = dict(turns[2] or {})
    assert invalid_turn.get("decision") == "need_revision"
    violations = [str(item) for item in (invalid_turn.get("validation_violations") or [])]
    assert any("safety rule 'SAFE_1'" in item for item in violations)
    findings = list(invalid_turn.get("outline_validation_findings") or [])
    assert any(
        dict(item).get("task_id") == "OT2"
        and "safety_rule_violation" in str(dict(item).get("failed_axes") or [])
        and str(dict(item).get("rule_id") or "") == "SAFE_1"
        for item in findings
        if isinstance(item, dict)
    )
    fourth_prompt_text = str(turns[3].get("prompt_text") or "")
    assert "\"safety_rule_violation\"" in fourth_prompt_text


def test_case3_bridge_multi_turn_outline_validation_rejects_illegal_holder_swap() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    invalid_outline_tasks = [
        deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS[0]),
        {
            "outline_id": "OT2",
            "resource_jid": "ur5e@localhost",
            "description": "Swap directly from MCP recovery to LG recovery in one macro-step.",
            "rationale": "This intentionally exercises illegal holder swap validation.",
            "action_target": {
                "source_location": "observed_pose",
                "requirement_id": "REQ_2",
            },
            "expected_start_state": {
                "current_state": "picked",
                "gripper_state": "closed",
                "held_part": "MCP",
            },
            "expected_end_state": {
                "current_state": "picked",
                "gripper_state": "closed",
                "held_part": "LG",
            },
            "depends_on": ["outline_clear_xarm6"],
        },
    ]

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This outline incorrectly swaps ur5e directly from MCP to LG.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": invalid_outline_tasks,
            },
            {
                "thought": "The revised outline releases MCP before assigning LG to ur5e.",
                "addressed_validation_findings": deepcopy(
                    MOCK_ILLEGAL_HOLDER_SWAP_REQUIRED_FINDING_REFS
                ),
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    turns = list(multi_turn_session.get("turns") or [])
    invalid_turn = dict(turns[2] or {})
    assert invalid_turn.get("decision") == "need_revision"
    violations = [str(item) for item in (invalid_turn.get("validation_violations") or [])]
    assert any("already holds 'MCP'" in item for item in violations)
    findings = list(invalid_turn.get("outline_validation_findings") or [])
    assert any(
        dict(item).get("task_id") == "OT2"
        and "holder_conflict" in str(dict(item).get("failed_axes") or [])
        for item in findings
        if isinstance(item, dict)
    )
    fourth_prompt_text = str(turns[3].get("prompt_text") or "")
    assert "\"holder_conflict\"" in fourth_prompt_text


def test_case3_bridge_multi_turn_outline_validation_rejects_unknown_named_pose() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This outline names a robot pose that does not exist.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": [
                    {
                        "outline_id": "OT1",
                        "resource_jid": "xarm6@localhost",
                        "description": "Move xarm6 to a nonexistent named pose before recovery.",
                        "rationale": "This intentionally exercises named-pose validation.",
                        "action_target": {
                            "named_pose": "service_bay",
                        },
                        "expected_start_state": {
                            "current_state": "failed",
                            "held_part": None,
                            "gripper_state": "open",
                        },
                        "expected_end_state": {
                            "current_state": "idle",
                            "held_part": None,
                            "gripper_state": "open",
                        },
                        "depends_on": [],
                    }
                ],
            },
            {
                "thought": "The revised outline uses an available named pose.",
                "addressed_validation_findings": deepcopy(
                    MOCK_NAMED_POSE_NOT_AVAILABLE_REQUIRED_FINDING_REFS
                ),
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    turns = list(multi_turn_session.get("turns") or [])
    invalid_turn = dict(turns[2] or {})
    assert invalid_turn.get("decision") == "need_revision"
    violations = [str(item) for item in (invalid_turn.get("validation_violations") or [])]
    assert any("named pose 'service_bay' is not available" in item for item in violations)
    findings = list(invalid_turn.get("outline_validation_findings") or [])
    assert any(
        dict(item).get("task_id") == "OT1"
        and "named_pose_unavailable" in str(dict(item).get("failed_axes") or [])
        for item in findings
        if isinstance(item, dict)
    )
    fourth_prompt_text = str(turns[3].get("prompt_text") or "")
    assert "\"named_pose_unavailable\"" in fourth_prompt_text


def test_case3_bridge_multi_turn_outline_validation_rejects_missing_post_task_anchor() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    invalid_outline_tasks = [
        deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS[0]),
        deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS[1]),
        {
            "outline_id": "OT2",
            "resource_jid": "ur5e@localhost",
            "description": "Reposition LG somewhere more convenient for later recovery.",
            "rationale": "This intentionally leaves the post-task LG holder/location unspecified.",
            "part_name": "LG",
            "expected_start_state": {
                "part_name": "LG",
                "current_state": "misplaced",
            },
            "expected_end_state": {
                "held_part": None,
                "gripper_state": "open",
            },
            "depends_on": ["outline_return_mcp_to_printer"],
        },
    ]

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This outline tries to move LG without making the resulting LG state explicit.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": invalid_outline_tasks,
            },
            {
                "thought": "The revised outline makes LG's post-task holder and destination explicit.",
                "addressed_validation_findings": deepcopy(
                    MOCK_MISSING_POST_TASK_ANCHOR_REQUIRED_FINDING_REFS
                ),
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_outline"
    turns = list(multi_turn_session.get("turns") or [])
    invalid_turn = dict(turns[2] or {})
    assert invalid_turn.get("decision") == "need_revision"
    violations = [str(item) for item in (invalid_turn.get("validation_violations") or [])]
    assert any("does not specify enough target or effect information" in item for item in violations)
    findings = list(invalid_turn.get("outline_validation_findings") or [])
    assert any(
        dict(item).get("task_id") == "OT2"
        and "task_not_projectable" in str(dict(item).get("failed_axes") or [])
        for item in findings
        if isinstance(item, dict)
    )
    fourth_prompt_text = str(turns[3].get("prompt_text") or "")
    assert "\"task_not_projectable\"" in fourth_prompt_text


def test_case3_bridge_multi_turn_outline_validation_rejects_continuation_before_prerequisites() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    invalid_outline_tasks = [
        {
            "outline_id": "resume_mcp",
            "resource_jid": "ur5e@localhost",
            "description": "Resume MCP assembly immediately.",
            "rationale": "This incorrectly attempts to continue the blocked nominal suffix before recovery prerequisites clear.",
            "part_name": "MCP",
            "action_target": {
                "target_location": "assembly_board-v1",
                "requirement_id": "REQ_1",
            },
            "expected_start_state": {
                "current_state": "idle",
                "held_part": None,
                "part_name": "MCP",
            },
            "expected_end_state": {
                "current_state": "idle",
                "held_part": None,
                "part_name": "MCP",
                "location": "assembly_board-v1",
            },
            "depends_on": [],
        },
        deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS[0]),
        deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS[1]),
        deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS[2]),
        deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS[3]),
    ]

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This outline incorrectly resumes the blocked MCP continuation before the factual blockers are cleared.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": invalid_outline_tasks,
            },
            {
                "thought": "The revised outline restores the factual prerequisites before resuming MCP.",
                "addressed_validation_findings": deepcopy(
                    MOCK_CONTINUATION_PREREQUISITE_REQUIRED_FINDING_REFS
                ),
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_outline"
    turns = list(multi_turn_session.get("turns") or [])
    invalid_turn = dict(turns[2] or {})
    assert invalid_turn.get("decision") == "need_revision"
    assert any(
        dict(row).get("outline_id") == "resume_mcp"
        and dict(row).get("task_type") == "continuation_resume"
        for row in (invalid_turn.get("outline_task_types") or [])
        if isinstance(row, dict)
    )
    violations = [str(item) for item in (invalid_turn.get("validation_violations") or [])]
    assert any("missing prerequisite dependencies" in item for item in violations)
    assert any("appears before prerequisite tasks" in item for item in violations)
    assert any("continuation blockers are still uncleared" in item for item in violations)
    findings = list(invalid_turn.get("outline_validation_findings") or [])
    assert any(
        dict(item).get("task_id") == "resume_mcp"
        and "dependency_unsatisfied" in str(dict(item).get("failed_axes") or [])
        for item in findings
        if isinstance(item, dict)
    )
    assert any(
        dict(item).get("task_id") == "resume_mcp"
        and "order_violation" in str(dict(item).get("failed_axes") or [])
        for item in findings
        if isinstance(item, dict)
    )
    assert any(
        dict(item).get("task_id") == "resume_mcp"
        and "blocker_open" in str(dict(item).get("failed_axes") or [])
        for item in findings
        if isinstance(item, dict)
    )
    fourth_prompt_text = str(turns[3].get("prompt_text") or "")
    assert "\"dependency_unsatisfied\"" in fourth_prompt_text
    assert "\"order_violation\"" in fourth_prompt_text
    assert "\"blocker_open\"" in fourth_prompt_text


def test_case3_bridge_multi_turn_outline_validation_uses_predecessor_symbolic_part_state() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    symbolic_rollout_outline = [
        deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS[0]),
        deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS[1]),
        {
            "outline_id": "outline_reposition_lg_for_xarm6",
            "resource_jid": "ur5e@localhost",
            "description": "Move LG from the observed slip pose to a reachable recovery pose for xarm6.",
            "rationale": "This task makes LG reachable to xarm6 without adding prompt-side hints.",
            "part_name": "LG",
            "expected_start_state": {
                "part_name": "LG",
                "current_state": "misplaced",
                "position": {"x": 0.0, "y": 0.2, "z": 1.035},
                "held_part": None,
                "gripper_state": "open",
            },
            "expected_end_state": {
                "part_name": "LG",
                "current_state": "placed",
                "location": "recovery_buffer",
                "position": {"x": 0.0, "y": 0.05, "z": 1.035},
                "held_part": None,
                "gripper_state": "open",
            },
            "depends_on": ["outline_return_mcp_to_printer"],
        },
        {
            "outline_id": "outline_pick_lg_with_xarm6",
            "resource_jid": "xarm6@localhost",
            "description": "Pick LG from the explicit recovery pose prepared for xarm6.",
            "rationale": "Once LG has been concretely repositioned, xarm6 can finish the recovery locally.",
            "part_name": "LG",
            "expected_start_state": {
                "current_state": "idle",
                "gripper_state": "open",
                "position": {"named_pose": "home"},
            },
            "expected_end_state": {
                "part_name": "LG",
                "current_state": "picked",
                "held_part": "LG",
                "gripper_state": "closed",
            },
            "depends_on": ["outline_reposition_lg_for_xarm6"],
        },
        {
            "outline_id": "outline_place_lg_with_xarm6",
            "resource_jid": "xarm6@localhost",
            "description": "Place LG at assembly_board-v1 after grasping it from the recovery pose.",
            "rationale": "This clears SAFE_1 using the predecessor-produced symbolic LG pose.",
            "part_name": "LG",
            "action_target": {
                "target_location": "assembly_board-v1",
                "requirement_id": "REQ_2",
            },
            "expected_start_state": {
                "part_name": "LG",
                "held_part": "LG",
                "gripper_state": "closed",
                "current_state": "picked",
            },
            "expected_end_state": {
                "part_name": "LG",
                "current_state": "assembled",
                "held_part": None,
                "gripper_state": "open",
                "location": "assembly_board-v1",
            },
            "depends_on": ["outline_pick_lg_with_xarm6"],
        },
    ]

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This outline first changes LG's concrete state, then has xarm6 act on the new pose.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": symbolic_rollout_outline,
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_outline"
    accepted_outline = list(multi_turn_session.get("accepted_outline") or [])
    assert accepted_outline == symbolic_rollout_outline
    turns = list(multi_turn_session.get("turns") or [])
    outline_turn = dict(turns[2] or {})
    assert outline_turn.get("decision") == "outline_ready"
    assert not outline_turn.get("validation_violations")
    assert dict(outline_turn.get("outline_validation") or {}).get("status") == "passed"


def test_case3_bridge_multi_turn_outline_validation_does_not_reenter_grounding() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This draft incorrectly keeps LG recovery on xarm6.",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_INVALID_XARM_OUTLINE_TASKS),
            },
            {
                "thought": "The revised outline reassigns LG recovery to ur5e while keeping xarm6 limited to clearing the failed state.",
                "addressed_validation_findings": deepcopy(
                    MOCK_INVALID_XARM_REQUIRED_FINDING_REFS
                ),
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
            },
            deepcopy(MOCK_MULTI_TURN_RESPONSES[3]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[4]),
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal == MOCK_MULTI_TURN_FINAL_PROPOSAL
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    turns = list(multi_turn_session.get("turns") or [])
    assert [turn.get("phase") for turn in turns] == [
        "grounding",
        "grounding",
        "outline",
        "outline",
        "primitive_generation",
        "finalize",
    ]
    assert all(turn.get("phase") != "grounding" for turn in turns[2:])
    assert turns[2].get("decision") == "need_revision"
    assert turns[2].get("validation_violations")


def test_case3_bridge_multi_turn_outline_validation_rejects_abstract_task() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(
            configure_live=False,
            fixture_mode="live",
            bridge_reasoning_mode="multi_turn",
        )
    )
    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    bridge_session["stop_after_phase"] = "outline"
    prepared_bridge_request["bridge_session"] = bridge_session
    prepared_bridge_request["multi_turn_session_seed"] = build_multi_turn_session_seed(
        prepared_bridge_request
    )

    scripted_responses = iter(
        [
            deepcopy(MOCK_MULTI_TURN_RESPONSES[0]),
            deepcopy(MOCK_MULTI_TURN_RESPONSES[1]),
            {
                "thought": "This outline includes an abstract diagnosis step instead of a grounded recovery task.",
                "decision": "need_grounding",
                "addressed_validation_findings": deepcopy(
                    MOCK_EMPTY_ADDRESSED_VALIDATION_FINDINGS
                ),
                "outline_tasks": [
                    {
                        "outline_id": "OT1",
                        "resource_jid": "xarm6@localhost",
                        "description": "Move xarm6 out of the failed posture.",
                        "rationale": "The failed robot must clear its blocked posture first.",
                        "action_target": {
                            "named_pose": "home",
                        },
                        "expected_start_state": {
                            "current_state": "failed",
                            "held_part": None,
                            "position": {
                                "x": 0.1,
                                "y": 0.08,
                                "z": 1.1994999760206477,
                            },
                            "gripper_state": "open",
                        },
                        "expected_end_state": {
                            "current_state": "idle",
                            "held_part": None,
                            "gripper_state": "open",
                        },
                        "depends_on": [],
                    },
                    {
                        "outline_id": "OT2",
                        "resource_jid": "ur5e@localhost",
                        "description": "Diagnose the failure and decide the next recovery step.",
                        "rationale": "This intentionally exercises abstract-task validation.",
                        "expected_start_state": {},
                        "expected_end_state": {},
                        "depends_on": ["OT1"],
                    },
                ],
            },
            {
                "thought": "The revised outline replaces the abstract diagnosis row with grounded recovery tasks without reopening grounding.",
                "addressed_validation_findings": deepcopy(
                    MOCK_ABSTRACT_TASK_REQUIRED_FINDING_REFS
                ),
                "outline_tasks": deepcopy(MOCK_MULTI_TURN_OUTLINE_TASKS),
            },
        ]
    )

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        return deepcopy(next(scripted_responses))

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        proposal = asyncio.run(planner.execute_prepared_bridge_request(prepared_bridge_request))

    assert proposal is None
    multi_turn_session = (planner.get_last_bridge_debug() or {}).get("multi_turn_session") or {}
    assert multi_turn_session.get("status") == "paused_after_outline"
    assert multi_turn_session.get("current_phase") == "outline"
    turns = list(multi_turn_session.get("turns") or [])
    assert [turn.get("phase") for turn in turns] == [
        "grounding",
        "grounding",
        "outline",
        "outline",
    ]
    invalid_turn = dict(turns[2] or {})
    assert invalid_turn.get("decision") == "need_revision"
    assert dict(invalid_turn.get("raw_response") or {}).get("decision") == "need_grounding"
    assert invalid_turn.get("decision_compatibility") is None
    invalid_findings = list(invalid_turn.get("outline_validation_findings") or [])
    assert any(
        "task_not_projectable" in str(dict(item).get("failed_axes") or [])
        for item in invalid_findings
    )
    fourth_prompt_text = str(turns[3].get("prompt_text") or "")
    assert "Repair Contract" in fourth_prompt_text
    assert "Grounded Feasibility Facts" not in fourth_prompt_text
    assert "Recovery Gap State" not in fourth_prompt_text
    assert "\"task_not_projectable\"" in fourth_prompt_text
    assert "\"outline_id\": \"OT2\"" not in fourth_prompt_text


def test_case3_bridge_multi_turn_debug_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.modules[__name__], "DEBUG_DIR", tmp_path)
    scripted_responses = iter(deepcopy(MOCK_MULTI_TURN_RESPONSES))

    async def _mock_ask_llm_structured(
        self: FakeProductAgent,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        del prompt, response_format, tools, tool_executor, max_tool_rounds
        response = deepcopy(next(scripted_responses))
        self._turn_index += 1
        self.turn_log.append(
            {
                "turn_index": self._turn_index,
                "prompt": "<mocked-structured>",
                "response": deepcopy(response),
            }
        )
        return response

    with patch.object(FakeProductAgent, "ask_llm_structured", new=_mock_ask_llm_structured):
        result = asyncio.run(
            run_case3_bridge_dryrun(
                write_debug=True,
                bridge_reasoning_mode="multi_turn",
                multi_turn_stop_after_phase="",
            )
        )

    prompt_artifact_path = Path(str(result.get("prompt_artifact_path") or ""))
    latest_prompt_artifact_path = Path(str(result.get("latest_prompt_artifact_path") or ""))
    response_artifact_path = Path(str(result.get("response_artifact_path") or ""))
    latest_response_artifact_path = Path(str(result.get("latest_response_artifact_path") or ""))
    session_artifact_path = Path(str(result.get("session_transcript_artifact_path") or ""))
    latest_session_artifact_path = Path(
        str(result.get("latest_session_transcript_artifact_path") or "")
    )

    assert prompt_artifact_path.exists()
    assert prompt_artifact_path.name.startswith("multi_turn_turn05_finalize_prompt_")
    assert latest_prompt_artifact_path.exists()
    assert latest_prompt_artifact_path.name == "multi_turn_turn05_finalize_prompt_latest.txt"
    assert response_artifact_path.exists()
    assert response_artifact_path.name.startswith("multi_turn_turn05_finalize_response_")
    assert latest_response_artifact_path.exists()
    assert latest_response_artifact_path.name == "multi_turn_turn05_finalize_response_latest.txt"
    assert session_artifact_path.exists()
    assert session_artifact_path.name.startswith("multi_turn_session_prepare_")
    assert latest_session_artifact_path.exists()
    assert latest_session_artifact_path.name.startswith("multi_turn_session_prepare_")
    assert latest_session_artifact_path.name.endswith("_latest.txt")
    turn01_response_latest_path = tmp_path / "multi_turn_turn01_grounding_response_latest.txt"
    turn02_prompt_latest_path = tmp_path / "multi_turn_turn02_grounding_prompt_latest.txt"
    assert turn01_response_latest_path.exists()
    assert turn02_prompt_latest_path.exists()
    turn01_response_text = turn01_response_latest_path.read_text(encoding="utf-8")
    turn02_prompt_text = turn02_prompt_latest_path.read_text(encoding="utf-8")
    assert "\"z\": 1.035" not in turn01_response_text
    assert "\"z\": 1.035" in turn02_prompt_text
    assert "Observation Fulfillment Status" in turn02_prompt_text
    assert "Latest Session Delta" in turn02_prompt_text
    assert "Session Observation Store" not in turn02_prompt_text
    assert "Observation Fact Ledger" not in turn02_prompt_text
    assert "Session Turn History" not in turn02_prompt_text
    assert turn02_prompt_text.find("Observation Fulfillment Status") < turn02_prompt_text.find(
        "Current Resource Facts"
    )
    session_text = latest_session_artifact_path.read_text(encoding="utf-8")
    assert "\"current_phase\": \"finalize\"" in session_text
    assert "\"final_proposal\"" in session_text


# ---------------------------------------------------------------------------
# CLI entry  (F5 / python test/test_case3_bridge_dryrun.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Case 3 LG-slippage bridge dry-run harness"
    )
    parser.add_argument("--model", default=DEFAULT_LIVE_MODEL, help="OpenAI model name")
    parser.add_argument(
        "--reasoning-mode",
        choices=("single_shot", "multi_turn"),
        default="multi_turn",
        help="Bridge reasoning mode to exercise when running the script directly",
    )
    parser.add_argument(
        "--stop-after-phase",
        choices=("grounding", "outline", "primitive_generation", "finalize"),
        default=None,
        help="Optional multi-turn pause point for debugging.",
    )
    parser.add_argument("--no-debug", action="store_true", help="Skip writing debug artifact")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare and log the context trace, then exit before any LLM stage",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Disable the default multi-turn grounding pause and continue through later phases.",
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
        "--show-prompt",
        dest="show_prompt",
        action="store_true",
        help="Print the current mode prompt text in the terminal",
    )
    parser.add_argument(
        "--hide-prompt",
        dest="show_prompt",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--show-single-shot-prompt",
        dest="show_prompt",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--hide-single-shot-prompt",
        dest="show_prompt",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(show_llm_input=False, show_prompt=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.prepare_only:
        result = asyncio.run(
            run_case3_bridge_prepare_trace(
                write_debug=not args.no_debug,
                llm_model=args.model,
                bridge_reasoning_mode=args.reasoning_mode,
            )
        )
        print()
        print(f"Prepare-trace context build completed ({args.reasoning_mode}).")
        _print_prepare_context_summary(result.get("context_summary") or {})
        if args.show_llm_input:
            _print_llm_input(result.get("llm_input") or {})
        if args.show_prompt:
            _print_prompt(
                _extract_latest_prompt_text(result),
                reasoning_mode=str(result.get("reasoning_mode") or args.reasoning_mode),
            )
        _print_debug_artifact_paths(result)
        sys.exit(0)

    effective_stop_after_phase = args.stop_after_phase
    if args.reasoning_mode == "multi_turn" and args.full_run and effective_stop_after_phase is None:
        effective_stop_after_phase = ""

    result = asyncio.run(
        run_case3_bridge_dryrun(
            write_debug=not args.no_debug,
            llm_model=args.model,
            bridge_reasoning_mode=args.reasoning_mode,
            multi_turn_stop_after_phase=effective_stop_after_phase,
        )
    )

    print()
    print(f"Bridge status ({result.get('reasoning_mode') or args.reasoning_mode}):", result.get("status") or "-")
    _print_prepare_context_summary(result.get("context_summary") or {})
    if args.show_prompt:
        _print_prompt(
            _extract_latest_prompt_text(result),
            reasoning_mode=str(result.get("reasoning_mode") or args.reasoning_mode),
        )
    if args.show_llm_input:
        _print_llm_input(result.get("llm_input") or {})
    _print_debug_artifact_paths(result)

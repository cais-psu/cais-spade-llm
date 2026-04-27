"""Dry-run scenario for Case 3 LG slippage -> LLM bridge recovery.

Run directly:
    python test/test_case3_bridge_dryrun.py
    python test/test_case3_bridge_dryrun.py --model gpt-5
    python test/test_case3_bridge_dryrun.py --reasoning-mode multi_turn
    python test/test_case3_bridge_dryrun.py --focus primitive_generation
    python test/test_case3_bridge_dryrun.py --focus recovery_safety
    python test/test_case3_bridge_dryrun.py --show-llm-input
    python test/test_case3_bridge_dryrun.py --show-prompt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
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
from cais_spade_llm.agents.central_controller.central_controller_agent import (
    CentralControllerAgent,
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
from cais_spade_llm.agents.central_controller.recovery_safety_generation import (
    generate_recovery_safety_bundle,
    _validate_grounded_rule_result,
)
from cais_spade_llm.agents.central_controller.online_safety_monitor import (
    OnlineSafetyMonitor,
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


def _resolve_case3_archived_final_output_path() -> Path:
    filename = "multi_turn_turn09_final_output_response_20260423T013259.txt"
    worked_root = (
        ROOT
        / "cais_spade_llm"
        / "agents"
        / "intelligent_product"
        / "replanner"
        / "llm_bridge"
        / "runtime_data"
        / "imported"
        / "worked"
    )
    base_path = (
        worked_root
        / "1"
        / filename
    )
    candidate_paths = [
        base_path.parent / "recovery_final" / base_path.name,
        base_path,
        *sorted(worked_root.glob(f"*/recovery_final/{filename}")),
    ]
    for candidate_path in candidate_paths:
        if candidate_path.is_file():
            return candidate_path
    return base_path


CASE3_ARCHIVED_FINAL_OUTPUT_PATH = _resolve_case3_archived_final_output_path()


def _configure_dryrun_logging() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


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
                parsed, end_idx = decoder.raw_decode(text[start_idx:])
            except json.JSONDecodeError:
                continue
            trailing = text[start_idx + end_idx :].strip()
            if trailing:
                logging.getLogger("case3_bridge_dryrun").warning(
                    "[DryRun] Ignoring trailing text after structured JSON payload"
                )
            return parsed
        raise exc

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


def _build_dryrun_recovery_safety_generation_payload(
    *,
    prepared_bridge_request: dict[str, Any],
    multi_turn_session: dict[str, Any] | None = None,
    recovery_safety_scope_id: str = "dryrun_recovery_scope",
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
            for task in (resource_entry.get("pending_tasks") or []):
                if not isinstance(task, dict):
                    continue
                task_id = str(task.get("id") or "").strip()
                if not task_id:
                    continue
                pending_task_rows_by_id[task_id] = deepcopy(task)
                requirement_id = str(
                    task.get("requirement_id")
                    or task_requirement_map.get(task_id)
                    or ""
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
            for raw_task in (requirement_task_index.get(requirement_id) or []):
                if not isinstance(raw_task, dict):
                    continue
                task_id = str(raw_task.get("task_id") or raw_task.get("id") or "").strip()
                if not task_id or task_id in seen_task_ids:
                    continue
                seen_task_ids.add(task_id)
                enriched = dict(pending_task_rows_by_id.get(task_id) or {})
                pending_row = dict(pending_by_id.get(task_id) or {})
                params = dict(
                    enriched.get("params")
                    or raw_task.get("params")
                    or {}
                )
                product_jid = str(params.get("product_jid") or "").strip()
                destination_location = str(params.get("destination_location") or "").strip()
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
                        "product_jid": product_jid,
                        "destination_location": destination_location,
                        "blocked_by_condition_ids": [
                            str(token).strip()
                            for token in (pending_row.get("blocked_by_condition_ids") or [])
                            if str(token).strip()
                        ],
                        "expected_start_state": deepcopy(
                            pending_row.get("expected_start_state") or {}
                        ),
                        "expected_end_state": deepcopy(
                            pending_row.get("expected_end_state") or {}
                        ),
                        "projected_outline_state": deepcopy(
                            pending_row.get("projected_outline_state") or {}
                        ),
                    }
                )
        return candidate_rows

    session_state = _resolve_dryrun_multi_turn_session_state(
        prepared_bridge_request,
        multi_turn_session=multi_turn_session,
        require_outline_trace=True,
    )
    accepted_outline_prefix = _dryrun_outline_trace_from_session_state(session_state)
    if not accepted_outline_prefix:
        return {}
    projected_outline_state = (
        deepcopy(accepted_outline_prefix[-1].get("projected_outline_state") or {})
        if accepted_outline_prefix
        else deepcopy(session_state.get("projected_outline_state") or {})
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
    debug_root = _resolve_debug_root()
    recovery_safety_dir = debug_root / "recovery_safety"
    return {
        "product_jid": "assembly_board-v1@localhost",
        "recovery_safety_scope_id": str(recovery_safety_scope_id or "").strip(),
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
    }


def _dryrun_outline_trace_from_session_state(
    session_state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
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
    turns = [
        dict(row)
        for row in (session_state.get("turns") or [])
        if isinstance(row, dict)
    ]
    for turn in reversed(turns):
        if str(turn.get("phase") or "").strip().lower() != "final_output":
            continue
        if str(turn.get("final_output_stage") or "").strip() != "outline_ready":
            continue
        response_artifact_path = str(turn.get("response_artifact_path") or "").strip()
        if not response_artifact_path:
            continue
        try:
            artifact_payload = json.loads(
                Path(response_artifact_path).read_text(encoding="utf-8")
            )
        except Exception:
            continue
        trace = [
            deepcopy(row)
            for row in (artifact_payload.get("transition_trace") or [])
            if isinstance(row, dict)
        ]
        if trace:
            return trace
    return []


def _resolve_dryrun_multi_turn_session_state(
    prepared_bridge_request: dict[str, Any],
    *,
    multi_turn_session: dict[str, Any] | None = None,
    planner_bridge_debug: dict[str, Any] | None = None,
    require_outline_trace: bool = False,
) -> dict[str, Any]:
    session_candidates = [
        multi_turn_session,
        dict(planner_bridge_debug or {}).get("multi_turn_session"),
        dict(prepared_bridge_request.get("bridge_debug") or {}).get("multi_turn_session"),
        prepared_bridge_request.get("multi_turn_session_state"),
        prepared_bridge_request.get("multi_turn_session_seed"),
    ]
    session_state: dict[str, Any] = {}
    for candidate in session_candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_state = dict(candidate or {})
        if require_outline_trace and _dryrun_outline_trace_from_session_state(candidate_state):
            return candidate_state
        if not session_state:
            session_state = candidate_state
    if require_outline_trace:
        artifact_directory = _payload_artifact_directory(
            {"prepared_bridge_request": prepared_bridge_request}
        )
        recovery_outline_dir = artifact_directory / "recovery_outline"
        if recovery_outline_dir.is_dir():
            for artifact_path in sorted(
                recovery_outline_dir.glob("multi_turn_turn*_final_output_response_*.txt"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            ):
                try:
                    artifact_payload = json.loads(
                        artifact_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    continue
                if (
                    str(artifact_payload.get("final_output_stage") or "").strip()
                    != "outline_ready"
                ):
                    continue
                transition_trace = [
                    deepcopy(row)
                    for row in (artifact_payload.get("transition_trace") or [])
                    if isinstance(row, dict)
                ]
                if not transition_trace:
                    continue
                return {
                    "status": str(artifact_payload.get("status") or "").strip()
                    or "ready_for_primitive_generation",
                    "current_phase": str(
                        artifact_payload.get("current_phase") or ""
                    ).strip()
                    or "primitive_generation",
                    "accepted_outline_prefix": deepcopy(transition_trace),
                    "transition_trace": deepcopy(transition_trace),
                    "final_output": deepcopy(artifact_payload),
                    "turns": [
                        {
                            "phase": "final_output",
                            "final_output_stage": "outline_ready",
                            "response_artifact_path": str(artifact_path),
                        }
                    ],
                }
    return session_state


async def _generate_dryrun_recovery_safety_artifacts(
    *,
    product_agent: FakeProductAgent,
    prepared_bridge_request: dict[str, Any],
    multi_turn_session: dict[str, Any] | None = None,
    recovery_safety_scope_id: str = "dryrun_recovery_scope",
) -> dict[str, Any]:
    payload = _build_dryrun_recovery_safety_generation_payload(
        prepared_bridge_request=prepared_bridge_request,
        multi_turn_session=multi_turn_session,
        recovery_safety_scope_id=recovery_safety_scope_id,
    )
    if not payload:
        return {}
    return await generate_recovery_safety_bundle(product_agent, payload)


def _fake_case3_recovery_safety_grounding_response(
    payload: dict[str, Any],
) -> dict[str, Any]:
    accepted_outline_prefix = [
        deepcopy(row)
        for row in (payload.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]
    accepted_by_outline_id = {
        str(row.get("outline_id") or "").strip(): dict(row)
        for row in accepted_outline_prefix
        if isinstance(row, dict) and str(row.get("outline_id") or "").strip()
    }
    recovery_seq3 = dict(accepted_by_outline_id.get("RECOVERY_SEQ3") or {})
    recovery_seq4 = dict(accepted_by_outline_id.get("RECOVERY_SEQ4") or {})
    selector_rules: list[dict[str, Any]] = []
    for rule in (payload.get("loaded_safety_rules") or []):
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id") or rule.get("rule_id") or "").strip()
        if rule_id == "SAFE_1":
            selector_rules.append(
                {
                    "rule_id": rule_id,
                    "status": "selected",
                    "reason": (
                        "RECOVERY_SEQ3 and RECOVERY_SEQ4 are the LG recovery rows in scope."
                    ),
                    "selected_recovery_outline_ids": [
                        str(recovery_seq3.get("outline_id") or "").strip(),
                        str(recovery_seq4.get("outline_id") or "").strip(),
                    ],
                }
            )
            continue
        if rule_id == "SAFE_2":
            selector_rules.append(
                {
                    "rule_id": rule_id,
                    "status": "selected",
                    "reason": (
                        "RECOVERY_SEQ4 is the ur5e recovery move into assembly_board-v1."
                    ),
                    "selected_recovery_outline_ids": [
                        str(recovery_seq4.get("outline_id") or "").strip(),
                    ],
                }
            )
            continue
        selector_rules.append(
            {
                "rule_id": rule_id,
                "status": "not_involved",
                "reason": "",
                "selected_recovery_outline_ids": [],
            }
        )
    return {"rules": selector_rules}


def _dryrun_recovery_safety_status_from_result(
    result: dict[str, Any] | None,
) -> str:
    payload = dict(result or {})
    status = str(payload.get("recovery_safety_status") or "").strip().lower()
    if status:
        return status
    if payload:
        return "ready"
    return "none"


def _dryrun_outline_approval_reached(
    session_state: dict[str, Any] | None,
) -> bool:
    if not isinstance(session_state, dict):
        return False
    outline_trace = _dryrun_outline_trace_from_session_state(session_state)
    if not outline_trace:
        return False
    current_phase = str(session_state.get("current_phase") or "").strip().lower()
    if current_phase == "primitive_generation":
        return True
    final_output = dict(session_state.get("final_output") or {})
    if (
        final_output
        and str(final_output.get("final_output_stage") or "").strip() == "outline_ready"
        and bool(list(final_output.get("transition_trace") or []))
    ):
        return True
    turns = [
        dict(row)
        for row in (session_state.get("turns") or [])
        if isinstance(row, dict)
    ]
    for turn in reversed(turns):
        if str(turn.get("phase") or "").strip().lower() != "final_output":
            continue
        if str(turn.get("final_output_stage") or "").strip() == "outline_ready":
            return True
    return False


def _dryrun_primitive_program_ready_payload(
    prepared_bridge_request: dict[str, Any],
    *,
    multi_turn_session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session_candidates = [
        multi_turn_session,
        dict(prepared_bridge_request.get("bridge_debug") or {}).get("multi_turn_session"),
        prepared_bridge_request.get("multi_turn_session_state"),
    ]
    for candidate in session_candidates:
        if not isinstance(candidate, dict):
            continue
        final_output = dict(candidate.get("final_output") or {})
        if (
            final_output
            and str(final_output.get("final_output_stage") or "").strip()
            == "primitive_program_ready"
            and bool(final_output.get("primitive_program_complete"))
        ):
            return deepcopy(final_output)
    bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
    final_output = dict(bridge_debug.get("final_output") or {})
    if (
        final_output
        and str(final_output.get("final_output_stage") or "").strip()
        == "primitive_program_ready"
        and bool(final_output.get("primitive_program_complete"))
    ):
        return deepcopy(final_output)
    return {}


def _dryrun_primitive_program_ready_source_path(
    prepared_bridge_request: dict[str, Any],
    *,
    multi_turn_session: dict[str, Any] | None = None,
) -> str:
    session_candidates = [
        multi_turn_session,
        dict(prepared_bridge_request.get("bridge_debug") or {}).get("multi_turn_session"),
        prepared_bridge_request.get("multi_turn_session_state"),
    ]
    for candidate in session_candidates:
        if not isinstance(candidate, dict):
            continue
        turns = [
            dict(row)
            for row in (candidate.get("turns") or [])
            if isinstance(row, dict)
        ]
        for turn in reversed(turns):
            if str(turn.get("phase") or "").strip().lower() != "final_output":
                continue
            if str(turn.get("final_output_stage") or "").strip() != "primitive_program_ready":
                continue
            candidate_path = str(turn.get("response_artifact_path") or "").strip()
            if candidate_path:
                return candidate_path
    return ""


def _dryrun_primitive_program_ready_turn_index(
    prepared_bridge_request: dict[str, Any],
    *,
    multi_turn_session: dict[str, Any] | None = None,
) -> int:
    session_candidates = [
        multi_turn_session,
        dict(prepared_bridge_request.get("bridge_debug") or {}).get("multi_turn_session"),
        prepared_bridge_request.get("multi_turn_session_state"),
    ]
    for candidate in session_candidates:
        if not isinstance(candidate, dict):
            continue
        turns = [
            dict(row)
            for row in (candidate.get("turns") or [])
            if isinstance(row, dict)
        ]
        for turn in reversed(turns):
            if str(turn.get("phase") or "").strip().lower() != "final_output":
                continue
            if str(turn.get("final_output_stage") or "").strip() != "primitive_program_ready":
                continue
            return int(turn.get("turn_index") or 0)
    return 0


def _write_dryrun_recovery_final_bundle(
    *,
    prepared_bridge_request: dict[str, Any],
    multi_turn_session: dict[str, Any] | None = None,
    recovery_safety_generation: dict[str, Any] | None = None,
) -> dict[str, str]:
    if not isinstance(recovery_safety_generation, dict) or not bool(
        recovery_safety_generation.get("ok")
    ):
        return {}
    final_output_payload = _dryrun_primitive_program_ready_payload(
        prepared_bridge_request,
        multi_turn_session=multi_turn_session,
    )
    if not final_output_payload:
        return {}
    recovery_safety_logic_json = str(
        recovery_safety_generation.get("recovery_safety_logic_json") or ""
    ).strip()
    if not recovery_safety_logic_json:
        return {}
    recovery_safety_logic_path = Path(recovery_safety_logic_json)
    if not recovery_safety_logic_path.exists():
        return {}

    recovery_final_dir = _resolve_debug_root() / "recovery_final"
    recovery_final_dir.mkdir(parents=True, exist_ok=True)
    source_final_output_path = _dryrun_primitive_program_ready_source_path(
        prepared_bridge_request,
        multi_turn_session=multi_turn_session,
    )
    final_output_turn_index = _dryrun_primitive_program_ready_turn_index(
        prepared_bridge_request,
        multi_turn_session=multi_turn_session,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if source_final_output_path:
        source_path = Path(source_final_output_path)
        target_final_output_path = recovery_final_dir / source_path.name
        if source_path.exists() and source_path.resolve() != target_final_output_path.resolve():
            shutil.copy2(source_path, target_final_output_path)
        elif source_path.exists():
            target_final_output_path = source_path
        else:
            target_final_output_path.write_text(
                json.dumps(final_output_payload, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
    else:
        target_final_output_path = recovery_final_dir / (
            f"multi_turn_turn{max(final_output_turn_index, 1):02d}_"
            f"final_output_response_{timestamp}.txt"
        )
        target_final_output_path.write_text(
            json.dumps(final_output_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    target_logic_path = recovery_final_dir / "cca_safety_logic.json"
    target_logic_tmp_path = recovery_final_dir / "cca_safety_logic.json.tmp"
    shutil.copy2(recovery_safety_logic_path, target_logic_tmp_path)
    target_logic_tmp_path.replace(target_logic_path)
    for stale_dfa_path in sorted(recovery_final_dir.glob("*_dfa.dot")):
        stale_dfa_path.unlink(missing_ok=True)
    for source_dfa_path in sorted(recovery_safety_logic_path.parent.glob("*_dfa.dot")):
        shutil.copy2(source_dfa_path, recovery_final_dir / source_dfa_path.name)

    bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
    bridge_debug["recovery_safety_logic_json"] = str(target_logic_path.resolve())
    bridge_debug["recovery_final_safety_logic_json"] = str(target_logic_path.resolve())
    bridge_debug["recovery_final_dir"] = str(recovery_final_dir)
    bridge_debug["recovery_final_output_path"] = str(target_final_output_path.resolve())
    prepared_bridge_request["bridge_debug"] = bridge_debug
    return {
        "recovery_safety_logic_json": str(target_logic_path.resolve()),
        "recovery_final_dir": str(recovery_final_dir),
        "recovery_final_output_path": str(target_final_output_path.resolve()),
        "recovery_final_safety_logic_json": str(target_logic_path.resolve()),
    }


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case3_archived_transition_trace() -> list[dict[str, Any]]:
    final_output_payload = _load_json(CASE3_ARCHIVED_FINAL_OUTPUT_PATH)
    if not isinstance(final_output_payload, dict):
        raise TypeError("case3 archived final output did not decode to an object")
    for key in ("executable_recovery_trace", "transition_trace"):
        trace = [
            deepcopy(row)
            for row in (final_output_payload.get(key) or [])
            if isinstance(row, dict)
        ]
        if trace:
            return trace
    raise ValueError("case3 archived final output is missing transition_trace")


def _case3_archived_projected_outline_state() -> dict[str, Any]:
    final_output_payload = _load_json(CASE3_ARCHIVED_FINAL_OUTPUT_PATH)
    if not isinstance(final_output_payload, dict):
        raise TypeError("case3 archived final output did not decode to an object")
    accepted_program = [
        deepcopy(row)
        for row in (final_output_payload.get("accepted_primitive_program") or [])
        if isinstance(row, dict)
    ]
    if accepted_program:
        return deepcopy(dict(accepted_program[-1].get("projected_outline_state") or {}))
    return {}


def _case3_archived_outline_ready_session() -> dict[str, Any]:
    transition_trace = _case3_archived_transition_trace()
    return {
        "status": "ready_for_primitive_generation",
        "current_phase": "primitive_generation",
        "accepted_outline_prefix": deepcopy(transition_trace),
        "transition_trace": deepcopy(transition_trace),
        "final_output": {
            "engine": "multi_turn",
            "decision": "final_output_ready",
            "final_output_stage": "outline_ready",
            "status": "ready_for_primitive_generation",
            "current_phase": "primitive_generation",
            "accepted_trace_length": len(transition_trace),
            "transition_trace": deepcopy(transition_trace),
            "primitive_program_complete": False,
        },
        "turns": [
            {
                "turn_index": 6,
                "phase": "final_output",
                "final_output_stage": "outline_ready",
            }
        ],
    }


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
                return _parse_structured_json_text(choice.content or "{}")
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


def test_case3_archived_primitive_context_uses_event_local_start_state_for_seq2() -> None:
    session_state = {
        "accepted_outline_prefix": _case3_archived_transition_trace(),
    }
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


def test_parse_structured_json_text_ignores_trailing_extra_data() -> None:
    parsed = _parse_structured_json_text(
        '{"decision":"accept","items":[1,2]}\n{"debug":"extra"}'
    )

    assert parsed == {"decision": "accept", "items": [1, 2]}


def test_case3_recovery_safety_filter_preserves_nominal_only_cca_violations() -> None:
    plan = {
        "nodes": [
            {
                "type": "task",
                "id": "RECOVERY_BRIDGE_123456",
                "function_name": "execute_recovery_macro",
            },
            {
                "type": "task",
                "id": "RECOVERY_BRIDGE_ABCDEF",
                "function_name": "execute_recovery_macro",
            },
            {"type": "task", "id": "REQ_1_T3", "function_name": "place_approach"},
            {"type": "task", "id": "REQ_4_T3", "function_name": "place_approach"},
        ]
    }
    violations = [
        {
            "violated_rule_id": "SAFE_2",
            "witness_task_ids": ["RECOVERY_BRIDGE_123456", "REQ_1_T3"],
        },
        {
            "violated_rule_id": "SAFE_2",
            "witness_task_ids": ["RECOVERY_BRIDGE_123456", "RECOVERY_BRIDGE_ABCDEF"],
        },
        {
            "violated_rule_id": "SAFE_2",
            "witness_task_ids": ["REQ_1_T3", "REQ_4_T3"],
        },
    ]

    filtered, suppressed = (
        CentralControllerAgent._filter_recovery_safety_validation_violations(
            violations,
            plan,
        )
    )

    assert suppressed == 2
    assert filtered == [violations[2]]


def test_case3_outline_validation_rejects_xarm6_lg_pick_outside_workspace() -> None:
    _, _, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(reasoning_mode="multi_turn")
    )
    session_state = deepcopy(
        prepared_bridge_request.get("multi_turn_session_seed") or {}
    )
    session_state["symbolic_resources"] = {
        "xarm6@localhost": {
            "resource_jid": "xarm6@localhost",
            "resource_state": "idle",
            "current_state": "idle",
            "held_part": None,
            "current_pose": {"x": 0.1, "y": 0.08, "z": 1.1994999760206477},
        },
        "ur5e@localhost": {
            "resource_jid": "ur5e@localhost",
            "resource_state": "picked",
            "current_state": "picked",
            "held_part": "MCP",
            "current_pose": {"x": -0.25, "y": 0.22, "z": 1.18},
        },
    }
    session_state["symbolic_parts"] = {
        "LG": {
            "part_name": "LG",
            "part_state": "misplaced",
            "current_state": "misplaced",
            "part_location": None,
            "current_location": None,
            "part_holder_resource_jid": None,
            "current_holder_resource_jid": None,
            "observed_pose": {"x": 0.0, "y": 0.2, "z": 1.035},
            "goal_location": "assembly_board-v1",
        },
        "MCP": {
            "part_name": "MCP",
            "part_state": "in_gripper",
            "current_state": "in_gripper",
            "part_location": "ur5e@localhost_gripper",
            "current_location": "ur5e@localhost_gripper",
            "part_holder_resource_jid": "ur5e@localhost",
            "current_holder_resource_jid": "ur5e@localhost",
            "goal_location": "assembly_board-v1",
        },
    }

    findings, grounded_action = multi_turn_mode._validate_single_outline_task(
        planner=planner,
        session_state=session_state,
        prepared_bridge_request=prepared_bridge_request,
        task={
            "outline_id": "RECOVERY_SEQ4",
            "event_name": "pick_LG_for_recovery",
            "resource_jid": "xarm6@localhost",
            "part_name": "LG",
            "rationale": "Pick the misplaced LG.",
            "expected_start_state": {
                "resource_state": "idle",
                "held_part": None,
                "part_state": "misplaced",
            },
            "expected_end_state": {
                "resource_state": "picked",
                "held_part": "LG",
                "part_state": "in_gripper",
                "part_location": "xarm6@localhost_gripper",
            },
        },
    )

    assert grounded_action is not None
    assert any(
        str(row.get("constraint_code") or "") == "workspace_unreachable"
        and str(row.get("constraint_family") or "") == "resource_feasibility"
        for row in findings
    )


def test_case3_archived_bridge_selects_first_ready_task_for_serial_dispatch(
    tmp_path: Path,
) -> None:
    _, product_agent, planner, _, recovery = _approve_case3_archived_bridge(tmp_path)

    assert recovery["status"] == "validating"
    ready_node = product_agent._active_bridge_next_ready_task()

    assert isinstance(ready_node, dict)
    assert str(ready_node.get("bridge_outline_id") or "").strip() == "RECOVERY_SEQ1"
    assert str(ready_node.get("resource_jid") or "").strip() == "xarm6@localhost"
    assert str(ready_node.get("status") or "").strip() == "pending"


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


async def _immediate_to_thread(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    return func(*args, **kwargs)


def _configure_runtime_bridge_approval_harness(
    *,
    product_agent: FakeProductAgent,
    planner: ProcessPlanner,
    fixture: dict[str, Any],
    tmp_path: Path,
    bridge_mode: str = "pre_ran",
    validation_policy: str = "no_validation",
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
    product_agent._runtime_bridge_mode = str(bridge_mode or "pre_ran")
    product_agent._runtime_bridge_validation_policy = str(
        validation_policy or "no_validation"
    )
    product_agent._runtime_bridge_archive_path = str(CASE3_ARCHIVED_FINAL_OUTPUT_PATH)
    product_agent._runtime_bridge_archive_label = CASE3_ARCHIVED_FINAL_OUTPUT_PATH.name
    product_agent._orphaned_bridge_task_warning_ids = set()
    product_agent._generated_bridge_gazebo_verification_enabled = False
    product_agent.cca_jid = "cca@localhost"
    product_agent.plan_path = tmp_path / "case3_plan.json"
    product_agent.global_fsa_path = tmp_path / "case3_global_fsa.json"
    product_agent.product_state_path = tmp_path / "case3_product_state.json"
    product_agent.resource_state_path = tmp_path / "case3_resource_state.json"
    product_agent.dispatched_agent_messages = []
    product_agent._build_plan_validation_payload = lambda **kwargs: (
        ProductAgent._build_plan_validation_payload(product_agent, **kwargs)
    )
    product_agent._build_runtime_plan_context = lambda: (
        ProductAgent._build_runtime_plan_context(product_agent)
    )
    product_agent._ensure_plan_result_inbox = lambda: None
    product_agent._dispatch_agent_message_sync = (
        lambda msg, trace_category="": product_agent.dispatched_agent_messages.append(
            {
                "to": str(getattr(msg, "to", "") or "").strip(),
                "metadata": dict(getattr(msg, "metadata", {}) or {}),
                "body": json.loads(str(getattr(msg, "body", "") or "{}")),
                "trace_category": str(trace_category or "").strip(),
            }
        )
    )
    product_agent._run_callable_on_agent_loop_sync = (
        lambda func, timeout_sec=10.0, operation_name="": func()
    )
    product_agent.recovery_controller = ProductRecoveryController(product_agent)
    product_agent.recovery_controller.bind_methods()
    product_agent.runtime_recovery = product_agent._empty_runtime_recovery()
    product_agent._runtime_recovery_context = {}


def _approve_case3_archived_bridge(
    tmp_path: Path,
    *,
    bridge_source: str = "archived_final_output",
    bridge_mode: str = "pre_ran",
    validation_policy: str = "no_validation",
    verification_only: bool = False,
    recovery_safety_scope_id: str = "",
    recovery_safety_status: str = "none",
) -> tuple[dict[str, Any], FakeProductAgent, ProcessPlanner, dict[str, Any], dict[str, Any]]:
    fixture, product_agent, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(reasoning_mode="multi_turn")
    )
    _configure_runtime_bridge_approval_harness(
        product_agent=product_agent,
        planner=planner,
        fixture=fixture,
        tmp_path=tmp_path,
        bridge_mode=bridge_mode,
        validation_policy=validation_policy,
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
    execution_policy = {
        "complete_full_tail": not verification_only,
    }
    if verification_only:
        execution_policy["verification_only"] = True
        execution_policy["pause_after_verification"] = True
    bridge_debug = {
        "execution_policy": execution_policy,
    }
    if bridge_source:
        bridge_debug["source"] = str(bridge_source)
    if bridge_source == "archived_final_output":
        bridge_debug["archive_replay"] = {
            "source_path": str(CASE3_ARCHIVED_FINAL_OUTPUT_PATH),
            "source_label": CASE3_ARCHIVED_FINAL_OUTPUT_PATH.name,
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
        recovery_safety_scope_id=recovery_safety_scope_id,
        recovery_safety_status=recovery_safety_status,
        recovery_safety_logic_json=(
            str(tmp_path / "recovery_safety" / "cca_safety_logic.json")
            if recovery_safety_scope_id and recovery_safety_status == "ready"
            else ""
        ),
        recovery_safety_dir=(
            str(tmp_path / "recovery_safety") if recovery_safety_scope_id else ""
        ),
        recovery_plan_dir=(
            str(tmp_path / "recovery_safety") if recovery_safety_scope_id else ""
        ),
        recovery_safery_dir=(
            str(tmp_path / "recovery_safety") if recovery_safety_scope_id else ""
        ),
        violations=violations,
    )
    with patch(
        "cais_spade_llm.agents.intelligent_product.product_recovery_controller.threading.Thread",
        new=_ImmediateThread,
    ):
        recovery = product_agent.approve_runtime_bridge_proposal_sync()
    return fixture, product_agent, planner, prepared_bridge_request, recovery


def _copy_case3_archive_with_recovery_safety(
    tmp_path: Path,
    *,
    include_recovery_safety: bool,
) -> Path:
    worked_dir = tmp_path / "worked" / "1"
    recovery_final_dir = worked_dir / "recovery_final"
    recovery_final_dir.mkdir(parents=True, exist_ok=True)
    archive_path = recovery_final_dir / CASE3_ARCHIVED_FINAL_OUTPUT_PATH.name
    shutil.copyfile(CASE3_ARCHIVED_FINAL_OUTPUT_PATH, archive_path)
    if include_recovery_safety:
        recovery_safety_dir = worked_dir / "recovery_safety"
        recovery_safety_dir.mkdir(parents=True, exist_ok=True)
        logic_path = recovery_safety_dir / "cca_safety_logic.json"
        logic_path.write_text(json.dumps({"rules": []}), encoding="utf-8")
        result = {
            "ok": True,
            "recovery_safety_scope_id": "archive_recovery_scope_case3",
            "recovery_safety_status": "ready",
            "recovery_safety_dir": str(recovery_safety_dir),
            "recovery_plan_dir": str(recovery_safety_dir),
            "recovery_safery_dir": str(recovery_safety_dir),
            "recovery_safety_logic_json": str(logic_path),
            "rule_ids": [],
            "rules": [],
            "rule_dfas": {},
        }
        (recovery_safety_dir / "recovery_safety_generation_result.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
    return archive_path


def test_case3_archived_bridge_approval_prunes_redundant_xarm6_move_home_after_home_idle_bridge(
    tmp_path: Path,
) -> None:
    _, product_agent, planner, _, recovery = _approve_case3_archived_bridge(tmp_path)

    assert recovery["status"] == "validating"
    active_bridge_sequence = dict(product_agent.runtime_recovery.get("active_bridge_sequence") or {})
    execution_policy = dict(active_bridge_sequence.get("execution_policy") or {})
    assert execution_policy["complete_full_tail"] is True

    bridge_nodes_by_outline_id = {
        str(node.get("bridge_outline_id") or node.get("params", {}).get("outline_id") or "").strip(): node
        for node in planner.nodes
        if isinstance(node, dict)
        and str(node.get("function_name") or "").strip() == "execute_recovery_macro"
    }
    seq1 = dict(bridge_nodes_by_outline_id["RECOVERY_SEQ1"])
    approval_debug = dict(
        dict(product_agent.runtime_recovery.get("bridge_debug") or {}).get("approval") or {}
    )

    assert list(seq1.get("predecessors") or []) == [ANCHOR_TASK_ID]
    assert str(seq1.get("resource_jid") or "").strip() == "xarm6@localhost"
    assert str(seq1.get("requirement_id") or "").strip() == "REQ_2"
    assert planner._find_node("REQ_2_T5") is None
    assert "REQ_2_T5" not in list(approval_debug.get("entry_task_ids") or [])
    assert "REQ_2_T5" in list(approval_debug.get("deleted_task_ids") or [])


def test_case3_archived_bridge_approval_splices_ur5e_bridge_before_remaining_nominal_chain(
    tmp_path: Path,
) -> None:
    _, _, planner, _, _ = _approve_case3_archived_bridge(tmp_path)

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
    acquire_entity = next(
        (
            dict(node)
            for node in planner.nodes
            if isinstance(node, dict)
            and str(node.get("repair_operator") or "").strip() == "acquire_entity"
            and str(node.get("restores_event_id") or "").strip() == "REQ_1_T3"
        ),
        {},
    )
    req_1_t2 = dict(planner._find_node("REQ_1_T2") or {})
    req_1_t3 = dict(planner._find_node("REQ_1_T3") or {})
    req_1_t4 = dict(planner._find_node("REQ_1_T4") or {})
    req_1_t5 = dict(planner._find_node("REQ_1_T5") or {})

    assert str(seq2.get("resource_jid") or "").strip() == "ur5e@localhost"
    assert str(seq3.get("resource_jid") or "").strip() == "ur5e@localhost"
    assert str(seq4.get("resource_jid") or "").strip() == "ur5e@localhost"
    assert str(seq2.get("requirement_id") or "").strip() == "REQ_1"
    assert str(seq3.get("requirement_id") or "").strip() == "REQ_1"
    assert str(seq4.get("requirement_id") or "").strip() == "REQ_1"
    assert int(seq2.get("sequence_index") or 0) < int(seq3.get("sequence_index") or 0)
    assert int(seq3.get("sequence_index") or 0) < int(seq4.get("sequence_index") or 0)
    assert int(seq4.get("sequence_index") or 0) < int(req_1_t3.get("sequence_index") or 0)
    assert acquire_entity
    assert set(seq2.get("predecessors") or []) == {ANCHOR_TASK_ID, req_1_t2["id"]}
    assert seq1["id"] not in list(seq2.get("predecessors") or [])
    assert list(seq3.get("predecessors") or []) == [seq2["id"]]
    assert list(seq4.get("predecessors") or []) == [seq3["id"]]
    assert list(acquire_entity.get("predecessors") or []) == [seq4["id"]]
    assert list(req_1_t3.get("predecessors") or []) == [acquire_entity["id"]]
    assert list(req_1_t4.get("predecessors") or []) == ["REQ_1_T3"]
    assert list(req_1_t5.get("predecessors") or []) == ["REQ_1_T4"]


def test_case3_archived_bridge_ready_scan_skips_blocked_pending_non_root(
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

    seq1["status"] = "pending"
    seq2["status"] = "pending"
    seq3["status"] = "pending"
    seq4["status"] = "pending"
    active_bridge_sequence = dict(product_agent.runtime_recovery.get("active_bridge_sequence") or {})
    active_bridge_sequence["bridge_task_ids"] = [
        seq3["id"],
        seq1["id"],
        seq2["id"],
        seq4["id"],
    ]
    product_agent.runtime_recovery["active_bridge_sequence"] = active_bridge_sequence

    ready_node = product_agent._active_bridge_next_ready_task()

    assert isinstance(ready_node, dict)
    assert str(ready_node.get("id") or "").strip() == seq1["id"]


def test_case3_prune_redundant_xarm6_move_home_retarges_resume_entry_to_next_nominal_task() -> None:
    class _FakePlanner:
        def __init__(self) -> None:
            self.nodes = [
                {
                    "id": "REQ_3_T5",
                    "type": "task",
                    "function_name": "move_home",
                    "resource_jid": "xarm6@localhost",
                    "sequence_index": 5,
                    "predecessors": ["RECOVERY_BRIDGE_HOME"],
                    "successors": ["REQ_4_T1"],
                },
                {
                    "id": "REQ_4_T1",
                    "type": "task",
                    "function_name": "pick_approach",
                    "resource_jid": "xarm6@localhost",
                    "sequence_index": 6,
                    "predecessors": ["REQ_3_T5"],
                    "successors": [],
                },
            ]

        def _find_node(self, node_id: str) -> dict[str, Any] | None:
            for node in self.nodes:
                if str(node.get("id") or "").strip() == str(node_id or "").strip():
                    return node
            return None

        @staticmethod
        def _task_sequence_index_key(value: Any) -> int:
            return int(value) if value is not None else 10**9

        def _apply_replan_patch(self, modified_tasks: list[dict[str, Any]]) -> None:
            deleted_ids = {
                str(task.get("id") or "").strip()
                for task in modified_tasks
                if task.get("delete") is True and str(task.get("id") or "").strip()
            }
            self.nodes = [
                node
                for node in self.nodes
                if str(node.get("id") or "").strip() not in deleted_ids
            ]
            for node in self.nodes:
                node["predecessors"] = [
                    pred_id
                    for pred_id in (node.get("predecessors") or [])
                    if str(pred_id or "").strip() not in deleted_ids
                ]
                node["successors"] = [
                    succ_id
                    for succ_id in (node.get("successors") or [])
                    if str(succ_id or "").strip() not in deleted_ids
                ]

    fake_controller = SimpleNamespace(
        process_planner=_FakePlanner(),
        _bridge_projected_plant_state_for_resource=lambda **kwargs: {
            "resources": {
                "xarm6@localhost": {
                    "current_state": "idle",
                    "current_pose_ref": "home",
                }
            }
        },
        _plant_resource_field=lambda plant_state, resource_jid, field: (
            ProductRecoveryController._plant_resource_field(
                plant_state,
                resource_jid,
                field,
            )
        ),
    )
    resumable_task_ids = ["REQ_3_T5", "REQ_4_T1"]
    resumable_task_ids_by_resource = {"xarm6@localhost": ["REQ_3_T5", "REQ_4_T1"]}
    resume_entry_task_ids_by_resource = {"xarm6@localhost": "REQ_3_T5"}
    deleted_task_ids: list[str] = []
    tasks = [{"id": "RECOVERY_BRIDGE_HOME", "resource_jid": "xarm6@localhost"}]

    ProductRecoveryController._prune_redundant_bridge_resume_move_home_if_satisfied(
        fake_controller,
        tasks=tasks,
        resumable_task_ids=resumable_task_ids,
        resumable_task_ids_by_resource=resumable_task_ids_by_resource,
        resume_entry_task_ids_by_resource=resume_entry_task_ids_by_resource,
        deleted_task_ids=deleted_task_ids,
    )

    assert fake_controller.process_planner._find_node("REQ_3_T5") is None
    assert resumable_task_ids == ["REQ_4_T1"]
    assert resumable_task_ids_by_resource == {"xarm6@localhost": ["REQ_4_T1"]}
    assert resume_entry_task_ids_by_resource == {"xarm6@localhost": "REQ_4_T1"}
    assert deleted_task_ids == ["REQ_3_T5"]


def test_case3_cca_validation_witness_ordering_repair_uses_only_witness_task_ids(tmp_path: Path) -> None:
    product_agent = SimpleNamespace(
        jid="assembly_board-v1@localhost",
        logger=logging.getLogger("case3_bridge_dryrun"),
        global_fsa_path=tmp_path / "global_fsa.json",
    )
    planner = ProcessPlanner(product_agent, [])
    planner.nodes = [
        {
            "id": "TASK_BLOCKER",
            "type": "task",
            "status": "pending",
            "function_name": "alpha",
            "resource_jid": "resource-A",
            "sequence_index": 0,
            "predecessors": [],
            "successors": [],
            "params": {},
        },
        {
            "id": "TASK_TARGET",
            "type": "task",
            "status": "pending",
            "function_name": "beta",
            "resource_jid": "resource-B",
            "sequence_index": 0,
            "predecessors": [],
            "successors": [],
            "params": {},
        },
    ]
    violations = [
        {
            "violated_rule_id": "SAFE_SYMBOL",
            "witness_transitions": [
                {
                    "from": "(resource-A=(k=0,run=TASK_BLOCKER:alpha),resource-B=(k=0,idle))",
                    "event": "TASK_TARGET.start",
                    "task_id": "TASK_TARGET",
                    "_sigma": ["ap-left", "ap-right"],
                }
            ],
        }
    ]

    repaired = planner.apply_validation_witness_ordering_repairs(violations)

    assert repaired is True
    blocker = next(node for node in planner.nodes if node["id"] == "TASK_BLOCKER")
    target = next(node for node in planner.nodes if node["id"] == "TASK_TARGET")
    assert target["predecessors"] == ["TASK_BLOCKER"]
    assert "TASK_TARGET" in blocker["successors"]


def test_case3_plan_executor_prefers_next_dispatchable_task_node_over_global_bridge_shortcut() -> None:
    sent_messages: list[dict[str, Any]] = []
    nominal_node = {
        "id": "REQ_2_T5",
        "status": "pending",
        "function_name": "move_home",
        "resource_jid": "xarm6@localhost",
        "params": {},
    }
    bridge_node = {
        "id": "RECOVERY_BRIDGE_UR5E",
        "status": "pending",
        "function_name": "execute_recovery_macro",
        "resource_jid": "xarm6@localhost",
        "bridge_sequence_id": "BRIDGE_SEQ",
        "params": {},
    }
    fake_agent = SimpleNamespace(
        resource_jids=["xarm6@localhost", "ur5e@localhost"],
        logger=logging.getLogger("case3_bridge_dryrun"),
        task_states={},
        runtime_recovery={},
        _runtime_recovery_blocks_execution=lambda: False,
        _next_dispatchable_task_node=lambda: nominal_node,
        _active_bridge_next_ready_task=lambda: bridge_node,
        _active_bridge_blocks_nominal_dispatch=lambda: False,
        _active_bridge_sequence=lambda: None,
        _reconstruct_active_bridge_sequence_for_validation=lambda: None,
        _bridge_sequence_task_ids=lambda ids: list(ids or []),
        _set_runtime_recovery=lambda **kwargs: None,
        _compose_task_msg=lambda **kwargs: dict(kwargs),
        _dispatch_params_for_task_node=lambda task_node: dict(task_node.get("params") or {}),
    )
    executor = ProductAgent._PlanExecutor()
    executor.agent = fake_agent  # type: ignore[attr-defined]

    async def _fake_send(msg: dict[str, Any]) -> None:
        sent_messages.append(dict(msg))

    executor.send = _fake_send  # type: ignore[method-assign]

    asyncio.run(executor.run())

    assert sent_messages
    assert str(sent_messages[0].get("task_id") or "").strip() == "REQ_2_T5"
    assert str(nominal_node.get("status") or "").strip() == "dispatched"
    assert str(fake_agent.task_states.get("REQ_2_T5") or "").strip() == "dispatched"


def test_case3_plan_executor_dispatch_guard_allows_different_resource_active_task() -> None:
    sent_messages: list[dict[str, Any]] = []
    nominal_node = {
        "id": "REQ_4_T2",
        "status": "pending",
        "function_name": "pick_grasp",
        "resource_jid": "xarm6@localhost",
        "params": {"part_name": "LCP"},
    }
    recovery_node = {
        "id": "RECOVERY_BRIDGE_ACTIVE",
        "status": "running",
        "function_name": "execute_recovery_macro",
        "resource_jid": "ur5e@localhost",
        "bridge_sequence_id": "BRIDGE_SEQ",
        "params": {
            "destination_location": "assembly_board-v1",
            "primitive_steps": [
                {
                    "primitive": "move_relative",
                    "params": {"dx": 0, "dy": 0, "dz": 0.05},
                }
            ]
        },
    }
    active_bridge_sequence = {
        "bridge_sequence_id": "BRIDGE_SEQ",
        "bridge_task_ids": ["RECOVERY_BRIDGE_ACTIVE"],
        "dispatched_bridge_task_ids": ["RECOVERY_BRIDGE_ACTIVE"],
        "completed_bridge_task_ids": [],
        "state": "executing",
    }
    fake_agent = SimpleNamespace(
        resource_jids=["xarm6@localhost", "ur5e@localhost"],
        logger=logging.getLogger("case3_bridge_dryrun"),
        task_states={},
        runtime_recovery={"active_bridge_sequence": active_bridge_sequence},
        process_planner=SimpleNamespace(nodes=[recovery_node, nominal_node]),
        _runtime_recovery_blocks_execution=lambda: False,
        _next_dispatchable_task_node=lambda: nominal_node,
        _active_bridge_blocks_nominal_dispatch=lambda: False,
        _active_bridge_sequence=lambda: active_bridge_sequence,
        _reconstruct_active_bridge_sequence_for_validation=lambda: None,
        _bridge_sequence_task_ids=lambda ids: list(ids or []),
        _set_runtime_recovery=lambda **kwargs: None,
        _compose_task_msg=lambda **kwargs: dict(kwargs),
        _dispatch_params_for_task_node=lambda task_node: dict(task_node.get("params") or {}),
    )
    executor = ProductAgent._PlanExecutor()
    executor.agent = fake_agent  # type: ignore[attr-defined]

    async def _fake_send(msg: dict[str, Any]) -> None:
        sent_messages.append(dict(msg))

    executor.send = _fake_send  # type: ignore[method-assign]

    asyncio.run(executor.run())

    assert sent_messages
    assert str(sent_messages[0].get("task_id") or "").strip() == "REQ_4_T2"
    assert str(nominal_node.get("status") or "").strip() == "dispatched"
    assert str(fake_agent.task_states.get("REQ_4_T2") or "").strip() == "dispatched"


def test_case3_plan_executor_dispatch_guard_suppresses_same_resource_active_task() -> None:
    sent_messages: list[dict[str, Any]] = []
    active_node = {
        "id": "RECOVERY_BRIDGE_ACTIVE",
        "status": "running",
        "function_name": "execute_recovery_macro",
        "resource_jid": "ur5e@localhost",
        "bridge_sequence_id": "BRIDGE_SEQ",
        "params": {
            "primitive_steps": [
                {
                    "primitive": "move_relative",
                    "params": {"dx": 0, "dy": 0, "dz": 0.05},
                }
            ]
        },
    }
    nominal_node = {
        "id": "REQ_4_T2",
        "status": "pending",
        "function_name": "pick_grasp",
        "resource_jid": "ur5e@localhost",
        "params": {"part_name": "LCP"},
    }
    active_bridge_sequence = {
        "bridge_sequence_id": "BRIDGE_SEQ",
        "bridge_task_ids": ["RECOVERY_BRIDGE_ACTIVE"],
        "dispatched_bridge_task_ids": ["RECOVERY_BRIDGE_ACTIVE"],
        "completed_bridge_task_ids": [],
        "state": "executing",
    }
    fake_agent = SimpleNamespace(
        resource_jids=["xarm6@localhost", "ur5e@localhost"],
        logger=logging.getLogger("case3_bridge_dryrun"),
        task_states={},
        runtime_recovery={"active_bridge_sequence": active_bridge_sequence},
        process_planner=SimpleNamespace(nodes=[active_node, nominal_node]),
        _runtime_recovery_blocks_execution=lambda: False,
        _next_dispatchable_task_node=lambda: nominal_node,
        _active_bridge_blocks_nominal_dispatch=lambda: False,
        _active_bridge_sequence=lambda: active_bridge_sequence,
        _reconstruct_active_bridge_sequence_for_validation=lambda: None,
        _bridge_sequence_task_ids=lambda ids: list(ids or []),
        _set_runtime_recovery=lambda **kwargs: None,
        _compose_task_msg=lambda **kwargs: dict(kwargs),
        _dispatch_params_for_task_node=lambda task_node: dict(task_node.get("params") or {}),
    )
    executor = ProductAgent._PlanExecutor()
    executor.agent = fake_agent  # type: ignore[attr-defined]

    async def _fake_send(msg: dict[str, Any]) -> None:
        sent_messages.append(dict(msg))

    executor.send = _fake_send  # type: ignore[method-assign]

    asyncio.run(executor.run())

    assert sent_messages == []
    assert str(nominal_node.get("status") or "").strip() == "pending"
    assert "REQ_4_T2" not in fake_agent.task_states


class _RuntimeInterlockPlanner:
    def __init__(self, nodes: list[dict[str, Any]], ready_ids: list[str]) -> None:
        self.nodes = nodes
        self.ready_ids = ready_ids

    def _find_node(self, task_id: str) -> dict[str, Any] | None:
        for node in self.nodes:
            if str(node.get("id") or "").strip() == str(task_id or "").strip():
                return node
        return None

    def _bridge_sequence_nodes(self, bridge_sequence_id: str) -> list[dict[str, Any]]:
        return [
            node
            for node in self.nodes
            if str(node.get("bridge_sequence_id") or "").strip()
            == str(bridge_sequence_id or "").strip()
        ]

    def graph_ready_task_nodes(self) -> list[dict[str, Any]]:
        return [
            node
            for task_id in self.ready_ids
            for node in [self._find_node(task_id)]
            if isinstance(node, dict)
        ]

    def next_ready_task(self) -> dict[str, Any] | None:
        ready = self.graph_ready_task_nodes()
        return ready[0] if ready else None

    def _tool_row_for_task(
        self,
        *,
        resource_jid: str,
        function_name: str,
        tools_catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del resource_jid, function_name, tools_catalog
        return {}

    def _resource_by_jid(self, resource_jid: str) -> Any:
        del resource_jid
        return None

    def _extract_resource_states(self, system_state: dict[str, Any]) -> dict[str, Any]:
        return dict((system_state or {}).get("resource_states") or {})


def _runtime_interlock_product(
    *,
    bridge_node: dict[str, Any],
    nominal_nodes: list[dict[str, Any]],
    ready_ids: list[str],
    validation_policy: str = "no_validation",
) -> Any:
    active_bridge_sequence = {
        "bridge_sequence_id": str(bridge_node.get("bridge_sequence_id") or "").strip(),
        "bridge_task_ids": [str(bridge_node.get("id") or "").strip()],
        "dispatched_bridge_task_ids": [str(bridge_node.get("id") or "").strip()],
        "completed_bridge_task_ids": [],
        "state": "executing",
        "validation_policy": validation_policy,
        "execution_policy": {"validation_policy": validation_policy},
    }
    product = SimpleNamespace(
        logger=logging.getLogger("case3_bridge_dryrun"),
        process_planner=_RuntimeInterlockPlanner(
            [bridge_node] + list(nominal_nodes),
            ready_ids,
        ),
        runtime_recovery={
            "active_bridge_sequence": active_bridge_sequence,
            "validation_policy": validation_policy,
        },
        _runtime_recovery_context={"validation_policy": validation_policy},
        _runtime_bridge_validation_policy=validation_policy,
        _orphaned_bridge_task_warning_ids=set(),
        tools_catalog=[],
        part_tracker={},
        resource_states={},
        task_states={},
        _utc_now_iso=lambda: "2026-04-26T00:00:00+00:00",
    )
    controller = ProductRecoveryController(product)
    controller.bind_methods()
    return product


def test_case3_product_selection_does_not_inspect_destination_location_for_recovery_safety_blocking() -> None:
    bridge_node = {
        "id": "RECOVERY_BRIDGE_ACTIVE",
        "type": "task",
        "status": "running",
        "function_name": "execute_recovery_macro",
        "resource_jid": "ur5e@localhost",
        "bridge_sequence_id": "BRIDGE_SEQ",
        "params": {
            "destination_location": "assembly_board-v1",
            "primitive_steps": [
                {
                    "primitive": "compute_place_targets",
                    "params": {"destination_location": "assembly_board-v1"},
                }
            ],
        },
    }
    nominal_node = {
        "id": "REQ_4_T3",
        "type": "task",
        "status": "pending",
        "function_name": "place_approach",
        "resource_jid": "xarm6@localhost",
        "params": {"destination_location": "assembly_board-v1"},
    }
    product = _runtime_interlock_product(
        bridge_node=bridge_node,
        nominal_nodes=[nominal_node],
        ready_ids=["REQ_4_T3"],
    )

    selected = product._select_runtime_event()

    assert isinstance(selected, dict)
    assert str(selected.get("id") or "").strip() == "REQ_4_T3"


def test_case3_product_selection_allows_xarm6_lcp_pick_tasks_while_ur5e_recovery_active() -> None:
    for task_id, function_name in (
        ("REQ_4_T1", "pick_approach"),
        ("REQ_4_T2", "pick_grasp"),
    ):
        bridge_node = {
            "id": "RECOVERY_BRIDGE_ACTIVE",
            "type": "task",
            "status": "running",
            "function_name": "execute_recovery_macro",
            "resource_jid": "ur5e@localhost",
            "bridge_sequence_id": "BRIDGE_SEQ",
            "params": {
                "destination_location": "assembly_board-v1",
                "primitive_steps": [
                    {
                        "primitive": "move_relative",
                        "params": {"dx": 0, "dy": 0, "dz": 0.05},
                    }
                ],
            },
        }
        nominal_node = {
            "id": task_id,
            "type": "task",
            "status": "pending",
            "function_name": function_name,
            "resource_jid": "xarm6@localhost",
            "params": {"part_name": "LCP"},
        }
        product = _runtime_interlock_product(
            bridge_node=bridge_node,
            nominal_nodes=[nominal_node],
            ready_ids=[task_id],
            validation_policy="validated",
        )

        selected = product._select_runtime_event()

        assert isinstance(selected, dict)
        assert str(selected.get("id") or "").strip() == task_id


def test_case3_product_selection_does_not_inspect_recovery_primitive_steps_for_safety_blocking() -> None:
    bridge_node = {
        "id": "RECOVERY_BRIDGE_ACTIVE",
        "type": "task",
        "status": "running",
        "function_name": "execute_recovery_macro",
        "resource_jid": "ur5e@localhost",
        "bridge_sequence_id": "BRIDGE_SEQ",
        "params": {
            "primitive_steps": [
                {
                    "primitive": "move_relative",
                    "params": {"dx": 0, "dy": 0, "dz": 0.05},
                }
            ],
        },
    }
    nominal_node = {
        "id": "REQ_4_T2",
        "type": "task",
        "status": "pending",
        "function_name": "pick_grasp",
        "resource_jid": "xarm6@localhost",
        "params": {"part_name": "LCP"},
    }
    product = _runtime_interlock_product(
        bridge_node=bridge_node,
        nominal_nodes=[nominal_node],
        ready_ids=["REQ_4_T2"],
        validation_policy="validated",
    )

    selected = product._select_runtime_event()

    assert isinstance(selected, dict)
    assert str(selected.get("id") or "").strip() == "REQ_4_T2"


def test_case3_product_selection_does_not_inspect_missing_primitive_steps_for_safety_blocking() -> None:
    bridge_node = {
        "id": "RECOVERY_BRIDGE_ACTIVE",
        "type": "task",
        "status": "running",
        "function_name": "execute_recovery_macro",
        "resource_jid": "ur5e@localhost",
        "bridge_sequence_id": "BRIDGE_SEQ",
        "params": {"macro_name": "recover_to_home_idle"},
    }
    nominal_node = {
        "id": "REQ_4_T3",
        "type": "task",
        "status": "pending",
        "function_name": "place_approach",
        "resource_jid": "xarm6@localhost",
        "params": {"destination_location": "assembly_board-v1"},
    }
    product = _runtime_interlock_product(
        bridge_node=bridge_node,
        nominal_nodes=[nominal_node],
        ready_ids=["REQ_4_T3"],
        validation_policy="validated",
    )

    selected = product._select_runtime_event()

    assert isinstance(selected, dict)
    assert str(selected.get("id") or "").strip() == "REQ_4_T3"


def test_case3_no_validation_active_bridge_dispatch_params_do_not_force_fast_path_for_bridge_and_nominal_tasks(
    tmp_path: Path,
) -> None:
    _, product_agent, planner, _, recovery = _approve_case3_archived_bridge(
        tmp_path,
        validation_policy="no_validation",
    )

    assert recovery["status"] == "validating"
    seq1 = next(
        node
        for node in planner.nodes
        if isinstance(node, dict)
        and str(node.get("bridge_outline_id") or "").strip() == "RECOVERY_SEQ1"
    )
    req_4_t1 = dict(planner._find_node("REQ_4_T1") or {})
    req_1_t3 = dict(planner._find_node("REQ_1_T3") or {})

    seq1_params = product_agent._dispatch_params_for_task_node(seq1)
    req_4_t1_params = product_agent._dispatch_params_for_task_node(req_4_t1)
    req_1_t3_params = product_agent._dispatch_params_for_task_node(req_1_t3)

    assert "start_safety_mode" not in seq1_params
    assert "start_safety_mode" not in req_4_t1_params
    assert "start_safety_mode" not in req_1_t3_params


def test_case3_archived_bridge_approval_waits_for_recovery_safety_ready(
    tmp_path: Path,
) -> None:
    with patch.object(
        ProductRecoveryController,
        "_send_runtime_plan_validation_check_sync",
        autospec=True,
        return_value=None,
    ) as mocked_validation:
        _, product_agent, _, _, recovery = _approve_case3_archived_bridge(
            tmp_path,
            bridge_source="live_multi_turn",
            bridge_mode="auto",
            validation_policy="validated",
            recovery_safety_scope_id="recovery_scope_case3",
            recovery_safety_status="generating",
        )

    assert str(recovery.get("status") or "").strip() == "llm_bridge"
    assert dict(recovery.get("action_feedback") or {}).get("kind") == "warning"
    assert "still generating" in str(
        dict(recovery.get("action_feedback") or {}).get("text") or ""
    )
    assert product_agent.runtime_recovery.get("active_bridge_sequence") is None
    mocked_validation.assert_not_called()


def test_case3_validated_archive_loads_sibling_recovery_safety_result(
    tmp_path: Path,
) -> None:
    fixture, product_agent, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(reasoning_mode="multi_turn")
    )
    archive_path = _copy_case3_archive_with_recovery_safety(
        tmp_path,
        include_recovery_safety=True,
    )
    _configure_runtime_bridge_approval_harness(
        product_agent=product_agent,
        planner=planner,
        fixture=fixture,
        tmp_path=tmp_path,
        bridge_mode="pre_ran",
        validation_policy="validated",
    )
    product_agent._runtime_bridge_archive_path = str(archive_path)
    product_agent._runtime_bridge_archive_label = archive_path.name
    product_agent._runtime_bridge_data_root = lambda: tmp_path
    product_agent._persist_product_state = lambda: None
    product_agent._persist_plan_snapshot = lambda: None
    product_agent._persist_resource_state = lambda: None
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
        status="bridge_ready",
        trigger="runtime_des_replan",
        failed_task_id=FAILED_TASK_ID,
        message="Awaiting archived bridge load.",
        bridge_debug=deepcopy(prepared_bridge_request.get("bridge_debug") or {}),
        bridge_approval_state="ready",
        violations=violations,
    )

    with patch.object(
        ProductRecoveryController,
        "_runtime_bridge_data_root",
        autospec=True,
        return_value=tmp_path,
    ), patch(
        "cais_spade_llm.agents.intelligent_product.product_recovery_controller.asyncio.to_thread",
        new=_immediate_to_thread,
    ):
        recovery = asyncio.run(
            product_agent.load_runtime_bridge_archive_proposal(str(archive_path))
        )

    assert str(recovery.get("recovery_safety_scope_id") or "").strip() == "archive_recovery_scope_case3"
    assert str(recovery.get("recovery_safety_status") or "").strip() == "ready"

    with patch(
        "cais_spade_llm.agents.intelligent_product.product_recovery_controller.threading.Thread",
        new=_ImmediateThread,
    ):
        recovery = product_agent.approve_runtime_bridge_proposal_sync()

    assert recovery["status"] == "validating"
    active_bridge_sequence = dict(
        product_agent.runtime_recovery.get("active_bridge_sequence") or {}
    )
    assert str(active_bridge_sequence.get("recovery_safety_scope_id") or "").strip() == "archive_recovery_scope_case3"
    plan_safety_checks = [
        dict(row.get("body") or {})
        for row in product_agent.dispatched_agent_messages
        if dict(row.get("metadata") or {}).get("type") == "plan_safety_check"
    ]
    assert plan_safety_checks
    assert str(
        dict(plan_safety_checks[-1].get("recovery_safety_result") or {}).get(
            "recovery_safety_scope_id"
        )
        or ""
    ).strip() == "archive_recovery_scope_case3"
    seq1 = next(
        node
        for node in planner.nodes
        if isinstance(node, dict)
        and str(node.get("bridge_outline_id") or "").strip() == "RECOVERY_SEQ1"
    )
    seq1_params = product_agent._dispatch_params_for_task_node(seq1)
    assert str(seq1_params.get("recovery_safety_scope_id") or "").strip() == "archive_recovery_scope_case3"
    assert str(seq1_params.get("start_safety_mode") or "").strip() == "cca_check"


def test_case3_validated_archive_without_recovery_safety_result_defers_approval(
    tmp_path: Path,
) -> None:
    fixture, product_agent, planner, prepared_bridge_request = asyncio.run(
        _prepare_bridge_dryrun_harness(reasoning_mode="multi_turn")
    )
    archive_path = _copy_case3_archive_with_recovery_safety(
        tmp_path,
        include_recovery_safety=False,
    )
    _configure_runtime_bridge_approval_harness(
        product_agent=product_agent,
        planner=planner,
        fixture=fixture,
        tmp_path=tmp_path,
        bridge_mode="pre_ran",
        validation_policy="validated",
    )
    product_agent._runtime_bridge_archive_path = str(archive_path)
    product_agent._runtime_bridge_archive_label = archive_path.name
    product_agent._runtime_bridge_data_root = lambda: tmp_path
    product_agent._persist_product_state = lambda: None
    product_agent._persist_plan_snapshot = lambda: None
    product_agent._persist_resource_state = lambda: None
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
        status="bridge_ready",
        trigger="runtime_des_replan",
        failed_task_id=FAILED_TASK_ID,
        message="Awaiting archived bridge load.",
        bridge_debug=deepcopy(prepared_bridge_request.get("bridge_debug") or {}),
        bridge_approval_state="ready",
        violations=violations,
    )

    with patch.object(
        ProductRecoveryController,
        "_runtime_bridge_data_root",
        autospec=True,
        return_value=tmp_path,
    ), patch(
        "cais_spade_llm.agents.intelligent_product.product_recovery_controller.asyncio.to_thread",
        new=_immediate_to_thread,
    ):
        recovery = asyncio.run(
            product_agent.load_runtime_bridge_archive_proposal(str(archive_path))
        )

    assert str(recovery.get("recovery_safety_scope_id") or "").strip()
    assert str(recovery.get("recovery_safety_status") or "").strip() == "generating"
    assert any(
        dict(row.get("metadata") or {}).get("type") == "recovery_safety_generate"
        for row in product_agent.dispatched_agent_messages
    )
    with patch.object(
        ProductRecoveryController,
        "_send_runtime_plan_validation_check_sync",
        autospec=True,
        return_value=None,
    ) as mocked_validation:
        recovery = product_agent.approve_runtime_bridge_proposal_sync()

    mocked_validation.assert_not_called()
    assert str(recovery.get("status") or "").strip() == "llm_bridge"
    assert dict(recovery.get("action_feedback") or {}).get("kind") == "warning"
    assert "still generating" in str(
        dict(recovery.get("action_feedback") or {}).get("text") or ""
    )


def test_case3_recovery_scope_dispatch_params_force_cca_check_for_bridge_and_nominal_tasks(
    tmp_path: Path,
) -> None:
    _, product_agent, planner, _, recovery = _approve_case3_archived_bridge(
        tmp_path,
        bridge_source="live_multi_turn",
        bridge_mode="auto",
        validation_policy="validated",
        recovery_safety_scope_id="recovery_scope_case3",
        recovery_safety_status="ready",
    )

    assert recovery["status"] == "validating"
    active_bridge_sequence = dict(product_agent.runtime_recovery.get("active_bridge_sequence") or {})
    seq1 = next(
        node
        for node in planner.nodes
        if isinstance(node, dict)
        and str(node.get("bridge_outline_id") or "").strip() == "RECOVERY_SEQ1"
    )
    req_1_t3 = dict(planner._find_node("REQ_1_T3") or {})

    assert "REQ_1_T3" in list(active_bridge_sequence.get("recovery_enforced_task_ids") or [])

    seq1_params = product_agent._dispatch_params_for_task_node(seq1)
    req_1_t3_params = product_agent._dispatch_params_for_task_node(req_1_t3)

    assert str(seq1_params.get("recovery_safety_scope_id") or "").strip() == "recovery_scope_case3"
    assert str(req_1_t3_params.get("recovery_safety_scope_id") or "").strip() == "recovery_scope_case3"
    assert str(seq1_params.get("start_safety_mode") or "").strip() == "cca_check"
    assert str(req_1_t3_params.get("start_safety_mode") or "").strip() == "cca_check"


def test_case3_recovery_scope_dispatch_params_force_cca_check_for_bridge_task_even_if_enforced_list_is_stale(
    tmp_path: Path,
) -> None:
    _, product_agent, planner, _, recovery = _approve_case3_archived_bridge(
        tmp_path,
        bridge_source="live_multi_turn",
        bridge_mode="auto",
        validation_policy="validated",
        recovery_safety_scope_id="recovery_scope_case3",
        recovery_safety_status="ready",
    )

    assert recovery["status"] == "validating"
    active_bridge_sequence = dict(product_agent.runtime_recovery.get("active_bridge_sequence") or {})
    active_bridge_sequence["recovery_enforced_task_ids"] = []
    active_bridge_sequence["bridge_task_ids"] = []
    active_bridge_sequence["dispatched_bridge_task_ids"] = []
    product_agent.runtime_recovery["active_bridge_sequence"] = active_bridge_sequence
    product_agent.runtime_recovery["recovery_enforced_task_ids"] = []

    seq1 = next(
        node
        for node in planner.nodes
        if isinstance(node, dict)
        and str(node.get("bridge_outline_id") or "").strip() == "RECOVERY_SEQ1"
    )

    seq1_params = product_agent._dispatch_params_for_task_node(seq1)

    assert str(seq1_params.get("recovery_safety_scope_id") or "").strip() == "recovery_scope_case3"
    assert str(seq1_params.get("start_safety_mode") or "").strip() == "cca_check"


def test_case3_recovery_scope_dispatch_params_block_validated_recovery_task_without_scope(
    tmp_path: Path,
) -> None:
    _, product_agent, planner, _, recovery = _approve_case3_archived_bridge(
        tmp_path,
        bridge_source="live_multi_turn",
        bridge_mode="auto",
        validation_policy="validated",
    )

    assert recovery["status"] == "llm_bridge"
    assert dict(recovery.get("action_feedback") or {}).get("kind") == "warning"
    assert "no recovery_safety_scope_id" in str(
        dict(recovery.get("action_feedback") or {}).get("text") or ""
    )
    assert product_agent.runtime_recovery.get("active_bridge_sequence") is None

    _, product_agent, planner, _, recovery = _approve_case3_archived_bridge(
        tmp_path,
        bridge_source="live_multi_turn",
        bridge_mode="auto",
        validation_policy="validated",
        recovery_safety_scope_id="recovery_scope_case3",
        recovery_safety_status="ready",
    )
    assert recovery["status"] == "validating"
    active_bridge_sequence = dict(
        product_agent.runtime_recovery.get("active_bridge_sequence") or {}
    )
    active_bridge_sequence["recovery_safety_scope_id"] = ""
    product_agent.runtime_recovery["active_bridge_sequence"] = active_bridge_sequence
    product_agent.runtime_recovery["recovery_safety_scope_id"] = ""
    seq1 = next(
        node
        for node in planner.nodes
        if isinstance(node, dict)
        and str(node.get("bridge_outline_id") or "").strip() == "RECOVERY_SEQ1"
    )

    try:
        product_agent._dispatch_params_for_task_node(seq1)
    except RuntimeError as exc:
        assert "Recovery Safety Check dispatch blocked" in str(exc)
    else:
        raise AssertionError("validated recovery task without recovery_safety_scope_id was not blocked")

    assert str(product_agent.runtime_recovery.get("status") or "").strip() == "human_required"
    assert product_agent._runtime_recovery_blocks_execution() is True


def test_case3_no_validation_recovery_scope_dispatch_params_do_not_route_scoped_recovery_safety_runtime_check(
    tmp_path: Path,
) -> None:
    _, product_agent, planner, _, recovery = _approve_case3_archived_bridge(
        tmp_path,
        bridge_source="live_multi_turn",
        bridge_mode="auto",
        validation_policy="no_validation",
        recovery_safety_scope_id="recovery_scope_case3",
        recovery_safety_status="ready",
    )

    assert recovery["status"] == "validating"
    seq1 = next(
        node
        for node in planner.nodes
        if isinstance(node, dict)
        and str(node.get("bridge_outline_id") or "").strip() == "RECOVERY_SEQ1"
    )
    req_1_t3 = dict(planner._find_node("REQ_1_T3") or {})

    seq1_params = product_agent._dispatch_params_for_task_node(seq1)
    req_1_t3_params = product_agent._dispatch_params_for_task_node(req_1_t3)

    assert "recovery_safety_scope_id" not in seq1_params
    assert "recovery_safety_scope_id" not in req_1_t3_params
    assert "start_safety_mode" not in seq1_params
    assert "start_safety_mode" not in req_1_t3_params


def test_case3_recovery_scope_dispatch_params_include_recovery_outline_grounding_metadata() -> None:
    fake_agent = SimpleNamespace()
    fake_agent.runtime_recovery = {
        "recovery_safety_scope_id": "recovery_scope_case3",
        "recovery_enforced_task_ids": ["RECOVERY_BRIDGE_SEQ4"],
    }
    fake_agent._runtime_recovery_context = {}
    fake_agent._geometry_for_part = lambda part_name: {}
    fake_agent._enrich_observed_pose_recovery_params = lambda params: dict(params)
    fake_agent._active_bridge_sequence = lambda: {
        "recovery_safety_scope_id": "recovery_scope_case3",
        "recovery_enforced_task_ids": ["RECOVERY_BRIDGE_SEQ4"],
    }
    fake_agent._bridge_sequence_is_live = lambda _: False
    fake_agent._runtime_bridge_session_validation_policy = lambda: "validated"

    task_node = {
        "id": "RECOVERY_BRIDGE_SEQ4",
        "function_name": "execute_recovery_macro",
        "bridge_outline_id": "RECOVERY_SEQ4",
        "llm_outline_id": "recovery_ur5e_lg_place_004",
        "event_name": "recover_place_LG_to_assembly_board-v1",
        "projected_outline_state": {
            "resource_state": "idle",
            "held_part": None,
            "part_state": "placed",
            "part_location": "assembly_board-v1",
        },
        "params": {
            "macro_name": "recover_place_LG_to_assembly_board-v1",
            "outline_id": "RECOVERY_SEQ4",
            "expected_start_state": "picked",
            "outline_expected_start_state": {
                "resource_state": "picked",
                "held_part": "LG",
                "part_state": "held",
            },
            "expected_end_state": {
                "resource_state": "idle",
                "held_part": None,
                "part_state": "placed",
                "part_location": "assembly_board-v1",
            },
            "part_name": "LG",
        },
    }

    params = ProductRecoveryController._dispatch_params_for_task_node(fake_agent, task_node)

    assert str(params.get("task_id") or "").strip() == "RECOVERY_BRIDGE_SEQ4"
    assert str(params.get("recovery_safety_scope_id") or "").strip() == "recovery_scope_case3"
    assert str(params.get("start_safety_mode") or "").strip() == "cca_check"
    assert str(params.get("bridge_outline_id") or "").strip() == "RECOVERY_SEQ4"
    assert str(params.get("outline_id") or "").strip() == "RECOVERY_SEQ4"
    assert str(params.get("llm_outline_id") or "").strip() == "recovery_ur5e_lg_place_004"
    assert str(params.get("event_name") or "").strip() == "recover_place_LG_to_assembly_board-v1"
    assert dict(params.get("outline_expected_start_state") or {}).get("held_part") == "LG"
    assert dict(params.get("expected_end_state") or {}).get("part_location") == "assembly_board-v1"
    assert dict(params.get("projected_outline_state") or {}).get("part_state") == "placed"


def test_case3_dryrun_recovery_safety_generation_writes_debug_artifacts(
    tmp_path: Path,
) -> None:
    async def _run() -> dict[str, Any]:
        _, product_agent, _, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
            reasoning_mode="multi_turn"
        )
        multi_turn_session = {
            "accepted_outline_prefix": _case3_archived_transition_trace(),
            "projected_outline_state": _case3_archived_projected_outline_state(),
        }
        payload = _build_dryrun_recovery_safety_generation_payload(
            prepared_bridge_request=prepared_bridge_request,
            multi_turn_session=multi_turn_session,
            recovery_safety_scope_id="dryrun_recovery_scope_case3",
        )

        async def _fake_ask_llm_structured(
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
            max_tool_rounds: int = 3,
        ) -> dict[str, Any]:
            del tools, tool_executor, max_tool_rounds
            assert "\"outline_id\": \"RECOVERY_SEQ4\"" in prompt
            assert "\"nominal_candidate_tasks\"" not in prompt
            assert "\"nominal_candidate_task_ids\"" not in prompt
            assert "\"recovery_safety_scope_id\"" not in prompt
            assert "dryrun_recovery_scope_case3" not in prompt
            assert "Use the supplied recovery outline row fields" in prompt
            assert "Copy only supplied outline_id values" in prompt
            assert "selected_nominal_task_ids" not in prompt
            assert "selected_nominal_state_tokens" not in prompt
            assert "selected_recovery_state_tokens" not in prompt
            assert "Do not select nominal task rows." in prompt
            assert "Use only the supplied row ids and supplied state_tokens." not in prompt
            assert "return rule_id, status, reason, and selected_recovery_outline_ids" in prompt
            assert "Do not generate aps." not in prompt
            assert "Do not generate ltlf." not in prompt
            assert "generated_aps" not in prompt
            assert "\"ltlf\"" not in prompt
            assert "Do not adapt one resource to another resource." in prompt
            prompt_payload = json.loads(prompt.split("\n\n", 1)[1])
            assert "nominal_candidate_tasks" not in prompt_payload
            response_schema = json.dumps(response_format, sort_keys=True)
            assert "selected_recovery_outline_ids" in response_schema
            assert "selected_nominal_task_ids" not in response_schema
            assert "selected_nominal_state_tokens" not in response_schema
            assert "selected_recovery_state_tokens" not in response_schema
            return _fake_case3_recovery_safety_grounding_response(payload)

        product_agent.ask_llm_structured = _fake_ask_llm_structured  # type: ignore[method-assign]
        with patch("test.test_case3_bridge_dryrun.DEBUG_DIR", tmp_path):
            return await _generate_dryrun_recovery_safety_artifacts(
                product_agent=product_agent,
                prepared_bridge_request=prepared_bridge_request,
                multi_turn_session=multi_turn_session,
                recovery_safety_scope_id="dryrun_recovery_scope_case3",
            )

    result = asyncio.run(_run())

    assert result["ok"] is True
    assert Path(str(result.get("recovery_safety_dir") or "")) == tmp_path / "recovery_safety"
    assert Path(str(result.get("recovery_plan_dir") or "")) == tmp_path / "recovery_safety"
    assert Path(str(result.get("recovery_safery_dir") or "")) == tmp_path / "recovery_safety"
    assert Path(str(result.get("recovery_safety_dir") or "")).is_dir()
    assert not (tmp_path / "recovery_plan").exists()
    assert Path(str(result.get("snapshot_artifact_path") or "")).is_file()
    assert Path(str(result.get("grounding_prompt_artifact_path") or "")).is_file()
    assert Path(str(result.get("latest_grounding_prompt_artifact_path") or "")).is_file()
    assert Path(str(result.get("grounding_response_artifact_path") or "")).is_file()
    assert Path(str(result.get("latest_grounding_response_artifact_path") or "")).is_file()
    assert Path(str(result.get("grounding_llm_response_artifact_path") or "")).is_file()
    assert Path(
        str(result.get("latest_grounding_llm_response_artifact_path") or "")
    ).is_file()
    assert Path(str(result.get("recovery_safety_logic_json") or "")).is_file()
    assert list(result.get("rule_ids") or []) == ["SAFE_2"]
    assert Path(str(result.get("recovery_safety_dir") or "")).joinpath(
        "SAFE_2_dfa.dot"
    ).is_file()
    assert not Path(str(result.get("recovery_safety_dir") or "")).joinpath(
        "SAFE_1_dfa.dot"
    ).exists()
    all_rule_results = {
        str(row.get("rule_id") or "").strip(): dict(row)
        for row in (result.get("all_rule_results") or [])
        if isinstance(row, dict)
    }
    assert len(result.get("accepted_outline_prefix") or []) == 4
    assert len(result.get("pending_nominal_tasks") or []) >= 1
    assert any(
        str(row.get("id") or "").strip() == "REQ_2_T3"
        for row in (result.get("nominal_candidate_tasks") or [])
        if isinstance(row, dict)
    )
    nominal_by_id = {
        str(row.get("id") or "").strip(): dict(row)
        for row in (result.get("nominal_candidate_tasks") or [])
        if isinstance(row, dict)
    }
    assert str(nominal_by_id["REQ_2_T3"].get("destination_location") or "").strip() == (
        "Assembly Station"
    )
    assert str(nominal_by_id["REQ_2_T4"].get("destination_location") or "").strip() == (
        "Assembly Station"
    )
    for task_id in ("REQ_1_T5", "REQ_2_T1", "REQ_2_T2", "REQ_2_T5"):
        assert str(nominal_by_id[task_id].get("destination_location") or "").strip() == ""
    assert str(all_rule_results["SAFE_1"].get("status") or "").strip() == "not_involved"
    assert str(all_rule_results["SAFE_2"].get("status") or "").strip() == "grounded"
    assert list(all_rule_results["SAFE_1"].get("selected_recovery_outline_ids") or []) == [
        "RECOVERY_SEQ3",
        "RECOVERY_SEQ4",
    ]
    assert list(all_rule_results["SAFE_1"].get("selected_nominal_task_ids") or []) == []
    assert list(all_rule_results["SAFE_2"].get("selected_nominal_task_ids") or []) == [
        "REQ_2_T3",
        "REQ_2_T4",
    ]
    recovery_seq4 = next(
        row
        for row in _case3_archived_transition_trace()
        if str(row.get("outline_id") or "").strip() == "RECOVERY_SEQ4"
    )
    recovery_seq4_end_state = dict(
        recovery_seq4.get("expected_end_state")
        or recovery_seq4.get("projected_outline_state")
        or {}
    )
    recovery_seq4_part_state = str(
        recovery_seq4_end_state.get("part_state") or ""
    ).strip()
    assert recovery_seq4_part_state
    safe_2_result = dict(all_rule_results["SAFE_2"])
    safe_2_ap_details = [
        dict(row) for row in (safe_2_result.get("aps") or []) if isinstance(row, dict)
    ]
    recovery_event_fulls = [
        str(row.get("full") or "").strip()
        for row in safe_2_ap_details
        if str(row.get("source") or "").strip() == "recovery"
        and str(row.get("kind") or "").strip() == "ap_event"
    ]
    recovery_state_fulls = [
        str(row.get("full") or "").strip()
        for row in safe_2_ap_details
        if str(row.get("source") or "").strip() == "recovery"
        and str(row.get("kind") or "").strip() == "ap_state"
    ]
    nominal_ap_fulls = [
        str(row.get("full") or "").strip()
        for row in safe_2_ap_details
        if str(row.get("source") or "").strip() == "nominal"
    ]
    nominal_state_fulls = [
        str(row.get("full") or "").strip()
        for row in safe_2_ap_details
        if str(row.get("source") or "").strip() == "nominal"
        and str(row.get("kind") or "").strip() == "ap_state"
    ]
    assert any("execute_recovery_macro" in token for token in recovery_event_fulls)
    assert any("task_id=REQ_2_T3" in token for token in nominal_ap_fulls)
    assert any(
        "positioned" in token and "task_id=REQ_2_T3" in token
        for token in nominal_state_fulls
    )
    assert any(
        "placed" in token and "task_id=REQ_2_T4" in token
        for token in nominal_state_fulls
    )
    grounded_nominal_states = [
        dict(row)
        for row in (safe_2_result.get("grounded_nominal_states") or [])
        if isinstance(row, dict)
    ]
    assert any(
        str(row.get("id") or "").strip() == "REQ_2_T3"
        and str(row.get("value") or "").strip() == "positioned"
        for row in grounded_nominal_states
    )
    assert any(
        str(row.get("id") or "").strip() == "REQ_2_T4"
        and str(row.get("value") or "").strip() == "placed"
        for row in grounded_nominal_states
    )
    nominal_side_aps = [
        str(token or "").strip()
        for token in (safe_2_result.get("nominal_side_aps") or [])
        if str(token or "").strip()
    ]
    assert any(
        "positioned" in token and "task_id=REQ_2_T3" in token
        for token in nominal_side_aps
    )
    assert any(
        "placed" in token and "task_id=REQ_2_T4" in token
        for token in nominal_side_aps
    )
    assert any(
        str(row.get("outline_id") or "").strip() == "RECOVERY_SEQ4"
        and str(row.get("field") or "").strip() == "part_state"
        and str(row.get("value") or "").strip() == recovery_seq4_part_state
        for row in safe_2_ap_details
        if str(row.get("source") or "").strip() == "recovery"
        and str(row.get("kind") or "").strip() == "ap_state"
    )
    assert any(
        recovery_seq4_part_state in token and "outline_id=RECOVERY_SEQ4" in token
        for token in recovery_state_fulls
    )
    safe_2_ltlf = str(safe_2_result.get("ltlf") or "").strip()
    assert safe_2_ltlf.startswith("G !((")
    assert ") & (" in safe_2_ltlf
    for token in recovery_event_fulls + recovery_state_fulls + nominal_ap_fulls:
        assert token in safe_2_ltlf
    assert recovery_event_fulls and recovery_state_fulls
    same_side_pair = f"{recovery_event_fulls[0]} & {recovery_state_fulls[0]}"
    reverse_same_side_pair = f"{recovery_state_fulls[0]} & {recovery_event_fulls[0]}"
    assert same_side_pair not in safe_2_ltlf
    assert reverse_same_side_pair not in safe_2_ltlf
    raw_grounding_response_payload = json.loads(
        Path(
            str(result.get("latest_grounding_llm_response_artifact_path") or "")
        ).read_text(encoding="utf-8")
    )
    raw_safe_2 = next(
        row
        for row in (raw_grounding_response_payload.get("rules") or [])
        if isinstance(row, dict) and str(row.get("rule_id") or "").strip() == "SAFE_2"
    )
    assert list(raw_safe_2.get("selected_recovery_outline_ids") or []) == [
        "RECOVERY_SEQ4",
    ]
    assert "selected_nominal_task_ids" not in raw_safe_2
    grounding_response_payload = json.loads(
        Path(str(result.get("latest_grounding_response_artifact_path") or "")).read_text(
            encoding="utf-8"
        )
    )
    assert "accepted_outline_prefix" not in grounding_response_payload
    assert "pending_nominal_tasks" not in grounding_response_payload
    selected_safe_2 = next(
        row
        for row in (grounding_response_payload.get("rules") or [])
        if isinstance(row, dict) and str(row.get("rule_id") or "").strip() == "SAFE_2"
    )
    assert list(selected_safe_2.get("selected_recovery_outline_ids") or []) == [
        "RECOVERY_SEQ4",
    ]
    assert list(selected_safe_2.get("selected_nominal_task_ids") or []) == [
        "REQ_2_T3",
        "REQ_2_T4",
    ]
    rules = [dict(row) for row in (result.get("rules") or []) if isinstance(row, dict)]
    aps = list(rules[0].get("aps") or [])
    assert any(
        "execute_recovery_macro" in str(ap.get("full") or "")
        for ap in aps
    )
    assert any(
        "task_id=REQ_2_T3" in str(ap.get("full") or "")
        for ap in aps
    )
    assert any(
        "positioned" in str(ap.get("full") or "")
        and "task_id=REQ_2_T3" in str(ap.get("full") or "")
        for ap in aps
    )
    assert any(
        "placed" in str(ap.get("full") or "")
        and "task_id=REQ_2_T4" in str(ap.get("full") or "")
        for ap in aps
    )
    assert any(
        str(ap.get("source") or "").strip() == "recovery"
        and str(ap.get("kind") or "").strip() == "ap_state"
        and str(ap.get("outline_id") or "").strip() == "RECOVERY_SEQ4"
        for ap in aps
        if isinstance(ap, dict)
    )
    assert not any(
        "recover_place_LG_to_assembly_board-v1" in str(ap.get("full") or "")
        for ap in aps
    )


def test_case3_recovery_safety_generation_preserves_recovery_completion_state_symbols(
    tmp_path: Path,
) -> None:
    async def _run(part_state: str, debug_dir: Path) -> dict[str, Any]:
        _, product_agent, _, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
            reasoning_mode="multi_turn"
        )
        trace = _case3_archived_transition_trace()
        for row in trace:
            if str(row.get("outline_id") or "").strip() != "RECOVERY_SEQ4":
                continue
            expected_end_state = dict(row.get("expected_end_state") or {})
            expected_end_state["part_state"] = part_state
            row["expected_end_state"] = expected_end_state
            if isinstance(row.get("projected_outline_state"), dict):
                projected_outline_state = dict(row.get("projected_outline_state") or {})
                projected_outline_state["part_state"] = part_state
                row["projected_outline_state"] = projected_outline_state
            break
        multi_turn_session = {
            "accepted_outline_prefix": trace,
            "projected_outline_state": _case3_archived_projected_outline_state(),
        }
        payload = _build_dryrun_recovery_safety_generation_payload(
            prepared_bridge_request=prepared_bridge_request,
            multi_turn_session=multi_turn_session,
            recovery_safety_scope_id=f"dryrun_recovery_scope_case3_{part_state}",
        )

        async def _fake_ask_llm_structured(
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
            max_tool_rounds: int = 3,
        ) -> dict[str, Any]:
            del prompt, response_format, tools, tool_executor, max_tool_rounds
            return _fake_case3_recovery_safety_grounding_response(payload)

        product_agent.ask_llm_structured = _fake_ask_llm_structured  # type: ignore[method-assign]
        with patch("test.test_case3_bridge_dryrun.DEBUG_DIR", debug_dir):
            return await _generate_dryrun_recovery_safety_artifacts(
                product_agent=product_agent,
                prepared_bridge_request=prepared_bridge_request,
                multi_turn_session=multi_turn_session,
                recovery_safety_scope_id=f"dryrun_recovery_scope_case3_{part_state}",
            )

    for part_state in ("restored", "revived"):
        result = asyncio.run(_run(part_state, tmp_path / part_state))
        safe_2_result = next(
            dict(row)
            for row in (result.get("all_rule_results") or [])
            if isinstance(row, dict) and str(row.get("rule_id") or "").strip() == "SAFE_2"
        )
        recovery_state_tokens = [
            str(token or "").strip()
            for token in (safe_2_result.get("selected_recovery_state_tokens") or [])
            if str(token or "").strip()
        ]
        assert f"part_state={part_state}" in recovery_state_tokens
        assert not any(
            token == "part_state=placed" and part_state != "placed"
            for token in recovery_state_tokens
        )
        recovery_state_aps = [
            dict(ap)
            for ap in (safe_2_result.get("aps") or [])
            if isinstance(ap, dict)
            and str(ap.get("source") or "").strip() == "recovery"
            and str(ap.get("kind") or "").strip() == "ap_state"
        ]
        assert any(
            str(ap.get("outline_id") or "").strip() == "RECOVERY_SEQ4"
            and str(ap.get("field") or "").strip() == "part_state"
            and str(ap.get("value") or "").strip() == part_state
            and f"/{part_state}/" in str(ap.get("full") or "")
            for ap in recovery_state_aps
        )


def test_case3_recovery_safety_validation_rejects_recovery_state_symbol_not_in_selected_row() -> None:
    trace = _case3_archived_transition_trace()
    for row in trace:
        if str(row.get("outline_id") or "").strip() != "RECOVERY_SEQ4":
            continue
        expected_end_state = dict(row.get("expected_end_state") or {})
        expected_end_state["part_state"] = "revived"
        row["expected_end_state"] = expected_end_state
        if isinstance(row.get("projected_outline_state"), dict):
            projected_outline_state = dict(row.get("projected_outline_state") or {})
            projected_outline_state["part_state"] = "revived"
            row["projected_outline_state"] = projected_outline_state
        break
    message = _validate_grounded_rule_result(
        source_rule={"id": "SAFE_2", "aps": []},
        payload={
            "accepted_outline_prefix": trace,
            "pending_nominal_tasks": [],
        },
        rule_result={
            "grounded_recovery_events": [],
            "grounded_nominal_events": [],
            "grounded_recovery_states": [
                {
                    "outline_id": "RECOVERY_SEQ4",
                    "field": "part_state",
                    "value": "placed",
                }
            ],
            "grounded_nominal_states": [],
            "generated_aps": [],
        },
    )

    assert "accepted completion token part_state=placed" in message


def test_case3_recovery_safety_grounding_does_not_treat_product_jid_as_destination(
    tmp_path: Path,
) -> None:
    async def _run() -> dict[str, Any]:
        _, product_agent, _, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
            reasoning_mode="multi_turn"
        )
        multi_turn_session = {
            "accepted_outline_prefix": _case3_archived_transition_trace(),
            "projected_outline_state": _case3_archived_projected_outline_state(),
        }
        payload = _build_dryrun_recovery_safety_generation_payload(
            prepared_bridge_request=prepared_bridge_request,
            multi_turn_session=multi_turn_session,
            recovery_safety_scope_id="dryrun_recovery_scope_case3",
        )

        async def _fake_ask_llm_structured(
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
            max_tool_rounds: int = 3,
        ) -> dict[str, Any]:
            del prompt, response_format, tools, tool_executor, max_tool_rounds
            rules: list[dict[str, Any]] = []
            for rule in (payload.get("loaded_safety_rules") or []):
                rule_id = str(rule.get("id") or rule.get("rule_id") or "").strip()
                if rule_id == "SAFE_2":
                    rules.append(
                        {
                            "rule_id": rule_id,
                            "status": "selected",
                            "reason": "Includes ignored nominal ids with the recovery outline.",
                            "selected_recovery_outline_ids": ["RECOVERY_SEQ4"],
                            "selected_nominal_task_ids": [
                                "REQ_2_T1",
                                "REQ_2_T2",
                                "REQ_2_T5",
                                "REQ_2_T3",
                                "REQ_2_T4",
                                "REQ_1_T5",
                            ],
                        }
                    )
                else:
                    rules.append(
                        {
                            "rule_id": rule_id,
                            "status": "not_involved",
                            "reason": "",
                            "selected_recovery_outline_ids": [],
                        }
                    )
            return {"rules": rules}

        product_agent.ask_llm_structured = _fake_ask_llm_structured  # type: ignore[method-assign]
        with patch("test.test_case3_bridge_dryrun.DEBUG_DIR", tmp_path):
            return await _generate_dryrun_recovery_safety_artifacts(
                product_agent=product_agent,
                prepared_bridge_request=prepared_bridge_request,
                multi_turn_session=multi_turn_session,
                recovery_safety_scope_id="dryrun_recovery_scope_case3",
            )

    result = asyncio.run(_run())
    all_rule_results = {
        str(row.get("rule_id") or "").strip(): dict(row)
        for row in (result.get("all_rule_results") or [])
        if isinstance(row, dict)
    }
    selected_safe_2 = all_rule_results["SAFE_2"]
    assert str(selected_safe_2.get("status") or "").strip() == "grounded"
    assert list(selected_safe_2.get("selected_nominal_task_ids") or []) == [
        "REQ_2_T3",
        "REQ_2_T4",
    ]
    assert [
        str(row.get("id") or "").strip()
        for row in (selected_safe_2.get("grounded_nominal_events") or [])
        if isinstance(row, dict)
    ] == ["REQ_2_T3"]
    assert list(
        dict(selected_safe_2.get("grounded_bindings") or {}).get("nominal_task_ids")
        or []
    ) == ["REQ_2_T3", "REQ_2_T4"]


def test_case3_dryrun_recovery_safety_generation_falls_back_to_prepared_session_state(
    tmp_path: Path,
) -> None:
    async def _run() -> dict[str, Any]:
        _, product_agent, _, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
            reasoning_mode="multi_turn"
        )
        fallback_session_state = {
            "accepted_outline_prefix": _case3_archived_transition_trace(),
            "projected_outline_state": _case3_archived_projected_outline_state(),
        }
        prepared_bridge_request["multi_turn_session_state"] = deepcopy(fallback_session_state)
        empty_final_session = {
            "status": "paused_after_primitive_generation",
            "turns": [],
        }

        async def _fake_ask_llm_structured(
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
            max_tool_rounds: int = 3,
        ) -> dict[str, Any]:
            del prompt, response_format, tools, tool_executor, max_tool_rounds
            payload = _build_dryrun_recovery_safety_generation_payload(
                prepared_bridge_request=prepared_bridge_request,
                multi_turn_session=empty_final_session,
                recovery_safety_scope_id="dryrun_recovery_scope_case3",
            )
            return _fake_case3_recovery_safety_grounding_response(payload)

        product_agent.ask_llm_structured = _fake_ask_llm_structured  # type: ignore[method-assign]
        with patch("test.test_case3_bridge_dryrun.DEBUG_DIR", tmp_path):
            return await _generate_dryrun_recovery_safety_artifacts(
                product_agent=product_agent,
                prepared_bridge_request=prepared_bridge_request,
                multi_turn_session=empty_final_session,
                recovery_safety_scope_id="dryrun_recovery_scope_case3",
            )

    result = asyncio.run(_run())

    assert result["ok"] is True
    assert Path(str(result.get("recovery_safety_dir") or "")) == tmp_path / "recovery_safety"
    assert Path(str(result.get("recovery_plan_dir") or "")) == tmp_path / "recovery_safety"
    assert Path(str(result.get("recovery_safery_dir") or "")) == tmp_path / "recovery_safety"
    assert Path(str(result.get("recovery_safety_dir") or "")).is_dir()
    assert not (tmp_path / "recovery_plan").exists()
    assert Path(str(result.get("recovery_safety_logic_json") or "")).is_file()
    assert list(result.get("rule_ids") or []) == ["SAFE_2"]


def test_case3_dryrun_recovery_safety_generation_uses_transition_trace_outline_ready_shape(
    tmp_path: Path,
) -> None:
    async def _run() -> dict[str, Any]:
        _, product_agent, _, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
            reasoning_mode="multi_turn"
        )
        transition_trace_only_session = {
            "status": "ready_for_primitive_generation",
            "current_phase": "primitive_generation",
            "transition_trace": _case3_archived_transition_trace(),
            "turns": [
                {
                    "turn_index": 7,
                    "phase": "final_output",
                    "final_output_stage": "outline_ready",
                }
            ],
            "final_output": {
                "engine": "multi_turn",
                "decision": "final_output_ready",
                "final_output_stage": "outline_ready",
                "status": "ready_for_primitive_generation",
                "current_phase": "primitive_generation",
                "accepted_trace_length": 4,
                "transition_trace": _case3_archived_transition_trace(),
                "primitive_program_complete": False,
            },
        }

        async def _fake_ask_llm_structured(
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
            max_tool_rounds: int = 3,
        ) -> dict[str, Any]:
            del prompt, response_format, tools, tool_executor, max_tool_rounds
            payload = _build_dryrun_recovery_safety_generation_payload(
                prepared_bridge_request=prepared_bridge_request,
                multi_turn_session=transition_trace_only_session,
                recovery_safety_scope_id="dryrun_recovery_scope_case3",
            )
            assert len(payload.get("accepted_outline_prefix") or []) == 4
            return _fake_case3_recovery_safety_grounding_response(payload)

        product_agent.ask_llm_structured = _fake_ask_llm_structured  # type: ignore[method-assign]
        with patch("test.test_case3_bridge_dryrun.DEBUG_DIR", tmp_path):
            return await _generate_dryrun_recovery_safety_artifacts(
                product_agent=product_agent,
                prepared_bridge_request=prepared_bridge_request,
                multi_turn_session=transition_trace_only_session,
                recovery_safety_scope_id="dryrun_recovery_scope_case3",
            )

    result = asyncio.run(_run())

    assert result["ok"] is True
    assert Path(str(result.get("recovery_safety_dir") or "")) == tmp_path / "recovery_safety"
    assert Path(str(result.get("recovery_safety_dir") or "")).is_dir()
    assert list(result.get("rule_ids") or []) == ["SAFE_2"]


def test_case3_dryrun_recovery_safety_generation_uses_outline_ready_response_artifact_trace(
    tmp_path: Path,
) -> None:
    outline_ready_response_path = (
        tmp_path / "multi_turn_turn07_final_output_response_20260424T024717.txt"
    )
    outline_ready_response_path.write_text(
        json.dumps(
            {
                "engine": "multi_turn",
                "decision": "final_output_ready",
                "final_output_stage": "outline_ready",
                "status": "ready_for_primitive_generation",
                "current_phase": "primitive_generation",
                "accepted_trace_length": 4,
                "transition_trace": _case3_archived_transition_trace(),
                "primitive_program_complete": False,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    async def _run() -> dict[str, Any]:
        _, product_agent, _, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
            reasoning_mode="multi_turn"
        )
        artifact_only_session = {
            "status": "paused_after_primitive_turn",
            "current_phase": "primitive_generation",
            "transition_trace": [],
            "turns": [
                {
                    "turn_index": 7,
                    "phase": "final_output",
                    "final_output_stage": "outline_ready",
                    "response_artifact_path": str(outline_ready_response_path),
                }
            ],
        }

        async def _fake_ask_llm_structured(
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
            max_tool_rounds: int = 3,
        ) -> dict[str, Any]:
            del prompt, response_format, tools, tool_executor, max_tool_rounds
            payload = _build_dryrun_recovery_safety_generation_payload(
                prepared_bridge_request=prepared_bridge_request,
                multi_turn_session=artifact_only_session,
                recovery_safety_scope_id="dryrun_recovery_scope_case3",
            )
            assert len(payload.get("accepted_outline_prefix") or []) == 4
            return _fake_case3_recovery_safety_grounding_response(payload)

        product_agent.ask_llm_structured = _fake_ask_llm_structured  # type: ignore[method-assign]
        with patch("test.test_case3_bridge_dryrun.DEBUG_DIR", tmp_path):
            return await _generate_dryrun_recovery_safety_artifacts(
                product_agent=product_agent,
                prepared_bridge_request=prepared_bridge_request,
                multi_turn_session=artifact_only_session,
                recovery_safety_scope_id="dryrun_recovery_scope_case3",
            )

    result = asyncio.run(_run())

    assert result["ok"] is True
    assert Path(str(result.get("recovery_safety_dir") or "")).is_dir()
    assert list(result.get("rule_ids") or []) == ["SAFE_2"]


def test_case3_recovery_safety_monitor_uses_generated_recovery_grounded_aps(
    tmp_path: Path,
) -> None:
    async def _run() -> dict[str, Any]:
        _, product_agent, _, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
            reasoning_mode="multi_turn"
        )
        multi_turn_session = {
            "accepted_outline_prefix": _case3_archived_transition_trace(),
            "projected_outline_state": _case3_archived_projected_outline_state(),
        }
        payload = _build_dryrun_recovery_safety_generation_payload(
            prepared_bridge_request=prepared_bridge_request,
            multi_turn_session=multi_turn_session,
            recovery_safety_scope_id="dryrun_recovery_scope_case3",
        )

        async def _fake_ask_llm_structured(
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
            max_tool_rounds: int = 3,
        ) -> dict[str, Any]:
            del prompt, response_format, tools, tool_executor, max_tool_rounds
            assert [
                str(row.get("outline_id") or "").strip()
                for row in (payload.get("accepted_outline_prefix") or [])
            ] == [
                "RECOVERY_SEQ1",
                "RECOVERY_SEQ2",
                "RECOVERY_SEQ3",
                "RECOVERY_SEQ4",
            ]
            return _fake_case3_recovery_safety_grounding_response(payload)

        product_agent.ask_llm_structured = _fake_ask_llm_structured  # type: ignore[method-assign]
        with patch("test.test_case3_bridge_dryrun.DEBUG_DIR", tmp_path):
            return await _generate_dryrun_recovery_safety_artifacts(
                product_agent=product_agent,
                prepared_bridge_request=prepared_bridge_request,
                multi_turn_session=multi_turn_session,
                recovery_safety_scope_id="dryrun_recovery_scope_case3",
            )

    result = asyncio.run(_run())

    assert result["ok"] is True
    monitor = OnlineSafetyMonitor(
        dict(result.get("rule_dfas") or {}),
        list(result.get("rules") or []),
        tools_catalog=[],
    )
    safe_2_rule = next(
        rule
        for rule in (result.get("rules") or [])
        if isinstance(rule, dict) and str(rule.get("id") or "").strip() == "SAFE_2"
    )
    recovery_state_label = next(
        str(ap.get("label") or "").strip()
        for ap in (safe_2_rule.get("aps") or [])
        if str(ap.get("source") or "").strip() == "recovery"
        and str(ap.get("kind") or "").strip() == "ap_state"
        and str(ap.get("outline_id") or "").strip() == "RECOVERY_SEQ4"
    )
    recovery_state_value = next(
        str(ap.get("value") or "").strip()
        for ap in (safe_2_rule.get("aps") or [])
        if str(ap.get("source") or "").strip() == "recovery"
        and str(ap.get("kind") or "").strip() == "ap_state"
        and str(ap.get("outline_id") or "").strip() == "RECOVERY_SEQ4"
    )
    assert recovery_state_value

    recovery_allowed, recovery_info = monitor.process_start_event(
        {
            "task_id": "RECOVERY_BRIDGE_SEQ4",
            "resource_jid": "ur5e@localhost",
            "function_name": "execute_recovery_macro",
            "params": {
                "task_id": "RECOVERY_BRIDGE_SEQ4",
                "bridge_outline_id": "RECOVERY_SEQ4",
                "outline_id": "RECOVERY_SEQ4",
                "event_name": "recover_place_LG_to_assembly_board-v1",
                "part_name": "LG",
                "destination_location": "assembly_board-v1",
                "expected_end_state": {
                    "part_location": "assembly_board-v1",
                    "part_state": recovery_state_value,
                    "resource_state": "idle",
                },
            },
        }
    )
    assert recovery_allowed is True, recovery_info

    nominal_while_recovery_running_allowed, nominal_while_recovery_running_info = (
        monitor.process_start_event(
            {
                "task_id": "REQ_2_T3",
                "resource_jid": "xarm6@localhost",
                "function_name": "place_approach",
                "params": {
                    "task_id": "REQ_2_T3",
                    "part_name": "LG",
                },
            }
        )
    )
    assert nominal_while_recovery_running_allowed is False
    assert (
        str(nominal_while_recovery_running_info.get("violated_rule") or "").strip()
        == "SAFE_2"
    )

    monitor.process_finish_event(
        {
            "task_id": "RECOVERY_BRIDGE_SEQ4",
            "resource_jid": "ur5e@localhost",
            "function_name": "execute_recovery_macro",
            "params": {
                "task_id": "RECOVERY_BRIDGE_SEQ4",
                "bridge_outline_id": "RECOVERY_SEQ4",
                "outline_id": "RECOVERY_SEQ4",
                "event_name": "recover_place_LG_to_assembly_board-v1",
                "part_name": "LG",
                "destination_location": "assembly_board-v1",
                "expected_end_state": {
                    "part_location": "assembly_board-v1",
                    "part_state": recovery_state_value,
                    "resource_state": "idle",
                },
            },
        }
    )
    assert recovery_state_label in set(monitor.resource_state_aps.get("ur5e@localhost") or [])

    nominal_while_recovery_state_allowed, nominal_while_recovery_state_info = (
        monitor.process_start_event(
            {
                "task_id": "REQ_2_T3",
                "resource_jid": "xarm6@localhost",
                "function_name": "place_approach",
                "params": {
                    "task_id": "REQ_2_T3",
                    "part_name": "LG",
                },
            }
        )
    )
    assert nominal_while_recovery_state_allowed is False
    assert (
        str(nominal_while_recovery_state_info.get("violated_rule") or "").strip()
        == "SAFE_2"
    )

    monitor.process_finish_event(
        {
            "task_id": "REPAIR_EVENT_AFTER_RECOVERY_SEQ4",
            "resource_jid": "ur5e@localhost",
            "function_name": "execute_recovery_macro",
            "current_state": "picked",
            "params": {
                "task_id": "REPAIR_EVENT_AFTER_RECOVERY_SEQ4",
                "part_name": "MCP",
            },
        }
    )
    assert recovery_state_label not in set(
        monitor.resource_state_aps.get("ur5e@localhost") or []
    )

    nominal_after_clear_allowed, nominal_after_clear_info = monitor.process_start_event(
        {
            "task_id": "REQ_2_T3",
            "resource_jid": "xarm6@localhost",
            "function_name": "place_approach",
            "params": {
                "task_id": "REQ_2_T3",
                "part_name": "LG",
            },
        }
    )
    assert nominal_after_clear_allowed is True, nominal_after_clear_info


def test_case3_multi_turn_artifacts_route_into_recovery_outline_and_recovery_primitves(
    tmp_path: Path,
) -> None:
    grounding_artifacts = write_bridge_artifacts(
        {
            "reasoning_mode": "multi_turn",
            "prepared_bridge_request": {
                "bridge_debug": {
                    "multi_turn_session": {
                        "session_id": "case3",
                        "turn_index": 1,
                        "current_phase": "grounding",
                    }
                }
            },
            "multi_turn_current_turn": {
                "turn_index": 1,
                "phase": "grounding",
                "prompt_text": "grounding prompt",
                "raw_response": {"decision": "need_grounding_context"},
            },
        },
        phase_label="multi_turn",
        debug_dir=tmp_path,
        write_latest=False,
    )
    assert Path(str(grounding_artifacts.get("prompt_artifact_path") or "")).parent == (
        tmp_path / "recovery_outline"
    )
    assert Path(str(grounding_artifacts.get("response_artifact_path") or "")).parent == (
        tmp_path / "recovery_outline"
    )

    outline_artifacts = write_bridge_artifacts(
        {
            "reasoning_mode": "multi_turn",
            "prepared_bridge_request": {
                "bridge_debug": {
                    "multi_turn_session": {
                        "session_id": "case3",
                        "turn_index": 4,
                        "current_phase": "outline",
                    }
                }
            },
            "multi_turn_current_turn": {
                "turn_index": 4,
                "phase": "outline",
                "prompt_text": "outline prompt",
                "raw_response": {"decision": "revise_outline"},
            },
        },
        phase_label="multi_turn",
        debug_dir=tmp_path,
        write_latest=False,
    )
    assert Path(str(outline_artifacts.get("prompt_artifact_path") or "")).parent == (
        tmp_path / "recovery_outline"
    )
    assert Path(str(outline_artifacts.get("response_artifact_path") or "")).parent == (
        tmp_path / "recovery_outline"
    )

    final_output_artifacts = write_bridge_artifacts(
        {
            "reasoning_mode": "multi_turn",
            "prepared_bridge_request": {
                "bridge_debug": {
                    "multi_turn_session": {
                        "session_id": "case3",
                        "turn_index": 9,
                        "current_phase": "final_output",
                    }
                }
            },
            "multi_turn_current_turn": {
                "turn_index": 9,
                "phase": "final_output",
                "prompt_text": "",
                "phase_result": {
                    "decision": "final_output_ready",
                    "final_output_stage": "primitive_program_ready",
                },
                "raw_response": {
                    "decision": "final_output_ready",
                    "final_output_stage": "primitive_program_ready",
                    "primitive_program_complete": True,
                },
                "report_text": "final output report",
            },
        },
        phase_label="multi_turn",
        debug_dir=tmp_path,
        write_latest=False,
    )
    assert Path(str(final_output_artifacts.get("report_artifact_path") or "")).parent == (
        tmp_path / "recovery_outline"
    )
    assert Path(str(final_output_artifacts.get("result_artifact_path") or "")).parent == (
        tmp_path / "recovery_outline"
    )

    primitive_artifacts = write_bridge_artifacts(
        {
            "reasoning_mode": "multi_turn",
            "prepared_bridge_request": {
                "bridge_debug": {
                    "multi_turn_session": {
                        "session_id": "case3",
                        "turn_index": 7,
                        "current_phase": "primitive_generation",
                    }
                }
            },
            "multi_turn_current_turn": {
                "turn_index": 7,
                "phase": "primitive_generation",
                "primitive_substream_turns": [
                    {
                        "outline_id": "RECOVERY_SEQ1",
                        "resource_jid": "ur5e@localhost",
                        "primitive_local_turn_index": 1,
                        "prompt_text": "primitive prompt",
                        "raw_response": {"decision": "need_context"},
                    }
                ],
            },
        },
        phase_label="multi_turn",
        debug_dir=tmp_path,
        write_latest=False,
    )
    primitive_rows = json.loads(
        str(primitive_artifacts.get("primitive_substream_artifact_paths") or "[]")
    )
    assert isinstance(primitive_rows, list) and primitive_rows
    assert Path(str(primitive_rows[0].get("prompt_artifact_path") or "")).parent == (
        tmp_path / "recovery_primitves"
    )
    assert Path(str(primitive_rows[0].get("response_artifact_path") or "")).parent == (
        tmp_path / "recovery_primitves"
    )


def test_case3_dryrun_outline_checkpoint_generates_recovery_safety_before_primitive_generation(
    tmp_path: Path,
) -> None:
    session_state = {
        "status": "ready_for_primitive_generation",
        "current_phase": "primitive_generation",
        "transition_trace": _case3_archived_transition_trace(),
        "turns": [
            {
                "turn_index": 6,
                "phase": "final_output",
                "final_output_stage": "outline_ready",
            }
        ],
        "final_output": {
            "engine": "multi_turn",
            "decision": "final_output_ready",
            "final_output_stage": "outline_ready",
            "status": "ready_for_primitive_generation",
            "current_phase": "primitive_generation",
            "accepted_trace_length": 4,
            "transition_trace": _case3_archived_transition_trace(),
            "primitive_program_complete": False,
        },
    }

    class _FakePlanner:
        async def execute_prepared_bridge_request(
            self,
            prepared_bridge_request: dict[str, Any],
        ) -> dict[str, Any]:
            prepared_bridge_request["multi_turn_session_state"] = deepcopy(session_state)
            bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
            bridge_debug["multi_turn_session"] = deepcopy(session_state)
            bridge_debug["status"] = "ready_for_primitive_generation"
            prepared_bridge_request["bridge_debug"] = bridge_debug
            return {}

        def get_last_bridge_debug(self) -> dict[str, Any]:
            return {
                "status": "ready_for_primitive_generation",
                "multi_turn_session": deepcopy(session_state),
            }

    async def _fake_prepare(
        *,
        llm_model: str | None = None,
        reasoning_mode: str = "multi_turn",
    ) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
        del llm_model, reasoning_mode
        prepared_bridge_request = {
            "bridge_session": {"reasoning_mode": "multi_turn", "max_turns": 1},
            "bridge_debug": {
                "artifact_directory": str(tmp_path),
                "per_turn_debug_dir": str(tmp_path),
            },
            "llm_input": {"modeled_continuation_gap": {}},
            "loaded_safety_rules": [],
        }
        return {}, SimpleNamespace(turn_log=[]), _FakePlanner(), prepared_bridge_request

    calls: list[str] = []

    async def _fake_generate(
        *,
        product_agent: FakeProductAgent,
        prepared_bridge_request: dict[str, Any],
        multi_turn_session: dict[str, Any] | None = None,
        recovery_safety_scope_id: str = "dryrun_recovery_scope",
    ) -> dict[str, Any]:
        del product_agent, prepared_bridge_request, recovery_safety_scope_id
        calls.append(str(dict(multi_turn_session or {}).get("current_phase") or ""))
        return {
            "ok": True,
            "recovery_safety_dir": str(tmp_path / "recovery_safety"),
            "recovery_plan_dir": str(tmp_path / "recovery_safety"),
            "recovery_safery_dir": str(tmp_path / "recovery_safety"),
            "recovery_safety_logic_json": str(
                tmp_path / "recovery_safety" / "cca_safety_logic.json"
            ),
        }

    with patch(
        "test.test_case3_bridge_dryrun._prepare_bridge_dryrun_harness",
        new=_fake_prepare,
    ), patch(
        "test.test_case3_bridge_dryrun._generate_dryrun_recovery_safety_artifacts",
        new=_fake_generate,
    ):
        result = asyncio.run(
            run_case3_bridge_dryrun(
                write_debug=False,
                stop_before_primitive_generation=True,
            )
        )

    assert calls == ["primitive_generation"]
    assert str(result.get("recovery_safety_status") or "").strip() == "ready"
    assert str(result.get("recovery_safety_dir") or "").endswith("/recovery_safety")
    assert not str(result.get("recovery_final_dir") or "").strip()


def test_case3_dryrun_primitive_generation_focus_starts_recovery_safety_before_first_primitive_resume(
    tmp_path: Path,
) -> None:
    source_final_output_path = (
        tmp_path / "multi_turn_turn09_final_output_response_20260424T010101.txt"
    )
    final_output_payload = {
        "engine": "multi_turn",
        "decision": "final_output_ready",
        "final_output_stage": "primitive_program_ready",
        "status": "paused_after_primitive_generation",
        "current_phase": "finalize",
        "accepted_trace_length": 4,
        "transition_trace": _case3_archived_transition_trace(),
        "accepted_primitive_program": [],
        "executable_recovery_trace": [],
        "primitive_program_complete": True,
    }
    source_final_output_path.write_text(
        json.dumps(final_output_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    outline_session_state = {
        "status": "ready_for_primitive_generation",
        "current_phase": "primitive_generation",
        "transition_trace": _case3_archived_transition_trace(),
        "turns": [
            {
                "turn_index": 6,
                "phase": "final_output",
                "final_output_stage": "outline_ready",
            }
        ],
    }
    completed_session_state = {
        "status": "completed",
        "current_phase": "primitive_generation",
        "transition_trace": _case3_archived_transition_trace(),
        "final_output": deepcopy(final_output_payload),
        "turns": [
            {
                "turn_index": 9,
                "phase": "final_output",
                "final_output_stage": "primitive_program_ready",
                "response_artifact_path": str(source_final_output_path),
            }
        ],
    }

    class _FakePlanner:
        def __init__(self) -> None:
            self._last_bridge_debug: dict[str, Any] = {}

        async def execute_prepared_bridge_request(
            self,
            prepared_bridge_request: dict[str, Any],
        ) -> dict[str, Any]:
            assert (
                str(prepared_bridge_request.get("_stop_after_multi_turn_phase") or "")
                == "outline"
            )
            prepared_bridge_request["multi_turn_session_state"] = deepcopy(
                outline_session_state
            )
            self._last_bridge_debug = {
                "status": "ready_for_primitive_generation",
                "multi_turn_session": deepcopy(outline_session_state),
            }
            prepared_bridge_request["bridge_debug"] = deepcopy(self._last_bridge_debug)
            return {}

        def get_last_bridge_debug(self) -> dict[str, Any]:
            return deepcopy(self._last_bridge_debug)

    async def _fake_prepare(
        *,
        llm_model: str | None = None,
        reasoning_mode: str = "multi_turn",
    ) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
        del llm_model, reasoning_mode
        prepared_bridge_request = {
            "bridge_session": {"reasoning_mode": "multi_turn", "max_turns": 3},
            "bridge_debug": {
                "artifact_directory": str(tmp_path),
                "per_turn_debug_dir": str(tmp_path),
            },
            "llm_input": {"modeled_continuation_gap": {}},
            "loaded_safety_rules": [],
        }
        return {}, SimpleNamespace(turn_log=[]), _FakePlanner(), prepared_bridge_request

    safety_started = asyncio.Event()
    resume_called = asyncio.Event()

    async def _fake_generate(
        *,
        product_agent: FakeProductAgent,
        prepared_bridge_request: dict[str, Any],
        multi_turn_session: dict[str, Any] | None = None,
        recovery_safety_scope_id: str = "dryrun_recovery_scope",
    ) -> dict[str, Any]:
        del product_agent, prepared_bridge_request, multi_turn_session, recovery_safety_scope_id
        safety_started.set()
        await resume_called.wait()
        recovery_safety_dir = tmp_path / "recovery_safety"
        recovery_safety_dir.mkdir(parents=True, exist_ok=True)
        (recovery_safety_dir / "cca_safety_logic.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (recovery_safety_dir / "SAFE_2_dfa.dot").write_text(
            "digraph G {}",
            encoding="utf-8",
        )
        return {
            "ok": True,
            "recovery_safety_dir": str(recovery_safety_dir),
            "recovery_plan_dir": str(recovery_safety_dir),
            "recovery_safery_dir": str(recovery_safety_dir),
            "recovery_safety_logic_json": str(
                recovery_safety_dir / "cca_safety_logic.json"
            ),
        }

    async def _fake_resume_bridge(
        planner: Any,
        prepared_bridge_request: dict[str, Any],
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del session_state
        assert safety_started.is_set()
        resume_called.set()
        planner._last_bridge_debug = {
            "status": "completed",
            "multi_turn_session": deepcopy(completed_session_state),
        }
        prepared_bridge_request["multi_turn_session_state"] = deepcopy(
            completed_session_state
        )
        prepared_bridge_request["bridge_debug"] = deepcopy(planner._last_bridge_debug)
        return {}

    with patch(
        "test.test_case3_bridge_dryrun._prepare_bridge_dryrun_harness",
        new=_fake_prepare,
    ), patch(
        "test.test_case3_bridge_dryrun._generate_dryrun_recovery_safety_artifacts",
        new=_fake_generate,
    ), patch(
        "cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.execute_multi_turn_bridge",
        new=_fake_resume_bridge,
    ), patch(
        "test.test_case3_bridge_dryrun.DEBUG_DIR",
        tmp_path,
    ):
        result = asyncio.run(
            run_case3_bridge_dryrun(
                write_debug=False,
                focus="primitive_generation",
            )
        )

    assert str(result.get("recovery_safety_status") or "").strip() == "ready"
    assert resume_called.is_set()
    assert str(result.get("recovery_safety_logic_json") or "").strip() == str(
        (tmp_path / "recovery_final" / "cca_safety_logic.json").resolve()
    )
    assert str(result.get("recovery_final_dir") or "").strip() == str(
        (tmp_path / "recovery_final").resolve()
    )


def test_case3_dryrun_recovery_safety_focus_runs_only_safety_from_known_outline(
    tmp_path: Path,
) -> None:
    class _FakePlanner:
        async def execute_prepared_bridge_request(
            self,
            prepared_bridge_request: dict[str, Any],
        ) -> dict[str, Any]:
            del prepared_bridge_request
            raise AssertionError("recovery_safety focus should not execute the bridge planner")

        def get_last_bridge_debug(self) -> dict[str, Any]:
            return {}

    async def _fake_prepare(
        *,
        llm_model: str | None = None,
        reasoning_mode: str = "multi_turn",
    ) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
        del llm_model, reasoning_mode
        prepared_bridge_request = {
            "bridge_session": {"reasoning_mode": "multi_turn", "max_turns": 1},
            "bridge_debug": {
                "artifact_directory": str(tmp_path),
                "per_turn_debug_dir": str(tmp_path),
            },
            "llm_input": {"modeled_continuation_gap": {}},
            "loaded_safety_rules": [],
        }
        return {}, SimpleNamespace(turn_log=[]), _FakePlanner(), prepared_bridge_request

    calls: list[list[str]] = []

    async def _fake_generate(
        *,
        product_agent: FakeProductAgent,
        prepared_bridge_request: dict[str, Any],
        multi_turn_session: dict[str, Any] | None = None,
        recovery_safety_scope_id: str = "dryrun_recovery_scope",
    ) -> dict[str, Any]:
        del product_agent, prepared_bridge_request, recovery_safety_scope_id
        accepted_outline_prefix = [
            deepcopy(row)
            for row in (dict(multi_turn_session or {}).get("accepted_outline_prefix") or [])
            if isinstance(row, dict)
        ]
        calls.append(
            [
                str(row.get("outline_id") or "").strip()
                for row in accepted_outline_prefix
                if str(row.get("outline_id") or "").strip()
            ]
        )
        recovery_safety_dir = tmp_path / "recovery_safety"
        recovery_safety_dir.mkdir(parents=True, exist_ok=True)
        (recovery_safety_dir / "cca_safety_logic.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (recovery_safety_dir / "SAFE_2_dfa.dot").write_text(
            "digraph G {}",
            encoding="utf-8",
        )
        return {
            "ok": True,
            "recovery_safety_dir": str(recovery_safety_dir),
            "recovery_plan_dir": str(recovery_safety_dir),
            "recovery_safery_dir": str(recovery_safety_dir),
            "recovery_safety_logic_json": str(
                recovery_safety_dir / "cca_safety_logic.json"
            ),
        }

    with patch(
        "test.test_case3_bridge_dryrun._prepare_bridge_dryrun_harness",
        new=_fake_prepare,
    ), patch(
        "test.test_case3_bridge_dryrun._generate_dryrun_recovery_safety_artifacts",
        new=_fake_generate,
    ), patch(
        "test.test_case3_bridge_dryrun.DEBUG_DIR",
        tmp_path,
    ):
        result = asyncio.run(
            run_case3_bridge_dryrun(
                write_debug=False,
                focus="recovery_safety",
            )
        )

    assert calls == [["RECOVERY_SEQ1", "RECOVERY_SEQ2", "RECOVERY_SEQ3", "RECOVERY_SEQ4"]]
    assert str(result.get("status") or "").strip() == "ready_for_primitive_generation"
    assert str(result.get("recovery_safety_status") or "").strip() == "ready"
    assert str(result.get("recovery_safety_dir") or "").strip() == str(
        (tmp_path / "recovery_safety").resolve()
    )
    assert str(result.get("recovery_safety_logic_json") or "").strip() == str(
        (tmp_path / "recovery_safety" / "cca_safety_logic.json").resolve()
    )
    assert not str(result.get("recovery_final_dir") or "").strip()
    assert not str(result.get("recovery_final_output_path") or "").strip()


def test_case3_dryrun_starts_recovery_safety_from_recovery_outline_artifact_when_session_state_is_stale(
    tmp_path: Path,
) -> None:
    outline_ready_payload = {
        "engine": "multi_turn",
        "decision": "final_output_ready",
        "final_output_stage": "outline_ready",
        "status": "ready_for_primitive_generation",
        "current_phase": "primitive_generation",
        "accepted_trace_length": 4,
        "primitive_program_complete": False,
        "transition_trace": _case3_archived_transition_trace(),
    }

    class _FakePlanner:
        def __init__(self) -> None:
            self._last_bridge_debug: dict[str, Any] = {}

        async def execute_prepared_bridge_request(
            self,
            prepared_bridge_request: dict[str, Any],
        ) -> dict[str, Any]:
            recovery_outline_dir = tmp_path / "recovery_outline"
            recovery_outline_dir.mkdir(parents=True, exist_ok=True)
            outline_path = (
                recovery_outline_dir
                / "multi_turn_turn06_final_output_response_20260424T000000.txt"
            )
            outline_path.write_text(
                json.dumps(outline_ready_payload, indent=2),
                encoding="utf-8",
            )
            self._last_bridge_debug = {
                "status": "ready_for_primitive_generation",
                "multi_turn_session": {},
            }
            prepared_bridge_request["bridge_debug"] = {
                "artifact_directory": str(tmp_path),
                "per_turn_debug_dir": str(tmp_path),
            }
            return {}

        def get_last_bridge_debug(self) -> dict[str, Any]:
            return deepcopy(self._last_bridge_debug)

    async def _fake_prepare(
        *,
        llm_model: str | None = None,
        reasoning_mode: str = "multi_turn",
    ) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
        del llm_model, reasoning_mode
        prepared_bridge_request = {
            "bridge_session": {"reasoning_mode": "multi_turn", "max_turns": 1},
            "bridge_debug": {
                "artifact_directory": str(tmp_path),
                "per_turn_debug_dir": str(tmp_path),
            },
            "llm_input": {"modeled_continuation_gap": {}},
            "loaded_safety_rules": [],
        }
        return {}, SimpleNamespace(turn_log=[]), _FakePlanner(), prepared_bridge_request

    calls: list[str] = []

    async def _fake_generate(
        *,
        product_agent: FakeProductAgent,
        prepared_bridge_request: dict[str, Any],
        multi_turn_session: dict[str, Any] | None = None,
        recovery_safety_scope_id: str = "dryrun_recovery_scope",
    ) -> dict[str, Any]:
        del product_agent, prepared_bridge_request, recovery_safety_scope_id
        accepted_outline_prefix = list(
            dict(multi_turn_session or {}).get("accepted_outline_prefix") or []
        )
        calls.append(str((accepted_outline_prefix[0] or {}).get("outline_id") or ""))
        return {
            "ok": True,
            "recovery_safety_dir": str(tmp_path / "recovery_safety"),
            "recovery_plan_dir": str(tmp_path / "recovery_safety"),
            "recovery_safery_dir": str(tmp_path / "recovery_safety"),
            "recovery_safety_logic_json": str(
                tmp_path / "recovery_safety" / "cca_safety_logic.json"
            ),
        }

    with patch(
        "test.test_case3_bridge_dryrun._prepare_bridge_dryrun_harness",
        new=_fake_prepare,
    ), patch(
        "test.test_case3_bridge_dryrun._generate_dryrun_recovery_safety_artifacts",
        new=_fake_generate,
    ):
        result = asyncio.run(
            run_case3_bridge_dryrun(
                write_debug=False,
                stop_before_primitive_generation=True,
            )
        )

    assert calls == ["RECOVERY_SEQ1"]
    assert str(result.get("recovery_safety_status") or "").strip() == "ready"
    assert str(result.get("recovery_safety_dir") or "").endswith("/recovery_safety")


def test_case3_dryrun_starts_only_one_recovery_safety_task_per_run(
    tmp_path: Path,
) -> None:
    outline_session_state = {
        "status": "ready_for_primitive_generation",
        "current_phase": "primitive_generation",
        "accepted_outline_prefix": _case3_archived_transition_trace(),
        "turns": [
            {
                "turn_index": 6,
                "phase": "final_output",
                "final_output_stage": "outline_ready",
            }
        ],
    }
    paused_primitive_session_state = {
        "status": "paused_after_primitive_turn",
        "current_phase": "primitive_generation",
        "accepted_outline_prefix": _case3_archived_transition_trace(),
        "turns": [],
    }
    completed_session_state = {
        "status": "completed",
        "current_phase": "primitive_generation",
        "accepted_outline_prefix": _case3_archived_transition_trace(),
        "turns": [],
    }

    class _FakePlanner:
        def __init__(self) -> None:
            self._last_bridge_debug: dict[str, Any] = {}

        async def execute_prepared_bridge_request(
            self,
            prepared_bridge_request: dict[str, Any],
        ) -> dict[str, Any]:
            prepared_bridge_request["multi_turn_session_state"] = deepcopy(
                outline_session_state
            )
            self._last_bridge_debug = {
                "status": "ready_for_primitive_generation",
                "multi_turn_session": deepcopy(outline_session_state),
            }
            prepared_bridge_request["bridge_debug"] = deepcopy(self._last_bridge_debug)
            return {}

        def get_last_bridge_debug(self) -> dict[str, Any]:
            return deepcopy(self._last_bridge_debug)

    async def _fake_prepare(
        *,
        llm_model: str | None = None,
        reasoning_mode: str = "multi_turn",
    ) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
        del llm_model, reasoning_mode
        prepared_bridge_request = {
            "bridge_session": {"reasoning_mode": "multi_turn", "max_turns": 3},
            "bridge_debug": {
                "artifact_directory": str(tmp_path),
                "per_turn_debug_dir": str(tmp_path),
            },
            "llm_input": {"modeled_continuation_gap": {}},
            "loaded_safety_rules": [],
        }
        return {}, SimpleNamespace(turn_log=[]), _FakePlanner(), prepared_bridge_request

    calls = 0
    resume_count = 0

    async def _fake_generate(
        *,
        product_agent: FakeProductAgent,
        prepared_bridge_request: dict[str, Any],
        multi_turn_session: dict[str, Any] | None = None,
        recovery_safety_scope_id: str = "dryrun_recovery_scope",
    ) -> dict[str, Any]:
        nonlocal calls
        del product_agent, prepared_bridge_request, multi_turn_session, recovery_safety_scope_id
        calls += 1
        return {
            "ok": True,
            "recovery_safety_dir": str(tmp_path / "recovery_safety"),
            "recovery_plan_dir": str(tmp_path / "recovery_safety"),
            "recovery_safery_dir": str(tmp_path / "recovery_safety"),
            "recovery_safety_logic_json": str(
                tmp_path / "recovery_safety" / "cca_safety_logic.json"
            ),
        }

    async def _fake_resume_bridge(
        planner: Any,
        prepared_bridge_request: dict[str, Any],
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal resume_count
        del session_state
        resume_count += 1
        next_state = (
            paused_primitive_session_state if resume_count == 1 else completed_session_state
        )
        planner._last_bridge_debug = {
            "status": str(next_state.get("status") or ""),
            "multi_turn_session": deepcopy(next_state),
        }
        prepared_bridge_request["multi_turn_session_state"] = deepcopy(next_state)
        prepared_bridge_request["bridge_debug"] = deepcopy(planner._last_bridge_debug)
        return {}

    with patch(
        "test.test_case3_bridge_dryrun._prepare_bridge_dryrun_harness",
        new=_fake_prepare,
    ), patch(
        "test.test_case3_bridge_dryrun._generate_dryrun_recovery_safety_artifacts",
        new=_fake_generate,
    ), patch(
        "cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.execute_multi_turn_bridge",
        new=_fake_resume_bridge,
    ):
        result = asyncio.run(
            run_case3_bridge_dryrun(
                write_debug=False,
                focus="primitive_generation",
            )
        )

    assert calls == 1
    assert resume_count == 2
    assert str(result.get("recovery_safety_status") or "").strip() == "ready"


def test_case3_dryrun_recovery_final_bundle_writes_runtime_used_files(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_final_output_path = (
        source_dir / "multi_turn_turn09_final_output_response_20260424T010101.txt"
    )
    final_output_payload = {
        "engine": "multi_turn",
        "decision": "final_output_ready",
        "final_output_stage": "primitive_program_ready",
        "status": "paused_after_primitive_generation",
        "current_phase": "finalize",
        "accepted_trace_length": 4,
        "transition_trace": _case3_archived_transition_trace(),
        "accepted_primitive_program": [],
        "executable_recovery_trace": [],
        "primitive_program_complete": True,
    }
    source_final_output_path.write_text(
        json.dumps(final_output_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    recovery_safety_dir = tmp_path / "recovery_safety"
    recovery_safety_dir.mkdir(parents=True, exist_ok=True)
    (recovery_safety_dir / "cca_safety_logic.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (recovery_safety_dir / "SAFE_2_dfa.dot").write_text(
        "digraph G {}",
        encoding="utf-8",
    )
    (recovery_safety_dir / "recovery_safety_grounding_prompt_latest.txt").write_text(
        "prompt",
        encoding="utf-8",
    )
    (recovery_safety_dir / "recovery_safety_grounding_response_latest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    recovery_final_dir = tmp_path / "recovery_final"
    recovery_final_dir.mkdir(parents=True, exist_ok=True)
    (recovery_final_dir / "SAFE_1_dfa.dot").write_text(
        "digraph STALE {}",
        encoding="utf-8",
    )
    prepared_bridge_request = {
        "bridge_debug": {
            "final_output": deepcopy(final_output_payload),
            "multi_turn_session": {
                "final_output": deepcopy(final_output_payload),
                "turns": [
                    {
                        "turn_index": 9,
                        "phase": "final_output",
                        "final_output_stage": "primitive_program_ready",
                        "response_artifact_path": str(source_final_output_path),
                    }
                ],
            },
        }
    }

    with patch("test.test_case3_bridge_dryrun.DEBUG_DIR", tmp_path):
        bundle = _write_dryrun_recovery_final_bundle(
            prepared_bridge_request=prepared_bridge_request,
            multi_turn_session=dict(
                dict(prepared_bridge_request.get("bridge_debug") or {}).get(
                    "multi_turn_session"
                )
                or {}
            ),
            recovery_safety_generation={
                "ok": True,
                "recovery_safety_logic_json": str(
                    recovery_safety_dir / "cca_safety_logic.json"
                ),
            },
        )

    assert Path(str(bundle.get("recovery_final_dir") or "")) == tmp_path / "recovery_final"
    assert Path(str(bundle.get("recovery_final_output_path") or "")).is_file()
    assert str(bundle.get("recovery_safety_logic_json") or "").strip() == str(
        (tmp_path / "recovery_final" / "cca_safety_logic.json").resolve()
    )
    assert (tmp_path / "recovery_final" / "cca_safety_logic.json").is_file()
    assert (tmp_path / "recovery_final" / "SAFE_2_dfa.dot").is_file()
    assert not (tmp_path / "recovery_final" / "SAFE_1_dfa.dot").exists()
    assert not (tmp_path / "recovery_final" / "recovery_safety_grounding_prompt_latest.txt").exists()
    assert not (tmp_path / "recovery_final" / "recovery_safety_grounding_response_latest.json").exists()


def test_case3_live_runtime_reuses_worked_dir_for_recovery_outline_primitves_and_safety(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "worked"

    async def _run() -> tuple[dict[str, Any], dict[str, str]]:
        _, product_agent, _, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
            reasoning_mode="multi_turn"
        )
        product_agent.recovery_controller = ProductRecoveryController(product_agent)
        product_agent.recovery_controller.bind_methods()
        product_agent.runtime_recovery = {}
        product_agent._runtime_recovery_context = {}
        with patch.object(
            ProductRecoveryController,
            "_runtime_recovery_safety_live_root",
            return_value=live_root,
        ):
            bridge_debug = product_agent._ensure_live_bridge_per_turn_debug_dir(
                prepared_bridge_request
            )
            dirs = product_agent._recovery_safety_generation_dirs()
            return bridge_debug, dirs

    bridge_debug, dirs = asyncio.run(_run())

    assert Path(str(bridge_debug.get("artifact_directory") or "")) == live_root / "1"
    assert Path(str(dirs.get("recovery_safety_dir") or "")) == (
        live_root / "1" / "recovery_safety"
    )
    assert Path(str(dirs.get("recovery_plan_dir") or "")) == (
        live_root / "1" / "recovery_safety"
    )
    assert Path(str(dirs.get("recovery_safery_dir") or "")) == (
        live_root / "1" / "recovery_safety"
    )


def test_case3_live_runtime_recovery_safety_generation_writes_prompt_response_and_logic_into_recovery_safety(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "worked"

    async def _run() -> dict[str, Any]:
        _, product_agent, _, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
            reasoning_mode="multi_turn"
        )
        controller = ProductRecoveryController(product_agent)
        product_agent.recovery_controller = controller
        controller.bind_methods()
        product_agent.runtime_recovery = {}
        product_agent._runtime_recovery_context = {}
        prepared_bridge_request["multi_turn_session_state"] = {
            "accepted_outline_prefix": _case3_archived_transition_trace(),
            "projected_outline_state": _case3_archived_projected_outline_state(),
        }
        with patch.object(
            ProductRecoveryController,
            "_runtime_recovery_safety_live_root",
            return_value=live_root,
        ):
            dirs = controller._recovery_safety_generation_dirs()
            payload = controller._build_recovery_safety_generation_payload(
                prepared_bridge_request=prepared_bridge_request,
                recovery_safety_scope_id="live_recovery_scope_case3",
                request_id="live_recovery_safety_generate_case3",
                recovery_safety_dir=str(dirs.get("recovery_safety_dir") or "").strip(),
                recovery_plan_dir=str(dirs.get("recovery_plan_dir") or "").strip(),
                recovery_safery_dir=str(dirs.get("recovery_safery_dir") or "").strip(),
            )

        async def _fake_ask_llm_structured(
            prompt: str,
            *,
            response_format: dict[str, Any],
            tools: list[dict[str, Any]] | None = None,
            tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
            max_tool_rounds: int = 3,
        ) -> dict[str, Any]:
            del prompt, response_format, tools, tool_executor, max_tool_rounds
            return _fake_case3_recovery_safety_grounding_response(payload)

        product_agent.ask_llm_structured = _fake_ask_llm_structured  # type: ignore[method-assign]
        return await generate_recovery_safety_bundle(product_agent, payload)

    result = asyncio.run(_run())

    recovery_safety_dir = live_root / "1" / "recovery_safety"
    assert result["ok"] is True
    assert Path(str(result.get("recovery_safety_dir") or "")) == recovery_safety_dir
    assert (recovery_safety_dir / "recovery_safety_input_snapshot.json").is_file()
    assert (recovery_safety_dir / "recovery_safety_grounding_prompt_latest.txt").is_file()
    assert (recovery_safety_dir / "recovery_safety_grounding_response_latest.json").is_file()
    assert (recovery_safety_dir / "recovery_safety_generation_result.json").is_file()
    assert (recovery_safety_dir / "cca_safety_logic.json").is_file()
    assert (recovery_safety_dir / "SAFE_2_dfa.dot").is_file()
    assert not (recovery_safety_dir / "SAFE_1_dfa.dot").exists()
    assert not (live_root / "1" / "recovery_final").exists()


def test_case3_live_runtime_writes_recovery_final_bundle_from_exact_final_output(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "worked"

    async def _run() -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        _, product_agent, _, prepared_bridge_request = await _prepare_bridge_dryrun_harness(
            reasoning_mode="multi_turn"
        )
        controller = ProductRecoveryController(product_agent)
        product_agent.recovery_controller = controller
        controller.bind_methods()
        product_agent.agent_name = "assembly_board-v1"
        product_agent._utc_now_iso = staticmethod(
            lambda: datetime.now(timezone.utc).isoformat()
        )
        product_agent._runtime_repair_max_attempts = 3
        product_agent._runtime_bridge_mode = "pre_ran"
        product_agent._runtime_bridge_validation_policy = "validated"
        product_agent._runtime_bridge_archive_path = ""
        product_agent._runtime_bridge_archive_label = ""
        product_agent.runtime_recovery = {
            "recovery_safety_status": "ready",
            "recovery_safety_logic_json": str(
                live_root / "1" / "recovery_safety" / "cca_safety_logic.json"
            ),
            "recovery_safety_dir": str(live_root / "1" / "recovery_safety"),
            "recovery_plan_dir": str(live_root / "1" / "recovery_safety"),
            "recovery_safery_dir": str(live_root / "1" / "recovery_safety"),
            "bridge_debug": None,
            "history": [],
        }
        product_agent._runtime_recovery_context = {}

        recovery_safety_dir = live_root / "1" / "recovery_safety"
        recovery_safety_dir.mkdir(parents=True, exist_ok=True)
        (recovery_safety_dir / "cca_safety_logic.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (recovery_safety_dir / "SAFE_2_dfa.dot").write_text(
            "digraph G {}",
            encoding="utf-8",
        )
        (recovery_safety_dir / "recovery_safety_grounding_prompt_latest.txt").write_text(
            "prompt",
            encoding="utf-8",
        )
        (recovery_safety_dir / "recovery_safety_grounding_response_latest.json").write_text(
            "{}",
            encoding="utf-8",
        )
        source_final_output_path = (
            live_root / "1" / "multi_turn_turn09_final_output_response_20260424T010101.txt"
        )
        final_output_payload = {
            "engine": "multi_turn",
            "decision": "final_output_ready",
            "final_output_stage": "primitive_program_ready",
            "status": "paused_after_primitive_generation",
            "current_phase": "finalize",
            "accepted_trace_length": 4,
            "transition_trace": _case3_archived_transition_trace(),
            "accepted_primitive_program": [],
            "executable_recovery_trace": [],
            "primitive_program_complete": True,
        }
        source_final_output_path.write_text(
            json.dumps(final_output_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        prepared_bridge_request["bridge_debug"] = {
            "artifact_directory": str(live_root / "1"),
            "multi_turn_session": {
                "final_output": deepcopy(final_output_payload),
                "turns": [
                    {
                        "turn_index": 9,
                        "phase": "final_output",
                        "final_output_stage": "primitive_program_ready",
                        "response_artifact_path": str(source_final_output_path),
                    }
                ],
            },
            "final_output": deepcopy(final_output_payload),
        }

        with patch.object(
            ProductRecoveryController,
            "_runtime_recovery_safety_live_root",
            return_value=live_root,
        ):
            bundle = controller._maybe_write_recovery_final_bundle(
                prepared_bridge_request=prepared_bridge_request,
            )
        return bundle, prepared_bridge_request, deepcopy(product_agent.runtime_recovery)

    bundle, prepared_bridge_request, runtime_recovery = asyncio.run(_run())

    assert Path(str(bundle.get("recovery_final_dir") or "")) == live_root / "1" / "recovery_final"
    assert Path(str(bundle.get("recovery_final_output_path") or "")).is_file()
    assert str(bundle.get("recovery_safety_logic_json") or "").strip() == str(
        (live_root / "1" / "recovery_final" / "cca_safety_logic.json").resolve()
    )
    assert (live_root / "1" / "recovery_final" / "cca_safety_logic.json").is_file()
    assert (live_root / "1" / "recovery_final" / "SAFE_2_dfa.dot").is_file()
    assert not (live_root / "1" / "recovery_final" / "recovery_safety_grounding_prompt_latest.txt").exists()
    assert not (live_root / "1" / "recovery_final" / "recovery_safety_grounding_response_latest.json").exists()
    assert str(runtime_recovery.get("recovery_safety_logic_json") or "").strip() == str(
        (live_root / "1" / "recovery_final" / "cca_safety_logic.json").resolve()
    )
    assert str(runtime_recovery.get("recovery_safety_dir") or "").strip() == str(
        live_root / "1" / "recovery_safety"
    )
    assert str(
        dict(prepared_bridge_request.get("bridge_debug") or {}).get(
            "recovery_safety_logic_json"
        )
        or ""
    ).strip() == str(bundle.get("recovery_safety_logic_json") or "")
    assert str(
        dict(prepared_bridge_request.get("bridge_debug") or {}).get(
            "recovery_final_output_path"
        )
        or ""
    ).strip() == str(bundle.get("recovery_final_output_path") or "")


def test_case3_dispatch_params_stop_force_fast_path_after_bridge_session_resolves(
    tmp_path: Path,
) -> None:
    _, product_agent, planner, _, recovery = _approve_case3_archived_bridge(
        tmp_path,
        validation_policy="no_validation",
    )

    assert recovery["status"] == "validating"
    product_agent.runtime_recovery["active_bridge_sequence"] = None

    req_4_t1 = dict(planner._find_node("REQ_4_T1") or {})
    params = product_agent._dispatch_params_for_task_node(req_4_t1)

    assert "start_safety_mode" not in params


def test_case3_archived_bridge_approval_fails_when_resume_acquire_entity_cannot_compile(
    tmp_path: Path,
) -> None:
    with patch.object(
        ProductRecoveryController,
        "_resource_can_reach_location",
        return_value=False,
    ):
        _, _, planner, _, recovery = _approve_case3_archived_bridge(tmp_path)

    assert recovery["status"] == "human_required"
    assert "could not be compiled" in str(recovery.get("message") or "")
    assert not any(
        isinstance(node, dict)
        and str(node.get("function_name") or "").strip() == "execute_recovery_macro"
        for node in planner.nodes
    )


def test_case3_archived_bridge_approval_validated_dispatches_runtime_safety_check(
    tmp_path: Path,
) -> None:
    with patch.object(
        ProductRecoveryController,
        "_send_runtime_plan_validation_check_sync",
        autospec=True,
        return_value=None,
    ) as mocked_validation:
        _, product_agent, planner, _, recovery = _approve_case3_archived_bridge(
            tmp_path,
            validation_policy="validated",
            recovery_safety_scope_id="recovery_scope_case3",
            recovery_safety_status="ready",
        )

    assert recovery["status"] == "validating"
    assert str(recovery.get("validation_policy") or "").strip() == "validated"
    mocked_validation.assert_called_once_with(
        product_agent.recovery_controller,
        skip_revalidation=False,
        skip_recovery_safety_validation=False,
    )


def test_case3_archived_bridge_approval_no_validation_dispatches_normal_cca_validation(
    tmp_path: Path,
) -> None:
    with patch.object(
        ProductRecoveryController,
        "_send_runtime_plan_validation_check_sync",
        autospec=True,
        return_value=None,
    ) as mocked_validation:
        _, product_agent, _, _, recovery = _approve_case3_archived_bridge(
            tmp_path,
            validation_policy="no_validation",
        )

    assert recovery["status"] == "validating"
    assert str(recovery.get("validation_policy") or "").strip() == "no_validation"
    mocked_validation.assert_called_once_with(
        product_agent.recovery_controller,
        skip_revalidation=False,
        skip_recovery_safety_validation=True,
    )
    active_bridge_sequence = dict(recovery.get("active_bridge_sequence") or {})
    execution_policy = dict(active_bridge_sequence.get("execution_policy") or {})
    assert execution_policy["validation_policy"] == "no_validation"
    assert execution_policy["complete_full_tail"] is True
    assert product_agent._runtime_recovery_blocks_execution() is True


def test_case3_archived_bridge_approval_no_validation_dispatches_recovery_safety_filter(
    tmp_path: Path,
) -> None:
    _, product_agent, _, _, recovery = _approve_case3_archived_bridge(
        tmp_path,
        validation_policy="no_validation",
    )

    assert recovery["status"] == "validating"
    dispatched_payloads = [
        row
        for row in list(getattr(product_agent, "dispatched_agent_messages", []) or [])
        if str(dict(row.get("metadata") or {}).get("type") or "").strip()
        == "plan_safety_check"
    ]

    assert dispatched_payloads
    assert all(
        not bool(dict(row.get("body") or {}).get("skip_revalidation"))
        for row in dispatched_payloads
    )
    assert all(
        bool(dict(row.get("body") or {}).get("skip_recovery_safety_validation"))
        for row in dispatched_payloads
    )
    assert all(
        "skip_recovery_nominal_validation" not in dict(row.get("body") or {})
        for row in dispatched_payloads
    )


def test_case3_no_validation_ignores_late_runtime_safety_result(
    tmp_path: Path,
) -> None:
    _, product_agent, _, _, recovery = _approve_case3_archived_bridge(
        tmp_path,
        validation_policy="no_validation",
    )

    assert recovery["status"] == "validating"
    product_agent._runtime_recovery_context = {}
    product_agent.runtime_recovery["status"] = "idle"
    violations = [
        {
            "violated_rule_id": "SAFE_2",
            "witness_task_ids": ["REQ_2_T5"],
        }
    ]
    with patch.object(
        ProductRecoveryController,
        "_run_des_runtime_recovery_attempt",
        autospec=True,
        side_effect=AssertionError("late runtime validation result should be ignored"),
    ):
        handled = asyncio.run(
            product_agent._handle_runtime_plan_validation_result(
                ok=False,
                violations=violations,
                request_id="runtime_plan_validation_stale_case3",
            )
        )

    assert handled is True
    assert str(product_agent.runtime_recovery.get("status") or "").strip() == "idle"
    assert str(product_agent.runtime_recovery.get("validation_policy") or "").strip() == "no_validation"


def test_case3_no_validation_terminal_bridge_completion_dispatches_normal_cca_validation(
    tmp_path: Path,
) -> None:
    _, product_agent, planner, _, recovery = _approve_case3_archived_bridge(
        tmp_path,
        validation_policy="no_validation",
    )

    assert recovery["status"] == "validating"
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
    acquire_entity = next(
        (
            node
            for node in planner.nodes
            if isinstance(node, dict)
            and str(node.get("repair_operator") or "").strip() == "acquire_entity"
            and str(node.get("restores_event_id") or "").strip() == "REQ_1_T3"
        ),
        None,
    )
    assert isinstance(acquire_entity, dict)

    for node in (seq1, seq2, seq3, seq4, acquire_entity):
        node["status"] = "completed"

    async def _inline_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    with patch.object(
        ProductRecoveryController,
        "_refresh_bridge_snapshot",
        autospec=True,
        return_value=deepcopy(dict(acquire_entity.get("projected_snapshot") or {})),
    ), patch.object(
        ProductRecoveryController,
        "_bridge_snapshot_mismatch",
        autospec=True,
        return_value="",
    ), patch.object(
        ProductRecoveryController,
        "_bridge_part_entry_mismatch",
        autospec=True,
        return_value="",
    ), patch.object(
        ProductRecoveryController,
        "_bridge_sequence_tail_task_ids",
        autospec=True,
        return_value=[],
    ), patch.object(
        ProductRecoveryController,
        "_system_coordination_state_with_bridge_snapshot",
        autospec=True,
        return_value={},
    ), patch.object(
        ProductRecoveryController,
        "_bridge_continuation_disabled_frontier",
        autospec=True,
        return_value=[],
    ), patch.object(
        ProductRecoveryController,
        "_send_runtime_plan_validation_check",
        autospec=True,
        return_value=None,
    ) as mocked_validation, patch.object(
        ProductRecoveryController,
        "_persist_plan_snapshot",
        autospec=True,
        return_value=None,
    ), patch.object(
        ProductRecoveryController,
        "_persist_product_state",
        autospec=True,
        return_value=None,
    ), patch.object(
        ProductRecoveryController,
        "_persist_resource_state",
        autospec=True,
        return_value=None,
    ), patch.object(
        asyncio,
        "to_thread",
        new=_inline_to_thread,
    ):
        handled = asyncio.run(
            product_agent._handle_bridge_macro_ack(
                task_node=acquire_entity,
                status="completed",
            )
        )

    mocked_validation.assert_called_once_with(
        product_agent.recovery_controller,
        skip_recovery_safety_validation=True,
    )
    assert handled is True
    assert str(product_agent.runtime_recovery.get("status") or "").strip() == "validating"
    assert product_agent.runtime_recovery.get("active_bridge_sequence") is None
    assert product_agent._runtime_recovery_blocks_execution() is True


def test_case3_live_bridge_approval_validated_dispatches_runtime_safety_check(
    tmp_path: Path,
) -> None:
    with patch.object(
        ProductRecoveryController,
        "_send_runtime_plan_validation_check_sync",
        autospec=True,
        return_value=None,
    ) as mocked_validation:
        _, product_agent, _, _, recovery = _approve_case3_archived_bridge(
            tmp_path,
            bridge_source="live_multi_turn",
            bridge_mode="auto",
            validation_policy="validated",
            recovery_safety_scope_id="recovery_scope_case3",
            recovery_safety_status="ready",
        )

    assert recovery["status"] == "validating"
    assert str(recovery.get("validation_policy") or "").strip() == "validated"
    mocked_validation.assert_called_once_with(
        product_agent.recovery_controller,
        skip_revalidation=False,
        skip_recovery_safety_validation=False,
    )


def test_case3_live_bridge_approval_no_validation_dispatches_normal_cca_validation(
    tmp_path: Path,
) -> None:
    with patch.object(
        ProductRecoveryController,
        "_send_runtime_plan_validation_check_sync",
        autospec=True,
        return_value=None,
    ) as mocked_validation:
        _, product_agent, _, _, recovery = _approve_case3_archived_bridge(
            tmp_path,
            bridge_source="live_multi_turn",
            bridge_mode="auto",
            validation_policy="no_validation",
        )

    assert recovery["status"] == "validating"
    assert str(recovery.get("validation_policy") or "").strip() == "no_validation"
    mocked_validation.assert_called_once_with(
        product_agent.recovery_controller,
        skip_revalidation=False,
        skip_recovery_safety_validation=True,
    )


def test_case3_verification_only_bridge_approval_validated_dispatches_runtime_safety_check(
    tmp_path: Path,
) -> None:
    with patch.object(
        ProductRecoveryController,
        "_send_runtime_plan_validation_check_sync",
        autospec=True,
        return_value=None,
    ) as mocked_validation:
        _, product_agent, _, _, recovery = _approve_case3_archived_bridge(
            tmp_path,
            bridge_source="live_multi_turn",
            bridge_mode="auto",
            validation_policy="validated",
            verification_only=True,
            recovery_safety_scope_id="recovery_scope_case3",
            recovery_safety_status="ready",
        )

    assert recovery["status"] == "validating"
    active_bridge_sequence = dict(product_agent.runtime_recovery.get("active_bridge_sequence") or {})
    execution_policy = dict(active_bridge_sequence.get("execution_policy") or {})
    assert execution_policy["verification_only"] is True
    mocked_validation.assert_called_once_with(
        product_agent.recovery_controller,
        skip_revalidation=False,
        skip_recovery_safety_validation=False,
    )


def test_case3_verification_only_bridge_approval_no_validation_dispatches_normal_cca_validation(
    tmp_path: Path,
) -> None:
    with patch.object(
        ProductRecoveryController,
        "_send_runtime_plan_validation_check_sync",
        autospec=True,
        return_value=None,
    ) as mocked_validation:
        _, product_agent, _, _, recovery = _approve_case3_archived_bridge(
            tmp_path,
            bridge_source="live_multi_turn",
            bridge_mode="auto",
            validation_policy="no_validation",
            verification_only=True,
        )

    assert recovery["status"] == "validating"
    active_bridge_sequence = dict(product_agent.runtime_recovery.get("active_bridge_sequence") or {})
    execution_policy = dict(active_bridge_sequence.get("execution_policy") or {})
    assert execution_policy["verification_only"] is True
    assert execution_policy["validation_policy"] == "no_validation"
    mocked_validation.assert_called_once_with(
        product_agent.recovery_controller,
        skip_revalidation=False,
        skip_recovery_safety_validation=True,
    )


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
    if normalized_focus not in {"full", "primitive_generation", "recovery_safety"}:
        raise ValueError("focus must be 'full', 'primitive_generation', or 'recovery_safety'")
    checkpoint_path: Path | None = None
    resume_payload: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        checkpoint_path, resume_payload = _load_resume_checkpoint(resume_checkpoint)
    if normalized_focus in {"primitive_generation", "recovery_safety"}:
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
    recovery_safety_generation: dict[str, Any] = {}
    recovery_safety_status = "none"
    recovery_safety_task: asyncio.Task[dict[str, Any]] | None = None

    def _recovery_safety_started() -> bool:
        return recovery_safety_task is not None or bool(recovery_safety_generation)

    def _start_dryrun_recovery_safety_generation(
        session_state: dict[str, Any] | None,
    ) -> None:
        nonlocal recovery_safety_status, recovery_safety_task
        if _recovery_safety_started():
            return
        planner_bridge_debug = {}
        if callable(getattr(planner, "get_last_bridge_debug", None)):
            planner_bridge_debug = dict(planner.get_last_bridge_debug() or {})
        resolved_session_state = _resolve_dryrun_multi_turn_session_state(
            prepared_bridge_request,
            multi_turn_session=session_state,
            planner_bridge_debug=planner_bridge_debug,
            require_outline_trace=True,
        )
        if not _dryrun_outline_approval_reached(resolved_session_state):
            return
        payload = _build_dryrun_recovery_safety_generation_payload(
            prepared_bridge_request=prepared_bridge_request,
            multi_turn_session=resolved_session_state,
        )
        if not payload:
            return
        Path(str(payload.get("recovery_safety_dir") or "")).mkdir(
            parents=True,
            exist_ok=True,
        )
        recovery_safety_status = "generating"
        recovery_safety_task = asyncio.create_task(
            _generate_dryrun_recovery_safety_artifacts(
                product_agent=product_agent,
                prepared_bridge_request=prepared_bridge_request,
                multi_turn_session=deepcopy(resolved_session_state),
            )
        )

    async def _collect_dryrun_recovery_safety_generation(
        *,
        wait: bool = False,
    ) -> dict[str, Any]:
        nonlocal recovery_safety_generation, recovery_safety_status, recovery_safety_task
        task = recovery_safety_task
        if task is None:
            return recovery_safety_generation
        if not wait and not task.done():
            await asyncio.sleep(0)
            if not task.done():
                return recovery_safety_generation
        recovery_safety_task = None
        try:
            result = await task
        except asyncio.CancelledError:
            recovery_safety_status = "none"
            raise
        recovery_safety_generation = deepcopy(dict(result or {}))
        recovery_safety_status = _dryrun_recovery_safety_status_from_result(
            recovery_safety_generation
        )
        return recovery_safety_generation
    if normalized_focus == "recovery_safety":
        recovery_safety_session = (
            deepcopy(resume_payload.get("session_state") or {})
            if isinstance(resume_payload, dict)
            else {}
        )
        if not _dryrun_outline_approval_reached(recovery_safety_session):
            recovery_safety_session = _case3_archived_outline_ready_session()
        prepared_bridge_request["multi_turn_session_state"] = deepcopy(
            recovery_safety_session
        )
        bridge_debug = dict(prepared_bridge_request.get("bridge_debug") or {})
        bridge_debug["status"] = str(
            recovery_safety_session.get("status") or "ready_for_primitive_generation"
        ).strip()
        bridge_debug["multi_turn_session"] = deepcopy(recovery_safety_session)
        prepared_bridge_request["bridge_debug"] = bridge_debug
        _start_dryrun_recovery_safety_generation(recovery_safety_session)
        await _collect_dryrun_recovery_safety_generation(wait=True)

        result: dict[str, Any] = {
            "scenario": "case3_lg_slippage",
            "reasoning_mode": normalized_reasoning_mode,
            "status": str(bridge_debug.get("status") or ""),
            "proposal": None,
            "bridge_debug": deepcopy(bridge_debug),
            "prepared_bridge_request": prepared_bridge_request,
            "context_summary": deepcopy(prepared_bridge_request.get("context_summary") or {}),
            "llm_input": deepcopy(prepared_bridge_request.get("llm_input") or {}),
            "multi_turn_session": deepcopy(recovery_safety_session),
            "turns": [
                deepcopy(row)
                for row in (recovery_safety_session.get("turns") or [])
                if isinstance(row, dict)
            ],
            "turn_log": deepcopy(product_agent.turn_log),
            "resume_checkpoint_source_path": str(checkpoint_path or ""),
            "prompt_artifact_path": None,
            "latest_prompt_artifact_path": None,
            "response_artifact_path": None,
            "latest_response_artifact_path": None,
            "session_transcript_artifact_path": None,
            "latest_session_transcript_artifact_path": None,
            "resume_checkpoint_artifact_path": None,
            "latest_resume_checkpoint_artifact_path": None,
            "primitive_resume_checkpoint_artifact_path": None,
            "latest_primitive_resume_checkpoint_artifact_path": None,
            "recovery_safety_status": recovery_safety_status,
        }
        if recovery_safety_generation:
            result["recovery_safety_generation"] = deepcopy(recovery_safety_generation)
            result["recovery_safety_dir"] = str(
                recovery_safety_generation.get("recovery_safety_dir") or ""
            ).strip()
            result["recovery_plan_dir"] = str(
                recovery_safety_generation.get("recovery_plan_dir") or ""
            ).strip()
            result["recovery_safery_dir"] = str(
                recovery_safety_generation.get("recovery_safery_dir") or ""
            ).strip()
            result["recovery_safety_logic_json"] = str(
                recovery_safety_generation.get("recovery_safety_logic_json") or ""
            ).strip()
        if write_debug:
            artifact_paths = _write_debug_artifacts(result)
            result.update(artifact_paths)
        return result
    if resume_payload is not None and str(resume_payload.get("kind") or "").strip() == "primitive_batch_resume_checkpoint":
        resume_session_state = deepcopy(resume_payload.get("session_state") or {})
        if _dryrun_outline_approval_reached(resume_session_state):
            _start_dryrun_recovery_safety_generation(resume_session_state)
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
        await _collect_dryrun_recovery_safety_generation(wait=False)
        if _dryrun_primitive_program_ready_payload(
            prepared_bridge_request,
            multi_turn_session=primitive_session,
        ):
            await _collect_dryrun_recovery_safety_generation(wait=True)
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
            "recovery_safety_status": recovery_safety_status,
        }
        if recovery_safety_generation:
            result["recovery_safety_generation"] = deepcopy(recovery_safety_generation)
            result["recovery_safety_dir"] = str(
                recovery_safety_generation.get("recovery_safety_dir") or ""
            ).strip()
            result["recovery_plan_dir"] = str(
                recovery_safety_generation.get("recovery_plan_dir") or ""
            ).strip()
            result["recovery_safery_dir"] = str(
                recovery_safety_generation.get("recovery_safery_dir") or ""
            ).strip()
            result["recovery_safety_logic_json"] = str(
                recovery_safety_generation.get("recovery_safety_logic_json") or ""
            ).strip()
        recovery_final = _write_dryrun_recovery_final_bundle(
            prepared_bridge_request=prepared_bridge_request,
            multi_turn_session=primitive_session,
            recovery_safety_generation=recovery_safety_generation,
        )
        if recovery_final:
            result["recovery_safety_logic_json"] = str(
                recovery_final.get("recovery_safety_logic_json")
                or recovery_final.get("recovery_final_safety_logic_json")
                or result.get("recovery_safety_logic_json")
                or ""
            ).strip()
            result["recovery_final_dir"] = str(
                recovery_final.get("recovery_final_dir") or ""
            ).strip()
            result["recovery_final_output_path"] = str(
                recovery_final.get("recovery_final_output_path") or ""
            ).strip()
        if write_debug:
            artifact_paths = _write_debug_artifacts(result)
            result.update(artifact_paths)
        return result
    if resume_payload is not None:
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
            execute_multi_turn_bridge as _resume_bridge,
        )
        resume_session_state = deepcopy(resume_payload.get("session_state") or {})
        _start_dryrun_recovery_safety_generation(resume_session_state)
        proposal = await _resume_bridge(
            planner,
            prepared_bridge_request,
            session_state=resume_session_state,
        )
    else:
        # First run: stop at outline approval so dryrun can start recovery_safety
        if normalized_reasoning_mode == "multi_turn":
            prepared_bridge_request["_stop_after_multi_turn_phase"] = "outline"
        proposal = await planner.execute_prepared_bridge_request(prepared_bridge_request)
        _start_dryrun_recovery_safety_generation(
            _resolve_dryrun_multi_turn_session_state(
                prepared_bridge_request,
                planner_bridge_debug=(
                    dict(planner.get_last_bridge_debug() or {})
                    if callable(getattr(planner, "get_last_bridge_debug", None))
                    else {}
                ),
            )
        )
        await _collect_dryrun_recovery_safety_generation(wait=False)
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
            ss = _resolve_dryrun_multi_turn_session_state(
                prepared_bridge_request,
                planner_bridge_debug=(
                    dict(planner.get_last_bridge_debug() or {})
                    if callable(getattr(planner, "get_last_bridge_debug", None))
                    else {}
                ),
            )
            _start_dryrun_recovery_safety_generation(ss)
            await _collect_dryrun_recovery_safety_generation(wait=False)
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
            await _collect_dryrun_recovery_safety_generation(wait=False)
            if post_validation_resume_budget > 0:
                post_validation_resume_budget -= 1
                next_ss = _resolve_dryrun_multi_turn_session_state(
                    prepared_bridge_request,
                    planner_bridge_debug=(
                        dict(planner.get_last_bridge_debug() or {})
                        if callable(getattr(planner, "get_last_bridge_debug", None))
                        else {}
                    ),
                )
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

    if stop_before_primitive_generation and _dryrun_outline_approval_reached(
        multi_turn_session
    ):
        await _collect_dryrun_recovery_safety_generation(wait=True)
    elif _dryrun_primitive_program_ready_payload(
        prepared_bridge_request,
        multi_turn_session=multi_turn_session,
    ):
        await _collect_dryrun_recovery_safety_generation(wait=True)
    else:
        await _collect_dryrun_recovery_safety_generation(wait=False)

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
        "recovery_safety_status": recovery_safety_status,
    }

    if recovery_safety_generation:
        result["recovery_safety_generation"] = deepcopy(recovery_safety_generation)
        result["recovery_safety_dir"] = str(
            recovery_safety_generation.get("recovery_safety_dir") or ""
        ).strip()
        result["recovery_plan_dir"] = str(
            recovery_safety_generation.get("recovery_plan_dir") or ""
        ).strip()
        result["recovery_safery_dir"] = str(
            recovery_safety_generation.get("recovery_safery_dir") or ""
        ).strip()
        result["recovery_safety_logic_json"] = str(
            recovery_safety_generation.get("recovery_safety_logic_json") or ""
        ).strip()
    recovery_final = _write_dryrun_recovery_final_bundle(
        prepared_bridge_request=prepared_bridge_request,
        multi_turn_session=multi_turn_session,
        recovery_safety_generation=recovery_safety_generation,
    )
    if recovery_final:
        result["recovery_safety_logic_json"] = str(
            recovery_final.get("recovery_safety_logic_json")
            or recovery_final.get("recovery_final_safety_logic_json")
            or result.get("recovery_safety_logic_json")
            or ""
        ).strip()
        result["recovery_final_dir"] = str(
            recovery_final.get("recovery_final_dir") or ""
        ).strip()
        result["recovery_final_output_path"] = str(
            recovery_final.get("recovery_final_output_path") or ""
        ).strip()

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
    recovery_safety_dir = result.get("recovery_safety_dir")
    recovery_plan_dir = result.get("recovery_plan_dir")
    recovery_safery_dir = result.get("recovery_safery_dir")
    recovery_safety_dir = recovery_safety_dir or recovery_safery_dir or recovery_plan_dir
    recovery_safety_logic_json = result.get("recovery_safety_logic_json")
    recovery_final_dir = result.get("recovery_final_dir")
    recovery_final_output_path = result.get("recovery_final_output_path")
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
            recovery_safety_dir,
            recovery_plan_dir,
            recovery_safery_dir,
            recovery_safety_logic_json,
            recovery_final_dir,
            recovery_final_output_path,
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
    if recovery_safety_dir:
        print("Recovery safety dir:      ", recovery_safety_dir)
    if recovery_safety_logic_json:
        print("Recovery safety logic:    ", recovery_safety_logic_json)
    if recovery_final_dir:
        print("Recovery final dir:       ", recovery_final_dir)
    if recovery_final_output_path:
        print("Recovery final output:    ", recovery_final_output_path)


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
        choices=("full", "primitive_generation", "recovery_safety"),
        help=(
            "Run the full harness, start directly from the known Case 3 "
            "primitive-generation outline, or run only recovery_safety from that outline"
        ),
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

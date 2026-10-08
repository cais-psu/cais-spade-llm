"""Case 3 runtime failure-context dry-run for recovery stages.

The reusable entrypoint is cais_spade_llm.recovery_framework.scenario_runner.
The legacy script also delegates to it; provide --runtime-context explicitly
and choose --mode outline, primitive, safety, or full.

Each mode loads runtime_context.json and derives the Case 3 failure facts from
the configured bundle, failure event, part tracker, and resource snapshots.
The Case 3 response fixtures are only used by tests as mocked LLM responses.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import pytest
import json
import logging
import os
import shutil
import sys
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
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

from jsonschema import Draft202012Validator  # noqa: E402

from cais_spade_llm.agents.central_controller.recovery_safety_generation import (
    generate_recovery_safety_bundle,
)  # noqa: E402
from cais_spade_llm.agents.central_controller import (  # noqa: E402
    outline_macro_safety as outline_macro_safety_module,
)
from cais_spade_llm.agents.central_controller.outline_macro_safety import (  # noqa: E402
    validate_outline_macro_recovery_safety,
)
from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner  # noqa: E402
from cais_spade_llm.agents.intelligent_product.product_recovery_controller import (  # noqa: E402
    _private_pending_nominal_tasks,
)
from cais_spade_llm.agents.intelligent_product.replanner.failure_context import (  # noqa: E402
    build_failure_event,
    failure_context_from_scenario_config,
    load_failure_scenario_config,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (  # noqa: E402
    multi_turn as multi_turn_mode,
    multi_turn_outline_generation,
    multi_turn_outline_state,
    multi_turn_prompts,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import (  # noqa: E402
    recovery_artifacts,
    recovery_validation_service,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (  # noqa: E402
    write_recovery_artifacts,
)
from cais_spade_llm.agents.resource_agent.resource_agent import (  # noqa: E402
    ResourceAgent,
)
from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent  # noqa: E402
from cais_spade_llm.agents.shared_information import llm_agent as llm_agent_module  # noqa: E402
from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (  # noqa: E402
    recovery_validation_fingerprint,
)
from cais_spade_llm.resources.resource_primitives import (  # noqa: E402
    get_resource_recovery_snapshot,
)


DEBUG_ROOT = ROOT / "cais_spade_llm" / "monitor" / "debug"
CASE3_RESPONSE_FIXTURES = ROOT / "test" / "fixtures" / "part_slippage"
CASE3_RUNTIME_CONTEXT = CASE3_RESPONSE_FIXTURES / "runtime_context.json"
AUTO_CANDIDATE_COUNT_CAP = 8


from cais_spade_llm.recovery_framework.scenario_runner import (
    DEFAULT_LIVE_MODEL,
    DEFAULT_REASONING_EFFORT,
    _load_local_env,
    _env_default,
    _normalize_reasoning_effort_for_model,
    _parse_structured_json_text,
    _load_json,
    _write_json,
    _utc_token,
    _debug_root,
    _repo_path,
    _load_case3_runtime_context,
    _bundle_artifact_path,
    _case3_paths,
    _case3_bundle_context,
    _normalize_recovery_outline_experiment_settings,
    _recovery_action_horizon_fields,
    _recovery_candidate_count_fields,
    _load_recovery_outline_experiment_settings,
    _load_robot_config,
    ProcessPlannerPrepareTrace,
    FakeProductAgent,
    FakeRecoveryRobot,
    _task_node_by_id,
    _apply_runtime_status_snapshot,
    _merge_part_tracker,
    _build_live_style_failure_payload,
    _build_live_style_slippage_fixture,
    _normalized_observation_pose,
    _holder_resource_jid_for_part_row,
    _build_shared_grounding_observation_catalog,
    _configure_live_recovery_session,
    _prepare_recovery_dryrun_harness,
    _outline_trace_from_session_state,
    _latest_session,
    _primitive_program_from_session,
    _execute_recovery_until,
    _build_recovery_safety_payload,
    _run_safety_from_outline,
    _primitive_ready_payload,
    _primitive_ready_source_path,
    _write_recovery_final_bundle,
    _run_actual_recovery,
    run_recovery_outline_only,
    run_primitive_composition,
    run_safety_synthsis,
    run_full,
    _outline_ids,
    _outline_trace_from_safety,
    _validate_outline_trace,
    _validate_primitive_program,
    _validate_safety_result,
    _compact_json,
    _emit,
    _print_result,
    _print_experiment_settings,
    _print_fault_event,
    _print_turns,
    _print_artifact_paths,
    _print_outline,
    _print_primitives,
    _print_safety,
    _parse_args,
    main,
 )


@pytest.fixture(autouse=True)
def archived_outline_ap_fixture(monkeypatch):
    """Keep archived task-abstraction scenarios separate from physical admission.

    These fixtures predate physical primitive grounding. Their explicit abstract
    selectors remain useful for testing symbolic planning and DFA history, but
    they are never used as current runtime safety definitions.
    """
    from ppr_ap_migration import migrate_ap_key
    from cais_spade_llm.agents.central_controller.ppr_ap import (
        make_ap_definition, canonical_ap_key, parse_ap_definition,
    )

    def archived_projection(rule):
        destination = (rule.get("context") or {}).get("destination", "")
        priority = rule.get("constraint_type") == "ordering_place_approach_priority"
        products = rule.get("product") or []
        gateway = str(products[0]).lower() if products else ""
        rows = []
        for ap in rule.get("aps") or []:
            definition = parse_ap_definition(migrate_ap_key(ap["full"]))
            kind = definition["kind"]
            part, resource = definition["product"], definition["resource"]
            symbol = definition["state" if kind == "ap_state" else "event"]["symbol"]
            if priority:
                part, resource = part.lower(), "any"
                condition = "part_goal_satisfied" if part == gateway else "move_part_to_destination"
                kind = "ap_state" if part == gateway else "ap_event"
                selector = {"mode": condition, "part": part, "resource": resource,
                            "destination": destination}
                if part == gateway:
                    selector["states"] = ["assembled", "placed"]
            else:
                condition = "resource_move_to_destination" if kind == "ap_event" else "resource_in_destination"
                selector = {"mode": condition, "part": part, "resource": resource,
                            "destination": destination}
            projected = make_ap_definition(kind, part, "*", resource, condition,
                                           {"destination": destination})
            rows.append({"label": ap["label"], "full": canonical_ap_key(projected),
                         "definition": projected, "selector": selector})
        return rows
    monkeypatch.setattr(ProcessPlannerPrepareTrace, "_recovery_rule_recovery_aps",
                        staticmethod(archived_projection))


def _load_response_fixture(filename: str) -> dict[str, Any]:
    payload = _load_json(CASE3_RESPONSE_FIXTURES / filename)
    if not isinstance(payload, dict):
        raise TypeError(f"response fixture {filename} did not decode to an object")
    return payload


def _without_llm_private_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_llm_private_fields(item)
            for key, item in value.items()
            if key not in {"description", "expected_start_state"}
        }
    if isinstance(value, list):
        return [_without_llm_private_fields(item) for item in value]
    return deepcopy(value)


def _fixture_outline_responses() -> list[dict[str, Any]]:
    fixture_specs = (
        (
            "turn01_grounding_response.json",
            ("thought", "decision", "observe_requests"),
        ),
        (
            "turn02_grounding_response.json",
            ("thought", "decision", "observe_requests"),
        ),
        (
            "turn03_outline_response.json",
            ("thought", "selected_candidate_index", "candidate_events"),
        ),
        (
            "turn04_outline_response.json",
            ("thought", "selected_candidate_index", "candidate_events"),
        ),
        (
            "turn05_outline_response.json",
            ("thought", "selected_candidate_index", "candidate_events"),
        ),
        (
            "turn06_outline_response.json",
            ("thought", "selected_candidate_index", "candidate_events"),
        ),
    )
    responses: list[dict[str, Any]] = []
    for filename, response_keys in fixture_specs:
        payload = _load_response_fixture(filename)
        model_response = {
            key: deepcopy(payload[key]) for key in response_keys if key in payload
        }
        if (
            "candidate_events" in model_response
            and _load_recovery_outline_experiment_settings()[
                "recovery_selection_mode"
            ]
            == "neurosymbolic"
        ):
            selected_index = int(model_response.pop("selected_candidate_index", 0) or 0)
            candidate_events = list(model_response.get("candidate_events") or [])
            model_response["candidate_events"] = [
                deepcopy(candidate_events[selected_index])
            ]
        responses.append(_without_llm_private_fields(model_response))
    return [_without_llm_private_fields(response) for response in responses]


def _with_three_mocked_candidate_events(response: dict[str, Any]) -> dict[str, Any]:
    """Expand a test-only response to the three-candidate runtime contract."""
    expanded = deepcopy(response)
    candidate_events = [
        deepcopy(row)
        for row in (expanded.get("candidate_events") or [])
        if isinstance(row, dict)
    ]
    selected_index = int(expanded.get("selected_candidate_index") or 0)
    selected_event = deepcopy(candidate_events[selected_index])
    while len(candidate_events) < 3:
        alternative_number = len(candidate_events) + 1
        alternative = deepcopy(selected_event)
        alternative["outline_id"] = (
            f"{str(selected_event.get('outline_id') or '').strip()}_alternative_"
            f"{alternative_number}"
        )
        alternative["event_name"] = (
            f"{str(selected_event.get('event_name') or '').strip()}_alternative_"
            f"{alternative_number}"
        )
        alternative["rationale"] = (
            f"Mocked alternative {alternative_number} with the same declared physical effect."
        )
        candidate_events.append(alternative)
    expanded["candidate_events"] = candidate_events
    return expanded


def _novel_symbol_outline_responses() -> list[dict[str, Any]]:
    responses = _fixture_outline_responses()[:2]
    novel_responses = [
        {
            "thought": "Use explicit occupancy and a grounded named pose.",
            "selected_candidate_index": 0,
            "candidate_events": [
                {
                    "outline_id": "novel_clear_xarm6",
                    "event_name": "evt_q7",
                    "resource_jid": 'recovery-resource-3@localhost',
                    "expected_end_state": {
                        "resource_state": "xarm6_clear_state",
                        "resource_location": "home",
                    },
                    "rationale": "Leave the explicitly occupied protected destination.",
                }
            ],
        },
        {
            "thought": "Free the reachable manipulator without entering the destination.",
            "selected_candidate_index": 0,
            "candidate_events": [
                {
                    "outline_id": "novel_stage_mcp",
                    "event_name": "evt_z9",
                    "resource_jid": 'recovery-resource-4@localhost',
                    "part_name": 'gear_large',
                    "expected_end_state": {
                        "resource_state": "mcp_buffer_clear",
                        "held_part": None,
                        "part_state": "mcp_waiting_recovery",
                        "part_location": '3D Printing Station',
                    },
                    "rationale": "Release MCP at a supplied reachable location.",
                }
            ],
        },
        {
            "thought": "Acquire the affected part from the grounded observation.",
            "selected_candidate_index": 0,
            "candidate_events": [
                {
                    "outline_id": "novel_acquire_lg",
                    "event_name": "evt_n4",
                    "resource_jid": 'recovery-resource-4@localhost',
                    "part_name": 'KET4_Square_4mm',
                    "expected_end_state": {
                        "resource_state": "lg_secured",
                        "held_part": 'KET4_Square_4mm',
                        "part_state": "lg_under_recovery_control",
                        "part_location": 'recovery-resource-4@localhost',
                    },
                    "rationale": 'Acquire KET4_Square_4mm using its observed pose.',
                }
            ],
        },
        {
            "thought": "Restore the held affected part to its supplied goal location.",
            "selected_candidate_index": 0,
            "candidate_events": [
                {
                    "outline_id": "novel_restore_lg",
                    "event_name": "evt_v2",
                    "resource_jid": 'recovery-resource-4@localhost',
                    "part_name": 'KET4_Square_4mm',
                    "expected_end_state": {
                        "resource_state": "lg_recovery_complete",
                        "resource_location": "assembly_board-v1",
                        "held_part": None,
                        "part_state": "assembled",
                        "part_location": "assembly_board-v1",
                    },
                    "rationale": 'Release KET4_Square_4mm at assembly_board-v1 after occupancy clears.',
                }
            ],
        },
    ]
    if (
        _load_recovery_outline_experiment_settings()["recovery_selection_mode"]
        == "neurosymbolic"
    ):
        for response in novel_responses:
            response.pop("selected_candidate_index", None)
        responses.extend(deepcopy(novel_responses))
    else:
        responses.extend(
            _with_three_mocked_candidate_events(response) for response in novel_responses
        )
    return [_without_llm_private_fields(response) for response in responses]


def _render_non_case3_candidate_prompt(
    *,
    reverse_order: bool = False,
    candidate_rejection_feedback: list[dict[str, Any]] | None = None,
    accepted_outline_prefix: list[dict[str, Any]] | None = None,
    recovery_des_models: dict[str, Any] | None = None,
    action_horizon: str = "1",
) -> str:
    resources = [
        {
            "resource_jid": "ur5e@localhost",
            "current_state": "idle",
            "current_location": "Assembly Station",
            "held_part": None,
            "gripper_state": "open",
            "current_pose": {"x": -0.25, "y": 0.22, "z": 1.18},
        },
        {
            "resource_jid": "xarm6@localhost",
            "current_state": "failed",
            "current_location": "station",
            "held_part": None,
            "gripper_state": "open",
            "current_pose": {"x": 0.1, "y": 0.08, "z": 1.2},
        },
    ]
    parts = [
        {
            "part_name": "LCP",
            "current_state": "misplaced",
            "current_location": "station",
            "current_holder_resource_jid": None,
            "origin_location": "prusa-mk4-1",
            "goal_location": "station",
            "goal_requirement_id": "REQ_1",
            "observed_pose": {"x": 0.0, "y": 0.2, "z": 1.035},
        },
        {
            "part_name": "MG",
            "current_state": "assembled",
            "current_location": "Assembly Station",
            "current_holder_resource_jid": None,
            "origin_location": "prusa-mk4-2",
            "goal_location": "Assembly Station",
            "goal_requirement_id": "REQ_3",
            "observed_pose": {"x": 0.0, "y": -0.08, "z": 1.025},
        },
    ]
    if reverse_order:
        resources.reverse()
        parts.reverse()

    recovery_resources = {
        row["resource_jid"]: {
            "static_capabilities": {
                "named_poses": ["home", row["current_location"]],
                "reachable_locations": ["prusa-mk4-1", "prusa-mk4-2", "station"],
            },
            "recovery_adapter": {
                "supports_executable_recovery": True,
                "supports_manipulator_pick_place": True,
            },
            "recovery_snapshot": deepcopy(row),
        }
        for row in resources
    }
    llm_input = {
        "fault_event": {
            "focused_resource_jid": "xarm6@localhost",
            "blocked_at_task_id": "REQ_1_T3",
            "blocked_at_function": "place_approach",
            "affected_part_names": ["LCP"],
        },
        "observed_runtime_state": {"resources": deepcopy(resources)},
        "part_facts": deepcopy(parts),
        "relevant_assembly_requirements": [
            {
                "requirement_id": "REQ_1",
                "status": "failed",
                "summary": "xarm6@localhost place_approach LCP to station",
            }
        ],
        "goal_conditions": [
            {
                "condition_id": "goal_lcp_state",
                "kind": "goal_part_state",
                "condition_family": "goal",
                "entity_kind": "part",
                "entity": "LCP",
                "field": "state",
                "expected": "assembled",
            },
            {
                "condition_id": "goal_lcp_location",
                "kind": "goal_part_location",
                "condition_family": "goal",
                "entity_kind": "part",
                "entity": "LCP",
                "field": "location",
                "expected": "station",
            },
        ],
        "loaded_safety_rules": [
            {
                "id": "SAFE_1",
                "raw_text": "LCP by xarm6@localhost must place_approach before MG by ur5e@localhost.",
                "constraint_type": "ordering_place_approach_priority",
                "product": ["LCP", "MG"],
                "event": "place_approach",
                "context": {"destination": "station"},
            }
        ],
    }
    session_state = {
        "outline_mode": "incremental_candidates_validated",
        "recovery_selection_mode": "pure_llm",
        "action_horizon": action_horizon,
        "action_horizon_steps": 1 if action_horizon == "1" else 3,
        "action_horizon_k": 3,
        "candidate_count": "auto",
        "accepted_outline_prefix": deepcopy(accepted_outline_prefix or []),
        "observation_store": {},
        "outline_lookahead": [],
        "pruned_actions": [],
        "outline_validation_findings": [],
        "candidate_rejection_feedback": deepcopy(candidate_rejection_feedback or []),
        "primitive_escalation_diagnostics": [],
        "recovery_des_models": deepcopy(recovery_des_models or {}),
        "symbolic_resources": {row["resource_jid"]: deepcopy(row) for row in resources},
        "symbolic_parts": {row["part_name"]: deepcopy(row) for row in parts},
    }
    prompt_input = multi_turn_prompts.build_multi_turn_phase_prompt_input(
        phase="outline",
        llm_input=llm_input,
        session_state=session_state,
        recovery_resources=recovery_resources,
        current_recovery_blockers=deepcopy(llm_input["goal_conditions"]),
    )
    return multi_turn_prompts.render_multi_turn_phase_prompt(prompt_input)


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
        "outline_validation_findings": [],
        "pruned_actions": [],
        "symbolic_resources": {},
        "symbolic_parts": {},
    }
    session.update(_recovery_action_horizon_fields(action_horizon_setting))
    session.update(_recovery_candidate_count_fields(candidate_count))
    return session


async def _run_mocked_candidate_handler(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    invalid_candidate_indexes: set[int] | None = None,
    recovery_des_models: dict[str, Any] | None = None,
    admissible_nominal_reentry_event_ids_after: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_outline_generation,
    )

    def _mock_remaining_counts(
        *,
        session_state: dict[str, Any],
        prepared_recovery_request: dict[str, Any],
    ) -> tuple[int, int]:
        del prepared_recovery_request
        accepted_count = len(session_state.get("accepted_outline_prefix") or [])
        return (0, 0) if accepted_count >= 2 else (1, 0)

    async def _mock_validate_candidate_sequence(
        *,
        candidate: dict[str, Any],
        sequence_index: int,
        session_state: dict[str, Any],
        prepared_recovery_request: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        events = [
            deepcopy(row)
            for row in (candidate.get("surface_events") or [])
            if isinstance(row, dict)
        ]
        committed = [
            multi_turn_mode._commit_selected_candidate_task(
                task=event,
                sequence_index=sequence_index + index,
            )
            for index, event in enumerate(events)
        ]
        candidate_index = int(candidate.get("candidate_index") or 0)
        invalid = candidate_index in set(invalid_candidate_indexes or set())
        findings = (
            [
                multi_turn_mode._candidate_schema_finding(
                    task=events[0] if events else {},
                    reason="selected candidate test rejection",
                    evidence={"candidate_index": candidate_index},
                )
            ]
            if invalid
            else []
        )
        return {
            "candidate_index": candidate_index,
            "valid": not invalid,
            "recovery_des_models": deepcopy(recovery_des_models or {}),
            "task": deepcopy(events[0] if events else {}),
            "surface_events": events,
            "validated_events": deepcopy(events),
            "committed_events": committed,
            "grounded_actions": [],
            "validation_findings": findings,
            "validation_stages": [],
            "remaining_blocked_issues": 0,
            "resource_switch_count": 0,
            "admissible_nominal_reentry_event_ids_after": deepcopy(
                admissible_nominal_reentry_event_ids_after or []
            ),
            "pa_state_fingerprint": multi_turn_outline_generation._pa_state_fingerprint(
                session_state=session_state,
                prepared_recovery_request=prepared_recovery_request,
            ),
        }

    with (
        patch.object(multi_turn_mode, "_active_pruned_actions", return_value=[]),
        patch.object(
            multi_turn_outline_generation,
            "_validate_candidate_sequence",
            side_effect=_mock_validate_candidate_sequence,
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
            prepared_recovery_request={},
            planner=object(),
        )


async def _run_fixture_outline_prefix(
    *, debug_root: Path, responses: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run all supplied fixture turns, retaining an incomplete accepted prefix."""
    _, product_agent, planner, request = await _prepare_recovery_dryrun_harness(
        debug_root=debug_root, scripted_responses=responses,
    )
    request["_stop_after_multi_turn_phase"] = "outline"
    session = None
    while product_agent._scripted_responses:
        if session is None:
            await planner.execute_prepared_recovery_request(request)
        else:
            await multi_turn_mode.execute_multi_turn_recovery(
                planner, request, session_state=session,
            )
        session = deepcopy(request.get("multi_turn_session_state") or _latest_session(request, planner))
        if session.get("current_phase") != "outline":
            break
    assert not product_agent._scripted_responses
    return {
        "prepared_recovery_request": request,
        "transition_trace": _outline_trace_from_session_state(session),
        "multi_turn_session": session,
        "turns": deepcopy(session["turns"]),
        "llm_response_source": "mocked_scripted_fixture",
        "accepted_primitive_program": _primitive_program_from_session(session),
    }


def test_case3_actual_outline_uses_runtime_failure_context(tmp_path: Path) -> None:
    runtime_context = _load_case3_runtime_context()
    expected_failed_task_id = str(
        dict(runtime_context.get("failure_event") or {}).get("failed_task_id") or ""
    ).strip()
    responses = _fixture_outline_responses()
    # Request the predefined picked successor; geometric evidence is still absent.
    responses[4]["candidate_events"][0]["expected_end_state"]["resource_state"] = "picked"
    result = asyncio.run(_run_fixture_outline_prefix(debug_root=tmp_path, responses=responses))
    trace = result["transition_trace"]
    prepared_recovery_request = result["prepared_recovery_request"]
    failure_context = prepared_recovery_request["failure_context_raw"]
    plan_payload = _load_json(_case3_paths(runtime_context)["plan"])
    assert isinstance(plan_payload, dict)
    failed_task = _task_node_by_id(plan_payload, expected_failed_task_id)
    expected_p_id = sorted(
        part_name
        for part_name, part_row in dict(runtime_context.get("part_tracker") or {}).items()
        if isinstance(part_row, dict)
        and str(part_row.get("state") or "").strip()
        != str(runtime_context.get("goal_state") or "").strip()
    )

    assert trace
    assert failure_context["failed_task_id"] == expected_failed_task_id
    assert failure_context["failed_function_name"] == "place_insert"
    assert prepared_recovery_request["ra_jid"] == failed_task["resource_jid"]
    assert prepared_recovery_request["P_id"] == expected_p_id
    assert result["llm_response_source"] == "mocked_scripted_fixture"
    assert (tmp_path / "recovery_outline").is_dir()
    grounding_request_paths = sorted(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_grounding_request_*.txt"
        )
    )
    grounding_result_paths = sorted(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_grounding_result_*.json"
        )
    )
    assert len(grounding_request_paths) == 2
    assert len(grounding_result_paths) == 2
    assert not list(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_grounding_llm_response_*.txt"
        )
    )
    assert not list(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_grounding_response_*.txt"
        )
    )

    expected_lg_observation = deepcopy(
        next(
            row
            for row in (runtime_context.get("resource_snapshots") or [])
            if isinstance(row, dict)
            and str(row.get("resource_jid") or "").strip() == 'recovery-resource-3@localhost'
        )["observations"]['KET4_Square_4mm']
    )
    first_grounding_result = json.loads(
        grounding_result_paths[0].read_text(encoding="utf-8")
    )
    second_grounding_result = json.loads(
        grounding_result_paths[1].read_text(encoding="utf-8")
    )
    assert first_grounding_result["effective_decision"] == "observe"
    assert first_grounding_result["next_phase"] == "grounding"
    assert first_grounding_result["llm_response"]["decision"] == "observe"
    assert first_grounding_result["observation_results"][0]["observation_key"] == (
        'observed_pose_KET4_Square_4mm'
    )
    assert first_grounding_result["observation_results"][0]["output"] == (
        expected_lg_observation
    )
    assert first_grounding_result["observation_store_after_turn"]['observed_pose_KET4_Square_4mm'] == (
        expected_lg_observation
    )
    assert second_grounding_result["effective_decision"] == "grounded"
    assert second_grounding_result["next_phase"] == "outline"
    assert second_grounding_result["observation_results"] == []
    assert second_grounding_result["observation_store_after_turn"]['observed_pose_KET4_Square_4mm'] == (
        expected_lg_observation
    )

    first_request_text = grounding_request_paths[0].read_text(encoding="utf-8")
    request_message_text = first_request_text.split("Response Format", maxsplit=1)[0]
    assert "role=system" not in first_request_text
    assert "role=user" in first_request_text
    assert '"name": "multi_turn_grounding_response"' in first_request_text
    assert "multi_turn_grounding_response" not in request_message_text
    assert '"goal_location"' not in request_message_text
    assert '"goal_requirement_id"' not in request_message_text
    assert 'restore KET4_Square_4mm to assembly_board-v1' in request_message_text

    grounding_turns = [
        turn
        for turn in result["turns"]
        if isinstance(turn, dict)
        and str(turn.get("phase") or "").strip().lower() == "grounding"
    ]
    assert len(grounding_turns) == 2
    for turn in grounding_turns:
        assert turn["request_artifact_path"] == turn["prompt_artifact_path"]
        assert turn["grounding_result_artifact_path"] == turn["response_artifact_path"]
        assert "llm_response_artifact_path" not in turn

    outline_request_paths = sorted(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_outline_request_*.txt"
        )
    )
    outline_result_paths = sorted(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_outline_result_*.json"
        )
    )
    outline_audit_paths = sorted(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_outline_audit_*.json"
        )
    )
    outline_stack_paths = sorted(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_outline_stack_*.json"
        )
    )
    outline_turns = [
        turn
        for turn in result["turns"]
        if isinstance(turn, dict)
        and str(turn.get("phase") or "").strip().lower() == "outline"
    ]
    llm_outline_turns = [
        turn for turn in outline_turns if turn.get("llm_called") is not False
    ]
    assert len(outline_request_paths) == len(llm_outline_turns)
    assert len(outline_result_paths) == len(outline_turns)
    assert len(outline_audit_paths) == len(outline_turns)
    assert len(outline_stack_paths) == len(outline_turns)
    assert not list(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_outline_llm_response_*.txt"
        )
    )
    assert not list(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_outline_response_*.txt"
        )
    )
    assert not list(
        (tmp_path / "recovery_outline").glob("multi_turn_turn*_outline_index_*.json")
    )

    first_outline_request = outline_request_paths[0].read_text(encoding="utf-8")
    outline_message_text = first_outline_request.split(
        "Response Format", maxsplit=1
    )[0]
    assert len(outline_message_text) < 6_000
    assert len(first_outline_request) < 12_000
    for private_model_token in (
        "RA-Owned Recovery DES Models",
        "local_event_alphabet",
        "descriptor_fingerprint",
        '"guards"',
        '"updates"',
        "detect_parts",
        "compute_pick_targets",
        "get_current_pose",
            "move_relative",
            "gripper_state",
            "goal_recovery_events",
            "cca_admissible_goal_recovery_event_ids",
        ):
        assert private_model_token not in outline_message_text
    assert "gripper_state" not in first_outline_request
    assert '"held_part_location": "recovery-resource-3@localhost"' not in outline_message_text
    assert '"held_part_location": "recovery-resource-4@localhost"' in outline_message_text
    expected_pose = dict(expected_lg_observation.get("pose") or {})
    assert '"observed_pose": {' in first_outline_request
    for axis in ("x", "y", "z"):
        assert f'"{axis}": {expected_pose[axis]}' in first_outline_request
    ur5e_capability_lines = [
        line
        for line in outline_message_text.splitlines()
        if 'recovery-resource-4@localhost' in line
        and ("reachability=" in line or "grounded_target_refs=" in line)
    ]
    xarm6_capability_lines = [
        line
        for line in outline_message_text.splitlines()
        if 'recovery-resource-3@localhost' in line
        and ("reachability=" in line or "grounded_target_refs=" in line)
    ]
    assert ur5e_capability_lines
    assert all('3D Printing Station' in line for line in ur5e_capability_lines)
    assert all('Buffer For Machined parts' not in line for line in ur5e_capability_lines)
    assert xarm6_capability_lines
    assert all('Buffer For Machined parts' in line for line in xarm6_capability_lines)
    assert all('3D Printing Station' not in line for line in xarm6_capability_lines)
    assert 'reachability=["Buffer For Machined parts", "assembly_board-v1", "Exit"]' in (
        outline_message_text
    )
    assert "Location order is lexical and does not express a preference." not in (
        outline_message_text
    )
    assert "named poses" not in outline_message_text

    first_outline_result = json.loads(
        outline_result_paths[0].read_text(encoding="utf-8")
    )
    first_outline_audit = json.loads(
        outline_audit_paths[0].read_text(encoding="utf-8")
    )
    llm_response = first_outline_audit["llm_response"]
    assert set(llm_response) == {"thought", "candidate_events"}
    assert len(llm_response["candidate_events"]) == 1
    assert "candidate_evaluation_summary" not in llm_response
    assert len(first_outline_result["candidate_evaluation_summary"]) == 1
    assert len(first_outline_audit["candidate_evaluation_summary"]) == 1
    assert "private_recovery_des_models" not in first_outline_result
    assert "llm_response" not in first_outline_result
    assert "thought" not in first_outline_result
    assert "candidate_events" not in first_outline_result
    assert "transition_trace" not in first_outline_result
    assert "projected_successor" not in first_outline_result[
        "candidate_evaluation_summary"
    ][0]
    assert "gripper_state" not in json.dumps(first_outline_result)
    for evaluation in first_outline_result["candidate_evaluation_summary"]:
        stages = list(evaluation.get("validation_stages") or [])
        assert [stage["validator_role"] for stage in stages] == [
            "PA",
            "RA",
            "RA",
            "CCA",
        ]
        assert stages[0]["validation_category"] == "syntax_and_grounding_validation"
        assert stages[1]["validation_category"] == "transition_feasibility"
        assert stages[2]["validation_category"] == "physical_feasibility"
        assert stages[3]["validation_category"] == "safety"
        assert all("mocked" not in stage for stage in stages)
        assert all("state_fingerprint" not in stage for stage in stages)
    audit_stages = first_outline_audit["candidate_evaluation_summary"][0][
        "validation_stages"
    ]
    assert audit_stages[0]["mocked"] is False
    assert audit_stages[1]["mocked"] is True
    assert audit_stages[2]["mocked"] is True
    assert audit_stages[3]["mocked"] is True
    first_selection_evidence = dict(
        first_outline_audit.get("selection_evidence") or {}
    )
    # At the two-pickup checkpoint no robot occupies the destination. A home
    # move passes validation but clears no recovery obligation or CCA blocker.
    assert first_selection_evidence == {}
    assert first_outline_result["selection_status"] == "need_revision"
    assert not first_outline_result.get("selected_transition")
    for result_path, audit_path in zip(
        outline_result_paths,
        outline_audit_paths,
        strict=True,
    ):
        outline_result = json.loads(result_path.read_text(encoding="utf-8"))
        outline_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        result_artifact_paths = outline_result["artifact_paths"]
        assert result_artifact_paths["outline_result_artifact_path"] == str(result_path)
        assert result_artifact_paths["response_artifact_path"] == str(result_path)
        assert result_artifact_paths["outline_audit_artifact_path"] == str(audit_path)
        if outline_audit.get("llm_called") is False:
            assert outline_audit["candidate_source"] == "robot_task_program"
            assert "llm_response" not in outline_audit
            assert "request_artifact_path" not in result_artifact_paths
            assert "prompt_artifact_path" not in result_artifact_paths
        else:
            assert result_artifact_paths["request_artifact_path"] == (
                result_artifact_paths["prompt_artifact_path"]
            )
            assert "llm_response" in outline_audit
        assert Path(result_artifact_paths["outline_audit_artifact_path"]).exists()
        assert Path(result_artifact_paths["outline_stack_artifact_path"]).exists()
        serialized_result = json.dumps(outline_result, sort_keys=True)
        assert "llm_outline_id" not in serialized_result
        assert '"outline_id": "enabledness_' not in serialized_result
        for private_result_token in (
            '"llm_response":',
            '"snapshot":',
            '"fingerprint":',
            '"event_id":',
            '"safety_dfa_states_before":',
            '"safety_dfa_states_after":',
            '"recovery_des_model":',
            '"projected_symbolic_resources":',
            '"projected_symbolic_parts":',
            '"selection_evidence":',
            '"recovery_enabledness_validation_after":',
            '"candidate_id":',
            '"mocked":',
            '"reason":',
            '"rationale":',
            '"findings":',
            '"candidate_comparison":',
        ):
            assert private_result_token not in serialized_result
        assert '"recovery_admission"' not in serialized_result
        serialized_audit = json.dumps(outline_audit, sort_keys=True)
        assert "llm_outline_id" not in serialized_audit
        assert '"outline_id": "enabledness_' not in serialized_audit
        assert '"recovery_admission"' not in serialized_audit
        for audit_token in (
            '"validation_stages":',
            '"state_fingerprint":',
            '"selection_evidence":',
        ):
            assert audit_token in serialized_audit
        assert set(outline_result).issubset(
            {
                "turn_index",
                "phase",
                "decision",
                "next_phase",
                "accepted_trace_length",
                "remaining_blocked_issue_count",
                "selected_by",
                "selection_status",
                "constraint_codes",
                "selected_transition",
                "candidate_evaluation_summary",
                "artifact_paths",
            }
        )
        assert set(outline_result.get("selected_transition") or {}).issubset(
            {
                "outline_id",
                "candidate_source",
                "event_name",
                "resource_jid",
                "part_name",
                "expected_end_state",
            }
        )
        for candidate_row in outline_result["candidate_evaluation_summary"]:
            assert set(candidate_row).issubset(
                {
                    "candidate_index",
                    "valid",
                    "task",
                    "selection_status",
                    "constraint_codes",
                    "validation_stages",
                }
            )
            assert set(candidate_row.get("task") or {}).issubset(
                {
                    "outline_id",
                    "candidate_source",
                    "event_name",
                    "resource_jid",
                    "part_name",
                    "expected_end_state",
                }
            )
            for stage in candidate_row.get("validation_stages") or []:
                assert set(stage).issubset(
                    {
                        "validation_category",
                        "validator_role",
                        "status",
                        "constraint_codes",
                    }
                )
    assert [
        json.loads(path.read_text(encoding="utf-8"))["remaining_blocked_issue_count"]
        for path in outline_result_paths
    ] == [2, 2, 2, 2]
    assert [row["candidate_source"] for row in trace] == ["llm"]
    assert [row["event_name"] for row in trace] == [
        'stage_held_part',
    ]
    for event in trace:
        assert event["primitive_support"]["primitives"]
        assert all(
            witness["feasibility_status"] == "FEASIBLE"
            for witness in event["primitive_support"]["primitives"]
        )

    seq3_audit = json.loads(outline_audit_paths[2].read_text(encoding="utf-8"))
    seq3_evaluation = seq3_audit["candidate_evaluation_summary"][0]
    assert seq3_evaluation["task"]["event_name"] == 'pick_observed_part'
    assert seq3_evaluation["task"]["resource_jid"] == 'recovery-resource-4@localhost'
    assert seq3_evaluation["task"]["part_name"] == 'KET4_Square_4mm'
    assert seq3_evaluation["task"]["expected_end_state"]["held_part"] == 'KET4_Square_4mm'
    assert seq3_evaluation["task"]["expected_end_state"]["part_location"] == 'recovery-resource-4@localhost'
    assert [stage["status"] for stage in seq3_evaluation["validation_stages"]] == [
        "passed", "passed", "rejected", "skipped",
    ]
    assert seq3_evaluation["constraint_codes"] == ["resource_validation_unavailable"]
    assert seq3_audit["selection_status"] == "need_revision"

    mcp_release_audit = json.loads(outline_audit_paths[1].read_text(encoding="utf-8"))
    selected_release_evaluation = next(
        row for row in mcp_release_audit["candidate_evaluation_summary"]
        if row.get("selection_status") == "nondominated"
    )
    enabledness_after = selected_release_evaluation["recovery_enabledness_validation_after"]
    # The slipping robot remains failed because its unnecessary home candidate
    # was not selected. The peer is idle after its acknowledged staged release.
    disabled_pick = json.dumps(
        {"event_name": "pick_approach", "part_name": "KET4_Square_4mm",
         "resource_jid": "recovery-resource-3@localhost"},
        separators=(",", ":"), sort_keys=True,
    )
    assert disabled_pick not in enabledness_after["symbolically_enabled_event_ids"]
    for resource_jid in ("recovery-resource-4@localhost",):
        event_id = json.dumps(
            {"event_name": "pick_approach", "part_name": 'KET4_Square_4mm', "resource_jid": resource_jid},
            separators=(",", ":"), sort_keys=True,
        )
        assert event_id in enabledness_after["symbolically_enabled_event_ids"]
        assert event_id not in enabledness_after["ra_admissible_event_ids"]
        evaluation = next(
            row for row in enabledness_after["event_evaluations"] if row["event_id"] == event_id
        )
        assert evaluation["ra_status"] == "rejected"
        assert evaluation["cca_status"] == "skipped"
        assert evaluation["constraint_codes"] == ["resource_validation_unavailable"]
    assert any(
        '"task_id":"REQ_1_T1"' in event_id
        for event_id in mcp_release_audit["selection_evidence"][
            "admissible_nominal_reentry_event_ids_after"
        ]
    )

    post_release_request = outline_request_paths[2].read_text(encoding="utf-8")
    post_release_message = post_release_request.split("Response Format", maxsplit=1)[0]
    rejected_request = outline_request_paths[3].read_text(encoding="utf-8")
    rejected_message = rejected_request.split("Response Format", maxsplit=1)[0]
    assert "resource_validation_unavailable" in rejected_message
    assert 'recovery-resource-4@localhost/KET4_Square_4mm' in rejected_message
    assert "NEEDS_CONTEXT" in rejected_message
    assert "Accepted Transition Prefix (already applied; do not repeat)" in post_release_message
    assert '"event_name": "stage_held_part"' in post_release_message
    assert "The only immediate blocker to recovering LG" not in post_release_message
    assert '"held_part_location": "recovery-resource-4@localhost"' not in post_release_message
    for index, stack_path in enumerate(outline_stack_paths):
        stack_payload = json.loads(stack_path.read_text(encoding="utf-8"))
        assert isinstance(stack_payload, list)
        assert len(stack_payload) == min(index, 1)
        assert [row["candidate_source"] for row in stack_payload] == ["llm"] * len(stack_payload)
        serialized_stack = json.dumps(stack_payload, sort_keys=True)
        assert "llm_outline_id" not in serialized_stack
        assert '"outline_id": "enabledness_' not in serialized_stack
    assert _load_json(outline_stack_paths[1]) == _load_json(outline_stack_paths[2])
    assert _load_json(outline_stack_paths[2]) == _load_json(outline_stack_paths[3])
    session = result["multi_turn_session"]
    assert session["status"] == "paused_after_outline_turn"
    assert session["current_phase"] == "outline"
    assert not session.get("final_output")
    assert session["accepted_primitive_program"] == []
    latest_outline_turn = outline_turns[-1]
    assert latest_outline_turn["outline_result_artifact_path"] == latest_outline_turn["response_artifact_path"]
    assert Path(latest_outline_turn["outline_audit_artifact_path"]).exists()
    assert Path(latest_outline_turn["outline_stack_artifact_path"]).exists()
    assert "llm_response_artifact_path" not in latest_outline_turn
    assert "turn_index_artifact_path" not in latest_outline_turn


def test_case3_robot_capabilities_match_verified_plan_origins() -> None:
    runtime_context = _load_case3_runtime_context()
    snapshots = {
        str(row.get("resource_jid") or "").strip(): dict(row)
        for row in (runtime_context.get("resource_snapshots") or [])
        if isinstance(row, dict)
    }
    ur5e_snapshot = snapshots['recovery-resource-4@localhost']
    xarm6_snapshot = snapshots['recovery-resource-3@localhost']
    ur5e = _load_robot_config(
        _repo_path(ur5e_snapshot["resource_config"]),
        str(ur5e_snapshot["resource_config_key"]),
    )
    xarm6 = _load_robot_config(
        _repo_path(xarm6_snapshot["resource_config"]),
        str(xarm6_snapshot["resource_config_key"]),
    )
    ur5e_capabilities = dict(dict(ur5e["gazebo"])["static_capabilities"])
    xarm6_capabilities = dict(dict(xarm6["gazebo"])["static_capabilities"])

    from cais_spade_llm.ui.recovery_setup import default_setup, validate_setup

    inputs = validate_setup(default_setup())
    for rid, capabilities, part in (
        ("ur5e-3", xarm6_capabilities, "KET4_Square_4mm"),
        ("ur5e-4", ur5e_capabilities, "gear_large"),
    ):
        model = inputs["models"][rid]
        origin = model["assignments"]["source"]
        assert capabilities["reachability"] == [
            value for value in model["state_variables"]["resource_location"]["domain"]
            if value is not None and value != "home"
        ]
        source = inputs["scene"][origin]
        point = source["output_poses"][part] if "output_poses" in source else source["pickup_pose"]
        assert capabilities["staging_areas"][origin]["anchor_pose"] == dict(
            zip(("x", "y", "z"), point[:3])
        )


def test_case3_mocked_novel_symbol_sequence_converges_without_semantic_cycles(
    tmp_path: Path,
) -> None:
    """The historical novel-state sequence now requires supported successors."""
    result = asyncio.run(_run_fixture_outline_prefix(
        debug_root=tmp_path, responses=_novel_symbol_outline_responses(),
    ))
    session_state = result["multi_turn_session"]
    assert result["transition_trace"] == []
    assert "semantic_state_history" not in session_state
    assert session_state["status"] == "paused_after_outline_turn"
    assert session_state["current_phase"] == "outline"
    assert session_state["accepted_primitive_program"] == []
    assert result["accepted_primitive_program"] == []
    event_names = []
    artifact_dir = tmp_path / "recovery_outline"
    result_paths = sorted(artifact_dir.glob("*_outline_result_*.json"))
    audit_paths = sorted(artifact_dir.glob("*_outline_audit_*.json"))
    stack_paths = sorted(artifact_dir.glob("*_outline_stack_*.json"))
    assert len(result_paths) == len(audit_paths) == len(stack_paths) == 4
    for result_path, audit_path, stack_path in zip(
        result_paths, audit_paths, stack_paths, strict=True,
    ):
        outline_result = _load_json(result_path)
        assert outline_result["selection_status"] == "need_revision"
        assert outline_result["remaining_blocked_issue_count"] > 0
        assert outline_result["accepted_trace_length"] == 0
        for evaluation in outline_result["candidate_evaluation_summary"]:
            event_names.append(evaluation["task"]["event_name"])
            assert evaluation["valid"] is False
        audit_evaluations = _load_json(audit_path)["candidate_evaluation_summary"]
        for evaluation in audit_evaluations:
            expected_code = (
                "unsupported_successor_condition"
                if evaluation["task"]["event_name"] in {"evt_q7", "evt_z9"}
                else "part_traceability_violation"
            )
            assert evaluation["constraint_codes"] == [expected_code]
            assert evaluation["findings"][0]["constraint_owner"] == "resource"
        assert _load_json(stack_path) == []
        for artifact_path in (result_path, audit_path, stack_path):
            serialized_artifact = json.dumps(_load_json(artifact_path), sort_keys=True)
            assert "modeled_task_steps" not in serialized_artifact
            assert "primitive_steps" not in serialized_artifact
    assert event_names == ["evt_q7", "evt_z9", "evt_n4", "evt_v2"]
    serialized_outline = json.dumps(session_state, sort_keys=True)
    assert "modeled_task_steps" not in serialized_outline
    assert "primitive_steps" not in serialized_outline
    assert result["llm_response_source"] == "mocked_scripted_fixture"


def test_case3_runtime_context_has_no_expected_recovery_answers() -> None:
    runtime_context = _load_case3_runtime_context()
    paths = _case3_paths(runtime_context)
    manifest = _load_json(paths["bundle_manifest"])

    assert isinstance(manifest, dict)
    artifacts = dict(manifest.get("artifacts") or {})
    bundle_root = paths["bundle_manifest"].parent
    assert paths["plan"] == bundle_root / str(artifacts["plan_json"])
    assert paths["requirements"] == bundle_root / str(artifacts["requirements_json"])
    assert paths["tools"] == bundle_root / str(artifacts["tools_json"])
    assert paths["safety_logic"] == bundle_root / str(artifacts["safety_logic_json"])

    serialized_context = json.dumps(runtime_context, sort_keys=True)
    for forbidden_token in (
        "candidate_events",
        "candidate_traces",
        "selected_candidate_index",
        "recover_to_home_idle",
        'stage_held_part',
        'pick_observed_part',
        'place_recovered_part',
    ):
        assert forbidden_token not in serialized_context


def test_ur5e_driver_no_longer_rejects_the_example_box_but_rejects_invalid_numbers() -> None:
    import ast
    import math

    path = Path(__file__).resolve().parents[1] / "ros2/cais_lab_robotics/scripts/ur5e_rtde_trajectory_server.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_workspace_error")
    # Isolate the production numeric guard so the offline regression cannot load
    # hardware drivers or contact RTDE. Controller safety checks stay in the driver.
    namespace = {"math": math, "RigidTransform": tuple,
                 "UR5E_RTDE_CARTESIAN_REACH_ORIGIN": (0.0, 0.5, 1.021),
                 "UR5E_RTDE_CARTESIAN_REACH_RADIUS_M": 0.8}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    check = namespace["_workspace_error"]
    rotation = (0.0, 0.0, 0.0, 1.0)
    assert check(((0.0, 0.5, 1.7), rotation)) is None
    assert check(((0.0, 0.5, float("nan")), rotation)) is not None
    assert check(((2.0, 0.5, 1.0), rotation)) is not None


def test_robot_recovery_requires_live_validation_instead_of_an_example_box() -> None:
    from types import SimpleNamespace

    robot = SimpleNamespace(jid="xarm6@localhost", static_capabilities={})
    result = RobotAgent.check_recovery_physical_feasibility(
        robot, part_context={}, recovery_snapshot={},
        grounded_action={"target": {"pose": {"x": 0.0, "y": 0.144, "z": 1.06}}},
    )
    assert result["allowed"] is False
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["constraint_code"] == "resource_validation_unavailable"
    assert "workspace_bounds" not in result["evidence"]


def test_case3_dryrun_uses_production_ra_and_cca_validator_functions() -> None:
    assert (
        FakeRecoveryRobot.check_recovery_physical_feasibility
        is RobotAgent.check_recovery_physical_feasibility
    )
    assert (
        FakeRecoveryRobot.recovery_physical_validation_snapshot
        is RobotAgent.recovery_physical_validation_snapshot
    )
    assert (
        validate_outline_macro_recovery_safety
        is outline_macro_safety_module.validate_outline_macro_recovery_safety
    )


def test_case3_recovery_context_has_no_nominal_terminal_or_stale_resource_location() -> None:
    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )

    marked_reentry_context = dict(
        prepared_recovery_request.get("marked_reentry_context") or {}
    )
    assert all(
        str(condition.get("kind") or "") != "focused_resource_terminal_state"
        for condition in (marked_reentry_context.get("marked_reentry_conditions") or [])
        if isinstance(condition, dict)
    )
    goal_conditions = list(
        dict(prepared_recovery_request.get("llm_input") or {}).get(
            "goal_conditions", []
        )
    )
    assert {str(condition.get("entity") or "") for condition in goal_conditions} == {
        'KET4_Square_4mm'
    }
    assert {str(condition.get("kind") or "") for condition in goal_conditions} == {
        "goal_part_state",
        "goal_part_location",
    }
    serialized_prepared = json.dumps(prepared_recovery_request, sort_keys=True)
    assert "Modeled Continuation Gap" not in serialized_prepared
    assert "safety_blocked_suffix_task" not in serialized_prepared
    assert "safety_destination_occupancy" not in serialized_prepared

    context_resources = {
        str(row.get("resource_jid") or ""): row
        for row in (
            dict(prepared_recovery_request.get("context_summary") or {})
            .get("current_product_state", {})
            .get("resources", [])
        )
        if isinstance(row, dict)
    }
    assert context_resources['recovery-resource-4@localhost']["current_location"] == "3D Printing Station"
    assert context_resources['recovery-resource-4@localhost']["current_location_basis"] == "recovery_snapshot"
    assert context_resources['recovery-resource-3@localhost']["current_location"] == "Buffer For Machined parts"
    assert context_resources['recovery-resource-3@localhost']["current_location_basis"] == "recovery_snapshot"

    explicit_location, explicit_basis = planner._resolve_resource_location(
        resource_jid='recovery-resource-4@localhost',
        snapshot={"current_location": "prusa-mk3"},
        profile=object(),
        prepared_recovery_request=prepared_recovery_request,
    )
    assert explicit_location == "prusa-mk3"
    assert explicit_basis == "recovery_snapshot"

    loaded_rules = {
        str(rule.get("id") or rule.get("rule_id") or ""): rule
        for rule in (prepared_recovery_request.get("loaded_safety_rules") or [])
        if isinstance(rule, dict)
    }
    safe_1_resources = {
        str(dict(ap.get("selector") or {}).get("resource") or "")
        for ap in (dict(loaded_rules["SAFE_1"]).get("recovery_aps") or [])
        if isinstance(ap, dict)
    }
    safe_2_resources = {
        str(dict(ap.get("selector") or {}).get("resource") or "")
        for ap in (dict(loaded_rules["SAFE_2"]).get("recovery_aps") or [])
        if isinstance(ap, dict)
    }
    assert safe_1_resources == {"any"}
    assert {'recovery-resource-4', 'recovery-resource-3'}.issubset(safe_2_resources)

    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])
    grounding_prompt = multi_turn_prompts.render_multi_turn_phase_prompt(
        multi_turn_prompts.build_multi_turn_phase_prompt_input(
            phase="grounding",
            llm_input=prepared_recovery_request["llm_input"],
            session_state=session_state,
            recovery_resources=prepared_recovery_request["recovery_resources"],
        )
    )
    assert "Assembly Requirements" not in grounding_prompt
    assert "pending_nominal_tasks" not in grounding_prompt
    assert "LG by xarm6" not in grounding_prompt
    assert "MCP by ur5e" not in grounding_prompt
    assert (
        "KET4_Square_4mm place_approach must occur before gear_large place_approach to assembly_board-v1."
        in grounding_prompt
    )

    session_state["current_phase"] = "outline"
    outline_prompt_input, outline_prompt = multi_turn_mode._build_phase_prompt(
        prepared_recovery_request,
        session_state,
    )
    assert {
        str(blocker.get("kind") or "")
        for blocker in (outline_prompt_input.get("current_recovery_blockers") or [])
    } == {"goal_part_state", "goal_part_location"}
    assert (
        'recovery-resource-3@localhost currently occupies assembly_board-v1 under SAFE_2'
        not in outline_prompt
    )
    assert "Modeled Continuation Gap" not in outline_prompt
    assert '"resource_location": "Buffer For Machined parts"' in outline_prompt
    assert '"resource_location": "3D Printing Station"' in outline_prompt
    assert (
        "SAFE_2: The two robots must not occupy the assembly destination together."
        in outline_prompt
    )
    assert 'restore KET4_Square_4mm to assembly_board-v1' in outline_prompt
    assert '"origin_location"' not in outline_prompt
    assert 'reachability=["Buffer For Machined parts", "assembly_board-v1", "Exit"]' in (
        outline_prompt
    )
    assert "Location order is lexical and does not express a preference." not in (
        outline_prompt
    )
    assert "named poses" not in outline_prompt
    resource_vocabulary = outline_prompt.split("Resource States", maxsplit=1)[1].split(
        "Part States", maxsplit=1
    )[0]
    part_vocabulary = outline_prompt.split("Part States", maxsplit=1)[1].split(
        "Resource Fields", maxsplit=1
    )[0]
    vocabulary = resource_vocabulary + part_vocabulary
    current_des_state = outline_prompt.split("Current DES State", maxsplit=1)[1]
    assert "held_part:" not in vocabulary
    assert '"gear_large"' not in vocabulary
    assert '"held_part": "gear_large"' in current_des_state
    assert '"part_name": "gear_large"' in current_des_state
    assert '"part_name": "KET4_Square_4mm"' in current_des_state
    assert "any" not in vocabulary
    assert '"failed"' not in resource_vocabulary
    assert '"misplaced"' not in part_vocabulary
    assert '"resource_state": "failed"' in current_des_state
    assert '"part_state": "misplaced"' in current_des_state
    assert '"goal_location"' not in outline_prompt
    assert '"goal_requirement_id"' not in outline_prompt
    for line in vocabulary.splitlines():
        if "resource_state:" in line or "part_state:" in line:
            assert "null" not in line

    session_state["symbolic_parts"]['KET4_Square_4mm'].update(
        {
            "current_state": "assembled",
            "part_state": "assembled",
            "current_location": "assembly_board-v1",
            "part_location": "assembly_board-v1",
            "current_holder_resource_jid": 'recovery-resource-4@localhost',
            "part_holder_resource_jid": 'recovery-resource-4@localhost',
        }
    )
    remaining_findings, remaining_conditions = multi_turn_mode._remaining_blocked_issue_counts(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert remaining_findings == 0
    assert remaining_conditions == 0
    assert session_state["symbolic_resources"]['recovery-resource-3@localhost']["current_state"] == "failed"


def test_non_case3_prompt_contains_only_supplied_runtime_facts() -> None:
    prompt = _render_non_case3_candidate_prompt()

    for supplied_token in (
        "ur5e@localhost",
        "xarm6@localhost",
        "LCP",
        "MG",
        "station",
        "Assembly Station",
        "prusa-mk4-1",
        "prusa-mk4-2",
        "SAFE_1",
        "LCP place_approach must occur before MG place_approach to station.",
    ):
        assert supplied_token in prompt
    for leaked_token in (
        "LG",
        "MCP",
        "REQ_2_T4",
        "recover_to_home_idle",
        "place_mcp_to_prusa_mk4_2_temp",
        "pick_lg_from_observed_pose",
        "place_lg_to_assembly_board_v1",
    ):
        assert leaked_token not in prompt

    assert "LCP by xarm6@localhost" not in prompt
    assert "MG by ur5e@localhost" not in prompt
    assert "Open Guard / Marking Conditions" not in prompt
    assert "Grounded Event Facts" not in prompt
    assert "Marked-State Conditions" not in prompt
    assert "Assembly Requirements" not in prompt
    assert "Candidate Count:" not in prompt
    assert "Action Horizon:" not in prompt
    assert "Selection Mode:" not in prompt
    assert "current_pose(x=" not in prompt
    assert prompt.count('"x": -0.25') == 1
    assert "Recovery Goals" in prompt
    assert "restore LCP to station" in prompt
    assert "restore MG" not in prompt


def test_candidate_prompt_resource_and_part_order_is_stable() -> None:
    assert _render_non_case3_candidate_prompt() == _render_non_case3_candidate_prompt(
        reverse_order=True
    )


def test_acquire_and_release_propagate_part_holder_and_location() -> None:
    acquire = {
        "resource_jid": "xarm6@localhost",
        "part_name": "LCP",
        "expected_end_state": {
            "resource_state": "picked",
            "held_part": "LCP",
            "part_state": "in_gripper",
            "part_location": "xarm6@localhost",
        },
    }
    release = {
        "resource_jid": "xarm6@localhost",
        "part_name": "LCP",
        "expected_end_state": {
            "resource_state": "positioned",
            "held_part": None,
            "part_state": "assembled",
            "part_location": "station",
        },
    }
    session_state = {
        "symbolic_resources": {
            "xarm6@localhost": {
                "resource_jid": "xarm6@localhost",
                "current_state": "idle",
                "held_part": None,
            }
        },
        "symbolic_parts": {
            "LCP": {
                "part_name": "LCP",
                "current_state": "misplaced",
                "current_location": "prusa-mk4-1",
                "current_holder_resource_jid": None,
            }
        },
    }

    multi_turn_mode._apply_task_effects_to_symbolic_state(acquire, session_state)
    acquired_part = session_state["symbolic_parts"]["LCP"]
    assert acquired_part["part_holder_resource_jid"] == "xarm6@localhost"
    assert acquired_part["current_holder_resource_jid"] == "xarm6@localhost"
    assert acquired_part["part_location"] == "xarm6@localhost"
    assert acquired_part["current_location"] == "xarm6@localhost"

    multi_turn_mode._apply_task_effects_to_symbolic_state(release, session_state)
    released_part = session_state["symbolic_parts"]["LCP"]
    assert released_part["part_holder_resource_jid"] is None
    assert released_part["current_holder_resource_jid"] is None
    assert released_part["part_location"] == "station"
    assert released_part["current_location"] == "station"

    projected_resources = deepcopy(session_state["symbolic_resources"])
    projected_parts = {
        "LCP": {
            "part_name": "LCP",
            "current_state": "misplaced",
            "current_location": "prusa-mk4-1",
            "current_holder_resource_jid": None,
        }
    }
    multi_turn_outline_state._apply_outline_task_effects(
        acquire,
        resources_by_jid=projected_resources,
        parts_by_name=projected_parts,
        task_type="part_handling",
    )
    assert projected_parts["LCP"]["current_location"] == "xarm6@localhost"
    multi_turn_outline_state._apply_outline_task_effects(
        release,
        resources_by_jid=projected_resources,
        parts_by_name=projected_parts,
        task_type="part_handling",
    )
    assert projected_parts["LCP"]["current_holder_resource_jid"] is None
    assert projected_parts["LCP"]["current_location"] == "station"


def test_pa_custody_propagation_does_not_infer_resource_mechanism_or_location() -> None:
    acquire = {
        "resource_jid": "handler@localhost",
        "part_name": "LCP",
        "expected_end_state": {
            "resource_state": "loaded",
            "held_part": "LCP",
            "part_state": "controlled",
        },
    }
    session_state = {
        "recovery_des_models": {
            "handler@localhost": {
                "state_variables": {
                    "resource_state": {"scope": "resource"},
                    "held_part": {"scope": "resource"},
                    "part_state": {"scope": "part"},
                    "part_location": {"scope": "part"},
                }
            }
        },
        "symbolic_resources": {
            "handler@localhost": {
                "resource_jid": "handler@localhost",
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            }
        },
        "symbolic_parts": {
            "LCP": {
                "part_name": "LCP",
                "current_state": "available",
                "current_location": "station",
                "part_location": "station",
                "current_holder_resource_jid": None,
            }
        },
    }

    multi_turn_mode._apply_task_effects_to_symbolic_state(acquire, session_state)

    resource = session_state["symbolic_resources"]["handler@localhost"]
    part = session_state["symbolic_parts"]["LCP"]
    assert resource["held_part"] == "LCP"
    assert resource["gripper_state"] == "open"
    assert part["current_holder_resource_jid"] == "handler@localhost"
    assert part["part_location"] == "station"
    assert part["current_location"] == "station"

    projected_resources = {
        "handler@localhost": {
            "resource_jid": "handler@localhost",
            "current_state": "idle",
            "held_part": None,
            "gripper_state": "open",
        }
    }
    projected_parts = {
        "LCP": {
            "part_name": "LCP",
            "current_state": "available",
            "current_location": "station",
            "part_location": "station",
            "current_holder_resource_jid": None,
        }
    }
    multi_turn_outline_state._apply_outline_task_effects(
        acquire,
        resources_by_jid=projected_resources,
        parts_by_name=projected_parts,
        task_type="part_handling",
        state_field_scopes={
            "resource_state": "resource",
            "held_part": "resource",
            "part_state": "part",
            "part_location": "part",
        },
    )
    assert projected_resources["handler@localhost"]["gripper_state"] == "open"
    assert projected_parts["LCP"]["current_holder_resource_jid"] == (
        "handler@localhost"
    )
    assert projected_parts["LCP"]["part_location"] == "station"


def test_candidate_completeness_uses_responsible_ra_state_variables() -> None:
    common = {
        "outline_id": "resource_specific_fields",
        "event_name": "authored_event",
        "resource_jid": "resource@localhost",
        "part_name": "LCP",
    }
    resource_only_states = {
        "resource_state": {"scope": "resource", "domain": ["idle", "ready"]}
    }
    findings = multi_turn_mode._candidate_state_completeness_findings(
        candidate_task=common,
        part_name="LCP",
        end_state={"resource_state": "ready"},
        state_variables=resource_only_states,
    )
    assert findings == []

    custody_states = {
        **resource_only_states,
        "held_part": {"scope": "resource", "domain": [None]},
        "part_state": {
            "scope": "part",
            "domain": [None, "available", "controlled"],
        },
        "part_location": {"scope": "part", "domain": ["station"]},
        "part_quality": {"scope": "part", "domain": ["unknown", "accepted"]},
    }
    findings = multi_turn_mode._candidate_state_completeness_findings(
        candidate_task=common,
        part_name="LCP",
        end_state={
            "resource_state": "ready",
            "held_part": "LCP",
            "part_state": "controlled",
        },
        state_variables=custody_states,
    )
    assert findings[0]["constraint_code"] == "candidate_schema_violation"
    assert findings[0]["evidence"]["missing"] == ["part_location"]

    response_schema = multi_turn_prompts.multi_turn_phase_response_schema(
        "outline",
        outline_mode="incremental_candidates_validated",
        recovery_selection_mode="neurosymbolic",
        declared_state_variables=custody_states,
    )
    assert response_schema["strict"] is False
    serialized_schema = json.dumps(response_schema["schema"], sort_keys=True)
    for unsupported_keyword in ('"allOf"', '"if"', '"then"', '"else"', '"not"'):
        assert unsupported_keyword not in serialized_schema
    definitions = response_schema["schema"]["$defs"]
    outline_event_schema = definitions["outline_event"]
    branch_names = [
        branch["$ref"].rsplit("/", maxsplit=1)[-1]
        for branch in outline_event_schema["anyOf"]
    ]
    assert branch_names == [
        "resource_only_outline_event",
        "part_outline_event",
        "custody_outline_event",
    ]

    resource_event = definitions["resource_only_outline_event"]
    part_event = definitions["part_outline_event"]
    custody_event = definitions["custody_outline_event"]
    for event_schema in (resource_event, part_event, custody_event):
        assert event_schema["additionalProperties"] is False
        assert "expected_start_state" not in event_schema["properties"]
    assert "part_name" not in resource_event["properties"]
    assert "part_name" in part_event["required"]
    assert "part_name" in custody_event["required"]

    resource_state_schema = definitions["resource_only_outline_state"]
    part_state_schema = definitions["part_outline_state"]
    custody_state_schema = definitions["custody_outline_state"]
    for state_schema in (
        resource_state_schema,
        part_state_schema,
        custody_state_schema,
    ):
        assert state_schema["additionalProperties"] is False
    assert set(resource_state_schema["properties"]) == {
        "resource_state",
        "resource_location",
    }
    assert "held_part" not in part_state_schema["properties"]
    assert {"part_state", "part_location", "part_quality"}.issubset(
        part_state_schema["properties"]
    )
    assert custody_state_schema["required"] == [
        "resource_state",
        "held_part",
        "part_state",
        "part_location",
    ]
    assert custody_state_schema["properties"]["resource_state"] == {
        "type": "string",
        "minLength": 1,
    }
    assert custody_state_schema["properties"]["part_state"] == {
        "type": "string",
        "minLength": 1,
    }
    assert custody_state_schema["properties"]["held_part"] == {
        "type": ["string", "null"],
        "minLength": 1,
    }


def test_candidate_schema_separates_resource_part_and_custody_shapes() -> None:
    response_schema = multi_turn_prompts.multi_turn_phase_response_schema(
        "outline",
        outline_mode="incremental_candidates_validated",
        recovery_selection_mode="neurosymbolic",
        declared_state_variables={
            "resource_state": {"scope": "resource", "domain": ["idle", "ready"]},
            "resource_location": {"scope": "resource", "domain": ["home"]},
            "held_part": {"scope": "resource", "domain": [None, "LG"]},
            "part_state": {
                "scope": "part",
                "domain": ["available", "in_gripper"],
            },
            "part_location": {
                "scope": "part",
                "domain": ["station", "resource@localhost"],
            },
        },
    )["schema"]
    Draft202012Validator.check_schema(response_schema)
    validator = Draft202012Validator(response_schema)

    def response(candidate: dict[str, Any]) -> dict[str, Any]:
        return {"thought": "propose one transition", "candidate_events": [candidate]}

    common = {
        "outline_id": "candidate_1",
        "event_name": "recover_to_home",
        "resource_jid": "resource@localhost",
        "rationale": "recover the resource",
    }
    resource_only = {
        **common,
        "expected_end_state": {
            "resource_state": "ready",
            "resource_location": "home",
        },
    }
    part_without_custody = {
        **common,
        "part_name": "LG",
        "expected_end_state": {
            "resource_state": "ready",
            "part_state": "available",
        },
    }
    acquisition = {
        **common,
        "part_name": "LG",
        "expected_end_state": {
            "resource_state": "ready",
            "held_part": "LG",
            "part_state": "in_gripper",
            "part_location": "resource@localhost",
        },
    }
    release = deepcopy(acquisition)
    release["expected_end_state"] = {
        "resource_state": "ready",
        "held_part": None,
        "part_state": "available",
        "part_location": "station",
    }
    turn03_shape = {
        **common,
        "part_name": "",
        "expected_end_state": {
            "resource_state": "ready",
            "resource_location": "home",
            "held_part": None,
            "part_state": "",
            "part_location": None,
        },
    }
    incomplete_custody = deepcopy(acquisition)
    incomplete_custody["expected_end_state"].pop("part_location")

    for candidate in (resource_only, part_without_custody, acquisition, release):
        assert list(validator.iter_errors(response(candidate))) == []
    assert list(validator.iter_errors(response(turn03_shape)))
    assert list(validator.iter_errors(response(incomplete_custody)))


def test_candidate_prompt_has_no_numeric_selected_index_example() -> None:
    prompt = _render_non_case3_candidate_prompt()
    schema = multi_turn_prompts.multi_turn_phase_response_schema(
        "outline",
        outline_mode="incremental_candidates_validated",
        candidate_bound=8,
        recovery_selection_mode="pure_llm",
        action_horizon="1",
    )["schema"]

    assert '"selected_candidate_index": 0' not in prompt
    assert "integer `selected_candidate_index`" in prompt
    assert "Outline Candidate Contract" not in prompt
    assert "Recovery Candidate Rules" in prompt
    assert "`candidate_events` must contain exactly three candidates" in prompt
    assert (
        "Each candidate is one transition for one listed resource and includes"
        in prompt
    )
    assert "Without `part_name`, omit every part-scoped state field" in prompt
    assert "Use `held_part` only in a candidate branch" not in prompt
    assert "one physical action" not in prompt
    assert "Resource-only actions may include" not in prompt
    assert "selected_candidate_index" in schema["required"]
    assert schema["properties"]["selected_candidate_index"]["type"] == "integer"
    assert schema["properties"]["selected_candidate_index"]["maximum"] == 2
    assert schema["properties"]["candidate_events"]["minItems"] == 3
    assert schema["properties"]["candidate_events"]["maxItems"] == 3

    trace_prompt = _render_non_case3_candidate_prompt(action_horizon="k")
    assert "Each event in a candidate trace is one transition" in trace_prompt
    assert "Each candidate is one transition" not in trace_prompt


def test_candidate_prompt_shows_accepted_prefix_and_allows_new_event_state_symbols() -> None:
    accepted_event_name = "previously_authored_event"
    prompt = _render_non_case3_candidate_prompt(
        accepted_outline_prefix=[
            {
                "outline_id": "RECOVERY_SEQ1",
                "event_name": accepted_event_name,
                "resource_jid": "xarm6@localhost",
                "expected_start_state": {"resource_state": "failed"},
                "expected_end_state": {
                    "resource_state": "new_clear_state",
                    "resource_location": "home",
                },
            }
        ]
    )

    assert "Accepted Transition Prefix (already applied; do not repeat)" in prompt
    assert accepted_event_name in prompt
    assert '"outline_id": "RECOVERY_SEQ1"' in prompt
    accepted_history = prompt.split(
        "Accepted Transition Prefix (already applied; do not repeat)", maxsplit=1
    )[1].split("Resource Capabilities", maxsplit=1)[0]
    assert "expected_start_state" not in accepted_history
    assert "expected_end_state" not in accepted_history
    assert "new_clear_state" not in prompt
    assert "These transitions have already been applied" in prompt
    assert "`event_name` may be new" in prompt
    assert "Its intended resource condition must match a predefined successor" in prompt
    assert "requires feasible parameter assignments for every primitive in the union of all matching successor transitions" in prompt
    assert '"resource_location": "station"' in prompt
    assert "known resource locations" not in prompt
    assert "Location tokens must belong to the same RA state field" in prompt
    assert "exact current `part_location` may be preserved" in prompt
    assert "named poses" not in prompt
    assert "Outline Candidate Contract" not in prompt
    assert "Current DES State" in prompt

    initial_prompt = _render_non_case3_candidate_prompt()
    assert "Accepted Transition Prefix" not in initial_prompt


def test_candidate_prompt_keeps_ra_location_vocabulary_field_specific() -> None:
    prompt = _render_non_case3_candidate_prompt(
        recovery_des_models={
            "xarm6@localhost": {
                "state_variables": {
                    "resource_location": {
                        "scope": "resource",
                        "domain": ["home", None],
                    },
                    "part_location": {
                        "scope": "part",
                        "domain": ["assembly_board-v1", None],
                    },
                }
            }
        }
    )
    resource_fields = prompt.split("Resource Fields", maxsplit=1)[1].split(
        "Current DES State", maxsplit=1
    )[0]

    assert (
        'resource_location: {"domain": ["home"], "scope": "resource"}'
        in resource_fields
    )
    assert "part_location" not in resource_fields
    assert "Part Exact Fields" not in prompt
    assert 'part_location="assembly_board-v1"' not in prompt
    assert "\n- part_location=" not in prompt
    assert "null" not in resource_fields


def test_production_cca_rejects_safe1_out_of_order_mcp_placement() -> None:
    _fixture, _product_agent, _planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])
    session_state["symbolic_resources"]['recovery-resource-3@localhost'].update(
        {
            "current_state": "idle",
            "resource_state": "idle",
            "current_location": "home",
            "resource_location": "home",
            "occupancy": {"location": "home"},
        }
    )
    task = {
        "outline_id": "mcp_early",
        "event_name": "evt_mcp_early",
        "resource_jid": 'recovery-resource-4@localhost',
        "part_name": 'gear_large',
        "expected_start_state": {
            "resource_state": "picked",
            "held_part": 'gear_large',
            "part_state": "in_gripper",
            "part_location": 'recovery-resource-4@localhost',
        },
        "expected_end_state": {
            "resource_state": "idle",
            "resource_location": "assembly_board-v1",
            "held_part": None,
            "part_state": "assembled",
            "part_location": "assembly_board-v1",
        },
    }
    safety_input = recovery_validation_service.build_recovery_safety_validation_input(
        task=task,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    safety_result = validate_outline_macro_recovery_safety(**safety_input)

    assert safety_result["is_safe"] is False
    assert any(
        str(finding.get("rule_id") or "") == "SAFE_1"
        and str(finding.get("constraint_code") or "") == "safety_rule_violation"
        for finding in (safety_result.get("findings") or [])
    )


def test_novel_event_and_state_symbols_use_declared_effects_and_clear_safe2() -> None:
    from cais_spade_llm.resources.robot import robot_primitives, robot_profile

    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])

    # The real checkpoint is at the sources. Explicitly project an occupied
    # destination here to preserve the independent SAFE_2 regression.
    session_state["symbolic_resources"]["recovery-resource-3@localhost"].update(
        current_location="assembly_board-v1", resource_location="assembly_board-v1",
        occupancy={"location": "assembly_board-v1"},
    )

    def _validate(candidate: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        validated, schema_findings = multi_turn_mode._derive_candidate_outline_task(
            candidate_task=deepcopy(candidate),
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        assert schema_findings == []
        assert validated is not None
        findings, _grounded_action = multi_turn_mode._validate_single_outline_task(
            planner=planner,
            task=validated,
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        return validated, findings

    stage_mcp = {
        "outline_id": "novel_stage_mcp",
        "event_name": "evt_z9",
        "resource_jid": 'recovery-resource-4@localhost',
        "part_name": 'gear_large',
        "expected_end_state": {
            "resource_state": "mcp_buffer_clear",
            "held_part": None,
            "part_state": "mcp_waiting_recovery",
            "part_location": '3D Printing Station',
        },
        "rationale": "Declare a reachable release that frees the gripper.",
    }
    validated_stage, findings = _validate(stage_mcp)
    assert findings == []
    assert validated_stage["expected_start_state"] == {
        "held_part": 'gear_large',
        "part_location": 'recovery-resource-4@localhost',
        "part_state": "in_gripper",
        "resource_state": "picked",
    }
    assert robot_profile._robot_event_family(validated_stage) == "place"
    assert robot_primitives._robot_event_family(validated_stage) == "place"
    multi_turn_mode._apply_task_effects_to_symbolic_state(validated_stage, session_state)
    assert session_state["symbolic_resources"]['recovery-resource-4@localhost']["current_state"] == (
        "mcp_buffer_clear"
    )
    assert session_state["symbolic_parts"]['gear_large']["current_state"] == (
        "mcp_waiting_recovery"
    )
    session_state["symbolic_parts"]['KET4_Square_4mm']["observed_pose"] = {
        "x": 0.0,
        "y": 0.2,
        "z": 1.035,
    }

    acquire_lg = {
        "outline_id": "novel_acquire_lg",
        "event_name": "evt_n4",
        "resource_jid": 'recovery-resource-4@localhost',
        "part_name": 'KET4_Square_4mm',
        "expected_end_state": {
            "resource_state": "lg_secured",
            "held_part": 'KET4_Square_4mm',
            "part_state": "lg_under_recovery_control",
            "part_location": 'recovery-resource-4@localhost',
        },
        "rationale": "Acquire the observed part using its grounded pose.",
    }
    validated_acquire, findings = _validate(acquire_lg)
    assert findings == []
    assert validated_acquire["expected_start_state"] == {
        "held_part": None,
        "part_location": None,
        "part_state": "misplaced",
        "resource_state": "mcp_buffer_clear",
    }
    assert robot_profile._robot_event_family(validated_acquire) == "pick"
    assert robot_primitives._robot_event_family(validated_acquire) == "pick"
    multi_turn_mode._apply_task_effects_to_symbolic_state(validated_acquire, session_state)

    restore_lg = {
        "outline_id": "novel_restore_lg",
        "event_name": "evt_v2",
        "resource_jid": 'recovery-resource-4@localhost',
        "part_name": 'KET4_Square_4mm',
        "expected_end_state": {
            "resource_state": "lg_recovery_complete",
            "resource_location": "assembly_board-v1",
            "held_part": None,
            "part_state": "assembled",
            "part_location": "assembly_board-v1",
        },
        "rationale": "Restore the held part to its supplied goal location.",
    }
    validated_restore, blocked_findings = _validate(restore_lg)
    assert blocked_findings == []
    safety_input = recovery_validation_service.build_recovery_safety_validation_input(
        task=validated_restore,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    safety_result = validate_outline_macro_recovery_safety(**safety_input)
    assert any(
        str(finding.get("constraint_code") or "") == "safety_rule_violation"
        and str(finding.get("rule_id") or "") == "SAFE_2"
        for finding in (safety_result.get("findings") or [])
    )

    clear_xarm6 = {
        "outline_id": "novel_clear_xarm6",
        "event_name": "evt_q7",
        "resource_jid": 'recovery-resource-3@localhost',
        "expected_end_state": {
            "resource_state": "xarm6_clear_state",
            "resource_location": "home",
        },
        "rationale": "Leave the explicitly occupied protected destination.",
    }
    validated_clear, findings = _validate(clear_xarm6)
    assert findings == []
    assert robot_profile._robot_event_family(validated_clear) == "home"
    assert robot_primitives._robot_event_family(validated_clear) == "home"
    multi_turn_mode._apply_task_effects_to_symbolic_state(validated_clear, session_state)

    validated_restore, findings = _validate(restore_lg)
    assert findings == []
    safety_input = recovery_validation_service.build_recovery_safety_validation_input(
        task=validated_restore,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert validate_outline_macro_recovery_safety(**safety_input)["is_safe"] is True
    multi_turn_mode._apply_task_effects_to_symbolic_state(validated_restore, session_state)
    remaining_findings, remaining_conditions = multi_turn_mode._remaining_blocked_issue_counts(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert remaining_findings == 0
    assert remaining_conditions == 0
    assert session_state["symbolic_resources"]['recovery-resource-3@localhost']["current_state"] == (
        "xarm6_clear_state"
    )
    assert session_state["symbolic_parts"]['KET4_Square_4mm']["current_state"] == "assembled"

    assert "semantic_state_history" not in session_state


def test_gripper_state_is_private_and_nominal_reentry_selects_exact_origin(tmp_path) -> None:
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_outline_generation,
    )

    context = _load_json(CASE3_RESPONSE_FIXTURES / "runtime_context_reverse.json")
    snapshot = next(row for row in context["resource_snapshots"]
                    if row["resource_id"] == "ur5e-4")
    snapshot.update(current_location="assembly_board-v1",
                    occupancy={"location": "assembly_board-v1"})
    context["task_statuses"].update(REQ_2_T3="completed", REQ_2_T4="failed")
    context["failure_event"]["state_before"]["current_state"] = "positioned"
    context["part_tracker"]["gear_large"]["last_successful_task"] = "REQ_2_T3"
    peer_snapshot = next(row for row in context["resource_snapshots"] if row["resource_id"] == "ur5e-3")
    peer_snapshot.pop("current_location")
    context_path = tmp_path / "occupied_context.json"
    context_path.write_text(json.dumps(context))

    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(runtime_context_path=context_path, scripted_responses=[])
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])
    session_state["recovery_selection_mode"] = "neurosymbolic"
    session_state["candidate_count"] = "auto"
    # Explicit counterfactual occupancy preserves the CCA clearance regression.
    session_state["symbolic_resources"]["recovery-resource-4@localhost"].update(
        current_location="assembly_board-v1", resource_location="assembly_board-v1",
        occupancy={"location": "assembly_board-v1"},
    )

    clear_xarm6 = {
        "outline_id": "clear_xarm6_without_private_gripper_state",
        "event_name": "clear_board_home",
        "resource_jid": 'recovery-resource-4@localhost',
        "expected_end_state": {
            "resource_state": "idle",
            "resource_location": "home",
        },
        "rationale": "Clear the explicitly occupied destination.",
    }
    decision, clear_turn = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={
                "thought": "clear occupancy",
                "candidate_events": [clear_xarm6],
            },
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    )
    assert decision == "need_next_task"
    clearance = clear_turn["selection_evidence"]
    assert clearance["cleared_recovery_obligation_ids"] == []
    assert any(
        '"event_name":"place_insert"' in event_id
        and '"part_name":"gear_large"' in event_id
        and '"resource_jid":"recovery-resource-3@localhost"' in event_id
        for event_id in clearance["newly_cca_admissible_goal_recovery_event_ids"]
    )
    assert [
        stage["status"]
        for stage in clear_turn["candidate_evaluations"][0]["validation_stages"]
    ] == ["passed", "passed", "passed", "passed"]

    def _stage_candidate(location: str) -> dict[str, Any]:
        return {
            "outline_id": f"stage_mcp_{location}",
            "event_name": f"stage_mcp_at_{location}",
            "resource_jid": 'recovery-resource-3@localhost',
            "part_name": 'KET4_Square_4mm',
            "expected_end_state": {
                "resource_state": "idle",
                "resource_location": "home",
                "held_part": None,
                "part_state": "placed",
                "part_location": location,
            },
            "rationale": "Release MCP at one exact reachable staging location and return to home.",
        }

    stage_prusa_mk3 = _stage_candidate("Exit")
    stage_prusa_mk4_2 = _stage_candidate('Buffer For Machined parts')
    stage_prusa_mk3["event_name"] = "opaque_stage_a"
    stage_prusa_mk4_2["event_name"] = "opaque_stage_b"
    decision, selected_turn = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={
                "thought": "compare staging successors",
                "candidate_events": [stage_prusa_mk3, stage_prusa_mk4_2],
            },
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    )
    assert decision == "need_next_task"
    assert selected_turn["selected_transition"]["expected_end_state"][
        "part_location"
    ] == (
        'Buffer For Machined parts'
    )
    assert len(session_state["accepted_outline_prefix"]) == 2
    selected_evidence = selected_turn["selection_evidence"]
    assert any(
        '"task_id":"REQ_1_T1"' in event_id
        for event_id in selected_evidence[
            "admissible_nominal_reentry_event_ids_after"
        ]
    )
    assert "gripper_state" not in session_state["recovery_des_models"][
        'recovery-resource-3@localhost'
    ]["state_variables"]
    assert "gripper_state" not in selected_turn["selected_transition"][
        "expected_start_state"
    ]
    assert "gripper_state" not in selected_turn["selected_transition"][
        "expected_end_state"
    ]
    assert session_state["selection_revision_count"] == 0
    assert session_state["active_selection_ambiguity_feedback"] == {}

    _fixture, _product_agent, reverse_planner, reverse_request = asyncio.run(
        _prepare_recovery_dryrun_harness(runtime_context_path=context_path, scripted_responses=[])
    )
    reverse_session = deepcopy(reverse_request["multi_turn_session_seed"])
    reverse_session["recovery_selection_mode"] = "neurosymbolic"
    reverse_session["candidate_count"] = "auto"
    # Explicit counterfactual occupancy preserves the CCA clearance regression.
    reverse_session["symbolic_resources"]["recovery-resource-4@localhost"].update(
        current_location="assembly_board-v1", resource_location="assembly_board-v1",
        occupancy={"location": "assembly_board-v1"},
    )
    asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=reverse_session,
            parsed_response={"thought": "clear", "candidate_events": [clear_xarm6]},
            prepared_recovery_request=reverse_request,
            planner=reverse_planner,
        )
    )
    renamed_mk3 = deepcopy(stage_prusa_mk3)
    renamed_mk4_2 = deepcopy(stage_prusa_mk4_2)
    renamed_mk3["event_name"] = "renamed_x"
    renamed_mk4_2["event_name"] = "renamed_y"
    reverse_decision, reverse_turn = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=reverse_session,
            parsed_response={
                "thought": "reverse order",
                "candidate_events": [renamed_mk4_2, renamed_mk3],
            },
            prepared_recovery_request=reverse_request,
            planner=reverse_planner,
        )
    )
    assert reverse_decision == "need_next_task"
    assert reverse_turn["selected_transition"]["expected_end_state"][
        "part_location"
    ] == 'Buffer For Machined parts'

    projection_seed = deepcopy(prepared_recovery_request["multi_turn_session_seed"])
    validated_stage, schema_findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=stage_prusa_mk4_2,
        session_state=projection_seed,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert schema_findings == []
    safety_input = recovery_validation_service.build_recovery_safety_validation_input(
        task=dict(validated_stage or {}),
        session_state=projection_seed,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert "gripper_state" not in safety_input["task"]["expected_start_state"]
    assert "gripper_state" not in safety_input["task"]["expected_end_state"]


def test_case3_unvalidated_pose_and_nonprogressing_home_remain_authoritative() -> None:
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_outline_generation,
    )

    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])
    session_state["recovery_selection_mode"] = "neurosymbolic"
    session_state["candidate_count"] = "auto"
    clear_xarm6 = {
        "outline_id": "RECOVERY_SEQ1",
        "event_name": "clear_board_home",
        "resource_jid": 'recovery-resource-3@localhost',
        "expected_start_state": {
            "resource_state": "failed",
            "resource_location": "Buffer For Machined parts",
            "held_part": None,
        },
        "expected_end_state": {
            "resource_state": "home",
            "resource_location": "home",
            "held_part": None,
        },
        "rationale": "Clear the occupied board.",
    }
    session_state["accepted_outline_prefix"] = [deepcopy(clear_xarm6)]
    multi_turn_mode._apply_task_effects_to_symbolic_state(clear_xarm6, session_state)
    session_state["symbolic_parts"]['KET4_Square_4mm']["observed_pose"] = {
        "x": 0.0,
        "y": 0.2,
        "z": 1.035,
    }

    acquire_lg_with_xarm6 = {
        "outline_id": "xarm6_acquire_lg",
        "event_name": "acquire_lg",
        "resource_jid": 'recovery-resource-3@localhost',
        "part_name": 'KET4_Square_4mm',
        "expected_end_state": {
            "resource_state": "picked",
            "resource_location": "home",
            "held_part": 'KET4_Square_4mm',
            "part_state": "in_gripper",
            "part_location": 'recovery-resource-3@localhost',
        },
        "rationale": 'Attempt the observed KET4_Square_4mm pose.',
    }
    ur5e_home_with_mcp = {
        "outline_id": "ur5e_home_with_mcp",
        "event_name": "move_home_with_mcp",
        "resource_jid": 'recovery-resource-4@localhost',
        "part_name": 'gear_large',
        "expected_end_state": {
            "resource_state": "home",
            "resource_location": "home",
            "held_part": 'gear_large',
            "part_state": "in_gripper",
            "part_location": 'recovery-resource-4@localhost',
        },
        "rationale": "Retain MCP while moving to home.",
    }
    decision, turn_entry = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={
                "thought": "validate physical and symbolic alternatives",
                "candidate_events": [acquire_lg_with_xarm6, ur5e_home_with_mcp],
            },
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    )
    evaluations = turn_entry["candidate_evaluations"]
    assert evaluations[0]["validation_findings"][0]["constraint_code"] == (
        "resource_validation_unavailable"
    )
    assert evaluations[1]["valid"] is False
    assert evaluations[1]["validation_findings"][0]["constraint_code"] == (
        "unsupported_successor_condition"
    )
    assert decision == "need_revision"


def test_pa_allows_label_only_change_and_selection_rejects_no_progress() -> None:
    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])

    label_only = {
        "outline_id": "label_only",
        "event_name": "evt_l1",
        "resource_jid": 'recovery-resource-3@localhost',
        "expected_end_state": {
            "resource_state": "new_label_only",
            "resource_location": "Buffer For Machined parts",
        },
        "rationale": "Only relabel the state.",
    }
    validated, schema_findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=label_only,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert schema_findings == []
    findings, _grounded_action = multi_turn_mode._validate_single_outline_task(
        planner=planner,
        task=validated,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings == []

    decision, turn_entry = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={
                "thought": "test a state-label-only successor",
                "candidate_events": [label_only],
            },
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    )
    evaluation = turn_entry["candidate_evaluations"][0]
    assert decision == "need_revision"
    assert evaluation["valid"] is False
    assert evaluation["validation_findings"][0]["constraint_code"] == "unsupported_successor_condition"
    progressing, codes = multi_turn_outline_generation._classify_candidate_selection_progress(
        candidate={"surface_events": [validated]}, evaluation={}, progressing=False,
    )
    assert progressing is False
    assert codes == ["label_only_state_change"]
    assert [
        (stage["validation_category"], stage["status"])
        for stage in evaluation["validation_stages"]
    ] == [
        ("syntax_and_grounding_validation", "passed"),
        ("transition_feasibility", "passed"),
        ("physical_feasibility", "rejected"),
        ("safety", "skipped"),
    ]

    clear = {
        **label_only,
        "outline_id": "clear",
        "event_name": "evt_clear",
        "expected_end_state": {
            "resource_state": "clear_state",
            "resource_location": "home",
        },
    }
    validated_clear, schema_findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=clear,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert schema_findings == []
    findings, _grounded_action = multi_turn_mode._validate_single_outline_task(
        planner=planner,
        task=validated_clear,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings == []
    multi_turn_mode._apply_task_effects_to_symbolic_state(validated_clear, session_state)

    return_to_prior = {
        **label_only,
        "outline_id": "return_to_prior",
        "event_name": "evt_return",
        "expected_end_state": {
            "resource_state": "returned_state",
            "resource_location": "Buffer For Machined parts",
        },
    }
    validated_return, schema_findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=return_to_prior,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert schema_findings == []
    findings, _grounded_action = multi_turn_mode._validate_single_outline_task(
        planner=planner,
        task=validated_return,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings == []
    assert "semantic_state_history" not in session_state


def test_exact_no_op_passes_validators_and_is_excluded_by_selection() -> None:
    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])
    no_op = {
        "outline_id": "exact_no_op",
        "event_name": "exact_no_op_symbol",
        "resource_jid": 'recovery-resource-3@localhost',
        "expected_end_state": {
            "resource_state": "failed",
            "resource_location": "Buffer For Machined parts",
        },
        "rationale": "Exercise selection-level no-progress classification.",
    }

    decision, turn_entry = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={
                "thought": "test an exact no-op",
                "candidate_events": [no_op],
            },
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    )

    evaluation = turn_entry["candidate_evaluations"][0]
    assert decision == "need_revision"
    assert evaluation["valid"] is False
    assert evaluation["validation_findings"][0]["constraint_code"] == "unsupported_successor_condition"
    validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=no_op, session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings == []
    progressing, codes = multi_turn_outline_generation._classify_candidate_selection_progress(
        candidate={"surface_events": [validated]}, evaluation={}, progressing=False,
    )
    assert progressing is False
    assert codes == ["no_state_change"]
    assert [
        (stage["validation_category"], stage["status"])
        for stage in evaluation["validation_stages"]
    ] == [
        ("syntax_and_grounding_validation", "passed"),
        ("transition_feasibility", "passed"),
        ("physical_feasibility", "rejected"),
        ("safety", "skipped"),
    ]
    assert "candidate_revision_targets" not in turn_entry


def test_transition_feasibility_checks_successor_custody_and_release_location() -> None:
    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])
    session_state["symbolic_parts"]['KET4_Square_4mm']["observed_pose"] = {
        "x": 0.0,
        "y": 0.2,
        "z": 1.035,
    }
    latest_stages: list[dict[str, Any]] = []

    def _transition_findings(
        candidate: dict[str, Any],
        *,
        active_session: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        _decision, turn_entry = asyncio.run(
            multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
                session_state=deepcopy(active_session or session_state),
                parsed_response={
                    "thought": "exercise ResourceAgent transition consistency",
                    "candidate_events": [deepcopy(candidate)],
                },
                prepared_recovery_request=prepared_recovery_request,
                planner=planner,
            )
        )
        stages = turn_entry["candidate_evaluations"][0]["validation_stages"]
        latest_stages[:] = stages
        pa_stage = next(
            stage
            for stage in stages
            if stage["validation_category"] == "syntax_and_grounding_validation"
        )
        assert pa_stage["validator_role"] == "PA"
        assert pa_stage["findings"] == []
        transition_stage = next(
            stage
            for stage in stages
            if stage["validation_category"] == "transition_feasibility"
        )
        assert transition_stage["validator_role"] == "RA"
        return list(transition_stage["findings"])

    release = {
        "outline_id": "release_with_explicit_successor",
        "event_name": "authored_release",
        "resource_jid": 'recovery-resource-4@localhost',
        "part_name": 'gear_large',
        "expected_end_state": {
            "resource_state": "released_state",
            "held_part": None,
            "part_state": "released_state",
            "part_location": None,
        },
        "rationale": "Release the held part.",
    }
    findings = _transition_findings(release)
    assert findings[0]["constraint_code"] == "part_traceability_violation"
    assert findings[0]["validation_category"] == "transition_feasibility"
    assert findings[0]["invariant_id"] == "part_traceability"
    assert next(
        stage
        for stage in latest_stages
        if stage["validation_category"] == "physical_feasibility"
    )["status"] == "skipped"
    assert next(
        stage
        for stage in latest_stages
        if stage["validation_category"] == "safety"
    )["status"] == "skipped"

    release["expected_end_state"]["part_location"] = '3D Printing Station'
    assert _transition_findings(release) == []

    direct_relocation = {
        "outline_id": "direct_relocation_without_custody",
        "event_name": "authored_direct_relocation",
        "resource_jid": 'recovery-resource-3@localhost',
        "part_name": 'KET4_Square_4mm',
        "expected_end_state": {
            "resource_state": "failed",
            "held_part": None,
            "part_state": "restored",
            "part_location": "assembly_board-v1",
        },
        "rationale": "Relocate without declaring custody.",
    }
    assert _transition_findings(direct_relocation) == []

    acquire = {
        "outline_id": "acquire_with_explicit_successor",
        "event_name": "authored_acquisition",
        "resource_jid": 'recovery-resource-3@localhost',
        "part_name": 'KET4_Square_4mm',
        "expected_end_state": {
            "resource_state": "controlled_state",
            "held_part": 'KET4_Square_4mm',
            "part_state": "controlled_state",
            "part_location": None,
        },
        "rationale": "Acquire the affected part.",
    }
    findings = _transition_findings(acquire)
    assert findings[0]["constraint_code"] == "held_part_location_mismatch"
    assert findings[0]["validation_category"] == "transition_feasibility"
    assert findings[0]["invariant_id"] == "part_traceability"

    acquire["expected_end_state"]["part_location"] = 'Buffer For Machined parts'
    findings = _transition_findings(acquire)
    assert findings[0]["constraint_code"] == "held_part_location_mismatch"
    assert findings[0]["validation_category"] == "transition_feasibility"
    assert findings[0]["invariant_id"] == "part_traceability"
    assert findings[0]["evidence"] == {
        "field": "expected_end_state.part_location",
        "proposed_part_location": 'Buffer For Machined parts',
        "expected_carried_part_location": 'recovery-resource-3@localhost',
    }

    acquire["expected_end_state"]["part_location"] = 'recovery-resource-3@localhost_gripper'
    findings = _transition_findings(acquire)
    assert findings[0]["constraint_code"] == "unknown_location_token"
    assert findings[0]["validation_category"] == "transition_feasibility"

    acquire["expected_end_state"]["part_location"] = 'recovery-resource-3@localhost'
    assert _transition_findings(acquire) == []

    held_transport = deepcopy(acquire)
    held_transport["outline_id"] = "transport_while_holding"
    held_transport["expected_end_state"] = {
        "resource_state": "approaching_destination",
        "resource_location": 'Buffer For Machined parts',
        "held_part": 'KET4_Square_4mm',
        "part_state": "controlled_state",
        "part_location": "assembly_board-v1",
    }
    held_session = deepcopy(session_state)
    multi_turn_mode._apply_task_effects_to_symbolic_state(acquire, held_session)
    held_session["accepted_outline_prefix"] = [deepcopy(acquire)]

    findings = _transition_findings(held_transport, active_session=held_session)
    assert findings[0]["constraint_code"] == "held_part_location_mismatch"
    held_transport["expected_end_state"]["part_location"] = 'recovery-resource-3@localhost'
    assert _transition_findings(held_transport, active_session=held_session) == []

    duplicate_holder = deepcopy(acquire)
    duplicate_holder["outline_id"] = "duplicate_mcp_holder"
    duplicate_holder["part_name"] = 'gear_large'
    duplicate_holder["expected_end_state"]["held_part"] = 'gear_large'
    duplicate_holder["expected_end_state"]["part_location"] = 'recovery-resource-3@localhost'
    findings = _transition_findings(duplicate_holder)
    assert findings[0]["constraint_code"] == "part_traceability_violation"
    assert findings[0]["invariant_id"] == "part_traceability"


def test_pa_rejects_malformed_candidate_without_contacting_ra() -> None:
    _fixture, product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_seed = deepcopy(prepared_recovery_request["multi_turn_session_seed"])
    ra_calls = 0
    request_ra = product_agent.request_recovery_outline_physical_validation

    async def _count_ra_calls(**kwargs: Any) -> dict[str, Any]:
        nonlocal ra_calls
        ra_calls += 1
        return await request_ra(**kwargs)

    product_agent.request_recovery_outline_physical_validation = _count_ra_calls
    common_candidate = {
        "event_name": "generated_event_name",
        "resource_jid": 'recovery-resource-3@localhost',
        "expected_end_state": {"resource_state": "generated_resource_state"},
        "rationale": "Submit one malformed candidate.",
    }
    malformed_candidates = [
        {
            **common_candidate,
            "outline_id": "custody_without_part_name",
            "expected_end_state": {
                "resource_state": "generated_resource_state",
                "held_part": 'KET4_Square_4mm',
            },
        },
        {**common_candidate, "outline_id": "empty_part_name", "part_name": ""},
        {
            **common_candidate,
            "outline_id": "whitespace_part_name",
            "part_name": "   ",
        },
        {
            **common_candidate,
            "outline_id": "padded_part_name",
            "part_name": ' KET4_Square_4mm ',
        },
        {
            **common_candidate,
            "outline_id": "non_string_part_name",
            "part_name": 7,
        },
    ]

    for candidate in malformed_candidates:
        decision, turn_entry = asyncio.run(
            multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
                session_state=deepcopy(session_seed),
                parsed_response={
                    "thought": "submit one malformed candidate",
                    "candidate_events": [candidate],
                },
                prepared_recovery_request=prepared_recovery_request,
                planner=planner,
            )
        )

        assert decision == "need_revision"
        stages = turn_entry["candidate_evaluations"][0]["validation_stages"]
        assert [stage["status"] for stage in stages] == [
            "rejected",
            "skipped",
            "skipped",
            "skipped",
        ]
        assert [stage["validator_role"] for stage in stages] == [
            "PA",
            "RA",
            "RA",
            "CCA",
        ]
        finding = stages[0]["findings"][0]
        assert finding["constraint_code"] == "candidate_schema_violation"
        if candidate["outline_id"] == "custody_without_part_name":
            assert finding["evidence"]["unexpected"] == ["held_part"]
        else:
            assert finding["part_name"] is None
            assert finding["evidence"]["field"] == "part_name"
            assert finding["evidence"]["value"] == candidate["part_name"]

    assert ra_calls == 0


def test_held_part_location_mismatch_skips_ra_cca_and_returns_scoped_feedback() -> None:
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_outline_generation,
    )

    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    mismatch_session = deepcopy(
        prepared_recovery_request["multi_turn_session_seed"]
    )
    mismatch_session["recovery_selection_mode"] = "neurosymbolic"
    mismatched_acquisition = {
        "outline_id": "live_shaped_acquisition_mismatch",
        "event_name": "recover_pick",
        "resource_jid": 'recovery-resource-3@localhost',
        "part_name": 'KET4_Square_4mm',
        "expected_end_state": {
            "resource_state": "controlled_state",
            "held_part": 'KET4_Square_4mm',
            "part_state": "controlled_state",
            "part_location": 'Buffer For Machined parts',
        },
        "rationale": "Attempt a grounded acquisition.",
    }
    decision, mismatch_turn = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=mismatch_session,
            parsed_response={
                "thought": "attempt a grounded acquisition",
                "candidate_events": [mismatched_acquisition],
            },
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    )
    assert decision == "need_revision"
    stages = mismatch_turn["candidate_evaluations"][0]["validation_stages"]
    transition_stage = next(
        stage
        for stage in stages
        if stage["validation_category"] == "transition_feasibility"
    )
    assert transition_stage["status"] == "rejected"
    assert transition_stage["findings"][0]["constraint_code"] == (
        "held_part_location_mismatch"
    )
    assert next(
        stage
        for stage in stages
        if stage["validation_category"] == "physical_feasibility"
    )["status"] == "skipped"
    assert next(
        stage for stage in stages if stage["validation_category"] == "safety"
    )["status"] == "skipped"

    mismatch_session["current_phase"] = "outline"
    _prompt_input, mismatch_prompt = multi_turn_mode._build_phase_prompt(
        prepared_recovery_request,
        mismatch_session,
    )
    assert "held_part_location_mismatch" in mismatch_prompt
    assert "expected_carried_part_location" in mismatch_prompt
    assert '"proposed_part_location": "Buffer For Machined parts"' in mismatch_prompt
    assert '"expected_carried_part_location": "recovery-resource-3@localhost"' in mismatch_prompt
    assert "held_part and part_location do not describe the same ResourceAgent transition" in (
        mismatch_prompt
    )
    assert "resource and part custody facts disagree" not in mismatch_prompt
    assert "recover_pick" not in mismatch_prompt
    assert "Candidate Revision Targets" not in mismatch_prompt
    assert "one materially revised candidate for every listed target" not in mismatch_prompt


def test_novel_states_do_not_allow_invented_resources_parts_or_locations() -> None:
    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])

    unknown_resource = {
        "outline_id": "unknown_resource",
        "event_name": "evt_u1",
        "resource_jid": "new_robot@localhost",
        "expected_end_state": {
            "resource_state": "new_state",
            "resource_location": "home",
        },
        "rationale": "Invalid resource binding.",
    }
    _validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=unknown_resource,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert "unknown resource" in findings[0]["reason"]

    unknown_part = {
        "outline_id": "unknown_part",
        "event_name": "evt_u2",
        "resource_jid": 'recovery-resource-4@localhost',
        "part_name": "NEW_PART",
        "expected_end_state": {
            "resource_state": "new_state",
            "held_part": None,
            "part_state": "new_part_state",
            "part_location": '3D Printing Station',
        },
        "rationale": "Invalid part binding.",
    }
    _validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=unknown_part,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert "unknown part" in findings[0]["reason"]

    unknown_location = {
        "outline_id": "unknown_location",
        "event_name": "evt_u3",
        "resource_jid": 'recovery-resource-4@localhost',
        "part_name": 'gear_large',
        "expected_end_state": {
            "resource_state": "new_state",
            "held_part": None,
            "part_state": "new_part_state",
            "part_location": "invented_location",
        },
        "rationale": "Invalid location binding.",
    }
    validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=unknown_location,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings == []
    decision, turn_entry = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={
                "thought": "exercise ResourceAgent location consistency",
                "candidate_events": [unknown_location],
            },
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    )
    assert decision == "need_revision"
    transition_stage = next(
        stage
        for stage in turn_entry["candidate_evaluations"][0]["validation_stages"]
        if stage["validation_category"] == "transition_feasibility"
    )
    assert transition_stage["validator_role"] == "RA"
    assert transition_stage["findings"][0]["constraint_code"] == (
        "unknown_location_token"
    )


def test_llm_resource_only_candidate_omits_held_part_and_part_fields_require_part_name() -> None:
    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])

    resource_only = {
        "outline_id": "resource_only_clear_xarm6",
        "event_name": "evt_resource_only_clear",
        "resource_jid": 'recovery-resource-3@localhost',
        "expected_end_state": {
            "resource_state": "xarm6_clear_state",
            "resource_location": "home",
        },
        "rationale": "Clear the occupied destination without changing any part state.",
    }
    validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=resource_only,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings == []
    assert validated is not None
    assert set(validated["expected_start_state"]) == {
        "resource_location",
        "resource_state",
    }
    resource_only_with_null = deepcopy(resource_only)
    resource_only_with_null["part_name"] = None
    validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=resource_only_with_null,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings == []
    assert validated is not None
    assert "part_name" not in validated
    _decision, resource_turn_entry = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=deepcopy(session_state),
            parsed_response={
                "thought": "clear the resource",
                "candidate_events": [resource_only_with_null],
            },
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    )
    resource_stages = resource_turn_entry["candidate_evaluations"][0][
        "validation_stages"
    ]
    assert resource_stages[0]["validator_role"] == "PA"
    assert resource_stages[0]["status"] == "passed"
    assert resource_stages[1]["validator_role"] == "RA"
    assert resource_stages[1]["validation_category"] == "transition_feasibility"
    assert resource_stages[1]["status"] == "passed"

    empty_part_name = deepcopy(resource_only)
    empty_part_name["part_name"] = ""
    _validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=empty_part_name,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings
    assert "exact supplied nonempty string token" in str(
        findings[0].get("reason") or ""
    )

    ambiguous_named_pose = deepcopy(resource_only)
    ambiguous_named_pose["outline_id"] = "resource_only_ambiguous_home"
    ambiguous_named_pose["expected_end_state"] = {
        "resource_state": "home",
        "resource_location": "assembly_board-v1",
    }
    validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=ambiguous_named_pose,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings == []
    assert validated is not None
    _decision, turn_entry = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=deepcopy(session_state),
            parsed_response={
                "thought": "exercise ResourceAgent named-pose consistency",
                "candidate_events": [ambiguous_named_pose],
            },
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    )
    transition_stage = next(
        stage
        for stage in turn_entry["candidate_evaluations"][0]["validation_stages"]
        if stage["validation_category"] == "transition_feasibility"
    )
    assert transition_stage["validator_role"] == "RA"
    assert transition_stage["status"] == "rejected"
    assert "resource location do not match" in str(
        transition_stage["findings"][0].get("reason") or ""
    )

    missing_part_name = {
        "outline_id": "missing_part_name",
        "event_name": "evt_missing_part_name",
        "resource_jid": 'recovery-resource-4@localhost',
        "expected_end_state": {
            "resource_state": "mcp_buffer_clear",
            "held_part": None,
            "part_state": "mcp_waiting_recovery",
            "part_location": '3D Printing Station',
        },
        "rationale": "Invalid because part fields require part_name.",
    }
    _validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=missing_part_name,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings
    reason = str(findings[0].get("reason") or "")
    assert "part_name" in reason
    assert "part_state" in reason
    assert "part_location" in reason
    assert "held_part" in reason

    inconsistent_held_part = {
        "outline_id": "inconsistent_held_part",
        "event_name": "evt_inconsistent_held_part",
        "resource_jid": 'recovery-resource-4@localhost',
        "part_name": 'KET4_Square_4mm',
        "expected_end_state": {
            "resource_state": "invalid_hold",
            "held_part": 'gear_large',
            "part_state": "lg_under_recovery_control",
            "part_location": 'recovery-resource-4@localhost',
        },
        "rationale": "Invalid because held_part contradicts part_name.",
    }
    validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=inconsistent_held_part,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings == []
    assert validated is not None
    _decision, turn_entry = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=deepcopy(session_state),
            parsed_response={
                "thought": "exercise ResourceAgent custody consistency",
                "candidate_events": [inconsistent_held_part],
            },
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    )
    transition_stage = next(
        stage
        for stage in turn_entry["candidate_evaluations"][0]["validation_stages"]
        if stage["validation_category"] == "transition_feasibility"
    )
    assert transition_stage["validator_role"] == "RA"
    assert transition_stage["status"] == "rejected"
    assert "contradicts the candidate part_name" in str(
        transition_stage["findings"][0].get("reason") or ""
    )


def test_grounding_schema_contains_only_structural_decision_fields() -> None:
    schema = multi_turn_prompts.multi_turn_phase_response_schema("grounding")["schema"]

    assert schema["required"] == ["thought", "decision", "observe_requests"]
    assert set(schema["properties"]) == {"thought", "decision", "observe_requests"}
    assert schema["additionalProperties"] is False
    request_schema = schema["properties"]["observe_requests"]["items"]
    assert set(request_schema["properties"]) == {"fact_type", "entity", "reason"}
    assert request_schema["required"] == ["fact_type", "entity"]


def test_production_structured_request_records_actual_messages() -> None:
    calls: list[dict[str, Any]] = []

    def _create(**kwargs: Any) -> Any:
        calls.append(deepcopy(kwargs))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "thought": "enough context",
                                "decision": "grounded",
                                "observe_requests": [],
                            }
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )

    async def _direct_to_thread(
        func: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        return func(*args, **kwargs)

    agent = object.__new__(llm_agent_module.LlmAgent)
    agent.model = "test-model"
    agent.reasoning_effort = "medium"
    agent.instructions = "generic product system instructions"
    response_format = multi_turn_prompts.multi_turn_phase_response_schema("grounding")

    with (
        patch.object(llm_agent_module, "_client", fake_client),
        patch.object(llm_agent_module.asyncio, "to_thread", new=_direct_to_thread),
    ):
        parsed = asyncio.run(
            llm_agent_module.LlmAgent.ask_llm_structured(
                agent,
                "grounding user prompt",
                response_format=response_format,
            )
        )

    assert parsed["decision"] == "grounded"
    assert calls
    recorded_request = agent._last_structured_request
    assert recorded_request["model"] == "test-model"
    assert recorded_request["messages"] == [
        {"role": "system", "content": "generic product system instructions"},
        {"role": "user", "content": "grounding user prompt"},
    ]
    assert calls[0]["messages"] == recorded_request["messages"]
    assert calls[0]["response_format"] == recorded_request["response_format"]

    calls.clear()
    with (
        patch.object(llm_agent_module, "_client", fake_client),
        patch.object(llm_agent_module.asyncio, "to_thread", new=_direct_to_thread),
    ):
        parsed = asyncio.run(
            llm_agent_module.LlmAgent.ask_llm_structured(
                agent,
                "recovery user prompt",
                response_format=response_format,
                include_agent_instructions=False,
            )
        )

    assert parsed["decision"] == "grounded"
    assert agent._last_structured_request["messages"] == [
        {"role": "user", "content": "recovery user prompt"}
    ]
    assert calls[0]["messages"] == agent._last_structured_request["messages"]
    assert "tools" not in agent._last_structured_request


def test_recovery_safety_never_calls_llm_or_accepts_legacy_empty_rules(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    agent = SimpleNamespace(ask_llm_structured=AsyncMock(
        side_effect=AssertionError("Recovery cannot select its own safety rules")))
    with pytest.raises(ValueError, match="recompile"):
        asyncio.run(generate_recovery_safety_bundle(agent, {
            "recovery_safety_dir": str(tmp_path), "recovery_safety_scope_id": "test_scope",
            "loaded_safety_rules": [],
        }))
    agent.ask_llm_structured.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_candidate_validation_feedback_is_rendered_once() -> None:
    prompt = _render_non_case3_candidate_prompt(
        candidate_rejection_feedback=[
            {
                "candidate_index": 0,
                "task": {
                    "outline_id": "candidate_1",
                    "resource_jid": "xarm6@localhost",
                },
                "validation_findings": [
                    {
                        "stage": "syntax_and_grounding_validation",
                        "constraint_code": "candidate_schema_violation",
                        "reason": "candidate contains an unsupported field",
                    }
                ],
            }
        ]
    )

    assert prompt.count("candidate_schema_violation") == 1
    assert prompt.count("Validation Feedback") == 1
    assert "Disabled And Blocked Candidate Events" not in prompt


def test_ra_feedback_preserves_exact_constraint_details() -> None:
    reason = "Task releases 'LCP' without specifying a concrete grounded destination."
    prompt = _render_non_case3_candidate_prompt(
        candidate_rejection_feedback=[
            {
                "candidate_index": 0,
                "task": {
                    "outline_id": "candidate_release",
                    "event_name": "authored_release",
                    "resource_jid": "xarm6@localhost",
                    "part_name": "LCP",
                },
                "validation_findings": [
                    {
                        "validation_category": "transition_feasibility",
                        "constraint_code": "missing_release_destination",
                        "reason": reason,
                        "resource_jid": "xarm6@localhost",
                        "part_name": "LCP",
                        "evidence": {
                            "field": "expected_end_state.part_location",
                            "proposed_part_location": None,
                        },
                    }
                ],
            }
        ]
    )

    assert "missing_release_destination" in prompt
    assert reason in prompt
    assert '"field": "expected_end_state.part_location"' in prompt
    assert '"proposed_part_location": null' in prompt
    assert "resource and part custody facts disagree" not in prompt
    assert "no_progressing_candidate" not in prompt


def test_concise_result_hides_internal_custody_codes() -> None:
    audit_payload = {
        "turn_index": 3,
        "phase": "outline",
        "decision": "need_revision",
        "candidate_evaluation_summary": [
            {
                "candidate_index": 0,
                "valid": False,
                "task": {
                    "outline_id": "custody_mismatch",
                    "event_name": "authored_event",
                    "resource_jid": "xarm6@localhost",
                    "part_name": "LG",
                    "expected_end_state": {
                        "resource_state": "picked",
                        "held_part": "LG",
                        "part_state": "in_gripper",
                        "part_location": "prusa-mk4-1",
                    },
                },
                "constraint_codes": ["held_part_location_mismatch"],
                "validation_stages": [
                    {
                        "validation_category": "transition_feasibility",
                        "validator_role": "PA",
                        "status": "rejected",
                        "findings": [
                            {
                                "constraint_code": "held_part_location_mismatch",
                                "reason": "detailed internal reason",
                            }
                        ],
                    }
                ],
            }
        ],
    }

    result = recovery_artifacts._outline_result_payload(audit_payload)

    assert "held_part_location_mismatch" not in json.dumps(result)
    stage = result["candidate_evaluation_summary"][0]["validation_stages"][0]
    assert stage == {
        "validation_category": "transition_feasibility",
        "validator_role": "PA",
        "status": "rejected",
        "reason": "resource and part custody facts disagree",
    }
    assert audit_payload["candidate_evaluation_summary"][0]["constraint_codes"] == [
        "held_part_location_mismatch"
    ]


def test_outline_request_result_and_stack_artifacts_are_complete(tmp_path: Path) -> None:
    transition_trace = [_candidate_event("better_score", outline_id="RECOVERY_SEQ1")]
    transition_trace[0]["candidate_source"] = "llm"
    candidate_evaluations = [
        {
            "candidate_index": candidate_index,
            "valid": True,
            "remaining_blocked_issues": 0,
            "validated_task": _candidate_event(event_name),
            "validation_findings": [],
        }
        for candidate_index, event_name in enumerate(
            ("low_score", "better_score", "third_choice")
        )
    ]
    llm_raw_response = {
        "thought": "select the second candidate",
        "selected_candidate_index": 1,
        "candidate_events": [
            _candidate_event("low_score"),
            _candidate_event("better_score"),
            _candidate_event("third_choice"),
        ],
    }
    enriched_response = {
        **deepcopy(llm_raw_response),
        "candidate_evaluations": deepcopy(candidate_evaluations),
        "selected_transition": deepcopy(transition_trace[0]),
        "selected_candidate_index": 1,
        "transition_trace": deepcopy(transition_trace),
    }
    turn = {
        "turn_index": 3,
        "phase": "outline",
        "decision": "need_next_task",
        "prompt_text": "candidate prompt",
        "llm_request": {
            "model": "test-model",
            "messages": [{"role": "user", "content": "candidate prompt"}],
            "response_format": {"type": "json_schema", "json_schema": {}},
            "response_source": "mocked_scripted_fixture",
            "request_sent": False,
        },
        "llm_raw_response": deepcopy(llm_raw_response),
        "raw_response": deepcopy(enriched_response),
        "candidate_evaluations": deepcopy(candidate_evaluations),
        "selected_candidate_index": 1,
        "selected_transition": deepcopy(transition_trace[0]),
        "transition_trace": deepcopy(transition_trace),
    }
    artifact_paths = write_recovery_artifacts(
        {
            "reasoning_mode": "multi_turn",
            "multi_turn_current_turn": deepcopy(turn),
            "multi_turn_llm_raw_response": deepcopy(llm_raw_response),
            "recovery_debug": {
                "multi_turn_session": {
                    "session_id": "artifact_test",
                    "turn_index": 3,
                    "turns": [deepcopy(turn)],
                }
            },
        },
        phase_label="multi_turn",
        debug_dir=tmp_path,
    )

    request_artifact = Path(artifact_paths["request_artifact_path"])
    result_artifact = Path(artifact_paths["outline_result_artifact_path"])
    audit_artifact = Path(artifact_paths["outline_audit_artifact_path"])
    stack_artifact = Path(artifact_paths["outline_stack_artifact_path"])
    assert request_artifact.name.startswith("multi_turn_turn03_outline_request_")
    assert result_artifact.name.startswith("multi_turn_turn03_outline_result_")
    assert audit_artifact.name.startswith("multi_turn_turn03_outline_audit_")
    assert stack_artifact.name.startswith("multi_turn_turn03_outline_stack_")
    assert artifact_paths["prompt_artifact_path"] == str(request_artifact)
    assert artifact_paths["response_artifact_path"] == str(result_artifact)
    assert "llm_response_artifact_path" not in artifact_paths
    assert "turn_index_artifact_path" not in artifact_paths
    assert {path.name for path in (tmp_path / "recovery_outline").iterdir()} == {
        request_artifact.name,
        result_artifact.name,
        audit_artifact.name,
        stack_artifact.name,
    }

    request_text = request_artifact.read_text(encoding="utf-8")
    assert "Model: test-model" in request_text
    assert "role=system" not in request_text
    assert "role=user" in request_text
    assert "candidate prompt" in request_text

    result_payload = json.loads(result_artifact.read_text(encoding="utf-8"))
    audit_payload = json.loads(audit_artifact.read_text(encoding="utf-8"))
    assert "llm_response" not in result_payload
    assert audit_payload["llm_response"] == llm_raw_response
    assert len(result_payload["candidate_evaluation_summary"]) == 3
    assert len(audit_payload["candidate_evaluation_summary"]) == 3
    assert result_payload["selected_transition"] == {
        "outline_id": "RECOVERY_SEQ1",
        "candidate_source": "llm",
        "event_name": "better_score",
        "resource_jid": "xarm6@localhost",
        "part_name": "LG",
        "expected_end_state": {
            "resource_state": "picked",
            "held_part": "LG",
        },
    }
    assert audit_payload["selected_transition"] == transition_trace[0]
    assert "transition_trace" not in result_payload
    assert result_payload["turn_index"] == 3
    assert result_payload["phase"] == "outline"
    assert result_payload["decision"] == "need_next_task"
    assert "selected_candidate_index" not in result_payload
    assert "selected_transition_outline_id" not in result_payload
    assert audit_payload["selected_candidate_index"] == 1
    assert audit_payload["selected_transition_outline_id"] == "RECOVERY_SEQ1"
    assert result_payload["accepted_trace_length"] == 1
    assert result_payload["remaining_blocked_issue_count"] == 0
    assert result_payload["artifact_paths"] == {
        "request_artifact_path": str(request_artifact),
        "outline_result_artifact_path": str(result_artifact),
        "outline_audit_artifact_path": str(audit_artifact),
        "outline_stack_artifact_path": str(stack_artifact),
        "prompt_artifact_path": str(request_artifact),
        "response_artifact_path": str(result_artifact),
    }
    assert audit_payload["artifact_paths"] == result_payload["artifact_paths"]
    stack_payload = json.loads(stack_artifact.read_text(encoding="utf-8"))
    assert stack_payload == transition_trace
    for serialized_payload in (
        json.dumps(result_payload, sort_keys=True),
        json.dumps(audit_payload, sort_keys=True),
        json.dumps(stack_payload, sort_keys=True),
    ):
        assert "llm_outline_id" not in serialized_payload
        assert '"outline_id": "enabledness_' not in serialized_payload


def test_outline_latest_paths_link_concise_result_and_detailed_audit(
    tmp_path: Path,
) -> None:
    transition = _candidate_event("selected", outline_id="RECOVERY_SEQ1")
    turn = {
        "turn_index": 3,
        "phase": "outline",
        "decision": "outline_ready",
        "prompt_text": "candidate prompt",
        "llm_raw_response": {
            "thought": "select candidate",
            "selected_candidate_index": 0,
            "candidate_events": [deepcopy(transition)],
        },
        "raw_response": {
            "selected_transition": deepcopy(transition),
            "selected_candidate_index": 0,
            "transition_trace": [deepcopy(transition)],
        },
        "selected_transition": deepcopy(transition),
        "transition_trace": [deepcopy(transition)],
    }

    artifact_paths = write_recovery_artifacts(
        {
            "reasoning_mode": "multi_turn",
            "multi_turn_current_turn": deepcopy(turn),
            "recovery_debug": {
                "multi_turn_session": {
                    "session_id": "latest_artifact_test",
                    "turn_index": 3,
                    "turns": [deepcopy(turn)],
                }
            },
        },
        phase_label="multi_turn",
        debug_dir=tmp_path,
        write_latest=True,
    )

    latest_result = Path(artifact_paths["latest_response_artifact_path"])
    latest_audit = Path(artifact_paths["latest_outline_audit_artifact_path"])
    latest_result_payload = json.loads(latest_result.read_text(encoding="utf-8"))
    latest_audit_payload = json.loads(latest_audit.read_text(encoding="utf-8"))

    assert latest_result.name == "multi_turn_turn03_outline_result_latest.json"
    assert latest_audit.name == "multi_turn_turn03_outline_audit_latest.json"
    assert "llm_response" not in latest_result_payload
    assert latest_audit_payload["llm_response"]["thought"] == "select candidate"
    assert latest_result_payload["artifact_paths"][
        "latest_outline_audit_artifact_path"
    ] == str(latest_audit)


def test_rejected_outline_stack_preserves_only_the_accepted_trace(
    tmp_path: Path,
) -> None:
    transition_trace = [_candidate_event("accepted_before", outline_id="RECOVERY_SEQ1")]
    turn = {
        "turn_index": 4,
        "phase": "outline",
        "decision": "need_revision",
        "prompt_text": "candidate prompt",
        "llm_raw_response": {
            "thought": "invalid selected candidate",
            "selected_candidate_index": 0,
            "candidate_events": [_candidate_event("rejected_candidate")],
        },
        "raw_response": {
            "decision": "need_revision",
            "selected_candidate_index": 0,
            "transition_trace": deepcopy(transition_trace),
            "transition_validation": {"status": "rejected"},
        },
    }
    artifact_paths = write_recovery_artifacts(
        {
            "reasoning_mode": "multi_turn",
            "multi_turn_current_turn": deepcopy(turn),
            "recovery_debug": {
                "multi_turn_session": {
                    "session_id": "rejected_stack_test",
                    "turn_index": 4,
                    "turns": [deepcopy(turn)],
                }
            },
        },
        phase_label="multi_turn",
        debug_dir=tmp_path,
    )

    stack_payload = json.loads(
        Path(artifact_paths["outline_stack_artifact_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert stack_payload == transition_trace


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

    assert settings["recovery_selection_mode"] == "neurosymbolic"
    assert settings["action_horizon"] == 1
    assert settings["candidate_count"] == "auto"
    assert settings["candidate_proposal_budget"] == 5


def test_case3_configured_neurosymbolic_uses_adaptive_symbolic_selection() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="neurosymbolic",
        action_horizon="1",
    )
    schema = multi_turn_mode._get_response_schema(
        "outline",
        {
            **session_state,
            "outline_mode": "incremental_candidates_validated",
            "candidate_bound": 5,
        },
    )

    assert session_state["recovery_selection_mode"] == "neurosymbolic"
    assert "selected_candidate_index" not in schema["schema"]["properties"]
    candidate_schema = schema["schema"]["properties"]["candidate_events"]
    assert candidate_schema["minItems"] == 1
    assert candidate_schema["maxItems"] == 5


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
                "selected_candidate_index": 1,
                "candidate_events": [
                    _candidate_event("low_score"),
                    _candidate_event("better_score"),
                    _candidate_event("third_choice"),
                ],
            },
        )
    )

    assert turn_entry["selected_by"] == "pure_llm"
    assert turn_entry["selected_candidate_index"] == 1
    assert [row["valid"] for row in turn_entry["candidate_evaluations"]] == [
        True,
        True,
        True,
    ]
    assert session_state["accepted_outline_prefix"][0]["event_name"] == "better_score"


def test_selected_verified_ra_vocabulary_is_used_by_the_next_prompt() -> None:
    resource_jid = "resource@localhost"
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="1",
    )
    session_state["outline_mode"] = "incremental_candidates_validated"
    session_state["recovery_des_models"] = {
        resource_jid: {
            "state_variables": {
                "resource_state": {
                    "scope": "resource",
                    "domain": ["previous_vocabulary_token"],
                }
            },
            "events": [
                {
                    "updates": {
                        "resource_state": {
                            "set": "previous_vocabulary_token"
                        }
                    }
                }
            ],
        }
    }
    refreshed_model = {
        "state_variables": {
            "resource_state": {
                "scope": "resource",
                "domain": ["next_request_vocabulary_token"],
            }
        },
        "events": [
            {
                "updates": {
                    "resource_state": {"set": "next_request_vocabulary_token"}
                }
            }
        ],
    }

    asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            recovery_des_models={resource_jid: refreshed_model},
            parsed_response={
                "thought": "select the refreshed candidate",
                "selected_candidate_index": 0,
                "candidate_events": [
                    _candidate_event("first_choice"),
                    _candidate_event("second_choice"),
                    _candidate_event("third_choice"),
                ],
            },
        )
    )
    assert session_state["recovery_des_models"][resource_jid] == refreshed_model

    prompt = multi_turn_prompts.render_multi_turn_phase_prompt(
        multi_turn_prompts.build_multi_turn_phase_prompt_input(
            phase="outline",
            llm_input={
                "observed_runtime_state": {"resources": []},
                "part_facts": [],
                "goal_conditions": [],
                "loaded_safety_rules": [],
            },
            session_state=session_state,
            recovery_resources={},
        )
    )
    assert "next_request_vocabulary_token" in prompt
    assert "previous_vocabulary_token" not in prompt


def test_case3_invalid_llm_selected_candidate_is_not_substituted() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="1",
    )
    decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            invalid_candidate_indexes={0},
            parsed_response={
                "thought": "select the first candidate",
                "selected_candidate_index": 0,
                "candidate_events": [
                    _candidate_event("invalid_selected"),
                    _candidate_event("valid_alternative"),
                    _candidate_event("valid_third"),
                ],
            },
        )
    )

    assert decision == "need_revision"
    assert session_state["accepted_outline_prefix"] == []
    assert turn_entry["transition_validation"]["selected_candidate_index"] == 0
    assert turn_entry["candidate_evaluations"][0]["valid"] is False
    assert turn_entry["candidate_evaluations"][1]["valid"] is True
    assert [
        row["candidate_index"]
        for row in turn_entry["candidate_rejection_feedback"]
    ] == [0]


def test_case3_missing_selected_candidate_index_is_rejected() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="1",
    )
    decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "forgot to select",
                "candidate_events": [
                    _candidate_event("low_score"),
                    _candidate_event("better_score"),
                    _candidate_event("third_choice"),
                ],
            },
        )
    )

    assert decision == "need_revision"
    assert turn_entry["transition_validation"]["status"] == "rejected"
    assert "selected_candidate_index" in turn_entry["validation_findings"][0]["reason"]


def test_case3_out_of_range_selected_candidate_index_is_rejected() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="1",
    )
    decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "bad selection",
                "selected_candidate_index": 3,
                "candidate_events": [
                    _candidate_event("low_score"),
                    _candidate_event("better_score"),
                    _candidate_event("third_choice"),
                ],
            },
        )
    )

    assert decision == "need_revision"
    assert turn_entry["transition_validation"]["status"] == "rejected"
    assert "selected_candidate_index" in turn_entry["validation_findings"][0]["reason"]


def test_case3_pure_llm_k_horizon_commits_selected_sequence() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="k",
    )
    _decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "propose short traces",
                "selected_candidate_index": 1,
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

    assert turn_entry["selected_by"] == "pure_llm"
    assert turn_entry["selected_candidate_index"] == 1
    assert len(turn_entry["selected_transition_sequence"]) == 2
    assert [row["event_name"] for row in session_state["accepted_outline_prefix"]] == [
        "step_a",
        "step_b",
    ]


def test_case3_pure_llm_full_horizon_requires_complete_trace() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="full",
    )
    decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "propose full traces",
                "selected_candidate_index": 1,
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


def test_case3_one_step_candidate_count_rejects_fewer_than_three() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="1",
        candidate_count="auto",
    )
    decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "too few alternatives",
                "candidate_events": [
                    _candidate_event("low_score"),
                    _candidate_event("better_score"),
                ],
            },
        )
    )

    assert decision == "need_revision"
    assert "exactly 3 candidates" in turn_entry["error"]


def test_case3_one_step_candidate_count_rejects_more_than_three() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="1",
        candidate_count="auto",
    )
    decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "too many alternatives",
                "selected_candidate_index": 0,
                "candidate_events": [
                    _candidate_event("first_choice"),
                    _candidate_event("second_choice"),
                    _candidate_event("third_choice"),
                    _candidate_event("fourth_choice"),
                ],
            },
        )
    )

    assert decision == "need_revision"
    assert "exactly 3 candidates" in turn_entry["error"]


if __name__ == "__main__":
    raise SystemExit(main())


def test_recovery_dfa_accepting_self_loop_and_multiple_accepting_states() -> None:
    from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor

    monitor = OnlineSafetyMonitor({"SAFE_1": """
        digraph DFA {
            node [shape = doublecircle]; 2; 3;
            node [shape = circle]; 1;
            init -> 1;
            1 -> 2 [label="ap1"];
            1 -> 3 [label="!ap1"];
            2 -> 2 [label="true"];
            3 -> 3 [label="true"];
        }
    """}, [])
    assert monitor.dfas["SAFE_1"]["accepting_states"] == ["2", "3"]
    assert monitor.dfas["SAFE_1"]["violation_state"] is None
    assert monitor.online_safety_validation(["ap1"])[0] is True
    assert monitor.online_safety_validation([])[0] is True
    assert monitor.current_states == {"SAFE_1": "1"}


def test_recovery_dfa_rejects_nonaccepting_cycle_and_unsatisfiable_escape() -> None:
    from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor

    monitor = OnlineSafetyMonitor({"SAFE_1": """
        digraph DFA {
            node [shape = doublecircle]; 1;
            node [shape = circle]; 2; 3;
            init -> 1;
            1 -> 2 [label="ap1"];
            1 -> 1 [label="!ap1"];
            2 -> 1 [label="ap1 & !ap1"];
            2 -> 3 [label="true"];
            3 -> 2 [label="true"];
        }
    """}, [])
    assert monitor.dfas["SAFE_1"]["accepting_reachable_states"] == ["1"]
    allowed, result = monitor.online_safety_validation(["ap1"])
    assert allowed is False
    assert result["violated_rule"] == "SAFE_1"
    assert result["violated_to"] == "2"
    assert monitor.current_states == {"SAFE_1": "1"}


def test_recovery_dfa_allows_nonaccepting_prefix_with_accepting_continuation() -> None:
    from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor

    monitor = OnlineSafetyMonitor({"SAFE_1": """
        digraph DFA {
            node [shape = doublecircle]; 3;
            node [shape = circle]; 1; 2;
            init -> 1;
            1 -> 2 [label="true"];
            2 -> 3 [label="ap1"];
            2 -> 2 [label="!ap1"];
            3 -> 3 [label="true"];
        }
    """}, [])
    allowed, result = monitor.online_safety_validation([])
    assert allowed is True
    assert result["next_states"] == {"SAFE_1": "2"}


def test_recovery_dfa_empty_label_is_a_trace_step() -> None:
    from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor

    monitor = OnlineSafetyMonitor({"SAFE_1": """
        digraph DFA {
            node [shape = doublecircle]; 1;
            node [shape = circle]; 2;
            init -> 1;
            1 -> 1 [label="ap1"];
            1 -> 2 [label="!ap1"];
            2 -> 2 [label="true"];
        }
    """}, [])
    assert monitor.online_safety_validation([], successor_state_aps=[])[0] is False


def test_recovery_successor_replaces_old_facts_and_keeps_running_tasks() -> None:
    from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor

    monitor = OnlineSafetyMonitor({"SAFE_2": """
        digraph DFA {
            node [shape = doublecircle]; 1;
            node [shape = circle]; 2;
            init -> 1;
            1 -> 2 [label="ap1 & ap2"];
            1 -> 1 [label="!ap1 | !ap2"];
            2 -> 2 [label="true"];
        }
    """}, [])
    monitor.resource_state_aps["recovery_scope"] = {"ap1"}
    monitor.running_aps = {"ap2"}
    allowed, result = monitor.online_safety_validation([], successor_state_aps=[])
    assert allowed is True
    assert result["successor_state_aps"] == []
    assert result["running_snapshot"] == ["ap2"]
    assert monitor.online_safety_validation([], successor_state_aps=["ap1"])[0] is False
    assert monitor.resource_state_aps == {"recovery_scope": {"ap1"}}
    assert monitor.current_states == {"SAFE_2": "1"}


def test_recovery_macro_uses_successor_history_and_other_tasks() -> None:
    rule = {
        "id": "SAFE_1", "ap_scope": "nominal",
        "text": "Require the projected idle condition while other work remains active.",
        "dfa_dot": """
            digraph DFA {
                node [shape = doublecircle]; 3;
                node [shape = circle]; 1; 2; 4;
                init -> 1;
                1 -> 4 [label="true"];
                2 -> 3 [label="!ap1 & ap2 & ap3 & ap4"];
                2 -> 4 [label="ap1 | !ap2 | !ap3 | !ap4"];
                3 -> 3 [label="!ap1 & ap2 & ap3 & ap4"];
                3 -> 4 [label="ap1 | !ap2 | !ap3 | !ap4"];
                4 -> 4 [label="true"];
            }
        """,
        "recovery_aps": [
            {"label": "ap1", "full": '{"kind":"ap_state","process":"p","product":"any","resource":"xarm6","state":{"arguments":{},"symbol":"failed"}}',
             "selector": {"mode": "resource_state", "resource": "xarm6", "state": "failed"}},
            {"label": "ap2", "full": '{"kind":"ap_state","process":"p","product":"any","resource":"xarm6","state":{"arguments":{},"symbol":"idle"}}',
             "selector": {"mode": "resource_state", "resource": "xarm6", "state": "idle"}},
            {"label": "ap3", "full": '{"kind":"ap_state","process":"p","product":"any","resource":"ur5e","state":{"arguments":{},"symbol":"printing"}}',
             "selector": {"mode": "resource_state", "resource": "ur5e", "state": "printing"}},
            {"label": "ap4", "full": '{"event":{"arguments":{},"symbol":"place_approach"},"kind":"ap_event","process":"p","product":"MCP","resource":"ur5e"}',
             "selector": {"mode": "move_part_to_destination", "part": "MCP",
                          "resource": "ur5e", "destination": "assembly_board-v1"}},
        ],
    }
    pre_resources = {
        "xarm6@localhost": {"current_state": "failed"},
        "ur5e@localhost": {"current_state": "printing"},
    }
    projected = deepcopy(pre_resources)
    projected["xarm6@localhost"]["current_state"] = "idle"
    kwargs = {
        "task": {"outline_id": "RECOVERY_SEQ1", "event_name": "evt_q7",
                 "resource_jid": "xarm6@localhost"},
        "signature": {}, "pre_resources": pre_resources, "pre_parts": {},
        "projected_resources": projected, "projected_parts": {},
        "llm_input": {"loaded_safety_rules": [rule],
                      "recovery_safety_context": {"running_aps": ["ap4"]}},
        "safety_dfa_states_before": {"SAFE_1": "2"},
    }
    before = deepcopy(kwargs)
    accepted = validate_outline_macro_recovery_safety(**kwargs)
    assert accepted["is_safe"] is True
    assert accepted["safety_dfa_states_before"] == {"SAFE_1": "2"}
    assert accepted["safety_dfa_states_after"] == {"SAFE_1": "3"}
    assert set(accepted["safety_ctx"]["predicted_state_aps"]) == {"ap2", "ap3"}
    assert kwargs == before

    kwargs["safety_dfa_states_before"] = accepted["safety_dfa_states_after"]
    kwargs["projected_resources"] = deepcopy(pre_resources)
    rejected = validate_outline_macro_recovery_safety(**kwargs)
    assert rejected["is_safe"] is False
    assert rejected["safety_dfa_states_after"] == {"SAFE_1": "3"}
    finding = rejected["findings"][0]
    assert finding["rule_id"] == "SAFE_1"
    assert finding["evidence"]["safety_rule"] == rule | {
        "rule_id": "SAFE_1", "dfa_dot": rule["dfa_dot"].strip(),
    }
    assert finding["evidence"]["projected_resources"] == pre_resources
    assert finding["evidence"]["projected_parts"] == {}


def test_recovery_dfa_missing_or_ambiguous_step_cannot_stutter() -> None:
    from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor

    for edges in (
        '1 -> 1 [label="ap1"];',
        '1 -> 1 [label="true"]; 1 -> 2 [label="true"]; 2 -> 2 [label="true"];',
    ):
        monitor = OnlineSafetyMonitor({"SAFE_1": (
            "digraph DFA { node [shape = doublecircle]; 1; 2; "
            "node [shape = circle]; init -> 1; " + edges + " }"
        )}, [])
        allowed, evidence = monitor.online_safety_validation([], successor_state_aps=[])
        assert allowed is False
        assert evidence["violated_rule"] == "SAFE_1"
        assert monitor.current_states == {"SAFE_1": "1"}


def test_recovery_applicable_constant_rule_and_missing_projection() -> None:
    import pytest

    rule = {
        "id": "SAFE_1", "ap_scope": "recovery",
        "dfa_dot": 'digraph DFA { node [shape = doublecircle]; 1; node [shape = circle]; '
                   'init -> 1; 1 -> 1 [label="true"]; }',
        "recovery_aps": [],
    }
    kwargs = {
        "task": {}, "signature": {}, "pre_resources": {}, "pre_parts": {},
        "projected_resources": {}, "projected_parts": {},
        "llm_input": {"loaded_safety_rules": [rule]},
        "safety_dfa_states_before": {"SAFE_1": "1"},
    }
    assert validate_outline_macro_recovery_safety(**kwargs)["is_safe"] is True
    rule["dfa_dot"] = rule["dfa_dot"].replace('"true"', '"ap1"')
    with pytest.raises(ValueError, match="unprojected propositions"):
        validate_outline_macro_recovery_safety(**kwargs)
    rule["dfa_dot"] = ""
    with pytest.raises(ValueError, match="no DFA"):
        validate_outline_macro_recovery_safety(**kwargs)


def test_recovery_completion_requires_admissible_nominal_reentry() -> None:
    for reentry, expected in (([], "need_next_task"), (["REQ_1_T1"], "outline_ready")):
        session = _candidate_session(recovery_selection_mode="pure_llm", action_horizon="k")
        with patch.object(
            multi_turn_outline_generation, "_nominal_reentry_event_rows",
            return_value=[{"event_id": "REQ_1_T1"}],
        ):
            decision, _ = asyncio.run(_run_mocked_candidate_handler(
                session_state=session,
                parsed_response={
                    "thought": "resolve conditions",
                    "selected_candidate_index": 0,
                    "candidate_traces": [{"events": [
                        _candidate_event("step_a"), _candidate_event("step_b"),
                    ]}],
                },
                admissible_nominal_reentry_event_ids_after=reentry,
            ))
        assert decision == expected
        assert len(session["accepted_outline_prefix"]) == 2
        assert not session.get("accepted_primitive_program")


def test_shared_dfa_step_rejects_missing_and_ambiguous_guards() -> None:
    import pytest
    from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor

    for edges, reason in (
        ('1 -> 1 [label="ap1"];', "dfa_missing_transition"),
        ('1 -> 1 [label="true"]; 1 -> 2 [label="true"];', "dfa_ambiguous_transition"),
    ):
        monitor = OnlineSafetyMonitor({"SAFE_1": 'digraph DFA { node [shape = doublecircle]; 1; 2; node [shape = circle]; init -> 1; ' + edges + ' }'}, [])
        allowed, evidence = monitor.online_safety_validation([])
        assert not allowed
        assert evidence["rule_checks"][0]["reason"] == reason
        with pytest.raises(ValueError, match=reason):
            monitor._delta("SAFE_1", "1", frozenset())
        assert monitor.current_states == {"SAFE_1": "1"}


def test_cca_records_successful_and_failed_rules_without_advancing_history() -> None:
    from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor

    good = 'digraph DFA { node [shape = doublecircle]; 1; node [shape = circle]; init -> 1; 1 -> 1 [label="true"]; }'
    bad = 'digraph DFA { node [shape = doublecircle]; 1; node [shape = circle]; 2; init -> 1; 1 -> 2 [label="true"]; 2 -> 2 [label="true"]; }'
    monitor = OnlineSafetyMonitor({"SAFE_1": bad, "SAFE_2": good}, [])
    before = deepcopy(monitor.current_states)
    allowed, evidence = monitor.online_safety_validation([])
    assert not allowed
    assert [row["status"] for row in evidence["rule_checks"]] == ["rejected", "passed"]
    assert evidence["rule_checks"][1]["label"] == []
    assert monitor.current_states == before


def test_missing_state_evidence_is_not_a_false_safety_ap() -> None:
    import pytest
    from cais_spade_llm.agents.central_controller.outline_macro_safety import _require_state_evidence

    selector = {"mode": "resource_in_destination", "resource": "ur5e-3", "destination": "Assembly Station"}
    with pytest.raises(ValueError, match="missing"):
        _require_state_evidence(selector, {}, {})
    with pytest.raises(ValueError, match="missing"):
        _require_state_evidence(selector, {"ur5e-3@localhost": {"current_state": "failed"}}, {})
    _require_state_evidence(selector, {"ur5e-3@localhost": {"current_location": "Assembly Station"}}, {})


def test_malformed_committed_dfa_history_holds_later_admission() -> None:
    import pytest
    from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor

    monitor = OnlineSafetyMonitor({"SAFE_1": 'digraph DFA { node [shape = doublecircle]; 1; node [shape = circle]; init -> 1; 1 -> 1 [label="ap1"]; }'}, [])
    event = {"resource_jid": "ur5e-4@localhost", "function_name": "release_part", "current_state": "idle"}
    with pytest.raises(ValueError, match="dfa_missing_transition"):
        monitor.process_finish_event(event)
    assert monitor.current_states == {"SAFE_1": "1"}
    allowed, evidence = monitor.online_safety_validation(["ap1"])
    assert not allowed
    assert evidence["reason"] == "monitor_history_unavailable"
    assert monitor.resource_states["ur5e-4@localhost"]["current_state"] == "idle"


def test_live_cca_refuses_recovery_after_a_monitor_history_error(monkeypatch) -> None:
    from unittest.mock import AsyncMock
    from cais_spade_llm.agents.central_controller import central_controller_agent as cca_module

    agent = SimpleNamespace(
        jid="central_controller@localhost", safety_file=None, safety_rules=[],
        safety_monitor=SimpleNamespace(current_states={}, running_aps=set(), history_error={"reason": "missing step"}),
        _wait_for_safety_monitor_ready=AsyncMock(return_value=True), logger=logging.getLogger(__name__),
    )
    payload = {"product_jid": "product@localhost", "request_id": "history-test", "candidates": [{"candidate_index": 0, "event_id": "proposed_recovery", "safety_input": {}}]}
    behaviour = SimpleNamespace(agent=agent, receive=AsyncMock(return_value=SimpleNamespace(body=json.dumps(payload), sender="product@localhost")))
    send = AsyncMock()
    monkeypatch.setattr(cca_module, "send_agent_message", send)
    asyncio.run(cca_module.CentralControllerAgent._RecoveryOutlineSafetyValidation.run(behaviour))
    response = json.loads(send.call_args.args[1].body)
    assert response["admissible_recovery_event_ids"] == []
    assert response["results"][0]["is_safe"] is False
    assert response["results"][0]["findings"][0]["constraint_code"] == "safety_validation_unavailable"
    assert "history" in response["results"][0]["findings"][0]["reason"]


def test_local_analysis_copies_history_and_reuses_compiled_DFAs() -> None:
    from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor
    from cais_spade_llm.recovery_framework.environment_composition import detached_checker

    monitor = OnlineSafetyMonitor({"SPEC": """
        digraph DFA { node [shape = doublecircle]; 0; init -> 0;
          0 -> 0 [label="true"]; }
    """}, [])
    monitor.running_aps = {"ap1"}
    monitor.resource_state_aps = {"M1": {"ap2"}}
    copied = detached_checker(monitor)
    assert copied.dfas is monitor.dfas
    copied.current_states["SPEC"] = "analysis_only"
    copied.running_aps.clear()
    copied.resource_state_aps["M1"].clear()
    assert monitor.current_states == {"SPEC": "0"}
    assert monitor.running_aps == {"ap1"}
    assert monitor.resource_state_aps == {"M1": {"ap2"}}


def test_local_independence_checks_persistent_AP_valuations() -> None:
    from cais_spade_llm.agents.central_controller.local_composition import Budget
    from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor
    from cais_spade_llm.recovery_framework.environment_composition import EnvironmentPlant

    rule = {"id": "SPEC", "aps": [
        {"label": "ap1", "full": '{"kind":"ap_state","process":"operation","product":"any","resource":"m1","state":{"arguments":{},"symbol":"loaded"}}'}]}
    monitor = OnlineSafetyMonitor({"SPEC": """
        digraph DFA { node [shape = doublecircle]; 0; 1; init -> 0;
          0 -> 1 [label="ap1"]; 0 -> 0 [label="!ap1"]; 1 -> 1 [label="true"]; }
    """}, [rule])
    plant = object.__new__(EnvironmentPlant)
    plant.checker = monitor
    assert monitor.transition_evidence("SPEC", "0", frozenset())["to"] == "0"
    assert not plant._unmatched_stutters("SPEC", rule, Budget())

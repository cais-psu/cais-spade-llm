from __future__ import annotations

from copy import deepcopy
from typing import Any

from test_case3_recovery_main import (
    LG_DROP_POSE,
    MAIN_V1_VARIANT,
    MAIN_V2_VARIANT,
    _bridge_clear_event,
    _bridge_pick_place_event,
    _prepare_case3_harness_state,
    _record_bridge_observation,
    run_case3_recovery_dry_run,
)


def _bridge_replaced_part_names(prepared_bridge_request: dict[str, Any]) -> set[str]:
    marked_reentry = prepared_bridge_request.get("marked_reentry_context") or {}
    return {
        str(condition.get("entity") or "").strip()
        for condition in (marked_reentry.get("marked_reentry_conditions") or [])
        if isinstance(condition, dict)
        and str(condition.get("entity_kind") or "").strip() == "part"
        and str(condition.get("role") or "").strip() == "bridge_replaced"
        and str(condition.get("entity") or "").strip()
    }


def _misplaced_mcp_overrides(mcp_pose: dict[str, float]) -> dict[str, Any]:
    return {
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


def test_bias_regression_main_variants_choose_distinct_recovery_topologies() -> None:
    main_v1 = run_case3_recovery_dry_run(write_debug=False, variant=MAIN_V1_VARIANT)
    main_v2 = run_case3_recovery_dry_run(write_debug=False, variant=MAIN_V2_VARIANT)

    v1_events = main_v1["proposal"].get("bridge_event_summary") or []
    v2_events = main_v2["proposal"].get("bridge_event_summary") or []

    v1_lg_resources = {
        str(event.get("resource_jid") or "").strip()
        for event in v1_events
        if isinstance(event, dict) and str(event.get("part_name") or "").strip() == "LG"
    }
    v2_lg_resources = {
        str(event.get("resource_jid") or "").strip()
        for event in v2_events
        if isinstance(event, dict) and str(event.get("part_name") or "").strip() == "LG"
    }

    assert len(v1_events) == 5
    assert len(v2_events) == 2
    assert v1_lg_resources == {"ur5e@localhost"}
    assert v2_lg_resources == {"xarm6@localhost"}
    assert any(
        str(event.get("part_name") or "").strip() == "MCP"
        for event in v1_events
        if isinstance(event, dict)
    )
    assert not any(
        str(event.get("part_name") or "").strip() == "MCP"
        for event in v2_events
        if isinstance(event, dict)
    )


def test_bias_regression_branch_swap_inverts_bridge_replaced_and_resume_suffix_roles() -> None:
    _, _, _, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
        scenario_overrides={"ra_jid": "ur5e@localhost"},
    )

    marked_reentry = prepared_bridge_request.get("marked_reentry_context") or {}
    resume_suffix_resources = {
        str(item.get("resource_jid") or "").strip()
        for item in (marked_reentry.get("pending_suffix_summary") or [])
        if isinstance(item, dict) and str(item.get("role") or "").strip() == "resume_suffix"
    }

    assert marked_reentry.get("focused_resource_jid") == "ur5e@localhost"
    assert _bridge_replaced_part_names(prepared_bridge_request) == {"MCP"}
    assert resume_suffix_resources == {"xarm6@localhost"}


def test_bias_regression_displaced_part_swap_compiles_without_lg_leakage() -> None:
    mcp_pose = {"x": 0.0, "y": -0.50, "z": 1.034}
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
        scenario_overrides=_misplaced_mcp_overrides(mcp_pose),
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
    assert {
        str(task.get("part_name") or "").strip()
        for task in (compiled_plan.get("macro_tasks") or [])
        if isinstance(task, dict) and str(task.get("part_name") or "").strip()
    } == {"MCP"}


def test_bias_regression_prompt_handles_non_recovery_required_blocking_state() -> None:
    _, _, planner, prepared_bridge_request, _ = _prepare_case3_harness_state(
        llm_mode="scripted",
        llm_model=None,
        scenario_overrides={
            "fixture": {
                "resource_states": {
                    "xarm6@localhost": {
                        "current_state": "faulted",
                    }
                },
                "stuck_state": {
                    "resource_state": "faulted",
                },
            },
            "robots": {
                "xarm6@localhost": {
                    "current_state": "faulted",
                }
            },
        },
    )
    _record_bridge_observation(
        planner,
        prepared_bridge_request,
        resource_jid="ur5e@localhost",
        part_name="LG",
        pose=LG_DROP_POSE,
    )

    prompt = planner._build_bridge_turn_prompt_preview(prepared_bridge_request)

    assert planner._bridge_current_phase(prepared_bridge_request) == "bridge_events"
    assert "Resource xarm6@localhost is in state 'faulted'." in prompt
    assert "Bring it toward 'idle' before proposing pick/place actions" in prompt

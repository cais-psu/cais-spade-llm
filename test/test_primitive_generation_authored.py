"""Unit tests for the LLM-authored primitive-generation phase (multi-turn v2).

These tests exercise ``_handle_primitive_generation_phase`` directly, seeding the
session state as if the bridge had just entered turn 08 (primitive phase) after
an accepted outline prefix. No XMPP, no LLM, no Gazebo — the LLM is replaced by
a pre-canned ``parsed_response`` per test.

Plan cases covered:
  1. Acquisition authored end-to-end
  2. Placement authored end-to-end
  3. Agentic retrieval loop (need_context)
  4. Derived-offset path (no hardcoded dz constants)
  5. Witness violation (motion not bound to target_pose)
  6. Need-context refusal
  7. Need-primitive-revision refusal
  8. Hidden-primitive rejection (move_pose / get_current_pose)
  9. Unmodeled recovery coverage
 10. Load-bearing-LLM ablation (empty steps halts the phase)
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from test.test_case3_bridge_dryrun import (
    _prepare_bridge_dryrun_harness,
    _seed_case3_primitive_generation_focus,
)

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
    multi_turn_v2 as multi_turn_v2_mode,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
    multi_turn_primitive_generation as primitive_mode,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.prompts import (
    multi_turn_v2 as prompt_module,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


async def _harness() -> tuple[Any, dict[str, Any]]:
    _, _, planner, prepared_bridge_request = await _prepare_bridge_dryrun_harness()
    seed = multi_turn_v2_mode.build_multi_turn_session_seed(prepared_bridge_request)
    session_state = _seed_case3_primitive_generation_focus(seed)
    return planner, prepared_bridge_request, session_state


def _ctx(alias: str, field: str) -> dict[str, Any]:
    return {"context_ref": f"step_outputs.{alias}.{field}"}


def _acquire_seq3_steps() -> list[dict[str, Any]]:
    """Canonical acquisition steps for RECOVERY_SEQ3 (LG pick)."""
    return [
        {
            "primitive": "detect_parts",
            "params": {"part_name": "LG"},
            "store_as": "obs_lg",
        },
        {
            "primitive": "compute_pick_targets",
            "params": {"part_name": "LG"},
            "store_as": "pick",
        },
        {
            "primitive": "move_cartesian",
            "params": {
                "x": _ctx("pick", "approach_pose.x"),
                "y": _ctx("pick", "approach_pose.y"),
                "z": _ctx("pick", "approach_pose.z"),
            },
        },
        {
            "primitive": "move_cartesian",
            "params": {
                "x": _ctx("pick", "target_pose.x"),
                "y": _ctx("pick", "target_pose.y"),
                "z": _ctx("pick", "target_pose.z"),
            },
        },
        {
            "primitive": "grasp_part",
            "params": {"model_name": "LG", "part_name": "LG"},
        },
    ]


def _release_steps(
    part_name: str,
    *,
    destination_location: str | None = None,
) -> list[dict[str, Any]]:
    compute_params: dict[str, Any] = {"part_name": part_name}
    if destination_location is not None:
        compute_params["destination_location"] = destination_location
    return [
        {
            "primitive": "compute_place_targets",
            "params": compute_params,
            "store_as": "place",
        },
        {
            "primitive": "move_cartesian",
            "params": {
                "x": _ctx("place", "approach_pose.x"),
                "y": _ctx("place", "approach_pose.y"),
                "z": _ctx("place", "approach_pose.z"),
            },
        },
        {
            "primitive": "move_cartesian",
            "params": {
                "x": _ctx("place", "target_pose.x"),
                "y": _ctx("place", "target_pose.y"),
                "z": _ctx("place", "target_pose.z"),
            },
        },
        {
            "primitive": "release_part",
            "params": {"model_name": part_name, "part_name": part_name},
        },
    ]


def _release_seq1_steps() -> list[dict[str, Any]]:
    """Canonical placement steps for RECOVERY_SEQ1 (MCP release)."""
    return _release_steps("MCP")


def _release_seq4_steps(
    *,
    destination_location: str | None = None,
) -> list[dict[str, Any]]:
    """Canonical placement steps for RECOVERY_SEQ4 (LG release)."""
    return _release_steps("LG", destination_location=destination_location)


def _ungrounded_release_steps() -> list[dict[str, Any]]:
    return [
        {
            "primitive": "move_relative",
            "params": {"dx": 0.0, "dy": 0.0, "dz": 0.02, "speed": 0.5},
        },
        {
            "primitive": "release_part",
            "params": {"model_name": "MCP", "part_name": "MCP"},
        },
    ]


def _non_target_grasp_steps() -> list[dict[str, Any]]:
    return [
        {"primitive": "detect_parts", "params": {"part_name": "LG"}, "store_as": "obs"},
        {
            "primitive": "compute_pick_targets",
            "params": {"part_name": "LG"},
            "store_as": "pick",
        },
        {
            "primitive": "move_cartesian",
            "params": {
                "x": _ctx("pick", "approach_pose.x"),
                "y": _ctx("pick", "approach_pose.y"),
                "z": _ctx("pick", "approach_pose.z"),
            },
        },
        {
            "primitive": "grasp_part",
            "params": {"model_name": "LG", "part_name": "LG"},
        },
    ]


def _authored_response(
    *,
    outline_id: str,
    resource_jid: str,
    primitive_steps: list[dict[str, Any]],
    decision: str = "primitive_steps_ready",
) -> dict[str, Any]:
    return {
        "thought": "Authoring the primitive plan directly.",
        "decision": decision,
        "outline_id": outline_id,
        "resource_jid": resource_jid,
        "primitive_steps": primitive_steps,
        "rationale": "",
        "notes": [],
    }


def _set_cursor_to(session_state: dict[str, Any], outline_id: str) -> None:
    prefix = list(session_state.get("accepted_outline_prefix") or [])
    for idx, row in enumerate(prefix):
        if row.get("outline_id") == outline_id:
            session_state["primitive_generation_cursor"] = idx
            return
    raise AssertionError(f"outline_id {outline_id!r} not in accepted prefix")


def _active_event(session_state: dict[str, Any]) -> dict[str, Any]:
    _, active_event, _ = primitive_mode._active_primitive_outline_event(session_state)
    assert active_event is not None
    return active_event


def _primitive_prompt_text(
    *,
    prepared: dict[str, Any],
    session_state: dict[str, Any],
) -> str:
    prompt_input = prompt_module.build_multi_turn_v2_phase_prompt_input(
        phase="primitive_generation",
        llm_input=dict(prepared.get("llm_input") or {}),
        session_state=session_state,
        bridge_resources=dict(prepared.get("bridge_resources") or {}),
    )
    prompt_input.update(
        primitive_mode.build_primitive_generation_prompt_context(
            session_state=session_state,
            prepared_bridge_request=prepared,
        )
    )
    return prompt_module.render_multi_turn_v2_phase_prompt(prompt_input)


def _outline_prompt_text(
    *,
    prepared: dict[str, Any],
    session_state: dict[str, Any],
) -> str:
    prompt_input = prompt_module.build_multi_turn_v2_phase_prompt_input(
        phase="outline",
        llm_input=dict(prepared.get("llm_input") or {}),
        session_state=session_state,
        bridge_resources=dict(prepared.get("bridge_resources") or {}),
    )
    return prompt_module.render_multi_turn_v2_phase_prompt(prompt_input)


def _section_body(text: str, title: str) -> str:
    marker = f"\n{title}\n"
    start = text.find(marker)
    if start == -1:
        raise AssertionError(f"section {title!r} not found")
    start += len(marker)
    end = text.find("\n\n", start)
    if end == -1:
        end = len(text)
    return text[start:end]


# ---------------------------------------------------------------------------
# 0. Slim prompt + retrievable context surface
# ---------------------------------------------------------------------------


def test_initial_prompt_is_minimal() -> None:
    async def _run() -> None:
        _, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")

        text = _primitive_prompt_text(prepared=prepared, session_state=session_state)

        assert "Active DES Transition (token)" in text
        assert '"outline_id": "RECOVERY_SEQ3"' in text
        assert '"name": "grasp_part"' in text
        assert '"primitive_kind"' in text
        assert "/primitive_contracts/<name>" in text
        assert "/capability_decompositions/<function_name>" in text
        assert "Available Capability Decompositions" in text
        assert "decomposition_examples" not in text
        assert "primitive_event_ready" not in text
        assert "need_outline_revision" not in text
        assert "primitive_steps_ready" in text
        assert "primitive_blocked" in text
        assert '"place_approach"' in text
        assert '"place_insert"' in text
        assert "/outline_event" in text
        assert "closest applicable /capability_decompositions/<function_name>" in text
        assert "Primitive names like grasp_part and release_part must be retrieved via /primitive_contracts/<name> instead." in text
        assert "known pick/place/release/home behavior" not in text

        # Full cards and full event bodies should be retrievable, not injected.
        for banned in (
            '"preconditions"',
            '"effects"',
            '"output_schema"',
            '"expected_start_state"',
        ):
            assert banned not in text

    asyncio.run(_run())


def test_primitive_prompt_has_named_pose_rules_section_with_names_only() -> None:
    async def _run() -> None:
        _, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")

        text = _primitive_prompt_text(prepared=prepared, session_state=session_state)
        named_pose_section = _section_body(text, "Named Pose Rules")
        output_contract_section = _section_body(text, "Output Contract")

        assert "Named Pose Rules" in text
        assert text.index("Named Pose Rules") < text.index("Output Contract")
        assert '"home"' in named_pose_section
        assert "1.637161" not in named_pose_section
        assert (
            "move_to_named_pose may only use named poses advertised for the active resource."
            in named_pose_section
        )
        assert "/resources/ur5e@localhost/static_capabilities" in named_pose_section
        assert "move_to_named_pose('home')" not in output_contract_section
        assert "home admissibility" not in named_pose_section
        assert "requires home cleanup" not in named_pose_section
        assert "home admissibility" not in output_contract_section
        assert "requires home cleanup" not in output_contract_section

    asyncio.run(_run())


def test_named_pose_rules_section_is_present_for_release_event_without_home_note() -> None:
    async def _run() -> None:
        _, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ1")

        text = _primitive_prompt_text(prepared=prepared, session_state=session_state)
        named_pose_section = _section_body(text, "Named Pose Rules")

        assert '"home"' in named_pose_section
        assert "Current event home admissibility" not in named_pose_section
        assert "requires home cleanup" not in named_pose_section

    asyncio.run(_run())


def test_primitive_prompt_renders_input_diagnostics_for_target_mismatch() -> None:
    async def _run() -> None:
        _, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ1")

        text = _primitive_prompt_text(prepared=prepared, session_state=session_state)

        assert "Input Diagnostics" in text
        assert "target_location_mismatch" in text
        assert '"outline_target_ref": "prusa-mk3"' in text
        assert '"grounded_goal_location": "assembly_board-v1"' in text
        assert "Suggested Capability Decompositions For This Event" in text
        assert '"place_approach"' in text
        assert '"place_insert"' in text

    asyncio.run(_run())


def test_full_catalog_card_retrievable_via_ref() -> None:
    async def _run() -> None:
        _, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        served, errors = primitive_mode._serve_context_requests(
            session_state=session_state,
            prepared_bridge_request=prepared,
            outline_event=_active_event(session_state),
            context_requests=["/primitive_contracts/grasp_part"],
        )

        assert errors == []
        card = dict(served["/primitive_contracts/grasp_part"])
        assert card["name"] == "grasp_part"
        assert "preconditions" in card
        assert "effects" in card
        assert "output_schema" in card

    asyncio.run(_run())


def test_full_event_retrievable_via_ref() -> None:
    async def _run() -> None:
        _, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        served, errors = primitive_mode._serve_context_requests(
            session_state=session_state,
            prepared_bridge_request=prepared,
            outline_event=_active_event(session_state),
            context_requests=["/outline_event"],
        )

        assert errors == []
        event = dict(served["/outline_event"])
        assert event["outline_id"] == "RECOVERY_SEQ3"
        assert "expected_start_state" in event
        assert "expected_end_state" in event

    asyncio.run(_run())


def test_capability_decompositions_retrievable_for_place_approach_and_insert() -> None:
    async def _run() -> None:
        _, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        served, errors = primitive_mode._serve_context_requests(
            session_state=session_state,
            prepared_bridge_request=prepared,
            outline_event=_active_event(session_state),
            context_requests=[
                "/capability_decompositions/place_approach",
                "/capability_decompositions/place_insert",
            ],
        )

        assert errors == []
        place_approach = dict(served["/capability_decompositions/place_approach"])
        place_insert = dict(served["/capability_decompositions/place_insert"])
        assert place_approach["source"] == "robot_agent.py"
        assert place_approach["modeled_transition"] == "picked -> positioned"
        approach_names = [
            step["primitive"] for step in place_approach["bridge_visible_steps"]
        ]
        assert approach_names == [
            "compute_place_targets",
            "move_cartesian",
            "move_cartesian",
        ]
        serialized_approach = str(place_approach)
        assert "step_outputs.place_targets.approach_pose" in serialized_approach
        assert "step_outputs.place_targets.target_pose" in serialized_approach

        insert_names = [
            step["primitive"] for step in place_insert["bridge_visible_steps"]
        ]
        assert insert_names == ["release_part", "move_relative"]
        assert "Positive dz retreat" in str(place_insert)

    asyncio.run(_run())


def test_all_robot_capability_decompositions_resolve_and_hide_execution_primitives() -> None:
    async def _run() -> None:
        _, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        function_names = [
            "pick_approach",
            "pick_grasp",
            "place_approach",
            "place_insert",
            "move_home",
        ]
        refs = [f"/capability_decompositions/{name}" for name in function_names]
        served, errors = primitive_mode._serve_context_requests(
            session_state=session_state,
            prepared_bridge_request=prepared,
            outline_event=_active_event(session_state),
            context_requests=refs,
        )

        assert errors == []
        visible_names = {
            row["name"]
            for row in primitive_mode._visible_primitive_catalog_names(
                primitive_mode._primitive_catalog_for_resource(
                    prepared,
                    "ur5e@localhost",
                )
            )
        }
        hidden_names = {
            "open_gripper",
            "close_gripper",
            "attach_part",
            "detach_part",
            "move_pose",
            "get_current_pose",
        }
        for function_name in function_names:
            payload = dict(served[f"/capability_decompositions/{function_name}"])
            assert payload["function_name"] == function_name
            assert payload["source"] == "robot_agent.py"
            step_names = [
                str(step.get("primitive") or "").strip()
                for step in payload["bridge_visible_steps"]
            ]
            assert step_names
            assert set(step_names) <= visible_names
            assert not (set(step_names) & hidden_names)
            serialized_steps = str(payload["bridge_visible_steps"])
            for hidden_name in hidden_names:
                assert hidden_name not in serialized_steps

        assert "close_gripper" in str(
            served["/capability_decompositions/pick_grasp"]["execution_notes"]
        )
        assert "detach_part" in str(
            served["/capability_decompositions/place_insert"]["execution_notes"]
        )

    asyncio.run(_run())


def test_new_refs_resolve() -> None:
    async def _run() -> None:
        _, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        session_state["primitive_authoring_memo"] = [
            {
                "action": "acquire",
                "part": "SG",
                "part_name": "SG",
                "resource_jid": "ur5e@localhost",
                "outline_id": "PRIOR",
                "steps_summary": ["detect_parts", "grasp_part"],
                "accepted_at_turn": 1,
            }
        ]
        refs = [
            "/accepted_outline_prefix",
            "/remaining_outline_events",
            "/safety_rules",
            "/capability_decompositions/place_approach",
            "/memo/primitive_authoring",
        ]
        served, errors = primitive_mode._serve_context_requests(
            session_state=session_state,
            prepared_bridge_request=prepared,
            outline_event=_active_event(session_state),
            context_requests=refs,
        )

        assert errors == []
        assert set(served) == set(refs)
        assert isinstance(served["/accepted_outline_prefix"], list)
        assert isinstance(served["/remaining_outline_events"], list)
        assert isinstance(served["/safety_rules"], list)
        assert served["/capability_decompositions/place_approach"]["function_name"] == (
            "place_approach"
        )
        assert served["/memo/primitive_authoring"][0]["outline_id"] == "PRIOR"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 1. Acquisition authored end-to-end
# ---------------------------------------------------------------------------


def test_acquisition_authored_end_to_end() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        # LG must be seen as in workspace by start snapshot.
        decision, turn_entry = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response=_authored_response(
                outline_id="RECOVERY_SEQ3",
                resource_jid="ur5e@localhost",
                primitive_steps=_acquire_seq3_steps(),
            ),
            prepared_bridge_request=prepared,
            planner=planner,
        )
        assert decision in {"primitive_steps_ready", "draft_ready"}, (
            f"expected acceptance, got {decision!r}: "
            f"{session_state.get('primitive_rejection_feedback')}"
        )
        accepted = list(session_state.get("accepted_primitive_program") or [])
        assert accepted and accepted[-1]["outline_id"] == "RECOVERY_SEQ3"
        primitive_names = [s["primitive"] for s in accepted[-1]["primitive_steps"]]
        assert primitive_names[-1] == "grasp_part"
        assert "move_cartesian" in primitive_names

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2. Placement authored end-to-end
# ---------------------------------------------------------------------------


def test_placement_authored_end_to_end() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ1")
        decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response=_authored_response(
                outline_id="RECOVERY_SEQ1",
                resource_jid="ur5e@localhost",
                primitive_steps=_release_seq1_steps(),
            ),
            prepared_bridge_request=prepared,
            planner=planner,
        )
        assert decision in {"primitive_steps_ready", "draft_ready"}, (
            f"expected acceptance, got {decision!r}: "
            f"{session_state.get('primitive_rejection_feedback')}"
        )
        accepted = list(session_state.get("accepted_primitive_program") or [])
        assert accepted[-1]["outline_id"] == "RECOVERY_SEQ1"
        assert accepted[-1]["primitive_steps"][-1]["primitive"] == "release_part"

    asyncio.run(_run())


def test_explicit_release_destination_mismatch_is_rejected() -> None:
    async def _run() -> None:
        _, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ1")
        active_event = _active_event(session_state)

        _result, feedback = primitive_mode._validate_single_event_primitive_steps(
            session_state=session_state,
            prepared_bridge_request=prepared,
            outline_event=active_event,
            primitive_steps=_release_steps(
                "MCP",
                destination_location="assembly_board-v1",
            ),
        )

        codes = {str(row.get("constraint_code") or "") for row in feedback}
        reasons = " ".join(str(row.get("reason") or "") for row in feedback)
        assert "release_destination_mismatch" in codes
        assert "assembly_board-v1" in reasons
        assert "prusa-mk3" in reasons

    asyncio.run(_run())


def test_release_seq4_accepts_matching_explicit_destination() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ4")

        decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response=_authored_response(
                outline_id="RECOVERY_SEQ4",
                resource_jid="ur5e@localhost",
                primitive_steps=_release_seq4_steps(
                    destination_location="assembly_board-v1",
                ),
            ),
            prepared_bridge_request=prepared,
            planner=planner,
        )

        assert decision in {"primitive_steps_ready", "draft_ready"}, (
            f"expected acceptance, got {decision!r}: "
            f"{session_state.get('primitive_rejection_feedback')}"
        )
        accepted = list(session_state.get("accepted_primitive_program") or [])
        assert accepted and accepted[-1]["outline_id"] == "RECOVERY_SEQ4"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. Agentic retrieval loop (need_context -> served_context -> authored plan)
# ---------------------------------------------------------------------------


def test_agentic_retrieval_loop_serves_requested_refs() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        # First turn: ask for context.
        decision1, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response={
                "thought": "I need the active event details and the pick contract.",
                "decision": "need_context",
                "outline_id": "RECOVERY_SEQ3",
                "resource_jid": "ur5e@localhost",
                "context_requests": [
                    "/outline_event",
                    "/resources/ur5e@localhost/current_pose",
                    "/primitive_contracts/grasp_part",
                    "/parts/LG/observed_pose",
                ],
                "primitive_steps": [],
                "rationale": "",
                "notes": [],
            },
            prepared_bridge_request=prepared,
            planner=planner,
        )
        assert decision1 == "need_context"
        served = dict(session_state.get("primitive_served_context") or {})
        assert "/outline_event" in served
        assert "/primitive_contracts/grasp_part" in served
        assert str(dict(served["/primitive_contracts/grasp_part"]).get("name")) == "grasp_part"
        # Cursor must not advance on need_context.
        assert session_state.get("primitive_generation_cursor") == next(
            i for i, r in enumerate(session_state["accepted_outline_prefix"])
            if r["outline_id"] == "RECOVERY_SEQ3"
        )
        # Second turn: author using retrieved context.
        decision2, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response=_authored_response(
                outline_id="RECOVERY_SEQ3",
                resource_jid="ur5e@localhost",
                primitive_steps=_acquire_seq3_steps(),
            ),
            prepared_bridge_request=prepared,
            planner=planner,
        )
        assert decision2 in {"primitive_steps_ready", "draft_ready"}
        # Served context is cleared after the event is accepted.
        assert session_state.get("primitive_served_context") == {}

    asyncio.run(_run())


def test_invalid_context_request_errors_render_in_next_prompt() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response={
                "thought": "I want a capability that does not exist.",
                "decision": "need_context",
                "outline_id": "RECOVERY_SEQ3",
                "resource_jid": "ur5e@localhost",
                "context_requests": ["/capability_decompositions/release_approach"],
                "primitive_steps": [],
                "rationale": "",
                "notes": [],
            },
            prepared_bridge_request=prepared,
            planner=planner,
        )

        assert decision == "need_context"
        text = _primitive_prompt_text(prepared=prepared, session_state=session_state)
        assert "Context Request Errors (previous turn)" in text
        assert "/capability_decompositions/release_approach" in text
        assert "no capability decomposition registered" in text

    asyncio.run(_run())


def test_capability_decomposition_request_for_primitive_name_points_to_primitive_contract() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ1")
        decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response={
                "thought": "I want the release primitive details.",
                "decision": "need_context",
                "outline_id": "RECOVERY_SEQ1",
                "resource_jid": "ur5e@localhost",
                "context_requests": ["/capability_decompositions/release_part"],
                "primitive_steps": [],
                "rationale": "",
                "notes": [],
            },
            prepared_bridge_request=prepared,
            planner=planner,
        )

        assert decision == "need_context"
        text = _primitive_prompt_text(prepared=prepared, session_state=session_state)
        assert "/capability_decompositions/release_part" in text
        assert "is a primitive name, not a capability decomposition" in text
        assert "/primitive_contracts/release_part" in text
        assert "place_approach, place_insert" in text

    asyncio.run(_run())


def test_home_named_pose_is_no_longer_rejected_by_authored_plan_gate() -> None:
    async def _run() -> None:
        _, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        active_event = _active_event(session_state)
        visible_catalog = primitive_mode._primitive_catalog_for_resource(
            prepared,
            "ur5e@localhost",
        )

        steps, error = primitive_mode._validate_authored_plan(
            parsed_response=_authored_response(
                outline_id="RECOVERY_SEQ3",
                resource_jid="ur5e@localhost",
                primitive_steps=[
                    {
                        "primitive": "move_to_named_pose",
                        "params": {"pose_name": "home"},
                    }
                ],
            ),
            active_event=active_event,
            visible_catalog=visible_catalog,
        )

        assert error is None
        assert steps == [
            {
                "primitive": "move_to_named_pose",
                "params": {"pose_name": "home"},
            }
        ]

    asyncio.run(_run())


def test_non_advertised_named_pose_is_still_rejected() -> None:
    async def _run() -> None:
        _, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        active_event = _active_event(session_state)

        _result, feedback = primitive_mode._validate_single_event_primitive_steps(
            session_state=session_state,
            prepared_bridge_request=prepared,
            outline_event=active_event,
            primitive_steps=[
                {
                    "primitive": "move_to_named_pose",
                    "params": {"pose_name": "bogus_pose"},
                }
            ],
        )

        codes = {str(row.get("constraint_code") or "") for row in feedback}
        assert "named_pose_unavailable" in codes

    asyncio.run(_run())


def test_projected_snapshot_mismatch_checks_resource_state_in_primitive_mode() -> None:
    matches, reason = primitive_mode._primitive_projected_snapshot_matches_event(
        {"held_part": None, "current_state": "picked"},
        {"expected_end_state": {"held_part": None, "resource_state": "idle"}},
    )

    assert not matches
    assert reason is not None
    assert "current_state" in reason
    assert "'idle'" in reason
    assert "'picked'" in reason


def test_projected_snapshot_mismatch_checks_resource_state_in_v2_mode() -> None:
    matches, reason = multi_turn_v2_mode._primitive_projected_snapshot_matches_event(
        {"held_part": None, "current_state": "picked"},
        {"expected_end_state": {"held_part": None, "resource_state": "idle"}},
    )

    assert not matches
    assert reason is not None
    assert "current_state" in reason
    assert "'idle'" in reason
    assert "'picked'" in reason


def test_repeated_validator_feedback_pauses_on_fifth_repeat() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        response = _authored_response(
            outline_id="RECOVERY_SEQ3",
            resource_jid="ur5e@localhost",
            primitive_steps=_non_target_grasp_steps(),
        )

        for _ in range(4):
            decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
                session_state=session_state,
                parsed_response=deepcopy(response),
                prepared_bridge_request=prepared,
                planner=planner,
            )
            assert decision == "need_primitive_revision"

        decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response=deepcopy(response),
            prepared_bridge_request=prepared,
            planner=planner,
        )

        assert decision == "primitive_event_stuck"
        assert session_state.get("status") == "paused_after_primitive_stuck"
        feedback = list(session_state.get("primitive_rejection_feedback") or [])
        assert feedback and feedback[0]["constraint_code"] == "primitive_event_stuck"
        diagnostics = list(session_state.get("primitive_escalation_diagnostics") or [])
        assert diagnostics and diagnostics[0]["trigger"] == "same_signature_streak"
        assert diagnostics[0]["same_signature_streak"] >= 5

    asyncio.run(_run())


def test_mixed_no_progress_turns_pause_on_sixth_turn() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        responses = [
            {
                "thought": "Need a bogus ref A.",
                "decision": "need_context",
                "outline_id": "RECOVERY_SEQ3",
                "resource_jid": "ur5e@localhost",
                "context_requests": ["/missing_ref/a"],
                "primitive_steps": [],
                "rationale": "",
                "notes": [],
            },
            {
                "thought": "Need a bogus ref B.",
                "decision": "need_context",
                "outline_id": "RECOVERY_SEQ3",
                "resource_jid": "ur5e@localhost",
                "context_requests": ["/missing_ref/b"],
                "primitive_steps": [],
                "rationale": "",
                "notes": [],
            },
            {
                "thought": "",
                "decision": "need_primitive_revision",
                "outline_id": "RECOVERY_SEQ3",
                "resource_jid": "ur5e@localhost",
                "context_requests": [],
                "primitive_steps": [],
                "rationale": "gap-a",
                "notes": [],
            },
            {
                "thought": "",
                "decision": "need_primitive_revision",
                "outline_id": "RECOVERY_SEQ3",
                "resource_jid": "ur5e@localhost",
                "context_requests": [],
                "primitive_steps": [],
                "rationale": "gap-b",
                "notes": [],
            },
            _authored_response(
                outline_id="RECOVERY_SEQ3",
                resource_jid="ur5e@localhost",
                primitive_steps=[{"primitive": "move_pose", "params": {}}],
            ),
            _authored_response(
                outline_id="RECOVERY_SEQ3",
                resource_jid="ur5e@localhost",
                primitive_steps=[{"primitive": "get_current_pose", "params": {}}],
            ),
        ]

        for response in responses[:5]:
            decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
                session_state=session_state,
                parsed_response=deepcopy(response),
                prepared_bridge_request=prepared,
                planner=planner,
            )
            assert decision in {"need_context", "need_primitive_revision"}

        decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response=deepcopy(responses[5]),
            prepared_bridge_request=prepared,
            planner=planner,
        )

        assert decision == "primitive_event_stuck"
        assert session_state.get("status") == "paused_after_primitive_stuck"
        diagnostics = list(session_state.get("primitive_escalation_diagnostics") or [])
        assert diagnostics and diagnostics[0]["trigger"] == "no_progress_turns"
        assert diagnostics[0]["no_progress_turns"] >= 6

    asyncio.run(_run())


def test_stuck_summary_keeps_repeated_release_blockers_visible() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ1")
        response = _authored_response(
            outline_id="RECOVERY_SEQ1",
            resource_jid="ur5e@localhost",
            primitive_steps=_ungrounded_release_steps(),
        )

        for _ in range(5):
            decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
                session_state=session_state,
                parsed_response=deepcopy(response),
                prepared_bridge_request=prepared,
                planner=planner,
            )

        assert decision == "primitive_event_stuck"
        diagnostics = list(session_state.get("primitive_escalation_diagnostics") or [])
        assert diagnostics
        blocker_codes = {
            row["constraint_code"]
            for row in (diagnostics[0].get("repeated_validator_blockers") or [])
            if isinstance(row, dict)
        }
        assert "release_target_grounding_required" in blocker_codes
        assert "trace_fact_required" in blocker_codes
        reason = str((diagnostics[0] or {}).get("reason") or "")
        assert "release_target_grounding_required" in reason
        assert "trace_fact_required" in reason

    asyncio.run(_run())


def test_need_context_with_new_ref_resets_stuck_guard() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")

        for ref in ("/missing_ref/a", "/missing_ref/b"):
            decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
                session_state=session_state,
                parsed_response={
                    "thought": "",
                    "decision": "need_context",
                    "outline_id": "RECOVERY_SEQ3",
                    "resource_jid": "ur5e@localhost",
                    "context_requests": [ref],
                    "primitive_steps": [],
                    "rationale": "",
                    "notes": [],
                },
                prepared_bridge_request=prepared,
                planner=planner,
            )
            assert decision == "need_context"

        decision, turn_entry = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response={
                "thought": "Need the active event body.",
                "decision": "need_context",
                "outline_id": "RECOVERY_SEQ3",
                "resource_jid": "ur5e@localhost",
                "context_requests": ["/outline_event"],
                "primitive_steps": [],
                "rationale": "",
                "notes": [],
            },
            prepared_bridge_request=prepared,
            planner=planner,
        )

        assert decision == "need_context"
        assert turn_entry["newly_served_context_refs"] == ["/outline_event"]
        guard = dict(session_state.get("primitive_event_guard") or {})
        assert guard["no_progress_turns"] == 0
        assert guard["same_signature_streak"] == 0

    asyncio.run(_run())


def test_accepting_event_resets_stuck_guard() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")

        for ref in ("/missing_ref/a", "/missing_ref/b"):
            decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
                session_state=session_state,
                parsed_response={
                    "thought": "",
                    "decision": "need_context",
                    "outline_id": "RECOVERY_SEQ3",
                    "resource_jid": "ur5e@localhost",
                    "context_requests": [ref],
                    "primitive_steps": [],
                    "rationale": "",
                    "notes": [],
                },
                prepared_bridge_request=prepared,
                planner=planner,
            )
            assert decision == "need_context"

        decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response=_authored_response(
                outline_id="RECOVERY_SEQ3",
                resource_jid="ur5e@localhost",
                primitive_steps=_acquire_seq3_steps(),
            ),
            prepared_bridge_request=prepared,
            planner=planner,
        )

        assert decision in {"primitive_steps_ready", "draft_ready"}
        assert session_state.get("primitive_event_guard") == {}

    asyncio.run(_run())


def test_legacy_need_outline_revision_maps_to_primitive_blocked() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ1")
        session_state["primitive_served_context"] = {
            "/outline_event": {"outline_id": "RECOVERY_SEQ1"}
        }
        session_state["primitive_context_errors"] = [
            {"context_ref": "/parts/MCP/target", "reason": "target mismatch remains"}
        ]

        decision, turn_entry = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response={
                "thought": "The active event is contradictory.",
                "decision": "need_outline_revision",
                "outline_id": "RECOVERY_SEQ1",
                "resource_jid": "ur5e@localhost",
                "context_requests": [],
                "primitive_steps": [],
                "rationale": (
                    "outline says prusa-mk3 while grounded goal remains assembly_board-v1"
                ),
                "notes": [],
            },
            prepared_bridge_request=prepared,
            planner=planner,
        )

        assert decision == "primitive_blocked"
        assert session_state.get("status") == "paused_after_primitive_blocked"
        assert session_state.get("primitive_served_context") == {
            "/outline_event": {"outline_id": "RECOVERY_SEQ1"}
        }
        assert session_state.get("primitive_context_errors") == [
            {"context_ref": "/parts/MCP/target", "reason": "target mismatch remains"}
        ]
        feedback = list(session_state.get("primitive_rejection_feedback") or [])
        assert feedback and feedback[0]["constraint_code"] == "primitive_blocked"
        assert "assembly_board-v1" in feedback[0]["reason"]
        assert turn_entry["primitive_context_errors"][0]["context_ref"] == "/parts/MCP/target"

    asyncio.run(_run())


def test_legacy_primitive_event_ready_maps_to_primitive_steps_ready() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")

        decision, turn_entry = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response=_authored_response(
                outline_id="RECOVERY_SEQ3",
                resource_jid="ur5e@localhost",
                primitive_steps=_acquire_seq3_steps(),
                decision="primitive_event_ready",
            ),
            prepared_bridge_request=prepared,
            planner=planner,
        )

        assert decision == "primitive_steps_ready"
        assert turn_entry["primitive_response"]["decision"] == "primitive_steps_ready"
        assert turn_entry["primitive_response"]["legacy_decision"] == "primitive_event_ready"

    asyncio.run(_run())


def test_primitive_event_stuck_transition_is_self_loop() -> None:
    assert (
        multi_turn_v2_mode.transition_multi_turn_phase(
            "primitive_generation",
            "primitive_event_stuck",
        )
        == "primitive_generation"
    )


def test_primitive_blocked_transition_is_self_loop() -> None:
    assert (
        multi_turn_v2_mode.transition_multi_turn_phase(
            "primitive_generation",
            "primitive_blocked",
        )
        == "primitive_generation"
    )


def test_primitive_turn_logs_decision_summary(caplog: Any) -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        logging.disable(logging.NOTSET)
        primitive_mode._logger.addHandler(caplog.handler)

        try:
            with caplog.at_level(logging.INFO, logger=primitive_mode.__name__):
                decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
                    session_state=session_state,
                    parsed_response={
                        "thought": "Need the full event and grasp contract before authoring.",
                        "decision": "need_context",
                        "outline_id": "RECOVERY_SEQ3",
                        "resource_jid": "ur5e@localhost",
                        "context_requests": ["/outline_event", "/primitive_contracts/grasp_part"],
                        "primitive_steps": [],
                        "rationale": "Need grounded contract details.",
                        "notes": ["agentic retrieval turn"],
                    },
                    prepared_bridge_request=prepared,
                    planner=planner,
                )
        finally:
            primitive_mode._logger.removeHandler(caplog.handler)

        assert decision == "need_context"

    asyncio.run(_run())
    assert "decision=need_context" in caplog.text
    assert "refs=2" in caplog.text
    assert "thought=" not in caplog.text
    assert "rationale=" not in caplog.text


def test_authoring_memo_written_on_accept() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response=_authored_response(
                outline_id="RECOVERY_SEQ3",
                resource_jid="ur5e@localhost",
                primitive_steps=_acquire_seq3_steps(),
            ),
            prepared_bridge_request=prepared,
            planner=planner,
        )

        assert decision in {"primitive_steps_ready", "draft_ready"}
        memo = list(session_state.get("primitive_authoring_memo") or [])
        assert memo
        assert memo[-1]["action"] == "acquire"
        assert memo[-1]["part"] == "LG"
        assert memo[-1]["part_name"] == "LG"
        assert memo[-1]["steps_summary"] == [
            "detect_parts",
            "compute_pick_targets",
            "move_cartesian",
            "move_cartesian",
            "grasp_part",
        ]

    asyncio.run(_run())


def test_authoring_memo_rendered_for_matching_event() -> None:
    async def _run() -> None:
        _, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        active_index = int(session_state["primitive_generation_cursor"])
        session_state["accepted_outline_prefix"][active_index]["part_name"] = "SG"
        session_state["primitive_authoring_memo"] = [
            {
                "action": "acquire",
                "part": "SG",
                "part_name": "SG",
                "resource_jid": "xarm6@localhost",
                "outline_id": "PRIOR_ACQUIRE_SG",
                "steps_summary": ["detect_parts", "compute_pick_targets", "grasp_part"],
                "accepted_at_turn": 3,
            }
        ]

        text = _primitive_prompt_text(prepared=prepared, session_state=session_state)
        assert "Prior Accepted Decompositions" in text
        assert "PRIOR_ACQUIRE_SG" in text
        assert '"part": "SG"' in text
        assert "compute_pick_targets" in text

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4. Derived-offset path: no hardcoded retreat/descent constants in source
# ---------------------------------------------------------------------------


def test_no_hardcoded_retreat_constants_in_source() -> None:
    root = Path(__file__).resolve().parents[1]
    target = (
        root
        / "cais_spade_llm"
        / "agents"
        / "intelligent_product"
        / "replanner"
        / "llm_bridge"
        / "modes"
        / "multi_turn_primitive_generation.py"
    )
    text = target.read_text(encoding="utf-8")
    for banned in (
        "_DEFAULT_PICK_RETREAT_DZ_M",
        "_DEFAULT_DESCEND_DZ_M",
        "_DEFAULT_PLACE_RETREAT_DZ_M",
        "_compose_pick_steps",
        "_compose_release_steps",
        "_compose_guided_primitive_steps",
        "_retrieve_decomposition_examples",
        "_load_decomposition_examples",
        "_DECOMPOSITION_EXAMPLES",
        "primitive_decomposition_examples.json",
    ):
        assert banned not in text, f"{banned!r} must be removed from {target.name}"


# ---------------------------------------------------------------------------
# 5. Witness violation: grasp_part after motion not bound to target_pose
# ---------------------------------------------------------------------------


def test_witness_violation_rejects_grasp_after_non_target_motion() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        steps = [
            {"primitive": "detect_parts", "params": {"part_name": "LG"}, "store_as": "obs"},
            {
                "primitive": "compute_pick_targets",
                "params": {"part_name": "LG"},
                "store_as": "pick",
            },
            # Motion bound to APPROACH_POSE only — no motion lands on target_pose.
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": _ctx("pick", "approach_pose.x"),
                    "y": _ctx("pick", "approach_pose.y"),
                    "z": _ctx("pick", "approach_pose.z"),
                },
            },
            {
                "primitive": "grasp_part",
                "params": {"model_name": "LG", "part_name": "LG"},
            },
        ]
        decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response=_authored_response(
                outline_id="RECOVERY_SEQ3",
                resource_jid="ur5e@localhost",
                primitive_steps=steps,
            ),
            prepared_bridge_request=prepared,
            planner=planner,
        )
        assert decision == "need_primitive_revision"
        feedback = list(session_state.get("primitive_rejection_feedback") or [])
        assert feedback
        codes = {row.get("constraint_code") for row in feedback}
        assert "trace_fact_required" in codes or "pick_target_grounding_required" in codes
        # Cursor stays put.
        target_index = next(
            i for i, r in enumerate(session_state["accepted_outline_prefix"])
            if r["outline_id"] == "RECOVERY_SEQ3"
        )
        assert session_state.get("primitive_generation_cursor") == target_index

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 6. Need-context refusal surfaces served context
# ---------------------------------------------------------------------------


def test_need_context_empty_requests_is_schema_violation() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response={
                "thought": "",
                "decision": "need_context",
                "outline_id": "RECOVERY_SEQ3",
                "resource_jid": "ur5e@localhost",
                "context_requests": [],
                "primitive_steps": [],
            },
            prepared_bridge_request=prepared,
            planner=planner,
        )
        assert decision == "need_primitive_revision"
        feedback = list(session_state.get("primitive_rejection_feedback") or [])
        assert feedback[0]["constraint_code"] == "primitive_schema_violation"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 7. Need-primitive-revision refusal requires a rationale
# ---------------------------------------------------------------------------


def test_need_primitive_revision_requires_rationale() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        # Without rationale: schema violation.
        decision_no_reason, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response={
                "thought": "",
                "decision": "need_primitive_revision",
                "outline_id": "RECOVERY_SEQ3",
                "resource_jid": "ur5e@localhost",
                "context_requests": [],
                "primitive_steps": [],
                "rationale": "",
                "notes": [],
            },
            prepared_bridge_request=prepared,
            planner=planner,
        )
        assert decision_no_reason == "need_primitive_revision"
        assert session_state["primitive_rejection_feedback"][0]["constraint_code"] == (
            "primitive_schema_violation"
        )
        # With rationale: accepted as a refusal with the named contract gap.
        decision_with_reason, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response={
                "thought": "",
                "decision": "need_primitive_revision",
                "outline_id": "RECOVERY_SEQ3",
                "resource_jid": "ur5e@localhost",
                "rationale": (
                    "no visible primitive establishes part orientation for flipped-SG insert"
                ),
                "context_requests": [],
                "primitive_steps": [],
                "notes": [],
            },
            prepared_bridge_request=prepared,
            planner=planner,
        )
        assert decision_with_reason == "need_primitive_revision"
        feedback = list(session_state["primitive_rejection_feedback"])
        assert feedback[0]["constraint_code"] == "primitive_revision_requested"
        assert "flipped-SG" in feedback[0]["reason"]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 8. Hidden primitive rejection (move_pose / get_current_pose)
# ---------------------------------------------------------------------------


def test_hidden_primitive_is_rejected() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        for hidden_name in ("move_pose", "get_current_pose"):
            state = deepcopy(session_state)
            decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
                session_state=state,
                parsed_response=_authored_response(
                    outline_id="RECOVERY_SEQ3",
                    resource_jid="ur5e@localhost",
                    primitive_steps=[
                        {"primitive": hidden_name, "params": {}},
                    ],
                ),
                prepared_bridge_request=prepared,
                planner=planner,
            )
            assert decision == "need_primitive_revision"
            feedback = list(state["primitive_rejection_feedback"])
            assert feedback[0]["constraint_code"] == "primitive_schema_violation"
            assert hidden_name in feedback[0]["reason"]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 9. Unmodeled-recovery coverage: trace validator accepts a non-canonical
# sequence that no deleted _compose_* recipe could have produced.
# ---------------------------------------------------------------------------


def test_unmodeled_recovery_sequence_accepted() -> None:
    """A two-phase plan for SEQ3 that interleaves an extra observation step and
    a retreat before the grasp. None of the deleted ``_compose_*`` helpers
    generated such a shape; the LLM must author it directly."""

    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        steps = [
            {"primitive": "detect_parts", "params": {"part_name": "LG"}, "store_as": "obs1"},
            # Second observation, which no composer recipe produced.
            {"primitive": "detect_parts", "params": {"part_name": "LG"}, "store_as": "obs2"},
            {
                "primitive": "compute_pick_targets",
                "params": {"part_name": "LG"},
                "store_as": "pick",
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": _ctx("pick", "approach_pose.x"),
                    "y": _ctx("pick", "approach_pose.y"),
                    "z": _ctx("pick", "approach_pose.z"),
                },
            },
            {
                "primitive": "move_cartesian",
                "params": {
                    "x": _ctx("pick", "target_pose.x"),
                    "y": _ctx("pick", "target_pose.y"),
                    "z": _ctx("pick", "target_pose.z"),
                },
            },
            {
                "primitive": "grasp_part",
                "params": {"model_name": "LG", "part_name": "LG"},
            },
        ]
        decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response=_authored_response(
                outline_id="RECOVERY_SEQ3",
                resource_jid="ur5e@localhost",
                primitive_steps=steps,
            ),
            prepared_bridge_request=prepared,
            planner=planner,
        )
        assert decision in {"primitive_steps_ready", "draft_ready"}, (
            f"rejected: {session_state.get('primitive_rejection_feedback')}"
        )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 10. Load-bearing-LLM ablation: empty primitive_steps halts the phase
# ---------------------------------------------------------------------------


def test_empty_primitive_steps_halts_phase() -> None:
    async def _run() -> None:
        planner, prepared, session_state = await _harness()
        _set_cursor_to(session_state, "RECOVERY_SEQ3")
        pre_cursor = session_state["primitive_generation_cursor"]
        decision, _ = await multi_turn_v2_mode._handle_primitive_generation_phase(
            session_state=session_state,
            parsed_response={
                "thought": "",
                "decision": "primitive_steps_ready",
                "outline_id": "RECOVERY_SEQ3",
                "resource_jid": "ur5e@localhost",
                "context_requests": [],
                "primitive_steps": [],
                "rationale": "",
                "notes": [],
            },
            prepared_bridge_request=prepared,
            planner=planner,
        )
        assert decision == "need_primitive_revision"
        assert session_state["primitive_generation_cursor"] == pre_cursor
        assert session_state.get("accepted_primitive_program") in (None, [])
        feedback = list(session_state["primitive_rejection_feedback"])
        assert feedback[0]["constraint_code"] == "primitive_schema_violation"
        assert "non-empty primitive_steps" in feedback[0]["reason"]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Sanity: the phase handler is still wired into the multi-turn v2 dispatcher
# ---------------------------------------------------------------------------


def test_primitive_phase_handler_is_wired() -> None:
    assert (
        multi_turn_v2_mode._PHASE_HANDLERS["primitive_generation"]
        is primitive_mode._handle_primitive_generation_phase
    )

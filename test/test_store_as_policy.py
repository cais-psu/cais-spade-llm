from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any

from test.test_case3_bridge_dryrun import _prepare_bridge_dryrun_harness
from test.test_primitive_generation_authored import (
    _active_event,
    _authored_response,
    _harness,
    _set_cursor_to,
)

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
    multi_turn as legacy_multi_turn_mode,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
    multi_turn_v2 as multi_turn_v2_mode,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
    multi_turn_primitive_generation as primitive_mode,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.bridge_generation import (
    normalize_bridge_turn_response,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.function_synthesis import (
    validate_synthesized_function,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.v3.mutation_types import (
    SynthesizedTaskFn,
)
from cais_spade_llm.resources.resource_profile import get_resource_profile


def test_active_primitive_plan_rejects_legacy_store_as_on_move_to_named_pose() -> None:
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
                        "params": {"pose_name": "home", "speed": 0.8},
                        "store_as": "recover_home_pose",
                    }
                ],
            ),
            active_event=active_event,
            visible_catalog=visible_catalog,
        )

        assert steps is None
        assert error is not None
        assert "legacy field 'store_as'" in error

    asyncio.run(_run())


def test_legacy_multi_turn_grounding_dedupes_by_observation_key() -> None:
    async def _run() -> None:
        _, _, planner, prepared_bridge_request = await _prepare_bridge_dryrun_harness()
        session_state = legacy_multi_turn_mode.build_multi_turn_session_seed(
            prepared_bridge_request
        )
        fact_key = json.dumps(
            {"entity": "LG", "fact_type": "part_pose"},
            sort_keys=True,
            ensure_ascii=True,
        )
        observation_payload = {
            "part_name": "LG",
            "x": 0.0,
            "y": 0.2,
            "z": 1.035,
            "pose": {"x": 0.0, "y": 0.2, "z": 1.035},
        }
        session_state["observation_store"] = {
            "observed_pose_LG": deepcopy(observation_payload),
        }
        session_state["observation_fact_ledger"] = {
            fact_key: {
                "fact_key": fact_key,
                "fact_type": "part_pose",
                "entity": "LG",
                "entity_kind": "part",
                "primitive": "detect_parts",
                "params": {"part_name": "LG"},
                "output": deepcopy(observation_payload),
                "turn_index": 1,
                "validity": "current",
                "freshness": "current_session",
                "observation_key": "observed_pose_LG",
            }
        }

        results, error = await legacy_multi_turn_mode._execute_observe_requests(
            planner,
            prepared_bridge_request,
            session_state,
            [{"fact_type": "part_pose", "entity": "LG", "reason": "repeat"}],
        )

        assert results == []
        assert error is not None
        assert "already succeeded earlier" in error

    asyncio.run(_run())


def test_v3_rejects_store_as_in_observe_and_function_steps() -> None:
    async def _run() -> None:
        _, _, _planner, prepared_bridge_request = await _prepare_bridge_dryrun_harness()

        observe_row, observe_error = normalize_bridge_turn_response(
            raw={
                "type": "observe",
                "resource_jid": "ur5e@localhost",
                "primitive": "detect_parts",
                "params": {"part_name": "LG"},
                "store_as": "detected_lg",
            },
            available_resource_jids=["ur5e@localhost", "xarm6@localhost"],
            allowed_observation_primitives=["detect_parts"],
            bridge_resources=dict(prepared_bridge_request.get("bridge_resources") or {}),
        )

        assert observe_row is None
        assert observe_error is not None
        assert "store_as is no longer supported" in observe_error

        bridge_entry = dict(
            dict(prepared_bridge_request.get("bridge_resources") or {}).get("ur5e@localhost")
            or {}
        )
        primitive_catalog = list(bridge_entry.get("primitive_catalog") or [])
        resource_snapshot = dict(bridge_entry.get("bridge_snapshot") or {})
        fn_def = SynthesizedTaskFn(
            name="recover_home",
            intent="Return the robot to home.",
            resource_constraints={},
            inputs={},
            preconditions={},
            effects={},
            primitive_program=[
                {
                    "primitive": "move_to_named_pose",
                    "params": {"pose_name": "home", "speed": 0.8},
                    "store_as": "recover_home_pose",
                }
            ],
            expected_post_state={},
        )

        is_valid, _projected, errors = validate_synthesized_function(
            fn_def,
            primitive_catalog,
            resource_snapshot,
            observation_store={},
            capability_flags={},
            resource_jid="ur5e@localhost",
        )

        assert not is_valid
        assert any("legacy field 'store_as' is not supported" in error for error in errors)

    asyncio.run(_run())


def test_move_to_named_pose_remains_non_output_producing() -> None:
    profile = get_resource_profile("robot")

    assert "move_to_named_pose" not in dict(profile.preview_output_map or {})
    assert "move_to_named_pose" not in dict(profile.extract_output_map or {})
    assert "move_to_named_pose" not in dict(profile.event_fact_contract_map or {})

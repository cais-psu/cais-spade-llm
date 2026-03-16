from __future__ import annotations

from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_adapters import (
    bridge_adapter_capabilities,
    canonical_bridge_event,
    canonical_bridge_resource,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_compiler import (
    BridgeCompilerMixin,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_generation import (
    _bridge_event_contract_error,
    normalize_bridge_turn_response,
)
from cais_spade_llm.resources.resource_profile import (
    ResourceProfile,
    all_registered_operation_kinds,
    get_resource_profile,
    get_resource_profile_for_agent,
    register_resource_profile,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
    build_primitive_catalog,
    expected_snapshot_from_bridge_snapshot,
)
from cais_spade_llm.resources.machine.printer_profile import (
    PRINTER_PROFILE,
)
from cais_spade_llm.resources.robot.robot_profile import (
    ROBOT_PROFILE,
)


def _drilling_facet(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "spindle_speed": snapshot.get("spindle_speed"),
        "bit_position": snapshot.get("bit_position"),
    }


def _drilling_event_family(event: dict[str, Any]) -> str:
    event_name = str(event.get("event_name", "") or "").strip().lower()
    if event_name.startswith("start_drill"):
        return "start_drill"
    if event_name.startswith("stop_drill"):
        return "stop_drill"
    return ""


DRILLING_PROFILE = ResourceProfile(
    resource_type="drilling",
    snapshot_fields=("current_state", "spindle_speed", "bit_position"),
    facet_key="drilling",
    facet_builder=_drilling_facet,
    sync_map={
        "spindle_speed": "_spindle_speed",
        "bit_position": "_bit_position",
    },
    primitive_kind_map={
        "start_drill": "drilling",
        "stop_drill": "drilling",
    },
    event_family_resolver=_drilling_event_family,
    family_to_primitive={
        "start_drill": "start_drill",
        "stop_drill": "stop_drill",
    },
    capability_flags={"supports_drilling": True},
)


register_resource_profile(DRILLING_PROFILE)


def _bridge_resource_entry(
    *,
    resource_jid: str,
    resource_type: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    bridge_snapshot = canonical_bridge_resource(
        resource_jid=resource_jid,
        resource_type=resource_type,
        snapshot=deepcopy(snapshot),
        modeled_state={},
    )
    return {
        "resource_jid": resource_jid,
        "resource_type": resource_type,
        "bridge_snapshot": bridge_snapshot,
        "resource_core": deepcopy(bridge_snapshot.get("resource_core") or {}),
        "resource_facets": deepcopy(bridge_snapshot.get("resource_facets") or {}),
        "primitive_catalog": [],
        "bridge_adapter": bridge_adapter_capabilities(resource_type, primitive_catalog=[]),
    }


class _DummyCompiler(BridgeCompilerMixin):
    def _bridge_effective_resource_facts(
        self,
        *,
        bridge_resources: dict[str, dict[str, Any]],
        projected_resource_snapshots: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        effective: dict[str, dict[str, Any]] = {}
        for resource_jid, entry in (bridge_resources or {}).items():
            snapshot = dict((entry or {}).get("bridge_snapshot") or {})
            effective[str(resource_jid)] = {
                "current_state": str(snapshot.get("current_state") or "idle"),
            }
        return effective

    def _bridge_event_semantic_kind(self, event: dict[str, Any]) -> str:
        return str(event.get("operation_family") or "").strip()


class _FakeAgentWithDescriptor:
    _RESOURCE_PROFILE = DRILLING_PROFILE
    _BRIDGE_PRIMITIVES = ["start_drill", "stop_drill"]

    def __init__(self) -> None:
        self.agent_name = "drill"
        self.jid = "drilling@localhost"
        self.static_capabilities = {"resource_type": "drilling"}
        self._current_state = "idle"
        self._spindle_speed = 0
        self._bit_position = "retracted"

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "resource_type": "drilling",
            "current_state": self._current_state,
            "spindle_speed": self._spindle_speed,
            "bit_position": self._bit_position,
        }

    async def start_drill(self, *, speed: int = 1200, **kwargs: Any) -> dict[str, Any]:
        """
        ---
        preconditions:
          current_state:
            equals: "idle"
        effects:
          current_state:
            set: "drilling"
          spindle_speed:
            set_from_param: "speed"
          bit_position:
            set: "engaged"
        ---
        Start the drill spindle.
        """
        self._current_state = "drilling"
        self._spindle_speed = int(speed)
        self._bit_position = "engaged"
        return {"success": True, "state": "drilling"}

    async def stop_drill(self, **kwargs: Any) -> dict[str, Any]:
        """
        ---
        preconditions:
          current_state:
            equals: "drilling"
        effects:
          current_state:
            set: "idle"
          spindle_speed:
            set: 0
          bit_position:
            set: "retracted"
        ---
        Stop the drill spindle.
        """
        self._current_state = "idle"
        self._spindle_speed = 0
        self._bit_position = "retracted"
        return {"success": True, "state": "idle"}


class _FakeAgentWithoutDescriptor:
    def __init__(self) -> None:
        self.static_capabilities = {"resource_type": "robot"}


def test_profile_registry_returns_builtin_and_default_profiles() -> None:
    assert get_resource_profile("robot") is ROBOT_PROFILE
    assert get_resource_profile("printer") is PRINTER_PROFILE
    assert get_resource_profile("unknown").resource_type == "resource"


def test_get_resource_profile_for_agent_prefers_agent_local_profile() -> None:
    assert get_resource_profile_for_agent(_FakeAgentWithDescriptor()) is DRILLING_PROFILE


def test_get_resource_profile_for_agent_falls_back_to_registry() -> None:
    assert get_resource_profile_for_agent(_FakeAgentWithoutDescriptor()) is ROBOT_PROFILE


def test_all_registered_operation_kinds_include_profile_kinds() -> None:
    kinds = all_registered_operation_kinds()
    assert "pick_place" in kinds
    assert "pause_job" in kinds
    assert "start_drill" in kinds


def test_bridge_event_contract_error_stays_generic() -> None:
    assert _bridge_event_contract_error({}) == "bridge_events.events[].operation_family is required"
    assert (
        _bridge_event_contract_error(
            {"_operation_family_explicit": True, "operation_family": ""}
        )
        == "bridge_events.events[].operation_family must not be empty"
    )

    unknown_error = _bridge_event_contract_error(
        {"_operation_family_explicit": True, "operation_family": "mystery_op"}
    )
    assert unknown_error is not None
    assert "not recognized for any registered resource profile" in unknown_error
    assert "in_gripper" not in unknown_error
    assert "picked" not in unknown_error
    assert "assembled" not in unknown_error
    assert "ready" not in unknown_error

    assert (
        _bridge_event_contract_error(
            {"_operation_family_explicit": True, "operation_family": "pick"}
        )
        is None
    )


def test_custom_profile_registration_smoke() -> None:
    assert get_resource_profile("drilling") is DRILLING_PROFILE


def test_explicit_operation_family_is_preserved() -> None:
    bridge_resources = {
        "drilling@localhost": _bridge_resource_entry(
            resource_jid="drilling@localhost",
            resource_type="drilling",
            snapshot={
                "resource_type": "drilling",
                "current_state": "idle",
                "spindle_speed": 0,
                "bit_position": "retracted",
            },
        )
    }
    normalized = canonical_bridge_event(
        {
            "event_name": "start_drill_cycle",
            "resource_jid": "drilling@localhost",
            "operation_family": "stop_drill",
            "expected_resource_delta": {"from": "drilling", "to": "idle"},
        },
        bridge_resources=bridge_resources,
    )
    assert normalized["operation_family"] == "stop_drill"


def test_unknown_resource_with_no_primitives_is_not_executable() -> None:
    capabilities = bridge_adapter_capabilities("buffer", primitive_catalog=[])
    assert capabilities["supports_executable_bridge"] is False


def test_new_resource_type_uses_profile_without_bridge_core_edits() -> None:
    agent = _FakeAgentWithDescriptor()
    catalog = build_primitive_catalog(agent)
    assert {entry["name"] for entry in catalog} == {"start_drill", "stop_drill"}
    assert {
        (entry.get("bridge_semantics") or {}).get("operation_kind")
        for entry in catalog
    } == {"drilling"}

    capabilities = bridge_adapter_capabilities("drilling", primitive_catalog=catalog)
    assert capabilities["supports_executable_bridge"] is True
    assert capabilities["supports_drilling"] is True

    bridge_snapshot = canonical_bridge_resource(
        resource_jid="drilling@localhost",
        resource_type="drilling",
        snapshot=agent._snapshot_state(),
        modeled_state={},
    )
    expected = expected_snapshot_from_bridge_snapshot(
        bridge_snapshot,
        resource_type="drilling",
    )
    assert expected == {
        "current_state": "idle",
        "spindle_speed": 0,
        "bit_position": "retracted",
    }

    bridge_resources = {
        "drilling@localhost": {
            "resource_jid": "drilling@localhost",
            "resource_type": "drilling",
            "primitive_catalog": deepcopy(catalog),
            "bridge_snapshot": deepcopy(bridge_snapshot),
            "resource_core": deepcopy(bridge_snapshot.get("resource_core") or {}),
            "resource_facets": deepcopy(bridge_snapshot.get("resource_facets") or {}),
            "bridge_adapter": deepcopy(capabilities),
        }
    }
    normalized_event = canonical_bridge_event(
        {
            "event_name": "start_drill_cycle",
            "resource_jid": "drilling@localhost",
            "expected_resource_delta": {"from": "idle", "to": "drilling"},
        },
        bridge_resources=bridge_resources,
    )
    assert normalized_event["operation_family"] == "start_drill"

    compiler = _DummyCompiler()
    compiled_plan, error = compiler._compile_bridge_events_to_macro_tasks(
        {
            "bridge_resources": bridge_resources,
            "obligation_targets": [],
        },
        approved_events=[normalized_event],
    )
    assert error is None
    macro_tasks = list((compiled_plan or {}).get("macro_tasks") or [])
    assert len(macro_tasks) == 1
    assert macro_tasks[0]["primitive_steps"] == [
        {"primitive": "start_drill", "params": {}}
    ]


def test_robot_profile_validates_pick_contract_outside_bridge_core() -> None:
    bridge_resources = {
        "ur5e@localhost": _bridge_resource_entry(
            resource_jid="ur5e@localhost",
            resource_type="robot",
            snapshot={
                "resource_type": "robot",
                "current_state": "idle",
                "current_location": "board_area",
                "held_part": None,
                "gripper_state": "open",
            },
        )
    }

    response, error = normalize_bridge_turn_response(
        raw={
            "type": "bridge_events",
            "events": [
                {
                    "event_name": "bad_pick",
                    "resource_jid": "ur5e@localhost",
                    "operation_family": "pick",
                    "part_name": "LG",
                    "expected_resource_delta": {"from": "idle", "to": "picked"},
                    "expected_part_delta": {
                        "part_name": "LG",
                        "from": "ready",
                        "to": "assembled",
                    },
                }
            ],
        },
        available_resource_jids=["ur5e@localhost"],
        allowed_observation_primitives=[],
        bridge_resources=bridge_resources,
    )

    assert response is None
    assert error is not None
    assert "inconsistent for robot resources" in error
    assert "'pick' requires expected_part_delta.to='in_gripper'" in error


def test_printer_profile_accepts_job_control_family() -> None:
    bridge_resources = {
        "printer@localhost": _bridge_resource_entry(
            resource_jid="printer@localhost",
            resource_type="printer",
            snapshot={
                "resource_type": "printer",
                "current_state": "printing",
                "current_location": "printer_bay",
                "active_job": "JOB_42",
                "job_state": "printing",
            },
        )
    }

    response, error = normalize_bridge_turn_response(
        raw={
            "type": "bridge_events",
            "events": [
                {
                    "event_name": "pause_job_for_bridge",
                    "resource_jid": "printer@localhost",
                    "operation_family": "pause_job",
                    "expected_resource_delta": {"from": "printing", "to": "paused"},
                }
            ],
        },
        available_resource_jids=["printer@localhost"],
        allowed_observation_primitives=[],
        bridge_resources=bridge_resources,
    )

    assert error is None
    assert response is not None
    assert response["events"][0]["operation_family"] == "pause_job"


def test_printer_profile_rejects_manipulator_style_event() -> None:
    bridge_resources = {
        "printer@localhost": _bridge_resource_entry(
            resource_jid="printer@localhost",
            resource_type="printer",
            snapshot={
                "resource_type": "printer",
                "current_state": "paused",
                "current_location": "printer_bay",
                "active_job": "JOB_42",
                "job_state": "paused",
            },
        )
    }

    response, error = normalize_bridge_turn_response(
        raw={
            "type": "bridge_events",
            "events": [
                {
                    "event_name": "printer_pick_lg",
                    "resource_jid": "printer@localhost",
                    "operation_family": "pick",
                    "part_name": "LG",
                    "expected_resource_delta": {"from": "idle", "to": "picked"},
                    "expected_part_delta": {
                        "part_name": "LG",
                        "from": "ready",
                        "to": "in_gripper",
                    },
                }
            ],
        },
        available_resource_jids=["printer@localhost"],
        allowed_observation_primitives=[],
        bridge_resources=bridge_resources,
    )

    assert response is None
    assert error is not None
    assert "inconsistent for printer resources" in error
    assert "manipulator-style part transitions are not supported" in error

"""Tests for PA-driven two-state resource grounding."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from rdflib import Namespace, URIRef
from rdflib.namespace import RDF

from cais_spade_llm.spec2primitives.adapters.moveit_plan_only import (
    _cartesian_waypoints,
)
from cais_spade_llm.spec2primitives.agents.pa import resource_grounding
from cais_spade_llm.spec2primitives.agents.pa.presentation_records import (
    AllocationEvidenceSource,
    load_allocation_presentation,
    load_or_create_allocation_presentation,
    load_or_create_evidence_presentation,
)
from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    ABoxSnapshot,
    initialize_interaction_abox,
    load_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
    CartesianReachabilityRequest,
    ReachabilityCheckRecord,
    ResourceGroundingError,
    RobotFrameLocationEvidenceError,
    candidate_resource_catalog,
    commit_resource_assignment,
    persist_cartesian_reachability,
    persist_pa_resource_selection,
    prepare_cartesian_reachability,
)
from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
    check_resource_reachability as _runtime_check_resource_reachability,
)
from cais_spade_llm.spec2primitives.agents.ra.feasibility_validation import (
    validate_provisional_allocation,
)
from cais_spade_llm.spec2primitives.config import load_workcell_profile
from cais_spade_llm.spec2primitives.ontology import (
    PredefinedWorkcellSnapshot,
    ResourceRegistrySnapshot,
    TBoxSnapshot,
    load_ppr_tbox,
    load_predefined_resource_registry,
    load_predefined_workcell,
)

SPEC2PRIMITIVES_ROOT = Path(__file__).parents[1]
TBOX_PATH = SPEC2PRIMITIVES_ROOT / "ontology/spec2primitives_ppr_tbox.owl"
PPR_NAMESPACE = "http://PAonto.com#"
PROCESS_IRI = "https://cais-spade-llm.local/process/assembly"
RESOURCE_NAMESPACE = "https://cais-spade-llm.local/resource/"
EVIDENCE_REF = "products/user_requirement/product_requirement.json"


class _CapturingFeasibilityRuntime:
    def __init__(self) -> None:
        self.requests: list[Mapping[str, object]] = []

    async def validate_plan_only_allocation(
        self,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        self.requests.append(dict(request))
        state_locations = request["state_locations"]
        return {
            "status": "accepted",
            "state_locations": {
                state_name: [
                    {
                        "evidence_handle": item["evidence_handle"],
                        "status": "accepted",
                        "message": "arbitrary-profile location accepted",
                        "error_code": 1,
                    }
                    for item in locations
                ]
                for state_name, locations in state_locations.items()
            },
            "feedback": None,
        }


class _CapturingCartesianRuntime:
    def __init__(self, *, status: str = "accepted") -> None:
        self.status = status
        self.requests: list[Mapping[str, object]] = []

    async def validate_plan_only_allocation(
        self,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        self.requests.append(dict(request))
        live_start_pose = {
            "frame_id": "world",
            "link_name": str(request["end_effector_link"]),
            "position_m": [0.0, -0.5, 1.3],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        ee_to_tcp = {
            "parent_link": str(request["end_effector_link"]),
            "child_link": str(request["tcp_link"]),
            "translation_m": [0.0, 0.0, -0.17],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        waypoints = _cartesian_waypoints(
            request,
            live_start_pose=live_start_pose,
            ee_to_tcp=ee_to_tcp,
        )

        def phase(name: str, roles: list[str]) -> Mapping[str, object]:
            phase_status = (
                "accepted" if self.status == "rejected" and name == "pick" else self.status
            )
            fraction = 1.0 if phase_status == "accepted" else 0.5
            error_code = 1 if phase_status == "accepted" else -1
            return {
                "phase": name,
                "status": phase_status,
                "waypoint_roles": roles,
                "fraction": fraction,
                "moveit_error_code": error_code,
                "terminal_state_available": phase_status == "accepted",
                "message": f"{name} Cartesian path {phase_status}",
            }

        return {
            "status": self.status,
            "live_start_pose": live_start_pose,
            "ee_to_tcp_transform": ee_to_tcp,
            "waypoints": waypoints,
            "phases": {
                "pick": phase(
                    "pick",
                    ["pick_approach", "grasp", "pick_retreat"],
                ),
                "place": phase(
                    "place",
                    ["transfer", "place_approach", "placement", "place_retreat"],
                ),
            },
            "feedback": (None if self.status == "accepted" else "Cartesian path rejected"),
            "motion_executed": False,
        }


def test_removed_need_and_automatic_selection_are_not_exported() -> None:
    assert not hasattr(resource_grounding, "ResourceAssignmentNeed")
    assert not hasattr(resource_grounding, "derive_resource_assignment_need")
    assert not hasattr(resource_grounding, "select_predefined_resource")
    assert "first_reachable_in_predefined_registry_order" not in Path(
        resource_grounding.__file__
    ).read_text(encoding="utf-8")


def test_catalog_is_unordered_and_never_selects_a_resource(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    tbox_a, registry_a, workcell_a, _ = _authorities(first)
    tbox_b, registry_b, workcell_b, _ = _authorities(
        second,
        resource_order=("ur5e", "xarm6"),
    )
    abox_a = _grounded_abox(first / "interaction", tbox_a)
    abox_b = _grounded_abox(second / "interaction", tbox_b)

    catalog_a = candidate_resource_catalog(abox_a, registry_a, workcell_a)
    catalog_b = candidate_resource_catalog(abox_b, registry_b, workcell_b)

    assert catalog_a == catalog_b
    assert set(catalog_a) == {"xarm6", "ur5e"}
    assert all("selected" not in entry for entry in catalog_a.values())


def test_arbitrary_process_and_resource_symbols_flow_without_code_dependencies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell = _arbitrary_authorities(tmp_path)
    process_iri = "https://example.local/process/joining"
    abox = _grounded_abox(
        root,
        tbox,
        product_requirement="join the panel",
        process_iri=process_iri,
    )
    current_path = _write_world_location(root, "neutral_a", (0.0, 0.2, 1.1))
    current_extra_path = _write_world_location(root, "neutral_c", (0.0, 0.1, 1.1))
    desired_path = _write_world_location(root, "neutral_b", (0.0, 0.5, 1.1))
    desired_extra_path = _write_world_location(root, "neutral_d", (0.0, 0.6, 1.1))
    presentation, handles = _allocation_presentation_for_locations(
        root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        location_paths=(
            current_path,
            current_extra_path,
            desired_path,
            desired_extra_path,
        ),
    )
    catalog = candidate_resource_catalog(abox, registry, workcell)

    assert tuple(catalog) == ("alpha_bot", "beta_bot")
    assert "gamma_bot" not in catalog
    reachability = _runtime_check_resource_reachability(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        resource_symbol="beta_bot",
        allocation_presentation=presentation,
        state_location_record_paths={
            "current_state": [
                (handles[_location_candidate_key(root, current_path)], current_path),
                (
                    handles[_location_candidate_key(root, current_extra_path)],
                    current_extra_path,
                ),
            ],
            "desired_state": [
                (handles[_location_candidate_key(root, desired_path)], desired_path),
                (
                    handles[_location_candidate_key(root, desired_extra_path)],
                    desired_extra_path,
                ),
            ],
        },
    )
    runtime = _CapturingFeasibilityRuntime()
    validation = asyncio.run(
        validate_provisional_allocation(
            runtime,
            interaction_root=root,
            workcell=workcell,
            reachability=reachability,
        )
    )
    selection = persist_pa_resource_selection(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        reachability=reachability,
        allocation_presentation=presentation,
        robot_agent_validation_path=validation.record_path,
    )
    committed = commit_resource_assignment(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        selection=selection,
    )

    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request["process_symbol"] == "joining"
    assert request["process_iri"] == process_iri
    assert request["feature_iri"].endswith("feature_0001")
    assert request["resource_symbol"] == "beta_bot"
    assert len(request["state_locations"]["current_state"]) == 2
    assert len(request["state_locations"]["desired_state"]) == 2
    assert request["validation_scope"] == "state_location_reachability"
    serialized_request = json.dumps(request, sort_keys=True)
    assert "assembly" not in serialized_request
    assert "xarm6" not in serialized_request
    assert "ur5e" not in serialized_request
    assert selection.process_symbol == "joining"
    assert selection.process_iri == process_iri
    assert selection.schema_version == 5
    assert selection.state_location_handles is not None
    assert selection.candidate_resource_symbols == ("alpha_bot", "beta_bot")
    assert committed.abox.accepted_assertion_count == 11
    assert (
        URIRef(f"{abox.namespace}process_execution_0001"),
        Namespace(PPR_NAMESPACE).runsOnResource,
        URIRef("https://example.local/resource/beta_bot"),
    ) in committed.abox.graph


@pytest.mark.parametrize(
    ("resource_symbol", "current", "desired", "expected_status"),
    [
        ("xarm6", (0.0, -0.7, 1.1), (0.0, -0.2, 1.1), "accepted"),
        ("ur5e", (0.0, 0.2, 1.1), (0.0, 0.8, 1.1), "accepted"),
        ("xarm6", (0.0, -0.7, 1.1), (0.0, 0.8, 1.1), "rejected"),
    ],
)
def test_reachability_checks_both_states_for_the_explicit_pa_resource(
    tmp_path: Path,
    resource_symbol: str,
    current: tuple[float, float, float],
    desired: tuple[float, float, float],
    expected_status: str,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, _ = _authorities(tmp_path)
    _grounded_abox(root, tbox)
    current_path = _write_world_location(root, "current", current)
    desired_path = _write_world_location(root, "desired", desired)

    check = check_resource_reachability(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        resource_symbol=resource_symbol,
        current_location_record_path=current_path,
        desired_location_record_path=desired_path,
    )

    assert check.resource_symbol == resource_symbol
    assert check.status == expected_status
    assert check.current_state.state_name == "current_state"
    assert check.desired_state.state_name == "desired_state"
    assert check.current_state.location_record_ref != (check.desired_state.location_record_ref)
    assert check.current_state.distance_from_reach_origin_m >= 0.0
    assert check.desired_state.distance_from_reach_origin_m >= 0.0
    assert "selected_resource" not in check.to_record()
    check.assert_unchanged()


def test_live_cartesian_reachability_ignores_static_workspace_box(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, manifest_paths = _authorities(tmp_path)
    _grounded_abox(root, tbox)
    fixture = _cartesian_grounding_fixture(
        root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        desired_y=0.144,
    )

    prepared = prepare_cartesian_reachability(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        resource_symbol="xarm6",
        allocation_presentation=fixture["presentation"],
        current_state_evidence_handle=fixture["current_handle"],
        desired_state_evidence_handle=fixture["desired_handle"],
        current_location_record_path=fixture["current_location"],
        desired_location_record_path=fixture["desired_location"],
        current_cad_correspondence_record_path=fixture["current_correspondence"],
        desired_cad_correspondence_record_path=fixture["desired_correspondence"],
    )
    runtime = _CapturingCartesianRuntime()
    validation = asyncio.run(
        validate_provisional_allocation(
            runtime,
            interaction_root=root,
            workcell=workcell,
            reachability=prepared,
        )
    )
    reachability = persist_cartesian_reachability(
        interaction_root=root,
        prepared=prepared,
        robot_agent_validation_path=validation.record_path,
    )
    selection = persist_pa_resource_selection(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        reachability=reachability,
        allocation_presentation=fixture["presentation"],
        robot_agent_validation_path=validation.record_path,
        state_location_handles={
            "current_state": (fixture["current_handle"],),
            "desired_state": (fixture["desired_handle"],),
        },
    )

    assert isinstance(prepared, CartesianReachabilityRequest)
    xarm_manifest = json.loads(manifest_paths["xarm6"].read_text(encoding="utf-8"))
    assert (
        xarm_manifest["xarm6"]["gazebo"]["static_capabilities"]["workspace_bounds"]["y_max_m"]
        == 0.1
    )
    assert prepared.desired_state.translation_m[1] == pytest.approx(0.144)
    assert prepared.cartesian_targets["place_object_center_m"] != list(
        prepared.desired_state.translation_m
    )
    assert runtime.requests[0]["motion_mode"] == "cartesian_pick_place"
    assert runtime.requests[0]["cartesian_parameters"] == {
        "max_step_m": 0.01,
        "jump_threshold": 0.0,
        "avoid_collisions": True,
        "minimum_fraction": 0.999,
    }
    assert reachability.schema_version == 3
    assert reachability.status == "accepted"
    assert "in_workspace" not in reachability.to_record()["desired_state"]
    assert "gripper_reach" not in json.dumps(reachability.to_record())
    assert validation.schema_version == 3
    assert validation.status == "accepted"
    assert validation.to_record()["motion_executed"] is False
    assert selection.schema_version == 5
    assert selection.state_location_handles == {
        "current_state": (fixture["current_handle"],),
        "desired_state": (fixture["desired_handle"],),
    }
    assert selection.selected_resource_symbol == "xarm6"
    assert selection.allocation_status == "accepted"


@pytest.mark.parametrize("resource_symbol", ["xarm6", "ur5e"])
def test_cartesian_request_uses_pa_chosen_resource_controller_profile(
    tmp_path: Path,
    resource_symbol: str,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, _ = _authorities(tmp_path)
    _grounded_abox(root, tbox)
    fixture = _cartesian_grounding_fixture(
        root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        desired_y=0.144,
    )

    prepared = prepare_cartesian_reachability(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        resource_symbol=resource_symbol,
        allocation_presentation=fixture["presentation"],
        current_state_evidence_handle=fixture["current_handle"],
        desired_state_evidence_handle=fixture["desired_handle"],
        current_location_record_path=fixture["current_location"],
        desired_location_record_path=fixture["desired_location"],
        current_cad_correspondence_record_path=fixture["current_correspondence"],
        desired_cad_correspondence_record_path=fixture["desired_correspondence"],
    )

    assert prepared.resource_symbol == resource_symbol
    assert prepared.resource_jid == f"{resource_symbol}@localhost"
    assert prepared.controller.moveit_group == f"{resource_symbol}_manipulator"
    assert prepared.controller.end_effector_link == f"{resource_symbol}_ee"
    assert prepared.controller.tcp_link == f"{resource_symbol}_tcp"
    assert prepared.controller.cartesian_path_service == "/compute_cartesian_path"


def test_cartesian_reachability_fails_closed_without_support_plane(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, _ = _authorities(tmp_path)
    _grounded_abox(root, tbox)
    fixture = _cartesian_grounding_fixture(
        root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        desired_y=0.144,
        support_status="unavailable",
    )

    with pytest.raises(ResourceGroundingError, match="support-plane evidence"):
        prepare_cartesian_reachability(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            resource_symbol="xarm6",
            allocation_presentation=fixture["presentation"],
            current_state_evidence_handle=fixture["current_handle"],
            desired_state_evidence_handle=fixture["desired_handle"],
            current_location_record_path=fixture["current_location"],
            desired_location_record_path=fixture["desired_location"],
            current_cad_correspondence_record_path=fixture["current_correspondence"],
            desired_cad_correspondence_record_path=fixture["desired_correspondence"],
        )


def test_cartesian_reachability_rejects_tampered_robot_frame_location(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, _ = _authorities(tmp_path)
    _grounded_abox(root, tbox)
    fixture = _cartesian_grounding_fixture(
        root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        desired_y=0.144,
    )
    current_location = fixture["current_location"]
    assert isinstance(current_location, Path)
    record = json.loads(current_location.read_text(encoding="utf-8"))
    record["translated_location_m"][0] += 0.1
    current_location.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ResourceGroundingError, match="calibrated candidate"):
        prepare_cartesian_reachability(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            resource_symbol="xarm6",
            allocation_presentation=fixture["presentation"],
            current_state_evidence_handle=fixture["current_handle"],
            desired_state_evidence_handle=fixture["desired_handle"],
            current_location_record_path=current_location,
            desired_location_record_path=fixture["desired_location"],
            current_cad_correspondence_record_path=fixture["current_correspondence"],
            desired_cad_correspondence_record_path=fixture["desired_correspondence"],
        )


def test_rejected_cartesian_plan_cannot_commit_or_substitute_resource(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, _ = _authorities(tmp_path)
    _grounded_abox(root, tbox)
    fixture = _cartesian_grounding_fixture(
        root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        desired_y=0.144,
    )
    prepared = prepare_cartesian_reachability(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        resource_symbol="xarm6",
        allocation_presentation=fixture["presentation"],
        current_state_evidence_handle=fixture["current_handle"],
        desired_state_evidence_handle=fixture["desired_handle"],
        current_location_record_path=fixture["current_location"],
        desired_location_record_path=fixture["desired_location"],
        current_cad_correspondence_record_path=fixture["current_correspondence"],
        desired_cad_correspondence_record_path=fixture["desired_correspondence"],
    )
    validation = asyncio.run(
        validate_provisional_allocation(
            _CapturingCartesianRuntime(status="rejected"),
            interaction_root=root,
            workcell=workcell,
            reachability=prepared,
        )
    )
    reachability = persist_cartesian_reachability(
        interaction_root=root,
        prepared=prepared,
        robot_agent_validation_path=validation.record_path,
    )

    assert reachability.status == "rejected"
    assert validation.status == "rejected"
    with pytest.raises(ResourceGroundingError, match="accepted reachability"):
        persist_pa_resource_selection(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            reachability=reachability,
            allocation_presentation=fixture["presentation"],
            robot_agent_validation_path=validation.record_path,
        )
    assert not (root / "products/grounding/resource_selection").exists()


@pytest.mark.parametrize(
    ("resource_order", "resource_symbol", "current_y", "desired_y"),
    [
        (("xarm6", "ur5e"), "ur5e", 0.2, 0.8),
        (("ur5e", "xarm6"), "xarm6", -0.7, -0.2),
    ],
)
def test_resource_presentation_order_cannot_override_the_pa_choice(
    tmp_path: Path,
    resource_order: tuple[str, str],
    resource_symbol: str,
    current_y: float,
    desired_y: float,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, _ = _authorities(
        tmp_path,
        resource_order=resource_order,
    )
    _grounded_abox(root, tbox)
    current_path = _write_world_location(root, "current", (0.0, current_y, 1.1))
    desired_path = _write_world_location(root, "desired", (0.0, desired_y, 1.1))
    presentation, handles = _allocation_presentation_for_locations(
        root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        location_paths=(desired_path, current_path),
    )

    check = _runtime_check_resource_reachability(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        resource_symbol=resource_symbol,
        allocation_presentation=presentation,
        state_location_record_paths={
            "current_state": [
                (handles[_location_candidate_key(root, current_path)], current_path)
            ],
            "desired_state": [
                (handles[_location_candidate_key(root, desired_path)], desired_path)
            ],
        },
    )
    validation = asyncio.run(
        validate_provisional_allocation(
            _CapturingFeasibilityRuntime(),
            interaction_root=root,
            workcell=workcell,
            reachability=check,
        )
    )
    selection = persist_pa_resource_selection(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        reachability=check,
        allocation_presentation=presentation,
        robot_agent_validation_path=validation.record_path,
    )

    assert check.status == "accepted"
    assert check.schema_version == 4
    assert check.resource_symbol == resource_symbol
    assert selection.schema_version == 5
    assert selection.selected_resource_symbol == resource_symbol


def test_v4_selection_rejects_any_unreachable_submitted_location_without_substitution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, _ = _authorities(tmp_path)
    before = _grounded_abox(root, tbox)
    current_path = _write_world_location(root, "current", (0.0, -0.7, 1.1))
    desired_path = _write_world_location(root, "desired", (0.0, 0.8, 1.1))
    presentation, handles = _allocation_presentation_for_locations(
        root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        location_paths=(current_path, desired_path),
    )

    check = _runtime_check_resource_reachability(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        resource_symbol="xarm6",
        allocation_presentation=presentation,
        state_location_record_paths={
            "current_state": [
                (handles[_location_candidate_key(root, current_path)], current_path)
            ],
            "desired_state": [
                (handles[_location_candidate_key(root, desired_path)], desired_path)
            ],
        },
    )

    assert check.status == "rejected"
    assert check.state_locations is not None
    assert check.state_locations["current_state"][0].reachable is True
    assert check.state_locations["desired_state"][0].reachable is False
    with pytest.raises(ResourceGroundingError, match="accepted reachability"):
        persist_pa_resource_selection(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            reachability=check,
            allocation_presentation=presentation,
            robot_agent_validation_path=root / "missing_validation.json",
        )
    assert set(load_interaction_abox(root, tbox).graph) == set(before.graph)
    assert not (root / "products/grounding/resource_selection").exists()


def test_neutral_location_requires_an_intact_hash_chain(tmp_path: Path) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, _ = _authorities(tmp_path)
    _grounded_abox(root, tbox)
    current = _write_world_location(root, "current", (0.0, -0.7, 1.1))
    desired = _write_world_location(root, "desired", (0.0, -0.2, 1.1))
    record = json.loads(current.read_text(encoding="utf-8"))
    record.pop("source_evidence")
    current.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RobotFrameLocationEvidenceError, match="hash references"):
        check_resource_reachability(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            resource_symbol="xarm6",
            current_location_record_path=current,
            desired_location_record_path=desired,
        )


def test_rejected_robot_agent_validation_does_not_commit_or_substitute(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, _ = _authorities(tmp_path)
    before = _grounded_abox(root, tbox)
    check = _accepted_check(root, tbox, registry, workcell, "xarm6")
    validation_path = _write_validation(root, check, status="rejected")

    selection = persist_pa_resource_selection(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        reachability=check,
        allocation_presentation=load_allocation_presentation(root),
        robot_agent_validation_path=validation_path,
    )

    assert selection.provisional_resource_symbol == "xarm6"
    assert selection.robot_agent_validation_status == "rejected"
    assert selection.selected_resource_symbol is None
    assert selection.selected_resource_iri is None
    with pytest.raises(ResourceGroundingError, match="accepted PA choice"):
        commit_resource_assignment(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            selection=selection,
        )
    assert set(load_interaction_abox(root, tbox).graph) == set(before.graph)


def test_accepted_validation_commits_exactly_one_assignment_without_motion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, _ = _authorities(tmp_path)
    before = _grounded_abox(root, tbox)
    check = _accepted_check(root, tbox, registry, workcell, "ur5e")
    validation_path = _write_validation(root, check, status="accepted")
    selection = persist_pa_resource_selection(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        reachability=check,
        allocation_presentation=load_allocation_presentation(root),
        robot_agent_validation_path=validation_path,
    )

    result = commit_resource_assignment(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        selection=selection,
    )

    ppr = Namespace(PPR_NAMESPACE)
    execution = URIRef(f"{before.namespace}process_execution_0001")
    assert set(result.abox.graph) - set(before.graph) == {
        (URIRef(before.specification_iri), ppr.hasProcessExecution, execution),
        (execution, RDF.type, ppr.processExecution),
        (execution, ppr.runsProcess, URIRef(PROCESS_IRI)),
        (
            execution,
            ppr.runsOnResource,
            URIRef(f"{RESOURCE_NAMESPACE}ur5e"),
        ),
    }
    selection_record = selection.to_record()
    assert selection_record["authority"] == "ProductAgent"
    assert selection_record["current_state_iri"].endswith("currentstate_0001")
    assert selection_record["desired_state_iri"].endswith("desiredstate_0001")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["mode"] == "plan_only"
    assert validation["motion_executed"] is False
    assert result.assertion_count == 4


def test_selection_requires_accepted_reachability_and_unchanged_records(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, _ = _authorities(tmp_path)
    _grounded_abox(root, tbox)
    current_path = _write_world_location(root, "current", (0.0, -0.7, 1.1))
    desired_path = _write_world_location(root, "desired", (0.0, 0.8, 1.1))
    check = check_resource_reachability(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        resource_symbol="xarm6",
        current_location_record_path=current_path,
        desired_location_record_path=desired_path,
    )
    validation_path = _write_validation(root, check, status="rejected")

    with pytest.raises(ResourceGroundingError, match="accepted reachability"):
        persist_pa_resource_selection(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            reachability=check,
            allocation_presentation=load_allocation_presentation(root),
            robot_agent_validation_path=validation_path,
        )

    presentation, handles = _allocation_presentation_for_locations(
        root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        location_paths=(current_path, desired_path),
    )
    desired_handle = handles[_location_candidate_key(root, desired_path)]
    accepted = _runtime_check_resource_reachability(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        resource_symbol="ur5e",
        allocation_presentation=presentation,
        current_state_evidence_handle=desired_handle,
        desired_state_evidence_handle=desired_handle,
        current_location_record_path=desired_path,
        desired_location_record_path=desired_path,
        check_number=2,
    )
    accepted.record_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ResourceGroundingError, match="changed"):
        accepted.assert_unchanged()


def test_forged_accepted_selection_cannot_change_the_provisional_resource(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, _ = _authorities(tmp_path)
    _grounded_abox(root, tbox)
    check = _accepted_check(root, tbox, registry, workcell, "xarm6")
    selection = persist_pa_resource_selection(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        reachability=check,
        allocation_presentation=load_allocation_presentation(root),
        robot_agent_validation_path=_write_validation(root, check, status="accepted"),
    )
    forged = replace(
        selection,
        selected_resource_symbol="ur5e",
        selected_resource_iri=f"{RESOURCE_NAMESPACE}ur5e",
        selected_resource_jid="ur5e@localhost",
    )

    with pytest.raises(ResourceGroundingError, match="changed"):
        commit_resource_assignment(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            selection=forged,
        )


def check_resource_reachability(  # noqa: PLR0913
    *,
    interaction_root: Path,
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
    resource_symbol: str,
    current_location_record_path: Path,
    desired_location_record_path: Path,
    check_number: int = 1,
) -> ReachabilityCheckRecord:
    """Exercise the v2 checker with a pinned neutral presentation fixture."""
    presentation, handles = _allocation_presentation_for_locations(
        interaction_root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        location_paths=(current_location_record_path, desired_location_record_path),
    )
    current_key = _location_candidate_key(interaction_root, current_location_record_path)
    desired_key = _location_candidate_key(interaction_root, desired_location_record_path)
    return _runtime_check_resource_reachability(
        interaction_root=interaction_root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        resource_symbol=resource_symbol,
        allocation_presentation=presentation,
        current_state_evidence_handle=handles[current_key],
        desired_state_evidence_handle=handles[desired_key],
        current_location_record_path=current_location_record_path,
        desired_location_record_path=desired_location_record_path,
        check_number=check_number,
    )


def _cartesian_grounding_fixture(  # noqa: PLR0913
    root: Path,
    *,
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
    desired_y: float,
    support_status: str = "detected",
) -> dict[str, object]:
    fixture_root = root / "products/grounding/cartesian_fixture"
    fixture_root.mkdir(parents=True, exist_ok=True)

    def write(relative: str, value: Mapping[str, object]) -> Path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    current_cad = write(
        "products/grounding/cartesian_fixture/current_cad.json",
        {
            "schema_version": 1,
            "record_type": "CADMeshRecord",
            "stored_units": "m",
            "bounds_m": {"size": [0.042, 0.042, 0.02]},
        },
    )
    desired_cad = write(
        "products/grounding/cartesian_fixture/desired_cad.json",
        {
            "schema_version": 1,
            "record_type": "CADMeshRecord",
            "stored_units": "m",
            "bounds_m": {"size": [0.01, 0.01, 0.02]},
        },
    )
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    current_calibration = write(
        "products/grounding/cartesian_fixture/current_calibration.json",
        {
            "schema_version": 1,
            "record_type": "CameraToRobotCalibrationRecord",
            "source_frame": "neutral_current_frame",
            "target_frame": "world",
            "target_from_camera_transform": identity,
        },
    )
    desired_calibration = write(
        "products/grounding/cartesian_fixture/desired_calibration.json",
        {
            "schema_version": 1,
            "record_type": "CameraToRobotCalibrationRecord",
            "source_frame": "neutral_desired_frame",
            "target_frame": "world",
            "target_from_camera_transform": identity,
        },
    )
    support_plane = {
        "status": support_status,
        "candidate_filtering_applied": support_status == "detected",
        "normal": [0.0, 0.0, 1.0],
        "offset_m": -1.04,
        "rms_distance_m": 0.001,
    }
    segmentation = write(
        "products/grounding/cartesian_fixture/segmentation.json",
        {
            "schema_version": 2,
            "record_type": "RGBDSegmentationRecord",
            "cameras": [
                {
                    "observation_handle": "view_current",
                    "frame": "neutral_current_frame",
                    "support_plane": support_plane,
                    "candidates": [
                        {
                            "candidate_handle": "candidate_current",
                            "centroid_m": [0.4, -0.3, 1.06],
                        }
                    ],
                },
                {
                    "observation_handle": "view_desired",
                    "frame": "neutral_desired_frame",
                    "support_plane": support_plane,
                    "candidates": [
                        {
                            "candidate_handle": "candidate_desired",
                            "centroid_m": [0.0, desired_y, 1.06],
                        }
                    ],
                },
            ],
        },
    )
    segmentation_ref = segmentation.relative_to(root).as_posix()
    segmentation_sha256 = hashlib.sha256(segmentation.read_bytes()).hexdigest()

    def location(
        name: str,
        *,
        calibration: Path,
        observation_handle: str,
        candidate_handle: str,
        source_frame: str,
        translation: list[float],
    ) -> Path:
        return write(
            f"products/grounding/cartesian_fixture/{name}_location.json",
            {
                "schema_version": 2,
                "record_type": "RobotFrameLocationRecord",
                "method": "calibrated_neutral_candidate_center",
                "source_segmentation": {
                    "ref": segmentation_ref,
                    "sha256": segmentation_sha256,
                },
                "source_calibration": {
                    "ref": calibration.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
                },
                "candidate_reference": {
                    "observation_handle": observation_handle,
                    "candidate_handle": candidate_handle,
                },
                "observation_timestamp_ns": 1,
                "source_frame": source_frame,
                "target_frame": "world",
                "translated_location_m": translation,
                "location": "available",
                "robot_frame_conversion": "accepted",
            },
        )

    current_location = location(
        "current",
        calibration=current_calibration,
        observation_handle="view_current",
        candidate_handle="candidate_current",
        source_frame="neutral_current_frame",
        translation=[0.4, -0.3, 1.06],
    )
    desired_location = location(
        "desired",
        calibration=desired_calibration,
        observation_handle="view_desired",
        candidate_handle="candidate_desired",
        source_frame="neutral_desired_frame",
        translation=[0.0, desired_y, 1.06],
    )

    def correspondence(
        name: str,
        *,
        cad_path: Path,
        observation_handle: str,
        candidate_handle: str,
        status: str,
    ) -> Path:
        candidate = {
            "observation_handle": observation_handle,
            "candidate_handle": candidate_handle,
            "within_size_tolerance": status in {"accepted", "ambiguous"},
        }
        return write(
            f"products/grounding/cartesian_fixture/{name}_correspondence.json",
            {
                "schema_version": 3,
                "record_type": "CADSizeCorrespondenceRecord",
                "CAD": {
                    "record": {
                        "ref": cad_path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(cad_path.read_bytes()).hexdigest(),
                    }
                },
                "segmentation": {
                    "record": {
                        "ref": segmentation_ref,
                        "sha256": segmentation_sha256,
                    }
                },
                "candidate_measurements": [candidate],
                "measurement": "accepted",
                "CAD_correspondence": "not_evaluated",
            },
        )

    current_correspondence = correspondence(
        "current",
        cad_path=current_cad,
        observation_handle="view_current",
        candidate_handle="candidate_current",
        status="accepted",
    )
    desired_correspondence = correspondence(
        "desired",
        cad_path=desired_cad,
        observation_handle="view_desired",
        candidate_handle="candidate_desired",
        status="ambiguous",
    )

    evidence_presentation = load_or_create_evidence_presentation(
        root,
        sources=((None, "observation", "fresh_on_call"),),
        explicit_order=("__live_observation__",),
    )
    evidence_sources = (
        AllocationEvidenceSource(
            canonical_key=f"{segmentation_ref}#/cameras/0/candidates/0",
            record_type="RGBDSegmentationRecord",
            record_ref=segmentation_ref,
            record_sha256=segmentation_sha256,
            field_path="/cameras/0/candidates/0",
            observation_handle="view_current",
            candidate_handle="candidate_current",
            source_frame="neutral_current_frame",
            neutral_projection={"candidate_available": True},
        ),
        AllocationEvidenceSource(
            canonical_key=f"{segmentation_ref}#/cameras/1/candidates/0",
            record_type="RGBDSegmentationRecord",
            record_ref=segmentation_ref,
            record_sha256=segmentation_sha256,
            field_path="/cameras/1/candidates/0",
            observation_handle="view_desired",
            candidate_handle="candidate_desired",
            source_frame="neutral_desired_frame",
            neutral_projection={"candidate_available": True},
        ),
    )
    abox = load_interaction_abox(root, tbox)
    catalog = candidate_resource_catalog(abox, registry, workcell)
    presentation = load_or_create_allocation_presentation(
        root,
        evidence_presentation=evidence_presentation,
        process_symbol="assembly",
        process_iri=PROCESS_IRI,
        feature_iri=f"{abox.namespace}feature_0001",
        current_state_iri=f"{abox.namespace}currentstate_0001",
        desired_state_iri=f"{abox.namespace}desiredstate_0001",
        resources=tuple(
            (symbol, str(entry["resource_iri"]), str(entry["resource_jid"]))
            for symbol, entry in catalog.items()
        ),
        evidence_sources=evidence_sources,
        explicit_resource_order=tuple(catalog),
        explicit_candidate_order=tuple(source.canonical_key for source in evidence_sources),
    )
    entries = {entry.canonical_key: entry.pa_handle for entry in presentation.evidence_entries}
    return {
        "presentation": presentation,
        "current_handle": entries[evidence_sources[0].canonical_key],
        "desired_handle": entries[evidence_sources[1].canonical_key],
        "current_location": current_location,
        "desired_location": desired_location,
        "current_correspondence": current_correspondence,
        "desired_correspondence": desired_correspondence,
    }


def _allocation_presentation_for_locations(  # noqa: PLR0913
    root: Path,
    *,
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
    location_paths: tuple[Path, ...],
):
    interaction_root = Path(root).resolve()
    abox = load_interaction_abox(interaction_root, tbox)
    evidence_presentation = load_or_create_evidence_presentation(
        interaction_root,
        sources=((None, "observation", "fresh_on_call"),),
        explicit_order=("__live_observation__",),
    )
    sources: list[AllocationEvidenceSource] = []
    for path in dict.fromkeys(location_paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        record_ref = path.resolve().relative_to(interaction_root).as_posix()
        source_frame = str(record.get("source_frame") or "neutral_frame")
        candidate_reference = record.get("candidate_reference")
        observation_handle = (
            candidate_reference.get("observation_handle")
            if isinstance(candidate_reference, Mapping)
            else None
        )
        candidate_handle = (
            candidate_reference.get("candidate_handle")
            if isinstance(candidate_reference, Mapping)
            else None
        )
        sources.append(
            AllocationEvidenceSource(
                canonical_key=f"{record_ref}#/translated_location_m",
                record_type="RobotFrameLocationRecord",
                record_ref=record_ref,
                record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                field_path="/translated_location_m",
                observation_handle=(
                    str(observation_handle) if observation_handle is not None else None
                ),
                candidate_handle=(str(candidate_handle) if candidate_handle is not None else None),
                source_frame=source_frame,
                neutral_projection={"location_record_available": True},
            )
        )
    catalog = candidate_resource_catalog(abox, registry, workcell)
    process_symbol, process_iri = workcell.processes[0]
    presentation = load_or_create_allocation_presentation(
        interaction_root,
        evidence_presentation=evidence_presentation,
        process_symbol=process_symbol,
        process_iri=process_iri,
        feature_iri=f"{abox.namespace}feature_0001",
        current_state_iri=f"{abox.namespace}currentstate_0001",
        desired_state_iri=f"{abox.namespace}desiredstate_0001",
        resources=tuple(
            (symbol, str(entry["resource_iri"]), str(entry["resource_jid"]))
            for symbol, entry in catalog.items()
        ),
        evidence_sources=tuple(sources),
        explicit_resource_order=tuple(catalog),
        explicit_candidate_order=tuple(source.canonical_key for source in sources),
    )
    handles = {entry.canonical_key: entry.pa_handle for entry in presentation.evidence_entries}
    return presentation, handles


def _location_candidate_key(root: Path, path: Path) -> str:
    record_ref = path.resolve().relative_to(Path(root).resolve()).as_posix()
    return f"{record_ref}#/translated_location_m"


def _accepted_check(
    root: Path,
    tbox: TBoxSnapshot,
    registry: ResourceRegistrySnapshot,
    workcell: PredefinedWorkcellSnapshot,
    resource_symbol: str,
    *,
    check_number: int = 1,
    current_name: str = "current",
    desired_name: str = "desired",
):
    locations = {
        "xarm6": ((0.0, -0.7, 1.1), (0.0, -0.2, 1.1)),
        "ur5e": ((0.0, 0.2, 1.1), (0.0, 0.8, 1.1)),
    }
    current, desired = locations[resource_symbol]
    return check_resource_reachability(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        resource_symbol=resource_symbol,
        current_location_record_path=_write_world_location(root, current_name, current),
        desired_location_record_path=_write_world_location(root, desired_name, desired),
        check_number=check_number,
    )


def _authorities(
    root: Path,
    *,
    resource_order: tuple[str, str] = ("xarm6", "ur5e"),
) -> tuple[
    TBoxSnapshot,
    ResourceRegistrySnapshot,
    PredefinedWorkcellSnapshot,
    dict[str, Path],
]:
    manifest_root = root / "manifests"
    manifest_root.mkdir(parents=True)
    manifest_paths = {
        "xarm6": manifest_root / "robot_xarm6.json",
        "ur5e": manifest_root / "robot_ur5e.json",
    }
    _write_manifest(
        manifest_paths["xarm6"],
        "xarm6",
        origin_y=-0.5,
        workspace_y=(-1.0, 0.1),
    )
    _write_manifest(
        manifest_paths["ur5e"],
        "ur5e",
        origin_y=0.5,
        workspace_y=(-0.35, 1.1),
    )
    profile_path = root / "workcell_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "processes": [{"symbol": "assembly", "iri": PROCESS_IRI}],
                "resources": [
                    {
                        "symbol": symbol,
                        "iri": f"{RESOURCE_NAMESPACE}{symbol}",
                        "manifest_ref": manifest_paths[symbol].relative_to(root).as_posix(),
                        "capable_process_iris": [PROCESS_IRI],
                    }
                    for symbol in resource_order
                ],
            }
        ),
        encoding="utf-8",
    )
    profile = load_workcell_profile(profile_path, repository_root=root)
    tbox = load_ppr_tbox(TBOX_PATH, ppr_namespace=PPR_NAMESPACE)
    registry = load_predefined_resource_registry(tbox, profile=profile)
    return (
        tbox,
        registry,
        load_predefined_workcell(tbox, registry, profile=profile),
        manifest_paths,
    )


def _arbitrary_authorities(
    root: Path,
) -> tuple[TBoxSnapshot, ResourceRegistrySnapshot, PredefinedWorkcellSnapshot]:
    process_iris = {
        "joining": "https://example.local/process/joining",
        "inspection": "https://example.local/process/inspection",
    }
    resource_profiles = (
        ("alpha_bot", -0.4, (-0.9, 0.2), (process_iris["joining"],)),
        (
            "beta_bot",
            0.3,
            (-0.2, 0.9),
            (process_iris["joining"], process_iris["inspection"]),
        ),
        ("gamma_bot", 0.8, (0.3, 1.3), (process_iris["inspection"],)),
    )
    manifest_root = root / "arbitrary_manifests"
    manifest_root.mkdir(parents=True)
    resources: list[dict[str, object]] = []
    for symbol, origin_y, workspace_y, capabilities in resource_profiles:
        manifest_path = manifest_root / f"{symbol}.json"
        _write_manifest(
            manifest_path,
            symbol,
            origin_y=origin_y,
            workspace_y=workspace_y,
        )
        resources.append(
            {
                "symbol": symbol,
                "iri": f"https://example.local/resource/{symbol}",
                "manifest_ref": manifest_path.relative_to(root).as_posix(),
                "capable_process_iris": list(capabilities),
            }
        )
    profile_path = root / "arbitrary_workcell_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "processes": [
                    {"symbol": symbol, "iri": process_iri}
                    for symbol, process_iri in process_iris.items()
                ],
                "resources": resources,
            }
        ),
        encoding="utf-8",
    )
    profile = load_workcell_profile(profile_path, repository_root=root)
    tbox = load_ppr_tbox(TBOX_PATH, ppr_namespace=PPR_NAMESPACE)
    registry = load_predefined_resource_registry(tbox, profile=profile)
    return tbox, registry, load_predefined_workcell(tbox, registry, profile=profile)


def _write_manifest(
    path: Path,
    symbol: str,
    *,
    origin_y: float,
    workspace_y: tuple[float, float],
) -> None:
    path.write_text(
        json.dumps(
            {
                symbol: {
                    "type": "robot",
                    "jid": f"{symbol}@localhost",
                    "password": "ignored",
                    "execution_mode": "simulation",
                    "gazebo": {
                        "static_capabilities": {
                            "workspace_bounds": {
                                "x_min_m": -0.7,
                                "x_max_m": 0.7,
                                "y_min_m": workspace_y[0],
                                "y_max_m": workspace_y[1],
                                "z_min_m": 0.9,
                                "z_max_m": 1.5,
                            },
                            "gripper_reach": {
                                "frame": "world",
                                "origin_pose": {
                                    "x": 0.0,
                                    "y": origin_y,
                                    "z": 1.021,
                                },
                                "max_xy_radius_m": 0.7,
                                "z_min_m": 0.9,
                                "z_max_m": 1.5,
                                "tolerance_m": 0.01,
                            },
                        },
                        "controller": {
                            "move_group": {
                                "group_name": f"{symbol}_manipulator",
                                "ee_link": f"{symbol}_ee",
                                "tcp_link": f"{symbol}_tcp",
                                "frame_id": "world",
                            },
                            "services": {
                                "cartesian_path": "/compute_cartesian_path",
                            },
                            "motion": {
                                "approach_height_m": 0.2,
                                "recovery_observed_pick_approach_height_m": 0.06,
                                "recovery_observed_pick_surface_clearance_m": 0.005,
                                "pick_tcp_z_bias_min_m": 0.003,
                                "pick_tcp_z_bias_max_m": 0.02,
                            },
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def _grounded_abox(
    root: Path,
    tbox: TBoxSnapshot,
    *,
    product_requirement: str = "assemble Medium Gear",
    process_iri: str = PROCESS_IRI,
) -> ABoxSnapshot:
    abox = initialize_interaction_abox(root, product_requirement, tbox)
    ppr = Namespace(PPR_NAMESPACE)
    feature_iri = f"{abox.namespace}feature_0001"
    current_state_iri = f"{abox.namespace}currentstate_0001"
    desired_state_iri = f"{abox.namespace}desiredstate_0001"
    result = validate_and_merge_triple_delta(
        root,
        tbox,
        "ontology_grounding_host",
        {
            "assertions": [
                _assertion(feature_iri, str(RDF.type), str(ppr.feature)),
                _assertion(abox.specification_iri, str(ppr.defines), feature_iri),
                _assertion(process_iri, str(ppr.realizes), feature_iri),
                _assertion(current_state_iri, str(RDF.type), str(ppr.state)),
                _assertion(desired_state_iri, str(RDF.type), str(ppr.state)),
                _assertion(
                    feature_iri,
                    str(ppr.hascurrentstate),
                    current_state_iri,
                ),
                _assertion(
                    feature_iri,
                    str(ppr.hasdesiredstate),
                    desired_state_iri,
                ),
            ]
        },
        authorized_evidence_refs={EVIDENCE_REF},
        authorized_external_process_iris={process_iri},
    )
    return result.abox


def _assertion(subject: str, predicate: str, object_iri: str) -> dict[str, object]:
    return {
        "subject": subject,
        "predicate": predicate,
        "object": {"kind": "iri", "value": object_iri},
        "evidence_refs": [EVIDENCE_REF],
    }


def _write_world_location(
    root: Path,
    name: str,
    translation: tuple[float, float, float],
) -> Path:
    destination = root / "products/grounding/synthetic_world_location" / name
    destination.mkdir(parents=True, exist_ok=True)
    source_path = destination / "source_evidence.json"
    source_path.write_text('{"source":"synthetic"}', encoding="utf-8")
    source_ref = source_path.relative_to(root).as_posix()
    record = {
        "schema_version": 2,
        "record_type": "RobotFrameLocationRecord",
        "producer": "synthetic_world_location_provider",
        "method": "neutral_fixture",
        "robot_frame_conversion": "accepted",
        "location": "available",
        "target_frame": "world",
        "observation_timestamp_ns": 11,
        "translated_location_m": list(translation),
        "candidate_reference": {
            "observation_handle": f"observation_{name}",
            "candidate_handle": f"candidate_{name}",
        },
        "source_evidence": {
            "ref": source_ref,
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
    }
    path = destination / "robot_frame_location_record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _write_validation(
    root: Path,
    reachability: ReachabilityCheckRecord,
    *,
    status: str,
) -> Path:
    resource_symbol = reachability.resource_symbol
    resource_iri = reachability.resource_iri
    resource_jid = reachability.resource_jid
    execution_mode = reachability.execution_mode
    target_frame = reachability.target_frame
    request_fingerprint = reachability.fingerprint
    endpoint_status = "accepted" if status == "accepted" else "rejected"
    endpoint = {
        "status": endpoint_status,
        "message": f"fixture {endpoint_status}",
        "error_code": 1 if endpoint_status == "accepted" else -1,
    }
    payload: dict[str, object] = {
        "schema_version": 2,
        "record_type": "PlanOnlyFeasibilityValidationRecord",
        "validation_number": 1,
        "validator_authority": resource_jid,
        "process_symbol": reachability.process_symbol,
        "process_iri": reachability.process_iri,
        "feature_iri": reachability.feature_iri,
        "current_state_iri": reachability.current_state.state_iri,
        "desired_state_iri": reachability.desired_state.state_iri,
        "resource_symbol": resource_symbol,
        "resource_iri": resource_iri,
        "resource_jid": resource_jid,
        "execution_mode": execution_mode,
        "moveit_group": f"{resource_symbol}_manipulator",
        "end_effector_link": f"{resource_symbol}_tcp",
        "target_frame": target_frame,
        "validation_scope": "endpoint_motion",
        "checked_constraints": [
            "positional_ik",
            "collision_aware_endpoints",
            "path_between_endpoints",
        ],
        "unvalidated_constraints": [
            "grasping",
            "end_effector_orientation",
            "attached_object_geometry",
            f"{reachability.process_symbol}_tolerance",
            "force_contact",
            "insertion_constraints",
        ],
        "current_state": dict(endpoint),
        "desired_state": dict(endpoint),
        "mode": "plan_only",
        "motion_executed": False,
        "status": status,
        "feedback": None if status == "accepted" else "fixture rejection",
        "validated_at_ns": 12,
        "request_fingerprint": request_fingerprint,
    }
    payload["fingerprint"] = _fingerprint(payload)
    path = (
        root
        / "resources"
        / resource_jid
        / "validation"
        / "plan_only_validation_0001"
        / "plan_only_feasibility_validation_record.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

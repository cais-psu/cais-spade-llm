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
    ReachabilityCheckRecord,
    ResourceGroundingError,
    RobotFrameLocationEvidenceError,
    candidate_resource_catalog,
    commit_resource_assignment,
    persist_pa_resource_selection,
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
        endpoint = {
            "status": "accepted",
            "message": "arbitrary-profile endpoint accepted",
            "error_code": 1,
        }
        return {
            "status": "accepted",
            "current_state": dict(endpoint),
            "desired_state": dict(endpoint),
            "feedback": None,
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
    desired_path = _write_world_location(root, "neutral_b", (0.0, 0.5, 1.1))
    presentation, handles = _allocation_presentation_for_locations(
        root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        location_paths=(current_path, desired_path),
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
        current_state_evidence_handle=handles[
            _location_candidate_key(root, current_path)
        ],
        desired_state_evidence_handle=handles[
            _location_candidate_key(root, desired_path)
        ],
        current_location_record_path=current_path,
        desired_location_record_path=desired_path,
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
    assert request["current_state"]["evidence_handle"]
    assert request["desired_state"]["evidence_handle"]
    serialized_request = json.dumps(request, sort_keys=True)
    assert "assembly" not in serialized_request
    assert "xarm6" not in serialized_request
    assert "ur5e" not in serialized_request
    assert selection.process_symbol == "joining"
    assert selection.process_iri == process_iri
    assert selection.candidate_resource_symbols == ("alpha_bot", "beta_bot")
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


def test_reversing_registry_order_cannot_override_the_pa_choice(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, _ = _authorities(
        tmp_path,
        resource_order=("ur5e", "xarm6"),
    )
    _grounded_abox(root, tbox)

    check = check_resource_reachability(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        resource_symbol="ur5e",
        current_location_record_path=_write_world_location(root, "current", (0.0, 0.2, 1.1)),
        desired_location_record_path=_write_world_location(root, "desired", (0.0, 0.8, 1.1)),
    )

    assert check.status == "accepted"
    assert check.resource_symbol == "ur5e"
    assert check.resource_jid == "ur5e@localhost"


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
                candidate_handle=(
                    str(candidate_handle) if candidate_handle is not None else None
                ),
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
    handles = {
        entry.canonical_key: entry.pa_handle for entry in presentation.evidence_entries
    }
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
                                "tcp_link": f"{symbol}_tcp",
                                "frame_id": "world",
                            }
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

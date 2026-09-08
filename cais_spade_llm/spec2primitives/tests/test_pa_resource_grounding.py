from __future__ import annotations

"""Tests for PA-driven two-state resource grounding."""


import asyncio
import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdflib import Namespace
from rdflib.namespace import RDF

from cais_spade_llm.spec2primitives.agents.pa import resource_grounding
from cais_spade_llm.spec2primitives.agents.pa.presentation_records import (
    AllocationEvidenceSource,
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
    ResourceGroundingError,
    candidate_resource_catalog,
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


def _proximity_check(symbol="xarm6", status="accepted"):
    record = {
        "resource_symbol": symbol,
        "target_frame": "world",
        "state_locations": {
            "current_state": [
                {"evidence_handle": "source_1", "translation_m": [3.0, 0.0, 0.0]},
                {"evidence_handle": "source_2", "translation_m": [0.0, 4.0, 0.0]},
            ],
            "desired_state": [
                {"evidence_handle": "destination", "translation_m": [0.0, 0.0, 5.0]}
            ],
        },
    }
    return SimpleNamespace(
        resource_symbol=symbol, target_frame="world", status=status, to_record=lambda: record
    )


class _BasePoseRuntime:
    def __init__(self, translation=(0.0, 0.0, 0.0), **changes):
        self.translation = translation
        self.changes = changes

    async def read_resource_base_pose(self, *, base_frame, target_frame):
        return {
            "base_frame": base_frame,
            "target_frame": target_frame,
            "translation_m": list(self.translation),
            "observed_at_ns": 123,
            **self.changes,
        }


@pytest.mark.parametrize("verdict", ["accepted", "rejected", "needs_context"])
def test_proximity_measures_all_locations_without_changing_eligibility(verdict):
    from cais_spade_llm.spec2primitives.agents.pa.resource_proximity import (
        read_resource_proximity,
        validate_resource_proximity,
    )

    check = _proximity_check(status=verdict)
    result = asyncio.run(read_resource_proximity(_BasePoseRuntime(), check))

    assert result["mean_current_state_distance_m"] == 3.5
    assert [item["distance_m"] for item in result["state_locations"]["current_state"]] == [3, 4]
    assert result["state_locations"]["desired_state"][0]["distance_m"] == 5
    assert check.status == verdict
    assert "selected_resource_symbol" not in result
    validate_resource_proximity(result, check.to_record())


def test_proximity_follows_swapped_robot_positions():
    from cais_spade_llm.spec2primitives.agents.pa.resource_proximity import read_resource_proximity

    near = _BasePoseRuntime((0.0, 0.0, 0.0))
    far = _BasePoseRuntime((20.0, 0.0, 0.0))
    for first, second in ((near, far), (far, near)):
        xarm = asyncio.run(read_resource_proximity(first, _proximity_check("xarm6")))
        ur5e = asyncio.run(read_resource_proximity(second, _proximity_check("ur5e")))
        assert (xarm["mean_current_state_distance_m"] < ur5e["mean_current_state_distance_m"]) is (
            first is near
        )


@pytest.mark.parametrize(
    "runtime",
    [
        object(),
        _BasePoseRuntime(target_frame="camera"),
        _BasePoseRuntime(base_frame="other_robot"),
        _BasePoseRuntime((float("nan"), 0, 0)),
        _BasePoseRuntime((True, 0, 0)),
        _BasePoseRuntime(observed_at_ns=0),
    ],
)
def test_invalid_or_missing_base_pose_does_not_become_zero_distance(runtime):
    from cais_spade_llm.spec2primitives.agents.pa.resource_proximity import read_resource_proximity

    check = _proximity_check()
    result = asyncio.run(read_resource_proximity(runtime, check))

    assert result["status"] == "unavailable"
    assert set(result) == {"status", "feedback"}
    assert check.status == "accepted"


@pytest.mark.parametrize("changed", ["distance", "mean", "handle", "frame"])
def test_proximity_validator_recomputes_pinned_measurements(changed):
    from cais_spade_llm.spec2primitives.agents.pa.resource_proximity import (
        read_resource_proximity,
        validate_resource_proximity,
    )

    check = _proximity_check()
    result = deepcopy(asyncio.run(read_resource_proximity(_BasePoseRuntime(), check)))
    if changed == "distance":
        result["state_locations"]["current_state"][0]["distance_m"] = 0.0
    elif changed == "mean":
        result["mean_current_state_distance_m"] = 0.0
    elif changed == "handle":
        result["state_locations"]["current_state"][0]["evidence_handle"] = "unbound"
    else:
        result["base_pose"]["target_frame"] = "camera"
    with pytest.raises(ValueError):
        validate_resource_proximity(result, check.to_record())


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
    from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import OfflineMoveItPlanning

    reachability = asyncio.run(
        resource_grounding.check_live_resource_reachability(
            runtime=OfflineMoveItPlanning(),
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
    )
    assert reachability.status == "accepted"
    assert reachability.resource_symbol == "beta_bot"
    assert len(reachability.state_locations["current_state"]) == 2
    assert len(reachability.state_locations["desired_state"]) == 2


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
            "record_type": "CADMeshRecord",
            "stored_units": "m",
            "bounds_m": {"size": [0.042, 0.042, 0.02]},
        },
    )
    desired_cad = write(
        "products/grounding/cartesian_fixture/desired_cad.json",
        {
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
            "record_type": "CameraToRobotCalibrationRecord",
            "source_frame": "neutral_current_frame",
            "target_frame": "world",
            "target_from_camera_transform": identity,
        },
    )
    desired_calibration = write(
        "products/grounding/cartesian_fixture/desired_calibration.json",
        {
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


@pytest.mark.parametrize(
    "verdict", ["accepted", "rejected", "needs_context", "missing_plan", "missing_location"]
)
def test_live_location_planning_uses_robot_result_outside_example_limits(tmp_path, verdict):
    from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
        check_live_resource_reachability,
    )
    from cais_spade_llm.spec2primitives.tests.pa_grounding_test_support import OfflineMoveItPlanning

    root = tmp_path / "interaction"
    tbox, registry, workcell, _ = _authorities(tmp_path)
    _grounded_abox(root, tbox)
    fixture = _cartesian_grounding_fixture(
        root, tbox=tbox, registry=registry, workcell=workcell, desired_y=0.144
    )

    class Planning(OfflineMoveItPlanning):
        async def validate_plan_only_allocation(self, request):
            assert request["moveit_group"] == "xarm6_manipulator"
            assert request["state_locations"]["desired_state"][0]["translation_m"][
                1
            ] == pytest.approx(0.144)
            assert request["motion_executed"] is False
            assert "workspace_bounds" not in json.dumps(request)
            assert "gripper_reach" not in json.dumps(request)
            response = await super().validate_plan_only_allocation(request)
            result = response["state_locations"]["desired_state"][0]
            if verdict in {"rejected", "needs_context"}:
                response["status"] = result["status"] = verdict
                result["error_code"] = -1 if verdict == "rejected" else None
                result["plan"] = None
            elif verdict == "missing_plan":
                result["plan"] = None
            elif verdict == "missing_location":
                response["state_locations"]["desired_state"] = []
            return response

    async def run():
        return await check_live_resource_reachability(
            runtime=Planning(),
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            resource_symbol="xarm6",
            allocation_presentation=fixture["presentation"],
            state_location_record_paths={
                "current_state": [(fixture["current_handle"], fixture["current_location"])],
                "desired_state": [(fixture["desired_handle"], fixture["desired_location"])],
            },
        )

    if verdict in {"missing_plan", "missing_location"}:
        with pytest.raises(ResourceGroundingError):
            asyncio.run(run())
        assert not (root / "products/grounding/reachability").exists()
    else:
        check = asyncio.run(run())
        assert not hasattr(check, "schema_version")
        assert check.status == verdict
        assert check.state_locations["desired_state"][0].reachable is (verdict == "accepted")
        assert "in_workspace" not in json.dumps(check.to_record())
        check.assert_unchanged()


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
                                "position_tolerance_m": 0.005,
                            },
                            "services": {
                                "cartesian_path": "/compute_cartesian_path",
                                "motion_plan": "/plan_kinematic_path",
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


def test_direct_allocation_entrypoint_rejects_historical_pa_proposal(tmp_path):
    from cais_spade_llm.spec2primitives.tests.test_pa_completion import (
        persist_native_completion_fixture,
    )

    completion = persist_native_completion_fixture(tmp_path).to_record()
    path = tmp_path / completion["ontology_projection_ref"]
    proposal = json.loads(path.read_text())
    proposal["schema_version"] = 12
    path.write_text(json.dumps(proposal))
    with pytest.raises(ResourceGroundingError, match="unchanged grounding evidence"):
        resource_grounding._validate_pa_grounding_evidence(tmp_path, completion["feature_iri"])


def test_direct_allocation_entrypoint_revalidates_snapshot_hash(tmp_path):
    from cais_spade_llm.spec2primitives.tests.test_pa_completion import (
        persist_native_completion_fixture,
    )

    completion = persist_native_completion_fixture(tmp_path).to_record()
    resource_grounding._validate_pa_grounding_evidence(tmp_path, completion["feature_iri"])
    proposal = json.loads((tmp_path / completion["ontology_projection_ref"]).read_text())
    snapshot_ref = next(
        item["ref"] for item in proposal["grounding_evidence"]["input_artifacts"]
        if item["ref"].startswith("products/grounding/product_context/")
    )
    snapshot_path = tmp_path / snapshot_ref
    snapshot_path.write_text(snapshot_path.read_text() + " ")
    with pytest.raises(ResourceGroundingError, match="unchanged grounding evidence"):
        resource_grounding._validate_pa_grounding_evidence(tmp_path, completion["feature_iri"])


def test_direct_allocation_rejects_unissued_state_value_citation_with_valid_native_hash(tmp_path):
    from cais_spade_llm.spec2primitives.tests.test_pa_completion import (
        persist_native_completion_fixture,
    )

    completion = persist_native_completion_fixture(tmp_path).to_record()
    path = tmp_path / completion["ontology_projection_ref"]
    proposal = json.loads(path.read_text())
    original_delta = json.dumps(proposal["compiled_delta"], sort_keys=True)
    unissued = tmp_path / "products/grounding/unissued_local.json"
    unissued.write_text('{"description":"Never issued to PA."}')
    ref = unissued.relative_to(tmp_path).as_posix()
    proposal["output"]["target_feature"]["current_state"]["state_values"][0][
        "evidence_refs"
    ].append(ref)
    proposal["grounding_evidence"]["source_refs"].append(
        {"ref": ref, "sha256": hashlib.sha256(unissued.read_bytes()).hexdigest()}
    )
    path.write_text(json.dumps(proposal))
    assert json.dumps(proposal["compiled_delta"], sort_keys=True) == original_delta
    with pytest.raises(ResourceGroundingError, match="unchanged grounding evidence"):
        resource_grounding._validate_pa_grounding_evidence(tmp_path, completion["feature_iri"])

"""Tests for ontology-driven predefined-resource grounding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from rdflib import Namespace, URIRef
from rdflib.namespace import RDF

from cais_spade_llm.spec2primitives.agents.pa.product_context import (
    ABoxSnapshot,
    initialize_interaction_abox,
    load_interaction_abox,
    validate_and_merge_triple_delta,
)
from cais_spade_llm.spec2primitives.agents.pa.resource_grounding import (
    ResourceGroundingError,
    RobotFrameLocationEvidenceError,
    commit_resource_assignment,
    derive_resource_assignment_need,
    select_predefined_resource,
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


def test_need_is_derived_only_after_the_exact_semantic_join(tmp_path: Path) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, _manifest_paths = _authorities(tmp_path)
    abox = initialize_interaction_abox(root, "assemble Medium Gear", tbox)

    assert derive_resource_assignment_need(abox, workcell) is None

    abox = _merge_semantic_grounding(root, tbox, abox)
    need = derive_resource_assignment_need(abox, workcell)

    assert need is not None
    assert need.specification_iri == abox.specification_iri
    assert need.feature_iri == f"{abox.namespace}feature_0001"
    assert need.process_iri == PROCESS_IRI
    assert need.candidate_resource_iris == (
        f"{RESOURCE_NAMESPACE}xarm6",
        f"{RESOURCE_NAMESPACE}ur5e",
    )
    assert need.required_record_type == "RobotFrameLocationRecord"
    assert need.target_frame == "world"


def test_need_uses_primary_join_and_ignores_defined_supporting_feature(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox, _registry, workcell, _manifest_paths = _authorities(tmp_path)
    abox = _merge_semantic_grounding(
        root,
        tbox,
        initialize_interaction_abox(root, "assemble supported features", tbox),
    )
    supporting_feature_iri = f"{abox.namespace}feature_0002"
    supporting = validate_and_merge_triple_delta(
        root,
        tbox,
        "ontology_grounding",
        {
            "assertions": [
                _assertion(
                    supporting_feature_iri,
                    str(RDF.type),
                    f"{PPR_NAMESPACE}feature",
                ),
                _assertion(
                    abox.specification_iri,
                    f"{PPR_NAMESPACE}defines",
                    supporting_feature_iri,
                ),
            ]
        },
        authorized_evidence_refs={EVIDENCE_REF},
    )

    need = derive_resource_assignment_need(supporting.abox, workcell)

    assert need is not None
    assert need.feature_iri == f"{abox.namespace}feature_0001"


@pytest.mark.parametrize(
    ("translation", "expected_symbol"),
    [
        ((0.0, 0.0, 1.1), "xarm6"),
        ((0.0, -0.8, 1.1), "xarm6"),
        ((0.0, 0.8, 1.1), "ur5e"),
        ((2.0, 0.0, 1.1), None),
    ],
)
def test_selection_uses_registry_order_and_only_coarse_manifest_geometry(
    tmp_path: Path,
    translation: tuple[float, float, float],
    expected_symbol: str | None,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, manifest_paths = _authorities(tmp_path)
    abox = _grounded_abox(root, tbox)
    need = derive_resource_assignment_need(abox, workcell)
    assert need is not None
    location_path = _write_world_location(root, translation)

    selection = select_predefined_resource(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        need=need,
        grounding_record_path=location_path,
    )

    assert selection.selected_resource_symbol == expected_symbol
    assert selection.selected_resource_iri == (
        None if expected_symbol is None else f"{RESOURCE_NAMESPACE}{expected_symbol}"
    )
    assert selection.selected_resource_jid == (
        None if expected_symbol is None else f"{expected_symbol}@localhost"
    )
    assert [item.resource_symbol for item in selection.candidate_reach_evidence] == [
        "xarm6",
        "ur5e",
    ]
    assert selection.record_path == (
        root
        / "products/grounding/resource_selection/selection_0001"
        / "resource_selection_record.json"
    )
    record_text = selection.record_path.read_text(encoding="utf-8")
    for forbidden in (
        "reachability",
        "parts_tuning",
        "password",
        "controller",
        "spawn",
    ):
        assert forbidden not in record_text
    selection.assert_unchanged()


def test_unrelated_manifest_details_cannot_change_the_reach_decision(
    tmp_path: Path,
) -> None:
    decisions: list[tuple[str | None, tuple[bool, ...]]] = []
    for index, hidden_value in enumerate(("first", "completely-different"), start=1):
        case_root = tmp_path / f"case_{index}"
        interaction_root = case_root / "interaction"
        tbox, registry, workcell, manifest_paths = _authorities(
            case_root,
            hidden_value=hidden_value,
        )
        need = derive_resource_assignment_need(
            _grounded_abox(interaction_root, tbox), workcell
        )
        assert need is not None
        selection = select_predefined_resource(
            interaction_root=interaction_root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            need=need,
            grounding_record_path=_write_world_location(
                interaction_root, (0.0, 0.0, 1.1)
            ),
        )
        decisions.append(
            (
                selection.selected_resource_symbol,
                tuple(item.reachable for item in selection.candidate_reach_evidence),
            )
        )

    assert decisions == [("xarm6", (True, True)), ("xarm6", (True, True))]


def test_default_selection_reads_the_registry_pinned_shared_manifests(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox = load_ppr_tbox(TBOX_PATH, ppr_namespace=PPR_NAMESPACE)
    registry = load_predefined_resource_registry(tbox)
    workcell = load_predefined_workcell(tbox, registry)
    need = derive_resource_assignment_need(_grounded_abox(root, tbox), workcell)
    assert need is not None

    selection = select_predefined_resource(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        need=need,
        grounding_record_path=_write_world_location(root, (0.0, 0.0, 1.1)),
    )

    assert selection.selected_resource_symbol == "xarm6"
    assert selection.selected_resource_jid == "xarm6@localhost"
    assert all(
        candidate.execution_mode == "simulation"
        for candidate in selection.candidate_reach_evidence
    )


@pytest.mark.parametrize(
    ("record_update", "message"),
    [
        ({"target_frame": "camera"}, "exact RobotFrameLocationRecord"),
        ({"robot_frame_conversion": "ambiguous"}, "accepted"),
        ({"robot_frame_conversion": "stale"}, "accepted"),
        ({"CAD_correspondence": "rejected"}, "CAD_correspondence"),
        ({"location": "ambiguous"}, "location"),
    ],
)
def test_selection_rejects_unusable_location_without_creating_a_record(
    tmp_path: Path,
    record_update: dict[str, object],
    message: str,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, manifest_paths = _authorities(tmp_path)
    need = derive_resource_assignment_need(_grounded_abox(root, tbox), workcell)
    assert need is not None
    location_path = _write_world_location(root, (0.0, 0.0, 1.1), **record_update)

    with pytest.raises(RobotFrameLocationEvidenceError, match=message):
        select_predefined_resource(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            need=need,
            grounding_record_path=location_path,
        )

    assert not (root / "products/grounding/resource_selection").exists()


@pytest.mark.parametrize(
    "translated_location_m",
    [
        None,
        [0.0, 0.0],
        [0.0, -0.5, "invalid"],
    ],
)
def test_location_record_rejects_missing_or_malformed_translation(
    tmp_path: Path,
    translated_location_m: object,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, manifest_paths = _authorities(tmp_path)
    need = derive_resource_assignment_need(_grounded_abox(root, tbox), workcell)
    assert need is not None
    location_path = _write_world_location(
        root,
        (0.0, -0.5, 1.1),
        translated_location_m=translated_location_m,
    )

    with pytest.raises(RobotFrameLocationEvidenceError):
        select_predefined_resource(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            need=need,
            grounding_record_path=location_path,
        )

    assert not (root / "products/grounding/resource_selection").exists()


def test_selection_requires_an_intact_location_source_hash_chain(tmp_path: Path) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, manifest_paths = _authorities(tmp_path)
    need = derive_resource_assignment_need(_grounded_abox(root, tbox), workcell)
    assert need is not None
    location_path = _write_world_location(root, (0.0, 0.0, 1.1))
    location_record = json.loads(location_path.read_text(encoding="utf-8"))
    source_ref = location_record["source_hashes"][0]["ref"]

    location_record.pop("source_hashes")
    location_path.write_text(json.dumps(location_record), encoding="utf-8")
    with pytest.raises(RobotFrameLocationEvidenceError, match="hash reference"):
        select_predefined_resource(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            need=need,
            grounding_record_path=location_path,
        )

    location_path = _write_world_location(root, (0.0, 0.0, 1.1))
    (root / source_ref).write_text("changed evidence", encoding="utf-8")
    with pytest.raises(RobotFrameLocationEvidenceError, match="hash does not match"):
        select_predefined_resource(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            need=need,
            grounding_record_path=location_path,
        )


def test_manifest_errors_are_not_location_evidence_errors(tmp_path: Path) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, manifest_paths = _authorities(tmp_path)
    need = derive_resource_assignment_need(_grounded_abox(root, tbox), workcell)
    assert need is not None
    manifest_paths["xarm6"].write_text("{}", encoding="utf-8")

    with pytest.raises(ResourceGroundingError) as raised:
        select_predefined_resource(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            need=need,
            grounding_record_path=_write_world_location(root, (0.0, 0.0, 1.1)),
        )

    assert not isinstance(raised.value, RobotFrameLocationEvidenceError)


def test_selection_detects_changed_location_and_manifest_sources(tmp_path: Path) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, manifest_paths = _authorities(tmp_path)
    need = derive_resource_assignment_need(_grounded_abox(root, tbox), workcell)
    assert need is not None
    location_path = _write_world_location(root, (0.0, 0.0, 1.1))
    selection = select_predefined_resource(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        need=need,
        grounding_record_path=location_path,
    )

    location_record = json.loads(location_path.read_text(encoding="utf-8"))
    location_record["observation_timestamp_ns"] = 12
    location_path.write_text(json.dumps(location_record), encoding="utf-8")
    with pytest.raises(ResourceGroundingError, match="location changed"):
        selection.assert_unchanged()

    location_record["observation_timestamp_ns"] = 11
    location_path.write_text(json.dumps(location_record), encoding="utf-8")
    manifest_paths["xarm6"].write_text("{}", encoding="utf-8")
    with pytest.raises(ResourceGroundingError, match="authority changed"):
        selection.assert_unchanged()


def test_system_commit_adds_only_the_derived_execution_assignment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, manifest_paths = _authorities(tmp_path)
    before = _grounded_abox(root, tbox)
    need = derive_resource_assignment_need(before, workcell)
    assert need is not None
    selection = select_predefined_resource(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        need=need,
        grounding_record_path=_write_world_location(root, (0.0, 0.0, 1.1)),
    )

    result = commit_resource_assignment(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        need=need,
        selection=selection,
    )

    ppr = Namespace(PPR_NAMESPACE)
    execution = URIRef(f"{before.namespace}process_execution_0001")
    expected = {
        (URIRef(before.specification_iri), ppr.hasProcessExecution, execution),
        (execution, RDF.type, ppr.processExecution),
        (execution, ppr.runsProcess, URIRef(PROCESS_IRI)),
        (
            execution,
            ppr.runsOnResource,
            URIRef(f"{RESOURCE_NAMESPACE}xarm6"),
        ),
    }
    assert result.accepted is True
    assert result.assertion_count == 4
    assert set(result.abox.graph) - set(before.graph) == expected
    assert derive_resource_assignment_need(result.abox, workcell) is None
    assert set(workcell.graph) == set(load_predefined_workcell(tbox, registry).graph)
    assert not any(
        "@localhost" in str(node)
        for triple in result.abox.graph
        for node in triple
    )
    assert load_interaction_abox(root, tbox).graph.isomorphic(result.abox.graph)


def test_configured_process_identity_reaches_the_final_abox(tmp_path: Path) -> None:
    custom_process_iri = "https://cais-spade-llm.local/process/configured_process"
    root = tmp_path / "interaction"
    tbox, registry, workcell, _manifest_paths = _authorities(
        tmp_path,
        process_symbol="configured_process",
        process_iri=custom_process_iri,
    )
    abox = _grounded_abox(root, tbox, process_iri=custom_process_iri)
    need = derive_resource_assignment_need(abox, workcell)
    assert need is not None
    selection = select_predefined_resource(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        need=need,
        grounding_record_path=_write_world_location(root, (0.0, 0.0, 1.1)),
    )

    result = commit_resource_assignment(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        need=need,
        selection=selection,
    )

    ppr = Namespace(PPR_NAMESPACE)
    execution = URIRef(f"{abox.namespace}process_execution_0001")
    assert (execution, ppr.runsProcess, URIRef(custom_process_iri)) in result.abox.graph
    assert load_interaction_abox(root, tbox).graph.isomorphic(result.abox.graph)


def test_no_reachable_resource_and_tampered_selection_cannot_commit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, manifest_paths = _authorities(tmp_path)
    before = _grounded_abox(root, tbox)
    need = derive_resource_assignment_need(before, workcell)
    assert need is not None
    selection = select_predefined_resource(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        need=need,
        grounding_record_path=_write_world_location(root, (2.0, 0.0, 1.1)),
    )

    with pytest.raises(ResourceGroundingError, match="No predefined resource"):
        commit_resource_assignment(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            need=need,
            selection=selection,
        )
    assert set(load_interaction_abox(root, tbox).graph) == set(before.graph)

    forged = replace(
        selection,
        selected_resource_symbol="xarm6",
        selected_resource_iri=f"{RESOURCE_NAMESPACE}xarm6",
        selected_resource_jid="xarm6@localhost",
        selected_execution_mode="simulation",
    )
    with pytest.raises(ResourceGroundingError, match="decision changed"):
        commit_resource_assignment(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            need=need,
            selection=forged,
        )
    assert set(load_interaction_abox(root, tbox).graph) == set(before.graph)


def _authorities(
    root: Path,
    *,
    hidden_value: str = "ignored",
    process_symbol: str = "assembly",
    process_iri: str = PROCESS_IRI,
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
        hidden_value=hidden_value,
    )
    _write_manifest(
        manifest_paths["ur5e"],
        "ur5e",
        origin_y=0.5,
        workspace_y=(-0.35, 1.1),
        hidden_value=hidden_value,
    )
    profile_path = root / "workcell_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "process": {"symbol": process_symbol, "iri": process_iri},
                "resources": [
                    {
                        "symbol": symbol,
                        "iri": f"{RESOURCE_NAMESPACE}{symbol}",
                        "manifest_ref": path.relative_to(root).as_posix(),
                    }
                    for symbol, path in manifest_paths.items()
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


def _write_manifest(
    path: Path,
    symbol: str,
    *,
    origin_y: float,
    workspace_y: tuple[float, float],
    hidden_value: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                symbol: {
                    "type": "robot",
                    "jid": f"{symbol}@localhost",
                    "password": hidden_value,
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
                            "reachability": [hidden_value],
                        },
                        "controller": {"parts_tuning": hidden_value},
                        "spawn": hidden_value,
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
    process_iri: str = PROCESS_IRI,
) -> ABoxSnapshot:
    abox = initialize_interaction_abox(root, "assemble Medium Gear", tbox)
    return _merge_semantic_grounding(root, tbox, abox, process_iri=process_iri)


def _merge_semantic_grounding(
    root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
    *,
    process_iri: str = PROCESS_IRI,
) -> ABoxSnapshot:
    ppr = Namespace(PPR_NAMESPACE)
    feature_iri = f"{abox.namespace}feature_0001"
    result = validate_and_merge_triple_delta(
        root,
        tbox,
        "ontology_grounding_host",
        {
            "assertions": [
                _assertion(feature_iri, str(RDF.type), str(ppr.feature)),
                _assertion(abox.specification_iri, str(ppr.defines), feature_iri),
                _assertion(process_iri, str(ppr.realizes), feature_iri),
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
    translation: tuple[float, float, float],
    **updates: object,
) -> Path:
    destination = root / "products/grounding/synthetic_world_location"
    destination.mkdir(parents=True, exist_ok=True)
    source_path = destination / "source_evidence.json"
    source_path.write_text('{"source":"synthetic"}', encoding="utf-8")
    source_ref = source_path.relative_to(root).as_posix()
    record: dict[str, object] = {
        "schema_version": 1,
        "record_type": "RobotFrameLocationRecord",
        "producer": "synthetic_world_location_provider",
        "robot_frame_conversion": "accepted",
        "CAD_correspondence": "accepted",
        "location": "available",
        "target_frame": "world",
        "observation_timestamp_ns": 11,
        "translated_location_m": list(translation),
        "source_hashes": [
            {
                "ref": source_ref,
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            }
        ],
    }
    record.update(updates)
    path = destination / "robot_frame_location_record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path

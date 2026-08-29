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
    RobotFramePoseEvidenceError,
    commit_resource_assignment,
    derive_resource_assignment_need,
    select_predefined_resource,
)
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
    assert need.feature_iri == f"{abox.namespace}medium_gear_feature"
    assert need.process_iri == PROCESS_IRI
    assert need.candidate_resource_iris == (
        f"{RESOURCE_NAMESPACE}xarm6",
        f"{RESOURCE_NAMESPACE}ur5e",
    )
    assert need.required_record_type == "RobotFramePoseRecord"
    assert need.target_frame == "world"


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
    pose_path = _write_world_pose(root, translation)

    selection = select_predefined_resource(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        need=need,
        robot_frame_pose_path=pose_path,
        manifest_paths=manifest_paths,
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
            robot_frame_pose_path=_write_world_pose(
                interaction_root, (0.0, 0.0, 1.1)
            ),
            manifest_paths=manifest_paths,
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
        robot_frame_pose_path=_write_world_pose(root, (0.0, 0.0, 1.1)),
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
        ({"target_frame": "camera"}, "world-frame"),
        ({"status": "ambiguous"}, "accepted"),
        ({"status": "stale"}, "accepted"),
        ({"status": "rejected"}, "accepted"),
    ],
)
def test_selection_rejects_unusable_pose_without_creating_a_record(
    tmp_path: Path,
    record_update: dict[str, object],
    message: str,
) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, manifest_paths = _authorities(tmp_path)
    need = derive_resource_assignment_need(_grounded_abox(root, tbox), workcell)
    assert need is not None
    pose_path = _write_world_pose(root, (0.0, 0.0, 1.1), **record_update)

    with pytest.raises(RobotFramePoseEvidenceError, match=message):
        select_predefined_resource(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            need=need,
            robot_frame_pose_path=pose_path,
            manifest_paths=manifest_paths,
        )

    assert not (root / "products/grounding/resource_selection").exists()


def test_selection_requires_an_intact_pose_source_hash_chain(tmp_path: Path) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, manifest_paths = _authorities(tmp_path)
    need = derive_resource_assignment_need(_grounded_abox(root, tbox), workcell)
    assert need is not None
    pose_path = _write_world_pose(root, (0.0, 0.0, 1.1))
    pose_record = json.loads(pose_path.read_text(encoding="utf-8"))
    source_ref = pose_record["source_hashes"][0]["ref"]

    pose_record.pop("source_hashes")
    pose_path.write_text(json.dumps(pose_record), encoding="utf-8")
    with pytest.raises(RobotFramePoseEvidenceError, match="hash reference"):
        select_predefined_resource(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            need=need,
            robot_frame_pose_path=pose_path,
            manifest_paths=manifest_paths,
        )

    pose_path = _write_world_pose(root, (0.0, 0.0, 1.1))
    (root / source_ref).write_text("changed evidence", encoding="utf-8")
    with pytest.raises(RobotFramePoseEvidenceError, match="hash does not match"):
        select_predefined_resource(
            interaction_root=root,
            tbox=tbox,
            registry=registry,
            workcell=workcell,
            need=need,
            robot_frame_pose_path=pose_path,
            manifest_paths=manifest_paths,
        )


def test_manifest_errors_are_not_pose_evidence_errors(tmp_path: Path) -> None:
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
            robot_frame_pose_path=_write_world_pose(root, (0.0, 0.0, 1.1)),
            manifest_paths=manifest_paths,
        )

    assert not isinstance(raised.value, RobotFramePoseEvidenceError)


def test_selection_detects_changed_pose_and_manifest_sources(tmp_path: Path) -> None:
    root = tmp_path / "interaction"
    tbox, registry, workcell, manifest_paths = _authorities(tmp_path)
    need = derive_resource_assignment_need(_grounded_abox(root, tbox), workcell)
    assert need is not None
    pose_path = _write_world_pose(root, (0.0, 0.0, 1.1))
    selection = select_predefined_resource(
        interaction_root=root,
        tbox=tbox,
        registry=registry,
        workcell=workcell,
        need=need,
        robot_frame_pose_path=pose_path,
        manifest_paths=manifest_paths,
    )

    pose_record = json.loads(pose_path.read_text(encoding="utf-8"))
    pose_record["observation_timestamp_ns"] = 12
    pose_path.write_text(json.dumps(pose_record), encoding="utf-8")
    with pytest.raises(ResourceGroundingError, match="pose changed"):
        selection.assert_unchanged()

    pose_record["observation_timestamp_ns"] = 11
    pose_path.write_text(json.dumps(pose_record), encoding="utf-8")
    manifest_paths["xarm6"].write_text("{}", encoding="utf-8")
    with pytest.raises(ResourceGroundingError, match="authority changed"):
        selection.assert_unchanged()


def test_host_commit_adds_only_the_derived_execution_assignment(
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
        robot_frame_pose_path=_write_world_pose(root, (0.0, 0.0, 1.1)),
        manifest_paths=manifest_paths,
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
        robot_frame_pose_path=_write_world_pose(root, (2.0, 0.0, 1.1)),
        manifest_paths=manifest_paths,
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
    tbox = load_ppr_tbox(TBOX_PATH, ppr_namespace=PPR_NAMESPACE)
    registry = load_predefined_resource_registry(
        tbox,
        manifest_paths=manifest_paths,
        source_root=root,
    )
    return tbox, registry, load_predefined_workcell(tbox, registry), manifest_paths


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
                            "supports_manipulator_pick_place": True,
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


def _grounded_abox(root: Path, tbox: TBoxSnapshot) -> ABoxSnapshot:
    abox = initialize_interaction_abox(root, "assemble Medium Gear", tbox)
    return _merge_semantic_grounding(root, tbox, abox)


def _merge_semantic_grounding(
    root: Path,
    tbox: TBoxSnapshot,
    abox: ABoxSnapshot,
) -> ABoxSnapshot:
    ppr = Namespace(PPR_NAMESPACE)
    feature_iri = f"{abox.namespace}medium_gear_feature"
    result = validate_and_merge_triple_delta(
        root,
        tbox,
        "ontology_grounding_host",
        {
            "assertions": [
                _assertion(feature_iri, str(RDF.type), str(ppr.feature)),
                _assertion(abox.specification_iri, str(ppr.defines), feature_iri),
                _assertion(PROCESS_IRI, str(ppr.realizes), feature_iri),
            ]
        },
        authorized_evidence_refs={EVIDENCE_REF},
    )
    return result.abox


def _assertion(subject: str, predicate: str, object_iri: str) -> dict[str, object]:
    return {
        "subject": subject,
        "predicate": predicate,
        "object": {"kind": "iri", "value": object_iri},
        "evidence_refs": [EVIDENCE_REF],
    }


def _write_world_pose(
    root: Path,
    translation: tuple[float, float, float],
    **updates: object,
) -> Path:
    destination = root / "products/grounding/synthetic_world_pose"
    destination.mkdir(parents=True, exist_ok=True)
    source_path = destination / "source_evidence.json"
    source_path.write_text('{"source":"synthetic"}', encoding="utf-8")
    source_ref = source_path.relative_to(root).as_posix()
    record: dict[str, object] = {
        "schema_version": 1,
        "record_type": "RobotFramePoseRecord",
        "producer": "synthetic_world_pose_provider",
        "status": "accepted",
        "target_frame": "world",
        "observation_timestamp_ns": 11,
        "robot_frame_pose": {"CAD_origin_translation_m": list(translation)},
        "source_hashes": [
            {
                "ref": source_ref,
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            }
        ],
    }
    record.update(updates)
    path = destination / "robot_frame_pose_record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path

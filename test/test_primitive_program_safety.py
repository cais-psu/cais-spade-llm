"""Offline physical observations and reusable safety specification acceptance tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ppr_ap_migration import migrate_ap_key, migrate_ppr_fixture

from cais_spade_llm.agents.central_controller import primitive_program_safety
from cais_spade_llm.agents.central_controller.primitive_program_safety import (
    validate_primitive_program_safety,
)


def _pose(x: float) -> list[float]:
    return [x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _move(points: list[tuple[float, float]], *, scalar_params: bool = False) -> dict:
    if scalar_params:
        params = dict(zip(
            ("x", "y", "z", "qx", "qy", "qz", "qw"), _pose(points[-1][1]), strict=True,
        ))
        params["speed"] = 0.1
    else:
        params = {"target": _pose(points[-1][1])}
    return {
        "primitive": "move_cartesian",
        "resolved_params": params,
        "start_time": points[0][0],
        "end_time": points[-1][0],
        "model_evidence": {
            "frame": "world",
            "trajectory": [{"time": time, "pose": _pose(x)} for time, x in points],
        },
    }


def _compute(start: float, end: float, x: float, destination: str = "M2") -> dict:
    return {
        "primitive": "compute_place_targets",
        "resolved_params": {"part_name": "p", "destination_location": destination},
        "start_time": start,
        "end_time": end,
        "model_evidence": {"frame": "world", "outputs": {"target": _pose(x)}},
    }


def _release(start: float, end: float, x: float, contained_by: str | None) -> dict:
    return {
        "primitive": "release_part",
        "resolved_params": {"model_name": "p", "part_name": "p"},
        "start_time": start,
        "end_time": end,
        "model_evidence": {
            "frame": "world",
            "released_part": {
                "part_name": "p",
                "frame": "world",
                "pose": _pose(x),
                "contained_by": contained_by,
                "stationary_until": 1.0,
            }
        },
    }


def _program(resource_id: str, *steps: dict) -> dict:
    records = deepcopy(list(steps))
    for index, record in enumerate(records):
        record["step_index"] = index
    return {
        "resource_id": resource_id,
        "event_name": "recovery_event_42",
        "in_state": "llm_state_17",
        "out_state": "llm_state_18",
        "primitive_steps": [
            {"primitive": record["primitive"], "params": deepcopy(record["resolved_params"])}
            for record in records
        ],
        "step_results": records,
    }


def _resource_geometry(footprint: float = 0.05) -> dict:
    return {"frame": "world", "footprint": [-footprint, footprint], "attachment_offset": 0.0}


def _entry_case(
    *, start: float = 0.80, end: float = 1.10, occupied: bool = True,
    resource: str = "KMR", receiver: str = "M2",
    region: str = "M2 receiving region", interval: tuple[float, float] = (1.0, 1.2),
    footprint: float = 0.05,
) -> dict:
    return {
        "programs": [_program(resource, _move([(0.0, start), (1.0, end)]))],
        "snapshot": {
            "resources": {
                resource: {"current_pose": _pose(start), "held_part": "p"},
                receiver: {"contained_parts": ["q"] if occupied else []},
            },
            "parts": {
                "p": {"current_pose": _pose(start), "contained_by": None},
                **({"q": {"current_pose": _pose(sum(interval) / 2), "contained_by": receiver}}
                   if occupied else {}),
            },
        },
        "geometry": {
            "frame": "world", "axis": 0,
            "regions": {region: {"frame": "world", "interval": list(interval)}},
            "resources": {resource: _resource_geometry(footprint)},
            "parts": {"p": {"frame": "world", "footprint": [-footprint, footprint]}},
        },
        "horizon": [0.0, 1.0],
        "stationary": {receiver: [[0.0, 1.0]]},
        "bindings": [{
            "rule_id": "receiving_rule", "specification": "receiving_region_entry",
            "resource": resource, "receiving_resource": receiver,
            "part": "p", "region": region,
        }],
    }


def _mutex_case(*, first: float = 0.80, second: float = 1.40) -> dict:
    return {
        "programs": [],
        "snapshot": {
            "resources": {
                "KMR": {"current_pose": _pose(first), "held_part": None},
                "xarm6": {"current_pose": _pose(second), "held_part": None},
            },
            "parts": {},
        },
        "geometry": {
            "frame": "world", "axis": 0,
            "regions": {"assembly_board-v1": {"frame": "world", "interval": [1.0, 1.2]}},
            "resources": {resource: _resource_geometry() for resource in ("KMR", "xarm6")},
            "parts": {},
        },
        "horizon": [0.0, 1.0],
        "stationary": {resource: [[0.0, 1.0]] for resource in ("KMR", "xarm6")},
        "bindings": [{
            "rule_id": "assembly_mutex", "specification": "shared_area_mutex",
            "resources": ["KMR", "xarm6"], "region": "assembly_board-v1",
        }],
    }


def _check(case: dict) -> dict:
    return validate_primitive_program_safety(**case)


def _violation(result: dict) -> dict:
    assert result["is_safe"] is False, result
    finding = next(row for row in result["findings"] if row["constraint_code"] == "safety_rule_violation")
    return finding["evidence"]


def _unavailable(result: dict) -> None:
    assert result["is_safe"] is False, result
    assert any(row["constraint_code"] == "safety_validation_unavailable" for row in result["findings"])
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["safety_ctx"]["status"] == "unavailable"
    assert result["safety_dfa_states_after"] == result["safety_dfa_states_before"]


def test_original_entry_example_finds_first_contact_inside_generated_event() -> None:
    """Find the intermediate contact rather than relying on the event endpoint."""
    result = _check(_entry_case())
    finding = _violation(result)
    assert finding["time"] == pytest.approx(0.5)
    assert finding["ap_values"] == {"ap001": True, "ap002": True}
    row = next(row for row in result["observations"] if row["time"] == pytest.approx(0.5))
    assert row["resources"]["KMR"]["current_pose"][0] == pytest.approx(0.95)
    assert row["region_occupancy"]["M2 receiving region"]["KMR"] is True
    assert result["safety_dfa_states_after"] == result["safety_dfa_states_before"]


def test_staging_motion_avoiding_receiving_region_passes() -> None:
    """Check a supported recovery path entirely outside the occupied region."""
    result = _check(_entry_case(end=0.40))
    assert result["is_safe"] is True, result
    assert result["projected_snapshot"]["resources"]["KMR"]["current_pose"][0] == 0.40


def test_existing_scalar_move_cartesian_interface_is_preserved() -> None:
    """A robot's scalar pose parameters retain the same observation semantics."""
    case = _entry_case(resource="xarm6")
    case["programs"] = [_program("xarm6", _move([(0.0, 0.8), (1.0, 1.1)], scalar_params=True))]
    before = deepcopy(case["programs"])
    assert _violation(_check(case))["time"] == pytest.approx(0.5)
    assert case["programs"] == before


@pytest.mark.parametrize("points", [
    [(0.0, 0.8), (1.0, 1.4)],
    [(0.0, 0.8), (0.5, 1.1), (1.0, 0.8)],
])
def test_safe_endpoints_do_not_hide_entry_between_them(points: list[tuple[float, float]]) -> None:
    """Reject both transit and an out-and-back excursion through the region."""
    case = _entry_case()
    case["programs"] = [_program("KMR", _move(points))]
    assert _violation(_check(case))["time"] < 1.0


@pytest.mark.parametrize("resource,receiver,region,interval,footprint,start,end,contact", [
    ("KMR", "M2", "M2 receiving region", (1.0, 1.2), 0.05, 0.8, 1.1, 0.95),
    ("Robot-17", "M1", "Receiving Area A", (4.0, 4.8), 0.2, 3.0, 4.5, 3.8),
    ("Robot 17", "Resource_exact-v3", "Region_Exact", (-3.0, -2.0), 0.1, -4.0, -2.5, -3.1),
])
def test_bindings_and_geometry_drive_the_same_evaluator(
    resource: str, receiver: str, region: str, interval: tuple[float, float],
    footprint: float, start: float, end: float, contact: float,
) -> None:
    """Preserve identifiers while deriving entry from each configured interval."""
    result = _check(_entry_case(
        resource=resource, receiver=receiver, region=region, interval=interval,
        footprint=footprint, start=start, end=end,
    ))
    finding = _violation(result)
    assert finding["time"] == pytest.approx((contact - start) / (end - start))


def test_generated_state_and_event_names_cannot_change_ap_valuations() -> None:
    """New names describe no physical evidence and cannot bypass a requirement."""
    case = _entry_case()
    original = _check(case)
    case["programs"][0].update(
        event_name="nominal_place", in_state="safe_to_load", out_state="ready_for_transfer",
    )
    renamed = _check(case)
    assert original["is_safe"] == renamed["is_safe"] is False
    assert _violation(original)["time"] == _violation(renamed)["time"]
    assert [row["ap_values"] for row in original["safety_ctx"]["rule_checks"]] == [
        row["ap_values"] for row in renamed["safety_ctx"]["rule_checks"]
    ]
    assert [row["region_occupancy"] for row in original["observations"]] == [
        row["region_occupancy"] for row in renamed["observations"]
    ]


def test_computing_targets_does_not_move_the_incoming_part() -> None:
    """A target inside the receiving region is data until motion begins."""
    case = _entry_case()
    case["programs"] = [_program("KMR", _compute(0.0, 1.0, 1.1))]
    result = _check(case)
    assert result["is_safe"] is True, result
    assert all(row["resources"]["KMR"]["current_pose"][0] == 0.8 for row in result["observations"])
    assert result["projected_snapshot"]["parts"]["p"]["current_pose"][0] == 0.8
    case["programs"] = [_program("KMR", _compute(0.0, 0.2, 1.1), _move([(0.2, 0.8), (1.0, 1.1)]))]
    assert _violation(_check(case))["time"] == pytest.approx(0.6)


@pytest.mark.parametrize("start,end,safe", [
    (1.10, 1.15, True),
    (1.10, 1.10, True),
    (0.80, 0.80, True),
    (0.80, 0.95, False),
    (1.40, 1.10, False),
])
def test_entry_initial_state_boundary_reverse_and_zero_movement(start: float, end: float, safe: bool) -> None:
    """Entry requires a transition; touching the closed region is sufficient."""
    result = _check(_entry_case(start=start, end=end))
    assert result["is_safe"] is safe, result
    if not safe:
        _violation(result)


def test_leaving_initial_occupancy_then_reentering_produces_a_new_entry() -> None:
    """Starting inside is allowed by the entry rule until a later reentry."""
    case = _entry_case(start=1.1)
    case["programs"] = [_program("KMR", _move([(0.0, 1.1), (0.5, 0.8), (1.0, 1.1)]))]
    assert _violation(_check(case))["time"] == pytest.approx(0.75)


def test_incoming_part_is_not_its_own_other_occupant_after_release() -> None:
    """A single modeled deposit does not invent another part in the receiver."""
    case = _entry_case(occupied=False)
    case["programs"] = [_program(
        "KMR", _move([(0.0, 0.8), (0.6, 1.1)]), _release(0.6, 0.7, 1.1, "M2"),
        _move([(0.7, 1.1), (1.0, 0.8)]),
    )]
    result = _check(case)
    assert result["is_safe"] is True, result
    assert result["projected_snapshot"]["resources"]["M2"]["contained_parts"] == ["p"]
    assert result["projected_snapshot"]["parts"]["p"]["contained_by"] == "M2"
    assert result["projected_snapshot"]["resources"]["KMR"]["held_part"] is None


def test_initial_shared_area_overlap_rejects_without_motion() -> None:
    """The mutex is a persistent state constraint, including initial occupancy."""
    assert _violation(_check(_mutex_case(first=1.1, second=1.15)))["time"] == 0.0


def test_two_approaches_are_checked_on_one_shared_time_axis() -> None:
    """Concurrent approaches conflict even though both started outside."""
    case = _mutex_case()
    case["stationary"] = {}
    case["programs"] = [
        _program("KMR", _move([(0.0, 0.8), (1.0, 1.1)])),
        _program("xarm6", _move([(0.0, 1.4), (1.0, 1.1)])),
    ]
    assert _violation(_check(case))["time"] == pytest.approx(0.5)


def test_simultaneous_boundary_exit_and_entry_violates_closed_mutex() -> None:
    """Zero-duration simultaneous contact still occupies the shared area."""
    case = _mutex_case(first=1.1)
    case["stationary"] = {}
    case["programs"] = [
        _program("KMR", _move([(0.0, 1.1), (1.0, 0.8)])),
        _program("xarm6", _move([(0.0, 1.4), (1.0, 1.1)])),
    ]
    assert _violation(_check(case))["time"] == pytest.approx(0.5)


def test_mutex_allows_entry_after_complete_retreat() -> None:
    """Disjoint occupancy intervals permit an ordered handoff."""
    case = _mutex_case(first=1.1)
    case["stationary"] = {"KMR": [[0.4, 1.0]], "xarm6": [[0.0, 0.6]]}
    case["programs"] = [
        _program("KMR", _move([(0.0, 1.1), (0.4, 0.8)])),
        _program("xarm6", _move([(0.6, 1.4), (1.0, 1.1)])),
    ]
    assert _check(case)["is_safe"] is True


@pytest.mark.parametrize("retreat", [1.1, 0.98, 0.80])
def test_release_preserves_robot_occupancy_until_complete_retreat(retreat: float) -> None:
    """A deposited part leaves custody, while partial robot retreat retains mutex."""
    case = _mutex_case(first=1.1)
    case["snapshot"]["resources"]["KMR"]["held_part"] = "p"
    case["snapshot"]["parts"]["p"] = {"current_pose": _pose(1.1), "contained_by": None}
    case["geometry"]["parts"]["p"] = {"frame": "world", "footprint": [-0.05, 0.05]}
    case["stationary"] = {"KMR": [[0.4, 1.0]], "xarm6": [[0.0, 0.6]]}
    case["programs"] = [
        _program("KMR", _release(0.0, 0.1, 1.1, None), _move([(0.1, 1.1), (0.4, retreat)])),
        _program("xarm6", _move([(0.6, 1.4), (1.0, 1.1)])),
    ]
    result = _check(case)
    assert result["is_safe"] is (retreat == 0.80), result
    if retreat == 0.80:
        final = result["projected_snapshot"]
        assert final["resources"]["KMR"]["held_part"] is None
        assert final["parts"]["p"]["current_pose"][0] == 1.1
        assert final["resources"]["KMR"]["current_pose"][0] == 0.8
    else:
        _violation(result)


def test_held_part_geometry_and_attachment_offset_extend_resource_occupancy() -> None:
    """A carried part can enter before the tool's own footprint does."""
    case = _mutex_case(first=0.8, second=1.1)
    case["snapshot"]["resources"]["KMR"]["held_part"] = "p"
    case["snapshot"]["parts"]["p"] = {"current_pose": _pose(0.98), "contained_by": None}
    case["geometry"]["resources"]["KMR"]["attachment_offset"] = 0.18
    case["geometry"]["parts"]["p"] = {"frame": "world", "footprint": [-0.05, 0.05]}
    assert _violation(_check(case))["time"] == 0.0


def test_final_occupancy_is_retained_when_checking_the_next_program() -> None:
    """A completed program cannot clear the next program's initial mutex facts."""
    case = _mutex_case()
    case["stationary"] = {"xarm6": [[0.0, 1.0]]}
    case["programs"] = [_program("KMR", _move([(0.0, 0.8), (1.0, 1.1)]))]
    first = _check(case)
    assert first["is_safe"] is True, first
    case["snapshot"] = first["projected_snapshot"]
    case["safety_dfa_states_before"] = first["safety_dfa_states_after"]
    case["stationary"] = {"KMR": [[0.0, 1.0]]}
    case["programs"] = [_program("xarm6", _move([(0.0, 1.4), (1.0, 1.1)]))]
    _violation(_check(case))


def test_deposited_part_blocks_a_later_incoming_part() -> None:
    """Containment projected by release supplies the next entry check's facts."""
    case = _entry_case(occupied=False)
    case["programs"] = [_program(
        "KMR", _move([(0.0, 0.8), (0.6, 1.1)]), _release(0.6, 0.7, 1.1, "M2"),
        _move([(0.7, 1.1), (1.0, 0.8)]),
    )]
    first = _check(case)
    assert first["is_safe"] is True, first
    case["snapshot"] = first["projected_snapshot"]
    case["snapshot"]["resources"]["xarm6"] = {"current_pose": _pose(0.8), "held_part": "r"}
    case["snapshot"]["parts"]["r"] = {"current_pose": _pose(0.8), "contained_by": None}
    case["geometry"]["resources"]["xarm6"] = _resource_geometry()
    case["geometry"]["parts"]["r"] = {"frame": "world", "footprint": [-0.05, 0.05]}
    case["programs"] = [_program("xarm6", _move([(0.0, 0.8), (1.0, 1.1)]))]
    case["stationary"]["KMR"] = [[0.0, 1.0]]
    case["bindings"][0].update(resource="xarm6", part="r")
    assert _violation(_check(case))["time"] == pytest.approx(0.5)


@pytest.mark.parametrize("missing", ["region", "resource", "part", "occupancy", "trajectory", "receiver_coverage"])
def test_missing_authoritative_inputs_remain_unresolved(missing: str) -> None:
    """Absence of required evidence must never be interpreted as a safe path."""
    case = _entry_case()
    if missing == "region":
        case["geometry"]["regions"].clear()
    elif missing == "resource":
        case["geometry"]["resources"].clear()
    elif missing == "part":
        case["geometry"]["parts"].clear()
    elif missing == "occupancy":
        del case["snapshot"]["resources"]["M2"]["contained_parts"]
    elif missing == "trajectory":
        case["programs"][0]["step_results"][0]["model_evidence"].clear()
    else:
        case["stationary"].clear()
    _unavailable(_check(case))


@pytest.mark.parametrize("source", ["region", "resource", "part", "trajectory"])
def test_frame_mismatch_remains_unresolved(source: str) -> None:
    """Coordinates from different frames cannot be compared without evidence."""
    case = _entry_case()
    values = {
        "region": case["geometry"]["regions"]["M2 receiving region"],
        "resource": case["geometry"]["resources"]["KMR"],
        "part": case["geometry"]["parts"]["p"],
        "trajectory": case["programs"][0]["step_results"][0]["model_evidence"],
    }
    values[source]["frame"] = "robot_base"
    _unavailable(_check(case))


def test_missing_other_part_containment_is_unresolved() -> None:
    """Receiver inventory must agree with explicit part containment evidence."""
    case = _entry_case()
    del case["snapshot"]["parts"]["q"]["contained_by"]
    _unavailable(_check(case))


@pytest.mark.parametrize("primitive", ["compute_place_targets", "release_part"])
def test_non_motion_primitives_require_their_modeled_outputs(primitive: str) -> None:
    """A known primitive name cannot substitute for target or release evidence."""
    case = _entry_case(start=1.1)
    step = _compute(0.0, 1.0, 1.1) if primitive == "compute_place_targets" else _release(0.0, 1.0, 1.1, "M2")
    step["model_evidence"].clear()
    case["programs"] = [_program("KMR", step)]
    _unavailable(_check(case))


def test_incomplete_other_participant_motion_coverage_is_unresolved() -> None:
    """All mutex participants require motion or stationary coverage throughout."""
    case = _mutex_case()
    case["stationary"] = {"xarm6": [[0.0, 0.4]]}
    case["programs"] = [_program("KMR", _move([(0.0, 0.8), (1.0, 1.1)]))]
    _unavailable(_check(case))


def test_uncovered_gap_between_motion_steps_requires_stationary_evidence() -> None:
    """The observer does not infer how a participant behaves during a gap."""
    case = _entry_case(end=0.4)
    case["programs"] = [_program(
        "KMR", _move([(0.0, 0.8), (0.3, 0.7)]), _move([(0.6, 0.7), (1.0, 0.4)]),
    )]
    _unavailable(_check(case))
    case["stationary"]["KMR"] = [[0.3, 0.6]]
    assert _check(case)["is_safe"] is True


@pytest.mark.parametrize("outcome", ["safe", "violation", "unresolved"])
def test_candidate_checks_do_not_mutate_input_state_or_evidence(outcome: str) -> None:
    """Safety checking is offline and must not advance caller-owned monitors."""
    case = _entry_case(end=0.4 if outcome == "safe" else 1.1)
    if outcome == "unresolved":
        case["geometry"]["regions"].clear()
    case["safety_dfa_states_before"] = {"receiving_rule": "1"}
    before = deepcopy(case)
    result = _check(case)
    assert case == before
    if outcome == "unresolved":
        _unavailable(result)
    elif outcome == "violation":
        _violation(result)
    else:
        assert result["is_safe"] is True, result


def test_rule_bindings_do_not_share_ap_values() -> None:
    """A different occupied resource must not contaminate an empty receiver."""
    case = _entry_case(occupied=False)
    case["snapshot"]["resources"]["M1"] = {"contained_parts": ["q"]}
    case["snapshot"]["parts"]["q"] = {"current_pose": _pose(3.1), "contained_by": "M1"}
    case["geometry"]["regions"]["M1 receiving region"] = {"frame": "world", "interval": [3.0, 3.2]}
    case["stationary"]["M1"] = [[0.0, 1.0]]
    case["bindings"].append({
        "rule_id": "other_receiving_rule", "specification": "receiving_region_entry",
        "resource": "KMR", "receiving_resource": "M1", "part": "p", "region": "M1 receiving region",
    })
    result = _check(case)
    assert result["is_safe"] is True, result
    assert set(result["safety_dfa_states_after"]) == {"receiving_rule", "other_receiving_rule"}


def test_simultaneous_release_updates_inventory_before_another_entry_is_checked() -> None:
    """A new deposit counts as another part at the same instant as entry."""
    case = _entry_case(resource="xarm6", occupied=False)
    case["snapshot"]["resources"]["xarm6"]["held_part"] = "r"
    case["snapshot"]["parts"]["r"] = case["snapshot"]["parts"].pop("p")
    case["snapshot"]["resources"]["KMR"] = {"current_pose": _pose(1.1), "held_part": "p"}
    case["snapshot"]["parts"]["p"] = {"current_pose": _pose(1.1), "contained_by": None}
    case["geometry"]["resources"]["KMR"] = _resource_geometry()
    case["geometry"]["parts"]["r"] = deepcopy(case["geometry"]["parts"]["p"])
    case["stationary"]["KMR"] = [[0.5, 1.0]]
    case["bindings"][0]["part"] = "r"
    case["programs"].append(_program("KMR", _release(0.0, 0.5, 1.1, "M2")))
    evidence = _violation(_check(case))
    assert evidence["time"] == pytest.approx(0.5)
    assert evidence["resources"]["M2"]["contained_parts"] == ["p"]
    assert evidence["ap_values"] == {"ap001": True, "ap002": True}


@pytest.mark.parametrize("mismatch", [
    "target", "resolved_params", "discontinuous_start", "orientation", "other_axis",
    "unknown_primitive", "nan", "infinity",
])
def test_motion_evidence_must_cover_the_exact_supported_primitive(mismatch: str) -> None:
    """Unrelated trajectories and unsupported motion cannot justify safety."""
    case = _entry_case(end=0.4)
    program = case["programs"][0]
    authored = program["primitive_steps"][0]
    traced = program["step_results"][0]
    trajectory = traced["model_evidence"]["trajectory"]
    if mismatch == "target":
        authored["params"]["target"][0] = 0.3
        traced["resolved_params"]["target"][0] = 0.3
    elif mismatch == "resolved_params":
        traced["resolved_params"]["target"][0] = 0.3
    elif mismatch == "discontinuous_start":
        trajectory[0]["pose"][0] = 0.7
    elif mismatch in {"orientation", "other_axis"}:
        coordinate = 3 if mismatch == "orientation" else 1
        trajectory[-1]["pose"][coordinate] = 0.2
        authored["params"]["target"][coordinate] = 0.2
        traced["resolved_params"]["target"][coordinate] = 0.2
    elif mismatch == "unknown_primitive":
        authored["primitive"] = traced["primitive"] = "unmodeled_motion"
    else:
        trajectory[0]["pose"][0] = float(mismatch)
    _unavailable(_check(case))


@pytest.mark.parametrize("missing", ["no_bindings", "duplicate_rule_id", "unknown_specification", "unknown_dfa_state"])
def test_rule_binding_and_monitor_errors_are_unresolved(missing: str) -> None:
    """The checker cannot silently omit a rule or invent a monitor state."""
    case = _entry_case(end=0.4)
    if missing == "no_bindings":
        case["bindings"].clear()
    elif missing == "duplicate_rule_id":
        case["bindings"].append(deepcopy(case["bindings"][0]))
    elif missing == "unknown_specification":
        case["bindings"][0]["specification"] = "missing_specification"
    else:
        case["safety_dfa_states_before"] = {"receiving_rule": "missing_state"}
    _unavailable(_check(case))


@pytest.mark.parametrize("corruption", [
    "document", "version", "specifications", "specification", "missing_specification",
    "duplicate_specification", "unknown_specification", "missing_full", "duplicate_full",
    "unknown_full", "duplicate_label", "invalid_label", "formula", "meaning", "requirement",
])
def test_inconsistent_ap_definitions_do_not_silently_select_other_semantics(
    corruption: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Invalid definitions preserve monitor state and require corrected evidence."""
    definitions = json.loads(primitive_program_safety._DEFINITIONS.read_text(encoding="utf-8"))
    receiving = definitions["specifications"][0]
    first, second = receiving["aps"]
    if corruption == "document":
        definitions = []
    elif corruption == "version":
        definitions["version"] = 99
    elif corruption == "specifications":
        definitions["specifications"] = {}
    elif corruption == "specification":
        definitions["specifications"][0] = None
    elif corruption == "missing_specification":
        definitions["specifications"].pop()
    elif corruption == "duplicate_specification":
        definitions["specifications"].append(deepcopy(receiving))
    elif corruption == "unknown_specification":
        receiving["id"] = "other_requirement"
    elif corruption == "missing_full":
        del first["full"]
    elif corruption == "duplicate_full":
        second["full"] = first["full"]
    elif corruption == "unknown_full":
        first["full"] = "ap_event/physical_observation/unknown"
    elif corruption == "duplicate_label":
        second["label"] = first["label"]
    elif corruption == "invalid_label":
        first["label"] = "not_an_ap"
    elif corruption == "formula":
        receiving["formula"] = "G (ap001 | ap002)"
    elif corruption == "meaning":
        first["meaning"] = "True when motion is safe."
    else:
        receiving["requirement"] = "Permit all incoming parts."
    path = tmp_path / "primitive_observation_safety.json"
    path.write_text(json.dumps(definitions), encoding="utf-8")
    monkeypatch.setattr(primitive_program_safety, "_DEFINITIONS", path)
    case = _entry_case()
    case["safety_dfa_states_before"] = {"receiving_rule": "1"}
    result = _check(case)
    _unavailable(result)
    assert result["safety_dfa_states_after"] == {"receiving_rule": "1"}


def test_ap_full_definitions_select_predicates_independently_of_label_and_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Use each supplied exact AP label with the predicate named by its full ID."""
    definitions = json.loads(primitive_program_safety._DEFINITIONS.read_text(encoding="utf-8"))
    receiving = definitions["specifications"][0]
    receiving["aps"][0]["label"] = "ap101"
    receiving["aps"][1]["label"] = "ap202"
    receiving["aps"].reverse()
    receiving["formula"] = "G !(ap101 & ap202)"
    path = tmp_path / "primitive_observation_safety.json"
    path.write_text(json.dumps(definitions), encoding="utf-8")
    monkeypatch.setattr(primitive_program_safety, "_DEFINITIONS", path)
    result = _check(_entry_case())
    assert _violation(result)["ap_values"] == {"ap101": True, "ap202": True}
    assert all(set(row["ap_values"]) == {"ap101", "ap202"} for row in result["safety_ctx"]["rule_checks"])


def test_receiving_binding_cannot_hide_the_actual_held_part() -> None:
    """A binding for a different incoming part cannot suppress the entry AP."""
    case = _entry_case()
    case["snapshot"]["parts"]["r"] = {"current_pose": _pose(0.8), "contained_by": None}
    case["geometry"]["parts"]["r"] = {"frame": "world", "footprint": [-0.05, 0.05]}
    case["bindings"][0]["part"] = "r"
    _unavailable(_check(case))


def test_empty_retreat_does_not_fabricate_an_incoming_part() -> None:
    """The same receiving binding remains meaningful after custody is released."""
    case = _entry_case(start=1.1, end=0.8)
    case["snapshot"]["resources"]["KMR"]["held_part"] = None
    case["snapshot"]["resources"]["M2"]["contained_parts"].append("p")
    case["snapshot"]["parts"]["p"]["contained_by"] = "M2"
    result = _check(case)
    assert result["is_safe"] is True, result
    assert all(not row["ap_values"]["ap001"] for row in result["safety_ctx"]["rule_checks"])


@pytest.mark.parametrize("inconsistency", ["duplicate_custody", "wrong_receiver", "missing_inventory", "duplicate_inventory"])
def test_initial_custody_and_containment_must_be_consistent(inconsistency: str) -> None:
    """Joint snapshots cannot claim incompatible possession of the same part."""
    case = _entry_case()
    if inconsistency == "duplicate_custody":
        case["snapshot"]["resources"]["xarm6"] = {"current_pose": _pose(0.8), "held_part": "p"}
        case["geometry"]["resources"]["xarm6"] = _resource_geometry()
        case["stationary"]["xarm6"] = [[0.0, 1.0]]
        case["bindings"].append({
            "rule_id": "duplicate_custody_mutex", "specification": "shared_area_mutex",
            "resources": ["KMR", "xarm6"], "region": "M2 receiving region",
        })
    elif inconsistency == "wrong_receiver":
        case["snapshot"]["parts"]["q"]["contained_by"] = "M1"
    elif inconsistency == "missing_inventory":
        case["snapshot"]["resources"]["M2"]["contained_parts"].clear()
    else:
        case["snapshot"]["resources"]["M2"]["contained_parts"].append("q")
    _unavailable(_check(case))


@pytest.mark.parametrize("outputs", [
    {"target": {"context_ref": "unknown_target"}},
    {"target": None},
    {"metadata": "target not supplied"},
])
def test_compute_place_targets_requires_resolved_coordinate_outputs(outputs: dict) -> None:
    """A nonempty output mapping alone does not establish a computed target."""
    case = _entry_case()
    step = _compute(0.0, 1.0, 1.1)
    step["model_evidence"]["outputs"] = outputs
    case["programs"] = [_program("KMR", step)]
    _unavailable(_check(case))


def test_compute_place_targets_accepts_existing_robot_target_pose_output() -> None:
    """The observer keeps the robot's coordinate output interface intact."""
    case = _entry_case(resource="xarm6")
    step = _compute(0.0, 1.0, 1.1)
    step["model_evidence"]["outputs"] = {"target_pose": {"x": 1.1, "y": 0.0, "z": 0.0}}
    case["programs"] = [_program("xarm6", step)]
    result = _check(case)
    assert result["is_safe"] is True, result
    assert result["projected_snapshot"]["resources"]["xarm6"]["current_pose"] == _pose(0.8)


@pytest.mark.parametrize("assembly_slot", [{}, {"x": 1.1, "y": 0.0, "z": 0.0}])
def test_release_part_with_assembly_slot_needs_a_placement_correction_model(assembly_slot: dict) -> None:
    """Gazebo placement correction is outside the fixed-position release model."""
    case = _entry_case(resource="xarm6", start=1.1, occupied=False)
    step = _release(0.0, 1.0, 1.1, "M2")
    step["resolved_params"]["assembly_slot"] = assembly_slot
    case["programs"] = [_program("xarm6", step)]
    _unavailable(_check(case))


def test_release_part_with_null_assembly_slot_preserves_the_modeled_pose() -> None:
    """Explicitly disabling placement correction keeps ordinary release supported."""
    case = _entry_case(resource="xarm6", start=1.1, occupied=False)
    step = _release(0.0, 1.0, 1.1, "M2")
    step["resolved_params"]["assembly_slot"] = None
    case["programs"] = [_program("xarm6", step)]
    result = _check(case)
    assert result["is_safe"] is True, result
    assert result["projected_snapshot"]["parts"]["p"]["current_pose"] == _pose(1.1)


@pytest.mark.parametrize("level,field,value", [
    ("program", "valid", False),
    ("program", "validation_error", "resource validation failed"),
    ("step", "valid", False),
    ("step", "validation_error", "primitive validation failed"),
])
def test_supplied_validation_failure_cannot_be_overridden_by_safe_geometry(
    level: str, field: str, value: str | bool,
) -> None:
    """Known invalid resource evidence remains unresolved at either trace level."""
    case = _entry_case(end=0.4)
    row = case["programs"][0]
    if level == "step":
        row = row["step_results"][0]
    row[field] = value
    _unavailable(_check(case))


@pytest.mark.parametrize("snapshot_field", ["start_snapshot", "projected_snapshot"])
@pytest.mark.parametrize("fact,value", [("current_pose", _pose(4.0)), ("held_part", None)])
def test_step_snapshots_cannot_contradict_modeled_pose_or_custody(
    snapshot_field: str, fact: str, value: list[float] | None,
) -> None:
    """Physical evidence must agree across primitive trajectories and snapshots."""
    case = _entry_case(end=0.4)
    case["programs"][0]["step_results"][0][snapshot_field] = {fact: value}
    _unavailable(_check(case))


def test_consistent_trace_snapshots_and_unfamiliar_state_names_remain_supported() -> None:
    """Physical facts determine checking while state identifiers retain their spelling."""
    case = _entry_case(end=0.4)
    program = case["programs"][0]
    program["valid"] = True
    step = program["step_results"][0]
    step["valid"] = True
    step["start_snapshot"] = {
        "current_pose": _pose(0.8), "held_part": "p", "current_state": "llm_state_17",
    }
    step["projected_snapshot"] = {
        "current_pose": _pose(0.4), "held_part": "p", "current_state": "safe_to_load",
    }
    original = deepcopy(case)
    assert _check(case)["is_safe"] is True
    assert case == original


def test_release_step_snapshots_establish_custody_change_without_movement() -> None:
    """Consistent release snapshots confirm clearing custody at the same pose."""
    case = _entry_case(resource="xarm6", start=1.1, occupied=False)
    step = _release(0.0, 1.0, 1.1, "M2")
    step["start_snapshot"] = {"current_pose": _pose(1.1), "held_part": "p"}
    step["projected_snapshot"] = {"current_pose": _pose(1.1), "held_part": None}
    case["programs"] = [_program("xarm6", step)]
    result = _check(case)
    assert result["is_safe"] is True, result
    assert result["projected_snapshot"]["resources"]["xarm6"]["held_part"] is None


_REVIEWED_FIXTURE = (
    Path(__file__).parent / "fixtures" / "KMR_assembly_board-v1_recovery" / "storage_interruption"
)


def _reviewed_case() -> dict:
    return migrate_ppr_fixture(json.loads((_REVIEWED_FIXTURE / "safety_evidence.json").read_text())["inputs"])


def _reviewed_check(case: dict, **kwargs) -> dict:
    from cais_spade_llm.agents.central_controller.primitive_program_safety import (
        validate_reviewed_primitive_program_safety,
    )

    return validate_reviewed_primitive_program_safety(**case, **kwargs)


def _reviewed_formula(case: dict, formula: str) -> None:
    next(row for row in case["catalog"]["specifications"]
         if row["id"] == "KET4_Square_4mm_trim_precedence")["formula"] = formula


def _reviewed_unavailable(result: dict) -> None:
    assert result["status"] == "unavailable", result
    assert result["is_safe"] is False
    assert result["feasibility_status"] == "NEEDS_CONTEXT"
    assert result["safety_dfa_states_after"] == result["safety_dfa_states_before"]
    assert result["candidate_continuation"] is None


def _reviewed_ledger(case: dict, records: list[dict]) -> None:
    """Keep every supplied KET4 ledger consistent in negative variants."""
    def replace(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "KET4_Square_4mm" and isinstance(child, dict) and "processCompleted" in child:
                    child["processCompleted"] = deepcopy(records)
                replace(child)
        elif isinstance(value, list):
            for child in value:
                replace(child)
    replace(case)


def test_reviewed_storage_interruption_preserves_programs_and_nominal_continuation() -> None:
    case = _reviewed_case()
    original = deepcopy(case)
    result = _reviewed_check(case, trace_complete=True)
    assert result["status"] == "satisfied", result
    assert result["pending_rule_ids"] == []
    assert case == original
    assert len(result["bindings"]) == 11
    assert sum(binding["specification"] == "shared_area_mutex"
               for binding in result["bindings"]) == 10
    records = json.loads((_REVIEWED_FIXTURE / "primitive_programs.json").read_text())
    initial = json.loads((_REVIEWED_FIXTURE / "initial_context.json").read_text())
    steps = case["programs"][0]["primitive_steps"]
    results = case["programs"][0]["step_results"]
    assert len(steps) == len(results) == 22
    flattened = [(event, index, step) for event in records["primitive_program"]
                 for index, step in enumerate(event["primitive_steps"])]
    for global_index, (event, local_index, step) in enumerate(flattened):
        assert steps[global_index]["primitive"] == step["primitive"]
        assert steps[global_index]["params"] == step["params"]
        source = results[global_index]["source"]
        for key in ("outline_id", "des_event_id", "event_name", "resource_jid"):
            assert source[key] == event[key]
        assert source["step_index"] == local_index
        assert results[global_index]["step_index"] == global_index
    final = result["projected_snapshot"]
    assert final["resources"]["KMR"]["held_part"] == "KET8_Square_8mm"
    assert final["resources"]["KMR"]["current_pose"] == pytest.approx(
        initial["required_resume_state"]["current_pose"])
    assert final["resources"]["KMR"]["base_pose"] == pytest.approx(
        initial["required_resume_state"]["base_pose"])
    assert final["parts"]["KET8_Square_8mm"]["contained_by"] is None
    assert final["parts"]["KET4_Square_4mm"]["contained_by"] == "assembly_board-v1"
    assert final["parts"]["KET4_Square_4mm"]["processCompleted"] == [
        {"process": "trim", "result": "square"}]
    assert final["parts"]["KET8_Square_8mm"]["processCompleted"] == []
    for key in ("nominal_tasks", "task_statuses", "resume_entry_task_ids_by_resource",
                "resumable_task_ids_by_resource"):
        assert final[key] == initial[key]
    for row in result["observations"]:
        assert row["parts"]["KET4_Square_4mm"]["processCompleted"] == [
            {"process": "trim", "result": "square"}]
        if 3 <= row["time"] < 20:
            assert row["parts"]["KET8_Square_8mm"]["contained_by"] == "Storage"
        if row["time"] >= 13:
            assert row["parts"]["KET4_Square_4mm"]["contained_by"] == "assembly_board-v1"
    assert final["task_statuses"]["KMR_STORAGE_KET8_MOVE_TO_M1"] == "pending"


@pytest.mark.parametrize("records", [[], [{"process": "trim", "result": "circle"}],
                                    [{"process": "assembly", "result": "square"}]])
def test_reviewed_precedence_requires_exact_completed_process(records) -> None:
    case = _reviewed_case()
    _reviewed_ledger(case, records)
    result = _reviewed_check(case, trace_complete=True)
    assert result["status"] == "violated", result
    assert result["candidate_continuation"] is None
    assert result["safety_dfa_states_after"] == result["safety_dfa_states_before"]


@pytest.mark.parametrize("change", ["missing_ledger", "incomplete", "missing_provenance"])
def test_reviewed_missing_completion_evidence_is_not_a_negative_ledger(change) -> None:
    case = _reviewed_case()
    part = case["snapshot"]["parts"]["KET4_Square_4mm"]
    if change == "missing_ledger":
        part.pop("processCompleted")
    elif change == "incomplete":
        part["processCompleted_evidence"]["complete"] = False
        part["processCompleted_complete"] = False
    else:
        part.pop("processCompleted_evidence")
    _reviewed_unavailable(_reviewed_check(case, trace_complete=True))


@pytest.mark.parametrize("resources", [("ur5e-3",), ("ur5e-3", "ur5e-4")])
def test_reviewed_mutex_includes_conflicts_independent_of_kmr(resources) -> None:
    case = _reviewed_case()
    for resource in resources:
        case["snapshot"]["resources"][resource]["current_pose"][:3] = [0, 0, 1.04]
    result = _reviewed_check(case, trace_complete=True)
    assert result["status"] == "violated", result
    if len(resources) == 2:
        assert result["counterexample"]["time"] == 0


@pytest.mark.parametrize("formula", ["G (ap001 -> ap002)", "F ap001", "ap002 U ap001", "X ap002"])
def test_reviewed_general_formulas_accept_the_actual_finite_trace(formula) -> None:
    case = _reviewed_case()
    _reviewed_formula(case, formula)
    result = _reviewed_check(case, trace_complete=True)
    assert result["status"] == "satisfied", result
    assert result["pending_rule_ids"] == []


def test_reviewed_eventually_pending_and_contiguous_slices_match_full_trace() -> None:
    case = _reviewed_case()
    _reviewed_formula(case, "F ap001")
    full = _reviewed_check(case, trace_complete=True)
    first = _reviewed_check(case, observation_slice=[0, 2])
    assert first["status"] == "prefix_checked", first
    assert len(first["pending_rule_ids"]) == 1
    assert first["is_safe"] is False  # A pending prefix is not full satisfaction.
    second = _reviewed_check(case, observation_slice=[2, full["trace_length"]],
                             continuation=first["candidate_continuation"], trace_complete=True)
    assert second["status"] == "satisfied", second
    assert second["safety_dfa_states_after"] == full["safety_dfa_states_after"]
    assert first["observations"] + second["observations"] == full["observations"]
    assert second["trace_id"] == full["trace_id"]


def test_reviewed_pending_requirement_rejects_at_completion_without_empty_tick() -> None:
    case = _reviewed_case()
    _reviewed_formula(case, "F (!ap002)")
    prefix = _reviewed_check(case)
    assert prefix["status"] == "prefix_checked", prefix
    assert len(prefix["pending_rule_ids"]) == 1
    continuation = deepcopy(prefix["candidate_continuation"])
    final = _reviewed_check(case, continuation=continuation, trace_complete=True,
                            observation_slice=[prefix["trace_length"], prefix["trace_length"]])
    assert final["status"] == "violated", final
    assert final["observations"] == []
    assert final["continuation"] == continuation
    assert final["candidate_continuation"] is None
    assert prefix["observations"][-1]["time"] == 22


@pytest.mark.parametrize("change", ["gap", "duplicate", "formula", "geometry", "state", "early_completion"])
def test_reviewed_continuation_rejects_changed_or_noncontiguous_history(change) -> None:
    case = _reviewed_case()
    first = _reviewed_check(case, observation_slice=[0, 2])
    continuation = deepcopy(first["candidate_continuation"])
    kwargs = {"continuation": continuation, "observation_slice": [2, 4]}
    if change == "gap":
        kwargs["observation_slice"] = [3, 4]
    elif change == "duplicate":
        kwargs["observation_slice"] = [1, 4]
    elif change == "formula":
        _reviewed_formula(case, "F ap001")
    elif change == "geometry":
        case["geometry"]["regions"]["assembly_board-v1"]["bounds"][0][0] -= 0.01
    elif change == "state":
        continuation["states"] = {"made_up": "9999"}
    else:
        kwargs["trace_complete"] = True
    original = deepcopy(continuation)
    result = _reviewed_check(case, **kwargs)
    _reviewed_unavailable(result)
    assert result["continuation"] == original
    assert continuation == original


def _reviewed_short_case() -> dict:
    case = _reviewed_case()
    case["snapshot"] = {field: case["snapshot"][field] for field in ("resources", "parts")}
    case["horizon"] = [0, 1]
    case["programs"] = []
    case["stationary"] = {resource: [[0, 1]] for resource in case["snapshot"]["resources"]}
    kmr = case["snapshot"]["resources"]["KMR"]
    kmr.update(current_pose=[1.1, 0, 1, 0, 0, 0, 1], held_part="KET4_Square_4mm",
               gripper_state="closed", grasp_transform=[-0.3, 0, 0, 0, 0, 0, 1])
    kmr.pop("base_pose")
    case["geometry"]["resources"]["KMR"] = {
        "frame": "world", "footprint": [[-0.2, 0.2], [-0.01, 0.01], [-0.01, 0.01]]}
    case["snapshot"]["parts"].pop("KET8_Square_8mm")
    part = case["snapshot"]["parts"]["KET4_Square_4mm"]
    part.update(current_pose=[0.8, 0, 1, 0, 0, 0, 1], contained_by=None)
    case["geometry"]["parts"].pop("KET8_Square_8mm")
    case["geometry"]["parts"]["KET4_Square_4mm"]["footprint"] = [
        [-0.05, 0.05], [-0.01, 0.01], [-0.01, 0.01]]
    case["geometry"]["regions"]["assembly_board-v1"]["bounds"] = [
        [1, 1.2], [-0.1, 0.1], [0.9, 1.1]]
    return case


def _reviewed_move(resource: str, points: list[tuple[float, list[float]]]) -> dict:
    return _program(resource, {
        "primitive": "move_cartesian", "resolved_params": {"target": points[-1][1]},
        "start_time": points[0][0], "end_time": points[-1][0],
        "model_evidence": {"frame": "world", "trajectory": [
            {"time": time, "pose": pose} for time, pose in points]},
    })


def test_reviewed_part_entry_is_detected_when_tool_was_already_inside() -> None:
    case = _reviewed_short_case()
    _reviewed_ledger(case, [])
    case["stationary"].pop("KMR")
    case["programs"] = [_reviewed_move("KMR", [
        (0, [1.1, 0, 1, 0, 0, 0, 1]), (1, [1.3, 0, 1, 0, 0, 0, 1])])]
    result = _reviewed_check(case, trace_complete=True)
    assert result["status"] == "violated", result
    assert result["counterexample"]["time"] == 0.75
    assert result["counterexample"]["ap_values"] == {"ap001": True, "ap002": False}
    assert result["observations"][0]["region_occupancy"]["assembly_board-v1"]["KMR"] is True
    assert result["observations"][0]["part_region_occupancy"]["assembly_board-v1"]["KET4_Square_4mm"] is False


def test_reviewed_initial_occupancy_has_no_invented_entry_or_earlier_history() -> None:
    case = _reviewed_short_case()
    _reviewed_ledger(case, [])
    case["geometry"]["regions"]["assembly_board-v1"]["bounds"][0] = [0.7, 1.2]
    result = _reviewed_check(case, trace_complete=True)
    assert result["status"] == "satisfied", result
    checks = [row for row in result["rule_checks"] if row["specification"] == "KET4_Square_4mm_trim_precedence"]
    assert all(row["ap_values"]["ap001"] is False for row in checks)


def test_reviewed_custody_boundaries_do_not_create_part_entry() -> None:
    result = _reviewed_check(_reviewed_case(), trace_complete=True)
    checks = [row for row in result["rule_checks"] if row["specification"] == "KET4_Square_4mm_trim_precedence"]
    assert any(row["ap_values"]["ap001"] for row in checks)
    for row in checks:
        if row["time"] in {3, 9, 13, 20}:
            assert row["ap_values"]["ap001"] is False
    release = next(row for row in result["observations"] if row["time"] == 13)
    assert release["resources"]["KMR"]["held_part"] is None
    assert release["region_occupancy"]["assembly_board-v1"]["KMR"] is True
    assert release["part_region_occupancy"]["assembly_board-v1"]["KET4_Square_4mm"] is True
    final = result["observations"][-1]
    assert final["region_occupancy"]["assembly_board-v1"]["KMR"] is False
    assert final["part_region_occupancy"]["assembly_board-v1"]["KET4_Square_4mm"] is True


@pytest.mark.parametrize("start,end", [(11.5, 12.5), (13.01, 13.2)])
def test_reviewed_concurrent_internal_motion_and_retreat_conflict(start, end) -> None:
    case = _reviewed_case()
    resource = "ur5e-3"
    pose = deepcopy(case["snapshot"]["resources"][resource]["current_pose"])
    inside = [0, 0, 1.04, *pose[3:]]
    case["programs"].append(_reviewed_move(resource, [
        (start, pose), ((start + end) / 2, inside), (end, pose)]))
    case["stationary"][resource] = [[0, start], [end, 22]]
    result = _reviewed_check(case, trace_complete=True)
    assert result["status"] == "violated", result
    assert start < result["counterexample"]["time"] < end
    assert result["observations"][0]["region_occupancy"]["assembly_board-v1"][resource] is False
    assert result["observations"][-1]["region_occupancy"]["assembly_board-v1"][resource] is False


def test_reviewed_simultaneous_exit_entry_contact_is_joint() -> None:
    case = _reviewed_short_case()
    region = case["geometry"]["regions"]["assembly_board-v1"]
    region["bounds"] = [[10, 11], [-0.1, 0.1], [0.9, 1.1]]
    for resource, start, end in (("ur5e-3", 10.5, 11.7), ("ur5e-4", 9.3, 10.5)):
        pose = [start, 0, 1, 0, 0, 0, 1]
        case["snapshot"]["resources"][resource]["current_pose"] = pose
        case["stationary"].pop(resource)
        case["programs"].append(_reviewed_move(resource, [(0, pose), (1, [end, 0, 1, 0, 0, 0, 1])]))
    result = _reviewed_check(case, trace_complete=True)
    assert result["status"] == "violated", result
    assert result["counterexample"]["time"] == 0.5
    assert result["counterexample"]["ap_values"] == {"ap001": True, "ap002": True}


@pytest.mark.parametrize("change", [
    "participant_snapshot", "participant_geometry", "participant_population", "omitted_participant",
    "stationary", "applicable_rule", "catalog_rule", "unknown_predicate", "changed_meaning",
    "undefined_ap", "invalid_formula",
])
def test_reviewed_incomplete_scope_or_unsupported_formulas_cannot_pass(change) -> None:
    case = _reviewed_case()
    if change == "participant_snapshot":
        case["snapshot"]["resources"].pop("ur5e-4")
    elif change == "participant_geometry":
        case["geometry"]["resources"].pop("ur5e-4")
    elif change == "participant_population":
        case["applicability"]["regions"]["assembly_board-v1"]["resources"].remove("ur5e-4")
    elif change == "omitted_participant":
        case["applicability"]["scene_resources"] = [row for row in case["applicability"]["scene_resources"]
                                                    if row["resource_id"] != "ur5e-4"]
        case["applicability"]["regions"]["assembly_board-v1"]["resources"].remove("ur5e-4")
    elif change == "stationary":
        case["stationary"].pop("ur5e-2")
    elif change == "applicable_rule":
        case["applicability"]["rules"].pop()
    elif change == "catalog_rule":
        case["catalog"]["specifications"].pop()
    elif change == "unknown_predicate":
        case["catalog"]["specifications"][1]["aps"][0]["full"] = "ap_event/unsupported"
    elif change == "changed_meaning":
        case["catalog"]["specifications"][1]["aps"][0]["meaning"] = "A generated name is entry"
    elif change == "undefined_ap":
        _reviewed_formula(case, "G ap003")
    elif change == "invalid_formula":
        _reviewed_formula(case, "G(")
    _reviewed_unavailable(_reviewed_check(case, trace_complete=True))


@pytest.mark.parametrize("change", [
    "trajectory", "helper_output", "helper_binding", "grasp_transform", "release_transform",
    "part_stationary", "orientation", "base_trajectory", "unknown_primitive",
    "contradictory_custody", "ledger_record", "invented_completion",
])
def test_reviewed_incomplete_or_contradictory_primitive_evidence_cannot_pass(change) -> None:
    case = _reviewed_case()
    steps = case["programs"][0]["step_results"]
    if change == "trajectory":
        steps[10]["model_evidence"].pop("trajectory")
    elif change == "helper_output":
        steps[16]["model_evidence"]["outputs"].pop("target")
    elif change == "helper_binding":
        steps[16]["model_evidence"]["outputs"]["target"][0] += 0.01
    elif change == "grasp_transform":
        case["snapshot"]["resources"]["KMR"]["grasp_transform"][2] += 0.01
    elif change == "release_transform":
        steps[12]["resolved_params"]["transform"][2] += 0.01
        case["programs"][0]["primitive_steps"][12]["params"]["transform"][2] += 0.01
    elif change == "part_stationary":
        steps[2]["model_evidence"]["released_part"]["stationary_until"] = 19
    elif change == "orientation":
        steps[0]["model_evidence"]["trajectory"][1]["pose"][3:] = [0, 0, 0, 1]
    elif change == "base_trajectory":
        steps[5]["model_evidence"].pop("base_trajectory")
    elif change == "unknown_primitive":
        steps[0]["primitive"] = "unknown_motion"
        case["programs"][0]["primitive_steps"][0]["primitive"] = "unknown_motion"
    elif change == "contradictory_custody":
        steps[0]["projected_snapshot"]["held_part"] = None
    elif change == "ledger_record":
        _reviewed_ledger(case, [{}])
    else:
        steps[12]["projected_snapshot"]["processCompleted"] = [{"process": "assembly", "result": "square"}]
    _reviewed_unavailable(_reviewed_check(case, trace_complete=True))


def test_reviewed_missing_compiler_fails_even_after_cached_success(monkeypatch) -> None:
    from cais_spade_llm.agents.central_controller import (
        reviewed_primitive_program_safety as reviewed,
    )

    case = _reviewed_case()
    assert _reviewed_check(case, trace_complete=True)["status"] == "satisfied"
    monkeypatch.setattr(reviewed.shutil, "which", lambda _: None)
    _reviewed_unavailable(_reviewed_check(case, trace_complete=True))


@pytest.mark.parametrize("formula,expected", [("true", "satisfied"), ("false", "violated"),
                                              ("ap001 & !ap001", "violated")])
def test_reviewed_constant_and_unsatisfiable_formulas_are_not_mutex_substitutes(formula, expected) -> None:
    case = _reviewed_case()
    _reviewed_formula(case, formula)
    assert _reviewed_check(case, trace_complete=True)["status"] == expected


def test_reviewed_x_requires_next_declared_observation_not_next_primitive() -> None:
    case = _reviewed_case()
    _reviewed_formula(case, "X ap002")
    first = _reviewed_check(case, observation_slice=[0, 1])
    assert first["status"] == "prefix_checked"
    assert first["pending_rule_ids"]
    second = _reviewed_check(case, observation_slice=[1, 2], continuation=first["candidate_continuation"])
    assert second["pending_rule_ids"] == []
    assert second["observations"][0]["time"] < 1  # Still within the first primitive.


def test_reviewed_all_primitive_endpoints_agree_with_original_expectations() -> None:
    case = _reviewed_case()
    result = _reviewed_check(case, trace_complete=True)
    assert result["status"] == "satisfied", result
    originals = json.loads((_REVIEWED_FIXTURE / "primitive_programs.json").read_text())
    for record in case["programs"][0]["step_results"]:
        source = record["source"]
        expected = originals["step_expectations"][source["outline_id"]][source["step_index"]]
        observed = next(row for row in result["observations"] if row["time"] == record["end_time"])
        resource = observed["resources"]["KMR"]
        for field in ("current_pose", "base_pose"):
            assert resource[field] == pytest.approx(expected["projected_snapshot"][field])
        for field in ("held_part", "gripper_state"):
            assert resource[field] == expected["projected_snapshot"][field]
        for part, tracker in expected["part_tracker"].items():
            assert observed["parts"][part]["current_pose"] == pytest.approx(tracker["pose"])
            assert observed["parts"][part]["processCompleted"] == tracker["processCompleted"]
        storage_inventory = observed["resources"]["Storage"]["contained_parts"]
        for part in ("KET8_Square_8mm", "KET4_Square_4mm"):
            assert (part in storage_inventory) == expected["Storage"]["inventory." + part]


def test_reviewed_boundary_entry_survives_slice_and_failed_candidate_retains_history() -> None:
    case = _reviewed_case()
    full = _reviewed_check(case, trace_complete=True)
    entry = next(row["observation_index"] for row in full["rule_checks"]
                 if row["specification"] == "KET4_Square_4mm_trim_precedence" and row["ap_values"]["ap001"])
    before = _reviewed_check(case, observation_slice=[0, entry])
    at = _reviewed_check(case, continuation=before["candidate_continuation"],
                         observation_slice=[entry, entry + 1])
    assert next(row for row in at["rule_checks"] if row["specification"] == "KET4_Square_4mm_trim_precedence")["ap_values"]["ap001"] is True
    _reviewed_ledger(case, [])
    before = _reviewed_check(case, observation_slice=[0, entry])
    assert before["status"] == "prefix_checked"
    original = deepcopy(before["candidate_continuation"])
    rejected = _reviewed_check(case, continuation=original, observation_slice=[entry, entry + 1])
    assert rejected["status"] == "violated"
    assert rejected["continuation"] == original
    assert rejected["safety_dfa_states_after"] == original["states"]


@pytest.mark.parametrize("change", ["unknown_parameter", "seed_shape", "nonunit_quaternion", "named_output", "custody_motion"])
def test_reviewed_primitive_parameter_and_effect_contracts_are_required(change) -> None:
    case = _reviewed_case()
    program = case["programs"][0]
    if change in {"unknown_parameter", "seed_shape"}:
        field, value = ("typo_target", 1) if change == "unknown_parameter" else ("seed", [0])
        program["primitive_steps"][0]["params"][field] = value
        program["step_results"][0]["resolved_params"][field] = value
    elif change == "nonunit_quaternion":
        case["snapshot"]["resources"]["KMR"]["current_pose"][3] = 2
    elif change == "named_output":
        program["step_results"][4]["model_evidence"]["outputs"]["pose_name"] = "different_pose"
    else:
        program["step_results"][2]["model_evidence"]["trajectory"] = [
            {"time": 2, "pose": [0, 0, 0, 0, 0, 0, 1]},
            {"time": 3, "pose": [1, 0, 0, 0, 0, 0, 1]},
        ]
    _reviewed_unavailable(_reviewed_check(case, trace_complete=True))


def test_reviewed_motion_observations_are_independent_of_formulas() -> None:
    case = _reviewed_case()
    first = _reviewed_check(case, trace_complete=True)
    _reviewed_formula(case, "F ap001")
    second = _reviewed_check(case, trace_complete=True)
    assert first["observations"] == second["observations"]
    assert first["trace_id"] != second["trace_id"]


@pytest.mark.parametrize("pose_name", ["home", "unsupported_name"])
def test_reviewed_unmodeled_named_pose_cannot_bypass_custody_effects(pose_name) -> None:
    case = _reviewed_case()
    program = case["programs"][0]
    program["primitive_steps"][21]["params"]["pose_name"] = pose_name
    program["step_results"][21]["resolved_params"]["pose_name"] = pose_name
    program["step_results"][21]["model_evidence"]["outputs"]["pose_name"] = pose_name
    _reviewed_unavailable(_reviewed_check(case, trace_complete=True))


@pytest.mark.parametrize(("formula", "expected"), [
    ("ap001", lambda word: word[0][0]),
    ("ap002", lambda word: word[0][1]),
    ("!ap001", lambda word: not word[0][0]),
    ("X ap001", lambda word: len(word) > 1 and word[1][0]),
    ("F ap001", lambda word: any(row[0] for row in word)),
    ("G ap001", lambda word: all(row[0] for row in word)),
    ("ap001 U ap002", lambda word: any(
        row[1] and all(prior[0] for prior in word[:index])
        for index, row in enumerate(word))),
    ("ap001 R ap002", lambda word: all(
        row[1] or any(prior[0] for prior in word[:index])
        for index, row in enumerate(word))),
    ("X (F ap001)", lambda word: len(word) > 1 and any(row[0] for row in word[1:])),
    ("F (X ap001)", lambda word: any(row[0] for row in word[1:])),
    ("G (X ap001)", lambda word: False),
    ("X (G ap001)", lambda word: len(word) > 1 and all(row[0] for row in word[1:])),
    ("G (ap001 -> F ap002)", lambda word: all(
        not row[0] or any(future[1] for future in word[index:])
        for index, row in enumerate(word))),
    ("(!ap001) U (X ap002)", lambda word: any(
        word[index + 1][1] and all(not prior[0] for prior in word[:index])
        for index in range(len(word) - 1))),
    ("!(X ap001)", lambda word: not (len(word) > 1 and word[1][0])),
    ("X (!ap001)", lambda word: len(word) > 1 and not word[1][0]),
])
def test_reviewed_compiled_formulas_match_finite_truth_tables(formula, expected) -> None:
    """Compare compilation with direct finite-word semantics, independent of AP grounding."""
    from itertools import product

    from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
        _compile_formula,
    )

    labels = ("ap001", "ap002")
    checker = BaseSafetyChecker({"reviewed": _compile_formula(formula, set(labels))}, [])
    dfa = checker.dfas["reviewed"]
    for length in range(1, 5):
        for word in product(product((False, True), repeat=2), repeat=length):
            state = dfa["initial"]
            reachable = True
            for row in word:
                transition = checker.transition_evidence(
                    "reviewed", state,
                    frozenset(label for label, active in zip(labels, row, strict=True) if active),
                )
                if transition["status"] != "passed":
                    assert transition["reason"] == "accepting_state_unreachable", transition
                    reachable = False
                    break
                state = transition["to"]
            accepted = reachable and state in dfa["accepting_states"]
            assert accepted == expected(word), (formula, word, state)


@pytest.mark.parametrize("step_index,field,value", [
    (0, "tcp_pose", [0, 0, 0, 0, 0, 0, 1]),
    (5, "base_pose", [0, 0, 0]),
    (2, "held_part", "KET8_Square_8mm"),
    (8, "attached", False),
    (0, "success", False),
])
def test_reviewed_reported_primitive_outputs_must_agree_with_effects(step_index, field, value) -> None:
    case = _reviewed_case()
    case["programs"][0]["step_results"][step_index]["model_evidence"]["outputs"][field] = value
    _reviewed_unavailable(_reviewed_check(case, trace_complete=True))


@pytest.mark.parametrize("returncode,stdout,stderr", [(0, "incomplete DFA", ""), (1, "", "MONA failed")])
def test_reviewed_compilation_failure_never_omits_a_rule(monkeypatch, returncode, stdout, stderr) -> None:
    from types import SimpleNamespace

    from cais_spade_llm.agents.central_controller import (
        reviewed_primitive_program_safety as reviewed,
    )

    reviewed._compile_formula_cached.cache_clear()
    monkeypatch.setattr(reviewed.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=returncode, stdout=stdout, stderr=stderr))
    result = _reviewed_check(_reviewed_case(), trace_complete=True)
    _reviewed_unavailable(result)
    assert result["bindings"] == []


def _grounded_case() -> dict:
    return migrate_ppr_fixture(json.loads((_REVIEWED_FIXTURE / "automatic_grounding_evidence.json").read_text())["inputs"])


def _grounded_check(case: dict, **kwargs) -> dict:
    from cais_spade_llm.agents.central_controller.offline_safety_grounding import (
        validate_grounded_primitive_program_safety,
    )

    return validate_grounded_primitive_program_safety(**case, **kwargs)


def _gear_precedence_case(*, completion_time: float | None = None) -> dict:
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
        _ENTRY,
        _MEANINGS,
        _TARGET_COMPLETED,
    )
    from cais_spade_llm.resources.environment_models import build_environment_models

    case, short = _grounded_case(), _reviewed_short_case()
    for field in ("snapshot", "geometry"):
        case[field]["parts"] = deepcopy(short[field]["parts"])
        case[field]["resources"]["KMR"] = deepcopy(short[field]["resources"]["KMR"])
    case["geometry"]["regions"] = deepcopy(short["geometry"]["regions"])
    case["horizon"] = [0, 1]
    case["stationary"] = {rid: [[0, 1]] for rid in case["snapshot"]["resources"]}
    case["stationary"]["KMR"] = []
    first = case["snapshot"]["resources"]["KMR"]["current_pose"]
    last = [1.5, *first[1:]]
    case["programs"] = [_reviewed_move("KMR", [(0, first), (1, last)])]
    target = "Gear_Plate/Gear_Shaft_1"
    case["snapshot"]["parts"]["gear_small"] = {
        "current_pose": [1.1, 0, 1, 0, 0, 0, 1], "contained_by": None, "stationary_until": 1,
        "processCompleted": [{"process": "print_part"}], "processCompleted_complete": True,
        "processCompleted_evidence": {"complete": True, "source_kind": "synthetic",
                                      "checkpoint": "before_gear_assembly_acknowledgement"},
    }
    case["geometry"]["parts"]["gear_small"] = {
        "frame": "world", "footprint": [[-.01, .01]] * 3, "target": target,
    }
    case["catalog"] = {"version": 2, "specifications": [{
        "id": "gear_small_before_KET4_Square_4mm",
        "requirement": "gear_small assembly completes before KET4_Square_4mm entry.",
        "formula": "(!ap001 U (ap002 & !ap001)) | G !ap001",
        "aps": [{"label": label, "full": full, "meaning": _MEANINGS[full]}
                for label, full in (("ap001", _ENTRY), ("ap002", _TARGET_COMPLETED))],
    }]}
    case["requirement_scopes"] = [{
        "specification": "gear_small_before_KET4_Square_4mm",
        "physical_ap_bindings": {
            "ap001": {"part": "KET4_Square_4mm", "region": "assembly_board-v1"},
            "ap002": {"part": "gear_small", "process": "assembly", "target": target},
        },
    }]
    if completion_time is not None:
        model = build_environment_models(case["scene"])["ur5e-4"]
        event = next(row for row in model["events"] if row["event_name"] == "place_insert")
        params = {key: row["equals"] for key, row in event["parameter_bindings"].items()
                  if "equals" in row}
        params.update(part_name="gear_small", target=target)
        case["task_evidence"] = {
            "complete": True, "source_kind": "synthetic", "horizon": [0, 1],
            "events": [{"task_id": "gear_assembly", "resource_id": "ur5e-4", "process": "assembly",
                        "product": "gear_small", "function": "place_insert",
                        "context": "destination=assembly_board-v1", "start_time": 0,
                        "end_time": completion_time}],
        }
        case["product_effect_evidence"] = {
            "complete": True, "source_kind": "synthetic", "horizon": [0, 1],
            "updates": [{"task_id": "gear_assembly", "resource_id": "ur5e-4",
                         "time": completion_time, "kind": "acknowledged",
                         "declaration": {"event_id": event["event_id"], "event_name": "place_insert",
                                         "parameters": params},
                         "product_effects": {"gear_small": {"state": "assembled", "target": target,
                             "processCompleted": [{"process": "assembly", "target": target}]}}}],
        }
    return case


@pytest.mark.parametrize("completion_time,expected", [(None, "violated"), (.25, "satisfied"),
                                                      (.375, "violated"), (.5, "violated")])
def test_distinct_part_assembly_precedence_requires_strict_earlier_completion(completion_time, expected) -> None:
    case = _gear_precedence_case(completion_time=completion_time)
    before = deepcopy(case)
    result = _grounded_check(case, trace_complete=True)
    assert result["status"] == expected, result["reason"]
    assert case == before
    if completion_time is not None:
        at_completion = next(row for row in result["observations"]
                             if float(row["time"]) == completion_time and row["phase"] == "at")
        ledger = at_completion["parts"]["gear_small"]["processCompleted"]
        assert ledger == [{"process": "print_part"}, {"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"}]
        assert at_completion["parts"]["KET4_Square_4mm"]["processCompleted"] == before["snapshot"]["parts"]["KET4_Square_4mm"]["processCompleted"]


@pytest.mark.parametrize("change", ["missing_ledger", "incomplete_ledger", "wrong_target", "wrong_part",
                                   "wrong_owner", "wrong_time", "wrong_event", "wrong_effect",
                                   "duplicate", "predicted", "missing_task", "unassociated_generated"])
def test_assembly_effect_evidence_cannot_invent_completion(change: str) -> None:
    case = _gear_precedence_case(completion_time=.25)
    update = case["product_effect_evidence"]["updates"][0]
    if change == "missing_ledger":
        case["snapshot"]["parts"]["gear_small"].pop("processCompleted_evidence")
    elif change == "incomplete_ledger":
        case["snapshot"]["parts"]["gear_small"]["processCompleted_complete"] = False
    elif change == "wrong_target":
        update["declaration"]["parameters"]["target"] = "Gear_Plate/Gear_Shaft_3"
    elif change == "wrong_part":
        update["declaration"]["parameters"]["part_name"] = "KET4_Square_4mm"
    elif change == "wrong_owner":
        update["resource_id"] = "ur5e-3"
    elif change == "wrong_time":
        update["time"] = .125
    elif change == "wrong_event":
        update["declaration"]["event_name"] = "release_part"
    elif change == "wrong_effect":
        update["product_effects"]["gear_small"]["processCompleted"][0] = {"process": "assembly", "result": "assembled"}
    elif change == "duplicate":
        case["product_effect_evidence"]["updates"].append(deepcopy(update))
    elif change == "predicted":
        update["kind"] = "predicted"
    elif change == "missing_task":
        case.pop("task_evidence")
    else:
        case["task_evidence"]["events"][0]["function"] = "generated_assembly_17"
    _reviewed_unavailable(_grounded_check(case, trace_complete=True))


def test_assembly_effect_explicit_generated_task_association_preserves_names() -> None:
    case = _gear_precedence_case(completion_time=.25)
    task = case["task_evidence"]["events"][0]
    task["function"] = "generated_assembly_17"
    task["declared_task"] = deepcopy(case["product_effect_evidence"]["updates"][0]["declaration"])
    result = _grounded_check(case, trace_complete=True)
    assert result["status"] == "satisfied", result["reason"]
    assert result["evidence"]["product_effect_evidence"][0]["kind"] == "acknowledged"


def test_initial_assembly_completion_and_initial_occupancy_create_no_entry() -> None:
    case = _gear_precedence_case()
    case["snapshot"]["parts"]["gear_small"]["processCompleted"].append(
        {"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"})
    assert _grounded_check(case, trace_complete=True)["status"] == "satisfied"
    case = _gear_precedence_case()
    case["programs"] = []
    case["stationary"]["KMR"] = [[0, 1]]
    case["snapshot"]["resources"]["KMR"]["current_pose"][0] = 1.4
    case["snapshot"]["parts"]["KET4_Square_4mm"]["current_pose"][0] = 1.1
    result = _grounded_check(case, trace_complete=True)
    assert result["status"] == "satisfied", result["reason"]
    assert all(row["value"] is False for row in result["ap_evidence"] if row["descriptor"]["label"] == "ap001")


def test_assembly_product_effect_clock_continuation_and_evidence_identity() -> None:
    case = _gear_precedence_case(completion_time=.25)
    full = _grounded_check(case, trace_complete=True)
    assert full["status"] == "satisfied", full["reason"]
    cut = next(index for index, row in enumerate(full["observations"]) if row["time"] == .25)
    prefix = _grounded_check(case, observation_slice=[0, cut])
    continuation = deepcopy(prefix["continuation"])
    suffix = _grounded_check(case, continuation=continuation, trace_complete=True)
    assert suffix["status"] == "satisfied", suffix["reason"]
    assert prefix["observations"] + suffix["observations"] == full["observations"]
    assert suffix["safety_dfa_states_after"] == full["safety_dfa_states_after"]
    assert prefix["projected_snapshot"]["parts"]["gear_small"]["processCompleted"] == [{"process": "print_part"}]
    case["product_effect_evidence"]["source_kind"] = "different_acknowledgement_source"
    rejected = _grounded_check(case, continuation=continuation, trace_complete=True)
    _reviewed_unavailable(rejected)
    assert rejected["continuation"] == continuation


def test_predicted_assembly_effects_remain_explicit_in_composition_preparation() -> None:
    from cais_spade_llm.agents.central_controller.offline_safety_grounding import (
        _prepare_grounded_primitive_trace,
    )

    case = _gear_precedence_case(completion_time=.25)
    case["product_effect_evidence"]["updates"][0]["kind"] = "predicted"
    prepared = _prepare_grounded_primitive_trace(**case, allow_predicted_product_effects=True)
    record = prepared["projected_snapshot"]["parts"]["gear_small"]["product_effect_evidence"][0]
    assert record["kind"] == "predicted"
    assert prepared["frozen"]["allow_predicted_product_effects"] is True


@pytest.mark.parametrize("change", ["missing_ap", "extra_ap", "extra_field", "shared_part",
                                   "wrong_region", "wrong_part", "wrong_target", "malformed_geometry"])
def test_per_ap_physical_bindings_are_complete_and_exact(change: str) -> None:
    case = _gear_precedence_case(completion_time=.25)
    scope = case["requirement_scopes"][0]
    bindings = scope["physical_ap_bindings"]
    if change == "missing_ap":
        bindings.pop("ap002")
    elif change == "extra_ap":
        bindings["ap003"] = deepcopy(bindings["ap002"])
    elif change == "extra_field":
        bindings["ap002"]["result"] = "assembled"
    elif change == "shared_part":
        scope["part"] = "gear_small"
    elif change == "wrong_region":
        bindings["ap001"]["region"] = "assembly_board-v1 "
    elif change == "wrong_part":
        bindings["ap001"]["part"] = "ket4_square_4mm"
    elif change == "wrong_target":
        bindings["ap002"]["target"] = "Gear_Plate/Gear_Shaft_3"
    else:
        case["geometry"]["parts"]["gear_small"] = None
    _reviewed_unavailable(_grounded_check(case, trace_complete=True))


def test_wrong_initial_assembly_record_and_release_alone_do_not_complete_gear() -> None:
    case = _gear_precedence_case()
    gear = case["snapshot"]["parts"]["gear_small"]
    gear["processCompleted"].append({"process": "assembly", "target": "Gear_Plate/Gear_Shaft_3"})
    assert _grounded_check(case, trace_complete=True)["status"] == "violated"
    gear["state"] = "assembled"
    gear["target"] = "Gear_Plate/Gear_Shaft_1"
    assert _grounded_check(case, trace_complete=True)["status"] == "violated"


def test_predicted_checkpoint_is_not_reused_as_acknowledged_product_history() -> None:
    case = _gear_precedence_case()
    case["snapshot"]["parts"]["gear_small"]["product_effect_evidence"] = [{"kind": "predicted"}]
    result = _grounded_check(case, trace_complete=True)
    _reviewed_unavailable(result)
    assert "predicted product effect" in result["reason"]


def test_reviewed_checker_accepts_distinct_initial_part_target_bindings() -> None:
    case = _gear_precedence_case()
    part = case["snapshot"]["parts"]["gear_small"]
    part["processCompleted"].append({"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"})
    scope = case.pop("requirement_scopes")[0]
    case.pop("scene")
    population = sorted(case["geometry"]["resources"])
    case["applicability"] = {"version": 1,
        "scene_resources": [{"resource_id": rid, "resource_type": "configured_resource"} for rid in population],
        "participant_resource_types": ["configured_resource"],
        "regions": {"assembly_board-v1": {"resources": population, "excluded_resources": []}},
        "rules": [scope]}
    result = _reviewed_check(case, trace_complete=True)
    assert result["status"] == "satisfied", result["reason"]


def _grounded_extra_resource(case: dict) -> None:
    robot = deepcopy(case["scene"]["robots"][0])
    robot["resource_id"] = "ur5e-5"
    robot["cartesian_motion"]["resource_id"] = "ur5e-5"
    case["scene"]["robots"].append(robot)
    programs = case["scene"]["resource_programs"]["resources"]
    programs["ur5e-5"] = deepcopy(programs["ur5e-1"])
    case["geometry"]["resources"]["ur5e-5"] = deepcopy(case["geometry"]["resources"]["ur5e-1"])
    case["snapshot"]["resources"]["ur5e-5"] = deepcopy(case["snapshot"]["resources"]["ur5e-1"])
    case["stationary"]["ur5e-5"] = [[0, 22]]


def _grounded_structured(case: dict, formula: str = "G (ap002 -> F ap008)") -> None:
    """Bind saved descriptors to complete, explicitly synthetic task/state records."""
    context = "destination=assembly_board-v1"
    case["catalog"]["specifications"].append({
        "id": "structured_place_approach", "requirement": "place_approach reaches positioned.",
        "formula": formula,
        "aps": [
            {"label": "ap002", "full": f"ap_event/assembly/any/recovery-resource-3/place_approach/{context}",
             "meaning": "The bound place_approach task is executing."},
            {"label": "ap008", "full": f"ap_state/assembly/any/recovery-resource-3/positioned/{context}",
             "meaning": "The resource-owned resource_state is positioned in the bound context."},
        ],
    })
    common = {"resource_id": "ur5e-3", "resource_symbol": "recovery-resource-3",
              "process": "assembly", "product": "any", "context": context}
    case["requirement_scopes"].append({
        "specification": "structured_place_approach",
        "ap_groundings": {
            "ap002": {**common, "source": "task_event", "function": "place_approach"},
            "ap008": {**common, "source": "resource_state", "state_field": "resource_state",
                      "state_value": "positioned"},
        },
    })
    identity = {"resource_id": "ur5e-3", "process": "assembly",
                "product": "KET4_Square_4mm", "context": context}
    case["task_evidence"] = {
        "complete": True, "source_kind": "synthetic", "horizon": [0, 22],
        "events": [{**identity, "task_id": "synthetic_place_approach", "function": "place_approach",
                    "start_time": 10.123, "end_time": 13.123}],
    }
    case["state_evidence"] = {
        "complete": True, "source_kind": "synthetic", "horizon": [0, 22],
        "initial": [{**identity, "values": {"resource_state": "picked"}}],
        "updates": [{**identity, "task_id": "synthetic_place_approach", "time": 13.123,
                     "values": {"resource_state": "positioned"}}],
    }
    migrated = migrate_ppr_fixture(case)
    case.clear()
    case.update(migrated)


def test_grounded_frozen_scene_derives_all_66_pairs_and_preserves_recovery() -> None:
    from itertools import combinations

    from cais_spade_llm.resources.environment_models import build_environment_models

    case = _grounded_case()
    original = deepcopy(case)
    result = _grounded_check(case, trace_complete=True)
    assert result["status"] == "satisfied", result["reason"]
    assert case == original
    resources = sorted(build_environment_models(case["scene"]))
    assert len(resources) == 12
    assert "applicability" not in case
    assert all("resources" not in row and "participants" not in row and "pairs" not in row
               for row in case["requirement_scopes"])
    pairs = [tuple(row["resources"]) for row in result["bindings"]
             if row["specification"] == "shared_area_mutex"]
    assert len(pairs) == 66
    assert set(pairs) == set(combinations(resources, 2))
    assert result["trace_length"] == 107
    legacy_case = _reviewed_case()
    assert case["programs"] == legacy_case["programs"]
    assert len(case["programs"][0]["primitive_steps"]) == 22
    legacy = _reviewed_check(legacy_case, trace_complete=True)
    final = result["projected_snapshot"]
    for key in ("parts", "nominal_tasks", "task_statuses", "resume_entry_task_ids_by_resource",
                "resumable_task_ids_by_resource"):
        assert final[key] == legacy["projected_snapshot"][key]
    assert final["resources"]["KMR"] == legacy["projected_snapshot"]["resources"]["KMR"]
    for observed, prior in zip(result["observations"], legacy["observations"], strict=True):
        assert observed["time_exact"] == prior["time_exact"]
        assert observed["parts"] == prior["parts"]
        assert observed["resources"]["KMR"] == prior["resources"]["KMR"]
        for resource in legacy_case["geometry"]["resources"]:
            assert observed["region_occupancy"]["assembly_board-v1"][resource] == (
                prior["region_occupancy"]["assembly_board-v1"][resource])
    for resource, shape in case["geometry"]["resources"].items():
        if shape.get("stationary_only"):
            assert not {"held_part", "gripper_state", "grasp_transform"}.intersection(
                final["resources"][resource])


def test_grounded_additional_configured_resource_generates_78_pairs_and_detects_conflict() -> None:
    from cais_spade_llm.resources.environment_models import build_environment_models

    case = _grounded_case()
    catalog = deepcopy(case["catalog"])
    scopes = deepcopy(case["requirement_scopes"])
    _grounded_extra_resource(case)
    models = build_environment_models(case["scene"])
    assert len(models) == 13
    assert models["ur5e-5"]["local_event_alphabet"] == ["move_home"]
    result = _grounded_check(case, trace_complete=True)
    assert result["status"] == "satisfied", result["reason"]
    pairs = [row for row in result["bindings"] if row["specification"] == "shared_area_mutex"]
    assert len(pairs) == 78
    assert sum("ur5e-5" in row["resources"] for row in pairs) == 12
    case["snapshot"]["resources"]["ur5e-5"]["current_pose"][:3] = [0, 0, 1.04]
    rejected = _grounded_check(case, trace_complete=True)
    assert rejected["status"] == "violated", rejected["reason"]
    assert "ur5e-5" in rejected["counterexample"]["binding"]["resources"]
    assert case["catalog"] == catalog and case["requirement_scopes"] == scopes


@pytest.mark.parametrize("resource", ["M1", "M2", "Storage", "Conveyor", "Buffer For Machined parts",
                                     "3D Printing Station", "Exit", "ur5e-4"])
@pytest.mark.parametrize("missing", ["geometry", "snapshot", "stationary"])
def test_grounded_every_configured_resource_requires_physical_coverage(resource, missing) -> None:
    case = _grounded_case()
    if missing == "geometry":
        case["geometry"]["resources"].pop(resource)
    elif missing == "snapshot":
        case["snapshot"]["resources"].pop(resource)
    else:
        case["stationary"].pop(resource)
    _reviewed_unavailable(_grounded_check(case, trace_complete=True))


def test_grounded_fixed_equipment_conflict_is_not_pruned() -> None:
    case = _grounded_case()
    for resource in ("M1", "M2"):
        case["snapshot"]["resources"][resource]["current_pose"][:3] = [0, 0, 1.04]
    result = _grounded_check(case, trace_complete=True)
    assert result["status"] == "violated", result["reason"]
    assert result["counterexample"]["time"] == 0
    assert result["counterexample"]["binding"]["resources"] == ["M1", "M2"]


@pytest.mark.parametrize("mutation", ["missing_custody", "fake_stationary_only", "fixed_gripper",
                                    "fixed_motion", "missing_scope", "manual_pairs", "unknown_predicate"])
def test_grounded_unsupported_or_incomplete_grounding_is_unavailable(mutation) -> None:
    case = _grounded_case()
    if mutation == "missing_custody":
        case["snapshot"]["resources"]["ur5e-3"].pop("held_part")
    elif mutation == "fake_stationary_only":
        case["geometry"]["resources"]["ur5e-3"]["stationary_only"] = True
        for key in ("held_part", "gripper_state", "grasp_transform"):
            case["snapshot"]["resources"]["ur5e-3"].pop(key, None)
    elif mutation == "fixed_gripper":
        case["snapshot"]["resources"]["M1"]["gripper_state"] = "open"
    elif mutation == "fixed_motion":
        pose = deepcopy(case["snapshot"]["resources"]["M1"]["current_pose"])
        case["programs"].append(_reviewed_move("M1", [(0, pose), (22, pose)]))
        case["stationary"].pop("M1")
    elif mutation == "missing_scope":
        case["requirement_scopes"].pop()
    elif mutation == "manual_pairs":
        case["requirement_scopes"][0]["resources"] = ["KMR", "ur5e-3"]
    else:
        case["catalog"]["specifications"][1]["aps"][0]["full"] = "ap_event/unknown_predicate"
    _reviewed_unavailable(_grounded_check(case, trace_complete=True))


@pytest.mark.parametrize("variant", ["clear", "mutex", "precedence"])
def test_grounded_new_event_identities_do_not_change_physical_meanings(variant) -> None:
    case = _grounded_case()
    if variant == "mutex":
        case["snapshot"]["resources"]["ur5e-3"]["current_pose"][:3] = [0, 0, 1.04]
    elif variant == "precedence":
        _reviewed_ledger(case, [])
    renamed = deepcopy(case)
    for row in renamed["programs"][0]["step_results"]:
        for key in ("event_name", "des_event_id", "outline_id"):
            row["source"][key] = "independently_authored_" + row["source"][key]
    first = _grounded_check(case, trace_complete=True)
    second = _grounded_check(renamed, trace_complete=True)
    assert first["status"] == second["status"] == ("satisfied" if variant == "clear" else "violated")
    assert first["trace_id"] != second["trace_id"]
    assert first["bindings"] == second["bindings"]
    assert [(row["observation_index"], row["rule_id"], row["ap_values"]) for row in first["rule_checks"]] == [
        (row["observation_index"], row["rule_id"], row["ap_values"]) for row in second["rule_checks"]]
    if variant != "clear":
        old_sources = [row["source"] for row in first["counterexample"]["active_steps"]
                       if row["resource_id"] == "KMR"]
        new_sources = [row["source"] for row in second["counterexample"]["active_steps"]
                       if row["resource_id"] == "KMR"]
        assert old_sources and new_sources
        assert {row["step_index"] for row in old_sources} == {row["step_index"] for row in new_sources}
        assert all(row["event_name"].startswith("independently_authored_") for row in new_sources)


def test_grounded_complete_trim_ledger_is_required() -> None:
    case = _grounded_case()
    case["snapshot"]["parts"]["KET4_Square_4mm"].pop("processCompleted_complete")
    _reviewed_unavailable(_grounded_check(case, trace_complete=True))


def test_reviewed_unavailable_ledger_preserves_existing_initial_dfa_outputs() -> None:
    case = _reviewed_case()
    case["snapshot"]["parts"]["KET4_Square_4mm"].pop("processCompleted_complete")
    result = _reviewed_check(case, trace_complete=True)
    _reviewed_unavailable(result)
    assert len(result["safety_dfa_states_before"]) == 11
    assert len(result["observations"]) == 107


def test_grounded_structured_task_state_descriptors_keep_their_meanings() -> None:
    case = _grounded_case()
    _grounded_structured(case)
    # Physical occupancy changes inside this one declared task interval. The
    # structured task AP remains tied to the record, never to region crossings.
    outside = deepcopy(case["snapshot"]["resources"]["ur5e-3"]["current_pose"])
    inside = [0, 0, 1.04, *outside[3:]]
    case["programs"].append(_reviewed_move("ur5e-3", [(10.123, outside), (11, inside), (12, outside)]))
    case["stationary"]["ur5e-3"] = [[0, 10.123], [12, 22]]
    # Use the supplied task/state requirement alone to inspect its trace through
    # physical overlap, which is deliberately prohibited by the separate mutex.
    case["catalog"]["specifications"] = case["catalog"]["specifications"][-1:]
    case["requirement_scopes"] = case["requirement_scopes"][-1:]
    result = _grounded_check(case, trace_complete=True)
    assert result["status"] == "satisfied", result["reason"]
    checks = result["rule_checks"]
    for row in checks:
        assert row["ap_values"]["ap002"] == (10.123 <= row["time"] < 13.123)
        assert row["ap_values"]["ap008"] == (row["time"] >= 13.123)
    end = next(row for row in checks if row["time"] == 13.123)
    assert end["ap_values"] == {"ap002": False, "ap008": True}
    inside_event = [row for row in result["observations"] if 10.123 <= row["time"] < 13.123]
    assert {row["region_occupancy"]["assembly_board-v1"]["ur5e-3"] for row in inside_event} == {False, True}
    descriptors = {ap["label"]: ap for ap in case["catalog"]["specifications"][0]["aps"]}
    assert len(result["ap_evidence"]) == len(result["observations"]) * 2
    for row in result["ap_evidence"]:
        assert row["descriptor"] == descriptors[row["descriptor"]["label"]]
        assert type(row["value"]) is bool
        assert row["observation_index"] < result["trace_length"]
        assert row["binding"]
        assert row["evidence_source"]


@pytest.mark.parametrize("mutation", ["task_missing", "task_incomplete", "state_missing", "state_incomplete",
                                    "descriptor_mismatch", "duplicate_task", "overlapping_tasks", "state_conflict",
                                    "update_not_completed", "unknown_task", "missing_state_field"])
def test_grounded_ambiguous_task_state_evidence_cannot_pass(mutation) -> None:
    case = _grounded_case()
    _grounded_structured(case)
    if mutation == "task_missing":
        case.pop("task_evidence")
    elif mutation == "task_incomplete":
        case["task_evidence"]["complete"] = False
    elif mutation == "state_missing":
        case.pop("state_evidence")
    elif mutation == "state_incomplete":
        case["state_evidence"]["complete"] = False
    elif mutation == "descriptor_mismatch":
        case["requirement_scopes"][-1]["ap_groundings"]["ap002"]["function"] = "move_cartesian"
    elif mutation in {"duplicate_task", "overlapping_tasks"}:
        event = deepcopy(case["task_evidence"]["events"][0])
        if mutation == "overlapping_tasks":
            event["task_id"] = "another_task"
        case["task_evidence"]["events"].append(event)
    elif mutation == "state_conflict":
        update = deepcopy(case["state_evidence"]["updates"][0])
        update["values"]["resource_state"] = "idle"
        case["state_evidence"]["updates"].append(update)
    elif mutation == "update_not_completed":
        case["state_evidence"]["updates"][0]["time"] = 12
    elif mutation == "unknown_task":
        case["state_evidence"]["updates"][0]["task_id"] = "unprovided_task"
    else:
        case["state_evidence"]["initial"][0]["values"].pop("resource_state")
    _reviewed_unavailable(_grounded_check(case, trace_complete=True))


def test_grounded_generated_physical_event_does_not_fabricate_place_approach() -> None:
    case = _grounded_case()
    _grounded_structured(case, "G !ap002")
    case["task_evidence"]["events"] = []
    case["state_evidence"]["updates"] = []
    result = _grounded_check(case, trace_complete=True)
    assert result["status"] == "satisfied", result["reason"]
    assert any(row["region_occupancy"]["assembly_board-v1"]["KMR"] for row in result["observations"])
    assert all(not row["ap_values"]["ap002"] for row in result["rule_checks"]
               if row["specification"] == "structured_place_approach")


def test_grounded_combined_clock_pending_completion_and_continuation() -> None:
    case = _grounded_case()
    _grounded_structured(case, "F ap008")
    full = _grounded_check(case, trace_complete=True)
    assert full["status"] == "satisfied", full["reason"]
    assert full["trace_length"] > 107
    assert any(row["time"] == 10.123 for row in full["observations"])
    assert any(row["time"] == 13.123 for row in full["observations"])
    cut = next(index for index, row in enumerate(full["observations"]) if row["time"] == 13.123)
    prefix = _grounded_check(case, observation_slice=[0, cut])
    assert prefix["status"] == "prefix_checked" and prefix["is_safe"] is False
    assert len(prefix["pending_rule_ids"]) == 1
    suffix = _grounded_check(case, continuation=prefix["continuation"], observation_slice=[cut, full["trace_length"]])
    assert suffix["status"] == "prefix_checked" and suffix["is_safe"] is False
    complete = _grounded_check(case, continuation=suffix["continuation"],
                              observation_slice=[full["trace_length"], full["trace_length"]], trace_complete=True)
    assert complete["status"] == "satisfied"
    assert complete["observations"] == []
    assert prefix["observations"] + suffix["observations"] == full["observations"]
    assert complete["safety_dfa_states_after"] == full["safety_dfa_states_after"]


def test_grounded_X_uses_next_combined_observation() -> None:
    case = _grounded_case()
    _grounded_structured(case, "X ap002")
    event = case["task_evidence"]["events"][0]
    event["start_time"], event["end_time"] = 0.01, 0.02
    case["state_evidence"]["updates"][0]["time"] = 0.02
    rejected = _grounded_check(case, trace_complete=True)
    assert rejected["status"] == "violated", rejected["reason"]
    assert rejected["observations"][1]["time"] == 0.005
    case["catalog"]["specifications"][-1]["formula"] = "X (X ap002)"
    accepted = _grounded_check(case, trace_complete=True)
    assert accepted["status"] == "satisfied", accepted["reason"]
    assert accepted["observations"][2]["time"] == 0.01
    assert rejected["observations"] == accepted["observations"]


@pytest.mark.parametrize("mutation", ["gap", "duplicate", "old_clock", "state_history", "changed_evidence",
                                    "changed_scene", "early_complete"])
def test_grounded_continuation_rejects_incompatible_history_without_advancing(mutation) -> None:
    case = _grounded_case()
    _grounded_structured(case)
    prefix = _grounded_check(case, observation_slice=[0, 5])
    assert prefix["status"] == "prefix_checked", prefix["reason"]
    prior = deepcopy(prefix["continuation"])
    selection = [5, 10]
    if mutation == "gap":
        selection[0] = 6
    elif mutation == "duplicate":
        selection[0] = 4
    elif mutation == "old_clock":
        prior["clock_version"] = "primitive_observations_joint_trace_v1"
    elif mutation == "state_history":
        prior["previous_observation"]["time"] += 1
    elif mutation == "changed_evidence":
        case["task_evidence"]["events"][0]["start_time"] = 10.124
    elif mutation == "changed_scene":
        case["scene"]["robots"][0]["tf_lookup_timeout_sec"] += 1
    result = _grounded_check(case, continuation=prior, observation_slice=selection,
                             trace_complete=mutation == "early_complete")
    _reviewed_unavailable(result)
    assert result["continuation"] == prior


def test_grounded_missing_compiler_does_not_use_fixed_mutex_dfa(monkeypatch) -> None:
    from cais_spade_llm.agents.central_controller import (
        reviewed_primitive_program_safety as reviewed,
    )

    monkeypatch.setattr(reviewed.shutil, "which", lambda name: None)
    _reviewed_unavailable(_grounded_check(_grounded_case(), trace_complete=True))


def test_grounded_task_function_any_is_not_an_invented_wildcard() -> None:
    case = _grounded_case()
    _grounded_structured(case, "F ap002")
    ap = case["catalog"]["specifications"][-1]["aps"][0]
    ap["full"] = ap["full"].replace("/place_approach/", "/any/")
    case["requirement_scopes"][-1]["ap_groundings"]["ap002"]["function"] = "any"
    result = _grounded_check(case, trace_complete=True)
    assert result["status"] in {"violated", "unavailable"}, result["reason"]


@pytest.mark.parametrize("mutation", ["state_value", "shape", "task_horizon", "nonfinite_state"])
def test_grounded_malformed_or_impossible_evidence_is_unavailable(mutation) -> None:
    case = _grounded_case()
    _grounded_structured(case)
    if mutation == "state_value":
        case["state_evidence"]["initial"][0]["values"]["resource_state"] = "undeclared_state"
    elif mutation == "shape":
        case["geometry"]["resources"]["M1"] = []
    elif mutation == "task_horizon":
        case["task_evidence"]["horizon"] = [0, 21]
    else:
        case["state_evidence"]["initial"][0]["values"]["resource_state"] = float("nan")
    _reviewed_unavailable(_grounded_check(case, trace_complete=True))


def _grounded_state_field(case: dict, resource: str, field: str, value: str, observed) -> None:
    descriptor = f"ap_state/any/any/{resource}/{field}={value}/any"
    case["catalog"] = {"version": 2, "specifications": [{
        "id": "exact_state_field", "requirement": "The supplied state predicate must remain false.",
        "formula": "G !ap001", "aps": [{"label": "ap001", "full": descriptor,
                                       "meaning": "Exact resource-owned state field equality."}],
    }]}
    case["requirement_scopes"] = [{"specification": "exact_state_field", "ap_groundings": {
        "ap001": {"source": "resource_state", "resource_id": resource, "resource_symbol": resource,
                  "process": "any", "product": "any", "context": "any", "state_field": field,
                  "state_value": value},
    }}]
    case["state_evidence"] = {
        "complete": True, "source_kind": "synthetic", "horizon": [0, 22],
        "initial": [{"resource_id": resource, "process": "any", "product": "any", "context": "any",
                     "values": {field: observed}}], "updates": [],
    }
    migrated = migrate_ppr_fixture(case)
    case.clear()
    case.update(migrated)


def test_grounded_boolean_state_cannot_silently_become_false() -> None:
    case = _grounded_case()
    _grounded_state_field(case, "Conveyor", "belt_stopped", "true", True)
    result = _grounded_check(case, trace_complete=True)
    assert result["status"] == "violated", result["reason"]
    assert result["counterexample"]["ap_values"] == {"ap001": True}


def test_grounded_state_ledger_cannot_contradict_primitive_custody() -> None:
    case = _grounded_case()
    _grounded_state_field(case, "KMR", "held_part", "KET8_Square_8mm", "KET4_Square_4mm")
    _reviewed_unavailable(_grounded_check(case, trace_complete=True))


def test_grounded_state_tokens_are_never_trimmed_to_match_descriptors() -> None:
    case = _grounded_case()
    _grounded_state_field(case, "ur5e-3", "task_ctx.part_name", "KET4_Square_4mm", " KET4_Square_4mm ")
    case["catalog"]["specifications"][0]["formula"] = "F ap001"
    result = _grounded_check(case, trace_complete=True)
    assert result["status"] in {"violated", "unavailable"}, result["reason"]


@pytest.mark.parametrize("mutation", ["ambiguous_symbol", "changed_configured_identity"])
def test_grounded_resource_symbols_cannot_change_identity(mutation) -> None:
    case = _grounded_case()
    _grounded_structured(case)
    groundings = case["requirement_scopes"][-1]["ap_groundings"]
    if mutation == "ambiguous_symbol":
        groundings["ap002"]["resource_id"] = "ur5e-4"
    else:
        ap = case["catalog"]["specifications"][-1]["aps"][0]
        ap["full"] = ap["full"].replace("/recovery-resource-3/", "/ur5e-4/")
        groundings["ap002"]["resource_symbol"] = "ur5e-4"
    _reviewed_unavailable(_grounded_check(case, trace_complete=True))


def test_grounded_adjacent_tasks_and_simultaneous_state_updates_are_joint() -> None:
    case = _grounded_case()
    _grounded_structured(case, "G (ap002 -> F ap008)")
    first = case["task_evidence"]["events"][0]
    first["start_time"], first["end_time"] = 12, 13
    case["state_evidence"]["updates"][0]["time"] = 13
    second = deepcopy(first)
    second.update(task_id="synthetic_place_release", function="place_release", start_time=13, end_time=14)
    case["task_evidence"]["events"].append(second)
    result = _grounded_check(case, trace_complete=True)
    assert result["status"] == "satisfied", result["reason"]
    at_completion = [row for row in result["observations"] if row["time"] == 13]
    assert len(at_completion) == 1
    assert at_completion[0]["resources"]["KMR"]["held_part"] is None
    check = next(row for row in result["rule_checks"]
                 if row["time"] == 13 and row["specification"] == "structured_place_approach")
    assert check["ap_values"] == {"ap002": False, "ap008": True}


@pytest.fixture(scope="module")
def part_slippage_checker(tmp_path_factory):
    """Replay an explicitly migrated copy while keeping archived evidence intact."""
    import functools
    import hashlib
    import runpy
    import shutil

    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_part_slippage_safety.py"))
    archived = module["FIXTURES"]
    directory = tmp_path_factory.mktemp("ppr_part_slippage")
    for name in ("initial_context.json", "mutex.json", "precedence.json", "safe.json"):
        shutil.copyfile(archived / name, directory / name)
    context = json.loads((directory / "initial_context.json").read_text())
    source = context["sources"]["predefined_safety"]
    source["sha256"] = hashlib.sha256((module["ROOT"] / source["path"]).read_bytes()).hexdigest()
    (directory / "initial_context.json").write_text(json.dumps(context))
    module["load_candidate"] = functools.partial(module["load_candidate"], directory=directory)
    module["check_candidates"] = functools.partial(module["check_candidates"], directory=directory)
    module["FIXTURES"] = directory
    module["ARCHIVED_FIXTURES"] = archived
    return module


@pytest.fixture(scope="module")
def part_slippage_report(part_slippage_checker):
    return part_slippage_checker["check_candidates"]()


@pytest.mark.parametrize("name,expected,violation", [
    ("mutex", "violated", "SAFE_shared_area_mutex"),
    ("precedence", "violated", "SAFE_gear_small_before_KET4_Square_4mm"),
    ("safe", "satisfied", None),
])
def test_mock_part_slippage_predefined_results(part_slippage_report, name, expected, violation) -> None:
    report = part_slippage_report
    assert report["expectations_met"] and report["llm_calls"] == 0
    assert report["dispatch_authorized"] is False
    result = next(row for row in report["cases"] if row["file"] == name + ".json")
    assert result["status"] == expected
    assert result["concrete_rule_count"] == 67
    assert len(result["configured_resources"]) == 12
    assert result["expectations_met"]
    assert result["specifications"]["SAFE_shared_area_mutex"]["concrete_rule_count"] == 66
    assert result["specifications"]["SAFE_gear_small_before_KET4_Square_4mm"]["concrete_rule_count"] == 1
    if violation is not None:
        counterexample = result["counterexample"]
        assert counterexample["specification"] == violation
        assert counterexample["transition"]["accepting_reachable"] is False
        assert counterexample["active_steps"][0]["primitive"] == "move_cartesian"
        assert counterexample["active_steps"][0]["primitive_index"] == 1
        assert len(counterexample["aps"]) == 2
    else:
        assert result["counterexample"] is None
        completion = next(row for row in result["ap_changes"]
                          if row["specification"] == "SAFE_gear_small_before_KET4_Square_4mm"
                          and row["ap_values"]["ap002"])
        entry = next(row for row in result["ap_changes"]
                     if row["specification"] == "SAFE_gear_small_before_KET4_Square_4mm"
                     and row["ap_values"]["ap001"])
        assert completion["time"] == 9 < entry["time"]
    final = result["modeled_final_state"]
    assert all(row["held_part"] is None for row in final["resources"].values())
    assert final["parts"]["gear_small"]["processCompleted"] == [
        {"process": "print_part"}, {"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"}]
    assert final["parts"]["KET4_Square_4mm"]["processCompleted"] == [{"process": "trim", "result": "square"}]
    assert final["pending_tasks"][0]["status"] == "pending"


@pytest.mark.parametrize("name", ["mutex", "precedence", "safe"])
def test_mock_part_slippage_event_names_do_not_determine_physical_aps(part_slippage_checker, name) -> None:
    candidate, inputs, _ = part_slippage_checker["load_candidate"](name)
    original = _grounded_check(inputs, trace_complete=True)
    mapping = {}
    for index, event in enumerate(candidate["outline_events"]):
        for field in ("outline_id", "des_event_id", "event_name"):
            mapping[event[field]] = f"previously_unseen_recovery_{index}_{field}"

    def rename(value):
        if isinstance(value, dict):
            return {key: rename(child) for key, child in value.items()}
        if isinstance(value, list):
            return [rename(child) for child in value]
        return mapping.get(value, value) if isinstance(value, str) else value

    renamed_candidate, renamed_inputs = rename(candidate), rename(inputs)
    part_slippage_checker["validate_event_records"](renamed_candidate, renamed_inputs)
    renamed = _grounded_check(renamed_inputs, trace_complete=True)
    assert renamed["status"] == original["status"]
    assert renamed["trace_id"] != original["trace_id"]
    assert [(row["rule_id"], row["observation_index"], row["value"]) for row in renamed["ap_evidence"]] == [
        (row["rule_id"], row["observation_index"], row["value"]) for row in original["ap_evidence"]]
    if renamed["counterexample"]:
        source = renamed["counterexample"]["active_steps"][0]
        assert source["outline_id"].startswith("previously_unseen_recovery_")
        assert source["event_name"].startswith("previously_unseen_recovery_")


@pytest.mark.parametrize("change,expected", [
    ("missing_participant", "unavailable"), ("missing_trajectory", "unavailable"),
    ("missing_ledger", "unavailable"), ("missing_acknowledgement", "violated"),
    ("wrong_target", "unavailable"), ("predicted_acknowledgement", "unavailable"),
    ("conflicting_custody", "unavailable"),
])
def test_mock_part_slippage_incomplete_or_forged_evidence_cannot_pass(part_slippage_checker, change, expected) -> None:
    _, inputs, _ = part_slippage_checker["load_candidate"]("safe")
    if change == "missing_participant":
        del inputs["geometry"]["resources"]["M1"]
    elif change == "missing_trajectory":
        del inputs["programs"][0]["step_results"][0]["model_evidence"]["trajectory"]
    elif change == "missing_ledger":
        del inputs["snapshot"]["parts"]["gear_small"]["processCompleted_evidence"]
    elif change == "missing_acknowledgement":
        inputs["product_effect_evidence"]["updates"] = []
    elif change == "wrong_target":
        inputs["product_effect_evidence"]["updates"][0]["product_effects"]["gear_small"]["processCompleted"][0]["target"] = "Gear_Plate/Gear_Shaft_3"
    elif change == "predicted_acknowledgement":
        inputs["product_effect_evidence"]["updates"][0]["kind"] = "predicted"
    else:
        inputs["snapshot"]["resources"]["ur5e-4"].update(held_part="gear_small", gripper_state="closed")
    before = deepcopy(inputs)
    result = _grounded_check(inputs, trace_complete=True)
    assert result["status"] == expected, result["reason"]
    assert result["is_safe"] is False
    assert inputs == before
    assert result["safety_dfa_states_after"] == result["safety_dfa_states_before"]
    if change == "missing_acknowledgement":
        assert result["counterexample"]["specification"] == "SAFE_gear_small_before_KET4_Square_4mm"


@pytest.mark.parametrize("change", ["primitive_reference", "predecessor", "task_reference"])
def test_mock_part_slippage_records_preserve_exact_provenance(part_slippage_checker, change) -> None:
    candidate, inputs, _ = part_slippage_checker["load_candidate"]("safe")
    if change == "primitive_reference":
        inputs["programs"][0]["primitive_steps"][0]["source"]["outline_id"] = "another_event"
    elif change == "predecessor":
        event = candidate["outline_events"][0]
        event["predecessors"] = [candidate["outline_events"][-1]["outline_id"]]
    else:
        inputs["task_evidence"]["events"][0]["function"] = "assembly_completed_because_the_name_says_so"
    with pytest.raises(ValueError):
        part_slippage_checker["validate_event_records"](candidate, inputs)


def test_mock_part_slippage_report_keeps_supplied_formulas_aps_and_saved_results(part_slippage_checker, part_slippage_report) -> None:
    import hashlib

    directory = part_slippage_checker["FIXTURES"]
    _, _, document = part_slippage_checker["load_candidate"]("safe")
    for supplied, compiled in zip(document["catalog"]["specifications"], part_slippage_report["definitions"], strict=True):
        assert compiled["id"] == supplied["id"]
        assert compiled["ltlf"] == supplied["formula"]
        assert compiled["aps"] == supplied["aps"]
        assert "digraph" in compiled["dfa_dot"]
    archived = part_slippage_checker["ARCHIVED_FIXTURES"]
    saved = json.loads((archived / "report.json").read_text())
    for old, current in zip(saved["cases"], part_slippage_report["cases"], strict=True):
        for field in ("case_id", "events", "expected", "status", "configured_resources",
                      "concrete_rule_count", "modeled_final_state", "expectations_met"):
            assert current[field] == old[field]
        for identifier, old_rule in old["specifications"].items():
            assert current["specifications"][identifier]["status"] == old_rule["status"]
        if old["counterexample"]:
            for field in ("specification", "time", "ap_values", "active_steps"):
                assert current["counterexample"][field] == old["counterexample"][field]
    for name, expected in saved["fixture_sha256"].items():
        assert hashlib.sha256((archived / name).read_bytes()).hexdigest() == expected
    for name, expected in part_slippage_report["fixture_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("extra_resource,expected_pairs", [(False, 66), (True, 78)])
def test_ppr_wildcard_mutex_authoring_expands_every_configured_resource(extra_resource, expected_pairs):
    from cais_spade_llm.agents.central_controller.ppr_ap import (
        build_mutex_specification, make_ap_definition,
    )
    case = _grounded_case()
    if extra_resource:
        _grounded_extra_resource(case)
    definition = build_mutex_specification(
        "SAFE_shared_area_mutex", "At most one resource may be in the region.",
        make_ap_definition("ap_state", "*", "*", "*", "any", {"region": "assembly_board-v1"}),
    )
    case["catalog"] = {"version": 2, "specifications": [definition]}
    case["requirement_scopes"] = [{"specification": "SAFE_shared_area_mutex"}]
    result = _grounded_check(case, trace_complete=True)
    assert result["status"] in {"satisfied", "violated"}, result["reason"]
    assert len(result["bindings"]) == expected_pairs
    observed = {}
    for row in result["ap_evidence"]:
        ap = row["descriptor"]["definition"]
        assert ap["kind"] == "ap_state"
        assert ap["state"] == {"symbol": "any", "arguments": {"region": "assembly_board-v1"}}
        assert not ap["resource"].startswith("$")
        observed.setdefault(row["rule_id"], set()).add(ap["resource"])
    assert all(len(pair) == 2 for pair in observed.values())
    assert len(observed) == expected_pairs

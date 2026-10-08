from __future__ import annotations

"""Continuous interpolation bounds and uncertainty in recovery composition."""

import math
from copy import deepcopy

import pytest

from cais_spade_llm.resources.continuous_motion import (
    ContinuousMotion,
    cosine,
    joint_coefficients,
    occupancy,
    polynomial_bounds,
    sine,
)


def configuration():
    return {
        "joint_names": ["slide"],
        "interpolation": "splines",
        "root": "world",
        "root_xyz": [0, 0, 0],
        "root_rpy": [0, 0, 0],
        "reference_link": "body",
        "joints": [
            {
                "name": "slide",
                "parent": "world",
                "child": "body",
                "type": "prismatic",
                "xyz": [0, 0, 0],
                "rpy": [0, 0, 0],
                "axis": [1, 0, 0],
            }
        ],
        "components": [{"id": "body_box", "link": "body", "bounds": [[-0.05, 0.05]] * 3}],
    }


def trajectory(first=0.0, last=0.0, velocities=(4.0, -4.0)):
    return {
        "joint_names": ["slide"],
        "duration_ns": 1_000_000_000,
        "points": [
            {
                "positions": [q],
                "velocities": [] if velocities is None else [velocities[i]],
                "accelerations": [],
                "effort": [],
                "time_from_start": {"sec": i, "nanosec": 0},
            }
            for i, q in enumerate((first, last))
        ],
    }


@pytest.mark.parametrize("value", [-7.9, -math.pi, -1.0, 0.0, 1.0, math.pi, 7.9])
def test_trigonometric_bounds_include_known_values(value):
    for function, reference in ((sine, math.sin), (cosine, math.cos)):
        lo, hi = function((value, value))
        assert lo <= reference(value) <= hi


def test_trigonometric_bounds_include_interior_extrema():
    assert sine((0, math.pi))[1] == 1
    assert cosine((-math.pi, math.pi)) == (-1.0, 1.0)


def test_cubic_motion_crosses_region_between_identical_endpoints():
    motion = ContinuousMotion(trajectory(), configuration())
    region = [[0.8, 1.2], [-0.1, 0.1], [-0.1, 0.1]]
    assert occupancy(motion.boxes(0, 0), region) == [False]
    assert occupancy(motion.boxes(1, 1), region) == [False]
    assert occupancy(motion.boxes(0.5, 0.5), region) == [True]
    assert occupancy(motion.boxes(0, 1), region) == [False, True]
    assert motion.joints(0, 1)["slide"][1] >= 1


def test_quintic_preserves_endpoint_derivatives():
    first, last = trajectory()["points"]
    first["accelerations"], last["accelerations"] = [2.0], [-3.0]
    c = joint_coefficients(first, last, 0, 1)
    assert c[0] == 0 and sum(c) == 0
    assert c[1] == 4 and sum(i * c[i] for i in range(1, 6)) == -4
    assert 2 * c[2] == 2 and sum(i * (i - 1) * c[i] for i in range(2, 6)) == -3
    low, high = polynomial_bounds(c)
    assert low <= 0 <= high and high > 0


def test_continuous_identity_and_interpolation_are_required():
    config = configuration()
    config["interpolation"] = "none"
    with pytest.raises(ValueError, match="interpolation"):
        ContinuousMotion(trajectory(), config)
    config = configuration()
    config["joint_names"] = ["different"]
    with pytest.raises(ValueError, match="joint"):
        ContinuousMotion(trajectory(), config)


def test_rotation_can_cross_without_endpoint_collision():
    config = configuration()
    config["joints"][0].update(type="revolute", axis=[0, 0, 1])
    config["components"][0]["bounds"] = [[0.9, 1.1], [-0.01, 0.01], [-0.01, 0.01]]
    motion = ContinuousMotion(trajectory(0, math.pi, None), config)
    region = [[-0.1, 0.1], [0.8, 1.2], [-0.1, 0.1]]
    assert occupancy(motion.boxes(0, 0), region) == [False]
    assert occupancy(motion.boxes(1, 1), region) == [False]
    assert occupancy(motion.boxes(0.5, 0.5), region) == [True]


def test_motion_configuration_and_commands_are_not_mutated():
    raw, config = trajectory(), configuration()
    before = deepcopy([raw, config])
    ContinuousMotion(raw, config).boxes(0.2, 0.8)
    assert [raw, config] == before


def composition_case(*, conflict=False, crossing=False):
    from test_predefined_safety import _document
    from test_primitive_program_safety import _gear_precedence_case

    from cais_spade_llm.resources.resource_safety_preparation import PrimitiveModel

    case = _gear_precedence_case()
    document = _document()
    case.update({key: document[key] for key in ("catalog", "requirement_scopes")})
    case["geometry"]["regions"]["assembly_board-v1"]["bounds"] = [
        [0.8, 1.2],
        [-0.1, 0.1],
        [-0.1, 0.1],
    ]
    for key in ("snapshot", "geometry"):
        case[key]["resources"].pop("assembly_board-v1", None)
    for index, (rid, state) in enumerate(case["snapshot"]["resources"].items()):
        state["current_pose"] = [5 + index, 0, 0, 0, 0, 0, 1]
        state["contained_parts"] = []
        if "held_part" in state:
            state.update(held_part=None, grasp_transform=None, gripper_state="open")
        state.pop("base_pose", None)
        case["geometry"]["resources"][rid].pop("base_footprint", None)
        case["geometry"]["resources"][rid]["footprint"] = [[-0.05, 0.05]] * 3
    for part in case["snapshot"]["parts"].values():
        part.update(contained_by=None, stationary_until=1, current_pose=[-3, 0, 0, 0, 0, 0, 1])
    rid = "ur5e-4"
    state = case["snapshot"]["resources"][rid]
    state.update(current_pose=[0, 0, 0, 0, 0, 0, 1], joint_positions=[0.0], joint_names=["slide"])
    if conflict:
        case["snapshot"]["resources"]["ur5e-3"]["current_pose"] = [1, 0, 0, 0, 0, 0, 1]
    source = {
        "outline_id": "SUPPLIED_MOTION",
        "des_event_id": "SUPPLIED_MOTION_EVENT",
        "event_name": "new_recovery_motion",
        "step_index": 0,
    }
    if state.get("resource_jid"):
        source["resource_jid"] = state["resource_jid"]
    raw = trajectory(0, 1, None) if not crossing else trajectory()
    params = {"x": 0 if crossing else 1, "y": 0, "z": 0}
    program = {
        "resource_id": rid,
        "primitive_steps": [{"primitive": "move_cartesian", "params": params, "source": source}],
        "step_results": [
            {
                "primitive": "move_cartesian",
                "resolved_params": params,
                "source": source,
                "start_time": 0,
                "end_time": 1,
                "success": True,
                "model_evidence": {
                    "continuous_motion": {"joint_trajectory": raw, "configuration": configuration()}
                },
            }
        ],
    }

    def effects(**kwargs):
        return {
            "trajectory": None,
            "base_trajectory": None,
            "part_trajectories": {},
            "transfers": [],
            "resource_updates": {},
            "continuous_motion": kwargs["evidence"]["continuous_motion"],
        }

    model = PrimitiveModel("test_continuous", 2, configuration(), effects)
    case["programs"] = [program]
    case["stationary"] = {
        name: ([[0, 1]] if name != rid else []) for name in case["snapshot"]["resources"]
    }
    inputs = {
        "grounding_inputs": case,
        "recovery_events": [
            {
                "outline_id": source["outline_id"],
                "des_event_id": source["des_event_id"],
                "event_name": source["event_name"],
                "resource_id": rid,
                "predecessors": [],
                "primitive_step_indices": [0],
            }
        ],
        "running_work": [],
        "event_start_choices": [
            {
                "id": "supplied",
                "starts": {source["outline_id"]: 0},
                "programs": [deepcopy(program)],
                "stationary": deepcopy(case["stationary"]),
            }
        ],
        "completion": {"resources": {rid: {"held_part": None}}, "parts": {}},
    }
    return inputs, {rid: model}


@pytest.mark.parametrize("conflict,expected", [(False, "allowed"), (True, "held")])
def test_continuous_composition_retains_all_resource_pairs(conflict, expected):
    from cais_spade_llm.agents.central_controller.local_composition import Budget
    from cais_spade_llm.agents.central_controller.offline_recovery_composition import (
        analyze_grounded_recovery_composition,
    )

    case, models = composition_case(conflict=conflict)
    result = analyze_grounded_recovery_composition(
        **case, primitive_models=models, budget=Budget(seconds=30)
    )
    assert result["status"] == expected, result["reason"]
    assert len(result["bindings"]) == 67
    assert result["clock_version"] == "continuous_physical_boundaries_v1"
    if conflict:
        assert result["counterexample"]
    else:
        assert result["completion_witness"]["completed_event_ids"] == ["SUPPLIED_MOTION"]


def test_endpoint_clearance_cannot_hide_an_internal_conflict():
    from cais_spade_llm.agents.central_controller.local_composition import Budget
    from cais_spade_llm.agents.central_controller.offline_recovery_composition import (
        analyze_grounded_recovery_composition,
    )

    case, models = composition_case(conflict=True, crossing=True)
    result = analyze_grounded_recovery_composition(
        **case, primitive_models=models, budget=Budget(seconds=30)
    )
    assert result["status"] in {"held", "inconclusive"}, result
    assert result["completion_witness"] is None


def _analyze(case, models, **kwargs):
    from cais_spade_llm.agents.central_controller.local_composition import Budget
    from cais_spade_llm.agents.central_controller.offline_recovery_composition import (
        analyze_grounded_recovery_composition,
    )

    return analyze_grounded_recovery_composition(
        **case, primitive_models=models, budget=Budget(seconds=30), **kwargs
    )


def _custody_case(*, completed=True, conflict=False):
    from cais_spade_llm.resources.resource_safety_preparation import PrimitiveModel

    case, _ = composition_case(conflict=conflict)
    inputs, rid, part = case["grounding_inputs"], "ur5e-4", "KET4_Square_4mm"
    inputs["horizon"] = [0, 4]
    for row in inputs["snapshot"]["parts"].values():
        row["stationary_until"] = 4
    inputs["snapshot"]["parts"][part]["current_pose"] = [0, 0, 0, 0, 0, 0, 1]
    inputs["geometry"]["parts"][part]["local_bounds"] = [[-.02, .02]] * 3
    if completed:
        inputs["snapshot"]["parts"]["gear_small"]["processCompleted"].append(
            {"process": "assembly", "target": "Gear_Plate/Gear_Shaft_1"})
    program = inputs["programs"][0]
    source = deepcopy(program["primitive_steps"][0]["source"])
    program["primitive_steps"], program["step_results"] = [], []
    for index, (primitive, first, last) in enumerate((
        ("grasp_part", 0, 0), ("move_cartesian", 0, 1),
        ("release_part", 1, 1), ("move_cartesian", 1, 0),
    )):
        params = ({"x": last, "y": 0, "z": 0} if primitive == "move_cartesian" else
                  {"model_name": part, "part_name": part})
        command = {"primitive": primitive, "params": params, "source": {**source, "step_index": index}}
        custody = []
        if primitive == "grasp_part":
            custody = [{"kind": "grasp", "part": part, "pose": [0, 0, 0, 0, 0, 0, 1],
                        "grasp_transform": [0, 0, 0, 0, 0, 0, 1]}]
        elif primitive == "release_part":
            custody = [{"kind": "release", "part": part, "pose": [1, 0, 0, 0, 0, 0, 1],
                        "contained_by": None, "stationary_until": 4}]
        program["primitive_steps"].append(command)
        program["step_results"].append({
            "primitive": primitive, "resolved_params": params, "source": command["source"],
            "success": True, "start_time": index, "end_time": index + 1,
            "model_evidence": {"continuous_motion": {"joint_trajectory": trajectory(first, last, None),
                                                      "configuration": configuration()},
                               "custody_effects": custody},
        })
    inputs["stationary"] = {name: ([[0, 4]] if name != rid else []) for name in inputs["snapshot"]["resources"]}
    case["recovery_events"][0]["primitive_step_indices"] = [0, 1, 2, 3]
    choice = case["event_start_choices"][0]
    choice.update(programs=deepcopy(inputs["programs"]), stationary=deepcopy(inputs["stationary"]))

    def effects(**kwargs):
        evidence = kwargs["evidence"]
        return {"trajectory": None, "base_trajectory": None, "part_trajectories": {},
                "transfers": [], "resource_updates": {},
                "continuous_motion": evidence["continuous_motion"],
                "custody_effects": evidence["custody_effects"]}

    return case, {rid: PrimitiveModel("test_continuous_custody", 3, configuration(), effects)}


@pytest.mark.parametrize("completed,conflict,expected", [(True, False, "allowed"), (False, False, "held"), (True, True, "held")])
def test_continuous_custody_checks_both_fixed_specifications(completed, conflict, expected):
    case, models = _custody_case(completed=completed, conflict=conflict)
    result = _analyze(case, models)
    assert result["status"] == expected, result["reason"]
    assert result["clock_version"] == "continuous_physical_boundaries_v2"
    assert len(result["bindings"]) == 67
    if expected == "allowed":
        assert result["completion_witness"]


@pytest.mark.parametrize("change", ["geometry", "grasp", "release", "stationary", "duplicate", "malformed_effect"])
def test_continuous_custody_requires_complete_consistent_evidence(change):
    case, models = _custody_case()
    rows = case["event_start_choices"][0]["programs"][0]["step_results"]
    if change == "geometry":
        del case["grounding_inputs"]["geometry"]["parts"]["KET4_Square_4mm"]["local_bounds"]
    elif change == "grasp":
        rows[0]["model_evidence"]["custody_effects"][0]["grasp_transform"][0] = 1
    elif change == "release":
        rows[2]["model_evidence"]["custody_effects"][0]["part"] = "gear_small"
    elif change == "stationary":
        rows[2]["model_evidence"]["custody_effects"][0]["stationary_until"] = 3
    elif change == "duplicate":
        rows[0]["model_evidence"]["custody_effects"] *= 2
    else:
        rows[0]["model_evidence"]["custody_effects"] = [None]
    assert _analyze(case, models)["status"] == "inconclusive"


def test_tracking_bounds_enclose_more_than_the_commanded_endpoint():
    config = configuration()
    config["joint_position_error"] = {"slide": .1}
    motion = ContinuousMotion(trajectory(0, 0, None), config)
    assert motion.joints(0, 0)["slide"][0] <= -.1
    assert motion.joints(0, 0)["slide"][1] >= .1
    assert occupancy(motion.boxes(0, 1), [[.1,.12], [-.1,.1], [-.1,.1]]) == [False, True]


def test_tracking_bounds_prevent_admission_beside_a_stationary_occupied_region():
    case, models = composition_case(conflict=True)
    inputs = case["grounding_inputs"]
    inputs["geometry"]["regions"]["assembly_board-v1"]["bounds"][0] = [1.12, 1.3]
    inputs["snapshot"]["resources"]["ur5e-3"]["current_pose"][0] = 1.22
    clear = _analyze(case, models)
    assert clear["status"] == "allowed", clear

    # This supplied bound tests geometry only; it is not native execution evidence.
    for program in (inputs["programs"][0], case["event_start_choices"][0]["programs"][0]):
        continuous = program["step_results"][0]["model_evidence"]["continuous_motion"]
        continuous["configuration"]["joint_position_error"] = {"slide": .2}
    bounded = _analyze(case, models)
    assert bounded["status"] in {"held", "inconclusive"}, bounded
    assert bounded["completion_witness"] is None
    assert any(row.get("possible_values") == [False, True] for row in bounded["ap_evidence"])


@pytest.mark.parametrize(
    "change",
    [
        "resource",
        "stationary",
        "model",
        "interpolation",
        "custody",
        "joint_start",
        "program",
        "ledger",
    ],
)
def test_incomplete_continuous_evidence_cannot_pass(change):
    case, models = composition_case()
    inputs = case["grounding_inputs"]
    if change == "resource":
        inputs["geometry"]["resources"].pop("M1")
    elif change == "stationary":
        case["event_start_choices"][0]["stationary"].pop("M1")
    elif change == "model":
        models.clear()
    elif change == "interpolation":
        for program in [inputs["programs"][0], case["event_start_choices"][0]["programs"][0]]:
            program["step_results"][0]["model_evidence"]["continuous_motion"]["configuration"][
                "interpolation"
            ] = "unknown"
    elif change == "custody":
        inputs["snapshot"]["resources"]["ur5e-4"]["held_part"] = "gear_small"
    elif change == "joint_start":
        inputs["snapshot"]["resources"]["ur5e-4"]["joint_positions"] = [0.1]
    elif change == "ledger":
        inputs["snapshot"]["parts"]["gear_small"].pop("processCompleted_evidence")
    else:
        case["event_start_choices"][0]["programs"][0]["primitive_steps"][0]["params"]["x"] = 3
    result = _analyze(case, models)
    assert result["status"] == "inconclusive", result
    assert result["completion_witness"] is None


def test_uncertain_contact_remains_inconclusive_with_definite_boolean_evidence_separate():
    case, models = composition_case(conflict=True)
    # The final box touches the region boundary. Outward arithmetic cannot prove
    # which side of the boundary the point occupies; no samples erase that fact.
    case["grounding_inputs"]["geometry"]["regions"]["assembly_board-v1"]["bounds"][0] = [1.05, 1.2]
    result = _analyze(case, models)
    assert result["status"] == "inconclusive"
    assert any(
        row["value"] is None and row["possible_values"] == [False, True]
        for row in result["ap_evidence"]
    )
    assert result["completion_witness"] is None


@pytest.mark.parametrize(
    "formula,pending",
    [
        ("F ap001", True),
        ("(!ap001 U ap001)", True),
        ("F (ap001 & ap002)", True),
        ("X ap001", False),
    ],
)
def test_continuous_pending_formulas_require_acceptance_and_reject_next(formula, pending):
    case, models = composition_case()
    case["grounding_inputs"]["catalog"]["specifications"][0]["formula"] = formula
    result = _analyze(case, models)
    assert result["status"] == "inconclusive", result
    # F for every configured resource cannot be discharged by only ur5e-4.
    if pending:
        assert result["pending_rule_ids"]
    else:
        assert "next" in result["reason"].lower() or "X" in result["reason"]


@pytest.mark.parametrize("formula", ["F !(ap001 & ap002)", "ap001 U !ap001"])
def test_continuous_eventually_and_until_accept_resolved_obligations(formula):
    case, models = composition_case()
    case["grounding_inputs"]["catalog"]["specifications"][0]["formula"] = formula
    result = _analyze(case, models)
    assert result["status"] == "allowed", result
    assert result["completion_witness"]


def test_renamed_continuous_events_keep_physical_values_and_distinct_sources():
    case, models = composition_case(conflict=True)
    original = _analyze(case, models)
    event = case["recovery_events"][0]
    event.update(
        outline_id="ANOTHER_OUTLINE", des_event_id="ANOTHER_EVENT", event_name="another_motion"
    )
    choice = case["event_start_choices"][0]
    choice["starts"] = {"ANOTHER_OUTLINE": 0}
    for program in (case["grounding_inputs"]["programs"][0], choice["programs"][0]):
        for step in (*program["primitive_steps"], *program["step_results"]):
            step["source"].update(
                {key: event[key] for key in ("outline_id", "des_event_id", "event_name")}
            )
    renamed = _analyze(case, models)
    assert original["status"] == renamed["status"] == "held"
    assert [(row["rule_id"], row["value"]) for row in original["ap_evidence"]] == [
        (row["rule_id"], row["value"]) for row in renamed["ap_evidence"]
    ]
    assert "ANOTHER_OUTLINE" in str(renamed["counterexample"])
    assert "SUPPLIED_MOTION" not in str(renamed["counterexample"])


def test_mixed_derivatives_and_reversed_intervals_are_unavailable():
    raw = trajectory()
    raw["points"][0]["velocities"] = []
    with pytest.raises(ValueError, match="derivative"):
        ContinuousMotion(raw, configuration())
    motion = ContinuousMotion(trajectory(), configuration())
    for start, end in ((-0.1, 1), (2, 1), (1, 0.5)):
        with pytest.raises(ValueError, match="interval"):
            motion.boxes(start, end)


def test_stationary_component_geometry_preserves_the_same_occupancy_meaning():
    case, models = composition_case(conflict=True)
    geometry = case["grounding_inputs"]["geometry"]["resources"]["ur5e-3"]
    geometry["footprint"] = [[-0.6, 0.6], [-0.1, 0.1], [-0.1, 0.1]]
    geometry["component_bounds"] = [
        {"id": "left", "bounds": [[0.4, 0.6], [-0.1, 0.1], [-0.1, 0.1]]},
        {"id": "right", "bounds": [[1.4, 1.6], [-0.1, 0.1], [-0.1, 0.1]]},
    ]
    clear = _analyze(case, models)
    assert clear["status"] == "allowed", clear
    geometry["component_bounds"][0]["bounds"][0] = [0.9, 1.1]
    conflict = _analyze(case, models)
    assert conflict["status"] == "held", conflict
    by_label = {
        label: {
            row["value"] for row in conflict["ap_evidence"] if row["descriptor"]["label"] == label
        }
        for label in ("ap001", "ap002")
    }
    assert all(False in values and True in values for values in by_label.values())


@pytest.mark.parametrize("change,expected", [
    ("inside", "allowed"), ("boundary", "allowed"), ("outside", "inconclusive"),
    ("missing_bound", "inconclusive"), ("missing_names", "inconclusive"),
    ("changed_names", "inconclusive"), ("nonfinite", "inconclusive"),
    ("incomplete", "inconclusive"), ("boolean", "inconclusive"),
])
def test_observed_initial_joints_use_only_the_registered_error_bound(change, expected):
    case, models = composition_case()
    observed = case["grounding_inputs"]["snapshot"]["resources"]["ur5e-4"]
    observed["joint_positions"] = [.004]
    observed["current_pose"][0] = .004
    for program in (case["grounding_inputs"]["programs"][0], case["event_start_choices"][0]["programs"][0]):
        config = program["step_results"][0]["model_evidence"]["continuous_motion"]["configuration"]
        if change != "missing_bound":
            config["joint_position_error"] = {"slide": .01}
    if change == "boundary":
        observed["joint_positions"] = [.01]
    elif change == "outside":
        observed["joint_positions"] = [.011]
    elif change == "missing_names":
        observed.pop("joint_names")
    elif change == "changed_names":
        observed["joint_names"] = ["different"]
    elif change == "nonfinite":
        observed["joint_positions"] = [float("inf")]
    elif change == "incomplete":
        observed["joint_positions"] = []
    elif change == "boolean":
        observed["joint_positions"] = [True]
    original = deepcopy(case)
    result = _analyze(case, models)
    assert result["status"] == expected, result
    assert case == original
    trajectory_start = case["grounding_inputs"]["programs"][0]["step_results"][0]["model_evidence"]["continuous_motion"]["joint_trajectory"]["points"][0]["positions"]
    assert trajectory_start == [0]
    if expected == "allowed":
        assert observed["joint_positions"] != trajectory_start
        assert result["completion_witness"] is not None
    else:
        assert result["completion_witness"] is None

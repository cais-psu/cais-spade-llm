from __future__ import annotations

"""Tests for the dual Gazebo adapter and streamlined ProductAgent UI."""

import ast
import json
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from cais_spade_llm.spec2primitives import spec2primitives_ui
from cais_spade_llm.spec2primitives.adapters.dual_gazebo import (
    DUAL_GAZEBO_NAME,
    read_dual_gazebo_started_at_ns,
    read_dual_gazebo_status,
    start_dual_gazebo,
    stop_dual_gazebo,
    GazeboResetProbe,
)


@pytest.mark.parametrize('world,passive,expected_direct', [
    ('table_spec2primitives.world', False, True),
    ('table_spec2primitives.world', True, False),
    ('table_recovery_framework.world', False, True),
    ('table_recovery_framework.world', True, False),
    ('single_table.world', False, False),
    ('single_table.world', True, False),
])
def test_icra_gripper_followers_preserve_other_simulation_modes(world, passive, expected_direct):
    """Change only follower control, preserving every other robot model element."""
    launch = Path(__file__).resolve().parents[3] / 'ros2/cais_lab_robotics/launch/xarm6_ur5e_gazebo.launch.py'
    module = ast.parse(launch.read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_configure_spec2primitives_gripper_followers')
    scope = {'ET': ET}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(launch), 'exec'), scope)
    configure = scope[function.name]
    root = ET.fromstring('<robot><joint name="arm"/><gazebo><plugin filename="other"/></gazebo></robot>')
    names = ['left_finger_joint', 'left_inner_knuckle_joint', 'right_outer_knuckle_joint',
             'right_finger_joint', 'right_inner_knuckle_joint']
    for name in names:
        plugin = ET.SubElement(ET.SubElement(root, 'gazebo'), 'plugin',
                               filename='libgazebo_mimic_joint_plugin.so')
        for key, value in {'joint': 'xarm6_drive_joint', 'mimicJoint': 'xarm6_' + name,
                           'multiplier': '1', 'offset': '0', 'maxEffort': '10', 'hasPID': ''}.items():
            ET.SubElement(plugin, key).text = value
    before = ET.tostring(root)
    configure(root, 'xarm6_', world_file=world, passive=passive)
    if not expected_direct:
        assert ET.tostring(root) == before
        return
    assert root.findall('.//hasPID') == []
    for plugin in root.findall('./gazebo/plugin')[1:]:
        ET.SubElement(plugin, 'hasPID').text = ''
    assert ET.tostring(root) == before
    root.findall('./gazebo/plugin')[-1].find('mimicJoint').text = 'xarm6_left_finger_joint'
    with pytest.raises(RuntimeError, match='exactly five'):
        configure(root, 'xarm6_', world_file=world, passive=passive)


@pytest.mark.parametrize(('world', 'passive', 'expected'), [
    ('table_spec2primitives.world', False, 0.0),
    ('table_spec2primitives.world', True, 0.85),
    ('table_recovery_framework.world', False, 0.0),
    ('table_recovery_framework.world', True, 0.85),
    ('single_table.world', False, 0.85),
    ('single_table.world', True, 0.85),
])
def test_icra_xarm_gripper_starts_open_without_changing_other_modes(
    world, passive, expected,
):
    """Start the active ICRA and recovery framework xArm gripper open."""
    launch = Path(__file__).resolve().parents[3] / 'ros2/cais_lab_robotics/launch/xarm6_ur5e_gazebo.launch.py'
    module = ast.parse(launch.read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_xarm_gripper_initial_position')
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(launch), 'exec'), scope)

    assert scope[function.name](world_file=world, passive=passive) == expected


class FakeRuntime:
    """In-memory runtime implementing only the Spec2Primitives adapter protocol."""

    def __init__(self) -> None:
        self.execution_mode, self.robot_env, self.system_running = "simulation", "gazebo", False
        self.statuses = {DUAL_GAZEBO_NAME: "stopped"}
        self.hardware_statuses = {
            "xarm6": {"overall": "stopped"},
            "ur5e": {"overall": "stopped"},
            "dual robots": {"overall": "stopped"},
        }
        self.start_error: str | None = None
        self.status_error: RuntimeError | None = None
        self.start_calls: list[str] = []
        self.stop_calls: list[str] = []

    def ros2_all_statuses(self) -> dict[str, str]:
        if self.status_error is not None:
            raise self.status_error
        return dict(self.statuses)

    def hardware_stack_status(self, robot: str) -> dict[str, object]:
        return dict(self.hardware_statuses[robot])

    def ros2_start(self, name: str) -> str | None:
        self.start_calls.append(name)
        return self.start_error

    def ros2_stop(self, name: str) -> None:
        self.stop_calls.append(name)


@pytest.mark.parametrize("fault", [None, "old_clock", "paused", "reset_clock", "stale", "missing", "nonfinite", "old_feedback"])
def test_reset_probe_requires_new_clock_and_fresh_both_robot_feedback(monkeypatch, fault):
    """A stopped/restarted process name alone never proves a new simulation baseline."""
    from cais_spade_llm.spec2primitives.adapters import dual_gazebo

    profile = {"joint_states_topics": ["/joint_states"], "readiness_timeout_sec": 2.0, "state_max_age_sec": 0.2}
    probe = GazeboResetProbe(profile, threading.Event())
    probe.context = SimpleNamespace(get_domain_id=lambda: 7)
    probe.description = '<robot><ros2_control><joint name="arm_a"/><joint name="arm_b"/></ros2_control></robot>'
    wall = 0.0
    old = {"publishers": {"/clock": ["old_clock"], "/joint_states": ["old_joints"]}, "services": []}
    publishers = {"/clock": ["old_clock" if fault == "old_clock" else "new_clock"], "/joint_states": ["new_joints"]}

    def spin(*, timeout_sec):
        nonlocal wall
        wall += timeout_sec
        clock = 1.0 if fault == "paused" else (0.5 if fault == "reset_clock" and wall > 0.15 else 1.0 + wall)
        probe.clock_ns, probe.clock_gid = int(clock * 1e9), publishers["/clock"][0]
        stamp = int((0.5 if fault == "stale" else clock) * 1e9)
        probe.joints["/joint_states"] = (SimpleNamespace(
            name=["arm_a"] if fault == "missing" else ["arm_a", "arm_b"],
            position=[0.1] if fault == "missing" else [float("nan") if fault == "nonfinite" else 0.1, 0.2],
            header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp // 10**9, nanosec=stamp % 10**9)),
        ), "old_joints" if fault == "old_feedback" else "new_joints", wall)

    monkeypatch.setattr(dual_gazebo, "time", SimpleNamespace(monotonic=lambda: wall))
    probe.executor = SimpleNamespace(spin_once=spin)
    monkeypatch.setattr(probe, "_endpoints", lambda: {"publishers": publishers, "services": []})
    if fault:
        with pytest.raises((RuntimeError, TimeoutError)):
            probe.wait_ready(old, ["arm_a", "arm_b"])
    else:
        baseline = probe.wait_ready(old, ["arm_a", "arm_b"])
        assert baseline["clock_publisher_gid"] == "new_clock"
        assert set(baseline["joint_feedback"]) == {"arm_a", "arm_b"}
        assert baseline["clock_ns"] > baseline["first_clock_ns"]


@pytest.mark.parametrize("fault", [None, "process", "endpoints", "interlock"])
def test_reset_probe_requires_stopped_process_and_absent_endpoints(monkeypatch, fault):
    from cais_spade_llm.spec2primitives.adapters import dual_gazebo

    runtime = FakeRuntime()
    if fault == "process":
        runtime.statuses[DUAL_GAZEBO_NAME] = "running"
    wall = 0.0
    probe = GazeboResetProbe({"stop_timeout_sec": 2}, threading.Event())
    monkeypatch.setattr(probe, "_subscribe", lambda: None)

    def spin(*, timeout_sec):
        nonlocal wall
        wall += timeout_sec
        if fault == "interlock" and wall > 0.2:
            runtime.hardware_statuses["ur5e"]["overall"] = "starting"

    monkeypatch.setattr(dual_gazebo, "time", SimpleNamespace(monotonic=lambda: wall))
    probe.executor = SimpleNamespace(spin_once=spin)
    monkeypatch.setattr(probe, "_endpoints", lambda: {
        "publishers": {"/clock": ["old"] if fault == "endpoints" else []}, "services": [],
    })
    if fault:
        with pytest.raises((RuntimeError, TimeoutError)):
            probe.wait_stopped(runtime)
    else:
        assert probe.wait_stopped(runtime)["process_status"] == "stopped"
        assert wall >= 1.0


def test_reads_stopped_and_running_status() -> None:
    runtime = FakeRuntime()

    assert read_dual_gazebo_status(runtime).state == "stopped"

    runtime.statuses[DUAL_GAZEBO_NAME] = "running"
    status = read_dual_gazebo_status(runtime)

    assert status.state == "running"
    assert status.blocked_reason is None


@pytest.mark.parametrize("fault", [None, "missing", "name", "pid", "stopped", "future", "nonfinite"])
def test_custody_launch_time_uses_existing_process_identity_without_ros(monkeypatch, fault):
    """Only matching live launch metadata can establish the history boundary."""
    from cais_spade_llm.spec2primitives.adapters import dual_gazebo

    runtime = FakeRuntime()
    launch = {"name": DUAL_GAZEBO_NAME, "pid": 123, "t0": 10.0}
    runtime._ros2_procs = {DUAL_GAZEBO_NAME: SimpleNamespace(
        pid=123, poll=lambda: 0 if fault == "stopped" else None,
    )}
    runtime._gazebo_launch_timing_snapshot = lambda: None if fault == "missing" else dict(launch)
    if fault == "name":
        launch["name"] = "gazebo_dual"
    elif fault == "pid":
        launch["pid"] = 456
    elif fault == "future":
        launch["t0"] = 30.0
    elif fault == "nonfinite":
        launch["t0"] = float("nan")
    monkeypatch.setattr(dual_gazebo, "time", SimpleNamespace(
        monotonic=lambda: 20.0, time_ns=lambda: 100_000_000_000,
    ))
    runtime.status_error = RuntimeError("This metadata read must not probe runtime status.")
    assert read_dual_gazebo_started_at_ns(runtime) == (90_000_000_000 if fault is None else 0)
    assert runtime.start_calls == runtime.stop_calls == []


@pytest.mark.parametrize("hardware_stack", ["xarm6", "ur5e", "dual robots"])
def test_hardware_stack_blocks_start(hardware_stack: str) -> None:
    runtime = FakeRuntime()
    runtime.hardware_statuses[hardware_stack]["overall"] = "running"

    error = start_dual_gazebo(runtime)

    assert error == "Blocked: hardware stack is running. Stop hardware first."
    assert runtime.start_calls == []


def test_start_uses_exact_dual_gazebo_name() -> None:
    runtime = FakeRuntime()

    assert start_dual_gazebo(runtime) is None
    assert runtime.start_calls == [DUAL_GAZEBO_NAME]


def test_duplicate_start_is_rejected() -> None:
    runtime = FakeRuntime()
    runtime.statuses[DUAL_GAZEBO_NAME] = "running"

    error = start_dual_gazebo(runtime)

    assert error == "Dual Robots (xArm6 + UR5e) is already running."
    assert runtime.start_calls == []


def test_runtime_start_error_is_returned_unchanged() -> None:
    runtime = FakeRuntime()
    runtime.start_error = "ROS2 workspace is not built yet."

    assert start_dual_gazebo(runtime) == runtime.start_error
    assert runtime.start_calls == [DUAL_GAZEBO_NAME]


def test_start_rechecks_hardware_state() -> None:
    runtime = FakeRuntime()
    assert read_dual_gazebo_status(runtime).blocked_reason is None
    runtime.hardware_statuses["dual robots"]["overall"] = "running"

    error = start_dual_gazebo(runtime)

    assert error == "Blocked: hardware stack is running. Stop hardware first."
    assert runtime.start_calls == []


def test_failed_fresh_state_check_never_starts() -> None:
    runtime = FakeRuntime()
    runtime.status_error = RuntimeError("fresh status unavailable")

    with pytest.raises(RuntimeError, match="fresh status unavailable"):
        start_dual_gazebo(runtime)

    assert runtime.start_calls == []


def test_stop_uses_exact_dual_gazebo_name() -> None:
    runtime = FakeRuntime()

    stop_dual_gazebo(runtime)

    assert runtime.stop_calls == [DUAL_GAZEBO_NAME]


def test_pa_ui_declares_the_streamlined_grounding_workspace() -> None:
    source = Path(spec2primitives_ui.__file__).read_text(encoding="utf-8")

    for exact_ui_term in (
        'label="product_requirement"',
        'value=""',
        '"Start ProductAgent"',
        'ui.label("ProductAgent Grounding")',
        'ui.label("ProductAgent")',
        'ui.label("ProductAgent Timeline")',
        'ui.label("Final Grounding Result")',
        'ui.label("Final Ontology")',
        'ui.label("Authoritative final interaction ABox")',
        'ui.expansion("Raw Turtle"',
        'ui.expansion("Developer diagnostics"',
        'ui.label("Failure details")',
        'ui.label("Recovered latest interaction")',
        'f"Interaction: {diagnostic_values.get',
        'f"Persisted path: {diagnostic_values.get',
    ):
        assert exact_ui_term in source

    assert 'requirement_input.on_value_change' in source
    assert '.props("outlined")' in source
    assert '.classes("w-full")' in source
    assert 'requirement_input.props("disable")' in source
    assert 'requirement_input.props(remove="disable")' in source
    assert 'start_button.on_click(_start_pa_interaction)' in source
    assert 'diagnostic_failure_card.set_visibility(False)' in source
    assert '"w-full border border-red-200 bg-red-50 shadow-none"' in source
    assert 'cancel_interaction_button.on_click(_cancel_clarification)' in source
    assert "start_pa_context_interaction(" in source
    assert "submit_pa_clarification_reply(" in source
    for removed_diagnostic in (
        "Detailed ProductAgent transcript",
        "Ordered interaction record",
        "Phase 4.1 Document Interpretation Diagnostic",
        "RGB-D Observation Processing",
        "No needed_context decision is available.",
    ):
        assert removed_diagnostic not in source
    assert "_render_ra" not in source
    assert '"RA Interaction"' not in source
    assert 'ui.badge("not connected")' not in source
    for removed_placeholder in (
        'ui.label("Assembly Plan")',
        'ui.label("Robot-independent assembly plan")',
        'ui.badge("Phase 5")',
        'ui.button("Open Assembly Plan"',
        'ui.label("Proposed Workflow")',
        'ui.label("RobotAgent execution")',
        'ui.label("User ↔ ProductAgent Messages")',
        '"No assembly plan is available."',
    ):
        assert removed_placeholder not in source


def test_phase_2_top_controls_are_full_width_and_responsive() -> None:
    source = Path(spec2primitives_ui.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    functions = {
        node.name: ast.get_source_segment(source, node)
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    dual_gazebo_source = functions["_render_dual_gazebo"]
    pa_source = functions["_render_pa_interaction"]
    render_source = functions["render"]

    assert dual_gazebo_source is not None
    assert pa_source is not None
    assert render_source is not None
    assert 'ui.card().classes("w-full border border-slate-200 shadow-sm")' in (
        dual_gazebo_source
    )
    assert '"w-full items-center justify-between gap-4 flex-wrap"' in (
        dual_gazebo_source
    )
    assert 'ui.card().classes("w-full border border-slate-200 shadow-sm")' in pa_source
    assert render_source.index("_render_dual_gazebo") < render_source.index(
        "_render_pa_interaction"
    )
    assert (
        "        _render_dual_gazebo(runtime.dual_gazebo)\n"
        "        _render_pa_interaction(runtime)"
    ) in render_source


@pytest.mark.parametrize(
    ("status", "available", "busy", "expected"),
    [
        ("ready_for_assignment", True, False, ("Capture RobotAgent context", True)),
        ("waiting_for_ra", True, False, ("Retry context capture", True)),
        ("waiting_for_ra", True, True, ("Retry context capture", False)),
        ("waiting_for_phase_4", True, False, ("Capture RobotAgent context", False)),
        ("blocked", True, False, ("Capture RobotAgent context", False)),
        ("context_captured", True, False, ("Refresh RobotAgent context", True)),
        ("context_captured", True, True, ("Refresh RobotAgent context", False)),
        ("ready_for_assignment", False, False, ("Capture RobotAgent context", False)),
    ],
)
def test_phase_5_1_action_state_is_fail_closed(
    status: str,
    available: bool,
    busy: bool,
    expected: tuple[str, bool],
) -> None:
    assert spec2primitives_ui._phase_5_1_action_state(
        status,
        activation_available=available,
        activation_busy=busy,
    ) == expected


def test_phase_5_1_ui_activates_only_from_the_persisted_interaction() -> None:
    source = Path(spec2primitives_ui.__file__).read_text(encoding="utf-8")

    assert 'ui.button(\n                    "Capture RobotAgent context"' in source
    assert 'label = "Retry context capture"' in source
    assert 'label = "Refresh RobotAgent context"' in source
    assert "async def _start_phase_5_1()" in source
    assert "interaction_root = interaction.get(\"interaction_root\")" in source
    assert 'ui.badge("temporary diagnostic")' not in source
    assert "await activate_selected_ra_context(" in source
    assert '"context_captured",' in source
    assert 'phase_5_elements["start_button"].on_click(_start_phase_5_1)' in source
    assert "runtime.robot_agent_context_runtime" in source


def test_phase_2_pa_start_requires_configured_grounding() -> None:
    source = Path(spec2primitives_ui.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    pa_source = next(
        ast.get_source_segment(source, node)
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_render_pa_interaction"
    )

    assert pa_source is not None
    assert "runtime.ontology_config is not None" in pa_source
    assert "runtime.grounding_runtime is not None" in pa_source
    assert "runtime.document_diagnostic_unavailable_reason" in pa_source
    assert '"grounding unavailable"' in pa_source
    assert "grounding_ready\n                and not action_state" in pa_source
    assert "not grounding_ready\n                or action_state" in pa_source


def test_pa_ui_imports_only_authorized_spec2primitives_agent_boundaries() -> None:
    source_path = Path(spec2primitives_ui.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert (
        "cais_spade_llm.spec2primitives.agents.pa" in imported_modules
    )
    assert (
        "cais_spade_llm.spec2primitives.adapters.ui_runtime" in imported_modules
    )
    assert "cais_spade_llm.spec2primitives.agents.ra" in imported_modules
    assert not any(
        module.startswith("cais_spade_llm.agents.")
        for module in imported_modules
    )


def test_spec2primitives_does_not_import_bridge() -> None:
    spec2primitives_root = Path(__file__).resolve().parents[1]

    for path in spec2primitives_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imported_modules.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert "cais_spade_llm.ui.bridge" not in imported_modules


def test_spec2primitives_production_code_has_no_ground_truth_reference() -> None:
    spec2primitives_root = Path(__file__).resolve().parents[1]
    forbidden_references = (
        "gazebo_msgs",
        "/gazebo/model_states",
        "/get_entity_state",
        "gazebo_camera_detector",
        "table_spec2primitives.world",
    )

    production_paths = [
        path
        for path in spec2primitives_root.rglob("*.py")
        if "tests" not in path.relative_to(spec2primitives_root).parts
    ]

    for path in production_paths:
        source = path.read_text(encoding="utf-8")
        for forbidden_reference in forbidden_references:
            assert forbidden_reference not in source, (
                f"{path} contains forbidden ground-truth reference "
                f"{forbidden_reference!r}"
            )


def test_program_details_preserve_context_without_a_draft_panel() -> None:
    """The compact program keeps source context available under expandable details."""
    context = {
        "target_feature": {"product_requirement": "assemble medium gear"},
        "selected_resource": {"resource_jid": "xarm6@localhost"},
        "ontology_projection": {
            "tbox_fingerprint": "tbox",
            "abox_fingerprint": "abox",
            "assertions": [],
        },
        "robot_state": {"held_part": None},
        "primitive_catalog": [],
        "grounded_context": {"typed_records": []},
    }
    with spec2primitives_ui.ui.column() as container:
        elements = spec2primitives_ui._render_phase_5_diagnostics()
    try:
        spec2primitives_ui._apply_primitive_composition_diagnostic(
            elements,
            {"status": "ready_for_composition", "composition_input": context},
            authoring_available=True,
        )
        assert elements["create_candidate_button"].text == "Compose Primitive Program"
        assert not elements["create_candidate_button"]._props.get("disable", False)
        assert "create_draft_button" not in elements
        assert "draft" not in elements
        assert elements["candidate_trace_expansion"].visible
        assert json.loads(elements["candidate_trace"].content)["composition_input"] == context
    finally:
        container.delete()

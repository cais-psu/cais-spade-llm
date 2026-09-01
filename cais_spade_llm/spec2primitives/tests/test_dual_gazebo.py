"""Tests for the dual Gazebo adapter and streamlined ProductAgent UI."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cais_spade_llm.spec2primitives import spec2primitives_ui
from cais_spade_llm.spec2primitives.adapters.dual_gazebo import (
    DUAL_GAZEBO_NAME,
    read_dual_gazebo_status,
    start_dual_gazebo,
    stop_dual_gazebo,
)


class FakeRuntime:
    """In-memory runtime implementing only the Spec2Primitives adapter protocol."""

    def __init__(self) -> None:
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


def test_reads_stopped_and_running_status() -> None:
    runtime = FakeRuntime()

    assert read_dual_gazebo_status(runtime).state == "stopped"

    runtime.statuses[DUAL_GAZEBO_NAME] = "running"
    status = read_dual_gazebo_status(runtime)

    assert status.state == "running"
    assert status.blocked_reason is None


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
        ("ready_for_assignment", True, False, ("Start Phase 5", True)),
        ("waiting_for_ra", True, False, ("Retry Phase 5", True)),
        ("waiting_for_ra", True, True, ("Retry Phase 5", False)),
        ("waiting_for_phase_4", True, False, ("Start Phase 5", False)),
        ("blocked", True, False, ("Start Phase 5", False)),
        ("context_captured", True, False, ("Restart Phase 5", True)),
        ("context_captured", True, True, ("Restart Phase 5", False)),
        ("ready_for_assignment", False, False, ("Start Phase 5", False)),
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

    assert 'ui.button(\n                    "Start Phase 5"' in source
    assert 'label = "Retry Phase 5"' in source
    assert 'label = "Restart Phase 5"' in source
    assert "async def _start_phase_5_1()" in source
    assert "interaction_root = interaction.get(\"interaction_root\")" in source
    assert (
        '"Starting or reusing only the exact RobotAgent selected by Phase 4."'
        in source
    )
    assert 'ui.badge("temporary diagnostic")' not in source
    assert "await activate_selected_ra_context(" in source
    assert '"context_captured",' in source
    assert 'phase_5_elements["start_button"].on_click(_start_phase_5_1)' in source
    assert "runtime.robot_agent_context_runtime" in source


@pytest.mark.parametrize(
    ("status", "available", "busy", "expected"),
    [
        ("ready_for_draft", True, False, True),
        ("ready_for_draft", True, True, False),
        ("ready_for_draft", False, False, False),
        ("waiting_for_context", True, False, False),
        ("draft_authored", True, False, False),
        ("unsupported", True, False, False),
        ("blocked", True, False, False),
    ],
)
def test_phase_5_2_action_state_is_fail_closed(
    status: str,
    available: bool,
    busy: bool,
    expected: bool,
) -> None:
    assert (
        spec2primitives_ui._phase_5_2_action_enabled(
            status,
            authoring_available=available,
            authoring_busy=busy,
        )
        is expected
    )


def test_phase_5_2_composition_evidence_summary_uses_exact_input_sections() -> None:
    summary = spec2primitives_ui._phase_5_2_composition_evidence_summary(
        {
            "task": {"product_requirement": "assemble medium gear"},
            "selected_resource": {"resource_jid": "xarm6@localhost"},
            "ontology_projection": {
                "tbox_fingerprint": "tbox-fingerprint",
                "abox_fingerprint": "abox-fingerprint",
                "assertions": [{"subject": "s"}, {"subject": "t"}],
            },
            "robot_state": {"controller_ready": True},
            "primitive_catalog": [{"primitive_symbol": "detect_parts"}],
            "grounded_context": {
                "typed_records": [
                    {
                        "record_type": "RobotFrameLocationRecord",
                        "record_ref": "products/grounding/location.json",
                    }
                ]
            },
        }
    )

    assert summary == {
        "assertion_count": 2,
        "primitive_count": 1,
        "typed_record_count": 1,
        "tbox_fingerprint": "tbox-fingerprint",
        "abox_fingerprint": "abox-fingerprint",
    }
    assert spec2primitives_ui._phase_5_2_waiting_view()["composition_input"] is None


def test_phase_5_2_ui_authors_only_from_the_active_persisted_context() -> None:
    source = Path(spec2primitives_ui.__file__).read_text(encoding="utf-8")

    assert 'ui.label("5.2 · RA-authored structural primitive draft")' in source
    assert '"Create Primitive Draft"' in source
    assert "async def _start_phase_5_2()" in source
    assert 'interaction_root = interaction.get("interaction_root")' in source
    assert 'current_diagnostic.status != "ready_for_draft"' in source
    assert "await author_primitive_program_draft(" in source
    assert "runtime.robot_agent_draft_runtime" in source
    assert 'ui.label("RA composition evidence")' in source
    assert '"COMPOSITION_INPUT delivered to RA"' in source
    assert 'composition_input = diagnostic.get("composition_input")' in source
    assert 'status in {"draft_authored", "unsupported"}' in source
    assert 'elements["composition_evidence_card"].set_visibility(' in source
    assert "json.dumps(\n            composition_input," in source
    assert (
        'phase_5_elements["create_draft_button"].on_click(_start_phase_5_2)'
        in source
    )


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

"""Experiment setting persistence, exact NIST bindings, and UI-only behavior."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from nicegui import context, core, ui
from nicegui.client import Client
from starlette.requests import Request

from cais_spade_llm.product.nominal import NominalProductContext
from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.ui import recovery_setup as settings
from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.recovery_setup import render_setup
from cais_spade_llm.ui.pages import recovery_framework, recovery_run, safety


@pytest.fixture
def project(tmp_path):
    setup = settings.default_setup()
    for field in (
        "selected_product",
        "selected_product_order_file",
        "product_geometry_file",
        "scene_file",
        "selected_safety_file",
        "recovery_experiment_settings_file",
    ):
        relative = setup[field]
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def slippage(setup, root, resource="ur5e-3", part="KET4_Square_4mm"):
    models = settings.validate_setup({**setup, "failure_scenario": None}, root=root)["models"]
    failure = settings.slippage_example(models, resource, part)
    failure["drop_pose"] = {"x": 0.0, "y": -0.2, "z": 1.04}
    return failure


def snapshot(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_defaults_and_save_leave_all_source_definitions_unchanged(project):
    before = snapshot(project)
    path = project / settings.SETUP_RELATIVE
    setup = settings.load_setup(path, root=project)
    assert len(setup["permitted_resources"]) == 12
    assert setup["failure_scenario"] is None
    assert setup["execution_mode"] == "simulation"
    assert not path.exists()
    assert settings.startup_block_reason(setup, root=project) == ""
    settings.save_setup(setup, path, root=project)
    assert settings.load_setup(path, root=project) == setup
    after = snapshot(project)
    after.pop(settings.SETUP_RELATIVE)
    assert before == after
    assert "password" not in path.read_text()


@pytest.mark.parametrize(
    "part",
    [
        "KET4_Square_4mm",
        "KET8_Square_8mm",
        "KET12_Square_12mm",
        "KET16_Square_16mm",
        "RGOCG4-50_Round_4mm",
        "RGOCG8-50_8mm",
        "RGOCG12-50_12mm",
        "RGOCG16-50_16mm",
        "gear_small",
        "gear_medium",
        "gear_large",
    ],
)
def test_all_exact_nist_parts_can_be_saved_with_their_assembly_resource(project, part):
    setup = settings.default_setup(project)
    resource = "ur5e-4" if part.startswith("gear_") else "ur5e-3"
    setup["failure_scenario"] = slippage(setup, project, resource, part)
    path = project / settings.SETUP_RELATIVE
    settings.save_setup(setup, path, root=project)
    assert settings.load_setup(path, root=project)["failure_scenario"]["part_name"] == part
    assert "Part slippage: execution not integrated" in settings.startup_block_reason(
        setup, root=project
    )


def test_registered_part_binding_is_not_a_resource_eligibility_list(project):
    setup = settings.default_setup(project)
    setup["failure_scenario"] = slippage(setup, project, "ur5e-3", "gear_large")
    path = project / settings.SETUP_RELATIVE
    settings.save_setup(setup, path, root=project)
    assert settings.load_setup(path, root=project)["failure_scenario"]["part_name"] == "gear_large"


@pytest.mark.parametrize(
    "field,value",
    [
        ("part_name", "LG"),
        ("part_name", "KET4_square_4mm"),
        ("checkpoint", "during_motion"),
        ("resource_id", "ur5e"),
        ("event_name", "place_release"),
        ("event_id", True),
        ("mode", "always"),
        ("parameter_bindings", {}),
        ("drop_pose", {"x": float("nan"), "y": 0, "z": 1}),
        ("drop_pose", {"x": 0, "y": 0}),
        ("orientation_quat", {"qx": 0, "qy": 0, "qz": 0, "qw": 0}),
    ],
)
def test_invalid_slippage_never_replaces_saved_setup(project, field, value):
    setup = settings.default_setup(project)
    path = project / settings.SETUP_RELATIVE
    settings.save_setup(setup, path, root=project)
    before = path.read_bytes()
    setup["failure_scenario"] = slippage(setup, project)
    setup["failure_scenario"][field] = value
    with pytest.raises(ValueError):
        settings.save_setup(setup, path, root=project)
    assert path.read_bytes() == before


def test_exact_machine_roles_and_subset_orders_restrict_slippage(project):
    setup = settings.default_setup(project)
    inputs = settings.validate_setup(setup, root=project)
    assert "KET4_Square_4mm" in settings.eligible_parts(
        inputs["models"]["ur5e-1"], inputs["selected_parts"]
    )
    assert "RGOCG4-50_Round_4mm" in settings.eligible_parts(
        inputs["models"]["ur5e-1"], inputs["selected_parts"]
    )
    assert "KET4_Square_4mm" in settings.eligible_parts(
        inputs["models"]["ur5e-2"], inputs["selected_parts"]
    )
    failure = slippage(setup, project)
    order_path = project / setup["selected_product_order_file"]
    order = json.loads(order_path.read_text())
    order["parts"] = ["gear_large"]
    order_path.write_text(json.dumps(order))
    setup["failure_scenario"] = failure
    with pytest.raises(ValueError, match="exact selected NIST part"):
        settings.validate_setup(setup, root=project)


def test_permitted_resources_and_another_held_part_are_checked(project):
    setup = settings.default_setup(project)
    failure = slippage(setup, project)
    failure["additional_condition"] = {"resource_id": "ur5e-4", "part_name": "gear_small"}
    setup["failure_scenario"] = failure
    settings.validate_setup(setup, root=project)
    failure["additional_condition"]["part_name"] = "KET4_Square_4mm"
    with pytest.raises(ValueError, match="another exact selected part"):
        settings.validate_setup(setup, root=project)
    failure["additional_condition"] = None
    setup["permitted_resources"].remove("ur5e-3")
    with pytest.raises(ValueError, match="Failure resource must be permitted"):
        settings.validate_setup(setup, root=project)
    setup["failure_scenario"] = None
    settings.validate_setup(setup, root=project)
    assert settings.startup_block_reason(setup, root=project) == ""


@pytest.mark.parametrize(
    "scenario,resource",
    [
        ("Conveyor breakdown", "Conveyor"),
        ("ur5e-1 breakdown", "ur5e-1"),
        ("Machining breakdown during part processing", "M1"),
    ],
)
def test_documented_resource_scenarios_save_as_unintegrated(project, scenario, resource):
    setup = settings.default_setup(project)
    setup["failure_scenario"] = {
        "scenario": scenario,
        "resource_id": resource,
        "mode": "once",
        "checkpoint": "before_execute",
    }
    settings.save_setup(setup, project / settings.SETUP_RELATIVE, root=project)
    assert "execution not integrated" in settings.startup_block_reason(setup, root=project)


@pytest.mark.parametrize(
    "field,value",
    [
        ("selected_product_order_file", "missing.json"),
        ("product_geometry_file", "another.json"),
        ("selected_safety_file", "missing.txt"),
        ("permitted_resources", ["ur5e"]),
        ("permitted_resources", ["M1", "M1"]),
        ("execution_mode", "dry_run"),
        ("runtime_recovery_mode", "unknown"),
        ("recovery_experiment_settings", {}),
        ("runtime_recovery_archive_path", "missing.json"),
    ],
)
def test_invalid_references_are_not_substituted(project, field, value):
    setup = settings.default_setup(project)
    setup[field] = value
    assert settings.startup_block_reason(setup, root=project).startswith(
        "Invalid experiment setup:"
    )
    assert setup[field] == value


def _capture_buttons(monkeypatch):
    buttons = {}
    original = ui.button.on_click

    def capture(button, callback):
        buttons[button.text] = callback
        return original(button, callback)

    monkeypatch.setattr(ui.button, "on_click", capture)
    return buttons


def _element(client, cls, label):
    return next(
        element
        for element in client.elements.values()
        if isinstance(element, cls) and element._props.get("label") == label
    )


def test_simulation_controls_poll_without_dispatch_or_settings_changes(project, monkeypatch):
    import time
    from cais_spade_llm.ui.components.simulation_controls import render_simulation_controls

    path = project / settings.SETUP_RELATIVE
    settings.save_setup(settings.default_setup(project), path, root=project)
    initial_simulation = settings.load_setup(path, root=project)['simulation']
    saved_bytes = path.read_bytes()
    monkeypatch.setattr(settings, 'SETUP_PATH', path)
    bridge = _run_bridge()
    bridge.simulation_environment_running.return_value = False
    bridge.ros2_start.return_value = bridge.ros2_stop.return_value = None
    buttons = _capture_buttons(monkeypatch)
    callbacks = []
    client = Client(context.client.page)

    async def check():
        with client:
            render_simulation_controls(bridge, lambda period, callback: callbacks.append((period, callback)), lambda: False)
            assert callbacks[0][0] == 2.0
            await callbacks[0][1]()
            bridge.ros2_start.assert_not_called()
            bridge.ros2_stop.assert_not_called()
            bridge.start_system.assert_not_called()
            assert not any(isinstance(e, (ui.select, ui.checkbox)) for e in client.elements.values())
            assert set(buttons) == {'Open RViz', 'Close RViz', 'Stop Simulation'}
            labels = [e.text for e in client.elements.values() if isinstance(e, ui.label)]
            assert 'Simulation is stopped.' in labels
            assert not any('settings apply' in text.lower() for text in labels)
            bridge.simulation_environment_running.return_value = True
            bridge.ros2_exec.return_value = (True, json.dumps({
                'observed_at_unix': time.time(), 'real_time_factor': .4,
                'settings': initial_simulation,
            }))
            await callbacks[0][1]()
            labels = [e.text for e in client.elements.values() if isinstance(e, ui.label)]
            assert any('Measured simulation speed: 0.40×' in text for text in labels)
            assert not any('Requested' in text for text in labels)
            assert path.read_bytes() == saved_bytes
            bridge.ros2_start.assert_not_called()
            bridge.ros2_stop.assert_not_called()
            await buttons['Open RViz']()
            bridge.ros2_start.assert_called_once_with('recovery_rviz')
            await buttons['Close RViz']()
            bridge.ros2_stop.assert_called_once_with('recovery_rviz')
            bridge.ros2_exec.return_value = (True, '{}')
            await callbacks[0][1]()
            assert any('observation unavailable' in e.text for e in client.elements.values() if isinstance(e, ui.label))
            assert path.read_bytes() == saved_bytes
        client.delete()

    asyncio.run(check())


def test_complete_setup_form_only_saves_explicitly_and_preserves_nist_bindings(
    project, monkeypatch
):
    bridge = SimpleNamespace(system_running=False, _starting=False, _stopping=False)
    before_bridge = vars(bridge).copy()
    before = snapshot(project)
    buttons = _capture_buttons(monkeypatch)
    monkeypatch.setattr(
        NominalProductContext,
        "plan",
        Mock(side_effect=AssertionError("planning during settings editing")),
    )
    timer = Mock(side_effect=AssertionError("setup added a polling path"))
    monkeypatch.setattr(ui, "timer", timer)
    client = Client(context.client.page)

    async def check_page():
        monkeypatch.setattr(core, "loop", asyncio.get_running_loop())
        with client:
            render_setup(bridge, root=project)
            resources = next(
                element for element in client.elements.values() if isinstance(element, ui.table)
            )
            assert len(resources.rows) == 12
            assert len(resources.selected) == 12
            buttons["Example: ur5e-3 / KET4_Square_4mm"]()
            await asyncio.sleep(0)
            assert "gear_large" in _element(client, ui.select, "NIST part").options
            for axis, value in {"x": 0.0, "y": -0.2, "z": 1.04}.items():
                _element(client, ui.number, f"drop_pose.{axis}").value = value
            assert snapshot(project) == before
            assert vars(bridge) == before_bridge
            buttons["Save setup"]()
            path = project / settings.SETUP_RELATIVE
            saved = settings.load_setup(path, root=project)
            assert saved["failure_scenario"]["part_name"] == "KET4_Square_4mm"
            assert saved["failure_scenario"]["drop_pose"] == {"x": 0.0, "y": -0.2, "z": 1.04}
            buttons["Example: ur5e-4 / gear_large"]()
            await asyncio.sleep(0)
            assert "KET4_Square_4mm" in _element(client, ui.select, "NIST part").options
            buttons["Reload saved setup"]()
            await asyncio.sleep(0)
            assert _element(client, ui.select, "NIST part").value == "KET4_Square_4mm"
            assert settings.load_setup(path, root=project) == saved
            _element(client, ui.select, "Failure resource").value = "ur5e-2"
            await asyncio.sleep(0)
            parts = _element(client, ui.select, "NIST part")
            assert "LG" not in parts.options
            assert "KET4_Square_4mm" in parts.options
            parts.value = "RGOCG4-50_Round_4mm"
            await asyncio.sleep(0)
            task = _element(client, ui.select, "Task")
            task.value = next(iter(task.options))
            bridge.system_running = True
            buttons["Save setup"]()
            assert settings.load_setup(path, root=project) == saved
            bridge.system_running = False
            buttons["Save setup"]()
            assert (
                settings.load_setup(path, root=project)["failure_scenario"]["resource_id"]
                == "ur5e-2"
            )
            after = snapshot(project)
            after.pop(settings.SETUP_RELATIVE)
            assert before == after
            assert vars(bridge) == before_bridge
            timer.assert_not_called()
            await asyncio.sleep(0)
        client.delete()

    asyncio.run(check_page())


def _run_bridge():
    bridge = MagicMock(spec=SystemBridge)
    for name, value in {
        "system_running": False,
        "_starting": False,
        "_stopping": False,
        "execution_mode": "simulation",
        "robot_env": "gazebo",
        "last_error": "",
        "cca": None,
    }.items():
        setattr(bridge, name, value)
    for name in (
        "get_agent_statuses",
        "get_plan_nodes",
        "get_current_task_dag_nodes",
        "get_plan_safety_alerts",
        "get_runtime_recoveries",
        "get_execution_timeline",
        "get_safety_rules",
    ):
        getattr(bridge, name).return_value = []
    for name in ("get_robot_states", "get_task_states", "get_safety_state", "get_environment_capabilities"):
        getattr(bridge, name).return_value = {}
    bridge.get_environment_capabilities_revision.return_value = None
    bridge.simulation_environment_running.return_value = True
    bridge.ros2_exec.return_value = (False, 'Performance observation unavailable')
    bridge.passive_digital_twin_environment_running.return_value = False
    bridge.simulation_start_ready.return_value = (True, "")
    bridge.ros2_proc_status.return_value = "running"
    return bridge


@pytest.mark.parametrize("failure", [False, True])
def test_run_reads_saved_setup_and_blocks_unsupported_settings_before_dispatch(
    project, monkeypatch, failure
):
    setup = settings.default_setup(project)
    if failure:
        setup["failure_scenario"] = slippage(setup, project)
    else:
        setup["permitted_resources"].remove("KMR")
    path = project / settings.SETUP_RELATIVE
    settings.save_setup(setup, path, root=project)
    monkeypatch.setattr(settings, "SETUP_PATH", path)
    original_gate = settings.startup_block_reason
    monkeypatch.setattr(
        settings, "startup_block_reason", lambda value, **_kwargs: original_gate(value, root=project)
    )
    original_validation = settings.StartupValidation
    monkeypatch.setattr(settings, "StartupValidation", lambda: original_validation(root=project))
    bridge = _run_bridge()
    buttons = _capture_buttons(monkeypatch)
    callbacks = []
    monkeypatch.setattr(
        ui, "timer", lambda interval, callback, **kwargs: callbacks.append(callback) or Mock()
    )
    before = snapshot(project)
    client = Client(context.client.page)
    async def check_controls():
        with client:
            recovery_run.render(bridge)
            for callback in callbacks:
                await callback()
            await buttons["Refresh saved setup"]()
            await buttons["Start System"]()
            for _ in range(100):
                if failure or bridge.start_system.await_count:
                    break
                await asyncio.sleep(0.01)
            if not failure:
                await buttons["Stop System"]()
            if failure:
                bridge.start_system.assert_not_called()
            else:
                bridge.start_system.assert_awaited_once()
            bridge.ros2_start.assert_not_called()
            if failure:
                bridge.set_runtime_recovery_mode.assert_not_called()
                bridge.set_runtime_recovery_archive_selection.assert_not_called()
            assert snapshot(project) == before
            assert not failure or any(
                "not integrated" in str(getattr(element, "text", ""))
                for element in client.elements.values()
            )
    asyncio.run(check_controls())
    client.delete()


def test_start_rechecks_saved_settings_and_stop_keeps_existing_dispatch(project, monkeypatch):
    setup = settings.default_setup(project)
    path = project / settings.SETUP_RELATIVE
    settings.save_setup(setup, path, root=project)
    monkeypatch.setattr(settings, "SETUP_PATH", path)
    original_gate = settings.startup_block_reason
    monkeypatch.setattr(
        settings, "startup_block_reason", lambda value, **_kwargs: original_gate(value, root=project)
    )
    monkeypatch.setattr(ui, "timer", lambda *args, **kwargs: Mock())
    monkeypatch.setattr(recovery_run, "_check_prerequisites", Mock(return_value=True))
    original_validation = settings.StartupValidation
    monkeypatch.setattr(settings, "StartupValidation", lambda: original_validation(root=project))
    bridge = _run_bridge()
    bridge.consume_notice.return_value = ""
    buttons = _capture_buttons(monkeypatch)
    client = Client(context.client.page)

    async def started():
        bridge.system_running = True

    async def stopped():
        bridge.system_running = False

    bridge.start_system.side_effect = started
    bridge.stop_system.side_effect = stopped

    async def check_controls():
        with client:
            recovery_run.render(bridge)
            bridge.start_system.assert_not_called()
            setup["failure_scenario"] = slippage(setup, project)
            settings.save_setup(setup, path, root=project)
            await buttons["Start System"]()
            bridge.start_system.assert_not_called()
            setup["failure_scenario"] = None
            setup["runtime_recovery_mode"] = "manual"
            settings.save_setup(setup, path, root=project)
            await buttons["Start System"]()
            for _ in range(100):
                if bridge.start_system.await_count:
                    break
                await asyncio.sleep(0.01)
            bridge.start_system.assert_awaited_once()
            assert bridge.selected_product == setup["selected_product"]
            assert bridge.selected_product_order_file == setup["selected_product_order_file"]
            bridge.set_runtime_recovery_mode.assert_called_once_with("manual")
            summary = next(item for item in client.elements.values() if isinstance(item, ui.table))
            assert summary.rows == settings.setup_summary(setup)
            await buttons["Stop System"]()
            bridge.stop_system.assert_awaited_once()
            bridge.ros2_start.assert_not_called()

    asyncio.run(check_controls())
    client.delete()


@pytest.mark.parametrize("content", ["{", '{"schema_version": 99}', '{"schema_version": 1}'])
def test_malformed_saved_settings_display_an_error_without_replacing_them(
    project, monkeypatch, content
):
    path = project / settings.SETUP_RELATIVE
    path.write_text(content)
    before = snapshot(project)
    monkeypatch.setattr(ui, "timer", Mock(side_effect=AssertionError("unexpected polling")))
    client = Client(context.client.page)
    with client:
        render_setup(SimpleNamespace(), root=project)
        assert any(
            "could not be loaded" in str(getattr(item, "text", ""))
            for item in client.elements.values()
        )
    client.delete()
    assert snapshot(project) == before


def test_setup_tab_direct_link_keeps_one_draft_and_no_polling(project, monkeypatch):
    monkeypatch.setattr(recovery_framework, "_ROOT", project)
    monkeypatch.setattr(recovery_framework, "_PAPER", project / "missing.md")
    monkeypatch.setattr(recovery_run, "render", Mock())
    monkeypatch.setattr(ui, "timer", Mock(side_effect=AssertionError("unexpected polling")))
    request = Request(
        {"type": "http", "path": "/recovery-framework", "headers": [], "query_string": b"tab=setup"}
    )
    client = Client(context.client.page, request=request)
    before = snapshot(project)
    with client:
        recovery_framework.render(SimpleNamespace())
        tabs = next(item for item in client.elements.values() if isinstance(item, ui.tabs))
        assert tabs.value == "setup"
        mode = _element(client, ui.select, "Mode")
        mode.value = "physical"
        tabs.value = "run"
        tabs.value = "setup"
        assert _element(client, ui.select, "Mode") is mode
        assert mode.value == "physical"
        assert (
            len(
                [
                    item
                    for item in client.elements.values()
                    if isinstance(item, ui.button) and item.text == "Save setup"
                ]
            )
            == 1
        )
        assert snapshot(project) == before
    client.delete()


def test_complete_safety_page_keeps_its_editors_and_verification_controls(project, monkeypatch):
    bridge = _run_bridge()
    bridge.evaluate_safety_intent_approval.return_value = {"approved": False}
    bridge.get_safety_rule_preview.return_value = {"available": False, "rules": []}
    monkeypatch.setattr(safety, "_SAFETY_DIR", project / "cais_spade_llm/specification/safety")
    polls = []
    monkeypatch.setattr(
        ui,
        "timer",
        lambda interval, callback, **kwargs: polls.append((interval, callback)) or Mock(),
    )
    client = Client(context.client.page)
    before = snapshot(project)
    with client:
        safety.render(bridge)
        for _, callback in polls:
            callback()
        texts = [getattr(item, "text", "") for item in client.elements.values()]
        assert "Safety Requirements" in texts
        assert "Verify Safety" in texts
        assert "Generated Rule Interpretation" in texts
        assert [interval for interval, _ in polls] == [0.1, 2.0]
        bridge.generate_safety_rule_preview.assert_not_called()
        bridge.approve_safety_intent.assert_not_called()
        bridge.start_system.assert_not_called()
        assert snapshot(project) == before
    client.delete()


@pytest.mark.parametrize("field", ["selected_product", "selected_product_order_file", "product_geometry_file", "scene_file", "selected_safety_file", "recovery_experiment_settings_file"])
def test_startup_validation_caches_unchanged_inputs_and_invalidates_dependencies(project, monkeypatch, field):
    setup = settings.default_setup(project)
    path = project / settings.SETUP_RELATIVE
    settings.save_setup(setup, path, root=project)
    monkeypatch.setattr(settings, "SETUP_PATH", path)
    validate = Mock(wraps=settings.startup_block_reason)
    monkeypatch.setattr(settings, "startup_block_reason", validate)
    cache = settings.StartupValidation(root=project)
    assert cache.read()[1] == ""
    assert cache.read()[1] == ""
    assert validate.call_count == 1
    changed = project / setup[field]
    changed.write_text(changed.read_text() + "\n")
    cache.read()
    assert validate.call_count == 2
    cache.read(force=True)
    assert validate.call_count == 3


@pytest.mark.parametrize('delivery, ready, reason, expected_reason', [
    (True, True, 'Perception is still warming up: /detect_all.', ''),
    (False, True, 'Perception is still warming up: /detect_all.',
     'Perception is still warming up: /detect_all.'),
    (True, False, 'Perception is still warming up: /detect_all.',
     'Perception is still warming up: /detect_all.'),
    (True, False, 'Waiting for /DETACHLINK', 'Waiting for /DETACHLINK'),
    (True, True, 'Other readiness notice', 'Other readiness notice'),
])
def test_delivery_prerequisites_only_omit_optional_perception_note(
    delivery, ready, reason, expected_reason
):
    bridge = _run_bridge()
    bridge.simulation_start_ready.return_value = (ready, reason)
    status = recovery_run._read_prerequisites(bridge, 'simulation', delivery=delivery)
    assert status['ready'] is ready
    assert status['reason'] == expected_reason
    assert status['gazebo_running'] is True
    bridge.simulation_start_ready.assert_called_once_with()
    bridge.ros2_start.assert_not_called()
    bridge.start_system.assert_not_called()


def test_run_controls_enable_after_readiness_and_reuse_unchanged_widgets(project, monkeypatch):
    setup = settings.default_setup(project)
    path = project / settings.SETUP_RELATIVE
    settings.save_setup(setup, path, root=project)
    monkeypatch.setattr(settings, "SETUP_PATH", path)
    original = settings.StartupValidation
    monkeypatch.setattr(settings, "StartupValidation", lambda: original(root=project))
    validate = Mock(wraps=settings.startup_block_reason)
    monkeypatch.setattr(settings, "startup_block_reason", validate)
    bridge = _run_bridge()
    bridge.simulation_start_ready.return_value = (False, "Waiting for /DETACHLINK")
    polls = []
    monkeypatch.setattr(ui, "timer", lambda interval, callback, **kwargs: polls.append(callback) or Mock())
    client = Client(context.client.page)

    async def check():
        with client:
            recovery_run.render(bridge)
            for refresh in polls:
                await refresh()
            start = next(e for e in client.elements.values() if isinstance(e, ui.button) and e.text == "Start System")
            simulation = next(e for e in client.elements.values() if isinstance(e, ui.button) and e.text == "Start Simulation")
            assert not start.enabled
            assert any(getattr(e, "text", "") == "Waiting for /DETACHLINK" for e in client.elements.values())
            assert start.parent_slot.parent == simulation.parent_slot.parent
            toolbar = start.parent_slot.parent.parent_slot.parent
            assert toolbar._style["position"] == "sticky"
            bridge.simulation_start_ready.return_value = (True, "")
            for refresh in polls:
                await refresh()
            assert start.enabled
            graph = next(e for e in client.elements.values() if isinstance(e, ui.mermaid))
            graph.update = Mock(wraps=graph.update)
            elements = set(client.elements)
            for _ in range(3):
                for refresh in polls:
                    await refresh()
            assert set(client.elements) == elements
            graph.update.assert_not_called()
            assert validate.call_count == 1
            bridge.get_execution_timeline.assert_not_called()
            bridge.simulation_environment_running.return_value = False
            for refresh in polls:
                await refresh()
            assert not start.enabled and simulation.enabled
            assert any(getattr(e, "text", "") == "Gazebo is not running." for e in client.elements.values())

    asyncio.run(check())
    client.delete()


def test_selected_order_summary_refreshes_when_order_contents_change(project, monkeypatch):
    from cais_spade_llm.recovery_framework import delivery

    setup = settings.default_setup(project)
    setup['selected_product_order_file'] = str(delivery.ORDER_PATH.relative_to(ROOT))
    order_path = project / setup['selected_product_order_file']
    shutil.copyfile(delivery.ORDER_PATH, order_path)
    path = project / settings.SETUP_RELATIVE
    settings.save_setup(setup, path, root=project)
    monkeypatch.setattr(settings, 'SETUP_PATH', path)
    validation = settings.StartupValidation(root=project)
    assert validation.read()[2]
    assert 'Parts: KET4_Square_4mm' in validation.order_summary
    assert 'Destination: M1' in validation.order_summary
    order = json.loads(order_path.read_text())
    order['objective'] = 'Updated delivery objective'
    order_path.write_text(json.dumps(order))
    assert validation.read()[2]
    assert 'Objective: Updated delivery objective' in validation.order_summary
    order_path.write_text('{')
    assert validation.read()[1]
    assert validation.order_summary == ''


@pytest.mark.parametrize('status', ['execution_unavailable', 'blocked'])
def test_run_keeps_execution_blocker_visible_without_stopping_agents(monkeypatch, status):
    from cais_spade_llm.resources.environment_models import build_environment_models

    setup = settings.default_setup()
    monkeypatch.setattr(settings, 'load_setup', lambda *args, **kwargs: deepcopy(setup))
    monkeypatch.setattr(recovery_run, 'prepare_environment_start', Mock())
    bridge = _run_bridge()
    bridge.consume_notice.return_value = ''
    models = build_environment_models(settings.validate_setup(setup)['scene'])
    task = {'resource_id': 'KMR', 'event_name': 'pick_part'}
    outcome = {'status': status, 'reason': 'Controller completion evidence unavailable'}
    if status == 'execution_unavailable':
        outcome['execution_unavailable'] = [task]
    runtime = {'models': models, 'outcome': outcome,
               'environment_model': {'selected_path': [task]}}
    bridge.get_environment_capabilities.side_effect = lambda: deepcopy(runtime) if bridge.system_running else {}
    bridge.get_environment_capabilities_revision.side_effect = lambda: ('run', 1) if bridge.system_running else None
    bridge.get_robot_states.side_effect = lambda: (
        {'KMR': {'resource_state': 'idle', 'resource_location': 'Storage'}} if bridge.system_running else {}
    )

    async def started():
        bridge.system_running = True

    async def stopped():
        bridge.system_running = False

    bridge.start_system = AsyncMock(side_effect=started)
    bridge.stop_system = AsyncMock(side_effect=stopped)
    buttons = _capture_buttons(monkeypatch)
    polls = []
    monkeypatch.setattr(ui, 'timer', lambda interval, callback, **kwargs: polls.append(callback) or Mock())
    client = Client(context.client.page)

    async def scenario():
        with client:
            recovery_run.render(bridge)
            bridge.start_system.assert_not_called()
            await buttons['Start System']()
            for _ in range(200):
                await asyncio.sleep(.01)
                if any(getattr(element, 'text', '') == 'Agents started.' for element in client.elements.values()):
                    break
            for _ in range(2):
                for refresh in polls:
                    await refresh()
                texts = [getattr(element, 'text', '') for element in client.elements.values()]
                assert 'Agents started.' in texts
                assert 'System started successfully.' not in texts
                assert 'Execution status' in texts and status in texts
                assert outcome['reason'] in texts
                assert 'KMR' in texts
                rows = [row for element in client.elements.values() if isinstance(element, ui.table) for row in element.rows]
                assert any(row.get('resource_id') == 'KMR' and row.get('event_name') == 'pick_part' for row in rows)
                assert bridge.system_running
                bridge.stop_system.assert_not_called()
            await buttons['Stop System']()
            for refresh in polls:
                await refresh()
            texts = [getattr(element, 'text', '') for element in client.elements.values()]
            assert status not in texts and outcome['reason'] not in texts
            assert 'KMR' not in texts
            assert 'No resources available — start the system first' in texts
    asyncio.run(scenario())
    client.delete()


def test_run_slow_validation_yields_and_duplicate_start_does_not_dispatch_twice(project, monkeypatch):
    setup = settings.default_setup(project)
    path = project / settings.SETUP_RELATIVE
    settings.save_setup(setup, path, root=project)
    monkeypatch.setattr(settings, "SETUP_PATH", path)
    original = settings.StartupValidation
    monkeypatch.setattr(settings, "StartupValidation", lambda: original(root=project))
    entered, release = threading.Event(), threading.Event()
    original_gate = settings.startup_block_reason

    def slow_gate(value, **kwargs):
        entered.set()
        assert release.wait(2)
        return original_gate(value, **kwargs)

    monkeypatch.setattr(settings, "startup_block_reason", slow_gate)
    monkeypatch.setattr(ui, "timer", lambda *args, **kwargs: Mock())
    buttons = _capture_buttons(monkeypatch)
    bridge = _run_bridge()
    bridge.consume_notice.return_value = ""
    client = Client(context.client.page)

    async def check():
        with client:
            recovery_run.render(bridge)
            start = asyncio.create_task(buttons["Start System"]())
            assert await asyncio.to_thread(entered.wait, 1)
            heartbeat = time.monotonic()
            await asyncio.sleep(0.02)
            assert time.monotonic() - heartbeat < 0.2
            await buttons["Start System"]()
            bridge.start_system.assert_not_called()
            release.set()
            await start
            for _ in range(100):
                if bridge.start_system.await_count:
                    break
                await asyncio.sleep(0.01)
            bridge.start_system.assert_awaited_once()
            await buttons["Stop System"]()

    asyncio.run(check())
    client.delete()


def test_stop_during_validation_cancels_start_and_allows_retry(project, monkeypatch):
    setup = settings.default_setup(project)
    path = project / settings.SETUP_RELATIVE
    settings.save_setup(setup, path, root=project)
    monkeypatch.setattr(settings, "SETUP_PATH", path)
    original = settings.StartupValidation
    monkeypatch.setattr(settings, "StartupValidation", lambda: original(root=project))
    entered, release = threading.Event(), threading.Event()
    original_gate = settings.startup_block_reason

    def check_gate(value, **kwargs):
        entered.set()
        assert release.wait(2)
        return original_gate(value, **kwargs)

    monkeypatch.setattr(settings, "startup_block_reason", check_gate)
    monkeypatch.setattr(ui, "timer", lambda *args, **kwargs: Mock())
    bridge = _run_bridge()
    bridge.consume_notice.return_value = ""
    buttons = _capture_buttons(monkeypatch)
    client = Client(context.client.page)

    async def check():
        with client:
            recovery_run.render(bridge)
            starting = asyncio.create_task(buttons["Start System"]())
            assert await asyncio.to_thread(entered.wait, 1)
            await buttons["Stop System"]()
            release.set()
            await starting
            bridge.start_system.assert_not_called()
            await buttons["Start System"]()
            for _ in range(100):
                if bridge.start_system.await_count:
                    break
                await asyncio.sleep(0.01)
            bridge.start_system.assert_awaited_once()
            await buttons["Stop System"]()

    asyncio.run(check())
    client.delete()

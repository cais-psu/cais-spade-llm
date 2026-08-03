"""UI contract tests for confirmed physical UR5e pick execution."""

from __future__ import annotations

import ast
import inspect
import textwrap

from cais_spade_llm.ui.pages import control


def _predefined_function_body_tree() -> ast.Module:
    source = textwrap.dedent(inspect.getsource(control._predefined_function_record_body))
    return ast.parse(source)


def _directly_awaited_bridge_methods() -> dict[str, ast.Call]:
    calls: dict[str, ast.Call] = {}
    for node in ast.walk(_predefined_function_body_tree()):
        if not isinstance(node, ast.Await) or not isinstance(node.value, ast.Call):
            continue
        function = node.value.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "bridge"
        ):
            calls[function.attr] = node.value
    return calls


def test_pick_controls_use_separate_explicit_confirmations() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert '"Run pick_approach"' in source
    assert '"Confirm Run pick_approach"' in source
    assert '"Run pick_grasp"' in source
    assert '"Confirm Run pick_grasp"' in source
    assert "with ui.dialog() as pick_approach_confirm" in source
    assert "with ui.dialog() as pick_grasp_confirm" in source
    assert "This commands physical robot motion." in source
    assert "This closes the physical RG2 gripper and lifts the gear." in source
    assert '"Move Above Part"' not in source


def test_pick_handlers_directly_await_confirmed_bridge_apis() -> None:
    calls = _directly_awaited_bridge_methods()

    for method_name in (
        "digital_twin_execute_pick_approach",
        "digital_twin_execute_pick_grasp",
    ):
        call = calls[method_name]
        confirmed = next(keyword.value for keyword in call.keywords if keyword.arg == "confirmed")
        assert isinstance(confirmed, ast.Constant)
        assert confirmed.value is True
        assert len(call.args) == 4


def test_pick_ui_keeps_preview_read_only_and_disables_controls_while_busy() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert "Preview Target is read-only." in source
    assert "requests a separate fresh" in source
    assert "pick_execution.get(\"busy\")" in source
    assert "pick_approach_button.set_enabled(enabled)" in source
    assert "pick_grasp_button.set_enabled(enabled)" in source
    assert "preview_button.disable()" in source


def test_pick_staging_step_explains_that_no_recording_is_required() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert 'step_name == "move_to_origin_resource_location"' in source
    assert "Automatic physical staging from `origin_resource_location`; " in source
    assert "no recording required." in source


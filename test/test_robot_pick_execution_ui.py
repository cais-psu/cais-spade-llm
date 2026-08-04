"""UI contract tests for confirmed physical robot-function execution."""

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


def test_robot_functions_panel_has_one_dynamic_run_control() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    panel_source = inspect.getsource(control._function_record_panel)

    assert 'ui.label("Robot Functions")' in panel_source
    assert 'ui.button("Run", on_click=_check_run_readiness' in source
    assert 'run_button.set_text(f"Run {function_name}"' in source
    assert "with ui.dialog() as run_confirm" in source
    assert '"Confirm Run"' in source
    assert '"Run pick_approach"' not in source
    assert '"Run pick_grasp"' not in source
    assert "This commands physical robot motion." in source


def test_run_checks_no_motion_readiness_before_confirmation_and_rechecks_on_execute() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    calls = _directly_awaited_bridge_methods()

    readiness_call = calls["digital_twin_robot_function_execution_readiness"]
    execute_call = calls["digital_twin_execute_robot_function"]
    assert len(readiness_call.args) == 3
    assert len(execute_call.args) == 3
    assert {keyword.arg for keyword in readiness_call.keywords} == {
        "origin_resource_location",
        "destination_location",
        "part_name",
    }
    confirmed = next(
        keyword.value for keyword in execute_call.keywords if keyword.arg == "confirmed"
    )
    assert isinstance(confirmed, ast.Constant)
    assert confirmed.value is True
    assert source.index("digital_twin_robot_function_execution_readiness") < source.index(
        "run_confirm.open()"
    )


def test_exact_dynamic_selectors_cover_all_function_arguments() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert 'label="origin_resource_location"' in source
    assert 'label="destination_location"' in source
    assert 'label="part_name"' in source
    for function_name in (
        "pick_approach",
        "pick_grasp",
        "place_approach",
        "place_insert",
    ):
        assert f'"{function_name}"' in source
    assert "origin_select.set_visibility" in source
    assert "destination_select.set_visibility" in source
    assert "part_select.set_visibility" in source
    assert "Releasing the part is irreversible." in source


def test_position_recording_is_conditional_and_capture_checks_readiness_automatically() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert 'ui.label("Position Recording")' in source
    assert "recording_container.set_visibility(bool(required))" in source
    assert '"Capture Position"' in source
    assert '"Save/Replace Position"' in source
    assert '"Clear Position"' in source
    assert '"Test Position"' in source
    assert "digital_twin_capture_function_step" in source
    assert "digital_twin_function_capture_readiness" not in source
    assert "Check Capture Readiness" not in source
    assert "Preview Target" not in source


def test_pick_staging_step_explains_that_no_recording_is_required() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert 'step_name == "move_to_origin_resource_location"' in source
    assert "Automatic physical staging from `origin_resource_location`; " in source
    assert "no recording required." in source

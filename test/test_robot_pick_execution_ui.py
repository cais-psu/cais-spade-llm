"""UI contract tests for confirmed physical robot-function execution."""

from __future__ import annotations

import ast
import inspect
import math
import textwrap

import pytest

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


def test_gripper_close_test_is_separate_confirmed_gripper_only_control() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    calls = _directly_awaited_bridge_methods()

    assert '"Gripper Close Test"' in source
    assert '"Confirm Gripper Close Test"' in source
    assert "The arm will not move." in source
    assert "approximately 26.16 mm" in source
    assert "Gripper Close Test only" in source
    assert "gripper_close_position" in source
    close_test_call = calls["digital_twin_execute_gripper_close_test"]
    assert len(close_test_call.args) == 4
    confirmed = next(
        keyword.value for keyword in close_test_call.keywords if keyword.arg == "confirmed"
    )
    assert isinstance(confirmed, ast.Constant)
    assert confirmed.value is True


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
        "operator_confirmed_held_part",
    }
    confirmed = next(
        keyword.value for keyword in execute_call.keywords if keyword.arg == "confirmed"
    )
    assert isinstance(confirmed, ast.Constant)
    assert confirmed.value is True
    execute_operator_held_part = next(
        keyword.value
        for keyword in execute_call.keywords
        if keyword.arg == "operator_confirmed_held_part"
    )
    assert isinstance(execute_operator_held_part, ast.Name)
    assert execute_operator_held_part.id == "operator_confirmed_held_part"
    assert source.index("digital_twin_robot_function_execution_readiness") < source.index(
        "run_confirm.open()"
    )
    assert "execution_blocker.set_visibility(bool(blocker))" in source
    assert "and not blocker" not in source
    assert "not require the assembly sequence resource_state" in source
    assert "Held-part, gripper, task-context, readiness, and confirmation checks" in source
    assert "reset to idle" not in source


def test_place_run_refreshes_part_name_from_exact_held_part() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert "bridge.digital_twin_function_held_part(_current_robot())" in source
    assert 'if function_name in {"place_approach", "place_insert"}' in source
    assert "if held_part in options:" in source
    assert "part_select.value = held_part" in source
    assert source.index("_sync_part_name()", source.index("def _check_run_readiness")) < source.index(
        "values = _execution_kwargs()", source.index("def _check_run_readiness")
    )
    assert "Independent place_approach may run with " in source
    assert "place_insert at assembly_board-v1 requires the held part" in source
    assert "run place_approach independently with held_part" in source
    assert "cannot run place_insert at assembly_board-v1" in source
    assert "no retained held_part context. Select place_approach" in source
    assert "standalone run assumes exact part" in source
    assert "physically clamped" in source
    assert "open the empty gripper and retreat 0.08 m" in source


def test_standalone_mg_automatically_restores_handoff_before_supervised_place_insert() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    calls = _directly_awaited_bridge_methods()

    assert "operator_held_part_checkbox" not in source
    assert "I visually confirm MG is physically clamped in the UR5e gripper" not in source
    assert '_current_part_name() in _MOVE_INSERT_SUPPORTED_PARTS' in source
    assert "requires its retained pick_grasp and place_approach context" in source
    assert "def _physical_function_execution_selected()" in source
    assert "_physical_function_execution_selected()" in source
    assert "def _operator_held_part_origin_resource_location()" in source
    assert '"pick_approach",' in source
    assert 'if "prusa-mk4-2" in options:' in source
    assert 'return "prusa-mk4-2"' in source
    assert '_current_robot() == "ur5e"' in source
    assert '_current_function() == "place_approach"' in source
    assert '_current_destination_location() == "assembly_board-v1"' in source
    assert '_current_part_name() in _MOVE_INSERT_SUPPORTED_PARTS' in source
    assert 'not bridge.digital_twin_function_held_part("ur5e")' in source
    assert "return _operator_held_part_option_visible()" in source
    assert source.count(
        "operator_confirmed_held_part=operator_confirmed_held_part"
    ) == 3
    assert "operator_handoff_origin_resource_location=(" in source
    assert "pick_approach.descend handoff" in source
    assert "latest successful RG2 close command" not in source
    assert "fresh robot TF" in source
    assert "assumes {part_name} is physically" in source
    assert source.count("pick recording") >= 2
    assert "place_approach and retain the exact {part_name}" in source
    assert "handoff. On success, return to place_insert" in source
    assert "Then return to place_insert;" in source
    assert "Supervised Test move_insert will recheck readiness" in source
    assert "This does not insert" in source
    assert "never uses this custody recovery" in source
    for method_name in (
        "digital_twin_robot_function_execution_readiness",
        "digital_twin_execute_robot_function",
    ):
        call = calls[method_name]
        origin_keyword = next(
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "origin_resource_location"
        )
        assert isinstance(origin_keyword, ast.IfExp)
        assert isinstance(origin_keyword.test, ast.Name)
        assert origin_keyword.test.id == "operator_confirmed_held_part"
        assert isinstance(origin_keyword.body, ast.Name)
        assert origin_keyword.body.id == "operator_handoff_origin_resource_location"
        assert isinstance(origin_keyword.orelse, ast.Subscript)


def test_operator_confirmed_handoff_is_available_for_all_six_move_insert_parts() -> None:
    source = inspect.getsource(control)
    body_source = inspect.getsource(control._predefined_function_record_body)

    assert (
        '_MOVE_INSERT_SUPPORTED_PARTS = ("SG", "MG", "LG", "SCP", "MCP", "LCP")'
        in source
    )
    assert body_source.count(
        "_current_part_name() in _MOVE_INSERT_SUPPORTED_PARTS"
    ) >= 3
    assert "retain the exact {part_name}" in body_source
    assert "{_current_part_name()} remains clamped" in body_source


def test_cartesian_jog_exposes_world_step_and_watchdog_smooth_hold_controls() -> None:
    source = inspect.getsource(control._teleop_section)
    button_source = inspect.getsource(control._jog_btn)
    refresh_source = source.split(
        "def _refresh_cartesian_controls() -> None:",
        maxsplit=1,
    )[1]
    active_hold_controls = refresh_source.split(
        'if cartesian_jog_mode["preparing"] or cartesian_command_state["pending"]:',
        maxsplit=1,
    )[0]
    smooth_dispatch = source.split(
        "async def _run_smooth_hold(", maxsplit=1
    )[1].split("def _start_smooth_hold", maxsplit=1)[0]
    step_dispatch = source.split(
        "async def _send_cartesian_step(", maxsplit=1
    )[1].split("def _cartesian_jog_button", maxsplit=1)[0]

    assert '["Step", "Smooth Hold"]' in source
    assert '["Off", "Step", "Smooth Hold"]' not in source
    assert 'value="Smooth Hold"' in source
    assert '"mode": "smooth"' in source
    assert "Preparing Mode" in source
    assert 'prepared = await _apply_cartesian_mode("Smooth Hold")' in source
    assert 'not smooth_hold["pressed"]' in source
    assert 'if robot == "xarm6":' in source
    assert 'readiness = cartesian_readiness_refresh["readiness"]' in source
    assert "bridge.teleop_cartesian_readiness" not in smooth_dispatch
    assert "bridge.teleop_cartesian_readiness" not in step_dispatch
    assert "Starting World" in source
    assert "Smooth Hold is the Hardware Stack default" in source
    assert "finite Step (mm) distance per click" in source
    assert 'cartesian_jog_mode["mode"] = "smooth"' in source
    assert '_set_cartesian_toggle_value("Smooth Hold")' in source
    assert "smooth_available = True" in source
    assert "Step selected; Mode 1 active" in source
    assert 'prepared = await _apply_cartesian_mode("Step")' in source
    assert "Motion Speed -" in source
    assert "Cartesian speed (mm/s)" in source
    assert "Joint jog speed (deg/s)" in source
    assert "Set a jog speed to 0 to disable that jog type" in source
    assert "Cartesian jog disabled: set Cartesian speed above 0 mm/s." in source
    assert "joint_jog_buttons" in source
    assert '"World X / World Y"' in source
    assert '"World Z"' in source
    assert '"pointerdown"' in button_source
    assert '"pointerup"' in button_source
    assert '"pointercancel"' in button_source
    assert '"mouseleave"' not in button_source
    assert "setPointerCapture" in button_source
    assert "touch-action: none" in button_source
    assert "cartesian_jog_buttons" not in active_hold_controls
    assert "conflicting_motion_buttons" in active_hold_controls
    assert 'f"Smooth Hold stopped by {reason}."' in source
    assert 'expected=event_name == "pointerup"' in button_source
    assert "window.addEventListener('pointerup', stop)" not in source
    assert "window.addEventListener('pointercancel', stop)" not in source
    assert "window.addEventListener('blur', stop)" in source
    assert "document.addEventListener('visibilitychange'" in source
    assert "ui.context.client.on_disconnect" in source
    assert "bridge.teleop_cartesian_smooth" in source
    assert "await asyncio.sleep(0.1)" in source
    assert source.count(
        'smooth_hold["pressed"] or smooth_hold["active"] or smooth_hold["stopping"]'
    ) >= 3
    assert 'label.set_text(f"Cartesian motion blocked: {message}")' in source


def test_motion_speed_status_refresh_does_not_rewrite_slider_values() -> None:
    source = inspect.getsource(control._teleop_section)
    refresh_source = source.split(
        "def _refresh_motion_speed_labels() -> None:",
        maxsplit=1,
    )[1].split("def _smooth_speed_mm_s", maxsplit=1)[0]
    configure_source = source.split(
        "def _configure_motion_speed_controls(robot: str) -> None:",
        maxsplit=1,
    )[1].split("def _refresh_motion_speed_labels", maxsplit=1)[0]

    assert ".value =" not in refresh_source
    assert ".props(" not in refresh_source
    assert 'motion_speed_controls[f"{control_key}_slider"]' in configure_source
    assert 'motion_speed_controls[f"{control_key}_number"]' in configure_source
    assert 'control._props["min"] = minimum' in configure_source
    assert 'control._props["max"] = maximum' in configure_source
    assert 'control._props["step"] = step' in configure_source
    assert 'control.props(' not in configure_source
    assert 'motion_settings[initial_robot][' in source
    assert '"cartesian_speed_max_mm_s"' in source
    assert "_PHYSICAL_CARTESIAN_SPEED_STEP_MM_S = 0.025" in inspect.getsource(
        control
    )
    assert 'motion_speed_controls["range"]' in source
    assert 'f"Applied Cartesian speed: {cartesian_value:.3f} mm/s"' in source
    assert "control.set_value(value)" in configure_source
    assert "control.update()" in configure_source
    assert "_configure_motion_speed_controls(selected_robot)" in source


def test_motion_speed_sliders_commit_on_release_without_reconfiguring() -> None:
    source = inspect.getsource(control._teleop_section)
    set_speed_source = source.split(
        "def _set_motion_speed(",
        maxsplit=1,
    )[1].split("def _configure_motion_speed_controls", maxsplit=1)[0]
    cartesian_slider_source = source.split(
        "cartesian_speed_slider = ui.slider(",
        maxsplit=1,
    )[1].split(
        'motion_speed_controls["cartesian_slider"]',
        maxsplit=1,
    )[0]

    assert "on_change=" not in cartesian_slider_source
    assert 'cartesian_speed_slider.on(\n                            "change"' in source
    assert 'joint_speed_slider.on(\n                            "change"' in source
    assert 'source="slider"' in source
    assert "def _motion_speed_slider_release_value(event) -> object:" in source
    assert "if isinstance(values, (list, tuple)) and len(values) == 1:" in source
    assert source.count("_motion_speed_slider_release_value(event)") == 3
    assert "cartesian_speed_slider.LOOPBACK = False" in source
    assert 'cartesian_speed_slider._props["loopback"] = False' in source
    assert "joint_speed_slider.LOOPBACK = False" in source
    assert 'joint_speed_slider._props["loopback"] = False' in source
    assert "settings = motion_settings[robot_key]" in set_speed_source
    assert "_motion_settings_for(robot_key)" not in set_speed_source
    assert "bridge.teleop_cartesian_readiness" not in set_speed_source


def test_cartesian_readiness_never_blocks_slider_or_timer_event_handlers() -> None:
    source = inspect.getsource(control._teleop_section)
    render_source = source.split(
        "def _refresh_cartesian_controls() -> None:",
        maxsplit=1,
    )[1].split(
        "async def _refresh_cartesian_readiness_async(",
        maxsplit=1,
    )[0]
    async_source = source.split(
        "async def _refresh_cartesian_readiness_async(",
        maxsplit=1,
    )[1].split(
        "def _schedule_cartesian_readiness_refresh(",
        maxsplit=1,
    )[0]

    assert "bridge.teleop_cartesian_readiness" not in render_source
    assert "bridge.teleop_target" not in render_source
    assert "await asyncio.to_thread(_load)" in async_source
    assert 'ui.timer(0.5, _schedule_cartesian_readiness_refresh)' in source
    assert 'cartesian_readiness_refresh["busy"]' in source
    assert 'cartesian_readiness_refresh["revision"]' in source


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
    assert "recording_container.set_visibility(bool(recordable_steps))" in source
    assert '"Capture Pose"' in source
    assert '"Save/Replace Pose"' in source
    assert '"Clear Position"' in source
    assert '"Preview Resolved Pose"' not in source
    assert "digital_twin_preview_function_position" not in source
    assert '"Apply Axis Sources"' not in source
    assert "Saved calibration XYZ offset" in source
    assert '"Z+ 1 mm"' not in source
    assert '"Z− 1 mm"' not in source
    assert '"Test Position"' in source
    assert "digital_twin_capture_function_step" in source
    assert "digital_twin_prepare_function_capture" in source
    assert "digital_twin_function_capture_readiness" not in source
    assert "Check Capture Readiness" not in source
    assert "Preview Target" not in source


def test_assembly_board_v1_readiness_panel_has_optional_no_motion_acceptance() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    refresh_source = source.split(
        "async def _refresh_assembly_board_v1_readiness(",
        maxsplit=1,
    )[1].split("def _default_location", maxsplit=1)[0]

    assert 'ui.label("assembly_board-v1 Board Readiness")' in source
    assert 'ui.icon("warning", color="red")' in source
    assert 'function_name in {"place_approach", "place_insert"}' in source
    assert '_current_destination_location() == "assembly_board-v1"' in source
    assert "bridge.perception_assembly_board_v1_aruco_status" in source
    assert "bridge.list_named_positions" in source
    assert 'ui.button(\n                "Locate & Accept Board"' in source
    assert 'ui.button(\n            "Re-accept Board"' not in source
    assert "bridge.perception_locate_and_accept_assembly_board_v1" in source
    assert 'accept_visible = function_name == "place_approach"' in source
    assert 'status.get("ready_to_accept")' in source
    assert "This does not move the robot." in source
    assert "Confirmed Run place_approach first moves" in source
    assert "Using the assembly_board-v1 pose frozen by place_approach." in source
    assert 'function_name == "place_insert"' in source
    board_render_source = source.split(
        "def _render_assembly_board_v1_readiness() -> None:",
        maxsplit=1,
    )[1].split(
        "async def _refresh_assembly_board_v1_readiness() -> None:",
        maxsplit=1,
    )[0]
    assert "held_part empty" not in board_render_source
    assert "Complete pick_grasp" not in board_render_source
    assert "ui.timer(2.0, _refresh_assembly_board_v1_readiness, immediate=True)" in source
    assert "digital_twin_execute_robot_function" not in refresh_source
    assert "digital_twin_test_function_position" not in refresh_source


def test_assembly_board_v1_manual_acceptance_invalidates_only_unsaved_capture_state() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    accept_source = source.split(
        "async def _locate_and_accept_assembly_board_v1() -> None:",
        maxsplit=1,
    )[1].split("def _default_location", maxsplit=1)[0]

    assert "_assembly_board_v1_automatic_accept_observation" not in source
    assert "async def _automatically_accept_assembly_board_v1" not in source
    assert "async def _reaccept_assembly_board_v1" not in source
    assert "bridge.perception_locate_and_accept_assembly_board_v1" in accept_source
    assert '_current_function() == "place_approach"' in accept_source
    assert '_current_destination_location() == "assembly_board-v1"' in accept_source
    assert "bridge.digital_twin_clear_function_steps" in accept_source
    assert '"place_approach",' in accept_source
    assert '"assembly_board-v1",' in accept_source
    assert "part_name=part_name" in accept_source
    assert "digital_twin_save_function_position" not in accept_source
    assert "digital_twin_clear_function_position" not in accept_source
    assert "digital_twin_execute_robot_function" not in accept_source
    assert "digital_twin_test_function_position" not in accept_source
    assert accept_source.index("if not result.get(\"success\")") < accept_source.index(
        "bridge.digital_twin_clear_function_steps"
    )
    assert "pending_execution.clear()" in accept_source
    assert "pending_assembly.clear()" in accept_source
    assert "run_confirm.close()" in accept_source
    assert "assembly_confirm.close()" in accept_source
    assert "_invalidate_move_insert_trial()" in accept_source
    assert "await _refresh_assembly_board_v1_readiness()" in accept_source
    assert "await _load_move_insert_trial_readiness()" in accept_source
    assert "automatically accept or reaccept the current board" in source
    assert "run_confirm.open()" in source
    assert "confirmed=True" in source


def test_assembly_board_v1_readiness_has_green_amber_and_red_states() -> None:
    green = control._assembly_board_v1_readiness(
        {
            "accepted_baseline_ready": True,
            "accepted": True,
            "visible": True,
            "valid": True,
            "frame_age_sec": 0.1,
        }
    )
    amber = control._assembly_board_v1_readiness(
        {
            "accepted_baseline_ready": True,
            "accepted": True,
            "visible": False,
            "movement_blocked": True,
        }
    )
    red = control._assembly_board_v1_readiness(
        {
            "accepted_baseline_ready": False,
            "accepted_baseline_error": "camera calibration changed",
            "accepted": True,
        }
    )

    assert green == {
        "usable": True,
        "level": "green",
        "message": "Accepted board baseline is usable and ArUco ID 70 is visible.",
    }
    assert amber["usable"] is True
    assert amber["level"] == "amber"
    assert "currently occluded" in str(amber["message"])
    assert red == {
        "usable": False,
        "level": "red",
        "message": "camera calibration changed",
    }


def test_assembly_board_v1_readiness_directs_acceptance_to_confirmed_run() -> None:
    automatic = control._assembly_board_v1_readiness(
        {
            "accepted_baseline_ready": False,
            "accepted": False,
            "post_staging_acceptance_allowed": True,
        }
    )
    blocked = control._assembly_board_v1_readiness(
        {
            "accepted_baseline_ready": False,
            "accepted_baseline_error": (
                "camera calibration changed; use Locate & Accept Board again."
            ),
            "accepted": True,
            "post_staging_acceptance_allowed": False,
        }
    )

    assert "Confirmed Run place_approach" in str(automatic["message"])
    assert "accept the board automatically" in str(automatic["message"])
    assert "Locate & Accept Board" not in str(blocked["message"])
    assert "confirmed Run place_approach" in str(blocked["message"])


def test_assembly_board_v1_readiness_treats_a_stale_visible_snapshot_as_amber() -> None:
    readiness = control._assembly_board_v1_readiness(
        {
            "accepted_baseline_ready": True,
            "accepted": True,
            "visible": True,
            "valid": True,
            "frame_age_sec": 2.01,
        }
    )

    assert readiness["usable"] is True
    assert readiness["level"] == "amber"
    assert "not fresh" in str(readiness["message"])


def test_generalized_recording_guidance_disables_raw_replay() -> None:
    string_constants = [
        node.value
        for node in ast.walk(_predefined_function_body_tree())
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    guidance = next(
        value for value in string_constants if "Raw waypoint Replay is disabled" in value
    )

    assert "resolve current geometry first" in guidance
    assert "apply this robot correction" in guidance


def test_assembly_board_v1_readiness_fallback_does_not_treat_occlusion_as_movement() -> None:
    readiness = control._assembly_board_v1_readiness(
        {
            "accepted": True,
            "visible": False,
            "valid": False,
            "movement_blocked": True,
            "calibration_changed": False,
        }
    )

    assert readiness["usable"] is True
    assert readiness["level"] == "amber"


def test_assembly_board_v1_cross_view_diagnostic_preserves_accepted_pose_authority() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert 'status.get("excessive_movement")' in source
    assert "This cross-view difference is diagnostic and does not replace the " in source
    assert "accepted board pose. Use Locate & Accept Board if the board actually " in source
    assert 'and not _assembly_board_v1_accepted_usable()' in source


def test_assembly_board_v1_capture_stays_gated_while_run_can_stage_and_accept() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert 'function_name == "place_approach"' in source
    assert 'and name == "assembly_board-v1"' in source
    assert "and not _assembly_board_v1_accepted_usable()" in source
    assert "or board_capture_blocked" in source
    assert 'board_status.get("post_staging_acceptance_allowed")' in source
    assert "and not post_staging_acceptance_allowed" in source
    assert "and not board_run_blocked" in source
    assert "No manual camera staging is required." in source
    assert "collect ten fresh post-motion observations" in source
    assert "automatically accept or reaccept the current board" in source


def test_pick_staging_step_explains_that_no_recording_is_required() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert 'step_name == "move_to_origin_resource_location"' in source
    assert "Automatic physical staging from `origin_resource_location`; " in source
    assert "no recording required." in source


def test_assembly_is_a_separate_card_not_a_sixth_robot_function() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert 'ui.label("Assembly")' in source
    assert 'ui.button(\n                    "Run Assembly"' in source
    assert "functions = bridge.digital_twin_function_names()" in source
    assert "function_select = (\n            ui.select(functions" in source
    assert "assembly_functions = (" in source
    for function_name in (
        "pick_approach",
        "pick_grasp",
        "place_approach",
        "place_insert",
        "move_home",
    ):
        assert f'        "{function_name}",' in source
    assert '        "assembly",' not in source


def test_assembly_uses_selected_robot_and_its_own_exact_argument_selectors() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    tree = _predefined_function_body_tree()

    assert 'assembly_origin_select = (\n                    ui.select([], label="origin_resource_location")' in source
    assert 'assembly_destination_select = (\n                    ui.select([], label="destination_location")' in source
    assert 'assembly_part_select = (\n                    ui.select([], label="part_name")' in source
    assert "robot = _current_robot()" in source
    location_functions = {
        call.args[1].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "digital_twin_function_location_options"
        and len(call.args) >= 2
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "robot"
        and isinstance(call.args[1], ast.Constant)
    }
    assert {"pick_approach", "place_approach"} <= location_functions
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "digital_twin_function_part_options"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "pick_approach"
        for call in ast.walk(tree)
    )


def test_assembly_checks_readiness_before_one_confirmation_and_execution() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    calls = _directly_awaited_bridge_methods()
    execute_calls = [
        node
        for node in ast.walk(_predefined_function_body_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "bridge"
        and node.func.attr == "digital_twin_execute_assembly"
    ]

    readiness_call = calls["digital_twin_assembly_readiness"]
    assert len(execute_calls) == 1
    execute_call = execute_calls[0]
    assert len(readiness_call.args) == 2
    assert len(execute_call.args) == 2
    expected_arguments = {
        "origin_resource_location",
        "destination_location",
        "part_name",
    }
    assert {keyword.arg for keyword in readiness_call.keywords} == expected_arguments
    assert {keyword.arg for keyword in execute_call.keywords} == {
        *expected_arguments,
        "confirmed",
    }
    confirmed = next(
        keyword.value for keyword in execute_call.keywords if keyword.arg == "confirmed"
    )
    assert isinstance(confirmed, ast.Constant)
    assert confirmed.value is True
    assert source.index("digital_twin_assembly_readiness") < source.index(
        "assembly_confirm.open()"
    )
    assert source.count("with ui.dialog() as assembly_confirm") == 1
    assert source.count('"Confirm Run Assembly"') == 1


def test_assembly_confirmation_lists_exact_order_and_irreversible_release() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    exact_order = (
        "pick_approach → pick_grasp → place_approach → place_insert → move_home"
    )

    assert source.count(exact_order) == 2
    assert "place_insert releases the part irreversibly" in source
    assert "Assembly stops immediately" in source
    assert "never retries or continues automatically" in source
    assert "This commands physical robot motion." in source
    assert "The Hardware Stack may remain running." in source


def test_assembly_correction_guidance_matches_readiness_gate() -> None:
    string_constants = [
        node.value
        for node in ast.walk(_predefined_function_body_tree())
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    guidance = next(
        value
        for value in string_constants
        if "saved-unconfirmed robot corrections" in value
    )

    assert "Missing optional robot corrections are allowed." in guidance
    assert "Buffered or saved-unconfirmed robot corrections" in guidance
    assert "block Assembly" in guidance
    assert "Save/Replace Pose or Clear Position" in guidance
    assert "ignored" not in guidance


def test_assembly_card_shows_both_exact_correction_states() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    string_constants = [
        node.value
        for node in ast.walk(_predefined_function_body_tree())
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    tracked_steps = next(
        value
        for value in string_constants
        if "pick_approach.descend" in value
        and "place_approach.move_above_destination" in value
    )

    assert 'ui.label("Assembly Robot Corrections")' in source
    assert "pick_approach.descend" in tracked_steps
    assert "place_approach.move_above_destination" in tracked_steps
    assert "place_approach.descend" in tracked_steps
    assert '"pick_approach",\n                _current_assembly_origin_resource_location()' in source
    assert '"place_approach",\n                _current_assembly_destination_location()' in source
    assert "bridge.digital_twin_list_function_buffer_steps(" in source
    assert "bridge.digital_twin_list_function_file_steps(" in source
    assert "bridge.digital_twin_function_template(function_name)" in source
    assert 'f"{function_name}.{step_name}: {state}"' in source
    assert '"buffered"' in source
    assert '"unconfirmed"' in source
    assert '"active"' in source
    assert '"missing"' in source
    assert "Missing optional robot corrections are allowed." in source
    assert "Assembly is blocked; use Save/Replace Pose or Clear Position." in source
    assert "_refresh_assembly_correction_status()" in source


def test_assembly_ui_propagates_client_task_cancellation_to_the_bridge() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    panel_source = inspect.getsource(control._function_record_panel)

    assert "assembly_task = asyncio.create_task(" in source
    assert "bridge.digital_twin_execute_assembly(" in source
    assert 'execution["assembly_task"] = assembly_task' in source
    assert "result = await assembly_task" in source
    assert "def _cancel_active_assembly" in source
    assert "task.cancel()" in source
    assert "client.on_disconnect(_cancel_active_assembly)" in source
    assert "client.on_delete(_cancel_active_assembly)" in source
    assert "return _cancel_body_tasks" in source
    assert 'body_cleanup["callback"]()' in panel_source
    assert "body.clear()" in panel_source
    assert 'body_cleanup["callback"] = _predefined_function_record_body(' in panel_source


def test_assembly_progress_reports_steps_completed_and_failed_function() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert 'progress.get("assembly_step_index", 0)' in source
    assert 'progress.get("assembly_step_count", len(assembly_functions))' in source
    assert 'progress.get("completed_functions", [])' in source
    assert 'progress.get("failed_function")' in source
    assert 'f"Assembly step {assembly_step_index}/{assembly_step_count}."' in source
    assert 'f"Failed function: {failed_function}."' in source


def test_supervised_move_insert_replaces_every_operator_tuning_control() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert 'ui.label("Supervised move_insert")' in source
    assert '"Supervised Test move_insert"' in source
    assert '"Stop Supervised move_insert"' in source
    assert '"Confirm Physical Recovery"' in source
    assert '"Confirm Completion"' in source
    assert source.count("ui.number(") == 0
    for removed_text in (
        "move_insert Tuning",
        "Insertion force (N)",
        "Maximum spiral radius (mm)",
        "Save/Replace Override",
        "Use Shared",
        "Clear Override",
        "Discard Draft",
    ):
        assert removed_text not in source


def test_insertion_demonstration_is_button_only_and_never_replays_motion() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert 'ui.label("Insertion Demonstration")' in source
    for button_label in (
        "Start Recording",
        "Save Recording",
        "Cancel Recording",
        "Download Recording Bundle",
        "Reanalyze Saved Recording",
        "Delete Previous Recording",
    ):
        assert f'"{button_label}"' in source
    assert 'insertion_demonstration_container.set_visibility(selected)' in source
    assert '_current_function() == "place_insert"' in source
    assert "is never replayed as robot motion" in source
    assert "digital_twin_save_insertion_recording" in source
    assert "digital_twin_delete_insertion_recording" in source
    assert "digital_twin_reanalyze_insertion_recording" in source
    assert 'ui.button(\n                "Capture Seated"' not in source
    assert 'ui.button(\n                "Stop Recording"' not in source
    assert source.count("ui.number(") == 0
    for forbidden_label in (
        "Capture MG Seated",
        "Insertion force (N)",
        "Maximum spiral radius (mm)",
    ):
        assert forbidden_label not in source


def test_saved_insertion_recording_refreshes_and_can_recheck_supervised_readiness() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    save_source = source.split(
        "async def _save_insertion_recording()", maxsplit=1
    )[1].split("async def _cancel_insertion_recording()", maxsplit=1)[0]
    render_source = source.split(
        "def _render_move_insert_trial()", maxsplit=1
    )[1].split("with assembly_container", maxsplit=1)[0]

    assert '== "recording_saved_return_to_pre_insertion"' in save_source
    assert "_invalidate_move_insert_trial()" in save_source
    assert "await _load_move_insert_trial_readiness()" in save_source
    assert 'state == "ready_to_test"' in render_source
    assert 'status.get("ready")' in render_source
    assert 'state != "confirmed"' in render_source


def test_saved_insertion_recording_polls_fresh_pose_readiness_without_motion() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    loader = source.split(
        "async def _load_move_insert_trial_readiness(", maxsplit=1
    )[1].split("async def _check_move_insert_trial_readiness", maxsplit=1)[0]
    refresher = source.split(
        "def _refresh_move_insert_trial_readiness_after_recording()", maxsplit=1
    )[1].split("def _refresh_function_execution_progress", maxsplit=1)[0]

    assert '"readiness_revision": 0' in source
    assert 'readiness_revision != move_insert_trial["readiness_revision"]' in loader
    assert 'if not result.get("ready"):' in loader
    assert "move_insert_trial_confirm.close()" in loader
    assert "if not background:" in loader
    assert 'move_insert_trial["loading"] = True' in loader
    assert "(not background or status_changed)" in loader
    assert '!= "recording_saved_return_to_pre_insertion"' in refresher
    assert "_load_move_insert_trial_readiness(background=True)" in refresher
    assert 'trial_status.get("ready")' in refresher
    assert 'trial_state == "ready_to_test"' in refresher
    assert 'ui.timer(1.0, _refresh_move_insert_trial_readiness_after_recording)' in source
    for forbidden_motion in (
        "digital_twin_execute_move_insert_trial",
        "digital_twin_execute_robot_function",
        "digital_twin_execute_assembly",
    ):
        assert forbidden_motion not in refresher


def test_move_insert_expected_start_diagnostic_uses_bridge_measurements() -> None:
    diagnostic = control._move_insert_expected_start_diagnostic(
        {
            "expected_start_delta_m": {
                "x": 0.001,
                "y": -0.002,
                "z": 0.003,
            },
            "expected_start_position_error_m": 0.004,
            "expected_start_position_tolerance_m": 0.005,
            "expected_start_rotation_error_rad": math.radians(1.0),
            "expected_start_orientation_tolerance_rad": math.radians(2.0),
            "expected_start_tf_age_sec": 0.25,
            "dispatch_attempted": False,
        }
    )

    assert "ΔX +1.00 mm" in diagnostic
    assert "ΔY -2.00 mm" in diagnostic
    assert "ΔZ +3.00 mm" in diagnostic
    assert "Distance 4.00 mm / limit 5.00 mm" in diagnostic
    assert "Rotation 1.00 deg / limit 2.00 deg" in diagnostic
    assert "TF age 0.25 s" in diagnostic
    assert "Readiness only: move_insert was not dispatched" in diagnostic


def test_insertion_demonstration_disconnect_cancels_and_preserves_trace() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    panel_source = inspect.getsource(control._function_record_panel)

    assert "bridge.digital_twin_cancel_insertion_recording" in source
    assert '"Control client disconnected during recording."' in source
    assert "client.on_disconnect(_cancel_active_assembly)" in source
    assert "client.on_delete(_cancel_active_assembly)" in source
    assert "return _cancel_body_tasks" in source
    assert 'body_cleanup["callback"]()' in panel_source


@pytest.mark.parametrize(
    ("function_name", "expected_visible"),
    [
        ("pick_approach", False),
        ("pick_grasp", False),
        ("place_approach", False),
        ("place_insert", True),
        ("move_home", False),
    ],
)
def test_supervised_move_insert_is_visible_only_for_exact_place_insert_selection(
    function_name: str,
    expected_visible: bool,
) -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    tree = _predefined_function_body_tree()
    visibility_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_visibility"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "move_insert_trial_container"
    ]

    assert (
        'ui.column().classes("w-full gap-2 mt-2") as move_insert_trial_container,'
        in source
    )
    assert len(visibility_calls) == 1
    expression = visibility_calls[0].args[0]
    assert isinstance(expression, ast.Compare)
    assert len(expression.ops) == 1
    assert isinstance(expression.ops[0], ast.Eq)
    assert isinstance(expression.left, ast.Call)
    assert isinstance(expression.left.func, ast.Name)
    assert expression.left.func.id == "_current_function"
    assert expression.left.args == []
    assert expression.left.keywords == []
    assert len(expression.comparators) == 1
    comparator = expression.comparators[0]
    assert isinstance(comparator, ast.Constant)
    assert comparator.value == "place_insert"
    assert (function_name == comparator.value) is expected_visible


def test_supervised_move_insert_uses_exact_guarded_bridge_surface() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    for method in (
        "digital_twin_move_insert_trial_readiness",
        "digital_twin_execute_move_insert_trial",
        "digital_twin_cancel_move_insert_trial",
        "digital_twin_move_insert_trial_status",
        "digital_twin_confirm_move_insert_completion",
    ):
        assert f"bridge.{method}" in source
    assert source.count("confirmed=True") >= 3
    assert 'trial_id=str(move_insert_trial.get("trial_id") or "")' in source
    assert "The part remains clamped: this test never releases, lifts," in source
    assert "or runs move_home" in source


def test_supervised_move_insert_shows_recoverable_soft_overload_phases() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    for message in (
        "Searching for pin center — local spiral",
        "No entry detected — expanding touch search",
        "Pin capture detected — inserting straight",
        "appears cocked — withdrawing completely",
        "Returning above tactile center",
        "Alignment normal — retrying insertion",
        "Soft load limit detected — unloading force first",
        "Load persisted — performing the bounded micro-backoff",
        "Load cleared — retrying direct insertion",
        "Checking engagement and stable seating",
    ):
        assert message in source
    assert 'f" (cycle {max(1, relief_cycle)}/3)"' in source
    assert 'f" (cycle {max(1, disengagement_cycle)}/6)"' in source
    assert 'status.get("hard_limit_reason")' in source
    assert 'status.get("last_soft_overload_reason")' in source
    assert 'status.get("hardware_stack_repair_required")' in source


def test_supervised_move_insert_qualification_gates_release_and_normal_run() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert "def _move_insert_trial_is_qualified" in source
    assert "elif not _move_insert_trial_is_qualified()" in source
    assert "and not place_insert_run_blocked" in source
    assert "no retained held_part context. Select place_approach" in source
    assert "finishes and you use Confirm Completion" in source
    assert "state == \"awaiting_visual_confirmation\"" in source
    assert 'status.get("completion_eligible")' in source
    assert "release the part irreversibly and lift exactly once" in source
    assert "One successful release and lift confirms the selected exact" in source
    assert '"confirmation_progress"' in source
    assert "Confirmed {confirmed_trial_count} of" in source
    assert 'if _current_robot() == "xarm6":' in source
    assert "Physical xarm6 move_insert remains blocked in this version" in source
    assert "installed UFactory six-axis force/torque sensor" in source


def test_supervised_move_insert_states_stop_and_failure_diagnostics_are_visible() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    for state_label in (
        "Not confirmed",
        "Ready to test",
        "Testing",
        "Awaiting visual confirmation",
        "Confirmed",
    ):
        assert f'"{state_label}"' in source
    assert 'status.get("completion_motion_active")' in source
    assert 'if state == "testing" and not bool(' in source
    assert 'move_insert_trial.get("active") or status.get("active")' in source
    assert "stop_move_insert_button.set_visibility(True)" in source
    assert 'move_insert_trial["stop_requested"] = True' in source
    assert "while True:" in source
    assert 'if bool(status.get("active")):' in source
    assert "active = bool(move_insert_active or completion_motion_active)" in source
    assert 'local_active or status.get("active")' in source
    assert 'local_active or status.get("active") or state == "testing"' not in source
    assert "Supervised move_insert is already terminal" in source
    assert "Stop cancels active force/search motion" in source
    assert 'status.get("failure_id")' in source
    assert 'status.get("diagnostic_bundle_path")' in source
    assert 'status.get("insertion_depth_diagnostic")' in source
    assert "Failure ID:" in source
    assert "Diagnostic bundle:" in source
    assert '"Download Diagnostic Bundle"' in source
    assert 'Path("~/.local/share/cais-spade-llm/move_insert_trials")' in source
    assert ".resolve(strict=True)" in source
    assert "bundle_path.relative_to(allowed_root)" in source
    assert 'bundle_path.name != "diagnostic_bundle.zip"' in source
    assert "ui.download(bundle_path" in source
    assert "download_move_insert_diagnostic_button.set_visibility(bool(bundle_path))" in source


def test_supervised_move_insert_has_exact_no_motion_recovery_control() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    recovery_handler = source.split(
        "async def _confirmed_move_insert_recovery()", maxsplit=1
    )[1].split("async def _confirmed_move_insert_completion()", maxsplit=1)[0]

    assert 'ui.button(\n                        "Record Failure"' not in source
    assert "bridge.digital_twin_record_move_insert_failure" not in source
    assert '"Confirm Physical Recovery"' in source
    assert "bridge.digital_twin_confirm_move_insert_recovery" in recovery_handler
    assert "confirmed=True" in recovery_handler
    assert 'values["target"]' in recovery_handler
    assert 'values["robot"]' in recovery_handler
    assert 'destination_location=values["destination_location"]' in recovery_handler
    assert 'part_name=values["part_name"]' in recovery_handler
    assert 'trial_id=values["trial_id"]' in recovery_handler
    assert "digital_twin_execute_move_insert_trial" not in recovery_handler
    assert "digital_twin_confirm_move_insert_completion" not in recovery_handler
    assert "physically moved the part clear using" in source
    assert "approved manual recovery" in source
    assert "This action commands no robot motion" in source
    assert 'recovery_required = bool(status.get("recovery_required"))' in source
    assert (
        "confirm_move_insert_recovery_button.set_visibility(recovery_required)"
        in source
    )


def test_supervised_move_insert_restores_pending_review_after_selection_change() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    loader = source.split(
        "async def _load_move_insert_trial_readiness(", maxsplit=1
    )[1].split("async def _check_move_insert_trial_readiness", maxsplit=1)[0]

    assert loader.index("digital_twin_move_insert_trial_status") < loader.index(
        "digital_twin_move_insert_trial_readiness"
    )
    assert 'current_status.get("review_required")' in loader
    assert 'current_status.get("active")' in loader
    assert 'current_status.get("recovery_required")' in loader
    assert 'current_status.get("hardware_stack_repair_required")' in loader
    assert 'current_status.get("normal_repair_required")' in loader
    assert 'current_state == "awaiting_visual_confirmation"' in loader
    assert 'current_state in {"testing", "awaiting_visual_confirmation"}' not in loader
    assert "Finish the current move_insert review before another test" in loader


def test_supervised_move_insert_ui_exception_never_fabricates_recorded_failure() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    executor = source.split(
        "async def _confirmed_execute_move_insert_trial()", maxsplit=1
    )[1].split("async def _stop_move_insert_trial()", maxsplit=1)[0]

    assert "digital_twin_move_insert_trial_status" in executor
    assert '"state": "testing"' in executor
    assert '"active": True' in executor
    assert '"settlement is unknown; use Stop Supervised move_insert "' in executor
    assert "backend_may_be_active" in executor
    assert '"state": "failure_recorded"' not in executor


def test_assembly_confirmation_and_progress_surface_internal_move_insert() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert "place_insert.move_insert keeps the part gripped" in source
    assert "only after move_insert succeeds" in source
    assert 'progress.get("active_step")' in source
    assert 'active_step == "place_insert.move_insert"' in source
    assert 'progress.get("insert_phase")' in source


def test_robot_function_selection_batches_render_and_readiness_refresh_work() -> None:
    source = inspect.getsource(control._predefined_function_record_body)
    render_execution = source.split("def _render_execution()", maxsplit=1)[1].split(
        "async def _capture_position", maxsplit=1
    )[0]

    assert 'selection_update: dict[str, object] = {' in source
    assert 'if selection_update.get("active"):' in source
    assert 'selection_update["active"] = True' in source
    assert "def _schedule_selection_readiness_refresh()" in source
    assert 'current_task.cancel()' in source
    assert "await asyncio.sleep(0.05)" in source
    assert "await asyncio.gather(" in source
    assert "_refresh_assembly_board_v1_readiness()," in source
    assert "_load_insertion_demonstration_readiness()," in source
    assert "_load_move_insert_trial_readiness()," in source
    assert "teleop_cartesian_readiness" not in render_execution
    assert "bridge.teleop_cartesian_smooth_active(_current_robot())" in (
        render_execution
    )
    assert "_robot_function_hardware_snapshot()" in render_execution

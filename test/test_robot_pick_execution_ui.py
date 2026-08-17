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
    }
    confirmed = next(
        keyword.value for keyword in execute_call.keywords if keyword.arg == "confirmed"
    )
    assert isinstance(confirmed, ast.Constant)
    assert confirmed.value is True
    assert source.index("digital_twin_robot_function_execution_readiness") < source.index(
        "run_confirm.open()"
    )
    assert "execution_blocker.set_visibility(bool(blocker))" in source
    assert "and not blocker" not in source
    assert "not require the assembly sequence resource_state" in source
    assert "Held-part, gripper, task-context, readiness, and confirmation checks" in source
    assert "reset to idle" not in source


def test_cartesian_jog_exposes_world_step_and_watchdog_smooth_hold_controls() -> None:
    source = inspect.getsource(control._teleop_section)
    button_source = inspect.getsource(control._jog_btn)

    assert '["Step", "Smooth Hold"]' in source
    assert '["Off", "Step", "Smooth Hold"]' not in source
    assert 'value="Step"' in source
    assert "Preparing Mode" in source
    assert "Step selected; Mode 1 active" in source
    assert 'prepared = await _apply_cartesian_mode("Step")' in source
    assert "Motion Speed -" in source
    assert "Cartesian speed (mm/s)" in source
    assert "Joint jog speed (deg/s)" in source
    assert '"World X / World Y"' in source
    assert '"World Z"' in source
    assert '"pointerdown"' in button_source
    assert '"pointerup"' in button_source
    assert '"pointercancel"' in button_source
    assert '"mouseleave"' not in button_source
    assert "setPointerCapture" in button_source
    assert "touch-action: none" in button_source
    assert "window.addEventListener('pointerup', stop)" in source
    assert "window.addEventListener('pointercancel', stop)" in source
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
    assert "settings = motion_settings[robot_key]" in set_speed_source
    assert "_motion_settings_for(robot_key)" not in set_speed_source


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


def test_assembly_board_v1_readiness_panel_is_read_only_for_selected_robot() -> None:
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
    assert 'ui.button(\n            "Locate & Accept Board"' not in source
    assert 'ui.button(\n            "Re-accept Board"' not in source
    assert "bridge.perception_locate_and_accept_assembly_board_v1" not in source
    assert "Confirmed Run place_approach first moves" in source
    assert "Using the assembly_board-v1 pose frozen by place_approach." in source
    assert 'function_name == "place_insert"' in source
    assert "ui.timer(2.0, _refresh_assembly_board_v1_readiness, immediate=True)" in source
    assert "digital_twin_execute_robot_function" not in refresh_source
    assert "digital_twin_test_function_position" not in refresh_source


def test_assembly_board_v1_acceptance_is_only_part_of_confirmed_place_approach() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert "_assembly_board_v1_automatic_accept_observation" not in source
    assert "async def _automatically_accept_assembly_board_v1" not in source
    assert "async def _reaccept_assembly_board_v1" not in source
    assert "bridge.perception_locate_and_accept_assembly_board_v1" not in source
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


def test_assembly_board_v1_recording_guidance_describes_full_se3_replay() -> None:
    source = inspect.getsource(control._predefined_function_record_body)

    assert 'relative_reference.get("source")' in source
    assert '== "assembly_board-v1_aruco"' in source
    assert "Replay composes the current ArUco ID 70 world pose" in source
    assert "applying board " in source
    assert '"translation and rotation."' in source


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

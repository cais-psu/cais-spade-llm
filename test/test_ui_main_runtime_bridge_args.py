from __future__ import annotations

import os

import pytest

from cais_spade_llm import ui_main


def test_bridge_fixture_cli_args_set_runtime_env(monkeypatch, tmp_path, capsys) -> None:
    fixture = tmp_path / "multi_turn_turn24_final_output_response_test.txt"
    fixture.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT", raising=False)
    monkeypatch.delenv("CAIS_VERIFY_GENERATED_BRIDGE_IN_GAZEBO", raising=False)

    parser = ui_main._build_arg_parser()
    args = parser.parse_args(
        [
            "--bridge-fixture-final-output",
            str(fixture),
            "--verify-generated-bridge-in-gazebo",
        ]
    )
    ui_main._apply_runtime_bridge_test_args(args, parser)

    assert os.environ["CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT"] == str(
        fixture.resolve()
    )
    assert os.environ["CAIS_VERIFY_GENERATED_BRIDGE_IN_GAZEBO"] == "1"
    output = capsys.readouterr().out
    assert "Runtime bridge fixture final_output:" in output
    assert str(fixture.resolve()) in output
    assert "Generated bridge Gazebo verification: enabled" in output


def test_bridge_fixture_cli_arg_rejects_directory(tmp_path) -> None:
    parser = ui_main._build_arg_parser()
    args = parser.parse_args(["--bridge-fixture-final-output", str(tmp_path)])

    with pytest.raises(SystemExit):
        ui_main._apply_runtime_bridge_test_args(args, parser)


def test_f5_current_file_debug_run_sets_fixture_defaults(monkeypatch, capsys) -> None:
    monkeypatch.delenv("CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT", raising=False)
    monkeypatch.delenv("CAIS_VERIFY_GENERATED_BRIDGE_IN_GAZEBO", raising=False)
    monkeypatch.setattr(ui_main.sys, "gettrace", lambda: object())

    parser = ui_main._build_arg_parser()
    args = parser.parse_args([])
    ui_main._apply_runtime_bridge_test_args(args, parser)

    assert os.environ["CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT"] == str(
        ui_main._F5_DEBUG_BRIDGE_FIXTURE_FINAL_OUTPUT.resolve()
    )
    assert os.environ["CAIS_VERIFY_GENERATED_BRIDGE_IN_GAZEBO"] == "1"
    output = capsys.readouterr().out
    assert "F5 debug runtime bridge fixture final_output:" in output
    assert str(ui_main._F5_DEBUG_BRIDGE_FIXTURE_FINAL_OUTPUT.resolve()) in output
    assert "F5 debug generated bridge Gazebo verification: enabled" in output

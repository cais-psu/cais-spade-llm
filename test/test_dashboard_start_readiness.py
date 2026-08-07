"""Focused UI contracts for Dashboard execution-mode readiness."""

from __future__ import annotations

import inspect

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.pages import dashboard


def test_dashboard_initial_mode_preserves_bridge_execution_mode() -> None:
    source = inspect.getsource(dashboard.render)

    assert dashboard._MODE_LABEL_BY_VALUE["physical"] == "Physical"
    assert "value=_MODE_LABEL_BY_VALUE.get(" in source
    assert 'value="Simulation"' not in source


def test_digital_twin_start_establishes_physical_bridge_context() -> None:
    source = inspect.getsource(SystemBridge.digital_twin_start)

    physical_assignment = 'self.execution_mode = "physical"'
    real_assignment = 'self.robot_env = "real"'
    assert physical_assignment in source
    assert real_assignment in source
    assert source.index(physical_assignment) < source.index(real_assignment)


def test_dashboard_simulation_prerequisite_distinguishes_passive_twin() -> None:
    source = inspect.getsource(dashboard._check_prerequisites)

    assert "simulation_environment_running" in source
    assert "passive_digital_twin_environment_running" in source
    assert "Select Physical mode to start CAIS with this Digital Twin." in source

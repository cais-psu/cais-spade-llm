"""Saved full-scene performance settings, applied only by an explicit launch."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from cais_spade_llm.recovery_framework import ROOT

SIMULATION_SPEEDS = (1, 2, 5, 10)
CONTROLLER_RATES = (250, 500, 1000)
DEFAULT_SETTINGS = {
    "speed": 1,
    "ur_controller_rate_hz": 1000,
    "enable_camera_streams": False,
    "dynamic_shadows": False,
    "gazebo_gui_rate_hz": 30,
}


def simulation_settings(setup: dict) -> dict:
    """Validate optional settings without replacing missing live observations."""
    supplied = setup.get("simulation", {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_SETTINGS):
        raise ValueError("Invalid simulation settings")
    settings = {**DEFAULT_SETTINGS, **supplied}
    if type(settings["speed"]) is not int or settings["speed"] not in SIMULATION_SPEEDS:
        raise ValueError("Simulation speed must be 1, 2, 5, or 10")
    if (type(settings["ur_controller_rate_hz"]) is not int
            or settings["ur_controller_rate_hz"] not in CONTROLLER_RATES):
        raise ValueError("UR controller rate must be 250, 500, or 1000 Hz")
    for key in ("enable_camera_streams", "dynamic_shadows"):
        if type(settings[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if type(settings['gazebo_gui_rate_hz']) is not int or settings['gazebo_gui_rate_hz'] not in (30, 60):
        raise ValueError('Gazebo viewer rate must be 30 or 60 Hz')
    return settings


def launch_arguments(path: Path | None = None) -> str:
    """Read the saved settings at launch, never when constructing a bridge."""
    path = path or ROOT / "cais_spade_llm/initialization/recovery_framework_setup.json"
    setup = json.loads(path.read_text()) if path.exists() else {}
    settings = simulation_settings(setup)
    initial_part = ''
    order_path = setup.get('selected_product_order_file')
    scene_path = setup.get('scene_file')
    if order_path and scene_path:
        order = json.loads((ROOT / order_path).read_text())
        scene = json.loads((ROOT / scene_path).read_text())
        initial_part = next((part for part in order.get('parts', [])
                             if part in scene['Storage']['slots']), '')
    return " ".join((
        shlex.quote(f"kmr_initial_part:={initial_part}"),
        f"simulation_speed:={settings['speed']}",
        f"ur_controller_rate_hz:={settings['ur_controller_rate_hz']}",
        f"enable_camera_streams:={str(settings['enable_camera_streams']).lower()}",
        f"dynamic_shadows:={str(settings['dynamic_shadows']).lower()}",
        f"gazebo_gui_rate_hz:={settings['gazebo_gui_rate_hz']}",
    ))

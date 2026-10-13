from __future__ import annotations

"""Selected CCA specifications for the existing recovery cases."""


_SAFETY_FILES = {
    "Part slippage": (
        "cais_spade_llm/specification/safety/safety_assembly_board-v1_mutex.txt",
        "cais_spade_llm/specification/safety/safety_assembly_board-v1_gear_small_before_KET4_Square_4mm.txt",
    ),
    "Conveyor breakdown": ("cais_spade_llm/specification/safety/safety_Conveyor_breakdown.txt",),
    "ur5e-1 breakdown": ("cais_spade_llm/specification/safety/safety_ur5e-1_breakdown.txt",),
    "Machining breakdown during part processing": (
        "cais_spade_llm/specification/safety/safety_M2_mutex.txt",
        "cais_spade_llm/specification/safety/safety_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing.txt",
    ),
}


def default_safety_files(scenario: str) -> list[str]:
    """Return independent default selections without altering saved selections."""
    if scenario not in _SAFETY_FILES:
        raise ValueError("Select an existing failure scenario")
    selected = list(_SAFETY_FILES[scenario])
    if scenario in {"Conveyor breakdown", "ur5e-1 breakdown"}:
        selected.extend(_SAFETY_FILES["Part slippage"])
    return selected

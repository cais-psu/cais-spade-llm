from __future__ import annotations

"""One plant-wide CCA specification set shared by the existing recovery cases."""


_PLANT_SAFETY_FILES = (
    "cais_spade_llm/specification/safety/safety_assembly_board-v1_mutex.txt",
    "cais_spade_llm/specification/safety/safety_M1_mutex.txt",
    "cais_spade_llm/specification/safety/safety_M2_mutex.txt",
    "cais_spade_llm/specification/safety/safety_assembly_board-v1_trim_before_entry.txt",
    "cais_spade_llm/specification/safety/safety_assembly_board-v1_gear_small_before_KET4_Square_4mm.txt",
    "cais_spade_llm/specification/safety/safety_assembly_board-v1_KET4_Square_4mm_before_RGOCG4-50_Round_4mm.txt",
    "cais_spade_llm/specification/safety/safety_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing.txt",
)

_SAFETY_FILES = {
    scenario: _PLANT_SAFETY_FILES
    for scenario in (
        "Part slippage",
        "Conveyor breakdown",
        "ur5e-1 breakdown",
        "Machining breakdown during part processing",
    )
}


def default_safety_files(scenario: str) -> list[str]:
    """Return independent default selections without altering saved selections.

    Args:
        scenario: Exact configured failure scenario.

    Returns:
        A new list of the scenario's safety specification paths.
    """
    if scenario not in _SAFETY_FILES:
        raise ValueError("Select an existing failure scenario")
    return list(_SAFETY_FILES[scenario])

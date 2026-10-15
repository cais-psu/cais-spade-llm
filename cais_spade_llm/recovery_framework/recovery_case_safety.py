from __future__ import annotations

"""Selected CCA specifications for the existing recovery cases."""


_SAFETY_FILES = {
    "Part slippage": (
        "cais_spade_llm/specification/safety/safety_assembly_board-v1_mutex.txt",
        "cais_spade_llm/specification/safety/safety_assembly_board-v1_gear_small_before_KET4_Square_4mm.txt",
    ),
    "Conveyor breakdown": (
        "cais_spade_llm/specification/safety/safety_assembly_board-v1_mutex.txt",
        "cais_spade_llm/specification/safety/safety_assembly_board-v1_KET4_Square_4mm_before_gear_small.txt",
    ),
    "ur5e-1 breakdown": ("cais_spade_llm/specification/safety/safety_M1_mutex.txt",),
    "Machining breakdown during part processing": (
        "cais_spade_llm/specification/safety/safety_M2_mutex.txt",
        "cais_spade_llm/specification/safety/safety_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing.txt",
    ),
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

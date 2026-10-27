from __future__ import annotations

"""One plant-wide CCA specification set shared by the existing recovery cases, and each case's declared
failure-state assumptions for the CCA safety check."""

from copy import deepcopy

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

# Captures run with the CCA bypassed. The Part slippage harness holds ur5e-3 at its buffer zone 4 pick pose,
# which reaches into the board area while ur5e-4 places there; with the CCA on, ur5e-3 would have waited clear
# (user decision, October 26). Every other failure state that already breaks a rule stops the check.
_CCA_ASSUMPTIONS = {
    "Part slippage": [{"resource": "ur5e-3", "clear_of": ["assembly_board-v1"],
                       "reason": "ur5e-3 is clear of the assembly_board-v1 area until an event moves it, as it "
                                 "would have waited with the CCA on (the capture ran with the CCA bypassed)"}],
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


def default_cca_assumptions(scenario: str) -> list[dict]:
    """Return the failure-state facts the CCA safety check assumes for a scenario's capture.

    Args:
        scenario: Exact configured failure scenario.

    Returns:
        New ``{"resource", "clear_of", "reason"}`` rows; empty when the capture shows all the check reads.
    """
    if scenario not in _SAFETY_FILES:
        raise ValueError("Select an existing failure scenario")
    return deepcopy(_CCA_ASSUMPTIONS.get(scenario, []))

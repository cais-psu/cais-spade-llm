"""
Thin wrapper around pyperplan for in-process PDDL planning.

Returns a list of (action_name, [param1, param2, ...]) tuples
representing the ordered plan, or raises PlannerNoSolutionError.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


class PlannerNoSolutionError(Exception):
    """Raised when pyperplan finds no valid plan."""


def solve(domain_pddl: str, problem_pddl: str) -> list[tuple[str, list[str]]]:
    """
    Run pyperplan BFS on the given PDDL domain and problem strings.

    Returns:
        List of (action_name, [parameters...]) tuples in plan order.
        e.g. [("move-to-pick-location", ["ur5e-localhost", "sg", "loc1"]), ...]

    Raises:
        PlannerNoSolutionError: if no plan exists.
        ImportError: if pyperplan is not installed.
    """
    try:
        from pyperplan.pddl.parser import Parser
        from pyperplan import grounding
        from pyperplan.search.breadth_first_search import breadth_first_search
    except ImportError as e:
        raise ImportError(
            "pyperplan is required for PDDL replanning. "
            "Install with: pip install pyperplan"
        ) from e

    # pyperplan reads from disk
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as df:
        df.write(domain_pddl)
        domain_path = df.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as pf:
        pf.write(problem_pddl)
        problem_path = pf.name

    try:
        parser = Parser(domain_path, problem_path)
        domain = parser.parse_domain()
        problem = parser.parse_problem(domain)
        task = grounding.ground(problem)

        plan = breadth_first_search(task)
    finally:
        os.unlink(domain_path)
        os.unlink(problem_path)

    if plan is None:
        raise PlannerNoSolutionError(
            "pyperplan found no solution for this problem."
        )

    return _parse_plan(plan)


def _parse_plan(plan: Any) -> list[tuple[str, list[str]]]:
    """
    Convert pyperplan plan (list of operators) to (name, params) tuples.

    Pyperplan operator names look like:
        "(move-to-pick-location ur5e-localhost sg loc1)"
    """
    result: list[tuple[str, list[str]]] = []
    for op in plan:
        raw = op.name.strip("() ")
        tokens = raw.split()
        action_name = tokens[0]
        params = tokens[1:]
        result.append((action_name, params))
    return result

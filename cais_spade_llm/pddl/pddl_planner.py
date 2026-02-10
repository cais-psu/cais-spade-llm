"""
Thin wrapper around pyperplan for in-process PDDL planning.

Returns a list of (action_name, [param1, param2, ...]) tuples
representing the ordered plan, or raises PlannerNoSolutionError if
no plan exists.
"""

from __future__ import annotations
import tempfile
import os
import logging
from typing import Any

logger = logging.getLogger(__name__)


class PlannerNoSolutionError(Exception):
    """Raised when pyperplan finds no valid plan."""


def solve(domain_pddl: str, problem_pddl: str) -> list[tuple[str, list[str]]]:
    """
    Run pyperplan on the given domain and problem strings.

    Returns:
        List of (action_name, [parameters...]) tuples in plan order.
        e.g. [("pick-part", ["ur5e-localhost", "sg", "failed-loc-sg"]), ...]

    Raises:
        PlannerNoSolutionError: if pyperplan finds no plan.
        ImportError: if pyperplan is not installed.
    """
    try:
        import pyperplan
        from pyperplan import planner as pyperplan_planner
    except ImportError as e:
        raise ImportError(
            "pyperplan is required for PDDL replanning. "
            "Install it with: pip install pyperplan"
        ) from e

    # Write domain and problem to temp files (pyperplan reads from disk)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as df:
        df.write(domain_pddl)
        domain_path = df.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as pf:
        pf.write(problem_pddl)
        problem_path = pf.name

    try:
        plan = _run_pyperplan(domain_path, problem_path)
    finally:
        os.unlink(domain_path)
        os.unlink(problem_path)

    if plan is None:
        raise PlannerNoSolutionError("pyperplan found no solution for this problem.")

    return _parse_plan(plan)


def _run_pyperplan(domain_path: str, problem_path: str) -> Any:
    """Call pyperplan internals and return the raw plan object."""
    from pyperplan.pddl.parser import Parser
    from pyperplan import grounding
    from pyperplan.search.breadth_first_search import breadth_first_search

    parser = Parser(domain_path, problem_path)
    domain  = parser.parse_domain()
    problem = parser.parse_problem(domain)
    task    = grounding.ground(problem)

    # BFS is complete and correct for small STRIPS problems
    plan = breadth_first_search(task)
    return plan


def _parse_plan(plan: Any) -> list[tuple[str, list[str]]]:
    """
    Convert pyperplan plan (list of operators) to list of (name, params).

    pyperplan operator names look like: "(pick-part ur5e-localhost sg failed-loc-sg)"
    """
    result: list[tuple[str, list[str]]] = []
    for op in plan:
        # op.name is the full string e.g. "(pick-part ur5e-localhost sg failed-loc-sg)"
        raw = op.name.strip("() ")
        tokens = raw.split()
        action_name = tokens[0]
        params      = tokens[1:]
        result.append((action_name, params))
    return result

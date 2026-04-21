from __future__ import annotations

import inspect

from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent
from cais_spade_llm.function_analyzer import FunctionAnalyzer
from cais_spade_llm.resources.robot.robot_primitives import robot_capability_decompositions
from cais_spade_llm.resources.robot.robot_task_specs import (
    robot_task_docstring,
    robot_task_spec_names,
)


def test_robot_task_spec_registry_drives_docstrings_and_decompositions() -> None:
    expected_names = (
        "pick_approach",
        "pick_grasp",
        "place_approach",
        "move_home",
        "place_insert",
    )
    assert robot_task_spec_names() == expected_names

    for function_name in expected_names:
        method = getattr(RobotAgent, function_name)
        assert inspect.getdoc(method) == robot_task_docstring(function_name)

        analyzed = FunctionAnalyzer().analyze_function(method)
        assert analyzed["name"] == function_name
        assert analyzed["description"]

        decomposition = robot_capability_decompositions(
            function_name=function_name,
            resource_jid="ur5e@localhost",
        )
        assert decomposition["function_name"] == function_name
        assert decomposition["bridge_visible_steps"]
        assert decomposition["execution_notes"]

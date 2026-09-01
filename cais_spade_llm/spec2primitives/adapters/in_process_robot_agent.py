"""Read Phase 5.1 context from the exact live in-process RobotAgent."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from copy import deepcopy
from typing import Any, Protocol

from cais_spade_llm.spec2primitives.adapters.dual_gazebo import DUAL_GAZEBO_NAME
from cais_spade_llm.spec2primitives.agents.ra import (
    RAContextHandoffError,
    SelectedRAAssignmentEnvelope,
)

_ROBOT_AGENT_STARTUP_TIMEOUT_SECONDS = 90.0
_ROBOT_AGENT_STARTUP_POLL_SECONDS = 1.0


class InProcessRobotAgentHost(Protocol):
    """Expose the narrow application runtime needed by the RA adapter."""

    resource_agents: Sequence[object]
    system_running: bool
    execution_mode: str
    robot_env: str

    def ros2_proc_status(self, name: str) -> str:
        """Return the fresh state of one application-owned ROS2 process."""
        ...

    def simulation_start_ready(self, force: bool = False) -> tuple[bool, str]:
        """Return whether the selected RobotAgent may use the simulation."""
        ...

    async def start_spec2primitives_robot_agent(
        self,
        resource_jid: str,
        execution_mode: str,
    ) -> object:
        """Start or reuse only the exact RobotAgent needed for context capture."""
        ...

    async def _run_on_agent_runtime(self, coroutine: Awaitable[Any]) -> Any:
        """Run one read operation on the shared SPADE agent event loop."""
        ...


class InProcessRobotAgentCompositionRuntime:
    """Adapt one exact live RobotAgent to the Phase 5.1 context contract."""

    def __init__(self, host: InProcessRobotAgentHost) -> None:
        self._host = host

    async def request_assigned_context(
        self,
        assignment: SelectedRAAssignmentEnvelope,
    ) -> Mapping[str, object]:
        """Return fresh state and the complete composer-visible atomic catalog."""
        selected_agent = await self._selected_or_started_agent(assignment)
        assignment.assert_addressed_to(str(getattr(selected_agent, "jid", "")))
        self._require_alive(selected_agent, assignment.selected_resource_jid)
        self._require_execution_mode(selected_agent, assignment)

        async def _read_context() -> dict[str, object]:
            self._require_alive(selected_agent, assignment.selected_resource_jid)
            snapshot_reader = getattr(selected_agent, "get_recovery_snapshot", None)
            catalog_reader = getattr(
                selected_agent,
                "recovery_synthesis_primitive_catalog",
                None,
            )
            if not callable(snapshot_reader):
                raise RAContextHandoffError(
                    "Selected live RobotAgent does not expose a state snapshot."
                )
            if not callable(catalog_reader):
                raise RAContextHandoffError(
                    "Selected live RobotAgent does not expose a primitive catalog."
                )
            robot_state = snapshot_reader()
            raw_catalog = catalog_reader()
            if not isinstance(robot_state, Mapping) or not robot_state:
                raise RAContextHandoffError(
                    "Selected live RobotAgent returned an empty state snapshot."
                )
            return {
                "resource_jid": assignment.selected_resource_jid,
                "assignment_fingerprint": assignment.fingerprint,
                "robot_state": deepcopy(dict(robot_state)),
                "primitive_catalog": _phase_5_1_primitive_catalog(raw_catalog),
            }

        response = await self._host._run_on_agent_runtime(_read_context())
        if not isinstance(response, Mapping):
            raise RAContextHandoffError(
                "Selected live RobotAgent context response is unavailable."
            )
        return response

    async def author_structural_draft(
        self,
        assignment: SelectedRAAssignmentEnvelope,
        *,
        prompt: str,
        response_format: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Ask only the exact selected RobotAgent to author an unbound draft."""
        selected_agent = await self._selected_or_started_agent(assignment)
        assignment.assert_addressed_to(str(getattr(selected_agent, "jid", "")))
        self._require_alive(selected_agent, assignment.selected_resource_jid)
        self._require_execution_mode(selected_agent, assignment)
        structured_call = getattr(selected_agent, "ask_llm_structured", None)
        if not callable(structured_call):
            raise RAContextHandoffError(
                "Selected live RobotAgent does not expose structured composition."
            )

        async def _author() -> Mapping[str, object]:
            self._require_alive(selected_agent, assignment.selected_resource_jid)
            response = await structured_call(
                prompt,
                response_format=deepcopy(dict(response_format)),
                tools=None,
                max_tool_rounds=0,
                include_agent_instructions=False,
            )
            if not isinstance(response, Mapping):
                raise RAContextHandoffError(
                    "Selected live RobotAgent returned an invalid structural draft."
                )
            return deepcopy(dict(response))

        response = await self._host._run_on_agent_runtime(_author())
        if not isinstance(response, Mapping):
            raise RAContextHandoffError(
                "Selected live RobotAgent structural draft response is unavailable."
            )
        return response

    async def _selected_or_started_agent(
        self,
        assignment: SelectedRAAssignmentEnvelope,
    ) -> object:
        selected_agent = self._selected_agent_or_none(
            assignment.selected_resource_jid
        )
        if selected_agent is not None:
            self._require_execution_mode(selected_agent, assignment)
            if self._is_alive(selected_agent):
                return selected_agent

        if bool(getattr(self._host, "system_running", False)):
            raise RAContextHandoffError(
                f"Selected RobotAgent {assignment.selected_resource_jid} is not running."
            )
        if assignment.selected_execution_mode != "simulation":
            raise RAContextHandoffError(
                "Spec2Primitives can start a context RobotAgent only for the "
                "Phase 4 simulation execution mode."
            )

        try:
            gazebo_state = self._host.ros2_proc_status(DUAL_GAZEBO_NAME)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RAContextHandoffError(
                "Spec2Primitives Dual Gazebo status is unavailable."
            ) from exc
        if gazebo_state != "running":
            raise RAContextHandoffError(
                "Spec2Primitives Dual Gazebo Environment is not running. "
                "Start it above, then retry Phase 5."
            )

        # Phase 4 remains the sole authority for the mode used to create agents.
        self._host.execution_mode = assignment.selected_execution_mode
        self._host.robot_env = "gazebo"
        await self._wait_for_simulation_readiness()
        try:
            selected_agent = (
                await self._host.start_spec2primitives_robot_agent(
                    assignment.selected_resource_jid,
                    assignment.selected_execution_mode,
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RAContextHandoffError(
                f"Selected RobotAgent failed to start: {exc}"
            ) from exc
        if selected_agent is None:
            raise RAContextHandoffError(
                f"Selected RobotAgent {assignment.selected_resource_jid} is not running."
            )

        assignment.assert_addressed_to(str(getattr(selected_agent, "jid", "")))
        self._require_alive(selected_agent, assignment.selected_resource_jid)
        self._require_execution_mode(selected_agent, assignment)
        return selected_agent

    async def _wait_for_simulation_readiness(self) -> None:
        deadline = (
            asyncio.get_running_loop().time()
            + _ROBOT_AGENT_STARTUP_TIMEOUT_SECONDS
        )
        last_reason = "Simulation startup is not ready."
        while True:
            try:
                # Phase 5 is an operator action, so use the authoritative bounded
                # probe instead of the UI timer cache, which may still be pending.
                readiness = self._host.simulation_start_ready(force=True)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise RAContextHandoffError(
                    "Spec2Primitives Dual Gazebo readiness is unavailable."
                ) from exc
            if (
                not isinstance(readiness, tuple)
                or len(readiness) != 2
                or not isinstance(readiness[0], bool)
            ):
                raise RAContextHandoffError(
                    "Spec2Primitives Dual Gazebo readiness response is malformed."
                )
            ready, reason = readiness
            if ready:
                return
            reason_text = str(reason or "").strip()
            if reason_text:
                last_reason = reason_text
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise RAContextHandoffError(
                    "Spec2Primitives Dual Gazebo did not become ready within "
                    f"{_ROBOT_AGENT_STARTUP_TIMEOUT_SECONDS:.0f} seconds: "
                    f"{last_reason}"
                )
            await asyncio.sleep(
                min(_ROBOT_AGENT_STARTUP_POLL_SECONDS, remaining)
            )

    def _selected_agent_or_none(self, selected_resource_jid: str) -> object | None:
        matches = [
            agent
            for agent in tuple(getattr(self._host, "resource_agents", ()) or ())
            if str(getattr(agent, "jid", "")) == selected_resource_jid
        ]
        if len(matches) != 1:
            if not matches:
                return None
            raise RAContextHandoffError(
                f"Selected RobotAgent {selected_resource_jid} is not unique."
            )
        return matches[0]

    @staticmethod
    def _require_execution_mode(
        agent: object,
        assignment: SelectedRAAssignmentEnvelope,
    ) -> None:
        execution_mode = _required_text(
            getattr(agent, "execution_mode", None),
            "selected RobotAgent execution_mode",
        )
        if execution_mode != assignment.selected_execution_mode:
            raise RAContextHandoffError(
                "Selected live RobotAgent execution_mode does not match Phase 4."
            )

    @staticmethod
    def _is_alive(agent: object) -> bool:
        is_alive = getattr(agent, "is_alive", None)
        return callable(is_alive) and is_alive() is True

    @staticmethod
    def _require_alive(agent: object, selected_resource_jid: str) -> None:
        if not InProcessRobotAgentCompositionRuntime._is_alive(agent):
            raise RAContextHandoffError(
                f"Selected RobotAgent {selected_resource_jid} is not running."
            )


def _phase_5_1_primitive_catalog(value: object) -> list[dict[str, object]]:
    """Convert the RA-owned synthesis catalog without changing its symbols."""
    if not isinstance(value, list) or not value:
        raise RAContextHandoffError(
            "Selected live RobotAgent primitive catalog must be non-empty."
        )

    converted: list[dict[str, object]] = []
    symbols: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise RAContextHandoffError(
                f"RobotAgent primitive catalog entry {index} must be an object."
            )
        if item.get("synthesis_hidden") is True:
            raise RAContextHandoffError(
                f"RobotAgent primitive catalog entry {index} is hidden."
            )
        if (
            "primitive_steps" in item
            or "composite_expansion" in item
            or item.get("primitive_kind") == "composite"
        ):
            raise RAContextHandoffError(
                f"RobotAgent primitive catalog entry {index} is composite."
            )

        symbol = _required_text(
            item.get("name"),
            f"primitive catalog entry {index} name",
        )
        if symbol in symbols:
            raise RAContextHandoffError(
                "Selected live RobotAgent primitive symbols must be unique."
            )
        symbols.add(symbol)
        description = _required_text(
            item.get("description"),
            f"primitive catalog entry {index} description",
        )
        parameters = item.get("params")
        required_parameters = item.get("required_params")
        output_schema = item.get("output_schema")
        conditions = item.get("preconditions")
        effects = item.get("effects")
        if not isinstance(parameters, Mapping):
            raise RAContextHandoffError(
                f"RobotAgent primitive catalog entry {index} params must be an object."
            )
        required_names = _required_parameter_names(
            required_parameters,
            parameters,
            index,
        )
        if not isinstance(output_schema, Mapping):
            raise RAContextHandoffError(
                f"RobotAgent primitive catalog entry {index} output_schema must be an object."
            )
        if not isinstance(conditions, Mapping) or not isinstance(effects, Mapping):
            raise RAContextHandoffError(
                f"RobotAgent primitive catalog entry {index} conditions/effects are invalid."
            )

        typed_parameters: list[dict[str, object]] = []
        for name, schema in parameters.items():
            parameter_name = _required_text(
                name,
                f"primitive catalog entry {index} parameter name",
            )
            typed_parameters.append(
                {
                    "name": parameter_name,
                    "type": _parameter_type(
                        schema,
                        index=index,
                        name=parameter_name,
                    ),
                    "required": parameter_name in required_names,
                }
            )
        typed_results: list[dict[str, object]] = []
        for name, schema in output_schema.items():
            result_name = _required_text(
                name,
                f"primitive catalog entry {index} result name",
            )
            typed_results.append(
                {
                    "name": result_name,
                    "type": _result_type(
                        schema,
                        index=index,
                        name=result_name,
                    ),
                }
            )
        converted.append(
            {
                "primitive_symbol": symbol,
                "operation_description": description,
                "typed_parameters": typed_parameters,
                "typed_results": typed_results,
                "invocation_binding": symbol,
                "truthful_limits": _optional_text_list(item, "truthful_limits", index),
                "direct_evidence": _optional_text_list(item, "direct_evidence", index),
                "evaluator_endpoints": _optional_text_list(
                    item,
                    "evaluator_endpoints",
                    index,
                ),
                "conditions": deepcopy(dict(conditions)),
                "effects": deepcopy(dict(effects)),
            }
        )

    if not converted:
        raise RAContextHandoffError(
            "Selected live RobotAgent has no composer-visible atomic primitives."
        )
    return converted


def _required_parameter_names(
    value: object,
    parameters: Mapping[object, object],
    entry_index: int,
) -> set[str]:
    if not isinstance(value, list):
        raise RAContextHandoffError(
            f"RobotAgent primitive catalog entry {entry_index} required_params must be a list."
        )
    names: set[str] = set()
    parameter_names = {str(name) for name in parameters}
    for index, item in enumerate(value):
        name = _required_text(
            item,
            f"primitive catalog entry {entry_index} required_params[{index}]",
        )
        if name not in parameter_names:
            raise RAContextHandoffError(
                f"RobotAgent primitive catalog entry {entry_index} requires an unknown parameter."
            )
        if name in names:
            raise RAContextHandoffError(
                f"RobotAgent primitive catalog entry {entry_index} repeats a required parameter."
            )
        names.add(name)
    return names


def _parameter_type(value: object, *, index: int, name: str) -> str:
    if isinstance(value, str):
        return _required_text(
            value,
            f"primitive catalog entry {index} parameter {name} type",
        )
    if not isinstance(value, Mapping):
        raise RAContextHandoffError(
            f"RobotAgent primitive catalog entry {index} parameter {name} is untyped."
        )
    return _required_text(
        value.get("type"),
        f"primitive catalog entry {index} parameter {name} type",
    )


def _result_type(value: object, *, index: int, name: str) -> str:
    if isinstance(value, str):
        return _required_text(
            value,
            f"primitive catalog entry {index} result {name} type",
        )
    if not isinstance(value, Mapping):
        raise RAContextHandoffError(
            f"RobotAgent primitive catalog entry {index} result {name} is untyped."
        )
    declared_type = value.get("type")
    if declared_type is not None:
        return _required_text(
            declared_type,
            f"primitive catalog entry {index} result {name} type",
        )
    if value:
        return "object"
    raise RAContextHandoffError(
        f"RobotAgent primitive catalog entry {index} result {name} is untyped."
    )


def _optional_text_list(
    item: Mapping[object, object],
    field: str,
    entry_index: int,
) -> list[str]:
    value = item.get(field)
    if value is None:
        return []
    if not isinstance(value, list):
        raise RAContextHandoffError(
            f"RobotAgent primitive catalog entry {entry_index} {field} must be a list."
        )
    return [
        _required_text(
            entry,
            f"primitive catalog entry {entry_index} {field}[{index}]",
        )
        for index, entry in enumerate(value)
    ]


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RAContextHandoffError(f"{field} must be non-empty text.")
    return value

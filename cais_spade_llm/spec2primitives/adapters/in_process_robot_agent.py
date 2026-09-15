from __future__ import annotations

"""Read context and author primitive programs with the exact live RobotAgent."""


import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any, Protocol
from pathlib import Path

from cais_spade_llm.spec2primitives.adapters.dual_gazebo import DUAL_GAZEBO_NAME, read_dual_gazebo_started_at_ns
from cais_spade_llm.spec2primitives.adapters.moveit_plan_only import (
    MoveItPlanOnlyRuntime,
)
from cais_spade_llm.spec2primitives.agents.ra import (
    RAContextHandoffError,
    SelectedRAAssignmentEnvelope,
)

_ROBOT_AGENT_STARTUP_TIMEOUT_SECONDS = 90.0
_ROBOT_AGENT_STARTUP_POLL_SECONDS = 1.0


class InProcessRobotAgentHost(Protocol):
    """Expose the narrow application runtime needed by the RA adapter."""

    resource_agents: Sequence[object]
    _spec2primitives_robot_agent: object | None
    system_running: bool
    execution_mode: str
    robot_env: str

    def ros2_proc_status(self, name: str) -> str:
        """Return the fresh state of one application-owned ROS2 process."""
        ...

    def simulation_start_ready(self, force: bool = False) -> tuple[bool, str]:
        """Return whether the selected RobotAgent may use the simulation."""
        ...

    def hardware_stack_status(self, robot: str) -> dict[str, object]:
        """Read the hardware interlock without starting a hardware process."""
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
    """Adapt the selected RobotAgent for context capture and program authoring."""

    def __init__(
        self,
        host: InProcessRobotAgentHost,
        *,
        moveit_plan_only_runtime: object | None = None,
        contexts_root: Path | None = None,
    ) -> None:
        """Create the adapter with an injectable no-motion MoveIt boundary."""
        self._host = host
        self._moveit_plan_only_runtime = moveit_plan_only_runtime or MoveItPlanOnlyRuntime()
        self._contexts_root = contexts_root or Path(__file__).resolve().parents[1] / "contexts"

    async def request_assigned_context(
        self,
        assignment: SelectedRAAssignmentEnvelope,
    ) -> Mapping[str, object]:
        """Return fresh state and the complete composer-visible atomic catalog."""
        from ..agents.ra.execution_state import execution_busy, execution_custody

        if execution_busy():
            raise RAContextHandoffError("Robot context capture is unavailable during Gazebo execution.")
        acknowledged = await asyncio.to_thread(
            execution_custody, self._contexts_root, assignment.selected_resource_jid,
            _started_at_ns=read_dual_gazebo_started_at_ns(self._host),
        )
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
            state = deepcopy(dict(robot_state))
            if acknowledged is not None:
                state.update(acknowledged)
            # Configuration describes coordinate interpretation, not live TCP
            # feedback. Do not turn absent configuration into a measured pose.
            configuration = getattr(selected_agent, "controller_config", None)
            move_group = (
                configuration.get("move_group") if isinstance(configuration, Mapping) else None
            )
            state["motion_context"] = {
                "source": "controller_config.move_group",
                **{
                    name: (
                        move_group[name]
                        if isinstance(move_group, Mapping)
                        and isinstance(move_group.get(name), str)
                        and move_group[name].strip()
                        else None
                    )
                    for name in ("frame_id", "ee_link", "tcp_link")
                },
            }
            return {
                "resource_jid": assignment.selected_resource_jid,
                "assignment_fingerprint": assignment.fingerprint,
                "robot_state": state,
                "primitive_catalog": _phase_5_1_primitive_catalog(raw_catalog),
            }

        response = await self._host._run_on_agent_runtime(_read_context())
        if not isinstance(response, Mapping):
            raise RAContextHandoffError("Selected live RobotAgent context response is unavailable.")
        return response

    async def author_composition_action(
        self,
        assignment: SelectedRAAssignmentEnvelope,
        *,
        prompt: str,
        response_format: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Return one RA-authored semantic proposal or unsupported decision.

        The owned composer serves evidence requests; shared execution tools and
        recovery instructions must never participate in this call.
        """
        from ..agents.ra.execution_state import execution_busy

        if execution_busy():
            raise RAContextHandoffError("Primitive composition is unavailable during Gazebo execution.")
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
                    "Selected live RobotAgent returned an invalid composition response."
                )
            return deepcopy(dict(response))

        response = await self._host._run_on_agent_runtime(_author())
        if not isinstance(response, Mapping):
            raise RAContextHandoffError(
                "Selected live RobotAgent composition response is unavailable."
            )
        return response

    async def request_primitive_context(
        self, assignment: SelectedRAAssignmentEnvelope, *, root: Path, recipient: str,
        request_ref: Mapping[str, str], thread: str, deadline: float,
    ) -> Mapping[str, Any]:
        """Send one pinned evidence request from the exact assigned RA over SPADE.

        Args:
            assignment: Current selected RA authority.
            root: Authorized interaction root.
            recipient: PA's registered context inbox JID.
            request_ref: Immutable request owned by the active refinement run.
            thread: Host-issued conversation identifier.
            deadline: Absolute monotonic deadline.

        Returns:
            PA's verified context response and its record pin.
        """
        from ..agents.pa.primitive_context_messages import request_context_message
        from ..agents.ra.refinement_records import verify_record

        request = await asyncio.to_thread(verify_record, root, request_ref)
        if request.get("assignment_fingerprint") != assignment.fingerprint:
            raise RAContextHandoffError("Primitive context request belongs to another assignment.")
        agent = await self._selected_or_started_agent(assignment)
        self._require_alive(agent, assignment.selected_resource_jid)
        return await self._host._run_on_agent_runtime(request_context_message(
            agent, root=root, recipient=recipient, request_ref=request_ref,
            thread=thread, deadline=deadline,
        ))

    async def capture_validation_context(
        self, assignment: SelectedRAAssignmentEnvelope, *, profile: Mapping[str, Any],
        _execution_custody: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Measure the selected resource through owned read-only ROS interfaces."""
        async with self.validation_context(assignment, profile=profile, _execution_custody=_execution_custody) as record:
            return dict(record)

    @asynccontextmanager
    async def validation_context(
        self, assignment: SelectedRAAssignmentEnvelope, *, profile: Mapping[str, Any],
        _execution_custody: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Finish authority/custody reads before delivering fresh measured feedback."""
        from .robot_validation_context import MeasuredRobotContextRuntime
        from ..agents.ra.refinement_records import fingerprint
        from ..agents.ra.execution_state import execution_busy, execution_custody

        if _execution_custody is None and execution_busy():
            raise RAContextHandoffError("Composition validation is unavailable during Gazebo execution.")

        # Execution starts its selected RA once, before fresh validation. Later
        # captures must not replace a stopped agent while a program owns it.
        selected = (
            self._require_execution_agent(assignment)
            if _execution_custody is not None
            else await self._selected_or_started_agent(assignment)
        )
        self._require_alive(selected, assignment.selected_resource_jid)
        self._require_execution_mode(selected, assignment)
        if assignment.selected_execution_mode != "simulation":
            raise RAContextHandoffError("The current no-motion validation model supports simulation context only.")

        async def configuration() -> dict[str, Any]:
            value = getattr(selected, "controller_config", None)
            if not isinstance(value, Mapping):
                raise RAContextHandoffError("Selected robot controller configuration is unavailable.")
            return deepcopy(dict(value))

        config = await self._host._run_on_agent_runtime(configuration())
        async def custody() -> Any:
            state = selected.get_recovery_snapshot()
            if not isinstance(state, Mapping) or "held_part" not in state:
                raise RAContextHandoffError("The selected RA's held_part state is unavailable.")
            return deepcopy(state["held_part"])
        from ..agents.ra.composition_context import _without_model_name

        held_part = _without_model_name(await self._host._run_on_agent_runtime(custody()))
        acknowledged = _execution_custody
        if acknowledged is None:
            acknowledged = await asyncio.to_thread(
                execution_custody, self._contexts_root, assignment.selected_resource_jid,
                _started_at_ns=read_dual_gazebo_started_at_ns(self._host),
            )
        if acknowledged is not None:
            held_part = acknowledged["held_part"]
        async with MeasuredRobotContextRuntime().validation_context(
            resource_jid=assignment.selected_resource_jid, assignment_fingerprint=assignment.fingerprint,
            configuration=config, profile=profile,
        ) as measured:
            measured["held_part"] = held_part
            yield measured
            current = await self._host._run_on_agent_runtime(configuration())
            if fingerprint(current) != measured["configuration_sha256"]:
                raise RAContextHandoffError("Robot tool/configuration changed during measurement.")

    async def execution_configuration(
        self, assignment: SelectedRAAssignmentEnvelope, *, start_if_needed: bool = False,
    ) -> Mapping[str, Any]:
        """Read simulation command configuration without starting agents or probing ROS.

        Args:
            assignment: The saved selected simulation resource.
            start_if_needed: Retained for existing callers; execution no longer starts an RA.

        Returns:
            The configured controller and, when present, the selected RA catalog.
        """
        selected = await asyncio.to_thread(self._require_execution_agent, assignment, allow_stopped=True)
        if selected is not None:
            return {
                "configuration": deepcopy(selected.controller_config),
                "primitive_catalog": _phase_5_1_primitive_catalog(selected.recovery_synthesis_primitive_catalog()),
            }
        # Restored programs need controller endpoints, not a SPADE agent or a
        # second robot controller. Read the same exact resource's Gazebo manifest.
        def read() -> dict[str, Any]:
            directory = Path(__file__).resolve().parents[2] / "initialization/resources"
            matches = [
                resource["gazebo"]["controller"]
                for path in sorted(directory.glob("robot_*.json"))
                for resource in json.loads(path.read_text()).values()
                if resource.get("jid") == assignment.selected_resource_jid
            ]
            if len(matches) != 1:
                raise RAContextHandoffError("The selected resource has no unique Gazebo controller configuration.")
            return {"configuration": matches[0], "primitive_catalog": []}

        return await asyncio.to_thread(read)

    def _require_execution_agent(
        self, assignment: SelectedRAAssignmentEnvelope, *, allow_stopped: bool = False,
    ) -> Any:
        """Check simulation ownership; configuration reads may use an absent or stopped RA."""
        if assignment.selected_execution_mode != "simulation":
            raise RAContextHandoffError("Gazebo execution requires simulation mode.")
        if self._host.ros2_proc_status(DUAL_GAZEBO_NAME) != "running":
            raise RAContextHandoffError("Spec2Primitives Dual Gazebo Environment is not running.")
        if (
            self._host.robot_env != "gazebo"
            or self._host.execution_mode != "simulation"
            or self._host.system_running
        ):
            raise RAContextHandoffError(
                "The runtime is not exclusively available for Spec2Primitives simulation."
            )
        for robot in ("xarm6", "ur5e", "dual robots"):
            if self._host.hardware_stack_status(robot).get("overall") not in {"stopped", "idle"}:
                raise RAContextHandoffError(
                    "Hardware stack is active or its stopped state is unavailable."
                )
        selected = self._selected_agent_or_none(assignment.selected_resource_jid)
        # The standalone context-only RA is owned separately from the shared
        # Agent System's resource list. Read that owner without starting an agent.
        standalone = getattr(self._host, "_spec2primitives_robot_agent", None)
        if str(getattr(standalone, "jid", "")) == assignment.selected_resource_jid:
            if selected is not None and selected is not standalone:
                raise RAContextHandoffError(
                    f"Selected RobotAgent {assignment.selected_resource_jid} is not unique."
                )
            selected = standalone
        if selected is None:
            if allow_stopped:
                return None
            raise RAContextHandoffError("The selected RobotAgent is not running.")
        assignment.assert_addressed_to(str(getattr(selected, "jid", "")))
        if not allow_stopped:
            self._require_alive(selected, assignment.selected_resource_jid)
        self._require_execution_mode(selected, assignment)
        if getattr(selected, "context_only", False) is not True:
            raise RAContextHandoffError(
                "Gazebo execution requires the isolated context-only RobotAgent."
            )

        return selected

    async def capture_execution_context(
        self,
        assignment: SelectedRAAssignmentEnvelope,
        *,
        profile: Mapping[str, Any],
        custody: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Measure preparation state using the execution journal's semantic custody."""
        return await self.capture_validation_context(
            assignment, profile=profile, _execution_custody=custody
        )

    async def validate_plan_only_allocation(
        self,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Run no-motion MoveIt checks without activating a RobotAgent."""
        from ..agents.pa.resource_grounding import validate_location_planning_request

        validate_location_planning_request(request)
        try:
            gazebo_state = self._host.ros2_proc_status(DUAL_GAZEBO_NAME)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RAContextHandoffError(
                "Spec2Primitives Dual Gazebo status is unavailable."
            ) from exc
        if gazebo_state != "running":
            raise RAContextHandoffError(
                "Spec2Primitives Dual Gazebo Environment is not running. "
                "Start it above, then retry arm reachability checks."
            )

        # MoveIt supplies planning readiness and live state; RobotAgent startup
        # belongs to the later explicit selected-RA context capture.
        response = await self._moveit_plan_only_runtime.validate_state_locations(
            deepcopy(dict(request))
        )
        if not isinstance(response, Mapping):
            raise RAContextHandoffError("MoveIt location planning response is invalid.")
        return deepcopy(dict(response))

    async def read_resource_base_pose(
        self, *, base_frame: str, target_frame: str
    ) -> Mapping[str, object]:
        """Read advisory TF evidence through the owned running-simulation boundary."""
        if self._host.ros2_proc_status(DUAL_GAZEBO_NAME) != "running":
            raise RAContextHandoffError("Spec2Primitives Dual Gazebo Environment is not running.")
        return await self._moveit_plan_only_runtime.read_resource_base_pose(
            base_frame=base_frame, target_frame=target_frame
        )

    async def _selected_or_started_agent(
        self,
        assignment: SelectedRAAssignmentEnvelope,
    ) -> object:
        selected_agent = await self._selected_or_started_resource(
            assignment.selected_resource_jid,
            assignment.selected_execution_mode,
        )
        assignment.assert_addressed_to(str(getattr(selected_agent, "jid", "")))
        return selected_agent

    async def _selected_or_started_resource(
        self,
        resource_jid: str,
        execution_mode: str,
    ) -> object:
        selected_agent = self._selected_agent_or_none(resource_jid)
        if selected_agent is not None:
            self._require_execution_mode_value(
                selected_agent,
                resource_jid=resource_jid,
                execution_mode=execution_mode,
            )
            if self._is_alive(selected_agent):
                return selected_agent

        if bool(getattr(self._host, "system_running", False)):
            raise RAContextHandoffError(f"Selected RobotAgent {resource_jid} is not running.")
        if execution_mode != "simulation":
            raise RAContextHandoffError(
                "Spec2Primitives can start a context RobotAgent only for the "
                "validated simulation execution mode."
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
                "Start it above, then retry selected RobotAgent context capture."
            )

        # Phase 4 remains the sole authority for the mode used to create agents.
        self._host.execution_mode = execution_mode
        self._host.robot_env = "gazebo"
        await self._wait_for_simulation_readiness()
        return await self._start_selected_resource(resource_jid, execution_mode)

    async def _start_selected_resource(self, resource_jid: str, execution_mode: str) -> object:
        """Start or reuse the exact RA through the host's context-only lifecycle."""
        try:
            selected_agent = await self._host.start_spec2primitives_robot_agent(
                resource_jid,
                execution_mode,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RAContextHandoffError(f"Selected RobotAgent failed to start: {exc}") from exc
        if selected_agent is None:
            raise RAContextHandoffError(f"Selected RobotAgent {resource_jid} is not running.")

        if str(getattr(selected_agent, "jid", "")) != resource_jid:
            raise RAContextHandoffError(
                "Started RobotAgent does not match the PA provisional choice."
            )
        self._require_alive(selected_agent, resource_jid)
        self._require_execution_mode_value(
            selected_agent,
            resource_jid=resource_jid,
            execution_mode=execution_mode,
        )
        return selected_agent

    async def _wait_for_simulation_readiness(self) -> None:
        deadline = asyncio.get_running_loop().time() + _ROBOT_AGENT_STARTUP_TIMEOUT_SECONDS
        last_reason = "Simulation startup is not ready."
        while True:
            try:
                # Phase 5 is an operator action, so use the authoritative bounded
                # probe instead of the UI timer cache, which may still be pending.
                readiness = await asyncio.to_thread(
                    self._host.simulation_start_ready, force=True
                )
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
            await asyncio.sleep(min(_ROBOT_AGENT_STARTUP_POLL_SECONDS, remaining))

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
        InProcessRobotAgentCompositionRuntime._require_execution_mode_value(
            agent,
            resource_jid=assignment.selected_resource_jid,
            execution_mode=assignment.selected_execution_mode,
        )

    @staticmethod
    def _require_execution_mode_value(
        agent: object,
        *,
        resource_jid: str,
        execution_mode: str,
    ) -> None:
        agent_execution_mode = _required_text(
            getattr(agent, "execution_mode", None),
            "selected RobotAgent execution_mode",
        )
        if agent_execution_mode != execution_mode:
            raise RAContextHandoffError(
                f"Selected live RobotAgent {resource_jid} execution_mode does not match."
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
        raise RAContextHandoffError("Selected live RobotAgent primitive catalog must be non-empty.")

    converted: list[dict[str, object]] = []
    symbols: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise RAContextHandoffError(
                f"RobotAgent primitive catalog entry {index} must be an object."
            )
        if item.get("synthesis_hidden") is True:
            raise RAContextHandoffError(f"RobotAgent primitive catalog entry {index} is hidden.")
        if (
            "primitive_steps" in item
            or "composite_expansion" in item
            or item.get("primitive_kind") == "composite"
        ):
            raise RAContextHandoffError(f"RobotAgent primitive catalog entry {index} is composite.")

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
                "parameter_schemas": deepcopy(dict(parameters)),
                "result_schemas": deepcopy(dict(output_schema)),
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

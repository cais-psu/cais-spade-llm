"""Focused no-motion regression tests for dual robots Start Twin RViz startup."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import pytest

from cais_spade_llm.ui import ros2_processes
from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.ros2_processes import ROS2_ENV

_SPEC2PRIMITIVES_GAZEBO = "gazebo_dual_spec2primitives"


def test_ros2_environment_removes_opencv_qt_paths_and_preserves_wsl_display() -> None:
    unset_command = ROS2_ENV.split(";", maxsplit=1)[0]

    assert "QT_QPA_PLATFORM_PLUGIN_PATH" in unset_command
    assert "QT_QPA_FONTDIR" in unset_command
    assert "DISPLAY" not in unset_command.split()
    assert "WAYLAND_DISPLAY" not in unset_command.split()


def test_gazebo_reset_reads_stored_nist_parts_without_robot_commands() -> None:
    """Load the recovery world's stored pegs and completed printer output."""
    bridge = object.__new__(SystemBridge)
    bridge._gazebo_reset_pose_cache = None

    poses = bridge._load_gazebo_reset_model_poses()

    assert set(poses) == {
        "KET4_Square_4mm", "KET8_Square_8mm", "KET12_Square_12mm", "KET16_Square_16mm",
        "RGOCG4-50_Round_4mm", "RGOCG8-50_8mm", "RGOCG12-50_12mm", "RGOCG16-50_16mm",
        "gear_small", "gear_medium", "gear_large",
    }
    assert poses["KET4_Square_4mm"] == (-8.95, 1.88, 1.165, 0.0, 0.0, 1.57079632679)
    assert poses["RGOCG16-50_16mm"] == (-8.95, 2.72, 0.645, 0.0, 0.0, 1.57079632679)
    assert poses["gear_small"] == (0.44, -0.58, 1.11, 0.0, 0.0, 0.0)
    assert poses["gear_medium"] == (0.44, -0.50, 1.11, 0.0, 0.0, 0.0)
    assert poses["gear_large"] == (0.44, -0.42, 1.11, 0.0, 0.0, 0.0)
    assert bridge._load_gazebo_reset_model_poses() == poses


class _GazeboStatusRecorder:
    _BASE_GAZEBO_PROCESS_NAMES = SystemBridge._BASE_GAZEBO_PROCESS_NAMES

    def __init__(self) -> None:
        self.names: set[str] = set()

    def _any_running(self, names: set[str]) -> bool:
        self.names = set(names)
        return _SPEC2PRIMITIVES_GAZEBO in names


def test_spec2primitives_gazebo_is_a_shared_simulation_environment() -> None:
    bridge = _GazeboStatusRecorder()

    assert SystemBridge.simulation_environment_running(bridge) is True
    assert _SPEC2PRIMITIVES_GAZEBO in bridge.names
    assert _SPEC2PRIMITIVES_GAZEBO in SystemBridge._GAZEBO_PROCESS_NAMES


def test_spec2primitives_gazebo_skips_perception_dependent_controller_prewarm() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._gazebo_prewarm_pending = set()

    SystemBridge._queue_gazebo_prewarm(bridge, _SPEC2PRIMITIVES_GAZEBO)

    assert bridge._gazebo_prewarm_pending == set()


def test_forced_simulation_readiness_allows_ros_discovery_spin_to_finish() -> None:
    bridge = object.__new__(SystemBridge)
    bridge._gazebo_prewarm_lock = threading.Lock()
    bridge._gazebo_prewarm_done = threading.Event()
    bridge._gazebo_prewarm_thread = None
    bridge._gazebo_prewarm_pending = set()
    bridge._sim_ready_probe_inflight = False
    bridge._sim_ready_cache_ts = 0.0
    bridge._sim_ready_cache = (False, "Simulation startup check pending.")
    bridge.simulation_environment_running = lambda: True
    probe_timeouts: list[float] = []

    def _probe_sim_services(timeout_sec: float) -> tuple[bool, str]:
        probe_timeouts.append(timeout_sec)
        return True, ""

    bridge._probe_sim_services = _probe_sim_services

    assert SystemBridge.simulation_start_ready(bridge, force=True) == (True, "")
    assert probe_timeouts == [10.0]


class _StandaloneRobotAgent:
    def __init__(self, jid: str = "xarm6@localhost") -> None:
        self.jid = jid
        self.execution_mode = "simulation"
        self.context_only = True
        self.executables: dict[str, Any] = {}
        self.failure_scenarios: list[dict[str, Any]] = []
        self._controller = None
        self.alive = False
        self.start_calls = 0
        self.stop_calls = 0
        self.teardown_calls = 0
        self.container = None

    async def start(self, *, auto_register: bool) -> None:
        assert auto_register is True
        self.start_calls += 1
        self.alive = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self.alive = False

    async def teardown(self) -> None:
        self.teardown_calls += 1

    def is_alive(self) -> bool:
        return self.alive


class _ResourceOnlyAgentCreator:
    def __init__(self, *agents: _StandaloneRobotAgent) -> None:
        self.agents = {
            str(agent.jid).split("@", maxsplit=1)[0]: agent for agent in agents
        }
        self.resource_calls: list[tuple[list[str], str, bool]] = []

    def create_resource_agents(
        self,
        resource_paths: list[str],
        cca_path: str,
        *,
        robot_context_only: bool = False,
    ) -> list[_StandaloneRobotAgent]:
        self.resource_calls.append(
            (resource_paths, cca_path, robot_context_only)
        )
        resource_name = "ur5e" if resource_paths[0].endswith("robot_ur5e.json") else "xarm6"
        return [self.agents[resource_name]]

    def create_product_agents(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("ProductAgent startup is outside Phase 5.1")

    def create_central_controller(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("CCA startup is outside Phase 5.1")

    def create_user(self) -> None:
        raise AssertionError("UserAgent startup is outside Phase 5.1")


def test_spec2primitives_starts_and_reuses_only_the_exact_robot_agent() -> None:
    bridge = object.__new__(SystemBridge)
    agent = _StandaloneRobotAgent()
    creator = _ResourceOnlyAgentCreator(agent)
    bridge.resource_agents = []
    bridge.system_running = False
    bridge._starting = False
    bridge._stopping = False
    bridge.execution_mode = "simulation"
    bridge.robot_env = "gazebo"
    bridge._spec2primitives_robot_agent = None
    bridge._spec2primitives_robot_agent_lifecycle_lock = asyncio.Lock()
    bridge._agent_creator_cached = creator
    xmpp_calls: list[bool] = []
    configure_calls: list[tuple[str, str, str]] = []

    async def _ensure_xmpp_server() -> None:
        xmpp_calls.append(True)

    async def _run_on_agent_runtime(coroutine: Any) -> Any:
        return await coroutine

    def _configure_agent_creator_runtime(
        _creator: Any,
        robot_env: str,
        execution_mode: str,
        perception_backend: str,
    ) -> None:
        configure_calls.append((robot_env, execution_mode, perception_backend))

    bridge._ensure_xmpp_server = _ensure_xmpp_server
    bridge._run_on_agent_runtime = _run_on_agent_runtime
    bridge._configure_agent_creator_runtime = _configure_agent_creator_runtime

    async def _scenario() -> None:
        first = await asyncio.wait_for(
            SystemBridge.start_spec2primitives_robot_agent(
                bridge,
                "xarm6@localhost",
                "simulation",
            ),
            timeout=1.0,
        )
        second = await asyncio.wait_for(
            SystemBridge.start_spec2primitives_robot_agent(
                bridge,
                "xarm6@localhost",
                "simulation",
            ),
            timeout=1.0,
        )
        assert first is agent
        assert second is agent
        await asyncio.wait_for(
            SystemBridge.shutdown_spec2primitives_robot_agent(bridge),
            timeout=1.0,
        )

    asyncio.run(_scenario())

    assert agent.start_calls == 1
    assert agent.stop_calls == 1
    assert agent.teardown_calls == 1
    assert len(creator.resource_calls) == 1
    assert creator.resource_calls[0][0][0].endswith("robot_xarm6.json")
    assert creator.resource_calls[0][2] is True
    assert xmpp_calls == [True]
    assert configure_calls == [("gazebo", "simulation", "none")]
    assert agent.model == "gpt-5.6"
    assert agent.reasoning_effort == "medium"
    assert agent.non_function_model == "gpt-5.6"
    assert agent.non_function_reasoning_effort == "medium"
    assert bridge.resource_agents == []
    assert bridge.system_running is False


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("context_only", False, "not context-only"),
        ("executables", {"pick_approach": object()}, "exposes task tools"),
        (
            "failure_scenarios",
            [{"scenario_id": "lg_slippage"}],
            "exposes failure scenarios",
        ),
        ("_controller", object(), "constructed a controller"),
    ],
)
def test_spec2primitives_rejects_an_invalid_context_only_profile(
    attribute: str,
    value: Any,
    message: str,
) -> None:
    bridge = object.__new__(SystemBridge)
    agent = _StandaloneRobotAgent()
    setattr(agent, attribute, value)
    creator = _ResourceOnlyAgentCreator(agent)
    bridge.resource_agents = []
    bridge.system_running = False
    bridge._starting = False
    bridge._stopping = False
    bridge.execution_mode = "simulation"
    bridge.robot_env = "gazebo"
    bridge._spec2primitives_robot_agent = None
    bridge._spec2primitives_robot_agent_lifecycle_lock = asyncio.Lock()
    bridge._agent_creator_cached = creator

    async def _no_op() -> None:
        return None

    async def _run_on_agent_runtime(coroutine: Any) -> Any:
        return await coroutine

    bridge._ensure_xmpp_server = _no_op
    bridge._run_on_agent_runtime = _run_on_agent_runtime
    bridge._configure_agent_creator_runtime = lambda *_args: None

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(
            SystemBridge.start_spec2primitives_robot_agent(
                bridge,
                "xarm6@localhost",
                "simulation",
            )
        )

    assert agent.start_calls == 0
    assert agent.teardown_calls == 1


def test_spec2primitives_replaces_an_earlier_context_only_robot_agent() -> None:
    bridge = object.__new__(SystemBridge)
    xarm6 = _StandaloneRobotAgent()
    ur5e = _StandaloneRobotAgent("ur5e@localhost")
    creator = _ResourceOnlyAgentCreator(xarm6, ur5e)
    bridge.resource_agents = []
    bridge.system_running = False
    bridge._starting = False
    bridge._stopping = False
    bridge.execution_mode = "simulation"
    bridge.robot_env = "gazebo"
    bridge._spec2primitives_robot_agent = None
    bridge._spec2primitives_robot_agent_lifecycle_lock = asyncio.Lock()
    bridge._agent_creator_cached = creator

    async def _no_op() -> None:
        return None

    async def _run_on_agent_runtime(coroutine: Any) -> Any:
        return await coroutine

    bridge._ensure_xmpp_server = _no_op
    bridge._run_on_agent_runtime = _run_on_agent_runtime
    bridge._configure_agent_creator_runtime = lambda *_args: None

    async def _scenario() -> None:
        await asyncio.wait_for(
            SystemBridge.start_spec2primitives_robot_agent(
                bridge,
                "xarm6@localhost",
                "simulation",
            ),
            timeout=1.0,
        )
        selected = await asyncio.wait_for(
            SystemBridge.start_spec2primitives_robot_agent(
                bridge,
                "ur5e@localhost",
                "simulation",
            ),
            timeout=1.0,
        )
        assert selected is ur5e

    asyncio.run(_scenario())

    assert xarm6.start_calls == 1
    assert xarm6.stop_calls == 1
    assert xarm6.teardown_calls == 1
    assert ur5e.start_calls == 1
    assert ur5e.is_alive() is True
    assert bridge._spec2primitives_robot_agent is ur5e


def test_full_system_start_disposes_spec2primitives_robot_agent_first() -> None:
    bridge = object.__new__(SystemBridge)
    bridge.system_running = False
    bridge._starting = False
    bridge._stopping = False
    bridge.last_error = None
    bridge._ur5e_robot_function_execution_lock = threading.Lock()
    bridge._ur5e_robot_function_execution_active = None
    bridge._ur5e_robot_function_preflight_lock = threading.Lock()
    bridge._ur5e_robot_function_agent_lifecycle_lock = asyncio.Lock()
    bridge._xarm6_robot_function_agent_lifecycle_lock = asyncio.Lock()
    bridge._spec2primitives_robot_agent_lifecycle_lock = asyncio.Lock()
    calls: list[str] = []

    bridge._ur5e_robot_function_agent_handoff_error = lambda: ""
    bridge._xarm6_robot_function_agent_handoff_error = lambda: ""

    async def _record(name: str) -> None:
        calls.append(name)

    bridge._dispose_ur5e_robot_function_agent = lambda: _record("dispose_ur5e")
    bridge._dispose_xarm6_robot_function_agent = lambda: _record("dispose_xarm6")
    bridge._dispose_spec2primitives_robot_agent = lambda: _record("dispose_spec2")
    bridge._start_system_after_ur5e_handoff = lambda: _record("start_full")

    asyncio.run(SystemBridge.start_system(bridge))

    assert calls == [
        "dispose_ur5e",
        "dispose_xarm6",
        "dispose_spec2",
        "start_full",
    ]


class _LaunchRecorder:
    def __init__(self) -> None:
        self.command = ""
        self.ensure_calls: list[dict[str, Any]] = []

    def _ros2_launch_prereq_error(self, _launch_name: str) -> None:
        return None

    def _render_ros2_launch_cmd(self, _launch_name: str) -> str:
        return (
            "ros2 launch cais_lab_robotics dual_robots_hardware_moveit.launch.py "
            "launch_rviz:=false"
        )

    def _start_tracked_ros2_command(
        self,
        _process_name: str,
        command: str,
        *,
        ros_domain_id: int,
    ) -> None:
        assert ros_domain_id == 40
        self.command = command
        return None

    def _digital_twin_dual_robots_processes(
        self,
        _cfg: dict[str, Any],
    ) -> tuple[str, str, str, str]:
        return "xarm6_driver", "ur5e_driver", "", "dual_moveit"

    def _ensure_digital_twin_launch(
        self,
        process_name: str,
        launch_name: str,
        *,
        ros_domain_id: int,
        extra_args: str = "",
    ) -> None:
        self.ensure_calls.append(
            {
                "process_name": process_name,
                "launch_name": launch_name,
                "ros_domain_id": ros_domain_id,
                "extra_args": extra_args,
            }
        )
        return None

    def _start_ur5e_rtde_trajectory_server(
        self,
        _process_name: str,
        *,
        ros_domain_id: int,
    ) -> None:
        assert ros_domain_id == 40
        return None

    def _wait_with_ros2_daemon_retry(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def test_digital_twin_launch_extra_argument_replaces_hard_coded_value() -> None:
    bridge = _LaunchRecorder()

    result = SystemBridge._start_digital_twin_launch(
        bridge,
        "dual_moveit",
        "hardware_dual_robots_moveit",
        ros_domain_id=40,
        extra_args="launch_rviz:=true",
    )

    assert result is None
    assert bridge.command.endswith("launch_rviz:=true")
    assert "launch_rviz:=false" not in bridge.command


def test_dual_robots_monitor_start_explicitly_enables_rviz() -> None:
    bridge = _LaunchRecorder()

    result = SystemBridge._start_digital_twin_dual_robots_hardware_launches(
        bridge,
        {},
        ros_domain_id=40,
        launch_rviz=True,
    )

    assert result is None
    moveit_call = next(
        call
        for call in bridge.ensure_calls
        if call["launch_name"] == "hardware_dual_robots_moveit"
    )
    assert moveit_call["extra_args"] == "launch_rviz:=true"


@pytest.mark.parametrize(
    "parser",
    [SystemBridge._topic_publisher_count_from_output, ros2_processes._topic_publisher_count_from_output],
    ids=["SystemBridge", "ros2_processes"],
)
@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("", None),
        ("Subscription count: 1", None),
        ("Publisher count: 0", 0),
        ("Publisher count: 12\nSubscription count: 0", 12),
        ("Publisher count:\t3", 3),
        ("Publisher count: unknown", None),
        ("Publisher count: -1", None),
        ("Publisher count: 1x", None),
    ],
)
def test_topic_publisher_count_from_output(
    parser: Callable[[str], int | None],
    output: str,
    expected: int | None,
) -> None:
    """Preserve the distinction between zero publishers and unavailable counts."""
    assert parser(output) == expected


@pytest.mark.parametrize(
    "parser",
    [
        SystemBridge._controller_states_from_list_controllers_output,
        ros2_processes._controller_states_from_list_controllers_output,
    ],
    ids=["SystemBridge", "ros2_processes"],
)
@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("", {}),
        ("controller name state\nmissing_state", {}),
        (
            "ControllerState(name='joint_state_broadcaster', state='active')\n"
            "ControllerState(name='ur5e_joint_trajectory_controller', state='inactive')",
            {"joint_state_broadcaster": "active", "ur5e_joint_trajectory_controller": "inactive"},
        ),
        (
            "joint_state_broadcaster joint_state_broadcaster/JointStateBroadcaster active\n"
            "ur5e_joint_trajectory_controller joint_trajectory_controller/JointTrajectoryController inactive\n"
            "unconfigured_controller controller/type unconfigured\n"
            "finalized_controller controller/type finalized",
            {
                "joint_state_broadcaster": "active",
                "ur5e_joint_trajectory_controller": "inactive",
                "unconfigured_controller": "unconfigured",
                "finalized_controller": "finalized",
            },
        ),
        (
            "\x1b[32mur5e_joint_trajectory_controller[controller/type]\x1b[0m \x1b[1;32mACTIVE\x1b[0m",
            {"ur5e_joint_trajectory_controller": "active"},
        ),
        (
            "ControllerState(name='ur5e_joint_trajectory_controller', state='starting')",
            {"ur5e_joint_trajectory_controller": "starting"},
        ),
        ("ur5e_joint_trajectory_controller controller/type starting", {}),
        ("ControllerState(name='ur5e_joint_trajectory_controller', state=)", {}),
        (
            "ControllerState(name='ur5e_joint_trajectory_controller', state='inactive')\n"
            "joint_state_broadcaster joint_state_broadcaster/JointStateBroadcaster active",
            {"ur5e_joint_trajectory_controller": "inactive"},
        ),
    ],
)
def test_controller_states_from_list_controllers_output(
    parser: Callable[[str], dict[str, str]],
    output: str,
    expected: dict[str, str],
) -> None:
    """Retain response-format precedence and existing unknown-state handling."""
    assert parser(output) == expected

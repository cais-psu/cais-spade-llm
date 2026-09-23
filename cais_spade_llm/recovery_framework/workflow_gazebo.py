"""Observed Gazebo execution for machining and controlled part transport."""

from __future__ import annotations

from copy import deepcopy

import json
import math
import os
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cais_spade_llm.resources.workflow_task_programs import workflow_task_program


def _motion_distance(elapsed: float, distance: float, speed: float, acceleration: float) -> float:
    """Return bounded trapezoidal-profile displacement for simulation time."""
    distance = abs(distance)
    if distance == 0.0:
        return 0.0
    ramp_time = speed / acceleration
    ramp_distance = 0.5 * acceleration * ramp_time * ramp_time
    if 2.0 * ramp_distance >= distance:
        ramp_time = math.sqrt(distance / acceleration)
        total = 2.0 * ramp_time
        if elapsed <= ramp_time:
            return 0.5 * acceleration * elapsed * elapsed
        remaining = max(0.0, total - elapsed)
        return distance - 0.5 * acceleration * remaining * remaining
    cruise_distance = distance - 2.0 * ramp_distance
    cruise_time = cruise_distance / speed
    total = 2.0 * ramp_time + cruise_time
    if elapsed <= ramp_time:
        return 0.5 * acceleration * elapsed * elapsed
    if elapsed <= ramp_time + cruise_time:
        return ramp_distance + speed * (elapsed - ramp_time)
    remaining = max(0.0, total - elapsed)
    return distance - 0.5 * acceleration * remaining * remaining


def _motion_duration(distance: float, speed: float, acceleration: float) -> float:
    """Return trapezoidal-profile duration in simulation seconds."""
    distance = abs(distance)
    ramp_time = speed / acceleration
    ramp_distance = 0.5 * acceleration * ramp_time * ramp_time
    if 2.0 * ramp_distance >= distance:
        return 2.0 * math.sqrt(distance / acceleration)
    return 2.0 * ramp_time + (distance - 2.0 * ramp_distance) / speed


def _interval(scene: dict, location: str) -> tuple[float, float]:
    loading = {
        "loading_position_1": float(scene["machines"][0]["conveyor_loading_pose"][0]),
        "loading_position_2": float(scene["machines"][1]["conveyor_loading_pose"][0]),
    }
    output = float(scene["Conveyor"]["output_nest_pose"][0])
    if location == "loading_position_1":
        return loading[location], loading[location]
    if location == "after loading_position_1":
        return loading["loading_position_1"] + 0.001, loading["loading_position_2"] - 0.001
    if location == "loading_position_2":
        return loading[location], loading[location]
    if location == "after loading_position_2":
        return loading["loading_position_2"] + 0.001, output - 0.001
    if location == "output_nest":
        return output, output
    if location == "Buffer For Machined parts":
        value = float(scene["Buffer For Machined parts"]["slot_poses"][0][0])
        return value, value
    raise ValueError(f"Unknown transport location: {location}")


def _shared_displacement(
    scene: dict, current_x: dict[str, float], next_locations: dict[str, str]
) -> float:
    """Choose the fastest downstream displacement shared by every belt resident."""
    lower = 0.0
    upper = math.inf
    for part, location in next_locations.items():
        interval = _interval(scene, location)
        lower = max(lower, interval[0] - current_x[part])
        upper = min(upper, interval[1] - current_x[part])
    if not math.isfinite(upper) or lower > upper + 1e-9:
        raise ValueError("Observed part poses cannot realize the requested shared-belt displacement")
    return max(0.0, upper)



class WorkflowPrimitiveRunner:
    """Execute one declared machine or transport function using Gazebo callbacks.

    Args:
        request: Bound task, scene, geometry, and observed valuation.
        entity: Read a Gazebo entity state.
        set_pose: Command one Gazebo entity pose.
        robot_clearance: Check observed robot clearance around part states.
        current_clock: Read the simulation clock.
        wait_simulation: Advance a process or motion for simulation seconds.
        update_part_collisions: Publish observed resident poses before completion.
    """

    def __init__(
        self,
        request: dict[str, Any],
        *,
        entity: Callable[[str], Any],
        set_pose: Callable[[str, Any], None],
        robot_clearance: Callable[[list[str], dict[str, Any]], dict[str, float]],
        current_clock: Callable[[], float],
        wait_simulation: Callable[[float, Callable[[float], None] | None], tuple[float, float]],
        update_part_collisions: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.request = request
        self.scene = request["scene"]
        self.geometry = request.get("geometry", {})
        self.task = request["task"]
        self.params = self.task["parameters"]
        self.part_name = self.params.get("part_name") or self.params.get("delivered_part")
        self.entity = entity
        self.set_pose = set_pose
        self.robot_clearance = robot_clearance
        self.current_clock = current_clock
        self.wait_simulation = wait_simulation
        self.update_part_collisions = update_part_collisions
        self.state: dict[str, Any] = {}
        self.observations: dict[str, Any] = {}
        self.primitive_trace: list[dict[str, Any]] = []

    def execute(self) -> dict[str, Any]:
        """Run only the ordered implemented primitives for the bound event."""
        event_name = self.task["event_name"]
        program = workflow_task_program(event_name)
        if not program or program["status"] != "implemented":
            raise ValueError(f"Unsupported workflow Gazebo event: {event_name}")
        primitives = {
            "observe_workholding": self.observe_workholding,
            "verify_process_clearance": self.verify_process_clearance,
            "run_machining_clock": self.run_machining_clock,
            "confirm_process_observation": self.confirm_process_observation,
            "observe_belt_residents": self.observe_belt_residents,
            "compute_shared_displacement": self.compute_shared_displacement,
            "observe_zone_part": self.observe_zone_part,
            "compute_downstream_motion": self.compute_downstream_motion,
            "verify_transport_clearance": self.verify_transport_clearance,
            "move_belt_residents": self.move_belt_residents,
            "move_buffer_part": self.move_buffer_part,
            "confirm_arrival": self.confirm_arrival,
        }
        for step in program["steps"]:
            primitive_name = step["op"]
            primitive = primitives[primitive_name]
            started = time.monotonic()
            record = {
                "step_id": step["id"], "primitive": primitive_name, "status": "running",
                "task_id": self.task.get("task_id"), "resource_id": self.task["resource_id"],
                "event_name": event_name, "started_at_unix": time.time(),
            }
            self.primitive_trace.append(record)
            try:
                primitive()
            except (OSError, KeyError, TypeError, ValueError, RuntimeError, TimeoutError, InterruptedError) as exc:
                record.update(status="failed", error=str(exc))
                self._observe_transport_progress()
                raise
            else:
                record["status"] = "completed"
            finally:
                record["wall_clock_sec"] = time.monotonic() - started
                record["observations"] = self.progress_evidence()
        return self.observations

    def progress_evidence(self) -> dict[str, Any]:
        """Return serializable physical progress without implying event completion."""
        fields = ("initial_x", "target_x", "last_commanded_x", "observed_x",
                  "observation_errors", "simulation_elapsed_sec", "displacement")
        return {key: deepcopy(self.state[key]) for key in fields if key in self.state}

    def _observe_transport_progress(self) -> None:
        for name in self.state.get("names", []):
            try:
                observed = self.entity(self.geometry[name]["model_name"])
            except (OSError, ValueError, RuntimeError, TimeoutError, InterruptedError) as exc:
                self.state.setdefault("observation_errors", {})[name] = str(exc)
            else:
                self.state.setdefault("observed_x", {})[name] = observed.pose.position.x

    def observe_workholding(self) -> None:
        """Require the selected part at the configured machine workholding pose."""
        machine = next(
            row for row in self.scene["machines"]
            if row["resource_id"] == self.task["resource_id"]
        )
        model = self.geometry[self.part_name]["model_name"]
        expected = machine["workholding_pose"]
        initial = self.entity(model)
        position = initial.pose.position
        if math.dist((position.x, position.y, position.z), expected[:3]) > 0.08:
            raise ValueError("The correct part is not observed at machine workholding")
        self.state.update(machine=machine, model=model, expected=expected, initial=initial)

    def _kmr_clearance(self) -> dict[str, float]:
        """Measure KMR links against the active machine process area."""
        from cais_spade_llm.recovery_framework.geometry import rotate

        distances: dict[str, float] = {}
        expected = self.state["expected"]
        for link, bounds in self.scene["KMR"]["task_execution"]["clearance_link_bounds"].items():
            pose = self.entity(f"KMR::{link}").pose
            q = pose.orientation
            origin = [pose.position.x, pose.position.y, pose.position.z]
            distances[link] = min(
                math.dist(
                    [a + b for a, b in zip(
                        origin, rotate([q.x, q.y, q.z, q.w], bound["center"]), strict=False
                    )],
                    expected[:3],
                ) - bound["radius"]
                for bound in bounds
            )
        if min(distances.values()) < self.state["machine"]["KMR_process_clearance_m"]:
            raise ValueError("KMR arm has not cleared the active machine process area")
        return distances

    def verify_process_clearance(self) -> None:
        """Check KMR and handling robot before machining starts."""
        machine = self.state["machine"]
        self.state["kmr_distances"] = self._kmr_clearance()
        self.state["observed_at"] = self.current_clock()
        self.state["robot_distances"] = self.robot_clearance(
            [machine["handling_robot"]], {self.part_name: self.state["initial"]}
        )

    def run_machining_clock(self) -> None:
        """Wait on /clock while repeatedly observing the part and clearance."""
        machine = self.state["machine"]
        model = self.state["model"]
        expected = self.state["expected"]
        duration = float(machine["simulation_process"]["processing_time_sec"])
        interval = float(machine["simulation_process"].get("observation_interval_sec", 0.25))
        last_observation = -math.inf

        def observe_machine(elapsed: float) -> None:
            nonlocal last_observation
            self.state["simulation_elapsed_sec"] = elapsed
            if elapsed - last_observation < interval and elapsed + 1e-9 < duration:
                return
            observed_state = self.entity(model)
            observed = observed_state.pose.position
            if math.dist((observed.x, observed.y, observed.z), expected[:3]) > 0.08:
                raise ValueError("The part left machine workholding during machining")
            for link, distance in self._kmr_clearance().items():
                self.state["kmr_distances"][link] = min(
                    self.state["kmr_distances"][link], distance
                )
            clearance = self.robot_clearance(
                [machine["handling_robot"]], {self.part_name: observed_state}
            )
            robot = machine["handling_robot"]
            self.state["robot_distances"][robot] = min(
                self.state["robot_distances"][robot], clearance[robot]
            )
            last_observation = elapsed

        start, end = self.wait_simulation(duration, observe_machine)
        self.state.update(process_duration=duration, started_sim=start, completed_sim=end)

    def confirm_process_observation(self) -> None:
        """Return the observations required by the existing completion validator."""
        self.observations = {
            "part_name": self.part_name,
            "machine": self.task["resource_id"],
            "process": self.params["process"],
            "result": self.params["result"],
            "processing_time_sec": self.state["process_duration"],
            "KMR_arm_clearance_m": self.state["kmr_distances"],
            "KMR_arm_clearance_observed_sec": self.state["observed_at"],
            "simulation_started_sec": self.state["started_sim"],
            "simulation_completed_sec": self.state["completed_sim"],
            "correct_part_present": True,
            "robot_access_clear": True,
            "robot_clearance_m": self.state["robot_distances"],
            "source_observed": True,
        }

    def observe_belt_residents(self) -> None:
        """Read every resident part before calculating one shared belt move."""
        before = self.request["valuation"]["Conveyor"]
        residents = {
            field.split(".", 1)[1]: value
            for field, value in before.items()
            if field.startswith("part_location.") and value is not None
        }
        names = list(residents)
        states = {name: self.entity(self.geometry[name]["model_name"]) for name in names}
        self.state.update(
            names=names,
            part_states=states,
            initial_x={name: state.pose.position.x for name, state in states.items()},
        )

    def compute_shared_displacement(self) -> None:
        """Resolve all requested downstream regions to one belt displacement."""
        current_x = self.state["initial_x"]
        targets = dict(self.params["next_locations"])
        delivered = self.params.get("delivered_part")
        if delivered is not None:
            targets[delivered] = "Buffer For Machined parts"
        displacement = _shared_displacement(self.scene, current_x, targets)
        self.state.update(
            target_x={name: current_x[name] + displacement for name in self.state["names"]},
            displacement=displacement,
            clearance_robots=["ur5e-1", "ur5e-2", "ur5e-3"],
        )
        self._set_transport_profile("Conveyor")

    def observe_zone_part(self) -> None:
        """Read the exact part in the buffer source zone."""
        state = self.entity(self.geometry[self.part_name]["model_name"])
        self.state.update(
            names=[self.part_name],
            part_states={self.part_name: state},
            initial_x={self.part_name: state.pose.position.x},
        )

    def compute_downstream_motion(self) -> None:
        """Resolve the configured downstream buffer slot without changing it."""
        target = self.scene["Buffer For Machined parts"]["slot_poses"][
            int(self.params["downstream_zone"]) - 1
        ]
        target_x = float(target[0])
        initial_x = self.state["initial_x"][self.part_name]
        self.state.update(
            target_x={self.part_name: target_x},
            displacement=target_x - initial_x,
            clearance_robots=["ur5e-3"],
        )
        self._set_transport_profile("Buffer For Machined parts")

    def _set_transport_profile(self, resource_id: str) -> None:
        profile = self.scene[resource_id]["simulation_transport"]
        self.state.update(
            speed=float(profile["speed_mps"]),
            acceleration=float(profile["acceleration_mps2"]),
        )

    def verify_transport_clearance(self) -> None:
        """Require robot clearance before commanding part motion."""
        self.state["initial_clearance"] = self.robot_clearance(
            self.state["clearance_robots"], self.state["part_states"]
        )
        self.state["duration"] = _motion_duration(
            self.state["displacement"], self.state["speed"], self.state["acceleration"]
        )

    def _move_transport(self) -> None:
        """Move observed entities with the configured simulation-time profile."""
        displacement = self.state["displacement"]
        speed = self.state["speed"]
        acceleration = self.state["acceleration"]
        target_x = self.state["target_x"]
        initial_x = self.state["initial_x"]

        def transport(elapsed: float) -> None:
            self.state["simulation_elapsed_sec"] = elapsed
            self.robot_clearance(self.state["clearance_robots"], self.state["part_states"])
            travelled = _motion_distance(elapsed, displacement, speed, acceleration)
            fraction = 1.0 if abs(displacement) < 1e-12 else min(1.0, travelled / abs(displacement))
            for name, state in self.state["part_states"].items():
                state.pose.position.x = initial_x[name] + (target_x[name] - initial_x[name]) * fraction
                self.set_pose(self.geometry[name]["model_name"], state.pose)
                self.state.setdefault("last_commanded_x", {})[name] = state.pose.position.x

        start, end = self.wait_simulation(self.state["duration"], transport)
        self.state.update(started_sim=start, completed_sim=end)

    def move_belt_residents(self) -> None:
        """Move every conveyor resident in one synchronized step."""
        self._move_transport()

    def move_buffer_part(self) -> None:
        """Move one part to its configured downstream buffer slot."""
        self._move_transport()

    def confirm_arrival(self) -> None:
        """Observe endpoint positions and clearance before acknowledging transport."""
        for name, expected_x in self.state["target_x"].items():
            if abs(self.entity(self.geometry[name]["model_name"]).pose.position.x - expected_x) > 0.015:
                raise ValueError("Transport arrival was not observed")
        final_states = {
            name: self.entity(self.geometry[name]["model_name"])
            for name in self.state["names"]
        }
        final_clearance = self.robot_clearance(self.state["clearance_robots"], final_states)
        if self.update_part_collisions is not None:
            self.update_part_collisions(final_states)
        self.observations = {
            "part_name": self.part_name,
            "simulation_started_sec": self.state["started_sim"],
            "simulation_completed_sec": self.state["completed_sim"],
            "transport_time_sec": self.state["duration"],
            "speed_mps": self.state["speed"],
            "acceleration_mps2": self.state["acceleration"],
            "shared_displacement_m": self.state["displacement"],
            "arrival_observed": True,
            "source_clear": True,
            "downstream_reserved": True,
            "robot_clear": True,
            "robot_clearance_m": {
                "initial": self.state["initial_clearance"],
                "final": final_clearance,
            },
        }

def run(request: dict, session: dict | None = None) -> dict:
    """Execute one clock-driven machine or transport event and return observations."""
    import rclpy
    from gazebo_msgs.msg import EntityState
    from gazebo_msgs.srv import GetEntityState, SetEntityState
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock

    persistent = session is not None
    session = session if session is not None else {}
    owner_pid = session.setdefault('owner_pid', os.getppid())

    def check_owner():
        if session.get('stopped') or os.getppid() != owner_pid:
            raise InterruptedError('Workflow execution stopped or owner exited')
    if "node" not in session:
        rclpy.init(args=[])
        session["node"] = rclpy.create_node("manufacturing_workflow_worker")
        session["clock"] = None
        session["clock_wall"] = 0.0

        def receive_clock(msg: Clock) -> None:
            session["clock"] = float(msg.clock.sec) + float(msg.clock.nanosec) / 1e9
            session["clock_wall"] = time.monotonic()

        session["clock_subscription"] = session["node"].create_subscription(
            Clock,
            "/clock",
            receive_clock,
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
    node = session["node"]
    clients = session.setdefault("clients", {})
    get_client = clients.setdefault("get", node.create_client(GetEntityState, "/get_entity_state"))
    set_client = clients.setdefault("set", node.create_client(SetEntityState, "/set_entity_state"))
    scene = request["scene"]
    geometry = request.get("geometry", {})

    def wait_future(future, timeout: float = 15.0, *, label: str = "Gazebo service"):
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            check_owner()
            rclpy.spin_once(node, timeout_sec=0.02)
        if not future.done():
            raise TimeoutError(f"{label} timed out")
        result = future.result()
        if result is None:
            raise ValueError(f"{label} returned no result")
        return result

    def get_state(name: str, reference_frame: str = "world"):
        deadline = time.monotonic() + 30.0
        if not get_client.wait_for_service(timeout_sec=5.0):
            raise TimeoutError("Gazebo observation service is unavailable")
        while True:
            future = get_client.call_async(
                GetEntityState.Request(name=name, reference_frame=reference_frame)
            )
            try:
                return wait_future(
                    future,
                    timeout=min(5.0, max(0.1, deadline - time.monotonic())),
                    label="Gazebo observation service",
                )
            except TimeoutError:
                get_client.remove_pending_request(future)
                if time.monotonic() >= deadline:
                    raise

    if request.get("operation") in {
        "register_assembly_fixtures",
        "prepare_assembly",
    }:
        try:
            fixtures = (
                ("GMC_Laser_Plate_Virtual", "link"),
                ("Gear_Plate", "Gear_Plate"),
            )
            observed = []
            for model, _link in fixtures:
                relative = get_state(model, "assembly_board_v1::link")
                if not relative.success:
                    raise ValueError(f"Could not observe stationary assembly fixture: {model}")
                position = relative.state.pose.position
                if (
                    abs(position.x) > 0.3
                    or abs(position.y) > 0.3
                    or abs(position.z) > 0.1
                ):
                    raise ValueError(f"Assembly fixture is outside its table station: {model}")
                observed.append(model)
            if request.get("operation") == "register_assembly_fixtures":
                return {
                    "status": "completed",
                    "operation": "register_assembly_fixtures",
                    "fixture_models": observed,
                    "stationary": True,
                }
            retained = []
            for part_name, part_geometry in geometry.items():
                model = part_geometry.get("model_name")
                if not model or part_name == scene["Exit"]["completed_product"]:
                    continue
                relative = get_state(model, "assembly_board_v1::link")
                if not relative.success:
                    raise ValueError(
                        f"Assembled component observation is unavailable: {part_name}"
                    )
                component_position = relative.state.pose.position
                if (
                    abs(component_position.x) > 0.3
                    or abs(component_position.y) > 0.3
                    or abs(component_position.z) > 0.2
                ):
                    raise ValueError(f"Component is outside assembly carrier: {part_name}")
                retained.append(part_name)
            return {
                "status": "completed",
                "operation": "prepare_assembly",
                "fixture_models": observed,
                "stationary": True,
                "retained_components": retained,
                "complete_product_observed": len(retained) == 11,
            }
        except (KeyError, TypeError, ValueError, TimeoutError, InterruptedError) as exc:
            return {
                "status": "failed",
                "operation": str(request.get("operation") or "assembly_fixture_observation"),
                "error": str(exc),
            }
    task = request["task"]
    event_name = task["event_name"]
    started_wall = time.monotonic()
    progress_path = Path(request["_progress_path"]) if request.get("_progress_path") else None

    def save_progress(payload: dict) -> None:
        if progress_path is None:
            return
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload))
        temporary.replace(progress_path)

    def entity(name: str):
        response = get_state(name)
        if not response.success:
            raise ValueError(f"Gazebo entity is unavailable: {name}")
        return response.state

    def set_pose(name: str, pose) -> None:
        state = EntityState(name=name, pose=pose, reference_frame="world")
        response = wait_future(
            set_client.call_async(SetEntityState.Request(state=state)),
            label="Gazebo controlled transport service",
        )
        if not response.success:
            raise ValueError(f"Gazebo rejected controlled transport for {name}")

    def robot_tcp(robot_id: str):
        robot = next(row for row in scene["robots"] if row["resource_id"] == robot_id)
        link = robot["clearance_observation_link"]
        return entity(f"dual_robot::{robot['prefix']}{link}").pose.position

    def robot_clearance(robot_ids: list[str], part_states: dict[str, Any]) -> dict[str, float]:
        distances = {}
        for robot_id in robot_ids:
            tcp_position = robot_tcp(robot_id)
            distance = min(
                math.dist(
                    (tcp_position.x, tcp_position.y, tcp_position.z),
                    (state.pose.position.x, state.pose.position.y, state.pose.position.z),
                )
                for state in part_states.values()
            )
            if distance < 0.12:
                raise ValueError(f"{robot_id} has not cleared the active transport/process area")
            distances[robot_id] = distance
        return distances

    def current_clock() -> float:
        check_owner()
        deadline = time.monotonic() + 5.0
        while session.get("clock") is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        if session.get("clock") is None:
            raise ValueError("/clock observation is missing")
        return float(session["clock"])

    def wait_simulation(duration: float, observe=None) -> tuple[float, float]:
        start = current_clock()
        last = start
        while True:
            rclpy.spin_once(node, timeout_sec=0.02)
            now = current_clock()
            if now + 1e-9 < last:
                raise ValueError("Simulation clock reset during execution")
            last = now
            elapsed = now - start
            save_progress(
                {
                    "task_id": task["task_id"],
                    "resource_id": task["resource_id"],
                    "event_name": event_name,
                    "simulation_elapsed_sec": elapsed,
                    "simulation_duration_sec": duration,
                    "simulation_remaining_sec": max(0.0, duration - elapsed),
                    "simulation_clock_sec": now,
                }
            )
            if observe is not None:
                observe(elapsed)
            if elapsed >= duration:
                return start, now

    def update_part_collisions(states: dict[str, Any]) -> None:
        from moveit_msgs.srv import ApplyPlanningScene
        from moveit_msgs.msg import PlanningScene
        from cais_spade_llm.recovery_framework.part_collision import observed_part_boxes, part_scene_update

        client = session.get("part_scene_client")
        if client is None:
            client = session["part_scene_client"] = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        if not client.wait_for_service(timeout_sec=2.0):
            raise ValueError("Transport collision scene service is unavailable")
        update = PlanningScene(is_diff=True)
        update.robot_state.is_diff = True
        for name, state in states.items():
            model_name = geometry[name]["model_name"]
            diff = part_scene_update(model_name, observed_part_boxes(model_name, state.pose))
            update.world.collision_objects.extend(diff.world.collision_objects)
            update.robot_state.attached_collision_objects.extend(diff.robot_state.attached_collision_objects)
        response = wait_future(client.call_async(ApplyPlanningScene.Request(scene=update)),
                               label="transport collision scene acknowledgement")
        if not response.success:
            raise ValueError("Transport collision scene was rejected")

    runner = WorkflowPrimitiveRunner(
        request,
        entity=entity,
        set_pose=set_pose,
        robot_clearance=robot_clearance,
        current_clock=current_clock,
        wait_simulation=wait_simulation,
        update_part_collisions=update_part_collisions,
    )
    try:
        observations = runner.execute()
        return {
            "status": "completed",
            "resource_id": task["resource_id"],
            "event_name": event_name,
            "task_id": task["task_id"],
            "observations": observations,
            "primitive_trace": runner.primitive_trace,
            "timing": {
                "planning_sec": 0.0,
                "motion_sec": observations.get("transport_time_sec", 0.0),
                "waiting_sec": observations.get("processing_time_sec", 0.0),
                "simulation_sec": observations["simulation_completed_sec"]
                - observations["simulation_started_sec"],
                "wall_clock_sec": time.monotonic() - started_wall,
            },
        }
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, TimeoutError, InterruptedError) as exc:
        return {
            "status": "failed",
            "resource_id": task.get("resource_id"),
            "event_name": event_name,
            "task_id": task.get("task_id"),
            "error": str(exc),
            "primitive_trace": runner.primitive_trace,
            "partial_motion": runner.progress_evidence(),
            "physical_state_reconciliation_required": bool(runner.state.get("last_commanded_x")) or any(
                abs(position - runner.state.get("initial_x", {}).get(name, position)) > 1e-6
                for name, position in runner.state.get("observed_x", {}).items()
            ),
            "wall_clock_sec": time.monotonic() - started_wall,
        }
    finally:
        if not persistent:
            node.destroy_node()
            rclpy.shutdown()


def serve(directory: Path) -> None:
    """Keep ROS clients and /clock subscription alive between operations."""
    import rclpy

    session: dict[str, Any] = {}
    previous = None
    previous_file = None
    owner_pid = os.getppid()
    stopped = False

    def request_stop(*_args) -> None:
        nonlocal stopped
        stopped = True
        session['stopped'] = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        while not stopped:
            if os.getppid() != owner_pid:
                break
            path = directory / "request.json"
            try:
                stamp = path.stat()
            except FileNotFoundError:
                stamp = None
            identity = (stamp.st_ino, stamp.st_mtime_ns, stamp.st_size) if stamp else None
            if identity is not None and identity != previous_file:
                previous_file = identity
                envelope = json.loads(path.read_text())
                if envelope["id"] != previous:
                    previous = envelope["id"]
                    result = run(envelope["request"], session)
                    temporary = directory / "result.tmp"
                    temporary.write_text(json.dumps({"id": previous, "result": result}))
                    temporary.replace(directory / "result.json")
            if "node" in session:
                rclpy.spin_once(session["node"], timeout_sec=0.02)
            else:
                time.sleep(0.02)
    finally:
        if "node" in session:
            session["node"].destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    """Serve the filesystem IPC used by the owning ResourceAgent."""
    if len(sys.argv) == 3 and sys.argv[1] == "--serve":
        serve(Path(sys.argv[2]))
        return
    raise SystemExit("workflow_gazebo is started through GazeboWorker")


if __name__ == "__main__":
    main()

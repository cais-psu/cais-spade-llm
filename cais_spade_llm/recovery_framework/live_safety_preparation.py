"""Registered Gazebo readers and CCA-owned, non-dispatching evidence assembly.

Only the configured idle, unchanged-custody contract is supported. No ROS writer,
primitive executor or CCA permission method is exposed by this module.
"""

from __future__ import annotations

import json
import math
import threading
import time
from copy import deepcopy
from fractions import Fraction
from functools import partial

from cais_spade_llm.recovery_framework import fingerprint
from cais_spade_llm.resources.environment_models import build_environment_models


def controller_goal_identity(observations: dict | None) -> dict | None:
    """Compare controller incarnations and command revisions, retaining raw stamps separately."""
    if observations is None:
        return None
    changing = {"sequence", "simulation_time", "observed_monotonic", "positions", "received_monotonic"}
    return {name: {key: deepcopy(value) for key, value in row.items() if key not in changing}
            for name, row in observations.items()}


class GazeboSafetyReader:
    """Read one physics-thread snapshot through the registered observation service."""

    def __init__(self, configuration: dict):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from std_srvs.srv import Trigger

        self.configuration = deepcopy(configuration)
        self._context = rclpy.context.Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node("recovery_safety_reader", context=self._context)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        self._client = self._node.create_client(Trigger, configuration["observation_service"])
        self._type = Trigger
        self._cached = None
        self._lock = threading.RLock()
        self._snapshot_lock = threading.RLock()
        from action_msgs.msg import GoalStatusArray
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        self._goal_status = {}
        self._subscriptions = []
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        for topic in configuration["controller_status_topics"]:
            self._subscriptions.append(
                self._node.create_subscription(
                    GoalStatusArray,
                    topic,
                    partial(self._record_goals, topic),
                    qos,
                )
            )

    def _record_goals(self, topic, message, info):
        with self._lock:
            self._goal_status[topic] = {
                "received_monotonic": time.monotonic(),
                "publisher_gid": list(info.publisher_gid),
                "goals": [
                    {"id": list(row.goal_info.goal_id.uuid), "status": row.status}
                    for row in message.status_list
                ],
            }

    def idle_goals(self):
        """Require a connected authoritative status publisher and no active goal."""
        if self.configuration.get("controller_state_services"):
            return self._idle_service_goals()
        with self._lock:
            for topic in self.configuration["controller_status_topics"]:
                publishers = self._node.get_publishers_info_by_topic(topic)
                if (
                    len(publishers) != 1
                    or topic not in self._goal_status
                    or list(publishers[0].endpoint_gid) != self._goal_status[topic]["publisher_gid"]
                ):
                    raise ValueError("Complete controller goal observations unavailable: " + topic)
                if any(row["status"] in (1, 2, 3) for row in self._goal_status[topic]["goals"]):
                    raise ValueError(
                        "An active controller goal prevents idle preparation: " + topic
                    )
            return deepcopy(self._goal_status)

    def _idle_service_goals(self):
        sources = self.configuration["controller_state_services"]
        covered = [topic for row in sources.values() for topic in row["covers"]]
        if (len(covered) != len(set(covered))
                or set(covered) != set(self.configuration["controller_status_topics"])):
            raise ValueError("Controller observation owners do not cover all configured endpoints")
        result = {}
        for service, configuration in sources.items():
            client = self._node.create_client(self._type, service)
            try:
                reply = self._call(client, self._type.Request())
                if not reply.success:
                    raise ValueError("Controller state unavailable: " + service + ": " + reply.message)
                row = json.loads(reply.message)
            finally:
                self._node.destroy_client(client)
            stamp = row.get("simulation_time")
            now = self.snapshot()["simulation_time"]
            if (row.get("version") != 1 or not row.get("instance_id")
                    or type(stamp) not in (int, float) or not math.isfinite(stamp)
                    or abs(now - stamp) > 2):
                raise ValueError("Stale or incomplete controller state: " + service)
            if (row.get("has_active_goal") is not False or row.get("has_pending_goal") is not False
                    or row.get("holding") is not True):
                raise ValueError("Controller is not holding without active goals: " + service)
            if any(row.get(key) != value for key, value in configuration.get("required_values", {}).items()):
                raise ValueError("Controller observation contract is unavailable: " + service)
            result[service] = {**row, "observed_monotonic": time.monotonic(),
                               "covers": deepcopy(configuration["covers"])}
        return result

    def _call(self, client, request, timeout=2.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise ValueError("Gazebo read-only service unavailable: " + client.srv_name)
        future = client.call_async(request)
        completed = threading.Event()
        future.add_done_callback(lambda _: completed.set())
        if not completed.wait(timeout):
            future.cancel()
            raise ValueError("Gazebo read-only service timed out: " + client.srv_name)
        result = future.result()
        if result is None:
            raise ValueError("Gazebo read-only service returned no observation")
        return result

    def parameter(self, node: str, name: str):
        """Read a declared string parameter without changing controller state."""
        from rcl_interfaces.srv import GetParameters

        client = self._node.create_client(GetParameters, node.rstrip("/") + "/get_parameters")
        try:
            result = self._call(client, GetParameters.Request(names=[name]))
            if len(result.values) != 1 or result.values[0].type != 4:
                raise ValueError("Required string parameter unavailable: " + node + "/" + name)
            return result.values[0].string_value
        finally:
            self._node.destroy_client(client)

    def snapshot(self, *, max_age=2.0, refresh=False):
        """Return a finite stamped physics snapshot, retaining one capture window."""
        with self._snapshot_lock:
            if (
                refresh
                or self._cached is None
                or time.monotonic() - self._cached["observed_monotonic"] > max_age
            ):
                result = self._call(self._client, self._type.Request())
                if not result.success:
                    raise ValueError(result.message)
                row = json.loads(result.message)
                json.dumps(row, allow_nan=False)
                if (
                    row.get("version") != 1
                    or row.get("attachment_complete") is not True
                    or not row.get("instance_id")
                ):
                    raise ValueError("Incomplete Gazebo attachment snapshot")
                row["observed_monotonic"] = time.monotonic()
                self._cached = row
            return deepcopy(self._cached)

    def entity(self, name: str, *, max_age=2.0):
        """Return one model pose from the complete stamped physics observation."""
        row = self.snapshot(max_age=max_age)
        model = row["models"][name]
        return {
            "name": name,
            "frame": "world",
            "pose": deepcopy(model["pose"]),
            "simulation_time": row["simulation_time"],
            "simulation_stamp": row["simulation_time"],
            "observed_monotonic": row["observed_monotonic"],
            "source": "GETRECOVERYSTATE",
            "instance_id": row["instance_id"],
        }

    def close(self):
        """Release this read-only ROS context without touching the simulation."""
        self._executor.shutdown(timeout_sec=2.0)
        self._node.destroy_node()
        self._context.shutdown()
        self._thread.join(timeout=2.0)


def _links(record, configuration):
    links = record["models"][configuration["model"]]["links"]
    prefix = configuration.get("link_prefix", "")
    result = {name: row for name, row in links.items() if name.startswith(prefix)}
    if not result:
        raise ValueError("Configured resource links are missing")
    return result


def _envelope(links, pose):
    boxes = [row["bounds"] for link in links.values() for row in link["collisions"]]
    if not boxes or any(row is None for row in boxes):
        raise ValueError("Configured resource collision geometry is missing")
    return [
        [
            math.nextafter(min(row[i][0] for row in boxes) - pose[i], -math.inf),
            math.nextafter(max(row[i][1] for row in boxes) - pose[i], math.inf),
        ]
        for i in range(3)
    ]


def _configured_static_part_geometry(record: dict, declaration: dict, part: dict) -> dict:
    """Enclose configured static fixtures relative to their observed carrier.

    An empty carrier link has no collision envelope of its own. Constituent
    models supply geometry only while they and the carrier are observed static
    in the same physics snapshot.
    """
    if set(declaration) != {"models", "requires_static"} or declaration["requires_static"] is not True:
        raise ValueError("Configured constituent geometry requires an explicit static contract")
    names = declaration["models"]
    if (not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or part["name"] in names):
        raise ValueError("Configured constituent model identities must be distinct")
    carrier = record["models"][part["name"]]
    if (carrier["static"] is not True or carrier["pose"] != part["pose"]
            or part.get("instance_id") != record["instance_id"]
            or part.get("simulation_time") != record["simulation_time"]):
        raise ValueError("Constituent geometry requires the same observed static carrier")
    links = {}
    for name in names:
        model = record["models"][name]
        if model["static"] is not True:
            raise ValueError("Configured constituent model is not static: " + name)
        # One empty constituent must not be hidden by another usable envelope.
        _envelope(model["links"], part["pose"])
        links.update({name + "/" + key: value for key, value in model["links"].items()})
    return {"frame": "world", "footprint": _envelope(links, part["pose"])}


class GazeboResourceObservation:
    """Resource-owned idle observation support, selected entirely by configuration."""

    def __init__(self, provider, owner, configuration):
        self.provider, self.owner = provider, owner
        self.configuration = deepcopy(configuration)
        self.motion_configuration = None
        self.support_error = None

    @property
    def reader(self):
        """Expose the registered entity reader without a robot-controller dependency."""
        return self.provider.reader.entity

    def capture(self, *, max_age=2.0, native=None):
        """Capture physical facts; record observed motion without claiming idle coverage."""
        record = self.provider.reader.snapshot(max_age=max_age)
        config, rid = self.configuration, self.owner.agent_name
        links = _links(record, config)
        moving = {
            name: {key: row[key] for key in ("linear_speed", "angular_speed")}
            for name, row in links.items()
            if row["linear_speed"] > config["stationary_linear_speed_max"]
            or row["angular_speed"] > config["stationary_angular_speed_max"]
        }
        state = deepcopy(self.provider.runtime.context.snapshot()[rid])
        model = record["models"][config["model"]]
        reference = config.get("reference_link")
        if native is not None:
            # Fixed URDF links can be lumped out of Gazebo's physics tree. The
            # controller owns the stamped tool transform; body bounds still
            # come from the physics owner, not from the transform endpoint.
            pose = native["current_pose"]
            if reference in links and math.dist(pose[:3], links[reference]["pose"][:3]) > 0.001:
                raise ValueError("Controller and physics-thread tool observations disagree")
        else:
            pose = links[reference]["pose"] if reference else model["pose"]
        attached = [
            row
            for row in record["attachments"]
            if any(
                row[f"model{side}"] == config["model"] and row[f"link{side}"] in links
                for side in (1, 2)
            )
        ]
        # This slice does not silently reinterpret assembly containment as grasp.
        if attached:
            raise ValueError(
                "Idle motion preparation requires observed empty resource custody: " + rid
            )
        if state.get("held_part") is not None:
            raise ValueError("Symbolic custody disagrees with the observed empty attachment set")
        state["current_pose"] = deepcopy(pose)
        state["contained_parts"] = sorted(
            part
            for part, row in self.provider.runtime.context.part_tracker.items()
            if row.get("location") == rid
        )
        if "held_part" in state:
            grip = config["gripper"]
            width = model["joints"][grip["joint"]]["position"]
            matches = [
                name for name in ("open", "closed") if abs(width - grip[name]) <= grip["tolerance"]
            ]
            if len(matches) != 1:
                raise ValueError("Gripper observation is not a supported configured state")
            state["gripper_state"] = matches[0]
            state["grasp_transform"] = None
        if native is not None:
            state["joint_positions"] = deepcopy(native["joint_positions"])
        physical = {
            **deepcopy(native or {}),
            "source": "registered owner / GETRECOVERYSTATE",
            "model_static": model["static"],
            "frame": "world",
            "current_pose": deepcopy(pose),
            "observation_state": state,
            "launch_id": self.provider.launch_id,
            "observed_monotonic": record["observed_monotonic"],
            "simulation_time": record["simulation_time"],
            "custody_complete": True,
            "attachment": {
                "model_name": None,
                "observed_attachment_complete": True,
                "instance_id": record["instance_id"],
                "revision": record["attachment_revision"],
            },
            "stationary_contract": deepcopy(config["stationary_contract"]),
            "idle": not moving,
            "moving_links": moving,
            "geometry_source": {
                "service": "GETRECOVERYSTATE",
                "model": config["model"],
                "links": sorted(links),
                "configuration": fingerprint(config),
            },
            "component_bounds": [
                deepcopy(collision) for link in links.values() for collision in link["collisions"]
            ],
            "footprint": _envelope(links, pose),
        }
        return physical

    def stationary_coverage(self, intervals: list, checkpoint: dict) -> dict:
        """Bind the owner's declared hold contract to explicit intervals in this check."""
        contract = self.configuration.get('stationary_contract')
        kinds = ('idle_commanded_hold', 'static_body_and_idle_containment')
        supported = [{'kind': kind, 'requires_no_running_tasks': True,
                      'requires_no_active_goals': True, 'future_execution_tracking': 'not_established'}
                     for kind in kinds]
        if contract not in supported:
            raise ValueError('Supported owner stationary coverage is unavailable: '+self.owner.agent_name)
        physical = checkpoint['observations'][self.owner.agent_name]['physical']
        if physical.get('idle') is not True:
            raise ValueError('Owner has not observed an idle checkpoint: '+self.owner.agent_name)
        if contract['kind'] == 'static_body_and_idle_containment' and physical.get('model_static') is not True:
            raise ValueError('Static coverage requires observed static equipment: '+self.owner.agent_name)
        return {'contract':deepcopy(contract),'resource_id':self.owner.agent_name,
                'checkpoint_id':checkpoint['checkpoint_id'], 'intervals':deepcopy(intervals),
                'configuration_fingerprint':fingerprint(self.configuration)}


class LiveSafetyPreparation:
    """CCA-owned assembler for an idle supplied candidate; never an admission proof."""

    def __init__(self, runtime, cca, configuration):
        self.runtime, self.cca, self.configuration = runtime, cca, deepcopy(configuration)
        self.reader = None
        self.launch_id = None
        self.observers = {}
        self.last_prepared = None

    def initialize(self):
        """Initialize read-only support before capturing the atomic runtime record."""
        if self.configuration.get("provider") != "gazebo_idle_continuous_v1":
            raise ValueError("Unregistered live preparation provider")
        population = set(build_environment_models(self.runtime.context.inputs["scene"]))
        owners = {owner.agent_name: owner for owner in self.runtime.resource_agents}
        if set(self.configuration["resources"]) != population or set(owners) != population:
            raise ValueError("Live preparation configuration must cover every configured resource")
        if self.reader is None:
            self.reader = GazeboSafetyReader(self.configuration)
        self.launch_id = self.reader.parameter(
            self.configuration["launch_parameter_node"], "launch_id"
        )
        if not self.launch_id or self.launch_id == "recovery_framework":
            raise ValueError("Observed simulation launch identity is unavailable")
        record = self.reader.snapshot(refresh=True)
        for rid, owner in owners.items():
            if rid not in self.observers:
                self.observers[rid] = GazeboResourceObservation(
                    self, owner, self.configuration["resources"][rid]
                )
                owner.recovery_safety_observer = self.observers[rid]
            hook = getattr(owner, "configure_recovery_safety_observer", None)
            if callable(hook):
                try:
                    hook(self.observers[rid], record)
                    self.observers[rid].support_error = None
                except (KeyError, ValueError, RuntimeError, TypeError) as exc:
                    self.observers[rid].motion_configuration = None
                    self.observers[rid].support_error = str(exc)

    def geometry(self, observations, parts, unresolved):
        """Freeze configured regions and observed configured collision envelopes."""
        record = self.reader.snapshot()
        result = {
            "dimension": 3,
            "frame": "world",
            "regions": deepcopy(self.configuration["regions"]),
            "resources": {},
            "parts": {},
        }
        for rid, row in observations.items():
            if "physical" not in row:
                continue
            physical = row["physical"]
            shape = {
                "frame": "world",
                "footprint": physical["footprint"],
                "component_bounds": deepcopy(physical["component_bounds"]),
            }
            if "held_part" not in physical["observation_state"]:
                shape["stationary_only"] = True
            result["resources"][rid] = shape
        for name, row in parts.items():
            model = record["models"][row["name"]]
            try:
                declaration = self.configuration.get("part_geometry", {}).get(name)
                if declaration is None:
                    result["parts"][name] = {
                        "frame": "world",
                        "footprint": _envelope(model["links"], row["pose"]),
                    }
                else:
                    result["parts"][name] = _configured_static_part_geometry(record, declaration, row)
                    row["geometry_source"] = {
                        "service": "GETRECOVERYSTATE",
                        "instance_id": record["instance_id"],
                        "simulation_time": record["simulation_time"],
                        "declaration": deepcopy(declaration),
                        "declaration_fingerprint": fingerprint(declaration),
                        "models_fingerprint": fingerprint({key: record["models"][key] for key in declaration["models"]}),
                    }
            except (ValueError, KeyError, TypeError) as exc:
                unresolved.append({"part": name, "reason": str(exc)})
        return result

    def validate_idle(self, checkpoint):
        """Require an observed idle checkpoint and the known initialization ledger."""
        goals = self.reader.idle_goals()
        if controller_goal_identity(checkpoint.get("controller_goals")) != controller_goal_identity(goals):
            raise ValueError("Controller goals changed or were not captured at the checkpoint")
        runtime = checkpoint["runtime"]
        admission = runtime.get("admission") or {}
        if admission.get("running") or runtime["reservations"]:
            raise ValueError(
                "Prepare and check requires an idle checkpoint without running work or reservations"
            )
        if any(monitor["running_aps"] for monitor in runtime["monitors"]):
            raise ValueError("Active task monitor work is outside idle preparation")
        if any(runtime["physical_sessions"].values()):
            raise ValueError(
                "Existing physical history requires a compatible continuous checkpoint"
            )
        if (
            runtime["acknowledgements"]
            or self.runtime.context.part_tracker != self.runtime.context.initial_product_states
        ):
            raise ValueError(
                "Live ledger replay beyond the known initialization checkpoint is unavailable"
            )
        if runtime["revision"] != 0:
            raise ValueError("The initialized preparation checkpoint has changed")

    def revalidate(self, checkpoint, *, max_age=2.0):
        """Recapture after planning without changing the preparation's original start."""
        record = self.reader.snapshot(refresh=True)
        self.reader.idle_goals()
        observations = {}
        for rid, observer in self.observers.items():
            fresh = observer.owner.capture_recovery_safety_state(max_age=max_age)
            prior = checkpoint["observations"][rid]["physical"]
            for field in ("launch_id", "attachment", "custody_complete", "stationary_contract"):
                if fresh[field] != prior[field]:
                    raise ValueError("Physical identity or custody changed during planning: " + rid)
            # No observation-error envelope is currently provided. A tolerance
            # here would accept an initial state absent from the prepared proof.
            if fresh["current_pose"] != prior["current_pose"]:
                raise ValueError("Resource moved during preparation: " + rid)
            old, new = deepcopy(prior["observation_state"]), deepcopy(fresh["observation_state"])
            old.pop("current_pose", None)
            new.pop("current_pose", None)
            old_joints, new_joints = old.pop("joint_positions", []), new.pop("joint_positions", [])
            if (
                old != new
                or old_joints != new_joints
                or fresh["footprint"] != prior["footprint"]
                or fresh.get("component_bounds") != prior.get("component_bounds")
            ):
                raise ValueError("Resource checkpoint changed during preparation: " + rid)
            observations[rid] = fresh
        parts = {}
        for name, old in checkpoint["parts"].items():
            fresh = self.reader.entity(old["name"], max_age=max_age)
            if fresh["pose"] != old["pose"]:
                raise ValueError("Part moved during preparation: " + name)
            declaration = self.configuration.get("part_geometry", {}).get(name)
            if declaration is not None:
                current = _configured_static_part_geometry(record, declaration, fresh)
                source = old.get("geometry_source", {})
                if (current != checkpoint["geometry"]["parts"][name]
                        or source.get("declaration_fingerprint") != fingerprint(declaration)
                        or source.get("models_fingerprint") != fingerprint({
                            key: record["models"][key] for key in declaration["models"]})):
                    raise ValueError("Configured constituent geometry changed during preparation: " + name)
            parts[name] = fresh
        result = {
            "observations": observations,
            "parts": parts,
            "original_checkpoint_id": checkpoint["checkpoint_id"],
        }
        result["checkpoint_id"] = fingerprint(result)
        return result

    def __call__(self, *, checkpoint, prepared, request):
        """Assemble one finite, fully bound schedule from exact prepared programs."""
        self.validate_idle(checkpoint)
        if checkpoint["unresolved"] or any(row["status"] != "prepared" for row in prepared):
            raise ValueError("Live checkpoint or primitive preparation is incomplete")
        configured = self.configuration["candidates"].get(request["recovery_id"])
        if configured is None:
            raise ValueError("Supplied candidate has no registered finite start schedule")
        schedules = configured["start_offsets"]
        if set(schedules) != {row["resource_id"] for row in prepared}:
            raise ValueError("Candidate starts do not cover its supplied programs")
        programs, events, ends = [], [], []
        for program in prepared:
            rid = program["resource_id"]
            cursor = Fraction(str(schedules[rid]))
            if cursor < 0:
                raise ValueError("Candidate start offsets must be nonnegative")
            results = []
            previous = None
            for index, step in enumerate(program["steps"]):
                command = program["program"]["primitive_steps"][index]
                if step.get("observation_status") != "prepared":
                    raise ValueError("Continuous prepared motion is unavailable: " + rid)
                end = cursor + Fraction(step["joint_trajectory"]["duration_ns"], 1_000_000_000)
                source = command["source"]
                if previous != source["outline_id"]:
                    events.append(
                        {
                            "outline_id": source["outline_id"],
                            "des_event_id": source["des_event_id"],
                            "event_name": source["event_name"],
                            "resource_id": rid,
                            "predecessors": [] if previous is None else [previous],
                            "primitive_step_indices": [],
                        }
                    )
                    previous = source["outline_id"]
                events[-1]["primitive_step_indices"].append(index)
                evidence = {
                    "frame": "world",
                    "preparation_id": step["preparation_id"],
                    "joint_trajectory": deepcopy(step["joint_trajectory"]),
                    "continuous_motion": deepcopy(step["continuous_motion"]),
                }
                results.append(
                    {
                        "primitive": command["primitive"],
                        "resolved_params": deepcopy(command["params"]),
                        "start_time": float(cursor),
                        "end_time": float(end),
                        "success": True,
                        "source": deepcopy(source),
                        "model_evidence": evidence,
                    }
                )
                cursor = end
            ends.append(cursor)
            programs.append(
                {
                    "resource_id": rid,
                    "primitive_steps": deepcopy(program["program"]["primitive_steps"]),
                    "step_results": results,
                }
            )
        horizon = [0.0, float(max(ends))]
        stationary = {rid: [deepcopy(horizon)] for rid in checkpoint["observations"]}
        for program in programs:
            a, b = program["step_results"][0]["start_time"], program["step_results"][-1]["end_time"]
            stationary[program["resource_id"]] = ([[0.0, a]] if a else []) + (
                [[b, horizon[1]]] if b < horizon[1] else []
            )
        coverage = {rid:self.observers[rid].stationary_coverage(intervals,checkpoint)
                    for rid,intervals in stationary.items()}
        resources = {
            rid: deepcopy(row["physical"]["observation_state"])
            for rid, row in checkpoint["observations"].items()
        }
        parts = {}
        ledger_evidence = {
            "source_kind": "validated_initialization",
            "checkpoint": checkpoint["checkpoint_id"],
            "complete": True,
            "run_id": checkpoint["runtime"]["run_id"],
            "initial_product_states_fingerprint": fingerprint(
                self.runtime.context.initial_product_states
            ),
        }
        for name, ledger in checkpoint["runtime"]["part_tracker"].items():
            parts[name] = {
                **deepcopy(ledger),
                "current_pose": deepcopy(checkpoint["parts"][name]["pose"]),
                "contained_by": ledger.get("location")
                if ledger.get("location") in resources
                else None,
                "stationary_until": horizon[1],
                "processCompleted_complete": True,
                "processCompleted_evidence": deepcopy(ledger_evidence),
            }
        snapshot = {"resources": resources, "parts": parts}
        # These records preserve outstanding work; the candidate does not complete it.
        snapshot["nominal_tasks"] = deepcopy(checkpoint["runtime"]["pending_tasks"])
        inputs = {
            "grounding_inputs": {
                "scene": deepcopy(checkpoint["scene"]),
                "programs": programs,
                "snapshot": snapshot,
                "geometry": deepcopy(checkpoint["geometry"]),
                "horizon": horizon,
                "stationary": stationary,
            },
            "recovery_events": events,
            "running_work": [],
            "event_start_choices": [
                {
                    "id": request["recovery_id"],
                    "starts": {
                        event["outline_id"]: next(
                            step["start_time"]
                            for program in programs
                            for step in program["step_results"]
                            if step["source"]["outline_id"] == event["outline_id"]
                        )
                        for event in events
                    },
                    "programs": deepcopy(programs),
                    "stationary": deepcopy(stationary),
                }
            ],
            "completion": {
                "resources": {
                    rid: {"held_part": None} for rid in resources if "held_part" in resources[rid]
                },
                "parts": {
                    name: {
                        "processCompleted": deepcopy(row["processCompleted"]),
                        "contained_by": row["contained_by"],
                    }
                    for name, row in parts.items()
                },
            },
        }
        self.last_prepared = fingerprint(prepared)
        return {
            "synthetic": False,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "prepared_fingerprint": fingerprint(prepared),
            "composition_inputs": inputs,
            "physical_rule_activation": "prospective",
            "ledger_evidence": ledger_evidence,
            "stationary_contracts": coverage,
        }


def install_live_preparation(runtime, cca):
    """Register only the trusted configured provider, without activating a proof."""
    configuration = runtime.context.inputs["scene"].get("safety_preparation")
    if configuration is None:
        return
    previous = getattr(cca, "recovery_safety_preparation_provider", None)
    if previous is None:
        cca.recovery_safety_preparation_provider = LiveSafetyPreparation(
            runtime, cca, configuration
        )
    elif isinstance(previous, LiveSafetyPreparation) and previous.runtime is not runtime:
        if previous.reader is not None:
            previous.reader.close()
        cca.recovery_safety_preparation_provider = LiveSafetyPreparation(
            runtime, cca, configuration
        )


def build_supplied_candidate(bridge, candidate_id: str) -> dict:
    """Resolve a configured mock motion recipe against actual observed retreat poses."""
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import capture_checkpoint

    runtimes = {
        id(owner.environment_runtime): owner.environment_runtime
        for owner in bridge.resource_agents
        if getattr(owner, "environment_runtime", None) is not None
    }
    if len(runtimes) != 1 or bridge.cca is None:
        raise ValueError("One registered runtime and CCA are required")
    runtime = next(iter(runtimes.values()))
    install_live_preparation(runtime, bridge.cca)
    provider = bridge.cca.recovery_safety_preparation_provider
    if not isinstance(provider, LiveSafetyPreparation):
        raise ValueError("Configured live preparation is unavailable")
    provider.initialize()
    checkpoint = capture_checkpoint(runtime, bridge.cca)
    recipe = provider.configuration["candidates"][candidate_id]
    programs = []
    for rid, targets in recipe["poses"].items():
        if "physical" not in checkpoint["observations"][rid]:
            raise ValueError("Candidate targets need an observed resource pose: " + rid)
        start = checkpoint["observations"][rid]["physical"]["current_pose"]
        identity = candidate_id + "/" + rid
        steps = []
        for index, target in enumerate((targets["approach"], targets["entry"], start[:3])):
            steps.append(
                {
                    "primitive": "move_cartesian",
                    "params": dict(zip(("x", "y", "z"), target, strict=True)),
                    "source": {
                        "outline_id": identity,
                        "des_event_id": identity,
                        "event_name": identity,
                        "step_index": index,
                    },
                }
            )
        programs.append({"resource_id": rid, "primitive_steps": steps})
    return {"recovery_id": candidate_id, "programs": programs}

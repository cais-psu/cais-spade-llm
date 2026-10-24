from __future__ import annotations

"""Physical admission of nominal starts and their native sends against predefined specifications.

A start is checked on a conservative prediction of its whole saved program: the regions its
resource and carried part may occupy and the completions its finish applies. Running work counts
as possibly anywhere in its claimed regions, and a pending completion as possibly done. The native
program then runs unchanged; each arm trajectory it sends is swept exactly and checked again before
dispatch. A send outside the claim extends it when the specifications allow, otherwise the robot
waits stopped at that primitive boundary until a timeout faults the task. A claim is released at
the first physics snapshot taken after its execution ended, which gives the resource's resting place.
"""

import logging
import math
import time
from copy import deepcopy
from fractions import Fraction
from itertools import product

logger = logging.getLogger(__name__)

SUPPORTED_KINDS = frozenset({
    "resource_region", "part_region_entry", "process_target_completed", "process_result_completed",
})
_UNKNOWN = (False, True)


def _overlap(box: list, region: list) -> bool:
    return all(box[i][0] <= region[i][1] and box[i][1] >= region[i][0] for i in range(3))


def _hull(boxes: list) -> list:
    return [[min(box[i][0] for box in boxes), max(box[i][1] for box in boxes)] for i in range(3)]


def _possible(boxes: list, region: list) -> tuple:
    """A predicted envelope only shows where a resource may be, never that it is there."""
    return _UNKNOWN if any(_overlap(box, region) for box in boxes) else (False,)


def _completion_key(part: str, process: str, field: str | None, value: str | None) -> tuple:
    return (part, process, field, value)


def _record_key(part: str, record: dict) -> tuple:
    field = "target" if "target" in record else "result" if "result" in record else None
    return _completion_key(part, record["process"], field, record.get(field) if field else None)


def _read_result(result) -> dict:
    """Normalize a read primitive's result as the robot agent does for its native program."""
    if isinstance(result, bool):
        return {"success": result}
    if isinstance(result, list):
        return {"success": True, "data": result}
    return dict(result) if isinstance(result, dict) else {"success": False}


def _entries(cells: list) -> list:
    """Derive part-entry possibilities from consecutive part occupancy, as continuous observations do."""
    preceding = None
    for phase, cell in cells:
        cell["part_entry"] = {}
        for region, possible in cell["part_occupancy"].items():
            if preceding is None:
                entry = (False,)
            elif phase == "between":
                entry = ((False,) if possible == (False,) or possible == (True,) and preceding[region] == (True,)
                         else _UNKNOWN)
            else:
                entry = tuple(sorted({now and not before for now in possible for before in preceding[region]}))
            cell["part_entry"][region] = entry
        if phase == "at":
            preceding = cell["part_occupancy"]
    return cells


def _configured_route(step_id: str, settings: dict) -> list[list[float]]:
    """Waypoints the native Cartesian controller inserts for this saved step, as XYZ points."""
    if step_id == "move_above_part":
        route = settings.get("pick_transit_waypoints", [])
    elif step_id == "move_above_destination" and settings.get("transit_waypoints"):
        route = settings["transit_waypoints"][0]
    elif step_id == "retreat_from_source" and settings.get("rotation_waypoint"):
        route = [settings["rotation_waypoint"]]
    else:
        route = []
    return [list(point[:3]) for point in route]


def _location_dock(scene: dict, location: str) -> list:
    """A location's recorded KMR dock as planar x, y, yaw."""
    from cais_spade_llm.recovery_framework.kmr_recovery import recorded_location

    source, _ = recorded_location(scene, location)
    dock = source["KMR_docking_pose"]
    return [dock[0], dock[1], dock[-1]]


def _transfer_arc(center: list, start: list, end: list, radius: float) -> list:
    """Gripper positions on the configured transfer arc around the arm mount between two positions."""
    left = math.atan2(start[1] - center[1], start[0] - center[0])
    right = math.atan2(end[1] - center[1], end[0] - center[0])
    angle = math.atan2(math.sin(right - left), math.cos(right - left))
    count = max(1, math.ceil(abs(angle) / 0.1))
    height = max(start[2], end[2])
    return [[center[0] + radius * math.cos(left + angle * index / count),
             center[1] + radius * math.sin(left + angle * index / count), height] for index in range(count + 1)]


def arm_chain(configuration: dict, matrices: dict) -> dict:
    """Arm links from the first commanded joint to the reference link, with chain lengths and sizes.

    Args:
        configuration: Configured URDF kinematics with commanded joint names and link components.
        matrices: Link transforms at one configuration, used to size links and their side branches.

    Returns:
        The chain links; for each link frame, the summed joint offsets from the first link frame and
        to the reference link; each link's thickness around the segment from its frame to the next
        (side branches such as fingers included); and the chain link that owns every arm link.
    """
    from itertools import product as corners

    from cais_spade_llm.resources.continuous_motion import transformed_box

    rows = {row["child"]: row for row in configuration["joints"]}
    first = next(row["child"] for row in configuration["joints"] if row["name"] == configuration["joint_names"][0])
    links, link = [], configuration["reference_link"]
    while link != first:
        links.append(link)
        link = rows[link]["parent"]
    links = [first, *reversed(links)]
    lengths = [0.0]
    for link in links[1:]:
        lengths.append(lengths[-1] + math.hypot(*rows[link]["xyz"]))
    children = {}
    for row in configuration["joints"]:
        children.setdefault(row["parent"], []).append(row["child"])
    owners = {}
    for index, link in enumerate(links):
        stack = [link]
        while stack:
            name = stack.pop()
            owners[name] = index
            stack.extend(child for child in children.get(name, []) if child not in links)

    def origin(name):
        return [(row[3][0] + row[3][1]) / 2 for row in matrices[name][:3]]

    def to_segment(point, start, end):
        span = [b - a for a, b in zip(start, end, strict=True)]
        length = sum(value * value for value in span)
        fraction = 0.0 if length == 0 else max(0.0, min(1.0, sum(
            (p - a) * s for p, a, s in zip(point, start, span, strict=True)) / length))
        return math.dist(point, [a + fraction * s for a, s in zip(start, span, strict=True)])

    thickness = [0.0] * len(links)
    for component in configuration["components"]:
        index = owners.get(component["link"])
        if index is None:
            continue
        box = transformed_box(matrices[component["link"]], component["bounds"], component["id"])
        start, end = origin(links[index]), origin(links[min(index + 1, len(links) - 1)])
        thickness[index] = max(thickness[index], max(to_segment(point, start, end) for point in corners(
            *[(box["minimum"][i][0], box["maximum"][i][1]) for i in range(3)])))
    return {"links": links, "from_first": lengths,
            "to_reference": [lengths[-1] - value for value in lengths], "radius": thickness, "owners": owners}


def arm_link_bounds(chain: dict, first: list, points: list, margin: float) -> list:
    """Bound every arm link from its frames' chain distances to the first arm frame and the gripper.

    Each link frame lies within its chain length of the first arm frame and of the reference link;
    a link lies within its thickness of the segment between its frame and the next. So any joint
    solution that puts the reference link at one of ``points``, or on the straight paths between
    them, keeps each link within these boxes.
    """
    low = [min(point[i] for point in points) for i in range(3)]
    high = [max(point[i] for point in points) for i in range(3)]
    frames = []
    for near, far in zip(chain["from_first"], chain["to_reference"], strict=True):
        box = [[max(first[i] - near, low[i] - far), min(first[i] + near, high[i] + far)] for i in range(3)]
        if any(axis[0] > axis[1] for axis in box):
            # Unreachable as stated: fall back to the frame's whole reach around the first frame.
            box = [[first[i] - near, first[i] + near] for i in range(3)]
        frames.append(box)
    boxes = []
    for index, size in enumerate(chain["radius"]):
        ends = _hull([frames[index], frames[min(index + 1, len(frames) - 1)]])
        boxes.append([[axis[0] - size - margin, axis[1] + size + margin] for axis in ends])
    return boxes

_ARM_OPERATIONS = frozenset({"move_cartesian", "move_relative", "move_to_named_pose", "grasp_part", "release_part",
                             "open_gripper", "close_gripper", "compute_pick_targets", "compute_place_targets"})


class _KMRGeometry:
    """The KMR's configured kinematics and observed body, placed at any base pose along its route."""

    def __init__(self, owner, record: dict, configuration: dict, settings: dict, margin: float) -> None:
        from cais_spade_llm.recovery_framework.gazebo_safety_preparation import _links
        from cais_spade_llm.resources.continuous_geometry import observed_joint_configuration
        from cais_spade_llm.resources.continuous_motion import interval, kinematic_transforms

        self.model = record["models"][configuration["model"]]
        self.observed = owner.recovery_safety_observer.motion_configuration
        self.root = self.model["links"][self.observed["root"]]["pose"]
        self.configuration = observed_joint_configuration(self.observed, self.model["joints"], self.root)
        names = list(self.configuration["joint_names"])
        self.current = {name: interval(self.model["joints"][name]["position"]) for name in names}
        self.chain = arm_chain(self.configuration, kinematic_transforms(self.configuration, self.current))
        self.base_boxes = [row["bounds"] for name, link in _links(record, configuration).items()
                           if name not in self.chain["owners"] for row in link["collisions"]]
        root = self.root
        self.here = [root[0], root[1], math.atan2(2 * (root[6] * root[5] + root[3] * root[4]),
                                                  1 - 2 * (root[4] ** 2 + root[5] ** 2))]
        waypoints = settings["task_execution"]["cartesian_waypoints"]
        self.transport_offset = waypoints["transport_tcp_position_in_base_frame_m"]
        self.turn = waypoints["turn_radius_m"]
        self.mount = settings["arm_mount_xyz"]
        self.parked = [self.current, *({name: interval(value) for name, value in zip(names, joints, strict=True)}
                                       for joints in (settings["parked_arm_configuration"],
                                                      settings["task_execution"]["carrying_arm_configuration"]))]
        self.margin = margin

    def at(self, dock: list, offset: list) -> list:
        # The KMR worker composes base-frame offsets with the KMR model pose.
        cosine, sine = math.cos(dock[2]), math.sin(dock[2])
        return [dock[0] + cosine * offset[0] - sine * offset[1], dock[1] + sine * offset[0] + cosine * offset[1],
                self.model["pose"][2] + offset[2]]

    def transport(self, dock: list) -> list:
        return self.at(dock, self.transport_offset)

    def body(self, dock: list) -> list:
        dx, dy = dock[0] - self.here[0], dock[1] - self.here[1]
        return [[[box[0][0] + dx, box[0][1] + dx], [box[1][0] + dy, box[1][1] + dy], list(box[2])]
                for box in self.base_boxes]

    def _placed(self, dock: list) -> dict:
        from cais_spade_llm.resources.continuous_geometry import observed_joint_configuration

        pose = [dock[0], dock[1], self.root[2], 0.0, 0.0, math.sin(dock[2] / 2), math.cos(dock[2] / 2)]
        return observed_joint_configuration(self.observed, self.model["joints"], pose)

    def tcp(self) -> list:
        from cais_spade_llm.resources.continuous_motion import kinematic_transforms

        matrix = kinematic_transforms(self.configuration, self.current)[self.chain["links"][-1]]
        return [(row[3][0] + row[3][1]) / 2 for row in matrix[:3]]

    def arm(self, dock: list, points: list) -> list:
        from cais_spade_llm.resources.continuous_motion import kinematic_transforms

        first = kinematic_transforms(self._placed(dock), self.current)[self.chain["links"][0]]
        return arm_link_bounds(self.chain, [(row[3][0] + row[3][1]) / 2 for row in first[:3]], points, self.margin)

    def parked_arm(self, dock: list) -> list:
        """Exact arm geometry of the current, parked and carrying configurations at a base pose."""
        from cais_spade_llm.resources.continuous_motion import kinematic_transforms, transformed_box

        boxes = []
        for joints in self.parked:
            matrices = kinematic_transforms(self._placed(dock), joints)
            for component in self.configuration["components"]:
                if component["link"] in self.chain["owners"]:
                    box = transformed_box(matrices[component["link"]], component["bounds"], component["id"])
                    boxes.append([[box["minimum"][i][0] - self.margin, box["maximum"][i][1] + self.margin]
                                  for i in range(3)])
        return boxes

    def gripper_points(self, dock: list, work: list) -> list:
        """The gripper's positions at a dock: transport, work and their transfer arcs, with lifts above."""
        center, points = self.at(dock, self.mount), [self.transport(dock), *work]
        points += [[point[0], point[1], point[2] + 0.3] for point in points]
        arcs = [arc for start in points[:1] + work for end in points
                for arc in _transfer_arc(center, start, end, self.turn)]
        return points + arcs


def grounded_rules(cca, runtime, geometry: dict) -> list[dict]:
    """Ground the CCA's predefined specifications over the complete configured scene.

    Args:
        cca: Central controller holding the loaded predefined document and compiled DFAs.
        runtime: Environment runtime whose scene defines the resource population.
        geometry: Live safety geometry with resources, regions and parts.

    Returns:
        Grounded rule instances, each with its template DFA and resolved AP bindings.

    Raises:
        ValueError: A specification uses an AP kind this check cannot evaluate.
    """
    from cais_spade_llm.agents.central_controller.offline_safety_grounding import _catalog, _rules
    from cais_spade_llm.agents.central_controller.ppr_ap import physical_ap_kind
    from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import (
        _ap_binding,
    )
    from cais_spade_llm.resources.environment_models import build_environment_models

    document = cca.predefined_safety
    rules, _ = _rules(_catalog(document["catalog"], structured_groundings=True),
                      document["requirement_scopes"],
                      build_environment_models(runtime.context.inputs["scene"]), geometry)
    for rule in rules:
        bindings = {}
        for ap in rule["aps"]:
            kind = physical_ap_kind(ap)
            if kind not in SUPPORTED_KINDS:
                raise ValueError(f"Nominal physical check cannot evaluate {kind or 'structured'} APs: "
                                 + rule["rule_id"])
            bindings[ap["label"]] = (kind, _ap_binding(rule, ap))
        rule["dfa_dot"] = cca.safety_logic.rule_dfas[rule["id"]]
        rule["ap_kinds"] = bindings
    return rules


class NominalPhysicalGate:
    """Own nominal physical claims and the physical monitor states of one run."""

    def __init__(self, runtime, cca, preparation) -> None:
        """Ground the specifications on an idle scene and observe every resource at rest.

        Args:
            runtime: The run's environment runtime; its admission lock guards every decision.
            cca: Central controller with the predefined specifications.
            preparation: Live safety preparation providing the physics reader and resource configuration.
        """
        from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
        from cais_spade_llm.recovery_framework.gazebo_safety_preparation import capture_checkpoint

        self.runtime, self.cca, self.preparation = runtime, cca, preparation
        self.lock = runtime.context.admission_lock
        settings = preparation.configuration
        self.resources = deepcopy(settings["resources"])
        self.margin = float(settings.get("nominal_prediction_margin_m", 0.05))
        self.wait_timeout_s = float(settings.get("nominal_send_wait_timeout_sec", 300.0))
        self.poll_s = 0.5
        preparation.initialize(model_execution=True)
        checkpoint = capture_checkpoint(runtime, cca, model_execution=True)
        if checkpoint["geometry"] is None:
            raise ValueError("Nominal physical check needs the live safety geometry: "
                             + str(checkpoint["unresolved"]))
        geometry = deepcopy(checkpoint["geometry"])
        targets = runtime.context.inputs["geometry"].get("parts", {}).get("assembly_target_map", {})
        for part in runtime.context.part_tracker:
            # Grounding needs each scoped part's assembly target, not its pose; printed parts appear later.
            row = geometry["parts"].setdefault(part, {})
            if part in targets:
                row.setdefault("target", targets[part])
        self.rules = grounded_rules(cca, runtime, geometry)
        used = {binding["region"] for rule in self.rules for _, binding in rule["ap_kinds"].values()
                if "region" in binding}
        self.regions = {name: deepcopy(row["bounds"]) for name, row in settings["regions"].items() if name in used}
        self.checker = BaseSafetyChecker({rule["rule_id"]: rule["dfa_dot"] for rule in self.rules}, self.rules)
        self.states = {rule["rule_id"]: {self.checker.dfas[rule["rule_id"]]["initial"]} for rule in self.rules}
        self.states_observed = 0.0
        geometry = runtime.context.inputs["geometry"].get("parts", {})
        self.part_models = {name: geometry.get("model_map", {}).get(name) or geometry.get(name, {}).get("model_name")
                            for name in runtime.context.part_tracker}
        self.work: dict[str, dict] = {}
        self.history: list[dict] = []
        self._observe()

    # ------------------------------------------------------------------ observation

    def _observe(self) -> dict:
        """Read one physics snapshot: each resource's and part's region occupancy and who is stopped."""
        from cais_spade_llm.recovery_framework.gazebo_safety_preparation import _links

        record = self.preparation.reader.snapshot(max_age=0.25)
        occupancy = {}
        for rid, configuration in self.resources.items():
            boxes = [row["bounds"] for link in _links(record, configuration).values() for row in link["collisions"]]
            if not boxes or any(box is None for box in boxes):
                raise ValueError("Resource collision geometry is missing: " + rid)
            occupancy[rid] = {name: any(_overlap(box, bounds) for box in boxes)
                              for name, bounds in self.regions.items()}
        parts = {}
        for name, model in self.part_models.items():
            row = record["models"].get(model)
            if row is None:
                continue
            boxes = [collision["bounds"] for link in row["links"].values() for collision in link["collisions"]]
            parts[name] = {region: any(_overlap(box, bounds) for box in boxes)
                           for region, bounds in self.regions.items()}
        return {"record": record, "occupancy": occupancy, "parts": parts}

    def _refresh(self, observation: dict) -> None:
        """Release claims of retired work once a snapshot taken after its end shows where it rests."""
        pending = self.runtime.context.pending_tasks
        for task_id, entry in list(self.work.items()):
            if entry["status"] == "retired" and observation["record"]["observed_monotonic"] > entry["retired_at"]:
                entry.update(regions=set(), part_regions={}, status="released")
            if entry["status"] == "released" and task_id not in pending:
                del self.work[task_id]

    def _base(self, observation: dict, own: str | None) -> dict:
        """Valuations of everything except the checked resource's own motion."""
        claims, part_entries, pending = {}, {}, {}
        tasks = self.runtime.context.pending_tasks
        for task_id, entry in self.work.items():
            if entry["resource_id"] != own and entry["status"] != "released":
                claims.setdefault(entry["resource_id"], set()).update(entry["regions"])
                for part, regions in entry["part_regions"].items():
                    part_entries.setdefault(part, {}).update({region: _UNKNOWN for region in regions})
            if task_id in tasks:
                pending.update({key: _UNKNOWN for key in entry["completions"]})
        occupancy = {}
        for rid in self.resources:
            if rid in claims:
                occupancy[rid] = {name: _UNKNOWN if name in claims[rid] else (False,) for name in self.regions}
            else:
                occupancy[rid] = {name: (observation["occupancy"][rid][name],) for name in self.regions}
        for task in tasks.values():
            pending.update({key: _UNKNOWN for key in self._observed_completions(task)})
        return {"occupancy": occupancy, "part_entry": part_entries, "completions": {**pending, **self._done()}}

    def _done(self) -> dict:
        return {_record_key(part, record): (True,) for part, row in self.runtime.context.part_tracker.items()
                for record in row.get("processCompleted", [])}

    def _observed_completions(self, task: dict) -> set:
        """Completions a pending uncontrollable event, such as a running job's end, may record."""
        from cais_spade_llm.resources.environment_models import project_transition

        context = self.runtime.context
        event = next((row for row in context.models[task["resource_id"]]["events"]
                      if row["event_id"] == task["event_id"]), {})
        if event.get("controllable") is not False or event.get("start_event_id") is not None:
            return set()
        try:
            _, products = project_transition(context.models, context.snapshot(), context.part_tracker, task,
                                             context.product_name, context.requirements)
        except (ValueError, KeyError, TypeError):
            return set()
        return self._new_completions(products)

    def _new_completions(self, products: dict) -> set:
        tracker = self.runtime.context.part_tracker
        return {_record_key(part, record) for part, row in products.items()
                for record in row.get("processCompleted", [])
                if record not in tracker.get(part, {}).get("processCompleted", [])}

    # ------------------------------------------------------------------ monitor

    def _observe_states(self, observation: dict) -> str:
        """Advance every monitor by one observed scene and drop the states it rules out.

        Each change was checked before it happened, so the true state survives; a send commits every
        state along its trajectory, and later observations narrow them again. The specification
        monitors use no next operator, so an extra observation of an unchanged scene is a stutter.

        Returns:
            The first rule left without a state, meaning the observed history violates it, or "".
        """
        if observation["record"]["observed_monotonic"] < self.states_observed:
            return ""
        base = {"occupancy": {rid: {name: (value,) for name, value in row.items()}
                              for rid, row in observation["occupancy"].items()},
                "part_entry": {}, "completions": self._done()}
        cell = {"resource": None, "occupancy": {}, "part": None, "part_entry": {}, "completions": frozenset()}
        observed = {}
        for rule in self.rules:
            values = self._values(rule, cell, base)
            alphabet = frozenset(label for label, value in values.items() if value[0])
            reached = set()
            for state in self.states[rule["rule_id"]]:
                transition = self.checker.transition_evidence(rule["rule_id"], state, alphabet)
                if transition["status"] == "passed":
                    reached.add(transition["to"])
            if not reached:
                return rule["rule_id"]
            observed[rule["rule_id"]] = reached
        self.states = observed
        self.states_observed = observation["record"]["observed_monotonic"]
        return ""

    def _values(self, rule: dict, cell: dict, base: dict) -> dict:
        values = {}
        for label, (kind, binding) in rule["ap_kinds"].items():
            if kind == "resource_region":
                rid, region = binding["resource"], binding["region"]
                value = cell["occupancy"][region] if rid == cell["resource"] else base["occupancy"][rid][region]
            elif kind == "part_region_entry":
                part, region = binding["part"], binding["region"]
                value = (cell["part_entry"][region] if part == cell["part"]
                         else base["part_entry"].get(part, {}).get(region, (False,)))
            else:
                field = "target" if kind == "process_target_completed" else "result"
                key = _completion_key(binding["part"], binding["process"], field, binding[field])
                value = (True,) if key in cell["completions"] else base["completions"].get(key, (False,))
            values[label] = value
        return values

    def _step(self, cells: list, base: dict) -> dict:
        """Step every monitor over the cells; return the reached and all prefix states or a violation."""
        states = {key: set(value) for key, value in self.states.items()}
        prefixes = {key: set(value) for key, value in states.items()}
        for rule in self.rules:
            rule_id = rule["rule_id"]
            sequence = []
            for phase, cell in cells:
                values = self._values(rule, cell, base)
                if sequence and sequence[-1][1] == values:
                    # Repeating one valuation is covered by the closure of any nonempty word over it.
                    sequence[-1] = ("between", values)
                else:
                    sequence.append((phase, values))
            current = states[rule_id]
            for phase, values in sequence:
                labels = list(values)
                alphabets = {frozenset(label for label, bit in zip(labels, bits, strict=True) if bit)
                             for bits in product(*values.values())}
                pending, reached = list(current), set()
                while pending:
                    state = pending.pop()
                    for alphabet in alphabets:
                        transition = self.checker.transition_evidence(rule_id, state, alphabet)
                        if transition["status"] != "passed":
                            return {"status": "held", "reason": "possible_physical_requirement_violation",
                                    "rule_id": rule_id, "transition": transition}
                        if transition["to"] not in reached:
                            reached.add(transition["to"])
                            if phase == "between":
                                pending.append(transition["to"])
                current = reached
                prefixes[rule_id] |= reached
            states[rule_id] = current
        return {"status": "allowed", "states": states, "prefixes": prefixes}

    # ------------------------------------------------------------------ start prediction

    def _cell(self, phase: str, rid: str, occupancy: dict, part: str | None, part_occupancy: dict,
              completions=frozenset()) -> tuple[str, dict]:
        return phase, {"resource": rid, "occupancy": occupancy, "part": part,
                       "part_occupancy": part_occupancy, "completions": completions}

    def _completions(self, task: dict) -> frozenset:
        """Completions of the task's own finish and of the uncontrollable events its start makes
        inevitable, such as the end of a started machine job."""
        from cais_spade_llm.resources.environment_models import project_activity, project_transition

        context = self.runtime.context
        values, products = project_activity(context.models, context.snapshot(), context.part_tracker, task,
                                            context.product_name, context.requirements,
                                            started=task["task_id"] in getattr(context, "started_tasks", {}))
        keys = self._new_completions(products)
        for event in context.models[task["resource_id"]]["events"]:
            if event.get("controllable") is not False or event.get("start_event_id") is not None:
                continue
            follow = {**task, "event_id": event["event_id"], "event_name": event["event_name"]}
            try:
                _, after = project_transition(context.models, values, products, follow,
                                              context.product_name, context.requirements)
            except (ValueError, KeyError, TypeError):
                continue
            keys |= self._new_completions(after)
        return frozenset(keys)

    def _held_part(self, rid: str) -> str | None:
        held = [part for part, row in self.runtime.context.part_tracker.items() if row.get("location") == rid]
        return held[0] if len(held) == 1 else None

    def _task_part(self, task: dict) -> str | None:
        part = task["parameters"].get("part_name")
        return part if part in self.part_models else None

    def _part_boxes(self, record: dict, part: str) -> list:
        row = record["models"][self.part_models[part]]
        return [collision["bounds"] for link in row["links"].values() for collision in link["collisions"]]

    def predict(self, task: dict) -> dict:
        """Predict a start's whole saved program from a fresh snapshot, outside the admission lock.

        Args:
            task: Exact pending nominal task.

        Returns:
            Its cells, possible regions and completions, ready for ``decide``.
        """
        started = time.monotonic()
        observation = self._observe()
        rid = task["resource_id"]
        owner = next(row for row in self.runtime.resource_agents if row.agent_name == rid)
        completions = self._completions(task)
        if getattr(getattr(owner, "_controller", None), "recovery_safety_observer", None) is not None:
            cells = self._robot_cells(owner, task, observation)
        elif rid == "KMR":
            cells = self._kmr_cells(owner, task, observation)
        else:
            cells = self._static_cells(rid, observation)
        cells.append(self._cell("at", rid, cells[-1][1]["occupancy"], cells[-1][1]["part"],
                                cells[-1][1]["part_occupancy"], completions))
        _entries(cells)
        regions = {name for _, cell in cells for name, value in cell["occupancy"].items() if True in value}
        part_regions = {}
        for _, cell in cells:
            if cell["part"] is not None:
                part_regions.setdefault(cell["part"], set()).update(
                    name for name, value in cell["part_occupancy"].items() if True in value)
        return {"task": deepcopy(task), "observation": observation, "cells": cells, "regions": regions,
                "part_regions": part_regions, "completions": completions,
                "prediction_time_s": time.monotonic() - started}

    def _robot_cells(self, owner, task: dict, observation: dict) -> list:
        """Envelope the program's tool and carried-part path through its targets and waypoints."""
        from cais_spade_llm.resources.continuous_geometry import observed_joint_configuration
        from cais_spade_llm.resources.continuous_motion import (
            interval,
            kinematic_transforms,
            representative_pose,
            transformed_box,
        )

        rid, controller = owner.agent_name, owner._controller
        record = observation["record"]
        observer = controller.recovery_safety_observer
        model = record["models"][self.resources[rid]["model"]]
        configuration = observed_joint_configuration(
            observer.motion_configuration, model["joints"],
            model["links"][observer.motion_configuration["root"]]["pose"])
        names = list(controller.arm_joint_names)

        def transforms(positions):
            return kinematic_transforms(configuration, {name: interval(value)
                                                       for name, value in zip(names, positions, strict=True)})

        joints = [model["joints"][name]["position"] for name in names]
        matrices = transforms(joints)
        reference = configuration["reference_link"]
        tool = representative_pose(matrices[reference])[:3]
        children = {}
        for row in configuration["joints"]:
            children.setdefault(row["parent"], []).append(row["child"])
        attached, stack = set(), [reference]
        while stack:
            link = stack.pop()
            attached.add(link)
            stack.extend(children.get(link, []))
        corners = [box for component in configuration["components"] if component["link"] in attached
                   for box in [transformed_box(matrices[component["link"]], component["bounds"], component["id"])]]
        if not corners:
            raise ValueError("Tool geometry below the controlled link is unavailable: " + rid)
        # The gripper only turns about the vertical axis on these paths, so a radius bounds it in XY.
        radius = max(math.hypot(x - tool[0], y - tool[1]) for box in corners
                     for x in (box["minimum"][0][0], box["maximum"][0][1])
                     for y in (box["minimum"][1][0], box["maximum"][1][1]))
        low = min(box["minimum"][2][0] for box in corners) - tool[2]
        high = max(box["maximum"][2][1] for box in corners) - tool[2]
        held = self._held_part(rid)
        part_offset, part_half = None, 0.0
        if held is not None:
            boxes = self._part_boxes(record, held)
            hull = _hull(boxes)
            center = [(axis[0] + axis[1]) / 2 for axis in hull]
            part_offset = [center[i] - tool[i] for i in range(3)]
            part_half = max((axis[1] - axis[0]) / 2 for axis in hull)
        tcp = controller.tcp_link if controller.tcp_link in matrices else reference
        grasp_offset = [a - b for a, b in zip(representative_pose(matrices[tcp])[:3], tool, strict=True)]
        margin = self.margin

        def tool_box(point):
            return [[point[0] - radius - margin, point[0] + radius + margin],
                    [point[1] - radius - margin, point[1] + radius + margin],
                    [point[2] + low - margin, point[2] + high + margin]]

        def part_box(point, offset, half):
            return [[point[i] + offset[i] - half - margin, point[i] + offset[i] + half + margin] for i in range(3)]

        def cell(phase, points, link_boxes, holding, frozen):
            hull_box = _hull([tool_box(point) for point in points] + link_boxes)
            occupancy = {name: _possible([hull_box], bounds) for name, bounds in self.regions.items()}
            part_occupancy = frozen
            if holding is not None:
                offset, half = ((part_offset, part_half) if holding == held and part_offset is not None
                                else (grasp_offset, self._part_half(record, holding)))
                part_hull = _hull([part_box(point, offset, half) for point in points])
                part_occupancy = {name: _possible([part_hull], bounds) for name, bounds in self.regions.items()}
            return self._cell(phase, rid, occupancy, part if part_occupancy else None, part_occupancy)

        part = held or self._task_part(task)
        resting = ({name: (value,) for name, value in observation["parts"][part].items()}
                   if part in observation["parts"] else {})
        cells = [self._cell("at", rid, {name: (value,) for name, value in observation["occupancy"][rid].items()},
                            part if resting else None, resting)]
        holding, frozen = held, resting
        for kind, points, link_boxes in self._robot_path(owner, task, tool, transforms, names):
            if kind == "grasp":
                holding = part
                continue
            if kind == "release":
                # A released part stays where it was put; only its holder's occupancy ends there.
                holding, frozen = None, cells[-1][1]["part_occupancy"]
                continue
            for start, end in zip(points, points[1:], strict=False):
                cells.append(cell("between", [start, end], link_boxes, holding, frozen))
                cells.append(cell("at", [end], link_boxes, holding, frozen))
        return cells

    def _part_half(self, record: dict, part: str) -> float:
        hull = _hull(self._part_boxes(record, part))
        return max((axis[1] - axis[0]) / 2 for axis in hull)

    def _robot_path(self, owner, task: dict, tool: list, transforms, names: list) -> list:
        """Walk the saved program to its targets, waypoints and custody changes, without moving."""
        from cais_spade_llm.recovery_framework.nominal_safety_adapter import _READS, _guard
        from cais_spade_llm.recovery_framework.workflow_execution import _robot_arguments
        from cais_spade_llm.resources.continuous_motion import representative_pose, transformed_box
        from cais_spade_llm.resources.gazebo_programs import saved_robot_definition
        from cais_spade_llm.resources.robot.robot_task_model import _resolve_value
        from cais_spade_llm.resources.robot.robot_task_runtime import (
            _build_runtime_state,
            _primitive_payload_from_result,
        )

        controller, rid = owner._controller, owner.agent_name
        settings = controller.controller_config.get("cartesian_motion", {})
        program, arguments = _robot_arguments(self.runtime.context, owner, task)
        definition = saved_robot_definition(self.runtime.context.inputs["scene"], rid, program)
        args = {row.name: deepcopy(row.default) for row in definition.arguments if row.default is not None}
        args.update(arguments)
        state, outputs, path = _build_runtime_state(owner), {}, []
        configuration = controller.recovery_safety_observer.motion_configuration
        configuration_components = configuration["components"]
        current = list(tool)
        for step in definition.program.steps:
            try:
                enabled = all(_guard(guard, owner, args, state, outputs) for guard in step.when)
            except (ValueError, KeyError, TypeError):
                # A prediction keeps a step whose condition it cannot read; it only widens the envelope.
                enabled = True
            if not enabled:
                continue
            params = _resolve_value(step.params, args=args, runtime_state=state, step_outputs=outputs)
            params = {key: value for key, value in params.items() if not key.startswith("_")}
            if step.op in _READS:
                result = _read_result(getattr(controller, step.op)(**deepcopy(params)))
                if result.get("success") is not True:
                    raise ValueError("Nominal prediction read unavailable: " + step.id)
                if step.store_as:
                    outputs[step.store_as] = _primitive_payload_from_result(
                        step=step, params=params, primitive_result=result, runtime_state=state)
                continue
            if step.op in {"grasp_part", "attach_part"}:
                path.append(("grasp", params.get("part_name") or args.get("part_name"), []))
                continue
            if step.op in {"release_part", "detach_part"}:
                path.append(("release", None, []))
                continue
            link_boxes = []
            if step.op == "move_to_named_pose":
                positions = controller.named_positions[params["pose_name"]]
                matrices = transforms(positions)
                target = representative_pose(matrices[configuration["reference_link"]])[:3]
                link_boxes = [[[box["minimum"][i][0], box["maximum"][i][1]] for i in range(3)]
                              for box in (transformed_box(matrices[component["link"]], component["bounds"],
                                                          component["id"])
                                          for component in configuration_components)]
            elif step.op == "move_cartesian":
                target = [float(params[key]) for key in ("x", "y", "z")]
            elif step.op == "move_relative":
                target = [current[i] + float(params.get(key, 0.0)) for i, key in enumerate(("dx", "dy", "dz"))]
            else:
                continue
            route = _configured_route(step.id, settings)
            if route and math.dist(current, route[-1]) < math.dist(current, route[0]):
                route.reverse()
            path.append(("move", [current, *route, target], link_boxes))
            current = target
        return path

    def _kmr_cells(self, owner, task: dict, observation: dict) -> list:
        """Envelope the KMR base along its route and its arm around the task's gripper positions.

        The KMR runs a whole task in its worker process, so no single trajectory reaches the send
        check; this envelope is its only physical check. Where the arm works, each arm link is
        bounded by its configured chain lengths around the gripper positions the worker computes
        from, whatever joint solution the planner picks. While the base moves the arm stays parked,
        so the exact geometry of its current, parked and carrying configurations is swept piece by
        piece along each route segment.
        """
        rid, record = owner.agent_name, observation["record"]
        scene = self.runtime.context.inputs["scene"]
        kmr = _KMRGeometry(owner, record, self.resources[rid], scene["KMR"], self.margin)
        steps = [row["op"] for row in scene["resource_programs"]["resources"][rid]["functions"]
                 [task["event_name"]]["program"]["steps"]]
        split = steps.index("move_base") if "move_base" in steps else len(steps)
        part = self._held_part(rid) or self._task_part(task)
        path = self._kmr_route(kmr.here, task["parameters"], part) if split < len(steps) else [kmr.here]
        # Without a base move the arm works at the current dock; otherwise it only parks there.
        work = self._kmr_work(task["parameters"], part, record)
        work_here = [kmr.tcp(), *(work if split == len(steps) else [])]
        carried = self._held_part(rid) is not None or "grasp_part" in steps
        resting = ({name: (value,) for name, value in observation["parts"][part].items()}
                   if part in observation["parts"] else {})
        half = self._part_half(record, part) + self.margin if carried and resting else 0.0

        def cell(phase, boxes, points):
            occupancy = {name: _possible(boxes, bounds) for name, bounds in self.regions.items()}
            moved = resting
            if carried and resting:
                spot = _hull([[[point[i] - half, point[i] + half] for i in range(3)] for point in points])
                moved = {name: _possible([spot], bounds) for name, bounds in self.regions.items()}
            return self._cell(phase, rid, occupancy, part if moved else None, moved)

        cells = [self._cell("at", rid, {name: (value,) for name, value in observation["occupancy"][rid].items()},
                            part if resting else None, resting)]
        if _ARM_OPERATIONS & set(steps[:split]):
            points = kmr.gripper_points(kmr.here, work_here)
            boxes = kmr.body(kmr.here) + kmr.arm(kmr.here, points)
            cells += [cell("between", boxes, points), cell("at", boxes, points)]
        for start, end in zip(path, path[1:], strict=False):
            ends = [kmr.transport(start), kmr.transport(end)]
            pieces = kmr.body(end) + kmr.parked_arm(end)
            # Each piece translates with the base, so its sweep is the hull of its two ends.
            swept = [_hull([first, last]) for first, last in zip(
                kmr.body(start) + kmr.parked_arm(start), pieces, strict=True)]
            cells += [cell("between", swept, ends), cell("at", pieces, ends[1:])]
        if split < len(steps) and _ARM_OPERATIONS & set(steps[split + 1:]):
            points = kmr.gripper_points(path[-1], work)
            boxes = kmr.body(path[-1]) + kmr.arm(path[-1], points)
            cells += [cell("between", boxes, points), cell("at", boxes, points)]
        return cells

    def _kmr_route(self, here: list, parameters: dict, part: str | None) -> list:
        """Base poses from the current dock through the configured route to the task's dock."""
        from cais_spade_llm.recovery_framework.kmr_recovery import pickup_docking_pose

        scene = self.runtime.context.inputs["scene"]
        routes = scene["KMR"].get("predefined_route_waypoints", [])
        docks = {location: _location_dock(scene, location)
                 for location in {name for route in routes for name in route["resources"]}}
        source = min(docks, key=lambda name: math.dist(docks[name][:2], here[:2]))
        destination = (parameters.get("origin_resource_location") or parameters.get("target_resource")
                       or parameters.get("destination_location"))
        if destination not in docks:
            raise ValueError("KMR task destination has no configured dock: " + str(destination))
        target = (pickup_docking_pose(scene, destination, part) if "origin_resource_location" in parameters
                  and part else docks[destination])
        path = [here]
        for route in routes:
            if set(route["resources"]) == {source, destination}:
                poses = route["poses"] if route["resources"][0] == source else list(reversed(route["poses"]))
                path.extend(list(pose) for pose in poses)
        return [*path, list(target)]

    def _kmr_work(self, parameters: dict, part: str | None, record: dict) -> list:
        """The gripper's work positions: the part where it lies, or the recorded fixture access poses."""
        scene = self.runtime.context.inputs["scene"]
        destination = parameters.get("destination_location") or parameters.get("target_resource")
        machine = next((row for row in scene["machines"] if row["resource_id"] == destination), None)
        if machine is not None and "KMR_access" in machine:
            return [list(machine["KMR_access"][key][:3]) for key in ("approach_pose", "target_pose")]
        if part is not None and self.part_models.get(part) in record["models"]:
            return [list(record["models"][self.part_models[part]]["pose"][:3])]
        return []

    def _static_cells(self, rid: str, observation: dict) -> list:
        """Static equipment keeps its body; a part it moves stays within its own extent."""
        from cais_spade_llm.recovery_framework.gazebo_safety_preparation import _links

        record = observation["record"]
        boxes = [row["bounds"] for link in _links(record, self.resources[rid]).values() for row in link["collisions"]]
        hull = _hull(boxes)
        occupancy = {name: (value,) for name, value in observation["occupancy"][rid].items()}
        carried = [part for part, row in self.runtime.context.part_tracker.items() if row.get("location") == rid]
        part = carried[0] if len(carried) == 1 else None
        start_part = ({name: (value,) for name, value in observation["parts"][part].items()}
                      if part is not None and part in observation["parts"] else {})
        moved = {name: _possible([hull], bounds) for name, bounds in self.regions.items()} if part else {}
        return [self._cell("at", rid, occupancy, part, start_part),
                self._cell("between", rid, occupancy, part, moved)]

    # ------------------------------------------------------------------ admission

    def decide(self, prediction: dict) -> dict:
        """Check a prediction against the current claims and pending completions.

        Run under the admission lock, in the same section as the symbolic commit and ``claim``.
        """
        task = prediction["task"]
        self._refresh(prediction["observation"])
        started = time.monotonic()
        violated = self._observe_states(prediction["observation"])
        result = ({"status": "held", "reason": "observed_history_violates_specification", "rule_id": violated,
                   "transition": None} if violated else
                  self._step(prediction["cells"], self._base(prediction["observation"], task["resource_id"])))
        self.history.append({
            "kind": "start", "time": time.time(), "task_id": task["task_id"], "resource_id": task["resource_id"],
            "event_name": task["event_name"], "status": result["status"], "rule_id": result.get("rule_id"),
            "prediction_time_s": prediction["prediction_time_s"], "check_time_s": time.monotonic() - started,
            "regions": sorted(prediction["regions"])})
        if result["status"] != "allowed":
            return {"status": "held", "reason": result["reason"], "rule_id": result["rule_id"],
                    "transition": result["transition"]}
        return {"status": "allowed", "regions": sorted(prediction["regions"])}

    def claim(self, prediction: dict) -> None:
        """Hold an admitted prediction's regions, carried-part regions and pending completions."""
        task = prediction["task"]
        self.work[task["task_id"]] = {
            "resource_id": task["resource_id"], "status": "active", "regions": set(prediction["regions"]),
            "part_regions": {part: set(regions) for part, regions in prediction["part_regions"].items()},
            "completions": prediction["completions"]}

    def admitted(self, task_id: str | None, resource_id: str) -> bool:
        """Whether a task of this resource holds an active nominal claim."""
        with self.lock:
            entry = self.work.get(task_id)
            return entry is not None and entry["resource_id"] == resource_id and entry["status"] == "active"

    def retire(self, task_id: str) -> None:
        """Mark a task's execution ended; its claim stays until its resource is observed stopped."""
        with self.lock:
            entry = self.work.get(task_id)
            if entry is not None and entry["status"] == "active":
                entry.update(status="retired", retired_at=time.monotonic())

    # ------------------------------------------------------------------ native sends

    def guard_send(self, owner, topic: str, trajectory) -> None:
        """Sweep one native arm trajectory exactly and admit it, waiting stopped while it is held.

        Raises:
            ValueError: The send stayed held past the configured wait, or cannot be swept.
        """
        controller = owner._controller
        if topic != controller.arm_trajectory_topic:
            return
        rid = owner.agent_name
        started = time.monotonic()
        observation = self._observe()
        cells = self._sweep(owner, trajectory, observation)
        waited = 0.0
        while True:
            with self.lock:
                self._refresh(observation)
                entry = next((row for row in self.work.values()
                              if row["resource_id"] == rid and row["status"] == "active"), None)
                violated = self._observe_states(observation)
                result = ({"status": "held", "rule_id": violated} if violated
                          else self._step(cells, self._base(observation, rid)))
                if result["status"] == "allowed":
                    # Until later observations narrow them, any state along this trajectory may hold.
                    self.states = result["prefixes"]
                    self.states_observed = time.monotonic()
                    if entry is None:
                        # A motion outside any admitted task holds its regions until the next snapshot.
                        entry = self.work.setdefault(rid + ":untracked", {
                            "resource_id": rid, "status": "retired", "regions": set(), "part_regions": {},
                            "completions": frozenset()})
                        entry["retired_at"] = time.monotonic()
                    for _, cell in cells:
                        entry["regions"].update(name for name, value in cell["occupancy"].items() if True in value)
                        if cell["part"] is not None:
                            entry["part_regions"].setdefault(cell["part"], set()).update(
                                name for name, value in cell["part_occupancy"].items() if True in value)
                    self.history.append({"kind": "send", "time": time.time(), "resource_id": rid,
                                         "status": "allowed", "waited_s": waited,
                                         "check_time_s": time.monotonic() - started - waited})
                    return
            waited = time.monotonic() - started
            if waited >= self.wait_timeout_s:
                with self.lock:
                    self.history.append({"kind": "send", "time": time.time(), "resource_id": rid,
                                         "status": "held", "rule_id": result["rule_id"], "waited_s": waited})
                raise ValueError(f"Nominal send held past {self.wait_timeout_s:.0f} s by {result['rule_id']}")
            time.sleep(self.poll_s)
            observation = self._observe()

    def _sweep(self, owner, trajectory, observation: dict) -> list:
        """Cells over one native trajectory from configured kinematics, refined where a region is uncertain."""
        from cais_spade_llm.recovery_framework.gazebo_safety_preparation import _part_local_bounds
        from cais_spade_llm.recovery_framework.geometry import multiply, rotate
        from cais_spade_llm.resources.continuous_geometry import observed_joint_configuration
        from cais_spade_llm.resources.continuous_motion import ContinuousMotion, occupancy
        from cais_spade_llm.resources.resource_safety_preparation import trajectory_record

        controller, rid = owner._controller, owner.agent_name
        record = observation["record"]
        observer = controller.recovery_safety_observer
        model = record["models"][self.resources[rid]["model"]]
        configuration = observed_joint_configuration(
            observer.motion_configuration, model["joints"],
            model["links"][observer.motion_configuration["root"]]["pose"])
        names = list(configuration["joint_names"])
        raw = trajectory_record(trajectory, validate=False)
        order = [raw["joint_names"].index(name) for name in names]
        points = [{**row, "positions": [row["positions"][i] for i in order],
                   **{field: [row[field][i] for i in order] for field in ("velocities", "accelerations", "effort")
                      if len(row.get(field, [])) == len(order)}} for row in raw["points"]]
        current = [model["joints"][name]["position"] for name in names]
        first = points[0]["time_from_start"]
        if first["sec"] or first["nanosec"]:
            # The controller starts from where the arm stands; make that start explicit at time zero.
            points.insert(0, {"positions": current, "velocities": [0.0] * len(names),
                              "accelerations": [0.0] * len(names), "effort": [],
                              "time_from_start": {"sec": 0, "nanosec": 0}})
        last = points[-1]["time_from_start"]
        motion = ContinuousMotion({"joint_names": names, "points": points,
                                   "duration_ns": last["sec"] * 1_000_000_000 + last["nanosec"]}, configuration)
        attached = getattr(controller, "_attached_model", None)
        part = next((name for name, model_name in self.part_models.items() if model_name == attached),
                    None) if attached else None
        grasp = bounds = None
        if part is not None:
            row = record["models"][self.part_models[part]]
            pose = row["pose"]
            tool = motion.reference_pose(0)
            inverse = [-tool[3], -tool[4], -tool[5], tool[6]]
            grasp = [*rotate(inverse, [pose[i] - tool[i] for i in range(3)]), *multiply(inverse, pose[3:])]
            bounds = _part_local_bounds(row["links"], pose)

        def evaluate(start, end):
            boxes = motion.boxes(start, end)
            robot = {name: tuple(occupancy(boxes, region)) for name, region in self.regions.items()}
            held = {}
            if part is not None:
                box = motion.carried_part_box(start, end, grasp, bounds, part)
                held = {name: tuple(occupancy([box], region)) for name, region in self.regions.items()}
            return robot, held

        def uncertain(values):
            return any(len(value) > 1 for row in values for value in row.values())

        times = motion.exact_times
        # A send only needs which regions it may touch and when a carried part may enter one.
        intervals, budget = [], 1000
        stack = [(0, len(times) - 1, times[0], times[-1])]
        while stack:
            first, last, start, end = stack.pop()
            values = evaluate(start, end)
            budget -= 1
            if uncertain(values) and budget > 0 and end - start > Fraction(1, 10):
                if last - first > 1:
                    middle = (first + last) // 2
                    stack += [(middle, last, times[middle], end), (first, middle, start, times[middle])]
                else:
                    middle = (start + end) / 2
                    stack += [(first, last, middle, end), (first, last, start, middle)]
                continue
            intervals.append((start, end, values))
        intervals.sort(key=lambda row: row[0])
        cells = []
        for index, (start, end, (robot, held)) in enumerate(intervals):
            if index == 0:
                cells.append(self._cell("at", rid, evaluate(start, start)[0], part, evaluate(start, start)[1]))
            cells.append(self._cell("between", rid, robot, part, held))
            point = evaluate(end, end)
            cells.append(self._cell("at", rid, point[0], part, point[1]))
        return _entries(cells)

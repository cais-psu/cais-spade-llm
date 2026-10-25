from __future__ import annotations

"""RA feasibility of one proposed recovery event: Algorithm 1 and Eq. 12 of the recovery paper.

Backward derivation (Alg. 1) goes from each condition the event changes to the acting resource's
primitives whose declared effects produce the stated value. Nominal events are never used: a recovery
event is a new transition. The forward check (Eq. 12) then needs, for every changed condition, one
producer whose preconditions hold in the projected state v, whose parameters lie in their declared
domain, and whose continuous constraint holds: a pose within the resource's own kinematic chain, a
mobile base's footprint on free floor. The event's primitive set is the union of one feasible producer
per condition, in no order; composition orders it later.

Primitives never declare where a part or resource is (they must not be symbolically connectable), so
location, slot and custody conditions are decided by the acting resource's capability: a known place
(recorded geometry), within its physical reach, plus navigation for a mobile base. Some conditions need
no derivation: ``resource_state`` is the free DES state name, an unchanged value needs nothing, and
bookkeeping (``task_ctx.*``, private fields, the KMR's ``approached_part``) is skipped.

Results state facts only, never a fix: FEASIBLE with the primitive set, INFEASIBLE with the fact that
refuses the event, or NEEDS_CONTEXT when recorded evidence or a bounded search leaves it unresolved.
"""

import math
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from scipy.spatial.transform import Rotation

FEASIBLE, INFEASIBLE, NEEDS_CONTEXT = "FEASIBLE", "INFEASIBLE", "NEEDS_CONTEXT"
# The user decided on October 24 that the KMR's approach record is bookkeeping, like task_ctx.*.
_BOOKKEEPING = frozenset({"approached_part"})
# Gripper headings tried for a top-down grasp or release, about the world vertical.
_TOP_DOWN_YAWS_RAD = (0.0, math.pi / 2, math.pi, -math.pi / 2)
# Largest recorded surfaces of a place body tried as its work area.
_BODY_SURFACES = 3
_ORDER = {FEASIBLE: 0, NEEDS_CONTEXT: 1, INFEASIBLE: 2}


@dataclass(frozen=True)
class RecordedCell:
    """What a check reads besides the projected state: the recorded cell and current declarations.

    Attributes:
        scene: The captured Gazebo scene.
        resources: Recorded per-resource physical evidence (joint state, poses, TCP offset).
        models: Recorded per-resource kinematic models (``configuration`` with joint bounds).
        regions: Recorded named areas, such as a tray, a buffer zone or a belt loading position.
        bodies: Recorded scene models with their collision boxes.
        parts: Recorded part poses at the failure.
        floor: The navigation floor map that mobile bases plan around.
        declarations: Every resource's current primitive contracts, per primitive.
        rids: Resource id per JID.
    """

    scene: dict
    resources: dict
    models: dict
    regions: dict
    bodies: dict
    parts: dict
    floor: dict
    declarations: dict
    rids: dict


def recorded_cell(observations: dict, floor: dict) -> RecordedCell:
    """Read the recorded cell of a saved failure capture.

    Args:
        observations: The saved capture.
        floor: The navigation floor map (``floor_extent_xy`` and ``floor_obstacles_xy``).

    Returns:
        The cell a check reads.
    """
    from cais_spade_llm.resources.resource_primitives import current_primitive_declarations

    physical = observations["physical_evidence"]
    scene = observations["inputs"]["scene"]
    return RecordedCell(
        scene=scene, resources=physical.get("resources", {}), models=physical.get("resource_models", {}),
        regions=physical.get("geometry", {}).get("regions", {}),
        bodies=physical.get("physics_evidence", {}).get("models", {}), parts=physical.get("parts", {}), floor=floor,
        declarations=current_primitive_declarations(scene),
        rids={jid: rid for rid, jid in observations.get("jids", {}).items()})


def _worst(statuses: Iterable[str]) -> str:
    return max(statuses, key=_ORDER.__getitem__, default=FEASIBLE)


def _best(statuses: Iterable[str]) -> str:
    return min(statuses, key=_ORDER.__getitem__, default=INFEASIBLE)


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {_text(item)}" for key, item in value.items()) + "}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return "null" if value is None else str(value)


class _Check:
    """One event checked against the projected state v and the recorded cell."""

    def __init__(self, event: dict, session: dict, prepared: dict, cell: RecordedCell) -> None:
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
            multi_turn,
        )

        self.event, self.cell = event, cell
        self.jid = event.get("resource_jid")
        self.rid = cell.rids.get(self.jid, self.jid)
        self.part = event.get("part_name")
        self.resources, self.parts = multi_turn._projected_outline_validation_context(
            session_state=session, prepared_recovery_request=prepared)
        self.models = session.get("recovery_des_models") or {}
        self.failure_parts = session.get("base_symbolic_parts") or {}
        self.part_names = set(self.parts) | set(session.get("symbolic_parts") or {})
        self.end = dict(event.get("expected_end_state") or {})
        self.neighbors = dict(event.get("neighbor_end_state") or {})

    # Projected state v.

    def value(self, jid: str, name: str, scope: str = "resource") -> tuple[bool, Any]:
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
            multi_turn,
        )

        row = self.parts.get(self.part, {}) if scope == "part" else self.resources.get(jid, {})
        return multi_turn._projected_state_value(row=row, field_name=name, scope=scope)

    def variables(self, jid: str) -> dict:
        return (self.models.get(jid) or {}).get("state_variables") or {}

    def name_of(self, jid: str) -> str:
        return self.cell.rids.get(jid, jid)

    # Alg. 1: the changed conditions and how each is derived.

    def conditions(self) -> list[dict]:
        rows = [self.condition(self.jid, name, after) for name, after in self.end.items()]
        for jid, values in self.neighbors.items():
            rows += [self.condition(jid, name, after) for name, after in (values or {}).items()]
        return rows

    def condition(self, jid: str, name: str, after: Any) -> dict:
        # A part's variables are the part-scoped ones any resource declares, as the knowledge base lists them.
        declaration = self.variables(jid).get(name) or next(
            (row for model in self.models.values() for key, row in (model.get("state_variables") or {}).items()
             if key == name and row.get("scope") == "part"), None)
        row = {"entity": self.name_of(jid), "condition": name, "after": deepcopy(after)}
        if name == "resource_state":
            return {**row, "derivation": "free", "status": FEASIBLE}
        if name in _BOOKKEEPING or name.startswith(("task_ctx.", "_")) or (declaration or {}).get("private"):
            return {**row, "derivation": "bookkeeping", "status": FEASIBLE}
        if declaration is None:
            return {**row, "derivation": "undeclared", "status": INFEASIBLE,
                    "reason": f"{name} is not a declared variable of {self.name_of(jid)}'s current model"}
        scope = declaration.get("scope", "resource")
        known, before = self.value(jid, name, scope)
        row["before"] = deepcopy(before) if known else "unknown"
        if known and before == after:
            return {**row, "derivation": "unchanged", "status": FEASIBLE}
        if jid == self.jid:
            producers = self.producers(name, declaration, before if known else None, after, scope)
            if producers:
                status = _best(producer["status"] for producer in producers)
                result = {**row, "derivation": "producers", "status": status, "producers": producers}
                if status != FEASIBLE:
                    # Primitives that make the stated value but fail on v say why; the others only that they don't.
                    making = [producer for producer in producers if producer.get("makes_value")] or producers
                    result["reason"] = "; ".join(producer["reason"] for producer in making if producer.get("reason"))
                return result
        if self.is_place(jid, name, after, declaration):
            return {**row, "derivation": "place", "status": FEASIBLE}
        return {**row, "derivation": "no_producer", "status": INFEASIBLE,
                "reason": f"no {self.rid} primitive sets {self.name_of(jid)}.{name} to {_text(after)}"}

    def is_place(self, jid: str, name: str, after: Any, declaration: dict) -> bool:
        """Return whether a condition says where a part or resource is: a location, slot or custody."""
        domain = [value for value in declaration.get("domain", []) if value is not None]
        holds_parts = bool(domain) and set(domain) <= self.part_names
        per_part = any(name.endswith("." + part) for part in self.part_names)
        # A condition whose declared values are all places says where something is, null (no named place) too.
        place_domain = bool(domain) and all(self.is_place_value(value) for value in domain)
        places = after is None or after in self.part_names or self.is_place_value(after)
        if declaration.get("scope") == "part":
            return places
        return holds_parts or per_part or place_domain or (isinstance(after, str) and places)

    def is_place_value(self, value: Any) -> bool:
        """Return whether a value names a place: recorded geometry, a resource, or a named posture."""
        return isinstance(value, str) and (value in self.cell.rids or value in self.cell.rids.values()
                                           or value in self.named_poses() or bool(self.areas(value)))

    # Producers: declared effects that make the stated value, their preconditions and parameters.

    def effect_fields(self, name: str, declaration: dict, after: Any) -> tuple[str, ...]:
        if name == "pose":
            # The derived pose condition comes from the base_pose effect for a mobile base, else current_pose.
            return ("base_pose",) if declaration.get("locates_resource") else ("current_pose",)
        if name == "resource_location" and after in self.named_poses():
            # Only the resource's own named postures come from a primitive; any other place is capability.
            return ("current_pose_ref", "occupancy.location")
        return (name,)

    def producers(self, name: str, declaration: dict, before: Any, after: Any, scope: str) -> list[dict]:
        effects_key = "product_effects" if scope == "part" else "effects"
        found = []
        for primitive, contracts in (self.cell.declarations.get(self.rid) or {}).items():
            for contract in contracts:
                fields = [field for field in self.effect_fields(name, declaration, after)
                          if field in (contract.get(effects_key) or {})]
                rules = [contract[effects_key][field] for field in fields
                         if contract[effects_key][field] != {"set_unknown": True}]
                if rules:
                    found.append(self.producer(primitive, contract, rules[0], name, before, after))
        return found

    def producer(self, primitive: str, contract: dict, rule: Any, name: str, before: Any, after: Any) -> dict:
        theta = {"part_name": self.part} if self.part else {}
        row = {"primitive": primitive}
        if contract.get("controllable") is False:
            row["controllable"] = False
        status, reason, continuous = _produces(rule, before, after, theta, contract.get("parameters") or {})
        if status != FEASIBLE:
            return {**row, "status": status, "reason": f"{primitive}: {reason}"}
        row["makes_value"] = True
        failures = self.preconditions(contract, theta)
        if failures:
            return {**row, "status": _worst(status for status, _ in failures),
                    "reason": f"{primitive} needs " + "; ".join(reason for _, reason in failures)}
        if continuous:
            checked = self.pose_constraint(name, after)
            return {**row, **checked, "reason": f"{primitive}: {checked['reason']}"} if checked.get("reason") \
                else {**row, **checked}
        return {**row, "status": FEASIBLE}

    def preconditions(self, contract: dict, theta: dict) -> list[tuple[str, str]]:
        failures = []
        for source, scope in (("preconditions", "resource"), ("product_guards", "part")):
            for field, rule in (contract.get(source) or {}).items():
                known, value = self.value(self.jid, field, scope)
                if field == "current_pose" and not known:
                    known, value = self.value(self.jid, "pose", scope)
                holds = _holds(rule, value, theta) if known else None
                if holds is None:
                    failures.append((NEEDS_CONTEXT, f"{field} {_rule_text(rule, theta)}, which the projected "
                                                    "state does not record"))
                elif not holds:
                    failures.append((INFEASIBLE, f"{field} {_rule_text(rule, theta)}; projected {field} is "
                                                 f"{_text(value)}"))
        return failures

    # Eq. 12's continuous constraint on a stated pose.

    def pose_constraint(self, name: str, after: Any) -> dict:
        if self.mobile():
            if self.base_fits(after):
                return {"status": FEASIBLE}
            from cais_spade_llm.resources.kinematic_reach import footprint_obstacles

            footprint = self.cell.scene[self.rid]["platform_overall_dimensions_m"]
            stated = [after.get(axis, 0.0) for axis in ("x", "y", "yaw")] if isinstance(after, dict) else []
            numeric = len(stated) == 3 and all(isinstance(value, (int, float)) for value in stated)
            overlaps = footprint_obstacles(self.cell.floor, footprint, stated) if numeric and self.cell.floor else []
            where = f"overlaps {', '.join(overlaps)} on the floor map" if overlaps else "is not on free, connected floor"
            return {"status": INFEASIBLE, "reason": f"{self.rid}'s footprint at {_text(after)} {where}"}
        target = self.stated_tool_pose(after)
        if target is None:
            return {"status": NEEDS_CONTEXT, "reason": f"{name} {_text(after)} has no orientation to check"}
        return self.reach_pose(target, f"{name} {_text(after)}")

    def stated_tool_pose(self, after: Any) -> list[float] | None:
        if not isinstance(after, dict) or not all(isinstance(after.get(axis), (int, float)) for axis in "xyz"):
            return None
        orientation = [after.get(axis) for axis in ("qx", "qy", "qz", "qw")]
        if not all(isinstance(value, (int, float)) for value in orientation):
            # The controller keeps the current orientation when a move states none.
            current = self.current_pose()
            if current is None:
                return None
            orientation = current[3:]
        return [after["x"], after["y"], after["z"], *orientation]

    def current_pose(self) -> list[float] | None:
        known, pose = self.value(self.jid, "pose")
        if known and isinstance(pose, dict) and all(isinstance(pose.get(axis), (int, float))
                                                   for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")):
            return [pose[axis] for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")]
        recorded = (self.cell.resources.get(self.rid) or {}).get("current_pose")
        return list(recorded) if isinstance(recorded, list) and len(recorded) == 7 else None

    # Capability (Step 1 item 6): known place, physical reach, navigation.

    def named_poses(self) -> set[str]:
        return set((self.cell.models.get(self.rid) or {}).get("named_joint_poses") or {})

    def chain(self) -> dict | None:
        configuration = (self.cell.models.get(self.rid) or {}).get("configuration")
        return configuration if isinstance(configuration, dict) and configuration.get("joints") else None

    def mobile(self) -> bool:
        return isinstance((self.cell.scene.get(self.rid) or {}).get("platform_overall_dimensions_m"), list)

    def transport(self) -> bool:
        return isinstance((self.cell.scene.get(self.rid) or {}).get("simulation_transport"), dict)

    def areas(self, place: Any, slot: str | None = None) -> list[dict]:
        """Return a place's recorded areas: its named regions, else its scene model's collision surfaces.

        A region named ``<place>.<sub-area>`` is part of the place unless its surface belongs to another
        recorded body (a staging tray beside a machine is its own place). With a slot, the sub-areas the
        slot names are kept: the slot's value or the start of its variable name.
        """
        if not isinstance(place, str) or place in self.cell.rids:
            return []
        own, other = [], []
        for name, row in self.cell.regions.items():
            body = str((row.get("source") or {}).get("surface_collision") or "").split("::")[0]
            if not row.get("bounds"):
                continue
            if body == place:
                own.append({"name": name, **row})
            elif name == place or name.startswith(place + "."):
                (other if body and body != place else own).append({"name": name, **row})
        if slot:
            named = [row for row in own + other if row["name"] == place or slot == row["name"].split(".", 1)[-1]
                     or slot.startswith(row["name"].split(".", 1)[-1] + "_")]
            if named:
                return named
        if own:
            return own
        model = (self.cell.scene.get(place) or {}).get("gazebo_model", place)
        surfaces = [{"name": collision["id"], "bounds": collision["bounds"], "surface": True}
                    for link in ((self.cell.bodies.get(model) or self.cell.bodies.get(place) or {})
                                 .get("links") or {}).values()
                    for collision in link.get("collisions", []) if collision.get("bounds")]
        surfaces.sort(key=lambda row: -(row["bounds"][0][1] - row["bounds"][0][0])
                      * (row["bounds"][1][1] - row["bounds"][1][0]))
        return surfaces[:_BODY_SURFACES]

    def work_point(self, area: dict) -> list[float]:
        (x0, x1), (y0, y1), (z0, z1) = area["bounds"]
        # A collision surface is worked at its top face, a region that holds parts (it has a capacity) on
        # its floor, and any other region at its centre.
        z = z1 if area.get("surface") else z0 if "capacity" in area else (z0 + z1) / 2
        return [(x0 + x1) / 2, (y0 + y1) / 2, z]

    def orientations(self) -> list[list[float]]:
        down = Rotation.from_euler("x", math.pi)
        candidates = []
        current = self.current_pose()
        if current is not None and Rotation.from_quat(current[3:]).apply([0, 0, 1])[2] < -0.9:
            candidates.append(list(current[3:]))
        candidates += [(Rotation.from_euler("z", yaw) * down).as_quat().tolist() for yaw in _TOP_DOWN_YAWS_RAD]
        return candidates[:3] if self.mobile() else candidates

    def reference_pose(self, point: list[float], orientation: list[float]) -> list[float]:
        """Return the chain's reference-link pose that puts the gripper point at a point."""
        recorded = self.cell.resources.get(self.rid) or {}
        offset = recorded.get("tcp_offset_z")
        chain = self.chain() or {}
        if isinstance(offset, (int, float)) and recorded.get("tcp_link") != chain.get("reference_link"):
            point = (point - Rotation.from_quat(orientation).apply([0.0, 0.0, abs(offset)])).tolist()
        return [*point, *orientation]

    def seeds(self) -> list[list[float]]:
        recorded = self.cell.resources.get(self.rid) or {}
        chain = self.chain() or {}
        seeds = []
        if recorded.get("joint_names") == chain.get("joint_names") and isinstance(recorded.get("joint_positions"), list):
            seeds.append(list(recorded["joint_positions"]))
        for pose in ((self.cell.models.get(self.rid) or {}).get("named_joint_poses") or {}).values():
            if isinstance(pose, dict) and pose.get("joint_names") == chain.get("joint_names"):
                seeds.append(list(pose["positions"]))
        return seeds

    def reach_pose(self, target: list[float], what: str) -> dict:
        from cais_spade_llm.resources.kinematic_reach import reach

        chain = self.chain()
        if chain is None:
            return {"status": NEEDS_CONTEXT, "reason": f"{self.rid} has no recorded kinematic chain"}
        checked = reach(chain, target, seeds=self.seeds())
        if checked["feasibility_status"] == FEASIBLE:
            return {"status": FEASIBLE, "witness": {"joint_positions": checked["joint_positions"]}}
        if checked["feasibility_status"] == INFEASIBLE:
            return {"status": INFEASIBLE, "reason": f"{what} lies beyond {self.rid}'s kinematic chain"}
        return {"status": NEEDS_CONTEXT, "reason": f"{self.rid}'s bounded joint search found no configuration "
                                                   f"for {what}; reachability is unresolved"}

    def base_pose(self) -> list[float] | None:
        known, pose = self.value(self.jid, "pose")
        # A planar base pose carries a yaw; a recorded arm pose carries an orientation quaternion instead.
        if known and isinstance(pose, dict) and all(isinstance(pose.get(axis), (int, float)) for axis in ("x", "y", "yaw")):
            return [pose["x"], pose["y"], pose["yaw"]]
        recorded = (self.cell.resources.get(self.rid) or {}).get("base_pose")
        if isinstance(recorded, list) and len(recorded) == 7:
            return [recorded[0], recorded[1], Rotation.from_quat(recorded[3:]).as_euler("xyz")[2]]
        return None

    def docks(self) -> list[list[float]]:
        key = f"{self.rid}_docking_pose"
        entries = [row for row in self.cell.scene.values() if isinstance(row, dict)] + list(self.cell.scene.get("machines", []))
        return [[row[key][0], row[key][1], row[key][5]] for row in entries
                if isinstance(row.get(key), list) and len(row[key]) == 6]

    def base_candidates(self, around: list[float]) -> list[list[float]]:
        from cais_spade_llm.resources.kinematic_reach import free_base_poses

        start = self.base_pose()
        footprint = self.cell.scene[self.rid]["platform_overall_dimensions_m"]
        if start is None or not self.cell.floor:
            return []
        return free_base_poses(self.cell.floor, footprint, start[:2], around[:2], docks=self.docks())

    def base_fits(self, pose: Any) -> bool:
        from cais_spade_llm.resources.kinematic_reach import free_base_poses

        start = self.base_pose()
        if not isinstance(pose, dict) or start is None or not self.cell.floor:
            return False
        stated = [pose.get("x"), pose.get("y"), pose.get("yaw", 0.0)]
        if not all(isinstance(value, (int, float)) for value in stated):
            return False
        footprint = self.cell.scene[self.rid]["platform_overall_dimensions_m"]
        fitting = free_base_poses(self.cell.floor, footprint, start[:2], stated[:2], docks=[stated], radius_m=0.0)
        return bool(fitting) and fitting[0] == stated

    def reach_area(self, area: dict, what: str, *, part_start: list[float] | None = None) -> dict:
        """Check whether the acting resource can bring its gripper, or a carried part, to a recorded area."""
        from cais_spade_llm.resources.kinematic_reach import reach_from_base

        point = self.work_point(area)
        if self.transport():
            return self.transport_reach(area, point, part_start, what)
        chain = self.chain()
        if chain is None:
            return {"status": INFEASIBLE, "reason": f"{self.rid} declares no motion that moves parts"}
        if not self.mobile():
            results = []
            for orientation in self.orientations():
                checked = self.reach_pose(self.reference_pose(point, orientation), what)
                if checked["status"] == FEASIBLE or (checked["status"] == INFEASIBLE):
                    return checked
                results.append(checked)
            return results[-1]
        stated = self.end.get("pose")
        bases = ([[stated["x"], stated["y"], stated.get("yaw", 0.0)]] if isinstance(stated, dict)
                 and self.base_fits(stated) else self.base_candidates(point))
        if not bases:
            return {"status": NEEDS_CONTEXT, "reason": f"no free, connected base pose for {self.rid} near {what}"}
        for orientation in self.orientations():
            checked = reach_from_base(chain, self.reference_pose(point, orientation), bases,
                                      base_height_m=chain["root_xyz"][2])
            if checked["feasibility_status"] == FEASIBLE:
                return {"status": FEASIBLE, "witness": {"base_pose": checked["base_pose"],
                                                        "joint_positions": checked["joint_positions"]}}
        return {"status": NEEDS_CONTEXT, "reason": f"no candidate base pose gave {self.rid} a configuration at "
                                                   f"{what}; reachability is unresolved"}

    def transport_reach(self, area: dict, point: list[float], start: list[float] | None, what: str) -> dict:
        from cais_spade_llm.recovery_framework.workflow_gazebo import transport_chain
        from cais_spade_llm.resources.kinematic_reach import reach

        if start is None:
            return {"status": NEEDS_CONTEXT, "reason": f"the projected state does not record where {self.part} "
                                                       f"is on {self.rid}"}
        chain = transport_chain(self.cell.scene, self.rid, start)
        axis = chain["joints"][0]["axis"]
        along = sum((point[index] - start[index]) * axis[index] for index in range(3))
        target = [start[index] + along * axis[index] for index in range(3)]
        (x0, x1), (y0, y1), _ = area["bounds"]
        if not (x0 <= target[0] <= x1 and y0 <= target[1] <= y1):
            return {"status": INFEASIBLE, "reason": f"{what} is not on {self.rid}'s path for {self.part}"}
        checked = reach(chain, [*target, 0.0, 0.0, 0.0, 1.0])
        if checked["feasibility_status"] == FEASIBLE:
            return {"status": FEASIBLE, "witness": {"travel_m": checked["joint_positions"][0]}}
        return {"status": INFEASIBLE if checked["feasibility_status"] == INFEASIBLE else NEEDS_CONTEXT,
                "reason": f"{what} is upstream of {self.part} or beyond {self.rid}'s last stop"}

    def reach_place(self, place: Any, slot: str | None, what: str, *, part_start: list[float] | None = None) -> dict:
        areas = self.areas(place, slot)
        if not areas:
            return {"status": NEEDS_CONTEXT, "reason": f"{_text(place)} has no recorded geometry"}
        results = [self.reach_area(area, f"{what} ({area['name']})", part_start=part_start) for area in areas]
        best = _best(result["status"] for result in results)
        chosen = next(result for result in results if result["status"] == best)
        return {**chosen, "place": place, **({"slot": slot} if slot else {})}

    # Custody: where the event's part is in v and where it goes.

    def slot_of(self, part: str, place: str) -> str | None:
        """Return the slot of a place that holds a part in v: a part-valued or per-part variable of its owner."""
        jid = next((jid for jid, rid in self.cell.rids.items() if rid == place), None)
        if jid is None:
            return None
        row = self.resources.get(jid, {})
        for name in self.variables(jid):
            if row.get(name) == part and name != "held_part":
                return name
            if name.endswith("." + part) and isinstance(row.get(name), str):
                return row[name]
        return None

    def part_place(self) -> tuple[Any, str | None, list[float] | None]:
        """Return where the event's part is in v: a place, its slot, and its position if recorded.

        The position is the part's observed pose, else its pose recorded at the failure while it is still
        where it was then.
        """
        known, place = self.value(self.jid, "part_location", "part")
        row = self.parts.get(self.part, {})
        pose = row.get("observed_pose") or row.get("pose")
        recorded = (self.cell.parts.get(self.part) or {}).get("pose")
        if pose is None and known and place == (self.failure_parts.get(self.part) or {}).get("part_location"):
            pose = recorded
        position = None
        if isinstance(pose, dict) and all(isinstance(pose.get(axis), (int, float)) for axis in "xyz"):
            position = [pose["x"], pose["y"], pose["z"]]
        elif isinstance(pose, list) and len(pose) >= 3:
            position = list(pose[:3])
        place = place if known else None
        if isinstance(place, str) and place in self.cell.rids.values():
            slot = self.slot_of(self.part, place)
        else:
            slot = None
        return place, slot, position

    def destination(self) -> tuple[Any, str | None, dict | None]:
        """Return where the event puts its part: a neighbor's slot, else the stated part location."""
        for jid, values in self.neighbors.items():
            place = self.name_of(jid)
            for name, after in (values or {}).items():
                if after == self.part:
                    return place, name, {"jid": jid, "condition": name}
                if name.endswith("." + str(self.part)) and isinstance(after, str):
                    return place, after, {"jid": jid, "condition": name}
        place = self.end.get("part_location")
        return place, None, None

    def capability(self) -> list[dict]:
        """Check the custody and location changes of the event against the acting resource's capability."""
        rows = self.custody() if self.part is not None else []
        picks_or_places = {row["check"] for row in rows} & {"pick", "place"}
        location = self.end.get("resource_location")
        known, before = self.value(self.jid, "resource_location")
        if isinstance(location, str) and location != before and location not in self.named_poses():
            if self.mobile():
                areas = self.areas(location)
                bases = self.base_candidates(self.work_point(areas[0])) if areas else []
                rows.append({"check": "navigate", "place": location,
                             "status": FEASIBLE if bases else NEEDS_CONTEXT,
                             **({} if bases else {"reason": f"no free, connected base pose for {self.rid} near "
                                                            f"{location}"})})
            elif not picks_or_places or location not in {row.get("place") for row in rows}:
                rows.append({"check": "move", **self.reach_place(location, None, location)})
        return rows

    def custody(self) -> list[dict]:
        """Check where the event takes its part from and puts it, as the acting resource's reach."""
        own_slot = next((name for name in self.end if name.endswith("." + self.part)), None)
        if own_slot is not None and self.transport():
            return [self.transport_check(own_slot)]
        held_known, held_before = self.value(self.jid, "held_part")
        location_known, location_before = self.value(self.jid, "part_location", "part")
        holding = (held_known and held_before == self.part) or location_before == self.jid
        held_after = self.end.get("held_part", held_before if held_known else None)
        place, slot, neighbor = self.destination()
        takes = held_after == self.part or place in (self.jid, self.rid)
        puts = place not in (None, self.jid, self.rid) and (neighbor is not None or place != location_before)
        rows = []
        if (takes or puts) and not holding:
            rows.append({"check": "pick", "part": self.part, **self.reach_part()})
        if puts:
            rows.append({"check": "place", "part": self.part,
                         **self.reach_place(place, slot, f"{_text(place)}" + (f" {slot}" if slot else ""))})
            if neighbor is not None:
                rows.append(self.capacity(neighbor, slot))
        elif holding and "held_part" in self.end and held_after is None:
            # Released without a stated place, the part rests at the release pose (a pose place).
            rows.append({"check": "place", "part": self.part, "status": NEEDS_CONTEXT,
                         "reason": f"the support and clearance of {self.part} at a release pose are not checked yet"
                         if isinstance(self.end.get("pose"), dict) else f"the event does not state where {self.part} goes"})
        return rows

    def reach_part(self) -> dict:
        place, slot, position = self.part_place()
        if position is not None:
            area = {"name": "observed pose", "bounds": [[position[0]] * 2, [position[1]] * 2, [position[2]] * 2]}
            return self.reach_area(area, f"{self.part}'s observed position")
        return self.reach_place(place, slot, f"{self.part} at {_text(place)}" + (f" {slot}" if slot else ""))

    def transport_check(self, condition: str) -> dict:
        place = self.rid
        known, before = self.value(self.jid, condition)
        start_area = self.areas(place, before) if known and isinstance(before, str) else []
        _, _, position = self.part_place()
        start = position or (self.work_point(start_area[0]) if start_area else None)
        result = self.reach_place(place, self.end[condition], f"{self.end[condition]}", part_start=start)
        return {"check": "transport", "part": self.part, **result}

    def capacity(self, neighbor: dict, slot: str | None) -> dict:
        jid, condition = neighbor["jid"], neighbor["condition"]
        row = self.resources.get(jid, {})
        place = self.name_of(jid)
        holder = row.get(condition) if not condition.endswith("." + str(self.part)) else next(
            (name.rsplit(".", 1)[-1] for name, value in row.items() if value == slot and name != condition
             and name.startswith(condition.rsplit(".", 1)[0] + ".")), None)
        if holder not in (None, self.part, "unknown"):
            return {"check": "capacity", "place": place, "slot": slot, "status": INFEASIBLE,
                    "reason": f"{place} {slot or condition} holds {holder} in the projected state"}
        return {"check": "capacity", "place": place, "slot": slot, "status": FEASIBLE}


def _param_in_domain(name: str, value: Any, parameters: dict) -> tuple[bool, str]:
    schema = parameters.get(name) or {}
    if "enum" in schema and value not in schema["enum"]:
        return False, f"{name} {_text(value)} is outside its declared values {schema['enum']}"
    for bound, holds in (("minimum", lambda limit: value >= limit), ("maximum", lambda limit: value <= limit)):
        if bound in schema and not (isinstance(value, (int, float)) and holds(schema[bound])):
            return False, f"{name} {_text(value)} is outside its declared {bound} {schema[bound]}"
    return True, ""


def _bind(template: Any, value: Any, theta: dict, parameters: dict) -> tuple[bool, str]:
    """Match a stated value to an effect template, binding the template's parameters into theta."""
    if isinstance(template, dict) and set(template) == {"set_from_param"}:
        name = template["set_from_param"]
        if name in theta and theta[name] != value:
            return False, f"{name} is {_text(theta[name])}, not {_text(value)}"
        theta[name] = value
        return _param_in_domain(name, value, parameters)
    if isinstance(template, dict):
        if not isinstance(value, dict) or set(template) != set(value):
            return False, f"it sets {_text(template)}, not {_text(value)}"
        for key, item in template.items():
            matched, reason = _bind(item, value[key], theta, parameters)
            if not matched:
                return False, reason
        return True, ""
    if isinstance(template, list):
        if not isinstance(value, list) or len(template) != len(value):
            return False, f"it sets {_text(template)}, not {_text(value)}"
        for item, stated in zip(template, value, strict=True):
            matched, reason = _bind(item, stated, theta, parameters)
            if not matched:
                return False, reason
        return True, ""
    return (template == value), f"it sets {_text(template)}, not {_text(value)}"


def _produces(rule: Any, before: Any, after: Any, theta: dict, parameters: dict) -> tuple[str, str, bool]:
    """Return whether an effect rule can make the stated value, and whether it is a continuous effect.

    Returns:
        Status, the fact when it cannot, and whether Eq. 12's continuous constraint still applies.
    """
    if isinstance(rule, list):
        # A process record effect appends its entries to the part's record.
        if not isinstance(after, list) or not isinstance(before, list) or after[:len(before)] != before:
            return INFEASIBLE, f"it adds to the record, which is {_text(before)}, not {_text(after)}", False
        matched, reason = _bind(rule, after[len(before):], theta, parameters)
        return (FEASIBLE, "", False) if matched else (INFEASIBLE, reason, False)
    if not isinstance(rule, dict) or len(rule) != 1:
        return NEEDS_CONTEXT, f"effect {_text(rule)} is not interpreted", False
    (form, argument), = rule.items()
    if form == "set":
        matched, reason = _bind(argument, after, theta, parameters)
        return (FEASIBLE, "", False) if matched else (INFEASIBLE, reason, False)
    if form in {"set_from_param", "set_from_param_any_of"}:
        names = [argument] if form == "set_from_param" else list(argument)
        reasons = []
        for name in names:
            matched, reason = _bind({"set_from_param": name}, after, dict(theta), parameters)
            if matched:
                theta[name] = after
                return FEASIBLE, "", False
            reasons.append(reason)
        return INFEASIBLE, "; ".join(reasons), False
    if form == "set_status":
        expected = {**before, "status": argument} if isinstance(before, dict) else None
        if expected is not None and expected == after:
            return FEASIBLE, "", False
        return INFEASIBLE, f"it sets the status of {_text(before)} to {argument}, not {_text(after)}", False
    if form.startswith(("pose_absolute_from_", "pose_relative_from_")):
        fields = list(argument) if isinstance(argument, list) else []
        if not isinstance(after, dict) or not all(isinstance(after.get(field), (int, float)) for field in fields):
            return INFEASIBLE, f"it sets a pose ({', '.join(fields)}), not {_text(after)}", False
        if form.startswith("pose_relative_from_") and not isinstance(before, dict):
            return NEEDS_CONTEXT, "a relative move needs the projected pose", False
        return FEASIBLE, "", True
    return NEEDS_CONTEXT, f"effect form {form} is not interpreted", False


def _holds(rule: Any, value: Any, theta: dict) -> bool | None:
    """Evaluate one declared precondition on a projected value; None when it cannot be evaluated."""
    if not isinstance(rule, dict):
        return None
    checks = []
    for form, argument in rule.items():
        if form == "equals":
            checks.append(value == argument)
        elif form == "not_equals":
            checks.append(value != argument)
        elif form == "in":
            checks.append(value in argument)
        elif form == "exists":
            checks.append((value is not None) == bool(argument))
        elif form == "equals_from_param":
            if argument not in theta:
                return None
            checks.append(value == theta[argument])
        elif form in {"matches", "not_matches"}:
            matched = isinstance(value, Mapping) and all(value.get(key) == item for key, item in argument.items())
            checks.append(matched if form == "matches" else not matched)
        else:
            return None
    return all(checks)


def _rule_text(rule: Any, theta: dict) -> str:
    if not isinstance(rule, dict):
        return _text(rule)
    words = {"equals": "=", "not_equals": "≠", "in": "in", "exists": "exists:", "matches": "matches",
             "not_matches": "does not match"}
    parts = []
    for form, argument in rule.items():
        if form == "equals_from_param":
            parts.append(f"= {_text(theta.get(argument, argument))}")
        else:
            parts.append(f"{words.get(form, form)} {_text(argument)}")
    return " and ".join(parts)


def check_event(event: dict, session: dict, prepared: dict, cell: RecordedCell) -> dict:
    """Check one candidate event: backward derivation (Alg. 1), then the forward check (Eq. 12).

    Args:
        event: One candidate event as the LLM states it.
        session: The projected planning state before the event (v).
        prepared: Recorded failure context.
        cell: The recorded cell and current declarations.

    Returns:
        ``feasibility_status``, the derivation of every changed condition, the capability checks, the
        primitive set (one feasible producer per derived condition) and, unless FEASIBLE, the facts that
        refuse or leave the event unresolved.
    """
    from cais_spade_llm.resources.kinematic_reach import JOINT_SEARCH_CACHE

    token = JOINT_SEARCH_CACHE.set({})
    try:
        check = _Check(event, session, prepared, cell)
        conditions = check.conditions()
        capability = check.capability()
    finally:
        JOINT_SEARCH_CACHE.reset(token)
    status = _worst([row["status"] for row in conditions] + [row["status"] for row in capability])
    primitive_set = sorted({next(producer["primitive"] for producer in row["producers"]
                                 if producer["status"] == FEASIBLE)
                            for row in conditions if row["derivation"] == "producers" and row["status"] == FEASIBLE})
    reasons = list(dict.fromkeys(row["reason"] for row in conditions + capability
                                 if row["status"] != FEASIBLE and row.get("reason")))
    result = {"feasibility_status": status, "resource_jid": event.get("resource_jid"),
              "event_name": event.get("event_name"), "conditions": conditions, "capability": capability,
              "primitive_set": primitive_set}
    if reasons:
        result["reasons"] = reasons
    return result


def check_candidates(events: list[dict], selected: int, session: dict, prepared: dict,
                     cell: RecordedCell) -> tuple[int | None, list[dict]]:
    """Check the LLM's selected candidate first, then its other candidates in the LLM's order.

    Args:
        events: The LLM's candidate events.
        selected: The index of the candidate the LLM chose.
        session: The projected planning state before the event.
        prepared: Recorded failure context.
        cell: The recorded cell and current declarations.

    Returns:
        The index of the first feasible candidate in that order, or None, and every checked result.
    """
    order = [selected, *(index for index in range(len(events)) if index != selected)]
    results = []
    for index in order:
        result = {"candidate_index": index, **check_event(events[index], session, prepared, cell)}
        results.append(result)
        if result["feasibility_status"] == FEASIBLE:
            return index, results
    return None, results


def refusal_findings(events: list[dict], results: list[dict]) -> list[dict]:
    """Return one outline validation finding per refused candidate, stating only recorded facts."""
    findings = []
    for result in results:
        event = events[result["candidate_index"]]
        reason = "; ".join(result.get("reasons", []))
        if result["feasibility_status"] == NEEDS_CONTEXT:
            reason = "NEEDS_CONTEXT: " + reason
        findings.append({"phase": "outline", "constraint_code": "ra_feasibility", "validation_category": "RA",
                         "feasibility_status": result["feasibility_status"],
                         "resource_jid": event.get("resource_jid"), "part_name": event.get("part_name"),
                         "event_name": event.get("event_name"), "outline_id": event.get("outline_id"),
                         "task_id": event.get("outline_id"), "candidate_index": result["candidate_index"],
                         "reason": reason})
    return findings

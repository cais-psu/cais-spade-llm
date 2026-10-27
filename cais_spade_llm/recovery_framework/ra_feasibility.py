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
(recorded geometry), within its physical reach, plus navigation for a mobile base. A part released where
no declared place names the spot (a pose place) must rest right there on a recorded surface that nothing
recorded overlaps. Some conditions need no derivation: ``resource_state`` is the free DES state name, an
unchanged value needs nothing, and bookkeeping (``task_ctx.*``, private fields) is skipped.

Results state facts only, never a fix: FEASIBLE with the primitive set, INFEASIBLE with the fact that
refuses the event, or NEEDS_CONTEXT when recorded evidence or a bounded search leaves it unresolved.
"""

import math
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

FEASIBLE, INFEASIBLE, NEEDS_CONTEXT = "FEASIBLE", "INFEASIBLE", "NEEDS_CONTEXT"
# Gripper headings tried for a top-down grasp or release, about the world vertical.
_TOP_DOWN_YAWS_RAD = (0.0, math.pi / 2, math.pi, -math.pi / 2)
# Largest recorded surfaces of a place body tried as its work area.
_BODY_SURFACES = 3
# Recorded shapes that only touch do not overlap.
_TOUCH_M = 1e-6
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
        part_geometry: Recorded part extents, ``local_bounds`` about each part's own frame.
        contact_layer_m: The physics engine's recorded contact surface layer, how far resting bodies may
            sink into each other, or None when the capture records none.
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
    part_geometry: dict
    contact_layer_m: float | None


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
        rids={jid: rid for rid, jid in observations.get("jids", {}).items()},
        part_geometry=physical.get("geometry", {}).get("parts", {}),
        contact_layer_m=_contact_layer(physical.get("physics_evidence", {})))


def _contact_layer(physics: dict) -> float | None:
    """Return the recorded ODE contact surface layer, as the projected part support reads it."""
    import xml.etree.ElementTree as ET

    layer = physics.get("contact_surface_layer")
    sdf = (physics.get("physics_configuration") or {}).get("sdf")
    if layer is None and isinstance(sdf, str):
        try:
            layer = float(ET.fromstring(sdf).findtext("./ode/constraints/contact_surface_layer"))
        except (ET.ParseError, TypeError, ValueError):
            return None
    return float(layer) if isinstance(layer, (int, float)) and math.isfinite(layer) and layer >= 0 else None


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
        self.checks = session.get("ra_feasibility_checks") or []
        self.accepted = session.get("accepted_outline_prefix") or []
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
        if name.startswith(("task_ctx.", "_")) or (declaration or {}).get("private"):
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
                    # Alternatives whose guards on the part fail concern another job or part, unless all do.
                    making = [producer for producer in producers if producer.get("makes_value")] or producers
                    making = [producer for producer in making if producer.get("part_guards_hold", True)] or making
                    result["reason"] = "; ".join(dict.fromkeys(producer["reason"] for producer in making
                                                               if producer.get("reason")))
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
        places = after is None or (isinstance(after, str) and after in self.part_names) or self.is_place_value(after)
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
        if name == "resource_location" and isinstance(after, str) and after in self.named_poses():
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
        row["part_guards_hold"] = not self.preconditions({"product_guards": contract.get("product_guards")}, theta)
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

    def part_pose(self, part: str) -> tuple[Any, list[float] | None]:
        """Return where a part is in v: its place, and its pose if one is known.

        The pose is the part's observed pose; else, while no resource holds it and no place names where
        it is (or only a resource that has released it), the pose this check found where an applied event
        left it (a pose place); else its pose recorded at the failure while it is still where it was then.
        """
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
            multi_turn,
        )

        row = self.parts.get(part, {})
        known, place = multi_turn._projected_state_value(row=row, field_name="part_location", scope="part")
        pose = _pose_list(row.get("observed_pose") or row.get("pose"))
        held = row.get("part_holder_resource_jid") or row.get("current_holder_resource_jid")
        released = place is not None and self.released_by(place, part)
        if pose is None and known and not held and (place is None or released):
            pose = self.rest_pose(part)
        # A part an accepted event set down is no longer where the failure recorded it, although its
        # location may still name the resource that set it down.
        moved = released and any(self.moves_part(task, part) for task in self.accepted)
        if pose is None and known and not moved and place == (self.failure_parts.get(part) or {}).get("part_location"):
            pose = _pose_list((self.cell.parts.get(part) or {}).get("pose"))
        return (place if known else None), pose

    def part_place(self) -> tuple[Any, str | None, list[float] | None]:
        """Return where the event's part is in v: a place, its slot, and its position if known.

        A location that names a resource which put the part down without stating where says nothing; the
        resource whose own state records the part does.
        """
        place, pose = self.part_pose(self.part)
        if pose is None and place is not None and self.released_by(place, self.part):
            place, slot = self.recorded_place(self.part) or (None, None)
            return place, slot, None
        if isinstance(place, str) and place in self.cell.rids.values():
            slot = self.slot_of(self.part, place)
        else:
            slot = None
        return place, slot, pose[:3] if pose else None

    def recorded_place(self, part: str) -> tuple[str, str | None] | None:
        """Return a place whose own state records the part, and the variable or slot that records it.

        Gripping resources (they declare a held part) and bookkeeping do not say where a part rests, and two
        records of the part leave it unresolved.
        """
        records = []
        for jid, row in self.resources.items():
            variables = self.variables(jid)
            if "held_part" in variables:
                continue
            for name, declaration in variables.items():
                if name.startswith(("task_ctx.", "_")) or (declaration or {}).get("private"):
                    continue
                if row.get(name) == part:
                    records.append((self.name_of(jid), name))
                elif name.endswith("." + part) and row.get(name):
                    records.append((self.name_of(jid), row[name] if isinstance(row[name], str) else None))
        return records[0] if len(records) == 1 else None

    def rest_pose(self, part: str) -> list[float] | None:
        """Return the pose at which this check let an applied event leave a part at a pose place.

        Only the last applied event that took or put the part counts: a later one moved it again.
        """
        for turn in reversed(self.checks):
            applied = next((row for row in turn.get("results", [])
                            if row.get("candidate_index") == turn.get("applied_candidate_index")), None)
            rows = [row for row in (applied or {}).get("capability", []) if row.get("part") == part]
            if rows:
                return next((_pose_list(row["rest_pose"]) for row in rows if row.get("rest_pose")), None)
        return None

    def released_by(self, place: Any, part: str) -> bool:
        """Return whether a part's location names a gripping resource whose own held part says it does not hold it."""
        jid = next((key for key, rid in self.cell.rids.items() if place in (key, rid)), None)
        if jid is None or "held_part" not in self.variables(jid):
            return False
        known, held = self.value(jid, "held_part")
        return known and held != part

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
        # A gripping resource's own held part decides custody; for any other resource the part's location does.
        grips = held_known and "held_part" in self.variables(self.jid)
        holding = held_before == self.part if grips else location_before == self.jid
        held_after = self.end.get("held_part", held_before if held_known else None)
        place, slot, neighbor = self.destination()
        takes = held_after == self.part or place in (self.jid, self.rid)
        puts = place not in (None, self.jid, self.rid) and (neighbor is not None or place != location_before or holding)
        # Set down where no declared place names the spot, the part rests at the stated pose (a pose place):
        # released from the gripper, or taken from where it is and set down in one event.
        released = ("part_location" in self.end and place is None and neighbor is None and location_before is not None
                    or holding and "held_part" in self.end and held_after is None)
        rows = []
        if (takes or puts or released) and not holding:
            rows.append({"check": "pick", "part": self.part, **self.reach_part()})
        if puts:
            rows.append({"check": "place", "part": self.part,
                         **self.reach_place(place, slot, f"{_text(place)}" + (f" {slot}" if slot else ""))})
            if neighbor is not None:
                rows.append(self.capacity(neighbor, slot))
        elif released:
            rows.append({"check": "place", "part": self.part, **self.pose_place()})
        return rows

    def pose_place(self) -> dict:
        """Check a part released at the stated pose where no declared place names the spot (a pose place).

        The part leaves the gripper as the grasp the resource recorded at the failure holds it, while the
        resource still holds it from then; any other grasp is not recorded, so the release is unresolved.
        The part must rest right there, flat on recorded surfaces of one height: its bottom on their top
        faces, sunk at most by the recorded contact allowance and never above them (settling is not
        modelled), its recorded centre of mass over where it touches down, its footprint on them, where no
        declared place names the spot and no recorded object or part overlaps it. A mobile resource's pose
        places its base, not the part.
        """
        stated = self.end.get("pose")
        if self.mobile() or not isinstance(stated, dict):
            return {"status": NEEDS_CONTEXT, "reason": f"the event does not state where {self.part} goes"}
        if not all(isinstance(stated.get(axis), (int, float)) for axis in "xyz"):
            return {"status": NEEDS_CONTEXT, "reason": f"pose {_text(stated)} does not state x, y and z"}
        tool = self.stated_tool_pose(stated)
        if tool is None:
            return {"status": NEEDS_CONTEXT, "reason": f"pose {_text(stated)} has no orientation to check"}
        bounds = (self.cell.part_geometry.get(self.part) or {}).get("local_bounds")
        if not bounds:
            return {"status": NEEDS_CONTEXT, "reason": f"the capture records no geometry for {self.part}"}
        layer, depth = self.cell.contact_layer_m, self.contact_depth()
        if layer is None or depth is None:
            return {"status": NEEDS_CONTEXT, "reason": f"the capture records no contact surface layer or contact "
                                                       f"depth for {self.part}"}
        grasp = self.recorded_grasp()
        if grasp is None:
            return {"status": NEEDS_CONTEXT, "reason": f"the capture records no grasp of {self.part} by {self.rid} "
                                                       "that still holds"}
        rotation = Rotation.from_quat(tool[3:]) * Rotation.from_quat(grasp[3:])
        origin = np.add(tool[:3], Rotation.from_quat(tool[3:]).apply(grasp[:3]))
        return self.rest_at(f"{self.part} released at pose {_text(stated)}", origin, rotation, bounds, min(layer, depth))

    def support_under(self, where: str, box: list[list[float]], sink: float) -> tuple[dict | None, dict | None]:
        """Return the recorded surface right under a part's bottom, or the fact that it has none there.

        As the projected part support reads the physics record, a part may sink into a surface by the smaller
        of the contact layer and both collisions' contact depths (``sink`` holds the part's and the layer),
        and contact never bridges an air gap.
        """
        (bottom, top), center = box[2], [(box[0][0] + box[0][1]) / 2, (box[1][0] + box[1][1]) / 2]
        under = [{**surface, "sink": min(sink, surface["depth"])} for surface in self.surfaces()
                 if _footprint_contains(surface, center)]
        inside = [surface for surface in under
                  if surface["bottom"] < top and surface["top"] > bottom + surface["sink"] + _TOUCH_M]
        if inside:
            return None, {"status": INFEASIBLE, "reason": f"{where} would sit inside {inside[0]['id']}"}
        support = max((surface for surface in under if surface["top"] <= bottom + surface["sink"] + _TOUCH_M),
                      key=lambda surface: surface["top"], default=None)
        # A shape recorded only by its bounds (a mesh, a cylinder, a tilted box) between that surface and the
        # part may hold the part instead, or the part may sit inside it.
        floor = support["top"] if support else -math.inf
        unchecked = [collision["id"] for _, collision in self.static_collisions()
                     if _vertical_box(collision) is None and _bounds_contain(collision["bounds"], center)
                     and collision["bounds"][2][0] < top and collision["bounds"][2][1] > floor + _TOUCH_M]
        if unchecked:
            return None, {"status": NEEDS_CONTEXT, "reason": f"{where} would rest on or in the recorded bounds of "
                                                             f"{', '.join(unchecked)}, whose shapes are not checked"}
        if support is None:
            return None, {"status": INFEASIBLE, "reason": f"no recorded surface lies under {where}"}
        if support["top"] < bottom - _TOUCH_M:
            return None, {"status": INFEASIBLE, "reason": f"{where} would drop {(bottom - support['top']) * 1000:.1f} mm "
                                                         f"onto {support['id']}"}
        return support, None

    def rest_at(self, where: str, origin: np.ndarray, rotation: Rotation, bounds: list[list[float]],
                sink: float) -> dict:
        """Check that a part placed at a pose rests right there, flat on recorded surfaces, at a free spot."""
        box = _part_box(origin, rotation, bounds)
        (bottom, top), center = box[2], [(box[0][0] + box[0][1]) / 2, (box[1][0] + box[1][1]) / 2]
        support, refused = self.support_under(where, box, sink)
        if refused:
            return refused
        tipping = self.tips_over(origin, rotation, bounds, support["sink"])
        if tipping:
            return {"status": NEEDS_CONTEXT, "reason": f"{where} {tipping} on {support['id']}"}
        corners = [[x, y] for x in box[0] for y in box[1]]
        level = [surface for surface in self.surfaces() if abs(surface["top"] - support["top"]) <= support["sink"] + _TOUCH_M
                 and any(_footprint_contains(surface, point) for point in [center, *corners])]
        if not all(any(_footprint_contains(surface, corner) for surface in level) for corner in corners):
            return {"status": INFEASIBLE, "reason": f"{where} would rest past the edge of {support['id']}"}
        rest = [box[0], box[1], [support["top"], support["top"] + top - bottom]]
        place = self.place_named_at(rest, level)
        if place:
            return {"status": INFEASIBLE,
                    "reason": f"{where} would rest in {place}, which the event does not state as its location"}
        exact, bounded = self.overlaps(rest, {surface["id"] for surface in level})
        if exact:
            return {"status": INFEASIBLE, "reason": f"{where} would overlap {', '.join(exact)} where it rests"}
        if bounded:
            return {"status": NEEDS_CONTEXT, "reason": f"{where} would rest within the recorded bounds of "
                                                       f"{', '.join(bounded)}, whose shapes are not checked"}
        resting = np.add(origin, [0.0, 0.0, support["top"] - bottom])
        return {"status": FEASIBLE, "support": support["id"],
                "rest_pose": dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"),
                                      [*resting.tolist(), *rotation.as_quat().tolist()], strict=True))}

    def contact_depth(self) -> float | None:
        """Return the event's part's recorded collision contact depth, or None when the capture records none."""
        depths = [collision.get("contact_min_depth") for link in ((self.cell.bodies.get(self.part) or {}).get("links")
                                                                  or {}).values() for collision in link.get("collisions", [])]
        valid = [depth for depth in depths if isinstance(depth, (int, float)) and math.isfinite(depth) and depth >= 0]
        return min(valid) if valid and len(valid) == len(depths) else None

    def tips_over(self, origin: np.ndarray, rotation: Rotation, bounds: list[list[float]], sink: float) -> str | None:
        """Return why a part resting on its lowest points would not stay there, or None when it would.

        Its recorded extent touches the surface at the corners within the contact allowance of its lowest
        one; its recorded centre of mass must lie strictly inside them, as the static support model of the
        projected part support requires.
        """
        from cais_spade_llm.resources.recovery_geometry import _support_polygon

        body = self.cell.bodies.get(self.part) or {}
        link = next(iter((body.get("links") or {}).values()), {})
        pose, centre = _pose_list(body.get("pose")), link.get("center_of_mass")
        if not pose or len(pose) != 7 or not (isinstance(centre, list) and len(centre) == 3):
            return "has no recorded centre of mass, so its rest is not checked"
        local_centre = Rotation.from_quat(pose[3:]).inv().apply(np.subtract(centre, pose[:3]))
        points = rotation.apply([[x, y, z] for x in bounds[0] for y in bounds[1] for z in bounds[2]]) + origin
        contacts = [point.tolist() for point in points if point[2] <= points[:, 2].min() + sink + _TOUCH_M]
        try:
            _support_polygon(contacts, (rotation.apply(local_centre) + origin).tolist(), [0.0, 0.0, -1.0])
        except ValueError:
            return "would tip over, its recorded centre of mass outside where it touches down,"
        return None

    def recorded_grasp(self) -> list[float] | None:
        """Return the grasp the resource recorded on the event's part at the failure, while it holds it from then.

        Once an accepted event has taken or released the part, the grasp it is held by is no longer recorded.
        """
        attachment = (self.cell.resources.get(self.rid) or {}).get("attachment") or {}
        grasp = attachment.get("grasp_transform")
        if (attachment.get("part_name") != self.part or attachment.get("observed_attachment_complete") is not True
                or not (isinstance(grasp, list) and len(grasp) == 7) or self.value(self.jid, "held_part") != (True, self.part)):
            return None
        return None if any(self.moves_part(task, self.part) for task in self.accepted) else grasp

    def moves_part(self, task: dict, part: str) -> bool:
        """Return whether an accepted event took or released a part: who holds it, or where it is, changed."""
        start, end = task.get("expected_start_state") or {}, task.get("expected_end_state") or {}
        held = "held_part" in end and start.get("held_part") != end["held_part"] and part in (
            start.get("held_part"), end["held_part"])
        located = (task.get("part_name") == part and "part_location" in end
                   and self.name_of(start.get("part_location")) != self.name_of(end["part_location"]))
        handed = any((values or {}).get("held_part") == part
                     for values in (task.get("neighbor_end_state") or {}).values())
        return held or located or handed

    def static_collisions(self) -> list[tuple[str, dict]]:
        """Return every recorded collision shape of the scene's fixed models, with its model's name."""
        return [(name, collision) for name, model in self.cell.bodies.items() if model.get("static") is True
                for link in (model.get("links") or {}).values() for collision in link.get("collisions", [])
                if collision.get("bounds")]

    def surfaces(self) -> list[dict]:
        """Return the recorded fixed boxes with a horizontal top face, which can support a part."""
        return [{**surface, "model": name} for name, collision in self.static_collisions()
                if (surface := _vertical_box(collision)) is not None]

    def place_named_at(self, rest: list[list[float]], supports: list[dict]) -> str | None:
        """Return the declared place a resting part would be in: a recorded area it meets, or a resource's body."""
        x, y = (rest[0][0] + rest[0][1]) / 2, (rest[1][0] + rest[1][1]) / 2
        for name, region in self.cell.regions.items():
            if not region.get("bounds"):
                continue
            (x0, x1), (y0, y1), (z0, z1) = region["bounds"]
            if x0 <= x <= x1 and y0 <= y <= y1 and rest[2][0] < z1 and z0 < rest[2][1]:
                return name
        bodies = {(self.cell.scene.get(rid) or {}).get("gazebo_model") or rid: rid for rid in self.cell.rids.values()}
        return next((bodies[surface["model"]] for surface in supports if surface["model"] in bodies), None)

    def overlaps(self, rest: list[list[float]], supports: set[str]) -> tuple[list[str], list[str]]:
        """Return what overlaps a resting part: recorded boxes exactly, other shapes and parts by their bounds."""
        exact, bounded = [], []
        for _, collision in self.static_collisions():
            if collision["id"] in supports or not _boxes_overlap(rest, collision["bounds"]):
                continue
            box = _vertical_box(collision)
            if box is None:
                bounded.append(collision["id"])
            elif _footprints_overlap(box, rest):
                exact.append(collision["id"])
        for part in sorted(self.part_names - {self.part}):
            row = self.parts.get(part, {})
            bounds = (self.cell.part_geometry.get(part) or {}).get("local_bounds")
            _, pose = self.part_pose(part)
            if row.get("part_holder_resource_jid") or row.get("current_holder_resource_jid") or not bounds or not pose:
                continue
            orientation = pose[3:] if len(pose) == 7 else [0.0, 0.0, 0.0, 1.0]
            if _boxes_overlap(rest, _part_box(pose[:3], Rotation.from_quat(orientation), bounds)):
                bounded.append(part)
        return exact, bounded

    def reach_part(self) -> dict:
        place, slot, position = self.part_place()
        if position is not None:
            area = {"name": "observed pose", "bounds": [[position[0]] * 2, [position[1]] * 2, [position[2]] * 2]}
            return self.reach_area(area, f"{self.part}'s observed position")
        if place is None:
            return {"status": NEEDS_CONTEXT, "reason": f"the projected state does not record where {self.part} is"}
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


def _pose_list(pose: Any) -> list[float] | None:
    """Return a recorded or projected pose as [x, y, z] or [x, y, z, qx, qy, qz, qw], or None."""
    axes = ("x", "y", "z", "qx", "qy", "qz", "qw")
    if isinstance(pose, dict) and all(isinstance(pose.get(axis), (int, float)) for axis in axes[:3]):
        values = [pose.get(axis) for axis in axes]
        return values if all(isinstance(value, (int, float)) for value in values) else values[:3]
    if isinstance(pose, list) and len(pose) >= 3 and all(isinstance(value, (int, float)) for value in pose):
        return list(pose[:7]) if len(pose) >= 7 else list(pose[:3])
    return None


def _part_box(origin: Iterable[float], rotation: Rotation, bounds: list[list[float]]) -> list[list[float]]:
    """Return the world bounds of a part's recorded extent placed at a pose."""
    corners = rotation.apply([[x, y, z] for x in bounds[0] for y in bounds[1] for z in bounds[2]]) + np.asarray(origin)
    return [[float(corners[:, axis].min()), float(corners[:, axis].max())] for axis in range(3)]


def _bounds_contain(bounds: list[list[float]], point: list[float]) -> bool:
    return bounds[0][0] <= point[0] <= bounds[0][1] and bounds[1][0] <= point[1] <= bounds[1][1]


def _boxes_overlap(first: list[list[float]], second: list[list[float]]) -> bool:
    return all(first[axis][0] < second[axis][1] - _TOUCH_M and second[axis][0] < first[axis][1] - _TOUCH_M
               for axis in range(3))


def _vertical_box(collision: dict) -> dict | None:
    """Return a recorded box's horizontal footprint and its height range when one of its axes is vertical."""
    shape = collision.get("geometry") or {}
    if shape.get("type") != "box" or not collision.get("bounds"):
        return None
    axes = Rotation.from_quat(collision["pose"][3:]).apply(np.eye(3))
    vertical = next((index for index, axis in enumerate(axes) if abs(abs(axis[2]) - 1.0) < 1e-6), None)
    if vertical is None:
        return None
    level = [index for index in range(3) if index != vertical]
    return {"id": collision["id"], "center": list(collision["pose"][:2]), "axes": [axes[index][:2] for index in level],
            "half": [shape["size"][index] / 2 for index in level],
            "bottom": collision["bounds"][2][0], "top": collision["bounds"][2][1],
            # A depth the record lacks allows no sinking.
            "depth": collision["contact_min_depth"] if isinstance(collision.get("contact_min_depth"), (int, float)) else 0.0}


def _footprint_contains(box: dict, point: list[float]) -> bool:
    offset = np.subtract(point, box["center"])
    return all(abs(float(np.dot(offset, axis))) <= half + _TOUCH_M
               for axis, half in zip(box["axes"], box["half"], strict=True))


def _footprints_overlap(box: dict, rest: list[list[float]]) -> bool:
    """Return whether a box's footprint and a resting part's footprint overlap, by separating axes."""
    corners = [np.add(box["center"], sign_a * box["half"][0] * box["axes"][0] + sign_b * box["half"][1] * box["axes"][1])
               for sign_a in (-1, 1) for sign_b in (-1, 1)]
    square = [np.array([x, y]) for x in rest[0] for y in rest[1]]
    for axis in (*box["axes"], np.array([1.0, 0.0]), np.array([0.0, 1.0])):
        first, second = [float(np.dot(corner, axis)) for corner in corners], [float(np.dot(corner, axis)) for corner in square]
        if max(first) <= min(second) + _TOUCH_M or max(second) <= min(first) + _TOUCH_M:
            return False
    return True


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

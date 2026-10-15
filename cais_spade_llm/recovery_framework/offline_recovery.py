from __future__ import annotations

"""Recovery generation from saved facts with no equipment or agent transport."""

import asyncio
import json
import logging
import math
import os
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)
from cais_spade_llm.recovery_framework.recovery_observations import (
    RECORDS,
    offline_preparation_missing,
)
from cais_spade_llm.resources.recovery_feasibility import derive_primitive_union

logger = logging.getLogger(__name__)


class SavedRecoveryResource:
    """Expose only recorded observation methods and declared resource metadata."""

    def validate_recovery_primitive_composition(self, **kwargs) -> dict:
        """Use the same nonexecuting composition contract as the live RA."""
        return ResourceAgent.validate_recovery_primitive_composition(self, **kwargs)

    def __init__(self, jid: str, entry: dict, observations: dict, *, _copy_observations: bool = True) -> None:
        """Keep a detached resource record without constructing a controller.

        The factory shares its own read-only capture across detached resources.
        Direct callers retain a defensive copy; all observation accessors return
        copies and supplementary evidence is stored separately.
        """
        self.jid = jid
        self.agent_name = next((rid for rid, value in observations.get("jids", {}).items() if value == jid), jid)
        self.static_capabilities = deepcopy(entry.get("static_capabilities", {}))
        self._entry = deepcopy(entry)
        self._observations = deepcopy(observations) if _copy_observations else observations
        self._resource_evidence_supplement: dict | None = None
        self._scene_capability_calculation: dict | None = None
        evidence = entry.get('resource_evidence') or observations.get('resource_evidence', {}).get(jid, {})
        if evidence.get('environment_models') and entry.get('resource_type') != 'robot':
            from cais_spade_llm.resources.environment_recovery import environment_declared_capabilities

            for field, value in environment_declared_capabilities(evidence).items():
                if field in self.static_capabilities and recovery_validation_fingerprint(self.static_capabilities[field]) != recovery_validation_fingerprint(value):
                    raise ValueError(f'Conflicting recorded resource capabilities: {self.jid}.{field}')
                self.static_capabilities[field] = deepcopy(value)

    def prepare_current_scene_capabilities(self) -> dict:
        """Derive current cell capabilities for a new attempt, retaining captured declarations."""
        if self._entry.get("resource_type") != "robot":
            return {}
        from .workflow_execution import _robot_configuration
        from cais_spade_llm.recovery_framework import ROOT

        scene = self._observations.get("inputs", {}).get("scene", {})
        robots = [row for row in scene.get("robots", []) if row.get("resource_id") == self.agent_name]
        if len(robots) != 1:
            return {}
        robot = robots[0]
        template = json.loads((ROOT / "cais_spade_llm/initialization/resources/robot_ur5e.json").read_text())["ur5e"]["gazebo"]["static_capabilities"]
        inherited = deepcopy(template["gripper_reach"])
        inherited["origin_pose"] = dict(zip(("x", "y", "z"), robot["base_xyz"], strict=True))
        _, _, configured = _robot_configuration(scene, robot)
        before = deepcopy(self.static_capabilities)
        for field, old in (("gripper_reach", inherited), ("staging_areas", template["staging_areas"])):
            if field in before and before[field] != old:
                continue
            if field in configured:
                self.static_capabilities[field] = deepcopy(configured[field])
            else:
                self.static_capabilities.pop(field, None)
        self._scene_capability_calculation = {
            "kind": "calculated", "source": "recorded scene and workflow_execution._robot_configuration",
            "scene_fingerprint": recovery_validation_fingerprint(scene),
            "recorded_static_capabilities": before, "static_capabilities": deepcopy(self.static_capabilities),
            "resource_id": self.agent_name, "execution_authorized": False}
        return deepcopy(self._scene_capability_calculation)

    def install_recovery_resource_evidence(self, *, observations: dict, source_fingerprint: str) -> dict:
        """Retain compatible supplementary facts separately from the source capture."""
        from cais_spade_llm.recovery_framework.outline_gazebo_evidence import checkpoint_differences
        from cais_spade_llm.recovery_framework.recovery_observations import observation_fingerprint

        if observation_fingerprint(self._observations) != source_fingerprint:
            raise ValueError("Supplementary resource evidence belongs to another source capture")
        differences = checkpoint_differences(self._observations, observations)
        if differences:
            raise ValueError("Supplementary resource evidence changes the checkpoint: " + "; ".join(differences))
        evidence = (observations.get("resource_evidence") or {}).get(self.jid)
        if not isinstance(evidence, dict) or evidence.get("conflicts"):
            raise ValueError("Supplementary resource evidence is unavailable or conflicting")
        self._resource_evidence_supplement = deepcopy(evidence)
        return self.get_recovery_resource_evidence()

    def get_recovery_resource_evidence(self) -> dict:
        """Restore supported owner facts from this capture without changing it."""
        recorded = self._resource_evidence_supplement
        if recorded is None:
            recorded = self._entry.get("resource_evidence")
        if recorded is None:
            recorded = (self._observations.get("resource_evidence") or {}).get(self.jid)
        evidence = deepcopy(recorded) if isinstance(recorded, dict) else {}
        evidence.setdefault("resource_id", self.agent_name)
        if self._scene_capability_calculation:
            evidence["scene_capability_calculation"] = deepcopy(self._scene_capability_calculation)
        evidence.setdefault("resource_type", self._entry.get("resource_type"))
        if "execution_mode" in (self._observations.get("setup") or {}):
            evidence.setdefault("execution_mode", self._observations["setup"]["execution_mode"])
        if "inputs" not in evidence:
            evidence["inputs"] = deepcopy(self._observations.get("inputs") or {})
        evidence.setdefault("source", {"run_id": self._observations.get("run_id"),
                                       "captured_at": self._observations.get("captured_at"),
                                       "revision": self._observations.get("context_revision"), "kind": "saved_observations"})
        physical = self._observations.get("physical_evidence") or {}
        provenance = physical.get("provenance") or {}
        if (evidence.get("resource_type") == "robot" and self.agent_name in physical.get("resources", {})
                and provenance.get("run_id") == self._observations.get("run_id")
                and provenance.get("context_revision") == self._observations.get("context_revision")
                and bool(provenance.get("run_id")) and "context_revision" in provenance):
            from cais_spade_llm.recovery_framework.recovery_observations import reconcile_resource_snapshot

            evidence = reconcile_resource_snapshot(evidence, physical, resource_id=self.agent_name,
                run_id=self._observations["run_id"], revision=self._observations["context_revision"],
                inputs=self._observations.get("inputs") or {})
        sources = evidence.setdefault("sources", {})
        conflicts = evidence.setdefault("conflicts", [])
        recorded_snapshot = evidence.get("snapshot")
        snapshot = deepcopy(recorded_snapshot if isinstance(recorded_snapshot, dict)
                            else self._entry.get("recovery_snapshot") or {})
        # Legacy captures stored the measured quaternion outside the prepared
        # snapshot. Merge missing fields only when their XYZ observations agree.
        pose = (self._observations.get("robot_poses") or {}).get(self.agent_name)
        existing_pose = snapshot.get("current_pose")
        if (isinstance(pose, dict) and isinstance(existing_pose, dict)
                and any(key not in existing_pose for key in ("qx", "qy", "qz", "qw"))):
            compatible = all(type(pose.get(key)) in (int, float) and type(existing_pose.get(key)) in (int, float)
                             and math.isfinite(pose[key]) and math.isfinite(existing_pose[key])
                             and abs(pose[key] - existing_pose[key]) <= 1e-6 for key in ("x", "y", "z"))
            if compatible:
                for key in ("qx", "qy", "qz", "qw"):
                    if key not in existing_pose and key in pose:
                        existing_pose[key] = deepcopy(pose[key])
                        sources["snapshot/current_pose/" + key] = f"/robot_poses/{self.agent_name}/{key}"
            else:
                conflicts.append({"ref": f"/resources/{self.jid}/resource_evidence/snapshot/current_pose",
                                  "reason": "Saved robot pose conflicts with the prepared resource snapshot"})
        if "snapshot" not in evidence or isinstance(recorded_snapshot, dict):
            evidence["snapshot"] = snapshot
        state = evidence.setdefault("state", {})
        for name, field in (("_held_part", "held_part"), ("_position", "current_pose")):
            if name not in state and field in snapshot:
                state[name] = deepcopy(snapshot[field])
                sources["state/" + name] = f"/prepared_recovery_request/recovery_resources/{self.jid}/recovery_snapshot/{field}"
        held = snapshot.get("held_part")
        pickup = (self._observations.get("failure", {}).get("pickups") or {}).get(self.agent_name) or {}
        controller_observations = (pickup.get("observations", {}).get("controller_result") or {}).get("observations") or {}
        if held is not None and controller_observations.get("part_name") == held and "attachment" not in evidence:
            primitive_results = (controller_observations.get("simulation_execution") or {}).get("primitive_results") or []
            attachments = [(index, row.get("command_evidence", {}).get("payload_collision"))
                           for index, row in enumerate(primitive_results)
                           if row.get("primitive") == "grasp_part" and row.get("status") == "completed"
                           and row.get("parameters", {}).get("part_name") == held
                           and isinstance(row.get("command_evidence", {}).get("payload_collision"), dict)]
            if len(attachments) == 1:
                index, attachment = attachments[0]
                evidence["attachment"] = deepcopy(attachment)
                sources["attachment"] = (f"/failure/pickups/{self.agent_name}/observations/controller_result/observations/"
                                         f"simulation_execution/primitive_results/{index}/command_evidence/payload_collision")
        for field in ('unavailable_resources', 'recovery_permitted_resources'):
            if field in self._observations:
                evidence[field] = deepcopy(self._observations[field])
        _, restored_sources = self._recorded_recovery_snapshot()
        for field, paths in restored_sources.items():
            sources['recovery_snapshot/' + field] = paths
        return evidence

    def prepare_recovery_primitive_parameters(self, *, task: dict, recovery_des_model: dict,
                                              recovery_snapshot: dict, physical_input: dict) -> dict:
        """Calculate parameter evidence through the recorded resource's owner."""
        evidence = physical_input.get("resource_evidence") or {}
        if self._entry.get("resource_type") == "robot":
            from cais_spade_llm.resources.robot.recovery_capability import prepare_robot_primitive_parameters

            return prepare_robot_primitive_parameters(task=task, recovery_des_model=recovery_des_model,
                recovery_snapshot=recovery_snapshot, resource_evidence=evidence, resource_jid=self.jid,
                physical_evidence=physical_input.get("physical_evidence"))
        if self._entry.get('resource_type') == 'kmr':
            from cais_spade_llm.recovery_framework.kmr_recovery import prepare_kmr_primitive_parameters
            from cais_spade_llm.resources.environment_recovery import kmr_validation_evidence

            return prepare_kmr_primitive_parameters(task=task, recovery_des_model=recovery_des_model,
                recovery_snapshot=recovery_snapshot, resource_evidence=kmr_validation_evidence(
                    evidence, recovery_snapshot, physical_input.get('physical_evidence')), resource_jid=self.jid)
        return ResourceAgent.prepare_recovery_primitive_parameters(self, task=task,
            recovery_des_model=recovery_des_model, recovery_snapshot=recovery_snapshot, physical_input=physical_input)

    def project_recovery_outline_event(self, **kwargs) -> dict:
        """Project recorded event evidence without attaching a controller."""
        evidence = kwargs.get('physical_input', {}).get('resource_evidence') or self.get_recovery_resource_evidence()
        if evidence.get('environment_models') and self._entry.get('resource_type') != 'robot':
            from cais_spade_llm.resources.environment_recovery import project_environment_event

            kwargs['physical_input'] = {**kwargs.get('physical_input', {}), 'resource_evidence': evidence}
            return project_environment_event(**kwargs)
        return ResourceAgent.project_recovery_outline_event(self, **kwargs)

    def get_recovery_snapshot(self) -> dict:
        """Restore declared owner facts without changing the original capture."""
        snapshot, _ = self._recorded_recovery_snapshot()
        if self._scene_capability_calculation:
            for field in ("gripper_reach", "staging_areas"):
                snapshot.pop(field, None)
                if field in self.static_capabilities:
                    snapshot[field] = deepcopy(self.static_capabilities[field])
            from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_resource_adapter import (
                adapt_recovery_resource_snapshot,
            )

            snapshot = adapt_recovery_resource_snapshot(resource_jid=self.jid,
                resource_type=self._entry["resource_type"], snapshot=snapshot,
                modeled_state=self._entry.get("modeled_state", {}),
                state_variables=self._entry.get("recovery_des_model", {}).get("state_variables", {}))
        return snapshot

    def _recorded_recovery_snapshot(self) -> tuple[dict, dict]:
        snapshot = self._entry.get("recovery_snapshot")
        if not isinstance(snapshot, dict) or not snapshot:
            raise ValueError(f"NEEDS_CONTEXT: {self.jid}: recovery_snapshot")
        result, sources = deepcopy(snapshot), {}
        evidence_rows = []
        if self._resource_evidence_supplement is not None:
            evidence_rows.append(('supplementary_resource_evidence', self._resource_evidence_supplement))
        if isinstance(self._entry.get('resource_evidence'), dict):
            evidence_rows.append((f'/prepared_recovery_request/recovery_resources/{self.jid}/resource_evidence', self._entry['resource_evidence']))
        recorded = self._observations.get('resource_evidence', {}).get(self.jid)
        if isinstance(recorded, dict):
            evidence_rows.append((f'/resource_evidence/{self.jid}', recorded))
        saved_model = self._entry.get('recovery_des_model')
        declarations = saved_model.get('state_variables', {}) if isinstance(saved_model, dict) else {}
        fields = {field for field, row in declarations.items() if row.get('scope', 'resource') == 'resource'}
        rows = [(f'/prepared_recovery_request/recovery_resources/{self.jid}/recovery_snapshot', snapshot)]
        for path, evidence in evidence_rows:
            model = (evidence.get('environment_models', {}).get(self.agent_name, {})
                     if self._entry.get('resource_type') != 'robot' else {})
            for field, declaration in model.get('state_variables', {}).items():
                if declaration.get('scope', 'resource') != 'resource':
                    continue
                names = evidence.get('environment_products', {})
                if '{part_name}' in field:
                    fields.update(field.replace('{part_name}', name) for name in names)
                else:
                    fields.add(field)
            rows.extend([(path + '/snapshot', evidence.get('snapshot', {})),
                         (path + '/environment_valuation/' + self.agent_name,
                          evidence.get('environment_valuation', {}).get(self.agent_name, {}))])
        for field in sorted(fields):
            if field in {"active_job", "job_state", "selected_program"} and any(
                    evidence.get("snapshot_reconciliation_version") == 1
                    and not evidence.get("processing_observations")
                    and any(row.get("ref") == "processing_observations/" + self.agent_name
                            for row in evidence.get("missing", []) if isinstance(row, dict))
                    for _, evidence in evidence_rows):
                result.pop(field, None)
                continue
            values = [(path + '/' + field, row[field]) for path, row in rows if isinstance(row, dict) and field in row]
            if not values:
                continue
            first = recovery_validation_fingerprint(values[0][1])
            if any(recovery_validation_fingerprint(value) != first for _, value in values[1:]):
                raise ValueError(f'Conflicting recorded resource state: {self.jid}.{field}: ' + ', '.join(path for path, _ in values))
            result[field] = deepcopy(values[0][1])
            sources[field] = [path for path, _ in values]
        return result, sources

    def recovery_des_model(self, *, snapshot: dict | None = None) -> dict:
        """Return the recorded descriptor without adapting its symbols."""
        evidence = self.get_recovery_resource_evidence()
        if (snapshot is not None and self._entry.get('resource_type') != 'robot'
                and evidence.get('environment_models', {}).get(evidence.get('resource_id'))):
            from cais_spade_llm.resources.environment_recovery import environment_recovery_descriptor

            return environment_recovery_descriptor(evidence, snapshot, resource_jid=self.jid)
        model = self._entry.get("recovery_des_model")
        if not isinstance(model, dict) or not model.get("descriptor_fingerprint"):
            raise ValueError(f"NEEDS_CONTEXT: {self.jid}: recovery_des_model, descriptor_fingerprint")
        return deepcopy(model)

    def validate_recovery_primitive_support(
        self, *, task: dict, recovery_des_model: dict, recovery_snapshot: dict, physical_input: dict,
    ) -> dict:
        """Use recorded capability witnesses without constructing a live resource."""
        physical_input = dict(physical_input)
        if physical_input.get("primitive_support_semantics") == "successor_primitive_union":
            supplied = physical_input.get("resource_evidence")
            known = self.get_recovery_resource_evidence()
            if physical_input.get("physical_evidence", {}).get("endpoint_evidence_policy_version") in (1, 2):
                from .outline_evidence_policy import sanitize_validation_input

                checked = {**deepcopy(physical_input), "projected_snapshot": deepcopy(recovery_snapshot),
                           "resource_evidence": known}
                sanitize_validation_input(checked, self.jid)
                known = checked["resource_evidence"]
            if supplied is not None:
                def matches(recorded, proposed):
                    if isinstance(recorded, dict) and isinstance(proposed, dict):
                        return all(key in recorded and matches(recorded[key], value) for key, value in proposed.items())
                    return type(recorded) is type(proposed) and recorded == proposed

                if not isinstance(supplied, dict) or not matches(known, supplied):
                    return {"allowed": False, "feasibility_status": "NEEDS_CONTEXT",
                            "reason": "Resource evidence is not linked to this saved resource capture",
                            "context_requests": [f"/resources/{self.jid}/resource_evidence"]}
            # Supplementary legacy facts were verified against this immutable
            # capture above. An empty prepared record must not hide them.
            physical_input["resource_evidence"] = known
            witnesses = physical_input.get("parameter_witnesses") or []
            recorded = self._entry.get("parameter_witnesses") or []
            if not isinstance(witnesses, list) or any(not isinstance(row, dict) or row not in recorded for row in witnesses):
                return {"allowed": False, "feasibility_status": "NEEDS_CONTEXT",
                        "reason": "Parameter witnesses are not linked to this saved resource capture",
                        "context_requests": [f"/resources/{self.jid}/parameter_witnesses"]}
            witnesses = deepcopy(witnesses)
            for row in witnesses:
                row.setdefault("source", f"/resources/{self.jid}/parameter_witnesses")
                row["source_capture"] = {key: deepcopy(self._observations[key])
                                         for key in ("run_id", "captured_at", "context_revision") if key in self._observations}
                if "captured_at" in self._observations:
                    row.setdefault("captured_at", self._observations["captured_at"])
                if "context_revision" in self._observations:
                    row.setdefault("revision", self._observations["context_revision"])
            physical_input["parameter_witnesses"] = witnesses
        return ResourceAgent.validate_recovery_primitive_support(
            self, task=task, recovery_des_model=recovery_des_model,
            recovery_snapshot=recovery_snapshot, physical_input=physical_input)

    def check_recovery_primitive_feasibility(
        self, *, primitive: str, params: dict, task: dict, recovery_snapshot: dict,
        part_context: dict, grounded_action: dict, resource_evidence: dict | None = None,
        physical_evidence: dict | None = None,
    ) -> dict:
        """Check independent capabilities using only recorded declarations and facts."""
        evidence = {"evidence_source": "saved_observations", "source_run_id": self._observations.get("run_id"),
                    "captured_at": self._observations.get("captured_at"),
                    "primitive": primitive, "params": deepcopy(params), "command_sent": False,
                    "scope": "Capability evidence only; ordering and physical projection are not established."}

        def result(status: str, reason: str, **facts) -> dict:
            return {"allowed": status == "FEASIBLE", "feasibility_status": status, "reason": reason,
                    "evidence": {**evidence, **deepcopy(facts)}}

        catalog = (self._entry.get("recovery_des_model") or {}).get("primitive_catalog") or []
        entry = next((row for row in catalog if row.get("name") == primitive), None)
        if entry is None:
            return result("NEEDS_CONTEXT", "Primitive declaration is not recorded for this resource")
        if self._entry.get("resource_type") == "robot" and resource_evidence is not None:
            from cais_spade_llm.resources.robot.recovery_capability import check_robot_primitive_capability

            checked = check_robot_primitive_capability(primitive=primitive, params=params, task=task,
                recovery_snapshot=recovery_snapshot, resource_evidence=resource_evidence,
                static_capabilities=self.static_capabilities, primitive_catalog=catalog, resource_jid=self.jid,
                part_context=part_context, physical_evidence=physical_evidence)
            if checked is not None:
                return {**checked, "evidence": {**evidence, **checked.get("evidence", {})}}
        owner_evidence = resource_evidence if resource_evidence is not None else self.get_recovery_resource_evidence()
        if owner_evidence.get('environment_models') and self._entry.get('resource_type') != 'robot':
            from cais_spade_llm.resources.environment_recovery import check_environment_primitive_capability

            return check_environment_primitive_capability(primitive=primitive, params=params, task=task,
                recovery_snapshot=recovery_snapshot, resource_evidence=owner_evidence,
                part_context=part_context, physical_evidence=physical_evidence)
        if self._entry.get("resource_type") == "printer":
            from cais_spade_llm.agents.resource_agent.printing_agent import PrintingAgent

            owner = SimpleNamespace(_RECOVERY_PRIMITIVES=PrintingAgent._RECOVERY_PRIMITIVES)
            owner.check_recovery_physical_feasibility = lambda **kwargs: PrintingAgent.check_recovery_physical_feasibility(owner, **kwargs)
            checked = PrintingAgent.check_recovery_primitive_feasibility(
                owner, primitive=primitive, params=deepcopy(params), task=deepcopy(task),
                recovery_snapshot=deepcopy(recovery_snapshot), part_context=deepcopy(part_context),
                grounded_action=deepcopy(grounded_action))
            return {**checked, "evidence": {**evidence, **deepcopy(checked.get("evidence") or {})}}
        if primitive == "get_current_pose":
            pose = recovery_snapshot.get("current_pose")
            if (not isinstance(pose, dict) or any(isinstance(pose.get(axis), bool)
                    or not isinstance(pose.get(axis), (int, float)) or not math.isfinite(pose[axis])
                    for axis in ("x", "y", "z"))):
                return result("NEEDS_CONTEXT", "Recorded current_pose is unavailable")
            return result("FEASIBLE", "Recorded pose observation is available", current_pose=pose)
        if primitive == "detect_parts":
            part = params.get("part_name")
            context = part_context if part_context else (self._observations.get("part_tracker") or {}).get(part, {})
            pose = context.get("observed_pose")
            if (context.get("part_name") not in (None, part) or context.get("observed_pose_current") is False
                    or not isinstance(pose, dict) or any(isinstance(pose.get(axis), bool)
                    or not isinstance(pose.get(axis), (int, float)) or not math.isfinite(pose[axis])
                    for axis in ("x", "y", "z"))):
                return result("NEEDS_CONTEXT", "Requested part has no current recorded observed_pose")
            return result("FEASIBLE", "Recorded part observation is available", part_name=part, observed_pose=pose)
        if self._entry.get("resource_type") != "robot":
            return result("NEEDS_CONTEXT", "Recorded resource has no local evaluator for this primitive")
        if primitive == "move_to_named_pose":
            declarations = [row["named_poses"] for row in (recovery_snapshot, self.static_capabilities)
                            if "named_poses" in row]
            if (not declarations or any(not isinstance(row, (dict, list)) or not row
                    or any(not isinstance(name, str) for name in row) for row in declarations)
                    or any(row != declarations[0] for row in declarations[1:])):
                return result("NEEDS_CONTEXT", "Recorded named_poses declarations are missing or conflict")
            allowed = params.get("pose_name") in declarations[0]
            return result("FEASIBLE" if allowed else "INFEASIBLE",
                          "Checked the bound pose_name against recorded named_poses", named_poses=declarations[0])
        if primitive in {"move_cartesian", "move_pose", "move_relative"}:
            frames = [row["frame"] for row in (params, entry) if "frame" in row]
            if not frames or any(frame != "world" for frame in frames):
                return result("NEEDS_CONTEXT", "Primitive parameters require an explicitly recorded world frame")
            pose = {axis: params.get(axis) for axis in ("x", "y", "z")}
            if primitive == "move_relative":
                current = recovery_snapshot.get("current_pose") or {}
                frame = recovery_snapshot.get("current_pose_frame", current.get("frame"))
                if frame != "world":
                    return result("NEEDS_CONTEXT", "Relative capability requires the recorded current_pose frame")
                values = [current.get(axis) for axis in ("x", "y", "z")] + [params.get("d" + axis) for axis in ("x", "y", "z")]
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                       for value in values):
                    return result("NEEDS_CONTEXT", "Relative capability requires recorded current_pose and explicit dx, dy, dz")
                pose = {axis: current[axis] + params["d" + axis] for axis in ("x", "y", "z")}
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in pose.values()):
                return result("NEEDS_CONTEXT", "Cartesian capability requires explicit finite x, y, z")
            declarations = [row["gripper_reach"] for row in (recovery_snapshot, self.static_capabilities,
                            self._entry.get("recovery_adapter") or {}) if "gripper_reach" in row]
            if (not declarations or any(row != declarations[0] for row in declarations[1:])
                    or not isinstance(declarations[0], dict) or declarations[0].get("frame") != "world"):
                return result("NEEDS_CONTEXT", "Recorded gripper_reach declarations are missing or conflict")
            from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

            error = RobotAgent._recovery_pose_gripper_reach_error(pose, declarations[0])
            status = ("INFEASIBLE" if error.startswith("pose is outside gripper_reach:")
                      else "NEEDS_CONTEXT") if error else "FEASIBLE"
            return result(status, error or "Bound pose is inside recorded gripper_reach",
                          checked_pose=pose, pose_frame="world", gripper_reach=declarations[0])
        return result("NEEDS_CONTEXT", "This primitive requires capability evidence not available in the saved capture")

    def get_current_pose(self) -> dict:
        """Read a recorded pose, never query a robot or fabricate a pose."""
        snapshot = self.get_recovery_snapshot()
        pose = snapshot.get("current_pose")
        if not isinstance(pose, dict) or not all(axis in pose for axis in ("x", "y", "z")):
            raise ValueError(f"NEEDS_CONTEXT: no saved current_pose for {self.jid}")
        return {"success": True, "pose": deepcopy(pose), "evidence_source": "saved_observations",
                "captured_at": snapshot.get("current_pose_captured_at")}

    def detect_parts(self, part_name: str | None = None, **params) -> list[dict]:
        """Answer perception requests exclusively from recorded part poses."""
        part_name = part_name or params.get("filter_part_name")
        tracker = self._observations.get("part_tracker", {})
        names = [part_name] if part_name else list(tracker)
        result = []
        for name in names:
            row = tracker.get(name, {})
            pose = row.get("observed_pose")
            if isinstance(pose, dict) and all(axis in pose for axis in ("x", "y", "z")):
                result.append({"part_name": name, **deepcopy(pose),
                               "evidence_source": "saved_observations",
                               "captured_at": self._observations.get("captured_at")})
        if not result:
            raise ValueError(f"NEEDS_CONTEXT: no saved observed_pose for {part_name or 'requested parts'}")
        return result

    def check_recorded_recovery_feasibility(  # noqa: C901, PLR0912, PLR0915 - recorded facts only
        self, *, task: dict, recovery_snapshot: dict, physical_input: dict,
    ) -> dict:
        """Check saved robot custody and reach bounds without claiming a motion plan."""
        findings, checks = [], []
        result = {"findings": findings, "checks": checks, "evidence_source": "saved_observations",
                  "source_run_id": self._observations.get("run_id"),
                  "captured_at": self._observations.get("captured_at")}
        if self._entry.get("resource_type") != "robot":
            return result
        from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

        part = task.get("part_name") or physical_input.get("part_name")
        context = physical_input.get("part_context") or {}
        action = physical_input.get("grounded_action") or {}
        end = task.get("expected_end_state") or {}
        held = recovery_snapshot.get("held_part")
        holder = (context["part_holder_resource_jid"] if "part_holder_resource_jid" in context
                  else context.get("current_holder_resource_jid"))
        effects = (action.get("expected_effect") or {}).get("part") or {}
        affects_part = bool(part and (effects or any(field in end for field in (
            "part_state", "part_location", "part_holder_resource_jid")) or end.get("held_part") == part))
        current_location = context.get("part_location", context.get("current_location", context.get("location")))
        changes_custody = end.get("held_part") == part or end.get("part_holder_resource_jid") == self.jid
        relocates = "part_location" in end and end["part_location"] != current_location
        acquires = bool(part and (
            ((action.get("preconditions") or {}).get("part") or {}).get("requires_acquisition")
            or ((changes_custody or relocates) and held != part and holder != self.jid)))

        def reject(code: str, reason: str, *, missing: bool = False, **evidence) -> None:
            findings.append({"validation_category": "physical_feasibility", "constraint_owner": "resource",
                             "constraint_code": code, "resource_jid": self.jid, "part_name": part,
                             "feasibility_status": "NEEDS_CONTEXT" if missing else "INFEASIBLE",
                             "reason": reason, "evidence": {
                                 "evidence_source": "saved_observations", "source_run_id": result["source_run_id"],
                                 "captured_at": result["captured_at"], **evidence}})

        if acquires:
            if "held_part" not in recovery_snapshot:
                reject("resource_validation_unavailable", "NEEDS_CONTEXT: recorded held_part is missing", missing=True)
            elif held is not None and held != part:
                reject("holder_conflict", f"Resource holds '{held}' and cannot acquire '{part}'", held_part=held)
            if holder and holder != self.jid:
                reject("holder_conflict", f"Part '{part}' is held by '{holder}'", current_holder_resource_jid=holder)
            if held is None and recovery_snapshot.get("gripper_state") == "closed":
                reject("gripper_occupancy_conflict", "Gripper is closed without holding the requested part")

        sources = [("static_capabilities", self._entry.get("static_capabilities") or {}),
                   ("recovery_snapshot", self._entry.get("recovery_snapshot") or {}),
                   ("recovery_adapter", self._entry.get("recovery_adapter") or {})]

        def capability(field: str):
            fields = ("reachability", "reachable_locations") if field == "reachability" else (field,)
            declarations = [(f"{source}.{key}", values[key]) for source, values in sources
                            for key in fields if key in values]
            if not declarations:
                reject("resource_validation_unavailable", f"NEEDS_CONTEXT: recorded {field} is missing", missing=True, field=field)
                return None
            if any(value != declarations[0][1] for _, value in declarations[1:]):
                reject("resource_validation_unavailable", f"NEEDS_CONTEXT: recorded {field} declarations conflict",
                       missing=True, field=field, declarations=deepcopy(declarations))
                return None
            return declarations[0][1]

        def check_pose(pose: dict, *, role: str, frame: str | None, source: str, frame_source: str | None = None) -> None:
            bounds = capability("gripper_reach")
            if bounds is None:
                return
            if not isinstance(bounds, dict) or bounds.get("frame") != "world" or frame != "world":
                reject("resource_validation_unavailable", f"NEEDS_CONTEXT: {role} and gripper_reach require a recorded world frame",
                       missing=True, checked_pose=deepcopy(pose), pose_frame=frame, pose_source=source,
                       gripper_reach=deepcopy(bounds), frame_source=frame_source)
                return
            error = RobotAgent._recovery_pose_gripper_reach_error(pose, bounds)
            evidence = {"checked_pose": deepcopy(pose), "pose_frame": frame, "pose_source": source,
                        "gripper_reach": deepcopy(bounds), "role": role, "frame_source": frame_source}
            if error:
                missing = not error.startswith("pose is outside gripper_reach:")
                reject("resource_validation_unavailable" if missing else "workspace_unreachable",
                       ("NEEDS_CONTEXT: " if missing else "") + error, missing=missing, **evidence)
            else:
                checks.append({"status": "passed", "check": "gripper_reach", **evidence,
                               "scope": "Recorded bound only; trajectory and controller checks remain deferred."})

        if acquires:
            source_ref = (action.get("preconditions") or {}).get("source_ref") or {}
            pose = context.get("observed_pose") or source_ref.get("pose")
            if context.get("observed_pose_current") is False:
                reject("source_reference_unavailable", "NEEDS_CONTEXT: no current part pose is recorded after projected movement",
                       missing=True, observed_pose_current=False, projected_pose=deepcopy(context.get("projected_pose")))
            elif not isinstance(pose, dict) or not all(axis in pose for axis in ("x", "y", "z")):
                reject("source_reference_unavailable", "NEEDS_CONTEXT: acquisition requires a recorded part pose", missing=True)
            else:
                frame = (context.get("observed_pose_frame") or context.get("frame") or context.get("frame_id")
                         or pose.get("frame") or pose.get("frame_id") or source_ref.get("frame"))
                source = "part_context.observed_pose" if context.get("observed_pose") else "source_ref.pose"
                frame_source = None
                failure = self._observations.get("failure") or {}
                # This producer queries GetEntityState(reference_frame="world");
                # only its exact captured result establishes the legacy frame.
                if (not frame and self._observations.get("evidence_source") == "Gazebo"
                        and failure.get("part_name") == part and pose == failure.get("observed_drop_pose")):
                    frame = "world"
                    source = "failure.observed_drop_pose"
                    frame_source = "failure_effects._observe_part reference_frame=world"
                check_pose(pose, role="source", frame=frame, source=source, frame_source=frame_source)

        target = action.get("target") or {}
        destination = end.get("part_location") if affects_part else None
        destination = destination or target.get("target_location") or target.get("destination_location")
        if destination == self.jid or destination == holder:
            destination = None
        if affects_part and not destination and end.get("resource_location") != recovery_snapshot.get(
                "resource_location", recovery_snapshot.get("current_location")):
            destination = end.get("resource_location")
            named_poses = recovery_snapshot.get("named_poses") or self.static_capabilities.get("named_poses") or []
            if destination and isinstance(named_poses, (dict, list)) and destination in named_poses:
                checks.append({"status": "passed", "check": "named_poses", "named_pose": destination,
                               "named_poses": deepcopy(named_poses),
                               "scope": "Declared resource pose only; controller and trajectory checks remain deferred."})
                destination = None
        if affects_part and destination:
            reachability = capability("reachability")
            if reachability is not None:
                if not isinstance(reachability, list):
                    reject("resource_validation_unavailable", "NEEDS_CONTEXT: recorded reachability must be a list", missing=True)
                elif destination not in reachability:
                    reject("workspace_unreachable", f"Destination '{destination}' is outside declared reachability",
                           destination=destination, reachability=deepcopy(reachability))
                else:
                    checks.append({"status": "passed", "check": "reachability", "destination": destination,
                                   "reachability": deepcopy(reachability),
                                   "scope": "Declared destination only; placement geometry and trajectory remain deferred."})
        pose = target.get("pose") or target.get("slot_pose") or target.get("destination_pose")
        if isinstance(pose, dict):
            frame = (target.get("frame") or target.get("frame_id") or target.get("pose_frame")
                     or pose.get("frame") or pose.get("frame_id"))
            check_pose(pose, role="target", frame=frame, source="grounded_action.target")
        return result


class SavedRecoveryProduct:
    """Use the existing LLM request writer and pure saved-state validation."""

    def __init__(
        self, observations: dict, resources: list[SavedRecoveryResource], *,
        model: str | None = None, reasoning_effort: str | None = None,
    ) -> None:
        """Select attempt settings without changing the recorded observation settings."""
        from cais_spade_llm.agents.shared_information.llm_agent import (
            _ALLOWED_REASONING_EFFORTS,
            _default_model_name,
            _default_reasoning_effort,
            _normalize_reasoning_effort_for_model,
        )

        self.jid = observations.get("product_jid") or "assembly_board-v1@localhost"
        self.cca_jid = "central_controller@localhost"
        self.logger = logger
        self.instructions = ""
        self.precomputed_bundle = {}
        if model is not None and (not isinstance(model, str) or not model or model != model.strip()):
            raise ValueError("model must be a nonempty model identifier without surrounding whitespace")
        if reasoning_effort is not None and (
            not isinstance(reasoning_effort, str) or reasoning_effort not in _ALLOWED_REASONING_EFFORTS
        ):
            raise ValueError("reasoning_effort must be one of " + ", ".join(sorted(_ALLOWED_REASONING_EFFORTS)))
        model_env = ("CAIS_SPADE_LLM_MODEL", "OPENAI_MODEL", "CASE3_RECOVERY_MODEL")
        effort_env = ("CAIS_SPADE_REASONING_EFFORT", "OPENAI_REASONING_EFFORT")
        saved_model = observations.get("llm_model")
        saved_effort = observations.get("llm_reasoning_effort")
        self.model = model if model is not None else saved_model or _default_model_name(
            *model_env, fallback="gpt-5.4")
        effort = reasoning_effort if reasoning_effort is not None else saved_effort or _default_reasoning_effort(
            *effort_env, fallback="medium")
        self.reasoning_effort = (None if self.model == "gpt-4o"
                                 else _normalize_reasoning_effort_for_model(self.model, effort))
        if reasoning_effort is not None and self.reasoning_effort != reasoning_effort:
            raise ValueError(f"reasoning_effort={reasoning_effort!r} is unsupported for {self.model}")
        model_source = "saved" if saved_model else (
            "environment" if any(os.environ.get(name, "").strip() for name in model_env) else "default")
        environment_effort = _default_model_name(*effort_env, fallback="").lower()
        effort_source = "saved" if saved_effort else (
            "environment" if environment_effort in _ALLOWED_REASONING_EFFORTS else "default")
        self.llm_settings = {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "model_source": "override" if model is not None else model_source,
            "reasoning_effort_source": "override" if reasoning_effort is not None else effort_source,
            "model_override": model,
            "reasoning_effort_override": reasoning_effort,
            "saved_model": deepcopy(saved_model),
            "saved_reasoning_effort": deepcopy(saved_effort),
        }
        if self.model == "gpt-4o":
            self.llm_settings["reasoning_effort_source"] = "not_applicable"
        self.resources = {str(resource.jid): resource for resource in resources}
        self._last_structured_request = {}

    def restore_recovery_resource_evidence(self, session: dict, *, source_fingerprint: str) -> None:
        """Restore explicitly saved supplements after constructing detached owners."""
        for resource in self.resources.values():
            resource._resource_evidence_supplement = None
        captures = session.get("validation_resource_evidence_captures") or []
        if not isinstance(captures, list):
            raise ValueError("Saved supplementary validation captures must be a list")
        for record in captures:
            if record.get("source_observation_fingerprint") != source_fingerprint:
                raise ValueError("Saved supplementary validation evidence belongs to another source capture")
            observations = record.get("observations")
            if not isinstance(observations, dict):
                raise ValueError("Saved supplementary validation capture is unavailable")
            for jid in observations.get("resource_evidence", {}):
                if jid not in self.resources:
                    raise ValueError("Saved supplementary validation evidence identifies an unknown resource")
                self.resources[jid].install_recovery_resource_evidence(
                    observations=observations, source_fingerprint=source_fingerprint)
        physical = session.get('projected_physical_evidence') or {}
        if physical.get('additional_binding_records'):
            from .kmr_calculation import validate_calculation_records

            owner = self.resources.get(physical.get('jids', {}).get('KMR'))
            if owner is None:
                raise ValueError('Saved additional bindings have no recorded resource owner')
            validate_calculation_records(physical, session_id=session.get('session_id', ''),
                inputs=owner.get_recovery_resource_evidence().get('inputs', {}))
            model = owner.recovery_des_model(snapshot=owner.get_recovery_snapshot())
            if any(record.get('source_model_fingerprint') != recovery_validation_fingerprint(model)
                   for record in physical['additional_binding_records']):
                raise ValueError('Saved additional bindings use a changed resource model')

    async def ask_llm_structured(self, prompt: str, **kwargs) -> dict:
        """Capture actual provider requests through the existing implementation."""
        from cais_spade_llm.agents.shared_information.llm_agent import LlmAgent

        return await LlmAgent.ask_llm_structured(self, prompt, **kwargs)

    async def request_recovery_outline_physical_validation(
        self, *, resource_jid: str, payload: dict, timeout_s: float = 10.0,
    ) -> dict:
        """Check declared transitions; explicitly defer fresh physical evidence."""
        from cais_spade_llm.recovery_framework.outline_validation import uses_outline_evidence, validate_resource_outline_payload

        if uses_outline_evidence(payload):
            resource = self.resources.get(resource_jid)
            if resource is None:
                raise ValueError(f"NEEDS_CONTEXT: unknown resource {resource_jid}")
            return validate_resource_outline_payload(resource, payload, offline=True)
        resource = self.resources.get(resource_jid)
        if resource is None:
            raise ValueError(f"NEEDS_CONTEXT: unknown resource {resource_jid}")
        snapshot = resource.get_recovery_snapshot()
        model = resource.recovery_des_model()
        recorded_catalog = resource._entry.get("execution_primitive_catalog")
        if not isinstance(recorded_catalog, list):
            raise ValueError(f"NEEDS_CONTEXT: {resource_jid}: execution_primitive_catalog")
        catalog = {row["name"] for row in recorded_catalog}
        project_snapshot = ResourceAgent.recovery_physical_validation_snapshot
        if resource._entry.get("resource_type") == "robot":
            from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent

            project_snapshot = RobotAgent.recovery_physical_validation_snapshot
        results = []
        for candidate in payload.get("candidates", []):
            task, physical = candidate["task"], candidate.get("physical_input", {})
            projected = project_snapshot(
                live_snapshot=snapshot, physical_input=physical, recovery_des_model=model)
            transition = ResourceAgent.check_recovery_transition_feasibility(
                resource, task=task, recovery_snapshot=projected,
                part_context=physical.get("part_context", {}), recovery_des_model=model)
            support = derive_primitive_union(task=task, recovery_des_model=model, recovery_snapshot=projected)
            missing = [row["primitive"] for row in support["primitives"] if row["primitive"] not in catalog]
            if support["missing_support"] or not support["primitives"] or missing:
                transition = {"allowed": False, "constraint_code": "resource_validation_unavailable",
                              "reason": "NEEDS_CONTEXT: saved descriptor or primitive catalog lacks transition support"}
            findings = [] if transition.get("allowed") else [{
                "constraint_code": transition.get("constraint_code", "transition_feasibility_rejected"),
                "reason": transition.get("reason"), "validation_category": "transition_feasibility",
                "resource_jid": resource_jid, "evidence": transition.get("evidence", {}),
            }]
            transition["findings"] = deepcopy(findings)
            recorded = resource.check_recorded_recovery_feasibility(
                task=task, recovery_snapshot=projected, physical_input=physical,
            ) if transition.get("allowed") else {"findings": [], "checks": []}
            physical_findings = recorded["findings"]
            findings.extend(physical_findings)
            physical_status = "rejected" if physical_findings else "deferred"
            results.append({
                "candidate_index": candidate.get("candidate_index", 0), "event_id": candidate.get("event_id", ""),
                "allowed": False, "planning_only": True, "findings": findings,
                "transition_feasibility": transition,
                "physical_feasibility": {
                    "allowed": False, "status": physical_status, "planning_only": True,
                    "reason": (physical_findings[0]["reason"] if physical_findings else
                               "Fresh controller and trajectory checks require Test in Gazebo"),
                    "findings": physical_findings, "recorded_checks": recorded,
                    "primitive_support": {**support, "status": "deferred"},
                },
            })
        return {"validator_jid": resource_jid, "evidence_source": "saved_observations",
                "snapshot": snapshot, "snapshot_fingerprint": recovery_validation_fingerprint(snapshot),
                "recovery_des_model": model, "recovery_des_model_fingerprint": model["descriptor_fingerprint"],
                "results": results}

    async def request_recovery_outline_safety_validation(self, **kwargs) -> dict:
        """Run detached CCA evaluation for new attempts, retaining legacy policy."""
        from cais_spade_llm.recovery_framework.outline_validation import uses_outline_evidence, validate_safety_outline_payload

        payload = kwargs.get("payload") or {}
        if uses_outline_evidence(payload):
            return validate_safety_outline_payload(payload, offline=True)
        raise ValueError("Formal safety checks are skipped in this offline example")


def create_offline_recovery(
    observations: dict, *, outline_mode: str, primitive_mode: str,
    model: str | None = None, reasoning_effort: str | None = None,
    source_observation_path: str | None = None,
):
    """Create a Manual session with optional attempt-specific LLM settings.

    Explicit model and reasoning effort override saved settings independently.
    Omitted values retain saved, environment, then existing default precedence.
    The mode arguments remain accepted for existing callers; Offline is Manual.
    """
    from cais_spade_llm.recovery_framework.part_slippage_recovery import PartSlippageRecovery

    from cais_spade_llm.ui.recovery_setup import FAILURE_SCENARIOS

    if observations.get("scenario") not in FAILURE_SCENARIOS:
        raise ValueError("Unknown saved failure scenario")
    prepared = observations.get("prepared_recovery_request") or {}
    missing = offline_preparation_missing(observations)
    if missing:
        raise ValueError("NEEDS_CONTEXT: " + ", ".join(missing))
    # The complete capture includes mesh geometry and a full planning scene.
    # Keep one factory-owned copy instead of duplicating it for every resource.
    observations = deepcopy(observations)
    prepared = observations["prepared_recovery_request"]
    resources = [SavedRecoveryResource(jid, entry, observations, _copy_observations=False)
                 for jid, entry in prepared["recovery_resources"].items()]
    product = SavedRecoveryProduct(observations, resources, model=model, reasoning_effort=reasoning_effort)
    product.process_planner = ProcessPlanner(product, resources)
    product.process_planner.nodes = deepcopy(observations.get("pending_nodes", []))
    product.part_tracker = deepcopy(observations.get("part_tracker", {}))
    setup = deepcopy(observations.get("setup") or {})
    setup.update(execution_mode="simulation", diagnostic_cca_bypass=True,
                 runtime_recovery_validation_policy="no_validation", selected_safety_file="__NONE__",
                 runtime_recovery_mode="manual", runtime_recovery_primitive_mode="manual")
    setup["failure_scenario"] = {**setup.get("failure_scenario", {}), "scenario": observations["scenario"]}
    setup["outline_only"] = False
    runtime = SimpleNamespace(
        diagnostic_cca_bypass=True, setup=setup, path=RECORDS.parent, agent=product,
        context=SimpleNamespace(run_id=f"offline_{uuid4().hex}", revision=observations.get("context_revision", 0)),
        nominal_idle=asyncio.Event(), queue_save=lambda: None,
        source_observation_path=source_observation_path,
    )
    runtime.nominal_idle.set()
    recovery = PartSlippageRecovery(runtime, observations=observations)
    runtime.part_slippage_recovery = recovery
    product.environment_runtime = runtime
    recovery._publish(llm_model=product.model, llm_reasoning_effort=product.reasoning_effort,
                      llm_settings=product.llm_settings)
    return recovery

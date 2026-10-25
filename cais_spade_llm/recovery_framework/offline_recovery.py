from __future__ import annotations

"""Recovery generation from saved facts with no equipment or agent transport."""

import logging
import math
import os
from copy import deepcopy
from types import SimpleNamespace

from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)

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

            # The owner's recorded kinematics live in the capture's physical evidence.
            if not (physical_evidence or {}).get("resource_models"):
                physical_evidence = self._observations.get("physical_evidence") or physical_evidence
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
            from cais_spade_llm.resources.robot.recovery_capability import _cartesian_joint_witness

            # Reach is the owner's recorded kinematics; without a stated orientation the tool keeps its own.
            quaternion = ("qx", "qy", "qz", "qw")
            current = recovery_snapshot.get("current_pose") or {}
            orientation = params if all(axis in params for axis in quaternion) else current
            pose.update({axis: orientation[axis] for axis in quaternion if axis in orientation})
            checked = _cartesian_joint_witness(pose=pose, resource_jid=self.jid, recovery_snapshot=recovery_snapshot,
                resource_evidence=self.get_recovery_resource_evidence(),
                physical_evidence=self._observations.get("physical_evidence") or {})
            return {**checked, "evidence": {**evidence, **checked.get("evidence", {})}}
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


    async def ask_llm_structured(self, prompt: str, **kwargs) -> dict:
        """Capture actual provider requests through the existing implementation."""
        from cais_spade_llm.agents.shared_information.llm_agent import LlmAgent

        return await LlmAgent.ask_llm_structured(self, prompt, **kwargs)



"""High-level product agent that reads build instructions and instructs a resource agent."""

from __future__ import annotations

import asyncio
import json
import os, uuid
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message
from spade.template import Template

from agents.shared_information.llm_agent import LlmAgent
from agents.intelligent_product.process_planner import ProcessPlanner
from resources.sensor.camera_module import CameraModule


class ProductAgent(LlmAgent):
    """
    SPADE ProductAgent
    - Reads a product specification (optional) and sends a task to ResourceAgents.
    - Receives ACKs from Resource/Robot agents and logs status.
    - Keeps the PA simple: RA is the single broker that asks the LLM with tools.
    """

    agent_role = "product"  # Registered role so the shared LLM base class can fetch the right prompts.

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str,
        resource_jids: Optional[Iterable[str]] = None,
        resource_agents: Optional[Iterable[Any]] = None,
        product_specification_file: Optional[str] = None,
        product_geometry_file: Optional[str] = None,
        safety_file: Optional[str] = None,
        instruction_override: Optional[str] = None,
        cca_jid: Optional[str] = None,
        camera: Optional["CameraModule"] = None,
        replan_mode: str = "llm",
        precomputed_bundle: Optional[Dict[str, Any]] = None,
        **kw,
    ) -> None:
        """
        :param resource_jids: List of RA JIDs to target (first is used).
        :param product_specification_file: Path to spec text (utf-8). Optional.
        :param instruction_override: If provided, this text is used instead of reading a file.
        :param replan_mode: Online replanning strategy — "llm" (prompt-level guidance, default).
        """
        super().__init__(jid, password, name=name, agent_role="product", **kw)

        self.cca_jid = cca_jid or "cca@localhost"

        # Resource agents are the downstream executors; keep them in order and avoid mutating caller lists.
        self.resource_jids = list(resource_jids or [])
        self._resource_agent_refs = list(resource_agents or [])
        self.product_specification_file = product_specification_file
        self.product_geometry_file = product_geometry_file
        self.product_geometry: Dict[str, Any] = self._load_product_geometry(product_geometry_file)
        self.safety_file = safety_file
        # Manual instruction text provided at runtime overrides any file read.
        self.instruction_override = instruction_override

        # Cache safety text for use during replanning
        self.safety_text: str = ""

        self.replan_mode = replan_mode
        self.precomputed_bundle: dict[str, Any] = dict(precomputed_bundle or {})

        # Planner scaffolding
        base_plan_dir = Path("cais_spade_llm/monitor/plan")
        base_state_dir = Path("cais_spade_llm/monitor/state")
        # For now: requirements file (NL → structured requirements)
        self.structured_requirements_path = base_plan_dir / f"{name}_requirements.json"
        # Reserved for later: full DAG task plan (requirements → task graph)
        self.plan_path = base_plan_dir / f"{name}_plan.json"
        self.global_fsa_path = base_plan_dir / f"{name}_global_fsa.json"
        # Runtime snapshots (overwritten each step)
        self.product_state_path = base_state_dir / f"{name}_product_state.json"
        self.resource_state_path = base_state_dir / f"{name}_resource_state.json"

        planner_resources = self._match_resource_objects(
            self._resource_agent_refs, self.resource_jids
        )
        self.process_planner = ProcessPlanner(self, planner_resources)
        
        #keep resolved resource agents on the ProductAgent for caps overview
        self.resource_agents = planner_resources

        # Simple in-memory map of task_id -> latest status string so UI/debug tooling can query progress.
        self.task_states: dict[str, str] = {}

        # Sensor: camera module for post-placement verification
        self.camera = camera if camera is not None else CameraModule()

        # Runtime tracking for replanning context (PRODUCT STATE ONLY)
        self.part_tracker: dict[str, dict[str, Any]] = {}  # part_name -> {location, state, last_task}
        self.execution_timeline: list[dict[str, Any]] = []  # [{timestamp, task_id, status, ...}]
        # NOTE: Robot states are queried directly from ResourceAgents, not cached here
        self._plan_result_inbox_registered = False
        self._runtime_repair_inflight = False
        self._runtime_repair_fail_streak = 0
        self._runtime_repair_max_attempts = 3

        self.logger.info(f"ProductAgent '{name}' initialized.")

    # ------------------------------------------------------------------ #
    # Persistence helper
    # ------------------------------------------------------------------ #
    def _build_plan_validation_payload(self):
        """Package plan + FSA for offline safety validation by the CCA."""
        fsa = self.process_planner.global_fsa
        nodes = self.process_planner.nodes

        if fsa is None:
            raise RuntimeError("Global FSA is None. Did you call save_global_fsa()?")

        return {
            "fsa": fsa,                      # <-- upload FSA here
            "product_jid": str(self.jid),
            "plan": {"nodes": nodes},
        }



    def _persist_plan_snapshot(self) -> None:
        """Persist the current process planner graph (DAG nodes only) to disk."""
        if not self.plan_path:
            return

        try:
            import json

            self.plan_path.parent.mkdir(parents=True, exist_ok=True)
            with self.plan_path.open("w", encoding="utf-8") as f:
                json.dump({"nodes": self.process_planner.nodes}, f, indent=2)

            self.logger.info(f"[Product] Saved plan to {self.plan_path.resolve()}")
        except Exception:
            self.logger.exception("[Product] Failed to persist plan snapshot.")

    def _persist_product_state(self) -> None:
        """Persist product state (part tracker, execution timeline) to disk."""
        if not self.product_state_path:
            return

        try:
            from datetime import datetime, timezone
            import json

            payload = {
                "part_tracker": self.part_tracker,
                "execution_timeline": self.execution_timeline,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.product_state_path.parent.mkdir(parents=True, exist_ok=True)
            with self.product_state_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            self.logger.info(f"[Product] Saved product state to {self.product_state_path.resolve()}")
        except Exception:
            self.logger.exception("[Product] Failed to persist product state.")

    def _persist_resource_state(self) -> None:
        """Persist each resource agent's current state snapshot to disk."""
        if not self.resource_state_path:
            return

        try:
            import json

            state = {str(ra.jid): ra._snapshot_state() for ra in self.resource_agents}
            self.resource_state_path.parent.mkdir(parents=True, exist_ok=True)
            with self.resource_state_path.open("w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            self.logger.info(f"[Product] Saved resource state to {self.resource_state_path.resolve()}")
        except Exception:
            self.logger.exception("[Product] Failed to persist resource state.")


    def _get_part_transition(self, function_name: str) -> Dict[str, Any]:
        """Look up the part_transition map for a function from the shared tools catalogue."""
        self.__class__._load_shared_tools_catalogue()
        row = LlmAgent._TOOLS_BY_FUNC.get(function_name, {})
        return row.get("part_transition", {})

    def _apply_part_tracker_update(
        self,
        part_name: str,
        function_name: str,
        status: str,
        params: Dict[str, Any],
        resource_jid: str,
        task_id: str,
    ) -> None:
        """Generic interpreter: applies the part_transition declared in each function's docstring."""
        transition_map = self._get_part_transition(function_name)
        if not transition_map:
            return  # function has no declared part_transition — nothing to track

        # Most-specific match wins: "failed:misplaced" → "failed" → nothing
        transition = transition_map.get(status) or transition_map.get(status.split(":")[0])
        if not transition:
            return

        self.part_tracker.setdefault(part_name, {"state": "unknown", "location": None})
        entry: Dict[str, Any] = {"state": transition["state"]}

        if "location_template" in transition:
            entry["location"] = transition["location_template"].format(
                resource_jid=resource_jid,
            )

        if "location_param" in transition:
            entry["location"] = params.get(transition["location_param"])

        exec_mode = str(os.environ.get("EXECUTION_MODE", "dry_run")).strip().lower()
        perception_backend = str(os.environ.get("PERCEPTION_BACKEND", "none")).strip().lower()

        if transition.get("verify_camera"):
            if exec_mode == "dry_run" or perception_backend in {"", "none"}:
                entry["state"] = "assembled"
                entry["camera_verification"] = "skipped_dry_run"
            elif perception_backend == "yolo":
                # Placeholder for future physical-camera verification.
                entry["state"] = "assembled"
                entry["camera_verification"] = "todo_yolo_placeholder"
            else:
                position = self.camera.observe(part_name)
                if position is not None:
                    entry["state"] = "assembled"
                    entry["position"] = position
                    entry["camera_verification"] = "detected"
                else:
                    entry["state"] = "untracked"
                    entry["observation_required"] = True
                    entry["camera_verification"] = "not_detected"
                    self.logger.error(
                        "[Product] %s is untracked after placement; camera backend '%s' could not locate it.",
                        part_name,
                        perception_backend or "unknown",
                    )

        if transition.get("camera_locate"):
            last_known = (
                params.get(transition["last_known_param"])
                if "last_known_param" in transition else None
            )
            entry["location"] = None
            if last_known:
                entry["last_known_location"] = last_known

            if exec_mode == "dry_run" or perception_backend in {"", "none", "yolo"}:
                entry["state"] = "untracked"
                entry["observation_required"] = True
                entry["camera_verification"] = "unavailable"
            else:
                position = self.camera.observe(part_name)
                if position is not None:
                    entry["state"] = "misplaced"
                    entry["position"] = position
                    entry["camera_verification"] = "detected"
                else:
                    entry["state"] = "untracked"
                    entry["observation_required"] = True
                    entry["camera_verification"] = "not_detected"
                    self.logger.error(
                        "[Product] %s is untracked; camera backend '%s' could not locate it.",
                        part_name,
                        perception_backend or "unknown",
                    )

        if transition.get("observation_required"):
            entry["observation_required"] = True
            entry["location"] = None
            if "last_known_param" in transition:
                entry["last_known_location"] = params.get(transition["last_known_param"])
            elif "last_known_template" in transition:
                entry["last_known_location"] = transition["last_known_template"].format(resource_jid=resource_jid)

        if status == "completed":
            entry["last_successful_task"] = task_id

        self.part_tracker[part_name].update(entry)



    def _restore_product_state(self) -> None:
        """Restore product state (part tracker, timeline) and plan nodes from disk."""
        import json

        # --- Restore product state ---
        # Prefer the dedicated product_state file; fall back to legacy plan.json key.
        product_state: dict = {}
        if self.product_state_path and self.product_state_path.exists():
            try:
                with self.product_state_path.open("r", encoding="utf-8") as f:
                    product_state = json.load(f)
            except Exception:
                self.logger.exception("[Product] Failed to read product state file.")
        elif self.plan_path and self.plan_path.exists():
            try:
                with self.plan_path.open("r", encoding="utf-8") as f:
                    product_state = json.load(f).get("product_state", {})
            except Exception:
                self.logger.exception("[Product] Failed to read legacy product state from plan file.")

        if product_state:
            self.part_tracker = product_state.get("part_tracker", {})
            self.execution_timeline = product_state.get("execution_timeline", [])
            self.logger.info(
                f"[Product] Restored product state: {len(self.part_tracker)} parts, "
                f"{len(self.execution_timeline)} timeline events"
            )

        # --- Restore plan nodes ---
        if self.plan_path and self.plan_path.exists():
            try:
                with self.plan_path.open("r", encoding="utf-8") as f:
                    nodes = json.load(f).get("nodes", [])
                if nodes:
                    self.process_planner.nodes = nodes
                    self.logger.info(f"[Product] Restored {len(nodes)} plan nodes")
            except Exception:
                self.logger.exception("[Product] Failed to restore plan nodes.")

    # --------------------------------------------------------------------- #
    # SPADE lifecycle
    # --------------------------------------------------------------------- #

    async def setup(self):
        await super().setup()

        # Kickoff behaviour (runs once) to build the plan.
        # Template ensures _Kickoff only receives plan_safety_result messages
        # and does not steal ACKs or other messages from the queue.
        t_kickoff = Template()
        t_kickoff.set_metadata("type", "plan_safety_result")
        self.add_behaviour(self._Kickoff(), t_kickoff)

        # ACK inbox
        t_ack = Template()
        t_ack.set_metadata("type", "ack")
        self.add_behaviour(self._AckInbox(), t_ack)

        # Replan request inbox (from CCA)
        t_replan = Template()
        t_replan.set_metadata("type", "replan_request")
        self.add_behaviour(self._ReplanInbox(), t_replan)

        # Plan executor (runs cycles, dispatches DAG tasks)
        # self.add_behaviour(self._PlanExecutor())

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _read_safety_text(self) -> str:
        """
        Read the NL safety file if provided. Returns empty string if missing.
        """
        if not self.safety_file:
            return ""
        
        try:
            p = Path(self.safety_file)
            if p.exists():
                txt = p.read_text(encoding="utf-8").strip()
                self.logger.info(f"[Product] Loaded safety constraints from {p}")
                return txt
            else:
                self.logger.warning(f"[Product] Safety file path provided but not found: {p}")
        except Exception as e:
            self.logger.exception(f"[Product] Failed to read safety file: {e}")
        
        return ""
    
    def _read_spec_text(self) -> Optional[str]:
        """Return the instruction text: prefer override, else read file, else None."""
        if self.instruction_override:
            # Honour explicit overrides so tests and manual runs can inject free-form prompts.
            txt = self.instruction_override.strip()
            if txt:
                return txt

        if self.product_specification_file:
            try:
                cwd = os.getcwd()
                self.logger.info(f"[Product] Current working directory: {cwd}")
                p = Path(self.product_specification_file)
                self.logger.info(f"[Product] Attempting to open: {p.resolve()}")
                txt = p.read_text(encoding="utf-8").strip()
                if txt:
                    return txt
                self.logger.warning(f"[Product] Spec file is empty: {p}")
            except Exception as e:
                self.logger.exception(f"[Product] Failed to read spec: {e}")

        return None

    def _load_product_geometry(self, geometry_file: Optional[str]) -> Dict[str, Any]:
        """Load product geometry JSON, selecting the environment block (gazebo/real)."""
        if not geometry_file:
            return {}
        env = os.environ.get("ROBOT_ENV", "gazebo").strip().lower()
        try:
            p = Path(geometry_file)
            if not p.exists():
                self.logger.warning("[Product] Geometry file not found: %s", p)
                return {}
            raw = json.loads(p.read_text(encoding="utf-8"))
            geo = raw.get(env, {})
            if geo:
                self.logger.info(
                    "[Product] Loaded geometry for env='%s' from %s", env, p,
                )
                return geo

            self.logger.warning(
                "[Product] Geometry file %s has no usable '%s' block.", p, env
            )
            return {}
        except Exception:
            self.logger.exception("[Product] Failed to load geometry from %s", geometry_file)
            return {}

    def _geometry_for_part(self, part_name: str) -> Dict[str, Any]:
        """Extract placement geometry for a single part from loaded product geometry."""
        if not self.product_geometry:
            return {}
        board = self.product_geometry.get("assembly_board", {})
        parts = self.product_geometry.get("parts", {})
        slot_xy = board.get("slots", {}).get(part_name)
        if slot_xy is None:
            return {}
        return {
            "slot_xy": slot_xy,
            "part_height_m": parts.get("heights_m", {}).get(part_name),
            "model_name": parts.get("model_map", {}).get(part_name),
            "slot_floor_z_m": board.get("slot_floor_z_m"),
            "board_center": board.get("center", {}),
        }

    def _compose_task_msg(
        self,
        *,
        to: str,
        task_id: str,
        instruction: str | dict,
        phase_id: str | None = None,
        protocol: str = "plan/1.0",
    ) -> Message:
        """Create the structured SPADE Message that drives resource-agent task execution."""
        # Build a JSON payload with the instruction plus optional phase metadata.
        body = {"task_id": task_id, "instruction": instruction}
        if phase_id:
            body["phase_id"] = phase_id
        msg = Message(to=to)
        # Metadata drives the SPADE routing/filtering logic downstream.
        msg.set_metadata("type", "task")
        msg.set_metadata("protocol", protocol)
        msg.body = json.dumps(body)
        return msg

    def _build_product_state(self) -> Dict[str, Any]:
        """
        Build product-specific state for replanning context.

        NOTE: This only includes product/part state. System coordination state
        (robot states, running tasks, FSA states) is provided by CentralControllerAgent.

        Includes:
        - Part locations and states
        - Execution timeline
        - Requirements progress
        """
        requirements_status: Dict[str, Dict[str, Any]] = {}
        task_nodes = [n for n in self.process_planner.nodes if n.get("type") == "task"]

        # Seed from explicit requirement nodes when available.
        for node in self.process_planner.nodes:
            if node.get("type") != "requirement":
                continue
            req_id = node.get("id")
            if not req_id:
                continue
            requirements_status[str(req_id)] = {
                "goal": node.get("description") or node.get("raw_text", ""),
                "status": "unknown",
                "completion": 0,
            }

        # Ensure every requirement_id referenced by tasks is represented.
        req_ids_from_tasks = {
            str(n.get("requirement_id"))
            for n in task_nodes
            if n.get("requirement_id")
        }
        for req_id in req_ids_from_tasks:
            requirements_status.setdefault(
                req_id,
                {"goal": "", "status": "unknown", "completion": 0},
            )

        # Aggregate per-requirement progress from task statuses.
        for req_id in req_ids_from_tasks:
            req_tasks = [n for n in task_nodes if str(n.get("requirement_id")) == req_id]
            total_tasks = len(req_tasks)
            completed_tasks = sum(1 for n in req_tasks if n.get("status") == "completed")
            has_failed = any(
                isinstance(n.get("status"), str) and n.get("status", "").startswith("failed")
                for n in req_tasks
            )
            all_completed = bool(req_tasks) and all(n.get("status") == "completed" for n in req_tasks)

            if total_tasks > 0:
                requirements_status[req_id]["completion"] = int((completed_tasks / total_tasks) * 100)

            if has_failed:
                requirements_status[req_id]["status"] = "failed"
            elif all_completed:
                requirements_status[req_id]["status"] = "completed"
            elif completed_tasks > 0:
                requirements_status[req_id]["status"] = "in_progress"
            else:
                requirements_status[req_id]["status"] = "pending"

        final_timeline = [
            e for e in self.execution_timeline
            if e.get("status", "") in {"completed", "blocked"}
            or str(e.get("status", "")).startswith("failed:")
        ]
        return {
            "parts": dict(self.part_tracker),
            "execution_timeline": final_timeline,
            "requirements_status": requirements_status,
        }

    def _extract_requirement_text(self) -> Optional[str]:
        """
        Load requirement snippets from either an override path or the default
        specification/products/requirements/<product>.txt file.
        """
        candidates = [Path(self.product_specification_file)]
        for req_path in candidates:
            try:
                if not req_path.exists():
                    continue
                txt = req_path.read_text(encoding="utf-8").strip()
                if txt:
                    self.logger.info(f"[Product] Using requirement file: {req_path.resolve()}")
                    return txt
                self.logger.warning(f"[Product] Requirement file empty: {req_path}")
            except Exception as exc:
                self.logger.exception(
                    "[Product] Failed to read requirements from %s: %s",
                    req_path,
                    exc,
                )
        return None

    def _ensure_plan_result_inbox(self) -> None:
        """Register runtime plan_safety_result inbox exactly once."""
        if self._plan_result_inbox_registered:
            return
        t_plan_result = Template()
        t_plan_result.set_metadata("type", "plan_safety_result")
        self.add_behaviour(self._PlanSafetyResultInbox(), t_plan_result)
        self._plan_result_inbox_registered = True

    def _reactivate_blocked_tasks(
        self,
        *,
        candidate_task_ids: Optional[set[str]] = None,
    ) -> int:
        """
        Convert blocked tasks back to pending so they can be retried after
        replanning. If candidate_task_ids is provided, only reactivate those.
        """
        if candidate_task_ids is not None:
            candidate_task_ids = {str(tid) for tid in candidate_task_ids if tid}

        reactivated = 0
        for node in self.process_planner.nodes:
            if node.get("type") != "task":
                continue
            if node.get("status") != "blocked":
                continue

            node_id = str(node.get("id") or "")
            if candidate_task_ids is not None and node_id not in candidate_task_ids:
                continue

            node["status"] = "pending"
            reactivated += 1

        return reactivated

    async def _build_plan(self, requirement_text: str, safety_text: str = ""):
        """Build requirements, expand to tasks, and compile the global FSA."""
        # 1) NL → structured requirements
        await self.process_planner.build_high_level(requirement_text)
        self.process_planner.save(self.structured_requirements_path)

        # 2) Expand → DAG
        await self.process_planner.expand_requirements_to_tasks(safety_text=safety_text)
        self.process_planner.save(self.plan_path)

        # 3) Compile + save FSA (also sets self.process_planner.global_fsa)
        self.process_planner.save_global_fsa(self.global_fsa_path)

        # 4) Return both artifacts
        return self.process_planner.nodes, self.process_planner.global_fsa

    def _load_precomputed_plan_bundle(self) -> bool:
        """Load precomputed plan/global FSA artifacts when provided by startup bundle context."""
        bundle = dict(self.precomputed_bundle or {})
        artifacts = bundle.get("artifacts", {}) if isinstance(bundle, dict) else {}
        if not isinstance(artifacts, dict):
            return False

        plan_path = artifacts.get("plan_json")
        fsa_path = artifacts.get("global_fsa_json")
        req_path = artifacts.get("requirements_json")

        if not plan_path or not fsa_path:
            return False

        p_plan = Path(str(plan_path))
        p_fsa = Path(str(fsa_path))
        if not p_plan.exists() or not p_fsa.exists():
            self.logger.warning(
                "[Bundle] Precomputed plan artifacts missing plan=%s fsa=%s",
                p_plan,
                p_fsa,
            )
            return False

        self.process_planner.load(p_plan)
        self.process_planner.load_global_fsa(p_fsa)

        # Keep monitor snapshots aligned with runtime expectations.
        self.process_planner.save(self.plan_path)
        self.process_planner.save_global_fsa(self.global_fsa_path)

        if req_path:
            p_req = Path(str(req_path))
            if p_req.exists():
                self.structured_requirements_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p_req, self.structured_requirements_path)

        self.logger.info(
            "[Bundle] Using precomputed plan bundle_id=%s plan=%s fsa=%s",
            bundle.get("bundle_id", ""),
            p_plan,
            p_fsa,
        )
        return True



    # --------------------------------------------------------------------- #
    # Behaviours
    # --------------------------------------------------------------------- #
    class _Kickoff(OneShotBehaviour):
        async def run(self):
            agent: "ProductAgent" = self.agent

            # Initial Plan Build
            instruction = agent._extract_requirement_text()
            safety_text = agent._read_safety_text()
            agent.safety_text = safety_text  # Store for later use in replanning

            used_precomputed = agent._load_precomputed_plan_bundle()
            if not used_precomputed:
                await agent._build_plan(instruction, safety_text)

            # Retry Loop for Safety
            max_retries = 1 if used_precomputed else 3
            attempt = 0
            
            while attempt < max_retries:
                attempt += 1
                agent.logger.info(f"[Product] Validating Plan (Attempt {attempt}/{max_retries})...")
                
                # 1. Send to CCA
                payload = agent._build_plan_validation_payload()
                msg = Message(to=agent.cca_jid)
                msg.set_metadata("type", "plan_safety_check")
                msg.body = json.dumps(payload)
                await self.send(msg)
                
                # 2. Wait for Reply
                reply = None
                while reply is None:
                    reply = await self.receive(timeout=5.0)

                if reply.metadata.get("type") != "plan_safety_result":
                    continue # or handle error

                data = json.loads(reply.body)
                is_safe = data.get("ok", False)
                violations = data.get("violations", [])

                if is_safe:
                    agent.logger.info("[Product] Plan PASSED safety validation.")
                    agent._ensure_plan_result_inbox()
                    agent.add_behaviour(agent._PlanExecutor())
                    return # Exit Kickoff successfully

                if used_precomputed:
                    agent.logger.error(
                        "[Bundle] Precomputed plan failed safety validation (%d violations). Aborting kickoff.",
                        len(violations),
                    )
                    return

                # 3. Handle Failure
                agent.logger.warning(f"[Product] Plan FAILED safety check ({len(violations)} violations). Triggering Re-plan...")
                
                # Call the new Re-planning method
                await agent.process_planner.replan_with_feedback_offline(violations)
                
                # Loop continues to validate the NEW plan

            agent.logger.error("[Product] Max replanning attempts reached. Aborting.")

    class _AckInbox(CyclicBehaviour):
        """Background behaviour that listens for acknowledgements from resource agents."""

        async def run(self):
            agent: "ProductAgent" = self.agent  # type: ignore
            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            try:
                payload = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Product] Malformed ACK body (not JSON).")
                return

            task_id = payload.get("task_id", "?")
            status = payload.get("status", "unknown")

            # 1) Keep existing state map for UI/debug
            agent.task_states[task_id] = status

            # 2) ALSO update node status in the planner DAG if exists
            updated_node = False
            task_node = None
            for node in agent.process_planner.nodes:
                if node.get("id") == task_id:
                    # Map RA status → planner status; for now use it directly
                    if node.get("status") != status:
                        node["status"] = status
                        updated_node = True
                    task_node = node
                    break

            # 3) Track execution timeline
            from datetime import datetime, timezone
            agent.execution_timeline.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": task_id,
                "status": status,
                "resource_jid": str(msg.sender),
            })

            # 4) Update part location tracking
            if task_node:
                function_name = task_node.get("function_name", "")
                params = task_node.get("params", {})
                part_name = params.get("part_name")

                if part_name:
                    agent._apply_part_tracker_update(
                        part_name=part_name,
                        function_name=function_name,
                        status=status,
                        params=params,
                        resource_jid=str(msg.sender),
                        task_id=task_id,
                    )

            # NOTE: Robot states are NOT cached here - they're collected by CentralControllerAgent
            # and provided in the replan_request message (system_coordination_state)

            if updated_node:
                await asyncio.to_thread(agent._persist_plan_snapshot)
                await asyncio.to_thread(agent._persist_product_state)
                await asyncio.to_thread(agent._persist_resource_state)

            agent.logger.info(
                f"[Product] ACK ({task_id}) status='{status}' from={msg.sender}"
            )

    class _ReplanInbox(CyclicBehaviour):
        """Handle online replan requests from the CCA."""

        async def run(self):
            agent: "ProductAgent" = self.agent  # type: ignore
            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            try:
                payload = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Product] Malformed replan_request body.")
                return

            reason = payload.get("reason", "unknown")
            event = payload.get("event") or {}
            plan_ctx = payload.get("plan_ctx") or {}
            safety_ctx = payload.get("safety_ctx") or {}
            failure_ctx = event.get("failure_context") or {}
            system_coordination_state = payload.get("system_coordination_state") or {}

            failed_task_id = event.get("task_id") or plan_ctx.get("failed_task_id")
            if not failed_task_id:
                agent.logger.warning("[Product] Replan request missing failed_task_id.")
                return

            violations = agent._build_online_replan_violations(
                reason=reason,
                failed_task_id=str(failed_task_id),
                plan_ctx=plan_ctx,
                safety_ctx=safety_ctx,
                failure_ctx=failure_ctx,
            )

            agent.logger.info(
                "[Product] Online replan payload: %s", json.dumps(violations, ensure_ascii=False)
            )
            agent.logger.warning(
                "[Product] Online replan requested (%s) for failed task %s.",
                reason,
                failed_task_id,
            )
            await agent.process_planner.replan_with_feedback_online(
                violations,
                system_coordination_state=system_coordination_state,
            )

            # After online replan, let previously blocked tasks be re-evaluated via
            # fresh safety_check when they are dispatched again.
            candidate_ids: set[str] = set()
            for v in violations:
                if not isinstance(v, dict):
                    continue
                for k in ("failed_task_id", "task_id"):
                    tid = v.get(k)
                    if tid:
                        candidate_ids.add(str(tid))
                for k in ("affected_task_ids", "blocked_task_ids", "unreachable_task_ids"):
                    vals = v.get(k)
                    if isinstance(vals, (list, tuple, set)):
                        candidate_ids.update(str(x) for x in vals if x)

            reactivated = agent._reactivate_blocked_tasks(
                candidate_task_ids=(candidate_ids or None)
            )
            if reactivated:
                agent.logger.info(
                    "[Product] Reactivated %d blocked task(s) to pending after online replan.",
                    reactivated,
                )

            # Rebuild + re-register runtime FSA monitor context at CCA
            # so online monitoring tracks the repaired plan structure.
            try:
                agent.process_planner.compile_global_fsa()
                agent.process_planner.save_global_fsa(agent.global_fsa_path)

                payload = agent._build_plan_validation_payload()
                agent._ensure_plan_result_inbox()
                msg_check = Message(to=agent.cca_jid)
                msg_check.set_metadata("type", "plan_safety_check")
                msg_check.body = json.dumps(payload)
                await self.send(msg_check)

                agent.logger.info(
                    "[Product] Recompiled plan FSA after online replan and sent plan_safety_check to CCA."
                )
            except Exception:
                agent.logger.exception(
                    "[Product] Failed to rebuild/re-register FSA after online replan."
                )

            await asyncio.to_thread(agent._persist_plan_snapshot)
            await asyncio.to_thread(agent._persist_product_state)
            await asyncio.to_thread(agent._persist_resource_state)

    class _PlanSafetyResultInbox(CyclicBehaviour):
        """Handle runtime plan_safety_result replies from CCA."""

        async def run(self):
            agent: "ProductAgent" = self.agent  # type: ignore
            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            try:
                payload = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Product] Malformed plan_safety_result body.")
                return

            ok = bool(payload.get("ok", False))
            violations = payload.get("violations")
            if not isinstance(violations, list):
                violations = []

            if ok:
                if agent._runtime_repair_fail_streak:
                    agent.logger.info(
                        "[Product] Runtime plan validation recovered after %d repair attempt(s).",
                        agent._runtime_repair_fail_streak,
                    )
                agent._runtime_repair_fail_streak = 0
                return

            if agent._runtime_repair_inflight:
                agent.logger.warning(
                    "[Product] Runtime plan validation failed while repair is already in progress; ignoring duplicate result."
                )
                return

            if agent._runtime_repair_fail_streak >= agent._runtime_repair_max_attempts:
                agent.logger.error(
                    "[Product] Runtime plan validation still failing after %d repair attempt(s); giving up automatic retries.",
                    agent._runtime_repair_fail_streak,
                )
                return

            agent._runtime_repair_inflight = True
            agent._runtime_repair_fail_streak += 1
            try:
                agent.logger.warning(
                    "[Product] Runtime plan validation failed (%d violation(s)); triggering corrective replan attempt %d/%d.",
                    len(violations),
                    agent._runtime_repair_fail_streak,
                    agent._runtime_repair_max_attempts,
                )
                await agent.process_planner.replan_with_feedback_offline(violations)
                agent.process_planner.compile_global_fsa()
                agent.process_planner.save_global_fsa(agent.global_fsa_path)

                check_payload = agent._build_plan_validation_payload()
                check_msg = Message(to=agent.cca_jid)
                check_msg.set_metadata("type", "plan_safety_check")
                check_msg.body = json.dumps(check_payload)
                await self.send(check_msg)
                await asyncio.to_thread(agent._persist_plan_snapshot)
                await asyncio.to_thread(agent._persist_product_state)
                await asyncio.to_thread(agent._persist_resource_state)
            except Exception:
                agent.logger.exception(
                    "[Product] Corrective runtime replan attempt failed."
                )
            finally:
                agent._runtime_repair_inflight = False

    class _PlanExecutor(CyclicBehaviour):
        """
        Periodically checks the DAG for the next ready task and dispatches it
        as a SPADE message to the resource agent.
        """

        async def run(self):
            agent: "ProductAgent" = self.agent  # type: ignore

            # No resources? nothing to do
            if not agent.resource_jids:
                return

            # Ask planner for one ready task
            task_node = agent.process_planner.next_ready_task()
            if not task_node:
                await asyncio.sleep(0.5)
                return

            # NEW: prefer resource_jid chosen by the planner (LLM)
            to = task_node.get("resource_jid")
            if not to:
                # Fallback: first configured resource JID
                to = agent.resource_jids[0]
                agent.logger.warning(
                    "[Product] Task %s has no resource_jid, falling back to %s",
                    task_node.get("id"),
                    to,
                )

            # Build the instruction for the RobotAgent from the DAG node.
            # Enrich params with product geometry when a part_name is present.
            params = dict(task_node.get("params", {}))
            part_name = params.get("part_name")
            if part_name:
                geo = agent._geometry_for_part(part_name)
                if geo:
                    params["product_geometry"] = geo

            instruction = {
                "function_name": task_node.get("function_name"),
                "params": params,
            }

            task_id = task_node["id"]

            msg = agent._compose_task_msg(
                to=to,
                task_id=task_id,
                instruction=instruction,
                phase_id=None,
            )

            # Mark as "dispatched" (still waiting for ACK to flip to "completed")
            task_node["status"] = "dispatched"

            await self.send(msg)
            agent.logger.info(
                f"[Product] Dispatched DAG task {task_id} -> {to} ({instruction})"
            )

            # Short sleep so we don't hammer the RA with a storm of tasks
            await asyncio.sleep(0.1)

    @staticmethod
    def _match_resource_objects(
        resources: Iterable[Any], target_jids: Iterable[str]
    ) -> list[Any]:
        """Return the subset of *resources* whose JIDs appear in *target_jids* (case-insensitive)."""
        resources = list(resources or [])
        if not resources:
            return []

        target_jids_lower = {
            str(jid).lower() for jid in target_jids if jid is not None
        }
        if not target_jids_lower:
            return resources

        matched = []
        for agent in resources:
            # Each agent is expected to expose a `jid`; treat missing values as empty strings.
            agent_jid = str(getattr(agent, "jid", "")).lower()
            # Append resource agents whose JID matches one of the requested targets.
            if agent_jid and agent_jid in target_jids_lower:
                matched.append(agent)
        return matched

    def _collect_descendants(self, root_id: str) -> list[str]:
        """Return all descendant task IDs in the current planner DAG."""
        succ_map: dict[str, list[str]] = {}
        for node in self.process_planner.nodes:
            if node.get("type") != "task":
                continue
            nid = node.get("id")
            if not nid:
                continue
            succs = node.get("successors") or []
            succ_map[str(nid)] = [str(s) for s in succs if s]

        seen: set[str] = set()
        stack = [root_id]
        while stack:
            cur = stack.pop()
            for nxt in succ_map.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return sorted(seen)

    def _build_online_replan_violations(
        self,
        *,
        reason: str,
        failed_task_id: str,
        plan_ctx: dict,
        safety_ctx: dict,
        failure_ctx: dict,
    ) -> list[dict]:
        """
        Build a standardized online replanning payload for the planner.
        """
        descendants = self._collect_descendants(failed_task_id)
        # Prefer explicit failed-task descendants; fall back to legacy names if present.
        unreachable = (
            plan_ctx.get("failed_task_descendant_ids")
            or plan_ctx.get("failure_blocked_task_ids")
            or plan_ctx.get("blocked_by_failure_task_ids")
            or plan_ctx.get("unreachable_task_ids")
            or []
        )
        # Ensure the failed task itself is included if present.
        plan_failed_task_id = plan_ctx.get("failed_task_id")
        if plan_failed_task_id:
            failed_task_id = str(plan_failed_task_id)
        if failed_task_id:
            unreachable = sorted(set(unreachable) | {str(failed_task_id)})
        affected_task_ids = sorted(set(unreachable) | set(descendants))

        return [
            {
                "type": reason,
                "failed_task_id": str(failed_task_id),
                "affected_task_ids": affected_task_ids,
                "failure_context": failure_ctx,
                "safety_ctx": safety_ctx,
            }
        ]

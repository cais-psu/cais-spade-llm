"""High-level product agent that reads build instructions and instructs a resource agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message
from spade.template import Template

from cais_spade_llm.agents.shared_information.llm_agent import LlmAgent
from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.agents.intelligent_product.product_recovery_controller import (
    ProductRecoveryController,
)
from cais_spade_llm.product.profile import ProductProfile
from cais_spade_llm.resources.sensor.camera_module import CameraModule

_UNSET = object()
_LEGACY_PROCEDURAL_DES_BRIDGE_MODE = "procedural" + "_des_v1"


def _env_flag_enabled(*names: str, default: bool = False) -> bool:
    for name in names:
        token = str(os.environ.get(name) or "").strip().lower()
        if not token:
            continue
        return token in {"1", "true", "yes", "on"}
    return bool(default)


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
        precomputed_bundle: Optional[Dict[str, Any]] = None,
        **kw,
    ) -> None:
        """
        :param resource_jids: List of RA JIDs to target (first is used).
        :param product_specification_file: Path to spec text (utf-8). Optional.
        :param instruction_override: If provided, this text is used instead of reading a file.
        """
        super().__init__(jid, password, name=name, agent_role="product", **kw)

        self.cca_jid = cca_jid or "cca@localhost"

        # Resource agents are the downstream executors; keep them in order and avoid mutating caller lists.
        self.resource_jids = list(resource_jids or [])
        self._resource_agent_refs = list(resource_agents or [])
        self.product_profile = ProductProfile(
            name=name,
            product_specification_file=product_specification_file,
            product_geometry_file=product_geometry_file,
            safety_file=safety_file,
            instruction_override=instruction_override,
            precomputed_bundle=dict(precomputed_bundle or {}),
            logger=self.logger,
        )
        self.product_specification_file = self.product_profile.product_specification_file
        self.product_geometry_file = self.product_profile.product_geometry_file
        self.product_geometry: Dict[str, Any] = dict(self.product_profile.product_geometry)
        self.safety_file = self.product_profile.safety_file
        self.robot_env = self.product_profile.robot_env
        # Manual instruction text provided at runtime overrides any file read.
        self.instruction_override = self.product_profile.instruction_override

        # Cache safety text for use during replanning
        self.safety_text: str = ""

        self.precomputed_bundle: dict[str, Any] = dict(self.product_profile.precomputed_bundle or {})

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
        self.recovery_controller = ProductRecoveryController(self)
        self.recovery_controller.bind_methods()
        
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
        self._bridge_generation_mode = "auto"
        self._bridge_reasoning_mode = "multi_turn"
        self._runtime_bridge_mode = "pre_ran"
        self._runtime_bridge_validation_policy = "validated"
        self._runtime_bridge_archive_path = ""
        self._runtime_bridge_archive_label = ""
        self._orphaned_bridge_task_warning_ids: set[str] = set()
        self._generated_bridge_gazebo_verification_enabled = _env_flag_enabled(
            "CAIS_VERIFY_GENERATED_BRIDGE_IN_GAZEBO",
            "CAIS_GENERATED_BRIDGE_GAZEBO_VERIFICATION",
            default=False,
        )
        self.runtime_repair_state = "idle"
        self.plan_safety_alert: dict[str, Any] | None = None
        self.runtime_recovery: dict[str, Any] = self._empty_runtime_recovery()
        self._runtime_recovery_context: dict[str, Any] = {}
        self.kickoff_result: dict[str, Any] = {
            "success": False,
            "message": "kickoff pending",
            "retries_used": 0,
            "retries_max": 0,
            "violated_rules": [],
            "witness_count": 0,
            "updated_at_utc": "",
            "product_name": name,
            "product_jid": str(self.jid),
            "stage": "kickoff",
            "alert": None,
        }
        self._kickoff_result_event = asyncio.Event()
        precomputed_policy = (
            self.precomputed_bundle.get("replan_policy", {})
            if isinstance(self.precomputed_bundle.get("replan_policy"), dict)
            else {}
        )
        if precomputed_policy:
            try:
                self._runtime_repair_max_attempts = max(
                    0,
                    min(int(precomputed_policy.get("auto_replan_max_attempts", 3) or 0), 10),
                )
            except Exception:
                self._runtime_repair_max_attempts = 3
            self._bridge_generation_mode = str(
                precomputed_policy.get("bridge_generation_mode", "auto") or "auto"
            ).strip().lower()
            if self._bridge_generation_mode not in {"auto", "manual"}:
                self._bridge_generation_mode = "auto"
            self._bridge_reasoning_mode = str(
                precomputed_policy.get("bridge_reasoning_mode", "multi_turn")
                or "multi_turn"
            ).strip().lower()
            if self._bridge_reasoning_mode in {
                "hybrid",
                "procedural",
                _LEGACY_PROCEDURAL_DES_BRIDGE_MODE,
            }:
                self._bridge_reasoning_mode = "multi_turn"
            if self._bridge_reasoning_mode != "multi_turn":
                self._bridge_reasoning_mode = "multi_turn"
            if "generated_bridge_gazebo_verification" in precomputed_policy:
                self._generated_bridge_gazebo_verification_enabled = bool(
                    precomputed_policy.get("generated_bridge_gazebo_verification")
                )

        self.logger.info(f"ProductAgent '{name}' initialized.")

    # ------------------------------------------------------------------ #
    # Persistence helper
    # ------------------------------------------------------------------ #
    def _build_plan_validation_payload(
        self,
        *,
        skip_revalidation: bool = False,
        skip_offline_validation: bool | None = None,
    ):
        """Package plan + FSA for plan validation by the CCA."""
        if skip_offline_validation is not None:
            skip_revalidation = bool(skip_offline_validation)

        fsa = self.process_planner.global_fsa
        nodes = self.process_planner.nodes

        if fsa is None and not skip_revalidation:
            raise RuntimeError("Global FSA is None. Did you call save_global_fsa()?")

        return {
            "fsa": fsa or {},                # <-- upload FSA here
            "product_jid": str(self.jid),
            "plan": {"nodes": nodes},
            "runtime_context": self._build_runtime_plan_context(),
            "skip_revalidation": bool(skip_revalidation),
        }

    def _build_runtime_plan_context(self) -> dict[str, Any]:
        completed_task_ids: list[str] = []
        seen_completed: set[str] = set()
        for event in self.execution_timeline:
            if not isinstance(event, dict):
                continue
            task_id = str(event.get("task_id", "")).strip()
            status = str(event.get("status", "")).strip().lower()
            if not task_id:
                continue
            if status in {"completed", "finished"} and task_id not in seen_completed:
                seen_completed.add(task_id)
                completed_task_ids.append(task_id)

        running_task_ids: list[str] = []
        failed_task_ids: list[str] = []
        for node in self.process_planner.nodes:
            if not isinstance(node, dict) or node.get("type") != "task":
                continue
            task_id = str(node.get("id", "")).strip()
            if not task_id:
                continue
            status = str(node.get("status", "")).strip().lower()
            if status == "running":
                running_task_ids.append(task_id)
            elif status == "failed" or status.startswith("failed"):
                failed_task_ids.append(task_id)

        return {
            "completed_task_ids": completed_task_ids,
            "running_task_ids": running_task_ids,
            "failed_task_ids": failed_task_ids,
        }


    def _dispatch_agent_message_sync(
        self,
        msg: Message,
        *,
        trace_category: str = "agent",
    ) -> None:
        """Send a SPADE message synchronously from local helper code."""
        if msg.empty_sender():
            msg.sender = str(self.jid)

        if self.container.has_agent(str(msg.to)):
            self.container.get_agent(str(msg.to)).dispatch(msg)
        else:
            if self.client is None:
                raise RuntimeError("agent client is not connected")
            slixmpp_msg = msg.prepare(self.client)
            slixmpp_msg.send()

        msg.sent = True
        self.traces.append(msg, category=trace_category)


    def _run_callable_on_agent_loop_sync(
        self,
        callback: Any,
        *,
        timeout_sec: float = 10.0,
        operation_name: str = "product agent callback",
    ) -> Any:
        loop = getattr(self, "loop", None)
        if loop is None:
            raise RuntimeError("product agent loop is unavailable")
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            return callback()

        future: ConcurrentFuture[Any] = ConcurrentFuture()

        def _invoke() -> None:
            try:
                future.set_result(callback())
            except Exception as exc:
                future.set_exception(exc)

        loop.call_soon_threadsafe(_invoke)
        try:
            return future.result(timeout=max(1.0, float(timeout_sec)))
        except FutureTimeoutError as exc:
            raise RuntimeError(
                f"{str(operation_name or 'product agent callback').strip() or 'product agent callback'} "
                f"timed out after {max(1.0, float(timeout_sec)):.0f}s"
            ) from exc

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _violation_summary(violations: list[dict[str, Any]] | None) -> tuple[list[str], int]:
        items = violations if isinstance(violations, list) else []
        violated_rules = sorted(
            {
                str(v.get("violated_rule_id"))
                for v in items
                if isinstance(v, dict) and v.get("violated_rule_id")
            }
        )
        return violated_rules, len(items)

    def _task_nodes_hash(self) -> str:
        task_nodes = [
            node
            for node in self.process_planner.nodes
            if isinstance(node, dict) and node.get("type") == "task"
        ]
        canonical = json.dumps(task_nodes, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()






































































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

        # Retry-ready inbox (from CCA) for transient safety blocks that cleared.
        t_retry_ready = Template()
        t_retry_ready.set_metadata("type", "task_retry_ready")
        self.add_behaviour(self._TaskRetryReadyInbox(), t_retry_ready)

        # Plan executor (runs cycles, dispatches DAG tasks)
        # self.add_behaviour(self._PlanExecutor())

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _read_safety_text(self) -> str:
        """Compatibility wrapper for ProductProfile safety text loading."""
        return ProductProfile.read_safety_file(self.safety_file, logger=self.logger)
    


    def _geometry_for_part(self, part_name: str) -> Dict[str, Any]:
        """Compatibility wrapper for ProductProfile per-part geometry lookup."""
        profile = getattr(self, "product_profile", None)
        if isinstance(profile, ProductProfile):
            return profile.geometry_for_part(part_name, product_geometry=self.product_geometry)
        return ProductProfile.geometry_for_part_from_geometry(part_name, self.product_geometry)

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
        """Compatibility wrapper for ProductProfile requirement text loading."""
        return ProductProfile.extract_requirement_file(
            self.product_specification_file,
            logger=self.logger,
        )

    def _ensure_plan_result_inbox(self) -> None:
        """Register runtime plan_safety_result inbox exactly once."""
        if self._plan_result_inbox_registered:
            return
        t_plan_result = Template()
        t_plan_result.set_metadata("type", "plan_safety_result")
        self.add_behaviour(self._PlanSafetyResultInbox(), t_plan_result)
        self._plan_result_inbox_registered = True























































    @staticmethod






























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
            retries_used = 0
            used_precomputed = False
            max_retries = 3

            try:
                instruction = agent._extract_requirement_text()
                safety_text = agent._read_safety_text()
                agent.safety_text = safety_text
                agent.runtime_repair_state = "idle"
                agent._runtime_repair_fail_streak = 0
                agent._clear_plan_safety_alert()
                agent._clear_runtime_recovery()

                used_precomputed = agent._load_precomputed_plan_bundle()
                max_retries = 0 if used_precomputed else 3
                if not used_precomputed:
                    if not instruction:
                        raise RuntimeError("no product requirement text available for startup planning")
                    await agent._build_plan(instruction, safety_text)

                while True:
                    if used_precomputed:
                        agent.logger.info(
                            "[Product] Initializing startup runtime monitors for verified plan bundle (offline revalidation skipped)."
                        )
                    else:
                        agent.logger.info(
                            "[Product] Validating startup plan (auto-replans %d/%d)...",
                            retries_used,
                            max_retries,
                        )

                    payload = agent._build_plan_validation_payload(
                        skip_revalidation=used_precomputed,
                    )
                    msg = Message(to=agent.cca_jid)
                    msg.set_metadata("type", "plan_safety_check")
                    msg.body = json.dumps(payload)
                    await self.send(msg)

                    reply = None
                    while reply is None:
                        reply = await self.receive(timeout=5.0)

                    if reply.metadata.get("type") != "plan_safety_result":
                        continue

                    data = json.loads(reply.body or "{}")
                    is_safe = bool(data.get("ok", False))
                    violations = data.get("violations")
                    if not isinstance(violations, list):
                        violations = []

                    if is_safe:
                        message = (
                            f"{agent.agent_name}: loaded verified plan set; startup skipped offline safety revalidation."
                            if used_precomputed
                            else f"{agent.agent_name}: startup plan passed safety validation."
                        )
                        agent.logger.info("[Product] Plan PASSED safety validation.")
                        agent._clear_plan_safety_alert()
                        agent._ensure_plan_result_inbox()
                        agent.add_behaviour(agent._PlanExecutor())
                        agent._set_kickoff_result(
                            success=True,
                            message=message,
                            retries_used=retries_used,
                            retries_max=max_retries,
                            violations=[],
                        )
                        return

                    if used_precomputed:
                        message = (
                            f"{agent.agent_name}: verified plan set failed startup safety validation "
                            f"({len(violations)} witness(es)); startup aborted."
                        )
                        agent.logger.error(
                            "[Bundle] Precomputed plan failed safety validation (%d violations). Aborting kickoff.",
                            len(violations),
                        )
                        alert = agent._set_plan_safety_alert(
                            stage="kickoff",
                            message=message,
                            retries_used=0,
                            retries_max=0,
                            violations=violations,
                            paused=False,
                        )
                        agent._set_kickoff_result(
                            success=False,
                            message=message,
                            retries_used=0,
                            retries_max=0,
                            violations=violations,
                            alert=alert,
                        )
                        return

                    if retries_used >= max_retries:
                        message = (
                            f"{agent.agent_name}: startup plan still violates safety after "
                            f"{retries_used}/{max_retries} auto-replan attempt(s); startup aborted."
                        )
                        agent.logger.error("[Product] Max replanning attempts reached. Aborting.")
                        alert = agent._set_plan_safety_alert(
                            stage="kickoff",
                            message=message,
                            retries_used=retries_used,
                            retries_max=max_retries,
                            violations=violations,
                            paused=False,
                        )
                        agent._set_kickoff_result(
                            success=False,
                            message=message,
                            retries_used=retries_used,
                            retries_max=max_retries,
                            violations=violations,
                            alert=alert,
                        )
                        return

                    next_attempt = retries_used + 1
                    before_hash = agent._task_nodes_hash()
                    agent.logger.warning(
                        "[Product] Startup plan failed safety check (%d violation(s)); triggering auto-replan %d/%d.",
                        len(violations),
                        next_attempt,
                        max_retries,
                    )
                    try:
                        retries_used = next_attempt
                        await agent.process_planner.replan_with_feedback_offline(violations)
                        if before_hash == agent._task_nodes_hash():
                            message = (
                                f"{agent.agent_name}: startup auto-replan produced no plan change on "
                                f"attempt {retries_used}/{max_retries}; startup aborted."
                            )
                            alert = agent._set_plan_safety_alert(
                                stage="kickoff",
                                message=message,
                                retries_used=retries_used,
                                retries_max=max_retries,
                                violations=violations,
                                paused=False,
                            )
                            agent._set_kickoff_result(
                                success=False,
                                message=message,
                                retries_used=retries_used,
                                retries_max=max_retries,
                                violations=violations,
                                alert=alert,
                            )
                            return
                        agent.process_planner.compile_global_fsa()
                        await asyncio.to_thread(agent._persist_plan_snapshot)
                        await asyncio.to_thread(agent.process_planner.save_global_fsa, agent.global_fsa_path)
                    except Exception as exc:
                        message = (
                            f"{agent.agent_name}: startup auto-replan attempt {retries_used}/{max_retries} "
                            f"failed ({exc}); startup aborted."
                        )
                        agent.logger.exception("[Product] Startup auto-replan attempt failed.")
                        alert = agent._set_plan_safety_alert(
                            stage="kickoff",
                            message=message,
                            retries_used=retries_used,
                            retries_max=max_retries,
                            violations=violations,
                            paused=False,
                        )
                        agent._set_kickoff_result(
                            success=False,
                            message=message,
                            retries_used=retries_used,
                            retries_max=max_retries,
                            violations=violations,
                            alert=alert,
                        )
                        return
            except Exception as exc:
                message = f"{agent.agent_name}: kickoff failed ({exc})."
                agent.logger.exception("[Product] Kickoff failed.")
                alert = agent._set_plan_safety_alert(
                    stage="kickoff",
                    message=message,
                    retries_used=retries_used,
                    retries_max=max_retries,
                    violations=[],
                    paused=False,
                )
                agent._set_kickoff_result(
                    success=False,
                    message=message,
                    retries_used=retries_used,
                    retries_max=max_retries,
                    violations=[],
                    alert=alert,
                )

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
            content = str(payload.get("content") or "").strip()
            observations = payload.get("observations")
            if not isinstance(observations, dict):
                observations = None

            existing_task_node = None
            for node in agent.process_planner.nodes:
                if node.get("id") == task_id:
                    existing_task_node = node
                    break

            if agent._should_ignore_stale_recovery_macro_ack(
                task_node=existing_task_node,
                incoming_status=str(status),
            ):
                agent.logger.warning(
                    "[Product] Ignoring stale ACK for completed recovery task %s: current_status=%s incoming_status=%s",
                    task_id,
                    str((existing_task_node or {}).get("status") or "").strip() or "<unknown>",
                    str(status or "").strip() or "<unknown>",
                )
                return

            # 1) Keep existing state map for UI/debug
            agent.task_states[task_id] = status

            # 2) ALSO update node status in the planner DAG if exists
            updated_node = False
            task_node = existing_task_node
            if task_node:
                # Map RA status → planner status; for now use it directly
                if task_node.get("status") != status:
                    task_node["status"] = status
                    updated_node = True

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
                part_name = agent._tracked_part_name_for_task(task_node)

                if part_name:
                    agent._apply_part_tracker_update(
                        part_name=part_name,
                        function_name=function_name,
                        status=status,
                        params=params,
                        resource_jid=str(msg.sender),
                        task_id=task_id,
                        task_node=task_node,
                        observations=observations,
                    )

            # NOTE: Robot states are NOT cached here - they're collected by CentralControllerAgent
            # and provided in the replan_request message (system_coordination_state)

            handled_bridge_ack = False
            if task_node:
                handled_bridge_ack = await agent._handle_bridge_macro_ack(
                    task_node=task_node,
                    status=str(status),
                    content=content,
                    observations=observations,
                )

            if updated_node and not handled_bridge_ack:
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
            await agent._handle_runtime_des_replan_request(
                reason=str(reason),
                failed_task_id=str(failed_task_id),
                violations=violations,
                system_coordination_state=system_coordination_state,
            )
            await asyncio.to_thread(agent._persist_product_state)
            return

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

            if await agent._handle_runtime_plan_validation_result(
                ok=ok,
                violations=violations,
            ):
                return

            if ok:
                if agent._runtime_repair_fail_streak:
                    agent.logger.info(
                        "[Product] Runtime plan validation recovered after %d repair attempt(s).",
                        agent._runtime_repair_fail_streak,
                    )
                agent._runtime_repair_fail_streak = 0
                agent.runtime_repair_state = "idle"
                agent._clear_plan_safety_alert()
                await asyncio.to_thread(agent._persist_product_state)
                return

            if agent._runtime_repair_inflight:
                agent.logger.warning(
                    "[Product] Runtime plan validation failed while repair is already in progress; ignoring duplicate result."
                )
                return

            if agent._runtime_repair_fail_streak >= agent._runtime_repair_max_attempts:
                message = (
                    f"{agent.agent_name}: runtime plan validation still fails after "
                    f"{agent._runtime_repair_fail_streak}/{agent._runtime_repair_max_attempts} "
                    "auto-replan attempt(s); execution paused."
                )
                agent.logger.error(
                    "[Product] Runtime plan validation still failing after %d repair attempt(s); pausing execution.",
                    agent._runtime_repair_fail_streak,
                )
                agent.runtime_repair_state = "paused_after_failure"
                agent._set_plan_safety_alert(
                    stage="runtime",
                    message=message,
                    retries_used=agent._runtime_repair_fail_streak,
                    retries_max=agent._runtime_repair_max_attempts,
                    violations=violations,
                    paused=True,
                )
                await asyncio.to_thread(agent._persist_product_state)
                return

            agent._runtime_repair_inflight = True
            agent.runtime_repair_state = "repairing"
            agent._runtime_repair_fail_streak += 1
            try:
                before_hash = agent._task_nodes_hash()
                agent.logger.warning(
                    "[Product] Runtime plan validation failed (%d violation(s)); triggering corrective replan attempt %d/%d.",
                    len(violations),
                    agent._runtime_repair_fail_streak,
                    agent._runtime_repair_max_attempts,
                )
                await agent.process_planner.replan_with_feedback_offline(violations)
                if before_hash == agent._task_nodes_hash():
                    message = (
                        f"{agent.agent_name}: runtime auto-replan produced no plan change on "
                        f"attempt {agent._runtime_repair_fail_streak}/{agent._runtime_repair_max_attempts}; "
                        "execution paused."
                    )
                    agent.runtime_repair_state = "paused_after_failure"
                    agent._set_plan_safety_alert(
                        stage="runtime",
                        message=message,
                        retries_used=agent._runtime_repair_fail_streak,
                        retries_max=agent._runtime_repair_max_attempts,
                        violations=violations,
                        paused=True,
                    )
                    await asyncio.to_thread(agent._persist_product_state)
                    return
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
                message = (
                    f"{agent.agent_name}: runtime auto-replan attempt "
                    f"{agent._runtime_repair_fail_streak}/{agent._runtime_repair_max_attempts} failed; "
                    "execution paused."
                )
                agent.runtime_repair_state = "paused_after_failure"
                agent._set_plan_safety_alert(
                    stage="runtime",
                    message=message,
                    retries_used=agent._runtime_repair_fail_streak,
                    retries_max=agent._runtime_repair_max_attempts,
                    violations=violations,
                    paused=True,
                )
                await asyncio.to_thread(agent._persist_product_state)
            finally:
                agent._runtime_repair_inflight = False

    class _TaskRetryReadyInbox(CyclicBehaviour):
        """Requeue blocked tasks after CCA clears a transient safety block."""

        async def run(self):
            agent: "ProductAgent" = self.agent  # type: ignore
            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            try:
                payload = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Product] Malformed task_retry_ready body.")
                return

            task_ids = payload.get("task_ids")
            if not isinstance(task_ids, list):
                task_id = payload.get("task_id")
                task_ids = [task_id] if task_id else []

            reactivated = agent._handle_task_retry_ready(task_ids)
            if not reactivated:
                return

            agent.logger.info(
                "[Product] Requeued %d blocked task(s) after CCA cleared transient safety block: %s",
                reactivated,
                ", ".join(str(task_id) for task_id in task_ids if task_id),
            )
            await asyncio.to_thread(agent._persist_plan_snapshot)
            await asyncio.to_thread(agent._persist_product_state)
            await asyncio.to_thread(agent._persist_resource_state)

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

            if agent._runtime_recovery_blocks_execution():
                await asyncio.sleep(0.05)
                return

            # Prioritize active bridge sequences over nominal DAG work. The first
            # recovery macro may be anchored after the failed task, which is
            # intentionally not "completed" during runtime recovery.
            task_node = agent._next_dispatchable_task_node()
            if not task_node and agent._active_bridge_blocks_nominal_dispatch():
                agent.logger.debug(
                    "[Product] Active bridge sequence is executing; suppressing nominal DAG dispatch."
                )
                await asyncio.sleep(0.05)
                return
            if not task_node:
                await asyncio.sleep(0.05)
                return

            task_id = str(task_node.get("id") or "").strip()
            planner_status = str(task_node.get("status") or "").strip().lower()
            tracked_status = str(agent.task_states.get(task_id) or "").strip().lower()
            current_sequence_for_guard = agent._active_bridge_sequence()
            bridge_recorded_status = ""
            if isinstance(current_sequence_for_guard, dict):
                completed_task_ids = set(
                    agent._bridge_sequence_task_ids(
                        current_sequence_for_guard.get("completed_bridge_task_ids") or []
                    )
                )
                dispatched_task_ids = set(
                    agent._bridge_sequence_task_ids(
                        current_sequence_for_guard.get("dispatched_bridge_task_ids") or []
                    )
                )
                if task_id in completed_task_ids:
                    bridge_recorded_status = "completed"
                elif task_id in dispatched_task_ids:
                    bridge_recorded_status = "dispatched"
            effective_tracked_status = bridge_recorded_status or tracked_status
            if planner_status == "pending" and effective_tracked_status in {
                "accepted",
                "running",
                "dispatched",
                "completed",
                "blocked",
            }:
                task_node["status"] = effective_tracked_status
                agent.logger.warning(
                    "[Product] Suppressing duplicate dispatch for %s: planner_status=pending tracked_status=%s",
                    task_id,
                    effective_tracked_status,
                )
                if str(task_node.get("function_name") or "").strip() == "execute_recovery_macro":
                    await asyncio.to_thread(agent._persist_plan_snapshot)
                    await asyncio.to_thread(agent._persist_product_state)
                await asyncio.sleep(0.05)
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
            # Recovery macro primitive steps own their destination intent.
            params = agent._dispatch_params_for_task_node(task_node)

            instruction = {
                "function_name": task_node.get("function_name"),
                "params": params,
            }

            msg = agent._compose_task_msg(
                to=to,
                task_id=task_id,
                instruction=instruction,
                phase_id=None,
            )

            # Mark as "dispatched" (still waiting for ACK to flip to "completed")
            task_node["status"] = "dispatched"
            agent.task_states[task_id] = "dispatched"

            is_bridge_task = (
                str(task_node.get("function_name") or "").strip() == "execute_recovery_macro"
                and str(task_node.get("bridge_sequence_id") or "").strip()
            )
            if is_bridge_task:
                current_sequence = agent._active_bridge_sequence()
                if (
                    isinstance(current_sequence, dict)
                    and str(current_sequence.get("bridge_sequence_id") or "").strip()
                    == str(task_node.get("bridge_sequence_id") or "").strip()
                ):
                    repair_task_id = str(current_sequence.get("repair_task_id") or "").strip()
                    repair_target_task_id = str(
                        current_sequence.get("repair_target_task_id") or ""
                    ).strip()
                    if repair_task_id and repair_task_id == task_id:
                        agent.logger.info(
                            "[Product] Dispatching runtime DES repair %s before restored task %s.",
                            task_id,
                            repair_target_task_id or "<unknown>",
                        )
                    next_sequence = agent._bridge_sequence_with_dispatched_task(
                        current_sequence,
                        task_id=task_id,
                    )
                    if isinstance(next_sequence, dict):
                        agent._set_runtime_recovery(
                            message=str(agent.runtime_recovery.get("message", "") or "").strip(),
                            active_bridge_sequence=next_sequence,
                        )

            await self.send(msg)
            agent.logger.info(
                f"[Product] Dispatched DAG task {task_id} -> {to} ({instruction})"
            )

            if is_bridge_task:
                await asyncio.to_thread(agent._persist_plan_snapshot)
                await asyncio.to_thread(agent._persist_product_state)

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
        return self._collect_descendants_from_nodes(self.process_planner.nodes, root_id)

    @staticmethod
    def _collect_descendants_from_nodes(nodes: list[dict[str, Any]], root_id: str) -> list[str]:
        """Return all descendant task IDs from the provided task graph snapshot."""
        succ_map: dict[str, list[str]] = {}
        for node in nodes or []:
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

    @staticmethod
    def _direct_predecessors_from_nodes(nodes: list[dict[str, Any]], task_id: str) -> list[str]:
        lookup = {
            str(node.get("id", "")).strip(): node
            for node in nodes or []
            if isinstance(node, dict) and str(node.get("id", "")).strip()
        }
        task = lookup.get(str(task_id or "").strip()) or {}
        return [
            str(pred).strip()
            for pred in (task.get("predecessors") or [])
            if str(pred or "").strip() and str(pred or "").strip() in lookup
        ]

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

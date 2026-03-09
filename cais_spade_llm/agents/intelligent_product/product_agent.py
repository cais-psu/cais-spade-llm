"""High-level product agent that reads build instructions and instructs a resource agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os, uuid
import shutil
from copy import deepcopy
from datetime import datetime, timezone
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
        :param replan_mode: Online replanning strategy — "des", "llm", or "none".
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
        fsa = self.process_planner.global_fsa
        nodes = self.process_planner.nodes

        if fsa is None:
            raise RuntimeError("Global FSA is None. Did you call save_global_fsa()?")

        if skip_offline_validation is not None:
            skip_revalidation = bool(skip_offline_validation)

        return {
            "fsa": fsa,                      # <-- upload FSA here
            "product_jid": str(self.jid),
            "plan": {"nodes": nodes},
            "skip_revalidation": bool(skip_revalidation),
        }

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

    def _build_plan_safety_alert(
        self,
        *,
        stage: str,
        message: str,
        retries_used: int,
        retries_max: int,
        violations: list[dict[str, Any]] | None = None,
        paused: bool = False,
    ) -> dict[str, Any]:
        violated_rules, witness_count = self._violation_summary(violations)
        return {
            "product_name": self.agent_name,
            "product_jid": str(self.jid),
            "stage": str(stage),
            "message": str(message),
            "retries_used": int(retries_used),
            "retries_max": int(retries_max),
            "violated_rules": violated_rules,
            "witness_count": witness_count,
            "paused": bool(paused),
            "updated_at_utc": self._utc_now_iso(),
        }

    def _set_plan_safety_alert(
        self,
        *,
        stage: str,
        message: str,
        retries_used: int,
        retries_max: int,
        violations: list[dict[str, Any]] | None = None,
        paused: bool = False,
    ) -> dict[str, Any]:
        alert = self._build_plan_safety_alert(
            stage=stage,
            message=message,
            retries_used=retries_used,
            retries_max=retries_max,
            violations=violations,
            paused=paused,
        )
        self.plan_safety_alert = alert
        return alert

    def _clear_plan_safety_alert(self) -> None:
        self.plan_safety_alert = None

    def get_plan_safety_alert(self) -> dict[str, Any] | None:
        return dict(self.plan_safety_alert) if isinstance(self.plan_safety_alert, dict) else None

    def _empty_runtime_recovery(self) -> dict[str, Any]:
        return {
            "product_name": self.agent_name,
            "product_jid": str(self.jid),
            "replan_mode": str(self.replan_mode or "llm").strip().lower() or "llm",
            "status": "idle",
            "resolution_class": "none",
            "trigger": "",
            "failed_task_id": "",
            "message": "No active runtime recovery session.",
            "violated_rules": [],
            "witness_count": 0,
            "attempts_used": 0,
            "attempts_max": int(self._runtime_repair_max_attempts),
            "used_llm_bridge": False,
            "operator_guidance": "",
            "history": [],
            "updated_at_utc": self._utc_now_iso(),
        }

    def _sync_runtime_repair_state(self) -> None:
        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        if status in {"des_search", "llm_bridge", "validating"}:
            self.runtime_repair_state = "repairing"
        elif status == "human_required":
            self.runtime_repair_state = "paused_after_failure"
        else:
            self.runtime_repair_state = "idle"

    def _set_runtime_recovery(
        self,
        *,
        reset: bool = False,
        status: str | None = None,
        resolution_class: str | None = None,
        trigger: str | None = None,
        failed_task_id: str | None = None,
        message: str | None = None,
        attempts_used: int | None = None,
        attempts_max: int | None = None,
        used_llm_bridge: bool | None = None,
        operator_guidance: str | None = None,
        violations: list[dict[str, Any]] | None = None,
        append_history: bool = False,
        history_message: str | None = None,
    ) -> dict[str, Any]:
        current = self._empty_runtime_recovery() if reset else deepcopy(self.runtime_recovery)
        current["product_name"] = self.agent_name
        current["product_jid"] = str(self.jid)
        current["replan_mode"] = str(self.replan_mode or "llm").strip().lower() or "llm"
        current["attempts_max"] = int(self._runtime_repair_max_attempts)

        if status is not None:
            current["status"] = str(status or "idle").strip() or "idle"
        if resolution_class is not None:
            current["resolution_class"] = str(resolution_class or "none").strip() or "none"
        if trigger is not None:
            current["trigger"] = str(trigger).strip()
        if failed_task_id is not None:
            current["failed_task_id"] = str(failed_task_id).strip()
        if message is not None:
            current["message"] = str(message).strip() or current.get("message", "")
        if attempts_used is not None:
            current["attempts_used"] = max(0, int(attempts_used))
        if attempts_max is not None:
            current["attempts_max"] = max(0, int(attempts_max))
        if used_llm_bridge is not None:
            current["used_llm_bridge"] = bool(used_llm_bridge)
        if operator_guidance is not None:
            current["operator_guidance"] = str(operator_guidance).strip()
        if violations is not None:
            violated_rules, witness_count = self._violation_summary(violations)
            current["violated_rules"] = violated_rules
            current["witness_count"] = witness_count

        event_message = str(history_message if history_message is not None else message or "").strip()
        if append_history and event_message:
            history = list(current.get("history") or [])
            history.append(
                {
                    "timestamp": self._utc_now_iso(),
                    "status": current.get("status", ""),
                    "message": event_message,
                }
            )
            current["history"] = history[-12:]

        current["updated_at_utc"] = self._utc_now_iso()
        self.runtime_recovery = current
        self._sync_runtime_repair_state()
        return deepcopy(current)

    def _clear_runtime_recovery(self) -> None:
        self.runtime_recovery = self._empty_runtime_recovery()
        self._runtime_recovery_context = {}
        self._sync_runtime_repair_state()

    def _runtime_recovery_blocks_execution(self) -> bool:
        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        return status not in {"", "idle", "resolved"}

    def get_runtime_recovery(self) -> dict[str, Any]:
        return deepcopy(self.runtime_recovery) if isinstance(self.runtime_recovery, dict) else self._empty_runtime_recovery()

    def _set_kickoff_result(
        self,
        *,
        success: bool,
        message: str,
        retries_used: int,
        retries_max: int,
        violations: list[dict[str, Any]] | None = None,
        alert: dict[str, Any] | None = None,
    ) -> None:
        violated_rules, witness_count = self._violation_summary(violations)
        self.kickoff_result = {
            "success": bool(success),
            "message": str(message),
            "retries_used": int(retries_used),
            "retries_max": int(retries_max),
            "violated_rules": violated_rules,
            "witness_count": witness_count,
            "updated_at_utc": self._utc_now_iso(),
            "product_name": self.agent_name,
            "product_jid": str(self.jid),
            "stage": "kickoff",
            "alert": dict(alert) if isinstance(alert, dict) else None,
        }
        if not self._kickoff_result_event.is_set():
            self._kickoff_result_event.set()

    async def wait_for_kickoff_result(self, timeout: float | None = None) -> dict[str, Any]:
        if not self._kickoff_result_event.is_set():
            try:
                if timeout is None:
                    await self._kickoff_result_event.wait()
                else:
                    await asyncio.wait_for(self._kickoff_result_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                message = (
                    f"{self.agent_name}: startup plan validation timed out after {timeout:.1f}s."
                )
                return {
                    "success": False,
                    "message": message,
                    "retries_used": 0,
                    "retries_max": 0,
                    "violated_rules": [],
                    "witness_count": 0,
                    "updated_at_utc": self._utc_now_iso(),
                    "product_name": self.agent_name,
                    "product_jid": str(self.jid),
                    "stage": "kickoff",
                    "alert": self._build_plan_safety_alert(
                        stage="kickoff",
                        message=message,
                        retries_used=0,
                        retries_max=0,
                        violations=[],
                    ),
                }
        return dict(self.kickoff_result)



    def _persist_plan_snapshot(self) -> None:
        """Persist the current process planner graph (DAG nodes only) to disk."""
        if not self.plan_path:
            return

        try:
            import json

            self.plan_path.parent.mkdir(parents=True, exist_ok=True)
            with self.plan_path.open("w", encoding="utf-8") as f:
                json.dump({"nodes": self.process_planner.nodes}, f, indent=2)

            self.logger.debug(f"[Product] Saved plan to {self.plan_path.resolve()}")
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
                "runtime_repair_state": self.runtime_repair_state,
                "runtime_recovery": self.runtime_recovery,
                "plan_safety_alert": self.plan_safety_alert,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.product_state_path.parent.mkdir(parents=True, exist_ok=True)
            with self.product_state_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            self.logger.debug(f"[Product] Saved product state to {self.product_state_path.resolve()}")
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

            self.logger.debug(f"[Product] Saved resource state to {self.resource_state_path.resolve()}")
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
            runtime_repair_state = str(product_state.get("runtime_repair_state", "")).strip()
            if runtime_repair_state:
                self.runtime_repair_state = runtime_repair_state
            runtime_recovery = product_state.get("runtime_recovery")
            if isinstance(runtime_recovery, dict):
                self.runtime_recovery = deepcopy(runtime_recovery)
                self._sync_runtime_repair_state()
            alert = product_state.get("plan_safety_alert")
            self.plan_safety_alert = dict(alert) if isinstance(alert, dict) else None
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
                self.logger.debug(f"[Product] Current working directory: {cwd}")
                p = Path(self.product_specification_file)
                self.logger.debug(f"[Product] Attempting to open: {p.resolve()}")
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

    def _candidate_task_ids_from_violations(
        self,
        violations: list[dict[str, Any]] | None,
    ) -> set[str]:
        candidate_ids: set[str] = set()
        for violation in violations or []:
            if not isinstance(violation, dict):
                continue
            for key in ("failed_task_id", "task_id"):
                task_id = violation.get(key)
                if task_id:
                    candidate_ids.add(str(task_id))
            for key in ("affected_task_ids", "blocked_task_ids", "unreachable_task_ids"):
                values = violation.get(key)
                if isinstance(values, (list, tuple, set)):
                    candidate_ids.update(str(value) for value in values if value)
        return candidate_ids

    async def _send_runtime_plan_validation_check(self) -> None:
        self.process_planner.compile_global_fsa()
        self.process_planner.save_global_fsa(self.global_fsa_path)

        payload = self._build_plan_validation_payload()
        self._ensure_plan_result_inbox()
        msg_check = Message(to=self.cca_jid)
        msg_check.set_metadata("type", "plan_safety_check")
        msg_check.body = json.dumps(payload)
        await self.send(msg_check)

        self.logger.info(
            "[Product] Recompiled plan FSA after runtime recovery and sent plan_safety_check to CCA."
        )

    async def _run_des_runtime_recovery_attempt(
        self,
        *,
        violations: list[dict[str, Any]],
        trigger: str,
        failed_task_id: str,
        system_coordination_state: dict | None = None,
        reset_attempts: bool = False,
        history_message: str | None = None,
    ) -> dict[str, Any]:
        if reset_attempts:
            self._runtime_repair_fail_streak = 0

        attempt_number = self._runtime_repair_fail_streak + 1
        self._runtime_repair_fail_streak = attempt_number
        self._runtime_recovery_context = {
            "trigger": str(trigger or "").strip(),
            "failed_task_id": str(failed_task_id or "").strip(),
            "violations": deepcopy(list(violations or [])),
            "system_coordination_state": deepcopy(system_coordination_state or {}),
        }
        self._set_runtime_recovery(
            status="des_search",
            resolution_class="none",
            trigger=trigger,
            failed_task_id=failed_task_id,
            message=(
                f"Running DES runtime recovery attempt "
                f"{attempt_number}/{self._runtime_repair_max_attempts}."
            ),
            attempts_used=attempt_number,
            attempts_max=self._runtime_repair_max_attempts,
            used_llm_bridge=False,
            violations=violations,
            append_history=True,
            history_message=history_message or (
                f"DES runtime recovery attempt {attempt_number}/{self._runtime_repair_max_attempts} started."
            ),
        )

        self._runtime_repair_inflight = True
        try:
            result = await self.process_planner.replan_with_feedback_online(
                violations,
                system_coordination_state=system_coordination_state,
            )
            if not isinstance(result, dict):
                result = {}

            plan_changed = bool(result.get("plan_changed", False))
            used_llm_bridge = bool(result.get("used_llm_bridge", False))
            human_required = bool(result.get("human_required", False))
            base_message = str(result.get("message", "")).strip()
            bridge_summary = result.get("bridge_summary") or []
            if used_llm_bridge:
                bridge_text = ", ".join(str(item) for item in bridge_summary if item) or "bridge step(s)"
                self._set_runtime_recovery(
                    status="llm_bridge",
                    trigger=trigger,
                    failed_task_id=failed_task_id,
                    message=base_message or "DES recovery used the LLM bridge to extend the model.",
                    attempts_used=attempt_number,
                    attempts_max=self._runtime_repair_max_attempts,
                    used_llm_bridge=True,
                    violations=violations,
                    append_history=True,
                    history_message=f"LLM bridge added {bridge_text}.",
                )

            if human_required or not plan_changed:
                message = (
                    base_message
                    or "DES recovery could not produce a valid continuation. Human intervention required."
                )
                recovery = self._set_runtime_recovery(
                    status="human_required",
                    resolution_class="human_required",
                    trigger=trigger,
                    failed_task_id=failed_task_id,
                    message=message,
                    attempts_used=attempt_number,
                    attempts_max=self._runtime_repair_max_attempts,
                    used_llm_bridge=used_llm_bridge,
                    violations=violations,
                    append_history=True,
                    history_message=message,
                )
                self._set_plan_safety_alert(
                    stage="runtime",
                    message=message,
                    retries_used=attempt_number,
                    retries_max=self._runtime_repair_max_attempts,
                    violations=violations,
                    paused=True,
                )
                await asyncio.to_thread(self._persist_product_state)
                return recovery

            reactivated = self._reactivate_blocked_tasks(
                candidate_task_ids=self._candidate_task_ids_from_violations(violations) or None
            )
            if reactivated:
                self.logger.info(
                    "[Product] Reactivated %d blocked task(s) to pending after DES recovery.",
                    reactivated,
                )

            validation_message = (
                "DES recovery candidate generated; validating updated plan."
                if not used_llm_bridge
                else "DES + LLM bridge candidate generated; validating updated plan."
            )
            recovery = self._set_runtime_recovery(
                status="validating",
                resolution_class="none",
                trigger=trigger,
                failed_task_id=failed_task_id,
                message=validation_message,
                attempts_used=attempt_number,
                attempts_max=self._runtime_repair_max_attempts,
                used_llm_bridge=used_llm_bridge,
                violations=violations,
                append_history=True,
                history_message=validation_message,
            )
            self._clear_plan_safety_alert()
            await self._send_runtime_plan_validation_check()
            await asyncio.to_thread(self._persist_plan_snapshot)
            await asyncio.to_thread(self._persist_product_state)
            await asyncio.to_thread(self._persist_resource_state)
            return recovery
        except Exception as exc:
            self.logger.exception("[Product] DES runtime recovery attempt failed.")
            message = (
                f"{self.agent_name}: DES runtime recovery attempt "
                f"{attempt_number}/{self._runtime_repair_max_attempts} failed ({exc})."
            )
            recovery = self._set_runtime_recovery(
                status="human_required",
                resolution_class="human_required",
                trigger=trigger,
                failed_task_id=failed_task_id,
                message=message,
                attempts_used=attempt_number,
                attempts_max=self._runtime_repair_max_attempts,
                violations=violations,
                append_history=True,
                history_message=message,
            )
            self._set_plan_safety_alert(
                stage="runtime",
                message=message,
                retries_used=attempt_number,
                retries_max=self._runtime_repair_max_attempts,
                violations=violations,
                paused=True,
            )
            await asyncio.to_thread(self._persist_product_state)
            return recovery
        finally:
            self._runtime_repair_inflight = False

    async def _handle_runtime_des_replan_request(
        self,
        *,
        reason: str,
        failed_task_id: str,
        violations: list[dict[str, Any]],
        system_coordination_state: dict | None = None,
    ) -> dict[str, Any]:
        current_status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        if current_status in {"des_search", "llm_bridge", "validating", "human_required"}:
            self.logger.warning(
                "[Product] Runtime recovery already active for %s; ignoring duplicate replan request.",
                self.runtime_recovery.get("failed_task_id") or failed_task_id,
            )
            return self.get_runtime_recovery()

        self._clear_plan_safety_alert()
        self._set_runtime_recovery(
            reset=True,
            status="des_search",
            resolution_class="none",
            trigger=reason,
            failed_task_id=failed_task_id,
            message=f"Runtime DES recovery triggered by {reason}.",
            attempts_used=0,
            attempts_max=self._runtime_repair_max_attempts,
            violations=violations,
            append_history=True,
            history_message=f"Runtime DES recovery triggered by {reason}.",
        )
        return await self._run_des_runtime_recovery_attempt(
            violations=violations,
            trigger=reason,
            failed_task_id=failed_task_id,
            system_coordination_state=system_coordination_state,
            reset_attempts=True,
        )

    async def _handle_runtime_plan_validation_result(
        self,
        *,
        ok: bool,
        violations: list[dict[str, Any]],
    ) -> bool:
        if str(self.replan_mode or "llm").strip().lower() != "des":
            return False

        status = str(self.runtime_recovery.get("status", "idle") or "idle").strip().lower()
        if not self._runtime_recovery_context and status in {"", "idle", "resolved"}:
            return False

        if ok:
            resolution_class = (
                "des_with_llm_bridge"
                if bool(self.runtime_recovery.get("used_llm_bridge", False))
                else "des_only"
            )
            attempts_used = self._runtime_repair_fail_streak
            self._runtime_repair_fail_streak = 0
            self._set_runtime_recovery(
                status="resolved",
                resolution_class=resolution_class,
                message="Plan validation passed; runtime recovery resolved.",
                attempts_used=attempts_used,
                used_llm_bridge=bool(self.runtime_recovery.get("used_llm_bridge", False)),
                violations=[],
                append_history=True,
                history_message="Plan validation passed; runtime recovery resolved.",
            )
            self._runtime_recovery_context = {}
            self._clear_plan_safety_alert()
            await asyncio.to_thread(self._persist_product_state)
            return True

        if self._runtime_repair_inflight:
            self.logger.warning(
                "[Product] Runtime plan validation failed while DES recovery is already running; ignoring duplicate result."
            )
            return True

        if not self._runtime_recovery_context:
            message = (
                f"{self.agent_name}: plan validation failed without an active DES recovery context; "
                "execution paused for human intervention."
            )
            self._set_runtime_recovery(
                reset=True,
                status="human_required",
                resolution_class="human_required",
                message=message,
                attempts_used=self._runtime_repair_fail_streak,
                attempts_max=self._runtime_repair_max_attempts,
                violations=violations,
                append_history=True,
                history_message=message,
            )
            self._set_plan_safety_alert(
                stage="runtime",
                message=message,
                retries_used=self._runtime_repair_fail_streak,
                retries_max=self._runtime_repair_max_attempts,
                violations=violations,
                paused=True,
            )
            await asyncio.to_thread(self._persist_product_state)
            return True

        if self._runtime_repair_fail_streak >= self._runtime_repair_max_attempts:
            message = (
                f"{self.agent_name}: runtime plan validation still fails after "
                f"{self._runtime_repair_fail_streak}/{self._runtime_repair_max_attempts} "
                "DES recovery attempt(s); execution paused."
            )
            self._set_runtime_recovery(
                status="human_required",
                resolution_class="human_required",
                trigger=str(self._runtime_recovery_context.get("trigger", "")),
                failed_task_id=str(self._runtime_recovery_context.get("failed_task_id", "")),
                message=message,
                attempts_used=self._runtime_repair_fail_streak,
                attempts_max=self._runtime_repair_max_attempts,
                used_llm_bridge=bool(self.runtime_recovery.get("used_llm_bridge", False)),
                violations=violations,
                append_history=True,
                history_message=message,
            )
            self._set_plan_safety_alert(
                stage="runtime",
                message=message,
                retries_used=self._runtime_repair_fail_streak,
                retries_max=self._runtime_repair_max_attempts,
                violations=violations,
                paused=True,
            )
            await asyncio.to_thread(self._persist_product_state)
            return True

        self._runtime_recovery_context["violations"] = deepcopy(list(violations or []))
        await self._run_des_runtime_recovery_attempt(
            violations=violations,
            trigger=str(self._runtime_recovery_context.get("trigger", "")),
            failed_task_id=str(self._runtime_recovery_context.get("failed_task_id", "")),
            system_coordination_state=dict(self._runtime_recovery_context.get("system_coordination_state") or {}),
            reset_attempts=False,
            history_message=(
                f"Plan validation failed; rerunning DES recovery attempt "
                f"{self._runtime_repair_fail_streak + 1}/{self._runtime_repair_max_attempts}."
            ),
        )
        return True

    async def submit_runtime_recovery_guidance(self, message: str) -> dict[str, Any]:
        guidance = str(message or "").strip()
        if not guidance:
            raise ValueError("operator guidance is empty")
        recovery = self._set_runtime_recovery(
            operator_guidance=guidance,
            append_history=True,
            history_message=f"Operator guidance recorded: {guidance}",
        )
        await asyncio.to_thread(self._persist_product_state)
        return recovery

    async def retry_runtime_recovery_des(self) -> dict[str, Any]:
        if str(self.replan_mode or "llm").strip().lower() != "des":
            raise RuntimeError("runtime DES retry is only available when replan_mode=des")
        if not self._runtime_recovery_context:
            raise RuntimeError("no active runtime DES recovery context is available")
        trigger = str(self._runtime_recovery_context.get("trigger", "") or "operator_retry")
        failed_task_id = str(self._runtime_recovery_context.get("failed_task_id", "")).strip()
        violations = deepcopy(list(self._runtime_recovery_context.get("violations") or []))
        system_coordination_state = dict(self._runtime_recovery_context.get("system_coordination_state") or {})
        self._clear_plan_safety_alert()
        return await self._run_des_runtime_recovery_attempt(
            violations=violations,
            trigger=trigger,
            failed_task_id=failed_task_id,
            system_coordination_state=system_coordination_state,
            reset_attempts=True,
            history_message="Operator requested a DES retry for the active runtime recovery session.",
        )

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
                            f"{agent.agent_name}: loaded verified plan set passed startup safety validation."
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
            if str(agent.replan_mode or "llm").strip().lower() == "des":
                await agent._handle_runtime_des_replan_request(
                    reason=str(reason),
                    failed_task_id=str(failed_task_id),
                    violations=violations,
                    system_coordination_state=system_coordination_state,
                )
                await asyncio.to_thread(agent._persist_product_state)
                return

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
                await asyncio.sleep(0.5)
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

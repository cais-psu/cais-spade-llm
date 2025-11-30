"""High-level product agent that reads build instructions and instructs a resource agent."""

from __future__ import annotations

import asyncio
import json
import os, uuid
from pathlib import Path
from typing import Any, Iterable, Optional

from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message
from spade.template import Template

from agents.shared_information.llm_agent import LlmAgent
from agents.intelligent_product.process_planner import ProcessPlanner


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
        safety_file: Optional[str] = None,
        instruction_override: Optional[str] = None,
        cca_jid: Optional[str] = None,
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
        self.product_specification_file = product_specification_file
        self.safety_file = safety_file
        # Manual instruction text provided at runtime overrides any file read.
        self.instruction_override = instruction_override

        # Planner scaffolding
        base_plan_dir = Path("cais_spade_llm/plan")
        # For now: requirements file (NL → structured requirements)
        self.structured_requirements_path = base_plan_dir / f"{name}_requirements.json"
        # Reserved for later: full DAG task plan (requirements → task graph)
        self.plan_path = base_plan_dir / f"{name}_plan.json"

        planner_resources = self._match_resource_objects(
            self._resource_agent_refs, self.resource_jids
        )
        self.process_planner = ProcessPlanner(self, planner_resources)
        
        #keep resolved resource agents on the ProductAgent for caps overview
        self.resource_agents = planner_resources

        # Simple in-memory map of task_id -> latest status string so UI/debug tooling can query progress.
        self.task_states: dict[str, str] = {}

        self.logger.info(f"ProductAgent '{name}' initialized.")

    # ------------------------------------------------------------------ #
    # Persistence helper
    # ------------------------------------------------------------------ #
    def _build_plan_validation_payload(self, nodes: list[dict[str, Any]]) -> dict:
        """
        Build the JSON payload to send to the CCA for offline plan validation.
        This does NOT send anything; sending is done from behaviours.
        """
        return {
            "plan": {"nodes": nodes},
            "product_jid": str(self.jid),
        }

    def _persist_plan_snapshot(self) -> None:
        """Persist the current process planner graph to disk."""
        if not self.plan_path:
            return

        try:
            self.process_planner.save(self.plan_path)
        except Exception:
            self.logger.exception("[Product] Failed to persist plan snapshot.")

    # --------------------------------------------------------------------- #
    # SPADE lifecycle
    # --------------------------------------------------------------------- #

    async def setup(self):
        await super().setup()

        # Kickoff behaviour (runs once) to build the plan
        self.add_behaviour(self._Kickoff())

        # ACK inbox
        t_ack = Template()
        t_ack.set_metadata("type", "ack")
        self.add_behaviour(self._AckInbox(), t_ack)

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

    async def _build_plan(self, requirement_text: str, safety_text: str = ""):
        """
        Build structured requirements → save them → expand into task DAG → save DAG.
        """
        # 1. NL → structured requirements
        await self.process_planner.build_high_level(requirement_text)

        # (NEW) Save only structured requirements before expansion
        self.process_planner.save(self.structured_requirements_path)

        # 2. Expand structured requirements → DAG tasks
        await self.process_planner.expand_requirements_to_tasks(safety_text=safety_text)

        # 3. Save DAG plan separately
        self.process_planner.save(self.plan_path)

        return self.process_planner.nodes


    # --------------------------------------------------------------------- #
    # Behaviours
    # --------------------------------------------------------------------- #
    class _Kickoff(OneShotBehaviour):
        async def run(self):
            agent: "ProductAgent" = self.agent

            # Initial Plan Build
            instruction = agent._extract_requirement_text()
            safety_text = agent._read_safety_text()

            dag_nodes = await agent._build_plan(instruction, safety_text)
            
            # Retry Loop for Safety
            max_retries = 3
            attempt = 0
            
            while attempt < max_retries:
                attempt += 1
                agent.logger.info(f"[Product] Validating Plan (Attempt {attempt}/{max_retries})...")
                
                # 1. Send to CCA
                payload = agent._build_plan_validation_payload(agent.process_planner.nodes)
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
                    agent.add_behaviour(agent._PlanExecutor())
                    return # Exit Kickoff successfully
                
                # 3. Handle Failure
                agent.logger.warning(f"[Product] Plan FAILED safety check ({len(violations)} violations). Triggering Re-plan...")
                
                # Call the new Re-planning method
                await agent.process_planner.replan_with_feedback(violations)
                
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
            for node in agent.process_planner.nodes:
                if node.get("id") == task_id:
                    # Map RA status → planner status; for now use it directly
                    if node.get("status") != status:
                        node["status"] = status
                        updated_node = True
                    break

            if updated_node:
                agent._persist_plan_snapshot()

            agent.logger.info(
                f"[Product] ACK ({task_id}) status='{status}' from={msg.sender}"
            )

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

            # Build the instruction for the RobotAgent from the DAG node
            instruction = {
                "function_name": task_node.get("function_name"),
                "params": task_node.get("params", {}),
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

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional, Iterable

from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message

from agents.shared_information.llm_agent import LlmAgent
from agents.central_controller.safety_logic import SafetyLogic
from agents.central_controller.safety_monitor import SafetyMonitor

from prompts import build_safety_replan_prompt

class CentralControllerAgent(LlmAgent):
    """
    Central Controller Agent (CCA)

    - Loads NL safety requirements and builds structured rules + LTLf via SafetyLogic.
    - Builds per-rule DFAs and wraps them in a SafetyMonitor.
    - Runs a _Monitor behaviour that listens for 'resource_event' messages
      from ResourceAgents and performs runtime safety checks.
    """

    agent_role = "controller"

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str,
        resource_agents: Optional[Iterable[Any]] = None,
        safety_file: str | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(jid, password, name=name, agent_role="controller", **kw)

        self.agent_name = name
        self.safety_file = Path(safety_file) if safety_file else None
        # Resource agents (used for grounding capability overviews in safety prompts)
        self.resource_agents = list(resource_agents or [])

        # Where the safety rules and logic will be saved (like ProductAgent plan.json)
        base_safety_dir = Path("cais_spade_llm/safety")
        self.safety_logic_path = base_safety_dir / f"{name}_safety_logic.json"

        # Safety logic scaffolding
        self.safety_logic: Optional[SafetyLogic] = None
        if self.safety_file:
            self.safety_logic = SafetyLogic(self, self.safety_file)

        # In-memory safety rules (for later DFA/LTLf integration)
        self.safety_rules: list[dict[str, Any]] = []

        # Runtime safety monitor (wraps all per-rule DFAs)
        self.safety_monitor: Optional[SafetyMonitor] = None

        # Track which APs are currently considered "running" across resources.
        # This lets the DFA see simultaneous actions: σ = running_aps ∪ candidate_aps.
        self.running_aps: set[str] = set()

        self.logger.info(
            "CentralControllerAgent '%s' initialized. safety_file=%s",
            name,
            str(self.safety_file) if self.safety_file else "(none)",
        )

    # ------------------------------------------------------------------ #
    # SPADE setup
    # ------------------------------------------------------------------ #
    async def setup(self) -> None:
        await super().setup()
        self.logger.info("[CCA] setup completed.")

        # One-shot init behaviour (build safety rules once at startup)
        self.add_behaviour(self._InitCCA())

    # ------------------------------------------------------------------ #
    # High-level safety-violation logic (NO messaging here)
    # ------------------------------------------------------------------ #
    async def handle_safety_violation(
        self,
        *,
        task_id: str,
        resource_jid: str,
        function_name: str,
        params: dict[str, Any],
        candidate_aps: list[str],
        running_aps: list[str],
        violated_rule: str | None,
    ) -> None:
        """
        High-level safety-violation handler.

        This method is intentionally *pure* with respect to messaging:
        - NO calls to .send() here.
        - Messaging is done in the _Monitor behaviour.
        - Here you can:
            * build replan context
            * call the LLM
            * update any internal plan copies
            * decide when/how to "release" blocked tasks logically
        """
        self.logger.info(
            "[CCA] (handle_safety_violation) task=%s rule=%s candidate_aps=%s running_aps=%s",
            task_id,
            violated_rule,
            candidate_aps,
            running_aps,
        )

        # ------------------------------------------------------------------
        # 1) Figure out which product this task belongs to
        # ------------------------------------------------------------------
        product_jid = params.get("product_jid")
        if not product_jid:
            self.logger.warning(
                "[CCA] safety violation for task=%s but params.product_jid is missing.",
                task_id,
            )

        # ------------------------------------------------------------------
        # 2) Load the current plan DAG for that product
        # ------------------------------------------------------------------
        plan_snapshot = await self._load_current_plan_for_replan(
            product_jid=product_jid,
        )

        # ------------------------------------------------------------------
        # 3) Safety-logic snapshot (rules + the violated one)
        # ------------------------------------------------------------------
        violated_rule_detail = None
        if violated_rule and self.safety_rules:
            violated_rule_detail = next(
                (r for r in self.safety_rules if r.get("id") == violated_rule),
                None,
            )

        safety_logic_snapshot: dict[str, Any] = {
            "rules": self.safety_rules,
            "violated_rule_detail": violated_rule_detail,
            "logic_file": str(self.safety_logic_path),
        }

        # ------------------------------------------------------------------
        # 4) Build violation context for the LLM
        # ------------------------------------------------------------------
        violation_context = {
            "task": {
                "task_id": task_id,
                "resource_jid": resource_jid,
                "function_name": function_name,
                "params": params,
            },
            "safety": {
                "violated_rule": violated_rule,
                "running_aps": running_aps,
                "candidate_aps": candidate_aps,
            },
            "plan": plan_snapshot,
            "safety_logic": safety_logic_snapshot,
        }

        self.logger.debug(
            "[CCA] (handle_safety_violation) Violation context prepared keys=%s",
            list(violation_context.keys()),
        )

        # ------------------------------------------------------------------
        # 5) Ask the LLM to repair the plan
        # ------------------------------------------------------------------
        replanned_dag = await self._llm_replan_on_violation(violation_context)
        print(replanned_dag)
        if replanned_dag is None:
            self.logger.warning(
                "[CCA] LLM replanning failed or returned None; keeping original plan."
            )
            return

        # ------------------------------------------------------------------
        # 6) Apply the new DAG and logically release the blocked task
        # ------------------------------------------------------------------
        # await self._apply_replanned_dag(
        #     replanned_dag=replanned_dag,
        #     product_jid=product_jid,
        # )

        # NOTE: you can add a more sophisticated unblocking strategy here
        # (e.g., only unblocking certain successors, etc.)
        # await self._release_task_block(task_id=task_id)

    async def _load_current_plan_for_replan(
        self,
        *,
        product_jid: str | None,
    ) -> Optional[dict[str, Any]]:
        """
        Load the plan DAG for the ProductAgent that owns this task.

        We assume:
          - ProductAgent name == localpart of product_jid
          - Plan file path: cais_spade_llm/plan/{name}_plan.json
        """
        if not product_jid:
            self.logger.warning("[CCA] No product_jid in params; cannot load plan.")
            return None

        localpart = product_jid.split("@", 1)[0]
        base_plan_dir = Path("cais_spade_llm/plan")
        plan_path = base_plan_dir / f"{localpart}_plan.json"

        if not plan_path.exists():
            self.logger.warning("[CCA] Plan file not found for %s: %s", product_jid, plan_path)
            return None

        try:
            with plan_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if "nodes" not in data:
                self.logger.warning("[CCA] Plan file %s has no 'nodes' key.", plan_path)
            return data
        except Exception:
            self.logger.exception("[CCA] Failed to load plan for %s from %s", product_jid, plan_path)
            return None




    async def _llm_replan_on_violation(self, context: dict) -> dict | None:
        """Ask the LLM for a patch that fixes only the violated task."""
        
        # Build prompt from prompts.py
        prompt = build_safety_replan_prompt(context)

        # Call LLM
        raw = await self.ask_llm(
            prompt=prompt,
            with_functions=False,
            temperature=0.0,
        )

        # Parse JSON output
        try:
            patch = json.loads(raw)
        except:
            self.logger.warning("[CCA] LLM replan: invalid JSON")
            return None

        # Validate minimal structure
        if not isinstance(patch, dict):
            return None
        if "target_task_id" not in patch or "updated_task" not in patch:
            return None

        return patch


    async def _apply_replanned_dag(self, replanned_dag: dict[str, Any]) -> None:
        """
        Placeholder: Apply the replanned DAG into your plan storage.

        This might talk to ProductAgents, update a shared plan store, etc.
        """
        self.logger.debug("[CCA] (APPLY REPLAN PLACEHOLDER) Got DAG with keys=%s",
                          list(replanned_dag.keys()))

    async def _release_task_block(self, *, task_id: str) -> None:
        """
        Placeholder: Release a logical block on a task inside the plan.

        For now, this is a stub; real plan updates happen on the ProductAgent
        when it receives ACK messages from CCA.
        """
        self.logger.debug("[CCA] (RELEASE BLOCK PLACEHOLDER) task_id=%s", task_id)

    # ------------------------------------------------------------------ #
    # Helpers for modularizing _Monitor
    # ------------------------------------------------------------------ #
    def _parse_resource_event(self, msg) -> Optional[dict[str, Any]]:
        """Parse and validate a resource_event message body into a dict."""
        try:
            data = json.loads(msg.body or "{}")
        except Exception:
            self.logger.error("[CCA] Malformed resource_event body.")
            return None

        task_id       = data.get("task_id")
        resource_jid  = data.get("resource_jid")
        function_name = data.get("function_name")
        params        = data.get("params") or {}
        status        = data.get("status") or "running"

        if not (resource_jid and function_name):
            self.logger.warning(
                "[CCA] resource_event missing resource_jid or function_name."
            )
            return None

        return {
            "task_id": task_id,
            "resource_jid": resource_jid,
            "function_name": function_name,
            "params": params,
            "status": status,
        }

    def _compute_candidate_aps(self, event: dict[str, Any]) -> list[str]:
        """Use SafetyMonitor to map a resource event to AP labels."""
        if not self.safety_monitor:
            return []

        return self.safety_monitor._map_task_to_aps(
            resource_jid=event["resource_jid"],
            function_name=event["function_name"],
            params=event["params"],
        )

    async def _handle_start_event_logic(
        self,
        *,
        event: dict[str, Any],
        candidate_aps: list[str],
    ) -> tuple[bool, dict[str, Any]]:
        """
        Handle 'running' (start) events logically.

        Returns:
          (allowed, info)
            - allowed == True  → APs already added into running_aps, info empty-ish
            - allowed == False → violation, info contains:
                  {
                    "violated_rule": ...,
                    "running_snapshot": [...],
                    "product_jid": str | None,
                  }
        """
        running_snapshot = list(self.running_aps)

        allowed, meta = self.safety_monitor.check(
            running_aps=running_snapshot,
            candidate_aps=candidate_aps,
        )

        if allowed:
            # DFA accepted → mark these APs as running
            self.running_aps.update(candidate_aps)
            return True, {
                "running_snapshot": running_snapshot,
            }

        # Violation
        info = {
            "violated_rule": meta.get("violated_rule"),
            "running_snapshot": running_snapshot,
            "product_jid": event["params"].get("product_jid"),
        }
        return False, info

    def _handle_finish_event_logic(
        self,
        *,
        event: dict[str, Any],
        candidate_aps: list[str],
    ) -> None:
        """Handle non-running (finish) events logically: remove APs."""
        for ap in candidate_aps:
            if ap in self.running_aps:
                self.running_aps.discard(ap)

    # ------------------------------------------------------------------ #
    # Behaviours
    # ------------------------------------------------------------------ #
    class _Monitor(CyclicBehaviour):
        """
        Behaviour that receives 'resource_event' messages from ResourceAgents
        and applies the SafetyMonitor. Messaging stays here; logic is delegated
        to CentralControllerAgent helper methods.
        """

        async def run(self) -> None:
            agent: "CentralControllerAgent" = self.agent  # type: ignore

            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            if msg.metadata.get("type") != "resource_event":
                return

            # 1) Parse event
            event = agent._parse_resource_event(msg)
            if not event:
                return

            task_id = event["task_id"]
            status  = event["status"]

            # 2) Map to AP labels
            candidate_aps = agent._compute_candidate_aps(event)
            if not candidate_aps:
                agent.logger.info(
                    "[CCA] No AP mapping for task=%s (%s, %s, status=%s); skipping.",
                    task_id,
                    event["resource_jid"],
                    event["function_name"],
                    status,
                )
                return

            # 3) START event → run safety
            if status == "safety_check":
                allowed, info = await agent._handle_start_event_logic(
                    event=event,
                    candidate_aps=candidate_aps,
                )
                resource_jid = event["resource_jid"]

                if not allowed:
                    violated_rule    = info.get("violated_rule")
                    running_snapshot = info.get("running_snapshot", [])

                    agent.logger.warning(
                        "[CCA] SAFETY VIOLATION: task=%s rule=%s aps=%s (running=%s)",
                        task_id,
                        violated_rule,
                        candidate_aps,
                        running_snapshot,
                    )

                    # 3a) Tell the ResourceAgent: DO NOT START (block)
                    decision = {
                        "task_id": task_id,
                        "decision": "block",
                        "reason": violated_rule,
                    }
                    ra_msg = Message(to=resource_jid)
                    ra_msg.set_metadata("type", "safety_decision")
                    ra_msg.body = json.dumps(decision)
                    await self.send(ra_msg)
                    agent.logger.info(
                        "[CCA] Sent safety_decision=block for task=%s to %s",
                        task_id, resource_jid
                    )

                    # 3b) LLM replanning hook
                    try:
                        await agent.handle_safety_violation(
                            task_id=task_id or "?",
                            resource_jid=resource_jid,
                            function_name=event["function_name"],
                            params=event["params"],
                            candidate_aps=candidate_aps,
                            running_aps=running_snapshot,
                            violated_rule=violated_rule,
                        )
                    except Exception:
                        agent.logger.exception(
                            "[CCA] handle_safety_violation (no-send) failed."
                        )

                    return

                # allowed == True
                agent.logger.info(
                    "[CCA] Safety OK: task=%s aps=%s (now running=%s)",
                    task_id,
                    candidate_aps,
                    sorted(agent.running_aps),
                )

                # Tell the RA it's allowed to start
                decision = {
                    "task_id": task_id,
                    "decision": "allow",
                }
                ra_msg = Message(to=resource_jid)
                ra_msg.set_metadata("type", "safety_decision")
                ra_msg.body = json.dumps(decision)
                await self.send(ra_msg)
                agent.logger.info(
                    "[CCA] Sent safety_decision=allow for task=%s to %s",
                    task_id, resource_jid
                )
                return

            # 3b) RUNNING event (informational only)
            if status == "running":
                agent.logger.info(
                    "[CCA] Task now running: task=%s aps=%s (running=%s)",
                    task_id,
                    candidate_aps,
                    sorted(agent.running_aps),
                )
                # No DFA finish here; APs should already be in running_aps
                return

            # 4) FINISH event → remove APs
            agent._handle_finish_event_logic(
                event=event,
                candidate_aps=candidate_aps,
            )

            agent.logger.info(
                "[CCA] Task finished: task=%s status=%s removed_aps=%s (now running=%s)",
                task_id,
                status,
                candidate_aps,
                sorted(agent.running_aps),
            )

    class _InitCCA(OneShotBehaviour):
        async def run(self) -> None:
            agent: "CentralControllerAgent" = self.agent  # type: ignore

            safety_logic = agent.safety_logic
            if not safety_logic:
                agent.logger.warning("[CCA] No SafetyPlanner configured.")
                return

            safety_text = safety_logic.load_nl_safety_text()
            if not safety_text:
                agent.logger.warning("[CCA] No NL safety text.")
                return

            await safety_logic.build_safety_rules_and_logic(safety_text)
            safety_logic.save(agent.safety_logic_path)

            agent.safety_rules = safety_logic.rules
            dfa_map = safety_logic.build_dfas_per_rule()

            agent.safety_monitor = SafetyMonitor(dfa_map, agent.safety_rules)

            agent.logger.info(
                "[CCA] _InitCCA completed. %d rules, %d DFAs.",
                len(agent.safety_rules),
                len(dfa_map),
            )

            # Now that safety_monitor exists, start the monitor behaviour
            agent.add_behaviour(agent._Monitor())
            agent.logger.info("[CCA] _Monitor behaviour started.")

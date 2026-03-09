"""Central Controller Agent (CCA) orchestrating safety checks and replanning."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional, Iterable, Dict, List

from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message
from spade.template import Template

from agents.shared_information.llm_agent import LlmAgent
from agents.central_controller.safety_logic import SafetyLogic
# Import the updated monitor
from agents.central_controller.online_safety_monitor import OnlineSafetyMonitor
from agents.central_controller.online_fsa_monitor import OnlineFsaMonitor
from agents.central_controller.online_safety_supervisor import OnlineSafetySupervisor
from cais_spade_llm.agents.central_controller.plan_safety_validator import PlanSafetyValidator

class CentralControllerAgent(LlmAgent):
    """
    Central Controller Agent (CCA).
    Coordinates safety validation, online monitoring, and replanning signals.
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
        precomputed_bundle: Optional[dict[str, Any]] = None,
        **kw: Any,
    ) -> None:
        """Initialize controller state, safety logic, and monitoring scaffolding."""
        super().__init__(jid, password, name=name, agent_role="controller", **kw)

        self.agent_name = name
        self.safety_file = Path(safety_file) if safety_file else None
        self.resource_agents = list(resource_agents or [])
        self.precomputed_bundle: dict[str, Any] = dict(precomputed_bundle or {})
        
        base_safety_dir = Path("cais_spade_llm/safety")
        self.safety_logic_path = base_safety_dir / f"{name}_safety_logic.json"

        self.safety_logic: Optional[SafetyLogic] = None
        if self.safety_file:
            self.safety_logic = SafetyLogic(self, self.safety_file)

        self.safety_rules: list[dict[str, Any]] = []
        
        # We use the NEW OnlineSafetyMonitor
        self.safety_monitor: Optional[OnlineSafetyMonitor] = None

        # Runtime Plan FSA monitor
        self.plan_fsa_monitor: Optional[OnlineFsaMonitor] = None
        self.online_supervisor: Optional[OnlineSafetySupervisor] = None
        self.runtime_supervisor_mode: str = "preventive"
        
        # NOTE: self.running_aps is removed; the monitor tracks it now.
        
        self.blocked_tasks: dict[str, dict[str, Any]] = {}
        # Stores the most recent task failure event so plan_block replans can
        # include the root-cause failure context, not just the blocked task's event.
        self.last_failure_event: Optional[dict[str, Any]] = None

        self.logger.info(
            "CentralControllerAgent '%s' initialized. safety_file=%s",
            name, str(self.safety_file)
        )

    async def setup(self) -> None:
        """Attach startup, runtime monitor, and plan validation behaviours."""
        await super().setup()
        self.logger.info("[CCA] setup completed.")
        self.add_behaviour(self._InitCCA())
        t_resource = Template()
        t_resource.set_metadata("type", "resource_event")
        self.add_behaviour(self._Monitor(), t_resource)

        t_plan = Template()
        t_plan.set_metadata("type", "plan_safety_check")
        self.add_behaviour(self._PlanValidation(), t_plan)

    def _collect_system_coordination_state(self) -> dict[str, Any]:
        """
        Collect system-level coordination state for replanning context.

        This includes:
        - Resource states from ResourceAgents
        - Running tasks from safety monitor
        - Plan FSA state
        - Safety DFA states

        Returns:
            Dictionary containing system coordination state
        """
        coord_state: dict[str, Any] = {}

        # 1. Collect resource states from ResourceAgents.
        resource_states = {}
        for ra in self.resource_agents:
            if hasattr(ra, '_snapshot_state'):
                resource_states[str(ra.jid)] = ra._snapshot_state()
            else:
                # Fallback for agents without _snapshot_state
                resource_states[str(ra.jid)] = {
                    "current_state": "unknown",
                    "held_part": getattr(ra, '_held_part', None),
                }
        coord_state["resource_states"] = resource_states

        # 2. Collect running tasks from safety monitor
        if self.safety_monitor:
            # running_aps is a set of AP labels; keep it JSON-serializable and stable.
            coord_state["running_tasks"] = sorted(self.safety_monitor.running_aps)
        else:
            coord_state["running_tasks"] = []

        # 3. Collect plan FSA state
        if self.plan_fsa_monitor:
            coord_state["plan_fsa_state"] = self.plan_fsa_monitor.current_state
            coord_state["plan_fsa_completed_tasks"] = list(
                self.plan_fsa_monitor.completed_task_ids
            )
        else:
            coord_state["plan_fsa_state"] = None
            coord_state["plan_fsa_completed_tasks"] = []

        # 4. Collect safety DFA states (per-rule DFA states)
        safety_dfa_states = {}
        if self.safety_monitor:
            # OnlineSafetyMonitor stores current DFA pointers in current_states.
            if hasattr(self.safety_monitor, "current_states"):
                safety_dfa_states = dict(getattr(self.safety_monitor, "current_states", {}))
            # Fallback for alternate monitor implementations.
            elif hasattr(self.safety_monitor, "dfa_map"):
                for rule_id, dfa_obj in getattr(self.safety_monitor, "dfa_map", {}).items():
                    if hasattr(dfa_obj, "current_state"):
                        safety_dfa_states[rule_id] = dfa_obj.current_state
        coord_state["safety_dfa_states"] = safety_dfa_states

        if self.online_supervisor:
            try:
                coord_state["safety_supervisor"] = self.online_supervisor.classify()
            except Exception:
                self.logger.exception("[CCA] Failed to classify online supervisor state.")
                coord_state["safety_supervisor"] = {"status": "error"}
        else:
            coord_state["safety_supervisor"] = None

        return coord_state

    @staticmethod
    def _extract_resource_states(system_coordination_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if not isinstance(system_coordination_state, dict):
            return {}

        for key in ("resource_states", "resources", "robot_states", "robots"):
            payload = system_coordination_state.get(key)
            if isinstance(payload, dict):
                return payload
        return {}

    def _classify_replan_reason(
        self,
        *,
        current_state: str,
        marked_states: set,
        next_task_ids: list,
        any_running: bool,
    ) -> Optional[str]:
        """
        Classify why replanning is needed based on FSA state (completion path only).
        Task-failure replanning is handled separately via the safety monitor.

        Returns:
            None if no replanning needed, otherwise:
            - "plan_stuck_non_accepting": FSA has no forward transitions and is not
              in an accepting state — plan cannot progress after a completion.
        """
        if current_state in marked_states:
            return None

        if any_running:
            return None

        if next_task_ids:
            return None

        return "plan_stuck_non_accepting"

    def _build_replan_message(
        self,
        *,
        product_jid: str,
        reason: str,
        event: dict[str, Any],
        safety_info: Optional[dict[str, Any]],
    ) -> Message:
        """Build a structured replanning request message for a ProductAgent."""
        plan_ctx = {}
        if self.plan_fsa_monitor:
            plan_ctx = self.plan_fsa_monitor.build_replan_context(
                failure_event={
                    "failed_task_id": event.get("task_id"),
                    "task_id": event.get("task_id"),
                }
            )

        # Failure-focused safety context: keep only what helps replanning decisions.
        safety_ctx = {}
        if safety_info:
            # Preserve all provided fields so runtime context is not dropped.
            safety_ctx = dict(safety_info)

            # Add normalized aliases expected by replanning prompt builders.
            safety_ctx.setdefault("violated_rule_id", safety_info.get("violated_rule"))
            safety_ctx.setdefault("running_aps", safety_info.get("running_snapshot") or safety_info.get("running_tasks") or [])
            safety_ctx.setdefault("candidate_aps", safety_info.get("candidate_aps") or [])

        # Collect system coordination state (robot states, running tasks, FSA states)
        system_coordination_state = self._collect_system_coordination_state()
        if safety_ctx:
            safety_ctx["obligation_targets"] = self._build_obligation_targets(
                event=event,
                safety_info=safety_ctx,
                system_coordination_state=system_coordination_state,
            )

        # Debug log: capture the full context we are about to send for replanning.
        # Keep logs bounded to avoid flooding if the context grows large.
        self.logger.info(
            "[CCA] Replan request -> %s reason=%s task_id=%s plan_ctx=%s safety_ctx=%s coord_state=%s",
            product_jid,
            reason,
            event.get("task_id"),
            json.dumps(plan_ctx, ensure_ascii=False)[:1000],
            json.dumps(safety_ctx, ensure_ascii=False)[:1000],
            json.dumps(system_coordination_state, ensure_ascii=False)[:1000],
        )

        msg = Message(to=str(product_jid))
        msg.set_metadata("type", "replan_request")
        msg.body = json.dumps(
            {
                "reason": reason,
                "event": event,
                "plan_ctx": plan_ctx,
                "safety_ctx": safety_ctx,
                "system_coordination_state": system_coordination_state,
            }
        )
        return msg

    @staticmethod
    def _normalize_rule_ids(value: Any) -> list[str]:
        if isinstance(value, (list, tuple, set)):
            items = value
        elif value is None:
            items = []
        else:
            items = [value]
        return [str(item).strip() for item in items if str(item).strip()]

    def _resource_jid_by_token(self) -> dict[str, str]:
        if not self.safety_monitor:
            return {}
        mapping: dict[str, str] = {}
        for ra in self.resource_agents:
            jid_text = str(getattr(ra, "jid", "")).strip()
            if not jid_text:
                continue
            mapping[self.safety_monitor._resource_short_name(jid_text)] = jid_text
        return mapping

    def _rule_resource_tokens(
        self,
        rule: dict[str, Any],
        *,
        fallback_resource: str = "",
    ) -> list[str]:
        if not self.safety_monitor:
            return []
        concrete: list[str] = []
        for value in (rule.get("resources") or []):
            token = self.safety_monitor._resource_short_name(str(value or "").strip())
            if not token or token in {"any", "robot"}:
                continue
            if token not in concrete:
                concrete.append(token)
        if concrete:
            return concrete
        if fallback_resource:
            return [fallback_resource]
        return sorted(self._resource_jid_by_token().keys())

    def _tool_matches_ap_descriptor(
        self,
        row: dict[str, Any],
        descriptor: dict[str, str],
        *,
        resource_token: str,
    ) -> bool:
        if not self.safety_monitor:
            return False

        row_owner = self.safety_monitor._resource_short_name(
            str(row.get("function_owner_agent", "")).strip()
        )
        if resource_token and row_owner and row_owner != resource_token:
            return False

        desc_resource = self.safety_monitor._resource_short_name(
            str(descriptor.get("resource", "")).strip()
        )
        if desc_resource not in {"", "any", "robot"} and desc_resource != row_owner:
            return False

        desc_process = str(descriptor.get("process", "")).strip().lower()
        row_process = str(row.get("process", "")).strip().lower()
        if desc_process not in {"", "any"} and row_process and desc_process != row_process:
            return False

        prefix = str(descriptor.get("prefix", "")).strip().lower()
        symbol = str(descriptor.get("symbol", "")).strip()
        if prefix in {"ap", "ap_event"}:
            return symbol == str(row.get("function", "")).strip()
        if prefix in {"ap_state", "sp"}:
            out_state = str(row.get("out_state", "")).strip()
            return bool(out_state) and out_state.lower() != "any" and out_state == symbol
        return False

    def _candidate_tools_for_obligation(
        self,
        *,
        resource_token: str,
        required_event_aps: list[dict[str, Any]],
        required_state_aps: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.safety_monitor:
            return []

        candidates: list[dict[str, Any]] = []
        resource_jid = self._resource_jid_by_token().get(resource_token, resource_token)

        for row in getattr(self, "tools_catalog", []) or []:
            if not isinstance(row, dict):
                continue

            matched_event_aps = [
                ap
                for ap in required_event_aps
                if self._tool_matches_ap_descriptor(row, ap, resource_token=resource_token)
            ]
            matched_state_aps = [
                ap
                for ap in required_state_aps
                if self._tool_matches_ap_descriptor(row, ap, resource_token=resource_token)
            ]

            if len(matched_event_aps) != len(required_event_aps):
                continue
            if len(matched_state_aps) != len(required_state_aps):
                continue

            candidates.append(
                {
                    "function_name": str(row.get("function", "")).strip(),
                    "resource_jid": resource_jid,
                    "tool_signature": self.safety_monitor._tool_signature(row),
                    "in_state": str(row.get("in_state", "")).strip(),
                    "out_state": str(row.get("out_state", "")).strip(),
                    "description": str(row.get("description", "")).strip(),
                    "matched_event_aps": [
                        {"label": ap.get("label", ""), "full": ap.get("full", "")}
                        for ap in matched_event_aps
                    ],
                    "matched_state_aps": [
                        {"label": ap.get("label", ""), "full": ap.get("full", "")}
                        for ap in matched_state_aps
                    ],
                }
            )

        return candidates

    def _build_obligation_targets(
        self,
        *,
        event: dict[str, Any],
        safety_info: dict[str, Any],
        system_coordination_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not self.safety_monitor or not self.safety_rules:
            return []

        rule_ids = self._normalize_rule_ids(
            safety_info.get("rule_ids")
            or safety_info.get("violated_rule")
            or safety_info.get("violated_rule_id")
        )
        if not rule_ids:
            return []

        rule_lookup = {
            str(rule.get("id", "")).strip(): rule
            for rule in self.safety_rules
            if isinstance(rule, dict) and str(rule.get("id", "")).strip()
        }
        trigger_event = str((event or {}).get("function_name") or "").strip()
        trigger_resource = self.safety_monitor._resource_short_name(
            str((event or {}).get("resource_jid") or "").strip()
        )
        resource_states = self._extract_resource_states(system_coordination_state or {})
        resource_jid_by_token = self._resource_jid_by_token()
        targets: list[dict[str, Any]] = []

        for rule_id in rule_ids:
            rule = rule_lookup.get(rule_id)
            if not isinstance(rule, dict):
                continue

            rule_context = dict(rule.get("context") or {})
            trigger_symbol = str(rule_context.get("trigger_event") or "").strip()
            required_event_aps: list[dict[str, Any]] = []
            required_state_aps: list[dict[str, Any]] = []

            for ap in rule.get("aps") or []:
                if not isinstance(ap, dict):
                    continue
                full = str(ap.get("full", "")).strip()
                label = str(ap.get("label", "")).strip()
                descriptor = self.safety_monitor._parse_ap_descriptor(full)
                if not descriptor:
                    continue
                payload = {**descriptor, "label": label, "full": full}
                prefix = str(payload.get("prefix", "")).strip().lower()
                symbol = str(payload.get("symbol", "")).strip()
                if prefix in {"ap", "ap_event"}:
                    if trigger_symbol and symbol == trigger_symbol:
                        continue
                    required_event_aps.append(payload)
                elif prefix in {"ap_state", "sp"}:
                    required_state_aps.append(payload)

            if not required_event_aps and not required_state_aps:
                fallback_event = str(rule.get("event", "")).strip()
                if fallback_event:
                    for resource_token in self._rule_resource_tokens(
                        rule,
                        fallback_resource=trigger_resource,
                    ):
                        required_event_aps.append(
                            {
                                "prefix": "ap_event",
                                "process": str(rule.get("process", "")).strip().lower() or "any",
                                "product": "any",
                                "resource": resource_token,
                                "symbol": fallback_event,
                                "context": "",
                                "label": "",
                                "full": "",
                            }
                        )

            target_resource_tokens = sorted(
                {
                    self.safety_monitor._resource_short_name(ap.get("resource", ""))
                    for ap in required_event_aps + required_state_aps
                    if self.safety_monitor._resource_short_name(ap.get("resource", ""))
                    not in {"", "any", "robot"}
                }
            )
            if not target_resource_tokens:
                target_resource_tokens = self._rule_resource_tokens(
                    rule,
                    fallback_resource=trigger_resource,
                )

            for resource_token in target_resource_tokens:
                resource_jid = resource_jid_by_token.get(resource_token, resource_token)
                resource_event_aps = [
                    ap for ap in required_event_aps
                    if self.safety_monitor._resource_short_name(ap.get("resource", ""))
                    in {"", "any", "robot", resource_token}
                ]
                resource_state_aps = [
                    ap for ap in required_state_aps
                    if self.safety_monitor._resource_short_name(ap.get("resource", ""))
                    in {"", "any", "robot", resource_token}
                ]
                candidate_tools = self._candidate_tools_for_obligation(
                    resource_token=resource_token,
                    required_event_aps=resource_event_aps,
                    required_state_aps=resource_state_aps,
                )
                current_snapshot = dict(resource_states.get(resource_jid) or {})
                targets.append(
                    {
                        "rule_id": rule_id,
                        "resource_jid": resource_jid,
                        "required_event_aps": [
                            {"label": ap.get("label", ""), "full": ap.get("full", "")}
                            for ap in resource_event_aps
                        ],
                        "required_state_aps": [
                            {"label": ap.get("label", ""), "full": ap.get("full", "")}
                            for ap in resource_state_aps
                        ],
                        "trigger_event": trigger_symbol,
                        "generated_interpretation": str(
                            rule.get("generated_interpretation", "")
                        ).strip(),
                        "candidate_tools": candidate_tools,
                        "required_in_states": sorted(
                            {
                                str(tool.get("in_state", "")).strip()
                                for tool in candidate_tools
                                if str(tool.get("in_state", "")).strip()
                                and str(tool.get("in_state", "")).strip().lower() != "any"
                            }
                        ),
                        "required_out_states": sorted(
                            {
                                str(tool.get("out_state", "")).strip()
                                for tool in candidate_tools
                                if str(tool.get("out_state", "")).strip()
                                and str(tool.get("out_state", "")).strip().lower() != "any"
                            }
                        ),
                        "current_resource_state": str(
                            current_snapshot.get("current_state", "")
                        ).strip(),
                    }
                )

        return targets

    # ------------------------------------------------------------------ #
    # Behaviours
    # ------------------------------------------------------------------ #
    class _Monitor(CyclicBehaviour):
        """
        Listen for 'resource_event', delegate logic to OnlineSafetyMonitor,
        and send decisions back to resources.
        """

        async def run(self) -> None:
            agent: "CentralControllerAgent" = self.agent  # type: ignore

            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            # Safety guard: ensure monitor is loaded
            if not agent.safety_monitor:
                agent.logger.warning("[CCA] SafetyMonitor not loaded yet.")
                return

            # 1. DELEGATE PARSING to the Monitor
            #
            event = agent.safety_monitor.parse_resource_event(msg)
            if not event:
                return

            task_id = event["task_id"]
            status = event["status"]
            resource_jid = event["resource_jid"]
            function_name = event["function_name"]
            product_jid = (event.get("params") or {}).get("product_jid")

            # 2. HANDLE 'SAFETY_CHECK' (Start Event)
            if status == "safety_check":
                await self._handle_safety_check(
                    event=event,
                    task_id=task_id,
                    resource_jid=resource_jid,
                    product_jid=product_jid,
                )
                return

            await self._handle_runtime_event(
                event=event,
                task_id=task_id,
                status=status,
                resource_jid=resource_jid,
                function_name=function_name,
                product_jid=product_jid,
            )

        async def _handle_safety_check(
            self,
            *,
            event: dict[str, Any],
            task_id: str,
            resource_jid: str,
            product_jid: Optional[str],
        ) -> None:
            agent: "CentralControllerAgent" = self.agent  # type: ignore

            # --- FSA check: is this task enabled given the current plan state? ---
            # If a predecessor failed, the FSA is stuck and this task won't be enabled.
            if agent.plan_fsa_monitor:
                enabled = agent.plan_fsa_monitor._next_task_ids_from_state(
                    agent.plan_fsa_monitor.current_state
                )
                if task_id not in enabled:
                    agent.logger.warning(
                        "[CCA] PLAN BLOCK: task=%s not enabled in FSA (predecessor may have failed).",
                        task_id,
                    )
                    await self._send_decision(resource_jid, task_id, "block")
                    if product_jid:
                        # Use the root-cause failure event so the LLM gets the
                        # actual failure_context (mode, retryable, etc.), not
                        # the empty safety_check event of the blocked task.
                        failure_event = agent.last_failure_event or event
                        replan_msg = agent._build_replan_message(
                            product_jid=product_jid,
                            reason="plan_block",
                            event=failure_event,
                            safety_info={
                                "running_tasks": sorted(agent.safety_monitor.running_aps),
                                "blocked_task_id": task_id,
                            },
                        )
                        await self.send(replan_msg)
                    return

            candidate_aps = agent.safety_monitor._map_task_to_aps(
                event["resource_jid"],
                event["function_name"],
                event.get("params") or {},
            )
            predicted_state_aps = agent.safety_monitor._predict_state_aps(
                event["resource_jid"],
                event["function_name"],
                event.get("params") or {},
            )
            allowed, info = agent.safety_monitor.online_safety_validation(
                candidate_aps,
                predicted_state_aps=predicted_state_aps,
            )

            if not allowed:
                violated_rule = info.get("violated_rule")
                running_snapshot = info.get("running_snapshot", [])

                agent.logger.warning(
                    "[CCA] SAFETY VIOLATION: task=%s rule=%s (running=%s)",
                    task_id, violated_rule, running_snapshot
                )

                # Queue the task to retry later
                agent.blocked_tasks[task_id] = {
                    "event": event,
                    "violated_rule": violated_rule
                }

                agent.logger.info(
                    "[CCA] Queued task=%s as temporarily unsafe.", task_id
                )

                await self._send_decision(resource_jid, task_id, "block")

                if product_jid:
                    replan_msg = agent._build_replan_message(
                        product_jid=product_jid,
                        reason="safety_block",
                        event=event,
                        safety_info=info,
                    )
                    await self.send(replan_msg)
                return

            if agent.online_supervisor:
                allowed_by_supervisor, diagnosis = agent.online_supervisor.check_candidate(event)
                if not allowed_by_supervisor:
                    agent.logger.warning(
                        "[CCA] SUPERVISOR BLOCK: task=%s status=%s safe_next=%s reason=%s",
                        task_id,
                        diagnosis.get("status"),
                        diagnosis.get("safe_next_task_ids"),
                        diagnosis.get("reason"),
                    )

                    if diagnosis.get("status") == "blocked_candidate":
                        agent.blocked_tasks[task_id] = {
                            "event": event,
                            "violated_rule": diagnosis.get("rule_ids"),
                        }

                    await self._send_decision(resource_jid, task_id, "block")

                    if product_jid and diagnosis.get("status") in {"inevitable_violation", "violated"}:
                        replan_msg = agent._build_replan_message(
                            product_jid=product_jid,
                            reason=str(diagnosis.get("status")),
                            event=event,
                            safety_info=diagnosis,
                        )
                        await self.send(replan_msg)
                    return
                if diagnosis.get("status") == "deferred_monitoring":
                    agent.logger.info(
                        "[CCA] SUPERVISOR DEFERRED: task=%s mode=%s reason=%s",
                        task_id,
                        diagnosis.get("enforcement_mode"),
                        diagnosis.get("reason"),
                    )

            allowed, info = agent.safety_monitor.process_start_event(event)
            if not allowed:
                agent.logger.warning(
                    "[CCA] Safety state changed before task=%s could be committed; blocking start.",
                    task_id,
                )
                await self._send_decision(resource_jid, task_id, "block")
                return

            # If allowed
            agent.logger.debug("[CCA] Safety OK: task=%s allowed.", task_id)
            await self._send_decision(resource_jid, task_id, "allow")

        async def _handle_runtime_event(
            self,
            *,
            event: dict[str, Any],
            task_id: str,
            status: str,
            resource_jid: str,
            function_name: str,
            product_jid: Optional[str],
        ) -> None:
            agent: "CentralControllerAgent" = self.agent  # type: ignore

            # ----- PLAN FSA TRACE: START EVENT ----- #
            if status == "running" and agent.plan_fsa_monitor:
                agent.plan_fsa_monitor.process_event(
                    event_type="start",
                    task_id=task_id,
                    function_name=function_name,
                    resource_jid=resource_jid,
                    status=status,
                )

            # 3. HANDLE 'FINISHED' / 'FAILED' (End Event)
            # Treat any "failed:*" status as a failed end event.
            is_failed = (status == "failed") or (isinstance(status, str) and status.startswith("failed"))
            is_completed = status in ("completed", "finished")
            if is_completed or is_failed:
                # ----- PLAN FSA TRACE: END EVENT ----- #
                if agent.plan_fsa_monitor:
                    agent.plan_fsa_monitor.process_event(
                        event_type=("fail" if is_failed else "done"),
                        task_id=task_id,
                        function_name=function_name,
                        resource_jid=resource_jid,
                        status=status,
                    )
                if is_failed:
                    agent.safety_monitor.process_fail_event(event)
                    agent.last_failure_event = event
                    agent.logger.info("[CCA] Task %s failed. State updated.", task_id)
                else:
                    agent.safety_monitor.process_finish_event(event)
                    agent.logger.info("[CCA] Task %s finished. State updated.", task_id)

                # Retry any blocked tasks now that state has changed
                await self._retry_blocked_tasks()

                if agent.online_supervisor and product_jid:
                    diagnosis = agent.online_supervisor.classify(
                        event_kind=("fail" if is_failed else "done")
                    )
                    status_token = str(diagnosis.get("status") or "")
                    if status_token in {"inevitable_violation", "violated"}:
                        agent.logger.warning(
                            "[CCA] Supervisor detected %s after task=%s. Triggering replanning.",
                            status_token,
                            task_id,
                        )
                        replan_msg = agent._build_replan_message(
                            product_jid=product_jid,
                            reason=status_token,
                            event=event,
                            safety_info=diagnosis,
                        )
                        await self.send(replan_msg)
                    elif status_token == "pending_obligation":
                        agent.logger.info(
                            "[CCA] Supervisor pending obligation after task=%s: rules=%s safe_next=%s",
                            task_id,
                            diagnosis.get("rule_ids"),
                            diagnosis.get("safe_next_task_ids"),
                        )

                # Progress detection for completions: check if plan can continue
                if is_completed and agent.plan_fsa_monitor and product_jid:
                    pm = agent.plan_fsa_monitor
                    cur_state = pm.current_state
                    if cur_state:
                        marked = set((pm.fsa or {}).get("A", {}).get("Xm") or [])
                        plan_running_tasks = pm.running_task_ids_from_state(cur_state)
                        any_running = bool(plan_running_tasks)
                        next_task_ids = pm._next_task_ids_from_state(cur_state)

                        replan_reason = agent._classify_replan_reason(
                            current_state=cur_state,
                            marked_states=marked,
                            next_task_ids=next_task_ids,
                            any_running=any_running,
                        )

                        if replan_reason:
                            replan_msg = agent._build_replan_message(
                                product_jid=product_jid,
                                reason=replan_reason,
                                event=event,
                                safety_info={
                                    "current_state": cur_state,
                                    "marked_states": list(marked),
                                    "available_tasks": next_task_ids,
                                    "running_tasks": plan_running_tasks,
                                },
                            )
                            await self.send(replan_msg)

        async def _retry_blocked_tasks(self) -> None:
            """
            Re-evaluate queued blocked tasks and prune entries that are no longer
            blocked. We intentionally do NOT send deferred "allow" decisions:
            resources must always receive allow/block only as a response to a
            fresh safety_check request for that task attempt.
            """
            agent: "CentralControllerAgent" = self.agent # type: ignore
            
            if not agent.blocked_tasks or not agent.safety_monitor:
                return

            to_clear = []

            # Check all blocked tasks against the NEW state.
            for task_id, data in agent.blocked_tasks.items():
                event = data["event"]
                
                # Re-check safety status WITHOUT mutating running_aps/current DFA state.
                try:
                    candidate_aps = agent.safety_monitor._map_task_to_aps(
                        event["resource_jid"],
                        event["function_name"],
                        event.get("params") or {},
                    )
                    predicted_state_aps = agent.safety_monitor._predict_state_aps(
                        event["resource_jid"],
                        event["function_name"],
                        event.get("params") or {},
                    )
                    allowed, _ = agent.safety_monitor.online_safety_validation(
                        candidate_aps,
                        predicted_state_aps=predicted_state_aps,
                    )
                    if allowed and agent.online_supervisor:
                        allowed, _ = agent.online_supervisor.check_candidate(event)
                except Exception:
                    agent.logger.exception(
                        "[CCA] Failed to non-mutating re-check for blocked task=%s; keeping queued.",
                        task_id,
                    )
                    continue

                if allowed:
                    agent.logger.info(
                        "[CCA] Clearing blocked queue entry task=%s (rule %s no longer violated). "
                        "Waiting for fresh safety_check before allowing execution.",
                        task_id, data.get("violated_rule")
                    )
                    to_clear.append(task_id)
                else:
                    # Still blocked, keep queued.
                    pass

            # Cleanup
            for tid in to_clear:
                agent.blocked_tasks.pop(tid, None)

        async def _send_decision(self, to_jid: str, task_id: str, decision: str):
            msg = Message(to=to_jid)
            msg.set_metadata("type", "safety_decision")
            msg.body = json.dumps({"task_id": task_id, "decision": decision})
            await self.send(msg)


    class _InitCCA(OneShotBehaviour):
        async def run(self) -> None:
            agent: "CentralControllerAgent" = self.agent  # type: ignore

            safety_logic = agent.safety_logic
            if not safety_logic:
                agent.logger.warning("[CCA] No SafetyPlanner configured.")
                return

            # Bundle fast-path: load precomputed structured safety + DFA artifacts.
            bundle = dict(agent.precomputed_bundle or {})
            artifacts = bundle.get("artifacts", {}) if isinstance(bundle, dict) else {}
            precomputed_logic = artifacts.get("safety_logic_json") if isinstance(artifacts, dict) else None
            if precomputed_logic:
                try:
                    p_logic = Path(str(precomputed_logic))
                    if p_logic.exists():
                        await asyncio.to_thread(safety_logic.load, p_logic)
                        agent.safety_rules = safety_logic.rules or []

                        dfa_map: dict[str, str] = {}
                        expected_rule_ids = [str(r.get("id")) for r in agent.safety_rules if r.get("id")]
                        for rid in expected_rule_ids:
                            dot_path = p_logic.parent / f"{rid}_dfa.dot"
                            if dot_path.exists():
                                dfa_map[rid] = dot_path.read_text(encoding="utf-8")

                        if len(dfa_map) < len(expected_rule_ids):
                            dfa_map = await asyncio.to_thread(
                                safety_logic.build_dfas_per_rule,
                                p_logic.parent,
                            )

                        safety_logic.rule_dfas = dfa_map
                        agent.safety_monitor = OnlineSafetyMonitor(
                            dfa_map,
                            agent.safety_rules,
                            tools_catalog=getattr(agent, "tools_catalog", []),
                        )
                        agent.safety_monitor.seed_resource_states(
                            {
                                str(getattr(ra, "jid", "")): ra._snapshot_state()
                                for ra in agent.resource_agents
                                if hasattr(ra, "_snapshot_state")
                            }
                        )
                        agent.logger.info(
                            "[Bundle] Using precomputed safety bundle_id=%s path=%s rules=%d",
                            bundle.get("bundle_id", ""),
                            p_logic,
                            len(agent.safety_rules),
                        )
                        agent.logger.info(
                            "[CCA] _InitCCA completed. Monitor online with %d rules.",
                            len(agent.safety_rules),
                        )
                        return
                    else:
                        agent.logger.warning(
                            "[Bundle] Precomputed safety logic missing at %s; falling back to runtime generation.",
                            p_logic,
                        )
                except Exception:
                    agent.logger.exception(
                        "[Bundle] Failed loading precomputed safety. Falling back to runtime generation."
                    )

            safety_text = safety_logic.load_nl_safety_text()
            if not safety_text:
                agent.logger.warning("[CCA] No NL safety text.")
                return

            await safety_logic.build_safety_rules_and_logic(safety_text)
            await asyncio.to_thread(safety_logic.save, agent.safety_logic_path)

            agent.safety_rules = safety_logic.rules
            dfa_map = await asyncio.to_thread(safety_logic.build_dfas_per_rule)

            # Initialize the NEW OnlineSafetyMonitor
            #
            agent.safety_monitor = OnlineSafetyMonitor(
                dfa_map,
                agent.safety_rules,
                tools_catalog=getattr(agent, "tools_catalog", []),
            )
            agent.safety_monitor.seed_resource_states(
                {
                    str(getattr(ra, "jid", "")): ra._snapshot_state()
                    for ra in agent.resource_agents
                    if hasattr(ra, "_snapshot_state")
                }
            )

            agent.logger.info(
                "[CCA] _InitCCA completed. Monitor online with %d rules.",
                len(agent.safety_rules)
            )

    class _PlanValidation(CyclicBehaviour):
        """
        Behaviour that listens for 'plan_safety_check', validates a compiled
        plan FSA, and replies with the result.
        """

        async def run(self) -> None:
            agent: "CentralControllerAgent" = self.agent  # type: ignore

            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            try:
                data = json.loads(msg.body or "{}")
                fsa = data.get("fsa")            # REQUIRED
                plan = data.get("plan")          # OPTIONAL (semantic AP mapping)
                product_jid = data.get("product_jid")
                runtime_context = data.get("runtime_context") or {}
                skip_revalidation = bool(
                    data.get("skip_revalidation", data.get("skip_offline_validation", False))
                )
            except Exception:
                agent.logger.exception("[CCA] Malformed plan_safety_check.")
                return

            if not fsa:
                agent.logger.warning("[CCA] No FSA provided for plan validation.")
                return

            # Reuse the live runtime monitor when the validated FSA is unchanged.
            if agent.plan_fsa_monitor and agent.plan_fsa_monitor.matches_fsa(fsa):
                plan_fsa_monitor = agent.plan_fsa_monitor
            else:
                plan_fsa_monitor = OnlineFsaMonitor(fsa)
                if isinstance(runtime_context, dict):
                    plan_fsa_monitor.restore_runtime_progress(
                        completed_task_ids=runtime_context.get("completed_task_ids") or [],
                        running_task_ids=runtime_context.get("running_task_ids") or [],
                        failed_task_ids=runtime_context.get("failed_task_ids") or [],
                    )
            agent.plan_fsa_monitor = plan_fsa_monitor

            # Delegate to the plan validator. A verified zero-rule bundle is
            # still "ready" even though its DFA map is empty.
            if agent.safety_logic is not None and agent.safety_monitor is not None:
                validator = PlanSafetyValidator(
                    rules=agent.safety_rules,
                    dfa_map=dict(agent.safety_logic.rule_dfas),
                    tools_catalog=getattr(agent, "tools_catalog", []),
                )

                if skip_revalidation:
                    ok, violations = True, []
                else:
                    ok, violations = validator.validate_plan_fsa(
                        fsa=fsa,
                        plan=plan,
                        product_jid=product_jid
                    )
                try:
                    winning_set_data = validator.compute_winning_set(fsa=fsa, plan=plan)
                    policy = (
                        dict(agent.precomputed_bundle.get("replan_policy", {}))
                        if isinstance(agent.precomputed_bundle.get("replan_policy"), dict)
                        else {}
                    )
                    verified_bundle_runtime = (
                        str(agent.precomputed_bundle.get("status", "")).strip().lower() == "verified"
                    )
                    runtime_validation = (
                        isinstance(runtime_context, dict)
                        and any(
                            runtime_context.get(key)
                            for key in ("completed_task_ids", "running_task_ids", "failed_task_ids")
                        )
                    )
                    prior_mode = str(getattr(agent, "runtime_supervisor_mode", "") or "").strip().lower()
                    if verified_bundle_runtime:
                        supervisor_mode = "reactive"
                    else:
                        supervisor_mode = str(
                            policy.get("runtime_supervisor_mode")
                            or (
                                prior_mode
                                if runtime_validation and prior_mode in {"preventive", "reactive"}
                                else ("reactive" if skip_revalidation else "preventive")
                            )
                        ).strip().lower()
                    if supervisor_mode not in {"preventive", "reactive"}:
                        supervisor_mode = (
                            prior_mode
                            if runtime_validation and prior_mode in {"preventive", "reactive"}
                            else ("reactive" if skip_revalidation else "preventive")
                        )
                    agent.runtime_supervisor_mode = supervisor_mode
                    agent.online_supervisor = OnlineSafetySupervisor(
                        winning_set_data=winning_set_data,
                        fsa_monitor=plan_fsa_monitor,
                        safety_monitor=agent.safety_monitor,
                        enforcement_mode=supervisor_mode,
                    ) if agent.safety_monitor else None
                except Exception:
                    agent.online_supervisor = None
                    agent.logger.exception(
                        "[CCA] Failed to initialize online safety supervisor."
                    )
            else:
                agent.logger.warning("[CCA] Safety logic not ready; skipping validation.")
                ok, violations = True, []
                agent.online_supervisor = None

            # ---- NEW: log summary + details ----
            violated_rules = sorted({v.get("violated_rule_id") for v in violations if v.get("violated_rule_id")})
            if skip_revalidation:
                agent.logger.info(
                    "[CCA] Verified bundle startup: skipped plan revalidation and initialized runtime monitors product=%s supervisor_mode=%s",
                    product_jid,
                    agent.runtime_supervisor_mode,
                )
            else:
                agent.logger.info(
                    "[CCA] Plan FSA Validation: %s (Violated rules: %d, Witnesses: %d) product=%s supervisor_mode=%s",
                    "OK" if ok else "FAIL",
                    len(violated_rules),
                    len(violations),
                    product_jid,
                    agent.runtime_supervisor_mode,
                )

            if not ok and violations:
                # Cap to avoid log spam
                max_witnesses = 3
                for i, v in enumerate(violations[:max_witnesses], start=1):
                    agent.logger.error(
                        "[CCA] VIOLATION #%d | rule=%s | %s | ltlf=%s",
                        i,
                        v.get("violated_rule_id"),
                        v.get("violation_text"),
                        v.get("violation_logic"),
                    )
                    agent.logger.error(
                        "[CCA]   witness_task_ids=%s witness_events=%s",
                        v.get("witness_task_ids"),
                        v.get("witness_events"),
                    )

                    # Per-transition debug (only present if you added _sigma/_q_from/_q_to in validator)
                    for t in v.get("witness_transitions", [])[:50]:
                        agent.logger.error(
                            "[CCA]     ↳ from=%s --%s--> %s | task=%s | sigma=%s | DFA:%s→%s",
                            t.get("from"),
                            t.get("event"),
                            t.get("to"),
                            t.get("task_id"),
                            t.get("_sigma"),   # may be None if you didn't patch validator
                            t.get("_q_from"),  # may be None if you didn't patch validator
                            t.get("_q_to"),    # may be None if you didn't patch validator
                        )

                if len(violations) > max_witnesses:
                    agent.logger.error(
                        "[CCA] ... %d more witness(es) suppressed",
                        len(violations) - max_witnesses
                    )
            # ---- END NEW ----

            # Reply
            try:
                reply = msg.make_reply()
                reply.set_metadata("type", "plan_safety_result")
                reply.body = json.dumps({
                    "ok": ok,
                    "violations": violations
                })
                await self.send(reply)
            except Exception:
                agent.logger.exception("Failed to send reply.")

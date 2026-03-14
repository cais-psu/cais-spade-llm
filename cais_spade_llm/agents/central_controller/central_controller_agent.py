"""Central Controller Agent (CCA) orchestrating safety checks and replanning."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path
from typing import Any, Optional, Iterable, Dict, List, Set, Tuple

from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message
from spade.template import Template

from cais_spade_llm.agents.shared_information.llm_agent import LlmAgent
from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic
# Import the updated monitor
from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor
from cais_spade_llm.agents.central_controller.online_fsa_monitor import OnlineFsaMonitor
from cais_spade_llm.agents.central_controller.online_safety_supervisor import OnlineSafetySupervisor
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

    # ------------------------------------------------------------------ #
    # Forward simulation: decide wait vs. replan
    # ------------------------------------------------------------------ #
    def _will_violation_resolve(
        self,
        violated_rule: str,
        blocked_task_event: dict[str, Any],
    ) -> bool:
        """Check if the safety violation will naturally resolve by simulating
        the composite plan FSA forward through all remaining task interleavings
        and checking the safety DFA at each reached state.

        Returns True if *any* reachable interleaving clears the violation so
        that the blocked task would be allowed.
        """
        if not self.plan_fsa_monitor or not self.safety_monitor:
            return False

        blocked_task_id = str(blocked_task_event.get("task_id", "")).strip()
        if not blocked_task_id:
            return False

        # APs the blocked task would introduce (needed to test "would it be allowed?")
        blocked_candidate_aps = frozenset(
            self.safety_monitor._map_task_to_aps(
                blocked_task_event["resource_jid"],
                blocked_task_event["function_name"],
                blocked_task_event.get("params") or {},
            )
        )
        blocked_predicted_state_aps = frozenset(
            self.safety_monitor._predict_state_aps(
                blocked_task_event["resource_jid"],
                blocked_task_event["function_name"],
                blocked_task_event.get("params") or {},
            )
        )

        # Initial simulation state
        fsa_state = self.plan_fsa_monitor.current_state
        init_running_aps = frozenset(self.safety_monitor.running_aps)
        init_state_aps: dict[str, frozenset[str]] = {
            rjid: frozenset(aps)
            for rjid, aps in self.safety_monitor.resource_state_aps.items()
        }
        init_dfa_states = {
            rid: self.safety_monitor.current_states.get(rid, "1")
            for rid in self.safety_monitor.dfas
        }

        # BFS state: (fsa_state, running_aps, resource_state_aps_key, dfa_states_key)
        # resource_state_aps is tracked as a tuple of (resource_jid, frozenset) pairs
        def _state_aps_key(rsa: dict[str, frozenset[str]]) -> tuple:
            return tuple(sorted((k, v) for k, v in rsa.items()))

        def _dfa_key(ds: dict[str, str]) -> tuple:
            return tuple(sorted(ds.items()))

        def _all_state_aps(rsa: dict[str, frozenset[str]]) -> frozenset[str]:
            result: Set[str] = set()
            for labels in rsa.values():
                result |= labels
            return frozenset(result)

        # Test if blocked task would be allowed given simulated state
        def _blocked_task_allowed(
            running: frozenset[str],
            rsa: dict[str, frozenset[str]],
            dfa_states: dict[str, str],
        ) -> bool:
            sigma = frozenset(
                set(running)
                | set(_all_state_aps(rsa))
                | set(blocked_candidate_aps)
                | set(blocked_predicted_state_aps)
            )
            for rid in self.safety_monitor.dfas:
                curr = dfa_states.get(rid, "1")
                nxt = self.safety_monitor._delta(rid, curr, sigma)
                vio = self.safety_monitor.dfas[rid].get("violation_state")
                if vio and nxt == vio:
                    return False
            return True

        start = (
            fsa_state,
            init_running_aps,
            init_state_aps,
            init_dfa_states,
        )

        queue: deque[tuple] = deque([start])
        visited: set[tuple] = set()
        max_states = 2000  # bound to prevent explosion on large plans

        while queue and len(visited) < max_states:
            cur_fsa, cur_running, cur_rsa, cur_dfa = queue.popleft()

            visit_key = (cur_fsa, cur_running, _state_aps_key(cur_rsa), _dfa_key(cur_dfa))
            if visit_key in visited:
                continue
            visited.add(visit_key)

            # Get transitions from the current FSA state
            for tr in self.plan_fsa_monitor._from_map.get(cur_fsa or "", []):
                tr_task_id = str(tr.get("task_id", "")).strip()
                event_label = str(tr.get("event", "")).strip()

                # Skip the blocked task's own transitions
                if tr_task_id == blocked_task_id:
                    continue

                next_fsa = tr.get("to") or cur_fsa
                tr_resource_jid = str(tr.get("resource_jid", "")).strip()
                tr_function_name = str(tr.get("function_name", "")).strip()
                tr_params = tr.get("params") or {}

                # Simulate AP changes based on .start vs .done events
                next_running = set(cur_running)
                next_rsa = dict(cur_rsa)
                next_dfa = dict(cur_dfa)

                if event_label.endswith(".start"):
                    # .start adds event APs to running set
                    event_aps = self.safety_monitor._map_task_to_aps(
                        tr_resource_jid, tr_function_name, tr_params,
                    )
                    next_running |= set(event_aps)

                elif event_label.endswith(".done"):
                    # .done removes event APs and updates state APs
                    event_aps = self.safety_monitor._map_task_to_aps(
                        tr_resource_jid, tr_function_name, tr_params,
                    )
                    next_running -= set(event_aps)

                    # Update resource state APs based on out_state
                    out_state = str(tr.get("out_state", "")).strip()
                    if out_state:
                        new_state_aps = self.safety_monitor._map_state_to_aps(
                            tr_resource_jid, out_state, tr_params,
                        )
                        next_rsa[tr_resource_jid] = frozenset(new_state_aps)

                    # Advance safety DFA on .done
                    sigma = frozenset(
                        next_running
                        | set(_all_state_aps(next_rsa))
                        | set(event_aps)
                    )
                    for rid in self.safety_monitor.dfas:
                        prev = next_dfa.get(rid, "1")
                        next_dfa[rid] = self.safety_monitor._delta(rid, prev, sigma)

                next_running_fs = frozenset(next_running)

                # Check if blocked task would now be allowed
                if _blocked_task_allowed(next_running_fs, next_rsa, next_dfa):
                    return True

                queue.append((next_fsa, next_running_fs, next_rsa, next_dfa))

        return False

    # ------------------------------------------------------------------ #
    # DFA-guided recovery: find tools that exit violating states
    # ------------------------------------------------------------------ #
    def _find_dfa_recovery_tools(
        self,
        rule_id: str,
        resource_token: str,
    ) -> list[dict[str, Any]]:
        """Find tools that transition a resource OUT of states that contribute
        to the safety DFA violation.

        Instead of checking currently active APs (which may not yet reflect
        the future violating state), this method extracts the set of
        *violating state symbols* directly from the rule's AP definitions
        for the target resource.  It then searches the tools catalog for
        tools whose ``in_state`` is one of those violating states and whose
        ``out_state`` is NOT, meaning execution of that tool would clear the
        resource's contribution to the violation.

        Returns candidate tool dicts in the same format as
        ``_candidate_tools_for_obligation``.
        """
        if not self.safety_monitor:
            return []

        # 1. Find the rule and extract state AP symbols for the target resource.
        violating_states: set[str] = set()
        matched_state_ap_info: list[dict[str, str]] = []

        for rule in self.safety_rules:
            if str(rule.get("id", "")).strip() != rule_id:
                continue
            for ap in rule.get("aps") or []:
                full = str(ap.get("full", "")).strip()
                label = str(ap.get("label", "")).strip()
                if not full or not label:
                    continue
                descriptor = self.safety_monitor._parse_ap_descriptor(full)
                if not descriptor:
                    continue
                prefix = str(descriptor.get("prefix", "")).strip().lower()
                ap_resource = self.safety_monitor._resource_short_name(
                    str(descriptor.get("resource", "")).strip()
                )
                # Only consider state APs belonging to the target resource.
                if prefix not in {"ap_state", "sp"}:
                    continue
                if ap_resource not in {"", "any", "robot", resource_token}:
                    continue
                symbol = str(descriptor.get("symbol", "")).strip()
                if symbol:
                    violating_states.add(symbol)
                    matched_state_ap_info.append({"label": label, "full": full})
            break  # only process the matching rule

        if not violating_states:
            return []

        # 2. Find tools for this resource whose in_state is a violating
        #    state and whose out_state is NOT a violating state.
        resource_jid = self._resource_jid_by_token().get(resource_token, resource_token)
        candidates: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()

        for row in getattr(self, "tools_catalog", []) or []:
            if not isinstance(row, dict):
                continue
            row_owner = self.safety_monitor._resource_short_name(
                str(row.get("function_owner_agent", "")).strip()
            )
            if row_owner != resource_token:
                continue

            out_state = str(row.get("out_state", "")).strip()

            # Only require that the tool's out_state exits the violating
            # state set.  We deliberately skip in_state filtering — DES
            # bidding will verify reachability through its own BFS.
            if not out_state:
                continue
            if out_state in violating_states:
                continue

            sig = self.safety_monitor._tool_signature(row)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            candidates.append(
                {
                    "function_name": str(row.get("function", "")).strip(),
                    "resource_jid": resource_jid,
                    "tool_signature": sig,
                    "in_state": str(row.get("in_state", "")).strip(),
                    "out_state": out_state,
                    "description": str(row.get("description", "")).strip(),
                    "matched_event_aps": [],
                    "matched_state_aps": list(matched_state_ap_info),
                }
            )

        return candidates

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
                # Fallback: if direct AP matching found nothing, use
                # DFA-guided recovery to discover tools that transition
                # the resource out of the violating state set.
                if not candidate_tools:
                    candidate_tools = self._find_dfa_recovery_tools(
                        rule_id=rule_id,
                        resource_token=resource_token,
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
                    # Forward-simulate the composite FSA to check if the
                    # violation will naturally resolve without replanning.
                    will_resolve = agent._will_violation_resolve(
                        violated_rule=str(violated_rule or ""),
                        blocked_task_event=event,
                    )
                    if will_resolve:
                        agent.logger.info(
                            "[CCA] Safety block on task=%s (rule=%s) will resolve "
                            "naturally; waiting for remaining tasks to complete.",
                            task_id, violated_rule,
                        )
                    else:
                        agent.logger.info(
                            "[CCA] Forward simulation: violation persists after "
                            "all remaining tasks; requesting replan for task=%s.",
                            task_id,
                        )
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
            retry_ready_by_product: dict[str, list[str]] = {}

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
                    product_jid = str((event.get("params") or {}).get("product_jid", "")).strip()
                    if product_jid:
                        retry_ready_by_product.setdefault(product_jid, []).append(str(task_id))
                else:
                    # Still blocked, keep queued.
                    pass

            # Cleanup
            for tid in to_clear:
                agent.blocked_tasks.pop(tid, None)

            for product_jid, task_ids in retry_ready_by_product.items():
                if not task_ids:
                    continue
                retry_msg = Message(to=product_jid)
                retry_msg.set_metadata("type", "task_retry_ready")
                retry_msg.body = json.dumps({
                    "task_ids": task_ids,
                    "reason": "safety_unblocked",
                })
                await self.send(retry_msg)
                agent.logger.info(
                    "[CCA] Notified product=%s to requeue %d task(s) after transient safety block cleared: %s",
                    product_jid,
                    len(task_ids),
                    ", ".join(task_ids),
                )

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

            prior_plan_fsa_monitor = agent.plan_fsa_monitor

            # Reuse the live runtime monitor when the validated FSA is unchanged.
            if prior_plan_fsa_monitor and prior_plan_fsa_monitor.matches_fsa(fsa):
                plan_fsa_monitor = prior_plan_fsa_monitor
            else:
                plan_fsa_monitor = OnlineFsaMonitor(fsa)
                restored_from_live_state = plan_fsa_monitor.restore_from_prior_monitor(
                    prior_plan_fsa_monitor
                )
                if restored_from_live_state:
                    prior_progress = plan_fsa_monitor.runtime_progress_snapshot()
                    agent.logger.info(
                        "[CCA] Restored repaired FSA from live monitor state "
                        "state=%s completed=%d running=%d failed=%d",
                        plan_fsa_monitor.current_state,
                        len(prior_progress.get("completed_task_ids") or []),
                        len(prior_progress.get("running_task_ids") or []),
                        len(prior_progress.get("failed_task_ids") or []),
                    )
                elif isinstance(runtime_context, dict):
                    completed_task_ids = [
                        str(task_id).strip()
                        for task_id in (runtime_context.get("completed_task_ids") or [])
                        if str(task_id).strip()
                    ]
                    seen_completed = set(completed_task_ids)
                    running_task_ids = [
                        str(task_id).strip()
                        for task_id in (runtime_context.get("running_task_ids") or [])
                        if str(task_id).strip()
                    ]
                    failed_task_ids = [
                        str(task_id).strip()
                        for task_id in (runtime_context.get("failed_task_ids") or [])
                        if str(task_id).strip()
                    ]
                    if prior_plan_fsa_monitor:
                        prior_progress = prior_plan_fsa_monitor.runtime_progress_snapshot()
                        for task_id in prior_progress.get("completed_task_ids") or []:
                            task_id = str(task_id).strip()
                            if task_id and task_id not in seen_completed:
                                completed_task_ids.append(task_id)
                                seen_completed.add(task_id)
                        running_task_ids = list(
                            dict.fromkeys(
                                running_task_ids + [
                                    str(task_id).strip()
                                    for task_id in (prior_progress.get("running_task_ids") or [])
                                    if str(task_id).strip()
                                ]
                            )
                        )
                        failed_task_ids = list(
                            dict.fromkeys(
                                failed_task_ids + [
                                    str(task_id).strip()
                                    for task_id in (prior_progress.get("failed_task_ids") or [])
                                    if str(task_id).strip()
                                ]
                            )
                        )
                    plan_fsa_monitor.restore_runtime_progress(
                        completed_task_ids=completed_task_ids,
                        running_task_ids=running_task_ids,
                        failed_task_ids=failed_task_ids,
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
                    winning_set_data = {}
                else:
                    ok, violations = validator.validate_plan_fsa(
                        fsa=fsa,
                        plan=plan,
                        product_jid=product_jid
                    )
                try:
                    if not skip_revalidation:
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

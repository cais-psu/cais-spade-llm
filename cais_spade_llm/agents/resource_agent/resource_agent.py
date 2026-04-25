"""Resource agent that receives work orders, selects a tool via LLM, and executes it."""

# agents/resource_agent/resource_agent.py
from __future__ import annotations

import asyncio
import inspect
import json
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Tuple

from spade.behaviour import CyclicBehaviour  # Behaviour base used for our inbox loop.
from spade.message import Message  # SPADE message objects (XMPP stanzas under the hood).
from spade.template import Template  # Filters incoming messages by metadata.

from cais_spade_llm.agents.shared_information.llm_agent import LlmAgent
from cais_spade_llm.agents.intelligent_product.replanner.failure_context import (
    build_failure_event,
)


class ResourceAgent(LlmAgent):
    """
    SPADE ResourceAgent.

    - Receives tasks from ProductAgents (metadata[type] == "task").
    - Calls the LLM with tools enabled to select a function.
    - Dispatches to a registered executable (async function).
    - Sends ACK messages back to the ProductAgent with status.
    """

    agent_role = "resource"  # Used by LlmAgent to pick prompts and instructions for this class.

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str,
        function_names: Optional[Iterable[str]] = None,
        static_capabilities: Optional[Dict[str, Any]] = None,
        allowed_senders: Optional[Iterable[str]] = None,
        llm_timeout_s: int = 30,
        tool_timeout_s: int = 300,
        cca_jid: Optional[str] = None,   # <-- NEW
        **kw: Any,
    ) -> None:
        """
        :param function_names: Names of methods on this class to expose as tools.
        :param static_capabilities: Optional metadata for discovery/logging.
        :param allowed_senders: Optional allowlist of JIDs allowed to send tasks.
        :param llm_timeout_s: Timeout for LLM tool selection.
        :param tool_timeout_s: Timeout for tool execution.
        """
        super().__init__(
            jid,
            password,
            name=name,
            agent_role="resource",
            function_names=list(function_names or []),
            **kw,
        )

        self.cca_jid = cca_jid

        # Optional metadata (payload limits, tool list, etc.) exposed to other agents or dashboards.
        self.static_capabilities: Dict[str, Any] = static_capabilities or {}
        # Optional sender allow-list: if populated, only those JIDs can submit work.
        self.allowed_senders = set(allowed_senders or [])
        # Separate timeouts keep LLM latency (planning) independent from tool runtime (execution).
        self.llm_timeout_s = int(llm_timeout_s)
        self.tool_timeout_s = int(tool_timeout_s)

        self._safety_decisions: dict[str, str] = {}
        self._bridge_execution_primitive_catalog_cache: list[dict[str, Any]] | None = None
        self._bridge_synthesis_primitive_catalog_cache: list[dict[str, Any]] | None = None

        # Register bridge recovery executor for all resource types.
        # RobotAgent overrides the method but no longer needs to re-register.
        self.executables["execute_recovery_macro"] = self.execute_recovery_macro

    # ------------------------------------------------------------------ #
    # SPADE lifecycle
    # ------------------------------------------------------------------ #
    async def setup(self) -> None:
        await super().setup()

        # Tasks from ProductAgent
        t_task = Template()
        t_task.set_metadata("type", "task")
        self.add_behaviour(self._TaskInbox(), t_task)

        # Safety decisions from CCA
        t_safety = Template()
        t_safety.set_metadata("type", "safety_decision")
        self.add_behaviour(self._SafetyDecisionInbox(), t_safety)

    def _snapshot_state(self) -> Dict[str, Any]:
        """
        Best-effort snapshot of resource state for failure context.
        Subclasses can override to provide richer state.
        """
        return {}

    def bridge_resource_type(self) -> str:
        snapshot = self._snapshot_state() if hasattr(self, "_snapshot_state") else {}
        if isinstance(snapshot, dict):
            token = str(snapshot.get("resource_type", "") or "").strip().lower()
            if token:
                return token
        token = str(self.static_capabilities.get("resource_type", "") or "").strip().lower()
        if token:
            return token
        return "resource"

    def get_bridge_snapshot(self) -> Dict[str, Any]:
        """Return the current descriptor-driven bridge snapshot for this resource."""
        from cais_spade_llm.resources.resource_primitives import (
            get_resource_bridge_snapshot,
        )

        return get_resource_bridge_snapshot(self)

    def bridge_execution_primitive_catalog(self) -> list[dict[str, Any]]:
        """Return the resource-owned execution primitive catalog."""
        if self._bridge_execution_primitive_catalog_cache is None:
            from cais_spade_llm.resources.resource_primitives import (
                build_execution_primitive_catalog,
            )

            self._bridge_execution_primitive_catalog_cache = (
                build_execution_primitive_catalog(self)
            )
        return deepcopy(self._bridge_execution_primitive_catalog_cache)

    def bridge_synthesis_primitive_catalog(self) -> list[dict[str, Any]]:
        """Return the resource-owned LLM-facing primitive catalog."""
        if self._bridge_synthesis_primitive_catalog_cache is None:
            from cais_spade_llm.resources.resource_primitives import (
                build_synthesis_primitive_catalog,
            )

            self._bridge_synthesis_primitive_catalog_cache = (
                build_synthesis_primitive_catalog(
                    primitive_catalog=self.bridge_execution_primitive_catalog()
                )
            )
        return deepcopy(self._bridge_synthesis_primitive_catalog_cache)

    def invalidate_bridge_primitive_catalog(self) -> None:
        """Clear cached primitive catalogs after resource primitive changes."""
        self._bridge_execution_primitive_catalog_cache = None
        self._bridge_synthesis_primitive_catalog_cache = None

    def bridge_feasibility_oracle(
        self,
        *,
        event_instance: Any | None = None,
        schema: Any | None = None,
        projection: Any | None = None,
        part_context: Dict[str, Any] | None = None,
        bridge_snapshot: Dict[str, Any] | None = None,
        **_compat_kwargs: Any,
    ) -> Dict[str, Any]:
        """Default permissive bridge feasibility oracle.

        Subclasses (RobotAgent, PrintingAgent) can override with
        resource-specific checks.
        """
        del event_instance, schema, projection, part_context, bridge_snapshot
        return {"allowed": True, "reason": "default permissive oracle"}

    async def generate_bridge_primitives_batch(
        self,
        *,
        bridge_session_id: str = "",
        resource_jid: str = "",
        assigned_outline_events: list[dict[str, Any]] | None = None,
        prepared_bridge_request: Dict[str, Any] | None = None,
        carried_session_state: Dict[str, Any] | None = None,
        max_turns: int = 24,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn_primitive_generation import (
            generate_primitive_batch_with_llm_agent,
        )

        normalized_resource_jid = str(resource_jid or getattr(self, "jid", "") or "").strip()
        if normalized_resource_jid and normalized_resource_jid != str(getattr(self, "jid", "") or "").strip():
            raise ValueError(
                f"resource-owned primitive batch was assigned to {normalized_resource_jid!r} "
                f"but invoked on {str(getattr(self, 'jid', '') or '').strip()!r}"
            )
        return await generate_primitive_batch_with_llm_agent(
            llm_agent=self,
            prepared_bridge_request=dict(prepared_bridge_request or {}),
            assigned_outline_events=[
                dict(row)
                for row in (assigned_outline_events or [])
                if isinstance(row, dict)
            ],
            bridge_session_id=str(bridge_session_id or "").strip(),
            carried_session_state=dict(carried_session_state or {}),
            max_turns=max_turns,
        )

    async def execute_recovery_macro(
        self,
        *,
        macro_name: str = "",
        primitive_steps: list | None = None,
        expected_start_state: str = "",
        expected_snapshot: Dict[str, Any] | None = None,
        product_jid: str | None = None,
        task_id: str | None = None,
        in_state: str | None = None,
        out_state: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generic bridge recovery macro executor.

        Validates the starting snapshot, semantically validates the primitive
        sequence, executes each primitive on the resolved owner, and applies
        projected bridge state back onto the resource agent.
        """
        from cais_spade_llm.resources.resource_profile import (
            get_resource_profile_for_agent,
            resource_snapshot_set_field,
        )
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_primitives import (
            apply_effects_to_snapshot,
            event_fact_key_for_primitive,
            expand_composite_steps,
            extract_step_output,
            resolve_param_refs,
            snapshot_matches_expected,
            validate_and_project_steps,
        )
        from cais_spade_llm.resources.resource_primitives import (
            get_resource_bridge_snapshot,
            sync_agent_from_bridge_snapshot,
        )

        steps = list(primitive_steps or [])
        runtime_snapshot = get_resource_bridge_snapshot(self)
        actual_state = str(runtime_snapshot.get("current_state") or getattr(self, "_current_state", "") or "").strip()

        if expected_start_state and actual_state != expected_start_state:
            msg = (
                f"Recovery macro '{macro_name}' expected start state "
                f"'{expected_start_state}' but resource is in '{actual_state}'"
            )
            return {
                "status": "failed",
                "content": msg,
                "observations": {
                    "macro_name": macro_name,
                    "expected_start_state": expected_start_state,
                    "actual_state": actual_state,
                    "step_index": -1,
                },
            }

        if expected_snapshot:
            matches, mismatch_message = snapshot_matches_expected(runtime_snapshot, expected_snapshot)
            if not matches:
                msg = (
                    f"Recovery macro '{macro_name}' expected snapshot mismatch: "
                    f"{mismatch_message}"
                )
                return {
                    "status": "failed",
                    "content": msg,
                    "observations": {
                        "macro_name": macro_name,
                        "expected_snapshot": expected_snapshot,
                        "actual_snapshot": runtime_snapshot,
                        "step_index": -1,
                    },
                }

        if not steps:
            return {
                "status": "failed",
                "content": f"Recovery macro '{macro_name}' has no primitive steps",
            }

        primitive_catalog = self.bridge_execution_primitive_catalog()
        try:
            steps = expand_composite_steps(steps, primitive_catalog)
        except Exception as exc:
            return {
                "status": "failed",
                "content": (
                    f"Recovery macro '{macro_name}' could not expand composite primitives: {exc}"
                ),
                "observations": {
                    "macro_name": macro_name,
                    "step_index": -1,
                },
            }
        primitive_meta_by_name = {
            str(entry.get("name", "")).strip(): entry
            for entry in primitive_catalog
            if isinstance(entry, dict) and str(entry.get("name", "")).strip()
        }
        grounding_context = deepcopy(dict(kwargs or {}))
        semantic_ok, _projected_runtime_snapshot, semantic_error = validate_and_project_steps(
            steps,
            primitive_catalog,
            runtime_snapshot,
            grounding_context=grounding_context,
        )
        if not semantic_ok:
            msg = (
                f"Recovery macro '{macro_name}' failed runtime semantic validation: "
                f"{semantic_error}"
            )
            return {
                "status": "failed",
                "content": msg,
                "observations": {
                    "macro_name": macro_name,
                    "expected_snapshot": expected_snapshot,
                    "actual_snapshot": runtime_snapshot,
                    "semantic_error": semantic_error,
                    "step_index": -1,
                },
            }

        profile = get_resource_profile_for_agent(self)
        owner = self
        if profile.primitive_owner_resolver is not None:
            owner = profile.primitive_owner_resolver(self) or self

        results: list[Dict[str, Any]] = []
        event_facts: dict[str, Any] = {}
        resource_type = str(
            dict(runtime_snapshot.get("resource_core") or {}).get("resource_type")
            or runtime_snapshot.get("resource_type")
            or "resource"
        ).strip().lower() or "resource"
        for step_idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            primitive = str(step.get("primitive", "")).strip()
            raw_params = dict(step.get("params") or {})

            fn = getattr(owner, primitive, None) or getattr(self, primitive, None)
            if not callable(fn):
                msg = (
                    f"Unknown primitive '{primitive}' at step {step_idx} "
                    f"in macro '{macro_name}'"
                )
                return {
                    "status": "failed",
                    "content": msg,
                    "observations": {
                        "macro_name": macro_name,
                        "step_index": step_idx,
                        "primitive": primitive,
                        "completed_steps": len(results),
                    },
                }

            try:
                params = resolve_param_refs(
                    raw_params,
                    grounding_context,
                    event_facts=event_facts,
                )
            except Exception as exc:
                msg = (
                    f"Macro '{macro_name}' could not resolve params at step {step_idx} "
                    f"({primitive}): {exc}"
                )
                return {
                    "status": "failed",
                    "content": msg,
                    "observations": {
                        "macro_name": macro_name,
                        "step_index": step_idx,
                        "primitive": primitive,
                        "raw_params": raw_params,
                        "event_facts": deepcopy(event_facts),
                    },
                }

            try:
                maybe_result = fn(**params)
                step_result = await maybe_result if inspect.isawaitable(maybe_result) else maybe_result
            except Exception as exc:
                return {
                    "status": "failed",
                    "content": (
                        f"Macro '{macro_name}' failed at step {step_idx} "
                        f"({primitive}): {type(exc).__name__}: {exc}"
                    ),
                    "observations": {
                        "macro_name": macro_name,
                        "step_index": step_idx,
                        "primitive": primitive,
                        "completed_steps": len(results),
                        "total_steps": len(steps),
                    },
                }

            if isinstance(step_result, dict):
                normalized_result = dict(step_result)
                normalized_result.setdefault("success", True)
            elif isinstance(step_result, bool):
                normalized_result = {
                    "success": step_result,
                    "message": f"{primitive} {'ok' if step_result else 'failed'}",
                }
            elif isinstance(step_result, list):
                normalized_result = {
                    "success": True,
                    "message": f"{primitive} returned {len(step_result)} items",
                    "data": step_result,
                }
            else:
                normalized_result = {
                    "success": True,
                    "message": f"{primitive} completed",
                    "data": step_result,
                }

            results.append({"primitive": primitive, "result": normalized_result})
            if not normalized_result.get("success", False):
                return {
                    "status": "failed",
                    "content": (
                        f"Macro '{macro_name}' failed at step {step_idx} "
                        f"({primitive}): {normalized_result.get('message') or 'unknown failure'}"
                    ),
                    "observations": {
                        "macro_name": macro_name,
                        "step_index": step_idx,
                        "primitive": primitive,
                        "primitive_result": normalized_result,
                        "completed_steps": len(results) - 1,
                        "total_steps": len(steps),
                    },
                }

            primitive_meta = primitive_meta_by_name.get(primitive)
            if primitive_meta is not None:
                runtime_snapshot = apply_effects_to_snapshot(
                    {**dict(step), "params": params},
                    primitive_meta,
                    runtime_snapshot,
                )
                sync_agent_from_bridge_snapshot(self, runtime_snapshot)

            event_fact_key, event_fact_error = event_fact_key_for_primitive(
                primitive=primitive,
                params=params,
                resource_type=resource_type,
            )
            if event_fact_error:
                return {
                    "status": "failed",
                    "content": (
                        f"Macro '{macro_name}' could not publish event facts at step {step_idx} "
                        f"({primitive}): {event_fact_error}"
                    ),
                    "observations": {
                        "macro_name": macro_name,
                        "step_index": step_idx,
                        "primitive": primitive,
                        "primitive_result": normalized_result,
                        "completed_steps": len(results),
                    },
                }
            if event_fact_key:
                step_output, output_error = extract_step_output(
                    primitive=primitive,
                    params=params,
                    step_result=normalized_result,
                    resource_type=resource_type,
                )
                if output_error:
                    return {
                        "status": "failed",
                        "content": (
                            f"Macro '{macro_name}' could not publish event facts at step {step_idx} "
                            f"({primitive}): {output_error}"
                        ),
                        "observations": {
                            "macro_name": macro_name,
                            "step_index": step_idx,
                            "primitive": primitive,
                            "primitive_result": normalized_result,
                            "event_fact_path": f"event_facts.{event_fact_key}",
                            "completed_steps": len(results),
                        },
                    }
                target = event_facts
                tokens = [token for token in str(event_fact_key).split(".") if token]
                for token in tokens[:-1]:
                    child = target.get(token)
                    if not isinstance(child, dict):
                        child = {}
                        target[token] = child
                    target = child
                if tokens:
                    target[tokens[-1]] = deepcopy(step_output)

        if out_state:
            runtime_snapshot = resource_snapshot_set_field(
                runtime_snapshot,
                "current_state",
                out_state,
                profile=profile,
            )
            sync_agent_from_bridge_snapshot(self, runtime_snapshot)

        return {
            "status": "completed",
            "content": f"Recovery macro '{macro_name}' completed successfully",
            "observations": {
                "macro_name": macro_name,
                "completed_steps": len(results),
                "total_steps": len(steps),
                "event_facts": deepcopy(event_facts),
            },
        }

    def _build_failure_context(
        self,
        *,
        fn_name: str,
        fn_args: Dict[str, Any],
        result: Dict[str, Any] | None,
        final_status: str,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a normalized failure context payload for any failure type.
        """
        is_failed = isinstance(final_status, str) and final_status.startswith("failed")
        if not is_failed:
            if isinstance(result, dict) and isinstance(result.get("failure_context"), dict):
                return deepcopy(result.get("failure_context") or {})
            return {}

        raw_failure_context = {}
        raw_observations = {}
        if isinstance(result, dict):
            if isinstance(result.get("failure_context"), dict):
                raw_failure_context = deepcopy(result.get("failure_context") or {})
            if isinstance(result.get("observations"), dict):
                raw_observations = deepcopy(result.get("observations") or {})

        failure_event = build_failure_event(
            failed_task_id=str(fn_args.get("task_id") or "").strip(),
            failed_resource_jid=str(getattr(self, "jid", "") or "").strip(),
            failed_function_name=str(fn_name or "").strip(),
            final_status=str(final_status or "").strip(),
            part_name=str(fn_args.get("part_name") or "").strip(),
            base_failure_context=raw_failure_context,
            observations=raw_observations,
            state_before=state_before,
            state_after=state_after,
        )
        return deepcopy(failure_event.get("failure_context") or {})

    async def _wait_for_safety_decision(self, task_id: str) -> Optional[str]:
        """
        Block until a safety_decision is available for this task_id.
        No timeout: waits indefinitely until CCA replies.
        """
        while True:
            decision = self._safety_decisions.pop(task_id, None)
            if decision is not None:
                return decision
            await asyncio.sleep(0.1)
    # ------------------------------------------------------------------ #
    # Behaviours
    # ------------------------------------------------------------------ ##
    class _TaskInbox(CyclicBehaviour):
        """Long-running behaviour that processes incoming tasks sequentially."""
        async def run(self) -> None:
            agent: "ResourceAgent" = self.agent  # type: ignore

            # Poll inbox frequently but yield control if nothing arrives to keep agent responsive.
            msg = await self.receive(timeout=0.05)
            if not msg:
                return

            # ----- trust boundary ----- #
            # Give operators a simple safety net: reject unexpected senders early.
            if agent.allowed_senders and str(msg.sender) not in agent.allowed_senders:
                agent.logger.warning(
                    f"[Resource] Rejecting task from {msg.sender} (not allowed)"
                )
                await self._ack(msg, task_id="?", status="rejected:unauthorized")
                return

            # ----- envelope ----- #
            # Protocol metadata lets us bump behaviours later (e.g., plan/exec distinctions).
            protocol = msg.metadata.get("protocol", "")

            # ----- parse body ----- #
            try:
                data = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Resource] Malformed task body (not JSON).")
                await self._ack(msg, task_id="?", status="failed:bad_json")
                return

            task_id = data.get("task_id")
            instruction = data.get("instruction", "")
            phase_id = data.get("phase_id")  # optional multi-phase flow identifier

            if not task_id:
                agent.logger.warning("[Resource] Task without task_id.")
                await self._ack(msg, task_id="?", status="failed:missing_task_id")
                return

            agent.logger.info(
                f"[Resource] ← Task ({task_id}) from={msg.sender} proto={protocol}"
            )

            # ----- Tool selection ----- #
            # If the instruction already specifies a tool, honor it and skip the LLM.
            fn_name = None
            fn_args: Dict[str, Any] = {}
            if isinstance(instruction, dict):
                fn_name = instruction.get("function_name") or instruction.get("function")
                if isinstance(instruction.get("params"), dict):
                    fn_args = dict(instruction.get("params") or {})

            if not fn_name:
                # ----- EARLY ACK (non-recovery only) ----- #
                await self._ack(msg, task_id=task_id, status="accepted")
                try:
                    # Force the LLM to pick an explicit tool so we never free-text a task.
                    llm_resp = await asyncio.wait_for(
                        agent.ask_llm(
                            instruction,
                            with_functions=True,
                            force_tool=True,
                        ),
                        timeout=agent.llm_timeout_s,
                    )
                except asyncio.TimeoutError:
                    await self._ack(msg, task_id=task_id, status="llm_timeout")
                    return
                except Exception as e:
                    agent.logger.exception("[Resource] LLM failure")
                    await self._ack(
                        msg,
                        task_id=task_id,
                        status=f"failed:llm:{type(e).__name__}",
                    )
                    return

                fn_name, fn_args = _parse_function_call(llm_resp)
            if not fn_name:
                agent.logger.info(
                    f"[Resource] ({task_id}) no_tool_match; responding."
                )
                await self._ack(msg, task_id=task_id, status="no_tool_match")
                return

            start_safety_mode = str(
                fn_args.get("start_safety_mode") or ""
            ).strip().lower()
            # Recovery bridge macros keep the legacy default fast path unless
            # they explicitly request cca_check. Any task can now opt into the
            # same bypass with start_safety_mode=fast_path.
            is_recovery_macro = fn_name == "execute_recovery_macro"
            use_fast_path = bool(
                start_safety_mode == "fast_path"
                or (is_recovery_macro and start_safety_mode != "cca_check")
            )

            # ----- plumb routing/context ----- #
            # Pass routing info into the tool implementation for downstream logging/rpc calls.
            fn_args.setdefault("product_jid", str(msg.sender))
            fn_args.setdefault("task_id", task_id)
            if phase_id and "phase_id" not in fn_args:
                fn_args["phase_id"] = phase_id

            # ----- dispatch ----- #
            func = agent.executables.get(fn_name)
            if not func:
                agent.logger.warning(f"[Resource] Unknown tool '{fn_name}'")
                await self._ack(
                    msg,
                    task_id=task_id,
                    status=f"failed:unknown_tool:{fn_name}",
                )
                return

            if use_fast_path:
                # Fast path: self-allow, log, and proceed directly to execution.
                # Skip the safety_check round-trip but still notify CCA of
                # the running state so the plan FSA can track .start events.
                agent.logger.info(
                    "[Resource] Fast-path task %s fn=%s (skipping safety round-trip)",
                    task_id,
                    fn_name,
                )
                agent._safety_decisions[task_id] = "allow"
                try:
                    running_msg = Message(to=agent.cca_jid)
                    running_msg.set_metadata("type", "resource_event")
                    running_msg.body = json.dumps({
                        "task_id": task_id,
                        "resource_jid": str(agent.jid),
                        "function_name": fn_name,
                        "params": fn_args,
                        "status": "running",
                    })
                    await self.send(running_msg)
                except Exception:
                    agent.logger.exception(
                        "[Resource] Failed to send running event for fast-path task %s (ignored).",
                        task_id,
                    )
            else:
                # ----- EARLY ACK (normal tasks) ----- #
                await self._ack(msg, task_id=task_id, status="accepted")

                # ----- RESOURCE EVENT (FOR SAFETY) NOTIFICATION TO CCA ----- #
                try:
                    # 1) Send request permission, not running
                    resource_msg = Message(to=agent.cca_jid)
                    resource_msg.set_metadata("type", "resource_event")
                    resource_msg.body = json.dumps({
                        "task_id": task_id,
                        "resource_jid": str(agent.jid),
                        "function_name": fn_name,
                        "params": fn_args,
                        "status": "safety_check",   # <-- REQUEST permission
                    })
                    await self.send(resource_msg)
                except Exception:
                    agent.logger.exception("[Resource] Failed to send resource_event to CCA (ignored).")

            # 2) Wait for CCA decision (instant for recovery macros, blocks for normal tasks)
            decision = await agent._wait_for_safety_decision(task_id)

            if decision == "block":
                await self._ack(msg, task_id=task_id, status="blocked")
                return

            if not use_fast_path:
                # ---------------------------
                #  SAFETY PASSED → RUNNING
                # ---------------------------
                await self._ack(msg, task_id=task_id, status="running")

                running_msg = Message(to=agent.cca_jid)
                running_msg.set_metadata("type", "resource_event")
                running_msg.body = json.dumps({
                    "task_id": task_id,
                    "resource_jid": str(agent.jid),
                    "function_name": fn_name,
                    "params": fn_args,
                    "status": "running",
                })
                await self.send(running_msg)

            # ---------------------------
            #  EXECUTE THE TOOL
            # ---------------------------
            state_before = agent._snapshot_state()
            state_after = state_before
            result: Dict[str, Any] | None = None
            try:
                # Filter fn_args to only params the function accepts.
                # Functions that declare **kwargs receive everything;
                # others get only the params in their signature.
                sig = inspect.signature(func)
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                if accepts_var_kw:
                    filtered_args = fn_args
                else:
                    accepted = set(sig.parameters.keys())
                    filtered_args = {k: v for k, v in fn_args.items() if k in accepted}

                result = await asyncio.wait_for(
                    func(**filtered_args),
                    timeout=agent.tool_timeout_s,
                )
                state_after = agent._snapshot_state()
                final_status = (result or {}).get("status") or "completed"

                # ----- RESOURCE EVENT: TASK FINISHED (notify CCA) ----- #
                try:
                    failure_context = agent._build_failure_context(
                        fn_name=fn_name,
                        fn_args=fn_args,
                        result=result if isinstance(result, dict) else None,
                        final_status=final_status,
                        state_before=state_before,
                        state_after=state_after,
                    )

                    done_msg = Message(to=agent.cca_jid)
                    done_msg.set_metadata("type", "resource_event")
                    done_msg.body = json.dumps({
                        "task_id": task_id,
                        "resource_jid": str(agent.jid),
                        "function_name": fn_name,
                        "params": fn_args,
                        "status": final_status,  # e.g. "completed", "blocked", etc.
                        "failure_context": failure_context,
                        "current_state": state_after.get("current_state", "idle"),
                    })
                    # fire-and-forget so we don't block on CCA
                    asyncio.create_task(self.send(done_msg))
                except Exception:
                    agent.logger.exception(
                        "[Resource] Failed to send final resource_event to CCA (ignored)."
                    )

            except asyncio.TimeoutError:
                agent.logger.exception("[Resource] Tool execution timeout")
                final_status = "failed:tool_timeout"
            except Exception as e:
                agent.logger.exception("[Resource] Tool execution failed")
                final_status = f"failed:tool:{type(e).__name__}"

            # Notify CCA of failure so it can clean up running_aps / FSA state.
            if isinstance(final_status, str) and final_status.startswith("failed"):
                try:
                    state_after = agent._snapshot_state()
                    failure_context = agent._build_failure_context(
                        fn_name=fn_name,
                        fn_args=fn_args,
                        result=result if isinstance(result, dict) else None,
                        final_status=final_status,
                        state_before=state_before,
                        state_after=state_after,
                    )
                    fail_msg = Message(to=agent.cca_jid)
                    fail_msg.set_metadata("type", "resource_event")
                    fail_msg.body = json.dumps({
                        "task_id": task_id,
                        "resource_jid": str(agent.jid),
                        "function_name": fn_name,
                        "params": fn_args,
                        "status": final_status,
                        "failure_context": failure_context,
                        "current_state": state_after.get("current_state", "idle"),
                    })
                    await self.send(fail_msg)
                except Exception:
                    agent.logger.exception(
                        "[Resource] Failed to send fail resource_event to CCA."
                    )

            # ---------------------------
            #  SEND FINAL ACK TO PA
            # ---------------------------
            ack_content = ""
            ack_observations = None
            if isinstance(result, dict):
                ack_content = str(result.get("content") or "").strip()
                raw_observations = result.get("observations")
                if isinstance(raw_observations, dict):
                    ack_observations = raw_observations
            await self._ack(
                msg,
                task_id=task_id,
                status=final_status,
                content=ack_content,
                observations=ack_observations,
            )

        async def _ack(
            self,
            msg: Message,
            *,
            task_id: Optional[str],
            status: str,
            content: str = "",
            observations: Dict[str, Any] | None = None,
        ) -> None:
            """Send an acknowledgement/status update back to the originating ProductAgent."""
            reply = Message(to=str(msg.sender))
            reply.set_metadata("type", "ack")
            payload = {"task_id": task_id, "status": status}
            if content:
                payload["content"] = str(content).strip()
            if isinstance(observations, dict) and observations:
                payload["observations"] = observations
            reply.body = json.dumps(payload)
            await self.send(reply)

    class _SafetyDecisionInbox(CyclicBehaviour):
        """Receives safety_decision messages from CCA and stores them on the agent."""
        async def run(self) -> None:
            agent: "ResourceAgent" = self.agent  # type: ignore

            msg = await self.receive(timeout=0.05)
            if not msg:
                return

            try:
                data = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Resource] Malformed safety_decision body")
                return

            task_id = data.get("task_id")
            decision = data.get("decision")

            if not task_id or decision not in ("allow", "block"):
                agent.logger.warning(
                    "[Resource] Invalid safety_decision message: %s", data
                )
                return

            agent._safety_decisions[task_id] = decision
            agent.logger.info(
                "[Resource] Stored safety_decision=%s for task=%s",
                decision,
                task_id,
            )

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _parse_function_call(resp):
    """Normalize the OpenAI response into (tool_name, args_dict)."""
    fc = resp.get("function_call")
    if not isinstance(fc, dict):
        return None, {}

    name = fc.get("name")
    args_raw = fc.get("arguments", {})

    if isinstance(args_raw, str):
        # OpenAI may return arguments as a JSON string; parse defensively.
        try:
            return name, json.loads(args_raw)
        except json.JSONDecodeError:
            return name, {}

    return name, args_raw

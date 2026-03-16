"""Descriptor-driven deterministic bridge macro compilation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cais_spade_llm.resources.resource_profile import (
    get_resource_profile,
)


class BridgeCompilerMixin:
    @staticmethod
    def _bridge_ref(path: str) -> dict[str, str]:
        return {"context_ref": str(path)}

    def _bridge_compile_single_primitive_macro(
        self,
        prepared_bridge_request: dict[str, Any] | None = None,
        *,
        event: dict[str, Any],
        resource_jid: str,
        start_state: str,
        primitive_name: str,
    ) -> dict[str, Any]:
        params = deepcopy(dict(event.get("primitive_params") or {}))
        out_state = str(
            (event.get("expected_resource_delta") or {}).get("to", "") or start_state
        ).strip() or start_state
        return {
            "resource_jid": resource_jid,
            "macro_name": str(event.get("event_name") or primitive_name).strip() or primitive_name,
            "description": str(
                event.get("rationale") or f"Execute {primitive_name} as a bridge primitive."
            ).strip(),
            "rationale": str(event.get("rationale") or "").strip(),
            "expected_start_state": start_state,
            "part_name": str(event.get("part_name") or "").strip(),
            "task_params": {},
            "task_metadata": {
                "in_state": start_state,
                "out_state": out_state,
                "required_context_keys": [],
                "context_mapping": {},
                "part_transition": None,
            },
            "primitive_steps": [{"primitive": primitive_name, "params": params}],
        }

    def _compile_bridge_events_to_macro_tasks(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        approved_events: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not approved_events:
            return None, "no approved bridge events available for deterministic compilation"

        bridge_resources = deepcopy(prepared_bridge_request.get("bridge_resources") or {})
        effective_resources = self._bridge_effective_resource_facts(
            bridge_resources=bridge_resources,
            projected_resource_snapshots=None,
        )
        current_states = {
            jid: str((info or {}).get("current_state") or "").strip()
            for jid, info in effective_resources.items()
        }
        macro_tasks: list[dict[str, Any]] = []
        for event in approved_events:
            if not isinstance(event, dict):
                continue
            resource_jid = str(event.get("resource_jid", "")).strip()
            if not resource_jid:
                return None, "approved bridge event is missing resource_jid"
            start_state = (
                str((event.get("expected_resource_delta") or {}).get("from", "")).strip()
                or current_states.get(resource_jid)
                or str(
                    (bridge_resources.get(resource_jid) or {})
                    .get("bridge_snapshot", {})
                    .get("current_state")
                    or ""
                ).strip()
                or "idle"
            )
            semantic_kind = self._bridge_event_semantic_kind(event)
            resource_entry = dict(bridge_resources.get(resource_jid) or {})
            bridge_snapshot = dict(resource_entry.get("bridge_snapshot") or {})
            resource_type = str(
                resource_entry.get("resource_type")
                or bridge_snapshot.get("resource_type")
                or dict(bridge_snapshot.get("resource_core") or {}).get("resource_type")
                or "resource"
            ).strip().lower() or "resource"
            profile = get_resource_profile(resource_type)

            compiler_hook = (profile.compiler_map or {}).get(semantic_kind)
            if callable(compiler_hook):
                macro = compiler_hook(
                    self,
                    prepared_bridge_request,
                    event=event,
                    resource_jid=resource_jid,
                    start_state=start_state,
                    primitive_name=semantic_kind,
                )
            elif isinstance(compiler_hook, str) and compiler_hook.strip():
                method = getattr(self, compiler_hook.strip(), None)
                if not callable(method):
                    return None, (
                        f"profile compiler '{compiler_hook}' is not available for "
                        f"resource type '{resource_type}'"
                    )
                macro = method(
                    prepared_bridge_request,
                    event=event,
                    resource_jid=resource_jid,
                    start_state=start_state,
                    primitive_name=semantic_kind,
                )
            else:
                primitive_name = str(
                    (profile.family_to_primitive or {}).get(semantic_kind) or semantic_kind
                ).strip()
                if not primitive_name:
                    return None, (
                        f"deterministic compiler does not yet support bridge event "
                        f"'{str(event.get('event_name', '')).strip() or semantic_kind}'"
                    )
                macro = self._bridge_compile_single_primitive_macro(
                    prepared_bridge_request,
                    event=event,
                    resource_jid=resource_jid,
                    start_state=start_state,
                    primitive_name=primitive_name,
                )
            macro_tasks.append(macro)
            out_state = str((macro.get("task_metadata") or {}).get("out_state") or "").strip()
            if out_state:
                current_states[resource_jid] = out_state

        primary_obligation = deepcopy(
            (prepared_bridge_request.get("obligation_targets") or [None])[0]
        )
        return {
            "primary_obligation": primary_obligation,
            "bridge_event_summary": deepcopy(approved_events),
            "macro_tasks": macro_tasks,
        }, None

    def _bridge_final_plan_realizes_approved_events(
        self,
        *,
        approved_events: list[dict[str, Any]],
        realized_events: list[dict[str, Any]],
    ) -> tuple[bool, str | None]:
        if not approved_events:
            return True, None

        realized_index = 0
        for approved_event in approved_events:
            if not isinstance(approved_event, dict):
                continue
            approved_keys = self._bridge_event_condition_keys([approved_event])
            if not approved_keys:
                continue
            expected_resource = str(approved_event.get("resource_jid", "") or "").strip()
            matched = False
            while realized_index < len(realized_events):
                realized_event = realized_events[realized_index]
                realized_index += 1
                realized_resource = str(realized_event.get("resource_jid", "") or "").strip()
                realized_keys = self._bridge_event_condition_keys([realized_event])
                if expected_resource and realized_resource and expected_resource != realized_resource:
                    continue
                if approved_keys <= realized_keys:
                    matched = True
                    break
            if not matched:
                return False, (
                    f"final_plan did not realize approved bridge event "
                    f"'{str(approved_event.get('event_name', '')).strip() or expected_resource}'"
                )
        return True, None

    def _bridge_synthesize_plan_rewrite(
        self,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, Any]:
        marked_reentry_context = deepcopy(
            prepared_bridge_request.get("marked_reentry_context")
            or prepared_bridge_request.get("continuation_context")
            or {}
        )
        resume_task_ids: list[str] = []
        seen: set[str] = set()
        for suffix in (marked_reentry_context.get("pending_suffix_summary") or []):
            if not isinstance(suffix, dict):
                continue
            if str(suffix.get("role", "")).strip() != "resume_suffix":
                continue
            task_ids = [
                str(task_id).strip()
                for task_id in (suffix.get("pending_task_ids") or [])
                if str(task_id).strip()
            ]
            entry_task_id = str(suffix.get("entry_task_id", "")).strip()
            if entry_task_id and entry_task_id in task_ids:
                task_ids = task_ids[task_ids.index(entry_task_id):]
            for task_id in task_ids:
                if task_id in seen:
                    continue
                seen.add(task_id)
                resume_task_ids.append(task_id)
        return {
            "replace_failed_branch": True,
            "resume_task_ids": resume_task_ids,
        }

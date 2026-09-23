"""Read-only capability functions, primitive steps, and generated recovery programs."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any

from nicegui import context, ui

from cais_spade_llm.ui.bridge import SystemBridge


def resource_function_rows(model: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Group owned functions and passive event participation for one resource.

    Args:
        model: Configured or live capability descriptor.

    Returns:
        Function groups and events performed by other resources.
    """
    resource_id = str(model["resource_id"])
    executable = model.get("executable_tasks")
    owned: dict[tuple[str, str], dict[str, Any]] = {}
    participating: list[dict[str, Any]] = []
    for event in sorted(model["events"], key=lambda row: row["event_id"]):
        actor = str(event["parameter_bindings"]["resource_id"]["equals"])
        variant = {
            "event_id": event["event_id"],
            "parameter_bindings": deepcopy(event["parameter_bindings"]),
            "guards": deepcopy(event["guards"]),
            "updates": deepcopy(event["updates"]),
            "participants": list(event.get("participants") or []),
        }
        if actor != resource_id:
            participating.append(
                {"event_name": event["event_name"], "actor": actor, **variant}
            )
            continue
        event_name = str(event["event_name"])
        function_name = str(event.get("function_name") or event_name)
        key = (event_name, function_name)
        if key not in owned:
            status = str(event.get("program_status") or "")
            if status == "planned":
                note = str((event.get("program") or {}).get("availability_note") or "")
                availability = "Planned only — no Gazebo executor"
                if note:
                    availability += f". {note}"
            elif executable is None:
                availability = "Configured — start the system to check Gazebo execution"
            elif event_name in executable:
                availability = "Gazebo executable"
            else:
                availability = "No Gazebo executor bound"
            owned[key] = {
                "event_name": event_name,
                "function_name": function_name,
                "program_status": status,
                "availability": availability,
                "program": deepcopy(event.get("program") or {}),
                "variants": [],
            }
        owned[key]["variants"].append(variant)
    return {"functions": list(owned.values()), "participating": participating}


def _render_json(label: str, value: Any) -> None:
    with ui.expansion(label, icon="data_object").classes("w-full"):
        ui.code(json.dumps(value, indent=2, ensure_ascii=False), language="json").classes(
            "w-full"
        )


def render_resource_function_rows(rows: dict[str, list[dict[str, Any]]]) -> None:
    """Render a selected resource's functions without dispatch controls."""
    functions = rows["functions"]
    if not functions:
        ui.label("This resource owns no capability functions.").classes("text-sm text-slate-600")
    for item in functions:
        event_name = item["event_name"]
        function_name = item["function_name"]
        with ui.expansion(event_name, icon="precision_manufacturing").classes("w-full"):
            if function_name != event_name:
                ui.label(f"Executed function: {function_name}").classes("text-sm font-medium")
            ui.label(item["availability"]).classes("text-sm text-slate-600")
            program = item["program"]
            steps = list(program.get("steps") or [])
            if not steps:
                ui.label("No ordered primitive program is declared.").classes(
                    "text-sm text-amber-700"
                )
            for index, step in enumerate(steps, start=1):
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label(f"{index}. {step.get('id', '')}").classes("text-sm font-medium")
                    ui.label("→").classes("text-slate-400")
                    ui.label(str(step.get("op") or "")).classes("text-sm text-blue-700")
                if step.get("note"):
                    ui.label(str(step["note"])).classes("text-xs text-slate-600")
                if step.get("physical_position_required"):
                    ui.label("Recorded physical position required").classes(
                        "text-xs text-blue-700"
                    )
                if step.get("params"):
                    _render_json(f"{step.get('id', '')} parameter sources", step["params"])
                if step.get("when"):
                    _render_json(f"{step.get('id', '')} conditions", step["when"])
            _render_json("Function conditions and effects", {
                key: program[key]
                for key in ("entry_state", "success_state", "entry_guards", "effects")
                if key in program
            })
            with ui.expansion(
                f"Capability event variants ({len(item['variants'])})", icon="account_tree"
            ).classes("w-full"):
                for variant in item["variants"]:
                    _render_json(f"event_id {variant['event_id']}", variant)
    if rows["participating"]:
        with ui.expansion(
            f"Shared events this resource participates in ({len(rows['participating'])})",
            icon="hub",
        ).classes("w-full"):
            for event in rows["participating"]:
                _render_json(
                    f"event_id {event['event_id']}: {event['event_name']} — {event['actor']}",
                    event,
                )


def recovery_program_rows(
    recoveries: list[dict[str, Any]], agents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Project current generated programs without changing their approval state."""
    names = {
        str(agent.get("jid") or ""): str(agent.get("name") or "")
        for agent in agents
        if isinstance(agent, dict)
    }
    rows: list[dict[str, Any]] = []
    for recovery in recoveries:
        if not isinstance(recovery, dict):
            continue
        debug = recovery.get("recovery_debug")
        if not isinstance(debug, dict):
            continue
        source = "final_output"
        final = debug.get("final_output")
        program = final.get("accepted_primitive_program") if isinstance(final, dict) else None
        if not isinstance(program, list) or not program:
            source = "multi_turn_session"
            session = debug.get("multi_turn_session")
            program = session.get("accepted_primitive_program") if isinstance(session, dict) else None
        if not isinstance(program, list):
            continue
        for item in program:
            if not isinstance(item, dict):
                continue
            jid = str(item.get("resource_jid") or "")
            rows.append({
                "product_jid": str(recovery.get("product_jid") or ""),
                "product_name": str(recovery.get("product_name") or recovery.get("product_jid") or "product"),
                "resource_jid": jid,
                "resource_name": names.get(jid) or jid or "Resource unavailable",
                "event_name": str(item.get("event_name") or item.get("description") or ""),
                "approval_state": str(recovery.get("recovery_approval_state") or "none"),
                "status": str(recovery.get("status") or ""),
                "source": source,
                "primitive_steps": [
                    deepcopy(step) for step in (item.get("primitive_steps") or [])
                    if isinstance(step, dict)
                ],
            })
    return rows


def render_generated_recovery_programs(
    bridge: SystemBridge,
) -> Callable[[], Awaitable[None]]:
    """Render generated Gazebo recovery programs through existing bridge reads."""
    client = context.client
    previous: list[dict[str, Any]] | None = None
    with ui.card().classes("w-full"):
        ui.label("Generated recovery programs").classes("text-lg font-semibold")
        ui.label(
            "Current ProductAgent recovery composition; approval and execution remain on the Run page."
        ).classes("text-sm text-slate-500")
        body = ui.column().classes("w-full gap-2")

    async def refresh() -> None:
        nonlocal previous
        if getattr(client, "_deleted", False):
            return
        recoveries = await asyncio.to_thread(bridge.get_runtime_recoveries)
        agents = await asyncio.to_thread(bridge.get_agent_statuses) if recoveries else []
        rows = recovery_program_rows(recoveries, agents)
        if rows == previous or getattr(client, "_deleted", False):
            return
        previous = deepcopy(rows)
        body.clear()
        with body:
            if not rows:
                ui.label("No generated recovery program is available.").classes(
                    "text-sm text-slate-500"
                )
            for item in rows:
                with ui.expansion(
                    f"{item['product_name']} · {item['resource_name']} · {item['event_name']}",
                    icon="route",
                ).classes("w-full"):
                    ui.label(
                        f"Approval: {item['approval_state']} · Recovery: {item['status']} · Source: {item['source']}"
                    ).classes("text-xs text-slate-600")
                    for index, step in enumerate(item["primitive_steps"], start=1):
                        primitive = str(step.get("primitive") or step.get("function_name") or "")
                        ui.label(f"{index}. {primitive}").classes("text-sm font-medium")
                        if step.get("params"):
                            _render_json("Parameters", step["params"])
    return refresh

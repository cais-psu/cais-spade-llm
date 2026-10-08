from __future__ import annotations

"""Explicit non-dispatching preparation on the recovery evidence page."""

import asyncio
import json

from nicegui import ui

from cais_spade_llm.recovery_framework.gazebo_safety_preparation import prepare_and_check


def render_safety_preparation(bridge) -> None:
    """Prepare exact candidate programs and display saved evidence on request."""
    with ui.expansion("Gazebo safety preparation", icon="fact_check").classes("w-full"):
        ui.label("Capture the current scene, prepare supported resource programs, and check the selected predefined specifications.")
        ui.label("Preparation and checking only; execution remains under CCA admission.").classes("text-sm text-slate-600")
        editor = ui.textarea("Recovery programs (JSON)", value=json.dumps(
            {"recovery_id": "", "programs": []}, indent=2)).classes("w-full").props("autogrow")
        candidate = ui.select(['GAZEBO_MOTION_SAFE','GAZEBO_MOTION_CONFLICT'],
                              value='GAZEBO_MOTION_SAFE',label='Supplied mock motion candidate')

        async def load_candidate():
            from cais_spade_llm.recovery_framework.gazebo_safety_preparation import (
                build_supplied_candidate,
            )
            try:
                request = await asyncio.to_thread(build_supplied_candidate,bridge,candidate.value)
                editor.value = json.dumps(request,indent=2)
            except (ValueError,KeyError,RuntimeError,OSError) as exc:
                ui.notify(str(exc),type='negative')

        ui.button('Load supplied candidate',on_click=load_candidate)
        ui.label('Mock candidate input; its checkpoint and motion must come from Gazebo and the planner.').classes('text-sm')
        ui.label("Use exact resource_id, primitive_steps, params, and source event/step references from the recovery plan.").classes("text-sm")
        output = ui.column().classes("w-full")

        async def prepare() -> None:
            button.disable()
            try:
                request = json.loads(editor.value)
                result = await asyncio.to_thread(prepare_and_check, bridge, request)
                output.clear()
                with output:
                    ui.label("Safety check: " + result["status"]).classes("font-semibold")
                    ui.label('Execution permission: not issued').classes('text-sm')
                    for program in result["prepared_programs"]:
                        ui.label(f"{program['resource_id']}: {program['status']}")
                    for row in result.get("resource_support", []):
                        contract = row["primitive_model"]
                        support = f"{contract['id']} v{contract['version']}" if contract else "primitive model unavailable"
                        ui.label(f"{row['resource_id']}: {row['observation_status']}; {support}")
                    for row in result["unresolved"]:
                        ui.label((row.get("resource_id") or row.get("part") or "Evidence") + ": " + row["reason"])
                    analysis = result.get("analysis") or {}
                    if analysis.get("reason"):
                        ui.label(analysis["reason"])
                    if analysis.get("pending_rule_ids"):
                        ui.label("Pending requirements: " + ", ".join(analysis["pending_rule_ids"]))
                    with ui.expansion("Checkpoint, trajectories and safety evidence").classes("w-full"):
                        ui.code(json.dumps(result, indent=2), language="json").classes("w-full")
                    ui.button("Download evidence", icon="download",
                              on_click=lambda: ui.download.file(result["artifact_path"]))
            except (ValueError, TypeError, OSError) as exc:
                ui.notify(str(exc), type="negative")
            finally:
                button.enable()

        button = ui.button("Prepare and check", icon="fact_check", on_click=prepare)

"""Read saved Gazebo delivery evidence without executing or rewriting anything."""

from __future__ import annotations

import json
from pathlib import Path

from nicegui import ui

from cais_spade_llm.recovery_framework import read_json
from cais_spade_llm.recovery_framework.delivery import RUN_DIRECTORY


def render_gazebo_delivery_runs(directory: Path = RUN_DIRECTORY) -> None:
    """Display recorded configuration, acknowledged transitions, and incomplete outcomes."""
    with ui.expansion('Gazebo delivery runs', icon='precision_manufacturing').classes('w-full'):
        ui.label('Recorded Gazebo evidence. Selecting a report performs reads only.').classes('text-sm')
        selector = ui.select({}, label='Saved Gazebo run').classes('w-full')
        content = ui.column().classes('w-full')

        def display() -> None:
            content.clear()
            if not selector.value:
                return
            with content:
                try:
                    report = read_json(selector.value)
                    if report.get('schema_version') != 1 or report.get('evidence') != 'gazebo':
                        raise ValueError('Unsupported Gazebo report format')
                    ui.label(f"Outcome: {report['outcome']['status']}").classes('font-semibold')
                    if report['outcome'].get('reason'):
                        ui.label(report['outcome']['reason']).classes('text-sm')
                    rows = [{'revision': record['revision'], 'task': record['task']['event_name'],
                             'resource': record['task']['resource_id'],
                             'part': record['acknowledgement']['observations']['part_name']}
                            for record in report['transitions']]
                    ui.table(columns=[{'name': key, 'label': key, 'field': key}
                                      for key in ('revision', 'task', 'resource', 'part')],
                             rows=rows, row_key='revision').classes('w-full')
                    with ui.expansion('Recorded setup and observations').classes('w-full'):
                        ui.code(json.dumps(report, indent=2), language='json').classes('w-full')
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    ui.label(f'Gazebo report unavailable: {exc}').classes('text-amber-700')

        def refresh() -> None:
            selector.options = {str(path): path.parent.name for path in sorted(directory.glob('*/run.json'))}
            selector.update()
            display()

        selector.on_value_change(display)
        ui.button('Refresh Gazebo reports', on_click=refresh, icon='refresh')
        refresh()

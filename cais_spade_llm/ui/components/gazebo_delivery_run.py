"""Read recovery evidence and archive only on an explicit operator action."""

from __future__ import annotations

import json
from pathlib import Path

from nicegui import ui

from cais_spade_llm.recovery_framework import read_json
from cais_spade_llm.recovery_framework.delivery import RUN_DIRECTORY
from cais_spade_llm.recovery_framework.reports import archive_latest


def render_gazebo_delivery_runs(directory: Path = RUN_DIRECTORY, *, environment: bool = False) -> None:
    """Display recorded configuration, acknowledged transitions, and incomplete outcomes."""
    title = 'Environmental runs' if environment else 'Gazebo delivery runs'
    with ui.expansion(title, icon='precision_manufacturing').classes('w-full'):
        ui.label(f'Recorded {"environmental" if environment else "Gazebo"} evidence. Selecting a report performs reads only.').classes('text-sm')
        selector = ui.select({}, label='Saved Gazebo run').classes('w-full')
        content = ui.column().classes('w-full')
        displayed_run = {'run_id': None}

        def display() -> None:
            content.clear()
            if not selector.value:
                return
            with content:
                try:
                    report = read_json(selector.value)
                    displayed_run['run_id'] = report.get('run_id')
                    if (not environment and (report.get('schema_version') != 1 or report.get('evidence') != 'gazebo')):
                        raise ValueError('Unsupported Gazebo report format')
                    if environment and not {'outcome', 'models', 'transitions'} <= report.keys():
                        raise ValueError('Unsupported environmental report format')
                    ui.label(f"Outcome: {report['outcome']['status']}").classes('font-semibold')
                    if report['outcome'].get('reason'):
                        ui.label(report['outcome']['reason']).classes('text-sm')
                    rows = [{'revision': record['revision'], 'task': record['task']['event_name'],
                             'resource': record['task']['resource_id'],
                             'part': record['acknowledgement']['observations']['part_name']}
                            for record in report['transitions']] if not environment else []
                    ui.table(columns=[{'name': key, 'label': key, 'field': key}
                                      for key in ('revision', 'task', 'resource', 'part')],
                             rows=rows, row_key='revision').classes('w-full')
                    with ui.expansion('Recorded setup and observations').classes('w-full'):
                        ui.code(json.dumps(report, indent=2), language='json').classes('w-full')
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    ui.label(f'Gazebo report unavailable: {exc}').classes('text-amber-700')

        def refresh() -> None:
            paths = sorted(directory.glob('*/run.json')) + sorted(directory.glob('archive/*/run.json'))
            selector.options = {str(path): path.parent.name for path in paths}
            if not selector.value and (directory / 'latest/run.json').exists():
                selector.value = str(directory / 'latest/run.json')
            selector.update()
            display()

        def archive() -> None:
            try:
                if selector.value != str(directory / 'latest/run.json') or not displayed_run['run_id']:
                    raise ValueError('Select the latest run before archiving')
                path = archive_latest(directory, expected_run_id=displayed_run['run_id'])
                ui.notify(f'Archived {path.parent.name}')
                refresh()
            except (OSError, ValueError) as exc:
                ui.notify(str(exc), type='warning')

        selector.on_value_change(display)
        ui.button('Refresh Gazebo reports', on_click=refresh, icon='refresh')
        ui.button('Archive this run', on_click=archive, icon='archive')
        refresh()

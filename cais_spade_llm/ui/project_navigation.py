"""Keep project links, retained NiceGUI tabs, and browser history in sync."""

from __future__ import annotations

import json

from nicegui import context, ui


def bind_project_tabs(tabs: ui.tabs, path: str, names: tuple[str, ...]) -> None:
    """Select same-project anchors locally while preserving normal link behavior."""
    client = context.client
    event = f"project_tab_{tabs.id}"
    history_navigation = False

    def selected(e) -> None:
        nonlocal history_navigation
        history_navigation = True
        try:
            tabs.value = e.args if e.args in names else "run"
        finally:
            history_navigation = False

    def changed() -> None:
        if not history_navigation and client.has_socket_connection:
            ui.run_javascript(f"window.caisProjectTabHistory?.({json.dumps(tabs.value)})")

    ui.on(event, selected)
    tabs.on_value_change(changed)
    ui.add_body_html(
        """<script>
    (() => {
        const path = PATH, names = NAMES, event = EVENT;
        function selection(url) {
            const name = url.searchParams.get('tab');
            return names.includes(name) ? name : 'run';
        }
        window.caisProjectTabHistory = name => {
            const url = new URL(location.href);
            if (url.pathname !== path || url.searchParams.get('tab') === name) return;
            url.searchParams.set('tab', name);
            history.pushState(null, '', url);
        };
        document.addEventListener('click', e => {
            const link = e.target.closest('a[href]');
            if (!link || e.defaultPrevented || e.button !== 0 || e.ctrlKey || e.metaKey ||
                e.shiftKey || e.altKey || (link.target && link.target !== '_self') || link.hasAttribute('download')) return;
            const url = new URL(link.href, location.href);
            if (url.origin !== location.origin || url.pathname !== path) return;
            e.preventDefault();
            const name = selection(url);
            window.caisProjectTabHistory(name);
            emitEvent(event, name);
        });
        window.addEventListener('popstate', () => {
            if (location.pathname === path) emitEvent(event, selection(new URL(location.href)));
        });
    })();
    </script>""".replace("PATH", json.dumps(path))
        .replace("NAMES", json.dumps(names))
        .replace("EVENT", json.dumps(event))
    )

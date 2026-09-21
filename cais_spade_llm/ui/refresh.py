"""Page-owned polling that pauses while the operator cannot see the page."""

from __future__ import annotations

import inspect
from collections.abc import Callable

from nicegui import context, ui


class PageRefresh:
    """Keep timer work within one connected, visible page's lifetime."""

    def __init__(self, is_active: Callable[[], bool] | None = None) -> None:
        self.client = context.client
        self.is_active = is_active
        self.connected = True
        self.timers = []
        self.client.on_connect(lambda: setattr(self, "connected", True))
        self.client.on_disconnect(lambda: setattr(self, "connected", False))
        self.client.on_delete(self.cancel)

    def active(self) -> bool:
        """Return whether polling can publish updates to this page."""
        return (
            not getattr(self.client, "_deleted", False)
            and self.connected
            and getattr(self.client, "_cais_visible", True)
            and (self.is_active is None or self.is_active())
        )

    def timer(self, interval: float, callback: Callable, **kwargs):
        """Await each refresh before the next tick; cancel it with the page."""

        async def refresh() -> None:
            if not self.active():
                return
            result = callback()
            if inspect.isawaitable(result):
                await result

        timer = ui.timer(interval, refresh, **kwargs)
        self.timers.append(timer)
        return timer

    def cancel(self) -> None:
        """Stop callbacks when navigation deletes their page."""
        for timer in self.timers:
            timer.cancel(with_current_invocation=True)
        self.timers.clear()

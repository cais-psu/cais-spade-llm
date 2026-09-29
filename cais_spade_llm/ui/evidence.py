"""Bounded, deferred views of saved evidence shared by project pages."""

from __future__ import annotations

import asyncio
import codecs
import json
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Any

from nicegui import context, ui

PREVIEW_BYTES = 64 * 1024
_CACHE_BYTES = 16 * 1024 * 1024
_cache: OrderedDict[tuple[str, int, int], Any] = OrderedDict()
_cache_lock = Lock()


def read_json_cached(path: Path) -> Any:
    """Read unchanged JSON once, with a bounded cache of file contents."""
    path = path.resolve()
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    value = json.loads(path.read_text(encoding="utf-8"))
    if stat.st_size <= _CACHE_BYTES:
        with _cache_lock:
            for old in list(_cache):
                if old[0] == str(path):
                    del _cache[old]
            _cache[key] = value
            while sum(item[2] for item in _cache) > _CACHE_BYTES:
                _cache.popitem(last=False)
    return value


def contained_path(root: Path, reference: str | Path) -> Path:
    """Resolve a file reference without escaping its evidence directory."""
    path = (root / reference).resolve()
    path.relative_to(root.resolve())
    return path


def read_text_page(path: Path, offset: int = 0) -> tuple[str, int, int]:
    """Read at most 64 KiB, retaining UTF-8 boundaries between pages."""
    with path.open("rb") as stream:
        stream.seek(max(0, offset))
        data = stream.read(PREVIEW_BYTES)
        size = stream.seek(0, 2)
    decoder = codecs.getincrementaldecoder("utf-8")()
    text = decoder.decode(data, final=offset + len(data) >= size)
    return text, offset + len(data) - len(decoder.getstate()[0]), size


class BackgroundSection:
    """Read outside the UI loop and discard obsolete or deleted-page updates."""

    def __init__(self, container: Any = None) -> None:
        self.client = context.client
        self.container = container if container is not None else ui.column().classes("w-full")
        self.revision = 0
        self.timer = None
        self.pending: tuple[Callable, Callable] | None = None
        self.client.on_delete(self.cancel)
        self.client.on_disconnect(self.cancel)
        self.client.on_connect(self._reconnect)

    def cancel(self) -> None:
        """Invalidate pending UI work when the client goes away."""
        self.revision += 1
        if self.timer is not None:
            self.timer.cancel(with_current_invocation=True)
            self.timer = None

    def _reconnect(self) -> None:
        if self.pending and not self.container.is_deleted:
            self.load(*self.pending)

    def load(self, reader: Callable, render: Callable) -> None:
        """Schedule one read and render only its most recent result."""
        self.cancel()
        if self.container.is_deleted or self.client._deleted:
            return
        self.pending = (reader, render)
        revision = self.revision
        self.container.clear()
        with self.container:
            ui.label("Loading…").classes("text-sm text-slate-500")

        async def finish() -> None:
            try:
                value = await asyncio.to_thread(reader)
            except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
                value = exc
            if revision != self.revision or self.client._deleted or self.container.is_deleted:
                return
            self.pending = None
            self.container.clear()
            with self.container:
                if isinstance(value, Exception):
                    ui.label(f"Unavailable: {value}").classes("text-amber-700")
                    ui.button("Retry", on_click=lambda: self.load(reader, render)).props("flat")
                else:
                    render(value)

        with self.container:
            self.timer = ui.timer(0.01, finish, once=True)


def render_file(path: Path) -> None:
    """Show exact recorded text in bounded pages with the original download."""
    offsets = [0]
    next_offset = 0
    total = 0
    with ui.row().classes("items-center gap-2"):
        previous = ui.button("Previous text", on_click=lambda: move(False)).props("flat dense")
        following = ui.button("Next text", on_click=lambda: move(True)).props("flat dense")
        ui.button(
            "Download original", icon="download", on_click=lambda: ui.download.file(path)
        ).props("flat dense")
        ui.button("Refresh text", icon="refresh", on_click=lambda: load()).props("flat dense")
        position = ui.label().classes("text-xs text-slate-500")
    section = BackgroundSection()

    def display(page: tuple[str, int, int]) -> None:
        nonlocal next_offset, total
        text, next_offset, total = page
        position.text = f"Bytes {offsets[-1]}–{next_offset} of {total}"
        previous.set_enabled(len(offsets) > 1)
        following.set_enabled(next_offset < total)
        ui.label(text or "Empty file").classes("w-full text-xs").style(
            "white-space: pre-wrap; overflow-wrap: anywhere; font-family: monospace"
        )

    def move(forward: bool) -> None:
        if forward and next_offset < total:
            offsets.append(next_offset)
        elif not forward and len(offsets) > 1:
            offsets.pop()
        load()

    def load() -> None:
        previous.disable()
        following.disable()
        offset = offsets[-1]
        section.load(lambda: read_text_page(path, offset), display)

    load()


def lazy_file(label: str, path: Path) -> None:
    """Read a file only when its detail expansion is opened."""
    with ui.expansion(label, icon="description").classes("w-full") as expansion:
        content = ui.column().classes("w-full")
    built = False

    def opened() -> None:
        nonlocal built
        if expansion.value and not built:
            built = True
            with content:
                render_file(path)

    expansion.on_value_change(opened)


class TablePager:
    """Send one page of rows to the browser instead of an entire record set."""

    def __init__(self, table: Any, rows: Callable[[], list[dict]], size: int = 25) -> None:
        self.table = table
        self.rows = rows
        self.size = size
        self.page = 0
        with ui.row().classes("items-center gap-2"):
            self.previous = ui.button("Previous rows", on_click=lambda: self.move(-1)).props(
                "flat dense"
            )
            self.next = ui.button("Next rows", on_click=lambda: self.move(1)).props("flat dense")
            self.position = ui.label().classes("text-xs text-slate-500")

    def move(self, delta: int) -> None:
        """Select an adjacent server-side page."""
        self.page += delta
        self.update(reset=False)

    def update(self, *, reset: bool = True) -> None:
        """Refresh the visible bounded rows after data or filters change."""
        rows = self.rows()
        self.page = 0 if reset else max(0, min(self.page, max(0, (len(rows) - 1) // self.size)))
        first = self.page * self.size
        self.table.rows = rows[first : first + self.size]
        self.table.selected = []
        self.previous.set_enabled(self.page > 0)
        self.next.set_enabled(first + self.size < len(rows))
        self.position.text = (
            f"{first + 1 if rows else 0}–{min(first + self.size, len(rows))} of {len(rows)}"
        )

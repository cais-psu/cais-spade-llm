"""Async log tailer and file watcher utilities."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import AsyncIterator


class LogTailer:
    """Async generator that tails a log file (like ``tail -f``).

    Usage::

        tailer = LogTailer("/path/to/file.log", initial_lines=100)
        async for line in tailer:
            print(line)
    """

    def __init__(self, path: str | Path, initial_lines: int = 200, poll_interval: float = 0.5) -> None:
        self.path = Path(path)
        self.initial_lines = initial_lines
        self.poll_interval = poll_interval
        self._offset: int = 0
        self._running = True

    def stop(self) -> None:
        self._running = False

    async def _read_initial(self) -> list[str]:
        """Read the last N lines of the file."""
        if not self.path.exists():
            return []
        try:
            text = await asyncio.to_thread(self.path.read_text, encoding="utf-8", errors="replace")
        except Exception:
            return []
        lines = text.splitlines()
        tail = lines[-self.initial_lines:]
        self._offset = len(text.encode("utf-8"))
        return tail

    async def __aiter__(self) -> AsyncIterator[str]:
        # Yield initial lines.
        for line in await self._read_initial():
            yield line

        # Poll for new content.
        while self._running:
            await asyncio.sleep(self.poll_interval)
            if not self.path.exists():
                continue
            try:
                size = os.path.getsize(self.path)
            except OSError:
                continue

            if size < self._offset:
                # File was truncated/rotated — re-read from start.
                self._offset = 0

            if size <= self._offset:
                continue

            try:
                with open(self.path, "rb") as f:
                    f.seek(self._offset)
                    chunk = f.read()
                self._offset += len(chunk)
                text = chunk.decode("utf-8", errors="replace")
                for line in text.splitlines():
                    yield line
            except Exception:
                continue

"""One non-blocking background-planning permit shared by the Gazebo scene."""

from __future__ import annotations

import fcntl
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _permit_path(launch_id: str) -> Path:
    identity = hashlib.sha256(str(launch_id).encode("utf-8")).hexdigest()[:20]
    return Path("/tmp") / f"cais_spade_llm_gazebo_planning_{identity}.lock"


@contextmanager
def scene_planning_permit(launch_id: str) -> Iterator[bool]:
    """Yield whether this process acquired the scene's planning permit."""
    path = _permit_path(launch_id)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            pass
        yield acquired
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

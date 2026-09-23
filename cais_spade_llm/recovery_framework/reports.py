"""Latest-only recovery reports with atomic writes and explicit archival."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


@contextmanager
def _locked(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.reports.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, allow_nan=False, separators=(",", ":")))
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class LatestReport:
    """Own one logical run independently of its replaceable report location."""

    def __init__(self, directory: Path, run_id: str | None = None) -> None:
        self.directory = directory
        self.path = directory / 'latest'
        self.run_id = run_id or uuid4().hex
        self.owner = uuid4().hex
        with _locked(directory):
            _atomic_json(directory / '.latest-owner.json', {
                'owner': self.owner, 'run_id': self.run_id,
            })

    def save(self, report: dict, processes: dict[str, dict] | None = None) -> bool:
        """Replace current evidence; a superseded runtime cannot write latest."""
        with _locked(self.directory):
            ownership = json.loads((self.directory / '.latest-owner.json').read_text())
            if ownership.get('owner') != self.owner:
                logger.warning('Ignored stale report writer for run %s', self.run_id)
                return False
            if processes is not None:
                destination = self.path / 'processes'
                destination.mkdir(parents=True, exist_ok=True)
                for name, process in processes.items():
                    if not name or Path(name).name != name:
                        raise ValueError('Process export name must be a filename')
                    path = destination / f'{name}.json'
                    if not path.exists() or json.loads(path.read_text()) != process:
                        _atomic_json(path, process)
                for path in destination.glob('*.json'):
                    if path.stem not in processes:
                        path.unlink()
            _atomic_json(self.path / 'run.json', {**report, 'run_id': self.run_id})
            return True


def archive_latest(directory: Path, *, expected_run_id: str) -> Path:
    """Archive exactly the displayed run at an operator's explicit request."""
    with _locked(directory):
        source = directory / 'latest'
        report = json.loads((source / 'run.json').read_text())
        if report.get('run_id') != expected_run_id:
            raise ValueError('The latest run changed; refresh before archiving')
        target = directory / 'archive' / f'{expected_run_id}_{uuid4().hex[:8]}'
        if Path(expected_run_id).name != expected_run_id:
            raise ValueError('Invalid report run ID')
        target.parent.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.archive-', dir=directory))
        try:
            shutil.copytree(source, temporary, dirs_exist_ok=True)
            if json.loads((temporary / 'run.json').read_text()) != report:
                raise OSError('Archive verification failed')
            temporary.replace(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return target / 'run.json'

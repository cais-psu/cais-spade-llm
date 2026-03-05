"""Filesystem persistence for verified plan+safety bundles."""

from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import INDEX_SCHEMA_VERSION, atomic_json_write


class BundleStore:
    """Manage bundle index, manifests, active selection, and generation locks."""

    def __init__(self, root_dir: Path | str = "cais_spade_llm/user_verified_plan") -> None:
        self.root_dir = Path(root_dir)
        self.bundles_dir = self.root_dir / "bundles"
        self.index_path = self.root_dir / "index.json"
        self.lock_path = self.root_dir / ".bundle_generation.lock"
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.bundles_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._save_index(
                {
                    "schema_version": INDEX_SCHEMA_VERSION,
                    "active_bundle_id": None,
                    "bundles": [],
                }
            )

    def _load_index(self) -> dict[str, Any]:
        self._ensure_layout()
        try:
            with self.index_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = {}

        bundles = payload.get("bundles")
        if not isinstance(bundles, list):
            bundles = []

        return {
            "schema_version": int(payload.get("schema_version", INDEX_SCHEMA_VERSION)),
            "active_bundle_id": payload.get("active_bundle_id"),
            "bundles": bundles,
        }

    def _save_index(self, data: dict[str, Any]) -> None:
        atomic_json_write(self.index_path, data)

    def list_bundles(self) -> list[dict[str, Any]]:
        data = self._load_index()
        bundles = list(data.get("bundles", []))
        bundles.sort(key=lambda b: str(b.get("created_at_utc", "")), reverse=True)
        return bundles

    def upsert_bundle_summary(self, summary: dict[str, Any]) -> None:
        bundle_id = str(summary.get("bundle_id", "")).strip()
        if not bundle_id:
            raise ValueError("bundle summary missing bundle_id")

        data = self._load_index()
        bundles = list(data.get("bundles", []))
        replaced = False
        for idx, row in enumerate(bundles):
            if str(row.get("bundle_id")) == bundle_id:
                bundles[idx] = summary
                replaced = True
                break
        if not replaced:
            bundles.append(summary)

        data["bundles"] = bundles
        self._save_index(data)

    def get_bundle_summary(self, bundle_id: str) -> dict[str, Any] | None:
        bid = str(bundle_id or "").strip()
        if not bid:
            return None
        for row in self.list_bundles():
            if str(row.get("bundle_id")) == bid:
                return row
        return None

    def update_bundle_summary(self, bundle_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        bid = str(bundle_id or "").strip()
        if not bid:
            raise ValueError("bundle_id is required")
        current = self.get_bundle_summary(bid)
        if current is None:
            raise ValueError(f"unknown bundle_id: {bid}")
        merged = dict(current)
        merged.update(dict(patch or {}))
        self.upsert_bundle_summary(merged)
        return merged

    def delete_bundle_summary(self, bundle_id: str) -> bool:
        """Remove a bundle from the index. Returns True if an entry was removed."""
        bid = str(bundle_id or "").strip()
        if not bid:
            raise ValueError("bundle_id is required")
        data = self._load_index()
        bundles = list(data.get("bundles", []))
        kept = [row for row in bundles if str(row.get("bundle_id", "")).strip() != bid]
        removed = len(kept) != len(bundles)
        data["bundles"] = kept
        if str(data.get("active_bundle_id", "")).strip() == bid:
            data["active_bundle_id"] = None
        self._save_index(data)
        return removed

    def get_active_bundle_id(self) -> str | None:
        bid = self._load_index().get("active_bundle_id")
        return str(bid) if bid else None

    def set_active_bundle_id(self, bundle_id: str | None) -> None:
        data = self._load_index()
        if bundle_id is None:
            data["active_bundle_id"] = None
            self._save_index(data)
            return

        bid = str(bundle_id).strip()
        if not bid:
            data["active_bundle_id"] = None
            self._save_index(data)
            return

        if self.get_bundle_summary(bid) is None:
            raise ValueError(f"unknown bundle_id: {bid}")

        data["active_bundle_id"] = bid
        self._save_index(data)

    def bundle_dir(self, bundle_id: str) -> Path:
        return self.bundles_dir / str(bundle_id)

    def manifest_path(self, bundle_id: str) -> Path:
        return self.bundle_dir(bundle_id) / "bundle_manifest.json"

    def load_manifest(self, bundle_id: str) -> dict[str, Any] | None:
        p = self.manifest_path(bundle_id)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def save_manifest(self, bundle_dir: Path | str, manifest: dict[str, Any]) -> Path:
        d = Path(bundle_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "bundle_manifest.json"
        atomic_json_write(p, manifest)
        return p

    def overwrite_manifest(self, bundle_id: str, manifest: dict[str, Any]) -> Path:
        bid = str(bundle_id or "").strip()
        if not bid:
            raise ValueError("bundle_id is required")
        return self.save_manifest(self.bundle_dir(bid), manifest)

    def create_temp_bundle_dir(self, bundle_id: str) -> Path:
        tmp = self.bundles_dir / f".tmp_{bundle_id}_{os.getpid()}_{int(time.time() * 1000)}"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp

    def finalize_bundle_dir(self, tmp_dir: Path | str, bundle_id: str) -> Path:
        src = Path(tmp_dir)
        dst = self.bundle_dir(bundle_id)
        if dst.exists():
            raise FileExistsError(f"bundle already exists: {dst}")
        os.replace(src, dst)
        return dst

    @contextmanager
    def generation_lock(
        self,
        *,
        timeout_sec: float = 0.0,
        poll_sec: float = 0.2,
    ) -> Iterator[None]:
        """
        Exclusive lock using O_EXCL lock file.
        If timeout is 0, fail immediately when locked.
        """
        start = time.monotonic()
        fd: int | None = None
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("utf-8"))
                break
            except FileExistsError:
                if timeout_sec <= 0:
                    raise RuntimeError("bundle generation already in progress")
                if (time.monotonic() - start) >= timeout_sec:
                    raise RuntimeError("timed out waiting for bundle generation lock")
                time.sleep(max(0.05, poll_sec))

        try:
            yield
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
            try:
                self.lock_path.unlink(missing_ok=True)
            except Exception:
                pass

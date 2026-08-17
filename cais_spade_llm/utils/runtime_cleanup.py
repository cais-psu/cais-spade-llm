"""Bound generated CAIS and ROS runtime artifacts safely."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

HARDWARE_RUN_LOGS_PER_COMPONENT = 10
ROS_LOG_RETENTION_DAYS = 7
ACTION_LOG_MAX_BYTES = 10 * 1024 * 1024
ACTION_LOG_BACKUP_COUNT = 5

_CATEGORY_ORDER = (
    "hardware_run_logs",
    "cais_ros_logs",
    "test_caches",
    "global_ros_logs",
)


def project_root() -> Path:
    """Return the checked-out CAIS-SPADE-LLM project root."""
    return Path(__file__).resolve().parents[2]


def cais_log_dir(root: Path | None = None) -> Path:
    """Return the CAIS-owned runtime log directory."""
    resolved_root = Path(root) if root is not None else project_root()
    return resolved_root / "cais_spade_llm" / "log"


def cais_ros_log_dir(root: Path | None = None) -> Path:
    """Return the ROS log directory owned by CAIS-launched processes."""
    return cais_log_dir(root) / "ros"


def install_action_log_handlers(logger: logging.Logger, log_path: Path) -> None:
    """Install one rotating file handler and one console handler on a logger."""
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    resolved_path = Path(log_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = RotatingFileHandler(
        resolved_path,
        mode="a",
        maxBytes=ACTION_LOG_MAX_BYTES,
        backupCount=ACTION_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def _empty_report(*, applied: bool) -> dict[str, Any]:
    return {
        "applied": bool(applied),
        "items": [],
        "entries": dict.fromkeys(_CATEGORY_ORDER, 0),
        "counts": dict.fromkeys(_CATEGORY_ORDER, 0),
        "bytes": dict.fromkeys(_CATEGORY_ORDER, 0),
        "errors": [],
    }


def _validated_root(path: Path, *, forbidden: Iterable[Path]) -> Path:
    expanded = Path(path).expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise ValueError(
                f"cleanup root must not contain a directory symlink: {candidate}"
            )
    resolved = expanded.resolve(strict=False)
    forbidden_roots = {Path(item).expanduser().resolve(strict=False) for item in forbidden}
    if resolved == Path(resolved.anchor) or resolved in forbidden_roots:
        raise ValueError(f"refusing broad cleanup root: {resolved}")
    return resolved


def _safe_descendant(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if not relative.parts:
        return False
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return False
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError:
        return False
    return True


def _path_size(path: Path) -> int:
    if path.is_symlink() or not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_symlink():
            continue
        try:
            if child.is_file():
                total += int(child.stat().st_size)
        except OSError:
            continue
    return total


def _path_file_count(path: Path) -> int:
    if path.is_symlink() or not path.exists():
        return 0
    if path.is_file():
        return 1
    count = 0
    for child in path.rglob("*"):
        if child.is_symlink():
            continue
        try:
            if child.is_file():
                count += 1
        except OSError:
            continue
    return count


def _latest_entry_mtime(path: Path) -> float:
    latest = float(path.stat().st_mtime)
    if not path.is_dir() or path.is_symlink():
        return latest
    for child in path.rglob("*"):
        if child.is_symlink():
            continue
        try:
            latest = max(latest, float(child.stat().st_mtime))
        except OSError:
            continue
    return latest


def _record_candidate(
    report: dict[str, Any],
    *,
    category: str,
    path: Path,
    allowed_root: Path,
    apply: bool,
) -> None:
    if not _safe_descendant(path, allowed_root):
        return
    size = _path_size(path)
    file_count = _path_file_count(path)
    item = {
        "category": category,
        "path": str(path),
        "files": file_count,
        "bytes": size,
    }
    if apply:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            report["errors"].append(f"{path}: {exc}")
            return
    report["items"].append(item)
    report["entries"][category] = int(report["entries"].get(category, 0)) + 1
    report["counts"][category] = (
        int(report["counts"].get(category, 0)) + file_count
    )
    report["bytes"][category] = int(report["bytes"].get(category, 0)) + size


def prune_hardware_run_logs(
    log_dir: Path,
    *,
    keep_per_component: int = HARDWARE_RUN_LOGS_PER_COMPONENT,
    apply: bool = False,
) -> dict[str, Any]:
    """Keep only the newest Hardware Stack run logs for each exact component."""
    requested_root = Path(log_dir).expanduser()
    root = _validated_root(
        requested_root,
        forbidden=(Path.home(), project_root()),
    )
    report = _empty_report(applied=apply)
    keep = max(0, int(keep_per_component))
    if not root.exists():
        return report
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in root.glob("*__run_*.log"):
        if not path.is_file() or path.is_symlink():
            continue
        component, separator, _run_token = path.name.partition("__run_")
        if separator and component:
            groups[component].append(path)
    for paths in groups.values():
        ranked_paths: list[tuple[int, str, Path]] = []
        for path in paths:
            try:
                ranked_paths.append((path.stat().st_mtime_ns, path.name, path))
            except OSError as exc:
                report["errors"].append(f"{path}: {exc}")
        ranked_paths.sort(reverse=True)
        for _modified_at, _name, path in ranked_paths[keep:]:
            _record_candidate(
                report,
                category="hardware_run_logs",
                path=path,
                allowed_root=root,
                apply=apply,
            )
    return report


def _collect_expired_ros_entries(
    report: dict[str, Any],
    *,
    root: Path,
    category: str,
    cutoff_timestamp: float,
    apply: bool,
) -> None:
    if not root.exists():
        return
    for path in root.iterdir():
        if path.name == "latest" or path.is_symlink():
            continue
        try:
            modified_at = _latest_entry_mtime(path)
        except OSError as exc:
            report["errors"].append(f"{path}: {exc}")
            continue
        if modified_at >= cutoff_timestamp:
            continue
        _record_candidate(
            report,
            category=category,
            path=path,
            allowed_root=root,
            apply=apply,
        )


def _collect_test_caches(
    report: dict[str, Any],
    *,
    root: Path,
    apply: bool,
) -> None:
    candidates = [root / ".pytest_cache", root / ".ruff_cache"]
    excluded_top_level = {
        ".git",
        ".claude",
        ".venv",
        ".venv-windows",
        "env",
        "venv",
    }
    if root.exists() and not root.is_symlink():
        candidates.extend(root.glob("__pycache__"))
        for source_root in root.iterdir():
            if (
                source_root.name in excluded_top_level
                or not source_root.is_dir()
                or source_root.is_symlink()
            ):
                continue
            candidates.extend(source_root.rglob("__pycache__"))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists() or candidate.is_symlink():
            continue
        seen.add(candidate)
        _record_candidate(
            report,
            category="test_caches",
            path=candidate,
            allowed_root=root,
            apply=apply,
        )


def _merge_report(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["items"].extend(source["items"])
    target["errors"].extend(source["errors"])
    for category in _CATEGORY_ORDER:
        target["entries"][category] += int(source["entries"].get(category, 0))
        target["counts"][category] += int(source["counts"].get(category, 0))
        target["bytes"][category] += int(source["bytes"].get(category, 0))


def cleanup_runtime_artifacts(
    *,
    root: Path | None = None,
    home: Path | None = None,
    apply: bool = False,
    include_cais: bool = True,
    include_caches: bool = True,
    include_global_ros: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """Inspect or prune bounded runtime artifacts under exact owned roots."""
    requested_root = Path(root).expanduser() if root is not None else project_root()
    requested_home = Path(home).expanduser() if home is not None else Path.home()
    report = _empty_report(applied=apply)
    cutoff = float(now if now is not None else time.time()) - (
        ROS_LOG_RETENTION_DAYS * 86400.0
    )
    if include_cais:
        log_root = _validated_root(
            cais_log_dir(requested_root),
            forbidden=(requested_root, requested_home),
        )
        _merge_report(
            report,
            prune_hardware_run_logs(log_root, apply=apply),
        )
        ros_root = _validated_root(
            cais_ros_log_dir(requested_root),
            forbidden=(requested_root, requested_home),
        )
        _collect_expired_ros_entries(
            report,
            root=ros_root,
            category="cais_ros_logs",
            cutoff_timestamp=cutoff,
            apply=apply,
        )
    if include_caches:
        safe_project_root = _validated_root(
            requested_root,
            forbidden=(requested_home,),
        )
        _collect_test_caches(report, root=safe_project_root, apply=apply)
    if include_global_ros:
        global_ros_root = _validated_root(
            requested_home / ".ros" / "log",
            forbidden=(requested_home, requested_root),
        )
        _collect_expired_ros_entries(
            report,
            root=global_ros_root,
            category="global_ros_logs",
            cutoff_timestamp=cutoff,
            apply=apply,
        )
    return report


def format_cleanup_report(report: dict[str, Any]) -> str:
    """Format a cleanup result for an operator terminal."""
    mode = "APPLIED" if bool(report.get("applied")) else "DRY RUN"
    lines = [f"Runtime cleanup: {mode}"]
    total_count = 0
    total_bytes = 0
    for category in _CATEGORY_ORDER:
        entries = int(report.get("entries", {}).get(category, 0))
        count = int(report.get("counts", {}).get(category, 0))
        size = int(report.get("bytes", {}).get(category, 0))
        total_count += count
        total_bytes += size
        lines.append(
            f"  {category}: {count} files in {entries} entries, "
            f"{size / (1024 * 1024):.1f} MiB"
        )
    lines.append(f"  total: {total_count} files, {total_bytes / (1024 * 1024):.1f} MiB")
    errors = [str(error) for error in report.get("errors", [])]
    if errors:
        lines.append("  errors:")
        lines.extend(f"    - {error}" for error in errors)
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or prune generated CAIS and ROS runtime artifacts.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete eligible artifacts; the default is a dry run.",
    )
    parser.add_argument(
        "--scope",
        choices=("all", "cais", "global-ros"),
        default="cais",
        help="Select CAIS-owned artifacts, global ROS logs, or both.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the runtime cleanup command-line interface."""
    args = _build_arg_parser().parse_args(argv)
    include_cais = args.scope in {"all", "cais"}
    include_global_ros = args.scope in {"all", "global-ros"}
    report = cleanup_runtime_artifacts(
        apply=bool(args.apply),
        include_cais=include_cais,
        include_caches=include_cais,
        include_global_ros=include_global_ros,
    )
    sys.stdout.write(format_cleanup_report(report) + "\n")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Isolated recovery generation and validation jobs using saved inputs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import signal
import sys
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "cais_spade_llm/monitor/debug/test_runs"
MODES = ("outline", "primitive", "safety", "full")
logger = logging.getLogger(__name__)


def read_object(path: Path) -> dict:
    """Read a JSON object without changing its identifiers."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def write_record(path: Path, value: dict) -> None:
    """Atomically publish a diagnostic record for polling clients."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def inspect_inputs(
    context_path: Path, scenario_path: Path | None = None, *, root: Path = ROOT
) -> dict:
    """Validate saved task/resource references and fingerprint their exact inputs."""
    runtime_context = read_object(context_path)
    scenario_id = runtime_context.get("failure_scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id or Path(scenario_id).name != scenario_id:
        raise ValueError("runtime_context failure_scenario_id is missing or invalid")
    expected = root / "cais_spade_llm/initialization/failure_scenarios" / f"{scenario_id}.json"
    if scenario_path is not None and scenario_path.resolve() != expected.resolve():
        raise ValueError("Selected scenario does not match runtime_context.failure_scenario_id")
    scenario = read_object(expected)
    if scenario.get("scenario_id") != scenario_id:
        raise ValueError("Scenario filename and scenario_id do not match")

    def reference(value: str, base: Path = root) -> Path:
        path = (base / value).resolve()
        path.relative_to(root.resolve())
        if not path.is_file():
            raise ValueError(f"Input file is missing: {path}")
        return path

    manifest_path = reference("bundle_manifest.json", root / runtime_context["bundle_root"])
    manifest = read_object(manifest_path)
    artifacts = manifest["artifacts"]
    files = {manifest_path, expected.resolve(), reference(runtime_context["product_geometry"])}
    bundle_files = {
        key: reference(artifacts[key], manifest_path.parent)
        for key in ("tools_json", "plan_json", "requirements_json", "safety_logic_json")
    }
    files.update(bundle_files.values())
    nodes = read_object(bundle_files["plan_json"]).get("nodes", [])
    tasks = {node["id"]: node for node in nodes if isinstance(node, dict) and node.get("id")}
    event = runtime_context.get("failure_event") or {}
    failed_task = tasks.get(event.get("failed_task_id"))
    if failed_task is None:
        raise ValueError("failure_event.failed_task_id is not in the selected plan")
    unknown = set(runtime_context.get("task_statuses") or {}) - set(tasks)
    if unknown:
        raise ValueError(f"task_statuses contains unknown task IDs: {sorted(unknown)}")
    trigger = scenario.get("trigger") or {}
    if (
        trigger.get("function_names")
        and failed_task.get("function_name") not in trigger["function_names"]
    ):
        raise ValueError("The failed task function does not match the selected scenario trigger")
    if not runtime_context.get("part_tracker") or not runtime_context.get("goal_state"):
        raise ValueError("runtime_context requires part_tracker and goal_state")
    resources = set()
    for snapshot in runtime_context.get("resource_snapshots") or []:
        config_path = reference(snapshot["resource_config"])
        files.add(config_path)
        config = read_object(config_path)[snapshot["resource_config_key"]]
        jid = snapshot.get("resource_jid")
        if not jid or jid in resources or config.get("jid") != jid:
            raise ValueError("Resource snapshot identity does not match its configuration")
        if snapshot.get("execution_env") not in config:
            raise ValueError(f"Resource {jid} has no matching execution_env configuration")
        resources.add(jid)
    if failed_task.get("resource_jid") not in resources:
        raise ValueError("The failed task has no matching resource snapshot")
    settings = root / "cais_spade_llm/initialization/recovery_outline_experiment_settings.json"
    if settings.is_file():
        files.add(settings)
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }
    hashes["runtime_context"] = hashlib.sha256(context_path.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {
        "runtime_context": runtime_context,
        "scenario": scenario_id,
        "files": hashes,
        "fingerprint": fingerprint,
        "context_source": str(context_path.resolve()),
        "scenario_source": str(expected.resolve()),
    }


def validator_fingerprint(root: Path = ROOT) -> str:
    """Bind reusable acceptance to the exact validator and runner source files."""
    paths = {root / "test/test_case3_recovery_dryrun.py"}
    for directory in (
        "agents/intelligent_product/replanner/llm_recovery",
        "agents/central_controller",
        "resources",
    ):
        paths.update((root / "cais_spade_llm" / directory).rglob("*.py"))
    paths.update(
        root / "cais_spade_llm/agents/resource_agent" / name
        for name in ("resource_agent.py", "robot_agent.py")
    )
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def load_outline_checkpoint(path: Path, inputs: dict, *, root: Path = ROOT) -> dict:
    """Accept only a complete outline checkpoint with unchanged inputs and validators."""
    checkpoint = read_object(path)
    if checkpoint.get("input_fingerprint") != inputs["fingerprint"]:
        raise ValueError("Outline inputs differ; generate a new outline for this context")
    if checkpoint.get("validator_fingerprint") != validator_fingerprint(root):
        raise ValueError("Outline validator code has changed; generate a new outline")
    session = checkpoint.get("multi_turn_session") or {}
    if session.get("current_phase") not in {"primitive_generation", "finalize"} or session.get(
        "status"
    ) not in {"ready_for_primitive_generation", "completed"}:
        raise ValueError("A complete accepted outline checkpoint is required")
    if not (session.get("accepted_outline_prefix") or session.get("transition_trace")):
        raise ValueError("Outline checkpoint has no accepted trace")
    return checkpoint


def snapshot_inputs(directory: Path, inputs: dict, *, root: Path = ROOT) -> None:
    """Save original input bytes separately from existing examples and settings."""
    destination = directory / "inputs"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(inputs["context_source"], destination / "runtime_context.json")
    for relative in inputs["files"]:
        if relative == "runtime_context":
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    write_record(directory / "inputs.json", inputs)


class DiagnosticJobs:
    """Own one non-executing diagnostic subprocess across page navigation."""

    async def recover_interrupted(self) -> None:
        """Mark saved jobs whose owning UI process no longer exists as interrupted."""

        def recover() -> None:
            for path in self.runs.glob("*/run.json"):
                try:
                    record = read_object(path)
                    if record.get("status") != "running":
                        continue
                    owner = record.get("owner_pid")
                    if isinstance(owner, int) and Path(f"/proc/{owner}").exists():
                        continue
                    record.update(
                        status="interrupted", finished_at=datetime.now(timezone.utc).isoformat()
                    )
                    write_record(path, record)
                except (OSError, ValueError) as exc:
                    logger.warning("Unable to read diagnostic job %s: %s", path, exc)

        await asyncio.to_thread(recover)

    def __init__(self, *, root: Path = ROOT, runs: Path = RUNS) -> None:
        self.root = root
        self.runs = runs
        self.lock = asyncio.Lock()
        self.process: asyncio.subprocess.Process | None = None
        self.task: asyncio.Task | None = None
        self.current: Path | None = None
        self.cancelled = False

    async def start(
        self,
        mode: str,
        context_path: Path,
        scenario_path: Path,
        outline_checkpoint: Path | None = None,
    ) -> Path:
        """Validate explicit inputs, snapshot them, then launch an isolated test."""
        async with self.lock:
            if self.task and not self.task.done():
                raise ValueError("A recovery test is already running")
            if mode not in MODES:
                raise ValueError("Unsupported recovery test mode")
            inputs = await asyncio.to_thread(
                inspect_inputs, context_path, scenario_path, root=self.root
            )
            if outline_checkpoint:
                if mode == "outline":
                    raise ValueError("Select primitive, safety, or full to reuse an outline")
                await asyncio.to_thread(
                    load_outline_checkpoint, outline_checkpoint, inputs, root=self.root
                )
            identifier = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
            )
            directory = self.runs / identifier
            directory.mkdir(parents=True)
            await asyncio.to_thread(snapshot_inputs, directory, inputs, root=self.root)
            record = {
                "id": identifier,
                "scenario": inputs["scenario"],
                "mode": mode,
                "owner_pid": os.getpid(),
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "context_source": inputs["context_source"],
                "scenario_source": inputs["scenario_source"],
                "outline_source": str(outline_checkpoint) if outline_checkpoint else None,
            }
            write_record(directory / "run.json", record)
            command = [
                sys.executable,
                str(self.root / "test/test_case3_recovery_dryrun.py"),
                "--mode",
                mode,
                "--runtime-context",
                str(directory / "inputs/runtime_context.json"),
                "--debug-root",
                str(directory),
            ]
            if outline_checkpoint:
                shutil.copyfile(outline_checkpoint, directory / "source_outline_checkpoint.json")
                command.extend(
                    ["--outline-checkpoint", str(directory / "source_outline_checkpoint.json")]
                )
            self.current, self.cancelled = directory, False
            try:
                with (directory / "runner.log").open("wb") as output:
                    self.process = await asyncio.create_subprocess_exec(
                        *command,
                        cwd=self.root,
                        stdout=output,
                        stderr=asyncio.subprocess.STDOUT,
                        env={**os.environ, "PYTHONUNBUFFERED": "1"},
                        start_new_session=True,
                    )
            except OSError as exc:
                record.update(status="failed", error=str(exc))
                write_record(directory / "run.json", record)
                raise
            self.task = asyncio.create_task(self._wait(self.process, directory, record))
            return directory

    async def _wait(
        self, process: asyncio.subprocess.Process, directory: Path, record: dict
    ) -> None:
        code = await process.wait()
        record.update(
            status="cancelled" if self.cancelled else ("completed" if code == 0 else "failed"),
            returncode=code,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        write_record(directory / "run.json", record)

    async def cancel(self) -> None:
        """Cancel the active test while retaining its partial evidence."""
        async with self.lock:
            if self.process is None or self.process.returncode is not None:
                return
            self.cancelled = True
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
            if self.task:
                await self.task


jobs = DiagnosticJobs()

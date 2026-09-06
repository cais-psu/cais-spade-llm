from __future__ import annotations

"""NiceGUI page for the ICRA 2027 Spec2Primitives case study."""


import asyncio
import base64
import hashlib
import io
import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from html import escape
from pathlib import Path
from typing import Any

from nicegui import ui
from nicegui.elements.badge import Badge
from nicegui.elements.button import Button
from nicegui.elements.label import Label
from PIL import Image

from cais_spade_llm.spec2primitives.adapters.dual_gazebo import (
    DualGazeboRuntime,
    DualGazeboStatus,
    read_dual_gazebo_status,
    start_dual_gazebo,
    stop_dual_gazebo,
)
from cais_spade_llm.spec2primitives.adapters.ui_runtime import (
    Spec2PrimitivesUIRuntime,
)
from cais_spade_llm.spec2primitives.agents.pa import (
    ProductAgentContextRuntime,
    cancel_pa_context_interaction,
    load_pa_context_grounding_completion,
    start_pa_context_interaction,
    submit_pa_clarification_reply,
)
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    _GROUNDING_VALIDATION_CODES,
    _grounding_progress_validation_error,
)
from cais_spade_llm.spec2primitives.agents.ra import (
    PrimitiveDraftError,
    RAContextHandoffError,
    activate_selected_ra_context,
    author_primitive_program_draft,
    read_phase_5_1_diagnostic,
    read_phase_5_2_diagnostic,
)

_TURTLE_PREFIX_PATTERN = re.compile(r"^@prefix\s+([A-Za-z][A-Za-z0-9_-]*):\s+<([^>]+)>\s+\.\s*$")


_GROUNDING_PROGRESS_INTERVAL_SECONDS = 1.0
logger = logging.getLogger(__name__)


class _PAUIRuntimeObserver:
    """Report semantic ProductAgent stages around structured calls."""

    def __init__(
        self,
        product_agent: ProductAgentContextRuntime,
        *,
        on_pa_stage: Callable[[str], None],
        on_pa_event: Callable[[dict[str, str]], None] | None = None,
    ) -> None:
        self._product_agent = product_agent
        self._on_pa_stage = on_pa_stage
        self._on_pa_event = on_pa_event
        self._grounding_attempt = 0
        self.activity = "Investigating approved evidence and authoring the target feature."

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]]
        | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        """Report the response contract's semantic stage and validated result."""
        if response_format.get("name") == "spec2primitives_grounding_result":
            self._grounding_attempt += 1
        if (
            response_format.get("name") == "spec2primitives_grounding_result"
            and self._grounding_attempt > 1
        ):
            self.activity = f"Revising the target feature · attempt {self._grounding_attempt}."
            if self._on_pa_event is not None:
                self._on_pa_event(
                    _timeline_event(
                        "running",
                        f"ProductAgent revision · attempt {self._grounding_attempt}",
                        "ProductAgent is revising its proposal using validation feedback.",
                    )
                )
        else:
            self.activity = _live_pa_stage(response_format)
        self._on_pa_stage(self.activity)
        response = await self._product_agent.ask_llm_structured(
            prompt,
            response_format=response_format,
            tools=tools,
            tool_executor=tool_executor,
            max_tool_rounds=max_tool_rounds,
        )
        if self._on_pa_event is not None:
            event = _live_pa_response_event(response, response_format)
            if event is not None:
                self._on_pa_event(event)
        return response


def _grounding_progress_record(
    path: Path, cache: dict[Path, tuple[int, dict[str, object]]]
) -> dict[str, object]:
    """Read a changed outcome once and tolerate records still being written."""
    try:
        modified = path.stat().st_mtime_ns
        previous = cache.get(path)
        if previous is not None and previous[0] == modified:
            return previous[1]
        record = _read_json_object(path)
    except (OSError, ValueError):
        return {}
    cache[path] = (modified, record)
    return record


def _grounding_progress_text(progress: Mapping[str, object]) -> str:
    """Present host-owned limits and validation feedback without interpreting evidence."""
    if _grounding_progress_validation_error(progress) is not None:
        return ""
    detail = (
        f"Evidence operations: {progress['evidence_operations_used']}/{progress['evidence_operations_limit']}"
        f" · Proposal attempt: {progress['proposals_used']}/{progress['proposals_limit']}."
    )
    feedback = progress.get("last_feedback")
    if isinstance(feedback, list) and feedback:
        for item in feedback:
            message = item.get("message")
            if isinstance(message, str) and message:
                detail += " Validation feedback: " + message
            states = item.get("missing_states")
            if isinstance(states, list) and all(isinstance(state, str) for state in states):
                detail += " Affected states: " + ", ".join(states) + "."
    reason = progress.get("stop_reason")
    if isinstance(reason, str) and reason:
        detail += f" Stop reason: {reason}."
    return detail


def _persisted_grounding_progress(records: Mapping[str, object]) -> Mapping[str, object]:
    """Select the latest host counters without constructing a new persisted record."""
    turns = [record for name, record in records.items() if name.startswith("turn_")]
    if turns and isinstance(turns[-1], Mapping):
        output = turns[-1].get("PA_output")
        progress = output.get("grounding_progress") if isinstance(output, Mapping) else None
        if isinstance(progress, Mapping) and _grounding_progress_text(progress):
            return progress
    snapshots = []
    for record in records.values():
        if not isinstance(record, Mapping):
            continue
        output = record.get("PA_output")
        owner = output if isinstance(output, Mapping) else record
        progress = owner.get("grounding_progress")
        if isinstance(progress, Mapping) and _grounding_progress_text(progress):
            snapshots.append(progress)
    return max(
        snapshots,
        key=lambda item: (
            item["proposals_used"],
            item["evidence_operations_used"],
            bool(item.get("stop_reason")),
        ),
        default={},
    )


async def _monitor_grounding_progress(
    interaction_root: Path,
    on_pa_stage: Callable[[str], None],
    is_attached: Callable[[], bool] | None,
    observer: _PAUIRuntimeObserver | None,
    initial_paths: set[Path],
) -> None:
    """Read changed host progress snapshots while the original PA call continues."""
    grounding = interaction_root / "products/grounding"
    cache: dict[Path, tuple[int, dict[str, object]]] = {}
    started = time.monotonic()
    try:
        while is_attached is None or is_attached():
            paths = list((grounding / "ontology_grounding").glob("request_*.json"))
            paths.extend((interaction_root / "interaction_record").glob("tool_call_*.json"))
            for path in sorted(
                set(paths) - initial_paths, key=lambda item: item.stat().st_mtime_ns, reverse=True
            ):
                progress = _grounding_progress_record(path, cache).get("grounding_progress")
                if isinstance(progress, Mapping) and (detail := _grounding_progress_text(progress)):
                    activity = (
                        observer.activity if observer is not None else "Grounding product context."
                    )
                    on_pa_stage(f"{activity} {detail} Elapsed: {int(time.monotonic() - started)}s.")
                    break
            await asyncio.sleep(_GROUNDING_PROGRESS_INTERVAL_SECONDS)
    except (OSError, RuntimeError, TypeError, ValueError):
        # Progress is observational; losing its page must never fail the producer.
        logger.debug("Grounding progress display became unavailable.", exc_info=True)


async def _await_with_grounding_progress(
    operation: Awaitable[dict[str, object]],
    interaction_root: Path,
    on_pa_stage: Callable[[str], None] | None,
    is_attached: Callable[[], bool] | None,
    observer: _PAUIRuntimeObserver | None = None,
) -> dict[str, object]:
    """Keep monitor teardown separate from producer errors and cancellation."""
    if on_pa_stage is None:
        return await operation
    initial_paths = set(
        (interaction_root / "products/grounding/ontology_grounding").glob("request_*.json")
    )
    initial_paths.update((interaction_root / "interaction_record").glob("tool_call_*.json"))
    monitor = asyncio.create_task(
        _monitor_grounding_progress(
            interaction_root, on_pa_stage, is_attached, observer, initial_paths
        )
    )
    try:
        return await operation
    finally:
        monitor.cancel()
        # Treat monitor cancellation as a result; caller cancellation must still propagate.
        result = (await asyncio.gather(monitor, return_exceptions=True))[0]
        if isinstance(result, Exception):
            logger.debug("Grounding progress monitor stopped unexpectedly.", exc_info=result)


def _status_color(state: str) -> str:
    if state == "running":
        return "green"
    if state == "stopped":
        return "grey"
    return "amber"


def _set_enabled(element: Button, enabled: bool) -> None:
    if enabled:
        element.props(remove="disable")
    else:
        element.props("disable")


def _apply_dual_gazebo_status(
    status: DualGazeboStatus,
    *,
    busy: bool,
    status_badge: Badge,
    status_message: Label,
    start_button: Button,
    stop_button: Button,
    refresh_button: Button,
) -> None:
    """Apply a fresh runtime snapshot to the launcher controls."""
    status_badge.set_text(status.state)
    status_badge.props(f"color={_status_color(status.state)}")
    if status.blocked_reason:
        status_message.set_text(status.blocked_reason)
        status_message.classes(replace="text-sm text-amber-700")
    elif status.state == "running":
        status_message.set_text("Dual Gazebo is running under existing UI ownership.")
        status_message.classes(replace="text-sm text-green-700")
    elif status.state == "stopped":
        status_message.set_text("Dual Gazebo is stopped and ready to launch.")
        status_message.classes(replace="text-sm text-slate-600")
    else:
        status_message.set_text(f"Dual Gazebo status: {status.state}")
        status_message.classes(replace="text-sm text-amber-700")

    _set_enabled(
        start_button,
        not busy and status.state == "stopped" and not status.blocked_reason,
    )
    _set_enabled(stop_button, not busy and status.state == "running")
    _set_enabled(refresh_button, not busy)


def _render_dual_gazebo(runtime: DualGazeboRuntime) -> None:
    """Render the isolated `gazebo_dual_spec2primitives` launcher and controls."""
    with ui.card().classes("w-full border border-slate-200 shadow-sm"):
        with ui.row().classes("w-full items-center justify-between gap-4 flex-wrap"):
            with ui.column().classes("flex-1 min-w-72 gap-0"):
                ui.label("Dual Gazebo Environment").classes("text-lg font-semibold text-slate-900")
                ui.label("xArm6 + UR5e · NIST CAD · Gazebo + MoveIt/RViz · No hardware").classes(
                    "text-xs text-slate-500"
                )

            with ui.row().classes("flex-1 min-w-72 items-center gap-3 flex-wrap"):
                status_badge = ui.badge("checking").props("color=grey outline")
                status_message = ui.label("Reading fresh runtime status...").classes(
                    "flex-1 min-w-48 text-sm text-slate-600"
                )

            with ui.row().classes("items-center gap-2 flex-wrap"):
                start_button = ui.button("Start", icon="play_arrow").props("disable")
                stop_button = ui.button("Stop", icon="stop").props("outline disable")
                refresh_button = ui.button("Refresh", icon="refresh").props("flat")

        action_state = {
            "busy": False,
            "refreshing": False,
            "status": DualGazeboStatus(state="checking"),
        }

        def _apply_status(status: DualGazeboStatus) -> None:
            action_state["status"] = status
            _apply_dual_gazebo_status(
                status,
                busy=bool(action_state["busy"]),
                status_badge=status_badge,
                status_message=status_message,
                start_button=start_button,
                stop_button=stop_button,
                refresh_button=refresh_button,
            )

        async def _refresh_status() -> None:
            if action_state["refreshing"]:
                return
            action_state["refreshing"] = True
            try:
                status = await asyncio.to_thread(read_dual_gazebo_status, runtime)
            except (OSError, RuntimeError, ValueError) as exc:
                status_badge.set_text("unavailable")
                status_badge.props("color=red")
                status_message.set_text(f"Unable to read dual Gazebo status: {exc}")
                status_message.classes(replace="text-sm text-red-700")
                _set_enabled(start_button, False)
                _set_enabled(stop_button, False)
            else:
                _apply_status(status)
            finally:
                action_state["refreshing"] = False

        async def _start() -> None:
            if action_state["busy"]:
                return
            action_state["busy"] = True
            _apply_status(action_state["status"])
            try:
                error = await asyncio.to_thread(start_dual_gazebo, runtime)
                if error:
                    ui.notify(error, type="warning", timeout=5000)
                else:
                    ui.notify("Started Dual Robots (xArm6 + UR5e)", type="positive")
            except (OSError, RuntimeError, ValueError) as exc:
                ui.notify(f"Dual Gazebo start failed: {exc}", type="negative", timeout=5000)
            finally:
                action_state["busy"] = False
                await _refresh_status()

        async def _stop() -> None:
            if action_state["busy"]:
                return
            action_state["busy"] = True
            _apply_status(action_state["status"])
            try:
                await asyncio.to_thread(stop_dual_gazebo, runtime)
                ui.notify("Stopped Dual Robots (xArm6 + UR5e)", type="info")
            except (OSError, RuntimeError, ValueError) as exc:
                ui.notify(f"Dual Gazebo stop failed: {exc}", type="negative", timeout=5000)
            finally:
                action_state["busy"] = False
                await _refresh_status()

        start_button.on_click(_start)
        stop_button.on_click(_stop)
        refresh_button.on_click(_refresh_status)
        ui.timer(0.1, _refresh_status, once=True)
        ui.timer(3.0, _refresh_status)


async def _run_pa_ui_interaction(
    runtime: Spec2PrimitivesUIRuntime,
    product_requirement: str,
    max_pa_turns: int = 12,
    on_pa_stage: Callable[[str], None] | None = None,
    on_pa_event: Callable[[dict[str, str]], None] | None = None,
    is_attached: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Run one connected native ProductAgent grounding workflow."""
    del max_pa_turns
    interaction_identifier = f"interaction_{uuid.uuid4().hex}"
    interaction_root = runtime.contexts_root / interaction_identifier
    product_agent = runtime.product_agent
    if on_pa_stage is not None:
        product_agent = _PAUIRuntimeObserver(
            product_agent,
            on_pa_stage=on_pa_stage,
            on_pa_event=on_pa_event,
        )
    phase_3_1 = await _await_with_grounding_progress(
        start_pa_context_interaction(
            product_agent,
            interaction_root,
            product_requirement,
            ontology_config=runtime.ontology_config,
            grounding_runtime=runtime.grounding_runtime,
        ),
        interaction_root,
        on_pa_stage,
        is_attached,
        product_agent if isinstance(product_agent, _PAUIRuntimeObserver) else None,
    )
    return {
        "interaction_identifier": interaction_identifier,
        "interaction_root": interaction_root,
        "product_requirement": product_requirement,
        "phase_3_1": phase_3_1,
        "phase_3_2": None,
        "phase_3_3": None,
        "max_pa_turns": 12,
    }


async def _submit_pa_ui_clarification(
    runtime: Spec2PrimitivesUIRuntime,
    interaction: dict[str, object],
    user_reply: str,
    *,
    on_pa_stage: Callable[[str], None] | None = None,
    on_pa_event: Callable[[dict[str, str]], None] | None = None,
    is_attached: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Submit one clarification reply and update the same UI interaction."""
    interaction_root = interaction.get("interaction_root")
    max_pa_turns = interaction.get("max_pa_turns")
    if not isinstance(interaction_root, Path) or not isinstance(max_pa_turns, int):
        raise ValueError("PA UI clarification interaction is malformed.")
    product_agent = runtime.product_agent
    if on_pa_stage is not None:
        product_agent = _PAUIRuntimeObserver(
            product_agent,
            on_pa_stage=on_pa_stage,
            on_pa_event=on_pa_event,
        )
    result = await _await_with_grounding_progress(
        submit_pa_clarification_reply(
            product_agent,
            interaction_root,
            user_reply,
            ontology_config=runtime.ontology_config,
            grounding_runtime=runtime.grounding_runtime,
        ),
        interaction_root,
        on_pa_stage,
        is_attached,
        product_agent if isinstance(product_agent, _PAUIRuntimeObserver) else None,
    )
    interaction["phase_3_4"] = result
    interaction["phase_3_3"] = result
    return interaction


def _cancel_pa_ui_interaction(
    interaction: dict[str, object],
) -> dict[str, object]:
    """Cancel one pending clarification without invoking ProductAgent."""
    interaction_root = interaction.get("interaction_root")
    if not isinstance(interaction_root, Path):
        raise ValueError("PA UI cancellation interaction is malformed.")
    result = cancel_pa_context_interaction(interaction_root)
    interaction["phase_3_4"] = result
    interaction["phase_3_3"] = result
    return interaction


def _timeline_event(state: str, title: str, detail: str) -> dict[str, str]:
    """Return one compact operator-facing ProductAgent event."""
    return {"state": state, "title": title, "detail": detail}


def _live_pa_stage(response_format: Mapping[str, object]) -> str:
    """Describe the current structured ProductAgent task in one sentence."""
    if response_format.get("name") == "spec2primitives_grounding_result":
        return "Investigating approved evidence and authoring the target feature."
    return "Grounding the validated product context."


def _live_pa_response_event(
    response: Mapping[str, object],
    response_format: Mapping[str, object],
) -> dict[str, str] | None:
    """Translate one structured ProductAgent result without exposing raw payloads."""
    del response_format
    wrapped = response.get("result")
    payload = wrapped if isinstance(wrapped, Mapping) else response
    if isinstance(payload.get("clarification_question"), str):
        return _timeline_event(
            "waiting",
            "Clarification requested",
            str(payload["clarification_question"]),
        )
    if isinstance(payload.get("target_feature"), Mapping):
        return _timeline_event(
            "running",
            "Target feature returned",
            "The evidence-backed target feature is being validated.",
        )
    return None


def _grounding_incomplete_detail(output: Mapping[str, object]) -> str:
    """Render only controller-coded grounding diagnostics."""
    code = output.get("grounding_validation_code")
    message = output.get("insufficient_evidence")
    if code not in _GROUNDING_VALIDATION_CODES:
        return "This record has no valid grounding result. Start a fresh interaction."
    if isinstance(code, str) and code and isinstance(message, str) and message:
        return f"{message} Validation code: {code}."
    return "This record has no valid grounding result. Start a fresh interaction."


def _interaction_ref_path(interaction_root: Path, ref: object) -> Path:
    """Resolve one persisted interaction reference without escaping its root."""
    if not isinstance(ref, str) or not ref:
        raise ValueError("Interaction record reference is invalid.")
    root = interaction_root.resolve()
    path = (root / ref).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Interaction record reference escapes its root.") from exc
    if not path.is_file():
        raise ValueError(f"Interaction record does not exist: {ref}")
    return path


def _read_json_object(path: Path) -> dict[str, object]:
    """Read one JSON object used by the validated UI result."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Interaction record is not an object: {path.name}")
    return value


def _latest_interaction_json_object(
    interaction_root: Path,
    relative_pattern: str,
) -> dict[str, object]:
    """Read the latest matching JSON object through the interaction root."""
    root = interaction_root.resolve()
    paths = sorted(root.glob(relative_pattern))
    if not paths:
        raise ValueError(f"Final interaction record is unavailable: {relative_pattern}")
    return _read_json_object(_interaction_ref_path(root, str(paths[-1].relative_to(root))))


def _latest_record(
    records: Mapping[str, object],
    prefix: str,
) -> dict[str, object] | None:
    """Return the latest ordered persisted object with the exact prefix."""
    values = [
        value
        for name, value in records.items()
        if name.startswith(prefix) and isinstance(value, dict)
    ]
    return values[-1] if values else None


def _turtle_prefixes(raw_turtle: str) -> tuple[tuple[str, str], ...]:
    """Return prefixes exactly as declared by the authoritative Turtle file."""
    prefixes = []
    for line in raw_turtle.splitlines():
        match = _TURTLE_PREFIX_PATTERN.match(line)
        if match is not None:
            prefixes.append((match.group(1), match.group(2)))
    return tuple(sorted(prefixes, key=lambda item: (-len(item[1]), item[0])))


def _compact_iri(iri: object, prefixes: tuple[tuple[str, str], ...]) -> str:
    """Display one exact IRI using a prefix declared by the persisted ABox."""
    if not isinstance(iri, str):
        raise ValueError("Ontology IRI is invalid.")
    for prefix, namespace in prefixes:
        if iri.startswith(namespace):
            return f"{prefix}:{iri.removeprefix(namespace)}"
    return iri


def _ontology_object_text(
    value: object,
    prefixes: tuple[tuple[str, str], ...],
) -> tuple[str, str]:
    """Return the exact readable object term and its Mermaid node kind."""
    if not isinstance(value, Mapping):
        raise ValueError("Ontology assertion object is invalid.")
    kind = value.get("kind")
    object_value = value.get("value")
    if kind == "iri":
        return _compact_iri(object_value, prefixes), "iri"
    if kind != "literal" or not isinstance(object_value, (str, int, float, bool)):
        raise ValueError("Ontology assertion object kind is invalid.")
    text = json.dumps(object_value, ensure_ascii=False, allow_nan=False)
    datatype = value.get("datatype")
    language = value.get("language")
    if isinstance(datatype, str):
        text = f"{text}^^{_compact_iri(datatype, prefixes)}"
    elif isinstance(language, str):
        text = f"{text}@{language}"
    return text, "literal"


def _mermaid_text(value: str) -> str:
    """Escape one exact ontology term for a Mermaid label."""
    return escape(value.replace("\n", " "), quote=True).replace("|", "&#124;")


def _final_ontology_view(
    product_context: Mapping[str, object],
    raw_turtle: str,
) -> dict[str, object]:
    """Build deterministic graph and table views of the final validated ABox."""
    assertions = product_context.get("assertions")
    if not isinstance(assertions, list):
        raise ValueError("Final ProductContextView assertions are invalid.")
    prefixes = _turtle_prefixes(raw_turtle)
    rows: list[dict[str, str]] = []
    node_kinds: dict[str, str] = {}
    for assertion in assertions:
        if not isinstance(assertion, Mapping):
            raise ValueError("Final ProductContextView assertion is invalid.")
        subject = _compact_iri(assertion.get("subject"), prefixes)
        predicate = _compact_iri(assertion.get("predicate"), prefixes)
        object_text, object_kind = _ontology_object_text(
            assertion.get("object"),
            prefixes,
        )
        rows.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": object_text,
            }
        )
        node_kinds.setdefault(subject, "iri")
        node_kinds.setdefault(object_text, object_kind)
    rows.sort(key=lambda row: (row["subject"], row["predicate"], row["object"]))

    node_ids = {term: f"n{index}" for index, term in enumerate(sorted(node_kinds))}
    lines = [
        "flowchart LR",
        "    classDef iri fill:#eef2ff,stroke:#4f46e5,color:#1e1b4b",
        "    classDef literal fill:#f8fafc,stroke:#64748b,color:#0f172a",
    ]
    for term, node_id in node_ids.items():
        lines.append(f'    {node_id}["{_mermaid_text(term)}"]:::{node_kinds[term]}')
    for row in rows:
        lines.append(
            f'    {node_ids[row["subject"]]} -->|"'
            f'{_mermaid_text(row["predicate"])}"| {node_ids[row["object"]]}'
        )
    return {
        "mermaid": "\n".join(lines),
        "rows": rows,
        "raw_turtle": raw_turtle,
    }


def _final_result_evidence(  # noqa: C901
    *,
    product_context: Mapping[str, object],
) -> list[dict[str, str]]:
    """Build compact evidence badges from validated final records."""
    evidence: list[dict[str, str]] = []

    def _add(label: object, status: object, kind: str) -> None:
        if not isinstance(label, str) or not label:
            return
        item = {"label": label, "status": str(status), "kind": kind}
        if item not in evidence:
            evidence.append(item)

    typed_bindings = product_context.get("typed_bindings")
    if isinstance(typed_bindings, list):
        for binding in typed_bindings:
            if isinstance(binding, Mapping):
                _add(
                    binding.get("output_symbol"),
                    binding.get("status", "recorded"),
                    "typed record",
                )
    return evidence


def _target_feature_evidence_refs(
    target_feature: Mapping[str, object],
) -> set[str]:
    """Collect direct citations from the PA-authored feature structure."""
    refs: set[str] = set()

    def _visit(value: object) -> None:
        if isinstance(value, Mapping):
            evidence_refs = value.get("evidence_refs")
            if isinstance(evidence_refs, list):
                refs.update(item for item in evidence_refs if isinstance(item, str) and item)
            for child in value.values():
                _visit(child)
        elif isinstance(value, list):
            for child in value:
                _visit(child)

    _visit(target_feature)
    return refs


def _cad_identity(
    interaction_root: Path,
    *,
    typed_bindings: list[object],
    target_feature: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Resolve the target-cited CAD identity without inferring a target pose."""
    cited = (
        _target_feature_evidence_refs(target_feature)
        if isinstance(target_feature, Mapping)
        else set()
    )
    candidates: list[tuple[bool, dict[str, object]]] = []
    for binding in typed_bindings:
        if (
            not isinstance(binding, Mapping)
            or binding.get("output_symbol") != "CADMeshRecord"
            or binding.get("status") != "accepted"
        ):
            continue
        record_ref = binding.get("record_ref")
        if not isinstance(record_ref, str):
            continue
        try:
            record = _read_json_object(_interaction_ref_path(interaction_root, record_ref))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
        source = record.get("source")
        context_ref = source.get("context_ref") if isinstance(source, Mapping) else None
        binding_refs = binding.get("evidence_refs")
        direct_refs = {record_ref}
        if isinstance(binding_refs, list):
            direct_refs.update(item for item in binding_refs if isinstance(item, str) and item)
        if isinstance(context_ref, str):
            direct_refs.add(context_ref)
        candidates.append(
            (
                bool(cited.intersection(direct_refs)),
                {
                    "context_ref": context_ref,
                    "record_ref": record_ref,
                    "coordinate_frame": record.get("coordinate_frame"),
                    "bounds_m": record.get("bounds_m"),
                },
            )
        )
    cited_candidates = [value for is_cited, value in candidates if is_cited]
    if len(cited_candidates) == 1:
        return {"status": "accepted", **cited_candidates[0]}
    if len(cited_candidates) > 1:
        return {"status": "ambiguous", "candidates": cited_candidates}
    if len(candidates) > 1:
        return {
            "status": "ambiguous",
            "candidates": [candidate for _is_cited, candidate in candidates],
        }
    return None


def _annotated_candidate_rgb(
    interaction_root: Path,
    *,
    segmentation_ref: str,
    candidate_reference: Mapping[str, object],
    observation_review_bindings: Mapping[str, str],
    state_label: str,
) -> dict[str, object] | None:
    """Build a crop-first visual with an optional annotated source view."""
    try:
        segmentation = _read_json_object(_interaction_ref_path(interaction_root, segmentation_ref))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    cameras = segmentation.get("cameras")
    observation_handle = candidate_reference.get("observation_handle")
    candidate_handle = candidate_reference.get("candidate_handle")
    if (
        segmentation.get("record_type") != "RGBDSegmentationRecord"
        or not isinstance(cameras, list)
        or not isinstance(observation_handle, str)
        or not isinstance(candidate_handle, str)
    ):
        return None
    for camera in cameras:
        if (
            not isinstance(camera, Mapping)
            or camera.get("observation_handle") != observation_handle
        ):
            continue
        candidates = camera.get("candidates")
        if not isinstance(candidates, list):
            continue
        candidate = next(
            (
                item
                for item in candidates
                if isinstance(item, Mapping) and item.get("candidate_handle") == candidate_handle
            ),
            None,
        )
        source_artifacts = camera.get("source_artifacts")
        rgb = source_artifacts.get("rgb") if isinstance(source_artifacts, Mapping) else None
        bounds = candidate.get("pixel_bounds_uv") if isinstance(candidate, Mapping) else None
        minimum = bounds.get("minimum") if isinstance(bounds, Mapping) else None
        maximum = bounds.get("maximum") if isinstance(bounds, Mapping) else None
        mask = camera.get("label_mask_artifact")
        shape = mask.get("shape") if isinstance(mask, Mapping) else None
        rgb_ref = rgb.get("ref") if isinstance(rgb, Mapping) else None
        rgb_sha256 = rgb.get("sha256") if isinstance(rgb, Mapping) else None
        if (
            not isinstance(rgb_ref, str)
            or not isinstance(minimum, list)
            or not isinstance(maximum, list)
            or len(minimum) != 2
            or len(maximum) != 2
            or not all(isinstance(value, int) for value in [*minimum, *maximum])
            or not isinstance(shape, list)
            or len(shape) != 2
            or not all(isinstance(value, int) and value > 0 for value in shape)
        ):
            return None
        try:
            rgb_path = _interaction_ref_path(interaction_root, rgb_ref)
            rgb_bytes = rgb_path.read_bytes()
        except (OSError, ValueError):
            return None
        if isinstance(rgb_sha256, str) and hashlib.sha256(rgb_bytes).hexdigest() != rgb_sha256:
            return None
        image_mime = "image/jpeg" if rgb_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        image_uri = f"data:{image_mime};base64," + base64.b64encode(rgb_bytes).decode("ascii")
        height, width = shape
        x_min, y_min = minimum
        x_max, y_max = maximum
        rectangle_width = max(1, x_max - x_min + 1)
        rectangle_height = max(1, y_max - y_min + 1)
        label = escape(f"{state_label} · {candidate_handle}", quote=True)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
            f'<image width="{width}" height="{height}" href="{image_uri}"/>'
            f'<rect x="{x_min}" y="{y_min}" width="{rectangle_width}" '
            f'height="{rectangle_height}" fill="none" stroke="#22c55e" '
            'stroke-width="4"/>'
            f'<rect x="{x_min}" y="{max(0, y_min - 24)}" '
            f'width="{min(width - x_min, max(160, rectangle_width))}" height="24" '
            'fill="#052e16" fill-opacity="0.88"/>'
            f'<text x="{x_min + 5}" y="{max(17, y_min - 7)}" '
            f'fill="#f0fdf4" font-size="14" font-family="sans-serif">{label}</text>'
            "</svg>"
        )
        source_view_data_uri = "data:image/svg+xml;base64," + base64.b64encode(
            svg.encode("utf-8")
        ).decode("ascii")
        crop = _reviewed_candidate_crop(
            interaction_root,
            segmentation_ref=segmentation_ref,
            segmentation_sha256=hashlib.sha256(
                _interaction_ref_path(interaction_root, segmentation_ref).read_bytes()
            ).hexdigest(),
            observation_handle=observation_handle,
            candidate_handle=candidate_handle,
            observation_review_bindings=observation_review_bindings,
        )
        if crop is None:
            try:
                with Image.open(io.BytesIO(rgb_bytes)) as source_image:
                    crop_image = source_image.convert("RGB").crop(
                        (x_min, y_min, x_max + 1, y_max + 1)
                    )
                    crop_buffer = io.BytesIO()
                    crop_image.save(crop_buffer, format="PNG")
                crop = {
                    "data_uri": "data:image/png;base64,"
                    + base64.b64encode(crop_buffer.getvalue()).decode("ascii"),
                }
            except OSError:
                crop = {"data_uri": source_view_data_uri}
        return {
            **crop,
            "source_view_data_uri": source_view_data_uri,
            "rgb_evidence_ref": rgb_ref,
            "segmentation_ref": segmentation_ref,
            "pixel_bounds_uv": {
                "minimum": list(minimum),
                "maximum": list(maximum),
            },
        }
    return None


def _reviewed_candidate_crop(
    interaction_root: Path,
    *,
    segmentation_ref: str,
    segmentation_sha256: str,
    observation_handle: str,
    candidate_handle: str,
    observation_review_bindings: Mapping[str, str],
) -> dict[str, object] | None:
    """Load the hash-pinned VLM review crop for one exact candidate."""
    for review_ref, review_sha256 in sorted(observation_review_bindings.items()):
        try:
            review_path = _interaction_ref_path(interaction_root, review_ref)
            if hashlib.sha256(review_path.read_bytes()).hexdigest() != review_sha256:
                continue
            review = _read_json_object(review_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
        source = review.get("source_segmentation")
        candidates = review.get("candidates")
        if (
            review.get("record_type") != "ObservationCandidateReview"
            or review.get("status") != "accepted"
            or not isinstance(source, Mapping)
            or source.get("ref") != segmentation_ref
            or source.get("sha256") != segmentation_sha256
            or not isinstance(candidates, list)
        ):
            continue
        candidate = next(
            (
                item
                for item in candidates
                if isinstance(item, Mapping)
                and item.get("observation_handle") == observation_handle
                and item.get("candidate_handle") == candidate_handle
            ),
            None,
        )
        artifact = candidate.get("crop_artifact") if isinstance(candidate, Mapping) else None
        crop_ref = artifact.get("ref") if isinstance(artifact, Mapping) else None
        crop_sha256 = artifact.get("sha256") if isinstance(artifact, Mapping) else None
        if not isinstance(crop_ref, str) or not isinstance(crop_sha256, str):
            continue
        try:
            crop_bytes = _interaction_ref_path(interaction_root, crop_ref).read_bytes()
        except (OSError, ValueError):
            continue
        if hashlib.sha256(crop_bytes).hexdigest() != crop_sha256:
            continue
        return {
            "data_uri": "data:image/png;base64," + base64.b64encode(crop_bytes).decode("ascii"),
            "crop_evidence_ref": crop_ref,
            "observation_review_ref": review_ref,
            "description": candidate.get("description"),
            "uncertainty": candidate.get("uncertainty"),
        }
    return None


def _state_visuals(
    root: Path,
    state: Mapping[str, object],
    artifact_hashes: Mapping[str, str],
    observation_review_bindings: Mapping[str, str],
) -> list[dict[str, object]]:
    """Load every exact observed candidate cited by a grounded state's values."""
    from .agents.pa.ontology_grounding import resolve_json_pointer

    visuals: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for value in state["state_values"]:
        reference = value["value_ref"]
        record_ref = reference["record_ref"]
        field_path = reference["field_path"]
        if (record_ref, field_path) in seen:
            continue
        seen.add((record_ref, field_path))
        record_path = _interaction_ref_path(root, record_ref)
        if hashlib.sha256(record_path.read_bytes()).hexdigest() != artifact_hashes.get(record_ref):
            raise ValueError("State evidence has changed.")
        record = _read_json_object(record_path)
        segmentation_ref = None
        candidate_reference = None
        if record.get("record_type") == "RGBDSegmentationRecord":
            selected = resolve_json_pointer(record, field_path)
            for camera in record["cameras"]:
                for candidate in camera["candidates"]:
                    if candidate is selected:
                        segmentation_ref = record_ref
                        candidate_reference = {
                            "observation_handle": camera["observation_handle"],
                            "candidate_handle": candidate["candidate_handle"],
                        }
        elif record.get("record_type") == "RobotFrameLocationRecord":
            source = record.get("source_segmentation")
            segmentation_ref = source.get("ref") if isinstance(source, Mapping) else None
            candidate_reference = record.get("candidate_reference")
        if not isinstance(segmentation_ref, str) or not isinstance(candidate_reference, Mapping):
            continue
        visual = _annotated_candidate_rgb(
            root,
            segmentation_ref=segmentation_ref,
            candidate_reference=candidate_reference,
            observation_review_bindings=observation_review_bindings,
            state_label=value["name"],
        )
        if visual is not None:
            visuals.append(
                {
                    **visual,
                    "state_value_name": value["name"],
                    "source_record_ref": record_ref,
                    "source_field_path": field_path,
                    "candidate_reference": dict(candidate_reference),
                }
            )
    return visuals


def _validated_grounding_result(root: Path, proposal_ref: str) -> dict[str, object]:
    """Present PA claims and pinned evidence even when allocation is deferred."""
    from .agents.pa.grounding_contracts import validate_grounding_evidence

    proposal = _read_json_object(_interaction_ref_path(root, proposal_ref))
    if proposal.get("status") != "accepted":
        raise ValueError("Product grounding has not been accepted.")
    evidence_view = validate_grounding_evidence(root, proposal)
    target = proposal["output"]["target_feature"]
    caveats = proposal["compiled_delta"]["uncertainty"]
    product_context = _latest_interaction_json_object(
        root, "products/grounding/product_context/view_*.json"
    )
    raw_turtle = _interaction_ref_path(
        root, "products/grounding/ontology/interaction_abox.ttl"
    ).read_text(encoding="utf-8")
    artifact_hashes = {
        item["ref"]: item["sha256"] for item in proposal["grounding_evidence"]["input_artifacts"]
    }
    observation_review_bindings = {
        binding["record_ref"]: binding["record_sha256"]
        for binding in evidence_view.to_record()["typed_bindings"]
        if binding.get("output_symbol") == "ObservationCandidateReview"
        and binding.get("status") == "accepted"
        and artifact_hashes.get(binding["record_ref"]) == binding["record_sha256"]
    }

    def affected(value: object) -> list[object]:
        refs: set[str] = set()

        def visit(item: object) -> None:
            if isinstance(item, Mapping):
                refs.update(item.get("evidence_refs", []))
                state_name = item.get("state_name")
                if state_name in {"current_state", "desired_state"}:
                    for state_value in target[state_name]["state_values"]:
                        if state_value["name"] == item.get("state_value_name"):
                            visit(state_value)
                for nested in item.values():
                    visit(nested)
            elif isinstance(item, list):
                for nested in item:
                    visit(nested)

        visit(value)
        for artifact_ref in artifact_hashes:
            if artifact_ref in refs and artifact_ref.endswith(".json"):
                record = _read_json_object(_interaction_ref_path(root, artifact_ref))
                refs.update(record.get("evidence_refs", []))
        return [item for item in caveats if refs.intersection(item["evidence_refs"])]

    result: dict[str, object] = {
        "product_requirement": evidence_view.product_requirement,
        "target_feature": {**target, "source_uncertainty": caveats},
        "allocation_label": "Product grounding; allocation deferred",
        "resource_assignment_status": "deferred",
        "motion_executed": False,
        "process": _compact_iri(
            target["required_process"]["process_iri"], _turtle_prefixes(raw_turtle)
        ),
        "cad_identity": _cad_identity(
            root, typed_bindings=product_context["typed_bindings"], target_feature=target
        ),
        "ontology": _final_ontology_view(product_context, raw_turtle),
        "evidence": _final_result_evidence(product_context=product_context),
        "source_uncertainty": caveats,
        "limitations": [
            item["description"] + " [" + ", ".join(item["evidence_refs"]) + "]" for item in caveats
        ],
        "assembly_feature_association": [
            {**association, "source_uncertainty": affected(association)}
            for association in target.get("assembly_feature_association", [])
        ],
    }
    for state in ("current_state", "desired_state"):
        relationships = [
            association
            for association in result["assembly_feature_association"]
            if state in association.get("state_names", [])
        ]
        relationship_visuals = []
        for association in relationships:
            for endpoint in association["assembly_features"]:
                source_state = endpoint.get("state_name")
                if source_state not in {"current_state", "desired_state"} or source_state == state:
                    continue
                endpoint_values = [
                    value
                    for value in target[source_state]["state_values"]
                    if value["name"] == endpoint.get("state_value_name")
                ]
                relationship_visuals.extend(
                    {**visual, "source_state": source_state}
                    for visual in _state_visuals(
                        root,
                        {"state_values": endpoint_values},
                        artifact_hashes,
                        observation_review_bindings,
                    )
                )
        result[f"{state}_evidence"] = {
            "statement": target[state]["statement"]["text"],
            "state_values": target[state]["state_values"],
            "source_uncertainty": affected(target[state]),
            "relationships": relationships,
            "relationship_visuals": relationship_visuals,
            "reference_note": "Images are directly bound to desired_state; they do not establish that assembly is complete."
            if (state == "desired_state")
            else None,
            "candidate_visuals": _state_visuals(
                root,
                target[state],
                artifact_hashes,
                observation_review_bindings,
            ),
        }
    result["resource_comparison"] = _resource_comparison(root)
    return result


def _resource_comparison(
    root: Path,
    *,
    selected: str | None = None,
    tool_refs: list[str] | None = None,
) -> list[dict[str, object]]:
    """Show planning coverage without treating unavailable or unchecked arms as unreachable."""
    from .agents.pa.presentation_records import load_allocation_presentation
    from .agents.pa.resource_grounding import validate_location_planning_response

    presentation_path = root / "products/grounding/presentation/allocation_presentation_record.json"
    if not presentation_path.is_file():
        return []
    presentation = load_allocation_presentation(root)
    rows = {
        symbol: {
            "resource": symbol,
            "current_state": "Unchecked",
            "desired_state": "Unchecked",
            "selected": "Selected" if symbol == selected else "",
        }
        for symbol in presentation.resource_order
    }
    refs = (
        tool_refs
        if tool_refs is not None
        else [
            path.relative_to(root).as_posix()
            for path in sorted((root / "interaction_record").glob("allocation_tool_call_*.json"))
        ]
    )
    labels = {
        "accepted": "Reachable",
        "rejected": "Planning rejected",
        "needs_context": "Unavailable",
    }
    for ref in refs:
        call = _read_json_object(_interaction_ref_path(root, ref))
        if call.get("record_type") != "ProductAgentAllocationToolCall":
            continue
        symbol = call.get("arguments", {}).get("resource_symbol")
        if symbol not in rows:
            continue
        rows[symbol].update(current_state="Unavailable", desired_state="Unavailable")
        if not isinstance(call.get("result_ref"), str):
            continue
        path = _interaction_ref_path(root, call["result_ref"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != call.get("result_sha256"):
            raise ValueError("Resource comparison evidence changed.")
        check = _read_json_object(path)
        validation = check["validation"]
        validate_location_planning_response(validation["request"], validation["response"])
        for state in ("current_state", "desired_state"):
            statuses = {item["status"] for item in validation["response"]["state_locations"][state]}
            rows[symbol][state] = " / ".join(
                labels[status] for status in labels if status in statuses
            )
    return list(rows.values())


def _final_grounding_result(
    interaction_root: Path,
    completion: Mapping[str, object],
) -> dict[str, object]:
    """Build the operator result from one already validated completion bundle."""
    result = _validated_grounding_result(
        interaction_root, str(completion["ontology_projection_ref"])
    )
    selection = _read_json_object(
        _interaction_ref_path(interaction_root, completion["resource_selection_ref"])
    )
    reachability = _read_json_object(
        _interaction_ref_path(interaction_root, completion["reachability_check_ref"])
    )
    result.update(
        status=completion["status"],
        resource_assignment_status="complete",
        selected_resource=selection["selected_resource_symbol"],
        selected_resource_jid=selection["selected_resource_jid"],
        execution_mode=selection["selected_execution_mode"],
        process_symbol=selection["process_symbol"],
        allocation_label=completion["allocation_label"],
        validation_scope=completion["validation_scope"],
        motion_validation_performed=completion.get("motion_validation_performed", False),
        target_frame=reachability["target_frame"],
        reachability={**reachability, "record_ref": completion["reachability_check_ref"]},
        robot_agent_validation=reachability.get("validation"),
        phase_5_available=True,
        checked_constraints=completion["checked_constraints"],
        unvalidated_constraints=completion["unvalidated_constraints"],
    )
    result["resource_comparison"] = _resource_comparison(
        interaction_root,
        selected=selection["selected_resource_symbol"],
        tool_refs=[item["ref"] for item in completion["tool_call_refs"]],
    )
    for state in ("current_state", "desired_state"):
        result[f"{state}_evidence"]["reachability_locations"] = reachability["state_locations"][
            state
        ]
    result["limitations"].extend(completion["unvalidated_constraints"])
    return result


def _persisted_pa_timeline(
    product_requirement: str,
    *,
    turns: list[dict[str, object]],
    retrievals: list[dict[str, object]],
    clarifications: list[dict[str, object]],
    records: Mapping[str, object],
    final_result: Mapping[str, object] | None,
    terminal_failure: object,
) -> list[dict[str, str]]:
    """Build a compact semantic timeline from persisted interaction records."""
    del retrievals, clarifications
    events = [_timeline_event("accepted", "Requirement received", product_requirement)]
    tool_calls = [
        value
        for name, value in records.items()
        if name.startswith("tool_call_") and isinstance(value, Mapping)
    ]
    successful_calls = [item for item in tool_calls if item.get("failure") is None]
    evidence_detail = (
        f"{len(successful_calls)} approved evidence operations completed."
        if tool_calls
        else "No external evidence operation was needed for this grounding."
    )
    events.append(
        _timeline_event(
            "accepted" if successful_calls or not tool_calls else "failed",
            "Evidence investigated",
            evidence_detail,
        )
    )
    progress = _persisted_grounding_progress(records)
    if detail := _grounding_progress_text(progress):
        stopped = progress.get("stop_reason") not in {None, "complete", "clarification_required"}
        events.append(
            _timeline_event("waiting" if stopped else "accepted", "Grounding investigation", detail)
        )
    if final_result is not None and terminal_failure is None:
        events.append(
            _timeline_event(
                "accepted",
                "Target feature grounded",
                (
                    "The PA-authored target feature passed deterministic structure and evidence validation."
                ),
            )
        )
    if (
        final_result is not None
        and terminal_failure is None
        and final_result.get("resource_assignment_status") == "complete"
    ):
        resource = final_result.get("selected_resource")
        events.extend(
            [
                _timeline_event(
                    "accepted",
                    "Arm assigned",
                    f"{resource} · every capable arm checked against the grounded positions.",
                ),
                _timeline_event(
                    "accepted",
                    "Product grounding and arm assignment complete",
                    "Product goal grounded and RA selected. Position plans were validated; grasping and insertion remain unvalidated.",
                ),
            ]
        )
        return events
    latest_output = next(
        (
            turn.get("PA_output")
            for turn in reversed(turns)
            if isinstance(turn.get("PA_output"), Mapping)
        ),
        None,
    )
    if terminal_failure is not None:
        events.append(
            _timeline_event("failed", "Grounding incomplete", _failure_message(terminal_failure))
        )
    elif isinstance(latest_output, Mapping) and latest_output.get("grounding_status") == "complete":
        resource_status = latest_output.get("resource_assignment_status")
        events.append(
            _timeline_event(
                "accepted",
                "Product-state grounding complete",
                (
                    "The current and desired states are validated; resource assignment "
                    f"is {resource_status}."
                    if resource_status in {"deferred", "incomplete"}
                    else "The current and desired states and resource assignment are validated."
                ),
            )
        )
    elif isinstance(latest_output, Mapping) and latest_output.get("grounding_status") != "complete":
        message = latest_output.get("clarification_question")
        title = (
            "Clarification requested"
            if latest_output.get("grounding_status") == "clarification_required"
            else "Grounding incomplete"
        )
        events.append(
            _timeline_event(
                "waiting",
                title,
                (
                    str(message or "More context is required.")
                    if latest_output.get("grounding_status") == "clarification_required"
                    else _grounding_incomplete_detail(latest_output)
                ),
            )
        )
    return events


def _pa_ui_view(  # noqa: C901, PLR0915
    interaction: dict[str, object],
) -> dict[str, object]:
    """Build operator-facing text from one connected PA interaction."""
    product_requirement = interaction["product_requirement"]
    phase_3_1 = interaction["phase_3_1"]
    phase_3_2 = interaction["phase_3_2"]
    phase_3_3 = interaction.get("phase_3_3")
    if not isinstance(product_requirement, str) or not isinstance(phase_3_1, dict):
        raise ValueError("PA UI interaction result is malformed.")

    interaction_root = interaction["interaction_root"]
    interaction_identifier = interaction["interaction_identifier"]
    if not isinstance(interaction_root, Path) or not isinstance(interaction_identifier, str):
        raise ValueError("PA UI interaction path is malformed.")
    records = _interaction_records(interaction_root)
    turns = [
        record
        for name, record in records.items()
        if name.startswith("turn_") and isinstance(record, dict)
    ]
    retrievals = [
        record
        for name, record in records.items()
        if name.startswith("retrieval_") and isinstance(record, dict)
    ]
    clarifications = [
        record
        for name, record in records.items()
        if name.startswith("clarification_") and isinstance(record, dict)
    ]
    clarification_by_turn = {
        record.get("question_turn"): record
        for record in clarifications
        if isinstance(record.get("question_turn"), int)
    }
    pending_clarification_turn: int | None = None
    clarification_question: str | None = None
    latest_turn = turns[-1] if turns else None
    latest_output = latest_turn.get("PA_output") if isinstance(latest_turn, Mapping) else None
    latest_turn_number = latest_turn.get("turn") if isinstance(latest_turn, Mapping) else None
    if (
        isinstance(latest_output, Mapping)
        and latest_output.get("grounding_status") == "clarification_required"
        and isinstance(latest_output.get("clarification_question"), str)
        and isinstance(latest_turn_number, int)
        and latest_turn_number not in clarification_by_turn
    ):
        pending_clarification_turn = latest_turn_number
        clarification_question = str(latest_output["clarification_question"])
    latest_clarification = clarifications[-1] if clarifications else None
    cancelled = (
        isinstance(latest_clarification, dict) and latest_clarification.get("action") == "cancelled"
    )
    completion_candidates = [
        value
        for name, value in records.items()
        if name.startswith("context_completion_") and isinstance(value, dict)
    ]
    completion_record = _validated_persisted_completion(records, interaction_root)
    final_result: dict[str, object] | None = None
    presentation_failure: dict[str, str] | None = (
        {
            "reason": "invalid_context_completion",
            "message": "The persisted context completion is incompatible or changed. Start a fresh interaction.",
        }
        if completion_candidates and completion_record is None
        else None
    )
    if completion_record is not None:
        try:
            final_result = _final_grounding_result(
                interaction_root,
                completion_record,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            presentation_failure = {
                "reason": "final_result_unavailable",
                "message": f"Final grounding result could not be loaded: {type(exc).__name__}: {exc}",
            }
    completed = final_result is not None
    if (
        not completion_candidates
        and isinstance(latest_output, Mapping)
        and isinstance(latest_output.get("ontology_projection_ref"), str)
    ):
        try:
            final_result = _validated_grounding_result(
                interaction_root, str(latest_output["ontology_projection_ref"])
            )
            final_result["deferred_reason"] = latest_output.get("deferred_reason")
            if latest_output.get("grounding_status") == "incomplete":
                final_result["resource_assignment_status"] = "incomplete"
                final_result["deferred_reason"] = latest_output.get("insufficient_evidence")
        except (OSError, KeyError, TypeError, ValueError):
            presentation_failure = {
                "reason": "invalid_grounding",
                "message": "Grounding evidence could not be validated. Start a fresh interaction.",
            }
    grounding_status = (
        latest_output.get("grounding_status") if isinstance(latest_output, Mapping) else None
    )
    if (
        grounding_status == "incomplete"
        and isinstance(latest_output, Mapping)
        and latest_output.get("grounding_validation_code") not in _GROUNDING_VALIDATION_CODES
    ):
        presentation_failure = presentation_failure or {
            "reason": "invalid_grounding_result",
            "message": "The persisted grounding result is incompatible. Start a fresh interaction.",
        }
    terminal_failure = presentation_failure or _terminal_failure(
        phase_3_1,
        phase_3_2,
        phase_3_3,
        latest_turn,
    )
    grounding_unavailable = (
        isinstance(terminal_failure, dict)
        and terminal_failure.get("reason") == "grounding_unavailable"
    )
    if grounding_unavailable:
        activity_state, activity_color = "grounding unavailable", "amber"
        activity_message = (
            f"{_failure_message(terminal_failure)} No PA evidence decision was requested."
        )
    elif terminal_failure is not None:
        activity_state, activity_color = "failed", "red"
        activity_message = _failure_message(terminal_failure)
    elif cancelled:
        activity_state, activity_color = "cancelled", "grey"
        activity_message = "The operator cancelled the pending clarification."
    elif isinstance(clarification_question, str):
        activity_state, activity_color = "clarification needed", "amber"
        activity_message = "ProductAgent is waiting for the operator's reply."
    elif completed:
        activity_state, activity_color = "grounding complete", "green"
        activity_message = "The grounded states and validated arm assignment are available below."
    elif grounding_status == "complete":
        resource_status = (
            latest_output.get("resource_assignment_status")
            if isinstance(latest_output, Mapping)
            else None
        )
        activity_state, activity_color = "product state grounded", "green"
        activity_message = (
            "The current and desired states are validated; resource assignment "
            f"is {resource_status}."
            if resource_status in {"deferred", "incomplete"}
            else "The current and desired states are validated."
        )
    elif grounding_status in {"incomplete", "ontology_gap"}:
        activity_state, activity_color = "grounding incomplete", "amber"
        activity_message = (
            _grounding_incomplete_detail(latest_output)
            if isinstance(latest_output, Mapping)
            else "Grounding is incomplete."
        )
    elif grounding_status in {"waiting_for_evidence", "waiting_for_user"}:
        activity_state, activity_color = "grounding waiting", "amber"
        activity_message = f"PA grounding is {str(grounding_status).replace('_', ' ')}."
    elif any(
        name.startswith("tool_call_")
        and isinstance(value, Mapping)
        and value.get("failure") is None
        for name, value in records.items()
    ):
        activity_state, activity_color = "stopped", "grey"
        activity_message = (
            "The interaction stopped after evidence serving and has no validated completion."
        )
    else:
        activity_state, activity_color = "stopped", "grey"
        activity_message = "The interaction stopped before validated completion."

    timeline = _persisted_pa_timeline(
        product_requirement,
        turns=turns,
        retrievals=retrievals,
        clarifications=clarifications,
        records=records,
        final_result=final_result,
        terminal_failure=terminal_failure,
    )
    progress = _persisted_grounding_progress(records)
    if detail := _grounding_progress_text(progress):
        activity_message += " " + detail
    phase_5_1 = read_phase_5_1_diagnostic(interaction_root).to_view()
    phase_5_2 = read_phase_5_2_diagnostic(interaction_root).to_view()
    return {
        "activity_state": activity_state,
        "activity_color": activity_color,
        "activity_message": activity_message,
        "timeline": timeline,
        "final_result": final_result,
        "clarification": (
            clarification_question if isinstance(clarification_question, str) else ""
        ),
        "pending_clarification_turn": str(pending_clarification_turn or ""),
        "phase_5_1": phase_5_1,
        "phase_5_2": phase_5_2,
        "diagnostics": {
            "interaction_identifier": interaction_identifier,
            "interaction_path": str(interaction_root),
            "turn_count": len(turns),
            "evidence_count": sum(
                1
                for name, value in records.items()
                if name.startswith("tool_call_")
                and isinstance(value, Mapping)
                and value.get("failure") is None
            ),
            "decision_count": len(turns),
            "failure": _product_agent_request_failure_text(terminal_failure),
            "grounding_progress": dict(progress),
        },
    }


def _validated_persisted_assessment(
    decision: dict[str, object],
) -> dict[str, object] | None:
    if decision.get("failure") is not None:
        return None
    assessment = decision.get("Phase_4_3_output")
    expected_keys = {
        "unresolved_semantic_need",
        "needed_context",
        "context understanding complete",
    }
    session_keys = expected_keys | {"grounding_status"}
    if not isinstance(assessment, dict) or frozenset(assessment) not in {
        frozenset(expected_keys),
        frozenset(session_keys),
    }:
        return None
    complete = assessment["context understanding complete"]
    needed_context = assessment["needed_context"]
    if not isinstance(complete, bool):
        return None
    if complete:
        terminal_values = (
            needed_context,
            assessment["unresolved_semantic_need"],
        )
        return assessment if all(value is None for value in terminal_values) else None
    if assessment.get("grounding_status") in {"incomplete", "ontology_gap"}:
        return (
            assessment
            if needed_context is None and assessment["unresolved_semantic_need"] is None
            else None
        )
    if not isinstance(needed_context, dict):
        return None
    semantic_need = assessment["unresolved_semantic_need"]
    if not isinstance(semantic_need, dict) or set(semantic_need) != {
        "kind",
        "symbol",
        "description",
    }:
        return None
    clarification = needed_context.get("clarification_question")
    if isinstance(clarification, str) and (semantic_need.get("kind") != "user_intent"):
        return None
    return assessment


def _interaction_records(interaction_root: Path) -> dict[str, object]:
    records: dict[str, object] = {}
    ontology_manifest_path = interaction_root / "products/grounding/ontology/abox_manifest.json"
    if ontology_manifest_path.is_file():
        try:
            records["ontology_initialization"] = json.loads(
                ontology_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            records["ontology_initialization"] = {
                "record_error": f"{type(exc).__name__}: {exc}",
            }
    record_root = interaction_root / "interaction_record"
    paths = list(record_root.glob("turn_*.json"))
    paths.extend(record_root.glob("retrieval_*.json"))
    paths.extend(record_root.glob("interpretation_*.json"))
    paths.extend(record_root.glob("decision_*.json"))
    paths.extend(record_root.glob("clarification_*.json"))
    paths.extend(record_root.glob("context_completion_*.json"))
    paths.extend(record_root.glob("producer_selection_*.json"))
    paths.extend(record_root.glob("tool_call_*.json"))
    settings_path = record_root / "pa_context_settings.json"
    if settings_path.is_file():
        paths.append(settings_path)
    for record_path in sorted(paths, key=_interaction_record_sort_key):
        record_name = record_path.stem
        try:
            records[record_name] = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            records[record_name] = {
                "record_error": f"{type(exc).__name__}: {exc}",
            }
    grounding_paths = [
        (
            "ontology_assertion_provenance",
            interaction_root / "products/grounding/ontology/assertion_provenance.json",
        )
    ]
    grounding_paths.extend(
        (f"ontology_{path.stem}", path)
        for path in sorted((interaction_root / "products/grounding/ontology").glob("delta_*.json"))
    )
    grounding_paths.extend(
        (f"product_context_{path.stem}", path)
        for path in sorted(
            (interaction_root / "products/grounding/product_context").glob("view_*.json")
        )
    )
    grounding_paths.extend(
        (f"ontology_grounding_{path.stem}", path)
        for path in sorted(
            (interaction_root / "products/grounding/ontology_grounding").glob("proposal_*.json")
        )
    )
    grounding_paths.extend(
        (f"grounding_{path.stem}", path)
        for path in sorted(
            (interaction_root / "products/grounding/ontology_grounding").glob("request_*.json")
        )
    )
    grounding_paths.extend(
        (f"grounding_completion_{path.stem}", path)
        for path in sorted((interaction_root / "products/grounding/completion").glob("*.json"))
    )
    for record_name, record_path in grounding_paths:
        if not record_path.is_file():
            continue
        try:
            records[record_name] = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            records[record_name] = {
                "record_error": f"{type(exc).__name__}: {exc}",
            }
    return records


def _persisted_turn_result(turn: Mapping[str, object]) -> dict[str, object]:
    """Reconstruct the public turn result from one persisted interaction record."""
    failure = turn.get("failure")
    if isinstance(failure, Mapping):
        return {"failure": dict(failure)}
    output = turn.get("PA_output")
    return dict(output) if isinstance(output, Mapping) else {}


def _latest_pa_ui_interaction(contexts_root: Path) -> dict[str, object] | None:
    """Recover the newest real interaction by requirement-record modification time."""
    root = contexts_root.resolve()
    candidates: list[tuple[int, Path]] = []
    for candidate in root.glob("interaction_*"):
        if not candidate.is_dir():
            continue
        try:
            candidate.resolve().relative_to(root)
            requirement_path = _interaction_ref_path(
                candidate,
                "products/user_requirement/product_requirement.json",
            )
            candidates.append((requirement_path.stat().st_mtime_ns, candidate))
        except (OSError, ValueError):
            continue
    if not candidates:
        return None

    _, interaction_root = max(candidates, key=lambda item: item[0])
    requirement = _read_json_object(
        _interaction_ref_path(
            interaction_root,
            "products/user_requirement/product_requirement.json",
        )
    ).get("product_requirement")
    if not isinstance(requirement, str):
        return None

    records = _interaction_records(interaction_root)
    turns = [
        value
        for name, value in records.items()
        if name.startswith("turn_") and isinstance(value, Mapping)
    ]
    retrievals = [
        value
        for name, value in records.items()
        if name.startswith("retrieval_") and isinstance(value, dict)
    ]
    settings = records.get("pa_context_settings")
    max_pa_turns = settings.get("max_pa_turns") if isinstance(settings, Mapping) else 12
    if not isinstance(max_pa_turns, int) or isinstance(max_pa_turns, bool):
        max_pa_turns = 12

    phase_3_1 = (
        _persisted_turn_result(turns[0])
        if turns
        else {"needed_context": None, "context understanding complete": False}
    )
    phase_3_3 = _persisted_turn_result(turns[-1]) if len(turns) > 1 else None
    if phase_3_3 is None:
        latest_decision = _latest_record(records, "decision_")
        assessment = (
            _validated_persisted_assessment(latest_decision)
            if isinstance(latest_decision, dict)
            else None
        )
        if assessment is not None:
            phase_3_3 = assessment
    return {
        "interaction_identifier": interaction_root.name,
        "interaction_root": interaction_root,
        "product_requirement": requirement,
        "phase_3_1": phase_3_1,
        "phase_3_2": retrievals[0] if retrievals else None,
        "phase_3_3": phase_3_3,
        "max_pa_turns": max_pa_turns,
        "recovered": True,
    }


def _interaction_record_sort_key(path: Path) -> tuple[int, int]:
    if path.stem == "pa_context_settings":
        return (0, 0)
    prefix, _, suffix = path.stem.rpartition("_")
    order = {
        "turn": 0,
        "retrieval": 1,
        "interpretation": 2,
        "decision": 3,
        "clarification": 4,
        "context_completion": 5,
        "producer_selection": 6,
        "tool_call": 1,
    }.get(prefix, 4)
    return (int(suffix), order)


def _validated_persisted_completion(
    records: dict[str, object],
    interaction_root: Path,
) -> dict[str, object] | None:
    values = [
        value
        for name, value in records.items()
        if name.startswith("context_completion_") and isinstance(value, dict)
    ]
    if len(values) != 1:
        return None
    try:
        return load_pa_context_grounding_completion(interaction_root).to_record()
    except (OSError, TypeError, ValueError):
        return None


def _terminal_failure(*results: object) -> object:
    for result in reversed(results):
        if isinstance(result, dict) and result.get("failure") is not None:
            return result["failure"]
    return None


def _failure_message(failure: object) -> str:
    if not isinstance(failure, dict):
        return "The connected PA workflow returned a malformed failure."
    message = failure.get("message")
    return message if isinstance(message, str) else _json_text(failure)


def _product_agent_request_failure_text(failure: object) -> str:
    """Format compact failure details without exposing requests or raw payloads."""
    if not isinstance(failure, dict):
        return ""
    lines = [
        f"reason: {_json_text(failure.get('reason'))}",
        f"message: {_json_text(failure.get('message'))}",
    ]
    diagnostic = failure.get("diagnostic")
    if not isinstance(diagnostic, dict):
        return "\n".join(lines)
    lines.append("diagnostic")
    for field in (
        "stage",
        "exception",
        "status_code",
        "request_id",
        "error_type",
        "param",
        "code",
        "message",
    ):
        lines.append(f"{field}: {_json_text(diagnostic.get(field))}")
    return "\n".join(lines)


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)


def _concise_final_grounding_summary(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Return only the operator-readable facts for the visible result card."""
    cad_identity = result.get("cad_identity")
    context_ref = (
        cad_identity.get("context_ref")
        if isinstance(cad_identity, Mapping)
        and cad_identity.get("status") == "accepted"
        and isinstance(cad_identity.get("context_ref"), str)
        else None
    )
    cad_name = Path(context_ref).name if isinstance(context_ref, str) else None
    current = result.get("current_state_evidence")
    desired = result.get("desired_state_evidence")
    current_mapping = current if isinstance(current, Mapping) else {}
    desired_mapping = desired if isinstance(desired, Mapping) else {}
    reachability = result.get("reachability")
    reachability_mapping = reachability if isinstance(reachability, Mapping) else {}
    validation = result.get("robot_agent_validation")
    validation_mapping = validation.get("response", {}) if isinstance(validation, Mapping) else {}
    deferred = result.get("resource_assignment_status") in {"deferred", "incomplete"}
    target_lines = [
        f"CAD: {cad_name}" if cad_name else "CAD identity unresolved",
    ]
    if deferred:
        target_lines.extend(
            [
                "Resource assignment is still incomplete.",
                str(result.get("deferred_reason") or "Resource assignment is deferred."),
            ]
        )
    else:
        target_lines.extend(
            [
                f"target frame: {result.get('target_frame')}",
                f"current references checked: {len(current_mapping.get('reachability_locations', []))}",
                f"destination references checked: {len(desired_mapping.get('reachability_locations', []))}",
            ]
        )
    return {
        "target": "\n".join(line for line in target_lines if line),
        "current_state": _concise_state_summary(current_mapping),
        "desired_state": _concise_state_summary(desired_mapping),
        "reachability": "Not checked: resource assignment is deferred."
        if deferred
        else "\n".join(
            (
                f"status: {reachability_mapping.get('status')}",
                f"resource: {reachability_mapping.get('resource_symbol')}",
                f"target frame: {reachability_mapping.get('target_frame')}",
            )
        ),
        "validation": "Motion validation not performed. Primitive composition remains a separate step."
        if result.get("motion_validation_performed") is False
        else "Not checked: resource assignment is deferred."
        if deferred
        else "\n".join(
            (
                f"status: {validation_mapping.get('status')}",
                "Position checks · no motion executed",
            )
        ),
    }


def _concise_state_summary(state: Mapping[str, object]) -> str:
    """Keep the PA statement verbatim; evidence and caveats belong in details."""
    statement = state.get("statement")
    return statement if isinstance(statement, str) else "No grounded state statement is available."


def _render_state_visuals(container: Any, state: Mapping[str, object]) -> None:
    """Render only the supplied direct state bindings, with descriptions expandable."""
    visuals = state.get("candidate_visuals", [])
    container.clear()
    with container:
        if not visuals:
            ui.label("No image is directly bound to this state.").classes("text-xs text-slate-500")
        for visual in visuals:
            label = visual.get("state_value_name") or "Observed candidate"
            if visual.get("source_state"):
                label = f"{label} · source: {visual['source_state']}"
            ui.label(label).classes("text-xs font-semibold text-slate-700")
            ui.image(visual["data_uri"]).classes(
                "w-full min-w-0 max-w-full max-h-80 object-contain rounded border border-slate-200"
            )
            with ui.expansion("Image evidence", icon="image").classes("w-full min-w-0"):
                for field in (
                    "description",
                    "uncertainty",
                    "source_record_ref",
                    "source_field_path",
                ):
                    if visual.get(field):
                        ui.label(str(visual[field])).classes("text-xs whitespace-pre-wrap")
                if visual.get("source_view_data_uri"):
                    ui.image(visual["source_view_data_uri"]).classes(
                        "w-full min-w-0 max-w-full max-h-80 object-contain"
                    )


def _render_pa_timeline_events(
    container: Any,
    events: list[dict[str, str]],
) -> None:
    """Replace the live timeline with the supplied compact events."""
    colors = {
        "accepted": "text-emerald-600",
        "running": "text-indigo-600",
        "waiting": "text-amber-600",
        "failed": "text-red-600",
        "stopped": "text-slate-500",
    }
    icons = {
        "accepted": "check_circle",
        "running": "pending",
        "waiting": "help",
        "failed": "error",
        "stopped": "cancel",
    }
    container.clear()
    with container:
        for event in events:
            state = event.get("state", "running")
            with ui.row().classes("w-full items-start gap-3 flex-nowrap"):
                ui.icon(icons.get(state, "circle")).classes(
                    f"mt-0.5 {colors.get(state, 'text-slate-500')}"
                )
                with ui.column().classes("min-w-0 flex-1 gap-0"):
                    ui.label(event.get("title", "ProductAgent event")).classes(
                        "text-sm font-semibold text-slate-800"
                    )
                    ui.label(event.get("detail", "")).classes(
                        "text-xs text-slate-600 whitespace-pre-wrap break-words"
                    )


def _render_state_statement() -> dict[str, Any]:
    """Render an exact statement with a four-line preview and responsive expansion."""
    statement = ui.label("").classes("s2p-state-statement text-sm text-slate-700")
    expanded = False
    revision = 0
    measuring = False

    def toggle() -> None:
        nonlocal expanded, revision
        revision += 1
        expanded = not expanded
        statement.classes(
            add="s2p-state-expanded" if expanded else "",
            remove="" if expanded else "s2p-state-expanded",
        )
        button.set_text("Show less" if expanded else "Show more")

    button = ui.button("Show more", on_click=toggle).props("flat dense no-caps size=sm")
    button.set_visibility(False)

    async def measure() -> None:
        nonlocal measuring
        if (
            expanded
            or measuring
            or statement.is_deleted
            or button.is_deleted
            or not statement.client.has_socket_connection
        ):
            return
        measured_revision = revision
        measuring = True
        try:
            overflowing = await statement.client.run_javascript(
                f"(() => {{ const e = document.getElementById('c{statement.id}'); return !!e && e.scrollHeight > e.clientHeight + 1; }})()"
            )
        except TimeoutError:
            # Browser measurement is optional; the exact statement must stay accessible.
            overflowing = bool(statement.text)
        finally:
            measuring = False
        if (
            measured_revision == revision
            and not statement.is_deleted
            and not button.is_deleted
            and statement.client.has_socket_connection
        ):
            button.set_visibility(bool(overflowing))

    def reset() -> None:
        nonlocal expanded, revision
        revision += 1
        expanded = False
        statement.classes(remove="s2p-state-expanded")
        button.set_text("Show more")
        button.set_visibility(bool(statement.text))
        ui.timer(0.1, measure, once=True)

    ui.element("q-resize-observer").on("resize", measure)
    return {"statement": statement, "show_more": button, "reset": reset}


def _render_final_grounding_result() -> dict[str, Any]:
    """Render the hidden final grounding result and return updateable elements."""
    ui.add_css("""
        .s2p-grounding-result, .s2p-grounding-result * { min-width: 0; box-sizing: border-box; }
        .s2p-grounding-result { max-width: 100%; overflow-wrap: anywhere; }
        .s2p-grounding-result .q-expansion-item, .s2p-grounding-result .q-expansion-item__content,
        .s2p-grounding-result .q-card, .s2p-grounding-result .q-img { max-width: 100%; }
        .s2p-grounding-result pre, .s2p-grounding-result code { max-width: 100%; white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; }
        .s2p-grounding-result .q-table { table-layout: fixed; width: 100%; }
        .s2p-grounding-result .q-table td { white-space: normal; overflow-wrap: anywhere; }
        .s2p-state-statement { white-space: pre-wrap; overflow-wrap: anywhere; display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 4; overflow: hidden; }
        .s2p-state-statement.s2p-state-expanded { display: block; -webkit-line-clamp: unset; overflow: visible; }
    """)
    with ui.card().classes(
        "s2p-grounding-result w-full min-w-0 max-w-full border-2 border-emerald-200 bg-white shadow-sm"
    ) as card:
        with ui.row().classes("w-full items-start justify-between gap-3"):
            with ui.column().classes("gap-0"):
                ui.label("Final Grounding Result").classes("text-xl font-semibold text-slate-900")
                result_kind_value = ui.label(
                    "Validated plan-only allocation · no motion executed"
                ).classes("text-xs text-slate-500")
            status_badge = ui.badge("validated plan-only allocation").props("color=green outline")

        ui.label("product_requirement").classes("text-xs font-semibold text-slate-500")
        requirement_value = ui.label("").classes(
            "text-base font-medium text-slate-900 whitespace-pre-wrap"
        )
        with ui.expansion("PA-authored target feature details", icon="data_object").classes(
            "w-full border border-slate-200 rounded"
        ):
            target_feature_value = ui.code("", language="json").classes(
                "w-full min-w-0 max-w-full text-xs overflow-x-auto"
            )

        with ui.element("div").classes(
            "grid grid-cols-1 md:grid-cols-2 w-full min-w-0 gap-3 items-stretch"
        ):
            with ui.card().classes(
                "min-w-0 max-w-full border border-slate-200 bg-slate-50 shadow-none"
            ):
                ui.label("Process").classes("text-xs font-semibold text-slate-500")
                process_value = ui.label("").classes("text-base font-semibold text-slate-900")
            with ui.card().classes(
                "min-w-0 max-w-full border border-slate-200 bg-slate-50 shadow-none"
            ):
                ui.label("Selected Resource").classes("text-xs font-semibold text-slate-500")
                resource_value = ui.label("").classes("text-base font-semibold text-slate-900")
            with ui.card().classes(
                "min-w-0 max-w-full border border-slate-200 bg-slate-50 shadow-none"
            ):
                ui.label("CAD and grounded states").classes("text-xs font-semibold text-slate-500")
                target_value = ui.label("").classes(
                    "text-sm font-medium text-slate-900 whitespace-pre-wrap"
                )
                with ui.expansion("Grounding record details", icon="data_object").classes(
                    "w-full border border-slate-200 rounded"
                ):
                    target_details_value = ui.code("", language="json").classes(
                        "w-full min-w-0 max-w-full text-xs overflow-x-auto"
                    )

        relationships_value = ui.column().classes("w-full min-w-0 gap-2")
        ui.label("PA-Selected Feature-State Evidence").classes(
            "text-sm font-semibold text-slate-800"
        )
        state_elements = {}
        with ui.element("div").classes(
            "grid grid-cols-1 md:grid-cols-2 w-full min-w-0 gap-3 items-stretch"
        ):
            for prefix, title, color in (
                ("current", "Current State · currentstate_0001", "sky"),
                ("desired", "Desired State · desiredstate_0001", "violet"),
            ):
                with ui.card().classes(
                    f"min-w-0 max-w-full border border-{color}-200 bg-{color}-50 shadow-none"
                ):
                    ui.label(title).classes(f"text-sm font-semibold text-{color}-900")
                    state_elements[f"{prefix}_images"] = ui.column().classes("w-full gap-3")
                    statement_widget = _render_state_statement()
                    state_elements[f"{prefix}_state"] = statement_widget["statement"]
                    state_elements[f"{prefix}_preview_reset"] = statement_widget["reset"]
                    state_elements[f"{prefix}_show_more"] = statement_widget["show_more"]
                    state_elements[f"{prefix}_caveat_count"] = ui.label("").classes(
                        "text-xs text-amber-800"
                    )
                    with ui.expansion("Evidence details", icon="data_object").classes(
                        f"w-full border border-{color}-200 rounded"
                    ):
                        state_elements[f"{prefix}_caveats"] = ui.column().classes(
                            "w-full min-w-0 gap-2"
                        )
                        state_elements[f"{prefix}_relationship_images"] = ui.column().classes(
                            "w-full min-w-0 gap-2"
                        )
                        state_elements[f"{prefix}_state_details"] = ui.code(
                            "", language="json"
                        ).classes("w-full min-w-0 max-w-full text-xs overflow-x-auto")

        with ui.element("div").classes(
            "grid grid-cols-1 md:grid-cols-2 w-full min-w-0 gap-3 items-stretch"
        ):
            with ui.card().classes(
                "min-w-0 max-w-full border border-slate-200 bg-slate-50 shadow-none"
            ):
                ui.label("Reachability · both states").classes(
                    "text-xs font-semibold text-slate-500"
                )
                reachability_value = ui.label("").classes(
                    "text-sm font-medium text-slate-900 whitespace-pre-wrap"
                )
                with ui.expansion("Reachability record details", icon="data_object").classes(
                    "w-full border border-slate-200 rounded"
                ):
                    reachability_details = ui.code("", language="json").classes(
                        "w-full min-w-0 max-w-full text-xs overflow-x-auto"
                    )
            with ui.card().classes(
                "min-w-0 max-w-full border border-slate-200 bg-slate-50 shadow-none"
            ):
                ui.label("MoveIt position planning").classes("text-xs font-semibold text-slate-500")
                validation_value = ui.label("").classes(
                    "text-sm font-medium text-slate-900 whitespace-pre-wrap"
                )
                with ui.expansion("RobotAgent record details", icon="data_object").classes(
                    "w-full border border-slate-200 rounded"
                ):
                    validation_details = ui.code("", language="json").classes(
                        "w-full min-w-0 max-w-full text-xs overflow-x-auto"
                    )

        ui.label("Resource comparison").classes("text-sm font-semibold text-slate-800")
        comparison = (
            ui.table(
                columns=[
                    {"name": field, "label": label, "field": field}
                    for field, label in (
                        ("resource", "Resource"),
                        ("current_state", "Current location"),
                        ("desired_state", "Desired location"),
                        ("selected", "Selection"),
                    )
                ],
                rows=[],
                row_key="resource",
            )
            .props("flat bordered wrap-cells hide-bottom")
            .classes("w-full min-w-0 max-w-full")
        )
        ui.label("Validated Evidence").classes("text-sm font-semibold text-slate-800")
        evidence_container = ui.row().classes("w-full gap-2 flex-wrap")

        with ui.expansion("Known non-blocking context limits", icon="warning").classes(
            "w-full border border-amber-200 rounded"
        ) as limitations_expansion:
            limitations_value = ui.label("").classes(
                "text-xs text-amber-800 p-2 whitespace-pre-wrap break-words"
            )
        limitations_expansion.set_visibility(False)

        ui.separator()
        ui.label("Final Ontology").classes("text-lg font-semibold text-slate-900")
        ui.label("Authoritative final interaction ABox").classes("text-xs text-slate-500")
        ontology_graph = ui.mermaid("flowchart LR\n    empty[No final ABox]").classes(
            "w-full overflow-x-auto"
        )
        ontology_table = (
            ui.table(
                columns=[
                    {"name": "subject", "label": "Subject", "field": "subject"},
                    {
                        "name": "predicate",
                        "label": "Predicate",
                        "field": "predicate",
                    },
                    {"name": "object", "label": "Object", "field": "object"},
                ],
                rows=[],
                row_key="row_id",
            )
            .props("flat bordered wrap-cells hide-bottom")
            .classes("w-full")
        )
        with ui.expansion("Raw Turtle", icon="data_object").classes(
            "w-full border border-slate-200 rounded"
        ):
            raw_turtle_value = ui.code("", language="turtle").classes(
                "w-full min-w-0 max-w-full text-xs overflow-x-auto"
            )
    card.set_visibility(False)
    return {
        "card": card,
        "status_badge": status_badge,
        "result_kind": result_kind_value,
        "requirement": requirement_value,
        "target_feature": target_feature_value,
        "relationships": relationships_value,
        "resource_comparison": comparison,
        "process": process_value,
        "resource": resource_value,
        "target": target_value,
        "target_details": target_details_value,
        **state_elements,
        "reachability": reachability_value,
        "reachability_details": reachability_details,
        "validation": validation_value,
        "validation_details": validation_details,
        "evidence": evidence_container,
        "limitations_expansion": limitations_expansion,
        "limitations": limitations_value,
        "ontology_graph": ontology_graph,
        "ontology_table": ontology_table,
        "raw_turtle": raw_turtle_value,
    }


def _apply_final_grounding_result(
    elements: Mapping[str, Any],
    result: Mapping[str, object] | None,
) -> None:
    """Apply one validated final result or hide the entire result section."""
    card = elements["card"]
    if result is None:
        card.set_visibility(False)
        return
    allocation_label = str(
        result.get("allocation_label") or "resource assignment validated by MoveIt"
    )
    elements["status_badge"].set_text(allocation_label)
    elements["result_kind"].set_text(f"{allocation_label} · no motion executed")
    elements["requirement"].set_text(str(result.get("product_requirement", "")))
    target_feature = result.get("target_feature")
    elements["target_feature"].content = (
        json.dumps(
            target_feature,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        if isinstance(target_feature, Mapping)
        else ""
    )
    elements["target_feature"].update()
    relationships = result.get("assembly_feature_association", [])
    elements["relationships"].clear()
    with elements["relationships"]:
        for association in relationships:
            with ui.expansion(association["assembly"]["name"], icon="link").classes(
                "w-full min-w-0"
            ):
                ui.label(f"{len(association.get('source_uncertainty', []))} caveats").classes(
                    "text-xs text-amber-800"
                )
                for endpoint in association["assembly_features"]:
                    ui.label(f"{endpoint['owner']['name']} / {endpoint['name']}").classes(
                        "text-sm whitespace-pre-wrap"
                    )
                ui.code(_json_text(association), language="json").classes(
                    "w-full min-w-0 max-w-full text-xs"
                )
    elements["relationships"].set_visibility(bool(relationships))
    elements["resource_comparison"].rows = result.get("resource_comparison", [])
    elements["resource_comparison"].update()
    process_text = " · ".join(
        str(result[field])
        for field in ("process_symbol", "process")
        if result.get(field) not in {None, ""}
    )
    elements["process"].set_text(process_text)
    resource_text = " · ".join(
        str(result[field])
        for field in (
            "selected_resource",
            "selected_resource_jid",
            "execution_mode",
        )
        if result.get(field) not in {None, ""}
    )
    elements["resource"].set_text(resource_text)
    if result.get("resource_assignment_status") == "deferred":
        elements["resource"].set_text("Deferred")
    elif result.get("resource_assignment_status") == "incomplete":
        elements["resource"].set_text("Not assigned — grounding incomplete")
    summary = _concise_final_grounding_summary(result)
    elements["target"].set_text(str(summary["target"]))
    elements["target_details"].content = _json_text(
        {
            "cad_identity": result.get("cad_identity"),
            "target_context_ref": result.get("target_context_ref"),
            "target_frame": result.get("target_frame"),
            "location": result.get("location"),
            "pose": result.get("pose"),
            "robot_frame_conversion": result.get("robot_frame_conversion"),
            "CAD_centroid_translation_m": result.get("CAD_centroid_translation_m"),
        }
    )
    elements["target_details"].update()

    for prefix, result_key in (
        ("current", "current_state_evidence"),
        ("desired", "desired_state_evidence"),
    ):
        state_evidence = result.get(result_key)
        state_mapping = state_evidence if isinstance(state_evidence, Mapping) else {}
        elements[f"{prefix}_state"].set_text(str(summary[f"{prefix}_state"]))
        elements[f"{prefix}_preview_reset"]()
        caveats = state_mapping.get("source_uncertainty", [])
        elements[f"{prefix}_caveat_count"].set_text(f"{len(caveats)} caveats")
        elements[f"{prefix}_caveats"].clear()
        with elements[f"{prefix}_caveats"]:
            for caveat in caveats:
                ui.label(caveat["description"]).classes("text-xs whitespace-pre-wrap")
                ui.label("\n".join(caveat["evidence_refs"])).classes(
                    "text-xs whitespace-pre-wrap text-slate-500"
                )
        relationship_visuals = state_mapping.get("relationship_visuals", [])
        elements[f"{prefix}_relationship_images"].set_visibility(bool(relationship_visuals))
        _render_state_visuals(
            elements[f"{prefix}_relationship_images"], {"candidate_visuals": relationship_visuals}
        )
        elements[f"{prefix}_state_details"].content = _json_text(
            {
                key: value
                for key, value in state_mapping.items()
                if key
                not in {
                    "statement",
                    "candidate_visuals",
                    "relationship_visuals",
                    "source_uncertainty",
                }
            }
        )
        elements[f"{prefix}_state_details"].update()
        _render_state_visuals(elements[f"{prefix}_images"], state_mapping)

    for element_key, result_key in (
        ("reachability", "reachability"),
        ("validation", "robot_agent_validation"),
    ):
        value = result.get(result_key)
        elements[element_key].set_text(str(summary[element_key]))
        elements[f"{element_key}_details"].content = (
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
            if isinstance(value, Mapping)
            else "unavailable"
        )
        elements[f"{element_key}_details"].update()

    evidence_container = elements["evidence"]
    evidence_container.clear()
    evidence = result.get("evidence")
    with evidence_container:
        if isinstance(evidence, list):
            for item in evidence:
                if not isinstance(item, Mapping):
                    continue
                status = str(item.get("status", "recorded"))
                color = "green" if status == "accepted" else "amber"
                ui.badge(f"{item.get('label')} · {status}").props(f"color={color} outline")

    limitations = result.get("limitations")
    limitation_values = limitations if isinstance(limitations, list) else []
    elements["limitations"].set_text("\n".join(f"• {value}" for value in limitation_values))
    elements["limitations_expansion"].set_text(
        f"Caveats ({len(result.get('source_uncertainty', []))}) and validation limits"
    )
    elements["limitations_expansion"].set_visibility(bool(limitation_values))

    ontology = result.get("ontology")
    if not isinstance(ontology, Mapping):
        raise ValueError("Final ontology view is malformed.")
    elements["ontology_graph"].content = str(ontology.get("mermaid", ""))
    elements["ontology_graph"].update()
    rows = ontology.get("rows")
    elements["ontology_table"].rows = [
        {"row_id": index, **row}
        for index, row in enumerate(rows if isinstance(rows, list) else [])
        if isinstance(row, dict)
    ]
    elements["raw_turtle"].content = str(ontology.get("raw_turtle", ""))
    elements["raw_turtle"].update()
    card.set_visibility(True)


def _phase_5_1_waiting_view(
    message: str = "Complete ProductAgent grounding to select a RobotAgent.",
) -> dict[str, object]:
    """Return the empty read-only Phase 5.1 card state."""
    return {
        "status": "waiting_for_phase_4",
        "message": message,
        "product_requirement": None,
        "selected_resource_jid": None,
        "selected_execution_mode": None,
        "assignment_ref": None,
        "state_snapshot_count": 0,
        "latest_state_ref": None,
        "catalog_snapshot_count": 0,
        "latest_catalog_ref": None,
        "catalog_fingerprint": None,
        "robot_state": None,
        "primitive_count": 0,
        "primitive_symbols": [],
        "primitive_catalog": [],
        "failure": None,
    }


def _phase_5_1_status_color(status: str) -> str:
    """Return the diagnostic badge color for one exact Phase 5.1 status."""
    return {
        "context_captured": "green",
        "ready_for_assignment": "amber",
        "waiting_for_ra": "amber",
        "blocked": "red",
        "waiting_for_phase_4": "grey",
    }.get(status, "grey")


def _phase_5_1_action_state(
    status: str,
    *,
    activation_available: bool,
    activation_busy: bool,
) -> tuple[str, bool]:
    """Return the Phase 5.1 action label and enabled state."""
    if status == "waiting_for_ra":
        label = "Retry context capture"
    elif status == "context_captured":
        label = "Refresh RobotAgent context"
    else:
        label = "Capture RobotAgent context"
    enabled = (
        activation_available
        and not activation_busy
        and status
        in {
            "ready_for_assignment",
            "waiting_for_ra",
            "context_captured",
        }
    )
    return label, enabled


def _phase_5_2_waiting_view(
    message: str = "Capture one valid RobotAgent context first.",
) -> dict[str, object]:
    """Return the empty read-only Phase 5.2 card state."""
    return {
        "status": "waiting_for_context",
        "message": message,
        "draft_count": 0,
        "latest_draft_ref": None,
        "primitive_symbols": [],
        "unsupported_reason": None,
        "draft": None,
        "composition_input": None,
        "failure": None,
    }


def _phase_5_2_composition_evidence_summary(
    composition_input: object,
) -> dict[str, object]:
    """Summarize the exact composition input without changing its contents."""
    if not isinstance(composition_input, Mapping):
        return {
            "assertion_count": 0,
            "primitive_count": 0,
            "typed_record_count": 0,
            "tbox_fingerprint": None,
            "abox_fingerprint": None,
        }
    ontology_projection = composition_input.get("ontology_projection")
    projection = ontology_projection if isinstance(ontology_projection, Mapping) else {}
    assertions = projection.get("assertions")
    primitive_catalog = composition_input.get("primitive_catalog")
    grounded_context = composition_input.get("grounded_context")
    grounded = grounded_context if isinstance(grounded_context, Mapping) else {}
    typed_records = grounded.get("typed_records")
    return {
        "assertion_count": len(assertions) if isinstance(assertions, list) else 0,
        "primitive_count": (len(primitive_catalog) if isinstance(primitive_catalog, list) else 0),
        "typed_record_count": (len(typed_records) if isinstance(typed_records, list) else 0),
        "tbox_fingerprint": projection.get("tbox_fingerprint"),
        "abox_fingerprint": projection.get("abox_fingerprint"),
    }


def _phase_5_2_status_color(status: str) -> str:
    """Return the diagnostic badge color for one exact Phase 5.2 status."""
    return {
        "ready_for_draft": "amber",
        "draft_authored": "green",
        "unsupported": "amber",
        "blocked": "red",
        "waiting_for_context": "grey",
    }.get(status, "grey")


def _phase_5_2_action_enabled(
    status: str,
    *,
    authoring_available: bool,
    authoring_busy: bool,
) -> bool:
    """Return whether the structural-draft authoring action is available."""
    return authoring_available and not authoring_busy and status == "ready_for_draft"


def _render_phase_5_diagnostics() -> dict[str, Any]:
    """Render the temporary RobotAgent activation and diagnostics card."""
    with ui.card().classes("w-full border-2 border-violet-200 bg-violet-50 shadow-sm"):
        with ui.row().classes("w-full items-start justify-between gap-3 flex-wrap"):
            with ui.column().classes("gap-0"):
                ui.label("RobotAgent context and primitive draft").classes(
                    "text-xl font-semibold text-slate-900"
                )
                ui.label(
                    "Capture the assigned RobotAgent context and create a structural primitive draft."
                ).classes("text-xs text-slate-500")
            with ui.row().classes("items-center gap-2 flex-wrap"):
                start_phase_5_button = ui.button(
                    "Capture RobotAgent context",
                    icon="play_arrow",
                ).props("flat disable")
                refresh_button = ui.button("Refresh", icon="refresh").props("flat disable")

        with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap"):
            ui.label("Assigned RA activation and context snapshot").classes(
                "text-sm font-semibold text-violet-900"
            )
            with ui.row().classes("items-center gap-2 flex-wrap"):
                status_badge = ui.badge("Waiting for grounding").props("color=grey outline")
                ui.badge("contract-first").props("color=violet outline")
        message_value = ui.label("").classes(
            "text-sm text-slate-700 whitespace-pre-wrap break-words"
        )

        with ui.row().classes("w-full gap-3 items-stretch flex-wrap"):
            with ui.card().classes("flex-1 min-w-64 border border-slate-200 bg-white shadow-none"):
                ui.label("Selected assignment").classes("text-xs font-semibold text-slate-500")
                requirement_value = ui.label("not selected").classes(
                    "text-sm font-semibold text-slate-900 whitespace-pre-wrap"
                )
                resource_value = ui.label("RobotAgent: not selected").classes(
                    "text-xs text-slate-700 break-all"
                )
                assignment_value = ui.label("Assignment: not recorded").classes(
                    "text-xs text-slate-600 break-all"
                )
            with ui.card().classes("flex-1 min-w-64 border border-slate-200 bg-white shadow-none"):
                ui.label("Paired immutable snapshots").classes(
                    "text-xs font-semibold text-slate-500"
                )
                snapshot_counts_value = ui.label("State: 0 · catalog: 0").classes(
                    "text-sm font-semibold text-slate-900"
                )
                state_ref_value = ui.label("Latest state: none").classes(
                    "text-xs text-slate-600 break-all"
                )
                catalog_ref_value = ui.label("Latest catalog: none").classes(
                    "text-xs text-slate-600 break-all"
                )
            with ui.card().classes("flex-1 min-w-64 border border-slate-200 bg-white shadow-none"):
                ui.label("Complete primitive-only catalog").classes(
                    "text-xs font-semibold text-slate-500"
                )
                primitive_count_value = ui.label("Primitives: 0").classes(
                    "text-sm font-semibold text-slate-900"
                )
                primitive_symbols_container = ui.row().classes("w-full gap-1 flex-wrap")
                catalog_fingerprint_value = ui.label("Fingerprint: none").classes(
                    "text-xs text-slate-600 break-all"
                )

        with ui.expansion("Current robot_state", icon="smart_toy").classes(
            "w-full border border-slate-200 bg-white rounded"
        ) as robot_state_expansion:
            robot_state_value = ui.code("", language="json").classes(
                "w-full text-xs overflow-x-auto"
            )
        robot_state_expansion.set_visibility(False)

        with ui.expansion("Full primitive_catalog", icon="account_tree").classes(
            "w-full border border-slate-200 bg-white rounded"
        ) as primitive_catalog_expansion:
            primitive_catalog_value = ui.code("", language="json").classes(
                "w-full text-xs overflow-x-auto"
            )
        primitive_catalog_expansion.set_visibility(False)

        with ui.card().classes(
            "w-full border border-red-200 bg-red-50 shadow-none"
        ) as failure_card:
            ui.label("Fail-closed diagnostic").classes("text-sm font-semibold text-red-900")
            failure_value = ui.label("").classes(
                "text-xs text-red-800 whitespace-pre-wrap break-words"
            )
        failure_card.set_visibility(False)

        ui.separator().classes("my-1 bg-violet-200")

        with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap"):
            ui.label("RA-authored structural primitive draft").classes(
                "text-sm font-semibold text-violet-900"
            )
            with ui.row().classes("items-center gap-2 flex-wrap"):
                draft_status_badge = ui.badge("waiting_for_context").props("color=grey outline")
                create_draft_button = ui.button(
                    "Create Primitive Draft",
                    icon="account_tree",
                ).props("flat disable")
        draft_message_value = ui.label("").classes(
            "text-sm text-slate-700 whitespace-pre-wrap break-words"
        )

        with ui.card().classes("w-full border border-slate-200 bg-white shadow-none"):
            draft_count_value = ui.label("Drafts: 0").classes(
                "text-sm font-semibold text-slate-900"
            )
            latest_draft_ref_value = ui.label("Latest draft: none").classes(
                "text-xs text-slate-600 break-all"
            )
            draft_symbols_container = ui.row().classes("w-full gap-1 flex-wrap")
            unsupported_reason_value = ui.label("").classes(
                "text-xs text-amber-800 whitespace-pre-wrap break-words"
            )
            unsupported_reason_value.set_visibility(False)

        with ui.card().classes(
            "w-full border border-violet-200 bg-white shadow-none"
        ) as composition_evidence_card:
            ui.label("RA composition evidence").classes("text-sm font-semibold text-violet-900")
            ui.label(
                "Reconstructed from the current draft's hash-pinned inputs. "
                "This is input provenance, not private model reasoning, feasibility "
                "validation, or execution evidence."
            ).classes("text-xs text-slate-600 whitespace-pre-wrap")
            ui.label("Input sections").classes("text-xs font-semibold text-slate-500")
            with ui.row().classes("w-full gap-1 flex-wrap"):
                for section in (
                    "target_feature",
                    "selected_resource",
                    "ontology_projection",
                    "robot_state",
                    "primitive_catalog",
                    "grounded_context",
                ):
                    ui.badge(section).props("color=violet outline")
            with ui.row().classes("w-full gap-4 items-start flex-wrap"):
                composition_assertion_count_value = ui.label("Assertions: 0").classes(
                    "text-xs font-semibold text-slate-800"
                )
                composition_primitive_count_value = ui.label("Primitives: 0").classes(
                    "text-xs font-semibold text-slate-800"
                )
                composition_typed_record_count_value = ui.label(
                    "Typed-record identities: 0"
                ).classes("text-xs font-semibold text-slate-800")
            composition_tbox_fingerprint_value = ui.label("TBox: none").classes(
                "text-xs text-slate-600 break-all"
            )
            composition_abox_fingerprint_value = ui.label("ABox: none").classes(
                "text-xs text-slate-600 break-all"
            )
            ui.label(
                "target_feature includes the PA-authored desired state and bounded "
                "resolved state-value projections. grounded_context otherwise contains "
                "typed-record identities only. Excluded: raw RDF, unrelated typed-record "
                "payloads, unrelated ProductContextView fields, parameter bindings, "
                "task tools, private model reasoning, feasibility claims, and execution "
                "evidence."
            ).classes("text-xs text-slate-500 whitespace-pre-wrap")
            with ui.expansion(
                "COMPOSITION_INPUT delivered to RA",
                icon="fact_check",
            ).classes(
                "w-full border border-slate-200 bg-slate-50 rounded"
            ) as composition_input_expansion:
                composition_input_value = ui.code("", language="json").classes(
                    "w-full text-xs overflow-x-auto"
                )
        composition_evidence_card.set_visibility(False)

        with ui.expansion("PrimitiveProgramDraft", icon="schema").classes(
            "w-full border border-slate-200 bg-white rounded"
        ) as draft_expansion:
            draft_value = ui.code("", language="json").classes("w-full text-xs overflow-x-auto")
        draft_expansion.set_visibility(False)

        with ui.card().classes(
            "w-full border border-red-200 bg-red-50 shadow-none"
        ) as draft_failure_card:
            ui.label("Fail-closed draft diagnostic").classes("text-sm font-semibold text-red-900")
            draft_failure_value = ui.label("").classes(
                "text-xs text-red-800 whitespace-pre-wrap break-words"
            )
        draft_failure_card.set_visibility(False)

    return {
        "status_badge": status_badge,
        "start_button": start_phase_5_button,
        "refresh_button": refresh_button,
        "message": message_value,
        "requirement": requirement_value,
        "resource": resource_value,
        "assignment": assignment_value,
        "snapshot_counts": snapshot_counts_value,
        "state_ref": state_ref_value,
        "catalog_ref": catalog_ref_value,
        "primitive_count": primitive_count_value,
        "primitive_symbols": primitive_symbols_container,
        "catalog_fingerprint": catalog_fingerprint_value,
        "robot_state_expansion": robot_state_expansion,
        "robot_state": robot_state_value,
        "primitive_catalog_expansion": primitive_catalog_expansion,
        "primitive_catalog": primitive_catalog_value,
        "failure_card": failure_card,
        "failure": failure_value,
        "draft_status_badge": draft_status_badge,
        "create_draft_button": create_draft_button,
        "draft_message": draft_message_value,
        "draft_count": draft_count_value,
        "latest_draft_ref": latest_draft_ref_value,
        "draft_symbols": draft_symbols_container,
        "unsupported_reason": unsupported_reason_value,
        "composition_evidence_card": composition_evidence_card,
        "composition_assertion_count": composition_assertion_count_value,
        "composition_primitive_count": composition_primitive_count_value,
        "composition_typed_record_count": composition_typed_record_count_value,
        "composition_tbox_fingerprint": composition_tbox_fingerprint_value,
        "composition_abox_fingerprint": composition_abox_fingerprint_value,
        "composition_input_expansion": composition_input_expansion,
        "composition_input": composition_input_value,
        "draft_expansion": draft_expansion,
        "draft": draft_value,
        "draft_failure_card": draft_failure_card,
        "draft_failure": draft_failure_value,
    }


def _apply_phase_5_1_diagnostic(
    elements: Mapping[str, Any],
    diagnostic: Mapping[str, object],
    *,
    activation_available: bool = False,
    activation_busy: bool = False,
) -> None:
    """Apply one JSON-safe persisted Phase 5.1 diagnostic to the UI card."""
    status = str(diagnostic.get("status", "waiting_for_phase_4"))
    elements["status_badge"].set_text(
        "Waiting for grounding" if status == "waiting_for_phase_4" else status
    )
    elements["status_badge"].props(f"color={_phase_5_1_status_color(status)} outline")
    elements["message"].set_text(str(diagnostic.get("message", "")))

    product_requirement = diagnostic.get("product_requirement")
    selected_resource_jid = diagnostic.get("selected_resource_jid")
    selected_execution_mode = diagnostic.get("selected_execution_mode")
    assignment_ref = diagnostic.get("assignment_ref")
    elements["requirement"].set_text(
        str(product_requirement) if product_requirement else "not selected"
    )
    resource_parts = [
        str(value)
        for value in (selected_resource_jid, selected_execution_mode)
        if value not in {None, ""}
    ]
    elements["resource"].set_text(
        f"RobotAgent: {' · '.join(resource_parts)}"
        if resource_parts
        else "RobotAgent: not selected"
    )
    elements["assignment"].set_text(
        f"Assignment: {assignment_ref}" if assignment_ref else "Assignment: not recorded"
    )

    state_count = diagnostic.get("state_snapshot_count", 0)
    catalog_count = diagnostic.get("catalog_snapshot_count", 0)
    elements["snapshot_counts"].set_text(f"State: {state_count} · catalog: {catalog_count}")
    elements["state_ref"].set_text(f"Latest state: {diagnostic.get('latest_state_ref') or 'none'}")
    elements["catalog_ref"].set_text(
        f"Latest catalog: {diagnostic.get('latest_catalog_ref') or 'none'}"
    )

    primitive_count = diagnostic.get("primitive_count", 0)
    elements["primitive_count"].set_text(f"Primitives: {primitive_count}")
    primitive_symbols = diagnostic.get("primitive_symbols")
    symbols = primitive_symbols if isinstance(primitive_symbols, list) else []
    elements["primitive_symbols"].clear()
    with elements["primitive_symbols"]:
        for symbol in symbols:
            ui.badge(str(symbol)).props("color=violet outline")
    elements["catalog_fingerprint"].set_text(
        f"Fingerprint: {diagnostic.get('catalog_fingerprint') or 'none'}"
    )

    robot_state = diagnostic.get("robot_state")
    has_robot_state = isinstance(robot_state, Mapping)
    elements["robot_state"].content = (
        json.dumps(robot_state, indent=2, ensure_ascii=False, allow_nan=False)
        if has_robot_state
        else ""
    )
    elements["robot_state"].update()
    elements["robot_state_expansion"].set_visibility(has_robot_state)

    primitive_catalog = diagnostic.get("primitive_catalog")
    has_primitive_catalog = isinstance(primitive_catalog, list) and bool(primitive_catalog)
    elements["primitive_catalog"].content = (
        json.dumps(
            primitive_catalog,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        if has_primitive_catalog
        else ""
    )
    elements["primitive_catalog"].update()
    elements["primitive_catalog_expansion"].set_visibility(has_primitive_catalog)

    failure = diagnostic.get("failure")
    elements["failure"].set_text(str(failure or ""))
    elements["failure_card"].set_visibility(bool(failure))
    action_label, action_enabled = _phase_5_1_action_state(
        status,
        activation_available=activation_available,
        activation_busy=activation_busy,
    )
    elements["start_button"].set_text(action_label)
    _set_enabled(elements["start_button"], action_enabled)


def _apply_phase_5_2_diagnostic(
    elements: Mapping[str, Any],
    diagnostic: Mapping[str, object],
    *,
    authoring_available: bool = False,
    authoring_busy: bool = False,
) -> None:
    """Apply one JSON-safe persisted Phase 5.2 diagnostic to the UI card."""
    status = str(diagnostic.get("status", "waiting_for_context"))
    elements["draft_status_badge"].set_text(status)
    elements["draft_status_badge"].props(f"color={_phase_5_2_status_color(status)} outline")
    elements["draft_message"].set_text(str(diagnostic.get("message", "")))
    elements["draft_count"].set_text(f"Drafts: {diagnostic.get('draft_count', 0)}")
    elements["latest_draft_ref"].set_text(
        f"Latest draft: {diagnostic.get('latest_draft_ref') or 'none'}"
    )

    primitive_symbols = diagnostic.get("primitive_symbols")
    symbols = primitive_symbols if isinstance(primitive_symbols, list) else []
    elements["draft_symbols"].clear()
    with elements["draft_symbols"]:
        for index, symbol in enumerate(symbols, start=1):
            ui.badge(f"{index}. {symbol}").props("color=violet outline")

    unsupported_reason = diagnostic.get("unsupported_reason")
    elements["unsupported_reason"].set_text(
        f"Unsupported: {unsupported_reason}" if unsupported_reason else ""
    )
    elements["unsupported_reason"].set_visibility(bool(unsupported_reason))

    composition_input = diagnostic.get("composition_input")
    has_composition_input = status in {"draft_authored", "unsupported"} and isinstance(
        composition_input, Mapping
    )
    evidence_summary = _phase_5_2_composition_evidence_summary(composition_input)
    elements["composition_assertion_count"].set_text(
        f"Assertions: {evidence_summary['assertion_count']}"
    )
    elements["composition_primitive_count"].set_text(
        f"Primitives: {evidence_summary['primitive_count']}"
    )
    elements["composition_typed_record_count"].set_text(
        f"Typed-record identities: {evidence_summary['typed_record_count']}"
    )
    elements["composition_tbox_fingerprint"].set_text(
        f"TBox: {evidence_summary['tbox_fingerprint'] or 'none'}"
    )
    elements["composition_abox_fingerprint"].set_text(
        f"ABox: {evidence_summary['abox_fingerprint'] or 'none'}"
    )
    elements["composition_input"].content = (
        json.dumps(
            composition_input,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        if has_composition_input
        else ""
    )
    elements["composition_input"].update()
    elements["composition_input_expansion"].set_visibility(has_composition_input)
    elements["composition_evidence_card"].set_visibility(has_composition_input)

    draft = diagnostic.get("draft")
    has_draft = isinstance(draft, Mapping)
    elements["draft"].content = (
        json.dumps(draft, indent=2, ensure_ascii=False, allow_nan=False) if has_draft else ""
    )
    elements["draft"].update()
    elements["draft_expansion"].set_visibility(has_draft)

    failure = diagnostic.get("failure")
    elements["draft_failure"].set_text(str(failure or ""))
    elements["draft_failure_card"].set_visibility(bool(failure))
    _set_enabled(
        elements["create_draft_button"],
        _phase_5_2_action_enabled(
            status,
            authoring_available=authoring_available,
            authoring_busy=authoring_busy,
        ),
    )


def _calibration_readiness(
    runtime: Spec2PrimitivesUIRuntime,
) -> tuple[str, str, str]:
    """Return operator-facing fixed-camera calibration readiness."""
    if runtime.camera_to_world_calibration_runtime is not None:
        return (
            "calibration ready",
            "green",
            "The runtime-selected camera frame will use its matching approved "
            "camera-to-world calibration.",
        )
    reason = runtime.camera_to_world_calibration_unavailable_reason or (
        "Approved camera-to-world calibration is unavailable."
    )
    return "calibration unavailable", "amber", reason


def _render_pa_interaction(  # noqa: C901, PLR0915
    runtime: Spec2PrimitivesUIRuntime,
) -> None:
    """Render the streamlined ProductAgent grounding interaction."""
    phase_5_1_activation_available = runtime.robot_agent_context_runtime is not None
    phase_5_2_authoring_available = runtime.robot_agent_draft_runtime is not None
    grounding_ready = runtime.ontology_config is not None and runtime.grounding_runtime is not None
    grounding_unavailable_reason = (
        runtime.document_diagnostic_unavailable_reason
        or "Authoritative TBox and controlled ProductAgent grounding runtime are unavailable."
    )

    with ui.card().classes("w-full border border-slate-200 shadow-sm") as interaction_card:
        with ui.row().classes("w-full items-center justify-between gap-3 flex-wrap"):
            with ui.column().classes("gap-0"):
                ui.label("ProductAgent Grounding").classes("text-lg font-semibold text-slate-900")
                ui.label("Requirement to validated product context").classes(
                    "text-xs text-slate-500"
                )
            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.badge("grounding ready" if grounding_ready else "grounding unavailable").props(
                    f"color={'green' if grounding_ready else 'amber'} outline"
                )
                calibration_state, calibration_color, calibration_message = _calibration_readiness(
                    runtime
                )
                ui.badge(calibration_state).props(f"color={calibration_color} outline")

        if not grounding_ready:
            ui.label(grounding_unavailable_reason).classes("text-sm text-amber-700")

        with ui.row().classes("w-full items-end gap-3 flex-wrap"):
            requirement_input = (
                ui.input(
                    label="product_requirement",
                    value="",
                    placeholder="Describe the assembly requirement",
                )
                .props("outlined")
                .classes("flex-1 min-w-72")
            )
            start_button = ui.button(
                "Start ProductAgent",
                icon="play_arrow",
            ).props("disable")

        with ui.row().classes(
            "w-full items-center gap-3 rounded border border-indigo-100 "
            "bg-indigo-50 px-3 py-2 flex-wrap"
        ) as activity_strip:
            ui.label("ProductAgent").classes("text-sm font-semibold text-indigo-900")
            activity_badge = ui.badge("idle" if grounding_ready else "grounding unavailable").props(
                f"color={'indigo' if grounding_ready else 'amber'} outline"
            )
            activity_message = ui.label(
                "Ready for a product_requirement."
                if grounding_ready
                else grounding_unavailable_reason
            ).classes(
                "flex-1 min-w-64 text-xs text-indigo-700"
                if grounding_ready
                else "flex-1 min-w-64 text-xs text-amber-700"
            )
        activity_strip.set_visibility(False)

        with ui.card().classes(
            "w-full border border-indigo-100 bg-white shadow-none"
        ) as timeline_card:
            with ui.row().classes("w-full items-center justify-between gap-2"):
                ui.label("ProductAgent Timeline").classes("text-base font-semibold text-slate-900")
                timeline_badge = ui.badge("live").props("color=indigo outline")
            timeline_container = ui.column().classes("w-full gap-3")
        timeline_card.set_visibility(False)

        with ui.card().classes(
            "w-full border border-amber-200 bg-amber-50 shadow-none"
        ) as clarification_card:
            ui.label("ProductAgent clarification").classes("text-sm font-semibold text-amber-900")
            clarification_prompt = ui.label("").classes("text-xs text-amber-800")
            clarification_reply_input = (
                ui.input(label="User reply").props("outlined").classes("w-full")
            )
            with ui.row().classes("items-center gap-2"):
                submit_reply_button = ui.button("Submit Reply", icon="send").props("disable")
                cancel_interaction_button = ui.button("Cancel Interaction", icon="cancel").props(
                    "outline disable"
                )
        clarification_card.set_visibility(False)

        with ui.card().classes(
            "w-full border border-red-200 bg-red-50 shadow-none"
        ) as outcome_alert_card:
            outcome_alert_title = ui.label("Grounding did not complete").classes(
                "text-sm font-semibold text-red-900"
            )
            outcome_alert_value = ui.label("").classes(
                "text-xs text-red-800 whitespace-pre-wrap break-words"
            )
        outcome_alert_card.set_visibility(False)

        final_result_elements = _render_final_grounding_result()
        phase_5_elements = _render_phase_5_diagnostics()
        _apply_phase_5_1_diagnostic(
            phase_5_elements,
            _phase_5_1_waiting_view(),
            activation_available=phase_5_1_activation_available,
        )
        _apply_phase_5_2_diagnostic(
            phase_5_elements,
            _phase_5_2_waiting_view(),
            authoring_available=phase_5_2_authoring_available,
        )

        with ui.expansion("Developer diagnostics", icon="terminal", value=False).classes(
            "w-full border border-slate-200 rounded"
        ):
            model = (
                runtime.model_config.product_agent_llm.model
                if runtime.model_config is not None
                else "unconfigured"
            )
            ui.label(
                "Runtime: "
                f"ProductAgent model={model} · "
                f"grounding={'ready' if grounding_ready else 'unavailable'}"
            ).classes("text-xs text-slate-600")
            ui.label(f"Calibration: {calibration_message}").classes(
                "text-xs text-emerald-700"
                if calibration_state == "calibration ready"
                else "text-xs text-amber-700"
            )
            diagnostic_identifier = ui.label("Interaction: none").classes("text-xs text-slate-600")
            diagnostic_path = ui.label("Persisted path: none").classes(
                "text-xs text-slate-600 break-all"
            )
            diagnostic_counts = ui.label("Turns: 0 · evidence: 0 · decisions: 0").classes(
                "text-xs text-slate-600"
            )
            with ui.card().classes(
                "w-full border border-red-200 bg-red-50 shadow-none"
            ) as diagnostic_failure_card:
                ui.label("Failure details").classes("text-sm font-semibold text-red-900")
                diagnostic_failure_value = ui.label("").classes(
                    "text-xs text-red-800 whitespace-pre-wrap break-words"
                )
            diagnostic_failure_card.set_visibility(False)

        recovered_label = ui.label("Recovered latest interaction").classes(
            "text-xs font-medium text-indigo-700"
        )
        recovered_label.set_visibility(False)

        action_state: dict[str, object] = {
            "busy": False,
            "pending_clarification": False,
            "phase_5_1_activating": False,
            "phase_5_1_refreshing": False,
            "phase_5_2_authoring": False,
            "interaction": None,
            "timeline": [],
        }

        def _update_start_enabled() -> None:
            if interaction_card.is_deleted:
                return
            value = requirement_input.value
            _set_enabled(
                start_button,
                grounding_ready
                and not action_state["busy"]
                and not action_state["pending_clarification"]
                and isinstance(value, str)
                and bool(value.strip()),
            )

        def _apply_pa_view(
            interaction: dict[str, object],
            view: dict[str, object],
        ) -> None:
            action_state["interaction"] = interaction
            activity_strip.set_visibility(True)
            pending = view["activity_state"] == "clarification needed"
            action_state["pending_clarification"] = pending
            activity_badge.set_text(str(view["activity_state"]))
            activity_badge.props(f"color={view['activity_color']}")
            activity_message.set_text(str(view["activity_message"]))

            timeline = view.get("timeline")
            persisted_events = (
                [event for event in timeline if isinstance(event, dict)]
                if isinstance(timeline, list)
                else []
            )
            action_state["timeline"] = persisted_events
            _render_pa_timeline_events(timeline_container, persisted_events)
            timeline_card.set_visibility(bool(persisted_events))
            timeline_state = str(view["activity_state"])
            timeline_colors = {
                "grounding complete": "green",
                "clarification needed": "amber",
                "grounding waiting": "amber",
                "grounding incomplete": "amber",
                "failed": "red",
                "grounding unavailable": "red",
                "cancelled": "grey",
                "stopped": "grey",
            }
            recovered = interaction.get("recovered") is True
            timeline_badge.set_text(
                f"recovered · {timeline_state}" if recovered else timeline_state
            )
            timeline_badge.props(f"color={timeline_colors.get(timeline_state, 'indigo')} outline")
            recovered_label.set_visibility(recovered)
            result = view.get("final_result")
            _apply_final_grounding_result(
                final_result_elements,
                result if isinstance(result, Mapping) else None,
            )
            phase_5_1 = view.get("phase_5_1")
            _apply_phase_5_1_diagnostic(
                phase_5_elements,
                phase_5_1 if isinstance(phase_5_1, Mapping) else _phase_5_1_waiting_view(),
                activation_available=phase_5_1_activation_available,
                activation_busy=bool(action_state["phase_5_1_activating"]),
            )
            phase_5_2 = view.get("phase_5_2")
            _apply_phase_5_2_diagnostic(
                phase_5_elements,
                phase_5_2 if isinstance(phase_5_2, Mapping) else _phase_5_2_waiting_view(),
                authoring_available=phase_5_2_authoring_available,
                authoring_busy=bool(action_state["phase_5_2_authoring"]),
            )
            _set_enabled(phase_5_elements["refresh_button"], True)

            diagnostics = view.get("diagnostics")
            diagnostic_values = diagnostics if isinstance(diagnostics, Mapping) else {}
            diagnostic_identifier.set_text(
                f"Interaction: {diagnostic_values.get('interaction_identifier', 'none')}"
            )
            diagnostic_path.set_text(
                f"Persisted path: {diagnostic_values.get('interaction_path', 'none')}"
            )
            diagnostic_counts.set_text(
                f"Turns: {diagnostic_values.get('turn_count', 0)} · "
                f"evidence: {diagnostic_values.get('evidence_count', 0)} · "
                f"decisions: {diagnostic_values.get('decision_count', 0)}"
            )
            failure_text = str(diagnostic_values.get("failure", ""))
            diagnostic_failure_value.set_text(failure_text)
            diagnostic_failure_card.set_visibility(bool(failure_text))

            activity_state = str(view["activity_state"])
            terminal = activity_state in {
                "cancelled",
                "failed",
                "grounding incomplete",
                "grounding unavailable",
                "stopped",
            }
            outcome_alert_title.set_text(
                "Grounding incomplete"
                if activity_state == "grounding incomplete"
                else "Grounding did not complete"
            )
            outcome_alert_value.set_text(str(view["activity_message"]))
            outcome_alert_card.set_visibility(terminal)
            clarification_card.set_visibility(pending)
            if pending:
                clarification_prompt.set_text(str(view["clarification"]))
                clarification_reply_input.props(remove="disable")
                cancel_interaction_button.props(remove="disable")
            else:
                clarification_reply_input.value = ""
                clarification_reply_input.props("disable")
                submit_reply_button.props("disable")
                cancel_interaction_button.props("disable")

        def _show_pa_event(event: dict[str, str]) -> None:
            # Page lifetime must not determine whether the underlying PA decision completes.
            if interaction_card.is_deleted:
                return
            events = action_state["timeline"]
            if not isinstance(events, list):
                events = []
                action_state["timeline"] = events
            events.append(event)
            _render_pa_timeline_events(timeline_container, events)

        def _show_pa_stage(sentence: str) -> None:
            if interaction_card.is_deleted:
                return
            activity_badge.set_text("grounding")
            activity_badge.props("color=indigo")
            activity_message.set_text(sentence)

        async def _start_pa_interaction() -> None:
            if interaction_card.is_deleted:
                return
            value = requirement_input.value
            if (
                not grounding_ready
                or action_state["busy"]
                or not isinstance(value, str)
                or not value.strip()
            ):
                return
            action_state["busy"] = True
            action_state["pending_clarification"] = False
            action_state["interaction"] = None
            _set_enabled(start_button, False)
            requirement_input.props("disable")
            clarification_card.set_visibility(False)
            outcome_alert_card.set_visibility(False)
            diagnostic_failure_card.set_visibility(False)
            recovered_label.set_visibility(False)
            _apply_final_grounding_result(final_result_elements, None)
            _apply_phase_5_1_diagnostic(
                phase_5_elements,
                _phase_5_1_waiting_view(
                    "ProductAgent grounding is running; RobotAgent context capture is unavailable."
                ),
                activation_available=phase_5_1_activation_available,
            )
            _apply_phase_5_2_diagnostic(
                phase_5_elements,
                _phase_5_2_waiting_view(
                    "ProductAgent grounding is running; structural primitive drafting is unavailable."
                ),
                authoring_available=phase_5_2_authoring_available,
            )
            _set_enabled(phase_5_elements["refresh_button"], False)
            activity_strip.set_visibility(True)
            activity_badge.set_text("grounding")
            activity_badge.props("color=indigo")
            activity_message.set_text("Investigating approved evidence.")
            live_events = [
                _timeline_event("accepted", "Requirement received", value),
            ]
            action_state["timeline"] = live_events
            _render_pa_timeline_events(timeline_container, live_events)
            timeline_card.set_visibility(True)
            timeline_badge.set_text("live")
            timeline_badge.props("color=indigo outline")
            diagnostic_identifier.set_text("Interaction: pending")
            diagnostic_path.set_text("Persisted path: pending")
            diagnostic_counts.set_text("Turns: 0 · evidence: 0 · decisions: 0")

            try:
                interaction = await _run_pa_ui_interaction(
                    runtime,
                    value,
                    on_pa_stage=_show_pa_stage,
                    on_pa_event=_show_pa_event,
                    is_attached=lambda: not interaction_card.is_deleted,
                )
                if interaction_card.is_deleted:
                    return
                view = _pa_ui_view(interaction)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                if interaction_card.is_deleted:
                    raise
                failure_message = f"Connected PA workflow failed: {type(exc).__name__}: {exc}"
                activity_badge.set_text("failed")
                activity_badge.props("color=red")
                timeline_badge.set_text("failed")
                timeline_badge.props("color=red outline")
                activity_message.set_text(failure_message)
                outcome_alert_title.set_text("Grounding did not complete")
                outcome_alert_value.set_text(failure_message)
                outcome_alert_card.set_visibility(True)
                diagnostic_failure_value.set_text(failure_message)
                diagnostic_failure_card.set_visibility(True)
                _show_pa_event(_timeline_event("failed", "Grounding failed", failure_message))
                ui.notify("Connected PA workflow failed.", type="negative")
            else:
                _apply_pa_view(interaction, view)
                notification_type = (
                    "positive" if str(view["activity_state"]) == "grounding complete" else "warning"
                )
                ui.notify(str(view["activity_state"]), type=notification_type)
            finally:
                action_state["busy"] = False
                if not interaction_card.is_deleted:
                    if action_state["pending_clarification"]:
                        requirement_input.props("disable")
                    else:
                        requirement_input.props(remove="disable")
                    _update_start_enabled()

        def _update_reply_enabled() -> None:
            if interaction_card.is_deleted:
                return
            value = clarification_reply_input.value
            _set_enabled(
                submit_reply_button,
                not action_state["busy"]
                and bool(action_state["pending_clarification"])
                and isinstance(value, str)
                and bool(value.strip()),
            )

        async def _submit_clarification() -> None:
            if interaction_card.is_deleted:
                return
            interaction = action_state["interaction"]
            reply = clarification_reply_input.value
            if (
                action_state["busy"]
                or not action_state["pending_clarification"]
                or not isinstance(interaction, dict)
                or not isinstance(reply, str)
                or not reply.strip()
            ):
                return
            action_state["busy"] = True
            submit_reply_button.props("disable")
            cancel_interaction_button.props("disable")
            activity_badge.set_text("resuming")
            activity_badge.props("color=indigo")
            activity_message.set_text("Continuing the evidence investigation.")
            timeline_badge.set_text("live")
            timeline_badge.props("color=indigo outline")
            recovered_label.set_visibility(False)

            _show_pa_event(_timeline_event("accepted", "Clarification submitted", reply))

            try:
                interaction = await _submit_pa_ui_clarification(
                    runtime,
                    interaction,
                    reply,
                    on_pa_stage=_show_pa_stage,
                    on_pa_event=_show_pa_event,
                    is_attached=lambda: not interaction_card.is_deleted,
                )
                if interaction_card.is_deleted:
                    return
                view = _pa_ui_view(interaction)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                if interaction_card.is_deleted:
                    raise
                failure_message = f"Clarification failed: {type(exc).__name__}: {exc}"
                activity_badge.set_text("failed")
                activity_badge.props("color=red")
                timeline_badge.set_text("failed")
                timeline_badge.props("color=red outline")
                activity_message.set_text(failure_message)
                outcome_alert_title.set_text("Grounding did not complete")
                outcome_alert_value.set_text(failure_message)
                outcome_alert_card.set_visibility(True)
                diagnostic_failure_value.set_text(failure_message)
                diagnostic_failure_card.set_visibility(True)
                _show_pa_event(_timeline_event("failed", "Clarification failed", failure_message))
                ui.notify("Clarification failed.", type="negative")
            else:
                _apply_pa_view(interaction, view)
                ui.notify(str(view["activity_state"]), type="positive")
            finally:
                action_state["busy"] = False
                if not interaction_card.is_deleted:
                    _update_reply_enabled()
                    if not action_state["pending_clarification"]:
                        requirement_input.props(remove="disable")
                    _update_start_enabled()

        def _cancel_clarification() -> None:
            interaction = action_state["interaction"]
            if (
                action_state["busy"]
                or not action_state["pending_clarification"]
                or not isinstance(interaction, dict)
            ):
                return
            action_state["busy"] = True
            try:
                interaction = _cancel_pa_ui_interaction(interaction)
                view = _pa_ui_view(interaction)
                _apply_pa_view(interaction, view)
                ui.notify("cancelled", type="info")
            finally:
                action_state["busy"] = False
                requirement_input.props(remove="disable")
                _update_start_enabled()

        async def _start_phase_5_1() -> None:
            interaction = action_state["interaction"]
            phase_5_runtime = runtime.robot_agent_context_runtime
            if (
                action_state["phase_5_1_activating"]
                or action_state["phase_5_1_refreshing"]
                or action_state["phase_5_2_authoring"]
                or not isinstance(interaction, dict)
                or phase_5_runtime is None
            ):
                return
            interaction_root = interaction.get("interaction_root")
            if not isinstance(interaction_root, Path):
                return
            current_diagnostic = read_phase_5_1_diagnostic(interaction_root)
            if current_diagnostic.status not in {
                "ready_for_assignment",
                "waiting_for_ra",
                "context_captured",
            }:
                _apply_phase_5_1_diagnostic(
                    phase_5_elements,
                    current_diagnostic.to_view(),
                    activation_available=phase_5_1_activation_available,
                )
                return

            action_state["phase_5_1_activating"] = True
            _apply_phase_5_1_diagnostic(
                phase_5_elements,
                current_diagnostic.to_view(),
                activation_available=phase_5_1_activation_available,
                activation_busy=True,
            )
            phase_5_elements["message"].set_text(
                "Starting or reusing only the exact RobotAgent selected by ProductAgent."
            )
            _set_enabled(phase_5_elements["refresh_button"], False)
            _apply_phase_5_2_diagnostic(
                phase_5_elements,
                read_phase_5_2_diagnostic(interaction_root).to_view(),
                authoring_available=phase_5_2_authoring_available,
                authoring_busy=True,
            )
            diagnostic_view = current_diagnostic.to_view()
            try:
                await activate_selected_ra_context(
                    phase_5_runtime,
                    interaction_root,
                )
            except (
                OSError,
                RAContextHandoffError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                diagnostic_view = read_phase_5_1_diagnostic(interaction_root).to_view()
                diagnostic_view["failure"] = f"{type(exc).__name__}: {exc}"
                ui.notify("RobotAgent context capture failed closed.", type="negative")
            else:
                diagnostic_view = read_phase_5_1_diagnostic(interaction_root).to_view()
                ui.notify("RobotAgent context captured.", type="positive")
            finally:
                action_state["phase_5_1_activating"] = False
                _apply_phase_5_1_diagnostic(
                    phase_5_elements,
                    diagnostic_view,
                    activation_available=phase_5_1_activation_available,
                )
                _apply_phase_5_2_diagnostic(
                    phase_5_elements,
                    read_phase_5_2_diagnostic(interaction_root).to_view(),
                    authoring_available=phase_5_2_authoring_available,
                )
                _set_enabled(phase_5_elements["refresh_button"], True)

        async def _refresh_phase_5_1() -> None:
            interaction = action_state["interaction"]
            if (
                action_state["phase_5_1_activating"]
                or action_state["phase_5_1_refreshing"]
                or action_state["phase_5_2_authoring"]
                or not isinstance(interaction, dict)
            ):
                return
            interaction_root = interaction.get("interaction_root")
            if not isinstance(interaction_root, Path):
                return
            action_state["phase_5_1_refreshing"] = True
            _set_enabled(phase_5_elements["refresh_button"], False)
            try:
                diagnostic = await asyncio.to_thread(
                    read_phase_5_1_diagnostic,
                    interaction_root,
                )
                _apply_phase_5_1_diagnostic(
                    phase_5_elements,
                    diagnostic.to_view(),
                    activation_available=phase_5_1_activation_available,
                )
                _apply_phase_5_2_diagnostic(
                    phase_5_elements,
                    read_phase_5_2_diagnostic(interaction_root).to_view(),
                    authoring_available=phase_5_2_authoring_available,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                failure_view = _phase_5_1_waiting_view(
                    "Persisted RobotAgent context evidence could not be inspected."
                )
                failure_view["status"] = "blocked"
                failure_view["failure"] = f"{type(exc).__name__}: {exc}"
                _apply_phase_5_1_diagnostic(
                    phase_5_elements,
                    failure_view,
                    activation_available=phase_5_1_activation_available,
                )
                _apply_phase_5_2_diagnostic(
                    phase_5_elements,
                    _phase_5_2_waiting_view(
                        "Persisted RobotAgent evidence could not be inspected."
                    ),
                    authoring_available=phase_5_2_authoring_available,
                )
            finally:
                action_state["phase_5_1_refreshing"] = False
                _set_enabled(phase_5_elements["refresh_button"], True)

        async def _start_phase_5_2() -> None:
            interaction = action_state["interaction"]
            draft_runtime = runtime.robot_agent_draft_runtime
            if (
                action_state["phase_5_1_activating"]
                or action_state["phase_5_1_refreshing"]
                or action_state["phase_5_2_authoring"]
                or not isinstance(interaction, dict)
                or draft_runtime is None
            ):
                return
            interaction_root = interaction.get("interaction_root")
            if not isinstance(interaction_root, Path):
                return
            current_diagnostic = read_phase_5_2_diagnostic(interaction_root)
            if current_diagnostic.status != "ready_for_draft":
                _apply_phase_5_2_diagnostic(
                    phase_5_elements,
                    current_diagnostic.to_view(),
                    authoring_available=phase_5_2_authoring_available,
                )
                return

            action_state["phase_5_2_authoring"] = True
            _apply_phase_5_2_diagnostic(
                phase_5_elements,
                current_diagnostic.to_view(),
                authoring_available=phase_5_2_authoring_available,
                authoring_busy=True,
            )
            phase_5_elements["draft_message"].set_text(
                "The exact selected RobotAgent is authoring an unbound structural sequence."
            )
            _apply_phase_5_1_diagnostic(
                phase_5_elements,
                read_phase_5_1_diagnostic(interaction_root).to_view(),
                activation_available=phase_5_1_activation_available,
                activation_busy=True,
            )
            _set_enabled(phase_5_elements["refresh_button"], False)
            diagnostic_view = current_diagnostic.to_view()
            try:
                await author_primitive_program_draft(
                    draft_runtime,
                    interaction_root,
                )
            except (
                OSError,
                PrimitiveDraftError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                diagnostic_view = read_phase_5_2_diagnostic(interaction_root).to_view()
                diagnostic_view["failure"] = f"{type(exc).__name__}: {exc}"
                ui.notify("Primitive draft creation failed closed.", type="negative")
            else:
                diagnostic_view = read_phase_5_2_diagnostic(interaction_root).to_view()
                notification_type = (
                    "positive" if diagnostic_view["status"] == "draft_authored" else "warning"
                )
                ui.notify(str(diagnostic_view["status"]), type=notification_type)
            finally:
                action_state["phase_5_2_authoring"] = False
                _apply_phase_5_2_diagnostic(
                    phase_5_elements,
                    diagnostic_view,
                    authoring_available=phase_5_2_authoring_available,
                )
                _apply_phase_5_1_diagnostic(
                    phase_5_elements,
                    read_phase_5_1_diagnostic(interaction_root).to_view(),
                    activation_available=phase_5_1_activation_available,
                )
                _set_enabled(phase_5_elements["refresh_button"], True)

        requirement_input.on_value_change(lambda _: _update_start_enabled())
        clarification_reply_input.on_value_change(lambda _: _update_reply_enabled())
        start_button.on_click(_start_pa_interaction)
        submit_reply_button.on_click(_submit_clarification)
        cancel_interaction_button.on_click(_cancel_clarification)
        phase_5_elements["start_button"].on_click(_start_phase_5_1)
        phase_5_elements["create_draft_button"].on_click(_start_phase_5_2)
        phase_5_elements["refresh_button"].on_click(_refresh_phase_5_1)
        _update_start_enabled()

        try:
            recovered_interaction = _latest_pa_ui_interaction(runtime.contexts_root)
            recovered_view = (
                _pa_ui_view(recovered_interaction) if recovered_interaction is not None else None
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            recovered_interaction = None
            recovered_view = None
        if recovered_interaction is not None and recovered_view is not None:
            requirement_input.value = recovered_interaction["product_requirement"]
            _apply_pa_view(recovered_interaction, recovered_view)
            if action_state["pending_clarification"]:
                requirement_input.props("disable")
            _update_start_enabled()


def render(runtime: Spec2PrimitivesUIRuntime) -> None:
    """Render the ICRA-demo-oriented Spec2Primitives grounding page."""
    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-6"):
        with ui.row().classes("w-full items-start justify-between gap-4"):
            with ui.column().classes("gap-1"):
                ui.label("Spec2Primitives").classes("text-3xl font-bold text-slate-900")
                ui.label(
                    "A Multi-Agent Framework for Dynamic Primitive Composition in "
                    "Industrial Robotic Assembly"
                ).classes("text-lg text-slate-600")
            ui.badge("ICRA 2027 Case Study").props("color=indigo outline").classes(
                "text-sm px-3 py-2"
            )

        _render_dual_gazebo(runtime.dual_gazebo)
        _render_pa_interaction(runtime)

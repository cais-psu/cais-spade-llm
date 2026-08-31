"""NiceGUI page for the ICRA 2027 Spec2Primitives case study."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping
from html import escape
from pathlib import Path
from typing import Any

from nicegui import ui
from nicegui.elements.badge import Badge
from nicegui.elements.button import Button
from nicegui.elements.label import Label

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

_TURTLE_PREFIX_PATTERN = re.compile(
    r"^@prefix\s+([A-Za-z][A-Za-z0-9_-]*):\s+<([^>]+)>\s+\.\s*$"
)


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

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[
            [str, Mapping[str, object]], Awaitable[Mapping[str, object]]
        ]
        | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        """Report the response contract's semantic stage and validated result."""
        self._on_pa_stage(_live_pa_stage(response_format))
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
    phase_3_1 = await start_pa_context_interaction(
        product_agent,
        interaction_root,
        product_requirement,
        ontology_config=runtime.ontology_config,
        grounding_runtime=runtime.grounding_runtime,
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
    result = await submit_pa_clarification_reply(
        product_agent,
        interaction_root,
        user_reply,
        ontology_config=runtime.ontology_config,
        grounding_runtime=runtime.grounding_runtime,
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
        return "Investigating approved evidence and grounding the product context."
    return "Grounding the validated product context."


def _live_pa_response_event(
    response: Mapping[str, object],
    response_format: Mapping[str, object],
) -> dict[str, str] | None:
    """Translate one structured ProductAgent result without exposing raw payloads."""
    del response_format
    if isinstance(response.get("clarification_question"), str):
        return _timeline_event(
            "waiting",
            "Clarification requested",
            str(response["clarification_question"]),
        )
    if isinstance(response.get("insufficient_evidence"), str):
        return _timeline_event(
            "waiting",
            "Grounding incomplete",
            str(response["insufficient_evidence"]),
        )
    if isinstance(response.get("context_summary"), str):
        return _timeline_event(
            "running",
            "Context proposal returned",
            "The evidence-backed context proposal is being validated.",
        )
    return None


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
    return _read_json_object(
        _interaction_ref_path(root, str(paths[-1].relative_to(root)))
    )


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

    node_ids = {
        term: f"n{index}"
        for index, term in enumerate(sorted(node_kinds))
    }
    lines = [
        "flowchart LR",
        "    classDef iri fill:#eef2ff,stroke:#4f46e5,color:#1e1b4b",
        "    classDef literal fill:#f8fafc,stroke:#64748b,color:#0f172a",
    ]
    for term, node_id in node_ids.items():
        lines.append(
            f'    {node_id}["{_mermaid_text(term)}"]:::{node_kinds[term]}'
        )
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


def _unique_text(values: list[object]) -> list[str]:
    """Return non-empty display strings once in their persisted order."""
    result: list[str] = []
    for value in values:
        text = value if isinstance(value, str) else None
        if text and text not in result:
            result.append(text)
    return result


def _final_result_limitations(
    product_context: Mapping[str, object],
    contract: Mapping[str, object],
) -> list[str]:
    """Collect exact non-blocking limits for one validated completion."""
    missing_information = contract.get("missing_information")
    values: list[object] = (
        list(missing_information) if isinstance(missing_information, list) else []
    )
    uncertainty = product_context.get("uncertainty")
    if isinstance(uncertainty, list):
        values.extend(
            item.get("description") if isinstance(item, Mapping) else item
            for item in uncertainty
        )
    return _unique_text(values)


def _final_result_evidence(
    *,
    product_context: Mapping[str, object],
    contract: Mapping[str, object],
    session: Mapping[str, object] | None,
    selection: Mapping[str, object],
) -> list[dict[str, str]]:
    """Build compact evidence badges from validated final records."""
    evidence: list[dict[str, str]] = []

    def _add(label: object, status: object, kind: str) -> None:
        if not isinstance(label, str) or not label:
            return
        item = {"label": label, "status": str(status), "kind": kind}
        if item not in evidence:
            evidence.append(item)

    context_evidence_refs = contract.get("context_evidence_refs")
    if isinstance(context_evidence_refs, list):
        for ref in context_evidence_refs:
            _add(ref, "accepted", "source")
    attempts = session.get("attempted_actions") if session is not None else None
    if isinstance(attempts, list):
        for attempt in attempts:
            if isinstance(attempt, Mapping):
                _add(
                    attempt.get("source_ref"),
                    attempt.get("status", "recorded"),
                    "source",
                )
    typed_bindings = product_context.get("typed_bindings")
    if isinstance(typed_bindings, list):
        for binding in typed_bindings:
            if isinstance(binding, Mapping):
                _add(
                    binding.get("output_symbol"),
                    binding.get("status", "recorded"),
                    "typed record",
                )
    _add(
        "ResourceSelectionRecord",
        "accepted" if selection.get("selected_resource_symbol") else "unavailable",
        "decision",
    )
    return evidence


def _target_context_ref(
    pose_record: Mapping[str, object],
    pose_binding: Mapping[str, object],
    contract: Mapping[str, object],
) -> str | None:
    """Return the exact CAD reference associated with the final target."""
    CAD = pose_record.get("CAD")
    context_ref = CAD.get("context_ref") if isinstance(CAD, Mapping) else None
    if isinstance(context_ref, str):
        return context_ref
    evidence_refs = pose_binding.get("evidence_refs")
    if isinstance(evidence_refs, list):
        context_ref = next(
            (
                value
                for value in evidence_refs
                if isinstance(value, str) and value.endswith(".STL")
            ),
            None,
        )
    if isinstance(context_ref, str):
        return context_ref
    context_evidence_refs = contract.get("context_evidence_refs")
    if not isinstance(context_evidence_refs, list):
        return None
    return next(
        (
            value
            for value in context_evidence_refs
            if isinstance(value, str) and value.endswith(".STL")
        ),
        None,
    )


def _final_grounding_result(
    interaction_root: Path,
    completion: Mapping[str, object],
) -> dict[str, object]:
    """Build the operator result from one already validated completion bundle."""
    product_context = _latest_interaction_json_object(
        interaction_root,
        "products/grounding/product_context/view_*.json",
    )
    contract = _read_json_object(
        _interaction_ref_path(
            interaction_root,
            completion.get("typed_grounding_contract_ref"),
        )
    )
    selection = _read_json_object(
        _interaction_ref_path(
            interaction_root,
            completion.get("resource_selection_ref"),
        )
    )
    raw_turtle = _interaction_ref_path(
        interaction_root,
        "products/grounding/ontology/interaction_abox.ttl",
    ).read_text(encoding="utf-8")
    prefixes = _turtle_prefixes(raw_turtle)

    typed_bindings = product_context.get("typed_bindings")
    if not isinstance(typed_bindings, list):
        raise ValueError("Final typed context bindings are invalid.")
    grounding_record_type = (
        "RobotFrameLocationRecord"
        if completion.get("schema_version") == 3
        else "RobotFramePoseRecord"
    )
    pose_binding = next(
        (
            item
            for item in typed_bindings
            if isinstance(item, Mapping)
            and item.get("output_symbol") == grounding_record_type
        ),
        None,
    )
    if not isinstance(pose_binding, Mapping):
        raise ValueError(f"Final {grounding_record_type} binding is unavailable.")
    pose_record = _read_json_object(
        _interaction_ref_path(interaction_root, pose_binding.get("record_ref"))
    )

    context_summary = contract.get("context_summary")
    if not isinstance(context_summary, str) or not context_summary:
        raise ValueError("Final context summary is invalid.")
    session = None
    if completion.get("schema_version") == 2:
        session = _read_json_object(
            _interaction_ref_path(
                interaction_root,
                completion.get("grounding_session_ref"),
            )
        )
    robot_frame_pose = pose_record.get("robot_frame_pose")
    translation = pose_record.get("translated_location_m")
    if translation is None:
        translation = (
            robot_frame_pose.get("CAD_centroid_translation_m")
            if isinstance(robot_frame_pose, Mapping)
            else None
        )
    return {
        "status": completion.get("status"),
        "product_requirement": completion.get("product_requirement"),
        "context_summary": context_summary,
        "process": _compact_iri(selection.get("process_iri"), prefixes),
        "selected_resource": selection.get("selected_resource_symbol"),
        "selected_resource_jid": selection.get("selected_resource_jid"),
        "execution_mode": selection.get("selected_execution_mode"),
        "target_context_ref": _target_context_ref(
            pose_record,
            pose_binding,
            contract,
        ),
        "target_frame": pose_record.get("target_frame"),
        "location": pose_record.get("location"),
        "pose": pose_record.get("pose"),
        "robot_frame_conversion": pose_record.get("robot_frame_conversion"),
        "CAD_centroid_translation_m": translation,
        "evidence": _final_result_evidence(
            product_context=product_context,
            contract=contract,
            session=session,
            selection=selection,
        ),
        "limitations": _final_result_limitations(product_context, contract),
        "ontology": _final_ontology_view(product_context, raw_turtle),
    }


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
    events = [
        _timeline_event("accepted", "Requirement received", product_requirement)
    ]
    tool_calls = [
        value
        for name, value in records.items()
        if name.startswith("tool_call_") and isinstance(value, Mapping)
    ]
    successful_calls = [item for item in tool_calls if item.get("failure") is None]
    evidence_detail = (
        f"{len(successful_calls)} approved evidence retrievals were accepted."
        if tool_calls
        else "No external evidence retrieval was needed for this grounding."
    )
    events.append(
        _timeline_event(
            "accepted" if successful_calls or not tool_calls else "failed",
            "Evidence investigated",
            evidence_detail,
        )
    )
    proposal = _latest_record(records, "ontology_grounding_proposal_")
    if isinstance(proposal, Mapping) and proposal.get("status") == "accepted":
        events.append(
            _timeline_event(
                "accepted",
                "Context grounded",
                "The evidence-backed ontology context passed deterministic validation.",
            )
        )
    if final_result is not None:
        resource = final_result.get("selected_resource")
        mode = final_result.get("execution_mode")
        events.append(
            _timeline_event(
                "accepted",
                "Resource selected",
                " · ".join(
                    str(value)
                    for value in (resource, mode)
                    if value not in {None, ""}
                ),
            )
        )
        events.append(
            _timeline_event(
                "accepted",
                "Grounding complete",
                "The validated context and coarse resource assignment are available.",
            )
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
            _timeline_event(
                "failed",
                "Grounding incomplete",
                _failure_message(terminal_failure),
            )
        )
    elif (
        isinstance(latest_output, Mapping)
        and latest_output.get("grounding_status") != "complete"
    ):
        message = latest_output.get("insufficient_evidence") or latest_output.get(
            "clarification_question"
        )
        title = (
            "Clarification requested"
            if latest_output.get("grounding_status") == "clarification_required"
            else "Grounding incomplete"
        )
        events.append(
            _timeline_event(
                "waiting",
                title,
                str(message or "More context is required."),
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
    latest_output = (
        latest_turn.get("PA_output") if isinstance(latest_turn, Mapping) else None
    )
    latest_turn_number = (
        latest_turn.get("turn") if isinstance(latest_turn, Mapping) else None
    )
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
        isinstance(latest_clarification, dict)
        and latest_clarification.get("action") == "cancelled"
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
            "message": "The persisted context completion failed validation.",
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
    session_records = [
        record
        for name, record in records.items()
        if name.startswith("grounding_session_revision_")
        and isinstance(record, dict)
    ]
    latest_session = session_records[-1] if session_records else None
    grounding_status = (
        latest_output.get("grounding_status")
        if isinstance(latest_output, Mapping)
        else None
    )
    if grounding_status is None and isinstance(latest_session, dict):
        grounding_status = latest_session.get("status")
    terminal_failure = presentation_failure or _terminal_failure(
        phase_3_1,
        phase_3_2,
        phase_3_3,
    )
    phase_3_1_failure = phase_3_1.get("failure")
    grounding_unavailable = (
        isinstance(phase_3_1_failure, dict)
        and phase_3_1_failure.get("reason") == "grounding_unavailable"
    )
    if grounding_unavailable:
        activity_state, activity_color = "grounding unavailable", "amber"
        activity_message = (
            f"{_failure_message(phase_3_1_failure)} No PA evidence "
            "decision was requested."
        )
    elif phase_3_1_failure is not None:
        activity_state, activity_color = "failed", "red"
        activity_message = _failure_message(phase_3_1_failure)
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
        activity_message = (
            "The validated context and coarse resource assignment are available below."
        )
    elif grounding_status in {"incomplete", "ontology_gap"}:
        activity_state, activity_color = "grounding incomplete", "amber"
        reason = (
            latest_output.get("insufficient_evidence")
            if isinstance(latest_output, Mapping)
            else None
        )
        activity_message = (
            f"PA grounding stopped as {grounding_status}. "
            f"Reason: {reason or 'No safe grounding action remained.'}"
        )
    elif grounding_status in {"waiting_for_evidence", "waiting_for_user"}:
        activity_state, activity_color = "grounding waiting", "amber"
        activity_message = (
            f"PA grounding is {str(grounding_status).replace('_', ' ')}."
        )
    elif any(
        name.startswith("tool_call_")
        and isinstance(value, Mapping)
        and value.get("failure") is None
        for name, value in records.items()
    ):
        activity_state, activity_color = "stopped", "grey"
        activity_message = (
            "The interaction stopped after evidence serving and has no validated "
            "completion."
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
    return {
        "activity_state": activity_state,
        "activity_color": activity_color,
        "activity_message": activity_message,
        "timeline": timeline,
        "final_result": final_result,
        "clarification": (
            clarification_question
            if isinstance(clarification_question, str)
            else ""
        ),
        "pending_clarification_turn": str(pending_clarification_turn or ""),
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
            if needed_context is None
            and assessment["unresolved_semantic_need"] is None
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
            interaction_root
            / "products/grounding/ontology/assertion_provenance.json",
        )
    ]
    grounding_paths.extend(
        (f"ontology_{path.stem}", path)
        for path in sorted(
            (interaction_root / "products/grounding/ontology").glob("delta_*.json")
        )
    )
    grounding_paths.extend(
        (f"product_context_{path.stem}", path)
        for path in sorted(
            (interaction_root / "products/grounding/product_context").glob(
                "view_*.json"
            )
        )
    )
    grounding_paths.extend(
        (f"grounding_session_{path.stem}", path)
        for path in sorted(
            (interaction_root / "products/grounding/session").glob(
                "revision_*.json"
            )
        )
    )
    grounding_paths.extend(
        (f"ontology_grounding_{path.stem}", path)
        for path in sorted(
            (interaction_root / "products/grounding/ontology_grounding").glob(
                "proposal_*.json"
            )
        )
    )
    grounding_paths.extend(
        (f"grounding_completion_{path.stem}", path)
        for path in sorted(
            (interaction_root / "products/grounding/completion").glob("*.json")
        )
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
    max_pa_turns = (
        settings.get("max_pa_turns") if isinstance(settings, Mapping) else 12
    )
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


def _render_final_grounding_result() -> dict[str, Any]:
    """Render the hidden final grounding result and return updateable elements."""
    with ui.card().classes(
        "w-full border-2 border-emerald-200 bg-white shadow-sm"
    ) as card:
        with ui.row().classes("w-full items-start justify-between gap-3"):
            with ui.column().classes("gap-0"):
                ui.label("Final Grounding Result").classes(
                    "text-xl font-semibold text-slate-900"
                )
                ui.label("Validated ProductAgent context").classes(
                    "text-xs text-slate-500"
                )
            status_badge = ui.badge("complete").props("color=green outline")

        ui.label("product_requirement").classes(
            "text-xs font-semibold text-slate-500"
        )
        requirement_value = ui.label("").classes(
            "text-base font-medium text-slate-900 whitespace-pre-wrap"
        )
        ui.label("ProductAgent context summary").classes(
            "text-xs font-semibold text-slate-500"
        )
        context_summary_value = ui.label("").classes(
            "text-sm text-slate-700 whitespace-pre-wrap break-words"
        )

        with ui.row().classes("w-full gap-3 items-stretch flex-wrap"):
            with ui.card().classes(
                "flex-1 min-w-48 border border-slate-200 bg-slate-50 shadow-none"
            ):
                ui.label("Process").classes("text-xs font-semibold text-slate-500")
                process_value = ui.label("").classes(
                    "text-base font-semibold text-slate-900"
                )
            with ui.card().classes(
                "flex-1 min-w-48 border border-slate-200 bg-slate-50 shadow-none"
            ):
                ui.label("Selected Resource").classes(
                    "text-xs font-semibold text-slate-500"
                )
                resource_value = ui.label("").classes(
                    "text-base font-semibold text-slate-900"
                )
            with ui.card().classes(
                "flex-[2] min-w-72 border border-slate-200 bg-slate-50 shadow-none"
            ):
                ui.label("Grounded Location").classes(
                    "text-xs font-semibold text-slate-500"
                )
                target_value = ui.label("").classes(
                    "text-sm font-medium text-slate-900 whitespace-pre-wrap"
                )

        ui.label("Validated Evidence").classes(
            "text-sm font-semibold text-slate-800"
        )
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
        ui.label("Authoritative final interaction ABox").classes(
            "text-xs text-slate-500"
        )
        ontology_graph = ui.mermaid("flowchart LR\n    empty[No final ABox]").classes(
            "w-full overflow-x-auto"
        )
        ontology_table = ui.table(
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
        ).props("flat bordered wrap-cells hide-bottom").classes("w-full")
        with ui.expansion("Raw Turtle", icon="data_object").classes(
            "w-full border border-slate-200 rounded"
        ):
            raw_turtle_value = ui.code("", language="turtle").classes(
                "w-full text-xs overflow-x-auto"
            )
    card.set_visibility(False)
    return {
        "card": card,
        "status_badge": status_badge,
        "requirement": requirement_value,
        "context_summary": context_summary_value,
        "process": process_value,
        "resource": resource_value,
        "target": target_value,
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
    elements["status_badge"].set_text(str(result.get("status", "complete")))
    elements["requirement"].set_text(str(result.get("product_requirement", "")))
    elements["context_summary"].set_text(str(result.get("context_summary", "")))
    elements["process"].set_text(str(result.get("process", "")))
    resource_text = " · ".join(
        str(result[field])
        for field in ("selected_resource", "execution_mode")
        if result.get(field) not in {None, ""}
    )
    elements["resource"].set_text(resource_text)
    target_lines = [
        " · ".join(
            str(result[field])
            for field in ("target_context_ref", "target_frame")
            if result.get(field) not in {None, ""}
        ),
        f"location: {result.get('location')}",
    ]
    translation = result.get("CAD_centroid_translation_m")
    if isinstance(translation, list):
        target_lines.append(
            "CAD_centroid_translation_m: "
            + json.dumps(translation, ensure_ascii=False, allow_nan=False)
        )
    elements["target"].set_text("\n".join(line for line in target_lines if line))

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
                ui.badge(f"{item.get('label')} · {status}").props(
                    f"color={color} outline"
                )

    limitations = result.get("limitations")
    limitation_values = limitations if isinstance(limitations, list) else []
    elements["limitations"].set_text(
        "\n".join(f"• {value}" for value in limitation_values)
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
    grounding_ready = (
        runtime.ontology_config is not None and runtime.grounding_runtime is not None
    )
    grounding_unavailable_reason = (
        runtime.document_diagnostic_unavailable_reason
        or "Authoritative TBox and controlled Phase 4 grounding runtime are unavailable."
    )

    with ui.card().classes("w-full border border-slate-200 shadow-sm"):
        with ui.row().classes("w-full items-center justify-between gap-3 flex-wrap"):
            with ui.column().classes("gap-0"):
                ui.label("ProductAgent Grounding").classes(
                    "text-lg font-semibold text-slate-900"
                )
                ui.label("Requirement to validated product context").classes(
                    "text-xs text-slate-500"
                )
            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.badge(
                    "grounding ready" if grounding_ready else "grounding unavailable"
                ).props(
                    f"color={'green' if grounding_ready else 'amber'} outline"
                )
                calibration_state, calibration_color, calibration_message = (
                    _calibration_readiness(runtime)
                )
                ui.badge(calibration_state).props(
                    f"color={calibration_color} outline"
                )

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
            ui.label("ProductAgent").classes(
                "text-sm font-semibold text-indigo-900"
            )
            activity_badge = ui.badge(
                "idle" if grounding_ready else "grounding unavailable"
            ).props(f"color={'indigo' if grounding_ready else 'amber'} outline")
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
                ui.label("ProductAgent Timeline").classes(
                    "text-base font-semibold text-slate-900"
                )
                timeline_badge = ui.badge("live").props("color=indigo outline")
            timeline_container = ui.column().classes("w-full gap-3")
        timeline_card.set_visibility(False)

        with ui.card().classes(
            "w-full border border-amber-200 bg-amber-50 shadow-none"
        ) as clarification_card:
            ui.label("ProductAgent clarification").classes(
                "text-sm font-semibold text-amber-900"
            )
            clarification_prompt = ui.label("").classes("text-xs text-amber-800")
            clarification_reply_input = (
                ui.input(label="User reply").props("outlined").classes("w-full")
            )
            with ui.row().classes("items-center gap-2"):
                submit_reply_button = ui.button(
                    "Submit Reply", icon="send"
                ).props("disable")
                cancel_interaction_button = ui.button(
                    "Cancel Interaction", icon="cancel"
                ).props("outline disable")
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
            diagnostic_identifier = ui.label("Interaction: none").classes(
                "text-xs text-slate-600"
            )
            diagnostic_path = ui.label("Persisted path: none").classes(
                "text-xs text-slate-600 break-all"
            )
            diagnostic_counts = ui.label(
                "Turns: 0 · evidence: 0 · decisions: 0"
            ).classes("text-xs text-slate-600")
            with ui.card().classes(
                "w-full border border-red-200 bg-red-50 shadow-none"
            ) as diagnostic_failure_card:
                ui.label("Failure details").classes(
                    "text-sm font-semibold text-red-900"
                )
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
            "interaction": None,
            "timeline": [],
        }

        def _update_start_enabled() -> None:
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
            persisted_events = [
                event for event in timeline if isinstance(event, dict)
            ] if isinstance(timeline, list) else []
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
            timeline_badge.props(
                f"color={timeline_colors.get(timeline_state, 'indigo')} outline"
            )
            recovered_label.set_visibility(recovered)
            result = view.get("final_result")
            _apply_final_grounding_result(
                final_result_elements,
                result if isinstance(result, Mapping) else None,
            )

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

        async def _start_pa_interaction() -> None:
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

            def _show_pa_event(event: dict[str, str]) -> None:
                events = action_state["timeline"]
                if not isinstance(events, list):
                    events = []
                    action_state["timeline"] = events
                events.append(event)
                _render_pa_timeline_events(timeline_container, events)

            def _show_pa_stage(sentence: str) -> None:
                activity_badge.set_text("grounding")
                activity_badge.props("color=indigo")
                activity_message.set_text(sentence)

            try:
                interaction = await _run_pa_ui_interaction(
                    runtime,
                    value,
                    on_pa_stage=_show_pa_stage,
                    on_pa_event=_show_pa_event,
                )
                view = _pa_ui_view(interaction)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                failure_message = (
                    f"Connected PA workflow failed: {type(exc).__name__}: {exc}"
                )
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
                _show_pa_event(
                    _timeline_event("failed", "Grounding failed", failure_message)
                )
                ui.notify("Connected PA workflow failed.", type="negative")
            else:
                _apply_pa_view(interaction, view)
                notification_type = (
                    "positive"
                    if str(view["activity_state"]) == "grounding complete"
                    else "warning"
                )
                ui.notify(str(view["activity_state"]), type=notification_type)
            finally:
                action_state["busy"] = False
                if action_state["pending_clarification"]:
                    requirement_input.props("disable")
                else:
                    requirement_input.props(remove="disable")
                _update_start_enabled()

        def _update_reply_enabled() -> None:
            value = clarification_reply_input.value
            _set_enabled(
                submit_reply_button,
                not action_state["busy"]
                and bool(action_state["pending_clarification"])
                and isinstance(value, str)
                and bool(value.strip()),
            )

        async def _submit_clarification() -> None:
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

            def _show_pa_event(event: dict[str, str]) -> None:
                events = action_state["timeline"]
                if not isinstance(events, list):
                    events = []
                    action_state["timeline"] = events
                events.append(event)
                _render_pa_timeline_events(timeline_container, events)

            _show_pa_event(
                _timeline_event("accepted", "Clarification submitted", reply)
            )

            def _show_pa_stage(sentence: str) -> None:
                activity_badge.set_text("grounding")
                activity_badge.props("color=indigo")
                activity_message.set_text(sentence)

            try:
                interaction = await _submit_pa_ui_clarification(
                    runtime,
                    interaction,
                    reply,
                    on_pa_stage=_show_pa_stage,
                    on_pa_event=_show_pa_event,
                )
                view = _pa_ui_view(interaction)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                failure_message = (
                    f"Clarification failed: {type(exc).__name__}: {exc}"
                )
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
                _show_pa_event(
                    _timeline_event("failed", "Clarification failed", failure_message)
                )
                ui.notify("Clarification failed.", type="negative")
            else:
                _apply_pa_view(interaction, view)
                ui.notify(str(view["activity_state"]), type="positive")
            finally:
                action_state["busy"] = False
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

        requirement_input.on_value_change(lambda _: _update_start_enabled())
        clarification_reply_input.on_value_change(lambda _: _update_reply_enabled())
        start_button.on_click(_start_pa_interaction)
        submit_reply_button.on_click(_submit_clarification)
        cancel_interaction_button.on_click(_cancel_clarification)
        _update_start_enabled()

        try:
            recovered_interaction = _latest_pa_ui_interaction(runtime.contexts_root)
            recovered_view = (
                _pa_ui_view(recovered_interaction)
                if recovered_interaction is not None
                else None
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

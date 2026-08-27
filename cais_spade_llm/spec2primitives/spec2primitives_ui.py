"""NiceGUI page for the ICRA 2027 Spec2Primitives case study."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
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
    continue_pa_context_interaction,
    serve_pa_requested_context,
    start_pa_context_interaction,
)
from cais_spade_llm.spec2primitives.tools.document_evidence import (
    DOCUMENT_CONTEXT_REF,
    run_document_interpretation_diagnostic,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding import (
    read_rgbd_segmentation_status,
)

_FLOW_STEPS = (
    "product requirement: assemble Medium Gear",
    "PA retrieves manual/specification/CAD",
    "PA grounds target_feature, target pose, insertion axis, tolerances",
    "RA retrieves fresh resource state and resource-owned primitive catalog",
    "RA authors primitive_steps",
    "state checks + IK/collision/trajectory validation",
)
_PHASE_2_CONNECTED_MESSAGE = (
    "Connected through Phase 3.3's ontology-backed orchestration boundary. The "
    "live grounding workflow remains fail-closed until an authoritative TBox and "
    "controlled Phase 4 producers are configured; planning, RA, and robot "
    "action remain unavailable."
)
_PHASE_2_RESULT_AREAS = (
    ("needed_context", "No needed_context decision is available."),
    ("served context", "No document, CAD, or observation context was served."),
    (
        "Evidence Sources",
        "Document pages, CAD files, and observation references will appear here.",
    ),
    ("retrieval error", "No retrieval was attempted."),
    ("clarification", "No clarification_question is available."),
)
_ASSEMBLY_PLAN_COLUMNS = (
    "Step",
    "Assembly Task",
    "Required Outcome",
    "Evidence Sources",
)


class _PAUIRuntimeObserver:
    """Update the UI before delegating each structured ProductAgent call."""

    def __init__(
        self,
        product_agent: ProductAgentContextRuntime,
        *,
        max_pa_turns: int,
        on_pa_turn: Callable[[int, int], None],
    ) -> None:
        self._product_agent = product_agent
        self._max_pa_turns = max_pa_turns
        self._on_pa_turn = on_pa_turn
        self._turn_number = 0

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        """Report the next turn and call only the composed PA interface."""
        self._turn_number += 1
        self._on_pa_turn(self._turn_number, self._max_pa_turns)
        return await self._product_agent.ask_llm_structured(
            prompt,
            response_format=response_format,
        )


def _phase_2_connection_message() -> str:
    """Return the fixed Phase 2 connected-state boundary message."""
    return _PHASE_2_CONNECTED_MESSAGE


def _render_flow_step(number: int, text: str) -> None:
    """Render one static workflow step."""
    with (
        ui.card().classes("w-full border border-slate-200 shadow-sm"),
        ui.row().classes("items-center gap-4 w-full"),
    ):
        ui.badge(str(number)).props("color=indigo rounded")
        ui.label(text).classes("text-base text-slate-800")


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
    with ui.card().classes("flex-1 min-w-80 border border-slate-200 shadow-sm"):
        with ui.row().classes("w-full items-center justify-between gap-3"):
            with ui.column().classes("gap-0"):
                ui.label("Dual Gazebo Environment").classes("text-lg font-semibold text-slate-900")
                ui.label("xArm6 + UR5e · NIST CAD · Gazebo + MoveIt/RViz · No hardware").classes(
                    "text-xs text-slate-500"
                )
            status_badge = ui.badge("checking").props("color=grey outline")

        status_message = ui.label("Reading fresh runtime status...").classes(
            "text-sm text-slate-600"
        )
        action_state = {
            "busy": False,
            "refreshing": False,
            "status": DualGazeboStatus(state="checking"),
        }

        with ui.row().classes("items-center gap-2 flex-wrap"):
            start_button = ui.button("Start", icon="play_arrow").props("disable")
            stop_button = ui.button("Stop", icon="stop").props("outline disable")
            refresh_button = ui.button("Refresh", icon="refresh").props("flat")

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
    on_pa_turn: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Run the connected Phase 3.1 through Phase 3.3 backend workflow."""
    interaction_identifier = f"interaction_{uuid.uuid4().hex}"
    interaction_root = runtime.contexts_root / interaction_identifier
    product_agent = runtime.product_agent
    if on_pa_turn is not None:
        product_agent = _PAUIRuntimeObserver(
            product_agent,
            max_pa_turns=max_pa_turns,
            on_pa_turn=on_pa_turn,
        )
    phase_3_1 = await start_pa_context_interaction(
        product_agent,
        interaction_root,
        product_requirement,
        ontology_config=runtime.ontology_config,
        grounding_runtime=runtime.grounding_runtime,
    )
    phase_3_2 = None
    phase_3_3 = None
    if "needed_context" in phase_3_1:
        phase_3_2 = serve_pa_requested_context(interaction_root)
    if isinstance(phase_3_2, dict) and "served_context" in phase_3_2:
        phase_3_3 = await continue_pa_context_interaction(
            product_agent,
            interaction_root,
            ontology_config=runtime.ontology_config,
            grounding_runtime=runtime.grounding_runtime,
            max_pa_turns=max_pa_turns,
        )
    return {
        "interaction_identifier": interaction_identifier,
        "interaction_root": interaction_root,
        "product_requirement": product_requirement,
        "phase_3_1": phase_3_1,
        "phase_3_2": phase_3_2,
        "phase_3_3": phase_3_3,
        "max_pa_turns": max_pa_turns,
    }


async def _run_document_diagnostic_ui(
    runtime: Spec2PrimitivesUIRuntime,
    product_requirement: str,
) -> dict[str, object]:
    """Run Phase 4.1 in a fresh ABox that is separate from the PA loop."""
    if (
        runtime.ontology_config is None
        or runtime.model_config is None
        or runtime.document_vision_runtime is None
    ):
        return {
            "status": "unavailable",
            "failure": {
                "reason": "document_interpretation_unavailable",
                "message": runtime.document_diagnostic_unavailable_reason
                or "The document interpretation diagnostic is not configured.",
            },
        }
    interaction_root = runtime.contexts_root / f"document_diagnostic_{uuid.uuid4().hex}"
    return await run_document_interpretation_diagnostic(
        interaction_root=interaction_root,
        product_requirement=product_requirement,
        ontology_config=runtime.ontology_config,
        config=runtime.model_config.document_vlm,
        vision_runtime=runtime.document_vision_runtime,
    )


def _validated_max_pa_turns(value: object) -> int | None:
    """Return one valid whole-number UI limit or `None`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not float(value).is_integer():
        return None
    max_pa_turns = int(value)
    return max_pa_turns if 2 <= max_pa_turns <= 50 else None


def _pa_ui_view(interaction: dict[str, object]) -> dict[str, str]:
    """Build operator-facing text from one connected PA interaction."""
    product_requirement = interaction["product_requirement"]
    phase_3_1 = interaction["phase_3_1"]
    phase_3_2 = interaction["phase_3_2"]
    phase_3_3 = interaction.get("phase_3_3")
    max_pa_turns = interaction.get("max_pa_turns", 12)
    if not isinstance(product_requirement, str) or not isinstance(phase_3_1, dict):
        raise ValueError("PA UI interaction result is malformed.")

    interaction_root = interaction["interaction_root"]
    interaction_identifier = interaction["interaction_identifier"]
    if not isinstance(interaction_root, Path) or not isinstance(interaction_identifier, str):
        raise ValueError("PA UI interaction path is malformed.")
    records = _interaction_records(interaction_root)
    ontology_record = records.get("ontology_initialization")
    ontology_initialized = (
        isinstance(ontology_record, dict)
        and ontology_record.get("status") == "unresolved"
        and isinstance(ontology_record.get("tbox_fingerprint"), str)
    )
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
    decisions = [
        record
        for name, record in records.items()
        if name.startswith("decision_") and isinstance(record, dict)
    ]
    needed_contexts = [
        needed_context
        for turn in turns
        if isinstance((pa_output := turn.get("PA_output")), dict)
        and isinstance((needed_context := pa_output.get("needed_context")), dict)
    ]
    served_contexts = [
        served_context
        for retrieval in retrievals
        if isinstance((served_context := retrieval.get("served_context")), dict)
    ]
    retrieval_errors = [
        retrieval_error
        for retrieval in retrievals
        if isinstance((retrieval_error := retrieval.get("retrieval_error")), dict)
    ]
    evidence_sources = [
        provenance
        for served_context in served_contexts
        if isinstance((provenance := served_context.get("provenance")), dict)
    ]
    persisted_assessments = [
        assessment
        for decision in decisions
        if isinstance((assessment := _validated_persisted_assessment(decision)), dict)
    ]
    clarification_question = next(
        (
            clarification
            for assessment in reversed(persisted_assessments)
            if isinstance(
                (needed_context := assessment.get("needed_context")),
                dict,
            )
            and isinstance(
                (clarification := needed_context.get("clarification_question")),
                str,
            )
        ),
        None,
    )
    completed = any(
        assessment.get("context understanding complete") is True
        and assessment.get("needed_context") is None
        for assessment in persisted_assessments
    )
    completed_turns = {
        decision.get("turn")
        for decision in decisions
        if isinstance((assessment := _validated_persisted_assessment(decision)), dict)
        and assessment.get("context understanding complete") is True
        and assessment.get("needed_context") is None
    }
    clarification_turns = {
        decision.get("turn")
        for decision in decisions
        if isinstance((assessment := _validated_persisted_assessment(decision)), dict)
        and isinstance((needed_context := assessment.get("needed_context")), dict)
        and isinstance(needed_context.get("clarification_question"), str)
    }
    messages = _pa_messages(
        product_requirement,
        turns=turns,
        retrievals=retrievals,
        completed_turns=completed_turns,
        clarification_turns=clarification_turns,
        max_pa_turns=max_pa_turns,
    )
    if ontology_initialized:
        messages = "Phase 4.0 ontology initialization\n\ninitialized\n\n" + messages

    terminal_failure = _terminal_failure(phase_3_1, phase_3_2, phase_3_3)
    current_turn = max(
        (
            turn["turn"]
            for turn in turns
            if isinstance(turn.get("turn"), int) and not isinstance(turn.get("turn"), bool)
        ),
        default=1,
    )
    turn_status = f"PA turn {current_turn} of {max_pa_turns}."
    phase_3_1_failure = phase_3_1.get("failure")
    grounding_unavailable = (
        isinstance(phase_3_1_failure, dict)
        and phase_3_1_failure.get("reason") == "grounding_unavailable"
    )
    if grounding_unavailable:
        activity_state, activity_color = "grounding unavailable", "amber"
        activity_message = (
            f"{turn_status} {_failure_message(phase_3_1_failure)} No PA evidence "
            "decision was requested."
        )
    elif phase_3_1_failure is not None:
        activity_state, activity_color = "failed", "red"
        activity_message = f"{turn_status} {_failure_message(phase_3_1_failure)}"
    elif terminal_failure is not None:
        activity_state, activity_color = "failed", "red"
        activity_message = f"{turn_status} {_failure_message(terminal_failure)}"
    elif isinstance(clarification_question, str):
        activity_state, activity_color = "clarification needed", "amber"
        activity_message = (
            f"{turn_status} ProductAgent requested clarification. User reply "
            "handling begins in Phase 3.4."
        )
    elif completed:
        activity_state, activity_color = "context understanding complete", "green"
        activity_message = (
            f"{turn_status} The persisted Phase 4.3 assessment reports complete "
            "product context. Phase 5 planning remains unavailable."
        )
    elif served_contexts:
        activity_state, activity_color = "context served", "green"
        activity_message = f"{turn_status} The requested context was served."
    else:
        activity_state, activity_color = "stopped", "grey"
        activity_message = f"{turn_status} The workflow stopped before context serving."
    if ontology_initialized and not grounding_unavailable:
        activity_message = f"Phase 4.0 ontology initialized. {activity_message}"

    latest_needed_context = needed_contexts[-1] if needed_contexts else None
    return {
        "activity_state": activity_state,
        "activity_color": activity_color,
        "activity_message": activity_message,
        "messages": messages,
        "needed_context": (
            _json_text(latest_needed_context)
            if latest_needed_context is not None
            else (
                "No further needed_context. context understanding complete."
                if completed
                else "No needed_context decision is available."
            )
        ),
        "served_context": (
            "\n".join(_compact_served_context(value) for value in served_contexts)
            if served_contexts
            else "No document, CAD, or observation context was served."
        ),
        "Evidence Sources": (
            _json_text(evidence_sources)
            if evidence_sources
            else "No Evidence Sources are available."
        ),
        "retrieval error": (
            _json_text(retrieval_errors) if retrieval_errors else "No retrieval error is available."
        ),
        "clarification": (
            clarification_question
            if isinstance(clarification_question, str)
            else "No clarification_question is available."
        ),
        "interaction_record": _ordered_interaction_record_text(
            interaction_identifier,
            interaction_root,
        ),
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
    if not isinstance(assessment, dict) or set(assessment) != expected_keys:
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


def _ordered_interaction_record_text(
    interaction_identifier: str,
    interaction_root: Path,
) -> str:
    records: dict[str, object] = {
        "interaction_identifier": interaction_identifier,
    }
    records.update(_interaction_records(interaction_root))
    return _json_text(records)


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
    return records


def _interaction_record_sort_key(path: Path) -> tuple[int, int]:
    if path.stem == "pa_context_settings":
        return (0, 0)
    prefix, _, suffix = path.stem.partition("_")
    order = {
        "turn": 0,
        "retrieval": 1,
        "interpretation": 2,
        "decision": 3,
    }.get(prefix, 4)
    return (int(suffix), order)


def _pa_messages(
    product_requirement: str,
    *,
    turns: list[dict[str, object]],
    retrievals: list[dict[str, object]],
    completed_turns: set[object],
    clarification_turns: set[object],
    max_pa_turns: object,
) -> str:
    retrieval_by_number = {
        retrieval["retrieval"]: retrieval
        for retrieval in retrievals
        if isinstance(retrieval.get("retrieval"), int)
    }
    messages = [
        "User product_requirement",
        product_requirement,
        "Operator Maximum PA turns",
        str(max_pa_turns),
    ]
    for turn in turns:
        turn_number = turn.get("turn")
        pa_output = turn.get("PA_output")
        if isinstance(pa_output, dict):
            needed_context = pa_output.get("needed_context")
            if isinstance(needed_context, dict):
                messages.extend(
                    [
                        f"ProductAgent turn {turn_number} needed_context",
                        _json_text(needed_context),
                    ]
                )
                clarification = needed_context.get("clarification_question")
                if isinstance(clarification, str) and turn_number in clarification_turns:
                    messages.extend(
                        [
                            "User reply",
                            "Not available until Phase 3.4 is implemented.",
                        ]
                    )
            if (
                pa_output.get("context understanding complete") is True
                and turn_number in completed_turns
            ):
                messages.extend(
                    [
                        f"ProductAgent turn {turn_number} context understanding complete",
                        (
                            "Persisted Phase 4.3 product-context assessment is "
                            "complete; Phase 5 planning remains unavailable."
                        ),
                    ]
                )
        if turn.get("failure") is not None:
            messages.extend(
                [f"ProductAgent turn {turn_number} failure", _json_text(turn["failure"])]
            )
        retrieval = retrieval_by_number.get(turn_number)
        if isinstance(retrieval, dict):
            served_context = retrieval.get("served_context")
            if isinstance(served_context, dict):
                messages.extend(
                    [
                        f"Phase 3 retrieval {turn_number} served",
                        _compact_served_context(served_context),
                    ]
                )
            if retrieval.get("retrieval_error") is not None:
                messages.extend(
                    [
                        f"Phase 3 retrieval {turn_number} retrieval_error",
                        _json_text(retrieval["retrieval_error"]),
                    ]
                )
    return "\n\n".join(messages)


def _compact_served_context(served_context: dict[str, object]) -> str:
    evidence_type = served_context.get("evidence_type")
    context_ref = served_context.get("context_ref")
    if evidence_type == "document":
        evidence = served_context.get("document_evidence")
        page_count = evidence.get("page_count") if isinstance(evidence, dict) else None
        return f"{context_ref} · document · {page_count} pages"
    if evidence_type == "CAD":
        evidence = served_context.get("CAD_evidence")
        triangle_count = evidence.get("triangle_count") if isinstance(evidence, dict) else None
        return f"{context_ref} · CAD · {triangle_count} triangles"
    observation_ref = served_context.get("observation_ref")
    observation_evidence = served_context.get("observation_evidence")
    manifest = (
        observation_evidence.get("manifest") if isinstance(observation_evidence, dict) else None
    )
    cameras = manifest.get("cameras") if isinstance(manifest, dict) else None
    camera_count = len(cameras) if isinstance(cameras, list) else 0
    return f"{observation_ref} · live observation · {camera_count} cameras"


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


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)


def _render_pa_result_area(title: str, message: str) -> Label:
    """Render one PA result area and return its updateable value label."""
    with ui.card().classes("flex-1 min-w-56 border border-slate-200 bg-slate-50 shadow-none"):
        ui.label(title).classes("text-sm font-semibold text-slate-800")
        return ui.label(message).classes("text-xs text-slate-500 whitespace-pre-wrap break-all")


def _render_pa_messages() -> tuple[Badge, Label]:
    """Render the connected user and ProductAgent message transcript."""
    with ui.card().classes("w-full border border-slate-200 bg-white shadow-none"):
        with ui.row().classes("w-full items-center justify-between gap-2"):
            ui.label("User ↔ ProductAgent Messages").classes("text-sm font-semibold text-slate-900")
            message_badge = ui.badge("no messages").props("color=grey outline")
        message_value = ui.label(
            "Submit a product_requirement to start the connected Phase 3 loop."
        ).classes("text-xs text-slate-500 whitespace-pre-wrap break-all")
    return message_badge, message_value


def _render_assembly_plan_preview() -> None:
    """Render the disabled robot-independent Phase 5 assembly plan layout."""
    with ui.card().classes("w-full border border-slate-200 bg-white shadow-none"):
        with ui.row().classes("w-full items-start justify-between gap-3"):
            with ui.column().classes("gap-0"):
                ui.label("Assembly Plan").classes("text-base font-semibold text-slate-900")
                ui.label("Robot-independent assembly plan").classes("text-xs text-slate-500")
            with ui.row().classes("items-center gap-2"):
                ui.badge("Phase 5").props("color=indigo outline")
                ui.badge("not available").props("color=grey outline")

        with ui.row().classes("w-full gap-2 rounded-t bg-slate-100 px-3 py-2 flex-nowrap"):
            for column in _ASSEMBLY_PLAN_COLUMNS:
                ui.label(column).classes("flex-1 text-xs font-semibold text-slate-700")
        ui.label("No assembly plan is available.").classes(
            "w-full rounded-b border border-t-0 border-slate-200 px-3 py-4 "
            "text-center text-xs text-slate-500"
        )
        ui.button("Open Assembly Plan", icon="description").props("disable outline")


def _render_pa_interaction(runtime: Spec2PrimitivesUIRuntime) -> None:
    """Render the Phase 2 UI connected through Phase 3.3."""
    with ui.card().classes("flex-[2] min-w-96 border border-slate-200 shadow-sm"):
        with ui.row().classes("w-full items-start justify-between gap-3"):
            with ui.column().classes("gap-0"):
                ui.label("PA Interaction").classes("text-lg font-semibold text-slate-900")
                ui.label("Phase 2 PA interaction UI").classes("text-xs text-slate-500")
            ui.badge("connected through Phase 3.3").props("color=green outline")

        ui.label(_phase_2_connection_message()).classes("text-sm text-emerald-700")

        requirement_input = (
            ui.input(
                label="product_requirement",
                placeholder="assemble Medium Gear",
            )
            .props("outlined")
            .classes("w-full")
        )

        with (
            ui.card().classes("w-full border border-slate-200 bg-slate-50 shadow-none"),
            ui.row().classes("w-full items-center justify-between gap-3 flex-wrap"),
        ):
            with ui.column().classes("flex-1 min-w-64 gap-0"):
                ui.label("Run settings").classes("text-sm font-semibold text-slate-800")
                ui.label(
                    "Phase 3.1 counts as turn 1. This limit is only an "
                    "emergency stop for the PA context loop."
                ).classes("text-xs text-slate-500")

            with ui.row().classes("items-center gap-2 flex-wrap"):
                max_pa_turns_input = (
                    ui.number(
                        label="Maximum PA turns",
                        value=12,
                        min=2,
                        max=50,
                        step=1,
                        format="%.0f",
                    )
                    .props("outlined")
                    .classes("w-40")
                )
                start_button = ui.button(
                    "Start PA Context Interaction",
                    icon="play_arrow",
                ).props("disable")

        with ui.card().classes("w-full border border-indigo-100 bg-indigo-50 shadow-none"):
            with ui.row().classes("w-full items-center justify-between gap-2"):
                ui.label("ProductAgent").classes("text-sm font-semibold text-indigo-900")
                activity_badge = ui.badge("idle").props("color=indigo outline")
            activity_message = ui.label("No PA retrieval or clarification activity.").classes(
                "text-xs text-indigo-700"
            )

        message_badge, message_value = _render_pa_messages()

        result_values: dict[str, Label] = {}
        with ui.row().classes("w-full gap-2 items-stretch flex-wrap"):
            for title, message in _PHASE_2_RESULT_AREAS:
                result_values[title] = _render_pa_result_area(title, message)

        _render_assembly_plan_preview()

        with ui.expansion("ordered interaction record", icon="history").classes(
            "w-full border border-slate-200 rounded"
        ):
            interaction_record_value = ui.label(
                "No interaction record exists because Phase 3 was not started."
            ).classes("text-xs text-slate-500 p-2 whitespace-pre-wrap break-all")

        action_state = {"busy": False}

        def _update_start_enabled() -> None:
            value = requirement_input.value
            max_pa_turns = _validated_max_pa_turns(max_pa_turns_input.value)
            _set_enabled(
                start_button,
                not action_state["busy"]
                and isinstance(value, str)
                and bool(value.strip())
                and max_pa_turns is not None,
            )

        async def _start_pa_interaction() -> None:
            value = requirement_input.value
            max_pa_turns = _validated_max_pa_turns(max_pa_turns_input.value)
            if (
                action_state["busy"]
                or not isinstance(value, str)
                or not value.strip()
                or max_pa_turns is None
            ):
                return
            action_state["busy"] = True
            _set_enabled(start_button, False)
            requirement_input.props("disable")
            max_pa_turns_input.props("disable")
            activity_badge.set_text(f"PA turn 1 of {max_pa_turns}")
            activity_badge.props("color=indigo")
            activity_message.set_text("ProductAgent is assessing and requesting context.")
            message_badge.set_text("1 user message")
            message_value.set_text(f"User product_requirement\n\n{value}")

            def _show_pa_turn(turn_number: int, turn_limit: int) -> None:
                activity_badge.set_text(f"PA turn {turn_number} of {turn_limit}")
                activity_message.set_text(
                    "ProductAgent is assessing accumulated context and deciding "
                    "the next observable action."
                )

            try:
                interaction = await _run_pa_ui_interaction(
                    runtime,
                    value,
                    max_pa_turns=max_pa_turns,
                    on_pa_turn=_show_pa_turn,
                )
                view = _pa_ui_view(interaction)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                activity_badge.set_text("failed")
                activity_badge.props("color=red")
                activity_message.set_text(
                    f"Connected PA workflow failed: {type(exc).__name__}: {exc}"
                )
                ui.notify("Connected PA workflow failed.", type="negative")
            else:
                activity_badge.set_text(view["activity_state"])
                activity_badge.props(f"color={view['activity_color']}")
                activity_message.set_text(view["activity_message"])
                message_badge.set_text("interaction recorded")
                message_badge.props("color=indigo")
                message_value.set_text(view["messages"])
                for title, value_label in result_values.items():
                    value_label.set_text(view[title])
                interaction_record_value.set_text(view["interaction_record"])
                notification_type = (
                    "positive"
                    if view["activity_state"]
                    in {
                        "context served",
                        "context understanding complete",
                    }
                    else "warning"
                )
                ui.notify(view["activity_state"], type=notification_type)
            finally:
                action_state["busy"] = False
                requirement_input.props(remove="disable")
                max_pa_turns_input.props(remove="disable")
                _update_start_enabled()

        requirement_input.on_value_change(lambda _: _update_start_enabled())
        max_pa_turns_input.on_value_change(lambda _: _update_start_enabled())
        start_button.on_click(_start_pa_interaction)


def _render_document_interpretation_diagnostic(
    runtime: Spec2PrimitivesUIRuntime,
) -> None:
    """Render the non-authoritative Phase 4.1 document diagnostic."""
    ready = (
        runtime.ontology_config is not None
        and runtime.model_config is not None
        and runtime.document_vision_runtime is not None
    )
    model = (
        runtime.model_config.document_vlm.model
        if runtime.model_config is not None
        else "unconfigured"
    )
    with ui.card().classes("w-full border border-slate-200 shadow-sm"):
        with ui.row().classes("w-full items-start justify-between gap-3"):
            with ui.column().classes("gap-0"):
                ui.label("Phase 4.1 Document Interpretation Diagnostic").classes(
                    "text-lg font-semibold text-slate-900"
                )
                ui.label(
                    "OpenAI VLM · separate diagnostic ABox · no PA completion decision"
                ).classes("text-xs text-slate-500")
            readiness_badge = ui.badge("ready" if ready else "unavailable").props(
                f"color={'green' if ready else 'amber'} outline"
            )

        ui.label(f"Approved source: {DOCUMENT_CONTEXT_REF}").classes("text-sm text-slate-700")
        ui.label(f"Configured VLM: {model}").classes("text-sm text-slate-700")
        if not ready:
            ui.label(
                runtime.document_diagnostic_unavailable_reason
                or "Authoritative ontology and OpenAI vision runtime are required."
            ).classes("text-sm text-amber-700")

        requirement_input = (
            ui.input(
                label="diagnostic product_requirement",
                placeholder="assemble Medium Gear",
            )
            .props("outlined")
            .classes("w-full")
        )
        run_button = ui.button(
            "Interpret Approved Document",
            icon="document_scanner",
        ).props("disable")
        result_value = ui.label("No document diagnostic has run.").classes(
            "text-xs text-slate-500 whitespace-pre-wrap break-all"
        )
        action_state = {"busy": False}

        def _update_enabled() -> None:
            value = requirement_input.value
            _set_enabled(
                run_button,
                ready
                and not action_state["busy"]
                and isinstance(value, str)
                and bool(value.strip()),
            )

        async def _run() -> None:
            value = requirement_input.value
            if not ready or action_state["busy"] or not isinstance(value, str) or not value.strip():
                return
            action_state["busy"] = True
            _set_enabled(run_button, False)
            requirement_input.props("disable")
            readiness_badge.set_text("running")
            readiness_badge.props("color=indigo")
            result_value.set_text("Rendering and interpreting all approved PDF pages...")
            try:
                result = await _run_document_diagnostic_ui(runtime, value)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                readiness_badge.set_text("rejected")
                readiness_badge.props("color=red")
                result_value.set_text(f"Diagnostic failed: {type(exc).__name__}: {exc}")
                ui.notify("Document interpretation rejected.", type="negative")
            else:
                status = result.get("status")
                readiness_badge.set_text(str(status))
                readiness_badge.props(f"color={'green' if status == 'accepted' else 'red'}")
                result_value.set_text(_json_text(result))
                ui.notify(
                    f"Document interpretation {status}.",
                    type="positive" if status == "accepted" else "negative",
                )
            finally:
                action_state["busy"] = False
                requirement_input.props(remove="disable")
                _update_enabled()

        requirement_input.on_value_change(lambda _: _update_enabled())
        run_button.on_click(_run)


def _render_rgbd_segmentation_status(runtime: Spec2PrimitivesUIRuntime) -> None:
    """Render read-only status for automatic supporting RGB-D processing."""
    with ui.card().classes("w-full border border-slate-200 shadow-sm"):
        with ui.row().classes("w-full items-start justify-between gap-3"):
            with ui.column().classes("gap-0"):
                ui.label("RGB-D Observation Processing").classes(
                    "text-lg font-semibold text-slate-900"
                )
                ui.label(
                    "Automatic capture, preprocessing, and minimal segmentation status"
                ).classes("text-xs text-slate-500")
            status_badge = ui.badge("idle").props("color=grey outline")

        status_message = ui.label(
            "Waiting for an observation request from the runtime."
        ).classes("text-sm text-slate-600")
        with ui.row().classes("w-full gap-6 flex-wrap"):
            source_count = ui.label("Source candidates: 0").classes(
                "text-sm text-slate-700"
            )
            assembly_count = ui.label("Assembly candidates: 0").classes(
                "text-sm text-slate-700"
            )
        correspondence_status = ui.label("CAD correspondence: not_requested").classes(
            "text-sm text-amber-700"
        )
        location_status = ui.label("Location: not_requested").classes(
            "text-sm text-amber-700"
        )
        pose_status = ui.label("Pose: not_evaluated").classes(
            "text-sm text-amber-700"
        )
        robot_frame_conversion_status = ui.label(
            "Robot-frame conversion: not_evaluated"
        ).classes("text-sm text-amber-700")

        async def _refresh_status() -> None:
            try:
                record = await asyncio.to_thread(
                    read_rgbd_segmentation_status,
                    runtime.contexts_root,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record = {
                    "status": "failed",
                    "source_candidate_count": 0,
                    "assembly_candidate_count": 0,
                    "failure": {
                        "message": f"Status unavailable: {type(exc).__name__}: {exc}"
                    },
                }
            status = str(record.get("status", "failed"))
            colors = {
                "idle": "grey",
                "running": "indigo",
                "ready": "green",
                "failed": "red",
            }
            status_badge.set_text(status)
            status_badge.props(f"color={colors.get(status, 'red')}")
            source_count.set_text(
                f"Source candidates: {record.get('source_candidate_count', 0)}"
            )
            assembly_count.set_text(
                f"Assembly candidates: {record.get('assembly_candidate_count', 0)}"
            )
            CAD_correspondence = record.get("CAD_correspondence", "not_evaluated")
            location = record.get("location", "not_evaluated")
            correspondence_status.set_text(
                "CAD correspondence: "
                + (
                    "not_requested"
                    if CAD_correspondence == "not_evaluated"
                    else str(CAD_correspondence)
                )
            )
            location_status.set_text(
                "Location: "
                + (
                    "not_requested"
                    if location == "not_evaluated"
                    else str(location)
                )
            )
            pose = record.get("pose", "not_evaluated")
            pose_status.set_text(
                "Pose: "
                + ("not_requested" if pose == "not_evaluated" else str(pose))
            )
            robot_frame_conversion = record.get(
                "robot_frame_conversion",
                "not_evaluated",
            )
            robot_frame_conversion_status.set_text(
                "Robot-frame conversion: "
                + (
                    "not_requested"
                    if robot_frame_conversion == "not_evaluated"
                    else str(robot_frame_conversion)
                )
            )
            failure = record.get("failure")
            if status == "idle":
                message = "Waiting for an observation request from the runtime."
            elif status == "running":
                message = "Capturing and processing fresh RGB-D evidence automatically."
            elif status == "ready":
                message = (
                    "Robot-frame conversion status is ready for later use."
                    if robot_frame_conversion != "not_evaluated"
                    else "CAD pose status is ready for later use."
                    if pose != "not_evaluated"
                    else "CAD-size association status is ready for later use."
                    if CAD_correspondence != "not_evaluated"
                    else "Minimal segmentation records are ready for later use."
                )
            elif isinstance(failure, dict) and isinstance(failure.get("message"), str):
                message = failure["message"]
            else:
                message = "Automatic RGB-D processing failed closed."
            status_message.set_text(message)
            status_message.classes(
                replace=(
                    "text-sm text-red-700"
                    if status == "failed"
                    else "text-sm text-slate-600"
                )
            )

        ui.timer(0.1, _refresh_status, once=True)
        ui.timer(3.0, _refresh_status)


def render(runtime: Spec2PrimitivesUIRuntime) -> None:
    """Render the Spec2Primitives PA UI connected through Phase 3.3."""
    with ui.column().classes("w-full max-w-6xl mx-auto gap-6 p-6"):
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

        with ui.card().classes("w-full bg-indigo-50 border border-indigo-100 shadow-none"):
            ui.label("Phase 2: PA Interaction UI").classes(
                "text-base font-semibold text-indigo-900"
            )
            ui.label(
                "The PA workspace exposes Phase 3.1 intake, Phase 3.2 requested "
                "context serving, and the Phase 3.3 ontology-backed orchestration "
                "boundary. Production grounding, planning, and execution "
                "remain unavailable."
            ).classes("text-sm text-indigo-800")

        with ui.row().classes("w-full gap-4 items-stretch flex-wrap"):
            _render_dual_gazebo(runtime.dual_gazebo)
            _render_pa_interaction(runtime)

        _render_document_interpretation_diagnostic(runtime)
        _render_rgbd_segmentation_status(runtime)

        ui.label("Proposed Workflow").classes("text-xl font-semibold text-slate-900")

        with ui.column().classes("w-full gap-2"):
            for number, text in enumerate(_FLOW_STEPS, start=1):
                _render_flow_step(number, text)
                if number < len(_FLOW_STEPS):
                    ui.icon("south").classes("self-center text-slate-400")

        with ui.row().classes("w-full gap-4 items-stretch flex-wrap"):
            with ui.card().classes(
                "flex-1 min-w-72 bg-amber-50 border border-amber-200 shadow-none"
            ):
                ui.badge("rejected").props("color=amber outline")
                ui.label("concrete feedback → RA revision").classes(
                    "text-base font-semibold text-amber-900"
                )
            with ui.card().classes(
                "flex-1 min-w-72 bg-emerald-50 border border-emerald-200 shadow-none"
            ):
                ui.badge("accepted").props("color=green outline")
                ui.label("RobotAgent execution").classes("text-base font-semibold text-emerald-900")

        ui.label(
            "ProductAgent, ResourceAgent, and RobotAgent remain shared runtime "
            "authorities outside the Spec2Primitives package."
        ).classes("text-sm text-slate-500")

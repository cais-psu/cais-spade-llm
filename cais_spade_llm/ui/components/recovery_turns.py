from __future__ import annotations

"""Expandable recovery turns rendered from original request and response records."""

import json
from contextlib import suppress
from pathlib import Path

from nicegui import ui

from cais_spade_llm.ui.evidence import BackgroundSection, read_json_cached
from cais_spade_llm.ui.recovery_evidence import read_recovery_turn


def _text(value) -> None:
    ui.label(value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)).classes("w-full text-sm").style(
        "white-space: pre-wrap; overflow-wrap: anywhere; font-family: monospace")


def _download(path: str | Path, label: str) -> None:
    ui.button(label, icon="download", on_click=lambda: ui.download.file(path)).props("flat dense")


def _messages(messages: list[dict], *, pretty_json: bool = False) -> None:
    for message in messages:
        ui.label(f"role={message.get('role', 'not recorded')}").classes("font-semibold text-sm")
        if message.get("content") is not None:
            content = message["content"]
            if pretty_json and isinstance(content, str):
                with suppress(json.JSONDecodeError):
                    content = json.loads(content)
            _text(content)
        if message.get("tool_calls"):
            ui.label("Tool requests").classes("font-semibold text-sm")
            _text(message["tool_calls"])


def _event_table(response: object) -> None:
    if not isinstance(response, dict):
        ui.label("Proposed events unavailable: response is not a JSON object.").classes("text-amber-700")
        return
    groups = []
    if isinstance(response.get("candidate_events"), list):
        groups.append(("", response["candidate_events"]))
    if isinstance(response.get("candidate_traces"), list):
        groups.extend((f"Candidate {index + 1}", candidate.get("events", []))
                      for index, candidate in enumerate(response["candidate_traces"])
                      if isinstance(candidate, dict) and isinstance(candidate.get("events"), list))
    if isinstance(response.get("next_transition"), dict):
        suffix = response.get("transition_suffix")
        groups.append(("", [response["next_transition"], *(suffix if isinstance(suffix, list) else [])]))
    if isinstance(response.get("transition_trace"), list):
        groups.append(("", response["transition_trace"]))
    if not any(events for _, events in groups):
        ui.label("No events proposed in this turn").classes("text-sm text-slate-500")
        return
    for label, events in groups:
        if label:
            ui.label(label).classes("font-semibold text-sm")
        rows = [{"row": index, **{key: event.get(key) for key in ("event_name", "resource_jid", "part_name")}}
                for index, event in enumerate(events) if isinstance(event, dict)]
        ui.table(columns=[{"name": key, "label": label, "field": key, "align": "left"}
                          for key, label in (("event_name", "Event"), ("resource_jid", "Resource"), ("part_name", "Part"))],
                 rows=rows, row_key="row").classes("w-full").props("flat dense wrap-cells")
        if len(rows) != len(events):
            ui.label("Some proposed events are not JSON objects; see the full response.").classes("text-amber-700")


def _legacy_llm_response(result: dict) -> dict | None:
    payload = result["payload"]
    if isinstance(payload.get("llm_response"), dict):
        return payload["llm_response"]
    name = Path(result["path"]).name
    if "_primitive_generation_" in name and "_response_" in name and isinstance(payload.get("response"), dict):
        return payload["response"]
    return None


def _proposed_events(data: dict) -> None:
    ui.label("Proposed events").classes("font-semibold")
    responses = [capture for capture in data["captures"] if capture["status"] == "captured"
                 and capture["record"].get("kind") == "provider_response"]
    for capture in responses:
        record = capture["record"]
        ui.label(f"Round {record.get('tool_round', '?')} · attempt {record.get('attempt', '?')}").classes("text-sm text-slate-500")
        choices = record.get("raw_response", {}).get("choices", [])
        if not choices:
            ui.label("LLM response message missing from capture.").classes("text-amber-700")
        for index, choice in enumerate(choices):
            if len(choices) > 1:
                ui.label(f"Response {index + 1}").classes("text-sm")
            message = choice.get("message", {})
            if message.get("content") is None and message.get("tool_calls"):
                ui.label("No events proposed in this turn").classes("text-sm text-slate-500")
                continue
            try:
                content = message.get("content")
                response = json.loads(content) if isinstance(content, str) else None
            except json.JSONDecodeError:
                response = None
            _event_table(response)
    if not responses:
        # Runtime selections and projected traces must never become LLM proposals.
        legacy = [response for result in data["results"] if (response := _legacy_llm_response(result)) is not None]
        if legacy:
            for response in legacy:
                ui.label("Legacy parsed LLM response — raw capture unavailable").classes("text-amber-700")
                _event_table(response)
        else:
            ui.label("Proposed events unavailable: LLM response capture missing.").classes("text-amber-700")


def _expansion(label: str, key: str, reader, render):
    with ui.expansion(label, value=False).classes("w-full") as expansion:
        content = BackgroundSection()
    loaded = False
    restoring = False

    def changed() -> None:
        nonlocal loaded
        if not restoring:
            ui.run_javascript(f"localStorage.setItem({json.dumps(key)}, String({json.dumps(expansion.value)}));")
        if expansion.value and not loaded:
            loaded = True
            content.load(reader, render)

    async def restore() -> None:
        nonlocal restoring
        if expansion.is_deleted:
            return
        try:
            saved = await ui.run_javascript(f"localStorage.getItem({json.dumps(key)})")
        except (TimeoutError, RuntimeError):
            return
        if expansion.is_deleted:
            return
        restoring = True
        expansion.value = saved == "true"
        changed()
        restoring = False

    expansion.on_value_change(changed)
    ui.timer(.01, restore, once=True)
    content.client.on_connect(restore)

    def refresh() -> None:
        nonlocal loaded
        loaded = False
        if expansion.value:
            loaded = True
            content.load(reader, render)
    return expansion, refresh


def render_turn_content(data: dict, tab: str, *, state_key: str = "") -> None:
    """Separate exact model messages from runtime retrieval and validation."""
    if tab == "output":
        _proposed_events(data)
        ui.label("LLM response").classes("font-semibold")
    kinds = {"provider_request", "tool_result", "tool_error"} if tab == "input" else {
        "provider_response", "provider_error", "tool_result", "tool_error", "parse_error"}
    exact = False
    for capture in data["captures"]:
        record = capture["record"]
        if capture["status"] != "captured":
            ui.label("Referenced capture missing.").classes("text-amber-700")
            continue
        kind = record.get("kind")
        if kind not in kinds:
            continue
        ui.label(f"{kind} · round {record.get('tool_round', '?')} · attempt {record.get('attempt', '?')}").classes("font-semibold")
        _download(capture["path"], "Download original")
        if kind == "provider_request":
            exact = True
            payload = record.get("payload", {})
            ui.label(f"Model: {payload.get('model', 'not recorded')} · Reasoning effort: {payload.get('reasoning_effort', 'not recorded')}").classes("text-xs text-slate-500")
            _messages(payload.get("messages", []))
        elif kind == "provider_response":
            exact = True
            raw = record.get("raw_response", {})
            _messages([{**choice.get("message", {}), "role": "assistant"} for choice in raw.get("choices", [])], pretty_json=True)
        elif kind in {"tool_result", "tool_error"}:
            _expansion("Returned context", f"{state_key}:{capture['path']}:context", lambda record=record: record, _text)
        else:
            _text({key: value for key, value in record.items() if key not in {"run_id", "call_id", "recorded_at"}})
    if not exact:
        ui.label("Exact request capture missing." if tab == "input" else "Raw LLM response capture missing.").classes("text-amber-700")
    if tab == "input" and not exact:
        for prompt in data["prompts"]:
            ui.label("Legacy request summary — exact provider capture unavailable").classes("text-amber-700")
            _text(prompt["text"])
            _download(prompt["path"], "Download legacy summary")
    _runtime_records(data["results"], tab, state_key=state_key, exact=exact)


def _runtime_records(results: list[dict], tab: str, *, state_key: str, exact: bool) -> None:
    validations = []
    context_records = []
    nominal_records = {}
    for result in results:
        payload = result["payload"]
        if tab == "output":
            if not exact and (response := _legacy_llm_response(result)) is not None:
                ui.label("Legacy parsed response — raw capture unavailable").classes("text-amber-700")
                _text(response)
            evidence = {**payload, **(payload.get("validation_evidence") or {})}
            fields = {key: evidence[key] for key in ("decision", "candidate_evaluation_summary", "validation_stages",
                        "validation_findings", "selected_transition", "transition_validation", "nominal_continuation_evidence") if key in evidence}
            if fields:
                validations.append({"record": Path(result["path"]).name, **fields})
            nominal = result.get("nominal_continuation_artifact")
            if isinstance(nominal, dict):
                key = (nominal.get("path"), nominal.get("sha256"), nominal.get("status"), nominal.get("reason"))
                nominal_records.setdefault(key, nominal)
        else:
            if payload.get("context_selection"):
                _expansion("Included context and reasons", f"{state_key}:{result['path']}:inclusion",
                           lambda payload=payload: payload["context_selection"], _text)
        exchanges = {key: payload[key] for key in ("context_requests", "served_context", "context_errors", "observations", "validation_context_exchanges") if payload.get(key)}
        if exchanges:
            context_records.append({"record": Path(result["path"]).name, **exchanges})
        _download(result["path"], "Download runtime record")
    if validations:
        _expansion("Validation records", f"{state_key}:validation", lambda: validations,
                   lambda rows: _validation_records(rows, state_key=state_key))
    if context_records:
        _expansion("Returned context", f"{state_key}:context", lambda: context_records, _text)
    for nominal in nominal_records.values():
        if nominal["status"] == "unavailable":
            ui.label("Nominal continuation proof unavailable: " + nominal["reason"]).classes("text-amber-700")
            continue
        label = "Nominal continuation proof" if nominal["status"] == "verified" else "Recorded nominal continuation proof"
        _expansion(label, f"{state_key}:nominal:{nominal['path']}:{nominal.get('sha256', '')}",
                   lambda nominal=nominal: nominal["proof"], _text)
        _download(nominal["path"], "Download full nominal continuation proof")


def _validation_records(records: list[dict], *, state_key: str) -> None:
    for record in records:
        ui.label(record["record"]).classes("font-semibold text-sm")
        _text({key: value for key, value in record.items() if key != "validation_stages"})
        for index, stage in enumerate(record.get("validation_stages") or []):
            role = stage.get("validator_role")
            evidence = stage.get("evidence") or {}
            key = f"{state_key}:{record['record']}:{index}"
            if role == "ResourceAgent":
                _expansion("RA feasibility", key + ":ra",
                    lambda evidence=evidence: {name: value for name, value in evidence.items() if name != "geometry_evidence"}, _text)
                if "geometry_evidence" in evidence:
                    _expansion("Physical evidence", key + ":physical",
                               lambda evidence=evidence: evidence["geometry_evidence"], _text)
            elif role == "CentralControllerAgent":
                _expansion("CCA safety", key + ":cca", lambda evidence=evidence: evidence, _text)
            else:
                _text(stage)


class RecoveryTurns:
    """Keep individual turn expansions stable as background records arrive."""

    def __init__(self, container) -> None:
        """Own presentation state only, independently of recovery operations."""
        self.container = container
        self.directory: Path | None = None
        self.identity = ""
        self.rows: dict = {}
        self.files: set[str] = set()

    def _expansion(self, label: str, key: str, reader, render):
        return _expansion(label, key, reader, render)

    def update(self, state: dict, stage: str, turns: list[dict]) -> None:
        """Append new turns and refresh open content without collapsing it."""
        artifact_stage = "recovery_outline" if stage == "outline" else "recovery_primitves"
        raw = state.get(f"{stage}_artifact_directory")
        directory = Path(raw) if raw else Path(state["artifact_directory"]) / artifact_stage if state.get("artifact_directory") else None
        identity = f"recovery_turn:{state.get('run_id', '')}:{directory}"
        if identity != self.identity:
            self.container.clear()
            self.identity, self.directory = identity, directory
            self.rows, self.files = {}, set()
        if directory is None:
            return
        with self.container:
            preview = directory.parent / ("input_preview.json" if stage == "outline" else "primitive_input_preview.json")
            if preview.is_file() and str(preview) not in self.files:
                self.files.add(str(preview))
                def show_preview(value, path=preview):
                    _text(value.get("prompt", "Prepared prompt not captured"))
                    _download(path, "Download prepared input")
                self._expansion("Prepared input — not sent", identity + ":preview",
                                lambda path=preview: read_json_cached(path), show_preview)
            for turn in turns:
                key = (turn["turn"], turn["phase"])
                if turn.get("owner"):
                    key = (turn["phase"], turn["owner"]) if turn.get("call_id") else (*key, turn["owner"])
                if key not in self.rows:
                    with ui.card().classes("w-full"):
                        label = ui.label().classes("font-semibold")
                        row = {"data": turn, "label": label, "signature": None, "refresh": []}
                        for tab in ("input", "output"):
                            state_key = f"{identity}:{json.dumps(key)}:{tab}"
                            _, refresh = self._expansion(tab.title(), state_key,
                                lambda row=row, directory=directory: read_recovery_turn(directory, row["data"]),
                                lambda data, tab=tab, state_key=state_key: render_turn_content(data, tab, state_key=state_key))
                            row["refresh"].append(refresh)
                        self.rows[key] = row
                row = self.rows[key]
                row["label"].text = f"Turn {turn['turn']} · {turn['phase']} · {turn['status']}"
                if turn.get("resource_jid"):
                    row["label"].text += f" · {turn['resource_jid']}"
                row["data"] = turn
                if row["signature"] != turn["signature"]:
                    row["signature"] = turn["signature"]
                    for refresh in row["refresh"]:
                        refresh()
            result = directory.parent / ("outline.json" if stage == "outline" else "proposal.json")
            if result.is_file() and str(result) not in self.files:
                self.files.add(str(result))
                def show_result(value, path=result):
                    _text(value)
                    _download(path, "Download completed result")
                self._expansion("Completed recovery outline" if stage == "outline" else "Completed primitive composition",
                                identity + ":result", lambda path=result: read_json_cached(path), show_result)

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


def render_composition_programs(program: list[dict]) -> None:
    """Display authored steps, resolved bindings and resource validation evidence."""
    for index, event in enumerate(program, 1):
        evidence = event.get("composition_validation", {})
        status = evidence.get("feasibility_status", "Historical result — RA composition validation not recorded")
        with ui.expansion(f"{index}. {event.get('outline_id')} · {event.get('resource_jid')} · {status}").classes("w-full"):
            rows = []
            trace = evidence.get("trace", [])
            for step_index, step in enumerate(event.get("primitive_steps", [])):
                validated = trace[step_index] if step_index < len(trace) else {}
                rows.append({"step": step_index + 1, "primitive": step.get("primitive"),
                    "authored": json.dumps(step.get("params", {}), ensure_ascii=False),
                    "resolved": json.dumps(validated.get("resolved_params", "Not recorded"), ensure_ascii=False)})
            ui.table(columns=[{"name": key, "label": label, "field": key, "align": "left"}
                for key, label in (("step", "Step"), ("primitive", "Primitive"),
                                   ("authored", "Authored parameters"), ("resolved", "Resolved parameters"))],
                rows=rows, row_key="step").classes("w-full").props("flat dense wrap-cells")
            ui.label("Achieved postconditions").classes("font-semibold")
            _text(evidence.get("derived_end_state", "Not validated by the current composition validator"))
            ui.label("RA validation findings").classes("font-semibold")
            _text(evidence.get("findings") or evidence.get("reason") or "Not recorded")
            _expansion("Complete RA evidence", f"composition:{event.get('outline_id')}:{evidence.get('evidence_fingerprint')}",
                       lambda evidence=evidence: evidence, _text)


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


def _fields_text(value: object) -> str:
    """State an end state or parameters as compact ``field: value`` pairs."""
    if isinstance(value, dict):
        return "; ".join(f"{key}: {json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else item}"
                         for key, item in value.items())
    return "" if value in (None, "") else str(value)


def _places_text(value: object) -> str:
    """State each other resource's stated changes as ``resource: field: value`` groups."""
    if not isinstance(value, dict):
        return ""
    return " | ".join(f"{place}: {_fields_text(fields)}" for place, fields in value.items())


def _event_rows(events: list, selected: object) -> list[dict]:
    rows = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        rows.append({"row": index, "chosen": "✓" if type(selected) is int and index == selected else "",
                     "event_name": event.get("event_name"),
                     # The outline prompt names resources; older records carry JIDs.
                     "resource": event.get("resource") or event.get("resource_jid"),
                     "part_name": event.get("part_name"),
                     "expected_end_state": _fields_text(event.get("expected_end_state")),
                     "neighbor_end_state": _places_text(event.get("neighbor_end_state") or event.get("place_end_state")),
                     "params": _fields_text(event.get("params")), "rationale": event.get("rationale") or ""})
    return rows


def _event_table(response: object) -> None:
    if not isinstance(response, dict):
        ui.label("Proposed events unavailable: response is not a JSON object.").classes("text-amber-700")
        return
    groups = []
    if isinstance(response.get("candidate_events"), list):
        groups.append(("", response["candidate_events"], response.get("selected_candidate_index")))
    if isinstance(response.get("candidate_traces"), list):
        groups.extend((f"Candidate {index + 1}", candidate.get("events", []), None)
                      for index, candidate in enumerate(response["candidate_traces"])
                      if isinstance(candidate, dict) and isinstance(candidate.get("events"), list))
    if isinstance(response.get("next_transition"), dict):
        suffix = response.get("transition_suffix")
        groups.append(("", [response["next_transition"], *(suffix if isinstance(suffix, list) else [])], None))
    if isinstance(response.get("transition_trace"), list):
        groups.append(("", response["transition_trace"], None))
    if not any(events for _, events, _ in groups):
        ui.label("No events proposed in this turn").classes("text-sm text-slate-500")
        requests = response.get("context_requests")
        if isinstance(requests, list) and requests:
            ui.label("Requested context: " + ", ".join(str(ref) for ref in requests)).classes("text-sm")
        return
    for label, events, selected in groups:
        if label:
            ui.label(label).classes("font-semibold text-sm")
        rows = _event_rows(events, selected)
        columns = [("event_name", "Event"), ("resource", "Resource"), ("part_name", "Part"),
                   ("expected_end_state", "End state"), ("neighbor_end_state", "Neighbor changes"), ("params", "Params"),
                   ("rationale", "Rationale")]
        if type(selected) is int:
            columns.insert(0, ("chosen", "Chosen"))
        if not any(row["neighbor_end_state"] for row in rows):
            columns.remove(("neighbor_end_state", "Neighbor changes"))
        if not any(row["params"] for row in rows):
            columns.remove(("params", "Params"))
        ui.table(columns=[{"name": key, "label": title, "field": key, "align": "left"} for key, title in columns],
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


def _final_output(data: dict) -> dict | None:
    return next((result["payload"] for result in data["results"]
                 if result["payload"].get("decision") == "final_output_ready"), None)


def _named(value: object, names: dict) -> object:
    """Replace resource JIDs with resource names in a recorded value, for display."""
    if isinstance(value, str):
        for jid in sorted(names, key=len, reverse=True):
            value = value.replace(jid, names[jid])
        return value
    if isinstance(value, dict):
        return {key: _named(item, names) for key, item in value.items()}
    if isinstance(value, list):
        return [_named(item, names) for item in value]
    return value


def _steps_table(steps: list, names: dict) -> None:
    rows = [{"step": index, "event_name": step.get("event_name"), "resource": _named(step.get("resource_jid"), names),
             "part_name": step.get("part_name") or "", "expected_end_state": _fields_text(_named(step.get("expected_end_state"), names)),
             "neighbor_end_state": _places_text({_named(place, names): _named(fields, names)
                                              for place, fields in (step.get("neighbor_end_state") or step.get("place_end_state") or {}).items()})}
            for index, step in enumerate(steps, 1) if isinstance(step, dict)]
    columns = [("step", "Step"), ("event_name", "Event"), ("resource", "Resource"), ("part_name", "Part"),
               ("expected_end_state", "End state"), ("neighbor_end_state", "Neighbor changes")]
    if not any(row["neighbor_end_state"] for row in rows):
        columns.remove(("neighbor_end_state", "Neighbor changes"))
    ui.table(columns=[{"name": key, "label": title, "field": key, "align": "left"} for key, title in columns],
             rows=rows, row_key="step").classes("w-full").props("flat dense wrap-cells")


def _render_handoff(final: dict, tab: str) -> None:
    """Show what code did after the outline ended: the accepted events, then the nominal handoff."""
    names = final.get("resource_names") or {}
    ui.label("Handoff to nominal work — no LLM call").classes("font-semibold")
    if tab == "input":
        ui.label("Accepted recovery events").classes("text-sm")
        _steps_table(final.get("transition_trace") or [], names)
        return
    handoff = final.get("nominal_handoff") or {}
    if not handoff:
        ui.label("No handoff recorded for this run.").classes("text-sm text-slate-500")
        return
    if handoff.get("declared_complete_by_llm"):
        ui.label("Ended by the LLM's recovery_complete.").classes("text-sm")
    if handoff.get("complete"):
        ui.label("Nominal work takes over.").classes("text-sm")
        assumed = [row["reason"] for row in handoff.get("assumptions") or [] if row.get("reason")]
        if assumed:
            ui.label("Assumed, not checked: " + "; ".join(_named(assumed, names))).classes("text-sm text-amber-700")
    else:
        reasons = handoff.get("not_connected_reasons") or ["its recorded handoff conditions do not hold"]
        ui.label("Nominal work cannot take over: " + "; ".join(_named(reasons, names))).classes("text-sm text-amber-700")
    steps = handoff.get("nominal_connection") or []
    if steps:
        ui.label("Declared nominal events that connect to nominal work").classes("text-sm")
        _steps_table(steps, names)
    elif handoff.get("complete"):
        ui.label("No nominal events were needed to connect.").classes("text-sm text-slate-500")


def _code_grounding(data: dict) -> dict | None:
    return next((result["payload"]["code_grounding"] for result in data["results"]
                 if isinstance(result["payload"].get("code_grounding"), dict)), None)


def render_turn_content(data: dict, tab: str, *, state_key: str = "") -> None:
    """Separate exact model messages from runtime retrieval and validation."""
    final = _final_output(data)
    if final is not None:
        _render_handoff(final, tab)
        for result in data["results"]:
            _download(result["path"], "Download runtime record")
        return
    grounding = _code_grounding(data)
    if grounding is not None:
        ui.label("Code grounding — no LLM call").classes("font-semibold")
        if tab == "input":
            ui.label("Raw capture facts read by the execution-monitoring rules").classes("text-xs text-slate-500")
            _text([{key: row.get(key) for key in ("signal", "resource_id", "task_id", "part_name", "evidence", "values")
                    if row.get(key) is not None} for row in grounding["fault_event"]["signals"]])
        else:
            _text({key: grounding[key] for key in ("fault_event", "parts", "resources", "expected_state_conflicts")
                   if key in grounding})
        for result in data["results"]:
            _download(result["path"], "Download runtime record")
        return
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
                        "validation_findings", "selected_transition", "transition_validation", "nominal_continuation_evidence",
                        "per_event_results", "primitive_rejection_feedback", "accepted_primitive_event") if key in evidence}
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
                _expansion("RA feasibility", key + ":ra", lambda evidence=evidence: evidence, _text)
            elif role == "CentralControllerAgent":
                _expansion("CCA safety", key + ":cca", lambda evidence=evidence: evidence, _text)
            else:
                _text(stage)


_VERDICT_COLORS = {"FEASIBLE": "positive", "SAFE": "positive", "NONBLOCKING": "positive", "NEEDS_CONTEXT": "warning",
                   "INFEASIBLE": "negative", "UNSAFE": "negative", "BLOCKING": "negative"}
# How the RA derived a changed condition (Alg. 1) when no producing primitive is involved.
_DERIVATIONS = {"free": "free: resource_state names the DES state", "unchanged": "unchanged",
                "bookkeeping": "bookkeeping: not derived", "undeclared": "not a declared variable",
                "place": "a place: decided by capability", "no_producer": "no primitive produces it"}


def turn_checks(draft: dict | None, turn: int) -> dict:
    """Return what the RA and the CCA judged on one outline turn, from the draft's kept check records."""
    found = {}
    for key in ("ra_feasibility_checks", "cca_safety_checks", "cca_nonblocking_checks"):
        rows = [row for row in (draft or {}).get(key) or [] if row.get("turn_index") == turn]
        if rows:
            found[key] = rows[-1]
    return found


def _checks_summary(checks: dict) -> str:
    """State a turn's verdicts in one line: each candidate per check, then what the turn applied."""
    parts = []
    ra = checks.get("ra_feasibility_checks")
    if ra:
        parts.append("RA: " + ", ".join(f"candidate {row['candidate_index'] + 1} {row['feasibility_status']}"
                                        for row in ra["results"]))
    if "cca_safety_checks" in checks:
        parts.append("CCA safety: " + (", ".join(f"candidate {row['candidate_index'] + 1} {row['safety_status']}"
                                                  for row in checks["cca_safety_checks"]["results"])
                                        or "no feasible candidate to check"))
    if ra:
        applied = ra.get("applied_candidate_index")
        parts.append(f"applied candidate {applied + 1}" if type(applied) is int else "no candidate applied")
    nonblocking = checks.get("cca_nonblocking_checks")
    if nonblocking:
        parts.append(f"CCA nonblocking: {nonblocking['nonblocking_status']}")
    return " · ".join(parts)


def _llm_candidates(data: dict) -> list:
    """Return the candidate events of the turn's last LLM response that proposed any, as the LLM named them."""
    for capture in reversed(data["captures"]):
        record = capture["record"]
        if capture["status"] != "captured" or record.get("kind") != "provider_response":
            continue
        for choice in record.get("raw_response", {}).get("choices", []):
            content = choice.get("message", {}).get("content")
            try:
                response = json.loads(content) if isinstance(content, str) else None
            except json.JSONDecodeError:
                continue
            events = response.get("candidate_events") if isinstance(response, dict) else None
            if isinstance(events, list) and events:
                return events
    return []


def _verdict(title: str, status: object) -> None:
    with ui.row().classes("items-center gap-2"):
        ui.label(title).classes("font-semibold text-sm")
        ui.badge(str(status), color=_VERDICT_COLORS.get(status, "grey"))


def _rows_table(columns: list[tuple[str, str]], rows: list[dict]) -> None:
    ui.table(columns=[{"name": key, "label": title, "field": key, "align": "left"} for key, title in columns],
             rows=[{"row": index, **row} for index, row in enumerate(rows)], row_key="row").classes("w-full").props(
        "flat dense wrap-cells")


def _value(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _rounded(value: object) -> object:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def _ra_conditions(conditions: list[dict]) -> None:
    rows = []
    for row in conditions:
        if row.get("derivation") == "producers":
            derivation = "produced by " + ", ".join(f"{producer['primitive']} ({producer['status']})"
                                                     for producer in row.get("producers", []))
        else:
            derivation = _DERIVATIONS.get(row.get("derivation"), str(row.get("derivation")))
        change = _value(row.get("after"))
        if "before" in row:
            change = f"{_value(row['before'])} → {change}"
        rows.append({"entity": row.get("entity"), "condition": row.get("condition"), "change": change,
                     "derivation": derivation, "status": row.get("status"), "reason": row.get("reason", "")})
    _rows_table([("entity", "Entity"), ("condition", "Condition"), ("change", "Before → after"),
                 ("derivation", "Derivation (Alg. 1)"), ("status", "Status"), ("reason", "Reason")], rows)


def _ra_capability(capability: list[dict]) -> None:
    rows = [{"check": row.get("check"), "target": " ".join(str(row[key]) for key in ("part", "place", "slot") if row.get(key)),
             "status": row.get("status"),
             "evidence": row.get("reason") or _fields_text(_rounded({key: row[key] for key in ("witness", "support", "rest_pose")
                                                                     if key in row}))}
            for row in capability]
    _rows_table([("check", "Check"), ("target", "Part or place"), ("status", "Status"),
                 ("evidence", "Reason or witness (Eq. 12)")], rows)


def _cca_safety(result: dict | None) -> None:
    if result is None:
        ui.label("CCA safety (Eq. 13): not checked; the RA refused this candidate.").classes("text-sm text-slate-500")
        return
    _verdict("CCA safety (Eq. 13): every specification monitor can still accept", result["safety_status"])
    if result.get("steps"):
        _rows_table([("step", "Step"), ("status", "Status")], result["steps"])
    for reason in result.get("reasons") or []:
        ui.label(reason).classes("text-sm whitespace-pre-wrap")


def _cca_nonblocking(check: dict) -> None:
    with ui.card().classes("w-full"):
        ui.label("Declared recovery_complete").classes("font-semibold")
        _verdict("CCA nonblocking (Eq. 6): normal production can carry on", check["nonblocking_status"])
        for reason in check.get("reasons") or []:
            ui.label(reason).classes("text-sm whitespace-pre-wrap")
        if check.get("blocked"):
            ui.label("Blocked: " + ", ".join(check["blocked"])).classes("text-sm")
        evidence = check.get("evidence") or {}
        if evidence.get("parts"):
            seconds = evidence.get("seconds")
            ui.label(f"Composition searched to the end: {evidence.get('explored_states')} states"
                     + (f" in {seconds:.1f} s" if isinstance(seconds, (int, float)) else "")
                     + "; parts with work left: " + ", ".join(evidence["parts"])
                     + "; resources: " + ", ".join(evidence.get("included_resources") or [])).classes("text-sm text-slate-500")
        elif "explored_states" in evidence:
            ui.label("No part has work left; nothing to compose.").classes("text-sm text-slate-500")
        ui.label("The CCA accepts the declared completion." if check["nonblocking_status"] == "NONBLOCKING" else
                 "The CCA refuses the declared completion: the reasons return to the LLM and synthesis continues "
                 "from the same state.").classes("font-semibold text-sm")


def render_turn_validation(checks: dict, candidates: list) -> None:
    """Show how the RA and the CCA judged one outline turn: each candidate, then what the turn applied."""
    ra = checks.get("ra_feasibility_checks")
    safety_record = checks.get("cca_safety_checks")
    safety = {row["candidate_index"]: row for row in (safety_record or {}).get("results", [])}
    if ra:
        passes = "RA feasibility and CCA safety" if safety_record is not None else "RA feasibility"
        ui.label(f"The LLM chose candidate {ra['llm_selected_candidate_index'] + 1}. Candidates are checked in the "
                 f"LLM's order, its choice first; the first that passes {passes} is applied.").classes("text-sm")
        for result in ra["results"]:
            index = result["candidate_index"]
            event = candidates[index] if index < len(candidates) and isinstance(candidates[index], dict) else {}
            with ui.card().classes("w-full"):
                ui.label(f"Candidate {index + 1} · {result.get('event_name')} · "
                         f"{event.get('resource') or result.get('resource_jid')}").classes("font-semibold")
                _verdict("RA feasibility (Alg. 1, Eq. 12)", result["feasibility_status"])
                ui.label("Changed conditions: how each is produced").classes("text-sm")
                _ra_conditions(result.get("conditions") or [])
                if result.get("capability"):
                    ui.label("Capability: reach, place and navigation").classes("text-sm")
                    _ra_capability(result["capability"])
                if result.get("primitive_set"):
                    ui.label("Primitive set: " + ", ".join(result["primitive_set"])).classes("text-sm")
                if safety_record is not None:
                    _cca_safety(safety.get(index))
        applied = ra.get("applied_candidate_index")
        ui.label(f"Applied candidate {applied + 1}." if type(applied) is int else
                 "No candidate passed: nothing is applied, and the reasons above return to the LLM as validation "
                 "feedback.").classes("font-semibold text-sm")
    if checks.get("cca_nonblocking_checks"):
        _cca_nonblocking(checks["cca_nonblocking_checks"])
    with ui.expansion("Complete check records").classes("w-full"):
        _text(checks)


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

    def _show_checks(self, row: dict, checks: dict, key: str, directory: Path) -> None:
        """Summarize a turn's RA and CCA verdicts on its card; its validation expansion shows how each was reached."""
        row["checks"] = checks
        row["checks_label"].text = _checks_summary(checks)
        row["checks_label"].set_visibility(bool(checks))
        if not checks:
            return
        if row["validation"] is not None:
            row["validation"]()
            return
        with row["checks_box"]:
            _, row["validation"] = self._expansion("RA and CCA validation", key,
                lambda row=row: (row["checks"], _llm_candidates(read_recovery_turn(directory, row["data"]))),
                lambda value: render_turn_validation(*value))
        row["refresh"].append(row["validation"])

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
                        checks_label = ui.label().classes("text-sm")
                        checks_label.set_visibility(False)
                        row = {"data": turn, "label": label, "signature": None, "refresh": [],
                               "checks": {}, "checks_label": checks_label, "validation": None}
                        for tab in ("input", "output"):
                            state_key = f"{identity}:{json.dumps(key)}:{tab}"
                            _, refresh = self._expansion(tab.title(), state_key,
                                lambda row=row, directory=directory: read_recovery_turn(directory, row["data"]),
                                lambda data, tab=tab, state_key=state_key: render_turn_content(data, tab, state_key=state_key))
                            row["refresh"].append(refresh)
                        row["checks_box"] = ui.column().classes("w-full")
                        self.rows[key] = row
                row = self.rows[key]
                row["label"].text = (f"Turn {turn['turn']} · handoff to nominal work · no LLM call"
                                     if turn["status"] == "final_output_ready" else
                                     f"Turn {turn['turn']} · {turn['phase']} · {turn['status']}")
                if turn.get("resource_jid"):
                    row["label"].text += f" · {turn['resource_jid']}"
                row["data"] = turn
                # The draft keeps each check's verdicts by turn; turns of a run without a draft have none.
                checks = turn_checks(state.get("outline_draft"), turn["turn"]) if stage == "outline" else {}
                if checks != row["checks"]:
                    self._show_checks(row, checks, f"{identity}:{json.dumps(key)}:validation", directory)
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

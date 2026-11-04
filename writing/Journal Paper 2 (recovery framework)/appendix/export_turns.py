"""Export an outline draft or a primitive composition as readable Markdown for Appendix A or B.

Usage, from the repository root:
    poetry run python "writing/Journal Paper 2 (recovery framework)/appendix/export_turns.py" <draft or composition dir> <out.md>

Each turn shows the prompt the model received and its reply. The words are the recorded ones; only line breaks are
restored, one-line JSON facts are indented, and a marker line shows where Claude's prompt cache split the prompt
(`readable_prompt`, the same view as the Recovery page). Outline turns also list what the RA, CCA safety and CCA
nonblocking checks judged. The file ends with the accepted events or the accepted programs.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from cais_spade_llm.ui.recovery_evidence import readable_prompt

FENCE = "````"


def _reply_text(raw: dict) -> str:
    """Return a recorded reply's text: a Claude message's text pieces, or an OpenAI choice's content."""
    if raw.get("type") == "message":
        return "".join(block.get("text", "") for block in raw.get("content", []) if block.get("type") == "text")
    choices = raw.get("choices") or [{}]
    return str((choices[0].get("message") or {}).get("content") or "")


def _readable_reply(text: str) -> str:
    try:
        return readable_prompt(json.dumps(json.loads(text), ensure_ascii=False))
    except json.JSONDecodeError:
        return text


def _exchange(request_path: Path) -> tuple[dict, str, str]:
    """Return a request record, its prompt as text and its reply as text."""
    request = json.loads(request_path.read_text(encoding="utf-8"))
    messages = request["payload"]["messages"]
    prompt = "\n\n".join(readable_prompt(message["content"]) for message in messages)
    response_path = request_path.with_name(request_path.name.replace("_request.json", "_response.json"))
    reply = "(no recorded reply)"
    if response_path.exists():
        reply = _readable_reply(_reply_text(json.loads(response_path.read_text(encoding="utf-8")).get("raw_response") or {}))
    return request, prompt, reply


def _block(text: str) -> list[str]:
    return [f"{FENCE}text", text.rstrip("\n"), FENCE, ""]


def _settings(payload: dict) -> list[str]:
    effort = (payload.get("output_config") or {}).get("effort") or payload.get("reasoning_effort") or "not recorded"
    lines = [f"- Model: `{payload.get('model')}`, reasoning effort `{effort}`"]
    system = payload.get("system")
    if system:
        text = "\n\n".join(piece.get("text", "") for piece in system) if isinstance(system, list) else str(system)
        lines += ["", "System message (reply format), the same on every turn:", "", *_block(readable_prompt(text))]
    return lines


def _checks(draft: dict, turn: int) -> list[str]:
    """List what each check judged on one outline turn."""
    rows = []
    for record in draft.get("ra_feasibility_checks") or []:
        if record.get("turn_index") != turn:
            continue
        for result in record.get("results") or []:
            reason = "; ".join(result.get("reasons") or [])
            rows.append(f"- RA, candidate {result.get('candidate_index', 0) + 1} `{result.get('event_name')}`: "
                        f"{result.get('feasibility_status')}" + (f" — {reason}" if reason else ""))
        applied = record.get("applied_candidate_index")
        rows.append(f"- Applied: candidate {applied + 1}" if applied is not None else "- Applied: none (every candidate refused)")
    for record in draft.get("cca_safety_checks") or []:
        if record.get("turn_index") == turn:
            for result in record.get("results") or []:
                reason = "; ".join(result.get("reasons") or [])
                rows.append(f"- CCA safety, candidate {result.get('candidate_index', 0) + 1}: {result.get('safety_status')}"
                            + (f" — {reason}" if reason else ""))
    for record in draft.get("cca_nonblocking_checks") or []:
        if record.get("turn_index") == turn:
            reason = "; ".join(record.get("reasons") or [])
            rows.append(f"- CCA nonblocking (declared completion): {record.get('nonblocking_status')}" + (f" — {reason}" if reason else ""))
    return rows


def _names(source_path: str) -> dict[str, str]:
    capture = json.loads(Path(source_path).read_text(encoding="utf-8"))
    return {jid: name for name, jid in capture["jids"].items()}


def export_draft(directory: Path) -> list[str]:
    draft = json.loads((directory / "draft.json").read_text(encoding="utf-8"))
    names = _names(draft["source_path"])
    turns = sorted(draft.get("llm_turns") or [], key=lambda row: row["turn"])
    lines = [f"# Recovery outline — {draft['scenario']}", "", f"- Draft: `{directory.name}`",
             f"- Events: {len(draft.get('transition_trace') or [])}"
             + ("" if draft.get("hand_edit") else f"; LLM turns: {len(turns)}")]
    edit = draft.get("hand_edit")
    if edit:
        # A hand-edited reference copy holds only draft.json; its LLM turns are the model outline's it came from.
        lines += [f"- Hand edit (not a model output): {edit.get('label')}. {edit.get('note')}",
                  f"- From the model outline `{edit.get('from_draft')}`; its prompts and replies are in that outline's export.",
                  f"- Joined events: {edit.get('merges')}. Reason: {edit.get('reason')}"]
        turns = []
    for index, turn in enumerate(turns):
        request, prompt, reply = _exchange(directory / turn["request"])
        if index == 0:
            lines += _settings(request["payload"])
        lines += ["", f"## Turn {turn['turn']} ({turn.get('stage', 'outline')})", "", "### Prompt", "", *_block(prompt),
                  "### Reply", "", *_block(reply)]
        checks = _checks(draft, turn["turn"])
        if checks:
            lines += ["### Checks", "", *checks]
    source = " | From model event" if edit else ""
    lines += ["", "## Accepted events", "", f"| # | Event | Resource | Part | End state{source} |",
              "|---|---|---|---|---|" + ("---|" if edit else "")]
    for number, event in enumerate(draft.get("transition_trace") or [], 1):
        end = json.dumps({key: names.get(value, value) if isinstance(value, str) else value
                          for key, value in (event.get("expected_end_state") or {}).items()}, ensure_ascii=False)
        lines.append(f"| {number} | `{event.get('event_name')}` | {names.get(event['resource_jid'], event['resource_jid'])} | "
                     f"{event.get('part_name') or '—'} | `{end}` |" + (f" {event.get('source_outline_id', '')} |" if edit else ""))
    return lines


def export_composition(directory: Path) -> list[str]:
    composition = json.loads((directory / "composition.json").read_text(encoding="utf-8"))
    requests = sorted(directory.glob("recovery_primitves/requests/*/round*_request.json"),
                      key=lambda path: json.loads(path.read_text(encoding="utf-8")).get("recorded_at", ""))
    draft_id = composition.get("source_outline_draft", "")
    source = directory.parents[1] / "outline_drafts" / draft_id / "draft.json"
    names = _names(json.loads(source.read_text(encoding="utf-8"))["source_path"]) if source.exists() else {}
    programs = composition.get("accepted_primitive_program") or []
    lines = [f"# Primitive composition — `{directory.name}`", "", f"- Outline: `{draft_id}`",
             f"- Events composed: {len(programs)}; LLM calls: {len(requests)}"]
    for index, path in enumerate(requests):
        request, prompt, reply = _exchange(path)
        if index == 0:
            lines += _settings(request["payload"])
        active = re.search(r"Active DES Transition \(token\)\n.*?(RECOVERY_SEQ\d+)", prompt, re.S)
        lines += ["", f"## Call {index + 1} — {active[1] if active else 'event not named'} (turn {request.get('turn')})", "",
                  "### Prompt", "", *_block(prompt), "### Reply", "", *_block(reply)]
    lines += ["", "## Accepted programs", ""]
    for row in programs:
        steps = "\n".join(f"{number}. {step['primitive']} {json.dumps(step.get('params') or {}, ensure_ascii=False)}"
                          for number, step in enumerate(row.get("primitive_steps") or [], 1))
        lines += [f"### {row['outline_id']} `{row.get('event_name')}` — {names.get(row['resource_jid'], row['resource_jid'])}", "",
                  *_block(readable_prompt(steps))]
    return lines


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        raise SystemExit(__doc__)
    directory, output = Path(argv[1]).resolve(), Path(argv[2])
    lines: list[Any] = export_draft(directory) if (directory / "draft.json").exists() else export_composition(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(lines)} lines)")


if __name__ == "__main__":
    main(sys.argv)

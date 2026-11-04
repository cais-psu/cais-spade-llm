"""Build the paper's appendix tables and supplementary material from the recorded recovery runs.

The appendix set (`appendix_set.json`) names, per scenario, the model outline, an optional hand-edited reference
outline, and the composition. `build_paper_export` writes two zips:

- `appendix_overleaf.zip`: LaTeX for the appendix (prompt template, each scenario's turn-1 facts, turn table,
  events, reference edit and composition, a cross-scenario summary and one worked example), for `\\input` in Overleaf;
- `supplementary_material.zip`: every prompt and reply as readable Markdown, the raw records, and a README.

Prompts are the recorded words, shown with `readable_prompt`; nothing here calls a model or changes a record.
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cais_spade_llm.ui.recovery_evidence import CACHE_POINT, readable_prompt

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "cais_spade_llm/monitor/environment_runs/recovery"
APPENDIX = ROOT / "writing/Journal Paper 2 (recovery framework)/appendix"
MANIFEST = APPENDIX / "appendix_set.json"
BUILD = APPENDIX / "build"
FENCE = "````"
# Section headings of the outline prompt, in the words the prompt uses.
SECTIONS = ("Task and Role", "Failure", "Recovery Goal", "Candidate Rules", "Knowledge Base", "Retrieved context",
            "Recovery Events So Far (planned, not executed)", "Validation Feedback", "Earlier Refused Completions",
            "Resources", "Parts", "Retrieved current values", "Response Format")
_ASCII = {"→": "->", "←": "<-", "—": "--", "–": "-", "×": "x", "≤": "<=", "≥": ">=", "°": " deg", "“": '"', "”": '"',
          "‘": "'", "’": "'", "…": "...", "·": "-", "─": "-", "∩": "and", "θ": "theta", "φ": "phi"}


# Reading the records.

def reply_text(raw: dict) -> str:
    """Return a recorded reply's text: a Claude message's text pieces, or an OpenAI choice's content."""
    if raw.get("type") == "message":
        return "".join(block.get("text", "") for block in raw.get("content", []) if block.get("type") == "text")
    choices = raw.get("choices") or [{}]
    return str((choices[0].get("message") or {}).get("content") or "")


def exchange(request_path: Path) -> tuple[dict, str, str, dict | None]:
    """Return a request record, its prompt as text, its reply as text and the reply parsed, when it is JSON."""
    request = json.loads(request_path.read_text(encoding="utf-8"))
    prompt = "\n\n".join(readable_prompt(message["content"]) for message in request["payload"]["messages"])
    response_path = request_path.with_name(request_path.name.replace("_request.json", "_response.json"))
    text, parsed = "(no recorded reply)", None
    if response_path.exists():
        text = reply_text(json.loads(response_path.read_text(encoding="utf-8")).get("raw_response") or {})
        # A reply may arrive inside a ```json fence; the framework reads the object inside it.
        fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, re.S)
        try:
            parsed = json.loads(fenced[1] if fenced else text)
            text = readable_prompt(json.dumps(parsed, ensure_ascii=False))
        except json.JSONDecodeError:
            pass
    return request, prompt, text, parsed


def names_of(source_path: str) -> dict[str, str]:
    """Return resource names by JID, from the capture an outline was planned on."""
    capture = json.loads(Path(source_path).read_text(encoding="utf-8"))
    return {jid: name for name, jid in capture["jids"].items()}


def outline_turns(directory: Path) -> list[dict]:
    draft = json.loads((directory / "draft.json").read_text(encoding="utf-8"))
    return sorted(draft.get("llm_turns") or [], key=lambda row: row["turn"])


def composition_requests(directory: Path) -> list[Path]:
    return sorted(directory.glob("recovery_primitves/requests/*/round*_request.json"),
                  key=lambda path: json.loads(path.read_text(encoding="utf-8")).get("recorded_at", ""))


def active_event(prompt: str) -> str | None:
    found = re.search(r"Active DES Transition \(token\)\n.*?(RECOVERY_SEQ\d+)", prompt, re.S)
    return found[1] if found else None


def sections(prompt: str) -> dict[str, str]:
    """Split an outline prompt into its sections by their headings; the cache marker is dropped."""
    found, current, lines = {}, None, []
    for line in prompt.replace(CACHE_POINT, "").split("\n"):
        if line.strip() in SECTIONS:
            if current:
                found[current] = "\n".join(lines).strip("\n")
            current, lines = line.strip(), []
        elif current:
            lines.append(line)
    if current:
        found[current] = "\n".join(lines).strip("\n")
    return found


def turn_checks(draft: dict, turn: int) -> list[str]:
    """Return what each check judged on one outline turn, in the checks' own words."""
    rows, outcome = [], None
    for record in draft.get("ra_feasibility_checks") or []:
        if record.get("turn_index") != turn:
            continue
        for result in record.get("results") or []:
            reason = "; ".join(result.get("reasons") or [])
            rows.append(f"RA, candidate {result.get('candidate_index', 0) + 1} `{result.get('event_name')}`: "
                        f"{result.get('feasibility_status')}" + (f" — {reason}" if reason else ""))
        applied = record.get("applied_candidate_index")
        outcome = f"Applied: candidate {applied + 1}" if applied is not None else "Applied: none (every candidate refused)"
    for record in draft.get("cca_safety_checks") or []:
        if record.get("turn_index") == turn:
            for result in record.get("results") or []:
                reason = "; ".join(result.get("reasons") or [])
                rows.append(f"CCA safety, candidate {result.get('candidate_index', 0) + 1}: {result.get('safety_status')}"
                            + (f" — {reason}" if reason else ""))
    for record in draft.get("cca_nonblocking_checks") or []:
        if record.get("turn_index") == turn:
            reason = "; ".join(record.get("reasons") or [])
            rows.append(f"CCA nonblocking (declared completion): {record.get('nonblocking_status')}" + (f" — {reason}" if reason else ""))
    return rows + ([outcome] if outcome else [])


def tokens(paths: list[Path]) -> int:
    """Sum the recorded token usage of the replies beside these requests."""
    total = 0
    for path in paths:
        response = path.with_name(path.name.replace("_request.json", "_response.json"))
        if response.exists():
            usage = (json.loads(response.read_text(encoding="utf-8")).get("raw_response") or {}).get("usage") or {}
            total += sum(usage.get(key) or 0 for key in ("input_tokens", "cache_read_input_tokens",
                                                       "cache_creation_input_tokens", "output_tokens"))
    return total


# Markdown transcripts.

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


def markdown_outline(directory: Path) -> str:
    """Return an outline draft as Markdown: every turn's prompt, reply and checks, then the accepted events."""
    draft = json.loads((directory / "draft.json").read_text(encoding="utf-8"))
    names = names_of(draft["source_path"])
    edit = draft.get("hand_edit")
    turns = [] if edit else outline_turns(directory)
    lines = [f"# Recovery outline — {draft['scenario']}", "", f"- Draft: `{directory.name}`",
             f"- Events: {len(draft.get('transition_trace') or [])}" + ("" if edit else f"; LLM turns: {len(turns)}")]
    if edit:
        # A hand-edited reference copy holds only draft.json; its LLM turns are the model outline's it came from.
        lines += [f"- Hand edit (not a model output): {edit.get('label')}. {edit.get('note')}",
                  f"- From the model outline `{edit.get('from_draft')}`; its prompts and replies are in that outline's export.",
                  f"- Joined events: {edit.get('merges')}. Reason: {edit.get('reason')}"]
    for index, turn in enumerate(turns):
        request, prompt, reply, _ = exchange(directory / turn["request"])
        if index == 0:
            lines += _settings(request["payload"])
        lines += ["", f"## Turn {turn['turn']} ({turn.get('stage', 'outline')})", "", "### Prompt", "", *_block(prompt),
                  "### Reply", "", *_block(reply)]
        checks = turn_checks(draft, turn["turn"])
        if checks:
            lines += ["### Checks", "", *(f"- {row}" for row in checks)]
    lines += ["", "## Accepted events", "", "| # | Event | Resource | Part | End state" + (" | From model event" if edit else "") + " |",
              "|---|---|---|---|---|" + ("---|" if edit else "")]
    for number, event in enumerate(draft.get("transition_trace") or [], 1):
        end = json.dumps(_named(event.get("expected_end_state") or {}, names), ensure_ascii=False)
        lines.append(f"| {number} | `{event.get('event_name')}` | {names.get(event['resource_jid'], event['resource_jid'])} | "
                     f"{event.get('part_name') or '—'} | `{end}` |" + (f" {event.get('source_outline_id', '')} |" if edit else ""))
    return "\n".join(lines) + "\n"


def markdown_composition(directory: Path) -> str:
    """Return a composition as Markdown: every call's prompt and reply, then the accepted programs."""
    composition = json.loads((directory / "composition.json").read_text(encoding="utf-8"))
    requests = composition_requests(directory)
    names = _composition_names(composition)
    programs = composition.get("accepted_primitive_program") or []
    lines = [f"# Primitive composition — `{directory.name}`", "", f"- Outline: `{composition.get('source_outline_draft')}`",
             f"- Events composed: {len(programs)}; LLM calls: {len(requests)}"]
    for index, path in enumerate(requests):
        request, prompt, reply, _ = exchange(path)
        if index == 0:
            lines += _settings(request["payload"])
        lines += ["", f"## Call {index + 1} — {active_event(prompt) or 'event not named'} (turn {request.get('turn')})", "",
                  "### Prompt", "", *_block(prompt), "### Reply", "", *_block(reply)]
    lines += ["", "## Accepted programs", ""]
    for row in programs:
        steps = "\n".join(f"{number}. {step['primitive']} {json.dumps(step.get('params') or {}, ensure_ascii=False)}"
                          for number, step in enumerate(row.get("primitive_steps") or [], 1))
        lines += [f"### {row['outline_id']} `{row.get('event_name')}` — {names.get(row['resource_jid'], row['resource_jid'])}", "",
                  *_block(readable_prompt(steps))]
    return "\n".join(lines) + "\n"


def _named(values: dict, names: dict[str, str]) -> dict:
    return {key: names.get(value, value) if isinstance(value, str) else value for key, value in values.items()}


def _composition_names(composition: dict) -> dict[str, str]:
    source = RECORDS / "outline_drafts" / str(composition.get("source_outline_draft")) / "draft.json"
    return names_of(json.loads(source.read_text(encoding="utf-8"))["source_path"]) if source.exists() else {}


# LaTeX for the appendix.

def tex(text: Any) -> str:
    """Escape text for a LaTeX paragraph or table cell."""
    text = "".join(_ASCII.get(character, character) for character in str(text))
    for plain, escaped in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                           ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        text = text.replace(plain, escaped)
    return text.replace(r"\textbackslash\{\}", r"\textbackslash{}").replace(r"\textasciitilde\{\}", r"\textasciitilde{}") \
               .replace(r"\textasciicircum\{\}", r"\textasciicircum{}")


def listing(text: str) -> str:
    """Return text as a listing; characters pdfLaTeX listings cannot set are written in ASCII."""
    body = "".join(_ASCII.get(character, character) for character in text.strip("\n"))
    return "\\begin{lstlisting}\n" + body + "\n\\end{lstlisting}\n"


def _longtable(columns: str, header: list[str], rows: list[list[str]]) -> str:
    head = " & ".join(f"\\textbf{{{cell}}}" for cell in header) + r" \\ \midrule"
    lines = [f"\\begin{{longtable}}{{{columns}}}", r"\toprule", head, r"\endfirsthead", r"\toprule", head, r"\endhead",
             *(" & ".join(row) + r" \\" for row in rows), r"\bottomrule", r"\end{longtable}", ""]
    return "\n".join(lines)


def _short(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def outline_tex(entry: dict) -> str:
    """Return one scenario's outline: turn-1 facts, the turn table, the accepted events and the reference edit."""
    directory = RECORDS / "outline_drafts" / entry["outline"]
    draft = json.loads((directory / "draft.json").read_text(encoding="utf-8"))
    names = names_of(draft["source_path"])
    turns = outline_turns(directory)
    first = sections(exchange(directory / turns[0]["request"])[1])
    facts = "\n\n".join(f"{name}\n{first[name]}" for name in ("Failure", "Recovery Goal", "Resources", "Parts") if name in first)
    out = [f"\\subsection{{{tex(entry['scenario'])}}}\\label{{app:{entry['key']}}}", "",
           f"Model outline \\texttt{{{tex(entry['outline'][:14])}}}: {len(draft.get('transition_trace') or [])} events, "
           f"{len(turns)} turns, {tokens([directory / turn['request'] for turn in turns]):,} tokens.", "",
           "\\paragraph{Facts at the failure (turn~1).}", listing(facts), "\\paragraph{Turns.}"]
    rows = []
    for turn in turns:
        _, _, _, reply = exchange(directory / turn["request"])
        reply = reply or {}
        if reply.get("decision") == "need_context":
            action = "looked up " + ", ".join(reply.get("context_requests") or [])
            candidates = ""
        elif reply.get("decision") == "recovery_complete":
            action, candidates = "declared recovery complete", ""
        elif not reply.get("candidate_events"):
            action, candidates = f"decision {reply.get('decision')}" if reply.get("decision") else "no readable reply", ""
        else:
            events = reply.get("candidate_events") or []
            action = f"proposed {len(events)}"
            candidates = "; ".join(f"{number}. {event.get('event_name')}" for number, event in enumerate(events, 1))
        checks = [tex(_short(row.replace("`", ""))) for row in turn_checks(draft, turn["turn"])]
        rows.append([str(turn["turn"]), tex(action), tex(candidates).replace("; ", r" \newline "),
                     r" \newline ".join(checks) or "--"])
    out.append(_longtable(r"@{}r p{0.17\linewidth} p{0.27\linewidth} p{0.45\linewidth}@{}",
                          ["Turn", "Model", "Candidates", "RA and CCA"], rows))
    out.append("\\paragraph{Accepted events.}")
    out.append(_events_table(draft["transition_trace"], names))
    if entry.get("reference"):
        out.append(reference_tex(entry, names))
    return "\n".join(out)


def _events_table(events: list[dict], names: dict[str, str], *, source: bool = False) -> str:
    rows = []
    for number, event in enumerate(events, 1):
        end = "; ".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in
                        _named(event.get("expected_end_state") or {}, names).items())
        row = [str(number), tex(event.get("event_name")), tex(names.get(event["resource_jid"], event["resource_jid"])),
               tex(event.get("part_name") or "--"), tex(end)]
        if source:
            row.append(tex(event.get("source_outline_id", "")))
        rows.append(row)
    if source:
        return _longtable(r"@{}r p{0.2\linewidth} p{0.08\linewidth} p{0.12\linewidth} p{0.33\linewidth} p{0.13\linewidth}@{}",
                          ["\\#", "Event", "Resource", "Part", "End state", "From model event"], rows)
    return _longtable(r"@{}r p{0.22\linewidth} p{0.09\linewidth} p{0.14\linewidth} p{0.45\linewidth}@{}",
                      ["\\#", "Event", "Resource", "Part", "End state"], rows)


def reference_tex(entry: dict, names: dict[str, str]) -> str:
    draft = json.loads((RECORDS / "outline_drafts" / entry["reference"] / "draft.json").read_text(encoding="utf-8"))
    edit = draft["hand_edit"]
    return "\n".join([
        "\\paragraph{Reference outline for composition (hand edit, not a model output).}",
        f"{tex(edit.get('label'))}. {tex(edit.get('reason'))}. Every value is the model's; joined events: "
        f"{tex(', '.join('+'.join(group) for group in edit.get('merges') or []))}.", "",
        _events_table(draft["transition_trace"], names, source=True)])


def composition_tex(entry: dict) -> str:
    """Return one scenario's composition: per event, its primitive sequence and the calls it took."""
    directory = RECORDS / "draft_compositions" / entry["composition"]
    composition = json.loads((directory / "composition.json").read_text(encoding="utf-8"))
    names = _composition_names(composition)
    requests = composition_requests(directory)
    calls: dict[str, int] = {}
    for path in requests:
        event = active_event(exchange(path)[1]) or "?"
        calls[event] = calls.get(event, 0) + 1
    rows = []
    for row in composition["accepted_primitive_program"]:
        steps = [step["primitive"] for step in row.get("primitive_steps") or []]
        rows.append([tex(row["outline_id"].replace("RECOVERY_SEQ", "")), tex(row.get("event_name")),
                     tex(names.get(row["resource_jid"], row["resource_jid"])), tex(", ".join(steps)),
                     str(calls.get(row["outline_id"], 0))])
    return "\n".join([
        f"\\subsection{{{tex(entry['scenario'])}: primitive composition}}", "",
        f"Composition \\texttt{{{tex(entry['composition'][:20])}}} on outline \\texttt{{{tex(composition['source_outline_draft'][:14])}}}: "
        f"{len(rows)} events, {len(requests)} calls, {tokens(requests):,} tokens.", "",
        _longtable(r"@{}r p{0.24\linewidth} p{0.08\linewidth} p{0.5\linewidth} r@{}",
                   ["\\#", "Event", "Resource", "Primitive steps", "Calls"], rows)])


def summary_tex(entries: list[dict]) -> str:
    rows = []
    for entry in entries:
        outline = RECORDS / "outline_drafts" / entry["outline"]
        draft = json.loads((outline / "draft.json").read_text(encoding="utf-8"))
        turns = outline_turns(outline)
        events = str(len(draft["transition_trace"]))
        if entry.get("reference"):
            reference = json.loads((RECORDS / "outline_drafts" / entry["reference"] / "draft.json").read_text(encoding="utf-8"))
            events += f" ({len(reference['transition_trace'])})"
        requests = composition_requests(RECORDS / "draft_compositions" / entry["composition"])
        rows.append([tex(entry["scenario"]), events, str(len(turns)), f"{tokens([outline / turn['request'] for turn in turns]):,}",
                     str(len(requests)), f"{tokens(requests):,}"])
    return "\n".join(["\\begin{table}[ht]\\centering\\small", "\\caption{Reference runs: outline events (reference outline "
                      "in parentheses), LLM turns and calls, and tokens.}\\label{tab:appendix-summary}",
                      "\\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
                      r"Scenario & Events & Outline turns & Outline tokens & Composition calls & Composition tokens \\ \midrule",
                      *(" & ".join(row) + r" \\" for row in rows), r"\bottomrule", "\\end{tabular}", "\\end{table}", ""])


def template_tex(entry: dict) -> str:
    """Return the outline prompt's fixed sections once, with the scenario facts left as placeholders."""
    directory = RECORDS / "outline_drafts" / entry["outline"]
    request, prompt, _, _ = exchange(directory / outline_turns(directory)[0]["request"])
    found = sections(prompt)
    parts = []
    for name in ("Task and Role", "Failure", "Recovery Goal", "Candidate Rules", "Knowledge Base", "Resources", "Parts"):
        if name in ("Failure", "Recovery Goal", "Resources", "Parts"):
            parts.append(f"{name}\n<the scenario's {name.lower()} lines, as in its turn-1 facts>")
        elif name in found:
            parts.append(f"{name}\n{found[name]}")
    system = request["payload"].get("system") or []
    reply_format = "\n\n".join(piece.get("text", "") for piece in system) if isinstance(system, list) else str(system)
    return "\n".join(["\\subsection{Outline prompt}\\label{app:prompt}", "",
                      "The fixed sections are the same words in every scenario and turn; the scenario's facts fill the "
                      "placeholders, and later turns add the events so far, the check feedback and the facts looked up.", "",
                      listing("\n\n".join(parts)), "\\paragraph{Reply format (system message).}",
                      listing(readable_prompt(reply_format))])


def worked_example_tex(entry: dict, turns: list[int]) -> str:
    """Return consecutive turns of one outline: each reply's candidates, the checks, and the feedback the next turn showed."""
    directory = RECORDS / "outline_drafts" / entry["outline"]
    draft = json.loads((directory / "draft.json").read_text(encoding="utf-8"))
    by_turn = {turn["turn"]: turn for turn in outline_turns(directory)}
    out = [f"\\subsection{{Worked example: {tex(entry['scenario'])}, turns {turns[0]}--{turns[-1]}}}\\label{{app:worked}}", ""]
    for number in turns:
        _, prompt, _, reply = exchange(directory / by_turn[number]["request"])
        feedback = sections(prompt).get("Validation Feedback")
        out.append(f"\\paragraph{{Turn {number}.}}")
        if feedback:
            out += ["Feedback shown in the prompt:", listing(feedback)]
        reply = reply or {}
        if reply.get("candidate_events"):
            text = [f"plan: {reply.get('plan', '')}"] + [
                f"{index}. {event.get('event_name')} ({event.get('resource')}): {event.get('rationale', '')}"
                for index, event in enumerate(reply["candidate_events"], 1)]
            out += ["Reply (candidates, most preferred first):", listing("\n".join(text))]
        checks = turn_checks(draft, number)
        if checks:
            out += ["Checks:", "\\begin{itemize}", *(f"\\item {tex(row.replace('`', ''))}" for row in checks), "\\end{itemize}", ""]
    return "\n".join(out)


PREAMBLE = r"""% Packages the appendix files need; add to the main preamble.
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{listings}
\lstset{basicstyle=\ttfamily\scriptsize, breaklines=true, breakatwhitespace=false, columns=fullflexible,
        keepspaces=true, frame=single, framesep=2pt, xleftmargin=2pt, xrightmargin=2pt}
"""


# Building the two zips.

def read_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_paper_export(manifest: dict | None = None, build: Path = BUILD) -> tuple[Path, Path]:
    """Write the Overleaf appendix files and the supplementary material, each as a folder and a zip.

    Returns:
        The two zip paths: appendix_overleaf.zip and supplementary_material.zip.
    """
    manifest = manifest or read_manifest()
    entries = manifest["scenarios"]
    overleaf, supplement = build / "appendix_overleaf", build / "supplementary_material"
    for folder in (overleaf, supplement):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    worked = next(entry for entry in entries if entry["key"] == manifest["worked_example"]["key"])
    files = {"appendix_preamble.tex": PREAMBLE, "appendix_summary.tex": summary_tex(entries),
             "appendix_prompt_template.tex": template_tex(entries[0]),
             "appendix_worked_example.tex": worked_example_tex(worked, manifest["worked_example"]["turns"])}
    for entry in entries:
        files[f"{entry['key']}_outline.tex"] = outline_tex(entry)
        files[f"{entry['key']}_composition.tex"] = composition_tex(entry)
    order = ["appendix_summary", "appendix_prompt_template", *(f"{entry['key']}_outline" for entry in entries),
             "appendix_worked_example", *(f"{entry['key']}_composition" for entry in entries)]
    files["appendix_all.tex"] = "\n".join(f"\\input{{appendix/{name}}}" for name in order) + "\n"
    files["README.txt"] = (
        "Upload this folder to Overleaf as appendix/ (Upload > Zip). Add appendix_preamble.tex's packages to the main\n"
        "preamble, then \\input{appendix/<file>} where each part belongs, or \\input{appendix/appendix_all} for all of it.\n"
        f"Built {datetime.now(timezone.utc).isoformat(timespec='seconds')} from the records named in appendix_set.json.\n")
    for name, text in files.items():
        (overleaf / name).write_text(text, encoding="utf-8")
    _supplement(manifest, supplement)
    zips = []
    for folder in (overleaf, supplement):
        archive = folder.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    bundle.write(path, (Path("appendix") if folder == overleaf else Path(folder.name)) / path.relative_to(folder))
        zips.append(archive)
    return zips[0], zips[1]


def _supplement(manifest: dict, folder: Path) -> None:
    transcripts, records = folder / "transcripts", folder / "records"
    rows = []
    for entry in manifest["scenarios"]:
        outline = RECORDS / "outline_drafts" / entry["outline"]
        composition = RECORDS / "draft_compositions" / entry["composition"]
        (transcripts / entry["key"]).mkdir(parents=True)
        (transcripts / entry["key"] / "outline.md").write_text(markdown_outline(outline), encoding="utf-8")
        (transcripts / entry["key"] / "composition.md").write_text(markdown_composition(composition), encoding="utf-8")
        _copy(outline, records / entry["key"] / "outline", ["draft.json", "audit.json"], "recovery_outline/requests")
        _copy(composition, records / entry["key"] / "composition", ["composition.json", "composition_source.json"],
              "recovery_primitves/requests")
        if entry.get("reference"):
            reference = RECORDS / "outline_drafts" / entry["reference"]
            (transcripts / entry["key"] / "reference_outline.md").write_text(markdown_outline(reference), encoding="utf-8")
            _copy(reference, records / entry["key"] / "reference_outline", ["draft.json", "audit.json"], None)
        rows.append(f"| {entry['scenario']} | `{entry['outline']}` | `{entry.get('reference') or '—'}` | `{entry['composition']}` |")
    (folder / "README.md").write_text("\n".join([
        "# Supplementary material: recovery outlines and primitive compositions", "",
        f"Model: {manifest.get('model')}. Outlines from commit `{manifest.get('outline_commit')}`, compositions from "
        f"commit `{manifest.get('composition_commit')}`.", "",
        "| Scenario | Model outline | Reference outline (hand edit) | Composition |", "|---|---|---|---|", *rows, "",
        "- `transcripts/<scenario>/`: every prompt and reply as readable text. Prompts are the recorded words; only line",
        "  breaks are restored, one-line JSON facts are indented, and a marker shows where the prompt cache split them.",
        "- `records/<scenario>/`: the raw records. `requests/<call>/round00_attempt01_request.json` is the exact payload",
        "  sent and `..._response.json` the reply received; `draft.json` holds the accepted events and every check result;",
        "  `composition.json` the accepted programs with their resolved parameters and check evidence.",
        "- Reference outlines are hand edits of the model outline for composition (repeated moves removed, every value the",
        "  model's); they are not model outputs.", ""]), encoding="utf-8")


def _copy(source: Path, target: Path, names: list[str], requests: str | None) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    if requests and (source / requests).exists():
        shutil.copytree(source / requests, target / "requests")

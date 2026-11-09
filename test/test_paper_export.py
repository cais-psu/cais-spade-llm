"""The paper export reads recorded runs as they are and writes the appendix and supplementary zips."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from cais_spade_llm.ui import paper_export

PROMPT = ("Task and Role\nYou are the recovery planner.\n\nFailure\n- ur5e-1 did not complete pick_grasp\n\nRecovery Goal\n"
          "- KET4_Square_4mm: part_location Buffer For Machined parts\n\nCandidate Rules\n- One event per candidate.\n\n"
          "Knowledge Base\n- /regions: area bounds\n")
FACTS = "\n\nResources\nM1 (resource)  idle; capabilities: trim (circle, square)\n\nParts\nKET4_Square_4mm  held by ur5e-1\n"


def _call(directory: Path, call: str, turn: int, prompt: list, reply: str, recorded_at: str) -> str:
    folder = directory / "requests" / call
    folder.mkdir(parents=True)
    (folder / "round00_attempt01_request.json").write_text(json.dumps({
        "turn": turn, "recorded_at": recorded_at, "kind": "provider_request",
        "payload": {"model": "claude-opus-5-5", "output_config": {"effort": "medium"},
                    "system": [{"type": "text", "text": "Reply with only one JSON object."}],
                    "messages": [{"role": "user", "content": prompt}]}}))
    (folder / "round00_attempt01_response.json").write_text(json.dumps({"raw_response": {
        "type": "message", "content": [{"type": "text", "text": reply}],
        "usage": {"input_tokens": 100, "cache_read_input_tokens": 50, "output_tokens": 10}}}))
    return str((folder / "round00_attempt01_request.json").relative_to(directory.parent))


@pytest.fixture
def records(tmp_path, monkeypatch):
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps({"jids": {"ur5e-1": "r1@localhost", "KMR": "kmr@localhost"}}))
    drafts = tmp_path / "outline_drafts" / "draft_one"
    (drafts / "recovery_outline").mkdir(parents=True)
    pieces = [{"type": "text", "text": PROMPT, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": FACTS}]
    first = _call(drafts / "recovery_outline", "a", 1, pieces,
                  '```json\n{"plan": "Look first.", "decision": "need_context", "context_requests": ["/regions"]}\n```', "1")
    second = _call(drafts / "recovery_outline", "b", 2, pieces, json.dumps({"plan": "Hand off.", "candidate_events": [
        {"event_name": "kmr_pick_part_from_M1", "resource": "KMR", "rationale": "M1 holds it."},
        {"event_name": "kmr_travel_to_M1", "resource": "KMR", "rationale": "M1 is reachable."}]}), "2")
    (drafts / "draft.json").write_text(json.dumps({
        "scenario": "Conveyor breakdown", "source_path": str(capture),
        "llm_turns": [{"turn": 2, "stage": "outline", "request": second}, {"turn": 1, "stage": "outline", "request": first}],
        "ra_feasibility_checks": [{"turn_index": 2, "applied_candidate_index": 1, "results": [
            {"candidate_index": 0, "event_name": "kmr_pick_part_from_M1", "feasibility_status": "FEASIBLE"},
            {"candidate_index": 1, "event_name": "kmr_travel_to_M1", "feasibility_status": "FEASIBLE"}]}],
        "cca_safety_checks": [{"turn_index": 2, "results": [
            {"candidate_index": 0, "safety_status": "UNSAFE", "reasons": ["SAFE_M1_mutex is broken: 50% & #1"]},
            {"candidate_index": 1, "safety_status": "SAFE"}]}],
        "transition_trace": [{"event_name": "kmr_travel_to_M1", "resource_jid": "kmr@localhost", "part_name": None,
                              "expected_end_state": {"resource_location": "M1", "held_part": "r1@localhost"}}]}))
    composition = tmp_path / "draft_compositions" / "composition_one"
    (composition / "recovery_primitves").mkdir(parents=True)
    _call(composition / "recovery_primitves", "c", 1,
          [{"type": "text", "text": "Active DES Transition (token)\nRECOVERY_SEQ1 kmr_travel_to_M1"}], '{"primitive_steps": []}', "3")
    (composition / "composition.json").write_text(json.dumps({"source_outline_draft": "draft_one", "accepted_primitive_program": [
        {"outline_id": "RECOVERY_SEQ1", "event_name": "kmr_travel_to_M1", "resource_jid": "kmr@localhost",
         "primitive_steps": [{"primitive": "move_base", "params": {"target_pose": [-4.7, 2.2, 1.57]}}]}]}))
    monkeypatch.setattr(paper_export, "RECORDS", tmp_path)
    return {"model": "claude-opus-5-5", "outline_commit": "a", "composition_commit": "b",
            "scenarios": [{"key": "conveyor", "model": "Claude Opus 5.5", "scenario": "Conveyor breakdown", "outline": "draft_one",
                           "composition": "composition_one"}],
            "worked_example": {"key": "conveyor", "turns": [2]}}


def test_a_transcript_shows_each_turn_s_prompt_reply_checks_and_the_events_by_name(records, tmp_path):
    text = paper_export.markdown_outline(tmp_path / "outline_drafts" / "draft_one")
    # Turns in order; a fenced reply is read as the framework reads it.
    assert text.index("## Turn 1") < text.index("## Turn 2")
    assert '"decision": "need_context"' in text and "```json" not in text
    # The checks of a turn, applied last; the events by resource name, never by JID.
    checks = text.split("### Checks", 1)[1]
    assert checks.index("CCA safety, candidate 1: UNSAFE") < checks.index("Applied: candidate 2")
    assert '| `kmr_travel_to_M1` | KMR | — | `{"resource_location": "M1", "held_part": "ur5e-1"}` |' in text


def test_the_paper_export_writes_ascii_latex_for_overleaf_and_the_supplementary_records(records, tmp_path):
    overleaf, supplement = paper_export.build_paper_export(records, tmp_path / "build")
    with zipfile.ZipFile(overleaf) as bundle:
        files = {name: bundle.read(name).decode() for name in bundle.namelist()}
    assert {"appendix/appendix_all.tex", "appendix/conveyor_outline.tex", "appendix/conveyor_composition.tex",
            "appendix/appendix_worked_example.tex", "appendix/appendix_prompt_template.tex"} <= set(files)
    assert all(text.isascii() for text in files.values())
    outline = files["appendix/conveyor_outline.tex"]
    # Each scenario names its LLM, in the outline, the composition and the summary table.
    assert r"\texttt{draft\_one} (Claude Opus 5.5): 1 events" in outline
    assert r"(Claude Opus 5.5) on outline" in files["appendix/conveyor_composition.tex"]
    assert r"Conveyor breakdown & Claude Opus 5.5 & 1 & 2" in files["appendix/appendix_summary.tex"]
    # The turn table: a lookup, then the candidates with each check on its own line; LaTeX specials are escaped.
    assert "1 & looked up /regions" in outline
    assert r"UNSAFE -- SAFE\_M1\_mutex is broken: 50\% \& \#1 \newline" in outline
    # The worked example shows the reply's candidates and the checks; the template keeps fixed sections only.
    assert "1. kmr_pick_part_from_M1 (KMR): M1 holds it." in files["appendix/appendix_worked_example.tex"]
    template = files["appendix/appendix_prompt_template.tex"]
    assert "One event per candidate." in template and "ur5e-1 did not complete" not in template
    with zipfile.ZipFile(supplement) as bundle:
        names = bundle.namelist()
    assert "supplementary_material/transcripts/conveyor/outline.md" in names
    assert "supplementary_material/records/conveyor/outline/requests/b/round00_attempt01_request.json" in names
    assert "supplementary_material/records/conveyor/composition/composition.json" in names


def test_tokens_sum_claude_and_openai_usage_alike(tmp_path):
    usages = {"claude": {"input_tokens": 100, "cache_read_input_tokens": 50, "cache_creation_input_tokens": 5,
                         "output_tokens": 10},
              "openai": {"prompt_tokens": 8739, "completion_tokens": 204, "total_tokens": 8943,
                         "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 8736}}}
    requests = []
    for name, usage in usages.items():
        (tmp_path / f"{name}_response.json").write_text(json.dumps({"raw_response": {"usage": usage}}))
        requests.append(tmp_path / f"{name}_request.json")
    assert paper_export.tokens(requests[:1]) == 165
    assert paper_export.tokens(requests[1:]) == 8943


def test_latex_text_is_escaped_and_written_in_ascii():
    assert paper_export.tex("a_b & 50% #1 {x} → y") == r"a\_b \& 50\% \#1 \{x\} -> y"
    assert paper_export.tex("back\\slash") == r"back\textbackslash{}slash"

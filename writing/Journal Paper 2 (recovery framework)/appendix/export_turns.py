"""Export recovery runs for the paper: readable Markdown of one run, or the appendix and supplementary zips.

Usage, from the repository root:
    poetry run python "writing/Journal Paper 2 (recovery framework)/appendix/export_turns.py" <draft or composition dir> <out.md>
    poetry run python "writing/Journal Paper 2 (recovery framework)/appendix/export_turns.py" --paper

`--paper` builds `build/appendix_overleaf.zip` (LaTeX to `\\input` in Overleaf) and `build/supplementary_material.zip`
(every prompt and reply, the raw records and a README) from the runs named in `appendix_set.json`; the Recovery
page's "Export for paper" button builds the same. Prompts are the recorded words; only their layout is restored.
"""
from __future__ import annotations

import sys
from pathlib import Path

from cais_spade_llm.ui.paper_export import (
    build_paper_export,
    markdown_composition,
    markdown_outline,
)


def main(argv: list[str]) -> None:
    if argv[1:] == ["--paper"]:
        overleaf, supplement = build_paper_export()
        sys.stdout.write(f"wrote {overleaf}\nwrote {supplement}\n")
        return
    if len(argv) != 3:
        raise SystemExit(__doc__)
    directory, output = Path(argv[1]).resolve(), Path(argv[2])
    text = markdown_outline(directory) if (directory / "draft.json").exists() else markdown_composition(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    sys.stdout.write(f"wrote {output}\n")


if __name__ == "__main__":
    main(sys.argv)

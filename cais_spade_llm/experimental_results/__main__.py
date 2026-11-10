"""Build the paper's result figures and tables from the RCIM recovery results page.

Usage, from the repository root:
    poetry run python -m cais_spade_llm.experimental_results [--page PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cais_spade_llm.experimental_results import fig11_audited_outcome, fig12_model_comparison
from cais_spade_llm.experimental_results.page_data import RESULTS_PAGE, read_page_data

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Read the results page and write every figure and table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--page", type=Path, default=RESULTS_PAGE, help="rendered results page to read")
    parser.add_argument("--out", type=Path, default=HERE, help="folder that receives figures/ and tables/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # fontTools reports every table it prunes while embedding fonts.
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    data = read_page_data(args.page)
    logger.info("Read %s (generated %s)", args.page, data["generated_at"])
    for figure in (fig11_audited_outcome, fig12_model_comparison):
        for path in figure.build(data, args.out / "figures"):
            logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()

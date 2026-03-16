from __future__ import annotations

import argparse

from test_case3_recovery_main import (
    DEFAULT_LIVE_MODEL,
    MAIN_V2_VARIANT,
    run_test,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the case3 recovery harness main-v2 (mirrored xarm6 recovery variant)."
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Use the deterministic scripted mirror scenario instead of the live ReAct loop.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LIVE_MODEL,
        help=f"Model to use in live mode (default: {DEFAULT_LIVE_MODEL}).",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Skip writing the JSON debug artifact.",
    )
    args = parser.parse_args()
    run_test(
        llm_mode="scripted" if args.scripted else "live",
        llm_model=args.model,
        write_debug=not args.no_debug,
        variant=MAIN_V2_VARIANT,
    )


if __name__ == "__main__":
    main()

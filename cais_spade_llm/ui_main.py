"""Unified entry point for the CAIS-SPADE-LLM system.

Modes::

    # Operator console (default) -- web UI at http://localhost:8080
    python3 -m cais_spade_llm.ui_main

    # Headless -- same as the old spade_main.py, no web server
    python3 -m cais_spade_llm.ui_main --headless
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import warnings

# Ensure the cais_spade_llm package directory is on sys.path so that
# agent_creator, utils, etc. can be imported the same way spade_main.py does.
_pkg_dir = os.path.join(os.path.dirname(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# Silence noisy third-party loggers.
for _name in ("pyjabber", "winloop", "asyncio"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", message="Unknown stanza interface")


def _run_headless() -> None:
    """Run the SPADE agents without the web UI (legacy CLI mode)."""
    from spade import run as spade_run
    import utils, agent_creator
    from function_analyzer import FunctionAnalyzer
    from agent_creator import ALLOWED_FUNCS
    from cais_spade_llm.ui.bridge import SystemBridge

    async def _main():
        SystemBridge._archive_monitors()

        prod_files = utils.get_init_files("cais_spade_llm/initialization/products/")
        res_files = utils.get_init_files("cais_spade_llm/initialization/resources/")

        user = agent_creator.create_user()
        resources = agent_creator.create_resource_agents(
            res_files, "cais_spade_llm/initialization/cca.json"
        )
        products = agent_creator.create_product_agents(
            prod_files, resources, "cais_spade_llm/initialization/cca.json"
        )
        cca = agent_creator.create_central_controller(
            "cais_spade_llm/initialization/cca.json", resources
        )

        FunctionAnalyzer.build_tools_catalogue(
            agents=products + resources,
            allowed=ALLOWED_FUNCS,
            outfile="cais_spade_llm/initialization/tools.json",
        )

        for ra in resources:
            await ra.start(auto_register=True)
        if cca:
            await cca.start(auto_register=True)
        if user:
            await user.start(auto_register=True)
        for i, pa in enumerate(products):
            await asyncio.sleep(0.2 * i)
            await pa.start(auto_register=True)

        print("Agents running (headless). Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            for a in products + resources + ([cca] if cca else []) + ([user] if user else []):
                try:
                    await a.stop()
                except Exception:
                    pass

    spade_run(_main(), embedded_xmpp_server=True)


def _run_ui() -> None:
    """Run the NiceGUI operator console (default mode)."""
    print(
        "UI started. Use Control to launch Gazebo + MoveIt, then Dashboard > Start System to start agents.",
        flush=True,
    )
    from cais_spade_llm.ui.app import create_app
    create_app()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAIS-SPADE-LLM: multi-agent manufacturing system",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the web UI (agents only, Ctrl+C to stop)",
    )
    args = parser.parse_args()

    if args.headless:
        _run_headless()
    else:
        _run_ui()


if __name__ in {"__main__", "__mp_main__"}:
    main()

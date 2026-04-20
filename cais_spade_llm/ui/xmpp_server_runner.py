"""Standalone embedded XMPP server runner for UI mode.

Runs pyjabber in a separate process so heavy startup work does not block
NiceGUI's asyncio loop.
"""

from __future__ import annotations

import asyncio
import os

from cais_spade_llm.utils.logging_setup import install_startup_logging_filters
from cais_spade_llm.utils.xmpp_runtime import install_xmpp_runtime_patches

install_startup_logging_filters()
install_xmpp_runtime_patches()


async def _run() -> None:
    import loguru
    from pyjabber.server import Server
    from pyjabber.server_parameters import Parameters

    # Keep pyjabber output quiet; UI process keeps primary logs.
    loguru.logger.remove()
    host = str(os.environ.get("CAIS_XMPP_HOST", "localhost")).strip() or "localhost"
    in_memory = str(os.environ.get("CAIS_XMPP_DB_IN_MEMORY", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    server = Server(Parameters(host=host, database_in_memory=in_memory))
    await server.start()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

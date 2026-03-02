"""Operator console entry point: NiceGUI web UI + SPADE agent system in one process.

Usage::

    python3 -m cais_spade_llm.ui_main

Opens a browser at http://localhost:8080 with the operator console.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

# Ensure the cais_spade_llm package directory is on sys.path so that
# agent_creator, utils, etc. can be imported the same way spade_main.py does.
_pkg_dir = os.path.join(os.path.dirname(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# Silence noisy third-party loggers (same as spade_main.py).
for _name in ("pyjabber", "winloop", "asyncio"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", message="Unknown stanza interface")

# Launch the NiceGUI app (this blocks until the server shuts down).
from cais_spade_llm.ui.app import create_app

if __name__ in {"__main__", "__mp_main__"}:
    create_app()

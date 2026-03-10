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
import atexit
import logging
import os
import signal
import subprocess
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


_KILL_CMDS: list[str] = [
    "killall -9 gzserver gzclient 2>/dev/null",
    (
        "killall -9 move_group rviz2 robot_state_publisher "
        "joint_state_publisher static_transform_publisher "
        "ros2_control_node 2>/dev/null"
    ),
    "pkill -9 -f gazebo 2>/dev/null",
    "pkill -9 -f keyboard_teleop.py 2>/dev/null",
    "pkill -9 -f 'spawner' 2>/dev/null",
    "pkill -9 -f xarm_driver_node 2>/dev/null",
    "pkill -9 -f controller_manager 2>/dev/null",
    "pkill -9 -f ur_robot_driver 2>/dev/null",
]


def _kill_stale_ros2_processes(*, quiet: bool = False) -> None:
    """Kill orphan Gazebo/ROS2/MoveIt processes.

    Called both at startup (to clear leftovers from a previous bad exit)
    and at shutdown (via atexit / signal handler) so stale processes never
    survive across sessions.
    """
    log = logging.getLogger("ui_main")
    killed_any = False
    for cmd in _KILL_CMDS:
        try:
            result = subprocess.run(
                ["bash", "-c", cmd], capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                killed_any = True
        except Exception:
            pass
    if killed_any and not quiet:
        log.info("Startup cleanup: killed stale ROS2/Gazebo processes from previous session.")


def _cleanup_ros2_shm() -> None:
    """Remove stale ROS2 shared-memory segments that accumulate in WSL."""
    log = logging.getLogger("ui_main")
    try:
        result = subprocess.run(
            ["bash", "-c", "ls /dev/shm/fastrtps_* 2>/dev/null"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            subprocess.run(
                ["bash", "-c", "rm -f /dev/shm/fastrtps_* 2>/dev/null"],
                capture_output=True, timeout=5,
            )
            log.info("Startup cleanup: removed stale ROS2 shared-memory files.")
    except Exception:
        pass


def _install_exit_cleanup() -> None:
    """Register atexit + SIGTERM handler to guarantee process cleanup."""
    atexit.register(_kill_stale_ros2_processes, quiet=True)

    def _signal_handler(signum: int, _frame: object) -> None:
        _kill_stale_ros2_processes(quiet=True)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _signal_handler)


def _run_ui() -> None:
    """Run the NiceGUI operator console (default mode)."""
    # Clean slate: kill any leftover processes from a previous session.
    _kill_stale_ros2_processes()
    _cleanup_ros2_shm()
    # Register cleanup for when this session exits.
    _install_exit_cleanup()
    print(
        "UI started. Use Control to launch Gazebo + MoveIt, then Dashboard > Start System to start agents.",
        flush=True,
    )
    from cais_spade_llm.ui.app import create_app
    create_app()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAIS-SP ADE-LLM: multi-agent manufacturing system",
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

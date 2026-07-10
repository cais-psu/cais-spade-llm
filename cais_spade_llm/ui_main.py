"""Unified entry point for the CAIS-SPADE-LLM system.

Modes::

    # Operator console (default) -- web UI at http://localhost:8080
    python3 -m cais_spade_llm.ui_main

    # Headless -- SPADE agents only, no web server
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

from cais_spade_llm.ui.gazebo_cleanup import keep_gazebo_on_exit
from cais_spade_llm.utils.logging_setup import install_startup_logging_filters
from cais_spade_llm.utils.xmpp_runtime import install_xmpp_runtime_patches

install_startup_logging_filters()
install_xmpp_runtime_patches()

LOGGER = logging.getLogger("ui_main")

# Legacy import compatibility.
# Ensure the cais_spade_llm package directory is on sys.path so that
# agent_creator, utils, etc. resolve as bare module names (legacy import style).
_pkg_dir = os.path.join(os.path.dirname(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)



# Headless agent runner.
def _run_headless() -> None:
    """Run the SPADE agents without the web UI (legacy CLI mode)."""
    import agent_creator
    import utils
    from agent_creator import ALLOWED_FUNCS
    from function_analyzer import FunctionAnalyzer
    from spade import run as spade_run

    from cais_spade_llm.ui.bridge import SystemBridge

    async def _main() -> None:
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
                    LOGGER.debug("Headless agent stop failed for %r.", a, exc_info=True)

    spade_run(_main(), embedded_xmpp_server=True)


# ROS2/Gazebo cleanup.
_KILL_CMDS: list[str] = [
    "killall -9 gzserver gzclient 2>/dev/null",
    (
        "killall -9 move_group rviz2 robot_state_publisher "
        "joint_state_publisher static_transform_publisher "
        "ros2_control_node 2>/dev/null"
    ),
    "pkill -9 -f gazebo 2>/dev/null",
    # spawn_entity.py can be mid-spawn when switching simulations; kill it so
    # it does not hold /spawn_entity calls that block the next Gazebo startup.
    "pkill -9 -f spawn_entity.py 2>/dev/null",
    "pkill -9 -f keyboard_teleop.py 2>/dev/null",
    "pkill -9 -f 'spawner' 2>/dev/null",
    "pkill -9 -f xarm_driver_node 2>/dev/null",
    "pkill -9 -f xarm6_hardware_driver.launch.py 2>/dev/null",
    "pkill -9 -f xarm6_hardware_moveit.launch.py 2>/dev/null",
    "pkill -9 -f XArm6JointStateRelay 2>/dev/null",
    "pkill -9 -f controller_manager 2>/dev/null",
    "pkill -9 -f ur5e_rg2_rtde_gripper.py 2>/dev/null",
    "pkill -9 -f robot_state_publisher 2>/dev/null",
]


def _kill_stale_ros2_processes(*, quiet: bool = False, reason: str = "startup") -> None:
    """Kill orphan Gazebo/ROS2/MoveIt processes.

    Called both at startup (to clear leftovers from a previous bad exit)
    and at shutdown (via atexit / signal handler) so stale processes never
    survive across sessions.
    """
    LOGGER.info("ROS2/Gazebo hard cleanup requested reason=%s", reason)
    killed_any = False
    for cmd in _KILL_CMDS:
        try:
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                killed_any = True
        except Exception:
            LOGGER.debug(
                "ROS2/Gazebo cleanup command failed reason=%s command=%r",
                reason,
                cmd,
                exc_info=True,
            )
    if killed_any:
        # Give OS time to release ports/shared memory (critical on WSL2)
        import time

        time.sleep(3)
        if not quiet:
            LOGGER.info(
                "ROS2/Gazebo cleanup complete: killed stale processes reason=%s.",
                reason,
            )


def _cleanup_ros2_shm() -> None:
    """Remove stale ROS2/DDS shared-memory and Gazebo temp files (WSL2)."""
    shm_patterns = [
        "/dev/shm/fastrtps_*",
        "/dev/shm/cyclonedds_*",
    ]
    tmp_patterns = [
        "/tmp/gazebo-*",
        "/tmp/.gazebo-*",
    ]
    cleaned = False
    for pattern in shm_patterns + tmp_patterns:
        try:
            result = subprocess.run(
                ["bash", "-c", f"ls {pattern} 2>/dev/null"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.stdout.strip():
                subprocess.run(
                    ["bash", "-c", f"rm -rf {pattern} 2>/dev/null"],
                    capture_output=True,
                    timeout=5,
                )
                cleaned = True
        except Exception:
            LOGGER.debug(
                "Startup cleanup failed for shared-memory/temp pattern=%s",
                pattern,
                exc_info=True,
            )
    if cleaned:
        LOGGER.info("Startup cleanup: removed stale shared-memory / Gazebo temp files.")


# UI startup and shutdown cleanup.
def _install_exit_cleanup() -> None:
    """Register atexit + SIGTERM handler to guarantee process cleanup."""
    if keep_gazebo_on_exit():
        LOGGER.info("CAIS_KEEP_GAZEBO_ON_EXIT=1; skipping UI exit Gazebo hard-kill hooks.")
        return

    atexit.register(_kill_stale_ros2_processes, quiet=True, reason="app_shutdown")

    def _signal_handler(signum: int, _frame: object) -> None:
        _kill_stale_ros2_processes(quiet=True, reason="app_shutdown")
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _signal_handler)


def _run_ui() -> None:
    """Run the NiceGUI operator console (default mode)."""
    # Clean slate: kill any leftover processes from a previous session.
    _kill_stale_ros2_processes(reason="startup")
    _cleanup_ros2_shm()
    # Register cleanup for when this session exits.
    _install_exit_cleanup()
    print(
        "UI started. Use Control to launch Gazebo + MoveIt, then Dashboard > Start System to start agents.",
        flush=True,
    )
    from cais_spade_llm.ui.app import create_app

    create_app()


# CLI parsing.
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CAIS-SPADE-LLM: multi-agent manufacturing system",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the web UI (agents only, Ctrl+C to stop)",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.headless:
        _run_headless()
    else:
        _run_ui()


if __name__ in {"__main__", "__mp_main__"}:
    main()

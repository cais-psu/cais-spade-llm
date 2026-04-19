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
from pathlib import Path
import signal
import subprocess
import sys

from cais_spade_llm.logging_setup import install_startup_logging_filters
from cais_spade_llm.xmpp_runtime import install_xmpp_runtime_patches

install_startup_logging_filters()
install_xmpp_runtime_patches()

# Ensure the cais_spade_llm package directory is on sys.path so that
# agent_creator, utils, etc. can be imported the same way spade_main.py does.
_pkg_dir = os.path.join(os.path.dirname(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

_F5_DEBUG_BRIDGE_FIXTURE_FINAL_OUTPUT = (
    Path(__file__).resolve().parent
    / "monitor"
    / "debug"
    / "worked"
    / "1"
    / "multi_turn_turn24_final_output_response_20260416T160606.txt"
)

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
    # spawn_entity.py can be mid-spawn when switching simulations; kill it so
    # it does not hold /spawn_entity calls that block the next Gazebo startup.
    "pkill -9 -f spawn_entity.py 2>/dev/null",
    "pkill -9 -f keyboard_teleop.py 2>/dev/null",
    "pkill -9 -f 'spawner' 2>/dev/null",
    "pkill -9 -f xarm_driver_node 2>/dev/null",
    "pkill -9 -f controller_manager 2>/dev/null",
    "pkill -9 -f ur_robot_driver 2>/dev/null",
    "pkill -9 -f robot_state_publisher 2>/dev/null",
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
    if killed_any:
        # Give OS time to release ports/shared memory (critical on WSL2)
        import time
        time.sleep(3)
        if not quiet:
            log.info("Startup cleanup: killed stale ROS2/Gazebo processes from previous session.")


def _cleanup_ros2_shm() -> None:
    """Remove stale ROS2/DDS shared-memory and Gazebo temp files (WSL2)."""
    log = logging.getLogger("ui_main")
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
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                subprocess.run(
                    ["bash", "-c", f"rm -rf {pattern} 2>/dev/null"],
                    capture_output=True, timeout=5,
                )
                cleaned = True
        except Exception:
            pass
    if cleaned:
        log.info("Startup cleanup: removed stale shared-memory / Gazebo temp files.")


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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CAIS-SPADE-LLM: multi-agent manufacturing system",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the web UI (agents only, Ctrl+C to stop)",
    )
    parser.add_argument(
        "--bridge-fixture-final-output",
        "--runtime-bridge-fixture-final-output",
        dest="bridge_fixture_final_output",
        default="",
        metavar="PATH",
        help=(
            "Test-only: exact archived multi-turn final_output artifact to replay "
            "for runtime bridge generation."
        ),
    )
    parser.add_argument(
        "--verify-generated-bridge-in-gazebo",
        action="store_true",
        help=(
            "Test-only: enable auto-execution of generated bridge proposals when "
            "the UI/system is in simulation+gazebo mode."
        ),
    )
    return parser


def _apply_runtime_bridge_test_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    fixture_arg = str(getattr(args, "bridge_fixture_final_output", "") or "").strip()
    using_f5_debug_fixture = False
    if (
        not fixture_arg
        and not os.environ.get("CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT")
        and not bool(getattr(args, "headless", False))
        and sys.gettrace() is not None
        and _F5_DEBUG_BRIDGE_FIXTURE_FINAL_OUTPUT.exists()
    ):
        fixture_arg = str(_F5_DEBUG_BRIDGE_FIXTURE_FINAL_OUTPUT)
        using_f5_debug_fixture = True

    if fixture_arg:
        fixture_path = Path(fixture_arg).expanduser()
        try:
            fixture_path = fixture_path.resolve()
        except Exception:
            pass
        if not fixture_path.exists():
            parser.error(
                "--bridge-fixture-final-output must point to an existing final_output file"
            )
        if fixture_path.is_dir():
            parser.error(
                "--bridge-fixture-final-output must point to the exact final_output file, not a directory"
            )
        os.environ["CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT"] = str(fixture_path)
        if using_f5_debug_fixture:
            print(f"F5 debug runtime bridge fixture final_output: {fixture_path}", flush=True)
        else:
            print(f"Runtime bridge fixture final_output: {fixture_path}", flush=True)
    elif os.environ.get("CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT"):
        print(
            "Runtime bridge fixture final_output: "
            f"{os.environ['CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT']}",
            flush=True,
        )

    if bool(getattr(args, "verify_generated_bridge_in_gazebo", False)) or using_f5_debug_fixture:
        os.environ["CAIS_VERIFY_GENERATED_BRIDGE_IN_GAZEBO"] = "1"
        if using_f5_debug_fixture:
            print("F5 debug generated bridge Gazebo verification: enabled", flush=True)
        else:
            print("Generated bridge Gazebo verification: enabled", flush=True)
    elif os.environ.get("CAIS_VERIFY_GENERATED_BRIDGE_IN_GAZEBO"):
        print(
            "Generated bridge Gazebo verification: "
            f"{os.environ['CAIS_VERIFY_GENERATED_BRIDGE_IN_GAZEBO']}",
            flush=True,
        )


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _apply_runtime_bridge_test_args(args, parser)

    if args.headless:
        _run_headless()
    else:
        _run_ui()


if __name__ in {"__main__", "__mp_main__"}:
    main()

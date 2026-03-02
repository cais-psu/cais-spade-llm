#!/usr/bin/env python3
"""
Smoke-test robot wiring across:
  1) robot_*.json manifests
  2) controller construction
  3) RobotAgent construction and mode dispatch

This script is intentionally dependency-aware:
- It can validate JSON + controller wiring without ROS2 running.
- It can optionally run `move_home` through the controller and/or RobotAgent.
- If SPADE/OpenAI deps are missing, RobotAgent checks fail with a clear message.

Examples:
  python3 test/test_robot_wiring_smoke.py
  python3 test/test_robot_wiring_smoke.py --robot ur5e --env gazebo
  python3 test/test_robot_wiring_smoke.py --robot both --run-controller-home
  python3 test/test_robot_wiring_smoke.py --robot xarm6 --run-agent-home
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "cais_spade_llm"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))


EXPECTED_FUNCTIONS = {
    "pick_approach",
    "pick_grasp",
    "place_approach",
    "place_insert",
}

PART_MODEL_MAP = {
    "SG": "gear_small",
    "MG": "gear_medium",
    "LG": "gear_large",
    "SRP": "rect_pin_small",
    "MRP": "rect_pin_medium",
    "LRP": "rect_pin_large",
    "SCP": "circ_pin_small",
    "MCP": "circ_pin_medium",
    "LCP": "circ_pin_large",
}


def _load_robot_manifest(robot_name: str, init_dir: Path) -> tuple[dict[str, Any], Path]:
    path = init_dir / f"robot_{robot_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if robot_name not in data:
        raise KeyError(f"Expected top-level key '{robot_name}' in {path}")
    return data[robot_name], path


def _resolve_execution_mode(meta: dict[str, Any], env_block: dict[str, Any]) -> str:
    mode = str(env_block.get("execution_mode", meta.get("execution_mode", "simulate"))).strip().lower()
    if mode not in {"simulate", "ros2", "real"}:
        return "simulate"
    return mode


def _validate_manifest(
    robot_name: str,
    meta: dict[str, Any],
    env_name: str,
    env_block: dict[str, Any],
    resolved_mode: str,
) -> list[str]:
    issues: list[str] = []

    for key in ("type", "jid", "password"):
        if key not in meta or str(meta.get(key)).strip() == "":
            issues.append(f"missing required top-level field: {key}")

    declared = set(meta.get("functions") or meta.get("function_names") or [])
    missing_funcs = sorted(EXPECTED_FUNCTIONS - declared)
    if missing_funcs:
        issues.append(f"missing expected robot functions: {missing_funcs}")

    if env_name not in meta:
        issues.append(f"missing environment block: '{env_name}'")
        return issues

    if not isinstance(env_block, dict):
        issues.append(f"environment block '{env_name}' must be an object")
        return issues

    if resolved_mode in {"ros2", "real"}:
        controller_cfg = env_block.get("controller")
        if not isinstance(controller_cfg, dict) or not controller_cfg:
            issues.append(
                f"execution_mode='{resolved_mode}' requires non-empty '{env_name}.controller'"
            )

    return issues


def _build_controller(
    robot_name: str,
    controller_cfg: dict[str, Any],
    named_positions: dict[str, Any],
    execution_mode: str,
):
    from resources.robot import UR5eController, XArm6Controller

    cls = UR5eController if robot_name == "ur5e" else XArm6Controller
    return cls(
        controller_config=controller_cfg,
        named_positions=named_positions,
        execution_mode=execution_mode,
    )


def _controller_health(controller) -> tuple[bool, str]:
    valid = bool(getattr(controller, "_config_valid", True))
    if valid:
        return True, "controller config looks valid"
    msg = str(getattr(controller, "_last_failure_message", "") or "controller config invalid")
    return False, msg


def _format_dict(obj: dict[str, Any]) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=True, sort_keys=True)


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _try_build_robot_agent(
    *,
    robot_name: str,
    meta: dict[str, Any],
    env_block: dict[str, Any],
    execution_mode: str,
):
    # LlmAgent constructs an OpenAI client at import time.
    # A dummy key is enough for local smoke checks (no network calls are made here).
    os.environ.setdefault("OPENAI_API_KEY", "sk-local-smoke-test")

    try:
        from agents.resource_agent.robot_agent import RobotAgent
    except Exception as exc:
        hint = ""
        if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", "") == "spade":
            hint = (
                " | hint: run with project venv: "
                "`.venv/bin/python test/test_robot_wiring_smoke.py ...`"
            )
        return None, f"RobotAgent import failed: {type(exc).__name__}: {exc}{hint}"

    try:
        agent = RobotAgent(
            meta["jid"],
            meta.get("password", "none"),
            name=robot_name,
            instructions=meta.get("instructions", ""),
            function_names=(meta.get("functions") or meta.get("function_names") or []),
            static_capabilities=env_block.get("static_capabilities", {}),
            execution_mode=execution_mode,
            controller_config=env_block.get("controller", {}),
            named_positions=env_block.get("named_positions", {}),
            sg_slippage_mode=meta.get("sg_slippage_mode", "off"),
            sg_slippage_scope=meta.get("sg_slippage_scope", robot_name),
        )
        return agent, ""
    except Exception as exc:
        return None, f"RobotAgent init failed: {type(exc).__name__}: {exc}"


async def _run_agent_move_home(agent) -> dict[str, Any]:
    return await agent.move_home()


def _default_product_geometry(
    *,
    part_name: str,
    slot_x: float,
    slot_y: float,
    board_center_z: float,
    slot_floor_z: float,
    part_height_m: float,
) -> dict[str, Any]:
    upper = str(part_name or "").upper()
    return {
        "board_center": {"x": 0.0, "y": 0.0, "z": float(board_center_z)},
        "slot_xy": [float(slot_x), float(slot_y)],
        "slot_floor_z_m": float(slot_floor_z),
        "part_height_m": float(part_height_m),
        "model_name": PART_MODEL_MAP.get(upper, ""),
        "part_name": upper,
    }


async def _run_agent_pick_place_sequence(
    *,
    agent,
    part_name: str,
    origin_location: str,
    destination_location: str,
    product_geometry: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    steps: list[tuple[str, dict[str, Any]]] = []

    res = await agent.pick_approach(
        origin_resource_location=origin_location,
        part_name=part_name,
        product_geometry=product_geometry,
    )
    steps.append(("pick_approach", res if isinstance(res, dict) else {"result": res}))
    if str((res or {}).get("status", "")).lower() != "completed":
        return steps

    res = await agent.pick_grasp(
        part_name=part_name,
        origin_resource_location=origin_location,
        product_geometry=product_geometry,
    )
    steps.append(("pick_grasp", res if isinstance(res, dict) else {"result": res}))
    if str((res or {}).get("status", "")).lower() != "completed":
        return steps

    res = await agent.place_approach(
        destination_location=destination_location,
        part_name=part_name,
        product_geometry=product_geometry,
    )
    steps.append(("place_approach", res if isinstance(res, dict) else {"result": res}))
    if str((res or {}).get("status", "")).lower() != "completed":
        return steps

    res = await agent.place_insert(
        destination_location=destination_location,
        part_name=part_name,
        product_geometry=product_geometry,
    )
    steps.append(("place_insert", res if isinstance(res, dict) else {"result": res}))
    return steps


def run_one_robot(
    *,
    robot_name: str,
    env_name: str,
    init_dir: Path,
    run_controller_home: bool,
    run_agent_home: bool,
    run_agent_sequence: bool,
    skip_agent: bool,
    part_name: str,
    origin_location: str,
    destination_location: str,
    slot_x: float,
    slot_y: float,
    board_center_z: float,
    slot_floor_z: float,
    part_height_m: float,
) -> bool:
    _print_section(f"{robot_name.upper()} | env={env_name}")

    try:
        meta, path = _load_robot_manifest(robot_name, init_dir)
    except Exception as exc:
        print(f"[FAIL] Could not load manifest: {type(exc).__name__}: {exc}")
        return False

    env_block = meta.get(env_name, {})
    resolved_mode = _resolve_execution_mode(meta, env_block if isinstance(env_block, dict) else {})

    print(f"manifest: {path}")
    print(f"resolved execution_mode: {resolved_mode}")

    issues = _validate_manifest(robot_name, meta, env_name, env_block, resolved_mode)
    if issues:
        print("[WARN] Manifest checks found issues:")
        for item in issues:
            print(f"  - {item}")
    else:
        print("[PASS] Manifest checks passed")

    ok = True
    controller = None

    # Build controller only when non-sim mode is requested.
    if resolved_mode in {"ros2", "real"}:
        try:
            controller = _build_controller(
                robot_name=robot_name,
                controller_cfg=env_block.get("controller", {}),
                named_positions=env_block.get("named_positions", {}),
                execution_mode=resolved_mode,
            )
            cfg_ok, cfg_msg = _controller_health(controller)
            if cfg_ok:
                print(f"[PASS] Controller construction: {cfg_msg}")
            else:
                print(f"[FAIL] Controller construction: {cfg_msg}")
                ok = False
        except Exception as exc:
            print(f"[FAIL] Controller construction error: {type(exc).__name__}: {exc}")
            ok = False
    else:
        print("[SKIP] Controller construction (simulate mode)")

    if run_controller_home and controller is not None:
        print("controller.move_home() ...")
        try:
            result = controller.move_home()
            print(_format_dict(result if isinstance(result, dict) else {"result": result}))
            if not bool((result or {}).get("success")):
                ok = False
        except Exception as exc:
            print(f"[FAIL] controller.move_home exception: {type(exc).__name__}: {exc}")
            ok = False
        finally:
            try:
                controller.shutdown()
            except Exception:
                pass

    if skip_agent:
        print("[SKIP] RobotAgent wiring checks (--skip-agent)")
        return ok and not issues

    agent, err = _try_build_robot_agent(
        robot_name=robot_name,
        meta=meta,
        env_block=env_block if isinstance(env_block, dict) else {},
        execution_mode=resolved_mode,
    )
    if agent is None:
        print(f"[FAIL] {err}")
        return False

    snapshot = agent._snapshot_state()
    print("[PASS] RobotAgent init")
    print(_format_dict(snapshot))

    if run_agent_sequence:
        geometry = _default_product_geometry(
            part_name=part_name,
            slot_x=slot_x,
            slot_y=slot_y,
            board_center_z=board_center_z,
            slot_floor_z=slot_floor_z,
            part_height_m=part_height_m,
        )
        print("RobotAgent pick/place sequence payload:")
        print(_format_dict(geometry))
        print("RobotAgent sequence: pick_approach -> pick_grasp -> place_approach -> place_insert ...")
        try:
            sequence = asyncio.run(
                _run_agent_pick_place_sequence(
                    agent=agent,
                    part_name=part_name,
                    origin_location=origin_location,
                    destination_location=destination_location,
                    product_geometry=geometry,
                )
            )
            for step_name, step_result in sequence:
                print(f"[STEP] {step_name}")
                print(_format_dict(step_result if isinstance(step_result, dict) else {"result": step_result}))
                if str((step_result or {}).get("status", "")).lower() != "completed":
                    ok = False
                    break

            expected_order = ["pick_approach", "pick_grasp", "place_approach", "place_insert"]
            if [name for name, _ in sequence] != expected_order:
                ok = False
        except Exception as exc:
            print(f"[FAIL] RobotAgent sequence exception: {type(exc).__name__}: {exc}")
            ok = False
    else:
        print("[SKIP] RobotAgent pick/place sequence (--run-agent-sequence not set)")

    if run_agent_home:
        print("RobotAgent.move_home() ...")
        try:
            result = asyncio.run(_run_agent_move_home(agent))
            print(_format_dict(result if isinstance(result, dict) else {"result": result}))
            status = str((result or {}).get("status", "")).lower()
            if status not in {"completed"}:
                ok = False
        except Exception as exc:
            print(f"[FAIL] RobotAgent.move_home exception: {type(exc).__name__}: {exc}")
            ok = False
    else:
        print("[SKIP] RobotAgent.move_home (--run-agent-home not set)")

    # Best-effort cleanup for ROS2 controller internals.
    inner = getattr(agent, "_controller", None)
    if inner is not None and hasattr(inner, "shutdown"):
        try:
            inner.shutdown()
        except Exception:
            pass

    return ok and not issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test robot JSON/controller/agent wiring.")
    parser.add_argument(
        "--robot",
        choices=["ur5e", "xarm6", "both"],
        default="both",
        help="Which robot wiring to test.",
    )
    parser.add_argument(
        "--env",
        choices=["gazebo", "real"],
        default=os.environ.get("ROBOT_ENV", "gazebo").strip().lower() or "gazebo",
        help="Which environment block to resolve from robot JSON.",
    )
    parser.add_argument(
        "--init-dir",
        default=str(PKG_ROOT / "initialization" / "resources"),
        help="Directory containing robot_ur5e.json / robot_xarm6.json.",
    )
    parser.add_argument(
        "--run-controller-home",
        action="store_true",
        help="Call controller.move_home() after construction (requires ROS2 stack).",
    )
    parser.add_argument(
        "--run-agent-home",
        action="store_true",
        help="Call RobotAgent.move_home() after construction.",
    )
    parser.add_argument(
        "--run-agent-sequence",
        action="store_true",
        help="Run RobotAgent pick/place tool sequence through the current wiring.",
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Skip RobotAgent creation and only test manifest/controller wiring.",
    )
    parser.add_argument(
        "--part-name",
        default="MRP",
        help="Part name used by the RobotAgent sequence test.",
    )
    parser.add_argument(
        "--origin-location",
        default="prusa-mk3",
        help="Origin location string passed to pick tools.",
    )
    parser.add_argument(
        "--destination-location",
        default="assembly_board-v1",
        help="Destination location string passed to place tools.",
    )
    parser.add_argument(
        "--slot-x",
        type=float,
        default=0.0,
        help="Board-local slot x (meters) used in generated product_geometry.",
    )
    parser.add_argument(
        "--slot-y",
        type=float,
        default=0.0,
        help="Board-local slot y (meters) used in generated product_geometry.",
    )
    parser.add_argument(
        "--board-center-z",
        type=float,
        default=1.02,
        help="Board center z (meters) used in generated product_geometry.",
    )
    parser.add_argument(
        "--slot-floor-z",
        type=float,
        default=1.025,
        help="Slot floor z (meters) used in generated product_geometry.",
    )
    parser.add_argument(
        "--part-height-m",
        type=float,
        default=0.08,
        help="Part height (meters) used in generated product_geometry.",
    )
    args = parser.parse_args()

    print(f"[INFO] python: {sys.executable}")
    print(f"[INFO] python version: {sys.version.split()[0]}")
    if (args.run_controller_home or args.run_agent_sequence or args.run_agent_home) and sys.version_info[:2] != (3, 10):
        print(
            "[WARN] ROS2 Humble Python ABI is typically 3.10. "
            "Current interpreter is not 3.10; ROS2 imports may fail."
        )

    robots = ["ur5e", "xarm6"] if args.robot == "both" else [args.robot]
    init_dir = Path(args.init_dir).resolve()
    if not init_dir.exists():
        print(f"[FAIL] init-dir does not exist: {init_dir}")
        return 2

    overall = True
    for robot_name in robots:
        one_ok = run_one_robot(
            robot_name=robot_name,
            env_name=args.env,
            init_dir=init_dir,
            run_controller_home=args.run_controller_home,
            run_agent_home=args.run_agent_home,
            run_agent_sequence=args.run_agent_sequence,
            skip_agent=args.skip_agent,
            part_name=args.part_name,
            origin_location=args.origin_location,
            destination_location=args.destination_location,
            slot_x=args.slot_x,
            slot_y=args.slot_y,
            board_center_z=args.board_center_z,
            slot_floor_z=args.slot_floor_z,
            part_height_m=args.part_height_m,
        )
        overall = overall and one_ok

    _print_section("RESULT")
    print("PASS" if overall else "FAIL")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())

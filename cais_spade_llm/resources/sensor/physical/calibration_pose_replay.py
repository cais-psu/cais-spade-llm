"""Replay reviewed wrist-calibration poses through MoveIt and capture samples."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_control(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return "resume"
    return str(payload.get("action") or "resume").strip().lower()


def _write_status(path: Path, **values: Any) -> None:
    _atomic_json_write(path, {"updated_at": time.time(), **values})


def _wait_future(
    rclpy: Any,
    node: Any,
    future: Any,
    *,
    timeout_sec: float,
    control_path: Path,
) -> str:
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and time.monotonic() < deadline and not future.done():
        if _read_control(control_path) == "abort":
            return "abort"
        rclpy.spin_once(node, timeout_sec=0.1)
    return "done" if future.done() else "timeout"


def _move_group_goal(group_name: str, pose: dict[str, Any], *, plan_only: bool) -> Any:
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import Constraints, JointConstraint

    names = [str(value) for value in pose.get("names", [])]
    positions = [float(value) for value in pose.get("positions", [])]
    if not names or len(names) != len(positions):
        raise RuntimeError("reviewed calibration pose has incomplete joint values")
    goal = MoveGroup.Goal()
    goal.request.group_name = group_name
    goal.request.num_planning_attempts = 5
    goal.request.allowed_planning_time = 10.0
    goal.request.max_velocity_scaling_factor = 0.10
    goal.request.max_acceleration_scaling_factor = 0.10
    goal.request.start_state.is_diff = True
    constraints = Constraints()
    constraints.name = f"{group_name}_calibration_pose_{pose.get('sample_index', 0)}"
    for name, position in zip(names, positions, strict=True):
        joint = JointConstraint()
        joint.joint_name = name
        joint.position = position
        joint.tolerance_above = 0.002
        joint.tolerance_below = 0.002
        joint.weight = 1.0
        constraints.joint_constraints.append(joint)
    goal.request.goal_constraints = [constraints]
    goal.planning_options.plan_only = plan_only
    goal.planning_options.look_around = False
    goal.planning_options.replan = False
    return goal


def _run_move_group(
    rclpy: Any,
    node: Any,
    client: Any,
    group_name: str,
    pose: dict[str, Any],
    *,
    plan_only: bool,
    control_path: Path,
) -> tuple[bool, str]:
    goal_future = client.send_goal_async(_move_group_goal(group_name, pose, plan_only=plan_only))
    outcome = _wait_future(
        rclpy,
        node,
        goal_future,
        timeout_sec=15.0,
        control_path=control_path,
    )
    if outcome != "done":
        return False, f"MoveIt goal acceptance {outcome}"
    goal_handle = goal_future.result()
    if goal_handle is None or not goal_handle.accepted:
        return False, "MoveIt rejected the reviewed calibration pose"
    result_future = goal_handle.get_result_async()
    outcome = _wait_future(
        rclpy,
        node,
        result_future,
        timeout_sec=90.0,
        control_path=control_path,
    )
    if outcome == "abort":
        cancel_future = goal_handle.cancel_goal_async()
        _wait_future(
            rclpy,
            node,
            cancel_future,
            timeout_sec=5.0,
            control_path=control_path,
        )
        return False, "calibration replay aborted; active MoveIt goal was cancelled"
    if outcome != "done":
        return False, f"MoveIt result {outcome}"
    response = result_future.result()
    action_result = getattr(response, "result", None)
    error_code = getattr(getattr(action_result, "error_code", None), "val", None)
    if int(error_code or 0) != 1:
        return False, f"MoveIt failed with error code {error_code}"
    return True, "plan ready" if plan_only else "execution completed"


def _wait_for_operator_control(
    rclpy: Any,
    node: Any,
    control_path: Path,
) -> str:
    while rclpy.ok():
        action = _read_control(control_path)
        if action in {"resume", "skip", "abort"}:
            return action
        rclpy.spin_once(node, timeout_sec=0.1)
    return "abort"


def replay(args: argparse.Namespace) -> None:  # noqa: C901 - explicit safety state machine.
    """Run the explicitly confirmed teach-then-replay state machine."""
    if not args.confirmed and not args.preview_only:
        raise RuntimeError("automatic calibration requires explicit operator confirmation")
    try:
        import rclpy
        from moveit_msgs.action import MoveGroup
        from rclpy.action import ActionClient
        from rclpy.node import Node
    except ImportError as exc:
        raise RuntimeError("ROS2 MoveIt Python interfaces are unavailable") from exc

    pose_path = Path(args.poses).expanduser()
    payload = yaml.safe_load(pose_path.read_text(encoding="utf-8")) or {}
    poses = payload.get("poses", [])
    if not isinstance(poses, list) or len(poses) < 20:
        raise RuntimeError("at least 20 reviewed calibration poses are required")
    control_path = Path(args.control)
    status_path = Path(args.status)
    _atomic_json_write(control_path, {"action": "resume", "requested_at": time.time()})

    rclpy.init()
    node = Node(f"{args.camera_role}_calibration_pose_replay")
    client = ActionClient(node, MoveGroup, "/move_action")
    try:
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("/move_action is unavailable; start the correct hardware MoveIt stack")
        if args.preview_only:
            for index, pose in enumerate(poses, start=1):
                _write_status(
                    status_path,
                    state="previewing",
                    pose_index=index,
                    pose_count=len(poses),
                )
                ok, message = _run_move_group(
                    rclpy,
                    node,
                    client,
                    args.planning_group,
                    pose,
                    plan_only=True,
                    control_path=control_path,
                )
                if not ok:
                    _write_status(
                        status_path,
                        state="preview_failed",
                        pose_index=index,
                        pose_count=len(poses),
                        error=message,
                    )
                    return
            _write_status(
                status_path,
                state="preview_ready",
                pose_index=len(poses),
                pose_count=len(poses),
            )
            return
        for index, pose in enumerate(poses, start=1):
            while True:
                action = _wait_for_operator_control(rclpy, node, control_path)
                if action == "abort":
                    _write_status(
                        status_path,
                        state="aborted",
                        pose_index=index,
                        pose_count=len(poses),
                    )
                    return
                if action == "skip":
                    _atomic_json_write(
                        control_path,
                        {"action": "resume", "requested_at": time.time()},
                    )
                    _write_status(
                        status_path,
                        state="skipped",
                        pose_index=index,
                        pose_count=len(poses),
                    )
                    break

                _write_status(
                    status_path,
                    state="planning",
                    pose_index=index,
                    pose_count=len(poses),
                )
                ok, message = _run_move_group(
                    rclpy,
                    node,
                    client,
                    args.planning_group,
                    pose,
                    plan_only=True,
                    control_path=control_path,
                )
                if not ok:
                    if _read_control(control_path) == "abort":
                        continue
                    _write_status(
                        status_path,
                        state="paused",
                        pose_index=index,
                        pose_count=len(poses),
                        error=message,
                    )
                    _atomic_json_write(
                        control_path,
                        {"action": "pause", "requested_at": time.time()},
                    )
                    continue

                _write_status(
                    status_path,
                    state="executing",
                    pose_index=index,
                    pose_count=len(poses),
                )
                ok, message = _run_move_group(
                    rclpy,
                    node,
                    client,
                    args.planning_group,
                    pose,
                    plan_only=False,
                    control_path=control_path,
                )
                if not ok:
                    if _read_control(control_path) == "abort":
                        continue
                    _write_status(
                        status_path,
                        state="paused",
                        pose_index=index,
                        pose_count=len(poses),
                        error=message,
                    )
                    _atomic_json_write(
                        control_path,
                        {"action": "pause", "requested_at": time.time()},
                    )
                    continue

                time.sleep(1.0)
                _write_status(
                    status_path,
                    state="capturing",
                    pose_index=index,
                    pose_count=len(poses),
                )
                capture = subprocess.run(
                    args.capture_command,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if capture.returncode != 0:
                    _write_status(
                        status_path,
                        state="paused",
                        pose_index=index,
                        pose_count=len(poses),
                        error=capture.stderr.strip() or capture.stdout.strip(),
                    )
                    _atomic_json_write(
                        control_path,
                        {"action": "pause", "requested_at": time.time()},
                    )
                    continue
                break
        _write_status(status_path, state="completed", pose_index=len(poses), pose_count=len(poses))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-role", required=True, choices=("ur5e", "xarm6"))
    parser.add_argument("--planning-group", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--confirmed", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("capture_command", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    """Run a confirmed wrist-calibration replay."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args()
    if args.capture_command[:1] == ["--"]:
        args.capture_command = args.capture_command[1:]
    if not args.capture_command:
        args.capture_command = [sys.executable, "-c", "raise SystemExit(2)"]
    replay(args)


if __name__ == "__main__":
    main()

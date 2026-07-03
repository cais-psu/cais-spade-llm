#!/usr/bin/env python3
"""ROS2 bridge from UR5e RG2 joint trajectories to URScript RG2 commands."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import socket
import sys
import threading
import time
import xmlrpc.client
from dataclasses import fields
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "cais_spade_llm" / "initialization" / "resources" / "robot_ur5e.json").is_file():
            return parent
    return script_path.parents[3]


_PROJECT_ROOT = _project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cais_spade_llm.resources.robot.hardware_pick_place_controller import (  # noqa: E402
    UR5eRG2GripperController,
    UR5eRG2GripperControllerSettings,
)


def _default_config_path() -> Path:
    return _PROJECT_ROOT / "cais_spade_llm" / "initialization" / "resources" / "robot_ur5e.json"


def _default_status_path() -> Path:
    return Path("/tmp") / "cais_ur5e_rg2_gripper_status.json"


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _load_real_gripper_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ur5e = raw.get("ur5e", {}) if isinstance(raw, dict) else {}
    real = ur5e.get("real", {}) if isinstance(ur5e, dict) else {}
    controller = real.get("controller", {}) if isinstance(real, dict) else {}
    gripper = controller.get("gripper", {}) if isinstance(controller, dict) else {}
    return dict(gripper) if isinstance(gripper, dict) else {}


def _target_from_trajectory(msg: Any, joint_name: str) -> float | None:
    points = list(getattr(msg, "points", []) or [])
    if not points:
        return None
    point = points[-1]
    positions = list(getattr(point, "positions", []) or [])
    if not positions:
        return None
    joint_names = list(getattr(msg, "joint_names", []) or [])
    if joint_names and joint_name in joint_names:
        return float(positions[joint_names.index(joint_name)])
    return float(positions[0])


def _clamp(value: float, lo: float, hi: float) -> float:
    lower = min(float(lo), float(hi))
    upper = max(float(lo), float(hi))
    return max(lower, min(upper, float(value)))


def _width_mm_from_position(
    settings: UR5eRG2GripperControllerSettings,
    position: float,
) -> float:
    open_position = float(settings.open_position)
    close_position = float(settings.close_position)
    if abs(open_position - close_position) < 1e-9:
        return float(settings.open_width_mm)
    ratio = (float(position) - close_position) / (open_position - close_position)
    width = settings.close_width_mm + ratio * (
        settings.open_width_mm - settings.close_width_mm
    )
    return _clamp(width, settings.close_width_mm, settings.open_width_mm)


def _force_for_position(
    settings: UR5eRG2GripperControllerSettings,
    position: float,
) -> float:
    midpoint = (float(settings.open_position) + float(settings.close_position)) * 0.5
    if settings.open_position >= settings.close_position:
        return settings.open_force if float(position) >= midpoint else settings.close_force
    return settings.open_force if float(position) <= midpoint else settings.close_force


def _settle_sec_for_position(
    settings: UR5eRG2GripperControllerSettings,
    position: float,
) -> float:
    midpoint = (float(settings.open_position) + float(settings.close_position)) * 0.5
    if settings.open_position >= settings.close_position:
        return settings.open_settle_sec if float(position) >= midpoint else settings.close_settle_sec
    return settings.open_settle_sec if float(position) <= midpoint else settings.close_settle_sec


def _rg2_status_payload(
    *,
    source: str,
    joint_name: str,
    position: float,
    width_mm: float,
    force: float,
    backend: str,
    hostname: str,
    state: str,
    success: bool | None,
    error: str = "",
    used_backend: str = "",
    fallback_error: str = "",
    rtde_method: str = "",
    disable_remote_control_check: bool = False,
    xmlrpc_url: str = "",
) -> dict[str, Any]:
    return {
        "source": str(source),
        "joint": str(joint_name),
        "position": float(position),
        "width_mm": float(width_mm),
        "force": float(force),
        "backend": str(backend),
        "used_backend": str(used_backend or backend),
        "fallback_error": str(fallback_error),
        "rtde_method": str(rtde_method),
        "disable_remote_control_check": bool(disable_remote_control_check),
        "xmlrpc_url": str(xmlrpc_url),
        "hostname": str(hostname),
        "state": str(state),
        "success": success,
        "error": str(error),
        "updated_at": time.time(),
    }


def _resolve_backend(cli_backend: str | None, gripper_config: dict[str, Any]) -> str:
    backend = str(cli_backend or gripper_config.get("backend") or "secondary_urscript").strip()
    if backend not in {"xmlrpc", "secondary_urscript", "rtde", "urscript_interface", "auto"}:
        return "secondary_urscript"
    return backend


def _normalize_xmlrpc_path(path: Any) -> str:
    text = str(path or "/").strip()
    if not text:
        return "/"
    if not text.startswith("/"):
        text = f"/{text}"
    return text


def _xmlrpc_url(hostname: str, port: int, path: str) -> str:
    return f"http://{hostname}:{int(port)}{_normalize_xmlrpc_path(path)}"


def _settings_payload(settings: UR5eRG2GripperControllerSettings) -> dict[str, Any]:
    return {field.name: getattr(settings, field.name) for field in fields(settings)}


def _rtde_command_worker(
    settings_payload: dict[str, Any],
    width_mm: float,
    force: float,
    settle_sec: float,
    result_queue: Any,
) -> None:
    controller: UR5eRG2GripperController | None = None
    try:
        settings = UR5eRG2GripperControllerSettings(**settings_payload)
        controller = UR5eRG2GripperController.from_settings(settings)
        controller.command_width(
            width_mm,
            force,
            settle_sec=settle_sec,
            blocking=True,
        )
        result_queue.put({"ok": True})
    except Exception as exc:
        result_queue.put({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        if controller is not None:
            controller.disconnect()


def _xmlrpc_command_worker(
    hostname: str,
    port: int,
    path: str,
    timeout_sec: float,
    width_mm: float,
    force: float,
    settle_sec: float,
    result_queue: Any,
) -> None:
    old_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(float(timeout_sec))
        rg = xmlrpc.client.ServerProxy(
            _xmlrpc_url(hostname, int(port), path),
            allow_none=True,
        )
        result = rg.rg_grip(0, float(width_mm), float(force))
        if settle_sec > 0.0:
            time.sleep(float(settle_sec))
        result_queue.put({"ok": True, "result": result})
    except Exception as exc:
        result_queue.put({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        socket.setdefaulttimeout(old_timeout)


class UR5eRG2RTDEBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        import rclpy
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionServer
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from trajectory_msgs.msg import JointTrajectory

        self.rclpy = rclpy
        self.FollowJointTrajectory = FollowJointTrajectory
        self.ActionServer = ActionServer
        self.JointState = JointState
        self.String = String
        self.JointTrajectory = JointTrajectory

        self.node: Node = rclpy.create_node("ur5e_rg2_rtde_gripper")
        gripper_config = _load_real_gripper_config(Path(args.config).expanduser())
        settings = UR5eRG2GripperControllerSettings.from_config(
            gripper_config,
            hostname=str(args.robot_ip or "").strip() or None,
        )
        self.settings = settings
        self.joint_name = str(args.joint or gripper_config.get("joint") or "ur5e_rg2_finger_width")
        self.topic = str(
            args.topic
            or gripper_config.get("topic")
            or "/ur5e_rg2_gripper_traj_controller/joint_trajectory"
        )
        self.action_name = str(args.action or "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory")
        self.script_port = int(args.script_port or gripper_config.get("script_port") or 30002)
        self.script_timeout_sec = max(0.1, float(args.script_timeout_sec))
        self.rtde_timeout_sec = max(
            0.1,
            float(args.rtde_timeout_sec or gripper_config.get("rtde_timeout_sec") or 8.0),
        )
        self.script_command_topic = str(
            args.script_command_topic
            or gripper_config.get("script_command_topic")
            or "/urscript_interface/script_command"
        )
        xmlrpc_config = dict(gripper_config.get("xmlrpc") or {})
        self.xmlrpc_port = int(args.xmlrpc_port or xmlrpc_config.get("port") or 41414)
        self.xmlrpc_path = _normalize_xmlrpc_path(args.xmlrpc_path or xmlrpc_config.get("path") or "/")
        self.xmlrpc_timeout_sec = max(
            0.1,
            float(args.xmlrpc_timeout_sec or xmlrpc_config.get("timeout_sec") or 3.0),
        )
        self.xmlrpc_url = _xmlrpc_url(settings.hostname, self.xmlrpc_port, self.xmlrpc_path)
        self.publish_rate_hz = max(1.0, float(args.publish_rate_hz))
        self.status_file = Path(args.status_file or _default_status_path()).expanduser()
        self.backend = _resolve_backend(args.backend, gripper_config)
        self.current_position = (float(settings.open_position) + float(settings.close_position)) / 2.0
        self.publish_arm_joint_states = bool(args.publish_arm_joint_states)
        self._arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self._rtde_receive = None
        self._last_arm_positions: list[float] | None = None
        self._last_arm_warning_ts = 0.0
        self._lock = threading.Lock()

        if self.publish_arm_joint_states:
            try:
                import rtde_receive

                self._rtde_receive = rtde_receive.RTDEReceiveInterface(settings.hostname)
            except Exception as exc:
                self.node.get_logger().warning(
                    "UR5e arm joint-state relay disabled: "
                    f"{type(exc).__name__}: {exc}"
                )

        self._joint_state_pub = self.node.create_publisher(JointState, "/joint_states", 10)
        self._script_command_pub = self.node.create_publisher(String, self.script_command_topic, 10)
        self._trajectory_sub = self.node.create_subscription(
            JointTrajectory,
            self.topic,
            self._on_trajectory,
            10,
        )
        self._action_server = ActionServer(
            self.node,
            FollowJointTrajectory,
            self.action_name,
            execute_callback=self._execute_action,
        )
        self._timer = self.node.create_timer(1.0 / self.publish_rate_hz, self._publish_joint_state)
        self.node.get_logger().info(
            f"UR5e RG2 ROS bridge ready: backend={self.backend} host={settings.hostname} "
            f"joint={self.joint_name} topic={self.topic} action={self.action_name}; "
            f"script_command_topic={self.script_command_topic} "
            f"secondary_urscript_port={self.script_port} xmlrpc_url={self.xmlrpc_url} "
            f"status_file={self.status_file} "
            f"publish_arm_joint_states={self.publish_arm_joint_states}"
        )

    def _read_arm_positions(self) -> list[float] | None:
        if self._rtde_receive is None:
            return self._last_arm_positions
        try:
            joints = [float(value) for value in self._rtde_receive.getActualQ()]
            if len(joints) >= 6:
                self._last_arm_positions = joints[:6]
        except Exception as exc:
            now = time.time()
            if now - self._last_arm_warning_ts > 5.0:
                self.node.get_logger().warning(
                    "UR5e arm joint-state relay read failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._last_arm_warning_ts = now
        return self._last_arm_positions

    def _publish_joint_state(self) -> None:
        msg = self.JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        names: list[str] = []
        positions: list[float] = []
        if self.publish_arm_joint_states:
            arm_positions = self._read_arm_positions()
            if arm_positions is not None:
                names.extend(self._arm_joint_names)
                positions.extend(float(value) for value in arm_positions)
        names.append(self.joint_name)
        with self._lock:
            positions.append(float(self.current_position))
        msg.name = names
        msg.position = positions
        self._joint_state_pub.publish(msg)

    def _write_command_status(
        self,
        *,
        source: str,
        position: float,
        width_mm: float,
        force: float,
        state: str,
        success: bool | None,
        error: str = "",
        used_backend: str = "",
        fallback_error: str = "",
    ) -> None:
        try:
            _atomic_json_write(
                self.status_file,
                _rg2_status_payload(
                    source=source,
                    joint_name=self.joint_name,
                    position=position,
                    width_mm=width_mm,
                    force=force,
                    backend=self.backend,
                    used_backend=used_backend,
                    fallback_error=fallback_error,
                    rtde_method=self.settings.rtde_method,
                    disable_remote_control_check=self.settings.disable_remote_control_check,
                    xmlrpc_url=self.xmlrpc_url,
                    hostname=self.settings.hostname,
                    state=state,
                    success=success,
                    error=error,
                ),
            )
        except Exception as exc:
            self.node.get_logger().warning(f"Failed to write RG2 status file: {exc}")

    def _command_position(self, position: float, source: str = "command") -> bool:
        with self._lock:
            self.current_position = float(position)
        width_mm = _width_mm_from_position(self.settings, position)
        force = _force_for_position(self.settings, position)
        settle_sec = _settle_sec_for_position(self.settings, position)
        self._write_command_status(
            source=source,
            position=position,
            width_mm=width_mm,
            force=force,
            state="received",
            success=None,
        )
        used_backend = self.backend
        fallback_error = ""
        try:
            if self.backend == "xmlrpc":
                self._command_width_xmlrpc(width_mm, force, settle_sec)
            elif self.backend == "rtde":
                self._command_width_rtde(width_mm, force, settle_sec)
            elif self.backend == "urscript_interface":
                used_backend = "urscript_interface"
                self._command_width_urscript_interface(width_mm, force, settle_sec)
            elif self.backend == "auto":
                try:
                    used_backend = "xmlrpc"
                    self._command_width_xmlrpc(width_mm, force, settle_sec)
                except Exception as exc:
                    fallback_error = str(exc)
                    used_backend = "rtde"
                    self.node.get_logger().warning(
                        "RG2 XMLRPC backend failed; falling back to rtde: "
                        f"{fallback_error}"
                    )
                    self._command_width_rtde(width_mm, force, settle_sec)
            else:
                self._command_width_secondary_urscript(width_mm, force, settle_sec)
            self.node.get_logger().info(
                f"RG2 command: {self.joint_name}={float(position):.3f} "
                f"width_mm={width_mm:.1f} backend={self.backend} "
                f"used_backend={used_backend} "
                f"host={self.settings.hostname}"
            )
            self._write_command_status(
                source=source,
                position=position,
                width_mm=width_mm,
                force=force,
                state="success",
                success=True,
                used_backend=used_backend,
                fallback_error=fallback_error,
            )
            return True
        except Exception as exc:
            error = str(exc)
            self.node.get_logger().error(
                f"RG2 command failed via {self.backend}"
                f"{f'/{used_backend}' if used_backend != self.backend else ''}: {error}"
            )
            self._write_command_status(
                source=source,
                position=position,
                width_mm=width_mm,
                force=force,
                state="error",
                success=False,
                error=error,
                used_backend=used_backend,
                fallback_error=fallback_error,
            )
            return False

    def _command_width_xmlrpc(self, width_mm: float, force: float, settle_sec: float) -> None:
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        proc = ctx.Process(
            target=_xmlrpc_command_worker,
            args=(
                self.settings.hostname,
                self.xmlrpc_port,
                self.xmlrpc_path,
                self.xmlrpc_timeout_sec,
                float(width_mm),
                float(force),
                float(settle_sec),
                result_queue,
            ),
        )
        proc.start()
        proc.join(self.xmlrpc_timeout_sec + max(0.0, float(settle_sec)) + 1.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(1.0)
            raise TimeoutError(
                f"XMLRPC RG2 command timed out after {self.xmlrpc_timeout_sec:.1f}s"
            )
        if proc.exitcode != 0:
            raise RuntimeError(f"XMLRPC RG2 command process exited with code {proc.exitcode}")
        try:
            result = result_queue.get(timeout=0.2)
        except queue.Empty:
            raise RuntimeError("XMLRPC RG2 command process returned no result") from None
        error = result.get("error")
        if error:
            raise RuntimeError(str(error))

    def _command_width_rtde(self, width_mm: float, force: float, settle_sec: float) -> None:
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        proc = ctx.Process(
            target=_rtde_command_worker,
            args=(
                _settings_payload(self.settings),
                float(width_mm),
                float(force),
                float(settle_sec),
                result_queue,
            ),
        )
        proc.start()
        proc.join(self.rtde_timeout_sec)
        if proc.is_alive():
            proc.terminate()
            proc.join(1.0)
            raise TimeoutError(f"RTDE RG2 command timed out after {self.rtde_timeout_sec:.1f}s")
        if proc.exitcode != 0:
            raise RuntimeError(f"RTDE RG2 command process exited with code {proc.exitcode}")
        try:
            result = result_queue.get(timeout=0.2)
        except queue.Empty:
            raise RuntimeError("RTDE RG2 command process returned no result") from None
        error = result.get("error")
        if error:
            raise RuntimeError(str(error))

    def _command_width_urscript_interface(
        self,
        width_mm: float,
        force: float,
        settle_sec: float,
    ) -> None:
        if self._script_command_pub.get_subscription_count() <= 0:
            raise RuntimeError(f"{self.script_command_topic} has no subscribers")
        msg = self.String()
        msg.data = UR5eRG2GripperController.secondary_program_body(width_mm, force)
        self._script_command_pub.publish(msg)
        if settle_sec > 0.0:
            time.sleep(float(settle_sec))

    def _command_width_secondary_urscript(
        self,
        width_mm: float,
        force: float,
        settle_sec: float,
    ) -> None:
        self._send_secondary_program(
            UR5eRG2GripperController.secondary_program_body(width_mm, force)
        )
        if settle_sec > 0.0:
            time.sleep(float(settle_sec))

    def _send_secondary_program(self, program: str) -> None:
        payload = str(program).encode("utf-8")
        with socket.create_connection(
            (self.settings.hostname, self.script_port),
            timeout=self.script_timeout_sec,
        ) as sock:
            sock.settimeout(self.script_timeout_sec)
            sock.sendall(payload)

    def _on_trajectory(self, msg: Any) -> None:
        target = _target_from_trajectory(msg, self.joint_name)
        if target is None:
            self.node.get_logger().warning("Ignored RG2 trajectory with no target position")
            return
        threading.Thread(
            target=self._command_position,
            args=(target, "topic"),
            daemon=True,
        ).start()

    def _execute_action(self, goal_handle: Any) -> Any:
        result = self.FollowJointTrajectory.Result()
        target = _target_from_trajectory(goal_handle.request.trajectory, self.joint_name)
        if target is None:
            goal_handle.abort()
            result.error_code = -1
            result.error_string = "trajectory has no target position"
            return result
        if self._command_position(target, source="action"):
            goal_handle.succeed()
            result.error_code = 0
            result.error_string = ""
            return result
        goal_handle.abort()
        result.error_code = -4
        result.error_string = "RG2 command failed"
        return result

    def run(self) -> None:
        try:
            self.rclpy.spin(self.node)
        finally:
            if self._rtde_receive is not None:
                try:
                    self._rtde_receive.disconnect()
                except Exception:
                    pass
            self.node.destroy_node()
            if self.rclpy.ok():
                self.rclpy.shutdown()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UR5e RG2 RTDE gripper ROS2 bridge")
    parser.add_argument("--config", default=str(_default_config_path()))
    parser.add_argument("--robot-ip", default="")
    parser.add_argument("--joint", default="")
    parser.add_argument("--topic", default="")
    parser.add_argument("--action", default="")
    parser.add_argument(
        "--backend",
        choices=("xmlrpc", "secondary_urscript", "rtde", "urscript_interface", "auto"),
        default=None,
    )
    parser.add_argument("--script-port", type=int, default=30002)
    parser.add_argument("--rtde-timeout-sec", type=float, default=0.0)
    parser.add_argument("--xmlrpc-port", type=int, default=0)
    parser.add_argument("--xmlrpc-path", default="")
    parser.add_argument("--xmlrpc-timeout-sec", type=float, default=0.0)
    parser.add_argument("--script-command-topic", default="")
    parser.add_argument("--script-timeout-sec", type=float, default=2.0)
    parser.add_argument("--publish-rate-hz", type=float, default=20.0)
    parser.add_argument(
        "--publish-arm-joint-states",
        action="store_true",
        help="Also publish UR5e arm joint states from RTDE receive on /joint_states.",
    )
    parser.add_argument("--status-file", default=str(_default_status_path()))
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    import rclpy

    rclpy.init()
    UR5eRG2RTDEBridge(args).run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Mirror validated physical gear poses into the passive Gazebo world."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, TextIO

from ament_index_python import get_package_share_directory

PART_MODELS = {"SG": "gear_small", "MG": "gear_medium", "LG": "gear_large"}
DEFAULT_LOCK_PATH = Path("/tmp/cais_physical_part_twin_sync.lock")

log = logging.getLogger(__name__)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _acquire_singleton_lock(path: Path) -> TextIO | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


class PhysicalPartTwinSync:
    """Spawn, update, attach, and release physical gear mirrors."""

    def __init__(
        self,
        *,
        snapshot_path: Path,
        ownership_path: Path,
        status_path: Path,
        deadband_m: float,
        maximum_detection_age_sec: float,
        robot_model_names: list[str],
        attach_link_candidates: list[str],
    ) -> None:
        import rclpy
        from gazebo_msgs.srv import GetEntityState, SetEntityState, SpawnEntity
        from rclpy.node import Node

        self.rclpy = rclpy
        self.node = Node("physical_part_twin_sync")
        self.snapshot_path = snapshot_path
        self.ownership_path = ownership_path
        self.status_path = status_path
        self.deadband_m = float(deadband_m)
        self.maximum_detection_age_sec = float(maximum_detection_age_sec)
        self.robot_model_names = robot_model_names
        self.attach_link_candidates = attach_link_candidates
        self._GetEntityState = GetEntityState
        self._SetEntityState = SetEntityState
        self._SpawnEntity = SpawnEntity
        self._get_client = self.node.create_client(GetEntityState, "/get_entity_state")
        self._set_client = self.node.create_client(SetEntityState, "/set_entity_state")
        self._spawn_client = self.node.create_client(SpawnEntity, "/spawn_entity")
        self._attach_type, self._detach_type = self._load_linkattacher_services()
        self._attach_client = (
            self.node.create_client(self._attach_type, "/ATTACHLINK")
            if self._attach_type is not None
            else None
        )
        self._detach_client = (
            self.node.create_client(self._detach_type, "/DETACHLINK")
            if self._detach_type is not None
            else None
        )
        self._held_models: set[str] = set()
        self._release_after: dict[str, float] = {}
        self._last_ownership_sequence = ""
        self._degraded_reason = ""
        self._waiting_reason = "waiting for validated world pose"
        self._mirrored_models: set[str] = set()
        self._model_root = Path(get_package_share_directory("cais_lab_robotics")) / "models"

    @staticmethod
    def _load_linkattacher_services() -> tuple[Any | None, Any | None]:
        try:
            from linkattacher_msgs.srv import AttachLink, DetachLink
        except ImportError:
            return None, None
        return AttachLink, DetachLink

    def wait_for_gazebo(self, timeout_sec: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        for client, name in (
            (self._get_client, "/get_entity_state"),
            (self._set_client, "/set_entity_state"),
            (self._spawn_client, "/spawn_entity"),
        ):
            remaining = max(0.0, deadline - time.monotonic())
            if not client.wait_for_service(timeout_sec=remaining):
                self._degraded_reason = f"Gazebo service unavailable: {name}"
                self._write_status([])
                return False
        return True

    def _call(self, client: Any, request: Any, *, timeout_sec: float = 3.0) -> Any | None:
        future = client.call_async(request)
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
        if not future.done():
            return None
        return future.result()

    def _entity_state(self, model_name: str) -> Any | None:
        request = self._GetEntityState.Request()
        request.name = model_name
        request.reference_frame = "world"
        response = self._call(self._get_client, request)
        if response is None or not response.success:
            return None
        return response.state

    @staticmethod
    def _pose_from_row(row: dict[str, Any]) -> Any:
        from geometry_msgs.msg import Pose

        pose = Pose()
        pose.position.x = float(row["x"])
        pose.position.y = float(row["y"])
        pose.position.z = float(row["z"])
        pose.orientation.x = float(row.get("qx", 0.0))
        pose.orientation.y = float(row.get("qy", 0.0))
        pose.orientation.z = float(row.get("qz", 0.0))
        pose.orientation.w = float(row.get("qw", 1.0))
        return pose

    def _spawn(self, model_name: str, row: dict[str, Any]) -> bool:
        sdf_path = self._model_root / model_name / "model.sdf"
        try:
            xml = sdf_path.read_text(encoding="utf-8")
        except OSError as exc:
            self._degraded_reason = f"cannot read {sdf_path}: {exc}"
            return False
        request = self._SpawnEntity.Request()
        request.name = model_name
        request.xml = xml
        request.robot_namespace = ""
        request.initial_pose = self._pose_from_row(row)
        request.reference_frame = "world"
        response = self._call(self._spawn_client, request, timeout_sec=5.0)
        if response is None or not response.success:
            message = getattr(response, "status_message", "spawn timed out")
            self._degraded_reason = f"failed to spawn {model_name}: {message}"
            return False
        return True

    def _update(self, model_name: str, row: dict[str, Any], state: Any) -> bool:
        target = self._pose_from_row(row)
        horizontal_distance = math.sqrt(
            (float(state.pose.position.x) - target.position.x) ** 2
            + (float(state.pose.position.y) - target.position.y) ** 2
        )
        if horizontal_distance <= self.deadband_m:
            return True
        request = self._SetEntityState.Request()
        request.state.name = model_name
        request.state.pose = target
        request.state.reference_frame = "world"
        response = self._call(self._set_client, request)
        if response is None or not response.success:
            message = getattr(response, "status_message", "update timed out")
            self._degraded_reason = f"failed to update {model_name}: {message}"
            return False
        return True

    def _link_request(self, service_type: Any, robot_model: str, robot_link: str, model_name: str) -> Any:
        request = service_type.Request()
        request.model1_name = robot_model
        request.link1_name = robot_link
        request.model2_name = model_name
        request.link2_name = "link"
        return request

    def _attach(self, model_name: str) -> bool:
        if self._attach_client is None or self._attach_type is None:
            self._degraded_reason = "linkattacher_msgs is unavailable in the Gazebo domain"
            return False
        if not self._attach_client.wait_for_service(timeout_sec=2.0):
            self._degraded_reason = "/ATTACHLINK is unavailable"
            return False
        for robot_model in self.robot_model_names:
            for robot_link in self.attach_link_candidates:
                response = self._call(
                    self._attach_client,
                    self._link_request(self._attach_type, robot_model, robot_link, model_name),
                )
                if response is not None and bool(getattr(response, "success", False)):
                    self._held_models.add(model_name)
                    return True
        self._degraded_reason = f"failed to attach {model_name} to mirrored UR5e"
        return False

    def _detach(self, model_name: str) -> bool:
        if self._detach_client is None or self._detach_type is None:
            self._degraded_reason = "linkattacher_msgs is unavailable in the Gazebo domain"
            return False
        if not self._detach_client.wait_for_service(timeout_sec=2.0):
            self._degraded_reason = "/DETACHLINK is unavailable"
            return False
        for robot_model in self.robot_model_names:
            for robot_link in self.attach_link_candidates:
                response = self._call(
                    self._detach_client,
                    self._link_request(self._detach_type, robot_model, robot_link, model_name),
                )
                if response is not None and bool(getattr(response, "success", False)):
                    self._held_models.discard(model_name)
                    return True
        self._degraded_reason = f"failed to detach {model_name} from mirrored UR5e"
        return False

    def _process_ownership(self) -> None:
        event = _read_json(self.ownership_path)
        sequence = str(event.get("sequence") or "")
        if not sequence or sequence == self._last_ownership_sequence:
            return
        self._last_ownership_sequence = sequence
        model_name = str(event.get("model_name") or "").strip()
        if model_name not in PART_MODELS.values():
            return
        action = str(event.get("action") or "").strip()
        if action == "held":
            self._attach(model_name)
        elif action == "released":
            self._detach(model_name)
            self._release_after[model_name] = float(event.get("occurred_at", time.time()))

    def _accepted_rows(self) -> list[dict[str, Any]]:
        snapshot = _read_json(self.snapshot_path)
        if not bool(snapshot.get("table_plane_ready", False)):
            self._waiting_reason = "physical table plane is not calibrated"
            return []
        rows = snapshot.get("detections", [])
        now = time.time()
        if not isinstance(rows, list):
            self._waiting_reason = "perception snapshot has no detection list"
            return []
        accepted = [
            row
            for row in rows
            if isinstance(row, dict)
            and row.get("part_name") in PART_MODELS
            and row.get("model_name") == PART_MODELS[row["part_name"]]
            and now - float(row.get("captured_at", 0.0)) <= self.maximum_detection_age_sec
        ]
        if accepted:
            self._waiting_reason = ""
            return accepted
        last_error = str(snapshot.get("last_error") or "").strip()
        if last_error:
            self._waiting_reason = f"world pose rejected: {last_error}"
        elif rows:
            self._waiting_reason = "latest validated detection is stale or unsupported"
        else:
            self._waiting_reason = "waiting for validated world pose"
        return []

    def sync_once(self) -> None:
        self._degraded_reason = ""
        self._process_ownership()
        rows = self._accepted_rows()
        for row in rows:
            model_name = str(row["model_name"])
            if model_name in self._held_models:
                continue
            release_after = self._release_after.get(model_name, 0.0)
            if float(row.get("captured_at", 0.0)) <= release_after:
                continue
            state = self._entity_state(model_name)
            if state is None:
                if self._spawn(model_name, row):
                    self._mirrored_models.add(model_name)
            elif self._update(model_name, row, state):
                self._mirrored_models.add(model_name)
        self._write_status()

    def _write_status(self, mirrored: list[str] | None = None) -> None:
        if mirrored:
            self._mirrored_models.update(mirrored)
        heartbeat_at = time.time()
        if self._degraded_reason:
            state = "degraded"
        elif self._mirrored_models:
            state = "mirrored"
        elif self._waiting_reason:
            state = "waiting"
        else:
            state = "ready"
        _atomic_write_json(
            self.status_path,
            {
                "updated_at": heartbeat_at,
                "heartbeat_at": heartbeat_at,
                "synchronizer_pid": os.getpid(),
                "state": state,
                "degraded_reason": self._degraded_reason,
                "waiting_reason": self._waiting_reason,
                "mirrored_models": sorted(self._mirrored_models),
                "held_models": sorted(self._held_models),
                "deadband_m": self.deadband_m,
            },
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="/tmp/cais_physical_perception.json")
    parser.add_argument("--ownership", default="/tmp/cais_physical_part_ownership.json")
    parser.add_argument("--status", default="/tmp/cais_physical_part_twin_status.json")
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--deadband-m", type=float, default=0.002)
    parser.add_argument("--maximum-detection-age-sec", type=float, default=15.0)
    parser.add_argument("--poll-sec", type=float, default=0.5)
    parser.add_argument(
        "--robot-model-names",
        default="dual_robot,ur5e_rg2",
    )
    parser.add_argument(
        "--attach-link-candidates",
        default="ur5e_rg2_gripper_tcp,ur5e_tool0,ur5e_wrist_3_link",
    )
    return parser


def main() -> None:
    """Run the file bridge in the current Gazebo ROS domain."""
    import rclpy

    args = _build_parser().parse_args()
    lock_path = Path(args.lock_file)
    lock_handle = _acquire_singleton_lock(lock_path)
    if lock_handle is None:
        log.warning("Physical part twin synchronizer is already running: %s", lock_path)
        return
    rclpy.init()
    sync = PhysicalPartTwinSync(
        snapshot_path=Path(args.snapshot),
        ownership_path=Path(args.ownership),
        status_path=Path(args.status),
        deadband_m=args.deadband_m,
        maximum_detection_age_sec=args.maximum_detection_age_sec,
        robot_model_names=[item.strip() for item in args.robot_model_names.split(",") if item.strip()],
        attach_link_candidates=[
            item.strip() for item in args.attach_link_candidates.split(",") if item.strip()
        ],
    )
    try:
        if not sync.wait_for_gazebo():
            raise SystemExit(2)
        while rclpy.ok():
            sync.sync_once()
            rclpy.spin_once(sync.node, timeout_sec=max(0.05, float(args.poll_sec)))
    finally:
        sync.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    main()

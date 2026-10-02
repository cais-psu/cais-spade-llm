"""Display non-colliding, resource-specific failure markers in Gazebo Classic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

MODEL_NAME = "cais_Conveyor_breakdown"


def marker_geometry(scene: dict) -> dict:
    """Locate the selected marker from configured geometry or observed part pose."""
    failure = scene.get("_failure_marker") or {"scenario": "Conveyor breakdown", "resource_id": "Conveyor"}
    scenario, rid = failure["scenario"], failure["resource_id"]
    if scenario == "Conveyor breakdown":
        belt = scene["Conveyor"]
        return {"model_name": MODEL_NAME, "pose": belt["world_pose"],
                "length": float(belt["length"]), "width": float(belt["width"]),
                "height": float(belt["surface_height"]) - belt["world_pose"][2],
                "material": "CAIS/ConveyorBreakdown", "label": scenario}
    if scenario == "Machining breakdown during part processing":
        machine = next(row for row in scene["machines"] if row["resource_id"] == rid)
        return {"model_name": f"cais_{rid}_breakdown", "pose": machine["world_pose"],
                "length": 0.65, "width": 0.65,
                "height": machine["workholding_pose"][2] - machine["world_pose"][2],
                "material": f"CAIS/{rid}Breakdown", "label": scenario + " (" + rid + ")"}
    robot = next(row for row in scene["robots"] if row["resource_id"] == rid)
    pose = list(robot["base_xyz"]) + list(robot["base_rpy"])
    if scenario == "ur5e-1 breakdown":
        return {"model_name": "cais_ur5e_1_breakdown", "pose": pose,
                "length": 0.5, "width": 0.5, "height": 0.05,
                "material": "CAIS/RobotBreakdown", "label": scenario}
    if scenario != "Part slippage":
        raise ValueError("Unknown failure marker")
    observed = failure.get("evidence", {}).get("observed_drop_pose")
    if observed:
        pose = [observed[axis] for axis in ("x", "y", "z")] + [0, 0, 0]
    return {"model_name": "cais_Part_slippage", "pose": pose,
            "length": 0.18, "width": 0.18, "height": 0.0,
            "material": "CAIS/PartSlippage", "label": scenario + " (" + rid + ")"}


def marker_sdf(scene: dict) -> str:
    """Build an outline and two-sided sign without collision geometry."""
    geometry = marker_geometry(scene)
    x, y, z, roll, pitch, yaw = geometry["pose"]
    length, width, height = geometry["length"], geometry["width"], geometry["height"]
    rails = []
    for name, px, py, sx, sy in (
        ("left", 0, -width / 2 - 0.035, length, 0.04),
        ("right", 0, width / 2 + 0.035, length, 0.04),
        ("start", -length / 2, 0, 0.04, width + 0.11),
        ("end", length / 2, 0, 0.04, width + 0.11),
    ):
        rails.append(
            f'<visual name="{name}"><pose>{px} {py} {height + 0.04} 0 0 0</pose>'
            f"<geometry><box><size>{sx} {sy} .045</size></box></geometry>"
            "<material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse>"
            "<emissive>.8 0 0 1</emissive></material></visual>"
        )
    sign = (
        '<visual name="failure_label">'
        f"<pose>0 0 {height + 0.55} 0 0 0</pose>"
        "<geometry><box><size>1.4 .025 .28</size></box></geometry>"
        "<material><script><uri>model://conveyor_fault_marker/materials/scripts</uri>"
        "<uri>model://conveyor_fault_marker/materials/textures</uri>"
        f"<name>{escape(geometry['material'])}</name></script></material></visual>"
    )
    pose = " ".join(str(v) for v in (x, y, z, roll, pitch, yaw))
    return (
        f'<sdf version="1.6"><model name="{escape(geometry["model_name"])}"><static>true</static>'
        f'<pose>{escape(pose)}</pose><link name="marker">'
        + "".join(rails)
        + sign
        + "</link></model></sdf>"
    )


def run(scene: dict, action: str) -> dict:
    """Observe, spawn or remove the marker using bounded Gazebo service calls."""
    import rclpy
    from gazebo_msgs.srv import DeleteEntity, GetEntityState, SpawnEntity

    geometry = marker_geometry(scene)
    model_name = geometry["model_name"]
    rclpy.init(args=[])
    node = rclpy.create_node("conveyor_breakdown_marker")

    def call(kind, name, request):
        client = node.create_client(kind, name)
        try:
            if not client.wait_for_service(timeout_sec=8):
                raise RuntimeError(f"Gazebo service unavailable: {name}")
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=8)
            if not future.done() or future.result() is None:
                raise RuntimeError(f"Gazebo service timed out: {name}")
            return future.result()
        finally:
            node.destroy_client(client)

    try:
        request = GetEntityState.Request()
        request.name = model_name
        request.reference_frame = "world"
        present = call(GetEntityState, "/get_entity_state", request).success
        if action == "clear" and present:
            request = DeleteEntity.Request()
            request.name = model_name
            result = call(DeleteEntity, "/delete_entity", request)
            if not result.success:
                raise RuntimeError(result.status_message)
        elif action == "show" and not present:
            request = SpawnEntity.Request()
            request.name = model_name
            request.xml = marker_sdf(scene)
            request.reference_frame = "world"
            request.initial_pose.orientation.w = 1.0
            result = call(SpawnEntity, "/spawn_entity", request)
            if not result.success:
                raise RuntimeError(result.status_message)
        request = GetEntityState.Request()
        request.name = model_name
        request.reference_frame = "world"
        observed = call(GetEntityState, "/get_entity_state", request)
        if observed.success != (action == "show"):
            raise RuntimeError("Gazebo marker observation disagrees with requested fault state")
        return {
            "status": "completed",
            "model_name": model_name,
            "present": observed.success,
            "label": geometry["label"],
            "collision_geometry": False,
        }
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    """Run only the marker operation explicitly requested by the simulation runtime."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--action", required=True, choices=("show", "clear"))
    args = parser.parse_args()
    try:
        result = run(json.loads(args.scene.read_text()), args.action)
    except (ValueError, RuntimeError, OSError) as exc:
        sys.stderr.write(str(exc))
        raise SystemExit(1) from exc
    sys.stdout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()

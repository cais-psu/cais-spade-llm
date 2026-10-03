"""Record observed Gazebo failures and assembly_board-v1 mutex enforcement."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.failure_checkpoints import CHECKPOINTS
from cais_spade_llm.recovery_framework.gazebo_recording import RecordingAttempt, X11Capture, _write_json
from cais_spade_llm.ui.recovery_setup import (
    SETUP_PATH, load_setup, save_setup, slippage_example, validate_setup,
)

logger = logging.getLogger(__name__)
VIDEO_ROOT = ROOT / "cais_spade_llm/monitor/recovery_gazebo_runs/videos-20x"
EVIDENCE_ROOT = ROOT / "cais_spade_llm/monitor/recovery_gazebo_runs"
TITLES = (
    "Conveyor breakdown",
    "Machining station handling robot breakdown (ur5e-1)",
    "Machining breakdown during part processing (M1)",
    "Part slippage (ur5e-3 part slips into ur5e-4’s region)",
    "mutex rule applied into ur5e 3 and ur5e4",
)
SCENARIOS = tuple(CHECKPOINTS)
MUTEX_TEXT = (
    "[Safety Requirements]\n"
    "- ur5e-3 and ur5e-4 must not be in the assembly_board-v1 destination area "
    "at the same time.\n"
)


def mutex_rule() -> dict:
    """Bind entry events and persistent resource_location to the exact robots."""
    return {
        "id": "workspace_mutex", "raw_text": MUTEX_TEXT.splitlines()[1][2:],
        "constraint_type": "mutex", "process": "assembly", "product": "any",
        "resources": ["ur5e-3", "ur5e-4"],
        "ltlf": "G !((ap3 | ap5) & (ap4 | ap6))",
        "aps": [
            {"label": "ap3", "full": "ap_state/assembly/any/ur5e-3/resource_location=assembly_board-v1/any"},
            {"label": "ap4", "full": "ap_state/assembly/any/ur5e-4/resource_location=assembly_board-v1/any"},
            {"label": "ap5", "full": "ap_event/assembly/any/ur5e-3/place_approach/destination_location=assembly_board-v1"},
            {"label": "ap6", "full": "ap_event/assembly/any/ur5e-4/place_approach/destination_location=assembly_board-v1"},
        ],
    }


def scenario_setup(number: int, safety_file: str | None = None) -> dict:
    """Select the two-part order and existing once-per-run injection bindings."""
    setup = load_setup()
    setup.update(
        selected_product_order_file="cais_spade_llm/specification/products/orders/assembly_board-v1-two-parts.json",
        execution_mode="simulation", diagnostic_cca_bypass=False, failure_scenario=None,
    )
    if safety_file is not None:
        setup["selected_safety_file"] = safety_file
    if number == 5:
        setup["permitted_resources"] = [rid for rid in setup["permitted_resources"]
                                        if rid not in {"M2", "ur5e-2"}]
    if number < 5:
        scenario = SCENARIOS[number - 1]
        if scenario == "Part slippage":
            models = validate_setup(setup)["models"]
            failure = slippage_example(models, "ur5e-3", "KET4_Square_4mm")
            failure["additional_condition"]["part_name"] = "gear_small"
            failure["drop_pose"] = {"x": 0.0, "y": -0.2, "z": 1.04}
        else:
            failure = {
                "scenario": scenario, "resource_id": ("Conveyor", "ur5e-1", "M1")[number - 1],
                "checkpoint": CHECKPOINTS[scenario], "mode": "once",
            }
        setup["failure_scenario"] = failure
    validate_setup(setup)
    return setup


def validate_failure(sample: dict, configuration: dict) -> dict:
    """Require completed, correlated injection and the observed failure marker."""
    fault = sample["fault"]
    evidence = fault.get("evidence", {})
    if (fault.get("status") != "triggered" or fault.get("run_id") != sample["run_id"]
            or evidence.get("run_id") != sample["run_id"]
            or fault.get("scenario") != configuration["scenario"]
            or evidence.get("checkpoint") != configuration["checkpoint"]
            or evidence.get("injection_status") != "completed"
            or fault.get("visual", {}).get("status") != "completed"):
        raise ValueError("Failure injection or visible marker is not confirmed for this run")
    if sample["diagnostic_cca_bypass"]:
        raise ValueError("Recording requires CCA enabled")
    if configuration["resource_id"] not in sample["unavailable_resources"]:
        raise ValueError("Failed resource is still available")
    before = evidence.get("resource_values_before", {})
    part = evidence.get("part_name")
    product_before = evidence.get("part_tracker_before", {}).get(part, {})
    if configuration["scenario"] == "Conveyor breakdown":
        if (not part or before.get("ur5e-1", {}).get("held_part") != part
                or evidence.get("source") != "M1"
                or product_before.get("location") != "ur5e-1"
                or not product_before.get("processCompleted")):
            raise ValueError("Completed M1 part was not acquired before Conveyor breakdown")
    if configuration["scenario"] == "ur5e-1 breakdown":
        if (not part or before.get("M1", {}).get("resource_state") != "completed"
                or before.get("M1", {}).get("part_name") != part
                or before.get("ur5e-1", {}).get("held_part") is not None
                or product_before.get("location") != "M1"
                or not product_before.get("processCompleted")):
            raise ValueError("M1 completion before ur5e-1 pickup was not observed")
    if configuration["scenario"] == "Machining breakdown during part processing":
        fraction = evidence.get("processing_fraction")
        if (evidence.get("process_completed") is not False
                or evidence.get("source") != "gazebo_workholding_observation"
                or not isinstance(fraction, (int, float)) or not .499999 <= fraction <= .500001):
            raise ValueError("Interrupted machining halfway through processing was not observed")
    if configuration["scenario"] == "Part slippage":
        detached = evidence.get("detach", {})
        observed = evidence.get("observed_drop_pose", {})
        if (not detached.get("success") or str(detached.get("release_mode", "")).startswith("assumed")
                or not observed):
            raise ValueError("Part slippage has no confirmed detachment and observed pose")
        if any(not isinstance(observed.get(axis), (int, float))
               or not abs(observed[axis] - configuration["drop_pose"][axis]) <= .05 for axis in ("x", "y", "z")):
            raise ValueError("Slipped part was not observed in the configured ur5e-4 region")
        other = configuration["additional_condition"]
        if (before.get(configuration["resource_id"], {}).get("held_part") != configuration["part_name"]
                or before.get(other["resource_id"], {}).get("held_part") != other["part_name"]):
            raise ValueError("Both robots did not hold their parts before Part slippage")
        if sample["values"][other["resource_id"]].get("held_part") != other["part_name"]:
            raise ValueError("The other robot no longer holds its current part")
    return {"validated": True, "run_id": sample["run_id"], "scenario": configuration["scenario"],
            "failure": deepcopy(fault)}


def _held_entry(hold: dict) -> dict:
    decision = hold.get("decision", {})
    if (hold.get("kind") != "candidate_held" or decision.get("status") != "held"
            or "workspace_mutex" not in decision.get("included_specifications", [])):
        return {}
    bindings = decision.get("task_bindings", {})
    witness = decision.get("counterexample", [])
    candidate = bindings.get(witness[0].get("action"), {}) if witness else {}
    if not candidate:
        candidate = next((task for task in bindings.values()
                          if task.get("task_id") == hold["task_id"]), {})
    if (candidate.get("resource_id") in {"ur5e-3", "ur5e-4"}
            and candidate.get("event_name") == "place_approach"
            and candidate.get("parameters", {}).get("destination_location") == "assembly_board-v1"):
        return candidate
    return {}


def validate_mutex(samples: list[dict], negotiations: list[dict]) -> dict:
    """Require a CCA hold during occupancy and later acknowledged access."""
    if not samples or len({row["run_id"] for row in samples}) != 1:
        raise ValueError("Mutex observations must belong to one run")
    occupied = []
    for row in samples:
        robots = [rid for rid in ("ur5e-3", "ur5e-4")
                  if row["values"][rid].get("resource_location") == "assembly_board-v1"]
        if len(robots) > 1 or row["diagnostic_cca_bypass"]:
            raise ValueError("Mutex recording has overlapping occupancy or CCA bypass")
        if robots:
            occupied.append((row["observed_at_unix"], robots[0]))
    holds = [row for row in negotiations if row.get("kind") == "candidate_held"
             and row.get("decision", {}).get("status") == "held"
             and "workspace_mutex" in row.get("decision", {}).get("included_specifications", [])]
    for hold in holds:
        timestamp = hold.get("timestamp")
        # Some existing held records omit timestamps; match their enclosing CCA reply.
        if timestamp is None:
            timestamp = next((row["timestamp"] for row in negotiations
                              if row.get("kind") == "CCA" and hold["task_id"] in row.get("task_ids", [])), None)
        if timestamp is None:
            continue
        candidate = _held_entry(hold)
        held_robot = candidate.get("resource_id")
        if (held_robot not in {"ur5e-3", "ur5e-4"}
                or candidate.get("event_name") != "place_approach"
                or candidate.get("parameters", {}).get("destination_location") != "assembly_board-v1"):
            continue
        preceding = [row for row in samples if row["observed_at_unix"] <= timestamp]
        if not preceding:
            continue
        current = preceding[-1]
        other = "ur5e-4" if held_robot == "ur5e-3" else "ur5e-3"
        if (timestamp - current["observed_at_unix"] > 3
                or current["values"][other].get("resource_location") != "assembly_board-v1"):
            continue
        after = [(when, rid) for when, rid in occupied if when > timestamp and rid == held_robot]
        if not after:
            continue
        entry = after[0][0]
        withdrawals = [row for row in samples if timestamp < row["observed_at_unix"] < entry
                       and row["values"][other].get("resource_location") == "home"]
        sent = [row for row in negotiations if row.get("kind") == "task_sent"
                and row.get("resource_id") == held_robot and row.get("event_name") == "place_approach"
                and timestamp < row["timestamp"] <= entry]
        allowed = next((row for row in negotiations if row.get("kind") == "CCA"
                        and any(row.get("decision", {}).get("decisions", {}).get(task["task_id"], {}).get("status")
                                == "allowed" for task in sent)), None)
        if withdrawals and allowed:
            return {"validated": True, "run_id": samples[0]["run_id"],
                    "rule": mutex_rule(), "hold": hold, "allow": allowed, "held_at_unix": timestamp,
                    "first_robot": other, "waiting_robot": held_robot,
                    "withdrawn_at_unix": withdrawals[0]["observed_at_unix"],
                    "entered_at_unix": entry, "overlapping_occupancy": False}
    raise ValueError("No observed CCA mutex hold followed by the waiting robot's access")


def _probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
         "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=300,
    )
    return json.loads(result.stdout)


def _caption_elapsed(timestamp: float, frames: list[dict], fps: float) -> float:
    frame = min(frames, key=lambda row: abs(row["observed_at_unix"] - timestamp))
    if abs(frame["observed_at_unix"] - timestamp) > 2:
        raise ValueError("Caption event has no nearby recorded observation")
    return frame["frame"] / fps


def export_20x(attempt: RecordingAttempt, output: Path, title: str,
               validation: dict, captions: list[dict]) -> dict:
    """Publish only a fully decoded 20x copy, then remove the owned original."""
    if validation.get("validated") is not True or not validation.get("run_id"):
        raise ValueError("Observed run validation is required before publishing a video")
    if output.exists():
        raise FileExistsError(output)
    source = attempt.directory / "capture.partial.mp4"
    metadata = json.loads((attempt.directory / "capture.json").read_text())
    probe = _probe(source)
    streams = [stream for stream in probe["streams"] if stream["codec_type"] == "video"]
    if metadata.get("status") != "captured" or len(streams) != 1:
        raise ValueError("Source capture is incomplete")
    source_frames = int(streams[0]["nb_read_frames"])
    if source_frames != metadata["frames"]:
        raise ValueError("Source recording has missing frames")
    source_duration = float(probe["format"]["duration"])
    observations = (attempt.directory / "frames.jsonl").read_text().splitlines()
    last = json.loads(observations[-1])
    if abs(source_duration - (last["elapsed_sec"] + 1 / metadata["fps"])) > 1:
        raise ValueError("Source duration disagrees with observed frame timing")
    filters = ["setpts=(PTS-STARTPTS)/20", "fps=15"]
    for index, caption in enumerate([{"text": "20x speed | " + title}, *captions]):
        text_file = attempt.directory / f"caption-{index}.txt"
        text_file.write_text(caption["text"], encoding="utf-8")
        draw = (f"drawtext=textfile='{text_file}':fontsize=22:fontcolor=white:"
                f"box=1:boxcolor=black@0.75:boxborderw=8:x=16:y={16 + 42 * min(index, 1)}")
        if "start" in caption:
            draw += f":enable='between(t,{caption['start'] / 20},{caption['end'] / 20})'"
        filters.append(draw)
    partial = attempt.directory / "20x.partial.mp4"
    log = attempt.directory / "20x.encoder.log"
    with log.open("w") as encoder_log:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-i", str(source), "-vf", ",".join(filters), "-an", "-c:v", "h264_nvenc",
             "-preset", "p1", "-cq", "23", "-b:v", "0", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(partial)],
            stdout=subprocess.DEVNULL, stderr=encoder_log, check=True, timeout=1200,
        )
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(partial),
                        "-f", "null", "-"], stdout=subprocess.DEVNULL,
                       stderr=encoder_log, check=True, timeout=300)
    final = _probe(partial)
    duration = float(final["format"]["duration"])
    if abs(duration - source_duration / 20) > 2 / 15:
        raise ValueError("Final video does not have 20x duration")
    if len(final["streams"]) != 1 or final["streams"][0]["codec_type"] != "video":
        raise ValueError("Final recording must be silent")
    result = {**validation, "video_file": str(output), "speed": 20,
              "source_duration_sec": source_duration, "duration_sec": duration,
              "frames_decoded": int(final["streams"][0]["nb_read_frames"]),
              "captions": captions, "capture": metadata}
    _write_json(attempt.directory / "20x_validation.json", result)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(output)
    source.unlink()
    attempt.cancel()
    return result


async def _observe(bridge, negotiation_cursor: int = 0) -> dict:
    runtime = next((getattr(agent, "environment_runtime", None) for agent in bridge.product_agents
                    if getattr(agent, "environment_runtime", None) is not None), None)
    if runtime is None:
        raise RuntimeError("No environment runtime was started")
    with runtime.context.admission_lock:
        entry_release = getattr(runtime, "_recording_entry_release", None)
        if (entry_release is not None and not entry_release.is_set()
                and runtime.context.resources["ur5e-3"].valuation.get("held_part") == "KET4_Square_4mm"):
            entry_release.set()
        release = getattr(runtime, "_recording_mutex_release", None)
        if release is not None and not release.is_set():
            for hold in runtime.context.negotiations[negotiation_cursor:]:
                if (_held_entry(hold).get("resource_id") == "ur5e-3"
                        and runtime.context.resources["ur5e-4"].valuation.get("resource_location") == "assembly_board-v1"):
                    if runtime._recording_mutex_held_at is None:
                        runtime._recording_mutex_held_at = time.monotonic()
            if (runtime._recording_mutex_held_at is not None
                    and time.monotonic() - runtime._recording_mutex_held_at >= 60):
                release.set()
        return {"observed_at_unix": time.time(), "run_id": runtime.context.run_id,
                "values": runtime.context.snapshot(), "fault": runtime.conveyor_fault.snapshot(),
                "outcome": {key: deepcopy(runtime.outcome[key])
                            for key in ("status", "reason", "failure_evidence") if key in runtime.outcome},
                "stopped": runtime.stopped,
                "diagnostic_cca_bypass": runtime.diagnostic_cca_bypass,
                "pending_tasks": deepcopy(list(runtime.context.pending_tasks.values())),
                "unavailable_resources": sorted(runtime.context.unavailable_resources),
                "negotiations": deepcopy(runtime.context.negotiations[negotiation_cursor:]),
                "run_file": str(runtime.path)}


async def _retry_marker(bridge) -> dict:
    from cais_spade_llm.recovery_framework.conveyor_fault import marker

    runtime = next(agent.environment_runtime for agent in bridge.product_agents
                   if getattr(agent, "environment_runtime", None) is not None)
    fault = runtime.conveyor_fault
    if fault.status != "triggered" or fault.evidence.get("injection_status") != "completed":
        raise ValueError("Only a completed failure injection can retry its visible marker")
    result = await asyncio.to_thread(marker, fault.marker_scene(), "show")
    with runtime.context.admission_lock:
        fault.visual = result
        fault.revision += 1
        runtime.context.negotiations.append({"kind": "recording_marker_retry", "timestamp": time.time(),
                                             "run_id": runtime.context.run_id, "result": deepcopy(result)})
        runtime.queue_save()
    return result


def _read_render_observation() -> dict:
    import ctypes
    import numpy as np

    camera = X11Capture()
    try:
        camera.x.XMapRaised.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        camera.x.XFlush.argtypes = [ctypes.c_void_p]
        camera.x.XMapRaised(camera.display, camera.window)
        camera.x.XFlush(camera.display)
        frame = camera.frame()
        height, width = frame.shape[:2]
        luminance = frame.mean(axis=2)
        contrast = float(np.percentile(luminance, 95) - np.percentile(luminance, 5))
        return {"observed_at_unix": time.time(), "width": width, "height": height,
                "contrast": contrast, "ready": width >= 640 and height >= 360 and contrast >= 20}
    finally:
        camera.close()


def _render_observation() -> dict:
    code = ("import json, sys; "
            "from cais_spade_llm.recovery_framework.failure_videos import _read_render_observation; "
            "sys.stdout.write(json.dumps(_read_render_observation()))")
    result = subprocess.run(["/usr/bin/python3", "-c", code], capture_output=True,
                            text=True, timeout=10)
    if result.returncode:
        raise RuntimeError("Gazebo rendering observation failed: " + result.stderr.strip())
    return json.loads(result.stdout)


def _read_navigation_readiness() -> dict:
    import rclpy
    from lifecycle_msgs.srv import GetState
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    names = ["/KMR/" + name for name in
             ("map_server", "planner_server", "controller_server", "behavior_server", "bt_navigator")]
    rclpy.init()
    node = rclpy.create_node("recording_navigation_readiness_" + uuid4().hex[:12])
    states, pending, received_map = {}, {}, {}
    clients = {name: node.create_client(GetState, name + "/get_state") for name in names}
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)

    def map_received(message):
        received_map.update(frame_id=message.header.frame_id, width=message.info.width,
                            height=message.info.height, cells=len(message.data))

    node.create_subscription(OccupancyGrid, "/KMR/map", map_received, qos)
    deadline = time.monotonic() + 8
    try:
        while time.monotonic() < deadline:
            for name, client in clients.items():
                future = pending.get(name)
                if future is not None and future.done():
                    response = future.result()
                    if response is not None:
                        states[name] = {"id": response.current_state.id, "label": response.current_state.label}
                    pending.pop(name)
                if name not in pending and states.get(name, {}).get("id") != 3 and client.service_is_ready():
                    pending[name] = client.call_async(GetState.Request())
            rclpy.spin_once(node, timeout_sec=.05)
            if (all(states.get(name, {}).get("id") == 3 for name in names)
                    and received_map.get("frame_id") == "world"
                    and received_map.get("cells", 0) > 0):
                break
        graph = node.get_node_names_and_namespaces()
        counts = {name: graph.count((name.rsplit("/", 1)[1], "/KMR")) for name in names}
        return {"observed_at_unix": time.time(), "states": states, "map": received_map,
                "node_counts": counts,
                "ready": all(states.get(name, {}).get("id") == 3 and counts[name] == 1 for name in names)
                         and received_map.get("frame_id") == "world"
                         and received_map.get("cells", 0) > 0}
    finally:
        node.destroy_node()
        rclpy.shutdown()


async def _wait_for_navigation(directory: Path) -> None:
    observations = []
    deadline = time.monotonic() + 180
    code = ("import json, sys; "
            "from cais_spade_llm.recovery_framework.failure_videos import _read_navigation_readiness; "
            "sys.stdout.write(json.dumps(_read_navigation_readiness()))")
    while time.monotonic() < deadline:
        try:
            result = await asyncio.to_thread(subprocess.run, ["/usr/bin/python3", "-c", code],
                                             capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            observation = {"observed_at_unix": time.time(), "ready": False,
                           "error": "Nav2 readiness observation timed out"}
        else:
            if result.returncode:
                raise RuntimeError("Nav2 readiness observation failed: " + result.stderr.strip())
            observation = json.loads(result.stdout)
        observations.append(observation)
        _write_json(directory / "navigation_readiness.json", {"observations": observations})
        if observation["ready"]:
            return
        await asyncio.sleep(1)
    raise RuntimeError("Nav2 did not provide active controllers and one received navigation map before capture")


def _set_assembly_camera(directory: Path) -> None:
    message = ('position { x: 1.8 y: -0.6 z: 3.8 } '
               'orientation { x: -0.4624753906 y: 0.0750494243 z: 0.8720425231 w: 0.1415130202 }')
    result = subprocess.run(["gz", "topic", "-p", "/gazebo/default/user_camera/joy_pose", "-m", message],
                            capture_output=True, text=True, check=True, timeout=10)
    _write_json(directory / "camera.json", {"requested_gui_message": message,
                                          "stdout": result.stdout, "stderr": result.stderr})


async def _wait_for_rendering(directory: Path) -> None:
    observations = []
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            observation = await asyncio.to_thread(_render_observation)
        except (RuntimeError, subprocess.SubprocessError) as exc:
            observation = {"observed_at_unix": time.time(), "ready": False, "error": str(exc)}
        observations.append(observation)
        _write_json(directory / "rendering_readiness.json", {"observations": observations})
        if observation["ready"]:
            return
        await asyncio.sleep(1)
    raise RuntimeError("Gazebo did not provide a visible rendered frame before capture")


def _install_safety(directory: Path, *, mutex: bool) -> Path:
    from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic

    path = directory / "safety.txt"
    text = MUTEX_TEXT if mutex else "[Safety Requirements]\n- \n"
    path.write_text(text)
    payload = {"safety_text_sha256": SafetyLogic.compute_safety_text_sha256(text),
               "rules": [mutex_rule()] if mutex else []}
    _write_json(ROOT / "cais_spade_llm/safety/cca_safety_logic.json", payload)
    if mutex:
        # This is the existing workspace_mutex DFA, expanded to include entry APs.
        dot = ('digraph DFA { node [shape = doublecircle]; 0; init -> 0; '
               '0 -> 1 [label="(ap3 | ap5) & (ap4 | ap6)"]; '
               '0 -> 0 [label="!(ap3 | ap5) | !(ap4 | ap6)"]; 1 -> 1 [label="true"]; }')
        (ROOT / "cais_spade_llm/safety/workspace_mutex_dfa.dot").write_text(dot)
        (directory / "workspace_mutex_dfa.dot").write_text(dot)
    _write_json(directory / "cca_safety_logic.json", payload)
    return path


async def record_one(number: int, directory: Path, output: Path, timeout: float) -> dict:
    """Own one fresh simulation and publish it only after observed acceptance."""
    from cais_spade_llm.recovery_framework.delivery import prepare_start, reset_stop
    from cais_spade_llm.recovery_framework.environment_runtime import (
        prepare_environment_start, record_environment_startup_ready,
    )
    from cais_spade_llm.ui.bridge import SystemBridge

    directory.mkdir(parents=True)
    safety = _install_safety(directory, mutex=number == 5)
    setup = scenario_setup(number, str(safety))
    save_setup(setup)
    _write_json(directory / "setup.json", setup)
    bridge = SystemBridge()
    bridge.execution_mode, bridge.robot_env = "simulation", "gazebo"
    for field in ("selected_product", "selected_product_order_file", "selected_safety_file",
                  "runtime_recovery_mode", "runtime_recovery_validation_policy",
                  "runtime_recovery_archive_path", "runtime_recovery_archive_label"):
        setattr(bridge, field, setup[field])
    original_create_agents = bridge._create_agents
    created_resources, created_products = [], []

    def create_agents(*args, **kwargs):
        agents = original_create_agents(*args, **kwargs)
        _user, resources, products, _cca = agents
        created_resources[:] = resources
        created_products[:] = products
        observations = {}
        for resource in resources:
            controller = getattr(resource, "_controller", None)
            if controller is None or resource.agent_name not in {"ur5e-1", "ur5e-2", "ur5e-3", "ur5e-4"}:
                continue
            if number == 5:
                controller.tf_lookup_timeout_sec = max(controller.tf_lookup_timeout_sec, 30.0)
            target = dict(zip(controller.arm_joint_names, resource.named_positions["home"], strict=True))
            home_deadline = time.monotonic() + 10
            probes = []
            while time.monotonic() < home_deadline:
                positions, missing = controller._get_arm_joint_positions(timeout_sec=.2)
                probes.append({"observed_at_unix": time.time(), "positions": positions, "missing": missing})
                if controller._fresh_stable_joint_target(target, tolerance=.02, stable_for_sec=.1):
                    break
                time.sleep(.05)
            else:
                _write_json(directory / (resource.agent_name + "-home-readiness.json"), {
                    "targets": target, "probes": probes,
                })
                raise ValueError(resource.agent_name + " has no measured stable startup home")
            observations[resource.agent_name] = deepcopy(controller._last_joint_target_observation)
        for product in products:
            runtime = getattr(product, "environment_runtime", None)
            if runtime is None:
                continue
            runtime.composition_budget = {"max_states": 200_000, "seconds": 300.0 if number == 5 else 10.0}
            if number == 5:
                runtime.composition_timeout_s = 900.0
            with runtime.context.admission_lock:
                for rid, observation in observations.items():
                    actor = runtime.context.resources[rid]
                    if actor.valuation.get("resource_state") != "idle" or actor.valuation.get("held_part") is not None:
                        raise ValueError("Startup home observation cannot replace active custody")
                    actor.valuation["resource_location"] = "home"
                    actor.evidence = "measured stable Gazebo startup home"
                    actor.revision += 1
                    runtime.context.negotiations.append({"kind": "startup_home_observation", "resource_id": rid,
                                                         "observation": observation, "timestamp": time.time()})
                runtime.context.revision += 1
            _write_json(directory / "startup_home_observations.json", {
                "run_id": runtime.context.run_id, "observations": observations,
                "composition_budget": runtime.composition_budget,
                "tf_lookup_timeout_sec": {
                    resource.agent_name: resource._controller.tf_lookup_timeout_sec
                    for resource in resources if resource.agent_name in observations
                },
            })
            if number == 5:
                runtime._recording_entry_release = asyncio.Event()
                runtime._recording_mutex_release = asyncio.Event()
                runtime._recording_mutex_held_at = None
                actor = runtime.context.resources["ur5e-4"]
                original_entry = actor.executors["place_approach"]
                original_home = actor.executors["move_home"]
                for resource in resources:
                    if resource.agent_name == "ur5e-4":
                        resource.tool_timeout_s = max(resource.tool_timeout_s, 1200)

                async def place_approach(task: dict) -> dict:
                    runtime.context.negotiations.append({
                        "kind": "recording_entry_pause", "resource_id": "ur5e-4",
                        "task_id": task["task_id"], "timestamp": time.time(),
                        "reason": "Stage KET4_Square_4mm in ur5e-3 before assembly_board-v1 entry",
                    })
                    await asyncio.wait_for(runtime._recording_entry_release.wait(), 1100)
                    runtime.context.negotiations.append({
                        "kind": "recording_entry_pause_released", "resource_id": "ur5e-4",
                        "task_id": task["task_id"], "timestamp": time.time(),
                    })
                    return await original_entry(task)

                async def move_home(task: dict) -> dict:
                    if actor.valuation.get("resource_location") == "assembly_board-v1":
                        runtime.context.negotiations.append({
                            "kind": "recording_occupancy_pause", "resource_id": "ur5e-4",
                            "event_name": "move_home", "task_id": task["task_id"],
                            "timestamp": time.time(), "reason": "Keep actual occupancy until a CCA entry hold is observed",
                        })
                        await asyncio.wait_for(runtime._recording_mutex_release.wait(), 900)
                        runtime.context.negotiations.append({
                            "kind": "recording_occupancy_pause_released", "resource_id": "ur5e-4",
                            "task_id": task["task_id"], "timestamp": time.time(),
                        })
                    return await original_home(task)

                actor.executors["place_approach"] = place_approach
                actor.executors["move_home"] = move_home
        return agents

    bridge._create_agents = create_agents
    attempt = None
    try:
        reset_stop()
        prepare_start(None)
        error = await asyncio.to_thread(bridge.ros2_start, "gazebo_dual")
        if error:
            raise RuntimeError(error)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            ready, reason = await asyncio.to_thread(bridge.simulation_start_ready, True)
            if ready:
                break
            await asyncio.sleep(1)
        else:
            raise RuntimeError("Gazebo readiness failed: " + reason)
        await _wait_for_navigation(directory)
        await asyncio.to_thread(prepare_environment_start, setup, prewarm_controllers=True,
                                launch_identity=bridge._simulation_launch_key(),
                                requested_at_unix=time.time())
        if number in {4, 5}:
            await asyncio.to_thread(_set_assembly_camera, directory)
            await asyncio.sleep(1)
        await _wait_for_rendering(directory)
        attempt = RecordingAttempt(directory)
        await asyncio.to_thread(attempt.start)
        logger.info("Recording %s", TITLES[number - 1])
        await bridge.start_system()
        if not bridge.system_running:
            raise RuntimeError(bridge.last_error or "Start System failed")
        record_environment_startup_ready(bridge)
        deadline = time.monotonic() + timeout
        samples, negotiations, validation, observed_failure, observed_completion = [], [], None, None, None
        marker_retries = []
        with (directory / "observations.jsonl").open("w") as observations:
            while time.monotonic() < deadline:
                await asyncio.to_thread(attempt.check)
                sample = await bridge._run_on_agent_runtime(_observe(bridge, len(negotiations)))
                new_negotiations = sample.pop("negotiations")
                negotiations.extend(new_negotiations)
                samples.append(sample)
                observations.write(json.dumps(sample) + "\n")
                observations.flush()
                _write_json(directory / "progress.json", sample)
                if new_negotiations:
                    _write_json(directory / "negotiations.json", {"negotiations": negotiations})
                fault = sample["fault"]
                if number < 5 and fault["status"] == "triggered":
                    injection_status = fault.get("evidence", {}).get("injection_status")
                    visual_status = fault.get("visual", {}).get("status")
                    if visual_status == "failed" and injection_status == "completed" and len(marker_retries) < 2:
                        try:
                            result = await bridge._run_on_agent_runtime(_retry_marker(bridge))
                        except (RuntimeError, OSError, ValueError, TimeoutError) as exc:
                            result = {"status": "failed", "error": str(exc)}
                        marker_retries.append(result)
                        _write_json(directory / "marker_retries.json", {"retries": marker_retries})
                        await asyncio.sleep(.5)
                        continue
                    if injection_status in {"failed", "interrupted"} or visual_status == "failed":
                        raise ValueError("Failure injection or marker did not complete: " + str({
                            "injection_status": injection_status, "visual": fault.get("visual"),
                            "error": fault.get("evidence", {}).get("error"),
                        }))
                    if injection_status != "completed" or visual_status != "completed":
                        await asyncio.sleep(.5)
                        continue
                    validation = validate_failure(sample, setup["failure_scenario"])
                    if observed_failure is None:
                        observed_failure = time.monotonic()
                        logger.info("Observed %s; capturing 60-second aftermath", TITLES[number - 1])
                    if time.monotonic() - observed_failure >= 60:
                        break
                elif number == 5:
                    if observed_completion is None:
                        try:
                            validation = validate_mutex(samples, negotiations)
                        except ValueError as exc:
                            if str(exc) != "No observed CCA mutex hold followed by the waiting robot's access":
                                raise
                            if sample["stopped"] or sample["outcome"].get("status") == "completed":
                                raise RuntimeError("Run ended without the requested CCA mutex demonstration") from exc
                        else:
                            observed_completion = time.monotonic()
                            logger.info("Observed CCA mutex enforcement and withdrawal; capturing final condition")
                    if observed_completion is not None and time.monotonic() - observed_completion >= 60:
                        validation["final_outcome"] = deepcopy(sample["outcome"])
                        break
                elif sample["stopped"]:
                    raise RuntimeError("Run stopped before acceptance: " + str(sample["outcome"]))
                elif sample["outcome"].get("failure_evidence", {}).get("injection_status") == "not_reached":
                    raise RuntimeError("Failure checkpoint not reached: " + sample["outcome"].get("reason", ""))
                await asyncio.sleep(.5)
            else:
                raise TimeoutError("Recording did not reach its observed acceptance before timeout")
        if validation is None:
            raise ValueError("Recording has no observed validation")
        await asyncio.to_thread(attempt.finish)
        metadata = json.loads((attempt.directory / "capture.json").read_text())
        frames = [json.loads(line) for line in (attempt.directory / "frames.jsonl").read_text().splitlines()]
        if number < 5:
            captions = [{"text": "Failure observed | " + TITLES[number - 1],
                         "start": _caption_elapsed(validation["failure"]["evidence"]["triggered_at_unix"],
                                                   frames, metadata["fps"]),
                         "end": metadata["wall_duration_sec"]}]
        else:
            captions = [
                {"text": validation["waiting_robot"] + " waits | " + validation["first_robot"] + " occupies assembly_board-v1",
                 "start": _caption_elapsed(validation["held_at_unix"], frames, metadata["fps"]),
                 "end": _caption_elapsed(validation["withdrawn_at_unix"], frames, metadata["fps"])},
                {"text": validation["waiting_robot"] + " enters after withdrawal",
                 "start": _caption_elapsed(validation["entered_at_unix"], frames, metadata["fps"]),
                 "end": metadata["wall_duration_sec"]},
            ]
        return await asyncio.to_thread(export_20x, attempt, output, TITLES[number - 1], validation, captions)
    finally:
        if attempt is not None:
            await asyncio.to_thread(attempt.cancel)
            (attempt.directory / "20x.partial.mp4").unlink(missing_ok=True)
        await bridge.stop_system()
        for resource in created_resources:
            controller = getattr(resource, "_controller", None)
            if controller is not None:
                await asyncio.to_thread(controller.shutdown)
        for product in created_products:
            runtime = getattr(product, "environment_runtime", None)
            if runtime is not None:
                runtime._close_calculations()
        prepare_environment_start(None)
        prepare_start(None)
        await asyncio.to_thread(bridge.ros2_stop, "gazebo_dual")
        await bridge._stop_xmpp_server()
        await asyncio.to_thread(bridge._shutdown_agent_runtime_loop)


async def record_videos(numbers: list[int], output: Path, evidence: Path, timeout: float) -> dict:
    """Record requested videos sequentially and restore existing runtime inputs."""
    for number in numbers:
        target = output / (TITLES[number - 1] + "-20x.mp4")
        if target.exists():
            raise FileExistsError(target)
    protected = [SETUP_PATH, ROOT / "cais_spade_llm/initialization/tools.json",
                 ROOT / "cais_spade_llm/safety/cca_safety_logic.json",
                 ROOT / "cais_spade_llm/safety/workspace_mutex_dfa.dot"]
    backups = {path: path.read_bytes() if path.exists() else None for path in protected}
    evidence.mkdir(parents=True, exist_ok=False)
    backup_directory = evidence / "originals"
    backup_directory.mkdir()
    for index, (path, data) in enumerate(backups.items()):
        if data is not None:
            (backup_directory / str(index)).write_bytes(data)
    _write_json(backup_directory / "paths.json", {
        str(index): {"path": str(path), "existed": data is not None}
        for index, (path, data) in enumerate(backups.items())
    })
    result = {"videos": [], "blocked": [], "output_directory": str(output)}
    try:
        for number in numbers:
            directory = evidence / str(number)
            try:
                video = await record_one(number, directory, output / (TITLES[number - 1] + "-20x.mp4"), timeout)
                result["videos"].append(video)
                logger.info("Validated %s", video["video_file"])
            except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
                logger.exception("Recording %s blocked", TITLES[number - 1])
                blocked = {"scenario": TITLES[number - 1], "reason": str(exc)}
                result["blocked"].append(blocked)
                _write_json(directory / "blocked.json", blocked)
            _write_json(evidence / "results.json", result)
    finally:
        for path, data in backups.items():
            if data is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(data)
    return result


def main() -> None:
    """Record selected demonstrations through the existing simulation runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=int, choices=range(1, 6), action="append")
    parser.add_argument("--output", type=Path, default=VIDEO_ROOT)
    parser.add_argument("--evidence", type=Path,
                        default=EVIDENCE_ROOT / ("video-evidence-" + uuid4().hex))
    parser.add_argument("--timeout", type=float, default=2400)
    args = parser.parse_args()
    os.chdir(ROOT)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    result = asyncio.run(record_videos(args.scenario or list(range(1, 6)), args.output, args.evidence, args.timeout))
    logger.info("Results: %s", args.evidence / "results.json")
    raise SystemExit(bool(result["blocked"]))


if __name__ == "__main__":
    main()

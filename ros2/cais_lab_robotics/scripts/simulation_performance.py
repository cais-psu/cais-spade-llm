#!/usr/bin/env python3
"""Publish measured full-scene clock progress independently of any executor."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path


def main() -> None:
    """Keep one atomic, launch-identified observation for the operator UI."""
    import rclpy
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock as ClockMessage
    from std_msgs.msg import String

    rclpy.init()
    node = Node('simulation_performance')
    node.declare_parameter('launch_id', '')
    node.declare_parameter('settings', '{}')
    launch_id = node.get_parameter('launch_id').value
    settings = json.loads(node.get_parameter('settings').value)
    path = Path('/tmp') / f"cais_simulation_performance_{int(os.environ.get('ROS_DOMAIN_ID', '0'))}.json"
    publisher = node.create_publisher(String, '/simulation_performance', 1)
    samples = deque(maxlen=2000)
    started = time.monotonic()
    resets = 0
    first_clock_wall = None

    def clock_sample(message):
        nonlocal resets, first_clock_wall
        wall = time.monotonic()
        simulated = message.clock.sec + message.clock.nanosec / 1e9
        if first_clock_wall is None:
            first_clock_wall = wall
        if samples and simulated < samples[-1][1]:
            samples.clear()
            resets += 1
        samples.append((wall, simulated))
        while len(samples) > 2 and samples[1][0] < wall - 4.0:
            samples.popleft()

    def publish():
        wall = time.monotonic()
        elapsed = wall - samples[0][0] if samples else 0.0
        rate = ((samples[-1][1] - samples[0][1]) / elapsed
                if elapsed > 0 and len(samples) > 1 else None)
        status = {
            'launch_id': launch_id, 'settings': settings, 'observed_at_unix': time.time(),
            'requested_real_time_factor': settings.get('speed'),
            'real_time_factor': rate, 'simulation_time_sec': samples[-1][1] if samples else None,
            'wall_time_sec': wall - started,
            'first_clock_wall_time_sec': first_clock_wall - started if first_clock_wall else None,
            'clock_age_wall_sec': wall - samples[-1][0] if samples else None,
            'clock_resets': resets,
            'physics_step_sec': .001, 'kmr_controller_rate_hz': 225,
        }
        encoded = json.dumps(status, allow_nan=False)
        temporary = path.with_suffix(f'.{launch_id}.tmp')
        temporary.write_text(encoded)
        temporary.replace(path)
        publisher.publish(String(data=encoded))

    subscription = node.create_subscription(ClockMessage, '/clock', clock_sample, qos_profile_sensor_data)
    timer = node.create_timer(1.0, publish, clock=Clock(clock_type=ClockType.STEADY_TIME))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        timer.cancel()
        node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

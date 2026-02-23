"""
Test: Move UR5e elbow joint 90 degrees in Gazebo.

Prerequisites:
    1. Gazebo running:  ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5e
    2. Run this script: source /opt/ros/jazzy/setup.bash && python3 test/test_ur5e_elbow_bend.py
"""
import sys
import os
import time
import math

# Add project root to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cais_spade_llm.resources.robot.ur5e_controller import UR5eController


def main():
    ctrl = UR5eController()
    ctrl.init()

    # Give the publisher time to register with the ROS2 graph
    time.sleep(1.0)

    # All joints at 0 = straight up
    # Bend elbow_joint (index 2) to 90 degrees = pi/2 radians
    print("Moving elbow joint to 90 degrees...")
    ctrl.move_joints([0.0, 0.0, math.pi / 2, 0.0, 0.0, 0.0], duration_sec=3)

    # Wait for motion to complete
    time.sleep(4)
    print("Done! The arm should now be bent at the elbow.")

    ctrl.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
xArm6 real hardware controller bring-up without MoveIt/RViz.

This launch is used by the dual robots digital twin. It starts the xArm6
ros2_control stack under the xarm6 namespace so it can coexist with the UR5e
hardware driver in the same ROS_DOMAIN_ID.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


_JOINT_STATE_RELAY = r"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

try:
    from control_msgs.action._gripper_command import GripperCommand_FeedbackMessage
except Exception:
    GripperCommand_FeedbackMessage = None


class XArm6JointStateRelay(Node):
    def __init__(self):
        super().__init__("xarm6_joint_state_relay")
        self._publisher = self.create_publisher(JointState, "/joint_states", 20)
        self._drive_joint_position = 0.0
        for topic in (
            "/xarm/joint_states",
            "/xarm6/joint_states",
            "/xarm6/xarm/joint_states",
            "/xarm6/xarm_gripper/joint_states",
        ):
            self.create_subscription(JointState, topic, self._relay, 20)
        if GripperCommand_FeedbackMessage is not None:
            for topic in (
                "/xarm6/xarm_gripper/gripper_action/_action/feedback",
                "/xarm_gripper/gripper_action/_action/feedback",
            ):
                self.create_subscription(
                    GripperCommand_FeedbackMessage,
                    topic,
                    self._gripper_feedback,
                    20,
                )
        self.create_timer(0.05, self._publish_drive_joint)

    def _relay(self, msg):
        for index, name in enumerate(msg.name):
            if name == "drive_joint" and index < len(msg.position):
                self._drive_joint_position = float(msg.position[index])
        self._publisher.publish(msg)

    def _gripper_feedback(self, msg):
        feedback = getattr(msg, "feedback", None)
        if feedback is not None and hasattr(feedback, "position"):
            self._drive_joint_position = float(feedback.position)

    def _publish_drive_joint(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["drive_joint"]
        msg.position = [self._drive_joint_position]
        msg.velocity = [0.0]
        msg.effort = [0.0]
        self._publisher.publish(msg)


rclpy.init()
node = XArm6JointStateRelay()
try:
    rclpy.spin(node)
finally:
    node.destroy_node()
    rclpy.shutdown()
"""


def launch_setup(context, *args, **kwargs):
    robot_ip = LaunchConfiguration("robot_ip")
    xarm_namespace = LaunchConfiguration("xarm_namespace", default="xarm6")
    namespace = xarm_namespace.perform(context).strip("/") or "xarm6"

    ros2_control_launch = GroupAction(
        [
            PushRosNamespace(namespace),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("xarm_controller"), "launch", "_ros2_control.launch.py"]
                    )
                ),
                launch_arguments={
                    "robot_ip": robot_ip,
                    "dof": "6",
                    "robot_type": "xarm",
                    "hw_ns": "xarm",
                    "prefix": "",
                    "add_gripper": "true",
                    "ros_namespace": namespace,
                }.items(),
            ),
        ]
    )

    required_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "joint_state_broadcaster",
            "xarm6_traj_controller",
            "--controller-manager",
            f"/{namespace}/controller_manager",
            "--controller-manager-timeout",
            "60",
            "--activate-as-group",
        ],
    )

    joint_state_relay = ExecuteProcess(
        cmd=["python3.10", "-c", _JOINT_STATE_RELAY],
        output="screen",
    )

    return [
        ros2_control_launch,
        required_controller_spawner,
        joint_state_relay,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_ip",
                default_value="192.168.1.240",
                description="xArm6 hardware IP address.",
            ),
            DeclareLaunchArgument(
                "xarm_namespace",
                default_value="xarm6",
                description="Namespace for the xArm6 hardware controller_manager.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )

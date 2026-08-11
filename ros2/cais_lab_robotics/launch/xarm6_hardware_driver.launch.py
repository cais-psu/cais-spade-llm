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
from launch_ros.actions import Node, PushRosNamespace, SetRemap
from launch_ros.substitutions import FindPackageShare

_JOINT_STATE_RELAY = r"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from xarm_msgs.msg import RobotMsg

try:
    from control_msgs.action._gripper_command import GripperCommand_FeedbackMessage
except ImportError:
    GripperCommand_FeedbackMessage = None


class XArm6JointStateRelay(Node):
    def __init__(self):
        super().__init__("xarm6_joint_state_relay")
        self._publisher = self.create_publisher(JointState, "/joint_states", 20)
        self._drive_joint_position = 0.0
        self._arm_feedback_received = False
        self.create_subscription(
            RobotMsg,
            "/xarm6/xarm/robot_states",
            self._robot_state,
            20,
        )
        self.create_subscription(
            JointState,
            "/xarm6/xarm_gripper/joint_states",
            self._relay,
            20,
        )
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

    def _robot_state(self, msg):
        angles = list(getattr(msg, "angle", []))
        if len(angles) < 6:
            return
        joint_state = JointState()
        joint_state.header = msg.header
        joint_state.name = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "drive_joint",
        ]
        joint_state.position = [
            float(angles[0]),
            float(angles[1]),
            float(angles[2]),
            float(angles[3]),
            float(angles[4]),
            float(angles[5]),
            self._drive_joint_position,
        ]
        joint_state.velocity = [0.0] * 7
        joint_state.effort = [0.0] * 7
        self._publisher.publish(joint_state)
        if not self._arm_feedback_received:
            self._arm_feedback_received = True
            self.get_logger().info(
                "Publishing joint1..joint6 from /xarm6/xarm/robot_states to /joint_states"
            )

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
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
"""


def launch_setup(context, *args, **kwargs):
    robot_ip = LaunchConfiguration("robot_ip")
    xarm_namespace = LaunchConfiguration("xarm_namespace", default="xarm6")
    namespace = xarm_namespace.perform(context).strip("/") or "xarm6"

    ros2_control_launch = GroupAction(
        [
            PushRosNamespace(namespace),
            SetRemap(
                src="/controller_manager/list_controllers",
                dst=f"/{namespace}/controller_manager/list_controllers",
            ),
            SetRemap(
                src="/controller_manager/switch_controller",
                dst=f"/{namespace}/controller_manager/switch_controller",
            ),
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
                    "extra_robot_api_params_path": PathJoinSubstitution(
                        [
                            FindPackageShare("cais_lab_robotics"),
                            "config",
                            "hardware_runtime",
                            "xarm6_robot_api_services.yaml",
                        ]
                    ),
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
            "--service-call-timeout",
            "30",
            "--switch-timeout",
            "30",
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

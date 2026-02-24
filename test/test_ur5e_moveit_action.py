"""
Demonstrate controlling the UR5e in the dual-robot MoveIt setup using the MoveGroup action server.
This uses the standard ROS 2 ActionClient to command MoveIt (avoiding obstacles, respecting limits).
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, WorkspaceParameters, Constraints, JointConstraint

class MoveItActionClient(Node):
    def __init__(self):
        super().__init__('ur5e_moveit_client')
        self._action_client = ActionClient(self, MoveGroup, '/ur5e/move_action')
        self._timer = self.create_timer(1.0, self.timer_callback)
        self.target_pos = [0.0, -math.pi/4, math.pi/2, -math.pi/4, -math.pi/2, 0.0]
        
    def timer_callback(self):
        if not self._action_client.server_is_ready():
            self.get_logger().info('Waiting for /ur5e/move_action server...')
            return
        
        self.get_logger().info('Server is ready. Sending goal.')
        self._timer.cancel()
        
        goal_msg = MoveGroup.Goal()
        
        req = MotionPlanRequest()
        req.group_name = 'ur5e_ur_manipulator'
        req.allowed_planning_time = 5.0
        req.num_planning_attempts = 10
        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5
        
        # Set workspace bounds
        workspace = WorkspaceParameters()
        workspace.header.frame_id = 'world'
        workspace.min_corner.x, workspace.min_corner.y, workspace.min_corner.z = -1.0, 0.0, 0.5
        workspace.max_corner.x, workspace.max_corner.y, workspace.max_corner.z = 1.0, 1.5, 2.0
        req.workspace_parameters = workspace

        # Create Joint Constraints for the target pose
        constraint = Constraints()
        joints = [
            'ur5e_shoulder_pan_joint', 'ur5e_shoulder_lift_joint', 'ur5e_elbow_joint',
            'ur5e_wrist_1_joint', 'ur5e_wrist_2_joint', 'ur5e_wrist_3_joint'
        ]
        
        for name, pos in zip(joints, self.target_pos):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraint.joint_constraints.append(jc)
            
        req.goal_constraints.append(constraint)
        goal_msg.request = req
        
        # We tell MoveIt to plan AND execute the trajectory
        goal_msg.planning_options.plan_only = False
        
        self.get_logger().info('Sending goal to MoveIt for UR5e...')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return
        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
        
    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Feedback: {feedback_msg.feedback.state}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result code: {result.error_code.val}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = MoveItActionClient()
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()

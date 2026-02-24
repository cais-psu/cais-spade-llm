"""
Demonstrate controlling the UR5e RG2 gripper in the dual-robot MoveIt setup.
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint

class GripperActionClient(Node):
    def __init__(self):
        super().__init__('ur5e_rg2_action_client')
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        self._timer = self.create_timer(1.0, self.timer_callback)
        self.target_width = 0.05 # Close a bit (max is ~0.11)
        
    def timer_callback(self):
        if not self._action_client.server_is_ready():
            self.get_logger().info('Waiting for /move_action server...')
            return
        
        self.get_logger().info('Server is ready. Sending goal.')
        self._timer.cancel()
        
        goal_msg = MoveGroup.Goal()
        
        req = MotionPlanRequest()
        req.group_name = 'ur5e_rg2_gripper'
        req.allowed_planning_time = 5.0
        req.num_planning_attempts = 10
        req.max_velocity_scaling_factor = 1.0
        req.max_acceleration_scaling_factor = 1.0

        constraint = Constraints()
        jc = JointConstraint()
        jc.joint_name = 'ur5e_rg2_finger_width'
        jc.position = float(self.target_width)
        jc.tolerance_above = 0.005
        jc.tolerance_below = 0.005
        jc.weight = 1.0
        constraint.joint_constraints.append(jc)
            
        req.goal_constraints.append(constraint)
        goal_msg.request = req
        
        goal_msg.planning_options.plan_only = False
        
        self.get_logger().info(f'Sending goal to MoveIt for RG2 width: {self.target_width}')
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
        pass

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result code: {result.error_code.val}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = GripperActionClient()
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()

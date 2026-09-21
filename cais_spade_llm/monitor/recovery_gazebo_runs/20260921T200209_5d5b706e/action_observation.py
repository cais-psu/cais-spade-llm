from __future__ import annotations
import json
import sys
import time
import rclpy
from action_msgs.msg import GoalStatusArray
from rclpy.qos import QoSProfile, DurabilityPolicy
rclpy.init()
node=rclpy.create_node('KMR_acceptance_action_observer')
expected={int(value) for value in sys.argv[1].split(',')}
goal=sys.argv[2] if len(sys.argv)>2 else None
observed=None

def receive(message):
    global observed
    for entry in message.status_list:
        identity=bytes(entry.goal_info.goal_id.uuid).hex()
        if entry.status in expected and (goal is None or identity==goal):
            observed={'goal_id':identity,'status':entry.status}

subscription=node.create_subscription(GoalStatusArray,'/execute_trajectory/_action/status',receive,
    QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL))
deadline=time.monotonic()+55
while observed is None and time.monotonic()<deadline:
    rclpy.spin_once(node,timeout_sec=.1)
node.destroy_node()
rclpy.shutdown()
if observed is None: raise TimeoutError('Expected action status was not observed')
print(json.dumps(observed))

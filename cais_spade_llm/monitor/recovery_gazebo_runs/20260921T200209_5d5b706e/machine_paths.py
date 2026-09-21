from __future__ import annotations
import json
import hashlib
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from moveit_msgs.srv import GetPositionIK, GetStateValidity, GetPlanningScene, GetCartesianPath
from geometry_msgs.msg import PoseStamped, Pose
from builtin_interfaces.msg import Duration
rclpy.init()
n=Node('machine_access_validation',parameter_overrides=[Parameter('use_sim_time',value=True)])
def call(kind,name,req):
 c=n.create_client(kind,name)
 if not c.wait_for_service(timeout_sec=10): raise RuntimeError(name)
 f=c.call_async(req);rclpy.spin_until_future_complete(n,f,timeout_sec=15)
 if not f.done(): raise TimeoutError(name)
 return f.result()
def pose(v):
 p=Pose();p.position.x,p.position.y,p.position.z=v[:3];p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w=v[3:];return p
scene=json.loads(Path('cais_spade_llm/initialization/recovery_framework_gazebo.json').read_text())
rows=[]
for i,machine in enumerate(scene['machines'],1):
 for robot in ['KMR',machine['handling_robot']]:
  q=GetPlanningScene.Request();q.components.components=2|4
  state=call(GetPlanningScene,'/get_planning_scene',q).scene.robot_state
  if robot=='KMR':
   group,link='KMR_iiwa_arm','rg2_gripper_tcp'
   for joint,value in zip(['KMR_base_x_joint','KMR_base_y_joint','KMR_base_yaw_joint'],[machine['KMR_docking_pose'][0],machine['KMR_docking_pose'][1],machine['KMR_docking_pose'][5]]):
    state.joint_state.position[state.joint_state.name.index(joint)]=value
   target=[*machine['workholding_pose'][:3],.8660254037844386,0.,.5,0.];target[2]+=.025
   approach=target.copy();approach[0]-=.1;approach[2]+=.035
  else:
   group,link=f'ur5e_{i}_ur_manipulator',f'ur5e_{i}_rg2_gripper_tcp'
   target=[*machine['workholding_pose'][:3],0.,1.,0.,0.];target[2]+=.025
   approach=target.copy();approach[1]-=.18
  req=GetPositionIK.Request();req.ik_request.group_name=group;req.ik_request.ik_link_name=link;req.ik_request.robot_state=state;req.ik_request.pose_stamped=PoseStamped();req.ik_request.pose_stamped.header.frame_id='world';req.ik_request.pose_stamped.pose=pose(approach);req.ik_request.timeout=Duration(sec=5);req.ik_request.avoid_collisions=True
  ik=call(GetPositionIK,'/compute_ik',req)
  row={'machine':machine['resource_id'],'robot':robot,'approach':approach,'target':target,'ik_code':ik.error_code.val}
  if ik.error_code.val==1:
   req=GetCartesianPath.Request();req.header.frame_id='world';req.group_name=group;req.link_name=link;req.start_state=ik.solution;req.waypoints=[pose(target)];req.max_step=.003;req.jump_threshold=2.;req.avoid_collisions=True
   result=call(GetCartesianPath,'/compute_cartesian_path',req);row.update(cartesian_fraction=result.fraction,error_code=result.error_code.val)
  rows.append(row)
Path('/tmp/cais_machine_paths_final.json').write_text(json.dumps({'scope':'Collision-aware IK and Cartesian approach checks; no robot execution', 'scene_sha256':hashlib.sha256(Path('cais_spade_llm/initialization/recovery_framework_gazebo.json').read_bytes()).hexdigest(), 'checks':rows},indent=2))
assert all(row.get('ik_code')==1 and row.get('error_code')==1 and row.get('cartesian_fraction',0)>=.999 for row in rows), rows
n.destroy_node();rclpy.shutdown()
